// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! H.264 encoding on a V4L2 stateful encoder (kernel
//! `Documentation/userspace-api/media/v4l/dev-encoder.rst`).
//!
//! Source tensors are imported on the OUTPUT queue as DMA-BUF planes. NV12
//! uses the two-plane `NV12M` layout where the device offers it, so each
//! plane carries its own offset. A source the device cannot read in place
//! is copied into a DMA-BUF of the device's layout first. The bitstream
//! comes back in MMAP CAPTURE buffers and is copied out.
//!
//! Each source is tagged with a sequence number, and the caller's PTS is
//! looked up again when its access unit comes back.

use std::collections::{HashSet, VecDeque};
use std::io::{Seek, SeekFrom};
use std::os::fd::{AsFd, AsRawFd, BorrowedFd};
use std::time::{Duration, Instant};

use edgefirst_tensor::{CpuAccess, DType, PixelFormat, Tensor, TensorDyn, TensorMemory};
use edgefirst_v4l2::controls::{self, ControlInfo, ControlType, ControlValue};
use edgefirst_v4l2::device::Fraction;
use edgefirst_v4l2::m2m::M2m;
use edgefirst_v4l2::queue::{Mapping, Memory, Plane};
use edgefirst_v4l2::uapi;

use super::{candidates, device_error, M2mDevice};
use crate::video::copy;
use crate::video::h264::{self, AccessUnit};
use crate::video::{
    chroma_subsampling, Bitrate, EncodedFrame, EncoderBackend, EncoderConfig, EncoderInfo,
    FrameOptions, H264Level, H264Profile,
};
use crate::{Codec, CodecError, Engine, Result, Transfer};

/// Environment variable naming the encoder node to use.
pub(crate) const ENV_ENCODER: &str = "EDGEFIRST_CODEC_V4L2_ENCODER";
/// How long the device may take to read one source frame.
const INPUT_TIMEOUT: Duration = Duration::from_secs(2);
/// How long to wait, once a source is read, for its access unit.
const OUTPUT_GRACE: Duration = Duration::from_millis(100);
/// How long a drain may take.
const DRAIN_TIMEOUT: Duration = Duration::from_secs(2);
/// CAPTURE (bitstream) buffers, at least.
const CAPTURE_BUFFERS: u32 = 4;
/// OUTPUT (source) buffer slots, at least.
const OUTPUT_BUFFERS: u32 = 2;
/// `V4L2_CID_JPEG_COMPRESSION_QUALITY`: `V4L2_CID_JPEG_CLASS_BASE` + 3.
const CID_JPEG_COMPRESSION_QUALITY: u32 = 0x009d_0903;
/// Alignment of the chroma plane in a two-plane staging buffer.
const PLANE_ALIGN: usize = 4096;
/// The most source shapes whose copy is logged at info.
const LOGGED_SHAPES: usize = 256;

/// The CAPTURE formats that carry `codec`, in the order they are tried.
pub(crate) fn coded_fourccs(codec: Codec) -> &'static [u32] {
    match codec {
        Codec::H264 => &[uapi::V4L2_PIX_FMT_H264],
        Codec::Mjpeg | Codec::Jpeg => &[uapi::V4L2_PIX_FMT_JPEG, uapi::V4L2_PIX_FMT_MJPEG],
        _ => &[],
    }
}

/// How a source's pixels are handed to the device.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Layout {
    /// One plane (packed RGB and YUV formats).
    Packed,
    /// Semi-planar in one V4L2 plane: chroma follows luma at
    /// `bytesperline * height` (`NV12`).
    Contiguous,
    /// Semi-planar in two V4L2 planes, each with its own offset (`NV12M`).
    Split,
}

/// The V4L2 formats that can carry `format`, in order of preference.
fn input_candidates(format: PixelFormat) -> Vec<(u32, Layout)> {
    match format {
        PixelFormat::Nv12 => vec![
            (uapi::V4L2_PIX_FMT_NV12M, Layout::Split),
            (uapi::V4L2_PIX_FMT_NV12, Layout::Contiguous),
        ],
        PixelFormat::Nv16 | PixelFormat::Nv24 | PixelFormat::PlanarRgb => Vec::new(),
        PixelFormat::PlanarRgba => Vec::new(),
        other => match other.to_fourcc() {
            0 => Vec::new(),
            fourcc => vec![(fourcc, Layout::Packed)],
        },
    }
}

/// The pixel formats `device`, an H.264 encoder, takes as sources.
pub(crate) fn input_formats(device: &M2mDevice) -> Vec<PixelFormat> {
    PixelFormat::all()
        .iter()
        .copied()
        .filter(|&format| {
            input_candidates(format).iter().any(|&(fourcc, layout)| {
                (layout != Layout::Split || device.multiplanar())
                    && device.lists(device.output, fourcc)
            })
        })
        .collect()
}

/// One plane of a source as the device will read it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct SourcePlane {
    /// Byte offset of the first pixel to encode in the DMA-BUF.
    offset: usize,
    /// Rows of this plane in the encoded picture.
    rows: usize,
}

/// Where the planes of a frame start, given where its buffers start.
///
/// `base` is the offset of the source image in its luma buffer and
/// `chroma_base` that of the chroma plane in its buffer; `strides` are the
/// luma and chroma row strides; `crop` is the picture's origin in pixels.
fn source_planes(
    layout: Layout,
    format: PixelFormat,
    strides: [usize; 2],
    base: usize,
    chroma_base: usize,
    (x, y): (usize, usize),
    height: usize,
) -> Vec<SourcePlane> {
    let bpp = format.channels();
    let luma = SourcePlane {
        offset: base + y * strides[0] + x * bpp,
        rows: height,
    };
    match layout {
        Layout::Packed => vec![luma],
        Layout::Contiguous | Layout::Split => {
            let (_, vsub) = chroma_subsampling(format);
            let chroma = SourcePlane {
                offset: chroma_base + (y / vsub as usize) * strides[1] + x,
                rows: height / vsub as usize,
            };
            vec![luma, chroma]
        }
    }
}

/// The OUTPUT format the device set: how it reads a source.
#[derive(Debug, Clone, PartialEq, Eq)]
struct OutputFormat {
    /// `bytesperline` of the luma (or packed) plane and of the chroma plane.
    strides: [usize; 2],
    /// Height of the format. The one-plane `NV12` layout has its chroma
    /// `strides[0] * rows` bytes after the luma.
    rows: usize,
    /// `sizeimage` of each V4L2 plane.
    sizes: Vec<usize>,
}

impl OutputFormat {
    /// Byte offset and size of each V4L2 plane in one buffer that holds a
    /// whole picture in this format, and the buffer's size.
    fn buffer_planes(&self, layout: Layout) -> (Vec<(usize, usize)>, usize) {
        let [luma_stride, chroma_stride] = self.strides;
        let size = |i: usize| self.sizes.get(i).copied().unwrap_or(0);
        let luma = luma_stride * self.rows;
        let chroma = chroma_stride * self.rows.div_ceil(2);
        match layout {
            Layout::Packed => (vec![(0, size(0).max(luma))], size(0).max(luma)),
            Layout::Contiguous => {
                let len = size(0).max(luma + chroma);
                (vec![(0, len)], len)
            }
            Layout::Split => {
                let luma = size(0).max(luma);
                let start = luma.next_multiple_of(PLANE_ALIGN);
                let chroma = size(1).max(chroma);
                (vec![(0, luma), (start, chroma)], start + chroma)
            }
        }
    }

    /// Where each picture plane starts in a buffer laid out as
    /// [`buffer_planes`](Self::buffer_planes) describes, and its row pitch.
    fn picture_planes(&self, layout: Layout) -> Vec<copy::Plane> {
        let (planes, _) = self.buffer_planes(layout);
        let luma = copy::Plane {
            offset: 0,
            pitch: self.strides[0],
        };
        match layout {
            Layout::Packed => vec![luma],
            Layout::Contiguous => vec![
                luma,
                copy::Plane {
                    offset: self.strides[0] * self.rows,
                    pitch: self.strides[1],
                },
            ],
            Layout::Split => vec![
                luma,
                copy::Plane {
                    offset: planes[1].0,
                    pitch: self.strides[1],
                },
            ],
        }
    }
}

/// A plane of a source the device reads in place.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct ImportPlane {
    /// 0 for the source's DMA-BUF, 1 for its chroma tensor's.
    buffer: usize,
    offset: usize,
    end: usize,
}

/// The planes to import for a source whose planes are `planes`, with row
/// strides `strides`, in buffers of `lengths` bytes. `chroma_buffer` is the
/// buffer of the chroma plane; `same_buffer` says whether it is the luma's
/// DMA-BUF. Returns why the device cannot read the source in place instead.
fn import_planes(
    layout: Layout,
    multiplanar: bool,
    device: &OutputFormat,
    planes: &[SourcePlane],
    strides: [usize; 2],
    lengths: [usize; 2],
    (chroma_buffer, same_buffer): (usize, bool),
) -> std::result::Result<Vec<ImportPlane>, String> {
    if strides[0] != device.strides[0] {
        return Err(format!(
            "the device reads rows {} bytes apart, the source's are {} apart",
            device.strides[0], strides[0]
        ));
    }
    if layout != Layout::Packed && strides[1] != device.strides[1] {
        return Err(format!(
            "the device reads chroma rows {} bytes apart, the source's are {} apart",
            device.strides[1], strides[1]
        ));
    }
    let end = |p: &SourcePlane, stride: usize| p.offset + stride * p.rows;
    let imports = match layout {
        Layout::Packed => vec![ImportPlane {
            buffer: 0,
            offset: planes[0].offset,
            end: end(&planes[0], strides[0]),
        }],
        Layout::Contiguous => {
            if !same_buffer {
                return Err(
                    "the device reads chroma from the luma buffer, the source has it in another"
                        .into(),
                );
            }
            let want = device.strides[0] * device.rows;
            let gap = planes[1].offset.checked_sub(planes[0].offset);
            if gap != Some(want) {
                return Err(format!(
                    "the device reads chroma {want} bytes after the luma, the source has it {}",
                    gap.map_or_else(|| "before".to_owned(), |g| format!("{g} bytes after"))
                ));
            }
            vec![ImportPlane {
                buffer: 0,
                offset: planes[0].offset,
                end: end(&planes[1], strides[1]),
            }]
        }
        Layout::Split => vec![
            ImportPlane {
                buffer: 0,
                offset: planes[0].offset,
                end: end(&planes[0], strides[0]),
            },
            ImportPlane {
                buffer: chroma_buffer,
                offset: planes[1].offset,
                end: end(&planes[1], strides[1]),
            },
        ],
    };
    for (i, p) in imports.iter().enumerate() {
        let length = lengths[p.buffer];
        if p.end > length {
            return Err(format!(
                "plane {i} ends at byte {} of a {length}-byte DMA-BUF",
                p.end
            ));
        }
        let min = device.sizes.get(i).copied().unwrap_or(0);
        if length < min {
            return Err(format!(
                "plane {i} is in a {length}-byte DMA-BUF, the device needs {min}"
            ));
        }
        if !multiplanar && p.offset != 0 {
            return Err(format!(
                "the device reads whole buffers, the source starts at byte {}",
                p.offset
            ));
        }
    }
    Ok(imports)
}

/// The device's streaming state, created by the first frame.
struct Stream {
    m2m: M2m,
    capture_maps: Vec<Mapping>,
    format: OutputFormat,
    /// The buffer sources the device cannot read in place are copied into.
    staging: Option<TensorDyn>,
    next_output: u32,
    output_count: u32,
}

pub(crate) struct V4l2Encoder {
    device: M2mDevice,
    config: EncoderConfig,
    path: String,
    /// The CAPTURE format: H.264, or JPEG pictures.
    coded_fourcc: u32,
    input_fourcc: u32,
    layout: Layout,
    controls: Vec<ControlInfo>,
    can_stop: bool,
    stream: Option<Stream>,
    next_seq: u64,
    /// `(sequence, pts)` of sources whose access unit has not come back.
    in_flight: VecDeque<(u64, u64)>,
    pending: VecDeque<EncodedFrame>,
    /// NAL units that precede a picture, delivered in a buffer of their
    /// own, for the next access unit.
    headers: Vec<u8>,
    /// The stream's latest SPS and PPS, for key frames the device sends
    /// without them.
    parameter_sets: Vec<u8>,
    /// How the latest source reached the device.
    transfer: Transfer,
    /// The source shapes whose copy has been logged at info.
    copied: HashSet<CopyShape>,
}

/// A source's format, size, row stride and memory, and whether it is
/// cropped.
type CopyShape = (PixelFormat, (usize, usize), usize, String, bool);

impl V4l2Encoder {
    /// Opens the first device that encodes `config.codec` from
    /// `config.input_format`, and applies the configuration.
    pub fn open(config: EncoderConfig) -> Result<Self> {
        let formats = input_candidates(config.input_format);
        if formats.is_empty() {
            return Err(CodecError::UnsupportedFormat(config.input_format));
        }
        let codec = config.codec;
        let mut found_encoder = false;
        let mut last_err = None;
        let mut busy = None;
        for path in candidates(ENV_ENCODER) {
            let device = match M2mDevice::try_open(&path) {
                Ok(Some(device)) => device,
                Ok(None) => continue,
                Err(e) => {
                    log::debug!("{e}");
                    busy.get_or_insert(e);
                    continue;
                }
            };
            let Some(&coded_fourcc) = coded_fourccs(codec)
                .iter()
                .find(|&&f| device.lists(device.capture, f))
            else {
                continue;
            };
            found_encoder = true;
            let Some(&(fourcc, layout)) = formats.iter().find(|&&(fourcc, layout)| {
                (layout != Layout::Split || device.multiplanar())
                    && device.lists(device.output, fourcc)
            }) else {
                log::debug!(
                    "{} does not take {} input",
                    path.display(),
                    config.input_format
                );
                continue;
            };
            let mut enc = Self {
                device,
                config: config.clone(),
                path: path.display().to_string(),
                coded_fourcc,
                input_fourcc: fourcc,
                layout,
                controls: Vec::new(),
                can_stop: false,
                stream: None,
                next_seq: 1,
                in_flight: VecDeque::new(),
                pending: VecDeque::new(),
                headers: Vec::new(),
                parameter_sets: Vec::new(),
                transfer: Transfer::ZeroCopy,
                copied: HashSet::new(),
            };
            match enc.configure() {
                Ok(()) => {
                    log::info!(
                        "{codec} encoder {} ({}, {:?})",
                        enc.path,
                        uapi::fourcc_str(fourcc),
                        layout
                    );
                    return Ok(enc);
                }
                Err(e) => {
                    log::debug!("{}: {e}", enc.path);
                    last_err = Some(e);
                }
            }
        }
        Err(match (last_err, found_encoder, busy) {
            (Some(e), _, _) => e,
            (None, true, _) => CodecError::UnsupportedFormat(config.input_format),
            (None, false, Some(e)) => e,
            (None, false, None) => CodecError::NoDevice(format!("no V4L2 {codec} encoder found")),
        })
    }

    /// Applies the controls, sets the coded format and checks the picture
    /// size. Buffers wait for the first frame.
    fn configure(&mut self) -> Result<()> {
        let (w, h) = (self.config.width, self.config.height);
        let dev = &self.device.dev;

        // The controls are set before the CAPTURE format.
        self.controls = controls::query_all(dev).map_err(device_error)?;
        if self.is_h264() {
            self.apply_controls()?;
        } else {
            let quality = i32::from(self.config.jpeg_quality());
            if !self.set_control(CID_JPEG_COMPRESSION_QUALITY, quality)? {
                log::debug!("{}: no JPEG quality control", self.path);
            }
        }

        let mut cap = dev.format(self.device.capture).map_err(device_error)?;
        // SAFETY: the buffer type selects the union member.
        unsafe {
            if self.device.multiplanar() {
                let p = cap.pix_mp();
                p.pixelformat = self.coded_fourcc;
                p.width = w;
                p.height = h;
                p.num_planes = 1;
                p.plane_fmt[0].sizeimage = 0;
            } else {
                let p = cap.pix();
                p.pixelformat = self.coded_fourcc;
                p.width = w;
                p.height = h;
                p.sizeimage = 0;
            }
        }
        dev.set_format(&mut cap).map_err(device_error)?;
        if self.is_h264() {
            self.confirm_coding_controls()?;
        }

        let mut out = dev.format(self.device.output).map_err(device_error)?;
        self.fill_output_format(&mut out, None);
        dev.try_format(&mut out).map_err(device_error)?;
        let (aw, ah) = self.format_size(&mut out);
        if (aw, ah) != (w, h) {
            return Err(CodecError::InvalidConfig(format!(
                "{} encodes {aw}x{ah}, not {w}x{h}",
                self.path
            )));
        }

        let interval = Fraction::new(
            1000,
            (self.config.frame_rate * 1000.0)
                .round()
                .clamp(1.0, f64::from(u32::MAX)) as u32,
        );
        if let Err(e) = dev.set_frame_interval(self.device.output, interval) {
            log::debug!("{}: frame interval not set: {e}", self.path);
        }

        let mut cmd = uapi::v4l2_encoder_cmd {
            cmd: uapi::V4L2_ENC_CMD_STOP,
            ..Default::default()
        };
        // SAFETY: `cmd` is a valid v4l2_encoder_cmd for the duration of the call.
        self.can_stop = unsafe {
            edgefirst_v4l2::ioctl::vidioc_try_encoder_cmd(dev.as_fd().as_raw_fd(), &mut cmd)
        }
        .is_ok();
        Ok(())
    }

    /// Whether the stream is H.264 rather than JPEG pictures.
    fn is_h264(&self) -> bool {
        self.coded_fourcc == uapi::V4L2_PIX_FMT_H264
    }

    /// Writes the OUTPUT format for the configured size, with `stride` as
    /// every plane's `bytesperline` when given (the driver's choice when
    /// not).
    fn fill_output_format(&self, fmt: &mut uapi::v4l2_format, stride: Option<usize>) {
        let (w, h) = (self.config.width, self.config.height);
        let bpl = stride.map_or(0, |s| s as u32);
        // SAFETY: the buffer type selects the union member.
        unsafe {
            if self.device.multiplanar() {
                let p = fmt.pix_mp();
                p.pixelformat = self.input_fourcc;
                p.width = w;
                p.height = h;
                p.field = uapi::V4L2_FIELD_NONE;
                p.num_planes = if self.layout == Layout::Split { 2 } else { 1 };
                for plane in &mut p.plane_fmt[..usize::from(p.num_planes)] {
                    plane.bytesperline = bpl;
                    plane.sizeimage = 0;
                }
            } else {
                let p = fmt.pix();
                p.pixelformat = self.input_fourcc;
                p.width = w;
                p.height = h;
                p.field = uapi::V4L2_FIELD_NONE;
                p.bytesperline = bpl;
                p.sizeimage = 0;
            }
        }
    }

    /// Width and height of a format of either API.
    fn format_size(&self, fmt: &mut uapi::v4l2_format) -> (u32, u32) {
        // SAFETY: the buffer type selects the union member.
        unsafe {
            if self.device.multiplanar() {
                let p = fmt.pix_mp();
                (p.width, p.height)
            } else {
                let p = fmt.pix();
                (p.width, p.height)
            }
        }
    }

    fn control(&self, id: u32) -> Option<&ControlInfo> {
        self.controls.iter().find(|c| c.id == id)
    }

    /// Sets control `id` to `value`. `Ok(false)` when the device lacks it.
    fn set_control(&self, id: u32, value: i32) -> Result<bool> {
        let Some(info) = self.control(id) else {
            return Ok(false);
        };
        let applied = controls::set(&self.device.dev, info, &ControlValue::Integer(value))
            .map_err(device_error)?;
        if applied != ControlValue::Integer(value) && info.control_type != ControlType::Button {
            log::warn!("{}: {} is {applied:?}, not {value}", self.path, info.name);
        }
        Ok(true)
    }

    /// Sets a control the configuration requires.
    fn require_control(&self, id: u32, value: i32, what: &str) -> Result<()> {
        match self.set_control(id, value) {
            Ok(true) => Ok(()),
            Ok(false) => Err(CodecError::InvalidConfig(format!(
                "{} cannot set the {what}",
                self.path
            ))),
            Err(e) => Err(CodecError::InvalidConfig(format!(
                "{} rejects the {what}: {e}",
                self.path
            ))),
        }
    }

    /// Sets the profile and level again where setting the CAPTURE format
    /// changed the value the device reports.
    fn confirm_coding_controls(&self) -> Result<()> {
        for c in coding_controls(&self.config) {
            let Some(info) = self.control(c.id) else {
                continue;
            };
            match controls::get(&self.device.dev, info) {
                Ok(ControlValue::Integer(v)) if v == c.value => {}
                Ok(other) => {
                    log::debug!(
                        "{}: {} reads {other:?} after the CAPTURE format, setting it again",
                        self.path,
                        c.what
                    );
                    self.set_coding_control(&c)?;
                }
                Err(e) => log::debug!("{}: cannot read back the {}: {e}", self.path, c.what),
            }
        }
        Ok(())
    }

    /// Sets a profile or level: one the configuration asked for must take,
    /// one chosen for it falls back to the device default.
    fn set_coding_control(&self, c: &CodingControl) -> Result<()> {
        if c.required {
            return self.require_control(c.id, c.value, &c.what);
        }
        match self.set_control(c.id, c.value) {
            Ok(true) => {}
            Ok(false) => log::debug!("{}: no {} control, keeping its default", self.path, c.what),
            Err(e) => log::debug!(
                "{}: {} not set, keeping its default: {e}",
                self.path,
                c.what
            ),
        }
        Ok(())
    }

    fn apply_controls(&self) -> Result<()> {
        let cfg = &self.config;
        for (id, value) in RATE_CONTROLS {
            if let Err(e) = self.set_control(id, value) {
                log::debug!("{}: rate control {id:#x} not set: {e}", self.path);
            }
        }
        let bps = i32::try_from(cfg.target_bitrate()).unwrap_or(i32::MAX);
        match cfg.bitrate {
            Bitrate::Bps(_) => {
                self.require_control(uapi::V4L2_CID_MPEG_VIDEO_BITRATE, bps, "bitrate")?;
            }
            Bitrate::Auto => match self.set_control(uapi::V4L2_CID_MPEG_VIDEO_BITRATE, bps) {
                Ok(true) => {}
                Ok(false) => log::debug!("{}: no bitrate control, keeping its default", self.path),
                Err(e) => log::debug!(
                    "{}: bitrate {bps} not set, keeping its default: {e}",
                    self.path
                ),
            },
        }
        if let Some(gop) = cfg.gop {
            let gop = i32::try_from(gop).unwrap_or(i32::MAX);
            self.require_control(uapi::V4L2_CID_MPEG_VIDEO_GOP_SIZE, gop, "GOP size")?;
            self.set_control(uapi::V4L2_CID_MPEG_VIDEO_H264_I_PERIOD, gop)?;
        }
        for c in coding_controls(&self.config) {
            self.set_coding_control(&c)?;
        }
        // Ask for the headers joined to the first frame, where the device allows it.
        let _ = self.set_control(
            uapi::V4L2_CID_MPEG_VIDEO_HEADER_MODE,
            uapi::V4L2_MPEG_VIDEO_HEADER_MODE_JOINED_WITH_1ST_FRAME,
        );
        let repeat = i32::from(cfg.repeat_headers);
        let repeated = self.set_control(uapi::V4L2_CID_MPEG_VIDEO_REPEAT_SEQ_HEADER, repeat)?
            | self.set_control(uapi::V4L2_CID_MPEG_VIDEO_PREPEND_SPSPPS_TO_IDR, repeat)?;
        if cfg.repeat_headers && !repeated {
            return Err(CodecError::InvalidConfig(format!(
                "{} cannot repeat the SPS and PPS on key frames",
                self.path
            )));
        }
        Ok(())
    }

    /// Sets the OUTPUT format, asking for `stride` as the row stride (the
    /// device's choice when `None`), and returns the format the device set.
    fn set_output_format(&self, stride: Option<usize>) -> Result<OutputFormat> {
        let dev = &self.device.dev;
        let mut out = dev.format(self.device.output).map_err(device_error)?;
        self.fill_output_format(&mut out, stride);
        dev.set_format(&mut out).map_err(device_error)?;
        let (strides, sizes) = self.plane_formats(&mut out);
        let (_, rows) = self.format_size(&mut out);
        let (w, h) = (self.config.width as usize, self.config.height as usize);
        let luma = strides.first().copied().unwrap_or(0);
        let chroma = strides.get(1).copied().filter(|&s| s != 0).unwrap_or(luma);
        let row = copy::plane_shapes(self.config.input_format, w, h)
            .and_then(|s| s.first().map(|&(len, _)| len))
            .unwrap_or(w);
        if luma < row || (rows as usize) < h {
            return Err(CodecError::InvalidConfig(format!(
                "{} reports a {luma}-byte row stride and {rows} rows for {w}x{h} {}",
                self.path, self.config.input_format
            )));
        }
        Ok(OutputFormat {
            strides: [luma, chroma],
            rows: rows as usize,
            sizes,
        })
    }

    /// Sets the OUTPUT format for sources with rows `stride` bytes apart,
    /// allocates both queues and starts streaming.
    fn start(&mut self, stride: usize) -> Result<()> {
        let format = self.set_output_format(Some(stride))?;
        let dev = &self.device.dev;

        let min = |id| {
            self.control(id)
                .and_then(|info| controls::get(dev, info).ok())
                .and_then(|v| match v {
                    ControlValue::Integer(n) => u32::try_from(n).ok(),
                    _ => None,
                })
                .unwrap_or(0)
        };
        let output_want = OUTPUT_BUFFERS.max(min(uapi::V4L2_CID_MIN_BUFFERS_FOR_OUTPUT));
        let capture_want = CAPTURE_BUFFERS.max(min(uapi::V4L2_CID_MIN_BUFFERS_FOR_CAPTURE));

        let mut m2m = M2m::new(dev, self.device.multiplanar()).map_err(device_error)?;
        let mut capture_maps = Vec::new();
        let started = (|| {
            let output_count = m2m
                .output_mut()
                .request(Memory::DmaBuf, output_want)
                .map_err(device_error)?;
            let capture_count = m2m
                .capture_mut()
                .request(Memory::Mmap, capture_want)
                .map_err(device_error)?;
            if output_count == 0 || capture_count == 0 {
                return Err(CodecError::NoDevice(format!(
                    "{} allocated no buffers",
                    self.path
                )));
            }
            for index in 0..capture_count {
                let mut maps = m2m.capture().map(index).map_err(device_error)?;
                if maps.is_empty() {
                    return Err(CodecError::NoDevice(format!(
                        "{} CAPTURE buffer {index} has no plane",
                        self.path
                    )));
                }
                capture_maps.push(maps.swap_remove(0));
                m2m.capture()
                    .enqueue(index, &[Plane::mmap()], None)
                    .map_err(device_error)?;
            }
            m2m.stream_on().map_err(device_error)?;
            Ok(output_count)
        })();
        let output_count = match started {
            Ok(count) => count,
            Err(e) => {
                drop(capture_maps);
                release(&mut m2m);
                return Err(e);
            }
        };
        self.stream = Some(Stream {
            m2m,
            capture_maps,
            format,
            staging: None,
            next_output: 0,
            output_count,
        });
        Ok(())
    }

    /// `bytesperline` and `sizeimage` of each plane of a format.
    fn plane_formats(&self, fmt: &mut uapi::v4l2_format) -> (Vec<usize>, Vec<usize>) {
        // SAFETY: the buffer type selects the union member.
        unsafe {
            if self.device.multiplanar() {
                let p = fmt.pix_mp();
                let n = usize::from(p.num_planes).min(p.plane_fmt.len());
                p.plane_fmt[..n]
                    .iter()
                    .map(|pl| (pl.bytesperline as usize, pl.sizeimage as usize))
                    .unzip()
            } else {
                let p = fmt.pix();
                (vec![p.bytesperline as usize], vec![p.sizeimage as usize])
            }
        }
    }

    /// Stops streaming and frees the buffers; the next frame starts again.
    fn reset(&mut self) {
        if let Some(Stream {
            mut m2m,
            capture_maps,
            ..
        }) = self.stream.take()
        {
            // Unmap before the buffers are freed.
            drop(capture_maps);
            release(&mut m2m);
        }
        self.in_flight.clear();
        self.headers.clear();
        self.parameter_sets.clear();
    }

    /// Checks `src` against the configuration and returns the picture's
    /// origin in it.
    fn check_source(&self, src: &TensorDyn, opts: &FrameOptions) -> Result<(usize, usize)> {
        let cfg = &self.config;
        let invalid = |what: String| Err(CodecError::InvalidInput(what));
        if src.format() != Some(cfg.input_format) {
            return invalid(format!(
                "source format {:?} is not the configured {}",
                src.format(),
                cfg.input_format
            ));
        }
        if src.dtype() != DType::U8 {
            return invalid(format!(
                "source elements are {:?}; {} is 8-bit",
                src.dtype(),
                cfg.input_format
            ));
        }
        let (sw, sh) = match (src.width(), src.height()) {
            (Some(w), Some(h)) => (w, h),
            _ => return invalid("source is not an image".into()),
        };
        let (w, h) = (cfg.width as usize, cfg.height as usize);
        if src.view_origin().is_some() && self.layout != Layout::Packed {
            return invalid(format!(
                "a view of a {} image cannot be encoded in place; crop with FrameOptions::crop",
                cfg.input_format
            ));
        }
        let Some(crop) = opts.crop else {
            if (sw, sh) != (w, h) {
                return invalid(format!("source is {sw}x{sh}, the encoder {w}x{h}"));
            }
            return Ok((0, 0));
        };
        if (crop.width, crop.height) != (w, h) {
            return invalid(format!(
                "crop is {}x{}, the encoder {w}x{h}",
                crop.width, crop.height
            ));
        }
        if crop.x + crop.width > sw || crop.y + crop.height > sh {
            return invalid(format!(
                "crop {}x{}+{}+{} is outside the {sw}x{sh} source",
                crop.width, crop.height, crop.x, crop.y
            ));
        }
        let (hsub, vsub) = chroma_subsampling(cfg.input_format);
        if !crop.x.is_multiple_of(hsub as usize) || !crop.y.is_multiple_of(vsub as usize) {
            return invalid(format!(
                "crop origin {},{} is not on a {} chroma sample",
                crop.x, crop.y, cfg.input_format
            ));
        }
        Ok((crop.x, crop.y))
    }

    /// Queues `src`, or a copy of it in the device's layout, and waits
    /// until the device has read it.
    fn submit(
        &mut self,
        src: &TensorDyn,
        origin: (usize, usize),
        opts: &FrameOptions,
    ) -> Result<()> {
        let stride = src
            .effective_row_stride()
            .ok_or_else(|| CodecError::InvalidInput("source has no row stride".into()))?;
        let chroma = if self.layout != Layout::Packed && src.is_multiplane() {
            Some(src.chroma_dyn().ok_or_else(|| {
                CodecError::InvalidInput("multi-plane source has no chroma plane".into())
            })?)
        } else {
            None
        };
        if self.stream.is_none() {
            self.start(stride)?;
        }
        let import = match self.source_imports(src, chroma.as_ref(), stride, origin)? {
            Ok(imports) => Some(imports),
            Err(reason) => {
                self.note_copy(src, opts.crop.is_some(), &reason);
                self.copy_to_staging(src, origin, stride)?;
                None
            }
        };
        self.transfer = if import.is_some() {
            Transfer::ZeroCopy
        } else {
            Transfer::CpuCopy
        };

        // Force a key frame, for H.264 only.
        if opts.keyframe && self.coded_fourcc == uapi::V4L2_PIX_FMT_H264 {
            let info = self
                .controls
                .iter()
                .find(|c| c.id == uapi::V4L2_CID_MPEG_VIDEO_FORCE_KEY_FRAME)
                .ok_or_else(|| {
                    CodecError::InvalidConfig(format!("{} cannot force key frames", self.path))
                })?;
            controls::set(&self.device.dev, info, &ControlValue::Integer(1))
                .map_err(device_error)?;
        }

        let seq = self.next_seq;
        self.next_seq += 1;
        let stream = self.stream.as_mut().expect("started above");
        let index = stream.next_output;
        stream.next_output = (index + 1) % stream.output_count;
        let stream = self.stream.as_ref().expect("started above");
        let luma_fd;
        let chroma_fd;
        let planes: Vec<Plane<'_>> = match &import {
            Some(imports) => {
                luma_fd = src.dmabuf()?;
                chroma_fd = chroma.as_ref().map(TensorDyn::dmabuf).transpose()?;
                let fds = [luma_fd, chroma_fd.unwrap_or(luma_fd)];
                let lengths = [dmabuf_size(fds[0])?, dmabuf_size(fds[1])?];
                imports
                    .iter()
                    .map(|p| Plane::DmaBuf {
                        fd: fds[p.buffer],
                        length: lengths[p.buffer] as u32,
                        bytesused: p.end as u32,
                        data_offset: p.offset as u32,
                    })
                    .collect()
            }
            None => {
                let staging = stream.staging.as_ref().expect("staged above");
                let fd = staging.dmabuf()?;
                let length = dmabuf_size(fd)?;
                let (planes, _) = stream.format.buffer_planes(self.layout);
                planes
                    .iter()
                    .map(|&(offset, size)| Plane::DmaBuf {
                        fd,
                        length: length as u32,
                        bytesused: (offset + size) as u32,
                        data_offset: offset as u32,
                    })
                    .collect()
            }
        };
        stream
            .m2m
            .output()
            .enqueue(index, &planes, Some(Duration::from_micros(seq)))
            .map_err(device_error)?;
        self.in_flight.push_back((seq, opts.pts));

        let deadline = Instant::now() + INPUT_TIMEOUT;
        loop {
            let left = deadline.saturating_duration_since(Instant::now());
            if left.is_zero() {
                return Err(CodecError::Timeout(format!(
                    "{} did not read the source within {INPUT_TIMEOUT:?}",
                    self.path
                )));
            }
            let stream = self.stream.as_ref().expect("started above");
            let ready = stream.m2m.wait(Some(left)).map_err(device_error)?;
            if ready.capture {
                self.collect()?;
            }
            let stream = self.stream.as_ref().expect("started above");
            if ready.output {
                while let Some(done) = stream.m2m.output().dequeue().map_err(device_error)? {
                    if done.index == index {
                        return Ok(());
                    }
                }
            }
        }
    }

    /// The planes to import `src` from, or why the device cannot read it in
    /// place.
    fn source_imports(
        &self,
        src: &TensorDyn,
        chroma: Option<&TensorDyn>,
        stride: usize,
        origin: (usize, usize),
    ) -> Result<std::result::Result<Vec<ImportPlane>, String>> {
        if src.memory() != TensorMemory::DmaBuf {
            return Ok(Err(format!("the source is in {:?} memory", src.memory())));
        }
        let format = self.config.input_format;
        let stream = self
            .stream
            .as_ref()
            .expect("source_imports runs while streaming");
        let base = src.plane_offset().unwrap_or(0);
        let luma_fd = src.dmabuf()?;
        let luma_length = dmabuf_size(luma_fd)?;
        let (chroma_stride, chroma_base, chroma_length, same_buffer) = match chroma {
            Some(c) => {
                let fd = c.dmabuf()?;
                (
                    c.effective_row_stride().unwrap_or(stride),
                    c.plane_offset().unwrap_or(0),
                    dmabuf_size(fd)?,
                    same_dmabuf(luma_fd, fd)?,
                )
            }
            None => {
                let planes = format
                    .plane_table(src.width().unwrap_or(0), src.height().unwrap_or(0), stride)
                    .ok_or_else(|| {
                        CodecError::InvalidInput(format!("no plane layout for {format}"))
                    })?;
                let offset = planes.get(1).map_or(0, |p| p.offset as usize);
                (stride, base + offset, luma_length, true)
            }
        };
        let planes = source_planes(
            self.layout,
            format,
            [stride, chroma_stride],
            base,
            chroma_base,
            origin,
            self.config.height as usize,
        );
        Ok(import_planes(
            self.layout,
            self.device.multiplanar(),
            &stream.format,
            &planes,
            [stride, chroma_stride],
            [luma_length, chroma_length],
            (usize::from(chroma.is_some()), same_buffer),
        ))
    }

    /// Copies the picture at `origin` of `src` into the staging buffer, in
    /// the device's layout, repeating the last row of each plane down to
    /// the format's height.
    fn copy_to_staging(
        &mut self,
        src: &TensorDyn,
        origin: (usize, usize),
        stride: usize,
    ) -> Result<()> {
        let layout = self.layout;
        let format = self.config.input_format;
        let size = (self.config.width as usize, self.config.height as usize);
        let stream = self
            .stream
            .as_mut()
            .expect("copy_to_staging runs while streaming");
        if stream.staging.is_none() {
            let (_, len) = stream.format.buffer_planes(layout);
            let buffer = TensorDyn::new(&[len], DType::U8, Some(TensorMemory::DmaBuf), None)?;
            if buffer.memory() != TensorMemory::DmaBuf {
                return Err(CodecError::NoDevice(
                    "no DMA-BUF allocator for a staging buffer".into(),
                ));
            }
            stream.staging = Some(buffer);
        }
        let staging = stream.staging.as_ref().expect("allocated above");
        let to = stream.format.picture_planes(layout);
        let shapes = copy::plane_shapes(format, size.0, size.1)
            .ok_or(CodecError::UnsupportedFormat(format))?;
        let mut map = staging.map_bytes(CpuAccess::Write)?;
        copy::copy_source(format, size, src, origin, stride, &mut map, &to)?;
        for (i, (&plane, &shape)) in to.iter().zip(&shapes).enumerate() {
            let rows = if i == 0 {
                stream.format.rows
            } else {
                stream.format.rows.div_ceil(2)
            };
            copy::repeat_last_row(&mut map, plane, shape, rows)
                .map_err(CodecError::InvalidInput)?;
        }
        Ok(())
    }

    /// Logs a source copied into the device's layout: at info the first time
    /// for its shape, at debug afterwards.
    fn note_copy(&mut self, src: &TensorDyn, cropped: bool, reason: &str) {
        let size = (src.width().unwrap_or(0), src.height().unwrap_or(0));
        let stride = src.effective_row_stride().unwrap_or(0);
        let memory = format!("{:?}", src.memory());
        let shape = (self.config.input_format, size, stride, memory, cropped);
        if self.copied.len() >= LOGGED_SHAPES && !self.copied.contains(&shape) {
            self.copied.clear();
        }
        let level = if self.copied.insert(shape) {
            log::Level::Info
        } else {
            log::Level::Debug
        };
        log::log!(
            level,
            "{}: copying {} {}x{} sources ({stride}-byte rows, {:?} memory{}) into the device's layout: {reason}",
            self.path,
            self.config.input_format,
            size.0,
            size.1,
            src.memory(),
            if cropped { ", cropped" } else { "" },
        );
    }

    /// Allocates a source the device reads in place.
    fn source_tensor(&self) -> Result<TensorDyn> {
        let format = match &self.stream {
            Some(stream) => stream.format.clone(),
            None => self.set_output_format(None)?,
        };
        let pixel_format = self.config.input_format;
        let (w, h) = (self.config.width as usize, self.config.height as usize);
        let allocate = |len: usize| -> Result<std::os::fd::OwnedFd> {
            let t = TensorDyn::new(&[len], DType::U8, Some(TensorMemory::DmaBuf), None)?;
            if t.memory() != TensorMemory::DmaBuf {
                return Err(CodecError::NoDevice("no DMA-BUF allocator".into()));
            }
            Ok(t.clone_fd()?)
        };
        let (planes, len) = format.buffer_planes(self.layout);
        let mut t = match self.layout {
            Layout::Packed => {
                let fd = allocate(len)?;
                let mut t = Tensor::<u8>::from_fd(fd, &[h, w, pixel_format.channels()], None)?;
                t.set_format(pixel_format)?;
                t
            }
            Layout::Contiguous | Layout::Split => {
                let (luma_fd, chroma_fd, chroma_offset) = if self.layout == Layout::Split {
                    (allocate(planes[0].1)?, allocate(planes[1].1)?, 0)
                } else {
                    let fd = allocate(len)?;
                    let chroma = fd.try_clone()?;
                    (fd, chroma, format.strides[0] * format.rows)
                };
                let luma = Tensor::<u8>::from_fd(luma_fd, &[h, w], None)?;
                let mut chroma = Tensor::<u8>::from_fd(chroma_fd, &[h.div_ceil(2), w], None)?;
                chroma.set_row_stride_unchecked(format.strides[1]);
                chroma.set_plane_offset(chroma_offset);
                Tensor::from_planes(luma, chroma, pixel_format)?
            }
        };
        t.set_row_stride(format.strides[0])?;
        Ok(TensorDyn::from(t))
    }

    /// Dequeues every finished bitstream buffer. Returns whether the drain
    /// has ended: a buffer carried `V4L2_BUF_FLAG_LAST`, or the queue reports
    /// end of stream (`EPIPE`), after which nothing more is dequeued until
    /// the encoder restarts.
    fn collect(&mut self) -> Result<bool> {
        loop {
            let stream = self.stream.as_ref().expect("collect runs while streaming");
            let done = match stream.m2m.capture().dequeue() {
                Ok(Some(done)) => done,
                Ok(None) => return Ok(false),
                Err(e) if e.kind() == edgefirst_v4l2::ErrorKind::EndOfStream => return Ok(true),
                Err(e) => return Err(device_error(e)),
            };
            let last = done.flags.is_last();
            let usage = done.planes().first().copied().unwrap_or_default();
            let start = usage.data_offset as usize;
            let end = usage.bytesused as usize;
            let map = &stream.capture_maps[done.index as usize];
            let bytes = if done.flags.is_error() || end <= start || end > map.len() {
                if done.flags.is_error() {
                    log::warn!(
                        "{}: CAPTURE buffer {} has the error flag",
                        self.path,
                        done.index
                    );
                }
                Vec::new()
            } else {
                // SAFETY: the buffer is dequeued, so the device is not writing it,
                // and `start..end` is inside the mapping.
                unsafe { std::slice::from_raw_parts(map.as_ptr().add(start), end - start) }.to_vec()
            };
            stream
                .m2m
                .capture()
                .enqueue(done.index, &[Plane::mmap()], None)
                .map_err(device_error)?;
            if !bytes.is_empty() {
                self.accept(bytes, done.flags.is_keyframe(), done.timestamp);
            }
            if last {
                return Ok(true);
            }
        }
    }

    /// Queues an access unit, or holds a header-only buffer for the next
    /// one.
    fn accept(&mut self, bytes: Vec<u8>, keyframe_flag: bool, timestamp: Duration) {
        if !self.is_h264() {
            let pts = self.take_pts(timestamp.as_micros() as u64);
            self.pending.push_back(EncodedFrame {
                keyframe: true,
                data: bytes,
                pts,
            });
            return;
        }
        let au = AccessUnit::parse(&bytes);
        if au.ends_previous() {
            match self.pending.back_mut() {
                Some(previous) => previous.data.extend_from_slice(&bytes),
                None => log::debug!(
                    "{}: dropped {} bytes ending an access unit already returned",
                    self.path,
                    bytes.len()
                ),
            }
            return;
        }
        if !au.has_picture {
            // Parameter sets, SEI and delimiters precede a picture.
            self.headers.extend_from_slice(&bytes);
            return;
        }
        let pts = self.take_pts(timestamp.as_micros() as u64);
        let mut data = std::mem::take(&mut self.headers);
        let au = if data.is_empty() {
            data = bytes;
            au
        } else {
            data.extend_from_slice(&bytes);
            AccessUnit::parse(&data)
        };
        if au.parameter_sets {
            self.parameter_sets = h264::parameter_sets(&data);
        } else if au.idr && self.config.repeat_headers && !self.parameter_sets.is_empty() {
            // Prepend the parameter sets to an IDR that came without them.
            let mut with_headers = self.parameter_sets.clone();
            with_headers.extend_from_slice(&data);
            data = with_headers;
        }
        self.pending.push_back(EncodedFrame {
            keyframe: keyframe_flag || au.idr,
            data,
            pts,
        });
    }

    /// The PTS of the source tagged `seq`, dropping older tags whose access
    /// units the device skipped.
    fn take_pts(&mut self, seq: u64) -> u64 {
        while let Some(&(s, pts)) = self.in_flight.front() {
            self.in_flight.pop_front();
            if s == seq {
                return pts;
            }
            if s > seq {
                self.in_flight.push_front((s, pts));
                break;
            }
            log::debug!("{}: no access unit for source {s}", self.path);
        }
        log::warn!("{}: access unit with unknown timestamp {seq}", self.path);
        0
    }

    /// Waits up to `timeout` for a bitstream buffer and collects it.
    fn wait_capture(&mut self, timeout: Duration) -> Result<bool> {
        let deadline = Instant::now() + timeout;
        loop {
            let left = deadline.saturating_duration_since(Instant::now());
            if left.is_zero() {
                return Ok(false);
            }
            let stream = self
                .stream
                .as_ref()
                .expect("wait_capture runs while streaming");
            let ready = stream.m2m.wait(Some(left)).map_err(device_error)?;
            if ready.capture {
                return self.collect();
            }
            if ready.is_empty() {
                return Ok(false);
            }
            // Nothing else is in flight; drop stray OUTPUT completions.
            while stream
                .m2m
                .output()
                .dequeue()
                .map_err(device_error)?
                .is_some()
            {}
        }
    }
}

impl EncoderBackend for V4l2Encoder {
    fn encode(&mut self, src: &TensorDyn, opts: &FrameOptions) -> Result<Option<EncodedFrame>> {
        let origin = self.check_source(src, opts)?;
        if let Err(e) = self.submit(src, origin, opts) {
            if matches!(e, CodecError::Timeout(_) | CodecError::Device { .. }) {
                self.reset();
            }
            return Err(e);
        }
        if self.pending.is_empty() && self.stream.is_some() {
            self.wait_capture(OUTPUT_GRACE)?;
        }
        Ok(self.pending.pop_front())
    }

    fn flush(&mut self) -> Result<Vec<EncodedFrame>> {
        if self.stream.is_none() {
            return Ok(self.pending.drain(..).collect());
        }
        if self.can_stop {
            encoder_cmd(&self.device.dev, uapi::V4L2_ENC_CMD_STOP)?;
            let deadline = Instant::now() + DRAIN_TIMEOUT;
            loop {
                let left = deadline.saturating_duration_since(Instant::now());
                if left.is_zero() {
                    self.reset();
                    return Err(CodecError::Timeout(format!(
                        "{} did not finish draining within {DRAIN_TIMEOUT:?}",
                        self.path
                    )));
                }
                if self.wait_capture(left)? {
                    break;
                }
            }
        } else {
            // Without ENCODER_CMD, wait for the access units of every source
            // the device has read.
            while !self.in_flight.is_empty() {
                if !self.wait_capture(OUTPUT_GRACE)? {
                    break;
                }
            }
        }
        // The next frame sets up a new stream, which opens with an IDR and its
        // parameter sets.
        self.reset();
        Ok(self.pending.drain(..).collect())
    }

    fn device(&self) -> String {
        self.path.clone()
    }

    fn info(&self) -> EncoderInfo {
        EncoderInfo {
            engine: Engine::V4l2,
            device: self.path.clone(),
            hardware: true,
            transfer: self.transfer,
        }
    }

    fn create_source(&self) -> Result<TensorDyn> {
        self.source_tensor()
    }
}

impl Drop for V4l2Encoder {
    fn drop(&mut self) {
        self.reset();
    }
}

/// Issues `VIDIOC_ENCODER_CMD` with `cmd`.
fn encoder_cmd(dev: &edgefirst_v4l2::device::Device, cmd: u32) -> Result<()> {
    let mut c = uapi::v4l2_encoder_cmd {
        cmd,
        ..Default::default()
    };
    // SAFETY: `c` is a valid v4l2_encoder_cmd for the duration of the call.
    unsafe { edgefirst_v4l2::ioctl::vidioc_encoder_cmd(dev.as_fd().as_raw_fd(), &mut c) }
        .map(|_| ())
        .map_err(|errno| CodecError::Device {
            op: "VIDIOC_ENCODER_CMD".into(),
            source: std::io::Error::from_raw_os_error(errno as i32),
        })
}

/// Stops both queues and frees their buffers (`VIDIOC_REQBUFS(0)`).
fn release(m2m: &mut M2m) {
    if let Err(e) = m2m.capture_mut().free() {
        log::debug!("freeing the CAPTURE queue: {e}");
    }
    if let Err(e) = m2m.output_mut().free() {
        log::debug!("freeing the OUTPUT queue: {e}");
    }
}

/// Size of a DMA-BUF in bytes.
fn dmabuf_size(fd: BorrowedFd<'_>) -> Result<usize> {
    let mut file = std::fs::File::from(fd.try_clone_to_owned()?);
    Ok(file.seek(SeekFrom::End(0))? as usize)
}

/// Whether `a` and `b` refer to the same DMA-BUF.
fn same_dmabuf(a: BorrowedFd<'_>, b: BorrowedFd<'_>) -> Result<bool> {
    use std::os::unix::fs::MetadataExt;
    let stat = |fd: BorrowedFd<'_>| std::fs::File::from(fd.try_clone_to_owned()?).metadata();
    let (a, b) = (stat(a)?, stat(b)?);
    Ok((a.dev(), a.ino()) == (b.dev(), b.ino()))
}

/// Rate control, as `(control, value)`: constant bitrate, regulated per frame
/// and per macroblock.
const RATE_CONTROLS: [(u32, i32); 3] = [
    (
        uapi::V4L2_CID_MPEG_VIDEO_BITRATE_MODE,
        uapi::V4L2_MPEG_VIDEO_BITRATE_MODE_CBR,
    ),
    (uapi::V4L2_CID_MPEG_VIDEO_FRAME_RC_ENABLE, 1),
    (uapi::V4L2_CID_MPEG_VIDEO_MB_RC_ENABLE, 1),
];

/// A profile or level control to set.
#[derive(Debug, PartialEq)]
struct CodingControl {
    id: u32,
    value: i32,
    what: String,
    /// Asked for by the configuration rather than chosen for it.
    required: bool,
}

/// The profile and level to set for `cfg`: the ones it asks for, or the ones
/// [`EncoderConfig::target_profile`] and [`EncoderConfig::target_level`]
/// choose.
fn coding_controls(cfg: &EncoderConfig) -> Vec<CodingControl> {
    let profile = cfg.target_profile();
    let mut list = vec![CodingControl {
        id: uapi::V4L2_CID_MPEG_VIDEO_H264_PROFILE,
        value: profile_value(profile),
        what: format!("H.264 profile {profile:?}"),
        required: cfg.profile.is_some(),
    }];
    if let Some(level) = cfg.target_level() {
        list.push(CodingControl {
            id: uapi::V4L2_CID_MPEG_VIDEO_H264_LEVEL,
            value: level_value(level),
            what: format!("H.264 level {level:?}"),
            required: cfg.level.is_some(),
        });
    }
    list
}

/// The `V4L2_CID_MPEG_VIDEO_H264_PROFILE` menu value of `profile`.
fn profile_value(profile: H264Profile) -> i32 {
    match profile {
        H264Profile::Baseline => uapi::V4L2_MPEG_VIDEO_H264_PROFILE_BASELINE,
        H264Profile::ConstrainedBaseline => uapi::V4L2_MPEG_VIDEO_H264_PROFILE_CONSTRAINED_BASELINE,
        H264Profile::Main => uapi::V4L2_MPEG_VIDEO_H264_PROFILE_MAIN,
        H264Profile::High => uapi::V4L2_MPEG_VIDEO_H264_PROFILE_HIGH,
    }
}

/// The `V4L2_CID_MPEG_VIDEO_H264_LEVEL` menu value of `level`.
fn level_value(level: H264Level) -> i32 {
    use H264Level as L;
    match level {
        L::L1_0 => uapi::V4L2_MPEG_VIDEO_H264_LEVEL_1_0,
        L::L1b => uapi::V4L2_MPEG_VIDEO_H264_LEVEL_1B,
        L::L1_1 => uapi::V4L2_MPEG_VIDEO_H264_LEVEL_1_1,
        L::L1_2 => uapi::V4L2_MPEG_VIDEO_H264_LEVEL_1_2,
        L::L1_3 => uapi::V4L2_MPEG_VIDEO_H264_LEVEL_1_3,
        L::L2_0 => uapi::V4L2_MPEG_VIDEO_H264_LEVEL_2_0,
        L::L2_1 => uapi::V4L2_MPEG_VIDEO_H264_LEVEL_2_1,
        L::L2_2 => uapi::V4L2_MPEG_VIDEO_H264_LEVEL_2_2,
        L::L3_0 => uapi::V4L2_MPEG_VIDEO_H264_LEVEL_3_0,
        L::L3_1 => uapi::V4L2_MPEG_VIDEO_H264_LEVEL_3_1,
        L::L3_2 => uapi::V4L2_MPEG_VIDEO_H264_LEVEL_3_2,
        L::L4_0 => uapi::V4L2_MPEG_VIDEO_H264_LEVEL_4_0,
        L::L4_1 => uapi::V4L2_MPEG_VIDEO_H264_LEVEL_4_1,
        L::L4_2 => uapi::V4L2_MPEG_VIDEO_H264_LEVEL_4_2,
        L::L5_0 => uapi::V4L2_MPEG_VIDEO_H264_LEVEL_5_0,
        L::L5_1 => uapi::V4L2_MPEG_VIDEO_H264_LEVEL_5_1,
        L::L5_2 => uapi::V4L2_MPEG_VIDEO_H264_LEVEL_5_2,
        L::L6_0 => uapi::V4L2_MPEG_VIDEO_H264_LEVEL_6_0,
        L::L6_1 => uapi::V4L2_MPEG_VIDEO_H264_LEVEL_6_1,
        L::L6_2 => uapi::V4L2_MPEG_VIDEO_H264_LEVEL_6_2,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nv12_prefers_the_two_plane_layout() {
        assert_eq!(
            input_candidates(PixelFormat::Nv12),
            vec![
                (uapi::V4L2_PIX_FMT_NV12M, Layout::Split),
                (uapi::V4L2_PIX_FMT_NV12, Layout::Contiguous)
            ]
        );
        assert_eq!(
            input_candidates(PixelFormat::Yuyv),
            vec![(uapi::V4L2_PIX_FMT_YUYV, Layout::Packed)]
        );
        assert!(input_candidates(PixelFormat::PlanarRgb).is_empty());
        assert!(input_candidates(PixelFormat::Nv16).is_empty());
    }

    #[test]
    fn whole_nv12_frame_planes() {
        // 1920x1080 NV12 with a 2048-byte stride at offset 4096; chroma right
        // after the luma rows.
        let base = 4096;
        let chroma = base + 2048 * 1080;
        let p = source_planes(
            Layout::Split,
            PixelFormat::Nv12,
            [2048, 2048],
            base,
            chroma,
            (0, 0),
            1080,
        );
        assert_eq!(
            p,
            vec![
                SourcePlane {
                    offset: base,
                    rows: 1080
                },
                SourcePlane {
                    offset: chroma,
                    rows: 540
                }
            ]
        );
    }

    #[test]
    fn cropped_nv12_planes_move_by_row_and_column() {
        let stride = 1920;
        let chroma = stride * 1080;
        let p = source_planes(
            Layout::Split,
            PixelFormat::Nv12,
            [stride, stride],
            0,
            chroma,
            (64, 32),
            720,
        );
        assert_eq!(p[0].offset, 32 * stride + 64);
        assert_eq!(p[1].offset, chroma + 16 * stride + 64);
        assert_eq!((p[0].rows, p[1].rows), (720, 360));
    }

    #[test]
    fn cropped_packed_planes_use_bytes_per_pixel() {
        let p = source_planes(
            Layout::Packed,
            PixelFormat::Yuyv,
            [2560, 2560],
            0,
            0,
            (10, 3),
            480,
        );
        assert_eq!(
            p,
            vec![SourcePlane {
                offset: 3 * 2560 + 20,
                rows: 480
            }]
        );
    }

    #[test]
    fn menu_values_follow_the_kernel_enums() {
        assert_eq!(profile_value(H264Profile::High), 4);
        assert_eq!(profile_value(H264Profile::ConstrainedBaseline), 1);
        assert_eq!(level_value(H264Level::L1_0), 0);
        assert_eq!(level_value(H264Level::L1b), 1);
        assert_eq!(level_value(H264Level::L4_0), 11);
        assert_eq!(level_value(H264Level::L6_2), 19);
    }

    #[test]
    fn coding_controls_carry_the_requested_profile_and_level() {
        let summary = |cfg: &EncoderConfig| -> Vec<(u32, i32, bool)> {
            coding_controls(cfg)
                .into_iter()
                .map(|c| (c.id, c.value, c.required))
                .collect()
        };
        let profile = uapi::V4L2_CID_MPEG_VIDEO_H264_PROFILE;
        let level = uapi::V4L2_CID_MPEG_VIDEO_H264_LEVEL;
        let mut cfg = EncoderConfig::h264(640, 480, PixelFormat::Nv12, 30.0);
        // Unset: High at level 3.0, both best effort.
        assert_eq!(summary(&cfg), vec![(profile, 4, false), (level, 8, false)]);
        cfg.profile = Some(H264Profile::Main);
        cfg.level = Some(H264Level::L4_0);
        assert_eq!(summary(&cfg), vec![(profile, 2, true), (level, 11, true)]);
    }

    /// The one-plane layout of a device that pads 1080 rows to 1088.
    fn padded_1080p() -> OutputFormat {
        OutputFormat {
            strides: [1920, 1920],
            rows: 1088,
            sizes: vec![1920 * 1632],
        }
    }

    fn nv12_planes(stride: usize, chroma: usize, origin: (usize, usize)) -> Vec<SourcePlane> {
        source_planes(
            Layout::Contiguous,
            PixelFormat::Nv12,
            [stride, stride],
            0,
            chroma,
            origin,
            1080,
        )
    }

    #[test]
    fn a_tight_nv12_image_is_copied_where_the_device_pads_its_rows() {
        let planes = nv12_planes(1920, 1920 * 1080, (0, 0));
        let len = 1920 * 1620;
        let err = import_planes(
            Layout::Contiguous,
            true,
            &padded_1080p(),
            &planes,
            [1920, 1920],
            [len, len],
            (0, true),
        )
        .unwrap_err();
        assert!(err.contains("chroma 2088960 bytes after"), "{err}");
    }

    #[test]
    fn an_image_in_the_devices_layout_is_imported() {
        let planes = nv12_planes(1920, 1920 * 1088, (0, 0));
        let len = 1920 * 1632;
        let imports = import_planes(
            Layout::Contiguous,
            true,
            &padded_1080p(),
            &planes,
            [1920, 1920],
            [len, len],
            (1, true),
        )
        .unwrap();
        assert_eq!(
            imports,
            vec![ImportPlane {
                buffer: 0,
                offset: 0,
                end: 1920 * 1088 + 1920 * 540,
            }]
        );
    }

    #[test]
    fn a_stride_or_buffer_the_device_does_not_take_is_copied() {
        let device = padded_1080p();
        let planes = nv12_planes(2048, 2048 * 1088, (0, 0));
        let err = import_planes(
            Layout::Contiguous,
            true,
            &device,
            &planes,
            [2048, 2048],
            [1 << 22, 1 << 22],
            (0, true),
        )
        .unwrap_err();
        assert!(err.contains("2048 apart"), "{err}");

        let planes = nv12_planes(1920, 1920 * 1088, (0, 0));
        let err = import_planes(
            Layout::Contiguous,
            true,
            &device,
            &planes,
            [1920, 1920],
            [1920 * 1088, 1920 * 1088],
            (1, false),
        )
        .unwrap_err();
        assert!(err.contains("another"), "{err}");
    }

    #[test]
    fn a_crop_is_imported_in_two_planes() {
        // 1280x720 from a 1920x1080 source, on a device that takes the
        // source's stride.
        let device = OutputFormat {
            strides: [1920, 1920],
            rows: 720,
            sizes: vec![1920 * 720, 1920 * 360],
        };
        let chroma = 1920 * 1080;
        let planes = source_planes(
            Layout::Split,
            PixelFormat::Nv12,
            [1920, 1920],
            0,
            chroma,
            (320, 180),
            720,
        );
        let len = 1920 * 1620;
        let imports = import_planes(
            Layout::Split,
            true,
            &device,
            &planes,
            [1920, 1920],
            [len, len],
            (0, true),
        )
        .unwrap();
        assert_eq!(imports[0].offset, 180 * 1920 + 320);
        assert_eq!(imports[1].offset, chroma + 90 * 1920 + 320);
        assert_eq!(imports[1].buffer, 0);
        assert!(import_planes(
            Layout::Contiguous,
            true,
            &device,
            &planes,
            [1920, 1920],
            [len, len],
            (0, true),
        )
        .is_err());
    }

    #[test]
    fn staging_buffers_follow_the_device_layout() {
        let f = padded_1080p();
        assert_eq!(
            f.buffer_planes(Layout::Contiguous),
            (vec![(0, 1920 * 1632)], 1920 * 1632)
        );
        assert_eq!(f.picture_planes(Layout::Contiguous)[1].offset, 1920 * 1088);
        let split = OutputFormat {
            strides: [672, 672],
            rows: 480,
            sizes: vec![672 * 480, 672 * 240],
        };
        // The chroma plane starts on the next 4 KiB boundary.
        let (planes, len) = split.buffer_planes(Layout::Split);
        assert_eq!(planes, vec![(0, 672 * 480), (323_584, 672 * 240)]);
        assert_eq!(len, 323_584 + 672 * 240);
        assert_eq!(split.picture_planes(Layout::Split)[1].offset, 323_584);
    }
}
