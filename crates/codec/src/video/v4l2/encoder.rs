// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! H.264 encoding on a V4L2 stateful encoder (kernel
//! `Documentation/userspace-api/media/v4l/dev-encoder.rst`).
//!
//! Source tensors are imported on the OUTPUT queue as DMA-BUF planes. NV12
//! uses the two-plane `NV12M` layout where the device offers it, so each
//! plane carries its own offset: that is what makes crops and separately
//! allocated chroma planes possible without a copy. The bitstream comes back
//! in MMAP CAPTURE buffers and is copied out.
//!
//! V4L2 keeps buffer timestamps at microsecond resolution, so each source is
//! tagged with a sequence number and the caller's PTS is looked up again
//! when its access unit comes back.

use std::collections::VecDeque;
use std::io::{Seek, SeekFrom};
use std::os::fd::{AsFd, AsRawFd, BorrowedFd};
use std::time::{Duration, Instant};

use edgefirst_tensor::{PixelFormat, TensorDyn, TensorMemory};
use edgefirst_v4l2::controls::{self, ControlInfo, ControlType, ControlValue};
use edgefirst_v4l2::device::Fraction;
use edgefirst_v4l2::m2m::M2m;
use edgefirst_v4l2::queue::{Mapping, Memory, Plane};
use edgefirst_v4l2::uapi;

use super::{candidates, device_error, M2mDevice};
use crate::video::h264::{self, AccessUnit};
use crate::video::{
    chroma_subsampling, Bitrate, EncodedFrame, EncoderBackend, EncoderConfig, FrameOptions,
    H264Level, H264Profile,
};
use crate::{CodecError, Result};

/// Environment variable naming the encoder node to use.
const ENV_ENCODER: &str = "EDGEFIRST_CODEC_V4L2_ENCODER";
/// How long the device may take to read one source frame.
const INPUT_TIMEOUT: Duration = Duration::from_secs(2);
/// How long to wait, once a source is read, for its access unit.
const OUTPUT_GRACE: Duration = Duration::from_millis(100);
/// How long a drain may take.
const DRAIN_TIMEOUT: Duration = Duration::from_secs(2);
/// CAPTURE (bitstream) buffers, at least.
const CAPTURE_BUFFERS: u32 = 4;
/// OUTPUT (source) buffer slots, at least. A source is in flight only
/// until the device has read it, so few are needed.
const OUTPUT_BUFFERS: u32 = 2;

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
/// `chroma_base` that of the chroma plane in its buffer; `crop` is the
/// picture's origin in pixels.
fn source_planes(
    layout: Layout,
    format: PixelFormat,
    stride: usize,
    base: usize,
    chroma_base: usize,
    (x, y): (usize, usize),
    height: usize,
) -> Vec<SourcePlane> {
    let bpp = format.channels();
    let luma = SourcePlane {
        offset: base + y * stride + x * bpp,
        rows: height,
    };
    match layout {
        Layout::Packed => vec![luma],
        Layout::Contiguous | Layout::Split => {
            let (_, vsub) = chroma_subsampling(format);
            let chroma = SourcePlane {
                offset: chroma_base + (y / vsub as usize) * stride + x,
                rows: height / vsub as usize,
            };
            vec![luma, chroma]
        }
    }
}

/// The device's streaming state, created by the first frame.
struct Stream {
    m2m: M2m,
    capture_maps: Vec<Mapping>,
    /// Row stride every source of this stream has.
    stride: usize,
    /// `sizeimage` of each OUTPUT plane.
    min_lengths: Vec<usize>,
    next_output: u32,
    output_count: u32,
}

pub(crate) struct V4l2Encoder {
    device: M2mDevice,
    config: EncoderConfig,
    path: String,
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
}

impl V4l2Encoder {
    /// Opens the first device that encodes `config.codec` from
    /// `config.input_format`, and applies the configuration.
    pub fn open(config: EncoderConfig) -> Result<Self> {
        let formats = input_candidates(config.input_format);
        if formats.is_empty() {
            return Err(CodecError::UnsupportedFormat(config.input_format));
        }
        let mut found_encoder = false;
        let mut last_err = None;
        for path in candidates(ENV_ENCODER) {
            let Some(device) = M2mDevice::open(&path) else {
                continue;
            };
            if !device.lists(device.capture, uapi::V4L2_PIX_FMT_H264) {
                continue;
            }
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
            };
            match enc.configure() {
                Ok(()) => {
                    log::info!(
                        "h264 encoder {} ({}, {:?})",
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
        Err(match (last_err, found_encoder) {
            (Some(e), _) => e,
            (None, true) => CodecError::UnsupportedFormat(config.input_format),
            (None, false) => CodecError::NoDevice("no V4L2 H.264 encoder found".into()),
        })
    }

    /// Sets the coded format, checks the picture size and applies the
    /// controls. Buffers wait for the first frame.
    fn configure(&mut self) -> Result<()> {
        let (w, h) = (self.config.width, self.config.height);
        let dev = &self.device.dev;

        let mut cap = dev.format(self.device.capture).map_err(device_error)?;
        // SAFETY: the buffer type selects the union member.
        unsafe {
            if self.device.multiplanar() {
                let p = cap.pix_mp();
                p.pixelformat = uapi::V4L2_PIX_FMT_H264;
                p.width = w;
                p.height = h;
                p.num_planes = 1;
                p.plane_fmt[0].sizeimage = 0;
            } else {
                let p = cap.pix();
                p.pixelformat = uapi::V4L2_PIX_FMT_H264;
                p.width = w;
                p.height = h;
                p.sizeimage = 0;
            }
        }
        dev.set_format(&mut cap).map_err(device_error)?;

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

        self.controls = controls::query_all(dev).map_err(device_error)?;
        self.apply_controls()?;

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

    fn apply_controls(&self) -> Result<()> {
        let cfg = &self.config;
        if let Bitrate::Bps(bps) = cfg.bitrate {
            let _ = self.set_control(
                uapi::V4L2_CID_MPEG_VIDEO_BITRATE_MODE,
                uapi::V4L2_MPEG_VIDEO_BITRATE_MODE_CBR,
            );
            let bps = i32::try_from(bps).unwrap_or(i32::MAX);
            self.require_control(uapi::V4L2_CID_MPEG_VIDEO_BITRATE, bps, "bitrate")?;
        }
        if let Some(gop) = cfg.gop {
            let gop = i32::try_from(gop).unwrap_or(i32::MAX);
            self.require_control(uapi::V4L2_CID_MPEG_VIDEO_GOP_SIZE, gop, "GOP size")?;
            self.set_control(uapi::V4L2_CID_MPEG_VIDEO_H264_I_PERIOD, gop)?;
        }
        if let Some(profile) = cfg.profile {
            self.require_control(
                uapi::V4L2_CID_MPEG_VIDEO_H264_PROFILE,
                profile_value(profile),
                &format!("H.264 profile {profile:?}"),
            )?;
        }
        if let Some(level) = cfg.level {
            self.require_control(
                uapi::V4L2_CID_MPEG_VIDEO_H264_LEVEL,
                level_value(level),
                &format!("H.264 level {level:?}"),
            )?;
        }
        // Headers in a separate buffer are folded into the next access unit,
        // but joining them is cheaper where the device allows it.
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

    /// Sets the OUTPUT format for `stride`, allocates both queues and starts
    /// streaming.
    fn start(&mut self, stride: usize) -> Result<()> {
        let dev = &self.device.dev;
        let mut out = dev.format(self.device.output).map_err(device_error)?;
        self.fill_output_format(&mut out, Some(stride));
        dev.set_format(&mut out).map_err(device_error)?;
        let (strides, min_lengths) = self.plane_formats(&mut out);
        if let Some(&s) = strides.iter().find(|&&s| s != stride) {
            return Err(CodecError::InvalidInput(format!(
                "{} needs a row stride of {s} bytes for {}x{} {}, the source has {stride}",
                self.path, self.config.width, self.config.height, self.config.input_format
            )));
        }

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
        let mut capture_maps = Vec::with_capacity(capture_count as usize);
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
        self.stream = Some(Stream {
            m2m,
            capture_maps,
            stride,
            min_lengths,
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
            m2m, capture_maps, ..
        }) = self.stream.take()
        {
            // Unmap before the buffers are freed: drivers without orphaned
            // buffer support refuse to free mapped buffers.
            drop(capture_maps);
            let _ = m2m.stream_off();
            drop(m2m);
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
        if src.memory() != TensorMemory::DmaBuf {
            return invalid(format!(
                "source is in {:?} memory; the encoder imports DMA-BUF tensors",
                src.memory()
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
        if (crop.x, crop.y) != (0, 0) && self.layout == Layout::Contiguous {
            return invalid(format!(
                "{} takes {} in one plane and cannot crop it",
                self.path, cfg.input_format
            ));
        }
        Ok((crop.x, crop.y))
    }

    /// Queues `src` and waits until the device has read it.
    fn submit(
        &mut self,
        src: &TensorDyn,
        origin: (usize, usize),
        opts: &FrameOptions,
    ) -> Result<()> {
        let stride = src
            .effective_row_stride()
            .ok_or_else(|| CodecError::InvalidInput("source has no row stride".into()))?;
        let format = self.config.input_format;
        let sh = src.height().unwrap_or(0);
        let luma_fd = src.dmabuf().map_err(CodecError::Tensor)?;
        let base = src.plane_offset().unwrap_or(0);

        // A separately allocated chroma plane brings its own DMA-BUF.
        let chroma_tensor = if self.layout == Layout::Packed {
            None
        } else if src.is_multiplane() {
            let c = src.chroma_dyn().ok_or_else(|| {
                CodecError::InvalidInput("multi-plane source has no chroma plane".into())
            })?;
            if c.effective_row_stride() != Some(stride) {
                return Err(CodecError::InvalidInput(
                    "chroma and luma row strides differ".into(),
                ));
            }
            Some(c)
        } else {
            None
        };
        let chroma_base = match &chroma_tensor {
            Some(c) => c.plane_offset().unwrap_or(0),
            None => {
                let planes = format
                    .plane_table(src.width().unwrap_or(0), sh, stride)
                    .ok_or_else(|| {
                        CodecError::InvalidInput(format!("no plane layout for {format}"))
                    })?;
                base + planes.get(1).map_or(0, |p| p.offset as usize)
            }
        };
        if self.layout == Layout::Contiguous
            && (chroma_tensor.is_some()
                || chroma_base != base + stride * self.config.height as usize)
        {
            return Err(CodecError::InvalidInput(format!(
                "{} takes {format} in one plane, so chroma must follow luma directly",
                self.path
            )));
        }

        if self.stream.is_none() {
            self.start(stride)?;
        }
        let multiplanar = self.device.multiplanar();
        let stream = self.stream.as_mut().expect("started above");
        if stride != stream.stride {
            return Err(CodecError::InvalidInput(format!(
                "source row stride {stride} differs from the stream's {}",
                stream.stride
            )));
        }

        let planes = source_planes(
            self.layout,
            format,
            stride,
            base,
            chroma_base,
            origin,
            self.config.height as usize,
        );
        let chroma_fd = chroma_tensor
            .as_ref()
            .map(|c| c.dmabuf().map_err(CodecError::Tensor))
            .transpose()?;
        let fds: Vec<BorrowedFd<'_>> = match self.layout {
            Layout::Split => vec![luma_fd, chroma_fd.unwrap_or(luma_fd)],
            _ => vec![luma_fd],
        };
        let mut v4l2_planes = Vec::with_capacity(fds.len());
        for (i, (plane, fd)) in planes.iter().zip(&fds).enumerate() {
            let length = dmabuf_size(*fd)?;
            // The Contiguous layout puts both source planes in V4L2 plane 0.
            let rows = if self.layout == Layout::Contiguous {
                planes.iter().map(|p| p.rows).sum()
            } else {
                plane.rows
            };
            let end = plane.offset + stride * rows;
            if end > length {
                return Err(CodecError::InvalidInput(format!(
                    "plane {i} ends at byte {end} of a {length}-byte DMA-BUF"
                )));
            }
            let min = stream.min_lengths.get(i).copied().unwrap_or(0);
            if length < min {
                return Err(CodecError::InvalidInput(format!(
                    "plane {i} DMA-BUF is {length} bytes, {} needs {min}",
                    self.path
                )));
            }
            if !multiplanar && plane.offset != 0 {
                return Err(CodecError::InvalidInput(format!(
                    "{} reads whole buffers; the source starts at byte {}",
                    self.path, plane.offset
                )));
            }
            v4l2_planes.push(Plane::DmaBuf {
                fd: *fd,
                length: length as u32,
                bytesused: end as u32,
                data_offset: plane.offset as u32,
            });
            if self.layout == Layout::Contiguous {
                break;
            }
        }

        if opts.keyframe {
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
        let index = stream.next_output;
        stream.next_output = (index + 1) % stream.output_count;
        stream
            .m2m
            .output()
            .enqueue(index, &v4l2_planes, Some(Duration::from_micros(seq)))
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
            // Some devices repeat the headers only on the IDR that opens a
            // GOP, not on a forced one.
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
        // Drivers differ in how a drained encoder resumes (ENC_CMD_START,
        // or restarting one queue), so the next frame sets up a new stream,
        // which also opens it with an IDR and its parameter sets.
        self.reset();
        Ok(self.pending.drain(..).collect())
    }

    fn device(&self) -> String {
        self.path.clone()
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

/// Size of a DMA-BUF in bytes.
fn dmabuf_size(fd: BorrowedFd<'_>) -> Result<usize> {
    let mut file = std::fs::File::from(fd.try_clone_to_owned()?);
    Ok(file.seek(SeekFrom::End(0))? as usize)
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
            2048,
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
            stride,
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
        let p = source_planes(Layout::Packed, PixelFormat::Yuyv, 2560, 0, 0, (10, 3), 480);
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
}
