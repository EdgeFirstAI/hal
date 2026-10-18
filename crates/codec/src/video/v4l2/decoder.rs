// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! H.264 decoding on a V4L2 stateful decoder (kernel
//! `Documentation/userspace-api/media/v4l/dev-decoder.rst`).
//!
//! Access units are copied into MMAP OUTPUT buffers. The CAPTURE queue is
//! set up once the device reports the stream's format with a
//! `V4L2_EVENT_SOURCE_CHANGE`, and again on every later change, after the
//! pictures of the old format have drained. Pictures are decoded into
//! DMA-BUFs the decoder (or the caller) allocated and imports, or, where the
//! device cannot import, into its own buffers exported as DMA-BUFs. Either
//! way each buffer is wrapped once as an NV12 tensor of the visible picture,
//! and a frame lends that tensor to the caller until it is dropped.
//!
//! V4L2 keeps buffer timestamps at microsecond resolution, so each access
//! unit is tagged with a sequence number and the caller's PTS is looked up
//! again when its picture comes back.

use std::borrow::Cow;
use std::collections::{BTreeMap, VecDeque};
use std::os::fd::{AsFd, AsRawFd, OwnedFd};
use std::sync::{Arc, Weak};
use std::time::{Duration, Instant};

// `Tensor::from_fd` is a trait method under the static tensor backend and
// inherent under the dynamic one.
#[cfg(feature = "static")]
use edgefirst_tensor::TensorTrait;
use edgefirst_tensor::{Colorimetry, DType, PixelFormat, Region, Tensor, TensorDyn, TensorMemory};
use edgefirst_v4l2::controls::{self, ControlInfo, ControlValue};
use edgefirst_v4l2::events;
use edgefirst_v4l2::m2m::{M2m, Ready};
use edgefirst_v4l2::queue::{BufType, Mapping, Memory, Plane};
use edgefirst_v4l2::{uapi, ErrorKind};

use super::{candidates, chroma, device_error, M2mDevice};
use crate::video::h264::{self, AccessUnit};
use crate::video::{
    DecodeEvent, DecodedFrame, DecoderBackend, DecoderConfig, OutputMemory, OutputPool, Returns,
    StreamInfo,
};
use crate::{CodecError, Result};

/// Environment variable naming the decoder node to use.
const ENV_DECODER: &str = "EDGEFIRST_CODEC_V4L2_DECODER";
/// Bitstream (OUTPUT) buffers to ask for. One access unit is in the device
/// at a time (see `submit`); a driver may grant more.
const BITSTREAM_BUFFERS: u32 = 2;
/// Smallest bitstream buffer to ask for; one access unit must fit.
const BITSTREAM_SIZE: u32 = 2 << 20;
/// How long the device may take to free a bitstream buffer.
const INPUT_TIMEOUT: Duration = Duration::from_secs(2);
/// How long to wait, once an access unit is queued, for a result.
const OUTPUT_GRACE: Duration = Duration::from_millis(20);
/// How long a drain may take.
const DRAIN_TIMEOUT: Duration = Duration::from_secs(2);
/// Events handled per wake-up, so a flood cannot stall decoding.
const MAX_EVENTS: usize = 16;
/// Timestamps kept for pictures the device has not returned. Older ones
/// belong to access units the device dropped.
const MAX_TIMESTAMPS: usize = 256;

/// The NV12 formats a decoder can write, in order of preference, and
/// whether each keeps its chroma in a second buffer.
const CAPTURE_FORMATS: [(u32, bool); 2] = [
    (uapi::V4L2_PIX_FMT_NV12, false),
    (uapi::V4L2_PIX_FMT_NV12M, true),
];

/// How the pictures of one stream format are laid out in the buffers.
#[derive(Debug, Clone, PartialEq)]
struct Layout {
    /// The chroma plane is in a buffer of its own (`NV12M`).
    two_buffers: bool,
    /// `bytesperline` of the luma and chroma planes.
    strides: [usize; 2],
    /// `sizeimage` of each buffer.
    sizes: Vec<usize>,
    /// Height of the format, which places the chroma plane of `NV12`.
    format_height: usize,
    crop: Region,
}

/// Byte offsets of the visible luma and chroma planes in their buffers.
fn plane_offsets(layout: &Layout) -> [usize; 2] {
    let Region { x, y, .. } = layout.crop;
    let [luma, chroma] = layout.strides;
    let chroma_base = if layout.two_buffers {
        0
    } else {
        luma * layout.format_height
    };
    [y * luma + x, chroma_base + (y / 2) * chroma + x]
}

/// An NV12 tensor of the visible picture over a buffer's memory. `fds` holds
/// the luma buffer and, for `NV12M`, the chroma buffer.
fn frame_tensor(
    mut fds: Vec<OwnedFd>,
    layout: &Layout,
    colorimetry: Option<Colorimetry>,
) -> Result<TensorDyn> {
    let (w, h) = (layout.crop.width, layout.crop.height);
    let [luma_offset, chroma_offset] = plane_offsets(layout);
    let chroma_fd = if layout.two_buffers {
        fds.pop().ok_or_else(|| {
            CodecError::InvalidConfig("NV12M buffer without a chroma plane".into())
        })?
    } else {
        fds[0].try_clone()?
    };
    let luma_fd = fds.swap_remove(0);
    let luma = Tensor::<u8>::from_fd(luma_fd, &[h, w], None)?;
    // A raw plane has no format to check a stride against, so the chroma
    // plane takes its geometry unchecked; the luma plane's is checked once
    // the planes are joined as NV12.
    let mut chroma = Tensor::<u8>::from_fd(chroma_fd, &[h.div_ceil(2), w], None)?;
    chroma.set_row_stride_unchecked(layout.strides[1]);
    chroma.set_plane_offset(chroma_offset);
    let mut t = Tensor::from_planes(luma, chroma, PixelFormat::Nv12)?;
    t.set_row_stride(layout.strides[0])?;
    t.set_plane_offset(luma_offset);
    t.set_colorimetry(colorimetry);
    Ok(TensorDyn::from(t))
}

/// Memory behind one CAPTURE buffer.
enum Backing {
    /// DMA-BUFs imported on every enqueue, one per V4L2 plane, with their
    /// sizes.
    Import(Vec<(OwnedFd, usize)>),
    /// The device's own MMAP buffer, exported.
    Export,
}

/// One CAPTURE buffer and the tensor that describes it.
struct Slot {
    tensor: Arc<TensorDyn>,
    backing: Backing,
}

/// The CAPTURE side for one stream format.
struct Capture {
    slots: Vec<Slot>,
    generation: u64,
    sizes: Vec<usize>,
}

/// The device's streaming state, created by the first access unit.
struct Stream {
    m2m: M2m,
    bitstream: Vec<Mapping>,
    free: VecDeque<u32>,
    capture: Option<Capture>,
    /// A source change arrived; the old pictures drain until the buffer
    /// flagged last, then the CAPTURE side is set up again.
    changing: bool,
    /// `V4L2_DEC_CMD_STOP` was issued and the drain has not finished.
    draining: bool,
}

pub(crate) struct V4l2Decoder {
    device: M2mDevice,
    path: String,
    pool: OutputPool,
    controls: Vec<ControlInfo>,
    can_stop: bool,
    stream: Option<Stream>,
    info: Option<StreamInfo>,
    generation: u64,
    returns: Arc<Returns>,
    pending: VecDeque<DecodeEvent>,
    /// Provided buffers, by pool index, still shown by a frame of an
    /// earlier pool. A buffer is queued again only once that frame drops.
    leases: Vec<Option<Weak<TensorDyn>>>,
    next_seq: u64,
    /// The caller's PTS by the sequence number queued with its access unit.
    timestamps: BTreeMap<u64, u64>,
    /// The stream's latest SPS and PPS, sent again after a reset.
    parameter_sets: Vec<u8>,
    needs_headers: bool,
}

impl V4l2Decoder {
    /// Opens the first device that decodes H.264 to NV12.
    pub fn open(config: DecoderConfig) -> Result<Self> {
        for path in candidates(ENV_DECODER) {
            let Some(device) = M2mDevice::open(&path) else {
                continue;
            };
            if !device.lists(device.output, uapi::V4L2_PIX_FMT_H264) {
                continue;
            }
            if !CAPTURE_FORMATS.iter().any(|&(fourcc, two)| {
                (!two || device.multiplanar()) && device.lists(device.capture, fourcc)
            }) {
                log::debug!("{} does not decode to NV12", path.display());
                continue;
            }
            let controls = controls::query_all(&device.dev).map_err(device_error)?;
            let mut cmd = uapi::v4l2_decoder_cmd {
                cmd: uapi::V4L2_DEC_CMD_STOP,
                ..Default::default()
            };
            // SAFETY: `cmd` is a valid v4l2_decoder_cmd for the duration of the call.
            let can_stop = unsafe {
                edgefirst_v4l2::ioctl::vidioc_try_decoder_cmd(
                    device.dev.as_fd().as_raw_fd(),
                    &mut cmd,
                )
            }
            .is_ok();
            log::info!("h264 decoder {}", path.display());
            return Ok(Self {
                device,
                path: path.display().to_string(),
                pool: config.output,
                controls,
                can_stop,
                stream: None,
                info: None,
                generation: 0,
                returns: Arc::default(),
                pending: VecDeque::new(),
                leases: Vec::new(),
                next_seq: 1,
                timestamps: BTreeMap::new(),
                parameter_sets: Vec::new(),
                needs_headers: false,
            });
        }
        Err(CodecError::NoDevice("no V4L2 H.264 decoder found".into()))
    }

    /// Sets the coded format, subscribes to events and starts the bitstream
    /// queue. The CAPTURE side waits for the first source change.
    fn start(&mut self) -> Result<()> {
        let dev = &self.device.dev;
        let mut fmt = dev.format(self.device.output).map_err(device_error)?;
        // SAFETY: the buffer type selects the union member.
        unsafe {
            if self.device.multiplanar() {
                let p = fmt.pix_mp();
                p.pixelformat = uapi::V4L2_PIX_FMT_H264;
                p.num_planes = 1;
                p.plane_fmt[0].sizeimage = p.plane_fmt[0].sizeimage.max(BITSTREAM_SIZE);
            } else {
                let p = fmt.pix();
                p.pixelformat = uapi::V4L2_PIX_FMT_H264;
                p.sizeimage = p.sizeimage.max(BITSTREAM_SIZE);
            }
        }
        dev.set_format(&mut fmt).map_err(device_error)?;
        events::subscribe(dev, uapi::V4L2_EVENT_SOURCE_CHANGE, 0, 0).map_err(device_error)?;
        if let Err(e) = events::subscribe(dev, uapi::V4L2_EVENT_EOS, 0, 0) {
            log::debug!("{}: no end-of-stream events: {e}", self.path);
        }

        let mut m2m = M2m::new(dev, self.device.multiplanar()).map_err(device_error)?;
        let count = m2m
            .output_mut()
            .request(Memory::Mmap, BITSTREAM_BUFFERS)
            .map_err(device_error)?;
        if count == 0 {
            return Err(CodecError::NoDevice(format!(
                "{} allocated no bitstream buffers",
                self.path
            )));
        }
        let mut bitstream = Vec::with_capacity(count as usize);
        for index in 0..count {
            let mut maps = m2m.output().map(index).map_err(device_error)?;
            if maps.is_empty() {
                return Err(CodecError::NoDevice(format!(
                    "{} bitstream buffer {index} has no plane",
                    self.path
                )));
            }
            bitstream.push(maps.swap_remove(0));
        }
        m2m.output().stream_on().map_err(device_error)?;
        self.stream = Some(Stream {
            m2m,
            bitstream,
            free: (0..count).collect(),
            capture: None,
            changing: false,
            draining: false,
        });
        Ok(())
    }

    /// Stops streaming and frees the buffers. Frames the caller holds keep
    /// their memory; the next access unit starts again, with the last
    /// parameter sets in front of it.
    fn reset(&mut self) {
        if let Some(mut stream) = self.stream.take() {
            drop(std::mem::take(&mut stream.bitstream));
            if let Some(capture) = stream.capture.take() {
                self.retire(capture);
                let _ = stream.m2m.capture_mut().free();
            }
            let _ = stream.m2m.output_mut().free();
            let _ = events::unsubscribe(&self.device.dev, uapi::V4L2_EVENT_ALL, 0);
            drop(stream);
            // A torn-down `vsi_v4l2` instance takes no input when streamed
            // again; a new open is a new decoder instance on every driver.
            match M2mDevice::open(std::path::Path::new(&self.path)) {
                Some(device) => self.device = device,
                None => log::warn!("{}: could not reopen the decoder", self.path),
            }
        }
        self.generation += 1;
        self.timestamps.clear();
        self.needs_headers = !self.parameter_sets.is_empty();
    }

    fn min_capture_buffers(&self) -> usize {
        self.controls
            .iter()
            .find(|c| c.id == uapi::V4L2_CID_MIN_BUFFERS_FOR_CAPTURE)
            .and_then(|info| controls::get(&self.device.dev, info).ok())
            .and_then(|v| match v {
                ControlValue::Integer(n) => usize::try_from(n).ok(),
                _ => None,
            })
            .unwrap_or(1)
    }

    /// Reads the CAPTURE format the device chose, switching to NV12 where it
    /// chose something else.
    fn capture_format(&self) -> Result<(uapi::v4l2_format, bool)> {
        let dev = &self.device.dev;
        let mut fmt = dev.format(self.device.capture).map_err(device_error)?;
        let current = self.fourcc(&mut fmt);
        if let Some(&(_, two)) = CAPTURE_FORMATS.iter().find(|&&(f, _)| f == current) {
            return Ok((fmt, two));
        }
        for &(fourcc, two) in &CAPTURE_FORMATS {
            if two && !self.device.multiplanar() {
                continue;
            }
            let mut want = fmt;
            // SAFETY: the buffer type selects the union member.
            unsafe {
                if self.device.multiplanar() {
                    let p = want.pix_mp();
                    p.pixelformat = fourcc;
                    p.num_planes = if two { 2 } else { 1 };
                } else {
                    want.pix().pixelformat = fourcc;
                }
            }
            if dev.set_format(&mut want).is_ok() && self.fourcc(&mut want) == fourcc {
                return Ok((want, two));
            }
        }
        Err(CodecError::UnsupportedFormat(PixelFormat::Nv12))
    }

    fn fourcc(&self, fmt: &mut uapi::v4l2_format) -> u32 {
        // SAFETY: the buffer type selects the union member.
        unsafe {
            if self.device.multiplanar() {
                fmt.pix_mp().pixelformat
            } else {
                fmt.pix().pixelformat
            }
        }
    }

    /// The layout and colorimetry of a CAPTURE format.
    fn layout(
        &self,
        fmt: &mut uapi::v4l2_format,
        two_buffers: bool,
    ) -> (Layout, Option<Colorimetry>) {
        // SAFETY: the buffer type selects the union member.
        let (w, h, strides, sizes, c) = unsafe {
            if self.device.multiplanar() {
                let p = fmt.pix_mp();
                let n = usize::from(p.num_planes).clamp(1, p.plane_fmt.len());
                let planes = &p.plane_fmt[..n];
                let luma = planes[0].bytesperline as usize;
                let chroma = planes.get(1).map_or(luma, |pl| pl.bytesperline as usize);
                (
                    p.width,
                    p.height,
                    [luma, if chroma == 0 { luma } else { chroma }],
                    planes.iter().map(|pl| pl.sizeimage as usize).collect(),
                    (
                        p.colorspace,
                        u32::from(p.xfer_func),
                        u32::from(p.ycbcr_enc),
                        u32::from(p.quantization),
                    ),
                )
            } else {
                let p = fmt.pix();
                let s = p.bytesperline as usize;
                (
                    p.width,
                    p.height,
                    [s, s],
                    vec![p.sizeimage as usize],
                    (p.colorspace, p.xfer_func, p.ycbcr_enc, p.quantization),
                )
            }
        };
        let crop = self
            .visible_rect()
            .filter(|r| {
                r.width > 0
                    && r.height > 0
                    && r.x + r.width <= w as usize
                    && r.y + r.height <= h as usize
            })
            .unwrap_or(Region::new(0, 0, w as usize, h as usize));
        let colorimetry = (c.0 != uapi::V4L2_COLORSPACE_DEFAULT)
            .then(|| Colorimetry::from_v4l2(c.0, c.1, c.2, c.3, false));
        (
            Layout {
                two_buffers,
                strides,
                sizes,
                format_height: h as usize,
                crop,
            },
            colorimetry,
        )
    }

    /// The visible rectangle (`V4L2_SEL_TGT_COMPOSE`), which a decoder
    /// reports on the single-planar buffer type.
    fn visible_rect(&self) -> Option<Region> {
        let single = match self.device.capture {
            BufType::VideoCaptureMplane => BufType::VideoCapture,
            other => other,
        };
        [single, self.device.capture]
            .into_iter()
            .find_map(|t| {
                self.device
                    .dev
                    .selection(t, uapi::V4L2_SEL_TGT_COMPOSE)
                    .ok()
            })
            .map(|r| {
                Region::new(
                    r.left.max(0) as usize,
                    r.top.max(0) as usize,
                    r.width as usize,
                    r.height as usize,
                )
            })
    }

    /// Sets up the CAPTURE side for the format the device reports, and
    /// queues a [`DecodeEvent::Reconfigured`].
    fn setup_capture(&mut self) -> Result<()> {
        let (mut fmt, two_buffers) = self.capture_format()?;
        let (layout, colorimetry) = self.layout(&mut fmt, two_buffers);
        let min = self.min_capture_buffers();
        let want = match &self.pool {
            // The device minimum covers its reference pictures; Wave6 also
            // needs a buffer to decode into before it releases one, and
            // stalls at the end of a stream without it.
            OutputPool::Allocate { count } => min + count + 1,
            OutputPool::Provided(pool) => {
                if pool.len() < min + 1 {
                    return Err(CodecError::InvalidConfig(format!(
                        "{} provided buffers; {} needs at least {} for this stream",
                        pool.len(),
                        self.path,
                        min + 1
                    )));
                }
                pool.len()
            }
        };
        self.generation += 1;
        let generation = self.generation;
        let allocate = matches!(self.pool, OutputPool::Allocate { .. });
        let slots = if allocate {
            self.import_allocated(&layout, colorimetry, want)
        } else {
            self.import_provided(&layout, colorimetry, want)
        };
        let mut capture = match slots {
            Ok(slots) => Capture {
                slots,
                generation,
                sizes: layout.sizes.clone(),
            },
            Err(e) if allocate => {
                log::info!("{}: exporting device buffers ({e})", self.path);
                Capture {
                    slots: self.export(&layout, colorimetry, want)?,
                    generation,
                    sizes: layout.sizes.clone(),
                }
            }
            Err(e) => return Err(e),
        };
        // An allocated pool is new memory and never leased.
        let free: Vec<u32> = (0..capture.slots.len() as u32)
            .filter(|&i| !self.leased(i))
            .collect();
        let stream = self
            .stream
            .as_mut()
            .expect("setup_capture runs while streaming");
        let first = free
            .first()
            .map(|&i| enqueue_capture(&stream.m2m, &capture, i));
        if let Some(Err(e)) = first {
            // A device can list DMA-BUF import and still refuse the buffers.
            if !(allocate && matches!(capture.slots[0].backing, Backing::Import(_))) {
                return Err(e);
            }
            log::info!(
                "{}: exporting device buffers (import refused: {e})",
                self.path
            );
            stream.m2m.capture_mut().free().map_err(device_error)?;
            capture.slots = self.export(&layout, colorimetry, want)?;
            let stream = self.stream.as_mut().expect("streaming");
            enqueue_capture(&stream.m2m, &capture, free[0])?;
        }
        let stream = self.stream.as_mut().expect("streaming");
        for &index in free.iter().skip(1) {
            enqueue_capture(&stream.m2m, &capture, index)?;
        }
        stream.m2m.capture().stream_on().map_err(device_error)?;
        let info = StreamInfo {
            width: layout.crop.width,
            height: layout.crop.height,
            crop: layout.crop,
            coded_width: fmt_width(&mut fmt, self.device.multiplanar()),
            coded_height: layout.format_height,
            format: PixelFormat::Nv12,
            colorimetry,
            buffers: capture.slots.len(),
            memory: match capture.slots.first().map(|s| &s.backing) {
                Some(Backing::Export) => OutputMemory::Exported,
                _ => OutputMemory::Imported,
            },
        };
        log::info!(
            "{}: {}x{} (coded {}x{}, {} {:?} buffers)",
            self.path,
            info.width,
            info.height,
            info.coded_width,
            info.coded_height,
            info.buffers,
            info.memory
        );
        stream.capture = Some(capture);
        self.info = Some(info.clone());
        self.pending.push_back(DecodeEvent::Reconfigured(info));
        Ok(())
    }

    /// Allocates DMA-BUFs for the pool and has the device import them.
    fn import_allocated(
        &mut self,
        layout: &Layout,
        colorimetry: Option<Colorimetry>,
        want: usize,
    ) -> Result<Vec<Slot>> {
        let stream = self.stream.as_mut().expect("streaming");
        let count = stream
            .m2m
            .capture_mut()
            .request(Memory::DmaBuf, want as u32)
            .map_err(device_error)? as usize;
        let mut slots = Vec::with_capacity(count);
        for _ in 0..count {
            let planes = layout
                .sizes
                .iter()
                .map(|&size| TensorDyn::new(&[size], DType::U8, Some(TensorMemory::DmaBuf), None))
                .collect::<std::result::Result<Vec<_>, _>>()?;
            if planes.iter().any(|p| p.memory() != TensorMemory::DmaBuf) {
                return Err(CodecError::InvalidConfig("no DMA-BUF allocator".into()));
            }
            slots.push(import_slot(&planes, layout, colorimetry)?);
        }
        Ok(slots)
    }

    /// Has the device import the caller's tensors.
    fn import_provided(
        &mut self,
        layout: &Layout,
        colorimetry: Option<Colorimetry>,
        want: usize,
    ) -> Result<Vec<Slot>> {
        let OutputPool::Provided(pool) = &self.pool else {
            unreachable!("called for a provided pool");
        };
        let invalid = |what: String| Err(CodecError::InvalidConfig(what));
        let mut slots = Vec::with_capacity(want);
        for (i, t) in pool.iter().enumerate() {
            if t.plane_offset().unwrap_or(0) != 0 {
                return invalid(format!("provided tensor {i} starts inside its DMA-BUF"));
            }
            let chroma = if layout.two_buffers {
                Some(chroma(t).ok_or_else(|| {
                    CodecError::InvalidConfig(format!(
                        "{} writes chroma to a second buffer; provided tensor {i} has one plane",
                        self.path
                    ))
                })?)
            } else {
                None
            };
            let planes: Vec<&TensorDyn> = std::iter::once(t).chain(chroma.as_deref()).collect();
            for (p, (plane, &size)) in planes.iter().zip(&layout.sizes).enumerate() {
                if plane.capacity_bytes() < size {
                    return invalid(format!(
                        "provided tensor {i} plane {p} is {} bytes; {} needs {size} for this stream",
                        plane.capacity_bytes(),
                        self.path
                    ));
                }
            }
            slots.push(import_slot(&planes, layout, colorimetry)?);
        }
        let stream = self.stream.as_mut().expect("streaming");
        let count = stream
            .m2m
            .capture_mut()
            .request(Memory::DmaBuf, slots.len() as u32)
            .map_err(|e| {
                CodecError::InvalidConfig(format!("{} cannot import DMA-BUFs: {e}", self.path))
            })? as usize;
        if count < slots.len() {
            slots.truncate(count);
        }
        Ok(slots)
    }

    /// Has the device allocate the pool and exports it.
    fn export(
        &mut self,
        layout: &Layout,
        colorimetry: Option<Colorimetry>,
        want: usize,
    ) -> Result<Vec<Slot>> {
        let stream = self.stream.as_mut().expect("streaming");
        let queue = stream.m2m.capture_mut();
        let count = queue
            .request(Memory::Mmap, want as u32)
            .map_err(device_error)?;
        let mut slots = Vec::with_capacity(count as usize);
        for index in 0..count {
            let fds = (0..queue.num_planes() as u32)
                .map(|plane| queue.export(index, plane).map_err(device_error))
                .collect::<Result<Vec<_>>>()?;
            slots.push(Slot {
                tensor: Arc::new(frame_tensor(fds, layout, colorimetry)?),
                backing: Backing::Export,
            });
        }
        Ok(slots)
    }

    /// Gives the buffers of dropped frames back to the device, with the
    /// provided buffers whose frames of an earlier pool have dropped.
    fn give_back(&mut self) -> Result<()> {
        let Some(stream) = self.stream.as_ref() else {
            return Ok(());
        };
        let Some(capture) = stream.capture.as_ref() else {
            return Ok(());
        };
        let released = self.leases.iter_mut().enumerate().filter_map(|(i, lease)| {
            let free = lease.as_ref().is_some_and(|w| w.strong_count() == 0);
            free.then(|| {
                *lease = None;
                i as u32
            })
        });
        let mut returned = self.returns.take(capture.generation);
        returned.extend(released);
        for index in returned {
            if (index as usize) < capture.slots.len() && !stream.m2m.capture().is_queued(index) {
                enqueue_capture(&stream.m2m, capture, index)?;
            }
        }
        Ok(())
    }

    /// Whether provided buffer `index` is still shown by a frame of an
    /// earlier pool.
    fn leased(&self, index: u32) -> bool {
        self.leases
            .get(index as usize)
            .and_then(Option::as_ref)
            .is_some_and(|w| w.strong_count() > 0)
    }

    /// Drops a CAPTURE pool, recording the provided buffers that frames
    /// still show, which a later pool over the same memory must not queue.
    fn retire(&mut self, capture: Capture) {
        let OutputPool::Provided(pool) = &self.pool else {
            return;
        };
        self.leases.resize(pool.len(), None);
        for (lease, slot) in self.leases.iter_mut().zip(&capture.slots) {
            if Arc::strong_count(&slot.tensor) > 1 {
                *lease = Some(Arc::downgrade(&slot.tensor));
            }
        }
    }

    /// Waits up to `timeout` for the device, handling what it reports,
    /// until `done` holds. Returns whether it did.
    fn service(&mut self, timeout: Duration, done: impl Fn(&Self) -> bool) -> Result<bool> {
        let deadline = Instant::now() + timeout;
        loop {
            self.give_back()?;
            if done(self) {
                return Ok(true);
            }
            let left = deadline.saturating_duration_since(Instant::now());
            let stream = self.stream.as_ref().expect("service runs while streaming");
            let ready = match stream.m2m.wait(Some(left)) {
                Ok(ready) => ready,
                // Nothing is queued that could complete.
                Err(e) if e.kind() == ErrorKind::InvalidState => Ready::default(),
                Err(e) => return Err(device_error(e)),
            };
            if ready.is_empty() {
                return Ok(done(self));
            }
            if ready.output {
                self.reclaim_bitstream()?;
            }
            if ready.capture {
                self.collect()?;
            }
            if ready.event {
                self.handle_events()?;
            }
            // A queue can stay readable without yielding anything (one that
            // has returned its last buffer reports EPIPE until restarted).
            if left.is_zero() {
                return Ok(done(self));
            }
        }
    }

    /// Restarts both queues after a drain, as for a seek. Drivers differ in
    /// what resumes a drained decoder: `vsi_v4l2` stays at end of stream
    /// after `V4L2_DEC_CMD_START`, and Wave6 does not drain again after it,
    /// while restarting the queues resumes both. The buffers no frame holds
    /// are queued again.
    fn restart(&mut self) -> Result<()> {
        let stream = self.stream.as_mut().expect("streaming");
        stream.m2m.output().stream_off().map_err(device_error)?;
        stream.free = (0..stream.bitstream.len() as u32).collect();
        stream.m2m.output().stream_on().map_err(device_error)?;
        if stream.capture.is_none() {
            return Ok(());
        }
        stream.m2m.capture().stream_off().map_err(device_error)?;
        stream.m2m.capture().stream_on().map_err(device_error)?;
        let stream = self.stream.as_ref().expect("streaming");
        let capture = stream.capture.as_ref().expect("checked above");
        for (index, slot) in capture.slots.iter().enumerate() {
            let index = index as u32;
            if Arc::strong_count(&slot.tensor) == 1 && !self.leased(index) {
                enqueue_capture(&stream.m2m, capture, index)?;
            }
        }
        Ok(())
    }

    /// Buffer counts for diagnostics.
    fn queue_state(&self) -> String {
        let Some(stream) = self.stream.as_ref() else {
            return "not streaming".into();
        };
        let capture = stream.capture.as_ref().map_or(0, |c| c.slots.len());
        format!(
            "bitstream {} free of {}, capture {} queued of {capture}, {} results pending",
            stream.free.len(),
            stream.bitstream.len(),
            stream.m2m.capture().queued_count(),
            self.pending.len()
        )
    }

    /// Takes back the bitstream buffers the device has read.
    fn reclaim_bitstream(&mut self) -> Result<()> {
        let stream = self.stream.as_mut().expect("streaming");
        while let Some(done) = stream.m2m.output().dequeue().map_err(device_error)? {
            stream.free.push_back(done.index);
        }
        Ok(())
    }

    fn handle_events(&mut self) -> Result<()> {
        let evs = events::drain(&self.device.dev, MAX_EVENTS).map_err(device_error)?;
        for ev in evs {
            match ev.source_change() {
                Some(changes) if changes & uapi::V4L2_EVENT_SRC_CH_RESOLUTION != 0 => {
                    let stream = self.stream.as_mut().expect("streaming");
                    if stream.capture.is_some() {
                        stream.changing = true;
                    } else {
                        self.setup_capture()?;
                    }
                }
                Some(changes) => log::debug!("{}: source change {changes:#x}", self.path),
                None => log::debug!("{}: event {}", self.path, ev.kind()),
            }
        }
        Ok(())
    }

    /// Dequeues every decoded picture. Returns whether a buffer flagged last
    /// (or the end of the queue) was reached; a pending source change is
    /// then applied.
    fn collect(&mut self) -> Result<bool> {
        loop {
            let stream = self.stream.as_ref().expect("streaming");
            let Some(capture) = stream.capture.as_ref() else {
                return Ok(false);
            };
            let done = match stream.m2m.capture().dequeue() {
                Ok(Some(done)) => done,
                Ok(None) => return Ok(false),
                Err(e) if e.kind() == ErrorKind::EndOfStream => {
                    log::debug!("{}: CAPTURE at end of stream", self.path);
                    return self.after_last();
                }
                Err(e) => return Err(device_error(e)),
            };
            let last = done.flags.is_last();
            let used = done.planes().first().map_or(0, |p| p.bytesused);
            if used == 0 {
                // An empty buffer, usually the one carrying the last flag.
                enqueue_capture(&stream.m2m, capture, done.index)?;
            } else {
                let tensor = Arc::clone(&capture.slots[done.index as usize].tensor);
                let slot = (capture.generation, done.index);
                let pts = self.take_pts(done.timestamp.as_micros() as u64);
                let frame = DecodedFrame::new(
                    tensor,
                    pts,
                    done.flags.is_error(),
                    slot,
                    Arc::clone(&self.returns),
                );
                self.pending.push_back(DecodeEvent::Frame(frame));
            }
            if last {
                log::debug!("{}: last buffer {} ({used} bytes)", self.path, done.index);
                return self.after_last();
            }
        }
    }

    /// The CAPTURE queue ended a sequence: applies a pending source change,
    /// or reports the end of a drain.
    fn after_last(&mut self) -> Result<bool> {
        let stream = self.stream.as_mut().expect("streaming");
        if stream.changing {
            stream.changing = false;
            let old = stream.capture.take();
            stream.m2m.capture_mut().free().map_err(device_error)?;
            if let Some(old) = old {
                self.retire(old);
            }
            self.setup_capture()?;
        }
        Ok(true)
    }

    /// The caller's PTS for sequence number `seq`.
    fn take_pts(&mut self, seq: u64) -> u64 {
        if let Some(pts) = self.timestamps.remove(&seq) {
            return pts;
        }
        log::warn!("{}: picture with unknown timestamp {seq}", self.path);
        0
    }

    /// Queues one access unit, waiting for a free bitstream buffer.
    fn submit(&mut self, access_unit: &[u8], pts: u64) -> Result<()> {
        if self.stream.is_none() {
            self.start()?;
        }
        let data: Cow<'_, [u8]> =
            if self.needs_headers && !AccessUnit::parse(access_unit).parameter_sets {
                let mut d = self.parameter_sets.clone();
                d.extend_from_slice(access_unit);
                Cow::Owned(d)
            } else {
                Cow::Borrowed(access_unit)
            };

        // One access unit in the device at a time: each needs one buffer to
        // decode into, so a call never waits on more buffers than the pool's
        // spare. Deeper queues decode no faster on these devices, and frames
        // finished while a call waits would hold the buffers it needs.
        let has_free = |s: &Self| {
            s.stream
                .as_ref()
                .is_some_and(|st| st.free.len() == st.bitstream.len())
        };
        self.give_back()?;
        if !has_free(self) {
            // With no output buffer queued the device cannot finish a
            // picture, so no bitstream buffer will come back.
            if let Some((held, pool)) = self.held().filter(|&(held, pool)| held == pool) {
                return Err(CodecError::PoolExhausted { held, pool });
            }
        }
        if !self.service(INPUT_TIMEOUT, has_free)? {
            log::debug!(
                "{}: no free bitstream buffer ({})",
                self.path,
                self.queue_state()
            );
            return Err(self.stalled("take an access unit", INPUT_TIMEOUT));
        }
        let stream = self.stream.as_mut().expect("started above");
        let index = stream.free.pop_front().expect("checked above");
        let map = &stream.bitstream[index as usize];
        if data.len() > map.len() {
            stream.free.push_front(index);
            return Err(CodecError::InvalidInput(format!(
                "access unit of {} bytes; {} takes at most {}",
                data.len(),
                self.path,
                map.len()
            )));
        }
        // SAFETY: the buffer is not queued, so the device is not reading it,
        // and the copy stays inside the mapping.
        unsafe { std::ptr::copy_nonoverlapping(data.as_ptr(), map.as_ptr(), data.len()) };
        let seq = self.next_seq;
        self.next_seq += 1;
        stream
            .m2m
            .output()
            .enqueue(
                index,
                &[Plane::Mmap {
                    bytesused: data.len() as u32,
                }],
                Some(Duration::from_micros(seq)),
            )
            .map_err(device_error)?;
        self.timestamps.insert(seq, pts);
        // The oldest entries belong to pictures the device skipped.
        while self.timestamps.len() > MAX_TIMESTAMPS {
            self.timestamps.pop_first();
        }
        self.needs_headers = false;
        Ok(())
    }

    /// Output buffers the caller holds, and the pool size, once the CAPTURE
    /// side is set up: those neither queued to the device nor in a frame
    /// not yet returned.
    fn held(&self) -> Option<(usize, usize)> {
        let stream = self.stream.as_ref()?;
        let pool = stream.capture.as_ref()?.slots.len();
        let unreturned = self
            .pending
            .iter()
            .filter(|e| matches!(e, DecodeEvent::Frame(_)))
            .count();
        let out = pool - stream.m2m.capture().queued_count().min(pool);
        Some((out.saturating_sub(unreturned), pool))
    }

    /// The error for a device that stopped making progress: the caller's
    /// frames when it holds some, the device otherwise.
    fn stalled(&self, what: &str, waited: Duration) -> CodecError {
        match self.held() {
            Some((held, pool)) if held > 0 => CodecError::PoolExhausted { held, pool },
            _ => CodecError::Timeout(format!(
                "{} did not {what} within {waited:?} ({})",
                self.path,
                self.queue_state()
            )),
        }
    }
}

/// A slot over `planes` (luma buffer, then the chroma buffer of `NV12M`),
/// queued by import.
fn import_slot<T: std::borrow::Borrow<TensorDyn>>(
    planes: &[T],
    layout: &Layout,
    colorimetry: Option<Colorimetry>,
) -> Result<Slot> {
    let mut frame_fds = Vec::with_capacity(planes.len());
    let mut buffers = Vec::with_capacity(planes.len());
    for p in planes {
        let p = p.borrow();
        frame_fds.push(p.clone_fd()?);
        buffers.push((p.clone_fd()?, p.capacity_bytes()));
    }
    Ok(Slot {
        tensor: Arc::new(frame_tensor(frame_fds, layout, colorimetry)?),
        backing: Backing::Import(buffers),
    })
}

/// Width of a format of either API.
fn fmt_width(fmt: &mut uapi::v4l2_format, multiplanar: bool) -> usize {
    // SAFETY: the buffer type selects the union member.
    unsafe {
        if multiplanar {
            fmt.pix_mp().width as usize
        } else {
            fmt.pix().width as usize
        }
    }
}

/// Queues CAPTURE buffer `index` of `capture`.
fn enqueue_capture(m2m: &M2m, capture: &Capture, index: u32) -> Result<()> {
    let slot = &capture.slots[index as usize];
    match &slot.backing {
        Backing::Export => {
            let planes = vec![Plane::mmap(); m2m.capture().num_planes().max(1)];
            m2m.capture().enqueue(index, &planes, None)
        }
        Backing::Import(buffers) => {
            let planes: Vec<Plane<'_>> = buffers
                .iter()
                .zip(&capture.sizes)
                .map(|((fd, length), &size)| Plane::DmaBuf {
                    fd: fd.as_fd(),
                    length: (*length).max(size) as u32,
                    bytesused: 0,
                    data_offset: 0,
                })
                .collect();
            m2m.capture().enqueue(index, &planes, None)
        }
    }
    .map_err(device_error)
}

impl DecoderBackend for V4l2Decoder {
    fn decode(&mut self, access_unit: &[u8], pts: u64) -> Result<Vec<DecodeEvent>> {
        if access_unit.is_empty() {
            return Err(CodecError::InvalidInput("empty access unit".into()));
        }
        if AccessUnit::parse(access_unit).parameter_sets {
            self.parameter_sets = h264::parameter_sets(access_unit);
        }
        let result = self.submit(access_unit, pts).and_then(|()| {
            // Wait briefly for a result when there is none yet, then take
            // whatever else is ready.
            if self.pending.is_empty() {
                self.service(OUTPUT_GRACE, |s| !s.pending.is_empty())?;
            }
            self.service(Duration::ZERO, |_| false).map(|_| ())
        });
        if let Err(e) = result {
            if matches!(e, CodecError::Timeout(_) | CodecError::Device { .. }) {
                self.reset();
            }
            return Err(e);
        }
        Ok(self.pending.drain(..).collect())
    }

    fn flush(&mut self) -> Result<Vec<DecodeEvent>> {
        if self.stream.is_none() {
            return Ok(self.pending.drain(..).collect());
        }
        let capturing = |s: &Self| s.stream.as_ref().is_some_and(|st| st.capture.is_some());
        if self.can_stop && !capturing(self) {
            // `decode` can return before the device has parsed the first
            // headers; the drain needs the CAPTURE side they set up.
            self.service(DRAIN_TIMEOUT, capturing)?;
        }
        if self.can_stop && capturing(self) {
            if let Err(e) = self.drain() {
                // Frames the caller holds block the drain; it resumes on the
                // next flush once they are dropped.
                if !matches!(e, CodecError::PoolExhausted { .. }) {
                    self.reset();
                }
                return Err(e);
            }
            if let Err(e) = self.restart() {
                log::debug!("{}: restarting after the drain: {e}", self.path);
                self.reset();
            }
        } else {
            // Without DECODER_CMD, collect what the device finishes on its
            // own, then start a new stream for whatever comes next.
            self.service(DRAIN_TIMEOUT / 4, |_| false)?;
            self.reset();
        }
        Ok(self.pending.drain(..).collect())
    }

    fn stream_info(&self) -> Option<&StreamInfo> {
        self.info.as_ref()
    }

    fn device(&self) -> String {
        self.path.clone()
    }
}

impl V4l2Decoder {
    /// Issues `V4L2_DEC_CMD_STOP`, once per drain, and waits for the buffer
    /// flagged last.
    fn drain(&mut self) -> Result<()> {
        let stream = self.stream.as_mut().expect("streaming");
        if !stream.draining {
            decoder_cmd(&self.device.dev, uapi::V4L2_DEC_CMD_STOP)?;
            stream.draining = true;
        }
        let deadline = Instant::now() + DRAIN_TIMEOUT;
        loop {
            let left = deadline.saturating_duration_since(Instant::now());
            if left.is_zero() {
                return Err(self.stalled("finish draining", DRAIN_TIMEOUT));
            }
            if self.drain_step(left)? {
                break;
            }
        }
        if let Some(stream) = self.stream.as_mut() {
            stream.draining = false;
        }
        Ok(())
    }

    /// Waits for drain progress; returns whether the buffer flagged last
    /// arrived with no source change pending.
    fn drain_step(&mut self, timeout: Duration) -> Result<bool> {
        self.give_back()?;
        let stream = self.stream.as_ref().expect("streaming");
        let ready = match stream.m2m.wait(Some(timeout)) {
            Ok(ready) => ready,
            // Nothing is queued on either side: the caller holds every
            // output buffer, and the device has nowhere to drain into.
            Err(e) if e.kind() == ErrorKind::InvalidState => {
                return Err(self.stalled("finish draining", Duration::ZERO));
            }
            Err(e) => return Err(device_error(e)),
        };
        if ready.output {
            self.reclaim_bitstream()?;
        }
        if ready.event {
            self.handle_events()?;
        }
        if ready.capture {
            let changing = self.stream.as_ref().is_some_and(|s| s.changing);
            if self.collect()? && !changing {
                return Ok(true);
            }
        }
        Ok(false)
    }
}

impl Drop for V4l2Decoder {
    fn drop(&mut self) {
        self.pending.clear();
        self.reset();
    }
}

/// Issues `VIDIOC_DECODER_CMD` with `cmd`.
fn decoder_cmd(dev: &edgefirst_v4l2::device::Device, cmd: u32) -> Result<()> {
    let mut c = uapi::v4l2_decoder_cmd {
        cmd,
        ..Default::default()
    };
    // SAFETY: `c` is a valid v4l2_decoder_cmd for the duration of the call.
    unsafe { edgefirst_v4l2::ioctl::vidioc_decoder_cmd(dev.as_fd().as_raw_fd(), &mut c) }
        .map(|_| ())
        .map_err(|errno| CodecError::Device {
            op: "VIDIOC_DECODER_CMD".into(),
            source: std::io::Error::from_raw_os_error(errno as i32),
        })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn layout(two_buffers: bool, crop: Region) -> Layout {
        Layout {
            two_buffers,
            strides: [2048, 2048],
            sizes: vec![2048 * 1088 * 3 / 2],
            format_height: 1088,
            crop,
        }
    }

    #[test]
    fn chroma_of_a_padded_picture_follows_the_coded_height() {
        let l = layout(false, Region::new(0, 0, 1920, 1080));
        assert_eq!(plane_offsets(&l), [0, 2048 * 1088]);
    }

    #[test]
    fn crop_moves_both_planes() {
        let l = layout(false, Region::new(16, 8, 1280, 720));
        assert_eq!(
            plane_offsets(&l),
            [8 * 2048 + 16, 2048 * 1088 + 4 * 2048 + 16]
        );
    }

    #[test]
    fn separate_chroma_buffer_starts_at_zero() {
        let l = layout(true, Region::new(0, 0, 1920, 1080));
        assert_eq!(plane_offsets(&l), [0, 0]);
        let l = layout(true, Region::new(0, 2, 1920, 1078));
        assert_eq!(plane_offsets(&l), [2 * 2048, 2048]);
    }
}
