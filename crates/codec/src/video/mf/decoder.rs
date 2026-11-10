// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! H.264 decoding with a Media Foundation decoder transform.
//!
//! The decoder writes each picture to a system-memory sample, which is
//! copied into a free tensor of the pool and returned as a
//! [`DecodedFrame`]. Pictures the decoder finishes while every tensor is
//! held wait, in Media Foundation's memory, for a frame to drop; the next
//! call fails with [`CodecError::PoolExhausted`] without taking its access
//! unit until one has.
//!
//! The pool is allocated at the stream's visible size: D3D11 tensors where
//! the process has a D3D11 device, host memory otherwise. A provided pool
//! must hold NV12 images of exactly that size, since a D3D11 texture cannot
//! be described as a smaller image.
//!
//! Sample times are the access unit's sequence number; the caller's PTS is
//! looked up again by the time of each decoded picture, which the decoder
//! carries through its reordering.

use std::collections::{BTreeMap, VecDeque};
use std::sync::Arc;
use std::time::Duration;

use edgefirst_tensor::{
    ColorEncoding, ColorRange, ColorSpace, ColorTransfer, Colorimetry, CpuAccess, DType,
    PixelFormat, Region, TensorDyn, TensorMemory,
};
use windows::core::{Interface, GUID};
use windows::Win32::Media::MediaFoundation::{
    CODECAPI_AVLowLatencyMode, ICodecAPI, IMF2DBuffer2, IMFAttributes, IMFMediaType, IMFSample,
    MF2DBuffer_LockFlags_Read, MFMediaType_Video, MFNominalRange_0_255, MFNominalRange_16_235,
    MFVideoArea, MFVideoFormat_H264, MFVideoFormat_NV12, MFVideoPrimaries_BT2020,
    MFVideoPrimaries_BT709, MFVideoPrimaries_SMPTE170M, MFVideoTransFunc_10, MFVideoTransFunc_2084,
    MFVideoTransFunc_709, MFVideoTransFunc_HLG, MFVideoTransFunc_sRGB,
    MFVideoTransferMatrix_BT2020_10, MFVideoTransferMatrix_BT601, MFVideoTransferMatrix_BT709,
    MFT_CATEGORY_VIDEO_DECODER, MF_MT_DEFAULT_STRIDE, MF_MT_FRAME_SIZE, MF_MT_MAJOR_TYPE,
    MF_MT_MINIMUM_DISPLAY_APERTURE, MF_MT_SUBTYPE, MF_MT_TRANSFER_FUNCTION,
    MF_MT_VIDEO_NOMINAL_RANGE, MF_MT_VIDEO_PRIMARIES, MF_MT_YUV_MATRIX,
};

use windows::Win32::System::Variant::VARIANT;

use super::copy::{self, Plane};
use super::{com, error, pair, Output, Platform, Transform};
use crate::video::{
    DecodeEvent, DecodedFrame, DecoderBackend, DecoderConfig, OutputMemory, OutputPool, Returns,
    StreamInfo,
};
use crate::{CodecError, Result};

/// Environment variable keeping only the decoders whose friendly name
/// contains its value (or `hardware`, or `software`).
const ENV_DECODER: &str = "EDGEFIRST_CODEC_MF_DECODER";
/// How long the decoder may take to accept one access unit.
const INPUT_TIMEOUT: Duration = Duration::from_secs(2);
/// How long an asynchronous decoder may take, once an access unit is
/// accepted, to return a picture before `decode` returns without one.
const OUTPUT_GRACE: Duration = Duration::from_millis(100);
/// How long a drain may take.
const DRAIN_TIMEOUT: Duration = Duration::from_secs(2);
/// Sample times kept for access units whose picture has not come back.
const TIMES_KEPT: usize = 64;

/// Where the pixels of a decoded sample are.
#[derive(Debug, Clone, PartialEq)]
struct Layout {
    /// Size of the coded picture.
    coded: (usize, usize),
    /// The picture to show, within the coded one.
    visible: Region,
    /// Bytes from one row to the next in a contiguous sample.
    pitch: usize,
    colorimetry: Option<Colorimetry>,
}

impl Layout {
    /// The layout an NV12 output type describes.
    fn of(t: &IMFMediaType) -> Result<Self> {
        let attrs: IMFAttributes = t.cast().map_err(|e| error(e, "IMFAttributes"))?;
        let (w, h) = pair(&attrs, &MF_MT_FRAME_SIZE)
            .map(|(w, h)| (w as usize, h as usize))
            .ok_or_else(|| CodecError::InvalidData("the decoder reports no frame size".into()))?;
        let mut area = MFVideoArea::default();
        let size = std::mem::size_of::<MFVideoArea>();
        // SAFETY: the blob is an `MFVideoArea`, read into one of its size.
        let aperture = unsafe {
            attrs.GetBlob(
                &MF_MT_MINIMUM_DISPLAY_APERTURE,
                std::slice::from_raw_parts_mut((&mut area as *mut MFVideoArea).cast::<u8>(), size),
                None,
            )
        };
        let visible = match aperture {
            Ok(()) => visible_region(
                (w, h),
                (area.OffsetX.value, area.OffsetY.value),
                (area.Area.cx, area.Area.cy),
            ),
            Err(_) => Region::new(0, 0, w, h),
        };
        // SAFETY: a live attribute store.
        let stride = unsafe { attrs.GetUINT32(&MF_MT_DEFAULT_STRIDE) }.ok();
        Ok(Self {
            coded: (w, h),
            visible,
            pitch: stride
                .map_or(w, |s| (s as i32).unsigned_abs() as usize)
                .max(w),
            colorimetry: colorimetry(&attrs),
        })
    }

    /// The size of the frames: the visible size, rounded up to whole chroma
    /// samples within the coded picture.
    fn frame_size(&self) -> (usize, usize) {
        let even = |v: usize, max: usize| (v + (v & 1)).min(max & !1).max(2);
        (
            even(self.visible.width, self.coded.0),
            even(self.visible.height, self.coded.1),
        )
    }
}

/// The visible region from a display aperture, clamped to the coded size;
/// the whole picture when the aperture is empty.
fn visible_region(coded: (usize, usize), (x, y): (i16, i16), (cx, cy): (i32, i32)) -> Region {
    let (x, y) = (x.max(0) as usize, y.max(0) as usize);
    let (cx, cy) = (cx.max(0) as usize, cy.max(0) as usize);
    if cx == 0 || cy == 0 || x >= coded.0 || y >= coded.1 {
        return Region::new(0, 0, coded.0, coded.1);
    }
    Region::new(x & !1, y & !1, cx.min(coded.0 - x), cy.min(coded.1 - y))
}

/// The colorimetry an output type reports, or `None` when it reports none.
fn colorimetry(attrs: &IMFAttributes) -> Option<Colorimetry> {
    // SAFETY: a live attribute store.
    let get = |key: &GUID| unsafe { attrs.GetUINT32(key) }.ok().map(|v| v as i32);
    let c = Colorimetry {
        space: get(&MF_MT_VIDEO_PRIMARIES).and_then(|v| match v {
            v if v == MFVideoPrimaries_BT709.0 => Some(ColorSpace::Bt709),
            v if v == MFVideoPrimaries_BT2020.0 => Some(ColorSpace::Bt2020),
            v if v == MFVideoPrimaries_SMPTE170M.0 => Some(ColorSpace::Smpte170m),
            _ => None,
        }),
        transfer: get(&MF_MT_TRANSFER_FUNCTION).and_then(|v| match v {
            v if v == MFVideoTransFunc_709.0 => Some(ColorTransfer::Bt709),
            v if v == MFVideoTransFunc_sRGB.0 => Some(ColorTransfer::Srgb),
            v if v == MFVideoTransFunc_2084.0 => Some(ColorTransfer::Pq),
            v if v == MFVideoTransFunc_HLG.0 => Some(ColorTransfer::Hlg),
            v if v == MFVideoTransFunc_10.0 => Some(ColorTransfer::Linear),
            _ => None,
        }),
        encoding: get(&MF_MT_YUV_MATRIX).and_then(|v| match v {
            v if v == MFVideoTransferMatrix_BT601.0 => Some(ColorEncoding::Bt601),
            v if v == MFVideoTransferMatrix_BT709.0 => Some(ColorEncoding::Bt709),
            v if v == MFVideoTransferMatrix_BT2020_10.0 => Some(ColorEncoding::Bt2020),
            _ => None,
        }),
        range: get(&MF_MT_VIDEO_NOMINAL_RANGE).and_then(|v| match v {
            v if v == MFNominalRange_0_255.0 => Some(ColorRange::Full),
            v if v == MFNominalRange_16_235.0 => Some(ColorRange::Limited),
            _ => None,
        }),
    };
    (c != Colorimetry::default()).then_some(c)
}

/// What the decoder has finished, in order, not yet given to the caller.
enum Ready {
    Format(Layout),
    Picture(IMFSample),
}

pub(crate) struct MfDecoder {
    transform: Transform,
    device: String,
    output: OutputPool,
    pool: Vec<Arc<TensorDyn>>,
    free: Vec<u32>,
    generation: u64,
    returns: Arc<Returns>,
    /// The layout of the pictures the pool holds.
    layout: Option<Layout>,
    /// The layout of the latest picture queued in `ready`.
    queued_layout: Option<Layout>,
    info: Option<StreamInfo>,
    ready: VecDeque<Ready>,
    pending: Vec<DecodeEvent>,
    next_seq: i64,
    /// The caller's PTS by sample time.
    times: BTreeMap<i64, u64>,
    /// Last, so Media Foundation stays started until the samples above are
    /// released.
    platform: Platform,
}

// SAFETY: the transform and the samples waiting in `ready` are
// free-threaded Media Foundation objects, used through `&mut self` only,
// so from one thread at a time.
unsafe impl Send for MfDecoder {}

impl MfDecoder {
    /// Opens the first decoder transform that decodes H.264 to NV12.
    pub fn open(config: DecoderConfig) -> Result<Self> {
        let platform = Platform::start()?;
        let candidates = platform.transforms(
            MFT_CATEGORY_VIDEO_DECODER,
            Some(MFVideoFormat_H264),
            MFVideoFormat_NV12,
            false,
            ENV_DECODER,
        )?;
        if candidates.is_empty() {
            return Err(CodecError::NoDevice(
                "no Media Foundation H.264 decoder".into(),
            ));
        }
        let mut last = None;
        for candidate in candidates {
            let name = candidate.name.clone();
            match Transform::activate(candidate).and_then(|t| Self::configure(&platform, t)) {
                Ok(transform) => {
                    let device = format!(
                        "Media Foundation: {name} ({})",
                        if transform.hardware() {
                            "hardware"
                        } else {
                            "software"
                        }
                    );
                    log::info!("h264 decoder {device}");
                    return Ok(Self {
                        transform,
                        platform,
                        device,
                        output: config.output,
                        pool: Vec::new(),
                        free: Vec::new(),
                        generation: 0,
                        returns: Arc::new(Returns::default()),
                        layout: None,
                        queued_layout: None,
                        info: None,
                        ready: VecDeque::new(),
                        pending: Vec::new(),
                        next_seq: 0,
                        times: BTreeMap::new(),
                    });
                }
                Err(e) => {
                    log::debug!("{name}: {e}");
                    last = Some(e);
                }
            }
        }
        Err(last.expect("at least one candidate was tried"))
    }

    fn configure(platform: &Platform, mut transform: Transform) -> Result<Transform> {
        if let Ok(api) = transform.mft().cast::<ICodecAPI>() {
            // SAFETY: a live codec interface and a VT_UI4 value.
            if let Err(e) =
                unsafe { api.SetValue(&CODECAPI_AVLowLatencyMode, &VARIANT::from(1u32)) }
            {
                log::debug!("{}: low latency not set: {}", transform.name(), e.message());
            }
        }
        let input = platform.media_type()?;
        let attrs: IMFAttributes = input.cast().map_err(|e| error(e, "IMFAttributes"))?;
        // SAFETY: a live media type.
        unsafe { attrs.SetGUID(&MF_MT_MAJOR_TYPE, &MFMediaType_Video) }
            .map_err(|e| error(e, "IMFAttributes::SetGUID"))?;
        // SAFETY: as above.
        unsafe { attrs.SetGUID(&MF_MT_SUBTYPE, &MFVideoFormat_H264) }
            .map_err(|e| error(e, "IMFAttributes::SetGUID"))?;
        transform
            .set_input_type(&input)
            .map_err(|e| error(e, "set the H.264 input type"))?;
        let nv12 = transform
            .output_types()
            .into_iter()
            // SAFETY: live media types.
            .find(|t| unsafe { t.GetGUID(&MF_MT_SUBTYPE) }.is_ok_and(|s| s == MFVideoFormat_NV12))
            .ok_or(CodecError::UnsupportedFormat(PixelFormat::Nv12))?;
        transform
            .set_output_type(&nv12)
            .map_err(|e| error(e, "set the NV12 output type"))?;
        transform.begin()?;
        Ok(transform)
    }

    /// Gives back the buffers of frames the caller has dropped.
    fn reclaim(&mut self) {
        let generation = self.generation;
        self.free.extend(self.returns.take(generation));
    }

    /// Queues the transform's outputs as pictures and format changes.
    fn queue(&mut self, outputs: Vec<Output>) -> Result<()> {
        for output in outputs {
            match output {
                Output::Format(t) => {
                    let layout = Layout::of(&t)?;
                    if self.queued_layout.as_ref() != Some(&layout) {
                        self.queued_layout = Some(layout.clone());
                        self.ready.push_back(Ready::Format(layout));
                    }
                }
                Output::Sample(sample) => {
                    if self.queued_layout.is_none() {
                        let layout = Layout::of(&self.transform.output_type()?)?;
                        self.queued_layout = Some(layout.clone());
                        self.ready.push_back(Ready::Format(layout));
                    }
                    self.ready.push_back(Ready::Picture(sample));
                }
            }
        }
        Ok(())
    }

    /// Gives the caller what is ready, as far as the pool allows. With
    /// `all`, a picture left waiting for a buffer is an error.
    fn deliver(&mut self, all: bool) -> Result<()> {
        self.reclaim();
        while let Some(front) = self.ready.front() {
            match front {
                Ready::Format(layout) => {
                    let layout = layout.clone();
                    self.apply(layout)?;
                    self.ready.pop_front();
                }
                Ready::Picture(_) => {
                    let slot = match self.free.pop() {
                        Some(slot) => slot,
                        None => match self.grow()? {
                            Some(slot) => slot,
                            None => break,
                        },
                    };
                    let Some(Ready::Picture(sample)) = self.ready.pop_front() else {
                        unreachable!("the front is a picture");
                    };
                    match self.picture(&sample, slot) {
                        Ok(frame) => self.pending.push(DecodeEvent::Frame(frame)),
                        Err(e) => {
                            self.free.push(slot);
                            return Err(e);
                        }
                    }
                }
            }
        }
        let waiting = self.ready.iter().any(|r| matches!(r, Ready::Picture(_)));
        if all && waiting {
            return Err(CodecError::PoolExhausted {
                held: self.held_by_caller(),
                pool: self.pool.len(),
            });
        }
        Ok(())
    }

    /// Pool buffers in frames the caller has, not counting frames this call
    /// is about to return.
    fn held_by_caller(&self) -> usize {
        let returning = self
            .pending
            .iter()
            .filter(|e| matches!(e, DecodeEvent::Frame(_)))
            .count();
        self.pool.len() - self.free.len() - returning
    }

    /// A new buffer, when the pool is full of frames this call is about to
    /// return: the decoder released more pictures at once than the pool
    /// holds, and the caller cannot drop frames it has not been given.
    fn grow(&mut self) -> Result<Option<u32>> {
        if self.held_by_caller() > 0 {
            return Ok(None);
        }
        let Some((w, h)) = self.layout.as_ref().map(Layout::frame_size) else {
            return Ok(None);
        };
        if let OutputPool::Provided(_) = self.output {
            return Err(CodecError::InvalidConfig(format!(
                "{} released more pictures at once than the {} buffers of the provided pool",
                self.device,
                self.pool.len()
            )));
        }
        let mut tensor = allocate(1, w, h)?.pop().expect("one tensor was allocated");
        tensor.set_colorimetry(self.layout.as_ref().and_then(|l| l.colorimetry));
        self.pool.push(Arc::new(tensor));
        if let Some(info) = &mut self.info {
            info.buffers = self.pool.len();
        }
        log::debug!("{}: pool grown to {}", self.device, self.pool.len());
        Ok(Some(self.pool.len() as u32 - 1))
    }

    /// Makes the pool fit `layout` and reports the new stream format.
    fn apply(&mut self, layout: Layout) -> Result<()> {
        let (w, h) = layout.frame_size();
        let same_size = self.layout.as_ref().map(Layout::frame_size) == Some((w, h));
        if same_size && !self.pool.is_empty() {
            for t in &mut self.pool {
                // A tensor a held frame still shows keeps its colorimetry.
                if let Some(t) = Arc::get_mut(t) {
                    t.set_colorimetry(layout.colorimetry);
                }
            }
        } else {
            let mut tensors = match &mut self.output {
                OutputPool::Allocate { count } => allocate(*count + 1, w, h)?,
                OutputPool::Provided(pool) => {
                    if pool.is_empty() {
                        return Err(CodecError::InvalidConfig(format!(
                            "the provided pool holds {}x{} pictures; on Windows a provided                              pool cannot follow the stream to {w}x{h}",
                            self.info.as_ref().map_or(0, |i| i.width),
                            self.info.as_ref().map_or(0, |i| i.height),
                        )));
                    }
                    if let Some(t) = pool.iter().find(|t| !fits(t, w, h)) {
                        return Err(CodecError::InvalidConfig(format!(
                            "a provided {:?} {:?}x{:?} tensor cannot hold the stream's {w}x{h}                              NV12 pictures: on Windows each must be an NV12 image of that size",
                            t.format(),
                            t.width(),
                            t.height()
                        )));
                    }
                    std::mem::take(pool)
                }
            };
            for t in &mut tensors {
                t.set_colorimetry(layout.colorimetry);
            }
            self.generation += 1;
            self.pool = tensors.into_iter().map(Arc::new).collect();
            self.free = (0..self.pool.len() as u32).rev().collect();
        }
        let info = StreamInfo {
            width: w,
            height: h,
            crop: Region::new(layout.visible.x, layout.visible.y, w, h),
            coded_width: layout.coded.0,
            coded_height: layout.coded.1,
            format: PixelFormat::Nv12,
            colorimetry: layout.colorimetry,
            buffers: self.pool.len(),
            memory: OutputMemory::Imported,
        };
        log::debug!("{}: {info:?}", self.device);
        self.layout = Some(layout);
        self.info = Some(info.clone());
        self.pending.push(DecodeEvent::Reconfigured(info));
        Ok(())
    }

    /// Copies `sample` into pool buffer `slot`.
    fn picture(&mut self, sample: &IMFSample, slot: u32) -> Result<DecodedFrame> {
        let layout = self
            .layout
            .as_ref()
            .expect("a format precedes every picture");
        let size = layout.frame_size();
        let tensor = &self.pool[slot as usize];
        // SAFETY: a live sample.
        let time = unsafe { sample.GetSampleTime() }.ok();
        let pts = match time.and_then(|t| self.times.remove(&t)) {
            Some(pts) => pts,
            None => {
                log::warn!("{}: picture with unknown time {time:?}", self.device);
                0
            }
        };
        copy_picture(sample, layout, tensor, size)?;
        Ok(DecodedFrame::new(
            Arc::clone(tensor),
            pts,
            false,
            (self.generation, slot),
            Arc::clone(&self.returns),
        ))
    }

    /// Discards what the decoder holds after a failure.
    fn reset(&mut self) {
        self.transform.discard();
        self.times.clear();
    }
}

impl DecoderBackend for MfDecoder {
    fn decode(&mut self, access_unit: &[u8], pts: u64) -> Result<Vec<DecodeEvent>> {
        if access_unit.is_empty() {
            return Err(CodecError::InvalidInput("empty access unit".into()));
        }
        com();
        // Pictures still waiting for a buffer go first; until they have one,
        // the access unit is not taken.
        self.deliver(true)?;

        let (sample, buffer) = self.platform.sample(access_unit.len())?;
        let mut ptr = std::ptr::null_mut();
        // SAFETY: a live buffer; unlocked below.
        unsafe { buffer.Lock(&mut ptr, None, None) }
            .map_err(|e| error(e, "IMFMediaBuffer::Lock"))?;
        // SAFETY: the buffer is locked and was created `access_unit.len()`
        // bytes long.
        unsafe { std::ptr::copy_nonoverlapping(access_unit.as_ptr(), ptr, access_unit.len()) };
        // SAFETY: locked above.
        unsafe { buffer.Unlock() }.map_err(|e| error(e, "IMFMediaBuffer::Unlock"))?;
        // SAFETY: a live buffer; the length is its size.
        unsafe { buffer.SetCurrentLength(access_unit.len() as u32) }
            .map_err(|e| error(e, "IMFMediaBuffer::SetCurrentLength"))?;
        let time = self.next_seq;
        // SAFETY: a live sample.
        unsafe { sample.SetSampleTime(time) }.map_err(|e| error(e, "IMFSample::SetSampleTime"))?;
        self.next_seq += 1;
        self.times.insert(time, pts);
        while self.times.len() > TIMES_KEPT {
            self.times.pop_first();
        }

        let mut out = Vec::new();
        let result = self
            .transform
            .input(&self.platform, &sample, INPUT_TIMEOUT, &mut out)
            .and_then(|()| {
                self.transform.collect(
                    &self.platform,
                    OUTPUT_GRACE,
                    &mut out,
                    |out, wants_input| !out.is_empty() || wants_input,
                )
            });
        self.queue(out)?;
        if let Err(e) = result {
            self.times.remove(&time);
            if matches!(e, CodecError::Timeout(_)) {
                self.reset();
            }
            return Err(e);
        }
        self.deliver(false)?;
        Ok(std::mem::take(&mut self.pending))
    }

    fn flush(&mut self) -> Result<Vec<DecodeEvent>> {
        com();
        let mut out = Vec::new();
        let drained = self
            .transform
            .drain(&self.platform, DRAIN_TIMEOUT, &mut out);
        self.queue(out)?;
        if let Err(e) = drained {
            self.reset();
            return Err(e);
        }
        self.deliver(true)?;
        self.times.clear();
        Ok(std::mem::take(&mut self.pending))
    }

    fn stream_info(&self) -> Option<&StreamInfo> {
        self.info.as_ref()
    }

    fn device(&self) -> String {
        self.device.clone()
    }
}

/// `count` NV12 images of `w`x`h` for the pool: D3D11 textures, or host
/// memory when the process has no D3D11 device.
fn allocate(count: usize, w: usize, h: usize) -> Result<Vec<TensorDyn>> {
    let make = |memory| {
        (0..count)
            .map(|_| {
                TensorDyn::image(
                    w,
                    h,
                    PixelFormat::Nv12,
                    DType::U8,
                    Some(memory),
                    CpuAccess::ReadWrite,
                )
            })
            .collect::<std::result::Result<Vec<_>, _>>()
    };
    match make(TensorMemory::DmaBuf) {
        Ok(t) => Ok(t),
        Err(e) => {
            log::warn!("no D3D11 textures for decoded pictures ({e}); decoding into host memory");
            make(TensorMemory::Mem).map_err(CodecError::Tensor)
        }
    }
}

/// Whether a provided tensor can hold a `w`x`h` decoded picture.
fn fits(t: &TensorDyn, w: usize, h: usize) -> bool {
    t.format() == Some(PixelFormat::Nv12)
        && t.dtype() == DType::U8
        && t.width() == Some(w)
        && t.height() == Some(h)
        && !t.is_multiplane()
}

/// Copies the visible picture of `sample` into `tensor`.
fn copy_picture(
    sample: &IMFSample,
    layout: &Layout,
    tensor: &TensorDyn,
    size: (usize, usize),
) -> Result<()> {
    // SAFETY: a live sample.
    let buffer = unsafe { sample.ConvertToContiguousBuffer() }
        .map_err(|e| error(e, "IMFSample::ConvertToContiguousBuffer"))?;
    let stride = tensor
        .effective_row_stride()
        .ok_or_else(|| CodecError::InvalidConfig("a pool tensor has no row stride".into()))?;
    let to: Vec<Plane> = PixelFormat::Nv12
        .plane_table(size.0, size.1, stride)
        .ok_or(CodecError::UnsupportedFormat(PixelFormat::Nv12))?
        .iter()
        .map(|p| Plane {
            offset: p.offset as usize,
            pitch: p.stride as usize,
        })
        .collect();
    let mut map = match tensor.map_bytes(CpuAccess::Write) {
        Err(edgefirst_tensor::Error::PartialWriteRequiresReadWrite(_)) => {
            tensor.map_bytes(CpuAccess::ReadWrite)
        }
        other => other,
    }?;
    let origin = (layout.visible.x, layout.visible.y);
    let into = |bytes: &[u8], top: usize, pitch: usize, map: &mut [u8]| {
        let planes = copy::contiguous(PixelFormat::Nv12, pitch, layout.coded.1);
        let planes: Vec<Plane> = planes
            .iter()
            .map(|p| Plane {
                offset: p.offset + top,
                pitch: p.pitch,
            })
            .collect();
        copy::copy(
            PixelFormat::Nv12,
            size,
            bytes,
            &copy::offset(PixelFormat::Nv12, &planes, origin),
            map,
            &to,
        )
        .map_err(CodecError::InvalidData)
    };
    if let Ok(b) = buffer.cast::<IMF2DBuffer2>() {
        let (mut scan0, mut pitch) = (std::ptr::null_mut(), 0);
        let (mut start, mut len) = (std::ptr::null_mut(), 0);
        // SAFETY: valid out-pointers; unlocked below.
        unsafe {
            b.Lock2DSize(
                MF2DBuffer_LockFlags_Read,
                &mut scan0,
                &mut pitch,
                &mut start,
                &mut len,
            )
        }
        .map_err(|e| error(e, "IMF2DBuffer2::Lock2DSize"))?;
        let copied = if pitch <= 0 {
            Err(CodecError::InvalidData(format!(
                "a decoded picture has a pitch of {pitch}"
            )))
        } else {
            // SAFETY: the buffer is locked and `len` bytes long from `start`.
            let bytes = unsafe { std::slice::from_raw_parts(start, len as usize) };
            let top = (scan0 as usize).wrapping_sub(start as usize);
            into(bytes, top, pitch as usize, &mut map)
        };
        // SAFETY: locked above.
        let _ = unsafe { b.Unlock2D() };
        return copied;
    }
    let (mut ptr, mut len) = (std::ptr::null_mut(), 0u32);
    // SAFETY: valid out-pointers; unlocked below.
    unsafe { buffer.Lock(&mut ptr, None, Some(&mut len)) }
        .map_err(|e| error(e, "IMFMediaBuffer::Lock"))?;
    // SAFETY: the buffer is locked and holds `len` bytes.
    let bytes = unsafe { std::slice::from_raw_parts(ptr, len as usize) };
    let copied = into(bytes, 0, layout.pitch, &mut map);
    // SAFETY: locked above.
    let _ = unsafe { buffer.Unlock() };
    copied
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_aperture_is_the_visible_picture() {
        assert_eq!(
            visible_region((1920, 1088), (0, 0), (1920, 1080)),
            Region::new(0, 0, 1920, 1080)
        );
        assert_eq!(
            visible_region((640, 480), (8, 4), (624, 472)),
            Region::new(8, 4, 624, 472)
        );
    }

    #[test]
    fn an_empty_or_outside_aperture_is_the_whole_picture() {
        assert_eq!(
            visible_region((1920, 1088), (0, 0), (0, 0)),
            Region::new(0, 0, 1920, 1088)
        );
        assert_eq!(
            visible_region((640, 480), (700, 0), (16, 16)),
            Region::new(0, 0, 640, 480)
        );
        assert_eq!(
            visible_region((640, 480), (-4, 0), (2000, 480)),
            Region::new(0, 0, 640, 480)
        );
    }

    #[test]
    fn frames_have_even_sizes_within_the_coded_picture() {
        let layout = |w, h| Layout {
            coded: (1920, 1088),
            visible: Region::new(0, 0, w, h),
            pitch: 1920,
            colorimetry: None,
        };
        assert_eq!(layout(1920, 1080).frame_size(), (1920, 1080));
        assert_eq!(layout(1279, 719).frame_size(), (1280, 720));
    }

    #[test]
    fn a_provided_tensor_must_be_an_nv12_image_of_the_stream_size() {
        let image = |w, h, format| {
            TensorDyn::image(
                w,
                h,
                format,
                DType::U8,
                Some(TensorMemory::Mem),
                CpuAccess::ReadWrite,
            )
            .unwrap()
        };
        assert!(fits(&image(640, 480, PixelFormat::Nv12), 640, 480));
        assert!(!fits(&image(1280, 720, PixelFormat::Nv12), 640, 480));
        assert!(!fits(&image(640, 480, PixelFormat::Yuyv), 640, 480));
        let raw = TensorDyn::new(&[4096], DType::U8, Some(TensorMemory::Mem), None).unwrap();
        assert!(!fits(&raw, 640, 480));
    }
}
