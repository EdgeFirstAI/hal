// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! H.264 decoding with a VideoToolbox decompression session.
//!
//! Every access unit's SPS and PPS update a [`ParameterSets`]; the pair its
//! picture refers to makes the session's format description. The picture's
//! NAL units, length-prefixed, go to `VTDecompressionSessionDecodeFrame`
//! with the access unit's sequence number, and the output callback queues
//! each decoded pixel buffer under that number.
//!
//! Decoded pictures wait in a reorder queue keyed by output epoch and
//! picture order count, and leave it in display order. Each is then lent as
//! a multiplane NV12 tensor over its pixel buffer's IOSurface. Its pool slot
//! keeps the pixel buffer until the frame drops, so VideoToolbox does not
//! reuse it while it is lent.
//!
//! With a provided pool each picture is instead copied into the next free
//! tensor, and its pixel buffer is released at once.

use std::collections::{BTreeMap, HashMap, VecDeque};
use std::ffi::c_void;
use std::sync::{Arc, Mutex};

use edgefirst_tensor::{DType, PixelFormat, Region, TensorDyn};

use super::color::from_attachments;
use super::ffi::*;
use super::{chroma_offset, status, FOURCC_420F, FOURCC_420V};
use crate::video::copy::{copy_into_image, offset, Plane};
use crate::video::h264::{annex_b_to_avcc, nal_units, NAL_AUD, NAL_IDR, NAL_PPS, NAL_SPS};
use crate::video::h264_syntax::{ParameterSets, PocCounter, SliceHeader, Sps};
use crate::video::{
    DecodeEvent, DecodedFrame, DecoderBackend, DecoderConfig, OutputMemory, OutputPool, Returns,
    StreamInfo,
};
use crate::{CodecError, Engine, Result, Transfer};

const DEVICE: &str = "VideoToolbox H.264 decoder";
/// The pool generation of every lent frame: slots outlive formats and
/// sessions.
const GENERATION: u64 = 1;
/// `nal_unit_type` of a coded slice of a non-IDR picture.
const NAL_SLICE: u8 = 1;

/// The geometry and range one SPS gives its pictures.
#[derive(Debug, PartialEq)]
struct Format {
    visible: Region,
    coded: (usize, usize),
    full_range: bool,
    reorder_depth: usize,
}

impl Format {
    fn of(sps: &Sps) -> Self {
        Self {
            visible: sps.visible,
            coded: sps.coded,
            full_range: sps.vui.as_ref().and_then(|v| v.full_range).unwrap_or(false),
            reorder_depth: sps.reorder_depth(),
        }
    }
}

/// What the output callback received for one access unit.
struct Output {
    seq: u64,
    status: OSStatus,
    flags: VTDecodeInfoFlags,
    image: Option<CfOwned>,
}

/// State the output callback shares with the decoder, from any thread.
#[derive(Default)]
struct Shared {
    outputs: Mutex<Vec<Output>>,
}

impl Shared {
    fn take(&self) -> Vec<Output> {
        std::mem::take(&mut *self.outputs.lock().unwrap_or_else(|e| e.into_inner()))
    }
}

/// The session's output callback.
///
/// # Safety
///
/// `ref_con` is the `Shared` of a live [`Session`], and `image` is null or a
/// live `CVImageBuffer`.
unsafe extern "C" fn on_output(
    ref_con: *mut c_void,
    source_frame_ref_con: *mut c_void,
    status: OSStatus,
    flags: VTDecodeInfoFlags,
    image: CVImageBufferRef,
    _presentation_time_stamp: CMTime,
    _presentation_duration: CMTime,
) {
    // SAFETY: the session's `Arc<Shared>` keeps `ref_con` live until the
    // session is invalidated, after which the callback no longer runs.
    let shared = unsafe { &*ref_con.cast_const().cast::<Shared>() };
    // SAFETY: `image` is null or a live image buffer; the +0 reference is
    // retained to keep it.
    let image = unsafe { CfOwned::retained(image.cast_const()) };
    shared
        .outputs
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .push(Output {
            seq: source_frame_ref_con as u64,
            status,
            flags,
            image,
        });
}

/// A decompression session, waited for and invalidated on drop.
struct Session {
    session: CfOwned,
    full_range: bool,
    invalidated: bool,
    _shared: Arc<Shared>,
}

impl Session {
    fn new(description: &CfOwned, full_range: bool, shared: &Arc<Shared>) -> Result<Self> {
        let fourcc = if full_range { FOURCC_420F } else { FOURCC_420V };
        let alloc = || CodecError::Device {
            op: "CFDictionaryCreate".into(),
            source: std::io::Error::other("out of memory"),
        };
        let format = cf_number_i32(fourcc as i32).ok_or_else(alloc)?;
        // SAFETY: the keys are CoreVideo constants and the values live CF
        // objects held across the call.
        let attributes = unsafe {
            let surface = cf_dictionary(&[]).ok_or_else(alloc)?;
            cf_dictionary(&[
                (kCVPixelBufferPixelFormatTypeKey, format.as_ptr()),
                (kCVPixelBufferIOSurfacePropertiesKey, surface.as_ptr()),
            ])
            .ok_or_else(alloc)?
        };
        let record = VTDecompressionOutputCallbackRecord {
            callback: Some(on_output),
            ref_con: Arc::as_ptr(shared).cast_mut().cast(),
        };
        let mut session = std::ptr::null_mut();
        // SAFETY: the description and attributes are live, `record` is read
        // during the call, its `ref_con` is kept live by `_shared` until the
        // session is invalidated, and `session` is a valid out-pointer.
        let s = unsafe {
            VTDecompressionSessionCreate(
                std::ptr::null(),
                description.as_ptr(),
                std::ptr::null(),
                attributes.as_ptr(),
                &record,
                &mut session,
            )
        };
        // SAFETY: on success `session` holds a +1 reference.
        match unsafe { CfOwned::from_create(session) } {
            Some(session) if s == 0 => Ok(Self {
                session,
                full_range,
                invalidated: false,
                _shared: Arc::clone(shared),
            }),
            _ => Err(status("VTDecompressionSessionCreate", s)),
        }
    }

    /// Waits for every picture in flight to reach the callback.
    fn wait(&self) -> Result<()> {
        // SAFETY: a live session.
        match unsafe { VTDecompressionSessionWaitForAsynchronousFrames(self.session.as_ptr()) } {
            0 => Ok(()),
            s => Err(status("VTDecompressionSessionWaitForAsynchronousFrames", s)),
        }
    }

    /// Waits for the session, then invalidates it, once.
    fn invalidate(&mut self) {
        if !self.invalidated {
            let _ = self.wait();
            // SAFETY: a live session, invalidated once.
            unsafe { VTDecompressionSessionInvalidate(self.session.as_ptr()) };
            self.invalidated = true;
        }
    }

    fn accepts(&self, description: &CfOwned) -> bool {
        // SAFETY: a live session and a live format description.
        unsafe {
            VTDecompressionSessionCanAcceptFormatDescription(
                self.session.as_ptr(),
                description.as_ptr(),
            ) != 0
        }
    }
}

impl Drop for Session {
    fn drop(&mut self) {
        self.invalidate();
    }
}

/// Output epoch, picture order count and sequence number.
type Key = (u64, i32, u64);

/// An access unit's picture in the reorder queue.
struct Entry {
    pts: u64,
    format: Arc<Format>,
    /// The pixel buffer and whether it is corrupted, once decoded.
    image: Option<(CfOwned, bool)>,
}

/// A picture out of the reorder queue, waiting for a pool slot.
struct Picture {
    pts: u64,
    format: Arc<Format>,
    buffer: CfOwned,
    corrupted: bool,
}

/// A pool slot: the pixel buffer and tensor of the frame lent from it. A
/// provided pool's slots always hold their tensor and no pixel buffer.
#[derive(Default)]
struct Slot {
    /// Keeps the pixel buffer out of VideoToolbox's pool.
    _buffer: Option<CfOwned>,
    tensor: Option<Arc<TensorDyn>>,
}

/// A locked pixel buffer, unlocked on drop.
struct Locked(CVPixelBufferRef);

impl Locked {
    /// Locks `buffer` for reading, which must stay live while this does.
    fn read(buffer: CVPixelBufferRef) -> Result<Self> {
        // SAFETY: `buffer` is a live pixel buffer.
        match unsafe { CVPixelBufferLockBaseAddress(buffer, kCVPixelBufferLock_ReadOnly) } {
            0 => Ok(Self(buffer)),
            s => Err(status("CVPixelBufferLockBaseAddress", s)),
        }
    }
}

impl Drop for Locked {
    fn drop(&mut self) {
        // SAFETY: locked by `read`, once, and still live.
        unsafe { CVPixelBufferUnlockBaseAddress(self.0, kCVPixelBufferLock_ReadOnly) };
    }
}

/// Whether a provided tensor is an NV12 image of exactly `w`x`h`.
fn fits(t: &TensorDyn, w: usize, h: usize) -> bool {
    t.format() == Some(PixelFormat::Nv12)
        && t.dtype() == DType::U8
        && t.width() == Some(w)
        && t.height() == Some(h)
}

/// Copies the `visible` region of the NV12 pixel buffer `buffer` into the
/// image `tensor`.
fn copy_picture(buffer: &CfOwned, visible: Region, tensor: &TensorDyn) -> Result<()> {
    let buffer = buffer.as_ptr();
    let short = || CodecError::InvalidData("a decoded picture is smaller than the stream".into());
    // SAFETY: `buffer` is a live pixel buffer.
    let (planes, width, height) = unsafe {
        (
            CVPixelBufferGetPlaneCount(buffer),
            CVPixelBufferGetWidth(buffer),
            CVPixelBufferGetHeight(buffer),
        )
    };
    if planes != 2 {
        return Err(CodecError::InvalidData(format!(
            "a decoded picture has {planes} planes, not 2"
        )));
    }
    if visible.x + visible.width > width || visible.y + visible.height > height {
        return Err(short());
    }
    let _lock = Locked::read(buffer)?;
    // SAFETY: the buffer is locked, so its plane addresses are valid.
    let (luma, chroma, pitches, rows) = unsafe {
        (
            CVPixelBufferGetBaseAddressOfPlane(buffer, 0).cast::<u8>(),
            CVPixelBufferGetBaseAddressOfPlane(buffer, 1).cast::<u8>(),
            [
                CVPixelBufferGetBytesPerRowOfPlane(buffer, 0),
                CVPixelBufferGetBytesPerRowOfPlane(buffer, 1),
            ],
            [
                CVPixelBufferGetHeightOfPlane(buffer, 0),
                CVPixelBufferGetHeightOfPlane(buffer, 1),
            ],
        )
    };
    let (chroma_at, len) = chroma_offset(luma as usize, chroma as usize, pitches, rows)
        .ok_or_else(|| {
            CodecError::InvalidData("a decoded picture's planes are not in one buffer".into())
        })?;
    // SAFETY: the planes of an IOSurface-backed pixel buffer share one
    // mapping, which the lock keeps valid and nothing writes to while the
    // picture is held, and `len` ends at the end of the chroma plane.
    let src = unsafe { std::slice::from_raw_parts(luma, len) };
    let from = [
        Plane {
            offset: 0,
            pitch: pitches[0],
        },
        Plane {
            offset: chroma_at,
            pitch: pitches[1],
        },
    ];
    let from = offset(PixelFormat::Nv12, &from, (visible.x, visible.y));
    copy_into_image(
        PixelFormat::Nv12,
        (visible.width, visible.height),
        src,
        &from,
        tensor,
    )
}

pub(crate) struct VtDecoder {
    sets: ParameterSets,
    poc: PocCounter,
    /// Starts again at every IDR or MMCO 5 picture and every flush.
    epoch: u64,
    next_seq: u64,
    shared: Arc<Shared>,
    session: Option<Session>,
    /// The SPS and PPS NAL units of `description`.
    params: Option<(Vec<u8>, Vec<u8>)>,
    description: Option<CfOwned>,
    format: Option<Arc<Format>>,
    reorder: BTreeMap<Key, Entry>,
    /// The reorder key by sequence number.
    keys: HashMap<u64, Key>,
    ready: VecDeque<Picture>,
    pending: Vec<DecodeEvent>,
    slots: Vec<Slot>,
    free: Vec<u32>,
    /// Slots given back by frames whose tensor reference has not dropped.
    releasing: Vec<u32>,
    returns: Arc<Returns>,
    info: Option<StreamInfo>,
    /// Whether the pool is the caller's, which pictures are copied into.
    provided: bool,
    /// The size of the pictures the provided pool was checked against.
    provided_size: Option<(usize, usize)>,
    /// The format of the last picture lent.
    shown: Option<Arc<Format>>,
    /// Pictures decoded, and those whose callback ran before
    /// `VTDecompressionSessionDecodeFrame` returned.
    decoded: u64,
    inside: u64,
    #[cfg(test)]
    fail_next_decode: Option<OSStatus>,
}

impl VtDecoder {
    /// A decoder whose pool lends `count + 1` pictures before it grows, or
    /// whose provided pool takes the pictures.
    pub(crate) fn open(config: DecoderConfig) -> Result<Self> {
        let (slots, provided) = match config.output {
            OutputPool::Allocate { count } => {
                ((0..=count).map(|_| Slot::default()).collect(), false)
            }
            OutputPool::Provided(pool) => (
                pool.into_iter()
                    .map(|t| Slot {
                        _buffer: None,
                        tensor: Some(Arc::new(t)),
                    })
                    .collect::<Vec<_>>(),
                true,
            ),
        };
        log::info!("H.264 decoder {DEVICE}");
        Ok(Self {
            sets: ParameterSets::default(),
            poc: PocCounter::new(),
            epoch: 0,
            next_seq: 0,
            shared: Arc::new(Shared::default()),
            session: None,
            params: None,
            description: None,
            format: None,
            reorder: BTreeMap::new(),
            keys: HashMap::new(),
            ready: VecDeque::new(),
            pending: Vec::new(),
            free: (0..slots.len() as u32).rev().collect(),
            slots,
            releasing: Vec::new(),
            returns: Arc::new(Returns::default()),
            info: None,
            provided,
            provided_size: None,
            shown: None,
            decoded: 0,
            inside: 0,
            #[cfg(test)]
            fail_next_decode: None,
        })
    }

    /// Invalidates the session once every picture in flight has reached the
    /// callback, leaving it in place.
    #[cfg(test)]
    pub(crate) fn invalidate_session_for_test(&mut self) {
        if let Some(session) = &mut self.session {
            session.invalidate();
        }
    }

    /// Makes the next `VTDecompressionSessionDecodeFrame` that succeeds
    /// report `status` instead, once its picture has reached the callback.
    #[cfg(test)]
    pub(crate) fn fail_next_decode_for_test(&mut self, status: OSStatus) {
        self.fail_next_decode = Some(status);
    }

    /// Makes the format description, and the session, those of PPS
    /// `pps_id` and its SPS. A session that cannot take the new description
    /// is replaced.
    fn use_parameter_sets(&mut self, pps_id: u32) -> Result<()> {
        let (sps_nal, pps_nal) = self.sets.nal_units_for_pps(pps_id).ok_or_else(|| {
            CodecError::InvalidData(format!("H.264 PPS {pps_id} or its SPS is absent"))
        })?;
        let same = self
            .params
            .as_ref()
            .is_some_and(|(s, p)| s == sps_nal && p == pps_nal);
        if same && self.session.is_some() {
            return Ok(());
        }
        let params = (sps_nal.to_vec(), pps_nal.to_vec());
        let description = format_description(&params.0, &params.1)?;
        let format = Format::of(
            self.sets
                .sps_for_pps(pps_id)
                .expect("the PPS's SPS is present"),
        );
        self.check_pool(&format)?;
        let accepted = self
            .session
            .as_ref()
            .is_some_and(|s| s.full_range == format.full_range && s.accepts(&description));
        if !accepted {
            self.end_session();
            self.session = Some(Session::new(&description, format.full_range, &self.shared)?);
            log::debug!("{DEVICE}: new session for {format:?}");
        }
        if self.format.as_deref() != Some(&format) {
            self.format = Some(Arc::new(format));
        }
        self.params = Some(params);
        self.description = Some(description);
        Ok(())
    }

    /// Checks that a provided pool holds pictures of `format`'s size, and
    /// keeps to the size of the first.
    fn check_pool(&mut self, format: &Format) -> Result<()> {
        if !self.provided {
            return Ok(());
        }
        let (w, h) = (format.visible.width, format.visible.height);
        match self.provided_size {
            Some(size) if size == (w, h) => {}
            Some((pw, ph)) => {
                return Err(CodecError::InvalidConfig(format!(
                    "the provided pool holds {pw}x{ph} pictures; a provided pool cannot follow the stream to {w}x{h}"
                )))
            }
            None => {
                let bad = self
                    .slots
                    .iter()
                    .filter_map(|s| s.tensor.as_deref())
                    .find(|t| !fits(t, w, h));
                if let Some(t) = bad {
                    return Err(CodecError::InvalidConfig(format!(
                        "a provided {:?} {:?}x{:?} tensor cannot hold the stream's {w}x{h} NV12 pictures: each must be an NV12 image of that size",
                        t.format(),
                        t.width(),
                        t.height()
                    )));
                }
                self.provided_size = Some((w, h));
            }
        }
        Ok(())
    }

    /// Waits for the session, invalidates it and releases every decoded
    /// picture in the reorder queue.
    fn end_session(&mut self) {
        if let Some(session) = self.session.take() {
            if let Err(e) = session.wait() {
                log::debug!("{DEVICE}: {e}");
            }
        }
        self.collect();
        self.release(true);
    }

    /// Moves what the callback has received into the reorder queue.
    fn collect(&mut self) {
        for output in self.shared.take() {
            let Some(&key) = self.keys.get(&output.seq) else {
                continue;
            };
            if output.status != 0 {
                log::debug!(
                    "{DEVICE}: access unit {} decoded with OSStatus {}",
                    output.seq,
                    output.status
                );
            }
            match output.image {
                Some(buffer) => {
                    let corrupted =
                        output.status != 0 || output.flags & kVTDecodeInfo_FrameDropped != 0;
                    if let Some(entry) = self.reorder.get_mut(&key) {
                        entry.image = Some((buffer, corrupted));
                    }
                }
                None => {
                    log::debug!("{DEVICE}: access unit {} has no picture", output.seq);
                    self.reorder.remove(&key);
                    self.keys.remove(&output.seq);
                }
            }
        }
    }

    /// Moves pictures from the front of the reorder queue to `ready`: those
    /// of an earlier epoch, then any while the queue holds more than the
    /// reorder depth. A picture not yet decoded stops this, except with
    /// `all`, which empties the queue and drops what was not decoded.
    fn release(&mut self, all: bool) {
        let depth = self.format.as_ref().map_or(0, |f| f.reorder_depth);
        loop {
            let len = self.reorder.len();
            let Some(entry) = self.reorder.first_entry() else {
                break;
            };
            let earlier = entry.key().0 < self.epoch;
            if !(all || earlier || len > depth) {
                break;
            }
            if entry.get().image.is_none() && !all {
                break;
            }
            let ((_, _, seq), entry) = entry.remove_entry();
            self.keys.remove(&seq);
            if let Some((buffer, corrupted)) = entry.image {
                self.ready.push_back(Picture {
                    pts: entry.pts,
                    format: entry.format,
                    buffer,
                    corrupted,
                });
            }
        }
    }

    /// Frees the slots of frames the caller has dropped, once the pool holds
    /// the last reference to their tensor.
    fn reclaim(&mut self) {
        let mut returned = std::mem::take(&mut self.releasing);
        returned.extend(self.returns.take(GENERATION));
        for index in returned {
            let slot = &mut self.slots[index as usize];
            if slot
                .tensor
                .as_ref()
                .is_some_and(|t| Arc::strong_count(t) > 1)
            {
                self.releasing.push(index);
            } else {
                if !self.provided {
                    *slot = Slot::default();
                }
                self.free.push(index);
            }
        }
    }

    /// Slots lent to frames the caller has, not counting frames this call
    /// is about to return.
    fn held_by_caller(&self) -> usize {
        let returning = self
            .pending
            .iter()
            .filter(|e| matches!(e, DecodeEvent::Frame(_)))
            .count();
        self.slots.len() - self.free.len() - returning
    }

    /// Lends `ready` pictures as far as the pool allows. With `all`, a
    /// picture left waiting while the caller holds frames is
    /// [`CodecError::PoolExhausted`]. A provided pool whose every slot is
    /// lent in this call leaves the rest waiting for the next call.
    fn deliver(&mut self, all: bool) -> Result<()> {
        self.reclaim();
        while !self.ready.is_empty() {
            let slot = match self.free.pop() {
                Some(slot) => slot,
                None if self.held_by_caller() == 0 && !self.provided => self.grow(),
                None => break,
            };
            let picture = self.ready.pop_front().expect("a picture is ready");
            match self.lend(picture, slot) {
                Ok(frame) => self.pending.push(DecodeEvent::Frame(frame)),
                Err(e) => {
                    self.free.push(slot);
                    return Err(e);
                }
            }
        }
        if all && !self.ready.is_empty() && self.held_by_caller() > 0 {
            return Err(CodecError::PoolExhausted {
                held: self.held_by_caller(),
                pool: self.slots.len(),
            });
        }
        Ok(())
    }

    /// Lends every `ready` picture from the provided pool, or none. Frames
    /// an earlier call left to return that fill the tensors the pictures
    /// need are returned first, with the pictures left waiting.
    ///
    /// # Errors
    ///
    /// [`CodecError::InvalidConfig`] when the pool holds fewer tensors than
    /// the pictures waiting, and [`CodecError::PoolExhausted`] when they fit
    /// the pool but not its free tensors while the caller holds frames.
    fn deliver_every_or_none(&mut self) -> Result<()> {
        self.reclaim();
        let (waiting, free, pool) = (self.ready.len(), self.free.len(), self.slots.len());
        if waiting <= free {
            return self.deliver(false);
        }
        if waiting > pool {
            return Err(CodecError::InvalidConfig(format!(
                "the provided pool holds {pool} tensors and the drain needs {waiting}"
            )));
        }
        match self.held_by_caller() {
            0 => Ok(()),
            held => Err(CodecError::PoolExhausted { held, pool }),
        }
    }

    /// A new slot, for an allocated pool full of frames this call is about
    /// to return.
    fn grow(&mut self) -> u32 {
        self.slots.push(Slot::default());
        if let Some(info) = &mut self.info {
            info.buffers = self.slots.len();
        }
        log::debug!("{DEVICE}: pool grown to {}", self.slots.len());
        self.slots.len() as u32 - 1
    }

    /// Lends `picture` from `slot`, reporting its format first when it
    /// differs from the last one reported.
    fn lend(&mut self, picture: Picture, slot: u32) -> Result<DecodedFrame> {
        let buffer = picture.buffer.as_ptr();
        // SAFETY: `picture.buffer` is a live pixel buffer.
        let (surface, fourcc) = unsafe {
            (
                CVPixelBufferGetIOSurface(buffer),
                CVPixelBufferGetPixelFormatType(buffer),
            )
        };
        if surface.is_null() {
            return Err(CodecError::InvalidData(
                "a decoded picture is not on an IOSurface".into(),
            ));
        }
        // SAFETY: `buffer` is a live pixel buffer.
        let colorimetry = unsafe { from_attachments(buffer, fourcc) };
        let format = Arc::clone(&picture.format);
        let (tensor, held) = if self.provided {
            let tensor = self.slots[slot as usize]
                .tensor
                .as_mut()
                .expect("a provided slot holds its tensor");
            copy_picture(&picture.buffer, format.visible, tensor)?;
            if tensor.colorimetry() != Some(colorimetry) {
                Arc::get_mut(tensor)
                    .expect("a free slot's tensor has no other reference")
                    .set_colorimetry(Some(colorimetry));
            }
            (Arc::clone(tensor), None)
        } else {
            // SAFETY: the pixel buffer keeps `surface` live, and the tensor
            // retains it.
            let mut tensor = unsafe { TensorDyn::from_planar_iosurface(surface, None) }?;
            tensor.set_colorimetry(Some(colorimetry));
            (Arc::new(tensor), Some(picture.buffer))
        };
        if self.shown.as_ref().is_none_or(|s| !Arc::ptr_eq(s, &format)) {
            let info = StreamInfo {
                width: format.visible.width,
                height: format.visible.height,
                crop: format.visible,
                coded_width: format.coded.0,
                coded_height: format.coded.1,
                format: PixelFormat::Nv12,
                colorimetry: Some(colorimetry),
                buffers: self.slots.len(),
                memory: if self.provided {
                    OutputMemory::Imported
                } else {
                    OutputMemory::Exported
                },
                engine: Engine::VideoToolbox,
                transfer: if self.provided {
                    Transfer::CpuCopy
                } else {
                    Transfer::ZeroCopy
                },
            };
            if self.info.as_ref() != Some(&info) {
                log::debug!("{DEVICE}: {info:?}");
                self.pending.push(DecodeEvent::Reconfigured(info.clone()));
                self.info = Some(info);
            }
            self.shown = Some(format);
        }
        if let Some(buffer) = held {
            self.slots[slot as usize] = Slot {
                _buffer: Some(buffer),
                tensor: Some(Arc::clone(&tensor)),
            };
        }
        Ok(DecodedFrame::new(
            tensor,
            picture.pts,
            picture.corrupted,
            (GENERATION, slot),
            Arc::clone(&self.returns),
        ))
    }

    /// Submits the picture of `au`, whose first slice is `slice`.
    fn submit(&mut self, au: &[u8], slice: &[u8], pts: u64) -> Result<()> {
        let header = SliceHeader::parse(slice, &self.sets).map_err(CodecError::InvalidData)?;
        self.use_parameter_sets(header.pps_id)?;
        let sps = self
            .sets
            .sps_for_pps(header.pps_id)
            .expect("the slice's SPS is present");
        let poc = self.poc.picture_order(&header, sps);
        if header.idr || header.mmco5 {
            self.epoch += 1;
        }
        let (Some(session), Some(description), Some(format)) =
            (&self.session, &self.description, &self.format)
        else {
            unreachable!("a session follows the parameter sets");
        };
        let sample = sample(
            &annex_b_to_avcc(au, &[NAL_SPS, NAL_PPS, NAL_AUD]),
            description,
        )?;
        let seq = self.next_seq;
        self.next_seq += 1;
        let key = (self.epoch, poc, seq);
        self.reorder.insert(
            key,
            Entry {
                pts,
                format: Arc::clone(format),
                image: None,
            },
        );
        self.keys.insert(seq, key);
        let mut flags = 0;
        // SAFETY: a live session and sample; the sequence number travels
        // as the frame's pointer-sized ref-con and is never dereferenced.
        let s = unsafe {
            VTDecompressionSessionDecodeFrame(
                session.session.as_ptr(),
                sample.as_ptr(),
                0,
                seq as *mut c_void,
                &mut flags,
            )
        };
        #[cfg(test)]
        let s = match self.fail_next_decode.take() {
            Some(fail) if s == 0 => {
                let _ = session.wait();
                fail
            }
            _ => s,
        };
        self.decoded += 1;
        if s != 0 {
            self.collect();
            match self.reorder.get_mut(&key) {
                Some(Entry {
                    image: Some((_, corrupted)),
                    ..
                }) => *corrupted = true,
                _ => {
                    self.reorder.remove(&key);
                    self.keys.remove(&seq);
                }
            }
            if s == kVTInvalidSessionErr {
                self.end_session();
            }
            return Err(status("VTDecompressionSessionDecodeFrame", s));
        }
        let back = self
            .shared
            .outputs
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .iter()
            .any(|o| o.seq == seq);
        self.inside += u64::from(back);
        Ok(())
    }
}

impl DecoderBackend for VtDecoder {
    fn decode(&mut self, access_unit: &[u8], pts: u64) -> Result<Vec<DecodeEvent>> {
        if access_unit.is_empty() {
            return Err(CodecError::InvalidInput("empty access unit".into()));
        }
        self.collect();
        self.release(false);
        self.deliver(true)?;
        let mut slice = None;
        for nal in nal_units(access_unit) {
            match nal.first().map(|h| h & 0x1f) {
                Some(NAL_SPS | NAL_PPS) => {
                    self.sets.update(nal).map_err(CodecError::InvalidData)?;
                }
                Some(NAL_SLICE | NAL_IDR) if slice.is_none() => slice = Some(nal),
                _ => {}
            }
        }
        if let Some(slice) = slice {
            self.submit(access_unit, slice, pts)?;
            self.collect();
            self.release(false);
            self.deliver(false)?;
        }
        Ok(std::mem::take(&mut self.pending))
    }

    fn flush(&mut self) -> Result<Vec<DecodeEvent>> {
        if let Some(Err(e)) = self.session.as_ref().map(Session::wait) {
            log::warn!("{DEVICE}: {e}; the session ends");
            self.end_session();
        }
        self.collect();
        self.release(true);
        self.poc.reset();
        self.epoch += 1;
        log::debug!(
            "{DEVICE}: {} of {} pictures came back inside VTDecompressionSessionDecodeFrame",
            self.inside,
            self.decoded
        );
        if self.provided {
            self.deliver_every_or_none()?;
        } else {
            self.deliver(true)?;
        }
        Ok(std::mem::take(&mut self.pending))
    }

    fn stream_info(&self) -> Option<&StreamInfo> {
        self.info.as_ref()
    }

    fn device(&self) -> String {
        DEVICE.into()
    }

    fn engine(&self) -> Engine {
        Engine::VideoToolbox
    }
}

/// A format description of an SPS and PPS, with 4-byte NAL lengths.
fn format_description(sps: &[u8], pps: &[u8]) -> Result<CfOwned> {
    let pointers = [sps.as_ptr(), pps.as_ptr()];
    let sizes = [sps.len(), pps.len()];
    let mut description = std::ptr::null();
    // SAFETY: two parameter sets of the given sizes, live across the call,
    // and a valid out-pointer.
    let s = unsafe {
        CMVideoFormatDescriptionCreateFromH264ParameterSets(
            std::ptr::null(),
            2,
            pointers.as_ptr(),
            sizes.as_ptr(),
            4,
            &mut description,
        )
    };
    // SAFETY: on success `description` holds a +1 reference.
    match unsafe { CfOwned::from_create(description) } {
        Some(d) if s == 0 => Ok(d),
        _ => Err(status(
            "CMVideoFormatDescriptionCreateFromH264ParameterSets",
            s,
        )),
    }
}

/// A sample buffer of one access unit of length-prefixed NAL units.
fn sample(avcc: &[u8], description: &CfOwned) -> Result<CfOwned> {
    let len = avcc.len();
    let mut block = std::ptr::null_mut();
    // SAFETY: CoreMedia allocates `len` bytes; `block` is a valid
    // out-pointer.
    let s = unsafe {
        CMBlockBufferCreateWithMemoryBlock(
            std::ptr::null(),
            std::ptr::null_mut(),
            len,
            std::ptr::null(),
            std::ptr::null(),
            0,
            len,
            kCMBlockBufferAssureMemoryNowFlag,
            &mut block,
        )
    };
    // SAFETY: on success `block` holds a +1 reference.
    let block = match unsafe { CfOwned::from_create(block.cast_const()) } {
        Some(b) if s == 0 => b,
        _ => return Err(status("CMBlockBufferCreateWithMemoryBlock", s)),
    };
    // SAFETY: `block` holds `len` bytes, and `avcc` is `len` bytes.
    let s = unsafe { CMBlockBufferReplaceDataBytes(avcc.as_ptr().cast(), block.as_ptr(), 0, len) };
    if s != 0 {
        return Err(status("CMBlockBufferReplaceDataBytes", s));
    }
    let mut sample = std::ptr::null_mut();
    // SAFETY: a live block buffer and format description, one sample of
    // `len` bytes with no timing, and a valid out-pointer.
    let s = unsafe {
        CMSampleBufferCreateReady(
            std::ptr::null(),
            block.as_ptr(),
            description.as_ptr(),
            1,
            0,
            std::ptr::null(),
            1,
            &len,
            &mut sample,
        )
    };
    // SAFETY: on success `sample` holds a +1 reference.
    match unsafe { CfOwned::from_create(sample.cast_const()) } {
        Some(sample) if s == 0 => Ok(sample),
        _ => Err(status("CMSampleBufferCreateReady", s)),
    }
}
