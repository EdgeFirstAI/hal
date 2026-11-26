// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! The compression session, its output callback and the state the callback
//! shares with the encoder.

use std::collections::VecDeque;
use std::ffi::c_void;
use std::sync::{Arc, Mutex, MutexGuard};

use edgefirst_tensor::Colorimetry;

use super::{DEVICE, MAX_FRAME_DELAY};
use crate::video::h264::{avcc_to_annex_b, with_parameter_sets, AccessUnit, NAL_PPS, NAL_SPS};
use crate::video::h264_syntax::Pps;
use crate::video::vt::color::to_corevideo;
use crate::video::vt::ffi::*;
use crate::video::vt::status;
use crate::video::{EncodedFrame, EncoderConfig, H264Level, H264Profile};
use crate::{CodecError, Result};

/// `profile_idc` of the Baseline profile.
const BASELINE: u8 = 66;
/// Ticks per second of the presentation times given to the session.
pub(super) const TIMESCALE: i32 = 90_000;
/// The most bytes in any one second, as a multiple of the target bit rate.
const PEAK_RATE: f64 = 1.25;

/// A source read in place: its pixel buffer and one use count on the
/// buffer's IOSurface, so `IOSurfaceIsInUse` is true while this lives. Drop
/// lowers the use count, then releases the pixel buffer.
#[derive(Debug)]
pub(super) struct HeldSource(CfOwned);

impl HeldSource {
    /// Holds `buffer`, a pixel buffer over an IOSurface, and raises the
    /// surface's use count.
    pub(super) fn new(buffer: CfOwned) -> Result<Self> {
        // SAFETY: `buffer` is a live pixel buffer; the surface is borrowed
        // (+0) while it lives.
        let surface = unsafe { CVPixelBufferGetIOSurface(buffer.as_ptr()) };
        if surface.is_null() {
            return Err(CodecError::InvalidInput(
                "the source's pixel buffer has no IOSurface".into(),
            ));
        }
        // SAFETY: `buffer` keeps `surface` live.
        unsafe { IOSurfaceIncrementUseCount(surface) };
        Ok(Self(buffer))
    }
}

impl Drop for HeldSource {
    fn drop(&mut self) {
        // SAFETY: the pixel buffer is live and keeps its surface, whose use
        // count `new` raised once.
        unsafe { IOSurfaceDecrementUseCount(CVPixelBufferGetIOSurface(self.0.as_ptr())) };
    }
}

/// A source submitted and not yet out.
#[derive(Debug)]
pub(super) struct Frame {
    seq: u64,
    pts: u64,
    source: Option<HeldSource>,
}

/// Sources submitted and not yet out, in submit order.
#[derive(Debug, Default)]
pub(super) struct InFlight {
    next: u64,
    pub(super) frames: VecDeque<Frame>,
}

impl InFlight {
    /// Records a source with `pts`, holding `source` until the source is
    /// out, and returns its sequence number.
    pub(super) fn submit(&mut self, pts: u64, source: Option<HeldSource>) -> u64 {
        let seq = self.next;
        self.next += 1;
        self.frames.push_back(Frame { seq, pts, source });
        seq
    }

    /// The PTS of source `seq`, whose access unit is out, releasing what is
    /// held for it; `None` when `seq` is not in flight.
    fn emitted(&mut self, seq: u64) -> Option<u64> {
        let at = self.frames.iter().position(|f| f.seq == seq)?;
        self.frames.remove(at).map(|f| f.pts)
    }

    /// Forgets source `seq`, which has no access unit.
    pub(super) fn dropped(&mut self, seq: u64) {
        self.emitted(seq);
    }

    /// Forgets every source. Sequence numbers keep counting.
    pub(super) fn clear(&mut self) {
        self.frames.clear();
    }

    fn contains(&self, seq: u64) -> bool {
        self.frames.iter().any(|f| f.seq == seq)
    }

    /// The sources read in place that are held.
    pub(super) fn held(&self) -> usize {
        self.frames.iter().filter(|f| f.source.is_some()).count()
    }

    /// Whether the source of the frame with `pts` is read in place and held.
    #[cfg(test)]
    pub(super) fn holds(&self, pts: u64) -> bool {
        self.frames
            .iter()
            .any(|f| f.pts == pts && f.source.is_some())
    }
}

/// What the output callback and the encoder share.
#[derive(Default)]
pub(super) struct State {
    pub(super) in_flight: InFlight,
    pub(super) ready: VecDeque<EncodedFrame>,
    /// The first error the callback reported since the last restart.
    pub(super) error: Option<CodecError>,
    /// The next access unit is the first since the encoder restarted.
    pub(super) first: bool,
    /// A frame was dropped before the first access unit since the restart.
    pub(super) rekey: bool,
    pub(super) dropped: u64,
}

impl State {
    /// Queues the access unit of frame `seq`, with the parameter sets on the
    /// first since the restart and, with `repeat_headers`, on every IDR.
    /// Until that first one, an access unit that is not a key frame is
    /// dropped.
    fn queue(&mut self, seq: u64, sample: Sample, repeat_headers: bool) {
        let Some(pts) = self.in_flight.emitted(seq) else {
            return;
        };
        if self.first && !sample.keyframe {
            self.dropped += 1;
            self.rekey = true;
            log::debug!("{DEVICE}: frame {seq} dropped: not a key frame at the stream's start");
            return;
        }
        let first = std::mem::take(&mut self.first);
        let idr = AccessUnit::parse(&sample.data).idr;
        let data = if first || (repeat_headers && idr) {
            with_parameter_sets(&sample.data, &sample.parameter_sets)
        } else {
            sample.data
        };
        self.ready.push_back(EncodedFrame {
            data,
            keyframe: sample.keyframe,
            pts,
        });
    }

    /// Forgets frame `seq`, which the session dropped.
    fn drop_frame(&mut self, seq: u64) {
        self.in_flight.dropped(seq);
        self.dropped += 1;
        self.rekey |= self.first;
        log::debug!("{DEVICE}: frame {seq} dropped");
    }

    /// Forgets frame `seq` and keeps `error`, unless an earlier one is kept.
    pub(super) fn fail(&mut self, seq: u64, error: CodecError) {
        self.in_flight.dropped(seq);
        self.error.get_or_insert(error);
    }
}

/// State the output callback shares with the encoder, from any thread.
pub(super) struct Shared {
    pub(super) repeat_headers: bool,
    /// The stream is Constrained Baseline.
    pub(super) constrained: bool,
    pub(super) state: Mutex<State>,
}

impl Shared {
    pub(super) fn lock(&self) -> MutexGuard<'_, State> {
        self.state.lock().unwrap_or_else(|e| e.into_inner())
    }
}

/// One encoded sample, in Annex B.
struct Sample {
    data: Vec<u8>,
    parameter_sets: Vec<u8>,
    keyframe: bool,
}

/// The session's output callback.
///
/// # Safety
///
/// `ref_con` is the `Shared` of a live [`Session`], and `sample` is null or
/// a live `CMSampleBuffer`.
unsafe extern "C" fn on_output(
    ref_con: *mut c_void,
    source_frame_ref_con: *mut c_void,
    status_code: OSStatus,
    flags: VTEncodeInfoFlags,
    sample: CMSampleBufferRef,
) {
    // SAFETY: the session's `Arc<Shared>` keeps `ref_con` live until the
    // session is invalidated, after which the callback no longer runs.
    let shared = unsafe { &*ref_con.cast_const().cast::<Shared>() };
    let seq = source_frame_ref_con as u64;
    let produced = status_code == 0 && flags & kVTEncodeInfo_FrameDropped == 0 && !sample.is_null();
    // SAFETY: `sample` is a live sample buffer.
    let sample = produced.then(|| unsafe { read_sample(sample, shared.constrained) });
    let mut state = shared.lock();
    if !state.in_flight.contains(seq) {
        log::debug!("{DEVICE}: output for frame {seq}, which is no longer in flight");
        return;
    }
    match sample {
        _ if status_code != 0 => {
            state.fail(seq, status("VTCompressionOutputCallback", status_code));
        }
        None => state.drop_frame(seq),
        Some(Err(e)) => state.fail(seq, e),
        Some(Ok(sample)) => state.queue(seq, sample, shared.repeat_headers),
    }
}

/// The access unit, parameter sets and sync flag of an encoded sample, its
/// SPS marked Constrained Baseline when `constrained` is set.
///
/// # Safety
///
/// `sample` is a live `CMSampleBuffer`.
unsafe fn read_sample(sample: CMSampleBufferRef, constrained: bool) -> Result<Sample> {
    let invalid = |what: &str| CodecError::InvalidData(format!("an encoded sample {what}"));
    // SAFETY: `sample` is live; the block buffer and format description are
    // borrowed (+0) while it is.
    let (block, description) = unsafe {
        (
            CMSampleBufferGetDataBuffer(sample),
            CMSampleBufferGetFormatDescription(sample),
        )
    };
    if block.is_null() || description.is_null() {
        return Err(invalid("has no data or format description"));
    }
    // SAFETY: `block` is a live block buffer.
    let len = unsafe { CMBlockBufferGetDataLength(block) };
    let mut avcc = vec![0u8; len];
    // SAFETY: `avcc` is `len` writable bytes.
    let s = unsafe { CMBlockBufferCopyDataBytes(block, 0, len, avcc.as_mut_ptr().cast()) };
    if s != 0 {
        return Err(status("CMBlockBufferCopyDataBytes", s));
    }
    let mut sets = Vec::new();
    let (mut count, mut length_size) = (0usize, 0i32);
    let mut index = 0;
    loop {
        let (mut ptr, mut size) = (std::ptr::null(), 0usize);
        // SAFETY: a live H.264 format description and valid out-pointers.
        let s = unsafe {
            CMVideoFormatDescriptionGetH264ParameterSetAtIndex(
                description,
                index,
                &mut ptr,
                &mut size,
                &mut count,
                &mut length_size,
            )
        };
        if s != 0 {
            return Err(status(
                "CMVideoFormatDescriptionGetH264ParameterSetAtIndex",
                s,
            ));
        }
        // SAFETY: the description holds `size` bytes at `ptr` while it lives.
        sets.push(unsafe { std::slice::from_raw_parts(ptr, size) }.to_vec());
        index += 1;
        if index >= count {
            break;
        }
    }
    if constrained {
        constrain(&mut sets)?;
    }
    let parameter_sets = sets
        .iter()
        .flat_map(|set| [&[0, 0, 0, 1][..], set])
        .flatten()
        .copied()
        .collect();
    let data = avcc_to_annex_b(&avcc, length_size as usize).map_err(CodecError::InvalidData)?;
    // SAFETY: `sample` is live.
    let keyframe = unsafe { !not_sync(sample) };
    Ok(Sample {
        data,
        parameter_sets,
        keyframe,
    })
}

/// Sets `constraint_set0_flag` and `constraint_set1_flag` in the Baseline
/// SPS among the NAL units `sets`, which marks it Constrained Baseline.
///
/// # Errors
///
/// [`CodecError::InvalidData`] when the SPS is not Baseline, or a PPS uses
/// slice groups or redundant pictures.
fn constrain(sets: &mut [Vec<u8>]) -> Result<()> {
    for nal in sets.iter_mut() {
        match nal.first().map(|h| h & 0x1f) {
            Some(NAL_PPS) => {
                let pps = Pps::parse(nal).map_err(CodecError::InvalidData)?;
                if pps.slice_groups != 1 || pps.redundant_pic_cnt_present {
                    return Err(CodecError::InvalidData(
                        "the encoder's PPS uses slice groups or redundant pictures, \
                         which Constrained Baseline excludes"
                            .into(),
                    ));
                }
            }
            Some(NAL_SPS) if nal.len() > 3 && nal[1] == BASELINE => nal[2] |= 0xc0,
            Some(NAL_SPS) => {
                return Err(CodecError::InvalidData(
                    "the encoder's SPS is not Baseline".into(),
                ))
            }
            _ => {}
        }
    }
    Ok(())
}

/// Whether `sample` carries the `NotSync` attachment.
///
/// # Safety
///
/// `sample` is a live `CMSampleBuffer`.
unsafe fn not_sync(sample: CMSampleBufferRef) -> bool {
    // SAFETY: `sample` is live, the array and its dictionary are borrowed
    // (+0), and the key is a CoreMedia constant.
    unsafe {
        let attachments = CMSampleBufferGetSampleAttachmentsArray(sample, 0);
        if attachments.is_null() || CFArrayGetCount(attachments) == 0 {
            return false;
        }
        let first = CFArrayGetValueAtIndex(attachments, 0);
        let value = CFDictionaryGetValue(first, kCMSampleAttachmentKey_NotSync);
        !value.is_null() && CFBooleanGetValue(value) != 0
    }
}

/// The `kVTProfileLevel_H264_*` constant of `profile` at `level`, its
/// automatic level for `None`, or `None` when VideoToolbox has no constant
/// for the pair.
fn profile_level(profile: H264Profile, level: Option<H264Level>) -> Option<CFStringRef> {
    use H264Level as L;
    // SAFETY: reads immutable constants VideoToolbox defines.
    unsafe {
        Some(match (profile, level) {
            (H264Profile::Baseline | H264Profile::ConstrainedBaseline, level) => match level {
                None => kVTProfileLevel_H264_Baseline_AutoLevel,
                Some(L::L1_3) => kVTProfileLevel_H264_Baseline_1_3,
                Some(L::L3_0) => kVTProfileLevel_H264_Baseline_3_0,
                Some(L::L3_1) => kVTProfileLevel_H264_Baseline_3_1,
                Some(L::L3_2) => kVTProfileLevel_H264_Baseline_3_2,
                Some(L::L4_0) => kVTProfileLevel_H264_Baseline_4_0,
                Some(L::L4_1) => kVTProfileLevel_H264_Baseline_4_1,
                Some(L::L4_2) => kVTProfileLevel_H264_Baseline_4_2,
                Some(L::L5_0) => kVTProfileLevel_H264_Baseline_5_0,
                Some(L::L5_1) => kVTProfileLevel_H264_Baseline_5_1,
                Some(L::L5_2) => kVTProfileLevel_H264_Baseline_5_2,
                Some(_) => return None,
            },
            (H264Profile::Main, level) => match level {
                None => kVTProfileLevel_H264_Main_AutoLevel,
                Some(L::L3_0) => kVTProfileLevel_H264_Main_3_0,
                Some(L::L3_1) => kVTProfileLevel_H264_Main_3_1,
                Some(L::L3_2) => kVTProfileLevel_H264_Main_3_2,
                Some(L::L4_0) => kVTProfileLevel_H264_Main_4_0,
                Some(L::L4_1) => kVTProfileLevel_H264_Main_4_1,
                Some(L::L4_2) => kVTProfileLevel_H264_Main_4_2,
                Some(L::L5_0) => kVTProfileLevel_H264_Main_5_0,
                Some(L::L5_1) => kVTProfileLevel_H264_Main_5_1,
                Some(L::L5_2) => kVTProfileLevel_H264_Main_5_2,
                Some(_) => return None,
            },
            (H264Profile::High, level) => match level {
                None => kVTProfileLevel_H264_High_AutoLevel,
                Some(L::L3_0) => kVTProfileLevel_H264_High_3_0,
                Some(L::L3_1) => kVTProfileLevel_H264_High_3_1,
                Some(L::L3_2) => kVTProfileLevel_H264_High_3_2,
                Some(L::L4_0) => kVTProfileLevel_H264_High_4_0,
                Some(L::L4_1) => kVTProfileLevel_H264_High_4_1,
                Some(L::L4_2) => kVTProfileLevel_H264_High_4_2,
                Some(L::L5_0) => kVTProfileLevel_H264_High_5_0,
                Some(L::L5_1) => kVTProfileLevel_H264_High_5_1,
                Some(L::L5_2) => kVTProfileLevel_H264_High_5_2,
                Some(_) => return None,
            },
        })
    }
}

/// The profile and level constant for `config`: the target level's, or the
/// automatic level when VideoToolbox has no constant for a level the caller
/// did not set.
fn config_profile_level(config: &EncoderConfig) -> Result<CFStringRef> {
    let profile = config.target_profile();
    match profile_level(profile, config.target_level()) {
        Some(key) => Ok(key),
        None if config.level.is_none() => {
            Ok(profile_level(profile, None).expect("every profile has an automatic level"))
        }
        None => Err(CodecError::InvalidConfig(format!(
            "{DEVICE} has no H.264 {profile:?} level {:?}",
            config.level
        ))),
    }
}

/// The error for the Core Foundation allocation `op` failing.
pub(super) fn out_of_memory(op: &str) -> CodecError {
    CodecError::Device {
        op: op.into(),
        source: std::io::Error::other("out of memory"),
    }
}

/// A compression session, completed and invalidated on drop.
pub(super) struct Session {
    session: CfOwned,
    /// The pixel format of the session's pool.
    pub(super) fourcc: u32,
    /// The colour the session writes into the stream.
    pub(super) colour: Option<Colorimetry>,
    /// No frame has been submitted.
    pub(super) fresh: bool,
    pub(super) hardware: bool,
    pub(super) encoder_id: Option<String>,
    /// The session holds at most [`MAX_FRAME_DELAY`] frames before it
    /// emits one.
    pub(super) bounded_delay: bool,
    invalidated: bool,
    _shared: Arc<Shared>,
}

#[cfg(test)]
thread_local! {
    /// Sessions this thread creates treat the frame delay as refused.
    pub(super) static REFUSE_FRAME_DELAY: std::cell::Cell<bool> =
        const { std::cell::Cell::new(false) };
}

/// Whether an unbounded frame delay has been logged at info level.
static UNBOUNDED_LOGGED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

impl Session {
    /// A prepared session for `config` whose pool holds `fourcc` buffers,
    /// describing `colour` in the stream.
    pub(super) fn new(
        config: &EncoderConfig,
        fourcc: u32,
        colour: Option<&Colorimetry>,
        shared: &Arc<Shared>,
    ) -> Result<Self> {
        let alloc = || out_of_memory("CFDictionaryCreate");
        let number = |v: u32| {
            cf_number_i32(i32::try_from(v).unwrap_or(i32::MAX))
                .ok_or_else(|| out_of_memory("CFNumberCreate"))
        };
        let (format, width, height) = (
            number(fourcc)?,
            number(config.width)?,
            number(config.height)?,
        );
        // SAFETY: the keys are CoreVideo constants and the values live CF
        // objects held across the call.
        let attributes = unsafe {
            let surface = cf_dictionary(&[]).ok_or_else(alloc)?;
            cf_dictionary(&[
                (kCVPixelBufferPixelFormatTypeKey, format.as_ptr()),
                (kCVPixelBufferWidthKey, width.as_ptr()),
                (kCVPixelBufferHeightKey, height.as_ptr()),
                (kCVPixelBufferIOSurfacePropertiesKey, surface.as_ptr()),
            ])
            .ok_or_else(alloc)?
        };
        #[cfg(target_os = "macos")]
        // SAFETY: the key is a VideoToolbox constant and the value a CF
        // constant.
        let specification = unsafe {
            Some(
                cf_dictionary(&[(
                    kVTVideoEncoderSpecification_EnableHardwareAcceleratedVideoEncoder,
                    cf_bool(true),
                )])
                .ok_or_else(alloc)?,
            )
        };
        #[cfg(not(target_os = "macos"))]
        let specification: Option<CfOwned> = None;
        let mut session = std::ptr::null_mut();
        let (w, h) = (config.width as i32, config.height as i32);
        // SAFETY: the dictionaries are live, the callback's `ref_con` is
        // kept live by `_shared` until the session is invalidated, and
        // `session` is a valid out-pointer.
        let s = unsafe {
            VTCompressionSessionCreate(
                std::ptr::null(),
                w,
                h,
                kCMVideoCodecType_H264,
                specification
                    .as_ref()
                    .map_or(std::ptr::null(), |s| s.as_ptr()),
                attributes.as_ptr(),
                std::ptr::null(),
                Some(on_output),
                Arc::as_ptr(shared).cast_mut().cast(),
                &mut session,
            )
        };
        // SAFETY: on success `session` holds a +1 reference.
        let session = match unsafe { CfOwned::from_create(session) } {
            Some(session) if s == 0 => session,
            _ => return Err(status("VTCompressionSessionCreate", s)),
        };
        let mut session = Self {
            session,
            fourcc,
            colour: None,
            fresh: true,
            hardware: false,
            encoder_id: None,
            bounded_delay: false,
            invalidated: false,
            _shared: Arc::clone(shared),
        };
        session.configure(config)?;
        let delay_set = session.set_frame_delay();
        if let Some(colour) = colour {
            session.describe(colour)?;
        }
        // SAFETY: a live session.
        let s = unsafe { VTCompressionSessionPrepareToEncodeFrames(session.session.as_ptr()) };
        if s != 0 {
            return Err(status("VTCompressionSessionPrepareToEncodeFrames", s));
        }
        session.hardware = session.uses_hardware();
        // SAFETY: reads a VideoToolbox constant.
        session.encoder_id = session
            .copy(unsafe { kVTCompressionPropertyKey_EncoderID })
            // SAFETY: `id` is a live CF object.
            .and_then(|id| unsafe { string_of(id.as_ptr()) });
        session.bounded_delay = session.bounded(delay_set);
        Ok(session)
    }

    fn set(&self, key: CFStringRef, value: CFTypeRef) -> std::result::Result<(), OSStatus> {
        // SAFETY: a live session, and a key and value that are live CF
        // objects.
        match unsafe { VTSessionSetProperty(self.session.as_ptr(), key, value) } {
            0 => Ok(()),
            s => Err(s),
        }
    }

    /// The value of property `key`, when the session has one.
    fn copy(&self, key: CFStringRef) -> Option<CfOwned> {
        let mut value: CFTypeRef = std::ptr::null();
        // SAFETY: a live session and a valid out-pointer.
        let s = unsafe {
            VTSessionCopyProperty(
                self.session.as_ptr(),
                key,
                std::ptr::null(),
                (&mut value as *mut CFTypeRef).cast(),
            )
        };
        // SAFETY: on success `value` holds a +1 reference.
        (s == 0).then(|| unsafe { CfOwned::from_create(value) })?
    }

    /// Applies the rate, GOP, profile, frame rate and frame order properties
    /// of `config`.
    fn configure(&self, config: &EncoderConfig) -> Result<()> {
        let rejected = |what: &str, s| {
            CodecError::InvalidConfig(format!("{DEVICE} rejects {what}: OSStatus {s}"))
        };
        let alloc = || out_of_memory("CFNumberCreate");
        let bps = config.target_bitrate();
        let peak = ((f64::from(bps) * PEAK_RATE / 8.0) as i64).min(i64::from(i32::MAX)) as i32;
        let (average, peak, second) = (
            cf_number_i32(i32::try_from(bps).unwrap_or(i32::MAX)).ok_or_else(alloc)?,
            cf_number_i32(peak).ok_or_else(alloc)?,
            cf_number_f64(1.0).ok_or_else(alloc)?,
        );
        // SAFETY: the values are live CF objects.
        let limits = unsafe { cf_array(&[peak.as_ptr(), second.as_ptr()]) }.ok_or_else(alloc)?;
        let rate = cf_number_f64(config.frame_rate).ok_or_else(alloc)?;
        let profile = config_profile_level(config)?;
        // SAFETY: reads VideoToolbox's key constants.
        let (required, optional) = unsafe {
            (
                [
                    (
                        kVTCompressionPropertyKey_AllowFrameReordering,
                        cf_bool(false),
                        "no B-frames",
                    ),
                    (
                        kVTCompressionPropertyKey_ProfileLevel,
                        profile,
                        "the profile and level",
                    ),
                    (
                        kVTCompressionPropertyKey_AverageBitRate,
                        average.as_ptr(),
                        "the bit rate",
                    ),
                ],
                [
                    (
                        kVTCompressionPropertyKey_DataRateLimits,
                        limits.as_ptr(),
                        "a peak rate",
                    ),
                    (
                        kVTCompressionPropertyKey_ExpectedFrameRate,
                        rate.as_ptr(),
                        "the frame rate",
                    ),
                ],
            )
        };
        for (key, value, what) in required {
            self.set(key, value).map_err(|s| rejected(what, s))?;
        }
        for (key, value, what) in optional {
            if let Err(s) = self.set(key, value) {
                log::debug!("{DEVICE}: {what} not set: OSStatus {s}");
            }
        }
        if let Some(gop) = config.gop {
            let gop = cf_number_i32(i32::try_from(gop).unwrap_or(i32::MAX)).ok_or_else(alloc)?;
            // SAFETY: reads a VideoToolbox constant.
            let key = unsafe { kVTCompressionPropertyKey_MaxKeyFrameInterval };
            self.set(key, gop.as_ptr())
                .map_err(|s| rejected("the GOP", s))?;
        }
        Ok(())
    }

    /// Asks the session to hold at most [`MAX_FRAME_DELAY`] frames before
    /// it emits one.
    fn set_frame_delay(&self) -> std::result::Result<(), OSStatus> {
        #[cfg(test)]
        if REFUSE_FRAME_DELAY.with(std::cell::Cell::get) {
            return Err(kVTPropertyNotSupportedErr);
        }
        let delay = cf_number_i32(MAX_FRAME_DELAY as i32).ok_or(0)?;
        // SAFETY: reads a VideoToolbox constant.
        let key = unsafe { kVTCompressionPropertyKey_MaxFrameDelayCount };
        self.set(key, delay.as_ptr())
    }

    /// The session's frame delay, or `None` when it cannot be read.
    fn frame_delay(&self) -> Option<i32> {
        #[cfg(test)]
        if REFUSE_FRAME_DELAY.with(std::cell::Cell::get) {
            return None;
        }
        // SAFETY: reads a VideoToolbox constant.
        let value = self.copy(unsafe { kVTCompressionPropertyKey_MaxFrameDelayCount })?;
        let mut delay = 0i32;
        // SAFETY: `value` is a live CF object, read as a number only when it
        // is one, into a valid out-pointer.
        let read = unsafe {
            CFGetTypeID(value.as_ptr()) == CFNumberGetTypeID()
                && CFNumberGetValue(
                    value.as_ptr(),
                    kCFNumberSInt32Type,
                    (&mut delay as *mut i32).cast(),
                ) != 0
        };
        read.then_some(delay)
    }

    /// Whether the session holds at most [`MAX_FRAME_DELAY`] frames before
    /// it emits one: `set` succeeded, or the delay reads back within the
    /// bound. Logs an unbounded session once at info level, then at debug.
    fn bounded(&self, set: std::result::Result<(), OSStatus>) -> bool {
        let read = self.frame_delay();
        let bounded =
            set.is_ok() || read.is_some_and(|d| (0..=MAX_FRAME_DELAY as i32).contains(&d));
        let id = self.encoder_id.as_deref().unwrap_or("an unnamed encoder");
        let status = set.err().unwrap_or(0);
        if bounded {
            log::debug!("{DEVICE}: {id} frame delay {read:?} (setting it: OSStatus {status})");
            return true;
        }
        let first = !UNBOUNDED_LOGGED.swap(true, std::sync::atomic::Ordering::Relaxed);
        let level = if first {
            log::Level::Info
        } else {
            log::Level::Debug
        };
        log::log!(
            level,
            "{DEVICE}: {id} frame delay {read:?} is not at most {MAX_FRAME_DELAY} \
             (setting it: OSStatus {status}); CoreVideo sources are copied"
        );
        false
    }

    /// Sets the stream's colour primaries, transfer function and matrix to
    /// those of `colour`.
    ///
    /// # Errors
    ///
    /// [`CodecError::InvalidInput`] when a set axis of `colour` has no
    /// CoreVideo value on this platform.
    pub(super) fn describe(&mut self, colour: &Colorimetry) -> Result<()> {
        let cv = to_corevideo(colour)?;
        self.colour = Some(*colour);
        // SAFETY: reads VideoToolbox's key constants.
        let keys = unsafe {
            [
                (kVTCompressionPropertyKey_ColorPrimaries, cv.primaries),
                (kVTCompressionPropertyKey_TransferFunction, cv.transfer),
                (kVTCompressionPropertyKey_YCbCrMatrix, cv.matrix),
            ]
        };
        for (key, value) in keys {
            if let Some(value) = value {
                if let Err(s) = self.set(key, value) {
                    log::debug!("{DEVICE}: colour property not set: OSStatus {s}");
                }
            }
        }
        Ok(())
    }

    #[cfg(target_os = "macos")]
    fn uses_hardware(&self) -> bool {
        // SAFETY: reads a VideoToolbox constant.
        let key = unsafe { kVTCompressionPropertyKey_UsingHardwareAcceleratedVideoEncoder };
        self.copy(key).is_some_and(|v| {
            // SAFETY: `v` is a live CF object, read as a boolean only when it
            // is one.
            unsafe {
                CFGetTypeID(v.as_ptr()) == CFBooleanGetTypeID()
                    && CFBooleanGetValue(v.as_ptr()) != 0
            }
        })
    }

    #[cfg(not(target_os = "macos"))]
    fn uses_hardware(&self) -> bool {
        cfg!(not(target_abi = "sim"))
    }

    /// A pixel buffer from the session's pool.
    pub(super) fn pixel_buffer(&self) -> Result<CfOwned> {
        // SAFETY: a live session; the pool is borrowed (+0) while it lives.
        let pool = unsafe { VTCompressionSessionGetPixelBufferPool(self.session.as_ptr()) };
        if pool.is_null() {
            return Err(CodecError::Device {
                op: "VTCompressionSessionGetPixelBufferPool".into(),
                source: std::io::Error::other("the session has no pixel buffer pool"),
            });
        }
        let mut buffer = std::ptr::null_mut();
        // SAFETY: a live pool and a valid out-pointer.
        let s = unsafe { CVPixelBufferPoolCreatePixelBuffer(std::ptr::null(), pool, &mut buffer) };
        // SAFETY: on success `buffer` holds a +1 reference.
        match unsafe { CfOwned::from_create(buffer) } {
            Some(buffer) if s == 0 => Ok(buffer),
            _ => Err(status("CVPixelBufferPoolCreatePixelBuffer", s)),
        }
    }

    /// Another reference to the session.
    pub(super) fn handle(&self) -> CfOwned {
        self.session.clone()
    }

    /// Waits until every submitted frame has reached the callback.
    pub(super) fn complete(&self) -> Result<()> {
        // SAFETY: a live session; `kCMTimeInvalid` completes every frame.
        match unsafe { VTCompressionSessionCompleteFrames(self.session.as_ptr(), kCMTimeInvalid) } {
            0 => Ok(()),
            s => Err(status("VTCompressionSessionCompleteFrames", s)),
        }
    }

    pub(super) fn invalidate(&mut self) {
        if !self.invalidated {
            // SAFETY: a live session, invalidated once.
            unsafe { VTCompressionSessionInvalidate(self.session.as_ptr()) };
            self.invalidated = true;
        }
    }
}

impl Drop for Session {
    fn drop(&mut self) {
        if !self.invalidated {
            if let Err(e) = self.complete() {
                log::debug!("{DEVICE}: {e}");
            }
        }
        self.invalidate();
    }
}

#[cfg(test)]
mod tests {
    use edgefirst_tensor::PixelFormat;

    use super::*;

    const SETS: &[u8] = &[
        0, 0, 0, 1, 0x67, 0x42, 0x00, 0x1e, 0, 0, 0, 1, 0x68, 0xce, 0x3c, 0x80,
    ];
    const IDR: &[u8] = &[0, 0, 0, 1, 0x65, 0x88, 0x84, 0x00];
    const P_SLICE: &[u8] = &[0, 0, 0, 1, 0x41, 0x9a, 0x02];

    fn sample(data: &[u8], keyframe: bool) -> Sample {
        Sample {
            data: data.to_vec(),
            parameter_sets: SETS.to_vec(),
            keyframe,
        }
    }

    #[test]
    fn the_stream_starts_at_its_first_key_frame() {
        let mut state = State {
            first: true,
            ..State::default()
        };
        let seqs: Vec<u64> = (0..3)
            .map(|pts| state.in_flight.submit(pts, None))
            .collect();
        state.queue(seqs[0], sample(P_SLICE, false), false);
        assert!(state.ready.is_empty());
        assert_eq!((state.dropped, state.rekey), (1, true));
        state.queue(seqs[1], sample(IDR, true), false);
        state.queue(seqs[2], sample(P_SLICE, false), false);
        let out: Vec<EncodedFrame> = state.ready.drain(..).collect();
        assert_eq!(out.len(), 2);
        assert_eq!((out[0].pts, out[0].keyframe), (1, true));
        assert_eq!(out[0].data, [SETS, IDR].concat());
        assert_eq!((out[1].pts, out[1].data.as_slice()), (2, P_SLICE));
        assert!(!state.first);
    }

    #[test]
    fn constrained_baseline_sets_constraint_flags_0_and_1() {
        let mut sets = vec![
            vec![0x67, 0x42, 0x00, 0x1e, 0xab],
            vec![0x68, 0xce, 0x3c, 0x80],
        ];
        constrain(&mut sets).unwrap();
        assert_eq!(sets[0], [0x67, 0x42, 0xc0, 0x1e, 0xab]);
        assert_eq!(sets[1], [0x68, 0xce, 0x3c, 0x80]);
        let refused = [
            vec![vec![0x67, 0x64, 0x00, 0x28]],
            vec![vec![0x68, 0xc4, 0x57, 0x1e, 0x40]],
            vec![vec![0x68, 0xce, 0x3d, 0x80]],
        ];
        for mut sets in refused {
            assert!(
                matches!(constrain(&mut sets), Err(CodecError::InvalidData(_))),
                "{sets:02x?}"
            );
        }
    }

    #[test]
    fn a_dropped_frame_does_not_shift_later_timestamps() {
        let mut f = InFlight::default();
        let s: Vec<u64> = [10, 20, 30].iter().map(|&p| f.submit(p, None)).collect();
        f.dropped(s[1]);
        assert_eq!((f.emitted(s[0]), f.emitted(s[2])), (Some(10), Some(30)));
    }

    #[test]
    fn cleared_frames_are_not_emitted_and_numbers_keep_counting() {
        let mut f = InFlight::default();
        let first = f.submit(1, None);
        f.clear();
        assert_eq!(f.emitted(first), None);
        let next = f.submit(2, None);
        assert!(next > first);
        assert_eq!(f.emitted(next), Some(2));
        assert_eq!(f.emitted(next), None);
    }

    #[test]
    fn profiles_and_levels_map_to_videotoolbox_constants() {
        use H264Level as L;
        use H264Profile as P;
        // SAFETY: reads VideoToolbox constants.
        unsafe {
            assert_eq!(
                profile_level(P::High, Some(L::L4_0)),
                Some(kVTProfileLevel_H264_High_4_0)
            );
            assert_eq!(
                profile_level(P::Main, Some(L::L5_2)),
                Some(kVTProfileLevel_H264_Main_5_2)
            );
            assert_eq!(
                profile_level(P::ConstrainedBaseline, Some(L::L1_3)),
                Some(kVTProfileLevel_H264_Baseline_1_3)
            );
            assert_eq!(
                profile_level(P::Baseline, None),
                Some(kVTProfileLevel_H264_Baseline_AutoLevel)
            );
        }
        assert_eq!(profile_level(P::High, Some(L::L2_1)), None);
        assert_eq!(profile_level(P::Main, Some(L::L6_0)), None);
    }

    #[test]
    fn a_level_without_a_constant_is_automatic_unless_requested() {
        let mut config = EncoderConfig::h264(320, 240, PixelFormat::Nv12, 15.0);
        assert_eq!(config.target_level(), Some(H264Level::L2_0));
        // SAFETY: reads a VideoToolbox constant.
        let auto = unsafe { kVTProfileLevel_H264_High_AutoLevel };
        assert_eq!(config_profile_level(&config).unwrap(), auto);
        config.level = Some(H264Level::L2_0);
        assert!(matches!(
            config_profile_level(&config),
            Err(CodecError::InvalidConfig(_))
        ));
    }
}
