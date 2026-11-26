// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! H.264 encoding with a VideoToolbox compression session.
//!
//! A source that is a whole CoreVideo surface of the configured format and
//! size is read in place through a pixel buffer over that surface, held with
//! the surface's use count raised until its frame is out of the session.
//! Every other source is copied into a pixel buffer from the session's pool.
//! `encode` queues each frame to a thread that submits frames to the session
//! in order, with the frame's sequence number as its ref-con. The output
//! callback, which may run on a VideoToolbox thread, turns each sample into
//! an Annex B access unit and queues it with the PTS the shared state holds
//! for that sequence number. `encode` never waits for the session or the
//! submission thread: it returns the oldest queued access unit, if any.

mod session;
mod submit;

use std::sync::{Arc, Mutex};

use edgefirst_tensor::{ColorRange, Colorimetry, PixelFormat, TensorDyn};

use self::session::{HeldSource, Session, Shared, State, TIMESCALE};
use self::submit::{Job, Submitter};
use super::color::{from_attachments, surface_range, to_corevideo};
use super::ffi::*;
use super::{
    chroma_offset, h264_encoders, status, ListedEncoder, FOURCC_420F, FOURCC_420V, FOURCC_YUVF,
    FOURCC_YUVS,
};
use crate::video::copy::{copy_source, Plane};
use crate::video::{
    check_source, EncodedFrame, EncoderBackend, EncoderConfig, EncoderInfo, FrameOptions,
    H264Profile,
};
use crate::{CodecError, Engine, Result, Transfer};

const DEVICE: &str = "VideoToolbox H.264 encoder";
/// The most sources read in place that the encoder holds at once.
pub(crate) const MAX_IN_FLIGHT: usize = 8;
const _: () = assert!(MAX_IN_FLIGHT >= 8);
/// The bound [`crate::video::VideoEncoder::encode`] documents.
const _: () = assert!(MAX_IN_FLIGHT == 8);
/// The most frames a session holds before it emits one.
pub(crate) const MAX_FRAME_DELAY: usize = 4;
const _: () = assert!(MAX_FRAME_DELAY < MAX_IN_FLIGHT);
/// The most copied frames queued for submission at once.
pub(crate) const MAX_QUEUED_COPIES: usize = 8;

/// The pixel format of the CoreVideo surface `src` wraps, when the frame
/// is the whole surface and the surface has the configured size: `420v` or
/// `420f`, with the luma and chroma at its planes 0 and 1 and the row
/// strides of those planes, for NV12, and `yuvs` or `yuvf` at the surface's
/// row stride for YUYV.
fn in_place_format(config: &EncoderConfig, src: &TensorDyn, opts: &FrameOptions) -> Option<u32> {
    if opts.crop.is_some() || src.view_origin().is_some() {
        return None;
    }
    let surface = src.iosurface_ref()?;
    // SAFETY: the tensor keeps its surface live.
    let (format, size) = unsafe {
        (
            IOSurfaceGetPixelFormat(surface),
            (IOSurfaceGetWidth(surface), IOSurfaceGetHeight(surface)),
        )
    };
    if size != (config.width as usize, config.height as usize) {
        return None;
    }
    match (config.input_format, format) {
        (PixelFormat::Nv12, FOURCC_420V | FOURCC_420F) => {
            let chroma = src.chroma_dyn()?;
            let chroma_surface = chroma.iosurface_ref()?;
            // SAFETY: the tensors keep their surfaces live, and the plane
            // addresses are only compared.
            let (same, planes, at, pitches) = unsafe {
                let base = IOSurfaceGetBaseAddress(surface) as usize;
                let at = |plane| {
                    (IOSurfaceGetBaseAddressOfPlane(surface, plane) as usize).wrapping_sub(base)
                };
                let pitch = |plane| IOSurfaceGetBytesPerRowOfPlane(surface, plane);
                (
                    IOSurfaceGetID(chroma_surface) == IOSurfaceGetID(surface),
                    IOSurfaceGetPlaneCount(surface),
                    [at(0), at(1)],
                    [pitch(0), pitch(1)],
                )
            };
            let offsets = [
                src.plane_offset().unwrap_or(0),
                chroma.plane_offset().unwrap_or(0),
            ];
            let strides = [src.effective_row_stride(), chroma.effective_row_stride()];
            (same && planes == 2 && offsets == at && strides == pitches.map(Some)).then_some(format)
        }
        (PixelFormat::Yuyv, FOURCC_YUVS | FOURCC_YUVF) => {
            // SAFETY: the tensor keeps its surface live.
            let pitch = unsafe { IOSurfaceGetBytesPerRow(surface) };
            (src.plane_offset().unwrap_or(0) == 0 && src.effective_row_stride() == Some(pitch))
                .then_some(format)
        }
        _ => None,
    }
}

/// The pixel format of the IOSurface of `src`, when it is a CoreVideo
/// `420v`, `420f`, `yuvs` or `yuvf` surface.
fn corevideo_format(src: &TensorDyn) -> Option<u32> {
    let surface = src.iosurface_ref()?;
    // SAFETY: the tensor keeps its surface live.
    let format = unsafe { IOSurfaceGetPixelFormat(surface) };
    surface_range(format).map(|_| format)
}

/// A pixel buffer over the IOSurface of `src`.
fn wrap_surface(src: &TensorDyn) -> Result<CfOwned> {
    let surface = src
        .iosurface_ref()
        .ok_or_else(|| CodecError::InvalidInput("the source has no IOSurface".into()))?;
    let mut buffer = std::ptr::null_mut();
    // SAFETY: the tensor keeps its surface live, and `buffer` is a valid
    // out-pointer.
    let s = unsafe {
        CVPixelBufferCreateWithIOSurface(std::ptr::null(), surface, std::ptr::null(), &mut buffer)
    };
    // SAFETY: on success `buffer` holds a +1 reference.
    match unsafe { CfOwned::from_create(buffer) } {
        Some(buffer) if s == 0 => Ok(buffer),
        _ => Err(status("CVPixelBufferCreateWithIOSurface", s)),
    }
}

/// The colorimetry of `src`: its tag, with the range of the CoreVideo
/// surface it is on, then for each unset axis the colour attachments of
/// `surface`, a pixel buffer over that surface, then the height rule.
///
/// # Errors
///
/// [`CodecError::InvalidInput`] when the tag's range contradicts the
/// surface's.
fn source_colour(src: &TensorDyn, surface: Option<&CfOwned>) -> Result<Colorimetry> {
    let mut colour = src.colorimetry().unwrap_or_default();
    let format = corevideo_format(src);
    if let Some(range) = format.and_then(surface_range) {
        match colour.range {
            Some(tagged) if tagged != range => {
                return Err(CodecError::InvalidInput(format!(
                    "the source is tagged {tagged:?} range on a {range:?}-range CoreVideo surface"
                )))
            }
            _ => colour.range = Some(range),
        }
    }
    if let Some(buffer) = surface {
        // SAFETY: `buffer` is a live pixel buffer.
        let attached = unsafe { from_attachments(buffer.as_ptr(), format.unwrap_or_default()) };
        colour.space = colour.space.or(attached.space);
        colour.transfer = colour.transfer.or(attached.transfer);
        colour.encoding = colour.encoding.or(attached.encoding);
    }
    Ok(colour.with_height_defaults(src.height()))
}

/// The pool pixel format for `format` sources in `range`.
fn pool_format(format: PixelFormat, range: Option<ColorRange>) -> Result<u32> {
    match (format, range) {
        (PixelFormat::Nv12, Some(ColorRange::Full)) => Ok(FOURCC_420F),
        (PixelFormat::Nv12, _) => Ok(FOURCC_420V),
        (PixelFormat::Yuyv, Some(ColorRange::Full)) => Ok(FOURCC_YUVF),
        (PixelFormat::Yuyv, _) => Ok(FOURCC_YUVS),
        (other, _) => Err(CodecError::UnsupportedFormat(other)),
    }
}

/// A pixel buffer locked for writing, unlocked on drop.
struct Locked(CVPixelBufferRef);

impl Locked {
    /// Locks `buffer`, which must stay live while this does.
    fn write(buffer: CVPixelBufferRef) -> Result<Self> {
        // SAFETY: `buffer` is a live pixel buffer.
        match unsafe { CVPixelBufferLockBaseAddress(buffer, 0) } {
            0 => Ok(Self(buffer)),
            s => Err(status("CVPixelBufferLockBaseAddress", s)),
        }
    }
}

impl Drop for Locked {
    fn drop(&mut self) {
        // SAFETY: locked by `write`, once, and still live.
        unsafe { CVPixelBufferUnlockBaseAddress(self.0, 0) };
    }
}

/// Copies the configured region of `src` at `origin` into the pool buffer
/// `buffer`.
fn fill(
    buffer: &CfOwned,
    config: &EncoderConfig,
    src: &TensorDyn,
    origin: (usize, usize),
    stride: usize,
) -> Result<()> {
    let buffer = buffer.as_ptr();
    let _lock = Locked::write(buffer)?;
    let layout = || CodecError::Device {
        op: "CVPixelBufferGetBaseAddressOfPlane".into(),
        source: std::io::Error::other("the pool buffer's planes are not in one mapping"),
    };
    // SAFETY: the buffer is locked, so its base addresses are valid.
    let (base, len, to) = unsafe {
        if CVPixelBufferGetPlaneCount(buffer) >= 2 {
            let luma = CVPixelBufferGetBaseAddressOfPlane(buffer, 0).cast::<u8>();
            let chroma = CVPixelBufferGetBaseAddressOfPlane(buffer, 1).cast::<u8>();
            let pitches = [
                CVPixelBufferGetBytesPerRowOfPlane(buffer, 0),
                CVPixelBufferGetBytesPerRowOfPlane(buffer, 1),
            ];
            let rows = [
                CVPixelBufferGetHeightOfPlane(buffer, 0),
                CVPixelBufferGetHeightOfPlane(buffer, 1),
            ];
            let (chroma_at, len) =
                chroma_offset(luma as usize, chroma as usize, pitches, rows).ok_or_else(layout)?;
            let to = vec![
                Plane {
                    offset: 0,
                    pitch: pitches[0],
                },
                Plane {
                    offset: chroma_at,
                    pitch: pitches[1],
                },
            ];
            (luma, len, to)
        } else {
            let base = CVPixelBufferGetBaseAddress(buffer).cast::<u8>();
            let pitch = CVPixelBufferGetBytesPerRow(buffer);
            if base.is_null() {
                return Err(layout());
            }
            (
                base,
                pitch * CVPixelBufferGetHeight(buffer),
                vec![Plane { offset: 0, pitch }],
            )
        }
    };
    // SAFETY: the lock keeps the buffer's `len` bytes from `base` mapped and
    // writable, and nothing else reads or writes them until it is submitted.
    let dst = unsafe { std::slice::from_raw_parts_mut(base, len) };
    let size = (config.width as usize, config.height as usize);
    copy_source(config.input_format, size, src, origin, stride, dst, &to)
}

/// The display name of `session`'s encoder in `encoders`, else its ID.
fn device_name(session: &Session, encoders: &[ListedEncoder]) -> String {
    let id = session.encoder_id.as_ref();
    id.and_then(|id| encoders.iter().find(|e| &e.id == id))
        .map(|e| e.name.clone())
        .or_else(|| id.cloned())
        .unwrap_or_else(|| DEVICE.into())
}

pub(crate) struct VtEncoder {
    config: EncoderConfig,
    shared: Arc<Shared>,
    session: Option<Session>,
    submitter: Submitter,
    /// The H.264 encoders VideoToolbox listed when the encoder opened.
    encoders: Vec<ListedEncoder>,
    device: String,
    hardware: bool,
    /// One frame period in ticks of [`TIMESCALE`].
    duration: i64,
    /// The stream's colorimetry, from its first accepted frame.
    colour: Option<Colorimetry>,
    /// The next frame starts a new sequence: an IDR picture.
    restart: bool,
    /// How the last frame reached the session.
    transfer: Transfer,
    /// The last session created bounds its frame delay: CoreVideo sources
    /// may be read in place.
    in_place: bool,
    #[cfg(test)]
    pool_buffers_taken: usize,
}

impl VtEncoder {
    /// Opens a prepared session for `config`.
    pub(crate) fn open(config: EncoderConfig) -> Result<Self> {
        let fourcc = pool_format(config.input_format, None)?;
        let shared = Arc::new(Shared {
            repeat_headers: config.repeat_headers,
            constrained: config.target_profile() == H264Profile::ConstrainedBaseline,
            state: Mutex::new(State {
                first: true,
                ..State::default()
            }),
        });
        let encoders = h264_encoders();
        if encoders.is_empty() {
            return Err(CodecError::NoDevice(
                "VideoToolbox lists no H.264 encoder".into(),
            ));
        }
        let session = Session::new(&config, fourcc, None, &shared).map_err(|e| match e {
            CodecError::Device { op, source } => CodecError::InvalidConfig(format!(
                "{DEVICE} cannot encode {}x{} {} at {} fps: {op} failed with {source}",
                config.width, config.height, config.input_format, config.frame_rate
            )),
            e => e,
        })?;
        let device = device_name(&session, &encoders);
        log::info!(
            "h264 encoder VideoToolbox: {device} ({}, {}x{} {}, {} bps)",
            if session.hardware {
                "hardware"
            } else {
                "software"
            },
            config.width,
            config.height,
            config.input_format,
            config.target_bitrate()
        );
        Ok(Self {
            hardware: session.hardware,
            in_place: session.bounded_delay,
            duration: (f64::from(TIMESCALE) / config.frame_rate).round().max(1.0) as i64,
            config,
            submitter: Submitter::start(Arc::clone(&shared))?,
            shared,
            session: Some(session),
            encoders,
            device,
            colour: None,
            restart: true,
            transfer: Transfer::CpuCopy,
            #[cfg(test)]
            pool_buffers_taken: 0,
        })
    }

    /// Makes sessions this thread creates treat the frame delay as refused
    /// while `refuse` is set.
    #[cfg(test)]
    pub(crate) fn refuse_frame_delay_for_test(refuse: bool) {
        session::REFUSE_FRAME_DELAY.with(|r| r.set(refuse));
    }

    /// Whether the last session created bounds its frame delay.
    #[cfg(test)]
    pub(crate) fn bounded_delay_for_test(&self) -> bool {
        self.in_place
    }

    /// The ID of the encoder the session runs on.
    #[cfg(test)]
    pub(crate) fn encoder_id_for_test(&self) -> Option<String> {
        self.session.as_ref().and_then(|s| s.encoder_id.clone())
    }

    /// The sources read in place that the encoder holds.
    #[cfg(test)]
    pub(crate) fn held_for_test(&self) -> usize {
        self.shared.lock().in_flight.held()
    }

    /// Whether the source of the frame with `pts` is read in place and held.
    #[cfg(test)]
    pub(crate) fn holds_for_test(&self, pts: u64) -> bool {
        self.shared.lock().in_flight.holds(pts)
    }

    /// The pool buffers the encoder has taken for copies.
    #[cfg(test)]
    pub(crate) fn pool_buffers_for_test(&self) -> usize {
        self.pool_buffers_taken
    }

    /// Stops the submission thread taking frames while `paused` is set.
    #[cfg(test)]
    pub(crate) fn pause_submission_for_test(&self, paused: bool) {
        self.submitter.pause(paused);
    }

    /// Submits every queued frame, then invalidates the session, leaving it
    /// in place.
    #[cfg(test)]
    pub(crate) fn invalidate_session_for_test(&mut self) {
        self.submitter.drain().expect("the submission thread runs");
        if let Some(session) = &mut self.session {
            session.invalidate();
        }
    }

    /// Replaces the session with a new one whose pool holds `fourcc`
    /// buffers, describing `colour`.
    fn start_session(&mut self, fourcc: u32, colour: Option<&Colorimetry>) -> Result<()> {
        self.submitter.drain()?;
        self.session = None;
        let session = Session::new(&self.config, fourcc, colour, &self.shared)?;
        self.hardware = session.hardware;
        self.in_place = session.bounded_delay;
        self.device = device_name(&session, &self.encoders);
        self.session = Some(session);
        Ok(())
    }

    /// Makes the session describe `colour`. The session is kept when it
    /// already describes `colour`, or has encoded nothing and has the pool
    /// format `colour` needs; otherwise it is recreated.
    fn describe(&mut self, colour: Colorimetry) -> Result<()> {
        let fourcc = pool_format(self.config.input_format, colour.range)?;
        match &mut self.session {
            Some(s) if s.fourcc == fourcc && s.colour == Some(colour) => {}
            Some(s) if s.fourcc == fourcc && s.fresh => s.describe(&colour)?,
            _ => {
                let started = std::time::Instant::now();
                self.start_session(fourcc, Some(&colour))?;
                log::debug!(
                    "{DEVICE}: new session for {colour:?} in {:?}",
                    started.elapsed()
                );
            }
        }
        Ok(())
    }

    /// Returns the error the callback reported, after resetting.
    fn take_error(&mut self) -> Result<()> {
        let error = self.shared.lock().error.take();
        match error {
            Some(e) => {
                self.reset();
                Err(e)
            }
            None => Ok(()),
        }
    }

    /// Ends the session after a failure and forgets what it held in
    /// flight; the next frame opens a new session and sequence.
    fn reset(&mut self) {
        if let Err(e) = self.submitter.discard() {
            log::debug!("{DEVICE}: {e}");
        }
        if let Some(mut session) = self.session.take() {
            session.invalidate();
        }
        let mut state = self.shared.lock();
        state.in_flight.clear();
        state.error = None;
        state.first = true;
        state.rekey = false;
        self.restart = true;
    }

    /// Submits `src`, whose rows are `stride` bytes apart and whose
    /// colorimetry is `colour`: read in place through `in_place`, a pixel
    /// buffer over its surface, or else copied into a pool buffer.
    fn submit(
        &mut self,
        src: &TensorDyn,
        opts: &FrameOptions,
        origin: (usize, usize),
        (stride, colour): (usize, Colorimetry),
        in_place: Option<CfOwned>,
    ) -> Result<()> {
        if self.session.is_none() {
            let fourcc = pool_format(self.config.input_format, colour.range)?;
            self.start_session(fourcc, Some(&colour))?;
        }
        let session = self.session.as_ref().expect("a session");
        let (buffer, held, transfer) = match in_place {
            Some(buffer) if session.bounded_delay => (
                buffer.clone(),
                Some(HeldSource::new(buffer)?),
                Transfer::ZeroCopy,
            ),
            _ => {
                let queued = self.submitter.queued_copies();
                if queued >= MAX_QUEUED_COPIES {
                    return Err(CodecError::Timeout(format!(
                        "{DEVICE} has {queued} copied frames waiting to be submitted"
                    )));
                }
                let buffer = session.pixel_buffer()?;
                #[cfg(test)]
                {
                    self.pool_buffers_taken += 1;
                }
                fill(&buffer, &self.config, src, origin, stride)?;
                (buffer, None, Transfer::CpuCopy)
            }
        };
        let (seq, rekey) = {
            let mut state = self.shared.lock();
            (
                state.in_flight.submit(opts.pts, held),
                std::mem::take(&mut state.rekey),
            )
        };
        let keyframe = opts.keyframe || self.restart || rekey;
        let pushed = self.submitter.push(Job {
            session: session.handle(),
            buffer,
            seq,
            duration: self.duration,
            keyframe,
            copied: transfer == Transfer::CpuCopy,
        });
        if let Err(e) = pushed {
            self.shared.lock().in_flight.dropped(seq);
            return Err(e);
        }
        if let Some(session) = &mut self.session {
            session.fresh = false;
        }
        self.restart = false;
        self.transfer = transfer;
        Ok(())
    }
}

impl Drop for VtEncoder {
    fn drop(&mut self) {
        let queued = self.submitter.stop();
        {
            let mut state = self.shared.lock();
            for seq in queued {
                state.in_flight.dropped(seq);
            }
        }
        self.session = None;
        self.shared.lock().in_flight.clear();
    }
}

impl EncoderBackend for VtEncoder {
    fn encode(&mut self, src: &TensorDyn, opts: &FrameOptions) -> Result<Option<EncodedFrame>> {
        self.take_error()?;
        let origin = check_source(&self.config, src, opts)?;
        let stride = src
            .effective_row_stride()
            .ok_or_else(|| CodecError::InvalidInput("source has no row stride".into()))?;
        let in_place = self.in_place && in_place_format(&self.config, src, opts).is_some();
        if in_place {
            let held = self.shared.lock().in_flight.held();
            if held >= MAX_IN_FLIGHT {
                return Err(CodecError::Timeout(format!(
                    "{DEVICE} is still reading {held} sources in place"
                )));
            }
        }
        let surface = match corevideo_format(src) {
            Some(_) => Some(wrap_surface(src)?),
            None => None,
        };
        let colour = source_colour(src, surface.as_ref())?;
        to_corevideo(&colour)?;
        match self.colour {
            Some(c) if c != colour => {
                return Err(CodecError::InvalidInput(format!(
                    "source colorimetry {colour:?} is not the stream's {c:?}"
                )))
            }
            Some(_) => {}
            None => self.describe(colour)?,
        }
        let in_place = surface.filter(|_| in_place);
        if let Err(e) = self.submit(src, opts, origin, (stride, colour), in_place) {
            if matches!(e, CodecError::Device { .. }) {
                self.reset();
            }
            return Err(e);
        }
        self.colour = Some(colour);
        Ok(self.shared.lock().ready.pop_front())
    }

    fn flush(&mut self) -> Result<Vec<EncodedFrame>> {
        self.take_error()?;
        if let Err(e) = self.submitter.drain() {
            self.reset();
            return Err(e);
        }
        self.take_error()?;
        if let Some(session) = &self.session {
            if let Err(e) = session.complete() {
                self.reset();
                return Err(e);
            }
        }
        self.take_error()?;
        let mut state = self.shared.lock();
        if !state.in_flight.frames.is_empty() {
            log::debug!(
                "{DEVICE}: {} frames had no access unit",
                state.in_flight.frames.len()
            );
            state.in_flight.clear();
        }
        log::debug!("{DEVICE}: {} frames dropped", state.dropped);
        state.first = true;
        state.rekey = false;
        let out = state.ready.drain(..).collect();
        drop(state);
        self.restart = true;
        self.colour = None;
        Ok(out)
    }

    fn device(&self) -> String {
        self.device.clone()
    }

    fn info(&self) -> EncoderInfo {
        EncoderInfo {
            engine: Engine::VideoToolbox,
            device: self.device.clone(),
            hardware: self.hardware,
            transfer: self.transfer,
        }
    }

    fn create_source(&self) -> Result<TensorDyn> {
        let fourcc = pool_format(self.config.input_format, self.colour.and_then(|c| c.range))?;
        let alloc = || CodecError::Device {
            op: "CFDictionaryCreate".into(),
            source: std::io::Error::other("out of memory"),
        };
        // SAFETY: the dictionary holds a live CF object across the call.
        let attributes = unsafe {
            let surface = cf_dictionary(&[]).ok_or_else(alloc)?;
            cf_dictionary(&[(kCVPixelBufferIOSurfacePropertiesKey, surface.as_ptr())])
                .ok_or_else(alloc)?
        };
        let mut buffer = std::ptr::null_mut();
        // SAFETY: `attributes` is a live dictionary and `buffer` a valid
        // out-pointer.
        let s = unsafe {
            CVPixelBufferCreate(
                std::ptr::null(),
                self.config.width as usize,
                self.config.height as usize,
                fourcc,
                attributes.as_ptr(),
                &mut buffer,
            )
        };
        // SAFETY: on success `buffer` holds a +1 reference.
        let buffer = match unsafe { CfOwned::from_create(buffer) } {
            Some(buffer) if s == kCVReturnSuccess => buffer,
            _ => return Err(status("CVPixelBufferCreate", s)),
        };
        // SAFETY: the pixel buffer keeps its surface live across the call,
        // and the tensor retains the surface.
        let source = unsafe {
            TensorDyn::from_planar_iosurface(CVPixelBufferGetIOSurface(buffer.as_ptr()), None)
        }?;
        Ok(source)
    }
}
