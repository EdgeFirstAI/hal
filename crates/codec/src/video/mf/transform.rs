// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! One Media Foundation transform, synchronous or asynchronous, driven from
//! the caller's thread.
//!
//! A synchronous transform takes input until it answers
//! `MF_E_NOTACCEPTING` and gives output until it answers
//! `MF_E_TRANSFORM_NEED_MORE_INPUT`. An asynchronous one (every hardware
//! transform) says when it wants input (`METransformNeedInput`) and when it
//! has output (`METransformHaveOutput`) through its event queue, which is
//! polled here rather than served by callbacks on Media Foundation's threads.

use std::mem::ManuallyDrop;
use std::time::{Duration, Instant};

use windows::core::{Interface, GUID};
use windows::Win32::Media::MediaFoundation::{
    IMFActivate, IMFMediaEventGenerator, IMFMediaType, IMFSample, IMFTransform, MEError,
    METransformDrainComplete, METransformHaveOutput, METransformNeedInput,
    MFT_MESSAGE_COMMAND_DRAIN, MFT_MESSAGE_COMMAND_FLUSH, MFT_MESSAGE_NOTIFY_BEGIN_STREAMING,
    MFT_MESSAGE_NOTIFY_END_STREAMING, MFT_MESSAGE_NOTIFY_START_OF_STREAM, MFT_OUTPUT_DATA_BUFFER,
    MFT_OUTPUT_DATA_BUFFER_FORMAT_CHANGE, MFT_OUTPUT_STREAM_CAN_PROVIDE_SAMPLES,
    MFT_OUTPUT_STREAM_PROVIDES_SAMPLES, MF_EVENT_FLAG_NO_WAIT, MF_E_NOTACCEPTING,
    MF_E_NO_EVENTS_AVAILABLE, MF_E_TRANSFORM_NEED_MORE_INPUT, MF_E_TRANSFORM_STREAM_CHANGE,
    MF_MT_SUBTYPE, MF_TRANSFORM_ASYNC, MF_TRANSFORM_ASYNC_UNLOCK,
};

use super::{com, error, Candidate, Platform};
use crate::{CodecError, Result};

/// How long to sleep between polls of an asynchronous transform's events.
const POLL: Duration = Duration::from_micros(500);

/// What a transform produced.
pub(crate) enum Output {
    /// A finished sample.
    Sample(IMFSample),
    /// The output type changed to this one; samples after it have the new
    /// type.
    Format(IMFMediaType),
}

/// An activated transform with one input and one output stream.
pub(crate) struct Transform {
    mft: IMFTransform,
    activate: IMFActivate,
    events: Option<IMFMediaEventGenerator>,
    name: String,
    hardware: bool,
    input_id: u32,
    output_id: u32,
    /// Inputs an asynchronous transform has asked for and not yet had.
    credits: u32,
    /// An asynchronous transform has finished draining.
    drained: bool,
    /// The output subtype to choose again when the output type changes.
    output_subtype: GUID,
    /// The transform allocates its output samples.
    provides: bool,
    /// Bytes an output sample must hold, when the caller allocates them.
    output_size: usize,
    /// The least the caller allocates for an output sample.
    min_output_size: usize,
    /// A caller-allocated output sample the transform did not fill.
    spare: Option<IMFSample>,
    streaming: bool,
}

impl Transform {
    /// Activates `candidate`; for an asynchronous transform, also unlocks it
    /// for use outside a Media Foundation pipeline.
    pub(crate) fn activate(candidate: Candidate) -> Result<Self> {
        com();
        let Candidate {
            activate,
            name,
            hardware,
        } = candidate;
        // SAFETY: a live activation object.
        let mft: IMFTransform = unsafe { activate.ActivateObject() }
            .map_err(|e| error(e, "IMFActivate::ActivateObject"))?;
        // SAFETY: a live transform.
        let attributes = unsafe { mft.GetAttributes() }.ok();
        let asynchronous = attributes
            .as_ref()
            // SAFETY: a live attribute store.
            .and_then(|a| unsafe { a.GetUINT32(&MF_TRANSFORM_ASYNC) }.ok())
            .is_some_and(|v| v != 0);
        let events = if asynchronous {
            if let Some(a) = &attributes {
                // SAFETY: as above.
                unsafe { a.SetUINT32(&MF_TRANSFORM_ASYNC_UNLOCK, 1) }
                    .map_err(|e| error(e, "unlock an asynchronous transform"))?;
            }
            Some(
                mft.cast::<IMFMediaEventGenerator>()
                    .map_err(|e| error(e, "IMFMediaEventGenerator"))?,
            )
        } else {
            None
        };
        let (mut input, mut output) = ([0u32], [0u32]);
        // SAFETY: one-element arrays for a transform with one stream each;
        // E_NOTIMPL means the stream ids are 0.
        let (input_id, output_id) = match unsafe { mft.GetStreamIDs(&mut input, &mut output) } {
            Ok(()) => (input[0], output[0]),
            Err(_) => (0, 0),
        };
        Ok(Self {
            mft,
            activate,
            events,
            name,
            hardware,
            input_id,
            output_id,
            credits: 0,
            drained: false,
            output_subtype: GUID::zeroed(),
            provides: false,
            output_size: 0,
            min_output_size: 0,
            spare: None,
            streaming: false,
        })
    }

    pub(crate) fn mft(&self) -> &IMFTransform {
        &self.mft
    }

    pub(crate) fn name(&self) -> &str {
        &self.name
    }

    pub(crate) fn hardware(&self) -> bool {
        self.hardware
    }

    /// Sets the input type.
    pub(crate) fn set_input_type(&self, t: &IMFMediaType) -> windows::core::Result<()> {
        // SAFETY: live objects.
        unsafe { self.mft.SetInputType(self.input_id, t, 0) }
    }

    /// Sets the output type; a later change of output type selects the
    /// same subtype again.
    pub(crate) fn set_output_type(&mut self, t: &IMFMediaType) -> windows::core::Result<()> {
        // SAFETY: live objects.
        unsafe { self.mft.SetOutputType(self.output_id, t, 0) }?;
        // SAFETY: a live media type.
        if let Ok(subtype) = unsafe { t.GetGUID(&MF_MT_SUBTYPE) } {
            self.output_subtype = subtype;
        }
        Ok(())
    }

    /// The output types the transform offers, in its order of preference.
    pub(crate) fn output_types(&self) -> Vec<IMFMediaType> {
        (0..)
            // SAFETY: a live transform; the index runs until it fails.
            .map_while(|i| unsafe { self.mft.GetOutputAvailableType(self.output_id, i) }.ok())
            .collect()
    }

    /// The current output type.
    pub(crate) fn output_type(&self) -> Result<IMFMediaType> {
        // SAFETY: a live transform.
        unsafe { self.mft.GetOutputCurrentType(self.output_id) }
            .map_err(|e| error(e, "IMFTransform::GetOutputCurrentType"))
    }

    /// Sets the least the caller allocates for an output sample.
    pub(crate) fn set_min_output_size(&mut self, bytes: usize) {
        self.min_output_size = bytes;
    }

    /// Reads how output samples are allocated, after the output type is
    /// set or changes.
    fn refresh_output_info(&mut self) -> Result<()> {
        // SAFETY: a live transform.
        let info = unsafe { self.mft.GetOutputStreamInfo(self.output_id) }
            .map_err(|e| error(e, "IMFTransform::GetOutputStreamInfo"))?;
        let flags = info.dwFlags as i32;
        self.provides = flags
            & (MFT_OUTPUT_STREAM_PROVIDES_SAMPLES.0 | MFT_OUTPUT_STREAM_CAN_PROVIDE_SAMPLES.0)
            != 0;
        self.output_size = info.cbSize as usize;
        self.spare = None;
        Ok(())
    }

    /// Starts streaming: the transform may allocate its resources and, if
    /// asynchronous, starts asking for input.
    pub(crate) fn begin(&mut self) -> Result<()> {
        self.refresh_output_info()?;
        self.message(MFT_MESSAGE_NOTIFY_BEGIN_STREAMING, "begin streaming")?;
        self.message(MFT_MESSAGE_NOTIFY_START_OF_STREAM, "start the stream")?;
        self.streaming = true;
        Ok(())
    }

    fn message(
        &self,
        message: windows::Win32::Media::MediaFoundation::MFT_MESSAGE_TYPE,
        what: &str,
    ) -> Result<()> {
        // SAFETY: a live transform; these messages take no parameter.
        unsafe { self.mft.ProcessMessage(message, 0) }.map_err(|e| error(e, what))
    }

    /// Gives the transform `sample`, waiting up to `timeout` for it to take
    /// one. Output that becomes ready meanwhile is added to `out`.
    pub(crate) fn input(
        &mut self,
        platform: &Platform,
        sample: &IMFSample,
        timeout: Duration,
        out: &mut Vec<Output>,
    ) -> Result<()> {
        com();
        let deadline = Instant::now() + timeout;
        loop {
            if self.events.is_some() {
                self.pump(platform, out)?;
                if self.credits == 0 {
                    if Instant::now() >= deadline {
                        return Err(self.timeout("take a frame", timeout));
                    }
                    std::thread::sleep(POLL);
                    continue;
                }
            }
            // SAFETY: live objects.
            match unsafe { self.mft.ProcessInput(self.input_id, sample, 0) } {
                Ok(()) => {
                    self.credits = self.credits.saturating_sub(1);
                    return Ok(());
                }
                Err(e) if e.code() == MF_E_NOTACCEPTING => {
                    if self.events.is_some() {
                        self.credits = 0;
                    } else if self.pull_all(platform, out)? == 0 {
                        // A synchronous transform with no output to give
                        // and no room for input will not recover.
                        return Err(error(e, "IMFTransform::ProcessInput"));
                    }
                }
                Err(e) => return Err(error(e, "IMFTransform::ProcessInput")),
            }
        }
    }

    /// Collects output into `out`. A synchronous transform gives everything
    /// it has at once. An asynchronous one is polled until `done` holds,
    /// called with the outputs so far and whether the transform has asked
    /// for more input, or until `wait` passes.
    pub(crate) fn collect(
        &mut self,
        platform: &Platform,
        wait: Duration,
        out: &mut Vec<Output>,
        done: impl Fn(&[Output], bool) -> bool,
    ) -> Result<()> {
        com();
        if self.events.is_none() {
            self.pull_all(platform, out)?;
            return Ok(());
        }
        let deadline = Instant::now() + wait;
        loop {
            self.pump(platform, out)?;
            if done(out, self.credits > 0) || Instant::now() >= deadline {
                return Ok(());
            }
            std::thread::sleep(POLL);
        }
    }

    /// Asks the transform to finish everything it holds and collects the
    /// output into `out`, waiting up to `timeout`. The transform accepts
    /// input again afterwards.
    pub(crate) fn drain(
        &mut self,
        platform: &Platform,
        timeout: Duration,
        out: &mut Vec<Output>,
    ) -> Result<()> {
        com();
        if !self.streaming {
            return Ok(());
        }
        self.drained = false;
        self.message(MFT_MESSAGE_COMMAND_DRAIN, "drain")?;
        if self.events.is_none() {
            self.pull_all(platform, out)?;
        } else {
            let deadline = Instant::now() + timeout;
            while !self.drained {
                if Instant::now() >= deadline {
                    return Err(self.timeout("finish draining", timeout));
                }
                std::thread::sleep(POLL);
                self.pump(platform, out)?;
            }
            // Requests for input made before the drain lapse with it.
            self.credits = 0;
        }
        // A drained asynchronous transform asks for input again only once
        // told a new stream starts.
        self.message(MFT_MESSAGE_NOTIFY_START_OF_STREAM, "restart the stream")
    }

    /// Discards everything the transform holds.
    pub(crate) fn discard(&mut self) {
        com();
        if self.streaming {
            let _ = self.message(MFT_MESSAGE_COMMAND_FLUSH, "flush");
        }
    }

    fn timeout(&self, what: &str, waited: Duration) -> CodecError {
        CodecError::Timeout(format!("{} did not {what} within {waited:?}", self.name))
    }

    /// Handles every queued event of an asynchronous transform.
    fn pump(&mut self, platform: &Platform, out: &mut Vec<Output>) -> Result<()> {
        let Some(events) = self.events.clone() else {
            return Ok(());
        };
        loop {
            // SAFETY: a live event generator.
            let event = match unsafe { events.GetEvent(MF_EVENT_FLAG_NO_WAIT) } {
                Ok(event) => event,
                Err(e) if e.code() == MF_E_NO_EVENTS_AVAILABLE => return Ok(()),
                Err(e) => return Err(error(e, "IMFMediaEventGenerator::GetEvent")),
            };
            // SAFETY: a live event.
            let kind =
                unsafe { event.GetType() }.map_err(|e| error(e, "IMFMediaEvent::GetType"))?;
            match kind {
                k if k == METransformNeedInput.0 as u32 => self.credits += 1,
                k if k == METransformHaveOutput.0 as u32 => {
                    self.pull(platform, out)?;
                }
                k if k == METransformDrainComplete.0 as u32 => self.drained = true,
                k if k == MEError.0 as u32 => {
                    // SAFETY: a live event.
                    let status = unsafe { event.GetStatus() }
                        .map_err(|e| error(e, "IMFMediaEvent::GetStatus"))?;
                    return Err(error(status.into(), "the transform reported an error"));
                }
                other => log::trace!("{}: event {other}", self.name),
            }
        }
    }

    /// Takes output until a synchronous transform needs more input.
    /// Returns how many outputs were taken.
    fn pull_all(&mut self, platform: &Platform, out: &mut Vec<Output>) -> Result<usize> {
        let mut n = 0;
        while self.pull(platform, out)? {
            n += 1;
        }
        Ok(n)
    }

    /// Takes one output sample, choosing the output type again first when
    /// it has changed. Returns false when the transform has none.
    fn pull(&mut self, platform: &Platform, out: &mut Vec<Output>) -> Result<bool> {
        loop {
            let sample = if self.provides {
                None
            } else if let Some(spare) = self.spare.take() {
                Some(spare)
            } else {
                let size = self.output_size.max(self.min_output_size).max(1);
                Some(platform.sample(size)?.0)
            };
            let ours = sample.is_some();
            let mut buffer = [MFT_OUTPUT_DATA_BUFFER {
                dwStreamID: self.output_id,
                pSample: ManuallyDrop::new(sample),
                dwStatus: 0,
                pEvents: ManuallyDrop::new(None),
            }];
            let mut status = 0;
            // SAFETY: a live transform and one output buffer for its one
            // stream.
            let result = unsafe { self.mft.ProcessOutput(0, &mut buffer, &mut status) };
            let [b] = buffer;
            let sample = ManuallyDrop::into_inner(b.pSample);
            drop(ManuallyDrop::into_inner(b.pEvents));
            match result {
                Ok(()) => {
                    if b.dwStatus & MFT_OUTPUT_DATA_BUFFER_FORMAT_CHANGE.0 as u32 != 0 {
                        self.refresh_output_info()?;
                        out.push(Output::Format(self.output_type()?));
                    }
                    return match sample {
                        Some(s) => {
                            out.push(Output::Sample(s));
                            Ok(true)
                        }
                        None => Ok(false),
                    };
                }
                Err(e) if e.code() == MF_E_TRANSFORM_NEED_MORE_INPUT => {
                    if ours {
                        self.spare = sample;
                    }
                    return Ok(false);
                }
                Err(e) if e.code() == MF_E_TRANSFORM_STREAM_CHANGE => {
                    self.renegotiate()?;
                    out.push(Output::Format(self.output_type()?));
                }
                Err(e) => return Err(error(e, "IMFTransform::ProcessOutput")),
            }
        }
    }

    /// Selects the output type with the subtype set before, after the
    /// transform has changed what it offers.
    fn renegotiate(&mut self) -> Result<()> {
        let subtype = self.output_subtype;
        let chosen = self
            .output_types()
            .into_iter()
            // SAFETY: live media types.
            .find(|t| unsafe { t.GetGUID(&MF_MT_SUBTYPE) }.is_ok_and(|s| s == subtype))
            .ok_or_else(|| {
                CodecError::InvalidData(format!(
                    "{} no longer offers the output format after a stream change",
                    self.name
                ))
            })?;
        self.set_output_type(&chosen)
            .map_err(|e| error(e, "IMFTransform::SetOutputType"))?;
        self.refresh_output_info()
    }
}

impl Drop for Transform {
    fn drop(&mut self) {
        com();
        if self.streaming {
            let _ = self.message(MFT_MESSAGE_NOTIFY_END_STREAMING, "end streaming");
        }
        // SAFETY: a live activation object; the transform is not used after
        // this.
        let _ = unsafe { self.activate.ShutdownObject() };
    }
}
