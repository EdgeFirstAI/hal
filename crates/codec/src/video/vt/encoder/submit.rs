// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! The thread that submits frames to the compression session, in order.

use std::collections::VecDeque;
use std::ffi::c_void;
use std::sync::{Arc, Condvar, Mutex, MutexGuard};
use std::thread::JoinHandle;

use super::session::{out_of_memory, Shared, TIMESCALE};
use crate::video::vt::ffi::*;
use crate::video::vt::status;
use crate::{CodecError, Result};

/// A frame waiting to be submitted.
pub(super) struct Job {
    /// The session to submit to.
    pub(super) session: CfOwned,
    pub(super) buffer: CfOwned,
    pub(super) seq: u64,
    /// One frame period in ticks of [`TIMESCALE`].
    pub(super) duration: i64,
    pub(super) keyframe: bool,
    /// The buffer holds a copy of the source.
    pub(super) copied: bool,
}

#[derive(Default)]
struct Queue {
    jobs: VecDeque<Job>,
    /// The thread is submitting a job it has taken from `jobs`.
    busy: bool,
    stop: bool,
    /// The thread has exited.
    dead: bool,
    /// The thread takes no job while this is set.
    #[cfg(test)]
    paused: bool,
    /// The thread panics when it next wakes.
    #[cfg(test)]
    panic: bool,
}

impl Queue {
    /// The copied frames queued and not yet taken by the thread.
    fn copies(&self) -> usize {
        self.jobs.iter().filter(|job| job.copied).count()
    }
}

#[derive(Default)]
struct Channel {
    queue: Mutex<Queue>,
    changed: Condvar,
}

impl Channel {
    fn lock(&self) -> MutexGuard<'_, Queue> {
        self.queue.lock().unwrap_or_else(|e| e.into_inner())
    }

    fn wait<'a>(&self, queue: MutexGuard<'a, Queue>) -> MutexGuard<'a, Queue> {
        self.changed.wait(queue).unwrap_or_else(|e| e.into_inner())
    }
}

/// The error for a submission thread that has exited.
fn dead() -> CodecError {
    CodecError::Device {
        op: "the VideoToolbox submission thread".into(),
        source: std::io::Error::other("the thread has exited"),
    }
}

/// Marks the thread dead and not busy when `run` returns or unwinds.
struct Exit<'a>(&'a Channel);

impl Drop for Exit<'_> {
    fn drop(&mut self) {
        let mut queue = self.0.lock();
        queue.busy = false;
        queue.dead = true;
        drop(queue);
        self.0.changed.notify_all();
    }
}

/// The submission thread and its queue. Dropping it stops and joins the
/// thread, discarding jobs not yet submitted.
pub(super) struct Submitter {
    channel: Arc<Channel>,
    thread: Option<JoinHandle<()>>,
}

impl Submitter {
    /// Starts a thread that submits jobs and records a failed submission in
    /// `shared` as that frame's error.
    pub(super) fn start(shared: Arc<Shared>) -> Result<Self> {
        let channel = Arc::new(Channel::default());
        let worker = Arc::clone(&channel);
        let thread = std::thread::Builder::new()
            .name("vt-encode".into())
            .spawn(move || run(&worker, &shared))
            .map_err(CodecError::Io)?;
        Ok(Self {
            channel,
            thread: Some(thread),
        })
    }

    /// Queues `job` behind the jobs already queued.
    ///
    /// # Errors
    ///
    /// [`CodecError::Device`] when the thread has exited; `job` is dropped.
    pub(super) fn push(&self, job: Job) -> Result<()> {
        let mut queue = self.channel.lock();
        if queue.dead {
            return Err(dead());
        }
        queue.jobs.push_back(job);
        drop(queue);
        self.channel.changed.notify_all();
        Ok(())
    }

    /// The copied frames queued and not yet taken by the thread.
    pub(super) fn queued_copies(&self) -> usize {
        self.channel.lock().copies()
    }

    /// Stops the thread taking jobs while `paused` is set.
    #[cfg(test)]
    pub(super) fn pause(&self, paused: bool) {
        self.channel.lock().paused = paused;
        self.channel.changed.notify_all();
    }

    /// Makes the thread panic when it next wakes.
    #[cfg(test)]
    pub(super) fn panic_for_test(&self) {
        self.channel.lock().panic = true;
        self.channel.changed.notify_all();
    }

    /// Waits until every queued job has been submitted.
    ///
    /// # Errors
    ///
    /// [`CodecError::Device`] when the thread has exited.
    pub(super) fn drain(&self) -> Result<()> {
        let mut queue = self.channel.lock();
        while !queue.dead && (!queue.jobs.is_empty() || queue.busy) {
            queue = self.channel.wait(queue);
        }
        if queue.dead {
            return Err(dead());
        }
        Ok(())
    }

    /// Drops the jobs not yet submitted and waits for the one being
    /// submitted, if any.
    ///
    /// # Errors
    ///
    /// [`CodecError::Device`] when the thread has exited.
    pub(super) fn discard(&self) -> Result<()> {
        let mut queue = self.channel.lock();
        queue.jobs.clear();
        while !queue.dead && queue.busy {
            queue = self.channel.wait(queue);
        }
        if queue.dead {
            return Err(dead());
        }
        Ok(())
    }

    /// Stops and joins the thread, returning the sequence numbers of the
    /// jobs it did not submit.
    pub(super) fn stop(&mut self) -> Vec<u64> {
        self.channel.lock().stop = true;
        self.channel.changed.notify_all();
        if let Some(thread) = self.thread.take() {
            if thread.join().is_err() {
                log::error!("the VideoToolbox submission thread panicked");
            }
        }
        self.channel
            .lock()
            .jobs
            .drain(..)
            .map(|job| job.seq)
            .collect()
    }
}

impl Drop for Submitter {
    fn drop(&mut self) {
        self.stop();
    }
}

/// Submits jobs in order until stopped.
fn run(channel: &Channel, shared: &Shared) {
    let _exit = Exit(channel);
    let mut queue = channel.lock();
    loop {
        if queue.stop {
            return;
        }
        #[cfg(test)]
        if queue.panic {
            panic!("the submission thread was told to panic");
        }
        #[cfg(test)]
        if queue.paused {
            queue = channel.wait(queue);
            continue;
        }
        let Some(job) = queue.jobs.pop_front() else {
            queue = channel.wait(queue);
            continue;
        };
        queue.busy = true;
        drop(queue);
        channel.changed.notify_all();
        if let Err(e) = encode_frame(&job) {
            shared.lock().fail(job.seq, e);
        }
        drop(job);
        queue = channel.lock();
        queue.busy = false;
        channel.changed.notify_all();
    }
}

/// A presentation time of `ticks` of [`TIMESCALE`].
fn time(ticks: i64) -> CMTime {
    CMTime {
        value: ticks,
        timescale: TIMESCALE,
        flags: kCMTimeFlags_Valid,
        epoch: 0,
    }
}

/// Submits the job's buffer as frame `seq`, a key frame when `keyframe` is
/// set.
fn encode_frame(job: &Job) -> Result<()> {
    let properties = if job.keyframe {
        // SAFETY: the key is a VideoToolbox constant and the value a CF
        // constant.
        let p = unsafe { cf_dictionary(&[(kVTEncodeFrameOptionKey_ForceKeyFrame, cf_bool(true))]) };
        Some(p.ok_or_else(|| out_of_memory("CFDictionaryCreate"))?)
    } else {
        None
    };
    let mut flags = 0;
    // SAFETY: a live session, pixel buffer and properties; the sequence
    // number travels as the frame's pointer-sized ref-con and is never
    // dereferenced.
    let s = unsafe {
        VTCompressionSessionEncodeFrame(
            job.session.as_ptr(),
            job.buffer.as_ptr(),
            time(job.seq as i64 * job.duration),
            time(job.duration),
            properties.as_ref().map_or(std::ptr::null(), |p| p.as_ptr()),
            job.seq as *mut c_void,
            &mut flags,
        )
    };
    match s {
        0 => Ok(()),
        s => Err(status("VTCompressionSessionEncodeFrame", s)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn submitter() -> Submitter {
        let shared = Arc::new(Shared {
            repeat_headers: false,
            constrained: false,
            state: Mutex::default(),
        });
        Submitter::start(shared).expect("a submission thread")
    }

    /// A job over empty dictionaries, which a dead thread never submits.
    fn stand_in() -> Job {
        // SAFETY: an empty dictionary has no entries to keep live.
        let dictionary = || unsafe { cf_dictionary(&[]) }.expect("a dictionary");
        Job {
            session: dictionary(),
            buffer: dictionary(),
            seq: 0,
            duration: 1,
            keyframe: false,
            copied: false,
        }
    }

    #[test]
    fn a_dead_thread_is_a_device_error_not_a_wait() {
        let submitter = submitter();
        submitter.panic_for_test();
        let queued = submitter.push(stand_in());
        assert!(
            matches!(queued, Ok(()) | Err(CodecError::Device { .. })),
            "{queued:?}"
        );
        let device = |r: Result<()>| matches!(r, Err(CodecError::Device { .. }));
        assert!(device(submitter.drain()));
        assert!(device(submitter.discard()));
        assert!(device(submitter.push(stand_in())));
    }

    #[test]
    fn a_running_thread_drains_and_discards() {
        let submitter = submitter();
        submitter.drain().expect("drain");
        submitter.discard().expect("discard");
    }
}
