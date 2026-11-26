// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! Slots that bound the hardware codec sessions tests hold at once, across
//! the threads and processes of every test binary on the machine.
//!
//! Each slot is an exclusive `flock` on one of [`SLOTS`] files in the temp
//! directory. A test takes as many slots as the sessions it keeps open at
//! once, before it opens them, and frees them when the returned guard drops.

#![allow(dead_code)]

use std::fs::{File, OpenOptions, TryLockError};
use std::path::PathBuf;
use std::time::Duration;

/// Hardware codec sessions the tests hold at once.
pub const SLOTS: usize = 4;

/// Slots held by a test; dropping it frees them.
#[must_use = "the slots are freed when this is dropped"]
pub struct Sessions {
    _held: Vec<File>,
}

impl Sessions {
    /// The number of slots held.
    pub fn count(&self) -> usize {
        self._held.len()
    }
}

/// The lock file of slot `index`.
fn slot_path(index: usize) -> PathBuf {
    std::env::temp_dir().join(format!("edgefirst-codec-session-{index}.lock"))
}

/// Opens the lock file at `path`, creating it when missing.
fn open_slot(path: &PathBuf) -> File {
    if let Ok(f) = File::open(path) {
        return f;
    }
    if let Err(e) = OpenOptions::new()
        .write(true)
        .create(true)
        .truncate(false)
        .open(path)
    {
        panic!("creating {}: {e}", path.display());
    }
    File::open(path).unwrap_or_else(|e| panic!("opening {}: {e}", path.display()))
}

/// Takes `n` slots, waiting until `n` are free at the same time. The slots
/// are taken all at once or not at all.
///
/// # Panics
///
/// When `n` is more than [`SLOTS`], or a lock file cannot be opened or
/// locked.
pub fn take(n: usize) -> Sessions {
    assert!(
        n <= SLOTS,
        "{n} sessions requested; at most {SLOTS} are held at once"
    );
    let mut attempt: u64 = 0;
    loop {
        let mut held = Vec::with_capacity(n);
        for index in 0..SLOTS {
            if held.len() == n {
                break;
            }
            let path = slot_path(index);
            let file = open_slot(&path);
            match file.try_lock() {
                Ok(()) => held.push(file),
                Err(TryLockError::WouldBlock) => {}
                Err(TryLockError::Error(e)) => panic!("locking {}: {e}", path.display()),
            }
        }
        if held.len() == n {
            return Sessions { _held: held };
        }
        drop(held);
        attempt += 1;
        std::thread::sleep(backoff(attempt));
    }
}

/// A wait of 5 to 36 ms that differs between waiters.
fn backoff(attempt: u64) -> Duration {
    use std::hash::{Hash, Hasher};
    let mut h = std::collections::hash_map::DefaultHasher::new();
    (std::process::id(), std::thread::current().id(), attempt).hash(&mut h);
    Duration::from_millis(5 + h.finish() % 32)
}
