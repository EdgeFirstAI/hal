// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! Submitting a frame again while an encoder refuses it.

use std::time::{Duration, Instant};

/// Calls `encode` until it returns a result `refused` does not accept,
/// waiting 1 ms between calls, and returns that result. After 2 s it
/// returns the refusal.
pub fn retry<T, E>(
    mut encode: impl FnMut() -> Result<T, E>,
    refused: impl Fn(&E) -> bool,
) -> Result<T, E> {
    let deadline = Instant::now() + Duration::from_secs(2);
    loop {
        match encode() {
            Err(e) if refused(&e) && Instant::now() < deadline => {
                std::thread::sleep(Duration::from_millis(1));
            }
            result => return result,
        }
    }
}
