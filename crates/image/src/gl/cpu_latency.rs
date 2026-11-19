// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! Process-wide CPU wake-latency request held while GL work is in flight.

use std::io::Write;
use std::sync::{Mutex, PoisonError};

/// CPU wake latency requested while a hold is active, in microseconds.
const HELD_US: i32 = 0;
/// The kernel's value for a CPU latency request that sets no constraint.
const RELEASED_US: i32 = 2_000_000_000;
/// Environment override: `on` or `off`.
const ENV: &str = "EDGEFIRST_GL_CPU_LATENCY";
/// Path of the kernel's CPU latency request device.
#[cfg(target_os = "linux")]
const PATH: &str = "/dev/cpu_dma_latency";
/// What running without the request costs, and how to restore it.
const WITHOUT_REQUEST: &str =
    "GPU operations run with extra latency while idle CPU cores wake to relay \
     each completion; grant this process write access to /dev/cpu_dma_latency for optimal \
     performance (HAL README: Qualcomm Adreno on Linux)";

/// A reference-counted latency request written to `sink`: the held value
/// when the first holder arrives, the released value when the last leaves.
struct Requests<W> {
    sink: W,
    holders: usize,
}

impl<W: Write> Requests<W> {
    fn new(sink: W) -> Self {
        Self { sink, holders: 0 }
    }

    fn acquire(&mut self) -> std::io::Result<()> {
        if self.holders == 0 {
            self.sink.write_all(&HELD_US.to_ne_bytes())?;
        }
        self.holders += 1;
        Ok(())
    }

    fn release(&mut self) -> std::io::Result<()> {
        self.holders = self.holders.saturating_sub(1);
        if self.holders == 0 {
            self.sink.write_all(&RELEASED_US.to_ne_bytes())?;
        }
        Ok(())
    }
}

enum Device {
    Unopened,
    Open(Requests<std::fs::File>),
    Unavailable,
}

static DEVICE: Mutex<Device> = Mutex::new(Device::Unopened);

#[cfg(target_os = "linux")]
fn open() -> Device {
    match std::fs::OpenOptions::new().write(true).open(PATH) {
        Ok(file) => Device::Open(Requests::new(file)),
        Err(e) => {
            log::info!("cannot open {PATH} ({e}): {WITHOUT_REQUEST}");
            Device::Unavailable
        }
    }
}

#[cfg(not(target_os = "linux"))]
fn open() -> Device {
    Device::Unavailable
}

/// Holds the process-wide CPU wake-latency request until dropped.
pub(super) struct CpuLatencyHold(());

/// Takes the CPU wake-latency request, opening the device on first use.
/// Returns `None` when the device cannot be opened or written.
pub(super) fn hold() -> Option<CpuLatencyHold> {
    let mut device = DEVICE.lock().unwrap_or_else(PoisonError::into_inner);
    if matches!(*device, Device::Unopened) {
        *device = open();
    }
    let Device::Open(requests) = &mut *device else {
        return None;
    };
    match requests.acquire() {
        Ok(()) => Some(CpuLatencyHold(())),
        Err(e) => {
            log::info!("CPU wake-latency request failed ({e}): {WITHOUT_REQUEST}");
            *device = Device::Unavailable;
            None
        }
    }
}

impl Drop for CpuLatencyHold {
    fn drop(&mut self) {
        let mut device = DEVICE.lock().unwrap_or_else(PoisonError::into_inner);
        if let Device::Open(requests) = &mut *device {
            if let Err(e) = requests.release() {
                log::info!("CPU wake-latency release failed ({e}): {WITHOUT_REQUEST}");
                *device = Device::Unavailable;
            }
        }
    }
}

/// Whether GL operations hold the request: `EDGEFIRST_GL_CPU_LATENCY`
/// (`on` or `off`), otherwise `default`.
pub(super) fn enabled(default: bool) -> bool {
    parse(std::env::var(ENV).ok().as_deref(), default)
}

fn parse(value: Option<&str>, default: bool) -> bool {
    match value {
        Some("on") => true,
        Some("off") => false,
        _ => default,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn values(bytes: &[u8]) -> Vec<i32> {
        bytes
            .chunks_exact(4)
            .map(|c| i32::from_ne_bytes(c.try_into().unwrap()))
            .collect()
    }

    #[test]
    fn nested_holds_write_the_request_once_and_release_it_with_the_last() {
        let mut r = Requests::new(Vec::new());
        r.acquire().unwrap();
        r.acquire().unwrap();
        assert_eq!(values(&r.sink), [HELD_US]);
        r.release().unwrap();
        assert_eq!(values(&r.sink), [HELD_US]);
        r.release().unwrap();
        assert_eq!(values(&r.sink), [HELD_US, RELEASED_US]);
        r.acquire().unwrap();
        assert_eq!(values(&r.sink), [HELD_US, RELEASED_US, HELD_US]);
    }

    #[test]
    fn environment_value_overrides_the_default() {
        assert!(parse(Some("on"), false));
        assert!(!parse(Some("off"), true));
        assert!(parse(None, true));
        assert!(!parse(None, false));
        assert!(parse(Some("yes"), true));
        assert!(!parse(Some("yes"), false));
    }
}
