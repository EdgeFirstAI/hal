// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! Shared test-only helpers for reporting a skipped test. The report is written
//! directly to `std::io::stderr()`, which libtest's output capture does not
//! hide.

/// Reports a test skip with `reason`, writing `SKIPPED: {reason}` directly to
/// stderr (not via `eprintln!`).
#[cfg(test)]
pub(crate) fn report_skip(reason: &str) {
    use std::io::Write;
    let _ = writeln!(&mut std::io::stderr(), "WARNING: SKIPPED: {reason}");
}

#[cfg(test)]
#[path = "../tests/support/dma_require.rs"]
mod dma_require;

/// `true` when a DMA-heap test should run; `false` after reporting a skip.
///
/// # Panics
///
/// When `HAL_TEST_REQUIRE_DMA=1` and no DMA heap is usable.
#[cfg(test)]
pub(crate) fn dma_or_skip(what: &str) -> bool {
    dma_require::available_or_skip(crate::is_dma_available(), what)
}

#[cfg(test)]
#[path = "../tests/support/outcome.rs"]
#[allow(dead_code)]
mod outcome;
#[cfg(test)]
#[allow(unused_imports)]
pub(crate) use outcome::{allocated, Expect};

/// Unwraps a DMA-BUF allocation: it must succeed when the heap is usable (under
/// `HAL_TEST_REQUIRE_DMA`), and may fail with a warning otherwise.
#[cfg(test)]
#[allow(dead_code)]
pub(crate) fn dma_allocated<T, E: std::fmt::Debug>(result: Result<T, E>, what: &str) -> Option<T> {
    match result {
        Ok(value) => Some(value),
        Err(e) => {
            let usable = dma_require::decide(
                crate::is_dma_available(),
                dma_require::required(),
                what,
                "no usable DMA heap",
            );
            allocated(Err(e), what, outcome::expect_dma(usable))
        }
    }
}
