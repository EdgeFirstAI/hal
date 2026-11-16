// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! Test helper that reports a skipped test on stderr, outside libtest's
//! output capture.

/// Reports a test skip with `reason`, writing `SKIPPED: {reason}` directly
/// to stderr, outside libtest's output capture.
#[cfg(test)]
pub(crate) fn report_skip(reason: &str) {
    use std::io::Write;
    let _ = writeln!(&mut std::io::stderr(), "WARNING: SKIPPED: {reason}");
}

#[cfg(all(target_os = "linux", feature = "v4l2"))]
#[path = "../../tensor/tests/support/dma_require.rs"]
#[allow(dead_code)]
mod dma_require;

#[cfg(all(target_os = "linux", feature = "v4l2"))]
#[path = "../../tensor/tests/support/outcome.rs"]
#[allow(dead_code)]
mod outcome;

/// Unwraps a DMA-BUF allocation: it must succeed when the heap is usable (under
/// `HAL_TEST_REQUIRE_DMA`), and may fail with a warning otherwise.
#[cfg(all(target_os = "linux", feature = "v4l2"))]
pub(crate) fn dma_allocated<T, E: std::fmt::Debug>(result: Result<T, E>, what: &str) -> Option<T> {
    match result {
        Ok(value) => Some(value),
        Err(e) => {
            let usable = dma_require::decide(
                edgefirst_tensor::is_dma_available(),
                dma_require::required(),
                what,
                "no usable DMA heap",
            );
            outcome::allocated(Err(e), what, outcome::expect_dma(usable))
        }
    }
}
