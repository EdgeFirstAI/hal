// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! Shared test-only helper for reporting a skipped test so the report
//! survives libtest's output capture.
//!
//! `eprintln!`/`println!` route through `std::io::_eprint`/`_print`, which
//! libtest hooks to capture per-test output and replay it only for a
//! *failing* test. A hardware- or CPU-feature-gated test that returns
//! early after printing its reason with `eprintln!` therefore reports `ok`
//! with the reason silently discarded -- indistinguishable from having
//! actually exercised the code, both locally (`cargo test` without
//! `--nocapture`) and in CI. Writing directly to `std::io::stderr()`
//! bypasses that hook, so the `SKIPPED: ...` line survives capture in both
//! directions; `scripts/on-target-test.sh` greps captured logs for the
//! literal string `SKIPPED` to report a skip count per board.

/// Reports a test skip with `reason`, writing `SKIPPED: {reason}` directly
/// to stderr (not via `eprintln!`) so libtest's output capture cannot hide
/// it.
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
