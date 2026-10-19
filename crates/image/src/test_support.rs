// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! Shared test-only helper for reporting a skipped test so the report
//! survives libtest's output capture.
//!
//! `eprintln!`/`println!` route through `std::io::_eprint`/`_print`, which
//! libtest hooks to capture per-test output and replay it only for a
//! *failing* test. A hardware-gated test that returns early after printing
//! its reason with `eprintln!` therefore reports `ok` with the reason
//! silently discarded -- indistinguishable from having actually exercised
//! the code, both locally (`cargo test` without `--nocapture`) and in CI.
//! Writing directly to `std::io::stderr()` bypasses that hook, so the
//! `SKIPPED: ...` line survives capture in both directions; `scripts/
//! on-target-test.sh` greps captured logs for the literal string
//! `SKIPPED` to report a skip count per board.

/// Reports a test skip with `reason`, writing `SKIPPED: {reason}` directly
/// to stderr (not via `eprintln!`) so libtest's output capture cannot hide
/// it.
#[cfg(test)]
pub(crate) fn report_skip(reason: &str) {
    use std::io::Write;
    let _ = writeln!(&mut std::io::stderr(), "SKIPPED: {reason}");
}

#[path = "../../tensor/tests/support/dma_require.rs"]
#[allow(dead_code)]
mod dma_require;

/// libtest names each test's thread after the test.
fn current_test() -> String {
    std::thread::current()
        .name()
        .unwrap_or("this test")
        .to_string()
}

/// [`edgefirst_tensor::is_dma_available`] under the `HAL_TEST_REQUIRE_DMA=1`
/// policy: on a lane that requires a DMA heap, an unavailable one fails the
/// calling test instead of letting it skip while reporting success.
pub(crate) fn is_dma_available() -> bool {
    dma_require::decide(
        edgefirst_tensor::is_dma_available(),
        dma_require::required(),
        &current_test(),
        "no usable DMA heap (no /dev/dma_heap node, or it is not openable by this user)",
    )
}

/// [`edgefirst_tensor::is_gpu_buffer_available`], under the `HAL_TEST_REQUIRE_DMA=1` policy on Linux
/// where the buffers are DMA-BUFs.
pub(crate) fn is_gpu_buffer_available() -> bool {
    #[cfg(target_os = "linux")]
    return is_dma_available();
    #[cfg(not(target_os = "linux"))]
    edgefirst_tensor::is_gpu_buffer_available()
}

/// Applies the `HAL_TEST_REQUIRE_G2D=1` policy to a G2D availability probe.
///
/// # Panics
///
/// When `HAL_TEST_REQUIRE_G2D=1` and `available` is `false`.
pub(crate) fn g2d_required_or(available: bool) -> bool {
    let required = std::env::var("HAL_TEST_REQUIRE_G2D").as_deref() == Ok("1");
    g2d_decide(available, required, &current_test())
}

fn g2d_decide(available: bool, required: bool, what: &str) -> bool {
    assert!(
        available || !required,
        "HAL_TEST_REQUIRE_G2D=1 but {what} would have skipped while reporting success: \
         libg2d.so.2 did not load or G2D failed to initialize"
    );
    available
}

#[test]
fn g2d_policy_skips_unless_required() {
    assert!(g2d_decide(true, true, "t"));
    assert!(g2d_decide(true, false, "t"));
    assert!(!g2d_decide(false, false, "t"));
    let required_but_missing = std::panic::catch_unwind(|| g2d_decide(false, true, "t"));
    assert!(required_but_missing.is_err());
}
