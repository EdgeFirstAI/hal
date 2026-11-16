// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! DMA-BUF test policy.
//!
//! Tests that need a DMA heap return early when `is_dma_available()` is false.
//! `HAL_TEST_REQUIRE_DMA=1` turns that skip into a failure naming the test and
//! the reason.
//!
//! This file has no crate references; each compilation unit that includes it
//! passes its own `is_dma_available()` result in.

/// `true` when `HAL_TEST_REQUIRE_DMA=1` is set.
pub fn required() -> bool {
    std::env::var("HAL_TEST_REQUIRE_DMA").as_deref() == Ok("1")
}

/// The pure decision: `true` when `satisfied`; otherwise a failure or a skip
/// depending on `require`. The caller reports the skip on a `false` return.
///
/// # Panics
///
/// When `require` is `true` and `satisfied` is `false`. The message names
/// `HAL_TEST_REQUIRE_DMA=1`, `what`, and `why`.
pub fn decide(satisfied: bool, require: bool, what: &str, why: &str) -> bool {
    if satisfied {
        return true;
    }
    assert!(
        !require,
        "HAL_TEST_REQUIRE_DMA=1 but {what} would have skipped while \
         reporting success: {why}"
    );
    false
}

/// `true` when the caller should run; `false` when it should return early,
/// after writing `SKIPPED: {what} - {why}` straight to stderr.
///
/// # Panics
///
/// When `HAL_TEST_REQUIRE_DMA=1` and `available` is `false`.
pub fn available_or_skip(available: bool, what: &str) -> bool {
    use std::io::Write;
    let why = "no usable DMA heap (no /dev/dma_heap node, or the node is not \
               openable by this user)";
    let run = decide(available, required(), what, why);
    if !run {
        let _ = writeln!(&mut std::io::stderr(), "WARNING: SKIPPED: {what} - {why}");
    }
    run
}
