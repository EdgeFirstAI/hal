// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! DMA-BUF test policy: a skip must not be able to masquerade as coverage.
//!
//! Tests that need a DMA heap return early when `is_dma_available()` is
//! false and report `ok`. On a host whose heap node exists but is not
//! openable by the test user (GitHub's hosted runners ship
//! `/dev/dma_heap/system` as `root:video 0660`), that is indistinguishable
//! from real coverage. `HAL_TEST_REQUIRE_DMA=1` turns the skip into a
//! failure naming the test and the reason, so a lane that is meant to have
//! a heap cannot silently stop testing it.
//!
//! This file has no crate references so it can be `#[path]`-included from
//! both the library's `test_support` module and the integration tests; each
//! compilation unit supplies its own `is_dma_available()` to [`decide`].

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
/// after writing `SKIPPED: {what} - {why}` straight to stderr so libtest's
/// output capture cannot hide it.
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
        let _ = writeln!(&mut std::io::stderr(), "SKIPPED: {what} - {why}");
    }
    run
}
