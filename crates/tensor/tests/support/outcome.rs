// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! What a test does when it cannot exercise what it tests.
//!
//! A skip is a warning, never a silent pass. [`report_skip`] writes
//! `WARNING: SKIPPED: <reason>` straight to stderr, which neither libtest's
//! nor nextest's capture can swallow, so the line reaches the console, the
//! JUnit `system-err`, `scripts/on-target-run.sh`'s summary and
//! `.github/scripts/test_warnings.py`'s audit.
//!
//! An allocation a test depends on either must succeed ([`Expect::Succeeds`]),
//! when a failure fails the test, or may legitimately fail on this host
//! ([`Expect::MayFail`]), when a failure is a warning and the test skips.
//!
//! This file has no crate references so it can be `#[path]`-included from any
//! crate's library `test_support` module or integration tests.

use std::fmt::Debug;

/// Name of the running test; libtest names each test's thread after it.
pub fn current_test() -> String {
    std::thread::current()
        .name()
        .unwrap_or("this test")
        .to_string()
}

/// Reports that the calling test skips, as an auditable warning.
pub fn report_skip(reason: &str) {
    use std::io::Write;
    let _ = writeln!(&mut std::io::stderr(), "WARNING: SKIPPED: {reason}");
}

/// Whether an allocation is allowed to fail on this host.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Expect {
    /// The host can serve it: a failure is a bug and fails the test.
    Succeeds,
    /// The host may not serve it, for the stated reason: a failure warns and skips.
    MayFail(&'static str),
}

/// A DMA-BUF allocation must succeed when the heap probe said it is usable.
pub fn expect_dma(heap_usable: bool) -> Expect {
    if heap_usable {
        Expect::Succeeds
    } else {
        Expect::MayFail("no usable DMA heap on this host")
    }
}

/// Unwraps an allocation the test depends on, under `expect`.
///
/// Returns `None`, after a [`report_skip`] warning, when the allocation failed
/// and `expect` allows it; the caller then returns early.
///
/// # Panics
///
/// When the allocation failed and `expect` is [`Expect::Succeeds`].
pub fn allocated<T, E: Debug>(result: Result<T, E>, what: &str, expect: Expect) -> Option<T> {
    match (result, expect) {
        (Ok(value), _) => Some(value),
        (Err(e), Expect::Succeeds) => panic!(
            "{}: {what} failed on a host expected to serve it: {e:?}",
            current_test()
        ),
        (Err(e), Expect::MayFail(why)) => {
            report_skip(&format!("{}: {what} failed ({why}): {e:?}", current_test()));
            None
        }
    }
}
