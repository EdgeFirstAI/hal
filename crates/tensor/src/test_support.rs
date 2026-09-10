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
///
/// `cfg(unix)`, not just `cfg(test)`: every current call site
/// (`dma.rs`'s `target_os = "linux"` tests, `lib.rs`'s IOSurface tests
/// under `target_os = "macos"`, and its shm tests under plain `unix`) is
/// reachable only on Unix, so an unconditional `pub(crate) fn` here is
/// dead code on Windows and fails a `-D warnings` build there. Widen this
/// (or add a Windows-specific caller with its own cfg) if a Windows test
/// inside this crate's `src/` ever needs it directly --
/// `crates/tensor/tests/d3d11_tensor.rs` already carries its own local
/// copy for exactly that reason, since an integration-test binary can't
/// reach this `pub(crate)` item anyway.
#[cfg(all(test, unix))]
pub(crate) fn report_skip(reason: &str) {
    use std::io::Write;
    let _ = writeln!(&mut std::io::stderr(), "SKIPPED: {reason}");
}
