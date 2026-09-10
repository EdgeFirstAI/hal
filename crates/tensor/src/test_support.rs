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
/// Plain `cfg(test)`, not `cfg(all(test, unix))`: this crate's Unix-only
/// call sites (`dma.rs`'s `target_os = "linux"` tests, `lib.rs`'s IOSurface
/// tests under `target_os = "macos"`, its shm tests under plain `unix`)
/// once left this dead code on Windows, but `d3d11/adapter.rs`'s tests
/// (that whole module is `#[cfg(target_os = "windows")]`, see `lib.rs`)
/// are a real Windows-only caller now, so every platform this crate builds
/// for has at least one caller and the narrower gate is no longer needed.
/// `crates/tensor/tests/d3d11_tensor.rs` still carries its own local copy
/// rather than reaching this one -- it's a separate integration-test
/// compilation unit and can't reach a `pub(crate)` item in the lib crate.
#[cfg(test)]
pub(crate) fn report_skip(reason: &str) {
    use std::io::Write;
    let _ = writeln!(&mut std::io::stderr(), "SKIPPED: {reason}");
}
