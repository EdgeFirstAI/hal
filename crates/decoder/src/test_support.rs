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
///
/// `cfg(target_arch = "aarch64")`, not just `cfg(test)`: this crate's only
/// call site today is `per_scale/kernels/neon_baseline.rs`, whose module
/// declaration in `per_scale/kernels/mod.rs` carries an outer
/// `#[cfg(target_arch = "aarch64")]` (NEON kernels have no other target),
/// so the file compiles only on aarch64. An unconditional `pub(crate) fn`
/// here would be dead code on x86_64 and fail a `-D warnings` build there.
/// Widen this if a non-aarch64 caller is ever added.
#[cfg(all(test, target_arch = "aarch64"))]
pub(crate) fn report_skip(reason: &str) {
    use std::io::Write;
    let _ = writeln!(&mut std::io::stderr(), "SKIPPED: {reason}");
}
