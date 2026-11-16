// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! Helpers shared by unit tests.

/// Reports a skipped test on stderr directly, bypassing libtest's output capture.
#[cfg(target_arch = "aarch64")]
pub(crate) fn report_skip(reason: &str) {
    use std::io::Write;
    let _ = writeln!(&mut std::io::stderr(), "WARNING: SKIPPED: {reason}");
}
