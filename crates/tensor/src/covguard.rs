// SPDX-FileCopyrightText: Copyright 2025 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! Coverage-capture resilience for crash-on-shutdown hardware (e.g. the i.MX
//! Vivante EGL driver SIGABRTs during teardown after tests pass). Under
//! coverage instrumentation only, install a SIGABRT handler that flushes the
//! LLVM profile to LLVM_PROFILE_FILE *before* the abort completes, then
//! restores the default disposition and re-raises so the process exit status
//! is unchanged (the CI workflow already treats JUnit as the source of truth).
//!
//! No-op unless built with `-Cinstrument-coverage` (the `coverage` cfg, set by
//! build.rs) on Linux. Never installed in shipped release builds.

/// Install the coverage flush-on-abort handler. Idempotent; safe to call from
/// multiple artifact constructors. No-op outside instrumented Linux builds.
pub fn install() {
    #[cfg(all(coverage, target_os = "linux"))]
    {
        use std::sync::Once;
        static ONCE: Once = Once::new();
        ONCE.call_once(|| unsafe {
            libc::signal(
                libc::SIGABRT,
                flush_then_reraise as extern "C" fn(libc::c_int) as libc::sighandler_t,
            );
        });
    }
}

#[cfg(all(coverage, target_os = "linux"))]
extern "C" fn flush_then_reraise(_sig: libc::c_int) {
    extern "C" {
        // Provided by the LLVM profiling runtime under -Cinstrument-coverage.
        fn __llvm_profile_write_file() -> libc::c_int;
    }
    // SAFETY: async-signal context. The profile writer is the established
    // crash-coverage path; signal()/raise() are async-signal-safe. Best-effort:
    // the heap may already be corrupt, but this is strictly better than losing
    // all coverage. We do NOT _exit(0) — re-raising preserves the exit status.
    unsafe {
        __llvm_profile_write_file();
        libc::signal(libc::SIGABRT, libc::SIG_DFL);
        libc::raise(libc::SIGABRT);
    }
}
