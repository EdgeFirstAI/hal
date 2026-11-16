// SPDX-FileCopyrightText: Copyright 2025 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! Coverage capture for hardware that aborts on shutdown. Under coverage
//! instrumentation, installs a SIGABRT handler that flushes the LLVM profile to
//! `LLVM_PROFILE_FILE`, then restores the default disposition and re-raises.
//!
//! No-op unless built with `-Cinstrument-coverage` (the `coverage` cfg, set by
//! build.rs) on Linux.

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
    // SAFETY: runs in async-signal context; the profile write is best-effort,
    // and `signal()`/`raise()` are async-signal-safe.
    unsafe {
        __llvm_profile_write_file();
        libc::signal(libc::SIGABRT, libc::SIG_DFL);
        libc::raise(libc::SIGABRT);
    }
}
