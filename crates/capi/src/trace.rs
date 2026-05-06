// SPDX-FileCopyrightText: Copyright 2025 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! C-compatible trace capture API.
//!
//! Provides a simple start/stop interface for capturing performance traces
//! from C/C++ applications. The trace output is Chrome JSON format, viewable
//! at <https://ui.perfetto.dev/>.
//!
//! # Usage
//!
//! ```c
//! #include <edgefirst/hal.h>
//!
//! // Start capturing traces to a file
//! int rc = hal_start_tracing("/tmp/trace.json");
//! if (rc != 0) { /* handle error */ }
//!
//! // ... run inference pipeline ...
//!
//! // Stop and flush the trace file
//! hal_stop_tracing();
//! // Trace file is now ready to load in https://ui.perfetto.dev/
//! ```
//!
//! # Build Configuration
//!
//! The tracing feature is enabled by default. When compiled without the
//! `tracing` feature, these functions remain available but return `ENOSYS`
//! to indicate profiling support is not compiled in.

use libc::{c_char, c_int};

/// Start trace capture, writing Chrome JSON to the file at `path`.
///
/// Installs a process-wide tracing subscriber that records all HAL internal
/// spans (decode sub-steps, mask materialization, etc.) to the specified file.
/// The trace can be viewed at https://ui.perfetto.dev/ after calling
/// `hal_stop_tracing()`.
///
/// Only one trace session per process lifetime is supported.
///
/// @param path  Null-terminated UTF-8 file path for the trace output
/// @return 0 on success, -1 on error
/// @par Errors (errno):
/// - EINVAL:   `path` is NULL or not valid UTF-8
/// - EALREADY: a trace session is already active or was previously started and stopped
/// - ENOTSUP: another tracing subscriber was already installed by user code
/// - ENOSYS:  tracing support not compiled in (built without `tracing` feature)
///
/// @par Example
/// @code{.c}
/// hal_start_tracing("/tmp/trace.json");
/// // ... inference pipeline ...
/// hal_stop_tracing();
/// @endcode
#[no_mangle]
pub unsafe extern "C" fn hal_start_tracing(path: *const c_char) -> c_int {
    #[cfg(not(feature = "tracing"))]
    {
        let _ = path;
        errno::set_errno(errno::Errno(libc::ENOSYS));
        -1
    }
    #[cfg(feature = "tracing")]
    {
        if path.is_null() {
            errno::set_errno(errno::Errno(libc::EINVAL));
            return -1;
        }
        let path_str = match std::ffi::CStr::from_ptr(path).to_str() {
            Ok(s) => s,
            Err(_) => {
                errno::set_errno(errno::Errno(libc::EINVAL));
                return -1;
            }
        };
        match edgefirst_hal::trace::start_tracing(path_str) {
            Ok(()) => 0,
            Err(edgefirst_hal::trace::TracingError::AlreadyActive)
            | Err(edgefirst_hal::trace::TracingError::SessionExhausted) => {
                errno::set_errno(errno::Errno(libc::EALREADY));
                -1
            }
            Err(edgefirst_hal::trace::TracingError::SubscriberInstallFailed(_)) => {
                errno::set_errno(errno::Errno(libc::ENOTSUP));
                -1
            }
        }
    }
}

/// Stop trace capture and flush all buffered spans to the output file.
///
/// After this call the trace file is complete and can be loaded into
/// https://ui.perfetto.dev/. No-op if no session is active or if tracing
/// support is not compiled in.
///
/// @par Example
/// @code{.c}
/// hal_start_tracing("/tmp/trace.json");
/// // ... work ...
/// hal_stop_tracing();  // flushes and finalizes the trace file
/// @endcode
#[no_mangle]
pub extern "C" fn hal_stop_tracing() {
    #[cfg(feature = "tracing")]
    edgefirst_hal::trace::stop_tracing();
}

/// Check whether a trace capture session is currently active.
///
/// @return 1 if tracing is active, 0 otherwise (always 0 if tracing not compiled in)
#[no_mangle]
pub extern "C" fn hal_is_tracing_active() -> c_int {
    #[cfg(feature = "tracing")]
    {
        if edgefirst_hal::trace::is_tracing_active() {
            return 1;
        }
    }
    0
}
