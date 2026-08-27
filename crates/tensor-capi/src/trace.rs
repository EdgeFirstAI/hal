// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! Chrome-JSON trace capture.

use std::ffi::{c_char, c_int, CStr};
use std::panic::{catch_unwind, AssertUnwindSafe};

/// Start a process-wide trace to `path`. Only one session per process.
///
/// # Safety
/// `path` must be a NUL-terminated UTF-8 string.
#[no_mangle]
pub unsafe extern "C" fn ef_start_tracing(path: *const c_char) -> c_int {
    #[cfg(not(feature = "tracing"))]
    {
        let _ = path;
        errno::set_errno(errno::Errno(libc::ENOSYS));
        return -1;
    }
    #[cfg(feature = "tracing")]
    unsafe {
        catch_unwind(AssertUnwindSafe(|| {
            if path.is_null() {
                errno::set_errno(errno::Errno(libc::EINVAL));
                return -1;
            }
            let Ok(path_str) = CStr::from_ptr(path).to_str() else {
                errno::set_errno(errno::Errno(libc::EINVAL));
                return -1;
            };
            match edgefirst_tensor::trace::start_tracing(path_str) {
                Ok(()) => 0,
                Err(edgefirst_tensor::trace::TracingError::AlreadyActive)
                | Err(edgefirst_tensor::trace::TracingError::SessionExhausted) => {
                    errno::set_errno(errno::Errno(libc::EALREADY));
                    -1
                }
                Err(edgefirst_tensor::trace::TracingError::SubscriberInstallFailed(_)) => {
                    errno::set_errno(errno::Errno(libc::ENOTSUP));
                    -1
                }
            }
        }))
        .unwrap_or(-1)
    }
}

/// Stop tracing and flush. No-op if inactive or tracing is not compiled in.
#[no_mangle]
pub extern "C" fn ef_stop_tracing() {
    #[cfg(feature = "tracing")]
    {
        let _ = catch_unwind(edgefirst_tensor::trace::stop_tracing);
    }
}

/// 1 if a session is active, else 0.
#[no_mangle]
pub extern "C" fn ef_is_tracing_active() -> c_int {
    #[cfg(feature = "tracing")]
    {
        catch_unwind(|| i32::from(edgefirst_tensor::trace::is_tracing_active())).unwrap_or(0)
    }
    #[cfg(not(feature = "tracing"))]
    0
}
