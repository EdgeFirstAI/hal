// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! The errno plus thread-local last-error convention shared by every entry point.

use std::cell::RefCell;
use std::ffi::{c_char, c_int, CString};
use std::panic::{catch_unwind, AssertUnwindSafe};

use edgefirst_kernels::KernelError;

thread_local! {
    static LAST: RefCell<Option<CString>> = const { RefCell::new(None) };
}

/// Why an entry point failed, before it is mapped onto errno.
pub enum CError {
    /// A bad argument detected at the C boundary.
    Invalid(&'static str),
    /// An allocation the call needed was refused.
    NoMemory(&'static str),
    /// An error from the kernels library.
    Kernel(KernelError),
}

impl From<KernelError> for CError {
    fn from(e: KernelError) -> Self {
        Self::Kernel(e)
    }
}

impl CError {
    fn errno(&self) -> c_int {
        match self {
            Self::Invalid(_) => libc::EINVAL,
            Self::NoMemory(_) => libc::ENOMEM,
            Self::Kernel(KernelError::Unsupported(_))
            | Self::Kernel(KernelError::ForcedTierUnavailable { .. }) => libc::ENOTSUP,
            Self::Kernel(_) => libc::EINVAL,
        }
    }

    fn message(&self) -> String {
        match self {
            Self::Invalid(m) | Self::NoMemory(m) => (*m).to_string(),
            Self::Kernel(e) => e.to_string(),
        }
    }
}

fn set_last(msg: &str) {
    let c = CString::new(msg.replace('\0', "?")).unwrap_or_default();
    // A destroyed thread-local at thread exit just means there is nobody left to read it.
    let _ = LAST.try_with(|l| *l.borrow_mut() = Some(c));
}

fn clear_last() {
    let _ = LAST.try_with(|l| *l.borrow_mut() = None);
}

fn fail(e: &CError) -> c_int {
    set_last(&e.message());
    errno::set_errno(errno::Errno(e.errno()));
    -1
}

/// Runs an entry point body: clears the last error, maps `Err` onto errno plus a stored
/// message, and turns a panic into `-1` / `EINVAL` / "internal error".
pub fn ffi(body: impl FnOnce() -> Result<(), CError>) -> c_int {
    clear_last();
    match catch_unwind(AssertUnwindSafe(body)) {
        Ok(Ok(())) => 0,
        Ok(Err(e)) => fail(&e),
        Err(_) => fail(&CError::Invalid("internal error")),
    }
}

/// Message of the calling thread's last failed kernels call, or NULL if the last call succeeded.
///
/// The pointer is valid until the same thread's next `ef_*_plan_new*` or `ef_*_plan_run` call,
/// each of which replaces or clears the message; copy the text out if it must outlive that.
///
/// @return A NUL-terminated string, or NULL when there is no error.
#[no_mangle]
pub extern "C" fn ef_kernels_last_error() -> *const c_char {
    catch_unwind(|| {
        LAST.try_with(|l| l.borrow().as_ref().map_or(std::ptr::null(), |c| c.as_ptr()))
            .unwrap_or(std::ptr::null())
    })
    .unwrap_or(std::ptr::null())
}
