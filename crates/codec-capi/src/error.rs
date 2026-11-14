// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! Errors at the C boundary: an errno for the caller to branch on, and a
//! thread-local message saying what went wrong.

use std::cell::RefCell;
use std::ffi::{c_char, c_int, CString};
use std::panic::{catch_unwind, AssertUnwindSafe};

use edgefirst_codec::CodecError;

thread_local! {
    static LAST: RefCell<Option<CString>> = const { RefCell::new(None) };
}

pub(crate) fn set_last(msg: &str) {
    let c = CString::new(msg.replace('\0', "?")).unwrap_or_default();
    // A destroyed thread-local at thread exit means nobody is left to read it.
    let _ = LAST.try_with(|l| *l.borrow_mut() = Some(c));
}

fn clear_last() {
    let _ = LAST.try_with(|l| *l.borrow_mut() = None);
}

/// The errno for a codec error.
///
/// `ENOSPC` for a destination too small and `EACCES` for a tensor whose
/// declared CPU access does not cover the work are specific because a caller
/// can act on them; `EAGAIN` says the caller holds every decoded frame.
pub(crate) fn errno_for(e: &CodecError) -> c_int {
    if e.cpu_access_denied().is_some() {
        return libc::EACCES;
    }
    match e {
        CodecError::InsufficientCapacity { .. } => libc::ENOSPC,
        CodecError::UnsupportedDtype(_)
        | CodecError::UnsupportedFormat(_)
        | CodecError::Unsupported(_) => libc::ENOTSUP,
        CodecError::InvalidData(_) => libc::EBADMSG,
        CodecError::Io(io) if io.kind() == std::io::ErrorKind::NotFound => libc::ENOENT,
        CodecError::Io(_) | CodecError::Tensor(_) | CodecError::Device { .. } => libc::EIO,
        CodecError::NoDevice(_) => libc::ENODEV,
        CodecError::InvalidConfig(_) | CodecError::InvalidInput(_) => libc::EINVAL,
        CodecError::Timeout(_) => libc::ETIMEDOUT,
        CodecError::PoolExhausted { .. } => libc::EAGAIN,
    }
}

/// Records `e` as the thread's last error and returns its errno.
pub(crate) fn fail(e: &CodecError) -> c_int {
    set_last(&e.to_string());
    errno_for(e)
}

/// Records why a call failed and returns `rc`.
pub(crate) fn refuse(rc: c_int, why: &str) -> c_int {
    set_last(why);
    rc
}

/// Records a bad argument and returns `EINVAL`.
pub(crate) fn invalid(what: &str) -> c_int {
    refuse(libc::EINVAL, what)
}

/// Gives a failure that recorded no reason a generic one, so a nonzero
/// result always comes with a message.
fn ensure_message(rc: c_int) {
    let missing = LAST.try_with(|l| l.borrow().is_none()).unwrap_or(false);
    if rc != 0 && missing {
        set_last(&format!("failed with errno {rc}"));
    }
}

/// Runs an entry point that returns an errno: clears the last error first,
/// turns a panic into `EINVAL`, and leaves a message for every failure.
pub(crate) fn guard(body: impl FnOnce() -> c_int) -> c_int {
    clear_last();
    let rc = catch_unwind(AssertUnwindSafe(body)).unwrap_or_else(|_| {
        set_last("internal error");
        libc::EINVAL
    });
    ensure_message(rc);
    rc
}

/// Runs an entry point that returns a new object: `NULL` on failure, with
/// `errno` (the last-error code on Windows) and the last error set.
pub(crate) fn guard_new<T>(body: impl FnOnce() -> Result<T, c_int>) -> *mut T {
    clear_last();
    let rc = match catch_unwind(AssertUnwindSafe(body)) {
        Ok(Ok(v)) => return Box::into_raw(Box::new(v)),
        Ok(Err(rc)) => rc,
        Err(_) => {
            set_last("internal error");
            libc::EINVAL
        }
    };
    ensure_message(rc);
    errno::set_errno(errno::Errno(rc));
    std::ptr::null_mut()
}

/// Why the calling thread's last failed codec call failed, or `NULL` when
/// its last call succeeded.
///
/// The text is for people, not for parsing. The pointer stays valid until
/// the same thread's next codec call.
#[no_mangle]
pub extern "C" fn ef_codec_last_error() -> *const c_char {
    LAST.try_with(|l| l.borrow().as_ref().map_or(std::ptr::null(), |c| c.as_ptr()))
        .unwrap_or(std::ptr::null())
}

#[cfg(test)]
mod tests {
    use super::*;
    use edgefirst_codec::{Codec, Direction, Engine, UnsupportedFeature};

    #[test]
    fn each_error_class_has_its_errno() {
        let cases = [
            (CodecError::NoDevice("x".into()), libc::ENODEV),
            (
                CodecError::Unsupported(UnsupportedFeature::Codec {
                    codec: Codec::Hevc,
                    direction: Direction::Encode,
                    engine: Engine::Auto,
                }),
                libc::ENOTSUP,
            ),
            (CodecError::PoolExhausted { held: 4, pool: 4 }, libc::EAGAIN),
            (CodecError::Timeout("x".into()), libc::ETIMEDOUT),
            (CodecError::InvalidConfig("x".into()), libc::EINVAL),
            (CodecError::InvalidInput("x".into()), libc::EINVAL),
            (CodecError::InvalidData("x".into()), libc::EBADMSG),
            (
                CodecError::Io(std::io::Error::from(std::io::ErrorKind::NotFound)),
                libc::ENOENT,
            ),
            (
                CodecError::InsufficientCapacity {
                    image: (2, 2),
                    tensor: (1, 1),
                },
                libc::ENOSPC,
            ),
        ];
        for (e, errno) in cases {
            assert_eq!(errno_for(&e), errno, "{e}");
        }
    }

    #[test]
    fn a_failure_leaves_its_message_and_the_next_call_clears_it() {
        assert_eq!(guard(|| invalid("the thing is NULL")), libc::EINVAL);
        let msg = unsafe { std::ffi::CStr::from_ptr(ef_codec_last_error()) };
        assert_eq!(msg.to_str().unwrap(), "the thing is NULL");
        assert_eq!(guard(|| 0), 0);
        assert!(ef_codec_last_error().is_null());
    }

    #[test]
    fn a_failure_without_a_reason_still_leaves_a_message() {
        assert_eq!(guard(|| libc::EIO), libc::EIO);
        let msg = unsafe { std::ffi::CStr::from_ptr(ef_codec_last_error()) };
        assert_eq!(
            msg.to_str().unwrap(),
            format!("failed with errno {}", libc::EIO)
        );
        let p: *mut u8 = guard_new(|| -> Result<u8, c_int> { Err(libc::ENOTSUP) });
        assert!(p.is_null());
        assert!(!ef_codec_last_error().is_null());
    }

    #[test]
    fn a_panic_is_einval_not_an_abort() {
        assert_eq!(guard(|| panic!("boom")), libc::EINVAL);
        let p: *mut u8 = guard_new(|| -> Result<u8, c_int> { panic!("boom") });
        assert!(p.is_null());
        assert_eq!(errno::errno().0, libc::EINVAL);
    }
}
