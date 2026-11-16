// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! Advisory last-error detail, and the lazy quiet panic hook.
//!
//! `ef_tensor_last_error_message` follows the `dlerror` convention: the
//! errno return of a `tensor-capi` call is the contract, and this string is
//! an advisory one-line reason for a log line. It is `""` when nothing has
//! failed, thread-local, and valid on the calling thread only until that
//! thread's next failing `tensor-capi` call.
//!
//! # The panic hook
//!
//! The first shielded catch in the process installs, once, a panic hook that
//! captures the panic payload into [`set_last_error`] instead of printing
//! it. The hook stays installed for the rest of the process. It chains to
//! the hook that was active before it only when `EF_PANIC_VERBOSE=1` is set,
//! which is checked on every panic.

use std::ffi::{c_char, c_int, CString};

use edgefirst_tensor_abi::EfErrorClass;

thread_local! {
    /// The calling thread's most recent failure detail. Defaults to empty;
    /// only a failing call overwrites it.
    static LAST: std::cell::RefCell<CString> = std::cell::RefCell::new(CString::default());

    /// The class of that same failure. Written by every path that writes
    /// `LAST`: [`set_last_error`] resets it to `Unspecified`, and
    /// [`set_last_error_classified`] records a class.
    static LAST_CLASS: std::cell::Cell<EfErrorClass> =
        const { std::cell::Cell::new(EfErrorClass::Unspecified) };
}

/// Record one line of failure detail for the calling thread, with the class
/// `Unspecified`. NUL bytes in `msg` are replaced by `?`.
pub(crate) fn set_last_error(msg: &str) {
    set_last_error_classified(EfErrorClass::Unspecified, msg);
}

/// Record one line of failure detail and its class for the calling thread.
/// NUL bytes in `msg` are replaced by `?`.
pub(crate) fn set_last_error_classified(class: EfErrorClass, msg: &str) {
    let sanitized = if msg.contains('\0') {
        msg.replace('\0', "?")
    } else {
        msg.to_string()
    };
    let c = CString::new(sanitized).unwrap_or_default();
    LAST.with(|l| *l.borrow_mut() = c);
    LAST_CLASS.with(|c| c.set(class));
}

/// Re-class the failure just recorded, keeping its message.
///
/// Must be called immediately after the write it re-classes, on the same
/// thread.
pub(crate) fn reclass_last_error(class: EfErrorClass) {
    LAST_CLASS.with(|c| c.set(class));
}

/// The calling thread's last recorded class, for this crate's own tests.
#[cfg(test)]
pub(crate) fn last_class() -> EfErrorClass {
    LAST_CLASS.with(|c| c.get())
}

/// Set the calling thread's `errno`, for the entry points whose only
/// failure channel is a `NULL` return.
pub(crate) fn set_errno(code: c_int) {
    errno::set_errno(errno::Errno(code));
}

/// Map a backend error onto the wire class.
pub(crate) fn class_of(e: &edgefirst_tensor::Error) -> EfErrorClass {
    use edgefirst_tensor::Error as E;
    match e {
        E::InvalidArgument(_) | E::InvalidMemoryType(_) | E::InvalidSize(_) => {
            EfErrorClass::InvalidArgument
        }
        #[cfg(target_os = "linux")]
        E::UnknownBufferType(_) | E::UnknownDeviceType(..) => EfErrorClass::InvalidArgument,
        E::InvalidShape(_) | E::ShapeMismatch(_) | E::NdArrayError(_) => EfErrorClass::InvalidShape,
        E::InsufficientCapacity { .. } => EfErrorClass::InsufficientCapacity,
        E::BatchIndexOutOfBounds { .. } => EfErrorClass::BatchIndexOutOfBounds,
        E::RegionOutOfBounds { .. } => EfErrorClass::RegionOutOfBounds,
        E::NotImplemented(_) => EfErrorClass::NotSupported,
        E::InvalidOperation(_)
        | E::PboDisconnected
        | E::PboMapped
        | E::PartialWriteRequiresReadWrite(_)
        | E::CpuAccessDenied { .. } => EfErrorClass::InvalidOperation,
        E::QuantizationInvalid { .. } => EfErrorClass::QuantizationInvalid,
        E::IoError(_) => EfErrorClass::AllocationFailed,
        #[cfg(unix)]
        E::NixError(_) => EfErrorClass::AllocationFailed,
    }
}

/// Advisory detail for the calling thread's last failing `tensor-capi` call,
/// `""` if none has failed yet.
///
/// The returned pointer is valid only until this thread's next failing
/// `tensor-capi` call (or the thread's exit); a caller that wants to keep
/// the text must copy it out before calling in again. The string is a log
/// line and is not meant to be parsed.
///
/// # Safety
/// The returned pointer must not be read after this thread makes another
/// `tensor-capi` call, or after this thread exits.
#[no_mangle]
pub unsafe extern "C" fn ef_tensor_last_error_message() -> *const c_char {
    LAST.with(|l| l.borrow().as_ptr())
}

/// Which kind of failure the calling thread's last failing `ef_tensor_*`
/// call was; `EF_ERROR_CLASS_UNSPECIFIED` (0) if none has failed yet, or if
/// that failure recorded no class.
///
/// The companion to [`ef_tensor_last_error_message`]. Unlike that string,
/// this value may be programmed against. It serves the entry points that
/// report failure by returning `NULL`.
///
/// Thread-local, set by every failing call, unchanged by a successful one.
/// Read it immediately after the call that returned `NULL`, before making
/// another.
///
/// # Safety
/// Safe to call at any time.
#[no_mangle]
pub unsafe extern "C" fn ef_tensor_last_error_class() -> u32 {
    LAST_CLASS.with(std::cell::Cell::get) as u32 // NOSONAR
}

/// Runs the hook install once per process.
static HOOK_ONCE: std::sync::Once = std::sync::Once::new();

/// Install the quiet panic hook, once per process: a caught panic records
/// its payload through [`set_last_error`], which resets the class to
/// `Unspecified`, instead of printing.
pub(crate) fn ensure_hook_installed() {
    HOOK_ONCE.call_once(|| {
        let previous = std::panic::take_hook();
        std::panic::set_hook(Box::new(move |info| {
            let payload = info.payload();
            let msg = if let Some(s) = payload.downcast_ref::<&str>() {
                s.to_string()
            } else if let Some(s) = payload.downcast_ref::<String>() {
                s.clone()
            } else {
                "panic".to_string()
            };
            set_last_error(&msg);
            // Read on every panic.
            if std::env::var("EF_PANIC_VERBOSE").as_deref() == Ok("1") {
                previous(info);
            }
        }));
    });
}

/// Run `f`, catching any unwind and turning it into `EINVAL`. Installs the
/// quiet panic hook before invoking `f`.
pub(crate) fn shield_int(f: impl FnOnce() -> c_int) -> c_int {
    ensure_hook_installed();
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(f)).unwrap_or(libc::EINVAL)
}

#[cfg(test)]
mod tests {
    use crate::handle::{ef_tensor_free, ef_tensor_new};
    use edgefirst_tensor_abi::EfTensorView;

    #[test]
    fn a_failing_map_records_detail_and_success_does_not_disturb_it() {
        let dims = [4u64];
        let t = unsafe { ef_tensor_new(0, dims.as_ptr(), 1) };
        let mut view = EfTensorView {
            ptr: std::ptr::null_mut(),
            len: 0,
        };
        assert_eq!(
            unsafe { crate::map::ef_tensor_map(t, 9, &mut view) },
            libc::EINVAL
        );
        let msg = unsafe { std::ffi::CStr::from_ptr(super::ef_tensor_last_error_message()) };
        assert!(msg.to_str().unwrap().contains("access"));

        // A subsequent successful map and unmap leave the recorded detail in
        // place.
        assert_eq!(unsafe { crate::map::ef_tensor_map(t, 1, &mut view) }, 0); // Read
        let msg_after_success =
            unsafe { std::ffi::CStr::from_ptr(super::ef_tensor_last_error_message()) };
        assert!(
            msg_after_success.to_str().unwrap().contains("access"),
            "a successful call must not clear or overwrite the last failure detail"
        );
        assert_eq!(unsafe { crate::map::ef_tensor_unmap(t) }, 0);

        unsafe { ef_tensor_free(t) };
    }

    #[test]
    fn a_shielded_panic_is_captured_quietly() {
        let code = crate::last_error::shield_int(|| panic!("deliberate: boom"));
        assert_eq!(code, libc::EINVAL);
        let msg = unsafe { std::ffi::CStr::from_ptr(super::ef_tensor_last_error_message()) };
        assert!(msg.to_str().unwrap().contains("boom"));
    }

    #[test]
    fn a_caught_panic_resets_the_class_rather_than_leaving_a_stale_one() {
        use edgefirst_tensor_abi::EfErrorClass;
        // Leave a class behind, as an earlier failing call would.
        super::set_last_error_classified(EfErrorClass::RegionOutOfBounds, "earlier failure");
        assert_eq!(
            unsafe { super::ef_tensor_last_error_class() },
            EfErrorClass::RegionOutOfBounds as u32
        );

        // Install the hook, then catch, as a pointer-returning entry point
        // does.
        super::ensure_hook_installed();
        let caught = std::panic::catch_unwind(|| panic!("deliberate: pointer boom"));
        assert!(caught.is_err());
        assert_eq!(
            unsafe { super::ef_tensor_last_error_class() },
            EfErrorClass::Unspecified as u32,
            "a caught panic must reset the class to Unspecified, not leave a stale one"
        );
        let msg = unsafe { std::ffi::CStr::from_ptr(super::ef_tensor_last_error_message()) };
        assert!(msg.to_str().unwrap().contains("pointer boom"));
    }

    #[test]
    fn set_last_error_sanitizes_embedded_nul_bytes() {
        super::set_last_error("bad\0byte");
        let msg = unsafe { std::ffi::CStr::from_ptr(super::ef_tensor_last_error_message()) };
        assert_eq!(msg.to_str().unwrap(), "bad?byte");
    }

    #[test]
    fn no_failure_yet_on_this_thread_reads_as_empty() {
        // A fresh OS thread's thread-local starts at its default.
        let handle = std::thread::spawn(|| {
            let msg = unsafe { std::ffi::CStr::from_ptr(super::ef_tensor_last_error_message()) };
            msg.to_str().unwrap().to_string()
        });
        assert_eq!(handle.join().unwrap(), "");
    }
}
