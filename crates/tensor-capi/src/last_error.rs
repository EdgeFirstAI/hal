// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! Advisory last-error detail, and the lazy quiet panic hook.
//!
//! `ef_tensor_last_error_message` follows the `dlerror` convention: the
//! errno return of a `tensor-capi` call stays the contract a caller programs
//! against, and this string is advisory only, a one-line reason meant for a
//! log line, never parsed. It is `""` when nothing has failed, thread-local
//! (each OS thread has its own outstanding message), and valid on the
//! calling thread only until that thread's next failing `tensor-capi` call
//! -- a caller that wants to keep it must copy the bytes out before making
//! another call.
//!
//! Only the entry points introduced during the R3 map/refcount work (the map
//! family in `map.rs`, `ef_tensor_retain` in `handle.rs`) record a message
//! on their failure paths. Retrofitting the pre-R3 surface (the plain
//! accessors, the builder, serialize/detect) is out of scope here and left
//! to Plan R6.
//!
//! # The panic hook
//!
//! A shielded entry point already turns an unwind into a benign errno via
//! `catch_unwind`; without a custom hook, Rust's default hook still prints a
//! backtrace-shaped message to stderr on the way past, which is exactly the
//! noise a C caller embedding this library does not want. So the first
//! shielded catch anywhere in the process installs (once, via
//! [`std::sync::Once`]) a replacement hook that captures the panic payload
//! into [`set_last_error`] instead of printing it, and stays installed for
//! the rest of the process -- it is not scoped back down afterwards.
//!
//! Installation is lazy, on first use, rather than at library load: a
//! library that unconditionally imposes a process-global panic hook the
//! moment it is `dlopen`ed, before any of its functions have even been
//! called, is rude to whatever else lives in the same process. Once
//! installed the hook chains to whatever hook was active before it
//! (typically the default one) only when `EF_PANIC_VERBOSE=1` is set --
//! checked on every panic, not cached at install time, so a caller can flip
//! it on for one debugging run without relinking.

use std::ffi::{c_char, c_int, CString};

use edgefirst_tensor_abi::EfErrorClass;

thread_local! {
    /// The calling thread's most recent failure detail. Defaults to empty;
    /// a successful call never touches this -- only a failing one overwrites
    /// it, so the message from the last *failure* survives across
    /// intervening successes on the same thread.
    static LAST: std::cell::RefCell<CString> = std::cell::RefCell::new(CString::default());

    /// The class of that same failure. Kept beside the message rather than
    /// inside it because the message's own contract is "never parse this"
    /// -- and `ef_tensor_batch` was parsing it, for want of anywhere else
    /// to put the distinction.
    ///
    /// Written by every path that writes `LAST`, always, so the two cannot
    /// describe different failures. [`set_last_error`] resets it to
    /// `Unspecified`; only [`set_last_error_classified`] records a real
    /// class. That direction matters: a stale class left behind by an
    /// earlier failure would be read as *this* call's, which is the
    /// confident-falsehood shape this mechanism exists to remove.
    static LAST_CLASS: std::cell::Cell<EfErrorClass> =
        const { std::cell::Cell::new(EfErrorClass::Unspecified) };
}

/// Record one line of failure detail for the calling thread.
///
/// NUL bytes in `msg` are not a caller bug worth panicking over -- a payload
/// string or a backend error's `Display` could contain one incidentally --
/// so they are sanitized to `?` rather than rejected.
pub(crate) fn set_last_error(msg: &str) {
    set_last_error_classified(EfErrorClass::Unspecified, msg);
}

/// Record one line of failure detail **and its class** for the calling
/// thread.
///
/// Use this wherever a real [`edgefirst_tensor::Error`] is in hand;
/// [`class_of`] does the mapping. Plain [`set_last_error`] stays correct
/// for the ad-hoc argument checks that have no `Error` to classify -- it
/// records `Unspecified`, which is the honest answer and, critically, is
/// still a *write*: it clears any class an earlier failure left behind.
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

/// Map a backend error onto the wire class.
///
/// A deliberate many-to-one: the vocabulary names the distinctions a caller
/// acts on differently, not every Rust variant. `NixError`/`IoError` are
/// both "the syscall behind an allocation or an fd failed", and a consumer
/// does the same thing about either.
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
        E::InvalidOperation(_) | E::PboDisconnected | E::PboMapped => {
            EfErrorClass::InvalidOperation
        }
        E::QuantizationInvalid { .. } => EfErrorClass::QuantizationInvalid,
        E::IoError(_) => EfErrorClass::AllocationFailed,
        #[cfg(unix)]
        E::NixError(_) => EfErrorClass::AllocationFailed,
        // Deliberately exhaustive, with no `_` arm: `Error` is not
        // `#[non_exhaustive]`, so adding a variant should be a compile
        // error here rather than silently classifying as `Unspecified` and
        // costing a consumer the distinction without telling anyone.
    }
}

/// Advisory detail for the calling thread's last failing `tensor-capi` call,
/// `""` if none has failed yet.
///
/// The returned pointer is valid only until this thread's next failing
/// `tensor-capi` call (or the thread's exit); a caller that wants to keep
/// the text must copy it out before calling in again. Never parse this
/// string -- program against the errno return, this is a log line.
///
/// # Safety
/// The returned pointer must not be read after this thread makes another
/// `tensor-capi` call, or after this thread exits.
#[no_mangle]
pub unsafe extern "C" fn ef_tensor_last_error_message() -> *const c_char {
    LAST.with(|l| l.borrow().as_ptr())
}

/// Which *kind* of failure the calling thread's last failing
/// `ef_tensor_*` call was; `EF_ERROR_CLASS_UNSPECIFIED` (0) if none has
/// failed yet, or if that failure recorded no class.
///
/// The companion to [`ef_tensor_last_error_message`], and unlike that
/// string this **is** meant to be programmed against. It exists for the
/// entry points that report failure by returning `NULL`: those have no
/// errno to carry a class, so before this a caller rebuilding a typed error
/// from one had only the advisory message -- and `ef_tensor_batch`'s Rust
/// wrapper really did match on a fragment of it, because there was nowhere
/// else for the distinction to live.
///
/// Same lifetime rules as the message: thread-local, set by every failing
/// call, unchanged by a successful one. Read it immediately after the call
/// that returned `NULL`, before making another.
///
/// # Safety
/// Safe to call at any time; declared `unsafe` only for symmetry with the
/// rest of this ABI.
#[no_mangle]
pub unsafe extern "C" fn ef_tensor_last_error_class() -> u32 {
    LAST_CLASS.with(|c| c.get()) as u32
}

/// Runs the lazy hook install exactly once per process, on the first
/// shielded catch anywhere -- never at library load.
static HOOK_ONCE: std::sync::Once = std::sync::Once::new();

/// Install the quiet panic hook, so a caught panic writes the thread-local
/// instead of printing.
///
/// [`shield_int`] does this for the errno-returning entry points. The
/// **pointer**-returning ones must do it too, and for a sharper reason than
/// tidy stderr: a `NULL`-returning entry carries no errno, so a Rust
/// consumer reads [`ef_tensor_last_error_class`] to learn what failed --
/// and that read is only sound if *every* failure path of that call wrote
/// the thread-local. Without the hook, a caught panic writes nothing, and
/// the consumer then reads a class left behind by some **earlier** failure
/// and reports it as this call's. That is exactly the confident-falsehood
/// shape the class was added to remove, so an unwritten panic path would
/// have reintroduced it through the back door.
///
/// With the hook installed, a caught panic runs [`set_last_error`], which
/// writes the payload as the message and resets the class to `Unspecified`
/// -- "no class recorded", which is the truth.
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
            // Read per invocation, not cached at install time, so a caller
            // can turn this on for one run without relinking.
            if std::env::var("EF_PANIC_VERBOSE").as_deref() == Ok("1") {
                previous(info);
            }
        }));
    });
}

/// Run `f`, catching any unwind and turning it into `EINVAL`.
///
/// Ensures the quiet panic hook is installed *before* invoking `f` --
/// otherwise a panic inside the very call that would have installed it
/// prints the default backtrace-shaped message on its way past. Adopted by
/// the R3 entry points whose C return type is already `c_int`
/// (`ef_tensor_map`, `ef_tensor_unmap`, `ef_tensor_retain`);
/// `ef_tensor_copy_to` returns `i64` and so calls [`ensure_hook_installed`]
/// directly instead of going through this wrapper.
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

        // A subsequent success must not disturb the recorded detail: map a
        // valid access, unmap it, and confirm the message from the earlier
        // failure is still there.
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
        // Drives the shield with a panicking body and asserts the hook
        // recorded the message instead of printing a backtrace (asserted
        // via the thread-local, not by scraping stderr).
        let code = crate::last_error::shield_int(|| panic!("deliberate: boom"));
        assert_eq!(code, libc::EINVAL);
        let msg = unsafe { std::ffi::CStr::from_ptr(super::ef_tensor_last_error_message()) };
        assert!(msg.to_str().unwrap().contains("boom"));
    }

    #[test]
    fn a_caught_panic_resets_the_class_rather_than_leaving_a_stale_one() {
        use edgefirst_tensor_abi::EfErrorClass;
        // Leave a real class behind, as an earlier failing call would.
        super::set_last_error_classified(EfErrorClass::RegionOutOfBounds, "earlier failure");
        assert_eq!(
            unsafe { super::ef_tensor_last_error_class() },
            EfErrorClass::RegionOutOfBounds as u32
        );

        // A panic on a NULL-returning entry must not leave that class
        // standing: a consumer reading it would attribute someone else's
        // failure to this call.
        // What a pointer-returning entry point does: install the hook,
        // then catch. Without the install, nothing writes the thread-local
        // and the stale class above survives.
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
        // A fresh OS thread's thread-local starts at its default, never
        // touched by another test's failures -- proves the "" contract
        // without depending on test execution order.
        let handle = std::thread::spawn(|| {
            let msg = unsafe { std::ffi::CStr::from_ptr(super::ef_tensor_last_error_message()) };
            msg.to_str().unwrap().to_string()
        });
        assert_eq!(handle.join().unwrap(), "");
    }
}
