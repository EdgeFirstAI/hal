// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! The map window: `ef_tensor_map` / `ef_tensor_try_map` /
//! `ef_tensor_unmap` / `ef_tensor_copy_to`.
//!
//! One CPU-access window per tensor at a time, held for the caller between
//! `ef_tensor_map` and `ef_tensor_unmap` -- the C counterpart to
//! [`edgefirst_tensor::TensorDyn::map_bytes`]'s `HostView` guard, whose
//! `Drop` is the platform sync bracket (mmap unlock, IOSurface unlock,
//! dma-buf sync-for-device, ...). `ef_tensor_copy_to` sidesteps the window
//! entirely: it takes its own short-lived guard, copies, and drops it before
//! returning, so it needs no outstanding map and leaves none behind.
//!
//! Whole-tensor only: a caller wanting one plane's bytes maps the whole
//! tensor and offsets into it using the plane's `offset`/`stride` from
//! `ef_tensor_plane_at` -- there is no per-plane map entry point.

use std::ffi::c_int;
use std::panic::{catch_unwind, AssertUnwindSafe};

use edgefirst_tensor::{CpuAccess, TensorMapTrait};
use edgefirst_tensor_abi::EfTensorView;

use crate::handle::{impl_of, EfTensor};
use crate::last_error::{set_last_error, shield_int};

/// The live state behind an outstanding `ef_tensor_map`.
///
/// Holds the mapping guard itself; there is nothing else to track; `unmap`
/// is exactly "drop this". No lifetime games: `TensorDyn::map_bytes` already
/// returns a genuinely `'static` `HostView` (it shares ownership through the
/// pin's keepalive `Arc` rather than borrowing the tensor -- see that
/// method's doc comment), so this field just holds it as-is.
pub(crate) struct MapState {
    /// The RAII guard whose `Drop` is the platform sync bracket.
    ///
    /// Held, never read: its whole purpose is its `Drop`, same as
    /// `HostPin::keepalive` (`edgefirst-tensor/src/pin.rs`) -- clippy cannot
    /// see that a field exists only for the side effect of going out of
    /// scope.
    #[allow(dead_code)]
    view: edgefirst_tensor::view::HostView<'static, u8>,
}

/// Test-only observation points. `release_own`'s drain of an outstanding
/// map is otherwise invisible from a test (its warning goes to `log`, and
/// the handle is gone the instant it matters), so the drain bumps this
/// counter under `cfg(test)` -- deleting the drain turns the test red
/// instead of leaving the field-order fallback to pass it silently.
#[cfg(test)]
pub(crate) mod test_support {
    use std::sync::atomic::AtomicUsize;
    pub(crate) static FREED_WITH_OUTSTANDING_MAP: AtomicUsize = AtomicUsize::new(0);
}

/// Map a [`edgefirst_tensor::Error`] from the map/copy path to a POSIX
/// errno.
///
/// No shared conversion existed to extend: `builder.rs` and `serialize.rs`
/// each collapse every backend error to one hardcoded code inline
/// (`.map_err(|_| libc::EINVAL)`), which loses exactly the distinction this
/// entry points' contract needs (`EACCES` for a declined access, `ERANGE`
/// for a too-small window, `ENOTSUP` for an unimplemented backend, ...). This
/// is that conversion, written once here rather than inline at each of the
/// three call sites below.
///
/// `pub(crate)`: task 15's new mutators (`mutate.rs`, `quant.rs`) and
/// constructors (`image.rs`) reuse this rather than each collapsing to a
/// single hardcoded code the way `builder.rs`/`serialize.rs` still do --
/// the same reasoning that motivated writing this once applies to every
/// later call site, not just the original three.
pub(crate) fn errno_for(e: &edgefirst_tensor::Error) -> c_int {
    use edgefirst_tensor::Error;
    match e {
        Error::QuantizationInvalid { .. } => libc::EINVAL,
        Error::RegionOutOfBounds { .. } | Error::BatchIndexOutOfBounds { .. } => libc::EINVAL,
        Error::InvalidMemoryType(_) => libc::EINVAL,
        // `InvalidOperation` covers two shapes that deserve different
        // errnos. Misaligned-offset refusals (every backend's `view`/typed
        // map path phrases them "offset N not aligned to align_of") are a
        // caller-argument defect -- EINVAL. Unreachable through this entry
        // point today (`map_bytes` maps the whole extent at offset 0 as
        // `u8`, align 1), but `errno_for` is the shared conversion and a
        // future offset-taking entry point must not inherit the wrong code.
        Error::InvalidOperation(msg) if msg.contains("not aligned") => libc::EINVAL,
        // The remaining operation-refusal arm hard-refused by a live Rust
        // map path today is AHardwareBuffer's "a narrower map is held; this
        // request needs wider access" (`ahardwarebuffer.rs`) -- a platform
        // lock declining the request, the same shape of failure as "this
        // access is not permitted right now". A future backend that hard
        // -refuses a requested access outside the tensor's declared
        // `CpuAccess` (most backends today only log that case as
        // best-effort telemetry, never error) would fall in here too.
        Error::InvalidOperation(_) => libc::EACCES,
        Error::InvalidArgument(_) | Error::InvalidShape(_) | Error::ShapeMismatch(_) => {
            libc::EINVAL
        }
        Error::InvalidSize(_) => libc::EINVAL,
        Error::InsufficientCapacity { .. } => libc::ERANGE,
        Error::NotImplemented(_) => libc::ENOTSUP,
        Error::PboDisconnected | Error::PboMapped => libc::EBUSY,
        // `WouldBlock` is not a failure: it is `ef_tensor_try_map` reporting
        // that the GPU copy the map depends on has not finished and the
        // caller should come back. `EAGAIN` is the POSIX spelling of exactly
        // that, and keeping it out of the `EIO` arm below is what lets a
        // caller tell "retry me" from "this map is broken".
        Error::IoError(e) if e.kind() == std::io::ErrorKind::WouldBlock => libc::EAGAIN,
        Error::IoError(_) => libc::EIO,
        #[cfg(unix)]
        Error::NixError(_) => libc::EIO,
        _ => libc::EIO,
    }
}

/// Map a tensor's whole extent for CPU access.
///
/// Only one map may be outstanding per tensor at a time; a second call
/// before the matching `ef_tensor_unmap` returns `EBUSY`. `access` selects
/// the mapping direction: `EF_CPU_ACCESS_READ`, `_WRITE`, or `_READ_WRITE`.
/// `EF_CPU_ACCESS_NONE` is not a mappable direction.
///
/// A read-only map (`EF_CPU_ACCESS_READ`) still populates `out->ptr` --
/// `ef_tensor_view` is one shared shape for both directions -- but writing
/// through it is a contract violation this C signature cannot itself
/// enforce; the Rust-side guard does enforce it (a debug assertion fires if
/// the guard is ever asked for a mutable slice while read-only), so misuse
/// is a caught bug on the Rust side even though nothing stops the raw
/// pointer write from C directly.
///
/// Exclusive write, CPU-side only: a writable map (`_WRITE`/`_READ_WRITE`)
/// is refused with `EBUSY` unless the tensor's CPU-side handle count
/// (`ef_tensor_retain`/`ef_tensor_free`) is exactly one -- a second EBUSY
/// trigger distinct from the double-map one above, with its own
/// `ef_tensor_last_error_message` text so the two are distinguishable. This
/// is the C surface's honesty limit on exclusivity: the gate sees only the
/// CPU-side handle count, because a refcount cannot see a device -- the
/// GPU/NPU hold no reference of their own, so a dma-buf a device is
/// concurrently writing is not exclusive no matter what this count says.
/// Device-side ordering is not this gate's job; it stays with the fence
/// field.
///
/// @return 0 on success, `EINVAL` (null tensor/out, or bad access code),
///         `EBUSY` (a map is already outstanding on this tensor, or a
///         writable map was requested while the CPU-side handle count is
///         greater than one -- see above), `EACCES` (a writable map was
///         requested on a tensor whose declared CPU access is not writable
///         -- enforced at this boundary; read-direction mismatches follow
///         the Rust layer's warn-and-allow policy instead. Backend-level
///         mapping refusals, e.g. AHardwareBuffer lock exclusivity, also
///         surface as `EACCES`), or another errno translated from the
///         backend's error.
///
/// # Safety
/// `t` must be `NULL` or a live handle; `out` must be writable for one
/// `ef_tensor_view`.
#[no_mangle]
pub unsafe extern "C" fn ef_tensor_map(
    t: *mut EfTensor,
    access: u32,
    out: *mut EfTensorView,
) -> c_int {
    // SAFETY: `t` and `out` carry this function's own contract unchanged.
    unsafe { map_window(t, access, out, false) }
}

/// Non-blocking [`ef_tensor_map`]: identical in every way except that a
/// backing whose map has to wait for a GPU copy answers `EAGAIN` instead of
/// stalling until that copy lands.
///
/// Only the Windows D3D11 texture has such a copy today (its staging
/// refresh). Every other backing -- host memory, shared memory, dma-buf,
/// IOSurface, AHardwareBuffer, PBO -- reaches exactly the code
/// `ef_tensor_map` reaches and can never answer `EAGAIN`, so a caller can
/// use this unconditionally and only ever actually retry where the wait is
/// real.
///
/// `EAGAIN` takes nothing: no map is left outstanding, `out` is untouched,
/// and a later call (this one or `ef_tensor_map`) makes progress. Yield or
/// sleep between attempts: on the WARP software adapter the CPU threads are
/// the GPU, so a tight retry loop starves the copy it is waiting for.
///
/// @return the codes [`ef_tensor_map`] returns, plus `EAGAIN` when the map
///         would have had to wait for a GPU copy.
///
/// # Safety
/// `t` must be `NULL` or a live handle; `out` must be writable for one
/// `ef_tensor_view`.
#[no_mangle]
pub unsafe extern "C" fn ef_tensor_try_map(
    t: *mut EfTensor,
    access: u32,
    out: *mut EfTensorView,
) -> c_int {
    // SAFETY: `t` and `out` carry this function's own contract unchanged.
    unsafe { map_window(t, access, out, true) }
}

/// The shared body of [`ef_tensor_map`] and [`ef_tensor_try_map`].
///
/// Written once rather than twice: the two differ only in which `TensorDyn`
/// mapping call they make at the end, and every gate before it -- the null
/// checks, the access decode, the write-access refusal, the exclusive-write
/// refcount gate, the outstanding-map check -- is a contract both must
/// enforce identically. Two hand-written copies is exactly how one of those
/// gates ends up applying to only one of them.
///
/// # Safety
/// `t` must be `NULL` or a live handle; `out` must be writable for one
/// `ef_tensor_view`.
unsafe fn map_window(
    t: *mut EfTensor,
    access: u32,
    out: *mut EfTensorView,
    non_blocking: bool,
) -> c_int {
    let what = if non_blocking { "try_map" } else { "map" };
    unsafe {
        shield_int(|| {
            if t.is_null() || out.is_null() {
                set_last_error(&format!("{what}: null tensor or out"));
                return libc::EINVAL;
            }
            let Some(access) = crate::codes::cpu_access_from_code(access) else {
                set_last_error(&format!("{what}: unknown access code {access}"));
                return libc::EINVAL;
            };
            let Some(imp) = impl_of(t) else {
                set_last_error(&format!("{what}: could not resolve handle"));
                return libc::EINVAL;
            };
            // Narrow, write-only gate: a read request against a not-writable
            // declaration is NOT refused here -- that mismatch is the Rust
            // layer's own warn-and-allow territory (`note_unplanned_cpu_access`
            // in `edgefirst-tensor`, best-effort telemetry, not an error on most
            // backends). Only a *writable* request against a declaration that
            // does not permit writes is refused, at this boundary, before ever
            // reaching the backend.
            if access.writes() && !imp.inner.cpu_access().writes() {
                set_last_error(&format!(
                    "{what}: write access requested on a non-writable tensor"
                ));
                return libc::EACCES;
            }
            // A poisoned lock means a shielded panic unwound while the lock was
            // held. The slot is a single-assignment `Option` -- no intermediate
            // state a panic could expose -- so recovering the guard is sound,
            // and the alternative (refusing forever) would brick this tensor's
            // CPU access on the first caught panic.
            let mut slot = imp
                .map_state
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            // Exclusive write, CPU-side only (spec Revision 4, "Exclusive write
            // is adopted only as far as it is honest"): a refcount cannot see a
            // device, so a dma-buf the NPU is writing is not exclusive no matter
            // what this count says. This gate refuses a writable map only when
            // more than one CPU-side handle exists; device-side ordering stays
            // the fence field's job, not this gate's.
            if access.writes() {
                let refs = imp.refs.load(std::sync::atomic::Ordering::Acquire);
                if refs > 1 {
                    set_last_error(&format!(
                        "{what}: write map refused: tensor handle is shared (refcount \
                     {refs}); write access requires a unique handle"
                    ));
                    return libc::EBUSY;
                }
            }
            if slot.is_some() {
                set_last_error(&format!(
                    "{what}: a map is already outstanding on this tensor"
                ));
                return libc::EBUSY;
            }
            let mapped = if non_blocking {
                imp.inner.try_map_bytes(access)
            } else {
                imp.inner.map_bytes(access)
            };
            let mut view = match mapped {
                Ok(v) => v,
                Err(e) => {
                    let errno = errno_for(&e);
                    set_last_error(&format!("{what}: {e}"));
                    return errno;
                }
            };
            let len = view.as_slice().len();
            // Writable maps hand out `as_mut_ptr` -- the guard's own DerefMut
            // enforces writability, so this cannot silently succeed on a
            // read-only guard. A read map deliberately does NOT call
            // `as_mut_slice` (that would panic via `assert_map_writable`); it
            // still returns a `*mut u8` because `ef_tensor_view` has one field
            // for both directions, and the read-only contract is documentation
            // plus the Rust-side guard, not something this C struct can express.
            let ptr: *mut u8 = if access.writes() {
                view.as_mut_slice().as_mut_ptr()
            } else {
                view.as_slice().as_ptr() as *mut u8
            };
            *slot = Some(MapState { view });
            *out = EfTensorView { ptr, len };
            0
        })
    }
}

/// Release the outstanding map taken by `ef_tensor_map`.
///
/// Dropping the guard runs the platform sync bracket (mmap stays resident,
/// but e.g. IOSurface/dma-buf run their unlock/sync-for-device here). The
/// pointer handed out by the matching `ef_tensor_map` is invalid the instant
/// this returns 0.
///
/// @return 0 on success, `EINVAL` (null tensor, or no map is outstanding).
///
/// # Safety
/// `t` must be `NULL` or a live handle.
#[no_mangle]
pub unsafe extern "C" fn ef_tensor_unmap(t: *mut EfTensor) -> c_int {
    unsafe {
        shield_int(|| {
            if t.is_null() {
                set_last_error("unmap: null tensor");
                return libc::EINVAL;
            }
            let Some(imp) = impl_of(t) else {
                set_last_error("unmap: could not resolve handle");
                return libc::EINVAL;
            };
            // Poison recovery: same reasoning as in `ef_tensor_map` -- the slot
            // has no intermediate state, and a bricked unmap would leak the
            // outstanding guard's sync bracket forever.
            let mut slot = imp
                .map_state
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            match slot.take() {
                Some(state) => {
                    drop(state); // the sync bracket runs here
                    0
                }
                None => {
                    set_last_error("unmap: no map is outstanding on this tensor");
                    libc::EINVAL
                }
            }
        })
    }
}

/// Acquire the buffer for CPU access -- the standalone cache-maintenance
/// bracket, without a mapping.
///
/// `DMA_BUF_IOCTL_SYNC` with `DMA_BUF_SYNC_START` on Linux; the IOSurface
/// lock on Apple platforms; a no-op for coherent host memory. Pairs with
/// [`ef_tensor_sync_for_device`], which **must** be called with the same
/// `access`: the direction tells the kernel which half of the maintenance
/// this access needs, and a mismatched pair skips one of them (a read-only
/// bracket lets the kernel skip the writeback, a write-only one skips the
/// invalidate).
///
/// Distinct from `ef_tensor_map`, which establishes an address *and* the
/// coherency window together. This is for a caller that already holds the
/// address -- one that mapped once at init and now brackets each frame's
/// CPU access -- so it takes no map state and leaves none behind.
///
/// `EF_CPU_ACCESS_NONE` is not a sync direction and is refused with
/// `EINVAL`, exactly as it is for `ef_tensor_map`.
///
/// @return 0 on success, `EINVAL` (null tensor, or a bad/`NONE` access
///         code), `ENOTSUP` (a backing with no coherency window independent
///         of its map -- PBO, and AHardwareBuffer on Android), or another
///         errno translated from the backend's error.
///
/// # Safety
/// `t` must be `NULL` or a live handle.
#[no_mangle]
pub unsafe extern "C" fn ef_tensor_sync_for_cpu(t: *const EfTensor, access: u32) -> c_int {
    unsafe { sync_bracket(t, access, true) }
}

/// Release the buffer back to the device -- the CPU is done accessing it.
///
/// `DMA_BUF_SYNC_END`. See [`ef_tensor_sync_for_cpu`] for the pairing rule
/// and the direction's meaning; `access` must match the one that opened the
/// bracket.
///
/// @return the same codes as [`ef_tensor_sync_for_cpu`].
///
/// # Safety
/// `t` must be `NULL` or a live handle.
#[no_mangle]
pub unsafe extern "C" fn ef_tensor_sync_for_device(t: *const EfTensor, access: u32) -> c_int {
    unsafe { sync_bracket(t, access, false) }
}

/// The shared body of the two sync entry points.
///
/// Written once rather than twice: the two differ only in which end of the
/// bracket they call, and duplicating the null/access/handle checks is
/// exactly how the two ends drift into accepting different arguments.
/// Not itself `#[no_mangle]` -- see this crate's note in `handle.rs` about
/// cbindgen not expanding macros; a plain helper is invisible to cbindgen
/// too, which is what we want here.
///
/// # Safety
/// `t` must be `NULL` or a live handle.
unsafe fn sync_bracket(t: *const EfTensor, access: u32, to_cpu: bool) -> c_int {
    let what = if to_cpu {
        "sync_for_cpu"
    } else {
        "sync_for_device"
    };
    {
        shield_int(|| {
            if t.is_null() {
                set_last_error(&format!("{what}: null tensor"));
                return libc::EINVAL;
            }
            let Some(access) = crate::codes::cpu_access_from_code(access) else {
                set_last_error(&format!("{what}: unknown or non-directional access code"));
                return libc::EINVAL;
            };
            let Some(inner) = crate::handle::tensor_of(t) else {
                set_last_error(&format!("{what}: could not resolve handle"));
                return libc::EINVAL;
            };
            let r = if to_cpu {
                inner.sync_for_cpu(access)
            } else {
                inner.sync_for_device(access)
            };
            match r {
                Ok(()) => 0,
                Err(e) => {
                    let errno = errno_for(&e);
                    set_last_error(&format!("{what}: {e}"));
                    errno
                }
            }
        })
    }
}

/// Copy a tensor's whole extent into a caller-provided buffer.
///
/// Needs no outstanding `ef_tensor_map`: it takes its own short-lived read
/// guard, copies, and drops the guard before returning. On `edgefirst-tensor`
/// backends today (`Mem`, at minimum -- see the test below) a plain read
/// guard coexists freely with an outstanding stored map, because the
/// underlying platform mapping carries no single-writer lock of its own;
/// this call does not special-case that, so if a future backend's map ever
/// does refuse a second concurrent guard, the refusal surfaces here as
/// whatever errno `errno_for` gives that backend's error (in practice
/// `EACCES` or `EBUSY` depending on the backend), not a hardcoded one.
///
/// @return bytes written (`>= 0`) on success, or a negative errno: `-EINVAL`
///         (null tensor/out), `-ERANGE` (`cap` is smaller than the tensor's
///         byte length), or another negative errno translated from the
///         backend's error.
///
/// # Safety
/// `t` must be `NULL` or a live handle; `out` must be writable for `cap`
/// bytes.
#[no_mangle]
pub unsafe extern "C" fn ef_tensor_copy_to(t: *mut EfTensor, out: *mut u8, cap: usize) -> i64 {
    unsafe {
        // `shield_int` does not fit here -- it returns `c_int`, and this entry
        // point's contract is a signed byte count (`i64`) -- so the Once is
        // driven directly rather than through that wrapper.
        crate::last_error::ensure_hook_installed();
        catch_unwind(AssertUnwindSafe(|| {
            if t.is_null() || out.is_null() {
                set_last_error("copy_to: null tensor or out");
                return -(libc::EINVAL as i64);
            }
            let Some(imp) = impl_of(t) else {
                set_last_error("copy_to: could not resolve handle");
                return -(libc::EINVAL as i64);
            };
            let view = match imp.inner.map_bytes(CpuAccess::Read) {
                Ok(v) => v,
                Err(e) => {
                    let errno = errno_for(&e);
                    set_last_error(&format!("copy_to: {e}"));
                    return -(errno as i64);
                }
            };
            let bytes = view.as_slice();
            if cap < bytes.len() {
                set_last_error("copy_to: buffer is smaller than the tensor's byte length");
                return -(libc::ERANGE as i64);
            }
            std::ptr::copy_nonoverlapping(bytes.as_ptr(), out, bytes.len());
            bytes.len() as i64
            // `view` drops here -- the local guard's sync bracket runs, and it
            // was never stored, so there is nothing for a later `ef_tensor_unmap`
            // to find.
        }))
        .unwrap_or(-(libc::EINVAL as i64))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::handle::{ef_tensor_free, ef_tensor_new, ef_tensor_retain, EfTensorImpl};
    use edgefirst_tensor_abi::EfTensorView;

    fn empty_view() -> EfTensorView {
        EfTensorView {
            ptr: std::ptr::null_mut(),
            len: 0,
        }
    }

    #[test]
    fn a_write_map_on_a_read_only_declared_tensor_is_eacces_but_a_read_map_succeeds() {
        // Cheapest constructor yielding a non-writable declaration: a
        // Read-declared image tensor via `TensorDyn::image`, wrapped into a
        // handle the same way every constructor in `handle.rs` does.
        let t = edgefirst_tensor::TensorDyn::image(
            4,
            4,
            edgefirst_tensor::PixelFormat::Rgba,
            edgefirst_tensor::DType::U8,
            None,
            CpuAccess::Read,
        )
        .expect("Read-declared RGBA image tensor");
        let handle = crate::handle::into_handle(t);

        let mut view = empty_view();
        // The gate is write-only: a read map on the same tensor must succeed.
        assert_eq!(
            unsafe { ef_tensor_map(handle, 1, &mut view) }, // Read
            0,
            "a read map on a Read-declared tensor must succeed"
        );
        assert_eq!(unsafe { ef_tensor_unmap(handle) }, 0);

        // ReadWrite (3) needs the write bit, which this tensor did not
        // declare -- EACCES, without ever reaching the backend's map.
        assert_eq!(
            unsafe { ef_tensor_map(handle, 3, &mut view) },
            libc::EACCES,
            "a writable map on a Read-declared tensor must be refused"
        );
        // The refused request must not have left a stale outstanding map.
        assert_eq!(unsafe { ef_tensor_unmap(handle) }, libc::EINVAL);

        unsafe { ef_tensor_free(handle) };
    }

    #[test]
    fn write_map_requires_a_unique_cpu_side_handle() {
        // Spec Revision 4's exclusive-write rule (docs/superpowers/specs/
        // 2026-08-20-modular-tensor-abi-design.md:416, Resolved decision 8):
        // uniqueness checking applies to the CPU-side handle count only. A
        // second CPU-side reference is enough to refuse a writable map, even
        // though nothing here can or does see a device-side reference.
        let dims = [4u64];
        let t = unsafe { ef_tensor_new(0, dims.as_ptr(), 1) };
        let mut view = empty_view();

        // Two CPU-side handles to the same tensor: refcount 2.
        assert_eq!(unsafe { ef_tensor_retain(t) }, 0);

        assert_eq!(
            unsafe { ef_tensor_map(t, 3, &mut view) }, // ReadWrite
            libc::EBUSY,
            "a writable map must be refused while the CPU-side refcount is > 1"
        );
        let msg =
            unsafe { std::ffi::CStr::from_ptr(crate::last_error::ef_tensor_last_error_message()) };
        assert!(
            msg.to_str().unwrap().contains("shared"),
            "the exclusive-write EBUSY must carry a message distinct from the \
             double-map EBUSY, got: {msg:?}"
        );

        // The gate is write-only: a read map at the same refcount succeeds.
        assert_eq!(
            unsafe { ef_tensor_map(t, 1, &mut view) }, // Read
            0,
            "a read map must succeed regardless of the CPU-side refcount"
        );
        assert_eq!(unsafe { ef_tensor_unmap(t) }, 0);

        // Release one reference, back to refcount 1: the write gate lifts.
        unsafe { ef_tensor_free(t) };
        assert_eq!(
            unsafe { ef_tensor_map(t, 3, &mut view) },
            0,
            "a writable map must succeed once the handle is unique again"
        );
        assert_eq!(unsafe { ef_tensor_unmap(t) }, 0);

        unsafe { ef_tensor_free(t) };
    }

    #[test]
    fn host_view_static_u8_is_send() {
        // The Send question the brief flags: `HostView` holds `Option<HostPin>`
        // (explicitly `unsafe impl Send`/`Sync` in `pin.rs`), a `Vec<usize>`
        // shape, an `Option<usize>` override, a `bool`, and a
        // `PhantomData<u8>` -- every field is `Send` on its own, so
        // `HostView<'static, u8>` is automatically `Send` with no unsafe impl
        // needed. This is a compile-time proof, not a runtime assertion: it
        // exists so that if a future field ever breaks the auto-derive, this
        // fails to *compile* rather than passing silently.
        fn assert_send<T: Send>() {}
        assert_send::<MapState>();
        assert_send::<std::sync::Mutex<Option<MapState>>>();
    }

    #[test]
    fn tensor_dyn_is_send_and_sync() {
        // The map window's thread-safety story leans on this: concurrent C
        // threads calling `ef_tensor_map`/`ef_tensor_copy_to` share
        // `&TensorDyn` through `EfTensorImpl` (`&EfTensor` from C fans out to
        // `&TensorDyn` on every call). This is a compile-time proof, not a
        // runtime assertion, exactly like `host_view_static_u8_is_send`
        // above: it exists so a future `TensorDyn` variant that loses either
        // auto trait fails to *compile* rather than passing silently.
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<edgefirst_tensor::TensorDyn>();
    }

    #[test]
    fn map_write_unmap_map_read_roundtrip() {
        let dims = [2u64, 3];
        let t = unsafe { ef_tensor_new(0, dims.as_ptr(), 2) };
        let mut view = empty_view();
        assert_eq!(unsafe { ef_tensor_map(t, 3, &mut view) }, 0); // ReadWrite
        assert_eq!(view.len, 6);
        for i in 0..6 {
            unsafe { *view.ptr.add(i) = i as u8 };
        }
        assert_eq!(unsafe { ef_tensor_unmap(t) }, 0);
        assert_eq!(unsafe { ef_tensor_map(t, 1, &mut view) }, 0); // Read
        let bytes = unsafe { std::slice::from_raw_parts(view.ptr, view.len) };
        assert_eq!(bytes, &[0, 1, 2, 3, 4, 5]);
        assert_eq!(unsafe { ef_tensor_unmap(t) }, 0);
        unsafe { ef_tensor_free(t) };
    }

    #[test]
    fn map_error_contract() {
        let dims = [4u64];
        let t = unsafe { ef_tensor_new(0, dims.as_ptr(), 1) };
        let mut view = empty_view();
        assert_eq!(unsafe { ef_tensor_map(t, 0, &mut view) }, libc::EINVAL); // None
        assert_eq!(unsafe { ef_tensor_map(t, 7, &mut view) }, libc::EINVAL); // unknown
        assert_eq!(unsafe { ef_tensor_unmap(t) }, libc::EINVAL); // not mapped
        assert_eq!(unsafe { ef_tensor_map(t, 3, &mut view) }, 0);
        assert_eq!(unsafe { ef_tensor_map(t, 1, &mut view) }, libc::EBUSY); // double map
        assert_eq!(unsafe { ef_tensor_unmap(t) }, 0);
        unsafe { ef_tensor_free(t) };
    }

    #[test]
    fn free_with_an_outstanding_map_releases_the_guard_first() {
        use std::sync::atomic::Ordering;
        let before = test_support::FREED_WITH_OUTSTANDING_MAP.load(Ordering::Relaxed);
        let dims = [4u64];
        let t = unsafe { ef_tensor_new(0, dims.as_ptr(), 1) };
        let mut view = empty_view();
        assert_eq!(unsafe { ef_tensor_map(t, 3, &mut view) }, 0);
        unsafe { ef_tensor_free(t) }; // must not crash; guard dropped before the Box
        assert_eq!(
            test_support::FREED_WITH_OUTSTANDING_MAP.load(Ordering::Relaxed),
            before + 1,
            "the last-reference free must have gone through the drain path, \
             not left the guard to the struct's field-order fallback"
        );
    }

    #[test]
    fn a_poisoned_map_state_lock_recovers_instead_of_bricking_the_tensor() {
        // A shielded panic that unwinds while holding `map_state` poisons
        // the mutex. The slot is a single-assignment `Option`, so recovery
        // is sound -- and without it, every later map/unmap on this tensor
        // would fail forever.
        let dims = [4u64];
        let t = unsafe { ef_tensor_new(0, dims.as_ptr(), 1) };
        let addr = t as usize;
        let _ = std::thread::spawn(move || {
            let imp = unsafe { &*(addr as *const EfTensorImpl) };
            let _guard = imp.map_state.lock().unwrap();
            panic!("deliberately poisoning map_state");
        })
        .join(); // the Err is the panic we caused
        let imp = unsafe { &*(t as *const EfTensorImpl) };
        assert!(
            imp.map_state.is_poisoned(),
            "test setup must have poisoned the lock"
        );

        let mut view = empty_view();
        assert_eq!(
            unsafe { ef_tensor_map(t, 3, &mut view) },
            0,
            "a poisoned map-state lock must recover, not brick CPU access"
        );
        assert_eq!(unsafe { ef_tensor_unmap(t) }, 0);
        unsafe { ef_tensor_free(t) };
    }

    #[test]
    fn errno_for_distinguishes_misaligned_offsets_from_access_refusals() {
        use edgefirst_tensor::Error;
        // The two live `InvalidOperation` shapes (both message texts taken
        // verbatim from `edgefirst-tensor`'s map paths): misalignment is a
        // caller-argument defect, an access refusal is a permission one.
        let misaligned =
            Error::InvalidOperation("DMA map: offset 3 not aligned to align_of::<T>()=4".into());
        assert_eq!(errno_for(&misaligned), libc::EINVAL);
        let refused = Error::InvalidOperation(
            "AHardwareBuffer: a narrower map is already held; this request needs wider access"
                .into(),
        );
        assert_eq!(errno_for(&refused), libc::EACCES);
    }

    #[test]
    fn copy_to_needs_no_map_and_reports_short_buffers() {
        let dims = [2u64, 2];
        let t = unsafe { ef_tensor_new(0, dims.as_ptr(), 2) };
        let mut out = [0u8; 4];
        assert_eq!(unsafe { ef_tensor_copy_to(t, out.as_mut_ptr(), 4) }, 4);
        assert_eq!(
            unsafe { ef_tensor_copy_to(t, out.as_mut_ptr(), 2) },
            -(libc::ERANGE as i64)
        );
        unsafe { ef_tensor_free(t) };
    }

    #[test]
    fn null_arguments_are_errors_not_crashes() {
        // Follows the sibling convention in builder.rs/serialize.rs/detect.rs:
        // every null a C caller could plausibly pass is exercised, not just
        // the ones the RED tests happened to cover.
        let mut view = empty_view();
        assert_eq!(
            unsafe { ef_tensor_map(std::ptr::null_mut(), 3, &mut view) },
            libc::EINVAL
        );
        assert_eq!(
            unsafe { ef_tensor_unmap(std::ptr::null_mut()) },
            libc::EINVAL
        );
        assert_eq!(
            unsafe { ef_tensor_copy_to(std::ptr::null_mut(), std::ptr::null_mut(), 0) },
            -(libc::EINVAL as i64)
        );

        let dims = [4u64];
        let t = unsafe { ef_tensor_new(0, dims.as_ptr(), 1) };
        assert_eq!(
            unsafe { ef_tensor_map(t, 3, std::ptr::null_mut()) },
            libc::EINVAL,
            "null out for ef_tensor_map"
        );
        assert_eq!(
            unsafe { ef_tensor_copy_to(t, std::ptr::null_mut(), 4) },
            -(libc::EINVAL as i64),
            "null out for ef_tensor_copy_to"
        );
        unsafe { ef_tensor_free(t) };
    }

    #[test]
    fn dma_buf_map_write_unmap_copy_to_round_trip() {
        // Runtime-gated, not `#[cfg(target_os = "linux")]`: `is_dma_available`
        // is compiled on every platform and returns `false` wherever the
        // dma-heap device is missing or unprivileged (every non-Linux host,
        // and most Linux dev machines), so this test compiles and *runs* --
        // loudly skipped -- everywhere, the same gating pattern
        // `crates/tensor/tests/pin.rs`'s DMA tests use. It is the only path
        // that exercises the map window's `DMA_BUF_IOCTL_SYNC` sync-bracket
        // arm, which a `Mem`-backed tensor never reaches.
        if !edgefirst_tensor::is_dma_available() {
            // Straight to stderr, not `eprintln!`: libtest swallows
            // `eprintln!` output for passing tests (see check_abi.rs's
            // `artifact_is_fresh`, which documents this exact trap), and this
            // is the plan's only executable evidence for the
            // `DMA_BUF_IOCTL_SYNC` sync-bracket arm -- a skip nobody sees is
            // indistinguishable from a pass.
            use std::io::Write;
            let _ = writeln!(
                std::io::stderr(),
                "SKIP: dma_buf_map_write_unmap_copy_to_round_trip -- no usable \
                 dma-heap on this host (missing or unprivileged)"
            );
            return;
        }
        let t = edgefirst_tensor::TensorDyn::new(
            &[4096],
            edgefirst_tensor::DType::U8,
            Some(edgefirst_tensor::TensorMemory::DmaBuf),
            None,
        )
        .expect("alloc a DMA-BUF tensor");
        let handle = crate::handle::into_handle(t);

        let mut view = empty_view();
        assert_eq!(unsafe { ef_tensor_map(handle, 3, &mut view) }, 0); // ReadWrite
        assert_eq!(view.len, 4096);
        for i in 0..4096 {
            unsafe { *view.ptr.add(i) = (i % 256) as u8 };
        }
        // The sync-for-device bracket runs here, inside unmap.
        assert_eq!(unsafe { ef_tensor_unmap(handle) }, 0);

        let mut out = vec![0u8; 4096];
        assert_eq!(
            unsafe { ef_tensor_copy_to(handle, out.as_mut_ptr(), out.len()) },
            4096,
            "copy_to must read back exactly what was written through the map"
        );
        for (i, b) in out.iter().enumerate() {
            assert_eq!(*b, (i % 256) as u8, "byte {i} did not round-trip");
        }

        unsafe { ef_tensor_free(handle) };
    }

    #[test]
    fn copy_to_coexists_with_an_outstanding_stored_map() {
        // The empirical answer the brief asks for: on the `Mem` backend, a
        // local read guard (what `copy_to` takes) coexists freely with an
        // outstanding stored map -- `MemTensor::map_inner` has no
        // single-writer lock of its own (see `mem.rs`), unlike
        // AHardwareBuffer's refcounted CPU lock. So `copy_to` needs no
        // EBUSY special-case for this backend; it just works.
        let dims = [2u64, 2];
        let t = unsafe { ef_tensor_new(0, dims.as_ptr(), 2) };
        let mut view = empty_view();
        assert_eq!(unsafe { ef_tensor_map(t, 3, &mut view) }, 0);
        let mut out = [0u8; 4];
        assert_eq!(
            unsafe { ef_tensor_copy_to(t, out.as_mut_ptr(), 4) },
            4,
            "a local read guard must coexist with the outstanding stored map \
             on the Mem backend"
        );
        assert_eq!(unsafe { ef_tensor_unmap(t) }, 0);
        unsafe { ef_tensor_free(t) };
    }
}
