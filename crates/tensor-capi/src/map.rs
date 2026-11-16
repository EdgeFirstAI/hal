// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! The map window: `ef_tensor_map` / `ef_tensor_try_map` /
//! `ef_tensor_unmap` / `ef_tensor_copy_to`.
//!
//! One CPU-access window per tensor at a time, held for the caller between
//! `ef_tensor_map` and `ef_tensor_unmap`. `ef_tensor_copy_to` takes its own
//! short-lived guard, copies, and drops it before returning.
//!
//! The window covers the whole tensor: a caller wanting one plane's bytes
//! offsets into it using the plane's `offset`/`stride` from
//! `ef_tensor_plane_at`.

use std::ffi::c_int;
use std::panic::{catch_unwind, AssertUnwindSafe};

use edgefirst_tensor::{CpuAccess, TensorMapTrait};
use edgefirst_tensor_abi::{EfErrorClass, EfTensorView};

use crate::handle::{impl_of, EfTensor};
use crate::last_error::{class_of, set_last_error, set_last_error_classified, shield_int};

/// The live state behind an outstanding `ef_tensor_map`: the mapping guard.
pub(crate) struct MapState {
    /// The RAII guard whose `Drop` runs the platform sync bracket.
    #[allow(dead_code)]
    view: edgefirst_tensor::view::HostView<'static, u8>,
}

/// Test-only counter of tensors freed with an outstanding map, incremented
/// by `release_own`.
#[cfg(test)]
pub(crate) mod test_support {
    use std::sync::atomic::AtomicUsize;
    pub(crate) static FREED_WITH_OUTSTANDING_MAP: AtomicUsize = AtomicUsize::new(0);
}

/// Map a [`edgefirst_tensor::Error`] from the map/copy path to a POSIX
/// errno.
pub(crate) fn errno_for(e: &edgefirst_tensor::Error) -> c_int {
    use edgefirst_tensor::Error;
    if e.cpu_access_denied().is_some() {
        return libc::EACCES;
    }
    match e {
        Error::QuantizationInvalid { .. } => libc::EINVAL,
        Error::RegionOutOfBounds { .. } | Error::BatchIndexOutOfBounds { .. } => libc::EINVAL,
        Error::InvalidMemoryType(_) => libc::EINVAL,
        // A misaligned-offset refusal is `EINVAL`.
        Error::InvalidOperation(msg) if msg.contains("not aligned") => libc::EINVAL,
        // Every other operation refusal is `EACCES`.
        Error::InvalidOperation(_) => libc::EACCES,
        Error::PartialWriteRequiresReadWrite(_) => libc::EACCES,
        Error::InvalidArgument(_) | Error::InvalidShape(_) | Error::ShapeMismatch(_) => {
            libc::EINVAL
        }
        Error::InvalidSize(_) => libc::EINVAL,
        Error::InsufficientCapacity { .. } => libc::ERANGE,
        Error::NotImplemented(_) => libc::ENOTSUP,
        Error::PboDisconnected | Error::PboMapped => libc::EBUSY,
        // `WouldBlock` is `ef_tensor_try_map` reporting that the GPU copy
        // the map depends on has not finished.
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
/// A read-only map (`EF_CPU_ACCESS_READ`) populates `out->ptr`; the caller
/// must not write through it.
///
/// A writable map (`_WRITE`/`_READ_WRITE`) is refused with `EBUSY` unless
/// the tensor's CPU-side handle count (`ef_tensor_retain`/`ef_tensor_free`)
/// is exactly one. `ef_tensor_last_error_message` distinguishes this from
/// the double-map `EBUSY`. The count covers CPU-side handles only; it does
/// not order access against a device writing the same buffer.
///
/// @return 0 on success, `EINVAL` (null tensor/out, or bad access code),
///         `EBUSY` (a map is already outstanding on this tensor, or a
///         writable map was requested while the CPU-side handle count is
///         greater than one), `EACCES` (the requested direction is not
///         covered by the tensor's declared CPU access, see
///         `ef_tensor_cpu_access`, or the backend refused the mapping), or
///         another errno translated from the backend's error.
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

/// Non-blocking [`ef_tensor_map`]: identical except that a backing whose
/// map has to wait for a GPU copy returns `EAGAIN` instead of waiting.
///
/// Only the Windows D3D11 texture has such a copy. Every other backing
/// behaves as `ef_tensor_map` and never returns `EAGAIN`.
///
/// On `EAGAIN` no map is left outstanding and `out` is untouched. Yield or
/// sleep between attempts.
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
            // Refuse a direction the declared CPU access does not cover.
            if let Err(e) = imp.inner.cpu_access().require(access) {
                set_last_error_classified(class_of(&e), &format!("{what}: {e}"));
                return errno_for(&e);
            }
            // A poisoned lock is recovered.
            let mut slot = imp
                .map_state
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            // A writable map requires a single CPU-side handle.
            if access.writes() {
                let refs = imp.refs.load(std::sync::atomic::Ordering::Acquire);
                if refs > 1 {
                    set_last_error_classified(
                        EfErrorClass::InvalidOperation,
                        &format!(
                            "{what}: write map refused: tensor handle is shared (refcount \
                     {refs}); write access requires a unique handle"
                        ),
                    );
                    return libc::EBUSY;
                }
            }
            if slot.is_some() {
                set_last_error_classified(
                    EfErrorClass::InvalidOperation,
                    &format!("{what}: a map is already outstanding on this tensor"),
                );
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
/// Runs the platform sync bracket (for example the IOSurface unlock or the
/// dma-buf sync-for-device). The pointer returned by the matching
/// `ef_tensor_map` is invalid once this returns 0.
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
            // A poisoned lock is recovered.
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

/// Acquire the buffer for CPU access: the cache-maintenance bracket,
/// without a mapping.
///
/// `DMA_BUF_IOCTL_SYNC` with `DMA_BUF_SYNC_START` on Linux; the IOSurface
/// lock on Apple platforms; a no-op for coherent host memory. Pairs with
/// [`ef_tensor_sync_for_device`], which must be called with the same
/// `access`.
///
/// Unlike `ef_tensor_map`, this takes no map state and returns no address.
///
/// `EF_CPU_ACCESS_NONE` is not a sync direction and is refused with
/// `EINVAL`.
///
/// @return 0 on success, `EINVAL` (null tensor, or a bad/`NONE` access
///         code), `EACCES` (the direction is not covered by the tensor's
///         declared CPU access; see `ef_tensor_cpu_access`), `ENOTSUP` (a
///         backing with no coherency window independent of its map -- PBO,
///         and AHardwareBuffer on Android), or another errno translated
///         from the backend's error.
///
/// # Safety
/// `t` must be `NULL` or a live handle.
#[no_mangle]
pub unsafe extern "C" fn ef_tensor_sync_for_cpu(t: *const EfTensor, access: u32) -> c_int {
    unsafe { sync_bracket(t, access, true) }
}

/// Release the buffer back to the device after CPU access.
///
/// `DMA_BUF_SYNC_END` on Linux. `access` must match the one passed to the
/// [`ef_tensor_sync_for_cpu`] that opened the bracket.
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
/// guard, copies, and drops the guard before returning.
///
/// @return bytes written (`>= 0`) on success, or a negative errno: `-EINVAL`
///         (null tensor/out), `-EACCES` (the tensor's declared CPU access
///         has no read direction), `-ERANGE` (`cap` is smaller than the
///         tensor's byte length), or another negative errno translated from
///         the backend's error.
///
/// # Safety
/// `t` must be `NULL` or a live handle; `out` must be writable for `cap`
/// bytes.
#[no_mangle]
pub unsafe extern "C" fn ef_tensor_copy_to(t: *mut EfTensor, out: *mut u8, cap: usize) -> i64 {
    unsafe {
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
            // `view` drops here, running its sync bracket.
        }))
        .unwrap_or(-(libc::EINVAL as i64))
    }
}

/// Runs `f(ctx)` while no tensor CPU mapping can be created in this process.
///
/// Every tensor mapping this library creates waits until `f` returns; `f`
/// must not map a tensor itself. `f` is called exactly once, on the calling
/// thread; a NULL `f` is a no-op.
///
/// # Safety
///
/// `f`, when non-NULL, must be safe to call with `ctx`.
#[no_mangle]
pub unsafe extern "C" fn ef_tensor_with_cpu_mappings_excluded(
    f: Option<unsafe extern "C" fn(ctx: *mut std::ffi::c_void)>,
    ctx: *mut std::ffi::c_void,
) {
    if let Some(f) = f {
        // SAFETY: the caller guarantees `f` accepts `ctx`.
        edgefirst_tensor::pin::with_cpu_mappings_excluded(|| unsafe { f(ctx) });
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
        assert_eq!(
            unsafe { ef_tensor_map(handle, 1, &mut view) }, // Read
            0,
            "a read map on a Read-declared tensor must succeed"
        );
        assert_eq!(unsafe { ef_tensor_unmap(handle) }, 0);

        // ReadWrite (3).
        assert_eq!(
            unsafe { ef_tensor_map(handle, 3, &mut view) },
            libc::EACCES,
            "a writable map on a Read-declared tensor must be refused"
        );
        // The refused request leaves no outstanding map.
        assert_eq!(unsafe { ef_tensor_unmap(handle) }, libc::EINVAL);

        unsafe { ef_tensor_free(handle) };
    }

    #[test]
    fn a_read_map_on_a_write_or_none_declared_tensor_is_eacces() {
        for declared in [CpuAccess::Write, CpuAccess::None] {
            let t = edgefirst_tensor::TensorDyn::image(
                4,
                4,
                edgefirst_tensor::PixelFormat::Rgba,
                edgefirst_tensor::DType::U8,
                Some(edgefirst_tensor::TensorMemory::Mem),
                declared,
            )
            .expect("image tensor");
            let handle = crate::handle::into_handle(t);
            assert_eq!(
                unsafe { crate::handle::ef_tensor_cpu_access(handle) },
                declared.code()
            );
            let mut view = empty_view();
            assert_eq!(
                unsafe { ef_tensor_map(handle, 1, &mut view) }, // Read
                libc::EACCES,
                "a read map on a {declared:?}-declared tensor must be refused"
            );
            let msg = unsafe {
                std::ffi::CStr::from_ptr(crate::last_error::ef_tensor_last_error_message())
            };
            assert!(
                msg.to_str().unwrap().contains("CPU access denied"),
                "{msg:?}"
            );
            assert_eq!(unsafe { ef_tensor_unmap(handle) }, libc::EINVAL);
            unsafe { ef_tensor_free(handle) };
        }
    }

    fn declared_handle(declared: CpuAccess) -> *mut EfTensor {
        let t = edgefirst_tensor::TensorDyn::image(
            4,
            4,
            edgefirst_tensor::PixelFormat::Rgba,
            edgefirst_tensor::DType::U8,
            Some(edgefirst_tensor::TensorMemory::Mem),
            declared,
        )
        .expect("image tensor");
        crate::handle::into_handle(t)
    }

    #[test]
    fn write_map_on_read_or_none_is_eacces() {
        for declared in [CpuAccess::Read, CpuAccess::None] {
            let handle = declared_handle(declared);
            for access in [CpuAccess::Write, CpuAccess::ReadWrite] {
                let mut view = empty_view();
                assert_eq!(
                    unsafe { ef_tensor_map(handle, access.code(), &mut view) },
                    libc::EACCES,
                    "{access:?} map on a {declared:?}-declared tensor"
                );
                assert_eq!(unsafe { ef_tensor_unmap(handle) }, libc::EINVAL);
            }
            unsafe { ef_tensor_free(handle) };
        }
    }

    #[test]
    fn sync_outside_declaration_is_eacces() {
        let handle = declared_handle(CpuAccess::Read);
        unsafe {
            assert_eq!(
                ef_tensor_sync_for_cpu(handle, CpuAccess::Write.code()),
                libc::EACCES
            );
            assert_eq!(
                ef_tensor_sync_for_device(handle, CpuAccess::ReadWrite.code()),
                libc::EACCES
            );
            assert_eq!(ef_tensor_sync_for_cpu(handle, CpuAccess::Read.code()), 0);
            assert_eq!(ef_tensor_sync_for_device(handle, CpuAccess::Read.code()), 0);
            ef_tensor_free(handle);
        }
        let write_only = declared_handle(CpuAccess::Write);
        unsafe {
            assert_eq!(
                ef_tensor_sync_for_cpu(write_only, CpuAccess::Read.code()),
                libc::EACCES
            );
            ef_tensor_free(write_only);
        }
    }

    #[test]
    fn write_map_requires_a_unique_cpu_side_handle() {
        let dims = [4u64];
        let t = unsafe { ef_tensor_new(0, dims.as_ptr(), 1) };
        let mut view = empty_view();

        // Two CPU-side handles to the same tensor.
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

        // A read map at the same refcount succeeds.
        assert_eq!(
            unsafe { ef_tensor_map(t, 1, &mut view) }, // Read
            0,
            "a read map must succeed regardless of the CPU-side refcount"
        );
        assert_eq!(unsafe { ef_tensor_unmap(t) }, 0);

        // Release one reference, back to refcount 1.
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
        // Compile-time check that `MapState` is `Send`.
        fn assert_send<T: Send>() {}
        assert_send::<MapState>();
        assert_send::<std::sync::Mutex<Option<MapState>>>();
    }

    #[test]
    fn tensor_dyn_is_send_and_sync() {
        // Compile-time check that `TensorDyn` is `Send + Sync`.
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
        unsafe { ef_tensor_free(t) }; // the guard is dropped before the Box
        assert_eq!(
            test_support::FREED_WITH_OUTSTANDING_MAP.load(Ordering::Relaxed),
            before + 1,
            "the last-reference free must have gone through the drain path, \
             not left the guard to the struct's field-order fallback"
        );
    }

    #[test]
    fn a_poisoned_map_state_lock_recovers_instead_of_bricking_the_tensor() {
        // A panic while holding `map_state` poisons the mutex.
        let dims = [4u64];
        let t = unsafe { ef_tensor_new(0, dims.as_ptr(), 1) };
        let addr = t as usize;
        let _ = std::thread::spawn(move || {
            let imp = unsafe { &*(addr as *const EfTensorImpl) };
            let _guard = imp.map_state.lock().unwrap();
            panic!("deliberately poisoning map_state");
        })
        .join(); // the Err is the spawned thread's panic
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
        // Skipped at runtime where the dma-heap device is unavailable.
        if !crate::dma_require::available_or_skip(
            edgefirst_tensor::is_dma_available(),
            "dma_buf_map_write_unmap_copy_to_round_trip",
        ) {
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
        // The sync-for-device bracket runs inside unmap.
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

    /// A second concurrent map on one handle returns `EBUSY` with the
    /// `InvalidOperation` class, and the outstanding guard is neither
    /// dropped nor replaced.
    #[test]
    fn a_second_map_on_one_handle_is_refused_with_a_classified_error() {
        let dims = [2u64, 2];
        let t = unsafe { ef_tensor_new(0, dims.as_ptr(), 2) };
        let mut first = empty_view();
        assert_eq!(unsafe { ef_tensor_map(t, 1, &mut first) }, 0);

        let mut second = empty_view();
        assert_eq!(
            unsafe { ef_tensor_map(t, 1, &mut second) },
            libc::EBUSY,
            "a second outstanding map on one handle must be refused"
        );
        assert_eq!(
            crate::last_error::last_class(),
            edgefirst_tensor_abi::EfErrorClass::InvalidOperation,
            "the refusal must be classified, not left Unspecified for the \
             caller to parse out of the message"
        );
        assert!(
            second.ptr.is_null() && second.len == 0,
            "the refused map must not hand back a view"
        );

        // The first map still unmaps.
        assert_eq!(unsafe { ef_tensor_unmap(t) }, 0);
        unsafe { ef_tensor_free(t) };
    }

    /// A write map on a shared handle returns `EBUSY` with the
    /// `InvalidOperation` class.
    #[test]
    fn a_write_map_on_a_shared_handle_is_refused_with_a_classified_error() {
        let dims = [2u64, 2];
        let t = unsafe { ef_tensor_new(0, dims.as_ptr(), 2) };
        let second_ref = unsafe { crate::handle::ef_tensor_retain(t) };
        assert_eq!(second_ref, 0, "retain must succeed");

        let mut view = empty_view();
        assert_eq!(
            unsafe { ef_tensor_map(t, 2, &mut view) },
            libc::EBUSY,
            "a writable map on a shared handle must be refused"
        );
        assert_eq!(
            crate::last_error::last_class(),
            edgefirst_tensor_abi::EfErrorClass::InvalidOperation,
        );

        unsafe { ef_tensor_free(t) };
        unsafe { ef_tensor_free(t) };
    }

    unsafe extern "C" fn count_call(ctx: *mut std::ffi::c_void) {
        // SAFETY: the test passes a live `u32`.
        unsafe { *ctx.cast::<u32>() += 1 };
    }

    #[test]
    fn mapping_exclusion_runs_the_callback_once() {
        let mut calls = 0u32;
        // SAFETY: `count_call` takes a `*mut u32`, which `ctx` is.
        unsafe {
            ef_tensor_with_cpu_mappings_excluded(Some(count_call), (&mut calls as *mut u32).cast())
        };
        assert_eq!(calls, 1);
    }

    #[test]
    fn mapping_exclusion_accepts_a_null_callback() {
        // SAFETY: a NULL callback is a no-op.
        unsafe { ef_tensor_with_cpu_mappings_excluded(None, std::ptr::null_mut()) };
    }
}
