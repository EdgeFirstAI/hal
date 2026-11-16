// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! PBO-backed tensors: the constructor, and the three accessors a GL
//! consumer reads back.
//!
//! This library makes no GL call. The client passes a callback channel:
//! `ef_client_state` plus the map and unmap functions. This library builds a
//! `TensorStorage::Pbo` tensor over them, and every GL operation calls back
//! out through that channel.
//!
//! # Mapping a PBO handle
//!
//! `ef_tensor_map` is the mapping entry point for a PBO handle, and its
//! one-outstanding-map rule applies: a second concurrent map on the same
//! handle is refused with `EBUSY` and an `EF_ERROR_CLASS_INVALID_OPERATION`
//! last-error class. A caller that wants two simultaneous readers takes two
//! handles, via `ef_tensor_view_region`.

use std::ffi::{c_int, c_void};
use std::panic::{catch_unwind, AssertUnwindSafe};

use edgefirst_tensor::{DType, EfClientState, TensorDyn};
use edgefirst_tensor_abi::{EfErrorClass, EfPboMapFnNullable, EfPboUnmapFnNullable};

use crate::handle::{into_handle, read_dims, tensor_of, EfTensor};
use crate::last_error::{
    class_of, ensure_hook_installed, reclass_last_error, set_errno, set_last_error,
    set_last_error_classified,
};

/// Wrap a client-owned OpenGL Pixel Buffer Object as a tensor.
///
/// `state` is the callback channel this library keeps for the tensor's
/// life: it calls `state.retain` before returning and `state.release` once,
/// when the last reference to the returned tensor is freed. The caller
/// keeps its own reference and may drop it as soon as this returns.
///
/// `retain`/`release` govern the channel only. They do not transfer
/// ownership of the GL buffer: this library never deletes it, and the
/// caller remains the sole caller of `glDeleteBuffers`.
///
/// `size` is the GL allocation's full byte count, which may exceed the
/// product of `dims`.
///
/// `map_fn` and `unmap_fn` are called on the caller's own thread, never on
/// a thread this library creates. They must be safe to call concurrently
/// and must not unwind.
///
/// `state.ctx`, `state.retain`, `state.release`, `map_fn` and `unmap_fn`
/// must all be non-NULL; a NULL one is rejected before anything is
/// constructed.
///
/// @retval a new tensor the caller must free with `ef_tensor_free`.
/// @retval `NULL` for a NULL `state.ctx`/`state.retain`/`state.release`/
///         `map_fn`/`unmap_fn`, a NULL `dims`, `ndim == 0`, an
///         unrecognized `dtype`, or a `size` smaller than the shape needs
///         — `ef_tensor_last_error_message` carries the reason and
///         `ef_tensor_last_error_class` its class. `errno` is set to
///         `EINVAL` for these argument-validation refusals.
///
/// # Safety
/// `dims` must point to `ndim` readable `uint64_t`. `state.ctx` must remain
/// valid until this library's `release` call, and `map_fn`/`unmap_fn`, when
/// non-NULL, must remain callable for the same span.
#[no_mangle]
pub unsafe extern "C" fn ef_tensor_wrap_pbo(
    state: EfClientState,
    buffer_id: u32,
    size: usize,
    dtype: u32,
    dims: *const u64,
    ndim: u32,
    // Nullable: C can pass a NULL function pointer.
    map_fn: EfPboMapFnNullable,
    unmap_fn: EfPboUnmapFnNullable,
) -> *mut EfTensor {
    unsafe {
        ensure_hook_installed();
        catch_unwind(AssertUnwindSafe(|| {
            let Some(shape) = read_dims(dims, ndim, "wrap_pbo") else {
                set_errno(libc::EINVAL);
                reclass_last_error(EfErrorClass::InvalidArgument);
                return std::ptr::null_mut();
            };
            let Some(dt) = DType::from_code(dtype) else {
                set_errno(libc::EINVAL);
                set_last_error_classified(
                    EfErrorClass::InvalidArgument,
                    &format!("wrap_pbo: unknown dtype code {dtype}"),
                );
                return std::ptr::null_mut();
            };
            if state.ctx.is_null() || state.retain.is_none() || state.release.is_none() {
                set_errno(libc::EINVAL);
                set_last_error_classified(
                    EfErrorClass::InvalidArgument,
                    "wrap_pbo: client state must carry a non-NULL ctx, retain and release",
                );
                return std::ptr::null_mut();
            }
            let (Some(map_fn), Some(unmap_fn)) = (map_fn, unmap_fn) else {
                set_errno(libc::EINVAL);
                set_last_error_classified(
                    EfErrorClass::InvalidArgument,
                    "wrap_pbo: map_fn and unmap_fn must both be non-NULL",
                );
                return std::ptr::null_mut();
            };
            // SAFETY: this function's own contract, forwarded verbatim.
            match TensorDyn::from_client_pbo(state, map_fn, unmap_fn, buffer_id, size, &shape, dt) {
                Ok(t) => into_handle(t),
                Err(e) => {
                    set_errno(libc::EINVAL);
                    set_last_error_classified(class_of(&e), &format!("wrap_pbo: {e}"));
                    std::ptr::null_mut()
                }
            }
        }))
        .unwrap_or(std::ptr::null_mut())
    }
}

/// The GL buffer name behind a PBO-backed tensor.
///
/// @retval 0 success; `*out_id` holds the buffer name.
/// @retval EINVAL `t` is NULL/unresolvable, `out_id` is NULL, or `t` is not
///         PBO-backed. Use `ef_tensor_storage_kind` to test whether a tensor
///         is PBO-backed.
///
/// # Safety
/// `t` must be NULL or a live handle; `out_id` must be NULL or a writable
/// `uint32_t`.
#[no_mangle]
pub unsafe extern "C" fn ef_tensor_pbo_id(t: *const EfTensor, out_id: *mut u32) -> c_int {
    unsafe {
        ensure_hook_installed();
        catch_unwind(AssertUnwindSafe(|| {
            if out_id.is_null() {
                set_last_error("pbo_id: null out_id");
                return libc::EINVAL;
            }
            let Some(inner) = tensor_of(t) else {
                set_last_error("pbo_id: could not resolve handle");
                return libc::EINVAL;
            };
            match inner.pbo_id() {
                Some(id) => {
                    *out_id = id;
                    0
                }
                None => {
                    set_last_error("pbo_id: tensor is not PBO-backed");
                    libc::EINVAL
                }
            }
        }))
        .unwrap_or(libc::EINVAL)
    }
}

/// Whether a PBO-backed tensor holds, or is establishing, a CPU mapping.
///
/// @retval 1 mapped, or a map/unmap is in flight.
/// @retval 0 fully unmapped.
/// @retval -1 `t` is NULL/unresolvable or not PBO-backed.
///
/// @warning Three-valued: the error value `-1` is truthy. Test `< 0` before
/// testing for non-zero.
///
/// # Safety
/// `t` must be NULL or a live handle.
#[no_mangle]
pub unsafe extern "C" fn ef_tensor_pbo_is_mapped(t: *const EfTensor) -> c_int {
    ensure_hook_installed();
    catch_unwind(AssertUnwindSafe(|| {
        let Some(inner) = tensor_of(t) else {
            set_last_error("pbo_is_mapped: could not resolve handle");
            return -1;
        };
        match inner.pbo_is_mapped() {
            Some(true) => 1,
            Some(false) => 0,
            None => {
                set_last_error("pbo_is_mapped: tensor is not PBO-backed");
                -1
            }
        }
    }))
    .unwrap_or(-1)
}

/// The address of this PBO's callback vtable, as carried in the
/// cross-package descriptor's `ptr` field for the `PBO` kind.
///
/// Borrowed: valid for as long as `t` is. A consumer that reconstructs
/// operations from it takes its own reference through the `ef_client_state`
/// embedded in it.
///
/// @retval the vtable address.
/// @retval `NULL` when `t` is NULL/unresolvable or not PBO-backed.
///
/// # Safety
/// `t` must be NULL or a live handle.
#[no_mangle]
pub unsafe extern "C" fn ef_tensor_pbo_vtable(t: *const EfTensor) -> *const c_void {
    ensure_hook_installed();
    catch_unwind(AssertUnwindSafe(|| {
        let Some(inner) = tensor_of(t) else {
            set_last_error("pbo_vtable: could not resolve handle");
            return std::ptr::null();
        };
        inner.pbo_vtable_ptr().unwrap_or_else(|| {
            set_last_error("pbo_vtable: tensor is not PBO-backed");
            std::ptr::null()
        })
    }))
    .unwrap_or(std::ptr::null())
}

#[cfg(test)]
mod tests {
    use super::*;
    use edgefirst_tensor::{EfPboMapFn, EfPboUnmapFn, TensorMemory};
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Mutex;

    static RETAINS: AtomicUsize = AtomicUsize::new(0);
    static RELEASES: AtomicUsize = AtomicUsize::new(0);
    static STORAGE: Mutex<Vec<u8>> = Mutex::new(Vec::new());

    /// Serializes the tests that use the shared counters and backing
    /// storage.
    static CHANNEL: Mutex<()> = Mutex::new(());

    fn channel_guard() -> std::sync::MutexGuard<'static, ()> {
        CHANNEL.lock().unwrap_or_else(|e| e.into_inner())
    }

    unsafe extern "C" fn retain(_ctx: *const c_void) {
        RETAINS.fetch_add(1, Ordering::AcqRel);
    }
    unsafe extern "C" fn release(_ctx: *const c_void) {
        RELEASES.fetch_add(1, Ordering::AcqRel);
    }
    unsafe extern "C" fn map(
        _ctx: *const c_void,
        _id: u32,
        size: usize,
        out_ptr: *mut *mut u8,
        out_len: *mut usize,
    ) -> c_int {
        let mut s = STORAGE.lock().expect("lock");
        s.resize(size, 0);
        unsafe {
            *out_ptr = s.as_mut_ptr();
            *out_len = size;
        }
        0
    }
    unsafe extern "C" fn unmap(_ctx: *const c_void, _id: u32) -> c_int {
        0
    }

    /// A non-NULL sentinel `ctx` that `retain`/`release` never dereference.
    static CTX: u8 = 0;

    fn state() -> EfClientState {
        EfClientState {
            ctx: std::ptr::addr_of!(CTX) as *const c_void,
            retain: Some(retain),
            release: Some(release),
        }
    }

    /// The handle reports PBO storage, returns the buffer name, answers
    /// `is_mapped`, exposes a vtable, and holds the channel for its own
    /// lifetime.
    #[test]
    fn wrap_pbo_mints_a_pbo_handle_and_balances_the_channel() {
        let _serialized = channel_guard();
        RETAINS.store(0, Ordering::Release);
        RELEASES.store(0, Ordering::Release);
        let dims: [u64; 1] = [64];
        // SAFETY: `dims` is a live local; the callbacks above are valid.
        let t = unsafe {
            ef_tensor_wrap_pbo(state(), 7, 64, 0, dims.as_ptr(), 1, Some(map), Some(unmap))
        };
        assert!(!t.is_null(), "wrap_pbo must succeed on a valid channel");
        assert_eq!(
            RETAINS.load(Ordering::Acquire),
            1,
            "the library retains once"
        );

        // SAFETY: `t` is a live handle from the call above.
        unsafe {
            assert_eq!(
                crate::handle::ef_tensor_storage_kind(t),
                TensorMemory::Pbo.code(),
                "the handle must report PBO storage, not host memory"
            );
            let mut id = 0u32;
            assert_eq!(ef_tensor_pbo_id(t, &mut id), 0);
            assert_eq!(id, 7);
            assert_eq!(ef_tensor_pbo_is_mapped(t), 0, "a fresh PBO is unmapped");
            assert!(!ef_tensor_pbo_vtable(t).is_null());
            crate::handle::ef_tensor_free(t);
        }
        assert_eq!(
            RELEASES.load(Ordering::Acquire),
            1,
            "the channel is released exactly once, when the handle is freed"
        );
    }

    /// A map through `ef_tensor_map` lands in the client's own storage, via
    /// the client's `map_fn`.
    #[test]
    fn a_wrapped_pbo_maps_through_the_clients_own_callback() {
        let _serialized = channel_guard();
        let dims: [u64; 1] = [64];
        // SAFETY: `dims` is a live local; the callbacks above are valid.
        let t = unsafe {
            ef_tensor_wrap_pbo(state(), 9, 64, 0, dims.as_ptr(), 1, Some(map), Some(unmap))
        };
        assert!(!t.is_null());

        // SAFETY: `t` is a live handle.
        unsafe {
            let mut view = edgefirst_tensor_abi::EfTensorView {
                ptr: std::ptr::null_mut(),
                len: 0,
            };
            assert_eq!(crate::map::ef_tensor_map(t, 3, &mut view), 0, "map a PBO");
            assert_eq!(view.len, 64);
            assert_eq!(
                ef_tensor_pbo_is_mapped(t),
                1,
                "a mapped PBO reports itself mapped"
            );
            *view.ptr = 0xA5;

            // A second concurrent map on the same handle is refused.
            let mut second = edgefirst_tensor_abi::EfTensorView {
                ptr: std::ptr::null_mut(),
                len: 0,
            };
            assert_eq!(
                crate::map::ef_tensor_map(t, 1, &mut second),
                libc::EBUSY,
                "a second map on one PBO handle must be refused, not silently \
                 replace the first"
            );
            assert_eq!(
                crate::last_error::last_class(),
                EfErrorClass::InvalidOperation
            );

            assert_eq!(crate::map::ef_tensor_unmap(t), 0);
            assert_eq!(ef_tensor_pbo_is_mapped(t), 0, "unmapped again");
            crate::handle::ef_tensor_free(t);
        }
        assert_eq!(
            STORAGE.lock().expect("lock")[0],
            0xA5,
            "the write went into the client's own buffer, through map_fn"
        );
    }

    /// A NULL `ctx`, `retain` or `release` is refused with `EINVAL` and the
    /// `InvalidArgument` class.
    #[test]
    fn wrap_pbo_refuses_an_incomplete_channel() {
        let _serialized = channel_guard();
        RETAINS.store(0, Ordering::Release);
        let dims: [u64; 1] = [64];
        let broken = [
            EfClientState {
                ctx: std::ptr::null(),
                ..state()
            },
            EfClientState {
                retain: None,
                ..state()
            },
            EfClientState {
                release: None,
                ..state()
            },
        ];
        for s in broken {
            // SAFETY: `dims` is a live local.
            let t = unsafe {
                ef_tensor_wrap_pbo(s, 7, 64, 0, dims.as_ptr(), 1, Some(map), Some(unmap))
            };
            assert!(t.is_null(), "an incomplete client state must be refused");
            assert_eq!(
                crate::last_error::last_class(),
                EfErrorClass::InvalidArgument
            );
            assert_eq!(errno::errno().0, libc::EINVAL, "must be EINVAL");
        }
        assert_eq!(
            RETAINS.load(Ordering::Acquire),
            0,
            "a refused wrap must not have taken a reference on the channel"
        );
    }

    /// A NULL `map_fn` or `unmap_fn` is refused with `EINVAL`, and no
    /// channel reference is taken.
    #[test]
    fn wrap_pbo_refuses_a_null_op() {
        let _serialized = channel_guard();
        RETAINS.store(0, Ordering::Release);
        let dims: [u64; 1] = [64];
        let cases: [(Option<EfPboMapFn>, Option<EfPboUnmapFn>); 3] =
            [(None, Some(unmap)), (Some(map), None), (None, None)];
        for (m, u) in cases {
            // SAFETY: `dims` is a live local; a NULL op is rejected, not
            // called.
            let t = unsafe { ef_tensor_wrap_pbo(state(), 7, 64, 0, dims.as_ptr(), 1, m, u) };
            assert!(t.is_null(), "a NULL map_fn/unmap_fn must be refused");
            assert_eq!(
                crate::last_error::last_class(),
                EfErrorClass::InvalidArgument
            );
            assert_eq!(errno::errno().0, libc::EINVAL, "errno must say EINVAL");
        }
        assert_eq!(
            RETAINS.load(Ordering::Acquire),
            0,
            "a refused wrap must not have taken a reference on the channel"
        );
    }

    /// A NULL `dims`, a zero `ndim` and an unknown dtype code are each
    /// refused as `InvalidArgument` with `errno == EINVAL`.
    #[test]
    fn wrap_pbo_refuses_malformed_geometry() {
        let _serialized = channel_guard();
        RETAINS.store(0, Ordering::Release);
        let dims: [u64; 1] = [64];
        let cases: [(&str, *const u64, u32, u32, &str); 3] = [
            (
                "null dims",
                std::ptr::null(),
                1,
                0,
                "null dims or zero ndim",
            ),
            ("zero ndim", dims.as_ptr(), 0, 0, "null dims or zero ndim"),
            (
                "unknown dtype code",
                dims.as_ptr(),
                1,
                9999,
                "unknown dtype code 9999",
            ),
        ];
        for (what, d, n, dt, expected_msg) in cases {
            // SAFETY: each call passes either a live `dims` or an explicit
            // NULL, which `read_dims` checks before reading.
            let t = unsafe { ef_tensor_wrap_pbo(state(), 7, 64, dt, d, n, Some(map), Some(unmap)) };
            assert!(t.is_null(), "{what} must be refused");
            assert_eq!(
                crate::last_error::last_class(),
                EfErrorClass::InvalidArgument,
                "{what}: the refusal must be classified, not left Unspecified"
            );
            assert_eq!(errno::errno().0, libc::EINVAL, "{what}: errno");
            // The message is the one the failing layer wrote.
            // SAFETY: the pointer is this thread's own thread-local, read
            // before any further `tensor-capi` call.
            let msg = unsafe {
                std::ffi::CStr::from_ptr(crate::last_error::ef_tensor_last_error_message())
                    .to_string_lossy()
                    .into_owned()
            };
            assert!(
                msg.contains(expected_msg),
                "{what}: expected a message naming {expected_msg:?}, got {msg:?}"
            );
        }
        assert_eq!(
            RETAINS.load(Ordering::Acquire),
            0,
            "a refused wrap must not have taken a reference on the channel"
        );
    }

    /// A `size` smaller than the shape needs, and a shape whose byte
    /// footprint overflows `usize`, are each refused with `EINVAL` and the
    /// tensor layer's class.
    #[test]
    fn wrap_pbo_sets_einval_on_the_refusals_the_tensor_layer_makes() {
        let _serialized = channel_guard();
        RETAINS.store(0, Ordering::Release);
        RELEASES.store(0, Ordering::Release);
        // (what, size, dims, expected class)
        let undersized: [u64; 2] = [8, 8];
        let overflowing: [u64; 2] = [1u64 << 63, 4];
        let cases: [(&str, usize, &[u64], EfErrorClass); 2] = [
            (
                "size below the shape footprint",
                8,
                &undersized,
                EfErrorClass::InvalidShape,
            ),
            (
                "shape footprint overflows usize",
                8,
                &overflowing,
                EfErrorClass::InvalidShape,
            ),
        ];
        for (what, size, dims, class) in cases {
            errno::set_errno(errno::Errno(libc::EEXIST));
            // SAFETY: `dims` is a live local of `ndim` readable entries and
            // the callbacks above are valid.
            let t = unsafe {
                ef_tensor_wrap_pbo(
                    state(),
                    7,
                    size,
                    0,
                    dims.as_ptr(),
                    dims.len() as u32,
                    Some(map),
                    Some(unmap),
                )
            };
            assert!(t.is_null(), "{what} must be refused");
            assert_eq!(
                errno::errno().0,
                libc::EINVAL,
                "{what}: the header promises EINVAL for an argument refusal"
            );
            assert_eq!(
                crate::last_error::last_class(),
                class,
                "{what}: the class the tensor layer recorded must survive"
            );
        }
        // The channel is retained and then released again.
        assert_eq!(
            RETAINS.load(Ordering::Acquire),
            RELEASES.load(Ordering::Acquire),
            "a refused wrap must not strand a reference on the channel"
        );
    }

    /// The three accessors refuse a non-PBO handle; `is_mapped` returns a
    /// negative value.
    #[test]
    fn the_pbo_accessors_refuse_a_host_tensor() {
        let dims: [u64; 1] = [8];
        // SAFETY: `dims` is a live local.
        let t = unsafe { crate::handle::ef_tensor_new(0, dims.as_ptr(), 1) };
        assert!(!t.is_null());
        // SAFETY: `t` is a live handle.
        unsafe {
            let mut id = 0u32;
            assert_eq!(ef_tensor_pbo_id(t, &mut id), libc::EINVAL);
            assert_eq!(ef_tensor_pbo_is_mapped(t), -1);
            assert!(ef_tensor_pbo_vtable(t).is_null());
            crate::handle::ef_tensor_free(t);
        }
    }

    /// Every accessor refuses a NULL handle, and `pbo_id` a NULL `out_id`.
    #[test]
    fn the_pbo_accessors_refuse_null_arguments() {
        // SAFETY: NULL is a permitted argument on all three.
        unsafe {
            let mut id = 0u32;
            assert_eq!(ef_tensor_pbo_id(std::ptr::null(), &mut id), libc::EINVAL);
            assert_eq!(ef_tensor_pbo_is_mapped(std::ptr::null()), -1);
            assert!(ef_tensor_pbo_vtable(std::ptr::null()).is_null());

            let dims: [u64; 1] = [8];
            let t = crate::handle::ef_tensor_new(0, dims.as_ptr(), 1);
            assert_eq!(ef_tensor_pbo_id(t, std::ptr::null_mut()), libc::EINVAL);
            crate::handle::ef_tensor_free(t);
        }
    }
}
