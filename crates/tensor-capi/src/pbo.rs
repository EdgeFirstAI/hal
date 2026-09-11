// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! PBO-backed tensors: the constructor, and the three facts a GL consumer
//! needs back.
//!
//! A GL Pixel Buffer Object cannot move into this library — its map and
//! unmap must run on the client's GL-worker thread, and this library never
//! makes a GL call. What crosses instead is the **callback channel**:
//! `ef_client_state` plus the two op functions. This library builds a real
//! `TensorStorage::Pbo` tensor over them, so a PBO handle behaves like any
//! other tensor — `ef_tensor_view_region`, `ef_tensor_map`,
//! `ef_tensor_set_plane_offset` and the rest all work — while every actual
//! GL operation calls back out.
//!
//! **This is a constructor, not an attach.** The storage *is* the GL
//! buffer; there is no pre-existing tensor to decorate. An
//! `ef_tensor_pbo_attach` that decorated a host tensor would reproduce the
//! two-sources-of-truth shape this design exists to remove, just moved
//! across the ABI.
//!
//! # Mapping a PBO handle
//!
//! `ef_tensor_map` is the mapping entry point for a PBO handle like any
//! other, and its one-outstanding-map rule applies unchanged: a second
//! concurrent map on the same handle is refused with `EBUSY` and an
//! `EF_ERROR_CLASS_INVALID_OPERATION` last-error class, never silently
//! dropped or replaced. The underlying `PboHandle` does support several
//! concurrent readers, but that is reachable only from inside the library;
//! a caller that wants two simultaneous readers takes two handles, via
//! `ef_tensor_view_region`.

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
/// **`retain`/`release` govern the channel only.** They never transfer
/// ownership of the GL buffer: this library never deletes it, and the
/// caller's own destructor stays the sole caller of `glDeleteBuffers`.
/// Getting this backwards double-frees the buffer.
///
/// `size` is the GL allocation's full byte count, which may exceed the
/// product of `dims` — a PBO allocated at a 64-byte-aligned row stride is
/// larger than its shape implies, and clamping to the shape would lose the
/// padding a strided map needs.
///
/// `map_fn` and `unmap_fn` are called on the caller's own thread, never on
/// a thread this library creates. They must be safe to call concurrently
/// and must not unwind.
///
/// **The whole channel is checked, not assumed.** `state.ctx`,
/// `state.retain`, `state.release`, `map_fn` and `unmap_fn` are five
/// non-NULL requirements, and all five are rejected rather than
/// dereferenced: a NULL function pointer this library called would be
/// undefined behaviour, not a diagnosable failure, so the check happens
/// before anything is constructed. The two op parameters are nullable
/// function pointers in C for exactly that reason.
///
/// @retval a new tensor the caller must free with `ef_tensor_free`.
/// @retval `NULL` for a NULL `state.ctx`/`state.retain`/`state.release`/
///         `map_fn`/`unmap_fn`, a NULL `dims`, `ndim == 0`, an
///         unrecognized `dtype`, or a `size` smaller than the shape needs
///         — `ef_tensor_last_error_message` carries the reason and
///         `ef_tensor_last_error_class` its class. `errno` is set to
///         `EINVAL` for the argument-validation refusals, alongside the
///         `NULL` return that is this entry point's contract.
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
    // Nullable: C can pass a NULL function pointer, and reading one as a
    // bare `fn` is undefined behaviour before any check could run. Only the
    // entry point's parameters change -- `PboOpsVtable`'s own fields stay
    // bare, because a table this library built has already been checked.
    // See `EfPboMapFnNullable` for why the alias exists and why the header
    // signature still reads `ef_pbo_map_fn map_fn`.
    map_fn: EfPboMapFnNullable,
    unmap_fn: EfPboUnmapFnNullable,
) -> *mut EfTensor {
    unsafe {
        // The quiet hook, before the catch: a caught panic must WRITE the
        // thread-local, or a consumer reading `ef_tensor_last_error_class`
        // after this returns NULL gets a class left behind by an earlier
        // failure and reports it as this call's. See `ensure_hook_installed`.
        ensure_hook_installed();
        catch_unwind(AssertUnwindSafe(|| {
            let Some(shape) = read_dims(dims, ndim, "wrap_pbo") else {
                // `read_dims` is shared with `ef_tensor_wrap_host` and
                // records an accurate message but no class, so the class is
                // added here rather than in the helper: this entry point's
                // header groups the NULL-`dims`/zero-`ndim` refusal with its
                // other argument refusals in one sentence, and a caller
                // reading `ef_tensor_last_error_class` must find that true.
                // Fixing the helper would change `wrap_host` too -- a
                // pre-existing gap, deliberately left for Stage E.
                //
                // `reclass_last_error`, not `set_last_error_classified`:
                // `read_dims` distinguishes "null dims or zero ndim" from "a
                // dimension is out of range for this host's usize", and
                // restating a message here would collapse the two and
                // sometimes name the wrong one.
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
            // `client_state_pbo_ops` re-checks the three `state` members, but
            // checking here names the entry point in the message and keeps
            // the EINVAL contract in the header true independently of a
            // helper two crates away.
            //
            // Spec section 6 says a NULL `ctx`/`retain`/`release` is
            // `EINVAL`, so `errno` is set as well as the class -- the
            // convention `hardware.rs`, `cuda.rs` and `d3d11.rs` already
            // follow for entry points whose only failure channel is a NULL
            // return. `ef_tensor_wrap_host` sets no errno on any of its
            // refusals; that is a pre-existing gap, out of scope here, and
            // noted for Stage E rather than fixed under a PBO change.
            if state.ctx.is_null() || state.retain.is_none() || state.release.is_none() {
                set_errno(libc::EINVAL);
                set_last_error_classified(
                    EfErrorClass::InvalidArgument,
                    "wrap_pbo: client state must carry a non-NULL ctx, retain and release",
                );
                return std::ptr::null_mut();
            }
            // The two ops are checked HERE and nowhere else: past this point
            // they are bare `fn` values, and a NULL one is undefined
            // behaviour the moment Rust reads it -- not a failure any
            // downstream layer could still diagnose. Nothing is constructed
            // before this returns, so a refusal leaks no channel reference.
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
                    // Every failure `from_client_pbo` can return is an
                    // argument refusal -- an incomplete channel
                    // (`client_state_pbo_ops`), a `size` smaller than the
                    // shape needs, a zero `size`, or a shape footprint that
                    // overflows `usize` (`PboTensor::from_pbo`). The header
                    // above promises `EINVAL` for the argument-validation
                    // refusals and names the `size` one explicitly, so this
                    // arm has to set it too: without it a caller reading
                    // `errno` after a NULL return sees whatever an earlier,
                    // unrelated call left there. The *class* stays the one
                    // the failing layer recorded -- errno is the coarse
                    // channel, `ef_tensor_last_error_class` the precise one.
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
///         PBO-backed. `ef_tensor_storage_kind` is the unambiguous
///         predicate; do not infer the backing from this call's failure.
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

/// Whether a PBO-backed tensor currently holds (or is establishing) a CPU
/// mapping — "is this buffer free for GL operations?", not "does a CPU
/// pointer exist right now".
///
/// @retval 1 mapped, or a map/unmap is in flight.
/// @retval 0 fully unmapped.
/// @retval -1 `t` is NULL/unresolvable or not PBO-backed.
///
/// @warning Three-valued behind an `int`. Test `< 0` **before**
/// truthiness: `if (ef_tensor_pbo_is_mapped(t))` is true for the error
/// value, so a caller that writes it that way treats "not a PBO" as
/// "mapped". The shape is
/// `int r = ef_tensor_pbo_is_mapped(t); if (r < 0) { ...error... } else if
/// (r) { ...mapped... }`.
///
/// # Safety
/// `t` must be NULL or a live handle.
#[no_mangle]
pub unsafe extern "C" fn ef_tensor_pbo_is_mapped(t: *const EfTensor) -> c_int {
    ensure_hook_installed();
    catch_unwind(AssertUnwindSafe(|| {
        // `tensor_of` is safe to *call*; this function stays `unsafe` because
        // its contract on `t` (NULL or a live handle from this library) is
        // what makes that call sound.
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

/// The address of this PBO's callback vtable, for the cross-package
/// descriptor protocol's `ptr` field under its `PBO` kind.
///
/// Borrowed, never owned: valid for as long as `t` is, and a consumer that
/// reconstructs operations from it takes its own reference through the
/// `ef_client_state` embedded in it.
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
        // See `ef_tensor_pbo_is_mapped` on why the body needs no `unsafe`
        // block while the function itself stays `unsafe`.
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
    // The aliases the entry point's signature deliberately spells out: the
    // tests may name them, because nothing here goes through cbindgen.
    use edgefirst_tensor::{EfPboMapFn, EfPboUnmapFn, TensorMemory};
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Mutex;

    static RETAINS: AtomicUsize = AtomicUsize::new(0);
    static RELEASES: AtomicUsize = AtomicUsize::new(0);
    static STORAGE: Mutex<Vec<u8>> = Mutex::new(Vec::new());

    /// The counters and the backing storage above are one shared channel, so
    /// the tests that actually retain it must not overlap. The Makefile runs
    /// this crate with `--test-threads=1`, but a bare `cargo test` does not,
    /// and a test that flakes only outside the Makefile is worse than one
    /// that flakes always. Poison is recovered rather than propagated: a
    /// panic in one test must fail that test, not cascade into the next.
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

    /// A non-NULL, never-dereferenced sentinel: `retain`/`release` above
    /// ignore it, which is exactly what "opaque to the library" means.
    ///
    /// A real static's address rather than a fabricated integer, so a
    /// regression that *did* dereference `ctx` reads a valid byte and fails
    /// an assertion instead of becoming undefined behaviour.
    static CTX: u8 = 0;

    fn state() -> EfClientState {
        EfClientState {
            ctx: std::ptr::addr_of!(CTX) as *const c_void,
            retain: Some(retain),
            release: Some(release),
        }
    }

    /// The constructor round trip: the handle reports PBO storage, hands
    /// back the buffer name, answers `is_mapped`, exposes a vtable, and
    /// holds the channel for exactly its own lifetime.
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

    /// The bytes really are the client's: a map through `ef_tensor_map`
    /// lands in the client's own storage, via the client's `map_fn`. A
    /// constructor that quietly allocated host memory instead would pass
    /// every assertion above and fail this one.
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

            // A second concurrent map on the same handle is refused loudly
            // (controller ruling 2) -- never dropping the live guard.
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

    /// §6: a NULL `ctx`, `retain` or `release` is EINVAL, not a silently
    /// non-owning wrap. `errno` as well as the class, because §6 says
    /// `EINVAL` and this entry point's only other failure channel is the
    /// `NULL` return.
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
            assert_eq!(errno::errno().0, libc::EINVAL, "spec §6 says EINVAL");
        }
        assert_eq!(
            RETAINS.load(Ordering::Acquire),
            0,
            "a refused wrap must not have taken a reference on the channel"
        );
    }

    /// A NULL `map_fn` or `unmap_fn` is `EINVAL`, not a call through a NULL
    /// function pointer.
    ///
    /// This is the arm the `state` check cannot cover: `retain`/`release`
    /// are `Option` fields the library inspects, but the two ops used to be
    /// bare `fn` parameters, so a C caller passing NULL produced a value
    /// Rust may not even materialize — undefined behaviour before any check
    /// could run. They are nullable at the boundary now, and the refusal
    /// happens before anything is constructed, so no channel reference
    /// leaks.
    #[test]
    fn wrap_pbo_refuses_a_null_op() {
        let _serialized = channel_guard();
        RETAINS.store(0, Ordering::Release);
        let dims: [u64; 1] = [64];
        let cases: [(Option<EfPboMapFn>, Option<EfPboUnmapFn>); 3] =
            [(None, Some(unmap)), (Some(map), None), (None, None)];
        for (m, u) in cases {
            // SAFETY: `dims` is a live local; a NULL op is exactly what this
            // entry point contracts to reject rather than call.
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

    /// The malformed-geometry arms: a NULL `dims`, a zero `ndim` and an
    /// unknown dtype code each refuse rather than mint a handle over the
    /// client's buffer with a shape nobody agreed to.
    ///
    /// All three are `InvalidArgument` with `errno == EINVAL`, because the
    /// header groups them with the NULL-channel refusal in one sentence and
    /// a caller reading `ef_tensor_last_error_class` must find that true.
    /// The NULL-`dims` and zero-`ndim` arms go through `read_dims`, which is
    /// shared with `ef_tensor_wrap_host` and records `Unspecified`, so those
    /// are exactly the arms a regression would silently lose.
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
            // The message stays the one the failing layer wrote. The two
            // `read_dims` arms word themselves differently, and classifying
            // by restating a message here would collapse them and sometimes
            // name the wrong refusal -- which is why `wrap_pbo` re-classes
            // rather than re-records.
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

    /// The refusals that come back from the tensor layer rather than from
    /// this file's own argument checks: a `size` smaller than the shape
    /// needs, and a shape whose byte footprint overflows `usize` (`2^63 * 4`
    /// wraps to `0`, which an 8-byte allocation would otherwise satisfy).
    /// Both are argument-validation refusals, so both owe the header's
    /// `EINVAL` — `errno` is seeded with an unrelated value first, so a
    /// passing assertion means this call set it and not some earlier one.
    #[test]
    fn wrap_pbo_sets_einval_on_the_refusals_the_tensor_layer_makes() {
        let _serialized = channel_guard();
        RETAINS.store(0, Ordering::Release);
        RELEASES.store(0, Ordering::Release);
        // (what, size, dims, expected class)
        let undersized: [u64; 2] = [8, 8];
        let overflowing: [u64; 2] = [1u64 << 63, 4];
        let cases: [(&str, usize, &[u64], EfErrorClass); 2] = [
            ("size below the shape footprint", 8, &undersized, EfErrorClass::InvalidShape),
            ("shape footprint overflows usize", 8, &overflowing, EfErrorClass::InvalidShape),
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
        // Unlike the argument refusals above, these two are made *after*
        // `client_state_pbo_ops` has taken its reference, so the channel is
        // retained and then released again -- balanced, not untouched.
        assert_eq!(
            RETAINS.load(Ordering::Acquire),
            RELEASES.load(Ordering::Acquire),
            "a refused wrap must not strand a reference on the channel"
        );
    }

    /// The three accessors refuse a non-PBO handle rather than inventing an
    /// answer, and `is_mapped`'s refusal is negative so a caller testing
    /// `< 0` sees it.
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

    /// A NULL handle and a NULL `out_id` are refused on every accessor, and
    /// none of them dereferences.
    #[test]
    fn the_pbo_accessors_refuse_null_arguments() {
        // SAFETY: NULL is an explicitly permitted argument on all three.
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
