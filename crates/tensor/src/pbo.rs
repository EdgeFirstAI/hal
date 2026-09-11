// SPDX-FileCopyrightText: Copyright 2025 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

use crate::{
    error::{Error, Result},
    BufferIdentity, IdentityKind, TensorMemory, TensorTrait,
};
use log::trace;
use num_traits::Num;
use std::{
    ffi::c_void,
    fmt,
    marker::PhantomData,
    ops::Deref,
    ptr::NonNull,
    sync::{Arc, Condvar, Mutex},
};
// Used by `PboOpsVtable` and friends (the cross-cdylib capsule protocol's
// PBO support), which are compiled into both backends since task P2a — see
// that type's own doc comment.
use std::os::raw::c_int;
// `PboHandle::abi_vtable`'s cell, and so `static`-only with it: under
// `dynamic` the library owns the table (`ef_tensor_pbo_vtable`).
#[cfg(feature = "static")]
use std::sync::OnceLock;
// The frozen client-state vocabulary the table is expressed over. Re-exported
// (`crate::EfClientState` and friends) so a C entry point assembling the same
// three parts names one type, not a copy of it.
pub use edgefirst_tensor_abi::{EfClientState, EfPboMapFn, EfPboUnmapFn};

/// The refusal both backends give when asked to pin a PBO's host address.
///
/// One string, two arms: `Tensor::pin_host` (`lib.rs`, `static`) refuses its
/// own `TensorStorage::Pbo`, and `TensorDyn::pin_host`
/// (`tensor_dyn/dynamic_backend.rs`) has to repeat the refusal because the
/// ABI exposes `ef_tensor_map` and no `ef_tensor_pin_host`, so the library
/// cannot tell a pin from a map. Shared as a constant rather than copied so
/// the two cannot drift into refusing with different words.
///
/// The reason, at length: `glMapBufferRange`'s pointer is valid only until
/// `glUnmapBuffer`, and this backend maps with `MAP_READ_BIT |
/// MAP_WRITE_BIT` -- no `GL_MAP_PERSISTENT_BIT`, and no `glBufferStorage`
/// anywhere. Holding the map open for a pin's lifetime would keep the buffer
/// mapped across GL work, which is exactly what a PBO is not for. Persistent
/// mapping (`EXT_buffer_storage` on GLES) would make this implementable, but
/// that is a buffer-allocation feature, not wiring -- it changes how every
/// PBO is created and needs fences for coherency.
pub(crate) const PIN_HOST_PBO_REFUSAL: &str = "pin_host: a PBO has no host address outside \
     glMapBufferRange/glUnmapBuffer, so it cannot hand out one that \
     outlives a guard. Use map()/map_with(). Persistent mapping \
     (EXT_buffer_storage, GL_MAP_PERSISTENT_BIT) is not used by this \
     backend.";

/// Raw mapped pointer from a PBO. CPU-accessible while the buffer is mapped.
/// The pointer is only valid between map and unmap calls.
pub struct PboMapping {
    pub ptr: *mut u8,
    pub size: usize,
}

// SAFETY: PboMapping is only created by PboOps::map_buffer which runs on the
// GL thread, but the resulting pointer is used on the caller's thread. This is
// safe because glMapBufferRange returns a CPU-visible pointer that can be
// accessed from any thread while the buffer remains mapped.
unsafe impl Send for PboMapping {}

/// Trait for PBO GL operations, implemented by the image crate.
///
/// All methods are blocking — they send commands to the GL thread
/// and wait for completion. Implementations must ensure GL context
/// is current on the thread that executes the actual GL calls.
///
/// # Safety
///
/// Implementations must ensure:
/// - `map_buffer` returns a valid, aligned pointer to `size` bytes of
///   CPU-accessible memory that remains valid until `unmap_buffer` is called.
/// - `unmap_buffer` invalidates the pointer and releases the mapping.
/// - `delete_buffer` frees the GL buffer resources.
pub unsafe trait PboOps: Send + Sync {
    /// Map the PBO for CPU read/write access.
    /// The returned PboMapping is valid until `unmap_buffer` is called.
    ///
    /// A returned `PboMapping::size` below `size` is treated as a FAILED
    /// map, not a partial one: `PboHandle::acquire_map` refuses it and
    /// unmaps the buffer. Callers of the resulting pin size their host
    /// slice from the tensor's own allocation, so there is nowhere for a
    /// shorter mapping to be recorded. Report the shortfall as an `Err`
    /// rather than a short success if the implementation can tell.
    fn map_buffer(&self, buffer_id: u32, size: usize) -> Result<PboMapping>;

    /// Unmap a previously mapped PBO. Must be called before GL operations
    /// on this buffer (GLES 3.0 requirement).
    fn unmap_buffer(&self, buffer_id: u32) -> Result<()>;

    /// Delete the PBO. Fire-and-forget — no reply needed.
    /// Called from PboTensor's Drop impl.
    fn delete_buffer(&self, buffer_id: u32);
}

/// C-ABI-safe callback table for a PBO's GL operations — the same
/// technique `tensor-capi`'s `EfTensorVtable` uses to let one
/// independently-compiled `cdylib` dispatch into a tensor implementation it
/// did not compile, applied here to [`PboOps`].
///
/// **Why `PboOps` itself cannot cross a `cdylib` boundary.** `Arc<dyn
/// PboOps>`'s vtable pointer addresses a table Rust generates per
/// (concrete-type, trait) pair at compile time — its exact layout is not
/// part of any stable ABI. Two independently-compiled `cdylib`s (e.g.
/// `edgefirst.image`'s and `edgefirst.codec`'s Python extension modules,
/// each statically linking its own copy of this crate — confirmed by their
/// separate `.so` files on disk, one `edgefirst-tensor` copy embedded in
/// each) are not guaranteed to agree on that layout, and Rust's own "no
/// stable ABI" position is exactly why `tensor-capi` exists at all for
/// tensors. A same-process-only registry (buffer_id -> `Arc<dyn PboOps>`,
/// the design this replaces) does not solve this: it would need to be a
/// process-wide `static`, and `crates/tensor/tests/no_global_state.rs`
/// exists specifically to reject that class of state, with the exact
/// cross-package PBO scenario named in its own docstring as the motivating
/// example. `#[repr(C)]` plus `extern "C" fn` pointers are the one
/// representation the platform C ABI guarantees is stable across separate
/// compilations, so that is what crosses instead.
///
/// **Lifetime, and why there is no registry, no leak, and nothing to
/// unregister.** A `PboOpsVtable` is built once, lazily, as a field of the
/// `PboHandle` whose `ops` it dispatches into (see [`PboTensor::pbo_vtable`]),
/// and its address is only ever handed out through
/// [`crate::TensorDesc::ptr`] under [`crate::protocol::kind::PBO`] — valid
/// for exactly as long as that field's existing contract already promises
/// (the producer's capsule keepalive holding the producing tensor alive;
/// see [`crate::TensorDyn::import_descriptor`]'s doc comment, which already
/// establishes this for every other kind's use of `ptr`). This struct's own
/// address IS the `PboHandle`'s address plus a fixed field offset, so it
/// disappears exactly when `PboHandle` does — no separate allocation to
/// leak, no registry entry to remove on drop. The external contract only has
/// to hold for the *read*: an importer calls `state.retain` while reading
/// the table and holds its own reference from then on (see
/// [`client_state_pbo_ops`]).
///
/// Compiled into both backends. It was `static`-only while the
/// cross-package capsule protocol had no `dynamic` counterpart; task P2a
/// gave it one (`TensorDyn::descriptor_pinned`/`import_descriptor` now
/// exist on both), so the gate came off. Nothing here is
/// backend-specific — it is a C-ABI view onto `PboOps`, which was always
/// a backend-agnostic GL extension point.
#[repr(C)]
pub struct PboOpsVtable {
    /// The callback channel: an opaque `ctx` plus the pair that extends its
    /// life. `ctx` addresses the producing [`PboHandle`]; only this module's
    /// own `extern "C"` functions ever dereference it.
    ///
    /// **Borrowed, not owned.** [`Self::new`] takes no reference of its own,
    /// because this struct lives in a `OnceLock` *inside* the very
    /// `PboHandle` its `ctx` names — a count taken here could never reach
    /// zero and every PBO would leak its GL buffer. An importer takes its
    /// own reference through `state.retain` instead; see
    /// [`client_state_pbo_ops`].
    state: EfClientState,
    map_buffer_fn: EfPboMapFn,
    unmap_buffer_fn: EfPboUnmapFn,
    // No `delete_buffer_fn`: deliberately absent, not merely unused. A
    // consumer reconstructing a `PboTensor` from this vtable does not own
    // the GL buffer -- it holds the channel alive through `state.retain`
    // and lets the producer's own `PboHandle::Drop` do the deleting (see
    // `ImportedPboOps`'s own doc comment) -- so there is no legitimate
    // caller for it on that side. Including a pointer nothing may safely
    // call is worse than omitting it.
}

// SAFETY: every field is either a raw pointer this struct never
// dereferences itself (only its own `extern "C"` fns do, and those forward
// into `PboOps`, itself `Send + Sync`) or a plain function pointer — both
// safe to share and move across threads.
unsafe impl Send for PboOpsVtable {}
unsafe impl Sync for PboOpsVtable {}

impl PboOpsVtable {
    /// Build a vtable dispatching into `handle`'s `ops`.
    ///
    /// `ctx` is the `PboHandle`'s own address, taken with `Arc::as_ptr` so
    /// `retain`/`release` can drive its strong count — a pointer into the
    /// `ops` field (what this used to store) is not a value
    /// `Arc::increment_strong_count` accepts. No reference is taken here;
    /// see the field's own doc comment for why that would leak.
    ///
    /// `static`-only: under `dynamic` the library builds and owns the table.
    #[cfg(feature = "static")]
    fn new(handle: &Arc<PboHandle>) -> Self {
        PboOpsVtable {
            state: EfClientState {
                ctx: Arc::as_ptr(handle) as *const c_void,
                retain: Some(vt_retain),
                release: Some(vt_release),
            },
            map_buffer_fn: vt_map_buffer,
            unmap_buffer_fn: vt_unmap_buffer,
        }
    }
}

/// `0` success; `-1` the GL context is gone ([`Error::PboDisconnected`]);
/// any other value a generic failure. This boundary is internal to the
/// cross-package protocol (not `tensor-capi`'s public C ABI), so it does
/// not need POSIX errno parity — only enough to distinguish "gone" (retry
/// is pointless) from "the underlying `PboOps` call itself returned `Err`".
/// `0` success; `-1` the GL context is gone ([`Error::PboDisconnected`]);
/// any other value a generic failure, deliberately without the real
/// variant or message. This boundary is internal to the cross-package
/// protocol (not `tensor-capi`'s public C ABI), so it does not need POSIX
/// errno parity -- only enough to distinguish "gone" (retry is pointless)
/// from "the underlying `PboOps` call itself returned some other `Err`".
///
/// **Tried and reverted: a `tensor-capi`-style thread-local "last error"
/// string.** `tensor-capi`'s `ef_tensor_last_error_message` works precisely
/// because `libedgefirst_tensor.so` is the ONE shared library every
/// consumer dynamically links -- one compiled copy, one thread-local slot,
/// genuinely shared. `edgefirst-tensor` is the opposite: this crate is
/// statically linked separately into every consumer (the whole reason this
/// vtable exists at all), so a `thread_local!` declared here gets a
/// SEPARATE storage slot per linked copy -- the producer's `vt_map_buffer`
/// (compiled into the producer's `.so`) would write into ITS copy's slot,
/// and the consumer's `vtable_errno_to_error` (compiled into a DIFFERENT
/// `.so`) would read its OWN, never-written slot. It would silently work
/// in a same-process test and silently do nothing across the real `.so`
/// boundary this whole file exists for -- worse than the coarse errno it
/// would replace. A real fix needs the message to travel as plain data
/// (an out-param the producer writes into caller-owned memory, with its
/// own ownership/lifetime convention) -- real engineering, not the "cheap"
/// fix this was flagged as; left as coarse on purpose rather than shipping
/// something that looks richer and is not.
fn vtable_errno_to_error(rc: c_int) -> Error {
    match rc {
        -1 => Error::PboDisconnected,
        n => Error::NotImplemented(format!("cross-package PBO operation failed: errno {n}")),
    }
}

fn error_to_vtable_errno(e: &Error) -> c_int {
    match e {
        Error::PboDisconnected => -1,
        _ => 1,
    }
}

/// The `extern "C"` half of [`PboOpsVtable::new`]'s two dispatch
/// pointers — generic over any `Arc<dyn PboOps>`, so no producer needs to
/// write its own. `catch_unwind`-shielded: a panic unwinding across an
/// `extern "C"` boundary is undefined behaviour, the same reasoning
/// `tensor-capi`'s own exported functions are built around.
unsafe extern "C" fn vt_map_buffer(
    ctx: *const c_void,
    buffer_id: u32,
    size: usize,
    out_ptr: *mut *mut u8,
    out_len: *mut usize,
) -> c_int {
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        // SAFETY: the caller's contract (`client_state_pbo_ops`, or this
        // crate's own `PboOpsVtable`) guarantees `ctx` still addresses a
        // live `PboHandle`.
        let handle = unsafe { &*(ctx as *const PboHandle) };
        handle.ops.map_buffer(buffer_id, size)
    }));
    match result {
        Ok(Ok(mapping)) => {
            // SAFETY: `out_ptr`/`out_len` are valid out-params for the
            // duration of this call, per this function's own contract.
            unsafe {
                *out_ptr = mapping.ptr;
                *out_len = mapping.size;
            }
            0
        }
        Ok(Err(e)) => error_to_vtable_errno(&e),
        Err(_) => 1, // shielded panic: report a generic failure, not UB
    }
}

unsafe extern "C" fn vt_unmap_buffer(ctx: *const c_void, buffer_id: u32) -> c_int {
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        // SAFETY: see `vt_map_buffer`.
        let handle = unsafe { &*(ctx as *const PboHandle) };
        handle.ops.unmap_buffer(buffer_id)
    }));
    match result {
        Ok(Ok(())) => 0,
        Ok(Err(e)) => error_to_vtable_errno(&e),
        Err(_) => 1,
    }
}

/// The `retain` half of [`PboOpsVtable::new`]'s [`EfClientState`].
///
/// Compiled into the producing copy of this crate, which is the only one
/// that knows `PboHandle` — the whole reason `ctx` is opaque everywhere
/// else. `catch_unwind`-shielded like its siblings: an unwind across an
/// `extern "C"` boundary is undefined behaviour, and an `Arc` refcount
/// operation cannot fail in any other way, so there is nothing to report.
unsafe extern "C" fn vt_retain(ctx: *const c_void) {
    let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        // SAFETY: `ctx` came from `Arc::as_ptr` on a live `Arc<PboHandle>`
        // (or from a matching `retain` the caller has not released yet),
        // per `EfClientState`'s own contract.
        unsafe { Arc::increment_strong_count(ctx as *const PboHandle) }
    }));
}

/// The `release` half. See [`vt_retain`].
unsafe extern "C" fn vt_release(ctx: *const c_void) {
    let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        // SAFETY: balances exactly one prior `vt_retain`.
        unsafe { Arc::decrement_strong_count(ctx as *const PboHandle) }
    }));
}

/// Adapts a [`PboOpsVtable`]'s three parts back into a [`PboOps`]
/// implementation, so [`PboTensor::from_pbo`] needs no separate
/// cross-package constructor.
///
/// **Holds the channel open.** Construction calls `state.retain` and
/// [`Drop`] calls `state.release`, so this value keeps the producing
/// `PboHandle` alive by itself. That is what lets a child outlive its
/// parent, and it is the difference from the design this replaces, where
/// the lifetime was enforced externally by a `pbo_keepalive()` `Arc` some
/// unrelated object had to hold. The Python capsule still holds that
/// keepalive; it is now belt-and-braces rather than the only thing standing
/// between a consumer and freed memory.
///
/// `delete_buffer` is deliberately a no-op: this tensor does not own the GL
/// buffer, it holds the *channel* to it (see [`EfClientState`]'s own
/// contract), so dropping it must not delete a buffer the producer's own
/// tensor may still be using. The producer's own `PboHandle::Drop` is the
/// only thing that ever calls the real `delete_buffer`.
struct ImportedPboOps {
    state: EfClientState,
    map_buffer_fn: EfPboMapFn,
    unmap_buffer_fn: EfPboUnmapFn,
}

// SAFETY: the parts this holds are a raw pointer it never dereferences
// itself (only the function pointers do, and those forward into a real
// `PboOps` implementation, itself `Send + Sync`) plus plain function
// pointers, safe to call from any thread — see `PboOps`'s own trait-level
// contract, which the real implementation behind them already upholds.
unsafe impl Send for ImportedPboOps {}
unsafe impl Sync for ImportedPboOps {}

impl Drop for ImportedPboOps {
    fn drop(&mut self) {
        if let Some(release) = self.state.release {
            // SAFETY: balances the `retain` in `client_state_pbo_ops`,
            // which is this type's only constructor.
            unsafe { release(self.state.ctx) };
        }
    }
}

// SAFETY: forwards every call through the client's own op functions, which
// forward into a real `PboOps` implementation upholding this trait's
// contract already — this wrapper adds no unsafety beyond the calls
// `client_state_pbo_ops`'s own safety contract covers.
unsafe impl PboOps for ImportedPboOps {
    fn map_buffer(&self, buffer_id: u32, size: usize) -> Result<PboMapping> {
        let mut ptr: *mut u8 = std::ptr::null_mut();
        let mut len: usize = 0;
        // SAFETY: `client_state_pbo_ops` retained the channel, so `ctx` is
        // live for as long as `self` is.
        let rc =
            unsafe { (self.map_buffer_fn)(self.state.ctx, buffer_id, size, &mut ptr, &mut len) };
        if rc != 0 {
            return Err(vtable_errno_to_error(rc));
        }
        Ok(PboMapping { ptr, size: len })
    }

    fn unmap_buffer(&self, buffer_id: u32) -> Result<()> {
        // SAFETY: see `map_buffer` above.
        let rc = unsafe { (self.unmap_buffer_fn)(self.state.ctx, buffer_id) };
        if rc != 0 {
            return Err(vtable_errno_to_error(rc));
        }
        Ok(())
    }

    fn delete_buffer(&self, _buffer_id: u32) {
        // No-op: see this type's own doc comment.
    }
}

/// The three parts of a [`PboOpsVtable`], read out of a descriptor's `ptr`
/// or supplied directly by a C caller.
pub struct PboOpsVtableParts {
    pub state: EfClientState,
    pub map_fn: EfPboMapFn,
    pub unmap_fn: EfPboUnmapFn,
}

/// Read a [`crate::TensorDesc::ptr`]'s vtable into its parts, so a caller
/// can hand them to a constructor rather than reassembling a
/// `dyn PboOps` first.
///
/// # Safety
/// `vtable_ptr` must be a table this
/// module built, live for the duration of this call.
pub unsafe fn read_pbo_vtable_parts(vtable_ptr: *const c_void) -> Result<PboOpsVtableParts> {
    let Some(vtable) = NonNull::new(vtable_ptr as *mut PboOpsVtable) else {
        return Err(Error::InvalidArgument(
            "PBO descriptor carries no ops vtable".into(),
        ));
    };
    // SAFETY: caller's contract -- the table is live for this call.
    let v = unsafe { vtable.as_ref() };
    Ok(PboOpsVtableParts {
        state: v.state,
        map_fn: v.map_buffer_fn,
        unmap_fn: v.unmap_buffer_fn,
    })
}

/// Build a live [`PboOps`] from a client's [`EfClientState`] and its two op
/// functions — the one place the three parts are assembled, whether they
/// arrived through a [`crate::TensorDesc`] under
/// [`crate::protocol::kind::PBO`] or through `ef_tensor_wrap_pbo`'s
/// arguments.
///
/// Takes its own reference on the channel (`state.retain`) and releases it
/// when the returned `Arc`'s last clone drops.
///
/// # Errors
/// [`Error::InvalidArgument`] for a NULL `ctx`, `retain` or `release` — a
/// silently non-owning attach is exactly the class of quiet loss this
/// design exists to remove.
///
/// # Safety
/// `state.ctx` must be a genuine context the supplied `retain`/`release`
/// understand, and `map_fn`/`unmap_fn` must uphold [`PboOps`]'s own
/// contract for it.
pub unsafe fn client_state_pbo_ops(
    state: EfClientState,
    map_fn: EfPboMapFn,
    unmap_fn: EfPboUnmapFn,
) -> Result<Arc<dyn PboOps>> {
    if state.ctx.is_null() {
        return Err(Error::InvalidArgument(
            "PBO client state carries a NULL ctx".into(),
        ));
    }
    let Some(retain) = state.retain else {
        return Err(Error::InvalidArgument(
            "PBO client state carries a NULL retain".into(),
        ));
    };
    if state.release.is_none() {
        return Err(Error::InvalidArgument(
            "PBO client state carries a NULL release".into(),
        ));
    }
    // SAFETY: the caller's contract above. Taken before the value is stored
    // so `Drop`'s release always balances exactly one retain.
    unsafe { retain(state.ctx) };
    Ok(Arc::new(ImportedPboOps {
        state,
        map_buffer_fn: map_fn,
        unmap_buffer_fn: unmap_fn,
    }))
}

/// Opaque handle to a PBO's GL resources.
struct PboHandle {
    ops: Arc<dyn PboOps>,
    buffer_id: u32,
    size: usize,
    map_state: Mutex<MapState>,
    /// Wakes threads waiting out an in-flight GL map/unmap. Needed because
    /// `map_state` is deliberately NOT held across `PboOps` calls — see
    /// [`PboHandle::acquire_map`].
    map_cv: Condvar,
    /// C-ABI vtable dispatching into `ops`, for cross-cdylib export via
    /// [`crate::TensorDesc`]. Built once, lazily, on first
    /// [`PboTensor::pbo_vtable`] call — see [`PboOpsVtable`]'s own doc
    /// comment for why this exists and what it replaces (a same-process-only
    /// registry, which cannot cross a `cdylib` boundary at all).
    ///
    /// `static`-only since Stage B: under `dynamic` a PBO's storage lives
    /// inside `libedgefirst_tensor.so`, which builds and owns the table
    /// itself, and `TensorDyn::pbo_vtable_ptr` asks it through
    /// `ef_tensor_pbo_vtable` rather than building one here.
    #[cfg(feature = "static")]
    abi_vtable: OnceLock<PboOpsVtable>,
}

/// CPU-map state of a PBO's single GL buffer.
///
/// A GL buffer has exactly one mapping at a time -- but **this state
/// machine enforces that only within one `PboHandle`**, including its own
/// read-sharing (below); it has no visibility into a *different*
/// `PboHandle` for the same real `buffer_id`. Two `PboHandle`s legitimately
/// exist for one buffer today: a producer's own tensor and a tensor
/// [`crate::TensorDyn::import_descriptor`]'s `kind::PBO` arm reconstructs
/// from it cross-cdylib. What actually prevents *those* two from both
/// reaching `glMapBufferRange` on the same buffer is a second, separate
/// enforcement point downstream of this one: the real `PboOps`
/// implementor's own serialization of every call against one real buffer
/// (in `edgefirst-image`, `GLProcessorThreaded`'s message loop tracks
/// outstanding `buffer_id`s and refuses a second concurrent `PboMap`,
/// because its one GL thread is the only place every call -- from any
/// `PboHandle`, in any linked copy of this crate -- actually converges).
/// This state machine's own job stays exactly what it says below: same-
/// handle mapping/read-sharing bookkeeping.
///
/// So these describe that one mapping's ownership rather than per-[`PboMap`] state:
///
/// * `Unmapped` — no CPU mapping exists.
/// * `Exclusive` — one writable map holds it. Nothing else may map until
///   that map drops.
/// * `Shared` — `readers` read-only maps over ONE mapping, unmapped when
///   the last of them drops.
///
/// Read sharing exists because tiled (SAHI) pre-processing has several
/// worker threads convert different crops of the SAME source tensor
/// concurrently, each taking `map_read()`. Read-only holders cannot observe
/// one another's writes because there are none, so one mapping serves them
/// all. Under a single-map rule every reader but one failed with
/// [`Error::PboMapped`], which left callers no option but to serialize the
/// workers that share a source.
enum MapState {
    Unmapped,
    /// A thread is inside `PboOps::map_buffer` for this buffer right now.
    /// Others must wait it out rather than start a second GL map.
    Mapping,
    Exclusive,
    Shared {
        readers: usize,
        ptr: PboPtr,
    },
    /// A thread is inside `PboOps::unmap_buffer` for this buffer right now.
    Unmapping,
}

impl PboHandle {
    /// Acquire a CPU mapping of this buffer, returning the base pointer.
    ///
    /// Read-only acquisitions join an existing read-only mapping (bumping
    /// its refcount) or create one. Writable acquisitions are exclusive.
    /// Returns [`Error::PboMapped`] whenever the request cannot share what
    /// is already held: anything against an exclusive map, or a writable
    /// request against readers.
    ///
    /// # Locking
    ///
    /// `map_state` is NEVER held across a [`PboOps`] call. Those calls are
    /// blocking round-trips to the GL thread, and the work the GL thread
    /// runs (a convert, say) maps tensors itself — so a thread that waited
    /// on GL while holding this mutex would deadlock against the very GL
    /// thread it is waiting for. The in-flight GL call is published as
    /// [`MapState::Mapping`] / [`MapState::Unmapping`] instead, and other
    /// threads wait on `map_cv` until it resolves.
    fn acquire_map(&self, writable: bool) -> Result<PboPtr> {
        loop {
            let mut guard = self.map_state.lock().expect("PBO map state mutex poisoned");
            // Wait out any GL map/unmap another thread is mid-way through.
            while matches!(*guard, MapState::Mapping | MapState::Unmapping) {
                guard = self
                    .map_cv
                    .wait(guard)
                    .expect("PBO map state mutex poisoned");
            }
            match &mut *guard {
                MapState::Shared { readers, ptr } if !writable => {
                    *readers += 1;
                    return Ok(PboPtr(ptr.0));
                }
                MapState::Shared { .. } | MapState::Exclusive => return Err(Error::PboMapped),
                // Ruled out by the wait above; re-check rather than assume.
                MapState::Mapping | MapState::Unmapping => continue,
                MapState::Unmapped => *guard = MapState::Mapping,
            }
            drop(guard);

            // No lock held here — see this function's Locking note.
            let mapped = self.ops.map_buffer(self.buffer_id, self.size);
            let base = match mapped {
                // A SHORT mapping is a failed mapping. `scoped_pin` and
                // `map_internal` both hand out `self.size` bytes from this
                // base -- they have nothing else to go on -- so an
                // implementation that reports success while mapping fewer
                // bytes (a client's `map_fn` across the `ef_client_state`
                // channel is untrusted here, and its `out_len` is its own
                // claim) would turn into an out-of-bounds host slice with
                // no error anywhere. Refuse it the same way a NULL pointer
                // is refused below: GL still considers the buffer mapped,
                // so release it rather than strand it.
                Ok(mapping) if mapping.size < self.size => {
                    let _ = self.ops.unmap_buffer(self.buffer_id);
                    self.finish_map(None, writable);
                    return Err(Error::InsufficientCapacity {
                        needed: self.size,
                        capacity: mapping.size,
                    });
                }
                Ok(mapping) => NonNull::new(mapping.ptr as *mut c_void),
                Err(e) => {
                    self.finish_map(None, writable);
                    return Err(e);
                }
            };
            let Some(base) = base else {
                // GL reported success but handed back a null pointer. It
                // considers the buffer mapped, so release it rather than
                // strand it mapped forever.
                //
                // Order matters: the cleanup unmap runs while this thread
                // still holds the `Mapping` claim, so waiters stay parked
                // and cannot start a fresh `map_buffer` that would race
                // this unmap on the GL queue. Only once it has completed is
                // `Unmapped` published.
                let _ = self.ops.unmap_buffer(self.buffer_id);
                self.finish_map(None, writable);
                return Err(Error::InvalidSize(self.size));
            };
            self.finish_map(Some(base), writable);
            return Ok(PboPtr(base));
        }
    }

    /// Publish the outcome of a GL map this thread had claimed via
    /// [`MapState::Mapping`], waking anyone who waited on it. `None` means
    /// the map failed and the buffer is unmapped again.
    fn finish_map(&self, base: Option<NonNull<c_void>>, writable: bool) {
        let mut guard = self.map_state.lock().expect("PBO map state mutex poisoned");
        *guard = match base {
            None => MapState::Unmapped,
            Some(_) if writable => MapState::Exclusive,
            Some(base) => MapState::Shared {
                readers: 1,
                ptr: PboPtr(base),
            },
        };
        self.map_cv.notify_all();
    }

    /// Release one acquisition, unmapping the GL buffer once the last
    /// holder is gone. Same locking rule as [`Self::acquire_map`]: the GL
    /// unmap runs with no lock held.
    fn release_map(&self) {
        let mut guard = self.map_state.lock().expect("PBO map state mutex poisoned");
        let last_holder = match &mut *guard {
            MapState::Shared { readers, .. } => {
                *readers = readers.saturating_sub(1);
                *readers == 0
            }
            MapState::Exclusive => true,
            // Nothing held — a release with no matching acquire. (A caller
            // holding an acquisition cannot observe Mapping/Unmapping:
            // only the last holder enters Unmapping, and it is this one.)
            MapState::Unmapped | MapState::Mapping | MapState::Unmapping => false,
        };
        if !last_holder {
            return;
        }
        *guard = MapState::Unmapping;
        drop(guard);

        trace!("Unmapping PBO buffer_id={}", self.buffer_id);
        if let Err(e) = self.ops.unmap_buffer(self.buffer_id) {
            log::warn!("Failed to unmap PBO buffer {}: {e}", self.buffer_id);
        }

        let mut guard = self.map_state.lock().expect("PBO map state mutex poisoned");
        *guard = MapState::Unmapped;
        self.map_cv.notify_all();
    }

    /// Whether any CPU mapping is currently held or being established.
    fn is_mapped(&self) -> bool {
        !matches!(
            *self.map_state.lock().expect("PBO map state mutex poisoned"),
            MapState::Unmapped
        )
    }
}

impl Drop for PboHandle {
    fn drop(&mut self) {
        self.ops.delete_buffer(self.buffer_id);
    }
}

/// A tensor backed by an OpenGL Pixel Buffer Object.
pub struct PboTensor<T>
where
    T: Num + Clone + fmt::Debug + Send + Sync,
{
    pub name: String,
    pub shape: Vec<usize>,
    handle: Arc<PboHandle>,
    identity: BufferIdentity,
    /// Byte offset of this tensor's window into the shared GL buffer. Non-zero
    /// only for sub-views (`view`/`batch`), which share the `Arc<PboHandle>` and
    /// `BufferIdentity` and address a sub-region by this offset — mirrors
    /// `DmaTensor::mmap_offset` / `IoSurfaceTensor::view_offset`.
    pub(crate) view_offset: usize,
    _marker: PhantomData<T>,
}

impl<T> fmt::Debug for PboTensor<T>
where
    T: Num + Clone + fmt::Debug + Send + Sync,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("PboTensor")
            .field("name", &self.name)
            .field("shape", &self.shape)
            .field("buffer_id", &self.handle.buffer_id)
            .field("size", &self.handle.size)
            .finish()
    }
}

unsafe impl<T> Send for PboTensor<T> where T: Num + Clone + fmt::Debug + Send + Sync {}
unsafe impl<T> Sync for PboTensor<T> where T: Num + Clone + fmt::Debug + Send + Sync {}

impl<T> PboTensor<T>
where
    T: Num + Clone + fmt::Debug + Send + Sync,
{
    /// Create a new PBO tensor from an already-allocated GL buffer.
    ///
    /// Called by the image crate after creating the PBO on the GL thread.
    /// Users should not call this directly — use `ImageProcessor::create_image()`.
    ///
    /// # Errors
    ///
    /// Returns `Error::ShapeMismatch` if `size` does not equal
    /// `shape.iter().product::<usize>() * std::mem::size_of::<T>()`.
    /// Returns `Error::InvalidSize` if `size` is zero.
    /// Returns `Error::InvalidShape` if that shape footprint overflows
    /// `usize` — refused rather than wrapped to a small byte count an
    /// undersized `size` would then satisfy.
    pub fn from_pbo(
        buffer_id: u32,
        size: usize,
        shape: &[usize],
        name: Option<&str>,
        ops: Arc<dyn PboOps>,
    ) -> Result<Self> {
        if size == 0 {
            return Err(Error::InvalidSize(0));
        }
        // Checked, not `product() * size_of::<T>()`: this shape arrives
        // straight from a C caller through `ef_tensor_wrap_pbo`, and a
        // product that wraps `usize` in a release build computes a *small*
        // footprint, slips past the `size <` test below, and mints a tensor
        // whose logical extent runs far past the GL allocation. One shared
        // helper with `reshape`/`set_logical_shape`/`view` so all four
        // footprints in this file obey the same rule.
        let expected = crate::ahardwarebuffer_layout::checked_shape_bytes::<T>(shape)?;
        // Allow `size >= expected`: PBOs allocated with a 64-byte-aligned row
        // stride may be larger than the shape product.  Reject only if the
        // allocation is strictly smaller than the logical content.
        if size < expected {
            return Err(Error::ShapeMismatch(format!(
                "PBO size {size} is smaller than shape {shape:?} * sizeof({}) = {expected}",
                std::any::type_name::<T>(),
            )));
        }
        let name = name.unwrap_or("pbo_tensor").to_owned();
        Ok(Self {
            name,
            shape: shape.to_vec(),
            handle: Arc::new(PboHandle {
                ops,
                buffer_id,
                size,
                map_state: Mutex::new(MapState::Unmapped),
                map_cv: Condvar::new(),
                #[cfg(feature = "static")]
                abi_vtable: OnceLock::new(),
            }),
            // A GL buffer name is meaningful only inside its creating
            // context (no system-wide key exists), but it is unique among
            // this context's live PBOs, which is exactly what the
            // per-context import cache needs to key on.
            identity: BufferIdentity::derived(IdentityKind::Pbo, buffer_id as u64),
            view_offset: 0,
            _marker: PhantomData,
        })
    }

    /// Returns the GL buffer ID for this PBO.
    pub fn buffer_id(&self) -> u32 {
        self.handle.buffer_id
    }

    /// The C-ABI vtable dispatching into this PBO's `ops`, for cross-cdylib
    /// export via [`crate::TensorDesc`]. See [`PboOpsVtable`]'s own doc
    /// comment. Called from `static`'s `pbo_vtable_ptr`
    /// (`lib.rs`/`static_backend.rs`); `dynamic` reads the library's own
    /// table through `ef_tensor_pbo_vtable` instead, so this is
    /// `static`-only.
    #[cfg(feature = "static")]
    pub(crate) fn pbo_vtable(&self) -> &PboOpsVtable {
        self.handle
            .abi_vtable
            .get_or_init(|| PboOpsVtable::new(&self.handle))
    }

    /// A type-erased keepalive holding this PBO's `Arc<PboHandle>` alive --
    /// the same shape as [`crate::pin::HostPin`]'s own `Keepalive` (an
    /// `Arc<dyn Send + Sync>`), for a now-narrower reason: [`PboOpsVtable::
    /// new`] hands out `state.ctx`, this `PboHandle`'s own address, and that
    /// pointer must still be live at the moment an importer reads the table
    /// and calls `state.retain`. Past that call the importer holds its own
    /// reference (see [`client_state_pbo_ops`]) and needs nothing external,
    /// so this is belt-and-braces over the import window rather than the
    /// only thing standing between a consumer and freed memory. Cloning this
    /// `Arc` (not the `PboHandle`'s *contents*) is what a cross-package
    /// capsule holds alongside the descriptor, exactly the way `pin` already
    /// does for the `HOST` kind -- see `TensorCapsulePayload`
    /// (`edgefirst-python-common`). Type-erased so this crate's public
    /// surface never has to name `PboHandle`, which stays private.
    ///
    /// Called from `static`'s `pbo_keepalive`
    /// (`lib.rs`/`static_backend.rs`); `dynamic` holds a retained reference
    /// on the handle that owns the library-side `PboTensor` instead, so this
    /// is `static`-only.
    #[cfg(feature = "static")]
    pub(crate) fn pbo_keepalive(&self) -> Arc<dyn Send + Sync> {
        self.handle.clone()
    }

    /// Returns true unless the PBO is fully unmapped.
    ///
    /// That covers an exclusive map, one or more read-only holders, AND a
    /// map or unmap another thread currently has in flight — during those
    /// windows the GL buffer is (or is about to be) mapped, and no other
    /// mapping may begin, so reporting `false` would be misleading. Use it
    /// as "is this buffer free for GL operations?", not as "does a CPU
    /// pointer exist right now".
    pub fn is_mapped(&self) -> bool {
        self.handle.is_mapped()
    }
}

/// The C-ABI parts `ef_tensor_wrap_pbo` takes, from a [`PboTensor`] **at
/// the origin** of its buffer.
///
/// "At the origin", not "whole-buffer": what
/// [`PboTensor::into_client_parts`] guards is the byte offset, not the
/// extent. `view(0, &[16])` on a 32-byte buffer passes and yields
/// `shape = [16]`, `size = 32` — a correct, representable tensor, because
/// `size` is the GL allocation's full byte count and is *documented* to
/// exceed the shape's product (that is exactly how a stride-padded PBO is
/// carried). A non-zero `view_offset` is the only thing this struct cannot
/// express, and that is what is refused.
///
/// `state.ctx` **borrows** the channel this value still owns; the callee is
/// expected to `retain` it (which [`client_state_pbo_ops`] does) before this
/// struct drops. Dropping it without retaining releases the client's own
/// last reference and deletes the GL buffer — which is why the `Arc` is a
/// field here rather than something the caller has to remember to hold.
///
/// Deliberately neither `Send` nor `Sync`: `EfClientState` holds a raw
/// pointer and `edgefirst-tensor-abi` adds no `unsafe impl`. That is right
/// for a value whose only purpose is to be handed straight to a C entry
/// point on the calling thread. If this ever needs to cross a thread, that
/// is a design question to raise, not something to unblock with an
/// `unsafe impl`.
pub struct PboClientParts {
    /// The callback channel, borrowing this value's own reference.
    pub state: EfClientState,
    pub map_fn: EfPboMapFn,
    pub unmap_fn: EfPboUnmapFn,
    pub buffer_id: u32,
    /// The GL allocation's full byte count, which may exceed the shape's
    /// product (a 64-byte-aligned row stride).
    pub size: usize,
    pub shape: Vec<usize>,
    /// The client's own strong count on the channel, released when this
    /// value drops. Private: `PboHandle` is not part of this crate's
    /// public surface.
    _handle: Arc<PboHandle>,
}

impl<T> PboTensor<T>
where
    T: Num + Clone + fmt::Debug + Send + Sync,
{
    /// Decompose this tensor into the parts a C entry point takes.
    ///
    /// The shape is carried through as-is and is **not** required to fill
    /// the allocation: `size` is the GL buffer's full byte count and may
    /// exceed the shape's product by design (a 64-byte-aligned row stride
    /// is the usual reason). Only the *origin* is constrained.
    ///
    /// # Errors
    /// [`Error::InvalidOperation`] for a sub-view (`view_offset != 0`).
    /// A sub-view's window is not expressible in `ef_tensor_wrap_pbo`'s
    /// argument list, and a wrap that silently dropped it would land the
    /// far side at the parent's origin — issue #162 in reverse. Views are
    /// taken on the far side (`Tensor::view`) once the buffer is wrapped,
    /// so nothing needs this.
    pub fn into_client_parts(self) -> Result<PboClientParts> {
        if self.view_offset != 0 {
            return Err(Error::InvalidOperation(format!(
                "PboTensor::into_client_parts: cannot wrap a sub-view \
                 (view_offset = {}); wrap the whole buffer and take the view \
                 on the other side",
                self.view_offset
            )));
        }
        let state = EfClientState {
            ctx: Arc::as_ptr(&self.handle) as *const c_void,
            retain: Some(vt_retain),
            release: Some(vt_release),
        };
        Ok(PboClientParts {
            state,
            map_fn: vt_map_buffer,
            unmap_fn: vt_unmap_buffer,
            buffer_id: self.handle.buffer_id,
            size: self.handle.size,
            shape: self.shape.clone(),
            _handle: self.handle,
        })
    }
}

impl<T> TensorTrait<T> for PboTensor<T>
where
    T: Num + Clone + fmt::Debug + Send + Sync,
{
    fn new(_shape: &[usize], _name: Option<&str>) -> Result<Self> {
        Err(Error::NotImplemented(
            "PboTensor cannot be created directly — use ImageProcessor::create_image()".to_owned(),
        ))
    }

    #[cfg(unix)]
    fn from_fd(_fd: std::os::fd::OwnedFd, _shape: &[usize], _name: Option<&str>) -> Result<Self> {
        Err(Error::NotImplemented(
            "PboTensor does not support from_fd".to_owned(),
        ))
    }

    #[cfg(unix)]
    fn clone_fd(&self) -> Result<std::os::fd::OwnedFd> {
        Err(Error::NotImplemented(
            "PboTensor does not support clone_fd".to_owned(),
        ))
    }

    fn memory(&self) -> TensorMemory {
        TensorMemory::Pbo
    }

    fn name(&self) -> String {
        self.name.clone()
    }

    fn shape(&self) -> &[usize] {
        &self.shape
    }

    fn reshape(&mut self, shape: &[usize]) -> Result<()> {
        if shape.is_empty() {
            return Err(Error::InvalidSize(0));
        }
        let new_size = crate::ahardwarebuffer_layout::checked_shape_bytes::<T>(shape)?;
        if new_size != self.handle.size {
            return Err(Error::ShapeMismatch(format!(
                "Cannot reshape incompatible shape: {:?} to {:?}",
                self.shape, shape
            )));
        }
        self.shape = shape.to_vec();
        self.view_offset = 0;
        Ok(())
    }

    fn capacity_bytes(&self) -> usize {
        self.handle.size
    }

    /// Capacity-based reconfigure (mirrors Mem/Shm/DMA/IOSurface): allow any
    /// shape whose byte size fits the PBO allocation, so an oversized reusable
    /// pool can be `configure_image`d to a smaller image. Without this PBO fell
    /// back to the strict-`reshape` default and rejected pool reuse.
    fn set_logical_shape(&mut self, shape: &[usize]) -> Result<()> {
        if shape.is_empty() {
            return Err(Error::InvalidSize(0));
        }
        let needed = crate::ahardwarebuffer_layout::checked_shape_bytes::<T>(shape)?;
        if needed > self.handle.size {
            return Err(Error::InsufficientCapacity {
                needed,
                capacity: self.handle.size,
            });
        }
        self.shape = shape.to_vec();
        Ok(())
    }

    fn map_with<'a>(&self, access: crate::CpuAccess) -> Result<crate::view::HostView<'a, T>>
    where
        T: 'a,
    {
        self.map_internal(None, access)
    }

    fn buffer_identity(&self) -> &BufferIdentity {
        &self.identity
    }

    /// Zero-copy sub-region view sharing this PBO's GL buffer (via the
    /// `Arc<PboHandle>`) and [`BufferIdentity`], positioned at `offset_bytes`
    /// from this tensor's own window with logical `shape`. The GL backend keys
    /// the import on the shared identity and addresses the window via
    /// `glViewport` / the staged copy; a CPU map adds `view_offset` to the
    /// mapped base. Mirrors [`DmaTensor::view`](crate::TensorTrait::view).
    ///
    /// # Errors
    ///
    /// - [`Error::InvalidOperation`] if `offset_bytes` is mis-aligned for `T`.
    /// - [`Error::InsufficientCapacity`] if the window exceeds the allocation.
    fn view(&self, offset_bytes: usize, shape: &[usize]) -> Result<Self> {
        if !offset_bytes.is_multiple_of(std::mem::align_of::<T>()) {
            return Err(Error::InvalidOperation(format!(
                "PboTensor::view: offset {offset_bytes} not aligned to align_of::<T>()={}",
                std::mem::align_of::<T>()
            )));
        }
        let abs_offset = self
            .view_offset
            .checked_add(offset_bytes)
            .ok_or(Error::InvalidSize(offset_bytes))?;
        let logical = crate::ahardwarebuffer_layout::checked_shape_bytes::<T>(shape)?;
        let needed = abs_offset
            .checked_add(logical)
            .ok_or(Error::InvalidSize(logical))?;
        if needed > self.handle.size {
            return Err(Error::InsufficientCapacity {
                needed,
                capacity: self.handle.size,
            });
        }
        Ok(Self {
            name: self.name.clone(),
            shape: shape.to_vec(),
            handle: Arc::clone(&self.handle),
            identity: self.identity.clone(),
            view_offset: abs_offset,
            _marker: PhantomData,
        })
    }
}

impl<T> PboTensor<T>
where
    T: Num + Clone + fmt::Debug + Send + Sync,
{
    /// Map the PBO so `as_slice()` exposes the full padded buffer (`byte_size`
    /// bytes) rather than the shape-derived logical count. Mirrors
    /// [`DmaTensor::map_with_byte_size`]: a CPU producer (e.g. the JPEG decoder)
    /// or a strided convert source iterates rows via `effective_row_stride()`
    /// without running past the slice. Crate-private; the only caller is
    /// `Tensor::map()`, which already checks `byte_size <= capacity_bytes()`.
    ///
    /// `static`-only: `Tensor::map()` (`impl<T> TensorMapTrait<T> for
    /// Tensor<T>` in `lib.rs`) is itself `#[cfg(feature = "static")]`. Only
    /// this one accessor gates -- `PboTensor` and the rest of its API stay
    /// available under `dynamic` too, since `PboOps`/`PboMapping` are a
    /// backend-agnostic GL extension point `edgefirst-image` implements
    /// regardless of which tensor backend it links.
    #[cfg(feature = "static")]
    pub(crate) fn map_with_byte_size<'a>(
        &self,
        byte_size: usize,
        access: crate::CpuAccess,
    ) -> Result<crate::view::HostView<'a, T>>
    where
        T: 'a,
    {
        self.map_internal(Some(byte_size), access)
    }

    fn map_internal<'a>(
        &self,
        byte_size_override: Option<usize>,
        access: crate::CpuAccess,
    ) -> Result<crate::view::HostView<'a, T>>
    where
        T: 'a,
    {
        // Always map the full GL allocation (`handle.size`); the slice length is
        // narrowed by `byte_size_override` (or the logical shape) at access time.
        // scoped_pin acquires the GL map; its keepalive releases it exactly
        // once when the last clone drops, which is what the `released` flag
        // used to do by hand.
        Ok(crate::view::HostView::new(
            self.scoped_pin(access)?,
            self.shape.clone(),
            byte_size_override,
            access,
        ))
    }
}

/// Non-null base address of a live GL buffer mapping.
///
/// Still used by `PboHandle`'s refcounted acquire/release even though the
/// per-backend `PboMap` it once fed is gone: the handle serialises GL maps
/// itself, and this is what it hands back.
#[derive(Debug)]
struct PboPtr(NonNull<c_void>);

impl Deref for PboPtr {
    type Target = NonNull<c_void>;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

unsafe impl Send for PboPtr {}

/// Keepalive that holds a PBO's CPU map open for a [`HostPin`]'s lifetime.
///
/// A PBO's address is only valid between `glMapBufferRange` and
/// `glUnmapBuffer`, so — unlike DMA-BUF or IOSurface — the map IS the lifetime.
/// `PboTensor::scoped_pin` acquires it and this releases it when the last clone
/// of the pin drops, which reproduces `PboMap::unmap`'s once-only release
/// through `Arc` rather than a `released` flag.
// Staged for the HostView collapse (Plan 2b Task 4): constructed once
// map_with migrates to HostView. Kept compiling and reviewable rather than
// landing the whole collapse in one unverifiable change.
#[allow(dead_code)]
pub(crate) struct PboMapLock {
    handle: Arc<PboHandle>,
}

impl Drop for PboMapLock {
    fn drop(&mut self) {
        self.handle.release_map();
    }
}

impl<T> PboTensor<T>
where
    T: Num + Clone + fmt::Debug + Send + Sync,
{
    /// Acquire a host address valid for as long as the returned pin lives.
    ///
    /// Private on purpose: this is NOT `pin_host`, which refuses PBO precisely
    /// because the address cannot outlive the map. A guard built on this pin
    /// releases the map when it drops, which is the only correct lifetime.
    #[allow(dead_code)]
    pub(crate) fn scoped_pin<'a>(&self, access: crate::CpuAccess) -> Result<crate::pin::HostPin<'a>>
    where
        T: 'a,
    {
        let ptr = self.handle.acquire_map(access.writes())?;
        let base = unsafe { (ptr.as_ptr() as *mut u8).add(self.view_offset) };
        let len = self.handle.size.saturating_sub(self.view_offset);
        Ok(crate::pin::HostPin::new(
            Arc::new(PboMapLock {
                handle: Arc::clone(&self.handle),
            }),
            base,
            len,
        ))
    }
}

// -- PboMap --

impl<T> Clone for PboTensor<T>
where
    T: Num + Clone + fmt::Debug + Send + Sync,
{
    fn clone(&self) -> Self {
        Self {
            name: self.name.clone(),
            shape: self.shape.clone(),
            handle: Arc::clone(&self.handle),
            identity: self.identity.clone(),
            view_offset: self.view_offset,
            _marker: PhantomData,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::TensorMapTrait;

    /// Mock PboOps that uses a Vec<u8> as backing storage instead of GL.
    ///
    /// Counts map/unmap calls so tests can assert that N read-only holders
    /// share ONE GL mapping rather than each taking their own.
    struct MockPboOps {
        storage: Mutex<Vec<u8>>,
        maps: std::sync::atomic::AtomicUsize,
        unmaps: std::sync::atomic::AtomicUsize,
        deletes: std::sync::atomic::AtomicUsize,
    }

    impl MockPboOps {
        fn new(size: usize) -> Arc<Self> {
            Arc::new(Self {
                storage: Mutex::new(vec![0u8; size]),
                maps: std::sync::atomic::AtomicUsize::new(0),
                unmaps: std::sync::atomic::AtomicUsize::new(0),
                deletes: std::sync::atomic::AtomicUsize::new(0),
            })
        }

        fn map_count(&self) -> usize {
            self.maps.load(std::sync::atomic::Ordering::Acquire)
        }

        fn delete_count(&self) -> usize {
            self.deletes.load(std::sync::atomic::Ordering::Acquire)
        }

        fn unmap_count(&self) -> usize {
            self.unmaps.load(std::sync::atomic::Ordering::Acquire)
        }
    }

    // SAFETY: the returned pointer addresses a `Vec<u8>` that is allocated
    // once in `new` and never resized, so it stays valid for the whole life
    // of the `MockPboOps` — which outlives every map handed out, since the
    // tensor holds an `Arc` to it. The storage mutex is NOT what keeps the
    // pointer alive (`map_buffer` releases it before returning); it only
    // guards the length assertion.
    unsafe impl PboOps for MockPboOps {
        fn map_buffer(&self, _buffer_id: u32, size: usize) -> Result<PboMapping> {
            self.maps.fetch_add(1, std::sync::atomic::Ordering::AcqRel);
            let storage = self.storage.lock().expect("lock");
            assert_eq!(storage.len(), size);
            Ok(PboMapping {
                ptr: storage.as_ptr() as *mut u8,
                size,
            })
        }

        fn unmap_buffer(&self, _buffer_id: u32) -> Result<()> {
            self.unmaps
                .fetch_add(1, std::sync::atomic::Ordering::AcqRel);
            Ok(())
        }

        fn delete_buffer(&self, _buffer_id: u32) {
            self.deletes
                .fetch_add(1, std::sync::atomic::Ordering::AcqRel);
        }
    }

    #[test]
    fn test_pbo_tensor_create_and_metadata() {
        let ops = MockPboOps::new(24);
        let tensor = PboTensor::<u8>::from_pbo(42, 24, &[2, 3, 4], Some("test_pbo"), ops).unwrap();
        assert_eq!(tensor.memory(), TensorMemory::Pbo);
        assert_eq!(tensor.name(), "test_pbo");
        assert_eq!(tensor.shape(), &[2, 3, 4]);
        assert_eq!(tensor.buffer_id(), 42);
        assert!(!tensor.is_mapped());
    }

    #[test]
    fn test_pbo_tensor_map_write_read() {
        let ops = MockPboOps::new(12);
        let tensor = PboTensor::<u8>::from_pbo(1, 12, &[3, 4], Some("rw_test"), ops).unwrap();
        {
            let mut map = tensor.map().expect("map should succeed");
            assert_eq!(map.shape(), &[3, 4]);
            assert!(tensor.is_mapped());
            map.as_mut_slice().fill(0xAB);
            assert!(map.as_slice().iter().all(|&b| b == 0xAB));
        }
        assert!(!tensor.is_mapped());
    }

    #[test]
    fn test_pbo_tensor_double_map_fails() {
        let ops = MockPboOps::new(8);
        let tensor = PboTensor::<u8>::from_pbo(2, 8, &[8], None, ops).unwrap();
        let _map1 = tensor.map().expect("first map should succeed");
        assert!(tensor.is_mapped());
        let result = tensor.map();
        assert!(result.is_err(), "second map while mapped should fail");
    }

    /// Mock whose `map_buffer` reports success but hands back a null
    /// pointer — the one path where a claimed map has to be released again.
    #[derive(Default)]
    struct NullMappingOps {
        unmaps: std::sync::atomic::AtomicUsize,
    }

    impl NullMappingOps {
        fn unmap_count(&self) -> usize {
            self.unmaps.load(std::sync::atomic::Ordering::Acquire)
        }
    }

    // SAFETY: never produces a usable pointer, so no caller can dereference
    // one; the mapping it returns is rejected before reaching a slice.
    unsafe impl PboOps for NullMappingOps {
        fn map_buffer(&self, _buffer_id: u32, size: usize) -> Result<PboMapping> {
            Ok(PboMapping {
                ptr: std::ptr::null_mut(),
                size,
            })
        }

        fn unmap_buffer(&self, _buffer_id: u32) -> Result<()> {
            self.unmaps
                .fetch_add(1, std::sync::atomic::Ordering::AcqRel);
            Ok(())
        }

        fn delete_buffer(&self, _buffer_id: u32) {}
    }

    /// An implementation that reports success while mapping FEWER bytes than
    /// were asked for. The bytes it does map are real, so a caller that
    /// trusted the success would read `size` bytes from a `size / 2`-byte
    /// allocation — the shape of the bug, not a null-pointer stand-in.
    #[derive(Default)]
    struct ShortMappingOps {
        storage: Mutex<Vec<u8>>,
        unmaps: std::sync::atomic::AtomicUsize,
    }

    impl ShortMappingOps {
        fn unmap_count(&self) -> usize {
            self.unmaps.load(std::sync::atomic::Ordering::Acquire)
        }
    }

    // SAFETY: the pointer addresses a `Vec<u8>` this value owns for its
    // whole life, and the mapping is rejected before reaching a slice.
    unsafe impl PboOps for ShortMappingOps {
        fn map_buffer(&self, _buffer_id: u32, size: usize) -> Result<PboMapping> {
            let short = size / 2;
            let mut storage = self.storage.lock().expect("lock");
            storage.resize(short, 0);
            Ok(PboMapping {
                ptr: storage.as_mut_ptr(),
                size: short,
            })
        }

        fn unmap_buffer(&self, _buffer_id: u32) -> Result<()> {
            self.unmaps
                .fetch_add(1, std::sync::atomic::Ordering::AcqRel);
            Ok(())
        }

        fn delete_buffer(&self, _buffer_id: u32) {}
    }

    /// A short mapping is a failed mapping: every consumer of the returned
    /// base (`scoped_pin`, `map_internal`) sizes its host slice from
    /// `handle.size`, so accepting `out_len < size` would hand out an
    /// out-of-bounds slice with no error anywhere. The buffer GL does
    /// consider mapped is released, exactly as for a null mapping.
    #[test]
    fn short_mapping_is_refused_and_the_buffer_released() {
        let ops = Arc::new(ShortMappingOps::default());
        let dyn_ops: Arc<dyn PboOps> = ops.clone();
        let tensor = PboTensor::<u8>::from_pbo(13, 32, &[32], None, dyn_ops).unwrap();

        let Err(err) = tensor.map_read() else {
            panic!("a mapping shorter than the allocation must not succeed");
        };
        assert!(
            matches!(
                err,
                Error::InsufficientCapacity {
                    needed: 32,
                    capacity: 16
                }
            ),
            "unexpected error: {err:?}"
        );
        assert_eq!(
            ops.unmap_count(),
            1,
            "GL still holds the buffer mapped — the refused map must release it"
        );
        assert!(
            !tensor.is_mapped(),
            "state must return to Unmapped so later maps can proceed"
        );
    }

    /// A shape whose byte footprint wraps `usize` must be refused, not
    /// multiplied down to something an eight-byte allocation satisfies.
    /// `ef_tensor_wrap_pbo` forwards a C caller's `dims` here unchecked, so
    /// this is the boundary that has to hold: `2^63 * 4` is `0` modulo
    /// `usize`, and the accepted tensor would then report `2^65` elements
    /// over eight real bytes.
    #[test]
    fn from_pbo_refuses_a_shape_whose_footprint_overflows() {
        let ops = MockPboOps::new(8);
        let dyn_ops: Arc<dyn PboOps> = ops.clone();
        let result = PboTensor::<u8>::from_pbo(14, 8, &[1usize << 63, 4], None, dyn_ops);
        let Err(err) = result else {
            panic!("an overflowing shape footprint must be refused");
        };
        assert!(
            matches!(err, Error::InvalidShape(_)),
            "unexpected error: {err:?}"
        );
    }

    /// The same rule on the three reshaping paths, which recompute the
    /// footprint against an allocation that is already fixed.
    #[test]
    fn reshape_paths_refuse_an_overflowing_footprint() {
        let ops = MockPboOps::new(8);
        let dyn_ops: Arc<dyn PboOps> = ops.clone();
        let mut tensor = PboTensor::<u8>::from_pbo(15, 8, &[8], None, dyn_ops).unwrap();
        let huge = [1usize << 63, 4];

        assert!(
            matches!(tensor.reshape(&huge), Err(Error::InvalidShape(_))),
            "reshape must refuse a wrapping footprint"
        );
        assert!(
            matches!(tensor.set_logical_shape(&huge), Err(Error::InvalidShape(_))),
            "set_logical_shape must refuse a wrapping footprint"
        );
        assert!(
            matches!(tensor.view(0, &huge), Err(Error::InvalidShape(_))),
            "view must refuse a wrapping footprint"
        );
    }

    /// A map that "succeeds" with a null pointer leaves GL considering the
    /// buffer mapped, so it must be released and the state returned to
    /// unmapped — otherwise the buffer is stranded and every later map
    /// fails. The release happens before `Unmapped` is published, so no
    /// waiter can start a map that races the cleanup unmap.
    #[test]
    fn null_mapping_releases_the_buffer_and_resets_state() {
        let ops = Arc::new(NullMappingOps::default());
        let dyn_ops: Arc<dyn PboOps> = ops.clone();
        let tensor = PboTensor::<u8>::from_pbo(12, 8, &[8], None, dyn_ops).unwrap();

        let Err(err) = tensor.map_read() else {
            panic!("a null mapping must not succeed");
        };
        assert!(
            matches!(err, Error::InvalidSize(_)),
            "unexpected error: {err:?}"
        );
        assert_eq!(
            ops.unmap_count(),
            1,
            "GL still holds the buffer mapped — the failed map must release it"
        );
        assert!(
            !tensor.is_mapped(),
            "state must return to Unmapped so later maps can proceed"
        );

        // Not stranded: the buffer can be claimed again.
        let Err(err) = tensor.map_read() else {
            panic!("still null, so it must still fail");
        };
        assert!(matches!(err, Error::InvalidSize(_)));
        assert_eq!(ops.unmap_count(), 2, "and released again");
    }

    /// Several read-only holders share ONE GL mapping. This is what lets
    /// tiled pre-processing run several workers over one source tensor;
    /// before read sharing, every reader after the first failed.
    #[test]
    fn read_maps_share_one_gl_mapping() {
        let ops = MockPboOps::new(8);
        let dyn_ops: Arc<dyn PboOps> = ops.clone();
        let tensor = PboTensor::<u8>::from_pbo(9, 8, &[8], None, dyn_ops).unwrap();

        let r1 = tensor.map_read().expect("first read map");
        let r2 = tensor
            .map_read()
            .expect("second read map must share, not fail");
        let r3 = tensor
            .map_read()
            .expect("third read map must share, not fail");

        assert_eq!(ops.map_count(), 1, "one GL mapping serves every reader");
        assert_eq!(ops.unmap_count(), 0, "nothing unmapped while readers live");
        assert!(tensor.is_mapped());
        // All readers address the same bytes.
        assert_eq!(r1.as_slice(), r2.as_slice());
        assert_eq!(r2.as_slice(), r3.as_slice());

        drop(r1);
        drop(r2);
        assert_eq!(
            ops.unmap_count(),
            0,
            "the mapping outlives every reader but the last"
        );
        assert!(tensor.is_mapped());

        drop(r3);
        assert_eq!(ops.unmap_count(), 1, "last reader out unmaps, exactly once");
        assert!(!tensor.is_mapped());
    }

    /// Read sharing must not weaken write exclusion in either direction.
    #[test]
    fn writers_and_readers_still_exclude_each_other() {
        let ops = MockPboOps::new(8);
        let dyn_ops: Arc<dyn PboOps> = ops.clone();
        let tensor = PboTensor::<u8>::from_pbo(10, 8, &[8], None, dyn_ops).unwrap();

        let reader = tensor.map_read().expect("read map");
        assert!(
            tensor.map().is_err(),
            "a writer must not join an existing reader set"
        );
        drop(reader);

        let writer = tensor.map().expect("write map after readers drained");
        assert!(
            tensor.map_read().is_err(),
            "a reader must not join an exclusive writer"
        );
        assert!(
            tensor.map().is_err(),
            "a second writer must not join an exclusive writer"
        );
        drop(writer);

        // Fully released — a fresh map of either kind succeeds again.
        assert!(tensor.map_read().is_ok());
    }

    /// `unmap()` is public on the trait and also runs from `Drop`. With
    /// refcounted readers a double release would unmap a SIBLING reader's
    /// mapping, so each map must release at most once.
    #[test]
    fn explicit_unmap_then_drop_releases_only_once() {
        let ops = MockPboOps::new(8);
        let dyn_ops: Arc<dyn PboOps> = ops.clone();
        let tensor = PboTensor::<u8>::from_pbo(11, 8, &[8], None, dyn_ops).unwrap();

        let keeper = tensor.map_read().expect("reader that must survive");
        let mut early = tensor.map_read().expect("reader unmapped by hand");
        early.unmap();
        drop(early); // Drop must NOT release a second time.

        assert_eq!(
            ops.unmap_count(),
            0,
            "one reader still holds the mapping — it must not be unmapped"
        );
        assert!(tensor.is_mapped(), "the surviving reader still holds it");
        // The survivor's pointer is still valid and readable.
        assert_eq!(keeper.as_slice().len(), 8);

        drop(keeper);
        assert_eq!(ops.unmap_count(), 1);
        assert!(!tensor.is_mapped());
    }

    #[test]
    fn test_pbo_tensor_reshape() {
        let ops = MockPboOps::new(24);
        let mut tensor = PboTensor::<u8>::from_pbo(3, 24, &[2, 3, 4], None, ops).unwrap();
        tensor
            .reshape(&[4, 6])
            .expect("compatible reshape should succeed");
        assert_eq!(tensor.shape(), &[4, 6]);
        let result = tensor.reshape(&[100]);
        assert!(result.is_err(), "incompatible reshape should fail");
    }

    #[test]
    fn test_pbo_tensor_set_logical_shape_capacity_based() {
        // A 24-byte PBO can be reconfigured to any shape that fits (unlike the
        // strict `reshape`), so an oversized reusable pool can be
        // `configure_image`d to a smaller image (the native-chroma decode pool).
        let ops = MockPboOps::new(24);
        let mut tensor = PboTensor::<u8>::from_pbo(7, 24, &[24], None, ops).unwrap();
        // Smaller-than-capacity logical shape is accepted (reshape would reject).
        tensor
            .set_logical_shape(&[4, 5])
            .expect("shape within capacity should succeed");
        assert_eq!(tensor.shape(), &[4, 5]);
        // Exactly-capacity is fine.
        tensor.set_logical_shape(&[24]).unwrap();
        // Over-capacity is rejected.
        assert!(
            tensor.set_logical_shape(&[5, 5]).is_err(),
            "shape exceeding PBO capacity must be rejected"
        );
    }

    #[test]
    fn test_pbo_tensor_buffer_identity() {
        let ops1 = MockPboOps::new(8);
        let ops2 = MockPboOps::new(8);
        let t1 = PboTensor::<u8>::from_pbo(1, 8, &[8], None, ops1).unwrap();
        let t2 = PboTensor::<u8>::from_pbo(2, 8, &[8], None, ops2).unwrap();
        assert_eq!(t1.buffer_identity().kind(), IdentityKind::Pbo);
        assert_ne!(t1.buffer_identity().id(), t2.buffer_identity().id());
    }

    #[test]
    fn two_pbos_with_the_same_gl_name_share_an_identity() {
        // The identity is derived from the GL buffer name, not a per-call
        // counter -- two independent `from_pbo` wraps of the same name (the
        // in-process analog of two independently-linked copies of this crate
        // each importing the same PBO) must agree, or the per-context import
        // cache misses on every wrap instead of reusing the bound texture.
        let ops1 = MockPboOps::new(8);
        let ops2 = MockPboOps::new(8);
        let t1 = PboTensor::<u8>::from_pbo(7, 8, &[8], None, ops1).unwrap();
        let t2 = PboTensor::<u8>::from_pbo(7, 8, &[8], None, ops2).unwrap();
        assert_eq!(t1.buffer_identity().id(), t2.buffer_identity().id());
    }

    #[test]
    fn test_pbo_tensor_new_returns_error() {
        let result = PboTensor::<u8>::new(&[8], None);
        assert!(result.is_err(), "PboTensor::new() should fail");
    }

    #[cfg(unix)]
    #[test]
    fn test_pbo_tensor_fd_ops_return_error() {
        let ops = MockPboOps::new(8);
        let tensor = PboTensor::<u8>::from_pbo(1, 8, &[8], None, ops).unwrap();
        assert!(tensor.clone_fd().is_err());
    }

    #[test]
    fn test_pbo_tensor_from_pbo_size_mismatch() {
        let ops = MockPboOps::new(24);
        let result = PboTensor::<u8>::from_pbo(1, 24, &[2, 3, 5], None, ops);
        assert!(result.is_err(), "mismatched size/shape should fail");
    }

    #[test]
    fn test_pbo_tensor_from_pbo_zero_size() {
        let ops = MockPboOps::new(0);
        let result = PboTensor::<u8>::from_pbo(1, 0, &[0], None, ops);
        assert!(result.is_err(), "zero size should fail");
    }

    #[test]
    fn test_pbo_via_tensor_enum() {
        let ops = MockPboOps::new(12);
        let pbo = PboTensor::<u8>::from_pbo(10, 12, &[3, 4], Some("enum_test"), ops).unwrap();
        let tensor = crate::Tensor::wrap(crate::TensorStorage::Pbo(pbo));
        assert_eq!(tensor.memory(), TensorMemory::Pbo);
        assert_eq!(tensor.name(), "enum_test");
        assert_eq!(tensor.shape(), &[3, 4]);
        let mut map = tensor.map().expect("map via enum");
        map.as_mut_slice().fill(42);
        assert!(map.as_slice().iter().all(|&b| b == 42));
    }

    /// The Task-11 acceptance criterion at the Rust level: export a
    /// PBO-backed tensor's descriptor, reconstruct it, and read back real
    /// bytes -- not just a non-error. Models the producer/consumer split
    /// the cross-package capsule protocol exists for (`edgefirst.image`
    /// allocates and writes; a different `cdylib`, `edgefirst.codec`,
    /// reconstructs from just the `TensorDesc` and reads real bytes back),
    /// but both halves run in this one process, which is all `cargo test`
    /// can exercise. The genuine two-`.so` case -- the reason this needed a
    /// `#[repr(C)]` vtable rather than a same-process registry -- is what
    /// `tests/interop/test_cross_package.py::
    /// test_decode_into_gpu_backed_tensor_from_another_package` and
    /// `tests/image/test_image.py::test_decode_image_pipeline` prove.
    #[test]
    fn cross_package_pbo_round_trips_through_tensor_desc_with_real_bytes() {
        let ops = MockPboOps::new(8);
        let dyn_ops: Arc<dyn PboOps> = ops.clone();
        let producer = PboTensor::<u8>::from_pbo(42, 8, &[8], None, dyn_ops).unwrap();
        {
            let mut m = producer.map().expect("map producer");
            m.as_mut_slice().copy_from_slice(&[1, 2, 3, 4, 5, 6, 7, 8]);
        }

        // The producer side: build a real TensorDesc the way
        // `Tensor.__edgefirst_tensor__()` does under the hood.
        let producer_dyn =
            crate::TensorDyn::U8(crate::Tensor::wrap(crate::TensorStorage::Pbo(producer)));
        let desc = producer_dyn.descriptor();
        assert_eq!(desc.kind, crate::protocol::kind::PBO);
        assert_eq!(desc.handle, 42);
        assert!(
            !desc.ptr.is_null(),
            "a PBO descriptor must carry an ops vtable -- see TensorDesc::ptr's doc comment"
        );

        // The consumer side: reconstruct from ONLY the descriptor, exactly
        // what `import_descriptor` does for a genuinely foreign handle --
        // no access to `producer_dyn` itself from here on.
        let imported = crate::TensorDyn::import_descriptor(&desc)
            .expect("import_descriptor must reconstruct a PBO-backed tensor from its own vtable");
        assert_eq!(imported.memory(), TensorMemory::Pbo);

        let m = imported
            .map_bytes(crate::CpuAccess::Read)
            .expect("map the reconstructed tensor");
        assert_eq!(
            m.as_slice(),
            &[1u8, 2, 3, 4, 5, 6, 7, 8],
            "the reconstructed tensor must read the SAME bytes the producer wrote through the \
             SAME GL buffer, not a copy or garbage"
        );
    }

    /// Hazard 1, made structurally moot rather than merely handled: GL
    /// buffer ids are scoped per-context, so two DIFFERENT contexts can
    /// legitimately hand out the SAME buffer_id for UNRELATED buffers. A
    /// registry keyed on buffer_id alone would resolve the second import to
    /// the first context's ops -- a wrong-buffer bug with no error. This
    /// design has no id to key anything on: the descriptor's `ptr` IS the
    /// address of the specific `PboHandle` it was built from, so two
    /// tensors sharing a buffer_id (simulating two independent contexts
    /// that happen to collide on the number) cannot cross-resolve, full
    /// stop -- there is no shared table for them to collide *in*.
    #[test]
    fn two_pbo_tensors_sharing_a_buffer_id_do_not_cross_resolve() {
        let ops_a = MockPboOps::new(4);
        let ops_b = MockPboOps::new(4);
        ops_a.storage.lock().unwrap().copy_from_slice(&[0xAA; 4]);
        ops_b.storage.lock().unwrap().copy_from_slice(&[0xBB; 4]);
        let dyn_a: Arc<dyn PboOps> = ops_a.clone();
        let dyn_b: Arc<dyn PboOps> = ops_b.clone();

        // SAME buffer_id, two entirely unrelated backing buffers -- exactly
        // the scenario two independent GL contexts could produce.
        let tensor_a = PboTensor::<u8>::from_pbo(7, 4, &[4], None, dyn_a).unwrap();
        let tensor_b = PboTensor::<u8>::from_pbo(7, 4, &[4], None, dyn_b).unwrap();

        let dyn_a_wrapped =
            crate::TensorDyn::U8(crate::Tensor::wrap(crate::TensorStorage::Pbo(tensor_a)));
        let dyn_b_wrapped =
            crate::TensorDyn::U8(crate::Tensor::wrap(crate::TensorStorage::Pbo(tensor_b)));
        let desc_a = dyn_a_wrapped.descriptor();
        let desc_b = dyn_b_wrapped.descriptor();
        assert_eq!(
            desc_a.handle, desc_b.handle,
            "both descriptors share the same buffer_id by construction"
        );
        assert_ne!(
            desc_a.ptr.0, desc_b.ptr.0,
            "but each carries its own, distinct vtable address -- there is no shared key \
             for them to collide on"
        );

        let imported_a = crate::TensorDyn::import_descriptor(&desc_a).unwrap();
        let imported_b = crate::TensorDyn::import_descriptor(&desc_b).unwrap();
        assert_eq!(
            imported_a
                .map_bytes(crate::CpuAccess::Read)
                .unwrap()
                .as_slice(),
            &[0xAA; 4],
            "importing desc_a must reach ops_a's buffer, never ops_b's"
        );
        assert_eq!(
            imported_b
                .map_bytes(crate::CpuAccess::Read)
                .unwrap()
                .as_slice(),
            &[0xBB; 4],
            "importing desc_b must reach ops_b's buffer, never ops_a's"
        );
    }

    /// Hazard 2: an `Arc<dyn PboOps>` registry would hold its own strong
    /// reference to `ops`, so the failure mode is a **leak** -- specifically
    /// a GL context kept alive past its own teardown, unless something
    /// explicitly removes the entry when the registering `PboTensor` drops.
    /// This design has no separate reference to remove: `PboOpsVtable::ctx`
    /// is a raw pointer into the `PboHandle`'s own `ops` field, not a fresh
    /// `Arc` clone, so building and exporting a descriptor must not change
    /// `ops`'s strong count at all, and dropping the producing tensor must
    /// release its one real reference immediately -- proven directly via
    /// `Arc::strong_count`, not merely "nothing crashed".
    #[test]
    fn dropping_the_producing_pbo_tensor_leaks_nothing_the_vtable_touched() {
        let ops = MockPboOps::new(8);
        let dyn_ops: Arc<dyn PboOps> = ops.clone();
        assert_eq!(
            Arc::strong_count(&ops),
            2,
            "this test's own handle, plus the clone about to move into from_pbo"
        );

        let producer = PboTensor::<u8>::from_pbo(99, 8, &[8], None, dyn_ops).unwrap();
        assert_eq!(
            Arc::strong_count(&ops),
            2,
            "from_pbo must not clone ops again"
        );

        let producer_dyn =
            crate::TensorDyn::U8(crate::Tensor::wrap(crate::TensorStorage::Pbo(producer)));
        // Force the vtable to build -- the exact operation hazard 2 worried
        // about: does exporting a descriptor create a NEW strong reference
        // to `ops` that nothing ever releases?
        let _desc = producer_dyn.descriptor();
        assert_eq!(
            Arc::strong_count(&ops),
            2,
            "building/exporting the vtable must not clone ops -- see PboOpsVtable's own \
             doc comment for why ctx is a raw pointer, not an Arc clone"
        );

        assert_eq!(ops.delete_count(), 0, "not dropped yet");
        drop(producer_dyn);
        assert_eq!(
            Arc::strong_count(&ops),
            1,
            "dropping the producing tensor must release its ops reference immediately -- \
             nothing the vtable machinery built kept it alive past that"
        );
        assert_eq!(
            ops.delete_count(),
            1,
            "the real delete_buffer must still fire exactly once, from PboHandle::Drop, \
             completely unaffected by whether a descriptor was ever exported"
        );
    }

    /// The callback channel a `PboOpsVtable` hands out is self-sufficient:
    /// an importer that retains it keeps the producer's `PboHandle` — and
    /// therefore the real GL buffer — alive after the producing tensor has
    /// dropped, and the buffer is deleted exactly once, when the last
    /// reference goes.
    ///
    /// This is the case today's design cannot express. `PboOpsVtable::new`
    /// used to hand out a bare borrowed pointer with the lifetime enforced
    /// *externally* by a `pbo_keepalive()` `Arc` some other object had to
    /// hold; nothing in the vtable itself could extend it. A child that
    /// outlived its parent read freed memory. Against a change that added
    /// the retain but forgot the `release` in `ImportedPboOps::drop`,
    /// `delete_count()` stays 0 and the LAST assertion fails instead.
    #[test]
    fn an_imported_channel_keeps_the_producers_buffer_alive() {
        let ops = MockPboOps::new(64);
        let producer = PboTensor::<u8>::from_pbo(21, 64, &[64], None, ops.clone())
            .expect("PboTensor::from_pbo");
        let vtable_ptr = producer.pbo_vtable() as *const PboOpsVtable as *const c_void;

        // SAFETY: `vtable_ptr` came from `pbo_vtable()` on a live tensor.
        let parts = unsafe { read_pbo_vtable_parts(vtable_ptr) }.expect("read_pbo_vtable_parts");
        // SAFETY: the parts came from a table this module built.
        let imported = unsafe { client_state_pbo_ops(parts.state, parts.map_fn, parts.unmap_fn) }
            .expect("client_state_pbo_ops");
        let child = PboTensor::<u8>::from_pbo(21, 64, &[64], None, imported)
            .expect("reconstruct across the boundary");

        // The producing tensor goes away first — the case the external
        // keepalive design could not express at all.
        drop(producer);
        assert_eq!(
            ops.delete_count(),
            0,
            "the producer's PboHandle must still be alive: the imported \
             channel holds a reference to it"
        );

        // The child still works through the retained channel.
        {
            let mut m = child.map().expect("map through the imported channel");
            m.as_mut_slice()[0] = 0xA5;
        }
        assert_eq!(ops.map_count(), 1, "the map went through the real ops");

        drop(child);
        assert_eq!(
            ops.delete_count(),
            1,
            "the GL buffer is deleted exactly once, when the last reference \
             to the channel goes"
        );
    }

    /// `into_client_parts` hands out a *borrowed* channel: the parts carry
    /// the producer's own `Arc`, so a callee that retains before they drop
    /// keeps the GL buffer, and a callee that never retains does not.
    ///
    /// This is the shape `ef_tensor_wrap_pbo` consumes, exercised here
    /// against `client_state_pbo_ops` — the same assembly point a C caller
    /// reaches.
    #[test]
    fn client_parts_hand_over_a_channel_the_callee_retains() {
        let ops = MockPboOps::new(32);
        let tensor = PboTensor::<u8>::from_pbo(7, 32, &[32], None, ops.clone())
            .expect("PboTensor::from_pbo");
        let parts = tensor.into_client_parts().expect("into_client_parts");
        // Before the reattach, and on its own line: consuming the tensor must
        // not have dropped the channel. `PboClientParts::_handle` is what
        // keeps it, and a regression that removed that field would otherwise
        // survive to the `client_state_pbo_ops` call below and retain a dead
        // `Arc` -- undefined behaviour instead of a test failure.
        assert_eq!(
            ops.delete_count(),
            0,
            "into_client_parts must hand over a live channel, not a dead one"
        );
        assert_eq!(parts.buffer_id, 7);
        assert_eq!(parts.size, 32);
        assert_eq!(parts.shape, vec![32]);

        // SAFETY: the parts came from a live `PboTensor`, so they satisfy
        // `client_state_pbo_ops`'s contract by construction.
        let reattached = unsafe { client_state_pbo_ops(parts.state, parts.map_fn, parts.unmap_fn) }
            .expect("client_state_pbo_ops");

        // The producer's own last reference goes. The callee retained, so
        // the GL buffer is still there.
        drop(parts);
        assert_eq!(
            ops.delete_count(),
            0,
            "the callee's retain must outlive the parts it was handed"
        );

        reattached
            .map_buffer(7, 32)
            .expect("map through the channel");
        reattached.unmap_buffer(7).expect("unmap");
        assert_eq!(ops.map_count(), 1);

        drop(reattached);
        assert_eq!(
            ops.delete_count(),
            1,
            "the buffer goes exactly once, when the last reference does"
        );
    }

    /// A sub-view has a window `ef_tensor_wrap_pbo`'s argument list cannot
    /// carry, so it is refused rather than silently wrapped at the parent's
    /// origin — #162 in reverse.
    #[test]
    fn client_parts_refuse_a_sub_view() {
        let ops = MockPboOps::new(32);
        let tensor =
            PboTensor::<u8>::from_pbo(8, 32, &[32], None, ops).expect("PboTensor::from_pbo");
        let view = tensor.view(16, &[16]).expect("view");
        let Err(err) = view.into_client_parts() else {
            panic!("a sub-view must not wrap silently");
        };
        assert!(
            matches!(err, Error::InvalidOperation(_)),
            "expected InvalidOperation, got {err:?}"
        );
    }

    /// `TensorDyn::from_client_pbo` refuses a NULL `ctx`, `retain` or
    /// `release` with `Error::InvalidArgument` — the **static twin** of
    /// `dynamic_primitives.rs`'s
    /// `dynamic_from_client_pbo_refuses_an_incomplete_channel`.
    ///
    /// Both assert the same variant on the same call deliberately: a
    /// consumer must not get a different `Error` for the same malformed
    /// channel depending on which backend it linked. Under `dynamic` that
    /// only holds because the C entry point classifies the failure and
    /// `ffi_error` rebuilds it from the class rather than guessing.
    #[test]
    fn from_client_pbo_refuses_an_incomplete_channel() {
        let ops = MockPboOps::new(16);
        let tensor =
            PboTensor::<u8>::from_pbo(9, 16, &[16], None, ops).expect("PboTensor::from_pbo");
        let parts = tensor.into_client_parts().expect("into_client_parts");
        // A genuine `ctx`, for the reason
        // `a_client_state_missing_any_part_is_refused` gives below.
        let full = parts.state;
        for (name, state) in [
            (
                "ctx",
                EfClientState {
                    ctx: std::ptr::null(),
                    ..full
                },
            ),
            (
                "retain",
                EfClientState {
                    retain: None,
                    ..full
                },
            ),
            (
                "release",
                EfClientState {
                    release: None,
                    ..full
                },
            ),
        ] {
            // SAFETY: `full` came from a live `PboTensor`, so every variant
            // of it is a state this constructor may legally inspect.
            let r = unsafe {
                crate::TensorDyn::from_client_pbo(
                    state,
                    parts.map_fn,
                    parts.unmap_fn,
                    9,
                    16,
                    &[16],
                    crate::DType::U8,
                )
            };
            let Err(err) = r else {
                panic!("a NULL {name} must be refused");
            };
            assert!(
                matches!(err, Error::InvalidArgument(_)),
                "NULL {name}: expected InvalidArgument, got {err:?}"
            );
        }
    }

    /// The lower half of the same contract: `client_state_pbo_ops` itself,
    /// which both backends' `from_client_pbo` delegate to. A NULL `ctx`,
    /// `retain` or `release` is a loud `InvalidArgument`, not a silently
    /// non-owning attach — spec §6, the failure class this design exists to
    /// remove.
    #[test]
    fn a_client_state_missing_any_part_is_refused() {
        let ops = MockPboOps::new(16);
        let tensor =
            PboTensor::<u8>::from_pbo(9, 16, &[16], None, ops).expect("PboTensor::from_pbo");
        let parts = tensor.into_client_parts().expect("into_client_parts");
        // A *genuine* `ctx` on purpose. If one of the checks below ever
        // regressed, the call would retain a real `Arc` -- a refcount the
        // `else` arm's panic reports cleanly -- rather than drive
        // `Arc::increment_strong_count` through a forged pointer, which
        // would turn a test failure into undefined behaviour.
        let full = parts.state;
        for (name, state) in [
            (
                "ctx",
                EfClientState {
                    ctx: std::ptr::null(),
                    ..full
                },
            ),
            (
                "retain",
                EfClientState {
                    retain: None,
                    ..full
                },
            ),
            (
                "release",
                EfClientState {
                    release: None,
                    ..full
                },
            ),
        ] {
            // SAFETY: `full` came from a live `PboTensor`, so every variant
            // of it is a state `client_state_pbo_ops` may legally inspect.
            let Err(err) = (unsafe { client_state_pbo_ops(state, parts.map_fn, parts.unmap_fn) })
            else {
                panic!("a NULL {name} must be refused");
            };
            assert!(
                matches!(err, Error::InvalidArgument(_)),
                "NULL {name}: expected InvalidArgument, got {err:?}"
            );
        }
    }

    /// `set_plane_offset` must move a PBO's CPU-map window, not merely
    /// record a number on the wrapper.
    ///
    /// The PBO mirror of `set_plane_offset_moves_the_iosurface_map_window`
    /// (`crates/tensor/src/iosurface.rs`). `PboTensor::view_offset` is the
    /// offset `scoped_pin` adds to the mapped base, and `PboTensor::view`
    /// already set it on the fresh-view path — so, exactly like IOSurface,
    /// only *restoring* an offset onto an already-built tensor was broken,
    /// which is why a freshly created view was right and only the
    /// descriptor round trip was wrong. Before the `Pbo` arm this falls
    /// into `set_plane_offset`'s `_ => {}` and the map starts at the
    /// parent's origin.
    #[test]
    fn set_plane_offset_moves_the_pbo_map_window() {
        use crate::{Tensor, TensorTrait};

        const PARENT: usize = 256;
        const OFFSET: usize = 64;
        const WINDOW: usize = 64;

        // Ramp the buffer so byte i holds i: a window at the wrong origin
        // cannot coincidentally match the expected bytes.
        let ops = MockPboOps::new(PARENT);
        let whole = Tensor::<u8>::from_pbo(
            PboTensor::<u8>::from_pbo(31, PARENT, &[PARENT], None, ops.clone()).expect("from_pbo"),
        )
        .expect("wrap");
        {
            let mut m = whole.map().expect("map the whole buffer");
            for (i, b) in m.as_mut_slice().iter_mut().enumerate() {
                *b = (i & 0xff) as u8;
            }
        }

        // Re-wrap the WHOLE buffer, as an import does: the handle names the
        // parent, so this lands at its origin.
        let mut rebuilt = Tensor::<u8>::from_pbo(
            PboTensor::<u8>::from_pbo(31, PARENT, &[PARENT], None, ops.clone()).expect("from_pbo"),
        )
        .expect("wrap");
        rebuilt
            .set_logical_shape(&[WINDOW])
            .expect("narrow to the window's shape");
        {
            let m = rebuilt.map().expect("map before the offset");
            assert_eq!(
                m.as_slice()[0],
                0,
                "precondition: a bare wrap starts at the parent's origin"
            );
        }

        // The restore under test.
        rebuilt.set_plane_offset(OFFSET);
        assert_eq!(
            rebuilt.plane_offset(),
            Some(OFFSET),
            "the wrapper field records the offset on every backing"
        );

        let m = rebuilt.map().expect("map after the offset");
        let s = m.as_slice();
        assert_eq!(s.len(), WINDOW, "the window exposes its logical length");
        assert_ne!(
            s[0], 0,
            "map() still starts at the parent's origin — set_plane_offset was \
             a no-op on PBO storage (issue #161)"
        );
        for (i, b) in s.iter().enumerate() {
            assert_eq!(
                *b,
                ((i + OFFSET) & 0xff) as u8,
                "window byte {i} must be parent[{}]",
                i + OFFSET
            );
        }
    }

    /// A nested `subview` must not compound its offset now that
    /// `set_plane_offset` writes through to PBO storage.
    ///
    /// The double-apply guard, and the reason it needs checking rather than
    /// assuming: `Tensor::subview` sets the offset *twice* by two different
    /// routes — `PboTensor::view` computes `view_offset + offset_bytes`
    /// inside `TensorStorage::view`, and then `subview` calls
    /// `set_plane_offset(plane_offset + offset_bytes)` on the result. Those
    /// are the same number only while the wrapper field and the storage
    /// field stay in sync, which is what the new arm makes true. Two levels,
    /// because a single level cannot tell an idempotent write apart from one
    /// that compounds from zero.
    #[test]
    fn nested_pbo_subviews_do_not_compound_their_plane_offset() {
        use crate::{Tensor, TensorTrait};

        let ops = MockPboOps::new(256);
        let parent = Tensor::<u8>::from_pbo(
            PboTensor::<u8>::from_pbo(32, 256, &[256], None, ops).expect("from_pbo"),
        )
        .expect("wrap");
        {
            let mut m = parent.map().expect("map parent");
            for (i, b) in m.as_mut_slice().iter_mut().enumerate() {
                *b = (i & 0xff) as u8;
            }
        }

        let first = parent.subview(32, &[128]).expect("first subview");
        assert_eq!(first.plane_offset(), Some(32));
        let second = first.subview(16, &[64]).expect("nested subview");
        assert_eq!(
            second.plane_offset(),
            Some(48),
            "a nested subview composes 32 + 16, it does not compound"
        );
        let m = second.map().expect("map the nested view");
        for (i, b) in m.as_slice().iter().enumerate() {
            assert_eq!(*b, ((i + 48) & 0xff) as u8, "nested byte {i}");
        }
    }

    /// `set_format` must clear the PBO's storage-internal window, not only
    /// the wrapper field.
    ///
    /// The clear that pairs with the setter above: without it, changing the
    /// format leaves `plane_offset() == None` while `map()` still starts at
    /// the old offset — a *stale* window rather than a lost one, which is
    /// strictly harder to notice.
    #[test]
    fn set_format_clears_the_pbo_map_window() {
        use crate::{PixelFormat, Tensor, TensorTrait};

        let ops = MockPboOps::new(256);
        let mut t = Tensor::<u8>::from_pbo(
            PboTensor::<u8>::from_pbo(33, 256, &[256], None, ops).expect("from_pbo"),
        )
        .expect("wrap");
        {
            let mut m = t.map().expect("map");
            for (i, b) in m.as_mut_slice().iter_mut().enumerate() {
                *b = (i & 0xff) as u8;
            }
        }
        t.set_logical_shape(&[8, 8, 1]).expect("narrow");
        t.set_plane_offset(64);

        t.set_format(PixelFormat::Grey).expect("set_format");
        assert_eq!(t.plane_offset(), None, "the wrapper field is cleared");
        let m = t.map().expect("map after set_format");
        assert_eq!(
            m.as_slice()[0],
            0,
            "map() must be back at the origin: a cleared wrapper field over a \
             live storage offset is a stale window"
        );
    }
}
