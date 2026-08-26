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
use std::{os::raw::c_int, sync::OnceLock};

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
/// leak, no registry entry to remove on drop.
///
/// Compiled into both backends. It was `static`-only while the
/// cross-package capsule protocol had no `dynamic` counterpart; task P2a
/// gave it one (`TensorDyn::descriptor_pinned`/`import_descriptor` now
/// exist on both), so the gate came off. Nothing here is
/// backend-specific — it is a C-ABI view onto `PboOps`, which was always
/// a backend-agnostic GL extension point.
#[repr(C)]
pub struct PboOpsVtable {
    /// Opaque to every caller except this struct's own function pointers —
    /// a caller crossing a `cdylib` boundary calls through them and never
    /// dereferences `ctx` itself.
    ctx: *const c_void,
    map_buffer_fn: unsafe extern "C" fn(
        ctx: *const c_void,
        buffer_id: u32,
        size: usize,
        out_ptr: *mut *mut u8,
        out_len: *mut usize,
    ) -> c_int,
    unmap_buffer_fn: unsafe extern "C" fn(ctx: *const c_void, buffer_id: u32) -> c_int,
    // No `delete_buffer_fn`: deliberately absent, not merely unused. A
    // consumer reconstructing a `PboTensor` from this vtable does not own
    // the GL buffer -- it borrows it for the descriptor's lifetime (see
    // `ImportedPboOps`'s own doc comment) -- so there is no legitimate
    // caller for it on that side, and the producer's own `PboHandle::Drop`
    // already calls the real `delete_buffer` directly through its own
    // `Arc<dyn PboOps>`, never through this vtable. Including a pointer
    // nothing may safely call is worse than omitting it.
}

// SAFETY: every field is either a raw pointer this struct never
// dereferences itself (only its own `extern "C" fn`s do, and those forward
// into `PboOps`, itself `Send + Sync`) or a plain function pointer — both
// safe to share and move across threads.
unsafe impl Send for PboOpsVtable {}
unsafe impl Sync for PboOpsVtable {}

impl PboOpsVtable {
    /// Build a vtable dispatching into `ops`. `ctx` addresses `ops` itself,
    /// so `ops` must already be at its final, stable address — a field of a
    /// `PboHandle` that already lives inside its own `Arc` (see
    /// [`PboTensor::pbo_vtable`]), since nothing here pins or moves it.
    fn new(ops: &Arc<dyn PboOps>) -> Self {
        PboOpsVtable {
            ctx: ops as *const Arc<dyn PboOps> as *const c_void,
            map_buffer_fn: vt_map_buffer,
            unmap_buffer_fn: vt_unmap_buffer,
        }
    }

    /// # Safety
    /// `self` must be a table this module built (via [`Self::new`]) whose
    /// `ctx` is still live — the `PboHandle` it addresses must not have
    /// dropped. A raw pointer reconstructed from a [`crate::TensorDesc`]
    /// satisfies this for exactly as long as that descriptor's own borrow
    /// contract does.
    unsafe fn map_buffer(&self, buffer_id: u32, size: usize) -> Result<PboMapping> {
        let mut ptr: *mut u8 = std::ptr::null_mut();
        let mut len: usize = 0;
        // SAFETY: caller's contract above.
        let rc = unsafe { (self.map_buffer_fn)(self.ctx, buffer_id, size, &mut ptr, &mut len) };
        if rc != 0 {
            return Err(vtable_errno_to_error(rc));
        }
        Ok(PboMapping { ptr, size: len })
    }

    /// # Safety
    /// Same as [`Self::map_buffer`].
    unsafe fn unmap_buffer(&self, buffer_id: u32) -> Result<()> {
        // SAFETY: caller's contract above.
        let rc = unsafe { (self.unmap_buffer_fn)(self.ctx, buffer_id) };
        if rc != 0 {
            return Err(vtable_errno_to_error(rc));
        }
        Ok(())
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
        // SAFETY: the caller's contract (`PboOpsVtable::map_buffer`)
        // guarantees `ctx` still addresses a live `Arc<dyn PboOps>`.
        let ops = unsafe { &*(ctx as *const Arc<dyn PboOps>) };
        ops.map_buffer(buffer_id, size)
    }));
    match result {
        Ok(Ok(mapping)) => {
            // SAFETY: `out_ptr`/`out_len` are valid out-params for the
            // duration of this call, per this function's own contract with
            // its caller (`PboOpsVtable::map_buffer`).
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
        let ops = unsafe { &*(ctx as *const Arc<dyn PboOps>) };
        ops.unmap_buffer(buffer_id)
    }));
    match result {
        Ok(Ok(())) => 0,
        Ok(Err(e)) => error_to_vtable_errno(&e),
        Err(_) => 1,
    }
}

/// Adapts a [`PboOpsVtable`] reconstructed from a cross-package
/// [`crate::TensorDesc`] back into a [`PboOps`] implementation, so
/// [`PboTensor::from_pbo`] needs no separate cross-package constructor —
/// [`import_pbo_ops`] builds the exact same `Arc<dyn PboOps>` shape this
/// crate already uses for a same-process PBO.
///
/// `delete_buffer` is deliberately a no-op: this tensor does not own the
/// GL buffer, it borrows it for the descriptor's lifetime (the producer's
/// own capsule-keepalive contract), so dropping it must not delete a
/// buffer the producer's own tensor may still be using. The producer's own
/// `PboHandle::Drop` is the only thing that ever calls the real
/// `delete_buffer`.
struct ImportedPboOps {
    vtable: NonNull<PboOpsVtable>,
}

// SAFETY: the vtable this addresses is `Send + Sync` (see its own impl),
// and this struct never mutates through the pointer, only calls through
// its function pointers, which are themselves safe to call from any thread
// (see `PboOps`'s own trait-level contract, which the real implementation
// behind this vtable already upholds).
unsafe impl Send for ImportedPboOps {}
unsafe impl Sync for ImportedPboOps {}

// SAFETY: forwards every call through `PboOpsVtable`'s own dispatch, which
// forwards into a real `PboOps` implementation upholding this trait's
// contract already — this wrapper adds no new unsafety beyond the pointer
// dereference `import_pbo_ops`'s own safety contract covers.
unsafe impl PboOps for ImportedPboOps {
    fn map_buffer(&self, buffer_id: u32, size: usize) -> Result<PboMapping> {
        // SAFETY: `import_pbo_ops` (this type's only constructor) upholds
        // `PboOpsVtable::map_buffer`'s safety contract.
        unsafe { self.vtable.as_ref().map_buffer(buffer_id, size) }
    }

    fn unmap_buffer(&self, buffer_id: u32) -> Result<()> {
        // SAFETY: see `map_buffer` above.
        unsafe { self.vtable.as_ref().unmap_buffer(buffer_id) }
    }

    fn delete_buffer(&self, _buffer_id: u32) {
        // No-op: see this type's own doc comment.
    }
}

/// Reconstruct a live [`PboOps`] from a [`crate::TensorDesc::ptr`] carrying
/// a [`PboOpsVtable`] address under [`crate::protocol::kind::PBO`].
///
/// # Safety
/// `vtable_ptr` must be a genuine `*const PboOpsVtable` this module built
/// (i.e. it came from a real `TensorDesc.ptr` under `kind::PBO`, not a
/// forged or reinterpreted value), and the `PboHandle` it addresses must
/// stay alive for as long as the returned `Arc<dyn PboOps>` (and anything
/// built from it) is used — the same capsule-keepalive contract
/// `TensorDesc::ptr` already documents for every other kind.
pub(crate) unsafe fn import_pbo_ops(vtable_ptr: *const c_void) -> Result<Arc<dyn PboOps>> {
    let Some(vtable) = NonNull::new(vtable_ptr as *mut PboOpsVtable) else {
        return Err(Error::InvalidArgument(
            "PBO descriptor carries no ops vtable".into(),
        ));
    };
    Ok(Arc::new(ImportedPboOps { vtable }))
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
    /// registry, which cannot cross a `cdylib` boundary at all). Present on
    /// both backends — see [`PboOpsVtable`]'s own doc comment.
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
        let expected = shape.iter().product::<usize>() * std::mem::size_of::<T>();
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
    /// comment. Called from both backends' `pbo_vtable_ptr`
    /// (`lib.rs`/`static_backend.rs`, and `dynamic_backend.rs` via
    /// `with_pbo!`).
    pub(crate) fn pbo_vtable(&self) -> &PboOpsVtable {
        self.handle
            .abi_vtable
            .get_or_init(|| PboOpsVtable::new(&self.handle.ops))
    }

    /// A type-erased keepalive holding this PBO's `Arc<PboHandle>` alive --
    /// the same shape as [`crate::pin::HostPin`]'s own `Keepalive` (an
    /// `Arc<dyn Send + Sync>`), for the identical reason: [`PboOpsVtable::
    /// new`] hands out `ctx`, a raw pointer straight into this `PboHandle`'s
    /// own `ops` field, and that pointer is only valid for as long as the
    /// `PboHandle` it addresses is alive. Cloning this `Arc` (not the
    /// `PboHandle`'s *contents*) is what a cross-package capsule holds
    /// alongside the descriptor, exactly the way `pin` already does for the
    /// `HOST` kind -- see `TensorCapsulePayload` (`edgefirst-python-common`).
    /// Type-erased so this crate's public surface never has to name
    /// `PboHandle`, which stays private.
    ///
    /// Called from both backends' `pbo_keepalive`
    /// (`lib.rs`/`static_backend.rs`, and `dynamic_backend.rs` via
    /// `with_pbo!`).
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
        let new_size = shape.iter().product::<usize>() * std::mem::size_of::<T>();
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
        let needed = shape.iter().product::<usize>() * std::mem::size_of::<T>();
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
        let logical = shape.iter().product::<usize>() * std::mem::size_of::<T>();
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
}
