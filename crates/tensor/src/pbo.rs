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
// Used by `PboOpsVtable`.
use std::os::raw::c_int;
// `PboHandle::abi_vtable`'s cell.
#[cfg(not(edgefirst_tensor_dynamic))]
use std::sync::OnceLock;
// The client-state vocabulary the table is expressed over.
pub use edgefirst_tensor_abi::{EfClientState, EfPboMapFn, EfPboUnmapFn};

/// The refusal both backends give when asked to pin a PBO's host address.
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

// SAFETY: glMapBufferRange returns a CPU-visible pointer that can be accessed
// from any thread while the buffer remains mapped.
unsafe impl Send for PboMapping {}

/// Trait for PBO GL operations, implemented by the image crate.
///
/// All methods are blocking. Implementations must ensure the GL context is
/// current on the thread that executes the GL calls.
///
/// # Safety
///
/// Implementations must ensure:
/// - `map_buffer` returns a valid, aligned pointer to `size` bytes of
///   CPU-accessible memory that remains valid until `unmap_buffer` is called.
/// - `unmap_buffer` invalidates the pointer and releases the mapping.
/// - `delete_buffer` frees the GL buffer resources.
/// - No method unwinds.
pub unsafe trait PboOps: Send + Sync {
    /// Map the PBO for CPU read/write access.
    /// The returned PboMapping is valid until `unmap_buffer` is called.
    ///
    /// A returned `PboMapping::size` below `size` is treated as a failed
    /// map: `PboHandle::acquire_map` refuses it and unmaps the buffer.
    fn map_buffer(&self, buffer_id: u32, size: usize) -> Result<PboMapping>;

    /// Unmap a previously mapped PBO. Must be called before GL operations
    /// on this buffer (GLES 3.0 requirement).
    fn unmap_buffer(&self, buffer_id: u32) -> Result<()>;

    /// Delete the PBO; no reply is returned.
    /// Called from PboTensor's Drop impl.
    fn delete_buffer(&self, buffer_id: u32);
}

/// C-ABI-safe callback table for a PBO's GL operations: `#[repr(C)]` plus
/// `extern "C" fn` pointers dispatching into a [`PboOps`], so an
/// independently-compiled `cdylib` can call it.
///
/// A `PboOpsVtable` is built once, lazily, as a field of the `PboHandle`
/// whose `ops` it dispatches into (see [`PboTensor::pbo_vtable`]), and its
/// address is handed out through [`crate::TensorDesc::ptr`] under
/// [`crate::protocol::kind::PBO`]. It lives as long as that `PboHandle`. An
/// importer calls `state.retain` while reading the table and holds its own
/// reference from then on (see [`client_state_pbo_ops`]).
#[repr(C)]
pub struct PboOpsVtable {
    /// The callback channel: an opaque `ctx` plus the pair that extends its
    /// life. `ctx` addresses the producing [`PboHandle`]; only this module's
    /// own `extern "C"` functions dereference it.
    ///
    /// Borrowed, not owned: [`Self::new`] takes no reference of its own. An
    /// importer takes its own reference through `state.retain`; see
    /// [`client_state_pbo_ops`].
    state: EfClientState,
    map_buffer_fn: EfPboMapFn,
    unmap_buffer_fn: EfPboUnmapFn,
}

// SAFETY: every field is either a raw pointer this struct never dereferences
// itself or a plain function pointer.
unsafe impl Send for PboOpsVtable {}
unsafe impl Sync for PboOpsVtable {}

impl PboOpsVtable {
    /// Build a vtable dispatching into `handle`'s `ops`.
    ///
    /// `ctx` is the `PboHandle`'s own address, taken with `Arc::as_ptr`. No
    /// reference is taken here.
    #[cfg(not(edgefirst_tensor_dynamic))]
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

/// The vtable's return codes: `0` success; `-1` the GL context is gone
/// ([`Error::PboDisconnected`]); any other value a generic failure.
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

/// The `extern "C"` map-buffer dispatch of [`PboOpsVtable::new`].
/// `catch_unwind`-shielded.
unsafe extern "C" fn vt_map_buffer(
    ctx: *const c_void,
    buffer_id: u32,
    size: usize,
    out_ptr: *mut *mut u8,
    out_len: *mut usize,
) -> c_int {
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        // SAFETY: the caller guarantees `ctx` addresses a live `PboHandle`.
        let handle = unsafe { &*(ctx as *const PboHandle) };
        handle.ops.map_buffer(buffer_id, size)
    }));
    match result {
        Ok(Ok(mapping)) => {
            // SAFETY: `out_ptr`/`out_len` are valid out-params for the
            // duration of this call.
            unsafe {
                *out_ptr = mapping.ptr;
                *out_len = mapping.size;
            }
            0
        }
        Ok(Err(e)) => error_to_vtable_errno(&e),
        Err(_) => 1, // a caught panic reports a generic failure
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
/// `catch_unwind`-shielded.
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
/// implementation.
///
/// Construction calls `state.retain` and [`Drop`] calls `state.release`, so
/// this value keeps the producing `PboHandle` alive.
///
/// `delete_buffer` is a no-op: this value does not own the GL buffer.
struct ImportedPboOps {
    state: EfClientState,
    map_buffer_fn: EfPboMapFn,
    unmap_buffer_fn: EfPboUnmapFn,
}

// SAFETY: the parts this holds are a raw pointer it never dereferences
// itself plus plain function pointers that are safe to call from any thread.
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
// uphold this trait's contract.
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
        // No-op.
    }
}

/// The three parts of a [`PboOpsVtable`], read out of a descriptor's `ptr`
/// or supplied directly by a C caller.
pub struct PboOpsVtableParts {
    pub state: EfClientState,
    pub map_fn: EfPboMapFn,
    pub unmap_fn: EfPboUnmapFn,
}

/// Read a [`crate::TensorDesc::ptr`]'s vtable into its parts.
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
/// functions, whether they arrived through a [`crate::TensorDesc`] under
/// [`crate::protocol::kind::PBO`] or through `ef_tensor_wrap_pbo`'s
/// arguments.
///
/// Takes its own reference on the channel (`state.retain`) and releases it
/// when the returned `Arc`'s last clone drops.
///
/// # Errors
/// [`Error::InvalidArgument`] for a NULL `ctx`, `retain` or `release`.
///
/// # Safety
/// `state.ctx` must be a context the supplied `retain`/`release`
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
    // SAFETY: the caller's contract above.
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
    /// Wakes threads waiting out an in-flight GL map/unmap.
    map_cv: Condvar,
    /// C-ABI vtable dispatching into `ops`, for cross-cdylib export via
    /// [`crate::TensorDesc`]. Built once, lazily, on first
    /// [`PboTensor::pbo_vtable`] call.
    #[cfg(not(edgefirst_tensor_dynamic))]
    abi_vtable: OnceLock<PboOpsVtable>,
}

/// CPU-map state of a PBO's single GL buffer, within one `PboHandle`.
///
/// * `Unmapped` — no CPU mapping exists.
/// * `Exclusive` — one writable map holds it. Nothing else may map until
///   that map drops.
/// * `Shared` — `readers` read-only maps over one mapping, unmapped when
///   the last of them drops.
enum MapState {
    Unmapped,
    /// A thread is inside `PboOps::map_buffer` for this buffer; others wait.
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
    /// `map_state` is not held across a [`PboOps`] call. The in-flight GL
    /// call is published as [`MapState::Mapping`] / [`MapState::Unmapping`],
    /// and other threads wait on `map_cv` until it resolves.
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
                // Re-checked after the wait.
                MapState::Mapping | MapState::Unmapping => continue,
                MapState::Unmapped => *guard = MapState::Mapping,
            }
            drop(guard);

            // No lock is held here.
            let mapped = self.ops.map_buffer(self.buffer_id, self.size);
            let base = match mapped {
                // A short mapping is a failed mapping: the buffer is unmapped again.
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
                // GL reported success with a null pointer: the buffer is unmapped
                // while this thread holds the `Mapping` claim, then `Unmapped` is
                // published.
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
    /// holder is gone. The GL unmap runs with no lock held.
    fn release_map(&self) {
        let mut guard = self.map_state.lock().expect("PBO map state mutex poisoned");
        let last_holder = match &mut *guard {
            MapState::Shared { readers, .. } => {
                *readers = readers.saturating_sub(1);
                *readers == 0
            }
            MapState::Exclusive => true,
            // Nothing held: a release with no matching acquire.
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
    /// `BufferIdentity`.
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
    /// Called by the image crate after creating the PBO on the GL thread;
    /// `ImageProcessor::create_image()` is the user-facing entry point.
    ///
    /// # Errors
    ///
    /// Returns `Error::ShapeMismatch` if `size` is smaller than
    /// `shape.iter().product::<usize>() * std::mem::size_of::<T>()`.
    /// Returns `Error::InvalidSize` if `size` is zero.
    /// Returns `Error::InvalidShape` if that shape footprint overflows
    /// `usize`.
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
        // Overflow-checked shape footprint.
        let expected = crate::ahardwarebuffer_layout::checked_shape_bytes::<T>(shape)?;
        // `size >= expected` is accepted: a PBO with a 64-byte-aligned row stride
        // is larger than the shape product.
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
                #[cfg(not(edgefirst_tensor_dynamic))]
                abi_vtable: OnceLock::new(),
            }),
            // A GL buffer name is unique among its context's live PBOs.
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
    /// export via [`crate::TensorDesc`]. See [`PboOpsVtable`].
    #[cfg(not(edgefirst_tensor_dynamic))]
    pub(crate) fn pbo_vtable(&self) -> &PboOpsVtable {
        self.handle
            .abi_vtable
            .get_or_init(|| PboOpsVtable::new(&self.handle))
    }

    /// A type-erased keepalive holding this PBO's `Arc<PboHandle>` alive.
    /// `state.ctx` from [`PboOpsVtable::new`] is this `PboHandle`'s address and
    /// must be live when an importer reads the table and calls `state.retain`.
    #[cfg(not(edgefirst_tensor_dynamic))]
    pub(crate) fn pbo_keepalive(&self) -> Arc<dyn Send + Sync> {
        self.handle.clone()
    }

    /// Returns true unless the PBO is fully unmapped: an exclusive map, one
    /// or more read-only holders, or a map or unmap another thread has in
    /// flight.
    pub fn is_mapped(&self) -> bool {
        self.handle.is_mapped()
    }
}

/// The C-ABI parts `ef_tensor_wrap_pbo` takes, from a [`PboTensor`] at the
/// origin of its buffer (`view_offset == 0`).
///
/// `state.ctx` borrows the channel this value owns; the callee is expected
/// to `retain` it (which [`client_state_pbo_ops`] does) before this struct
/// drops. Dropping it without retaining releases the client's last
/// reference and deletes the GL buffer.
///
/// Neither `Send` nor `Sync`.
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
    /// value drops.
    _handle: Arc<PboHandle>,
}

impl<T> PboTensor<T>
where
    T: Num + Clone + fmt::Debug + Send + Sync,
{
    /// Decompose this tensor into the parts a C entry point takes.
    ///
    /// The shape is carried through as-is and need not fill the allocation:
    /// `size` is the GL buffer's full byte count and may exceed the shape's
    /// product.
    ///
    /// # Errors
    /// [`Error::InvalidOperation`] for a sub-view (`view_offset != 0`).
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

    /// Capacity-based reconfigure: allows any shape whose byte size fits the
    /// PBO allocation.
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
    /// from this tensor's own window with logical `shape`. A CPU map adds
    /// `view_offset` to the mapped base.
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
    /// bytes) rather than the shape-derived logical count. The caller checks
    /// `byte_size <= capacity_bytes()`.
    #[cfg(not(edgefirst_tensor_dynamic))]
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
        // Maps the full GL allocation (`handle.size`); the slice length is
        // narrowed by `byte_size_override` (or the logical shape) at access time.
        Ok(crate::view::HostView::new(
            self.scoped_pin(access)?,
            self.shape.clone(),
            byte_size_override,
            access,
        ))
    }
}

/// Non-null base address of a live GL buffer mapping.
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
/// `PboTensor::scoped_pin` acquires the map and this releases it when the last
/// clone of the pin drops.
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
    /// Acquire a host address valid for as long as the returned pin lives. A
    /// guard built on this pin releases the map when it drops.
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

    /// Mock PboOps that uses a Vec<u8> as backing storage instead of GL,
    /// counting map/unmap calls.
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
    // once in `new` and never resized, and the `MockPboOps` outlives every map
    // handed out.
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
    /// pointer.
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

    /// An implementation that reports success while mapping fewer bytes than
    /// were asked for.
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

    /// A short mapping is refused and the buffer released.
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

    /// A shape whose byte footprint wraps `usize` is refused.
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

    /// The three reshaping paths refuse an overflowing footprint.
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

    /// A map that succeeds with a null pointer releases the buffer and returns
    /// the state to unmapped.
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

        // The buffer can be claimed again.
        let Err(err) = tensor.map_read() else {
            panic!("still null, so it must still fail");
        };
        assert!(matches!(err, Error::InvalidSize(_)));
        assert_eq!(ops.unmap_count(), 2, "and released again");
    }

    /// Several read-only holders share one GL mapping.
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

    /// Writers and readers exclude each other.
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

        // Fully released: a fresh map succeeds.
        assert!(tensor.map_read().is_ok());
    }

    /// An explicit `unmap()` followed by `Drop` releases once.
    #[test]
    fn explicit_unmap_then_drop_releases_only_once() {
        let ops = MockPboOps::new(8);
        let dyn_ops: Arc<dyn PboOps> = ops.clone();
        let tensor = PboTensor::<u8>::from_pbo(11, 8, &[8], None, dyn_ops).unwrap();

        let keeper = tensor.map_read().expect("reader that must survive");
        let mut early = tensor.map_read().expect("reader unmapped by hand");
        early.unmap();
        drop(early); // Drop does not release a second time.

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
        // A 24-byte PBO can be reconfigured to any shape that fits.
        let ops = MockPboOps::new(24);
        let mut tensor = PboTensor::<u8>::from_pbo(7, 24, &[24], None, ops).unwrap();
        // Smaller-than-capacity logical shape is accepted.
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
        // Two independent `from_pbo` wraps of the same GL buffer name derive the
        // same identity.
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

    /// Exports a PBO-backed tensor's descriptor, reconstructs it, and reads
    /// back the bytes the producer wrote.
    #[test]
    fn cross_package_pbo_round_trips_through_tensor_desc_with_real_bytes() {
        let ops = MockPboOps::new(8);
        let dyn_ops: Arc<dyn PboOps> = ops.clone();
        let producer = PboTensor::<u8>::from_pbo(42, 8, &[8], None, dyn_ops).unwrap();
        {
            let mut m = producer.map().expect("map producer");
            m.as_mut_slice().copy_from_slice(&[1, 2, 3, 4, 5, 6, 7, 8]);
        }

        // The producer side: build a TensorDesc.
        let producer_dyn =
            crate::TensorDyn::U8(crate::Tensor::wrap(crate::TensorStorage::Pbo(producer)));
        let desc = producer_dyn.descriptor();
        assert_eq!(desc.kind, crate::protocol::kind::PBO);
        assert_eq!(desc.handle, 42);
        assert!(
            !desc.ptr.is_null(),
            "a PBO descriptor must carry an ops vtable -- see TensorDesc::ptr's doc comment"
        );

        // The consumer side: reconstruct from the descriptor alone.
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

    /// Two tensors sharing a buffer_id resolve to their own buffers: the
    /// descriptor's `ptr` is the address of the specific `PboHandle`.
    #[test]
    fn two_pbo_tensors_sharing_a_buffer_id_do_not_cross_resolve() {
        let ops_a = MockPboOps::new(4);
        let ops_b = MockPboOps::new(4);
        ops_a.storage.lock().unwrap().copy_from_slice(&[0xAA; 4]);
        ops_b.storage.lock().unwrap().copy_from_slice(&[0xBB; 4]);
        let dyn_a: Arc<dyn PboOps> = ops_a.clone();
        let dyn_b: Arc<dyn PboOps> = ops_b.clone();

        // The same buffer_id over two unrelated backing buffers.
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

    /// Building and exporting a descriptor leaves `ops`'s strong count
    /// unchanged, and dropping the producing tensor releases its reference.
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
        // Forces the vtable to build.
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

    /// An importer that retains the channel keeps the producer's `PboHandle`
    /// and GL buffer alive after the producing tensor drops; the buffer is
    /// deleted once, when the last reference goes.
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

        // The producing tensor goes away first.
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
        assert_eq!(ops.unmap_count(), 1, "so did the unmap");

        drop(child);
        assert_eq!(
            ops.delete_count(),
            1,
            "the GL buffer is deleted exactly once, when the last reference \
             to the channel goes"
        );
    }

    /// `into_client_parts` hands out a borrowed channel: a callee that retains
    /// before the parts drop keeps the GL buffer.
    #[test]
    fn client_parts_hand_over_a_channel_the_callee_retains() {
        let ops = MockPboOps::new(32);
        let tensor = PboTensor::<u8>::from_pbo(7, 32, &[32], None, ops.clone())
            .expect("PboTensor::from_pbo");
        let parts = tensor.into_client_parts().expect("into_client_parts");
        // Consuming the tensor has not dropped the channel.
        assert_eq!(
            ops.delete_count(),
            0,
            "into_client_parts must hand over a live channel, not a dead one"
        );
        assert_eq!(parts.buffer_id, 7);
        assert_eq!(parts.size, 32);
        assert_eq!(parts.shape, vec![32]);

        // SAFETY: the parts came from a live `PboTensor`, so they satisfy
        // `client_state_pbo_ops`'s contract.
        let reattached = unsafe { client_state_pbo_ops(parts.state, parts.map_fn, parts.unmap_fn) }
            .expect("client_state_pbo_ops");

        // The producer's last reference goes; the callee retained, so the GL
        // buffer remains.
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

    /// A sub-view is refused.
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
    /// `release` with `Error::InvalidArgument`.
    #[test]
    fn from_client_pbo_refuses_an_incomplete_channel() {
        let ops = MockPboOps::new(16);
        let tensor =
            PboTensor::<u8>::from_pbo(9, 16, &[16], None, ops).expect("PboTensor::from_pbo");
        let parts = tensor.into_client_parts().expect("into_client_parts");
        // A genuine `ctx`.
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

    /// `client_state_pbo_ops` refuses a NULL `ctx`, `retain` or `release` with
    /// `InvalidArgument`.
    #[test]
    fn a_client_state_missing_any_part_is_refused() {
        let ops = MockPboOps::new(16);
        let tensor =
            PboTensor::<u8>::from_pbo(9, 16, &[16], None, ops).expect("PboTensor::from_pbo");
        let parts = tensor.into_client_parts().expect("into_client_parts");
        // A genuine `ctx`.
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

    /// `set_plane_offset` moves a PBO's CPU-map window.
    #[test]
    fn set_plane_offset_moves_the_pbo_map_window() {
        use crate::{Tensor, TensorTrait};

        const PARENT: usize = 256;
        const OFFSET: usize = 64;
        const WINDOW: usize = 64;

        // Ramp the buffer so byte i holds i.
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

        // Re-wrap the whole buffer, as an import does; it lands at the parent's
        // origin.
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
             a no-op on PBO storage"
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

    /// A nested `subview` does not compound its offset.
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

    /// `set_format` clears the PBO's storage-internal window.
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

    /// Mock whose map and unmap both fail: with `PboDisconnected` when
    /// `disconnected`, otherwise with an unrelated error.
    struct FailingOps {
        disconnected: bool,
    }

    impl FailingOps {
        fn error(&self) -> Error {
            if self.disconnected {
                Error::PboDisconnected
            } else {
                Error::InvalidOperation("GL call failed".into())
            }
        }
    }

    // SAFETY: never produces a pointer at all.
    unsafe impl PboOps for FailingOps {
        fn map_buffer(&self, _buffer_id: u32, _size: usize) -> Result<PboMapping> {
            Err(self.error())
        }

        fn unmap_buffer(&self, _buffer_id: u32) -> Result<()> {
            Err(self.error())
        }

        fn delete_buffer(&self, _buffer_id: u32) {}
    }

    /// `ops` as an importer on the far side of the vtable sees it.
    fn import_through_the_vtable(ops: Arc<dyn PboOps>) -> Arc<dyn PboOps> {
        let producer = PboTensor::<u8>::from_pbo(4, 16, &[16], None, ops).expect("from_pbo");
        let vtable_ptr = producer.pbo_vtable() as *const PboOpsVtable as *const c_void;
        // SAFETY: `vtable_ptr` came from `pbo_vtable()` on a live tensor, and
        // the importer retains the channel before the producer drops.
        unsafe {
            let parts = read_pbo_vtable_parts(vtable_ptr).expect("read_pbo_vtable_parts");
            client_state_pbo_ops(parts.state, parts.map_fn, parts.unmap_fn)
                .expect("client_state_pbo_ops")
        }
    }

    /// A disconnected GL context crosses the vtable as `PboDisconnected`.
    #[test]
    fn a_disconnected_producer_is_reported_as_disconnected_across_the_vtable() {
        let imported = import_through_the_vtable(Arc::new(FailingOps { disconnected: true }));
        assert!(matches!(
            imported.map_buffer(4, 16),
            Err(Error::PboDisconnected)
        ));
        assert!(matches!(
            imported.unmap_buffer(4),
            Err(Error::PboDisconnected)
        ));
    }

    /// Any other producer failure crosses the vtable as a generic failure.
    #[test]
    fn other_producer_failures_cross_the_vtable_as_a_generic_errno() {
        let imported = import_through_the_vtable(Arc::new(FailingOps {
            disconnected: false,
        }));
        for result in [
            imported.map_buffer(4, 16).map(|_| ()),
            imported.unmap_buffer(4),
        ] {
            match result {
                Err(Error::NotImplemented(msg)) => assert!(msg.ends_with("errno 1"), "{msg}"),
                other => panic!("expected NotImplemented(.. errno 1), got {other:?}"),
            }
        }
    }

    #[test]
    fn debug_names_the_buffer() {
        let t = PboTensor::<u8>::from_pbo(77, 16, &[16], Some("dbg"), MockPboOps::new(16))
            .expect("from_pbo");
        let text = format!("{t:?}");
        assert!(text.contains("PboTensor"), "{text}");
        assert!(text.contains("buffer_id: 77"), "{text}");
        assert!(text.contains("size: 16"), "{text}");
        assert!(text.contains("\"dbg\""), "{text}");
    }
}
