// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! The dynamic backend: a thin newtype over the opaque C handle that calls
//! into `libedgefirst_tensor.so` instead of embedding this crate's own
//! implementation.
//!
//! Every method here is a caller of an already-`#[no_mangle]`-exported
//! `ef_tensor_*` function declared in `edgefirst-tensor-ffi` -- none of the
//! logic that produces a tensor's bytes, geometry, or metadata lives in this
//! file; it lives in `libedgefirst_tensor.so` (the `static`-backend build of
//! `edgefirst-tensor-capi`). See `docs/superpowers/plans/
//! PRIMITIVE-INVENTORY.md` for which methods are primitive (a direct
//! `ef_tensor_*` call, here) versus derived (expressible over the primitives,
//! in `derived.rs`, identical on both backends).
//!
//! Not `#[repr(transparent)]`: unlike the brief's original stub, this struct
//! also caches the shape as `Vec<usize>` (because `shape()` must return a
//! borrowed `&[usize]`, but `ef_tensor_shape` returns a borrowed `*const
//! u64` -- a different element width) and a locally-derived
//! [`crate::BufferIdentity`] (because there is no `ef_tensor_*` primitive
//! that exposes one). Neither extra field changes `into_raw`/`from_raw`'s
//! contract: they only ever read/write the handle field explicitly, never
//! `transmute` the whole struct.

use std::fmt;
use std::ptr::NonNull;

use edgefirst_tensor_ffi::EfTensor;

use crate::{
    BufferIdentity, Colorimetry, CpuAccess, DType, Error, IdentityKind, PixelFormat, PixelLayout,
    Region, Result, TensorMemory, TensorTrait, ViewOrigin,
};

/// The tensor handle, plus the facts this backend cannot re-derive from the
/// handle alone on every call (see the module docs).
pub struct TensorDyn {
    handle: NonNull<EfTensor>,
    shape_cache: Vec<usize>,
    identity: BufferIdentity,
    /// Lazily-fetched quantization, read back through the two-call
    /// `ef_tensor_quantization_{info,get}` idiom the first time
    /// [`Self::quantization`] is called on *this* Rust value, then served
    /// from cache. Exists because `quantization(&self) -> Option<&Quantization>`
    /// must hand back a borrow with a stable address for as long as `self`
    /// lives, and `Quantization` is variable-length -- there is nowhere
    /// else on `Tensor<T>` to put owned storage for it without adding a
    /// field there, which would break its `#[repr(transparent)]` layout
    /// over `TensorDyn` (see `dynamic_tensor.rs`'s module docs). `OnceLock`
    /// specifically (not a raw `&mut` cast, the F12 colorimetry race's
    /// shape) is what makes concurrent reads from multiple threads sharing
    /// one `&TensorDyn` sound: `get_or_init` serializes concurrent
    /// initializers itself, so no `&mut TensorDyn` is ever materialized
    /// while a `&TensorDyn` could be alive elsewhere. [`Self::set_quantization`]/
    /// [`Self::clear_quantization`] take `&mut self`, so Rust's own
    /// aliasing rules already guarantee no `&TensorDyn` (hence no
    /// in-flight `get_or_init`) can coexist with the write that replaces
    /// this field wholesale -- unlike colorimetry's cross-thread
    /// read/write race, this write path is Rust-borrow-checked, not
    /// merely documented. What this does **not** cover: a *different*
    /// `TensorDyn` Rust value wrapping the same underlying C handle (via a
    /// separate `retain` + `from_raw`) has its own, independent cache, so
    /// a `set_quantization` through one wrapper does not invalidate an
    /// already-cached read through another -- a documented staleness
    /// limitation, not a memory-safety one (each wrapper's cache is its
    /// own allocation).
    quantization_cache: std::sync::OnceLock<Option<crate::Quantization>>,
    /// If this handle is the combined tensor [`Tensor::from_planes`] built,
    /// an independent handle over the chroma plane's own allocation --
    /// `None` for every other tensor, including a genuinely combined-plane
    /// semi-planar image, which correctly has no separate chroma
    /// allocation. Exists because no `ef_tensor_*` primitive can recover a
    /// genuinely-multiplane handle's separate chroma fd after the fact:
    /// `ef_tensor_plane_at` derives every plane's geometry from the
    /// format's own plane table over ONE buffer and reports the SAME
    /// native handle for every plane index (see `vtable.rs::native_handle`
    /// in `tensor-capi`), so it cannot express "plane 1 lives in a
    /// different allocation." [`Tensor::from_planes`] therefore captures a
    /// `dup`'d handle onto the chroma plane's own fd *before* the
    /// consuming `ef_tensor_from_planes` call (see
    /// [`Self::shadow_multiplane_chroma`]) and stashes it here, the same
    /// "state the ABI cannot answer" reasoning `shape_cache`/`identity`
    /// already document. See task 17's report: before this field existed,
    /// `is_multiplane()`/`chroma()`/`chroma_mut()` unconditionally
    /// reported "no chroma" even for a tensor `from_planes` had just
    /// built, which downstream consumers (`edgefirst-image`'s CPU convert
    /// path) read as "read chroma from the combined buffer" -- wrong for a
    /// tensor that is actually two independent DMA-BUFs.
    pub(crate) multiplane_chroma: Option<Box<TensorDyn>>,
    /// This tensor's `PboTensor<T>` (type-erased), when it wraps a GL
    /// Pixel Buffer Object rather than data the real `ef_tensor_*` handle
    /// itself owns. `None` for every other tensor.
    ///
    /// Exists for the same "state the ABI cannot answer" reason
    /// `multiplane_chroma` does: `PboTensor<T>`'s own state (a GL
    /// `buffer_id`, an `Arc<dyn PboOps>` routing map/unmap to the owning
    /// process's own GL worker thread, and CPU-map bookkeeping) is
    /// in-process Rust state with no wire representation at all -- there is
    /// no `ef_tensor_*` primitive that could hand back an equivalent
    /// "something" here, the same class of gap task 17's report already
    /// closed for `multiplane_chroma` and left open for `cuda` below.
    /// `PboTensor<T>` itself is `static`/`dynamic`-agnostic already (see
    /// its own module doc in `pbo.rs`); only `static`'s `Tensor::as_pbo`/
    /// `from_pbo`, which store it inside `TensorStorage::Pbo`, were
    /// `static`-only. This handle still carries a real `ef_tensor_*`
    /// backing (see [`Self::handle`]) sized to match, purely so shape/
    /// dtype/format/stride metadata queries keep working the ordinary way;
    /// only CPU-mapping operations need to know to route through this
    /// field's `PboOps` instead (see [`Self::memory`]/`map_pin`). That
    /// backing is a REAL, full-sized host allocation, not a cheap
    /// placeholder -- task 18's review (F32) found and measured this: see
    /// `Tensor::from_pbo`'s own doc comment (`dynamic_tensor.rs`) and
    /// `tests/dynamic_primitives.rs`'s
    /// `from_pbo_metadata_handle_allocation_cost_is_the_full_pbo_byte_count`
    /// for the number and the follow-up primitive this should replace.
    ///
    /// `Box<dyn Any>` because `TensorDyn` itself carries no element type;
    /// the typed lens (`Tensor<T>::as_pbo`, `dynamic_tensor.rs`) downcasts
    /// back to `PboTensor<T>`, the same technique `lens.rs`'s `as_typed`
    /// already uses for the handle itself.
    pub(crate) pbo: Option<Box<dyn std::any::Any + Send + Sync>>,
    /// CUDA registration attached to this tensor, if any -- same "state the
    /// ABI cannot answer" reasoning as [`Self::pbo`] and `multiplane_chroma`
    /// above. `CudaHandle` (`crate::cuda`) is already `static`/`dynamic`-
    /// agnostic (see that module's own doc comment); only `static`'s
    /// `Tensor::cuda`/`cuda_map`/`set_cuda_handle` (`lib.rs`), which store
    /// it as a plain field of `Tensor<T>`, were `static`-only. Not
    /// type-erased via `Any` like `pbo`: `CudaHandle` itself carries no
    /// element type to erase.
    pub(crate) cuda: Option<Box<crate::cuda::CudaHandle>>,
}

// SAFETY: `TensorTrait<T>: Send + Sync` is a supertrait bound every consumer
// (e.g. `edgefirst-image`'s `gl/threaded.rs`, which wraps a `TensorDyn` for
// cross-thread GL work) relies on, and `NonNull<EfTensor>` is neither by
// default. The C ABI's handles are designed to cross threads: ownership is
// tracked by refcount (`ef_tensor_retain`/`ef_tensor_free`), not by Rust's
// aliasing rules, and every mutating entry point this backend calls
// (`ef_tensor_map`/`unmap`, `ef_tensor_set_colorimetry`, ...) is documented
// as safe to call from any thread holding a valid reference. `shape_cache`
// and `identity` are ordinary owned `Send + Sync` data; `quantization_cache`
// is `OnceLock<Option<Quantization>>`, `Sync` by construction whenever its
// contents are (`Quantization` is plain owned data, `Send + Sync`
// automatically) -- see that field's own doc comment for why concurrent
// access through it specifically is sound, not just asserted here. `pbo` is
// `Box<dyn Any + Send + Sync>` (the bound is part of the trait object type,
// checked by the compiler) wrapping a `PboTensor<T>`, itself `Send + Sync`
// by its own explicit `unsafe impl` in `pbo.rs`; `cuda` is `Box<CudaHandle>`,
// and `CudaHandle`'s own fields (`cuda.rs`) are process-global handles or
// routed through a `Send + Sync` GL-ops trait object, the same reasoning
// `static`'s own `Tensor<T>` (`lib.rs`) already relies on for these same two
// types as plain fields.
unsafe impl Send for TensorDyn {}
unsafe impl Sync for TensorDyn {}

impl Drop for TensorDyn {
    fn drop(&mut self) {
        // SAFETY: we hold one reference, taken at construction.
        unsafe { edgefirst_tensor_ffi::ef_tensor_free(self.handle.as_ptr()) }
    }
}

/// Same name, same meaning as the static backend's -- here it is the real
/// handle rather than a boxed Rust value.
pub type Raw = *mut EfTensor;

impl TensorDyn {
    /// Wrap an already-live handle, deriving the cached facts fresh.
    fn from_handle(handle: NonNull<EfTensor>) -> Self {
        let shape_cache = Self::query_shape(handle);
        let identity = Self::derive_identity(handle);
        TensorDyn {
            handle,
            shape_cache,
            identity,
            quantization_cache: std::sync::OnceLock::new(),
            multiplane_chroma: None,
            pbo: None,
            cuda: None,
        }
    }

    /// Derive this handle's [`BufferIdentity`].
    ///
    /// For a DMA-BUF-backed handle, uses the native fd's `(st_dev, st_ino)`
    /// -- survives `dup`, and is the same value for every independent
    /// import of the same underlying buffer, matching the static backend's
    /// own `IdentityKind::DmaBuf` strategy exactly (`dma.rs::
    /// identity_from_stat`) rather than the handle's own process-local
    /// address. Observable from the Rust side today via an `fstat` on the
    /// native fd [`Self::plane0`] already exposes, so it needs no new
    /// `ef_tensor_*` primitive (proposed by task 15's report, implemented
    /// here by task 17).
    ///
    /// **Why this matters beyond `aliases()` returning the right bool.**
    /// `edgefirst-image`'s GL import cache (`gl/cache.rs`) keys a cached
    /// EGLImage on `buffer_identity().id()` and is documented as safe to
    /// outlive the tensor that produced it *specifically because* a
    /// DMA-BUF/IOSurface-kind identity is an OS-level key: the kernel
    /// cannot recycle an inode onto a different buffer while any reference
    /// (including the cache's own retained import) is alive. The
    /// `HostPtr` fallback this replaces has no such guarantee -- it is a
    /// process address, and this library's own C-side allocator can (and,
    /// under allocation churn, will) reuse a just-freed `EfTensorImpl`'s
    /// address for the next handle it mints. A cache entry keyed on that
    /// reused address, for a request with matching geometry, would serve a
    /// *different* buffer's stale imported texture: wrong pixels, no
    /// error, no crash -- an ABA hazard the design a `IdentityKind` doc
    /// comment (`lib.rs`) explicitly says a process-local kind must not be
    /// exposed to a cache like this without exactly this kind of stable,
    /// unrecyclable-while-live key. Falls back to `HostPtr` (this handle's
    /// own address) for every other storage kind, or if the fd cannot be
    /// `fstat`-ed -- no `ef_tensor_*` primitive exposes a system-level
    /// identity key for those, so a process-local one is the best answer
    /// available, same as the static backend's own `Mem`/`Pbo` tensors.
    fn derive_identity(handle: NonNull<EfTensor>) -> BufferIdentity {
        #[cfg(unix)]
        {
            // SAFETY: `handle` is live for the duration of this call.
            let code = unsafe { edgefirst_tensor_ffi::ef_tensor_storage_kind(handle.as_ptr()) };
            if TensorMemory::from_code(code) == Some(TensorMemory::DmaBuf) {
                let mut plane = edgefirst_tensor_ffi::EfTensorPlane::default();
                // SAFETY: `handle` is live; `plane` is a valid local out-param.
                let rc = unsafe {
                    edgefirst_tensor_ffi::ef_tensor_plane_at(handle.as_ptr(), 0, &mut plane)
                };
                if rc == 0 && plane.handle >= 0 {
                    // SAFETY: `plane.handle` is a valid fd owned by this
                    // handle for at least the duration of this borrow.
                    let fd = unsafe { std::os::fd::BorrowedFd::borrow_raw(plane.handle as i32) };
                    if let Ok(stat) = nix::sys::stat::fstat(fd) {
                        return identity_from_stat(&stat);
                    }
                }
            }
        }
        BufferIdentity::derived(IdentityKind::HostPtr, handle.as_ptr() as u64)
    }

    /// Independently `dup` this handle's fd into a brand-new, standalone
    /// `TensorDyn` -- not a `retain` (which shares the same handle and
    /// refcount; `ef_tensor_from_planes` explicitly rejects an outstanding
    /// one on either input, so a retained keepalive is not an option here).
    /// [`crate::Tensor::from_planes`] (`dynamic_tensor.rs`) calls this on
    /// the chroma plane *before* consuming it into the combined handle, so
    /// [`Self::multiplane_chroma`] has something real to serve afterward.
    /// See that field's doc comment for why this is necessary at all.
    #[cfg(unix)]
    pub(crate) fn shadow_multiplane_chroma(&self) -> Result<TensorDyn> {
        let fd = self.clone_fd()?;
        Self::from_fd(fd, self.shape(), self.dtype(), None)
    }

    /// Read the handle's current shape via `ef_tensor_ndim`/`ef_tensor_shape`.
    fn query_shape(handle: NonNull<EfTensor>) -> Vec<usize> {
        // SAFETY: `handle` is a live handle for the duration of this call.
        unsafe {
            let ndim = edgefirst_tensor_ffi::ef_tensor_ndim(handle.as_ptr());
            let ptr = edgefirst_tensor_ffi::ef_tensor_shape(handle.as_ptr());
            if ptr.is_null() || ndim == 0 {
                return Vec::new();
            }
            std::slice::from_raw_parts(ptr, ndim as usize)
                .iter()
                .map(|&d| d as usize)
                .collect()
        }
    }

    /// Give up ownership, yielding the handle itself. No allocation beyond
    /// the cache this instance already carried, which is simply dropped: it
    /// is derived state, not something `from_raw` needs handed back to it.
    pub fn into_raw(self) -> Raw {
        let p = self.handle.as_ptr();
        std::mem::forget(self); // the caller owns the reference now
        p
    }

    /// # Safety
    /// `p` must be a live handle carrying a reference the caller owns.
    pub unsafe fn from_raw(p: Raw) -> TensorDyn {
        Self::from_handle(unsafe { NonNull::new_unchecked(p) })
    }

    /// # Safety
    /// `p` must be a live handle from an EdgeFirst library.
    pub unsafe fn with_raw<R>(p: Raw, f: impl FnOnce(&mut TensorDyn) -> R) -> R {
        let mut t =
            std::mem::ManuallyDrop::new(Self::from_handle(unsafe { NonNull::new_unchecked(p) }));
        f(&mut t)
    }

    // --- The 9 primitives `TensorTrait<T>`'s method set requires a
    // per-backend implementation for (`lib.rs`'s trait definition; see
    // `docs/superpowers/plans/PRIMITIVE-INVENTORY.md`). Each is a direct
    // `ef_tensor_*` call, same as the static backend's own inherent methods
    // of the same names dispatch to `Tensor<T>`'s.

    /// Create a type-erased tensor with the given shape and element type.
    ///
    /// Drives the builder (`ef_tensor_builder_{new,dtype,shape,storage,
    /// alloc}`) rather than the shortcut `ef_tensor_new`, because that
    /// shortcut only ever allocates host memory -- the builder is what lets
    /// `memory` name a specific backing, matching the static backend's own
    /// contract. `name` is accepted for signature parity but has no effect:
    /// no `ef_tensor_*` primitive carries a tensor's name across the ABI.
    pub fn new(
        shape: &[usize],
        dtype: DType,
        memory: Option<TensorMemory>,
        _name: Option<&str>,
    ) -> Result<Self> {
        // SAFETY: every call below either takes a value or a pointer to data
        // this function owns for the call's duration.
        unsafe {
            let b = edgefirst_tensor_ffi::ef_tensor_builder_new();
            if b.is_null() {
                return Err(Error::NotImplemented(
                    "ef_tensor_builder_new: allocation failed".into(),
                ));
            }
            let dims: Vec<u64> = shape.iter().map(|&d| d as u64).collect();
            edgefirst_tensor_ffi::ef_tensor_builder_dtype(b, dtype.code());
            edgefirst_tensor_ffi::ef_tensor_builder_shape(b, dims.as_ptr(), dims.len() as u32);
            if let Some(mem) = memory {
                edgefirst_tensor_ffi::ef_tensor_builder_storage(b, mem.code());
            }
            let handle = edgefirst_tensor_ffi::ef_tensor_builder_alloc(b);
            let result = match NonNull::new(handle) {
                Some(h) => Ok(Self::from_handle(h)),
                None => Err(builder_error(
                    edgefirst_tensor_ffi::ef_tensor_builder_error(b),
                )),
            };
            edgefirst_tensor_ffi::ef_tensor_builder_free(b);
            result
        }
    }

    /// Import an existing buffer as a type-erased tensor, taking ownership
    /// of its file descriptor. No bytes are copied.
    ///
    /// Drives the builder's adopt path (`ef_tensor_builder_{new,dtype,shape,
    /// add_plane,wrap}`) -- `ef_tensor_builder_wrap` is exactly this
    /// primitive: it adopts the first plane's handle and dispatches to the
    /// static backend's own `TensorDyn::from_fd` internally (backend
    /// detection from the fd's filesystem magic included), inside
    /// `libedgefirst_tensor.so`.
    #[cfg(unix)]
    pub fn from_fd(
        fd: std::os::fd::OwnedFd,
        shape: &[usize],
        dtype: DType,
        _name: Option<&str>,
    ) -> Result<Self> {
        use std::os::fd::{FromRawFd, IntoRawFd};
        // SAFETY: every call below either takes a value or a pointer to data
        // this function owns for the call's duration; `raw_fd` is reclaimed
        // into an `OwnedFd` (and so closed) on every path that does not hand
        // it to `wrap`.
        unsafe {
            let b = edgefirst_tensor_ffi::ef_tensor_builder_new();
            if b.is_null() {
                return Err(Error::NotImplemented(
                    "ef_tensor_builder_new: allocation failed".into(),
                ));
            }
            let dims: Vec<u64> = shape.iter().map(|&d| d as u64).collect();
            edgefirst_tensor_ffi::ef_tensor_builder_dtype(b, dtype.code());
            edgefirst_tensor_ffi::ef_tensor_builder_shape(b, dims.as_ptr(), dims.len() as u32);
            let raw_fd = fd.into_raw_fd();
            edgefirst_tensor_ffi::ef_tensor_builder_add_plane(b, raw_fd as i64, 0, 0, 0, 0, 0);
            let handle = edgefirst_tensor_ffi::ef_tensor_builder_wrap(b);
            let result = match NonNull::new(handle) {
                Some(h) => Ok(Self::from_handle(h)),
                None => {
                    // `wrap` failed before adopting the fd -- reclaim it so
                    // it is closed instead of leaked.
                    drop(std::os::fd::OwnedFd::from_raw_fd(raw_fd));
                    let errno = edgefirst_tensor_ffi::ef_tensor_builder_error(b);
                    Err(Error::NotImplemented(format!(
                        "ef_tensor_builder_wrap failed: errno {errno}"
                    )))
                }
            };
            edgefirst_tensor_ffi::ef_tensor_builder_free(b);
            result
        }
    }

    /// Return the tensor shape.
    pub fn shape(&self) -> &[usize] {
        &self.shape_cache
    }

    /// Reshape this tensor. Total element count must remain the same.
    ///
    /// No `ef_tensor_reshape` primitive exists yet -- `PRIMITIVE-INVENTORY.md`
    /// found no call site across any of the four `-capi` siblings that
    /// reaches this on `TensorDyn` under the dynamic backend, so adding one
    /// was out of this task's scope. Rather than silently updating only the
    /// local `shape_cache` (which `ef_tensor_shape` itself would then
    /// contradict for any other consumer of the same handle), this refuses
    /// with a clear error until a real primitive lands.
    pub fn reshape(&mut self, shape: &[usize]) -> Result<()> {
        self.reshape_impl(shape, false)
    }

    /// Set the logical shape to any shape whose bytes fit the allocation,
    /// without [`reshape`](Self::reshape)'s equal-count constraint --
    /// the pool-reuse primitive. Drives `ef_tensor_set_logical_shape`.
    ///
    /// Not a convenience over `reshape`: `TensorTrait::set_logical_shape`'s
    /// *default* body is `self.reshape(shape)`, which is precisely the bug
    /// task P2e fixed. Without a real implementation here the dynamic
    /// backend would inherit that default and apply the strict rule to a
    /// pool tensor, silently refusing the reconfigure the method exists for.
    pub fn set_logical_shape(&mut self, shape: &[usize]) -> Result<()> {
        self.reshape_impl(shape, true)
    }

    /// The shared body of [`reshape`](Self::reshape) and
    /// [`set_logical_shape`](Self::set_logical_shape) -- one place to
    /// marshal the dims and refresh the cache, so the two cannot drift into
    /// leaving different state behind.
    fn reshape_impl(&mut self, shape: &[usize], by_capacity: bool) -> Result<()> {
        let dims: Vec<u64> = shape.iter().map(|&d| d as u64).collect();
        // SAFETY: `self.handle` is live; `dims` is a live local for the call.
        let rc = unsafe {
            if by_capacity {
                edgefirst_tensor_ffi::ef_tensor_set_logical_shape(
                    self.handle.as_ptr(),
                    dims.as_ptr(),
                    dims.len() as u32,
                )
            } else {
                edgefirst_tensor_ffi::ef_tensor_reshape(
                    self.handle.as_ptr(),
                    dims.as_ptr(),
                    dims.len() as u32,
                )
            }
        };
        if rc != 0 {
            // The C side already prefixes its message with the operation
            // name, so this does not add a second one.
            let msg = ffi_last_error();
            // ERANGE is "this shape does not fit"; anything else is a
            // malformed argument. The static backend distinguishes further
            // and the ABI has no room for it, so several of its variants
            // collapse into these two. The message survives every one of
            // them, which is what a caller actually sees (`python-common`
            // renders `Display` into a `PyErr`), and nothing in this
            // workspace matches on the variants. Enumerated so the next
            // reader does not have to re-derive it:
            //
            //   static                              dynamic
            //   ShapeMismatch (count mismatch)      ShapeMismatch
            //   InsufficientCapacity {needed, cap}  ShapeMismatch (numbers
            //                                       do not cross the ABI)
            //   InvalidOperation ("cannot reshape   InvalidShape
            //     a multiplane tensor")
            //   InvalidSize(0) (empty shape)        InvalidShape -- the C
            //                                       entry rejects ndim == 0
            //                                       before the backend sees it
            return Err(if rc == libc::ERANGE {
                Error::ShapeMismatch(msg)
            } else {
                Error::InvalidShape(msg)
            });
        }
        // `shape()` serves the cached vector, not the handle, so it would
        // keep reporting the OLD geometry the moment after this succeeded.
        self.shape_cache = Self::query_shape(self.handle);
        // ...and a PBO-backed tensor carries geometry in a THIRD place: the
        // wrapped `PboTensor`, which `as_pbo()` hands to `edgefirst-image`.
        // The handle already validated this shape, so the same call on the
        // PBO cannot refuse for a reason the handle accepted -- but its
        // result is propagated rather than dropped, because a PBO whose
        // buffer is smaller than the companion allocation would be a real
        // disagreement worth surfacing rather than swallowing.
        self.sync_pbo_shape(shape, by_capacity)?;
        Ok(())
    }

    /// Apply a geometry change to the wrapped `PboTensor`, if there is one.
    ///
    /// Mirrors the operation rather than always using one of them: `reshape`
    /// resets the PBO's `view_offset` to 0 and `set_logical_shape` does not,
    /// which is the static backend's behaviour and the difference a caller
    /// of `as_pbo()` would see.
    fn sync_pbo_shape(&mut self, shape: &[usize], by_capacity: bool) -> Result<()> {
        let r = if by_capacity {
            with_pbo_mut!(self, set_logical_shape, shape)
        } else {
            with_pbo_mut!(self, reshape, shape)
        };
        match r {
            None => Ok(()), // not PBO-backed; nothing to keep in step
            Some(res) => res,
        }
    }

    /// Return the tensor name.
    ///
    /// Always empty: no `ef_tensor_*` primitive carries a tensor's name
    /// across the ABI (confirmed in `PRIMITIVE-INVENTORY.md` and by task 7's
    /// investigation of this same gap).
    pub fn name(&self) -> String {
        String::new()
    }

    /// Return the memory allocation type.
    ///
    /// `Pbo` when [`Self::pbo`] is set, regardless of what the backing
    /// `ef_tensor_*` handle's own storage kind reports: that handle exists
    /// only to carry shape/dtype/format metadata (see `pbo`'s own doc
    /// comment), and is allocated as ordinary host memory since no
    /// `ef_tensor_*` primitive can mint a genuinely GL-buffer-backed handle
    /// -- `memory()` must still report the tensor's real backing to match
    /// `static`'s own `PboTensor::memory() -> TensorMemory::Pbo`.
    pub fn memory(&self) -> TensorMemory {
        if self.pbo.is_some() {
            return TensorMemory::Pbo;
        }
        // SAFETY: `self.handle` is a live handle for as long as `self` exists.
        let code = unsafe { edgefirst_tensor_ffi::ef_tensor_storage_kind(self.handle.as_ptr()) };
        TensorMemory::from_code(code).unwrap_or(TensorMemory::Mem)
    }

    /// Clone the file descriptor backing this tensor, for any storage kind
    /// that has one.
    ///
    /// Drives `ef_tensor_clone_fd`. This used to read plane 0's native
    /// handle (`ef_tensor_plane_at`) and `dup` it directly, which needed no
    /// new primitive -- and was wrong for SHM. A plane's native handle is a
    /// dma-buf fd on Linux and an IOSurface id on Apple, and `-1` for every
    /// other backing, so deriving "clone this tensor's fd" from it refused
    /// SHM-backed tensors, which do have a real fd and which the static
    /// backend clones without complaint (`TensorStorage::Shm(t) =>
    /// t.clone_fd()`). Asking the library, which owns the storage and knows
    /// which kinds have an fd, is what the split is for -- the same rule
    /// every other method here follows.
    #[cfg(unix)]
    pub fn clone_fd(&self) -> Result<std::os::fd::OwnedFd> {
        use std::os::fd::FromRawFd;
        // SAFETY: `self.handle` is live for as long as `self` exists.
        let rc = unsafe { edgefirst_tensor_ffi::ef_tensor_clone_fd(self.handle.as_ptr()) };
        if rc < 0 {
            let msg = ffi_last_error();
            return Err(if -rc == libc::ENOTSUP {
                Error::NotImplemented(msg)
            } else {
                // `Error::other`, not `from_raw_os_error`: the latter
                // discarded `msg` on exactly the path where it is most
                // useful -- a `dup` that failed for a reason worth reading,
                // reduced to a bare errno. The errno is kept in the text.
                //
                // Variant note: the static backend returns `NixError` here
                // (it calls `nix::unistd::dup` directly), and this returns
                // `IoError`. The ABI carries an errno, not a Rust variant,
                // and no caller in this workspace matches on either -- both
                // render the same way through `Display`. Stated rather than
                // left for the next reader to re-derive.
                Error::IoError(std::io::Error::other(format!(
                    "clone_fd: {msg} (errno {})",
                    -rc
                )))
            });
        }
        // SAFETY: `ef_tensor_clone_fd` returned a freshly-`dup`'d fd whose
        // ownership it handed to this caller.
        Ok(unsafe { std::os::fd::OwnedFd::from_raw_fd(rc) })
    }

    /// Return the [`BufferIdentity`] of the underlying allocation.
    ///
    /// See [`from_handle`](Self::from_handle): derived once at construction
    /// from the handle's own address, because no `ef_tensor_*` primitive
    /// exposes a system-level identity key (a dma-buf `(st_dev, st_ino)`,
    /// say) to derive one from instead.
    pub fn buffer_identity(&self) -> &BufferIdentity {
        &self.identity
    }

    /// The shared core of [`map_bytes`](Self::map_bytes) and
    /// [`pin_host`](Self::pin_host): `ef_tensor_map` plus an
    /// `ef_tensor_retain`'d [`MapKeepalive`] so the resulting pin is
    /// genuinely `'static` (shares ownership through the keepalive, never a
    /// borrow of `self` -- same contract the static backend's `map_bytes`
    /// documents).
    ///
    /// `pub(crate)`, not `pub`: `dynamic_tensor.rs`'s `Tensor<T>::map_with`
    /// also calls this directly (to build a typed `HostView<T>` instead of
    /// the byte-level one [`map_bytes`](Self::map_bytes) returns), which is
    /// why this is not `pub(self)` -- it is shared plumbing within the
    /// crate, not a second public byte-mapping entry point.
    pub(crate) fn map_pin(&self, access: CpuAccess) -> Result<crate::pin::HostPin<'static>> {
        // A PBO-backed handle's real (GPU-resident) bytes live in the GL
        // buffer `pbo` addresses, not in this companion `ef_tensor_*`
        // handle's own host allocation (real and full-sized, not
        // metadata-only -- see `from_pbo`'s doc comment, `dynamic_tensor.rs`,
        // for why and what it costs) -- mapping the latter would silently
        // hand back unrelated host bytes, not the PBO's actual data. No real
        // caller in this
        // workspace reaches this: `edgefirst-image` always downcasts via
        // `Tensor::as_pbo()` first and calls `PboTensor::map`/`map_with`
        // directly (confirmed by reading every non-test call site). Failing
        // loudly here, rather than silently mapping the wrong bytes, is the
        // same rule `TensorDyn::reshape`'s own honest-`Err` above follows.
        if self.pbo.is_some() {
            return self.map_pin_pbo(access);
        }
        let code = match access {
            CpuAccess::None => {
                return Err(Error::InvalidArgument(
                    "map: CpuAccess::None is not a mappable direction".into(),
                ))
            }
            CpuAccess::Read => 1,
            CpuAccess::Write => 2,
            CpuAccess::ReadWrite => 3,
        };
        let mut view = edgefirst_tensor_ffi::EfTensorView {
            ptr: std::ptr::null_mut(),
            len: 0,
        };
        // SAFETY: `self.handle` is live; `view` is a valid local out-param.
        let rc =
            unsafe { edgefirst_tensor_ffi::ef_tensor_map(self.handle.as_ptr(), code, &mut view) };
        if rc != 0 {
            return Err(Error::InvalidOperation(format!(
                "ef_tensor_map failed: errno {rc}"
            )));
        }
        // Retain so the pin can genuinely outlive `self` -- released by
        // `MapKeepalive::drop`.
        // SAFETY: `self.handle` is live.
        let retain_rc = unsafe { edgefirst_tensor_ffi::ef_tensor_retain(self.handle.as_ptr()) };
        if retain_rc != 0 {
            // SAFETY: undoes the successful map above.
            unsafe { edgefirst_tensor_ffi::ef_tensor_unmap(self.handle.as_ptr()) };
            return Err(Error::NotImplemented(format!(
                "ef_tensor_retain failed: errno {retain_rc}"
            )));
        }
        let keepalive: std::sync::Arc<dyn Send + Sync> =
            std::sync::Arc::new(MapKeepalive(self.handle.as_ptr()));
        Ok(crate::pin::HostPin::new(keepalive, view.ptr, view.len))
    }

    /// [`map_pin`](Self::map_pin) for a PBO-backed tensor: route the mapping
    /// through the wrapped `PboTensor` instead of the companion handle.
    ///
    /// This used to refuse outright ("use `Tensor::as_pbo().map()` instead of
    /// mapping the type-erased handle directly"). That was defensible as far
    /// as it went -- mapping the companion handle really would hand back
    /// unrelated host bytes, since a PBO's real data lives in the GL buffer
    /// (see [`Self::pbo`]) -- but it made a refusal out of something the
    /// static backend simply does: `TensorStorage::Pbo(t) => t.map_with(..)`.
    /// Python's `normalize_to_numpy()` maps whatever `convert()` returned,
    /// and on a GL machine that is a PBO, so every GPU conversion result
    /// became unreadable under `dynamic`.
    ///
    /// The guard `PboTensor::map_with` returns *is* the keepalive: dropping
    /// it runs `glUnmapBuffer` through the owning process's GL worker. So it
    /// is moved into the pin whole rather than having its pointer copied out
    /// and its lifetime managed separately -- the address stays valid for
    /// exactly as long as the pin does, which is the contract `HostPin`
    /// exists to express.
    fn map_pin_pbo(&self, access: CpuAccess) -> Result<crate::pin::HostPin<'static>> {
        // `map_with` needs the concrete `T` to reach `PboTensor<T>`, so this
        // is the same dtype dispatch `with_pbo!` performs -- written out
        // here rather than reusing that macro because this one takes an
        // argument and normalises each arm's `HostView<T>` to bytes.
        macro_rules! map_arm {
            ($any:expr, $t:ty) => {
                $any.downcast_ref::<crate::PboTensor<$t>>().map(|p| {
                    <crate::PboTensor<$t> as crate::TensorTrait<$t>>::map_with(p, access)
                        .map(|v| v.into_bytes())
                })
            };
        }
        let any = self
            .pbo
            .as_ref()
            .ok_or_else(|| Error::NotImplemented("map: not a PBO-backed tensor".into()))?;
        // Each instantiation in turn, not the one `dtype()` names -- see
        // `with_pbo!` for why those two can disagree. Here `T` does affect
        // the result (the view's element count), which is exactly why the
        // *stored* type is the right one to use: it is the type the buffer
        // was created with.
        let mapped = None
            .or_else(|| map_arm!(any, u8))
            .or_else(|| map_arm!(any, i8))
            .or_else(|| map_arm!(any, u16))
            .or_else(|| map_arm!(any, i16))
            .or_else(|| map_arm!(any, half::f16))
            .or_else(|| map_arm!(any, u32))
            .or_else(|| map_arm!(any, i32))
            .or_else(|| map_arm!(any, f32))
            .or_else(|| map_arm!(any, u64))
            .or_else(|| map_arm!(any, i64))
            .or_else(|| map_arm!(any, f64));
        let mut view = mapped
            .ok_or_else(|| {
                Error::NotImplemented(
                    "map: the wrapped PboTensor's element type does not match this handle's dtype"
                        .into(),
                )
            })?
            .map_err(|e| Error::InvalidOperation(format!("map: PBO map failed: {e}")))?;
        // Same rule `ef_tensor_map` follows: only a writable map may take
        // `as_mut_slice` (the guard debug-asserts writability), and a read
        // map still yields a `*mut u8` because `HostPin` has one pointer for
        // both directions.
        let (ptr, len) = if access.writes() {
            let s = crate::TensorMapTrait::as_mut_slice(&mut view);
            (s.as_mut_ptr(), s.len())
        } else {
            let s = crate::TensorMapTrait::as_slice(&view);
            (s.as_ptr() as *mut u8, s.len())
        };
        // The pointer addresses the GL mapping, not `view` itself, so it
        // stays valid across this move.
        let keepalive: std::sync::Arc<dyn Send + Sync> = std::sync::Arc::new(view);
        Ok(crate::pin::HostPin::new(keepalive, ptr, len))
    }

    /// Map this tensor's whole extent for CPU access, type-erased to raw
    /// bytes. Returns a `'static` view, same contract as the static
    /// backend's own `map_bytes`.
    pub fn map_bytes(&self, access: CpuAccess) -> Result<crate::view::HostView<'static, u8>> {
        let pin = self.map_pin(access)?;
        let len = pin.len();
        Ok(crate::view::HostView::new(pin, vec![len], None, access))
    }

    /// Pin a stable host address for this tensor's data. Returns a
    /// `'static` pin, same contract as the static backend's own `pin_host`.
    ///
    /// No dedicated `ef_tensor_pin_host` primitive exists; `ef_tensor_map`
    /// plus a retained reference (the same primitive `map_bytes` uses) is
    /// the only mapping operation the ABI exposes, so this shares
    /// [`map_pin`](Self::map_pin) rather than duplicating it.
    pub fn pin_host(&self, access: CpuAccess) -> Result<crate::pin::HostPin<'static>> {
        self.map_pin(access)
    }

    /// This tensor's plane-0 geometry (`ef_tensor_plane_at`, already
    /// exported), or `None` for an invalid handle or one with no planes.
    /// Shared by [`clone_fd`](Self::clone_fd), [`aliases`](Self::aliases),
    /// and [`effective_row_stride`](Self::effective_row_stride).
    ///
    /// `pub`, not `pub(crate)`: each `-capi` leaf's own transition vtable
    /// (`vtable.rs`, scheduled for removal once G4 closes -- see the
    /// single-tensor-home plan's task 10) mints tensors under its own
    /// private `EfTensorImpl` wrapper and needs exactly this to answer its
    /// own `ef_tensor_plane_at`-equivalent dispatch (the formatless-tensor
    /// fallback geometry, and the shareable native handle for any plane
    /// index -- every plane of one tensor shares plane 0's handle). Before
    /// task 9 switched the leaves onto this backend, those call sites used
    /// `TensorDyn::capacity_bytes()`/`dmabuf()`/`iosurface_id()`, which
    /// only exist on the `static` backend this crate no longer embeds in
    /// those libraries; delegating to the already-correct primitive here
    /// (rather than re-deriving capacity/native-handle logic on the
    /// consumer side) is the same rule task 15's report states for every
    /// other dynamic-backend method.
    pub fn plane0(&self) -> Option<edgefirst_tensor_ffi::EfTensorPlane> {
        let mut plane = edgefirst_tensor_ffi::EfTensorPlane::default();
        // SAFETY: `self.handle` is live; `plane` is a valid local out-param.
        let rc = unsafe {
            edgefirst_tensor_ffi::ef_tensor_plane_at(self.handle.as_ptr(), 0, &mut plane)
        };
        (rc == 0).then_some(plane)
    }

    /// Effective row stride: the padded byte pitch a stride-aware CPU
    /// reader must honor, sourced from plane 0's stride -- which is exactly
    /// what the static backend's own `effective_row_stride()` computes
    /// (`vtable.rs`'s `ef_tensor_plane_at` implementation reads that same
    /// value on the producing side).
    pub fn effective_row_stride(&self) -> Option<usize> {
        self.plane0().map(|p| p.stride as usize)
    }

    /// Image width in pixels (`None` if not an image tensor). Pure function
    /// of `shape()` + `format()`, mirroring `Tensor::width`.
    pub fn width(&self) -> Option<usize> {
        let fmt = self.format()?;
        let shape = self.shape();
        match fmt.layout() {
            PixelLayout::Packed => shape.get(1).copied(),
            PixelLayout::Planar => shape.get(2).copied(),
            PixelLayout::SemiPlanar => shape.get(1).copied(),
        }
    }

    /// Image height in pixels (`None` if not an image tensor). Pure
    /// function of `shape()` + `format()`, mirroring `Tensor::height`.
    ///
    /// Unlike the static backend, this backend has no way to tell whether a
    /// semi-planar tensor was assembled from separate luma/chroma
    /// allocations (`Tensor::is_multiplane`, which reads a private field no
    /// `ef_tensor_*` primitive exposes) -- so it always takes the
    /// combined-plane branch. `PRIMITIVE-INVENTORY.md` found no call site
    /// that reaches a multiplane `TensorDyn` under the dynamic backend, so
    /// this is a documented gap rather than a live break.
    pub fn height(&self) -> Option<usize> {
        let fmt = self.format()?;
        let shape = self.shape();
        match fmt.layout() {
            PixelLayout::Packed => shape.first().copied(),
            PixelLayout::Planar => shape.get(1).copied(),
            PixelLayout::SemiPlanar => match fmt {
                PixelFormat::Nv12 => shape.first().map(|h| h * 2 / 3),
                PixelFormat::Nv16 => shape.first().map(|h| h / 2),
                PixelFormat::Nv24 => shape.first().map(|h| h / 3),
                _ => None,
            },
        }
    }

    /// True if `self` and `other` reference the same underlying buffer.
    /// Same matching rules as the static backend's `aliases`: identity
    /// equality, then (Linux DMA tensors only) a same-fd-number compare
    /// via plane 0's native handle.
    pub fn aliases(&self, other: &Self) -> bool {
        if self.buffer_identity().id() == other.buffer_identity().id() {
            return true;
        }
        if self.memory() != other.memory() {
            return false;
        }
        if self.memory() == TensorMemory::DmaBuf {
            if let (Some(a), Some(b)) = (self.plane0(), other.plane0()) {
                if a.handle >= 0 && a.handle == b.handle {
                    return true;
                }
            }
        }
        false
    }

    // --- The three primitives `PRIMITIVE-INVENTORY.md` found genuinely new:
    // `colorimetry`, `set_colorimetry`, `view_origin`.

    /// Colorimetry metadata (`None` = undefined; never auto-filled).
    pub fn colorimetry(&self) -> Option<Colorimetry> {
        // SAFETY: `self.handle` is live.
        let packed = unsafe { edgefirst_tensor_ffi::ef_tensor_colorimetry(self.handle.as_ptr()) };
        if packed == 0 {
            None
        } else {
            Some(Colorimetry::unpack(packed))
        }
    }

    /// Attach/clear colorimetry metadata.
    pub fn set_colorimetry(&mut self, c: Option<Colorimetry>) {
        let packed = c.map(|c| c.pack()).unwrap_or(0);
        // SAFETY: `self.handle` is live and own-mint (every dynamic-backend
        // handle is minted by this same `libedgefirst_tensor.so`).
        unsafe { edgefirst_tensor_ffi::ef_tensor_set_colorimetry(self.handle.as_ptr(), packed) };
    }

    /// Builder-style [`Self::set_colorimetry`]. Same signature as the
    /// static backend's, so a shared call site needs no backend branch.
    pub fn with_colorimetry(mut self, c: Colorimetry) -> Self {
        self.set_colorimetry(Some(c));
        self
    }

    /// The parent-image snapshot if this tensor is a `view`/`batch`
    /// sub-region; `None` for a whole tensor. See [`ViewOrigin`].
    pub fn view_origin(&self) -> Option<ViewOrigin> {
        let mut out = edgefirst_tensor_ffi::EfViewOrigin::default();
        // SAFETY: `self.handle` is live; `out` is a valid local out-param.
        let rc =
            unsafe { edgefirst_tensor_ffi::ef_tensor_view_origin(self.handle.as_ptr(), &mut out) };
        if rc != 0 || out.has_origin == 0 {
            return None;
        }
        Some(ViewOrigin {
            parent_width: out.parent_width as usize,
            parent_height: out.parent_height as usize,
            parent_row_stride: out.parent_row_stride as usize,
            x: out.x as usize,
            y: out.y as usize,
        })
    }

    /// Return the runtime element type discriminant. Already-exported
    /// (`ef_tensor_dtype`); needed by `derived.rs`'s pure functions of
    /// `dtype()`/`shape()`/`format()`.
    pub fn dtype(&self) -> DType {
        // SAFETY: `self.handle` is live.
        let code = unsafe { edgefirst_tensor_ffi::ef_tensor_dtype(self.handle.as_ptr()) };
        DType::from_code(code).unwrap_or(DType::U8)
    }

    /// Return the pixel format (`None` if not an image tensor).
    /// Already-exported (`ef_tensor_format`).
    pub fn format(&self) -> Option<PixelFormat> {
        // SAFETY: `self.handle` is live; the returned pointer is borrowed
        // for as long as the handle lives, which outlives this call.
        let ptr = unsafe { edgefirst_tensor_ffi::ef_tensor_format(self.handle.as_ptr()) };
        if ptr.is_null() {
            return None;
        }
        // SAFETY: `ef_tensor_format` returns a NUL-terminated C string.
        let s = unsafe { std::ffi::CStr::from_ptr(ptr) }.to_str().ok()?;
        if s.is_empty() {
            None
        } else {
            PixelFormat::from_str_code(s)
        }
    }

    /// Allocate an image tensor with the given geometry, memory backing,
    /// and CPU access declaration -- dispatching on `dtype`, same as
    /// `static`'s `TensorDyn::image` (`static_backend.rs`). Drives
    /// `ef_tensor_image_alloc`, a thin wrapper the producer side
    /// (`tensor-capi`, always the `static` backend) resolves by calling the
    /// real `static::TensorDyn::image` -- the platform image geometry lives
    /// there, not here. Real callers under `dynamic`: `edgefirst-image`'s
    /// `cpu/mod.rs`.
    pub fn image(
        width: usize,
        height: usize,
        format: PixelFormat,
        dtype: DType,
        memory: Option<TensorMemory>,
        access: CpuAccess,
    ) -> Result<Self> {
        let c_format = format_cstring(format)?;
        let (has_memory, memory_code) = memory_code(memory);
        // SAFETY: `c_format` is a valid NUL-terminated string for the
        // call's duration.
        let handle = unsafe {
            edgefirst_tensor_ffi::ef_tensor_image_alloc(
                width,
                height,
                c_format.as_ptr(),
                dtype.code(),
                has_memory,
                memory_code,
                access_code(access),
            )
        };
        match NonNull::new(handle) {
            Some(h) => Ok(Self::from_handle(h)),
            // `NotImplemented` was a guess: an allocation that failed for
            // want of memory, or a format the platform declined, is not
            // "not implemented". `ffi_error` takes the producing side's own
            // class, and falls back to the old variant only when the
            // library recorded none.
            None => Err(ffi_error(Error::NotImplemented)),
        }
    }

    /// See [`Self::image`]: same primitive family, full-featured request
    /// form. Drives `ef_tensor_image_desc_alloc`, which resolves the
    /// request (including any compression negotiation) through the
    /// producer side's real `static::TensorDyn::image_desc`.
    pub fn image_desc(desc: &crate::ImageDesc) -> Result<Self> {
        let c_format = format_cstring(desc.format())?;
        // SAFETY: every call below either takes a value or a pointer to
        // data this function owns for the call's duration.
        unsafe {
            let d = edgefirst_tensor_ffi::ef_tensor_image_desc_new(
                desc.width(),
                desc.height(),
                c_format.as_ptr(),
                desc.dtype().code(),
            );
            if d.is_null() {
                return Err(Error::NotImplemented(
                    "ef_tensor_image_desc_new: allocation failed".into(),
                ));
            }
            if let Some(m) = desc.memory() {
                edgefirst_tensor_ffi::ef_tensor_image_desc_set_memory(d, m.code());
            }
            edgefirst_tensor_ffi::ef_tensor_image_desc_set_access(d, access_code(desc.access()));
            if let Some(c) = desc.compression() {
                let code = match c {
                    crate::Compression::Any => 1,
                    // `Compression::Scheme(_)` has no C setter (see
                    // `ef_tensor_image_desc_set_compression`'s own doc) --
                    // request the closest expressible thing, "any scheme",
                    // rather than silently dropping the request. A caller
                    // that specifically needs a named scheme has no ABI
                    // path to it today.
                    crate::Compression::Scheme(_) => 1,
                };
                edgefirst_tensor_ffi::ef_tensor_image_desc_set_compression(d, code);
            }
            let handle = edgefirst_tensor_ffi::ef_tensor_image_desc_alloc(d);
            let result = match NonNull::new(handle) {
                Some(h) => Ok(Self::from_handle(h)),
                None => Err(ffi_error(Error::NotImplemented)),
            };
            edgefirst_tensor_ffi::ef_tensor_image_desc_free(d);
            result
        }
    }

    /// See [`Self::image`]: same primitive family, externally-strided form.
    /// Drives `ef_tensor_image_with_stride_alloc`.
    pub fn image_with_stride(
        width: usize,
        height: usize,
        format: PixelFormat,
        dtype: DType,
        row_stride_bytes: usize,
        memory: Option<TensorMemory>,
        access: CpuAccess,
    ) -> Result<Self> {
        let c_format = format_cstring(format)?;
        let (has_memory, memory_code) = memory_code(memory);
        // SAFETY: `c_format` is a valid NUL-terminated string for the
        // call's duration.
        let handle = unsafe {
            edgefirst_tensor_ffi::ef_tensor_image_with_stride_alloc(
                width,
                height,
                c_format.as_ptr(),
                dtype.code(),
                row_stride_bytes,
                has_memory,
                memory_code,
                access_code(access),
            )
        };
        match NonNull::new(handle) {
            Some(h) => Ok(Self::from_handle(h)),
            None => Err(ffi_error(Error::NotImplemented)),
        }
    }

    /// Borrow a rectangular spatial sub-region as a zero-copy view sharing
    /// this tensor's allocation. Same signature as `static`'s
    /// `TensorDyn::view` (`static_backend.rs`), which dispatches to
    /// `Tensor<T>::view` (`lib.rs`) -- real production code calls it
    /// (`edgefirst-image`'s tiled multi-slot convert path,
    /// `tiling.rs::render_tile`, views a row-band out of a tall parent
    /// destination tensor).
    ///
    /// Drives `ef_tensor_view_region`, which is *not* a composition over
    /// `ef_tensor_builder_wrap`/`add_plane` (that path would mint a fresh
    /// identity by duplicating a plane's fd, breaking
    /// [`TensorTrait::view`]'s "must never mint a fresh identity" contract)
    /// -- it calls the producer side's real `static::TensorDyn::view`
    /// directly, which shares the parent's allocation and
    /// [`BufferIdentity`] the same way the static backend's own callers get.
    pub fn view(&self, region: Region) -> Result<TensorDyn> {
        // SAFETY: `self.handle` is a live handle for the call's duration.
        let handle = unsafe {
            edgefirst_tensor_ffi::ef_tensor_view_region(
                self.handle.as_ptr(),
                region.x as u64,
                region.y as u64,
                region.width as u64,
                region.height as u64,
            )
        };
        match NonNull::new(handle) {
            Some(h) => Ok(Self::from_handle(h)),
            None => {
                // Only claim "out of bounds" when the producing side says
                // that is what happened. This used to assert it for *every*
                // NULL -- including an unresolvable handle and a region a
                // format rejected for reasons other than fit -- and to
                // fabricate the `bounds` from `width()`/`height()`, which
                // for a non-image tensor are `None` and rendered as `(0,
                // 0)`. Same shape of confident falsehood `batch` had, found
                // by the P2c sweep rather than by a failing test.
                if ffi_last_error_class() == edgefirst_tensor_ffi::EfErrorClass::RegionOutOfBounds {
                    // Rebuilt with real numbers rather than through
                    // `ffi_error`: this caller has them (`region` is its own
                    // argument, and the frame comes from the tensor).
                    Err(Error::RegionOutOfBounds {
                        region,
                        bounds: (self.width().unwrap_or(0), self.height().unwrap_or(0)),
                    })
                } else {
                    Err(ffi_error(Error::InvalidShape))
                }
            }
        }
    }

    /// Attach pixel format metadata to this tensor, validating shape
    /// compatibility. Same contract as `static`'s `TensorDyn::set_format`
    /// (`static_backend.rs`). Drives `ef_tensor_set_format`, the live-handle
    /// mutator (as opposed to the pre-alloc `ef_tensor_builder_format`
    /// setter). Real caller: `edgefirst-image`'s `import_image` (`lib.rs`),
    /// which tags an already-imported DMA-BUF fd with its pixel format
    /// immediately after `TensorDyn::from_fd`.
    ///
    /// **Concurrency**: `ef_tensor_set_format` is narrower than
    /// `ef_tensor_set_colorimetry` -- not safe to call while any other
    /// `tensor-capi` call is in flight on the same underlying handle from
    /// another thread (see `mutate.rs`'s module docs in `tensor-capi`).
    /// `&mut self` here already gives this call exclusive access to *this*
    /// Rust value, which is what every real caller (`import_image`, once,
    /// right after construction) relies on.
    pub fn set_format(&mut self, format: PixelFormat) -> Result<()> {
        let c_format = format_cstring(format)?;
        // SAFETY: `self.handle` is a live handle; `c_format` is valid for
        // the call's duration.
        let rc = unsafe {
            edgefirst_tensor_ffi::ef_tensor_set_format(self.handle.as_ptr(), c_format.as_ptr())
        };
        if rc != 0 {
            return Err(Error::InvalidArgument(format!(
                "ef_tensor_set_format failed: {}",
                ffi_last_error()
            )));
        }
        Ok(())
    }

    /// Builder-style variant of [`Self::set_format`].
    pub fn with_format(mut self, format: PixelFormat) -> Result<Self> {
        self.set_format(format)?;
        Ok(self)
    }

    /// Set the row stride without format validation. Same contract as
    /// `static`'s `TensorDyn::set_row_stride`. Drives
    /// `ef_tensor_set_row_stride`. Real caller: `edgefirst-image`'s
    /// `import_image` (`lib.rs`), immediately after [`Self::with_format`].
    /// Same concurrency note as [`Self::set_format`].
    pub fn set_row_stride(&mut self, stride: usize) -> Result<()> {
        // SAFETY: `self.handle` is a live handle.
        let rc =
            unsafe { edgefirst_tensor_ffi::ef_tensor_set_row_stride(self.handle.as_ptr(), stride) };
        if rc != 0 {
            return Err(Error::InvalidArgument(format!(
                "ef_tensor_set_row_stride failed: {}",
                ffi_last_error()
            )));
        }
        Ok(())
    }

    /// Set the row stride in bytes without format validation. Same contract
    /// as `static`'s `TensorDyn::set_row_stride_unchecked`
    /// (`static_backend.rs`). Drives `ef_tensor_set_row_stride_unchecked`
    /// (task 17) -- unlike [`Self::set_row_stride`], needs no format on this
    /// tensor first, which is what makes it usable on a raw multiplane
    /// chroma plane (`from_planes` requires that plane to carry no format;
    /// see [`crate::Tensor::from_planes`]'s doc comment). Same concurrency
    /// note as [`Self::set_format`].
    pub fn set_row_stride_unchecked(&mut self, stride: usize) {
        // SAFETY: `self.handle` is a live handle.
        unsafe {
            edgefirst_tensor_ffi::ef_tensor_set_row_stride_unchecked(self.handle.as_ptr(), stride)
        };
    }

    /// Retag this tensor's element type, keeping its bytes untouched.
    ///
    /// Drives `ef_tensor_set_dtype`. Needed here and not on the static
    /// backend because of where the element type lives: `static`'s
    /// `TensorDyn` is an enum over eleven `Tensor<T>`, so the dtype **is**
    /// the Rust type and a layout-identical `transmute` changes it for free.
    /// This backend's `Tensor<T>` is `#[repr(transparent)]` over a handle
    /// that records its own dtype, so the same transmute changes a
    /// `PhantomData` and nothing else -- the handle keeps saying `U8`. See
    /// [`crate::TensorDyn::set_dtype`]'s static counterpart for the contract
    /// and why a width change is refused.
    pub fn set_dtype(&mut self, dtype: DType) -> Result<()> {
        if self.dtype() == dtype {
            return Ok(());
        }
        // SAFETY: `self.handle` is live and own-mint.
        let rc = unsafe {
            edgefirst_tensor_ffi::ef_tensor_set_dtype(self.handle.as_ptr(), dtype.code())
        };
        if rc == 0 {
            return Ok(());
        }
        Err(Error::InvalidArgument(format!(
            "ef_tensor_set_dtype failed: {} (errno {rc})",
            ffi_last_error()
        )))
    }

    /// Set the byte offset within the DMA-BUF where image data starts.
    /// Drives `ef_tensor_set_plane_offset`. Same caller as
    /// [`Self::set_row_stride`] (`import_image`'s `image_offset`
    /// parameter), same concurrency note as [`Self::set_format`].
    ///
    /// `static`'s signature returns `()`, not `Result` -- there is no
    /// channel to report a failure through, but with the real primitive in
    /// place there is also nothing left that can fail here short of a
    /// process-level handle corruption `ef_tensor_set_plane_offset` itself
    /// would only hit for a null/foreign handle, neither reachable through
    /// `&mut self` on an owned, already-validated `TensorDyn`.
    pub fn set_plane_offset(&mut self, offset: usize) {
        // SAFETY: `self.handle` is a live handle.
        unsafe { edgefirst_tensor_ffi::ef_tensor_set_plane_offset(self.handle.as_ptr(), offset) };
    }

    /// Byte offset within the DMA-BUF where image data starts (`None` = 0).
    /// Drives `ef_tensor_plane_offset` -- found missing while proving
    /// `set_plane_offset` above (task 15): the only other candidate,
    /// `plane0().offset` (`ef_tensor_plane_at`), is `plane_table`'s
    /// intra-buffer layout offset, always 0 for plane 0 by construction, a
    /// different quantity entirely from what this setter writes. Same
    /// contract as `static`'s `TensorDyn::plane_offset` (`static_backend.rs`).
    pub fn plane_offset(&self) -> Option<usize> {
        // SAFETY: `self.handle` is live.
        let raw = unsafe { edgefirst_tensor_ffi::ef_tensor_plane_offset(self.handle.as_ptr()) };
        (raw >= 0).then_some(raw as usize)
    }

    /// Set this tensor's logical dimensions and pixel format to a decoded
    /// image, reusing the existing allocation -- the pool-reuse primitive a
    /// JPEG decode-into-pool destination tensor needs before each decode.
    /// Same contract as `static`'s `TensorDyn::configure_image`
    /// (`static_backend.rs`). Drives `ef_tensor_configure_image`, whose
    /// producer-side implementation is the real `static::Tensor::configure_image`
    /// -- the pool-reuse stride-preservation and alignment rules live there,
    /// not here. Real caller: `edgefirst-codec`'s JPEG decode-into-pool path,
    /// via [`dynamic_tensor::Tensor::configure_image`](crate::Tensor::configure_image),
    /// which forwards straight to this. Same concurrency note as
    /// [`Self::set_format`].
    pub fn configure_image(
        &mut self,
        width: usize,
        height: usize,
        format: PixelFormat,
    ) -> Result<()> {
        let c_format = format_cstring(format)?;
        // SAFETY: `self.handle` is a live handle; `c_format` is valid for
        // the call's duration.
        let rc = unsafe {
            edgefirst_tensor_ffi::ef_tensor_configure_image(
                self.handle.as_ptr(),
                width,
                height,
                c_format.as_ptr(),
            )
        };
        if rc != 0 {
            return Err(Error::InvalidArgument(format!(
                "ef_tensor_configure_image failed: {}",
                ffi_last_error()
            )));
        }
        // `shape_cache` was captured once at construction and does not
        // track `inner`'s live shape the way the C side's own
        // `EfTensorImpl` caches do -- `configure_image` just changed it, so
        // it must be re-derived the same way `from_handle` derives it the
        // first time, or `Self::shape()`/`Self::width()`/`Self::height()`
        // would report the pre-reconfigure geometry forever after.
        self.shape_cache = Self::query_shape(self.handle);
        // Third copy: the wrapped `PboTensor`. This is the decode-into-a-
        // pool path, so a stale geometry here is read by the very next GL
        // import. Capacity-based, matching the static backend, whose
        // `configure_image` calls `storage.set_logical_shape(&shape)`: the
        // new image is usually SMALLER than the pool buffer, which the
        // strict rule would refuse.
        let new_shape = self.shape_cache.clone();
        self.sync_pbo_shape(&new_shape, true)?;
        Ok(())
    }

    // --- Quantization: family 2's primitives. `Tensor<T>` (integer-gated
    // via `IntegerType`, `dynamic_tensor.rs`) forwards straight to these;
    // the type gating lives entirely at that layer, matching how
    // `static`'s own `TensorDyn::quantization`/etc. (`static_backend.rs`)
    // already fold every float variant to `None`/an error without a
    // separate type-erased gate of their own.

    /// Quantization metadata for this tensor (`None` for a tensor that
    /// carries none, *or* a float-dtype tensor). Drives the two-call
    /// `ef_tensor_quantization_info`/`ef_tensor_quantization_get` idiom,
    /// caching the result in [`Self::quantization_cache`] the first time
    /// this particular `TensorDyn` value reads it -- see that field's doc
    /// comment for why the cache is sound under concurrent access.
    pub fn quantization(&self) -> Option<&crate::Quantization> {
        self.quantization_cache
            .get_or_init(|| self.fetch_quantization())
            .as_ref()
    }

    /// The actual `ef_tensor_quantization_{info,get}` two-call fetch,
    /// isolated from [`Self::quantization`] so the `OnceLock::get_or_init`
    /// closure stays a one-line call. Reconstructs the exact `Quantization`
    /// mode (symmetric vs. not, per-tensor vs. per-channel) the producer
    /// attached, not merely a numerically-equivalent one -- the wire
    /// encoding always returns a full `zero_point` array (zero-filled for
    /// symmetric, see `ef_tensor_quantization_get`'s own doc comment), so
    /// this re-derives symmetry from an all-zero check rather than losing
    /// it.
    fn fetch_quantization(&self) -> Option<crate::Quantization> {
        let mut info = edgefirst_tensor_ffi::EfQuantizationInfo::default();
        // SAFETY: `self.handle` is live; `info` is a valid local out-param.
        let rc = unsafe {
            edgefirst_tensor_ffi::ef_tensor_quantization_info(self.handle.as_ptr(), &mut info)
        };
        if rc != 0 || info.has_quantization == 0 {
            return None;
        }
        let n = info.count as usize;
        let mut scales = vec![0f32; n];
        let mut zps = vec![0i32; n];
        // SAFETY: `self.handle` is live; `scales`/`zps` are sized to `n`,
        // the count `ef_tensor_quantization_info` just reported.
        let rc = unsafe {
            edgefirst_tensor_ffi::ef_tensor_quantization_get(
                self.handle.as_ptr(),
                scales.as_mut_ptr(),
                zps.as_mut_ptr(),
                n as u32,
            )
        };
        if rc != 0 {
            return None;
        }
        let symmetric = zps.iter().all(|&z| z == 0);
        if info.axis == -1 {
            Some(if symmetric {
                crate::Quantization::per_tensor_symmetric(scales[0])
            } else {
                crate::Quantization::per_tensor(scales[0], zps[0])
            })
        } else {
            let axis = info.axis as usize;
            let q = if symmetric {
                crate::Quantization::per_channel_symmetric(scales, axis)
            } else {
                crate::Quantization::per_channel(scales, zps, axis)
            };
            q.ok()
        }
    }

    /// Attach quantization metadata to this tensor. Drives
    /// `ef_tensor_quantization_set`, then refreshes
    /// [`Self::quantization_cache`] so a subsequent [`Self::quantization`]
    /// call on this same `TensorDyn` value observes it immediately rather
    /// than the value cached before this call (see that field's doc
    /// comment: `&mut self` here means no concurrent reader on *this*
    /// value can be mid-`get_or_init` while the field is replaced).
    ///
    /// # Errors
    ///
    /// [`Error::InvalidArgument`] if the underlying call fails -- including
    /// a float-dtype tensor, matching `static`'s own
    /// `QuantizationInvalid { field: "dtype_is_integer", .. }` refusal
    /// (folded here into the same generic errno-derived message every
    /// other setter in this file uses, not the structured variant).
    pub fn set_quantization(&mut self, q: crate::Quantization) -> Result<()> {
        let axis = q.axis().map(|a| a as i32).unwrap_or(-1);
        let scales = q.scale();
        let n = scales.len() as u32;
        let zps_owned = q.zero_point().map(|z| z.to_vec());
        let zps_ptr = zps_owned
            .as_ref()
            .map(|z| z.as_ptr())
            .unwrap_or(std::ptr::null());
        // SAFETY: `self.handle` is live; `scales` points to `n` elements;
        // `zps_ptr` is either null or points to `n` elements from the same
        // `Quantization`.
        let rc = unsafe {
            edgefirst_tensor_ffi::ef_tensor_quantization_set(
                self.handle.as_ptr(),
                axis,
                scales.as_ptr(),
                zps_ptr,
                n,
            )
        };
        if rc != 0 {
            return Err(Error::InvalidArgument(format!(
                "ef_tensor_quantization_set failed: {}",
                ffi_last_error()
            )));
        }
        self.quantization_cache = std::sync::OnceLock::new();
        let _ = self.quantization_cache.set(Some(q));
        Ok(())
    }

    /// Builder-style variant of [`Self::set_quantization`].
    pub fn with_quantization(mut self, q: crate::Quantization) -> Result<Self> {
        self.set_quantization(q)?;
        Ok(self)
    }

    /// Clear any quantization metadata on this tensor. Drives
    /// `ef_tensor_quantization_clear`, then resets
    /// [`Self::quantization_cache`] the same way [`Self::set_quantization`]
    /// does.
    pub fn clear_quantization(&mut self) {
        // SAFETY: `self.handle` is live.
        unsafe { edgefirst_tensor_ffi::ef_tensor_quantization_clear(self.handle.as_ptr()) };
        self.quantization_cache = std::sync::OnceLock::new();
        let _ = self.quantization_cache.set(None);
    }

    // --- The `edgefirst-python-common` surface.
    //
    // `crates/python-common/src/tensor.rs` is the first consumer to call
    // `TensorDyn` at the Rust level under `dynamic` -- every earlier one
    // reached it through the `ef_tensor_*` C ABI instead -- so this block
    // is the set of methods it needs that the `-capi` leaves never did.
    // Some are pure derivations over primitives that already existed; the
    // ones that are not name the `ef_tensor_*` entry they drive and why no
    // existing primitive could answer. See `.superpowers/sdd/
    // 2026-08-25-python-single-tensor-home/task-P2a-report.md`.

    /// Total logical size of this tensor in bytes.
    ///
    /// Derived, with no `ef_tensor_*` entry of its own:
    /// [`crate::TensorTrait::size`] is defined as `len() * size_of::<T>()`
    /// and `len()` as `shape().iter().product()`, both of which this
    /// backend can already answer exactly -- `DType::size()` is
    /// `size_of::<T>()` for every one of the eleven element types.
    ///
    /// Logical, **not** the allocation: a pitch-aligned or pool-sized
    /// tensor's allocation is larger. [`Self::capacity_bytes`] is that one.
    pub fn size(&self) -> usize {
        self.shape().iter().product::<usize>() * self.dtype().size()
    }

    /// Bytes of the underlying allocation, which is `>=` [`Self::size`].
    ///
    /// Drives `ef_tensor_capacity_bytes`. Not derivable from
    /// `ef_tensor_plane_at`: for a *formatted* tensor that reports per-plane
    /// geometry computed from the format's plane table, whose sum is the
    /// logical image size rather than the allocation's -- the two differ for
    /// exactly the pool-sized and pitch-padded tensors that make the
    /// distinction matter.
    pub fn capacity_bytes(&self) -> usize {
        // A PBO-backed tensor's allocation is the GL buffer's, not the
        // companion `Mem` handle's. `Tensor::from_pbo` sizes that companion
        // to `shape.product() * size_of::<T>()` exactly, while
        // `PboTensor::from_pbo` explicitly permits `size > shape.product()`
        // ("PBOs allocated with a 64-byte-aligned row stride may be larger
        // than the shape product"). Reading the companion would understate
        // the real buffer, clamping a `kind::PBO` descriptor's `capacity`
        // and leaving a consumer's `from_pbo_import` mapping only part of
        // it -- the same pool-reuse breakage this method already fixes for
        // `kind::HOST`, left standing for `PBO`. Same precedence
        // [`Self::memory`] applies, for the same reason.
        if let Some(bytes) = with_pbo!(self, capacity_bytes) {
            return bytes;
        }
        // SAFETY: `self.handle` is live for as long as `self` exists.
        let n = unsafe { edgefirst_tensor_ffi::ef_tensor_capacity_bytes(self.handle.as_ptr()) };
        // `-1` is the entry point's invalid-handle sentinel. Falling back to
        // the logical size matches `TensorTrait::capacity_bytes`'s own
        // default ("the logical size for storages without spare capacity")
        // rather than inventing a zero.
        usize::try_from(n).unwrap_or_else(|_| self.size())
    }

    /// The *recorded* row stride in bytes; `None` when the tensor is
    /// tightly packed.
    ///
    /// Drives `ef_tensor_row_stride`, and is deliberately **not**
    /// [`Self::effective_row_stride`]: that one reports plane 0's pitch,
    /// which falls back to a pitch computed from the format and width when
    /// nothing is recorded. The difference is load-bearing for
    /// [`Self::descriptor_pinned`] -- the cross-package protocol carries
    /// `None` for a tight tensor and lets the consumer recompute, and
    /// baking a computed pitch in instead would turn "no stride recorded"
    /// into "this exact stride is required" across a package boundary.
    pub fn row_stride(&self) -> Option<usize> {
        // SAFETY: `self.handle` is live for as long as `self` exists.
        let n = unsafe { edgefirst_tensor_ffi::ef_tensor_row_stride(self.handle.as_ptr()) };
        usize::try_from(n).ok()
    }

    /// The vendor tile-compression scheme recorded at allocation; `None`
    /// for a linear layout.
    ///
    /// Drives `ef_tensor_compression`. The scheme is chosen by the
    /// allocator inside `libedgefirst_tensor.so` (today only an Android
    /// AHardwareBuffer allocation ever resolves to one), so it is a fact
    /// only that library holds -- `ef_tensor_image_desc_get`'s own
    /// `compression` field is the *request* a caller made, which is a
    /// different question and routinely disagrees with the answer.
    /// An unrecognised code is reported as `None` and **logged**, not
    /// silently passed off as a linear layout. `Option<CompressionScheme>`
    /// has no variant for "some scheme this build cannot name", so `None`
    /// is the only value available -- but a consumer acting on it will
    /// treat a tile-compressed buffer as linear, and a compressed tensor
    /// has no meaningful linear row stride (see [`crate::Tensor::
    /// compression`]). The one way a code lands here unrecognised is a
    /// `libedgefirst_tensor.so` newer than this build; the loader's own
    /// undefined-symbol failure catches deployment skew that *adds a
    /// symbol*, and says nothing about skew that adds a *value* returned
    /// through an existing one. The log line is what makes that case
    /// visible instead of silent -- there is no ABI-version negotiation to
    /// refuse it first (`ef_tensor_abi_version` is defined and declared but
    /// nothing in the tree compares it to anything).
    pub fn compression(&self) -> Option<crate::CompressionScheme> {
        // SAFETY: `self.handle` is live for as long as `self` exists.
        let code = unsafe { edgefirst_tensor_ffi::ef_tensor_compression(self.handle.as_ptr()) };
        // 0 is the wire code for "linear", a real answer rather than an
        // unrecognised one, so it must not warn -- it is what almost every
        // allocation on almost every platform reports.
        let scheme = crate::CompressionScheme::from_code(code);
        if scheme.is_none() && code != 0 {
            log::warn!(
                "ef_tensor_compression returned scheme code {code}, which this build of \
                 edgefirst-tensor does not recognise; reporting a linear layout. The \
                 loaded libedgefirst_tensor.so is newer than this consumer."
            );
        }
        scheme
    }

    /// Acquire the buffer for CPU access -- the standalone cache-maintenance
    /// bracket. See [`crate::Tensor::sync_for_cpu`].
    ///
    /// Drives `ef_tensor_sync_for_cpu`. Not derivable here: the maintenance
    /// is per-backing (a dma-buf ioctl, an IOSurface lock, a no-op for
    /// coherent host memory, an honest refusal for a PBO), and this backend
    /// holds none of that state -- it holds a handle. Re-deriving the
    /// dma-buf arm locally from [`Self::clone_fd`] would work on Linux and
    /// be silently wrong everywhere else, which is exactly the shape of
    /// answer this backend must not invent.
    pub fn sync_for_cpu(&self, access: CpuAccess) -> Result<()> {
        self.sync(access, true)
    }

    /// Release the buffer back to the device. See
    /// [`crate::Tensor::sync_for_device`].
    ///
    /// Drives `ef_tensor_sync_for_device`; `access` must match the one that
    /// opened the bracket. See [`Self::sync_for_cpu`].
    pub fn sync_for_device(&self, access: CpuAccess) -> Result<()> {
        self.sync(access, false)
    }

    /// Shared body of the two sync brackets -- one place to translate the
    /// access code and the errno, so the two ends cannot drift into
    /// accepting or reporting different things.
    fn sync(&self, access: CpuAccess, to_cpu: bool) -> Result<()> {
        let what = if to_cpu {
            "sync_for_cpu"
        } else {
            "sync_for_device"
        };
        // A PBO-backed tensor's real bytes live in the GL buffer `pbo`
        // addresses; the `ef_tensor_*` handle beside it is ordinary host
        // memory (see [`Self::pbo`]'s own doc comment). Forwarding to the
        // ABI would therefore sync *that companion allocation* and report
        // `Ok(())` -- a bracket that ran no maintenance on the buffer the
        // caller meant, and said nothing about it. [`Self::map_pin`]
        // refuses here for the identical reason, and the static backend
        // gives the same refusal for its own `TensorStorage::Pbo`, so both
        // backends answer alike. Verified by inducing the failure: without
        // this guard, `sync_brackets_succeed_on_host_memory_and_are_
        // refused_by_a_pbo` (`tests/dynamic_primitives.rs`) sees `Ok(())`.
        if self.pbo.is_some() {
            return Err(Error::NotImplemented(format!(
                "{what}: a PBO has no coherency window independent of its map -- \
                 glMapBufferRange establishes the address and the visibility \
                 together, and glUnmapBuffer publishes the writes. Use \
                 Tensor::as_pbo().map() instead, which owns the pair."
            )));
        }
        let code = match access {
            CpuAccess::None => {
                return Err(Error::InvalidArgument(format!(
                    "{what}: CpuAccess::None is not a sync direction"
                )))
            }
            CpuAccess::Read => 1,
            CpuAccess::Write => 2,
            CpuAccess::ReadWrite => 3,
        };
        // SAFETY: `self.handle` is live for as long as `self` exists.
        let rc = unsafe {
            if to_cpu {
                edgefirst_tensor_ffi::ef_tensor_sync_for_cpu(self.handle.as_ptr(), code)
            } else {
                edgefirst_tensor_ffi::ef_tensor_sync_for_device(self.handle.as_ptr(), code)
            }
        };
        if rc == 0 {
            return Ok(());
        }
        // `ENOTSUP` is the C surface's translation of the
        // backing-has-no-independent-coherency-window refusal
        // (AHardwareBuffer on Android). Preserving it as `NotImplemented`
        // keeps the static backend's own error *kind* for the same
        // condition, so a caller matching on it behaves identically on both
        // backends.
        let msg = ffi_last_error();
        if rc == libc::ENOTSUP {
            return Err(Error::NotImplemented(msg));
        }
        Err(Error::InvalidOperation(format!(
            "{what}: {msg} (errno {rc})"
        )))
    }

    /// Borrow batch element `n` of a batched tensor (leading `N` dimension)
    /// as a zero-copy view sharing this tensor's allocation. See
    /// [`crate::Tensor::batch`].
    ///
    /// Drives `ef_tensor_batch`, which is a genuinely distinct primitive
    /// from `ef_tensor_view_region`: that one crops a spatial rectangle
    /// *within* one image and cannot express dropping the leading
    /// dimension. Like `view`, it calls the producer side's real
    /// `static::TensorDyn::batch`, so the result shares the parent's
    /// allocation instead of minting a fresh one.
    pub fn batch(&self, n: usize) -> Result<TensorDyn> {
        // SAFETY: `self.handle` is a live handle for the call's duration.
        let handle =
            unsafe { edgefirst_tensor_ffi::ef_tensor_batch(self.handle.as_ptr(), n as u64) };
        match NonNull::new(handle) {
            Some(h) => Ok(Self::from_handle(h)),
            None => {
                // Read the class before anything else on this thread can
                // overwrite it -- same lifetime rule as the message.
                let is_index = ffi_last_error_class()
                    == edgefirst_tensor_ffi::EfErrorClass::BatchIndexOutOfBounds;
                Err(batch_error(n, self.shape().first().copied(), is_index))
            }
        }
    }

    /// Clone the DMA-BUF file descriptor backing this tensor (Linux only).
    ///
    /// Derived, with no `ef_tensor_*` entry of its own: the same
    /// memory-kind check the static backend performs, then
    /// [`Self::clone_fd`], which already `dup`s plane 0's native fd.
    ///
    /// # Errors
    ///
    /// * [`Error::NotImplemented`] if the tensor is not DMA-backed
    /// * [`Error::IoError`] if the `dup` syscall fails
    #[cfg(target_os = "linux")]
    pub fn dmabuf_clone(&self) -> Result<std::os::fd::OwnedFd> {
        if self.memory() != TensorMemory::DmaBuf {
            return Err(Error::NotImplemented(format!(
                "dmabuf_clone requires DMA-backed tensor, got {:?}",
                self.memory()
            )));
        }
        self.clone_fd()
    }

    /// Borrow the raw `IOSurfaceRef` backing this handle (macOS/iOS).
    ///
    /// Drives `ef_tensor_iosurface_ref`. `None` when the producing library
    /// has no IOSurface (wrong backing, or not Apple).
    #[cfg(any(target_os = "macos", target_os = "ios"))]
    pub fn iosurface_ref(&self) -> Option<*mut std::ffi::c_void> {
        // SAFETY: `self.handle` is live for as long as `self` exists.
        let p = unsafe { edgefirst_tensor_ffi::ef_tensor_iosurface_ref(self.handle.as_ptr()) };
        if p.is_null() {
            None
        } else {
            Some(p)
        }
    }

    /// Physical IOSurface dimensions in texels, independent of logical shape.
    ///
    /// There is no `ef_tensor_iosurface_physical_dims` primitive (unlike the
    /// AHardwareBuffer counterpart), so this reads `IOSurfaceGetWidth` /
    /// `IOSurfaceGetHeight` on the borrowed ref. `None` when
    /// [`Self::iosurface_ref`] is `None`.
    #[cfg(any(target_os = "macos", target_os = "ios"))]
    pub fn iosurface_physical_dims(&self) -> Option<(usize, usize)> {
        let surface = self.iosurface_ref()?;
        // SAFETY: `surface` is a live IOSurfaceRef owned by the producing
        // library for at least as long as `self` (the handle) exists.
        unsafe { Some((IOSurfaceGetWidth(surface), IOSurfaceGetHeight(surface))) }
    }

    /// The CUDA registration for this tensor, if any.
    ///
    /// Reads [`Self::cuda`], with no `ef_tensor_*` entry: `CudaHandle` is
    /// already backend-agnostic in-process Rust state (see `crate::cuda`'s
    /// module doc and [`crate::Tensor::cuda`], which task 18 added for the
    /// typed lens) -- there is nothing about it to send across the
    /// boundary.
    pub fn cuda(&self) -> Option<&crate::cuda::CudaHandle> {
        self.cuda.as_deref()
    }

    /// Fast-fail CUDA map: `None` when no handle is attached; else a scoped
    /// device-pointer guard. Same contract as [`crate::Tensor::cuda_map`]
    /// and as the static backend's own `TensorDyn::cuda_map`.
    pub fn cuda_map(&self) -> Option<crate::cuda::CudaMap<'_>> {
        self.cuda()?.map()
    }

    /// GL buffer ID for this PBO; `None` when the tensor is not PBO-backed.
    ///
    /// Reads [`Self::pbo`], with no `ef_tensor_*` entry: a `PboTensor`'s
    /// state is in-process Rust state with no wire representation at all
    /// (see that field's own doc comment). The backing `ef_tensor_*` handle
    /// is ordinary host memory and knows nothing about the GL buffer.
    pub fn pbo_id(&self) -> Option<u32> {
        with_pbo!(self, buffer_id)
    }

    /// The C-ABI `PboOpsVtable` address for this PBO, for cross-cdylib
    /// export via [`crate::TensorDesc::ptr`]; `None` when not PBO-backed.
    /// See [`Self::pbo_id`] for why this needs no `ef_tensor_*` entry.
    pub fn pbo_vtable_ptr(&self) -> Option<*const std::ffi::c_void> {
        with_pbo!(self, pbo_vtable).map(|v| v as *const _ as *const std::ffi::c_void)
    }

    /// A type-erased keepalive that must stay alive for at least as long as
    /// [`Self::pbo_vtable_ptr`]'s address is used; `None` when not
    /// PBO-backed. See [`crate::pbo::PboTensor::pbo_keepalive`]'s own doc
    /// comment, and [`Self::pbo_id`] for why this needs no `ef_tensor_*`
    /// entry.
    pub fn pbo_keepalive(&self) -> Option<std::sync::Arc<dyn Send + Sync>> {
        with_pbo!(self, pbo_keepalive)
    }

    /// Wrap a producer's host pointer as a type-erased tensor without
    /// copying, aliasing rather than owning it -- the consumer half of the
    /// capsule protocol's [`crate::protocol::kind::HOST`].
    ///
    /// Drives `ef_tensor_wrap_host`. Not expressible over the builder the
    /// way [`Self::from_fd`] is: `ef_tensor_builder_add_plane` takes a
    /// *native handle* (an fd or surface id), and a host address is neither
    /// -- there was no primitive that could hand `libedgefirst_tensor.so` a
    /// raw pointer to alias.
    ///
    /// # Safety
    ///
    /// Same as the static backend's `from_raw_host_with_capacity`: `ptr`
    /// must be non-null, aligned to `dtype`, and valid for
    /// `max(capacity_bytes, shape.product() * dtype.size())` bytes for as
    /// long as the returned tensor -- and every view/map sharing its
    /// backing -- is used. Nothing here extends that lifetime.
    pub unsafe fn from_raw_host_with_capacity(
        ptr: *mut u8,
        shape: &[usize],
        capacity_bytes: usize,
        dtype: DType,
        _name: Option<&str>,
    ) -> Result<Self> {
        let dims: Vec<u64> = shape.iter().map(|&d| d as u64).collect();
        // SAFETY: the caller upholds `ptr`'s validity (this function's own
        // contract); `dims` is a live local for the call's duration.
        let handle = unsafe {
            edgefirst_tensor_ffi::ef_tensor_wrap_host(
                ptr,
                capacity_bytes,
                dtype.code(),
                dims.as_ptr(),
                dims.len() as u32,
            )
        };
        match NonNull::new(handle) {
            Some(h) => Ok(Self::from_handle(h)),
            None => Err(Error::InvalidArgument(format!(
                "ef_tensor_wrap_host failed: {}",
                ffi_last_error()
            ))),
        }
    }

    /// Wrap a live `IOSurfaceRef` (macOS/iOS). Looks up the surface's
    /// `IOSurfaceID` and drives [`Self::from_iosurface_id`]: the C ABI
    /// does not take a raw ref, so the id is the portable handle.
    #[cfg(any(target_os = "macos", target_os = "ios"))]
    pub unsafe fn from_iosurface(
        surface_ref: *mut std::ffi::c_void,
        shape: &[usize],
        dtype: DType,
        name: Option<&str>,
    ) -> Result<Self> {
        if surface_ref.is_null() {
            return Err(Error::InvalidArgument(
                "from_iosurface: surface_ref is null".into(),
            ));
        }
        // SAFETY: caller guarantees `surface_ref` is a live IOSurfaceRef.
        let id = unsafe { IOSurfaceGetID(surface_ref) };
        Self::from_iosurface_id(id, shape, dtype, name)
    }

    /// IOSurfaceID for cross-process sharing. `None` when this handle has
    /// no IOSurface (wrong backing, or `IOSurfaceGetID` returned 0).
    #[cfg(any(target_os = "macos", target_os = "ios"))]
    pub fn iosurface_id(&self) -> Option<u32> {
        let surface = self.iosurface_ref()?;
        let id = unsafe { IOSurfaceGetID(surface) };
        if id == 0 {
            None
        } else {
            Some(id)
        }
    }

    /// Wrap a live IOSurface named by its cross-process `IOSurfaceID`
    /// (macOS/iOS only) -- the consumer half of the capsule protocol's
    /// [`crate::protocol::kind::IOSURFACE`].
    ///
    /// Drives `ef_tensor_from_iosurface_id`. The lookup, the liveness
    /// check, and the retain all happen inside `libedgefirst_tensor.so`:
    /// `IOSurfaceRef` is a CoreFoundation object with its own refcount
    /// discipline, and splitting "look it up" from "retain it" across an
    /// ABI boundary would put a window between them in which the surface
    /// can be freed. Passing the *id* rather than the ref is what closes
    /// that window.
    #[cfg(any(target_os = "macos", target_os = "ios"))]
    pub fn from_iosurface_id(
        id: u32,
        shape: &[usize],
        dtype: DType,
        _name: Option<&str>,
    ) -> Result<Self> {
        let dims: Vec<u64> = shape.iter().map(|&d| d as u64).collect();
        // SAFETY: `dims` is a live local for the call's duration.
        let handle = unsafe {
            edgefirst_tensor_ffi::ef_tensor_from_iosurface_id(
                id,
                dtype.code(),
                dims.as_ptr(),
                dims.len() as u32,
            )
        };
        match NonNull::new(handle) {
            Some(h) => Ok(Self::from_handle(h)),
            None => Err(Error::InvalidArgument(format!(
                "ef_tensor_from_iosurface_id failed: {}",
                ffi_last_error()
            ))),
        }
    }

    /// Rebuild a type-erased PBO tensor from a cross-cdylib `ops` (see
    /// [`crate::pbo::import_pbo_ops`]) plus the geometry a
    /// [`crate::TensorDesc`] under [`crate::protocol::kind::PBO`] carries.
    ///
    /// Needs no `ef_tensor_*` primitive: a `PboTensor` is in-process Rust
    /// state (a GL buffer id plus an `Arc<dyn PboOps>` routing map/unmap
    /// back through the producer's vtable) with no wire representation at
    /// all -- see [`Self::pbo`]'s own doc comment. The per-dtype dispatch
    /// mirrors the static backend's `from_pbo_import` exactly; the
    /// difference is only where the resulting `PboTensor<T>` is stored
    /// (that field, versus a `TensorStorage::Pbo` variant).
    ///
    /// Inherits [`crate::Tensor::from_pbo`]'s cost: minting the companion
    /// metadata handle allocates the PBO's full byte count as ordinary host
    /// RAM. See that constructor's doc comment for the measurement and the
    /// follow-up primitive that should replace it.
    pub(crate) fn from_pbo_import(
        buffer_id: u32,
        size: usize,
        shape: &[usize],
        dtype: DType,
        ops: std::sync::Arc<dyn crate::PboOps>,
    ) -> Result<Self> {
        macro_rules! arm {
            ($t:ty) => {
                crate::PboTensor::<$t>::from_pbo(buffer_id, size, shape, None, ops)
                    .and_then(crate::Tensor::<$t>::from_pbo)
                    .map(crate::Tensor::<$t>::into_inner)
            };
        }
        match dtype {
            DType::U8 => arm!(u8),
            DType::I8 => arm!(i8),
            DType::U16 => arm!(u16),
            DType::I16 => arm!(i16),
            DType::U32 => arm!(u32),
            DType::I32 => arm!(i32),
            DType::U64 => arm!(u64),
            DType::I64 => arm!(i64),
            DType::F16 => arm!(half::f16),
            DType::F32 => arm!(f32),
            DType::F64 => arm!(f64),
        }
    }

    /// The per-kind construction switch [`TensorDyn::import_descriptor`]
    /// (`derived.rs`) drives, once the descriptor has been validated and
    /// its `dtype`/`shape` decoded.
    ///
    /// Backend-specific because it is the one part of the import that names
    /// *constructors*; everything around it (validation, and the
    /// format/stride/colorimetry restore afterward) is shared. See
    /// `derived.rs` for the whole function's contract.
    pub(crate) fn import_storage(
        desc: &crate::TensorDesc,
        shape: &[usize],
        dtype: DType,
    ) -> Result<Self> {
        match desc.kind {
            #[cfg(target_os = "linux")]
            crate::protocol::kind::DMABUF => {
                let fd = crate::protocol::dup_descriptor_fd(desc)?;
                Self::from_fd(fd, shape, dtype, None)
            }
            #[cfg(not(target_os = "linux"))]
            crate::protocol::kind::DMABUF => {
                Err(Error::NotImplemented("dma-buf import off Linux".into()))
            }
            #[cfg(any(target_os = "macos", target_os = "ios"))]
            crate::protocol::kind::IOSURFACE => {
                let id = crate::protocol::descriptor_surface_id(desc)?;
                Self::from_iosurface_id(id, shape, dtype, None)
            }
            #[cfg(not(any(target_os = "macos", target_os = "ios")))]
            crate::protocol::kind::IOSURFACE => Err(Error::NotImplemented(
                "IOSurface import off Apple platforms".into(),
            )),
            crate::protocol::kind::HOST => {
                crate::protocol::check_descriptor_host_ptr(desc)?;
                // SAFETY: the caller guarantees the producer's keepalive
                // outlives the returned tensor -- that is the capsule
                // contract `import_descriptor` documents. `desc.capacity`
                // comes from the same trusted producer as `ptr`/`shape`.
                unsafe {
                    Self::from_raw_host_with_capacity(
                        desc.ptr.0,
                        shape,
                        desc.capacity as usize,
                        dtype,
                        None,
                    )
                }
            }
            crate::protocol::kind::PBO => {
                let buffer_id = crate::protocol::descriptor_pbo_buffer_id(desc)?;
                // SAFETY: same capsule contract as the `HOST` arm above,
                // extended to what `desc.ptr` means under `kind::PBO` (see
                // `TensorDesc::ptr`'s own doc comment).
                let ops = unsafe { crate::pbo::import_pbo_ops(desc.ptr.0 as *const _)? };
                Self::from_pbo_import(buffer_id, desc.capacity as usize, shape, dtype, ops)
            }
            k => Err(Error::NotImplemented(format!(
                "tensor interop kind {k} cannot be imported by this build"
            ))),
        }
    }

    /// Descriptor for the cross-package tensor protocol; see
    /// [`crate::protocol`]'s module docs.
    pub fn descriptor(&self) -> crate::TensorDesc {
        self.descriptor_pinned(None)
    }

    /// Descriptor carrying a pinned host address. See the static backend's
    /// `TensorDyn::descriptor_pinned` for the contract -- this is the same
    /// function over the same [`crate::protocol::from_parts`], which is
    /// compiled into both backends precisely so the two cannot describe the
    /// same tensor differently.
    ///
    /// The one input fetched differently here is `handle`: the static
    /// backend branches on the platform to pick `dmabuf()`'s fd or
    /// `iosurface_id()`, whereas this reads plane 0's `handle` --
    /// `ef_tensor_plane_at`'s own `native_handle` (`tensor-capi`'s
    /// `vtable.rs`) already makes exactly that platform choice on the
    /// producing side, so re-deriving it here would be a second copy of the
    /// same decision, free to drift.
    pub fn descriptor_pinned(&self, pin: Option<&crate::HostPin<'_>>) -> crate::TensorDesc {
        let memory = self.memory();
        let handle: i64 = match memory {
            // A PBO's "native handle" is its GL buffer id, which lives in
            // `pbo` and never in the backing `ef_tensor_*` handle -- plane 0
            // would report `-1` here.
            TensorMemory::Pbo => self.pbo_id().map(|id| id as i64).unwrap_or(-1),
            _ => self.plane0().map(|p| p.handle).unwrap_or(-1),
        };
        crate::protocol::from_parts(crate::protocol::DescParts {
            dims: self.shape(),
            memory,
            dtype: self.dtype(),
            fourcc: self.format().map(|f| f.to_fourcc()).unwrap_or(0),
            format: self.format(),
            row_stride: self.row_stride(),
            handle,
            colorimetry: self.colorimetry().map(|c| c.pack()).unwrap_or(0),
            capacity: self.capacity_bytes() as u64,
            pin,
            pbo_vtable_ptr: self.pbo_vtable_ptr(),
        })
    }
}

/// Call a `PboTensor<T>` method through [`TensorDyn::pbo`]'s type erasure.
///
/// `pbo` is a `Box<dyn Any>` because `TensorDyn` carries no element type,
/// so reaching `PboTensor<T>`'s inherent methods needs a concrete `T` --
/// recovered here by matching on [`TensorDyn::dtype`], which
/// `Tensor::from_pbo` (`dynamic_tensor.rs`) guarantees agrees with the
/// stored value's `T` (it mints the backing handle with `T::DTYPE`). The
/// same downcast technique `lens.rs`'s `as_typed` already uses for the
/// handle itself, applied to the eleven element types instead of one.
///
/// Yields `None` for a tensor with no PBO, which is every tensor except one
/// built by `Tensor::from_pbo`.
/// [`with_pbo!`]'s mutable sibling, for the geometry mutators.
///
/// A PBO-backed `TensorDyn` carries geometry in two places -- the
/// `ef_tensor_*` handle and the `PboTensor` behind [`TensorDyn::pbo`] -- and
/// a mutator that updates only the first leaves them disagreeing, with
/// `shape()` saying one thing and `as_pbo().shape` another. Nothing errors,
/// because the byte length still matches; `edgefirst-image` just reads the
/// stale geometry off `as_pbo()`. Reviewed as F6 on task P2b.
///
/// Tries each concrete instantiation rather than picking one from
/// `dtype()`, for the same reason [`with_pbo!`] does: after a
/// [`TensorDyn::set_dtype`] retag the two can legitimately disagree, and
/// keying on the dtype finds nothing.
macro_rules! with_pbo_mut {
    ($self:expr, $method:ident, $arg:expr) => {{
        match $self.pbo.as_mut() {
            None => None,
            Some(any) => {
                macro_rules! arm {
                    ($t:ty) => {
                        any.downcast_mut::<crate::PboTensor<$t>>().map(|p| {
                            <crate::PboTensor<$t> as crate::TensorTrait<$t>>::$method(p, $arg)
                        })
                    };
                }
                None.or_else(|| arm!(u8))
                    .or_else(|| arm!(i8))
                    .or_else(|| arm!(u16))
                    .or_else(|| arm!(i16))
                    .or_else(|| arm!(half::f16))
                    .or_else(|| arm!(u32))
                    .or_else(|| arm!(i32))
                    .or_else(|| arm!(f32))
                    .or_else(|| arm!(u64))
                    .or_else(|| arm!(i64))
                    .or_else(|| arm!(f64))
            }
        }
    }};
}
use with_pbo_mut;

macro_rules! with_pbo {
    ($self:expr, $method:ident) => {{
        match $self.pbo.as_ref() {
            None => None,
            Some(any) => {
                // Try each concrete instantiation until one matches, rather
                // than picking one from `$self.dtype()`.
                //
                // Those two can legitimately disagree. `edgefirst-image`
                // allocates a PBO as `u8` and hands it back as an `i8`
                // tensor; `From<Tensor<T>> for TensorDyn` retags the
                // *handle*'s dtype to match (see its doc comment), but the
                // `PboTensor<u8>` in this box is a real value behind a real
                // `Any` vtable, which no transmute of the enclosing
                // `Tensor<T>` touches. Keying on `dtype()` therefore looked
                // for a `PboTensor<i8>`, found nothing, and reported the
                // tensor as having no PBO at all -- `pbo_id()` returned
                // `None`, so its descriptor carried no buffer id and a
                // cross-package re-import failed with "PBO descriptor
                // carries no buffer id".
                //
                // Sound because `downcast_ref` is an exact `TypeId` match,
                // so at most one arm can hit and the order is irrelevant,
                // and because every method reached through this macro
                // (`buffer_id`, `pbo_vtable`, `pbo_keepalive`) reads the
                // shared `PboHandle` and does not depend on `T`.
                let p = None
                    .or_else(|| {
                        any.downcast_ref::<crate::PboTensor<u8>>()
                            .map(|p| p.$method())
                    })
                    .or_else(|| {
                        any.downcast_ref::<crate::PboTensor<i8>>()
                            .map(|p| p.$method())
                    })
                    .or_else(|| {
                        any.downcast_ref::<crate::PboTensor<u16>>()
                            .map(|p| p.$method())
                    })
                    .or_else(|| {
                        any.downcast_ref::<crate::PboTensor<i16>>()
                            .map(|p| p.$method())
                    })
                    .or_else(|| {
                        any.downcast_ref::<crate::PboTensor<half::f16>>()
                            .map(|p| p.$method())
                    })
                    .or_else(|| {
                        any.downcast_ref::<crate::PboTensor<u32>>()
                            .map(|p| p.$method())
                    })
                    .or_else(|| {
                        any.downcast_ref::<crate::PboTensor<i32>>()
                            .map(|p| p.$method())
                    })
                    .or_else(|| {
                        any.downcast_ref::<crate::PboTensor<f32>>()
                            .map(|p| p.$method())
                    })
                    .or_else(|| {
                        any.downcast_ref::<crate::PboTensor<u64>>()
                            .map(|p| p.$method())
                    })
                    .or_else(|| {
                        any.downcast_ref::<crate::PboTensor<i64>>()
                            .map(|p| p.$method())
                    })
                    .or_else(|| {
                        any.downcast_ref::<crate::PboTensor<f64>>()
                            .map(|p| p.$method())
                    });
                p
            }
        }
    }};
}
use with_pbo;

/// Turn `ef_tensor_batch`'s `NULL` into the error [`crate::Tensor::batch`]
/// would have returned.
///
/// `ef_tensor_batch` returns `NULL` for four distinct conditions -- an
/// unresolvable handle, an `n` not representable as `usize` on the
/// producing side, an out-of-range index, and `Tensor::batch`'s "this
/// tensor is not batched" shape refusal. Collapsing them into
/// [`Error::BatchIndexOutOfBounds`] produced a statement that was not
/// merely unhelpful but **false**: `.batch(0)` on an ordinary NV12 image
/// reported "batch index 0 out of bounds for batch size 720", pointing the
/// reader at the one argument they got right.
///
/// The first fix for that reconstructed the variant by matching a fragment
/// of `BatchIndexOutOfBounds`'s own `Display` in the message -- an ABI
/// programmed against a `Display` string, tolerable only because it
/// degraded to a truth. `ef_tensor_last_error_class` (task P2c) is the
/// structured signal it wanted, and the string matching is gone.
///
/// The out-of-bounds case is rebuilt with its real numbers rather than
/// taking `ffi_error`'s generic mapping, because this caller has them: `n`
/// is its own argument and the leading dimension is on the cached shape.
/// Every other class falls through to [`ffi_error`], carrying the
/// producing side's own message.
fn batch_error(n: usize, leading: Option<usize>, class_is_index: bool) -> Error {
    match leading {
        Some(batch) if class_is_index => Error::BatchIndexOutOfBounds { index: n, batch },
        // `InvalidShape` is the fallback because it is what `Tensor::batch`
        // (`lib.rs`) itself returns for both of its shape refusals, which
        // are the failures an older library would report unclassified.
        _ => ffi_error(Error::InvalidShape),
    }
}

/// Turn a builder's sticky errno into the typed [`Error`] the static
/// backend would have returned.
///
/// Driven by the **builder's own errno**, not by the thread-local error
/// class P2c added. That is deliberate: `ef_tensor_builder_error(b)` is set
/// by this call on this builder, so it cannot be stale, whereas the
/// thread-local is only as fresh as the last path that wrote it -- and
/// `terminal`'s early returns (a null builder, a sticky error from an
/// earlier call) write no message at all. The errno is the sound channel
/// here; [`ffi_last_error`] supplies advisory text on top of it.
///
/// Before this, every refusal came back as
/// `NotImplemented("ef_tensor_builder_alloc failed: errno 12")` -- ENOMEM
/// for everything, with the real reason discarded on the producing side.
/// A caller could not tell "try a smaller allocation" from "this build has
/// no such backing", and `tests/vocabulary.rs`'s
/// `a_defined_but_unbacked_code_errors_instead_of_panicking` failed under
/// `dynamic` for exactly that reason -- which is why that file was excluded
/// by name from the dynamic test lane.
fn builder_error(errno: std::ffi::c_int) -> Error {
    let msg = ffi_last_error();
    // Mirrors `errno_for` (`tensor-capi/src/map.rs`) in reverse; that
    // function is the single place the forward mapping lives, so the two
    // are read together rather than each inventing a table.
    match errno {
        libc::ENOTSUP => Error::NotImplemented(msg),
        libc::ENOMEM => Error::IoError(std::io::Error::from_raw_os_error(libc::ENOMEM)),
        libc::ERANGE => Error::ShapeMismatch(msg),
        libc::EACCES => Error::InvalidOperation(msg),
        libc::EIO => Error::IoError(std::io::Error::other(msg)),
        libc::EINVAL => Error::InvalidArgument(msg),
        other => Error::InvalidArgument(format!("{msg} (errno {other})")),
    }
}

/// Derive a [`BufferIdentity`] from an fd's `(st_dev, st_ino)`./// Derive a [`BufferIdentity`] from an fd's `(st_dev, st_ino)`. See
/// [`TensorDyn::derive_identity`]'s doc comment for why this is the correct
/// key for a DMA-BUF handle. Mirrors `dma.rs::identity_from_stat` exactly
/// (that one is `static`-backend-private and `target_os = "linux"`-gated;
/// this one only needs `unix`, since it works from a bare fd via `nix`
/// rather than any Linux-specific DMA-BUF API).
// See `dma.rs::identity_from_stat`'s own `#[allow]` for why this cast is
// needed on exactly one platform per field and both are needed to compile
// on both.
#[cfg(unix)]
#[allow(clippy::unnecessary_cast)]
fn identity_from_stat(stat: &nix::sys::stat::FileStat) -> BufferIdentity {
    let key = ((stat.st_dev as u64) << 32) ^ (stat.st_ino as u64);
    BufferIdentity::derived(IdentityKind::DmaBuf, key)
}

/// `NUL`-terminate a [`PixelFormat`]'s wire string for a C call.
///
/// # Errors
///
/// [`Error::InvalidArgument`] in the (unreachable in practice) case that a
/// format's wire string somehow contains a NUL byte -- `PixelFormat::as_str`
/// always returns a static ASCII string, but the fallible `CString::new`
/// path is kept explicit rather than `.unwrap()`-ed, matching this crate's
/// standing rule against silently swallowing an error class just because
/// it looks unreachable today.
fn format_cstring(format: PixelFormat) -> Result<std::ffi::CString> {
    std::ffi::CString::new(format.as_str())
        .map_err(|e| Error::InvalidArgument(format!("pixel format string contains a NUL: {e}")))
}

/// Encode `Option<TensorMemory>` as the `(has_memory, memory)` pair every
/// `ef_tensor_image_*` constructor takes -- 0/0 for "auto-select", matching
/// `ef_tensor_image_desc_view`'s own `has_memory` convention rather than a
/// sentinel code (every `ef_storage_kind` value 0..=5 is real).
fn memory_code(memory: Option<TensorMemory>) -> (std::ffi::c_int, u32) {
    match memory {
        Some(m) => (1, m.code()),
        None => (0, 0),
    }
}

/// Encode [`CpuAccess`] as the `ef_cpu_access` wire code, `None` included
/// (0) -- unlike [`TensorDyn::map_pin`]'s access decode, which rejects 0 as
/// "not a mappable direction", image allocation legitimately wants
/// `CpuAccess::None` (e.g. a GPU-only render target).
fn access_code(access: CpuAccess) -> u32 {
    match access {
        CpuAccess::None => 0,
        CpuAccess::Read => 1,
        CpuAccess::Write => 2,
        CpuAccess::ReadWrite => 3,
    }
}

/// Read the calling thread's most recent `tensor-capi` failure detail, for
/// enriching the `Error` this backend returns on a constructor/mutator
/// failure -- see `ef_tensor_last_error_message`'s own doc comment for the
/// "advisory, `dlerror`-style, valid until this thread's next call"
/// contract. Never `""`-checked by the caller: an empty string is a
/// perfectly fine (if uninformative) suffix to a message.
pub(crate) fn ffi_last_error() -> String {
    // SAFETY: `ef_tensor_last_error_message` always returns a valid,
    // NUL-terminated string (possibly empty), read immediately and copied
    // out before any further `tensor-capi` call on this thread.
    unsafe {
        let ptr = edgefirst_tensor_ffi::ef_tensor_last_error_message();
        if ptr.is_null() {
            return String::new();
        }
        std::ffi::CStr::from_ptr(ptr).to_string_lossy().into_owned()
    }
}

/// Read the calling thread's most recent `tensor-capi` failure **class**,
/// beside [`ffi_last_error`]'s message.
///
/// The entry points that report failure by returning `NULL` carry no errno,
/// so this is the only structured signal available for them. Unlike the
/// message -- whose contract is "never parse this" -- this is meant to be
/// programmed against.
fn ffi_last_error_class() -> edgefirst_tensor_ffi::EfErrorClass {
    use edgefirst_tensor_ffi::EfErrorClass as C;
    // SAFETY: a plain thread-local read on the producing side; no pointers.
    let code = unsafe { edgefirst_tensor_ffi::ef_tensor_last_error_class() };
    match code {
        1 => C::InvalidArgument,
        2 => C::InvalidShape,
        3 => C::InsufficientCapacity,
        4 => C::BatchIndexOutOfBounds,
        5 => C::RegionOutOfBounds,
        6 => C::NotSupported,
        7 => C::InvalidOperation,
        8 => C::AllocationFailed,
        9 => C::QuantizationInvalid,
        // 0, and anything a newer library added that this build does not
        // know: "no class recorded", which is exactly how an unclassified
        // failure already reads. Never a guess at the nearest neighbour.
        _ => C::Unspecified,
    }
}

/// Rebuild a typed [`Error`] from the class and message the last failing
/// `ef_tensor_*` call recorded.
///
/// This replaces the string matching `TensorDyn::batch` used to do. That
/// coupling was tolerable only because it degraded to a truth rather than a
/// falsehood, but it was still an ABI programmed against a `Display`
/// fragment; `ef_tensor_last_error_class` is the structured signal it
/// wanted and did not have.
///
/// The message is always the producing side's own, whatever the class. An
/// `Unspecified` class -- an older library, or a failure path that recorded
/// only a message -- yields `fallback`, so this degrades exactly the way
/// the string match did: to a less specific truth, never to a confident
/// wrong answer.
///
/// Variants carrying structured fields (`RegionOutOfBounds`,
/// `BatchIndexOutOfBounds`, `InsufficientCapacity`) are not rebuilt here:
/// their numbers do not cross the boundary. The two callers that can supply
/// them from their own arguments do so themselves; everyone else gets a
/// message-carrying variant of the right *kind*, which is what a consumer
/// reads.
pub(crate) fn ffi_error(fallback: fn(String) -> Error) -> Error {
    use edgefirst_tensor_ffi::EfErrorClass as C;
    let msg = ffi_last_error();
    match ffi_last_error_class() {
        C::InvalidArgument => Error::InvalidArgument(msg),
        C::InvalidShape => Error::InvalidShape(msg),
        C::InsufficientCapacity | C::BatchIndexOutOfBounds | C::RegionOutOfBounds => {
            Error::ShapeMismatch(msg)
        }
        C::NotSupported => Error::NotImplemented(msg),
        C::InvalidOperation => Error::InvalidOperation(msg),
        C::AllocationFailed => Error::IoError(std::io::Error::other(msg)),
        C::QuantizationInvalid => Error::InvalidArgument(msg),
        C::Unspecified => fallback(msg),
    }
}

/// The keepalive behind [`TensorDyn::map_bytes`]'s pin. Holds its own
/// `ef_tensor_retain`'d reference (independent of the `TensorDyn` that
/// created it, matching the static backend's "genuinely `'static`" pin
/// contract) and releases it on `Drop`, after unmapping.
struct MapKeepalive(*mut EfTensor);

// SAFETY: the referenced handle is `libedgefirst_tensor`'s own C ABI object,
// designed to cross threads; ownership is tracked by refcount
// (`ef_tensor_retain`/`ef_tensor_free`), not by Rust's aliasing rules.
unsafe impl Send for MapKeepalive {}
unsafe impl Sync for MapKeepalive {}

impl Drop for MapKeepalive {
    fn drop(&mut self) {
        // SAFETY: `self.0` was retained in `map_bytes` and is released here,
        // symmetrically -- unmap first (the sync bracket), then the retained
        // reference.
        unsafe {
            edgefirst_tensor_ffi::ef_tensor_unmap(self.0);
            edgefirst_tensor_ffi::ef_tensor_free(self.0);
        }
    }
}

/// Lighter than the static backend's. `static_backend.rs`'s `Debug` prints
/// every field of the boxed `Tensor<T>` (`storage`, `format`, `chroma`,
/// `row_stride`, `quantization`, `colorimetry`, the CUDA handle, ...); this
/// one prints only `dtype` and `shape`, because those are the only two facts
/// `libedgefirst_tensor.so` exposes a *reader* for today (`ef_tensor_dtype`,
/// `ef_tensor_shape`/`ef_tensor_ndim`). Byte parity with the static backend
/// is not reachable without a new `ef_tensor_*` getter -- `quantization` in
/// particular has only a pre-alloc builder setter, no runtime reader -- so
/// matching it is a future primitive, not an impl choice made here. Safe to
/// reduce fidelity like this because nothing in the codebase parses or
/// compares `TensorDyn`'s `Debug` output; it exists only so
/// `detect::ProtoData`'s `#[derive(Debug)]` (unconditionally compiled, real
/// cross-backend surface `decoder` constructs) has something to call.
impl fmt::Debug for TensorDyn {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("TensorDyn")
            .field("dtype", &self.dtype())
            .field("shape", &self.shape())
            .finish()
    }
}

#[cfg(any(target_os = "macos", target_os = "ios"))]
#[link(name = "IOSurface", kind = "framework")]
extern "C" {
    fn IOSurfaceGetWidth(surface: *mut std::ffi::c_void) -> usize;
    fn IOSurfaceGetHeight(surface: *mut std::ffi::c_void) -> usize;
    fn IOSurfaceGetID(surface: *mut std::ffi::c_void) -> u32;
}
