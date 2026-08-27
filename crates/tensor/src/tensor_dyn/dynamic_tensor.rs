// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! `dynamic`'s `Tensor<T>` — a typed lens over [`TensorDyn`], not a second
//! storage-owning type.
//!
//! `static`'s `Tensor<T>` (in `crate::lib` proper) owns real per-backend
//! storage; `TensorDyn` there is an eleven-variant enum boxing one. Under
//! `dynamic` there is no such enum -- `TensorDyn` already *is* the one
//! handle -- so this `Tensor<T>` carries no storage of its own either. It
//! exists only to attach a compile-time-known element type to a `TensorDyn`
//! value, and every method here either forwards straight to the identical
//! method [`TensorDyn`] (`dynamic_backend.rs`) already implements, or drives
//! `edgefirst-tensor-ffi` directly for the handful of operations that need
//! `T::DTYPE` (allocation, import) or a typed view (`map_with`).
//!
//! # `#[repr(transparent)]` and why the cast in `lens.rs` is sound here
//!
//! This struct has exactly one field with nonzero size -- `inner:
//! TensorDyn` -- plus a zero-sized `PhantomData` marker, so
//! `#[repr(transparent)]` gives it the identical layout of `TensorDyn`
//! itself, whatever `TensorDyn`'s own fields happen to be (see
//! `dynamic_backend.rs`'s module docs: it is a handle plus two cached
//! facts, not a bare pointer). That is what makes `&TensorDyn as *const
//! Tensor<T>` sound in `lens.rs`'s `dynamic` `as_typed`/`as_typed_mut` --
//! contrast `static`, where `TensorDyn` is a tagged-union `enum` with
//! eleven differently-typed payloads and the same cast would be UB (see
//! `lens.rs`'s `static_lens` module docs, which uses `Any::downcast_ref`
//! instead for exactly that reason). Same source-level idea -- "borrow the
//! erased handle as a typed one" -- different validity argument per
//! backend, because the two `TensorDyn`s have different actual layouts.
use std::fmt;
use std::marker::PhantomData;

use half::f16;

use crate::lens::Element;
use crate::{
    BufferIdentity, Colorimetry, CpuAccess, DType, Error, IntegerType, PixelFormat, Quantization,
    Region, Result, TensorDyn, TensorMapTrait, TensorMemory, TensorTrait, ViewOrigin,
};

/// The dynamic backend's `Tensor<T>`. See the module docs.
#[repr(transparent)]
pub struct Tensor<T: Element> {
    inner: TensorDyn,
    // `fn() -> T`, not `T`: this tensor does not own a `T` (it owns a
    // type-erased handle `T::DTYPE` describes), so the marker should not
    // tie `Tensor<T>`'s variance or auto-trait derivation to `T`'s own --
    // `Element` already requires `Send + Sync + 'static` directly, so this
    // costs nothing and avoids implying a relationship that isn't there.
    _marker: PhantomData<fn() -> T>,
}

impl<T: Element> Tensor<T> {
    fn from_inner(inner: TensorDyn) -> Self {
        Tensor {
            inner,
            _marker: PhantomData,
        }
    }

    /// Discard the typed lens, yielding the type-erased handle it wraps.
    ///
    /// The inverse of [`from_inner`](Self::from_inner), for the one caller
    /// that builds through a typed constructor but must hand back a
    /// `TensorDyn`: `TensorDyn::from_pbo_import` (`dynamic_backend.rs`),
    /// whose per-dtype dispatch needs `T` only to reach
    /// [`Self::from_pbo`]. Free -- this type is `#[repr(transparent)]` over
    /// exactly this field.
    pub(crate) fn into_inner(self) -> TensorDyn {
        self.inner
    }

    /// Create a new tensor with the given shape, optional memory backing,
    /// and optional name. Same signature as `static`'s inherent `new` (the
    /// wider constructor `TensorTrait::new` delegates to) -- kept identical
    /// so shared call sites like [`crate::is_dma_available`] need no
    /// backend-specific branch.
    pub fn new(shape: &[usize], memory: Option<TensorMemory>, name: Option<&str>) -> Result<Self> {
        TensorDyn::new(shape, T::DTYPE, memory, name).map(Self::from_inner)
    }

    /// Import an existing buffer as a tensor, taking ownership of its file
    /// descriptor. No bytes are copied.
    #[cfg(unix)]
    pub fn from_fd(fd: std::os::fd::OwnedFd, shape: &[usize], name: Option<&str>) -> Result<Self> {
        TensorDyn::from_fd(fd, shape, T::DTYPE, name).map(Self::from_inner)
    }

    /// Construct a tensor from a row-major element slice + shape. Allocates
    /// a new buffer (`TensorMemory::Mem`) and memcpys the contents --
    /// derived over [`Self::new`] plus [`TensorTrait::map`], identical
    /// logic to `static`'s `Tensor::from_slice` (`lib.rs`); no primitive
    /// needed beyond what construction and mapping already cover.
    ///
    /// # Errors
    ///
    /// - [`Error::InvalidShape`] if `values.len() != shape.iter().product()`.
    /// - Propagates any allocation error from [`Self::new`].
    pub fn from_slice(values: &[T], shape: &[usize]) -> Result<Self>
    where
        T: Copy,
    {
        let expected: usize = shape.iter().product();
        if values.len() != expected {
            return Err(Error::InvalidShape(format!(
                "from_slice: values.len()={} but shape product={expected} (shape={shape:?})",
                values.len()
            )));
        }
        let t = Self::new(shape, Some(TensorMemory::Mem), None)?;
        {
            let mut m = TensorTrait::map(&t)?;
            m.as_mut_slice().copy_from_slice(values);
        }
        Ok(t)
    }

    /// Construct a tensor from a 3-D ndarray view. Respects strides -- one
    /// copy in all cases; contiguous views take a memcpy fast path. Derived
    /// the same way as [`Self::from_slice`] -- see its docs.
    ///
    /// Only available when the `ndarray` feature is enabled.
    #[cfg(feature = "ndarray")]
    pub fn from_arrayview3(view: ndarray::ArrayView3<'_, T>) -> Result<Self>
    where
        T: Copy,
    {
        let (h, w, c) = view.dim();
        let t = Self::new(&[h, w, c], Some(TensorMemory::Mem), None)?;
        {
            let mut m = TensorTrait::map(&t)?;
            let dst = m.as_mut_slice();
            if let Some(src) = view.as_slice() {
                dst.copy_from_slice(src);
            } else {
                for (d, &s) in dst.iter_mut().zip(view.iter()) {
                    *d = s;
                }
            }
        }
        Ok(t)
    }

    /// Runtime element type discriminant. Always `T::DTYPE` for a live
    /// lens, but sourced from the handle (`ef_tensor_dtype`) rather than
    /// the constant, so a lens opened by [`TensorDyn::as_typed`] reports
    /// what the handle itself carries.
    pub fn dtype(&self) -> DType {
        self.inner.dtype()
    }

    /// Whether this semi-planar tensor was assembled from separate
    /// luma/chroma *allocations* (`static`'s `from_planes`), as opposed to
    /// one contiguous combined-plane buffer.
    ///
    /// **History, because the previous answer here was wrong and is worth
    /// understanding.** Until task 17, this unconditionally returned
    /// `false`, on the reasoning that [`Self::from_planes`] -- the only
    /// constructor that could ever make it `true` -- itself always returned
    /// [`Error::NotImplemented`], so `false` was the correct answer for
    /// every tensor this backend could actually produce. Task 15 then wired
    /// a real `from_planes` (driving `ef_tensor_from_planes`), and nobody
    /// revisited this method's premise: the moment `from_planes` started
    /// succeeding, `is_multiplane()` started lying about the exact tensors
    /// it produces -- a live caller (`edgefirst-image`'s `import_image`)
    /// calls `Tensor::<u8>::from_planes` directly, and every downstream
    /// consumer that branches on `is_multiplane()`/`chroma()` (CPU convert,
    /// GL/G2D DMA import) would read a genuinely two-fd tensor as if it
    /// were one contiguous buffer. See task 17's report for the severity of
    /// that: not a proxy mismatch, a wrong-buffer read.
    ///
    /// Now driven by [`TensorDyn::multiplane_chroma`], real Rust-side state
    /// [`Self::from_planes`] populates from the chroma handle it consumes
    /// -- see that field's doc comment for why no `ef_tensor_*` primitive
    /// can answer this after the fact instead.
    ///
    /// **Residual risk, not closed by this fix, and not the old one.** The
    /// shadow is captured by *this crate's* `Tensor::from_planes` at the
    /// moment it calls `ef_tensor_from_planes` -- it is state this specific
    /// Rust value carries, not something the C handle itself carries. A
    /// caller that reaches a genuinely multiplane handle by any *other*
    /// path -- calling the exported `ef_tensor_from_planes` directly (it is
    /// `#[no_mangle] pub extern "C"`, reachable from any C caller, or from
    /// another Rust crate going through raw FFI rather than this crate's
    /// own wrapper) and then wrapping the result via `TensorDyn::from_raw`
    /// -- gets a `TensorDyn` whose `multiplane_chroma` was never populated,
    /// so `is_multiplane()` reports `false` for a tensor that is genuinely
    /// two allocations. **What would close it:** an `ef_tensor_*` primitive
    /// the *producer* side can answer from the handle alone (e.g. whether
    /// plane 0's and plane 1's native handles differ) rather than something
    /// only the constructing Rust value remembers -- `ef_tensor_plane_at`
    /// cannot serve this today, per task 17's report. No caller in this
    /// workspace takes this path currently (the only `ef_tensor_from_planes`
    /// call site is this file's own `Tensor::from_planes`, below), so this
    /// is unreached today, not theoretical-and-impossible.
    ///
    /// This replaces, rather than restates, the pre-task-17 caveat about a
    /// raw two-fd tensor built via `ef_tensor_builder_add_plane` called
    /// twice: task 16 closed that path outright (`ef_tensor_builder_wrap`
    /// now rejects a second plane with `ENOTSUP`), so restating it here
    /// would describe a hazard that no longer exists.
    pub fn is_multiplane(&self) -> bool {
        self.inner.multiplane_chroma.is_some()
    }

    /// The linked chroma tensor, if this is a `static`-style multiplane
    /// tensor built via [`Self::from_planes`]. See [`Self::is_multiplane`]
    /// for why this can now be genuinely `Some` under `dynamic`.
    pub fn chroma(&self) -> Option<&Self> {
        // `as_typed` cannot fail here: `from_planes` only ever stores a
        // shadow it built from a `chroma: Self` argument of this exact `T`
        // (see its own doc comment), so the dtype always matches.
        self.inner
            .multiplane_chroma
            .as_deref()
            .and_then(|td| td.as_typed::<T>())
    }

    /// See [`Self::chroma`]: mutable form.
    pub fn chroma_mut(&mut self) -> Option<&mut Self> {
        self.inner
            .multiplane_chroma
            .as_deref_mut()
            .and_then(|td| td.as_typed_mut::<T>())
    }

    // `as_dma` is deliberately ABSENT on this backend, where the static one
    // has it (`lib.rs`). It used to exist here returning `None`
    // unconditionally, on the reasoning that `DmaTensor<T>` is
    // `static`-backend-internal storage this backend never constructs, that
    // the return type matched, and that callers already handle `None`.
    //
    // Every part of that was true and the conclusion was still wrong. The
    // callers handled `None` by *declining*: `crates/image`'s EGLImage
    // import, its Path-B R8 import, `import_buffer_packed`, and all four
    // G2D sites refused a genuinely DMA-backed tensor and fell back to a
    // slower path. Nothing failed, nothing logged, and the suite stayed
    // green because both backends still produced correct pixels -- so a
    // user on i.MX lost the zero-copy hardware path with nothing reporting
    // it. See task P2b's review, F3.
    //
    // Every one of those callers wanted exactly one thing from the
    // `DmaTensor`: its `.fd`. They now use [`Self::dmabuf`], which both
    // backends implement with the same signature. With no caller left, a
    // stub that always answers "not DMA-backed" is a trap for the next one,
    // so it is gone: a future `dynamic` caller of `as_dma` gets a compile
    // error naming the problem instead of a silent, plausible `None`.

    /// Create from separate Y and UV planes (multiplane NV12/NV16). Same
    /// signature as `static`'s `Tensor::from_planes` (`lib.rs`).
    ///
    /// Drives `ef_tensor_from_planes`, which resolves the combination
    /// through the producer side's real `static::Tensor::from_planes` --
    /// the NV12/NV16 combined-plane geometry lives there, not here.
    /// **Consumes `luma` and `chroma`** (both `TensorDyn::into_raw`'d
    /// before the call), matching `static`'s by-value signature exactly:
    /// neither is usable afterward, on any outcome.
    ///
    /// # Errors
    ///
    /// [`Error::InvalidArgument`] if the underlying call fails -- either
    /// `Tensor::from_planes`'s own validation (format/shape mismatch; see
    /// its doc comment in `lib.rs`), or one of `ef_tensor_from_planes`'s own
    /// preconditions (see its Doxygen). A dtype mismatch between `luma` and
    /// `chroma` is unreachable through this typed entry point -- both are
    /// already the same `T` at compile time, unlike the type-erased C ABI
    /// this drives.
    ///
    /// **Known residual risk, not silently hidden**: `ef_tensor_from_planes`
    /// does *not* consume `luma`/`chroma` when it rejects them before ever
    /// reaching `Tensor::from_planes` (an outstanding `ef_tensor_retain` or
    /// `ef_tensor_map` on either handle -- reachable from safe Rust here via
    /// e.g. `tensor.map_with(access)` leaving a live `'static` view while
    /// `tensor` itself is still moved into this call). This wrapper cannot
    /// distinguish that case from an ordinary consumed failure using only
    /// the `NULL` return (both look identical from here), so on **any**
    /// failure it does not attempt to reclaim `luma`/`chroma` itself --
    /// guessing wrong in either direction would be a double-free or a
    /// use-after-free. In the ordinary (consumed) failure case this is
    /// exactly correct (the C library already freed them). In the rare
    /// non-consuming rejection case, the two underlying allocations are
    /// intentionally leaked rather than risking UB -- a documented trade-off,
    /// not an oversight; leaking under a caller-created precondition
    /// violation is a bounded, honest cost next to a double-free.
    ///
    /// **`dynamic`-only: preserves a chroma shadow before consuming
    /// `chroma`.** No `ef_tensor_*` primitive can recover a genuinely
    /// multiplane handle's separate chroma fd after this call returns (see
    /// [`TensorDyn::multiplane_chroma`]'s doc comment), so this captures an
    /// independent `dup`'d handle onto `chroma`'s own fd *before* it is
    /// consumed below, and stashes it on the result -- what makes
    /// [`Self::is_multiplane`]/[`Self::chroma`]/[`Self::chroma_mut`]
    /// honest afterward for the real, DMA-backed multiplane import every
    /// caller in this workspace actually builds (`edgefirst-image`'s
    /// `import_image`).
    ///
    /// If `chroma` is not fd-backed (e.g. `TensorMemory::Mem`, the shape a
    /// unit test unrelated to real hardware might use to exercise the
    /// combine logic in isolation) the shadow cannot be built at all --
    /// there is no fd to `dup`. Rather than fail an otherwise-valid combine
    /// over a case no real caller hits, this degrades to the pre-task-17
    /// behavior (`is_multiplane()` reports `false`) for that one tensor and
    /// logs a warning so the gap stays observable rather than silent.
    pub fn from_planes(luma: Self, chroma: Self, format: PixelFormat) -> Result<Self> {
        let c_format = std::ffi::CString::new(format.as_str()).map_err(|e| {
            Error::InvalidArgument(format!("pixel format string contains a NUL: {e}"))
        })?;
        #[cfg(unix)]
        let chroma_shadow = match chroma.inner.shadow_multiplane_chroma() {
            Ok(shadow) => Some(shadow),
            Err(e) => {
                log::warn!(
                    "Tensor::from_planes: could not preserve a chroma shadow ({e}) -- \
                     is_multiplane()/chroma() will report no chroma for this tensor, matching \
                     the behavior before task 17's fix; see TensorDyn::multiplane_chroma's doc \
                     comment"
                );
                None
            }
        };
        let luma_raw = TensorDyn::from(luma).into_raw();
        let chroma_raw = TensorDyn::from(chroma).into_raw();
        // SAFETY: both raw pointers are live, own-mint handles this
        // backend minted; ownership was just forgotten above, matching
        // `ef_tensor_from_planes`'s "consumes both inputs" contract.
        let handle = unsafe {
            edgefirst_tensor_ffi::ef_tensor_from_planes(luma_raw, chroma_raw, c_format.as_ptr())
        };
        match std::ptr::NonNull::new(handle) {
            Some(h) => {
                // SAFETY: `h` is a live handle `ef_tensor_from_planes`
                // returned as the caller-owned result.
                #[allow(unused_mut)]
                let mut inner = unsafe { TensorDyn::from_raw(h.as_ptr()) };
                #[cfg(unix)]
                {
                    inner.multiplane_chroma = chroma_shadow.map(Box::new);
                }
                Ok(Self::from_inner(inner))
            }
            None => Err(Error::InvalidArgument(format!(
                "ef_tensor_from_planes failed: {} -- see this method's doc comment for the \
                 known non-consuming-rejection case",
                crate::tensor_dyn::ffi_last_error()
            ))),
        }
    }

    /// Pixel format, if this tensor is an image (`None` otherwise).
    pub fn format(&self) -> Option<PixelFormat> {
        self.inner.format()
    }

    /// Image width in pixels (`None` if not an image tensor).
    pub fn width(&self) -> Option<usize> {
        self.inner.width()
    }

    /// Image height in pixels (`None` if not an image tensor).
    pub fn height(&self) -> Option<usize> {
        self.inner.height()
    }

    /// True if `self` and `other` reference the same underlying buffer.
    pub fn aliases(&self, other: &Self) -> bool {
        self.inner.aliases(&other.inner)
    }

    /// Colorimetry metadata (`None` = undefined; never auto-filled).
    pub fn colorimetry(&self) -> Option<Colorimetry> {
        self.inner.colorimetry()
    }

    /// Attach/clear colorimetry metadata.
    pub fn set_colorimetry(&mut self, c: Option<Colorimetry>) {
        self.inner.set_colorimetry(c)
    }

    /// Effective row stride: the padded byte pitch a stride-aware CPU
    /// reader must honor.
    pub fn effective_row_stride(&self) -> Option<usize> {
        self.inner.effective_row_stride()
    }

    /// The *recorded* row stride in bytes; `None` when the tensor is
    /// tightly packed. Deliberately distinct from
    /// [`Self::effective_row_stride`], which substitutes a computed pitch
    /// when nothing is recorded -- see [`TensorDyn::row_stride`]'s own doc
    /// comment for why the difference is load-bearing for the
    /// cross-package descriptor.
    pub fn row_stride(&self) -> Option<usize> {
        self.inner.row_stride()
    }

    /// The parent-image snapshot if this tensor is a `view`/`batch`
    /// sub-region; `None` for a whole tensor.
    pub fn view_origin(&self) -> Option<ViewOrigin> {
        self.inner.view_origin()
    }

    /// Byte offset within the DMA-BUF where image data starts (`None` = 0).
    /// Forwards to [`TensorDyn::plane_offset`], which drives the real
    /// `ef_tensor_plane_offset` primitive.
    ///
    /// **Not** plane 0's `ef_tensor_plane_at` offset -- that is
    /// `plane_table`'s intra-buffer layout, always 0 for plane 0 by
    /// construction, a different quantity from this field entirely; an
    /// earlier version of this method read that one by mistake, which
    /// always reported `Some(0)` or `None` regardless of what
    /// [`Self::set_plane_offset`] had actually been called with.
    pub fn plane_offset(&self) -> Option<usize> {
        self.inner.plane_offset()
    }

    /// Set the row stride, format-validated. Forwards to
    /// [`TensorDyn::set_row_stride`], which drives the real
    /// `ef_tensor_set_row_stride` primitive.
    pub fn set_row_stride(&mut self, stride: usize) -> Result<()> {
        self.inner.set_row_stride(stride)
    }

    /// Set the row stride without format validation. Same signature as
    /// `static`'s `Tensor::set_row_stride_unchecked` (`lib.rs`), which has
    /// no `TensorDyn`-level counterpart there either (it is a `Tensor<T>`-
    /// only raw escape hatch for sub-tensors without format metadata) --
    /// forwards to [`TensorDyn::set_row_stride_unchecked`], which drives
    /// the real `ef_tensor_set_row_stride_unchecked` primitive (task 17).
    /// This used to always panic, because no primitive backed it; that
    /// gap mattered in practice, not just in principle -- it is exactly
    /// what a multiplane chroma sub-tensor needs (see
    /// [`Self::chroma_mut`]/`from_planes`'s doc comments), and
    /// `edgefirst-image`'s `import_image` calls it on one.
    pub fn set_row_stride_unchecked(&mut self, stride: usize) {
        self.inner.set_row_stride_unchecked(stride)
    }

    /// Set the byte offset within the DMA-BUF where image data starts.
    /// Forwards to [`TensorDyn::set_plane_offset`], which drives the real
    /// `ef_tensor_set_plane_offset` primitive.
    pub fn set_plane_offset(&mut self, offset: usize) {
        self.inner.set_plane_offset(offset)
    }

    /// Borrow the DMA-BUF file descriptor backing this tensor.
    ///
    /// Derived from plane 0's native handle (`ef_tensor_plane_at`), the
    /// same primitive [`TensorTrait::clone_fd`] dup's -- this borrows
    /// instead of duplicating, matching `static`'s `dmabuf()` contract
    /// (`lib.rs`) of a borrow scoped to `&self`.
    ///
    /// # Errors
    ///
    /// [`Error::NotImplemented`] if this tensor is not DMA-backed, has no
    /// plane 0, or that plane carries no native fd.
    #[cfg(target_os = "linux")]
    pub fn dmabuf(&self) -> Result<std::os::fd::BorrowedFd<'_>> {
        if self.inner.memory() != TensorMemory::DmaBuf {
            return Err(Error::NotImplemented(format!(
                "dmabuf requires DMA-backed tensor, got {:?}",
                self.inner.memory()
            )));
        }
        let plane = self
            .inner
            .plane0()
            .ok_or_else(|| Error::NotImplemented("dmabuf: this tensor has no plane 0".into()))?;
        if plane.handle < 0 {
            return Err(Error::NotImplemented(
                "dmabuf: this tensor has no native fd (not DMA-backed)".into(),
            ));
        }
        // SAFETY: `plane.handle` is a valid fd owned by this tensor (via the
        // live handle behind `self.inner`) for at least as long as `self`
        // is borrowed.
        Ok(unsafe { std::os::fd::BorrowedFd::borrow_raw(plane.handle as i32) })
    }

    /// Borrow the raw `IOSurfaceRef` backing this tensor (macOS/iOS).
    #[cfg(any(target_os = "macos", target_os = "ios"))]
    pub fn iosurface_ref(&self) -> Option<*mut std::ffi::c_void> {
        self.inner.iosurface_ref()
    }

    /// Wrap a live `IOSurfaceRef` as a typed tensor (macOS/iOS).
    ///
    /// Same signature as `static`'s `Tensor::from_iosurface` so identity
    /// wrap tests compile against both backends.
    ///
    /// # Safety
    ///
    /// `surface_ref` must be a valid live `IOSurfaceRef`. `shape` must
    /// match the IOSurface's pixel dimensions and chosen element type.
    #[cfg(any(target_os = "macos", target_os = "ios"))]
    pub unsafe fn from_iosurface(
        surface_ref: *mut std::ffi::c_void,
        shape: &[usize],
        name: Option<&str>,
    ) -> Result<Self> {
        // SAFETY: caller guarantees `surface_ref` is a live IOSurfaceRef.
        unsafe { TensorDyn::from_iosurface(surface_ref, shape, T::DTYPE, name) }
            .map(Self::from_inner)
    }

    /// Physical IOSurface dimensions in texels. `None` when not IOSurface-backed.
    #[cfg(any(target_os = "macos", target_os = "ios"))]
    pub fn iosurface_physical_dims(&self) -> Option<(usize, usize)> {
        self.inner.iosurface_physical_dims()
    }

    /// CUDA registration for this tensor, if any (set at creation via
    /// [`Self::set_cuda_handle`]).
    ///
    /// `CudaHandle` (`crate::cuda`) is already `static`/`dynamic`-agnostic
    /// in-process Rust state -- see that module's own doc comment ("The
    /// rest of this module ... stays available under `dynamic`") -- so
    /// unlike a genuine ABI gap this needed no new `ef_tensor_*` primitive
    /// at all, only somewhere on `TensorDyn` to hold the value
    /// (`TensorDyn::cuda`, `dynamic_backend.rs`), the same "state the ABI
    /// cannot answer" reasoning `multiplane_chroma` already documents.
    /// Real caller: `edgefirst-image`'s `gl/threaded.rs::register_pbo_cuda`
    /// (via [`Self::set_cuda_handle`]) and `edgefirst-codec`'s nvJPEG
    /// decoder (`jpeg/nvjpeg/mod.rs`, this getter) -- which, before this
    /// existed, unconditionally took the non-CUDA fallback path on
    /// `dynamic` even for a genuinely CUDA-registered PBO destination.
    pub fn cuda(&self) -> Option<&crate::cuda::CudaHandle> {
        self.inner.cuda.as_deref()
    }

    /// See [`Self::cuda`]. `None` when no handle is attached, same as
    /// `static`'s own `Tensor::cuda_map`.
    pub fn cuda_map(&self) -> Option<crate::cuda::CudaMap<'_>> {
        self.cuda()?.map()
    }

    /// Attach a CUDA handle (called by `ImageProcessor::create_image` after
    /// registering a PBO with `cudaGraphicsGLRegisterBuffer`). See
    /// [`Self::cuda`] for why this needed no new `ef_tensor_*` primitive.
    pub fn set_cuda_handle(&mut self, h: crate::cuda::CudaHandle) {
        self.inner.cuda = Some(Box::new(h));
    }

    /// Construct a tensor from a PBO tensor (for GL backends that allocate
    /// PBOs). See [`TensorDyn::pbo`](crate::TensorDyn)'s own doc comment
    /// (`dynamic_backend.rs`) for the full design: this mints a companion
    /// `ef_tensor_*` handle sized to match `pbo`'s shape (so format/stride/
    /// shape queries keep working the ordinary way), then stashes `pbo`
    /// itself alongside it for [`Self::as_pbo`] and CPU mapping to read
    /// back.
    ///
    /// **This is NOT metadata-only** (an earlier version of this comment
    /// claimed it was -- corrected in task 18's review, F32). The only
    /// `ef_tensor_*` primitive `TensorDyn::new` can drive for
    /// `TensorMemory::Mem` is `ef_tensor_builder_alloc`, which always
    /// allocates real host memory sized to the full shape it is given --
    /// there is no ABI primitive today that mints a handle carrying a shape
    /// without backing it byte for byte. So this genuinely allocates
    /// `pbo`'s full byte count as ordinary host RAM purely to carry
    /// metadata, for a buffer whose real data already lives on the GPU --
    /// for a 4K RGBA16F PBO, ~63.3 MiB. See
    /// `from_pbo_metadata_handle_allocation_cost_is_the_full_pbo_byte_count`
    /// (`tests/dynamic_primitives.rs`) for the measured proof and the
    /// follow-up primitive this should eventually replace it with.
    ///
    /// Returns `Result`, unlike `static`'s infallible `Tensor::from_pbo`
    /// (`lib.rs`): `static` only wraps an already-validated `PboTensor<T>`
    /// with no new allocation, but this backend must mint that companion
    /// handle through the real `ef_tensor_*` builder, which can fail --
    /// the same allocation-failure surface every other `Tensor::new` call
    /// already has. `edgefirst-image`'s own call sites (`gl/threaded.rs`)
    /// propagate this with `?`, same as any other fallible constructor.
    pub fn from_pbo(pbo: crate::PboTensor<T>) -> Result<Self> {
        let mut inner = TensorDyn::new(&pbo.shape, T::DTYPE, Some(TensorMemory::Mem), None)?;
        inner.pbo = Some(Box::new(pbo));
        Ok(Self::from_inner(inner))
    }

    /// Downcast to PBO tensor reference (for GL backends). `None` when this
    /// tensor is not PBO-backed.
    ///
    /// The stored `PboTensor`'s element type is **not** required to be `T`.
    /// `edgefirst-image` allocates a PBO as `u8` and hands it back as an
    /// `i8` tensor -- under `static` the by-value transmute of
    /// `Tensor<u8>` -> `Tensor<i8>` carries the whole storage with it, so
    /// `as_pbo` there really does find a `PboTensor<i8>`. Under `dynamic`
    /// the `PboTensor<u8>` sits behind a real `Any` vtable that no
    /// transmute of the enclosing `Tensor<T>` touches, so an exact
    /// `downcast_ref::<PboTensor<T>>` found nothing and this returned
    /// `None` for exactly the int8 GPU path it exists to serve.
    ///
    /// So the stored value is found first and *then* reinterpreted as
    /// `PboTensor<T>`, which is sound: `PboTensor<T>` holds its element
    /// type only in a `PhantomData` (name, shape, an `Arc<PboHandle>`, an
    /// identity and a byte offset besides), so every instantiation is
    /// layout-identical. The `size_of::<T>()` the reinterpreted reference
    /// then reports is `T`'s, which is the caller's own type and the one it
    /// means -- and it agrees with the stored type's width, because the
    /// only way the two differ at all is `TensorDyn::set_dtype`, which
    /// refuses a width change.
    pub fn as_pbo(&self) -> Option<&crate::PboTensor<T>> {
        let any = self.inner.pbo.as_ref()?;
        // Exact match first: the ordinary case, where nothing was retagged.
        if let Some(p) = any.downcast_ref::<crate::PboTensor<T>>() {
            return Some(p);
        }
        // Otherwise find whichever instantiation is really stored. At most
        // one arm can hit -- `downcast_ref` is an exact `TypeId` match -- so
        // the order is irrelevant.
        macro_rules! reinterpret_arm {
            ($t:ty) => {{
                // The layout fact the cast below rests on, enforced rather
                // than documented. `PboTensor<T>` holds its element type
                // only in a `PhantomData` today, so every instantiation is
                // layout-identical -- but that is exactly the kind of
                // invariant a future field addition breaks with no compile
                // error and no failing test. This makes it a build failure
                // at the moment someone changes the layout. Per-arm, not
                // once for the function, so it guards each individual cast;
                // and evaluated per monomorphization, so it covers whichever
                // `T` a caller actually instantiates.
                //
                // It fires at **build**, not at `cargo check` -- an inline
                // `const` in a generic function is evaluated at codegen, and
                // `check` does not codegen. So a clean `cargo check` is not
                // evidence this holds; `make test-tensor-dynamic` (which
                // builds) is. Verified by adding an inline `Option<T>` field
                // to `PboTensor` and watching four `error[E0080]: evaluation
                // panicked` with this message.
                const {
                    assert!(
                        std::mem::size_of::<crate::PboTensor<$t>>()
                            == std::mem::size_of::<crate::PboTensor<T>>(),
                        "PboTensor is no longer layout-identical across element types; \
                         Tensor::as_pbo's reinterpretation is unsound -- see its doc comment"
                    );
                    assert!(
                        std::mem::align_of::<crate::PboTensor<$t>>()
                            == std::mem::align_of::<crate::PboTensor<T>>(),
                        "PboTensor's alignment now varies by element type; \
                         Tensor::as_pbo's reinterpretation is unsound -- see its doc comment"
                    );
                }
                any.downcast_ref::<crate::PboTensor<$t>>().map(|p| {
                    // SAFETY: layout-identical across element types, asserted
                    // just above rather than assumed; `p` is borrowed from
                    // `self`, and the result carries the same lifetime.
                    unsafe { &*(p as *const crate::PboTensor<$t> as *const crate::PboTensor<T>) }
                })
            }};
        }
        None.or_else(|| reinterpret_arm!(u8))
            .or_else(|| reinterpret_arm!(i8))
            .or_else(|| reinterpret_arm!(u16))
            .or_else(|| reinterpret_arm!(i16))
            .or_else(|| reinterpret_arm!(half::f16))
            .or_else(|| reinterpret_arm!(u32))
            .or_else(|| reinterpret_arm!(i32))
            .or_else(|| reinterpret_arm!(f32))
            .or_else(|| reinterpret_arm!(u64))
            .or_else(|| reinterpret_arm!(i64))
            .or_else(|| reinterpret_arm!(f64))
    }

    /// Allocate an image tensor with the given geometry, memory backing,
    /// and CPU access declaration. Same signature as `static`'s
    /// `Tensor::image` (`lib.rs`), which real production code calls
    /// (`edgefirst-codec`'s V4L2 hardware JPEG decoder,
    /// `jpeg/v4l2/mod.rs`) to allocate its destination tensor.
    ///
    /// Forwards to [`TensorDyn::image`], which drives the real
    /// `ef_tensor_image_alloc` primitive -- the platform image geometry
    /// (macOS IOSurface tiling, Android AHardwareBuffer, Linux DMA-BUF
    /// 64-byte pitch alignment, odd-dimension chroma handling, `DmaBuf` →
    /// `Mem` fallback) lives in `libedgefirst_tensor.so`'s `static::Tensor::image`,
    /// not here; this does not reimplement or approximate any of it.
    pub fn image(
        width: usize,
        height: usize,
        format: PixelFormat,
        memory: Option<TensorMemory>,
        access: CpuAccess,
    ) -> Result<Self> {
        TensorDyn::image(width, height, format, T::DTYPE, memory, access).map(Self::from_inner)
    }

    /// See [`Self::image`]: same primitive, full-featured request form.
    /// Forwards to [`TensorDyn::image_desc`].
    pub fn image_desc(desc: &crate::ImageDesc) -> Result<Self> {
        TensorDyn::image_desc(desc).map(Self::from_inner)
    }

    /// See [`Self::image`]: same primitive, externally-strided form.
    /// Forwards to [`TensorDyn::image_with_stride`]. Argument order matches
    /// `static`'s `Tensor::image_with_stride` (`lib.rs`) exactly --
    /// `row_stride_bytes` before `memory`/`access` -- which real callers
    /// (`edgefirst-image`'s `gl/dma_import.rs`, `lib.rs`) already call
    /// positionally in that order.
    pub fn image_with_stride(
        width: usize,
        height: usize,
        format: PixelFormat,
        row_stride_bytes: usize,
        memory: Option<TensorMemory>,
        access: CpuAccess,
    ) -> Result<Self> {
        TensorDyn::image_with_stride(
            width,
            height,
            format,
            T::DTYPE,
            row_stride_bytes,
            memory,
            access,
        )
        .map(Self::from_inner)
    }

    /// Set this tensor's pixel format. Same signature as `static`'s
    /// `Tensor::set_format` (`lib.rs`), which real production code (e.g.
    /// `edgefirst-image`'s `gl/threaded.rs::create_pbo_image`, right after
    /// [`Self::from_pbo`]) calls directly on the typed `Tensor<T>` --
    /// missing here (only `TensorDyn::set_format`, `dynamic_backend.rs`,
    /// existed) until `image-capi` was first built against `dynamic` with
    /// `opengl` enabled surfaced the gap.
    ///
    /// Forwards to [`TensorDyn::set_format`], which drives the real
    /// `ef_tensor_set_format` primitive.
    pub fn set_format(&mut self, format: PixelFormat) -> Result<()> {
        self.inner.set_format(format)
    }

    /// Set this tensor's logical dimensions and pixel format to a decoded
    /// image, reusing the existing allocation. Same signature as `static`'s
    /// `Tensor::configure_image` (`lib.rs`), which real production code
    /// calls (`edgefirst-codec`'s JPEG decode-into-pool path) to reconfigure
    /// a pool-allocated destination tensor's shape/format/row-stride before
    /// writing into it.
    ///
    /// Forwards to [`TensorDyn::configure_image`], which drives the real
    /// `ef_tensor_configure_image` primitive -- the pool-reuse
    /// stride-preservation and alignment rules live in
    /// `libedgefirst_tensor.so`'s `static::Tensor::configure_image`, not
    /// here.
    pub fn configure_image(
        &mut self,
        width: usize,
        height: usize,
        format: PixelFormat,
    ) -> Result<()> {
        self.inner.configure_image(width, height, format)
    }

    /// Pin a stable host address for this tensor's data, type-erased to
    /// raw bytes -- same contract as [`TensorTrait::map_with`]'s pin, just
    /// without the element-typed view wrapped around it. `static`'s
    /// `Tensor<T>` exposes this as an inherent method too (`lib.rs`), for
    /// callers (e.g. `edgefirst-image`'s `lib.rs`) that want the address
    /// without going through `TensorMapTrait`.
    pub fn pin_host<'a>(&self, access: CpuAccess) -> Result<crate::pin::HostPin<'a>> {
        self.inner.pin_host(access)
    }

    /// Borrow a rectangular spatial sub-region as a zero-copy view. Same
    /// signature as `static`'s `Tensor::view` (`lib.rs`). Forwards to
    /// [`TensorDyn::view`], which drives the real `ef_tensor_view_region`
    /// primitive.
    pub fn view(&self, region: Region) -> Result<Self> {
        self.inner.view(region).map(Self::from_inner)
    }
}

// Quantization accessors — type-gated to integer element types via the
// sealed `IntegerType` trait, same as `static`'s equivalent block in
// `lib.rs` (see `[[feedback_tensor_quantization_type_gated]]`: this must
// stay a compile-time rejection for `Tensor<f32>`, not a runtime one).
// Each method here forwards straight to `TensorDyn`'s real implementation
// (`dynamic_backend.rs`), which drives the `ef_tensor_quantization_*`
// primitives and owns the cache `quantization()`'s borrow is served from --
// see that struct's `quantization_cache` field doc comment for why the
// cache is sound under concurrent access, the question this block's
// previous stub-era comment (before the primitive existed) had no answer
// for.
impl<T: Element + IntegerType> Tensor<T> {
    /// Quantization metadata for this tensor.
    pub fn quantization(&self) -> Option<&Quantization> {
        self.inner.quantization()
    }

    /// Attach quantization metadata to this tensor.
    pub fn set_quantization(&mut self, q: Quantization) -> Result<()> {
        self.inner.set_quantization(q)
    }

    /// Builder-style variant of [`Self::set_quantization`].
    pub fn with_quantization(mut self, q: Quantization) -> Result<Self> {
        self.set_quantization(q)?;
        Ok(self)
    }

    /// Clear any quantization metadata on this tensor.
    pub fn clear_quantization(&mut self) {
        self.inner.clear_quantization()
    }
}

/// Generate the three downcast methods (ref, mut ref, owned) for one
/// element type on [`TensorDyn`] -- the `dynamic`-backend counterpart to
/// `static_backend.rs`'s own `downcast_methods!` macro (same names, same
/// signatures, so a caller compiled against either backend needs no
/// backend-specific branch). `static`'s version matches an enum variant;
/// this one is built entirely on [`TensorDyn::as_typed`]/`as_typed_mut`
/// (`lens.rs`), the existing dtype-checked `#[repr(transparent)]` lens
/// `dynamic` already uses for exactly this "reinterpret the erased handle
/// as a typed one" operation -- so unlike `PboTensor`/`CudaHandle` above,
/// this family needed no new `TensorDyn` state and no new `ef_tensor_*`
/// primitive, only these convenience wrappers. Found missing (`no method
/// named as_u8/as_f32/... found for struct TensorDyn`) when `image-capi`
/// was first built against `dynamic` with `opengl` enabled -- task 15/17's
/// own `--no-default-features` verification never compiled `opengl`, so
/// this gap was as invisible to them as the PBO/CUDA one (see task 9's
/// report, "The verification hole").
macro_rules! downcast_methods {
    ($ty:ty, $as_name:ident, $as_mut_name:ident, $into_name:ident) => {
        /// Returns a shared reference to the inner tensor if the type matches.
        pub fn $as_name(&self) -> Option<&Tensor<$ty>> {
            self.as_typed::<$ty>()
        }

        /// Returns a mutable reference to the inner tensor if the type matches.
        pub fn $as_mut_name(&mut self) -> Option<&mut Tensor<$ty>> {
            self.as_typed_mut::<$ty>()
        }

        /// Unwraps the inner tensor if the type matches, otherwise returns `self` as `Err`.
        /// The Err variant is necessarily large (returns the unconsumed TensorDyn).
        #[allow(clippy::result_large_err)]
        pub fn $into_name(self) -> std::result::Result<Tensor<$ty>, Self> {
            if self.dtype() == <$ty as Element>::DTYPE {
                Ok(Tensor::from_inner(self))
            } else {
                Err(self)
            }
        }
    };
}

impl TensorDyn {
    downcast_methods!(u8, as_u8, as_u8_mut, into_u8);
    downcast_methods!(i8, as_i8, as_i8_mut, into_i8);
    downcast_methods!(u16, as_u16, as_u16_mut, into_u16);
    downcast_methods!(i16, as_i16, as_i16_mut, into_i16);
    downcast_methods!(u32, as_u32, as_u32_mut, into_u32);
    downcast_methods!(i32, as_i32, as_i32_mut, into_i32);
    downcast_methods!(u64, as_u64, as_u64_mut, into_u64);
    downcast_methods!(i64, as_i64, as_i64_mut, into_i64);
    downcast_methods!(f16, as_f16, as_f16_mut, into_f16);
    downcast_methods!(f32, as_f32, as_f32_mut, into_f32);
    downcast_methods!(f64, as_f64, as_f64_mut, into_f64);
}

impl<T: Element> From<Tensor<T>> for TensorDyn {
    /// Erase the compile-time element type, re-asserting `T`'s dtype on the
    /// handle as it goes.
    ///
    /// `Tensor<T>` already *is* a `TensorDyn` plus a zero-sized marker (see
    /// the module docs), so the unwrap itself is free. The retag is not
    /// decoration: it repairs the invariant `Tensor<T>` exists to carry.
    ///
    /// Under `static`, `TensorDyn` is an enum over eleven `Tensor<T>`, so
    /// the dtype **is** the Rust type — `TensorDyn::from(t)` picks the
    /// variant from `T` and a layout-identical `transmute` between
    /// same-width `Tensor`s changes the reported dtype for free. Under
    /// `dynamic` the dtype lives in the C handle instead, so that same
    /// transmute changes a `PhantomData` and nothing else, and the erased
    /// tensor keeps reporting whatever dtype the handle was *minted* with.
    ///
    /// That is not hypothetical. `edgefirst-image` allocates a PBO or DMA
    /// buffer as `u8` and hands it back as `Tensor<i8>` by exactly that
    /// transmute (twice, in `crates/image/src/lib.rs`), because the int8
    /// shader applies an XOR 0x80 bias over the same bytes. Without the
    /// retag, `create_image(dtype="int8")` returned a tensor reporting
    /// `uint8` — a wrong answer rather than a refusal, and one that
    /// propagates silently into quantization and inference rather than
    /// stopping anything.
    ///
    /// Cost is one `ef_tensor_dtype` read on the common path, where the
    /// handle already agrees and [`TensorDyn::set_dtype`] returns early.
    ///
    /// A failure here can only mean `T` is a different **width** than the
    /// handle's dtype, which no layout-identical transmute can produce —
    /// `Tensor<u8>` to `Tensor<u32>` is not layout-identical in the sense
    /// this relies on. `From` cannot return an error, so that case is
    /// logged rather than swallowed, and the dtype mismatch it leaves
    /// behind is then caught by [`crate::TensorDyn::as_typed`], which
    /// checks the handle's dtype against `T` before handing back a lens.
    fn from(t: Tensor<T>) -> TensorDyn {
        let mut inner = t.inner;
        if let Err(e) = inner.set_dtype(T::DTYPE) {
            log::error!(
                "erasing a Tensor<{}> whose handle reports {:?}: {e}. The typed lens and its \
                 handle disagree on element width, which no layout-identical transmute can \
                 produce; the erased tensor keeps the handle's dtype.",
                std::any::type_name::<T>(),
                inner.dtype()
            );
        }
        inner
    }
}

impl<T: Element> fmt::Debug for Tensor<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Forward to `TensorDyn`'s own `Debug` rather than deriving: the
        // lens carries no fields of its own to print, and this keeps one
        // `Debug` implementation instead of two that could drift.
        fmt::Debug::fmt(&self.inner, f)
    }
}

impl<T: Element> TensorTrait<T> for Tensor<T> {
    fn new(shape: &[usize], name: Option<&str>) -> Result<Self>
    where
        Self: Sized,
    {
        Self::new(shape, None, name)
    }

    #[cfg(unix)]
    fn from_fd(fd: std::os::fd::OwnedFd, shape: &[usize], name: Option<&str>) -> Result<Self>
    where
        Self: Sized,
    {
        Self::from_fd(fd, shape, name)
    }

    #[cfg(unix)]
    fn clone_fd(&self) -> Result<std::os::fd::OwnedFd> {
        self.inner.clone_fd()
    }

    fn memory(&self) -> TensorMemory {
        self.inner.memory()
    }

    fn name(&self) -> String {
        self.inner.name()
    }

    fn shape(&self) -> &[usize] {
        self.inner.shape()
    }

    fn reshape(&mut self, shape: &[usize]) -> Result<()> {
        self.inner.reshape(shape)
    }

    /// Forwarded rather than left on `TensorTrait`'s default, whose body is
    /// `self.reshape(shape)` -- the strict rule under a name promising the
    /// capacity-based one. See the static backend's own override
    /// (`impl TensorTrait for Tensor<T>`, `lib.rs`) and task P2e.
    fn set_logical_shape(&mut self, shape: &[usize]) -> Result<()> {
        self.inner.set_logical_shape(shape)
    }

    /// The allocation's real byte count, forwarded to
    /// [`TensorDyn::capacity_bytes`] rather than left on
    /// [`TensorTrait::capacity_bytes`]'s "the logical size" default.
    ///
    /// The default is wrong for this backend, not merely imprecise: a
    /// pitch-aligned or pool-sized tensor really does have headroom past
    /// its shape, and reporting the tight size discards it. Concretely, a
    /// `Mem`-backed NV12 image at an odd width pads its rows to a 64-byte
    /// pitch -- so a `HOST` descriptor minted from it carried a `capacity`
    /// equal to its shape, and a consumer importing that alias could not
    /// `configure_image` back into memory the producer actually has. That
    /// is `host_import_inherits_the_producers_capacity_headroom`
    /// (`tests/protocol_roundtrip.rs`), which fails on its own
    /// *precondition* without this -- it was never run against this
    /// backend before task P2a, because `descriptor_pinned` did not exist
    /// here to run it with. There was also no way to override the default
    /// before P2a: no `ef_tensor_*` primitive reported an allocation's
    /// byte count until `ef_tensor_capacity_bytes`.
    fn capacity_bytes(&self) -> usize {
        self.inner.capacity_bytes()
    }

    /// One `ef_tensor_map` call for the pointer (the same
    /// `TensorDyn::map_pin` core `TensorDyn::map_bytes` shares), then an
    /// ordinary typed [`crate::view::HostView`] over the resulting window
    /// -- the slice operations a caller does with the mapped view
    /// (`chunks_mut`, `split_at_mut`, `iter`, indexing, ...) are plain
    /// `[T]` methods after that, not additional ABI calls.
    ///
    /// `byte_size_override` is `Some(pin.len())`, not `None`: `pin.len()`
    /// is the real, already stride-aware extent `ef_tensor_map` mapped
    /// (`self.inner.map_pin` -> `ef_tensor_plane_at`'s own geometry, task
    /// 17's `vt_plane_at` fix included), which for a padded row (e.g. a
    /// semi-planar image whose width is not a multiple of the 64-byte
    /// alignment `configure_image` picks) is *larger* than
    /// `self.inner.shape().iter().product()`, the tight logical size.
    /// `None` here made [`HostView::len_elems`] fall back to that tight
    /// product, silently truncating the mapped view to less than what the
    /// real handle actually maps -- exactly the "full padded buffer" the
    /// `static` backend's own `Tensor<T>::map_with` (`crate::lib`) exposes
    /// for the identical reason (see that function's own doc comment). Not
    /// reached by anything before task 9: no caller mapped a `dynamic`
    /// `Tensor<T>` over a semi-planar image whose stride exceeds its tight
    /// width until G3's host-memory JPEG decode exercised it for the first
    /// time.
    fn map_with<'a>(&self, access: CpuAccess) -> Result<crate::view::HostView<'a, T>>
    where
        T: 'a,
    {
        let pin = self.inner.map_pin(access)?;
        let byte_len = pin.len();
        Ok(crate::view::HostView::new(
            pin,
            self.inner.shape().to_vec(),
            Some(byte_len),
            access,
        ))
    }

    fn buffer_identity(&self) -> &BufferIdentity {
        self.inner.buffer_identity()
    }
}
