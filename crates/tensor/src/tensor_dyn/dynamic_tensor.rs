// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! `dynamic`'s `Tensor<T>`: a typed lens over [`TensorDyn`] that owns no
//! storage.
//!
//! It attaches a compile-time element type to a `TensorDyn` value. Each method
//! forwards to the same method on [`TensorDyn`] (`dynamic_backend.rs`), or
//! drives `edgefirst-tensor-ffi` directly for the operations that need
//! `T::DTYPE` (allocation, import) or a typed view (`map_with`).
//!
//! The struct is `#[repr(transparent)]` over `TensorDyn`: one `inner:
//! TensorDyn` field plus a zero-sized `PhantomData` marker.
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
    _marker: PhantomData<fn() -> T>,
}

impl<T: Element> Tensor<T> {
    fn from_inner(inner: TensorDyn) -> Self {
        Tensor {
            inner,
            _marker: PhantomData,
        }
    }

    /// Create a new tensor with the given shape, optional memory backing, and
    /// optional name.
    pub fn new(shape: &[usize], memory: Option<TensorMemory>, name: Option<&str>) -> Result<Self> {
        TensorDyn::new(shape, T::DTYPE, memory, name).map(Self::from_inner)
    }

    /// Import an existing buffer as a tensor, taking ownership of its file
    /// descriptor. No bytes are copied.
    #[cfg(unix)]
    pub fn from_fd(fd: std::os::fd::OwnedFd, shape: &[usize], name: Option<&str>) -> Result<Self> {
        TensorDyn::from_fd(fd, shape, T::DTYPE, name).map(Self::from_inner)
    }

    /// Construct a tensor from a row-major element slice + shape. Allocates a
    /// new buffer (`TensorMemory::Mem`) and copies the contents.
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

    /// Construct a tensor from a 3-D ndarray view. Respects strides: one copy
    /// in all cases; contiguous views take a memcpy fast path.
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

    /// Runtime element type discriminant, read from the handle
    /// (`ef_tensor_dtype`).
    pub fn dtype(&self) -> DType {
        self.inner.dtype()
    }

    /// Whether this semi-planar tensor was assembled from separate luma/chroma
    /// allocations (`from_planes`), as opposed to one contiguous combined-plane
    /// buffer. Read from the C handle (`ef_tensor_is_multiplane`).
    pub fn is_multiplane(&self) -> bool {
        self.inner.is_multiplane()
    }

    /// The chroma plane, if this is a two-allocation semi-planar tensor.
    /// Borrowed from a cached lens over the retained `ef_tensor_chroma` handle,
    /// valid for this parent's lifetime.
    pub fn chroma(&self) -> Option<&Self> {
        self.inner.chroma()?.as_typed::<T>()
    }

    /// Create from separate Y and UV planes (multiplane NV12/NV16). Drives
    /// `ef_tensor_from_planes`.
    ///
    /// Consumes `luma` and `chroma`: neither is usable afterward, on any
    /// outcome. When the library rejects the handles without consuming them (an
    /// outstanding `ef_tensor_retain` or `ef_tensor_map` on either), the two
    /// allocations are leaked.
    ///
    /// # Errors
    ///
    /// [`Error::InvalidArgument`] if the underlying call fails.
    pub fn from_planes(luma: Self, chroma: Self, format: PixelFormat) -> Result<Self> {
        let c_format = std::ffi::CString::new(format.as_str()).map_err(|e| {
            Error::InvalidArgument(format!("pixel format string contains a NUL: {e}"))
        })?;
        let luma_raw = TensorDyn::from(luma).into_raw();
        let chroma_raw = TensorDyn::from(chroma).into_raw();
        // SAFETY: both raw pointers are live handles this backend minted, and
        // their ownership passes to `ef_tensor_from_planes`.
        let handle = unsafe {
            edgefirst_tensor_ffi::ef_tensor_from_planes(luma_raw, chroma_raw, c_format.as_ptr())
        };
        match std::ptr::NonNull::new(handle) {
            Some(h) => {
                // SAFETY: `h` is a live handle `ef_tensor_from_planes`
                // returned as the caller-owned result.
                Ok(Self::from_inner(unsafe { TensorDyn::from_raw(h.as_ptr()) }))
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

    /// The CPU access declared for this tensor (see [`CpuAccess`]): set at
    /// allocation, possibly narrowed by [`Self::restrict_cpu_access`].
    pub fn cpu_access(&self) -> CpuAccess {
        self.inner.cpu_access()
    }

    /// Narrow the declared CPU access to the directions both it and `access`
    /// cover; never widens.
    pub fn restrict_cpu_access(&mut self, access: CpuAccess) {
        self.inner.restrict_cpu_access(access)
    }

    /// Colorimetry metadata. `None` = undefined; an image convert tags an untagged YUV destination.
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

    /// The recorded row stride in bytes; `None` when the tensor is tightly
    /// packed. Unlike [`Self::effective_row_stride`], it substitutes no
    /// computed pitch when none is recorded.
    pub fn row_stride(&self) -> Option<usize> {
        self.inner.row_stride()
    }

    /// The parent-image snapshot if this tensor is a `view`/`batch`
    /// sub-region; `None` for a whole tensor.
    pub fn view_origin(&self) -> Option<ViewOrigin> {
        self.inner.view_origin()
    }

    /// Whether this tensor addresses part of a larger buffer.
    pub fn is_view(&self) -> bool {
        self.inner.is_view()
    }

    /// Record that this tensor addresses part of a larger buffer.
    pub fn mark_view(&mut self) {
        self.inner.mark_view()
    }

    /// Byte offset within the DMA-BUF where image data starts (`None` = 0).
    /// Forwards to [`TensorDyn::plane_offset`].
    pub fn plane_offset(&self) -> Option<usize> {
        self.inner.plane_offset()
    }

    /// Set the row stride, format-validated. Forwards to
    /// [`TensorDyn::set_row_stride`].
    pub fn set_row_stride(&mut self, stride: usize) -> Result<()> {
        self.inner.set_row_stride(stride)
    }

    /// Set the row stride without format validation. Forwards to
    /// [`TensorDyn::set_row_stride_unchecked`].
    pub fn set_row_stride_unchecked(&mut self, stride: usize) {
        self.inner.set_row_stride_unchecked(stride)
    }

    /// Set the byte offset within the DMA-BUF where image data starts. Forwards
    /// to [`TensorDyn::set_plane_offset`].
    pub fn set_plane_offset(&mut self, offset: usize) {
        self.inner.set_plane_offset(offset)
    }

    /// Borrow the DMA-BUF file descriptor backing this tensor, read from plane
    /// 0's native handle.
    ///
    /// # Errors
    ///
    /// [`Error::NotImplemented`] if this tensor is not DMA-backed, has no
    /// plane 0, or that plane carries no native fd.
    #[cfg(target_os = "linux")]
    pub fn dmabuf(&self) -> Result<std::os::fd::BorrowedFd<'_>> {
        self.inner.dmabuf()
    }

    /// Borrow the raw `IOSurfaceRef` backing this tensor (macOS/iOS).
    #[cfg(any(target_os = "macos", target_os = "ios"))]
    pub fn iosurface_ref(&self) -> Option<*mut std::ffi::c_void> {
        self.inner.iosurface_ref()
    }

    /// Wrap a live `IOSurfaceRef` as a typed tensor (macOS/iOS).
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

    /// The borrowed `ID3D11Texture2D*` backing this tensor, valid while the
    /// tensor lives. `None` when the tensor is not texture-backed.
    #[cfg(target_os = "windows")]
    pub fn d3d11_texture(&self) -> Option<*mut std::ffi::c_void> {
        self.inner.d3d11_texture()
    }

    /// The texture geometry the HAL chose for this image: DXGI format, texel
    /// dimensions and the matching GL internal format. `None` when the tensor
    /// is not texture-backed.
    #[cfg(target_os = "windows")]
    pub fn d3d11_layout(&self) -> Option<crate::d3d11_layout::D3d11ImageLayout> {
        self.inner.d3d11_layout()
    }

    /// A duplicated NT handle the caller owns: open it on a D3D12 device,
    /// another D3D11 device, or CUDA, or duplicate it into another process.
    /// Closing it does not affect this tensor.
    #[cfg(target_os = "windows")]
    pub fn d3d11_shared_handle(&self) -> Result<std::os::windows::io::OwnedHandle> {
        self.inner.d3d11_shared_handle()
    }

    /// The fence handle plus value a GPU consumer waits on before reading this
    /// texture, or `None` when no GPU write has been recorded on it.
    #[cfg(target_os = "windows")]
    pub fn gpu_completion(&self) -> Result<Option<crate::d3d11::GpuCompletion>> {
        self.inner.gpu_completion()
    }

    /// The fence value of the newest GPU write recorded on this tensor, or 0
    /// when there is none: the `value` of `gpu_completion` without the
    /// duplicated fence handle. 0 for a tensor that is not a D3D11 texture,
    /// and on every platform other than Windows.
    pub fn gpu_write_value(&self) -> u64 {
        self.inner.gpu_write_value()
    }

    /// Record that GPU work writing this texture completes at `value` of the
    /// process device's fence. Monotonic: an older value never displaces a
    /// newer one.
    #[cfg(target_os = "windows")]
    pub fn set_gpu_write(&self, value: u64) -> Result<()> {
        self.inner.set_gpu_write(value)
    }

    /// Wrap an existing `ID3D11Texture2D` as a tensor. The texture is checked
    /// against the layout the HAL would have chosen for
    /// `format`/`width`/`height` and must live on the HAL's process device.
    /// This takes its own reference; ownership stays with the caller.
    ///
    /// # Safety
    ///
    /// `texture` must be null or a live `ID3D11Texture2D` created on the HAL
    /// device ([`crate::d3d11::device()`]).
    #[cfg(target_os = "windows")]
    pub unsafe fn from_d3d11_texture(
        texture: *mut std::ffi::c_void,
        width: usize,
        height: usize,
        format: PixelFormat,
        access: CpuAccess,
        name: Option<&str>,
    ) -> Result<Self> {
        // SAFETY: caller guarantees `texture` is null or a live texture on
        // the HAL device.
        unsafe {
            TensorDyn::from_d3d11_texture(texture, width, height, format, T::DTYPE, access, name)
        }
        .map(Self::from_inner)
    }

    /// Open a shared texture by its NT handle on the HAL's device. When
    /// `completion` is given, its fence value is waited for on the immediate
    /// context. Both handles stay owned by the caller: this duplicates what it
    /// keeps.
    ///
    /// # Safety
    ///
    /// `handle` must be an NT shared handle of a D3D11 texture, valid in this
    /// process, and `completion`'s handle a shared fence handle.
    #[cfg(target_os = "windows")]
    pub unsafe fn from_d3d11_shared_handle(
        handle: std::os::windows::io::RawHandle,
        width: usize,
        height: usize,
        format: PixelFormat,
        access: CpuAccess,
        completion: Option<(std::os::windows::io::RawHandle, u64)>,
        name: Option<&str>,
    ) -> Result<Self> {
        // SAFETY: caller guarantees the handles are valid in this process.
        unsafe {
            TensorDyn::from_d3d11_shared_handle(
                handle,
                width,
                height,
                format,
                T::DTYPE,
                access,
                completion,
                name,
            )
        }
        .map(Self::from_inner)
    }

    /// CUDA registration for this tensor. Always `None` on the dynamic lens,
    /// where the `CudaHandle` lives in the library. Use
    /// [`Self::is_cuda_attached`] or [`Self::cuda_map`] instead.
    #[deprecated(
        since = "0.32.0",
        note = "On the dynamic backend use is_cuda_attached() or cuda_map() — the CudaHandle is not borrowable through this lens."
    )]
    pub fn cuda(&self) -> Option<&crate::cuda::CudaHandle> {
        #[allow(deprecated)]
        {
            self.inner.cuda()
        }
    }

    /// Whether the library tensor carries a CUDA registration.
    pub fn is_cuda_attached(&self) -> bool {
        self.inner.is_cuda_attached()
    }

    /// Fast-fail CUDA map; `None` when the tensor carries no registration.
    pub fn cuda_map(&self) -> Option<crate::cuda::CudaMap<'_>> {
        self.inner.cuda_map()
    }

    /// Writable counterpart of [`Self::cuda_map`].
    pub fn cuda_map_mut(&self) -> Option<crate::cuda::CudaMap<'_>> {
        self.inner.cuda_map_mut()
    }

    /// Attach a CUDA handle via `ef_tensor_cuda_attach`.
    pub fn set_cuda_handle(&mut self, h: crate::cuda::CudaHandle) {
        self.inner.set_cuda_handle(h);
    }

    /// Construct a tensor from a PBO tensor (for GL backends that allocate
    /// PBOs).
    ///
    /// Drives `ef_tensor_wrap_pbo`: the GL buffer stays this process's, and the
    /// callback channel (an `EfClientState` plus the two op functions, see
    /// [`crate::PboTensor::into_client_parts`]) crosses into the library, which
    /// builds `TensorStorage::Pbo` storage over it. Allocates no host memory.
    ///
    /// # Errors
    /// Fails for a PBO sub-view (`view_offset != 0`) and for anything the
    /// library refuses (unknown dtype, a `size` smaller than the shape needs).
    pub fn from_pbo(pbo: crate::PboTensor<T>) -> Result<Self> {
        let parts = pbo.into_client_parts()?;
        // SAFETY: `parts.state.ctx` borrows the `Arc<PboHandle>` `parts` still
        // holds, and `ef_tensor_wrap_pbo` retains it before returning. The op
        // functions are this crate's own and uphold `PboOps`'s contract.
        let inner = unsafe {
            TensorDyn::from_client_pbo(
                parts.state,
                parts.map_fn,
                parts.unmap_fn,
                parts.buffer_id,
                parts.size,
                &parts.shape,
                T::DTYPE,
            )
        }?;
        drop(parts);
        Ok(Self::from_inner(inner))
    }

    /// The GL buffer name for a PBO-backed tensor; `None` otherwise. `T` does
    /// not gate the answer.
    pub fn pbo_id(&self) -> Option<u32> {
        self.inner.pbo_id()
    }

    /// Whether a PBO-backed tensor holds (or is establishing) a CPU mapping;
    /// `None` when the tensor is not PBO-backed.
    pub fn pbo_is_mapped(&self) -> Option<bool> {
        self.inner.pbo_is_mapped()
    }

    /// The C-ABI `PboOpsVtable` address backing this tensor; `None` when the
    /// tensor is not PBO-backed. Every `view()` of one PBO reports the same
    /// address. See [`TensorDyn::pbo_vtable_ptr`].
    pub fn pbo_vtable_ptr(&self) -> Option<*const std::ffi::c_void> {
        self.inner.pbo_vtable_ptr()
    }

    /// Allocate an image tensor with the given geometry, memory backing, and
    /// CPU access declaration. Forwards to [`TensorDyn::image`].
    pub fn image(
        width: usize,
        height: usize,
        format: PixelFormat,
        memory: Option<TensorMemory>,
        access: CpuAccess,
    ) -> Result<Self> {
        TensorDyn::image(width, height, format, T::DTYPE, memory, access).map(Self::from_inner)
    }

    /// Allocate an image tensor from a full [`crate::ImageDesc`] request.
    /// Forwards to [`TensorDyn::image_desc`].
    pub fn image_desc(desc: &crate::ImageDesc) -> Result<Self> {
        TensorDyn::image_desc(desc).map(Self::from_inner)
    }

    /// Allocate an image tensor with a caller-supplied row stride. Forwards to
    /// [`TensorDyn::image_with_stride`].
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

    /// Set this tensor's pixel format. Forwards to [`TensorDyn::set_format`].
    pub fn set_format(&mut self, format: PixelFormat) -> Result<()> {
        self.inner.set_format(format)
    }

    /// Set this tensor's logical dimensions and pixel format to a decoded
    /// image, reusing the existing allocation. Forwards to
    /// [`TensorDyn::configure_image`].
    pub fn configure_image(
        &mut self,
        width: usize,
        height: usize,
        format: PixelFormat,
    ) -> Result<()> {
        self.inner.configure_image(width, height, format)
    }

    /// Pin a stable host address for this tensor's data, type-erased to raw
    /// bytes.
    pub fn pin_host<'a>(&self, access: CpuAccess) -> Result<crate::pin::HostPin<'a>> {
        self.inner.pin_host(access)
    }

    /// Borrow a rectangular spatial sub-region as a zero-copy view. Forwards to
    /// [`TensorDyn::view`].
    pub fn view(&self, region: Region) -> Result<Self> {
        self.inner.view(region).map(Self::from_inner)
    }
}

// Quantization accessors, available for integer element types only
// (`IntegerType`). Each forwards to `TensorDyn`.
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

/// Generate the three downcast methods (ref, mut ref, owned) for one element
/// type on [`TensorDyn`], built on [`TensorDyn::as_typed`]/`as_typed_mut`
/// (`lens.rs`).
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
    /// Erase the compile-time element type, setting `T`'s dtype on the handle
    /// with [`TensorDyn::set_dtype`]. A `T` of a different width than the
    /// handle's dtype is logged, and the erased tensor keeps the handle's
    /// dtype.
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
        fmt::Debug::fmt(&self.inner, f)
    }
}

impl<T: Element> TensorTrait<T> for Tensor<T> {
    fn cpu_access(&self) -> CpuAccess {
        Tensor::cpu_access(self)
    }

    fn view_origin(&self) -> Option<ViewOrigin> {
        Tensor::view_origin(self)
    }

    fn is_view(&self) -> bool {
        Tensor::is_view(self)
    }

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

    /// Forwards to [`TensorDyn::set_logical_shape`].
    fn set_logical_shape(&mut self, shape: &[usize]) -> Result<()> {
        self.inner.set_logical_shape(shape)
    }

    /// The allocation's byte count, forwarded to [`TensorDyn::capacity_bytes`].
    fn capacity_bytes(&self) -> usize {
        self.inner.capacity_bytes()
    }

    /// One `ef_tensor_map` call for the pointer (the `TensorDyn::map_pin`
    /// core), then a typed [`crate::view::HostView`] over the mapped window.
    /// The view's byte size is `pin.len()`, the stride-aware extent that was
    /// mapped.
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

    /// The non-blocking counterpart of [`map_with`](Self::map_with), routed
    /// through `ef_tensor_try_map`.
    fn try_map_with<'a>(&self, access: CpuAccess) -> Result<crate::view::HostView<'a, T>>
    where
        T: 'a,
    {
        let pin = self.inner.map_pin_with(access, true)?;
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
