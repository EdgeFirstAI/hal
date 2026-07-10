// SPDX-FileCopyrightText: Copyright 2025 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

use crate::{DType, PixelFormat, Tensor, TensorMemory, TensorTrait};
use half::f16;
use std::fmt;

/// Type-erased tensor. Wraps a `Tensor<T>` with runtime element type.
#[non_exhaustive]
pub enum TensorDyn {
    /// Unsigned 8-bit integer tensor.
    U8(Tensor<u8>),
    /// Signed 8-bit integer tensor.
    I8(Tensor<i8>),
    /// Unsigned 16-bit integer tensor.
    U16(Tensor<u16>),
    /// Signed 16-bit integer tensor.
    I16(Tensor<i16>),
    /// Unsigned 32-bit integer tensor.
    U32(Tensor<u32>),
    /// Signed 32-bit integer tensor.
    I32(Tensor<i32>),
    /// Unsigned 64-bit integer tensor.
    U64(Tensor<u64>),
    /// Signed 64-bit integer tensor.
    I64(Tensor<i64>),
    /// 16-bit floating-point tensor.
    F16(Tensor<f16>),
    /// 32-bit floating-point tensor.
    F32(Tensor<f32>),
    /// 64-bit floating-point tensor.
    F64(Tensor<f64>),
}

/// Dispatch a method call across all TensorDyn variants.
macro_rules! dispatch {
    ($self:expr, $method:ident $(, $arg:expr)*) => {
        match $self {
            TensorDyn::U8(t) => t.$method($($arg),*),
            TensorDyn::I8(t) => t.$method($($arg),*),
            TensorDyn::U16(t) => t.$method($($arg),*),
            TensorDyn::I16(t) => t.$method($($arg),*),
            TensorDyn::U32(t) => t.$method($($arg),*),
            TensorDyn::I32(t) => t.$method($($arg),*),
            TensorDyn::U64(t) => t.$method($($arg),*),
            TensorDyn::I64(t) => t.$method($($arg),*),
            TensorDyn::F16(t) => t.$method($($arg),*),
            TensorDyn::F32(t) => t.$method($($arg),*),
            TensorDyn::F64(t) => t.$method($($arg),*),
        }
    };
}

/// Like [`dispatch!`], but for methods returning `Result<Tensor<T>>`: rewrap the
/// typed result back into the matching `TensorDyn` variant. Keeps sub-region
/// fan-out (`batch`, future `view`) to one line instead of an 11-arm match.
macro_rules! dyn_fanout {
    ($self:expr, $method:ident $(, $arg:expr)*) => {
        match $self {
            TensorDyn::U8(t) => t.$method($($arg),*).map(TensorDyn::U8),
            TensorDyn::I8(t) => t.$method($($arg),*).map(TensorDyn::I8),
            TensorDyn::U16(t) => t.$method($($arg),*).map(TensorDyn::U16),
            TensorDyn::I16(t) => t.$method($($arg),*).map(TensorDyn::I16),
            TensorDyn::U32(t) => t.$method($($arg),*).map(TensorDyn::U32),
            TensorDyn::I32(t) => t.$method($($arg),*).map(TensorDyn::I32),
            TensorDyn::U64(t) => t.$method($($arg),*).map(TensorDyn::U64),
            TensorDyn::I64(t) => t.$method($($arg),*).map(TensorDyn::I64),
            TensorDyn::F16(t) => t.$method($($arg),*).map(TensorDyn::F16),
            TensorDyn::F32(t) => t.$method($($arg),*).map(TensorDyn::F32),
            TensorDyn::F64(t) => t.$method($($arg),*).map(TensorDyn::F64),
        }
    };
}

/// Generate the three downcast methods (ref, mut ref, owned) for one variant.
macro_rules! downcast_methods {
    ($variant:ident, $ty:ty, $as_name:ident, $as_mut_name:ident, $into_name:ident) => {
        /// Returns a shared reference to the inner tensor if the type matches.
        pub fn $as_name(&self) -> Option<&Tensor<$ty>> {
            match self {
                Self::$variant(t) => Some(t),
                _ => None,
            }
        }

        /// Returns a mutable reference to the inner tensor if the type matches.
        pub fn $as_mut_name(&mut self) -> Option<&mut Tensor<$ty>> {
            match self {
                Self::$variant(t) => Some(t),
                _ => None,
            }
        }

        /// Unwraps the inner tensor if the type matches, otherwise returns `self` as `Err`.
        /// The Err variant is necessarily large (returns the unconsumed TensorDyn).
        #[allow(clippy::result_large_err)]
        pub fn $into_name(self) -> Result<Tensor<$ty>, Self> {
            match self {
                Self::$variant(t) => Ok(t),
                other => Err(other),
            }
        }
    };
}

impl TensorDyn {
    /// Return the runtime element type discriminant.
    pub fn dtype(&self) -> DType {
        match self {
            Self::U8(_) => DType::U8,
            Self::I8(_) => DType::I8,
            Self::U16(_) => DType::U16,
            Self::I16(_) => DType::I16,
            Self::U32(_) => DType::U32,
            Self::I32(_) => DType::I32,
            Self::U64(_) => DType::U64,
            Self::I64(_) => DType::I64,
            Self::F16(_) => DType::F16,
            Self::F32(_) => DType::F32,
            Self::F64(_) => DType::F64,
        }
    }

    /// Return the tensor shape.
    pub fn shape(&self) -> &[usize] {
        dispatch!(self, shape)
    }

    /// Return the tensor name.
    pub fn name(&self) -> String {
        dispatch!(self, name)
    }

    /// Return the pixel format (None if not an image tensor).
    pub fn format(&self) -> Option<PixelFormat> {
        dispatch!(self, format)
    }

    /// Return the image width (None if not an image tensor).
    pub fn width(&self) -> Option<usize> {
        dispatch!(self, width)
    }

    /// Return the image height (None if not an image tensor).
    pub fn height(&self) -> Option<usize> {
        dispatch!(self, height)
    }

    /// Return the total size of this tensor in bytes.
    pub fn size(&self) -> usize {
        dispatch!(self, size)
    }

    /// Return the memory allocation type.
    pub fn memory(&self) -> TensorMemory {
        dispatch!(self, memory)
    }

    /// Reshape this tensor. Total element count must remain the same.
    pub fn reshape(&mut self, shape: &[usize]) -> crate::Result<()> {
        dispatch!(self, reshape, shape)
    }

    /// Attach pixel format metadata to this tensor.
    ///
    /// Validates that the tensor's shape is compatible with the format's
    /// layout (packed, planar, or semi-planar).
    ///
    /// # Arguments
    ///
    /// * `format` - The pixel format to attach
    ///
    /// # Returns
    ///
    /// `Ok(())` on success, with the format stored as metadata on the tensor.
    ///
    /// # Errors
    ///
    /// Returns `Error::InvalidShape` if the tensor shape doesn't match
    /// the expected layout for the given format.
    pub fn set_format(&mut self, format: PixelFormat) -> crate::Result<()> {
        dispatch!(self, set_format, format)
    }

    /// Attach pixel format metadata, consuming and returning self.
    ///
    /// Enables builder-style chaining.
    ///
    /// # Arguments
    ///
    /// * `format` - The pixel format to attach
    ///
    /// # Returns
    ///
    /// The tensor with format metadata attached.
    ///
    /// # Errors
    ///
    /// Returns `Error::InvalidShape` if the tensor shape doesn't match
    /// the expected layout for the given format.
    pub fn with_format(mut self, format: PixelFormat) -> crate::Result<Self> {
        self.set_format(format)?;
        Ok(self)
    }

    /// Colorimetry metadata (`None` = undefined; never auto-filled).
    pub fn colorimetry(&self) -> Option<crate::Colorimetry> {
        dispatch!(self, colorimetry)
    }

    /// Attach/clear colorimetry metadata.
    pub fn set_colorimetry(&mut self, c: Option<crate::Colorimetry>) {
        dispatch!(self, set_colorimetry, c)
    }

    /// Builder-style colorimetry attach (consumes and returns self).
    pub fn with_colorimetry(mut self, c: crate::Colorimetry) -> Self {
        self.set_colorimetry(Some(c));
        self
    }

    /// Row stride in bytes (`None` = tightly packed).
    pub fn row_stride(&self) -> Option<usize> {
        dispatch!(self, row_stride)
    }

    /// Effective row stride: stored stride or computed from format and width.
    pub fn effective_row_stride(&self) -> Option<usize> {
        dispatch!(self, effective_row_stride)
    }

    /// Set logical dimensions + format to a decoded image, reusing the
    /// allocation. See [`Tensor::configure_image`].
    pub fn configure_image(
        &mut self,
        width: usize,
        height: usize,
        format: PixelFormat,
    ) -> crate::Result<()> {
        dispatch!(self, configure_image, width, height, format)
    }

    /// Set the row stride in bytes for externally allocated buffers with
    /// row padding.
    ///
    /// Must be called before the tensor is first used for rendering. The
    /// format must be set before calling this method.
    pub fn set_row_stride(&mut self, stride: usize) -> crate::Result<()> {
        dispatch!(self, set_row_stride, stride)
    }

    /// Builder-style: set row stride, consuming and returning self.
    pub fn with_row_stride(mut self, stride: usize) -> crate::Result<Self> {
        self.set_row_stride(stride)?;
        Ok(self)
    }

    /// Byte offset within the DMA-BUF where image data starts (`None` = 0).
    pub fn plane_offset(&self) -> Option<usize> {
        dispatch!(self, plane_offset)
    }

    /// The parent-image snapshot if this tensor is a [`view`](Self::view)/
    /// [`batch`](Self::batch) sub-region; `None` for a whole tensor. See
    /// [`Tensor::view_origin`].
    pub fn view_origin(&self) -> Option<crate::ViewOrigin> {
        dispatch!(self, view_origin)
    }

    /// Set the byte offset within the DMA-BUF where image data starts.
    pub fn set_plane_offset(&mut self, offset: usize) {
        dispatch!(self, set_plane_offset, offset)
    }

    /// Borrow batch element `n` of a batched tensor (leading `N` dimension) as a
    /// zero-copy view sharing this tensor's allocation. See [`Tensor::batch`].
    pub fn batch(&self, n: usize) -> crate::Result<TensorDyn> {
        dyn_fanout!(self, batch, n)
    }

    /// Borrow a rectangular spatial sub-region (the destination/source crop) as
    /// a zero-copy view sharing this tensor's allocation. See [`Tensor::view`].
    pub fn view(&self, region: crate::Region) -> crate::Result<TensorDyn> {
        dyn_fanout!(self, view, region)
    }

    /// The CUDA registration for this tensor, if any.
    ///
    /// Returns `None` when no CUDA handle has been attached (the common non-CUDA case).
    /// This check is a pure local field read — no thread routing occurs.
    pub fn cuda(&self) -> Option<&crate::cuda::CudaHandle> {
        dispatch!(self, cuda)
    }

    /// Fast-fail CUDA map: `None` when no handle is attached; else maps the
    /// PBO through the GL worker and returns a scoped device-pointer guard.
    ///
    /// The same try-`cuda_map`-then-[`map`](crate::TensorTrait::map) fallback pattern that applies to
    /// [`Tensor::cuda_map`](crate::Tensor::cuda_map) applies here: call `cuda_map()` first for a
    /// zero-copy device pointer; when it returns `None` (no CUDA handle attached), fall back to the
    /// typed host mapping via the inner tensor.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use edgefirst_tensor::TensorDyn;
    /// # fn feed_tensorrt(_dptr: *mut std::ffi::c_void, _bytes: usize) {}
    /// # fn demo(t: &TensorDyn) {
    /// if let Some(cuda) = t.cuda_map() {
    ///     feed_tensorrt(cuda.device_ptr(), cuda.len());
    /// } else {
    ///     // No CUDA handle — use the typed inner tensor for host access.
    ///     // See Tensor::cuda_map for the full fallback example.
    /// }
    /// # }
    /// ```
    pub fn cuda_map(&self) -> Option<crate::cuda::CudaMap<'_>> {
        dispatch!(self, cuda_map)
    }

    /// Quantization metadata. Returns `None` for float variants (F16, F32,
    /// F64) — quantization does not apply to floating-point tensors.
    /// Otherwise delegates to the typed `Tensor<T>::quantization()` accessor.
    pub fn quantization(&self) -> Option<&crate::Quantization> {
        match self {
            Self::U8(t) => t.quantization(),
            Self::I8(t) => t.quantization(),
            Self::U16(t) => t.quantization(),
            Self::I16(t) => t.quantization(),
            Self::U32(t) => t.quantization(),
            Self::I32(t) => t.quantization(),
            Self::U64(t) => t.quantization(),
            Self::I64(t) => t.quantization(),
            Self::F16(_) | Self::F32(_) | Self::F64(_) => None,
        }
    }

    /// Attach quantization metadata. Fails on float variants with
    /// [`Error::QuantizationInvalid`]; delegates to the typed setter for
    /// integer variants.
    pub fn set_quantization(&mut self, q: crate::Quantization) -> crate::Result<()> {
        match self {
            Self::U8(t) => t.set_quantization(q),
            Self::I8(t) => t.set_quantization(q),
            Self::U16(t) => t.set_quantization(q),
            Self::I16(t) => t.set_quantization(q),
            Self::U32(t) => t.set_quantization(q),
            Self::I32(t) => t.set_quantization(q),
            Self::U64(t) => t.set_quantization(q),
            Self::I64(t) => t.set_quantization(q),
            Self::F16(_) | Self::F32(_) | Self::F64(_) => Err(crate::Error::QuantizationInvalid {
                field: "dtype_is_integer",
                expected: "integer tensor dtype (u8/i8/u16/i16/u32/i32/u64/i64)".to_string(),
                got: format!("{:?}", self.dtype()),
            }),
        }
    }

    /// Builder-style variant of [`Self::set_quantization`]. Consumes self
    /// and returns it with quantization applied (or the original error).
    pub fn with_quantization(mut self, q: crate::Quantization) -> crate::Result<Self> {
        self.set_quantization(q)?;
        Ok(self)
    }

    /// Clear any quantization metadata. No-op on float variants.
    pub fn clear_quantization(&mut self) {
        match self {
            Self::U8(t) => t.clear_quantization(),
            Self::I8(t) => t.clear_quantization(),
            Self::U16(t) => t.clear_quantization(),
            Self::I16(t) => t.clear_quantization(),
            Self::U32(t) => t.clear_quantization(),
            Self::I32(t) => t.clear_quantization(),
            Self::U64(t) => t.clear_quantization(),
            Self::I64(t) => t.clear_quantization(),
            Self::F16(_) | Self::F32(_) | Self::F64(_) => {}
        }
    }

    /// Clone the file descriptor associated with this tensor.
    #[cfg(unix)]
    pub fn clone_fd(&self) -> crate::Result<std::os::fd::OwnedFd> {
        dispatch!(self, clone_fd)
    }

    /// Clone the DMA-BUF file descriptor backing this tensor (Linux only).
    ///
    /// # Returns
    ///
    /// An owned duplicate of the DMA-BUF file descriptor.
    ///
    /// # Errors
    ///
    /// * `Error::NotImplemented` if the tensor is not DMA-backed (Mem/Shm/Pbo)
    /// * `Error::IoError` if the fd clone syscall fails (e.g., fd limit reached)
    #[cfg(target_os = "linux")]
    pub fn dmabuf_clone(&self) -> crate::Result<std::os::fd::OwnedFd> {
        if self.memory() != TensorMemory::Dma {
            return Err(crate::Error::NotImplemented(format!(
                "dmabuf_clone requires DMA-backed tensor, got {:?}",
                self.memory()
            )));
        }
        self.clone_fd()
    }

    /// Borrow the DMA-BUF file descriptor backing this tensor (Linux only).
    ///
    /// # Returns
    ///
    /// A borrowed reference to the DMA-BUF file descriptor, tied to `self`'s
    /// lifetime.
    ///
    /// # Errors
    ///
    /// * `Error::NotImplemented` if the tensor is not DMA-backed
    #[cfg(target_os = "linux")]
    pub fn dmabuf(&self) -> crate::Result<std::os::fd::BorrowedFd<'_>> {
        dispatch!(self, dmabuf)
    }

    /// Return `true` if this tensor uses separate plane allocations.
    pub fn is_multiplane(&self) -> bool {
        dispatch!(self, is_multiplane)
    }

    /// Return the [`BufferIdentity`](crate::BufferIdentity) of the underlying
    /// allocation.
    ///
    /// Two `TensorDyn` values share a [`BufferIdentity::id`] iff they were
    /// produced by cloning the same allocation (e.g. through
    /// [`DmaTensor::try_clone`](crate::dma::DmaTensor::try_clone)). Separate
    /// imports of the same physical buffer (e.g. two `from_fd` calls on the
    /// same dmabuf fd) have **distinct** identities — use
    /// [`aliases`](Self::aliases) if you need to detect that case.
    pub fn buffer_identity(&self) -> &crate::BufferIdentity {
        dispatch!(self, buffer_identity)
    }

    /// Return `true` if `self` and `other` reference the same underlying
    /// buffer.
    ///
    /// This is the correct check for APIs that require distinct input and
    /// output tensors (e.g. `ImageProcessor::draw_decoded_masks`, where
    /// aliasing `dst` and `background` would cause the GL backend to read
    /// and write the same texture — undefined behaviour on most drivers).
    ///
    /// Matching is conservative:
    /// 1. Matching [`BufferIdentity::id`] → same buffer (always).
    /// 2. Matching backing type + matching dmabuf fd number (Linux, DMA
    ///    tensors only) → same buffer, even across separate `from_fd`
    ///    imports in the same process.
    ///
    /// Two distinct `dup`'d fds pointing at the same kernel dma-buf are
    /// **not** detected — there is no cheap way to resolve that without a
    /// round-trip through the kernel.
    pub fn aliases(&self, other: &Self) -> bool {
        if self.buffer_identity().id() == other.buffer_identity().id() {
            return true;
        }
        if self.memory() != other.memory() {
            return false;
        }
        #[cfg(target_os = "linux")]
        if self.memory() == TensorMemory::Dma {
            use std::os::fd::AsRawFd;
            if let (Ok(a), Ok(b)) = (self.dmabuf(), other.dmabuf()) {
                return a.as_raw_fd() == b.as_raw_fd();
            }
        }
        false
    }

    // --- Downcasting ---

    downcast_methods!(U8, u8, as_u8, as_u8_mut, into_u8);
    downcast_methods!(I8, i8, as_i8, as_i8_mut, into_i8);
    downcast_methods!(U16, u16, as_u16, as_u16_mut, into_u16);
    downcast_methods!(I16, i16, as_i16, as_i16_mut, into_i16);
    downcast_methods!(U32, u32, as_u32, as_u32_mut, into_u32);
    downcast_methods!(I32, i32, as_i32, as_i32_mut, into_i32);
    downcast_methods!(U64, u64, as_u64, as_u64_mut, into_u64);
    downcast_methods!(I64, i64, as_i64, as_i64_mut, into_i64);
    downcast_methods!(F16, f16, as_f16, as_f16_mut, into_f16);
    downcast_methods!(F32, f32, as_f32, as_f32_mut, into_f32);
    downcast_methods!(F64, f64, as_f64, as_f64_mut, into_f64);

    /// Create a type-erased tensor with the given shape and element type.
    pub fn new(
        shape: &[usize],
        dtype: DType,
        memory: Option<TensorMemory>,
        name: Option<&str>,
    ) -> crate::Result<Self> {
        match dtype {
            DType::U8 => Tensor::<u8>::new(shape, memory, name).map(Self::U8),
            DType::I8 => Tensor::<i8>::new(shape, memory, name).map(Self::I8),
            DType::U16 => Tensor::<u16>::new(shape, memory, name).map(Self::U16),
            DType::I16 => Tensor::<i16>::new(shape, memory, name).map(Self::I16),
            DType::U32 => Tensor::<u32>::new(shape, memory, name).map(Self::U32),
            DType::I32 => Tensor::<i32>::new(shape, memory, name).map(Self::I32),
            DType::U64 => Tensor::<u64>::new(shape, memory, name).map(Self::U64),
            DType::I64 => Tensor::<i64>::new(shape, memory, name).map(Self::I64),
            DType::F16 => Tensor::<f16>::new(shape, memory, name).map(Self::F16),
            DType::F32 => Tensor::<f32>::new(shape, memory, name).map(Self::F32),
            DType::F64 => Tensor::<f64>::new(shape, memory, name).map(Self::F64),
        }
    }

    /// Create a type-erased tensor from a file descriptor.
    #[cfg(unix)]
    pub fn from_fd(
        fd: std::os::fd::OwnedFd,
        shape: &[usize],
        dtype: DType,
        name: Option<&str>,
    ) -> crate::Result<Self> {
        match dtype {
            DType::U8 => Tensor::<u8>::from_fd(fd, shape, name).map(Self::U8),
            DType::I8 => Tensor::<i8>::from_fd(fd, shape, name).map(Self::I8),
            DType::U16 => Tensor::<u16>::from_fd(fd, shape, name).map(Self::U16),
            DType::I16 => Tensor::<i16>::from_fd(fd, shape, name).map(Self::I16),
            DType::U32 => Tensor::<u32>::from_fd(fd, shape, name).map(Self::U32),
            DType::I32 => Tensor::<i32>::from_fd(fd, shape, name).map(Self::I32),
            DType::U64 => Tensor::<u64>::from_fd(fd, shape, name).map(Self::U64),
            DType::I64 => Tensor::<i64>::from_fd(fd, shape, name).map(Self::I64),
            DType::F16 => Tensor::<f16>::from_fd(fd, shape, name).map(Self::F16),
            DType::F32 => Tensor::<f32>::from_fd(fd, shape, name).map(Self::F32),
            DType::F64 => Tensor::<f64>::from_fd(fd, shape, name).map(Self::F64),
        }
    }

    /// Wrap externally-owned memory as a type-erased tensor without copying.
    /// The tensor borrows `[ptr, ptr + shape.product() * dtype.size())` as
    /// [`TensorMemory::Mem`]; `owner`, when `Some`, co-owns the source so it
    /// outlives the tensor (and all derived views/maps). See
    /// [`crate::ForeignOwner`] and [`Tensor::from_foreign`].
    ///
    /// # Safety
    ///
    /// `ptr` must be non-null, aligned to the element type, and valid for
    /// `shape.product()` elements of `dtype` for as long as the returned
    /// tensor — and every view/map sharing its backing — is alive. Pass an
    /// `owner` that co-owns the source to uphold that contract.
    pub unsafe fn from_foreign_ptr(
        ptr: *mut u8,
        shape: &[usize],
        dtype: DType,
        owner: Option<crate::ForeignOwner>,
        name: Option<&str>,
    ) -> crate::Result<Self> {
        match dtype {
            DType::U8 => Tensor::<u8>::from_foreign(ptr.cast(), shape, owner, name).map(Self::U8),
            DType::I8 => Tensor::<i8>::from_foreign(ptr.cast(), shape, owner, name).map(Self::I8),
            DType::U16 => {
                Tensor::<u16>::from_foreign(ptr.cast(), shape, owner, name).map(Self::U16)
            }
            DType::I16 => {
                Tensor::<i16>::from_foreign(ptr.cast(), shape, owner, name).map(Self::I16)
            }
            DType::U32 => {
                Tensor::<u32>::from_foreign(ptr.cast(), shape, owner, name).map(Self::U32)
            }
            DType::I32 => {
                Tensor::<i32>::from_foreign(ptr.cast(), shape, owner, name).map(Self::I32)
            }
            DType::U64 => {
                Tensor::<u64>::from_foreign(ptr.cast(), shape, owner, name).map(Self::U64)
            }
            DType::I64 => {
                Tensor::<i64>::from_foreign(ptr.cast(), shape, owner, name).map(Self::I64)
            }
            DType::F16 => {
                Tensor::<f16>::from_foreign(ptr.cast(), shape, owner, name).map(Self::F16)
            }
            DType::F32 => {
                Tensor::<f32>::from_foreign(ptr.cast(), shape, owner, name).map(Self::F32)
            }
            DType::F64 => {
                Tensor::<f64>::from_foreign(ptr.cast(), shape, owner, name).map(Self::F64)
            }
        }
    }

    /// Wrap an externally-allocated IOSurface as a type-erased tensor
    /// (macOS only).
    ///
    /// # Safety
    ///
    /// `surface_ref` must be a valid live `IOSurfaceRef`. `shape` must
    /// match the IOSurface's pixel dimensions and chosen element type.
    #[cfg(target_os = "macos")]
    pub unsafe fn from_iosurface(
        surface_ref: *mut std::ffi::c_void,
        shape: &[usize],
        dtype: DType,
        name: Option<&str>,
    ) -> crate::Result<Self> {
        unsafe {
            match dtype {
                DType::U8 => Tensor::<u8>::from_iosurface(surface_ref, shape, name).map(Self::U8),
                DType::I8 => Tensor::<i8>::from_iosurface(surface_ref, shape, name).map(Self::I8),
                DType::U16 => {
                    Tensor::<u16>::from_iosurface(surface_ref, shape, name).map(Self::U16)
                }
                DType::I16 => {
                    Tensor::<i16>::from_iosurface(surface_ref, shape, name).map(Self::I16)
                }
                DType::U32 => {
                    Tensor::<u32>::from_iosurface(surface_ref, shape, name).map(Self::U32)
                }
                DType::I32 => {
                    Tensor::<i32>::from_iosurface(surface_ref, shape, name).map(Self::I32)
                }
                DType::U64 => {
                    Tensor::<u64>::from_iosurface(surface_ref, shape, name).map(Self::U64)
                }
                DType::I64 => {
                    Tensor::<i64>::from_iosurface(surface_ref, shape, name).map(Self::I64)
                }
                DType::F16 => {
                    Tensor::<f16>::from_iosurface(surface_ref, shape, name).map(Self::F16)
                }
                DType::F32 => {
                    Tensor::<f32>::from_iosurface(surface_ref, shape, name).map(Self::F32)
                }
                DType::F64 => {
                    Tensor::<f64>::from_iosurface(surface_ref, shape, name).map(Self::F64)
                }
            }
        }
    }

    /// IOSurfaceID for cross-process surface sharing (macOS only).
    /// Returns `None` when the tensor is not IOSurface-backed.
    #[cfg(target_os = "macos")]
    pub fn iosurface_id(&self) -> Option<u32> {
        dispatch!(self, iosurface_id)
    }

    /// Borrow the raw `IOSurfaceRef` backing this tensor (macOS only).
    /// Returns `None` when the tensor is not IOSurface-backed. The
    /// pointer's lifetime is tied to `self`.
    #[cfg(target_os = "macos")]
    pub fn iosurface_ref(&self) -> Option<*mut std::ffi::c_void> {
        dispatch!(self, iosurface_ref)
    }

    /// Physical IOSurface dimensions in texels, independent of the logical
    /// shape (macOS only). `None` when not IOSurface-backed. The GL backend
    /// binds the EGL pbuffer at these dims so one cached pbuffer serves every
    /// frame size a reused pool surface holds.
    #[cfg(target_os = "macos")]
    pub fn iosurface_physical_dims(&self) -> Option<(usize, usize)> {
        dispatch!(self, iosurface_physical_dims)
    }

    /// Wrap an externally-allocated AHardwareBuffer as a type-erased
    /// tensor (Android only). Used to import buffers from
    /// CameraX/ImageReader (via JNI), NNAPI, or cross-process binder
    /// transfers.
    ///
    /// # Safety
    ///
    /// `buffer_ptr` must be a valid live AHardwareBuffer pointer. `shape`
    /// must match the buffer's dimensions and chosen element type.
    #[cfg(target_os = "android")]
    pub unsafe fn from_hardware_buffer(
        buffer_ptr: *mut std::ffi::c_void,
        shape: &[usize],
        dtype: DType,
        name: Option<&str>,
    ) -> crate::Result<Self> {
        unsafe {
            match dtype {
                DType::U8 => {
                    Tensor::<u8>::from_hardware_buffer(buffer_ptr, shape, name).map(Self::U8)
                }
                DType::I8 => {
                    Tensor::<i8>::from_hardware_buffer(buffer_ptr, shape, name).map(Self::I8)
                }
                DType::U16 => {
                    Tensor::<u16>::from_hardware_buffer(buffer_ptr, shape, name).map(Self::U16)
                }
                DType::I16 => {
                    Tensor::<i16>::from_hardware_buffer(buffer_ptr, shape, name).map(Self::I16)
                }
                DType::U32 => {
                    Tensor::<u32>::from_hardware_buffer(buffer_ptr, shape, name).map(Self::U32)
                }
                DType::I32 => {
                    Tensor::<i32>::from_hardware_buffer(buffer_ptr, shape, name).map(Self::I32)
                }
                DType::U64 => {
                    Tensor::<u64>::from_hardware_buffer(buffer_ptr, shape, name).map(Self::U64)
                }
                DType::I64 => {
                    Tensor::<i64>::from_hardware_buffer(buffer_ptr, shape, name).map(Self::I64)
                }
                DType::F16 => {
                    Tensor::<f16>::from_hardware_buffer(buffer_ptr, shape, name).map(Self::F16)
                }
                DType::F32 => {
                    Tensor::<f32>::from_hardware_buffer(buffer_ptr, shape, name).map(Self::F32)
                }
                DType::F64 => {
                    Tensor::<f64>::from_hardware_buffer(buffer_ptr, shape, name).map(Self::F64)
                }
            }
        }
    }

    /// Borrow the raw AHardwareBuffer pointer backing this tensor
    /// (Android only). Returns `None` when the tensor is not
    /// AHardwareBuffer-backed. The pointer's lifetime is tied to `self`.
    #[cfg(target_os = "android")]
    pub fn hardware_buffer_ptr(&self) -> Option<*mut std::ffi::c_void> {
        dispatch!(self, hardware_buffer_ptr)
    }

    /// Physical AHardwareBuffer dimensions in texels, independent of the
    /// logical shape (Android only). `None` when not
    /// AHardwareBuffer-backed.
    #[cfg(target_os = "android")]
    pub fn hardware_buffer_physical_dims(&self) -> Option<(usize, usize)> {
        dispatch!(self, hardware_buffer_physical_dims)
    }

    /// Create a type-erased image tensor.
    ///
    /// # Arguments
    ///
    /// * `width` - Image width in pixels
    /// * `height` - Image height in pixels
    /// * `format` - Pixel format
    /// * `dtype` - Element type discriminant
    /// * `memory` - Optional memory backend (None selects the best available)
    ///
    /// # Returns
    ///
    /// A new `TensorDyn` wrapping an image tensor of the requested element type.
    ///
    /// # Errors
    ///
    /// Returns an error if the underlying `Tensor::image` call fails.
    pub fn image(
        width: usize,
        height: usize,
        format: PixelFormat,
        dtype: DType,
        memory: Option<TensorMemory>,
    ) -> crate::Result<Self> {
        match dtype {
            DType::U8 => Tensor::<u8>::image(width, height, format, memory).map(Self::U8),
            DType::I8 => Tensor::<i8>::image(width, height, format, memory).map(Self::I8),
            DType::U16 => Tensor::<u16>::image(width, height, format, memory).map(Self::U16),
            DType::I16 => Tensor::<i16>::image(width, height, format, memory).map(Self::I16),
            DType::U32 => Tensor::<u32>::image(width, height, format, memory).map(Self::U32),
            DType::I32 => Tensor::<i32>::image(width, height, format, memory).map(Self::I32),
            DType::U64 => Tensor::<u64>::image(width, height, format, memory).map(Self::U64),
            DType::I64 => Tensor::<i64>::image(width, height, format, memory).map(Self::I64),
            DType::F16 => Tensor::<f16>::image(width, height, format, memory).map(Self::F16),
            DType::F32 => Tensor::<f32>::image(width, height, format, memory).map(Self::F32),
            DType::F64 => Tensor::<f64>::image(width, height, format, memory).map(Self::F64),
        }
    }

    /// Create a DMA-backed image tensor with an explicit row stride that
    /// may exceed the natural `width * channels * sizeof(T)` pitch.
    ///
    /// See [`Tensor::image_with_stride`] for the detailed contract and
    /// constraints. The TensorDyn wrapper dispatches to the appropriate
    /// monomorphised `Tensor<T>` based on `dtype`.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use edgefirst_tensor::{TensorDyn, PixelFormat, DType, TensorMemory};
    /// # fn main() -> edgefirst_tensor::Result<()> {
    /// // Allocate a 3004×1688 RGBA8 canvas with 64-byte pitch alignment
    /// // (12032 bytes per row instead of the natural 12016).
    /// let img = TensorDyn::image_with_stride(
    ///     3004, 1688,
    ///     PixelFormat::Rgba, DType::U8,
    ///     12032,
    ///     Some(TensorMemory::Dma),
    /// )?;
    /// assert_eq!(img.width(), Some(3004));       // logical, unchanged
    /// assert_eq!(img.effective_row_stride(), Some(12032)); // padded
    /// # Ok(())
    /// # }
    /// ```
    pub fn image_with_stride(
        width: usize,
        height: usize,
        format: PixelFormat,
        dtype: DType,
        row_stride_bytes: usize,
        memory: Option<TensorMemory>,
    ) -> crate::Result<Self> {
        match dtype {
            DType::U8 => {
                Tensor::<u8>::image_with_stride(width, height, format, row_stride_bytes, memory)
                    .map(Self::U8)
            }
            DType::I8 => {
                Tensor::<i8>::image_with_stride(width, height, format, row_stride_bytes, memory)
                    .map(Self::I8)
            }
            DType::U16 => {
                Tensor::<u16>::image_with_stride(width, height, format, row_stride_bytes, memory)
                    .map(Self::U16)
            }
            DType::I16 => {
                Tensor::<i16>::image_with_stride(width, height, format, row_stride_bytes, memory)
                    .map(Self::I16)
            }
            DType::U32 => {
                Tensor::<u32>::image_with_stride(width, height, format, row_stride_bytes, memory)
                    .map(Self::U32)
            }
            DType::I32 => {
                Tensor::<i32>::image_with_stride(width, height, format, row_stride_bytes, memory)
                    .map(Self::I32)
            }
            DType::U64 => {
                Tensor::<u64>::image_with_stride(width, height, format, row_stride_bytes, memory)
                    .map(Self::U64)
            }
            DType::I64 => {
                Tensor::<i64>::image_with_stride(width, height, format, row_stride_bytes, memory)
                    .map(Self::I64)
            }
            DType::F16 => {
                Tensor::<f16>::image_with_stride(width, height, format, row_stride_bytes, memory)
                    .map(Self::F16)
            }
            DType::F32 => {
                Tensor::<f32>::image_with_stride(width, height, format, row_stride_bytes, memory)
                    .map(Self::F32)
            }
            DType::F64 => {
                Tensor::<f64>::image_with_stride(width, height, format, row_stride_bytes, memory)
                    .map(Self::F64)
            }
        }
    }
}

// --- From impls ---

impl From<Tensor<u8>> for TensorDyn {
    fn from(t: Tensor<u8>) -> Self {
        Self::U8(t)
    }
}

impl From<Tensor<i8>> for TensorDyn {
    fn from(t: Tensor<i8>) -> Self {
        Self::I8(t)
    }
}

impl From<Tensor<u16>> for TensorDyn {
    fn from(t: Tensor<u16>) -> Self {
        Self::U16(t)
    }
}

impl From<Tensor<i16>> for TensorDyn {
    fn from(t: Tensor<i16>) -> Self {
        Self::I16(t)
    }
}

impl From<Tensor<u32>> for TensorDyn {
    fn from(t: Tensor<u32>) -> Self {
        Self::U32(t)
    }
}

impl From<Tensor<i32>> for TensorDyn {
    fn from(t: Tensor<i32>) -> Self {
        Self::I32(t)
    }
}

impl From<Tensor<u64>> for TensorDyn {
    fn from(t: Tensor<u64>) -> Self {
        Self::U64(t)
    }
}

impl From<Tensor<i64>> for TensorDyn {
    fn from(t: Tensor<i64>) -> Self {
        Self::I64(t)
    }
}

impl From<Tensor<f16>> for TensorDyn {
    fn from(t: Tensor<f16>) -> Self {
        Self::F16(t)
    }
}

impl From<Tensor<f32>> for TensorDyn {
    fn from(t: Tensor<f32>) -> Self {
        Self::F32(t)
    }
}

impl From<Tensor<f64>> for TensorDyn {
    fn from(t: Tensor<f64>) -> Self {
        Self::F64(t)
    }
}

impl fmt::Debug for TensorDyn {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        dispatch!(self, fmt, f)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn from_typed_tensor() {
        let t = Tensor::<u8>::new(&[10], None, None).unwrap();
        let dyn_t: TensorDyn = t.into();
        assert_eq!(dyn_t.dtype(), DType::U8);
        assert_eq!(dyn_t.shape(), &[10]);
    }

    #[test]
    fn from_foreign_ptr_wraps_borrowed_memory() {
        use crate::TensorMapTrait;
        // The CUDA zero-copy export shape: wrap an externally-allocated buffer as
        // a type-erased Mem tensor, with an owner that frees it on last drop.
        let mut vec: Vec<f32> = vec![0.0; 4];
        let ptr = vec.as_mut_ptr() as *mut u8;
        let owner: crate::ForeignOwner = Box::new(vec);
        let t = unsafe {
            TensorDyn::from_foreign_ptr(ptr, &[2, 2], DType::F32, Some(owner), Some("trt_output"))
        }
        .unwrap();
        assert_eq!(t.dtype(), DType::F32);
        assert_eq!(t.memory(), TensorMemory::Mem);
        assert_eq!(t.shape(), &[2, 2]);
        {
            let mut m = t.as_f32().unwrap().map().unwrap();
            m.as_mut_slice().copy_from_slice(&[1.0, 2.0, 3.0, 4.0]);
        }
        let m = t.as_f32().unwrap().map().unwrap();
        assert_eq!(m.as_slice(), &[1.0, 2.0, 3.0, 4.0]);
    }

    // -------------------------------------------------------------------------
    // TensorDyn::from_foreign_ptr guard paths.
    //
    // The happy path (F32) is covered by `from_foreign_ptr_wraps_borrowed_memory`
    // above. These cells add the null-ptr, empty-shape, and overflow guards, plus
    // a U8 dtype to confirm the match-arm dispatch is exercised for integer types.
    // -------------------------------------------------------------------------

    #[test]
    fn from_foreign_ptr_rejects_null_ptr() {
        let err = unsafe {
            TensorDyn::from_foreign_ptr(std::ptr::null_mut(), &[4], DType::U8, None, None)
        }
        .unwrap_err();
        // The null guard fires inside Tensor<u8>::from_foreign.
        assert!(
            matches!(err, crate::error::Error::InvalidArgument(ref m) if m.contains("non-null")),
            "expected InvalidArgument(non-null), got {err:?}"
        );
    }

    #[test]
    fn from_foreign_ptr_rejects_empty_shape() {
        let mut dummy: u8 = 0;
        let err = unsafe {
            TensorDyn::from_foreign_ptr(&mut dummy as *mut u8, &[], DType::U8, None, None)
        }
        .unwrap_err();
        assert!(
            matches!(err, crate::error::Error::InvalidSize(0)),
            "expected InvalidSize(0) for empty shape, got {err:?}"
        );
    }

    #[test]
    fn from_foreign_ptr_rejects_overflow_shape() {
        let mut dummy: u8 = 0;
        let huge = [usize::MAX / 2 + 1, 2];
        let err = unsafe { TensorDyn::from_foreign_ptr(&mut dummy, &huge, DType::U8, None, None) }
            .unwrap_err();
        assert!(
            matches!(err, crate::error::Error::InvalidArgument(ref m) if m.contains("overflow")),
            "expected InvalidArgument(overflow), got {err:?}"
        );
    }

    #[test]
    fn from_foreign_ptr_u8_dtype_dispatch() {
        // Exercises the U8 arm of from_foreign_ptr's match, which wraps
        // the raw pointer as Tensor<u8> and downcasts correctly.
        let mut buf: Vec<u8> = vec![1, 2, 3, 4];
        let ptr = buf.as_mut_ptr();
        let owner: crate::ForeignOwner = Box::new(buf);
        let t = unsafe {
            TensorDyn::from_foreign_ptr(ptr, &[4], DType::U8, Some(owner), Some("u8_foreign"))
        }
        .unwrap();
        assert_eq!(t.dtype(), DType::U8);
        assert_eq!(t.shape(), &[4]);
        let m = t.as_u8().unwrap().map().unwrap();
        use crate::TensorMapTrait;
        assert_eq!(m.as_slice(), &[1u8, 2, 3, 4]);
    }

    #[test]
    fn downcast_ref() {
        let t = Tensor::<u8>::new(&[10], None, None).unwrap();
        let dyn_t: TensorDyn = t.into();
        assert!(dyn_t.as_u8().is_some());
        assert!(dyn_t.as_i8().is_none());
    }

    #[test]
    fn downcast_into() {
        let t = Tensor::<u8>::new(&[10], None, None).unwrap();
        let dyn_t: TensorDyn = t.into();
        let back = dyn_t.into_u8().unwrap();
        assert_eq!(back.shape(), &[10]);
    }

    #[test]
    fn image_accessors() {
        let t = Tensor::<u8>::image(640, 480, PixelFormat::Rgba, None).unwrap();
        let dyn_t: TensorDyn = t.into();
        assert_eq!(dyn_t.format(), Some(PixelFormat::Rgba));
        assert_eq!(dyn_t.width(), Some(640));
        assert_eq!(dyn_t.height(), Some(480));
        assert!(!dyn_t.is_multiplane());
    }

    #[test]
    fn image_constructor() {
        let dyn_t = TensorDyn::image(640, 480, PixelFormat::Rgb, DType::U8, None).unwrap();
        assert_eq!(dyn_t.dtype(), DType::U8);
        assert_eq!(dyn_t.format(), Some(PixelFormat::Rgb));
        assert_eq!(dyn_t.width(), Some(640));
    }

    #[test]
    fn image_constructor_i8() {
        let dyn_t = TensorDyn::image(640, 480, PixelFormat::Rgb, DType::I8, None).unwrap();
        assert_eq!(dyn_t.dtype(), DType::I8);
        assert_eq!(dyn_t.format(), Some(PixelFormat::Rgb));
    }

    #[test]
    fn set_format_packed() {
        let mut t = TensorDyn::new(&[480, 640, 3], DType::U8, None, None).unwrap();
        assert_eq!(t.format(), None);
        t.set_format(PixelFormat::Rgb).unwrap();
        assert_eq!(t.format(), Some(PixelFormat::Rgb));
        assert_eq!(t.width(), Some(640));
        assert_eq!(t.height(), Some(480));
    }

    #[test]
    fn set_format_planar() {
        let mut t = TensorDyn::new(&[3, 480, 640], DType::U8, None, None).unwrap();
        t.set_format(PixelFormat::PlanarRgb).unwrap();
        assert_eq!(t.format(), Some(PixelFormat::PlanarRgb));
        assert_eq!(t.width(), Some(640));
        assert_eq!(t.height(), Some(480));
    }

    #[test]
    fn set_format_rejects_wrong_shape() {
        let mut t = TensorDyn::new(&[480, 640, 4], DType::U8, None, None).unwrap();
        assert!(t.set_format(PixelFormat::Rgb).is_err());
    }

    #[test]
    fn with_format_builder() {
        let t = TensorDyn::new(&[480, 640, 4], DType::U8, None, None)
            .unwrap()
            .with_format(PixelFormat::Rgba)
            .unwrap();
        assert_eq!(t.format(), Some(PixelFormat::Rgba));
        assert_eq!(t.width(), Some(640));
        assert_eq!(t.height(), Some(480));
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn dmabuf_clone_mem_tensor_fails() {
        let t = TensorDyn::new(&[480, 640, 3], DType::U8, Some(TensorMemory::Mem), None).unwrap();
        assert_eq!(t.memory(), TensorMemory::Mem);
        assert!(t.dmabuf_clone().is_err());
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn dmabuf_mem_tensor_fails() {
        let t = TensorDyn::new(&[480, 640, 3], DType::U8, Some(TensorMemory::Mem), None).unwrap();
        assert!(t.dmabuf().is_err());
    }

    #[test]
    fn set_format_semi_planar_nv12() {
        // 720 rows = 480 * 3/2 (NV12: height + height/2 for chroma)
        let mut t = TensorDyn::new(&[720, 640], DType::U8, Some(TensorMemory::Mem), None).unwrap();
        t.set_format(PixelFormat::Nv12).unwrap();
        assert_eq!(t.format(), Some(PixelFormat::Nv12));
        assert_eq!(t.width(), Some(640));
        assert_eq!(t.height(), Some(480));
    }

    #[test]
    fn set_format_semi_planar_nv16() {
        // 960 rows = 480 * 2 (NV16: height + height for chroma)
        let mut t = TensorDyn::new(&[960, 640], DType::U8, Some(TensorMemory::Mem), None).unwrap();
        t.set_format(PixelFormat::Nv16).unwrap();
        assert_eq!(t.format(), Some(PixelFormat::Nv16));
        assert_eq!(t.width(), Some(640));
        assert_eq!(t.height(), Some(480));
    }

    #[test]
    fn with_format_rejects_wrong_shape() {
        let result = TensorDyn::new(&[480, 640, 4], DType::U8, None, None)
            .unwrap()
            .with_format(PixelFormat::Rgb);
        assert!(result.is_err());
    }

    #[test]
    fn set_format_preserved_after_rejection() {
        let mut t = TensorDyn::new(&[480, 640, 3], DType::U8, None, None).unwrap();
        t.set_format(PixelFormat::Rgb).unwrap();
        assert_eq!(t.format(), Some(PixelFormat::Rgb));

        // Rgba requires 4 channels, should fail on a 3-channel tensor
        assert!(t.set_format(PixelFormat::Rgba).is_err());

        // Original format should be preserved
        assert_eq!(t.format(), Some(PixelFormat::Rgb));
    }

    #[test]
    fn set_format_idempotent() {
        let mut t = TensorDyn::new(&[480, 640, 3], DType::U8, None, None).unwrap();
        t.set_format(PixelFormat::Rgb).unwrap();
        t.set_format(PixelFormat::Rgb).unwrap();
        assert_eq!(t.format(), Some(PixelFormat::Rgb));
        assert_eq!(t.width(), Some(640));
        assert_eq!(t.height(), Some(480));
    }

    // --- Row stride tests ---

    #[test]
    fn set_row_stride_valid() {
        // RGBA 100px wide: min stride = 400, set 512
        let mut t = TensorDyn::image(100, 100, PixelFormat::Rgba, DType::U8, None).unwrap();
        t.set_row_stride(512).unwrap();
        assert_eq!(t.row_stride(), Some(512));
        assert_eq!(t.effective_row_stride(), Some(512));
    }

    #[test]
    fn set_row_stride_equals_min() {
        // RGB 100px: min stride = 300, set exactly 300
        let mut t = TensorDyn::image(100, 100, PixelFormat::Rgb, DType::U8, None).unwrap();
        t.set_row_stride(300).unwrap();
        assert_eq!(t.row_stride(), Some(300));
    }

    #[test]
    fn set_row_stride_too_small() {
        // RGBA 64px (a 64-aligned width: 64*4 = 256, already a multiple of 64)
        // carries no implicit stride. min stride = 256; setting 200 must error
        // and leave row_stride unset. (Non-64-aligned widths now record the
        // padded stride at allocation — see `Tensor::image`.)
        let mut t = TensorDyn::image(64, 100, PixelFormat::Rgba, DType::U8, None).unwrap();
        assert!(t.set_row_stride(200).is_err());
        assert_eq!(t.row_stride(), None);
    }

    #[test]
    fn set_row_stride_zero() {
        let mut t = TensorDyn::image(100, 100, PixelFormat::Rgb, DType::U8, None).unwrap();
        assert!(t.set_row_stride(0).is_err());
    }

    #[test]
    fn set_row_stride_requires_format() {
        let mut t = TensorDyn::new(&[480, 640, 3], DType::U8, None, None).unwrap();
        assert!(t.set_row_stride(2048).is_err());
    }

    #[test]
    fn effective_row_stride_without_stride() {
        // A 64-aligned-width packed image carries no explicit stride; the
        // effective stride falls back to the computed tight pitch. (Width 64
        // RGB → 64*3 = 192, already a multiple of 64, so no padding is added.
        // Non-aligned widths now record the padded stride — see `Tensor::image`.)
        let t = TensorDyn::image(64, 100, PixelFormat::Rgb, DType::U8, None).unwrap();
        assert_eq!(t.row_stride(), None);
        assert_eq!(t.effective_row_stride(), Some(192)); // 64 * 3
    }

    #[test]
    fn effective_row_stride_padded_packed_dma() {
        // A non-64-aligned packed width on a DMA buffer records the 64-aligned
        // stride so the EGLImage import is accepted by Mali/Vivante (RGB 100px:
        // 100*3 = 300 → padded to 320). This padding is DMA-specific — host-only
        // memory keeps the tight pitch — so skip when DMA is unavailable (e.g. CI
        // without dma_heap); the behaviour is also validated on-target.
        let t = match TensorDyn::image(
            100,
            100,
            PixelFormat::Rgb,
            DType::U8,
            Some(TensorMemory::Dma),
        ) {
            Ok(t) if t.memory() == TensorMemory::Dma => t,
            _ => return,
        };
        assert_eq!(t.row_stride(), Some(320));
        assert_eq!(t.effective_row_stride(), Some(320));
    }

    #[test]
    fn effective_row_stride_no_format() {
        let t = TensorDyn::new(&[480, 640, 3], DType::U8, None, None).unwrap();
        assert_eq!(t.effective_row_stride(), None);
    }

    #[test]
    fn with_row_stride_builder() {
        let t = TensorDyn::image(100, 100, PixelFormat::Rgba, DType::U8, None)
            .unwrap()
            .with_row_stride(512)
            .unwrap();
        assert_eq!(t.row_stride(), Some(512));
        assert_eq!(t.effective_row_stride(), Some(512));
    }

    #[test]
    fn with_row_stride_rejects_small() {
        let result = TensorDyn::image(100, 100, PixelFormat::Rgba, DType::U8, None)
            .unwrap()
            .with_row_stride(200);
        assert!(result.is_err());
    }

    #[test]
    fn set_format_clears_row_stride() {
        let mut t = TensorDyn::new(&[480, 640, 3], DType::U8, None, None).unwrap();
        t.set_format(PixelFormat::Rgb).unwrap();
        t.set_row_stride(2048).unwrap();
        assert_eq!(t.row_stride(), Some(2048));

        // Incompatible format change (4-chan on 3-chan shape) fails — stride preserved
        let _ = t.set_format(PixelFormat::Bgra);
        assert_eq!(t.row_stride(), Some(2048));

        // Re-set to same format — stride preserved
        t.set_format(PixelFormat::Rgb).unwrap();
        assert_eq!(t.row_stride(), Some(2048));

        // Reshape clears format and stride
        t.reshape(&[480 * 640 * 3]).unwrap();
        assert_eq!(t.row_stride(), None);
        assert_eq!(t.format(), None);
    }

    #[test]
    fn set_format_different_compatible_clears_stride() {
        // RGBA and BGRA are both 4-channel packed — switching between them
        // succeeds and must clear the stored stride.
        let mut t = TensorDyn::new(&[480, 640, 4], DType::U8, None, None).unwrap();
        t.set_format(PixelFormat::Rgba).unwrap();
        t.set_row_stride(4096).unwrap();
        assert_eq!(t.row_stride(), Some(4096));

        // Successful format change to a different compatible format clears stride
        t.set_format(PixelFormat::Bgra).unwrap();
        assert_eq!(t.format(), Some(PixelFormat::Bgra));
        assert_eq!(t.row_stride(), None);
    }

    #[test]
    fn set_format_same_preserves_stride() {
        let mut t = TensorDyn::image(100, 100, PixelFormat::Rgb, DType::U8, None).unwrap();
        t.set_row_stride(512).unwrap();
        // Re-setting the same format should not clear stride
        t.set_format(PixelFormat::Rgb).unwrap();
        assert_eq!(t.row_stride(), Some(512));
    }

    #[test]
    fn effective_row_stride_planar() {
        let t = TensorDyn::image(640, 480, PixelFormat::PlanarRgb, DType::U8, None).unwrap();
        assert_eq!(t.effective_row_stride(), Some(640)); // planar: width only
    }

    #[test]
    fn effective_row_stride_nv12() {
        let t = TensorDyn::image(640, 480, PixelFormat::Nv12, DType::U8, None).unwrap();
        assert_eq!(t.effective_row_stride(), Some(640)); // semi-planar: width only
    }

    #[test]
    fn map_rejects_strided_tensor() {
        let mut t =
            Tensor::<u8>::image(100, 100, PixelFormat::Rgba, Some(TensorMemory::Mem)).unwrap();
        // Map works before stride is set
        assert!(t.map().is_ok());
        // After setting stride, map should be rejected
        t.set_row_stride(512).unwrap();
        let err = t.map();
        assert!(err.is_err());
    }

    // ── plane_offset tests ──────────────────────────────────────────

    #[test]
    fn plane_offset_default_none() {
        let t = TensorDyn::image(100, 100, PixelFormat::Rgba, DType::U8, None).unwrap();
        assert_eq!(t.plane_offset(), None);
    }

    #[test]
    fn set_plane_offset_basic() {
        let mut t = TensorDyn::image(100, 100, PixelFormat::Rgba, DType::U8, None).unwrap();
        t.set_plane_offset(4096);
        assert_eq!(t.plane_offset(), Some(4096));
    }

    #[test]
    fn set_plane_offset_zero() {
        let mut t = TensorDyn::image(100, 100, PixelFormat::Rgb, DType::U8, None).unwrap();
        t.set_plane_offset(0);
        assert_eq!(t.plane_offset(), Some(0));
    }

    #[test]
    fn set_plane_offset_no_format() {
        // plane_offset does not require format (it is format-independent)
        let mut t = TensorDyn::new(&[480, 640, 3], DType::U8, None, None).unwrap();
        t.set_plane_offset(4096);
        assert_eq!(t.plane_offset(), Some(4096));
    }

    #[test]
    fn set_format_clears_plane_offset() {
        let mut t = TensorDyn::new(&[480, 640, 3], DType::U8, None, None).unwrap();
        t.set_format(PixelFormat::Rgb).unwrap();
        t.set_plane_offset(4096);
        assert_eq!(t.plane_offset(), Some(4096));

        // Re-set same format — offset preserved
        t.set_format(PixelFormat::Rgb).unwrap();
        assert_eq!(t.plane_offset(), Some(4096));

        // Reshape clears everything
        t.reshape(&[480 * 640 * 3]).unwrap();
        assert_eq!(t.plane_offset(), None);
        assert_eq!(t.format(), None);
    }

    #[test]
    fn map_rejects_out_of_bounds_offset() {
        let mut t =
            Tensor::<u8>::image(100, 100, PixelFormat::Rgba, Some(TensorMemory::Mem)).unwrap();
        // Map works before offset is set.
        assert!(t.map().is_ok());
        // Heap offsets are now honored, but an offset that pushes the full
        // logical window (40000 bytes) past the allocation must be rejected.
        t.set_plane_offset(4096);
        assert!(t.map().is_err());
    }

    #[test]
    fn mem_subview_in_bounds_maps_at_offset() {
        // An in-bounds heap sub-view now maps at its offset (previously every
        // non-zero heap offset was rejected outright).
        let parent =
            Tensor::<u8>::image(100, 100, PixelFormat::Rgba, Some(TensorMemory::Mem)).unwrap();
        // A 10x10 RGBA window (400 bytes) at byte offset 4096 fits in 40000.
        let view = parent.subview(4096, &[10, 10, 4]).unwrap();
        assert_eq!(view.plane_offset(), Some(4096));
        assert!(view.map().is_ok());
    }

    #[test]
    fn dyn_batch_dispatches_every_dtype() {
        // `TensorDyn::batch` fans out across all 11 dtype arms via `dyn_fanout!`;
        // exercise each so element `n` preserves the element type and shape.
        // A `[N=2, 4]` raw parent: element 1 is the contiguous 4-element window.
        use DType::*;
        for dt in [U8, I8, U16, I16, U32, I32, U64, I64, F16, F32, F64] {
            let parent = TensorDyn::new(&[2, 4], dt, Some(TensorMemory::Mem), None).unwrap();
            let view = parent.batch(1).unwrap();
            assert_eq!(view.dtype(), dt, "batch must preserve dtype {dt:?}");
            assert_eq!(view.shape(), &[4], "{dt:?}");
        }
    }

    #[test]
    fn map_accepts_zero_offset_tensor() {
        let mut t =
            Tensor::<u8>::image(100, 100, PixelFormat::Rgba, Some(TensorMemory::Mem)).unwrap();
        t.set_plane_offset(0);
        // Zero offset is fine for CPU mapping
        assert!(t.map().is_ok());
    }

    #[test]
    fn dyn_configure_image_nv12() {
        let mut t = TensorDyn::image(640, 480, PixelFormat::Rgb, DType::U8, None).unwrap();
        t.configure_image(320, 240, PixelFormat::Nv12).unwrap();
        assert_eq!(t.format(), Some(PixelFormat::Nv12));
        assert_eq!((t.width(), t.height()), (Some(320), Some(240)));
    }

    #[test]
    fn tensordyn_colorimetry_roundtrip() {
        use crate::{ColorEncoding, Colorimetry, DType, PixelFormat};
        let mut t = TensorDyn::image(1280, 720, PixelFormat::Nv12, DType::U8, None).unwrap();
        assert_eq!(t.colorimetry(), None);
        let c = Colorimetry::default().with_encoding(ColorEncoding::Bt709);
        t.set_colorimetry(Some(c));
        assert_eq!(t.colorimetry(), Some(c));
    }

    #[test]
    fn from_planes_propagates_plane_offset() {
        let mut luma =
            Tensor::<u8>::new(&[480, 640], Some(TensorMemory::Mem), Some("luma")).unwrap();
        luma.set_plane_offset(4096);
        let chroma =
            Tensor::<u8>::new(&[240, 640], Some(TensorMemory::Mem), Some("chroma")).unwrap();
        let combined = Tensor::<u8>::from_planes(luma, chroma, PixelFormat::Nv12).unwrap();
        assert_eq!(combined.plane_offset(), Some(4096));
    }

    #[test]
    fn cuda_passthrough_none_for_mem_tensor() {
        // Build a Mem-backed dynamic tensor the same way the other tests here do,
        // then confirm the CUDA accessors pass through to None (no handle).
        let t: TensorDyn = Tensor::<f32>::new(&[10], Some(TensorMemory::Mem), None)
            .unwrap()
            .into();
        assert!(t.cuda().is_none());
        assert!(t.cuda_map().is_none());
    }
}
