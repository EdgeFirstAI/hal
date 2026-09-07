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
/// Rebuild a [`TensorDyn`] under a different variant of the **same element
/// width**, moving the storage across untouched.
///
/// Every `Tensor<T>` is layout-identical across same-width `T`: it holds
/// type-erased storage plus a `PhantomData`, with no inline `T` and no
/// `T`-dependent drop glue. `edgefirst-image` performs the same transmute
/// at two of its own call sites and pins that with `const` assertions; this
/// is the same operation, written once where the invariant is documented.
///
/// **Contains no panicking operation**, deliberately: [`TensorDyn::set_dtype`]
/// calls it between a `ptr::read` and a `ptr::write`, and an unwind in that
/// window would double-drop. The final arm therefore returns its input
/// unchanged instead of asserting — unreachable via `set_dtype`, which
/// rejects a width change before calling, and harmless if it ever were
/// reached (the tensor keeps the dtype it had).
///
/// # Safety
///
/// `dtype.size()` must equal `t`'s current dtype size.
// Each width group generates one arm whose target type equals its source --
// `U8 => u8` reached from a `U8` source, and so on. `transmute`-to-self is a
// no-op the compiler removes; special-casing it would mean an extra match
// arm per group and an extra branch to read, for no behavioural difference.
#[allow(clippy::useless_transmute)]
unsafe fn retag_same_width(t: TensorDyn, dtype: DType) -> TensorDyn {
    /// One width group: `$src` is the source variant's element type and
    /// `$same` its variant, followed by every target variant of that width.
    ///
    /// `transmute` names **both** type parameters. `<_, Tensor<$ty>>`
    /// compiled and worked but tripped
    /// `clippy::missing_transmute_annotations`, a lint that exists because
    /// an inferred source type silently follows whatever the call site
    /// later becomes -- exactly the drift a transmute must not have.
    macro_rules! group {
        ($inner:expr, $src:ty, $same:ident, $( $d:ident => $ty:ty ),+ $(,)?) => {
            match dtype {
                $(
                    // SAFETY: the caller guarantees equal element width, and
                    // same-width `Tensor<T>` are layout-identical.
                    DType::$d => TensorDyn::$d(unsafe {
                        std::mem::transmute::<Tensor<$src>, Tensor<$ty>>($inner)
                    }),
                )+
                _ => TensorDyn::$same($inner),
            }
        };
    }
    match t {
        TensorDyn::U8(i) => group!(i, u8, U8, U8 => u8, I8 => i8),
        TensorDyn::I8(i) => group!(i, i8, I8, U8 => u8, I8 => i8),
        TensorDyn::U16(i) => group!(i, u16, U16, U16 => u16, I16 => i16, F16 => f16),
        TensorDyn::I16(i) => group!(i, i16, I16, U16 => u16, I16 => i16, F16 => f16),
        TensorDyn::F16(i) => group!(i, f16, F16, U16 => u16, I16 => i16, F16 => f16),
        TensorDyn::U32(i) => group!(i, u32, U32, U32 => u32, I32 => i32, F32 => f32),
        TensorDyn::I32(i) => group!(i, i32, I32, U32 => u32, I32 => i32, F32 => f32),
        TensorDyn::F32(i) => group!(i, f32, F32, U32 => u32, I32 => i32, F32 => f32),
        TensorDyn::U64(i) => group!(i, u64, U64, U64 => u64, I64 => i64, F64 => f64),
        TensorDyn::I64(i) => group!(i, i64, I64, U64 => u64, I64 => i64, F64 => f64),
        TensorDyn::F64(i) => group!(i, f64, F64, U64 => u64, I64 => i64, F64 => f64),
    }
}

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
    /// Cross-package protocol descriptor (see [`crate::protocol`]).
    ///
    /// This is the contract between independently-linked EdgeFirst packages:
    /// they cannot share a Rust or PyO3 type, so they exchange this instead.
    /// Pin a stable host address for this tensor's data.
    ///
    /// See [`Tensor::pin_host`](crate::Tensor::pin_host). Dispatches to the
    /// concrete element type behind the erasure.
    /// Returns a `'static` pin.
    ///
    /// `TensorDyn` erases to a fixed set of concrete numeric element types, all
    /// of which are `'static`, so the keepalive genuinely can outlive any
    /// borrow — which is what lets a pin cross into a `PyCapsule` handed to
    /// Python. Eliding to `&self`'s lifetime here would make the pin escape its
    /// own producer.
    pub fn pin_host(&self, access: crate::CpuAccess) -> crate::Result<crate::HostPin<'static>> {
        dispatch!(self, pin_host, access)
    }

    /// Map this tensor's whole extent for CPU access, type-erased to raw
    /// bytes. Returns a `'static` view.
    ///
    /// `TensorDyn` has no byte-level counterpart to
    /// [`TensorTrait::map_with`](crate::TensorTrait::map_with) — every typed
    /// `Tensor<T>` maps to `HostView<'_, T>`, and a type-erased caller (a C
    /// ABI consumer, which does not know `T`) cannot use that. This dispatches
    /// exactly the way [`pin_host`](Self::pin_host) and the other `dispatch!`
    /// accessors above do — one arm per concrete element type, calling the
    /// same [`TensorTrait::map_with`](crate::TensorTrait::map_with) each typed
    /// `Tensor<T>` already implements (with its declared-vs-requested
    /// telemetry, row-stride bookkeeping, and plane-offset handling intact) —
    /// then re-views the resulting `HostView<'_, T>` as `HostView<'_, u8>` via
    /// [`HostView::into_bytes`](crate::view::HostView::into_bytes), which
    /// keeps the same underlying pin (so the same platform sync bracket still
    /// runs on `Drop`) and just restates element count as byte count.
    ///
    /// The `'static` lifetime is the same story as [`pin_host`](Self::pin_host):
    /// `map_with`'s `'a` is a free parameter bounding how long the backing
    /// allocation lives, not a borrow of `self` — the returned view shares
    /// ownership through its pin's keepalive `Arc`, never a reference into
    /// this tensor — and `TensorDyn` erases to a fixed set of concrete
    /// element types that are all `'static`, so nothing stops choosing `'a =
    /// 'static` here. Eliding to `&self`'s lifetime instead would tie the
    /// view to this call's borrow for no reason, forcing every caller (a C
    /// ABI entry point chief among them) into an artificial self-referential
    /// shape to hold onto it past this function returning.
    ///
    /// # Errors
    /// Same as `map_with`: `Error::InvalidArgument` for `CpuAccess::None`,
    /// plus whatever the concrete backend's map can fail with (capacity,
    /// stride, or platform-lock errors).
    pub fn map_bytes(
        &self,
        access: crate::CpuAccess,
    ) -> crate::Result<crate::view::HostView<'static, u8>> {
        use crate::TensorTrait;
        match self {
            Self::U8(t) => t.map_with(access).map(crate::view::HostView::into_bytes),
            Self::I8(t) => t.map_with(access).map(crate::view::HostView::into_bytes),
            Self::U16(t) => t.map_with(access).map(crate::view::HostView::into_bytes),
            Self::I16(t) => t.map_with(access).map(crate::view::HostView::into_bytes),
            Self::U32(t) => t.map_with(access).map(crate::view::HostView::into_bytes),
            Self::I32(t) => t.map_with(access).map(crate::view::HostView::into_bytes),
            Self::U64(t) => t.map_with(access).map(crate::view::HostView::into_bytes),
            Self::I64(t) => t.map_with(access).map(crate::view::HostView::into_bytes),
            Self::F16(t) => t.map_with(access).map(crate::view::HostView::into_bytes),
            Self::F32(t) => t.map_with(access).map(crate::view::HostView::into_bytes),
            Self::F64(t) => t.map_with(access).map(crate::view::HostView::into_bytes),
        }
    }

    /// Non-blocking [`map_bytes`](Self::map_bytes): returns
    /// `Err(Error::IoError(WouldBlock))` while a GPU copy the map depends on
    /// is still in flight, and makes progress on a retry. Dispatches to each
    /// typed tensor's
    /// [`TensorTrait::try_map_with`](crate::TensorTrait::try_map_with), which
    /// aliases `map_with` for every backing but the Windows D3D11 texture.
    pub fn try_map_bytes(
        &self,
        access: crate::CpuAccess,
    ) -> crate::Result<crate::view::HostView<'static, u8>> {
        use crate::TensorTrait;
        match self {
            Self::U8(t) => t
                .try_map_with(access)
                .map(crate::view::HostView::into_bytes),
            Self::I8(t) => t
                .try_map_with(access)
                .map(crate::view::HostView::into_bytes),
            Self::U16(t) => t
                .try_map_with(access)
                .map(crate::view::HostView::into_bytes),
            Self::I16(t) => t
                .try_map_with(access)
                .map(crate::view::HostView::into_bytes),
            Self::U32(t) => t
                .try_map_with(access)
                .map(crate::view::HostView::into_bytes),
            Self::I32(t) => t
                .try_map_with(access)
                .map(crate::view::HostView::into_bytes),
            Self::U64(t) => t
                .try_map_with(access)
                .map(crate::view::HostView::into_bytes),
            Self::I64(t) => t
                .try_map_with(access)
                .map(crate::view::HostView::into_bytes),
            Self::F16(t) => t
                .try_map_with(access)
                .map(crate::view::HostView::into_bytes),
            Self::F32(t) => t
                .try_map_with(access)
                .map(crate::view::HostView::into_bytes),
            Self::F64(t) => t
                .try_map_with(access)
                .map(crate::view::HostView::into_bytes),
        }
    }

    /// See [`Tensor::sync_for_cpu`](crate::Tensor::sync_for_cpu).
    pub fn sync_for_cpu(&self, access: crate::CpuAccess) -> crate::Result<()> {
        dispatch!(self, sync_for_cpu, access)
    }

    /// See [`Tensor::sync_for_device`](crate::Tensor::sync_for_device).
    pub fn sync_for_device(&self, access: crate::CpuAccess) -> crate::Result<()> {
        dispatch!(self, sync_for_device, access)
    }

    pub fn descriptor(&self) -> crate::TensorDesc {
        self.descriptor_pinned(None)
    }

    /// Descriptor carrying a pinned host address.
    ///
    /// A bare [`descriptor`](Self::descriptor) is *descriptive only*: a
    /// consumer learns the shape, format and backing kind but cannot reach the
    /// bytes. Supplying a [`HostPin`](crate::HostPin) fills in `ptr`, and the
    /// caller is responsible for keeping that pin alive at least as long as
    /// the descriptor is used — which is what the capsule keepalive does.
    pub fn descriptor_pinned(&self, pin: Option<&crate::HostPin<'_>>) -> crate::TensorDesc {
        // Native handle: the consumer re-imports zero-copy from this when
        // present. `-1` (the [`TensorDesc::handle`] "unused" sentinel)
        // covers every backend/platform combination not listed here (Mem,
        // Shm, and Dma on platforms without a dedicated arm below).
        let handle: i64 = match self.memory() {
            #[cfg(target_os = "linux")]
            TensorMemory::DmaBuf => {
                use std::os::fd::AsRawFd;
                self.dmabuf().map(|fd| fd.as_raw_fd() as i64).unwrap_or(-1)
            }
            #[cfg(any(target_os = "macos", target_os = "ios"))]
            TensorMemory::DmaBuf => self.iosurface_id().map(|id| id as i64).unwrap_or(-1),
            // The tensor's own handle, not a duplicate: the consumer's own
            // keepalive holds this tensor alive for the descriptor's whole
            // life, and a duplicate made here would have nowhere to be
            // closed.
            #[cfg(target_os = "windows")]
            TensorMemory::DmaBuf => self
                .d3d11_shared_handle_value()
                .map(|h| h as i64)
                .unwrap_or(-1),
            TensorMemory::Pbo => self.pbo_id().map(|id| id as i64).unwrap_or(-1),
            _ => -1,
        };
        // The fence is the process device's, shared by every module of this
        // process through the rendezvous, so its handle is borrowed the same
        // way the texture's is -- the device outlives the descriptor. A copy
        // that fell back to a fence of its own exports none, because the value
        // beside it may come from a copy on the shared one.
        #[cfg(target_os = "windows")]
        let (fence_handle, sync) = match self.memory() {
            TensorMemory::DmaBuf => (
                crate::d3d11::device()
                    .map(|d| d.exported_fence_handle_value())
                    .unwrap_or(0),
                Some(self.gpu_write_value()).filter(|v| *v != 0),
            ),
            _ => (0, None),
        };
        #[cfg(not(target_os = "windows"))]
        let (fence_handle, sync) = (0, None);
        crate::protocol::from_parts(crate::protocol::DescParts {
            dims: self.shape(),
            memory: self.memory(),
            dtype: self.dtype(),
            fourcc: self.format().map(|f| f.to_fourcc()).unwrap_or(0),
            format: self.format(),
            row_stride: self.row_stride(),
            handle,
            colorimetry: self.colorimetry().map(|c| c.pack()).unwrap_or(0),
            capacity: self.capacity_bytes() as u64,
            pin,
            pbo_vtable_ptr: self.pbo_vtable_ptr(),
            fence_handle,
            sync,
        })
    }

    pub fn shape(&self) -> &[usize] {
        dispatch!(self, shape)
    }

    /// Set the logical shape to any shape whose bytes fit the allocation,
    /// without [`reshape`](Self::reshape)'s equal-count constraint. See
    /// [`crate::TensorTrait::set_logical_shape`].
    pub fn set_logical_shape(&mut self, shape: &[usize]) -> crate::Result<()> {
        dispatch!(self, set_logical_shape, shape)
    }

    /// See [`Tensor::capacity_bytes`](crate::Tensor::capacity_bytes).
    pub fn capacity_bytes(&self) -> usize {
        dispatch!(self, capacity_bytes)
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

    /// Retag this tensor's element type, keeping its bytes untouched.
    ///
    /// The recorded dtype is metadata over the same allocation: `U8` and
    /// `I8` address identical bytes and differ only in how a consumer reads
    /// them. `edgefirst-image` relies on exactly that -- a PBO or DMA buffer
    /// is allocated as `u8` and handed back as `i8`, with the int8 shader
    /// applying an XOR 0x80 bias over the same buffer.
    ///
    /// Refuses a dtype of a different width. That is not a retag but a
    /// reinterpretation: `len()` and `size()` are derived from the shape and
    /// the element width, so widening or narrowing here would silently
    /// change how many elements the tensor claims to hold over an allocation
    /// whose size did not change.
    ///
    /// # Errors
    ///
    /// [`crate::Error::InvalidArgument`] when `dtype.size()` differs from
    /// the current dtype's. The tensor is left untouched.
    pub fn set_dtype(&mut self, dtype: DType) -> crate::Result<()> {
        let current = self.dtype();
        if current == dtype {
            return Ok(());
        }
        if current.size() != dtype.size() {
            return Err(crate::Error::InvalidArgument(format!(
                "set_dtype: {current:?} and {dtype:?} are different widths \
                 ({} vs {} bytes); retagging would change the element count \
                 over an allocation whose size did not change",
                current.size(),
                dtype.size()
            )));
        }
        // SAFETY: `read` duplicates the value and `write` stores exactly one
        // back, so the duplicate is never dropped twice -- provided nothing
        // between them can unwind. `retag_same_width` is a `match` of
        // `transmute`s with no panicking operation anywhere in it (its
        // width-mismatch arm returns its input unchanged rather than
        // panicking, precisely so this window has no unwind path), and
        // `self` is not observed in between.
        unsafe {
            let taken = std::ptr::read(self);
            std::ptr::write(self, retag_same_width(taken, dtype));
        }
        Ok(())
    }

    /// Return the memory allocation type.
    pub fn memory(&self) -> TensorMemory {
        dispatch!(self, memory)
    }

    /// The CPU access declared for this tensor at allocation.
    ///
    /// See [`Tensor::cpu_access`](crate::Tensor::cpu_access); dispatches to
    /// the concrete element type behind the erasure, the same way
    /// [`memory`](Self::memory) and [`dtype`](Self::dtype) do. `.writes()`
    /// on the result is the declaration query a type-erased caller (a C ABI
    /// consumer deciding whether a writable map is allowed) needs before
    /// mapping -- unlike [`TensorMapTrait::is_writable`](crate::TensorMapTrait::is_writable),
    /// which answers a different question about an *already-open* map guard
    /// ("was this particular mapping opened writable"), this answers it about
    /// the tensor itself, before any map exists.
    pub fn cpu_access(&self) -> crate::CpuAccess {
        dispatch!(self, cpu_access)
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

    /// Set the row stride in bytes without format validation. See
    /// [`Tensor::set_row_stride_unchecked`]. Added for `tensor-capi`'s
    /// `ef_tensor_set_row_stride_unchecked` (task 17): the `dynamic` backend
    /// needs a type-erased entry point the same way [`Self::set_row_stride`]
    /// already gives it one for the checked setter.
    pub fn set_row_stride_unchecked(&mut self, stride: usize) {
        dispatch!(self, set_row_stride_unchecked, stride)
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

    /// Writable counterpart of [`cuda_map`](Self::cuda_map). See
    /// [`CudaHandle::map_mut`](crate::cuda::CudaHandle::map_mut) for which
    /// backings distinguish the two.
    pub fn cuda_map_mut(&self) -> Option<crate::cuda::CudaMap<'_>> {
        dispatch!(self, cuda_map_mut)
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
    /// [`crate::Error::QuantizationInvalid`]; delegates to the typed setter for
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
        if self.memory() != TensorMemory::DmaBuf {
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
    /// Two `TensorDyn` values share a [`crate::BufferIdentity::id`] iff they were
    /// produced by cloning the same allocation (e.g. through
    /// `DmaTensor::try_clone`). Separate
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
    /// 1. Matching [`crate::BufferIdentity::id`] → same buffer (always).
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
        if self.memory() == TensorMemory::DmaBuf {
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

    /// Import an existing buffer as a type-erased tensor, taking ownership
    /// of its file descriptor. No bytes are copied.
    ///
    /// Dispatches to [`Tensor::from_fd`](crate::TensorTrait::from_fd) for
    /// `dtype` and inherits its contract in full: on Linux the backend is
    /// detected from the fd's filesystem magic — `DMA_BUF_MAGIC` imports as
    /// [`TensorMemory::DmaBuf`](crate::TensorMemory::DmaBuf), `TMPFS_MAGIC` (both
    /// `/dev/shm` and `memfd`) as [`TensorMemory::Shm`](crate::TensorMemory::Shm)
    /// — and any other filesystem is rejected rather than assumed to be
    /// shared memory. On non-Linux Unix the fd is always adopted as SHM.
    ///
    /// # Errors
    ///
    /// * [`Error::UnknownBufferType`](crate::Error::UnknownBufferType) - the
    ///   fd is neither a DMA-BUF nor tmpfs-backed; carries the observed
    ///   `fstatfs` magic. Linux only.
    /// * [`Error::UnknownDeviceType`](crate::Error::UnknownDeviceType) - the
    ///   fd lives on a real block device. Linux only.
    /// * [`Error::InvalidSize`](crate::Error::InvalidSize) - `shape` is empty
    ///   or describes zero elements.
    /// * [`Error::NixError`](crate::Error::NixError) - a syscall on the
    ///   descriptor failed.
    ///
    /// Callers that require zero-copy must check
    /// [`memory()`](crate::TensorDyn::memory) on the result rather than
    /// treating a successful import as proof of DMA backing.
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

    /// Rebuild a type-erased PBO tensor from a cross-cdylib `ops` (see
    /// [`crate::pbo::import_pbo_ops`]) plus the geometry a
    /// [`crate::TensorDesc`] under [`crate::protocol::kind::PBO`] carries.
    /// Mirrors [`Self::from_fd`]'s per-dtype dispatch shape exactly; the
    /// only caller is [`Self::import_descriptor`].
    pub(crate) fn from_pbo_import(
        buffer_id: u32,
        size: usize,
        shape: &[usize],
        dtype: DType,
        ops: std::sync::Arc<dyn crate::PboOps>,
    ) -> crate::Result<Self> {
        match dtype {
            DType::U8 => crate::PboTensor::<u8>::from_pbo(buffer_id, size, shape, None, ops)
                .and_then(|p| Tensor::from_pbo(p).map(Self::U8)),
            DType::I8 => crate::PboTensor::<i8>::from_pbo(buffer_id, size, shape, None, ops)
                .and_then(|p| Tensor::from_pbo(p).map(Self::I8)),
            DType::U16 => crate::PboTensor::<u16>::from_pbo(buffer_id, size, shape, None, ops)
                .and_then(|p| Tensor::from_pbo(p).map(Self::U16)),
            DType::I16 => crate::PboTensor::<i16>::from_pbo(buffer_id, size, shape, None, ops)
                .and_then(|p| Tensor::from_pbo(p).map(Self::I16)),
            DType::U32 => crate::PboTensor::<u32>::from_pbo(buffer_id, size, shape, None, ops)
                .and_then(|p| Tensor::from_pbo(p).map(Self::U32)),
            DType::I32 => crate::PboTensor::<i32>::from_pbo(buffer_id, size, shape, None, ops)
                .and_then(|p| Tensor::from_pbo(p).map(Self::I32)),
            DType::U64 => crate::PboTensor::<u64>::from_pbo(buffer_id, size, shape, None, ops)
                .and_then(|p| Tensor::from_pbo(p).map(Self::U64)),
            DType::I64 => crate::PboTensor::<i64>::from_pbo(buffer_id, size, shape, None, ops)
                .and_then(|p| Tensor::from_pbo(p).map(Self::I64)),
            DType::F16 => crate::PboTensor::<f16>::from_pbo(buffer_id, size, shape, None, ops)
                .and_then(|p| Tensor::from_pbo(p).map(Self::F16)),
            DType::F32 => crate::PboTensor::<f32>::from_pbo(buffer_id, size, shape, None, ops)
                .and_then(|p| Tensor::from_pbo(p).map(Self::F32)),
            DType::F64 => crate::PboTensor::<f64>::from_pbo(buffer_id, size, shape, None, ops)
                .and_then(|p| Tensor::from_pbo(p).map(Self::F64)),
        }
    }

    /// Wrap a producer's host pointer as a type-erased tensor without
    /// copying, aliasing rather than owning it.
    ///
    /// This is [`Self::from_foreign_ptr`] with `owner: None` — the consumer
    /// half of the capsule protocol's `HOST` kind
    /// ([`crate::protocol::kind::HOST`]). The descriptor's `ptr` is only
    /// meaningful while the producer's capsule keepalive is alive; nothing
    /// here takes a reference to extend that lifetime, so the caller (the
    /// capsule machinery) is responsible for it. See
    /// [`TensorDyn::import_descriptor`] for the full contract.
    ///
    /// # Safety
    ///
    /// Same as [`Self::from_foreign_ptr`]: `ptr` must be non-null, aligned to
    /// `dtype`, and valid for `shape.product()` elements of `dtype` for as
    /// long as the returned tensor — and every view/map sharing its backing
    /// — is used.
    pub unsafe fn from_raw_host(
        ptr: *mut u8,
        shape: &[usize],
        dtype: DType,
        name: Option<&str>,
    ) -> crate::Result<Self> {
        unsafe { Self::from_foreign_ptr(ptr, shape, dtype, None, name) }
    }

    /// [`Self::from_raw_host`], additionally recording that the producer's
    /// allocation holds `capacity_bytes` bytes rather than exactly
    /// `shape.product() * dtype.size()` -- the consumer half of
    /// [`crate::TensorDesc::capacity`]. See
    /// [`Tensor::from_foreign_with_capacity`](crate::Tensor::from_foreign_with_capacity)
    /// for why: a producer's pool tensor (or one padded to a decoder's
    /// MCU/pitch alignment) is larger than the shape it currently reports,
    /// and without this the imported alias would be clamped to today's
    /// shape and unable to grow back into memory the producer actually has.
    ///
    /// # Safety
    ///
    /// Same as [`Self::from_raw_host`], except `ptr` must be valid for
    /// `capacity_bytes` bytes (or the tight shape footprint, whichever is
    /// larger).
    pub unsafe fn from_raw_host_with_capacity(
        ptr: *mut u8,
        shape: &[usize],
        capacity_bytes: usize,
        dtype: DType,
        name: Option<&str>,
    ) -> crate::Result<Self> {
        unsafe {
            Self::from_foreign_ptr_with_capacity(ptr, shape, dtype, capacity_bytes, None, name)
        }
    }

    /// The per-kind construction switch [`TensorDyn::import_descriptor`]
    /// (`derived.rs`) drives, once the descriptor has been validated and
    /// its `dtype`/`shape` decoded.
    ///
    /// Backend-specific because it is the one part of the import that names
    /// *constructors*; everything around it (validation, and the
    /// format/stride/colorimetry restore afterward) is shared. See
    /// `derived.rs` for the whole function's contract -- including the
    /// borrow semantics every arm here relies on: the result *aliases* the
    /// producer's memory and nothing here takes a reference to keep it
    /// alive, except `DMABUF`, which `dup`s.
    pub(crate) fn import_storage(
        desc: &crate::TensorDesc,
        shape: &[usize],
        dtype: DType,
    ) -> crate::Result<Self> {
        match desc.kind {
            #[cfg(target_os = "linux")]
            crate::protocol::kind::DMABUF => {
                // `from_fd` takes ownership of the fd it is given, but the
                // producer still owns the original and will close it when
                // its own tensor drops -- `dup_descriptor_fd` gives this
                // import its own.
                let owned = crate::protocol::dup_descriptor_fd(desc)?;
                Self::from_fd(owned, shape, dtype, None)
            }
            #[cfg(not(target_os = "linux"))]
            crate::protocol::kind::DMABUF => Err(crate::Error::NotImplemented(
                "dma-buf import off Linux".into(),
            )),
            #[cfg(any(target_os = "macos", target_os = "ios"))]
            crate::protocol::kind::IOSURFACE => {
                let id = crate::protocol::descriptor_surface_id(desc)?;
                let surface_ref = crate::iosurface::lookup_by_id(id).ok_or_else(|| {
                    crate::Error::InvalidArgument(format!(
                        "IOSurface id {id} is not live (the producer's capsule may \
                         already have been released)"
                    ))
                })?;
                // SAFETY: `lookup_by_id` just returned a live IOSurfaceRef for
                // this id. `from_iosurface` (`OwnedIoSurface::from_external`)
                // takes its own independent CFRetain, so the lookup's +1
                // reference is released right after regardless of outcome.
                let result = unsafe { Self::from_iosurface(surface_ref, shape, dtype, None) };
                crate::iosurface::release(surface_ref);
                result
            }
            #[cfg(not(any(target_os = "macos", target_os = "ios")))]
            crate::protocol::kind::IOSURFACE => Err(crate::Error::NotImplemented(
                "IOSurface import off Apple platforms".into(),
            )),
            crate::protocol::kind::HOST => {
                crate::protocol::check_descriptor_host_ptr(desc)?;
                // SAFETY: the caller guarantees the producer's keepalive
                // outlives the returned tensor -- that is the capsule
                // contract `import_descriptor` documents. `desc.capacity`
                // comes from the same trusted producer as `ptr`/`shape`, so
                // extending trust to it is not a new hazard; see
                // `from_raw_host_with_capacity`'s docs.
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
                // SAFETY: the caller guarantees the producer's keepalive
                // outlives the returned tensor -- the same capsule contract
                // `import_descriptor`'s own doc comment already establishes
                // for `ptr`, extended here to what `ptr` means under
                // `kind::PBO` (see `TensorDesc::ptr`'s own doc comment).
                // `desc.ptr.0` came from the same trusted producer as
                // `desc.handle`, so interpreting it as a `PboOpsVtable*` is
                // not a new hazard relative to the `HOST` arm's own use of
                // `desc.ptr`.
                let ops = unsafe { crate::pbo::import_pbo_ops(desc.ptr.0 as *const _)? };
                Self::from_pbo_import(buffer_id, desc.capacity as usize, shape, dtype, ops)
            }
            #[cfg(target_os = "windows")]
            crate::protocol::kind::D3D11_TEXTURE => {
                let (tex, completion) = crate::protocol::descriptor_d3d11_handles(desc)?;
                // SAFETY: the caller guarantees the producer's keepalive
                // outlives the returned tensor -- the capsule contract
                // `import_descriptor` documents -- so both handles are the
                // producer's own and live for this call. `ReadWrite` is the
                // widest access an import can ask for; the descriptor
                // carries no access of its own.
                unsafe {
                    let (format, width, height) =
                        crate::protocol::descriptor_d3d11_geometry(desc, tex, shape)?;
                    Self::from_d3d11_shared_handle(
                        tex,
                        width,
                        height,
                        format,
                        dtype,
                        crate::CpuAccess::ReadWrite,
                        completion,
                        None,
                    )
                }
            }
            #[cfg(not(target_os = "windows"))]
            crate::protocol::kind::D3D11_TEXTURE => Err(crate::Error::NotImplemented(
                "D3D11 texture import off Windows".into(),
            )),
            k => Err(crate::Error::NotImplemented(format!(
                "tensor interop kind {k} cannot be imported by this build"
            ))),
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
        // SAFETY: this function's own contract (see its doc comment) is
        // exactly `Tensor::from_foreign`'s contract, uniformly across every
        // dtype arm below -- `ptr`/`shape`/`owner`/`name` are unchanged, only
        // the element type dispatched on `dtype` differs.
        unsafe {
            match dtype {
                DType::U8 => {
                    Tensor::<u8>::from_foreign(ptr.cast(), shape, owner, name).map(Self::U8)
                }
                DType::I8 => {
                    Tensor::<i8>::from_foreign(ptr.cast(), shape, owner, name).map(Self::I8)
                }
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
    }

    /// [`Self::from_foreign_ptr`], additionally recording that the
    /// allocation behind `ptr` holds `capacity_bytes` bytes rather than
    /// exactly `shape.product() * dtype.size()`. See
    /// [`Tensor::from_foreign_with_capacity`](crate::Tensor::from_foreign_with_capacity).
    ///
    /// # Safety
    ///
    /// Same as [`Self::from_foreign_ptr`], except `ptr` must be valid for
    /// `capacity_bytes` bytes (or the tight shape footprint, whichever is
    /// larger).
    pub unsafe fn from_foreign_ptr_with_capacity(
        ptr: *mut u8,
        shape: &[usize],
        dtype: DType,
        capacity_bytes: usize,
        owner: Option<crate::ForeignOwner>,
        name: Option<&str>,
    ) -> crate::Result<Self> {
        macro_rules! import {
            ($t:ty, $variant:ident) => {
                // SAFETY: this function's own contract (see its doc comment)
                // is exactly `Tensor::from_foreign_with_capacity`'s
                // contract, uniformly across every dtype this macro is
                // invoked for below.
                unsafe {
                    Tensor::<$t>::from_foreign_with_capacity(
                        ptr.cast(),
                        shape,
                        capacity_bytes,
                        owner,
                        name,
                    )
                }
                .map(Self::$variant)
            };
        }
        match dtype {
            DType::U8 => import!(u8, U8),
            DType::I8 => import!(i8, I8),
            DType::U16 => import!(u16, U16),
            DType::I16 => import!(i16, I16),
            DType::U32 => import!(u32, U32),
            DType::I32 => import!(i32, I32),
            DType::U64 => import!(u64, U64),
            DType::I64 => import!(i64, I64),
            DType::F16 => import!(f16, F16),
            DType::F32 => import!(f32, F32),
            DType::F64 => import!(f64, F64),
        }
    }

    /// Wrap an IOSurface named by its global ID (macOS/iOS only).
    ///
    /// The IOSurface counterpart to [`from_fd`](Self::from_fd): an ID is the
    /// shareable handle on this platform, the way a dma-buf fd is on Linux, so
    /// this is what a consumer needs when a tensor arrives from another library
    /// or process carrying only the ID.
    ///
    /// Returns an error when no live surface has that ID — IDs are reused after
    /// the surface is freed, so a stale one must fail rather than resolve to an
    /// unrelated buffer.
    #[cfg(any(target_os = "macos", target_os = "ios"))]
    pub fn from_iosurface_id(
        id: u32,
        shape: &[usize],
        dtype: DType,
        name: Option<&str>,
    ) -> crate::Result<Self> {
        let surface = crate::iosurface::lookup_by_id(id).ok_or_else(|| {
            crate::Error::InvalidArgument(format!("no live IOSurface with id {id}"))
        })?;
        // SAFETY: `lookup_by_id` returned a live (+1) reference;
        // `from_iosurface` (`OwnedIoSurface::from_external`) takes its own
        // independent `CFRetain`, so the lookup's reference is ours to give
        // back and must be given back on BOTH paths -- `crate::iosurface`'s
        // module docs state the rule without hedging ("every call site pairs
        // a successful lookup with `release` once it has taken, or failed to
        // take, its own independent retain"), and this function was the one
        // call site that did not. The leak was one full surface -- and its
        // whole pixel allocation -- per import that never returns to the
        // producer's baseline, so a per-frame capsule hand-off leaked a
        // frame buffer per frame. Unreached until task P2a routed the
        // `dynamic` backend's entire `kind::IOSURFACE` capsule import
        // through here (`ef_tensor_from_iosurface_id`); `blob.rs`'s
        // deserializer was the only prior caller and leaked the same way.
        let result = unsafe { Self::from_iosurface(surface, shape, dtype, name) };
        crate::iosurface::release(surface);
        result
    }

    /// Wrap an externally-allocated IOSurface as a type-erased tensor
    /// (macOS/iOS only).
    ///
    /// # Safety
    ///
    /// `surface_ref` must be a valid live `IOSurfaceRef`. `shape` must
    /// match the IOSurface's pixel dimensions and chosen element type.
    #[cfg(any(target_os = "macos", target_os = "ios"))]
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

    /// IOSurfaceID for cross-process surface sharing (macOS/iOS only).
    /// Returns `None` when the tensor is not IOSurface-backed.
    #[cfg(any(target_os = "macos", target_os = "ios"))]
    pub fn iosurface_id(&self) -> Option<u32> {
        dispatch!(self, iosurface_id)
    }

    /// GL buffer ID for this PBO. Returns `None` when the tensor is not
    /// PBO-backed.
    pub fn pbo_id(&self) -> Option<u32> {
        dispatch!(self, pbo_id)
    }

    /// The C-ABI `PboOpsVtable` address for this PBO, for cross-cdylib
    /// export via [`crate::TensorDesc::ptr`] (see that field's doc
    /// comment). Returns `None` when the tensor is not PBO-backed.
    pub fn pbo_vtable_ptr(&self) -> Option<*const std::ffi::c_void> {
        dispatch!(self, pbo_vtable_ptr)
    }

    /// A type-erased keepalive that must stay alive for at least as long as
    /// [`Self::pbo_vtable_ptr`]'s address is used. `None` when the tensor is
    /// not PBO-backed. See [`crate::pbo::PboTensor::pbo_keepalive`]'s own
    /// doc comment.
    pub fn pbo_keepalive(&self) -> Option<std::sync::Arc<dyn Send + Sync>> {
        dispatch!(self, pbo_keepalive)
    }

    /// Borrow the raw `IOSurfaceRef` backing this tensor (macOS/iOS
    /// only). Returns `None` when the tensor is not IOSurface-backed.
    /// The pointer's lifetime is tied to `self`.
    #[cfg(any(target_os = "macos", target_os = "ios"))]
    pub fn iosurface_ref(&self) -> Option<*mut std::ffi::c_void> {
        dispatch!(self, iosurface_ref)
    }

    /// Physical IOSurface dimensions in texels, independent of the logical
    /// shape (macOS/iOS only). `None` when not IOSurface-backed. The GL
    /// backend binds the EGL pbuffer at these dims so one cached pbuffer
    /// serves every frame size a reused pool surface holds.
    #[cfg(any(target_os = "macos", target_os = "ios"))]
    pub fn iosurface_physical_dims(&self) -> Option<(usize, usize)> {
        dispatch!(self, iosurface_physical_dims)
    }

    /// The borrowed `ID3D11Texture2D*` backing this tensor (Windows only).
    /// `None` when the tensor is not texture-backed.
    #[cfg(target_os = "windows")]
    pub fn d3d11_texture(&self) -> Option<*mut std::ffi::c_void> {
        dispatch!(self, d3d11_texture)
    }

    /// The texture geometry the HAL chose for this image (Windows only).
    /// `None` when the tensor is not texture-backed.
    #[cfg(target_os = "windows")]
    pub fn d3d11_layout(&self) -> Option<crate::d3d11_layout::D3d11ImageLayout> {
        dispatch!(self, d3d11_layout)
    }

    /// A duplicated NT handle the caller owns (Windows only). See
    /// [`Tensor::d3d11_shared_handle`](crate::Tensor::d3d11_shared_handle).
    #[cfg(target_os = "windows")]
    pub fn d3d11_shared_handle(&self) -> crate::Result<std::os::windows::io::OwnedHandle> {
        dispatch!(self, d3d11_shared_handle)
    }

    /// The fence handle plus value a GPU consumer waits on before reading this
    /// texture (Windows only), or `None` when no GPU write has been recorded.
    #[cfg(target_os = "windows")]
    pub fn gpu_completion(&self) -> crate::Result<Option<crate::d3d11::GpuCompletion>> {
        dispatch!(self, gpu_completion)
    }

    /// The fence value of the newest GPU write recorded on this tensor, or 0
    /// when there is none; every platform, 0 off Windows. See
    /// [`Tensor::gpu_write_value`](crate::Tensor::gpu_write_value).
    pub fn gpu_write_value(&self) -> u64 {
        dispatch!(self, gpu_write_value)
    }

    /// Record that GPU work writing this texture completes at `value` of the
    /// process device's fence (Windows only). Takes `&self` -- see
    /// [`Tensor::set_gpu_write`](crate::Tensor::set_gpu_write).
    #[cfg(target_os = "windows")]
    pub fn set_gpu_write(&self, value: u64) -> crate::Result<()> {
        dispatch!(self, set_gpu_write, value)
    }

    /// The tensor's own NT handle value -- not a duplicate -- for descriptors
    /// whose consumer keeps the producing tensor alive (Windows only).
    #[cfg(target_os = "windows")]
    pub(crate) fn d3d11_shared_handle_value(&self) -> Option<usize> {
        dispatch!(self, d3d11_shared_handle_value)
    }

    /// Wrap an existing `ID3D11Texture2D` as a type-erased tensor (Windows
    /// only). See [`Tensor::from_d3d11_texture`](crate::Tensor::from_d3d11_texture).
    ///
    /// # Safety
    ///
    /// `texture` must be null or a live `ID3D11Texture2D` created on the HAL
    /// device ([`crate::d3d11::device()`]).
    #[cfg(target_os = "windows")]
    #[allow(clippy::too_many_arguments)] // one image description, spelled out
    pub unsafe fn from_d3d11_texture(
        texture: *mut std::ffi::c_void,
        width: usize,
        height: usize,
        format: PixelFormat,
        dtype: DType,
        access: crate::CpuAccess,
        name: Option<&str>,
    ) -> crate::Result<Self> {
        // SAFETY: the caller guarantees `texture` is null or a live texture on
        // the HAL device; the typed constructor takes its own reference.
        unsafe {
            match dtype {
                DType::U8 => {
                    Tensor::<u8>::from_d3d11_texture(texture, width, height, format, access, name)
                        .map(Self::U8)
                }
                DType::I8 => {
                    Tensor::<i8>::from_d3d11_texture(texture, width, height, format, access, name)
                        .map(Self::I8)
                }
                DType::U16 => {
                    Tensor::<u16>::from_d3d11_texture(texture, width, height, format, access, name)
                        .map(Self::U16)
                }
                DType::I16 => {
                    Tensor::<i16>::from_d3d11_texture(texture, width, height, format, access, name)
                        .map(Self::I16)
                }
                DType::U32 => {
                    Tensor::<u32>::from_d3d11_texture(texture, width, height, format, access, name)
                        .map(Self::U32)
                }
                DType::I32 => {
                    Tensor::<i32>::from_d3d11_texture(texture, width, height, format, access, name)
                        .map(Self::I32)
                }
                DType::U64 => {
                    Tensor::<u64>::from_d3d11_texture(texture, width, height, format, access, name)
                        .map(Self::U64)
                }
                DType::I64 => {
                    Tensor::<i64>::from_d3d11_texture(texture, width, height, format, access, name)
                        .map(Self::I64)
                }
                DType::F16 => {
                    Tensor::<f16>::from_d3d11_texture(texture, width, height, format, access, name)
                        .map(Self::F16)
                }
                DType::F32 => {
                    Tensor::<f32>::from_d3d11_texture(texture, width, height, format, access, name)
                        .map(Self::F32)
                }
                DType::F64 => {
                    Tensor::<f64>::from_d3d11_texture(texture, width, height, format, access, name)
                        .map(Self::F64)
                }
            }
        }
    }

    /// Open a shared texture by its NT handle as a type-erased tensor (Windows
    /// only). See
    /// [`Tensor::from_d3d11_shared_handle`](crate::Tensor::from_d3d11_shared_handle).
    ///
    /// # Safety
    ///
    /// `handle` must be an NT shared handle of a D3D11 texture, valid in this
    /// process, and `completion`'s handle a shared fence handle.
    #[cfg(target_os = "windows")]
    #[allow(clippy::too_many_arguments)] // one image description, spelled out
    pub unsafe fn from_d3d11_shared_handle(
        handle: std::os::windows::io::RawHandle,
        width: usize,
        height: usize,
        format: PixelFormat,
        dtype: DType,
        access: crate::CpuAccess,
        completion: Option<(std::os::windows::io::RawHandle, u64)>,
        name: Option<&str>,
    ) -> crate::Result<Self> {
        // SAFETY: the caller guarantees the handles are valid in this process;
        // the typed constructor duplicates what it keeps.
        unsafe {
            match dtype {
                DType::U8 => Tensor::<u8>::from_d3d11_shared_handle(
                    handle, width, height, format, access, completion, name,
                )
                .map(Self::U8),
                DType::I8 => Tensor::<i8>::from_d3d11_shared_handle(
                    handle, width, height, format, access, completion, name,
                )
                .map(Self::I8),
                DType::U16 => Tensor::<u16>::from_d3d11_shared_handle(
                    handle, width, height, format, access, completion, name,
                )
                .map(Self::U16),
                DType::I16 => Tensor::<i16>::from_d3d11_shared_handle(
                    handle, width, height, format, access, completion, name,
                )
                .map(Self::I16),
                DType::U32 => Tensor::<u32>::from_d3d11_shared_handle(
                    handle, width, height, format, access, completion, name,
                )
                .map(Self::U32),
                DType::I32 => Tensor::<i32>::from_d3d11_shared_handle(
                    handle, width, height, format, access, completion, name,
                )
                .map(Self::I32),
                DType::U64 => Tensor::<u64>::from_d3d11_shared_handle(
                    handle, width, height, format, access, completion, name,
                )
                .map(Self::U64),
                DType::I64 => Tensor::<i64>::from_d3d11_shared_handle(
                    handle, width, height, format, access, completion, name,
                )
                .map(Self::I64),
                DType::F16 => Tensor::<f16>::from_d3d11_shared_handle(
                    handle, width, height, format, access, completion, name,
                )
                .map(Self::F16),
                DType::F32 => Tensor::<f32>::from_d3d11_shared_handle(
                    handle, width, height, format, access, completion, name,
                )
                .map(Self::F32),
                DType::F64 => Tensor::<f64>::from_d3d11_shared_handle(
                    handle, width, height, format, access, completion, name,
                )
                .map(Self::F64),
            }
        }
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

    /// Copy the tensor's logical bytes into `dst`, compacting away any
    /// recorded row-stride padding — see [`Tensor::copy_to_flat`] for the
    /// full contract. `dst.len()` must equal the tight byte footprint
    /// (`shape` product × element size).
    pub fn copy_to_flat(&self, dst: &mut [u8]) -> crate::Result<()> {
        dispatch!(self, copy_to_flat, dst)
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
        access: crate::CpuAccess,
    ) -> crate::Result<Self> {
        match dtype {
            DType::U8 => Tensor::<u8>::image(width, height, format, memory, access).map(Self::U8),
            DType::I8 => Tensor::<i8>::image(width, height, format, memory, access).map(Self::I8),
            DType::U16 => {
                Tensor::<u16>::image(width, height, format, memory, access).map(Self::U16)
            }
            DType::I16 => {
                Tensor::<i16>::image(width, height, format, memory, access).map(Self::I16)
            }
            DType::U32 => {
                Tensor::<u32>::image(width, height, format, memory, access).map(Self::U32)
            }
            DType::I32 => {
                Tensor::<i32>::image(width, height, format, memory, access).map(Self::I32)
            }
            DType::U64 => {
                Tensor::<u64>::image(width, height, format, memory, access).map(Self::U64)
            }
            DType::I64 => {
                Tensor::<i64>::image(width, height, format, memory, access).map(Self::I64)
            }
            DType::F16 => {
                Tensor::<f16>::image(width, height, format, memory, access).map(Self::F16)
            }
            DType::F32 => {
                Tensor::<f32>::image(width, height, format, memory, access).map(Self::F32)
            }
            DType::F64 => {
                Tensor::<f64>::image(width, height, format, memory, access).map(Self::F64)
            }
        }
    }

    /// Allocate an image tensor from a declarative [`crate::ImageDesc`]
    /// request — dispatching on `desc.dtype()`. See
    /// [`Tensor::image_desc`] for the compression-request semantics.
    pub fn image_desc(desc: &crate::ImageDesc) -> crate::Result<Self> {
        match desc.dtype() {
            DType::U8 => Tensor::<u8>::image_desc(desc).map(Self::U8),
            DType::I8 => Tensor::<i8>::image_desc(desc).map(Self::I8),
            DType::U16 => Tensor::<u16>::image_desc(desc).map(Self::U16),
            DType::I16 => Tensor::<i16>::image_desc(desc).map(Self::I16),
            DType::U32 => Tensor::<u32>::image_desc(desc).map(Self::U32),
            DType::I32 => Tensor::<i32>::image_desc(desc).map(Self::I32),
            DType::U64 => Tensor::<u64>::image_desc(desc).map(Self::U64),
            DType::I64 => Tensor::<i64>::image_desc(desc).map(Self::I64),
            DType::F16 => Tensor::<f16>::image_desc(desc).map(Self::F16),
            DType::F32 => Tensor::<f32>::image_desc(desc).map(Self::F32),
            DType::F64 => Tensor::<f64>::image_desc(desc).map(Self::F64),
        }
    }

    /// The recorded vendor tile-compression scheme (see
    /// [`Tensor::compression`]).
    pub fn compression(&self) -> Option<crate::CompressionScheme> {
        dispatch!(self, compression)
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
    /// use edgefirst_tensor::{CpuAccess, TensorDyn, PixelFormat, DType, TensorMemory};
    /// # fn main() -> edgefirst_tensor::Result<()> {
    /// // Allocate a 3004×1688 RGBA8 canvas with 64-byte pitch alignment
    /// // (12032 bytes per row instead of the natural 12016).
    /// let img = TensorDyn::image_with_stride(
    ///     3004, 1688,
    ///     PixelFormat::Rgba, DType::U8,
    ///     12032,
    ///     Some(TensorMemory::DmaBuf),
    ///     CpuAccess::ReadWrite,
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
        access: crate::CpuAccess,
    ) -> crate::Result<Self> {
        match dtype {
            DType::U8 => Tensor::<u8>::image_with_stride(
                width,
                height,
                format,
                row_stride_bytes,
                memory,
                access,
            )
            .map(Self::U8),
            DType::I8 => Tensor::<i8>::image_with_stride(
                width,
                height,
                format,
                row_stride_bytes,
                memory,
                access,
            )
            .map(Self::I8),
            DType::U16 => Tensor::<u16>::image_with_stride(
                width,
                height,
                format,
                row_stride_bytes,
                memory,
                access,
            )
            .map(Self::U16),
            DType::I16 => Tensor::<i16>::image_with_stride(
                width,
                height,
                format,
                row_stride_bytes,
                memory,
                access,
            )
            .map(Self::I16),
            DType::U32 => Tensor::<u32>::image_with_stride(
                width,
                height,
                format,
                row_stride_bytes,
                memory,
                access,
            )
            .map(Self::U32),
            DType::I32 => Tensor::<i32>::image_with_stride(
                width,
                height,
                format,
                row_stride_bytes,
                memory,
                access,
            )
            .map(Self::I32),
            DType::U64 => Tensor::<u64>::image_with_stride(
                width,
                height,
                format,
                row_stride_bytes,
                memory,
                access,
            )
            .map(Self::U64),
            DType::I64 => Tensor::<i64>::image_with_stride(
                width,
                height,
                format,
                row_stride_bytes,
                memory,
                access,
            )
            .map(Self::I64),
            DType::F16 => Tensor::<f16>::image_with_stride(
                width,
                height,
                format,
                row_stride_bytes,
                memory,
                access,
            )
            .map(Self::F16),
            DType::F32 => Tensor::<f32>::image_with_stride(
                width,
                height,
                format,
                row_stride_bytes,
                memory,
                access,
            )
            .map(Self::F32),
            DType::F64 => Tensor::<f64>::image_with_stride(
                width,
                height,
                format,
                row_stride_bytes,
                memory,
                access,
            )
            .map(Self::F64),
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

/// The C-facing handle pointer type. Declared in both backends with the same
/// name, so a `-capi` crate's signatures do not change with the backend.
pub type Raw = *mut std::ffi::c_void;

impl TensorDyn {
    /// Give up ownership, yielding a raw handle the caller must return to
    /// [`TensorDyn::from_raw`] or leak.
    pub fn into_raw(self) -> Raw {
        Box::into_raw(Box::new(self)) as Raw
    }

    /// Retake ownership of a handle produced by [`TensorDyn::into_raw`].
    ///
    /// # Safety
    /// `p` must have come from `into_raw` in this build and must not have
    /// been passed here before.
    pub unsafe fn from_raw(p: Raw) -> TensorDyn {
        *unsafe { Box::from_raw(p as *mut TensorDyn) }
    }

    /// Borrow a `TensorDyn` from a raw handle for the duration of `f`.
    /// `ManuallyDrop` prevents ownership — and the destructor — being taken.
    ///
    /// # Safety
    /// `p` must be a live handle from `into_raw` in this build.
    pub unsafe fn with_raw<R>(p: Raw, f: impl FnOnce(&mut TensorDyn) -> R) -> R {
        let mut t = std::mem::ManuallyDrop::new(unsafe { std::ptr::read(p as *mut TensorDyn) });
        f(&mut t)
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
    fn map_bytes_roundtrips_and_is_typed_agnostic() {
        use crate::TensorMapTrait;
        // f32: exercises the byte-length-vs-element-length distinction
        // `into_bytes` has to get right (4 elements, but 16 bytes).
        let t = Tensor::<f32>::new(&[4], None, None).unwrap();
        let dyn_t: TensorDyn = t.into();
        {
            let mut view = dyn_t.map_bytes(crate::CpuAccess::ReadWrite).unwrap();
            assert_eq!(view.len(), 16, "4 x f32 is 16 bytes, not 4");
            view.as_mut_slice()
                .copy_from_slice(&[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]);
        }
        let view = dyn_t.map_bytes(crate::CpuAccess::Read).unwrap();
        assert_eq!(
            view.as_slice(),
            &[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
        );
    }

    #[test]
    fn map_bytes_none_is_invalid() {
        let t = Tensor::<u8>::new(&[4], None, None).unwrap();
        let dyn_t: TensorDyn = t.into();
        assert!(matches!(
            dyn_t.map_bytes(crate::CpuAccess::None),
            Err(crate::error::Error::InvalidArgument(_))
        ));
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
        let t = Tensor::<u8>::image(
            640,
            480,
            PixelFormat::Rgba,
            None,
            crate::CpuAccess::ReadWrite,
        )
        .unwrap();
        let dyn_t: TensorDyn = t.into();
        assert_eq!(dyn_t.format(), Some(PixelFormat::Rgba));
        assert_eq!(dyn_t.width(), Some(640));
        assert_eq!(dyn_t.height(), Some(480));
        assert!(!dyn_t.is_multiplane());
    }

    #[test]
    fn cpu_access_reports_the_declared_access() {
        // A Read-declared image tensor reports Read...
        let read_t =
            Tensor::<u8>::image(4, 4, PixelFormat::Rgba, None, crate::CpuAccess::Read).unwrap();
        let dyn_read: TensorDyn = read_t.into();
        assert_eq!(dyn_read.cpu_access(), crate::CpuAccess::Read);
        assert!(!dyn_read.cpu_access().writes());

        // ...and a plain mem tensor reports whatever the typed constructor
        // actually defaults to (ReadWrite -- `Tensor::new`'s historical
        // implicit behavior, per `CpuAccess::ReadWrite`'s own doc comment).
        let mem_t = Tensor::<u8>::new(&[4], None, None).unwrap();
        let dyn_mem: TensorDyn = mem_t.into();
        assert_eq!(dyn_mem.cpu_access(), crate::CpuAccess::ReadWrite);
        assert!(dyn_mem.cpu_access().writes());
    }

    #[test]
    fn image_constructor() {
        let dyn_t = TensorDyn::image(
            640,
            480,
            PixelFormat::Rgb,
            DType::U8,
            None,
            crate::CpuAccess::ReadWrite,
        )
        .unwrap();
        assert_eq!(dyn_t.dtype(), DType::U8);
        assert_eq!(dyn_t.format(), Some(PixelFormat::Rgb));
        assert_eq!(dyn_t.width(), Some(640));
    }

    #[test]
    fn image_constructor_i8() {
        let dyn_t = TensorDyn::image(
            640,
            480,
            PixelFormat::Rgb,
            DType::I8,
            None,
            crate::CpuAccess::ReadWrite,
        )
        .unwrap();
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
        let mut t = TensorDyn::image(
            100,
            100,
            PixelFormat::Rgba,
            DType::U8,
            None,
            crate::CpuAccess::ReadWrite,
        )
        .unwrap();
        t.set_row_stride(512).unwrap();
        assert_eq!(t.row_stride(), Some(512));
        assert_eq!(t.effective_row_stride(), Some(512));
    }

    #[test]
    fn set_row_stride_equals_min() {
        // RGB 100px: min stride = 300, set exactly 300
        let mut t = TensorDyn::image(
            100,
            100,
            PixelFormat::Rgb,
            DType::U8,
            None,
            crate::CpuAccess::ReadWrite,
        )
        .unwrap();
        t.set_row_stride(300).unwrap();
        assert_eq!(t.row_stride(), Some(300));
    }

    #[test]
    fn set_row_stride_too_small() {
        // RGBA 64px (a 64-aligned width: 64*4 = 256, already a multiple of 64)
        // carries no implicit stride. min stride = 256; setting 200 must error
        // and leave row_stride unset. (Non-64-aligned widths now record the
        // padded stride at allocation — see `Tensor::image`.)
        let mut t = TensorDyn::image(
            64,
            100,
            PixelFormat::Rgba,
            DType::U8,
            None,
            crate::CpuAccess::ReadWrite,
        )
        .unwrap();
        assert!(t.set_row_stride(200).is_err());
        assert_eq!(t.row_stride(), None);
    }

    #[test]
    fn set_row_stride_zero() {
        let mut t = TensorDyn::image(
            100,
            100,
            PixelFormat::Rgb,
            DType::U8,
            None,
            crate::CpuAccess::ReadWrite,
        )
        .unwrap();
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
        let t = TensorDyn::image(
            64,
            100,
            PixelFormat::Rgb,
            DType::U8,
            None,
            crate::CpuAccess::ReadWrite,
        )
        .unwrap();
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
            Some(TensorMemory::DmaBuf),
            crate::CpuAccess::ReadWrite,
        ) {
            Ok(t) if t.memory() == TensorMemory::DmaBuf => t,
            _ => return,
        };
        // 100px * 3 bytes = 300, rounded up to the 64-byte alignment DMA-BUF
        // imports require -> 320. That is a DMA-BUF property, not a universal
        // one: Android's TensorMemory::DmaBuf is an AHardwareBuffer, whose
        // allocator reports no intrinsic pitch here (row_stride() is None), so
        // asserting 320 there tests the wrong platform's invariant.
        if cfg!(target_os = "android") {
            assert!(
                t.row_stride().is_none() || t.row_stride() >= Some(300),
                "AHardwareBuffer pitch must be absent or at least the tight row"
            );
            return;
        }
        // Windows' TensorMemory::DmaBuf is a D3D11 texture, whose pitch is the
        // driver's staging row pitch (384 on the RTX 3070 for this 300-byte
        // row) and is recorded only when it exceeds the tight row -- so, as on
        // Android, 320 is the wrong platform's invariant.
        if cfg!(target_os = "windows") {
            assert!(
                t.row_stride().is_none() || t.row_stride() >= Some(300),
                "D3D11 staging pitch must be absent or at least the tight row"
            );
            assert!(t.effective_row_stride() >= Some(300));
            return;
        }
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
        let t = TensorDyn::image(
            100,
            100,
            PixelFormat::Rgba,
            DType::U8,
            None,
            crate::CpuAccess::ReadWrite,
        )
        .unwrap()
        .with_row_stride(512)
        .unwrap();
        assert_eq!(t.row_stride(), Some(512));
        assert_eq!(t.effective_row_stride(), Some(512));
    }

    #[test]
    fn with_row_stride_rejects_small() {
        let result = TensorDyn::image(
            100,
            100,
            PixelFormat::Rgba,
            DType::U8,
            None,
            crate::CpuAccess::ReadWrite,
        )
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
        let mut t = TensorDyn::image(
            100,
            100,
            PixelFormat::Rgb,
            DType::U8,
            None,
            crate::CpuAccess::ReadWrite,
        )
        .unwrap();
        t.set_row_stride(512).unwrap();
        // Re-setting the same format should not clear stride
        t.set_format(PixelFormat::Rgb).unwrap();
        assert_eq!(t.row_stride(), Some(512));
    }

    #[test]
    fn effective_row_stride_planar() {
        let t = TensorDyn::image(
            640,
            480,
            PixelFormat::PlanarRgb,
            DType::U8,
            None,
            crate::CpuAccess::ReadWrite,
        )
        .unwrap();
        assert_eq!(t.effective_row_stride(), Some(640)); // planar: width only
    }

    #[test]
    fn effective_row_stride_nv12() {
        let t = TensorDyn::image(
            640,
            480,
            PixelFormat::Nv12,
            DType::U8,
            None,
            crate::CpuAccess::ReadWrite,
        )
        .unwrap();
        assert_eq!(t.effective_row_stride(), Some(640)); // semi-planar: width only
    }

    #[test]
    fn map_rejects_strided_tensor() {
        let mut t = Tensor::<u8>::image(
            100,
            100,
            PixelFormat::Rgba,
            Some(TensorMemory::Mem),
            crate::CpuAccess::ReadWrite,
        )
        .unwrap();
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
        let t = TensorDyn::image(
            100,
            100,
            PixelFormat::Rgba,
            DType::U8,
            None,
            crate::CpuAccess::ReadWrite,
        )
        .unwrap();
        assert_eq!(t.plane_offset(), None);
    }

    #[test]
    fn set_plane_offset_basic() {
        let mut t = TensorDyn::image(
            100,
            100,
            PixelFormat::Rgba,
            DType::U8,
            None,
            crate::CpuAccess::ReadWrite,
        )
        .unwrap();
        t.set_plane_offset(4096);
        assert_eq!(t.plane_offset(), Some(4096));
    }

    #[test]
    fn set_plane_offset_zero() {
        let mut t = TensorDyn::image(
            100,
            100,
            PixelFormat::Rgb,
            DType::U8,
            None,
            crate::CpuAccess::ReadWrite,
        )
        .unwrap();
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
        let mut t = Tensor::<u8>::image(
            100,
            100,
            PixelFormat::Rgba,
            Some(TensorMemory::Mem),
            crate::CpuAccess::ReadWrite,
        )
        .unwrap();
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
        let parent = Tensor::<u8>::image(
            100,
            100,
            PixelFormat::Rgba,
            Some(TensorMemory::Mem),
            crate::CpuAccess::ReadWrite,
        )
        .unwrap();
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
        let mut t = Tensor::<u8>::image(
            100,
            100,
            PixelFormat::Rgba,
            Some(TensorMemory::Mem),
            crate::CpuAccess::ReadWrite,
        )
        .unwrap();
        t.set_plane_offset(0);
        // Zero offset is fine for CPU mapping
        assert!(t.map().is_ok());
    }

    #[test]
    fn dyn_configure_image_nv12() {
        let mut t = TensorDyn::image(
            640,
            480,
            PixelFormat::Rgb,
            DType::U8,
            None,
            crate::CpuAccess::ReadWrite,
        )
        .unwrap();
        t.configure_image(320, 240, PixelFormat::Nv12).unwrap();
        assert_eq!(t.format(), Some(PixelFormat::Nv12));
        assert_eq!((t.width(), t.height()), (Some(320), Some(240)));
    }

    #[test]
    fn tensordyn_colorimetry_roundtrip() {
        use crate::{ColorEncoding, Colorimetry, DType, PixelFormat};
        let mut t = TensorDyn::image(
            1280,
            720,
            PixelFormat::Nv12,
            DType::U8,
            None,
            crate::CpuAccess::ReadWrite,
        )
        .unwrap();
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
