// SPDX-FileCopyrightText: Copyright 2025 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

/*!
EdgeFirst HAL - Tensor Module

The `edgefirst_tensor` crate provides a unified interface for managing multi-dimensional arrays (tensors)
across four memory backends: the platform's native zero-copy GPU buffer, POSIX shared memory, the system
heap, and an OpenGL Pixel Buffer Object. The crate defines traits and structures for creating, reshaping,
and mapping tensors into memory.

## Examples
```rust
use edgefirst_tensor::{Error, Tensor, TensorMemory, TensorTrait};
# fn main() -> Result<(), Error> {
let tensor = Tensor::<f32>::new(&[2, 3, 4], Some(TensorMemory::Mem), Some("test_tensor"))?;
assert_eq!(tensor.memory(), TensorMemory::Mem);
assert_eq!(tensor.name(), "test_tensor");
#    Ok(())
# }
```

## Overview
The main structures and traits provided by the `edgefirst_tensor` crate are [`TensorTrait`] and
[`TensorMapTrait`], which define the behavior of Tensors and their memory mappings, respectively.
The [`Tensor<T>`] struct wraps a backend-specific storage with optional image format metadata
([`PixelFormat`]); mapping a tensor yields a [`HostView`], which implements
[`TensorMapTrait`] and derefs to the underlying elements. The
[`TensorDyn`] type-erased enum wraps `Tensor<T>` for runtime element-type dispatch.

[`TensorMemory::DmaBuf`] is one variant with three implementations behind it — a Linux DMA-heap DMA-BUF,
a macOS/iOS `IOSurfaceRef`, or an Android `AHardwareBuffer`. Callers ask for `Dma` and get whichever
the platform provides.

## Image tensors
Images go through [`Tensor::image`] (or [`Tensor::image_desc`] for the full-featured request) rather
than [`Tensor::new`]. Two things set them apart:

- They take a [`CpuAccess`] declaration (`ReadWrite` is the default). Hardware access — GPU, NPU,
  ISP, codec — is always implied.
  A CPU map, pin or sync beyond the declaration fails with [`Error::CpuAccessDenied`].
- DMA-backed images carry a 64-byte-aligned row stride in every layout. Read the pitch from
  [`Tensor::effective_row_stride`].
 */
#[cfg(all(target_os = "android", not(edgefirst_tensor_dynamic)))]
mod ahardwarebuffer;
// AHardwareBuffer layout logic: format table, descriptor geometry and
// overflow-checked shape math.
#[allow(dead_code)]
mod ahardwarebuffer_layout;
pub mod colorimetry;
#[cfg(feature = "tracing")]
pub mod trace;

pub mod covguard;
mod cuda;
mod detect;
#[cfg(all(target_os = "linux", not(edgefirst_tensor_dynamic)))]
mod dma;
#[cfg(all(target_os = "linux", not(edgefirst_tensor_dynamic)))]
mod dmabuf;
mod error;
mod format;
// D3D11 texture backing (Windows) and its layout table.
#[cfg(target_os = "windows")]
pub mod d3d11;
pub mod d3d11_layout;
// IOSurface layout table.
#[cfg(any(target_os = "macos", target_os = "ios"))]
mod iosurface_layout;
// IOSurface backing.
#[cfg(all(
    any(target_os = "macos", target_os = "ios"),
    not(edgefirst_tensor_dynamic)
))]
mod iosurface;
pub mod lens;
#[cfg(not(edgefirst_tensor_dynamic))]
mod mem;
mod pbo;
pub mod pin;
mod preprocess;
pub mod protocol;
#[cfg(all(unix, not(edgefirst_tensor_dynamic)))]
mod shm;
mod tensor_dyn;
#[cfg(test)]
mod test_support;
pub mod view;
mod vocabulary;
pub use colorimetry::{
    ColorEncoding, ColorRange, ColorSpace, ColorTransfer, Colorimetry, MatrixWeights, RangeScaling,
};

/// Installs the coverage flush-on-abort handler for this crate's
/// instrumented test binary. See `covguard`. Only present under coverage on
/// Linux.
#[cfg(all(coverage, target_os = "linux"))]
#[used]
#[link_section = ".init_array"]
static __EDGEFIRST_COV_INSTALL: extern "C" fn() = {
    extern "C" fn ctor() {
        crate::covguard::install();
    }
    ctor
};

// Backing tensor and map types.
#[cfg(all(target_os = "android", not(edgefirst_tensor_dynamic)))]
pub use crate::ahardwarebuffer::image_ahardwarebuffer_layout;
#[cfg(all(target_os = "android", not(edgefirst_tensor_dynamic)))]
pub(crate) use crate::ahardwarebuffer::AHardwareBufferTensor;
#[cfg(all(target_os = "linux", not(edgefirst_tensor_dynamic)))]
pub(crate) use crate::dma::DmaTensor;
#[cfg(all(
    any(target_os = "macos", target_os = "ios"),
    not(edgefirst_tensor_dynamic)
))]
pub(crate) use crate::iosurface::IoSurfaceTensor;
// D3D11 texture geometry helpers.
#[cfg(target_os = "windows")]
pub use crate::d3d11::geometry::{d3d11_shared_handle_geometry, d3d11_texture_geometry};
#[cfg(any(target_os = "macos", target_os = "ios"))]
pub use crate::iosurface_layout::image_iosurface_layout;
#[cfg(not(edgefirst_tensor_dynamic))]
pub(crate) use crate::mem::MemTensor;
pub use crate::pbo::{
    client_state_pbo_ops, read_pbo_vtable_parts, EfClientState, EfPboMapFn, EfPboUnmapFn,
    PboClientParts, PboMapping, PboOps, PboOpsVtableParts, PboTensor,
};
#[cfg(all(unix, not(edgefirst_tensor_dynamic)))]
pub(crate) use crate::shm::ShmTensor;
pub use cuda::{
    client_state_cuda_ops, gl_map_resource, gl_register_buffer, gl_unmap_resource,
    gl_unregister_resource, is_cuda_available, memcpy_device_to_host, memcpy_host_to_device,
    runtime_path as cuda_runtime_path, stream_create, stream_destroy, stream_synchronize,
    CudaGlOps, CudaHandle, CudaMap, CudaStream,
};
pub use detect::{unletter_norm, BoundingBox, DetectBox, ProtoData, ProtoLayout, Segmentation};
pub use error::{Error, Result};
pub use preprocess::{
    broadcast_rgb, fit_code, interpolation_code, letterbox_placement_code, letterbox_rect,
    normalization_code, pad_space_code, ContentRegion, ContentRegionSource, Fit, FitMode,
    Interpolation, LetterboxPlacement, Normalization, NormalizationPreset, PadSpace,
    PreprocessSpec,
};
// Backs `tensor-capi`'s `ef_tensor_export`/`ef_tensor_import`.
#[cfg(not(edgefirst_tensor_dynamic))]
pub mod blob;
#[cfg(not(edgefirst_tensor_dynamic))]
pub use blob::{
    BlobError, BlobHeader, BlobPlane, BlobStrings, BlobView, Regions, TransportMode, HEADER_LEN,
    PLANE_RECORD_LEN,
};
pub use format::{ChromaLayout, PixelFormat, PixelLayout, PlaneGeometry};
pub use lens::Element;
use num_traits::Num;
pub use pin::HostPin;
pub use protocol::{
    dtype as tensor_dtype, dtype_to_dtype, format as tensor_format, format_from_code,
    kind as tensor_kind, SendPtr, TensorDesc, ABI_VERSION,
};
use serde::{Deserialize, Serialize};
#[cfg(unix)]
use std::os::fd::OwnedFd;
use std::{
    fmt,
    sync::{
        atomic::{AtomicU64, Ordering},
        Arc, Weak,
    },
};
#[cfg(edgefirst_tensor_dynamic)]
pub use tensor_dyn::Tensor;
pub use tensor_dyn::{BorrowedTensor, Raw, TensorDyn};
pub use view::HostView;
pub use vocabulary::vocabulary_demo;

/// Opaque keep-alive handle for a foreign-memory tensor (see
/// [`Tensor::from_foreign`] / [`TensorDyn::from_foreign_ptr`]).
///
/// The HAL borrows the foreign buffer without owning it; this handle co-owns
/// the source so the borrowed memory stays valid for the tensor's life. Its
/// `Drop` releases the source, once, after the last sharing tensor/view/map
/// drops.
pub type ForeignOwner = Box<dyn std::any::Any + Send + Sync>;

/// Re-export of `half::f16`.
pub use half::f16;

// =============================================================================
// RGBA16F packed-layout geometry
//
// A `PlanarRgb` [3,H,W] or `PlanarRgba` [4,H,W] f16 tensor is represented
// on the GPU as an RGBA16F surface. Four contiguous f16 elements are packed
// into each 8-byte RGBA16F texel, yielding a `(W/4, C*H)` surface.
// =============================================================================

/// Geometry of the RGBA16F-packed surface backing a planar F16 image tensor.
///
/// 4 contiguous f16 elements of a `[C, H, W]` planar tensor are packed into
/// each RGBA16F texel, yielding a surface of `(W/4, C*H)` texels at
/// 8 bytes/texel. The byte stream can be consumed as `&[f16]` with shape
/// `[1, C, H, W]` without rearrangement.
///
/// Obtained from [`packed_rgba16f_layout`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PackedRgba16fLayout {
    /// Surface width in texels (`width / 4`).
    pub surface_w: usize,
    /// Surface height in texels (`planes * height`).
    pub surface_h: usize,
    /// Bytes per RGBA16F texel (always 8).
    pub bytes_per_texel: usize,
    /// Row pitch in bytes (`surface_w * 8`).
    pub pitch: usize,
}

/// Geometry for the RGBA16F-packed surface backing a planar F16 image
/// tensor.
///
/// Returns `Some(layout)` only when all of the following hold:
///
/// - `dtype == DType::F16`
/// - `format` is `PixelFormat::PlanarRgb` (3 planes) or
///   `PixelFormat::PlanarRgba` (4 planes)
/// - `width % 4 == 0`
///
/// Returns `None` for any other `(format, dtype)` combination, misaligned
/// width, or when the surface geometry would overflow `usize`.
///
/// # Examples
///
/// ```rust
/// use edgefirst_tensor::{packed_rgba16f_layout, PixelFormat, DType};
///
/// let layout = packed_rgba16f_layout(PixelFormat::PlanarRgb, DType::F16, 640, 480).unwrap();
/// assert_eq!(layout.surface_w, 160);
/// assert_eq!(layout.surface_h, 1440);
/// assert_eq!(layout.bytes_per_texel, 8);
/// assert_eq!(layout.pitch, 1280);
/// ```
pub fn packed_rgba16f_layout(
    format: PixelFormat,
    dtype: DType,
    width: usize,
    height: usize,
) -> Option<PackedRgba16fLayout> {
    if dtype != DType::F16 {
        return None;
    }
    let planes: usize = match format {
        PixelFormat::PlanarRgb => 3,
        PixelFormat::PlanarRgba => 4,
        _ => return None,
    };
    if !width.is_multiple_of(4) {
        return None;
    }
    let surface_w = width / 4;
    // Checked arithmetic: overflow returns `None`.
    let surface_h = planes.checked_mul(height)?;
    let bytes_per_texel = 8;
    let pitch = surface_w.checked_mul(bytes_per_texel)?;
    Some(PackedRgba16fLayout {
        surface_w,
        surface_h,
        bytes_per_texel,
        pitch,
    })
}

/// Geometry of the RGBA8888-packed surface backing a packed RGB u8/i8 image
/// tensor.
///
/// The tight `[H, W, 3]` byte stream is written into an RGBA8888 surface:
/// each texel carries 4 consecutive RGB bytes, giving a `(W*3/4, H)` surface
/// at 4 bytes/texel whose rows are byte-identical to tight RGB.
///
/// Returns `Some(layout)` only when `width % 4 == 0` and the geometry does
/// not overflow `usize`.
///
/// # Examples
///
/// ```rust
/// use edgefirst_tensor::packed_rgb888_layout;
///
/// let layout = packed_rgb888_layout(640, 480).unwrap();
/// assert_eq!(layout.surface_w, 480); // 640*3/4
/// assert_eq!(layout.surface_h, 480);
/// assert_eq!(layout.bytes_per_texel, 4);
/// assert_eq!(layout.pitch, 1920); // 640*3
/// assert!(packed_rgb888_layout(641, 480).is_none());
/// ```
pub fn packed_rgb888_layout(width: usize, height: usize) -> Option<PackedRgb888Layout> {
    if !width.is_multiple_of(4) {
        return None;
    }
    let row_bytes = width.checked_mul(3)?;
    let surface_w = row_bytes / 4;
    Some(PackedRgb888Layout {
        surface_w,
        surface_h: height,
        bytes_per_texel: 4,
        pitch: row_bytes,
    })
}

/// Geometry of the RGBA8888 surface backing a packed RGB u8/i8 image tensor.
///
/// Obtained from [`packed_rgb888_layout`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PackedRgb888Layout {
    /// Surface width in texels (`width * 3 / 4`).
    pub surface_w: usize,
    /// Surface height in texels (`height`).
    pub surface_h: usize,
    /// Bytes per RGBA8888 texel (always 4).
    pub bytes_per_texel: usize,
    /// Row pitch in bytes (`surface_w * 4` = `width * 3`).
    pub pitch: usize,
}

/// The logical `(width, height)` an allocation shape describes for
/// `format`: the inverse of [`PixelFormat::allocation_shape`].
///
/// `shape` must be the allocation shape, not
/// [`PixelFormat::addressing_shape`]: for a semi-planar format both are rank
/// 2 and cannot be told apart.
///
/// | Layout | Shape | Result |
/// |--------|-------|--------|
/// | Packed | `[h, w, c]` | `(shape[1], shape[0])` |
/// | Planar | `[c, h, w]` | `(shape[2], shape[1])` |
/// | SemiPlanar | `[combined_h, w]` | `(shape[1], image height)` |
///
/// A semi-planar shape carries the combined plane height; the logical
/// height is recovered through
/// [`PixelFormat::image_height_from_combined`].
///
/// # Errors
///
/// [`Error::InvalidShape`] for a shape of the wrong rank, or a semi-planar
/// combined height with no image height behind it.
///
/// # Example
///
/// ```
/// use edgefirst_tensor::{image_dims_from_shape, PixelFormat};
/// assert_eq!(image_dims_from_shape(PixelFormat::Rgba, &[480, 640, 4]).unwrap(), (640, 480));
/// assert_eq!(image_dims_from_shape(PixelFormat::PlanarRgb, &[3, 480, 640]).unwrap(), (640, 480));
/// assert_eq!(image_dims_from_shape(PixelFormat::Nv12, &[722, 640]).unwrap(), (640, 481));
/// assert!(image_dims_from_shape(PixelFormat::Nv12, &[721, 640]).is_err());
/// ```
pub fn image_dims_from_shape(format: PixelFormat, shape: &[usize]) -> Result<(usize, usize)> {
    let wrong_rank = |expected: usize| {
        Error::InvalidShape(format!(
            "image_dims_from_shape: {format:?} describes a rank-{expected} allocation \
             shape, got rank {} ({shape:?})",
            shape.len()
        ))
    };
    match format.layout() {
        PixelLayout::Packed => {
            if shape.len() != 3 {
                return Err(wrong_rank(3));
            }
            Ok((shape[1], shape[0]))
        }
        PixelLayout::Planar => {
            if shape.len() != 3 {
                return Err(wrong_rank(3));
            }
            Ok((shape[2], shape[1]))
        }
        PixelLayout::SemiPlanar => {
            if shape.len() != 2 {
                return Err(wrong_rank(2));
            }
            let height = format.image_height_from_combined(shape[0]).ok_or_else(|| {
                Error::InvalidShape(format!(
                    "image_dims_from_shape: no {format:?} image height has combined plane \
                     height {} (shape {shape:?})",
                    shape[0]
                ))
            })?;
            Ok((shape[1], height))
        }
    }
}

/// Per-plane DMA-BUF descriptor for external buffer import.
///
/// Owns a duplicated file descriptor plus optional stride and offset metadata.
/// The fd is duplicated in [`new()`](Self::new). `import_image` consumes the
/// descriptor and takes ownership of the duped fd.
///
/// # Examples
///
/// ```rust,no_run
/// use edgefirst_tensor::PlaneDescriptor;
/// use std::os::fd::BorrowedFd;
///
/// // SAFETY: fd 42 is hypothetical; real code must pass a valid fd.
/// let pd = unsafe { PlaneDescriptor::new(BorrowedFd::borrow_raw(42)) }
///     .unwrap()
///     .with_stride(2048)
///     .with_offset(0);
/// ```
#[cfg(unix)]
pub struct PlaneDescriptor {
    fd: OwnedFd,
    stride: Option<usize>,
    offset: Option<usize>,
}

#[cfg(unix)]
impl PlaneDescriptor {
    /// Create a new plane descriptor by duplicating the given file descriptor.
    ///
    /// The caller retains ownership of the original fd.
    ///
    /// # Errors
    ///
    /// Returns an error if the `dup()` syscall fails (e.g. invalid fd or
    /// fd limit reached).
    pub fn new(fd: std::os::fd::BorrowedFd<'_>) -> Result<Self> {
        let owned = fd.try_clone_to_owned()?;
        Ok(Self {
            fd: owned,
            stride: None,
            offset: None,
        })
    }

    /// Set the row stride in bytes (consuming builder).
    pub fn with_stride(mut self, stride: usize) -> Self {
        self.stride = Some(stride);
        self
    }

    /// Set the plane offset in bytes (consuming builder).
    pub fn with_offset(mut self, offset: usize) -> Self {
        self.offset = Some(offset);
        self
    }

    /// Consume the descriptor and return the owned file descriptor.
    pub fn into_fd(self) -> OwnedFd {
        self.fd
    }

    /// Row stride in bytes, if set.
    pub fn stride(&self) -> Option<usize> {
        self.stride
    }

    /// Plane offset in bytes, if set.
    pub fn offset(&self) -> Option<usize> {
        self.offset
    }
}

/// A rectangular sub-region of a tensor's leading spatial frame, in pixels.
///
/// The argument to [`Tensor::view`] and the source sampling window in the
/// image crate's `Crop`. Coordinates are pixel/element units of the leading
/// spatial axes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Region {
    pub x: usize,
    pub y: usize,
    pub width: usize,
    pub height: usize,
}

impl Region {
    /// Create a region at `(x, y)` spanning `width` × `height` pixels.
    pub fn new(x: usize, y: usize, width: usize, height: usize) -> Self {
        Self {
            x,
            y,
            width,
            height,
        }
    }

    /// True when the region lies fully within a `width` × `height` frame.
    pub fn fits_within(&self, width: usize, height: usize) -> bool {
        self.x.saturating_add(self.width) <= width && self.y.saturating_add(self.height) <= height
    }
}

/// The parent image a [`view`](Tensor::view)/[`batch`](Tensor::batch) sub-region
/// was carved from, snapshotted at the time the view was created.
///
/// A view shares the parent's `BufferIdentity` and addresses a sub-rectangle of
/// it. `parent_width`/`parent_height` are the parent's logical pixel
/// dimensions; `x`/`y` are this view's top-left origin within the parent
/// (pixels). Nested views compose: the snapshot names the root parent, with
/// offsets accumulated.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ViewOrigin {
    /// Logical width of the root parent image, in pixels.
    pub parent_width: usize,
    /// Logical height of the root parent image, in pixels. For a `batch(n)` view
    /// of an `[N, H, W, C]` tensor this is `N * H` (the tiles stack vertically in
    /// the shared buffer).
    pub parent_height: usize,
    /// The parent's row stride in bytes.
    pub parent_row_stride: usize,
    /// This view's top-left x origin within the root parent, in pixels.
    pub x: usize,
    /// This view's top-left y origin within the root parent, in pixels.
    pub y: usize,
}

// Declared through `ef_vocabulary!` (see `vocabulary.rs`): `.code()` is this
// crate's wire numbering, mirrored by `protocol::dtype` and the C ABI's
// `hal_dtype`.
crate::ef_vocabulary! {
    /// Element type discriminant for runtime type identification.
    #[derive(Serialize, Deserialize)]
    #[non_exhaustive]
    pub enum DType {
        U8 = 0, "u8", U8,
        I8 = 1, "i8", I8,
        U16 = 2, "u16", U16,
        I16 = 3, "i16", I16,
        U32 = 4, "u32", U32,
        I32 = 5, "i32", I32,
        U64 = 6, "u64", U64,
        I64 = 7, "i64", I64,
        F16 = 8, "f16", F16,
        F32 = 9, "f32", F32,
        F64 = 10, "f64", F64,
    }
    #[doc(hidden)]
    pub mod dtype_wire;
}

impl DType {
    /// Size of one element in bytes.
    pub const fn size(&self) -> usize {
        match self {
            Self::U8 | Self::I8 => 1,
            Self::U16 | Self::I16 | Self::F16 => 2,
            Self::U32 | Self::I32 | Self::F32 => 4,
            Self::U64 | Self::I64 | Self::F64 => 8,
        }
    }

    /// Short type name (e.g., "u8", "f32", "f16"). Identical to
    /// [`DType::as_str`].
    pub fn name(&self) -> &'static str {
        self.as_str()
    }
}

impl fmt::Display for DType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

/// Map a static numeric type `T` to its `DType` discriminant, returning
/// `None` for types that do not have a `DType` representation.
#[cfg(not(edgefirst_tensor_dynamic))]
pub(crate) fn dtype_of<T: 'static>() -> Option<DType> {
    use std::any::TypeId;
    let id = TypeId::of::<T>();
    if id == TypeId::of::<u8>() {
        Some(DType::U8)
    } else if id == TypeId::of::<i8>() {
        Some(DType::I8)
    } else if id == TypeId::of::<u16>() {
        Some(DType::U16)
    } else if id == TypeId::of::<i16>() {
        Some(DType::I16)
    } else if id == TypeId::of::<u32>() {
        Some(DType::U32)
    } else if id == TypeId::of::<i32>() {
        Some(DType::I32)
    } else if id == TypeId::of::<u64>() {
        Some(DType::U64)
    } else if id == TypeId::of::<i64>() {
        Some(DType::I64)
    } else if id == TypeId::of::<half::f16>() {
        Some(DType::F16)
    } else if id == TypeId::of::<f32>() {
        Some(DType::F32)
    } else if id == TypeId::of::<f64>() {
        Some(DType::F64)
    } else {
        None
    }
}

// =============================================================================
// Quantization metadata, available on integer element types only
// =============================================================================

mod sealed {
    pub trait Sealed {}
    impl Sealed for u8 {}
    impl Sealed for i8 {}
    impl Sealed for u16 {}
    impl Sealed for i16 {}
    impl Sealed for u32 {}
    impl Sealed for i32 {}
    impl Sealed for u64 {}
    impl Sealed for i64 {}
    // f16 / f32 / f64 are sealed and not given `IntegerType`.
    impl Sealed for half::f16 {}
    impl Sealed for f32 {}
    impl Sealed for f64 {}
}

/// Integer element types that may carry quantization metadata.
///
/// Sealed trait: implemented for `u8`, `i8`, `u16`, `i16`, `u32`, `i32`,
/// `u64`, `i64`. Cannot be implemented downstream. Float element types
/// (`half::f16`, `f32`, `f64`) are excluded.
pub trait IntegerType: sealed::Sealed {}
impl IntegerType for u8 {}
impl IntegerType for i8 {}
impl IntegerType for u16 {}
impl IntegerType for i16 {}
impl IntegerType for u32 {}
impl IntegerType for i32 {}
impl IntegerType for u64 {}
impl IntegerType for i64 {}

/// Quantization parameters for an integer tensor.
///
/// Covers all four modes the edgefirst.json spec defines:
///
/// | Mode | `scale.len()` | `zero_point` | `axis` |
/// |---|---|---|---|
/// | Per-tensor symmetric | 1 | `None` | `None` |
/// | Per-tensor asymmetric | 1 | `Some(len == 1)` | `None` |
/// | Per-channel symmetric | >1 | `None` | `Some(c)` |
/// | Per-channel asymmetric | >1 | `Some(len == scale.len())` | `Some(c)` |
///
/// The quantized storage type is carried on the parent [`Tensor<T>`].
/// Constructed via the four named constructors.
///
/// Dequantization formula:
///
/// ```text
///   real_value = scale[c] × (quantized_value[c] - zero_point[c])
/// ```
///
/// where `c` is the channel index (always `0` for per-tensor).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Quantization {
    /// Per-tensor: `vec![scale]`. Per-channel: `vec![scale_0, scale_1, ...]`.
    #[serde(deserialize_with = "deserialize_scalar_or_vec_f32")]
    scale: Vec<f32>,

    /// `None` means symmetric (zero-point is 0). `Some(vec)` must have the
    /// same length as `scale`.
    #[serde(
        default,
        deserialize_with = "deserialize_opt_scalar_or_vec_i32",
        skip_serializing_if = "Option::is_none"
    )]
    zero_point: Option<Vec<i32>>,

    /// Channel axis for per-channel quantization. `Some(_)` iff
    /// `scale.len() > 1`. Validated against the parent tensor's shape at
    /// `set_quantization()` time.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    axis: Option<usize>,
}

/// Semantic mode discriminant for kernel dispatch, obtained from
/// [`Quantization::mode`]. Borrows the scales and zero-points as slices.
#[derive(Debug, Clone, Copy)]
pub enum QuantMode<'a> {
    PerTensorSymmetric {
        scale: f32,
    },
    PerTensor {
        scale: f32,
        zero_point: i32,
    },
    PerChannelSymmetric {
        scales: &'a [f32],
        axis: usize,
    },
    PerChannel {
        scales: &'a [f32],
        zero_points: &'a [i32],
        axis: usize,
    },
}

impl Quantization {
    /// Per-tensor symmetric (zero_point = 0).
    pub fn per_tensor_symmetric(scale: f32) -> Self {
        Self {
            scale: vec![scale],
            zero_point: None,
            axis: None,
        }
    }

    /// Per-tensor asymmetric.
    pub fn per_tensor(scale: f32, zero_point: i32) -> Self {
        Self {
            scale: vec![scale],
            zero_point: Some(vec![zero_point]),
            axis: None,
        }
    }

    /// Per-channel symmetric. Errors on empty `scales`.
    pub fn per_channel_symmetric(scales: Vec<f32>, axis: usize) -> Result<Self> {
        if scales.is_empty() {
            return Err(Error::QuantizationInvalid {
                field: "scale.len",
                expected: "non-empty per-channel scales".to_string(),
                got: "length 0".to_string(),
            });
        }
        Ok(Self {
            scale: scales,
            zero_point: None,
            axis: Some(axis),
        })
    }

    /// Per-channel asymmetric. Errors on length mismatch between `scales`
    /// and `zero_points`, or empty arrays.
    pub fn per_channel(scales: Vec<f32>, zero_points: Vec<i32>, axis: usize) -> Result<Self> {
        if scales.is_empty() {
            return Err(Error::QuantizationInvalid {
                field: "scale.len",
                expected: "non-empty per-channel scales".to_string(),
                got: "length 0".to_string(),
            });
        }
        if scales.len() != zero_points.len() {
            return Err(Error::QuantizationInvalid {
                field: "zero_point.len",
                expected: format!("length matches scale ({})", scales.len()),
                got: format!("length {}", zero_points.len()),
            });
        }
        Ok(Self {
            scale: scales,
            zero_point: Some(zero_points),
            axis: Some(axis),
        })
    }

    /// Borrow-based dispatch view.
    pub fn mode(&self) -> QuantMode<'_> {
        match (self.scale.len(), self.zero_point.as_deref(), self.axis) {
            (1, None, _) => QuantMode::PerTensorSymmetric {
                scale: self.scale[0],
            },
            (1, Some(zps), _) => QuantMode::PerTensor {
                scale: self.scale[0],
                zero_point: zps.first().copied().unwrap_or(0),
            },
            (_, None, Some(axis)) => QuantMode::PerChannelSymmetric {
                scales: &self.scale,
                axis,
            },
            (_, Some(zps), Some(axis)) => QuantMode::PerChannel {
                scales: &self.scale,
                zero_points: zps,
                axis,
            },
            // Falls back to per-tensor symmetric using scale[0]; debug builds
            // assert.
            _ => {
                debug_assert!(
                    false,
                    "Quantization::mode: per-channel without axis is unreachable"
                );
                QuantMode::PerTensorSymmetric {
                    scale: self.scale.first().copied().unwrap_or(1.0),
                }
            }
        }
    }

    /// Returns `true` for per-tensor quantization (`scale.len() == 1`).
    pub fn is_per_tensor(&self) -> bool {
        self.scale.len() == 1
    }

    /// Returns `true` for per-channel quantization (`scale.len() > 1`).
    pub fn is_per_channel(&self) -> bool {
        self.scale.len() > 1
    }

    /// Returns `true` for symmetric quantization (no zero-point, or
    /// zero-point vector of all zeros).
    pub fn is_symmetric(&self) -> bool {
        match &self.zero_point {
            None => true,
            Some(zps) => zps.iter().all(|&z| z == 0),
        }
    }

    /// Borrow the scale array. Length 1 for per-tensor; `num_channels` for
    /// per-channel.
    pub fn scale(&self) -> &[f32] {
        &self.scale
    }

    /// Borrow the zero-point array. `None` for symmetric.
    pub fn zero_point(&self) -> Option<&[i32]> {
        self.zero_point.as_deref()
    }

    /// Channel axis for per-channel quantization. `None` for per-tensor.
    pub fn axis(&self) -> Option<usize> {
        self.axis
    }

    /// Validate against a target tensor shape. Rejects:
    ///   - empty `scale`
    ///   - `zero_point` length inconsistent with `scale` (per-tensor must
    ///     have len 1, per-channel must match `scale.len`)
    ///   - `axis >= shape.len()`
    ///   - `scale.len() != shape[axis]` for per-channel
    ///   - per-channel without axis
    ///   - per-tensor with redundant axis
    #[cfg(not(edgefirst_tensor_dynamic))]
    pub(crate) fn validate(&self, shape: &[usize]) -> Result<()> {
        if self.scale.is_empty() {
            return Err(Error::QuantizationInvalid {
                field: "scale.len",
                expected: ">= 1".to_string(),
                got: "0".to_string(),
            });
        }
        if let Some(zps) = self.zero_point.as_ref() {
            // Per-tensor: scale.len() == 1 and zero_point.len() must == 1.
            // Per-channel: zero_point.len() must == scale.len().
            let expected = if self.scale.len() == 1 {
                1
            } else {
                self.scale.len()
            };
            if zps.len() != expected {
                return Err(Error::QuantizationInvalid {
                    field: "zero_point.len",
                    expected: format!(
                        "{expected} (matching {})",
                        if self.scale.len() == 1 {
                            "per-tensor scale"
                        } else {
                            "per-channel scale.len"
                        }
                    ),
                    got: format!("length {}", zps.len()),
                });
            }
        }

        match (self.scale.len(), self.axis) {
            (1, None) => Ok(()),
            (1, Some(_)) => Err(Error::QuantizationInvalid {
                field: "per_tensor_redundant_axis",
                expected: "axis=None for per-tensor quantization".to_string(),
                got: format!("axis={:?}", self.axis),
            }),
            (_, None) => Err(Error::QuantizationInvalid {
                field: "per_channel_requires_axis",
                expected: format!(
                    "axis=Some(_) for per-channel quantization (scale.len={})",
                    self.scale.len()
                ),
                got: "axis=None".to_string(),
            }),
            (n, Some(axis)) => {
                if axis >= shape.len() {
                    return Err(Error::QuantizationInvalid {
                        field: "axis",
                        expected: format!("axis < tensor rank ({})", shape.len()),
                        got: format!("axis={axis}"),
                    });
                }
                if shape[axis] != n {
                    return Err(Error::QuantizationInvalid {
                        field: "scale.len",
                        expected: format!("length matches shape[{axis}] ({})", shape[axis]),
                        got: format!("length {n}"),
                    });
                }
                Ok(())
            }
        }
    }
}

impl From<(f32, i32)> for Quantization {
    /// Construction from a `(scale, zero_point)` tuple.
    fn from((scale, zero_point): (f32, i32)) -> Self {
        Self::per_tensor(scale, zero_point)
    }
}

fn deserialize_scalar_or_vec_f32<'de, D: serde::Deserializer<'de>>(
    de: D,
) -> std::result::Result<Vec<f32>, D::Error> {
    use serde::de::{self, Visitor};
    struct V;
    impl<'de> Visitor<'de> for V {
        type Value = Vec<f32>;
        #[cfg_attr(test, mutants::skip)]
        fn expecting(&self, f: &mut fmt::Formatter) -> fmt::Result {
            f.write_str("f32 or array of f32")
        }
        fn visit_f64<E: de::Error>(self, v: f64) -> std::result::Result<Self::Value, E> {
            Ok(vec![v as f32])
        }
        #[allow(clippy::cast_possible_truncation)]
        fn visit_i64<E: de::Error>(self, v: i64) -> std::result::Result<Self::Value, E> {
            Ok(vec![v as f32])
        }
        #[allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]
        fn visit_u64<E: de::Error>(self, v: u64) -> std::result::Result<Self::Value, E> {
            Ok(vec![v as f32])
        }
        fn visit_seq<A: de::SeqAccess<'de>>(
            self,
            mut seq: A,
        ) -> std::result::Result<Self::Value, A::Error> {
            let mut out = Vec::with_capacity(seq.size_hint().unwrap_or(1));
            while let Some(x) = seq.next_element::<f32>()? {
                out.push(x);
            }
            Ok(out)
        }
    }
    de.deserialize_any(V)
}

fn deserialize_opt_scalar_or_vec_i32<'de, D: serde::Deserializer<'de>>(
    de: D,
) -> std::result::Result<Option<Vec<i32>>, D::Error> {
    use serde::de::{self, Visitor};
    struct V;
    impl<'de> Visitor<'de> for V {
        type Value = Option<Vec<i32>>;
        #[cfg_attr(test, mutants::skip)]
        fn expecting(&self, f: &mut fmt::Formatter) -> fmt::Result {
            f.write_str("null, i32, or array of i32")
        }
        #[cfg_attr(test, mutants::skip)]
        fn visit_none<E: de::Error>(self) -> std::result::Result<Self::Value, E> {
            Ok(None)
        }
        #[cfg_attr(test, mutants::skip)]
        fn visit_unit<E: de::Error>(self) -> std::result::Result<Self::Value, E> {
            Ok(None)
        }
        fn visit_some<D2: serde::Deserializer<'de>>(
            self,
            de: D2,
        ) -> std::result::Result<Self::Value, D2::Error> {
            struct Inner;
            impl<'de> Visitor<'de> for Inner {
                type Value = Vec<i32>;
                #[cfg_attr(test, mutants::skip)]
                fn expecting(&self, f: &mut fmt::Formatter) -> fmt::Result {
                    f.write_str("i32 or array of i32")
                }
                #[allow(clippy::cast_possible_truncation)]
                fn visit_i64<E: de::Error>(self, v: i64) -> std::result::Result<Self::Value, E> {
                    Ok(vec![v as i32])
                }
                #[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
                fn visit_u64<E: de::Error>(self, v: u64) -> std::result::Result<Self::Value, E> {
                    Ok(vec![v as i32])
                }
                fn visit_seq<A: de::SeqAccess<'de>>(
                    self,
                    mut seq: A,
                ) -> std::result::Result<Self::Value, A::Error> {
                    let mut out = Vec::with_capacity(seq.size_hint().unwrap_or(1));
                    while let Some(x) = seq.next_element::<i32>()? {
                        out.push(x);
                    }
                    Ok(out)
                }
            }
            de.deserialize_any(Inner).map(Some)
        }
        #[allow(clippy::cast_possible_truncation)]
        fn visit_i64<E: de::Error>(self, v: i64) -> std::result::Result<Self::Value, E> {
            Ok(Some(vec![v as i32]))
        }
        #[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
        fn visit_u64<E: de::Error>(self, v: u64) -> std::result::Result<Self::Value, E> {
            Ok(Some(vec![v as i32]))
        }
        fn visit_seq<A: de::SeqAccess<'de>>(
            self,
            mut seq: A,
        ) -> std::result::Result<Self::Value, A::Error> {
            let mut out = Vec::with_capacity(seq.size_hint().unwrap_or(1));
            while let Some(x) = seq.next_element::<i32>()? {
                out.push(x);
            }
            Ok(Some(out))
        }
    }
    de.deserialize_option(V)
}

/// Uniform guard for mutable access through a read-only map. Every
/// `TensorMap` backend calls this from `as_mut_slice`/`deref_mut`.
#[inline]
pub(crate) fn assert_map_writable(writable: bool, backend: &str) {
    assert!(
        writable,
        "{backend} map is read-only (obtained via map_read()/CpuAccess::Read) — \
         use map_mut() or map_write() for mutable access"
    );
}

crate::ef_vocabulary! {
    /// Declared CPU involvement for a tensor, chosen at allocation.
    ///
    /// The declaration covers all CPU access to the buffer: the caller's
    /// maps, pins and sync brackets, and the HAL's own CPU work (the CPU
    /// image backend's converts and draws, codec decodes, decoder reads and
    /// writes). Hardware access (ISP, codec, GPU, NPU, G2D, a compositor)
    /// needs no declaration and is always allowed.
    ///
    /// The HAL enforces it. A CPU map, pin or sync whose direction the
    /// declaration does not cover fails with [`Error::CpuAccessDenied`] on
    /// every backend, before the memory is touched. The HAL's own CPU passes
    /// go through the same check.
    ///
    /// | Level | CPU may | Memory selected |
    /// |---|---|---|
    /// | `None` | nothing | Linux DMA-BUF: uncached CMA where available. Android: no CPU usage bits, so tile compression stays possible. |
    /// | `Read` | read | cached; read-direction syncs only |
    /// | `Write` | write whole passes | Linux DMA-BUF: uncached CMA where available, written write-combined |
    /// | `ReadWrite` (default) | read and write | cached |
    ///
    /// On Linux a `None` or `Write` DMA-BUF comes from
    /// `/dev/dma_heap/linux,cma-uncached` when the system has it, and from
    /// the cached heaps otherwise. A write-only pass must write the whole
    /// buffer (see [`TensorTrait::map_for_write`]).
    ///
    /// On Android the `ReadWrite` default asks gralloc for
    /// `CPU_READ_OFTEN | CPU_WRITE_OFTEN`, which may give linear, CPU-cached
    /// memory; `None` lets gralloc keep a tiled or compressed allocation.
    ///
    /// # Examples
    ///
    /// A CPU-drawn overlay: written whole every frame, never read back.
    ///
    /// ```rust,no_run
    /// use edgefirst_tensor::{
    ///     CpuAccess, Error, PixelFormat, Tensor, TensorMapTrait, TensorMemory, TensorTrait,
    /// };
    ///
    /// let overlay = Tensor::<u8>::image(
    ///     640,
    ///     480,
    ///     PixelFormat::Rgba,
    ///     Some(TensorMemory::DmaBuf),
    ///     CpuAccess::Write,
    /// )?;
    /// overlay.map_for_write()?.as_mut_slice().fill(0);
    /// assert!(matches!(overlay.map_read(), Err(Error::CpuAccessDenied { .. })));
    /// # Ok::<(), Error>(())
    /// ```
    ///
    /// [`CpuAccess::code`] is the wire code the C header's `ef_cpu_access`
    /// enumerators are asserted against.
    #[derive(Default)]
    pub enum CpuAccess {
        /// Hardware-only buffer: no CPU access. Required for vendor tile
        /// compression on platforms with compressed layouts; uncached CMA
        /// for a Linux DMA-BUF where available.
        None = 0, "none", NONE,
        /// CPU reads only (verification, CPU consumers) — cached mapping.
        Read = 1, "read", READ,
        /// CPU writes only, whole-buffer passes (decode targets, CPU-drawn
        /// overlays) — write-combined mapping; uncached CMA for a Linux
        /// DMA-BUF where available.
        Write = 2, "write", WRITE,
        /// CPU reads and writes (the default).
        #[default]
        ReadWrite = 3, "readwrite", READ_WRITE,
    }
    #[doc(hidden)]
    pub mod cpu_access_wire;
}

impl CpuAccess {
    /// The declaration with exactly the given directions.
    pub(crate) const fn from_directions(read: bool, write: bool) -> CpuAccess {
        match (read, write) {
            (true, true) => CpuAccess::ReadWrite,
            (true, false) => CpuAccess::Read,
            (false, true) => CpuAccess::Write,
            (false, false) => CpuAccess::None,
        }
    }

    /// Whether this declaration includes CPU reads.
    pub fn reads(self) -> bool {
        matches!(self, CpuAccess::Read | CpuAccess::ReadWrite)
    }

    /// Whether this declaration includes CPU writes.
    pub fn writes(self) -> bool {
        matches!(self, CpuAccess::Write | CpuAccess::ReadWrite)
    }

    /// Whether this declaration covers `requested` (every direction the
    /// request needs is declared).
    pub fn covers(self, requested: CpuAccess) -> bool {
        (!requested.reads() || self.reads()) && (!requested.writes() || self.writes())
    }

    /// The declaration covering every direction of both `self` and `other`.
    ///
    /// ```
    /// use edgefirst_tensor::CpuAccess;
    /// assert_eq!(CpuAccess::Read.union(CpuAccess::Write), CpuAccess::ReadWrite);
    /// assert_eq!(CpuAccess::None.union(CpuAccess::Write), CpuAccess::Write);
    /// ```
    pub fn union(self, other: CpuAccess) -> CpuAccess {
        Self::from_directions(
            self.reads() || other.reads(),
            self.writes() || other.writes(),
        )
    }

    /// The declaration covering only the directions both `self` and `other`
    /// cover.
    ///
    /// ```
    /// use edgefirst_tensor::CpuAccess;
    /// assert_eq!(CpuAccess::ReadWrite.intersection(CpuAccess::Write), CpuAccess::Write);
    /// assert_eq!(CpuAccess::Read.intersection(CpuAccess::Write), CpuAccess::None);
    /// ```
    pub fn intersection(self, other: CpuAccess) -> CpuAccess {
        Self::from_directions(
            self.reads() && other.reads(),
            self.writes() && other.writes(),
        )
    }

    /// `Ok` when this declaration covers `requested`, otherwise
    /// [`Error::CpuAccessDenied`] naming both.
    ///
    /// ```
    /// use edgefirst_tensor::{CpuAccess, Error};
    /// assert!(CpuAccess::ReadWrite.require(CpuAccess::Write).is_ok());
    /// assert!(matches!(
    ///     CpuAccess::Read.require(CpuAccess::Write),
    ///     Err(Error::CpuAccessDenied { .. })
    /// ));
    /// ```
    pub fn require(self, requested: CpuAccess) -> Result<()> {
        if self.covers(requested) {
            Ok(())
        } else {
            Err(Error::CpuAccessDenied {
                declared: self,
                requested,
            })
        }
    }

    /// [`require`](Self::require) for a map, pin or sync, where
    /// `CpuAccess::None` is no direction at all: [`Error::InvalidArgument`]
    /// naming `what`.
    pub(crate) fn require_direction(self, requested: CpuAccess, what: &str) -> Result<()> {
        if requested == CpuAccess::None {
            return Err(Error::InvalidArgument(format!(
                "{what}: CpuAccess::None is not a CPU access direction"
            )));
        }
        self.require(requested)
    }
}

/// Requested tile-compression behavior for an image allocation (set via
/// [`ImageDesc::with_compression`]).
///
/// Vendor GPUs store textures in proprietary compressed tile layouts
/// (UBWC, AFBC, PVRIC, DCC). Eligibility requires a hardware-only buffer
/// ([`CpuAccess::None`]). The request records the outcome on the tensor
/// ([`Tensor::compression`]); it does not change what bytes a consumer sees
/// through the GPU/NPU.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Compression {
    /// Let the platform use its native scheme when the format is
    /// eligible; otherwise allocate linear and count the fallback
    /// ([`compression_fallback_count`]).
    Any,
    /// Require one specific scheme: allocation fails with
    /// [`Error::InvalidArgument`] when the device's scheme differs and
    /// [`Error::NotImplemented`] on platforms without vendor tile
    /// compression.
    Scheme(CompressionScheme),
}

// Declared through `ef_vocabulary!` (see `vocabulary.rs`); the wire codes
// match the C enumerators. Code 0 is reserved for "linear", which the C side
// names `EF_COMPRESSION_NONE`.
crate::ef_vocabulary! {
    /// Vendor tile-compression schemes the HAL recognizes and records.
    ///
    /// Unrecognized platforms record `None` (linear); there is no `Unknown`
    /// variant.
    #[non_exhaustive]
    pub enum CompressionScheme {
        /// Qualcomm Adreno Universal Bandwidth Compression.
        Ubwc = 1, "ubwc", UBWC,
        /// Arm Mali/Immortalis Framebuffer Compression.
        Afbc = 2, "afbc", AFBC,
        /// Imagination PowerVR Image Compression (Google Tensor G5+).
        Pvric = 3, "pvric", PVRIC,
        /// Samsung Xclipse (AMD RDNA) Delta Color Compression.
        Dcc = 4, "dcc", DCC,
    }
    #[doc(hidden)]
    pub mod compression_wire;
}

crate::ef_vocabulary! {
    /// Whether a tensor's memory is known to be physically contiguous
    /// ([`Tensor::contiguity`]).
    ///
    /// Only memory this crate allocated from a Linux DMA heap is known
    /// either way: the CMA heap is contiguous and the system heap is not.
    /// An imported fd, and every other kind of memory, is `Unknown`.
    #[non_exhaustive]
    pub enum Contiguity {
        /// Not known: an imported fd, or memory not allocated from a DMA heap.
        Unknown = 0, "unknown", UNKNOWN,
        /// Allocated from the CMA heap.
        Contiguous = 1, "contiguous", CONTIGUOUS,
        /// Allocated from the system heap.
        NonContiguous = 2, "non_contiguous", NON_CONTIGUOUS,
    }
    #[doc(hidden)]
    pub mod contiguity_wire;
}

/// Count of image allocations that requested [`Compression::Any`] but
/// resolved to a linear layout (ineligible format/dtype, unrecognized
/// vendor, or a platform without vendor tile compression). Read via
/// [`compression_fallback_count`].
static COMPRESSION_FALLBACKS: AtomicU64 = AtomicU64::new(0);

/// Number of [`Compression::Any`] requests since process start that
/// resolved to a linear layout instead of a vendor tile scheme.
pub fn compression_fallback_count() -> u64 {
    COMPRESSION_FALLBACKS.load(Ordering::Relaxed)
}

/// Record a `Compression::Any` request resolving linear.
#[cfg(not(edgefirst_tensor_dynamic))]
pub(crate) fn note_compression_fallback(detail: &str) {
    COMPRESSION_FALLBACKS.fetch_add(1, Ordering::Relaxed);
    log::debug!("compression request fell back to linear: {detail}");
}

/// Whether this platform can allocate `(format, dtype)` images in a
/// vendor tile-compressed layout. `true` requires an Android build, an
/// eligible format (RGBA8888 `u8`/`i8`), and a positively identified GPU
/// vendor; everywhere else the answer is `false` and
/// [`Compression::Any`] requests fall back to linear.
pub fn compression_support(format: PixelFormat, dtype: DType) -> bool {
    #[cfg(target_os = "android")]
    {
        crate::ahardwarebuffer_layout::compression_eligible(format, dtype)
            && crate::ahardwarebuffer::device_compression_scheme().is_some()
    }
    #[cfg(not(target_os = "android"))]
    {
        let _ = (format, dtype);
        false
    }
}

/// Declarative image-allocation request for image tensors
/// ([`TensorDyn::image_desc`] and the image crate's
/// `ImageProcessor::create_image_desc`).
///
/// Carries the optional requests, such as the [`Compression`] request, that
/// the classic constructors (`image`, `create_image`) do not take. Fields
/// are private and the builders consume/return by value.
///
/// ```
/// use edgefirst_tensor::{Compression, CpuAccess, DType, ImageDesc, PixelFormat};
/// let desc = ImageDesc::new(640, 640, PixelFormat::Rgba, DType::U8)
///     .with_access(CpuAccess::None)
///     .with_compression(Compression::Any);
/// ```
#[derive(Debug, Clone)]
pub struct ImageDesc {
    width: usize,
    height: usize,
    format: PixelFormat,
    dtype: DType,
    memory: Option<TensorMemory>,
    access: CpuAccess,
    compression: Option<Compression>,
    contiguous: bool,
}

impl ImageDesc {
    /// A new image request: auto-selected memory, [`CpuAccess::ReadWrite`],
    /// no compression request.
    pub fn new(width: usize, height: usize, format: PixelFormat, dtype: DType) -> Self {
        Self {
            width,
            height,
            format,
            dtype,
            memory: None,
            access: CpuAccess::default(),
            compression: None,
            contiguous: false,
        }
    }

    /// Request a specific memory backing (`None` = auto-select).
    pub fn with_memory(mut self, memory: Option<TensorMemory>) -> Self {
        self.memory = memory;
        self
    }

    /// Declare the CPU access (see [`CpuAccess`]). Any declaration other
    /// than `None` makes a compression request invalid.
    pub fn with_access(mut self, access: CpuAccess) -> Self {
        self.access = access;
        self
    }

    /// Request a tile-compressed layout (see [`Compression`]).
    pub fn with_compression(mut self, compression: Compression) -> Self {
        self.compression = Some(compression);
        self
    }

    /// Require physically contiguous memory (see [`Contiguity`]).
    ///
    /// The image is allocated from the Linux CMA DMA heap only. Allocation
    /// fails, naming the heap, when there is no CMA heap or it cannot fit
    /// the image. Memory must be auto-selected or [`TensorMemory::DmaBuf`].
    /// Other platforms return [`Error::NotImplemented`].
    pub fn with_contiguous(mut self, contiguous: bool) -> Self {
        self.contiguous = contiguous;
        self
    }

    /// Requested width in pixels.
    pub fn width(&self) -> usize {
        self.width
    }

    /// Requested height in pixels.
    pub fn height(&self) -> usize {
        self.height
    }

    /// Requested pixel format.
    pub fn format(&self) -> PixelFormat {
        self.format
    }

    /// Requested element type.
    pub fn dtype(&self) -> DType {
        self.dtype
    }

    /// Requested memory backing (`None` = auto-select).
    pub fn memory(&self) -> Option<TensorMemory> {
        self.memory
    }

    /// Declared CPU access.
    pub fn access(&self) -> CpuAccess {
        self.access
    }

    /// The compression request, if any.
    pub fn compression(&self) -> Option<Compression> {
        self.compression
    }

    /// Whether physically contiguous memory is required.
    pub fn contiguous(&self) -> bool {
        self.contiguous
    }
}

/// What kind of system key an identity was derived from. Participates in the
/// id, so equal keys of different kinds yield different ids.
///
/// The key passed to [`BufferIdentity::derived`] is unique among live buffers
/// of that kind that own an allocation. Where no system-wide key exists a
/// process-local one is used (a pointer, a GL name). Zero-element tensors own
/// no allocation and share one identity.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IdentityKind {
    /// `(st_dev, st_ino)` of a dma-buf or memfd. Survives `dup`, and is the
    /// same value in every process that holds the buffer.
    DmaBuf,
    Shm,
    /// `IOSurfaceGetID`.
    IoSurface,
    /// `AHardwareBuffer_getId`, API 31+.
    AHardwareBufferId,
    /// The `AHardwareBuffer*` pointer, pre-API-31 where no id exists.
    /// Process-local: unique among live buffers here, meaningless elsewhere.
    AHardwareBufferPtr,
    /// An `ID3D11Texture2D*`. Process-local, like `AHardwareBufferPtr`, and
    /// the same pointer for every kind of texture tensor: one the HAL
    /// allocated, one wrapped from a caller's pointer, and one opened from a
    /// shared NT handle all key on the texture object they hold.
    ///
    /// Both backends derive it from the same pointer, so their ids are equal.
    ///
    /// A handle import does not compare equal to the tensor it was exported
    /// from, nor to a second import of the same handle: `OpenSharedResource1`
    /// mints a fresh `ID3D11Texture2D` per open.
    D3d11Texture,
    /// A host mapping address. Process-local.
    HostPtr,
    /// A GL buffer name. Valid only in the context that created it.
    Pbo,
}

/// Unique identity for a tensor's underlying buffer.
///
/// Created fresh on every buffer allocation or import. The `id` is a cache
/// key, `kind` records what system key (if any) it was derived from, and the
/// `guard` is an `Arc<()>` whose weak references allow downstream caches to
/// detect when the buffer has been dropped.
#[derive(Debug, Clone)]
pub struct BufferIdentity {
    id: u64,
    kind: IdentityKind,
    guard: Arc<()>,
}

impl BufferIdentity {
    /// Derive an identity from a key the operating system already guarantees.
    ///
    /// Two independently-linked copies of this crate derive the same id for
    /// the same buffer.
    pub fn derived(kind: IdentityKind, key: u64) -> Self {
        // Mixes the discriminant into the high bits.
        let id = ((kind as u64) << 56) ^ key;
        Self {
            id,
            kind,
            guard: Arc::new(()),
        }
    }

    /// Unique identifier for this buffer. Changes when the buffer changes.
    pub fn id(&self) -> u64 {
        self.id
    }

    /// What kind of system key this identity was derived from.
    pub fn kind(&self) -> IdentityKind {
        self.kind
    }

    /// Returns a weak reference to the buffer guard. Goes dead when the
    /// owning Tensor is dropped (and no clones remain).
    pub fn weak(&self) -> Weak<()> {
        Arc::downgrade(&self.guard)
    }
}

#[cfg(all(target_os = "linux", not(edgefirst_tensor_dynamic)))]
use nix::sys::stat::{major, minor};

/// Filesystem magic of the internal `dma_buf` mount, from
/// `include/uapi/linux/magic.h` (`DMA_BUF_MAGIC`, "DMAB").
#[cfg(all(target_os = "linux", not(edgefirst_tensor_dynamic)))]
const DMA_BUF_MAGIC: u32 = 0x444d_4142;

/// Filesystem magic of tmpfs, from `include/uapi/linux/magic.h`
/// (`TMPFS_MAGIC`). Covers both POSIX `shm_open` segments under `/dev/shm`
/// and anonymous `memfd_create` files.
#[cfg(all(target_os = "linux", not(edgefirst_tensor_dynamic)))]
const TMPFS_MAGIC: u32 = 0x0102_1994;

/// Normalize a raw `fstatfs` `f_type` to the 32-bit unsigned magic that
/// `include/uapi/linux/magic.h` documents, by truncating it to `u32`.
#[cfg(all(target_os = "linux", not(edgefirst_tensor_dynamic)))]
const fn fs_magic(raw: i64) -> u32 {
    raw as u32
}

/// The operations every memory backend implements.
///
/// The implementors are `DmaTensor` (Linux DMA-BUF), `IoSurfaceTensor`
/// (macOS/iOS), `AHardwareBufferTensor` (Android), `ShmTensor`, `MemTensor`,
/// and [`PboTensor`]; `TensorStorage<T>` dispatches to whichever is active.
/// Backend types are not public: allocate a [`Tensor<T>`] or [`TensorDyn`]
/// and call these methods through it.
///
/// Import the trait to get `shape`, `size`, `map`, `clone_fd`,
/// `buffer_identity`, and the zero-copy sub-region `view` on a tensor value.
///
/// [`TensorDyn`]: crate::TensorDyn
pub trait TensorTrait<T>: Send + Sync
where
    T: Num + Clone + fmt::Debug,
{
    /// Create a new tensor with the given shape and optional name. If no name
    /// is given, a random name will be generated.
    fn new(shape: &[usize], name: Option<&str>) -> Result<Self>
    where
        Self: Sized;

    #[cfg(unix)]
    /// Import an existing buffer as a tensor, taking ownership of its file
    /// descriptor. The buffer is adopted in place: no bytes are copied.
    ///
    /// The backend is detected from the fd. On Linux it is decided by the
    /// fd's filesystem magic (`include/uapi/linux/magic.h`):
    ///
    /// | Filesystem | Magic | Resulting [`TensorMemory`] |
    /// |------------|-------|----------------------------|
    /// | `dma_buf` | `DMA_BUF_MAGIC` (`0x444d4142`) | [`TensorMemory::DmaBuf`] |
    /// | `tmpfs` (`/dev/shm` **and** `memfd`) | `TMPFS_MAGIC` (`0x01021994`) | [`TensorMemory::Shm`] |
    /// | anything else | — | *rejected* — see Errors |
    ///
    /// An unrecognized filesystem is an error, not a fallback to shared
    /// memory. The device minor number is not consulted.
    ///
    /// On non-Linux Unix (macOS/iOS/Android) there is no fd-based DMA import;
    /// the fd is always adopted as [`TensorMemory::Shm`].
    ///
    /// # Arguments
    ///
    /// * `fd` - Owned descriptor for the buffer to import. Ownership
    ///   transfers to the returned tensor and the fd is closed on drop; pass
    ///   [`clone_fd`](TensorTrait::clone_fd) output to keep your own handle.
    /// * `shape` - Logical shape to interpret the buffer with. Must describe
    ///   no more elements than the buffer holds.
    /// * `name` - Optional name; a random one is generated when `None`.
    ///
    /// # Returns
    ///
    /// A tensor sharing the imported buffer's memory, whose
    /// [`memory()`](TensorTrait::memory) reports the detected backend.
    ///
    /// # Errors
    ///
    /// * [`Error::UnknownBufferType`] - the fd is on a filesystem that is
    ///   neither `dma_buf` nor `tmpfs`, so its buffer type cannot be
    ///   determined. Carries the observed `fstatfs` magic as a `u32`,
    ///   normalized so it can be looked up in `magic.h` directly on both
    ///   32- and 64-bit targets. Typical causes: a regular file, a pipe or
    ///   socket, or a `MFD_HUGETLB` memfd (hugetlbfs, not tmpfs). Linux only.
    /// * [`Error::UnknownDeviceType`] - the fd's `st_dev` major is non-zero,
    ///   i.e. it lives on a real block device rather than an anonymous or
    ///   in-memory filesystem. Linux only.
    /// * [`Error::InvalidSize`] - `shape` is empty or describes zero
    ///   elements.
    /// * [`Error::NixError`] - `fstat`, `fstatfs`, or `mmap` failed on the
    ///   descriptor.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use edgefirst_tensor::{Tensor, TensorMemory, TensorTrait};
    ///
    /// # fn main() -> edgefirst_tensor::Result<()> {
    /// let src = Tensor::<u8>::new(&[480, 640, 3], Some(TensorMemory::DmaBuf), None)?;
    ///
    /// // Round-tripping a DMA-BUF fd preserves the backend — the import is
    /// // still zero-copy, and still eligible for GPU/NPU paths.
    /// let imported = Tensor::<u8>::from_fd(src.clone_fd()?, src.shape(), None)?;
    /// assert_eq!(imported.memory(), TensorMemory::DmaBuf);
    /// # Ok(())
    /// # }
    /// ```
    fn from_fd(fd: std::os::fd::OwnedFd, shape: &[usize], name: Option<&str>) -> Result<Self>
    where
        Self: Sized;

    #[cfg(unix)]
    /// Clone the file descriptor associated with this tensor.
    fn clone_fd(&self) -> Result<std::os::fd::OwnedFd>;

    /// Get the memory type of this tensor.
    fn memory(&self) -> TensorMemory;

    /// Get the name of this tensor.
    fn name(&self) -> String;

    /// Get the number of elements in this tensor.
    fn len(&self) -> usize {
        self.shape().iter().product()
    }

    /// Check if the tensor is empty.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Get the size in bytes of this tensor.
    fn size(&self) -> usize {
        self.len() * std::mem::size_of::<T>()
    }

    /// Get the shape of this tensor.
    fn shape(&self) -> &[usize];

    /// Reshape this tensor to the given shape. The total number of elements
    /// must remain the same.
    fn reshape(&mut self, shape: &[usize]) -> Result<()>;

    /// Bytes of the underlying allocation (>= the current logical `size()`).
    fn capacity_bytes(&self) -> usize;

    /// Set the logical shape to any shape whose byte size fits the allocation
    /// capacity, without the equal-size constraint of `reshape`.
    fn set_logical_shape(&mut self, shape: &[usize]) -> Result<()>;

    /// Map the tensor into memory with the given access direction and
    /// return a TensorMap for accessing the data.
    ///
    /// `access` selects the platform mapping mode (read-only IOSurface
    /// lock, dma-buf sync direction, AHardwareBuffer lock usage) and the
    /// map's mutability: a map obtained with [`CpuAccess::Read`] rejects
    /// `as_mut_slice`. [`CpuAccess::None`] is not a mappable direction
    /// and returns [`Error::InvalidArgument`]; a direction the tensor's
    /// declared [`CpuAccess`] does not cover returns
    /// [`Error::CpuAccessDenied`].
    ///
    /// [`map_read`](Self::map_read), [`map_write`](Self::map_write) and
    /// [`map_mut`](Self::map_mut) are the typed wrappers.
    ///
    /// `'a` bounds how long the backing allocation lives; it does not borrow
    /// the tensor, so a map can coexist with a `&mut` on the same tensor.
    ///
    /// # Windows D3D11 textures: a write-only map of a partial window
    ///
    /// A [`CpuAccess::Write`] map is not refreshed from the texture first, so
    /// the bytes the caller leaves untouched are undefined.
    ///
    /// A tensor allocated [`CpuAccess::Write`] (no staging texture) publishes
    /// only the window's own rows on unmap. A sub-view (a
    /// [`view`](Tensor::view), or a shape narrowed with `set_logical_shape`)
    /// whose window is not a whole, pitch-aligned row range fails with
    /// [`Error::InvalidArgument`].
    ///
    /// On a tensor allocated [`CpuAccess::Read`] or [`CpuAccess::ReadWrite`]
    /// (staging texture), unmapping copies the whole staging texture back,
    /// and a write-only sub-view is refused with
    /// [`Error::PartialWriteRequiresReadWrite`]; map it
    /// [`CpuAccess::ReadWrite`] instead.
    fn map_with<'a>(&self, access: CpuAccess) -> Result<crate::view::HostView<'a, T>>
    where
        T: 'a;

    /// Map the tensor read-write (equivalent to
    /// `map_with(CpuAccess::ReadWrite)`).
    ///
    /// Fails with [`Error::CpuAccessDenied`] on a `Read`- or `Write`-declared
    /// tensor: use [`map_read`](Self::map_read) to read one and
    /// [`map_for_write`](Self::map_for_write) to fill one.
    fn map<'a>(&self) -> Result<crate::view::HostView<'a, T>>
    where
        T: 'a,
    {
        self.map_with(CpuAccess::ReadWrite)
    }

    /// Map the tensor for CPU reading only. The returned map rejects
    /// `as_mut_slice`; on macOS this takes the read-only IOSurface lock, on
    /// Linux the dma-buf read-direction sync. Fails with
    /// [`Error::CpuAccessDenied`] on a `Write`- or `None`-declared tensor.
    fn map_read<'a>(&self) -> Result<crate::view::HostView<'a, T>>
    where
        T: 'a,
    {
        self.map_with(CpuAccess::Read)
    }

    /// Map the tensor for CPU writing (fill-only: reading through a
    /// write map may see write-combined memory — do not read the slice).
    /// Valid only for a pass that overwrites the whole mapping; see
    /// [`write_access`](Self::write_access). Fails with
    /// [`Error::CpuAccessDenied`] on a `Read`- or `None`-declared tensor.
    fn map_write<'a>(&self) -> Result<crate::view::HostView<'a, T>>
    where
        T: 'a,
    {
        self.map_with(CpuAccess::Write)
    }

    /// Map the tensor read-write (alias of [`map`](Self::map) with the
    /// intent spelled out); refused on the same declarations.
    fn map_mut<'a>(&self) -> Result<crate::view::HostView<'a, T>>
    where
        T: 'a,
    {
        self.map_with(CpuAccess::ReadWrite)
    }

    /// The CPU access declared for this tensor (see [`CpuAccess`]). A bare
    /// backing carries no declaration of its own and reports `ReadWrite`.
    fn cpu_access(&self) -> CpuAccess {
        CpuAccess::ReadWrite
    }

    /// The parent-image snapshot when this tensor is a sub-region of a larger
    /// buffer (see [`ViewOrigin`]); a bare backing reports `None`.
    fn view_origin(&self) -> Option<ViewOrigin> {
        None
    }

    /// Whether this tensor addresses part of a larger buffer (see
    /// [`Tensor::is_view`]); a bare backing reports `false`.
    fn is_view(&self) -> bool {
        self.view_origin().is_some()
    }

    /// The direction a CPU pass that overwrites every element of this tensor
    /// maps with: a whole tensor takes its declaration plus `Write` (`Write`
    /// on a write-only declaration); a [`view`](Tensor::view) takes
    /// `ReadWrite`.
    fn write_access(&self) -> CpuAccess {
        if self.is_view() {
            CpuAccess::ReadWrite
        } else {
            self.cpu_access().union(CpuAccess::Write)
        }
    }

    /// Map for a CPU pass that overwrites every element, in
    /// [`write_access`](Self::write_access)'s direction.
    ///
    /// # Errors
    ///
    /// [`Error::CpuAccessDenied`] when the declaration does not cover that
    /// direction, otherwise whatever [`map_with`](Self::map_with) returns.
    fn map_for_write<'a>(&self) -> Result<crate::view::HostView<'a, T>>
    where
        T: 'a,
    {
        self.map_with(self.write_access())
    }

    /// Map for a CPU pass that writes only part of the buffer and keeps the
    /// rest (a letterbox border around a device-rendered interior): always
    /// `ReadWrite`, see [`write_access`](Self::write_access).
    ///
    /// # Errors
    ///
    /// [`Error::CpuAccessDenied`] unless the declaration is `ReadWrite`,
    /// otherwise whatever [`map_with`](Self::map_with) returns.
    fn map_for_partial_write<'a>(&self) -> Result<crate::view::HostView<'a, T>>
    where
        T: 'a,
    {
        self.map_with(CpuAccess::ReadWrite)
    }

    /// Non-blocking [`map_with`](Self::map_with): returns
    /// `Err(Error::IoError(WouldBlock))` while a GPU copy the map depends on
    /// is still in flight, and makes progress on a retry.
    ///
    /// Backings with no such copy (host memory, shared memory, dma-buf,
    /// IOSurface, AHardwareBuffer, PBO) alias `map_with`. Only the Windows
    /// D3D11 texture's staging refresh returns `WouldBlock`.
    ///
    /// A retry loop must yield or sleep between attempts.
    ///
    /// [`CpuAccess::Write`] is refused on a Windows D3D11 texture tensor whose
    /// window is shorter than its backing, as for
    /// [`map_with`](Self::map_with).
    fn try_map_with<'a>(&self, access: CpuAccess) -> Result<crate::view::HostView<'a, T>>
    where
        T: 'a,
    {
        self.map_with(access)
    }

    /// Non-blocking [`map`](Self::map): read-write, and `WouldBlock` rather
    /// than a stall. See [`try_map_with`](Self::try_map_with).
    fn try_map<'a>(&self) -> Result<crate::view::HostView<'a, T>>
    where
        T: 'a,
    {
        self.try_map_with(CpuAccess::ReadWrite)
    }

    /// Get the buffer identity for cache keying and liveness tracking.
    fn buffer_identity(&self) -> &BufferIdentity;

    /// Create a zero-copy sub-region view of this backing that shares the
    /// underlying allocation and [`BufferIdentity`].
    ///
    /// The window is `[offset_bytes, offset_bytes + shape.product() *
    /// size_of::<T>())` measured from this tensor's own logical start, so a
    /// sub-view of a sub-view composes by adding offsets.
    ///
    /// Defaults to [`Error::NotImplemented`]; every backend that supports
    /// sub-views overrides it (`Mem`, `Shm`, Linux DMA, macOS IOSurface, `Pbo`).
    fn view(&self, offset_bytes: usize, shape: &[usize]) -> Result<Self>
    where
        Self: Sized,
    {
        let _ = (offset_bytes, shape);
        Err(Error::NotImplemented(
            "view (zero-copy sub-region) is not supported for this tensor backend".to_owned(),
        ))
    }
}

pub trait TensorMapTrait<T>
where
    T: Num + Clone + fmt::Debug,
{
    /// Get the shape of this tensor map.
    fn shape(&self) -> &[usize];

    /// Unmap the tensor from memory.
    fn unmap(&mut self);

    /// Get the number of elements in this tensor map.
    fn len(&self) -> usize {
        self.shape().iter().product()
    }

    /// Check if the tensor map is empty.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Get the size in bytes of this tensor map.
    fn size(&self) -> usize {
        self.len() * std::mem::size_of::<T>()
    }

    /// Get a slice to the data in this tensor map.
    fn as_slice(&self) -> &[T];

    /// Get a mutable slice to the data in this tensor map.
    fn as_mut_slice(&mut self) -> &mut [T];

    /// Whether this mapping permits writes.
    fn is_writable(&self) -> bool;

    #[cfg(feature = "ndarray")]
    /// Get an ndarray ArrayView of the tensor data.
    fn view(&'_ self) -> Result<ndarray::ArrayView<'_, T, ndarray::Dim<ndarray::IxDynImpl>>> {
        Ok(ndarray::ArrayView::from_shape(
            self.shape(),
            self.as_slice(),
        )?)
    }

    #[cfg(feature = "ndarray")]
    /// Get an ndarray ArrayViewMut of the tensor data.
    fn view_mut(
        &'_ mut self,
    ) -> Result<ndarray::ArrayViewMut<'_, T, ndarray::Dim<ndarray::IxDynImpl>>> {
        let shape = self.shape().to_vec();
        Ok(ndarray::ArrayViewMut::from_shape(
            shape,
            self.as_mut_slice(),
        )?)
    }
}

// Declared through `ef_vocabulary!` (see `vocabulary.rs`): one numbering of
// backing stores shared by `code()`, the `tensor_memory_wire` constants, the
// C header's `ef_storage_kind`, the Python `TensorMemory` enum and
// `TensorDesc::kind`.
crate::ef_vocabulary! {
    /// Which memory backend a tensor is (or should be) allocated from.
    ///
    /// Pass `Some(..)` to a constructor to pin the backend, or `None` to
    /// auto-select. Auto-selection differs by constructor: [`Tensor::new`]
    /// tries `DmaBuf` → `Mem`, while the image constructors try `DmaBuf` →
    /// `Mem` and skip `Shm`. `EDGEFIRST_TENSOR_FORCE_MEM=1` short-circuits
    /// either chain to `Mem`.
    ///
    /// A pinned request has no fallback: if the backend cannot serve it, the
    /// constructor fails. [`TensorTrait::memory`] reports what a tensor
    /// ended up with.
    ///
    /// Every variant exists on every platform;
    /// [`TensorMemory::is_available`] reports at runtime whether this build
    /// can allocate one.
    ///
    /// The enum is `#[non_exhaustive]`: match it with a wildcard arm.
    #[non_exhaustive]
    pub enum TensorMemory {
        /// Regular system memory allocation. Available everywhere.
        Mem = 0, "mem", MEM,
        /// POSIX Shared Memory allocation. Suitable for inter-process
        /// communication, but not for hardware acceleration. Only
        /// allocatable on unix; nameable everywhere.
        Shm = 1, "shm", SHM,
        /// Platform-native zero-copy GPU buffer.
        ///
        /// On Linux this is a DMA-BUF allocated via the DRM/dma-heap
        /// subsystem. On macOS/iOS this variant yields an IOSurface, and on
        /// Android an AHardwareBuffer. No backend reports `IoSurface` or
        /// `D3d11Texture` as its `memory()`.
        ///
        /// Allows hardware-accelerated paths (OpenGL backend on Linux via
        /// `EGL_EXT_image_dma_buf_import`; macOS via
        /// `EGL_ANGLE_iosurface_client_buffer`).
        ///
        /// On Windows it is an `ID3D11Texture2D` on the process D3D11 device,
        /// allocated only through [`Tensor::image`](crate::Tensor::image); CPU
        /// access goes through a staging copy. The first such allocation
        /// creates the process device, so a host supplying its own device
        /// must call `d3d11::use_external_device` before it and before
        /// [`crate::is_gpu_buffer_available`].
        DmaBuf = 2, "dmabuf", DMABUF,
        /// Apple IOSurface, named specifically rather than through the
        /// portable `DmaBuf` spelling.
        ///
        /// No backend produces or accepts it: macOS/iOS allocate and report
        /// `DmaBuf`. Pinning a request to it fails with `NotImplemented`.
        IoSurface = 3, "iosurface", IOSURFACE,
        /// OpenGL Pixel Buffer Object memory. Created by `ImageProcessor`
        /// when DMA-buf is unavailable but OpenGL is present; not by
        /// [`Tensor::new`].
        Pbo = 4, "pbo", PBO,
        /// CUDA device memory: a device pointer, not host-addressable.
        ///
        /// No backend produces or accepts it; pinning a request to it fails
        /// with `NotImplemented`.
        Cuda = 5, "cuda", CUDA,
        /// Windows `ID3D11Texture2D`, named specifically rather than through
        /// the portable `DmaBuf` spelling.
        ///
        /// This is the code a Windows producer writes into
        /// [`TensorDesc::kind`](crate::TensorDesc::kind), where the handle
        /// field means an NT shared handle rather than an fd. No backend
        /// allocates or reports it as a [`TensorMemory`]: Windows allocates
        /// and reports `DmaBuf`, and pinning a request to it fails with
        /// `NotImplemented`.
        D3d11Texture = 6, "d3d11", D3D11_TEXTURE,
    }
    #[doc(hidden)]
    pub mod tensor_memory_wire;
}

impl TensorMemory {
    /// Whether this backing can be allocated in this process, now.
    ///
    /// `DmaBuf` needs `/dev/dma_heap` and permission on it, and `Shm` needs a
    /// writable `/dev/shm`. Each probe allocates once and caches.
    ///
    /// Four variants always answer `false`:
    ///
    /// * `IoSurface`: macOS/iOS serve IOSurface under the `DmaBuf` spelling,
    ///   so `DmaBuf` is the variant that answers `true` there.
    /// * `Pbo`: PBO tensors are created by `ImageProcessor::create_image`
    ///   with a GL context current, not by [`Tensor::new`].
    /// * `Cuda`: there is no CUDA backing in this crate.
    /// * `D3d11Texture`: Windows serves D3D11 textures under the `DmaBuf`
    ///   spelling.
    ///
    /// On Windows `DmaBuf` is an `ID3D11Texture2D`, which is image-formatted,
    /// so `true` there means [`Tensor::image`] can serve it and
    /// [`Tensor::new`] with `Some(DmaBuf)` answers `NotImplemented`.
    pub fn is_available(self) -> bool {
        match self {
            TensorMemory::Mem => true,
            TensorMemory::Shm => is_shm_available(),
            TensorMemory::DmaBuf => is_gpu_buffer_available(),
            TensorMemory::IoSurface
            | TensorMemory::Pbo
            | TensorMemory::Cuda
            | TensorMemory::D3d11Texture => false,
        }
    }

    /// The error to return when a caller pins a request to a backing this
    /// build knows the name of but cannot serve.
    #[cfg(not(edgefirst_tensor_dynamic))]
    pub(crate) fn unsupported_here(self) -> Error {
        let reason = match self {
            TensorMemory::Mem => "Mem is servable on every platform, so this should not happen",
            TensorMemory::Shm => "POSIX shared memory is unix-only",
            TensorMemory::DmaBuf => {
                "this target has no platform GPU-buffer allocator (DMA-BUF is Linux, \
                 IOSurface macOS/iOS, AHardwareBuffer Android, D3D11 textures on \
                 Windows)"
            }
            TensorMemory::IoSurface => {
                "no backend produces or accepts IOSurface under its own code yet -- \
                 macOS/iOS allocate and report TensorMemory::DmaBuf"
            }
            TensorMemory::Pbo => {
                "PBO tensors are created by ImageProcessor::create_image with a GL \
                 context current, never by Tensor::new"
            }
            TensorMemory::Cuda => "there is no CUDA backing in this build",
            TensorMemory::D3d11Texture => {
                "no backend produces or accepts a D3D11 texture under its own code yet -- \
                 Windows allocates and reports TensorMemory::DmaBuf"
            }
        };
        Error::NotImplemented(format!(
            "TensorMemory::{self:?} is not supported by this build: {reason}"
        ))
    }
}

impl From<TensorMemory> for String {
    /// The wire string, via [`TensorMemory::as_str`].
    ///
    /// `DmaBuf` renders as `"dmabuf"`. `TryFrom<&str>` also accepts
    /// `"dma"`.
    fn from(memory: TensorMemory) -> Self {
        memory.as_str().to_owned()
    }
}

impl TryFrom<&str> for TensorMemory {
    type Error = Error;

    /// Parses [`TensorMemory::as_str`]'s spelling, plus the `"dma"` alias for
    /// `DmaBuf`.
    fn try_from(s: &str) -> Result<Self> {
        TensorMemory::from_str_code(s)
            .or(if s == "dma" {
                Some(TensorMemory::DmaBuf)
            } else {
                None
            })
            .ok_or_else(|| Error::InvalidMemoryType(s.to_owned()))
    }
}

// The `static` backend's storage-owning `Tensor<T>`: `TensorStorage<T>` (the
// per-memory-backend enum) and the `Tensor<T>` struct that wraps it.
#[cfg(not(edgefirst_tensor_dynamic))]
#[derive(Debug)]
#[allow(dead_code)]
pub(crate) enum TensorStorage<T>
where
    T: Num + Clone + fmt::Debug + Send + Sync,
{
    /// Platform-native zero-copy GPU buffer. The inner type differs per
    /// target: `DmaTensor` on Linux, `IoSurfaceTensor` on macOS/iOS,
    /// `AHardwareBufferTensor` on Android, `D3d11TextureTensor` on Windows.
    #[cfg(target_os = "linux")]
    Dma(DmaTensor<T>),
    #[cfg(any(target_os = "macos", target_os = "ios"))]
    Dma(IoSurfaceTensor<T>),
    #[cfg(target_os = "android")]
    Dma(AHardwareBufferTensor<T>),
    #[cfg(target_os = "windows")]
    Dma(crate::d3d11::texture::D3d11TextureTensor<T>),
    #[cfg(unix)]
    Shm(ShmTensor<T>),
    Mem(MemTensor<T>),
    Pbo(PboTensor<T>),
}

#[cfg(not(edgefirst_tensor_dynamic))]
impl<T> TensorStorage<T>
where
    T: Num + Clone + fmt::Debug + Send + Sync,
{
    /// The backing allocation's intrinsic physical row pitch in bytes, if it
    /// has one that is fixed independent of the logical shape. An IOSurface,
    /// an AHardwareBuffer and a D3D11 texture report their row pitch; other
    /// backings return `None`.
    #[cfg(any(
        target_os = "macos",
        target_os = "ios",
        target_os = "android",
        target_os = "windows"
    ))]
    pub(crate) fn backing_row_stride(&self) -> Option<usize> {
        match self {
            // Only image-formatted IOSurfaces (height > 1) report a pitch.
            #[cfg(any(target_os = "macos", target_os = "ios"))]
            TensorStorage::Dma(t) => t.image_backing_row_stride(),
            // Android AHardwareBuffer: only 2D image-formatted buffers
            // (height > 1) report a pitch.
            #[cfg(target_os = "android")]
            TensorStorage::Dma(t) => t.image_backing_row_stride(),
            // Windows D3D11: the driver's staging row pitch. `None` when the
            // tensor has no staging, except for a semi-planar tensor.
            #[cfg(target_os = "windows")]
            TensorStorage::Dma(t) => t.image_backing_row_stride(),
            _ => None,
        }
    }

    /// Create a new tensor storage with the given shape, memory type, and
    /// optional name. If no name is given, a random name will be generated.
    /// If no memory type is given, the best available memory type will be
    /// chosen based on the platform and environment variables.
    fn new(shape: &[usize], memory: Option<TensorMemory>, name: Option<&str>) -> Result<Self> {
        match memory {
            #[cfg(target_os = "linux")]
            Some(TensorMemory::DmaBuf) => {
                DmaTensor::<T>::new(shape, name).map(TensorStorage::Dma)
            }
            #[cfg(any(target_os = "macos", target_os = "ios"))]
            Some(TensorMemory::DmaBuf) => {
                IoSurfaceTensor::<T>::new(shape, name).map(TensorStorage::Dma)
            }
            #[cfg(target_os = "android")]
            Some(TensorMemory::DmaBuf) => {
                AHardwareBufferTensor::<T>::new(shape, name).map(TensorStorage::Dma)
            }
            // On Windows the DmaBuf backing is a D3D11 texture, which
            // `Tensor::image` allocates.
            #[cfg(target_os = "windows")]
            Some(TensorMemory::DmaBuf) => Err(crate::error::Error::NotImplemented(
                "D3D11 texture tensors are image-formatted; use Tensor::image with \
                 TensorMemory::DmaBuf"
                    .to_owned(),
            )),
            #[cfg(not(any(
                target_os = "linux",
                target_os = "macos",
                target_os = "ios",
                target_os = "android",
                target_os = "windows"
            )))]
            Some(TensorMemory::DmaBuf) => Err(crate::error::Error::NotImplemented(
                "TensorMemory::DmaBuf is only available on Linux (DMA-BUF), macOS/iOS (IOSurface), \
                 Android (AHardwareBuffer), and Windows (D3D11 textures, via Tensor::image)"
                    .to_owned(),
            )),
            #[cfg(unix)]
            Some(TensorMemory::Shm) => {
                ShmTensor::<T>::new(shape, name).map(TensorStorage::Shm)
            }
            #[cfg(not(unix))]
            Some(m @ TensorMemory::Shm) => Err(m.unsupported_here()),
            Some(TensorMemory::Mem) => {
                MemTensor::<T>::new(shape, name).map(TensorStorage::Mem)
            }
            Some(TensorMemory::Pbo) => Err(crate::error::Error::NotImplemented(
                "PboTensor cannot be created via Tensor::new() — use ImageProcessor::create_image()".to_owned(),
            )),
            // Defined codes with no backing behind them.
            Some(
                m @ (TensorMemory::IoSurface | TensorMemory::Cuda | TensorMemory::D3d11Texture),
            ) => Err(m.unsupported_here()),
            None => {
                if std::env::var("EDGEFIRST_TENSOR_FORCE_MEM")
                    .is_ok_and(|x| x != "0" && x.to_lowercase() != "false")
                {
                    MemTensor::<T>::new(shape, name).map(TensorStorage::Mem)
                } else {
                    // Auto-select priority: Dma > Mem. Shm is not auto-selected.
                    #[cfg(target_os = "linux")]
                    {
                        // Linux: Try DMA -> Mem
                        match DmaTensor::<T>::new(shape, name) {
                            Ok(tensor) => Ok(TensorStorage::Dma(tensor)),
                            Err(_) => MemTensor::<T>::new(shape, name).map(TensorStorage::Mem),
                        }
                    }
                    #[cfg(any(target_os = "macos", target_os = "ios"))]
                    {
                        // macOS/iOS: Try IOSurface -> Mem.
                        match IoSurfaceTensor::<T>::new(shape, name) {
                            Ok(tensor) => Ok(TensorStorage::Dma(tensor)),
                            Err(_) => MemTensor::<T>::new(shape, name).map(TensorStorage::Mem),
                        }
                    }
                    #[cfg(target_os = "android")]
                    {
                        // Android: Mem.
                        MemTensor::<T>::new(shape, name).map(TensorStorage::Mem)
                    }
                    #[cfg(all(
                        unix,
                        not(any(
                            target_os = "linux",
                            target_os = "macos",
                            target_os = "ios",
                            target_os = "android"
                        ))
                    ))]
                    {
                        // Other Unix (BSD): Mem only
                        MemTensor::<T>::new(shape, name).map(TensorStorage::Mem)
                    }
                    #[cfg(not(unix))]
                    {
                        // Windows/other: Mem only
                        MemTensor::<T>::new(shape, name).map(TensorStorage::Mem)
                    }
                }
            }
        }
    }

    /// Create a DMA-backed tensor storage with an explicit byte size that
    /// may exceed `shape.product() * sizeof(T)`. Used for image tensors
    /// with row-padded layouts (see `DmaTensor::new_with_byte_size`).
    ///
    /// `policy` picks the heaps (see [`crate::dma::HeapPolicy`]): with
    /// `contiguous` the CMA heaps only (see [`ImageDesc::with_contiguous`]),
    /// and with `uncached` the uncached CMA heap first.
    #[cfg(target_os = "linux")]
    pub(crate) fn new_dma_with_byte_size(
        shape: &[usize],
        byte_size: usize,
        name: Option<&str>,
        policy: crate::dma::HeapPolicy,
    ) -> Result<Self> {
        DmaTensor::<T>::new_with_byte_size(shape, byte_size, name, policy).map(TensorStorage::Dma)
    }

    /// Create a Mem-backed tensor storage with an explicit byte size that may
    /// exceed `shape.product() * sizeof(T)`.  Used for image tensors with
    /// 64-byte-aligned row strides (see `MemTensor::with_capacity_bytes`).
    pub(crate) fn new_mem_with_byte_size(
        shape: &[usize],
        byte_size: usize,
        name: Option<&str>,
    ) -> Result<Self>
    where
        T: 'static,
    {
        MemTensor::<T>::with_capacity_bytes(shape, byte_size, name).map(TensorStorage::Mem)
    }

    /// Create a Shm-backed tensor storage with an explicit byte size that may
    /// exceed `shape.product() * sizeof(T)`.  Used for image tensors with
    /// 64-byte-aligned row strides (see `ShmTensor::new_with_byte_size`).
    #[cfg(unix)]
    pub(crate) fn new_shm_with_byte_size(
        shape: &[usize],
        byte_size: usize,
        name: Option<&str>,
    ) -> Result<Self> {
        ShmTensor::<T>::new_with_byte_size(shape, byte_size, name).map(TensorStorage::Shm)
    }

    /// Allocate an image-formatted IOSurface-backed storage (macOS).
    #[cfg(any(target_os = "macos", target_os = "ios"))]
    pub(crate) fn new_image_iosurface(
        width: usize,
        height: usize,
        format: PixelFormat,
        dtype: DType,
        shape: &[usize],
        name: Option<&str>,
    ) -> Result<Self> {
        IoSurfaceTensor::<T>::new_image(width, height, format, dtype, shape, name)
            .map(TensorStorage::Dma)
    }

    /// Allocate an image-formatted AHardwareBuffer-backed storage (Android).
    #[cfg(target_os = "android")]
    pub(crate) fn new_image_ahardwarebuffer(
        width: usize,
        height: usize,
        format: PixelFormat,
        dtype: DType,
        shape: &[usize],
        name: Option<&str>,
        access: CpuAccess,
    ) -> Result<Self> {
        AHardwareBufferTensor::<T>::new_image(width, height, format, dtype, shape, name, access)
            .map(TensorStorage::Dma)
    }

    /// Allocate an image-formatted D3D11 texture storage (Windows). A format
    /// with no texture layout is an error.
    #[cfg(target_os = "windows")]
    pub(crate) fn new_image_d3d11(
        width: usize,
        height: usize,
        format: PixelFormat,
        dtype: DType,
        shape: &[usize],
        name: Option<&str>,
        access: CpuAccess,
    ) -> Result<Self> {
        crate::d3d11::texture::D3d11TextureTensor::<T>::new_image(
            width, height, format, dtype, shape, name, access,
        )
        .map(TensorStorage::Dma)
    }

    /// Create a new tensor storage using the given file descriptor, shape,
    /// and optional name.
    #[cfg(unix)]
    fn from_fd(fd: OwnedFd, shape: &[usize], name: Option<&str>) -> Result<Self> {
        #[cfg(target_os = "linux")]
        {
            use nix::sys::stat::fstat;
            use nix::sys::statfs::fstatfs;

            let stat = fstat(&fd)?;
            let major = major(stat.st_dev);
            let minor = minor(stat.st_dev);

            if major != 0 {
                // Dma and Shm tensors are expected to have major number 0
                return Err(Error::UnknownDeviceType(major, minor));
            }

            // Classifies by filesystem magic.
            let magic = fs_magic(fstatfs(&fd)?.filesystem_type().0 as i64);

            log::debug!(
                "Creating tensor from fd: major={major}, minor={minor}, magic={magic:#010x}"
            );

            match magic {
                DMA_BUF_MAGIC => DmaTensor::<T>::from_fd(fd, shape, name).map(TensorStorage::Dma),
                TMPFS_MAGIC => ShmTensor::<T>::from_fd(fd, shape, name).map(TensorStorage::Shm),
                other => Err(Error::UnknownBufferType(other)),
            }
        }
        #[cfg(all(unix, not(target_os = "linux")))]
        {
            // On macOS/iOS/BSD the fd is adopted as SHM.
            ShmTensor::<T>::from_fd(fd, shape, name).map(TensorStorage::Shm)
        }
    }
}

#[cfg(not(edgefirst_tensor_dynamic))]
impl<T> TensorTrait<T> for TensorStorage<T>
where
    T: Num + Clone + fmt::Debug + Send + Sync,
{
    fn new(shape: &[usize], name: Option<&str>) -> Result<Self> {
        Self::new(shape, None, name)
    }

    #[cfg(unix)]
    fn from_fd(fd: OwnedFd, shape: &[usize], name: Option<&str>) -> Result<Self> {
        Self::from_fd(fd, shape, name)
    }

    #[cfg(unix)]
    fn clone_fd(&self) -> Result<OwnedFd> {
        match self {
            #[cfg(any(
                target_os = "linux",
                target_os = "macos",
                target_os = "ios",
                target_os = "android"
            ))]
            TensorStorage::Dma(t) => t.clone_fd(),
            TensorStorage::Shm(t) => t.clone_fd(),
            TensorStorage::Mem(t) => t.clone_fd(),
            TensorStorage::Pbo(t) => t.clone_fd(),
        }
    }

    fn memory(&self) -> TensorMemory {
        match self {
            #[cfg(any(
                target_os = "linux",
                target_os = "macos",
                target_os = "ios",
                target_os = "android",
                target_os = "windows"
            ))]
            TensorStorage::Dma(_) => TensorMemory::DmaBuf,
            #[cfg(unix)]
            TensorStorage::Shm(_) => TensorMemory::Shm,
            TensorStorage::Mem(_) => TensorMemory::Mem,
            TensorStorage::Pbo(_) => TensorMemory::Pbo,
        }
    }

    fn name(&self) -> String {
        match self {
            #[cfg(any(
                target_os = "linux",
                target_os = "macos",
                target_os = "ios",
                target_os = "android",
                target_os = "windows"
            ))]
            TensorStorage::Dma(t) => t.name(),
            #[cfg(unix)]
            TensorStorage::Shm(t) => t.name(),
            TensorStorage::Mem(t) => t.name(),
            TensorStorage::Pbo(t) => t.name(),
        }
    }

    fn shape(&self) -> &[usize] {
        match self {
            #[cfg(any(
                target_os = "linux",
                target_os = "macos",
                target_os = "ios",
                target_os = "android",
                target_os = "windows"
            ))]
            TensorStorage::Dma(t) => t.shape(),
            #[cfg(unix)]
            TensorStorage::Shm(t) => t.shape(),
            TensorStorage::Mem(t) => t.shape(),
            TensorStorage::Pbo(t) => t.shape(),
        }
    }

    fn reshape(&mut self, shape: &[usize]) -> Result<()> {
        match self {
            #[cfg(any(
                target_os = "linux",
                target_os = "macos",
                target_os = "ios",
                target_os = "android",
                target_os = "windows"
            ))]
            TensorStorage::Dma(t) => t.reshape(shape),
            #[cfg(unix)]
            TensorStorage::Shm(t) => t.reshape(shape),
            TensorStorage::Mem(t) => t.reshape(shape),
            TensorStorage::Pbo(t) => t.reshape(shape),
        }
    }

    fn capacity_bytes(&self) -> usize {
        match self {
            #[cfg(any(
                target_os = "linux",
                target_os = "macos",
                target_os = "ios",
                target_os = "android",
                target_os = "windows"
            ))]
            TensorStorage::Dma(t) => t.capacity_bytes(),
            #[cfg(unix)]
            TensorStorage::Shm(t) => t.capacity_bytes(),
            TensorStorage::Mem(t) => t.capacity_bytes(),
            TensorStorage::Pbo(t) => t.capacity_bytes(),
        }
    }

    fn set_logical_shape(&mut self, shape: &[usize]) -> Result<()> {
        match self {
            #[cfg(any(
                target_os = "linux",
                target_os = "macos",
                target_os = "ios",
                target_os = "android",
                target_os = "windows"
            ))]
            TensorStorage::Dma(t) => t.set_logical_shape(shape),
            #[cfg(unix)]
            TensorStorage::Shm(t) => t.set_logical_shape(shape),
            TensorStorage::Mem(t) => t.set_logical_shape(shape),
            TensorStorage::Pbo(t) => t.set_logical_shape(shape),
        }
    }

    fn map_with<'a>(&self, access: CpuAccess) -> Result<crate::view::HostView<'a, T>>
    where
        T: 'a,
    {
        match self {
            #[cfg(any(
                target_os = "linux",
                target_os = "macos",
                target_os = "ios",
                target_os = "android",
                target_os = "windows"
            ))]
            TensorStorage::Dma(t) => t.map_with(access),
            #[cfg(unix)]
            TensorStorage::Shm(t) => t.map_with(access),
            TensorStorage::Mem(t) => t.map_with(access),
            TensorStorage::Pbo(t) => t.map_with(access),
        }
    }

    fn try_map_with<'a>(&self, access: CpuAccess) -> Result<crate::view::HostView<'a, T>>
    where
        T: 'a,
    {
        match self {
            #[cfg(any(
                target_os = "linux",
                target_os = "macos",
                target_os = "ios",
                target_os = "android",
                target_os = "windows"
            ))]
            TensorStorage::Dma(t) => t.try_map_with(access),
            #[cfg(unix)]
            TensorStorage::Shm(t) => t.try_map_with(access),
            TensorStorage::Mem(t) => t.try_map_with(access),
            TensorStorage::Pbo(t) => t.try_map_with(access),
        }
    }

    fn buffer_identity(&self) -> &BufferIdentity {
        match self {
            #[cfg(any(
                target_os = "linux",
                target_os = "macos",
                target_os = "ios",
                target_os = "android",
                target_os = "windows"
            ))]
            TensorStorage::Dma(t) => t.buffer_identity(),
            #[cfg(unix)]
            TensorStorage::Shm(t) => t.buffer_identity(),
            TensorStorage::Mem(t) => t.buffer_identity(),
            TensorStorage::Pbo(t) => t.buffer_identity(),
        }
    }

    /// Forward a sub-region view to the active backend, re-wrapping the
    /// backend's view (which shares the parent's allocation and
    /// [`BufferIdentity`]) back into the matching `TensorStorage` variant.
    fn view(&self, offset_bytes: usize, shape: &[usize]) -> Result<Self> {
        match self {
            #[cfg(any(
                target_os = "linux",
                target_os = "macos",
                target_os = "ios",
                target_os = "android",
                target_os = "windows"
            ))]
            TensorStorage::Dma(t) => t.view(offset_bytes, shape).map(TensorStorage::Dma),
            #[cfg(unix)]
            TensorStorage::Shm(t) => t.view(offset_bytes, shape).map(TensorStorage::Shm),
            TensorStorage::Mem(t) => t.view(offset_bytes, shape).map(TensorStorage::Mem),
            TensorStorage::Pbo(t) => t.view(offset_bytes, shape).map(TensorStorage::Pbo),
        }
    }
}

/// Multi-backend tensor with optional image format metadata.
///
/// When `format` is `Some`, this tensor represents an image. Width, height,
/// and channels are derived from `shape` + `format`. When `format` is `None`,
/// this is a raw tensor.
///
/// This is the `static` backend's `Tensor<T>`; it owns its storage.
#[cfg(not(edgefirst_tensor_dynamic))]
#[derive(Debug)]
pub struct Tensor<T>
where
    T: Num + Clone + fmt::Debug + Send + Sync,
{
    /// CUDA registration for this tensor, if any. Declared before `storage`,
    /// so it drops first.
    cuda: Option<crate::cuda::CudaHandle>,
    pub(crate) storage: TensorStorage<T>,
    format: Option<PixelFormat>,
    chroma: Option<Box<Tensor<T>>>,
    /// Row stride in bytes for externally allocated buffers with row padding.
    /// `None` means tightly packed (stride == width * bytes_per_pixel).
    row_stride: Option<usize>,
    /// Byte offset within the DMA-BUF where image data starts.
    /// `None` means offset 0 (data starts at the beginning of the buffer).
    plane_offset: Option<usize>,
    /// Quantization metadata for integer-typed tensors.
    pub(crate) quantization: Option<Quantization>,
    /// Optional colorimetry metadata. `None` = undefined; never auto-filled.
    colorimetry: Option<crate::Colorimetry>,
    /// Declared CPU access (see [`CpuAccess`]). Image constructors set it
    /// from their `access` parameter; non-image tensors (`new`, imports,
    /// numpy) default to `ReadWrite`. Maps, pins and sync brackets beyond
    /// this declaration are refused.
    cpu_access: CpuAccess,
    /// Recorded vendor tile-compression scheme from allocation time (see
    /// [`Compression`]). `Some` only for Android hardware-only
    /// AHardwareBuffers whose allocation requested compression on an
    /// eligible format with a recognized vendor. `configure_image` preserves
    /// it; views inherit it.
    compression: Option<CompressionScheme>,
    /// Parent-image snapshot when this tensor is a [`view`](Self::view)/
    /// [`batch`](Self::batch) sub-region; `None` for a whole tensor. See
    /// [`ViewOrigin`].
    view_origin: Option<ViewOrigin>,
    /// Set by [`mark_view`](Self::mark_view) on a tensor rebuilt from a view's
    /// descriptor, which addresses part of a larger buffer but has no
    /// [`ViewOrigin`], and inherited by its sub-views. See
    /// [`is_view`](Self::is_view).
    view_marked: bool,
}

#[cfg(not(edgefirst_tensor_dynamic))]
impl<T> Tensor<T>
where
    T: Num + Clone + fmt::Debug + Send + Sync,
{
    /// Wrap a TensorStorage in a Tensor with no image metadata.
    pub(crate) fn wrap(storage: TensorStorage<T>) -> Self {
        Self {
            storage,
            format: None,
            chroma: None,
            row_stride: None,
            plane_offset: None,
            quantization: None,
            cuda: None,
            colorimetry: None,
            cpu_access: CpuAccess::ReadWrite,
            compression: None,
            view_origin: None,
            view_marked: false,
        }
    }

    /// Construct a tensor from a row-major element slice + shape. Allocates a
    /// new buffer (`TensorMemory::Mem`) and memcpys the contents; caller
    /// retains ownership of the input slice.
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
            let mut m = t.map()?;
            m.as_mut_slice().copy_from_slice(values);
        }
        Ok(t)
    }

    /// Wrap externally-owned memory as a tensor without copying. The tensor
    /// borrows `[ptr, ptr + shape.product() * size_of::<T>())` as
    /// [`TensorMemory::Mem`]; `owner`, when `Some`, co-owns the source so it
    /// outlives the tensor (and all derived views/maps). See [`ForeignOwner`].
    ///
    /// # Safety
    ///
    /// `ptr` must be non-null, aligned to `align_of::<T>()`, and valid for
    /// `shape.product()` elements of `T` for as long as the returned tensor —
    /// and every view/map sharing its backing — is alive. Pass an `owner` that
    /// co-owns the source to uphold that contract.
    ///
    /// # Errors
    ///
    /// [`Error::InvalidSize`] if `shape` is empty.
    pub unsafe fn from_foreign(
        ptr: *mut T,
        shape: &[usize],
        owner: Option<crate::ForeignOwner>,
        name: Option<&str>,
    ) -> Result<Self> {
        // SAFETY: forwarded to `from_foreign_with_capacity`'s contract; a capacity
        // of `0` clamps to the tight shape-derived footprint.
        unsafe { Self::from_foreign_with_capacity(ptr, shape, 0, owner, name) }
    }

    /// [`Self::from_foreign`], additionally recording that the allocation
    /// behind `ptr` holds at least `capacity_bytes` bytes: headroom beyond
    /// the tight `shape.product() * size_of::<T>()` footprint that a later
    /// [`configure_image`](Self::configure_image) / `set_logical_shape` can
    /// grow into. `capacity_bytes` below the tight footprint is clamped up to
    /// it.
    ///
    /// # Safety
    ///
    /// Same as [`Self::from_foreign`], except `ptr` must be valid for
    /// `capacity_bytes` bytes (or the tight shape footprint, whichever is
    /// larger), not just the shape-implied footprint.
    ///
    /// # Errors
    ///
    /// [`Error::InvalidSize`] if `shape` is empty.
    pub unsafe fn from_foreign_with_capacity(
        ptr: *mut T,
        shape: &[usize],
        capacity_bytes: usize,
        owner: Option<crate::ForeignOwner>,
        name: Option<&str>,
    ) -> Result<Self> {
        if shape.is_empty() {
            return Err(Error::InvalidSize(0));
        }
        if ptr.is_null() {
            return Err(Error::InvalidArgument(
                "from_foreign: ptr must be non-null".to_owned(),
            ));
        }
        shape
            .iter()
            .copied()
            .try_fold(1usize, usize::checked_mul)
            .ok_or_else(|| {
                Error::InvalidArgument(format!(
                    "from_foreign: shape.product() overflows usize (shape={shape:?})"
                ))
            })?;
        let capacity_elems = capacity_bytes / std::mem::size_of::<T>();
        // SAFETY: this function's contract is
        // `MemTensor::from_foreign_with_capacity`'s contract.
        let mem = unsafe {
            MemTensor::<T>::from_foreign_with_capacity(ptr, shape, capacity_elems, owner, name)
        };
        Ok(Self::wrap(TensorStorage::Mem(mem)))
    }

    /// Pin a stable host address for this tensor's data.
    ///
    /// The returned [`HostPin`] carries no borrow of the tensor. The address
    /// stays valid until the pin is dropped, including across
    /// `configure_image` and `set_logical_shape`.
    ///
    /// A pin is a snapshot: [`HostPin::len`] reports the extent at pin time
    /// and does not follow a later re-layout.
    ///
    /// The address is not a coherency guarantee: on non-coherent backends
    /// bracket CPU access with
    /// [`sync_for_cpu`](Self::sync_for_cpu) and
    /// [`sync_for_device`](Self::sync_for_device).
    ///
    /// # Errors
    ///
    /// - [`Error::InvalidArgument`] when `access` is [`CpuAccess::None`].
    /// - [`Error::CpuAccessDenied`] when the tensor's declared
    ///   [`CpuAccess`] does not cover `access`.
    /// - [`Error::NotImplemented`] for a PBO and an AHardwareBuffer, which
    ///   cannot pin; use `map()`/`map_with()` there. `Mem`, `Shm`, DMA-BUF,
    ///   IOSurface and the Windows D3D11 texture all pin. The texture pins a
    ///   persistent host buffer the storage owns, which
    ///   [`sync_for_cpu`](Self::sync_for_cpu) and
    ///   [`sync_for_device`](Self::sync_for_device) copy from and to the
    ///   texture.
    pub fn pin_host<'a>(&self, access: CpuAccess) -> Result<HostPin<'a>>
    where
        T: 'a,
    {
        self.cpu_access.require_direction(access, "pin_host")?;
        // Narrows the pin to the logical extent.
        let logical = self.len() * std::mem::size_of::<T>();
        let pin = match &self.storage {
            TensorStorage::Mem(m) => Ok(m.host_pin()),
            #[cfg(unix)]
            TensorStorage::Shm(s) => s.host_pin(),
            #[cfg(target_os = "linux")]
            TensorStorage::Dma(d) => d.host_pin(),
            #[cfg(any(target_os = "macos", target_os = "ios"))]
            TensorStorage::Dma(s) => s.host_pin(),
            // Windows D3D11: the pin is the storage's own persistent host buffer.
            #[cfg(target_os = "windows")]
            TensorStorage::Dma(t) => t.host_pin(access),
            // An AHardwareBuffer cannot pin.
            #[cfg(target_os = "android")]
            TensorStorage::Dma(_) => Err(Error::NotImplemented(
                "pin_host: AHardwareBuffer has no host address outside \
                 AHardwareBuffer_lock, so it cannot hand out one that outlives \
                 a guard. Use map()/map_with() for CPU access; for zero-copy \
                 handoff pass the buffer itself."
                    .to_owned(),
            )),
            // A PBO cannot pin.
            TensorStorage::Pbo(_) => Err(Error::NotImplemented(
                crate::pbo::PIN_HOST_PBO_REFUSAL.to_owned(),
            )),
        }?;
        Ok(pin.narrowed(logical))
    }

    /// Make CPU reads/writes of this tensor coherent: the CPU is about to
    /// access the buffer.
    ///
    /// Corresponds to `DMA_BUF_SYNC_START`. Pair with
    /// [`sync_for_device`](Self::sync_for_device).
    ///
    /// # Which backends owe maintenance
    ///
    /// | backend | behaviour |
    /// |---|---|
    /// | `Mem`, `Shm` | no-op |
    /// | `Dma` (Linux DMA-BUF) | one `DMA_BUF_IOCTL_SYNC`, direction-tagged |
    /// | `Dma` (macOS/iOS IOSurface) | `IOSurfaceLock` / `IOSurfaceUnlock`, options matched to `access` |
    /// | `Dma` (Android AHardwareBuffer) | [`Error::NotImplemented`]; use `map()`/`map_with()` |
    /// | `Dma` (Windows D3D11 texture) | refreshes the pinned host buffer from the texture through the staging copy, or uploads it back; a no-op until something pins |
    /// | `Pbo` | [`Error::NotImplemented`]; use `map()`/`map_with()` |
    ///
    /// # Errors
    ///
    /// [`Error::InvalidArgument`] when `access` is [`CpuAccess::None`];
    /// [`Error::CpuAccessDenied`] when the tensor's declared [`CpuAccess`]
    /// does not cover `access`.
    pub fn sync_for_cpu(&self, access: CpuAccess) -> Result<()> {
        self.check_sync_access(access)?;
        match &self.storage {
            TensorStorage::Mem(_) => Ok(()),
            #[cfg(unix)]
            TensorStorage::Shm(_) => Ok(()),
            #[cfg(target_os = "linux")]
            TensorStorage::Dma(d) => {
                // The direction tells the kernel which maintenance this access needs.
                crate::dmabuf::sync_access(&d.fd, true, access).map_err(Error::NixError)
            }
            #[cfg(any(target_os = "macos", target_os = "ios"))]
            TensorStorage::Dma(s) => s.sync_for_cpu(access),
            // Windows D3D11: refreshes the pinned host buffer from the texture
            // through the staging copy.
            #[cfg(target_os = "windows")]
            TensorStorage::Dma(t) => t.sync_for_cpu(access),
            #[cfg(target_os = "android")]
            TensorStorage::Dma(_) => Err(Error::NotImplemented(
                "sync_for_cpu: AHardwareBuffer has no sync independent of its \
                 lock. AHardwareBuffer_lock both establishes the address and \
                 performs the cache invalidate, and the NDK leaves locking an \
                 already-locked buffer undefined, so a bracket on top of a map \
                 guard would nest. Use map()/map_with(), which owns the pair."
                    .to_owned(),
            )),
            TensorStorage::Pbo(_) => Err(Error::NotImplemented(
                "sync_for_cpu: a PBO has no coherency window independent of its \
                 map -- glMapBufferRange establishes the address and the \
                 visibility together, and glUnmapBuffer publishes the writes. \
                 Use map()/map_with(), which owns the pair."
                    .to_owned(),
            )),
        }
    }

    /// Release the buffer back to the device: the CPU is done accessing it.
    ///
    /// Corresponds to `DMA_BUF_SYNC_END`. See
    /// [`sync_for_cpu`](Self::sync_for_cpu) for the pairing.
    pub fn sync_for_device(&self, access: CpuAccess) -> Result<()> {
        self.check_sync_access(access)?;
        match &self.storage {
            TensorStorage::Mem(_) => Ok(()),
            #[cfg(unix)]
            TensorStorage::Shm(_) => Ok(()),
            #[cfg(target_os = "linux")]
            TensorStorage::Dma(d) => {
                crate::dmabuf::sync_access(&d.fd, false, access).map_err(Error::NixError)
            }
            #[cfg(any(target_os = "macos", target_os = "ios"))]
            TensorStorage::Dma(s) => s.sync_for_device(access),
            // Windows D3D11: uploads the pinned host buffer to the texture.
            #[cfg(target_os = "windows")]
            TensorStorage::Dma(t) => t.sync_for_device(access),
            #[cfg(target_os = "android")]
            TensorStorage::Dma(_) => Err(Error::NotImplemented(
                "sync_for_device: AHardwareBuffer has no sync independent of \
                 its lock — see sync_for_cpu. Use map()/map_with()."
                    .to_owned(),
            )),
            TensorStorage::Pbo(_) => Err(Error::NotImplemented(
                "sync_for_device: a PBO has no coherency window independent of \
                 its map -- see sync_for_cpu. Use map()/map_with()."
                    .to_owned(),
            )),
        }
    }

    fn check_sync_access(&self, access: CpuAccess) -> Result<()> {
        self.cpu_access
            .require_direction(access, "sync_for_cpu/sync_for_device")
    }

    /// Construct a tensor from a 3-D ndarray view. Respects strides — one
    /// copy in all cases; contiguous views take a memcpy fast path.
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
            let mut m = t.map()?;
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

    /// Create a new tensor with the given shape, memory type, and optional
    /// name. If no name is given, a random name will be generated. If no
    /// memory type is given, the best available memory type will be chosen
    /// based on the platform and environment variables.
    ///
    /// On Linux platforms, the order of preference is: Dma -> Shm -> Mem.
    /// On other Unix platforms (macOS), the order is: Shm -> Mem.
    /// On non-Unix platforms, only Mem is available.
    ///
    /// # Environment Variables
    /// - `EDGEFIRST_TENSOR_FORCE_MEM`: If set to a non-zero and non-false
    ///   value, forces the use of regular system memory allocation
    ///   (`TensorMemory::Mem`) regardless of platform capabilities.
    ///
    /// # Example
    /// ```rust
    /// use edgefirst_tensor::{Error, Tensor, TensorMemory, TensorTrait};
    /// # fn main() -> Result<(), Error> {
    /// let tensor = Tensor::<f32>::new(&[2, 3, 4], Some(TensorMemory::Mem), Some("test_tensor"))?;
    /// assert_eq!(tensor.memory(), TensorMemory::Mem);
    /// assert_eq!(tensor.name(), "test_tensor");
    /// #    Ok(())
    /// # }
    /// ```
    pub fn new(shape: &[usize], memory: Option<TensorMemory>, name: Option<&str>) -> Result<Self> {
        let _span = tracing::trace_span!(
            "tensor.alloc",
            ?shape,
            memory = ?memory,
            dtype = std::any::type_name::<T>(),
        )
        .entered();
        #[cfg_attr(not(target_os = "linux"), allow(unused_mut))]
        let mut t = TensorStorage::new(shape, memory, name).map(Self::wrap)?;
        #[cfg(target_os = "linux")]
        t.try_init_dma_cuda();
        Ok(t)
    }

    /// Import an existing buffer as a tensor, taking ownership of its file
    /// descriptor. No bytes are copied.
    #[cfg(unix)]
    pub fn from_fd(fd: std::os::fd::OwnedFd, shape: &[usize], name: Option<&str>) -> Result<Self> {
        #[cfg_attr(not(target_os = "linux"), allow(unused_mut))]
        let mut t = Self::wrap(TensorStorage::from_fd(fd, shape, name)?);
        #[cfg(target_os = "linux")]
        t.try_init_dma_cuda();
        Ok(t)
    }

    /// Allocate an image tensor from a declarative request: the
    /// full-featured constructor behind [`Self::image`] and friends.
    ///
    /// Adds the [`Compression`] request to the classic parameters:
    ///
    /// - a request with any CPU access other than [`CpuAccess::None`] is
    ///   [`Error::InvalidArgument`];
    /// - [`Compression::Scheme`] fails with [`Error::NotImplemented`] on
    ///   platforms without vendor tile compression, and with
    ///   [`Error::InvalidArgument`] when the device's native scheme or
    ///   the format eligibility doesn't match;
    /// - [`Compression::Any`] never fails for compression reasons: it
    ///   records the scheme when the allocation is eligible and
    ///   otherwise resolves linear, incrementing
    ///   [`compression_fallback_count`].
    ///
    /// The recorded outcome is readable via [`Tensor::compression`].
    ///
    /// A [`ImageDesc::with_contiguous`] request allocates from the CMA heap
    /// only, and fails instead of falling back; [`Tensor::contiguity`]
    /// reports the heap.
    pub fn image_desc(desc: &ImageDesc) -> Result<Self>
    where
        T: 'static,
    {
        let Some(t_dtype) = dtype_of::<T>() else {
            return Err(Error::InvalidArgument(
                "image_desc: element type has no DType mapping".into(),
            ));
        };
        if t_dtype != desc.dtype {
            return Err(Error::InvalidArgument(format!(
                "image_desc: desc.dtype is {:?} but the tensor element type is {t_dtype:?}",
                desc.dtype
            )));
        }

        // A contiguous request is a CMA DMA-BUF or an error.
        if desc.contiguous {
            #[cfg(not(target_os = "linux"))]
            {
                return Err(Error::NotImplemented(
                    "image_desc: physically contiguous memory needs the Linux CMA DMA heap".into(),
                ));
            }
            #[cfg(target_os = "linux")]
            if !matches!(desc.memory, None | Some(TensorMemory::DmaBuf)) {
                return Err(Error::InvalidArgument(format!(
                    "image_desc: physically contiguous memory is DMA-BUF from the CMA heap, \
                     but {:?} was requested",
                    desc.memory
                )));
            }
        }

        // A compression request combined with any declared CPU access is an error.
        if desc.compression.is_some() && desc.access != CpuAccess::None {
            return Err(Error::InvalidArgument(format!(
                "image_desc: a compression request requires CpuAccess::None (declared {:?}) — CPU mapping pins the layout linear",
                desc.access
            )));
        }
        if let Some(Compression::Scheme(requested)) = desc.compression {
            #[cfg(not(target_os = "android"))]
            {
                return Err(Error::NotImplemented(format!(
                    "image_desc: Compression::Scheme({requested:?}) — no vendor tile compression on this platform (request Compression::Any for a portable fallback)"
                )));
            }
            #[cfg(target_os = "android")]
            {
                if matches!(
                    desc.memory,
                    Some(TensorMemory::Mem) | Some(TensorMemory::Shm) | Some(TensorMemory::Pbo)
                ) {
                    return Err(Error::InvalidArgument(format!(
                        "image_desc: Compression::Scheme({requested:?}) requires hardware memory (TensorMemory::DmaBuf or auto-select), got {:?}",
                        desc.memory
                    )));
                }
                if !crate::ahardwarebuffer_layout::compression_eligible(desc.format, desc.dtype) {
                    return Err(Error::InvalidArgument(format!(
                        "image_desc: ({:?}, {:?}) is not compression-eligible (RGBA8888 u8/i8)",
                        desc.format, desc.dtype
                    )));
                }
                let device = crate::ahardwarebuffer::device_compression_scheme();
                if device != Some(requested) {
                    return Err(Error::InvalidArgument(format!(
                        "image_desc: Compression::Scheme({requested:?}) but the device's native scheme is {device:?}"
                    )));
                }
            }
        }

        // With a compression request, auto memory tries the platform's zero-copy
        // allocation first. `Scheme` propagates its failure; `Any` falls back to
        // plain auto-select.
        #[allow(unused_mut)]
        let mut t = if desc.contiguous {
            Self::image_impl(
                desc.width,
                desc.height,
                desc.format,
                Some(TensorMemory::DmaBuf),
                desc.access,
                true,
            )?
        } else {
            match (desc.memory, desc.compression) {
                (None, Some(request)) => {
                    match Self::image(
                        desc.width,
                        desc.height,
                        desc.format,
                        Some(TensorMemory::DmaBuf),
                        desc.access,
                    ) {
                        Ok(t) => t,
                        Err(e) if matches!(request, Compression::Scheme(_)) => return Err(e),
                        Err(_) => {
                            Self::image(desc.width, desc.height, desc.format, None, desc.access)?
                        }
                    }
                }
                (memory, _) => {
                    Self::image(desc.width, desc.height, desc.format, memory, desc.access)?
                }
            }
        };

        // Records the scheme, or counts the fallback of an `Any` request that
        // resolved linear.
        if let Some(request) = desc.compression {
            #[cfg(target_os = "android")]
            {
                let eligible =
                    crate::ahardwarebuffer_layout::compression_eligible(desc.format, desc.dtype);
                let scheme = crate::ahardwarebuffer::device_compression_scheme();
                let is_ahb = t.memory() == TensorMemory::DmaBuf;
                match (request, scheme) {
                    (_, Some(s)) if eligible && is_ahb => {
                        t.set_compression_unchecked(Some(s));
                    }
                    (Compression::Scheme(requested), _) => {
                        // The allocation resolved off-AHB.
                        return Err(Error::InvalidOperation(format!(
                            "image_desc: Compression::Scheme({requested:?}) requested but the allocation resolved to {:?} (not an AHardwareBuffer)",
                            t.memory()
                        )));
                    }
                    (Compression::Any, _) => {
                        note_compression_fallback(&format!(
                            "({:?}, {:?}) {}x{}: eligible={eligible}, scheme={scheme:?}, memory={:?}",
                            desc.format,
                            desc.dtype,
                            desc.width,
                            desc.height,
                            t.memory()
                        ));
                    }
                }
            }
            #[cfg(not(target_os = "android"))]
            {
                debug_assert!(matches!(request, Compression::Any));
                let _ = request;
                note_compression_fallback(&format!(
                    "({:?}, {:?}) {}x{}: no vendor tile compression on this platform",
                    desc.format, desc.dtype, desc.width, desc.height
                ));
            }
        }
        Ok(t)
    }

    /// Allocate an image tensor of `width` × `height` in `format`.
    ///
    /// The shape follows from the format's layout — packed `[H, W, C]`, planar
    /// `[C, H, W]`, semi-planar `[H + chroma_rows, W]` — and the format is
    /// recorded on the tensor.
    ///
    /// `memory` requests a backend, or `None` to auto-select. Auto-select for
    /// images is DMA → Mem; SHM is used only when [`TensorMemory::Shm`] is
    /// requested explicitly.
    ///
    /// `access` declares what the CPU intends to do with the buffer; see
    /// [`CpuAccess`]. Hardware access (GPU, NPU, ISP, codec) is always
    /// implied. `None` keeps the allocation eligible for Android vendor tile
    /// compression; anything else pins the layout linear.
    ///
    /// DMA-backed images get a 64-byte-aligned row stride in every layout.
    /// [`Tensor::effective_row_stride`] reports the real pitch.
    ///
    /// For a compression request or any other less common knob, build an
    /// [`ImageDesc`] and call [`Tensor::image_desc`].
    ///
    /// # Errors
    ///
    /// - [`Error::InvalidArgument`] if `width` × `height` is not a valid size
    ///   for `format` (for example odd dimensions where the format's chroma
    ///   subsampling forbids them), or, on macOS with an explicit
    ///   [`TensorMemory::DmaBuf`], if the format cannot be expressed as an
    ///   image-formatted IOSurface at this width. The message names the
    ///   aligned width to use.
    /// - Whatever the chosen backend returns if the allocation itself fails —
    ///   for an explicit `memory` request there is no fallback.
    ///
    /// # Example
    ///
    /// ```rust
    /// use edgefirst_tensor::{CpuAccess, PixelFormat, Tensor, TensorMemory};
    ///
    /// # fn main() -> Result<(), edgefirst_tensor::Error> {
    /// // A camera frame the GPU converts and the CPU never touches.
    /// let frame = Tensor::<u8>::image(1920, 1080, PixelFormat::Nv12,
    ///                                 Some(TensorMemory::Mem), CpuAccess::None)?;
    /// assert_eq!(frame.format(), Some(PixelFormat::Nv12));
    /// # Ok(())
    /// # }
    /// ```
    pub fn image(
        width: usize,
        height: usize,
        format: PixelFormat,
        memory: Option<TensorMemory>,
        access: CpuAccess,
    ) -> Result<Self>
    where
        T: 'static,
    {
        Self::image_impl(width, height, format, memory, access, false)
    }

    /// [`Tensor::image`], plus the [`ImageDesc::with_contiguous`] request.
    /// `contiguous` is only honoured with `Some(TensorMemory::DmaBuf)` on
    /// Linux.
    fn image_impl(
        width: usize,
        height: usize,
        format: PixelFormat,
        memory: Option<TensorMemory>,
        access: CpuAccess,
        contiguous: bool,
    ) -> Result<Self>
    where
        T: 'static,
    {
        debug_assert!(!contiguous || memory == Some(TensorMemory::DmaBuf));
        #[cfg(not(target_os = "linux"))]
        let _ = contiguous;
        let shape = format.allocation_shape(width, height).ok_or_else(|| {
            Error::InvalidArgument(format!(
                "invalid dimensions {width}x{height} for format {format:?}"
            ))
        })?;

        // macOS Dma path: allocates a format-aware IOSurface (FourCC + 2D
        // dimensions).
        #[cfg(any(target_os = "macos", target_os = "ios"))]
        if matches!(memory, Some(TensorMemory::DmaBuf)) {
            // For planar formats the IOSurface stacks channels
            // vertically (channels * height rows), so the row stride is
            // single-channel width * sizeof(T). Packed formats keep the
            // natural width * channels * sizeof(T) stride.
            let natural_row_bytes = match format.layout() {
                PixelLayout::Planar => width * std::mem::size_of::<T>(),
                _ => width * format.channels() * std::mem::size_of::<T>(),
            };
            // A format with a real IOSurface FourCC tolerates a non-64-aligned
            // natural pitch; planar formats and formats without a FourCC require an
            // aligned pitch.
            let has_image_fourcc = dtype_of::<T>()
                .and_then(|dt| crate::iosurface_layout::image_iosurface_layout(format, dt))
                .is_some();
            let padded_ok = has_image_fourcc && format.layout() != PixelLayout::Planar;
            if !natural_row_bytes.is_multiple_of(64) && !padded_ok {
                let elem_size = std::mem::size_of::<T>();
                let per_pixel_bytes = match format.layout() {
                    PixelLayout::Planar => elem_size.max(1),
                    _ => format.channels().max(1) * elem_size.max(1),
                };
                // The suggested width is the natural row pitch rounded up to a
                // multiple of 64 and divided by per-pixel bytes; it is omitted when
                // that is not a whole pixel count.
                let aligned_row_bytes = natural_row_bytes.next_multiple_of(64);
                let pad_hint =
                    if per_pixel_bytes > 0 && aligned_row_bytes.is_multiple_of(per_pixel_bytes) {
                        let w = aligned_row_bytes / per_pixel_bytes;
                        format!("Pad width to {w} (the next 64-byte-aligned stride), ")
                    } else {
                        String::new()
                    };
                return Err(Error::InvalidArgument(format!(
                    "Tensor::image: {format:?} {width}x{height} with element \
                     size {elem_size} produces a {natural_row_bytes}-byte natural \
                     row pitch, which is not 64-byte aligned. \
                     IOSurface rounds bytes_per_row up to 64 bytes, so a \
                     contiguous CPU map of this tensor would read garbage. \
                     {pad_hint}pass memory=None to auto-fall-back to SHM, or \
                     pass memory=Some(TensorMemory::Shm) or \
                     Some(TensorMemory::Mem) explicitly."
                )));
            }
            // An explicit Dma request errors when no image-formatted IOSurface
            // exists.
            let dtype = dtype_of::<T>().ok_or_else(|| {
                Error::InvalidArgument(format!(
                    "Tensor::image: element type {} has no DType, so no \
                     image-formatted IOSurface exists. Pass memory=None or \
                     Some(TensorMemory::Mem) for a CPU tensor.",
                    std::any::type_name::<T>()
                ))
            })?;
            if crate::iosurface_layout::image_iosurface_layout(format, dtype).is_none() {
                return Err(Error::InvalidArgument(format!(
                    "Tensor::image: no zero-copy IOSurface mapping exists for \
                     {format:?}/{dtype:?} on macOS/iOS (supported: \
                     Rgba/Rgb @ U8/I8, Bgra/Yuyv/Grey/Nv12/Nv16/Nv24 @ U8, \
                     Rgba/PlanarRgb/PlanarRgba @ F16). Pass memory=None to \
                     auto-select, or Some(TensorMemory::Mem) explicitly, for \
                     a CPU tensor."
                )));
            }
            // Packed RGB u8/i8 uses an RGBA8888 surface at (W*3/4, H): rejects a
            // width the texel packing cannot express.
            if format == PixelFormat::Rgb
                && matches!(dtype, DType::U8 | DType::I8)
                && packed_rgb888_layout(width, height).is_none()
            {
                return Err(Error::InvalidArgument(format!(
                    "Tensor::image: Rgb {dtype:?} requires width%4==0 for the RGBA8888 \
                     IOSurface packing (got width={width}). Pad the width, or pass \
                     memory=Some(TensorMemory::Mem) for a CPU tensor."
                )));
            }
            let storage = TensorStorage::<T>::new_image_iosurface(
                width, height, format, dtype, &shape, None,
            )?;
            let mut t = Self::wrap(storage);
            t.format = Some(format);
            // Records the surface's `bytes_per_row` as the row stride when
            // `records_backing_pitch` says so.
            if let TensorStorage::Dma(ref io) = t.storage {
                let bpr = io.bytes_per_row();
                if let Some(natural) = t.effective_row_stride() {
                    if records_backing_pitch(format, bpr, natural) {
                        t.set_row_stride_unchecked(bpr);
                    }
                }
            }
            t.cpu_access = access;
            return Ok(t);
        }

        // Android Dma path: allocates a format-aware AHardwareBuffer.
        #[cfg(target_os = "android")]
        if matches!(memory, Some(TensorMemory::DmaBuf)) {
            // An explicit Dma request errors when no image-formatted
            // AHardwareBuffer exists.
            let dtype = dtype_of::<T>().ok_or_else(|| {
                Error::InvalidArgument(format!(
                    "Tensor::image: element type {} has no DType, so no \
                     image-formatted AHardwareBuffer exists. Pass memory=None \
                     or Some(TensorMemory::Mem) for a CPU tensor.",
                    std::any::type_name::<T>()
                ))
            })?;
            // Planar F16 requires width % 4 == 0 for the RGBA16F packing.
            if format.layout() == PixelLayout::Planar
                && dtype == DType::F16
                && packed_rgba16f_layout(format, dtype, width, height).is_none()
            {
                return Err(Error::InvalidArgument(format!(
                    "Tensor::image: {format:?} F16 requires width%4==0 for the RGBA16F \
                     AHardwareBuffer packing (got width={width}). Pad the width, or pass \
                     memory=Some(TensorMemory::Mem) for a CPU tensor."
                )));
            }
            // Packed RGB u8/i8 uses an RGBA8888 surface at (W*3/4, H), with the
            // same whole-texel constraint.
            if format == PixelFormat::Rgb
                && matches!(dtype, DType::U8 | DType::I8)
                && packed_rgb888_layout(width, height).is_none()
            {
                return Err(Error::InvalidArgument(format!(
                    "Tensor::image: Rgb {dtype:?} requires width%4==0 for the RGBA8888 \
                     AHardwareBuffer packing (got width={width}). Pad the width, or pass \
                     memory=Some(TensorMemory::Mem) for a CPU tensor."
                )));
            }
            if crate::ahardwarebuffer::image_ahardwarebuffer_layout(format, dtype).is_none() {
                return Err(Error::InvalidArgument(format!(
                    "Tensor::image: no zero-copy AHardwareBuffer mapping exists \
                     for {format:?}/{dtype:?} on Android (Grey/NV* need \
                     R8-format buffers, API 29+; the HAL floor is API 26 — see \
                     `image_ahardwarebuffer_layout`). Pass memory=None to \
                     auto-select, or Some(TensorMemory::Mem) explicitly, for a \
                     CPU tensor; camera NV12 stays zero-copy by wrapping the \
                     camera's own AHardwareBuffer instead of allocating one."
                )));
            }
            let storage = TensorStorage::<T>::new_image_ahardwarebuffer(
                width, height, format, dtype, &shape, None, access,
            )?;
            let mut t = Self::wrap(storage);
            t.format = Some(format);
            if let TensorStorage::Dma(ref ahb) = t.storage {
                // Records the buffer's row pitch as the row stride when
                // `records_backing_pitch` says so.
                let bpr = ahb.bytes_per_row();
                if let Some(natural) = t.effective_row_stride() {
                    if records_backing_pitch(format, bpr, natural) {
                        log::debug!(
                            "Tensor::image: the {format:?} AHardwareBuffer \
                             pitch is {bpr} bytes (natural {natural}); \
                             recording row stride"
                        );
                        t.set_row_stride_unchecked(bpr);
                    }
                }
            }
            t.cpu_access = access;
            return Ok(t);
        }

        // Windows Dma path: allocates an `ID3D11Texture2D` on the process
        // device. A format with no texture layout is an error.
        #[cfg(target_os = "windows")]
        if matches!(memory, Some(TensorMemory::DmaBuf)) {
            let dtype = dtype_of::<T>().ok_or_else(|| {
                Error::InvalidArgument(format!(
                    "Tensor::image: element type {} has no DType, so no D3D11 texture \
                     layout exists. Pass memory=None or Some(TensorMemory::Mem) for a \
                     CPU tensor.",
                    std::any::type_name::<T>()
                ))
            })?;
            let storage = TensorStorage::<T>::new_image_d3d11(
                width, height, format, dtype, &shape, None, access,
            )?;
            let mut t = Self::wrap(storage);
            t.format = Some(format);
            record_d3d11_row_stride(&mut t, format, "Tensor::image");
            t.cpu_access = access;
            t.try_init_d3d11_cuda();
            return Ok(t);
        }

        // The 64-byte-aligned row stride for every image layout.
        //
        // The per-layout natural pitch and total row count:
        //   * SemiPlanar `[total_h, width]`     — pitch = even(width)·elem, rows = total_h
        //   * Packed     `[height, width, ch]`  — pitch = width·ch·elem,    rows = height
        //   * Planar     `[ch, height, width]`  — pitch = width·elem,       rows = ch·height
        // Allocation byte size = `aligned_stride · total_rows`.
        let elem = std::mem::size_of::<T>();
        let channels = format.channels();
        let (natural_stride, total_rows) = match format.layout() {
            PixelLayout::SemiPlanar => (width.next_multiple_of(2) * elem, shape[0]),
            PixelLayout::Packed => (width * channels * elem, height),
            PixelLayout::Planar => (width * elem, channels * height),
        };
        let aligned_stride = natural_stride.next_multiple_of(64);
        let semi = format.layout() == PixelLayout::SemiPlanar;

        // DMA buffers and semi-planar images carry the 64-aligned pitch; packed
        // and planar images on host memory (Mem/Shm) keep the natural tight pitch.
        let host_stride = if semi { aligned_stride } else { natural_stride };
        let host_byte_size = host_stride * total_rows;
        #[cfg(target_os = "linux")]
        let dma_byte_size = aligned_stride * total_rows;

        // `used_stride` is the actual row pitch of the storage created below.
        let (storage, used_stride) = match memory {
            #[cfg(target_os = "linux")]
            Some(TensorMemory::DmaBuf) => (
                TensorStorage::<T>::new_dma_with_byte_size(
                    &shape,
                    dma_byte_size,
                    None,
                    crate::dma::HeapPolicy::for_access(access, contiguous),
                )?,
                aligned_stride,
            ),
            #[cfg(unix)]
            Some(TensorMemory::Shm) => (
                TensorStorage::<T>::new_shm_with_byte_size(&shape, host_byte_size, None)?,
                host_stride,
            ),
            Some(TensorMemory::Mem) => (
                TensorStorage::<T>::new_mem_with_byte_size(&shape, host_byte_size, None)?,
                host_stride,
            ),
            #[allow(unused_variables)]
            Some(other) => {
                // PBO and other variants: standard new().
                return {
                    let mut t = Self::new(&shape, Some(other), None)?;
                    t.format = Some(format);
                    t.cpu_access = access;
                    Ok(t)
                };
            }
            None => {
                // Auto-select priority: DMA → Mem. DMA gets the 64-aligned pitch; the
                // Mem fallback keeps the host pitch. Shm is not auto-selected.
                #[cfg(target_os = "linux")]
                {
                    match TensorStorage::<T>::new_dma_with_byte_size(
                        &shape,
                        dma_byte_size,
                        None,
                        crate::dma::HeapPolicy::for_access(access, false),
                    ) {
                        Ok(s) => (s, aligned_stride),
                        Err(_) => (
                            TensorStorage::<T>::new_mem_with_byte_size(
                                &shape,
                                host_byte_size,
                                None,
                            )?,
                            host_stride,
                        ),
                    }
                }
                #[cfg(not(target_os = "linux"))]
                {
                    (
                        TensorStorage::<T>::new_mem_with_byte_size(&shape, host_byte_size, None)?,
                        host_stride,
                    )
                }
            }
        };

        let mut t = Self::wrap(storage);
        t.format = Some(format);
        // Records the row stride when it exceeds the natural tight pitch, and for
        // every semi-planar image.
        if semi || used_stride > natural_stride {
            t.set_row_stride_unchecked(used_stride);
        }
        debug_assert!(
            t.row_stride.is_some() || !semi,
            "image() must always set row_stride for semi-planar tensors"
        );
        t.cpu_access = access;
        #[cfg(target_os = "linux")]
        t.try_init_dma_cuda();
        Ok(t)
    }

    /// Create a DMA-backed image tensor with an explicit row stride that
    /// may exceed the natural `width * channels * sizeof(T)` pitch.
    ///
    /// The underlying DMA-BUF is sized to `row_stride * height` bytes, and
    /// the tensor's logical shape stays at `[height, width, channels]`.
    /// `width()` / `height()` / `shape()` report the requested values; the
    /// padding is visible via `row_stride()` / `effective_row_stride()` and
    /// is propagated to the GL backend's EGLImage import.
    ///
    /// # Supported formats
    ///
    /// Only packed pixel layouts (RGBA8, BGRA8, RGB888, Grey, etc.).
    /// Semi-planar buffers from external allocators are imported via
    /// `TensorDyn::from_fd` + `set_row_stride`.
    ///
    /// # Supported memory
    ///
    /// Only `TensorMemory::DmaBuf`; any other memory type returns
    /// `NotImplemented`. `None` (auto-select) is treated as `Dma`.
    ///
    /// # Errors
    ///
    /// - `InvalidArgument` if `row_stride_bytes < width * channels * sizeof(T)`
    ///   (the requested stride would not fit a single row)
    /// - `NotImplemented` for non-packed formats or non-DMA memory
    /// - `IoError` if the DMA-heap allocation fails (propagated from
    ///   `DmaTensor::new_with_byte_size`)
    pub fn image_with_stride(
        width: usize,
        height: usize,
        format: PixelFormat,
        row_stride_bytes: usize,
        memory: Option<TensorMemory>,
        access: CpuAccess,
    ) -> Result<Self> {
        #[cfg(not(target_os = "linux"))]
        let _ = access;
        #[cfg(not(target_os = "linux"))]
        {
            let _ = (width, height, format, row_stride_bytes, memory);
            Err(Error::NotImplemented(
                "image_with_stride requires DMA support (Linux only)".to_owned(),
            ))
        }

        #[cfg(target_os = "linux")]
        {
            if format.layout() != PixelLayout::Packed {
                return Err(Error::NotImplemented(format!(
                    "Tensor::image_with_stride only supports packed pixel layouts, got {format:?}"
                )));
            }
            let elem = std::mem::size_of::<T>();
            let min_stride = width
                .checked_mul(format.channels())
                .and_then(|p| p.checked_mul(elem))
                .ok_or_else(|| {
                    Error::InvalidArgument(format!(
                        "image_with_stride: width {width} × channels {} × sizeof::<T>={elem} \
                         overflows usize",
                        format.channels()
                    ))
                })?;
            if row_stride_bytes < min_stride {
                return Err(Error::InvalidArgument(format!(
                    "image_with_stride: row_stride {row_stride_bytes} < minimum {min_stride} \
                     ({width} px × {} ch × {elem} B)",
                    format.channels()
                )));
            }
            let total_byte_size = row_stride_bytes.checked_mul(height).ok_or_else(|| {
                Error::InvalidArgument(format!(
                    "image_with_stride: row_stride {row_stride_bytes} × height {height} overflows usize"
                ))
            })?;

            let shape = vec![height, width, format.channels()];

            let storage = match memory {
                Some(TensorMemory::DmaBuf) | None => TensorStorage::<T>::new_dma_with_byte_size(
                    &shape,
                    total_byte_size,
                    None,
                    crate::dma::HeapPolicy::for_access(access, false),
                )?,
                Some(other) => {
                    return Err(Error::NotImplemented(format!(
                        "image_with_stride: only TensorMemory::DmaBuf is supported, got {other:?}"
                    )));
                }
            };

            let mut t = Self::wrap(storage);
            t.format = Some(format);
            t.row_stride = Some(row_stride_bytes);
            t.cpu_access = access;
            t.try_init_dma_cuda();
            Ok(t)
        }
    }

    /// Attach format metadata to an existing tensor.
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
    /// Returns `Error::InvalidShape` if the tensor shape is incompatible with
    /// the format's layout (packed expects `[H, W, C]`, planar expects
    /// `[C, H, W]`, semi-planar expects `[H*k, W]` with format-specific
    /// height constraints).
    pub fn set_format(&mut self, format: PixelFormat) -> Result<()> {
        let shape = self.shape();
        match format.layout() {
            PixelLayout::Packed => {
                if shape.len() != 3 || shape[2] != format.channels() {
                    return Err(Error::InvalidShape(format!(
                        "packed format {format:?} expects [H, W, {}], got {shape:?}",
                        format.channels()
                    )));
                }
            }
            PixelLayout::Planar => {
                if shape.len() != 3 || shape[0] != format.channels() {
                    return Err(Error::InvalidShape(format!(
                        "planar format {format:?} expects [{}, H, W], got {shape:?}",
                        format.channels()
                    )));
                }
            }
            PixelLayout::SemiPlanar => {
                if shape.len() != 2 {
                    return Err(Error::InvalidShape(format!(
                        "semi-planar format {format:?} expects [H*k, W], got {shape:?}"
                    )));
                }
                match format {
                    // Combined-plane height is `H + ceil(H/2)`: `3H/2` (≡ 0 mod 3) for
                    // even H and `(3H+1)/2` (≡ 2 mod 3) for odd H. Totals ≡ 1 mod 3
                    // are rejected.
                    PixelFormat::Nv12 if shape[0] % 3 == 1 => {
                        return Err(Error::InvalidShape(format!(
                            "NV12 contiguous shape[0] must be H + ceil(H/2) for some height; \
                             {} is unreachable (≡ 1 mod 3)",
                            shape[0]
                        )));
                    }
                    PixelFormat::Nv16 if !shape[0].is_multiple_of(2) => {
                        return Err(Error::InvalidShape(format!(
                            "NV16 contiguous shape[0] must be even, got {}",
                            shape[0]
                        )));
                    }
                    // NV24 (4:4:4): combined-plane height is 3H (Y + 2H chroma).
                    PixelFormat::Nv24 if !shape[0].is_multiple_of(3) => {
                        return Err(Error::InvalidShape(format!(
                            "NV24 contiguous shape[0] must be a multiple of 3 (= 3H), got {}",
                            shape[0]
                        )));
                    }
                    _ => {}
                }
            }
        }
        // Clears the stored stride and offset when the format changes.
        if self.format != Some(format) {
            self.row_stride = None;
            self.plane_offset = None;
            match self.storage {
                TensorStorage::Mem(ref mut m) => m.set_offset(0),
                #[cfg(target_os = "linux")]
                TensorStorage::Dma(ref mut dma) => dma.mmap_offset = 0,
                #[cfg(any(target_os = "macos", target_os = "ios"))]
                TensorStorage::Dma(ref mut io) => io.view_offset = 0,
                #[cfg(target_os = "windows")]
                TensorStorage::Dma(ref mut tex) => tex.set_view_offset(0),
                TensorStorage::Pbo(ref mut p) => p.view_offset = 0,
                #[cfg(unix)]
                TensorStorage::Shm(ref mut s) => s.set_offset(0),
                #[cfg(target_os = "android")]
                TensorStorage::Dma(_) => {}
            }
        }
        self.format = Some(format);
        Ok(())
    }

    /// Set this tensor's logical dimensions and pixel format to a decoded
    /// image, reusing the existing allocation. The shape is derived from the
    /// format layout; fails with `Error::InsufficientCapacity` if the
    /// allocation cannot hold `width`×`height` in `format`, or
    /// `Error::InvalidArgument` if the dimensions are invalid for the format.
    ///
    /// For NV12/NV16/NV24 the buffer width is rounded up to even; see
    /// [`PixelFormat::allocation_shape`].
    ///
    /// When the backing has a fixed physical row pitch (an IOSurface's
    /// 64-aligned `bytesPerRow`) that exceeds the new format's natural row
    /// stride, the physical pitch is kept as the tensor's `row_stride`.
    pub fn configure_image(
        &mut self,
        width: usize,
        height: usize,
        format: PixelFormat,
    ) -> Result<()> {
        let shape = format.allocation_shape(width, height).ok_or_else(|| {
            Error::InvalidArgument(format!(
                "invalid dimensions {width}x{height} for format {format:?}"
            ))
        })?;
        // The pre-existing row stride, read before `set_format` clears it.
        let prior_stride = self.row_stride;

        self.storage.set_logical_shape(&shape)?;
        self.set_format(format)?;
        // The new geometry's stride is chosen below.
        self.row_stride = None;

        // The row pitch for the new geometry: 64-byte-aligned for a DMA buffer of
        // any layout and for semi-planar on any backing; tight for packed/planar
        // on host memory (Mem/Shm).
        let elem = std::mem::size_of::<T>();
        let channels = format.channels();
        let (min_stride, total_rows) = match format.layout() {
            PixelLayout::SemiPlanar => (width.next_multiple_of(2) * elem, shape[0]),
            PixelLayout::Packed => (width * channels * elem, height),
            PixelLayout::Planar => (width * elem, channels * height),
        };
        let needs_align = self.storage.memory() == TensorMemory::DmaBuf
            || format.layout() == PixelLayout::SemiPlanar;

        // A backing with its own fixed pitch (IOSurface, AHardwareBuffer, a
        // staged D3D11 texture) keeps it whenever it exceeds the new row.
        #[cfg(any(
            target_os = "macos",
            target_os = "ios",
            target_os = "android",
            target_os = "windows"
        ))]
        if let Some(pitch) = self.storage.backing_row_stride() {
            let active_stride = if records_backing_pitch(format, pitch, min_stride) {
                self.set_row_stride_unchecked(pitch);
                pitch
            } else {
                min_stride
            };
            if !needs_align {
                return Ok(());
            }
            return self.check_configured_stride(active_stride, total_rows);
        }

        if !needs_align {
            return Ok(());
        }
        // Priority:
        //   1. The prior stride, when it is 64-aligned, >= this layout's minimum
        //      row, and fits the allocation.
        //   2. A fresh 64-aligned stride for the current width.
        let aligned = min_stride.next_multiple_of(64);
        let capacity = self.storage.capacity_bytes();

        let candidate = if let Some(ps) = prior_stride {
            if ps >= min_stride && ps % 64 == 0 && ps * total_rows <= capacity {
                ps
            } else {
                aligned
            }
        } else {
            aligned
        };

        let active_stride = if candidate * total_rows <= capacity {
            self.set_row_stride_unchecked(candidate);
            candidate
        } else {
            // Falls back to the tight row.
            min_stride
        };
        self.check_configured_stride(active_stride, total_rows)
    }

    /// The last step of [`configure_image`](Self::configure_image) for an
    /// aligned layout: refuses rows that do not fit the allocation.
    fn check_configured_stride(&self, active_stride: usize, total_rows: usize) -> Result<()> {
        let needed = active_stride * total_rows;
        let capacity = self.storage.capacity_bytes();
        if needed > capacity {
            return Err(Error::InsufficientCapacity { needed, capacity });
        }
        Ok(())
    }

    /// Allocate an image tensor sized to hold up to `width`×`height` in
    /// `format`, reusable for any smaller image via `configure_image`.
    pub fn image_with_capacity(
        width: usize,
        height: usize,
        format: PixelFormat,
        memory: Option<TensorMemory>,
        access: CpuAccess,
    ) -> Result<Self>
    where
        T: 'static,
    {
        Self::image(width, height, format, memory, access)
    }

    /// Pixel format (None if not an image).
    pub fn format(&self) -> Option<PixelFormat> {
        self.format
    }

    /// The format and the trailing shape dims that describe one image: packed
    /// `[H, W, C]`, planar `[C, H, W]` or semi-planar `[H*k, W]`, optionally
    /// behind a leading batch `N` (see [`batch`](Self::batch)). `None` when
    /// unformatted or when the shape has any other rank.
    fn image_dims(&self) -> Option<(PixelFormat, &[usize])> {
        let fmt = self.format?;
        let shape = self.shape();
        let rank = image_rank(fmt.layout());
        let batch_dims = shape.len().checked_sub(rank).filter(|&n| n <= 1)?;
        Some((fmt, &shape[batch_dims..]))
    }

    /// Image width (None if not an image). For a batched `[N, ...]` tensor
    /// this is the width of one element.
    pub fn width(&self) -> Option<usize> {
        let (fmt, dims) = self.image_dims()?;
        match fmt.layout() {
            PixelLayout::Packed => Some(dims[1]),
            PixelLayout::Planar => Some(dims[2]),
            PixelLayout::SemiPlanar => Some(dims[1]),
        }
    }

    /// Image height (None if not an image). For a batched `[N, ...]` tensor
    /// this is the height of one element.
    ///
    /// For semi-planar formats the combined-plane shape row count is divided
    /// by the format's luma-to-total ratio to recover the logical height.
    pub fn height(&self) -> Option<usize> {
        let (fmt, dims) = self.image_dims()?;
        match fmt.layout() {
            PixelLayout::Packed => Some(dims[0]),
            PixelLayout::Planar => Some(dims[1]),
            PixelLayout::SemiPlanar => {
                if self.is_multiplane() {
                    Some(dims[0])
                } else {
                    match fmt {
                        PixelFormat::Nv12 => Some(dims[0] * 2 / 3),
                        PixelFormat::Nv16 => Some(dims[0] / 2),
                        PixelFormat::Nv24 => Some(dims[0] / 3),
                        _ => None,
                    }
                }
            }
        }
    }

    /// Create from separate Y and UV planes (multiplane NV12/NV16).
    pub fn from_planes(luma: Tensor<T>, chroma: Tensor<T>, format: PixelFormat) -> Result<Self> {
        if format.layout() != PixelLayout::SemiPlanar {
            return Err(Error::InvalidArgument(format!(
                "from_planes requires a semi-planar format, got {format:?}"
            )));
        }
        if chroma.format.is_some() || chroma.chroma.is_some() {
            return Err(Error::InvalidArgument(
                "chroma tensor must be a raw tensor (no format or chroma metadata)".into(),
            ));
        }
        let luma_shape = luma.shape();
        let chroma_shape = chroma.shape();
        if luma_shape.len() != 2 || chroma_shape.len() != 2 {
            return Err(Error::InvalidArgument(format!(
                "from_planes expects 2D shapes, got luma={luma_shape:?} chroma={chroma_shape:?}"
            )));
        }
        if luma_shape[1] != chroma_shape[1] {
            return Err(Error::InvalidArgument(format!(
                "luma width {} != chroma width {}",
                luma_shape[1], chroma_shape[1]
            )));
        }
        match format {
            PixelFormat::Nv12 => {
                if luma_shape[0] % 2 != 0 {
                    return Err(Error::InvalidArgument(format!(
                        "NV12 requires even luma height, got {}",
                        luma_shape[0]
                    )));
                }
                if chroma_shape[0] != luma_shape[0] / 2 {
                    return Err(Error::InvalidArgument(format!(
                        "NV12 chroma height {} != luma height / 2 ({})",
                        chroma_shape[0],
                        luma_shape[0] / 2
                    )));
                }
            }
            PixelFormat::Nv16 => {
                if chroma_shape[0] != luma_shape[0] {
                    return Err(Error::InvalidArgument(format!(
                        "NV16 chroma height {} != luma height {}",
                        chroma_shape[0], luma_shape[0]
                    )));
                }
            }
            // Multiplane NV24 is not supported.
            _ => {
                return Err(Error::InvalidArgument(format!(
                    "from_planes only supports NV12 and NV16 (NV24 multiplane not yet \
                     supported — use a contiguous NV24 tensor), got {format:?}"
                )));
            }
        }

        Ok(Tensor {
            storage: luma.storage,
            format: Some(format),
            chroma: Some(Box::new(chroma)),
            row_stride: luma.row_stride,
            plane_offset: luma.plane_offset,
            quantization: luma.quantization,
            // A multiplane tensor carries no CUDA handle.
            cuda: None,
            colorimetry: luma.colorimetry,
            cpu_access: luma.cpu_access,
            compression: luma.compression,
            // A composed multiplane tensor is a whole image.
            view_origin: None,
            view_marked: false,
        })
    }

    /// Whether this tensor uses separate plane allocations.
    pub fn is_multiplane(&self) -> bool {
        self.chroma.is_some()
    }

    /// Access the chroma plane for multiplane semi-planar images.
    pub fn chroma(&self) -> Option<&Tensor<T>> {
        self.chroma.as_deref()
    }

    /// Mutable access to the chroma plane for multiplane semi-planar images.
    pub fn chroma_mut(&mut self) -> Option<&mut Tensor<T>> {
        self.chroma.as_deref_mut()
    }

    /// Independent tensor over this plane's allocation, copying geometry
    /// metadata.
    pub(crate) fn clone_chroma_plane(&self) -> Result<Self> {
        self.subview(0, self.shape())
    }

    /// Row stride in bytes (`None` = tightly packed).
    pub fn row_stride(&self) -> Option<usize> {
        self.row_stride
    }

    /// Effective row stride in bytes: the stored stride if set, otherwise the
    /// minimum stride computed from the format, width, and element size.
    /// Returns `None` only when no format is set and no explicit stride was
    /// stored via [`set_row_stride`](Self::set_row_stride).
    pub fn effective_row_stride(&self) -> Option<usize> {
        if let Some(s) = self.row_stride {
            return Some(s);
        }
        let fmt = self.format?;
        let w = self.width()?;
        let elem = std::mem::size_of::<T>();
        Some(match fmt.layout() {
            PixelLayout::Packed => w * fmt.channels() * elem,
            PixelLayout::Planar => w * elem,
            // Semi-planar: the minimum stride covers the even width.
            PixelLayout::SemiPlanar => w.next_multiple_of(2) * elem,
        })
    }

    /// Copy the tensor's logical bytes into `dst`, compacting away any
    /// recorded row-stride padding.
    ///
    /// `dst.len()` must equal the tight byte footprint (`shape` product ×
    /// element size). On a tight tensor this is one copy.
    pub fn copy_to_flat(&self, dst: &mut [u8]) -> Result<()> {
        let tight_bytes = crate::ahardwarebuffer_layout::checked_shape_bytes::<T>(self.shape())?;
        if dst.len() != tight_bytes {
            return Err(Error::InvalidArgument(format!(
                "copy_to_flat: dst is {} bytes but the tensor's tight \
                 footprint is {tight_bytes} bytes (shape {:?})",
                dst.len(),
                self.shape()
            )));
        }
        let map = self.map_read()?;
        // SAFETY: T is a plain numeric type (crate-wide bound); viewing the
        // mapped elements as bytes is sound.
        let src: &[u8] = unsafe {
            std::slice::from_raw_parts(
                map.as_slice().as_ptr() as *const u8,
                std::mem::size_of_val(map.as_slice()),
            )
        };
        let Some(stride) = self.row_stride else {
            // Tight layout: the mapped window is exactly the logical bytes.
            let Some(tight) = src.get(..tight_bytes) else {
                return Err(Error::InvalidOperation(format!(
                    "copy_to_flat: mapped {} bytes < tight footprint {tight_bytes}",
                    src.len()
                )));
            };
            dst.copy_from_slice(tight);
            return Ok(());
        };
        // Strided: the rows the strided map spans, each holding one tight
        // logical row.
        let rows = strided_rows(self)?;
        if rows == 0 {
            return Err(Error::InvalidOperation(
                "copy_to_flat: a strided tensor with no rows has nothing to copy".into(),
            ));
        }
        let row_bytes = tight_bytes / rows;
        let need = (rows - 1)
            .checked_mul(stride)
            .and_then(|b| b.checked_add(row_bytes))
            .ok_or_else(|| {
                Error::InvalidOperation(format!(
                    "copy_to_flat: stride {stride} × rows {rows} overflows usize"
                ))
            })?;
        if src.len() < need {
            return Err(Error::InvalidOperation(format!(
                "copy_to_flat: mapped {} bytes but strided rows need {need}",
                src.len()
            )));
        }
        for r in 0..rows {
            dst[r * row_bytes..(r + 1) * row_bytes]
                .copy_from_slice(&src[r * stride..r * stride + row_bytes]);
        }
        Ok(())
    }

    /// Set the row stride in bytes for externally allocated buffers with
    /// row padding (e.g. V4L2 or GStreamer allocators).
    ///
    /// The stride is propagated to the EGL DMA-BUF import attributes. Must be
    /// called after [`set_format`](Self::set_format) and before the tensor
    /// is first passed to `ImageProcessor::convert`. The stored stride
    /// is cleared if the pixel format is later changed.
    ///
    /// The stride is not validated against the backing allocation size.
    ///
    /// # Arguments
    ///
    /// * `stride` - Row stride in bytes. Must be >= the minimum stride for
    ///   the format (width * channels * sizeof(T) for packed,
    ///   width * sizeof(T) for planar/semi-planar).
    ///
    /// # Errors
    ///
    /// * `InvalidArgument` if no pixel format is set on this tensor
    /// * `InvalidArgument` if `stride` is less than the minimum for the
    ///   format and width
    pub fn set_row_stride(&mut self, stride: usize) -> Result<()> {
        let fmt = self.format.ok_or_else(|| {
            Error::InvalidArgument("cannot set row_stride without a pixel format".into())
        })?;
        let w = self.width().ok_or_else(|| {
            Error::InvalidArgument("cannot determine width for row_stride validation".into())
        })?;
        let elem = std::mem::size_of::<T>();
        let min_stride = match fmt.layout() {
            PixelLayout::Packed => w * fmt.channels() * elem,
            PixelLayout::Planar => w * elem,
            // Semi-planar: the minimum covers the even width.
            PixelLayout::SemiPlanar => w.next_multiple_of(2) * elem,
        };
        if stride < min_stride {
            return Err(Error::InvalidArgument(format!(
                "row_stride {stride} < minimum {min_stride} for {fmt:?} at width {w}"
            )));
        }
        self.row_stride = Some(stride);
        Ok(())
    }

    /// Set the row stride without format validation.
    ///
    /// For raw sub-tensors (e.g. chroma planes) that carry no format
    /// metadata. The caller is responsible for ensuring the stride is valid.
    pub fn set_row_stride_unchecked(&mut self, stride: usize) {
        self.row_stride = Some(stride);
    }

    /// Builder-style variant of [`set_row_stride`](Self::set_row_stride),
    /// consuming and returning `self`.
    ///
    /// # Errors
    ///
    /// Same conditions as [`set_row_stride`](Self::set_row_stride).
    pub fn with_row_stride(mut self, stride: usize) -> Result<Self> {
        self.set_row_stride(stride)?;
        Ok(self)
    }

    /// Byte offset within the DMA-BUF where image data starts (`None` = 0).
    pub fn plane_offset(&self) -> Option<usize> {
        self.plane_offset
    }

    /// The parent-image snapshot if this tensor is a [`view`](Self::view)/
    /// [`batch`](Self::batch) sub-region; `None` for a whole tensor. See
    /// [`ViewOrigin`].
    pub fn view_origin(&self) -> Option<ViewOrigin> {
        self.view_origin
    }

    /// Whether this tensor addresses part of a larger buffer: a
    /// [`view`](Self::view)/[`batch`](Self::batch) with a [`ViewOrigin`], or a
    /// tensor rebuilt from such a view's descriptor, which carries none.
    pub fn is_view(&self) -> bool {
        self.view_origin.is_some() || self.view_marked
    }

    /// Record that this tensor addresses part of a larger buffer, as
    /// [`is_view`](Self::is_view) reports. The mark is never cleared.
    pub fn mark_view(&mut self) {
        self.view_marked = true;
    }

    /// Set the byte offset within the DMA-BUF where image data starts.
    ///
    /// Propagated to `EGL_DMA_BUF_PLANE0_OFFSET_EXT` on GPU import. No format
    /// is required.
    pub fn set_plane_offset(&mut self, offset: usize) {
        self.plane_offset = Some(offset);
        // Writes the offset into the storage variant, which `map()` consults.
        match self.storage {
            TensorStorage::Mem(ref mut m) => m.set_offset(offset),
            #[cfg(target_os = "linux")]
            TensorStorage::Dma(ref mut dma) => dma.mmap_offset = offset,
            // The offset `IoSurfaceTensor::scoped_pin`/`map_inner` add to the
            // locked base address.
            #[cfg(any(target_os = "macos", target_os = "ios"))]
            TensorStorage::Dma(ref mut io) => io.view_offset = offset,
            // The offset every `D3d11TextureTensor` pin adds to its host or
            // staging base, in backing (pitched) space.
            #[cfg(target_os = "windows")]
            TensorStorage::Dma(ref mut tex) => tex.set_view_offset(offset),
            // The offset `PboTensor::scoped_pin` adds to the mapped base.
            TensorStorage::Pbo(ref mut p) => p.view_offset = offset,
            // The offset `ShmTensor::map_inner`/`host_pin` add to the mmap
            // base.
            #[cfg(unix)]
            TensorStorage::Shm(ref mut s) => s.set_offset(offset),
            // TODO: `AHardwareBufferTensor::view_offset` is not written here.
            #[cfg(target_os = "android")]
            TensorStorage::Dma(_) => {}
        }
    }

    /// The CPU access declared for this tensor (see [`CpuAccess`]): set at
    /// allocation, possibly narrowed by
    /// [`restrict_cpu_access`](Self::restrict_cpu_access). A view copies its
    /// parent's declaration when it is taken.
    pub fn cpu_access(&self) -> CpuAccess {
        self.cpu_access
    }

    /// Narrow the declared CPU access to the directions both it and `access`
    /// cover, on the chroma plane too. Never widens: a wider `access` leaves
    /// it unchanged.
    ///
    /// ```
    /// # use edgefirst_tensor::{CpuAccess, PixelFormat, Tensor, TensorMemory};
    /// # fn main() -> Result<(), edgefirst_tensor::Error> {
    /// let mut t = Tensor::<u8>::image(4, 4, PixelFormat::Rgba, Some(TensorMemory::Mem), CpuAccess::ReadWrite)?;
    /// t.restrict_cpu_access(CpuAccess::Read);
    /// assert_eq!(t.cpu_access(), CpuAccess::Read);
    /// t.restrict_cpu_access(CpuAccess::ReadWrite);
    /// assert_eq!(t.cpu_access(), CpuAccess::Read);
    /// # Ok(())
    /// # }
    /// ```
    pub fn restrict_cpu_access(&mut self, access: CpuAccess) {
        self.cpu_access = self.cpu_access.intersection(access);
        if let Some(chroma) = self.chroma.as_mut() {
            chroma.restrict_cpu_access(access);
        }
    }

    /// Set the declared CPU access without re-allocating.
    #[cfg(target_os = "android")]
    pub(crate) fn set_cpu_access_unchecked(&mut self, access: CpuAccess) {
        self.cpu_access = access;
    }

    /// The vendor tile-compression scheme recorded at allocation, or
    /// `None` for a linear layout. `Some` means the pixels live in a
    /// proprietary tile order: the row-stride accessors describe no
    /// meaningful linear layout and CPU maps are best-effort (see
    /// [`Compression`]). Only Android hardware-only allocations that
    /// requested compression record a scheme.
    pub fn compression(&self) -> Option<CompressionScheme> {
        self.compression
    }

    /// Whether this tensor's memory is known to be physically contiguous:
    /// [`Contiguity::Contiguous`] for a DMA-BUF this crate allocated from
    /// the CMA heap, [`Contiguity::NonContiguous`] for one from the system
    /// heap, and [`Contiguity::Unknown`] for an imported fd and every other
    /// kind of memory.
    pub fn contiguity(&self) -> Contiguity {
        match &self.storage {
            #[cfg(target_os = "linux")]
            TensorStorage::Dma(d) => d.contiguity,
            _ => Contiguity::Unknown,
        }
    }

    /// Record the compression scheme.
    #[cfg(target_os = "android")]
    pub(crate) fn set_compression_unchecked(&mut self, scheme: Option<CompressionScheme>) {
        self.compression = scheme;
    }

    pub fn colorimetry(&self) -> Option<crate::Colorimetry> {
        self.colorimetry
    }

    /// Attach/clear colorimetry metadata.
    pub fn set_colorimetry(&mut self, c: Option<crate::Colorimetry>) {
        self.colorimetry = c;
    }

    /// Builder-style colorimetry attach.
    pub fn with_colorimetry(mut self, c: crate::Colorimetry) -> Self {
        self.colorimetry = Some(c);
        self
    }

    /// Create a zero-copy sub-region view of this tensor's backing buffer.
    ///
    /// The returned tensor shares this tensor's allocation (no copy) and maps
    /// the window `[offset_bytes, offset_bytes + shape.product()*size_of::<T>())`
    /// measured from this tensor's own logical start.
    ///
    /// # Disjointness
    ///
    /// Sub-views whose byte ranges intersect alias the same memory: writing one
    /// while reading or writing the other is a data race. The caller keeps the
    /// windows disjoint; this method does not check for overlap.
    ///
    /// # Errors
    ///
    /// - [`Error::InvalidOperation`] if the backing is not `Mem` or `Dma`, or
    ///   if `offset_bytes` is not a multiple of `align_of::<T>()`.
    /// - [`Error::InsufficientCapacity`] / [`Error::InvalidSize`] if the window
    ///   exceeds the parent allocation.
    pub(crate) fn subview(&self, offset_bytes: usize, shape: &[usize]) -> Result<Tensor<T>> {
        // The offset is absolute into the backing allocation: this tensor's own
        // offset plus `offset_bytes`.
        let abs_offset = self
            .plane_offset
            .unwrap_or(0)
            .checked_add(offset_bytes)
            .ok_or(Error::InvalidSize(offset_bytes))?;
        // The storage view shares the resource and its `BufferIdentity`.
        let mut t = Tensor::wrap(self.storage.view(offset_bytes, shape)?);
        // Inherits the parent's image metadata, then sets the offset.
        if let Some(fmt) = self.format {
            t.set_format(fmt)?;
        }
        if let Some(rs) = self.row_stride {
            t.set_row_stride_unchecked(rs);
        }
        t.quantization = self.quantization.clone();
        t.set_colorimetry(self.colorimetry);
        t.cpu_access = self.cpu_access;
        t.compression = self.compression;
        t.view_marked = self.is_view();
        if abs_offset > 0 {
            t.set_plane_offset(abs_offset);
        }
        Ok(t)
    }

    /// Borrow batch element `n` of a batched tensor as a zero-copy view.
    ///
    /// A batched tensor prepends `N` as the leading dimension over the
    /// per-element image layout (`[N, H, W, C]` packed or `[N, C, H, W]`
    /// planar). `batch(n)` returns element `n`: the contiguous per-element
    /// region at byte offset `n * element_size`, sharing the parent's
    /// `BufferIdentity` and inheriting its format / row stride / colorimetry.
    /// `batch(0)` on a tensor with `N == 1` is equivalent to the whole tensor.
    ///
    /// # Errors
    ///
    /// - [`Error::BatchIndexOutOfBounds`] if `n >= N`.
    /// - [`Error::InvalidShape`] if the tensor is not batched (a formatted
    ///   tensor whose rank lacks the leading `N`, or an empty shape).
    pub fn batch(&self, n: usize) -> Result<Tensor<T>> {
        let shape = self.shape();
        // A formatted tensor must carry the leading `N` over its per-element
        // rank; a raw tensor takes shape[0] as `N`.
        if let Some(fmt) = self.format {
            let elem_rank = image_rank(fmt.layout());
            if shape.len() != elem_rank + 1 {
                return Err(Error::InvalidShape(format!(
                    "batch(): tensor is not batched ({fmt:?} expects a leading N over a \
                     {elem_rank}-D element, got shape {shape:?})"
                )));
            }
        }
        let batch = *shape
            .first()
            .ok_or_else(|| Error::InvalidShape("batch(): empty shape".into()))?;
        if n >= batch {
            return Err(Error::BatchIndexOutOfBounds { index: n, batch });
        }
        let elem_shape: Vec<usize> = shape[1..].to_vec();
        let elem_count: usize = elem_shape.iter().product();
        elem_count
            .checked_mul(std::mem::size_of::<T>())
            .ok_or(Error::InvalidSize(elem_count))?;
        // One element's footprint including row padding.
        let dims: Vec<u64> = shape.iter().map(|&d| d as u64).collect();
        let elem_bytes = crate::protocol::c_byte_strides(
            &dims,
            std::mem::size_of::<T>() as i64,
            self.format.map(|f| f.layout()),
            self.row_stride,
        )[0] as usize;
        let offset = n.checked_mul(elem_bytes).ok_or(Error::InvalidSize(n))?;
        // For a packed `[N, H, W, C]` tensor the N tiles stack vertically: the
        // parent is `(W, N*H)` and tile `n` is the row band at `y = n*H`. Other
        // layouts and raw tensors record no view origin.
        let view_origin = match self.format.map(|f| f.layout()) {
            Some(PixelLayout::Packed) => {
                let tile_h = elem_shape[0];
                let tile_w = elem_shape[1];
                // Per-row pitch of the `(W, N*H)` parent: the padded stride if set,
                // else the tight row width.
                let parent_stride = self
                    .effective_row_stride()
                    .expect("a formatted tensor of its batched rank has a width");
                Some(self.compose_view_origin(tile_w, batch * tile_h, parent_stride, 0, n * tile_h))
            }
            _ => None,
        };
        let mut t = self.subview(offset, &elem_shape)?;
        t.view_origin = view_origin;
        Ok(t)
    }

    /// Borrow a rectangular spatial sub-region of an image tensor as a
    /// zero-copy view.
    ///
    /// `region` is in pixels of the image's leading frame. The returned view
    /// shares the parent's `BufferIdentity` and addresses the sub-rectangle by
    /// offset + the parent's row pitch. `convert(src, &mut dst.view(rect), …)`
    /// renders into that sub-rectangle of `dst`.
    ///
    /// # Known limitation: bottom-right multi-row views
    ///
    /// A multi-row view is mapped as `parent_pitch × rows` bytes from its own
    /// offset, so a view that is both offset horizontally and reaches the
    /// parent's last row fails at [`map`](TensorTrait::map) time with
    /// [`Error::InsufficientCapacity`]. A single-row view keeps its own tight
    /// stride and maps.
    ///
    /// # Errors
    ///
    /// - [`Error::RegionOutOfBounds`] if `region` exceeds the image bounds.
    /// - [`Error::InvalidOperation`] if the tensor is not a packed-format image
    ///   (planar/semi-planar spatial sub-rects are not a single strided window;
    ///   use [`batch`](Self::batch) for batched planar tensors).
    pub fn view(&self, region: Region) -> Result<Tensor<T>> {
        let fmt = self.format.ok_or_else(|| {
            Error::InvalidOperation("view() requires a formatted image tensor".into())
        })?;
        if fmt.layout() != PixelLayout::Packed {
            return Err(Error::InvalidOperation(format!(
                "view() supports packed formats only (got {fmt:?}); use batch(n) for batched \
                 planar tensors"
            )));
        }
        let w = self
            .width()
            .ok_or_else(|| Error::InvalidOperation("view(): tensor has no image width".into()))?;
        let h = self
            .height()
            .ok_or_else(|| Error::InvalidOperation("view(): tensor has no image height".into()))?;
        if !region.fits_within(w, h) {
            return Err(Error::RegionOutOfBounds {
                region,
                bounds: (w, h),
            });
        }
        let elem = std::mem::size_of::<T>();
        let bpp = fmt.channels() * elem;
        let stride = self
            .effective_row_stride()
            .expect("a tensor with an image width has a row stride");
        let offset = region
            .y
            .checked_mul(stride)
            .and_then(|yo| yo.checked_add(region.x.checked_mul(bpp)?))
            .ok_or(Error::InvalidSize(region.y))?;
        let sub_shape = fmt
            .allocation_shape(region.width, region.height)
            .ok_or_else(|| Error::InvalidShape(format!("view(): invalid shape for {fmt:?}")))?;
        let mut t = self.subview(offset, &sub_shape)?;
        // A multi-row sub-rect advances rows by the parent pitch; a single-row
        // view uses its own tight stride.
        let view_stride = if region.height > 1 {
            stride
        } else {
            region.width * bpp
        };
        t.set_row_stride_unchecked(view_stride);
        // Snapshot the parent `(w, h, row_stride)` and this view's origin;
        // composes when viewing an existing view.
        t.view_origin = Some(self.compose_view_origin(w, h, stride, region.x, region.y));
        Ok(t)
    }

    /// Build the [`ViewOrigin`] for a new sub-region of `self`. When `self` is a
    /// whole tensor the snapshot names `self` as the parent; when `self` is
    /// already a view, the snapshot keeps the root parent and accumulates the
    /// local origin.
    fn compose_view_origin(
        &self,
        parent_width: usize,
        parent_height: usize,
        parent_row_stride: usize,
        x: usize,
        y: usize,
    ) -> ViewOrigin {
        match self.view_origin {
            Some(root) => ViewOrigin {
                parent_width: root.parent_width,
                parent_height: root.parent_height,
                parent_row_stride: root.parent_row_stride,
                x: root.x.saturating_add(x),
                y: root.y.saturating_add(y),
            },
            None => ViewOrigin {
                parent_width,
                parent_height,
                parent_row_stride,
                x,
                y,
            },
        }
    }

    /// The wrapped `PboTensor`. `None` when the tensor is not PBO-backed.
    pub(crate) fn as_pbo(&self) -> Option<&PboTensor<T>> {
        match &self.storage {
            TensorStorage::Pbo(p) => Some(p),
            _ => None,
        }
    }

    /// The GL buffer ID backing this tensor. `None` when the tensor is not
    /// PBO-backed.
    pub fn pbo_id(&self) -> Option<u32> {
        self.as_pbo().map(PboTensor::buffer_id)
    }

    /// Whether this PBO holds a CPU mapping; a map or unmap still in flight
    /// also answers `Some(true)`. `None` when the tensor is not PBO-backed.
    /// See [`PboTensor::is_mapped`].
    pub fn pbo_is_mapped(&self) -> Option<bool> {
        self.as_pbo().map(PboTensor::is_mapped)
    }

    /// The C-ABI `PboOpsVtable` address backing this tensor (the
    /// cross-package protocol's `ptr` under `kind::PBO`; see
    /// [`crate::protocol::TensorDesc::ptr`]). `None` when the tensor is not
    /// PBO-backed.
    pub fn pbo_vtable_ptr(&self) -> Option<*const std::ffi::c_void> {
        self.as_pbo()
            .map(|p| p.pbo_vtable() as *const _ as *const std::ffi::c_void)
    }

    /// A type-erased keepalive that must stay alive for at least as long as
    /// [`Self::pbo_vtable_ptr`]'s address is used; see
    /// [`PboTensor::pbo_keepalive`]. `None` when the tensor is not PBO-backed.
    pub fn pbo_keepalive(&self) -> Option<Arc<dyn Send + Sync>> {
        self.as_pbo().map(PboTensor::pbo_keepalive)
    }

    /// Downcast to DMA tensor reference (for EGL import, G2D).
    #[cfg(target_os = "linux")]
    pub fn as_dma(&self) -> Option<&DmaTensor<T>> {
        match &self.storage {
            TensorStorage::Dma(d) => Some(d),
            _ => None,
        }
    }

    /// Borrow the DMA-BUF file descriptor backing this tensor.
    ///
    /// # Returns
    ///
    /// A borrowed reference to the DMA-BUF file descriptor, tied to `self`'s
    /// lifetime.
    ///
    /// # Errors
    ///
    /// Returns `Error::NotImplemented` if the tensor is not DMA-backed.
    #[cfg(target_os = "linux")]
    pub fn dmabuf(&self) -> Result<std::os::fd::BorrowedFd<'_>> {
        use std::os::fd::AsFd;
        match &self.storage {
            TensorStorage::Dma(dma) => Ok(dma.fd.as_fd()),
            _ => Err(Error::NotImplemented(format!(
                "dmabuf requires DMA-backed tensor, got {:?}",
                self.storage.memory()
            ))),
        }
    }

    /// Construct a Tensor from a PBO tensor (for GL backends that allocate PBOs).
    ///
    /// Always returns `Ok` in this backend.
    pub fn from_pbo(pbo: PboTensor<T>) -> Result<Self> {
        Ok(Self {
            storage: TensorStorage::Pbo(pbo),
            format: None,
            chroma: None,
            row_stride: None,
            plane_offset: None,
            quantization: None,
            cuda: None,
            colorimetry: None,
            cpu_access: CpuAccess::ReadWrite,
            compression: None,
            view_origin: None,
            view_marked: false,
        })
    }

    /// The CUDA registration for this tensor, if any.
    pub fn cuda(&self) -> Option<&crate::cuda::CudaHandle> {
        self.cuda.as_ref()
    }

    /// Whether a CUDA registration is attached to this tensor.
    pub fn is_cuda_attached(&self) -> bool {
        self.cuda.is_some()
    }

    /// Attach a CUDA handle.
    pub fn set_cuda_handle(&mut self, h: crate::cuda::CudaHandle) {
        self.cuda = Some(h);
    }

    /// Map the tensor's CUDA device pointer, or `None` when no CUDA handle is
    /// attached.
    ///
    /// Returns a scoped [`CudaMap`] guard holding the raw CUDA device pointer
    /// for the duration of the mapping. For GL-buffer-backed tensors the unmap
    /// happens when the guard drops. When no CUDA handle is attached, returns
    /// `None` immediately.
    ///
    /// # Example — zero-copy CUDA input with host fallback
    ///
    /// ```no_run
    /// use edgefirst_tensor::{Tensor, TensorMemory, TensorTrait};
    /// # fn feed_tensorrt(_dptr: *mut std::ffi::c_void, _bytes: usize) {}
    /// # fn demo(t: &Tensor<f32>) {
    /// // Try the zero-copy CUDA device pointer first.
    /// if let Some(cuda) = t.cuda_map() {
    ///     feed_tensorrt(cuda.device_ptr(), cuda.len());
    ///     // `cuda` (a CudaMap guard) unmaps when it goes out of scope, freeing
    ///     // the GPU buffer for the next convert().
    /// } else {
    ///     // Fall back to the host mapping when no CUDA handle is attached.
    ///     let _host = t.map().expect("host map fallback must succeed");
    ///     // `_host` is a TensorMap<f32> that derefs to &[f32].
    /// }
    /// # }
    /// ```
    ///
    /// # D3D11 texture tensors: the mapping is tight rows
    ///
    /// On Windows the mapping is a linear copy of the texture laid out at
    /// `width * bytes_per_texel`, and `len()` is that times the row count,
    /// not [`row_stride`](Self::row_stride), which reports the D3D11 staging
    /// pitch a CPU map sees.
    ///
    /// For semi-planar formats (NV12, NV16, NV24) the texture is allocated as
    /// wide as its own row pitch, so the mapping equals
    /// `row_stride() x combined_height` bytes.
    pub fn cuda_map(&self) -> Option<crate::cuda::CudaMap<'_>> {
        self.cuda.as_ref()?.map()
    }

    /// Writable counterpart of [`cuda_map`](Self::cuda_map). See
    /// [`CudaHandle::map_mut`](crate::cuda::CudaHandle::map_mut) for which
    /// backings distinguish the two.
    ///
    /// On a D3D11 texture the guard's release copies the buffer back into the
    /// texture. The layout is [`cuda_map`](Self::cuda_map)'s: tight rows of
    /// `width * bytes_per_texel`, not `row_stride()`.
    pub fn cuda_map_mut(&self) -> Option<crate::cuda::CudaMap<'_>> {
        self.cuda.as_ref()?.map_mut()
    }

    /// Attempt to attach a CUDA `ExternalMemory` handle for DMA-backed tensors.
    ///
    /// On a CUDA-capable host, imports the DMA-BUF fd via
    /// `cudaImportExternalMemory(OpaqueFd)` and maps it to a device pointer.
    /// Sets `self.cuda` to a persistent `ExternalMem` handle on success. No-op
    /// if CUDA is unavailable, the tensor is not DMA-backed, or a handle is
    /// already set. Import failure is ignored: the tensor remains usable
    /// without a CUDA handle.
    #[cfg(target_os = "linux")]
    pub fn try_init_dma_cuda(&mut self) {
        if self.cuda.is_some() || !crate::cuda::is_cuda_available() {
            return;
        }
        let (raw_fd, buf_size) = match &self.storage {
            TensorStorage::Dma(dma) => {
                use std::os::fd::AsRawFd;
                (dma.fd.as_raw_fd(), dma.buf_size)
            }
            _ => return,
        };
        if let Some((ext, dptr)) = crate::cuda::import_dma_fd(raw_fd, buf_size) {
            self.cuda = Some(crate::cuda::CudaHandle::new_external(ext, dptr, buf_size));
        }
    }

    /// Attempt to attach a CUDA external-memory handle for D3D11 texture
    /// tensors: the Windows counterpart of
    /// [`try_init_dma_cuda`](Self::try_init_dma_cuda).
    ///
    /// Imports the texture's shared NT handle via
    /// `cudaImportExternalMemory(D3D11Resource)` and the process device's
    /// fence as an external semaphore, so [`cuda_map`](Self::cuda_map) hands
    /// out a linear device buffer whose refresh is ordered after the D3D11
    /// work a producer recorded with [`set_gpu_write`](Self::set_gpu_write).
    ///
    /// No-op when a handle is already attached or the tensor is not
    /// texture-backed. Import failure is ignored: the tensor stays valid and
    /// its `cuda_map()` is `None`.
    #[cfg(target_os = "windows")]
    pub fn try_init_d3d11_cuda(&mut self) {
        if self.cuda.is_some() {
            return;
        }
        let TensorStorage::Dma(tex) = &self.storage else {
            return;
        };
        if let Some(h) = tex.try_init_cuda() {
            self.cuda = Some(h);
        }
    }
}

/// Windows D3D11 texture accessors.
#[cfg(all(not(edgefirst_tensor_dynamic), target_os = "windows"))]
impl<T> Tensor<T>
where
    T: Num + Clone + fmt::Debug + Send + Sync,
{
    /// The borrowed `ID3D11Texture2D*` backing this tensor, valid while the
    /// tensor lives. `None` when the tensor is not texture-backed.
    pub fn d3d11_texture(&self) -> Option<*mut std::ffi::c_void> {
        match &self.storage {
            TensorStorage::Dma(t) => Some(t.texture_ptr()),
            _ => None,
        }
    }

    /// The texture geometry the HAL chose for this image: DXGI format, texel
    /// dimensions and the matching GL internal format. `None` when the tensor
    /// is not texture-backed.
    pub fn d3d11_layout(&self) -> Option<crate::d3d11_layout::D3d11ImageLayout> {
        match &self.storage {
            TensorStorage::Dma(t) => Some(t.layout()),
            _ => None,
        }
    }

    /// A duplicated NT handle the caller owns: open it on a D3D12 device,
    /// another D3D11 device, or CUDA, or duplicate it into another process.
    /// Closing it does not affect this tensor.
    pub fn d3d11_shared_handle(&self) -> Result<std::os::windows::io::OwnedHandle> {
        match &self.storage {
            TensorStorage::Dma(t) => t.shared_handle(),
            _ => Err(Error::NotImplemented(
                "not a D3D11 texture tensor".to_owned(),
            )),
        }
    }

    /// The fence handle plus value a GPU consumer waits on before reading this
    /// texture, or `None` when no GPU write has been recorded on it.
    pub fn gpu_completion(&self) -> Result<Option<crate::d3d11::GpuCompletion>> {
        match &self.storage {
            TensorStorage::Dma(t) => t.gpu_completion(),
            _ => Err(Error::NotImplemented(
                "not a D3D11 texture tensor".to_owned(),
            )),
        }
    }

    /// Record that GPU work writing this texture completes at `value` of the
    /// process device's fence. Monotonic: an older value never displaces a
    /// newer one.
    pub fn set_gpu_write(&self, value: u64) -> Result<()> {
        match &self.storage {
            TensorStorage::Dma(t) => {
                t.set_gpu_write(value);
                Ok(())
            }
            _ => Err(Error::NotImplemented(
                "not a D3D11 texture tensor".to_owned(),
            )),
        }
    }

    /// The tensor's own NT handle value, not a duplicate.
    pub(crate) fn d3d11_shared_handle_value(&self) -> Option<usize> {
        match &self.storage {
            TensorStorage::Dma(t) => Some(t.shared_handle_value()),
            _ => None,
        }
    }
}

/// The recorded GPU write value, on every platform.
#[cfg(not(edgefirst_tensor_dynamic))]
impl<T> Tensor<T>
where
    T: Num + Clone + fmt::Debug + Send + Sync,
{
    /// The fence value of the newest GPU write recorded on this tensor, or 0
    /// when there is none: the `value` of `gpu_completion` without the
    /// duplicated fence handle. 0 for a tensor that is not a D3D11 texture,
    /// and on every platform other than Windows.
    pub fn gpu_write_value(&self) -> u64 {
        #[cfg(target_os = "windows")]
        {
            if let TensorStorage::Dma(t) = &self.storage {
                return t.last_gpu_write();
            }
        }
        0
    }
}

/// Whether a pitched backing's row `pitch` is recorded as the row stride of
/// an image whose format and width give the `natural` stride: a wider pitch
/// always, an equal pitch for a semi-planar format, a narrower pitch never.
#[cfg(all(
    not(edgefirst_tensor_dynamic),
    any(
        target_os = "macos",
        target_os = "ios",
        target_os = "android",
        target_os = "windows"
    )
))]
fn records_backing_pitch(format: PixelFormat, pitch: usize, natural: usize) -> bool {
    pitch > natural || (pitch == natural && format.layout() == PixelLayout::SemiPlanar)
}

/// Records the D3D11 backing's row pitch on `t` when its format and width
/// call for it (see [`records_backing_pitch`]).
///
/// `t.format` must already be set.
#[cfg(all(not(edgefirst_tensor_dynamic), target_os = "windows"))]
fn record_d3d11_row_stride<T>(t: &mut Tensor<T>, format: PixelFormat, what: &str)
where
    T: Num + Clone + fmt::Debug + Send + Sync,
{
    let pitch = match &t.storage {
        TensorStorage::Dma(tex) => tex.image_backing_row_stride(),
        _ => None,
    };
    let (Some(pitch), Some(natural)) = (pitch, t.effective_row_stride()) else {
        return;
    };
    if records_backing_pitch(format, pitch, natural) {
        log::debug!(
            "{what}: the D3D11 backing pitch for the {format:?} texture is {pitch} bytes \
             (natural {natural}); recording row stride"
        );
        t.set_row_stride_unchecked(pitch);
    }
}

/// The two texture-importing constructors.
#[cfg(all(not(edgefirst_tensor_dynamic), target_os = "windows"))]
impl<T> Tensor<T>
where
    T: Num + Clone + fmt::Debug + Send + Sync + 'static,
{
    /// Wrap an existing `ID3D11Texture2D` as a tensor. The texture is checked
    /// against the layout the HAL would have chosen for `format`/`width`/
    /// `height` and must live on the HAL's process device. Ownership stays
    /// with the caller: this takes its own reference.
    ///
    /// The backing's row pitch is recorded as the tensor's
    /// [`row_stride`](Self::row_stride) when it exceeds the natural stride.
    ///
    /// # Safety
    ///
    /// `texture` must be null or a live `ID3D11Texture2D` created on the HAL
    /// device ([`crate::d3d11::device()`]).
    pub unsafe fn from_d3d11_texture(
        texture: *mut std::ffi::c_void,
        width: usize,
        height: usize,
        format: PixelFormat,
        access: CpuAccess,
        name: Option<&str>,
    ) -> Result<Self> {
        let dtype = dtype_of::<T>().ok_or_else(|| {
            Error::InvalidArgument(format!(
                "from_d3d11_texture: element type {} has no DType",
                std::any::type_name::<T>()
            ))
        })?;
        let shape = format.allocation_shape(width, height).ok_or_else(|| {
            Error::InvalidArgument(format!(
                "invalid dimensions {width}x{height} for format {format:?}"
            ))
        })?;
        // SAFETY: the caller guarantees `texture` is null or a live
        // `ID3D11Texture2D`; the storage takes its own reference.
        let storage = unsafe {
            crate::d3d11::texture::D3d11TextureTensor::<T>::from_d3d11_texture(
                texture, width, height, format, dtype, &shape, access, name,
            )
        }?;
        let mut t = Self::wrap(TensorStorage::Dma(storage));
        t.format = Some(format);
        record_d3d11_row_stride(&mut t, format, "from_d3d11_texture");
        t.cpu_access = access;
        t.try_init_d3d11_cuda();
        Ok(t)
    }

    /// Open a shared texture by its NT handle on the HAL's device. When
    /// `completion` is given, its fence value is waited for on the immediate
    /// context. Both handles stay owned by the caller: this duplicates what
    /// it keeps.
    ///
    /// Records the backing's row pitch as
    /// [`row_stride`](Self::row_stride) on the same rule as
    /// [`from_d3d11_texture`](Self::from_d3d11_texture).
    ///
    /// # Safety
    ///
    /// `handle` must be an NT shared handle of a D3D11 texture, valid in this
    /// process, and `completion`'s handle a shared fence handle.
    pub unsafe fn from_d3d11_shared_handle(
        handle: std::os::windows::io::RawHandle,
        width: usize,
        height: usize,
        format: PixelFormat,
        access: CpuAccess,
        completion: Option<(std::os::windows::io::RawHandle, u64)>,
        name: Option<&str>,
    ) -> Result<Self> {
        let dtype = dtype_of::<T>().ok_or_else(|| {
            Error::InvalidArgument(format!(
                "from_d3d11_shared_handle: element type {} has no DType",
                std::any::type_name::<T>()
            ))
        })?;
        let shape = format.allocation_shape(width, height).ok_or_else(|| {
            Error::InvalidArgument(format!(
                "invalid dimensions {width}x{height} for format {format:?}"
            ))
        })?;
        // SAFETY: the caller guarantees the handles are valid in this process.
        let storage = unsafe {
            crate::d3d11::texture::D3d11TextureTensor::<T>::from_d3d11_shared_handle(
                handle, width, height, format, dtype, &shape, access, completion, name,
            )
        }?;
        let mut t = Self::wrap(TensorStorage::Dma(storage));
        t.format = Some(format);
        record_d3d11_row_stride(&mut t, format, "from_d3d11_shared_handle");
        t.cpu_access = access;
        t.try_init_d3d11_cuda();
        Ok(t)
    }
}

// Quantization accessors, available on integer element types only.
#[cfg(not(edgefirst_tensor_dynamic))]
impl<T> Tensor<T>
where
    T: IntegerType + Num + Clone + fmt::Debug + Send + Sync,
{
    /// Quantization metadata for this tensor, if set.
    pub fn quantization(&self) -> Option<&Quantization> {
        self.quantization.as_ref()
    }

    /// Attach quantization metadata to this tensor. Validates against the
    /// tensor's shape — returns [`Error::QuantizationInvalid`] on any
    /// inconsistency (mismatched scale/zp lengths, out-of-range axis, etc.).
    pub fn set_quantization(&mut self, q: Quantization) -> Result<()> {
        q.validate(self.shape())?;
        self.quantization = Some(q);
        Ok(())
    }

    /// Builder-style variant of [`Self::set_quantization`]. Consumes `self`;
    /// on validation failure returns [`Error::QuantizationInvalid`] and drops
    /// the tensor.
    pub fn with_quantization(mut self, q: Quantization) -> Result<Self> {
        self.set_quantization(q)?;
        Ok(self)
    }

    /// Clear any quantization metadata on this tensor.
    pub fn clear_quantization(&mut self) {
        self.quantization = None;
    }
}

/// Rank of one image's shape in `layout`: `[H, W, C]` / `[C, H, W]` are 3-D,
/// semi-planar `[H*k, W]` is 2-D. A batched tensor has one more.
#[cfg(not(edgefirst_tensor_dynamic))]
fn image_rank(layout: PixelLayout) -> usize {
    match layout {
        PixelLayout::SemiPlanar => 2,
        PixelLayout::Packed | PixelLayout::Planar => 3,
    }
}

#[cfg(not(edgefirst_tensor_dynamic))]
fn map_strided_host_view<'a, T>(
    tensor: &Tensor<T>,
    stride: usize,
    access: CpuAccess,
    non_blocking: bool,
) -> Result<crate::view::HostView<'a, T>>
where
    T: Num + Clone + fmt::Debug + Send + Sync + 'a,
{
    // Rows sit at `stride`-byte spacing.
    let rows = strided_rows(tensor)?;
    let total_bytes = stride.checked_mul(rows).ok_or_else(|| {
        Error::InvalidOperation(format!(
            "Tensor::map: row_stride {stride} × rows {rows} overflows usize"
        ))
    })?;
    map_strided_storage(tensor, stride, rows, total_bytes, access, non_blocking)
}

/// Rows a strided map or flat copy of `tensor` spans at its row stride: a
/// raw tensor's leading dim, else every image row of every batch element --
/// `H` per packed image, `C * H` per planar one (C planes of H rows), and the
/// combined `H*k` per semi-planar one.
#[cfg(not(edgefirst_tensor_dynamic))]
fn strided_rows<T>(tensor: &Tensor<T>) -> Result<usize>
where
    T: Num + Clone + fmt::Debug + Send + Sync,
{
    let shape = tensor.shape();
    let Some(fmt) = tensor.format else {
        return shape.first().copied().ok_or_else(|| {
            Error::InvalidOperation("strided access requires a non-empty shape".into())
        });
    };
    let (_, dims) = tensor.image_dims().ok_or_else(|| {
        Error::InvalidOperation(format!(
            "strided access to a {fmt:?} tensor requires its image shape, optionally \
             batched; got {shape:?}"
        ))
    })?;
    let per_image = match fmt.layout() {
        PixelLayout::Planar => dims[0] * dims[1],
        PixelLayout::Packed | PixelLayout::SemiPlanar => dims[0],
    };
    let batch: usize = shape[..shape.len() - dims.len()].iter().product();
    Ok(batch * per_image)
}

#[cfg(not(edgefirst_tensor_dynamic))]
fn map_strided_storage<'a, T>(
    tensor: &Tensor<T>,
    _stride: usize,
    _rows: usize,
    total_bytes: usize,
    access: CpuAccess,
    _non_blocking: bool,
) -> Result<crate::view::HostView<'a, T>>
where
    T: Num + Clone + fmt::Debug + Send + Sync + 'a,
{
    match &tensor.storage {
        #[cfg(target_os = "linux")]
        TensorStorage::Dma(dma) => {
            // Bounds-checks `stride × rows` against the DMA-BUF size.
            let available_bytes = dma.buf_size.saturating_sub(dma.mmap_offset);
            if total_bytes > available_bytes {
                return Err(Error::InvalidOperation(format!(
                    "Tensor::map: strided mapping needs {total_bytes} bytes \
                     but DMA buffer only has {available_bytes} available \
                     (buf_size={}, mmap_offset={}, stride={_stride}, rows={_rows}); \
                     the row_stride was likely set larger than the original allocation",
                    dma.buf_size, dma.mmap_offset
                )));
            }
            dma.map_with_byte_size(total_bytes, access)
        }
        TensorStorage::Mem(mem) => {
            let capacity = tensor.storage.capacity_bytes();
            if total_bytes > capacity {
                return Err(Error::InsufficientCapacity {
                    needed: total_bytes,
                    capacity,
                });
            }
            mem.map_with_byte_size(total_bytes, access)
        }
        #[cfg(unix)]
        TensorStorage::Shm(shm) => {
            let capacity = tensor.storage.capacity_bytes();
            if total_bytes > capacity {
                return Err(Error::InsufficientCapacity {
                    needed: total_bytes,
                    capacity,
                });
            }
            shm.map_with_byte_size(total_bytes, access)
        }
        // macOS/iOS: `TensorStorage::Dma` is the IOSurface.
        #[cfg(any(target_os = "macos", target_os = "ios"))]
        TensorStorage::Dma(io) => {
            // A sub-view's window is `buf_size − view_offset`.
            let available = io.buf_size.saturating_sub(io.view_offset);
            if total_bytes > available {
                return Err(Error::InsufficientCapacity {
                    needed: total_bytes,
                    capacity: available,
                });
            }
            io.map_with_byte_size(total_bytes, access)
        }
        // Android: `TensorStorage::Dma` is the AHardwareBuffer.
        #[cfg(target_os = "android")]
        TensorStorage::Dma(ahb) => {
            // A sub-view's window is `buf_size − view_offset`.
            let available = ahb.buf_size.saturating_sub(ahb.view_offset);
            if total_bytes > available {
                return Err(Error::InsufficientCapacity {
                    needed: total_bytes,
                    capacity: available,
                });
            }
            ahb.map_with_byte_size(total_bytes, access)
        }
        // Windows: `TensorStorage::Dma` is the D3D11 texture, mapped through its
        // staging copy; `map_with_byte_size` bounds-checks the span.
        #[cfg(target_os = "windows")]
        TensorStorage::Dma(tex) => {
            if _non_blocking {
                tex.try_map_with_byte_size(total_bytes, access)
            } else {
                tex.map_with_byte_size(total_bytes, access)
            }
        }
        TensorStorage::Pbo(pbo) => {
            // Maps the full padded buffer. A sub-view's window is
            // `capacity − view_offset`.
            let available = pbo.capacity_bytes().saturating_sub(pbo.view_offset);
            if total_bytes > available {
                return Err(Error::InsufficientCapacity {
                    needed: total_bytes,
                    capacity: available,
                });
            }
            pbo.map_with_byte_size(total_bytes, access)
        }
        #[allow(unreachable_patterns)]
        _ => Err(Error::InvalidOperation(
            "CPU mapping of strided tensors is supported only for HAL-allocated \
             Mem/Shm (any platform), DMA (Linux, self-allocated or imported), \
             IOSurface (macOS), AHardwareBuffer (Android), D3D11 textures \
             (Windows), and PBO"
                .into(),
        )),
    }
}

#[cfg(not(edgefirst_tensor_dynamic))]
impl<T> Tensor<T>
where
    T: Num + Clone + fmt::Debug + Send + Sync,
{
    /// The shared body of [`TensorTrait::map_with`] and
    /// [`TensorTrait::try_map_with`]: the direction check, the strided routing
    /// and the backing's own map. `non_blocking` selects the backing's
    /// non-blocking map.
    fn map_impl<'a>(
        &self,
        access: CpuAccess,
        non_blocking: bool,
    ) -> Result<crate::view::HostView<'a, T>>
    where
        T: 'a,
    {
        let _span = tracing::trace_span!(
            "tensor.map",
            memory = ?self.storage.memory(),
            ?access,
            non_blocking,
        )
        .entered();
        self.cpu_access.require_direction(access, "map_with")?;
        // A strided tensor maps its full padded buffer (`row_stride × rows`).
        if let Some(stride) = self.row_stride {
            return map_strided_host_view(self, stride, access, non_blocking);
        }
        // Every storage applies its own plane offset inside its map.
        if non_blocking {
            self.storage.try_map_with(access)
        } else {
            self.storage.map_with(access)
        }
    }
}

#[cfg(all(
    not(edgefirst_tensor_dynamic),
    any(
        target_os = "macos",
        target_os = "ios",
        target_os = "android",
        target_os = "windows"
    )
))]
impl<T> Tensor<T>
where
    T: Num + Clone + fmt::Debug + Send + Sync,
{
    /// The pitched-backing half of [`TensorTrait::set_logical_shape`]: record
    /// the backing's own pitch as the row stride of the new `shape`.
    fn adopt_backing_pitch(&mut self, shape: &[usize]) {
        // Applies only to a backing with a pitch of its own, while the shape has
        // the rank the format's row is measured on.
        let Some(pitch) = self.storage.backing_row_stride() else {
            return;
        };
        let Some(format) = self.format else {
            return;
        };
        let (rank, rows) = match format.layout() {
            PixelLayout::Packed => (3, shape.first().copied()),
            PixelLayout::Planar => (3, shape.get(1).copied()),
            PixelLayout::SemiPlanar => (2, shape.first().copied()),
        };
        if rank != shape.len() {
            return;
        }
        // The natural stride of the new shape.
        let prior = self.row_stride.take();
        let Some(natural) = self.effective_row_stride() else {
            self.row_stride = prior;
            return;
        };
        if rows == Some(1) {
            self.set_row_stride_unchecked(natural);
        } else if records_backing_pitch(format, pitch, natural) {
            self.set_row_stride_unchecked(pitch);
        } else {
            self.row_stride = prior;
        }
    }
}

#[cfg(not(edgefirst_tensor_dynamic))]
impl<T> TensorTrait<T> for Tensor<T>
where
    T: Num + Clone + fmt::Debug + Send + Sync,
{
    fn cpu_access(&self) -> CpuAccess {
        self.cpu_access
    }

    fn view_origin(&self) -> Option<ViewOrigin> {
        self.view_origin
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
        self.storage.clone_fd()
    }

    fn memory(&self) -> TensorMemory {
        self.storage.memory()
    }

    fn name(&self) -> String {
        self.storage.name()
    }

    fn shape(&self) -> &[usize] {
        self.storage.shape()
    }

    /// Bytes available in the underlying allocation, from this tensor's
    /// window start. May exceed [`TensorTrait::size`] when the backing carries
    /// spare room beyond the current logical shape.
    fn capacity_bytes(&self) -> usize {
        self.storage.capacity_bytes()
    }

    fn reshape(&mut self, shape: &[usize]) -> Result<()> {
        if self.chroma.is_some() {
            return Err(Error::InvalidOperation(
                "cannot reshape a multiplane tensor — decompose planes first".into(),
            ));
        }
        self.storage.reshape(shape)?;
        self.format = None;
        self.row_stride = None;
        self.plane_offset = None;
        match self.storage {
            TensorStorage::Mem(ref mut m) => m.set_offset(0),
            #[cfg(target_os = "linux")]
            TensorStorage::Dma(ref mut dma) => dma.mmap_offset = 0,
            // `IoSurfaceTensor::reshape` zeroes its own `view_offset`.
            #[cfg(any(target_os = "macos", target_os = "ios"))]
            TensorStorage::Dma(_) => {}
            #[cfg(target_os = "windows")]
            TensorStorage::Dma(ref mut tex) => tex.set_view_offset(0),
            // `AHardwareBufferTensor::reshape` zeroes its own `view_offset`.
            #[cfg(target_os = "android")]
            TensorStorage::Dma(_) => {}
            // `PboTensor::reshape` zeroes its own `view_offset`.
            TensorStorage::Pbo(_) => {}
            #[cfg(unix)]
            TensorStorage::Shm(ref mut s) => s.set_offset(0),
        }
        Ok(())
    }

    /// Set the logical shape to any shape whose bytes fit the allocation,
    /// without [`reshape`](TensorTrait::reshape)'s equal-count constraint.
    ///
    /// The pixel format is kept. On a pitched backing (an IOSurface, an
    /// AHardwareBuffer, a D3D11 texture) the backing's pitch is recorded as
    /// the row stride when it exceeds the new shape's natural stride; a
    /// single-row shape records its own tight stride.
    fn set_logical_shape(&mut self, shape: &[usize]) -> Result<()> {
        self.storage.set_logical_shape(shape)?;
        #[cfg(any(
            target_os = "macos",
            target_os = "ios",
            target_os = "android",
            target_os = "windows"
        ))]
        self.adopt_backing_pitch(shape);
        Ok(())
    }

    fn map_with<'a>(&self, access: CpuAccess) -> Result<crate::view::HostView<'a, T>>
    where
        T: 'a,
    {
        self.map_impl(access, false)
    }

    fn try_map_with<'a>(&self, access: CpuAccess) -> Result<crate::view::HostView<'a, T>>
    where
        T: 'a,
    {
        self.map_impl(access, true)
    }

    fn buffer_identity(&self) -> &BufferIdentity {
        self.storage.buffer_identity()
    }
}

// ============================================================================
// Platform availability helpers
// ============================================================================

/// Cached result of the Linux DMA-BUF availability probe.
#[cfg(target_os = "linux")]
static DMA_AVAILABLE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
/// Cached result of the macOS/iOS IOSurface availability probe.
#[cfg(any(target_os = "macos", target_os = "ios"))]
static IOSURFACE_AVAILABLE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();

/// Check if Linux DMA-BUF allocation is available on this system.
///
/// Typically requires `/dev/dma_heap` access (root, or membership in a
/// video/render group). [`is_gpu_buffer_available`] is the portable probe
/// across platforms.
///
/// The result is cached after the first call.
#[cfg(target_os = "linux")]
pub fn is_dma_available() -> bool {
    *DMA_AVAILABLE
        .get_or_init(|| Tensor::<u8>::new(&[64], Some(TensorMemory::DmaBuf), None).is_ok())
}

/// Always returns `false` on non-Linux platforms.
#[cfg(not(target_os = "linux"))]
pub fn is_dma_available() -> bool {
    false
}

/// Check if macOS/iOS IOSurface allocation is available on this system.
///
/// The result is cached after the first call. Always returns `false` on
/// non-Apple platforms.
#[cfg(any(target_os = "macos", target_os = "ios"))]
pub fn is_iosurface_available() -> bool {
    *IOSURFACE_AVAILABLE.get_or_init(|| {
        // Probes by allocating a small Dma tensor.
        Tensor::<u8>::new(&[64], Some(TensorMemory::DmaBuf), None).is_ok()
    })
}

#[cfg(not(any(target_os = "macos", target_os = "ios")))]
pub fn is_iosurface_available() -> bool {
    false
}

/// Cached result of the Android AHardwareBuffer availability probe.
#[cfg(target_os = "android")]
static AHARDWAREBUFFER_AVAILABLE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();

/// Check if Android AHardwareBuffer allocation is available on this system.
///
/// The result is cached after the first call.
#[cfg(target_os = "android")]
pub fn is_ahardwarebuffer_available() -> bool {
    *AHARDWAREBUFFER_AVAILABLE.get_or_init(|| {
        // Probes by allocating a small Dma tensor.
        Tensor::<u8>::new(&[64], Some(TensorMemory::DmaBuf), None).is_ok()
    })
}

/// Always returns `false` on non-Android platforms.
#[cfg(not(target_os = "android"))]
pub fn is_ahardwarebuffer_available() -> bool {
    false
}

/// Portable probe for the platform's native zero-copy GPU buffer allocator
/// (DMA-BUF on Linux, IOSurface on macOS/iOS, AHardwareBuffer on Android,
/// D3D11 textures on Windows). Returns `false` on platforms with no
/// equivalent.
///
/// On Windows the `Dma` variant is image-formatted: `true` here means
/// [`Tensor::image`] can serve `TensorMemory::DmaBuf`, not [`Tensor::new`].
///
/// # Ordering on Windows
///
/// This probe creates the process D3D11 device. A host that supplies its own
/// device calls `d3d11::use_external_device` before this probe or any
/// allocation; installing one afterwards is refused.
pub fn is_gpu_buffer_available() -> bool {
    #[cfg(target_os = "linux")]
    {
        is_dma_available()
    }
    #[cfg(any(target_os = "macos", target_os = "ios"))]
    {
        is_iosurface_available()
    }
    #[cfg(target_os = "android")]
    {
        is_ahardwarebuffer_available()
    }
    #[cfg(target_os = "windows")]
    {
        crate::d3d11::device().is_ok()
    }
    #[cfg(not(any(
        target_os = "linux",
        target_os = "macos",
        target_os = "ios",
        target_os = "android",
        target_os = "windows"
    )))]
    {
        false
    }
}

/// Cached result of the POSIX shared memory availability probe.
#[cfg(unix)]
static SHM_AVAILABLE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();

/// Check if POSIX shared memory allocation is available on this system.
///
/// The result is cached after the first call.
#[cfg(unix)]
pub fn is_shm_available() -> bool {
    *SHM_AVAILABLE.get_or_init(|| Tensor::<u8>::new(&[64], Some(TensorMemory::Shm), None).is_ok())
}

/// Check if POSIX shared memory allocation is available on this system.
///
/// Always returns `false` on non-Unix platforms.
#[cfg(not(unix))]
pub fn is_shm_available() -> bool {
    false
}

#[cfg(test)]
mod dtype_tests {
    use super::*;

    #[test]
    fn dtype_size() {
        assert_eq!(DType::U8.size(), 1);
        assert_eq!(DType::I8.size(), 1);
        assert_eq!(DType::U16.size(), 2);
        assert_eq!(DType::I16.size(), 2);
        assert_eq!(DType::U32.size(), 4);
        assert_eq!(DType::I32.size(), 4);
        assert_eq!(DType::U64.size(), 8);
        assert_eq!(DType::I64.size(), 8);
        assert_eq!(DType::F16.size(), 2);
        assert_eq!(DType::F32.size(), 4);
        assert_eq!(DType::F64.size(), 8);
    }

    #[test]
    fn dtype_name() {
        assert_eq!(DType::U8.name(), "u8");
        assert_eq!(DType::F16.name(), "f16");
        assert_eq!(DType::F32.name(), "f32");
    }

    #[test]
    fn dtype_serde_roundtrip() {
        use serde_json;
        let dt = DType::F16;
        let json = serde_json::to_string(&dt).unwrap();
        let back: DType = serde_json::from_str(&json).unwrap();
        assert_eq!(dt, back);
    }
}

#[cfg(test)]
mod image_tests {
    use super::*;

    #[test]
    fn image_dims_from_shape_inverts_allocation_shape() {
        for (format, w, h) in [
            (PixelFormat::Rgba, 640usize, 480usize),
            (PixelFormat::Rgb, 641, 3),
            (PixelFormat::Grey, 32, 32),
            (PixelFormat::PlanarRgb, 640, 480),
            (PixelFormat::PlanarRgba, 8, 5),
            (PixelFormat::Nv12, 640, 481),
            (PixelFormat::Nv16, 33, 17),
            (PixelFormat::Nv24, 16, 9),
        ] {
            let shape = format.allocation_shape(w, h).unwrap();
            assert_eq!(
                image_dims_from_shape(format, &shape).unwrap(),
                (w, h),
                "{format:?} {w}x{h} via {shape:?}"
            );
        }
    }

    #[test]
    fn image_dims_from_shape_rejects_the_wrong_rank() {
        // Packed and planar want rank 3, semi-planar rank 2.
        for (format, shape) in [
            (PixelFormat::Rgba, vec![480usize, 640usize]),
            (PixelFormat::PlanarRgb, vec![3, 480]),
            (PixelFormat::Nv12, vec![720, 640, 1]),
        ] {
            let err = image_dims_from_shape(format, &shape).unwrap_err();
            assert!(
                matches!(err, Error::InvalidShape(ref m) if m.contains("rank")),
                "{format:?} {shape:?}: {err}"
            );
        }
    }

    #[test]
    fn image_dims_from_shape_rejects_an_unproduced_combined_height() {
        // NV12 maps 480 -> 720 and 481 -> 722; nothing maps to 721.
        let err = image_dims_from_shape(PixelFormat::Nv12, &[721, 640]).unwrap_err();
        assert!(
            matches!(err, Error::InvalidShape(ref m) if m.contains("combined plane height")),
            "{err}"
        );
    }

    #[test]
    fn allocation_shape_per_layout() {
        assert_eq!(
            PixelFormat::Rgb.allocation_shape(640, 480),
            Some(vec![480, 640, 3])
        );
        assert_eq!(
            PixelFormat::Grey.allocation_shape(640, 480),
            Some(vec![480, 640, 1])
        );
        assert_eq!(
            PixelFormat::Nv12.allocation_shape(640, 480),
            Some(vec![720, 640])
        );
        // Odd height: combined-plane height is `481 + ceil(481/2)` = 481 + 241
        // = 722 rows. Logical height is recovered as `722 * 2 / 3` = 481.
        assert_eq!(
            PixelFormat::Nv12.allocation_shape(640, 481),
            Some(vec![722, 640])
        );
        // Odd width: the shape carries the logical width (641).
        assert_eq!(
            PixelFormat::Nv12.allocation_shape(641, 480),
            Some(vec![720, 641])
        );
        // NV16 odd width: the shape carries the logical width.
        assert_eq!(
            PixelFormat::Nv16.allocation_shape(641, 480),
            Some(vec![960, 641])
        );
        assert_eq!(
            PixelFormat::PlanarRgb.allocation_shape(640, 480),
            Some(vec![3, 480, 640])
        );
        assert_eq!(
            PixelFormat::Nv16.allocation_shape(640, 480),
            Some(vec![960, 640])
        );
    }

    #[test]
    fn raw_tensor_has_no_format() {
        let t = Tensor::<u8>::new(&[480, 640, 3], None, None).unwrap();
        assert!(t.format().is_none());
        assert!(t.width().is_none());
        assert!(t.height().is_none());
        assert!(!t.is_multiplane());
        assert!(t.chroma().is_none());
    }

    #[test]
    fn image_tensor_packed() {
        let t = Tensor::<u8>::image(
            640,
            480,
            PixelFormat::Rgba,
            None,
            crate::CpuAccess::ReadWrite,
        )
        .unwrap();
        assert_eq!(t.format(), Some(PixelFormat::Rgba));
        assert_eq!(t.width(), Some(640));
        assert_eq!(t.height(), Some(480));
        assert_eq!(t.shape(), &[480, 640, 4]);
        assert!(!t.is_multiplane());
    }

    #[test]
    fn image_tensor_planar() {
        let t = Tensor::<u8>::image(
            640,
            480,
            PixelFormat::PlanarRgb,
            None,
            crate::CpuAccess::ReadWrite,
        )
        .unwrap();
        assert_eq!(t.format(), Some(PixelFormat::PlanarRgb));
        assert_eq!(t.width(), Some(640));
        assert_eq!(t.height(), Some(480));
        assert_eq!(t.shape(), &[3, 480, 640]);
    }

    #[test]
    #[cfg(target_os = "macos")]
    fn image_tensor_dma_non_aligned_packed_width_pads_zero_copy() {
        let _lock = crate::tests::fd_lock_shared();
        // RGBA u8 at width=4: 4*4 = 16 bytes/row, not 64-byte aligned.
        let t = Tensor::<u8>::image(
            4,
            4,
            PixelFormat::Rgba,
            Some(TensorMemory::DmaBuf),
            crate::CpuAccess::ReadWrite,
        )
        .expect("padded RGBA IOSurface should allocate");
        assert_eq!(t.format(), Some(PixelFormat::Rgba));
        assert_eq!(t.width(), Some(4));
        assert_eq!(t.height(), Some(4));
        let stride = t.effective_row_stride().expect("stride");
        assert_eq!(stride % 64, 0, "padded to 64-byte row alignment");
        assert!(stride >= 16);
        // A CPU map exposes the full padded surface.
        let m = t.map().expect("strided IOSurface map");
        assert_eq!(m.as_slice().len(), stride * 4);
    }

    /// An explicit-DMA request whose pixel pitch cannot reach 64-byte
    /// alignment errors without a "pad width to N" hint.
    #[test]
    #[cfg(target_os = "macos")]
    fn image_tensor_dma_rejects_indivisible_pixel_pitch_without_pad_hint() {
        let _lock = crate::tests::fd_lock_shared();
        // Width=10 RGB f32: 120 B/row, not 64-byte aligned; the next 64-multiple
        // (128 B) is not a multiple of 12 B/pixel.
        let err = Tensor::<f32>::image(
            10,
            10,
            PixelFormat::Rgb,
            Some(TensorMemory::DmaBuf),
            crate::CpuAccess::ReadWrite,
        )
        .expect_err("RGB f32 with 12 B/pixel and non-aligned width must be rejected");
        match err {
            Error::InvalidArgument(msg) => {
                assert!(
                    msg.contains("64-byte aligned"),
                    "error must still name the alignment requirement: {msg}"
                );
                assert!(
                    !msg.contains("Pad width"),
                    "indivisible per-pixel pitch makes a width suggestion impossible; \
                     hint must be omitted, got: {msg}"
                );
                assert!(
                    msg.contains("memory=None") && msg.contains("TensorMemory::Mem"),
                    "error must still list the always-applicable alternatives: {msg}"
                );
            }
            other => panic!("expected InvalidArgument, got {other:?}"),
        }
    }

    #[test]
    #[cfg(target_os = "macos")]
    fn image_tensor_dma_packed_rgb_u8_contract() {
        let _lock = crate::tests::fd_lock_shared();
        // Packed RGB u8 @Dma maps to RGBA8888 at (W*3/4, H); width%4 != 0 is
        // InvalidArgument.
        let err = Tensor::<u8>::image(
            10,
            10,
            PixelFormat::Rgb,
            Some(TensorMemory::DmaBuf),
            crate::CpuAccess::ReadWrite,
        )
        .expect_err("Rgb u8 width%4!=0 must be rejected");
        assert!(
            matches!(&err, Error::InvalidArgument(m) if m.contains("width%4==0")),
            "got {err:?}"
        );
        // width%4 == 0 with a non-64-aligned pitch allocates padded: 36 B rows at
        // a 64 B surface pitch.
        let t = Tensor::<u8>::image(
            12,
            4,
            PixelFormat::Rgb,
            Some(TensorMemory::DmaBuf),
            crate::CpuAccess::ReadWrite,
        )
        .expect("width 12 Rgb u8 must allocate padded");
        assert_eq!(t.memory(), TensorMemory::DmaBuf);
        assert!(
            t.row_stride()
                .is_some_and(|s| s >= 64 && s.is_multiple_of(64)),
            "padded pitch must be recorded: {:?}",
            t.row_stride()
        );
        // The aligned width stays flat: 640*3 = 1920 is 64-aligned, no recorded
        // stride.
        let t = Tensor::<u8>::image(
            640,
            8,
            PixelFormat::Rgb,
            Some(TensorMemory::DmaBuf),
            crate::CpuAccess::ReadWrite,
        )
        .expect("width 640 Rgb u8 must allocate flat");
        assert_eq!(t.row_stride(), None);
        // I8 shares the layout.
        let t = Tensor::<i8>::image(
            640,
            8,
            PixelFormat::Rgb,
            Some(TensorMemory::DmaBuf),
            crate::CpuAccess::ReadWrite,
        )
        .expect("Rgb i8 shares the RGBA8888 mapping");
        assert_eq!(t.memory(), TensorMemory::DmaBuf);
    }

    #[test]
    #[cfg(target_os = "macos")]
    fn image_tensor_dma_planar_f16_alignment() {
        let _lock = crate::tests::fd_lock_shared();
        // PlanarRgb F16 uses single-channel row pitch (width * 2 bytes).
        // Width=16 → 32 bytes/row (not aligned); width=32 → 64 bytes/row (aligned).
        let err = Tensor::<half::f16>::image(
            16,
            16,
            PixelFormat::PlanarRgb,
            Some(TensorMemory::DmaBuf),
            crate::CpuAccess::ReadWrite,
        )
        .expect_err("width=16 PlanarRgb F16 is 32-byte row, must reject");
        assert!(matches!(err, Error::InvalidArgument(_)), "got {err:?}");
        // 32 wide is accepted.
        let t = Tensor::<half::f16>::image(
            32,
            8,
            PixelFormat::PlanarRgb,
            Some(TensorMemory::DmaBuf),
            crate::CpuAccess::ReadWrite,
        )
        .expect("width=32 PlanarRgb F16 is 64-byte row, must succeed");
        assert_eq!(t.format(), Some(PixelFormat::PlanarRgb));
    }

    #[test]
    fn image_tensor_semi_planar_contiguous() {
        let t = Tensor::<u8>::image(
            640,
            480,
            PixelFormat::Nv12,
            None,
            crate::CpuAccess::ReadWrite,
        )
        .unwrap();
        assert_eq!(t.format(), Some(PixelFormat::Nv12));
        assert_eq!(t.width(), Some(640));
        assert_eq!(t.height(), Some(480));
        // NV12: H*3/2 = 720
        assert_eq!(t.shape(), &[720, 640]);
        assert!(!t.is_multiplane());
    }

    #[test]
    #[cfg(target_os = "linux")]
    fn image_tensor_with_stride_preserves_logical_width() {
        let _lock = crate::tests::fd_lock_shared();
        if !crate::test_support::dma_or_skip("image_tensor_with_stride_preserves_logical_width") {
            return;
        }
        // 3004×1688 RGBA8: natural pitch 12016, padded to 12032 (64-aligned).
        let stride = 12032;
        let t = Tensor::<u8>::image_with_stride(
            3004,
            1688,
            PixelFormat::Rgba,
            stride,
            Some(TensorMemory::DmaBuf),
            crate::CpuAccess::ReadWrite,
        )
        .unwrap();
        // Logical dimensions are unchanged by padding.
        assert_eq!(t.width(), Some(3004));
        assert_eq!(t.height(), Some(1688));
        assert_eq!(t.shape(), &[1688, 3004, 4]);
        // The stride reports the padded pitch.
        assert_eq!(t.effective_row_stride(), Some(stride));
        // The buffer is sized to stride × height.
        use crate::TensorMapTrait;
        {
            let map = t.map().unwrap();
            assert!(
                map.as_slice().len() >= stride * 1688,
                "mapped buffer {} bytes < expected {}",
                map.as_slice().len(),
                stride * 1688
            );
        }
        // Write rows at the padded stride, touching only the active
        // `width × bpp` region.
        {
            let mut map = t.map().unwrap();
            let slice = map.as_mut_slice();
            for y in 0..1688 {
                let row_start = y * stride;
                for x in 0..3004 {
                    let p = row_start + x * 4;
                    slice[p] = (y & 0xFF) as u8;
                    slice[p + 1] = (x & 0xFF) as u8;
                    slice[p + 2] = 0x42;
                    slice[p + 3] = 0xFF;
                }
            }
        }
        {
            let map = t.map().unwrap();
            let slice = map.as_slice();
            assert_eq!(slice[0], 0x00);
            assert_eq!(slice[1], 0x00);
            assert_eq!(slice[2], 0x42);
            assert_eq!(slice[3], 0xFF);
            let mid = 100 * stride + 50 * 4;
            assert_eq!(slice[mid], 100);
            assert_eq!(slice[mid + 1], 50);
            assert_eq!(slice[mid + 2], 0x42);
        }
    }

    #[test]
    #[cfg(target_os = "linux")]
    fn image_tensor_foreign_strided_map_is_bounds_checked_not_refused() {
        // A CPU map of an imported DMA-BUF with a row stride is bounds-checked
        // against `buf_size`.
        if !crate::test_support::dma_or_skip(
            "image_tensor_foreign_strided_map_is_bounds_checked_not_refused",
        ) {
            return;
        }
        let backing =
            Tensor::<u8>::new(&[240 * 320 * 4], Some(TensorMemory::DmaBuf), None).unwrap();

        // In bounds: the natural stride for the declared image fits the
        // allocation.
        {
            let fd = backing.clone_fd().unwrap();
            let shape = [240usize, 320, 4];
            let storage = TensorStorage::<u8>::from_fd(fd, &shape, None).unwrap();
            let mut t = Tensor::<u8>::wrap(storage);
            t.set_format(PixelFormat::Bgra).unwrap();
            t.set_row_stride(320 * 4).unwrap();
            assert!(
                t.map().is_ok(),
                "an in-bounds strided map of an imported DMA-BUF must be \
                 accepted -- refusing it is what broke decode-into-GPU-tensor"
            );
        }

        // Out of bounds: a stride the allocation cannot back is refused.
        {
            let fd = backing.clone_fd().unwrap();
            let shape = [240usize, 320, 4];
            let storage = TensorStorage::<u8>::from_fd(fd, &shape, None).unwrap();
            let mut t = Tensor::<u8>::wrap(storage);
            t.set_format(PixelFormat::Bgra).unwrap();
            // Double the pitch: 240 rows x 2560 bytes is twice the allocation.
            t.set_row_stride(320 * 4 * 2).unwrap();
            let err = t.map();
            assert!(
                matches!(err, Err(Error::InvalidOperation(_))),
                "a strided map larger than the DMA allocation must be refused"
            );
        }
    }

    #[test]
    #[cfg(target_os = "linux")]
    fn image_tensor_with_stride_map_rejects_tampered_stride() {
        let _lock = crate::tests::fd_lock_shared();
        if !crate::test_support::dma_or_skip("image_tensor_with_stride_map_rejects_tampered_stride")
        {
            return;
        }
        // A 640×480 RGBA8 padded canvas (stride = 3072 = 768 px); the backing
        // buffer is 3072 × 480 = 1,474,560 bytes.
        let mut t = Tensor::<u8>::image_with_stride(
            640,
            480,
            PixelFormat::Rgba,
            3072,
            Some(TensorMemory::DmaBuf),
            crate::CpuAccess::ReadWrite,
        )
        .unwrap();
        // Raise the stride to 4 × the original; it is >= min_stride (2560).
        t.set_row_stride(12288).unwrap();
        // 12288 × 480 = 5,898,240 > 1,474,560.
        let err = t.map();
        assert!(
            matches!(err, Err(Error::InvalidOperation(_))),
            "map() with oversized stride must return InvalidOperation"
        );
    }

    #[test]
    fn dma_tensor_new_with_byte_size_rejects_shape_overflow() {
        #[cfg(target_os = "linux")]
        {
            let err = crate::dma::DmaTensor::<u64>::new_with_byte_size(
                &[usize::MAX, 2, 2],
                usize::MAX,
                None,
                crate::dma::HeapPolicy::default(),
            );
            assert!(
                matches!(err, Err(Error::InvalidArgument(_))),
                "new_with_byte_size must detect shape.product() overflow"
            );
        }
    }

    #[test]
    #[cfg(target_os = "linux")]
    fn image_tensor_with_stride_rejects_too_small_stride() {
        let _lock = crate::tests::fd_lock_shared();
        // 640×480 RGBA8 natural pitch = 2560; a 2400 request errors.
        let err = Tensor::<u8>::image_with_stride(
            640,
            480,
            PixelFormat::Rgba,
            2400,
            Some(TensorMemory::DmaBuf),
            crate::CpuAccess::ReadWrite,
        );
        assert!(matches!(err, Err(Error::InvalidArgument(_))));
    }

    #[test]
    #[cfg(target_os = "linux")]
    fn image_tensor_with_stride_rejects_non_packed() {
        let _lock = crate::tests::fd_lock_shared();
        // NV12 is SemiPlanar.
        let err = Tensor::<u8>::image_with_stride(
            640,
            480,
            PixelFormat::Nv12,
            640,
            Some(TensorMemory::DmaBuf),
            crate::CpuAccess::ReadWrite,
        );
        assert!(matches!(err, Err(Error::NotImplemented(_))));
    }

    #[test]
    fn set_format_valid() {
        let mut t = Tensor::<u8>::new(&[480, 640, 3], None, None).unwrap();
        assert!(t.format().is_none());
        t.set_format(PixelFormat::Rgb).unwrap();
        assert_eq!(t.format(), Some(PixelFormat::Rgb));
        assert_eq!(t.width(), Some(640));
        assert_eq!(t.height(), Some(480));
    }

    #[test]
    fn set_format_invalid_shape() {
        let mut t = Tensor::<u8>::new(&[480, 640, 4], None, None).unwrap();
        // RGB expects 3 channels, not 4
        let err = t.set_format(PixelFormat::Rgb);
        assert!(err.is_err());
        assert!(t.format().is_none());
    }

    #[test]
    fn reshape_clears_format() {
        let mut t = Tensor::<u8>::image(
            640,
            480,
            PixelFormat::Rgba,
            None,
            crate::CpuAccess::ReadWrite,
        )
        .unwrap();
        assert_eq!(t.format(), Some(PixelFormat::Rgba));
        t.reshape(&[480 * 640 * 4]).unwrap();
        assert!(t.format().is_none());
    }

    #[test]
    fn from_planes_nv12() {
        let y = Tensor::<u8>::new(&[480, 640], None, None).unwrap();
        let uv = Tensor::<u8>::new(&[240, 640], None, None).unwrap();
        let img = Tensor::from_planes(y, uv, PixelFormat::Nv12).unwrap();
        assert_eq!(img.format(), Some(PixelFormat::Nv12));
        assert!(img.is_multiplane());
        assert!(img.chroma().is_some());
        assert_eq!(img.width(), Some(640));
        assert_eq!(img.height(), Some(480));
    }

    #[test]
    fn from_planes_rejects_non_semiplanar() {
        let y = Tensor::<u8>::new(&[480, 640], None, None).unwrap();
        let uv = Tensor::<u8>::new(&[240, 640], None, None).unwrap();
        let err = Tensor::from_planes(y, uv, PixelFormat::Rgb);
        assert!(err.is_err());
    }

    #[test]
    fn reshape_multiplane_errors() {
        let y = Tensor::<u8>::new(&[480, 640], None, None).unwrap();
        let uv = Tensor::<u8>::new(&[240, 640], None, None).unwrap();
        let mut img = Tensor::from_planes(y, uv, PixelFormat::Nv12).unwrap();
        let err = img.reshape(&[480 * 640 + 240 * 640]);
        assert!(err.is_err());
    }
}

#[cfg(test)]
mod compression_tests {
    use super::*;

    #[test]
    fn desc_builder_roundtrips() {
        let desc = ImageDesc::new(640, 480, PixelFormat::Rgba, DType::U8)
            .with_memory(Some(TensorMemory::Mem))
            .with_access(CpuAccess::Read)
            .with_compression(Compression::Any);
        assert_eq!(desc.width(), 640);
        assert_eq!(desc.height(), 480);
        assert_eq!(desc.format(), PixelFormat::Rgba);
        assert_eq!(desc.dtype(), DType::U8);
        assert_eq!(desc.memory(), Some(TensorMemory::Mem));
        assert_eq!(desc.access(), CpuAccess::Read);
        assert_eq!(desc.compression(), Some(Compression::Any));

        // Defaults: auto memory, CPU read/write, no request.
        let plain = ImageDesc::new(2, 2, PixelFormat::Grey, DType::U8);
        assert_eq!(plain.memory(), None);
        assert_eq!(plain.access(), CpuAccess::ReadWrite);
        assert_eq!(plain.compression(), None);
    }

    #[test]
    fn desc_dtype_must_match_element_type() {
        let desc = ImageDesc::new(4, 4, PixelFormat::Rgba, DType::F32);
        match Tensor::<u8>::image_desc(&desc) {
            Err(Error::InvalidArgument(msg)) => assert!(msg.contains("dtype")),
            other => panic!("expected InvalidArgument, got {other:?}"),
        }
    }

    #[test]
    fn compression_with_cpu_access_is_invalid() {
        let desc = ImageDesc::new(4, 4, PixelFormat::Rgba, DType::U8)
            .with_access(CpuAccess::ReadWrite)
            .with_compression(Compression::Any);
        match Tensor::<u8>::image_desc(&desc) {
            Err(Error::InvalidArgument(msg)) => assert!(msg.contains("CpuAccess::None")),
            other => panic!("expected InvalidArgument, got {other:?}"),
        }
    }

    #[cfg(not(target_os = "android"))]
    #[test]
    fn scheme_request_off_android_is_not_implemented() {
        let desc = ImageDesc::new(4, 4, PixelFormat::Rgba, DType::U8)
            .with_access(CpuAccess::None)
            .with_compression(Compression::Scheme(CompressionScheme::Ubwc));
        match Tensor::<u8>::image_desc(&desc) {
            Err(Error::NotImplemented(msg)) => assert!(msg.contains("Ubwc")),
            other => panic!("expected NotImplemented, got {other:?}"),
        }
    }

    #[cfg(not(target_os = "android"))]
    #[test]
    fn any_request_off_android_resolves_linear_and_counts() {
        let before = compression_fallback_count();
        let desc = ImageDesc::new(64, 64, PixelFormat::Rgba, DType::U8)
            .with_memory(Some(TensorMemory::Mem))
            .with_access(CpuAccess::None)
            .with_compression(Compression::Any);
        let t = Tensor::<u8>::image_desc(&desc).unwrap();
        assert_eq!(t.compression(), None);
        assert!(compression_fallback_count() > before);
    }

    #[test]
    fn desc_without_request_matches_classic_constructor() {
        let desc = ImageDesc::new(32, 32, PixelFormat::Rgba, DType::U8)
            .with_memory(Some(TensorMemory::Mem))
            .with_access(CpuAccess::ReadWrite);
        let t = Tensor::<u8>::image_desc(&desc).unwrap();
        assert_eq!(t.compression(), None);
        assert_eq!(t.cpu_access(), CpuAccess::ReadWrite);
        assert_eq!(t.width(), Some(32));
        let m = t.map_read().unwrap();
        assert_eq!(m.as_slice().len(), 32 * 32 * 4);
    }

    #[test]
    fn configure_image_preserves_compression_and_views_inherit() {
        // Emulates a recorded compression scheme.
        let mut t = Tensor::<u8>::image(
            64,
            64,
            PixelFormat::Rgba,
            Some(TensorMemory::Mem),
            CpuAccess::None,
        )
        .unwrap();
        t.compression = Some(CompressionScheme::Ubwc);
        t.configure_image(32, 32, PixelFormat::Rgba).unwrap();
        assert_eq!(t.compression(), Some(CompressionScheme::Ubwc));
        let view = t.subview(0, &[16, 32, 4]).unwrap();
        assert_eq!(view.compression(), Some(CompressionScheme::Ubwc));
    }

    #[test]
    fn tensor_dyn_dispatches_desc_and_compression() {
        let desc = ImageDesc::new(16, 16, PixelFormat::Rgba, DType::U8)
            .with_memory(Some(TensorMemory::Mem))
            .with_access(CpuAccess::ReadWrite);
        let t = TensorDyn::image_desc(&desc).unwrap();
        assert_eq!(t.compression(), None);
        assert!(matches!(t, TensorDyn::U8(_)));
    }
}

#[cfg(all(test, not(edgefirst_tensor_dynamic)))]
mod vocabulary_and_probe_tests {
    use super::*;

    #[test]
    fn dtype_displays_its_wire_name() {
        for (dt, name) in [
            (DType::U8, "u8"),
            (DType::I8, "i8"),
            (DType::U16, "u16"),
            (DType::I16, "i16"),
            (DType::U32, "u32"),
            (DType::I32, "i32"),
            (DType::U64, "u64"),
            (DType::I64, "i64"),
            (DType::F16, "f16"),
            (DType::F32, "f32"),
            (DType::F64, "f64"),
        ] {
            assert_eq!(dt.to_string(), name);
        }
    }

    #[test]
    fn dtype_of_maps_every_element_type_and_nothing_else() {
        assert_eq!(dtype_of::<u8>(), Some(DType::U8));
        assert_eq!(dtype_of::<i8>(), Some(DType::I8));
        assert_eq!(dtype_of::<u16>(), Some(DType::U16));
        assert_eq!(dtype_of::<i16>(), Some(DType::I16));
        assert_eq!(dtype_of::<u32>(), Some(DType::U32));
        assert_eq!(dtype_of::<i32>(), Some(DType::I32));
        assert_eq!(dtype_of::<u64>(), Some(DType::U64));
        assert_eq!(dtype_of::<i64>(), Some(DType::I64));
        assert_eq!(dtype_of::<half::f16>(), Some(DType::F16));
        assert_eq!(dtype_of::<f32>(), Some(DType::F32));
        assert_eq!(dtype_of::<f64>(), Some(DType::F64));
        assert_eq!(dtype_of::<bool>(), None);
    }

    #[test]
    fn tensor_memory_converts_to_and_from_its_wire_string() {
        assert_eq!(String::from(TensorMemory::DmaBuf), "dmabuf");
        assert_eq!(String::from(TensorMemory::Shm), "shm");
        assert_eq!(
            TensorMemory::try_from("dmabuf").unwrap(),
            TensorMemory::DmaBuf
        );
        assert_eq!(
            TensorMemory::try_from("dma").unwrap(),
            TensorMemory::DmaBuf,
            "the legacy spelling still parses"
        );
        assert_eq!(TensorMemory::try_from("mem").unwrap(), TensorMemory::Mem);
        match TensorMemory::try_from("bogus") {
            Err(Error::InvalidMemoryType(s)) => assert_eq!(s, "bogus"),
            other => panic!("expected InvalidMemoryType, got {other:?}"),
        }
    }

    #[test]
    fn tensor_memory_is_available_matches_a_real_allocation() {
        let _lock = crate::tests::fd_lock_shared();
        assert!(TensorMemory::Mem.is_available());
        assert!(!TensorMemory::IoSurface.is_available());
        assert!(!TensorMemory::Pbo.is_available());
        assert!(!TensorMemory::Cuda.is_available());
        assert!(!TensorMemory::D3d11Texture.is_available());
        #[cfg(unix)]
        assert_eq!(
            TensorMemory::Shm.is_available(),
            Tensor::<u8>::new(&[64], Some(TensorMemory::Shm), None).is_ok()
        );
        #[cfg(not(target_os = "windows"))]
        assert_eq!(
            TensorMemory::DmaBuf.is_available(),
            Tensor::<u8>::new(&[64], Some(TensorMemory::DmaBuf), None).is_ok()
        );
    }

    #[test]
    #[cfg(target_os = "linux")]
    fn linux_probes_report_what_linux_can_allocate() {
        let _lock = crate::tests::fd_lock_shared();
        let dma = DmaTensor::<u8>::new(&[64], None).is_ok();
        assert_eq!(is_dma_available(), dma);
        assert_eq!(is_gpu_buffer_available(), dma);
        assert!(!is_iosurface_available());
        assert!(!is_ahardwarebuffer_available());
        assert!(is_shm_available());
    }
}

#[cfg(all(test, not(edgefirst_tensor_dynamic)))]
mod quantization_serde_tests {
    use super::*;

    fn parse(json: &str) -> Quantization {
        serde_json::from_str(json).unwrap_or_else(|e| panic!("{json}: {e}"))
    }

    #[test]
    fn scale_accepts_every_json_number_form() {
        assert_eq!(parse(r#"{"scale": 0.5}"#).scale(), &[0.5]);
        assert_eq!(parse(r#"{"scale": 2}"#).scale(), &[2.0]);
        assert_eq!(parse(r#"{"scale": -2}"#).scale(), &[-2.0]);
        assert_eq!(parse(r#"{"scale": [0.25, 4]}"#).scale(), &[0.25, 4.0]);
    }

    #[test]
    fn zero_point_accepts_null_scalar_and_array() {
        assert_eq!(
            parse(r#"{"scale": 0.1, "zero_point": -5}"#).zero_point(),
            Some(&[-5][..])
        );
        assert_eq!(
            parse(r#"{"scale": 0.1, "zero_point": 3}"#).zero_point(),
            Some(&[3][..])
        );
        assert_eq!(
            parse(r#"{"scale": [0.1, 0.2], "zero_point": [1, -2], "axis": 0}"#).zero_point(),
            Some(&[1, -2][..])
        );
        assert_eq!(
            parse(r#"{"scale": 0.1, "zero_point": null}"#).zero_point(),
            None
        );
        assert_eq!(parse(r#"{"scale": 0.1}"#).zero_point(), None);
    }

    /// The zero-point visitor's scalar and sequence arms accept values that
    /// arrive without `visit_some`.
    #[test]
    fn zero_point_visitor_accepts_values_without_an_option_wrapper() {
        use serde::de::value::{
            Error as ValueError, I64Deserializer, SeqDeserializer, U64Deserializer,
        };
        assert_eq!(
            deserialize_opt_scalar_or_vec_i32(I64Deserializer::<ValueError>::new(-7)).unwrap(),
            Some(vec![-7])
        );
        assert_eq!(
            deserialize_opt_scalar_or_vec_i32(U64Deserializer::<ValueError>::new(9)).unwrap(),
            Some(vec![9])
        );
        let seq = SeqDeserializer::<_, ValueError>::new(vec![4i32, -4].into_iter());
        assert_eq!(
            deserialize_opt_scalar_or_vec_i32(seq).unwrap(),
            Some(vec![4, -4])
        );
    }

    #[test]
    fn per_channel_is_not_per_tensor() {
        let q = Quantization::per_channel(vec![0.1, 0.2], vec![0, 1], 0).unwrap();
        assert!(!q.is_per_tensor());
        assert!(Quantization::per_tensor(0.1, 0).is_per_tensor());
    }

    #[test]
    fn validate_accepts_matching_per_channel_and_names_the_failing_field() {
        let mut t = Tensor::<i8>::new(&[1, 1, 4], Some(TensorMemory::Mem), None).unwrap();
        let q = Quantization::per_channel(vec![0.1, 0.2, 0.3, 0.4], vec![0, 1, 2, 3], 2).unwrap();
        t.set_quantization(q)
            .expect("four channels on axis 2 of [1, 1, 4]");
        assert_eq!(
            t.quantization().and_then(Quantization::zero_point),
            Some(&[0, 1, 2, 3][..])
        );

        let q = Quantization::per_channel(vec![0.1, 0.2, 0.3, 0.4], vec![0, 1, 2, 3], 3).unwrap();
        match t.set_quantization(q) {
            Err(Error::QuantizationInvalid { field, got, .. }) => {
                assert_eq!(field, "axis");
                assert_eq!(got, "axis=3");
            }
            other => panic!("axis 3 of a rank-3 tensor must be refused, got {other:?}"),
        }

        let q = Quantization::per_channel_symmetric(vec![0.1, 0.2, 0.3, 0.4], 1).unwrap();
        match t.set_quantization(q) {
            Err(Error::QuantizationInvalid { field, got, .. }) => {
                assert_eq!(field, "scale.len");
                assert_eq!(got, "length 4");
            }
            other => panic!("four scales on a length-1 axis must be refused, got {other:?}"),
        }
    }
}

#[cfg(all(test, not(edgefirst_tensor_dynamic)))]
mod image_geometry_tests {
    use super::*;

    fn mem_image<T>(width: usize, height: usize, format: PixelFormat) -> Tensor<T>
    where
        T: Num + Clone + fmt::Debug + Send + Sync + 'static,
    {
        Tensor::<T>::image(
            width,
            height,
            format,
            Some(TensorMemory::Mem),
            CpuAccess::ReadWrite,
        )
        .unwrap_or_else(|e| panic!("Mem {format:?} {width}x{height}: {e}"))
    }

    fn raw_mem<T>(shape: &[usize]) -> Tensor<T>
    where
        T: Num + Clone + fmt::Debug + Send + Sync + 'static,
    {
        Tensor::<T>::new(shape, Some(TensorMemory::Mem), None).unwrap()
    }

    #[test]
    fn semi_planar_image_stride_scales_with_the_element_size() {
        // u16 NV12 at width 100: 200 B rows, aligned to 256.
        let t = mem_image::<u16>(100, 64, PixelFormat::Nv12);
        assert_eq!(t.row_stride(), Some(256));
        assert_eq!(t.capacity_bytes(), 256 * 96);
    }

    #[test]
    fn host_planar_image_is_tight_at_every_element_size() {
        let t = mem_image::<u8>(320, 240, PixelFormat::PlanarRgb);
        assert_eq!(t.row_stride(), None);
        assert_eq!(t.effective_row_stride(), Some(320));
        assert_eq!(t.capacity_bytes(), 320 * 240 * 3);

        let t = mem_image::<f32>(100, 40, PixelFormat::PlanarRgb);
        assert_eq!(t.row_stride(), None);
        assert_eq!(t.effective_row_stride(), Some(400));
        assert_eq!(t.capacity_bytes(), 400 * 120);
    }

    #[test]
    #[cfg(target_os = "linux")]
    fn dma_packed_image_records_the_aligned_pitch_only_when_padded() {
        let _lock = crate::tests::fd_lock_shared();
        if !crate::test_support::dma_or_skip(
            "dma_packed_image_records_the_aligned_pitch_only_when_padded",
        ) {
            return;
        }
        let t = Tensor::<u8>::image(
            100,
            10,
            PixelFormat::Rgba,
            Some(TensorMemory::DmaBuf),
            CpuAccess::ReadWrite,
        )
        .unwrap();
        assert_eq!(t.row_stride(), Some(448), "400 B rows pad to 448");
        let t = Tensor::<u8>::image(
            64,
            10,
            PixelFormat::Rgba,
            Some(TensorMemory::DmaBuf),
            CpuAccess::ReadWrite,
        )
        .unwrap();
        assert_eq!(t.row_stride(), None, "256 B rows are already aligned");
    }

    #[test]
    #[cfg(target_os = "linux")]
    fn image_with_stride_validates_the_stride_before_the_memory() {
        match Tensor::<u8>::image_with_stride(
            640,
            480,
            PixelFormat::Rgba,
            2559,
            Some(TensorMemory::Mem),
            CpuAccess::ReadWrite,
        ) {
            Err(Error::InvalidArgument(msg)) => {
                assert!(msg.contains("2559 < minimum 2560"), "{msg}")
            }
            other => panic!("expected InvalidArgument, got {other:?}"),
        }
        match Tensor::<u8>::image_with_stride(
            640,
            480,
            PixelFormat::Rgba,
            2560,
            Some(TensorMemory::Mem),
            CpuAccess::ReadWrite,
        ) {
            Err(Error::NotImplemented(msg)) => assert!(msg.contains("only TensorMemory::DmaBuf")),
            other => panic!("a valid stride on Mem must reach the memory check, got {other:?}"),
        }
    }

    #[test]
    fn set_format_checks_the_planar_and_semi_planar_shape() {
        let mut t = raw_mem::<u8>(&[480, 640, 3]);
        match t.set_format(PixelFormat::PlanarRgb) {
            Err(Error::InvalidShape(msg)) => assert!(msg.contains("[3, H, W]"), "{msg}"),
            other => panic!("[480, 640, 3] is not planar, got {other:?}"),
        }
        assert!(raw_mem::<u8>(&[3, 480, 640])
            .set_format(PixelFormat::PlanarRgb)
            .is_ok());

        for (format, bad, good) in [
            (PixelFormat::Nv12, 721, [720usize, 722]),
            (PixelFormat::Nv16, 481, [480, 482]),
            (PixelFormat::Nv24, 1441, [1440, 1443]),
        ] {
            match raw_mem::<u8>(&[bad, 640]).set_format(format) {
                Err(Error::InvalidShape(msg)) => {
                    assert!(msg.contains(&bad.to_string()), "{format:?}: {msg}")
                }
                other => panic!("{format:?} [{bad}, 640] must be refused, got {other:?}"),
            }
            for rows in good {
                raw_mem::<u8>(&[rows, 640])
                    .set_format(format)
                    .unwrap_or_else(|e| panic!("{format:?} [{rows}, 640]: {e}"));
            }
        }
    }

    #[test]
    fn contiguous_semi_planar_height_divides_the_combined_rows() {
        assert_eq!(
            mem_image::<u8>(320, 240, PixelFormat::Nv24).height(),
            Some(240)
        );
        assert_eq!(
            mem_image::<u8>(320, 240, PixelFormat::Nv16).height(),
            Some(240)
        );
        assert_eq!(
            mem_image::<u8>(320, 241, PixelFormat::Nv12).height(),
            Some(241)
        );
    }

    #[test]
    fn effective_row_stride_scales_with_the_element_size() {
        assert_eq!(
            mem_image::<f32>(10, 4, PixelFormat::Rgb).effective_row_stride(),
            Some(120)
        );
        assert_eq!(
            mem_image::<f32>(10, 4, PixelFormat::PlanarRgb).effective_row_stride(),
            Some(40)
        );
        let mut t = raw_mem::<u16>(&[360, 320]);
        t.set_format(PixelFormat::Nv12).unwrap();
        assert_eq!(t.effective_row_stride(), Some(640));
        let mut t = raw_mem::<u16>(&[360, 321]);
        t.set_format(PixelFormat::Nv12).unwrap();
        assert_eq!(
            t.effective_row_stride(),
            Some(644),
            "odd width rounds up to even"
        );
    }

    #[test]
    fn set_row_stride_minimum_scales_with_the_element_size() {
        let refuse = |mut t: Tensor<f32>, stride: usize, min: usize| match t.set_row_stride(stride)
        {
            Err(Error::InvalidArgument(msg)) => {
                assert!(msg.contains(&format!("{stride} < minimum {min}")), "{msg}")
            }
            other => panic!("stride {stride} under {min} must be refused, got {other:?}"),
        };
        refuse(mem_image::<f32>(10, 4, PixelFormat::Rgb), 119, 120);
        refuse(mem_image::<f32>(10, 4, PixelFormat::PlanarRgb), 39, 40);
        let mut t = mem_image::<f32>(10, 4, PixelFormat::Rgb);
        t.set_row_stride(120).unwrap();
        assert_eq!(t.row_stride(), Some(120));
        let mut t = mem_image::<f32>(10, 4, PixelFormat::PlanarRgb);
        t.set_row_stride(40).unwrap();
        assert_eq!(t.row_stride(), Some(40));

        let mut t = mem_image::<u16>(101, 64, PixelFormat::Nv12);
        match t.set_row_stride(203) {
            Err(Error::InvalidArgument(msg)) => assert!(msg.contains("203 < minimum 204"), "{msg}"),
            other => panic!("expected InvalidArgument, got {other:?}"),
        }
        t.set_row_stride(204).unwrap();
    }

    #[test]
    fn configure_image_semi_planar_stride_scales_with_the_element_size() {
        let mut t = mem_image::<u16>(640, 480, PixelFormat::Rgba);
        t.configure_image(100, 64, PixelFormat::Nv12).unwrap();
        assert_eq!(t.row_stride(), Some(256));
    }

    #[test]
    fn configure_image_keeps_host_packed_odd_widths_tight() {
        let mut t = mem_image::<u8>(640, 480, PixelFormat::Rgb);
        t.configure_image(321, 240, PixelFormat::Rgb).unwrap();
        assert_eq!(t.row_stride(), None);
        assert_eq!(t.effective_row_stride(), Some(963));
    }

    #[test]
    fn configure_image_does_not_inherit_a_stride_under_the_same_format() {
        // Host packed: the prior stride is below the new minimum row.
        let mut t = mem_image::<u8>(640, 480, PixelFormat::Rgb);
        t.configure_image(100, 100, PixelFormat::Rgb).unwrap();
        t.set_row_stride(400).unwrap();
        t.configure_image(300, 100, PixelFormat::Rgb).unwrap();
        assert_eq!(t.row_stride(), None);
        assert_eq!(t.effective_row_stride(), Some(900));

        // Semi-planar whose aligned stride does not fit: the tensor reports the
        // tight row.
        let mut t = mem_image::<u8>(64, 1000, PixelFormat::Nv12);
        assert_eq!(t.row_stride(), Some(64));
        t.configure_image(200, 300, PixelFormat::Nv12).unwrap();
        assert_eq!(t.row_stride(), None);
        assert_eq!(t.effective_row_stride(), Some(200));
    }

    #[test]
    fn configure_image_pool_reuse_picks_the_stride_that_fits() {
        // A prior stride narrower than the new row is replaced by the
        // aligned one.
        let mut t = mem_image::<u8>(64, 1024, PixelFormat::Nv12);
        t.configure_image(200, 100, PixelFormat::Nv12).unwrap();
        assert_eq!(t.row_stride(), Some(256));

        // An unaligned prior stride is replaced too.
        let mut t = mem_image::<u8>(64, 64, PixelFormat::Nv12);
        t.set_row_stride(100).unwrap();
        t.configure_image(64, 64, PixelFormat::Nv12).unwrap();
        assert_eq!(t.row_stride(), Some(64));

        // A wide prior stride whose rows would overrun the allocation is replaced
        // by the aligned stride of the new width.
        let mut t = mem_image::<u8>(640, 480, PixelFormat::Nv12);
        t.configure_image(320, 900, PixelFormat::Nv12).unwrap();
        assert_eq!(t.row_stride(), Some(320));

        // The tight row exactly fills a raw allocation.
        let mut t = raw_mem::<u8>(&[150 * 100]);
        t.configure_image(100, 100, PixelFormat::Nv12).unwrap();
        assert_eq!(t.row_stride(), None);
        assert_eq!(t.effective_row_stride(), Some(100));
    }

    #[test]
    fn configure_image_refuses_rows_that_overrun_the_allocation() {
        // Odd width 99: the shape [150, 99] fits 14850 bytes, but NV12 rows
        // are at least the even width, 100 * 150 = 15000 bytes.
        let mut t = raw_mem::<u8>(&[150 * 99]);
        match t.configure_image(99, 100, PixelFormat::Nv12) {
            Err(Error::InsufficientCapacity { needed, capacity }) => {
                assert_eq!((needed, capacity), (15000, 14850))
            }
            other => panic!("expected InsufficientCapacity, got {other:?}"),
        }
    }

    #[test]
    #[cfg(target_os = "linux")]
    fn configure_image_on_dma_aligns_packed_and_planar_rows() {
        let _lock = crate::tests::fd_lock_shared();
        if !crate::test_support::dma_or_skip("configure_image_on_dma_aligns_packed_and_planar_rows")
        {
            return;
        }
        // Packed u16 Rgb at width 100: 600 B rows align to 640.
        let mut t = Tensor::<u16>::image(
            640,
            480,
            PixelFormat::Rgba,
            Some(TensorMemory::DmaBuf),
            CpuAccess::ReadWrite,
        )
        .unwrap();
        t.configure_image(100, 50, PixelFormat::Rgb).unwrap();
        assert_eq!(t.row_stride(), Some(640));

        // Planar u16 at width 100: 200 B rows would align to 256, but 150
        // rows of 256 B overrun the 32 KiB buffer while 150 rows of 200 B fit.
        let mut t = Tensor::<u16>::new(&[16 * 1024], Some(TensorMemory::DmaBuf), None).unwrap();
        assert_eq!(t.capacity_bytes(), 32 * 1024);
        t.configure_image(100, 50, PixelFormat::PlanarRgb).unwrap();
        assert_eq!(t.row_stride(), None);
        assert_eq!(t.effective_row_stride(), Some(200));
        t.configure_image(100, 40, PixelFormat::PlanarRgb).unwrap();
        assert_eq!(t.row_stride(), Some(256), "120 rows of 256 B fit");
    }

    #[test]
    fn from_planes_requires_a_raw_two_dimensional_chroma() {
        let formatted_chroma = mem_image::<u8>(640, 240, PixelFormat::Grey);
        match Tensor::from_planes(
            raw_mem::<u8>(&[480, 640]),
            formatted_chroma,
            PixelFormat::Nv12,
        ) {
            Err(Error::InvalidArgument(msg)) => assert!(msg.contains("raw tensor"), "{msg}"),
            other => panic!("a formatted chroma plane must be refused, got {other:?}"),
        }
        match Tensor::from_planes(
            raw_mem::<u8>(&[480, 640, 1]),
            raw_mem::<u8>(&[240, 640]),
            PixelFormat::Nv12,
        ) {
            Err(Error::InvalidArgument(msg)) => assert!(msg.contains("2D shapes"), "{msg}"),
            other => panic!("a 3-D luma plane must be refused, got {other:?}"),
        }
    }

    #[test]
    fn from_planes_nv16_needs_full_height_chroma() {
        let img = Tensor::from_planes(
            raw_mem::<u8>(&[480, 640]),
            raw_mem::<u8>(&[480, 640]),
            PixelFormat::Nv16,
        )
        .expect("NV16 chroma is full height");
        assert_eq!(img.format(), Some(PixelFormat::Nv16));
        assert_eq!(img.height(), Some(480));

        match Tensor::from_planes(
            raw_mem::<u8>(&[480, 640]),
            raw_mem::<u8>(&[240, 640]),
            PixelFormat::Nv16,
        ) {
            Err(Error::InvalidArgument(msg)) => {
                assert!(msg.contains("NV16 chroma height"), "{msg}")
            }
            other => panic!("half-height NV16 chroma must be refused, got {other:?}"),
        }
    }

    #[test]
    fn chroma_mut_writes_the_chroma_plane() {
        let mut img = Tensor::from_planes(
            raw_mem::<u8>(&[4, 8]),
            raw_mem::<u8>(&[2, 8]),
            PixelFormat::Nv12,
        )
        .unwrap();
        let chroma = img.chroma_mut().expect("a multiplane image has chroma");
        chroma.map().unwrap().as_mut_slice().fill(0x80);
        let chroma = img.chroma().unwrap();
        assert!(chroma.map().unwrap().as_slice().iter().all(|&b| b == 0x80));
    }

    #[test]
    fn view_offset_and_stride_scale_with_the_element_size() {
        let t = mem_image::<u8>(100, 80, PixelFormat::Rgba);
        let v = t.view(Region::new(10, 20, 30, 40)).unwrap();
        assert_eq!(v.plane_offset(), Some(20 * 400 + 10 * 4));
        assert_eq!(v.row_stride(), Some(400));

        let t = mem_image::<u16>(100, 80, PixelFormat::Rgba);
        let v = t.view(Region::new(10, 20, 30, 40)).unwrap();
        assert_eq!(v.plane_offset(), Some(20 * 800 + 10 * 8));
        assert_eq!(v.row_stride(), Some(800));
        assert_eq!(v.view_origin().map(|o| o.parent_row_stride), Some(800));
    }

    #[test]
    fn single_row_view_keeps_a_tight_stride() {
        let t = mem_image::<u8>(8, 4, PixelFormat::Rgba);
        let row = t.view(Region::new(2, 3, 4, 1)).unwrap();
        assert_eq!(row.effective_row_stride(), Some(16));
        assert_eq!(row.map().unwrap().as_slice().len(), 16);
        assert_eq!(row.view_origin().map(|o| o.parent_row_stride), Some(32));

        let two_rows = t.view(Region::new(2, 2, 4, 2)).unwrap();
        assert_eq!(two_rows.effective_row_stride(), Some(32));
    }

    #[test]
    fn view_refuses_a_region_past_either_edge() {
        let t = mem_image::<u8>(64, 40, PixelFormat::Rgba);
        for region in [Region::new(0, 30, 10, 20), Region::new(60, 0, 10, 1)] {
            match t.view(region) {
                Err(Error::RegionOutOfBounds { region: r, bounds }) => {
                    assert_eq!((r, bounds), (region, (64, 40)))
                }
                other => panic!("{region:?} must be refused, got {other:?}"),
            }
        }
        assert!(Region::new(0, 0, 64, 40).fits_within(64, 40));
        assert!(!Region::new(1, 0, 64, 40).fits_within(64, 40));
        assert!(!Region::new(0, 1, 64, 40).fits_within(64, 40));
    }
}

#[cfg(all(test, not(edgefirst_tensor_dynamic)))]
mod strided_access_tests {
    use super::*;

    /// A patterned byte view of `t`'s whole mapping: byte `i` is `i % 251`.
    fn fill_pattern<T>(t: &Tensor<T>)
    where
        T: Num + Clone + fmt::Debug + Send + Sync,
    {
        let mut m = t.map().unwrap();
        let s = m.as_mut_slice();
        // SAFETY: plain numeric elements; every byte pattern is a valid value
        // for the integer and float types these tests use.
        let bytes = unsafe {
            std::slice::from_raw_parts_mut(s.as_mut_ptr() as *mut u8, std::mem::size_of_val(s))
        };
        for (i, b) in bytes.iter_mut().enumerate() {
            *b = (i % 251) as u8;
        }
    }

    fn expected_rows(rows: usize, stride: usize, row_bytes: usize) -> Vec<u8> {
        (0..rows)
            .flat_map(|r| (r * stride..r * stride + row_bytes).map(|i| (i % 251) as u8))
            .collect()
    }

    fn padded<T>(format: PixelFormat, stride: usize) -> Tensor<T>
    where
        T: Num + Clone + fmt::Debug + Send + Sync + 'static,
    {
        let mut t = Tensor::<T>::image_with_capacity(
            16,
            4,
            format,
            Some(TensorMemory::Mem),
            CpuAccess::ReadWrite,
        )
        .unwrap();
        t.configure_image(8, 4, format).unwrap();
        t.set_row_stride(stride).unwrap();
        t
    }

    #[test]
    fn copy_to_flat_compacts_padded_packed_rows() {
        // 8 px * 3 B = 24 B rows at a 32 B pitch.
        let t = padded::<u8>(PixelFormat::Rgb, 32);
        fill_pattern(&t);
        let mut dst = vec![0u8; 96];
        t.copy_to_flat(&mut dst).unwrap();
        assert_eq!(dst, expected_rows(4, 32, 24));
    }

    #[test]
    fn copy_to_flat_compacts_padded_planar_rows() {
        // f32 planar: 3 planes x 4 rows of 32 B at a 64 B pitch.
        let t = padded::<f32>(PixelFormat::PlanarRgb, 64);
        fill_pattern(&t);
        let mut dst = vec![0u8; 384];
        t.copy_to_flat(&mut dst).unwrap();
        assert_eq!(dst, expected_rows(12, 64, 32));
    }

    #[test]
    fn copy_to_flat_of_a_tight_stride_is_the_mapping() {
        let t = padded::<u8>(PixelFormat::Rgb, 24);
        fill_pattern(&t);
        let mut dst = vec![0u8; 96];
        t.copy_to_flat(&mut dst).unwrap();
        assert_eq!(dst, expected_rows(1, 96, 96));

        let t = Tensor::<u8>::image(
            8,
            4,
            PixelFormat::Rgb,
            Some(TensorMemory::Mem),
            CpuAccess::ReadWrite,
        )
        .unwrap();
        fill_pattern(&t);
        let mut dst = vec![0u8; 96];
        t.copy_to_flat(&mut dst).unwrap();
        assert_eq!(dst, expected_rows(1, 96, 96));
    }

    #[test]
    fn copy_to_flat_needs_a_tight_sized_destination() {
        let t = padded::<u8>(PixelFormat::Rgb, 32);
        let mut dst = vec![0u8; 95];
        match t.copy_to_flat(&mut dst) {
            Err(Error::InvalidArgument(msg)) => {
                assert!(
                    msg.contains("dst is 95 bytes") && msg.contains("96 bytes"),
                    "{msg}"
                )
            }
            other => panic!("expected InvalidArgument, got {other:?}"),
        }
    }

    #[test]
    fn copy_to_flat_compacts_a_padded_batch() {
        // Two 8x2 RGB tiles at a 32 B pitch: four rows in all.
        let mut t = padded::<u8>(PixelFormat::Rgb, 32);
        t.set_logical_shape(&[2, 2, 8, 3]).unwrap();
        fill_pattern(&t);
        let mut dst = vec![0u8; 96];
        t.copy_to_flat(&mut dst).unwrap();
        assert_eq!(dst, expected_rows(4, 32, 24));
    }

    #[test]
    fn strided_planar_map_spans_every_plane_row() {
        let t = padded::<u8>(PixelFormat::PlanarRgb, 16);
        assert_eq!(t.map().unwrap().as_slice().len(), 16 * 12);
    }

    #[test]
    fn batch_of_a_padded_batch_starts_at_its_padded_tile() {
        // Tile 1 starts after two 32 B rows.
        let mut t = padded::<u8>(PixelFormat::Rgb, 32);
        t.set_logical_shape(&[2, 2, 8, 3]).unwrap();
        fill_pattern(&t);
        let e = t.batch(1).unwrap();
        assert_eq!(e.plane_offset(), Some(64));
        let mut dst = vec![0u8; 48];
        e.copy_to_flat(&mut dst).unwrap();
        let tile1: Vec<u8> = expected_rows(4, 32, 24).split_off(48);
        assert_eq!(dst, tile1);
    }

    #[test]
    fn strided_packed_batch_map_spans_every_tile_row() {
        let mut t = padded::<u8>(PixelFormat::Rgb, 32);
        t.set_logical_shape(&[2, 2, 8, 3]).unwrap();
        assert_eq!(t.map().unwrap().as_slice().len(), 32 * 4);
    }

    #[test]
    fn strided_access_to_a_flattened_image_is_refused() {
        let mut t = padded::<u8>(PixelFormat::PlanarRgb, 16);
        t.set_logical_shape(&[96]).unwrap();
        match t.map() {
            Err(Error::InvalidOperation(msg)) => assert!(msg.contains("[96]"), "{msg}"),
            other => panic!("expected InvalidOperation, got {other:?}"),
        }
        let mut dst = vec![0u8; 96];
        match t.copy_to_flat(&mut dst) {
            Err(Error::InvalidOperation(msg)) => assert!(msg.contains("[96]"), "{msg}"),
            other => panic!("expected InvalidOperation, got {other:?}"),
        }
    }

    #[test]
    #[cfg(target_os = "linux")]
    fn strided_shm_map_is_bounded_by_the_segment() {
        let _lock = crate::tests::fd_lock_shared();
        let mut t = Tensor::<u8>::image(
            640,
            480,
            PixelFormat::Nv12,
            Some(TensorMemory::Shm),
            CpuAccess::ReadWrite,
        )
        .expect("Linux has POSIX shared memory");
        assert_eq!(t.capacity_bytes(), 640 * 720);
        assert_eq!(t.map().unwrap().as_slice().len(), 640 * 720);

        t.configure_image(320, 240, PixelFormat::Nv12).unwrap();
        assert_eq!(t.row_stride(), Some(640), "the pool pitch fits and is kept");
        assert_eq!(t.map().unwrap().as_slice().len(), 640 * 360);

        t.configure_image(640, 480, PixelFormat::Nv12).unwrap();
        t.set_row_stride(704).unwrap();
        match t.map() {
            Err(Error::InsufficientCapacity { needed, capacity }) => {
                assert_eq!((needed, capacity), (704 * 720, 640 * 720))
            }
            other => panic!("expected InsufficientCapacity, got {other:?}"),
        }
    }

    /// Minimal host-memory `PboOps`: a fixed buffer, never resized.
    struct HostPbo(Vec<u8>);

    // SAFETY: the pointer addresses `self.0`, allocated once and never
    // resized, and the tensor holds the `Arc` to `self` for every map's life.
    unsafe impl PboOps for HostPbo {
        fn map_buffer(&self, _buffer_id: u32, size: usize) -> Result<PboMapping> {
            assert!(size <= self.0.len());
            Ok(PboMapping {
                ptr: self.0.as_ptr() as *mut u8,
                size,
            })
        }
        fn unmap_buffer(&self, _buffer_id: u32) -> Result<()> {
            Ok(())
        }
        fn delete_buffer(&self, _buffer_id: u32) {}
    }

    fn pbo_rgba(width: usize, height: usize) -> Tensor<u8> {
        let bytes = width * height * 4;
        let pbo = PboTensor::<u8>::from_pbo(
            7,
            bytes,
            &[height, width, 4],
            None,
            Arc::new(HostPbo(vec![0; bytes])),
        )
        .unwrap();
        let mut t = Tensor::from_pbo(pbo).unwrap();
        t.set_format(PixelFormat::Rgba).unwrap();
        t
    }

    #[test]
    fn strided_pbo_map_is_bounded_by_the_buffer() {
        let mut t = pbo_rgba(8, 4);
        t.set_row_stride(32).unwrap();
        assert_eq!(t.map().unwrap().as_slice().len(), 128);

        t.set_row_stride(64).unwrap();
        match t.map() {
            Err(Error::InsufficientCapacity { needed, capacity }) => {
                assert_eq!((needed, capacity), (256, 128))
            }
            other => panic!("expected InsufficientCapacity, got {other:?}"),
        }
    }

    #[test]
    fn pbo_accessors_answer_only_for_a_pbo_tensor() {
        let mem = Tensor::<u8>::new(&[4], Some(TensorMemory::Mem), None).unwrap();
        assert_eq!(mem.pbo_is_mapped(), None);
        assert!(mem.pbo_keepalive().is_none());

        let t = pbo_rgba(8, 4);
        assert_eq!(t.pbo_is_mapped(), Some(false));
        assert!(t.pbo_keepalive().is_some());
        let m = t.map().unwrap();
        assert_eq!(t.pbo_is_mapped(), Some(true));
        drop(m);
        assert_eq!(t.pbo_is_mapped(), Some(false));
    }
}

#[cfg(all(test, not(edgefirst_tensor_dynamic)))]
mod accessor_tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    #[test]
    #[cfg(unix)]
    fn plane_descriptor_carries_its_stride_and_offset() {
        use std::os::fd::AsFd;
        let _lock = crate::tests::fd_lock_shared();
        let file = std::fs::File::open("/dev/null").unwrap();
        let pd = PlaneDescriptor::new(file.as_fd()).unwrap();
        assert_eq!((pd.stride(), pd.offset()), (None, None));
        let pd = pd.with_stride(832).with_offset(4096);
        assert_eq!((pd.stride(), pd.offset()), (Some(832), Some(4096)));
    }

    #[test]
    #[cfg(target_os = "linux")]
    fn as_dma_answers_only_for_a_dma_buffer() {
        use std::os::fd::AsRawFd;
        let _lock = crate::tests::fd_lock_shared();
        let mem = Tensor::<u8>::new(&[64], Some(TensorMemory::Mem), None).unwrap();
        assert!(mem.as_dma().is_none());
        if !crate::test_support::dma_or_skip("as_dma_answers_only_for_a_dma_buffer") {
            return;
        }
        let t = Tensor::<u8>::new(&[64], Some(TensorMemory::DmaBuf), None).unwrap();
        let dma = t.as_dma().expect("a DMA tensor downcasts");
        assert_eq!(
            dma.fd.as_raw_fd(),
            t.dmabuf().unwrap().as_raw_fd(),
            "the downcast is this tensor's own buffer"
        );
    }

    #[test]
    fn derived_identity_mixes_the_kind_into_the_key() {
        let id = BufferIdentity::derived(IdentityKind::Shm, u64::MAX);
        assert_eq!(id.id(), ((IdentityKind::Shm as u64) << 56) ^ u64::MAX);
        assert_ne!(
            id.id(),
            BufferIdentity::derived(IdentityKind::DmaBuf, u64::MAX).id()
        );
    }

    #[test]
    fn empty_is_a_zero_element_shape() {
        let mut t = Tensor::<f32>::new(&[2, 3], Some(TensorMemory::Mem), None).unwrap();
        assert!(!TensorTrait::is_empty(&t));
        {
            let m = t.map().unwrap();
            assert_eq!(TensorMapTrait::len(&m), 6);
            assert_eq!(TensorMapTrait::size(&m), 24);
            assert!(!TensorMapTrait::is_empty(&m));
        }
        t.set_logical_shape(&[0]).unwrap();
        assert!(TensorTrait::is_empty(&t));
        let m = t.map().unwrap();
        assert_eq!(TensorMapTrait::size(&m), 0);
        assert!(TensorMapTrait::is_empty(&m));
    }

    struct CountingCudaOps {
        maps: AtomicUsize,
    }

    impl CudaGlOps for CountingCudaOps {
        fn map(&self, _r: crate::cuda::GraphicsResource) -> Option<(*mut std::ffi::c_void, usize)> {
            self.maps.fetch_add(1, Ordering::SeqCst);
            Some((0x1000usize as *mut _, 64))
        }
        fn unmap(&self, _r: crate::cuda::GraphicsResource) {}
        fn unregister(&self, _r: crate::cuda::GraphicsResource) {}
    }

    #[test]
    fn cuda_accessors_follow_the_attached_handle() {
        let mut t = Tensor::<f32>::new(&[16], Some(TensorMemory::Mem), None).unwrap();
        assert!(!t.is_cuda_attached());
        assert!(t.cuda().is_none());
        assert!(t.cuda_map().is_none());
        assert!(t.cuda_map_mut().is_none());

        let ops = Arc::new(CountingCudaOps {
            maps: AtomicUsize::new(0),
        });
        t.set_cuda_handle(CudaHandle::new_gl(
            0x1usize as crate::cuda::GraphicsResource,
            64,
            ops.clone(),
        ));
        assert!(t.is_cuda_attached());
        assert!(t.cuda().is_some());
        assert_eq!(t.cuda_map().map(|m| m.device_ptr() as usize), Some(0x1000));
        assert_eq!(t.cuda_map_mut().map(|m| m.len()), Some(64));
        assert_eq!(ops.maps.load(Ordering::SeqCst), 2);
    }

    #[test]
    fn from_slice_copies_values_of_a_matching_shape() {
        let t = Tensor::<u16>::from_slice(&[1, 2, 3, 4, 5, 6], &[2, 3]).unwrap();
        assert_eq!(t.shape(), &[2, 3]);
        assert_eq!(t.map().unwrap().as_slice(), &[1, 2, 3, 4, 5, 6]);
        match Tensor::<u16>::from_slice(&[1, 2, 3, 4, 5], &[2, 3]) {
            Err(Error::InvalidShape(msg)) => assert!(msg.contains("values.len()=5"), "{msg}"),
            other => panic!("expected InvalidShape, got {other:?}"),
        }
    }

    /// An `Any` compression request with no pinned memory tries the
    /// platform's zero-copy image first and falls back when it cannot serve
    /// the format.
    #[cfg(not(target_os = "android"))]
    #[test]
    fn any_compression_request_falls_back_when_zero_copy_cannot_serve() {
        let _lock = crate::tests::fd_lock_shared();
        let desc = ImageDesc::new(10, 10, PixelFormat::Rgb, DType::F32)
            .with_access(CpuAccess::None)
            .with_compression(Compression::Any);
        let t = Tensor::<f32>::image_desc(&desc).expect("Any never fails for compression reasons");
        assert_eq!(t.shape(), &[10, 10, 3]);
        assert_eq!(t.compression(), None);
    }
}

#[cfg(test)]
mod cpu_access_tests {
    use super::*;

    #[test]
    fn covers_matrix() {
        use CpuAccess::*;
        // Every declaration covers a narrower or equal request.
        for a in [None, Read, Write, ReadWrite] {
            assert!(a.covers(None), "{a:?} must cover None");
            assert!(ReadWrite.covers(a), "ReadWrite must cover {a:?}");
        }
        assert!(Read.covers(Read));
        assert!(Write.covers(Write));
        // No declaration covers a wider one.
        assert!(!None.covers(Read));
        assert!(!None.covers(Write));
        assert!(!Read.covers(Write));
        assert!(!Read.covers(ReadWrite));
        assert!(!Write.covers(Read));
        assert!(!Write.covers(ReadWrite));
    }

    #[test]
    fn union_covers_both_and_require_names_both() {
        use CpuAccess::*;
        for a in [None, Read, Write, ReadWrite] {
            for b in [None, Read, Write, ReadWrite] {
                let u = a.union(b);
                assert!(u.covers(a) && u.covers(b), "{a:?} | {b:?} = {u:?}");
                assert_eq!(a.require(b).is_ok(), a.covers(b));
            }
        }
        match Read.require(Write) {
            Err(Error::CpuAccessDenied {
                declared: Read,
                requested: Write,
            }) => {}
            other => panic!("expected CpuAccessDenied, got {other:?}"),
        }
    }

    #[test]
    fn map_with_none_is_invalid() {
        let t = Tensor::<u8>::new(&[16], Some(TensorMemory::Mem), None).unwrap();
        match t.map_with(CpuAccess::None) {
            Err(Error::InvalidArgument(_)) => {}
            Err(other) => panic!("expected InvalidArgument, got {other:?}"),
            Ok(_) => panic!("map_with(CpuAccess::None) must not succeed"),
        }
    }

    #[test]
    fn read_map_rejects_mutation_uniformly() {
        // Mem backend: map_read yields a read view whose mutable accessor panics.
        let t = Tensor::<u8>::new(&[8], Some(TensorMemory::Mem), None).unwrap();
        t.map_mut().unwrap().as_mut_slice().copy_from_slice(&[7; 8]);
        let ro = t.map_read().unwrap();
        assert_eq!(ro.as_slice(), &[7; 8]);
        drop(ro);
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let mut ro = t.map_read().unwrap();
            let _ = ro.as_mut_slice();
        }));
        assert!(result.is_err(), "as_mut_slice through map_read must panic");
    }

    #[test]
    fn write_and_rw_maps_stay_mutable() {
        let t = Tensor::<u8>::new(&[4], Some(TensorMemory::Mem), None).unwrap();
        t.map_write()
            .unwrap()
            .as_mut_slice()
            .copy_from_slice(&[1; 4]);
        t.map().unwrap().as_mut_slice().copy_from_slice(&[2; 4]);
        assert_eq!(t.map_read().unwrap().as_slice(), &[2; 4]);
    }

    /// A `map_read` of an IOSurface observes data written through a prior
    /// read-write lock, and its unlock leaves later reads intact.
    #[test]
    #[cfg(target_os = "macos")]
    fn iosurface_read_only_lock_roundtrip() {
        let _lock = crate::tests::fd_lock_shared();
        let expect = if crate::is_iosurface_available() {
            crate::test_support::Expect::Succeeds
        } else {
            crate::test_support::Expect::MayFail("the IOSurface probe failed on this host")
        };
        let Some(t) = crate::test_support::allocated(
            Tensor::<u8>::new(&[64], Some(TensorMemory::DmaBuf), None),
            "IOSurface allocation",
            expect,
        ) else {
            return;
        };
        {
            let mut m = t.map_mut().unwrap();
            for (i, b) in m.as_mut_slice().iter_mut().enumerate() {
                *b = (i * 3) as u8;
            }
        }
        for _ in 0..2 {
            let ro = t.map_read().unwrap();
            for (i, b) in ro.as_slice().iter().enumerate() {
                assert_eq!(*b, (i * 3) as u8, "byte {i} through read-only lock");
            }
        }
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let mut ro = t.map_read().unwrap();
            let _ = ro.as_mut_slice();
        }));
        assert!(result.is_err(), "IOSurface read map must reject mutation");
    }
}

#[cfg(test)]
mod tests {
    use std::sync::{PoisonError, RwLock};

    use super::*;

    #[ctor::ctor(unsafe)]
    fn init() {
        env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();
    }

    /// Macro to get the current function name for logging in tests.
    #[cfg(target_os = "linux")]
    macro_rules! function {
        () => {{
            fn f() {}
            fn type_name_of<T>(_: T) -> &'static str {
                std::any::type_name::<T>()
            }
            let name = type_name_of(f);

            // Find and cut the rest of the path
            match &name[..name.len() - 3].rfind(':') {
                Some(pos) => &name[pos + 1..name.len() - 3],
                None => &name[..name.len() - 3],
            }
        }};
    }

    #[test]
    #[cfg(target_os = "linux")]
    fn test_tensor() {
        let _lock = crate::tests::fd_lock_shared();
        let shape = vec![1];
        let tensor = DmaTensor::<f32>::new(&shape, Some("dma_tensor"));
        let dma_enabled = tensor.is_ok();

        let tensor = Tensor::<f32>::new(&shape, None, None).expect("Failed to create tensor");
        // Auto-select priority is Dma > Mem; Shm is never auto-selected.
        match dma_enabled {
            true => assert_eq!(tensor.memory(), TensorMemory::DmaBuf),
            false => assert_eq!(tensor.memory(), TensorMemory::Mem),
        }
    }

    #[test]
    #[cfg(any(target_os = "macos", target_os = "ios"))]
    fn test_tensor() {
        let shape = vec![1];
        let tensor = Tensor::<f32>::new(&shape, None, None).expect("Failed to create tensor");
        // macOS/iOS auto-selection: IOSurface (Dma), then Mem.
        let m = tensor.memory();
        assert!(
            matches!(m, TensorMemory::DmaBuf | TensorMemory::Mem),
            "Unexpected auto-fallback result on macOS/iOS: {m:?}"
        );
    }

    #[test]
    #[cfg(all(
        unix,
        not(any(target_os = "linux", target_os = "macos", target_os = "ios"))
    ))]
    fn test_tensor() {
        let shape = vec![1];
        let tensor = Tensor::<f32>::new(&shape, None, None).expect("Failed to create tensor");
        // Other Unix (BSD): auto-selection is Mem.
        assert_eq!(tensor.memory(), TensorMemory::Mem);
    }

    #[test]
    #[cfg(not(unix))]
    fn test_tensor() {
        let shape = vec![1];
        let tensor = Tensor::<f32>::new(&shape, None, None).expect("Failed to create tensor");
        assert_eq!(tensor.memory(), TensorMemory::Mem);
    }

    #[test]
    #[cfg(target_os = "linux")]
    fn test_dma_tensor() {
        let _lock = crate::tests::fd_lock_shared();
        if !crate::test_support::dma_or_skip("test_dma_tensor") {
            return;
        }

        let shape = vec![2, 3, 4];
        let tensor =
            DmaTensor::<f32>::new(&shape, Some("test_tensor")).expect("Failed to create tensor");

        const DUMMY_VALUE: f32 = 12.34;

        assert_eq!(tensor.memory(), TensorMemory::DmaBuf);
        assert_eq!(tensor.name(), "test_tensor");
        assert_eq!(tensor.shape(), &shape);
        assert_eq!(tensor.size(), 2 * 3 * 4 * std::mem::size_of::<f32>());
        assert_eq!(tensor.len(), 2 * 3 * 4);

        {
            let mut tensor_map = tensor.map().expect("Failed to map DMA memory");
            tensor_map.fill(42.0);
            assert!(tensor_map.iter().all(|&x| x == 42.0));
        }

        {
            let shared = Tensor::<f32>::from_fd(
                tensor
                    .clone_fd()
                    .expect("Failed to duplicate tensor file descriptor"),
                &shape,
                Some("test_tensor_shared"),
            )
            .expect("Failed to create tensor from fd");

            assert_eq!(shared.memory(), TensorMemory::DmaBuf);
            assert_eq!(shared.name(), "test_tensor_shared");
            assert_eq!(shared.shape(), &shape);

            let mut tensor_map = shared.map().expect("Failed to map DMA memory from fd");
            tensor_map.fill(DUMMY_VALUE);
            assert!(tensor_map.iter().all(|&x| x == DUMMY_VALUE));
        }

        {
            let tensor_map = tensor.map().expect("Failed to map DMA memory");
            assert!(tensor_map.iter().all(|&x| x == DUMMY_VALUE));
        }

        let mut tensor = DmaTensor::<u8>::new(&shape, None).expect("Failed to create tensor");
        assert_eq!(tensor.shape(), &shape);
        let new_shape = vec![3, 4, 4];
        assert!(
            tensor.reshape(&new_shape).is_err(),
            "Reshape should fail due to size mismatch"
        );
        assert_eq!(tensor.shape(), &shape, "Shape should remain unchanged");

        let new_shape = vec![2, 3, 4];
        tensor.reshape(&new_shape).expect("Reshape should succeed");
        assert_eq!(
            tensor.shape(),
            &new_shape,
            "Shape should be updated after successful reshape"
        );

        {
            let mut tensor_map = tensor.map().expect("Failed to map DMA memory");
            tensor_map.fill(1);
            assert!(tensor_map.iter().all(|&x| x == 1));
        }

        {
            let mut tensor_map = tensor.map().expect("Failed to map DMA memory");
            tensor_map[2] = 42;
            assert_eq!(tensor_map[1], 1, "Value at index 1 should be 1");
            assert_eq!(tensor_map[2], 42, "Value at index 2 should be 42");
        }
    }

    #[test]
    #[cfg(unix)]
    fn test_shm_tensor() {
        let _lock = crate::tests::fd_lock_shared();
        if !is_shm_available() {
            crate::test_support::report_skip(
                "test_shm_tensor - SHM allocation unavailable on this platform",
            );
            return;
        }
        let shape = vec![2, 3, 4];
        let tensor =
            ShmTensor::<f32>::new(&shape, Some("test_tensor")).expect("Failed to create tensor");
        assert_eq!(tensor.shape(), &shape);
        assert_eq!(tensor.size(), 2 * 3 * 4 * std::mem::size_of::<f32>());
        assert_eq!(tensor.name(), "test_tensor");

        const DUMMY_VALUE: f32 = 12.34;
        {
            let mut tensor_map = tensor.map().expect("Failed to map shared memory");
            tensor_map.fill(42.0);
            assert!(tensor_map.iter().all(|&x| x == 42.0));
        }

        {
            let shared = Tensor::<f32>::from_fd(
                tensor
                    .clone_fd()
                    .expect("Failed to duplicate tensor file descriptor"),
                &shape,
                Some("test_tensor_shared"),
            )
            .expect("Failed to create tensor from fd");

            assert_eq!(shared.memory(), TensorMemory::Shm);
            assert_eq!(shared.name(), "test_tensor_shared");
            assert_eq!(shared.shape(), &shape);

            let mut tensor_map = shared.map().expect("Failed to map shared memory from fd");
            tensor_map.fill(DUMMY_VALUE);
            assert!(tensor_map.iter().all(|&x| x == DUMMY_VALUE));
        }

        {
            let tensor_map = tensor.map().expect("Failed to map shared memory");
            assert!(tensor_map.iter().all(|&x| x == DUMMY_VALUE));
        }

        let mut tensor = ShmTensor::<u8>::new(&shape, None).expect("Failed to create tensor");
        assert_eq!(tensor.shape(), &shape);
        let new_shape = vec![3, 4, 4];
        assert!(
            tensor.reshape(&new_shape).is_err(),
            "Reshape should fail due to size mismatch"
        );
        assert_eq!(tensor.shape(), &shape, "Shape should remain unchanged");

        let new_shape = vec![2, 3, 4];
        tensor.reshape(&new_shape).expect("Reshape should succeed");
        assert_eq!(
            tensor.shape(),
            &new_shape,
            "Shape should be updated after successful reshape"
        );

        {
            let mut tensor_map = tensor.map().expect("Failed to map shared memory");
            tensor_map.fill(1);
            assert!(tensor_map.iter().all(|&x| x == 1));
        }

        {
            let mut tensor_map = tensor.map().expect("Failed to map shared memory");
            tensor_map[2] = 42;
            assert_eq!(tensor_map[1], 1, "Value at index 1 should be 1");
            assert_eq!(tensor_map[2], 42, "Value at index 2 should be 42");
        }
    }

    #[test]
    fn mem_subview_partitions_parent_buffer() {
        // One heap [2,4] u8 parent (8 bytes); two [1,4] sub-views at byte offsets
        // 0 and 4.
        let parent = Tensor::<u8>::new(&[2, 4], Some(TensorMemory::Mem), None).unwrap();
        let view0 = parent.subview(0, &[1, 4]).expect("subview at offset 0");
        let view1 = parent.subview(4, &[1, 4]).expect("subview at offset 4");

        view1
            .map()
            .unwrap()
            .as_mut_slice()
            .copy_from_slice(&[10, 20, 30, 40]);
        view0
            .map()
            .unwrap()
            .as_mut_slice()
            .copy_from_slice(&[1, 2, 3, 4]);

        assert_eq!(view0.map().unwrap().as_slice(), &[1, 2, 3, 4]);
        assert_eq!(view1.map().unwrap().as_slice(), &[10, 20, 30, 40]);
        assert_eq!(
            parent.map().unwrap().as_slice(),
            &[1, 2, 3, 4, 10, 20, 30, 40]
        );
    }

    #[test]
    fn batch_partitions_leading_dim() {
        // Raw [4,2,2,3] u8 batched tensor: 4 elements of 12 bytes each.
        let parent = Tensor::<u8>::new(&[4, 2, 2, 3], Some(TensorMemory::Mem), None).unwrap();
        for i in 0..4u8 {
            let e = parent.batch(i as usize).expect("batch element");
            assert_eq!(e.shape(), &[2, 2, 3]);
            assert_eq!(e.buffer_identity().id(), parent.buffer_identity().id());
            for b in e.map().unwrap().as_mut_slice() {
                *b = i + 1;
            }
        }
        // Each element occupies its own 12-byte band of the parent.
        let whole = parent.map().unwrap();
        let s = whole.as_slice();
        for i in 0..4usize {
            assert!(
                s[i * 12..(i + 1) * 12].iter().all(|&b| b == (i as u8 + 1)),
                "band {i} not partitioned: {:?}",
                &s[i * 12..(i + 1) * 12]
            );
        }
    }

    #[test]
    fn view_origin_snapshots_parent_and_composes() {
        let parent = Tensor::<u8>::image(
            100,
            80,
            PixelFormat::Rgba,
            Some(TensorMemory::Mem),
            crate::CpuAccess::ReadWrite,
        )
        .unwrap();
        assert_eq!(
            parent.view_origin(),
            None,
            "whole tensor has no view_origin"
        );
        let v = parent.view(Region::new(10, 20, 30, 40)).unwrap();
        assert_eq!(
            v.view_origin(),
            Some(ViewOrigin {
                parent_width: 100,
                parent_height: 80,
                parent_row_stride: 100 * 4, // tight RGBA pitch
                x: 10,
                y: 20
            })
        );
        // A view of a view keeps the root parent and accumulates the origin.
        let v2 = v.view(Region::new(5, 5, 10, 10)).unwrap();
        assert_eq!(
            v2.view_origin(),
            Some(ViewOrigin {
                parent_width: 100,
                parent_height: 80,
                parent_row_stride: 100 * 4,
                x: 15,
                y: 25
            }),
            "nested view composes onto the root parent"
        );
    }

    #[test]
    fn view_origin_none_for_raw_batch() {
        // A raw (unformatted) batched tensor has no pixel geometry.
        let parent = Tensor::<u8>::new(&[4, 2, 2, 3], Some(TensorMemory::Mem), None).unwrap();
        assert_eq!(parent.batch(2).unwrap().view_origin(), None);
    }

    #[test]
    fn batch_rejects_out_of_bounds_index() {
        let parent = Tensor::<u8>::new(&[4, 2, 2, 3], Some(TensorMemory::Mem), None).unwrap();
        match parent.batch(4) {
            Err(Error::BatchIndexOutOfBounds { index, batch }) => {
                assert_eq!((index, batch), (4, 4));
            }
            other => panic!("expected BatchIndexOutOfBounds, got {other:?}"),
        }
    }

    #[test]
    fn batch_zero_on_unit_n_is_whole() {
        // N == 1: batch(0) is the whole per-element block at offset 0.
        let parent = Tensor::<u8>::new(&[1, 2, 2, 3], Some(TensorMemory::Mem), None).unwrap();
        let e = parent.batch(0).unwrap();
        assert_eq!(e.shape(), &[2, 2, 3]);
        assert_eq!(e.plane_offset(), None);
        assert_eq!(e.buffer_identity().id(), parent.buffer_identity().id());
    }

    #[test]
    fn mem_subview_rejects_unaligned_offset() {
        // A byte offset of 2 is not f32-aligned.
        let parent = Tensor::<f32>::new(&[8], Some(TensorMemory::Mem), None).unwrap();
        assert!(parent.subview(2, &[1]).is_err());
        // A correctly aligned offset is accepted.
        assert!(parent.subview(4, &[1]).is_ok());
    }

    #[test]
    fn mem_subview_rejects_out_of_bounds() {
        let parent = Tensor::<u8>::new(&[8], Some(TensorMemory::Mem), None).unwrap();
        // offset 6 + 4 bytes = 10 exceeds the 8-byte allocation.
        assert!(parent.subview(6, &[4]).is_err());
    }

    /// A `subview` shares the parent's `BufferIdentity` on every backend.
    #[test]
    fn subview_shares_buffer_identity_all_backends() {
        let _lock = crate::tests::fd_lock_shared();
        // `window` is the view shape.
        let assert_parent_shares = |parent: Tensor<u8>, window: &[usize], label: &str| {
            let parent_id = parent.buffer_identity().id();
            // Two offset-distinct windows carry the parent's identity.
            let v0 = parent
                .subview(0, window)
                .unwrap_or_else(|e| panic!("{label}: subview(0) failed: {e:?}"));
            let v1 = parent
                .subview(16, window)
                .unwrap_or_else(|e| panic!("{label}: subview(16) failed: {e:?}"));
            assert_eq!(
                v0.buffer_identity().id(),
                parent_id,
                "{label}: subview(0) minted a fresh BufferIdentity"
            );
            assert_eq!(
                v1.buffer_identity().id(),
                parent_id,
                "{label}: subview(16) minted a fresh BufferIdentity"
            );
        };
        let assert_shares = |memory: TensorMemory, label: &str| {
            let parent = Tensor::<u8>::new(&[64], Some(memory), None)
                .unwrap_or_else(|e| panic!("{label}: parent alloc failed: {e:?}"));
            assert_parent_shares(parent, &[16], label);
        };

        assert_shares(TensorMemory::Mem, "Mem");

        #[cfg(unix)]
        if crate::is_shm_available() {
            assert_shares(TensorMemory::Shm, "Shm");
        }

        // Dma is DMA-BUF on Linux, IOSurface on macOS and AHardwareBuffer on Android.
        #[cfg(target_os = "linux")]
        if crate::test_support::dma_or_skip("subview_shares_buffer_identity_all_backends (Dma)") {
            assert_shares(TensorMemory::DmaBuf, "Dma");
        }
        #[cfg(not(any(target_os = "linux", target_os = "windows")))]
        if crate::is_gpu_buffer_available() {
            assert_shares(TensorMemory::DmaBuf, "Dma");
        }
        // On Windows, Dma is an image-formatted D3D11 texture.
        #[cfg(target_os = "windows")]
        if crate::is_gpu_buffer_available() {
            let parent = Tensor::<u8>::image(
                64,
                1,
                PixelFormat::Grey,
                Some(TensorMemory::DmaBuf),
                CpuAccess::ReadWrite,
            )
            .expect("D3D11 texture alloc failed");
            assert_parent_shares(parent, &[1, 16, 1], "Dma");
        }
    }

    /// A sub-view of a D3D11 texture tensor maps at its plane offset.
    #[cfg(target_os = "windows")]
    #[test]
    fn subview_of_a_texture_tensor_maps_at_its_offset() {
        if !crate::is_gpu_buffer_available() {
            crate::test_support::report_skip(
                "subview_of_a_texture_tensor_maps_at_its_offset - no D3D11 device on this host",
            );
            return;
        }
        let t = Tensor::<u8>::image(
            32,
            8,
            PixelFormat::Rgba,
            Some(TensorMemory::DmaBuf),
            CpuAccess::ReadWrite,
        )
        .unwrap();
        let stride = t.effective_row_stride().unwrap();
        let v = t.subview(stride * 2, &[2, 32, 4]).unwrap();
        assert!(
            v.plane_offset().is_some_and(|o| o > 0),
            "a sub-view at a row offset must carry one"
        );
        let m = v
            .map_read()
            .expect("a sub-view of a texture tensor must map");
        assert_eq!(m.as_slice().len(), 2 * stride, "two rows of the window");
    }

    #[test]
    fn mem_subview_four_views_no_aliasing() {
        // One [4,3] f32 parent; four [1,3] views at 12-byte strides, each
        // written independently.
        let parent = Tensor::<f32>::new(&[4, 3], Some(TensorMemory::Mem), None).unwrap();
        let frame = 3 * std::mem::size_of::<f32>();
        for i in 0..4 {
            let v = parent.subview(i * frame, &[1, 3]).unwrap();
            let val = i as f32 + 1.0;
            v.map()
                .unwrap()
                .as_mut_slice()
                .copy_from_slice(&[val, val, val]);
        }
        assert_eq!(
            parent.map().unwrap().as_slice(),
            &[1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 3.0, 3.0, 3.0, 4.0, 4.0, 4.0]
        );
    }

    #[test]
    fn mem_subview_inherits_format_and_row_stride() {
        // A parent with a row stride wider than the tight row.
        let mut parent = Tensor::<u8>::image(
            100,
            100,
            PixelFormat::Rgba,
            Some(TensorMemory::Mem),
            crate::CpuAccess::ReadWrite,
        )
        .unwrap();
        parent.set_row_stride_unchecked(512); // padded stride (> 100*4)
        let view = parent.subview(4096, &[10, 10, 4]).unwrap();
        assert_eq!(view.format(), Some(PixelFormat::Rgba), "format inherited");
        assert_eq!(view.row_stride(), Some(512), "row_stride inherited");
    }

    #[test]
    fn mem_strided_subview_maps_offset_and_byte_size() {
        let parent = Tensor::<u8>::new(&[2048], Some(TensorMemory::Mem), None).unwrap();
        let mut view = parent.subview(128, &[8, 16]).unwrap(); // 8 rows × 16 @ off 128
        assert_eq!(view.plane_offset(), Some(128));
        view.set_row_stride_unchecked(32); // padded stride (> 16)

        {
            let mut m = view.map().unwrap();
            let s = m.as_mut_slice();
            // The padded window: stride(32) × rows(8) = 256.
            assert_eq!(
                s.len(),
                256,
                "strided map exposes the full padded byte window"
            );
            s[0] = 0xAA; // row 0, col 0
            s[32] = 0xBB; // row 1, col 0 (one stride in)
        }

        // The writes land in the parent at the view's offset.
        let p = parent.map().unwrap();
        let pb = p.as_slice();
        assert_eq!(pb[128], 0xAA, "row 0 writes at parent offset 128");
        assert_eq!(
            pb[128 + 32],
            0xBB,
            "row 1 writes at parent offset 128 + stride"
        );
    }

    #[test]
    #[cfg(unix)]
    fn shm_subview_partitions_parent_buffer() {
        let _lock = crate::tests::fd_lock_shared();
        // One [2,4] u8 shared segment; two [1,4] sub-views at byte offsets 0 and 4.
        if !crate::is_shm_available() {
            crate::test_support::report_skip("shm not available");
            return;
        }
        let parent = Tensor::<u8>::new(&[2, 4], Some(TensorMemory::Shm), None).unwrap();
        let view0 = parent.subview(0, &[1, 4]).expect("shm subview at offset 0");
        let view1 = parent.subview(4, &[1, 4]).expect("shm subview at offset 4");

        view1
            .map()
            .unwrap()
            .as_mut_slice()
            .copy_from_slice(&[10, 20, 30, 40]);
        view0
            .map()
            .unwrap()
            .as_mut_slice()
            .copy_from_slice(&[1, 2, 3, 4]);

        assert_eq!(view0.map().unwrap().as_slice(), &[1, 2, 3, 4]);
        assert_eq!(view1.map().unwrap().as_slice(), &[10, 20, 30, 40]);
        assert_eq!(
            parent.map().unwrap().as_slice(),
            &[1, 2, 3, 4, 10, 20, 30, 40]
        );
        // A sub-view of a sub-view composes the offset.
        let nested = view1.subview(2, &[1, 2]).expect("nested shm subview");
        assert_eq!(nested.map().unwrap().as_slice(), &[30, 40]);
    }

    #[test]
    #[cfg(unix)]
    fn shm_subview_rejects_unaligned_and_oob() {
        let _lock = crate::tests::fd_lock_shared();
        if !crate::is_shm_available() {
            crate::test_support::report_skip("shm not available");
            return;
        }
        // A 2-byte offset is not f32-aligned.
        let parent = Tensor::<f32>::new(&[8], Some(TensorMemory::Shm), None).unwrap();
        assert!(parent.subview(2, &[1]).is_err());
        assert!(parent.subview(4, &[1]).is_ok());
        // Out of bounds: offset 6 + 4 bytes = 10 > 8-byte (u8) segment.
        let p2 = Tensor::<u8>::new(&[8], Some(TensorMemory::Shm), None).unwrap();
        assert!(p2.subview(6, &[4]).is_err());
    }

    #[test]
    #[cfg(target_os = "linux")]
    fn dma_subview_matches_mem_subview() {
        let _lock = crate::tests::fd_lock_shared();
        if !crate::test_support::dma_or_skip("dma_subview_matches_mem_subview") {
            return;
        }
        let dma = Tensor::<u8>::new(&[8], Some(TensorMemory::DmaBuf), None).unwrap();
        let mem = Tensor::<u8>::new(&[8], Some(TensorMemory::Mem), None).unwrap();
        for parent in [&dma, &mem] {
            let v0 = parent.subview(0, &[4]).unwrap();
            let v1 = parent.subview(4, &[4]).unwrap();
            v0.map()
                .unwrap()
                .as_mut_slice()
                .copy_from_slice(&[1, 2, 3, 4]);
            v1.map()
                .unwrap()
                .as_mut_slice()
                .copy_from_slice(&[5, 6, 7, 8]);
            assert_eq!(parent.map().unwrap().as_slice(), &[1, 2, 3, 4, 5, 6, 7, 8]);
        }
    }

    #[test]
    #[cfg(target_os = "linux")]
    fn dma_strided_subview_maps_padded_window() {
        let _lock = crate::tests::fd_lock_shared();
        if !crate::test_support::dma_or_skip("dma_strided_subview_maps_padded_window") {
            return;
        }
        let parent = Tensor::<u8>::new(&[2048], Some(TensorMemory::DmaBuf), None).unwrap();
        let mut view = parent.subview(128, &[8, 16]).unwrap();
        assert_eq!(view.plane_offset(), Some(128));
        view.set_row_stride_unchecked(32); // padded stride (> 16)

        {
            let mut m = view.map().unwrap();
            let s = m.as_mut_slice();
            assert_eq!(s.len(), 256, "strided DMA map exposes stride(32) × rows(8)");
            s[0] = 0xAA; // row 0, col 0
            s[32] = 0xBB; // row 1, col 0 (one stride in)
        }

        let p = parent.map().unwrap();
        let pb = p.as_slice();
        assert_eq!(pb[128], 0xAA, "row 0 writes at parent offset 128");
        assert_eq!(
            pb[128 + 32],
            0xBB,
            "row 1 writes at parent offset 128 + stride"
        );
    }

    #[test]
    #[cfg(target_os = "linux")]
    fn view_single_row_snapshots_parent_stride() {
        let _lock = crate::tests::fd_lock_shared();
        // 8x4 RGBA with a padded 64-byte row stride (tight row = 8*4 = 32).
        if !crate::test_support::dma_or_skip("view_single_row_snapshots_parent_stride") {
            return;
        }
        let parent = Tensor::<u8>::image_with_stride(
            8,
            4,
            PixelFormat::Rgba,
            64,
            Some(TensorMemory::DmaBuf),
            crate::CpuAccess::ReadWrite,
        )
        .unwrap();
        assert_eq!(parent.effective_row_stride(), Some(64));
        // Bottom row (y=3) at x>0.
        let row = parent.view(Region::new(2, 3, 4, 1)).unwrap();
        // The view's own stride is tight (4*4 = 16); the parent pitch (64) is in
        // `view_origin`.
        assert_eq!(row.effective_row_stride(), Some(16));
        let vo = row.view_origin().expect("a view carries a view_origin");
        assert_eq!(
            vo.parent_row_stride, 64,
            "GL keys/pitches a view on the parent stride, not its tight one"
        );
        assert_eq!(row.map().unwrap().as_slice().len(), 16);
    }

    #[test]
    fn test_mem_tensor() {
        let shape = vec![2, 3, 4];
        let tensor =
            MemTensor::<f32>::new(&shape, Some("test_tensor")).expect("Failed to create tensor");
        assert_eq!(tensor.shape(), &shape);
        assert_eq!(tensor.size(), 2 * 3 * 4 * std::mem::size_of::<f32>());
        assert_eq!(tensor.name(), "test_tensor");

        {
            let mut tensor_map = tensor.map().expect("Failed to map memory");
            tensor_map.fill(42.0);
            assert!(tensor_map.iter().all(|&x| x == 42.0));
        }

        let mut tensor = MemTensor::<u8>::new(&shape, None).expect("Failed to create tensor");
        assert_eq!(tensor.shape(), &shape);
        let new_shape = vec![3, 4, 4];
        assert!(
            tensor.reshape(&new_shape).is_err(),
            "Reshape should fail due to size mismatch"
        );
        assert_eq!(tensor.shape(), &shape, "Shape should remain unchanged");

        let new_shape = vec![2, 3, 4];
        tensor.reshape(&new_shape).expect("Reshape should succeed");
        assert_eq!(
            tensor.shape(),
            &new_shape,
            "Shape should be updated after successful reshape"
        );

        {
            let mut tensor_map = tensor.map().expect("Failed to map memory");
            tensor_map.fill(1);
            assert!(tensor_map.iter().all(|&x| x == 1));
        }

        {
            let mut tensor_map = tensor.map().expect("Failed to map memory");
            tensor_map[2] = 42;
            assert_eq!(tensor_map[1], 1, "Value at index 1 should be 1");
            assert_eq!(tensor_map[2], 42, "Value at index 2 should be 42");
        }
    }

    #[test]
    #[cfg(target_os = "linux")]
    fn test_dma_no_fd_leaks() {
        let _lock = fd_lock_exclusive();
        if !crate::test_support::dma_or_skip(function!()) {
            return;
        }

        let baseline = open_fd_count(DMA_FD_TARGETS);

        // Checks that the counter sees a live DMA tensor.
        {
            let probe = Tensor::<u8>::new(&[100, 100], Some(TensorMemory::DmaBuf), None)
                .expect("Failed to create tensor");
            assert!(
                open_fd_count(DMA_FD_TARGETS) > baseline,
                "a live DMA-BUF tensor must be visible to the dmabuf fd counter, \
                 otherwise this test cannot detect a leak"
            );
            drop(probe);
        }

        for _ in 0..100 {
            let tensor = Tensor::<u8>::new(&[100, 100], Some(TensorMemory::DmaBuf), None)
                .expect("Failed to create tensor");
            let mut map = tensor.map().unwrap();
            map.as_mut_slice().fill(233);
        }

        let after = open_fd_count(DMA_FD_TARGETS);
        assert_eq!(
            baseline, after,
            "DMA-BUF file descriptor leak detected: {baseline} -> {after}"
        );
    }

    #[test]
    #[cfg(target_os = "linux")]
    fn test_dma_from_fd_no_fd_leaks() {
        let _lock = fd_lock_exclusive();
        if !crate::test_support::dma_or_skip(function!()) {
            return;
        }

        let baseline = open_fd_count(DMA_FD_TARGETS);
        let orig = Tensor::<u8>::new(&[100, 100], Some(TensorMemory::DmaBuf), None).unwrap();

        // Checks that the counter sees a live DMA tensor.
        let held = open_fd_count(DMA_FD_TARGETS);
        assert!(
            held > baseline,
            "a live DMA-BUF tensor must be visible to the dmabuf fd counter, \
             otherwise this test cannot detect a leak"
        );

        for _ in 0..100 {
            let tensor =
                Tensor::<u8>::from_fd(orig.clone_fd().unwrap(), orig.shape(), None).unwrap();
            assert_eq!(
                tensor.memory(),
                TensorMemory::DmaBuf,
                "DMA-BUF fd must import as Dma, not be silently downgraded"
            );
            let mut map = tensor.map().unwrap();
            map.as_mut_slice().fill(233);
        }

        let after_loop = open_fd_count(DMA_FD_TARGETS);
        assert_eq!(
            held, after_loop,
            "DMA-BUF file descriptor leak detected: {held} -> {after_loop}"
        );

        // The source tensor's fd is released on drop.
        drop(orig);
        let after_drop = open_fd_count(DMA_FD_TARGETS);
        assert_eq!(
            baseline, after_drop,
            "dropping the source tensor must release its fd: {baseline} -> {after_drop}"
        );
    }

    /// `fs_magic` returns the true 32-bit magic for a sign-extended `f_type`.
    #[test]
    #[cfg(target_os = "linux")]
    fn test_fs_magic_normalizes_sign_extended_values() {
        let _lock = crate::tests::fd_lock_shared();
        // Magics whose bit 31 is set.
        const HUGETLBFS_MAGIC: u32 = 0x9584_58f6;
        const F2FS_MAGIC: u32 = 0xf2f5_2010;
        const BTRFS_MAGIC: u32 = 0x9123_683e;

        for magic in [HUGETLBFS_MAGIC, F2FS_MAGIC, BTRFS_MAGIC] {
            // 64-bit gnu: f_type is i64 and holds the true value.
            assert_eq!(fs_magic(i64::from(magic)), magic);

            // 32-bit gnu / uclibc: f_type is i32 and arrives sign-extended once
            // widened.
            let sign_extended = i64::from(magic as i32);
            assert!(sign_extended < 0, "{magic:#x} should have bit 31 set");
            assert_eq!(
                fs_magic(sign_extended),
                magic,
                "{magic:#x} must survive a signed 32-bit f_type"
            );
        }

        // The two magics classified on are below 2^31.
        for magic in [DMA_BUF_MAGIC, TMPFS_MAGIC] {
            assert!(magic < 0x8000_0000);
            assert_eq!(fs_magic(i64::from(magic as i32)), magic);
        }
    }

    /// A DMA-BUF fd imports as [`TensorMemory::DmaBuf`].
    #[test]
    #[cfg(target_os = "linux")]
    fn test_from_fd_dma_imports_as_dma() {
        let _lock = crate::tests::fd_lock_shared();
        if !crate::test_support::dma_or_skip(function!()) {
            return;
        }

        let orig = Tensor::<u8>::new(&[64, 64], Some(TensorMemory::DmaBuf), None).unwrap();
        assert_eq!(orig.memory(), TensorMemory::DmaBuf);

        let imported = Tensor::<u8>::from_fd(orig.clone_fd().unwrap(), orig.shape(), None).unwrap();

        assert_eq!(
            imported.memory(),
            TensorMemory::DmaBuf,
            "a DMA-BUF fd must import as Dma"
        );
    }

    /// A tmpfs/SHM fd imports as [`TensorMemory::Shm`].
    #[test]
    #[cfg(target_os = "linux")]
    fn test_from_fd_shm_imports_as_shm() {
        let _lock = crate::tests::fd_lock_shared();
        if !is_shm_available() {
            crate::test_support::report_skip(&format!(
                "{} - SHM memory not available",
                function!()
            ));
            return;
        }

        let orig = Tensor::<u8>::new(&[64, 64], Some(TensorMemory::Shm), None).unwrap();
        assert_eq!(orig.memory(), TensorMemory::Shm);

        let imported = Tensor::<u8>::from_fd(orig.clone_fd().unwrap(), orig.shape(), None).unwrap();

        assert_eq!(
            imported.memory(),
            TensorMemory::Shm,
            "a tmpfs fd must import as Shm"
        );
    }

    /// An fd that is neither a DMA-BUF nor tmpfs is rejected.
    #[test]
    #[cfg(target_os = "linux")]
    fn test_from_fd_rejects_unknown_filesystem() {
        let _lock = crate::tests::fd_lock_shared();

        let (read_end, _write_end) = nix::unistd::pipe().unwrap();

        let result = Tensor::<u8>::from_fd(read_end, &[64], None);

        match result {
            Err(Error::UnknownBufferType(magic)) => {
                // PIPEFS_MAGIC, from include/uapi/linux/magic.h
                assert_eq!(magic, 0x5049_5045, "expected PIPEFS_MAGIC");
            }
            other => panic!("expected UnknownBufferType for a pipe fd, got {other:?}"),
        }
    }

    #[test]
    #[cfg(target_os = "linux")]
    fn test_shm_no_fd_leaks() {
        let _lock = fd_lock_exclusive();
        if !is_shm_available() {
            crate::test_support::report_skip(&format!(
                "{} - SHM memory allocation not available (permission denied or no SHM support)",
                function!()
            ));
            return;
        }

        let baseline = open_fd_count(SHM_FD_TARGETS);

        // Checks that the counter sees a live shm tensor.
        {
            let probe = Tensor::<u8>::new(&[100, 100], Some(TensorMemory::Shm), None)
                .expect("Failed to create tensor");
            assert!(
                open_fd_count(SHM_FD_TARGETS) > baseline,
                "a live shm tensor must be visible to the shm fd counter, \
                 otherwise this test cannot detect a leak"
            );
            drop(probe);
        }

        for _ in 0..100 {
            let tensor = Tensor::<u8>::new(&[100, 100], Some(TensorMemory::Shm), None)
                .expect("Failed to create tensor");
            let mut map = tensor.map().unwrap();
            map.as_mut_slice().fill(233);
        }

        let after = open_fd_count(SHM_FD_TARGETS);
        assert_eq!(
            baseline, after,
            "shm file descriptor leak detected: {baseline} -> {after}"
        );
    }

    #[test]
    #[cfg(target_os = "linux")]
    fn test_shm_from_fd_no_fd_leaks() {
        let _lock = fd_lock_exclusive();
        if !is_shm_available() {
            crate::test_support::report_skip(&format!(
                "{} - SHM memory allocation not available (permission denied or no SHM support)",
                function!()
            ));
            return;
        }

        let baseline = open_fd_count(SHM_FD_TARGETS);
        let orig = Tensor::<u8>::new(&[100, 100], Some(TensorMemory::Shm), None).unwrap();

        // Checks that the counter sees a live shm tensor.
        let held = open_fd_count(SHM_FD_TARGETS);
        assert!(
            held > baseline,
            "a live shm tensor must be visible to the shm fd counter, \
             otherwise this test cannot detect a leak"
        );

        for _ in 0..100 {
            let tensor =
                Tensor::<u8>::from_fd(orig.clone_fd().unwrap(), orig.shape(), None).unwrap();
            let mut map = tensor.map().unwrap();
            map.as_mut_slice().fill(233);
        }

        let after_loop = open_fd_count(SHM_FD_TARGETS);
        assert_eq!(
            held, after_loop,
            "shm file descriptor leak detected: {held} -> {after_loop}"
        );

        // The source tensor's fd is released on drop.
        drop(orig);
        let after_drop = open_fd_count(SHM_FD_TARGETS);
        assert_eq!(
            baseline, after_drop,
            "dropping the source tensor must release its fd: {baseline} -> {after_drop}"
        );
    }

    #[cfg(feature = "ndarray")]
    #[test]
    fn test_ndarray() {
        let _lock = crate::tests::fd_lock_shared();
        let shape = vec![2, 3, 4];
        let tensor = Tensor::<f32>::new(&shape, None, None).expect("Failed to create tensor");

        let mut tensor_map = tensor.map().expect("Failed to map tensor memory");
        tensor_map.fill(1.0);

        let view = tensor_map.view().expect("Failed to get ndarray view");
        assert_eq!(view.shape(), &[2, 3, 4]);
        assert!(view.iter().all(|&x| x == 1.0));

        let mut view_mut = tensor_map
            .view_mut()
            .expect("Failed to get mutable ndarray view");
        view_mut[[0, 0, 0]] = 42.0;
        assert_eq!(view_mut[[0, 0, 0]], 42.0);
        assert_eq!(tensor_map[0], 42.0, "Value at index 0 should be 42");
    }

    #[test]
    fn test_buffer_identity_derived_equality() {
        // Two identities derived from the same kind and key are equal.
        let id1 = BufferIdentity::derived(IdentityKind::DmaBuf, 4242);
        let id2 = BufferIdentity::derived(IdentityKind::DmaBuf, 4242);
        assert_eq!(
            id1.id(),
            id2.id(),
            "Identities derived from the same kind and key should match"
        );

        // A different key yields a different id.
        let id3 = BufferIdentity::derived(IdentityKind::DmaBuf, 4243);
        assert_ne!(
            id1.id(),
            id3.id(),
            "Identities derived from different keys should differ"
        );
    }

    #[test]
    fn test_buffer_identity_clone_shares_guard() {
        let id1 = BufferIdentity::derived(IdentityKind::DmaBuf, 4242);
        let weak = id1.weak();
        assert!(
            weak.upgrade().is_some(),
            "Weak should be alive while original exists"
        );

        let id2 = id1.clone();
        assert_eq!(id1.id(), id2.id(), "Cloned identity should have same id");

        drop(id1);
        assert!(
            weak.upgrade().is_some(),
            "Weak should still be alive (clone holds Arc)"
        );

        drop(id2);
        assert!(
            weak.upgrade().is_none(),
            "Weak should be dead after all clones dropped"
        );
    }

    #[test]
    fn test_tensor_buffer_identity() {
        let t1 = Tensor::<u8>::new(&[100], Some(TensorMemory::Mem), Some("t1")).unwrap();
        let t2 = Tensor::<u8>::new(&[100], Some(TensorMemory::Mem), Some("t2")).unwrap();
        assert_ne!(
            t1.buffer_identity().id(),
            t2.buffer_identity().id(),
            "Different tensors should have different buffer ids"
        );
    }

    // ------------------------------------------------------------------------
    // Quantization — constructor validation + accessor correctness.
    // ------------------------------------------------------------------------

    #[test]
    fn test_quantization_per_tensor_constructors() {
        let q = Quantization::per_tensor(0.1, -5);
        assert!(q.is_per_tensor());
        assert!(!q.is_per_channel());
        assert!(!q.is_symmetric());
        assert_eq!(q.scale(), &[0.1]);
        assert_eq!(q.zero_point(), Some(&[-5][..]));

        let qs = Quantization::per_tensor_symmetric(0.05);
        assert!(qs.is_per_tensor());
        assert!(qs.is_symmetric());
        assert_eq!(qs.zero_point(), None);
    }

    #[test]
    fn test_quantization_per_channel_constructors() {
        let q = Quantization::per_channel(vec![0.1, 0.2, 0.3], vec![0, -1, 1], 2).unwrap();
        assert!(q.is_per_channel());
        assert!(!q.is_symmetric());
        assert_eq!(q.axis(), Some(2));
        assert_eq!(q.scale().len(), 3);

        let qs = Quantization::per_channel_symmetric(vec![0.054, 0.089, 0.195], 0).unwrap();
        assert!(qs.is_per_channel());
        assert!(qs.is_symmetric());
        assert_eq!(qs.axis(), Some(0));
    }

    #[test]
    fn test_quantization_per_channel_length_mismatch_rejected() {
        // len(scales) != len(zero_points) is rejected at construction.
        let err = Quantization::per_channel(vec![0.1, 0.2], vec![0, 0, 0], 0).unwrap_err();
        assert!(matches!(err, Error::QuantizationInvalid { .. }));
    }

    #[test]
    fn test_quantization_per_channel_empty_rejected() {
        let err = Quantization::per_channel_symmetric(vec![], 0).unwrap_err();
        assert!(matches!(err, Error::QuantizationInvalid { .. }));
    }

    /// `set_quantization` rejects a malformed deserialized `Quantization`.
    #[test]
    fn test_quantization_validate_rejects_malformed_deserialize() {
        let mut t = Tensor::<i8>::new(&[1, 1, 4], Some(TensorMemory::Mem), None).unwrap();

        // Empty scale array.
        let q: Quantization = serde_json::from_str(r#"{"scale": []}"#).unwrap();
        assert!(matches!(
            t.set_quantization(q).unwrap_err(),
            Error::QuantizationInvalid { .. }
        ));

        // Per-tensor with multi-element zero_point.
        let q: Quantization =
            serde_json::from_str(r#"{"scale": 0.1, "zero_point": [0, 0, 0]}"#).unwrap();
        assert!(matches!(
            t.set_quantization(q).unwrap_err(),
            Error::QuantizationInvalid { .. }
        ));

        // Per-channel zero_point length != scale length.
        let q: Quantization = serde_json::from_str(
            r#"{"scale": [0.1, 0.2, 0.3, 0.4], "zero_point": [0, 0], "axis": 2}"#,
        )
        .unwrap();
        assert!(matches!(
            t.set_quantization(q).unwrap_err(),
            Error::QuantizationInvalid { .. }
        ));
    }

    #[test]
    fn test_quantization_mode_dispatch() {
        let pt = Quantization::per_tensor(0.1, -5);
        assert!(matches!(
            pt.mode(),
            QuantMode::PerTensor { scale, zero_point } if scale == 0.1 && zero_point == -5
        ));

        let pts = Quantization::per_tensor_symmetric(0.05);
        assert!(matches!(
            pts.mode(),
            QuantMode::PerTensorSymmetric { scale } if scale == 0.05
        ));

        let pc = Quantization::per_channel(vec![0.1, 0.2], vec![0, -1], 2).unwrap();
        assert!(matches!(pc.mode(), QuantMode::PerChannel { axis: 2, .. }));

        let pcs = Quantization::per_channel_symmetric(vec![0.1, 0.2], 0).unwrap();
        assert!(matches!(
            pcs.mode(),
            QuantMode::PerChannelSymmetric { axis: 0, .. }
        ));
    }

    #[test]
    fn test_tensor_quantization_roundtrip_integer() {
        let mut t = Tensor::<i8>::new(&[2, 3, 4], Some(TensorMemory::Mem), None).unwrap();
        assert!(t.quantization().is_none());
        t.set_quantization(Quantization::per_tensor(0.1, -5))
            .unwrap();
        let q = t.quantization().unwrap();
        assert_eq!(q.scale(), &[0.1]);
        t.clear_quantization();
        assert!(t.quantization().is_none());
    }

    #[test]
    fn test_tensor_with_quantization_builder() {
        let t = Tensor::<i8>::new(&[4, 4], Some(TensorMemory::Mem), None)
            .unwrap()
            .with_quantization(Quantization::per_tensor_symmetric(0.05))
            .unwrap();
        assert!(t.quantization().is_some());
    }

    #[test]
    fn test_tensor_dyn_quantization_float_arm_returns_none() {
        let t = Tensor::<f32>::new(&[2, 2], Some(TensorMemory::Mem), None).unwrap();
        let td = TensorDyn::F32(t);
        assert!(td.quantization().is_none());
    }

    #[test]
    fn test_tensor_dyn_set_quantization_float_arm_errors() {
        let t = Tensor::<f32>::new(&[2, 2], Some(TensorMemory::Mem), None).unwrap();
        let mut td = TensorDyn::F32(t);
        let err = td
            .set_quantization(Quantization::per_tensor(0.1, 0))
            .unwrap_err();
        assert!(matches!(err, Error::QuantizationInvalid { .. }));
    }

    /// `Tensor::<f32>::quantization()` fails to compile: `f32` is not
    /// `IntegerType`.
    ///
    /// ```compile_fail
    /// use edgefirst_tensor::{Tensor, TensorMemory};
    /// let t = Tensor::<f32>::new(&[2, 2], Some(TensorMemory::Mem), None).unwrap();
    /// let _ = t.quantization(); // compile error: f32 not IntegerType
    /// ```
    fn _compile_fail_doctest_anchor() {}

    // Held exclusively by tests that count fds and shared by tests that open or
    // close fds.
    pub static FD_LOCK: RwLock<()> = RwLock::new(());

    /// Take [`FD_LOCK`] shared, tolerating poisoning.
    pub fn fd_lock_shared() -> std::sync::RwLockReadGuard<'static, ()> {
        FD_LOCK.read().unwrap_or_else(PoisonError::into_inner)
    }

    /// Take [`FD_LOCK`] exclusively, tolerating poisoning.
    #[cfg(target_os = "linux")]
    pub fn fd_lock_exclusive() -> std::sync::RwLockWriteGuard<'static, ()> {
        FD_LOCK.write().unwrap_or_else(PoisonError::into_inner)
    }

    /// `/proc` fd-target fragments identifying a POSIX shared memory segment.
    #[cfg(target_os = "linux")]
    const SHM_FD_TARGETS: &[&str] = &["/dev/shm/"];

    /// `/proc` fd-target fragments identifying the fds a DMA tensor owns: the
    /// exported buffer and the heap it came from.
    #[cfg(target_os = "linux")]
    const DMA_FD_TARGETS: &[&str] = &["dmabuf", "dma_heap"];

    /// How many of this process's open fds point at an object whose `/proc`
    /// target contains one of `needles`.
    #[cfg(target_os = "linux")]
    fn open_fd_count(needles: &[&str]) -> usize {
        let proc = procfs::process::Process::myself()
            .expect("Failed to get current process using /proc/self");
        proc.fd()
            .expect("Failed to list open file descriptors")
            .filter_map(Result::ok)
            .filter(|entry| {
                let target = format!("{:?}", entry.target);
                needles.iter().any(|needle| target.contains(needle))
            })
            .count()
    }

    /// DMA is not available on non-Linux platforms.
    #[test]
    #[cfg(not(target_os = "linux"))]
    fn test_dma_not_available_on_non_linux() {
        assert!(
            !is_dma_available(),
            "DMA memory allocation should NOT be available on non-Linux platforms"
        );
    }

    #[test]
    fn colorimetry_defaults_none_and_roundtrips_without_auto_fill() {
        use crate::{ColorEncoding, ColorRange, Colorimetry, PixelFormat, TensorMemory};
        let mut t = Tensor::<u8>::image(
            1280,
            720,
            PixelFormat::Nv12,
            Some(TensorMemory::Mem),
            crate::CpuAccess::ReadWrite,
        )
        .unwrap();
        assert_eq!(t.colorimetry(), None); // default undefined
        let c = Colorimetry::default()
            .with_encoding(ColorEncoding::Bt709)
            .with_range(ColorRange::Limited);
        t.set_colorimetry(Some(c));
        assert_eq!(t.colorimetry(), Some(c));
        // configure_image leaves colorimetry unchanged.
        t.configure_image(640, 480, PixelFormat::Grey).unwrap();
        assert_eq!(t.colorimetry(), Some(c));
    }

    #[test]
    fn configure_image_within_capacity() {
        let mut t = Tensor::<u8>::image_with_capacity(
            640,
            480,
            PixelFormat::Rgb,
            None,
            crate::CpuAccess::ReadWrite,
        )
        .unwrap();
        t.configure_image(320, 240, PixelFormat::Nv12).unwrap();
        assert_eq!(t.format(), Some(PixelFormat::Nv12));
        assert_eq!(t.width(), Some(320));
        assert_eq!(t.height(), Some(240));
        assert_eq!(t.shape(), &[360, 320]); // 240*3/2
    }

    #[test]
    fn configure_image_too_large_errors() {
        let mut t = Tensor::<u8>::image_with_capacity(
            64,
            64,
            PixelFormat::Grey,
            None,
            crate::CpuAccess::ReadWrite,
        )
        .unwrap();
        let err = t
            .configure_image(1920, 1080, PixelFormat::Nv12)
            .unwrap_err();
        assert!(matches!(err, Error::InsufficientCapacity { .. }));
    }

    /// A reused IOSurface pool keeps its physical `bytesPerRow` when
    /// reconfigured to a smaller logical image.
    #[test]
    #[cfg(target_os = "macos")]
    fn configure_image_preserves_iosurface_physical_stride() {
        let _lock = crate::tests::fd_lock_shared();
        // Pool: GREY/R8 IOSurface 100 wide → bytesPerRow padded to 128.
        let mut pool = Tensor::<u8>::image(
            100,
            64,
            PixelFormat::Grey,
            Some(TensorMemory::DmaBuf),
            crate::CpuAccess::ReadWrite,
        )
        .unwrap();
        let pitch = pool.effective_row_stride().unwrap();
        assert!(
            pitch >= 128 && pitch.is_multiple_of(64),
            "padded bytesPerRow, got {pitch}"
        );

        // Reconfigure to a smaller NV12 frame; the physical pitch is kept.
        pool.configure_image(32, 16, PixelFormat::Nv12).unwrap();
        assert_eq!(pool.format(), Some(PixelFormat::Nv12));
        assert_eq!(pool.width(), Some(32));
        assert_eq!(pool.height(), Some(16));
        assert_eq!(
            pool.effective_row_stride(),
            Some(pitch),
            "configure_image must preserve the IOSurface physical bytesPerRow"
        );

        // Reconfigure to NV24; the pitch is kept.
        pool.configure_image(32, 16, PixelFormat::Nv24).unwrap();
        assert_eq!(pool.effective_row_stride(), Some(pitch));
    }

    /// `configure_image` on a Mem backing reconfigures to the format's
    /// 64-byte-aligned row stride.
    #[test]
    fn configure_image_mem_aligns_stride() {
        let mut t = Tensor::<u8>::image_with_capacity(
            64,
            64,
            PixelFormat::Rgba,
            Some(TensorMemory::Mem),
            crate::CpuAccess::ReadWrite,
        )
        .unwrap();
        t.configure_image(32, 16, PixelFormat::Nv12).unwrap();
        let s = t.effective_row_stride().unwrap();
        assert_eq!(s % 64, 0, "stride must be 64-aligned");
        assert!(s >= 32, "stride must cover the even-width minimum");
        assert_eq!(s, 64);
    }

    /// Host memory, plus the hardware slot where the host can allocate a
    /// semi-planar image in one. Android allocates none.
    fn semi_planar_pitch_memories(test: &str) -> Vec<TensorMemory> {
        let mut memories = vec![TensorMemory::Mem];
        #[cfg(unix)]
        memories.push(TensorMemory::Shm);
        let hardware = if cfg!(target_os = "android") {
            false
        } else if cfg!(target_os = "linux") {
            crate::test_support::dma_or_skip(test)
        } else {
            is_gpu_buffer_available()
        };
        if hardware {
            memories.push(TensorMemory::DmaBuf);
        }
        memories
    }

    /// Asserts `t` records a row stride of at least `even(width)` and maps
    /// `stride × rows` bytes.
    fn assert_semi_planar_grid(t: &Tensor<u8>, what: &str) {
        let width = t.width().unwrap();
        let rows = t.shape()[0];
        let stride = t
            .row_stride()
            .unwrap_or_else(|| panic!("{what}: a semi-planar tensor must record its row stride"));
        assert!(
            stride >= width.next_multiple_of(2),
            "{what}: stride {stride} splits a chroma pair at width {width}"
        );
        let map = t.map().unwrap_or_else(|e| panic!("{what}: map: {e}"));
        assert_eq!(
            map.as_slice().len(),
            stride * rows,
            "{what}: the map must span every pitched row"
        );
    }

    const SEMI_PLANAR_FORMATS: [PixelFormat; 3] =
        [PixelFormat::Nv12, PixelFormat::Nv16, PixelFormat::Nv24];

    #[test]
    fn semi_planar_image_maps_its_whole_pitched_grid() {
        let _lock = crate::tests::fd_lock_shared();
        for memory in semi_planar_pitch_memories("semi_planar_image_maps_its_whole_pitched_grid") {
            for format in SEMI_PLANAR_FORMATS {
                for width in [63, 64, 65, 127, 639, 640, 641] {
                    let what = format!("{memory:?} {format:?} {width}x16");
                    let t = Tensor::<u8>::image(
                        width,
                        16,
                        format,
                        Some(memory),
                        crate::CpuAccess::ReadWrite,
                    )
                    .unwrap_or_else(|e| panic!("{what}: {e}"));
                    assert_eq!(t.width(), Some(width), "{what}: logical width");
                    assert_semi_planar_grid(&t, &what);
                }
            }
        }
    }

    #[test]
    fn configure_image_semi_planar_maps_its_whole_pitched_grid() {
        let _lock = crate::tests::fd_lock_shared();
        for memory in
            semi_planar_pitch_memories("configure_image_semi_planar_maps_its_whole_pitched_grid")
        {
            for format in SEMI_PLANAR_FORMATS {
                for width in [639, 640] {
                    let what = format!("{memory:?} 640x48 Nv24 pool -> {format:?} {width}x16");
                    let mut t = Tensor::<u8>::image(
                        640,
                        48,
                        PixelFormat::Nv24,
                        Some(memory),
                        crate::CpuAccess::ReadWrite,
                    )
                    .unwrap_or_else(|e| panic!("{what}: {e}"));
                    t.configure_image(width, 16, format)
                        .unwrap_or_else(|e| panic!("{what}: {e}"));
                    assert_eq!(t.width(), Some(width), "{what}: logical width");
                    assert_semi_planar_grid(&t, &what);
                }
            }
        }
    }

    /// `set_logical_shape` adopts the backing's pitch for a semi-planar shape
    /// on a tensor that arrives with no stride recorded.
    #[test]
    #[cfg(target_os = "macos")]
    fn set_logical_shape_semi_planar_maps_its_whole_pitched_grid() {
        let _lock = crate::tests::fd_lock_shared();
        let mut t = Tensor::<u8>::image(
            640,
            24,
            PixelFormat::Grey,
            Some(TensorMemory::DmaBuf),
            crate::CpuAccess::ReadWrite,
        )
        .unwrap();
        assert_eq!(t.row_stride(), None, "a 64-aligned grey row is flat");
        t.reshape(&[24, 640]).unwrap();
        t.set_format(PixelFormat::Nv12).unwrap();
        assert_eq!(t.row_stride(), None, "set_format records no stride");

        t.set_logical_shape(&[24, 639]).unwrap();
        assert_eq!(t.width(), Some(639));
        assert_semi_planar_grid(&t, "IOSurface Grey 640x24 -> Nv12 [24, 639]");
    }

    #[test]
    fn strided_mem_tensor_cpu_maps_full_padded_buffer() {
        // A packed RGBA image with row padding: logical width 8 px (32 B/row) and
        // a 48-byte row stride.
        let mut t = Tensor::<u8>::image_with_capacity(
            16,
            3,
            PixelFormat::Rgba,
            Some(TensorMemory::Mem),
            crate::CpuAccess::ReadWrite,
        )
        .unwrap(); // capacity 3 × 16 × 4 = 192 B
        t.configure_image(8, 3, PixelFormat::Rgba).unwrap(); // logical [3, 8, 4] = 96 B
        t.set_row_stride(48).unwrap(); // padded stride (>= 32 B min)

        let map = t.map().expect("strided Mem tensor should CPU-map");
        // The full padded buffer: stride 48 × 3 rows = 144 B.
        assert_eq!(map.as_slice().len(), 144);
        assert_eq!(map.shape(), &[3, 8, 4]);
    }

    #[test]
    fn strided_mem_tensor_over_capacity_errors() {
        // Stride larger than the allocation: 64 B × 3 rows = 192 B > 96 B cap.
        let mut t = Tensor::<u8>::new(&[3, 8, 4], Some(TensorMemory::Mem), None).unwrap();
        t.set_format(PixelFormat::Rgba).unwrap();
        t.set_row_stride(64).unwrap();
        assert!(matches!(t.map(), Err(Error::InsufficientCapacity { .. })));
    }

    /// SHM memory allocation is available and usable on Unix systems.
    #[test]
    #[cfg(unix)]
    fn test_shm_available_and_usable() {
        let _lock = crate::tests::fd_lock_shared();
        if !is_shm_available() {
            crate::test_support::report_skip(
                "test_shm_available_and_usable - SHM allocation \
                 unavailable on this platform",
            );
            return;
        }

        let tensor = Tensor::<u8>::new(&[100, 100], Some(TensorMemory::Shm), None)
            .expect("Failed to create SHM tensor");

        let mut map = tensor.map().expect("Failed to map SHM tensor");
        map.as_mut_slice().fill(0xAB);

        assert!(
            map.as_slice().iter().all(|&b| b == 0xAB),
            "SHM tensor data should be writable and readable"
        );
    }

    // =========================================================================
    // packed_rgba16f_layout geometry
    // =========================================================================

    #[test]
    fn packed_rgba16f_layout_planar_rgb_f16() {
        let layout =
            packed_rgba16f_layout(PixelFormat::PlanarRgb, DType::F16, 640, 640).expect("Some");
        assert_eq!(layout.surface_w, 160);
        assert_eq!(layout.surface_h, 1920);
        assert_eq!(layout.bytes_per_texel, 8);
        assert_eq!(layout.pitch, 1280);
    }

    #[test]
    fn packed_rgba16f_layout_planar_rgba_f16() {
        let layout =
            packed_rgba16f_layout(PixelFormat::PlanarRgba, DType::F16, 640, 640).expect("Some");
        assert_eq!(layout.surface_w, 160);
        assert_eq!(layout.surface_h, 2560); // 4 planes
        assert_eq!(layout.bytes_per_texel, 8);
        assert_eq!(layout.pitch, 1280);
    }

    #[test]
    fn packed_rgba16f_layout_rejects_misaligned() {
        assert!(packed_rgba16f_layout(PixelFormat::PlanarRgb, DType::F16, 642, 640).is_none());
    }

    #[test]
    fn packed_rgba16f_layout_rejects_non_f16() {
        // Non-F16 dtype with planar RGB
        assert!(packed_rgba16f_layout(PixelFormat::PlanarRgb, DType::U8, 640, 640).is_none());
        // Non-planar format with F32
        assert!(packed_rgba16f_layout(PixelFormat::Rgb, DType::F32, 640, 640).is_none());
        // Packed Rgba with F16
        assert!(packed_rgba16f_layout(PixelFormat::Rgba, DType::F16, 640, 640).is_none());
    }

    #[test]
    fn cuda_map_fast_fails_to_none_without_handle() {
        let t = Tensor::<f32>::new(&[4], Some(TensorMemory::Mem), None).unwrap();
        assert!(t.cuda().is_none());
        assert!(t.cuda_map().is_none());
    }

    #[test]
    fn cuda_returns_none_without_handle() {
        let t = Tensor::<f32>::new(&[2, 2], Some(TensorMemory::Mem), None).unwrap();
        assert!(t.cuda().is_none(), "no CUDA handle on a Mem tensor");
        assert!(t.cuda_map().is_none(), "fast-fail map → None");
    }

    #[test]
    fn cuda_map_then_host_map_fallback() {
        // Try cuda_map() first; when it is None, fall back to map().
        let t = Tensor::<f32>::new(&[2, 2], Some(TensorMemory::Mem), None).unwrap();
        let cuda = t.cuda_map();
        if let Some(_c) = cuda {
            unreachable!("a Mem tensor has no CUDA handle");
        } else {
            let host = t.map().expect("host map fallback must succeed");
            // len() is the element count.
            assert_eq!(host.len(), 4); // 2*2 f32 elements
        }
    }

    // -------------------------------------------------------------------------
    // Tensor::from_foreign
    // -------------------------------------------------------------------------

    #[test]
    fn from_foreign_valid_wrap_and_readback() {
        let mut buf: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let ptr = buf.as_mut_ptr();
        let t = unsafe { Tensor::<f32>::from_foreign(ptr, &[2, 3], None, Some("test_foreign")) }
            .expect("valid from_foreign must succeed");
        assert_eq!(t.shape(), &[2, 3]);
        assert_eq!(t.memory(), TensorMemory::Mem);
        assert_eq!(t.name(), "test_foreign");
        let m = t.map().unwrap();
        assert_eq!(m.as_slice(), &[1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0]);
    }

    #[test]
    fn from_foreign_write_visible_in_caller_allocation() {
        let mut buf: Vec<u8> = vec![0u8; 6];
        let ptr = buf.as_mut_ptr();
        let t = unsafe { Tensor::<u8>::from_foreign(ptr, &[2, 3], None, None) }.unwrap();
        {
            let mut m = t.map().unwrap();
            m.as_mut_slice().copy_from_slice(&[10, 20, 30, 40, 50, 60]);
        }
        drop(t);
        assert_eq!(buf, vec![10, 20, 30, 40, 50, 60]);
    }

    #[test]
    fn from_foreign_rejects_null_ptr() {
        let err = unsafe { Tensor::<u8>::from_foreign(std::ptr::null_mut(), &[4], None, None) }
            .unwrap_err();
        assert!(
            matches!(err, Error::InvalidArgument(ref m) if m.contains("non-null")),
            "expected InvalidArgument(non-null), got {err:?}"
        );
    }

    #[test]
    fn from_foreign_rejects_empty_shape() {
        let mut dummy: u8 = 0;
        let err = unsafe { Tensor::<u8>::from_foreign(&mut dummy, &[], None, None) }.unwrap_err();
        assert!(
            matches!(err, Error::InvalidSize(0)),
            "expected InvalidSize(0) for empty shape, got {err:?}"
        );
    }

    #[test]
    fn from_foreign_rejects_overflow_shape() {
        // Two dimensions whose product overflows usize.
        let mut dummy: u8 = 0;
        let huge = [usize::MAX / 2 + 1, 2];
        let err = unsafe { Tensor::<u8>::from_foreign(&mut dummy, &huge, None, None) }.unwrap_err();
        assert!(
            matches!(err, Error::InvalidArgument(ref m) if m.contains("overflow")),
            "expected InvalidArgument(overflow), got {err:?}"
        );
    }

    #[test]
    fn from_foreign_owner_keeps_allocation_alive() {
        use std::sync::atomic::{AtomicBool, Ordering};
        let flag = std::sync::Arc::new(AtomicBool::new(false));
        let flag2 = flag.clone();
        struct Guard(std::sync::Arc<AtomicBool>);
        impl Drop for Guard {
            fn drop(&mut self) {
                self.0.store(true, Ordering::SeqCst);
            }
        }
        let mut buf: Vec<u32> = vec![42u32; 4];
        let ptr = buf.as_mut_ptr();
        let owner: ForeignOwner = Box::new(Guard(flag2));
        let t = unsafe { Tensor::<u32>::from_foreign(ptr, &[4], Some(owner), None) }.unwrap();
        // The map co-owns the backing Arc.
        let m = t.map().unwrap();
        assert_eq!(m.as_slice()[0], 42);
        drop(t); // tensor dropped while map is still live
        assert!(
            !flag.load(Ordering::SeqCst),
            "owner must not drop while a map shares the backing"
        );
        drop(m);
        assert!(
            flag.load(Ordering::SeqCst),
            "owner Drop must fire when the last Arc reference is released"
        );
    }
}
