// SPDX-FileCopyrightText: Copyright 2025 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

/*!
EdgeFirst HAL - Tensor Module

The `edgefirst_tensor` crate provides a unified interface for managing multi-dimensional arrays (tensors)
with support for different memory types, including Direct Memory Access (DMA), POSIX Shared Memory (Shm),
and system memory. The crate defines traits and structures for creating, reshaping, and mapping tensors into memory.

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
The main structures and traits provided by the `edgefirst_tensor` crate are `TensorTrait` and `TensorMapTrait`,
which define the behavior of Tensors and their memory mappings, respectively.
The `Tensor<T>` struct wraps a backend-specific storage with optional image format metadata (`PixelFormat`),
while the `TensorMap` enum provides access to the underlying data. The `TensorDyn` type-erased enum
wraps `Tensor<T>` for runtime element-type dispatch.
 */
pub mod colorimetry;
pub mod covguard;
mod cuda;
#[cfg(target_os = "linux")]
mod dma;
#[cfg(target_os = "linux")]
mod dmabuf;
mod error;
mod format;
#[cfg(target_os = "macos")]
mod iosurface;
mod mem;
mod pbo;
#[cfg(unix)]
mod shm;
mod tensor_dyn;
pub use colorimetry::{ColorEncoding, ColorRange, ColorSpace, ColorTransfer, Colorimetry};

/// Retained constructor: installs the coverage flush-on-abort handler for this
/// crate's instrumented test binary. See `covguard`. Only present under
/// coverage on Linux (`.init_array` is ELF-only; the i.MX flush is Linux-only).
#[cfg(all(coverage, target_os = "linux"))]
#[used]
#[link_section = ".init_array"]
static __EDGEFIRST_COV_INSTALL: extern "C" fn() = {
    extern "C" fn ctor() {
        crate::covguard::install();
    }
    ctor
};

// Backing tensor/map types are internal implementation details: callers
// allocate `Tensor<T>` / `TensorDyn` and map them, never naming the per-memory
// backing types directly. They are `pub(crate)` so they stay nameable for the
// `TensorStorage` / `TensorMap` enums without leaking into the public API.
// Exceptions kept public: `Pbo*` is a GL extension point implemented by the
// image crate, and `image_iosurface_layout` is a public helper.
#[cfg(target_os = "linux")]
pub(crate) use crate::dma::{DmaMap, DmaTensor};
#[cfg(target_os = "macos")]
pub use crate::iosurface::image_iosurface_layout;
#[cfg(target_os = "macos")]
pub(crate) use crate::iosurface::{IoSurfaceMap, IoSurfaceTensor};
pub(crate) use crate::mem::{MemMap, MemTensor};
pub use crate::pbo::{PboMap, PboMapping, PboOps, PboTensor};
#[cfg(unix)]
pub(crate) use crate::shm::{ShmMap, ShmTensor};
pub use cuda::{
    gl_map_resource, gl_register_buffer, gl_unmap_resource, gl_unregister_resource,
    is_cuda_available, memcpy_device_to_host, CudaGlOps, CudaHandle, CudaMap,
};
pub use error::{Error, Result};
pub use format::{ChromaLayout, PixelFormat, PixelLayout};
use num_traits::Num;
use serde::{Deserialize, Serialize};
#[cfg(unix)]
use std::os::fd::OwnedFd;
use std::{
    fmt,
    ops::{Deref, DerefMut},
    sync::{
        atomic::{AtomicU64, Ordering},
        Arc, Weak,
    },
};
pub use tensor_dyn::TensorDyn;

/// Re-export of `half::f16` so downstream crates can write
/// `Tensor::<edgefirst_tensor::f16>::from_iosurface(…)` without
/// adding `half` to their own dependency list. The version stays in
/// lockstep with the `half` workspace dep.
pub use half::f16;

// =============================================================================
// RGBA16F packed-layout geometry — single source of truth
//
// A `PlanarRgb` [3,H,W] or `PlanarRgba` [4,H,W] f16 tensor is represented
// on the GPU as an RGBA16F surface (the only float format accepted by the
// ANGLE IOSurface extension). Four contiguous f16 elements are packed into
// each 8-byte RGBA16F texel, yielding a `(W/4, C*H)` surface.
//
// All call sites that need these dimensions must use `packed_rgba16f_layout`
// so the rule lives in exactly one place. Currently consumed by:
//  - `crates/tensor/src/iosurface.rs` `new_image` (macOS IOSurface alloc)
//  - `crates/image/src/gl/iosurface_import.rs` (macOS GL IOSurface import)
//  - `crates/image/src/gl/processor/float.rs` (Linux GL float render — PBO
//    readback and DMA-BUF, also via the `dma_f16_packed_layout` wrapper)
// =============================================================================

/// Geometry of the RGBA16F-packed surface backing a planar F16 image tensor.
///
/// ANGLE only supports one float `(type, internal_format)` pair for IOSurface
/// import: `(GL_HALF_FLOAT, GL_RGBA)` = RGBA16F (8 bytes/texel). To map a
/// `[C, H, W]` f16 planar tensor onto such a surface, 4 contiguous f16
/// elements are packed into each RGBA16F texel, yielding a surface of
/// `(W/4, C*H)` texels at 8 bytes/texel. The byte stream is identical to a
/// (nonexistent) R16F `(W, C*H)` surface and can be consumed as `&[f16]`
/// with shape `[1, C, H, W]` without rearrangement.
///
/// Obtain via [`packed_rgba16f_layout`] — never construct directly.
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

/// Canonical geometry for the RGBA16F-packed surface backing a planar F16
/// image tensor.
///
/// Returns `Some(layout)` only when **all** of the following hold:
///
/// - `dtype == DType::F16`
/// - `format` is `PixelFormat::PlanarRgb` (3 planes) or
///   `PixelFormat::PlanarRgba` (4 planes)
/// - `width % 4 == 0`
///
/// Returns `None` for any other `(format, dtype)` combination, misaligned
/// width, or when the surface geometry would overflow `usize` — callers
/// must fall back to a non-packed path or return a context-appropriate
/// error.
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
    // Checked arithmetic: a degenerate (height, width) could otherwise wrap
    // and yield an under-sized layout, which downstream allocators trust for
    // GPU/CPU buffer sizing. Overflow → None (handled like any other
    // unsupported geometry).
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

/// Per-plane DMA-BUF descriptor for external buffer import.
///
/// Owns a duplicated file descriptor plus optional stride and offset metadata.
/// The fd is duplicated eagerly in [`new()`](Self::new) so that a bad fd is
/// caught immediately. `import_image` consumes the descriptor and takes
/// ownership of the duped fd — no further cleanup is needed by the caller.
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
    /// The fd is duped immediately — a bad fd fails here rather than inside
    /// `import_image`. The caller retains ownership of the original fd.
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

/// Element type discriminant for runtime type identification.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[repr(u8)]
#[non_exhaustive]
pub enum DType {
    U8,
    I8,
    U16,
    I16,
    U32,
    I32,
    U64,
    I64,
    F16,
    F32,
    F64,
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

    /// Short type name (e.g., "u8", "f32", "f16").
    pub const fn name(&self) -> &'static str {
        match self {
            Self::U8 => "u8",
            Self::I8 => "i8",
            Self::U16 => "u16",
            Self::I16 => "i16",
            Self::U32 => "u32",
            Self::I32 => "i32",
            Self::U64 => "u64",
            Self::I64 => "i64",
            Self::F16 => "f16",
            Self::F32 => "f32",
            Self::F64 => "f64",
        }
    }
}

impl fmt::Display for DType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

/// Map a static numeric type `T` to its `DType` discriminant, returning
/// `None` for types that do not have a `DType` representation (e.g.
/// user-defined wrappers in tests).
///
/// Used by image-tensor constructors that need the runtime dtype to
/// look up FourCC / pixel-format mappings on the macOS IOSurface path,
/// where the GPU backend cares about whether the bytes are u8 / f16 /
/// f32 even though the static `Tensor<T>` carries the same information
/// at the type level.
///
/// macOS-only: the sole caller is the IOSurface DMA branch in
/// `Tensor::<T>::image()`, which is itself `cfg(target_os = "macos")`.
/// Leaving this ungated makes it dead code on every other target, which
/// fails CI under `-D warnings`.
#[cfg(target_os = "macos")]
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
// Quantization metadata — type-gated to integer element types via sealed
// `IntegerType` trait. Accessors on `Tensor<T>` only compile when `T` is
// an integer type; calling them on `Tensor<f32>` / `Tensor<f16>` etc. is a
// compile error, not a runtime one.
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
    // Deliberately NOT implemented for f16 / f32 / f64.
}

/// Integer element types that may carry quantization metadata.
///
/// Sealed trait: implemented for `u8`, `i8`, `u16`, `i16`, `u32`, `i32`,
/// `u64`, `i64`. Cannot be implemented downstream. Float element types
/// (`half::f16`, `f32`, `f64`) are explicitly excluded — quantization
/// metadata does not apply to float tensors per the edgefirst.json spec.
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
/// The quantized storage type is carried on the parent [`Tensor<T>`]; this
/// struct does not duplicate it. Construct via the four named constructors
/// (the only public entry points); direct field mutation is not allowed so
/// invalid combinations cannot be represented.
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

/// Semantic mode discriminant for hot-path kernel dispatch.
///
/// Obtain via [`Quantization::mode`] once at kernel entry; never inside a
/// pixel-level loop. The enum is borrow-based so the hot kernel receives
/// the scales / zero-points as slices without reallocation.
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

    /// Per-tensor asymmetric — the most common runtime shape.
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

    /// Borrow-based dispatch view. Match once at kernel entry.
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
            // The `validate()` path prevents constructing a
            // per-channel Quantization without an axis, so the remaining
            // pattern is unreachable in practice. Fall back to
            // per-tensor symmetric using scale[0] to avoid panicking in
            // release; debug builds assert.
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

    /// Validate against a target tensor shape. Runs in
    /// `Tensor::set_quantization()`. Catches:
    ///   - empty `scale` (reject — must declare at least one factor)
    ///   - `zero_point` length inconsistent with `scale` (reject —
    ///     per-tensor must have len 1, per-channel must match `scale.len`)
    ///   - `axis >= shape.len()` (axis out of range)
    ///   - `scale.len() != shape[axis]` for per-channel
    ///   - per-channel without axis (reject)
    ///   - per-tensor with redundant axis (reject)
    pub(crate) fn validate(&self, shape: &[usize]) -> Result<()> {
        // `Quantization` is `Deserialize`, so malformed JSON like
        // `{"scale": [], "zero_point": []}` could otherwise produce an
        // ill-defined value that confuses `mode()` selection and the
        // per-channel kernels' indexing.
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
    /// Convenience construction from a `(scale, zero_point)` tuple. Matches
    /// the legacy `QuantTuple` / `Quantization::new` calling convention so
    /// existing `(0.1, -128).into()` sites keep working.
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
        fn expecting(&self, f: &mut fmt::Formatter) -> fmt::Result {
            f.write_str("null, i32, or array of i32")
        }
        fn visit_none<E: de::Error>(self) -> std::result::Result<Self::Value, E> {
            Ok(None)
        }
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

/// Monotonic counter for buffer identity IDs.
static NEXT_BUFFER_ID: AtomicU64 = AtomicU64::new(1);

/// Unique identity for a tensor's underlying buffer.
///
/// Created fresh on every buffer allocation or import. The `id` is a monotonic
/// u64 used as a cache key. The `guard` is an `Arc<()>` whose weak references
/// allow downstream caches to detect when the buffer has been dropped.
#[derive(Debug, Clone)]
pub struct BufferIdentity {
    id: u64,
    guard: Arc<()>,
}

impl BufferIdentity {
    /// Create a new unique buffer identity.
    pub fn new() -> Self {
        Self {
            id: NEXT_BUFFER_ID.fetch_add(1, Ordering::Relaxed),
            guard: Arc::new(()),
        }
    }

    /// Unique identifier for this buffer. Changes when the buffer changes.
    pub fn id(&self) -> u64 {
        self.id
    }

    /// Returns a weak reference to the buffer guard. Goes dead when the
    /// owning Tensor is dropped (and no clones remain).
    pub fn weak(&self) -> Weak<()> {
        Arc::downgrade(&self.guard)
    }
}

impl Default for BufferIdentity {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(target_os = "linux")]
use nix::sys::stat::{major, minor};

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
    /// Create a new tensor using the given file descriptor, shape, and optional
    /// name. If no name is given, a random name will be generated.
    ///
    /// On Linux: Inspects the fd to determine DMA vs SHM based on device major/minor.
    /// On other Unix (macOS): Always creates SHM tensor.
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
    /// Defaults to the logical size for storages without spare capacity.
    fn capacity_bytes(&self) -> usize {
        self.size()
    }

    /// Set the logical shape to any shape whose byte size fits the allocation
    /// capacity, without the equal-size constraint of `reshape`.
    fn set_logical_shape(&mut self, shape: &[usize]) -> Result<()> {
        self.reshape(shape)
    }

    /// Map the tensor into memory and return a TensorMap for accessing the
    /// data.
    fn map(&self) -> Result<TensorMap<T>>;

    /// Get the buffer identity for cache keying and liveness tracking.
    fn buffer_identity(&self) -> &BufferIdentity;
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TensorMemory {
    /// Platform-native zero-copy GPU buffer.
    ///
    /// On Linux this is a DMA-BUF (`DmaTensor` in `crates/tensor/src/dma.rs`)
    /// allocated via the DRM/dma-heap subsystem. On macOS this is an
    /// IOSurface (`IoSurfaceTensor` in `crates/tensor/src/iosurface.rs`).
    /// Both fit into the same `TensorStorage::Dma` slot at the trait
    /// level — the public C API discriminant (`HalTensorMemory::Dma=1`)
    /// works on both platforms with no ABI break.
    ///
    /// Allows hardware-accelerated paths (OpenGL backend on Linux via
    /// `EGL_EXT_image_dma_buf_import`; macOS via
    /// `EGL_ANGLE_iosurface_client_buffer`). CPU access via `map()`
    /// incurs cache-coherency overhead on Linux DMA-BUF and is similar
    /// in cost on IOSurface; SHM/Mem are cheaper for CPU-only workloads.
    Dma,
    #[cfg(unix)]
    /// POSIX Shared Memory allocation. Suitable for inter-process
    /// communication, but not suitable for hardware acceleration.
    Shm,

    /// Regular system memory allocation
    Mem,

    /// OpenGL Pixel Buffer Object memory. Created by ImageProcessor
    /// when DMA-buf is unavailable but OpenGL is present.
    Pbo,
}

impl From<TensorMemory> for String {
    fn from(memory: TensorMemory) -> Self {
        match memory {
            TensorMemory::Dma => "dma".to_owned(),
            #[cfg(unix)]
            TensorMemory::Shm => "shm".to_owned(),
            TensorMemory::Mem => "mem".to_owned(),
            TensorMemory::Pbo => "pbo".to_owned(),
        }
    }
}

impl TryFrom<&str> for TensorMemory {
    type Error = Error;

    fn try_from(s: &str) -> Result<Self> {
        match s {
            "dma" => Ok(TensorMemory::Dma),
            #[cfg(unix)]
            "shm" => Ok(TensorMemory::Shm),
            "mem" => Ok(TensorMemory::Mem),
            "pbo" => Ok(TensorMemory::Pbo),
            _ => Err(Error::InvalidMemoryType(s.to_owned())),
        }
    }
}

#[derive(Debug)]
#[allow(dead_code)] // Variants are constructed by downstream crates via pub(crate) helpers
pub(crate) enum TensorStorage<T>
where
    T: Num + Clone + fmt::Debug + Send + Sync,
{
    /// Platform-native zero-copy GPU buffer. Inner type differs per
    /// target: `DmaTensor` on Linux (DMA-BUF fd), `IoSurfaceTensor` on
    /// macOS (CFRetained IOSurface). The shared variant name keeps the
    /// public `TensorMemory::Dma` discriminant stable across platforms.
    #[cfg(target_os = "linux")]
    Dma(DmaTensor<T>),
    #[cfg(target_os = "macos")]
    Dma(IoSurfaceTensor<T>),
    #[cfg(unix)]
    Shm(ShmTensor<T>),
    Mem(MemTensor<T>),
    Pbo(PboTensor<T>),
}

impl<T> TensorStorage<T>
where
    T: Num + Clone + fmt::Debug + Send + Sync,
{
    /// The backing allocation's intrinsic physical row pitch in bytes, if it
    /// has one that is fixed independent of the logical shape. macOS IOSurface
    /// reports its 64-aligned `bytesPerRow`; other backings (Linux DMA, SHM,
    /// Mem, PBO) have no fixed pitch beyond the logical shape and return `None`.
    ///
    /// Used by `configure_image` to preserve the physical pitch when a reused
    /// pool tensor is reconfigured to a smaller logical image — so the decode
    /// writes rows at the surface's real stride and the GPU samples them with
    /// the same stride (the physical-grid / logical-ROI decoupling).
    pub(crate) fn backing_row_stride(&self) -> Option<usize> {
        match self {
            // Only genuine image-formatted IOSurfaces (height > 1) carry a real
            // per-row pitch; a generic byte-bag (height == 1) returns `None` so
            // `configure_image` does not adopt its whole-buffer "row" as a stride.
            #[cfg(target_os = "macos")]
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
            Some(TensorMemory::Dma) => {
                DmaTensor::<T>::new(shape, name).map(TensorStorage::Dma)
            }
            #[cfg(target_os = "macos")]
            Some(TensorMemory::Dma) => {
                IoSurfaceTensor::<T>::new(shape, name).map(TensorStorage::Dma)
            }
            #[cfg(not(any(target_os = "linux", target_os = "macos")))]
            Some(TensorMemory::Dma) => Err(crate::error::Error::NotImplemented(
                "TensorMemory::Dma is only available on Linux (DMA-BUF) and macOS (IOSurface)"
                    .to_owned(),
            )),
            #[cfg(unix)]
            Some(TensorMemory::Shm) => {
                ShmTensor::<T>::new(shape, name).map(TensorStorage::Shm)
            }
            Some(TensorMemory::Mem) => {
                MemTensor::<T>::new(shape, name).map(TensorStorage::Mem)
            }
            Some(TensorMemory::Pbo) => Err(crate::error::Error::NotImplemented(
                "PboTensor cannot be created via Tensor::new() — use ImageProcessor::create_image()".to_owned(),
            )),
            None => {
                if std::env::var("EDGEFIRST_TENSOR_FORCE_MEM")
                    .is_ok_and(|x| x != "0" && x.to_lowercase() != "false")
                {
                    MemTensor::<T>::new(shape, name).map(TensorStorage::Mem)
                } else {
                    #[cfg(target_os = "linux")]
                    {
                        // Linux: Try DMA -> SHM -> Mem
                        match DmaTensor::<T>::new(shape, name) {
                            Ok(tensor) => Ok(TensorStorage::Dma(tensor)),
                            Err(_) => {
                                match ShmTensor::<T>::new(shape, name)
                                    .map(TensorStorage::Shm)
                                {
                                    Ok(tensor) => Ok(tensor),
                                    Err(_) => MemTensor::<T>::new(shape, name)
                                        .map(TensorStorage::Mem),
                                }
                            }
                        }
                    }
                    #[cfg(target_os = "macos")]
                    {
                        // macOS: Try IOSurface -> SHM -> Mem. IOSurface
                        // is the GPU-shareable backend (zero-copy via
                        // ANGLE), filling the same role as DMA-BUF on
                        // Linux. Falls back to SHM if IOSurface alloc
                        // fails (memory pressure or sandboxed contexts).
                        match IoSurfaceTensor::<T>::new(shape, name) {
                            Ok(tensor) => Ok(TensorStorage::Dma(tensor)),
                            Err(_) => match ShmTensor::<T>::new(shape, name)
                                .map(TensorStorage::Shm)
                            {
                                Ok(tensor) => Ok(tensor),
                                Err(_) => MemTensor::<T>::new(shape, name)
                                    .map(TensorStorage::Mem),
                            },
                        }
                    }
                    #[cfg(all(unix, not(any(target_os = "linux", target_os = "macos"))))]
                    {
                        // Other Unix (BSD): Try SHM -> Mem (no DMA)
                        match ShmTensor::<T>::new(shape, name) {
                            Ok(tensor) => Ok(TensorStorage::Shm(tensor)),
                            Err(_) => {
                                MemTensor::<T>::new(shape, name).map(TensorStorage::Mem)
                            }
                        }
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
    /// This is intentionally DMA-only: padding is only meaningful for
    /// buffers that will be imported as GPU textures via EGLImage. PBO,
    /// Shm, and Mem storage doesn't benefit from pitch alignment and
    /// shouldn't pay the memory cost.
    #[cfg(target_os = "linux")]
    pub(crate) fn new_dma_with_byte_size(
        shape: &[usize],
        byte_size: usize,
        name: Option<&str>,
    ) -> Result<Self> {
        DmaTensor::<T>::new_with_byte_size(shape, byte_size, name).map(TensorStorage::Dma)
    }

    // No non-Linux stub: the only caller (`Tensor::image_with_stride`)
    // returns `NotImplemented` directly on non-Linux without ever
    // reaching the storage layer, so defining a stub here would be
    // dead code and fail the `-D warnings` clippy gate on macOS CI.

    /// Create a Mem-backed tensor storage with an explicit byte size that may
    /// exceed `shape.product() * sizeof(T)`.  Used for image tensors with
    /// 64-byte-aligned row strides (see `MemTensor::with_capacity_bytes`).
    pub(crate) fn new_mem_with_byte_size(
        shape: &[usize],
        byte_size: usize,
        name: Option<&str>,
    ) -> Result<Self> {
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
    ///
    /// Used by `Tensor::image()` when the caller requests
    /// `TensorMemory::Dma` and the format has an IOSurface FourCC
    /// mapping (YUYV, RGBA, BGRA today). Falls back to `new_with_byte_size`
    /// otherwise.
    #[cfg(target_os = "macos")]
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

    /// Create a new tensor storage using the given file descriptor, shape,
    /// and optional name.
    #[cfg(unix)]
    fn from_fd(fd: OwnedFd, shape: &[usize], name: Option<&str>) -> Result<Self> {
        #[cfg(target_os = "linux")]
        {
            use nix::sys::stat::fstat;

            let stat = fstat(&fd)?;
            let major = major(stat.st_dev);
            let minor = minor(stat.st_dev);

            log::debug!("Creating tensor from fd: major={major}, minor={minor}");

            if major != 0 {
                // Dma and Shm tensors are expected to have major number 0
                return Err(Error::UnknownDeviceType(major, minor));
            }

            match minor {
                9 | 10 => {
                    // minor number 9 & 10 indicates DMA memory
                    DmaTensor::<T>::from_fd(fd, shape, name).map(TensorStorage::Dma)
                }
                _ => {
                    // other minor numbers are assumed to be shared memory
                    ShmTensor::<T>::from_fd(fd, shape, name).map(TensorStorage::Shm)
                }
            }
        }
        #[cfg(all(unix, not(target_os = "linux")))]
        {
            // On macOS/BSD, always use SHM (no DMA support)
            ShmTensor::<T>::from_fd(fd, shape, name).map(TensorStorage::Shm)
        }
    }
}

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
            TensorStorage::Dma(t) => t.clone_fd(),
            TensorStorage::Shm(t) => t.clone_fd(),
            TensorStorage::Mem(t) => t.clone_fd(),
            TensorStorage::Pbo(t) => t.clone_fd(),
        }
    }

    fn memory(&self) -> TensorMemory {
        match self {
            #[cfg(any(target_os = "linux", target_os = "macos"))]
            TensorStorage::Dma(_) => TensorMemory::Dma,
            #[cfg(unix)]
            TensorStorage::Shm(_) => TensorMemory::Shm,
            TensorStorage::Mem(_) => TensorMemory::Mem,
            TensorStorage::Pbo(_) => TensorMemory::Pbo,
        }
    }

    fn name(&self) -> String {
        match self {
            #[cfg(any(target_os = "linux", target_os = "macos"))]
            TensorStorage::Dma(t) => t.name(),
            #[cfg(unix)]
            TensorStorage::Shm(t) => t.name(),
            TensorStorage::Mem(t) => t.name(),
            TensorStorage::Pbo(t) => t.name(),
        }
    }

    fn shape(&self) -> &[usize] {
        match self {
            #[cfg(any(target_os = "linux", target_os = "macos"))]
            TensorStorage::Dma(t) => t.shape(),
            #[cfg(unix)]
            TensorStorage::Shm(t) => t.shape(),
            TensorStorage::Mem(t) => t.shape(),
            TensorStorage::Pbo(t) => t.shape(),
        }
    }

    fn reshape(&mut self, shape: &[usize]) -> Result<()> {
        match self {
            #[cfg(any(target_os = "linux", target_os = "macos"))]
            TensorStorage::Dma(t) => t.reshape(shape),
            #[cfg(unix)]
            TensorStorage::Shm(t) => t.reshape(shape),
            TensorStorage::Mem(t) => t.reshape(shape),
            TensorStorage::Pbo(t) => t.reshape(shape),
        }
    }

    fn capacity_bytes(&self) -> usize {
        match self {
            #[cfg(any(target_os = "linux", target_os = "macos"))]
            TensorStorage::Dma(t) => t.capacity_bytes(),
            #[cfg(unix)]
            TensorStorage::Shm(t) => t.capacity_bytes(),
            TensorStorage::Mem(t) => t.capacity_bytes(),
            TensorStorage::Pbo(t) => t.capacity_bytes(),
        }
    }

    fn set_logical_shape(&mut self, shape: &[usize]) -> Result<()> {
        match self {
            #[cfg(any(target_os = "linux", target_os = "macos"))]
            TensorStorage::Dma(t) => t.set_logical_shape(shape),
            #[cfg(unix)]
            TensorStorage::Shm(t) => t.set_logical_shape(shape),
            TensorStorage::Mem(t) => t.set_logical_shape(shape),
            TensorStorage::Pbo(t) => t.set_logical_shape(shape),
        }
    }

    fn map(&self) -> Result<TensorMap<T>> {
        match self {
            #[cfg(any(target_os = "linux", target_os = "macos"))]
            TensorStorage::Dma(t) => t.map(),
            #[cfg(unix)]
            TensorStorage::Shm(t) => t.map(),
            TensorStorage::Mem(t) => t.map(),
            TensorStorage::Pbo(t) => t.map(),
        }
    }

    fn buffer_identity(&self) -> &BufferIdentity {
        match self {
            #[cfg(any(target_os = "linux", target_os = "macos"))]
            TensorStorage::Dma(t) => t.buffer_identity(),
            #[cfg(unix)]
            TensorStorage::Shm(t) => t.buffer_identity(),
            TensorStorage::Mem(t) => t.buffer_identity(),
            TensorStorage::Pbo(t) => t.buffer_identity(),
        }
    }
}

/// Multi-backend tensor with optional image format metadata.
///
/// When `format` is `Some`, this tensor represents an image. Width, height,
/// and channels are derived from `shape` + `format`. When `format` is `None`,
/// this is a raw tensor (identical to the pre-refactoring behavior).
#[derive(Debug)]
pub struct Tensor<T>
where
    T: Num + Clone + fmt::Debug + Send + Sync,
{
    /// CUDA registration for this tensor, if any. Set after creation by
    /// the image crate once a PBO is registered with CUDA interop.
    ///
    /// MUST be declared before `storage`: CUDA must unregister the GL buffer
    /// before storage's Drop deletes it (cudaGraphicsUnregisterResource before
    /// glDeleteBuffers). Rust drops fields in declaration order.
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
    /// Quantization metadata for integer-typed tensors. Public access is
    /// gated by the `IntegerType` trait — `Tensor<f32>` etc. carry the
    /// field for layout uniformity but have no way to read or write it.
    pub(crate) quantization: Option<Quantization>,
    /// Optional colorimetry metadata. `None` = undefined; never auto-filled.
    colorimetry: Option<crate::Colorimetry>,
}

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
        // Best-effort: attach a CUDA ExternalMemory handle for DMA tensors on
        // CUDA-capable hosts. Never blocks tensor creation on failure.
        // RUNTIME-UNVALIDATED: no CUDA+dma_heap test platform available; ABI
        // layout-asserted vs. CUDA 12.6 driver_types.h; mechanism proven by
        // gpu-probe O5 on Orin.
        #[cfg(target_os = "linux")]
        t.try_init_dma_cuda();
        Ok(t)
    }

    /// Create an image tensor with the given format.
    pub fn image(
        width: usize,
        height: usize,
        format: PixelFormat,
        memory: Option<TensorMemory>,
    ) -> Result<Self>
    where
        T: 'static,
    {
        // Shape comes from the shared `PixelFormat::image_shape` helper (packed /
        // planar / semi-planar NV12·NV16). NV12 supports odd dimensions via the
        // `H + ceil(H/2)` combined-plane height.
        // The `T: 'static` bound is required by the macOS IOSurface path below.
        let shape = format.image_shape(width, height).ok_or_else(|| {
            Error::InvalidArgument(format!(
                "invalid dimensions {width}x{height} for format {format:?}"
            ))
        })?;

        // macOS Dma path: allocate a format-aware IOSurface (FourCC +
        // 2D dimensions) so the GL backend can bind it via
        // `EGL_ANGLE_iosurface_client_buffer`. Without this, the IOSurface
        // would default to a generic byte buffer (FourCC 'L008') and
        // ANGLE would reject the import with `EGL_BAD_ATTRIBUTE`.
        //
        // Guard: IOSurface rounds `bytes_per_row` up to 64-byte alignment.
        // If the natural row pitch (`width * channels * sizeof(T)`) is not
        // already 64-byte aligned, the padded allocation cannot be mapped
        // as a contiguous packed tensor — CPU reads/writes would use the
        // wrong stride.
        //
        // Explicit-Dma contract: when the caller passes
        // `Some(TensorMemory::Dma)` they have asked for an
        // **image-formatted IOSurface**. Silently downgrading to the
        // generic 'L008' byte-bag when alignment fails buries the
        // mismatch — the caller only finds out hours later when ANGLE
        // (or any GL importer) rejects the bind with
        // `EGL_BAD_ATTRIBUTE`. Same anti-pattern bit us previously on
        // Mali GPUs with DMA-BUF padding. The right behaviour is to
        // fail loudly here with the alignment requirement spelled out
        // so the caller can either pick aligned dimensions, request
        // SHM/Mem explicitly, or pass `memory=None` for auto-select.
        #[cfg(target_os = "macos")]
        if matches!(memory, Some(TensorMemory::Dma)) {
            // For planar formats the IOSurface stacks channels
            // vertically (channels * height rows), so the row stride is
            // single-channel width * sizeof(T). Packed formats keep the
            // natural width * channels * sizeof(T) stride.
            let natural_row_bytes = match format.layout() {
                PixelLayout::Planar => width * std::mem::size_of::<T>(),
                _ => width * format.channels() * std::mem::size_of::<T>(),
            };
            // A format with a real IOSurface FourCC (RGBA/BGRA/YUYV packed,
            // GREY/NV12/NV16/NV24 as R8) tolerates a non-64-aligned natural
            // pitch: the surface is allocated with its own 64-aligned
            // `bytes_per_row`, the tensor records that stride below, and a CPU
            // map iterates rows correctly via the strided-map path while the GL
            // import uses the surface's pitch directly — fully zero-copy.
            // Planar (the F16 RGBA16F packing) is consumed flat as
            // `[1, C, H, W]` with no stride, so it still requires an aligned
            // pitch; and formats without a FourCC would fall through to a
            // generic byte-bag GL can't bind. Both fail loudly rather than
            // silently downgrade.
            let has_image_fourcc = dtype_of::<T>()
                .and_then(|dt| crate::iosurface::image_iosurface_layout(format, dt))
                .is_some();
            let padded_ok = has_image_fourcc && format.layout() != PixelLayout::Planar;
            if !natural_row_bytes.is_multiple_of(64) && !padded_ok {
                let elem_size = std::mem::size_of::<T>();
                let per_pixel_bytes = match format.layout() {
                    PixelLayout::Planar => elem_size.max(1),
                    _ => format.channels().max(1) * elem_size.max(1),
                };
                // Compute the next 64-byte-aligned width by rounding the
                // natural row pitch up to the next multiple of 64 and
                // dividing back by per-pixel bytes. This handles every
                // `per_pixel_bytes` value correctly:
                //
                //   * Divisors of 64 (1/2/4/8/16/32/64) → the suggestion
                //     is always 64-byte aligned.
                //   * Non-divisors of 64 (e.g. RGB u8 with 3 B/pixel) →
                //     the next aligned row pitch may not be an integer
                //     multiple of per_pixel_bytes (3 doesn't divide 64
                //     in any way), so a "pad width to N" suggestion is
                //     structurally impossible — omit the suggestion
                //     instead of printing a wrong number.
                //   * per_pixel_bytes > 64 → same situation, also
                //     omitted; the previous formula divided by zero.
                //
                // The error always names the alignment requirement
                // verbatim and lists the two non-DMA alternatives so
                // the caller has at least one always-applicable fix.
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
            // Alignment OK. Try image-formatted IOSurface; on any
            // structural failure (unsupported (format, dtype) combo,
            // out-of-memory, etc.) fall through to SHM. dtype_of
            // returns None for non-standard numeric T used in tests —
            // those legitimately have no IOSurface FourCC mapping.
            if let Some(dtype) = dtype_of::<T>() {
                if let Ok(storage) = TensorStorage::<T>::new_image_iosurface(
                    width, height, format, dtype, &shape, None,
                ) {
                    let mut t = Self::wrap(storage);
                    t.format = Some(format);
                    // IOSurface rounds `bytes_per_row` up to 64 bytes. When that
                    // pitch exceeds the natural packed/planar row stride, record
                    // it so CPU consumers iterate rows correctly (the GL import
                    // already uses the surface's own pitch). For 64-aligned rows
                    // — the common model-input case — the two match and no stride
                    // is stored, leaving the flat mapping unchanged.
                    if let TensorStorage::Dma(ref io) = t.storage {
                        let bpr = io.bytes_per_row();
                        if let Some(natural) = t.effective_row_stride() {
                            if bpr > natural {
                                t.set_row_stride_unchecked(bpr);
                            }
                        }
                    }
                    return Ok(t);
                }
            }
            // Unsupported (format, dtype) on the IOSurface path —
            // e.g. PlanarRgb u8 has no IOSurface FourCC mapping today
            // (only F16 PlanarRgb is wired). Fall through to SHM/Mem
            // since the caller asked for Dma but no image-formatted DMA
            // storage exists for this combination.
        }

        // Compute the **64-byte-aligned** row stride for every image layout.
        //
        // Embedded GPUs reject `eglCreateImage` DMA-BUF imports whose row pitch
        // is not 64-byte aligned: Mali returns `EGL_BAD_ALLOC`, Vivante
        // `EGL_BAD_ACCESS`. This bit packed RGBA/RGB destinations at odd widths
        // AND at even non-multiple-of-16 widths (e.g. 321→1284, 322→1288 bytes —
        // neither divisible by 64), so an odd-source → RGBA convert failed on
        // imx95/imx8mp while succeeding on V3D/Tegra. Semi-planar already aligned
        // here; we now align packed and planar identically so every image()
        // allocation is GPU-importable regardless of width.
        //
        // The per-layout natural pitch and total row count:
        //   * SemiPlanar `[total_h, width]`     — pitch = even(width)·elem, rows = total_h
        //   * Packed     `[height, width, ch]`  — pitch = width·ch·elem,    rows = height
        //   * Planar     `[ch, height, width]`  — pitch = width·elem,       rows = ch·height
        // Allocation byte size = `aligned_stride · total_rows` (NOT the shape
        // product, which reflects only the logical width and under-allocates the
        // padding on odd / unaligned widths).
        let elem = std::mem::size_of::<T>();
        let channels = format.channels();
        let (natural_stride, total_rows) = match format.layout() {
            PixelLayout::SemiPlanar => (width.next_multiple_of(2) * elem, shape[0]),
            PixelLayout::Packed => (width * channels * elem, height),
            PixelLayout::Planar => (width * elem, channels * height),
        };
        let aligned_stride = natural_stride.next_multiple_of(64);
        let semi = format.layout() == PixelLayout::SemiPlanar;

        // DMA buffers MUST carry a 64-aligned row pitch — Mali/Vivante reject a
        // DMA-BUF EGLImage whose pitch is not 64-aligned. Semi-planar also needs
        // the aligned pitch on every backend (its chroma-plane offset math
        // assumes it). Packed/planar on host-only memory (Mem/Shm) keep the
        // natural tight pitch so the many flat CPU consumers are unaffected.
        let host_stride = if semi { aligned_stride } else { natural_stride };
        let host_byte_size = host_stride * total_rows;
        #[cfg(target_os = "linux")]
        let dma_byte_size = aligned_stride * total_rows;

        // `used_stride` is the actual row pitch of the storage created below.
        let (storage, used_stride) = match memory {
            #[cfg(target_os = "linux")]
            Some(TensorMemory::Dma) => (
                TensorStorage::<T>::new_dma_with_byte_size(&shape, dma_byte_size, None)?,
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
                // PBO and any future variants: fall through to standard new().
                return {
                    let mut t = Self::new(&shape, Some(other), None)?;
                    t.format = Some(format);
                    Ok(t)
                };
            }
            None => {
                // Auto-select: DMA → Shm → Mem on Linux; Shm → Mem on macOS.
                // DMA gets the 64-aligned pitch; host fallbacks keep the tight
                // (host) pitch, so the recorded stride matches the storage used.
                #[cfg(target_os = "linux")]
                {
                    match TensorStorage::<T>::new_dma_with_byte_size(&shape, dma_byte_size, None) {
                        Ok(s) => (s, aligned_stride),
                        Err(_) => {
                            match TensorStorage::<T>::new_shm_with_byte_size(
                                &shape,
                                host_byte_size,
                                None,
                            ) {
                                Ok(s) => (s, host_stride),
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
                    }
                }
                #[cfg(all(unix, not(target_os = "linux")))]
                {
                    match TensorStorage::<T>::new_shm_with_byte_size(&shape, host_byte_size, None) {
                        Ok(s) => (s, host_stride),
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
                #[cfg(not(unix))]
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
        // Record the row stride when it exceeds the natural tight pitch (padding
        // is present — DMA packed/planar at an unaligned width, or always for
        // semi-planar), mirroring the IOSurface path above. Aligned-width and
        // host-only packed/planar images keep their flat layout with no explicit
        // stride; `effective_row_stride()` then falls back to the identical
        // computed pitch. When padding IS present, consumers must iterate rows by
        // `effective_row_stride()` to skip it.
        if semi || used_stride > natural_stride {
            t.set_row_stride_unchecked(used_stride);
        }
        debug_assert!(
            t.row_stride.is_some() || !semi,
            "image() must always set row_stride for semi-planar tensors"
        );
        #[cfg(target_os = "linux")]
        t.try_init_dma_cuda();
        Ok(t)
    }

    /// Create a DMA-backed image tensor with an explicit row stride that
    /// may exceed the natural `width * channels * sizeof(T)` pitch.
    ///
    /// Used for image tensors that need GPU pitch alignment padding: the
    /// underlying DMA-BUF is sized to `row_stride * height` bytes, but
    /// the tensor's logical shape stays at `[height, width, channels]`.
    /// `width()` / `height()` / `shape()` continue to report the
    /// user-requested values; the padding is visible only via
    /// `row_stride()` / `effective_row_stride()` and is automatically
    /// propagated to the GL backend's EGLImage import so Mali Valhall
    /// accepts the buffer.
    ///
    /// # Supported formats
    ///
    /// Currently only **packed** pixel layouts (RGBA8, BGRA8, RGB888,
    /// Grey, etc.) are supported — the formats the GL backend uses as
    /// render destinations. Semi-planar formats (NV12, NV16) come from
    /// external allocators (camera capture, video decoders) and are
    /// imported via `TensorDyn::from_fd` + `set_row_stride`, which
    /// already supports padded strides.
    ///
    /// # Supported memory
    ///
    /// Currently only `TensorMemory::Dma` is supported. PBO and Mem
    /// storage don't go through EGLImage import so they don't need
    /// pitch alignment; if you pass any other memory type this returns
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
    ) -> Result<Self> {
        // DMA backing (the only thing this constructor produces) is
        // Linux-only. On macOS/BSD/Windows the non-Linux block below is
        // the only compiled body and returns `NotImplemented` directly;
        // on Linux the non-Linux block is cfg-removed and the function
        // falls through to the real validation + allocation path. Each
        // target compiles exactly one of the two blocks, and the block
        // serves as the function's tail expression in both cases — so
        // neither needs an explicit `return` (avoids
        // `clippy::needless_return` on the macOS CI gate).
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
                Some(TensorMemory::Dma) | None => {
                    TensorStorage::<T>::new_dma_with_byte_size(&shape, total_byte_size, None)?
                }
                Some(other) => {
                    return Err(Error::NotImplemented(format!(
                        "image_with_stride: only TensorMemory::Dma is supported, got {other:?}"
                    )));
                }
            };

            let mut t = Self::wrap(storage);
            t.format = Some(format);
            t.row_stride = Some(row_stride_bytes);
            // Match new()/from_fd(): a DMA tensor must attempt CUDA external-
            // memory import so a strided DMA buffer is also zero-copy
            // CUDA-mappable (no-op when libcudart is absent).
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
                    // Combined-plane height is `H + ceil(H/2)` (luma + chroma
                    // rows). For even H that is `3H/2` (≡ 0 mod 3); for odd H it
                    // is `(3H+1)/2` (≡ 2 mod 3). Only totals ≡ 1 mod 3 are
                    // unreachable, so reject just those — odd-height NV12 is
                    // valid (e.g. 725 rows for a 483-tall image).
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
        // Clear stored stride/offset when format changes — they may be invalid
        // for the new format. Caller must re-set after changing format.
        if self.format != Some(format) {
            self.row_stride = None;
            self.plane_offset = None;
            match self.storage {
                TensorStorage::Mem(ref mut m) => m.set_offset(0),
                #[cfg(target_os = "linux")]
                TensorStorage::Dma(ref mut dma) => dma.mmap_offset = 0,
                _ => {}
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
    /// For NV12/NV16/NV24 the buffer width is rounded up to even (a chroma-plane
    /// interleaving requirement); the true odd width is reported by the decoder
    /// in `ImageInfo` and trimmed by a `convert()` crop. See
    /// [`PixelFormat::image_shape`].
    ///
    /// When the backing has a fixed physical row pitch (an IOSurface's
    /// 64-aligned `bytesPerRow`) that exceeds the new format's natural row
    /// stride — i.e. a reused max-sized pool tensor reconfigured to a smaller
    /// image — the physical pitch is preserved as the tensor's `row_stride`.
    /// This keeps the **physical grid** (allocation stride/surface) fixed while
    /// the **logical ROI** (this image's W×H) changes, so the decode writes rows
    /// at the surface's real stride and the GPU samples them at the same stride.
    /// Exact-sized buffers (pitch == natural) stay tightly packed unchanged.
    pub fn configure_image(
        &mut self,
        width: usize,
        height: usize,
        format: PixelFormat,
    ) -> Result<()> {
        let shape = format.image_shape(width, height).ok_or_else(|| {
            Error::InvalidArgument(format!(
                "invalid dimensions {width}x{height} for format {format:?}"
            ))
        })?;
        // Capture the pre-existing row stride before `set_format` clears it.
        // For pool tensors that were allocated at a larger width (e.g. 1920-wide
        // pool decoding a 789-wide image), this preserves the backing pitch so
        // rows are still written at the correct physical stride.
        let prior_stride = self.row_stride;

        self.storage.set_logical_shape(&shape)?;
        self.set_format(format)?; // clears any stale row_stride

        // Preserve the backing's physical row pitch when it is wider than the
        // natural (tightly-packed) row for this format — the physical-grid /
        // logical-ROI decoupling for reused pool tensors.
        let active_stride = if let Some(pitch) = self.storage.backing_row_stride() {
            // macOS IOSurface: use the surface's native pitch.
            let natural = self.effective_row_stride().unwrap_or(0);
            if pitch > natural {
                self.set_row_stride_unchecked(pitch);
                pitch
            } else {
                natural
            }
        } else if format.layout() == PixelLayout::SemiPlanar {
            // For self-allocated SemiPlanar tensors (Mem/Shm on any platform,
            // DMA on Linux), restore the correct 64-byte-aligned stride so the
            // buffer is fully exploited.
            //
            // Priority:
            //   1. Prior stride (pool reuse): if the pre-existing stride is
            //      64-aligned, >= even(width), and fits the allocation, keep it.
            //      This is the hot-loop reuse case (large pool, small image).
            //   2. Compute fresh 64-aligned stride for the current width.
            let elem = std::mem::size_of::<T>();
            let min_stride = width.next_multiple_of(2) * elem;
            let aligned = min_stride.next_multiple_of(64);
            let total_h = shape[0];
            let capacity = self.storage.capacity_bytes();

            let candidate = if let Some(ps) = prior_stride {
                // Keep the prior stride only when it still satisfies the current
                // minimum (even(width)) and the allocation can hold it.
                if ps >= min_stride && ps % 64 == 0 && ps * total_h <= capacity {
                    ps
                } else {
                    aligned
                }
            } else {
                aligned
            };

            if candidate * total_h <= capacity {
                self.set_row_stride_unchecked(candidate);
                candidate
            } else {
                // Shouldn't happen for legitimate pools, but don't crash.
                self.effective_row_stride().unwrap_or(0)
            }
        } else {
            self.effective_row_stride().unwrap_or(0)
        };

        // For semi-planar formats, ensure the active stride fits the allocation.
        // A pool reconfigured to a wider image than its backing would silently
        // SIGBUS on any subsequent map/write — catch it here instead.
        if format.layout() == PixelLayout::SemiPlanar && active_stride > 0 {
            let total_h = shape[0];
            let needed = active_stride * total_h;
            let capacity = self.storage.capacity_bytes();
            if needed > capacity {
                return Err(Error::InsufficientCapacity { needed, capacity });
            }
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
    ) -> Result<Self>
    where
        T: 'static,
    {
        Self::image(width, height, format, memory)
    }

    /// Pixel format (None if not an image).
    pub fn format(&self) -> Option<PixelFormat> {
        self.format
    }

    /// Image width (None if not an image).
    pub fn width(&self) -> Option<usize> {
        let fmt = self.format?;
        let shape = self.shape();
        match fmt.layout() {
            PixelLayout::Packed => Some(shape[1]),
            PixelLayout::Planar => Some(shape[2]),
            PixelLayout::SemiPlanar => Some(shape[1]),
        }
    }

    /// Image height (None if not an image).
    ///
    /// For semi-planar formats the combined-plane shape row count is divided
    /// by the format's luma-to-total ratio to recover logical height. This
    /// returns the exact logical height (including odd heights) only because
    /// the logical dimensions are tracked separately from the physical shape —
    /// `configure_image` stores the actual `(width, height)` in the format's
    /// `image_shape`, which round-trips losslessly via these accessors.
    pub fn height(&self) -> Option<usize> {
        let fmt = self.format?;
        let shape = self.shape();
        match fmt.layout() {
            PixelLayout::Packed => Some(shape[0]),
            PixelLayout::Planar => Some(shape[1]),
            PixelLayout::SemiPlanar => {
                if self.is_multiplane() {
                    Some(shape[0])
                } else {
                    match fmt {
                        PixelFormat::Nv12 => Some(shape[0] * 2 / 3),
                        PixelFormat::Nv16 => Some(shape[0] / 2),
                        PixelFormat::Nv24 => Some(shape[0] / 3),
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
            // NV24's chroma plane is full-resolution (2×-wide interleaved UV),
            // which the equal-width plane check above doesn't model. Multiplane
            // NV24 is unused (the JPEG decoder emits a contiguous NV24 buffer),
            // so it's not supported here yet.
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
            // A multiplane tensor spans two DMA-BUFs (luma + chroma); CUDA
            // external-memory import is per-fd, so there is no single device
            // pointer for the composite. Any CUDA handle the luma plane carried
            // is intentionally dropped — consumers needing CUDA access to
            // multiplane data must import each plane independently.
            cuda: None,
            colorimetry: luma.colorimetry,
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

    /// Row stride in bytes (`None` = tightly packed).
    pub fn row_stride(&self) -> Option<usize> {
        self.row_stride
    }

    /// Effective row stride in bytes: the stored stride if set, otherwise the
    /// minimum stride computed from the format, width, and element size.
    /// Returns `None` only when no format is set and no explicit stride was
    /// stored via [`set_row_stride`](Self::set_row_stride).
    ///
    /// **GREY note:** `effective_row_stride()` for a GREY tensor returns the
    /// tight `width` bytes (no padding), which is what `normalize_to_numpy` and
    /// the CPU convert path expect. The codec's internal `native_row_stride`
    /// (64-byte-aligned) is used only during decoding and is not propagated to
    /// the tensor's stored stride, so callers reading via
    /// `effective_row_stride()` always see the tight value for GREY.
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
            // Semi-planar: minimum stride must cover the even width so the
            // interleaved chroma columns are byte-aligned on odd-width images.
            PixelLayout::SemiPlanar => w.next_multiple_of(2) * elem,
        })
    }

    /// Set the row stride in bytes for externally allocated buffers with
    /// row padding (e.g. V4L2 or GStreamer allocators).
    ///
    /// The stride is propagated to the EGL DMA-BUF import attributes so
    /// the GPU interprets the padded buffer layout correctly. Must be
    /// called after [`set_format`](Self::set_format) and before the tensor
    /// is first passed to [`ImageProcessor::convert`]. The stored stride
    /// is cleared automatically if the pixel format is later changed.
    ///
    /// No stride-vs-buffer-size validation is performed because the
    /// backing allocation size is not reliably known: external DMA-BUFs
    /// may be over-allocated by the allocator, and internal tensors store
    /// a logical (unpadded) shape. An incorrect stride will be caught by
    /// the EGL driver at import time.
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
            // Semi-planar: minimum must cover even width for chroma alignment.
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
    /// Use this for raw sub-tensors (e.g. chroma planes) that don't carry
    /// format metadata. The caller is responsible for ensuring the stride
    /// is valid.
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

    /// Set the byte offset within the DMA-BUF where image data starts.
    ///
    /// Propagated to `EGL_DMA_BUF_PLANE0_OFFSET_EXT` on GPU import.
    /// Unlike [`set_row_stride`](Self::set_row_stride), no format is required
    /// since the offset is format-independent.
    pub fn set_plane_offset(&mut self, offset: usize) {
        self.plane_offset = Some(offset);
        // The offset consulted by `map()` lives inside the storage variant.
        // Keep it in sync with the wrapper field for every backing that
        // honors it (DMA and Mem); see also the clear sites in `set_format`
        // and `reshape`.
        match self.storage {
            TensorStorage::Mem(ref mut m) => m.set_offset(offset),
            #[cfg(target_os = "linux")]
            TensorStorage::Dma(ref mut dma) => dma.mmap_offset = offset,
            _ => {}
        }
    }

    /// Builder-style variant of [`set_plane_offset`](Self::set_plane_offset),
    /// consuming and returning `self`.
    pub fn with_plane_offset(mut self, offset: usize) -> Self {
        self.set_plane_offset(offset);
        self
    }

    /// Colorimetry metadata (`None` = undefined; never auto-filled).
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
    /// measured from this tensor's own logical start. N sub-views into one
    /// parent can be written independently, enabling batched assembly into a
    /// single buffer. Identical semantics across `Mem` (shared `Arc`) and
    /// `Dma` (shared fd) backings.
    ///
    /// # Disjointness
    ///
    /// Independent writes are sound *only* when the windows do not overlap. The
    /// shared backing uses interior mutability (`UnsafeCell` cells), so two
    /// sub-views whose byte ranges intersect alias the same cells: writing one
    /// while reading or writing the other is a data race and therefore
    /// **undefined behaviour**. The caller is responsible for keeping the
    /// windows disjoint; this method does not check for overlap.
    ///
    /// # Errors
    ///
    /// - [`Error::InvalidOperation`] if the backing is not `Mem` or `Dma`, or
    ///   if `offset_bytes` is not a multiple of `align_of::<T>()`.
    /// - [`Error::InsufficientCapacity`] / [`Error::InvalidSize`] if the window
    ///   exceeds the parent allocation.
    pub fn subview(&self, offset_bytes: usize, shape: &[usize]) -> Result<Tensor<T>> {
        // Offset is absolute into the backing allocation: a sub-view of a
        // sub-view composes by adding this tensor's own offset.
        let abs_offset = self
            .plane_offset
            .unwrap_or(0)
            .checked_add(offset_bytes)
            .ok_or(Error::InvalidSize(offset_bytes))?;
        let mut t = match &self.storage {
            TensorStorage::Mem(parent) => Tensor::wrap(TensorStorage::Mem(MemTensor::view(
                parent,
                offset_bytes,
                shape,
            )?)),
            #[cfg(target_os = "linux")]
            TensorStorage::Dma(parent) => {
                // Shares the parent's fd AND BufferIdentity (unlike from_fd,
                // which mints a fresh identity) so offset-distinct views of one
                // DMA-BUF are cached per (identity, offset) in the GL backend.
                Tensor::wrap(TensorStorage::Dma(parent.view(offset_bytes, shape)?))
            }
            _ => {
                return Err(Error::InvalidOperation(
                    "subview only supported for Mem and Dma tensors".into(),
                ))
            }
        };
        // Inherit the parent's image metadata so the view is a ready-to-use
        // sub-image (e.g. a `convert()` destination). The offset is applied
        // LAST because `set_format` deliberately clears it — the offset is a
        // structural property of the sub-region, not format-dependent metadata.
        if let Some(fmt) = self.format {
            t.set_format(fmt)?;
        }
        if let Some(rs) = self.row_stride {
            t.set_row_stride_unchecked(rs);
        }
        t.quantization = self.quantization.clone();
        // A sub-region of an image carries the parent's colorimetry — it is the
        // same pixels, same color encoding. Inherit it like the other image
        // metadata above so a sub-view is a faithful convert() source/target.
        t.set_colorimetry(self.colorimetry);
        if abs_offset > 0 {
            t.set_plane_offset(abs_offset);
        }
        Ok(t)
    }

    /// Downcast to PBO tensor reference (for GL backends).
    pub fn as_pbo(&self) -> Option<&PboTensor<T>> {
        match &self.storage {
            TensorStorage::Pbo(p) => Some(p),
            _ => None,
        }
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
    pub fn from_pbo(pbo: PboTensor<T>) -> Self {
        Self {
            storage: TensorStorage::Pbo(pbo),
            format: None,
            chroma: None,
            row_stride: None,
            plane_offset: None,
            quantization: None,
            cuda: None,
            colorimetry: None,
        }
    }

    /// The CUDA registration for this tensor, if any (set at creation on CUDA devices).
    pub fn cuda(&self) -> Option<&crate::cuda::CudaHandle> {
        self.cuda.as_ref()
    }

    /// Attach a CUDA handle (called by ImageProcessor::create_image after registering a PBO).
    pub fn set_cuda_handle(&mut self, h: crate::cuda::CudaHandle) {
        self.cuda = Some(h);
    }

    /// Fast-fail CUDA map: None (no GL routing) when no handle; else map (PBO routes to the GL worker).
    ///
    /// Returns a scoped [`CudaMap`](crate::cuda::CudaMap) guard holding the raw CUDA device pointer
    /// for the duration of the mapping. For GL-buffer-backed tensors the unmap is deferred until the
    /// guard drops, freeing the PBO for the next `convert()` call. When no CUDA handle is attached
    /// (the common case for plain `Mem`/`DMA` tensors without CUDA registration), returns `None`
    /// immediately — no GL routing, no allocation.
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
    pub fn cuda_map(&self) -> Option<crate::cuda::CudaMap<'_>> {
        self.cuda.as_ref()?.map()
    }

    /// Attempt to attach a CUDA `ExternalMemory` handle for DMA-backed tensors.
    ///
    /// On a CUDA-capable host, imports the DMA-BUF fd via
    /// `cudaImportExternalMemory(OpaqueFd)` and maps it to a device pointer.
    /// Sets `self.cuda` to a persistent `ExternalMem` handle on success. No-op
    /// if CUDA is unavailable, the tensor is not DMA-backed, or a handle is
    /// already set. Import failure is silently ignored — the tensor remains
    /// usable without a CUDA handle.
    ///
    /// # RUNTIME-UNVALIDATED
    ///
    /// No test platform has both `/dev/dma_heap` and a CUDA device. ABI is
    /// layout-asserted vs. CUDA 12.6 `driver_types.h`; the mechanism is proven
    /// by gpu-probe O5 on Orin. Best-effort: tensor creation never fails here.
    #[cfg(target_os = "linux")]
    pub fn try_init_dma_cuda(&mut self) {
        // Fast-path: already imported, CUDA not available, or not a DMA tensor.
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
}

// Quantization accessors — type-gated to integer element types via the
// sealed `IntegerType` trait. Calling `.quantization()` on a `Tensor<f32>`
// produces a compile error, not a runtime one.
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

    /// Builder-style variant of [`Self::set_quantization`]. Consumes `self`
    /// and returns `Result<Self>` — on success yields the tensor with the
    /// attached quantization; on validation failure returns
    /// [`Error::QuantizationInvalid`] and drops `self` (the tensor is not
    /// returned in the error arm).
    pub fn with_quantization(mut self, q: Quantization) -> Result<Self> {
        self.set_quantization(q)?;
        Ok(self)
    }

    /// Clear any quantization metadata on this tensor.
    pub fn clear_quantization(&mut self) {
        self.quantization = None;
    }
}

impl<T> TensorTrait<T> for Tensor<T>
where
    T: Num + Clone + fmt::Debug + Send + Sync,
{
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
        #[cfg_attr(not(target_os = "linux"), allow(unused_mut))]
        let mut t = Self::wrap(TensorStorage::from_fd(fd, shape, name)?);
        // Best-effort CUDA external memory import for DMA-backed tensors.
        // RUNTIME-UNVALIDATED: see try_init_dma_cuda().
        #[cfg(target_os = "linux")]
        t.try_init_dma_cuda();
        Ok(t)
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
            _ => {}
        }
        Ok(())
    }

    fn map(&self) -> Result<TensorMap<T>> {
        let _span = tracing::trace_span!(
            "tensor.map",
            memory = ?self.storage.memory(),
        )
        .entered();
        // CPU mapping of a strided tensor exposes the full padded buffer
        // (`row_stride × rows`) so callers can iterate rows via
        // `effective_row_stride()` without running past the slice. This is sound
        // only when the HAL owns and can size-check the allocation:
        //
        //   * Self-allocated Mem / Shm tensors (any platform) — the backing
        //     `Vec` / shm segment is sized by `capacity_bytes()`, checked here.
        //   * Self-allocated DMA tensors (Linux) — pitch padding from
        //     `image_with_stride()`; checked against the DMA-BUF `buf_size`.
        //
        // Foreign DMA-BUFs (`from_fd()` + `set_row_stride()`, the V4L2 /
        // GStreamer case), IOSurface, and PBO storages are rejected: their
        // layout comes from an external allocator / GPU driver the HAL cannot
        // validate for a strided CPU view, and they are intended for the GPU
        // path. (Earlier this rejected *all* non-Linux strided maps with
        // "DMA backing is Linux-only" — that was an unimplemented path, not a
        // platform limit; HAL-owned Mem/Shm are trivially mappable and now are.)
        if let Some(stride) = self.row_stride {
            // Rows sit at `stride`-byte spacing; the first shape dim is the row
            // count for packed `[H, W, C]` and semi-planar `[H*k, W]` alike.
            let rows = *self.shape().first().ok_or_else(|| {
                Error::InvalidOperation(
                    "Tensor::map: strided mapping requires a non-empty shape".into(),
                )
            })?;
            let total_bytes = stride.checked_mul(rows).ok_or_else(|| {
                Error::InvalidOperation(format!(
                    "Tensor::map: row_stride {stride} × rows {rows} overflows usize"
                ))
            })?;

            match &self.storage {
                #[cfg(target_os = "linux")]
                TensorStorage::Dma(dma) if !dma.is_imported => {
                    // `set_row_stride()` only validates `stride >= min_stride`,
                    // not that `stride × rows` fits the DMA-BUF, so re-check
                    // here — mapping past `buf_size` would SIGBUS on access.
                    let available_bytes = dma.buf_size.saturating_sub(dma.mmap_offset);
                    if total_bytes > available_bytes {
                        return Err(Error::InvalidOperation(format!(
                            "Tensor::map: strided mapping needs {total_bytes} bytes \
                             but DMA buffer only has {available_bytes} available \
                             (buf_size={}, mmap_offset={}, stride={stride}, rows={rows}); \
                             the row_stride was likely set larger than the original allocation",
                            dma.buf_size, dma.mmap_offset
                        )));
                    }
                    return dma.map_with_byte_size(total_bytes).map(TensorMap::Dma);
                }
                TensorStorage::Mem(mem) => {
                    let capacity = self.storage.capacity_bytes();
                    if total_bytes > capacity {
                        return Err(Error::InsufficientCapacity {
                            needed: total_bytes,
                            capacity,
                        });
                    }
                    return mem.map_with_byte_size(total_bytes);
                }
                #[cfg(unix)]
                TensorStorage::Shm(shm) => {
                    let capacity = self.storage.capacity_bytes();
                    if total_bytes > capacity {
                        return Err(Error::InsufficientCapacity {
                            needed: total_bytes,
                            capacity,
                        });
                    }
                    return shm.map_with_byte_size(total_bytes);
                }
                // macOS: `TensorStorage::Dma` is the IOSurface. The lock yields
                // the full surface base address, and the row pitch
                // (`IOSurfaceGetBytesPerRow`) is known from the API for both
                // self-allocated and imported surfaces — unlike a foreign
                // DMA-BUF — so a strided CPU view is sound and zero-copy.
                #[cfg(target_os = "macos")]
                TensorStorage::Dma(io) => {
                    if total_bytes > io.buf_size {
                        return Err(Error::InsufficientCapacity {
                            needed: total_bytes,
                            capacity: io.buf_size,
                        });
                    }
                    return io.map_with_byte_size(total_bytes);
                }
                TensorStorage::Pbo(pbo) => {
                    // PBO: the GPU-side allocation may have a padded stride; the
                    // map() call maps the full `handle.size` bytes and the caller
                    // iterates rows via row_stride.  Validate that the strided
                    // view fits the PBO capacity before mapping.
                    let capacity = pbo.capacity_bytes();
                    if total_bytes > capacity {
                        return Err(Error::InsufficientCapacity {
                            needed: total_bytes,
                            capacity,
                        });
                    }
                    return pbo.map();
                }
                // Reachable on Linux for an IMPORTED DMA-BUF (the `Dma` arm above
                // is guarded `if !dma.is_imported`). On macOS/Windows every
                // storage variant is matched explicitly, so this catch-all is
                // unreachable there — allow it rather than cfg-gating per platform.
                #[allow(unreachable_patterns)]
                _ => {
                    return Err(Error::InvalidOperation(
                        "CPU mapping of strided tensors is supported only for HAL-allocated \
                         Mem/Shm (any platform), self-allocated DMA (Linux), IOSurface \
                         (macOS), and PBO; imported DMA-BUF without self-allocation is \
                         GPU-path only"
                            .into(),
                    ));
                }
            }
        }
        // Offset tensors are supported for storages that apply the offset
        // inside their own `map()`: DMA (`DmaMap` adjusts the mmap range) and
        // Mem (`MemMap` adjusts the slice base). Other backings have no
        // sub-region concept, so a non-zero offset is rejected.
        if self.plane_offset.is_some_and(|o| o > 0) {
            let supported = matches!(self.storage, TensorStorage::Mem(_));
            #[cfg(target_os = "linux")]
            let supported = supported || matches!(self.storage, TensorStorage::Dma(_));
            if !supported {
                return Err(Error::InvalidOperation(
                    "plane offset only supported for DMA and Mem tensors".into(),
                ));
            }
        }
        self.storage.map()
    }

    fn buffer_identity(&self) -> &BufferIdentity {
        self.storage.buffer_identity()
    }
}

pub enum TensorMap<T>
where
    T: Num + Clone + fmt::Debug,
{
    #[cfg(target_os = "linux")]
    Dma(DmaMap<T>),
    #[cfg(target_os = "macos")]
    IoSurface(IoSurfaceMap<T>),
    #[cfg(unix)]
    Shm(ShmMap<T>),
    Mem(MemMap<T>),
    Pbo(PboMap<T>),
}

impl<T> TensorMapTrait<T> for TensorMap<T>
where
    T: Num + Clone + fmt::Debug,
{
    fn shape(&self) -> &[usize] {
        match self {
            #[cfg(target_os = "linux")]
            TensorMap::Dma(map) => map.shape(),
            #[cfg(target_os = "macos")]
            TensorMap::IoSurface(map) => map.shape(),
            #[cfg(unix)]
            TensorMap::Shm(map) => map.shape(),
            TensorMap::Mem(map) => map.shape(),
            TensorMap::Pbo(map) => map.shape(),
        }
    }

    fn unmap(&mut self) {
        match self {
            #[cfg(target_os = "linux")]
            TensorMap::Dma(map) => map.unmap(),
            #[cfg(target_os = "macos")]
            TensorMap::IoSurface(map) => map.unmap(),
            #[cfg(unix)]
            TensorMap::Shm(map) => map.unmap(),
            TensorMap::Mem(map) => map.unmap(),
            TensorMap::Pbo(map) => map.unmap(),
        }
    }

    fn as_slice(&self) -> &[T] {
        match self {
            #[cfg(target_os = "linux")]
            TensorMap::Dma(map) => map.as_slice(),
            #[cfg(target_os = "macos")]
            TensorMap::IoSurface(map) => map.deref(),
            #[cfg(unix)]
            TensorMap::Shm(map) => map.as_slice(),
            TensorMap::Mem(map) => map.as_slice(),
            TensorMap::Pbo(map) => map.as_slice(),
        }
    }

    fn as_mut_slice(&mut self) -> &mut [T] {
        match self {
            #[cfg(target_os = "linux")]
            TensorMap::Dma(map) => map.as_mut_slice(),
            #[cfg(target_os = "macos")]
            TensorMap::IoSurface(map) => map.deref_mut(),
            #[cfg(unix)]
            TensorMap::Shm(map) => map.as_mut_slice(),
            TensorMap::Mem(map) => map.as_mut_slice(),
            TensorMap::Pbo(map) => map.as_mut_slice(),
        }
    }
}

impl<T> Deref for TensorMap<T>
where
    T: Num + Clone + fmt::Debug,
{
    type Target = [T];

    fn deref(&self) -> &[T] {
        match self {
            #[cfg(target_os = "linux")]
            TensorMap::Dma(map) => map.deref(),
            #[cfg(target_os = "macos")]
            TensorMap::IoSurface(map) => map.deref(),
            #[cfg(unix)]
            TensorMap::Shm(map) => map.deref(),
            TensorMap::Mem(map) => map.deref(),
            TensorMap::Pbo(map) => map.deref(),
        }
    }
}

impl<T> DerefMut for TensorMap<T>
where
    T: Num + Clone + fmt::Debug,
{
    fn deref_mut(&mut self) -> &mut [T] {
        match self {
            #[cfg(target_os = "linux")]
            TensorMap::Dma(map) => map.deref_mut(),
            #[cfg(target_os = "macos")]
            TensorMap::IoSurface(map) => map.deref_mut(),
            #[cfg(unix)]
            TensorMap::Shm(map) => map.deref_mut(),
            TensorMap::Mem(map) => map.deref_mut(),
            TensorMap::Pbo(map) => map.deref_mut(),
        }
    }
}

// ============================================================================
// Platform availability helpers
// ============================================================================

/// Cached result of the Linux DMA-BUF availability probe.
#[cfg(target_os = "linux")]
static DMA_AVAILABLE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
/// Cached result of the macOS IOSurface availability probe.
#[cfg(target_os = "macos")]
static IOSURFACE_AVAILABLE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();

/// Check if Linux DMA-BUF allocation is available on this system.
///
/// Linux-specific availability check (typically requires `/dev/dma_heap`
/// access — running as root or membership in a video/render group). For
/// portable code that wants "any zero-copy GPU buffer", use
/// [`is_gpu_buffer_available`] which also covers IOSurface on macOS.
///
/// This function caches its result after the first call.
#[cfg(target_os = "linux")]
pub fn is_dma_available() -> bool {
    *DMA_AVAILABLE.get_or_init(|| Tensor::<u8>::new(&[64], Some(TensorMemory::Dma), None).is_ok())
}

/// Always returns `false` on non-Linux platforms.
#[cfg(not(target_os = "linux"))]
pub fn is_dma_available() -> bool {
    false
}

/// Check if macOS IOSurface allocation is available on this system.
///
/// IOSurface is part of the macOS OS and is essentially always present;
/// this probe catches degraded scenarios such as memory pressure or
/// sandboxed contexts where `IOSurfaceCreate` fails. The result is
/// cached after the first call.
///
/// Always returns `false` on non-macOS platforms.
#[cfg(target_os = "macos")]
pub fn is_iosurface_available() -> bool {
    *IOSURFACE_AVAILABLE.get_or_init(|| {
        // Probe via the same Dma path — on macOS this routes through
        // IoSurfaceTensor::new.
        Tensor::<u8>::new(&[64], Some(TensorMemory::Dma), None).is_ok()
    })
}

#[cfg(not(target_os = "macos"))]
pub fn is_iosurface_available() -> bool {
    false
}

/// Portable probe for the platform's native zero-copy GPU buffer
/// allocator (DMA-BUF on Linux, IOSurface on macOS). Returns `false` on
/// Windows and other platforms with no equivalent. Use this when writing
/// cross-platform code that cares whether the `Dma` tensor variant will
/// work, not which underlying mechanism is used.
pub fn is_gpu_buffer_available() -> bool {
    #[cfg(target_os = "linux")]
    {
        is_dma_available()
    }
    #[cfg(target_os = "macos")]
    {
        is_iosurface_available()
    }
    #[cfg(not(any(target_os = "linux", target_os = "macos")))]
    {
        false
    }
}

/// Check if POSIX shared memory allocation is available on this system.
///
/// Returns `true` on Unix systems (Linux, macOS, BSD) where POSIX shared memory
/// is supported. Always returns `false` on non-Unix platforms (Windows).
///
/// This function caches its result after the first call for efficiency.
#[cfg(unix)]
static SHM_AVAILABLE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();

/// Check if POSIX shared memory allocation is available on this system.
#[cfg(unix)]
pub fn is_shm_available() -> bool {
    *SHM_AVAILABLE.get_or_init(|| Tensor::<u8>::new(&[64], Some(TensorMemory::Shm), None).is_ok())
}

/// Check if POSIX shared memory allocation is available on this system.
///
/// Always returns `false` on non-Unix platforms since POSIX SHM is Unix-specific.
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
    fn image_shape_per_layout() {
        assert_eq!(
            PixelFormat::Rgb.image_shape(640, 480),
            Some(vec![480, 640, 3])
        );
        assert_eq!(
            PixelFormat::Grey.image_shape(640, 480),
            Some(vec![480, 640, 1])
        );
        assert_eq!(
            PixelFormat::Nv12.image_shape(640, 480),
            Some(vec![720, 640])
        );
        // Odd height: combined-plane height is `481 + ceil(481/2)` = 481 + 241
        // = 722 rows. Logical height is recovered as `722 * 2 / 3` = 481.
        assert_eq!(
            PixelFormat::Nv12.image_shape(640, 481),
            Some(vec![722, 640])
        );
        // Odd width: shape carries the LOGICAL width (641).
        // The 64-aligned stride (>= 642) is stored separately on the Tensor.
        assert_eq!(
            PixelFormat::Nv12.image_shape(641, 480),
            Some(vec![720, 641])
        );
        // NV16 odd width: same — logical width in shape, stride separate.
        assert_eq!(
            PixelFormat::Nv16.image_shape(641, 480),
            Some(vec![960, 641])
        );
        assert_eq!(
            PixelFormat::PlanarRgb.image_shape(640, 480),
            Some(vec![3, 480, 640])
        );
        assert_eq!(
            PixelFormat::Nv16.image_shape(640, 480),
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
        let t = Tensor::<u8>::image(640, 480, PixelFormat::Rgba, None).unwrap();
        assert_eq!(t.format(), Some(PixelFormat::Rgba));
        assert_eq!(t.width(), Some(640));
        assert_eq!(t.height(), Some(480));
        assert_eq!(t.shape(), &[480, 640, 4]);
        assert!(!t.is_multiplane());
    }

    #[test]
    fn image_tensor_planar() {
        let t = Tensor::<u8>::image(640, 480, PixelFormat::PlanarRgb, None).unwrap();
        assert_eq!(t.format(), Some(PixelFormat::PlanarRgb));
        assert_eq!(t.width(), Some(640));
        assert_eq!(t.height(), Some(480));
        assert_eq!(t.shape(), &[3, 480, 640]);
    }

    #[test]
    #[cfg(target_os = "macos")]
    fn image_tensor_dma_non_aligned_packed_width_pads_zero_copy() {
        // RGBA u8 at width=4 → 4*4 = 16 bytes/row, not 64-byte aligned. RGBA has
        // a real IOSurface FourCC, so an explicit `Some(TensorMemory::Dma)`
        // request now allocates a padded image IOSurface (64-aligned
        // `bytes_per_row`) and records the stride — a fully zero-copy buffer GL
        // can bind and the CPU can map via the strided path. (Previously this
        // failed loudly to avoid an 'L008' byte-bag downgrade; with a real
        // FourCC surface that concern no longer applies.)
        let t = Tensor::<u8>::image(4, 4, PixelFormat::Rgba, Some(TensorMemory::Dma))
            .expect("padded RGBA IOSurface should allocate");
        assert_eq!(t.format(), Some(PixelFormat::Rgba));
        assert_eq!(t.width(), Some(4));
        assert_eq!(t.height(), Some(4));
        let stride = t.effective_row_stride().expect("stride");
        assert_eq!(stride % 64, 0, "padded to 64-byte row alignment");
        assert!(stride >= 16);
        // A CPU map exposes the full padded surface for strided iteration.
        let m = t.map().expect("strided IOSurface map");
        assert_eq!(m.as_slice().len(), stride * 4);
    }

    /// `per_pixel_bytes` that doesn't divide 64 evenly (e.g. RGB u8 with
    /// 3 B/pixel) makes a "Pad width to N" suggestion structurally
    /// impossible — there is no integer width whose `width * 3` is a
    /// multiple of 64. The error must still fire (no silent SHM
    /// fallback for explicit-DMA requests) and must spell out the
    /// alignment requirement; it just omits the misleading "pad to N"
    /// hint instead of printing a number whose row pitch still won't
    /// align.
    #[test]
    #[cfg(target_os = "macos")]
    fn image_tensor_dma_rejects_indivisible_pixel_pitch_without_pad_hint() {
        // Width=10 RGB u8 → 30 B/row, not 64-byte aligned. The next
        // 64-multiple (64 B) isn't an integer multiple of 3 B/pixel,
        // so the "pad width to N" hint can't produce a valid number
        // and must be omitted. (Width=640 happens to align — 640*3 =
        // 1920 = 30*64 — so don't pick that for this regression
        // guard.)
        let err = Tensor::<u8>::image(10, 10, PixelFormat::Rgb, Some(TensorMemory::Dma))
            .expect_err("RGB u8 with 3 B/pixel and non-aligned width must be rejected");
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
    fn image_tensor_dma_planar_f16_alignment() {
        // PlanarRgb F16 uses single-channel row pitch (width * 2 bytes).
        // Width=16 → 32 bytes/row (not aligned); width=32 → 64 bytes/row (aligned).
        let err =
            Tensor::<half::f16>::image(16, 16, PixelFormat::PlanarRgb, Some(TensorMemory::Dma))
                .expect_err("width=16 PlanarRgb F16 is 32-byte row, must reject");
        assert!(matches!(err, Error::InvalidArgument(_)), "got {err:?}");
        // 32 wide should work.
        let t = Tensor::<half::f16>::image(32, 8, PixelFormat::PlanarRgb, Some(TensorMemory::Dma))
            .expect("width=32 PlanarRgb F16 is 64-byte row, must succeed");
        assert_eq!(t.format(), Some(PixelFormat::PlanarRgb));
    }

    #[test]
    fn image_tensor_semi_planar_contiguous() {
        let t = Tensor::<u8>::image(640, 480, PixelFormat::Nv12, None).unwrap();
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
        // Skip if DMA not available (e.g. sandboxed CI lacking dma_heap access).
        if !is_dma_available() {
            eprintln!("SKIPPED: DMA heap not available");
            return;
        }
        // 3004×1688 RGBA8: natural pitch 12016, padded to 12032 (64-aligned).
        let stride = 12032;
        let t = Tensor::<u8>::image_with_stride(
            3004,
            1688,
            PixelFormat::Rgba,
            stride,
            Some(TensorMemory::Dma),
        )
        .unwrap();
        // Logical dimensions unchanged by padding — this is the contract.
        assert_eq!(t.width(), Some(3004));
        assert_eq!(t.height(), Some(1688));
        assert_eq!(t.shape(), &[1688, 3004, 4]);
        // Stride is carried separately and reports the padded pitch.
        assert_eq!(t.effective_row_stride(), Some(stride));
        // Buffer is sized to stride × height so the full padded layout fits,
        // and CPU map() works for self-allocated strided DMA tensors.
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
        // CPU write access works too — iterate rows using the padded stride,
        // touch only the active `width × bpp` region, verify it round-trips.
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
            // Sample a few pixels to confirm the round-trip.
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
    fn image_tensor_with_stride_rejects_foreign_strided_map() {
        // A FOREIGN (imported via from_fd) DMA tensor with row_stride set
        // should still refuse CPU mapping — external allocator owns the
        // layout. This protects the V4L2 / GStreamer use case.
        //
        // We simulate a foreign import by wrapping our own allocation's
        // fd via `from_fd` and calling set_row_stride manually. The
        // `is_imported` flag on from_fd is true by construction.
        if !is_dma_available() {
            eprintln!("SKIPPED: DMA heap not available");
            return;
        }
        // Allocate a backing buffer large enough for a 320×240 BGRA8 image.
        let backing = Tensor::<u8>::new(&[240 * 320 * 4], Some(TensorMemory::Dma), None).unwrap();
        let fd = backing.clone_fd().unwrap();
        // Import it via from_fd — this marks is_imported=true.
        let shape = [240usize, 320, 4];
        let storage = TensorStorage::<u8>::from_fd(fd, &shape, None).unwrap();
        let mut t = Tensor::<u8>::wrap(storage);
        t.set_format(PixelFormat::Bgra).unwrap();
        t.set_row_stride(320 * 4).unwrap(); // natural, but still marks it as strided
        let err = t.map();
        assert!(
            matches!(err, Err(Error::InvalidOperation(_))),
            "foreign strided map should error"
        );
    }

    #[test]
    #[cfg(target_os = "linux")]
    fn image_tensor_with_stride_map_rejects_tampered_stride() {
        // Round-3 PR feedback (C1): `set_row_stride` is public and only
        // validates `stride >= min_stride`, not that the new stride × height
        // fits the underlying buffer. A caller that tampers with the stride
        // after allocation must not be able to coerce `Tensor::map()` into
        // returning a slice larger than the backing mmap (that would be UB
        // in `DmaMap::as_slice`).
        if !is_dma_available() {
            eprintln!("SKIPPED: DMA heap not available");
            return;
        }
        // Allocate a 640×480 RGBA8 padded canvas (stride = 3072 = 768 px).
        // Backing buffer is 3072 × 480 = 1,474,560 bytes.
        let mut t = Tensor::<u8>::image_with_stride(
            640,
            480,
            PixelFormat::Rgba,
            3072,
            Some(TensorMemory::Dma),
        )
        .unwrap();
        // Tamper: push the stride up to 4 × the original. This is >=
        // min_stride (2560), so `set_row_stride` accepts it.
        t.set_row_stride(12288).unwrap();
        // Map must now refuse — 12288 × 480 = 5,898,240 > 1,474,560.
        let err = t.map();
        assert!(
            matches!(err, Err(Error::InvalidOperation(_))),
            "map() with oversized stride must return InvalidOperation"
        );
    }

    #[test]
    fn dma_tensor_new_with_byte_size_rejects_shape_overflow() {
        // Round-3 PR feedback (C3): shape.product() * sizeof(T) must use
        // checked arithmetic so a pathological shape can't wrap usize and
        // make the byte_size-vs-logical-size comparison incorrect.
        //
        // This test only exercises the overflow rejection path, which is
        // pure-Rust and doesn't touch dma_heap — safe to run on any target.
        #[cfg(target_os = "linux")]
        {
            let err = crate::dma::DmaTensor::<u64>::new_with_byte_size(
                &[usize::MAX, 2, 2],
                usize::MAX,
                None,
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
        // 640×480 RGBA8 natural pitch = 2560, request 2400 → should error.
        let err = Tensor::<u8>::image_with_stride(
            640,
            480,
            PixelFormat::Rgba,
            2400,
            Some(TensorMemory::Dma),
        );
        assert!(matches!(err, Err(Error::InvalidArgument(_))));
    }

    #[test]
    #[cfg(target_os = "linux")]
    fn image_tensor_with_stride_rejects_non_packed() {
        // NV12 is SemiPlanar → not supported. (Linux-only because
        // `TensorMemory::Dma` itself is a Linux-only enum variant.)
        let err = Tensor::<u8>::image_with_stride(
            640,
            480,
            PixelFormat::Nv12,
            640,
            Some(TensorMemory::Dma),
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
        // Original tensor is unmodified
        assert!(t.format().is_none());
    }

    #[test]
    fn reshape_clears_format() {
        let mut t = Tensor::<u8>::image(640, 480, PixelFormat::Rgba, None).unwrap();
        assert_eq!(t.format(), Some(PixelFormat::Rgba));
        // Reshape to flat — format cleared
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
mod tests {
    #[cfg(target_os = "linux")]
    use nix::unistd::{access, AccessFlags};
    #[cfg(target_os = "linux")]
    use std::io::Write as _;
    use std::sync::RwLock;

    use super::*;

    #[ctor::ctor]
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
        let _lock = FD_LOCK.read().unwrap();
        let shape = vec![1];
        let tensor = DmaTensor::<f32>::new(&shape, Some("dma_tensor"));
        let dma_enabled = tensor.is_ok();

        let tensor = Tensor::<f32>::new(&shape, None, None).expect("Failed to create tensor");
        match dma_enabled {
            true => assert_eq!(tensor.memory(), TensorMemory::Dma),
            false => assert_eq!(tensor.memory(), TensorMemory::Shm),
        }
    }

    #[test]
    #[cfg(target_os = "macos")]
    fn test_tensor() {
        let shape = vec![1];
        let tensor = Tensor::<f32>::new(&shape, None, None).expect("Failed to create tensor");
        // macOS auto-fallback chain: IOSurface (Dma) → SHM → Mem.
        // Healthy systems always return Dma; SHM/Mem only appear under
        // memory pressure or sandboxed contexts where IOSurfaceCreate
        // fails.
        let m = tensor.memory();
        assert!(
            matches!(m, TensorMemory::Dma | TensorMemory::Shm | TensorMemory::Mem),
            "Unexpected auto-fallback result on macOS: {m:?}"
        );
    }

    #[test]
    #[cfg(all(unix, not(any(target_os = "linux", target_os = "macos"))))]
    fn test_tensor() {
        let shape = vec![1];
        let tensor = Tensor::<f32>::new(&shape, None, None).expect("Failed to create tensor");
        // Other Unix (BSD): auto-detection tries SHM first, falls back to Mem.
        assert!(
            tensor.memory() == TensorMemory::Shm || tensor.memory() == TensorMemory::Mem,
            "Expected SHM or Mem, got {:?}",
            tensor.memory()
        );
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
        let _lock = FD_LOCK.read().unwrap();
        match access(
            "/dev/dma_heap/linux,cma",
            AccessFlags::R_OK | AccessFlags::W_OK,
        ) {
            Ok(_) => println!("/dev/dma_heap/linux,cma is available"),
            Err(_) => match access(
                "/dev/dma_heap/system",
                AccessFlags::R_OK | AccessFlags::W_OK,
            ) {
                Ok(_) => println!("/dev/dma_heap/system is available"),
                Err(e) => {
                    writeln!(
                        &mut std::io::stdout(),
                        "[WARNING] DMA Heap is unavailable: {e}"
                    )
                    .unwrap();
                    return;
                }
            },
        }

        let shape = vec![2, 3, 4];
        let tensor =
            DmaTensor::<f32>::new(&shape, Some("test_tensor")).expect("Failed to create tensor");

        const DUMMY_VALUE: f32 = 12.34;

        assert_eq!(tensor.memory(), TensorMemory::Dma);
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

            assert_eq!(shared.memory(), TensorMemory::Dma);
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
        let _lock = FD_LOCK.read().unwrap();
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
        // One heap [2,4] u8 parent (8 bytes). Two [1,4] sub-views at byte
        // offsets 0 and 4 must share the parent allocation (zero-copy) and be
        // independently writable: view 0 owns bytes [0,4), view 1 owns [4,8).
        // Today this is impossible — heap offset is rejected and there is no
        // shared sub-view constructor.
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

        // Each view sees only its own window.
        assert_eq!(view0.map().unwrap().as_slice(), &[1, 2, 3, 4]);
        assert_eq!(view1.map().unwrap().as_slice(), &[10, 20, 30, 40]);
        // The parent buffer is correctly partitioned (shared, zero-copy).
        assert_eq!(
            parent.map().unwrap().as_slice(),
            &[1, 2, 3, 4, 10, 20, 30, 40]
        );
    }

    #[test]
    fn mem_subview_rejects_unaligned_offset() {
        // f32 has align 4; a byte offset of 2 cannot back a valid `*const f32`.
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

    #[test]
    fn mem_subview_four_views_no_aliasing() {
        // One [4,3] f32 parent; four [1,3] views at 12-byte strides, each
        // written independently. Exercises a multi-byte element type (offsets
        // must stay element-aligned) and N-way zero-copy sharing.
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
        // A sub-view is a ready-to-use sub-image: it inherits the parent's
        // pixel format and (crucially) its padded row stride, so a strided
        // parent yields strided windows. Set a stride wider than the tight row
        // to exercise the row_stride inheritance path specifically.
        let mut parent =
            Tensor::<u8>::image(100, 100, PixelFormat::Rgba, Some(TensorMemory::Mem)).unwrap();
        parent.set_row_stride_unchecked(512); // padded stride (> 100*4)
        let view = parent.subview(4096, &[10, 10, 4]).unwrap();
        assert_eq!(view.format(), Some(PixelFormat::Rgba), "format inherited");
        assert_eq!(view.row_stride(), Some(512), "row_stride inherited");
    }

    #[test]
    fn mem_strided_subview_maps_offset_and_byte_size() {
        // Integration of the sub-region offset (PR #89) and the strided-map
        // `byte_size_override` (PR #90): a strided sub-view exposes its full
        // padded window (`row_stride × rows`) starting at the view's byte
        // offset, mapped zero-copy into the parent.
        let parent = Tensor::<u8>::new(&[2048], Some(TensorMemory::Mem), None).unwrap();
        let mut view = parent.subview(128, &[8, 16]).unwrap(); // 8 rows × 16 @ off 128
        assert_eq!(view.plane_offset(), Some(128));
        view.set_row_stride_unchecked(32); // padded stride (> 16)

        {
            let mut m = view.map().unwrap();
            let s = m.as_mut_slice();
            // Strided map exposes the padded window: stride(32) × rows(8) = 256.
            assert_eq!(
                s.len(),
                256,
                "strided map exposes the full padded byte window"
            );
            s[0] = 0xAA; // row 0, col 0
            s[32] = 0xBB; // row 1, col 0 (one stride in)
        }

        // Zero-copy: the writes land in the parent at the view's offset.
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
    fn subview_rejects_unsupported_storage() {
        // subview shares either a heap `Arc` (Mem) or a dma-buf fd (Dma); any
        // other backing (here Shm) must be refused with InvalidOperation rather
        // than silently mishandled.
        if !crate::is_shm_available() {
            eprintln!("SKIPPED: shm not available");
            return;
        }
        let shm = Tensor::<u8>::new(&[64], Some(TensorMemory::Shm), None).unwrap();
        match shm.subview(0, &[4]) {
            Err(Error::InvalidOperation(_)) => {}
            Err(other) => panic!("expected InvalidOperation, got {other:?}"),
            Ok(_) => panic!("subview must reject Shm storage"),
        }
    }

    #[test]
    #[cfg(target_os = "linux")]
    fn dma_subview_matches_mem_subview() {
        // Serialize against the fd-leak tests: this test opens DMA fds (alloc +
        // clone_fd), which would otherwise perturb their fd counts.
        let _lock = FD_LOCK.read().unwrap();
        // Identical sub-view semantics across Dma (shared fd) and Mem (shared
        // Arc): same offsets → same logical windows → same partition.
        let dma = match Tensor::<u8>::new(&[8], Some(TensorMemory::Dma), None) {
            Ok(t) => t,
            Err(_) => {
                eprintln!("SKIPPED: DMA not available");
                return;
            }
        };
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
        // The strided-map path differs by backing: DMA maps through
        // `mmap_offset` + the `byte_size_override`, not the Mem `Arc` slice. A
        // padded sub-view of a DMA buffer must still expose its full
        // `row_stride × rows` window zero-copy at the view's offset (the GPU
        // batched-render-to-DMA case). Mirrors
        // `mem_strided_subview_maps_offset_and_byte_size` on a Dma parent.
        let _lock = FD_LOCK.read().unwrap();
        let parent = match Tensor::<u8>::new(&[2048], Some(TensorMemory::Dma), None) {
            Ok(t) => t,
            Err(_) => {
                eprintln!("SKIPPED: DMA not available");
                return;
            }
        };
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
        let _lock = FD_LOCK.write().unwrap();
        if !is_dma_available() {
            log::warn!(
                "SKIPPED: {} - DMA memory allocation not available (permission denied or no DMA-BUF support)",
                function!()
            );
            return;
        }

        let proc = procfs::process::Process::myself()
            .expect("Failed to get current process using /proc/self");

        let start_open_fds = proc
            .fd_count()
            .expect("Failed to get open file descriptor count");

        for _ in 0..100 {
            let tensor = Tensor::<u8>::new(&[100, 100], Some(TensorMemory::Dma), None)
                .expect("Failed to create tensor");
            let mut map = tensor.map().unwrap();
            map.as_mut_slice().fill(233);
        }

        let end_open_fds = proc
            .fd_count()
            .expect("Failed to get open file descriptor count");

        assert_eq!(
            start_open_fds, end_open_fds,
            "File descriptor leak detected: {} -> {}",
            start_open_fds, end_open_fds
        );
    }

    #[test]
    #[cfg(target_os = "linux")]
    fn test_dma_from_fd_no_fd_leaks() {
        let _lock = FD_LOCK.write().unwrap();
        if !is_dma_available() {
            log::warn!(
                "SKIPPED: {} - DMA memory allocation not available (permission denied or no DMA-BUF support)",
                function!()
            );
            return;
        }

        let proc = procfs::process::Process::myself()
            .expect("Failed to get current process using /proc/self");

        let start_open_fds = proc
            .fd_count()
            .expect("Failed to get open file descriptor count");

        let orig = Tensor::<u8>::new(&[100, 100], Some(TensorMemory::Dma), None).unwrap();

        for _ in 0..100 {
            let tensor =
                Tensor::<u8>::from_fd(orig.clone_fd().unwrap(), orig.shape(), None).unwrap();
            let mut map = tensor.map().unwrap();
            map.as_mut_slice().fill(233);
        }
        drop(orig);

        let end_open_fds = proc.fd_count().unwrap();

        assert_eq!(
            start_open_fds, end_open_fds,
            "File descriptor leak detected: {} -> {}",
            start_open_fds, end_open_fds
        );
    }

    #[test]
    #[cfg(target_os = "linux")]
    fn test_shm_no_fd_leaks() {
        let _lock = FD_LOCK.write().unwrap();
        if !is_shm_available() {
            log::warn!(
                "SKIPPED: {} - SHM memory allocation not available (permission denied or no SHM support)",
                function!()
            );
            return;
        }

        let proc = procfs::process::Process::myself()
            .expect("Failed to get current process using /proc/self");

        let start_open_fds = proc
            .fd_count()
            .expect("Failed to get open file descriptor count");

        for _ in 0..100 {
            let tensor = Tensor::<u8>::new(&[100, 100], Some(TensorMemory::Shm), None)
                .expect("Failed to create tensor");
            let mut map = tensor.map().unwrap();
            map.as_mut_slice().fill(233);
        }

        let end_open_fds = proc
            .fd_count()
            .expect("Failed to get open file descriptor count");

        assert_eq!(
            start_open_fds, end_open_fds,
            "File descriptor leak detected: {} -> {}",
            start_open_fds, end_open_fds
        );
    }

    #[test]
    #[cfg(target_os = "linux")]
    fn test_shm_from_fd_no_fd_leaks() {
        let _lock = FD_LOCK.write().unwrap();
        if !is_shm_available() {
            log::warn!(
                "SKIPPED: {} - SHM memory allocation not available (permission denied or no SHM support)",
                function!()
            );
            return;
        }

        let proc = procfs::process::Process::myself()
            .expect("Failed to get current process using /proc/self");

        let start_open_fds = proc
            .fd_count()
            .expect("Failed to get open file descriptor count");

        let orig = Tensor::<u8>::new(&[100, 100], Some(TensorMemory::Shm), None).unwrap();

        for _ in 0..100 {
            let tensor =
                Tensor::<u8>::from_fd(orig.clone_fd().unwrap(), orig.shape(), None).unwrap();
            let mut map = tensor.map().unwrap();
            map.as_mut_slice().fill(233);
        }
        drop(orig);

        let end_open_fds = proc.fd_count().unwrap();

        assert_eq!(
            start_open_fds, end_open_fds,
            "File descriptor leak detected: {} -> {}",
            start_open_fds, end_open_fds
        );
    }

    #[cfg(feature = "ndarray")]
    #[test]
    fn test_ndarray() {
        let _lock = FD_LOCK.read().unwrap();
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
    fn test_buffer_identity_unique() {
        let id1 = BufferIdentity::new();
        let id2 = BufferIdentity::new();
        assert_ne!(
            id1.id(),
            id2.id(),
            "Two identities should have different ids"
        );
    }

    #[test]
    fn test_buffer_identity_clone_shares_guard() {
        let id1 = BufferIdentity::new();
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
        // len(scales) != len(zero_points) → rejected at construction.
        let err = Quantization::per_channel(vec![0.1, 0.2], vec![0, 0, 0], 0).unwrap_err();
        assert!(matches!(err, Error::QuantizationInvalid { .. }));
    }

    #[test]
    fn test_quantization_per_channel_empty_rejected() {
        let err = Quantization::per_channel_symmetric(vec![], 0).unwrap_err();
        assert!(matches!(err, Error::QuantizationInvalid { .. }));
    }

    /// Constructors guard scale/zero_point length invariants, but
    /// `Quantization` is `Deserialize`, so malformed JSON (e.g. an
    /// empty `scale` array, or `zero_point` length that disagrees with
    /// `scale`) bypasses the constructor checks. `set_quantization`
    /// must reject these via `validate()` so they don't poison
    /// downstream `mode()` selection or per-channel kernel indexing.
    #[test]
    fn test_quantization_validate_rejects_malformed_deserialize() {
        let mut t = Tensor::<i8>::new(&[1, 1, 4], Some(TensorMemory::Mem), None).unwrap();

        // Empty scale array: must be rejected.
        let q: Quantization = serde_json::from_str(r#"{"scale": []}"#).unwrap();
        assert!(matches!(
            t.set_quantization(q).unwrap_err(),
            Error::QuantizationInvalid { .. }
        ));

        // Per-tensor with multi-element zero_point: must be rejected.
        let q: Quantization =
            serde_json::from_str(r#"{"scale": 0.1, "zero_point": [0, 0, 0]}"#).unwrap();
        assert!(matches!(
            t.set_quantization(q).unwrap_err(),
            Error::QuantizationInvalid { .. }
        ));

        // Per-channel zero_point length != scale length: must be rejected.
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
        // float path returns a QuantizationInvalid error.
        assert!(matches!(err, Error::QuantizationInvalid { .. }));
    }

    /// Compile-time type gate — calling `Tensor::<f32>::quantization()` must
    /// fail to compile (the `IntegerType` trait bound is not satisfied by
    /// `f32`). This doctest anchors the invariant.
    ///
    /// ```compile_fail
    /// use edgefirst_tensor::{Tensor, TensorMemory};
    /// let t = Tensor::<f32>::new(&[2, 2], Some(TensorMemory::Mem), None).unwrap();
    /// let _ = t.quantization(); // compile error: f32 not IntegerType
    /// ```
    fn _compile_fail_doctest_anchor() {}

    // Any test that cares about the fd count must grab it exclusively.
    // Any tests which modifies the fd count by opening or closing fds must grab it
    // shared.
    pub static FD_LOCK: RwLock<()> = RwLock::new(());

    /// Test that DMA is NOT available on non-Linux platforms.
    /// This verifies the cross-platform behavior of is_dma_available().
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
        let mut t =
            Tensor::<u8>::image(1280, 720, PixelFormat::Nv12, Some(TensorMemory::Mem)).unwrap();
        assert_eq!(t.colorimetry(), None); // default undefined
        let c = Colorimetry::default()
            .with_encoding(ColorEncoding::Bt709)
            .with_range(ColorRange::Limited);
        t.set_colorimetry(Some(c));
        assert_eq!(t.colorimetry(), Some(c));
        // configure_image must NOT touch colorimetry
        t.configure_image(640, 480, PixelFormat::Grey).unwrap();
        assert_eq!(t.colorimetry(), Some(c));
    }

    #[test]
    fn configure_image_within_capacity() {
        let mut t = Tensor::<u8>::image_with_capacity(640, 480, PixelFormat::Rgb, None).unwrap();
        t.configure_image(320, 240, PixelFormat::Nv12).unwrap();
        assert_eq!(t.format(), Some(PixelFormat::Nv12));
        assert_eq!(t.width(), Some(320));
        assert_eq!(t.height(), Some(240));
        assert_eq!(t.shape(), &[360, 320]); // 240*3/2
    }

    #[test]
    fn configure_image_too_large_errors() {
        let mut t = Tensor::<u8>::image_with_capacity(64, 64, PixelFormat::Grey, None).unwrap();
        let err = t
            .configure_image(1920, 1080, PixelFormat::Nv12)
            .unwrap_err();
        assert!(matches!(err, Error::InsufficientCapacity { .. }));
    }

    /// A reused max-sized IOSurface pool keeps its physical `bytesPerRow` when
    /// reconfigured to a smaller logical image (physical-grid / logical-ROI
    /// decoupling), instead of collapsing to the frame's natural row stride.
    #[test]
    #[cfg(target_os = "macos")]
    fn configure_image_preserves_iosurface_physical_stride() {
        // Pool: GREY/R8 IOSurface 100 wide → bytesPerRow padded to 128.
        let mut pool =
            Tensor::<u8>::image(100, 64, PixelFormat::Grey, Some(TensorMemory::Dma)).unwrap();
        let pitch = pool.effective_row_stride().unwrap();
        assert!(
            pitch >= 128 && pitch.is_multiple_of(64),
            "padded bytesPerRow, got {pitch}"
        );

        // Reconfigure to a smaller NV12 frame; the physical pitch must survive
        // (natural would be 32, but the surface stride is the 128-padded pitch).
        pool.configure_image(32, 16, PixelFormat::Nv12).unwrap();
        assert_eq!(pool.format(), Some(PixelFormat::Nv12));
        assert_eq!(pool.width(), Some(32));
        assert_eq!(pool.height(), Some(16));
        assert_eq!(
            pool.effective_row_stride(),
            Some(pitch),
            "configure_image must preserve the IOSurface physical bytesPerRow"
        );

        // Reconfigure again to NV24 — pitch still preserved.
        pool.configure_image(32, 16, PixelFormat::Nv24).unwrap();
        assert_eq!(pool.effective_row_stride(), Some(pitch));
    }

    /// `configure_image` on a Mem backing reconfigures to the format's
    /// **64-byte-aligned** row stride (the odd-dim contract: every image tensor
    /// carries a 64-aligned `row_stride`). For NV12 32×16 the minimum is
    /// `even(32)=32`, rounded up to the 64-byte alignment → 64. The capacity
    /// (64×64×4 RGBA = 16 KiB) easily holds the 24×64 = 1.5 KiB NV12 layout.
    #[test]
    fn configure_image_mem_aligns_stride() {
        let mut t =
            Tensor::<u8>::image_with_capacity(64, 64, PixelFormat::Rgba, Some(TensorMemory::Mem))
                .unwrap();
        t.configure_image(32, 16, PixelFormat::Nv12).unwrap();
        let s = t.effective_row_stride().unwrap();
        assert_eq!(s % 64, 0, "stride must be 64-aligned");
        assert!(s >= 32, "stride must cover the even-width minimum");
        assert_eq!(s, 64);
    }

    #[test]
    fn strided_mem_tensor_cpu_maps_full_padded_buffer() {
        // A packed RGBA image with row padding (GPU-pitch style): logical width
        // 8 px (32 B/row) but a 48-byte row stride. Over-allocate capacity (for
        // 16 px), narrow the logical width, then record the padded stride.
        // Previously `map()` rejected this on non-Linux with
        // "DMA backing is Linux-only"; HAL-owned Mem is now mappable.
        let mut t =
            Tensor::<u8>::image_with_capacity(16, 3, PixelFormat::Rgba, Some(TensorMemory::Mem))
                .unwrap(); // capacity 3 × 16 × 4 = 192 B
        t.configure_image(8, 3, PixelFormat::Rgba).unwrap(); // logical [3, 8, 4] = 96 B
        t.set_row_stride(48).unwrap(); // padded stride (>= 32 B min)

        let map = t.map().expect("strided Mem tensor should CPU-map");
        // Full padded buffer (stride 48 × 3 rows = 144 B), not the 96 B logical
        // view — callers iterate rows via `effective_row_stride()`.
        assert_eq!(map.as_slice().len(), 144);
        // Logical shape is still reported for shape-aware consumers.
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

    /// Test that SHM memory allocation is available and usable on Unix systems.
    /// This is a basic functional test; Linux has additional FD leak tests using procfs.
    #[test]
    #[cfg(unix)]
    fn test_shm_available_and_usable() {
        assert!(
            is_shm_available(),
            "SHM memory allocation should be available on Unix systems"
        );

        // Create a tensor with SHM backing
        let tensor = Tensor::<u8>::new(&[100, 100], Some(TensorMemory::Shm), None)
            .expect("Failed to create SHM tensor");

        // Verify we can map and write to it
        let mut map = tensor.map().expect("Failed to map SHM tensor");
        map.as_mut_slice().fill(0xAB);

        // Verify the data was written correctly
        assert!(
            map.as_slice().iter().all(|&b| b == 0xAB),
            "SHM tensor data should be writable and readable"
        );
    }

    // =========================================================================
    // packed_rgba16f_layout — host-runnable geometry unit tests (TDD)
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
        // Packed Rgba with F16 is not a planar format → None
        assert!(packed_rgba16f_layout(PixelFormat::Rgba, DType::F16, 640, 640).is_none());
    }

    #[test]
    fn cuda_map_fast_fails_to_none_without_handle() {
        let t = Tensor::<f32>::new(&[4], Some(TensorMemory::Mem), None).unwrap();
        assert!(t.cuda().is_none());
        assert!(t.cuda_map().is_none()); // pure local check, no GL routing
    }

    #[test]
    fn cuda_returns_none_without_handle() {
        // A plain Mem-backed tensor has no CUDA handle attached.
        let t = Tensor::<f32>::new(&[2, 2], Some(TensorMemory::Mem), None).unwrap();
        assert!(t.cuda().is_none(), "no CUDA handle on a Mem tensor");
        assert!(t.cuda_map().is_none(), "fast-fail map → None");
    }

    #[test]
    fn cuda_map_then_host_map_fallback() {
        // The documented client pattern: try cuda_map() first; when it is None
        // (no CUDA handle — the case for a plain Mem tensor), fall back to map().
        let t = Tensor::<f32>::new(&[2, 2], Some(TensorMemory::Mem), None).unwrap();
        // Bind to a named variable so the CudaMap guard (and its borrow of `t`)
        // is dropped at the end of this statement, before the else branch borrows `t` again.
        let cuda = t.cuda_map();
        if let Some(_c) = cuda {
            // On a CUDA-registered tensor we'd use the device ptr here.
            unreachable!("a Mem tensor has no CUDA handle");
        } else {
            let host = t.map().expect("host map fallback must succeed");
            // TensorMapTrait::len() returns the element count (not bytes).
            assert_eq!(host.len(), 4); // 2*2 f32 elements
        }
    }
}
