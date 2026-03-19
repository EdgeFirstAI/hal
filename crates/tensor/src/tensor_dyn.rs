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

    /// Clone the file descriptor associated with this tensor.
    #[cfg(unix)]
    pub fn clone_fd(&self) -> crate::Result<std::os::fd::OwnedFd> {
        dispatch!(self, clone_fd)
    }

    /// Return `true` if this tensor uses separate plane allocations.
    pub fn is_multiplane(&self) -> bool {
        dispatch!(self, is_multiplane)
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
}
