// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! Pixel type trait for image decoding.
//!
//! Types implementing [`ImagePixel`] can be used as the element type of a
//! [`Tensor`](edgefirst_tensor::Tensor) passed to
//! [`ImageLoad::load_image`](crate::ImageLoad::load_image).

use edgefirst_tensor::DType;
use std::fmt;

/// Marker trait for pixel element types supported by image decoding.
///
/// Provides the conversion from a decoded `u8` pixel value into the tensor's
/// native element type.
pub trait ImagePixel: num_traits::Num + Clone + fmt::Debug + Send + Sync + 'static {
    /// Convert a `[0, 255]` byte value to this pixel type.
    fn from_u8(v: u8) -> Self;

    /// The [`DType`] that corresponds to this Rust type.
    fn dtype() -> DType;
}

impl ImagePixel for u8 {
    #[inline]
    fn from_u8(v: u8) -> Self {
        v
    }

    fn dtype() -> DType {
        DType::U8
    }
}

impl ImagePixel for f32 {
    #[inline]
    fn from_u8(v: u8) -> Self {
        v as f32 / 255.0
    }

    fn dtype() -> DType {
        DType::F32
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn u8_identity() {
        assert_eq!(u8::from_u8(0), 0);
        assert_eq!(u8::from_u8(128), 128);
        assert_eq!(u8::from_u8(255), 255);
    }

    #[test]
    fn f32_normalised() {
        assert!((f32::from_u8(0) - 0.0).abs() < f32::EPSILON);
        assert!((f32::from_u8(255) - 1.0).abs() < f32::EPSILON);
        assert!((f32::from_u8(128) - 128.0 / 255.0).abs() < f32::EPSILON);
    }

    #[test]
    fn dtype_matches() {
        assert_eq!(u8::dtype(), DType::U8);
        assert_eq!(f32::dtype(), DType::F32);
    }
}
