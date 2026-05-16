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
/// Provides conversions from decoded `u8` and `u16` pixel values into the
/// tensor's native element type. The `from_u16` path is used for 16-bit PNG
/// images; JPEG always decodes to `u8`.
///
/// ## Supported types and conversions
///
/// | Type | `from_u8` | `from_u16` | Notes |
/// |------|-----------|------------|-------|
/// | `u8` | identity | `>> 8` | |
/// | `u16` | `* 257` | identity | 0..255 → 0..65535 |
/// | `i8` | `XOR 0x80` | `(>> 8) XOR 0x80` | unsigned-to-signed via sign-bit flip |
/// | `i16` | `* 257 XOR 0x8000` | `XOR 0x8000` | unsigned-to-signed via sign-bit flip |
/// | `f32` | `/ 255.0` | `/ 65535.0` | normalised to `[0.0, 1.0]` |
pub trait ImagePixel: num_traits::Num + Clone + fmt::Debug + Send + Sync + 'static {
    /// Convert a `[0, 255]` byte value to this pixel type.
    fn from_u8(v: u8) -> Self;

    /// Convert a `[0, 65535]` 16-bit value to this pixel type.
    ///
    /// Used for 16-bit PNG images. The default implementation converts via
    /// `from_u8(v >> 8)` which discards the low byte.
    fn from_u16(v: u16) -> Self {
        Self::from_u8((v >> 8) as u8)
    }

    /// The [`DType`] that corresponds to this Rust type.
    fn dtype() -> DType;
}

impl ImagePixel for u8 {
    #[inline]
    fn from_u8(v: u8) -> Self {
        v
    }

    #[inline]
    fn from_u16(v: u16) -> Self {
        (v >> 8) as u8
    }

    fn dtype() -> DType {
        DType::U8
    }
}

impl ImagePixel for u16 {
    #[inline]
    fn from_u8(v: u8) -> Self {
        // Scale 0..255 → 0..65535 exactly: 0→0, 128→32896, 255→65535
        v as u16 * 257
    }

    #[inline]
    fn from_u16(v: u16) -> Self {
        v
    }

    fn dtype() -> DType {
        DType::U16
    }
}

impl ImagePixel for i8 {
    #[inline]
    fn from_u8(v: u8) -> Self {
        // XOR sign-bit flip: 0→-128, 128→0, 255→127
        (v ^ 0x80) as i8
    }

    #[inline]
    fn from_u16(v: u16) -> Self {
        Self::from_u8((v >> 8) as u8)
    }

    fn dtype() -> DType {
        DType::I8
    }
}

impl ImagePixel for i16 {
    #[inline]
    fn from_u8(v: u8) -> Self {
        // Scale to u16 range first, then XOR sign-bit flip
        ((v as u16 * 257) ^ 0x8000) as i16
    }

    #[inline]
    fn from_u16(v: u16) -> Self {
        // XOR sign-bit flip: 0→-32768, 32768→0, 65535→32767
        (v ^ 0x8000) as i16
    }

    fn dtype() -> DType {
        DType::I16
    }
}

impl ImagePixel for f32 {
    #[inline]
    fn from_u8(v: u8) -> Self {
        v as f32 / 255.0
    }

    #[inline]
    fn from_u16(v: u16) -> Self {
        v as f32 / 65535.0
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
    fn u8_from_u16() {
        assert_eq!(u8::from_u16(0), 0);
        assert_eq!(u8::from_u16(32768), 128);
        assert_eq!(u8::from_u16(65535), 255);
    }

    #[test]
    fn u16_from_u8_scaling() {
        assert_eq!(u16::from_u8(0), 0);
        assert_eq!(u16::from_u8(1), 257);
        assert_eq!(u16::from_u8(128), 32896);
        assert_eq!(u16::from_u8(255), 65535);
    }

    #[test]
    fn u16_identity() {
        assert_eq!(u16::from_u16(0), 0);
        assert_eq!(u16::from_u16(32768), 32768);
        assert_eq!(u16::from_u16(65535), 65535);
    }

    #[test]
    fn i8_xor_trick() {
        // XOR with 0x80 flips the sign bit: u8 midpoint (128) maps to i8 zero
        assert_eq!(i8::from_u8(0), -128);
        assert_eq!(i8::from_u8(128), 0);
        assert_eq!(i8::from_u8(255), 127);
        assert_eq!(i8::from_u8(1), -127);
        assert_eq!(i8::from_u8(127), -1);
    }

    #[test]
    fn i16_xor_trick() {
        // From u8: scale to u16 first, then XOR
        assert_eq!(i16::from_u8(0), -32768);
        assert_eq!(i16::from_u8(128), 128); // 32896 ^ 0x8000 = 128
        assert_eq!(i16::from_u8(255), 32767);

        // From u16: direct XOR
        assert_eq!(i16::from_u16(0), -32768);
        assert_eq!(i16::from_u16(32768), 0);
        assert_eq!(i16::from_u16(65535), 32767);
    }

    #[test]
    fn f32_normalised() {
        assert!((f32::from_u8(0) - 0.0).abs() < f32::EPSILON);
        assert!((f32::from_u8(255) - 1.0).abs() < f32::EPSILON);
        assert!((f32::from_u8(128) - 128.0 / 255.0).abs() < f32::EPSILON);
    }

    #[test]
    fn f32_from_u16_normalised() {
        assert!((f32::from_u16(0) - 0.0).abs() < f32::EPSILON);
        assert!((f32::from_u16(65535) - 1.0).abs() < 1e-5);
        assert!((f32::from_u16(32768) - 32768.0 / 65535.0).abs() < 1e-5);
    }

    #[test]
    fn dtype_matches() {
        assert_eq!(u8::dtype(), DType::U8);
        assert_eq!(u16::dtype(), DType::U16);
        assert_eq!(i8::dtype(), DType::I8);
        assert_eq!(i16::dtype(), DType::I16);
        assert_eq!(f32::dtype(), DType::F32);
    }
}
