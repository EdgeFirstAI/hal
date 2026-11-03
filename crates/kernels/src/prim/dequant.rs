// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! Scalar per-tensor affine dequantization kernels.

use super::DequantKernel;
use crate::Quant;
use half::f16;

// Concrete cells (i8/u8/i16/u16/f16/f32 × f32/f16).

/// Dequantizes `i8` to `f32`: `output[i] = (input[i] − zero_point) · scale`.
///
/// # Panics
/// In debug builds, when `input` and `output` differ in length; release builds convert the
/// shorter length.
#[inline]
pub fn dequant_i8_to_f32(input: &[i8], q: Quant, output: &mut [f32]) {
    debug_assert_eq!(input.len(), output.len());
    let scale = q.scale;
    let zp = q.zero_point as f32;
    for (src, dst) in input.iter().zip(output.iter_mut()) {
        *dst = (*src as f32 - zp) * scale;
    }
}

/// Dequantizes `u8` to `f32`: `output[i] = (input[i] − zero_point) · scale`.
///
/// # Panics
/// In debug builds, when `input` and `output` differ in length; release builds convert the
/// shorter length.
#[inline]
pub fn dequant_u8_to_f32(input: &[u8], q: Quant, output: &mut [f32]) {
    debug_assert_eq!(input.len(), output.len());
    let scale = q.scale;
    let zp = q.zero_point as f32;
    for (src, dst) in input.iter().zip(output.iter_mut()) {
        *dst = (*src as f32 - zp) * scale;
    }
}

/// Dequantizes `i16` to `f32`: `output[i] = (input[i] − zero_point) · scale`.
///
/// # Panics
/// In debug builds, when `input` and `output` differ in length; release builds convert the
/// shorter length.
#[inline]
pub fn dequant_i16_to_f32(input: &[i16], q: Quant, output: &mut [f32]) {
    debug_assert_eq!(input.len(), output.len());
    let scale = q.scale;
    let zp = q.zero_point as f32;
    for (src, dst) in input.iter().zip(output.iter_mut()) {
        *dst = (*src as f32 - zp) * scale;
    }
}

/// Dequantizes `u16` to `f32`: `output[i] = (input[i] − zero_point) · scale`.
///
/// # Panics
/// In debug builds, when `input` and `output` differ in length; release builds convert the
/// shorter length.
#[inline]
pub fn dequant_u16_to_f32(input: &[u16], q: Quant, output: &mut [f32]) {
    debug_assert_eq!(input.len(), output.len());
    let scale = q.scale;
    let zp = q.zero_point as f32;
    for (src, dst) in input.iter().zip(output.iter_mut()) {
        *dst = (*src as f32 - zp) * scale;
    }
}

/// Converts `f16` to `f32` as `(x − zero_point) · scale`; `Quant::IDENTITY` is a plain conversion.
///
/// # Panics
/// In debug builds, when `input` and `output` differ in length; release builds convert the
/// shorter length.
#[inline]
pub fn dequant_f16_to_f32(input: &[f16], q: Quant, output: &mut [f32]) {
    debug_assert_eq!(input.len(), output.len());
    if q == Quant::IDENTITY {
        for (src, dst) in input.iter().zip(output.iter_mut()) {
            *dst = src.to_f32();
        }
    } else {
        let scale = q.scale;
        let zp = q.zero_point as f32;
        for (src, dst) in input.iter().zip(output.iter_mut()) {
            *dst = (src.to_f32() - zp) * scale;
        }
    }
}

/// Converts `f32` to `f32` as `(x − zero_point) · scale`; `Quant::IDENTITY` is a plain copy.
///
/// Writes nothing when `input` and `output` differ in length.
#[inline]
pub fn dequant_f32_to_f32(input: &[f32], q: Quant, output: &mut [f32]) {
    if input.len() != output.len() {
        return;
    }
    if q == Quant::IDENTITY {
        output.copy_from_slice(input);
    } else {
        let scale = q.scale;
        let zp = q.zero_point as f32;
        for (src, dst) in input.iter().zip(output.iter_mut()) {
            *dst = (*src - zp) * scale;
        }
    }
}

/// Dequantizes `i8` to `f16`: `output[i] = (input[i] − zero_point) · scale`.
///
/// # Panics
/// In debug builds, when `input` and `output` differ in length; release builds convert the
/// shorter length.
#[inline]
pub fn dequant_i8_to_f16(input: &[i8], q: Quant, output: &mut [f16]) {
    debug_assert_eq!(input.len(), output.len());
    let scale = q.scale;
    let zp = q.zero_point as f32;
    for (src, dst) in input.iter().zip(output.iter_mut()) {
        *dst = f16::from_f32((*src as f32 - zp) * scale);
    }
}

/// Dequantizes `u8` to `f16`: `output[i] = (input[i] − zero_point) · scale`.
///
/// # Panics
/// In debug builds, when `input` and `output` differ in length; release builds convert the
/// shorter length.
#[inline]
pub fn dequant_u8_to_f16(input: &[u8], q: Quant, output: &mut [f16]) {
    debug_assert_eq!(input.len(), output.len());
    let scale = q.scale;
    let zp = q.zero_point as f32;
    for (src, dst) in input.iter().zip(output.iter_mut()) {
        *dst = f16::from_f32((*src as f32 - zp) * scale);
    }
}

/// Dequantizes `i16` to `f16`: `output[i] = (input[i] − zero_point) · scale`.
///
/// # Panics
/// In debug builds, when `input` and `output` differ in length; release builds convert the
/// shorter length.
#[inline]
pub fn dequant_i16_to_f16(input: &[i16], q: Quant, output: &mut [f16]) {
    debug_assert_eq!(input.len(), output.len());
    let scale = q.scale;
    let zp = q.zero_point as f32;
    for (src, dst) in input.iter().zip(output.iter_mut()) {
        *dst = f16::from_f32((*src as f32 - zp) * scale);
    }
}

/// Dequantizes `u16` to `f16`: `output[i] = (input[i] − zero_point) · scale`.
///
/// # Panics
/// In debug builds, when `input` and `output` differ in length; release builds convert the
/// shorter length.
#[inline]
pub fn dequant_u16_to_f16(input: &[u16], q: Quant, output: &mut [f16]) {
    debug_assert_eq!(input.len(), output.len());
    let scale = q.scale;
    let zp = q.zero_point as f32;
    for (src, dst) in input.iter().zip(output.iter_mut()) {
        *dst = f16::from_f32((*src as f32 - zp) * scale);
    }
}

/// Converts `f16` to `f16` as `(x − zero_point) · scale`; `Quant::IDENTITY` is a plain copy.
///
/// Writes nothing when `input` and `output` differ in length.
#[inline]
pub fn dequant_f16_to_f16(input: &[f16], q: Quant, output: &mut [f16]) {
    if input.len() != output.len() {
        return;
    }
    if q == Quant::IDENTITY {
        output.copy_from_slice(input);
    } else {
        let scale = q.scale;
        let zp = q.zero_point as f32;
        for (src, dst) in input.iter().zip(output.iter_mut()) {
            *dst = f16::from_f32((src.to_f32() - zp) * scale);
        }
    }
}

/// Converts `f32` to `f16` as `(x − zero_point) · scale`; `Quant::IDENTITY` is a plain conversion.
///
/// # Panics
/// In debug builds, when `input` and `output` differ in length; release builds convert the
/// shorter length.
#[inline]
pub fn dequant_f32_to_f16(input: &[f32], q: Quant, output: &mut [f16]) {
    debug_assert_eq!(input.len(), output.len());
    if q == Quant::IDENTITY {
        for (src, dst) in input.iter().zip(output.iter_mut()) {
            *dst = f16::from_f32(*src);
        }
    } else {
        let scale = q.scale;
        let zp = q.zero_point as f32;
        for (src, dst) in input.iter().zip(output.iter_mut()) {
            *dst = f16::from_f32((src - zp) * scale);
        }
    }
}

// Trait impls for dispatch — one zero-sized type per cell selector.

pub struct ScalarDequant;

macro_rules! impl_scalar_dequant {
    ($i:ty, $f:ty, $fn_name:ident) => {
        impl DequantKernel<$i, $f> for ScalarDequant {
            #[inline]
            fn dequant_slice(input: &[$i], q: Quant, output: &mut [$f]) {
                $fn_name(input, q, output);
            }
        }
    };
}

impl_scalar_dequant!(i8, f32, dequant_i8_to_f32);
impl_scalar_dequant!(u8, f32, dequant_u8_to_f32);
impl_scalar_dequant!(i16, f32, dequant_i16_to_f32);
impl_scalar_dequant!(u16, f32, dequant_u16_to_f32);
impl_scalar_dequant!(f16, f32, dequant_f16_to_f32);
impl_scalar_dequant!(f32, f32, dequant_f32_to_f32);
impl_scalar_dequant!(i8, f16, dequant_i8_to_f16);
impl_scalar_dequant!(u8, f16, dequant_u8_to_f16);
impl_scalar_dequant!(i16, f16, dequant_i16_to_f16);
impl_scalar_dequant!(u16, f16, dequant_u16_to_f16);
impl_scalar_dequant!(f16, f16, dequant_f16_to_f16);
impl_scalar_dequant!(f32, f16, dequant_f32_to_f16);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Quant;
    use half::f16;

    #[test]
    fn dequant_i8_to_f32_per_tensor_round_trip() {
        let q = Quant::new(0.1, -10);
        let input = [-10_i8, 0, 10, 127];
        let mut out = [0.0_f32; 4];
        dequant_i8_to_f32(&input, q, &mut out);
        assert!((out[0] - 0.0).abs() < 1e-6);
        assert!((out[1] - 1.0).abs() < 1e-6);
        assert!((out[2] - 2.0).abs() < 1e-6);
        assert!((out[3] - 13.7).abs() < 1e-5);
    }

    #[test]
    fn dequant_u8_to_f32_zero_point_zero() {
        let q = Quant::new(0.5, 0);
        let input = [0_u8, 1, 2, 255];
        let mut out = [0.0_f32; 4];
        dequant_u8_to_f32(&input, q, &mut out);
        assert!((out[0] - 0.0).abs() < 1e-6);
        assert!((out[3] - 127.5).abs() < 1e-4);
    }

    #[test]
    fn dequant_i16_to_f32_basic() {
        let q = Quant::new(0.001, 0);
        let input = [0_i16, 1000, -1000];
        let mut out = [0.0_f32; 3];
        dequant_i16_to_f32(&input, q, &mut out);
        assert!((out[0] - 0.0).abs() < 1e-6);
        assert!((out[1] - 1.0).abs() < 1e-6);
        assert!((out[2] + 1.0).abs() < 1e-6);
    }

    #[test]
    fn dequant_i8_to_f16_matches_f32_within_envelope() {
        let q = Quant::new(0.1, -10);
        let input = [-10_i8, 0, 10, 127];
        let mut out_f16 = [f16::ZERO; 4];
        let mut out_f32 = [0.0_f32; 4];
        dequant_i8_to_f16(&input, q, &mut out_f16);
        dequant_i8_to_f32(&input, q, &mut out_f32);
        for i in 0..4 {
            let a: f32 = out_f16[i].to_f32();
            let b: f32 = out_f32[i];
            assert!((a - b).abs() < 1e-2, "i={i} f16={a} f32={b}");
        }
    }

    #[test]
    fn dequant_f32_passthrough_with_identity_quant() {
        let input = [0.0_f32, 1.5, -2.25];
        let mut out = [0.0_f32; 3];
        dequant_f32_to_f32(&input, Quant::IDENTITY, &mut out);
        assert_eq!(out, input);
    }

    #[test]
    fn float_passthrough_length_mismatch_writes_nothing() {
        for q in [Quant::IDENTITY, Quant::new(0.5, 1)] {
            let mut out = [7.0f32; 2];
            dequant_f32_to_f32(&[1.0, 2.0, 3.0], q, &mut out);
            assert_eq!(out, [7.0; 2]);
            let mut out = [f16::ONE; 4];
            dequant_f16_to_f16(&[f16::ZERO; 3], q, &mut out);
            assert_eq!(out, [f16::ONE; 4]);
        }
    }

    #[test]
    fn dequant_handles_empty_slices() {
        let mut out: Vec<f32> = Vec::new();
        dequant_i8_to_f32(&[], Quant::new(1.0, 0), &mut out);
        assert!(out.is_empty());
    }
}
