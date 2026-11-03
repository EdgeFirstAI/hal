// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! Slice conversions: per-tensor dequantization to `f32` or `f16`, and the NCHW → NHWC transpose.
//!
//! Every dequantization computes `(x − zero_point) · scale` in f32, with no fused multiply-add,
//! and rounds an `f16` output once, to nearest even; for `f16` and `f32` inputs `Quant::IDENTITY`
//! is a plain conversion (a copy when the types match). The aarch64 NEON kernels and the
//! portable code give the same bits on every input, NaN and ±inf included. A call whose
//! `input` and `output` differ in length writes nothing; none of these functions panics.
//!
//! ```
//! use edgefirst_kernels::convert::{dequant_i8_to_f32, nchw_to_nhwc};
//! use edgefirst_kernels::Quant;
//!
//! let mut real = [0f32; 3];
//! dequant_i8_to_f32(&[-128, 0, 2], Quant::new(0.5, 0), &mut real);
//! assert_eq!(real, [-64.0, 0.0, 1.0]);
//!
//! // Two channels of a 1×2 image, channel-major to pixel-major.
//! let mut nhwc = [0u8; 4];
//! nchw_to_nhwc(&[1, 2, 3, 4], 1, 2, 2, &mut nhwc);
//! assert_eq!(nhwc, [1, 3, 2, 4]);
//! ```

use half::f16;

pub use crate::prim::transpose::{nchw_to_nhwc, TransposeElement};
use crate::Quant;

macro_rules! dequant {
    ($(#[$doc:meta])* $name:ident, $i:ty, $o:ty, $neon:ident) => {
        $(#[$doc])*
        ///
        /// Writes nothing when `input` and `output` differ in length.
        #[cfg(target_arch = "aarch64")]
        #[inline]
        pub fn $name(input: &[$i], q: Quant, output: &mut [$o]) {
            // SAFETY: NEON is part of the aarch64 baseline; the kernel checks the lengths.
            unsafe { crate::prim::neon_convert::$neon(input, q, output) }
        }

        $(#[$doc])*
        ///
        /// Writes nothing when `input` and `output` differ in length.
        #[cfg(not(target_arch = "aarch64"))]
        #[inline]
        pub fn $name(input: &[$i], q: Quant, output: &mut [$o]) {
            if input.len() == output.len() {
                crate::prim::dequant::$name(input, q, output);
            }
        }
    };
}

dequant!(
    /// Dequantizes `i8` to `f32`: `output[i] = (input[i] − zero_point) · scale`.
    dequant_i8_to_f32,
    i8,
    f32,
    dequant_i8_to_f32_exact_neon
);
dequant!(
    /// Dequantizes `u8` to `f32`: `output[i] = (input[i] − zero_point) · scale`.
    dequant_u8_to_f32,
    u8,
    f32,
    dequant_u8_to_f32_exact_neon
);
dequant!(
    /// Dequantizes `i16` to `f32`: `output[i] = (input[i] − zero_point) · scale`.
    dequant_i16_to_f32,
    i16,
    f32,
    dequant_i16_to_f32_exact_neon
);
dequant!(
    /// Dequantizes `u16` to `f32`: `output[i] = (input[i] − zero_point) · scale`.
    dequant_u16_to_f32,
    u16,
    f32,
    dequant_u16_to_f32_exact_neon
);
dequant!(
    /// Converts `f16` to `f32` as `(x − zero_point) · scale`; `Quant::IDENTITY` widens exactly.
    dequant_f16_to_f32,
    f16,
    f32,
    dequant_f16_to_f32_neon
);
dequant!(
    /// Converts `f32` to `f32` as `(x − zero_point) · scale`; `Quant::IDENTITY` copies.
    dequant_f32_to_f32,
    f32,
    f32,
    dequant_f32_to_f32_neon
);
dequant!(
    /// Dequantizes `i8` to `f16`: `output[i] = f16((input[i] − zero_point) · scale)`.
    dequant_i8_to_f16,
    i8,
    f16,
    dequant_i8_to_f16_exact_neon
);
dequant!(
    /// Dequantizes `u8` to `f16`: `output[i] = f16((input[i] − zero_point) · scale)`.
    dequant_u8_to_f16,
    u8,
    f16,
    dequant_u8_to_f16_exact_neon
);
dequant!(
    /// Dequantizes `i16` to `f16`: `output[i] = f16((input[i] − zero_point) · scale)`.
    dequant_i16_to_f16,
    i16,
    f16,
    dequant_i16_to_f16_exact_neon
);
dequant!(
    /// Dequantizes `u16` to `f16`: `output[i] = f16((input[i] − zero_point) · scale)`.
    dequant_u16_to_f16,
    u16,
    f16,
    dequant_u16_to_f16_exact_neon
);
dequant!(
    /// Converts `f16` to `f16` as `f16((x − zero_point) · scale)`, computed in f32;
    /// `Quant::IDENTITY` copies.
    dequant_f16_to_f16,
    f16,
    f16,
    dequant_f16_to_f16_neon
);
dequant!(
    /// Converts `f32` to `f16` as `f16((x − zero_point) · scale)`; `Quant::IDENTITY` rounds `x`
    /// to nearest even.
    dequant_f32_to_f16,
    f32,
    f16,
    dequant_f32_to_f16_neon
);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::prim::dequant as scalar;

    /// The public wrappers give the scalar bits, and write nothing on a length mismatch.
    #[test]
    fn convert_dequant_matches_prim_and_checks_lengths() {
        let ints: Vec<i16> = (-300..300).collect();
        let floats: Vec<f32> = ints.iter().map(|&v| v as f32 * 0.37).collect();
        let halves: Vec<f16> = floats.iter().map(|&v| f16::from_f32(v)).collect();
        for q in [Quant::IDENTITY, Quant::new(0.1, 3), Quant::new(2.5e-3, -9)] {
            let n = ints.len();
            let (mut a, mut b) = (vec![0.0f32; n], vec![0.0f32; n]);
            dequant_i16_to_f32(&ints, q, &mut a);
            scalar::dequant_i16_to_f32(&ints, q, &mut b);
            assert_eq!(a, b);
            dequant_f32_to_f32(&floats, q, &mut a);
            scalar::dequant_f32_to_f32(&floats, q, &mut b);
            assert_eq!(a, b);
            dequant_f16_to_f32(&halves, q, &mut a);
            scalar::dequant_f16_to_f32(&halves, q, &mut b);
            assert_eq!(a, b);
            let (mut a, mut b) = (vec![f16::ZERO; n], vec![f16::ZERO; n]);
            dequant_f32_to_f16(&floats, q, &mut a);
            scalar::dequant_f32_to_f16(&floats, q, &mut b);
            assert_eq!(a, b);
            dequant_f16_to_f16(&halves, q, &mut a);
            scalar::dequant_f16_to_f16(&halves, q, &mut b);
            assert_eq!(a, b);
        }
        let mut out = [9.0f32; 2];
        dequant_u8_to_f32(&[1, 2, 3], Quant::new(1.0, 0), &mut out);
        assert_eq!(out, [9.0; 2]);
        let mut out = [f16::ONE; 4];
        dequant_i8_to_f16(&[1, 2, 3], Quant::new(1.0, 0), &mut out);
        assert_eq!(out, [f16::ONE; 4]);
        dequant_f16_to_f16(&[f16::ZERO; 3], Quant::IDENTITY, &mut out);
        assert_eq!(out, [f16::ONE; 4]);
        let mut out = [9.0f32; 2];
        dequant_f32_to_f32(&[0.0; 3], Quant::IDENTITY, &mut out);
        assert_eq!(out, [9.0; 2]);
    }
}
