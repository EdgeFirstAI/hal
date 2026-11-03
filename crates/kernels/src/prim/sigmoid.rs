// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! Scalar element-wise sigmoid for the per-scale score path.

use super::SigmoidKernel;
use half::f16;

/// Numerically-stable sigmoid:
/// - For x >= 0: 1 / (1 + exp(-x))
/// - For x <  0: exp(x) / (1 + exp(x))
///
/// Avoids overflow in either direction.
#[inline]
fn sigmoid_one_f32(x: f32) -> f32 {
    if x >= 0.0 {
        let e = (-x).exp();
        1.0 / (1.0 + e)
    } else {
        let e = x.exp();
        e / (1.0 + e)
    }
}

#[inline]
pub fn sigmoid_slice_f32(buf: &mut [f32]) {
    for v in buf.iter_mut() {
        *v = sigmoid_one_f32(*v);
    }
}

#[inline]
pub fn sigmoid_slice_f16(buf: &mut [f16]) {
    for v in buf.iter_mut() {
        *v = f16::from_f32(sigmoid_one_f32(v.to_f32()));
    }
}

/// Schraudolph's fast approximation of `eˣ`, bit for bit the `exp_raw` of the `fast-math` crate
/// (Huon Wilson, MIT/Apache-2.0); valid for about −88 ≤ x ≤ 88, which [`fast_sigmoid`]
/// guarantees, with relative error below about 0.011.
#[inline(always)]
fn exp_raw(x: f32) -> f32 {
    let mul = (FAST_EXP_A * x) as i32;
    let floor = mul & FAST_EXP_MASK;
    let frac = (mul - floor) as f32;
    let approx = (FAST_EXP_C0 * frac + FAST_EXP_C1) * frac + FAST_EXP_C2;
    f32::from_bits(approx.to_bits().wrapping_add(floor as u32))
}

/// `2^23 · log2(e)`: [`exp_raw`]'s input scale.
pub(crate) const FAST_EXP_A: f32 = (1u32 << 23) as f32 * core::f32::consts::LOG2_E;
/// The sign and exponent bits of an f32.
pub(crate) const FAST_EXP_MASK: i32 = 0xff800000u32 as i32;
const EXP2_23: f32 = 1.1920929e-7;
/// [`exp_raw`]'s mantissa polynomial, `(C0 · frac + C1) · frac + C2`, from `fast-math`'s
/// unrounded literals.
#[allow(clippy::excessive_precision)]
pub(crate) const FAST_EXP_C0: f32 = 0.3371894346 * EXP2_23 * EXP2_23;
#[allow(clippy::excessive_precision)]
pub(crate) const FAST_EXP_C1: f32 = 0.657636276 * EXP2_23;
#[allow(clippy::excessive_precision)]
pub(crate) const FAST_EXP_C2: f32 = 1.00172476;

/// The ModelPack decoder's sigmoid: `1 / (1 + exp_raw(−f))`, saturated to 0 or 1 beyond
/// `|f| > 80`; within 5e-4 of the exact sigmoid.
#[inline(always)]
pub fn fast_sigmoid(f: f32) -> f32 {
    if f.abs() > 80.0 {
        f.signum() * 0.5 + 0.5
    } else {
        // exp_raw is only valid for -88 < x < 88; the guard above keeps us in range
        1.0 / (1.0 + exp_raw(-f))
    }
}

pub struct ScalarSigmoid;

impl SigmoidKernel<f32> for ScalarSigmoid {
    #[inline]
    fn sigmoid_slice(buf: &mut [f32]) {
        sigmoid_slice_f32(buf);
    }
}

impl SigmoidKernel<f16> for ScalarSigmoid {
    #[inline]
    fn sigmoid_slice(buf: &mut [f16]) {
        sigmoid_slice_f16(buf);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The decoder's original `exp_raw` and `fast_sigmoid_impl`, verbatim.
    #[allow(clippy::excessive_precision)]
    fn original_fast_sigmoid(f: f32) -> f32 {
        fn exp_raw(x: f32) -> f32 {
            const A: f32 = (1u32 << 23) as f32;
            const MASK: i32 = 0xff800000u32 as i32;
            const EXP2_23: f32 = 1.1920929e-7;
            const C0: f32 = 0.3371894346 * EXP2_23 * EXP2_23;
            const C1: f32 = 0.657636276 * EXP2_23;
            const C2: f32 = 1.00172476;

            let a = A * core::f32::consts::LOG2_E;
            let mul = (a * x) as i32;
            let floor = mul & MASK;
            let frac = (mul - floor) as f32;
            let approx = (C0 * frac + C1) * frac + C2;
            f32::from_bits(approx.to_bits().wrapping_add(floor as u32))
        }
        if f.abs() > 80.0 {
            f.signum() * 0.5 + 0.5
        } else {
            1.0 / (1.0 + exp_raw(-f))
        }
    }

    /// `fast_sigmoid` equals the decoder's original code on every 61st f32 bit pattern.
    #[test]
    fn fast_sigmoid_matches_the_original_code() {
        for bits in (0..=u32::MAX).step_by(61) {
            let x = f32::from_bits(bits);
            assert_eq!(
                fast_sigmoid(x).to_bits(),
                original_fast_sigmoid(x).to_bits(),
                "{x:e} ({bits:#010x})"
            );
        }
    }

    #[test]
    fn sigmoid_zero_is_half() {
        let mut buf = [0.0_f32];
        sigmoid_slice_f32(&mut buf);
        assert!((buf[0] - 0.5).abs() < 1e-6);
    }

    #[test]
    fn sigmoid_large_positive_saturates_high() {
        let mut buf = [50.0_f32];
        sigmoid_slice_f32(&mut buf);
        assert!(buf[0] > 0.9999);
        assert!(buf[0].is_finite());
    }

    #[test]
    fn sigmoid_large_negative_saturates_low() {
        let mut buf = [-50.0_f32];
        sigmoid_slice_f32(&mut buf);
        assert!(buf[0] < 0.0001);
        assert!(buf[0].is_finite());
    }

    #[test]
    fn sigmoid_monotonic() {
        let mut buf = [-2.0_f32, -1.0, 0.0, 1.0, 2.0];
        sigmoid_slice_f32(&mut buf);
        for w in buf.windows(2) {
            assert!(w[0] < w[1]);
        }
    }
}
