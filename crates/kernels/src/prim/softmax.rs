// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! Scalar numerically-stable softmax for the per-scale DFL path, and the branch-free `e^x` of the
//! semantic decoder's softmax confidence.

use super::SoftmaxKernel;
use half::f16;

/// In-place softmax: subtract-max → exp → normalize.
#[inline]
pub fn softmax_inplace_f32(buf: &mut [f32]) {
    if buf.is_empty() {
        return;
    }
    let m = buf.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0_f32;
    for v in buf.iter_mut() {
        *v = (*v - m).exp();
        sum += *v;
    }
    if sum > 0.0 {
        let inv = 1.0 / sum;
        for v in buf.iter_mut() {
            *v *= inv;
        }
    }
}

#[inline]
pub fn softmax_inplace_f16(buf: &mut [f16]) {
    // Compute in f32 for stability, write back as f16.
    if buf.is_empty() {
        return;
    }
    let mut tmp: [f32; 64] = [0.0; 64];
    debug_assert!(
        buf.len() <= tmp.len(),
        "reg_max > 64 not supported in scalar f16 softmax"
    );
    let n = buf.len();
    for i in 0..n {
        tmp[i] = buf[i].to_f32();
    }
    softmax_inplace_f32(&mut tmp[..n]);
    for i in 0..n {
        buf[i] = f16::from_f32(tmp[i]);
    }
}

/// `exp_nonpositive`'s lower input clamp.
pub(crate) const EXP_MIN: f32 = -87.0;
/// `exp_nonpositive`'s Taylor coefficients of `2^f` after `1` and `ln 2`, highest degree first.
pub(crate) const EXP2_TAYLOR: [f32; 4] = [0.001_333_36, 0.009_618_13, 0.055_504_11, 0.240_226_5];

/// `e^x` for `x <= 0`, within 1e-5 relative error; inputs below -87 (and NaN) clamp to `e^-87`.
/// Branch-free so the softmax loops vectorize.
#[inline(always)]
pub fn exp_nonpositive(x: f32) -> f32 {
    let t = x.max(EXP_MIN) * std::f32::consts::LOG2_E;
    // Truncating `t - 0.5` rounds `t <= 0` to nearest without a libm call.
    let i = (t - 0.5) as i32 as f32;
    let f = t - i;
    let [c5, c4, c3, c2] = EXP2_TAYLOR;
    // Taylor series of 2^f on [-0.5, 0.5].
    let p = 1.0 + f * (std::f32::consts::LN_2 + f * (c2 + f * (c3 + f * (c4 + f * c5))));
    f32::from_bits(((i as i32).wrapping_add(127) as u32) << 23) * p
}

pub struct ScalarSoftmax;

impl SoftmaxKernel<f32> for ScalarSoftmax {
    #[inline]
    fn softmax_inplace(buf: &mut [f32]) {
        softmax_inplace_f32(buf);
    }
}

impl SoftmaxKernel<f16> for ScalarSoftmax {
    #[inline]
    fn softmax_inplace(buf: &mut [f16]) {
        softmax_inplace_f16(buf);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exp_nonpositive_is_accurate() {
        for i in 0..=1740 {
            let x = -(i as f32) * 0.05;
            let (fast, exact) = (exp_nonpositive(x), x.exp());
            assert!((fast - exact).abs() <= exact * 1e-5, "x={x} {fast} {exact}");
        }
        assert_eq!(exp_nonpositive(-1000.0), exp_nonpositive(-87.0));
        assert_eq!(exp_nonpositive(f32::NAN), exp_nonpositive(-87.0));
    }

    /// The decoder's original `exp_nonpositive`, verbatim.
    fn exp_nonpositive_original(x: f32) -> f32 {
        let t = x.max(-87.0) * std::f32::consts::LOG2_E;
        let i = (t - 0.5) as i32 as f32;
        let f = t - i;
        let p = 1.0
            + f * (std::f32::consts::LN_2
                + f * (0.240_226_5 + f * (0.055_504_11 + f * (0.009_618_13 + f * 0.001_333_36))));
        f32::from_bits(((i as i32 + 127) as u32) << 23) * p
    }

    #[test]
    fn exp_nonpositive_matches_the_original_code() {
        for i in 0..1u32 << 20 {
            let x = f32::from_bits(i.wrapping_mul(4099) | 0x8000_0000);
            assert_eq!(
                exp_nonpositive(x).to_bits(),
                exp_nonpositive_original(x).to_bits(),
                "{x:e}"
            );
        }
    }

    #[test]
    fn softmax_uniform_yields_uniform() {
        let mut buf = [1.0_f32; 8];
        softmax_inplace_f32(&mut buf);
        for &v in &buf {
            assert!((v - 0.125).abs() < 1e-6);
        }
    }

    #[test]
    fn softmax_concentrated_one_hot() {
        let mut buf = [0.0_f32; 16];
        buf[5] = 1000.0;
        softmax_inplace_f32(&mut buf);
        assert!((buf[5] - 1.0).abs() < 1e-6);
        for (i, &v) in buf.iter().enumerate() {
            if i != 5 {
                assert!(v.abs() < 1e-6);
            }
        }
    }

    #[test]
    fn softmax_stable_against_overflow() {
        let mut buf = [1000.0_f32, 1001.0, 1002.0];
        softmax_inplace_f32(&mut buf);
        let sum: f32 = buf.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);
        assert!(buf.iter().all(|v| v.is_finite()));
        assert!(buf[2] > buf[1] && buf[1] > buf[0]);
    }

    #[test]
    fn softmax_empty_is_noop() {
        let mut buf: [f32; 0] = [];
        softmax_inplace_f32(&mut buf);
    }

    #[test]
    fn softmax_f16_matches_f32_within_envelope() {
        use half::f16;
        let mut bf32 = [1.0_f32, 2.0, 3.0, 4.0];
        let mut bf16: [f16; 4] = bf32.map(f16::from_f32);
        softmax_inplace_f32(&mut bf32);
        softmax_inplace_f16(&mut bf16);
        for (a, b) in bf32.iter().zip(bf16.iter()) {
            assert!((a - b.to_f32()).abs() < 1e-3);
        }
    }
}
