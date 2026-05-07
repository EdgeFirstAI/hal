// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! Scalar numerically-stable softmax for the per-scale DFL path.

use super::SoftmaxKernel;
use half::f16;

/// In-place softmax: subtract-max → exp → normalize.
pub(crate) fn softmax_inplace_f32(buf: &mut [f32]) {
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

pub(crate) fn softmax_inplace_f16(buf: &mut [f16]) {
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

#[allow(dead_code)]
pub(crate) struct ScalarSoftmax;

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
