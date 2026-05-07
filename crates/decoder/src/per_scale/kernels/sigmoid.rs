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

pub(crate) fn sigmoid_slice_f32(buf: &mut [f32]) {
    for v in buf.iter_mut() {
        *v = sigmoid_one_f32(*v);
    }
}

pub(crate) fn sigmoid_slice_f16(buf: &mut [f16]) {
    for v in buf.iter_mut() {
        *v = f16::from_f32(sigmoid_one_f32(v.to_f32()));
    }
}

#[allow(dead_code)]
pub(crate) struct ScalarSigmoid;

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
