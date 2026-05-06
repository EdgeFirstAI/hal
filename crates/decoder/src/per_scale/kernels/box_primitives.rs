// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! Scalar DFL weighted-sum + dist2bbox primitives.

use super::{DflWeightedSumKernel, Dist2BboxKernel};
use half::f16;

/// Compute four LTRB distances by weighted-sum of softmax probabilities
/// against bins `[0, 1, …, reg_max-1]`.
///
/// `probs.len() == 4 * reg_max`, side-major: side 0 owns indices
/// `[0..reg_max]`, side 1 owns `[reg_max..2*reg_max]`, …
pub(crate) fn weighted_sum_4sides_f32(probs: &[f32], reg_max: usize) -> [f32; 4] {
    debug_assert_eq!(probs.len(), 4 * reg_max);
    let mut d = [0.0_f32; 4];
    for (side, slot) in d.iter_mut().enumerate() {
        let mut s = 0.0_f32;
        let base = side * reg_max;
        for i in 0..reg_max {
            s += probs[base + i] * (i as f32);
        }
        *slot = s;
    }
    d
}

pub(crate) fn weighted_sum_4sides_f16(probs: &[f16], reg_max: usize) -> [f16; 4] {
    // Compute in f32 for precision; result fits f16 trivially (max bin=15).
    debug_assert_eq!(probs.len(), 4 * reg_max);
    let mut d = [f16::ZERO; 4];
    for (side, slot) in d.iter_mut().enumerate() {
        let mut s = 0.0_f32;
        let base = side * reg_max;
        for i in 0..reg_max {
            s += probs[base + i].to_f32() * (i as f32);
        }
        *slot = f16::from_f32(s);
    }
    d
}

#[inline]
pub(crate) fn dist2bbox_anchor_f32(ltrb: [f32; 4], gx: f32, gy: f32, stride: f32) -> [f32; 4] {
    let [d_left, d_top, d_right, d_bottom] = ltrb;
    // Order matches Ultralytics: (grid + (r-l)/2) * stride.
    let xc = (gx + (d_right - d_left) * 0.5) * stride;
    let yc = (gy + (d_bottom - d_top) * 0.5) * stride;
    let w = (d_left + d_right) * stride;
    let h = (d_top + d_bottom) * stride;
    [xc, yc, w, h]
}

#[inline]
pub(crate) fn dist2bbox_anchor_f16(ltrb: [f16; 4], gx: f16, gy: f16, stride: f16) -> [f16; 4] {
    let r = dist2bbox_anchor_f32(
        [
            ltrb[0].to_f32(),
            ltrb[1].to_f32(),
            ltrb[2].to_f32(),
            ltrb[3].to_f32(),
        ],
        gx.to_f32(),
        gy.to_f32(),
        stride.to_f32(),
    );
    [
        f16::from_f32(r[0]),
        f16::from_f32(r[1]),
        f16::from_f32(r[2]),
        f16::from_f32(r[3]),
    ]
}

#[allow(dead_code)]
pub(crate) struct ScalarBoxPrimitives;

impl DflWeightedSumKernel<f32> for ScalarBoxPrimitives {
    #[inline]
    fn weighted_sum_4sides(probs: &[f32], reg_max: usize) -> [f32; 4] {
        weighted_sum_4sides_f32(probs, reg_max)
    }
}

impl DflWeightedSumKernel<f16> for ScalarBoxPrimitives {
    #[inline]
    fn weighted_sum_4sides(probs: &[f16], reg_max: usize) -> [f16; 4] {
        weighted_sum_4sides_f16(probs, reg_max)
    }
}

impl Dist2BboxKernel<f32> for ScalarBoxPrimitives {
    #[inline]
    fn dist2bbox_anchor(ltrb: [f32; 4], gx: f32, gy: f32, stride: f32) -> [f32; 4] {
        dist2bbox_anchor_f32(ltrb, gx, gy, stride)
    }
}

impl Dist2BboxKernel<f16> for ScalarBoxPrimitives {
    #[inline]
    fn dist2bbox_anchor(ltrb: [f16; 4], gx: f16, gy: f16, stride: f16) -> [f16; 4] {
        dist2bbox_anchor_f16(ltrb, gx, gy, stride)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn weighted_sum_uniform_distribution() {
        // reg_max=4, all probs uniform = 0.25 → expected = 0+1+2+3 / 4 = 1.5
        let probs = [0.25_f32; 16]; // 4 sides × 4 bins
        let d = weighted_sum_4sides_f32(&probs, 4);
        for &v in &d {
            assert!((v - 1.5).abs() < 1e-6);
        }
    }

    #[test]
    fn weighted_sum_one_hot_at_bin_5() {
        let mut probs = [0.0_f32; 64]; // 4 × 16
        for side in 0..4 {
            probs[side * 16 + 5] = 1.0;
        }
        let d = weighted_sum_4sides_f32(&probs, 16);
        for &v in &d {
            assert!((v - 5.0).abs() < 1e-5);
        }
    }

    #[test]
    fn dist2bbox_zero_distances_at_centre() {
        let xywh = dist2bbox_anchor_f32([0.0; 4], 0.5, 0.5, 8.0);
        // xc = (0.5 + 0)*8 = 4
        assert!((xywh[0] - 4.0).abs() < 1e-6);
        assert!((xywh[1] - 4.0).abs() < 1e-6);
        // w = h = 0
        assert!(xywh[2].abs() < 1e-6);
        assert!(xywh[3].abs() < 1e-6);
    }

    #[test]
    fn dist2bbox_symmetric_distances() {
        // d=[2,2,2,2] at grid (0.5,0.5), stride 8
        // xc = (0.5 + 0)*8 = 4; yc = (0.5+0)*8 = 4
        // w = (2+2)*8 = 32; h = 32
        let xywh = dist2bbox_anchor_f32([2.0, 2.0, 2.0, 2.0], 0.5, 0.5, 8.0);
        assert!((xywh[0] - 4.0).abs() < 1e-6);
        assert!((xywh[1] - 4.0).abs() < 1e-6);
        assert!((xywh[2] - 32.0).abs() < 1e-6);
        assert!((xywh[3] - 32.0).abs() < 1e-6);
    }

    #[test]
    fn dist2bbox_asymmetric_left_vs_right() {
        // d=[1,0,3,0] at grid (0,0), stride 1 → xc = 0 + (3-1)/2 = 1
        // w = 1+3 = 4
        let xywh = dist2bbox_anchor_f32([1.0, 0.0, 3.0, 0.0], 0.0, 0.0, 1.0);
        assert!((xywh[0] - 1.0).abs() < 1e-6);
        assert!((xywh[2] - 4.0).abs() < 1e-6);
    }
}
