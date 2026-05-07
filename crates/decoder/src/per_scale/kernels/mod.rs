// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! Per-scale decoder kernels — primitives, level orchestrators, dispatch.

use crate::Quantization;

pub(crate) mod cpu_features;
pub(crate) mod dequant;

#[allow(unused_imports)] // Consumed by Phase 1 dispatch tables in later tasks.
pub(crate) use cpu_features::CpuFeatures;

/// Per-tensor affine dequantization: `out[i] = (in[i] - zp) * scale`.
///
/// Generic over input integer/float type `I` and output float type `F`.
/// Float-to-float passthrough is implemented for `I = F` with
/// `Quantization::identity()`.
#[allow(dead_code)]
pub(crate) trait DequantKernel<I, F> {
    fn dequant_slice(input: &[I], q: Quantization, output: &mut [F]);
}

/// In-place numerically stable softmax over a length-`reg_max` slice.
/// Subtract-max prevents `exp` overflow on large logits.
#[allow(dead_code)]
pub(crate) trait SoftmaxKernel<F> {
    fn softmax_inplace(buf: &mut [F]);
}

pub(crate) mod softmax;

/// In-place element-wise sigmoid.
#[allow(dead_code)]
pub(crate) trait SigmoidKernel<F> {
    fn sigmoid_slice(buf: &mut [F]);
}

pub(crate) mod sigmoid;

/// DFL weighted-sum: given `4 * reg_max` softmax probabilities, compute
/// the four LTRB grid-unit distances `[d_left, d_top, d_right, d_bottom]`
/// for one anchor.
#[allow(dead_code)]
pub(crate) trait DflWeightedSumKernel<F> {
    fn weighted_sum_4sides(probs: &[F], reg_max: usize) -> [F; 4];
}

/// dist2bbox: given four LTRB grid-unit distances, anchor centre `(gx, gy)`
/// in grid units, and FPN stride, compute one anchor's `[xc, yc, w, h]` in
/// pixel coordinates. Multiplication order matches Ultralytics bit-for-bit.
#[allow(dead_code)]
pub(crate) trait Dist2BboxKernel<F> {
    fn dist2bbox_anchor(ltrb: [F; 4], gx: F, gy: F, stride: F) -> [F; 4];
}

pub(crate) mod box_primitives;

pub(crate) mod grids;

pub(crate) mod dispatch;

pub(crate) mod level_box;
pub(crate) mod level_mc;
pub(crate) mod level_score;

pub(crate) mod transpose;

#[cfg(target_arch = "aarch64")]
pub(crate) mod neon_baseline;
