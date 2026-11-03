// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! Low-level primitives shared with edgefirst-decoder; not covered by semver.

use crate::Quant;

pub mod dequant;

/// Per-tensor affine dequantization: `out[i] = (in[i] - zp) * scale`.
///
/// Generic over input integer/float type `I` and output float type `F`.
/// Float-to-float passthrough is implemented for `I = F` with
/// `Quant::IDENTITY`.
pub trait DequantKernel<I, F> {
    fn dequant_slice(input: &[I], q: Quant, output: &mut [F]);
}

/// In-place numerically stable softmax over a length-`reg_max` slice.
/// Subtract-max prevents `exp` overflow on large logits.
pub trait SoftmaxKernel<F> {
    fn softmax_inplace(buf: &mut [F]);
}

pub mod softmax;

/// In-place element-wise sigmoid.
pub trait SigmoidKernel<F> {
    fn sigmoid_slice(buf: &mut [F]);
}

pub mod sigmoid;

/// DFL weighted-sum: given `4 * reg_max` softmax probabilities, compute
/// the four LTRB grid-unit distances `[d_left, d_top, d_right, d_bottom]`
/// for one anchor.
pub trait DflWeightedSumKernel<F> {
    fn weighted_sum_4sides(probs: &[F], reg_max: usize) -> [F; 4];
}

/// dist2bbox: given four LTRB grid-unit distances, anchor centre `(gx, gy)`
/// in grid units, and FPN stride, compute one anchor's `[xc, yc, w, h]` in
/// pixel coordinates. Multiplication order matches Ultralytics bit-for-bit.
pub trait Dist2BboxKernel<F> {
    fn dist2bbox_anchor(ltrb: [F; 4], gx: F, gy: F, stride: F) -> [F; 4];
}

pub mod box_primitives;

pub mod grids;

pub mod dfl_lut;

pub mod transpose;

#[cfg(target_arch = "aarch64")]
pub(crate) mod fp16_asm;

#[cfg(target_arch = "aarch64")]
pub mod neon_baseline;

#[cfg(target_arch = "aarch64")]
pub mod neon_candidates;

#[cfg(target_arch = "aarch64")]
pub mod neon_convert;

#[cfg(target_arch = "aarch64")]
pub mod neon_fp16;

#[cfg(target_arch = "aarch64")]
pub mod neon_ltrb;

#[cfg(target_arch = "aarch64")]
pub mod neon_mask;

#[cfg(target_arch = "aarch64")]
pub mod neon_modelpack;

#[cfg(target_arch = "aarch64")]
pub mod neon_nms;

#[cfg(target_arch = "aarch64")]
pub mod neon_seg;

#[cfg(target_arch = "aarch64")]
pub mod neon_sigmoid;

#[cfg(target_arch = "aarch64")]
pub mod neon_semantic;
