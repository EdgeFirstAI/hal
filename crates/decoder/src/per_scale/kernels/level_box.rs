// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! Per-FPN-level box decode orchestrators (DFL and LTRB).
//!
//! Each function decodes one level's box tensor into 4*H*W floats
//! `[xc, yc, w, h]` per anchor, anchor-major.

use super::box_primitives::{
    dist2bbox_anchor_f16, dist2bbox_anchor_f32, weighted_sum_4sides_f16, weighted_sum_4sides_f32,
};
use super::dequant::{
    dequant_f16_to_f16, dequant_f16_to_f32, dequant_f32_to_f16, dequant_f32_to_f32,
    dequant_i16_to_f16, dequant_i16_to_f32, dequant_i8_to_f16, dequant_i8_to_f32,
    dequant_u16_to_f16, dequant_u16_to_f32, dequant_u8_to_f16, dequant_u8_to_f32,
};
use super::softmax::{softmax_inplace_f16, softmax_inplace_f32};
use crate::per_scale::plan::LevelPlan;
use crate::Quantization;
use half::f16;

const MAX_REG_MAX: usize = 64;

// ── DFL: dequant → softmax (per side) → weighted_sum → dist2bbox ─────────

/// Generate per-cell DFL level kernels.  Inputs are NHWC `[1, h, w, 4*reg_max]`.
macro_rules! impl_dfl_level_f32 {
    ($name:ident, $i:ty, $dequant:ident) => {
        #[allow(dead_code)]
        pub(crate) fn $name(input: &[$i], q: Quantization, level: &LevelPlan, dst: &mut [f32]) {
            let h = level.h;
            let w = level.w;
            let reg_max = level.reg_max;
            debug_assert_eq!(input.len(), h * w * 4 * reg_max);
            debug_assert_eq!(dst.len(), 4 * h * w);
            debug_assert!(reg_max <= MAX_REG_MAX);

            let mut deq: [f32; 4 * MAX_REG_MAX] = [0.0_f32; 4 * MAX_REG_MAX];
            for anchor in 0..(h * w) {
                let in_base = anchor * 4 * reg_max;
                $dequant(
                    &input[in_base..in_base + 4 * reg_max],
                    q,
                    &mut deq[..4 * reg_max],
                );
                for side in 0..4 {
                    softmax_inplace_f32(&mut deq[side * reg_max..(side + 1) * reg_max]);
                }
                let ltrb = weighted_sum_4sides_f32(&deq[..4 * reg_max], reg_max);
                let gx = level.grid_x[anchor];
                let gy = level.grid_y[anchor];
                let xywh = dist2bbox_anchor_f32(ltrb, gx, gy, level.stride);
                let out_base = anchor * 4;
                dst[out_base..out_base + 4].copy_from_slice(&xywh);
            }
        }
    };
}

macro_rules! impl_dfl_level_f16 {
    ($name:ident, $i:ty, $dequant:ident) => {
        #[allow(dead_code)]
        pub(crate) fn $name(input: &[$i], q: Quantization, level: &LevelPlan, dst: &mut [f16]) {
            let h = level.h;
            let w = level.w;
            let reg_max = level.reg_max;
            debug_assert_eq!(input.len(), h * w * 4 * reg_max);
            debug_assert_eq!(dst.len(), 4 * h * w);
            debug_assert!(reg_max <= MAX_REG_MAX);

            let mut deq: [f16; 4 * MAX_REG_MAX] = [f16::ZERO; 4 * MAX_REG_MAX];
            for anchor in 0..(h * w) {
                let in_base = anchor * 4 * reg_max;
                $dequant(
                    &input[in_base..in_base + 4 * reg_max],
                    q,
                    &mut deq[..4 * reg_max],
                );
                for side in 0..4 {
                    softmax_inplace_f16(&mut deq[side * reg_max..(side + 1) * reg_max]);
                }
                let ltrb = weighted_sum_4sides_f16(&deq[..4 * reg_max], reg_max);
                let gx = f16::from_f32(level.grid_x[anchor]);
                let gy = f16::from_f32(level.grid_y[anchor]);
                let stride = f16::from_f32(level.stride);
                let xywh = dist2bbox_anchor_f16(ltrb, gx, gy, stride);
                let out_base = anchor * 4;
                dst[out_base..out_base + 4].copy_from_slice(&xywh);
            }
        }
    };
}

// 12 DFL cells.
impl_dfl_level_f32!(decode_box_level_dfl_i8_to_f32, i8, dequant_i8_to_f32);
impl_dfl_level_f32!(decode_box_level_dfl_u8_to_f32, u8, dequant_u8_to_f32);
impl_dfl_level_f32!(decode_box_level_dfl_i16_to_f32, i16, dequant_i16_to_f32);
impl_dfl_level_f32!(decode_box_level_dfl_u16_to_f32, u16, dequant_u16_to_f32);
impl_dfl_level_f32!(decode_box_level_dfl_f16_to_f32, f16, dequant_f16_to_f32);
impl_dfl_level_f32!(decode_box_level_dfl_f32_to_f32, f32, dequant_f32_to_f32);

impl_dfl_level_f16!(decode_box_level_dfl_i8_to_f16, i8, dequant_i8_to_f16);
impl_dfl_level_f16!(decode_box_level_dfl_u8_to_f16, u8, dequant_u8_to_f16);
impl_dfl_level_f16!(decode_box_level_dfl_i16_to_f16, i16, dequant_i16_to_f16);
impl_dfl_level_f16!(decode_box_level_dfl_u16_to_f16, u16, dequant_u16_to_f16);
impl_dfl_level_f16!(decode_box_level_dfl_f16_to_f16, f16, dequant_f16_to_f16);
impl_dfl_level_f16!(decode_box_level_dfl_f32_to_f16, f32, dequant_f32_to_f16);

// ── LTRB: dequant → dist2bbox ────────────────────────────────────────────

macro_rules! impl_ltrb_level_f32 {
    ($name:ident, $i:ty, $dequant:ident) => {
        #[allow(dead_code)]
        pub(crate) fn $name(input: &[$i], q: Quantization, level: &LevelPlan, dst: &mut [f32]) {
            let h = level.h;
            let w = level.w;
            debug_assert_eq!(input.len(), h * w * 4);
            debug_assert_eq!(dst.len(), 4 * h * w);

            let mut deq: [f32; 4] = [0.0_f32; 4];
            for anchor in 0..(h * w) {
                let in_base = anchor * 4;
                $dequant(&input[in_base..in_base + 4], q, &mut deq);
                let gx = level.grid_x[anchor];
                let gy = level.grid_y[anchor];
                let xywh = dist2bbox_anchor_f32(deq, gx, gy, level.stride);
                let out_base = anchor * 4;
                dst[out_base..out_base + 4].copy_from_slice(&xywh);
            }
        }
    };
}

macro_rules! impl_ltrb_level_f16 {
    ($name:ident, $i:ty, $dequant:ident) => {
        #[allow(dead_code)]
        pub(crate) fn $name(input: &[$i], q: Quantization, level: &LevelPlan, dst: &mut [f16]) {
            let h = level.h;
            let w = level.w;
            debug_assert_eq!(input.len(), h * w * 4);
            debug_assert_eq!(dst.len(), 4 * h * w);

            let mut deq: [f16; 4] = [f16::ZERO; 4];
            for anchor in 0..(h * w) {
                let in_base = anchor * 4;
                $dequant(&input[in_base..in_base + 4], q, &mut deq);
                let gx = f16::from_f32(level.grid_x[anchor]);
                let gy = f16::from_f32(level.grid_y[anchor]);
                let stride = f16::from_f32(level.stride);
                let xywh = dist2bbox_anchor_f16(deq, gx, gy, stride);
                let out_base = anchor * 4;
                dst[out_base..out_base + 4].copy_from_slice(&xywh);
            }
        }
    };
}

// 12 LTRB cells.
impl_ltrb_level_f32!(decode_box_level_ltrb_i8_to_f32, i8, dequant_i8_to_f32);
impl_ltrb_level_f32!(decode_box_level_ltrb_u8_to_f32, u8, dequant_u8_to_f32);
impl_ltrb_level_f32!(decode_box_level_ltrb_i16_to_f32, i16, dequant_i16_to_f32);
impl_ltrb_level_f32!(decode_box_level_ltrb_u16_to_f32, u16, dequant_u16_to_f32);
impl_ltrb_level_f32!(decode_box_level_ltrb_f16_to_f32, f16, dequant_f16_to_f32);
impl_ltrb_level_f32!(decode_box_level_ltrb_f32_to_f32, f32, dequant_f32_to_f32);

impl_ltrb_level_f16!(decode_box_level_ltrb_i8_to_f16, i8, dequant_i8_to_f16);
impl_ltrb_level_f16!(decode_box_level_ltrb_u8_to_f16, u8, dequant_u8_to_f16);
impl_ltrb_level_f16!(decode_box_level_ltrb_i16_to_f16, i16, dequant_i16_to_f16);
impl_ltrb_level_f16!(decode_box_level_ltrb_u16_to_f16, u16, dequant_u16_to_f16);
impl_ltrb_level_f16!(decode_box_level_ltrb_f16_to_f16, f16, dequant_f16_to_f16);
impl_ltrb_level_f16!(decode_box_level_ltrb_f32_to_f16, f32, dequant_f32_to_f16);

// ────────────────────────────────────────────────────────────────────────
// NEON-baseline (Tier 1) DFL + LTRB box level kernels
//
// **DFL** keeps the per-anchor structure: each anchor's `4 * reg_max`
// channels (typically 64) is exactly 4 chunks of 16 i8 lanes — perfect
// shape for the chunked NEON dequant. Per-anchor: NEON dequant + scalar
// softmax + scalar weighted-sum + scalar dist2bbox. NEON softmax lands
// in task N-10 and replaces the per-side softmax inline.
//
// **LTRB** has only 4 channels per anchor — too few for NEON to amortize
// loop overhead. Instead the NEON variant **batch-dequants** the whole
// `(h * w * 4)` tensor into a heap scratch (sized 25600 f32 max for
// 80x80 LTRB, well under any per-frame budget), then runs a scalar
// per-anchor dist2bbox loop reading from the scratch.
// ────────────────────────────────────────────────────────────────────────

#[cfg(target_arch = "aarch64")]
macro_rules! impl_dfl_level_f32_neon {
    ($name:ident, $i:ty, $dequant_neon:ident, $softmax:ident) => {
        #[allow(dead_code)]
        pub(crate) fn $name(input: &[$i], q: Quantization, level: &LevelPlan, dst: &mut [f32]) {
            let h = level.h;
            let w = level.w;
            let reg_max = level.reg_max;
            debug_assert_eq!(input.len(), h * w * 4 * reg_max);
            debug_assert_eq!(dst.len(), 4 * h * w);
            debug_assert!(reg_max <= MAX_REG_MAX);

            let mut deq: [f32; 4 * MAX_REG_MAX] = [0.0_f32; 4 * MAX_REG_MAX];
            for anchor in 0..(h * w) {
                let in_base = anchor * 4 * reg_max;
                // SAFETY: NEON mandatory on aarch64; dispatch contract.
                // FP16-suffixed variants additionally require f.neon_fp16.
                unsafe {
                    crate::per_scale::kernels::neon_baseline::$dequant_neon(
                        &input[in_base..in_base + 4 * reg_max],
                        q,
                        &mut deq[..4 * reg_max],
                    );
                    for side in 0..4 {
                        crate::per_scale::kernels::neon_baseline::$softmax(
                            &mut deq[side * reg_max..(side + 1) * reg_max],
                        );
                    }
                }
                let ltrb = weighted_sum_4sides_f32(&deq[..4 * reg_max], reg_max);
                let gx = level.grid_x[anchor];
                let gy = level.grid_y[anchor];
                let xywh = dist2bbox_anchor_f32(ltrb, gx, gy, level.stride);
                let out_base = anchor * 4;
                dst[out_base..out_base + 4].copy_from_slice(&xywh);
            }
        }
    };
}

#[cfg(target_arch = "aarch64")]
impl_dfl_level_f32_neon!(
    decode_box_level_dfl_i8_to_f32_neon,
    i8,
    dequant_i8_to_f32_neon,
    softmax_inplace_f32_neon
);
#[cfg(target_arch = "aarch64")]
impl_dfl_level_f32_neon!(
    decode_box_level_dfl_u8_to_f32_neon,
    u8,
    dequant_u8_to_f32_neon,
    softmax_inplace_f32_neon
);
#[cfg(target_arch = "aarch64")]
impl_dfl_level_f32_neon!(
    decode_box_level_dfl_i16_to_f32_neon,
    i16,
    dequant_i16_to_f32_neon,
    softmax_inplace_f32_neon
);
#[cfg(target_arch = "aarch64")]
impl_dfl_level_f32_neon!(
    decode_box_level_dfl_u16_to_f32_neon,
    u16,
    dequant_u16_to_f32_neon,
    softmax_inplace_f32_neon
);

// FP16 DFL variants — same dequant, FP16 softmax. Reached only when
// CpuFeatures::neon_fp16 is true (gated upstream by dispatch::select).
#[cfg(target_arch = "aarch64")]
impl_dfl_level_f32_neon!(
    decode_box_level_dfl_i8_to_f32_neon_fp16,
    i8,
    dequant_i8_to_f32_neon,
    softmax_inplace_f32_neon_fp16
);
#[cfg(target_arch = "aarch64")]
impl_dfl_level_f32_neon!(
    decode_box_level_dfl_u8_to_f32_neon_fp16,
    u8,
    dequant_u8_to_f32_neon,
    softmax_inplace_f32_neon_fp16
);
#[cfg(target_arch = "aarch64")]
impl_dfl_level_f32_neon!(
    decode_box_level_dfl_i16_to_f32_neon_fp16,
    i16,
    dequant_i16_to_f32_neon,
    softmax_inplace_f32_neon_fp16
);
#[cfg(target_arch = "aarch64")]
impl_dfl_level_f32_neon!(
    decode_box_level_dfl_u16_to_f32_neon_fp16,
    u16,
    dequant_u16_to_f32_neon,
    softmax_inplace_f32_neon_fp16
);

#[cfg(target_arch = "aarch64")]
macro_rules! impl_ltrb_level_f32_neon {
    ($name:ident, $i:ty, $dequant_neon:ident) => {
        #[allow(dead_code)]
        pub(crate) fn $name(input: &[$i], q: Quantization, level: &LevelPlan, dst: &mut [f32]) {
            let h = level.h;
            let w = level.w;
            debug_assert_eq!(input.len(), h * w * 4);
            debug_assert_eq!(dst.len(), 4 * h * w);

            // Batch-dequant: NEON shines on the whole tensor (h*w*4 = up
            // to 25600 elements at level 0); per-anchor 4 elements alone
            // wouldn't fill a single NEON chunk. Heap scratch is small
            // and Phase 2-B can move this to a reusable per-frame buffer
            // (alongside the layout transpose scratch).
            let n = h * w * 4;
            let mut deq: Vec<f32> = vec![0.0_f32; n];
            // SAFETY: NEON mandatory on aarch64; dispatch contract.
            unsafe {
                crate::per_scale::kernels::neon_baseline::$dequant_neon(input, q, &mut deq);
            }
            // Per-anchor dist2bbox over the dequantized tensor.
            for anchor in 0..(h * w) {
                let base = anchor * 4;
                let dq = [deq[base], deq[base + 1], deq[base + 2], deq[base + 3]];
                let gx = level.grid_x[anchor];
                let gy = level.grid_y[anchor];
                let xywh = dist2bbox_anchor_f32(dq, gx, gy, level.stride);
                dst[base..base + 4].copy_from_slice(&xywh);
            }
        }
    };
}

#[cfg(target_arch = "aarch64")]
impl_ltrb_level_f32_neon!(
    decode_box_level_ltrb_i8_to_f32_neon,
    i8,
    dequant_i8_to_f32_neon
);
#[cfg(target_arch = "aarch64")]
impl_ltrb_level_f32_neon!(
    decode_box_level_ltrb_u8_to_f32_neon,
    u8,
    dequant_u8_to_f32_neon
);
#[cfg(target_arch = "aarch64")]
impl_ltrb_level_f32_neon!(
    decode_box_level_ltrb_i16_to_f32_neon,
    i16,
    dequant_i16_to_f32_neon
);
#[cfg(target_arch = "aarch64")]
impl_ltrb_level_f32_neon!(
    decode_box_level_ltrb_u16_to_f32_neon,
    u16,
    dequant_u16_to_f32_neon
);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::per_scale::kernels::grids::make_anchor_grid;
    use crate::per_scale::plan::LevelPlan;
    use crate::Quantization;

    fn level_1x1(reg_max: usize, stride: f32, box_channels: usize) -> LevelPlan {
        let (gx, gy) = make_anchor_grid(1, 1);
        LevelPlan {
            stride,
            h: 1,
            w: 1,
            reg_max,
            anchor_offset: 0,
            grid_x: gx,
            grid_y: gy,
            box_shape: vec![1, 1, 1, box_channels].into_boxed_slice(),
            score_shape: vec![1, 1, 1, 80].into_boxed_slice(),
            mc_shape: None,
            layout: crate::per_scale::plan::Layout::Nhwc,
        }
    }

    #[test]
    fn dfl_one_hot_at_bin_5_yields_distance_5() {
        // 1x1 grid, reg_max=16, stride=8. One-hot at bin 5 each side
        // → d_left = d_top = d_right = d_bottom = 5
        // → xc = (0.5 + 0)*8 = 4, w = (5+5)*8 = 80
        let mut logits = [0.0_f32; 64];
        for side in 0..4 {
            logits[side * 16 + 5] = 100.0;
        }
        let mut out_boxes = [0.0_f32; 4];
        decode_box_level_dfl_f32_to_f32(
            &logits,
            Quantization::identity(),
            &level_1x1(16, 8.0, 64),
            &mut out_boxes,
        );
        assert!((out_boxes[0] - 4.0).abs() < 1e-3);
        assert!((out_boxes[2] - 80.0).abs() < 1e-3);
    }

    #[test]
    fn ltrb_direct_distances_yield_dist2bbox() {
        // 1x1 grid, stride=8, ltrb=[1,1,3,3] post-dequant
        // → xc = (0.5 + 1)*8 = 12, yc = (0.5 + 1)*8 = 12
        // → w = 4*8 = 32, h = 4*8 = 32
        let logits = [1.0_f32, 1.0, 3.0, 3.0];
        let mut out_boxes = [0.0_f32; 4];
        decode_box_level_ltrb_f32_to_f32(
            &logits,
            Quantization::identity(),
            &level_1x1(1, 8.0, 4),
            &mut out_boxes,
        );
        assert!((out_boxes[0] - 12.0).abs() < 1e-3);
        assert!((out_boxes[2] - 32.0).abs() < 1e-3);
    }

    #[test]
    fn dfl_2x2_uniform_logits_each_anchor_at_centre() {
        // reg_max=4, all bins equal logit → softmax uniform → distance = 1.5 per side
        // 2x2 grid, stride=4. For anchor (0.5, 0.5):
        //   xc = (0.5 + 0)*4 = 2, w = (1.5+1.5)*4 = 12
        let logits = [1.0_f32; 2 * 2 * 4 * 4]; // (h=2, w=2, 4 sides * 4 bins)
        let (gx, gy) = make_anchor_grid(2, 2);
        let lvl = LevelPlan {
            stride: 4.0,
            h: 2,
            w: 2,
            reg_max: 4,
            anchor_offset: 0,
            grid_x: gx,
            grid_y: gy,
            box_shape: vec![1, 2, 2, 16].into_boxed_slice(),
            score_shape: vec![1, 2, 2, 80].into_boxed_slice(),
            mc_shape: None,
            layout: crate::per_scale::plan::Layout::Nhwc,
        };
        let mut out = [0.0_f32; 16]; // 4 anchors * 4 box-coords
        decode_box_level_dfl_f32_to_f32(&logits, Quantization::identity(), &lvl, &mut out);
        // Anchor 0 (0.5, 0.5)
        assert!((out[0] - 2.0).abs() < 1e-3);
        assert!((out[1] - 2.0).abs() < 1e-3);
        assert!((out[2] - 12.0).abs() < 1e-3);
        // Anchor 1 (1.5, 0.5)
        assert!((out[4] - 6.0).abs() < 1e-3, "expected xc=6, got {}", out[4]);
        // Anchor 2 (0.5, 1.5)
        assert!((out[9] - 6.0).abs() < 1e-3, "expected yc=6, got {}", out[9]);
    }

    #[test]
    fn dfl_i8_dequant_then_decode_matches_f32_input() {
        // Verify the i8→f32 path produces same boxes as f32 passthrough
        // when the dequantized logits are equivalent.
        let q = Quantization::new(0.1, -128);
        let mut input_i8 = [-128_i8; 64];
        // Set bin 5 of each side to a saturated value that dequants high.
        for side in 0..4 {
            input_i8[side * 16 + 5] = 127;
        }
        let mut out_i8 = [0.0_f32; 4];
        decode_box_level_dfl_i8_to_f32(&input_i8, q, &level_1x1(16, 8.0, 64), &mut out_i8);

        // Equivalent f32 logits: (-128 - -128)*0.1 = 0; (127 - -128)*0.1 = 25.5
        let mut logits_f32 = [0.0_f32; 64];
        for side in 0..4 {
            logits_f32[side * 16 + 5] = 25.5;
        }
        let mut out_f32 = [0.0_f32; 4];
        decode_box_level_dfl_f32_to_f32(
            &logits_f32,
            Quantization::identity(),
            &level_1x1(16, 8.0, 64),
            &mut out_f32,
        );

        for i in 0..4 {
            assert!(
                (out_i8[i] - out_f32[i]).abs() < 1e-2,
                "box[{i}]: i8={} f32={}",
                out_i8[i],
                out_f32[i]
            );
        }
    }
}
