// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! Score-level kernel: dequant + optional sigmoid.
//!
//! The input layout is NHWC `[1, h, w, num_classes]`. Dequantization
//! is element-wise so the layout is preserved → output is anchor-major
//! `[a0c0, a0c1, …, a0cNC-1, a1c0, …]`, exactly what NMS consumes.

use super::dequant::{
    dequant_f16_to_f16, dequant_f16_to_f32, dequant_f32_to_f16, dequant_f32_to_f32,
    dequant_i16_to_f16, dequant_i16_to_f32, dequant_i8_to_f16, dequant_i8_to_f32,
    dequant_u16_to_f16, dequant_u16_to_f32, dequant_u8_to_f16, dequant_u8_to_f32,
};
use super::sigmoid::{sigmoid_slice_f16, sigmoid_slice_f32};
use crate::per_scale::plan::LevelPlan;
use crate::per_scale::Activation;
use crate::Quantization;
use half::f16;

macro_rules! impl_score_level_f32 {
    ($name:ident, $i:ty, $dequant:ident) => {
        #[allow(dead_code)]
        pub(crate) fn $name(
            input: &[$i],
            q: Quantization,
            num_classes: usize,
            level: &LevelPlan,
            activation: Activation,
            dst: &mut [f32],
        ) {
            let n = level.h * level.w * num_classes;
            debug_assert_eq!(input.len(), n);
            debug_assert_eq!(dst.len(), n);
            $dequant(input, q, dst);
            if activation == Activation::Sigmoid {
                sigmoid_slice_f32(dst);
            }
        }
    };
}

macro_rules! impl_score_level_f16 {
    ($name:ident, $i:ty, $dequant:ident) => {
        #[allow(dead_code)]
        pub(crate) fn $name(
            input: &[$i],
            q: Quantization,
            num_classes: usize,
            level: &LevelPlan,
            activation: Activation,
            dst: &mut [f16],
        ) {
            let n = level.h * level.w * num_classes;
            debug_assert_eq!(input.len(), n);
            debug_assert_eq!(dst.len(), n);
            $dequant(input, q, dst);
            if activation == Activation::Sigmoid {
                sigmoid_slice_f16(dst);
            }
        }
    };
}

impl_score_level_f32!(decode_score_level_i8_to_f32, i8, dequant_i8_to_f32);
impl_score_level_f32!(decode_score_level_u8_to_f32, u8, dequant_u8_to_f32);
impl_score_level_f32!(decode_score_level_i16_to_f32, i16, dequant_i16_to_f32);
impl_score_level_f32!(decode_score_level_u16_to_f32, u16, dequant_u16_to_f32);
impl_score_level_f32!(decode_score_level_f16_to_f32, f16, dequant_f16_to_f32);
impl_score_level_f32!(decode_score_level_f32_to_f32, f32, dequant_f32_to_f32);

impl_score_level_f16!(decode_score_level_i8_to_f16, i8, dequant_i8_to_f16);
impl_score_level_f16!(decode_score_level_u8_to_f16, u8, dequant_u8_to_f16);
impl_score_level_f16!(decode_score_level_i16_to_f16, i16, dequant_i16_to_f16);
impl_score_level_f16!(decode_score_level_u16_to_f16, u16, dequant_u16_to_f16);
impl_score_level_f16!(decode_score_level_f16_to_f16, f16, dequant_f16_to_f16);
impl_score_level_f16!(decode_score_level_f32_to_f16, f32, dequant_f32_to_f16);

// ────────────────────────────────────────────────────────────────────────
// NEON-baseline (Tier 1) score level kernels
//
// Same shape as the scalar variants above but call the NEON dequant
// primitives in `kernels::neon_baseline`. Sigmoid still goes through
// the scalar `sigmoid_slice_f32` at this milestone — NEON sigmoid lands
// in task N-7 and replaces the activation step in-place. The dequant is
// the bandwidth-bound part of the score path; getting it onto NEON
// captures the bulk of the speedup before sigmoid is parallelized.
// ────────────────────────────────────────────────────────────────────────

#[cfg(target_arch = "aarch64")]
macro_rules! impl_score_level_f32_neon {
    ($name:ident, $i:ty, $dequant_neon:ident, $sigmoid:ident) => {
        #[allow(dead_code)]
        pub(crate) fn $name(
            input: &[$i],
            q: Quantization,
            num_classes: usize,
            level: &LevelPlan,
            activation: Activation,
            dst: &mut [f32],
        ) {
            let n = level.h * level.w * num_classes;
            debug_assert_eq!(input.len(), n);
            debug_assert_eq!(dst.len(), n);
            // SAFETY: NEON is mandatory on aarch64 and the dispatch
            // contract guarantees this function is only reached when
            // CpuFeatures::neon_baseline was probed true. Variants that
            // use the FP16 sigmoid additionally require f.neon_fp16
            // (gated upstream in dispatch::select).
            unsafe {
                crate::per_scale::kernels::neon_baseline::$dequant_neon(input, q, dst);
                if activation == Activation::Sigmoid {
                    crate::per_scale::kernels::neon_baseline::$sigmoid(dst);
                }
            }
        }
    };
}

#[cfg(target_arch = "aarch64")]
impl_score_level_f32_neon!(
    decode_score_level_i8_to_f32_neon,
    i8,
    dequant_i8_to_f32_neon,
    sigmoid_slice_f32_neon
);
#[cfg(target_arch = "aarch64")]
impl_score_level_f32_neon!(
    decode_score_level_u8_to_f32_neon,
    u8,
    dequant_u8_to_f32_neon,
    sigmoid_slice_f32_neon
);
#[cfg(target_arch = "aarch64")]
impl_score_level_f32_neon!(
    decode_score_level_i16_to_f32_neon,
    i16,
    dequant_i16_to_f32_neon,
    sigmoid_slice_f32_neon
);
#[cfg(target_arch = "aarch64")]
impl_score_level_f32_neon!(
    decode_score_level_u16_to_f32_neon,
    u16,
    dequant_u16_to_f32_neon,
    sigmoid_slice_f32_neon
);

// FP16 variants — same dequant path, FP16 sigmoid. Gated to be reached
// only when CpuFeatures::neon_fp16 is true.
#[cfg(target_arch = "aarch64")]
impl_score_level_f32_neon!(
    decode_score_level_i8_to_f32_neon_fp16,
    i8,
    dequant_i8_to_f32_neon,
    sigmoid_slice_f32_neon_fp16
);
#[cfg(target_arch = "aarch64")]
impl_score_level_f32_neon!(
    decode_score_level_u8_to_f32_neon_fp16,
    u8,
    dequant_u8_to_f32_neon,
    sigmoid_slice_f32_neon_fp16
);
#[cfg(target_arch = "aarch64")]
impl_score_level_f32_neon!(
    decode_score_level_i16_to_f32_neon_fp16,
    i16,
    dequant_i16_to_f32_neon,
    sigmoid_slice_f32_neon_fp16
);
#[cfg(target_arch = "aarch64")]
impl_score_level_f32_neon!(
    decode_score_level_u16_to_f32_neon_fp16,
    u16,
    dequant_u16_to_f32_neon,
    sigmoid_slice_f32_neon_fp16
);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::per_scale::kernels::grids::make_anchor_grid;

    fn level_2x2(num_classes: usize) -> LevelPlan {
        let (gx, gy) = make_anchor_grid(2, 2);
        LevelPlan {
            stride: 8.0,
            h: 2,
            w: 2,
            reg_max: 16,
            anchor_offset: 0,
            grid_x: gx,
            grid_y: gy,
            box_shape: vec![1, 2, 2, 64].into_boxed_slice(),
            score_shape: vec![1, 2, 2, num_classes].into_boxed_slice(),
            mc_shape: None,
            layout: crate::per_scale::plan::Layout::Nhwc,
        }
    }

    #[test]
    fn score_level_f32_passthrough_no_activation() {
        let nc = 3;
        let input = vec![
            0.0_f32, 0.5, 1.0, -1.0, 2.0, -2.0, 10.0, -10.0, 0.0, 0.5, 0.5, 0.5,
        ];
        let mut out = vec![0.0_f32; 12];
        decode_score_level_f32_to_f32(
            &input,
            Quantization::identity(),
            nc,
            &level_2x2(nc),
            Activation::None,
            &mut out,
        );
        assert_eq!(out, input);
    }

    #[test]
    fn score_level_f32_with_sigmoid_zero_yields_half() {
        let nc = 3;
        let input = vec![0.0_f32; 12];
        let mut out = vec![0.0_f32; 12];
        decode_score_level_f32_to_f32(
            &input,
            Quantization::identity(),
            nc,
            &level_2x2(nc),
            Activation::Sigmoid,
            &mut out,
        );
        for &v in &out {
            assert!((v - 0.5).abs() < 1e-6);
        }
    }

    #[test]
    fn score_level_i8_dequant_then_sigmoid() {
        let nc = 2;
        let q = Quantization::new(0.1, 0);
        // Input dequants to all-zeros except one large negative value.
        let input = vec![0_i8; 8]; // 2*2 anchors × 2 classes
        let mut out_no_act = vec![0.0_f32; 8];
        decode_score_level_i8_to_f32(
            &input,
            q,
            nc,
            &level_2x2(nc),
            Activation::None,
            &mut out_no_act,
        );
        for &v in &out_no_act {
            assert!(v.abs() < 1e-6);
        }

        let mut out_sig = vec![0.0_f32; 8];
        decode_score_level_i8_to_f32(
            &input,
            q,
            nc,
            &level_2x2(nc),
            Activation::Sigmoid,
            &mut out_sig,
        );
        for &v in &out_sig {
            assert!((v - 0.5).abs() < 1e-6);
        }
    }
}
