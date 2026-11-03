// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! Score-level kernel: dequant + optional sigmoid.
//!
//! The input layout is NHWC `[1, h, w, num_classes]`. Dequantization
//! is element-wise so the layout is preserved → output is anchor-major
//! `[a0c0, a0c1, …, a0cNC-1, a1c0, …]`, exactly what NMS consumes.

use crate::per_scale::plan::LevelPlan;
use crate::per_scale::Activation;
use crate::Quantization;
use edgefirst_kernels::prim::dequant::{
    dequant_f16_to_f16, dequant_f16_to_f32, dequant_f32_to_f16, dequant_f32_to_f32,
    dequant_i16_to_f16, dequant_i16_to_f32, dequant_i8_to_f16, dequant_i8_to_f32,
    dequant_u16_to_f16, dequant_u16_to_f32, dequant_u8_to_f16, dequant_u8_to_f32,
};
use edgefirst_kernels::prim::sigmoid::{sigmoid_slice_f16, sigmoid_slice_f32};
#[cfg(target_arch = "aarch64")]
use edgefirst_kernels::Quant;
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
            $dequant(input, q.into(), dst);
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
            $dequant(input, q.into(), dst);
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

// NEON score levels: the exact `neon_convert` dequantize, then `$sigmoid` in place.

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
                edgefirst_kernels::prim::neon_convert::$dequant_neon(input, q.into(), dst);
                if activation == Activation::Sigmoid {
                    edgefirst_kernels::prim::neon_baseline::$sigmoid(dst);
                }
            }
        }
    };
}

#[cfg(target_arch = "aarch64")]
impl_score_level_f32_neon!(
    decode_score_level_i8_to_f32_neon,
    i8,
    dequant_i8_to_f32_exact_neon,
    sigmoid_slice_f32_neon
);
#[cfg(target_arch = "aarch64")]
impl_score_level_f32_neon!(
    decode_score_level_u8_to_f32_neon,
    u8,
    dequant_u8_to_f32_exact_neon,
    sigmoid_slice_f32_neon
);
#[cfg(target_arch = "aarch64")]
impl_score_level_f32_neon!(
    decode_score_level_i16_to_f32_neon,
    i16,
    dequant_i16_to_f32_exact_neon,
    sigmoid_slice_f32_neon
);
#[cfg(target_arch = "aarch64")]
impl_score_level_f32_neon!(
    decode_score_level_u16_to_f32_neon,
    u16,
    dequant_u16_to_f32_exact_neon,
    sigmoid_slice_f32_neon
);

// ────────────────────────────────────────────────────────────────────────
// Fused dequant+sigmoid NEON score-level kernels
//
// For quantized integer inputs (i8, u8, i16, u16) with Sigmoid activation,
// these use a single-pass kernel that dequants and applies sigmoid without
// writing then re-reading the intermediate f32 buffer. When activation is
// not Sigmoid, falls back to the two-pass NEON dequant path (no sigmoid).
// ────────────────────────────────────────────────────────────────────────

#[cfg(target_arch = "aarch64")]
macro_rules! impl_score_level_f32_neon_fused {
    ($name:ident, $i:ty, $fused:ident, $dequant_neon:ident) => {
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
            // SAFETY: NEON is mandatory on aarch64; dispatch contract.
            unsafe {
                if activation == Activation::Sigmoid {
                    edgefirst_kernels::prim::neon_baseline::$fused(input, q.into(), dst);
                } else {
                    edgefirst_kernels::prim::neon_convert::$dequant_neon(input, q.into(), dst);
                }
            }
        }
    };
}

#[cfg(target_arch = "aarch64")]
impl_score_level_f32_neon_fused!(
    decode_score_level_i8_to_f32_neon_fused,
    i8,
    dequant_sigmoid_i8_to_f32_neon,
    dequant_i8_to_f32_exact_neon
);
#[cfg(target_arch = "aarch64")]
impl_score_level_f32_neon_fused!(
    decode_score_level_u8_to_f32_neon_fused,
    u8,
    dequant_sigmoid_u8_to_f32_neon,
    dequant_u8_to_f32_exact_neon
);
#[cfg(target_arch = "aarch64")]
impl_score_level_f32_neon_fused!(
    decode_score_level_i16_to_f32_neon_fused,
    i16,
    dequant_sigmoid_i16_to_f32_neon,
    dequant_i16_to_f32_exact_neon
);
#[cfg(target_arch = "aarch64")]
impl_score_level_f32_neon_fused!(
    decode_score_level_u16_to_f32_neon_fused,
    u16,
    dequant_sigmoid_u16_to_f32_neon,
    dequant_u16_to_f32_exact_neon
);

// NEON cells for f16 outputs and for float inputs into f32.

/// One input element type's NEON kernels for f16 outputs.
#[cfg(target_arch = "aarch64")]
pub(crate) trait NeonToF16: Copy {
    /// `prim::dequant` into f16, bit for bit.
    ///
    /// # Safety
    /// The CPU must support NEON.
    unsafe fn dequant_f16(src: &[Self], q: Quant, dst: &mut [f16]);
    /// The f32 NEON sigmoid rounded once to f16.
    ///
    /// # Safety
    /// The CPU must support NEON.
    unsafe fn sigmoid_f16(src: &[Self], q: Quant, dst: &mut [f16]);
    /// The native FP16 sigmoid.
    ///
    /// # Safety
    /// The CPU must support FEAT_FP16.
    unsafe fn sigmoid_f16_fp16(src: &[Self], q: Quant, dst: &mut [f16]);
}

/// A float input type's NEON kernels for f32 outputs.
#[cfg(target_arch = "aarch64")]
pub(crate) trait NeonFloatToF32: Copy {
    /// `prim::dequant` into f32, bit for bit.
    ///
    /// # Safety
    /// The CPU must support NEON.
    unsafe fn dequant_f32(src: &[Self], q: Quant, dst: &mut [f32]);
    /// The f32 NEON sigmoid of the dequantized value.
    ///
    /// # Safety
    /// The CPU must support NEON.
    unsafe fn sigmoid_f32(src: &[Self], q: Quant, dst: &mut [f32]);
}

#[cfg(target_arch = "aarch64")]
macro_rules! neon_to_f16 {
    ($t:ty, $dequant:ident, $sigmoid:ident, $fp16:ident) => {
        impl NeonToF16 for $t {
            #[inline]
            unsafe fn dequant_f16(src: &[Self], q: Quant, dst: &mut [f16]) {
                // SAFETY: the caller guarantees NEON.
                unsafe { edgefirst_kernels::prim::neon_convert::$dequant(src, q, dst) }
            }
            #[inline]
            unsafe fn sigmoid_f16(src: &[Self], q: Quant, dst: &mut [f16]) {
                // SAFETY: the caller guarantees NEON.
                unsafe { edgefirst_kernels::prim::neon_sigmoid::$sigmoid(src, q, dst) }
            }
            #[inline]
            unsafe fn sigmoid_f16_fp16(src: &[Self], q: Quant, dst: &mut [f16]) {
                // SAFETY: the caller guarantees FEAT_FP16.
                unsafe { edgefirst_kernels::prim::neon_fp16::$fp16(src, q, dst) }
            }
        }
    };
}

#[cfg(target_arch = "aarch64")]
neon_to_f16!(
    i8,
    dequant_i8_to_f16_exact_neon,
    dequant_sigmoid_i8_to_f16_neon,
    dequant_sigmoid_i8_to_f16_neon_fp16
);
#[cfg(target_arch = "aarch64")]
neon_to_f16!(
    u8,
    dequant_u8_to_f16_exact_neon,
    dequant_sigmoid_u8_to_f16_neon,
    dequant_sigmoid_u8_to_f16_neon_fp16
);
#[cfg(target_arch = "aarch64")]
neon_to_f16!(
    i16,
    dequant_i16_to_f16_exact_neon,
    dequant_sigmoid_i16_to_f16_neon,
    dequant_sigmoid_i16_to_f16_neon_fp16
);
#[cfg(target_arch = "aarch64")]
neon_to_f16!(
    u16,
    dequant_u16_to_f16_exact_neon,
    dequant_sigmoid_u16_to_f16_neon,
    dequant_sigmoid_u16_to_f16_neon_fp16
);
#[cfg(target_arch = "aarch64")]
neon_to_f16!(
    f16,
    dequant_f16_to_f16_neon,
    dequant_sigmoid_f16_to_f16_neon,
    dequant_sigmoid_f16_to_f16_neon_fp16
);
#[cfg(target_arch = "aarch64")]
neon_to_f16!(
    f32,
    dequant_f32_to_f16_neon,
    dequant_sigmoid_f32_to_f16_neon,
    dequant_sigmoid_f32_to_f16_neon_fp16
);

#[cfg(target_arch = "aarch64")]
macro_rules! neon_float_to_f32 {
    ($t:ty, $dequant:ident, $sigmoid:ident) => {
        impl NeonFloatToF32 for $t {
            #[inline]
            unsafe fn dequant_f32(src: &[Self], q: Quant, dst: &mut [f32]) {
                // SAFETY: the caller guarantees NEON.
                unsafe { edgefirst_kernels::prim::neon_convert::$dequant(src, q, dst) }
            }
            #[inline]
            unsafe fn sigmoid_f32(src: &[Self], q: Quant, dst: &mut [f32]) {
                // SAFETY: the caller guarantees NEON.
                unsafe { edgefirst_kernels::prim::neon_sigmoid::$sigmoid(src, q, dst) }
            }
        }
    };
}

#[cfg(target_arch = "aarch64")]
neon_float_to_f32!(
    f16,
    dequant_f16_to_f32_neon,
    dequant_sigmoid_f16_to_f32_neon
);
#[cfg(target_arch = "aarch64")]
neon_float_to_f32!(
    f32,
    dequant_f32_to_f32_neon,
    dequant_sigmoid_f32_to_f32_neon
);

/// Score level into f16 on the `neon` tier.
#[cfg(target_arch = "aarch64")]
pub(crate) fn decode_score_level_to_f16_neon<T: NeonToF16>(
    input: &[T],
    q: Quantization,
    activation: Activation,
    dst: &mut [f16],
) {
    // SAFETY: NEON is part of the aarch64 baseline.
    unsafe {
        if activation == Activation::Sigmoid {
            T::sigmoid_f16(input, q.into(), dst);
        } else {
            T::dequant_f16(input, q.into(), dst);
        }
    }
}

/// Score level into f16 on the `neon_fp16` tier.
///
/// # Safety
/// The CPU must support FEAT_FP16.
#[cfg(target_arch = "aarch64")]
pub(crate) unsafe fn decode_score_level_to_f16_neon_fp16<T: NeonToF16>(
    input: &[T],
    q: Quantization,
    activation: Activation,
    dst: &mut [f16],
) {
    // SAFETY: the caller guarantees FEAT_FP16, which implies NEON.
    unsafe {
        if activation == Activation::Sigmoid {
            T::sigmoid_f16_fp16(input, q.into(), dst);
        } else {
            T::dequant_f16(input, q.into(), dst);
        }
    }
}

/// Float score level into f32 on NEON.
#[cfg(target_arch = "aarch64")]
pub(crate) fn decode_score_level_float_to_f32_neon<T: NeonFloatToF32>(
    input: &[T],
    q: Quantization,
    activation: Activation,
    dst: &mut [f32],
) {
    // SAFETY: NEON is part of the aarch64 baseline.
    unsafe {
        if activation == Activation::Sigmoid {
            T::sigmoid_f32(input, q.into(), dst);
        } else {
            T::dequant_f32(input, q.into(), dst);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use edgefirst_kernels::prim::grids::make_anchor_grid;

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
