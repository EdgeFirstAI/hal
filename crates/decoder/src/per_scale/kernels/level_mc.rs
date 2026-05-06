// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! Mask-coefficient + proto level kernel: dequant only.
//!
//! Used for both per-scale mask_coefs (per-FPN-level) and the single
//! protos tensor (no per-scale structure — but the same kernel cells
//! apply since it's also pure dequant).

use super::dequant::{
    dequant_f16_to_f16, dequant_f16_to_f32, dequant_f32_to_f16, dequant_f32_to_f32,
    dequant_i16_to_f16, dequant_i16_to_f32, dequant_i8_to_f16, dequant_i8_to_f32,
    dequant_u16_to_f16, dequant_u16_to_f32, dequant_u8_to_f16, dequant_u8_to_f32,
};
use crate::per_scale::plan::LevelPlan;
use crate::Quantization;
use half::f16;

/// Per-FPN-level mask-coef dequant. NHWC `[1, h, w, num_mc]`.
macro_rules! impl_mc_level {
    ($name:ident, $i:ty, $f:ty, $dequant:ident) => {
        #[allow(dead_code)]
        pub(crate) fn $name(
            input: &[$i],
            q: Quantization,
            num_mc: usize,
            level: &LevelPlan,
            dst: &mut [$f],
        ) {
            let n = level.h * level.w * num_mc;
            debug_assert_eq!(input.len(), n);
            debug_assert_eq!(dst.len(), n);
            $dequant(input, q, dst);
        }
    };
}

impl_mc_level!(decode_mc_level_i8_to_f32, i8, f32, dequant_i8_to_f32);
impl_mc_level!(decode_mc_level_u8_to_f32, u8, f32, dequant_u8_to_f32);
impl_mc_level!(decode_mc_level_i16_to_f32, i16, f32, dequant_i16_to_f32);
impl_mc_level!(decode_mc_level_u16_to_f32, u16, f32, dequant_u16_to_f32);
impl_mc_level!(decode_mc_level_f16_to_f32, f16, f32, dequant_f16_to_f32);
impl_mc_level!(decode_mc_level_f32_to_f32, f32, f32, dequant_f32_to_f32);
impl_mc_level!(decode_mc_level_i8_to_f16, i8, f16, dequant_i8_to_f16);
impl_mc_level!(decode_mc_level_u8_to_f16, u8, f16, dequant_u8_to_f16);
impl_mc_level!(decode_mc_level_i16_to_f16, i16, f16, dequant_i16_to_f16);
impl_mc_level!(decode_mc_level_u16_to_f16, u16, f16, dequant_u16_to_f16);
impl_mc_level!(decode_mc_level_f16_to_f16, f16, f16, dequant_f16_to_f16);
impl_mc_level!(decode_mc_level_f32_to_f16, f32, f16, dequant_f32_to_f16);

/// Single-tensor proto dequant. Same dequant cells; different driver
/// (no per-level structure, takes flat input/output slices).
macro_rules! impl_proto {
    ($name:ident, $i:ty, $f:ty, $dequant:ident) => {
        #[allow(dead_code)]
        pub(crate) fn $name(input: &[$i], q: Quantization, dst: &mut [$f]) {
            debug_assert_eq!(input.len(), dst.len());
            $dequant(input, q, dst);
        }
    };
}

impl_proto!(decode_proto_i8_to_f32, i8, f32, dequant_i8_to_f32);
impl_proto!(decode_proto_u8_to_f32, u8, f32, dequant_u8_to_f32);
impl_proto!(decode_proto_i16_to_f32, i16, f32, dequant_i16_to_f32);
impl_proto!(decode_proto_u16_to_f32, u16, f32, dequant_u16_to_f32);
impl_proto!(decode_proto_f16_to_f32, f16, f32, dequant_f16_to_f32);
impl_proto!(decode_proto_f32_to_f32, f32, f32, dequant_f32_to_f32);
impl_proto!(decode_proto_i8_to_f16, i8, f16, dequant_i8_to_f16);
impl_proto!(decode_proto_u8_to_f16, u8, f16, dequant_u8_to_f16);
impl_proto!(decode_proto_i16_to_f16, i16, f16, dequant_i16_to_f16);
impl_proto!(decode_proto_u16_to_f16, u16, f16, dequant_u16_to_f16);
impl_proto!(decode_proto_f16_to_f16, f16, f16, dequant_f16_to_f16);
impl_proto!(decode_proto_f32_to_f16, f32, f16, dequant_f32_to_f16);

// ────────────────────────────────────────────────────────────────────────
// NEON-baseline (Tier 1) mc + proto kernels
//
// MC and protos are pure dequant — no sigmoid, no DFL machinery — so the
// NEON variants just call the bandwidth-optimised dequant primitives in
// `kernels::neon_baseline`. Same dispatch contract as score: only
// reachable via the *NeonBase variant when CpuFeatures::neon_baseline
// is true at probe time.
// ────────────────────────────────────────────────────────────────────────

#[cfg(target_arch = "aarch64")]
macro_rules! impl_mc_level_neon {
    ($name:ident, $i:ty, $dequant_neon:ident) => {
        #[allow(dead_code)]
        pub(crate) fn $name(
            input: &[$i],
            q: Quantization,
            num_mc: usize,
            level: &LevelPlan,
            dst: &mut [f32],
        ) {
            let n = level.h * level.w * num_mc;
            debug_assert_eq!(input.len(), n);
            debug_assert_eq!(dst.len(), n);
            // SAFETY: NEON is mandatory on aarch64; dispatch contract
            // ensures this is only called when select() picked the
            // *NeonBase variant.
            unsafe {
                crate::per_scale::kernels::neon_baseline::$dequant_neon(input, q, dst);
            }
        }
    };
}

#[cfg(target_arch = "aarch64")]
impl_mc_level_neon!(decode_mc_level_i8_to_f32_neon, i8, dequant_i8_to_f32_neon);
#[cfg(target_arch = "aarch64")]
impl_mc_level_neon!(decode_mc_level_u8_to_f32_neon, u8, dequant_u8_to_f32_neon);
#[cfg(target_arch = "aarch64")]
impl_mc_level_neon!(
    decode_mc_level_i16_to_f32_neon,
    i16,
    dequant_i16_to_f32_neon
);
#[cfg(target_arch = "aarch64")]
impl_mc_level_neon!(
    decode_mc_level_u16_to_f32_neon,
    u16,
    dequant_u16_to_f32_neon
);

#[cfg(target_arch = "aarch64")]
macro_rules! impl_proto_neon {
    ($name:ident, $i:ty, $dequant_neon:ident) => {
        #[allow(dead_code)]
        pub(crate) fn $name(input: &[$i], q: Quantization, dst: &mut [f32]) {
            debug_assert_eq!(input.len(), dst.len());
            // SAFETY: see impl_mc_level_neon.
            unsafe {
                crate::per_scale::kernels::neon_baseline::$dequant_neon(input, q, dst);
            }
        }
    };
}

#[cfg(target_arch = "aarch64")]
impl_proto_neon!(decode_proto_i8_to_f32_neon, i8, dequant_i8_to_f32_neon);
#[cfg(target_arch = "aarch64")]
impl_proto_neon!(decode_proto_u8_to_f32_neon, u8, dequant_u8_to_f32_neon);
#[cfg(target_arch = "aarch64")]
impl_proto_neon!(decode_proto_i16_to_f32_neon, i16, dequant_i16_to_f32_neon);
#[cfg(target_arch = "aarch64")]
impl_proto_neon!(decode_proto_u16_to_f32_neon, u16, dequant_u16_to_f32_neon);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::per_scale::kernels::grids::make_anchor_grid;

    #[test]
    fn mc_level_i8_dequant_round_trip() {
        let q = Quantization::new(0.5, -10);
        let nm = 2;
        let (gx, gy) = make_anchor_grid(2, 2);
        let lvl = LevelPlan {
            stride: 8.0,
            h: 2,
            w: 2,
            reg_max: 16,
            anchor_offset: 0,
            grid_x: gx,
            grid_y: gy,
            box_shape: vec![1, 2, 2, 64].into_boxed_slice(),
            score_shape: vec![1, 2, 2, 80].into_boxed_slice(),
            mc_shape: Some(vec![1, 2, 2, nm].into_boxed_slice()),
            layout: crate::per_scale::plan::Layout::Nhwc,
        };
        let input = vec![-10_i8, 0, 10, 20, -5, 5, 100, -100];
        let mut out = vec![0.0_f32; 8];
        decode_mc_level_i8_to_f32(&input, q, nm, &lvl, &mut out);
        // (q - zp) * scale: (-10 - -10)*0.5 = 0; (0 - -10)*0.5 = 5; etc
        assert!((out[0] - 0.0).abs() < 1e-6);
        assert!((out[1] - 5.0).abs() < 1e-6);
        assert!((out[2] - 10.0).abs() < 1e-6);
        assert!((out[7] - -45.0).abs() < 1e-6);
    }

    #[test]
    fn proto_i8_dequant_works() {
        let q = Quantization::new(0.1, 0);
        let input = vec![0_i8, 10, -10, 50];
        let mut out = vec![0.0_f32; 4];
        decode_proto_i8_to_f32(&input, q, &mut out);
        assert!((out[0] - 0.0).abs() < 1e-6);
        assert!((out[1] - 1.0).abs() < 1e-6);
        assert!((out[2] - -1.0).abs() < 1e-6);
        assert!((out[3] - 5.0).abs() < 1e-6);
    }
}
