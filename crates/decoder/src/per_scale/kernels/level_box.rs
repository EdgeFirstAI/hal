// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! Per-FPN-level box decode orchestrators (DFL and LTRB).
//!
//! Each function decodes one level's box tensor into 4*H*W floats
//! `[xc, yc, w, h]` per anchor, anchor-major.

use crate::per_scale::plan::LevelPlan;
use crate::Quantization;
use edgefirst_kernels::prim::box_primitives::{
    dist2bbox_anchor_f16, dist2bbox_anchor_f32, weighted_sum_4sides_f16, weighted_sum_4sides_f32,
};
use edgefirst_kernels::prim::dequant::{
    dequant_f16_to_f16, dequant_f16_to_f32, dequant_f32_to_f16, dequant_f32_to_f32,
    dequant_i16_to_f16, dequant_i16_to_f32, dequant_i8_to_f16, dequant_i8_to_f32,
    dequant_u16_to_f16, dequant_u16_to_f32, dequant_u8_to_f16, dequant_u8_to_f32,
};
use edgefirst_kernels::prim::softmax::{softmax_inplace_f16, softmax_inplace_f32};
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
                    q.into(),
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
                    q.into(),
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

// ── DFL per-channel: per-bin dequant → softmax → weighted_sum → dist2bbox ──

/// Generate per-cell DFL level kernels with per-channel quantization.
/// `scales[i]` and `zps[i]` index into the channel axis (`4 * reg_max`
/// values per anchor); each bin has its own scale and zero-point.
/// Float inputs never carry per-channel quant from NPUs — only integer
/// input types are instantiated.
macro_rules! impl_dfl_level_f32_per_channel {
    ($name:ident, $i:ty) => {
        #[allow(dead_code)]
        pub(crate) fn $name(
            input: &[$i],
            scales: &[f32],
            zps: &[i32],
            level: &LevelPlan,
            dst: &mut [f32],
        ) {
            let h = level.h;
            let w = level.w;
            let reg_max = level.reg_max;
            debug_assert_eq!(input.len(), h * w * 4 * reg_max);
            debug_assert_eq!(dst.len(), 4 * h * w);
            debug_assert!(reg_max <= MAX_REG_MAX);
            debug_assert_eq!(scales.len(), 4 * reg_max);
            // `zps` is either empty (the "all zero-points = 0" convention used
            // for symmetric quant) or one entry per channel. A mismatched
            // length would otherwise silently default missing channels to 0.
            debug_assert!(
                zps.is_empty() || zps.len() == 4 * reg_max,
                "per-channel zero-points must be empty or 4*reg_max ({}), got {}",
                4 * reg_max,
                zps.len(),
            );

            let mut deq: [f32; 4 * MAX_REG_MAX] = [0.0_f32; 4 * MAX_REG_MAX];
            for anchor in 0..(h * w) {
                let in_base = anchor * 4 * reg_max;
                // Per-channel: deq[i] = (in[i] - zp[i]) * scale[i]
                for i in 0..(4 * reg_max) {
                    let raw = input[in_base + i] as f32;
                    let zp = zps.get(i).copied().unwrap_or(0) as f32;
                    deq[i] = (raw - zp) * scales[i];
                }
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

impl_dfl_level_f32_per_channel!(decode_box_level_dfl_i8_to_f32_per_channel, i8);
impl_dfl_level_f32_per_channel!(decode_box_level_dfl_u8_to_f32_per_channel, u8);
impl_dfl_level_f32_per_channel!(decode_box_level_dfl_i16_to_f32_per_channel, i16);
impl_dfl_level_f32_per_channel!(decode_box_level_dfl_u16_to_f32_per_channel, u16);

/// Generate per-cell DFL level kernels with per-channel quant and f16 output.
macro_rules! impl_dfl_level_f16_per_channel {
    ($name:ident, $i:ty) => {
        #[allow(dead_code)]
        pub(crate) fn $name(
            input: &[$i],
            scales: &[f32],
            zps: &[i32],
            level: &LevelPlan,
            dst: &mut [f16],
        ) {
            let h = level.h;
            let w = level.w;
            let reg_max = level.reg_max;
            debug_assert_eq!(input.len(), h * w * 4 * reg_max);
            debug_assert_eq!(dst.len(), 4 * h * w);
            debug_assert!(reg_max <= MAX_REG_MAX);
            debug_assert_eq!(scales.len(), 4 * reg_max);
            // `zps` is either empty (the "all zero-points = 0" convention used
            // for symmetric quant) or one entry per channel. A mismatched
            // length would otherwise silently default missing channels to 0.
            debug_assert!(
                zps.is_empty() || zps.len() == 4 * reg_max,
                "per-channel zero-points must be empty or 4*reg_max ({}), got {}",
                4 * reg_max,
                zps.len(),
            );

            let mut deq: [f32; 4 * MAX_REG_MAX] = [0.0_f32; 4 * MAX_REG_MAX];
            for anchor in 0..(h * w) {
                let in_base = anchor * 4 * reg_max;
                for i in 0..(4 * reg_max) {
                    let raw = input[in_base + i] as f32;
                    let zp = zps.get(i).copied().unwrap_or(0) as f32;
                    deq[i] = (raw - zp) * scales[i];
                }
                for side in 0..4 {
                    softmax_inplace_f32(&mut deq[side * reg_max..(side + 1) * reg_max]);
                }
                let ltrb = weighted_sum_4sides_f32(&deq[..4 * reg_max], reg_max);
                let gx = level.grid_x[anchor];
                let gy = level.grid_y[anchor];
                let xywh = dist2bbox_anchor_f32(ltrb, gx, gy, level.stride);
                let out_base = anchor * 4;
                dst[out_base] = f16::from_f32(xywh[0]);
                dst[out_base + 1] = f16::from_f32(xywh[1]);
                dst[out_base + 2] = f16::from_f32(xywh[2]);
                dst[out_base + 3] = f16::from_f32(xywh[3]);
            }
        }
    };
}

impl_dfl_level_f16_per_channel!(decode_box_level_dfl_i8_to_f16_per_channel, i8);
impl_dfl_level_f16_per_channel!(decode_box_level_dfl_u8_to_f16_per_channel, u8);
impl_dfl_level_f16_per_channel!(decode_box_level_dfl_i16_to_f16_per_channel, i16);
impl_dfl_level_f16_per_channel!(decode_box_level_dfl_u16_to_f16_per_channel, u16);

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
                $dequant(&input[in_base..in_base + 4], q.into(), &mut deq);
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
                $dequant(&input[in_base..in_base + 4], q.into(), &mut deq);
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

// ── NEON (aarch64) ───────────────────────────────────────────────────────

#[cfg(target_arch = "aarch64")]
pub(crate) use neon::{dfl_f16_neon_fp16, dfl_neon, dfl_per_channel_neon, ltrb_neon, Exp};

#[cfg(target_arch = "aarch64")]
mod neon {
    use super::{dist2bbox_anchor_f32, f16, LevelPlan, Quantization, MAX_REG_MAX};
    use edgefirst_kernels::prim::neon_convert::PerChannelIn;
    use edgefirst_kernels::prim::neon_ltrb::LtrbIn;
    use edgefirst_kernels::prim::{neon_baseline, neon_convert, neon_fp16, neon_ltrb};
    use edgefirst_kernels::Quant;

    /// The exp of the NEON fused softmax and weighted sum.
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub(crate) enum Exp {
        /// `softmax_weighted_sum_4sides_f32_neon`.
        F32,
        /// `softmax_weighted_sum_4sides_f32_neon_fp16`, the fp16 exp; needs FEAT_FP16.
        Fp16,
    }

    /// An f32 or f16 box output of the NEON cells: `f16` boxes are the f32 boxes rounded once,
    /// except where a native FP16 kernel is named.
    pub(crate) trait BoxOut: Copy {
        /// Four anchors' boxes from their distances (`ltrb`, 16 values) into `out` (16 values),
        /// through `dist2bbox_4anchors_f32_neon`.
        fn boxes4(ltrb: &[f32; 16], gx: &[f32], gy: &[f32], stride: f32, out: &mut [Self]);

        /// One anchor's f32 box.
        fn store4(out: &mut [Self], b: [f32; 4]);

        /// One level's LTRB boxes through `prim::neon_ltrb`.
        fn ltrb<T: LtrbIn>(input: &[T], q: Quant, level: &LevelPlan, dst: &mut [Self]);

        /// One level's per-channel DFL boxes from the logits `deq` writes: the f32 path, or for
        /// `f16` boxes with `fp16` the native FP16 distances and boxes.
        ///
        /// # Safety
        /// `fp16` needs FEAT_FP16.
        unsafe fn per_channel<T>(
            input: &[T],
            level: &LevelPlan,
            fp16: bool,
            dst: &mut [Self],
            deq: impl FnMut(&[T], &mut [f32]),
        );
    }

    impl BoxOut for f32 {
        #[inline(always)]
        fn boxes4(ltrb: &[f32; 16], gx: &[f32], gy: &[f32], stride: f32, out: &mut [Self]) {
            if gx.len() >= 4 && gy.len() >= 4 && out.len() >= 16 {
                // SAFETY: NEON is part of the aarch64 baseline; the lengths are checked.
                unsafe { neon_baseline::dist2bbox_4anchors_f32_neon(ltrb, gx, gy, stride, out) };
            }
        }

        #[inline(always)]
        fn store4(out: &mut [Self], b: [f32; 4]) {
            out.iter_mut().zip(b).for_each(|(o, v)| *o = v);
        }

        #[inline(always)]
        fn ltrb<T: LtrbIn>(input: &[T], q: Quant, level: &LevelPlan, dst: &mut [Self]) {
            let (gx, gy, s) = (&level.grid_x[..], &level.grid_y[..], level.stride);
            // SAFETY: NEON is part of the aarch64 baseline.
            unsafe { neon_ltrb::ltrb_boxes_f32_neon(input, q, gx, gy, s, false, dst) };
        }

        #[inline(always)]
        unsafe fn per_channel<T>(
            input: &[T],
            level: &LevelPlan,
            _fp16: bool,
            dst: &mut [Self],
            deq: impl FnMut(&[T], &mut [f32]),
        ) {
            // SAFETY: the f32 exp runs on every NEON CPU.
            unsafe { dfl_level(input, level, Exp::F32, dst, deq) };
        }
    }

    impl BoxOut for f16 {
        #[inline(always)]
        fn boxes4(ltrb: &[f32; 16], gx: &[f32], gy: &[f32], stride: f32, out: &mut [Self]) {
            let mut b = [0.0f32; 16];
            <f32 as BoxOut>::boxes4(ltrb, gx, gy, stride, &mut b);
            if let Some(out) = out.get_mut(..16) {
                // SAFETY: NEON is part of the aarch64 baseline; both slices hold 16 values.
                unsafe { neon_convert::narrow_f32_to_f16_neon(&b, out) };
            }
        }

        #[inline(always)]
        fn store4(out: &mut [Self], b: [f32; 4]) {
            out.iter_mut()
                .zip(b)
                .for_each(|(o, v)| *o = f16::from_f32(v));
        }

        #[inline(always)]
        fn ltrb<T: LtrbIn>(input: &[T], q: Quant, level: &LevelPlan, dst: &mut [Self]) {
            let (gx, gy, s) = (&level.grid_x[..], &level.grid_y[..], level.stride);
            // SAFETY: NEON is part of the aarch64 baseline.
            unsafe { neon_ltrb::ltrb_boxes_f16_neon(input, q, gx, gy, s, false, dst) };
        }

        #[inline(always)]
        unsafe fn per_channel<T>(
            input: &[T],
            level: &LevelPlan,
            fp16: bool,
            dst: &mut [Self],
            deq: impl FnMut(&[T], &mut [f32]),
        ) {
            // SAFETY: the caller guarantees FEAT_FP16 when `fp16`.
            unsafe {
                if fp16 {
                    dfl_level_f16(input, level, dst, deq);
                } else {
                    dfl_level(input, level, Exp::F32, dst, deq);
                }
            }
        }
    }

    /// The anchor count of a DFL level call, or `None` when the slices do not match the level.
    #[inline(always)]
    fn dfl_anchors<T, O>(input: &[T], level: &LevelPlan, dst: &[O]) -> Option<usize> {
        let rm = level.reg_max;
        let n = dst.len() / 4;
        let ok = rm > 0
            && rm <= MAX_REG_MAX
            && dst.len().is_multiple_of(4)
            && Some(input.len()) == n.checked_mul(4 * rm)
            && level.grid_x.len() >= n
            && level.grid_y.len() >= n;
        ok.then_some(n)
    }

    /// One level's DFL boxes: `deq` writes an anchor's `4 · reg_max` logits in f32, the fused
    /// softmax and weighted sum (`exp`) gives its distances, and dist2bbox runs four anchors per
    /// NEON step, the last `n % 4` scalar. Writes nothing unless the slices match the level.
    ///
    /// # Safety
    /// [`Exp::Fp16`] needs FEAT_FP16.
    #[inline(always)]
    unsafe fn dfl_level<T, O: BoxOut>(
        input: &[T],
        level: &LevelPlan,
        exp: Exp,
        dst: &mut [O],
        mut deq: impl FnMut(&[T], &mut [f32]),
    ) {
        let Some(n) = dfl_anchors(input, level, dst) else {
            return;
        };
        let rm = level.reg_max;
        let per = 4 * rm;
        let mut store = [0.0f32; 4 * MAX_REG_MAX];
        let Some(logits) = store.get_mut(..per) else {
            return;
        };
        let mut ltrb = |bins: &[T]| -> [f32; 4] {
            deq(bins, logits);
            // SAFETY: the caller guarantees FEAT_FP16 for `Exp::Fp16`; `logits` holds 4·reg_max.
            unsafe {
                match exp {
                    Exp::F32 => neon_baseline::softmax_weighted_sum_4sides_f32_neon(logits, rm),
                    Exp::Fp16 => {
                        neon_baseline::softmax_weighted_sum_4sides_f32_neon_fp16(logits, rm)
                    }
                }
            }
        };
        let (gx, gy, stride) = (&level.grid_x[..], &level.grid_y[..], level.stride);
        let quads = input
            .chunks_exact(4 * per)
            .zip(dst.chunks_exact_mut(16))
            .zip(gx.chunks_exact(4).zip(gy.chunks_exact(4)))
            .take(n / 4);
        for ((bins, out), (x, y)) in quads {
            let mut d = [0.0f32; 16];
            for (d, a) in d.chunks_exact_mut(4).zip(bins.chunks_exact(per)) {
                d.copy_from_slice(&ltrb(a));
            }
            O::boxes4(&d, x, y, stride, out);
        }
        for a in n / 4 * 4..n {
            let (Some(bins), Some(out), Some(&x), Some(&y)) = (
                input.get(a * per..(a + 1) * per),
                dst.get_mut(a * 4..a * 4 + 4),
                gx.get(a),
                gy.get(a),
            ) else {
                return;
            };
            O::store4(out, dist2bbox_anchor_f32(ltrb(bins), x, y, stride));
        }
    }

    /// One level's DFL `f16` boxes in native FP16: `deq` writes an anchor's f32 logits,
    /// `dfl_softmax_ltrb_f16_neon_fp16` its f16 distances, then `dist2bbox_f16_neon_fp16` runs
    /// over blocks of 64 anchors. Writes nothing unless the slices match the level.
    ///
    /// # Safety
    /// The CPU must support FEAT_FP16.
    #[inline(always)]
    unsafe fn dfl_level_f16<T>(
        input: &[T],
        level: &LevelPlan,
        dst: &mut [f16],
        mut deq: impl FnMut(&[T], &mut [f32]),
    ) {
        const BLOCK: usize = 64;
        let Some(n) = dfl_anchors(input, level, dst) else {
            return;
        };
        let rm = level.reg_max;
        let per = 4 * rm;
        let mut store = [0.0f32; 4 * MAX_REG_MAX];
        let (Some(logits), Some(gx), Some(gy)) = (
            store.get_mut(..per),
            level.grid_x.get(..n),
            level.grid_y.get(..n),
        ) else {
            return;
        };
        let mut ltrb = [f16::ZERO; 4 * BLOCK];
        let blocks = input
            .chunks(BLOCK * per)
            .zip(dst.chunks_mut(4 * BLOCK))
            .zip(gx.chunks(BLOCK).zip(gy.chunks(BLOCK)));
        for ((bins, out), (x, y)) in blocks {
            let Some(l) = ltrb.get_mut(..out.len()) else {
                return;
            };
            for (a, d) in bins.chunks_exact(per).zip(l.chunks_exact_mut(4)) {
                deq(a, logits);
                // SAFETY: the caller guarantees FEAT_FP16; `d` holds four values.
                unsafe { neon_fp16::dfl_softmax_ltrb_f16_neon_fp16(logits, rm, d) };
            }
            // SAFETY: as above; `l`, `out` and the grids cover the block's anchors.
            unsafe { neon_fp16::dist2bbox_f16_neon_fp16(l, x, y, level.stride, false, out) };
        }
    }

    /// Per-tensor DFL boxes on NEON: the decoder's NEON dequantize (`LtrbIn::dequant_neon`: the
    /// fused FMA for integers, exact for floats), then [`dfl_level`].
    ///
    /// # Safety
    /// [`Exp::Fp16`] needs FEAT_FP16.
    #[inline]
    pub(crate) unsafe fn dfl_neon<T: LtrbIn, O: BoxOut>(
        input: &[T],
        q: Quantization,
        level: &LevelPlan,
        exp: Exp,
        dst: &mut [O],
    ) {
        let q = q.into();
        // SAFETY: the caller's guarantee.
        unsafe { dfl_level(input, level, exp, dst, |b, d| T::dequant_neon(b, q, d)) };
    }

    /// Per-tensor DFL `f16` boxes in native FP16 after the decoder's NEON dequantize.
    ///
    /// # Safety
    /// The CPU must support FEAT_FP16.
    #[inline]
    pub(crate) unsafe fn dfl_f16_neon_fp16<T: LtrbIn>(
        input: &[T],
        q: Quantization,
        level: &LevelPlan,
        dst: &mut [f16],
    ) {
        let q = q.into();
        // SAFETY: the caller's guarantee.
        unsafe { dfl_level_f16(input, level, dst, |b, d| T::dequant_neon(b, q, d)) };
    }

    /// Per-channel DFL boxes on NEON: the scalar cell's exact per-bin dequantize, with the zero
    /// points and scales converted once per call, then [`BoxOut::per_channel`]. Returns `false`,
    /// writing nothing, unless there is one scale per bin and no zero points or one per bin.
    ///
    /// # Safety
    /// `fp16` needs FEAT_FP16.
    #[inline]
    pub(crate) unsafe fn dfl_per_channel_neon<T: PerChannelIn, O: BoxOut>(
        input: &[T],
        scales: &[f32],
        zps: &[i32],
        level: &LevelPlan,
        fp16: bool,
        dst: &mut [O],
    ) -> bool {
        let ch = 4 * level.reg_max;
        if ch > 4 * MAX_REG_MAX || scales.len() != ch || !(zps.is_empty() || zps.len() == ch) {
            return false;
        }
        let (mut z, mut s) = ([0.0f32; 4 * MAX_REG_MAX], [0.0f32; 4 * MAX_REG_MAX]);
        z.iter_mut().zip(zps).for_each(|(d, &v)| *d = v as f32);
        s.iter_mut().zip(scales).for_each(|(d, &v)| *d = v);
        let (Some(z), Some(s)) = (z.get(..ch), s.get(..ch)) else {
            return false;
        };
        let deq = |b: &[T], d: &mut [f32]| T::dequant_per_channel_neon(b, z, s, d);
        // SAFETY: the caller's guarantee.
        unsafe { O::per_channel(input, level, fp16, dst, deq) };
        true
    }

    /// One level's LTRB boxes on NEON: `prim::neon_ltrb`'s fused dequantize and dist2bbox, f32
    /// or the f32 boxes rounded once to f16.
    #[inline]
    pub(crate) fn ltrb_neon<T: LtrbIn, O: BoxOut>(
        input: &[T],
        q: Quantization,
        level: &LevelPlan,
        dst: &mut [O],
    ) {
        O::ltrb(input, q.into(), level, dst);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::per_scale::plan::LevelPlan;
    use crate::Quantization;
    use edgefirst_kernels::prim::grids::make_anchor_grid;

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
