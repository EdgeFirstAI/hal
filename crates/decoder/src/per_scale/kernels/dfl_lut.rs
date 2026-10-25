// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! DFL box decode of 8-bit bins through an exp lookup table.
//!
//! With per-tensor quantization a bin's logit is `(q - zp) * scale`, so its
//! softmax weight relative to the largest bin `q_max` of its side is
//! `exp(-(q_max - q) * scale)`. For 8-bit inputs `q_max - q` is an integer in
//! `0..=255`: the weight is one lookup in a 256-entry table that depends
//! only on the scale, the zero point cancels, and the side's maximum is an
//! integer maximum over the raw bins. The largest bin weighs exactly 1, so a
//! side's weight sum is never below 1.
//!
//! Each side accumulates its weight sum and bin-weighted sum over even and
//! odd bins separately, then adds the two. The NEON kernel holds the four
//! sides in the four lanes of a vector, and the portable kernel performs the
//! same additions in the same order, so both produce identical boxes.

use super::box_primitives::dist2bbox_anchor_f32;
use crate::per_scale::plan::LevelPlan;
use half::f16;

/// `exp(-k * scale)` for `k` in `0..=255`, the softmax weight of a bin `k`
/// raw steps below its side's largest bin.
#[derive(Debug, Clone)]
pub(crate) struct DflExpTable {
    scale_bits: u32,
    exp: [f32; 256],
}

impl DflExpTable {
    pub(crate) fn new(scale: f32) -> Self {
        let mut table = Self {
            scale_bits: scale.to_bits(),
            exp: [0.0; 256],
        };
        table.fill(scale);
        table
    }

    /// Rebuild the table if it was built for a different scale.
    pub(crate) fn update(&mut self, scale: f32) {
        if !self.is_for(scale) {
            self.scale_bits = scale.to_bits();
            self.fill(scale);
        }
    }

    /// Whether a table can stand for `scale`: the weights `exp(-k * scale)`
    /// are at most 1 only for a finite, positive scale. Any other scale
    /// overflows the table, and the bins are decoded by the max-subtracted
    /// softmax instead.
    pub(crate) fn supports(scale: f32) -> bool {
        scale.is_finite() && scale > 0.0
    }

    pub(crate) fn is_for(&self, scale: f32) -> bool {
        self.scale_bits == scale.to_bits()
    }

    fn fill(&mut self, scale: f32) {
        let scale = f64::from(scale);
        for (k, e) in self.exp.iter_mut().enumerate() {
            *e = (-(k as f64) * scale).exp() as f32;
        }
    }
}

/// An 8-bit DFL bin type.
pub(crate) trait DflBin: Copy {
    fn raw(self) -> i32;
}

impl DflBin for i8 {
    #[inline(always)]
    fn raw(self) -> i32 {
        i32::from(self)
    }
}

impl DflBin for u8 {
    #[inline(always)]
    fn raw(self) -> i32 {
        i32::from(self)
    }
}

/// A box output element type.
pub(crate) trait BoxOut: Copy {
    fn store(dst: &mut [Self], xywh: [f32; 4]);
}

impl BoxOut for f32 {
    #[inline(always)]
    fn store(dst: &mut [Self], xywh: [f32; 4]) {
        dst[..4].copy_from_slice(&xywh);
    }
}

impl BoxOut for f16 {
    #[inline(always)]
    fn store(dst: &mut [Self], xywh: [f32; 4]) {
        for (d, v) in dst[..4].iter_mut().zip(xywh) {
            *d = f16::from_f32(v);
        }
    }
}

/// Portable decode of one level's NHWC `[h * w, 4 * reg_max]` 8-bit DFL bins
/// into `[xc, yc, w, h]` per anchor.
pub(crate) fn decode_dfl_lut<T: DflBin, O: BoxOut>(
    input: &[T],
    table: &DflExpTable,
    level: &LevelPlan,
    dst: &mut [O],
) {
    let reg_max = level.reg_max;
    let n = level.h * level.w;
    debug_assert!(reg_max > 0);
    debug_assert_eq!(input.len(), n * 4 * reg_max);
    debug_assert_eq!(dst.len(), 4 * n);
    let exp = &table.exp;
    for (a, (bins, out)) in input
        .chunks_exact(4 * reg_max)
        .zip(dst.chunks_exact_mut(4))
        .enumerate()
    {
        let mut ltrb = [0.0_f32; 4];
        for (d, side) in ltrb.iter_mut().zip(bins.chunks_exact(reg_max)) {
            let max = side.iter().map(|b| b.raw()).max().unwrap_or(0);
            let (mut sum_even, mut sum_odd) = (0.0_f32, 0.0_f32);
            let (mut ws_even, mut ws_odd) = (0.0_f32, 0.0_f32);
            for (i, b) in side.iter().enumerate() {
                let e = exp[(max - b.raw()) as usize];
                if i % 2 == 0 {
                    sum_even += e;
                    ws_even += e * i as f32;
                } else {
                    sum_odd += e;
                    ws_odd += e * i as f32;
                }
            }
            *d = (ws_even + ws_odd) / (sum_even + sum_odd);
        }
        let xywh = dist2bbox_anchor_f32(ltrb, level.grid_x[a], level.grid_y[a], level.stride);
        O::store(out, xywh);
    }
}

#[cfg(target_arch = "aarch64")]
pub(crate) use neon::decode_dfl_lut_neon;

#[cfg(target_arch = "aarch64")]
mod neon {
    use super::{decode_dfl_lut, BoxOut, DflBin, DflExpTable};
    use crate::per_scale::plan::LevelPlan;
    use std::arch::aarch64::*; // NOSONAR

    /// An 8-bit DFL bin type the NEON kernel can take the side maxima of.
    pub(crate) trait NeonDflBin: DflBin {
        /// Lane `s` of the result is the largest raw bin of side `s`, as the
        /// bit pattern of the input type.
        ///
        /// # Safety
        /// NEON must be available.
        unsafe fn side_max(s: [uint8x16_t; 4]) -> uint8x16_t;
    }

    impl NeonDflBin for i8 {
        #[inline(always)]
        unsafe fn side_max(s: [uint8x16_t; 4]) -> uint8x16_t {
            unsafe {
                let ab = vpmaxq_s8(vreinterpretq_s8_u8(s[0]), vreinterpretq_s8_u8(s[1]));
                let cd = vpmaxq_s8(vreinterpretq_s8_u8(s[2]), vreinterpretq_s8_u8(s[3]));
                let m = vpmaxq_s8(ab, cd);
                let m = vpmaxq_s8(m, m);
                vreinterpretq_u8_s8(vpmaxq_s8(m, m))
            }
        }
    }

    impl NeonDflBin for u8 {
        #[inline(always)]
        unsafe fn side_max(s: [uint8x16_t; 4]) -> uint8x16_t {
            unsafe {
                let m = vpmaxq_u8(vpmaxq_u8(s[0], s[1]), vpmaxq_u8(s[2], s[3]));
                let m = vpmaxq_u8(m, m);
                vpmaxq_u8(m, m)
            }
        }
    }

    /// [`decode_dfl_lut`] for `reg_max == 16` with the four sides in the
    /// lanes of one vector; other `reg_max` values take the portable kernel.
    pub(crate) fn decode_dfl_lut_neon<T: NeonDflBin, O: BoxOut>(
        input: &[T],
        table: &DflExpTable,
        level: &LevelPlan,
        dst: &mut [O],
    ) {
        if level.reg_max != 16 {
            return decode_dfl_lut(input, table, level, dst);
        }
        let n = level.h * level.w;
        assert_eq!(input.len(), n * 64);
        assert_eq!(dst.len(), 4 * n);
        assert!(level.grid_x.len() >= n && level.grid_y.len() >= n);
        // SAFETY: NEON is mandatory on aarch64, and the lengths are checked
        // above.
        unsafe { level_16(input, table, level, dst) }
    }

    /// Weighted and plain bin sums of one anchor's 64 bins at `p`, side `s`
    /// in lane `s`, accumulated as `decode_dfl_lut` does.
    #[inline(always)]
    unsafe fn anchor_sums<T: NeonDflBin>(
        p: *const u8,
        exp: *const f32,
    ) -> (float32x4_t, float32x4_t) {
        unsafe {
            let sides = [
                vld1q_u8(p),
                vld1q_u8(p.add(16)),
                vld1q_u8(p.add(32)),
                vld1q_u8(p.add(48)),
            ];
            let m = T::side_max(sides);
            // The true difference lies in 0..=255 for either signedness, so
            // the wrapping u8 subtraction yields it exactly.
            let k = [
                vreinterpretq_u64_u8(vsubq_u8(vdupq_laneq_u8::<0>(m), sides[0])),
                vreinterpretq_u64_u8(vsubq_u8(vdupq_laneq_u8::<1>(m), sides[1])),
                vreinterpretq_u64_u8(vsubq_u8(vdupq_laneq_u8::<2>(m), sides[2])),
                vreinterpretq_u64_u8(vsubq_u8(vdupq_laneq_u8::<3>(m), sides[3])),
            ];
            let lo = [
                vgetq_lane_u64::<0>(k[0]),
                vgetq_lane_u64::<0>(k[1]),
                vgetq_lane_u64::<0>(k[2]),
                vgetq_lane_u64::<0>(k[3]),
            ];
            let hi = [
                vgetq_lane_u64::<1>(k[0]),
                vgetq_lane_u64::<1>(k[1]),
                vgetq_lane_u64::<1>(k[2]),
                vgetq_lane_u64::<1>(k[3]),
            ];
            // The weights of bin `8 * half + j` of the four sides.
            macro_rules! weights {
                ($w:ident, $j:literal) => {{
                    let v = [
                        *exp.add((($w[0] >> (8 * $j)) & 0xff) as usize),
                        *exp.add((($w[1] >> (8 * $j)) & 0xff) as usize),
                        *exp.add((($w[2] >> (8 * $j)) & 0xff) as usize),
                        *exp.add((($w[3] >> (8 * $j)) & 0xff) as usize),
                    ];
                    vld1q_f32(v.as_ptr())
                }};
            }
            // Bin 0 adds nothing to the weighted sum, and bin 1 adds its
            // weight unscaled; adding either to a zero sum is exact.
            let mut sum_even = weights!(lo, 0);
            let e = weights!(lo, 1);
            let mut sum_odd = e;
            let mut ws_odd = e;
            let e = weights!(lo, 2);
            sum_even = vaddq_f32(sum_even, e);
            let mut ws_even = vmulq_n_f32(e, 2.0);
            macro_rules! bin {
                ($w:ident, $j:literal, $bin:literal, $sum:ident, $ws:ident) => {{
                    let e = weights!($w, $j);
                    $sum = vaddq_f32($sum, e);
                    $ws = vaddq_f32($ws, vmulq_n_f32(e, $bin));
                }};
            }
            bin!(lo, 3, 3.0, sum_odd, ws_odd);
            bin!(lo, 4, 4.0, sum_even, ws_even);
            bin!(lo, 5, 5.0, sum_odd, ws_odd);
            bin!(lo, 6, 6.0, sum_even, ws_even);
            bin!(lo, 7, 7.0, sum_odd, ws_odd);
            bin!(hi, 0, 8.0, sum_even, ws_even);
            bin!(hi, 1, 9.0, sum_odd, ws_odd);
            bin!(hi, 2, 10.0, sum_even, ws_even);
            bin!(hi, 3, 11.0, sum_odd, ws_odd);
            bin!(hi, 4, 12.0, sum_even, ws_even);
            bin!(hi, 5, 13.0, sum_odd, ws_odd);
            bin!(hi, 6, 14.0, sum_even, ws_even);
            bin!(hi, 7, 15.0, sum_odd, ws_odd);
            (vaddq_f32(ws_even, ws_odd), vaddq_f32(sum_even, sum_odd))
        }
    }

    /// `[l, t, r, b]` distances of anchor `a` to `[xc, yc, w, h]` in `out`:
    /// `(g + (r - l, b - t) / 2) * stride` and `(l + r, t + b) * stride`, the
    /// operations of `dist2bbox_anchor_f32`.
    #[inline(always)]
    unsafe fn store_box<O: BoxOut>(
        d: float32x4_t,
        level: &LevelPlan,
        a: usize,
        stride: float32x2_t,
        out: &mut [O],
    ) {
        unsafe {
            let rev = vextq_f32::<2>(d, d);
            let diff = vget_low_f32(vsubq_f32(rev, d));
            let size = vget_low_f32(vaddq_f32(d, rev));
            let g = vset_lane_f32::<1>(
                *level.grid_y.get_unchecked(a),
                vdup_n_f32(*level.grid_x.get_unchecked(a)),
            );
            let xy = vmul_f32(vadd_f32(g, vmul_n_f32(diff, 0.5)), stride);
            let wh = vmul_f32(size, stride);
            let mut xywh = [0.0_f32; 4];
            vst1q_f32(xywh.as_mut_ptr(), vcombine_f32(xy, wh));
            O::store(out, xywh);
        }
    }

    #[target_feature(enable = "neon")]
    unsafe fn level_16<T: NeonDflBin, O: BoxOut>(
        input: &[T],
        table: &DflExpTable,
        level: &LevelPlan,
        dst: &mut [O],
    ) {
        unsafe {
            let exp = table.exp.as_ptr();
            let src = input.as_ptr().cast::<u8>();
            let stride = vdup_n_f32(level.stride);
            let last = (dst.len() / 4).saturating_sub(1);
            // Two anchors per step give the in-order Cortex-A55 independent
            // work to issue while the other anchor's loads and divide retire.
            let mut pairs = dst.chunks_exact_mut(8);
            for (i, out) in pairs.by_ref().enumerate() {
                let a = 2 * i;
                let p = src.add(a * 64);
                let (ws0, sum0) = anchor_sums::<T>(p, exp);
                let (ws1, sum1) = anchor_sums::<T>(p.add(64), exp);
                let d0 = vdivq_f32(ws0, sum0);
                let d1 = vdivq_f32(ws1, sum1);
                let (out0, out1) = out.split_at_mut(4);
                store_box(d0, level, a, stride, out0);
                store_box(d1, level, a + 1, stride, out1);
            }
            let rest = pairs.into_remainder();
            if !rest.is_empty() {
                let (ws, sum) = anchor_sums::<T>(src.add(last * 64), exp);
                store_box(vdivq_f32(ws, sum), level, last, stride, rest);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::per_scale::kernels::grids::make_anchor_grid;
    use crate::per_scale::kernels::level_box::{
        decode_box_level_dfl_f32_to_f32, decode_box_level_dfl_i8_to_f32,
        decode_box_level_dfl_u8_to_f32,
    };
    use crate::per_scale::plan::Layout;
    use crate::Quantization;

    fn level(w: usize, h: usize, reg_max: usize) -> LevelPlan {
        let (grid_x, grid_y) = make_anchor_grid(h, w);
        LevelPlan {
            stride: 16.0,
            h,
            w,
            reg_max,
            anchor_offset: 0,
            grid_x,
            grid_y,
            box_shape: vec![1, h, w, 4 * reg_max].into_boxed_slice(),
            score_shape: vec![1, h, w, 1].into_boxed_slice(),
            mc_shape: None,
            layout: Layout::Nhwc,
        }
    }

    /// DFL logits shaped like a trained head: each side peaks at a bin and
    /// falls off, with noise, and saturated or flat sides mixed in.
    fn bins(n: usize, reg_max: usize, seed: u64) -> Vec<i8> {
        let mut x = seed | 1;
        let mut next = move || {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            (x >> 32) as i32
        };
        let mut v = Vec::with_capacity(n * 4 * reg_max);
        for _ in 0..n * 4 {
            let kind = next().rem_euclid(16);
            let peak = next().rem_euclid(reg_max as i32);
            let top = next().rem_euclid(200) - 100;
            let fall = next().rem_euclid(12);
            for i in 0..reg_max as i32 {
                let b = match kind {
                    0 => 127,
                    1 => -128,
                    2 => {
                        if i == peak {
                            127
                        } else {
                            -128
                        }
                    }
                    _ => top - fall * (i - peak).abs() + next().rem_euclid(9) - 4,
                };
                v.push(b.clamp(-128, 127) as i8);
            }
        }
        v
    }

    /// The f64 softmax expectation of each side, through dist2bbox in f64.
    fn reference(input: &[i8], q: Quantization, level: &LevelPlan) -> Vec<f64> {
        let rm = level.reg_max;
        let mut out = Vec::new();
        for (a, anchor) in input.chunks_exact(4 * rm).enumerate() {
            let mut d = [0.0_f64; 4];
            for (s, side) in anchor.chunks_exact(rm).enumerate() {
                let x: Vec<f64> = side
                    .iter()
                    .map(|&b| (f64::from(b) - f64::from(q.zero_point)) * f64::from(q.scale))
                    .collect();
                let m = x.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                let e: Vec<f64> = x.iter().map(|v| (v - m).exp()).collect();
                let sum: f64 = e.iter().sum();
                d[s] = e.iter().enumerate().map(|(i, e)| e * i as f64).sum::<f64>() / sum;
            }
            let (gx, gy, st) = (
                f64::from(level.grid_x[a]),
                f64::from(level.grid_y[a]),
                f64::from(level.stride),
            );
            out.extend([
                (gx + (d[2] - d[0]) * 0.5) * st,
                (gy + (d[3] - d[1]) * 0.5) * st,
                (d[0] + d[2]) * st,
                (d[1] + d[3]) * st,
            ]);
        }
        out
    }

    fn max_err(got: &[f32], want: &[f64]) -> f64 {
        got.iter()
            .zip(want)
            .map(|(&g, &w)| (f64::from(g) - w).abs())
            .fold(0.0, f64::max)
    }

    #[test]
    fn table_holds_exp_of_negative_steps() {
        let t = DflExpTable::new(0.25);
        assert_eq!(t.exp[0], 1.0);
        assert!((f64::from(t.exp[4]) - (-1.0_f64).exp()).abs() < 1e-7);
        assert!(t.exp.windows(2).all(|w| w[1] <= w[0]));
        let mut u = t.clone();
        u.update(0.25);
        assert_eq!(u.exp, t.exp);
        u.update(0.5);
        assert!(u.is_for(0.5) && !u.is_for(0.25));
        assert!((f64::from(u.exp[2]) - (-1.0_f64).exp()).abs() < 1e-7);
    }

    /// The table kernel is at least as close to an f64 reference as the f32
    /// dequant-then-softmax kernel, and within a few f32 rounding steps.
    #[test]
    fn matches_f64_reference_closer_than_f32_softmax() {
        for (scale, zp) in [(0.147, 8), (0.1, -40), (0.02, 0), (0.6, 100)] {
            let q = Quantization::new(scale, zp);
            let table = DflExpTable::new(scale);
            for reg_max in [16, 8, 7] {
                let lvl = level(9, 5, reg_max);
                let input = bins(45, reg_max, 7 + reg_max as u64);
                let want = reference(&input, q, &lvl);
                let mut lut = vec![0.0_f32; 4 * 45];
                decode_dfl_lut(&input, &table, &lvl, &mut lut);
                let mut f32_softmax = vec![0.0_f32; 4 * 45];
                decode_box_level_dfl_i8_to_f32(&input, q, &lvl, &mut f32_softmax);
                let (e_lut, e_f32) = (max_err(&lut, &want), max_err(&f32_softmax, &want));
                assert!(e_lut < 2e-4, "scale={scale} rm={reg_max}: lut err {e_lut}");
                assert!(
                    e_lut <= e_f32 * 1.01 + 1e-5,
                    "scale={scale} rm={reg_max}: lut {e_lut} f32 softmax {e_f32}"
                );
            }
        }
    }

    /// Unsigned bins decode as their signed equivalents shifted by 128.
    #[test]
    fn unsigned_bins_match_shifted_signed_bins() {
        let lvl = level(7, 3, 16);
        let input = bins(21, 16, 99);
        let unsigned: Vec<u8> = input.iter().map(|&b| (i16::from(b) + 128) as u8).collect();
        let table = DflExpTable::new(0.12);
        let mut s = vec![0.0_f32; 84];
        let mut u = vec![0.0_f32; 84];
        decode_dfl_lut(&input, &table, &lvl, &mut s);
        decode_dfl_lut(&unsigned, &table, &lvl, &mut u);
        assert_eq!(s, u);
        let mut legacy = vec![0.0_f32; 84];
        decode_box_level_dfl_u8_to_f32(&unsigned, Quantization::new(0.12, 128), &lvl, &mut legacy);
        assert!(
            max_err(
                &u,
                &legacy.iter().map(|&v| f64::from(v)).collect::<Vec<_>>()
            ) < 1e-3
        );
    }

    #[test]
    fn one_hot_bins_decode_to_their_index() {
        let lvl = level(1, 1, 16);
        let mut input = [-128_i8; 64];
        for (side, bin) in [3, 5, 7, 9].into_iter().enumerate() {
            input[side * 16 + bin] = 127;
        }
        let mut out = [0.0_f32; 4];
        decode_dfl_lut(&input, &DflExpTable::new(0.5), &lvl, &mut out);
        let mut want = [0.0_f32; 4];
        let mut logits = [0.0_f32; 64];
        for (side, bin) in [3, 5, 7, 9].into_iter().enumerate() {
            logits[side * 16 + bin] = 127.5;
        }
        decode_box_level_dfl_f32_to_f32(&logits, Quantization::identity(), &lvl, &mut want);
        assert_eq!(out, want);
    }

    #[test]
    fn f16_output_rounds_the_f32_boxes() {
        let lvl = level(4, 4, 16);
        let input = bins(16, 16, 3);
        let table = DflExpTable::new(0.147);
        let mut f = vec![0.0_f32; 64];
        let mut h = vec![f16::ZERO; 64];
        decode_dfl_lut(&input, &table, &lvl, &mut f);
        decode_dfl_lut(&input, &table, &lvl, &mut h);
        for (a, b) in f.iter().zip(&h) {
            assert_eq!(f16::from_f32(*a), *b);
        }
    }

    /// The NEON kernel produces the portable kernel's boxes bit for bit.
    #[cfg(target_arch = "aarch64")]
    #[test]
    fn neon_matches_portable_bit_for_bit() {
        for (scale, reg_max) in [(0.147, 16), (0.02, 16), (0.9, 16), (0.1, 8)] {
            let table = DflExpTable::new(scale);
            let lvl = level(17, 3, reg_max);
            let input = bins(51, reg_max, 11);
            let unsigned: Vec<u8> = input.iter().map(|&b| b as u8).collect();
            let mut want = vec![0.0_f32; 204];
            let mut got = vec![0.0_f32; 204];
            decode_dfl_lut(&input, &table, &lvl, &mut want);
            decode_dfl_lut_neon(&input, &table, &lvl, &mut got);
            assert_eq!(got, want, "i8 scale={scale} rm={reg_max}");
            decode_dfl_lut(&unsigned, &table, &lvl, &mut want);
            decode_dfl_lut_neon(&unsigned, &table, &lvl, &mut got);
            assert_eq!(got, want, "u8 scale={scale} rm={reg_max}");
            let mut want16 = vec![f16::ZERO; 204];
            let mut got16 = vec![f16::ZERO; 204];
            decode_dfl_lut(&input, &table, &lvl, &mut want16);
            decode_dfl_lut_neon(&input, &table, &lvl, &mut got16);
            assert_eq!(got16, want16, "f16 scale={scale} rm={reg_max}");
        }
    }
}
