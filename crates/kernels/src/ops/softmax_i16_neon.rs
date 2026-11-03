// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! NEON int16 softmax row kernel, bit-identical to the portable one.
//!
//! Eight lanes at a time: the row max, the input rescale (`vqrdmulhq_s32` is gemmlowp's
//! `SaturatingRoundingDoublingHighMul`; a sign fix-up makes `vrshlq_s32` round half away from
//! zero like `RoundingDivideByPOT`), the table interpolation, the i32 sum and the final rescale.
//! Only the two table reads per lane are scalar. The exps are cached in `dst` between passes.

#![cfg(target_arch = "aarch64")]

use std::arch::aarch64::*; // NOSONAR

use super::softmax_i16::{exp_q015, reciprocal, rescale, RowConsts};

/// NEON softmax of one row; processes `min(src.len(), dst.len())` elements.
///
/// # Safety
/// The CPU must support NEON, which every aarch64 CPU does.
#[target_feature(enable = "neon")]
pub(super) unsafe fn softmax_row_i16_neon(c: &RowConsts, src: &[i16], dst: &mut [i16]) {
    let n = src.len().min(dst.len());
    let (Some(src), Some(dst)) = (src.get(..n), dst.get_mut(..n)) else {
        return;
    };
    if n == 0 {
        return;
    }
    let max = row_max(src);
    let sum = exp_pass(c, max, src, dst);
    let (recip, shift) = reciprocal(c, sum);
    rescale_pass(recip, shift, dst);
}

#[target_feature(enable = "neon")]
fn row_max(src: &[i16]) -> i16 {
    let (chunks, tail) = src.as_chunks::<8>();
    let mut hi = i16::MIN;
    if !chunks.is_empty() {
        let mut v = vdupq_n_s16(i16::MIN);
        for ch in chunks {
            // SAFETY: `ch` is 8 readable i16s.
            v = vmaxq_s16(v, unsafe { vld1q_s16(ch.as_ptr()) });
        }
        hi = vmaxvq_s16(v);
    }
    tail.iter().fold(hi, |m, &x| m.max(x))
}

/// The input rescale of four differences: `MultiplyByQuantizedMultiplier` lane by lane.
#[target_feature(enable = "neon")]
fn rescale_diff(d: int32x4_t, left: int32x4_t, mult: int32x4_t, right: int32x4_t) -> int32x4_t {
    let p = vqrdmulhq_s32(vshlq_s32(d, left), mult);
    // -1 on negative lanes when shifting right, so the round-half-up shift rounds half away from zero.
    let fixup = vshrq_n_s32::<31>(vandq_s32(p, right));
    vrshlq_s32(vqaddq_s32(p, fixup), right)
}

/// `LUTLookup` of eight values; the two table reads per lane are scalar.
#[target_feature(enable = "neon")]
fn lookup8(v: int16x8_t, lut: &[i16; 513]) -> int16x8_t {
    let index = vreinterpretq_u16_s16(vaddq_s16(vshrq_n_s16::<7>(v), vdupq_n_s16(256)));
    let offset = vandq_s16(v, vdupq_n_s16(0x7f));
    let mut ix = [0u16; 8];
    // SAFETY: `ix` is 8 writable u16s.
    unsafe { vst1q_u16(ix.as_mut_ptr(), index) };
    // Each lane's `(lut[i], lut[i + 1])` as one u32, which LLVM reads with a single load.
    let mut pairs = [0u32; 8];
    for (p, &i) in pairs.iter_mut().zip(&ix) {
        // Indices are already 0..=511; the mask lets the bounds checks fold away.
        let i = usize::from(i & 0x1ff);
        *p = u32::from(lut[i] as u16) | (u32::from(lut[i + 1] as u16) << 16);
    }
    // SAFETY: `pairs` is 8 readable u32s.
    let (a, b) = unsafe {
        (
            vreinterpretq_s16_u32(vld1q_u32(pairs.as_ptr())),
            vreinterpretq_s16_u32(vld1q_u32(pairs.as_ptr().add(4))),
        )
    };
    let (base, next) = (vuzp1q_s16(a, b), vuzp2q_s16(a, b));
    let slope = vsubq_s16(next, base);
    // `(slope * offset + 64) >> 7` fits in an i16, so the narrowing keeps it exactly.
    let lo = vrshrn_n_s32::<7>(vmull_s16(vget_low_s16(slope), vget_low_s16(offset)));
    let delta = vrshrn_high_n_s32::<7>(lo, vmull_high_s16(slope, offset));
    vaddq_s16(base, delta)
}

/// Writes `exp(x - max)` for every element into `dst` and returns their wrapping i32 sum.
#[target_feature(enable = "neon")]
fn exp_pass(c: &RowConsts, max: i16, src: &[i16], dst: &mut [i16]) -> i32 {
    let maxv = vdupq_n_s16(max);
    let left = vdupq_n_s32(c.input_left_shift.clamp(0, 31));
    let right = vdupq_n_s32(-c.input_left_shift.saturating_neg().clamp(0, 31));
    let mult = vdupq_n_s32(c.input_multiplier);
    let recenter = vdupq_n_s32(32767);
    let mut acc = vdupq_n_s32(0);
    let (chunks, tail) = src.as_chunks::<8>();
    let (dchunks, dtail) = dst.as_chunks_mut::<8>();
    for (s, d) in chunks.iter().zip(dchunks.iter_mut()) {
        // SAFETY: `s` is 8 readable i16s.
        let x = unsafe { vld1q_s16(s.as_ptr()) };
        let lo = rescale_diff(
            vsubl_s16(vget_low_s16(x), vget_low_s16(maxv)),
            left,
            mult,
            right,
        );
        let hi = rescale_diff(vsubl_high_s16(x, maxv), left, mult, right);
        // Recentre, then saturate to i16 as the reference's clamp does.
        let sym = vqmovn_high_s32(vqmovn_s32(vaddq_s32(lo, recenter)), vaddq_s32(hi, recenter));
        let e = lookup8(sym, &c.exp_lut);
        // SAFETY: `d` is 8 writable i16s.
        unsafe { vst1q_s16(d.as_mut_ptr(), e) };
        acc = vpadalq_s16(acc, e);
    }
    let mut sum = vaddvq_s32(acc);
    for (d, &x) in dtail.iter_mut().zip(tail) {
        *d = exp_q015(c, x, max);
        sum = sum.wrapping_add(i32::from(*d));
    }
    sum
}

/// Replaces each cached exp in `dst` with `(exp * recip + round) >> shift`, clamped to `0..=32767`.
#[target_feature(enable = "neon")]
fn rescale_pass(recip: i16, shift: u32, dst: &mut [i16]) {
    let r = vdupq_n_s16(recip);
    // `shift` is at most 31, so the negation cannot wrap.
    let sh = vdupq_n_s32(-(shift.min(31) as i32));
    let zero = vdupq_n_s16(0);
    let (chunks, tail) = dst.as_chunks_mut::<8>();
    for d in chunks {
        // SAFETY: `d` is 8 readable and writable i16s.
        let e = unsafe { vld1q_s16(d.as_ptr()) };
        let lo = vrshlq_s32(vmull_s16(vget_low_s16(e), vget_low_s16(r)), sh);
        let hi = vrshlq_s32(vmull_high_s16(e, r), sh);
        let q = vmaxq_s16(vqmovn_high_s32(vqmovn_s32(lo), hi), zero);
        // SAFETY: `d` is 8 writable i16s.
        unsafe { vst1q_s16(d.as_mut_ptr(), q) };
    }
    for d in tail {
        *d = rescale(*d, recip, shift);
    }
}

#[cfg(test)]
mod tests {
    use super::super::softmax_i16::softmax_row_i16_portable;
    use super::*;
    use crate::tflite::quantize_multiplier;
    use rand::{RngExt, SeedableRng};

    fn check(c: &RowConsts, row: &[i16], what: &str) {
        let mut want = vec![0x5555i16; row.len()];
        let mut got = vec![0x5555i16; row.len()];
        softmax_row_i16_portable(c, row, &mut want);
        // SAFETY: NEON is part of the aarch64 baseline.
        unsafe { softmax_row_i16_neon(c, row, &mut got) };
        assert_eq!(
            got,
            want,
            "{what}: len {} multiplier {} shift {}",
            row.len(),
            c.input_multiplier,
            c.input_left_shift
        );
    }

    #[test]
    fn neon_row_bit_identical_to_portable() {
        let mut rng = rand::rngs::StdRng::seed_from_u64(0x5016_2348);
        let mut rescales: Vec<(i32, i32)> = [1e-9f64, 1e-6, 0.001, 0.05, 0.5, 1.0, 3.2, 100.0, 3e4]
            .iter()
            .map(|&r| quantize_multiplier(r))
            .collect();
        // Raw rescales spanning every right shift and left shifts that wrap in the i32 shift.
        for shift in [-31, -30, -17, -1, 0, 1, 15, 16, 24, 30, 31] {
            rescales.push((1 << 30, shift));
            rescales.push((i32::MAX, shift));
            rescales.push((rng.random_range(1 << 30..=i32::MAX), shift));
        }
        rescales.push((0, 0));
        let mut lens: Vec<usize> = (1..=70).collect();
        lens.extend([255, 256, 257, 1009, 4097]);
        for &(m, s) in &rescales {
            let c = RowConsts::new(m, s);
            for &n in &lens {
                for &span in &[1i32, 64, 4096, 65536] {
                    let lo = rng.random_range(-32768..=(32768 - span));
                    let row: Vec<i16> = (0..n)
                        .map(|_| (lo + rng.random_range(0..span)) as i16)
                        .collect();
                    check(&c, &row, &format!("random span {span}"));
                }
                for v in [i16::MIN, -1, 0, 1, i16::MAX] {
                    check(&c, &vec![v; n], "all equal");
                }
                let mut ext: Vec<i16> = (0..n)
                    .map(|i| if i % 2 == 0 { i16::MIN } else { i16::MAX })
                    .collect();
                check(&c, &ext, "alternating extremes");
                ext.reverse();
                check(&c, &ext, "alternating extremes reversed");
            }
        }
    }

    #[test]
    fn neon_row_handles_mismatched_and_empty_slices() {
        let c = RowConsts::new(1_717_986_918, 2);
        let src: Vec<i16> = (0..19).map(|v| v * 1000 - 9000).collect();
        let mut dst = [0x5555i16; 17];
        // SAFETY: NEON is part of the aarch64 baseline.
        unsafe { softmax_row_i16_neon(&c, &src, &mut dst) };
        let mut want = [0x5555i16; 17];
        softmax_row_i16_portable(&c, &src, &mut want);
        assert_eq!(dst, want);
        let mut empty: [i16; 0] = [];
        // SAFETY: NEON is part of the aarch64 baseline.
        unsafe { softmax_row_i16_neon(&c, &[], &mut empty) };
    }
}
