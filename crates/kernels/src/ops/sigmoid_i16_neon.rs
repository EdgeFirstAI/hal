// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! NEON int16 `LOGISTIC`, bit-identical to the scalar ports in `tflite.rs`.
//!
//! The power-of-two kernel is gemmlowp's `logistic` on `FixedPoint<int16x8_t, 3>`, as
//! `optimized_ops::Logistic` runs it, eight lanes per vector. The reference kernel is
//! `reference_integer_ops::Logistic` in `int32x4_t` lanes; its 256-entry table is read with one
//! 32-bit load per lane from a table of adjacent entry pairs.

#![cfg(target_arch = "aarch64")]

use std::arch::aarch64::*; // NOSONAR

use crate::tflite::{logistic_i16_pot, logistic_i16_reference, q16, SIGMOID_TABLE_U16};

/// gemmlowp `RoundingDivideByPOT<int16x8_t>`: negative lanes are lowered by one, then `VRSHR`.
#[inline(always)]
fn rounding_divide_by_pot<const E: i32>(x: int16x8_t) -> int16x8_t {
    // SAFETY: NEON is part of the aarch64 baseline.
    unsafe { vrshrq_n_s16::<E>(vqaddq_s16(x, vshrq_n_s16::<15>(x))) }
}

/// gemmlowp `exp_on_interval_between_negative_one_quarter_and_0_excl` on Q0.15.
#[inline(always)]
fn exp_on_interval_between_negative_one_quarter_and_0_excl(a: int16x8_t) -> int16x8_t {
    // SAFETY: NEON is part of the aarch64 baseline.
    unsafe {
        let x = vaddq_s16(a, vdupq_n_s16(1 << 12));
        let x2 = vqrdmulhq_s16(x, x);
        let x3 = vqrdmulhq_s16(x2, x);
        let x4 = vqrdmulhq_s16(x2, x2);
        let x4_over_4 = rounding_divide_by_pot::<2>(x4);
        let poly = rounding_divide_by_pot::<1>(vaddq_s16(
            vqrdmulhq_n_s16(vaddq_s16(x4_over_4, x3), q16::ONE_THIRD),
            x2,
        ));
        let c = vdupq_n_s16(q16::EXP_MINUS_1_8);
        vqaddq_s16(c, vqrdmulhq_s16(c, vaddq_s16(x, poly)))
    }
}

/// gemmlowp `exp_on_negative_values` from Q3.12 to Q0.15.
#[inline(always)]
fn exp_on_negative_values_q3(a: int16x8_t) -> int16x8_t {
    // SAFETY: NEON is part of the aarch64 baseline.
    unsafe {
        let quarter = vdupq_n_s16(1 << 10);
        let a_mod_quarter_minus_one_quarter =
            vsubq_s16(vandq_s16(a, vdupq_n_s16((1 << 10) - 1)), quarter);
        let mut result = exp_on_interval_between_negative_one_quarter_and_0_excl(
            vqshlq_n_s16::<3>(a_mod_quarter_minus_one_quarter),
        );
        let remainder = vsubq_s16(a_mod_quarter_minus_one_quarter, a);
        for (bit, multiplier) in q16::EXP_BARREL {
            let apply = vtstq_s16(remainder, vdupq_n_s16(1 << bit));
            result = vbslq_s16(apply, vqrdmulhq_n_s16(result, multiplier), result);
        }
        vbslq_s16(vceqzq_s16(a), vdupq_n_s16(i16::MAX), result)
    }
}

/// gemmlowp `one_over_one_plus_x_for_x_in_0_1` on Q0.15, by Newton-Raphson in Q2.13.
#[inline(always)]
fn one_over_one_plus_x_for_x_in_0_1(a: int16x8_t) -> int16x8_t {
    // SAFETY: NEON is part of the aarch64 baseline.
    unsafe {
        // `a` is an exp result, never negative, where `VRHADD` equals `RoundingHalfSum`.
        let half_denominator = vrhaddq_s16(a, vdupq_n_s16(i16::MAX));
        let mut x = vaddq_s16(
            vdupq_n_s16(q16::C48_OVER_17),
            vqrdmulhq_n_s16(half_denominator, q16::NEG_32_OVER_17),
        );
        let q2_one = vdupq_n_s16(1 << 13);
        for _ in 0..3 {
            let one_minus_half_denominator_times_x =
                vsubq_s16(q2_one, vqrdmulhq_s16(half_denominator, x));
            x = vaddq_s16(
                x,
                vqshlq_n_s16::<2>(vqrdmulhq_s16(x, one_minus_half_denominator_times_x)),
            );
        }
        vqshlq_n_s16::<1>(x)
    }
}

/// gemmlowp `logistic` from Q3.12 to Q0.15 on eight lanes; negations wrap.
#[inline(always)]
fn logistic_pot8(x: int16x8_t) -> int16x8_t {
    // SAFETY: NEON is part of the aarch64 baseline.
    unsafe {
        let positive = vcgtzq_s16(x);
        let abs = vbslq_s16(positive, x, vnegq_s16(x));
        let result_if_positive =
            one_over_one_plus_x_for_x_in_0_1(exp_on_negative_values_q3(vnegq_s16(abs)));
        let result_if_negative = vsubq_s16(vdupq_n_s16(i16::MAX), result_if_positive);
        vbslq_s16(
            vceqzq_s16(x),
            vdupq_n_s16(q16::ONE_HALF),
            vbslq_s16(positive, result_if_positive, result_if_negative),
        )
    }
}

/// Maps the first `min(src.len(), dst.len())` elements through `f`, `V` independent 8-lane
/// vectors per step to hide `f`'s dependency chains on in-order cores, then single vectors,
/// then `scalar` on the last few.
#[inline(always)]
fn map_vectors<const V: usize>(
    src: &[i16],
    dst: &mut [i16],
    f: impl Fn(int16x8_t) -> int16x8_t,
    scalar: impl Fn(i16) -> i16,
) {
    let n = src.len().min(dst.len());
    let (Some(src), Some(dst)) = (src.get(..n), dst.get_mut(..n)) else {
        return;
    };
    let mut s = src.chunks_exact(8 * V);
    let mut d = dst.chunks_exact_mut(8 * V);
    for (si, di) in (&mut s).zip(&mut d) {
        let r: [int16x8_t; V] = core::array::from_fn(|i| {
            // SAFETY: both chunks hold `8 · V` elements.
            f(unsafe { vld1q_s16(si.as_ptr().add(8 * i)) })
        });
        for (i, r) in r.into_iter().enumerate() {
            // SAFETY: as above.
            unsafe { vst1q_s16(di.as_mut_ptr().add(8 * i), r) };
        }
    }
    let mut s = s.remainder().chunks_exact(8);
    let mut d = d.into_remainder().chunks_exact_mut(8);
    for (si, di) in (&mut s).zip(&mut d) {
        // SAFETY: both chunks hold 8 elements.
        unsafe { vst1q_s16(di.as_mut_ptr(), f(vld1q_s16(si.as_ptr()))) };
    }
    for (o, &x) in d.into_remainder().iter_mut().zip(s.remainder()) {
        *o = scalar(x);
    }
}

/// [`logistic_i16_pot`] of the first `min(src.len(), dst.len())` elements.
///
/// # Safety
/// The CPU must support NEON, which every aarch64 CPU does.
#[target_feature(enable = "neon")]
pub(super) unsafe fn logistic_i16_pot_neon(src: &[i16], dst: &mut [i16]) {
    map_vectors::<4>(src, dst, logistic_pot8, logistic_i16_pot);
}

/// Last table index the reference kernel interpolates from; larger ones saturate.
const LAST_INDEX: u32 = 254;

/// `SIGMOID_TABLE_U16[i] | SIGMOID_TABLE_U16[i + 1] << 16`: both interpolation ends in one load.
static TABLE_PAIRS: [u32; LAST_INDEX as usize + 1] = {
    let mut pairs = [0u32; LAST_INDEX as usize + 1];
    let mut i = 0;
    while i < pairs.len() {
        pairs[i] = SIGMOID_TABLE_U16[i] as u32 | (SIGMOID_TABLE_U16[i + 1] as u32) << 16;
        i += 1;
    }
    pairs
};

/// The reference kernel's fixed operands, broadcast once per call.
#[derive(Clone, Copy)]
struct Reference {
    multiplier: int32x4_t,
    round: int32x4_t,
    neg_shift: int32x4_t,
}

impl Reference {
    /// Four lanes of `reference_integer_ops::Logistic` up to the final `>> 10`.
    #[inline(always)]
    fn eval4(self, x: int32x4_t) -> uint32x4_t {
        // SAFETY: NEON is baseline; table indices are clamped to `LAST_INDEX`, inside the table.
        unsafe {
            let input = vshlq_s32(vmlaq_s32(self.round, x, self.multiplier), self.neg_shift);
            let abs = vreinterpretq_u32_s32(vabsq_s32(input));
            let uh = vshrq_n_u32::<9>(abs);
            let saturated = vcgtq_u32(uh, vdupq_n_u32(LAST_INDEX));
            let idx = vminq_u32(uh, vdupq_n_u32(LAST_INDEX));
            let t = TABLE_PAIRS.as_ptr();
            let mut pair = vdupq_n_u32(0);
            pair = vsetq_lane_u32::<0>(*t.add(vgetq_lane_u32::<0>(idx) as usize), pair);
            pair = vsetq_lane_u32::<1>(*t.add(vgetq_lane_u32::<1>(idx) as usize), pair);
            pair = vsetq_lane_u32::<2>(*t.add(vgetq_lane_u32::<2>(idx) as usize), pair);
            pair = vsetq_lane_u32::<3>(*t.add(vgetq_lane_u32::<3>(idx) as usize), pair);
            let ua = vandq_u32(pair, vdupq_n_u32(0xffff));
            let ub = vshrq_n_u32::<16>(pair);
            let ut = vandq_u32(abs, vdupq_n_u32(0x1ff));
            let result = vmlaq_u32(vshlq_n_u32::<9>(ua), ut, vsubq_u32(ub, ua));
            let result = vbslq_u32(saturated, vdupq_n_u32(0x7fff << 10), result);
            let positive = vaddq_u32(result, vdupq_n_u32(1 << 9));
            let negative = vsubq_u32(vdupq_n_u32((1 << 25) + (1 << 9) - 1), result);
            vbslq_u32(vcltzq_s32(input), negative, positive)
        }
    }

    /// Eight lanes, narrowed to int16 as `(result >> 10) as i16`.
    #[inline(always)]
    fn eval8(self, x: int16x8_t) -> int16x8_t {
        // SAFETY: NEON is part of the aarch64 baseline.
        unsafe {
            let lo = self.eval4(vmovl_s16(vget_low_s16(x)));
            let hi = self.eval4(vmovl_high_s16(x));
            vreinterpretq_s16_u16(vcombine_u16(vshrn_n_u32::<10>(lo), vshrn_n_u32::<10>(hi)))
        }
    }
}

/// [`logistic_i16_reference`] of the first `min(src.len(), dst.len())` elements.
///
/// # Safety
/// The CPU must support NEON, which every aarch64 CPU does.
#[target_feature(enable = "neon")]
pub(super) unsafe fn logistic_i16_reference_neon(
    input_multiplier: i32,
    input_left_shift: i32,
    src: &[i16],
    dst: &mut [i16],
) {
    let (multiplier, shift) = if input_multiplier == 0 {
        (3i32.wrapping_shl(input_left_shift.clamp(0, 31) as u32), 0)
    } else {
        (input_multiplier, input_left_shift.clamp(0, 31))
    };
    let round = if shift > 0 { 1i32 << (shift - 1) } else { 0 };
    let k = Reference {
        multiplier: vdupq_n_s32(multiplier),
        round: vdupq_n_s32(round),
        neg_shift: vdupq_n_s32(-shift),
    };
    map_vectors::<2>(
        src,
        dst,
        |x| k.eval8(x),
        |x| logistic_i16_reference(input_multiplier, input_left_shift, x),
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ops::sigmoid::{i16_kernel, I16Kernel, MAX_I16_MULTIPLIER};

    fn all_i16() -> Vec<i16> {
        (i16::MIN..=i16::MAX).collect()
    }

    #[test]
    fn sigmoid_i16_pot_neon_exhaustive() {
        assert_eq!(i16_kernel(2f32.powi(-12)), Some(I16Kernel::Pot));
        let src = all_i16();
        let want: Vec<i16> = src.iter().map(|&x| logistic_i16_pot(x)).collect();
        // Offsets 0..32 move every input through every vector lane and every tail.
        for skip in 0..32 {
            let mut got = vec![0x5555i16; src.len() - skip];
            // SAFETY: NEON is part of the aarch64 baseline.
            unsafe { logistic_i16_pot_neon(&src[skip..], &mut got) };
            assert_eq!(got, want[skip..], "offset {skip}");
        }
    }

    #[test]
    fn sigmoid_i16_reference_neon_exhaustive() {
        // The largest f32 input scale the plan still sends to the reference kernel, from the
        // overflow bound `scale · 3 · 4096 < MAX_I16_MULTIPLIER + 1` downward.
        let mut largest = (f64::from(MAX_I16_MULTIPLIER + 1) / 12288.0) as f32;
        while i16_kernel(largest).is_none() {
            largest = f32::from_bits(largest.to_bits() - 1);
        }
        assert!(largest > 5.33 && i16_kernel(f32::from_bits(largest.to_bits() + 1)).is_none());
        let src = all_i16();
        for scale in [
            1e-5,
            0.7 / 4096.0,
            3.0 / 4096.0,
            0.001,
            0.01,
            0.1,
            1.0,
            largest,
        ] {
            let Some(I16Kernel::Reference {
                multiplier,
                left_shift,
            }) = i16_kernel(scale)
            else {
                panic!("scale {scale} does not take the reference kernel");
            };
            let want: Vec<i16> = src
                .iter()
                .map(|&x| logistic_i16_reference(multiplier, left_shift, x))
                .collect();
            for skip in [0, 3, 11] {
                let mut got = vec![0x5555i16; src.len() - skip];
                // SAFETY: NEON is part of the aarch64 baseline.
                unsafe {
                    logistic_i16_reference_neon(multiplier, left_shift, &src[skip..], &mut got)
                };
                assert_eq!(got, want[skip..], "scale {scale} offset {skip}");
            }
        }
        // The kernel's own power-of-two form, a zero multiplier, matches too.
        let want: Vec<i16> = src
            .iter()
            .map(|&x| logistic_i16_reference(0, 2, x))
            .collect();
        let mut got = vec![0i16; src.len()];
        // SAFETY: as above.
        unsafe { logistic_i16_reference_neon(0, 2, &src, &mut got) };
        assert_eq!(got, want, "zero multiplier");
    }

    #[test]
    fn mismatched_lengths_convert_the_shorter() {
        let src = all_i16();
        let mut got = vec![7i16; 21];
        // SAFETY: NEON is part of the aarch64 baseline.
        unsafe { logistic_i16_pot_neon(&src[..13], &mut got) };
        assert!(got[13..].iter().all(|&v| v == 7));
        let want: Vec<i16> = src[..13].iter().map(|&x| logistic_i16_pot(x)).collect();
        assert_eq!(got[..13], want);
        let mut got = vec![7i16; 5];
        // SAFETY: as above.
        unsafe { logistic_i16_reference_neon(25165, 12, &src[..21], &mut got) };
        let want: Vec<i16> = src[..5]
            .iter()
            .map(|&x| logistic_i16_reference(25165, 12, x))
            .collect();
        assert_eq!(got, want);
    }
}
