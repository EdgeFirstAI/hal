// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! NEON int8 `ADD`, `SUB` and `MUL`.
//!
//! Elements inside TFLite's NEON blocks repeat TFLite's NEON instruction sequence
//! (`MulElementwise`, `AddElementwiseInt8`); the rest vectorize its scalar code with an exact
//! `RoundingDivideByPOT`. Both equal [`Arith::neon_lane`] and [`Arith::reference`] bit for bit.

#![cfg(target_arch = "aarch64")]

use std::arch::aarch64::*; // NOSONAR

use super::binary_i8::{Arith, BinaryOp};

/// The 16 inputs of a step, offset and widened to four i32 vectors.
#[inline]
#[target_feature(enable = "neon")]
fn widen(v: int8x16_t, offset: int16x8_t) -> [int32x4_t; 4] {
    let lo = vaddq_s16(vmovl_s8(vget_low_s8(v)), offset);
    let hi = vaddq_s16(vmovl_s8(vget_high_s8(v)), offset);
    [
        vmovl_s16(vget_low_s16(lo)),
        vmovl_s16(vget_high_s16(lo)),
        vmovl_s16(vget_low_s16(hi)),
        vmovl_s16(vget_high_s16(hi)),
    ]
}

/// gemmlowp's NEON `RoundingDivideByPOT(int32x4_t, int)`.
#[inline]
#[target_feature(enable = "neon")]
fn rdbpot_gemmlowp(x: int32x4_t, exponent: i32) -> int32x4_t {
    let shift = vdupq_n_s32(-exponent);
    let fixup = vshrq_n_s32::<31>(vandq_s32(x, shift));
    vrshlq_s32(vqaddq_s32(x, fixup), shift)
}

/// gemmlowp's scalar `RoundingDivideByPOT`, lane by lane: `(x >> e) + (remainder > threshold)`.
#[inline]
#[target_feature(enable = "neon")]
pub(super) fn rdbpot_exact(x: int32x4_t, exponent: i32) -> int32x4_t {
    let mask = vdupq_n_s32(((1i64 << exponent.clamp(0, 31)) - 1) as i32);
    let remainder = vandq_s32(x, mask);
    // `x < 0` is all ones; adding its negation adds one to the threshold.
    let negative = vreinterpretq_s32_u32(vcltzq_s32(x));
    let threshold = vsubq_s32(vshrq_n_s32::<1>(mask), negative);
    let shifted = vshlq_s32(x, vdupq_n_s32(-exponent.clamp(0, 31)));
    let round_up = vreinterpretq_s32_u32(vcgtq_s32(remainder, threshold));
    vsubq_s32(shifted, round_up)
}

/// `MultiplyByQuantizedMultiplier` on four lanes, exactly as the scalar code.
#[inline]
#[target_feature(enable = "neon")]
pub(super) fn mbqm_exact(x: int32x4_t, mult: i32, shift: i32) -> int32x4_t {
    let x = vshlq_s32(x, vdupq_n_s32(shift.clamp(0, 31)));
    rdbpot_exact(vqrdmulhq_n_s32(x, mult), (-shift).clamp(0, 31))
}

/// Narrows four i32 vectors of values already clamped to the i8 range.
#[inline]
#[target_feature(enable = "neon")]
fn narrow_clamped(v: [int32x4_t; 4]) -> int8x16_t {
    let lo = vcombine_s16(vmovn_s32(v[0]), vmovn_s32(v[1]));
    let hi = vcombine_s16(vmovn_s32(v[2]), vmovn_s32(v[3]));
    vcombine_s8(vmovn_s16(lo), vmovn_s16(hi))
}

/// TFLite's scalar formula on 16 lanes.
#[inline]
#[target_feature(enable = "neon")]
fn reference16(a: &Arith, x: int8x16_t, y: int8x16_t) -> int8x16_t {
    let xs = widen(x, vdupq_n_s16(a.lhs_offset as i16));
    let ys = widen(y, vdupq_n_s16(a.rhs_offset as i16));
    let (lo, hi) = (vdupq_n_s32(a.act_min), vdupq_n_s32(a.act_max));
    let off = vdupq_n_s32(a.out_offset);
    let out = core::array::from_fn(|k| {
        let raw = match a.op {
            BinaryOp::Mul => mbqm_exact(vmulq_s32(xs[k], ys[k]), a.out_mult, a.out_shift),
            BinaryOp::Add | BinaryOp::Sub => {
                let xv = mbqm_exact(vshlq_n_s32::<20>(xs[k]), a.lhs_mult, a.lhs_shift);
                let yv = mbqm_exact(vshlq_n_s32::<20>(ys[k]), a.rhs_mult, a.rhs_shift);
                let s = if a.op == BinaryOp::Add {
                    vaddq_s32(xv, yv)
                } else {
                    vsubq_s32(xv, yv)
                };
                mbqm_exact(s, a.out_mult, a.out_shift)
            }
        };
        vminq_s32(vmaxq_s32(vaddq_s32(raw, off), lo), hi)
    });
    narrow_clamped(out)
}

/// TFLite's NEON loop body on 16 lanes (`MUL` and `ADD`).
#[inline]
#[target_feature(enable = "neon")]
fn neon16(a: &Arith, x: int8x16_t, y: int8x16_t) -> int8x16_t {
    let xs = widen(x, vdupq_n_s16(a.lhs_offset as i16));
    let ys = widen(y, vdupq_n_s16(a.rhs_offset as i16));
    let n: [int16x4_t; 4] = core::array::from_fn(|k| match a.op {
        BinaryOp::Mul => {
            let p = vshlq_s32(vmulq_s32(xs[k], ys[k]), vdupq_n_s32(a.out_shift.max(0)));
            let p = vqrdmulhq_n_s32(p, a.out_mult);
            vqmovn_s32(rdbpot_gemmlowp(p, (-a.out_shift).max(0)))
        }
        _ => {
            let xv = vshlq_s32(xs[k], vdupq_n_s32(20 + a.lhs_shift));
            let yv = vshlq_s32(ys[k], vdupq_n_s32(20 + a.rhs_shift));
            let s = vaddq_s32(
                vqrdmulhq_n_s32(xv, a.lhs_mult),
                vqrdmulhq_n_s32(yv, a.rhs_mult),
            );
            let s = vqrdmulhq_n_s32(s, a.out_mult);
            vmovn_s32(rdbpot_gemmlowp(s, -a.out_shift))
        }
    });
    let off = vdupq_n_s16(a.out_offset as i16);
    let lo = vaddq_s16(vcombine_s16(n[0], n[1]), off);
    let hi = vaddq_s16(vcombine_s16(n[2], n[3]), off);
    let q = vcombine_s8(vqmovn_s16(lo), vqmovn_s16(hi));
    vmaxq_s8(
        vdupq_n_s8(a.act_min as i8),
        vminq_s8(vdupq_n_s8(a.act_max as i8), q),
    )
}

/// Elements `start..start + out.len()` of a tensor whose scalar tail starts at flat index
/// `tail`: those before `tail` take the NEON build's arithmetic and the rest the reference
/// arithmetic. Both are lane-wise, so `start` may be any index.
///
/// # Safety
/// The CPU must support NEON, which every aarch64 CPU does.
#[target_feature(enable = "neon")]
pub(super) unsafe fn run(a: &Arith, start: usize, tail: usize, x: &[i8], y: &[i8], out: &mut [i8]) {
    let n = out.len().min(x.len()).min(y.len());
    let (Some(x), Some(y), Some(out)) = (x.get(..n), y.get(..n), out.get_mut(..n)) else {
        return;
    };
    let split = tail.saturating_sub(start).min(n);
    let neon = a.op != BinaryOp::Sub;
    let (xn, xr) = x.split_at(split);
    let (yn, yr) = y.split_at(split);
    let (on, or) = out.split_at_mut(split);
    span(a, neon, xn, yn, on);
    span(a, false, xr, yr, or);
}

/// One span of [`run`]: 16 elements at a time, then one at a time, all with the NEON build's
/// arithmetic when `neon` and with the reference arithmetic otherwise.
#[target_feature(enable = "neon")]
fn span(a: &Arith, neon: bool, x: &[i8], y: &[i8], out: &mut [i8]) {
    let (xq, xt) = x.as_chunks::<16>();
    let (yq, yt) = y.as_chunks::<16>();
    let (oq, ot) = out.as_chunks_mut::<16>();
    for ((xb, yb), ob) in xq.iter().zip(yq).zip(oq.iter_mut()) {
        // SAFETY: each chunk is 16 bytes.
        unsafe {
            let (xv, yv) = (vld1q_s8(xb.as_ptr()), vld1q_s8(yb.as_ptr()));
            let q = if neon {
                neon16(a, xv, yv)
            } else {
                reference16(a, xv, yv)
            };
            vst1q_s8(ob.as_mut_ptr(), q);
        }
    }
    for ((o, &xv), &yv) in ot.iter_mut().zip(xt).zip(yt) {
        *o = if neon {
            a.neon_lane(xv, yv)
        } else {
            a.reference(xv, yv)
        };
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ops::binary_i8::BinaryI8Params;
    use crate::quant::{Activation, Quant, TfliteKernels};

    /// Every (x, y) pair through the NEON kernel and the portable kernel, as block and tail.
    fn check(p: BinaryI8Params) {
        let a = Arith::new(&p).unwrap();
        let xs: Vec<i8> = (0..65536u32).map(|i| (i >> 8) as u8 as i8).collect();
        let ys: Vec<i8> = (0..65536u32).map(|i| i as u8 as i8).collect();
        for tail in [0usize, 65536, 65536 - 4096] {
            let mut neon = vec![0i8; 65536];
            let mut portable = vec![0i8; 65536];
            // SAFETY: NEON is part of the aarch64 baseline.
            unsafe { run(&a, 0, tail, &xs, &ys, &mut neon) };
            a.run_portable(0, tail, &xs, &ys, &mut portable);
            assert_eq!(neon, portable, "{p:?} tail {tail}");
            // Runs that start anywhere, as row segments do.
            let mut pieces = vec![0i8; 65536];
            let (mut s, mut k) = (0usize, 0usize);
            while s < 65536 {
                let len = [7usize, 9, 1000, 4103, 13][k % 5].min(65536 - s);
                let r = s..s + len;
                // SAFETY: NEON is part of the aarch64 baseline.
                unsafe { run(&a, s, tail, &xs[r.clone()], &ys[r.clone()], &mut pieces[r]) };
                s += len;
                k += 1;
            }
            assert_eq!(pieces, portable, "{p:?} tail {tail} in pieces");
        }
    }

    #[test]
    fn neon_matches_portable_on_every_pair() {
        let quants = [
            (
                Quant::new(0.0579, -128),
                Quant::new(0.000_196, -128),
                Quant::new(0.002_894, -128),
            ),
            (
                Quant::new(0.002_894, -128),
                Quant::new(0.003_897, -128),
                Quant::new(0.004_491, -128),
            ),
            (
                Quant::new(0.1251, -15),
                Quant::new(0.000_693, -128),
                Quant::new(0.010_18, 6),
            ),
            (
                Quant::new(0.5, 3),
                Quant::new(0.25, -7),
                Quant::new(0.01, 0),
            ),
            (
                Quant::new(0.01, 0),
                Quant::new(0.01, 0),
                Quant::new(0.000_01, 0),
            ),
        ];
        for op in [BinaryOp::Add, BinaryOp::Sub, BinaryOp::Mul] {
            for (lhs, rhs, output) in quants {
                for activation in [Activation::None, Activation::Relu6] {
                    for kernels in [TfliteKernels::NeonRuy, TfliteKernels::Reference] {
                        let p = BinaryI8Params {
                            op,
                            lhs,
                            rhs,
                            output,
                            activation,
                            kernels,
                        };
                        if Arith::new(&p).is_ok() {
                            check(p);
                        }
                    }
                }
            }
        }
    }
}
