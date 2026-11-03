// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! FP16-arithmetic NEON kernels (ARMv8.2 FEAT_FP16), eight half-precision lanes per vector.
//!
//! Stable Rust has no f16 intrinsics, so the arithmetic is inline asm with `.arch_extension
//! fp16` on `uint16x8_t` carriers. Every kernel here needs `CpuFeatures::neon_fp16` and
//! `is_aarch64_feature_detected!("fp16")`; CPUs without FEAT_FP16 fault on these instructions.
//!
//! The sigmoid kernels compute `exp(−|x|)` as `2^k · p(r)` with `r = −|x| − k·ln2`, a degree-4
//! polynomial `p`, then `s = e / (1 + e)` in f16 and `1 − s` for `x ≥ 0`. Integer and f32 inputs are
//! dequantized and range-reduced in f32 first, because rounding `x` itself to f16 costs up to
//! `|x|` ulps in the negative tail; f16 inputs are reduced in f16, where `x` is exact. The tests
//! hold every result within 2 f16 ulps of the float64 sigmoid rounded to f16, over every 8- and
//! 16-bit value and every f16 at fixed quantizations; `tests/sigmoid_lut.rs` adds a seeded sweep
//! of random quantizations for the integer inputs.
//!
//! The decode kernels compute DFL distances and boxes into f16. A side's softmax weights are
//! `exp(x − max)`, the same `2^k · p(r)` reduced in f16 from the f32 difference; eight bins per
//! vector are summed with their bin-weighted sum in two alternating f16 sets, and each side's
//! lanes are reduced and divided in f32, which keeps an f16 reduction tree's rounding out of the
//! distances.
//! dist2bbox runs eight anchors per vector in f16, on the f16-rounded grid and stride.
//! `scale_bias` converts integers to f16 (`SCVTF`/`UCVTF`) and multiplies and adds in f16, each
//! step correctly rounded, so it equals `half` arithmetic bit for bit.

#![cfg(target_arch = "aarch64")]

use std::arch::aarch64::*; // NOSONAR
use std::mem::MaybeUninit;

use half::f16;

use super::fp16_asm::{
    fadd_f16x8, fcvtns_s16_from_f16x8, fdiv_f16x8, fmax_f16x8, fmla_f16x8, fmls_f16x8, fmul_f16x8,
    fsub_f16x8, scvtf_f16_from_s16x8, ucvtf_f16_from_u16x8,
};
use super::neon_convert::{affine, map_steps, map_steps_uninit, narrow8, widen8, LoadF32};
use crate::Quant;

/// Binary16 bit patterns of the sigmoid constants.
mod bits {
    pub(super) const ONE: u16 = 0x3c00;
    pub(super) const HALF: u16 = 0x3800;
    /// 1/6 and 1/24, the polynomial's cubic and quartic coefficients.
    pub(super) const ONE_SIXTH: u16 = 0x3155;
    pub(super) const ONE_TWENTY_FOURTH: u16 = 0x2955;
    pub(super) const LOG2_E: u16 = 0x3dc5;
    /// ln 2 = `LN2_HI + LN2_LO`, `LN2_HI` = 0.693359375 exactly.
    pub(super) const LN2_HI: u16 = 0x398c;
    pub(super) const LN2_LO: u16 = 0x8af4;
    /// −17.5: below it the sigmoid rounds to zero in f16.
    pub(super) const T_MIN: u16 = 0xcc60;
    pub(super) const EIGHT: u16 = 0x4800;
    /// 0, 1, …, 7: the bin offsets of an eight-bin vector.
    pub(super) const IOTA: [u16; 8] = [
        0x0000, 0x3c00, 0x4000, 0x4200, 0x4400, 0x4500, 0x4600, 0x4700,
    ];
}

/// The exp argument's lower clamp in f32, the same −17.5.
const T_MIN_F32: f32 = -17.5;

/// ln 2 = `LN2_HI_F32 + LN2_LO_F32` for the f32 range reduction; the high part is exact.
#[allow(clippy::excessive_precision)]
const LN2_HI_F32: f32 = 0.693_359_375;
const LN2_LO_F32: f32 = -2.121_944_4e-4;

/// `2^k` for `k` in [−14, 15] by exponent injection.
#[inline(always)]
fn pow2_f16(k: int16x8_t) -> uint16x8_t {
    // SAFETY: NEON is part of the aarch64 baseline.
    unsafe { vreinterpretq_u16_s16(vshlq_n_s16::<10>(vaddq_s16(k, vdupq_n_s16(15)))) }
}

/// `2^k · exp(r)` for `k` in [−25, 0] and `|r|` about ln2/2 or less.
///
/// `2^k` is applied as `2^max(k, −13)` then `2^(k − max(k, −13))`: the first product stays
/// normal and exact, so a subnormal result is rounded once.
///
/// # Safety
/// The CPU must support FEAT_FP16.
#[inline(always)]
unsafe fn exp_reduced(r: uint16x8_t, k: int16x8_t) -> uint16x8_t {
    // SAFETY: the caller guarantees FEAT_FP16.
    unsafe {
        let one = vdupq_n_u16(bits::ONE);
        let q = fmla_f16x8(
            vdupq_n_u16(bits::ONE_SIXTH),
            vdupq_n_u16(bits::ONE_TWENTY_FOURTH),
            r,
        );
        let q = fmla_f16x8(vdupq_n_u16(bits::HALF), q, r);
        let r2 = fmul_f16x8(r, r);
        let p = fadd_f16x8(fmla_f16x8(r, q, r2), one);
        let k_hi = vmaxq_s16(k, vdupq_n_s16(-13));
        let k_lo = vsubq_s16(k, k_hi);
        fmul_f16x8(fmul_f16x8(p, pow2_f16(k_hi)), pow2_f16(k_lo))
    }
}

/// `s = e / (1 + e)` where `negative` is set, `1 − s` elsewhere; `e = exp(−|x|)`.
///
/// `s` is small and accurate, so near 1 the only rounding is that of `1 − s`, which reaches
/// every f16 below 1; `1 / (1 + e)` would round `1 + e` first and skip them.
///
/// # Safety
/// The CPU must support FEAT_FP16.
#[inline(always)]
unsafe fn sigmoid_from_exp(e: uint16x8_t, negative: uint16x8_t) -> uint16x8_t {
    // SAFETY: the caller guarantees FEAT_FP16.
    unsafe {
        let one = vdupq_n_u16(bits::ONE);
        let s = fdiv_f16x8(e, fadd_f16x8(one, e));
        vbslq_u16(negative, s, fsub_f16x8(one, s))
    }
}

/// Sigmoid of 8 f16 lanes, reduced in f16.
///
/// # Safety
/// The CPU must support FEAT_FP16.
#[inline(always)]
unsafe fn sigmoid_f16x8(x: uint16x8_t) -> uint16x8_t {
    // SAFETY: the caller guarantees FEAT_FP16.
    unsafe {
        let negative = vcltzq_s16(vreinterpretq_s16_u16(x));
        let t = fmax_f16x8(vorrq_u16(x, vdupq_n_u16(0x8000)), vdupq_n_u16(bits::T_MIN));
        let k = fcvtns_s16_from_f16x8(fmul_f16x8(t, vdupq_n_u16(bits::LOG2_E)));
        let kf = scvtf_f16_from_s16x8(k);
        let z = fmls_f16x8(t, kf, vdupq_n_u16(bits::LN2_HI));
        let r = fmls_f16x8(z, kf, vdupq_n_u16(bits::LN2_LO));
        sigmoid_from_exp(exp_reduced(r, k), negative)
    }
}

/// `−|x|` reduced in f32: `(r, k, x < 0)` with `−|x| = k·ln2 + r`.
#[inline(always)]
fn reduce_f32(x: float32x4_t) -> (float32x4_t, int32x4_t, uint32x4_t) {
    // SAFETY: NEON is part of the aarch64 baseline.
    unsafe {
        let t = vmaxq_f32(vnegq_f32(vabsq_f32(x)), vdupq_n_f32(T_MIN_F32));
        let k = vcvtnq_s32_f32(vmulq_f32(t, vdupq_n_f32(core::f32::consts::LOG2_E)));
        let kf = vcvtq_f32_s32(k);
        let z = vfmsq_f32(t, kf, vdupq_n_f32(LN2_HI_F32));
        let r = vfmsq_f32(z, kf, vdupq_n_f32(LN2_LO_F32));
        (r, k, vcltzq_f32(x))
    }
}

/// Sigmoid of 8 f32 lanes into f16: f32 range reduction, the rest in f16.
///
/// # Safety
/// The CPU must support FEAT_FP16.
#[inline(always)]
unsafe fn sigmoid_f32x8(lo: float32x4_t, hi: float32x4_t) -> uint16x8_t {
    let ((r0, k0, n0), (r1, k1, n1)) = (reduce_f32(lo), reduce_f32(hi));
    // SAFETY: the caller guarantees FEAT_FP16.
    unsafe {
        let k = vcombine_s16(vmovn_s32(k0), vmovn_s32(k1));
        let negative = vcombine_u16(vmovn_u32(n0), vmovn_u32(n1));
        sigmoid_from_exp(exp_reduced(narrow8(r0, r1), k), negative)
    }
}

/// Dequantizes 16 elements in f32 (as `prim::dequant` does) and stores their sigmoid as f16.
///
/// # Safety
/// The CPU must support FEAT_FP16; `s` must be readable and `d` writable for 16 elements.
#[inline(always)]
unsafe fn dequant_sigmoid_step<S: LoadF32>(
    s: *const S,
    d: *mut f16,
    q: Option<(float32x4_t, float32x4_t)>,
) {
    // SAFETY: the caller guarantees FEAT_FP16 and 16 valid elements at each pointer.
    unsafe {
        let v = S::load16(s);
        let x = match q {
            Some((zp, scale)) => affine(v, zp, scale),
            None => v,
        };
        let d = d.cast::<u16>();
        vst1q_u16(d, sigmoid_f32x8(x[0], x[1]));
        vst1q_u16(d.add(8), sigmoid_f32x8(x[2], x[3]));
    }
}

/// `sigmoid((x − zero_point) · scale)` into f16, dequantizing in f32; `Quant::IDENTITY` skips it.
///
/// # Safety
/// The CPU must support FEAT_FP16.
#[inline(always)]
unsafe fn dequant_sigmoid<S: LoadF32>(src: &[S], q: Quant, dst: &mut [f16]) {
    // SAFETY: NEON is part of the aarch64 baseline.
    let q = (q != Quant::IDENTITY)
        .then(|| unsafe { (vdupq_n_f32(q.zero_point as f32), vdupq_n_f32(q.scale)) });
    // SAFETY: the caller guarantees FEAT_FP16; `map_steps` hands out 16 valid elements.
    map_steps(src, dst, |s, d| unsafe { dequant_sigmoid_step(s, d, q) });
}

macro_rules! dequant_sigmoid_kernel {
    ($(#[$doc:meta])* $name:ident, $t:ty) => {
        $(#[$doc])*
        ///
        /// Tested within 2 f16 ulps of the float64 sigmoid rounded to f16; NaN stays NaN. Writes
        /// nothing when the slices differ in length.
        ///
        /// # Safety
        /// The CPU must support FEAT_FP16 (`CpuFeatures::neon_fp16`).
        #[inline]
        #[target_feature(enable = "neon")]
        pub unsafe fn $name(src: &[$t], q: Quant, dst: &mut [f16]) {
            // SAFETY: the caller guarantees FEAT_FP16.
            unsafe { dequant_sigmoid(src, q, dst) }
        }
    };
}

dequant_sigmoid_kernel!(
    /// `sigmoid((x − zero_point) · scale)` of i8 into f16.
    dequant_sigmoid_i8_to_f16_neon_fp16,
    i8
);
dequant_sigmoid_kernel!(
    /// `sigmoid((x − zero_point) · scale)` of u8 into f16.
    dequant_sigmoid_u8_to_f16_neon_fp16,
    u8
);
dequant_sigmoid_kernel!(
    /// `sigmoid((x − zero_point) · scale)` of i16 into f16.
    dequant_sigmoid_i16_to_f16_neon_fp16,
    i16
);
dequant_sigmoid_kernel!(
    /// `sigmoid((x − zero_point) · scale)` of u16 into f16.
    dequant_sigmoid_u16_to_f16_neon_fp16,
    u16
);
dequant_sigmoid_kernel!(
    /// `sigmoid((x − zero_point) · scale)` of f32 into f16; `Quant::IDENTITY` is a plain sigmoid.
    dequant_sigmoid_f32_to_f16_neon_fp16,
    f32
);

/// `sigmoid((x − zero_point) · scale)` of f16 into f16; `Quant::IDENTITY` runs entirely in f16.
///
/// Tested within 2 f16 ulps of the float64 sigmoid rounded to f16; NaN stays NaN. Writes nothing
/// when the slices differ in length.
///
/// # Safety
/// The CPU must support FEAT_FP16 (`CpuFeatures::neon_fp16`).
#[inline]
#[target_feature(enable = "neon")]
pub unsafe fn dequant_sigmoid_f16_to_f16_neon_fp16(src: &[f16], q: Quant, dst: &mut [f16]) {
    if q != Quant::IDENTITY {
        // SAFETY: the caller guarantees FEAT_FP16.
        unsafe { dequant_sigmoid(src, q, dst) };
        return;
    }
    // SAFETY: FEAT_FP16 by contract; 16 valid elements per step; `f16` is a transparent `u16`.
    map_steps(src, dst, |s, d| unsafe {
        let (s, d) = (s.cast::<u16>(), d.cast::<u16>());
        vst1q_u16(d, sigmoid_f16x8(vld1q_u16(s)));
        vst1q_u16(d.add(8), sigmoid_f16x8(vld1q_u16(s.add(8))));
    });
}

/// `exp(t)` of 8 f16 lanes with `t ≤ 0`; below −17.5 the result rounds to zero.
///
/// # Safety
/// The CPU must support FEAT_FP16.
#[inline(always)]
unsafe fn exp_nonpositive_f16x8(t: uint16x8_t) -> uint16x8_t {
    // SAFETY: the caller guarantees FEAT_FP16.
    unsafe {
        let t = fmax_f16x8(t, vdupq_n_u16(bits::T_MIN));
        let k = fcvtns_s16_from_f16x8(fmul_f16x8(t, vdupq_n_u16(bits::LOG2_E)));
        let kf = scvtf_f16_from_s16x8(k);
        let z = fmls_f16x8(t, kf, vdupq_n_u16(bits::LN2_HI));
        let r = fmls_f16x8(z, kf, vdupq_n_u16(bits::LN2_LO));
        exp_reduced(r, k)
    }
}

/// Eight-bin f16 accumulators of one side: the weight sums and bin-weighted sums, in two sets
/// that alternate by step so no lane adds more than every other step in f16.
#[derive(Clone, Copy)]
struct SideSums {
    sum: [uint16x8_t; 2],
    ws: [uint16x8_t; 2],
    /// The bin index of each lane of the next eight-bin step.
    bin: uint16x8_t,
}

impl SideSums {
    /// # Safety
    /// The CPU must support FEAT_FP16.
    #[inline(always)]
    unsafe fn new() -> Self {
        // SAFETY: NEON is part of the aarch64 baseline.
        unsafe {
            let zero = vdupq_n_u16(0);
            Self {
                sum: [zero; 2],
                ws: [zero; 2],
                bin: vld1q_u16(bits::IOTA.as_ptr()),
            }
        }
    }

    /// Adds the weights `e` of the next eight bins to set `SET`.
    ///
    /// # Safety
    /// The CPU must support FEAT_FP16.
    #[inline(always)]
    unsafe fn add<const SET: usize>(&mut self, e: uint16x8_t) {
        // SAFETY: the caller guarantees FEAT_FP16.
        unsafe {
            if let (Some(sum), Some(ws)) = (self.sum.get_mut(SET), self.ws.get_mut(SET)) {
                *sum = fadd_f16x8(*sum, e);
                *ws = fmla_f16x8(*ws, e, self.bin);
            }
            self.bin = fadd_f16x8(self.bin, vdupq_n_u16(bits::EIGHT));
        }
    }
}

/// `Σ ws / Σ sum` of the four sides, rounded to f16: lane `s` is side `s`'s distance. The f16
/// lanes of both sets are widened and reduced in f32, which keeps the rounding of an f16
/// reduction tree out of the distances.
///
/// # Safety
/// The CPU must support FEAT_FP16.
#[inline(always)]
unsafe fn side_distances(s: [SideSums; 4]) -> uint16x4_t {
    let fold = |v: [uint16x8_t; 2]| {
        let ([a, b], [c, d]) = (widen8(v[0]), widen8(v[1]));
        // SAFETY: NEON is part of the aarch64 baseline.
        unsafe { vaddq_f32(vaddq_f32(a, b), vaddq_f32(c, d)) }
    };
    // SAFETY: NEON is part of the aarch64 baseline.
    unsafe {
        let [a, b, c, d] = s;
        let sum = vpaddq_f32(
            vpaddq_f32(fold(a.sum), fold(b.sum)),
            vpaddq_f32(fold(c.sum), fold(d.sum)),
        );
        let ws = vpaddq_f32(
            vpaddq_f32(fold(a.ws), fold(b.ws)),
            vpaddq_f32(fold(c.ws), fold(d.ws)),
        );
        let q = vdivq_f32(ws, sum);
        vget_low_u16(narrow8(q, q))
    }
}

/// Checks that `dst` holds four values per anchor of `input`'s `4 · reg_max` bins.
#[inline(always)]
fn ltrb_lengths<S, D>(input: &[S], reg_max: usize, dst: &[D]) -> bool {
    reg_max != 0 && input.len().is_multiple_of(4 * reg_max) && dst.len() == input.len() / reg_max
}

/// The largest value of `side`, by NEON lane maxima over 4-value steps.
#[inline(always)]
fn max_f32(side: &[f32]) -> f32 {
    let mut steps = side.chunks_exact(4);
    let mut m = f32::NEG_INFINITY;
    if let Some(first) = steps.next() {
        // SAFETY: NEON is part of the aarch64 baseline; each step holds 4 floats.
        unsafe {
            let mut v = vld1q_f32(first.as_ptr());
            for s in &mut steps {
                v = vmaxq_f32(v, vld1q_f32(s.as_ptr()));
            }
            m = vmaxvq_f32(v);
        }
    }
    steps.remainder().iter().fold(m, |m, &v| m.max(v))
}

/// The f16 sums of one anchor's four sides of f32 logits, weighed by `exp(x − max)`; the
/// sides advance together so an in-order core overlaps their exps.
///
/// # Safety
/// The CPU must support FEAT_FP16.
#[inline(always)]
unsafe fn softmax_sides(sides: [&[f32]; 4]) -> [SideSums; 4] {
    let [a, b, c, d] = sides;
    // SAFETY: FEAT_FP16 by contract; every step reads eight floats, a rest from a padded copy.
    unsafe {
        let m = [
            vdupq_n_f32(max_f32(a)),
            vdupq_n_f32(max_f32(b)),
            vdupq_n_f32(max_f32(c)),
            vdupq_n_f32(max_f32(d)),
        ];
        let diff = |p: *const f32, m: float32x4_t| {
            narrow8(
                vsubq_f32(vld1q_f32(p), m),
                vsubq_f32(vld1q_f32(p.add(4)), m),
            )
        };
        let mut acc = [SideSums::new(); 4];
        let exps = |a: &[f32], b: &[f32], c: &[f32], d: &[f32]| {
            [
                exp_nonpositive_f16x8(diff(a.as_ptr(), m[0])),
                exp_nonpositive_f16x8(diff(b.as_ptr(), m[1])),
                exp_nonpositive_f16x8(diff(c.as_ptr(), m[2])),
                exp_nonpositive_f16x8(diff(d.as_ptr(), m[3])),
            ]
        };
        let pairs = a
            .chunks_exact(16)
            .zip(b.chunks_exact(16))
            .zip(c.chunks_exact(16).zip(d.chunks_exact(16)));
        for ((a, b), (c, d)) in pairs {
            let (a, b, c, d) = (a.split_at(8), b.split_at(8), c.split_at(8), d.split_at(8));
            let lo = exps(a.0, b.0, c.0, d.0);
            let hi = exps(a.1, b.1, c.1, d.1);
            for ((acc, lo), hi) in acc.iter_mut().zip(lo).zip(hi) {
                acc.add::<0>(lo);
                acc.add::<1>(hi);
            }
        }
        // At most one full eight-bin step remains after the pairs, then a partial one.
        fn tail(v: &[f32]) -> &[f32] {
            v.get(v.len() / 16 * 16..).unwrap_or(&[])
        }
        if let (Some(a), Some(b), Some(c), Some(d)) = (
            tail(a).get(..8),
            tail(b).get(..8),
            tail(c).get(..8),
            tail(d).get(..8),
        ) {
            for (acc, e) in acc.iter_mut().zip(exps(a, b, c, d)) {
                acc.add::<0>(e);
            }
        }
        for ((acc, side), m) in acc.iter_mut().zip(sides).zip(m) {
            let rest = side.get(side.len() / 8 * 8..).unwrap_or(&[]);
            if rest.is_empty() {
                continue;
            }
            // Padding with the max keeps every lane at t = 0, inside the exp's range.
            let mut x = [vgetq_lane_f32::<0>(m); 8];
            for (d, &v) in x.iter_mut().zip(rest) {
                *d = v;
            }
            let live = vcltq_u16(
                vld1q_u16([0u16, 1, 2, 3, 4, 5, 6, 7].as_ptr()),
                vdupq_n_u16(rest.len() as u16),
            );
            acc.add::<1>(vandq_u16(exp_nonpositive_f16x8(diff(x.as_ptr(), m)), live));
        }
        acc
    }
}

/// DFL `[l, t, r, b]` in f16 of each anchor's `4 · reg_max` f32 logits: `exp(x − max)` with
/// the difference taken in f32 and rounded to f16, the exp and eight-bin sums in f16, then
/// reduced across lanes and divided in f32. Writes nothing unless `dst` holds four values per
/// anchor of `logits`.
///
/// Tested against the float64 softmax within the bound `tests/decode_ops.rs` states.
///
/// # Safety
/// The CPU must support FEAT_FP16 (`CpuFeatures::neon_fp16`).
#[inline]
#[target_feature(enable = "neon")]
pub unsafe fn dfl_softmax_ltrb_f16_neon_fp16(logits: &[f32], reg_max: usize, dst: &mut [f16]) {
    if !ltrb_lengths(logits, reg_max, dst) {
        return;
    }
    for (anchor, out) in logits
        .chunks_exact(4 * reg_max)
        .zip(dst.chunks_exact_mut(4))
    {
        let Some(sides) = super::dfl_lut::sides(anchor, reg_max) else {
            return;
        };
        // SAFETY: the caller guarantees FEAT_FP16; `out` holds four f16.
        unsafe {
            let sums = softmax_sides(sides);
            vst1_u16(out.as_mut_ptr().cast::<u16>(), side_distances(sums));
        }
    }
}

/// Eight anchors' `[l, t, r, b]` as f16 lane vectors, read from 32 elements: f16 as is, f32
/// rounded to f16.
pub(crate) trait Ltrb8: Copy + Default + Send + Sync {
    /// # Safety
    /// `p` must be readable for 32 elements.
    unsafe fn load(p: *const Self) -> [uint16x8_t; 4];
}

impl Ltrb8 for f16 {
    #[inline(always)]
    unsafe fn load(p: *const Self) -> [uint16x8_t; 4] {
        // SAFETY: the caller guarantees 32 readable f16; `f16` is a transparent `u16`.
        let v = unsafe { vld4q_u16(p.cast::<u16>()) };
        [v.0, v.1, v.2, v.3]
    }
}

impl Ltrb8 for f32 {
    #[inline(always)]
    unsafe fn load(p: *const Self) -> [uint16x8_t; 4] {
        // SAFETY: the caller guarantees 32 readable f32.
        let (a, b) = unsafe { (vld4q_f32(p), vld4q_f32(p.add(16))) };
        [
            narrow8(a.0, b.0),
            narrow8(a.1, b.1),
            narrow8(a.2, b.2),
            narrow8(a.3, b.3),
        ]
    }
}

/// Boxes of eight anchors in f16: `[xc, yc, w, h]` in the Ultralytics order, or
/// `[(gx − l)·s, (gy − t)·s, (gx + r)·s, (gy + b)·s]` when `xyxy`.
///
/// # Safety
/// The CPU must support FEAT_FP16.
#[inline(always)]
unsafe fn boxes8(
    [l, t, r, b]: [uint16x8_t; 4],
    gx: uint16x8_t,
    gy: uint16x8_t,
    s: uint16x8_t,
    xyxy: bool,
) -> uint16x8x4_t {
    // SAFETY: the caller guarantees FEAT_FP16.
    unsafe {
        if xyxy {
            return uint16x8x4_t(
                fmul_f16x8(fsub_f16x8(gx, l), s),
                fmul_f16x8(fsub_f16x8(gy, t), s),
                fmul_f16x8(fadd_f16x8(gx, r), s),
                fmul_f16x8(fadd_f16x8(gy, b), s),
            );
        }
        let half = vdupq_n_u16(bits::HALF);
        uint16x8x4_t(
            fmul_f16x8(fmla_f16x8(gx, fsub_f16x8(r, l), half), s),
            fmul_f16x8(fmla_f16x8(gy, fsub_f16x8(b, t), half), s),
            fmul_f16x8(fadd_f16x8(l, r), s),
            fmul_f16x8(fadd_f16x8(t, b), s),
        )
    }
}

/// dist2bbox of `ltrb` (four per anchor) into f16 boxes, eight anchors per step, as
/// [`dist2bbox_f16_neon_fp16`] does.
///
/// # Safety
/// The CPU must support FEAT_FP16.
#[inline(always)]
pub(crate) unsafe fn dist2bbox_f16<L: Ltrb8>(
    ltrb: &[L],
    gx: &[f32],
    gy: &[f32],
    stride: f32,
    xyxy: bool,
    dst: &mut [f16],
) {
    let n = dst.len() / 4;
    if ltrb.len() != dst.len() || !dst.len().is_multiple_of(4) || gx.len() < n || gy.len() < n {
        return;
    }
    // SAFETY: FEAT_FP16 by contract; every step stays inside the slices or the padded copies.
    unsafe {
        let s = vdupq_n_u16(f16::from_f32(stride).to_bits());
        let step = |l: *const L, x: *const f32, y: *const f32, d: *mut f16| {
            let gx = narrow8(vld1q_f32(x), vld1q_f32(x.add(4)));
            let gy = narrow8(vld1q_f32(y), vld1q_f32(y.add(4)));
            vst4q_u16(d.cast::<u16>(), boxes8(L::load(l), gx, gy, s, xyxy));
        };
        let mut src = ltrb.chunks_exact(32);
        let mut out = dst.chunks_exact_mut(32);
        let grids = gx.chunks_exact(8).zip(gy.chunks_exact(8));
        for ((l, d), (x, y)) in (&mut src).zip(&mut out).zip(grids) {
            step(l.as_ptr(), x.as_ptr(), y.as_ptr(), d.as_mut_ptr());
        }
        let (src, out) = (src.remainder(), out.into_remainder());
        if out.is_empty() {
            return;
        }
        let first = n - out.len() / 4;
        let (Some(x), Some(y)) = (gx.get(first..n), gy.get(first..n)) else {
            return;
        };
        let mut l = [L::default(); 32];
        let (mut xs, mut ys) = ([0.0f32; 8], [0.0f32; 8]);
        let mut d = [f16::ZERO; 32];
        for (a, &b) in l.iter_mut().zip(src) {
            *a = b;
        }
        for ((a, &b), (c, &e)) in xs.iter_mut().zip(x).zip(ys.iter_mut().zip(y)) {
            *a = b;
            *c = e;
        }
        step(l.as_ptr(), xs.as_ptr(), ys.as_ptr(), d.as_mut_ptr());
        for (o, &v) in out.iter_mut().zip(&d) {
            *o = v;
        }
    }
}

/// dist2bbox in f16 of f16 `[l, t, r, b]` distances, four per anchor, with grid centres and the
/// stride rounded to f16: `[xc, yc, w, h]` in the Ultralytics order, or
/// `[(gx − l)·s, (gy − t)·s, (gx + r)·s, (gy + b)·s]` when `xyxy`. Writes nothing unless `ltrb`
/// and `dst` have the same length, a multiple of 4, and the grids cover every anchor.
///
/// Tested against the float64 boxes within the bound `tests/decode_ops.rs` states.
///
/// # Safety
/// The CPU must support FEAT_FP16 (`CpuFeatures::neon_fp16`).
#[inline]
#[target_feature(enable = "neon")]
pub unsafe fn dist2bbox_f16_neon_fp16(
    ltrb: &[f16],
    gx: &[f32],
    gy: &[f32],
    stride: f32,
    xyxy: bool,
    dst: &mut [f16],
) {
    // SAFETY: the caller guarantees FEAT_FP16.
    unsafe { dist2bbox_f16(ltrb, gx, gy, stride, xyxy, dst) }
}

/// Loads 16 integers as f16 lanes, each rounded once to nearest even.
trait LoadF16: Copy + Default {
    /// # Safety
    /// `p` must be valid for reading 16 elements; the CPU must support FEAT_FP16.
    unsafe fn load16_f16(p: *const Self) -> [uint16x8_t; 2];
}

impl LoadF16 for i8 {
    #[inline(always)]
    unsafe fn load16_f16(p: *const Self) -> [uint16x8_t; 2] {
        // SAFETY: the caller guarantees 16 readable elements and FEAT_FP16.
        unsafe {
            let v = vld1q_s8(p);
            [
                scvtf_f16_from_s16x8(vmovl_s8(vget_low_s8(v))),
                scvtf_f16_from_s16x8(vmovl_high_s8(v)),
            ]
        }
    }
}

impl LoadF16 for u8 {
    #[inline(always)]
    unsafe fn load16_f16(p: *const Self) -> [uint16x8_t; 2] {
        // SAFETY: as above.
        unsafe {
            let v = vld1q_u8(p);
            [
                ucvtf_f16_from_u16x8(vmovl_u8(vget_low_u8(v))),
                ucvtf_f16_from_u16x8(vmovl_high_u8(v)),
            ]
        }
    }
}

impl LoadF16 for i16 {
    #[inline(always)]
    unsafe fn load16_f16(p: *const Self) -> [uint16x8_t; 2] {
        // SAFETY: as above.
        unsafe {
            [
                scvtf_f16_from_s16x8(vld1q_s16(p)),
                scvtf_f16_from_s16x8(vld1q_s16(p.add(8))),
            ]
        }
    }
}

impl LoadF16 for u16 {
    #[inline(always)]
    unsafe fn load16_f16(p: *const Self) -> [uint16x8_t; 2] {
        // SAFETY: as above.
        unsafe {
            [
                ucvtf_f16_from_u16x8(vld1q_u16(p)),
                ucvtf_f16_from_u16x8(vld1q_u16(p.add(8))),
            ]
        }
    }
}

/// `x · scale + bias` (`x · scale` without a bias) with FP16 arithmetic.
///
/// # Safety
/// The CPU must support FEAT_FP16.
#[inline(always)]
unsafe fn scale_bias_fp16<S: LoadF16>(
    src: &[S],
    scale: f16,
    bias: Option<f16>,
    dst: &mut [MaybeUninit<f16>],
) {
    // SAFETY: NEON is part of the aarch64 baseline.
    let sc = unsafe { vdupq_n_u16(scale.to_bits()) };
    let b = bias.map(|b| unsafe { vdupq_n_u16(b.to_bits()) });
    // SAFETY: FEAT_FP16 by contract; 16 valid elements per step; `f16` is a transparent `u16`.
    map_steps_uninit(src, dst, |s, d| unsafe {
        let x = S::load16_f16(s);
        let d = d.cast::<u16>();
        for (k, v) in x.into_iter().enumerate() {
            let p = fmul_f16x8(v, sc);
            vst1q_u16(d.add(8 * k), b.map_or(p, |b| fadd_f16x8(p, b)));
        }
    });
}

macro_rules! scale_bias_kernel {
    ($(#[$doc:meta])* $name:ident, $t:ty) => {
        $(#[$doc])*
        ///
        /// Writes every element of `dst` when the lengths match, nothing otherwise.
        ///
        /// # Safety
        /// The CPU must support FEAT_FP16 (`CpuFeatures::neon_fp16`).
        #[inline]
        #[target_feature(enable = "neon")]
        pub unsafe fn $name(
            src: &[$t],
            scale: f16,
            bias: Option<f16>,
            dst: &mut [MaybeUninit<f16>],
        ) {
            // SAFETY: the caller guarantees FEAT_FP16.
            unsafe { scale_bias_fp16(src, scale, bias, dst) }
        }
    };
}

scale_bias_kernel!(
    /// `neon_convert::scale_bias_i8_to_f16_neon` with FP16 arithmetic, bit for bit: every step
    /// there is a correctly rounded f16 operation.
    scale_bias_i8_to_f16_neon_fp16,
    i8
);
scale_bias_kernel!(
    /// `neon_convert::scale_bias_u8_to_f16_neon` with FP16 arithmetic, bit for bit.
    scale_bias_u8_to_f16_neon_fp16,
    u8
);
scale_bias_kernel!(
    /// `neon_convert::scale_bias_i16_to_f16_neon` with FP16 arithmetic, bit for bit.
    scale_bias_i16_to_f16_neon_fp16,
    i16
);
scale_bias_kernel!(
    /// `neon_convert::scale_bias_u16_to_f16_neon` with FP16 arithmetic, bit for bit.
    scale_bias_u16_to_f16_neon_fp16,
    u16
);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::prim::dequant as dq;
    use crate::test_support::report_skip;

    /// The bound every kernel here states, in f16 ulps.
    const ULP_BOUND: i32 = 2;

    fn fp16() -> bool {
        let ok = std::arch::is_aarch64_feature_detected!("fp16");
        if !ok {
            report_skip("no FEAT_FP16 on this CPU");
        }
        ok
    }

    /// `f16` distance in ulps between two non-negative results.
    fn ulps(a: f16, b: f16) -> i32 {
        (i32::from(a.to_bits()) - i32::from(b.to_bits())).abs()
    }

    fn reference(x: f64) -> f16 {
        f16::from_f64(1.0 / (1.0 + (-x).exp()))
    }

    /// Asserts every `got` is within the bound of the f64 sigmoid of `xs`; NaN must give NaN.
    fn check(got: &[f16], xs: &[f64], what: &str) {
        for (&g, &x) in got.iter().zip(xs) {
            if x.is_nan() {
                assert!(g.is_nan(), "{what}: NaN gave {g}");
                continue;
            }
            let e = ulps(g, reference(x));
            assert!(
                e <= ULP_BOUND,
                "{what}: x {x}: got {g}, want {} ({e} ulp)",
                reference(x)
            );
        }
    }

    #[test]
    fn sigmoid_fp16_constants_are_the_f16_values() {
        let h = |v: f64| f16::from_f64(v).to_bits();
        assert_eq!(bits::ONE, h(1.0));
        assert_eq!(bits::HALF, h(0.5));
        assert_eq!(bits::ONE_SIXTH, h(1.0 / 6.0));
        assert_eq!(bits::ONE_TWENTY_FOURTH, h(1.0 / 24.0));
        assert_eq!(bits::LOG2_E, h(core::f64::consts::LOG2_E));
        assert_eq!(f16::from_bits(bits::LN2_HI).to_f64(), 0.693_359_375);
        assert_eq!(
            bits::LN2_LO,
            h(core::f64::consts::LN_2 - f16::from_bits(bits::LN2_HI).to_f64())
        );
        assert_eq!(bits::T_MIN, h(f64::from(T_MIN_F32)));
        assert_eq!(bits::EIGHT, h(8.0));
        assert_eq!(bits::IOTA, core::array::from_fn(|i| h(i as f64)));
        // Every reachable `k` stays in [−25, 0], so both injected exponents are in range.
        assert_eq!(
            (T_MIN_F32 * core::f32::consts::LOG2_E).round_ties_even(),
            -25.0
        );
        assert!(reference(f64::from(T_MIN_F32)).to_bits() == 0);
    }

    #[test]
    fn sigmoid_fp16_f16_input_every_value_within_bound() {
        if !fp16() {
            return;
        }
        let src: Vec<f16> = (0..=u16::MAX).map(f16::from_bits).collect();
        let xs: Vec<f64> = src.iter().map(|v| v.to_f64()).collect();
        let mut got = vec![f16::ZERO; src.len()];
        // SAFETY: FEAT_FP16 was detected.
        unsafe { dequant_sigmoid_f16_to_f16_neon_fp16(&src, Quant::IDENTITY, &mut got) };
        check(&got, &xs, "f16");
        assert_eq!(got[0x7c00], f16::ONE, "+inf");
        assert_eq!(got[0xfc00], f16::ZERO, "-inf");
        // A non-identity quant dequantizes in f32 first.
        let q = Quant::new(0.5, 3);
        let mut got = vec![f16::ZERO; src.len()];
        // SAFETY: as above.
        unsafe { dequant_sigmoid_f16_to_f16_neon_fp16(&src, q, &mut got) };
        let mut x32 = vec![0f32; src.len()];
        dq::dequant_f16_to_f32(&src, q, &mut x32);
        let xs: Vec<f64> = x32.iter().map(|&v| f64::from(v)).collect();
        check(&got, &xs, "f16 with a quant");
    }

    /// Every f16 in [0.999, 1] (0.99902, 0.99951 and 1) is produced where the float64 sigmoid
    /// rounds to it, and each result there is within the bound, for f32, f16 and i16 inputs.
    #[test]
    fn sigmoid_fp16_reaches_every_value_near_one() {
        if !fp16() {
            return;
        }
        let near = [0x3bfe_u16, 0x3bff, 0x3c00].map(f16::from_bits);
        let f32s: Vec<f32> = (0..400_001).map(|i| 6.0 + i as f32 * 2.5e-5).collect();
        let f16s: Vec<f16> = (0x4600..=0x4a00).map(f16::from_bits).collect();
        let i16s: Vec<i16> = (0..=i16::MAX).collect();
        let q = Quant::new(4e-4, 0);
        let mut got32 = vec![f16::ZERO; f32s.len()];
        let mut got16 = vec![f16::ZERO; f16s.len()];
        let mut goti16 = vec![f16::ZERO; i16s.len()];
        // SAFETY: FEAT_FP16 was detected.
        unsafe {
            dequant_sigmoid_f32_to_f16_neon_fp16(&f32s, Quant::IDENTITY, &mut got32);
            dequant_sigmoid_f16_to_f16_neon_fp16(&f16s, Quant::IDENTITY, &mut got16);
            dequant_sigmoid_i16_to_f16_neon_fp16(&i16s, q, &mut goti16);
        }
        let cases: [(&str, Vec<f64>, &[f16]); 3] = [
            ("f32", f32s.iter().map(|&v| f64::from(v)).collect(), &got32),
            ("f16", f16s.iter().map(|v| v.to_f64()).collect(), &got16),
            (
                "i16",
                i16s.iter()
                    .map(|&v| f64::from(v) * f64::from(q.scale))
                    .collect(),
                &goti16,
            ),
        ];
        for (what, xs, got) in cases {
            let mut worst = 0;
            let mut hit = [false; 3];
            for (&x, &g) in xs.iter().zip(got) {
                let want = reference(x);
                if want.to_f64() < 0.999 {
                    continue;
                }
                let e = ulps(g, want);
                worst = worst.max(e);
                assert!(
                    e <= ULP_BOUND,
                    "{what}: x {x}: got {g}, want {want} ({e} ulp)"
                );
                for (h, &v) in hit.iter_mut().zip(&near) {
                    *h |= g == v && want == v;
                }
            }
            assert_eq!(hit, [true; 3], "{what}: values near 1 produced");
            eprintln!("{what}: worst {worst} f16 ulp for sigmoid in [0.999, 1]");
        }
    }

    #[test]
    fn sigmoid_fp16_integer_and_f32_inputs_within_bound() {
        if !fp16() {
            return;
        }
        let i8s: Vec<i8> = (i8::MIN..=i8::MAX).collect();
        let u8s: Vec<u8> = (0..=u8::MAX).collect();
        let i16s: Vec<i16> = (i16::MIN..=i16::MAX).collect();
        let u16s: Vec<u16> = (0..=u16::MAX).collect();
        let exact = |v: f64, q: Quant| (v - f64::from(q.zero_point)) * f64::from(q.scale);
        for q in [
            Quant::new(0.1, 0),
            Quant::new(0.2, -20),
            Quant::new(0.07, -12),
            Quant::new(0.15, 30),
            Quant::new(1.0, 0),
        ] {
            let mut got = vec![f16::ZERO; i8s.len()];
            // SAFETY: FEAT_FP16 was detected.
            unsafe { dequant_sigmoid_i8_to_f16_neon_fp16(&i8s, q, &mut got) };
            let xs: Vec<f64> = i8s.iter().map(|&v| exact(f64::from(v), q)).collect();
            check(&got, &xs, &format!("i8 {q:?}"));
            let q = Quant::new(q.scale, q.zero_point.max(0));
            let mut got = vec![f16::ZERO; u8s.len()];
            // SAFETY: as above.
            unsafe { dequant_sigmoid_u8_to_f16_neon_fp16(&u8s, q, &mut got) };
            let xs: Vec<f64> = u8s.iter().map(|&v| exact(f64::from(v), q)).collect();
            check(&got, &xs, &format!("u8 {q:?}"));
        }
        for q in [
            Quant::new(0.0005, 0),
            Quant::new(0.001, 0),
            Quant::new(1e-5, 0),
            Quant::new(0.01, 100),
            Quant::new(1.0, 0),
        ] {
            let mut got = vec![f16::ZERO; i16s.len()];
            // SAFETY: FEAT_FP16 was detected.
            unsafe { dequant_sigmoid_i16_to_f16_neon_fp16(&i16s, q, &mut got) };
            let xs: Vec<f64> = i16s.iter().map(|&v| exact(f64::from(v), q)).collect();
            check(&got, &xs, &format!("i16 {q:?}"));
            let q = Quant::new(q.scale, q.zero_point + 32768);
            let mut got = vec![f16::ZERO; u16s.len()];
            // SAFETY: as above.
            unsafe { dequant_sigmoid_u16_to_f16_neon_fp16(&u16s, q, &mut got) };
            let xs: Vec<f64> = u16s.iter().map(|&v| exact(f64::from(v), q)).collect();
            check(&got, &xs, &format!("u16 {q:?}"));
        }
        let mut f32s: Vec<f32> = (0..200_001)
            .map(|i| (i as f32 - 100_000.0) * 2.5e-4)
            .collect();
        f32s.extend([
            f32::NAN,
            f32::INFINITY,
            f32::NEG_INFINITY,
            -0.0,
            1e30,
            -1e30,
        ]);
        let mut got = vec![f16::ZERO; f32s.len()];
        // SAFETY: FEAT_FP16 was detected.
        unsafe { dequant_sigmoid_f32_to_f16_neon_fp16(&f32s, Quant::IDENTITY, &mut got) };
        let xs: Vec<f64> = f32s.iter().map(|&v| f64::from(v)).collect();
        check(&got, &xs, "f32");
    }

    #[test]
    fn decode_fp16_mismatched_lengths_write_nothing() {
        if !fp16() {
            return;
        }
        let logits = [0.0f32; 4 * 16 * 2];
        let mut dst = [f16::ONE; 7];
        // SAFETY: FEAT_FP16 was detected.
        unsafe { dfl_softmax_ltrb_f16_neon_fp16(&logits, 16, &mut dst) };
        assert!(dst.iter().all(|&v| v == f16::ONE));
        let mut dst = [f16::ONE; 8];
        // SAFETY: as above.
        unsafe { dfl_softmax_ltrb_f16_neon_fp16(&logits[..127], 16, &mut dst) };
        // SAFETY: as above.
        unsafe { dfl_softmax_ltrb_f16_neon_fp16(&logits, 0, &mut dst) };
        assert!(dst.iter().all(|&v| v == f16::ONE));
        let (ltrb, grid) = ([f16::ONE; 12], [0.5f32; 3]);
        let mut boxes = [f16::ZERO; 12];
        // SAFETY: as above.
        unsafe {
            dist2bbox_f16_neon_fp16(&ltrb[..8], &grid, &grid, 8.0, false, &mut boxes);
            dist2bbox_f16_neon_fp16(&ltrb, &grid[..2], &grid, 8.0, false, &mut boxes);
            dist2bbox_f16_neon_fp16(&ltrb, &grid, &grid[..2], 8.0, true, &mut boxes);
            dist2bbox_f16_neon_fp16(&ltrb[..10], &grid, &grid, 8.0, true, &mut boxes[..10]);
        }
        assert!(boxes.iter().all(|&v| v == f16::ZERO));
        // Three anchors take the padded tail: `[1, 1, 1, 1]` at (0.5, 0.5), stride 8.
        // SAFETY: as above.
        unsafe { dist2bbox_f16_neon_fp16(&ltrb, &grid, &grid, 8.0, false, &mut boxes) };
        assert!(boxes
            .chunks_exact(4)
            .all(|b| b == [4.0, 4.0, 16.0, 16.0].map(f16::from_f32)));
        // SAFETY: as above.
        unsafe { dist2bbox_f16_neon_fp16(&ltrb, &grid, &grid, 8.0, true, &mut boxes) };
        assert!(boxes
            .chunks_exact(4)
            .all(|b| b == [-4.0, -4.0, 12.0, 12.0].map(f16::from_f32)));
    }

    #[test]
    fn sigmoid_fp16_mismatched_lengths_write_nothing() {
        if !fp16() {
            return;
        }
        let src = [0i8; 20];
        let mut dst = [f16::ONE; 19];
        // SAFETY: FEAT_FP16 was detected.
        unsafe { dequant_sigmoid_i8_to_f16_neon_fp16(&src, Quant::new(0.1, 0), &mut dst) };
        assert!(dst.iter().all(|&v| v == f16::ONE));
    }
}
