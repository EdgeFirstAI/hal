// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! NEON instance-mask kernels: one detection's mask coefficients dotted with every proto pixel
//! of a crop row, the sigmoid of those logits scaled to `u8`, the mask threshold, and the
//! dequantize of NCHW protos into NHWC.
//!
//! Protos are NHWC (a pixel's `K` channels contiguous, so a crop row is `n · K` contiguous
//! elements) or NCHW (channel `c` of pixel `j` at `planes[c · stride + j]`). The integer kernels
//! return the exact `Σₖ (mₖ − zm)(pₖ − zp)`:
//!
//! - 8-bit NHWC maps unsigned operands to signed by flipping the top bit (`u − 128`, zero point
//!   likewise) and computes `Σ m·p − zm·Σ p − zp·Σ m + K·zm·zp`. Every operand and zero point is
//!   then in [−128, 127], so no partial sum exceeds `K · 2^15` and the result `K · 2^16` in
//!   magnitude, which fits i32 for `K ≤ MAX_K`. `Σ m·p` is `sdot` on the dotprod kernels and
//!   `smull` + `sadalp` on the others. When every `m − zm` fits i8 (and `K ≤ 256`) they take
//!   `Σ (m − zm)·p − zp·Σ (m − zm)` instead, which needs no per-pixel `Σ p`.
//! - 8-bit NCHW multiplies `p` by `m − zm` (at most 255 in magnitude) with `smlal` and subtracts
//!   `zp · Σ (m − zm)`; both terms stay below `K · 2^16`.
//! - Kernels with a 16-bit operand work in i16 lanes (`u16 − 32768`) and accumulate in i64: the
//!   result is below `K · 2^33`, so rounding it to f32 once through f64 (exact below 2^53) is the
//!   correctly rounded integer.
//!
//! The float kernels accumulate `m · p` in f32 with FMA: NHWC in four interleaved partial sums
//! per pixel, so a logit is within `(K/4 + 5) · 2⁻²⁴ · Σ |m·p|` of the exact dot product; NCHW
//! one sequential sum per pixel, within `K · 2⁻²⁴ · Σ |m·p|`. Integer and `f16` protos convert to
//! f32 exactly. Only the `*_dotprod` kernels use FEAT_DotProd (inline asm, since stable Rust has
//! no `sdot` intrinsic); everything else is ARMv8.0 Advanced SIMD. A kernel returns `false` and
//! writes nothing when its arguments are invalid.

#![cfg(target_arch = "aarch64")]

use std::arch::aarch64::*; // NOSONAR

use half::f16;

use super::neon_convert::{affine, map_steps, widen8, LoadF32};
use super::neon_sigmoid::sigmoid4;
use crate::Quant;

/// Largest channel count the integer kernels take; the 8-bit bounds above need `K ≤ 32767`.
pub const MAX_K: usize = 32767;

/// Proto pixels per NHWC block.
const PX: usize = 4;
/// Most channels whose coefficient zero point the 8-bit NHWC kernels fold into the coefficients.
const FOLD_K: usize = 256;
/// Proto pixels per 8-bit and float NCHW block.
const PX_NCHW: usize = 16;
/// Proto pixels per wide NCHW block.
const PX_NCHW_WIDE: usize = 8;

mod sealed {
    pub trait Sealed {}
    impl Sealed for i8 {}
    impl Sealed for u8 {}
    impl Sealed for i16 {}
    impl Sealed for u16 {}
    impl Sealed for f32 {}
    impl Sealed for half::f16 {}
}

/// An integer coefficient or proto element: `i8`, `u8`, `i16` or `u16`.
pub trait MaskInt: Copy + Default + sealed::Sealed + 'static {
    /// Smallest value, and the smallest zero point the kernels take.
    const MIN: i32;
    /// Largest value, and the largest zero point the kernels take.
    const MAX: i32;
    /// Subtracted to land in i16: 32768 for `u16`, 0 otherwise.
    #[doc(hidden)]
    const OFF16: i32;

    /// The value.
    fn get(self) -> i32;

    /// Eight elements minus [`Self::OFF16`], as i16 lanes.
    ///
    /// # Safety
    /// `p` must be readable for 8 elements.
    #[doc(hidden)]
    unsafe fn load8_i16(p: *const Self) -> int16x8_t;
}

/// An 8-bit integer element: `i8` or `u8`.
pub trait MaskInt8: MaskInt {
    /// Subtracted to land in i8: 128 for `u8`, 0 for `i8`.
    #[doc(hidden)]
    const OFF8: i32;

    /// Sixteen elements minus [`Self::OFF8`], as i8 lanes.
    ///
    /// # Safety
    /// `p` must be readable for 16 elements.
    #[doc(hidden)]
    unsafe fn load16_i8(p: *const Self) -> int8x16_t;

    /// Eight elements minus [`Self::OFF8`], as i8 lanes.
    ///
    /// # Safety
    /// `p` must be readable for 8 elements.
    #[doc(hidden)]
    unsafe fn load8_i8(p: *const Self) -> int8x8_t;
}

impl MaskInt for i8 {
    const MIN: i32 = i8::MIN as i32;
    const MAX: i32 = i8::MAX as i32;
    const OFF16: i32 = 0;
    #[inline(always)]
    fn get(self) -> i32 {
        self.into()
    }
    #[inline(always)]
    unsafe fn load8_i16(p: *const Self) -> int16x8_t {
        // SAFETY: the caller guarantees 8 readable elements.
        unsafe { vmovl_s8(vld1_s8(p)) }
    }
}

impl MaskInt8 for i8 {
    const OFF8: i32 = 0;
    #[inline(always)]
    unsafe fn load16_i8(p: *const Self) -> int8x16_t {
        // SAFETY: the caller guarantees 16 readable elements.
        unsafe { vld1q_s8(p) }
    }
    #[inline(always)]
    unsafe fn load8_i8(p: *const Self) -> int8x8_t {
        // SAFETY: the caller guarantees 8 readable elements.
        unsafe { vld1_s8(p) }
    }
}

impl MaskInt for u8 {
    const MIN: i32 = 0;
    const MAX: i32 = u8::MAX as i32;
    const OFF16: i32 = 0;
    #[inline(always)]
    fn get(self) -> i32 {
        self.into()
    }
    #[inline(always)]
    unsafe fn load8_i16(p: *const Self) -> int16x8_t {
        // SAFETY: the caller guarantees 8 readable elements.
        unsafe { vreinterpretq_s16_u16(vmovl_u8(vld1_u8(p))) }
    }
}

impl MaskInt8 for u8 {
    const OFF8: i32 = 128;
    #[inline(always)]
    unsafe fn load16_i8(p: *const Self) -> int8x16_t {
        // SAFETY: the caller guarantees 16 readable elements.
        unsafe { vreinterpretq_s8_u8(veorq_u8(vld1q_u8(p), vdupq_n_u8(0x80))) }
    }
    #[inline(always)]
    unsafe fn load8_i8(p: *const Self) -> int8x8_t {
        // SAFETY: the caller guarantees 8 readable elements.
        unsafe { vreinterpret_s8_u8(veor_u8(vld1_u8(p), vdup_n_u8(0x80))) }
    }
}

impl MaskInt for i16 {
    const MIN: i32 = i16::MIN as i32;
    const MAX: i32 = i16::MAX as i32;
    const OFF16: i32 = 0;
    #[inline(always)]
    fn get(self) -> i32 {
        self.into()
    }
    #[inline(always)]
    unsafe fn load8_i16(p: *const Self) -> int16x8_t {
        // SAFETY: the caller guarantees 8 readable elements.
        unsafe { vld1q_s16(p) }
    }
}

impl MaskInt for u16 {
    const MIN: i32 = 0;
    const MAX: i32 = u16::MAX as i32;
    const OFF16: i32 = 32768;
    #[inline(always)]
    fn get(self) -> i32 {
        self.into()
    }
    #[inline(always)]
    unsafe fn load8_i16(p: *const Self) -> int16x8_t {
        // SAFETY: the caller guarantees 8 readable elements.
        unsafe { vreinterpretq_s16_u16(veorq_u16(vld1q_u16(p), vdupq_n_u16(0x8000))) }
    }
}

/// A float-path proto element, converted to f32 exactly: `f32`, `f16`, `i8` or `u8`.
pub trait MaskFloatIn: Copy + Default + sealed::Sealed + 'static {
    /// The value as f32.
    fn get(self) -> f32;

    /// Sixteen elements as f32 lanes.
    ///
    /// # Safety
    /// `p` must be readable for 16 elements.
    #[doc(hidden)]
    unsafe fn load16(p: *const Self) -> [float32x4_t; 4];

    /// Four elements as f32 lanes.
    ///
    /// # Safety
    /// `p` must be readable for 4 elements.
    #[doc(hidden)]
    unsafe fn load4(p: *const Self) -> float32x4_t;
}

macro_rules! float_in {
    ($t:ty, |$p:ident| $load4:expr) => {
        impl MaskFloatIn for $t {
            #[inline(always)]
            fn get(self) -> f32 {
                f32::from(self)
            }
            #[inline(always)]
            unsafe fn load16(p: *const Self) -> [float32x4_t; 4] {
                // SAFETY: the caller guarantees 16 readable elements.
                unsafe { <$t as LoadF32>::load16(p) }
            }
            #[inline(always)]
            unsafe fn load4($p: *const Self) -> float32x4_t {
                // SAFETY: the caller guarantees 4 readable elements.
                unsafe { $load4 }
            }
        }
    };
}

float_in!(f32, |p| vld1q_f32(p));
float_in!(f16, |p| {
    let [lo, _] = widen8(vcombine_u16(vld1_u16(p.cast()), vdup_n_u16(0)));
    lo
});
float_in!(i8, |p| {
    let v = [*p, *p.add(1), *p.add(2), *p.add(3)].map(f32::from);
    vld1q_f32(v.as_ptr())
});
float_in!(u8, |p| {
    let v = [*p, *p.add(1), *p.add(2), *p.add(3)].map(f32::from);
    vld1q_f32(v.as_ptr())
});

/// `SDOT`: `acc` plus the dot product of each group of four i8 lanes.
///
/// # Safety
/// The CPU must support FEAT_DotProd.
#[inline(always)]
unsafe fn sdot(acc: int32x4_t, a: int8x16_t, b: int8x16_t) -> int32x4_t {
    let r: int32x4_t;
    // SAFETY: register-only; the caller guarantees FEAT_DotProd.
    unsafe {
        core::arch::asm!(
            ".arch_extension dotprod",
            "sdot {acc:v}.4s, {a:v}.16b, {b:v}.16b",
            acc = inout(vreg) acc => r,
            a = in(vreg) a,
            b = in(vreg) b,
            options(pure, nomem, nostack),
        );
    }
    r
}

/// Lane sums of four vectors: `[Σ a, Σ b, Σ c, Σ d]`.
#[inline(always)]
fn sum4_s32(v: &[int32x4_t; PX]) -> int32x4_t {
    // SAFETY: NEON is part of the aarch64 baseline.
    unsafe { vpaddq_s32(vpaddq_s32(v[0], v[1]), vpaddq_s32(v[2], v[3])) }
}

/// Lane sums of four vectors: `[Σ a, Σ b, Σ c, Σ d]`.
#[inline(always)]
fn sum4_f32(v: &[float32x4_t; PX]) -> float32x4_t {
    // SAFETY: NEON is part of the aarch64 baseline.
    unsafe { vpaddq_f32(vpaddq_f32(v[0], v[1]), vpaddq_f32(v[2], v[3])) }
}

/// Whether `zp` lies in `T`'s range.
#[inline(always)]
fn zp_ok<T: MaskInt>(zp: i32) -> bool {
    (T::MIN..=T::MAX).contains(&zp)
}

/// `n · k`, the NHWC row length, when it matches `len`.
#[inline(always)]
fn nhwc_ok(n: usize, k: usize, len: usize) -> bool {
    n.checked_mul(k) == Some(len)
}

/// Whether `planes` holds `n` pixels of `k` channels `stride` apart.
#[inline(always)]
fn nchw_ok(n: usize, k: usize, stride: usize, len: usize) -> bool {
    n == 0
        || k == 0
        || (k - 1)
            .checked_mul(stride)
            .and_then(|o| o.checked_add(n))
            .is_some_and(|end| end <= len)
}

/// Whether a block of `block` pixels at `planes` can be read for every channel.
#[inline(always)]
fn nchw_room(block: usize, k: usize, stride: usize, len: usize) -> bool {
    nchw_ok(block, k, stride, len)
}

/// Per-pixel sums of `NP` pixels for the 8-bit NHWC kernels, before the lane reduction.
struct Acc8<const NP: usize> {
    dot: [int32x4_t; NP],
    sum: [int32x4_t; NP],
    dot_tail: [i32; NP],
    sum_tail: [i32; NP],
}

/// `Σ m·p` and `Σ p` of `NP` NHWC pixels at `px`, operands minus their `OFF8`.
///
/// # Safety
/// `c` readable for `k` elements, `px` for `NP · k`; `DP` needs FEAT_DotProd.
#[inline(always)]
unsafe fn acc8<M: MaskInt8, P: MaskInt8, const DP: bool, const SUMS: bool, const NP: usize>(
    c: *const M,
    px: *const P,
    k: usize,
) -> Acc8<NP> {
    // SAFETY: the caller guarantees the reads; `sdot` runs only with `DP`.
    unsafe {
        let mut dot = [vdupq_n_s32(0); NP];
        let mut sum = [vdupq_n_s32(0); NP];
        let ones = vdupq_n_s8(1);
        let mut j = 0;
        while j + 16 <= k {
            let cv = M::load16_i8(c.add(j));
            for i in 0..NP {
                let pv = P::load16_i8(px.add(i * k + j));
                if DP {
                    dot[i] = sdot(dot[i], cv, pv);
                    if SUMS {
                        sum[i] = sdot(sum[i], ones, pv);
                    }
                } else {
                    let lo = vmull_s8(vget_low_s8(cv), vget_low_s8(pv));
                    dot[i] = vpadalq_s16(vpadalq_s16(dot[i], lo), vmull_high_s8(cv, pv));
                    if SUMS {
                        sum[i] = vpadalq_s16(sum[i], vpaddlq_s8(pv));
                    }
                }
            }
            j += 16;
        }
        if j + 8 <= k {
            let cv = M::load8_i8(c.add(j));
            for i in 0..NP {
                let pv = P::load8_i8(px.add(i * k + j));
                dot[i] = vpadalq_s16(dot[i], vmull_s8(cv, pv));
                if SUMS {
                    sum[i] = vpadalq_s16(sum[i], vmovl_s8(pv));
                }
            }
            j += 8;
        }
        let mut dot_tail = [0i32; NP];
        let mut sum_tail = [0i32; NP];
        while j < k {
            let cj = (*c.add(j)).get() - M::OFF8;
            for i in 0..NP {
                let pj = (*px.add(i * k + j)).get() - P::OFF8;
                dot_tail[i] += cj * pj;
                sum_tail[i] += pj;
            }
            j += 1;
        }
        Acc8 {
            dot,
            sum,
            dot_tail,
            sum_tail,
        }
    }
}

/// The 8-bit NHWC row: `dst[j] = Σ (m − zm)(p − zp)` over pixel `j` of `px`.
///
/// # Safety
/// `DP` needs FEAT_DotProd.
#[inline(always)]
unsafe fn nhwc8<M: MaskInt8, P: MaskInt8, const DP: bool>(
    coeff: &[M],
    zm: i32,
    px: &[P],
    zp: i32,
    dst: &mut [i32],
) -> bool {
    let (k, n) = (coeff.len(), dst.len());
    if k > MAX_K || !zp_ok::<M>(zm) || !zp_ok::<P>(zp) || !nhwc_ok(n, k, px.len()) {
        return false;
    }
    let (zm, zp) = (zm - M::OFF8, zp - P::OFF8);
    if zm != 0 && k <= FOLD_K {
        // `m − zm` in i8 needs no `Σ p`: the result is `Σ (m − zm)·p − zp·Σ (m − zm)`.
        let mut folded = [0i8; FOLD_K];
        let mut fits = true;
        for (d, &m) in folded.iter_mut().zip(coeff) {
            let v = m.get() - M::OFF8 - zm;
            fits &= (-128..=127).contains(&v);
            *d = v as i8;
        }
        let c = &folded[..k];
        let sum: i32 = c.iter().map(|&v| i32::from(v)).sum();
        if fits {
            // SAFETY: as below; `c` has `k` elements.
            unsafe { nhwc8_rows::<i8, P, DP, false>(c, 0, px, -zp * sum, dst) };
            return true;
        }
    }
    let sum_m: i64 = coeff.iter().map(|&m| i64::from(m.get() - M::OFF8)).sum();
    let Ok(bias) = i32::try_from(k as i64 * i64::from(zm) * i64::from(zp) - i64::from(zp) * sum_m)
    else {
        return false;
    };
    // SAFETY: the lengths were checked above; the caller guarantees FEAT_DotProd for `DP`.
    unsafe {
        if zm == 0 {
            nhwc8_rows::<M, P, DP, false>(coeff, zm, px, bias, dst);
        } else {
            nhwc8_rows::<M, P, DP, true>(coeff, zm, px, bias, dst);
        }
    }
    true
}

/// [`nhwc8`] after validation: `bias` is `K·zm·zp − zp·Σ m` in the signed domain.
///
/// # Safety
/// `px.len() == dst.len() · coeff.len()`; `DP` needs FEAT_DotProd.
#[inline(always)]
unsafe fn nhwc8_rows<M: MaskInt8, P: MaskInt8, const DP: bool, const SUMS: bool>(
    coeff: &[M],
    zm: i32,
    px: &[P],
    bias: i32,
    dst: &mut [i32],
) {
    let (k, n) = (coeff.len(), dst.len());
    let (c, p, d) = (coeff.as_ptr(), px.as_ptr(), dst.as_mut_ptr());
    // SAFETY: every block reads whole pixels of `px` and writes their slots of `dst`.
    unsafe {
        let block = |i: usize| {
            let a = acc8::<M, P, DP, SUMS, PX>(c, p.add(i * k), k);
            let dot = vaddq_s32(sum4_s32(&a.dot), vld1q_s32(a.dot_tail.as_ptr()));
            let r = if SUMS {
                let s = vaddq_s32(sum4_s32(&a.sum), vld1q_s32(a.sum_tail.as_ptr()));
                vmlsq_s32(dot, s, vdupq_n_s32(zm))
            } else {
                dot
            };
            vst1q_s32(d.add(i), vaddq_s32(r, vdupq_n_s32(bias)));
        };
        let mut i = 0;
        while i + PX <= n {
            block(i);
            i += PX;
        }
        if i == n {
            return;
        }
        if n >= PX {
            block(n - PX);
            return;
        }
        for j in i..n {
            let a = acc8::<M, P, DP, SUMS, 1>(c, p.add(j * k), k);
            let mut r = vaddvq_s32(a.dot[0]) + a.dot_tail[0];
            if SUMS {
                r -= zm * (vaddvq_s32(a.sum[0]) + a.sum_tail[0]);
            }
            *d.add(j) = r + bias;
        }
    }
}

/// `dst[j] = Σₖ (coeff[k] − zm)(px[j·K + k] − zp)` for every NHWC pixel `j` of a crop row,
/// exactly, with `smull` + `sadalp`.
///
/// Needs `K = coeff.len() ≤ MAX_K`, zero points in their element type's range and
/// `px.len() == dst.len() · K`.
///
/// # Safety
/// The CPU must support NEON (every aarch64 CPU does).
#[inline]
#[target_feature(enable = "neon")]
pub unsafe fn mask_dot8_nhwc_neon<M: MaskInt8, P: MaskInt8>(
    coeff: &[M],
    zm: i32,
    px: &[P],
    zp: i32,
    dst: &mut [i32],
) -> bool {
    // SAFETY: `DP` is off, so only baseline NEON runs.
    unsafe { nhwc8::<M, P, false>(coeff, zm, px, zp, dst) }
}

/// [`mask_dot8_nhwc_neon`] with `sdot`, bit for bit.
///
/// # Safety
/// The CPU must support FEAT_DotProd (`is_aarch64_feature_detected!("dotprod")`).
#[inline]
#[target_feature(enable = "neon")]
pub unsafe fn mask_dot8_nhwc_dotprod<M: MaskInt8, P: MaskInt8>(
    coeff: &[M],
    zm: i32,
    px: &[P],
    zp: i32,
    dst: &mut [i32],
) -> bool {
    // SAFETY: the caller guarantees FEAT_DotProd.
    unsafe { nhwc8::<M, P, true>(coeff, zm, px, zp, dst) }
}

/// `Σ m·p` (i64) and `Σ p` of `NP` NHWC pixels at `px`, operands minus their `OFF16`.
///
/// # Safety
/// `c` readable for `k` elements, `px` for `NP · k`.
#[inline(always)]
unsafe fn acc16<M: MaskInt, P: MaskInt, const NP: usize>(
    c: *const M,
    px: *const P,
    k: usize,
) -> ([i64; NP], [i64; NP]) {
    // SAFETY: the caller guarantees the reads.
    unsafe {
        let mut dot = [vdupq_n_s64(0); NP];
        let mut sum = [vdupq_n_s32(0); NP];
        let mut j = 0;
        while j + 8 <= k {
            let cv = M::load8_i16(c.add(j));
            for i in 0..NP {
                let pv = P::load8_i16(px.add(i * k + j));
                let lo = vmull_s16(vget_low_s16(cv), vget_low_s16(pv));
                dot[i] = vpadalq_s32(vpadalq_s32(dot[i], lo), vmull_high_s16(cv, pv));
                sum[i] = vpadalq_s16(sum[i], pv);
            }
            j += 8;
        }
        let mut d = [0i64; NP];
        let mut s = [0i64; NP];
        for i in 0..NP {
            d[i] = vaddvq_s64(dot[i]);
            s[i] = i64::from(vaddvq_s32(sum[i]));
        }
        while j < k {
            let cj = i64::from((*c.add(j)).get() - M::OFF16);
            for i in 0..NP {
                let pj = i64::from((*px.add(i * k + j)).get() - P::OFF16);
                d[i] += cj * pj;
                s[i] += pj;
            }
            j += 1;
        }
        (d, s)
    }
}

/// `dst[j] = Σₖ (coeff[k] − zm)(px[j·K + k] − zp)` for every NHWC pixel `j` of a crop row,
/// computed exactly in i64 and rounded to f32 once.
///
/// For any pair of [`MaskInt`] types; needs `K ≤ MAX_K`, zero points in range and
/// `px.len() == dst.len() · K`.
///
/// # Safety
/// The CPU must support NEON (every aarch64 CPU does).
#[inline]
#[target_feature(enable = "neon")]
pub unsafe fn mask_dot_wide_nhwc_neon<M: MaskInt, P: MaskInt>(
    coeff: &[M],
    zm: i32,
    px: &[P],
    zp: i32,
    dst: &mut [f32],
) -> bool {
    let (k, n) = (coeff.len(), dst.len());
    if k > MAX_K || !zp_ok::<M>(zm) || !zp_ok::<P>(zp) || !nhwc_ok(n, k, px.len()) {
        return false;
    }
    let (zm, zp) = (i64::from(zm - M::OFF16), i64::from(zp - P::OFF16));
    let sum_m: i64 = coeff.iter().map(|&m| i64::from(m.get() - M::OFF16)).sum();
    let bias = k as i64 * zm * zp - zp * sum_m;
    let (c, p) = (coeff.as_ptr(), px.as_ptr());
    let finish = |d: i64, s: i64| (d - zm * s + bias) as f64 as f32;
    let mut i = 0;
    while i + PX <= n {
        // SAFETY: `PX` whole pixels from `i` lie inside `px`.
        let (d, s) = unsafe { acc16::<M, P, PX>(c, p.add(i * k), k) };
        for (o, (d, s)) in dst[i..i + PX].iter_mut().zip(d.into_iter().zip(s)) {
            *o = finish(d, s);
        }
        i += PX;
    }
    for (j, o) in dst.iter_mut().enumerate().skip(i) {
        // SAFETY: pixel `j` lies inside `px`.
        let ([d], [s]) = unsafe { acc16::<M, P, 1>(c, p.add(j * k), k) };
        *o = finish(d, s);
    }
    true
}

/// `(m − zm) · p` summed over channels for `PX_NCHW` pixels at `planes`, in i32.
///
/// # Safety
/// `planes` readable for `PX_NCHW` elements at each `c · stride`.
#[inline(always)]
unsafe fn nchw8_block<M: MaskInt8, P: MaskInt8>(
    coeff: &[M],
    zm: i32,
    planes: *const P,
    stride: usize,
) -> [int32x4_t; 4] {
    // SAFETY: the caller guarantees the reads.
    unsafe {
        let mut a = [vdupq_n_s32(0); 4];
        for (c, &m) in coeff.iter().enumerate() {
            // At most 255 in magnitude: both values lie in the 8-bit type's range.
            let w = (m.get() - zm) as i16;
            let p = planes.add(c * stride);
            let (lo, hi) = (P::load8_i16(p), P::load8_i16(p.add(8)));
            a[0] = vmlal_n_s16(a[0], vget_low_s16(lo), w);
            a[1] = vmlal_high_n_s16(a[1], lo, w);
            a[2] = vmlal_n_s16(a[2], vget_low_s16(hi), w);
            a[3] = vmlal_high_n_s16(a[3], hi, w);
        }
        a
    }
}

/// `dst[j] = Σ_c (coeff[c] − zm)(planes[c·stride + j] − zp)` for every pixel `j` of an NCHW crop
/// row, exactly, with `smlal`.
///
/// Needs `K = coeff.len() ≤ MAX_K`, zero points in range and `planes` to hold `dst.len()`
/// pixels in each of the `K` planes `stride` apart.
///
/// # Safety
/// The CPU must support NEON (every aarch64 CPU does).
#[inline]
#[target_feature(enable = "neon")]
pub unsafe fn mask_dot8_nchw_neon<M: MaskInt8, P: MaskInt8>(
    coeff: &[M],
    zm: i32,
    planes: &[P],
    stride: usize,
    zp: i32,
    dst: &mut [i32],
) -> bool {
    let (k, n) = (coeff.len(), dst.len());
    if k > MAX_K || !zp_ok::<M>(zm) || !zp_ok::<P>(zp) || !nchw_ok(n, k, stride, planes.len()) {
        return false;
    }
    if k == 0 {
        dst.fill(0);
        return true;
    }
    let sum_w: i32 = coeff.iter().map(|&m| m.get() - zm).sum();
    let corr = zp * sum_w;
    let base = planes.as_ptr();
    let store = |a: [int32x4_t; 4], out: &mut [i32]| {
        let mut v = [0i32; PX_NCHW];
        // SAFETY: `v` holds 16 lanes.
        unsafe {
            for (g, x) in a.into_iter().enumerate() {
                vst1q_s32(v.as_mut_ptr().add(4 * g), vsubq_s32(x, vdupq_n_s32(corr)));
            }
        }
        for (o, &x) in out.iter_mut().zip(&v) {
            *o = x;
        }
    };
    let mut i = 0;
    while i + PX_NCHW <= n {
        // SAFETY: pixels `i..i + 16` lie inside every plane.
        store(
            unsafe { nchw8_block(coeff, zm, base.add(i), stride) },
            &mut dst[i..],
        );
        i += PX_NCHW;
    }
    if i == n {
        return true;
    }
    if n >= PX_NCHW {
        // SAFETY: as above, for the last 16 pixels.
        let a = unsafe { nchw8_block(coeff, zm, base.add(n - PX_NCHW), stride) };
        store(a, &mut dst[n - PX_NCHW..]);
    } else if nchw_room(PX_NCHW, k, stride, planes.len()) {
        // SAFETY: the planes hold 16 readable pixels from the row start.
        store(unsafe { nchw8_block(coeff, zm, base, stride) }, dst);
    } else {
        for (j, o) in dst.iter_mut().enumerate() {
            let dot: i32 = coeff
                .iter()
                .enumerate()
                .map(|(c, &m)| (m.get() - zm) * planes[c * stride + j].get())
                .sum();
            *o = dot - corr;
        }
    }
    true
}

/// `(m − zm) · (p − OFF16)` summed over channels for `PX_NCHW_WIDE` pixels at `planes`, in i64.
///
/// # Safety
/// `planes` readable for `PX_NCHW_WIDE` elements at each `c · stride`.
#[inline(always)]
unsafe fn nchw16_block<M: MaskInt, P: MaskInt>(
    coeff: &[M],
    zm: i32,
    planes: *const P,
    stride: usize,
) -> [int64x2_t; 4] {
    // SAFETY: the caller guarantees the reads.
    unsafe {
        let mut a = [vdupq_n_s64(0); 4];
        for (c, &m) in coeff.iter().enumerate() {
            let w = m.get() - zm;
            let p = P::load8_i16(planes.add(c * stride));
            let (lo, hi) = (vmovl_s16(vget_low_s16(p)), vmovl_high_s16(p));
            a[0] = vmlal_n_s32(a[0], vget_low_s32(lo), w);
            a[1] = vmlal_high_n_s32(a[1], lo, w);
            a[2] = vmlal_n_s32(a[2], vget_low_s32(hi), w);
            a[3] = vmlal_high_n_s32(a[3], hi, w);
        }
        a
    }
}

/// `dst[j] = Σ_c (coeff[c] − zm)(planes[c·stride + j] − zp)` for every pixel `j` of an NCHW crop
/// row, computed exactly in i64 and rounded to f32 once.
///
/// For any pair of [`MaskInt`] types; needs `K ≤ MAX_K`, zero points in range and `planes` to
/// hold `dst.len()` pixels in each plane.
///
/// # Safety
/// The CPU must support NEON (every aarch64 CPU does).
#[inline]
#[target_feature(enable = "neon")]
pub unsafe fn mask_dot_wide_nchw_neon<M: MaskInt, P: MaskInt>(
    coeff: &[M],
    zm: i32,
    planes: &[P],
    stride: usize,
    zp: i32,
    dst: &mut [f32],
) -> bool {
    let (k, n) = (coeff.len(), dst.len());
    if k > MAX_K || !zp_ok::<M>(zm) || !zp_ok::<P>(zp) || !nchw_ok(n, k, stride, planes.len()) {
        return false;
    }
    if k == 0 {
        dst.fill(0.0);
        return true;
    }
    let sum_w: i64 = coeff.iter().map(|&m| i64::from(m.get() - zm)).sum();
    let corr = i64::from(zp - P::OFF16) * sum_w;
    let base = planes.as_ptr();
    let store = |a: [int64x2_t; 4], out: &mut [f32]| {
        let mut v = [0i64; PX_NCHW_WIDE];
        // SAFETY: `v` holds 8 lanes.
        unsafe {
            for (g, x) in a.into_iter().enumerate() {
                vst1q_s64(v.as_mut_ptr().add(2 * g), x);
            }
        }
        for (o, &x) in out.iter_mut().zip(&v) {
            *o = (x - corr) as f64 as f32;
        }
    };
    let mut i = 0;
    while i + PX_NCHW_WIDE <= n {
        // SAFETY: pixels `i..i + 8` lie inside every plane.
        store(
            unsafe { nchw16_block(coeff, zm, base.add(i), stride) },
            &mut dst[i..],
        );
        i += PX_NCHW_WIDE;
    }
    if i == n {
        return true;
    }
    if n >= PX_NCHW_WIDE {
        // SAFETY: as above, for the last 8 pixels.
        let a = unsafe { nchw16_block(coeff, zm, base.add(n - PX_NCHW_WIDE), stride) };
        store(a, &mut dst[n - PX_NCHW_WIDE..]);
    } else if nchw_room(PX_NCHW_WIDE, k, stride, planes.len()) {
        // SAFETY: the planes hold 8 readable pixels from the row start.
        store(unsafe { nchw16_block(coeff, zm, base, stride) }, dst);
    } else {
        for (j, o) in dst.iter_mut().enumerate() {
            let dot: i64 = coeff
                .iter()
                .enumerate()
                .map(|(c, &m)| {
                    i64::from(m.get() - zm) * i64::from(planes[c * stride + j].get() - P::OFF16)
                })
                .sum();
            *o = (dot - corr) as f64 as f32;
        }
    }
    true
}

/// `Σ c·p` of `NP` NHWC pixels at `px`: lane-interleaved FMA sums, then the scalar tail.
///
/// # Safety
/// `c` readable for `k` elements, `px` for `NP · k`.
#[inline(always)]
unsafe fn acc_f32<P: MaskFloatIn, const NP: usize>(
    c: *const f32,
    px: *const P,
    k: usize,
) -> [float32x4_t; NP] {
    // SAFETY: the caller guarantees the reads.
    unsafe {
        let mut acc = [vdupq_n_f32(0.0); NP];
        let mut j = 0;
        while j + 16 <= k {
            let cv = <f32 as MaskFloatIn>::load16(c.add(j));
            for (i, a) in acc.iter_mut().enumerate() {
                let pv = P::load16(px.add(i * k + j));
                for (cq, pq) in cv.into_iter().zip(pv) {
                    *a = vfmaq_f32(*a, cq, pq);
                }
            }
            j += 16;
        }
        while j + 4 <= k {
            let cv = vld1q_f32(c.add(j));
            for (i, a) in acc.iter_mut().enumerate() {
                *a = vfmaq_f32(*a, cv, P::load4(px.add(i * k + j)));
            }
            j += 4;
        }
        acc
    }
}

/// `dst[j] = Σₖ coeff[k] · px[j·K + k]` in f32 for every NHWC pixel `j` of a crop row.
///
/// Needs `px.len() == dst.len() · coeff.len()`.
///
/// # Safety
/// The CPU must support NEON (every aarch64 CPU does).
#[inline]
#[target_feature(enable = "neon")]
pub unsafe fn mask_dot_f32_nhwc_neon<P: MaskFloatIn>(
    coeff: &[f32],
    px: &[P],
    dst: &mut [f32],
) -> bool {
    let (k, n) = (coeff.len(), dst.len());
    if !nhwc_ok(n, k, px.len()) {
        return false;
    }
    let tail = k - k % 4;
    let finish = |j: usize, s: f32| {
        coeff[tail..]
            .iter()
            .zip(&px[j * k + tail..(j + 1) * k])
            .fold(s, |s, (&c, &p)| c.mul_add(p.get(), s))
    };
    let (c, p) = (coeff.as_ptr(), px.as_ptr());
    let mut i = 0;
    while i + PX <= n {
        let mut v = [0f32; PX];
        // SAFETY: `PX` whole pixels from `i` lie inside `px`; `v` holds 4 lanes.
        unsafe {
            let a = acc_f32::<P, PX>(c, p.add(i * k), k);
            vst1q_f32(v.as_mut_ptr(), sum4_f32(&a));
        }
        for (q, s) in v.into_iter().enumerate() {
            dst[i + q] = finish(i + q, s);
        }
        i += PX;
    }
    for (j, o) in dst.iter_mut().enumerate().skip(i) {
        // SAFETY: pixel `j` lies inside `px`.
        let [a] = unsafe { acc_f32::<P, 1>(c, p.add(j * k), k) };
        *o = finish(j, vaddvq_f32(a));
    }
    true
}

/// `Σ_c coeff[c] · p` sequentially for `PX_NCHW` pixels at `planes`.
///
/// # Safety
/// `planes` readable for `PX_NCHW` elements at each `c · stride`.
#[inline(always)]
unsafe fn nchw_f32_block<P: MaskFloatIn>(
    coeff: &[f32],
    planes: *const P,
    stride: usize,
) -> [float32x4_t; 4] {
    // SAFETY: the caller guarantees the reads.
    unsafe {
        let mut a = [vdupq_n_f32(0.0); 4];
        for (c, &m) in coeff.iter().enumerate() {
            let p = P::load16(planes.add(c * stride));
            for (x, q) in a.iter_mut().zip(p) {
                *x = vfmaq_n_f32(*x, q, m);
            }
        }
        a
    }
}

/// `dst[j] = Σ_c coeff[c] · planes[c·stride + j]` in f32, channels in order, for every pixel `j`
/// of an NCHW crop row.
///
/// Needs `planes` to hold `dst.len()` pixels in each of the `coeff.len()` planes.
///
/// # Safety
/// The CPU must support NEON (every aarch64 CPU does).
#[inline]
#[target_feature(enable = "neon")]
pub unsafe fn mask_dot_f32_nchw_neon<P: MaskFloatIn>(
    coeff: &[f32],
    planes: &[P],
    stride: usize,
    dst: &mut [f32],
) -> bool {
    let (k, n) = (coeff.len(), dst.len());
    if !nchw_ok(n, k, stride, planes.len()) {
        return false;
    }
    if k == 0 {
        dst.fill(0.0);
        return true;
    }
    let base = planes.as_ptr();
    let store = |a: [float32x4_t; 4], out: &mut [f32]| {
        let mut v = [0f32; PX_NCHW];
        // SAFETY: `v` holds 16 lanes.
        unsafe {
            for (g, x) in a.into_iter().enumerate() {
                vst1q_f32(v.as_mut_ptr().add(4 * g), x);
            }
        }
        for (o, &x) in out.iter_mut().zip(&v) {
            *o = x;
        }
    };
    let mut i = 0;
    while i + PX_NCHW <= n {
        // SAFETY: pixels `i..i + 16` lie inside every plane.
        store(
            unsafe { nchw_f32_block(coeff, base.add(i), stride) },
            &mut dst[i..],
        );
        i += PX_NCHW;
    }
    if i == n {
        return true;
    }
    if n >= PX_NCHW {
        // SAFETY: as above, for the last 16 pixels.
        let a = unsafe { nchw_f32_block(coeff, base.add(n - PX_NCHW), stride) };
        store(a, &mut dst[n - PX_NCHW..]);
    } else if nchw_room(PX_NCHW, k, stride, planes.len()) {
        // SAFETY: the planes hold 16 readable pixels from the row start.
        store(unsafe { nchw_f32_block(coeff, base, stride) }, dst);
    } else {
        for (j, o) in dst.iter_mut().enumerate() {
            *o = coeff
                .iter()
                .enumerate()
                .fold(0.0, |s, (c, &m)| m.mul_add(planes[c * stride + j].get(), s));
        }
    }
    true
}

/// Sixteen probabilities to `u8`: `round(255 · y)`, ties away from zero, NaN to 0.
#[inline(always)]
fn to_u8(y: [float32x4_t; 4]) -> uint8x16_t {
    // SAFETY: NEON is part of the aarch64 baseline.
    unsafe {
        let q = y.map(|v| vcvtaq_u32_f32(vmulq_n_f32(v, 255.0)));
        let lo = vcombine_u16(vqmovn_u32(q[0]), vqmovn_u32(q[1]));
        let hi = vcombine_u16(vqmovn_u32(q[2]), vqmovn_u32(q[3]));
        vcombine_u8(vqmovn_u16(lo), vqmovn_u16(hi))
    }
}

/// `dst[j] = round(255 · sigmoid(src[j] · scale))` for integer logits.
///
/// Within 1 of the scalar `(255 / (1 + exp(−x))).round()` (libm `exp`) for every input.
///
/// # Safety
/// The CPU must support NEON (every aarch64 CPU does).
#[inline]
#[target_feature(enable = "neon")]
pub unsafe fn mask_sigmoid_u8_i32_neon(src: &[i32], scale: f32, dst: &mut [u8]) -> bool {
    if src.len() != dst.len() {
        return false;
    }
    let s = vdupq_n_f32(scale);
    map_steps(src, dst, |sp, dp| {
        // SAFETY: `map_steps` hands out 16 valid elements at each pointer.
        unsafe {
            let x = [0, 4, 8, 12].map(|o| vmulq_f32(vcvtq_f32_s32(vld1q_s32(sp.add(o))), s));
            vst1q_u8(dp, to_u8(x.map(sigmoid4)));
        }
    });
    true
}

/// `dst[j] = round(255 · sigmoid(src[j] · scale))` for float logits; NaN maps to 0.
///
/// Within 1 of the scalar `(255 / (1 + exp(−x))).round()` (libm `exp`) for every input.
///
/// # Safety
/// The CPU must support NEON (every aarch64 CPU does).
#[inline]
#[target_feature(enable = "neon")]
pub unsafe fn mask_sigmoid_u8_f32_neon(src: &[f32], scale: f32, dst: &mut [u8]) -> bool {
    if src.len() != dst.len() {
        return false;
    }
    let s = vdupq_n_f32(scale);
    map_steps(src, dst, |sp, dp| {
        // SAFETY: `map_steps` hands out 16 valid elements at each pointer.
        unsafe {
            let x = <f32 as LoadF32>::load16(sp).map(|v| vmulq_f32(v, s));
            vst1q_u8(dp, to_u8(x.map(sigmoid4)));
        }
    });
    true
}

/// `dst[j] = 1` where `src[j] ≥ threshold`, else 0.
///
/// # Safety
/// The CPU must support NEON (every aarch64 CPU does).
#[inline]
#[target_feature(enable = "neon")]
pub unsafe fn mask_threshold_u8_neon(src: &[u8], threshold: u8, dst: &mut [u8]) -> bool {
    if src.len() != dst.len() {
        return false;
    }
    let (t, one) = (vdupq_n_u8(threshold), vdupq_n_u8(1));
    map_steps(src, dst, |sp, dp| {
        // SAFETY: `map_steps` hands out 16 valid elements at each pointer.
        unsafe { vst1q_u8(dp, vandq_u8(vcgeq_u8(vld1q_u8(sp), t), one)) };
    });
    true
}

/// Transposes four vectors: lane `j` of row `i` becomes lane `i` of row `j`.
#[inline(always)]
fn transpose4(r: [float32x4_t; 4]) -> [float32x4_t; 4] {
    // SAFETY: NEON is part of the aarch64 baseline.
    unsafe {
        let t0 = vreinterpretq_f64_f32(vtrn1q_f32(r[0], r[1]));
        let t1 = vreinterpretq_f64_f32(vtrn2q_f32(r[0], r[1]));
        let t2 = vreinterpretq_f64_f32(vtrn1q_f32(r[2], r[3]));
        let t3 = vreinterpretq_f64_f32(vtrn2q_f32(r[2], r[3]));
        [
            vreinterpretq_f32_f64(vtrn1q_f64(t0, t2)),
            vreinterpretq_f32_f64(vtrn1q_f64(t1, t3)),
            vreinterpretq_f32_f64(vtrn2q_f64(t0, t2)),
            vreinterpretq_f32_f64(vtrn2q_f64(t1, t3)),
        ]
    }
}

/// `(x − zero_point) · scale` of NCHW `[c, h, w]` `src` into NHWC `[h, w, c]` `dst`.
#[inline(always)]
fn dequant_nchw_to_nhwc<S: LoadF32 + MaskInt>(
    src: &[S],
    (h, w, c): (usize, usize, usize),
    q: Quant,
    dst: &mut [f32],
) -> bool {
    let hw = h.checked_mul(w);
    let total = hw.and_then(|hw| hw.checked_mul(c));
    let (Some(hw), Some(total)) = (hw, total) else {
        return false;
    };
    if src.len() != total || dst.len() != total {
        return false;
    }
    let (zpf, scale) = (q.zero_point as f32, q.scale);
    let one = |v: S| (v.get() as f32 - zpf) * scale;
    let (cv, wv) = (c - c % 4, w - w % 16);
    // SAFETY: NEON is part of the aarch64 baseline.
    let (zp, sc) = unsafe { (vdupq_n_f32(zpf), vdupq_n_f32(scale)) };
    for y in 0..h {
        for x in (0..wv).step_by(16) {
            let px = y * w + x;
            for ch in (0..cv).step_by(4) {
                // SAFETY: channels `ch..ch + 4` hold pixels `px..px + 16`.
                let v = [0, 1, 2, 3].map(|i| {
                    affine(
                        unsafe { S::load16(src.as_ptr().add((ch + i) * hw + px)) },
                        zp,
                        sc,
                    )
                });
                let [v0, v1, v2, v3] = v;
                for (g, (((a, b), c2), d)) in v0.into_iter().zip(v1).zip(v2).zip(v3).enumerate() {
                    let rows = transpose4([a, b, c2, d]);
                    for (j, r) in rows.into_iter().enumerate() {
                        let o = (px + 4 * g + j) * c + ch;
                        // SAFETY: pixel `px + 4g + j` has channels `ch..ch + 4` inside `dst`.
                        unsafe { vst1q_f32(dst.as_mut_ptr().add(o), r) };
                    }
                }
            }
            for ch in cv..c {
                for j in px..px + 16 {
                    dst[j * c + ch] = one(src[ch * hw + j]);
                }
            }
        }
        for j in y * w + wv..(y + 1) * w {
            for ch in 0..c {
                dst[j * c + ch] = one(src[ch * hw + j]);
            }
        }
    }
    true
}

macro_rules! dequant_nchw_kernel {
    ($name:ident, $t:ty) => {
        #[doc = concat!("`(x − zero_point) · scale` of NCHW `", stringify!($t), "` protos into NHWC f32, ")]
        /// as `prim::dequant` rounds it, bit for bit. Needs `src.len() == dst.len() == h·w·c`.
        ///
        /// # Safety
        /// The CPU must support NEON (every aarch64 CPU does).
        #[inline]
        #[target_feature(enable = "neon")]
        pub unsafe fn $name(
            src: &[$t],
            h: usize,
            w: usize,
            c: usize,
            q: Quant,
            dst: &mut [f32],
        ) -> bool {
            dequant_nchw_to_nhwc(src, (h, w, c), q, dst)
        }
    };
}

dequant_nchw_kernel!(dequant_nchw_to_nhwc_i8_neon, i8);
dequant_nchw_kernel!(dequant_nchw_to_nhwc_u8_neon, u8);
dequant_nchw_kernel!(dequant_nchw_to_nhwc_i16_neon, i16);
dequant_nchw_kernel!(dequant_nchw_to_nhwc_u16_neon, u16);

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests;
