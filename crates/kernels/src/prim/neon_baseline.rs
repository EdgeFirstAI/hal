// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! NEON kernels for per-scale dequantization and decode.
//!
//! Every function in this module is `#[cfg(target_arch = "aarch64")]`
//! and `unsafe fn` annotated `#[target_feature(enable = "neon")]`. The
//! FP16 kernels require `neon_fp16` in [`CpuFeatures`](crate::CpuFeatures).
//!
//! On non-aarch64 builds the module is empty.

#![cfg(target_arch = "aarch64")]

use super::fp16_asm::{
    fadd_f16x8, fcvtns_s16_from_f16x8, fmla_f16x8, fmls_f16x8, fmul_f16x8, scvtf_f16_from_s16x8,
};
use crate::Quant;
use std::arch::aarch64::*; // NOSONAR

// ────────────────────────────────────────────────────────────────────────
// Affine dequant: out[i] = (in[i] - zp) * scale, computed as
// scale * in[i] + bias with bias = -zp * scale.
// ────────────────────────────────────────────────────────────────────────

/// Computes the affine bias `-zp * scale` in f32.
#[inline(always)]
fn affine_bias(q: Quant) -> f32 {
    -(q.zero_point as f32) * q.scale
}

/// NEON i8 → f32 dequant, 16 lanes per iteration.
///
/// # Safety
/// Caller must ensure `input.len() == output.len()` and that the CPU
/// supports NEON.
#[inline]
#[target_feature(enable = "neon")]
pub unsafe fn dequant_i8_to_f32_neon(input: &[i8], q: Quant, output: &mut [f32]) {
    unsafe {
        debug_assert_eq!(input.len(), output.len());
        let scale = q.scale;
        let bias_v = vdupq_n_f32(affine_bias(q));
        let n = input.len();
        let chunks_16 = n / 16;
        let mut i = 0usize;

        let in_ptr = input.as_ptr();
        let out_ptr = output.as_mut_ptr();

        for _ in 0..chunks_16 {
            // Load 16 signed bytes.
            let v_i8 = vld1q_s8(in_ptr.add(i));
            // Widen to two i16 vectors (8 + 8).
            let lo_i16 = vmovl_s8(vget_low_s8(v_i8));
            let hi_i16 = vmovl_high_s8(v_i8);
            // Widen each to two i32 vectors (4 + 4 + 4 + 4 = 16 i32).
            let q0 = vmovl_s16(vget_low_s16(lo_i16));
            let q1 = vmovl_high_s16(lo_i16);
            let q2 = vmovl_s16(vget_low_s16(hi_i16));
            let q3 = vmovl_high_s16(hi_i16);
            // Convert to f32 and apply bias + scale*in via FMA.
            let f0 = vfmaq_n_f32(bias_v, vcvtq_f32_s32(q0), scale);
            let f1 = vfmaq_n_f32(bias_v, vcvtq_f32_s32(q1), scale);
            let f2 = vfmaq_n_f32(bias_v, vcvtq_f32_s32(q2), scale);
            let f3 = vfmaq_n_f32(bias_v, vcvtq_f32_s32(q3), scale);
            // Store 16 f32.
            vst1q_f32(out_ptr.add(i), f0);
            vst1q_f32(out_ptr.add(i + 4), f1);
            vst1q_f32(out_ptr.add(i + 8), f2);
            vst1q_f32(out_ptr.add(i + 12), f3);
            i += 16;
        }

        // Scalar tail.
        let zp = q.zero_point as f32;
        while i < n {
            *out_ptr.add(i) = (*in_ptr.add(i) as f32 - zp) * scale;
            i += 1;
        }
    }
}

/// NEON u8 → f32 dequant with unsigned widening (`vmovl_u8` /
/// `vcvtq_f32_u32`).
///
/// # Safety
/// See `dequant_i8_to_f32_neon`.
#[inline]
#[target_feature(enable = "neon")]
pub unsafe fn dequant_u8_to_f32_neon(input: &[u8], q: Quant, output: &mut [f32]) {
    unsafe {
        debug_assert_eq!(input.len(), output.len());
        let scale = q.scale;
        let bias_v = vdupq_n_f32(affine_bias(q));
        let n = input.len();
        let chunks_16 = n / 16;
        let mut i = 0usize;

        let in_ptr = input.as_ptr();
        let out_ptr = output.as_mut_ptr();

        for _ in 0..chunks_16 {
            let v_u8 = vld1q_u8(in_ptr.add(i));
            let lo_u16 = vmovl_u8(vget_low_u8(v_u8));
            let hi_u16 = vmovl_high_u8(v_u8);
            let q0 = vmovl_u16(vget_low_u16(lo_u16));
            let q1 = vmovl_high_u16(lo_u16);
            let q2 = vmovl_u16(vget_low_u16(hi_u16));
            let q3 = vmovl_high_u16(hi_u16);
            let f0 = vfmaq_n_f32(bias_v, vcvtq_f32_u32(q0), scale);
            let f1 = vfmaq_n_f32(bias_v, vcvtq_f32_u32(q1), scale);
            let f2 = vfmaq_n_f32(bias_v, vcvtq_f32_u32(q2), scale);
            let f3 = vfmaq_n_f32(bias_v, vcvtq_f32_u32(q3), scale);
            vst1q_f32(out_ptr.add(i), f0);
            vst1q_f32(out_ptr.add(i + 4), f1);
            vst1q_f32(out_ptr.add(i + 8), f2);
            vst1q_f32(out_ptr.add(i + 12), f3);
            i += 16;
        }

        let zp = q.zero_point as f32;
        while i < n {
            *out_ptr.add(i) = (*in_ptr.add(i) as f32 - zp) * scale;
            i += 1;
        }
    }
}

/// NEON i16 → f32 dequant, 8 lanes per iteration.
///
/// # Safety
/// See `dequant_i8_to_f32_neon`.
#[inline]
#[target_feature(enable = "neon")]
pub unsafe fn dequant_i16_to_f32_neon(input: &[i16], q: Quant, output: &mut [f32]) {
    unsafe {
        debug_assert_eq!(input.len(), output.len());
        let scale = q.scale;
        let bias_v = vdupq_n_f32(affine_bias(q));
        let n = input.len();
        let chunks_8 = n / 8;
        let mut i = 0usize;

        let in_ptr = input.as_ptr();
        let out_ptr = output.as_mut_ptr();

        for _ in 0..chunks_8 {
            let v_i16 = vld1q_s16(in_ptr.add(i));
            let lo_i32 = vmovl_s16(vget_low_s16(v_i16));
            let hi_i32 = vmovl_high_s16(v_i16);
            let f0 = vfmaq_n_f32(bias_v, vcvtq_f32_s32(lo_i32), scale);
            let f1 = vfmaq_n_f32(bias_v, vcvtq_f32_s32(hi_i32), scale);
            vst1q_f32(out_ptr.add(i), f0);
            vst1q_f32(out_ptr.add(i + 4), f1);
            i += 8;
        }

        let zp = q.zero_point as f32;
        while i < n {
            *out_ptr.add(i) = (*in_ptr.add(i) as f32 - zp) * scale;
            i += 1;
        }
    }
}

/// NEON u16 → f32 dequant.
///
/// # Safety
/// See `dequant_i8_to_f32_neon`.
#[inline]
#[target_feature(enable = "neon")]
pub unsafe fn dequant_u16_to_f32_neon(input: &[u16], q: Quant, output: &mut [f32]) {
    unsafe {
        debug_assert_eq!(input.len(), output.len());
        let scale = q.scale;
        let bias_v = vdupq_n_f32(affine_bias(q));
        let n = input.len();
        let chunks_8 = n / 8;
        let mut i = 0usize;

        let in_ptr = input.as_ptr();
        let out_ptr = output.as_mut_ptr();

        for _ in 0..chunks_8 {
            let v_u16 = vld1q_u16(in_ptr.add(i));
            let lo_u32 = vmovl_u16(vget_low_u16(v_u16));
            let hi_u32 = vmovl_high_u16(v_u16);
            let f0 = vfmaq_n_f32(bias_v, vcvtq_f32_u32(lo_u32), scale);
            let f1 = vfmaq_n_f32(bias_v, vcvtq_f32_u32(hi_u32), scale);
            vst1q_f32(out_ptr.add(i), f0);
            vst1q_f32(out_ptr.add(i + 4), f1);
            i += 8;
        }

        let zp = q.zero_point as f32;
        while i < n {
            *out_ptr.add(i) = (*in_ptr.add(i) as f32 - zp) * scale;
            i += 1;
        }
    }
}

// ────────────────────────────────────────────────────────────────────────
// NEON polynomial expf (Cephes-style minimax), 4 lanes
//
// Algorithm:
//   1. Clamp input to [-88.376, 88.376].
//   2. Range-reduce: `x = k*ln2 + r`, |r| <= ln2/2.
//      `fx = round_to_nearest(x * LOG2_E)`, `r = x - fx * ln2`,
//      with ln2 split into hi+lo parts.
//   3. Approximate exp(r) on [-ln2/2, ln2/2] with a 5-degree minimax
//      polynomial:
//        exp(r) ≈ ((((p0*r + p1)*r + p2)*r + p3)*r + p4)*r + p5)*r² + r + 1
//   4. Reconstruct 2^k via IEEE-754 exponent bit injection
//      (`(k + 127) << 23`, reinterpret as f32).
//   5. Result = exp(r) * 2^k.
//
// Accuracy: 3 ulp f32 across |x| < 88.
// ────────────────────────────────────────────────────────────────────────

/// Vectorised expf on 4 f32 lanes, polynomial approximation.
///
/// # Safety
/// See `dequant_i8_to_f32_neon`. Inputs outside [-88.376, 88.376] are
/// clamped.
#[allow(clippy::excessive_precision)]
#[inline]
#[target_feature(enable = "neon")]
pub unsafe fn expf_neon_f32x4(x: float32x4_t) -> float32x4_t {
    // Range and Cephes minimax constants.
    let exp_hi = vdupq_n_f32(88.376_26);
    let exp_lo = vdupq_n_f32(-88.376_26);
    let log2_e = vdupq_n_f32(core::f32::consts::LOG2_E);
    // 0.693359375 is exact in f32 (binary: 0x3F318000); the lo
    // correction is the residual ln(2) - 0.693359375.
    let ln2_hi = vdupq_n_f32(0.693_359_375);
    let ln2_lo = vdupq_n_f32(-2.121_944_400e-4);
    let one = vdupq_n_f32(1.0);
    let p0 = vdupq_n_f32(1.987_569_150e-4);
    let p1 = vdupq_n_f32(1.398_199_950e-3);
    let p2 = vdupq_n_f32(8.333_451_900e-3);
    let p3 = vdupq_n_f32(4.166_579_590e-2);
    let p4 = vdupq_n_f32(1.666_666_550e-1);
    let p5 = vdupq_n_f32(5.000_000_120e-1);

    // Clamp to the representable range.
    let x = vminq_f32(vmaxq_f32(x, exp_lo), exp_hi);

    // fx = round(x * LOG2_E), to nearest.
    let fx_f = vmulq_f32(x, log2_e);
    let fx_i = vcvtnq_s32_f32(fx_f);
    let fx = vcvtq_f32_s32(fx_i);

    // r = x - fx * ln2, computed as (x - fx*ln2_hi) - fx*ln2_lo.
    let z = vfmsq_f32(x, fx, ln2_hi);
    let r = vfmsq_f32(z, fx, ln2_lo);

    // Horner's method: y = ((((p0*r + p1)*r + p2)*r + p3)*r + p4)*r + p5
    let mut y = vfmaq_f32(p1, p0, r);
    y = vfmaq_f32(p2, y, r);
    y = vfmaq_f32(p3, y, r);
    y = vfmaq_f32(p4, y, r);
    y = vfmaq_f32(p5, y, r);

    // Final step: exp(r) ≈ y * r² + r + 1
    let r2 = vmulq_f32(r, r);
    let y_r2 = vmulq_f32(y, r2);
    let exp_r = vaddq_f32(vaddq_f32(y_r2, r), one);

    // 2^k via IEEE-754 exponent injection: (k + 127) << 23 reinterpreted
    // as f32.
    let bias = vdupq_n_s32(127);
    let pow2k_bits = vshlq_n_s32::<23>(vaddq_s32(fx_i, bias));
    let pow2k = vreinterpretq_f32_s32(pow2k_bits);

    vmulq_f32(exp_r, pow2k)
}

// ────────────────────────────────────────────────────────────────────────
// NEON sigmoid f32
//
//   x >= 0:  1 / (1 + e^-x)
//   x <  0:  e^x / (1 + e^x)
//
// The two branches are blended with `vbslq_f32`.
// ────────────────────────────────────────────────────────────────────────

/// NEON sigmoid f32, in-place.
///
/// # Safety
/// See `dequant_i8_to_f32_neon`.
#[inline]
#[target_feature(enable = "neon")]
pub unsafe fn sigmoid_slice_f32_neon(buf: &mut [f32]) {
    unsafe {
        let n = buf.len();
        let chunks_4 = n / 4;
        let mut i = 0usize;
        let zero = vdupq_n_f32(0.0);
        let one = vdupq_n_f32(1.0);

        let ptr = buf.as_mut_ptr();
        for _ in 0..chunks_4 {
            let x = vld1q_f32(ptr.add(i));
            // Mask: 0xFFFFFFFF for lanes where x >= 0, 0 otherwise.
            let mask = vcgeq_f32(x, zero);
            // Input to expf by sign:
            //   x >= 0  →  -x
            //   x <  0  →   x
            let neg_x = vnegq_f32(x);
            let exp_in = vbslq_f32(mask, neg_x, x);

            let e = expf_neon_f32x4(exp_in);

            let one_plus_e = vaddq_f32(one, e);
            let recip = vdivq_f32(one, one_plus_e);
            let pos_branch = recip; // x >= 0 path: 1 / (1 + e^-x)
            let neg_branch = vmulq_f32(e, recip); // x < 0 path: e^x / (1 + e^x)
            let r = vbslq_f32(mask, pos_branch, neg_branch);
            vst1q_f32(ptr.add(i), r);
            i += 4;
        }

        // Scalar tail.
        while i < n {
            let x = *ptr.add(i);
            *ptr.add(i) = if x >= 0.0 {
                let e = (-x).exp();
                1.0 / (1.0 + e)
            } else {
                let e = x.exp();
                e / (1.0 + e)
            };
            i += 1;
        }
    }
}

// ────────────────────────────────────────────────────────────────────────
// NEON softmax f32
//
// Three passes: max reduction, subtract + exp + sum, normalize. An empty
// buffer is a no-op.
// ────────────────────────────────────────────────────────────────────────

/// NEON softmax f32, in-place.
///
/// # Safety
/// See `dequant_i8_to_f32_neon`.
#[inline]
#[target_feature(enable = "neon")]
pub unsafe fn softmax_inplace_f32_neon(buf: &mut [f32]) {
    unsafe {
        if buf.is_empty() {
            return;
        }
        let n = buf.len();
        let chunks_4 = n / 4;
        let ptr = buf.as_mut_ptr();

        // Pass 1: find max via NEON horizontal-max reduce + scalar tail.
        let mut m = f32::NEG_INFINITY;
        if chunks_4 > 0 {
            let mut max_v = vld1q_f32(ptr);
            let mut i = 4;
            for _ in 1..chunks_4 {
                let v = vld1q_f32(ptr.add(i));
                max_v = vmaxq_f32(max_v, v);
                i += 4;
            }
            m = vmaxvq_f32(max_v);
        }
        {
            let mut i = chunks_4 * 4;
            while i < n {
                let v = *ptr.add(i);
                if v > m {
                    m = v;
                }
                i += 1;
            }
        }

        // Pass 2: subtract max, exp via polynomial NEON expf, sum.
        let m_v = vdupq_n_f32(m);
        let mut sum_v = vdupq_n_f32(0.0);
        {
            let mut i = 0;
            for _ in 0..chunks_4 {
                let v = vld1q_f32(ptr.add(i));
                let s = vsubq_f32(v, m_v);
                let e = expf_neon_f32x4(s);
                sum_v = vaddq_f32(sum_v, e);
                vst1q_f32(ptr.add(i), e);
                i += 4;
            }
        }
        let mut sum = vaddvq_f32(sum_v);
        {
            let mut i = chunks_4 * 4;
            while i < n {
                let e = (*ptr.add(i) - m).exp();
                *ptr.add(i) = e;
                sum += e;
                i += 1;
            }
        }

        // Pass 3: normalize when the sum is positive.
        if sum > 0.0 {
            let inv = 1.0 / sum;
            let inv_v = vdupq_n_f32(inv);
            let mut i = 0;
            for _ in 0..chunks_4 {
                let v = vld1q_f32(ptr.add(i));
                let r = vmulq_f32(v, inv_v);
                vst1q_f32(ptr.add(i), r);
                i += 4;
            }
            let mut i = chunks_4 * 4;
            while i < n {
                *ptr.add(i) *= inv;
                i += 1;
            }
        }
    }
}

// ────────────────────────────────────────────────────────────────────────
// NEON FP16 polynomial expf, 8 lanes, via inline asm
//
// Steps:
//   1. Receive 8 f32 lanes (as 2× `float32x4_t`).
//   2. Clamp to [-10, 10] in f32.
//   3. Pack to a single 8-lane f16 register via `fcvtn` / `fcvtn2`.
//   4. Polynomial in f16:
//        - Range reduction `x = k*ln2 + r` via `fcvtns` (round-to-nearest
//          int) + `scvtf` (back to f16) + two `fmls`.
//        - 5-degree minimax (Cephes coefficients rounded to f16).
//        - 2^k via integer exponent injection: f16 has 5 exponent bits
//          and bias 15, so the bit pattern of 2^k is `(k + 15) << 10`.
//      Operates on `int16x8_t` / `uint16x8_t` carriers holding f16 bit
//      patterns.
//   5. Unpack to 2× `float32x4_t` via `fcvtl` / `fcvtl2`.
//
// Accuracy: ~1% relative error in `exp(r)`.
//
// Every fp16-using function is `unsafe` and requires
// `CpuFeatures::neon_fp16`.
// ────────────────────────────────────────────────────────────────────────

/// Pack 2× f32x4 into a single f16x8 (bit-equivalent uint16x8_t).
///
/// `lo` lanes go to the lower 4 f16 slots; `hi` lanes to the upper 4.
#[inline]
#[target_feature(enable = "neon")]
unsafe fn pack_f16x8_from_f32x4_pair(lo: float32x4_t, hi: float32x4_t) -> uint16x8_t {
    unsafe {
        let result: uint16x8_t;
        core::arch::asm!(
            ".arch_extension fp16",
            "fcvtn  {r:v}.4h, {lo:v}.4s",
            "fcvtn2 {r:v}.8h, {hi:v}.4s",
            r = out(vreg) result,
            lo = in(vreg) lo,
            hi = in(vreg) hi,
            options(pure, nomem, nostack),
        );
        result
    }
}

/// Unpack f16x8 to 2× f32x4 (lo + hi halves).
#[inline]
#[target_feature(enable = "neon")]
unsafe fn unpack_f16x8_to_f32x4_pair(p: uint16x8_t) -> (float32x4_t, float32x4_t) {
    unsafe {
        let lo: float32x4_t;
        let hi: float32x4_t;
        core::arch::asm!(
            ".arch_extension fp16",
            "fcvtl  {lo:v}.4s, {p:v}.4h",
            "fcvtl2 {hi:v}.4s, {p:v}.8h",
            lo = out(vreg) lo,
            hi = out(vreg) hi,
            p = in(vreg) p,
            options(pure, nomem, nostack),
        );
        (lo, hi)
    }
}

/// 8-lane FP16 polynomial expf. Takes 2× f32x4, returns 2× f32x4.
///
/// * Clamps inputs to [-10, 10] in f32.
/// * Packs to f16x8.
/// * Runs Cephes 5-degree minimax in f16.
/// * Reconstructs `2^k` via int16 exponent-bit injection.
/// * Unpacks back to an f32x4 pair.
///
/// # Safety
/// Caller must ensure ARMv8.2-A FP16 hardware; executing on hardware
/// without FP16 raises SIGILL.
#[inline]
#[target_feature(enable = "neon")]
pub unsafe fn expf_neon_f32x8_via_f16(
    lo: float32x4_t,
    hi: float32x4_t,
) -> (float32x4_t, float32x4_t) {
    unsafe {
        // Cephes constants as IEEE 754 binary16 bit patterns.
        const LOG2_E: u16 = 0x3DC5; // 1.4424
        const LN2_HI: u16 = 0x398C; // 0.693359375 (exact)
        const LN2_LO: u16 = 0x8AF4; // -2.122e-4
        const ONE: u16 = 0x3C00;
        const P0: u16 = 0x0A83; // 1.987e-4
        const P1: u16 = 0x15BA; // 1.398e-3
        const P2: u16 = 0x2044; // 8.331e-3
        const P3: u16 = 0x2955; // 4.166e-2
        const P4: u16 = 0x3155; // 1.666e-1
        const P5: u16 = 0x3800; // 0.5

        // Clamp inputs to [-10, 10] in f32 before conversion.
        let max_v = vdupq_n_f32(10.0);
        let min_v = vdupq_n_f32(-10.0);
        let lo_c = vminq_f32(vmaxq_f32(lo, min_v), max_v);
        let hi_c = vminq_f32(vmaxq_f32(hi, min_v), max_v);

        let x = pack_f16x8_from_f32x4_pair(lo_c, hi_c);

        // Constants splatted across all 8 lanes as f16 bit patterns.
        let log2_e = vdupq_n_u16(LOG2_E);
        let ln2_hi = vdupq_n_u16(LN2_HI);
        let ln2_lo = vdupq_n_u16(LN2_LO);
        let one = vdupq_n_u16(ONE);
        let p0 = vdupq_n_u16(P0);
        let p1 = vdupq_n_u16(P1);
        let p2 = vdupq_n_u16(P2);
        let p3 = vdupq_n_u16(P3);
        let p4 = vdupq_n_u16(P4);
        let p5 = vdupq_n_u16(P5);

        // fx = x * log2_e (8 lanes f16). Then k = round_to_nearest(fx).
        let fx_f = fmul_f16x8(x, log2_e);
        let k_int = fcvtns_s16_from_f16x8(fx_f);
        let k_f = scvtf_f16_from_s16x8(k_int);

        // r = x - k*ln2_hi - k*ln2_lo.
        let z = fmls_f16x8(x, k_f, ln2_hi);
        let r = fmls_f16x8(z, k_f, ln2_lo);

        // Horner: y = ((((p0*r + p1)*r + p2)*r + p3)*r + p4)*r + p5
        let mut y = fmla_f16x8(p1, p0, r);
        y = fmla_f16x8(p2, y, r);
        y = fmla_f16x8(p3, y, r);
        y = fmla_f16x8(p4, y, r);
        y = fmla_f16x8(p5, y, r);

        // exp(r) ≈ y * r² + r + 1
        let r2 = fmul_f16x8(r, r);
        let exp_r = fadd_f16x8(fadd_f16x8(fmul_f16x8(y, r2), r), one);

        // 2^k via IEEE 754 f16 exponent injection: bias = 15, mantissa = 10.
        // (k + 15) << 10 gives the bit pattern of 2^k in f16.
        let bias = vdupq_n_s16(15);
        let pow2k_bits = vshlq_n_s16::<10>(vaddq_s16(k_int, bias));
        let pow2k = vreinterpretq_u16_s16(pow2k_bits);

        let result = fmul_f16x8(exp_r, pow2k);

        unpack_f16x8_to_f32x4_pair(result)
    }
}

/// FP16 NEON sigmoid, in place. 8-lane chunks via the f16 path; scalar
/// tail for the remainder.
///
/// # Safety
/// Caller must ensure ARMv8.2-A FP16 hardware.
#[target_feature(enable = "neon")]
pub unsafe fn sigmoid_slice_f32_neon_fp16(buf: &mut [f32]) {
    unsafe {
        let n = buf.len();
        let chunks_8 = n / 8;
        let one_v = vdupq_n_f32(1.0);
        let zero = vdupq_n_f32(0.0);
        let ptr = buf.as_mut_ptr();

        for c in 0..chunks_8 {
            let i = c * 8;
            let lo = vld1q_f32(ptr.add(i));
            let hi = vld1q_f32(ptr.add(i + 4));

            // Branch select: x >= 0 → exp(-x); x < 0 → exp(x).
            let lo_mask = vcgeq_f32(lo, zero);
            let hi_mask = vcgeq_f32(hi, zero);
            let lo_in = vbslq_f32(lo_mask, vnegq_f32(lo), lo);
            let hi_in = vbslq_f32(hi_mask, vnegq_f32(hi), hi);

            let (e_lo, e_hi) = expf_neon_f32x8_via_f16(lo_in, hi_in);

            let recip_lo = vdivq_f32(one_v, vaddq_f32(one_v, e_lo));
            let recip_hi = vdivq_f32(one_v, vaddq_f32(one_v, e_hi));
            let neg_lo = vmulq_f32(e_lo, recip_lo);
            let neg_hi = vmulq_f32(e_hi, recip_hi);
            let r_lo = vbslq_f32(lo_mask, recip_lo, neg_lo);
            let r_hi = vbslq_f32(hi_mask, recip_hi, neg_hi);

            vst1q_f32(ptr.add(i), r_lo);
            vst1q_f32(ptr.add(i + 4), r_hi);
        }

        // Scalar tail for fewer than 8 remaining lanes.
        let tail_start = chunks_8 * 8;
        for i in tail_start..n {
            let x = *ptr.add(i);
            let r = if x >= 0.0 {
                1.0 / (1.0 + (-x).exp())
            } else {
                let e = x.exp();
                e / (1.0 + e)
            };
            *ptr.add(i) = r;
        }
    }
}

/// FP16 NEON softmax, in place. 8-lane chunks via f16 expf; scalar tail
/// for the remainder.
///
/// # Safety
/// Caller must ensure ARMv8.2-A FP16 hardware.
#[inline]
#[target_feature(enable = "neon")]
pub unsafe fn softmax_inplace_f32_neon_fp16(buf: &mut [f32]) {
    unsafe {
        if buf.is_empty() {
            return;
        }
        let n = buf.len();
        let chunks_8 = n / 8;
        let ptr = buf.as_mut_ptr();

        // Pass 1: max via NEON horizontal reduce in f32, 4-lane chunks.
        let chunks_4 = n / 4;
        let mut m = f32::NEG_INFINITY;
        if chunks_4 > 0 {
            let mut max_v = vld1q_f32(ptr);
            let mut i = 4;
            for _ in 1..chunks_4 {
                let v = vld1q_f32(ptr.add(i));
                max_v = vmaxq_f32(max_v, v);
                i += 4;
            }
            m = vmaxvq_f32(max_v);
        }
        {
            let mut i = chunks_4 * 4;
            while i < n {
                let v = *ptr.add(i);
                if v > m {
                    m = v;
                }
                i += 1;
            }
        }

        // Pass 2: subtract max, exp via 8-lane f16, sum.
        let m_v = vdupq_n_f32(m);
        let mut sum_v = vdupq_n_f32(0.0);
        for c in 0..chunks_8 {
            let i = c * 8;
            let lo = vld1q_f32(ptr.add(i));
            let hi = vld1q_f32(ptr.add(i + 4));
            let s_lo = vsubq_f32(lo, m_v);
            let s_hi = vsubq_f32(hi, m_v);
            let (e_lo, e_hi) = expf_neon_f32x8_via_f16(s_lo, s_hi);
            sum_v = vaddq_f32(sum_v, e_lo);
            sum_v = vaddq_f32(sum_v, e_hi);
            vst1q_f32(ptr.add(i), e_lo);
            vst1q_f32(ptr.add(i + 4), e_hi);
        }
        let mut sum = vaddvq_f32(sum_v);

        // Tail (< 8 lanes): scalar.
        let tail_start = chunks_8 * 8;
        for i in tail_start..n {
            let e = (*ptr.add(i) - m).exp();
            *ptr.add(i) = e;
            sum += e;
        }

        // Pass 3: normalise.
        if sum > 0.0 {
            let inv = 1.0 / sum;
            let inv_v = vdupq_n_f32(inv);
            for c in 0..chunks_8 {
                let i = c * 8;
                let lo = vld1q_f32(ptr.add(i));
                let hi = vld1q_f32(ptr.add(i + 4));
                vst1q_f32(ptr.add(i), vmulq_f32(lo, inv_v));
                vst1q_f32(ptr.add(i + 4), vmulq_f32(hi, inv_v));
            }
            for i in tail_start..n {
                *ptr.add(i) *= inv;
            }
        }
    }
}

// ────────────────────────────────────────────────────────────────────────
// NEON weighted-sum for DFL
// ────────────────────────────────────────────────────────────────────────

/// NEON weighted-sum for 4 DFL sides.
///
/// `probs.len() == 4 * reg_max`, side-major layout. Returns `[d_left,
/// d_top, d_right, d_bottom]`.
///
/// # Safety
/// Caller must ensure NEON is available.
#[inline]
#[target_feature(enable = "neon")]
pub unsafe fn weighted_sum_4sides_f32_neon(probs: &[f32], reg_max: usize) -> [f32; 4] {
    unsafe {
        debug_assert_eq!(probs.len(), 4 * reg_max);
        let mut d = [0.0_f32; 4];
        let chunks_4 = reg_max / 4;
        let ptr = probs.as_ptr();

        for (side, slot) in d.iter_mut().enumerate() {
            let base = side * reg_max;
            let mut acc = vdupq_n_f32(0.0);
            let mut bin_idx = 0usize;

            for _ in 0..chunks_4 {
                let p = vld1q_f32(ptr.add(base + bin_idx));
                // Bin indices as f32: [bin_idx, bin_idx+1, bin_idx+2, bin_idx+3]
                let bins = [
                    bin_idx as f32,
                    (bin_idx + 1) as f32,
                    (bin_idx + 2) as f32,
                    (bin_idx + 3) as f32,
                ];
                let bins_v = vld1q_f32(bins.as_ptr());
                acc = vfmaq_f32(acc, p, bins_v);
                bin_idx += 4;
            }

            // Horizontal sum of acc.
            *slot = vaddvq_f32(acc);

            // Scalar tail for reg_max not divisible by 4.
            while bin_idx < reg_max {
                *slot += *ptr.add(base + bin_idx) * (bin_idx as f32);
                bin_idx += 1;
            }
        }
        d
    }
}

// ────────────────────────────────────────────────────────────────────────
// NEON fused softmax + weighted_sum
//
// For each side:
//
//   result = Σ(exp(x[i] - max) × i) / Σ(exp(x[i] - max))
//
// in two passes:
//   Pass 1: find max (NEON vmaxq_f32 reduction)
//   Pass 2: exp(x-max) accumulated into sum + weighted_sum in-register
// ────────────────────────────────────────────────────────────────────────

/// Fused softmax + weighted_sum for 4 DFL sides, NEON f32.
///
/// `logits` is a dequantized buffer with layout `[side0..side3]`, each
/// side containing `reg_max` contiguous f32 values. Returns `[d_left,
/// d_top, d_right, d_bottom]`.
///
/// # Safety
/// Caller must ensure NEON is available.
#[inline]
#[target_feature(enable = "neon")]
pub unsafe fn softmax_weighted_sum_4sides_f32_neon(logits: &[f32], reg_max: usize) -> [f32; 4] {
    unsafe {
        debug_assert_eq!(logits.len(), 4 * reg_max);
        let mut d = [0.0_f32; 4];
        let chunks_4 = reg_max / 4;
        let ptr = logits.as_ptr();

        for (side, slot) in d.iter_mut().enumerate() {
            let base = side * reg_max;

            // ── Pass 1: find max ──
            let mut m = f32::NEG_INFINITY;
            if chunks_4 > 0 {
                let mut max_v = vld1q_f32(ptr.add(base));
                for c in 1..chunks_4 {
                    let v = vld1q_f32(ptr.add(base + c * 4));
                    max_v = vmaxq_f32(max_v, v);
                }
                m = vmaxvq_f32(max_v);
            }
            // Scalar tail for non-multiple-of-4 reg_max.
            let tail_start = chunks_4 * 4;
            for i in tail_start..reg_max {
                let v = *ptr.add(base + i);
                if v > m {
                    m = v;
                }
            }

            // ── Pass 2: fused exp + sum + weighted_sum ──
            let m_v = vdupq_n_f32(m);
            let mut sum_v = vdupq_n_f32(0.0);
            let mut ws_v = vdupq_n_f32(0.0);
            let mut bin_idx = 0usize;

            for _ in 0..chunks_4 {
                let v = vld1q_f32(ptr.add(base + bin_idx));
                let e = expf_neon_f32x4(vsubq_f32(v, m_v));
                sum_v = vaddq_f32(sum_v, e);
                // Bin indices [bin_idx, bin_idx+1, bin_idx+2, bin_idx+3]
                let bins = [
                    bin_idx as f32,
                    (bin_idx + 1) as f32,
                    (bin_idx + 2) as f32,
                    (bin_idx + 3) as f32,
                ];
                let bins_v = vld1q_f32(bins.as_ptr());
                ws_v = vfmaq_f32(ws_v, e, bins_v);
                bin_idx += 4;
            }

            let mut sum = vaddvq_f32(sum_v);
            let mut ws = vaddvq_f32(ws_v);

            // Scalar tail.
            while bin_idx < reg_max {
                let e = (*ptr.add(base + bin_idx) - m).exp();
                sum += e;
                ws += e * (bin_idx as f32);
                bin_idx += 1;
            }

            // One divide per side.
            *slot = if sum > 0.0 { ws / sum } else { 0.0 };
        }
        d
    }
}

/// Fused softmax + weighted_sum for 4 DFL sides, FP16 exp path.
///
/// Same semantics as `softmax_weighted_sum_4sides_f32_neon` with the
/// 8-lane FP16 exp approximation. Max-finding and accumulation are in f32.
///
/// # Safety
/// Caller must ensure FP16 NEON is available.
#[inline]
#[target_feature(enable = "neon")]
pub unsafe fn softmax_weighted_sum_4sides_f32_neon_fp16(
    logits: &[f32],
    reg_max: usize,
) -> [f32; 4] {
    unsafe {
        debug_assert_eq!(logits.len(), 4 * reg_max);
        let mut d = [0.0_f32; 4];
        let chunks_8 = reg_max / 8;
        let chunks_4 = reg_max / 4;
        let ptr = logits.as_ptr();

        for (side, slot) in d.iter_mut().enumerate() {
            let base = side * reg_max;

            // ── Pass 1: find max in f32 ──
            let mut m = f32::NEG_INFINITY;
            if chunks_4 > 0 {
                let mut max_v = vld1q_f32(ptr.add(base));
                for c in 1..chunks_4 {
                    let v = vld1q_f32(ptr.add(base + c * 4));
                    max_v = vmaxq_f32(max_v, v);
                }
                m = vmaxvq_f32(max_v);
            }
            let tail_start_4 = chunks_4 * 4;
            for i in tail_start_4..reg_max {
                let v = *ptr.add(base + i);
                if v > m {
                    m = v;
                }
            }

            // ── Pass 2: fused exp(fp16) + sum + weighted_sum ──
            let m_v = vdupq_n_f32(m);
            let mut sum_v = vdupq_n_f32(0.0);
            let mut ws_v = vdupq_n_f32(0.0);
            let mut bin_idx = 0usize;

            // Process 8 elements at a time via FP16 exp.
            for _ in 0..chunks_8 {
                let lo = vld1q_f32(ptr.add(base + bin_idx));
                let hi = vld1q_f32(ptr.add(base + bin_idx + 4));
                let s_lo = vsubq_f32(lo, m_v);
                let s_hi = vsubq_f32(hi, m_v);
                let (e_lo, e_hi) = expf_neon_f32x8_via_f16(s_lo, s_hi);

                sum_v = vaddq_f32(sum_v, e_lo);
                sum_v = vaddq_f32(sum_v, e_hi);

                let bins_lo = [
                    bin_idx as f32,
                    (bin_idx + 1) as f32,
                    (bin_idx + 2) as f32,
                    (bin_idx + 3) as f32,
                ];
                let bins_hi = [
                    (bin_idx + 4) as f32,
                    (bin_idx + 5) as f32,
                    (bin_idx + 6) as f32,
                    (bin_idx + 7) as f32,
                ];
                ws_v = vfmaq_f32(ws_v, e_lo, vld1q_f32(bins_lo.as_ptr()));
                ws_v = vfmaq_f32(ws_v, e_hi, vld1q_f32(bins_hi.as_ptr()));
                bin_idx += 8;
            }

            // Handle 4-element remainder (reg_max % 8 >= 4).
            if bin_idx + 4 <= reg_max {
                let v = vld1q_f32(ptr.add(base + bin_idx));
                let e = expf_neon_f32x4(vsubq_f32(v, m_v));
                sum_v = vaddq_f32(sum_v, e);
                let bins = [
                    bin_idx as f32,
                    (bin_idx + 1) as f32,
                    (bin_idx + 2) as f32,
                    (bin_idx + 3) as f32,
                ];
                ws_v = vfmaq_f32(ws_v, e, vld1q_f32(bins.as_ptr()));
                bin_idx += 4;
            }

            let mut sum = vaddvq_f32(sum_v);
            let mut ws = vaddvq_f32(ws_v);

            // Scalar tail.
            while bin_idx < reg_max {
                let e = (*ptr.add(base + bin_idx) - m).exp();
                sum += e;
                ws += e * (bin_idx as f32);
                bin_idx += 1;
            }

            *slot = if sum > 0.0 { ws / sum } else { 0.0 };
        }
        d
    }
}

// ────────────────────────────────────────────────────────────────────────
// NEON dist2bbox, 4 anchors per batch
// ────────────────────────────────────────────────────────────────────────

/// Batch dist2bbox for 4 anchors. `ltrb` is `[l0,t0,r0,b0, l1,t1,r1,b1,
/// ...]` (anchor-major). `gx`/`gy` are 4 grid centres. Output is
/// `[xc0,yc0,w0,h0, xc1,...,h3]` written to `dst`.
///
/// # Safety
/// Caller must ensure `ltrb.len() >= 16`, `gx.len() >= 4`,
/// `gy.len() >= 4`, `dst.len() >= 16`, and NEON is available.
#[inline]
#[target_feature(enable = "neon")]
pub unsafe fn dist2bbox_4anchors_f32_neon(
    ltrb: &[f32],
    gx: &[f32],
    gy: &[f32],
    stride: f32,
    dst: &mut [f32],
) {
    unsafe {
        debug_assert!(ltrb.len() >= 16);
        debug_assert!(gx.len() >= 4 && gy.len() >= 4);
        debug_assert!(dst.len() >= 16);

        let half = vdupq_n_f32(0.5);
        let stride_v = vdupq_n_f32(stride);

        // Load LTRB for 4 anchors, laid out as
        // [l0,t0,r0,b0, l1,t1,r1,b1, l2,t2,r2,b2, l3,t3,r3,b3], and
        // transpose to [l0,l1,l2,l3], [t0,...], [r0,...], [b0,...].
        let a0 = vld1q_f32(ltrb.as_ptr()); // l0,t0,r0,b0
        let a1 = vld1q_f32(ltrb.as_ptr().add(4)); // l1,t1,r1,b1
        let a2 = vld1q_f32(ltrb.as_ptr().add(8)); // l2,t2,r2,b2
        let a3 = vld1q_f32(ltrb.as_ptr().add(12)); // l3,t3,r3,b3

        // 4×4 transpose using TRN + ZIP.
        let t01_lo = vtrn1q_f32(a0, a1); // l0,l1,r0,r1
        let t01_hi = vtrn2q_f32(a0, a1); // t0,t1,b0,b1
        let t23_lo = vtrn1q_f32(a2, a3); // l2,l3,r2,r3
        let t23_hi = vtrn2q_f32(a2, a3); // t2,t3,b2,b3

        // Interleave 64-bit halves to finish the transpose.
        let d_left = vreinterpretq_f32_f64(vtrn1q_f64(
            vreinterpretq_f64_f32(t01_lo),
            vreinterpretq_f64_f32(t23_lo),
        )); // l0,l1,l2,l3
        let d_top = vreinterpretq_f32_f64(vtrn1q_f64(
            vreinterpretq_f64_f32(t01_hi),
            vreinterpretq_f64_f32(t23_hi),
        )); // t0,t1,t2,t3
        let d_right = vreinterpretq_f32_f64(vtrn2q_f64(
            vreinterpretq_f64_f32(t01_lo),
            vreinterpretq_f64_f32(t23_lo),
        )); // r0,r1,r2,r3
        let d_bottom = vreinterpretq_f32_f64(vtrn2q_f64(
            vreinterpretq_f64_f32(t01_hi),
            vreinterpretq_f64_f32(t23_hi),
        )); // b0,b1,b2,b3

        let gx_v = vld1q_f32(gx.as_ptr());
        let gy_v = vld1q_f32(gy.as_ptr());

        // xc = (gx + (right - left) * 0.5) * stride
        let xc = vmulq_f32(vfmaq_f32(gx_v, vsubq_f32(d_right, d_left), half), stride_v);
        // yc = (gy + (bottom - top) * 0.5) * stride
        let yc = vmulq_f32(vfmaq_f32(gy_v, vsubq_f32(d_bottom, d_top), half), stride_v);
        // w = (left + right) * stride
        let w = vmulq_f32(vaddq_f32(d_left, d_right), stride_v);
        // h = (top + bottom) * stride
        let h = vmulq_f32(vaddq_f32(d_top, d_bottom), stride_v);

        // Transpose back to [xc0,yc0,w0,h0, xc1,...].
        let r01_lo = vtrn1q_f32(xc, yc); // xc0,yc0,xc2,yc2
        let r01_hi = vtrn2q_f32(xc, yc); // xc1,yc1,xc3,yc3
        let r23_lo = vtrn1q_f32(w, h); // w0,h0,w2,h2
        let r23_hi = vtrn2q_f32(w, h); // w1,h1,w3,h3

        let out0 = vreinterpretq_f32_f64(vtrn1q_f64(
            vreinterpretq_f64_f32(r01_lo),
            vreinterpretq_f64_f32(r23_lo),
        )); // xc0,yc0,w0,h0
        let out1 = vreinterpretq_f32_f64(vtrn1q_f64(
            vreinterpretq_f64_f32(r01_hi),
            vreinterpretq_f64_f32(r23_hi),
        )); // xc1,yc1,w1,h1
        let out2 = vreinterpretq_f32_f64(vtrn2q_f64(
            vreinterpretq_f64_f32(r01_lo),
            vreinterpretq_f64_f32(r23_lo),
        )); // xc2,yc2,w2,h2
        let out3 = vreinterpretq_f32_f64(vtrn2q_f64(
            vreinterpretq_f64_f32(r01_hi),
            vreinterpretq_f64_f32(r23_hi),
        )); // xc3,yc3,w3,h3

        vst1q_f32(dst.as_mut_ptr(), out0);
        vst1q_f32(dst.as_mut_ptr().add(4), out1);
        vst1q_f32(dst.as_mut_ptr().add(8), out2);
        vst1q_f32(dst.as_mut_ptr().add(12), out3);
    }
}

// ────────────────────────────────────────────────────────────────────────
// Fused dequant + sigmoid in one pass
// ────────────────────────────────────────────────────────────────────────

/// Fused NEON i8 dequant + sigmoid. Output is sigmoid((in - zp) * scale).
///
/// # Safety
/// See `dequant_i8_to_f32_neon`.
#[inline]
#[target_feature(enable = "neon")]
pub unsafe fn dequant_sigmoid_i8_to_f32_neon(input: &[i8], q: Quant, output: &mut [f32]) {
    unsafe {
        debug_assert_eq!(input.len(), output.len());
        let scale = q.scale;
        let bias_v = vdupq_n_f32(affine_bias(q));
        let zero = vdupq_n_f32(0.0);
        let one = vdupq_n_f32(1.0);
        let n = input.len();
        let chunks_16 = n / 16;
        let mut i = 0usize;
        let in_ptr = input.as_ptr();
        let out_ptr = output.as_mut_ptr();

        for _ in 0..chunks_16 {
            let v_i8 = vld1q_s8(in_ptr.add(i));
            let lo_i16 = vmovl_s8(vget_low_s8(v_i8));
            let hi_i16 = vmovl_high_s8(v_i8);
            let q0 = vmovl_s16(vget_low_s16(lo_i16));
            let q1 = vmovl_high_s16(lo_i16);
            let q2 = vmovl_s16(vget_low_s16(hi_i16));
            let q3 = vmovl_high_s16(hi_i16);

            // Dequant: bias + scale * in
            let x0 = vfmaq_n_f32(bias_v, vcvtq_f32_s32(q0), scale);
            let x1 = vfmaq_n_f32(bias_v, vcvtq_f32_s32(q1), scale);
            let x2 = vfmaq_n_f32(bias_v, vcvtq_f32_s32(q2), scale);
            let x3 = vfmaq_n_f32(bias_v, vcvtq_f32_s32(q3), scale);

            // Inline sigmoid for each quad.
            macro_rules! sigmoid4 {
                ($x:expr) => {{
                    let mask = vcgeq_f32($x, zero);
                    let neg_x = vnegq_f32($x);
                    let exp_in = vbslq_f32(mask, neg_x, $x);
                    let e = expf_neon_f32x4(exp_in);
                    let one_plus_e = vaddq_f32(one, e);
                    let recip = vdivq_f32(one, one_plus_e);
                    let neg_branch = vmulq_f32(e, recip);
                    vbslq_f32(mask, recip, neg_branch)
                }};
            }

            vst1q_f32(out_ptr.add(i), sigmoid4!(x0));
            vst1q_f32(out_ptr.add(i + 4), sigmoid4!(x1));
            vst1q_f32(out_ptr.add(i + 8), sigmoid4!(x2));
            vst1q_f32(out_ptr.add(i + 12), sigmoid4!(x3));
            i += 16;
        }

        // Scalar tail.
        let zp = q.zero_point as f32;
        while i < n {
            let x = (*in_ptr.add(i) as f32 - zp) * scale;
            *out_ptr.add(i) = if x >= 0.0 {
                let e = (-x).exp();
                1.0 / (1.0 + e)
            } else {
                let e = x.exp();
                e / (1.0 + e)
            };
            i += 1;
        }
    }
}

/// Fused NEON u8 dequant + sigmoid.
///
/// # Safety
/// See `dequant_i8_to_f32_neon`.
#[inline]
#[target_feature(enable = "neon")]
pub unsafe fn dequant_sigmoid_u8_to_f32_neon(input: &[u8], q: Quant, output: &mut [f32]) {
    unsafe {
        debug_assert_eq!(input.len(), output.len());
        let scale = q.scale;
        let bias_v = vdupq_n_f32(affine_bias(q));
        let zero = vdupq_n_f32(0.0);
        let one = vdupq_n_f32(1.0);
        let n = input.len();
        let chunks_16 = n / 16;
        let mut i = 0usize;
        let in_ptr = input.as_ptr();
        let out_ptr = output.as_mut_ptr();

        for _ in 0..chunks_16 {
            let v_u8 = vld1q_u8(in_ptr.add(i));
            let lo_u16 = vmovl_u8(vget_low_u8(v_u8));
            let hi_u16 = vmovl_high_u8(v_u8);
            let q0 = vmovl_u16(vget_low_u16(lo_u16));
            let q1 = vmovl_high_u16(lo_u16);
            let q2 = vmovl_u16(vget_low_u16(hi_u16));
            let q3 = vmovl_high_u16(hi_u16);

            let x0 = vfmaq_n_f32(bias_v, vcvtq_f32_u32(q0), scale);
            let x1 = vfmaq_n_f32(bias_v, vcvtq_f32_u32(q1), scale);
            let x2 = vfmaq_n_f32(bias_v, vcvtq_f32_u32(q2), scale);
            let x3 = vfmaq_n_f32(bias_v, vcvtq_f32_u32(q3), scale);

            macro_rules! sigmoid4 {
                ($x:expr) => {{
                    let mask = vcgeq_f32($x, zero);
                    let neg_x = vnegq_f32($x);
                    let exp_in = vbslq_f32(mask, neg_x, $x);
                    let e = expf_neon_f32x4(exp_in);
                    let one_plus_e = vaddq_f32(one, e);
                    let recip = vdivq_f32(one, one_plus_e);
                    let neg_branch = vmulq_f32(e, recip);
                    vbslq_f32(mask, recip, neg_branch)
                }};
            }

            vst1q_f32(out_ptr.add(i), sigmoid4!(x0));
            vst1q_f32(out_ptr.add(i + 4), sigmoid4!(x1));
            vst1q_f32(out_ptr.add(i + 8), sigmoid4!(x2));
            vst1q_f32(out_ptr.add(i + 12), sigmoid4!(x3));
            i += 16;
        }

        let zp = q.zero_point as f32;
        while i < n {
            let x = (*in_ptr.add(i) as f32 - zp) * scale;
            *out_ptr.add(i) = if x >= 0.0 {
                let e = (-x).exp();
                1.0 / (1.0 + e)
            } else {
                let e = x.exp();
                e / (1.0 + e)
            };
            i += 1;
        }
    }
}

/// Fused NEON i16 dequant + sigmoid.
///
/// # Safety
/// See `dequant_i8_to_f32_neon`.
#[inline]
#[target_feature(enable = "neon")]
pub unsafe fn dequant_sigmoid_i16_to_f32_neon(input: &[i16], q: Quant, output: &mut [f32]) {
    unsafe {
        debug_assert_eq!(input.len(), output.len());
        let scale = q.scale;
        let bias_v = vdupq_n_f32(affine_bias(q));
        let zero = vdupq_n_f32(0.0);
        let one = vdupq_n_f32(1.0);
        let n = input.len();
        let chunks_8 = n / 8;
        let mut i = 0usize;
        let in_ptr = input.as_ptr();
        let out_ptr = output.as_mut_ptr();

        for _ in 0..chunks_8 {
            let v_i16 = vld1q_s16(in_ptr.add(i));
            let lo_i32 = vmovl_s16(vget_low_s16(v_i16));
            let hi_i32 = vmovl_high_s16(v_i16);
            let x0 = vfmaq_n_f32(bias_v, vcvtq_f32_s32(lo_i32), scale);
            let x1 = vfmaq_n_f32(bias_v, vcvtq_f32_s32(hi_i32), scale);

            macro_rules! sigmoid4 {
                ($x:expr) => {{
                    let mask = vcgeq_f32($x, zero);
                    let neg_x = vnegq_f32($x);
                    let exp_in = vbslq_f32(mask, neg_x, $x);
                    let e = expf_neon_f32x4(exp_in);
                    let one_plus_e = vaddq_f32(one, e);
                    let recip = vdivq_f32(one, one_plus_e);
                    let neg_branch = vmulq_f32(e, recip);
                    vbslq_f32(mask, recip, neg_branch)
                }};
            }

            vst1q_f32(out_ptr.add(i), sigmoid4!(x0));
            vst1q_f32(out_ptr.add(i + 4), sigmoid4!(x1));
            i += 8;
        }

        let zp = q.zero_point as f32;
        while i < n {
            let x = (*in_ptr.add(i) as f32 - zp) * scale;
            *out_ptr.add(i) = if x >= 0.0 {
                let e = (-x).exp();
                1.0 / (1.0 + e)
            } else {
                let e = x.exp();
                e / (1.0 + e)
            };
            i += 1;
        }
    }
}

/// Fused NEON u16 dequant + sigmoid.
///
/// # Safety
/// See `dequant_i8_to_f32_neon`.
#[inline]
#[target_feature(enable = "neon")]
pub unsafe fn dequant_sigmoid_u16_to_f32_neon(input: &[u16], q: Quant, output: &mut [f32]) {
    unsafe {
        debug_assert_eq!(input.len(), output.len());
        let scale = q.scale;
        let bias_v = vdupq_n_f32(affine_bias(q));
        let zero = vdupq_n_f32(0.0);
        let one = vdupq_n_f32(1.0);
        let n = input.len();
        let chunks_8 = n / 8;
        let mut i = 0usize;
        let in_ptr = input.as_ptr();
        let out_ptr = output.as_mut_ptr();

        for _ in 0..chunks_8 {
            let v_u16 = vld1q_u16(in_ptr.add(i));
            let lo_u32 = vmovl_u16(vget_low_u16(v_u16));
            let hi_u32 = vmovl_high_u16(v_u16);
            let x0 = vfmaq_n_f32(bias_v, vcvtq_f32_u32(lo_u32), scale);
            let x1 = vfmaq_n_f32(bias_v, vcvtq_f32_u32(hi_u32), scale);

            macro_rules! sigmoid4 {
                ($x:expr) => {{
                    let mask = vcgeq_f32($x, zero);
                    let neg_x = vnegq_f32($x);
                    let exp_in = vbslq_f32(mask, neg_x, $x);
                    let e = expf_neon_f32x4(exp_in);
                    let one_plus_e = vaddq_f32(one, e);
                    let recip = vdivq_f32(one, one_plus_e);
                    let neg_branch = vmulq_f32(e, recip);
                    vbslq_f32(mask, recip, neg_branch)
                }};
            }

            vst1q_f32(out_ptr.add(i), sigmoid4!(x0));
            vst1q_f32(out_ptr.add(i + 4), sigmoid4!(x1));
            i += 8;
        }

        let zp = q.zero_point as f32;
        while i < n {
            let x = (*in_ptr.add(i) as f32 - zp) * scale;
            *out_ptr.add(i) = if x >= 0.0 {
                let e = (-x).exp();
                1.0 / (1.0 + e)
            } else {
                let e = x.exp();
                e / (1.0 + e)
            };
            i += 1;
        }
    }
}

// ────────────────────────────────────────────────────────────────────────
// Half-precision dequant output
//
// The dequant math runs in f32 and narrows to f16 on store via `fcvtn`.
// ────────────────────────────────────────────────────────────────────────

/// Convert 4 f32 lanes to 4 f16 lanes (lower half of a uint16x8_t).
///
/// # Safety
/// Caller must ensure NEON is available.
#[inline]
#[target_feature(enable = "neon")]
unsafe fn fcvtn_f16x4_from_f32x4(x: float32x4_t) -> uint16x4_t {
    unsafe {
        let result: uint16x4_t;
        core::arch::asm!(
            ".arch_extension fp16",
            "fcvtn {r:v}.4h, {x:v}.4s",
            r = out(vreg) result,
            x = in(vreg) x,
            options(pure, nomem, nostack),
        );
        result
    }
}

/// NEON i8 → f16 dequant. Dequantizes in f32, narrows to f16 on store.
///
/// # Safety
/// See `dequant_i8_to_f32_neon`. Additionally requires FP16 for `fcvtn`.
#[target_feature(enable = "neon")]
pub unsafe fn dequant_i8_to_f16_neon(input: &[i8], q: Quant, output: &mut [u16]) {
    unsafe {
        debug_assert_eq!(input.len(), output.len());
        let scale = q.scale;
        let bias_v = vdupq_n_f32(affine_bias(q));
        let n = input.len();
        let chunks_16 = n / 16;
        let mut i = 0usize;
        let in_ptr = input.as_ptr();
        let out_ptr = output.as_mut_ptr();

        for _ in 0..chunks_16 {
            let v_i8 = vld1q_s8(in_ptr.add(i));
            let lo_i16 = vmovl_s8(vget_low_s8(v_i8));
            let hi_i16 = vmovl_high_s8(v_i8);
            let q0 = vmovl_s16(vget_low_s16(lo_i16));
            let q1 = vmovl_high_s16(lo_i16);
            let q2 = vmovl_s16(vget_low_s16(hi_i16));
            let q3 = vmovl_high_s16(hi_i16);
            let f0 = vfmaq_n_f32(bias_v, vcvtq_f32_s32(q0), scale);
            let f1 = vfmaq_n_f32(bias_v, vcvtq_f32_s32(q1), scale);
            let f2 = vfmaq_n_f32(bias_v, vcvtq_f32_s32(q2), scale);
            let f3 = vfmaq_n_f32(bias_v, vcvtq_f32_s32(q3), scale);
            // Narrow to f16 and store.
            vst1_u16(out_ptr.add(i), fcvtn_f16x4_from_f32x4(f0));
            vst1_u16(out_ptr.add(i + 4), fcvtn_f16x4_from_f32x4(f1));
            vst1_u16(out_ptr.add(i + 8), fcvtn_f16x4_from_f32x4(f2));
            vst1_u16(out_ptr.add(i + 12), fcvtn_f16x4_from_f32x4(f3));
            i += 16;
        }

        let zp = q.zero_point as f32;
        while i < n {
            let val = (*in_ptr.add(i) as f32 - zp) * scale;
            *out_ptr.add(i) = half::f16::from_f32(val).to_bits();
            i += 1;
        }
    }
}

/// NEON u8 → f16 dequant.
///
/// # Safety
/// See `dequant_i8_to_f16_neon`.
#[target_feature(enable = "neon")]
pub unsafe fn dequant_u8_to_f16_neon(input: &[u8], q: Quant, output: &mut [u16]) {
    unsafe {
        debug_assert_eq!(input.len(), output.len());
        let scale = q.scale;
        let bias_v = vdupq_n_f32(affine_bias(q));
        let n = input.len();
        let chunks_16 = n / 16;
        let mut i = 0usize;
        let in_ptr = input.as_ptr();
        let out_ptr = output.as_mut_ptr();

        for _ in 0..chunks_16 {
            let v_u8 = vld1q_u8(in_ptr.add(i));
            let lo_u16 = vmovl_u8(vget_low_u8(v_u8));
            let hi_u16 = vmovl_high_u8(v_u8);
            let q0 = vmovl_u16(vget_low_u16(lo_u16));
            let q1 = vmovl_high_u16(lo_u16);
            let q2 = vmovl_u16(vget_low_u16(hi_u16));
            let q3 = vmovl_high_u16(hi_u16);
            let f0 = vfmaq_n_f32(bias_v, vcvtq_f32_u32(q0), scale);
            let f1 = vfmaq_n_f32(bias_v, vcvtq_f32_u32(q1), scale);
            let f2 = vfmaq_n_f32(bias_v, vcvtq_f32_u32(q2), scale);
            let f3 = vfmaq_n_f32(bias_v, vcvtq_f32_u32(q3), scale);
            vst1_u16(out_ptr.add(i), fcvtn_f16x4_from_f32x4(f0));
            vst1_u16(out_ptr.add(i + 4), fcvtn_f16x4_from_f32x4(f1));
            vst1_u16(out_ptr.add(i + 8), fcvtn_f16x4_from_f32x4(f2));
            vst1_u16(out_ptr.add(i + 12), fcvtn_f16x4_from_f32x4(f3));
            i += 16;
        }

        let zp = q.zero_point as f32;
        while i < n {
            let val = (*in_ptr.add(i) as f32 - zp) * scale;
            *out_ptr.add(i) = half::f16::from_f32(val).to_bits();
            i += 1;
        }
    }
}

/// NEON i16 → f16 dequant.
///
/// # Safety
/// See `dequant_i8_to_f16_neon`.
#[target_feature(enable = "neon")]
pub unsafe fn dequant_i16_to_f16_neon(input: &[i16], q: Quant, output: &mut [u16]) {
    unsafe {
        debug_assert_eq!(input.len(), output.len());
        let scale = q.scale;
        let bias_v = vdupq_n_f32(affine_bias(q));
        let n = input.len();
        let chunks_8 = n / 8;
        let mut i = 0usize;
        let in_ptr = input.as_ptr();
        let out_ptr = output.as_mut_ptr();

        for _ in 0..chunks_8 {
            let v_i16 = vld1q_s16(in_ptr.add(i));
            let lo_i32 = vmovl_s16(vget_low_s16(v_i16));
            let hi_i32 = vmovl_high_s16(v_i16);
            let f0 = vfmaq_n_f32(bias_v, vcvtq_f32_s32(lo_i32), scale);
            let f1 = vfmaq_n_f32(bias_v, vcvtq_f32_s32(hi_i32), scale);
            vst1_u16(out_ptr.add(i), fcvtn_f16x4_from_f32x4(f0));
            vst1_u16(out_ptr.add(i + 4), fcvtn_f16x4_from_f32x4(f1));
            i += 8;
        }

        let zp = q.zero_point as f32;
        while i < n {
            let val = (*in_ptr.add(i) as f32 - zp) * scale;
            *out_ptr.add(i) = half::f16::from_f32(val).to_bits();
            i += 1;
        }
    }
}

/// NEON u16 → f16 dequant.
///
/// # Safety
/// See `dequant_i8_to_f16_neon`.
#[target_feature(enable = "neon")]
pub unsafe fn dequant_u16_to_f16_neon(input: &[u16], q: Quant, output: &mut [u16]) {
    unsafe {
        debug_assert_eq!(input.len(), output.len());
        let scale = q.scale;
        let bias_v = vdupq_n_f32(affine_bias(q));
        let n = input.len();
        let chunks_8 = n / 8;
        let mut i = 0usize;
        let in_ptr = input.as_ptr();
        let out_ptr = output.as_mut_ptr();

        for _ in 0..chunks_8 {
            let v_u16 = vld1q_u16(in_ptr.add(i));
            let lo_u32 = vmovl_u16(vget_low_u16(v_u16));
            let hi_u32 = vmovl_high_u16(v_u16);
            let f0 = vfmaq_n_f32(bias_v, vcvtq_f32_u32(lo_u32), scale);
            let f1 = vfmaq_n_f32(bias_v, vcvtq_f32_u32(hi_u32), scale);
            vst1_u16(out_ptr.add(i), fcvtn_f16x4_from_f32x4(f0));
            vst1_u16(out_ptr.add(i + 4), fcvtn_f16x4_from_f32x4(f1));
            i += 8;
        }

        let zp = q.zero_point as f32;
        while i < n {
            let val = (*in_ptr.add(i) as f32 - zp) * scale;
            *out_ptr.add(i) = half::f16::from_f32(val).to_bits();
            i += 1;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::prim::dequant::{
        dequant_i16_to_f32, dequant_i8_to_f32, dequant_u16_to_f32, dequant_u8_to_f32,
    };
    use crate::prim::sigmoid::sigmoid_slice_f32;
    use crate::prim::softmax::softmax_inplace_f32;

    /// Whether `a` and `b` differ by less than 1e-7 or by less than `ulps` ulp.
    fn close_within_ulp(a: f32, b: f32, ulps: u32) -> bool {
        let diff = (a - b).abs();
        if diff < 1e-7 {
            return true;
        }
        let scale_max = a.abs().max(b.abs());
        diff < f32::EPSILON * scale_max * (ulps as f32)
    }

    #[test]
    fn dequant_i8_to_f32_neon_matches_scalar() {
        // 200 elements: twelve 16-lane chunks plus an 8-element scalar tail.
        let input: Vec<i8> = (-100..100).map(|i| i as i8).collect();
        let q = Quant {
            scale: 0.0326_f32,
            zero_point: -42,
        };
        let mut neon_out = vec![0f32; input.len()];
        let mut scalar_out = vec![0f32; input.len()];

        unsafe {
            dequant_i8_to_f32_neon(&input, q, &mut neon_out);
        }
        dequant_i8_to_f32(&input, q, &mut scalar_out);

        for (i, (&n_v, &s_v)) in neon_out.iter().zip(scalar_out.iter()).enumerate() {
            assert!(
                close_within_ulp(n_v, s_v, 2),
                "i8 dequant NEON/scalar mismatch at {i}: neon={n_v} scalar={s_v}"
            );
        }
    }

    #[test]
    fn dequant_u8_to_f32_neon_matches_scalar() {
        let input: Vec<u8> = (0..240).map(|i| i as u8).collect();
        let q = Quant {
            scale: 0.00392_f32,
            zero_point: 128,
        };
        let mut neon_out = vec![0f32; input.len()];
        let mut scalar_out = vec![0f32; input.len()];
        unsafe {
            dequant_u8_to_f32_neon(&input, q, &mut neon_out);
        }
        dequant_u8_to_f32(&input, q, &mut scalar_out);
        for (i, (&n_v, &s_v)) in neon_out.iter().zip(scalar_out.iter()).enumerate() {
            assert!(
                close_within_ulp(n_v, s_v, 2),
                "u8 dequant NEON/scalar mismatch at {i}: neon={n_v} scalar={s_v}"
            );
        }
    }

    #[test]
    fn dequant_i16_to_f32_neon_matches_scalar() {
        let input: Vec<i16> = (-1000..1000).step_by(7).collect();
        let q = Quant {
            scale: 0.0001_f32,
            zero_point: 0,
        };
        let mut neon_out = vec![0f32; input.len()];
        let mut scalar_out = vec![0f32; input.len()];
        unsafe {
            dequant_i16_to_f32_neon(&input, q, &mut neon_out);
        }
        dequant_i16_to_f32(&input, q, &mut scalar_out);
        for (i, (&n_v, &s_v)) in neon_out.iter().zip(scalar_out.iter()).enumerate() {
            assert!(
                close_within_ulp(n_v, s_v, 2),
                "i16 dequant NEON/scalar mismatch at {i}: neon={n_v} scalar={s_v}"
            );
        }
    }

    #[test]
    fn dequant_u16_to_f32_neon_matches_scalar() {
        let input: Vec<u16> = (0..2000).step_by(11).collect();
        let q = Quant {
            scale: 0.0001_f32,
            zero_point: 1024,
        };
        let mut neon_out = vec![0f32; input.len()];
        let mut scalar_out = vec![0f32; input.len()];
        unsafe {
            dequant_u16_to_f32_neon(&input, q, &mut neon_out);
        }
        dequant_u16_to_f32(&input, q, &mut scalar_out);
        for (i, (&n_v, &s_v)) in neon_out.iter().zip(scalar_out.iter()).enumerate() {
            assert!(
                close_within_ulp(n_v, s_v, 2),
                "u16 dequant NEON/scalar mismatch at {i}: neon={n_v} scalar={s_v}"
            );
        }
    }

    #[test]
    fn dequant_i8_to_f32_neon_handles_short_input_under_chunk_size() {
        // Length 12 < 16: only the scalar tail runs.
        let input: [i8; 12] = [-128, -64, -32, -1, 0, 1, 32, 64, 127, 50, -50, 25];
        let q = Quant {
            scale: 0.5_f32,
            zero_point: 0,
        };
        let mut neon_out = [0f32; 12];
        let mut scalar_out = [0f32; 12];
        unsafe {
            dequant_i8_to_f32_neon(&input, q, &mut neon_out);
        }
        dequant_i8_to_f32(&input, q, &mut scalar_out);
        assert_eq!(neon_out, scalar_out, "tail-only path must be exact");
    }

    #[test]
    fn expf_neon_f32x4_matches_libm() {
        let cases: Vec<f32> = (-100..=100).map(|i| i as f32 * 0.5).collect();
        for chunk in cases.chunks(4) {
            if chunk.len() < 4 {
                continue;
            }
            let arr = [chunk[0], chunk[1], chunk[2], chunk[3]];
            let neon = unsafe {
                let v = vld1q_f32(arr.as_ptr());
                let r = expf_neon_f32x4(v);
                let mut out = [0f32; 4];
                vst1q_f32(out.as_mut_ptr(), r);
                out
            };
            for (i, &x) in arr.iter().enumerate() {
                let oracle = x.exp();
                assert!(
                    close_within_ulp(neon[i], oracle, 8),
                    "expf NEON/libm mismatch at x={x}: neon={} libm={oracle}",
                    neon[i]
                );
            }
        }
    }

    #[test]
    fn sigmoid_slice_f32_neon_matches_scalar() {
        // Covers both branches of the vbslq_f32 blend and the saturation tails.
        let cases: Vec<f32> = (-50..=50).map(|i| i as f32 * 0.5).collect();
        let mut neon_buf = cases.clone();
        let mut scalar_buf = cases;
        unsafe {
            sigmoid_slice_f32_neon(&mut neon_buf);
        }
        sigmoid_slice_f32(&mut scalar_buf);
        for (i, (&n_v, &s_v)) in neon_buf.iter().zip(scalar_buf.iter()).enumerate() {
            let abs_diff = (n_v - s_v).abs();
            assert!(
                abs_diff < 1e-6 || close_within_ulp(n_v, s_v, 16),
                "sigmoid NEON/scalar mismatch at {i}: neon={n_v} scalar={s_v} diff={abs_diff}"
            );
        }
    }

    #[test]
    fn sigmoid_slice_f32_neon_short_input_under_chunk_size() {
        // Length 3: only the scalar tail runs.
        let mut neon_buf = [-2.0_f32, -1.0, 1.5];
        let mut scalar_buf = neon_buf;
        unsafe {
            sigmoid_slice_f32_neon(&mut neon_buf);
        }
        sigmoid_slice_f32(&mut scalar_buf);
        for (i, (&n_v, &s_v)) in neon_buf.iter().zip(scalar_buf.iter()).enumerate() {
            assert!(
                close_within_ulp(n_v, s_v, 1),
                "sigmoid tail-only mismatch at {i}: neon={n_v} scalar={s_v}"
            );
        }
    }

    #[test]
    fn sigmoid_slice_f32_neon_zero_is_half() {
        let mut buf = [0.0_f32; 4];
        unsafe {
            sigmoid_slice_f32_neon(&mut buf);
        }
        for &v in &buf {
            assert!((v - 0.5).abs() < 1e-7, "sigmoid(0) = {v}, expected 0.5");
        }
    }

    /// Whether `a` and `b` differ by less than 1e-5 or by less than 32 ulp.
    fn softmax_close(a: f32, b: f32) -> bool {
        let abs = (a - b).abs();
        abs < 1e-5 || close_within_ulp(a, b, 32)
    }

    #[test]
    fn softmax_neon_matches_scalar_reg_max_16() {
        // reg_max = 16: exactly 4 NEON chunks, no tail.
        let cases: Vec<f32> = (0..16).map(|i| (i as f32) * 0.5 - 4.0).collect();
        let mut neon_buf = cases.clone();
        let mut scalar_buf = cases;
        unsafe {
            softmax_inplace_f32_neon(&mut neon_buf);
        }
        softmax_inplace_f32(&mut scalar_buf);
        for (i, (&n_v, &s_v)) in neon_buf.iter().zip(scalar_buf.iter()).enumerate() {
            assert!(
                softmax_close(n_v, s_v),
                "softmax NEON/scalar mismatch at {i}: neon={n_v} scalar={s_v}"
            );
        }
        let sum: f32 = neon_buf.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5, "softmax sum != 1.0: {sum}");
    }

    #[test]
    fn softmax_neon_matches_scalar_with_tail() {
        // 18 elements: 4 chunks of 4 plus a 2-element tail.
        let cases: Vec<f32> = (0..18).map(|i| i as f32 * 0.3).collect();
        let mut neon_buf = cases.clone();
        let mut scalar_buf = cases;
        unsafe {
            softmax_inplace_f32_neon(&mut neon_buf);
        }
        softmax_inplace_f32(&mut scalar_buf);
        for (i, (&n_v, &s_v)) in neon_buf.iter().zip(scalar_buf.iter()).enumerate() {
            assert!(
                softmax_close(n_v, s_v),
                "softmax tail mismatch at {i}: neon={n_v} scalar={s_v}"
            );
        }
    }

    #[test]
    fn softmax_neon_overflow_safety() {
        let mut neon_buf = [1000.0_f32, 1001.0, 1002.0, 1003.0];
        let mut scalar_buf = neon_buf;
        unsafe {
            softmax_inplace_f32_neon(&mut neon_buf);
        }
        softmax_inplace_f32(&mut scalar_buf);
        for (i, (&n_v, &s_v)) in neon_buf.iter().zip(scalar_buf.iter()).enumerate() {
            assert!(
                softmax_close(n_v, s_v),
                "softmax overflow test mismatch at {i}: neon={n_v} scalar={s_v}"
            );
        }
        let sum: f32 = neon_buf.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);
    }

    #[test]
    fn softmax_neon_empty_is_noop() {
        let mut buf: [f32; 0] = [];
        unsafe {
            softmax_inplace_f32_neon(&mut buf);
        }
    }

    #[test]
    fn dequant_i8_to_f32_neon_handles_exact_chunk_boundary() {
        // Length 64: exactly 4 chunks of 16, no tail.
        let input: Vec<i8> = (-32..32).map(|i| i as i8).collect();
        let q = Quant {
            scale: 0.1_f32,
            zero_point: -10,
        };
        let mut neon_out = vec![0f32; 64];
        let mut scalar_out = vec![0f32; 64];
        unsafe {
            dequant_i8_to_f32_neon(&input, q, &mut neon_out);
        }
        dequant_i8_to_f32(&input, q, &mut scalar_out);
        for (i, (&n_v, &s_v)) in neon_out.iter().zip(scalar_out.iter()).enumerate() {
            assert!(
                close_within_ulp(n_v, s_v, 2),
                "exact-chunk boundary mismatch at {i}: neon={n_v} scalar={s_v}"
            );
        }
    }

    // ────────────────────────────────────────────────────────────────────
    // FP16 polynomial expf parity tests, skipped without ARMv8.2-A FP16.
    // ────────────────────────────────────────────────────────────────────

    fn fp16_supported() -> bool {
        std::arch::is_aarch64_feature_detected!("fp16")
    }

    #[test]
    fn expf_neon_f32x8_via_f16_matches_libm_relative() {
        if !fp16_supported() {
            crate::test_support::report_skip(
                "fp16 not supported on this CPU (FP16 expf parity test)",
            );
            return;
        }
        // The oracle saturates at exp(±10) as the kernel does.
        let cases: Vec<f32> = (-30..=30).map(|i| i as f32 * 0.5).collect();
        for chunk in cases.chunks(8) {
            if chunk.len() < 8 {
                continue;
            }
            let lo = unsafe { vld1q_f32(chunk.as_ptr()) };
            let hi = unsafe { vld1q_f32(chunk.as_ptr().add(4)) };
            let (out_lo, out_hi) = unsafe { expf_neon_f32x8_via_f16(lo, hi) };
            let mut out = [0f32; 8];
            unsafe {
                vst1q_f32(out.as_mut_ptr(), out_lo);
                vst1q_f32(out.as_mut_ptr().add(4), out_hi);
            }
            for (i, &x) in chunk.iter().enumerate() {
                let oracle = x.clamp(-10.0, 10.0).exp();
                let abs_err = (out[i] - oracle).abs();
                let rel_err = abs_err / oracle.max(1e-6);
                assert!(
                    rel_err < 0.02 || abs_err < 1e-4,
                    "f16 expf mismatch at x={x}: got={} oracle={oracle} rel_err={rel_err:.4}",
                    out[i]
                );
            }
        }
    }

    #[test]
    fn sigmoid_slice_f32_neon_fp16_matches_scalar() {
        if !fp16_supported() {
            crate::test_support::report_skip("fp16 not supported");
            return;
        }
        // 32 elements: 4 chunks of 8, no tail.
        let cases: Vec<f32> = (-16..16).map(|i| i as f32 * 0.5).collect();
        let mut neon_buf = cases.clone();
        let mut scalar_buf = cases;
        unsafe {
            sigmoid_slice_f32_neon_fp16(&mut neon_buf);
        }
        sigmoid_slice_f32(&mut scalar_buf);
        for (i, (&n_v, &s_v)) in neon_buf.iter().zip(scalar_buf.iter()).enumerate() {
            let abs_err = (n_v - s_v).abs();
            assert!(
                abs_err < 5e-3,
                "fp16 sigmoid mismatch at {i}: neon={n_v} scalar={s_v} err={abs_err:.5}"
            );
        }
    }

    #[test]
    fn softmax_neon_fp16_matches_scalar() {
        if !fp16_supported() {
            crate::test_support::report_skip("fp16 not supported");
            return;
        }
        // 16 elements: 2 chunks of 8.
        let cases: Vec<f32> = (0..16).map(|i| (i as f32) * 0.5 - 4.0).collect();
        let mut neon_buf = cases.clone();
        let mut scalar_buf = cases;
        unsafe {
            softmax_inplace_f32_neon_fp16(&mut neon_buf);
        }
        softmax_inplace_f32(&mut scalar_buf);
        for (i, (&n_v, &s_v)) in neon_buf.iter().zip(scalar_buf.iter()).enumerate() {
            let abs_err = (n_v - s_v).abs();
            assert!(
                abs_err < 5e-3,
                "fp16 softmax mismatch at {i}: neon={n_v} scalar={s_v} err={abs_err:.5}"
            );
        }
        let sum: f32 = neon_buf.iter().sum();
        assert!((sum - 1.0).abs() < 1e-3, "fp16 softmax sum drift: {sum}");
    }

    // ── NEON weighted sum ────────────────────────────────────────────
    use crate::prim::box_primitives::weighted_sum_4sides_f32;

    #[test]
    fn weighted_sum_neon_matches_scalar_uniform() {
        let probs = [0.25_f32; 64]; // 4 sides × 16 bins, uniform
        let scalar = weighted_sum_4sides_f32(&probs, 16);
        let neon = unsafe { weighted_sum_4sides_f32_neon(&probs, 16) };
        for (i, (&s, &n)) in scalar.iter().zip(neon.iter()).enumerate() {
            assert!(
                (s - n).abs() < 1e-4,
                "weighted_sum uniform mismatch side {i}: scalar={s} neon={n}"
            );
        }
    }

    #[test]
    fn weighted_sum_neon_matches_scalar_one_hot() {
        let mut probs = [0.0_f32; 64];
        for side in 0..4 {
            probs[side * 16 + 5] = 1.0;
        }
        let scalar = weighted_sum_4sides_f32(&probs, 16);
        let neon = unsafe { weighted_sum_4sides_f32_neon(&probs, 16) };
        for (i, (&s, &n)) in scalar.iter().zip(neon.iter()).enumerate() {
            assert!(
                (s - n).abs() < 1e-5,
                "weighted_sum one-hot mismatch side {i}: scalar={s} neon={n}"
            );
        }
    }

    // ── NEON dist2bbox ───────────────────────────────────────────────
    use crate::prim::box_primitives::dist2bbox_anchor_f32;

    #[test]
    fn dist2bbox_4anchors_matches_scalar() {
        let ltrb = [
            2.0_f32, 2.0, 2.0, 2.0, // anchor 0: symmetric
            1.0, 0.0, 3.0, 0.0, // anchor 1: l/r only
            0.0, 1.0, 0.0, 3.0, // anchor 2: t/b only
            5.0, 3.0, 7.0, 1.0, // anchor 3: asymmetric
        ];
        let gx = [0.5_f32, 0.5, 0.5, 0.5];
        let gy = [0.5_f32, 0.5, 0.5, 0.5];
        let stride = 8.0;
        let mut neon_out = [0.0_f32; 16];
        unsafe {
            dist2bbox_4anchors_f32_neon(&ltrb, &gx, &gy, stride, &mut neon_out);
        }
        for a in 0..4 {
            let ltrb_a = [
                ltrb[a * 4],
                ltrb[a * 4 + 1],
                ltrb[a * 4 + 2],
                ltrb[a * 4 + 3],
            ];
            let scalar = dist2bbox_anchor_f32(ltrb_a, gx[a], gy[a], stride);
            for c in 0..4 {
                assert!(
                    (scalar[c] - neon_out[a * 4 + c]).abs() < 1e-4,
                    "dist2bbox mismatch anchor={a} coord={c}: scalar={} neon={}",
                    scalar[c],
                    neon_out[a * 4 + c]
                );
            }
        }
    }

    // ── Fused dequant+sigmoid ────────────────────────────────────────

    #[test]
    fn fused_dequant_sigmoid_i8_matches_two_pass() {
        let q = Quant::new(0.1, 0);
        let input: Vec<i8> = (-64..64).collect();
        let mut two_pass = vec![0.0_f32; input.len()];
        let mut fused = vec![0.0_f32; input.len()];

        dequant_i8_to_f32(&input, q, &mut two_pass);
        sigmoid_slice_f32(&mut two_pass);

        unsafe {
            dequant_sigmoid_i8_to_f32_neon(&input, q, &mut fused);
        }

        for (i, (&t, &f)) in two_pass.iter().zip(fused.iter()).enumerate() {
            assert!(
                close_within_ulp(t, f, 8),
                "fused i8 mismatch at {i}: two_pass={t} fused={f}"
            );
        }
    }

    #[test]
    fn fused_dequant_sigmoid_u8_matches_two_pass() {
        let q = Quant::new(0.05, 128);
        let input: Vec<u8> = (0..=255).collect();
        let mut two_pass = vec![0.0_f32; input.len()];
        let mut fused = vec![0.0_f32; input.len()];

        dequant_u8_to_f32(&input, q, &mut two_pass);
        sigmoid_slice_f32(&mut two_pass);

        unsafe {
            dequant_sigmoid_u8_to_f32_neon(&input, q, &mut fused);
        }

        for (i, (&t, &f)) in two_pass.iter().zip(fused.iter()).enumerate() {
            assert!(
                close_within_ulp(t, f, 8),
                "fused u8 mismatch at {i}: two_pass={t} fused={f}"
            );
        }
    }

    // ── Half-precision dequant output ────────────────────────────────

    #[test]
    fn dequant_i8_to_f16_neon_matches_scalar() {
        if !std::arch::is_aarch64_feature_detected!("fp16") {
            crate::test_support::report_skip(
                "dequant_i8_to_f16_neon_matches_scalar - no FP16 instructions (e.g. Cortex-A53)",
            );
            return;
        }
        let q = Quant::new(0.1, -10);
        let input: Vec<i8> = (-20..20).collect();
        let mut neon_out = vec![0u16; input.len()];
        let mut scalar_f32 = vec![0.0_f32; input.len()];
        dequant_i8_to_f32(&input, q, &mut scalar_f32);

        unsafe {
            dequant_i8_to_f16_neon(&input, q, &mut neon_out);
        }

        for (i, (&neon_bits, &sf)) in neon_out.iter().zip(scalar_f32.iter()).enumerate() {
            let neon_f = half::f16::from_bits(neon_bits).to_f32();
            let scalar_f16 = half::f16::from_f32(sf).to_f32();
            assert!(
                (neon_f - scalar_f16).abs() < 1e-2,
                "i8→f16 mismatch at {i}: neon={neon_f} scalar_f16={scalar_f16}"
            );
        }
    }

    #[test]
    fn dequant_u8_to_f16_neon_matches_scalar() {
        if !std::arch::is_aarch64_feature_detected!("fp16") {
            crate::test_support::report_skip(
                "dequant_u8_to_f16_neon_matches_scalar - no FP16 instructions (e.g. Cortex-A53)",
            );
            return;
        }
        let q = Quant::new(0.5, 0);
        let input: Vec<u8> = (0..48).collect();
        let mut neon_out = vec![0u16; input.len()];
        let mut scalar_f32 = vec![0.0_f32; input.len()];
        dequant_u8_to_f32(&input, q, &mut scalar_f32);

        unsafe {
            dequant_u8_to_f16_neon(&input, q, &mut neon_out);
        }

        for (i, (&neon_bits, &sf)) in neon_out.iter().zip(scalar_f32.iter()).enumerate() {
            let neon_f = half::f16::from_bits(neon_bits).to_f32();
            let scalar_f16 = half::f16::from_f32(sf).to_f32();
            assert!(
                (neon_f - scalar_f16).abs() < 1e-2,
                "u8→f16 mismatch at {i}: neon={neon_f} scalar_f16={scalar_f16}"
            );
        }
    }

    #[test]
    fn fused_softmax_weighted_sum_matches_separate() {
        let reg_max = 16usize;
        // 4 sides of DFL logits in [-3, 3].
        let logits: Vec<f32> = (0..(4 * reg_max))
            .map(|i| ((i as f32) * 0.37 - 2.5).sin() * 3.0)
            .collect();

        // Reference: separate softmax + weighted_sum (scalar oracle).
        let mut probs = logits.clone();
        for side in 0..4 {
            softmax_inplace_f32(&mut probs[side * reg_max..(side + 1) * reg_max]);
        }
        let mut ref_ws = [0.0_f32; 4];
        for (side, ws) in ref_ws.iter_mut().enumerate() {
            let base = side * reg_max;
            for bin in 0..reg_max {
                *ws += probs[base + bin] * (bin as f32);
            }
        }

        let fused_ws = unsafe { softmax_weighted_sum_4sides_f32_neon(&logits, reg_max) };

        for (side, (&fused, &reference)) in fused_ws.iter().zip(ref_ws.iter()).enumerate() {
            let diff = (fused - reference).abs();
            assert!(
                diff < 1e-4,
                "fused softmax+ws mismatch on side {side}: fused={fused} ref={reference} diff={diff}",
            );
        }
    }

    #[test]
    fn fused_softmax_weighted_sum_reg_max_32() {
        // reg_max = 32: several chunks per side.
        let reg_max = 32usize;
        let logits: Vec<f32> = (0..(4 * reg_max))
            .map(|i| ((i as f32) * 0.23 - 1.8).cos() * 2.5)
            .collect();

        let mut probs = logits.clone();
        for side in 0..4 {
            softmax_inplace_f32(&mut probs[side * reg_max..(side + 1) * reg_max]);
        }
        let mut ref_ws = [0.0_f32; 4];
        for (side, ws) in ref_ws.iter_mut().enumerate() {
            let base = side * reg_max;
            for bin in 0..reg_max {
                *ws += probs[base + bin] * (bin as f32);
            }
        }

        let fused_ws = unsafe { softmax_weighted_sum_4sides_f32_neon(&logits, reg_max) };

        for (side, (&fused, &reference)) in fused_ws.iter().zip(ref_ws.iter()).enumerate() {
            let diff = (fused - reference).abs();
            assert!(
                diff < 1e-4,
                "fused softmax+ws (reg_max=32) mismatch on side {side}: fused={fused} ref={reference} diff={diff}",
            );
        }
    }
}
