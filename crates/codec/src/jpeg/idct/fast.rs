// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! Opt-in fast 8×8 IDCT (AAN factorisation, libjpeg `ifast` class).
//!
//! **Off by default.** Selected via [`crate::DctMethod::Fast`]; the default
//! path remains the accurate `islow`-class kernel, and every published
//! comparison against libjpeg-turbo quotes the accurate kernel. This kernel
//! trades a bounded accuracy loss (small quantised-domain rounding error, and
//! the documented `ifast` degeneration at very high quality factors ≥ ~97)
//! for roughly an eighth of the multiplies:
//!
//! - The per-coefficient AAN scale factors are folded into the **dequant
//!   table** at derive time ([`derive_fast_quant`]), so the butterfly needs
//!   only four constants.
//! - All four constants are applied as `x·(1+c)` or `x·(2+c)` with `c < 1`
//!   in Q15, i.e. one `SQDMULH` plus adds — no widening multiplies, the
//!   whole transform stays in 16-bit lanes end to end.
//! - Like the accurate kernels, arithmetic wraps on out-of-contract inputs
//!   (a hostile stream garbles only its own pixels; see `IdctFn`).
//!
//! The scalar kernel is the reference: it mirrors the NEON ops exactly
//! (including `SQDMULH` semantics and rounding shifts), so scalar↔NEON
//! parity is bit-exact — same discipline as `color.rs`.

/// AAN dequant scale factors in Q14: `2^14 · f(row) · f(col)` where
/// `f(0) = 1` and `f(k) = √2 · cos(kπ/16)` — the classic `aanscales` table.
/// Verified against the closed form in the tests below.
#[cfg_attr(not(target_arch = "aarch64"), allow(dead_code))] // fast kernels bind on NEON only
#[rustfmt::skip]
pub const AANSCALES: [u16; 64] = [
    16384, 22725, 21407, 19266, 16384, 12873,  8867,  4520,
    22725, 31521, 29692, 26722, 22725, 17855, 12299,  6270,
    21407, 29692, 27969, 25172, 21407, 16819, 11585,  5906,
    19266, 26722, 25172, 22654, 19266, 15137, 10426,  5315,
    16384, 22725, 21407, 19266, 16384, 12873,  8867,  4520,
    12873, 17855, 16819, 15137, 12873, 10114,  6967,  3552,
     8867, 12299, 11585, 10426,  8867,  6967,  4799,  2446,
     4520,  6270,  5906,  5315,  4520,  3552,  2446,  1247,
];

/// Q15 butterfly constants: the fractional parts of the four AAN multipliers.
/// `round((m − ⌊m⌋) · 2^15)` for m ∈ {1.414213562, 1.847759065, 1.082392200,
/// 2.613125930}.
const F_0_414: i16 = 13573; // 1.414213562 − 1
const F_0_847: i16 = 27779; // 1.847759065 − 1
const F_0_082: i16 = 2700; //  1.082392200 − 1
const F_0_613: i16 = 20091; // 2.613125930 − 2

/// Derive the AAN-prescaled dequant table: `(quant · aanscale + 2^11) >> 12`
/// (Q14 scale, keeping the `ifast` 2-bit headroom). Values stay well inside
/// `i16` (255 · 31521 ≫ 12 = 1962) and are always positive, so they are
/// stored in the same `[u16; 64]` shape the kernels already take and
/// reinterpreted as `i16` lanes at multiply time, exactly like the accurate
/// kernels' dequant.
#[cfg_attr(not(target_arch = "aarch64"), allow(dead_code))] // fast kernels bind on NEON only
pub fn derive_fast_quant(quant: &[u16; 64]) -> [u16; 64] {
    let mut out = [0u16; 64];
    for (o, (&q, &s)) in out.iter_mut().zip(quant.iter().zip(AANSCALES.iter())) {
        *o = (((q as u32) * (s as u32) + (1 << 11)) >> 12) as u16;
    }
    out
}

/// Scalar mirror of `vqdmulhq_n_s16`: `(2·a·b) >> 16` with saturation of the
/// doubling (only reachable at `a = b = i16::MIN`, which the Q15 constants
/// above never are).
#[cfg_attr(not(test), allow(dead_code))] // reference impl, exercised by the parity tests
#[inline(always)]
fn qdmulh(a: i16, b: i16) -> i16 {
    (((a as i32) * (b as i32) * 2) >> 16) as i16
}

/// One 8-point AAN butterfly on `i16` values, wrapping adds. `MULTIPLY(x, m)`
/// is expressed as `x + qdmulh(x, frac(m))` (or `2x + …` for m > 2), the same
/// shape the NEON kernel uses.
#[cfg_attr(not(test), allow(dead_code))] // reference impl, exercised by the parity tests
#[inline(always)]
fn butterfly8_fast(s: [i16; 8]) -> [i16; 8] {
    let mul_1_414 = |x: i16| x.wrapping_add(qdmulh(x, F_0_414));
    let mul_1_847 = |x: i16| x.wrapping_add(qdmulh(x, F_0_847));
    let mul_1_082 = |x: i16| x.wrapping_add(qdmulh(x, F_0_082));
    let mul_2_613 = |x: i16| x.wrapping_add(x).wrapping_add(qdmulh(x, F_0_613));

    // Even part.
    let tmp10 = s[0].wrapping_add(s[4]);
    let tmp11 = s[0].wrapping_sub(s[4]);
    let tmp13 = s[2].wrapping_add(s[6]);
    let tmp12 = mul_1_414(s[2].wrapping_sub(s[6])).wrapping_sub(tmp13);

    let e0 = tmp10.wrapping_add(tmp13);
    let e3 = tmp10.wrapping_sub(tmp13);
    let e1 = tmp11.wrapping_add(tmp12);
    let e2 = tmp11.wrapping_sub(tmp12);

    // Odd part.
    let z13 = s[5].wrapping_add(s[3]);
    let z10 = s[5].wrapping_sub(s[3]);
    let z11 = s[1].wrapping_add(s[7]);
    let z12 = s[1].wrapping_sub(s[7]);

    let t7 = z11.wrapping_add(z13);
    let t11 = mul_1_414(z11.wrapping_sub(z13));
    let z5 = mul_1_847(z10.wrapping_add(z12));
    let t10 = z5.wrapping_sub(mul_1_082(z12));
    let t12 = z5.wrapping_sub(mul_2_613(z10));

    let t6 = t12.wrapping_sub(t7);
    let t5 = t11.wrapping_sub(t6);
    let t4 = t10.wrapping_sub(t5);

    [
        e0.wrapping_add(t7),
        e1.wrapping_add(t6),
        e2.wrapping_add(t5),
        e3.wrapping_add(t4),
        e3.wrapping_sub(t4),
        e2.wrapping_sub(t5),
        e1.wrapping_sub(t6),
        e0.wrapping_sub(t7),
    ]
}

/// Scalar fast 8×8 IDCT on quantised coefficients with an AAN-prescaled
/// dequant table (from [`derive_fast_quant`]). Reference for the NEON kernel.
#[cfg_attr(not(test), allow(dead_code))] // reference impl, exercised by the parity tests
pub fn idct_8x8_scalar_fast(
    coeffs: &[i16; 64],
    fast_quant: &[u16; 64],
    output: &mut [u8],
    stride: usize,
) {
    let mut ws = [0i16; 64];

    // Pass 1: columns.
    for c in 0..8 {
        let mut s = [0i16; 8];
        for r in 0..8 {
            s[r] = coeffs[r * 8 + c].wrapping_mul(fast_quant[r * 8 + c] as i16);
        }
        let o = butterfly8_fast(s);
        for r in 0..8 {
            ws[r * 8 + c] = o[r];
        }
    }

    // Pass 2: rows, then descale (rounding >> 5) and centre.
    for r in 0..8 {
        let mut s = [0i16; 8];
        s.copy_from_slice(&ws[r * 8..r * 8 + 8]);
        let o = butterfly8_fast(s);
        for c in 0..8 {
            let v = ((o[c] as i32 + 16) >> 5) + 128;
            output[r * stride + c] = v.clamp(0, 255) as u8;
        }
    }
}

/// Scalar fast DC-only fill: both AAN passes broadcast the (prescaled)
/// dequantised DC, so the block is `clamp(((dc + 16) >> 5) + 128)`.
#[cfg_attr(not(test), allow(dead_code))] // reference impl, exercised by the parity tests
pub fn idct_dc_only_fast_scalar(dc_value: i32, output: &mut [u8], stride: usize) {
    let v = ((dc_value.wrapping_add(16) >> 5) + 128).clamp(0, 255) as u8;
    for row in output.chunks_mut(stride).take(8) {
        row[..8].fill(v);
    }
}

#[cfg(target_arch = "aarch64")]
pub use neon::{idct_8x8_neon_fast_k, idct_dc_only_fast_neon};

#[cfg(target_arch = "aarch64")]
mod neon {
    use std::arch::aarch64::*;

    use super::{F_0_082, F_0_414, F_0_613, F_0_847};

    /// Full-width 8-lane AAN butterfly (all eight columns at once — the
    /// 16-bit-only math is what lets the fast kernel skip the accurate
    /// kernel's 4-lane widening halves).
    #[inline(always)]
    unsafe fn butterfly8_fast_q(s: [int16x8_t; 8]) -> [int16x8_t; 8] {
        let mul_1_414 = |x: int16x8_t| vaddq_s16(x, vqdmulhq_n_s16(x, F_0_414));
        let mul_1_847 = |x: int16x8_t| vaddq_s16(x, vqdmulhq_n_s16(x, F_0_847));
        let mul_1_082 = |x: int16x8_t| vaddq_s16(x, vqdmulhq_n_s16(x, F_0_082));
        let mul_2_613 = |x: int16x8_t| vaddq_s16(vaddq_s16(x, x), vqdmulhq_n_s16(x, F_0_613));

        let tmp10 = vaddq_s16(s[0], s[4]);
        let tmp11 = vsubq_s16(s[0], s[4]);
        let tmp13 = vaddq_s16(s[2], s[6]);
        let tmp12 = vsubq_s16(mul_1_414(vsubq_s16(s[2], s[6])), tmp13);

        let e0 = vaddq_s16(tmp10, tmp13);
        let e3 = vsubq_s16(tmp10, tmp13);
        let e1 = vaddq_s16(tmp11, tmp12);
        let e2 = vsubq_s16(tmp11, tmp12);

        let z13 = vaddq_s16(s[5], s[3]);
        let z10 = vsubq_s16(s[5], s[3]);
        let z11 = vaddq_s16(s[1], s[7]);
        let z12 = vsubq_s16(s[1], s[7]);

        let t7 = vaddq_s16(z11, z13);
        let t11 = mul_1_414(vsubq_s16(z11, z13));
        let z5 = mul_1_847(vaddq_s16(z10, z12));
        let t10 = vsubq_s16(z5, mul_1_082(z12));
        let t12 = vsubq_s16(z5, mul_2_613(z10));

        let t6 = vsubq_s16(t12, t7);
        let t5 = vsubq_s16(t11, t6);
        let t4 = vsubq_s16(t10, t5);

        [
            vaddq_s16(e0, t7),
            vaddq_s16(e1, t6),
            vaddq_s16(e2, t5),
            vaddq_s16(e3, t4),
            vsubq_s16(e3, t4),
            vsubq_s16(e2, t5),
            vsubq_s16(e1, t6),
            vsubq_s16(e0, t7),
        ]
    }

    /// NEON fast 8×8 IDCT. Signature matches the accurate `idct_8x8_neon_k`
    /// so the MCU loop binds either as the same fn-item shape; the `last_k`
    /// sparsity hint is unused here — the AAN butterfly is cheap enough that
    /// tiering it has not been worth the branches so far.
    #[inline(always)]
    pub fn idct_8x8_neon_fast_k(
        coeffs: &[i16; 64],
        fast_quant: &[u16; 64],
        _last_k: u8,
        output: &mut [u8],
        stride: usize,
    ) {
        // SAFETY: caller is the aarch64 decode path, NEON presence probed.
        unsafe { idct_8x8_neon_fast_inner(coeffs, fast_quant, output, stride) }
    }

    #[target_feature(enable = "neon")]
    unsafe fn idct_8x8_neon_fast_inner(
        coeffs: &[i16; 64],
        fast_quant: &[u16; 64],
        output: &mut [u8],
        stride: usize,
    ) {
        // Dequantise all eight rows (prescaled table, wrapping 16-bit mul).
        let mut s = [vdupq_n_s16(0); 8];
        for (r, v) in s.iter_mut().enumerate() {
            *v = vmulq_s16(
                vld1q_s16(coeffs.as_ptr().add(r * 8)),
                vreinterpretq_s16_u16(vld1q_u16(fast_quant.as_ptr().add(r * 8))),
            );
        }

        // Pass 1 (columns: lanes are columns), transpose, pass 2 (rows).
        let w = butterfly8_fast_q(s);
        let (t0, t1, t2, t3, t4, t5, t6, t7) =
            super::super::neon::transpose_8x8_s16(w[0], w[1], w[2], w[3], w[4], w[5], w[6], w[7]);
        let o = butterfly8_fast_q([t0, t1, t2, t3, t4, t5, t6, t7]);

        // Descale (rounding >> 5), centre on 128, narrow to u8 — output
        // vectors are columns, so transpose bytes back to rows and store.
        let c128 = vdupq_n_s16(128);
        let narrow =
            |x: int16x8_t| -> uint8x8_t { vqmovun_s16(vaddq_s16(vrshrq_n_s16::<5>(x), c128)) };
        let (y0, y1, y2, y3, y4, y5, y6, y7) = super::super::neon::transpose_8x8_u8(
            narrow(o[0]),
            narrow(o[1]),
            narrow(o[2]),
            narrow(o[3]),
            narrow(o[4]),
            narrow(o[5]),
            narrow(o[6]),
            narrow(o[7]),
        );
        let out = output.as_mut_ptr();
        vst1_u8(out, y0);
        vst1_u8(out.add(stride), y1);
        vst1_u8(out.add(2 * stride), y2);
        vst1_u8(out.add(3 * stride), y3);
        vst1_u8(out.add(4 * stride), y4);
        vst1_u8(out.add(5 * stride), y5);
        vst1_u8(out.add(6 * stride), y6);
        vst1_u8(out.add(7 * stride), y7);
    }

    /// NEON fast DC-only fill (see the scalar version for the scaling).
    #[inline(always)]
    pub fn idct_dc_only_fast_neon(dc_value: i32, output: &mut [u8], stride: usize) {
        let v = ((dc_value.wrapping_add(16) >> 5) + 128).clamp(0, 255) as u8;
        // SAFETY: 8×8 block within `output` per the IdctFn contract.
        unsafe {
            let fill = vdup_n_u8(v);
            for row in 0..8 {
                vst1_u8(output.as_mut_ptr().add(row * stride), fill);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The literal AANSCALES table must equal `2^14·f(r)·f(c)` rounded.
    #[test]
    fn aanscales_matches_closed_form() {
        let f = |k: usize| -> f64 {
            if k == 0 {
                1.0
            } else {
                std::f64::consts::SQRT_2 * (k as f64 * std::f64::consts::PI / 16.0).cos()
            }
        };
        for r in 0..8 {
            for c in 0..8 {
                let expect = (16384.0 * f(r) * f(c)).round() as u16;
                assert_eq!(AANSCALES[r * 8 + c], expect, "({r},{c})");
            }
        }
    }

    /// Forward-DCT an 8×8 pixel block and quantise — the only way to get
    /// *valid* spectra. Random coefficient arrays are not valid DCT spectra:
    /// they decode to out-of-range "pixels" and legitimately overflow the
    /// fast kernel's 16-bit lanes (the documented `ifast` contract), so they
    /// cannot be used to judge its accuracy.
    fn quantised_spectrum(pixels: &[[u8; 8]; 8], quant: &[u16; 64]) -> [i16; 64] {
        let mut coeffs = [0i16; 64];
        for u in 0..8 {
            for v in 0..8 {
                let cu = if u == 0 {
                    std::f64::consts::FRAC_1_SQRT_2
                } else {
                    1.0
                };
                let cv = if v == 0 {
                    std::f64::consts::FRAC_1_SQRT_2
                } else {
                    1.0
                };
                let mut s = 0.0;
                for (y, row) in pixels.iter().enumerate() {
                    for (x, &p) in row.iter().enumerate() {
                        s += (p as f64 - 128.0)
                            * ((2 * y + 1) as f64 * u as f64 * std::f64::consts::PI / 16.0).cos()
                            * ((2 * x + 1) as f64 * v as f64 * std::f64::consts::PI / 16.0).cos();
                    }
                }
                let spec = cu * cv * s / 4.0;
                coeffs[u * 8 + v] = (spec / quant[u * 8 + v] as f64).round() as i16;
            }
        }
        coeffs
    }

    /// Fast scalar output must stay close to the accurate scalar reference on
    /// in-contract blocks (the whole point of `ifast`-class accuracy).
    #[test]
    fn fast_scalar_close_to_accurate() {
        use super::super::scalar::idct_8x8_scalar;
        let mut quant = [0u16; 64];
        for (i, q) in quant.iter_mut().enumerate() {
            *q = 2 + (i as u16 % 9);
        }
        let fq = derive_fast_quant(&quant);

        let mut state = 0x2468_ACE0u32;
        let mut next = move || {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            state
        };
        for t in 0..200 {
            // Gradient + edge + noise pixel content, then a real forward DCT.
            let mut px = [[0u8; 8]; 8];
            let edge = (next() % 8) as usize;
            for (y, row) in px.iter_mut().enumerate() {
                for (x, p) in row.iter_mut().enumerate() {
                    let base = 128.0
                        + 70.0 * ((x as f64 + t as f64) / 2.7).sin()
                        + 50.0 * ((y as f64 - t as f64) / 1.9).cos()
                        + if x >= edge { 40.0 } else { -40.0 };
                    let noise = (next() % 31) as f64 - 15.0;
                    *p = (base + noise).clamp(0.0, 255.0) as u8;
                }
            }
            let coeffs = quantised_spectrum(&px, &quant);
            let mut acc = [0u8; 64];
            let mut fast = [0u8; 64];
            idct_8x8_scalar(&coeffs, &quant, &mut acc, 8);
            idct_8x8_scalar_fast(&coeffs, &fq, &mut fast, 8);
            for i in 0..64 {
                let diff = (acc[i] as i32 - fast[i] as i32).abs();
                assert!(
                    diff <= 6,
                    "fast kernel drifted: idx {i} accurate={} fast={} diff={diff}",
                    acc[i],
                    fast[i]
                );
            }
        }
    }

    /// NEON fast kernel must be bit-identical to the fast scalar reference.
    #[cfg(target_arch = "aarch64")]
    #[test]
    fn fast_neon_matches_fast_scalar_exactly() {
        if !std::arch::is_aarch64_feature_detected!("neon") {
            eprintln!("SIMD feature not available, skipping");
            return;
        }
        let mut quant = [0u16; 64];
        for (i, q) in quant.iter_mut().enumerate() {
            *q = 1 + (i as u16 % 24);
        }
        let fq = derive_fast_quant(&quant);
        let mut state = 0x1357_9BDFu32;
        let mut next = move || {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            state
        };
        for _ in 0..500 {
            let mut coeffs = [0i16; 64];
            for c in coeffs.iter_mut() {
                if next() % 4 == 0 {
                    *c = (next() % 256) as i16 - 128;
                }
            }
            let mut s_out = [0u8; 64];
            let mut n_out = [0u8; 64];
            idct_8x8_scalar_fast(&coeffs, &fq, &mut s_out, 8);
            idct_8x8_neon_fast_k(&coeffs, &fq, 63, &mut n_out, 8);
            assert_eq!(s_out, n_out, "scalar/NEON fast parity");
        }
    }

    #[test]
    fn dc_only_fast_matches_full_kernel() {
        let quant = [3u16; 64];
        let fq = derive_fast_quant(&quant);
        for dc in [-200i16, -1, 0, 1, 77, 255] {
            let mut coeffs = [0i16; 64];
            coeffs[0] = dc;
            let mut full = [0u8; 64];
            let mut fill = [0u8; 64];
            idct_8x8_scalar_fast(&coeffs, &fq, &mut full, 8);
            let dcv = dc.wrapping_mul(fq[0] as i16) as i32;
            idct_dc_only_fast_scalar(dcv, &mut fill, 8);
            assert_eq!(full, fill, "dc={dc}");
        }
    }
}
