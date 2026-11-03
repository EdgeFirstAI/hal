// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! Baseline NEON fused dequantize and f32 sigmoid, stored as f32 or rounded once to f16.
//!
//! The sigmoid is the one `neon_baseline`'s fused kernels compute: `e = expf_neon_f32x4(−|x|)`,
//! then `1 / (1 + e)` for `x ≥ 0` and `e / (1 + e)` below. Integer inputs dequantize as
//! `bias + x · scale` with one FMA over whole 16-element blocks and pass the rest through the
//! matching `neon_baseline` f32 kernel itself, so every f16 result is exactly that kernel's f32
//! result rounded to nearest even. Float inputs dequantize as `prim::dequant` does
//! (`(x − zero_point) · scale`, no FMA; `Quant::IDENTITY` is a plain conversion) and run their
//! tail through the vector code on a padded block, so the f16 and f32 kernels agree on every
//! element. Every kernel writes nothing when the slices differ in length. Only ARMv8.0 Advanced
//! SIMD instructions are used.

#![cfg(target_arch = "aarch64")]

use std::arch::aarch64::*; // NOSONAR

use half::f16;

use super::neon_baseline::expf_neon_f32x4;
use super::neon_convert::{affine, map_steps, F32x16, LoadF32, StoreFloat};
use crate::Quant;

/// Sigmoid of 4 f32 lanes, the operation sequence of `neon_baseline`'s fused kernels.
#[inline(always)]
pub(crate) fn sigmoid4(x: float32x4_t) -> float32x4_t {
    // SAFETY: NEON is part of the aarch64 baseline.
    unsafe {
        let one = vdupq_n_f32(1.0);
        let mask = vcgeq_f32(x, vdupq_n_f32(0.0));
        let e = expf_neon_f32x4(vbslq_f32(mask, vnegq_f32(x), x));
        let recip = vdivq_f32(one, vaddq_f32(one, e));
        vbslq_f32(mask, recip, vmulq_f32(e, recip))
    }
}

/// Dequantizes 16 elements with `dq`, then stores their sigmoid.
#[inline(always)]
fn sigmoid_steps<S: LoadF32, D: StoreFloat>(
    src: &[S],
    dst: &mut [D],
    dq: impl Fn(F32x16) -> F32x16,
) {
    // SAFETY: `map_steps` hands out 16 valid elements at each pointer.
    map_steps(src, dst, |s, d| unsafe {
        D::store16(d, dq(S::load16(s)).map(sigmoid4));
    });
}

/// `sigmoid(bias + x · scale)`, `bias = −zero_point · scale`, one FMA per lane, into f16: whole
/// 16-element blocks in one pass, the rest through `f32_kernel` itself, then rounded.
#[inline(always)]
fn sigmoid_fma<S: LoadF32>(
    src: &[S],
    q: Quant,
    dst: &mut [f16],
    f32_kernel: impl Fn(&[S], Quant, &mut [f32]),
) {
    if src.len() != dst.len() {
        return;
    }
    let body = src.len() - src.len() % 16;
    let (src, tail_src) = src.split_at(body);
    let (dst, tail_dst) = dst.split_at_mut(body);
    let scale = q.scale;
    // SAFETY: NEON is part of the aarch64 baseline.
    let bias = unsafe { vdupq_n_f32(-(q.zero_point as f32) * scale) };
    // SAFETY: as above.
    sigmoid_steps(src, dst, |v| {
        v.map(|x| unsafe { vfmaq_n_f32(bias, x, scale) })
    });
    let mut buf = [0f32; 16];
    if let Some(buf) = buf.get_mut(..tail_src.len()) {
        f32_kernel(tail_src, q, buf);
        for (d, &y) in tail_dst.iter_mut().zip(buf.iter()) {
            *d = f16::from_f32(y);
        }
    }
}

/// `sigmoid((x − zero_point) · scale)` as `prim::dequant` rounds it; `Quant::IDENTITY` skips it.
#[inline(always)]
fn sigmoid_exact<S: LoadF32, D: StoreFloat>(src: &[S], q: Quant, dst: &mut [D]) {
    if q == Quant::IDENTITY {
        sigmoid_steps(src, dst, |v| v);
        return;
    }
    // SAFETY: NEON is part of the aarch64 baseline.
    let (zp, scale) = unsafe { (vdupq_n_f32(q.zero_point as f32), vdupq_n_f32(q.scale)) };
    sigmoid_steps(src, dst, |v| affine(v, zp, scale));
}

macro_rules! sigmoid_kernel {
    ($(#[$doc:meta])* $name:ident, $s:ty, $d:ty, $body:ident) => {
        $(#[$doc])*
        ///
        /// # Safety
        /// The CPU must support NEON (every aarch64 CPU does).
        #[inline]
        #[target_feature(enable = "neon")]
        pub unsafe fn $name(src: &[$s], q: Quant, dst: &mut [$d]) {
            $body(src, q, dst);
        }
    };
}

macro_rules! fma_kernel {
    ($(#[$doc:meta])* $name:ident, $s:ty, $f32:ident) => {
        $(#[$doc])*
        ///
        /// # Safety
        /// The CPU must support NEON (every aarch64 CPU does).
        #[inline]
        #[target_feature(enable = "neon")]
        pub unsafe fn $name(src: &[$s], q: Quant, dst: &mut [f16]) {
            // SAFETY: NEON is part of the aarch64 baseline; `sigmoid_fma` passes equal lengths.
            sigmoid_fma(src, q, dst, |s, q, d| unsafe { super::neon_baseline::$f32(s, q, d) });
        }
    };
}

fma_kernel!(
    /// `neon_baseline::dequant_sigmoid_i8_to_f32_neon` rounded to f16, element for element.
    dequant_sigmoid_i8_to_f16_neon,
    i8,
    dequant_sigmoid_i8_to_f32_neon
);
fma_kernel!(
    /// `neon_baseline::dequant_sigmoid_u8_to_f32_neon` rounded to f16, element for element.
    dequant_sigmoid_u8_to_f16_neon,
    u8,
    dequant_sigmoid_u8_to_f32_neon
);
fma_kernel!(
    /// `neon_baseline::dequant_sigmoid_i16_to_f32_neon` rounded to f16, element for element.
    dequant_sigmoid_i16_to_f16_neon,
    i16,
    dequant_sigmoid_i16_to_f32_neon
);
fma_kernel!(
    /// `neon_baseline::dequant_sigmoid_u16_to_f32_neon` rounded to f16, element for element.
    dequant_sigmoid_u16_to_f16_neon,
    u16,
    dequant_sigmoid_u16_to_f32_neon
);
sigmoid_kernel!(
    /// `sigmoid((x − zero_point) · scale)` of f16 into f16, in f32.
    dequant_sigmoid_f16_to_f16_neon,
    f16,
    f16,
    sigmoid_exact
);
sigmoid_kernel!(
    /// `sigmoid((x − zero_point) · scale)` of f32 into f16, in f32.
    dequant_sigmoid_f32_to_f16_neon,
    f32,
    f16,
    sigmoid_exact
);
sigmoid_kernel!(
    /// `sigmoid((x − zero_point) · scale)` of f16 into f32.
    dequant_sigmoid_f16_to_f32_neon,
    f16,
    f32,
    sigmoid_exact
);
sigmoid_kernel!(
    /// `sigmoid((x − zero_point) · scale)` of f32 into f32.
    dequant_sigmoid_f32_to_f32_neon,
    f32,
    f32,
    sigmoid_exact
);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::prim::neon_baseline as nb;

    /// The integer f16 kernels give the f32 kernels' results rounded once, tails included.
    #[test]
    fn integer_f16_is_the_f32_kernel_rounded() {
        let mut i8s: Vec<i8> = (i8::MIN..=i8::MAX).collect();
        i8s.extend(-7..6);
        let mut i16s: Vec<i16> = (i16::MIN..=i16::MAX).collect();
        i16s.extend(-5..4);
        for q in [
            Quant::new(0.1, 0),
            Quant::new(0.2, -20),
            Quant::new(0.07, 12),
        ] {
            let mut want = vec![0f32; i8s.len()];
            let mut got = vec![f16::ZERO; i8s.len()];
            // SAFETY: NEON is part of the aarch64 baseline.
            unsafe {
                nb::dequant_sigmoid_i8_to_f32_neon(&i8s, q, &mut want);
                dequant_sigmoid_i8_to_f16_neon(&i8s, q, &mut got);
            }
            for (g, w) in got.iter().zip(&want) {
                assert_eq!(g.to_bits(), f16::from_f32(*w).to_bits(), "i8 {q:?} {w}");
            }
        }
        for q in [Quant::new(0.001, 0), Quant::new(0.0005, 100)] {
            let mut want = vec![0f32; i16s.len()];
            let mut got = vec![f16::ZERO; i16s.len()];
            // SAFETY: NEON is part of the aarch64 baseline.
            unsafe {
                nb::dequant_sigmoid_i16_to_f32_neon(&i16s, q, &mut want);
                dequant_sigmoid_i16_to_f16_neon(&i16s, q, &mut got);
            }
            for (g, w) in got.iter().zip(&want) {
                assert_eq!(g.to_bits(), f16::from_f32(*w).to_bits(), "i16 {q:?} {w}");
            }
        }
    }

    /// `neon_baseline`'s scalar libm sigmoid of one dequantized value.
    fn libm_sigmoid(v: f32, q: Quant) -> f32 {
        let x = (v - q.zero_point as f32) * q.scale;
        if x >= 0.0 {
            1.0 / (1.0 + (-x).exp())
        } else {
            let e = x.exp();
            e / (1.0 + e)
        }
    }

    /// At every tail length 1..=15, the integer f16 kernels equal the f32 kernels rounded once,
    /// on values whose vector and libm sigmoids round to different f16s; a tail that took the
    /// wrong path for any element would fail.
    #[test]
    fn integer_f16_tails_match_the_f32_kernel() {
        macro_rules! check {
            ($t:ty, $all:expr, $f32k:ident, $f16k:ident, $quants:expr, $need:expr) => {{
                let all: Vec<$t> = $all;
                let mut found = 0;
                for q in $quants {
                    let mut vector = vec![0f32; all.len()];
                    // SAFETY: NEON is part of the aarch64 baseline; whole 16-element blocks.
                    unsafe { nb::$f32k(&all, q, &mut vector) };
                    let steep: Vec<$t> = all
                        .iter()
                        .zip(&vector)
                        .filter(|&(&v, &y)| {
                            f16::from_f32(y) != f16::from_f32(libm_sigmoid(f32::from(v), q))
                        })
                        .map(|(&v, _)| v)
                        .collect();
                    found += steep.len();
                    let pool = if steep.is_empty() { &all } else { &steep };
                    for tail in 1..=15 {
                        let src: Vec<$t> = pool.iter().copied().cycle().take(32 + tail).collect();
                        let mut want = vec![0f32; src.len()];
                        let mut got = vec![f16::ZERO; src.len()];
                        // SAFETY: NEON is part of the aarch64 baseline.
                        unsafe {
                            nb::$f32k(&src, q, &mut want);
                            $f16k(&src, q, &mut got);
                        }
                        for (i, (g, w)) in got.iter().zip(&want).enumerate() {
                            let w16 = f16::from_f32(*w);
                            let what = stringify!($t);
                            assert_eq!(
                                g.to_bits(),
                                w16.to_bits(),
                                "{what} {q:?} tail {tail} at {i}"
                            );
                        }
                    }
                }
                assert!(found > 0 || !$need, "{}: no steep values", stringify!($t));
            }};
        }
        let q8 = [
            Quant::new(0.1, 0),
            Quant::new(0.2, -20),
            Quant::new(0.07, 12),
        ];
        let q16 = [
            Quant::new(0.001, 0),
            Quant::new(0.0005, 100),
            Quant::new(0.0011, -3),
        ];
        let u16q = q16.map(|q| Quant::new(q.scale, q.zero_point + 32768));
        let u8q = q8.map(|q| Quant::new(q.scale, q.zero_point + 128));
        check!(
            i8,
            (i8::MIN..=i8::MAX).collect(),
            dequant_sigmoid_i8_to_f32_neon,
            dequant_sigmoid_i8_to_f16_neon,
            q8,
            false
        );
        check!(
            u8,
            (0..=u8::MAX).collect(),
            dequant_sigmoid_u8_to_f32_neon,
            dequant_sigmoid_u8_to_f16_neon,
            u8q,
            false
        );
        check!(
            i16,
            (i16::MIN..=i16::MAX).collect(),
            dequant_sigmoid_i16_to_f32_neon,
            dequant_sigmoid_i16_to_f16_neon,
            q16,
            true
        );
        check!(
            u16,
            (0..=u16::MAX).collect(),
            dequant_sigmoid_u16_to_f32_neon,
            dequant_sigmoid_u16_to_f16_neon,
            u16q,
            true
        );
    }

    /// A float kernel's tail element takes the vector path: it equals the same value in a full block.
    #[test]
    fn tail_matches_full_block() {
        let src: Vec<f32> = (0..16).map(|i| i as f32 * 0.7 - 5.0).collect();
        for n in [1usize, 3, 7, 15] {
            let mut full = vec![0f32; 16];
            let mut part = vec![0f32; n];
            let mut full16 = vec![f16::ZERO; 16];
            let mut part16 = vec![f16::ZERO; n];
            // SAFETY: NEON is part of the aarch64 baseline.
            unsafe {
                dequant_sigmoid_f32_to_f32_neon(&src, Quant::IDENTITY, &mut full);
                dequant_sigmoid_f32_to_f32_neon(&src[..n], Quant::IDENTITY, &mut part);
                dequant_sigmoid_f32_to_f16_neon(&src, Quant::IDENTITY, &mut full16);
                dequant_sigmoid_f32_to_f16_neon(&src[..n], Quant::IDENTITY, &mut part16);
            }
            assert_eq!(part, full[..n], "n={n}");
            assert_eq!(part16, full16[..n], "n={n}");
        }
    }

    #[test]
    fn specials_and_mismatched_lengths() {
        let src = [
            f32::NAN,
            f32::INFINITY,
            f32::NEG_INFINITY,
            0.0,
            -0.0,
            1e30,
            -1e30,
        ];
        let mut out = [0f32; 7];
        let mut out16 = [f16::ZERO; 7];
        // SAFETY: NEON is part of the aarch64 baseline.
        unsafe {
            dequant_sigmoid_f32_to_f32_neon(&src, Quant::IDENTITY, &mut out);
            dequant_sigmoid_f32_to_f16_neon(&src, Quant::IDENTITY, &mut out16);
        }
        assert!(out[0].is_nan() && out16[0].is_nan());
        assert_eq!(out[1..], [1.0, 0.0, 0.5, 0.5, 1.0, 0.0]);
        let want16 = [1.0, 0.0, 0.5, 0.5, 1.0, 0.0].map(f16::from_f32);
        assert_eq!(out16[1..], want16);
        let mut short = [7f32; 6];
        // SAFETY: as above.
        unsafe { dequant_sigmoid_f32_to_f32_neon(&src, Quant::IDENTITY, &mut short) };
        assert_eq!(short, [7.0; 6]);
    }
}
