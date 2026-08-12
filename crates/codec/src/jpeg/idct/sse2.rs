// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! SSE2 8×8 IDCT in 16-bit lanes (libjpeg-turbo `jpeg_idct_islow` shape).
//!
//! One `__m128i` holds a whole row of eight `i16` coefficients, so pass 1 is a
//! butterfly *across* the eight row registers with lane *j* carrying column
//! *j* — no transpose is needed until the two passes meet.
//!
//! Every multiply is a `pmaddwd`. The Loeffler butterfly computes each output
//! as a sum of products of pairs of inputs, so interleaving the two source rows
//! with `punpck?wd` turns each term into one `pmaddwd`: one uop and 5 cycles
//! for two multiplies *and* their add, against `pmulld`'s two uops and 10
//! cycles for a single multiply. The constant pairs in [`K`] are an exact
//! integer regrouping of [`super::scalar`], so this kernel is bit-exact with
//! the scalar reference rather than merely close (see `parity_vs_scalar`).
//!
//! Dequantisation is folded into the load as `pmullw`, which keeps the block in
//! `i16` and avoids materialising a 256-byte `[i32; 64]` scratch. As in
//! libjpeg-turbo, this assumes the dequantised coefficient fits in `i16`;
//! encoders pair large quantiser values with small coefficients, so real
//! streams stay well inside that range.
//!
//! Only SSE2 is required — `pmullw`, `pmaddwd`, `packssdw` and `packuswb` are
//! all baseline — so SSE4.1 targets share this kernel too.

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

/// Fixed-point precision (must match scalar).
const PASS1_BITS: i32 = 2;
const CONST_BITS: i32 = 13;

/// Descale shift applied at the end of each pass.
const PASS1_SHIFT: i32 = CONST_BITS - PASS1_BITS; // 11
const PASS2_SHIFT: i32 = CONST_BITS + PASS1_BITS + 3; // 18

/// Pack two `i16` multipliers into the `i32` lane layout `pmaddwd` wants:
/// applied to `punpck?wd(x, y)` it yields `x * a + y * b`.
const fn pair(a: i16, b: i16) -> i32 {
    (((b as u16 as u32) << 16) | (a as u16 as u32)) as i32
}

/// `pmaddwd` constant pairs, regrouped from the scalar Loeffler constants.
///
/// Each entry folds the scalar form's shared sub-expressions into per-input
/// coefficients. For example the scalar even part computes
/// `tmp13 = (s2 + s6) * FIX_0_541 + s2 * FIX_0_765`, which is exactly
/// `s2 * (FIX_0_541 + FIX_0_765) + s6 * FIX_0_541` — the [`K::EVEN_26_13`]
/// pair. The regrouping is an integer identity, so no precision is lost.
mod k {
    use super::pair;

    // Even part, applied to punpck(s0, s4) — (s0 ± s4) << CONST_BITS.
    pub const EVEN_04_ADD: i32 = pair(8192, 8192);
    pub const EVEN_04_SUB: i32 = pair(8192, -8192);
    // Even part, applied to punpck(s2, s6).
    pub const EVEN_26_13: i32 = pair(10703, 4433);
    pub const EVEN_26_12: i32 = pair(4433, -10704);
    // Odd part: each output is madd(punpck(s7, s5)) + madd(punpck(s3, s1)).
    pub const ODD0_75: i32 = pair(-11363, 9633);
    pub const ODD0_31: i32 = pair(-6436, 2260);
    pub const ODD1_75: i32 = pair(9633, 2261);
    pub const ODD1_31: i32 = pair(-11362, 6437);
    pub const ODD2_75: i32 = pair(-6436, -11362);
    pub const ODD2_31: i32 = pair(-2259, 9633);
    pub const ODD3_75: i32 = pair(2260, 6437);
    pub const ODD3_31: i32 = pair(9633, 11363);
}

/// Transpose eight `__m128i` of `i16` in place (three unpack stages).
macro_rules! transpose8x8_i16 {
    ($r0:expr, $r1:expr, $r2:expr, $r3:expr, $r4:expr, $r5:expr, $r6:expr, $r7:expr) => {{
        let a0 = _mm_unpacklo_epi16($r0, $r1);
        let a1 = _mm_unpackhi_epi16($r0, $r1);
        let a2 = _mm_unpacklo_epi16($r2, $r3);
        let a3 = _mm_unpackhi_epi16($r2, $r3);
        let a4 = _mm_unpacklo_epi16($r4, $r5);
        let a5 = _mm_unpackhi_epi16($r4, $r5);
        let a6 = _mm_unpacklo_epi16($r6, $r7);
        let a7 = _mm_unpackhi_epi16($r6, $r7);

        let b0 = _mm_unpacklo_epi32(a0, a2);
        let b1 = _mm_unpackhi_epi32(a0, a2);
        let b2 = _mm_unpacklo_epi32(a1, a3);
        let b3 = _mm_unpackhi_epi32(a1, a3);
        let b4 = _mm_unpacklo_epi32(a4, a6);
        let b5 = _mm_unpackhi_epi32(a4, a6);
        let b6 = _mm_unpacklo_epi32(a5, a7);
        let b7 = _mm_unpackhi_epi32(a5, a7);

        $r0 = _mm_unpacklo_epi64(b0, b4);
        $r1 = _mm_unpackhi_epi64(b0, b4);
        $r2 = _mm_unpacklo_epi64(b1, b5);
        $r3 = _mm_unpackhi_epi64(b1, b5);
        $r4 = _mm_unpacklo_epi64(b2, b6);
        $r5 = _mm_unpackhi_epi64(b2, b6);
        $r6 = _mm_unpacklo_epi64(b3, b7);
        $r7 = _mm_unpackhi_epi64(b3, b7);
    }};
}

/// One Loeffler pass over eight `i16` rows, lane *j* being column *j*.
///
/// Returns the eight results as `i32` lo/hi half-pairs (`[(lo, hi); 8]`) still
/// at full precision: the caller applies the pass-specific descale and pack,
/// which differ between pass 1 (to `i16`) and pass 2 (to `u8`).
///
/// `bias` is added to the even part exactly where the scalar reference adds it.
///
/// Inlined into the `sse2`-enabled caller rather than carrying its own
/// `target_feature`, so the two passes share one register allocation.
#[inline(always)]
unsafe fn idct_pass(s: &[__m128i; 8], bias: __m128i) -> [(__m128i, __m128i); 8] {
    let p04l = _mm_unpacklo_epi16(s[0], s[4]);
    let p04h = _mm_unpackhi_epi16(s[0], s[4]);
    let p26l = _mm_unpacklo_epi16(s[2], s[6]);
    let p26h = _mm_unpackhi_epi16(s[2], s[6]);
    let p75l = _mm_unpacklo_epi16(s[7], s[5]);
    let p75h = _mm_unpackhi_epi16(s[7], s[5]);
    let p31l = _mm_unpacklo_epi16(s[3], s[1]);
    let p31h = _mm_unpackhi_epi16(s[3], s[1]);

    let madd = |v, c: i32| _mm_madd_epi16(v, _mm_set1_epi32(c));

    // Even part: tmp10/tmp11 are (s0 ± s4) << CONST_BITS, tmp13/tmp12 the
    // s2/s6 rotation. The bias folds in here, as in the scalar reference.
    let t10l = madd(p04l, k::EVEN_04_ADD);
    let t10h = madd(p04h, k::EVEN_04_ADD);
    let t11l = madd(p04l, k::EVEN_04_SUB);
    let t11h = madd(p04h, k::EVEN_04_SUB);
    let t13l = madd(p26l, k::EVEN_26_13);
    let t13h = madd(p26h, k::EVEN_26_13);
    let t12l = madd(p26l, k::EVEN_26_12);
    let t12h = madd(p26h, k::EVEN_26_12);

    let b10l = _mm_add_epi32(t10l, bias);
    let b10h = _mm_add_epi32(t10h, bias);
    let b11l = _mm_add_epi32(t11l, bias);
    let b11h = _mm_add_epi32(t11h, bias);

    let e0 = (_mm_add_epi32(b10l, t13l), _mm_add_epi32(b10h, t13h));
    let e3 = (_mm_sub_epi32(b10l, t13l), _mm_sub_epi32(b10h, t13h));
    let e1 = (_mm_add_epi32(b11l, t12l), _mm_add_epi32(b11h, t12h));
    let e2 = (_mm_sub_epi32(b11l, t12l), _mm_sub_epi32(b11h, t12h));

    // Odd part: two madds per output, one over (s7, s5) and one over (s3, s1).
    let o0 = (
        _mm_add_epi32(madd(p75l, k::ODD0_75), madd(p31l, k::ODD0_31)),
        _mm_add_epi32(madd(p75h, k::ODD0_75), madd(p31h, k::ODD0_31)),
    );
    let o1 = (
        _mm_add_epi32(madd(p75l, k::ODD1_75), madd(p31l, k::ODD1_31)),
        _mm_add_epi32(madd(p75h, k::ODD1_75), madd(p31h, k::ODD1_31)),
    );
    let o2 = (
        _mm_add_epi32(madd(p75l, k::ODD2_75), madd(p31l, k::ODD2_31)),
        _mm_add_epi32(madd(p75h, k::ODD2_75), madd(p31h, k::ODD2_31)),
    );
    let o3 = (
        _mm_add_epi32(madd(p75l, k::ODD3_75), madd(p31l, k::ODD3_31)),
        _mm_add_epi32(madd(p75h, k::ODD3_75), madd(p31h, k::ODD3_31)),
    );

    [
        (_mm_add_epi32(e0.0, o3.0), _mm_add_epi32(e0.1, o3.1)),
        (_mm_add_epi32(e1.0, o2.0), _mm_add_epi32(e1.1, o2.1)),
        (_mm_add_epi32(e2.0, o1.0), _mm_add_epi32(e2.1, o1.1)),
        (_mm_add_epi32(e3.0, o0.0), _mm_add_epi32(e3.1, o0.1)),
        (_mm_sub_epi32(e3.0, o0.0), _mm_sub_epi32(e3.1, o0.1)),
        (_mm_sub_epi32(e2.0, o1.0), _mm_sub_epi32(e2.1, o1.1)),
        (_mm_sub_epi32(e1.0, o2.0), _mm_sub_epi32(e1.1, o2.1)),
        (_mm_sub_epi32(e0.0, o3.0), _mm_sub_epi32(e0.1, o3.1)),
    ]
}

/// SSE2 8×8 IDCT with in-kernel dequantisation, bit-exact with the scalar path.
pub fn idct_8x8_sse2(coeffs: &[i16; 64], quant: &[u16; 64], output: &mut [u8], stride: usize) {
    // SAFETY: SSE2 is baseline on x86_64; `output` is checked to hold 8 rows.
    assert!(output.len() >= 7 * stride + 8, "IDCT output too small");
    unsafe { idct_8x8_sse2_inner(coeffs, quant, output, stride) }
}

#[target_feature(enable = "sse2")]
unsafe fn idct_8x8_sse2_inner(
    coeffs: &[i16; 64],
    quant: &[u16; 64],
    output: &mut [u8],
    stride: usize,
) {
    islow_body(coeffs, quant, output, stride)
}

/// The kernel proper, kept free of `target_feature` so each tier can stamp out
/// its own copy: AVX2 callers get the same code VEX-encoded, which drops the
/// register-copy `movdqa`s that the two-operand SSE encoding forces.
///
/// # Safety
/// Caller must guarantee SSE2 and that `output` holds `7 * stride + 8` bytes.
#[inline(always)]
pub(super) unsafe fn islow_body(
    coeffs: &[i16; 64],
    quant: &[u16; 64],
    output: &mut [u8],
    stride: usize,
) {
    // Dequantise on load: pmullw keeps the block in i16.
    let c = coeffs.as_ptr() as *const __m128i;
    let q = quant.as_ptr() as *const __m128i;
    let mut r = [_mm_setzero_si128(); 8];
    for (i, slot) in r.iter_mut().enumerate() {
        *slot = _mm_mullo_epi16(_mm_loadu_si128(c.add(i)), _mm_loadu_si128(q.add(i)));
    }

    // Pass 1 over columns, descaled to PASS1_BITS and packed back to i16.
    let p1 = idct_pass(&r, _mm_set1_epi32(1 << (PASS1_SHIFT - 1)));
    for (slot, (lo, hi)) in r.iter_mut().zip(p1) {
        *slot = _mm_packs_epi32(
            _mm_srai_epi32::<PASS1_SHIFT>(lo),
            _mm_srai_epi32::<PASS1_SHIFT>(hi),
        );
    }

    transpose8x8_i16!(r[0], r[1], r[2], r[3], r[4], r[5], r[6], r[7]);

    // Pass 2 over rows. The bias carries both the rounding term and the +128
    // level shift, so the descale lands directly in [0, 255] before packuswb.
    let bias = _mm_set1_epi32((1 << (PASS2_SHIFT - 1)) + (128 << PASS2_SHIFT));
    let p2 = idct_pass(&r, bias);
    for (slot, (lo, hi)) in r.iter_mut().zip(p2) {
        *slot = _mm_packs_epi32(
            _mm_srai_epi32::<PASS2_SHIFT>(lo),
            _mm_srai_epi32::<PASS2_SHIFT>(hi),
        );
    }

    // A pass butterflies *across* registers, so its results come out
    // column-major; transposing here puts each output row back in one register.
    transpose8x8_i16!(r[0], r[1], r[2], r[3], r[4], r[5], r[6], r[7]);

    for (row, v) in r.into_iter().enumerate() {
        // packuswb saturates to [0, 255], matching the scalar clamp. storel
        // writes the low 8 bytes straight out of the register — going via a
        // stack temporary here costs a store-forwarding stall per row.
        let bytes = _mm_packus_epi16(v, v);
        _mm_storel_epi64(output.as_mut_ptr().add(row * stride) as *mut __m128i, bytes);
    }
}

/// DC-only fast path: all 64 output values are the same.
pub fn idct_dc_only_sse2(dc_value: i32, output: &mut [u8], stride: usize) {
    let round = 1 << (PASS2_SHIFT - 1);
    let bias = round + (128 << PASS2_SHIFT);
    // Wrapping: in-contract values cannot overflow here, but the argument is an
    // i32 and a caller that hands over a wider one must get the same defined
    // result the block kernels give, not a debug-build panic.
    let scaled = dc_value.wrapping_shl((CONST_BITS + PASS1_BITS) as u32);
    let val = (scaled.wrapping_add(bias) >> PASS2_SHIFT).clamp(0, 255) as u8;

    assert!(output.len() >= 7 * stride + 8, "IDCT output too small");
    // SAFETY: SSE2 is baseline on x86_64; the assert covers all eight stores.
    unsafe {
        let fill = _mm_set1_epi8(val as i8);
        for row in 0..8 {
            _mm_storel_epi64(output.as_mut_ptr().add(row * stride) as *mut __m128i, fill);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::jpeg::idct::scalar::{idct_8x8_scalar, idct_dc_only_scalar};

    /// Deterministic xorshift so failures reproduce exactly.
    fn rng(seed: &mut u64) -> u64 {
        *seed ^= *seed << 13;
        *seed ^= *seed >> 7;
        *seed ^= *seed << 17;
        *seed
    }

    /// The regrouped `pmaddwd` constants must reproduce the scalar reference
    /// exactly, not approximately — including the clamp at both output ends and
    /// with a stride wider than the block.
    #[test]
    fn parity_vs_scalar() {
        let mut seed = 0x1234_5678_9ABC_DEF0u64;
        for case in 0..512 {
            let mut coeffs = [0i16; 64];
            for (i, c) in coeffs.iter_mut().enumerate() {
                // Later cases push hard on the [0, 255] clamp while keeping the
                // dequantised product inside the i16 range the kernel assumes.
                let mag = if case < 64 { 32 } else { 256 };
                let v = (rng(&mut seed) % (2 * mag)) as i32 - mag as i32;
                *c = if i < 16 || rng(&mut seed).is_multiple_of(4) {
                    v as i16
                } else {
                    0
                };
            }
            let mut quant = [1u16; 64];
            for q in quant.iter_mut() {
                *q = (rng(&mut seed) % 4 + 1) as u16;
            }

            let stride = 24;
            let mut got = vec![0xAAu8; stride * 8];
            let mut want = vec![0xAAu8; stride * 8];
            idct_8x8_sse2(&coeffs, &quant, &mut got, stride);
            idct_8x8_scalar(&coeffs, &quant, &mut want, stride);
            assert_eq!(got, want, "case {case} diverged from scalar reference");
        }
    }

    /// A zero block is the DC level shift, and nothing is written past column 8.
    #[test]
    fn zero_block_and_stride() {
        let stride = 16;
        let mut out = vec![0xFFu8; stride * 8];
        idct_8x8_sse2(&[0i16; 64], &[1u16; 64], &mut out, stride);
        for row in 0..8 {
            for col in 0..8 {
                assert_eq!(out[row * stride + col], 128);
            }
            for col in 8..stride {
                assert_eq!(out[row * stride + col], 0xFF, "wrote past column 8");
            }
        }
    }

    #[test]
    fn dc_only_parity_vs_scalar() {
        for dc in [0i32, 8, 64, 128, -64, 255, -255, 4095, -4095] {
            let mut want = [0u8; 64];
            let mut got = [0u8; 64];
            idct_dc_only_scalar(dc, &mut want, 8);
            idct_dc_only_sse2(dc, &mut got, 8);
            assert_eq!(got, want, "dc_only mismatch at dc={dc}");
        }
    }
}
