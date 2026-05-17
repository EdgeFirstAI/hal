// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! SSE2-optimised 8×8 IDCT.
//!
//! Uses 32-bit integer arithmetic with the same Loeffler butterfly as the
//! scalar and NEON paths, but processes 4 columns/rows at a time using SSE2.
//! SSE2 lacks `_mm_mullo_epi32` (SSE4.1), so we emulate it via
//! `_mm_mul_epu32` + shuffle.

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

/// Fixed-point precision (must match scalar).
const PASS1_BITS: i32 = 2;
const CONST_BITS: i32 = 13;

/// Loeffler constants (must match scalar).
const FIX_0_298: i32 = 2446;
const FIX_0_390: i32 = 3196;
const FIX_0_541: i32 = 4433;
const FIX_0_765: i32 = 6270;
const FIX_0_899: i32 = 7373;
const FIX_1_175: i32 = 9633;
const FIX_1_501: i32 = 12299;
const FIX_1_847: i32 = 15137;
const FIX_1_961: i32 = 16069;
const FIX_2_053: i32 = 16819;
const FIX_2_562: i32 = 20995;
const FIX_3_072: i32 = 25172;

/// Emulate `_mm_mullo_epi32` (SSE4.1) using SSE2 primitives.
///
/// `_mm_mul_epu32` multiplies the 32-bit integers at positions 0 and 2,
/// producing 64-bit results. We call it twice (shifting inputs for positions
/// 1 and 3), then shuffle to recombine the four low-32 products.
#[inline(always)]
unsafe fn mm_mullo_epi32_sse2(a: __m128i, b: __m128i) -> __m128i {
    // Multiply elements at positions 0 and 2
    let mul02 = _mm_mul_epu32(a, b);
    // Shift right by 4 bytes to align elements 1 and 3
    let mul13 = _mm_mul_epu32(_mm_srli_si128::<4>(a), _mm_srli_si128::<4>(b));
    // Extract low 32 bits from each 64-bit product:
    // shuffle 0b00_00_10_00 = 0x08 selects elements 0 and 2 into positions 0 and 1
    let lo = _mm_shuffle_epi32::<0x08>(mul02);
    let hi = _mm_shuffle_epi32::<0x08>(mul13);
    // Interleave: [a0*b0, a1*b1, a2*b2, a3*b3]
    _mm_unpacklo_epi32(lo, hi)
}

/// SSE2 4×4 transpose of `__m128i` (i32 lanes).
macro_rules! transpose4x4_sse2 {
    ($r0:expr, $r1:expr, $r2:expr, $r3:expr) => {{
        let t0 = _mm_unpacklo_epi32($r0, $r1);
        let t1 = _mm_unpackhi_epi32($r0, $r1);
        let t2 = _mm_unpacklo_epi32($r2, $r3);
        let t3 = _mm_unpackhi_epi32($r2, $r3);
        $r0 = _mm_unpacklo_epi64(t0, t2);
        $r1 = _mm_unpackhi_epi64(t0, t2);
        $r2 = _mm_unpacklo_epi64(t1, t3);
        $r3 = _mm_unpackhi_epi64(t1, t3);
    }};
}

/// Loeffler butterfly on 4 lanes (even part).
#[inline(always)]
unsafe fn even_part_sse2(
    s0: __m128i,
    s2: __m128i,
    s4: __m128i,
    s6: __m128i,
    bias: __m128i,
) -> (__m128i, __m128i, __m128i, __m128i) {
    let tmp0 = _mm_slli_epi32::<CONST_BITS>(s0);
    let tmp2 = _mm_slli_epi32::<CONST_BITS>(s4);

    let tmp10 = _mm_add_epi32(tmp0, tmp2);
    let tmp11 = _mm_sub_epi32(tmp0, tmp2);

    let z1 = _mm_add_epi32(s2, s6);
    let fix_0541 = _mm_set1_epi32(FIX_0_541);
    let fix_0765 = _mm_set1_epi32(FIX_0_765);
    let fix_1847 = _mm_set1_epi32(FIX_1_847);

    let tmp13 = _mm_add_epi32(
        mm_mullo_epi32_sse2(z1, fix_0541),
        mm_mullo_epi32_sse2(s2, fix_0765),
    );
    let tmp12 = _mm_sub_epi32(
        mm_mullo_epi32_sse2(z1, fix_0541),
        mm_mullo_epi32_sse2(s6, fix_1847),
    );

    let t0 = _mm_add_epi32(_mm_add_epi32(tmp10, tmp13), bias);
    let t3 = _mm_add_epi32(_mm_sub_epi32(tmp10, tmp13), bias);
    let t1 = _mm_add_epi32(_mm_add_epi32(tmp11, tmp12), bias);
    let t2 = _mm_add_epi32(_mm_sub_epi32(tmp11, tmp12), bias);

    (t0, t1, t2, t3)
}

/// Loeffler butterfly on 4 lanes (odd part).
#[inline(always)]
unsafe fn odd_part_sse2(
    s1: __m128i,
    s3: __m128i,
    s5: __m128i,
    s7: __m128i,
) -> (__m128i, __m128i, __m128i, __m128i) {
    let z1 = _mm_add_epi32(s7, s1);
    let z2 = _mm_add_epi32(s5, s3);
    let z3 = _mm_add_epi32(s7, s3);
    let z4 = _mm_add_epi32(s5, s1);
    let z5 = mm_mullo_epi32_sse2(_mm_add_epi32(z3, z4), _mm_set1_epi32(FIX_1_175));

    let p7 = mm_mullo_epi32_sse2(s7, _mm_set1_epi32(FIX_0_298));
    let p5 = mm_mullo_epi32_sse2(s5, _mm_set1_epi32(FIX_2_053));
    let p3 = mm_mullo_epi32_sse2(s3, _mm_set1_epi32(FIX_3_072));
    let p1 = mm_mullo_epi32_sse2(s1, _mm_set1_epi32(FIX_1_501));

    let z1 = mm_mullo_epi32_sse2(z1, _mm_set1_epi32(-FIX_0_899));
    let z2 = mm_mullo_epi32_sse2(z2, _mm_set1_epi32(-FIX_2_562));
    let z3 = _mm_add_epi32(mm_mullo_epi32_sse2(z3, _mm_set1_epi32(-FIX_1_961)), z5);
    let z4 = _mm_add_epi32(mm_mullo_epi32_sse2(z4, _mm_set1_epi32(-FIX_0_390)), z5);

    let tmp0_odd = _mm_add_epi32(_mm_add_epi32(p7, z1), z3);
    let tmp1_odd = _mm_add_epi32(_mm_add_epi32(p5, z2), z4);
    let tmp2_odd = _mm_add_epi32(_mm_add_epi32(p3, z2), z3);
    let tmp3_odd = _mm_add_epi32(_mm_add_epi32(p1, z1), z4);

    (tmp0_odd, tmp1_odd, tmp2_odd, tmp3_odd)
}

/// SSE2 8×8 IDCT: processes 4 columns at a time, transposes, then 4 rows.
pub fn idct_8x8_sse2(coeffs: &[i32; 64], output: &mut [u8], stride: usize) {
    unsafe { idct_8x8_sse2_inner(coeffs, output, stride) }
}

#[target_feature(enable = "sse2")]
unsafe fn idct_8x8_sse2_inner(coeffs: &[i32; 64], output: &mut [u8], stride: usize) {
    let mut workspace = [0i32; 64];

    // Pass 1: columns. Process columns 0-3 then 4-7.
    let half = _mm_set1_epi32(1 << (CONST_BITS - 1));

    for col_group in 0..2 {
        let base = col_group * 4;

        let s0 = _mm_loadu_si128(coeffs.as_ptr().add(base) as *const __m128i);
        let s1 = _mm_loadu_si128(coeffs.as_ptr().add(8 + base) as *const __m128i);
        let s2 = _mm_loadu_si128(coeffs.as_ptr().add(16 + base) as *const __m128i);
        let s3 = _mm_loadu_si128(coeffs.as_ptr().add(24 + base) as *const __m128i);
        let s4 = _mm_loadu_si128(coeffs.as_ptr().add(32 + base) as *const __m128i);
        let s5 = _mm_loadu_si128(coeffs.as_ptr().add(40 + base) as *const __m128i);
        let s6 = _mm_loadu_si128(coeffs.as_ptr().add(48 + base) as *const __m128i);
        let s7 = _mm_loadu_si128(coeffs.as_ptr().add(56 + base) as *const __m128i);

        let (t0, t1, t2, t3) = even_part_sse2(s0, s2, s4, s6, half);
        let (o0, o1, o2, o3) = odd_part_sse2(s1, s3, s5, s7);

        // Combine and descale (CONST_BITS - PASS1_BITS = 11)
        let r0 = _mm_srai_epi32::<{ 13 - 2 }>(_mm_add_epi32(t0, o3));
        let r7 = _mm_srai_epi32::<{ 13 - 2 }>(_mm_sub_epi32(t0, o3));
        let r1 = _mm_srai_epi32::<{ 13 - 2 }>(_mm_add_epi32(t1, o2));
        let r6 = _mm_srai_epi32::<{ 13 - 2 }>(_mm_sub_epi32(t1, o2));
        let r2 = _mm_srai_epi32::<{ 13 - 2 }>(_mm_add_epi32(t2, o1));
        let r5 = _mm_srai_epi32::<{ 13 - 2 }>(_mm_sub_epi32(t2, o1));
        let r3 = _mm_srai_epi32::<{ 13 - 2 }>(_mm_add_epi32(t3, o0));
        let r4 = _mm_srai_epi32::<{ 13 - 2 }>(_mm_sub_epi32(t3, o0));

        _mm_storeu_si128(workspace.as_mut_ptr().add(base) as *mut __m128i, r0);
        _mm_storeu_si128(workspace.as_mut_ptr().add(8 + base) as *mut __m128i, r1);
        _mm_storeu_si128(workspace.as_mut_ptr().add(16 + base) as *mut __m128i, r2);
        _mm_storeu_si128(workspace.as_mut_ptr().add(24 + base) as *mut __m128i, r3);
        _mm_storeu_si128(workspace.as_mut_ptr().add(32 + base) as *mut __m128i, r4);
        _mm_storeu_si128(workspace.as_mut_ptr().add(40 + base) as *mut __m128i, r5);
        _mm_storeu_si128(workspace.as_mut_ptr().add(48 + base) as *mut __m128i, r6);
        _mm_storeu_si128(workspace.as_mut_ptr().add(56 + base) as *mut __m128i, r7);
    }

    // Pass 2: rows. Process rows 0-3 then 4-7.
    let range_shift = CONST_BITS + PASS1_BITS + 3;
    let bias_val = (1 << (range_shift - 1)) + (128 << range_shift);
    let bias = _mm_set1_epi32(bias_val);
    let max255 = _mm_set1_epi32(255);

    for row_group in 0..2 {
        let row_base = row_group * 4 * 8;

        // Load left half (columns 0-3) and transpose
        let mut r0l = _mm_loadu_si128(workspace.as_ptr().add(row_base) as *const __m128i);
        let mut r1l = _mm_loadu_si128(workspace.as_ptr().add(row_base + 8) as *const __m128i);
        let mut r2l = _mm_loadu_si128(workspace.as_ptr().add(row_base + 16) as *const __m128i);
        let mut r3l = _mm_loadu_si128(workspace.as_ptr().add(row_base + 24) as *const __m128i);
        transpose4x4_sse2!(r0l, r1l, r2l, r3l);

        // Load right half (columns 4-7) and transpose
        let mut r0r = _mm_loadu_si128(workspace.as_ptr().add(row_base + 4) as *const __m128i);
        let mut r1r = _mm_loadu_si128(workspace.as_ptr().add(row_base + 8 + 4) as *const __m128i);
        let mut r2r = _mm_loadu_si128(workspace.as_ptr().add(row_base + 16 + 4) as *const __m128i);
        let mut r3r = _mm_loadu_si128(workspace.as_ptr().add(row_base + 24 + 4) as *const __m128i);
        transpose4x4_sse2!(r0r, r1r, r2r, r3r);

        let s0 = r0l;
        let s1 = r1l;
        let s2 = r2l;
        let s3 = r3l;
        let s4 = r0r;
        let s5 = r1r;
        let s6 = r2r;
        let s7 = r3r;

        let (t0, t1, t2, t3) = even_part_sse2(s0, s2, s4, s6, bias);
        let (o0, o1, o2, o3) = odd_part_sse2(s1, s3, s5, s7);

        // Combine, descale, and clamp to [0, 255]
        let clamp = |v: __m128i| -> __m128i {
            let shifted = _mm_srai_epi32::<{ 13 + 2 + 3 }>(v);
            // SSE2 i32 clamp to [0, 255] via comparison + bitwise select
            // max(shifted, 0)
            let gt_zero = _mm_cmpgt_epi32(shifted, _mm_set1_epi32(-1));
            let clamped_low = _mm_and_si128(shifted, gt_zero);
            // min(clamped_low, 255)
            let lt_256 = _mm_cmplt_epi32(clamped_low, _mm_set1_epi32(256));
            _mm_or_si128(
                _mm_and_si128(clamped_low, lt_256),
                _mm_andnot_si128(lt_256, max255),
            )
        };

        let out0 = clamp(_mm_add_epi32(t0, o3));
        let out7 = clamp(_mm_sub_epi32(t0, o3));
        let out1 = clamp(_mm_add_epi32(t1, o2));
        let out6 = clamp(_mm_sub_epi32(t1, o2));
        let out2 = clamp(_mm_add_epi32(t2, o1));
        let out5 = clamp(_mm_sub_epi32(t2, o1));
        let out3 = clamp(_mm_add_epi32(t3, o0));
        let out4 = clamp(_mm_sub_epi32(t3, o0));

        // Write output via temp array (same approach as NEON)
        let mut tmp = [0i32; 32];
        _mm_storeu_si128(tmp.as_mut_ptr() as *mut __m128i, out0);
        _mm_storeu_si128(tmp.as_mut_ptr().add(4) as *mut __m128i, out1);
        _mm_storeu_si128(tmp.as_mut_ptr().add(8) as *mut __m128i, out2);
        _mm_storeu_si128(tmp.as_mut_ptr().add(12) as *mut __m128i, out3);
        _mm_storeu_si128(tmp.as_mut_ptr().add(16) as *mut __m128i, out4);
        _mm_storeu_si128(tmp.as_mut_ptr().add(20) as *mut __m128i, out5);
        _mm_storeu_si128(tmp.as_mut_ptr().add(24) as *mut __m128i, out6);
        _mm_storeu_si128(tmp.as_mut_ptr().add(28) as *mut __m128i, out7);

        for local_row in 0..4 {
            let row_idx = row_group * 4 + local_row;
            let out_ptr = output.as_mut_ptr().add(row_idx * stride);
            for col in 0..8 {
                *out_ptr.add(col) = tmp[col * 4 + local_row] as u8;
            }
        }
    }
}

/// SSE2 DC-only IDCT: fill 8×8 block with a single value.
pub fn idct_dc_only_sse2(dc_value: i32, output: &mut [u8], stride: usize) {
    let range_shift = CONST_BITS + PASS1_BITS + 3;
    let round = 1 << (range_shift - 1);
    let bias = round + (128 << range_shift);
    let scaled = dc_value << (CONST_BITS + PASS1_BITS);
    let val = ((scaled + bias) >> range_shift).clamp(0, 255) as u8;

    unsafe {
        let fill = _mm_set1_epi8(val as i8);
        for row in 0..8 {
            // Store 8 bytes (low 64 bits of the 128-bit register)
            _mm_storel_epi64(output.as_mut_ptr().add(row * stride) as *mut __m128i, fill);
        }
    }
}
