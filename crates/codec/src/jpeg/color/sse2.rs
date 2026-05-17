// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! SSE2-optimised YCbCr color conversion.
//!
//! Processes 8 pixels per iteration using SSE2 16-bit fixed-point arithmetic.
//! Uses the same 7-bit precision constants as the NEON path.
//!
//! Strategy:
//! 1. Load 8 Y/Cb/Cr values via `_mm_loadl_epi64` (64-bit load)
//! 2. Zero-extend u8→u16 via `_mm_unpacklo_epi8`
//! 3. Fixed-point multiply-accumulate for R/G/B
//! 4. Clamp to [0,255] via `_mm_packus_epi16` (saturating unsigned pack)
//! 5. Store interleaved pixels

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

/// BT.601 full-range constants at 7-bit fixed-point (must match NEON).
const CR_TO_R_7: i16 = 179;
const CB_TO_G_7: i16 = -44;
const CR_TO_G_7: i16 = -91;
const CB_TO_B_7: i16 = 227;

/// SSE2 YCbCr → RGB packed (3 bytes per pixel).
pub fn ycbcr_to_rgb_sse2(
    y_row: &[u8],
    cb_row: &[u8],
    cr_row: &[u8],
    output: &mut [u8],
    width: usize,
) {
    unsafe { ycbcr_to_rgb_sse2_inner(y_row, cb_row, cr_row, output, width) }
}

#[target_feature(enable = "sse2")]
unsafe fn ycbcr_to_rgb_sse2_inner(
    y_row: &[u8],
    cb_row: &[u8],
    cr_row: &[u8],
    output: &mut [u8],
    width: usize,
) {
    let chunks = width / 8;
    let zero = _mm_setzero_si128();
    let c128 = _mm_set1_epi16(128);
    let round = _mm_set1_epi16(64); // 1 << (7-1) for rounding
    let cr_r = _mm_set1_epi16(CR_TO_R_7);
    let cb_g = _mm_set1_epi16(CB_TO_G_7);
    let cr_g = _mm_set1_epi16(CR_TO_G_7);
    let cb_b = _mm_set1_epi16(CB_TO_B_7);

    for i in 0..chunks {
        let base = i * 8;

        // Load 8 u8 values (64-bit load into low half of __m128i)
        let y_u8 = _mm_loadl_epi64(y_row.as_ptr().add(base) as *const __m128i);
        let cb_u8 = _mm_loadl_epi64(cb_row.as_ptr().add(base) as *const __m128i);
        let cr_u8 = _mm_loadl_epi64(cr_row.as_ptr().add(base) as *const __m128i);

        // Zero-extend u8 → i16
        let y16 = _mm_unpacklo_epi8(y_u8, zero);
        let cb16 = _mm_sub_epi16(_mm_unpacklo_epi8(cb_u8, zero), c128);
        let cr16 = _mm_sub_epi16(_mm_unpacklo_epi8(cr_u8, zero), c128);

        // R = Y + (Cr * 179 + 64) >> 7
        let r_offset = _mm_srai_epi16::<7>(_mm_add_epi16(_mm_mullo_epi16(cr16, cr_r), round));
        let r16 = _mm_add_epi16(y16, r_offset);

        // G = Y + ((Cb * -44 + Cr * -91) + 64) >> 7
        let g_offset = _mm_srai_epi16::<7>(_mm_add_epi16(
            _mm_add_epi16(_mm_mullo_epi16(cb16, cb_g), _mm_mullo_epi16(cr16, cr_g)),
            round,
        ));
        let g16 = _mm_add_epi16(y16, g_offset);

        // B = Y + (Cb * 227 + 64) >> 7
        let b_offset = _mm_srai_epi16::<7>(_mm_add_epi16(_mm_mullo_epi16(cb16, cb_b), round));
        let b16 = _mm_add_epi16(y16, b_offset);

        // Clamp to [0, 255] via saturating unsigned pack
        let r8 = _mm_packus_epi16(r16, zero); // low 8 bytes = clamped R
        let g8 = _mm_packus_epi16(g16, zero);
        let b8 = _mm_packus_epi16(b16, zero);

        // Interleave and store RGB (24 bytes for 8 pixels)
        // SSE2 doesn't have vst3 — manually write interleaved bytes
        store_rgb_interleaved(r8, g8, b8, output.as_mut_ptr().add(base * 3));
    }

    // Scalar tail
    scalar_tail_rgb(y_row, cb_row, cr_row, output, chunks * 8, width);
}

/// SSE2 YCbCr → RGBA packed (4 bytes per pixel, alpha = 255).
pub fn ycbcr_to_rgba_sse2(
    y_row: &[u8],
    cb_row: &[u8],
    cr_row: &[u8],
    output: &mut [u8],
    width: usize,
) {
    unsafe { ycbcr_to_rgba_sse2_inner(y_row, cb_row, cr_row, output, width) }
}

#[target_feature(enable = "sse2")]
unsafe fn ycbcr_to_rgba_sse2_inner(
    y_row: &[u8],
    cb_row: &[u8],
    cr_row: &[u8],
    output: &mut [u8],
    width: usize,
) {
    let chunks = width / 8;
    let zero = _mm_setzero_si128();
    let c128 = _mm_set1_epi16(128);
    let round = _mm_set1_epi16(64);
    let cr_r = _mm_set1_epi16(CR_TO_R_7);
    let cb_g = _mm_set1_epi16(CB_TO_G_7);
    let cr_g = _mm_set1_epi16(CR_TO_G_7);
    let cb_b = _mm_set1_epi16(CB_TO_B_7);
    let alpha = _mm_set1_epi8(-1i8); // 0xFF

    for i in 0..chunks {
        let base = i * 8;

        let y_u8 = _mm_loadl_epi64(y_row.as_ptr().add(base) as *const __m128i);
        let cb_u8 = _mm_loadl_epi64(cb_row.as_ptr().add(base) as *const __m128i);
        let cr_u8 = _mm_loadl_epi64(cr_row.as_ptr().add(base) as *const __m128i);

        let y16 = _mm_unpacklo_epi8(y_u8, zero);
        let cb16 = _mm_sub_epi16(_mm_unpacklo_epi8(cb_u8, zero), c128);
        let cr16 = _mm_sub_epi16(_mm_unpacklo_epi8(cr_u8, zero), c128);

        let r_offset = _mm_srai_epi16::<7>(_mm_add_epi16(_mm_mullo_epi16(cr16, cr_r), round));
        let r16 = _mm_add_epi16(y16, r_offset);

        let g_offset = _mm_srai_epi16::<7>(_mm_add_epi16(
            _mm_add_epi16(_mm_mullo_epi16(cb16, cb_g), _mm_mullo_epi16(cr16, cr_g)),
            round,
        ));
        let g16 = _mm_add_epi16(y16, g_offset);

        let b_offset = _mm_srai_epi16::<7>(_mm_add_epi16(_mm_mullo_epi16(cb16, cb_b), round));
        let b16 = _mm_add_epi16(y16, b_offset);

        let r8 = _mm_packus_epi16(r16, zero);
        let g8 = _mm_packus_epi16(g16, zero);
        let b8 = _mm_packus_epi16(b16, zero);

        // Interleave RGBA: unpack R,G → RG pairs, then B,A → BA pairs,
        // then RG,BA → RGBA quads
        store_rgba_interleaved(r8, g8, b8, alpha, output.as_mut_ptr().add(base * 4));
    }

    // Scalar tail
    scalar_tail_rgba(y_row, cb_row, cr_row, output, chunks * 8, width);
}

/// SSE2 YCbCr → BGRA packed (4 bytes per pixel, alpha = 255).
pub fn ycbcr_to_bgra_sse2(
    y_row: &[u8],
    cb_row: &[u8],
    cr_row: &[u8],
    output: &mut [u8],
    width: usize,
) {
    unsafe { ycbcr_to_bgra_sse2_inner(y_row, cb_row, cr_row, output, width) }
}

#[target_feature(enable = "sse2")]
unsafe fn ycbcr_to_bgra_sse2_inner(
    y_row: &[u8],
    cb_row: &[u8],
    cr_row: &[u8],
    output: &mut [u8],
    width: usize,
) {
    let chunks = width / 8;
    let zero = _mm_setzero_si128();
    let c128 = _mm_set1_epi16(128);
    let round = _mm_set1_epi16(64);
    let cr_r = _mm_set1_epi16(CR_TO_R_7);
    let cb_g = _mm_set1_epi16(CB_TO_G_7);
    let cr_g = _mm_set1_epi16(CR_TO_G_7);
    let cb_b = _mm_set1_epi16(CB_TO_B_7);
    let alpha = _mm_set1_epi8(-1i8);

    for i in 0..chunks {
        let base = i * 8;

        let y_u8 = _mm_loadl_epi64(y_row.as_ptr().add(base) as *const __m128i);
        let cb_u8 = _mm_loadl_epi64(cb_row.as_ptr().add(base) as *const __m128i);
        let cr_u8 = _mm_loadl_epi64(cr_row.as_ptr().add(base) as *const __m128i);

        let y16 = _mm_unpacklo_epi8(y_u8, zero);
        let cb16 = _mm_sub_epi16(_mm_unpacklo_epi8(cb_u8, zero), c128);
        let cr16 = _mm_sub_epi16(_mm_unpacklo_epi8(cr_u8, zero), c128);

        let r_offset = _mm_srai_epi16::<7>(_mm_add_epi16(_mm_mullo_epi16(cr16, cr_r), round));
        let r16 = _mm_add_epi16(y16, r_offset);

        let g_offset = _mm_srai_epi16::<7>(_mm_add_epi16(
            _mm_add_epi16(_mm_mullo_epi16(cb16, cb_g), _mm_mullo_epi16(cr16, cr_g)),
            round,
        ));
        let g16 = _mm_add_epi16(y16, g_offset);

        let b_offset = _mm_srai_epi16::<7>(_mm_add_epi16(_mm_mullo_epi16(cb16, cb_b), round));
        let b16 = _mm_add_epi16(y16, b_offset);

        let r8 = _mm_packus_epi16(r16, zero);
        let g8 = _mm_packus_epi16(g16, zero);
        let b8 = _mm_packus_epi16(b16, zero);

        // BGRA: swap r and b in the interleave
        store_rgba_interleaved(b8, g8, r8, alpha, output.as_mut_ptr().add(base * 4));
    }

    // Scalar tail
    scalar_tail_bgra(y_row, cb_row, cr_row, output, chunks * 8, width);
}

// ─── Store helpers ──────────────────────────────────────────────────────────

/// Interleave and store 8 RGBA pixels using SSE2.
///
/// Input: r, g, b each have 8 u8 values in low 8 bytes; alpha is broadcast.
/// Output: 32 bytes of RGBA data.
#[inline(always)]
unsafe fn store_rgba_interleaved(r8: __m128i, g8: __m128i, b8: __m128i, a8: __m128i, dst: *mut u8) {
    // Step 1: interleave R,G → [r0,g0, r1,g1, r2,g2, r3,g3, r4,g4, r5,g5, r6,g6, r7,g7]
    let rg = _mm_unpacklo_epi8(r8, g8);
    // Step 2: interleave B,A → [b0,a0, b1,a1, b2,a2, b3,a3, b4,a4, b5,a5, b6,a6, b7,a7]
    let ba = _mm_unpacklo_epi8(b8, a8);
    // Step 3: interleave 16-bit pairs → RGBA quads
    // Low: [r0,g0,b0,a0, r1,g1,b1,a1, r2,g2,b2,a2, r3,g3,b3,a3]
    let rgba_lo = _mm_unpacklo_epi16(rg, ba);
    // High: [r4,g4,b4,a4, r5,g5,b5,a5, r6,g6,b6,a6, r7,g7,b7,a7]
    let rgba_hi = _mm_unpackhi_epi16(rg, ba);

    _mm_storeu_si128(dst as *mut __m128i, rgba_lo);
    _mm_storeu_si128(dst.add(16) as *mut __m128i, rgba_hi);
}

/// Interleave and store 8 RGB pixels using SSE2.
///
/// Without SSSE3 `_mm_shuffle_epi8`, we extract bytes and write manually.
/// The SIMD speedup comes from the compute path, not the store.
#[inline(always)]
unsafe fn store_rgb_interleaved(r8: __m128i, g8: __m128i, b8: __m128i, dst: *mut u8) {
    // Extract individual bytes from the low 8 bytes of each register.
    // Use a small temp buffer for clean extraction.
    let mut r = [0u8; 8];
    let mut g = [0u8; 8];
    let mut b = [0u8; 8];
    _mm_storel_epi64(r.as_mut_ptr() as *mut __m128i, r8);
    _mm_storel_epi64(g.as_mut_ptr() as *mut __m128i, g8);
    _mm_storel_epi64(b.as_mut_ptr() as *mut __m128i, b8);

    for j in 0..8 {
        *dst.add(j * 3) = r[j];
        *dst.add(j * 3 + 1) = g[j];
        *dst.add(j * 3 + 2) = b[j];
    }
}

// ─── Scalar tails ───────────────────────────────────────────────────────────

#[inline(always)]
fn scalar_tail_rgb(
    y_row: &[u8],
    cb_row: &[u8],
    cr_row: &[u8],
    output: &mut [u8],
    start: usize,
    width: usize,
) {
    for j in start..width {
        let y = y_row[j] as i32;
        let cb = cb_row[j] as i32 - 128;
        let cr = cr_row[j] as i32 - 128;
        let r = (y + ((cr * CR_TO_R_7 as i32 + 64) >> 7)).clamp(0, 255) as u8;
        let g =
            (y + ((cb * CB_TO_G_7 as i32 + cr * CR_TO_G_7 as i32 + 64) >> 7)).clamp(0, 255) as u8;
        let b = (y + ((cb * CB_TO_B_7 as i32 + 64) >> 7)).clamp(0, 255) as u8;
        output[j * 3] = r;
        output[j * 3 + 1] = g;
        output[j * 3 + 2] = b;
    }
}

#[inline(always)]
fn scalar_tail_rgba(
    y_row: &[u8],
    cb_row: &[u8],
    cr_row: &[u8],
    output: &mut [u8],
    start: usize,
    width: usize,
) {
    for j in start..width {
        let y = y_row[j] as i32;
        let cb = cb_row[j] as i32 - 128;
        let cr = cr_row[j] as i32 - 128;
        let r = (y + ((cr * CR_TO_R_7 as i32 + 64) >> 7)).clamp(0, 255) as u8;
        let g =
            (y + ((cb * CB_TO_G_7 as i32 + cr * CR_TO_G_7 as i32 + 64) >> 7)).clamp(0, 255) as u8;
        let b = (y + ((cb * CB_TO_B_7 as i32 + 64) >> 7)).clamp(0, 255) as u8;
        output[j * 4] = r;
        output[j * 4 + 1] = g;
        output[j * 4 + 2] = b;
        output[j * 4 + 3] = 255;
    }
}

#[inline(always)]
fn scalar_tail_bgra(
    y_row: &[u8],
    cb_row: &[u8],
    cr_row: &[u8],
    output: &mut [u8],
    start: usize,
    width: usize,
) {
    for j in start..width {
        let y = y_row[j] as i32;
        let cb = cb_row[j] as i32 - 128;
        let cr = cr_row[j] as i32 - 128;
        let r = (y + ((cr * CR_TO_R_7 as i32 + 64) >> 7)).clamp(0, 255) as u8;
        let g =
            (y + ((cb * CB_TO_G_7 as i32 + cr * CR_TO_G_7 as i32 + 64) >> 7)).clamp(0, 255) as u8;
        let b = (y + ((cb * CB_TO_B_7 as i32 + 64) >> 7)).clamp(0, 255) as u8;
        output[j * 4] = b;
        output[j * 4 + 1] = g;
        output[j * 4 + 2] = r;
        output[j * 4 + 3] = 255;
    }
}
