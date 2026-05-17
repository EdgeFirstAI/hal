// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! SSSE3-optimised YCbCr → RGB color conversion.
//!
//! Replaces the SSE2 scalar scatter in `store_rgb_interleaved` with SSSE3
//! `_mm_shuffle_epi8` for 3-channel byte interleaving. The color math is
//! identical to the SSE2 path (16-bit fixed-point BT.601 full-range).
//!
//! Only the RGB path benefits — RGBA/BGRA already use efficient SSE2
//! unpack-based 4-channel interleave.

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

/// BT.601 full-range constants at 7-bit fixed-point (shared with SSE2).
const CR_TO_R_7: i16 = 179;
const CB_TO_G_7: i16 = -44;
const CR_TO_G_7: i16 = -91;
const CB_TO_B_7: i16 = 227;

/// SSSE3 YCbCr → RGB packed (3 bytes per pixel).
pub fn ycbcr_to_rgb_ssse3(
    y_row: &[u8],
    cb_row: &[u8],
    cr_row: &[u8],
    output: &mut [u8],
    width: usize,
) {
    unsafe { ycbcr_to_rgb_ssse3_inner(y_row, cb_row, cr_row, output, width) }
}

#[target_feature(enable = "ssse3")]
unsafe fn ycbcr_to_rgb_ssse3_inner(
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

    // SSSE3 shuffle masks for 3-channel interleave.
    // Input layout after merge: [r0,g0,b0,r1,g1,b1,r2,g2, b2,r3,g3,b3,r4,g4,b4,r5]
    //
    // We have r8,g8,b8 each with 8 bytes in the low half.
    // Strategy: merge into two 16-byte registers and shuffle.
    //
    // Register A = [r0..r7, g0..g7] (via unpacklo_epi64)
    // Register B = [b0..b7, 0..0]
    //
    // Shuffle A with mask_a1: pick r0,g0,_,r1,g1,_,r2,g2,_,r3,g3,_,r4,g4,_,r5
    // Shuffle B with mask_b1: pick _,_,b0,_,_,b1,_,_,b2,_,_,b3,_,_,b4,_
    // OR the two → first 16 bytes of output (covers pixels 0..4 + partial 5)
    //
    // Actually, the classic 8-pixel RGB pack from 3 separate channels uses:
    // Merge r,g into one reg, b into another, then two shuffle+OR rounds.

    // Mask for first 16 bytes: pixels 0-4 (15 bytes) + pixel 5 byte 0
    // From A=[r0,r1,r2,r3,r4,r5,r6,r7, g0,g1,g2,g3,g4,g5,g6,g7]:
    //   want: r0,_,_,r1,_,_,r2,_,_,r3,_,_,r4,_,_,r5
    //   idxs:  0, X, X, 1, X, X, 2, X, X, 3, X, X, 4, X, X, 5
    // From B=[b0,b1,b2,b3,b4,b5,b6,b7, 0,0,0,0,0,0,0,0]:
    //   want: _,_,b0,_,_,b1,_,_,b2,_,_,b3,_,_,b4,_
    //   idxs: X, X, 0,  X, X, 1, X, X, 2, X, X, 3, X, X, 4, X
    // G from A:
    //   want: _,g0,_,_,g1,_,_,g2,_,_,g3,_,_,g4,_,_
    //   idxs: X, 8, X, X, 9, X, X,10, X, X,11, X, X,12, X, X

    // Combined mask for A (R+G channels): index or 0x80 for "zero this byte"
    let mask_a1 = _mm_setr_epi8(
        0, 8, -128i8, // pixel 0: r0, g0, (b from B)
        1, 9, -128i8, // pixel 1: r1, g1, (b from B)
        2, 10, -128i8, // pixel 2: r2, g2, (b from B)
        3, 11, -128i8, // pixel 3: r3, g3, (b from B)
        4, 12, -128i8, // pixel 4: r4, g4, (b from B)
        5,      // pixel 5: r5 (partial)
    );

    // Mask for B (blue channel): index or 0x80 for "zero this byte"
    let mask_b1 = _mm_setr_epi8(
        -128i8, -128i8, 0, // pixel 0: (from A), (from A), b0
        -128i8, -128i8, 1, // pixel 1
        -128i8, -128i8, 2, // pixel 2
        -128i8, -128i8, 3, // pixel 3
        -128i8, -128i8, 4,      // pixel 4
        -128i8, // pixel 5: (r5 from A)
    );

    // Second 8 bytes: pixels 5 (g5,b5), 6, 7 = bytes 16..23
    // From A: g5=13, r6=6, g6=14, r7=7, g7=15
    // From B: b5=5, b6=6, b7=7
    let mask_a2 = _mm_setr_epi8(
        13, -128i8, // pixel 5 cont: g5, (b5 from B)
        6, 14, -128i8, // pixel 6: r6, g6, (b6 from B)
        7, 15, -128i8, // pixel 7: r7, g7, (b7 from B)
        // pad remaining 8 bytes (unused)
        -128i8, -128i8, -128i8, -128i8, -128i8, -128i8, -128i8, -128i8,
    );

    let mask_b2 = _mm_setr_epi8(
        -128i8, 5, // pixel 5 cont: (g5 from A), b5
        -128i8, -128i8, 6, // pixel 6: (from A), (from A), b6
        -128i8, -128i8, 7, // pixel 7: (from A), (from A), b7
        // pad remaining 8 bytes (unused)
        -128i8, -128i8, -128i8, -128i8, -128i8, -128i8, -128i8, -128i8,
    );

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

        // SSSE3 shuffle-based RGB interleave
        // A = [r0..r7, g0..g7]
        let a = _mm_unpacklo_epi64(r8, g8);

        // First 16 bytes: pixels 0-5 (partial)
        let out1 = _mm_or_si128(_mm_shuffle_epi8(a, mask_a1), _mm_shuffle_epi8(b8, mask_b1));

        // Second 8 bytes: pixels 5 (cont) through 7
        let out2 = _mm_or_si128(_mm_shuffle_epi8(a, mask_a2), _mm_shuffle_epi8(b8, mask_b2));

        let dst = output.as_mut_ptr().add(base * 3);
        _mm_storeu_si128(dst as *mut __m128i, out1);
        // Store only 8 bytes for the second part (avoids overwrite)
        _mm_storel_epi64(dst.add(16) as *mut __m128i, out2);
    }

    // Scalar tail
    for j in (chunks * 8)..width {
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

#[cfg(test)]
#[cfg(target_arch = "x86_64")]
mod tests {
    use super::{ycbcr_to_rgb_ssse3, CB_TO_B_7, CB_TO_G_7, CR_TO_G_7, CR_TO_R_7};

    fn make_ycbcr_row() -> ([u8; 16], [u8; 16], [u8; 16]) {
        let y: [u8; 16] = [
            16, 40, 64, 80, 100, 120, 128, 140, 160, 180, 200, 210, 220, 230, 235, 200,
        ];
        let cb: [u8; 16] = [
            128, 100, 80, 64, 150, 200, 128, 90, 110, 170, 210, 128, 60, 220, 128, 140,
        ];
        let cr: [u8; 16] = [
            128, 200, 60, 180, 100, 128, 220, 128, 90, 160, 128, 50, 200, 128, 170, 100,
        ];
        (y, cb, cr)
    }

    /// 7-bit fixed-point scalar reference matching SSSE3 arithmetic.
    fn scalar_rgb_7bit(y: u8, cb: u8, cr: u8) -> (u8, u8, u8) {
        let y = y as i32;
        let cb = cb as i32 - 128;
        let cr = cr as i32 - 128;
        let r = (y + ((cr * CR_TO_R_7 as i32 + 64) >> 7)).clamp(0, 255) as u8;
        let g =
            (y + ((cb * CB_TO_G_7 as i32 + cr * CR_TO_G_7 as i32 + 64) >> 7)).clamp(0, 255) as u8;
        let b = (y + ((cb * CB_TO_B_7 as i32 + 64) >> 7)).clamp(0, 255) as u8;
        (r, g, b)
    }

    #[test]
    fn ycbcr_to_rgb_parity() {
        if !is_x86_feature_detected!("ssse3") {
            eprintln!("SIMD feature not available, skipping");
            return;
        }

        let (y, cb, cr) = make_ycbcr_row();
        let width = 16usize;
        let mut simd_out = vec![0u8; width * 3];
        ycbcr_to_rgb_ssse3(&y, &cb, &cr, &mut simd_out, width);

        for i in 0..width {
            let (sr, sg, sb) = scalar_rgb_7bit(y[i], cb[i], cr[i]);
            let vr = simd_out[i * 3];
            let vg = simd_out[i * 3 + 1];
            let vb = simd_out[i * 3 + 2];
            let dr = (sr as i32 - vr as i32).abs();
            let dg = (sg as i32 - vg as i32).abs();
            let db = (sb as i32 - vb as i32).abs();
            assert!(
                dr <= 1,
                "RGB R parity mismatch at pixel {i}: scalar={sr}, simd={vr}, diff={dr}"
            );
            assert!(
                dg <= 1,
                "RGB G parity mismatch at pixel {i}: scalar={sg}, simd={vg}, diff={dg}"
            );
            assert!(
                db <= 1,
                "RGB B parity mismatch at pixel {i}: scalar={sb}, simd={vb}, diff={db}"
            );
        }
    }

    #[test]
    fn ycbcr_to_rgb_parity_partial_chunk() {
        if !is_x86_feature_detected!("ssse3") {
            eprintln!("SIMD feature not available, skipping");
            return;
        }

        // Width of 11: one full chunk of 8 + 3 scalar tail pixels
        let (y_full, cb_full, cr_full) = make_ycbcr_row();
        let width = 11usize;
        let y = &y_full[..width];
        let cb = &cb_full[..width];
        let cr = &cr_full[..width];
        let mut simd_out = vec![0u8; width * 3];
        ycbcr_to_rgb_ssse3(y, cb, cr, &mut simd_out, width);

        for i in 0..width {
            let (sr, sg, sb) = scalar_rgb_7bit(y[i], cb[i], cr[i]);
            let vr = simd_out[i * 3];
            let vg = simd_out[i * 3 + 1];
            let vb = simd_out[i * 3 + 2];
            let dr = (sr as i32 - vr as i32).abs();
            let dg = (sg as i32 - vg as i32).abs();
            let db = (sb as i32 - vb as i32).abs();
            assert!(
                dr <= 1,
                "partial RGB R parity mismatch at pixel {i}: scalar={sr}, simd={vr}, diff={dr}"
            );
            assert!(
                dg <= 1,
                "partial RGB G parity mismatch at pixel {i}: scalar={sg}, simd={vg}, diff={dg}"
            );
            assert!(
                db <= 1,
                "partial RGB B parity mismatch at pixel {i}: scalar={sb}, simd={vb}, diff={db}"
            );
        }
    }
}
