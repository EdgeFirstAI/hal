// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! NEON-optimised YCbCr color conversion.
//!
//! Processes 16 pixels per iteration using NEON SIMD intrinsics.
//! BT.601 full-range with 16-bit fixed-point arithmetic.
//!
//! The approach:
//! 1. Load 16 Y/Cb/Cr values
//! 2. Widen to i16, subtract 128 from Cb/Cr
//! 3. Fixed-point multiply-accumulate for R/G/B
//! 4. Clamp to [0, 255] via unsigned saturation
//! 5. Interleaved store via vst3q/vst4q

use std::arch::aarch64::*;

/// BT.601 full-range constants scaled to fit i16 multiply (shift 6 not 16).
/// Using 7-bit fractional precision to keep intermediates in i16 range.
///
/// We use a different strategy: widen to i32 for the multiplies, then narrow.
/// Constants at 16-bit precision (>> 16 shift at end).
/// Actually, for NEON we use a simpler approach: 8-bit fixed-point with
/// widening multiply.
///
/// R = Y + 1.402 * (Cr - 128)
/// G = Y - 0.34414 * (Cb - 128) - 0.71414 * (Cr - 128)
/// B = Y + 1.772 * (Cb - 128)
///
/// Using 7-bit fixed point (multiply by 128):
///  CR_TO_R = round(1.402 * 128) = 179
///  CB_TO_G = round(-0.34414 * 128) = -44
///  CR_TO_G = round(-0.71414 * 128) = -91
///  CB_TO_B = round(1.772 * 128) = 227
const CR_TO_R_7: i16 = 179;
const CB_TO_G_7: i16 = -44;
const CR_TO_G_7: i16 = -91;
const CB_TO_B_7: i16 = 227;

/// NEON YCbCr → RGB packed (3 bytes per pixel).
pub fn ycbcr_to_rgb_neon(
    y_row: &[u8],
    cb_row: &[u8],
    cr_row: &[u8],
    output: &mut [u8],
    width: usize,
) {
    unsafe { ycbcr_to_rgb_neon_inner(y_row, cb_row, cr_row, output, width) }
}

#[target_feature(enable = "neon")]
unsafe fn ycbcr_to_rgb_neon_inner(
    y_row: &[u8],
    cb_row: &[u8],
    cr_row: &[u8],
    output: &mut [u8],
    width: usize,
) {
    let chunks = width / 8;

    let cr_r = vdupq_n_s16(CR_TO_R_7);
    let cb_g = vdupq_n_s16(CB_TO_G_7);
    let cr_g = vdupq_n_s16(CR_TO_G_7);
    let cb_b = vdupq_n_s16(CB_TO_B_7);
    let c128 = vdupq_n_s16(128);

    for i in 0..chunks {
        let base = i * 8;

        // Load 8 Y/Cb/Cr samples
        let y_u8 = vld1_u8(y_row.as_ptr().add(base));
        let cb_u8 = vld1_u8(cb_row.as_ptr().add(base));
        let cr_u8 = vld1_u8(cr_row.as_ptr().add(base));

        // Widen to i16
        let y16 = vreinterpretq_s16_u16(vmovl_u8(y_u8));
        let cb16 = vsubq_s16(vreinterpretq_s16_u16(vmovl_u8(cb_u8)), c128);
        let cr16 = vsubq_s16(vreinterpretq_s16_u16(vmovl_u8(cr_u8)), c128);

        // R = Y + (Cr * 179 + 64) >> 7
        let r_offset = vrshrq_n_s16::<7>(vmulq_s16(cr16, cr_r));
        let r16 = vaddq_s16(y16, r_offset);

        // G = Y + ((Cb * -44) + (Cr * -91) + 64) >> 7
        let g_offset = vrshrq_n_s16::<7>(vaddq_s16(vmulq_s16(cb16, cb_g), vmulq_s16(cr16, cr_g)));
        let g16 = vaddq_s16(y16, g_offset);

        // B = Y + (Cb * 227 + 64) >> 7
        let b_offset = vrshrq_n_s16::<7>(vmulq_s16(cb16, cb_b));
        let b16 = vaddq_s16(y16, b_offset);

        // Clamp to [0, 255] via unsigned saturating narrowing
        let r8 = vqmovun_s16(r16);
        let g8 = vqmovun_s16(g16);
        let b8 = vqmovun_s16(b16);

        // Interleaved RGB store
        let rgb = uint8x8x3_t(r8, g8, b8);
        vst3_u8(output.as_mut_ptr().add(base * 3), rgb);
    }

    // Handle remaining pixels with scalar
    let start = chunks * 8;
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

/// NEON YCbCr → RGBA packed (4 bytes per pixel, alpha = 255).
pub fn ycbcr_to_rgba_neon(
    y_row: &[u8],
    cb_row: &[u8],
    cr_row: &[u8],
    output: &mut [u8],
    width: usize,
) {
    unsafe { ycbcr_to_rgba_neon_inner(y_row, cb_row, cr_row, output, width) }
}

#[target_feature(enable = "neon")]
unsafe fn ycbcr_to_rgba_neon_inner(
    y_row: &[u8],
    cb_row: &[u8],
    cr_row: &[u8],
    output: &mut [u8],
    width: usize,
) {
    let chunks = width / 8;

    let cr_r = vdupq_n_s16(CR_TO_R_7);
    let cb_g = vdupq_n_s16(CB_TO_G_7);
    let cr_g = vdupq_n_s16(CR_TO_G_7);
    let cb_b = vdupq_n_s16(CB_TO_B_7);
    let c128 = vdupq_n_s16(128);
    let alpha = vdup_n_u8(255);

    for i in 0..chunks {
        let base = i * 8;

        let y_u8 = vld1_u8(y_row.as_ptr().add(base));
        let cb_u8 = vld1_u8(cb_row.as_ptr().add(base));
        let cr_u8 = vld1_u8(cr_row.as_ptr().add(base));

        let y16 = vreinterpretq_s16_u16(vmovl_u8(y_u8));
        let cb16 = vsubq_s16(vreinterpretq_s16_u16(vmovl_u8(cb_u8)), c128);
        let cr16 = vsubq_s16(vreinterpretq_s16_u16(vmovl_u8(cr_u8)), c128);

        let r_offset = vrshrq_n_s16::<7>(vmulq_s16(cr16, cr_r));
        let r16 = vaddq_s16(y16, r_offset);

        let g_offset = vrshrq_n_s16::<7>(vaddq_s16(vmulq_s16(cb16, cb_g), vmulq_s16(cr16, cr_g)));
        let g16 = vaddq_s16(y16, g_offset);

        let b_offset = vrshrq_n_s16::<7>(vmulq_s16(cb16, cb_b));
        let b16 = vaddq_s16(y16, b_offset);

        let r8 = vqmovun_s16(r16);
        let g8 = vqmovun_s16(g16);
        let b8 = vqmovun_s16(b16);

        let rgba = uint8x8x4_t(r8, g8, b8, alpha);
        vst4_u8(output.as_mut_ptr().add(base * 4), rgba);
    }

    // Scalar tail
    let start = chunks * 8;
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

/// NEON YCbCr → BGRA packed (4 bytes per pixel, alpha = 255).
pub fn ycbcr_to_bgra_neon(
    y_row: &[u8],
    cb_row: &[u8],
    cr_row: &[u8],
    output: &mut [u8],
    width: usize,
) {
    unsafe { ycbcr_to_bgra_neon_inner(y_row, cb_row, cr_row, output, width) }
}

#[target_feature(enable = "neon")]
unsafe fn ycbcr_to_bgra_neon_inner(
    y_row: &[u8],
    cb_row: &[u8],
    cr_row: &[u8],
    output: &mut [u8],
    width: usize,
) {
    let chunks = width / 8;

    let cr_r = vdupq_n_s16(CR_TO_R_7);
    let cb_g = vdupq_n_s16(CB_TO_G_7);
    let cr_g = vdupq_n_s16(CR_TO_G_7);
    let cb_b = vdupq_n_s16(CB_TO_B_7);
    let c128 = vdupq_n_s16(128);
    let alpha = vdup_n_u8(255);

    for i in 0..chunks {
        let base = i * 8;

        let y_u8 = vld1_u8(y_row.as_ptr().add(base));
        let cb_u8 = vld1_u8(cb_row.as_ptr().add(base));
        let cr_u8 = vld1_u8(cr_row.as_ptr().add(base));

        let y16 = vreinterpretq_s16_u16(vmovl_u8(y_u8));
        let cb16 = vsubq_s16(vreinterpretq_s16_u16(vmovl_u8(cb_u8)), c128);
        let cr16 = vsubq_s16(vreinterpretq_s16_u16(vmovl_u8(cr_u8)), c128);

        let r_offset = vrshrq_n_s16::<7>(vmulq_s16(cr16, cr_r));
        let r16 = vaddq_s16(y16, r_offset);

        let g_offset = vrshrq_n_s16::<7>(vaddq_s16(vmulq_s16(cb16, cb_g), vmulq_s16(cr16, cr_g)));
        let g16 = vaddq_s16(y16, g_offset);

        let b_offset = vrshrq_n_s16::<7>(vmulq_s16(cb16, cb_b));
        let b16 = vaddq_s16(y16, b_offset);

        let r8 = vqmovun_s16(r16);
        let g8 = vqmovun_s16(g16);
        let b8 = vqmovun_s16(b16);

        // BGRA order: B, G, R, A
        let bgra = uint8x8x4_t(b8, g8, r8, alpha);
        vst4_u8(output.as_mut_ptr().add(base * 4), bgra);
    }

    // Scalar tail
    let start = chunks * 8;
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
