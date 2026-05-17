// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! Scalar YCbCr → RGB/RGBA/BGRA/Grey color conversion.
//!
//! BT.601 full-range coefficients using 16-bit fixed-point arithmetic.

/// Fixed-point shift for color conversion.
const FIX: i32 = 16;
const HALF: i32 = 1 << (FIX - 1);

/// BT.601 full-range constants (scaled to 16-bit fixed-point).
/// R = Y + 1.402 * (Cr - 128)
/// G = Y - 0.34414 * (Cb - 128) - 0.71414 * (Cr - 128)
/// B = Y + 1.772 * (Cb - 128)
const CR_TO_R: i32 = 91881; // 1.402 * 2^16
const CB_TO_G: i32 = -22554; // -0.34414 * 2^16
const CR_TO_G: i32 = -46802; // -0.71414 * 2^16
const CB_TO_B: i32 = 116130; // 1.772 * 2^16

#[inline]
fn clamp_u8(v: i32) -> u8 {
    v.clamp(0, 255) as u8
}

/// Convert one pixel from YCbCr to (R, G, B).
#[inline]
fn ycbcr_to_rgb_pixel(y: u8, cb: u8, cr: u8) -> (u8, u8, u8) {
    let y = y as i32;
    let cb = cb as i32 - 128;
    let cr = cr as i32 - 128;

    let r = y + ((cr * CR_TO_R + HALF) >> FIX);
    let g = y + ((cb * CB_TO_G + cr * CR_TO_G + HALF) >> FIX);
    let b = y + ((cb * CB_TO_B + HALF) >> FIX);

    (clamp_u8(r), clamp_u8(g), clamp_u8(b))
}

/// YCbCr → RGB packed.
pub fn ycbcr_to_rgb(y_row: &[u8], cb_row: &[u8], cr_row: &[u8], output: &mut [u8], width: usize) {
    for i in 0..width {
        let (r, g, b) = ycbcr_to_rgb_pixel(y_row[i], cb_row[i], cr_row[i]);
        output[i * 3] = r;
        output[i * 3 + 1] = g;
        output[i * 3 + 2] = b;
    }
}

/// YCbCr → RGBA packed (alpha = 255).
pub fn ycbcr_to_rgba(y_row: &[u8], cb_row: &[u8], cr_row: &[u8], output: &mut [u8], width: usize) {
    for i in 0..width {
        let (r, g, b) = ycbcr_to_rgb_pixel(y_row[i], cb_row[i], cr_row[i]);
        output[i * 4] = r;
        output[i * 4 + 1] = g;
        output[i * 4 + 2] = b;
        output[i * 4 + 3] = 255;
    }
}

/// YCbCr → BGRA packed (alpha = 255).
pub fn ycbcr_to_bgra(y_row: &[u8], cb_row: &[u8], cr_row: &[u8], output: &mut [u8], width: usize) {
    for i in 0..width {
        let (r, g, b) = ycbcr_to_rgb_pixel(y_row[i], cb_row[i], cr_row[i]);
        output[i * 4] = b;
        output[i * 4 + 1] = g;
        output[i * 4 + 2] = r;
        output[i * 4 + 3] = 255;
    }
}

/// Greyscale copy: Y channel only.
pub fn grey_copy(y_row: &[u8], output: &mut [u8], width: usize) {
    output[..width].copy_from_slice(&y_row[..width]);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn white_pixel() {
        let (r, g, b) = ycbcr_to_rgb_pixel(255, 128, 128);
        assert_eq!((r, g, b), (255, 255, 255));
    }

    #[test]
    fn black_pixel() {
        let (r, g, b) = ycbcr_to_rgb_pixel(0, 128, 128);
        assert_eq!((r, g, b), (0, 0, 0));
    }

    #[test]
    fn red_pixel() {
        // Pure red in YCbCr (BT.601): Y≈76, Cb≈85, Cr≈255
        let (r, g, b) = ycbcr_to_rgb_pixel(76, 85, 255);
        // Should be close to (255, 0, 0) — allow ±2 for rounding
        assert!((r as i32 - 255).abs() <= 2, "r={r}");
        assert!((g as i32).abs() <= 2, "g={g}");
        assert!((b as i32).abs() <= 2, "b={b}");
    }

    #[test]
    fn clamping() {
        // Extreme values should not panic and should produce valid pixels
        let (r, g, b) = ycbcr_to_rgb_pixel(255, 0, 255);
        // Values are u8, so they are inherently in [0, 255].
        // Just verify we get reasonable output without panicking.
        let _ = (r, g, b);
    }

    #[test]
    fn grey_copy_works() {
        let y = [10, 20, 30, 40];
        let mut out = [0u8; 4];
        grey_copy(&y, &mut out, 4);
        assert_eq!(out, [10, 20, 30, 40]);
    }

    #[test]
    fn rgb_row_conversion() {
        let y = [128, 128];
        let cb = [128, 128];
        let cr = [128, 128];
        let mut out = [0u8; 6];
        ycbcr_to_rgb(&y, &cb, &cr, &mut out, 2);
        // Y=128, Cb=128, Cr=128 → (128, 128, 128)
        assert_eq!(out, [128, 128, 128, 128, 128, 128]);
    }
}
