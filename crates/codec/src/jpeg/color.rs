// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! Color conversion dispatcher.

pub mod scalar;

#[cfg(target_arch = "aarch64")]
pub mod neon;

#[cfg(target_arch = "x86_64")]
pub mod sse2;

#[cfg(target_arch = "x86_64")]
pub mod ssse3;

use edgefirst_tensor::PixelFormat;

/// Color conversion function: converts one row of YCbCr component data to
/// packed output pixels.
///
/// Arguments: `(y_row, cb_row, cr_row, output, width)`
/// - `y_row`: luma samples for this row
/// - `cb_row`: chroma-blue samples (may be upsampled to full width already)
/// - `cr_row`: chroma-red samples (may be upsampled to full width already)
/// - `output`: destination buffer for packed pixels
/// - `width`: number of pixels to convert
pub type ColorConvertFn =
    fn(y_row: &[u8], cb_row: &[u8], cr_row: &[u8], output: &mut [u8], width: usize);

/// Greyscale copy: just copies Y channel.
pub type GreyCopyFn = fn(y_row: &[u8], output: &mut [u8], width: usize);

/// Select the best color conversion function for the target format.
pub fn select_color_convert(format: PixelFormat) -> Option<ColorConvertFn> {
    match format {
        PixelFormat::Rgb => {
            #[cfg(target_arch = "aarch64")]
            if std::arch::is_aarch64_feature_detected!("neon") {
                return Some(neon::ycbcr_to_rgb_neon);
            }
            #[cfg(target_arch = "x86_64")]
            {
                if is_x86_feature_detected!("ssse3") {
                    return Some(ssse3::ycbcr_to_rgb_ssse3);
                }
                if is_x86_feature_detected!("sse2") {
                    return Some(sse2::ycbcr_to_rgb_sse2);
                }
            }
            Some(scalar::ycbcr_to_rgb)
        }
        PixelFormat::Rgba => {
            #[cfg(target_arch = "aarch64")]
            if std::arch::is_aarch64_feature_detected!("neon") {
                return Some(neon::ycbcr_to_rgba_neon);
            }
            #[cfg(target_arch = "x86_64")]
            if is_x86_feature_detected!("sse2") {
                return Some(sse2::ycbcr_to_rgba_sse2);
            }
            Some(scalar::ycbcr_to_rgba)
        }
        PixelFormat::Bgra => {
            #[cfg(target_arch = "aarch64")]
            if std::arch::is_aarch64_feature_detected!("neon") {
                return Some(neon::ycbcr_to_bgra_neon);
            }
            #[cfg(target_arch = "x86_64")]
            if is_x86_feature_detected!("sse2") {
                return Some(sse2::ycbcr_to_bgra_sse2);
            }
            Some(scalar::ycbcr_to_bgra)
        }
        _ => None,
    }
}

/// Select greyscale copy function.
pub fn select_grey_copy() -> GreyCopyFn {
    scalar::grey_copy
}
