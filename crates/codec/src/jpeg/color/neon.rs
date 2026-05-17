// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! NEON-optimised YCbCr color conversion.
//!
//! Placeholder: delegates to scalar. Full NEON implementation with merged
//! upsample+convert and vst3q/vst4q output will be added after scalar validation.

/// NEON YCbCr → RGB (currently delegates to scalar).
pub fn ycbcr_to_rgb_neon(
    y_row: &[u8],
    cb_row: &[u8],
    cr_row: &[u8],
    output: &mut [u8],
    width: usize,
) {
    super::scalar::ycbcr_to_rgb(y_row, cb_row, cr_row, output, width);
}

/// NEON YCbCr → RGBA (currently delegates to scalar).
pub fn ycbcr_to_rgba_neon(
    y_row: &[u8],
    cb_row: &[u8],
    cr_row: &[u8],
    output: &mut [u8],
    width: usize,
) {
    super::scalar::ycbcr_to_rgba(y_row, cb_row, cr_row, output, width);
}
