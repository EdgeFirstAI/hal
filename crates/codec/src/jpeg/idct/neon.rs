// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! NEON-optimised 8×8 IDCT.
//!
//! Placeholder: delegates to scalar. Full NEON implementation with vst4
//! transpose trick and sparsity bitmap will be added after scalar validation.

/// NEON 8×8 IDCT (currently delegates to scalar).
pub fn idct_8x8_neon(coeffs: &[i32; 64], output: &mut [u8], stride: usize) {
    super::scalar::idct_8x8_scalar(coeffs, output, stride);
}

/// NEON DC-only IDCT (currently delegates to scalar).
pub fn idct_dc_only_neon(dc_value: i32, output: &mut [u8], stride: usize) {
    super::scalar::idct_dc_only_scalar(dc_value, output, stride);
}
