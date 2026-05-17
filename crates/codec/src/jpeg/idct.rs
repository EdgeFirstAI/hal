// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! IDCT dispatcher — selects scalar or NEON implementation.

pub mod scalar;

#[cfg(target_arch = "aarch64")]
pub mod neon;

/// IDCT function signature: takes 64 dequantised coefficients in natural
/// order, writes 64 clamped u8 values into `output` at the given stride.
pub type IdctFn = fn(coeffs: &[i32; 64], output: &mut [u8], stride: usize);

/// Select the best available IDCT implementation for this CPU.
pub fn select_idct() -> IdctFn {
    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            return neon::idct_8x8_neon;
        }
    }
    scalar::idct_8x8_scalar
}

/// IDCT function for DC-only blocks (all AC coefficients are zero).
pub type IdctDcOnlyFn = fn(dc_value: i32, output: &mut [u8], stride: usize);

/// Select DC-only IDCT.
pub fn select_idct_dc_only() -> IdctDcOnlyFn {
    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            return neon::idct_dc_only_neon;
        }
    }
    scalar::idct_dc_only_scalar
}
