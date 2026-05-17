// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! IDCT dispatcher — selects scalar, NEON, SSE4.1, or SSE2 implementation.

pub mod scalar;

#[cfg(target_arch = "aarch64")]
pub mod neon;

#[cfg(target_arch = "x86_64")]
pub mod sse2;

#[cfg(target_arch = "x86_64")]
pub mod sse41;

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
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("sse4.1") {
            return sse41::idct_8x8_sse41;
        }
        if is_x86_feature_detected!("sse2") {
            return sse2::idct_8x8_sse2;
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
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("sse4.1") {
            return sse41::idct_dc_only_sse41;
        }
        if is_x86_feature_detected!("sse2") {
            return sse2::idct_dc_only_sse2;
        }
    }
    scalar::idct_dc_only_scalar
}
