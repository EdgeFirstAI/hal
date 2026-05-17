// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! Chroma upsampling dispatcher.

pub mod scalar;

#[cfg(target_arch = "aarch64")]
pub mod neon;

#[cfg(target_arch = "x86_64")]
pub mod sse2;

/// Horizontal 2× upsample function: expands `width` samples to `2*width`.
pub type UpsampleHFn = fn(input: &[u8], output: &mut [u8], width: usize);

/// Select horizontal upsample.
pub fn select_upsample_h() -> UpsampleHFn {
    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            return neon::upsample_h2_neon;
        }
    }
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("sse2") {
            return sse2::upsample_h2_sse2;
        }
    }
    scalar::upsample_h2
}
