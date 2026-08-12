// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! AVX2 8×8 IDCT.
//!
//! The islow kernel in [`super::sse2`] works in 16-bit lanes, so a whole row of
//! eight coefficients already fills a 128-bit register. Widening a *single*
//! block to 256-bit lanes would put two rows in one register and pay for the
//! lane-crossing shuffles that AVX2 charges for, which is why the previous
//! 32-bit-lane AVX2 kernel lost to libjpeg-turbo's plain SSE2 one.
//!
//! What AVX2 does buy here is the three-operand VEX encoding: recompiling the
//! same kernel with `avx2` enabled removes the register-copy `movdqa`s that the
//! destructive two-operand SSE encoding forces. Genuine 256-bit width needs two
//! blocks side by side, one per 128-bit half, which requires a two-block entry
//! point in the MCU loop rather than a wider kernel here.

/// AVX2 8×8 IDCT with in-kernel dequantisation, bit-exact with the scalar path.
pub fn idct_8x8_avx2(coeffs: &[i16; 64], quant: &[u16; 64], output: &mut [u8], stride: usize) {
    assert!(output.len() >= 7 * stride + 8, "IDCT output too small");
    // SAFETY: gated by `select_idct` on the AVX2 feature probe; the assert
    // covers the kernel's eight 8-byte row stores.
    unsafe { idct_8x8_avx2_inner(coeffs, quant, output, stride) }
}

#[target_feature(enable = "avx2")]
unsafe fn idct_8x8_avx2_inner(
    coeffs: &[i16; 64],
    quant: &[u16; 64],
    output: &mut [u8],
    stride: usize,
) {
    super::sse2::islow_body(coeffs, quant, output, stride)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::jpeg::idct::scalar::idct_8x8_scalar;

    /// The AVX2 build of the kernel must stay bit-exact with the scalar
    /// reference, including the clamp at both ends and on a strided
    /// destination, and must not write past column 8.
    #[test]
    fn idct_8x8_parity_random_strided() {
        if !is_x86_feature_detected!("avx2") {
            eprintln!("AVX2 not available, skipping");
            return;
        }
        const STRIDE: usize = 24;
        let mut seed = 0x1234_5678u32;
        let mut next = move || {
            seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            seed
        };

        for case in 0..256 {
            let mut coeffs = [0i16; 64];
            for (i, c) in coeffs.iter_mut().enumerate() {
                // Later cases push hard on the [0, 255] clamp. Dequantised
                // values stay inside the i16 range the islow IDCT assumes.
                let mag = if case < 64 { 32 } else { 256 };
                let v = (next() % (2 * mag)) as i32 - mag as i32;
                *c = if i < 16 || next().is_multiple_of(4) {
                    v as i16
                } else {
                    0
                };
            }
            let mut quant = [1u16; 64];
            for q in quant.iter_mut() {
                *q = (next() % 4 + 1) as u16;
            }

            let mut want = [0xAAu8; 8 * STRIDE];
            let mut got = [0xAAu8; 8 * STRIDE];
            idct_8x8_scalar(&coeffs, &quant, &mut want, STRIDE);
            idct_8x8_avx2(&coeffs, &quant, &mut got, STRIDE);

            assert_eq!(got, want, "case {case} diverged from scalar reference");
        }
    }
}
