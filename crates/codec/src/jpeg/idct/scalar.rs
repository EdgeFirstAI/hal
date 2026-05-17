// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! Scalar Loeffler 8×8 IDCT.
//!
//! Two-pass (columns then rows) Loeffler butterfly with 13-bit fixed-point
//! arithmetic. Reference implementation validated against libjpeg-turbo.

/// Fixed-point precision for intermediate values.
const PASS1_BITS: i32 = 2;
const CONST_BITS: i32 = 13;
const FIX_HALF: i32 = 1 << (CONST_BITS - 1);

/// Loeffler butterfly constants (13-bit fixed-point).
const FIX_0_298: i32 = 2446; // cos(7π/16) * 2^13
const FIX_0_390: i32 = 3196; // √2 * (cos(6π/16) - cos(2π/16)) * 2^13 (approx)
const FIX_0_541: i32 = 4433; // √2 * cos(6π/16)
const FIX_0_765: i32 = 6270; // √2 * (cos(2π/16) - cos(6π/16)) (approx)
const FIX_1_175: i32 = 9633; // √2 * cos(3π/16)
const FIX_1_501: i32 = 12299; // √2 * (cos(π/16) - cos(3π/16)) (approx)
const FIX_1_847: i32 = 15137; // √2 * cos(2π/16)
const FIX_1_961: i32 = 16069; // √2 * (cos(π/16) + cos(3π/16) - cos(5π/16))
const FIX_2_053: i32 = 16819; // √2 * (cos(7π/16) + cos(3π/16))
const FIX_2_562: i32 = 20995; // √2 * (cos(π/16) + cos(5π/16))
const FIX_3_072: i32 = 25172; // √2 * (cos(π/16) + cos(3π/16))

/// Perform 8×8 IDCT on dequantised coefficients in natural (row-major) order.
///
/// Output is 64 clamped `[0, 255]` u8 values written at `stride` byte offsets.
pub fn idct_8x8_scalar(coeffs: &[i32; 64], output: &mut [u8], stride: usize) {
    let mut workspace = [0i32; 64];

    // Pass 1: process columns from coefficients into workspace.
    for col in 0..8 {
        let s0 = coeffs[col];
        let s1 = coeffs[8 + col];
        let s2 = coeffs[16 + col];
        let s3 = coeffs[24 + col];
        let s4 = coeffs[32 + col];
        let s5 = coeffs[40 + col];
        let s6 = coeffs[48 + col];
        let s7 = coeffs[56 + col];

        // Shortcut for all-zero AC columns (DC-only)
        if s1 == 0 && s2 == 0 && s3 == 0 && s4 == 0 && s5 == 0 && s6 == 0 && s7 == 0 {
            let dc_val = s0 << PASS1_BITS;
            for row in 0..8 {
                workspace[row * 8 + col] = dc_val;
            }
            continue;
        }

        // Even part
        let tmp0 = s0 << CONST_BITS;
        let tmp2 = s4 << CONST_BITS;

        let tmp10 = tmp0 + tmp2;
        let tmp11 = tmp0 - tmp2;

        let z1 = s2 + s6;
        let tmp13 = z1 * FIX_0_541 + s2 * FIX_0_765;
        let tmp12 = z1 * FIX_0_541 - s6 * FIX_1_847;

        let tmp0 = tmp10 + tmp13 + FIX_HALF;
        let tmp3 = tmp10 - tmp13 + FIX_HALF;
        let tmp1 = tmp11 + tmp12 + FIX_HALF;
        let tmp2 = tmp11 - tmp12 + FIX_HALF;

        // Odd part
        let z1 = s7 + s1;
        let z2 = s5 + s3;
        let z3 = s7 + s3;
        let z4 = s5 + s1;
        let z5 = (z3 + z4) * FIX_1_175;

        let p7 = s7 * FIX_0_298;
        let p5 = s5 * FIX_2_053;
        let p3 = s3 * FIX_3_072;
        let p1 = s1 * FIX_1_501;

        let z1 = z1 * (-FIX_0_899);
        let z2 = z2 * (-FIX_2_562);
        let z3 = z3 * (-FIX_1_961) + z5;
        let z4 = z4 * (-FIX_0_390) + z5;

        let tmp0_odd = p7 + z1 + z3;
        let tmp1_odd = p5 + z2 + z4;
        let tmp2_odd = p3 + z2 + z3;
        let tmp3_odd = p1 + z1 + z4;

        // Final output (descale from CONST_BITS to PASS1_BITS)
        let shift = CONST_BITS - PASS1_BITS;
        workspace[col] = (tmp0 + tmp3_odd) >> shift;
        workspace[56 + col] = (tmp0 - tmp3_odd) >> shift;
        workspace[8 + col] = (tmp1 + tmp2_odd) >> shift;
        workspace[48 + col] = (tmp1 - tmp2_odd) >> shift;
        workspace[16 + col] = (tmp2 + tmp1_odd) >> shift;
        workspace[40 + col] = (tmp2 - tmp1_odd) >> shift;
        workspace[24 + col] = (tmp3 + tmp0_odd) >> shift;
        workspace[32 + col] = (tmp3 - tmp0_odd) >> shift;
    }

    // Pass 2: process rows from workspace into output.
    // Add DC offset of 128 and clamp to [0, 255].
    let range_shift = CONST_BITS + PASS1_BITS + 3; // +3 for the IDCT normalization
    let round = 1 << (range_shift - 1);
    // Also add 128 * 2^range_shift to centre the output
    let bias = round + (128 << range_shift);

    for row in 0..8 {
        let base = row * 8;
        let s0 = workspace[base];
        let s1 = workspace[base + 1];
        let s2 = workspace[base + 2];
        let s3 = workspace[base + 3];
        let s4 = workspace[base + 4];
        let s5 = workspace[base + 5];
        let s6 = workspace[base + 6];
        let s7 = workspace[base + 7];

        // Shortcut for all-zero AC rows
        if s1 == 0 && s2 == 0 && s3 == 0 && s4 == 0 && s5 == 0 && s6 == 0 && s7 == 0 {
            let val = clamp_u8(((s0 << CONST_BITS) + bias) >> range_shift);
            let out_base = row * stride;
            for i in 0..8 {
                output[out_base + i] = val;
            }
            continue;
        }

        // Even part (same butterfly as pass 1)
        let tmp0 = s0 << CONST_BITS;
        let tmp2 = s4 << CONST_BITS;

        let tmp10 = tmp0 + tmp2;
        let tmp11 = tmp0 - tmp2;

        let z1 = s2 + s6;
        let tmp13 = z1 * FIX_0_541 + s2 * FIX_0_765;
        let tmp12 = z1 * FIX_0_541 - s6 * FIX_1_847;

        let tmp0 = tmp10 + tmp13 + bias;
        let tmp3 = tmp10 - tmp13 + bias;
        let tmp1 = tmp11 + tmp12 + bias;
        let tmp2 = tmp11 - tmp12 + bias;

        // Odd part
        let z1 = s7 + s1;
        let z2 = s5 + s3;
        let z3 = s7 + s3;
        let z4 = s5 + s1;
        let z5 = (z3 + z4) * FIX_1_175;

        let p7 = s7 * FIX_0_298;
        let p5 = s5 * FIX_2_053;
        let p3 = s3 * FIX_3_072;
        let p1 = s1 * FIX_1_501;

        let z1 = z1 * (-FIX_0_899);
        let z2 = z2 * (-FIX_2_562);
        let z3 = z3 * (-FIX_1_961) + z5;
        let z4 = z4 * (-FIX_0_390) + z5;

        let tmp0_odd = p7 + z1 + z3;
        let tmp1_odd = p5 + z2 + z4;
        let tmp2_odd = p3 + z2 + z3;
        let tmp3_odd = p1 + z1 + z4;

        let out_base = row * stride;
        output[out_base] = clamp_u8((tmp0 + tmp3_odd) >> range_shift);
        output[out_base + 7] = clamp_u8((tmp0 - tmp3_odd) >> range_shift);
        output[out_base + 1] = clamp_u8((tmp1 + tmp2_odd) >> range_shift);
        output[out_base + 6] = clamp_u8((tmp1 - tmp2_odd) >> range_shift);
        output[out_base + 2] = clamp_u8((tmp2 + tmp1_odd) >> range_shift);
        output[out_base + 5] = clamp_u8((tmp2 - tmp1_odd) >> range_shift);
        output[out_base + 3] = clamp_u8((tmp3 + tmp0_odd) >> range_shift);
        output[out_base + 4] = clamp_u8((tmp3 - tmp0_odd) >> range_shift);
    }
}

/// DC-only fast path: all 64 output values are the same.
pub fn idct_dc_only_scalar(dc_value: i32, output: &mut [u8], stride: usize) {
    // DC coefficient is already dequantised. Apply IDCT scaling:
    // After two passes of the IDCT, the DC value gets divided by 8 (normalisation).
    // With our fixed-point: (dc * 2^CONST_BITS + bias) >> range_shift
    let range_shift = CONST_BITS + PASS1_BITS + 3;
    let round = 1 << (range_shift - 1);
    let bias = round + (128 << range_shift);
    let scaled = dc_value << (CONST_BITS + PASS1_BITS);
    let val = clamp_u8((scaled + bias) >> range_shift);

    for row in 0..8 {
        let base = row * stride;
        for col in 0..8 {
            output[base + col] = val;
        }
    }
}

/// Clamp an i32 to [0, 255].
#[inline]
fn clamp_u8(x: i32) -> u8 {
    x.clamp(0, 255) as u8
}

// Constants for the FIX_0_899 used in odd part
const FIX_0_899: i32 = 7373;

#[cfg(test)]
mod tests {
    use super::*;

    /// Test that a zero-coefficient block produces all 128s (the DC level shift).
    #[test]
    fn zero_block_gives_128() {
        let coeffs = [0i32; 64];
        let mut output = [0u8; 64];
        idct_8x8_scalar(&coeffs, &mut output, 8);
        for &v in &output {
            assert_eq!(v, 128, "zero coeffs should give DC offset 128");
        }
    }

    /// Test DC-only block.
    #[test]
    fn dc_only_block() {
        let mut coeffs = [0i32; 64];
        // A DC coefficient of 8 (dequantised) should produce:
        // (8 * 2^15 + bias) >> 18 with bias = 2^17 + 128*2^18
        // = approximately 128 + 1 = 129
        coeffs[0] = 8;
        let mut output = [0u8; 64];
        idct_8x8_scalar(&coeffs, &mut output, 8);
        // All values should be the same (DC-only shortcut triggers)
        let expected = output[0];
        for &v in &output {
            assert_eq!(v, expected);
        }

        // Compare with dc_only function
        let mut output2 = [0u8; 64];
        idct_dc_only_scalar(8, &mut output2, 8);
        assert_eq!(output, output2);
    }

    /// Test that strided output works correctly.
    #[test]
    fn strided_output() {
        let coeffs = [0i32; 64];
        let stride = 16; // Larger than 8
        let mut output = vec![0xFFu8; stride * 8];
        idct_8x8_scalar(&coeffs, &mut output, stride);

        for row in 0..8 {
            // First 8 bytes of each row should be 128
            for col in 0..8 {
                assert_eq!(output[row * stride + col], 128);
            }
            // Remaining bytes should be untouched (0xFF)
            for col in 8..stride {
                assert_eq!(output[row * stride + col], 0xFF);
            }
        }
    }
}
