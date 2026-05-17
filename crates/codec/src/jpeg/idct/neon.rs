// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! NEON-optimised 8×8 IDCT.
//!
//! Uses 32-bit integer arithmetic with the same Loeffler butterfly as the
//! scalar path, but processes 4 columns/rows at a time using NEON SIMD.
//! Includes a 4×4 transpose macro for the column→row transition.

use std::arch::aarch64::*;

/// Fixed-point precision (must match scalar).
const PASS1_BITS: i32 = 2;
const CONST_BITS: i32 = 13;

/// Loeffler constants (must match scalar).
const FIX_0_298: i32 = 2446;
const FIX_0_390: i32 = 3196;
const FIX_0_541: i32 = 4433;
const FIX_0_765: i32 = 6270;
const FIX_0_899: i32 = 7373;
const FIX_1_175: i32 = 9633;
const FIX_1_501: i32 = 12299;
const FIX_1_847: i32 = 15137;
const FIX_1_961: i32 = 16069;
const FIX_2_053: i32 = 16819;
const FIX_2_562: i32 = 20995;
const FIX_3_072: i32 = 25172;

/// Transpose a 4×4 matrix of int32x4_t vectors.
macro_rules! transpose4x4 {
    ($r0:expr, $r1:expr, $r2:expr, $r3:expr) => {{
        // Step 1: interleave pairs
        let a0 = vzip1q_s32($r0, $r2);
        let a1 = vzip2q_s32($r0, $r2);
        let a2 = vzip1q_s32($r1, $r3);
        let a3 = vzip2q_s32($r1, $r3);
        // Step 2: interleave quads
        $r0 = vzip1q_s32(a0, a2);
        $r1 = vzip2q_s32(a0, a2);
        $r2 = vzip1q_s32(a1, a3);
        $r3 = vzip2q_s32(a1, a3);
    }};
}

/// Loeffler butterfly on 4 lanes (even part).
/// Returns (tmp0, tmp1, tmp2, tmp3) with half-round bias added.
#[inline(always)]
unsafe fn even_part_neon(
    s0: int32x4_t,
    s2: int32x4_t,
    s4: int32x4_t,
    s6: int32x4_t,
    bias: int32x4_t,
) -> (int32x4_t, int32x4_t, int32x4_t, int32x4_t) {
    let tmp0 = vshlq_n_s32::<CONST_BITS>(s0);
    let tmp2 = vshlq_n_s32::<CONST_BITS>(s4);

    let tmp10 = vaddq_s32(tmp0, tmp2);
    let tmp11 = vsubq_s32(tmp0, tmp2);

    let z1 = vaddq_s32(s2, s6);
    let fix_0541 = vdupq_n_s32(FIX_0_541);
    let fix_0765 = vdupq_n_s32(FIX_0_765);
    let fix_1847 = vdupq_n_s32(FIX_1_847);

    let tmp13 = vaddq_s32(vmulq_s32(z1, fix_0541), vmulq_s32(s2, fix_0765));
    let tmp12 = vsubq_s32(vmulq_s32(z1, fix_0541), vmulq_s32(s6, fix_1847));

    let t0 = vaddq_s32(vaddq_s32(tmp10, tmp13), bias);
    let t3 = vaddq_s32(vsubq_s32(tmp10, tmp13), bias);
    let t1 = vaddq_s32(vaddq_s32(tmp11, tmp12), bias);
    let t2 = vaddq_s32(vsubq_s32(tmp11, tmp12), bias);

    (t0, t1, t2, t3)
}

/// Loeffler butterfly on 4 lanes (odd part).
/// Returns (tmp0_odd, tmp1_odd, tmp2_odd, tmp3_odd).
#[inline(always)]
unsafe fn odd_part_neon(
    s1: int32x4_t,
    s3: int32x4_t,
    s5: int32x4_t,
    s7: int32x4_t,
) -> (int32x4_t, int32x4_t, int32x4_t, int32x4_t) {
    let z1 = vaddq_s32(s7, s1);
    let z2 = vaddq_s32(s5, s3);
    let z3 = vaddq_s32(s7, s3);
    let z4 = vaddq_s32(s5, s1);
    let z5 = vmulq_s32(vaddq_s32(z3, z4), vdupq_n_s32(FIX_1_175));

    let p7 = vmulq_s32(s7, vdupq_n_s32(FIX_0_298));
    let p5 = vmulq_s32(s5, vdupq_n_s32(FIX_2_053));
    let p3 = vmulq_s32(s3, vdupq_n_s32(FIX_3_072));
    let p1 = vmulq_s32(s1, vdupq_n_s32(FIX_1_501));

    let z1 = vmulq_s32(z1, vdupq_n_s32(-FIX_0_899));
    let z2 = vmulq_s32(z2, vdupq_n_s32(-FIX_2_562));
    let z3 = vaddq_s32(vmulq_s32(z3, vdupq_n_s32(-FIX_1_961)), z5);
    let z4 = vaddq_s32(vmulq_s32(z4, vdupq_n_s32(-FIX_0_390)), z5);

    let tmp0_odd = vaddq_s32(vaddq_s32(p7, z1), z3);
    let tmp1_odd = vaddq_s32(vaddq_s32(p5, z2), z4);
    let tmp2_odd = vaddq_s32(vaddq_s32(p3, z2), z3);
    let tmp3_odd = vaddq_s32(vaddq_s32(p1, z1), z4);

    (tmp0_odd, tmp1_odd, tmp2_odd, tmp3_odd)
}

/// NEON 8×8 IDCT: processes 4 columns at a time, transposes, then 4 rows.
///
/// Two passes of the Loeffler butterfly, each processing 4 elements in
/// parallel. Between passes, a 4×4 transpose reorders the data.
pub fn idct_8x8_neon(coeffs: &[i32; 64], output: &mut [u8], stride: usize) {
    unsafe { idct_8x8_neon_inner(coeffs, output, stride) }
}

#[target_feature(enable = "neon")]
unsafe fn idct_8x8_neon_inner(coeffs: &[i32; 64], output: &mut [u8], stride: usize) {
    let mut workspace = [0i32; 64];

    // Pass 1: columns. Process columns 0-3 then 4-7 (4 at a time).
    let half = vdupq_n_s32(1 << (CONST_BITS - 1));

    for col_group in 0..2 {
        let base = col_group * 4;

        // Load 8 rows of 4 coefficients each
        let s0 = vld1q_s32(coeffs.as_ptr().add(base));
        let s1 = vld1q_s32(coeffs.as_ptr().add(8 + base));
        let s2 = vld1q_s32(coeffs.as_ptr().add(16 + base));
        let s3 = vld1q_s32(coeffs.as_ptr().add(24 + base));
        let s4 = vld1q_s32(coeffs.as_ptr().add(32 + base));
        let s5 = vld1q_s32(coeffs.as_ptr().add(40 + base));
        let s6 = vld1q_s32(coeffs.as_ptr().add(48 + base));
        let s7 = vld1q_s32(coeffs.as_ptr().add(56 + base));

        let (t0, t1, t2, t3) = even_part_neon(s0, s2, s4, s6, half);
        let (o0, o1, o2, o3) = odd_part_neon(s1, s3, s5, s7);

        // Combine and descale
        let r0 = vshrq_n_s32::<{ 13 - 2 }>(vaddq_s32(t0, o3));
        let r7 = vshrq_n_s32::<{ 13 - 2 }>(vsubq_s32(t0, o3));
        let r1 = vshrq_n_s32::<{ 13 - 2 }>(vaddq_s32(t1, o2));
        let r6 = vshrq_n_s32::<{ 13 - 2 }>(vsubq_s32(t1, o2));
        let r2 = vshrq_n_s32::<{ 13 - 2 }>(vaddq_s32(t2, o1));
        let r5 = vshrq_n_s32::<{ 13 - 2 }>(vsubq_s32(t2, o1));
        let r3 = vshrq_n_s32::<{ 13 - 2 }>(vaddq_s32(t3, o0));
        let r4 = vshrq_n_s32::<{ 13 - 2 }>(vsubq_s32(t3, o0));

        // Store to workspace
        vst1q_s32(workspace.as_mut_ptr().add(base), r0);
        vst1q_s32(workspace.as_mut_ptr().add(8 + base), r1);
        vst1q_s32(workspace.as_mut_ptr().add(16 + base), r2);
        vst1q_s32(workspace.as_mut_ptr().add(24 + base), r3);
        vst1q_s32(workspace.as_mut_ptr().add(32 + base), r4);
        vst1q_s32(workspace.as_mut_ptr().add(40 + base), r5);
        vst1q_s32(workspace.as_mut_ptr().add(48 + base), r6);
        vst1q_s32(workspace.as_mut_ptr().add(56 + base), r7);
    }

    // Pass 2: rows. Process rows 0-3 then 4-7 (4 at a time).
    // We need to transpose 4×8 blocks, process them, then write output.
    let range_shift = CONST_BITS + PASS1_BITS + 3;
    let bias_val = (1 << (range_shift - 1)) + (128 << range_shift);
    let bias = vdupq_n_s32(bias_val);
    let zero = vdupq_n_s32(0);
    let max255 = vdupq_n_s32(255);

    for row_group in 0..2 {
        let row_base = row_group * 4 * 8; // rows 0-3 or 4-7

        // Load 4 rows of 8 values each, but we need to transpose to get
        // columns. Load as 4×4 blocks and transpose.
        // Left half (columns 0-3)
        let mut r0l = vld1q_s32(workspace.as_ptr().add(row_base));
        let mut r1l = vld1q_s32(workspace.as_ptr().add(row_base + 8));
        let mut r2l = vld1q_s32(workspace.as_ptr().add(row_base + 16));
        let mut r3l = vld1q_s32(workspace.as_ptr().add(row_base + 24));
        transpose4x4!(r0l, r1l, r2l, r3l);

        // Right half (columns 4-7)
        let mut r0r = vld1q_s32(workspace.as_ptr().add(row_base + 4));
        let mut r1r = vld1q_s32(workspace.as_ptr().add(row_base + 8 + 4));
        let mut r2r = vld1q_s32(workspace.as_ptr().add(row_base + 16 + 4));
        let mut r3r = vld1q_s32(workspace.as_ptr().add(row_base + 24 + 4));
        transpose4x4!(r0r, r1r, r2r, r3r);

        // Now: r0l=col0(4rows), r1l=col1, r2l=col2, r3l=col3
        //      r0r=col4, r1r=col5, r2r=col6, r3r=col7
        // These are the 8 "columns" of the 4 rows we're processing
        let s0 = r0l;
        let s1 = r1l;
        let s2 = r2l;
        let s3 = r3l;
        let s4 = r0r;
        let s5 = r1r;
        let s6 = r2r;
        let s7 = r3r;

        let (t0, t1, t2, t3) = even_part_neon(s0, s2, s4, s6, bias);
        let (o0, o1, o2, o3) = odd_part_neon(s1, s3, s5, s7);

        // Combine, descale, and clamp to [0, 255]
        let clamp = |v: int32x4_t| -> int32x4_t {
            let shifted = vshrq_n_s32::<{ 13 + 2 + 3 }>(v);
            vminq_s32(vmaxq_s32(shifted, zero), max255)
        };

        let out0 = clamp(vaddq_s32(t0, o3));
        let out7 = clamp(vsubq_s32(t0, o3));
        let out1 = clamp(vaddq_s32(t1, o2));
        let out6 = clamp(vsubq_s32(t1, o2));
        let out2 = clamp(vaddq_s32(t2, o1));
        let out5 = clamp(vsubq_s32(t2, o1));
        let out3 = clamp(vaddq_s32(t3, o0));
        let out4 = clamp(vsubq_s32(t3, o0));

        // Write output: extract 4 rows, 8 pixels each.
        // Each outN vector holds column N's values for the 4 rows in this group.
        // We need to gather column 0..7 for each row.
        // Store to a temporary, then extract rows.
        let mut tmp = [0i32; 32]; // 4 rows × 8 columns
                                  // Store columns into row-major temp via scatter
        vst1q_s32(tmp.as_mut_ptr(), out0); // col 0, rows 0-3
        vst1q_s32(tmp.as_mut_ptr().add(4), out1); // col 1, rows 0-3
        vst1q_s32(tmp.as_mut_ptr().add(8), out2);
        vst1q_s32(tmp.as_mut_ptr().add(12), out3);
        vst1q_s32(tmp.as_mut_ptr().add(16), out4);
        vst1q_s32(tmp.as_mut_ptr().add(20), out5);
        vst1q_s32(tmp.as_mut_ptr().add(24), out6);
        vst1q_s32(tmp.as_mut_ptr().add(28), out7);

        // Extract each row (gather from column-major tmp)
        for local_row in 0..4 {
            let row_idx = row_group * 4 + local_row;
            let out_ptr = output.as_mut_ptr().add(row_idx * stride);
            for col in 0..8 {
                *out_ptr.add(col) = tmp[col * 4 + local_row] as u8;
            }
        }
    }
}

/// NEON DC-only IDCT: fill 8×8 block with a single value.
pub fn idct_dc_only_neon(dc_value: i32, output: &mut [u8], stride: usize) {
    let range_shift = CONST_BITS + PASS1_BITS + 3;
    let round = 1 << (range_shift - 1);
    let bias = round + (128 << range_shift);
    let scaled = dc_value << (CONST_BITS + PASS1_BITS);
    let val = ((scaled + bias) >> range_shift).clamp(0, 255) as u8;

    // Use NEON to fill 8 bytes at once
    unsafe {
        let fill = vdup_n_u8(val);
        for row in 0..8 {
            vst1_u8(output.as_mut_ptr().add(row * stride), fill);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::scalar::{idct_8x8_scalar, idct_dc_only_scalar};
    use super::{idct_8x8_neon, idct_dc_only_neon};

    /// Deterministic synthetic IDCT coefficient block: DC + a few AC terms.
    /// Values chosen so that no output pixel saturates (all outputs stay in ~30..220).
    fn make_test_coeffs() -> [i32; 64] {
        let mut c = [0i32; 64];
        // DC coefficient
        c[0] = 256;
        // A handful of AC terms at positions that give non-trivial output
        c[1] = 32;
        c[8] = 16;
        c[2] = -24;
        c[16] = 20;
        c[9] = -12;
        c[3] = 8;
        c[24] = -8;
        c
    }

    #[test]
    fn idct_8x8_parity() {
        if !std::arch::is_aarch64_feature_detected!("neon") {
            eprintln!("SIMD feature not available, skipping");
            return;
        }

        let coeffs = make_test_coeffs();
        let mut scalar_out = [0u8; 64];
        let mut simd_out = [0u8; 64];

        idct_8x8_scalar(&coeffs, &mut scalar_out, 8);
        idct_8x8_neon(&coeffs, &mut simd_out, 8);

        for i in 0..64 {
            let diff = (scalar_out[i] as i32 - simd_out[i] as i32).abs();
            assert!(
                diff <= 1,
                "parity mismatch at index {i}: scalar={}, simd={}, diff={}",
                scalar_out[i],
                simd_out[i],
                diff
            );
        }
    }

    #[test]
    fn idct_8x8_parity_zero_block() {
        if !std::arch::is_aarch64_feature_detected!("neon") {
            eprintln!("SIMD feature not available, skipping");
            return;
        }

        let coeffs = [0i32; 64];
        let mut scalar_out = [0u8; 64];
        let mut simd_out = [0u8; 64];

        idct_8x8_scalar(&coeffs, &mut scalar_out, 8);
        idct_8x8_neon(&coeffs, &mut simd_out, 8);

        for i in 0..64 {
            let diff = (scalar_out[i] as i32 - simd_out[i] as i32).abs();
            assert!(
                diff <= 1,
                "zero-block parity mismatch at index {i}: scalar={}, simd={}, diff={}",
                scalar_out[i],
                simd_out[i],
                diff
            );
        }
    }

    #[test]
    fn idct_8x8_parity_strided() {
        if !std::arch::is_aarch64_feature_detected!("neon") {
            eprintln!("SIMD feature not available, skipping");
            return;
        }

        let coeffs = make_test_coeffs();
        let stride = 16usize;
        let mut scalar_out = vec![0u8; stride * 8];
        let mut simd_out = vec![0u8; stride * 8];

        idct_8x8_scalar(&coeffs, &mut scalar_out, stride);
        idct_8x8_neon(&coeffs, &mut simd_out, stride);

        for row in 0..8 {
            for col in 0..8 {
                let i = row * stride + col;
                let diff = (scalar_out[i] as i32 - simd_out[i] as i32).abs();
                assert!(
                    diff <= 1,
                    "strided parity mismatch at row={row} col={col}: scalar={}, simd={}, diff={}",
                    scalar_out[i],
                    simd_out[i],
                    diff
                );
            }
        }
    }

    #[test]
    fn idct_dc_only_parity() {
        if !std::arch::is_aarch64_feature_detected!("neon") {
            eprintln!("SIMD feature not available, skipping");
            return;
        }

        for dc in [0i32, 8, 64, 128, -64, 255, -255] {
            let mut scalar_out = [0u8; 64];
            let mut simd_out = [0u8; 64];

            idct_dc_only_scalar(dc, &mut scalar_out, 8);
            idct_dc_only_neon(dc, &mut simd_out, 8);

            for i in 0..64 {
                let diff = (scalar_out[i] as i32 - simd_out[i] as i32).abs();
                assert!(
                    diff <= 1,
                    "dc_only parity mismatch dc={dc} at index {i}: scalar={}, simd={}, diff={}",
                    scalar_out[i],
                    simd_out[i],
                    diff
                );
            }
        }
    }
}
