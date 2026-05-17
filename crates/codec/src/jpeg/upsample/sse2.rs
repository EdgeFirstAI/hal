// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! SSE2-optimised chroma upsampling.
//!
//! Horizontal 2× bilinear upsample using widening multiply.
//! Processes 8 input samples → 16 output samples per iteration.

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

/// SSE2 horizontal 2× upsample with bilinear 3:1 blend.
///
/// For each pair of adjacent input samples (a, b), produces:
///   output[2i+1] = (3a + b + 2) >> 2
///   output[2i+2] = (a + 3b + 2) >> 2
pub fn upsample_h2_sse2(input: &[u8], output: &mut [u8], width: usize) {
    if width < 2 {
        super::scalar::upsample_h2(input, output, width);
        return;
    }

    unsafe { upsample_h2_sse2_inner(input, output, width) }
}

#[target_feature(enable = "sse2")]
unsafe fn upsample_h2_sse2_inner(input: &[u8], output: &mut [u8], width: usize) {
    // First pixel: duplicate
    output[0] = input[0];

    let chunks = (width - 1) / 8;
    let zero = _mm_setzero_si128();
    let two = _mm_set1_epi16(2);
    let three = _mm_set1_epi16(3);

    for chunk in 0..chunks {
        let base = chunk * 8;

        // Load 8 current samples and 8 next samples
        let a_u8 = _mm_loadl_epi64(input.as_ptr().add(base) as *const __m128i);
        let b_u8 = _mm_loadl_epi64(input.as_ptr().add(base + 1) as *const __m128i);

        // Zero-extend to u16
        let a16 = _mm_unpacklo_epi8(a_u8, zero);
        let b16 = _mm_unpacklo_epi8(b_u8, zero);

        // left = (3a + b + 2) >> 2
        let left16 = _mm_srli_epi16::<2>(_mm_add_epi16(
            _mm_add_epi16(_mm_mullo_epi16(a16, three), b16),
            two,
        ));
        // right = (a + 3b + 2) >> 2
        let right16 = _mm_srli_epi16::<2>(_mm_add_epi16(
            _mm_add_epi16(a16, _mm_mullo_epi16(b16, three)),
            two,
        ));

        // Narrow back to u8
        let left8 = _mm_packus_epi16(left16, zero);
        let right8 = _mm_packus_epi16(right16, zero);

        // Interleave left/right for output: [l0,r0, l1,r1, l2,r2, ...]
        let interleaved = _mm_unpacklo_epi8(left8, right8);

        // Store 16 bytes of interleaved output
        _mm_storeu_si128(
            output.as_mut_ptr().add(base * 2 + 1) as *mut __m128i,
            interleaved,
        );
    }

    // Scalar tail for remaining interior pixels
    let start = chunks * 8;
    for i in start..width - 1 {
        let a = input[i] as u16;
        let b = input[i + 1] as u16;
        output[2 * i + 1] = ((a * 3 + b + 2) >> 2) as u8;
        output[2 * i + 2] = ((a + b * 3 + 2) >> 2) as u8;
    }

    // Last pixel: duplicate
    output[2 * width - 1] = input[width - 1];
}

#[cfg(test)]
#[cfg(target_arch = "x86_64")]
mod tests {
    use super::super::scalar::upsample_h2;
    use super::upsample_h2_sse2;

    fn make_chroma_row() -> [u8; 16] {
        [
            0, 16, 32, 64, 80, 100, 128, 150, 170, 200, 220, 235, 255, 128, 64, 0,
        ]
    }

    #[test]
    fn upsample_h2_parity_full() {
        if !is_x86_feature_detected!("sse2") {
            eprintln!("SIMD feature not available, skipping");
            return;
        }

        let input = make_chroma_row();
        let width = 16usize;
        let mut scalar_out = vec![0u8; width * 2];
        let mut simd_out = vec![0u8; width * 2];

        upsample_h2(&input, &mut scalar_out, width);
        upsample_h2_sse2(&input, &mut simd_out, width);

        for i in 0..(width * 2) {
            let diff = (scalar_out[i] as i32 - simd_out[i] as i32).abs();
            assert!(
                diff <= 1,
                "parity mismatch at output index {i}: scalar={}, simd={}, diff={}",
                scalar_out[i],
                simd_out[i],
                diff
            );
        }
    }

    #[test]
    fn upsample_h2_parity_partial_chunk() {
        if !is_x86_feature_detected!("sse2") {
            eprintln!("SIMD feature not available, skipping");
            return;
        }

        // Width 11: one full chunk of 8 interior pairs + 2 scalar tail pairs + boundary
        let input = make_chroma_row();
        let width = 11usize;
        let mut scalar_out = vec![0u8; width * 2];
        let mut simd_out = vec![0u8; width * 2];

        upsample_h2(&input[..width], &mut scalar_out, width);
        upsample_h2_sse2(&input[..width], &mut simd_out, width);

        for i in 0..(width * 2) {
            let diff = (scalar_out[i] as i32 - simd_out[i] as i32).abs();
            assert!(
                diff <= 1,
                "partial parity mismatch at output index {i}: scalar={}, simd={}, diff={}",
                scalar_out[i],
                simd_out[i],
                diff
            );
        }
    }

    #[test]
    fn upsample_h2_parity_uniform() {
        if !is_x86_feature_detected!("sse2") {
            eprintln!("SIMD feature not available, skipping");
            return;
        }

        let input = [128u8; 16];
        let width = 16usize;
        let mut scalar_out = vec![0u8; width * 2];
        let mut simd_out = vec![0u8; width * 2];

        upsample_h2(&input, &mut scalar_out, width);
        upsample_h2_sse2(&input, &mut simd_out, width);

        for i in 0..(width * 2) {
            let diff = (scalar_out[i] as i32 - simd_out[i] as i32).abs();
            assert!(
                diff <= 1,
                "uniform parity mismatch at output index {i}: scalar={}, simd={}, diff={}",
                scalar_out[i],
                simd_out[i],
                diff
            );
        }
    }
}
