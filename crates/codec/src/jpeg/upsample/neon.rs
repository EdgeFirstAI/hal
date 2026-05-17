// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! NEON-optimised chroma upsampling.
//!
//! Horizontal 2× bilinear upsample using widening multiply-accumulate.
//! Processes 8 input samples → 16 output samples per iteration.

use std::arch::aarch64::*;

/// NEON horizontal 2× upsample with bilinear 3:1 blend.
///
/// For each pair of adjacent input samples (a, b), produces:
///   output[2i+1] = (3a + b + 2) >> 2
///   output[2i+2] = (a + 3b + 2) >> 2
pub fn upsample_h2_neon(input: &[u8], output: &mut [u8], width: usize) {
    if width < 2 {
        super::scalar::upsample_h2(input, output, width);
        return;
    }

    unsafe { upsample_h2_neon_inner(input, output, width) }
}

#[target_feature(enable = "neon")]
unsafe fn upsample_h2_neon_inner(input: &[u8], output: &mut [u8], width: usize) {
    // First pixel: duplicate
    output[0] = input[0];

    // Process interior pixels in chunks of 8
    let chunks = (width - 1) / 8;

    for chunk in 0..chunks {
        let base = chunk * 8;

        // Load 8 current samples and 8 next samples
        let a = vld1_u8(input.as_ptr().add(base));
        let b = vld1_u8(input.as_ptr().add(base + 1));

        // Widening operations for precise arithmetic
        let a16 = vmovl_u8(a);
        let b16 = vmovl_u8(b);

        // left = (3a + b + 2) >> 2
        let left16 = vshrq_n_u16::<2>(vaddq_u16(
            vaddq_u16(vmulq_n_u16(a16, 3), b16),
            vdupq_n_u16(2),
        ));
        // right = (a + 3b + 2) >> 2
        let right16 = vshrq_n_u16::<2>(vaddq_u16(
            vaddq_u16(a16, vmulq_n_u16(b16, 3)),
            vdupq_n_u16(2),
        ));

        // Narrow back to u8
        let left8 = vmovn_u16(left16);
        let right8 = vmovn_u16(right16);

        // Interleave left/right for output
        let interleaved = uint8x8x2_t(left8, right8);
        vst2_u8(output.as_mut_ptr().add(base * 2 + 1), interleaved);
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
