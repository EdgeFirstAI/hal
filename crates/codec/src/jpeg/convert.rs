// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! Vectorised u8 → T pixel conversion for JPEG decode output.
//!
//! Provides SIMD-accelerated conversions for f32, u16, and i16 target types.
//! The u8 and i8 paths are handled separately by the caller (direct decode
//! and XOR-copy respectively).

/// Convert a row of u8 pixel values to f32 (normalised to `[0.0, 1.0]`).
///
/// Uses NEON or SSE2 for the bulk of the row, with a scalar tail.
#[inline]
pub fn convert_u8_to_f32(src: &[u8], dst: &mut [f32]) {
    debug_assert!(dst.len() >= src.len());

    #[cfg(target_arch = "aarch64")]
    {
        convert_u8_to_f32_neon(src, dst);
        return;
    }
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("sse2") {
            // SAFETY: SSE2 detected
            unsafe { convert_u8_to_f32_sse2(src, dst) };
            return;
        }
    }
    #[allow(unreachable_code)]
    convert_u8_to_f32_scalar(src, dst);
}

/// Convert a row of u8 pixel values to u16 (scaled 0..255 → 0..65535).
#[inline]
pub fn convert_u8_to_u16(src: &[u8], dst: &mut [u16]) {
    debug_assert!(dst.len() >= src.len());

    #[cfg(target_arch = "aarch64")]
    {
        convert_u8_to_u16_neon(src, dst);
        return;
    }
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("sse2") {
            // SAFETY: SSE2 detected
            unsafe { convert_u8_to_u16_sse2(src, dst) };
            return;
        }
    }
    #[allow(unreachable_code)]
    convert_u8_to_u16_scalar(src, dst);
}

/// Convert a row of u8 pixel values to i16 (scaled + XOR 0x8000).
#[inline]
pub fn convert_u8_to_i16(src: &[u8], dst: &mut [i16]) {
    debug_assert!(dst.len() >= src.len());

    #[cfg(target_arch = "aarch64")]
    {
        convert_u8_to_i16_neon(src, dst);
        return;
    }
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("sse2") {
            // SAFETY: SSE2 detected
            unsafe { convert_u8_to_i16_sse2(src, dst) };
            return;
        }
    }
    #[allow(unreachable_code)]
    convert_u8_to_i16_scalar(src, dst);
}

// ── Scalar fallbacks ─────────────────────────────────────────────────

fn convert_u8_to_f32_scalar(src: &[u8], dst: &mut [f32]) {
    let recip = 1.0_f32 / 255.0;
    for (s, d) in src.iter().zip(dst.iter_mut()) {
        *d = *s as f32 * recip;
    }
}

fn convert_u8_to_u16_scalar(src: &[u8], dst: &mut [u16]) {
    for (s, d) in src.iter().zip(dst.iter_mut()) {
        *d = *s as u16 * 257;
    }
}

fn convert_u8_to_i16_scalar(src: &[u8], dst: &mut [i16]) {
    for (s, d) in src.iter().zip(dst.iter_mut()) {
        *d = ((*s as u16 * 257) ^ 0x8000) as i16;
    }
}

// ── NEON implementations ─────────────────────────────────────────────

#[cfg(target_arch = "aarch64")]
fn convert_u8_to_f32_neon(src: &[u8], dst: &mut [f32]) {
    use std::arch::aarch64::*;
    let len = src.len();

    // Process 16 bytes at a time → 16 f32s
    let chunks = len / 16;
    let recip = 1.0_f32 / 255.0;

    unsafe {
        let v_recip = vdupq_n_f32(recip);
        let mut i = 0;
        for _ in 0..chunks {
            let u8x16 = vld1q_u8(src.as_ptr().add(i));

            // Widen u8 → u16 → u32 → f32
            let lo8 = vget_low_u8(u8x16);
            let hi8 = vget_high_u8(u8x16);
            let lo16 = vmovl_u8(lo8);
            let hi16 = vmovl_u8(hi8);

            let u32_0 = vmovl_u16(vget_low_u16(lo16));
            let u32_1 = vmovl_high_u16(lo16);
            let u32_2 = vmovl_u16(vget_low_u16(hi16));
            let u32_3 = vmovl_high_u16(hi16);

            let f0 = vmulq_f32(vcvtq_f32_u32(u32_0), v_recip);
            let f1 = vmulq_f32(vcvtq_f32_u32(u32_1), v_recip);
            let f2 = vmulq_f32(vcvtq_f32_u32(u32_2), v_recip);
            let f3 = vmulq_f32(vcvtq_f32_u32(u32_3), v_recip);

            vst1q_f32(dst.as_mut_ptr().add(i), f0);
            vst1q_f32(dst.as_mut_ptr().add(i + 4), f1);
            vst1q_f32(dst.as_mut_ptr().add(i + 8), f2);
            vst1q_f32(dst.as_mut_ptr().add(i + 12), f3);

            i += 16;
        }

        // Scalar tail
        for j in i..len {
            *dst.get_unchecked_mut(j) = *src.get_unchecked(j) as f32 * recip;
        }
    }
}

#[cfg(target_arch = "aarch64")]
fn convert_u8_to_u16_neon(src: &[u8], dst: &mut [u16]) {
    use std::arch::aarch64::*;
    let len = src.len();
    let chunks = len / 16;

    unsafe {
        let v257 = vdupq_n_u16(257);
        let mut i = 0;
        for _ in 0..chunks {
            let u8x16 = vld1q_u8(src.as_ptr().add(i));
            let lo = vmovl_u8(vget_low_u8(u8x16));
            let hi = vmovl_u8(vget_high_u8(u8x16));
            vst1q_u16(dst.as_mut_ptr().add(i), vmulq_u16(lo, v257));
            vst1q_u16(dst.as_mut_ptr().add(i + 8), vmulq_u16(hi, v257));
            i += 16;
        }
        for j in i..len {
            *dst.get_unchecked_mut(j) = *src.get_unchecked(j) as u16 * 257;
        }
    }
}

#[cfg(target_arch = "aarch64")]
fn convert_u8_to_i16_neon(src: &[u8], dst: &mut [i16]) {
    use std::arch::aarch64::*;
    let len = src.len();
    let chunks = len / 16;

    unsafe {
        let v257 = vdupq_n_u16(257);
        let v_sign = vdupq_n_u16(0x8000);
        let mut i = 0;
        for _ in 0..chunks {
            let u8x16 = vld1q_u8(src.as_ptr().add(i));
            let lo = vmovl_u8(vget_low_u8(u8x16));
            let hi = vmovl_u8(vget_high_u8(u8x16));
            let lo_scaled = veorq_u16(vmulq_u16(lo, v257), v_sign);
            let hi_scaled = veorq_u16(vmulq_u16(hi, v257), v_sign);
            vst1q_u16(dst.as_mut_ptr().add(i) as *mut u16, lo_scaled);
            vst1q_u16(dst.as_mut_ptr().add(i + 8) as *mut u16, hi_scaled);
            i += 16;
        }
        for j in i..len {
            *dst.get_unchecked_mut(j) = ((*src.get_unchecked(j) as u16 * 257) ^ 0x8000) as i16;
        }
    }
}

// ── SSE2 implementations ─────────────────────────────────────────────

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse2")]
unsafe fn convert_u8_to_f32_sse2(src: &[u8], dst: &mut [f32]) {
    use std::arch::x86_64::*;
    let len = src.len();
    let chunks = len / 16;
    let recip = 1.0_f32 / 255.0;
    let v_recip = _mm_set1_ps(recip);
    let zero = _mm_setzero_si128();

    let mut i = 0;
    for _ in 0..chunks {
        let u8x16 = _mm_loadu_si128(src.as_ptr().add(i) as *const __m128i);

        // Unpack u8 → u16 → u32 → f32 (4 groups of 4)
        let lo16 = _mm_unpacklo_epi8(u8x16, zero);
        let hi16 = _mm_unpackhi_epi8(u8x16, zero);

        let u32_0 = _mm_unpacklo_epi16(lo16, zero);
        let u32_1 = _mm_unpackhi_epi16(lo16, zero);
        let u32_2 = _mm_unpacklo_epi16(hi16, zero);
        let u32_3 = _mm_unpackhi_epi16(hi16, zero);

        let f0 = _mm_mul_ps(_mm_cvtepi32_ps(u32_0), v_recip);
        let f1 = _mm_mul_ps(_mm_cvtepi32_ps(u32_1), v_recip);
        let f2 = _mm_mul_ps(_mm_cvtepi32_ps(u32_2), v_recip);
        let f3 = _mm_mul_ps(_mm_cvtepi32_ps(u32_3), v_recip);

        _mm_storeu_ps(dst.as_mut_ptr().add(i), f0);
        _mm_storeu_ps(dst.as_mut_ptr().add(i + 4), f1);
        _mm_storeu_ps(dst.as_mut_ptr().add(i + 8), f2);
        _mm_storeu_ps(dst.as_mut_ptr().add(i + 12), f3);

        i += 16;
    }

    for j in i..len {
        *dst.get_unchecked_mut(j) = *src.get_unchecked(j) as f32 * recip;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse2")]
unsafe fn convert_u8_to_u16_sse2(src: &[u8], dst: &mut [u16]) {
    use std::arch::x86_64::*;
    let len = src.len();
    let chunks = len / 16;
    let zero = _mm_setzero_si128();

    // 257 = 0x0101. Use mullo_epi16 which is available in SSE2.
    let v257 = _mm_set1_epi16(257_i16);

    let mut i = 0;
    for _ in 0..chunks {
        let u8x16 = _mm_loadu_si128(src.as_ptr().add(i) as *const __m128i);
        let lo16 = _mm_unpacklo_epi8(u8x16, zero);
        let hi16 = _mm_unpackhi_epi8(u8x16, zero);
        let lo_scaled = _mm_mullo_epi16(lo16, v257);
        let hi_scaled = _mm_mullo_epi16(hi16, v257);
        _mm_storeu_si128(dst.as_mut_ptr().add(i) as *mut __m128i, lo_scaled);
        _mm_storeu_si128(dst.as_mut_ptr().add(i + 8) as *mut __m128i, hi_scaled);
        i += 16;
    }

    for j in i..len {
        *dst.get_unchecked_mut(j) = *src.get_unchecked(j) as u16 * 257;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse2")]
unsafe fn convert_u8_to_i16_sse2(src: &[u8], dst: &mut [i16]) {
    use std::arch::x86_64::*;
    let len = src.len();
    let chunks = len / 16;
    let zero = _mm_setzero_si128();
    let v257 = _mm_set1_epi16(257_i16);
    let v_sign = _mm_set1_epi16(i16::MIN); // 0x8000

    let mut i = 0;
    for _ in 0..chunks {
        let u8x16 = _mm_loadu_si128(src.as_ptr().add(i) as *const __m128i);
        let lo16 = _mm_unpacklo_epi8(u8x16, zero);
        let hi16 = _mm_unpackhi_epi8(u8x16, zero);
        let lo_scaled = _mm_xor_si128(_mm_mullo_epi16(lo16, v257), v_sign);
        let hi_scaled = _mm_xor_si128(_mm_mullo_epi16(hi16, v257), v_sign);
        _mm_storeu_si128(dst.as_mut_ptr().add(i) as *mut __m128i, lo_scaled);
        _mm_storeu_si128(dst.as_mut_ptr().add(i + 8) as *mut __m128i, hi_scaled);
        i += 16;
    }

    for j in i..len {
        *dst.get_unchecked_mut(j) = ((*src.get_unchecked(j) as u16 * 257) ^ 0x8000) as i16;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn f32_conversion_matches_scalar() {
        let src: Vec<u8> = (0..=255).collect();
        let mut dst_vec = vec![0.0_f32; 256];
        let mut dst_scalar = vec![0.0_f32; 256];

        convert_u8_to_f32(&src, &mut dst_vec);
        convert_u8_to_f32_scalar(&src, &mut dst_scalar);

        for i in 0..256 {
            assert!(
                (dst_vec[i] - dst_scalar[i]).abs() < f32::EPSILON,
                "mismatch at {i}: vec={} scalar={}",
                dst_vec[i],
                dst_scalar[i]
            );
        }
    }

    #[test]
    fn u16_conversion_matches_scalar() {
        let src: Vec<u8> = (0..=255).collect();
        let mut dst_vec = vec![0u16; 256];
        let mut dst_scalar = vec![0u16; 256];

        convert_u8_to_u16(&src, &mut dst_vec);
        convert_u8_to_u16_scalar(&src, &mut dst_scalar);

        assert_eq!(dst_vec, dst_scalar);
    }

    #[test]
    fn i16_conversion_matches_scalar() {
        let src: Vec<u8> = (0..=255).collect();
        let mut dst_vec = vec![0i16; 256];
        let mut dst_scalar = vec![0i16; 256];

        convert_u8_to_i16(&src, &mut dst_vec);
        convert_u8_to_i16_scalar(&src, &mut dst_scalar);

        assert_eq!(dst_vec, dst_scalar);
    }

    #[test]
    fn f32_odd_lengths() {
        for len in [1, 7, 15, 17, 33] {
            let src: Vec<u8> = (0..len as u8).collect();
            let mut dst = vec![0.0_f32; len];
            convert_u8_to_f32(&src, &mut dst);
            for (i, d) in dst.iter().enumerate() {
                let expected = i as f32 / 255.0;
                assert!(
                    (d - expected).abs() < f32::EPSILON,
                    "len={len} i={i}: got={d} expected={expected}"
                );
            }
        }
    }
}
