// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! Fused YCbCr → interleaved RGB row conversion for the decode write stage.
//!
//! Full-range JFIF/BT.601 (the JPEG convention):
//!
//! ```text
//! R = Y + 1.40200 · (Cr − 128)
//! G = Y − 0.34414 · (Cb − 128) − 0.71414 · (Cr − 128)
//! B = Y + 1.77200 · (Cb − 128)
//! ```
//!
//! Fixed point matches libjpeg-turbo's `jdcolext-neon` shape: coefficients in
//! Q14, applied with the rounding-doubling high half multiply (`SQRDMULH`) on
//! a **doubled** chroma term — `sqrdmulh(2·c, round(k·2¹⁴)) = c·k` with one
//! rounding step. `SQRDMULH` is baseline ASIMD, so the same kernel runs on
//! Cortex-A53 through Apple Silicon. The scalar path reproduces the identical
//! arithmetic bit-for-bit, so NEON/scalar parity is exact (max error vs. the
//! real-valued formula is ±1).

/// Q14 coefficients: `round(k · 2^14)`.
const CR_R: i16 = 22970; // 1.40200
const CB_G: i16 = 5638; //  0.34414
const CR_G: i16 = 11700; // 0.71414
const CB_B: i16 = 29032; // 1.77200

/// Scalar model of `SQRDMULH` for the value ranges used here (no saturation
/// needed: |2·chroma| ≤ 256 and coefficients < 2¹⁵ keep `2ab` far from i32
/// overflow).
#[inline(always)]
fn qrdmulh(a: i32, b: i32) -> i32 {
    (2 * a * b + 0x8000) >> 16
}

#[inline(always)]
fn clamp_u8(v: i32) -> u8 {
    v.clamp(0, 255) as u8
}

/// Convert one row of planar Y/Cb/Cr samples to interleaved RGB bytes.
///
/// Writes `3·n` bytes to `dst`.
#[inline]
pub fn ycbcr_to_rgb_row(y: &[u8], cb: &[u8], cr: &[u8], dst: &mut [u8], n: usize) {
    assert!(y.len() >= n && cb.len() >= n && cr.len() >= n && dst.len() >= n * 3);
    #[cfg(target_arch = "aarch64")]
    {
        use super::cpu::{neon_tier, NeonTier};
        if !matches!(neon_tier(), NeonTier::Scalar)
            && std::arch::is_aarch64_feature_detected!("neon")
        {
            // SAFETY: lengths checked above; the NEON path only touches `n`
            // source samples and `3·n` destination bytes.
            unsafe { ycbcr_to_rgb_row_neon(y, cb, cr, dst, n) };
            return;
        }
    }
    #[cfg(target_arch = "x86_64")]
    {
        use super::cpu::intel_tier;
        let tier = intel_tier();
        if tier.has_avx2() && is_x86_feature_detected!("avx2") {
            unsafe { ycbcr_to_rgb_row_avx2(y, cb, cr, dst, n) };
            return;
        }
        if tier.has_sse41() && is_x86_feature_detected!("sse4.1") {
            unsafe { ycbcr_to_rgb_row_sse41(y, cb, cr, dst, n) };
            return;
        }
    }
    ycbcr_to_rgb_row_scalar(y, cb, cr, dst, 0, n);
}

#[inline]
fn ycbcr_to_rgb_row_scalar(y: &[u8], cb: &[u8], cr: &[u8], dst: &mut [u8], start: usize, n: usize) {
    for i in start..n {
        let yv = y[i] as i32;
        let cb2 = 2 * (cb[i] as i32 - 128);
        let cr2 = 2 * (cr[i] as i32 - 128);
        dst[i * 3] = clamp_u8(yv + qrdmulh(cr2, CR_R as i32));
        dst[i * 3 + 1] = clamp_u8(yv - qrdmulh(cb2, CB_G as i32) - qrdmulh(cr2, CR_G as i32));
        dst[i * 3 + 2] = clamp_u8(yv + qrdmulh(cb2, CB_B as i32));
    }
}

/// Convert one or two adjacent 8×8 4:4:4 MCU blocks to interleaved RGB.
///
/// Pairing two full blocks keeps the 16-pixel `vst3q` kernel and does one ISA
/// dispatch for the whole pair. Calling [`ycbcr_to_rgb_row`] per 8-pixel MCU
/// row was thousands of extra calls (and a feature-detect each time) on
/// in-order A53/A55.
#[allow(clippy::too_many_arguments)]
pub fn ycbcr_to_rgb_blocks(
    y0: &[u8; 64],
    cb0: &[u8; 64],
    cr0: &[u8; 64],
    right: Option<(&[u8; 64], &[u8; 64], &[u8; 64])>,
    dst: &mut [u8],
    stride: usize,
    x: usize,
    y: usize,
    cols0: usize,
    cols1: usize,
    rows: usize,
) {
    #[cfg(target_arch = "aarch64")]
    {
        use super::cpu::{neon_tier, NeonTier};
        if !matches!(neon_tier(), NeonTier::Scalar)
            && std::arch::is_aarch64_feature_detected!("neon")
        {
            if let Some((y1, cb1, cr1)) = right {
                if cols0 == 8 && cols1 == 8 {
                    // SAFETY: 8×`rows` source samples per plane; destination
                    // holds 16 RGB pixels per row at `stride`.
                    unsafe {
                        ycbcr_to_rgb_block16_neon(
                            y0, cb0, cr0, y1, cb1, cr1, dst, stride, x, y, rows,
                        );
                    }
                    return;
                }
            } else if cols0 == 8 {
                // SAFETY: 8×`rows` source samples; 8 RGB pixels per row.
                unsafe { ycbcr_to_rgb_block8_neon(y0, cb0, cr0, dst, stride, x, y, rows) };
                return;
            }
        }
    }
    #[cfg(target_arch = "x86_64")]
    {
        if super::cpu::intel_tier().has_sse41() {
            if let Some((y1, cb1, cr1)) = right {
                if cols0 == 8 && cols1 == 8 {
                    // SAFETY: SSE4.1 probed; same bounds contract as NEON above.
                    unsafe {
                        ycbcr_to_rgb_block16_sse41(
                            y0, cb0, cr0, y1, cb1, cr1, dst, stride, x, y, rows,
                        );
                    }
                    return;
                }
            }
        }
    }
    for r in 0..rows {
        let d = (y + r) * stride + x * 3;
        ycbcr_to_rgb_row(
            &y0[r * 8..],
            &cb0[r * 8..],
            &cr0[r * 8..],
            &mut dst[d..],
            cols0,
        );
        if let Some((y1, cb1, cr1)) = right {
            if cols1 > 0 {
                let d1 = d + cols0 * 3;
                ycbcr_to_rgb_row(
                    &y1[r * 8..],
                    &cb1[r * 8..],
                    &cr1[r * 8..],
                    &mut dst[d1..],
                    cols1,
                );
            }
        }
    }
}

/// Q14 butterfly on 8 i16 lanes → packed u8. Shared by the row and MCU-block
/// NEON writers so 8-wide and 16-wide stores stay bit-identical.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[inline]
unsafe fn ycbcr_half_neon(
    yv: core::arch::aarch64::int16x8_t,
    cbv: core::arch::aarch64::int16x8_t,
    crv: core::arch::aarch64::int16x8_t,
) -> (
    core::arch::aarch64::uint8x8_t,
    core::arch::aarch64::uint8x8_t,
    core::arch::aarch64::uint8x8_t,
) {
    use core::arch::aarch64::*;
    let c128 = vdupq_n_s16(128);
    let cb2 = vshlq_n_s16::<1>(vsubq_s16(cbv, c128));
    let cr2 = vshlq_n_s16::<1>(vsubq_s16(crv, c128));
    let r = vaddq_s16(yv, vqrdmulhq_n_s16(cr2, CR_R));
    let g = vsubq_s16(
        vsubq_s16(yv, vqrdmulhq_n_s16(cb2, CB_G)),
        vqrdmulhq_n_s16(cr2, CR_G),
    );
    let b = vaddq_s16(yv, vqrdmulhq_n_s16(cb2, CB_B));
    (vqmovun_s16(r), vqmovun_s16(g), vqmovun_s16(b))
}

/// 16 pixels per iteration: two 8-lane s16 halves through the Q14 butterfly,
/// then one `vst3q_u8` interleaved store.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn ycbcr_to_rgb_row_neon(y: &[u8], cb: &[u8], cr: &[u8], dst: &mut [u8], n: usize) {
    use core::arch::aarch64::*;

    let mut i = 0usize;
    while i + 16 <= n {
        let yq = vld1q_u8(y.as_ptr().add(i));
        let cbq = vld1q_u8(cb.as_ptr().add(i));
        let crq = vld1q_u8(cr.as_ptr().add(i));

        let (r_lo, g_lo, b_lo) = ycbcr_half_neon(
            vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(yq))),
            vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(cbq))),
            vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(crq))),
        );
        let (r_hi, g_hi, b_hi) = ycbcr_half_neon(
            vreinterpretq_s16_u16(vmovl_u8(vget_high_u8(yq))),
            vreinterpretq_s16_u16(vmovl_u8(vget_high_u8(cbq))),
            vreinterpretq_s16_u16(vmovl_u8(vget_high_u8(crq))),
        );

        let rgb = uint8x16x3_t(
            vcombine_u8(r_lo, r_hi),
            vcombine_u8(g_lo, g_hi),
            vcombine_u8(b_lo, b_hi),
        );
        vst3q_u8(dst.as_mut_ptr().add(i * 3), rgb);
        i += 16;
    }
    while i + 8 <= n {
        let (r, g, b) = ycbcr_half_neon(
            vreinterpretq_s16_u16(vmovl_u8(vld1_u8(y.as_ptr().add(i)))),
            vreinterpretq_s16_u16(vmovl_u8(vld1_u8(cb.as_ptr().add(i)))),
            vreinterpretq_s16_u16(vmovl_u8(vld1_u8(cr.as_ptr().add(i)))),
        );
        vst3_u8(dst.as_mut_ptr().add(i * 3), uint8x8x3_t(r, g, b));
        i += 8;
    }
    ycbcr_to_rgb_row_scalar(y, cb, cr, dst, i, n);
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[allow(clippy::too_many_arguments)]
unsafe fn ycbcr_to_rgb_block8_neon(
    y: &[u8; 64],
    cb: &[u8; 64],
    cr: &[u8; 64],
    dst: &mut [u8],
    stride: usize,
    x: usize,
    y0: usize,
    rows: usize,
) {
    use core::arch::aarch64::*;
    for r in 0..rows {
        let src = r * 8;
        let (rv, gv, bv) = ycbcr_half_neon(
            vreinterpretq_s16_u16(vmovl_u8(vld1_u8(y.as_ptr().add(src)))),
            vreinterpretq_s16_u16(vmovl_u8(vld1_u8(cb.as_ptr().add(src)))),
            vreinterpretq_s16_u16(vmovl_u8(vld1_u8(cr.as_ptr().add(src)))),
        );
        let d = (y0 + r) * stride + x * 3;
        vst3_u8(dst.as_mut_ptr().add(d), uint8x8x3_t(rv, gv, bv));
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[allow(clippy::too_many_arguments)]
unsafe fn ycbcr_to_rgb_block16_neon(
    y0: &[u8; 64],
    cb0: &[u8; 64],
    cr0: &[u8; 64],
    y1: &[u8; 64],
    cb1: &[u8; 64],
    cr1: &[u8; 64],
    dst: &mut [u8],
    stride: usize,
    x: usize,
    y: usize,
    rows: usize,
) {
    use core::arch::aarch64::*;
    for r in 0..rows {
        let src = r * 8;
        let yq = vcombine_u8(vld1_u8(y0.as_ptr().add(src)), vld1_u8(y1.as_ptr().add(src)));
        let cbq = vcombine_u8(
            vld1_u8(cb0.as_ptr().add(src)),
            vld1_u8(cb1.as_ptr().add(src)),
        );
        let crq = vcombine_u8(
            vld1_u8(cr0.as_ptr().add(src)),
            vld1_u8(cr1.as_ptr().add(src)),
        );
        let (r_lo, g_lo, b_lo) = ycbcr_half_neon(
            vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(yq))),
            vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(cbq))),
            vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(crq))),
        );
        let (r_hi, g_hi, b_hi) = ycbcr_half_neon(
            vreinterpretq_s16_u16(vmovl_u8(vget_high_u8(yq))),
            vreinterpretq_s16_u16(vmovl_u8(vget_high_u8(cbq))),
            vreinterpretq_s16_u16(vmovl_u8(vget_high_u8(crq))),
        );
        let rgb = uint8x16x3_t(
            vcombine_u8(r_lo, r_hi),
            vcombine_u8(g_lo, g_hi),
            vcombine_u8(b_lo, b_hi),
        );
        let d = (y + r) * stride + x * 3;
        vst3q_u8(dst.as_mut_ptr().add(d), rgb);
    }
}

/// `pshufb` selector byte meaning "write zero".
#[cfg(target_arch = "x86_64")]
const Z: i8 = -0x80;

/// `pshufb` selectors that scatter 16 single-channel bytes into their slots in
/// the 48-byte interleaved RGB run. Output byte `j` holds channel `j % 3` of
/// pixel `j / 3`, so each channel contributes 5 or 6 bytes per 16-byte chunk.
#[cfg(target_arch = "x86_64")]
const RGB_SCATTER: [[[i8; 16]; 3]; 3] = [
    // chunk 0: bytes 0..15
    [
        [0, Z, Z, 1, Z, Z, 2, Z, Z, 3, Z, Z, 4, Z, Z, 5], // R
        [Z, 0, Z, Z, 1, Z, Z, 2, Z, Z, 3, Z, Z, 4, Z, Z], // G
        [Z, Z, 0, Z, Z, 1, Z, Z, 2, Z, Z, 3, Z, Z, 4, Z], // B
    ],
    // chunk 1: bytes 16..31
    [
        [Z, Z, 6, Z, Z, 7, Z, Z, 8, Z, Z, 9, Z, Z, 10, Z],
        [5, Z, Z, 6, Z, Z, 7, Z, Z, 8, Z, Z, 9, Z, Z, 10],
        [Z, 5, Z, Z, 6, Z, Z, 7, Z, Z, 8, Z, Z, 9, Z, Z],
    ],
    // chunk 2: bytes 32..47
    [
        [Z, 11, Z, Z, 12, Z, Z, 13, Z, Z, 14, Z, Z, 15, Z, Z],
        [Z, Z, 11, Z, Z, 12, Z, Z, 13, Z, Z, 14, Z, Z, 15, Z],
        [10, Z, Z, 11, Z, Z, 12, Z, Z, 13, Z, Z, 14, Z, Z, 15],
    ],
];

/// Interleave 16 pixels' worth of packed R/G/B bytes and store 48 bytes.
///
/// Replaces the byte-at-a-time store the earlier i32 kernel used: three
/// shuffles and two ORs per 16-byte chunk instead of 16 scalar writes.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse4.1")]
#[inline]
unsafe fn store_rgb16(r: __m128i, g: __m128i, b: __m128i, dst: *mut u8) {
    use core::arch::x86_64::*;
    for (chunk, sel) in RGB_SCATTER.iter().enumerate() {
        let out = _mm_or_si128(
            _mm_or_si128(
                _mm_shuffle_epi8(r, _mm_loadu_si128(sel[0].as_ptr() as *const __m128i)),
                _mm_shuffle_epi8(g, _mm_loadu_si128(sel[1].as_ptr() as *const __m128i)),
            ),
            _mm_shuffle_epi8(b, _mm_loadu_si128(sel[2].as_ptr() as *const __m128i)),
        );
        _mm_storeu_si128(dst.add(chunk * 16) as *mut __m128i, out);
    }
}

#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::__m128i;

/// SSE4.1 path: 16 pixels/iter in i16 lanes.
///
/// `pmulhrsw` computes `(a·b + 0x4000) >> 15`, which is exactly the scalar
/// `qrdmulh` and NEON's `SQRDMULH`, so all three paths are bit-identical.
/// One 8-pixel half: Q14 butterfly on i16 lanes. Shared by the row and
/// MCU-block SSE4.1 writers so both stay bit-identical (`pmulhrsw` matches
/// scalar `qrdmulh` and NEON `SQRDMULH` exactly).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse4.1")]
#[inline]
unsafe fn ycbcr_half_sse41(
    yv: core::arch::x86_64::__m128i,
    cbv: core::arch::x86_64::__m128i,
    crv: core::arch::x86_64::__m128i,
) -> (
    core::arch::x86_64::__m128i,
    core::arch::x86_64::__m128i,
    core::arch::x86_64::__m128i,
) {
    use core::arch::x86_64::*;
    let c128 = _mm_set1_epi16(128);
    let cb2 = _mm_slli_epi16::<1>(_mm_sub_epi16(cbv, c128));
    let cr2 = _mm_slli_epi16::<1>(_mm_sub_epi16(crv, c128));
    let r = _mm_add_epi16(yv, _mm_mulhrs_epi16(cr2, _mm_set1_epi16(CR_R)));
    let g = _mm_sub_epi16(
        _mm_sub_epi16(yv, _mm_mulhrs_epi16(cb2, _mm_set1_epi16(CB_G))),
        _mm_mulhrs_epi16(cr2, _mm_set1_epi16(CR_G)),
    );
    let b = _mm_add_epi16(yv, _mm_mulhrs_epi16(cb2, _mm_set1_epi16(CB_B)));
    (r, g, b)
}

/// Paired 8×8 blocks → 16 RGB pixels per row (SSE4.1). x86 mirror of
/// `ycbcr_to_rgb_block16_neon`: without it the fused-RGB write fell to the
/// scalar row tail on x86 (the 16-wide row kernel never engages at 8-pixel
/// block width), which cost ~70% on the RGB arm of Rocket Lake after the
/// per-MCU write path landed.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse4.1")]
#[allow(clippy::too_many_arguments)]
unsafe fn ycbcr_to_rgb_block16_sse41(
    y0: &[u8; 64],
    cb0: &[u8; 64],
    cr0: &[u8; 64],
    y1: &[u8; 64],
    cb1: &[u8; 64],
    cr1: &[u8; 64],
    dst: &mut [u8],
    stride: usize,
    x: usize,
    y: usize,
    rows: usize,
) {
    use core::arch::x86_64::*;
    let zero = _mm_setzero_si128();
    for r in 0..rows {
        let src = r * 8;
        let pair = |a: &[u8; 64], b: &[u8; 64]| -> __m128i {
            _mm_unpacklo_epi64(
                _mm_loadl_epi64(a.as_ptr().add(src) as *const __m128i),
                _mm_loadl_epi64(b.as_ptr().add(src) as *const __m128i),
            )
        };
        let yq = pair(y0, y1);
        let cbq = pair(cb0, cb1);
        let crq = pair(cr0, cr1);
        let (r_lo, g_lo, b_lo) = ycbcr_half_sse41(
            _mm_unpacklo_epi8(yq, zero),
            _mm_unpacklo_epi8(cbq, zero),
            _mm_unpacklo_epi8(crq, zero),
        );
        let (r_hi, g_hi, b_hi) = ycbcr_half_sse41(
            _mm_unpackhi_epi8(yq, zero),
            _mm_unpackhi_epi8(cbq, zero),
            _mm_unpackhi_epi8(crq, zero),
        );
        let d = (y + r) * stride + x * 3;
        store_rgb16(
            _mm_packus_epi16(r_lo, r_hi),
            _mm_packus_epi16(g_lo, g_hi),
            _mm_packus_epi16(b_lo, b_hi),
            dst.as_mut_ptr().add(d),
        );
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse4.1")]
unsafe fn ycbcr_to_rgb_row_sse41(y: &[u8], cb: &[u8], cr: &[u8], dst: &mut [u8], n: usize) {
    use core::arch::x86_64::*;

    #[inline(always)]
    unsafe fn half(yv: __m128i, cbv: __m128i, crv: __m128i) -> (__m128i, __m128i, __m128i) {
        ycbcr_half_sse41(yv, cbv, crv)
    }

    let mut i = 0usize;
    while i + 16 <= n {
        let yq = _mm_loadu_si128(y.as_ptr().add(i) as *const __m128i);
        let cbq = _mm_loadu_si128(cb.as_ptr().add(i) as *const __m128i);
        let crq = _mm_loadu_si128(cr.as_ptr().add(i) as *const __m128i);
        let zero = _mm_setzero_si128();

        let (r_lo, g_lo, b_lo) = half(
            _mm_unpacklo_epi8(yq, zero),
            _mm_unpacklo_epi8(cbq, zero),
            _mm_unpacklo_epi8(crq, zero),
        );
        let (r_hi, g_hi, b_hi) = half(
            _mm_unpackhi_epi8(yq, zero),
            _mm_unpackhi_epi8(cbq, zero),
            _mm_unpackhi_epi8(crq, zero),
        );

        store_rgb16(
            _mm_packus_epi16(r_lo, r_hi),
            _mm_packus_epi16(g_lo, g_hi),
            _mm_packus_epi16(b_lo, b_hi),
            dst.as_mut_ptr().add(i * 3),
        );
        i += 16;
    }
    ycbcr_to_rgb_row_scalar(y, cb, cr, dst, i, n);
}

/// AVX2 path: same Q14 butterfly, but all 16 pixels in one 256-bit i16 vector
/// instead of two 128-bit halves. The interleaved store stays 128-bit —
/// `vpshufb` cannot cross lanes, and the arithmetic is the expensive half.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "sse4.1")]
unsafe fn ycbcr_to_rgb_row_avx2(y: &[u8], cb: &[u8], cr: &[u8], dst: &mut [u8], n: usize) {
    use core::arch::x86_64::*;

    /// Pack 16 i16 lanes down to 16 u8 lanes with JPEG's [0, 255] clamp.
    #[inline(always)]
    unsafe fn pack16(v: __m256i) -> __m128i {
        _mm_packus_epi16(_mm256_castsi256_si128(v), _mm256_extracti128_si256::<1>(v))
    }

    let mut i = 0usize;
    while i + 16 <= n {
        let load = |p: *const u8| _mm256_cvtepu8_epi16(_mm_loadu_si128(p as *const __m128i));
        let yv = load(y.as_ptr().add(i));
        let c128 = _mm256_set1_epi16(128);
        let cb2 = _mm256_slli_epi16::<1>(_mm256_sub_epi16(load(cb.as_ptr().add(i)), c128));
        let cr2 = _mm256_slli_epi16::<1>(_mm256_sub_epi16(load(cr.as_ptr().add(i)), c128));

        let r = _mm256_add_epi16(yv, _mm256_mulhrs_epi16(cr2, _mm256_set1_epi16(CR_R)));
        let g = _mm256_sub_epi16(
            _mm256_sub_epi16(yv, _mm256_mulhrs_epi16(cb2, _mm256_set1_epi16(CB_G))),
            _mm256_mulhrs_epi16(cr2, _mm256_set1_epi16(CR_G)),
        );
        let b = _mm256_add_epi16(yv, _mm256_mulhrs_epi16(cb2, _mm256_set1_epi16(CB_B)));

        store_rgb16(pack16(r), pack16(g), pack16(b), dst.as_mut_ptr().add(i * 3));
        i += 16;
    }
    ycbcr_to_rgb_row_scalar(y, cb, cr, dst, i, n);
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Real-valued reference for the JFIF equations.
    fn reference(y: u8, cb: u8, cr: u8) -> [u8; 3] {
        let yf = y as f64;
        let cbf = cb as f64 - 128.0;
        let crf = cr as f64 - 128.0;
        let r = yf + 1.40200 * crf;
        let g = yf - 0.34414 * cbf - 0.71414 * crf;
        let b = yf + 1.77200 * cbf;
        [
            r.round().clamp(0.0, 255.0) as u8,
            g.round().clamp(0.0, 255.0) as u8,
            b.round().clamp(0.0, 255.0) as u8,
        ]
    }

    fn lcg_bytes(seed: u32, n: usize) -> Vec<u8> {
        let mut s = seed;
        (0..n)
            .map(|_| {
                s = s.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                (s >> 24) as u8
            })
            .collect()
    }

    #[test]
    fn scalar_matches_float_reference_within_1() {
        let n = 4096;
        let y = lcg_bytes(1, n);
        let cb = lcg_bytes(2, n);
        let cr = lcg_bytes(3, n);
        let mut out = vec![0u8; n * 3];
        ycbcr_to_rgb_row_scalar(&y, &cb, &cr, &mut out, 0, n);
        for i in 0..n {
            let exp = reference(y[i], cb[i], cr[i]);
            for c in 0..3 {
                let got = out[i * 3 + c] as i32;
                let want = exp[c] as i32;
                assert!(
                    (got - want).abs() <= 1,
                    "pixel {i} ch {c}: got {got} want {want} (y={} cb={} cr={})",
                    y[i],
                    cb[i],
                    cr[i]
                );
            }
        }
    }

    #[test]
    fn dispatch_matches_scalar_exactly() {
        // Odd length exercises the 16-, 8-, and scalar tails together.
        let n = 133;
        let y = lcg_bytes(7, n);
        let cb = lcg_bytes(11, n);
        let cr = lcg_bytes(13, n);
        let mut got = vec![0u8; n * 3];
        let mut exp = vec![0u8; n * 3];
        ycbcr_to_rgb_row(&y, &cb, &cr, &mut got, n);
        ycbcr_to_rgb_row_scalar(&y, &cb, &cr, &mut exp, 0, n);
        assert_eq!(got, exp);
    }

    #[test]
    fn mcu_blocks_match_row_kernel() {
        let mut y0 = [0u8; 64];
        let mut cb0 = [0u8; 64];
        let mut cr0 = [0u8; 64];
        let mut y1 = [0u8; 64];
        let mut cb1 = [0u8; 64];
        let mut cr1 = [0u8; 64];
        let a = lcg_bytes(3, 64);
        let b = lcg_bytes(5, 64);
        let c = lcg_bytes(7, 64);
        let d = lcg_bytes(11, 64);
        let e = lcg_bytes(13, 64);
        let f = lcg_bytes(17, 64);
        y0.copy_from_slice(&a);
        cb0.copy_from_slice(&b);
        cr0.copy_from_slice(&c);
        y1.copy_from_slice(&d);
        cb1.copy_from_slice(&e);
        cr1.copy_from_slice(&f);

        let stride = 16 * 3;
        let mut got = vec![0u8; 8 * stride];
        ycbcr_to_rgb_blocks(
            &y0,
            &cb0,
            &cr0,
            Some((&y1, &cb1, &cr1)),
            &mut got,
            stride,
            0,
            0,
            8,
            8,
            8,
        );

        let mut exp = vec![0u8; 8 * stride];
        for r in 0..8 {
            let mut yrow = [0u8; 16];
            let mut cbrow = [0u8; 16];
            let mut crrow = [0u8; 16];
            yrow[..8].copy_from_slice(&y0[r * 8..r * 8 + 8]);
            yrow[8..].copy_from_slice(&y1[r * 8..r * 8 + 8]);
            cbrow[..8].copy_from_slice(&cb0[r * 8..r * 8 + 8]);
            cbrow[8..].copy_from_slice(&cb1[r * 8..r * 8 + 8]);
            crrow[..8].copy_from_slice(&cr0[r * 8..r * 8 + 8]);
            crrow[8..].copy_from_slice(&cr1[r * 8..r * 8 + 8]);
            let d = r * stride;
            ycbcr_to_rgb_row_scalar(&yrow, &cbrow, &crrow, &mut exp[d..], 0, 16);
        }
        assert_eq!(got, exp);
    }

    /// Both x86 kernels must be bit-identical to the scalar model, not just
    /// whichever one this host's tier happens to select.
    #[cfg(target_arch = "x86_64")]
    #[test]
    fn x86_kernels_match_scalar_exactly() {
        // Lengths straddling the 16-pixel main loop and its scalar tail.
        for n in [1usize, 15, 16, 17, 31, 32, 48, 133] {
            let y = lcg_bytes(17, n);
            let cb = lcg_bytes(19, n);
            let cr = lcg_bytes(23, n);
            let mut exp = vec![0u8; n * 3];
            ycbcr_to_rgb_row_scalar(&y, &cb, &cr, &mut exp, 0, n);

            if is_x86_feature_detected!("sse4.1") {
                let mut got = vec![0u8; n * 3];
                // SAFETY: feature checked; buffers are sized n / 3n.
                unsafe { ycbcr_to_rgb_row_sse41(&y, &cb, &cr, &mut got, n) };
                assert_eq!(got, exp, "sse4.1 n={n}");
            }
            if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("sse4.1") {
                let mut got = vec![0u8; n * 3];
                // SAFETY: features checked; buffers are sized n / 3n.
                unsafe { ycbcr_to_rgb_row_avx2(&y, &cb, &cr, &mut got, n) };
                assert_eq!(got, exp, "avx2 n={n}");
            }
        }
    }

    #[test]
    fn grey_input_stays_grey() {
        // Cb = Cr = 128 must reproduce Y exactly on all three channels.
        let y: Vec<u8> = (0..=255).collect();
        let c = vec![128u8; 256];
        let mut out = vec![0u8; 256 * 3];
        ycbcr_to_rgb_row(&y, &c, &c, &mut out, 256);
        for (i, &yv) in y.iter().enumerate() {
            assert_eq!(&out[i * 3..i * 3 + 3], &[yv, yv, yv], "pixel {i}");
        }
    }
}
