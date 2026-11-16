// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! SIMD row kernels for the CPU packed↔planar conversions.
//!
//! A NEON deinterleaving load (`vld3q_u8` / `vld4q_u8`) reads the packed
//! source once and splits R/G/B (and A) into separate registers, so one pass
//! writes every plane.
//!
//! The deinterleave and the u8→f32 widen use baseline AArch64 NEON. The
//! u8→f16 widen uses native half-precision arithmetic (FEAT_FP16) behind a
//! cached runtime probe ([`has_fp16`]), through inline `asm!` with
//! `.arch_extension fp16`. Non-aarch64 targets use the scalar fallback.

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::uint16x8_t;

/// Deinterleave one packed row into up to four destination planes.
///
/// `src_ch` must be 3 (RGB) or 4 (RGBA). The R, G, B planes are always written
/// from source channels 0, 1, 2. `a` is written from source channel 3 when
/// `Some` (requires `src_ch == 4`); a constant alpha plane is the caller's
/// responsibility.
///
/// Only the first `w` pixels of each plane and the first `w * src_ch` bytes of
/// `src` are touched; callers pass full row slices (stride length) and the
/// logical width `w`.
#[inline]
pub(super) fn deinterleave_row(
    src: &[u8],
    r: &mut [u8],
    g: &mut [u8],
    b: &mut [u8],
    a: Option<&mut [u8]>,
    w: usize,
    src_ch: usize,
) {
    debug_assert!(src_ch == 3 || src_ch == 4);
    debug_assert!(src.len() >= w * src_ch);
    debug_assert!(r.len() >= w && g.len() >= w && b.len() >= w);
    debug_assert!(a.as_ref().is_none_or(|a| a.len() >= w));
    // An alpha plane is sourced from a 4-channel input only.
    debug_assert!(
        a.is_none() || src_ch == 4,
        "alpha output plane requires a 4-channel source"
    );

    #[cfg(target_arch = "aarch64")]
    // SAFETY: NEON is baseline on aarch64; the caller provides the lengths asserted above.
    unsafe {
        deinterleave_row_neon(src, r, g, b, a, w, src_ch);
    }
    #[cfg(not(target_arch = "aarch64"))]
    deinterleave_row_scalar(src, r, g, b, a, w, src_ch);
}

/// Portable single-pass scalar deinterleave: one source read, contiguous plane
/// stores. Used on non-aarch64.
#[cfg(not(target_arch = "aarch64"))]
fn deinterleave_row_scalar(
    src: &[u8],
    r: &mut [u8],
    g: &mut [u8],
    b: &mut [u8],
    a: Option<&mut [u8]>,
    w: usize,
    src_ch: usize,
) {
    match a {
        // Alpha is sourced only when the input has a 4th channel.
        Some(a) if src_ch == 4 => {
            for x in 0..w {
                let p = &src[x * src_ch..];
                r[x] = p[0];
                g[x] = p[1];
                b[x] = p[2];
                a[x] = p[3];
            }
        }
        _ => {
            for x in 0..w {
                let p = &src[x * src_ch..];
                r[x] = p[0];
                g[x] = p[1];
                b[x] = p[2];
            }
        }
    }
}

/// Widen a `u8` buffer into an `f32` buffer, dividing each value by 255.0.
///
/// The NEON path converts 16 lanes per iteration with a true division
/// (`vdivq_f32` by 255.0); the result is bit-identical to the scalar
/// `b as f32 / 255.0`.
///
/// # Panics (debug)
/// Panics in debug mode if `src.len() != dst.len()`.
#[inline]
pub(super) fn widen_u8_to_f32_norm(src: &[u8], dst: &mut [f32]) {
    debug_assert_eq!(src.len(), dst.len());
    #[cfg(target_arch = "aarch64")]
    // SAFETY: NEON is baseline on aarch64; the lengths are asserted equal above.
    unsafe {
        widen_u8_to_f32_norm_neon(src, dst);
    }
    #[cfg(not(target_arch = "aarch64"))]
    for (o, &b) in dst.iter_mut().zip(src.iter()) {
        *o = b as f32 / 255.0;
    }
}

/// NEON u8 → f32 widen with `/255.0` normalisation, 16 lanes/iteration:
/// `vld1q_u8` → two `vmovl_u8` (u16) → four `vmovl_u16`+`vcvtq_f32_u32` (f32)
/// → `vdivq_f32` by 255.0. Scalar tail for the `len % 16` remainder.
///
/// # Safety
/// `src.len()` must equal `dst.len()`. NEON is guaranteed by the aarch64 target.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn widen_u8_to_f32_norm_neon(src: &[u8], dst: &mut [f32]) {
    unsafe {
        use std::arch::aarch64::*; // NOSONAR

        let n = src.len();
        let sp = src.as_ptr();
        let dp = dst.as_mut_ptr();
        let denom = vdupq_n_f32(255.0);

        let mut i = 0usize;
        while i + 16 <= n {
            let v = vld1q_u8(sp.add(i)); // 16 × u8
            let lo = vmovl_u8(vget_low_u8(v)); // 8 × u16
            let hi = vmovl_u8(vget_high_u8(v)); // 8 × u16
            let f0 = vdivq_f32(vcvtq_f32_u32(vmovl_u16(vget_low_u16(lo))), denom);
            let f1 = vdivq_f32(vcvtq_f32_u32(vmovl_u16(vget_high_u16(lo))), denom);
            let f2 = vdivq_f32(vcvtq_f32_u32(vmovl_u16(vget_low_u16(hi))), denom);
            let f3 = vdivq_f32(vcvtq_f32_u32(vmovl_u16(vget_high_u16(hi))), denom);
            vst1q_f32(dp.add(i), f0);
            vst1q_f32(dp.add(i + 4), f1);
            vst1q_f32(dp.add(i + 8), f2);
            vst1q_f32(dp.add(i + 12), f3);
            i += 16;
        }
        while i < n {
            *dp.add(i) = *sp.add(i) as f32 / 255.0;
            i += 1;
        }
    }
}

/// Runtime FEAT_FP16 (native half-precision arithmetic, `asimdhp`) availability,
/// probed once and cached. Setting `EDGEFIRST_IMAGE_NO_FP16` forces the scalar
/// fallback.
#[cfg(target_arch = "aarch64")]
fn has_fp16() -> bool {
    use std::sync::OnceLock;
    static FP16: OnceLock<bool> = OnceLock::new();
    *FP16.get_or_init(|| {
        if std::env::var_os("EDGEFIRST_IMAGE_NO_FP16").is_some() {
            return false;
        }
        std::arch::is_aarch64_feature_detected!("fp16")
    })
}

/// Widen a `u8` buffer into an `f16` buffer, dividing each value by 255.0.
///
/// On AArch64 CPUs with the FP16 extension (FEAT_FP16 / `asimdhp`) this
/// dispatches at runtime to a native half-precision kernel (`ucvtf.8h` +
/// `fdiv.8h`, 8 lanes/instruction). Other CPUs use the scalar
/// `half::f16::from_f32`.
///
/// The native path divides in the f16 domain, which differs from the scalar
/// f32-then-narrow result by at most ~1 f16 ULP.
///
/// # Panics (debug)
/// Panics in debug mode if `src.len() != dst.len()`.
#[inline]
pub(super) fn widen_u8_to_f16_norm(src: &[u8], dst: &mut [half::f16]) {
    debug_assert_eq!(src.len(), dst.len());
    #[cfg(target_arch = "aarch64")]
    if has_fp16() {
        // SAFETY: FEAT_FP16 was detected at runtime; the lengths are asserted equal above.
        unsafe {
            widen_u8_to_f16_norm_fp16(src, dst);
        }
        return;
    }
    for (o, &b) in dst.iter_mut().zip(src.iter()) {
        *o = half::f16::from_f32(b as f32 / 255.0);
    }
}

/// Native-FP16 u8 → f16 widen with `/255.0`, 16 u8/iteration: `vld1q_u8` →
/// two `vmovl_u8` (u16) → [`ucvtf_div255_f16x8`] per 8-lane half → `vst1q_u16`.
/// Scalar tail for the `len % 16` remainder.
///
/// # Safety
/// Requires FEAT_FP16 (caller gates on [`has_fp16`]); `src.len() == dst.len()`.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn widen_u8_to_f16_norm_fp16(src: &[u8], dst: &mut [half::f16]) {
    unsafe {
        use std::arch::aarch64::*; // NOSONAR

        let n = src.len();
        let sp = src.as_ptr();
        let dp = dst.as_mut_ptr() as *mut u16; // half::f16 is repr(transparent) over u16
        let d255 = vdupq_n_u16(half::f16::from_f32(255.0).to_bits());

        let mut i = 0usize;
        while i + 16 <= n {
            let v = vld1q_u8(sp.add(i)); // 16 × u8
            let lo = vmovl_u8(vget_low_u8(v)); // 8 × u16 (0..=255)
            let hi = vmovl_u8(vget_high_u8(v)); // 8 × u16
            vst1q_u16(dp.add(i), ucvtf_div255_f16x8(lo, d255));
            vst1q_u16(dp.add(i + 8), ucvtf_div255_f16x8(hi, d255));
            i += 16;
        }
        while i < n {
            *dst.get_unchecked_mut(i) = half::f16::from_f32(*sp.add(i) as f32 / 255.0);
            i += 1;
        }
    }
}

/// `f16(lane) / divisor` on 8 lanes via native FP16: `ucvtf.8h` converts the
/// unsigned-int u16 lanes (0..=255 from the widened u8) to f16, then `fdiv.8h`
/// divides by the broadcast f16 divisor. Uses inline `asm!` with
/// `.arch_extension fp16`; the f16 vectors are carried as `uint16x8_t`
/// bit-patterns.
///
/// # Safety
/// Requires FEAT_FP16 (caller gates on [`has_fp16`]).
#[cfg(target_arch = "aarch64")]
#[inline]
#[target_feature(enable = "neon")]
unsafe fn ucvtf_div255_f16x8(u16_lanes: uint16x8_t, divisor_f16: uint16x8_t) -> uint16x8_t {
    unsafe {
        let result: uint16x8_t;
        core::arch::asm!(
            ".arch_extension fp16",
            "ucvtf {r:v}.8h, {x:v}.8h",
            "fdiv {r:v}.8h, {r:v}.8h, {d:v}.8h",
            r = out(vreg) result,
            x = in(vreg) u16_lanes,
            d = in(vreg) divisor_f16,
            options(pure, nomem, nostack),
        );
        result
    }
}

/// NEON deinterleave: 16 pixels/iteration via `vld3q_u8` (RGB) or `vld4q_u8`
/// (RGBA), with a scalar tail for the `w % 16` remainder.
///
/// # Safety
/// `src` must hold at least `w * src_ch` bytes and each plane at least `w`
/// bytes; `src_ch` must be 3 or 4. NEON availability is guaranteed by the
/// aarch64 target (baseline ISA).
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn deinterleave_row_neon(
    src: &[u8],
    r: &mut [u8],
    g: &mut [u8],
    b: &mut [u8],
    a: Option<&mut [u8]>,
    w: usize,
    src_ch: usize,
) {
    unsafe {
        use std::arch::aarch64::*; // NOSONAR

        let sp = src.as_ptr();
        let rp = r.as_mut_ptr();
        let gp = g.as_mut_ptr();
        let bp = b.as_mut_ptr();
        // The alpha plane as a raw pointer.
        let ap: Option<*mut u8> = a.map(|s| s.as_mut_ptr());

        let mut x = 0usize;
        if src_ch == 3 {
            while x + 16 <= w {
                let v = vld3q_u8(sp.add(x * 3));
                vst1q_u8(rp.add(x), v.0);
                vst1q_u8(gp.add(x), v.1);
                vst1q_u8(bp.add(x), v.2);
                x += 16;
            }
        } else {
            while x + 16 <= w {
                let v = vld4q_u8(sp.add(x * 4));
                vst1q_u8(rp.add(x), v.0);
                vst1q_u8(gp.add(x), v.1);
                vst1q_u8(bp.add(x), v.2);
                if let Some(ap) = ap {
                    vst1q_u8(ap.add(x), v.3);
                }
                x += 16;
            }
        }

        // Scalar tail for the remaining < 16 pixels. The alpha write is gated on
        // `src_ch == 4`.
        let write_alpha = src_ch == 4;
        while x < w {
            *rp.add(x) = *sp.add(x * src_ch);
            *gp.add(x) = *sp.add(x * src_ch + 1);
            *bp.add(x) = *sp.add(x * src_ch + 2);
            if write_alpha {
                if let Some(ap) = ap {
                    *ap.add(x) = *sp.add(x * src_ch + 3);
                }
            }
            x += 1;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Deinterleave matches the scalar reference for widths that straddle the 16-pixel vector boundary.
    #[test]
    fn deinterleave_rgb_matches_scalar() {
        for w in [1usize, 7, 16, 17, 31, 64, 100] {
            let src: Vec<u8> = (0..w * 3).map(|i| (i * 7 % 256) as u8).collect();
            let (mut r, mut g, mut b) = (vec![0u8; w], vec![0u8; w], vec![0u8; w]);
            deinterleave_row(&src, &mut r, &mut g, &mut b, None, w, 3);
            for x in 0..w {
                assert_eq!(r[x], src[x * 3], "R w={w} x={x}");
                assert_eq!(g[x], src[x * 3 + 1], "G w={w} x={x}");
                assert_eq!(b[x], src[x * 3 + 2], "B w={w} x={x}");
            }
        }
    }

    /// Deinterleave of a 4-channel source into R,G,B,A planes.
    #[test]
    fn deinterleave_rgba_matches_scalar() {
        for w in [3usize, 16, 19, 48, 77] {
            let src: Vec<u8> = (0..w * 4).map(|i| (i * 5 % 256) as u8).collect();
            let (mut r, mut g, mut b, mut a) =
                (vec![0u8; w], vec![0u8; w], vec![0u8; w], vec![9u8; w]);
            deinterleave_row(&src, &mut r, &mut g, &mut b, Some(&mut a), w, 4);
            for x in 0..w {
                assert_eq!(r[x], src[x * 4], "R w={w} x={x}");
                assert_eq!(g[x], src[x * 4 + 1], "G w={w} x={x}");
                assert_eq!(b[x], src[x * 4 + 2], "B w={w} x={x}");
                assert_eq!(a[x], src[x * 4 + 3], "A w={w} x={x}");
            }
        }
    }

    /// The NEON u8→f32 `/255` widen is bit-identical to the scalar form.
    #[test]
    fn widen_f32_bit_identical() {
        let src: Vec<u8> = (0..=255u8).collect();
        let mut dst = vec![0f32; src.len()];
        widen_u8_to_f32_norm(&src, &mut dst);
        for (i, &v) in dst.iter().enumerate() {
            assert_eq!(v.to_bits(), (i as f32 / 255.0).to_bits(), "byte {i}");
        }
    }

    /// The u8→f16 widen is within ~1 f16 ULP of the scalar f32-then-narrow reference.
    #[test]
    fn widen_f16_within_tolerance() {
        let src: Vec<u8> = (0..=255u8).collect();
        let mut dst = vec![half::f16::ZERO; src.len()];
        widen_u8_to_f16_norm(&src, &mut dst);
        for (i, &v) in dst.iter().enumerate() {
            let expected = half::f16::from_f32(i as f32 / 255.0).to_f32();
            assert!(
                (v.to_f32() - expected).abs() <= 1e-3,
                "byte {i}: got {} expected {expected}",
                v.to_f32()
            );
        }
    }
}
