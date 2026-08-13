// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! IDCT dispatcher — selects scalar, NEON, AVX2, SSE4.1, or SSE2.

pub mod fast;
pub mod scalar;

#[cfg(target_arch = "aarch64")]
pub mod neon;

#[cfg(target_arch = "x86_64")]
pub mod sse2;

#[cfg(target_arch = "x86_64")]
pub mod avx2;

/// IDCT function signature: takes 64 **quantised** coefficients and the
/// component's quantisation table, both in natural (row-major) order.
/// Dequantisation happens inside the kernel (libjpeg-turbo model: the entropy
/// decoder emits raw `i16` coefficients; SIMD kernels dequantise 8 lanes at a
/// time). Writes 64 clamped u8 values into `output` at the given stride.
///
/// # Range
///
/// Both arguments come from the file, so `coefficient × quantiser` is not
/// evaluated over the full `i16 × u16` range the types allow. As in
/// libjpeg-turbo, the dequantised block is held in `i16` and the multiply
/// **wraps**, and the workspace between the two passes is narrowed back to
/// `i16` **saturating**. JPEG syntax permits products that overflow both
/// (1023 × 255), but no encoder emits them: a large quantiser means the
/// encoder already divided the coefficient down to match.
///
/// What every kernel guarantees for such a stream is that it is absorbed as
/// defined arithmetic — no panic under overflow checks, no write outside the
/// block ([`tests::extreme_coefficients_do_not_panic_or_overrun`]). What no
/// kernel guarantees is the resulting pixel values, which past the saturation
/// point depend on where each kernel loses range. [`scalar`] is written to
/// wrap and saturate at exactly the points the vector kernels are forced to,
/// so the x86 kernels stay bit-exact with it over the whole input space; the
/// NEON kernel splits its final descale and so tracks the reference to ±1, as
/// its `assert_parity` records. Decoding a *valid* stream is therefore
/// tier-independent on x86 and tier-independent to ±1 on aarch64, which is the
/// same tolerance libjpeg-turbo allows between its own scalar and SIMD paths.
pub type IdctFn = fn(coeffs: &[i16; 64], quant: &[u16; 64], output: &mut [u8], stride: usize);

/// Select the best available IDCT implementation for this CPU.
pub fn select_idct() -> IdctFn {
    #[cfg(target_arch = "aarch64")]
    {
        use super::cpu::{neon_tier, NeonTier};
        match neon_tier() {
            NeonTier::Scalar => {}
            NeonTier::Baseline | NeonTier::Plus | NeonTier::High => {
                if std::arch::is_aarch64_feature_detected!("neon") {
                    return neon::idct_8x8_neon;
                }
            }
        }
    }
    #[cfg(target_arch = "x86_64")]
    {
        use super::cpu::{intel_tier, IntelTier};
        match intel_tier() {
            IntelTier::Scalar => {}
            IntelTier::Avx2 => {
                if is_x86_feature_detected!("avx2") {
                    return avx2::idct_8x8_avx2;
                }
                if is_x86_feature_detected!("sse2") {
                    return sse2::idct_8x8_sse2;
                }
            }
            // The islow kernel needs only SSE2 (`pmullw`, `pmaddwd`, `packssdw`,
            // `packuswb`), so SSE4.1 has nothing to add over it.
            IntelTier::Sse41 | IntelTier::Sse2 => {
                if is_x86_feature_detected!("sse2") {
                    return sse2::idct_8x8_sse2;
                }
            }
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
        use super::cpu::{neon_tier, NeonTier};
        match neon_tier() {
            NeonTier::Scalar => {}
            NeonTier::Baseline | NeonTier::Plus | NeonTier::High => {
                if std::arch::is_aarch64_feature_detected!("neon") {
                    return neon::idct_dc_only_neon;
                }
            }
        }
    }
    #[cfg(target_arch = "x86_64")]
    {
        use super::cpu::{intel_tier, IntelTier};
        match intel_tier() {
            IntelTier::Scalar => {}
            IntelTier::Avx2 | IntelTier::Sse41 | IntelTier::Sse2 => {
                if is_x86_feature_detected!("sse2") {
                    return sse2::idct_dc_only_sse2;
                }
            }
        }
    }
    scalar::idct_dc_only_scalar
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Coefficients and quantisers both come from the file, so their product
    /// can overflow the `i16` the kernels dequantise into — JPEG syntax allows
    /// 1023 × 255 even though no encoder emits it. Every kernel, the scalar
    /// reference included, must absorb that as defined wrapping/saturating
    /// arithmetic: no panic under a debug build's overflow checks, and no write
    /// outside the 8×8 block. Pixel *values* past the overflow point are
    /// deliberately not asserted equal across tiers — see [`IdctFn`].
    #[test]
    fn extreme_coefficients_do_not_panic_or_overrun() {
        let mut seed = 0x9E37_79B9_7F4A_7C15u64;
        let mut next = move || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            seed
        };

        const STRIDE: usize = 12;
        let kernels: [(&str, IdctFn); 2] = [
            ("scalar", scalar::idct_8x8_scalar),
            ("selected", select_idct()),
        ];

        for (name, idct) in kernels {
            for case in 0..256 {
                let mut coeffs = [0i16; 64];
                let mut quant = [1u16; 64];
                for (c, q) in coeffs.iter_mut().zip(quant.iter_mut()) {
                    *c = match case % 4 {
                        0 => i16::MIN,
                        1 => i16::MAX,
                        _ => (next() % 65536) as u16 as i16,
                    };
                    *q = if case % 2 == 0 {
                        u16::MAX
                    } else {
                        (next() % 65536) as u16
                    };
                }

                let mut out = [0xAAu8; 8 * STRIDE];
                idct(&coeffs, &quant, &mut out, STRIDE);
                for row in 0..8 {
                    for col in 8..STRIDE {
                        assert_eq!(
                            out[row * STRIDE + col],
                            0xAA,
                            "{name} case {case}: wrote past column 8"
                        );
                    }
                }
            }
        }
    }

    /// The DC-only shortcut has to agree with the scalar reference over the
    /// whole `i32` argument, not just the range a valid stream produces: the
    /// value is a dequantised coefficient, and the MCU loop reaches the
    /// shortcut on corrupt blocks too. The fixed-point scaling inside these
    /// kernels overflows well before `i32::MAX`, which is a debug-build panic
    /// unless the arithmetic is written to wrap.
    #[test]
    fn selected_dc_only_matches_scalar() {
        let dc_only = select_idct_dc_only();
        let mut cases = vec![
            0i32,
            1,
            8,
            64,
            128,
            255,
            1023,
            4095,
            -1,
            -64,
            -1023,
            -4095,
            i32::MAX,
            i32::MIN,
            i32::MAX / 2,
            i32::MIN / 2,
        ];
        // Either side of where `dc << 15` stops fitting.
        cases.extend((14..20).flat_map(|s| [1i32 << s, -(1i32 << s), (1i32 << s) + 1]));

        for dc in cases {
            let mut want = [0u8; 64];
            let mut got = [0u8; 64];
            scalar::idct_dc_only_scalar(dc, &mut want, 8);
            dc_only(dc, &mut got, 8);
            assert_eq!(got, want, "dc={dc}: tier disagrees with scalar");
        }
    }

    /// The DC-only shortcut stands in for the full kernel on blocks whose AC
    /// coefficients are all zero, so it has to produce what the full kernel
    /// would — including on the corrupt blocks the MCU loop still routes
    /// through it. `dc_only_coefficient` in `mcu.rs` is what keeps the two
    /// paths dequantising the same way.
    #[test]
    fn dc_only_shortcut_matches_the_full_kernel() {
        let idct = select_idct();
        let dc_only = select_idct_dc_only();

        for &coeff in &[0i16, 1, -1, 17, -17, 255, 1023, -1023, i16::MAX, i16::MIN] {
            for &q in &[1u16, 2, 16, 255, 4096, u16::MAX] {
                let mut coeffs = [0i16; 64];
                coeffs[0] = coeff;
                let mut quant = [1u16; 64];
                quant[0] = q;

                let mut want = [0u8; 64];
                let mut got = [0u8; 64];
                idct(&coeffs, &quant, &mut want, 8);
                dc_only(coeff.wrapping_mul(q as i16) as i32, &mut got, 8);
                assert_eq!(got, want, "coeff={coeff} quant={q}: shortcut diverged");
            }
        }
    }
}

#[cfg(test)]
mod bench {
    use super::*;
    use std::time::Instant;

    /// Blocks with a JPEG-like sparsity profile: energy concentrated in the low
    /// frequencies, ~8 non-zero AC coefficients, so the kernels see the branch
    /// and data mix they meet on real photographic content.
    fn corpus(n: usize) -> Vec<([i16; 64], [u16; 64])> {
        let mut seed = 0x2545_F491_4F6C_DD1Du64;
        let mut next = move || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            seed
        };
        (0..n)
            .map(|_| {
                let mut c = [0i16; 64];
                c[0] = (next() % 2048) as i16 - 1024;
                for k in 1..64 {
                    // Decreasing occupancy with frequency, as in real DCT blocks.
                    if next() % 64 < (48 / (1 + k as u64 / 4)).max(1) {
                        let mag = (256 / (1 + k as i32 / 8)).max(2);
                        c[crate::jpeg::types::ZIGZAG[k] as usize] =
                            ((next() % (2 * mag as u64)) as i32 - mag) as i16;
                    }
                }
                let mut q = [1u16; 64];
                for (i, v) in q.iter_mut().enumerate() {
                    *v = (2 + (i as u16) / 4).min(32);
                }
                (c, q)
            })
            .collect()
    }

    /// Per-block cost of the selected IDCT. Not a correctness test — run with
    /// `cargo test -p edgefirst-codec --release -- --ignored --nocapture idct_kernel_cost`.
    #[test]
    #[ignore = "timing benchmark, not a correctness check"]
    fn idct_kernel_cost() {
        let blocks = corpus(4096);
        let idct = select_idct();
        let mut out = vec![0u8; 64 * 8];

        for _ in 0..8 {
            for (c, q) in &blocks {
                idct(c, q, &mut out, 8);
            }
        }

        let reps = 64;
        let t = Instant::now();
        for _ in 0..reps {
            for (c, q) in &blocks {
                idct(c, q, &mut out, 8);
            }
        }
        let el = t.elapsed();
        let n = (reps * blocks.len()) as f64;
        println!(
            "idct: {:.2} ns/block  ({:.0} blocks/ms, {} blocks timed) [{}]",
            el.as_secs_f64() * 1e9 / n,
            n / el.as_secs_f64() / 1e3,
            n as u64,
            std::hint::black_box(out[0]),
        );
    }
}
