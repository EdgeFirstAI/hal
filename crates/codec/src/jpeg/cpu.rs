// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! Runtime CPU tier selection for the software JPEG path.
//!
//! ## AArch64 — [`NeonTier`]
//!
//! A single aarch64 binary must run well on Cortex-A53 (in-order, no DOTPROD),
//! Cortex-A55 (in-order, DOTPROD), and big out-of-order cores: Cortex-A7x
//! (Jetson Orin A78AE, Raspberry Pi 5 A76) and Apple Silicon (M2+/A16+).
//!
//! | Tier | Probe | Typical cores | Huffman bits |
//! |------|-------|---------------|--------------|
//! | `Baseline` | NEON only | A53, A57, A72 | 10 |
//! | `Plus` | +`dotprod`, LITTLE core | A55, A510 | 10 |
//! | `High` | +`i8mm` **or** big-core MIDR | A76/A78(AE), X1+, Apple M2+ | 11 |
//!
//! `HWCAP` alone cannot split A55 from A76/A78 (both are `dotprod`, no
//! `i8mm`), so on Linux the probe also reads MIDR part numbers from sysfs and
//! promotes known big OoO cores to `High` (measured ~3% faster entropy decode
//! on Cortex-A76 from the wider Huffman lookahead). On Apple platforms `i8mm`
//! is present from M2/A15 onward and the HWCAP-style probe suffices.
//!
//! Override with `EDGEFIRST_CODEC_FORCE_NEON=scalar|baseline|plus|high` for A/B.
//!
//! ## x86_64 — [`IntelTier`]
//!
//! | Tier | Probe | Kernels |
//! |------|-------|---------|
//! | `Scalar` | forced / no SSE2 | scalar only |
//! | `Sse2` | `sse2` | SSE2 IDCT, UV interleave, UV downsample |
//! | `Sse41` | `sse4.1` | SSE4.1 IDCT + SSSE3/SSE4.1 color |
//! | `Avx2` | `avx2` | AVX2 IDCT + wider color/downsample |
//!
//! Override with `EDGEFIRST_CODEC_FORCE_INTEL=scalar|sse2|sse41|avx2` for A/B.

use std::sync::OnceLock;

/// Selected NEON / micro-arch tier for JPEG kernels.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NeonTier {
    /// Scalar fallbacks (no NEON, or forced off).
    Scalar,
    /// Baseline ASIMD (Cortex-A53-class).
    Baseline,
    /// DOTPROD-class (Cortex-A55 and many A7x).
    Plus,
    /// I8MM-class big cores.
    High,
}

impl NeonTier {
    /// Huffman fast-lookup width.
    ///
    /// Trade-off vs libjpeg-turbo's 8-bit `HUFF_LOOKAHEAD`: wider tables raise
    /// the fast-path hit rate (JPEG AC codes often need 9–11 bits) but cost L1
    /// (2^N × 2 bytes × ~4 tables). Measured on Cortex-A55 (32 KiB L1D): 10–11
    /// bits beat 8/9 despite the larger footprint; 12 bits starts to thrash
    /// when all four tables are hot. A53 originally stayed at 8, but with the
    /// combined code+magnitude fast-AC LUT a hit now covers the whole
    /// coefficient, and re-measurement on imx8mp put 10 bits ~0.3 ms/frame
    /// ahead of 8 on COCO (9 in between).
    #[inline]
    #[cfg_attr(not(target_arch = "aarch64"), allow(dead_code))]
    pub fn huffman_fast_bits(self) -> u8 {
        match self {
            Self::Scalar | Self::Baseline => 10, // A53: re-measured with fast-AC LUT
            Self::Plus => 10,                    // A55: measured sweet spot
            Self::High => 11,                    // A7x/X: 64 KiB L1, wider peek
        }
    }

    /// Prefetch distance (bytes ahead in the entropy stream).
    /// A53 dual-issues `prfm` for free; keep distances modest to avoid pollution.
    #[inline]
    #[cfg_attr(not(target_arch = "aarch64"), allow(dead_code))]
    pub fn entropy_prefetch(self) -> usize {
        match self {
            Self::Scalar => 0,
            Self::Baseline => 192,
            Self::Plus => 192,
            Self::High => 256,
        }
    }
}

/// Selected Intel / x86_64 ISA tier for JPEG kernels.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IntelTier {
    /// Scalar fallbacks (no SSE2, or forced off).
    Scalar,
    /// Baseline SSE2.
    Sse2,
    /// SSE4.1 (+ SSSE3 for color).
    Sse41,
    /// AVX2 hot paths.
    Avx2,
}

#[cfg_attr(not(target_arch = "x86_64"), allow(dead_code))]
impl IntelTier {
    /// Huffman fast-lookup width (same L1 trade-offs as [`NeonTier`]).
    #[inline]
    pub fn huffman_fast_bits(self) -> u8 {
        match self {
            Self::Scalar | Self::Sse2 | Self::Sse41 => 10,
            Self::Avx2 => 11,
        }
    }

    /// Prefetch distance for the entropy stream (`_mm_prefetch` on x86).
    #[inline]
    pub fn entropy_prefetch(self) -> usize {
        match self {
            Self::Scalar => 0,
            Self::Sse2 | Self::Sse41 => 192,
            Self::Avx2 => 256,
        }
    }

    /// True when SSE2 kernels may run.
    #[inline]
    pub fn has_sse2(self) -> bool {
        !matches!(self, Self::Scalar)
    }

    /// True when SSE4.1 kernels may run.
    #[inline]
    pub fn has_sse41(self) -> bool {
        matches!(self, Self::Sse41 | Self::Avx2)
    }

    /// True when AVX2 kernels may run.
    #[inline]
    pub fn has_avx2(self) -> bool {
        matches!(self, Self::Avx2)
    }
}

/// Probe once per process.
#[cfg_attr(not(target_arch = "aarch64"), allow(dead_code))]
pub fn neon_tier() -> NeonTier {
    static TIER: OnceLock<NeonTier> = OnceLock::new();
    *TIER.get_or_init(probe_neon_tier)
}

/// Probe once per process (x86_64 ISA tier).
#[cfg_attr(not(target_arch = "x86_64"), allow(dead_code))]
pub fn intel_tier() -> IntelTier {
    static TIER: OnceLock<IntelTier> = OnceLock::new();
    *TIER.get_or_init(probe_intel_tier)
}

/// Huffman fast-bits for the active arch tier.
#[inline]
pub fn entropy_huffman_fast_bits() -> u8 {
    #[cfg(target_arch = "aarch64")]
    {
        neon_tier().huffman_fast_bits()
    }
    #[cfg(target_arch = "x86_64")]
    {
        intel_tier().huffman_fast_bits()
    }
    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
    {
        10
    }
}

/// Entropy prefetch distance for the active arch tier.
#[inline]
pub fn entropy_prefetch_distance() -> usize {
    #[cfg(target_arch = "aarch64")]
    {
        neon_tier().entropy_prefetch()
    }
    #[cfg(target_arch = "x86_64")]
    {
        intel_tier().entropy_prefetch()
    }
    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
    {
        0
    }
}

fn probe_neon_tier() -> NeonTier {
    if let Ok(forced) = std::env::var("EDGEFIRST_CODEC_FORCE_NEON") {
        match forced.trim().to_ascii_lowercase().as_str() {
            "scalar" | "off" | "0" => return NeonTier::Scalar,
            "baseline" | "a53" | "neon" => return NeonTier::Baseline,
            "plus" | "a55" | "dotprod" => return NeonTier::Plus,
            "high" | "a76" | "a7x" | "i8mm" => return NeonTier::High,
            _ => {}
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        // NEON is mandatory for aarch64 userland we care about, but honour the
        // feature bit for completeness / exotic toolchains.
        if !std::arch::is_aarch64_feature_detected!("neon") {
            return NeonTier::Scalar;
        }
        if std::arch::is_aarch64_feature_detected!("i8mm") {
            return NeonTier::High;
        }
        if std::arch::is_aarch64_feature_detected!("dotprod") {
            if has_big_ooo_core() {
                return NeonTier::High;
            }
            return NeonTier::Plus;
        }
        NeonTier::Baseline
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        NeonTier::Scalar
    }
}

fn probe_intel_tier() -> IntelTier {
    if let Ok(forced) = std::env::var("EDGEFIRST_CODEC_FORCE_INTEL") {
        match forced.trim().to_ascii_lowercase().as_str() {
            "scalar" | "off" | "0" => return IntelTier::Scalar,
            "sse2" | "sse" => return IntelTier::Sse2,
            "sse41" | "sse4.1" | "sse4_1" | "ssse3" => return IntelTier::Sse41,
            "avx2" | "avx" => return IntelTier::Avx2,
            _ => {}
        }
    }

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            return IntelTier::Avx2;
        }
        if is_x86_feature_detected!("sse4.1") {
            return IntelTier::Sse41;
        }
        if is_x86_feature_detected!("sse2") {
            return IntelTier::Sse2;
        }
        IntelTier::Scalar
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        IntelTier::Scalar
    }
}

/// Detect a big out-of-order Arm core (Cortex-A76 and newer) from MIDR part
/// numbers exposed in sysfs. HWCAP cannot make this distinction: A55 and
/// A76/A78AE all report `dotprod` without `i8mm`.
///
/// Any big core qualifying promotes the tier — on big.LITTLE the decode worker
/// overwhelmingly runs on a big core under load, and the `High` policy is only
/// mildly worse on a LITTLE core (1 KiB more LUT).
#[cfg(all(target_arch = "aarch64", target_os = "linux"))]
fn has_big_ooo_core() -> bool {
    // Arm Ltd (implementer 0x41) part IDs for big OoO cores.
    // A76 0xD0B, N1 0xD0C, A77 0xD0D, V1 0xD40, A78 0xD41, A78AE 0xD42,
    // X1 0xD44, A710 0xD47, X2 0xD48, A715 0xD4D, X3 0xD4E, A720 0xD81.
    const BIG_PARTS: &[u32] = &[
        0xD0B, 0xD0C, 0xD0D, 0xD40, 0xD41, 0xD42, 0xD44, 0xD47, 0xD48, 0xD4D, 0xD4E, 0xD81,
    ];
    let Ok(entries) = std::fs::read_dir("/sys/devices/system/cpu") else {
        return false;
    };
    for entry in entries.flatten() {
        let name = entry.file_name();
        let Some(name) = name.to_str() else { continue };
        if !name.starts_with("cpu") || !name[3..].bytes().all(|b| b.is_ascii_digit()) {
            continue;
        }
        let midr_path = entry.path().join("regs/identification/midr_el1");
        let Ok(text) = std::fs::read_to_string(midr_path) else {
            continue;
        };
        let text = text.trim().trim_start_matches("0x");
        let Ok(midr) = u64::from_str_radix(text, 16) else {
            continue;
        };
        let implementer = ((midr >> 24) & 0xFF) as u32;
        let part = ((midr >> 4) & 0xFFF) as u32;
        if implementer == 0x41 && BIG_PARTS.contains(&part) {
            return true;
        }
    }
    false
}

#[cfg(all(target_arch = "aarch64", not(target_os = "linux")))]
fn has_big_ooo_core() -> bool {
    false // Apple Silicon reaches `High` via i8mm; no sysfs elsewhere.
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn probe_neon_returns_something_sane() {
        let t = probe_neon_tier();
        let _ = t.huffman_fast_bits();
        let _ = t.entropy_prefetch();
    }

    #[test]
    fn probe_intel_returns_something_sane() {
        let t = probe_intel_tier();
        let _ = t.huffman_fast_bits();
        let _ = t.entropy_prefetch();
        let _ = t.has_sse2();
        let _ = t.has_sse41();
        let _ = t.has_avx2();
    }

    #[test]
    fn entropy_helpers_are_stable() {
        let _ = entropy_huffman_fast_bits();
        let _ = entropy_prefetch_distance();
    }
}
