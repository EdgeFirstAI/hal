// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

use crate::DecoderError;

/// Detected CPU instruction-set features.
///
/// Probed once at `DecoderBuilder::build()` and cached in the plan.
/// `from_env_or_probe()` reads `EDGEFIRST_DECODER_FORCE_KERNEL` for
/// debugging / benchmarking overrides.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
#[allow(dead_code)] // Consumed by dispatch tables in later tasks.
pub(crate) struct CpuFeatures {
    pub(crate) neon_baseline: bool,
    pub(crate) neon_fp16: bool,
    pub(crate) neon_dotprod: bool,
    pub(crate) neon_i8mm: bool,
    pub(crate) avx2: bool,
    pub(crate) f16c: bool,
    pub(crate) avx512f: bool,
}

impl CpuFeatures {
    /// Detect features supported by the current CPU.
    pub(crate) fn probe() -> Self {
        let mut f = Self::default();
        #[cfg(target_arch = "aarch64")]
        {
            f.neon_baseline = true;
            f.neon_fp16 = std::arch::is_aarch64_feature_detected!("fp16");
            f.neon_dotprod = std::arch::is_aarch64_feature_detected!("dotprod");
            f.neon_i8mm = std::arch::is_aarch64_feature_detected!("i8mm");
        }
        #[cfg(target_arch = "x86_64")]
        {
            f.avx2 = std::arch::is_x86_feature_detected!("avx2");
            f.f16c = std::arch::is_x86_feature_detected!("f16c");
            f.avx512f = std::arch::is_x86_feature_detected!("avx512f");
        }
        f
    }

    /// Honour `EDGEFIRST_DECODER_FORCE_KERNEL=<tier>` if set.
    /// Recognised tiers (case-insensitive): `scalar`, `neon` (alias
    /// `neon_baseline`), `neon_fp16`, `neon_dotprod`. Anything else
    /// returns `ForcedKernelUnavailable`.
    pub(crate) fn from_env_or_probe() -> Result<Self, DecoderError> {
        let probed = Self::probe();
        let Ok(forced) = std::env::var("EDGEFIRST_DECODER_FORCE_KERNEL") else {
            return Ok(probed);
        };
        match forced.to_ascii_lowercase().as_str() {
            "scalar" => Ok(Self::default()),
            "neon" | "neon_baseline" => {
                if !probed.neon_baseline {
                    return Err(DecoderError::ForcedKernelUnavailable {
                        tier: "neon",
                        missing_feature: "neon",
                    });
                }
                Ok(Self {
                    neon_baseline: true,
                    ..Self::default()
                })
            }
            "neon_fp16" => {
                if !probed.neon_fp16 {
                    return Err(DecoderError::ForcedKernelUnavailable {
                        tier: "neon_fp16",
                        missing_feature: "fp16",
                    });
                }
                Ok(Self {
                    neon_baseline: true,
                    neon_fp16: true,
                    ..Self::default()
                })
            }
            "neon_dotprod" => {
                if !probed.neon_dotprod {
                    return Err(DecoderError::ForcedKernelUnavailable {
                        tier: "neon_dotprod",
                        missing_feature: "dotprod",
                    });
                }
                Ok(Self {
                    neon_baseline: true,
                    neon_dotprod: true,
                    ..Self::default()
                })
            }
            _ => Err(DecoderError::ForcedKernelUnavailable {
                tier: "unknown",
                missing_feature: "unknown tier name",
            }),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// RAII guard that overrides an env var for the lifetime of the
    /// guard, capturing the prior value on construction and restoring it
    /// on drop. Restores even on panic, and treats "unset" as a distinct
    /// state from "set to empty string" so we don't accidentally leave a
    /// stray empty value behind.
    ///
    /// The repo runs tests with `--test-threads=1`, so the per-test
    /// mutation is serialized with respect to other tests in this
    /// process. The guard ensures we also don't leak state to test
    /// invocations that follow this one (or to a developer's shell when
    /// the env var was set externally before `cargo test`).
    struct EnvGuard {
        key: &'static str,
        prev: Option<String>,
    }

    impl EnvGuard {
        fn set(key: &'static str, value: &str) -> Self {
            let prev = std::env::var(key).ok();
            std::env::set_var(key, value);
            Self { key, prev }
        }
    }

    impl Drop for EnvGuard {
        fn drop(&mut self) {
            match self.prev.take() {
                Some(v) => std::env::set_var(self.key, v),
                None => std::env::remove_var(self.key),
            }
        }
    }

    #[test]
    fn probe_does_not_panic() {
        let _ = CpuFeatures::probe();
    }

    #[test]
    fn probe_on_aarch64_has_neon_baseline() {
        let f = CpuFeatures::probe();
        #[cfg(target_arch = "aarch64")]
        assert!(f.neon_baseline, "aarch64 builds always have NEON baseline");
        #[cfg(not(target_arch = "aarch64"))]
        assert!(!f.neon_baseline);
    }

    #[test]
    fn from_env_with_scalar_clears_all_simd() {
        let _g = EnvGuard::set("EDGEFIRST_DECODER_FORCE_KERNEL", "scalar");
        let f = CpuFeatures::from_env_or_probe().unwrap();
        assert!(!f.neon_baseline);
        assert!(!f.neon_fp16);
        assert!(!f.neon_dotprod);
        assert!(!f.avx2);
    }

    #[test]
    fn from_env_with_unknown_tier_errors() {
        let _g = EnvGuard::set("EDGEFIRST_DECODER_FORCE_KERNEL", "wibble");
        let r = CpuFeatures::from_env_or_probe();
        assert!(r.is_err());
    }
}
