// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

use crate::error::{KernelError, Result};

/// Detected CPU instruction-set features.
///
/// [`CpuFeatures::from_env_or_probe`] applies the `EDGEFIRST_KERNELS_FORCE_TIER` override.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct CpuFeatures {
    pub neon_baseline: bool,
    pub neon_fp16: bool,
    pub neon_dotprod: bool,
    pub neon_i8mm: bool,
    pub avx2: bool,
    pub f16c: bool,
    pub avx512f: bool,
}

impl CpuFeatures {
    /// Detect features supported by the current CPU.
    pub fn probe() -> Self {
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

    /// Honour `EDGEFIRST_KERNELS_FORCE_TIER=<tier>` if set.
    /// Recognised tiers (case-insensitive): `scalar`, `neon` (alias
    /// `neon_baseline`), `neon_fp16`, `neon_dotprod`. Anything else
    /// returns `ForcedTierUnavailable`.
    pub fn from_env_or_probe() -> Result<Self> {
        Self::from_lookup(|name| std::env::var(name).ok())
    }

    /// [`CpuFeatures::from_env_or_probe`] with `EDGEFIRST_KERNELS_FORCE_TIER`
    /// read through `lookup`.
    fn from_lookup(lookup: impl Fn(&str) -> Option<String>) -> Result<Self> {
        Self::from_forced_tier(lookup("EDGEFIRST_KERNELS_FORCE_TIER").as_deref())
    }

    /// The features of the forced tier `forced`, or the probed features for `None`.
    pub fn from_forced_tier(forced: Option<&str>) -> Result<Self> {
        let probed = Self::probe();
        let Some(forced) = forced else {
            return Ok(probed);
        };
        match forced.to_ascii_lowercase().as_str() {
            "scalar" => Ok(Self::default()),
            "neon" | "neon_baseline" => {
                if !probed.neon_baseline {
                    return Err(KernelError::ForcedTierUnavailable {
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
                    return Err(KernelError::ForcedTierUnavailable {
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
                    return Err(KernelError::ForcedTierUnavailable {
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
            _ => Err(KernelError::ForcedTierUnavailable {
                tier: "unknown",
                missing_feature: "unknown tier name",
            }),
        }
    }

    /// Whether FP16 arithmetic may run: NEON and FEAT_FP16 requested, and FEAT_FP16 detected
    /// at run time.
    pub fn fp16_usable(&self) -> bool {
        #[cfg(target_arch = "aarch64")]
        {
            self.neon_baseline && self.neon_fp16 && std::arch::is_aarch64_feature_detected!("fp16")
        }
        #[cfg(not(target_arch = "aarch64"))]
        false
    }

    /// Whether `sdot`/`udot` may run: NEON and FEAT_DotProd requested, and FEAT_DotProd
    /// detected at run time.
    pub fn dotprod_usable(&self) -> bool {
        #[cfg(target_arch = "aarch64")]
        {
            self.neon_baseline
                && self.neon_dotprod
                && std::arch::is_aarch64_feature_detected!("dotprod")
        }
        #[cfg(not(target_arch = "aarch64"))]
        false
    }

    /// The best tier these features can run.
    pub fn tier(&self) -> Tier {
        if self.neon_baseline && self.neon_dotprod {
            Tier::NeonDotprod
        } else if self.neon_baseline && self.neon_fp16 {
            Tier::NeonFp16
        } else if self.neon_baseline {
            Tier::Neon
        } else {
            Tier::Scalar
        }
    }
}

/// Kernel implementation tier, in increasing order of required CPU features.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Tier {
    Scalar,
    Neon,
    NeonFp16,
    NeonDotprod,
}

impl Tier {
    /// The tier name accepted by `EDGEFIRST_KERNELS_FORCE_TIER`.
    pub fn as_str(self) -> &'static str {
        match self {
            Tier::Scalar => "scalar",
            Tier::Neon => "neon",
            Tier::NeonFp16 => "neon_fp16",
            Tier::NeonDotprod => "neon_dotprod",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The usable predicates need the requested feature, NEON and the CPU's own support.
    #[test]
    fn usable_predicates_need_runtime_support() {
        let all = CpuFeatures {
            neon_baseline: true,
            neon_fp16: true,
            neon_dotprod: true,
            ..CpuFeatures::default()
        };
        #[cfg(target_arch = "aarch64")]
        let (fp16, dotprod) = (
            std::arch::is_aarch64_feature_detected!("fp16"),
            std::arch::is_aarch64_feature_detected!("dotprod"),
        );
        #[cfg(not(target_arch = "aarch64"))]
        let (fp16, dotprod) = (false, false);
        assert_eq!(all.fp16_usable(), fp16);
        assert_eq!(all.dotprod_usable(), dotprod);
        let no_neon = CpuFeatures {
            neon_baseline: false,
            ..all
        };
        assert!(!no_neon.fp16_usable() && !no_neon.dotprod_usable());
        assert!(!CpuFeatures::default().fp16_usable());
        assert!(!CpuFeatures::default().dotprod_usable());
    }

    #[test]
    fn tier_ordering_and_names() {
        assert_eq!(CpuFeatures::default().tier(), Tier::Scalar);
        let f = CpuFeatures {
            neon_baseline: true,
            neon_fp16: true,
            neon_dotprod: true,
            ..CpuFeatures::default()
        };
        assert_eq!(f.tier(), Tier::NeonDotprod);
        assert_eq!(Tier::NeonFp16.as_str(), "neon_fp16");
        assert_eq!(Tier::Scalar.as_str(), "scalar");
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
    fn forced_tier_scalar_clears_all_simd() {
        let f = CpuFeatures::from_forced_tier(Some("scalar")).unwrap();
        assert!(!f.neon_baseline);
        assert!(!f.neon_fp16);
        assert!(!f.neon_dotprod);
        assert!(!f.avx2);
    }

    #[test]
    fn forced_tier_is_case_insensitive() {
        let f = CpuFeatures::from_forced_tier(Some("SCALAR")).unwrap();
        assert_eq!(f, CpuFeatures::default());
    }

    #[test]
    fn forced_tier_unknown_errors() {
        let err = CpuFeatures::from_forced_tier(Some("wibble")).unwrap_err();
        assert!(
            matches!(
                err,
                KernelError::ForcedTierUnavailable {
                    tier: "unknown",
                    ..
                }
            ),
            "{err:?}"
        );
    }

    #[test]
    fn no_forced_tier_reports_what_the_cpu_has() {
        assert_eq!(
            CpuFeatures::from_forced_tier(None).unwrap(),
            CpuFeatures::probe()
        );
    }

    /// A tier the CPU cannot provide is refused.
    #[test]
    fn forced_neon_tier_matches_what_the_cpu_can_do() {
        let r = CpuFeatures::from_forced_tier(Some("neon"));
        if CpuFeatures::probe().neon_baseline {
            assert!(r.unwrap().neon_baseline);
        } else {
            assert!(matches!(
                r.unwrap_err(),
                KernelError::ForcedTierUnavailable {
                    tier: "neon",
                    missing_feature: "neon"
                }
            ));
        }
    }

    /// `from_env_or_probe` reads `EDGEFIRST_KERNELS_FORCE_TIER`; other names are ignored.
    #[test]
    fn from_env_or_probe_reads_the_documented_variable() {
        let forced = |name: &str| (name == "EDGEFIRST_KERNELS_FORCE_TIER").then(|| "scalar".into());
        assert_eq!(
            CpuFeatures::from_lookup(forced).unwrap(),
            CpuFeatures::default()
        );
        let other =
            |name: &str| (name == "EDGEFIRST_DECODER_FORCE_KERNEL").then(|| "scalar".into());
        assert_eq!(
            CpuFeatures::from_lookup(other).unwrap(),
            CpuFeatures::probe()
        );
    }
}
