// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

use crate::error::{KernelError, Result};

/// Detected CPU instruction-set features.
///
/// Probe once when building a plan and cache the result.
/// [`CpuFeatures::from_env_or_probe`] reads `EDGEFIRST_KERNELS_FORCE_TIER` for
/// debugging and benchmarking overrides.
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
        Self::from_forced_tier(
            std::env::var("EDGEFIRST_KERNELS_FORCE_TIER")
                .ok()
                .as_deref(),
        )
    }

    /// The tier rules, with the environment factored out.
    ///
    /// An environment is per process, so taking the tier as an argument lets
    /// the rules be tested without a process-wide side effect.
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
    /// at run time, so features built by hand cannot reach an FP16 kernel on a CPU without it.
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
    use std::sync::{Mutex, MutexGuard};

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

    /// Serializes every test that overrides an env var.
    ///
    /// An environment is per process, not per thread, and plain `cargo test`
    /// runs tests as threads in one process. Two tests setting
    /// `EDGEFIRST_KERNELS_FORCE_TIER` could interleave between one's `set`
    /// and its read, so `from_env_or_probe_reads_the_documented_variable` would see the
    /// other's `wibble` and unwrap an `Err`. It is intermittent and invisible
    /// under nextest, which gives each test its own process, and under the
    /// mutation lane's `--test-threads=1`.
    ///
    /// Poisoning is ignored: the lock guards an env var that the guard below
    /// restores on unwind anyway, so a panicking test should fail on its own
    /// assertion rather than take every later test with it.
    static ENV_LOCK: Mutex<()> = Mutex::new(());

    /// RAII guard that overrides an env var for the lifetime of the
    /// guard, capturing the prior value on construction and restoring it
    /// on drop. Restores even on panic, and treats "unset" as a distinct
    /// state from "set to empty string" so we don't accidentally leave a
    /// stray empty value behind.
    ///
    /// Holding [`ENV_LOCK`] for the guard's lifetime is what makes the
    /// override safe against other tests; restoring on drop is what keeps it
    /// from leaking to the tests that follow, or to a developer's shell when
    /// the variable was already set before `cargo test`.
    struct EnvGuard {
        key: &'static str,
        prev: Option<String>,
        _lock: MutexGuard<'static, ()>,
    }

    impl EnvGuard {
        fn set(key: &'static str, value: &str) -> Self {
            let lock = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
            let prev = std::env::var(key).ok();
            std::env::set_var(key, value);
            Self {
                key,
                prev,
                _lock: lock,
            }
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

    /// A tier the CPU cannot provide is refused rather than silently downgraded.
    /// On x86_64 no NEON tier is available; on aarch64 the baseline always is.
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

    /// The only test that touches the process environment, and the one place
    /// the variable's name is load-bearing. `scalar` is chosen deliberately:
    /// it is a valid tier, so a plan built concurrently on another thread sees
    /// a different-but-valid dispatch rather than the `Err` an unknown tier
    /// would hand it.
    #[test]
    fn from_env_or_probe_reads_the_documented_variable() {
        let _g = EnvGuard::set("EDGEFIRST_KERNELS_FORCE_TIER", "scalar");
        assert_eq!(
            CpuFeatures::from_env_or_probe().unwrap(),
            CpuFeatures::default()
        );
    }
}
