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
        Self::from_forced_tier(
            std::env::var("EDGEFIRST_DECODER_FORCE_KERNEL")
                .ok()
                .as_deref(),
        )
    }

    /// The tier rules, with the environment factored out.
    ///
    /// An environment is per process, so a test that reached these rules by
    /// setting the variable changed them for every other thread too --
    /// including [`crate::per_scale::plan`], which reads the same variable
    /// while building a plan. Taking the tier as an argument means the rules
    /// can be tested without a process-wide side effect to serialize against.
    pub(crate) fn from_forced_tier(forced: Option<&str>) -> Result<Self, DecoderError> {
        let probed = Self::probe();
        let Some(forced) = forced else {
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
    use std::sync::{Mutex, MutexGuard};

    /// Serializes every test that overrides an env var.
    ///
    /// An environment is per process, not per thread, and plain `cargo test`
    /// runs tests as threads in one process. Two tests setting
    /// `EDGEFIRST_DECODER_FORCE_KERNEL` could interleave between one's `set`
    /// and its read, so `from_env_with_scalar_clears_all_simd` would see the
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
                DecoderError::ForcedKernelUnavailable {
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
                DecoderError::ForcedKernelUnavailable {
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
        let _g = EnvGuard::set("EDGEFIRST_DECODER_FORCE_KERNEL", "scalar");
        assert_eq!(
            CpuFeatures::from_env_or_probe().unwrap(),
            CpuFeatures::default()
        );
    }
}
