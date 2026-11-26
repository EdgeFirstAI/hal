// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! Per-scale decoder kernels — level orchestrators and dispatch over `edgefirst_kernels::prim`.

pub(crate) use edgefirst_kernels::CpuFeatures;

/// Probes CPU features, honouring `EDGEFIRST_DECODER_FORCE_KERNEL`.
pub(crate) fn decoder_cpu_features() -> Result<CpuFeatures, crate::DecoderError> {
    #[cfg(test)]
    if let Some(tier) = forced_for_test::get() {
        return decoder_cpu_features_from(Some(tier));
    }
    decoder_cpu_features_lookup(|name| std::env::var(name).ok())
}

/// [`decoder_cpu_features`] with `EDGEFIRST_DECODER_FORCE_KERNEL` read
/// through `lookup`.
fn decoder_cpu_features_lookup(
    lookup: impl Fn(&str) -> Option<String>,
) -> Result<CpuFeatures, crate::DecoderError> {
    decoder_cpu_features_from(lookup("EDGEFIRST_DECODER_FORCE_KERNEL").as_deref())
}

/// A forced tier for the calling thread only.
#[cfg(test)]
pub(crate) mod forced_for_test {
    use std::cell::Cell;

    thread_local! {
        static TIER: Cell<Option<&'static str>> = const { Cell::new(None) };
    }

    pub(crate) fn get() -> Option<&'static str> {
        TIER.with(Cell::get)
    }

    /// Runs `f` with `tier` forced on this thread, restoring the previous value even on panic.
    pub(crate) fn with<R>(tier: &'static str, f: impl FnOnce() -> R) -> R {
        struct Restore(Option<&'static str>);
        impl Drop for Restore {
            fn drop(&mut self) {
                TIER.with(|t| t.set(self.0));
            }
        }
        let _restore = Restore(TIER.with(|t| t.replace(Some(tier))));
        f()
    }
}

/// The forced-tier rules with the environment factored out.
pub(crate) fn decoder_cpu_features_from(
    forced: Option<&str>,
) -> Result<CpuFeatures, crate::DecoderError> {
    CpuFeatures::from_forced_tier(forced).map_err(|e| match e {
        edgefirst_kernels::KernelError::ForcedTierUnavailable {
            tier,
            missing_feature,
        } => crate::DecoderError::ForcedKernelUnavailable {
            tier,
            missing_feature,
        },
        _ => crate::DecoderError::ForcedKernelUnavailable {
            tier: "unknown",
            missing_feature: "unexpected kernels error",
        },
    })
}

pub(crate) mod dispatch;

pub(crate) mod level_box;
pub(crate) mod level_mc;
pub(crate) mod level_score;

#[cfg(test)]
mod dfl_lut_tests;

#[cfg(test)]
mod tier_tests;

#[cfg(test)]
mod box_tier_tests;

#[cfg(test)]
mod cpu_tests {
    use super::*;
    use crate::DecoderError;

    #[test]
    fn unknown_forced_tier_maps_to_forced_kernel_unavailable() {
        assert!(matches!(
            decoder_cpu_features_from(Some("wibble")).unwrap_err(),
            DecoderError::ForcedKernelUnavailable {
                tier: "unknown",
                missing_feature: "unknown tier name"
            }
        ));
    }

    #[test]
    fn scalar_and_unset_tiers() {
        assert_eq!(
            decoder_cpu_features_from(Some("scalar")).unwrap(),
            CpuFeatures::default()
        );
        assert_eq!(
            decoder_cpu_features_from(None).unwrap(),
            CpuFeatures::probe()
        );
    }

    /// `EDGEFIRST_DECODER_FORCE_KERNEL=scalar` gives the scalar tier; other names are ignored.
    #[test]
    fn decoder_cpu_features_reads_the_documented_variable() {
        let forced =
            |name: &str| (name == "EDGEFIRST_DECODER_FORCE_KERNEL").then(|| "scalar".into());
        assert_eq!(
            decoder_cpu_features_lookup(forced).unwrap(),
            CpuFeatures::default()
        );
        let other = |name: &str| (name == "EDGEFIRST_KERNELS_FORCE_TIER").then(|| "scalar".into());
        assert_eq!(
            decoder_cpu_features_lookup(other).unwrap(),
            CpuFeatures::probe()
        );
    }
}
