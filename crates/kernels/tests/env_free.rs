// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! The `*_with_features` constructors read no environment.
//!
//! These tests set `EDGEFIRST_KERNELS_FORCE_TIER` to an unknown tier, which makes every
//! env-reading constructor fail, so they live in their own test binary (process).

use std::sync::{Mutex, MutexGuard};

use edgefirst_kernels::{
    CpuFeatures, DType, KernelError, LutI8Plan, Quant, SigmoidParams, SigmoidPlan, Tier,
};

const VAR: &str = "EDGEFIRST_KERNELS_FORCE_TIER";

static ENV_LOCK: Mutex<()> = Mutex::new(());

/// Sets `VAR` for the guard's lifetime and restores the previous value on drop.
struct EnvGuard {
    prev: Option<String>,
    _lock: MutexGuard<'static, ()>,
}

impl EnvGuard {
    fn set(value: &str) -> Self {
        let lock = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let prev = std::env::var(VAR).ok();
        std::env::set_var(VAR, value);
        Self { prev, _lock: lock }
    }
}

impl Drop for EnvGuard {
    fn drop(&mut self) {
        match self.prev.take() {
            Some(v) => std::env::set_var(VAR, v),
            None => std::env::remove_var(VAR),
        }
    }
}

fn i8_params() -> SigmoidParams {
    SigmoidParams {
        input_dtype: DType::I8,
        input: Quant::new(0.1, 0),
        output_dtype: DType::I8,
        output: Quant::new(1.0 / 256.0, -128),
    }
}

#[test]
fn with_features_constructors_ignore_the_environment() {
    let _g = EnvGuard::set("wibble");
    let f = CpuFeatures::default();
    // The env-reading constructors see the unknown tier.
    assert!(matches!(
        SigmoidPlan::new(i8_params()),
        Err(KernelError::ForcedTierUnavailable { .. })
    ));
    assert!(matches!(
        LutI8Plan::from_table(DType::I8, [0; 256]),
        Err(KernelError::ForcedTierUnavailable { .. })
    ));
    assert!(matches!(
        LutI8Plan::sigmoid(DType::U8, Quant::new(0.1, 0), Quant::new(1.0 / 256.0, 0)),
        Err(KernelError::ForcedTierUnavailable { .. })
    ));
    // The env-free ones do not.
    let plan = SigmoidPlan::with_features(i8_params(), f).unwrap();
    assert_eq!(plan.tier(), Tier::Scalar);
    let lut = LutI8Plan::from_table_with_features(DType::I8, [0; 256], f).unwrap();
    assert_eq!(lut.tier(), Tier::Scalar);
    let lut = LutI8Plan::sigmoid_with_features(
        DType::U8,
        Quant::new(0.1, 0),
        Quant::new(1.0 / 256.0, 0),
        f,
    )
    .unwrap();
    assert_eq!(lut.tier(), Tier::Scalar);
}
