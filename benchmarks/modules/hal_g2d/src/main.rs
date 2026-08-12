// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! HAL software JPEG → NV12/NV24 + HAL G2D letterbox convert (i.MX).
//!
//! Forces `EDGEFIRST_FORCE_BACKEND=g2d` and `EDGEFIRST_DISABLE_V4L2=1`.

use clap::Parser;
use edgefirst_bench_common::{run_hal_module, BenchArgs, HalModuleConfig};

fn main() -> anyhow::Result<()> {
    let args = BenchArgs::parse();
    run_hal_module(
        HalModuleConfig {
            class: "hybrid_2d",
            module: "hal_g2d",
            force_backend: "g2d",
            disable_v4l2: true,
            prefer_heap_src: false,
        },
        &args,
    )?;
    Ok(())
}
