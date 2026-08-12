// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! HAL software JPEG → NV12/NV24 + HAL OpenGL letterbox convert.
//!
//! Forces `EDGEFIRST_FORCE_BACKEND=opengl` and `EDGEFIRST_DISABLE_V4L2=1`.
//! Decode source stays on DMA (auto) for zero-copy GL import.

use clap::Parser;
use edgefirst_bench_common::{run_hal_module, BenchArgs, HalModuleConfig};

fn main() -> anyhow::Result<()> {
    let args = BenchArgs::parse();
    run_hal_module(
        HalModuleConfig {
            class: "hybrid_gl",
            module: "hal_gl",
            force_backend: "opengl",
            disable_v4l2: true,
            prefer_heap_src: false,
        },
        &args,
    )?;
    Ok(())
}
