// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! HAL V4L2 mem2mem JPEG (e.g. mxc-jpeg) → NV12 + HAL OpenGL letterbox.
//!
//! Forces `EDGEFIRST_FORCE_BACKEND=opengl` and leaves V4L2 enabled.

use clap::Parser;
use edgefirst_bench_common::{run_hal_module, BenchArgs, HalModuleConfig};

fn main() -> anyhow::Result<()> {
    let args = BenchArgs::parse();
    run_hal_module(
        HalModuleConfig {
            class: "hw_gl",
            module: "hal_v4l2_gl",
            force_backend: "opengl",
            disable_v4l2: false,
            prefer_heap_src: false,
        },
        &args,
    )?;
    Ok(())
}
