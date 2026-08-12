// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! HAL software JPEG decode → HAL CPU letterbox convert.
//!
//! Forces `EDGEFIRST_FORCE_BACKEND=cpu` and `EDGEFIRST_DISABLE_V4L2=1`.
//! Defaults decode source to heap Mem (matches historical `codec_benchmark`);
//! override with `--tensor-mem dma` to measure DMA write-path cost.

use clap::Parser;
use edgefirst_bench_common::{run_hal_module, BenchArgs, HalModuleConfig};

fn main() -> anyhow::Result<()> {
    let args = BenchArgs::parse();
    run_hal_module(
        HalModuleConfig {
            class: "cpu",
            module: "hal_cpu",
            force_backend: "cpu",
            disable_v4l2: true,
            prefer_heap_src: true,
        },
        &args,
    )?;
    Ok(())
}
