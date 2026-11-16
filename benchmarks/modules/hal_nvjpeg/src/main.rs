// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! HAL nvJPEG GPU JPEG decode (Jetson Orin): the CUDA arm of the
//! hardware-decode table (BENCHMARKS.md § JPEG Decode).
//!
//! Sets `EDGEFIRST_ENABLE_NVJPEG`. nvJPEG decodes to interleaved RGB (RGBI).
//! Run with `--decode-fmt native` (the default): a decoded format of `Rgb`
//! means nvJPEG ran, `Nv12/16/24` means the CPU decoder handled that frame.
//!
//! Startup asserts `nvjpeg_available()`.

use clap::Parser;
use edgefirst_bench_common::{run_hal_module, BenchArgs, HalModuleConfig};

fn main() -> anyhow::Result<()> {
    std::env::set_var("EDGEFIRST_ENABLE_NVJPEG", "1");
    let args = BenchArgs::parse();
    anyhow::ensure!(
        edgefirst_codec::nvjpeg_available(),
        "nvJPEG unavailable (need Linux + CUDA device + loadable libnvjpeg.so.12); \
         refusing to run the GPU arm on the CPU decoder"
    );
    run_hal_module(
        HalModuleConfig {
            class: "hw_gpu",
            module: "hal_nvjpeg",
            force_backend: "opengl",
            disable_v4l2: true,
            prefer_heap_src: false,
        },
        &args,
    )?;
    Ok(())
}
