// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! HAL V4L2 mem2mem JPEG (e.g. mxc-jpeg) → NV12 + HAL OpenGL letterbox.
//!
//! Forces `EDGEFIRST_FORCE_BACKEND=opengl` and leaves V4L2 enabled.
//!
//! - `--decode-only`: V4L2 JPEG → native NV*/GREY, no convert.
//! - `--full-res-convert`: adds the full-resolution NV*→RGB GPU convert.
//!   `hal_v4l2_cpu` is the CPU-convert counterpart. The decode/convert
//!   split is reported per run.
//!
//! Startup asserts a V4L2 JPEG device is present.

use clap::Parser;
use edgefirst_bench_common::{run_hal_module, BenchArgs, DecodeFmt, HalModuleConfig};

fn main() -> anyhow::Result<()> {
    let args = BenchArgs::parse();
    anyhow::ensure!(
        edgefirst_codec::v4l2_available(),
        "no V4L2 JPEG decoder device found; refusing to run the hardware arm \
         on the CPU decoder"
    );
    anyhow::ensure!(
        args.decode_fmt == DecodeFmt::Native,
        "hardware arms require --decode-fmt native (fused outputs route to \
         the CPU decoder)"
    );
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
