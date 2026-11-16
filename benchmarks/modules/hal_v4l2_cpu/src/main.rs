// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! HAL V4L2 mem2mem JPEG (e.g. mxc-jpeg) + HAL CPU convert: the
//! CPU-second-pass variant of `hal_v4l2_gl`.
//!
//! Run with `--full-res-convert` to measure hardware decode +
//! full-resolution NV*→RGB on the CPU; `hal_v4l2_gl --full-res-convert` is
//! the GPU-convert counterpart, and `--decode-only` stops at the native
//! format. The per-frame decode/convert split is reported either way.

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
            class: "hw_cpu",
            module: "hal_v4l2_cpu",
            force_backend: "cpu",
            disable_v4l2: false,
            prefer_heap_src: false,
        },
        &args,
    )?;
    Ok(())
}
