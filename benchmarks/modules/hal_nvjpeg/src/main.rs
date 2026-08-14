// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! HAL nvJPEG GPU JPEG decode (Jetson Orin) — the CUDA arm of the
//! hardware-decode table (BENCHMARKS.md § JPEG Decode).
//!
//! **This is GPU decode, not dedicated JPEG hardware.** On Orin-class SoCs
//! only nvJPEG's GPU_HYBRID backend exists: Huffman decode stays on the CPU
//! and IDCT/postprocess run on the **shared CUDA cores** (the NVJPG ASIC is
//! unreachable through CUDA nvJPEG). Anything else using the GPU — above
//! all AI inference — will be impacted by concurrent decode, which is why
//! the backend is opt-in (`EDGEFIRST_ENABLE_NVJPEG`, set here) and off by
//! default in production. The decode-vs-inference contention is measured in
//! separate benchmarks, out of scope for this arm. CUDA 12.3.3 has no NV12
//! output, so nvJPEG's fixed decode surface is interleaved RGB (RGBI).
//!
//! **Run this arm with `--decode-fmt native` (the default), never `rgb`.**
//! The codec routes hardware decoders only when no fused output is requested
//! (`allow_hw = output_fmt == native_fmt` in `codec/src/jpeg/mod.rs`); a
//! `rgb` preference selects the CPU fused-RGB path and silently bypasses
//! nvJPEG. Under `native`, nvJPEG reconfigures the destination to RGB by
//! itself — which doubles as a per-frame engagement check: a decoded format
//! of `Rgb` means nvJPEG ran, `Nv12/16/24` means the CPU decoder handled
//! that frame.
//!
//! Startup asserts `nvjpeg_available()` so a missing libnvjpeg/CUDA fails
//! loudly instead of silently publishing CPU-decode numbers under a GPU
//! label (on JetPack, libnvjpeg comes from the `libnvjpeg-12-6` package —
//! NOT the same-named stub in `/usr/lib/aarch64-linux-gnu/nvidia/`).
//! Per-frame engagement is verifiable on-target with `EDGEFIRST_TRACE=…`:
//! the capture must show `codec.decode_jpeg.nvjpeg_*` spans for every frame.

use clap::Parser;
use edgefirst_bench_common::{run_hal_module, BenchArgs, HalModuleConfig};

fn main() -> anyhow::Result<()> {
    // Must be set before the first decode probes the backend.
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
