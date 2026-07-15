// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! Multi-processor GL throughput scaling benchmark.
//!
//! Measures aggregate convert throughput with {1, 2, 4} `ImageProcessor`
//! instances running concurrently, each on its own thread with its own GL
//! context. On `LifecycleOnly` platforms (Mali, V3D, Tegra, llvmpipe) the
//! processors execute GL work in parallel and the aggregate rate should
//! scale above 1×; on Vivante (`Full` serialization policy) the rate stays
//! ≈1× by design. The scaling factor S(n) is the demonstration evidence
//! for the per-driver GL serialization policy — evidence, not a hard gate.
//!
//! Cells:
//! - `gpu_bound`: NV12 1280×720 → RGB 640×640 letterbox (DMA where
//!   available) — a realistic model-input convert dominated by GPU work.
//! - `overhead`: RGBA 64×64 → RGB 64×64 (heap) — dominated by per-message
//!   dispatch, exposing channel-hop / lock overhead.
//!
//! Each cell runs a fixed wall-clock window; every thread counts completed
//! converts. The recorded `BenchResult` carries ONE synthetic sample: the
//! effective aggregate per-convert latency `window / total_converts`
//! (lower = better; `S(n) = t(1) / t(n)`), which keeps the `--json` output
//! compatible with `scripts/bench_compare.py`.
//!
//! ## Run (on-target)
//! ```bash
//! ./parallel_processors_benchmark --json out.json
//! ```

mod common;

use edgefirst_bench::{BenchResult, BenchSuite};
use edgefirst_image::{Crop, Flip, ImageProcessor, ImageProcessorTrait, Rotation};
use edgefirst_tensor::{DType, PixelFormat, TensorMemory, TensorTrait};
use std::sync::{Arc, Barrier};
use std::time::{Duration, Instant};

const WINDOW: Duration = Duration::from_secs(3);
const WARMUP_CONVERTS: usize = 5;

#[derive(Clone, Copy)]
struct Cell {
    tag: &'static str,
    src_fmt: PixelFormat,
    src_w: usize,
    src_h: usize,
    dst_fmt: PixelFormat,
    dst_w: usize,
    dst_h: usize,
    letterbox: bool,
    use_dma: bool,
}

const CELLS: [Cell; 6] = [
    Cell {
        tag: "gpu_bound",
        src_fmt: PixelFormat::Nv12,
        src_w: 1280,
        src_h: 720,
        dst_fmt: PixelFormat::Rgb,
        dst_w: 640,
        dst_h: 640,
        letterbox: true,
        use_dma: true,
    },
    Cell {
        tag: "overhead",
        src_fmt: PixelFormat::Rgba,
        src_w: 64,
        src_h: 64,
        dst_fmt: PixelFormat::Rgb,
        dst_w: 64,
        dst_h: 64,
        letterbox: false,
        use_dma: false,
    },
    // macOS-meaningful zero-copy cells (IOSurface src+dst): RGBA dst —
    // PlanarRgb-u8 dsts are Linux-only — and the profiler's F16 hot
    // path. Skip cleanly where unsupported (the per-cell error prints).
    Cell {
        tag: "gpu_bound_rgba",
        src_fmt: PixelFormat::Nv12,
        src_w: 1280,
        src_h: 720,
        dst_fmt: PixelFormat::Rgba,
        dst_w: 640,
        dst_h: 640,
        letterbox: true,
        use_dma: true,
    },
    Cell {
        tag: "f16_zero_copy",
        src_fmt: PixelFormat::Nv12,
        src_w: 1280,
        src_h: 720,
        dst_fmt: PixelFormat::PlanarRgb,
        dst_w: 640,
        dst_h: 640,
        letterbox: true,
        use_dma: true,
    },
    // The RGBA→PlanarF16 DIRECT path (not the fused NV two-pass): the only
    // cells that exercise per-frame float SOURCE feeding — zero-copy source
    // import vs CPU map+upload (`feed_float_src`). Two sizes because the
    // upload cost scales with source bytes (~2.5 GB/s): 720p ≈ 3.7 MB,
    // 1080p ≈ 8.3 MB per frame on the upload path.
    Cell {
        tag: "f16_rgba_src",
        src_fmt: PixelFormat::Rgba,
        src_w: 1280,
        src_h: 720,
        dst_fmt: PixelFormat::PlanarRgb,
        dst_w: 640,
        dst_h: 640,
        letterbox: true,
        use_dma: true,
    },
    Cell {
        tag: "f16_rgba_src_1080p",
        src_fmt: PixelFormat::Rgba,
        src_w: 1920,
        src_h: 1080,
        dst_fmt: PixelFormat::PlanarRgb,
        dst_w: 640,
        dst_h: 640,
        letterbox: true,
        use_dma: true,
    },
];

/// Run one (cell, n_procs) configuration; returns per-thread convert counts.
fn run_config(cell: Cell, n_procs: usize) -> Result<Vec<usize>, String> {
    // Zero-copy probe covers IOSurface on macOS too — is_dma_available()
    // is false there and would silently measure CPU-heap converts.
    let mem = if cell.use_dma && edgefirst_tensor::is_gpu_buffer_available() {
        Some(TensorMemory::Dma)
    } else {
        Some(TensorMemory::Mem)
    };
    // The DESTINATION auto-selects: explicit Some(Dma) now errors loudly
    // for (format, dtype) combos with no zero-copy mapping (the
    // explicit-Dma contract) — e.g. the gpu_bound cell's packed-RGB dst on
    // macOS, which historically byte-bagged and silently measured the CPU
    // fallback. Auto-select keeps every platform on its honest best path
    // (Linux: Dma; macOS packed-RGB: Mem + CPU convert).
    let dst_mem = if cell.use_dma && edgefirst_tensor::is_gpu_buffer_available() {
        None
    } else {
        Some(TensorMemory::Mem)
    };
    let barrier = Arc::new(Barrier::new(n_procs));

    let handles: Vec<_> = (0..n_procs)
        .map(|i| {
            let barrier = Arc::clone(&barrier);
            std::thread::spawn(move || -> Result<usize, String> {
                let mut proc = ImageProcessor::new()
                    .map_err(|e| format!("processor {i} creation failed: {e}"))?;
                let src = proc
                    .create_image(
                        cell.src_w,
                        cell.src_h,
                        cell.src_fmt,
                        DType::U8,
                        mem,
                        edgefirst_tensor::CpuAccess::ReadWrite,
                    )
                    .map_err(|e| format!("src alloc failed: {e}"))?;
                {
                    use edgefirst_tensor::TensorMapTrait;
                    let t = src.as_u8().unwrap();
                    let mut m = t.map().map_err(|e| format!("map failed: {e}"))?;
                    for (j, b) in m.as_mut_slice().iter_mut().enumerate() {
                        *b = ((i * 41 + j) % 223) as u8;
                    }
                }
                let crop = if cell.letterbox {
                    Crop::letterbox([114, 114, 114, 255])
                } else {
                    Crop::default()
                };
                // Pre-allocate and REUSE the destination: real pipelines pool
                // buffers, and per-iteration allocation would measure the
                // allocator (PBO/DMA churn) instead of convert dispatch.
                let dst_dtype = if cell.tag.starts_with("f16_") {
                    DType::F16
                } else {
                    DType::U8
                };
                let mut dst = proc
                    .create_image(
                        cell.dst_w,
                        cell.dst_h,
                        cell.dst_fmt,
                        dst_dtype,
                        dst_mem,
                        edgefirst_tensor::CpuAccess::ReadWrite,
                    )
                    .map_err(|e| format!("dst alloc failed: {e}"))?;
                let convert_once = |proc: &mut ImageProcessor, dst: &mut _| -> Result<(), String> {
                    proc.convert(&src, dst, Rotation::None, Flip::None, crop)
                        .map_err(|e| format!("convert failed: {e}"))?;
                    Ok(())
                };

                for _ in 0..WARMUP_CONVERTS {
                    convert_once(&mut proc, &mut dst)?;
                }
                barrier.wait();
                let deadline = Instant::now() + WINDOW;
                let mut count = 0usize;
                while Instant::now() < deadline {
                    convert_once(&mut proc, &mut dst)?;
                    count += 1;
                }
                Ok(count)
            })
        })
        .collect();

    handles
        .into_iter()
        .enumerate()
        .map(|(i, h)| {
            h.join()
                .map_err(|e| format!("thread {i} panicked: {e:?}"))?
        })
        .collect()
}

fn main() {
    let mut suite = BenchSuite::from_args();

    println!("== Parallel ImageProcessor throughput scaling ==");
    println!("   window = {WINDOW:?}/cell, warmup = {WARMUP_CONVERTS} converts/proc\n");

    for cell in CELLS {
        let mut base_rate: Option<f64> = None;
        for n in [1usize, 2, 4] {
            let name = format!("parallel/{}/n{}", cell.tag, n);
            let counts = match run_config(cell, n) {
                Ok(c) => c,
                Err(e) => {
                    println!("  {name:50} [unsupported: {e}]");
                    continue;
                }
            };
            let total: usize = counts.iter().sum();
            if total == 0 {
                println!("  {name:50} [no converts completed in window]");
                continue;
            }
            let rate = total as f64 / WINDOW.as_secs_f64();
            let scaling = base_rate.map(|b| rate / b);
            if n == 1 {
                base_rate = Some(rate);
            }
            let min = counts.iter().min().unwrap();
            let max = counts.iter().max().unwrap();
            let effective = Duration::from_secs_f64(WINDOW.as_secs_f64() / total as f64);
            println!(
                "  {name:50} agg {rate:8.1} conv/s  per-proc [{min}..{max}]  {}",
                scaling.map_or(String::from("S=1.00 (base)"), |s| format!("S={s:.2}")),
            );
            suite.record(&BenchResult {
                name,
                iterations: total,
                times: vec![effective],
            });
        }
        println!();
    }

    suite.finish();
}
