// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! CPU JPEG-preprocessing path benchmark (Jetson Orin Nano profile target).
//!
//! On the Orin Nano Super we deliberately keep JPEG decode + preprocessing +
//! postprocessing on the CPU so TensorRT owns the GPU. JPEG decodes to the NV
//! family (NV12 4:2:0, NV16 4:2:2, NV24 4:4:4) and the model wants planar RGB
//! (`PlanarRgb`, i.e. CHW) — optionally widened to F32. This bench isolates the
//! cells on that path so `perf` attribution is clean and before/after fusion +
//! NEON work has a named gate per cell.
//!
//! The CPU backend is pinned via `EDGEFIRST_FORCE_BACKEND=cpu` so GL/G2D never
//! absorb a cell. Self-contained: sources are synthesized (Y luma gradient +
//! neutral chroma), so NO testdata files are required on the target.
//!
//! ```bash
//! EDGEFIRST_FORCE_BACKEND=cpu ./cpu_preprocess_benchmark --json cpu.json
//! # profile a single cell under perf (crank iterations for sample density):
//! EDGEFIRST_FORCE_BACKEND=cpu EDGEFIRST_BENCH_ONLY=nv12/letterbox \
//!   EDGEFIRST_BENCH_ITERS=2000 perf record -e cycles:u -g -- ./cpu_preprocess_benchmark
//! ```

mod common;

use common::{run_bench, BenchSuite};

use edgefirst_image::{Crop, Flip, ImageProcessor, ImageProcessorTrait, Rotation};
use edgefirst_tensor::{DType, PixelFormat, TensorMapTrait, TensorTrait};

const WARMUP: usize = 10;
const DEFAULT_ITERATIONS: usize = 200;

// Camera 1280x720 → model 640x640 (the profiler's letterbox geometry).
const IN_W: usize = 1280;
const IN_H: usize = 720;
const OUT_W: usize = 640;
const OUT_H: usize = 640;

/// Combined semi-planar buffer byte length for a `w x h` image.
fn nv_len(fmt: PixelFormat, w: usize, h: usize) -> usize {
    match fmt {
        PixelFormat::Nv12 => w * h * 3 / 2,
        PixelFormat::Nv16 => w * h * 2,
        PixelFormat::Nv24 => w * h * 3,
        _ => w * h,
    }
}

/// Synthesize an NV* source: Y plane = luma gradient, chroma = neutral 128.
fn fill_nv(buf: &mut [u8], w: usize, h: usize) {
    for r in 0..h {
        for c in 0..w {
            buf[r * w + c] = ((r + c) * 255 / (w + h)) as u8;
        }
    }
    for b in buf.iter_mut().skip(w * h) {
        *b = 128;
    }
}

fn fmt_name(f: PixelFormat) -> &'static str {
    match f {
        PixelFormat::Nv12 => "nv12",
        PixelFormat::Nv16 => "nv16",
        PixelFormat::Nv24 => "nv24",
        _ => "other",
    }
}

fn main() {
    // Pin the CPU backend first, before anything that could spawn a thread:
    // this bench characterises the CPU preprocessing path the Orin Nano
    // deployment uses to keep the GPU free for TensorRT.
    // SAFETY: set_var runs at the very top of main(), before env_logger,
    // argument parsing, or any worker threads — the process is single-threaded.
    if std::env::var("EDGEFIRST_FORCE_BACKEND").is_err() {
        unsafe { std::env::set_var("EDGEFIRST_FORCE_BACKEND", "cpu") };
    }

    env_logger::init();
    let mut suite = BenchSuite::from_args();

    let iters: usize = std::env::var("EDGEFIRST_BENCH_ITERS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(DEFAULT_ITERATIONS);
    let only = std::env::var("EDGEFIRST_BENCH_ONLY").unwrap_or_default();

    println!(
        "\n== CPU preprocess benchmark ({IN_W}x{IN_H} -> {OUT_W}x{OUT_H}) backend={} ==",
        std::env::var("EDGEFIRST_FORCE_BACKEND").unwrap_or_else(|_| "auto".into())
    );
    println!("   warmup={WARMUP} iterations={iters} filter={only:?}\n");

    let mut proc = match ImageProcessor::new() {
        Ok(p) => p,
        Err(e) => {
            println!("  [skipped: ImageProcessor init failed: {e}]");
            suite.finish();
            return;
        }
    };

    let lb_crop = Crop::letterbox([114, 114, 114, 255]);

    for fmt in [PixelFormat::Nv12, PixelFormat::Nv16, PixelFormat::Nv24] {
        let src = match proc.create_image(
            IN_W,
            IN_H,
            fmt,
            DType::U8,
            None,
            edgefirst_tensor::CpuAccess::ReadWrite,
        ) {
            Ok(s) => s,
            Err(e) => {
                println!("  {fmt:?}: [skipped: source alloc failed: {e}]");
                continue;
            }
        };
        {
            let mut map = src.as_u8().unwrap().map().unwrap();
            let n = nv_len(fmt, IN_W, IN_H);
            fill_nv(&mut map.as_mut_slice()[..n], IN_W, IN_H);
        }
        let src_bytes = nv_len(fmt, IN_W, IN_H) as u64;

        // Each cell: (suffix, out dims, out fmt, out dtype, crop).
        let cells: &[(&str, usize, usize, PixelFormat, DType, Crop)] = &[
            // Pure convert (no resize) — isolates NV→RGB decode + planar scatter.
            (
                "convert_720p_to_rgb",
                IN_W,
                IN_H,
                PixelFormat::Rgb,
                DType::U8,
                Crop::no_crop(),
            ),
            (
                "convert_720p_to_planar_rgb",
                IN_W,
                IN_H,
                PixelFormat::PlanarRgb,
                DType::U8,
                Crop::no_crop(),
            ),
            // Letterbox (convert + resize) — the real pipeline op.
            (
                "letterbox_720p_to_640_planar_rgb",
                OUT_W,
                OUT_H,
                PixelFormat::PlanarRgb,
                DType::U8,
                lb_crop,
            ),
            // Full preprocess: planar RGB widened to F32 (model input dtype).
            (
                "letterbox_720p_to_640_planar_rgb_f32",
                OUT_W,
                OUT_H,
                PixelFormat::PlanarRgb,
                DType::F32,
                lb_crop,
            ),
            // Full preprocess to F16 (native-FP16 widen on capable CPUs).
            (
                "letterbox_720p_to_640_planar_rgb_f16",
                OUT_W,
                OUT_H,
                PixelFormat::PlanarRgb,
                DType::F16,
                lb_crop,
            ),
        ];

        for &(suffix, ow, oh, ofmt, odt, crop) in cells {
            let name = format!("{}/{}", fmt_name(fmt), suffix);
            if !only.is_empty() && !name.contains(&only) {
                continue;
            }
            let mut dst = match proc.create_image(
                ow,
                oh,
                ofmt,
                odt,
                None,
                edgefirst_tensor::CpuAccess::ReadWrite,
            ) {
                Ok(d) => d,
                Err(e) => {
                    println!("  {name:50} [dst alloc failed: {e}]");
                    continue;
                }
            };
            // Probe once: an unsupported cell is reported and skipped instead of
            // panicking inside the timing loop.
            if let Err(e) = proc.convert(&src, &mut dst, Rotation::None, Flip::None, crop) {
                println!("  {name:50} [unsupported: {e}]");
                continue;
            }
            let r = run_bench(&name, WARMUP, iters, || {
                proc.convert(&src, &mut dst, Rotation::None, Flip::None, crop)
                    .unwrap();
            });
            // Throughput is reported against source bytes; the f32/f16 cells
            // share the same source so the numbers are comparable across dtypes.
            r.print_summary_with_throughput(src_bytes);
            suite.record(&r);
        }
    }

    suite.finish();
}
