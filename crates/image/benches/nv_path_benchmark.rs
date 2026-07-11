// SPDX-FileCopyrightText: Copyright 2025 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! NV12/NV16/NV24 conversion-path A/B benchmark (`ExternalSampler` vs
//! `ShaderR8`), cross-platform.
//!
//! Self-contained: sources are synthesized (Y luma gradient + neutral chroma),
//! so NO testdata files are required on the target. Run twice per platform,
//! forcing each path, then compare:
//!
//! ```bash
//! EDGEFIRST_NV_CONVERT_PATH=sampler ./nv_path_benchmark --json sampler.json
//! EDGEFIRST_NV_CONVERT_PATH=shader  ./nv_path_benchmark --json shader.json
//! ```
//!
//! On non-DMA backends (e.g. orin) NV* takes the R8-upload `ShaderR8` path
//! regardless of the env var; the run still records its latency.

mod common;

use common::{run_bench, BenchSuite};

use edgefirst_image::{Crop, Flip, ImageProcessor, ImageProcessorTrait, Rotation};
use edgefirst_tensor::{DType, PixelFormat, TensorMapTrait, TensorTrait};

const WARMUP: usize = 10;
const ITERATIONS: usize = 100;

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

fn main() {
    let mut suite = BenchSuite::from_args();
    let path = std::env::var("EDGEFIRST_NV_CONVERT_PATH").unwrap_or_else(|_| "auto".into());
    println!("\n== NV conversion-path benchmark (EDGEFIRST_NV_CONVERT_PATH={path}) ==\n");

    let mut proc = match ImageProcessor::new() {
        Ok(p) => p,
        Err(e) => {
            println!("  [skipped: ImageProcessor init failed: {e}]");
            suite.finish();
            return;
        }
    };

    // Camera 1280x720 → model 640x640 (the profiler's letterbox geometry).
    // Placement is now derived by `Crop::resolve` from the src/dst dims.
    let (in_w, in_h) = (1280usize, 720usize);
    let (out_w, out_h) = (640usize, 640usize);
    let lb_crop = Crop::letterbox([114, 114, 114, 255]);

    for fmt in [PixelFormat::Nv12, PixelFormat::Nv16, PixelFormat::Nv24] {
        let src = match proc.create_image(
            in_w,
            in_h,
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
            let n = nv_len(fmt, in_w, in_h);
            fill_nv(&mut map.as_mut_slice()[..n], in_w, in_h);
        }
        let src_bytes = nv_len(fmt, in_w, in_h) as u64;

        // (1) Convert, same size → RGBA (sampling cost, no resize).
        if let Ok(mut dst) = proc.create_image(
            in_w,
            in_h,
            PixelFormat::Rgba,
            DType::U8,
            None,
            edgefirst_tensor::CpuAccess::ReadWrite,
        ) {
            let name = format!("{path}/{fmt:?}/convert_720p_rgba").to_lowercase();
            if let Err(e) =
                proc.convert(&src, &mut dst, Rotation::None, Flip::None, Crop::no_crop())
            {
                println!("  {name:50} [unsupported: {e}]");
            } else {
                let r = run_bench(&name, WARMUP, ITERATIONS, || {
                    proc.convert(&src, &mut dst, Rotation::None, Flip::None, Crop::no_crop())
                        .unwrap();
                });
                r.print_summary_with_throughput(src_bytes);
                suite.record(&r);
            }
        }

        // (2) Letterbox 720p → 640x640 packed RGB (the profiler op).
        if let Ok(mut dst) = proc.create_image(
            out_w,
            out_h,
            PixelFormat::Rgb,
            DType::U8,
            None,
            edgefirst_tensor::CpuAccess::ReadWrite,
        ) {
            let name = format!("{path}/{fmt:?}/letterbox_720p_to_640_rgb").to_lowercase();
            if let Err(e) = proc.convert(&src, &mut dst, Rotation::None, Flip::None, lb_crop) {
                println!("  {name:50} [unsupported: {e}]");
            } else {
                let r = run_bench(&name, WARMUP, ITERATIONS, || {
                    proc.convert(&src, &mut dst, Rotation::None, Flip::None, lb_crop)
                        .unwrap();
                });
                r.print_summary_with_throughput(src_bytes);
                suite.record(&r);
            }
        }
    }

    suite.finish();
}
