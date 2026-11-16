// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! CPU JPEG-preprocessing path benchmark.
//!
//! JPEG decodes to the NV family (NV12 4:2:0, NV16 4:2:2, NV24 4:4:4) and the
//! model takes planar RGB (`PlanarRgb`, i.e. CHW), optionally widened to F32.
//! This bench measures the cells on that path.
//!
//! The CPU backend is pinned via `EDGEFIRST_FORCE_BACKEND=cpu`. Sources are
//! synthesized (Y luma gradient + neutral chroma), so no testdata files are
//! required.
//!
//! ```bash
//! EDGEFIRST_FORCE_BACKEND=cpu ./cpu_preprocess_benchmark --json cpu.json
//! # profile a single cell under perf (crank iterations for sample density):
//! EDGEFIRST_FORCE_BACKEND=cpu EDGEFIRST_BENCH_ONLY=nv12/letterbox \
//!   EDGEFIRST_BENCH_ITERS=2000 perf record -e cycles:u -g -- ./cpu_preprocess_benchmark
//! ```

mod common;

use common::{run_bench, BenchSuite};

use edgefirst_image::{
    ConvertOptions, ImageProcessor, ImageProcessorTrait, LetterboxPlacement, Normalization,
};
use edgefirst_tensor::{DType, PixelFormat, TensorMapTrait, TensorTrait};

const WARMUP: usize = 10;
const DEFAULT_ITERATIONS: usize = 200;

// Camera 1280x720 → model 640x640.
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
    // Pins the CPU backend when `EDGEFIRST_FORCE_BACKEND` is unset.
    // SAFETY: the process is single-threaded at the top of main().
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

    let lb_crop = ConvertOptions::new().letterbox([114, 114, 114, 255]);
    let plain = ConvertOptions::new();
    // SegFormer-style input: ImageNet normalization, top-left letterbox, pad
    // 0.0 after normalization.
    let segformer = ConvertOptions::new()
        .letterbox([0, 0, 0, 255])
        .placement(LetterboxPlacement::TopLeft)
        .pad_output([0.0; 4])
        .normalization(Normalization::imagenet());

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

        // Each cell: (suffix, out dims, out fmt, out dtype, options, quantization).
        #[allow(clippy::type_complexity)]
        let cells: &[(
            &str,
            usize,
            usize,
            PixelFormat,
            DType,
            ConvertOptions,
            Option<(f32, i32)>,
        )] = &[
            // Pure convert (no resize).
            (
                "convert_720p_to_rgb",
                IN_W,
                IN_H,
                PixelFormat::Rgb,
                DType::U8,
                plain,
                None,
            ),
            (
                "convert_720p_to_planar_rgb",
                IN_W,
                IN_H,
                PixelFormat::PlanarRgb,
                DType::U8,
                plain,
                None,
            ),
            // Letterbox (convert + resize).
            (
                "letterbox_720p_to_640_planar_rgb",
                OUT_W,
                OUT_H,
                PixelFormat::PlanarRgb,
                DType::U8,
                lb_crop,
                None,
            ),
            // Full preprocess: planar RGB widened to F32 (model input dtype).
            (
                "letterbox_720p_to_640_planar_rgb_f32",
                OUT_W,
                OUT_H,
                PixelFormat::PlanarRgb,
                DType::F32,
                lb_crop,
                None,
            ),
            // Full preprocess to F16.
            (
                "letterbox_720p_to_640_planar_rgb_f16",
                OUT_W,
                OUT_H,
                PixelFormat::PlanarRgb,
                DType::F16,
                lb_crop,
                None,
            ),
            // Plain i8 (the XOR bias).
            (
                "letterbox_720p_to_640_planar_rgb_i8",
                OUT_W,
                OUT_H,
                PixelFormat::PlanarRgb,
                DType::I8,
                lb_crop,
                None,
            ),
            // Normalized F32 model input (per-channel affine on top of the widen).
            (
                "segformer_720p_to_640_planar_rgb_f32",
                OUT_W,
                OUT_H,
                PixelFormat::PlanarRgb,
                DType::F32,
                segformer,
                None,
            ),
            // Normalization fused with the NPU input quantization.
            (
                "segformer_720p_to_640_planar_rgb_i8_quant",
                OUT_W,
                OUT_H,
                PixelFormat::PlanarRgb,
                DType::I8,
                segformer,
                Some((0.0187, -14)),
            ),
        ];

        for &(suffix, ow, oh, ofmt, odt, opts, quant) in cells {
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
            if let Some((scale, zp)) = quant {
                dst.set_quantization(edgefirst_tensor::Quantization::per_tensor(scale, zp))
                    .unwrap();
            }
            // Probe once: an unsupported cell is reported and skipped.
            if let Err(e) = proc.convert(&src, &mut dst, Some(opts)) {
                println!("  {name:50} [unsupported: {e}]");
                continue;
            }
            let r = run_bench(&name, WARMUP, iters, || {
                proc.convert(&src, &mut dst, Some(opts)).unwrap();
            });
            // Throughput is reported against source bytes.
            r.print_summary_with_throughput(src_bytes);
            suite.record(&r);
        }
    }

    suite.finish();
}
