// SPDX-FileCopyrightText: Copyright 2025 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! Vision pipeline benchmarks using the edgefirst-bench harness.
//!
//! All benchmarks run through `ImageProcessor`, which selects the best
//! available backend (OpenGL > G2D > CPU) automatically. Set
//! `EDGEFIRST_FORCE_BACKEND=cpu|g2d|opengl` to pin a specific backend.
//!
//! The harness runs all benchmarks sequentially in a single process with
//! no forking, avoiding GPU driver state corruption across fork boundaries.
//!
//! ## Run benchmarks (host)
//! ```bash
//! cargo bench -p edgefirst-image --bench pipeline_benchmark
//! ```
//!
//! ## Run benchmarks (on-target, cross-compiled)
//! ```bash
//! ./pipeline_benchmark --bench
//! ```

mod common;

use common::{calculate_letterbox, get_test_data, run_bench, BenchConfig, BenchSuite};

use edgefirst_image::{Crop, Flip, ImageProcessor, ImageProcessorTrait, Rect, Rotation};
use edgefirst_image::{
    BGRA, GREY, NV12, NV16, PLANAR_RGB, PLANAR_RGB_INT8, RGB, RGBA, RGB_INT8, VYUY, YUYV,
};
use edgefirst_tensor::{TensorMapTrait, TensorTrait};

const WARMUP: usize = 10;
const ITERATIONS: usize = 100;

// =============================================================================
// Primary Benchmarks: Camera -> Model (letterbox)
// =============================================================================

fn bench_letterbox(configs: &[BenchConfig], proc: &mut ImageProcessor, suite: &mut BenchSuite) {
    println!("\n== Letterbox: Camera -> Model ==\n");

    for config in configs {
        let (left, top, new_w, new_h) =
            calculate_letterbox(config.in_w, config.in_h, config.out_w, config.out_h);
        let crop = Crop::new()
            .with_dst_rect(Some(Rect::new(left, top, new_w, new_h)))
            .with_dst_color(Some([114, 114, 114, 255]));
        let throughput = config.throughput();
        let name = format!("letterbox/{}", config.id());

        let Ok(src) = proc.create_image(config.in_w, config.in_h, config.in_fmt) else {
            println!("  {:50} [skipped: allocation failed]", name);
            continue;
        };
        let data = get_test_data(config.in_w, config.in_h, config.in_fmt);
        src.tensor().map().unwrap().as_mut_slice()[..data.len()].copy_from_slice(data);
        let Ok(mut dst) = proc.create_image(config.out_w, config.out_h, config.out_fmt) else {
            println!("  {:50} [skipped: allocation failed]", name);
            continue;
        };

        // Pre-flight: skip unsupported format combos for this backend
        if let Err(e) = proc.convert(&src, &mut dst, Rotation::None, Flip::None, crop) {
            println!("  {:50} [unsupported: {}]", name, e);
            continue;
        }

        let result = run_bench(&name, WARMUP, ITERATIONS, || {
            proc.convert(&src, &mut dst, Rotation::None, Flip::None, crop)
                .unwrap();
        });
        result.print_summary_with_throughput(throughput);
        suite.record(&result);
    }
}

// =============================================================================
// Secondary Benchmarks: Format Conversion (no resize)
// =============================================================================

fn bench_convert(configs: &[BenchConfig], proc: &mut ImageProcessor, suite: &mut BenchSuite) {
    println!("\n== Convert: Format Conversion ==\n");

    for config in configs {
        let throughput = config.throughput();
        let name = format!("convert/{}", config.id());

        let Ok(src) = proc.create_image(config.in_w, config.in_h, config.in_fmt) else {
            println!("  {:50} [skipped: allocation failed]", name);
            continue;
        };
        let data = get_test_data(config.in_w, config.in_h, config.in_fmt);
        src.tensor().map().unwrap().as_mut_slice()[..data.len()].copy_from_slice(data);
        let Ok(mut dst) = proc.create_image(config.out_w, config.out_h, config.out_fmt) else {
            println!("  {:50} [skipped: allocation failed]", name);
            continue;
        };

        // Pre-flight: skip unsupported format combos for this backend
        if let Err(e) = proc.convert(&src, &mut dst, Rotation::None, Flip::None, Crop::no_crop()) {
            println!("  {:50} [unsupported: {}]", name, e);
            continue;
        }

        let result = run_bench(&name, WARMUP, ITERATIONS, || {
            proc.convert(&src, &mut dst, Rotation::None, Flip::None, Crop::no_crop())
                .unwrap();
        });
        result.print_summary_with_throughput(throughput);
        suite.record(&result);
    }
}

// =============================================================================
// Resize Benchmarks: Resolution Change
// =============================================================================

fn bench_resize(configs: &[BenchConfig], proc: &mut ImageProcessor, suite: &mut BenchSuite) {
    println!("\n== Resize: Resolution Change ==\n");

    for config in configs {
        let throughput = config.throughput();
        let name = format!("resize/{}", config.id());

        let Ok(src) = proc.create_image(config.in_w, config.in_h, config.in_fmt) else {
            println!("  {:50} [skipped: allocation failed]", name);
            continue;
        };
        let data = get_test_data(config.in_w, config.in_h, config.in_fmt);
        src.tensor().map().unwrap().as_mut_slice()[..data.len()].copy_from_slice(data);
        let Ok(mut dst) = proc.create_image(config.out_w, config.out_h, config.out_fmt) else {
            println!("  {:50} [skipped: allocation failed]", name);
            continue;
        };

        // Pre-flight: skip unsupported format combos for this backend
        if let Err(e) = proc.convert(&src, &mut dst, Rotation::None, Flip::None, Crop::no_crop()) {
            println!("  {:50} [unsupported: {}]", name, e);
            continue;
        }

        let result = run_bench(&name, WARMUP, ITERATIONS, || {
            proc.convert(&src, &mut dst, Rotation::None, Flip::None, Crop::no_crop())
                .unwrap();
        });
        result.print_summary_with_throughput(throughput);
        suite.record(&result);
    }
}

// =============================================================================
// Rotation Benchmarks
// =============================================================================

fn bench_rotate(proc: &mut ImageProcessor, suite: &mut BenchSuite, max_width: usize) {
    println!("\n== Rotate ==\n");

    let rotations = [
        (Rotation::Clockwise90, "CW90"),
        (Rotation::Rotate180, "180"),
        (Rotation::CounterClockwise90, "CCW90"),
    ];
    let resolutions = [(1920, 1080, "1080p"), (3840, 2160, "4K")];

    for (w, h, res) in resolutions {
        if w > max_width {
            continue;
        }
        for (rotation, rot_name) in &rotations {
            let name = format!("rotate/{res}/{rot_name}/YUYV->RGBA");

            let Ok(src) = proc.create_image(w, h, YUYV) else {
                println!("  {:50} [skipped: allocation failed]", name);
                continue;
            };
            let data = get_test_data(w, h, YUYV);
            src.tensor().map().unwrap().as_mut_slice()[..data.len()].copy_from_slice(data);

            // For CW90/CCW90 rotations, dst dimensions are swapped
            let (dw, dh) = match rotation {
                Rotation::Clockwise90 | Rotation::CounterClockwise90 => (h, w),
                _ => (w, h),
            };
            let Ok(mut dst) = proc.create_image(dw, dh, RGBA) else {
                println!("  {:50} [skipped: allocation failed]", name);
                continue;
            };

            if let Err(e) = proc.convert(&src, &mut dst, *rotation, Flip::None, Crop::no_crop()) {
                println!("  {:50} [unsupported: {}]", name, e);
                continue;
            }

            let throughput = (w * h * 2) as u64; // YUYV = 2 bytes/pixel
            let result = run_bench(&name, WARMUP, ITERATIONS, || {
                proc.convert(&src, &mut dst, *rotation, Flip::None, Crop::no_crop())
                    .unwrap();
            });
            result.print_summary_with_throughput(throughput);
            suite.record(&result);
        }
    }
}

// =============================================================================
// Flip Benchmarks
// =============================================================================

fn bench_flip(proc: &mut ImageProcessor, suite: &mut BenchSuite, max_width: usize) {
    println!("\n== Flip ==\n");

    let flips = [
        (Flip::Horizontal, "horizontal"),
        (Flip::Vertical, "vertical"),
    ];
    let resolutions = [(1920, 1080, "1080p"), (3840, 2160, "4K")];

    for (w, h, res) in resolutions {
        if w > max_width {
            continue;
        }
        for (flip, flip_name) in &flips {
            let name = format!("flip/{res}/{flip_name}/YUYV->RGBA");

            let Ok(src) = proc.create_image(w, h, YUYV) else {
                println!("  {:50} [skipped: allocation failed]", name);
                continue;
            };
            let data = get_test_data(w, h, YUYV);
            src.tensor().map().unwrap().as_mut_slice()[..data.len()].copy_from_slice(data);
            let Ok(mut dst) = proc.create_image(w, h, RGBA) else {
                println!("  {:50} [skipped: allocation failed]", name);
                continue;
            };

            if let Err(e) = proc.convert(&src, &mut dst, Rotation::None, *flip, Crop::no_crop()) {
                println!("  {:50} [unsupported: {}]", name, e);
                continue;
            }

            let throughput = (w * h * 2) as u64; // YUYV = 2 bytes/pixel
            let result = run_bench(&name, WARMUP, ITERATIONS, || {
                proc.convert(&src, &mut dst, Rotation::None, *flip, Crop::no_crop())
                    .unwrap();
            });
            result.print_summary_with_throughput(throughput);
            suite.record(&result);
        }
    }
}

// =============================================================================
// Letterbox Pipeline Benchmarks: Realistic camera -> model with clear+resize
// =============================================================================

fn bench_letterbox_pipeline(
    proc: &mut ImageProcessor,
    suite: &mut BenchSuite,
    max_width: usize,
    skip_nv12_planar: bool,
) {
    println!("\n== Letterbox Pipeline: Realistic Camera -> Model ==\n");

    let pipelines: Vec<_> = [
        (1280, 720, "720p"),
        (1920, 1080, "1080p"),
        (3840, 2160, "4K"),
    ]
    .into_iter()
    .filter(|(w, _, _)| *w <= max_width)
    .collect();
    let all_formats = [
        (YUYV, RGBA),
        (YUYV, RGB),
        (YUYV, RGB_INT8),
        (YUYV, PLANAR_RGB),
        (YUYV, PLANAR_RGB_INT8),
        (NV12, RGBA),
        (NV12, RGB),
        (NV12, RGB_INT8),
        (NV12, PLANAR_RGB),
        (NV12, PLANAR_RGB_INT8),
    ];
    let formats: Vec<_> = all_formats
        .into_iter()
        .filter(|(inf, outf)| {
            !(skip_nv12_planar && *inf == NV12 && matches!(*outf, PLANAR_RGB | PLANAR_RGB_INT8))
        })
        .collect();

    for (w, h, res) in pipelines {
        for (in_fmt, out_fmt) in &formats {
            let name = format!(
                "pipeline/{res}/{}->{}/640x640",
                common::format_name(*in_fmt),
                common::format_name(*out_fmt),
            );

            let (left, top, new_w, new_h) = calculate_letterbox(w, h, 640, 640);
            let crop = Crop::new()
                .with_dst_rect(Some(Rect::new(left, top, new_w, new_h)))
                .with_dst_color(Some([114, 114, 114, 255]));

            let Ok(src) = proc.create_image(w, h, *in_fmt) else {
                println!("  {:50} [skipped: allocation failed]", name);
                continue;
            };
            let data = get_test_data(w, h, *in_fmt);
            src.tensor().map().unwrap().as_mut_slice()[..data.len()].copy_from_slice(data);
            let Ok(mut dst) = proc.create_image(640, 640, *out_fmt) else {
                println!("  {:50} [skipped: allocation failed]", name);
                continue;
            };

            if let Err(e) = proc.convert(&src, &mut dst, Rotation::None, Flip::None, crop) {
                println!("  {:50} [unsupported: {}]", name, e);
                continue;
            }

            // Calculate correct throughput based on actual input format bytes
            let throughput = BenchConfig::new(w, h, 640, 640, *in_fmt, *out_fmt).throughput();
            let result = run_bench(&name, WARMUP, ITERATIONS, || {
                proc.convert(&src, &mut dst, Rotation::None, Flip::None, crop)
                    .unwrap();
            });
            result.print_summary_with_throughput(throughput);
            suite.record(&result);
        }
    }
}

// =============================================================================
// Main
// =============================================================================

fn main() {
    let mut suite = BenchSuite::from_args();
    let mut proc = ImageProcessor::new().expect("Failed to create ImageProcessor");

    // Optional env var to limit maximum source resolution (e.g. BENCH_MAX_WIDTH=1920 skips 4K)
    let max_width: usize = std::env::var("BENCH_MAX_WIDTH")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(usize::MAX);
    // Optional: skip NV12 planar combos (hangs on Vivante GC7000UL)
    let skip_nv12_planar = std::env::var("BENCH_SKIP_NV12_PLANAR").is_ok();

    println!("Pipeline Benchmark — edgefirst-bench harness");
    println!("  warmup={WARMUP}  iterations={ITERATIONS}");
    if max_width < usize::MAX {
        println!("  max_width={max_width} (filtering enabled)");
    }
    if skip_nv12_planar {
        println!("  skip_nv12_planar=true");
    }

    // --- Letterbox configs (720p, 1080p, 4K) ---
    let letterbox_configs = vec![
        // 720p camera -> YOLO standard (640x640)
        BenchConfig::new(1280, 720, 640, 640, YUYV, RGBA),
        BenchConfig::new(1280, 720, 640, 640, YUYV, RGB),
        BenchConfig::new(1280, 720, 640, 640, YUYV, RGB_INT8),
        BenchConfig::new(1280, 720, 640, 640, YUYV, PLANAR_RGB),
        BenchConfig::new(1280, 720, 640, 640, YUYV, PLANAR_RGB_INT8),
        BenchConfig::new(1280, 720, 640, 640, NV12, RGBA),
        BenchConfig::new(1280, 720, 640, 640, NV12, RGB),
        BenchConfig::new(1280, 720, 640, 640, NV12, RGB_INT8),
        BenchConfig::new(1280, 720, 640, 640, NV12, PLANAR_RGB),
        BenchConfig::new(1280, 720, 640, 640, NV12, PLANAR_RGB_INT8),
        // 1080p camera -> YOLO standard (640x640)
        BenchConfig::new(1920, 1080, 640, 640, YUYV, RGBA),
        BenchConfig::new(1920, 1080, 640, 640, YUYV, RGB),
        BenchConfig::new(1920, 1080, 640, 640, YUYV, RGB_INT8),
        BenchConfig::new(1920, 1080, 640, 640, YUYV, PLANAR_RGB),
        BenchConfig::new(1920, 1080, 640, 640, YUYV, PLANAR_RGB_INT8),
        BenchConfig::new(1920, 1080, 640, 640, VYUY, RGBA),
        BenchConfig::new(1920, 1080, 640, 640, VYUY, RGB),
        BenchConfig::new(1920, 1080, 640, 640, NV12, RGBA),
        BenchConfig::new(1920, 1080, 640, 640, NV12, RGB),
        BenchConfig::new(1920, 1080, 640, 640, NV12, RGB_INT8),
        BenchConfig::new(1920, 1080, 640, 640, NV12, PLANAR_RGB),
        BenchConfig::new(1920, 1080, 640, 640, NV12, PLANAR_RGB_INT8),
        // 4K camera -> YOLO standard (640x640)
        BenchConfig::new(3840, 2160, 640, 640, YUYV, RGBA),
        BenchConfig::new(3840, 2160, 640, 640, YUYV, RGB),
        BenchConfig::new(3840, 2160, 640, 640, YUYV, RGB_INT8),
        BenchConfig::new(3840, 2160, 640, 640, YUYV, PLANAR_RGB),
        BenchConfig::new(3840, 2160, 640, 640, YUYV, PLANAR_RGB_INT8),
        BenchConfig::new(3840, 2160, 640, 640, NV12, RGBA),
        BenchConfig::new(3840, 2160, 640, 640, NV12, RGB),
        BenchConfig::new(3840, 2160, 640, 640, NV12, RGB_INT8),
        BenchConfig::new(3840, 2160, 640, 640, NV12, PLANAR_RGB),
        BenchConfig::new(3840, 2160, 640, 640, NV12, PLANAR_RGB_INT8),
        // 4K camera -> YOLO hi-res (1280x1280)
        BenchConfig::new(3840, 2160, 1280, 1280, YUYV, RGBA),
        BenchConfig::new(3840, 2160, 1280, 1280, YUYV, RGB),
        BenchConfig::new(3840, 2160, 1280, 1280, YUYV, RGB_INT8),
        BenchConfig::new(3840, 2160, 1280, 1280, YUYV, PLANAR_RGB_INT8),
        BenchConfig::new(3840, 2160, 1280, 1280, NV12, RGBA),
        BenchConfig::new(3840, 2160, 1280, 1280, NV12, RGB),
        BenchConfig::new(3840, 2160, 1280, 1280, NV12, RGB_INT8),
        BenchConfig::new(3840, 2160, 1280, 1280, NV12, PLANAR_RGB_INT8),
        // BGRA destinations
        BenchConfig::new(1920, 1080, 640, 640, YUYV, BGRA),
        BenchConfig::new(3840, 2160, 640, 640, YUYV, BGRA),
        // NV16 input
        BenchConfig::new(1920, 1080, 640, 640, NV16, RGBA),
    ];
    let is_nv12_planar =
        |c: &BenchConfig| c.in_fmt == NV12 && matches!(c.out_fmt, PLANAR_RGB | PLANAR_RGB_INT8);
    let letterbox_configs: Vec<_> = letterbox_configs
        .into_iter()
        .filter(|c| c.in_w <= max_width)
        .filter(|c| !(skip_nv12_planar && is_nv12_planar(c)))
        .collect();
    bench_letterbox(&letterbox_configs, &mut proc, &mut suite);

    // --- Convert configs ---
    let convert_configs = vec![
        // 1080p conversions
        BenchConfig::new(1920, 1080, 1920, 1080, YUYV, RGBA),
        BenchConfig::new(1920, 1080, 1920, 1080, YUYV, RGB),
        BenchConfig::new(1920, 1080, 1920, 1080, YUYV, RGB_INT8),
        BenchConfig::new(1920, 1080, 1920, 1080, VYUY, RGBA),
        BenchConfig::new(1920, 1080, 1920, 1080, VYUY, RGB),
        BenchConfig::new(1920, 1080, 1920, 1080, NV12, RGBA),
        BenchConfig::new(1920, 1080, 1920, 1080, NV12, RGB),
        BenchConfig::new(1920, 1080, 1920, 1080, RGB, RGBA),
        // 4K conversions
        BenchConfig::new(3840, 2160, 3840, 2160, YUYV, RGBA),
        BenchConfig::new(3840, 2160, 3840, 2160, YUYV, RGB),
        BenchConfig::new(3840, 2160, 3840, 2160, YUYV, RGB_INT8),
        BenchConfig::new(3840, 2160, 3840, 2160, NV12, RGBA),
        BenchConfig::new(3840, 2160, 3840, 2160, NV12, RGB),
        BenchConfig::new(3840, 2160, 3840, 2160, RGB, RGBA),
        // BGRA conversions
        BenchConfig::new(1920, 1080, 1920, 1080, RGBA, BGRA),
        BenchConfig::new(1920, 1080, 1920, 1080, RGB, BGRA),
        // NV16 conversions
        BenchConfig::new(1920, 1080, 1920, 1080, NV16, RGBA),
        // GREY destination
        BenchConfig::new(1920, 1080, 1920, 1080, RGBA, GREY),
    ];
    let convert_configs: Vec<_> = convert_configs
        .into_iter()
        .filter(|c| c.in_w <= max_width)
        .collect();
    bench_convert(&convert_configs, &mut proc, &mut suite);

    // --- Resize configs ---
    let resize_configs = vec![
        // 4K -> 1080p
        BenchConfig::new(3840, 2160, 1920, 1080, YUYV, RGBA),
        BenchConfig::new(3840, 2160, 1920, 1080, YUYV, RGB),
        // 4K -> 720p
        BenchConfig::new(3840, 2160, 1280, 720, YUYV, RGBA),
        // 1080p -> 720p
        BenchConfig::new(1920, 1080, 1280, 720, YUYV, RGBA),
    ];
    let resize_configs: Vec<_> = resize_configs
        .into_iter()
        .filter(|c| c.in_w <= max_width)
        .collect();
    bench_resize(&resize_configs, &mut proc, &mut suite);

    // --- Rotation benchmarks ---
    bench_rotate(&mut proc, &mut suite, max_width);

    // --- Flip benchmarks ---
    bench_flip(&mut proc, &mut suite, max_width);

    // --- Letterbox pipeline ---
    bench_letterbox_pipeline(&mut proc, &mut suite, max_width, skip_nv12_planar);

    suite.finish();
    println!("\nDone.");
}
