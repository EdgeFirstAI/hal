// SPDX-FileCopyrightText: Copyright 2025 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! Image processing benchmarks using the edgefirst-bench in-process harness.
//!
//! Divan runs each benchmark in a forked subprocess. On i.MX8MP this causes
//! `corrupted double-linked list` crashes after GPU driver state is poisoned
//! across the fork boundary. This harness runs all benchmarks sequentially in
//! a single process with no forking.
//!
//! ## Run benchmarks (host)
//! ```bash
//! cargo bench -p edgefirst-image --bench image_benchmark
//! ```
//!
//! ## Run benchmarks (on-target, cross-compiled)
//! ```bash
//! ./image_benchmark --bench
//! ```

mod common;

use common::{find_testdata_path, format_name, get_test_data, run_bench, BenchConfig};
use edgefirst_bench::BenchSuite;

use edgefirst_image::{
    Crop, Flip, ImageProcessor, ImageProcessorTrait, Rotation, TensorImage, GREY, NV16,
    PLANAR_RGB, RGB, RGBA, YUYV,
};
#[cfg(target_os = "linux")]
use edgefirst_tensor::TensorMemory;
use edgefirst_tensor::{TensorMapTrait, TensorTrait};

const WARMUP: usize = 10;
const ITERATIONS: usize = 100;

// Embedded test data — used in convert and hires sections
const CAMERA_1080P_RGBA: &[u8] = include_bytes!("../../../testdata/camera1080p.rgba");
const CAMERA_4K_RGBA: &[u8] = include_bytes!("../../../testdata/camera4k.rgba");

// =============================================================================
// 1. Load Benchmarks — JPEG loading to different backends and formats
// =============================================================================

fn bench_load(suite: &mut BenchSuite) {
    println!("\n== Load: JPEG to Memory Backends ==\n");

    // Load 3 test images to Mem backend in RGBA
    for name in &["jaguar.jpg", "person.jpg", "zidane.jpg"] {
        let path = find_testdata_path(name);
        let bench_name = format!("load/mem/RGBA/{}", name.strip_suffix(".jpg").unwrap());
        let file = std::fs::read(&path).unwrap();
        let result = run_bench(&bench_name, WARMUP, ITERATIONS, || {
            let img = TensorImage::load_jpeg(&file, Some(RGBA), Some(TensorMemory::Mem))
                .expect("Failed to load JPEG");
            std::hint::black_box(img);
        });
        result.print_summary();
        suite.record(&result);
    }

    // Load zidane to Shm and DMA backends
    #[cfg(target_os = "linux")]
    {
        let zidane = include_bytes!("../../../testdata/zidane.jpg");

        {
            let name = "load/shm/RGBA/zidane";
            let result = run_bench(name, WARMUP, ITERATIONS, || {
                let img = TensorImage::load_jpeg(zidane, Some(RGBA), Some(TensorMemory::Shm))
                    .expect("Failed to load JPEG");
                std::hint::black_box(img);
            });
            result.print_summary();
            suite.record(&result);
        }

        if common::dma_available() {
            let name = "load/dma/RGBA/zidane";
            let result = run_bench(name, WARMUP, ITERATIONS, || {
                let img = TensorImage::load_jpeg(zidane, Some(RGBA), Some(TensorMemory::Dma))
                    .expect("Failed to load JPEG");
                std::hint::black_box(img);
            });
            result.print_summary();
            suite.record(&result);
        } else {
            println!("  {:50} [skipped: DMA unavailable]", "load/dma/RGBA/zidane");
        }
    }

    // Load zidane to different formats (Mem backend)
    println!("\n== Load: JPEG to Different Formats ==\n");
    let zidane = include_bytes!("../../../testdata/zidane.jpg");
    for (fmt, fmt_name) in &[(RGBA, "RGBA"), (RGB, "RGB"), (GREY, "GREY")] {
        let bench_name = format!("load/mem/{}/zidane", fmt_name);
        let result = run_bench(&bench_name, WARMUP, ITERATIONS, || {
            let img = TensorImage::load_jpeg(zidane, Some(*fmt), Some(TensorMemory::Mem))
                .expect("Failed to load JPEG");
            std::hint::black_box(img);
        });
        result.print_summary();
        suite.record(&result);
    }
}

// =============================================================================
// 2. Resize Benchmarks — resize from zidane.jpg to target sizes
// =============================================================================

fn bench_resize(proc: &mut ImageProcessor, suite: &mut BenchSuite) {
    println!("\n== Resize: JPEG Source to Target Sizes ==\n");

    let zidane_path = find_testdata_path("zidane.jpg");
    let file = std::fs::read(&zidane_path).unwrap();
    let target_sizes: &[(usize, usize)] = &[(640, 640), (1280, 720), (1920, 1080)];

    // RGBA→RGBA resize
    {
        let src = TensorImage::load_jpeg(&file, Some(RGBA), None).unwrap();
        let (sw, sh) = (src.width(), src.height());

        for &(w, h) in target_sizes {
            let name = format!("resize/zidane/{}x{}/RGBA->RGBA", w, h);
            let throughput = (sw * sh * 4) as u64;
            let Ok(mut dst) = proc.create_image(w, h, RGBA) else {
                println!("  {:50} [skipped: allocation failed]", name);
                continue;
            };

            // Pre-flight
            if let Err(e) =
                proc.convert(&src, &mut dst, Rotation::None, Flip::None, Crop::no_crop())
            {
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

    // RGB→RGB resize
    {
        let src = TensorImage::load_jpeg(&file, Some(RGB), None).unwrap();
        let (sw, sh) = (src.width(), src.height());

        for &(w, h) in target_sizes {
            let name = format!("resize/zidane/{}x{}/RGB->RGB", w, h);
            let throughput = (sw * sh * 3) as u64;
            let Ok(mut dst) = proc.create_image(w, h, RGB) else {
                println!("  {:50} [skipped: allocation failed]", name);
                continue;
            };

            // Pre-flight
            if let Err(e) =
                proc.convert(&src, &mut dst, Rotation::None, Flip::None, Crop::no_crop())
            {
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
}

// =============================================================================
// 3. Convert Benchmarks — Format conversion at 1080p
// =============================================================================

fn bench_convert(proc: &mut ImageProcessor, suite: &mut BenchSuite) {
    println!("\n== Convert: Format Conversion at 1080p ==\n");

    let w = 1920usize;
    let h = 1080usize;

    let configs: &[(_, _, &[u8], u64)] = &[
        (YUYV, RGBA, get_test_data(w, h, YUYV), 2),
        (YUYV, RGB, get_test_data(w, h, YUYV), 2),
        (YUYV, YUYV, get_test_data(w, h, YUYV), 2),
        (RGBA, YUYV, CAMERA_1080P_RGBA, 4),
        (RGBA, RGB, CAMERA_1080P_RGBA, 4),
        (RGBA, RGBA, CAMERA_1080P_RGBA, 4),
        (RGBA, PLANAR_RGB, CAMERA_1080P_RGBA, 4),
        (RGBA, NV16, CAMERA_1080P_RGBA, 4),
    ];

    for &(in_fmt, out_fmt, data, bpp) in configs {
        let name = format!(
            "convert/1080p/{}->{}",
            format_name(in_fmt),
            format_name(out_fmt)
        );
        let throughput = w as u64 * h as u64 * bpp;

        let Ok(src) = proc.create_image(w, h, in_fmt) else {
            println!("  {:50} [skipped: allocation failed]", name);
            continue;
        };
        src.tensor().map().unwrap().as_mut_slice()[..data.len()].copy_from_slice(data);
        let Ok(mut dst) = proc.create_image(w, h, out_fmt) else {
            println!("  {:50} [skipped: allocation failed]", name);
            continue;
        };

        // Pre-flight: skip unsupported conversions
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
// 4. Rotate Benchmarks — 90, 180, 270 degree rotations
// =============================================================================

fn bench_rotate(proc: &mut ImageProcessor, suite: &mut BenchSuite) {
    println!("\n== Rotate: 90/180/270 degree rotations ==\n");

    let zidane = include_bytes!("../../../testdata/zidane.jpg");
    let src = TensorImage::load_jpeg(zidane, Some(RGBA), None).expect("Failed to load zidane.jpg");
    let (w, h) = (src.width(), src.height());

    let rotations: &[(Rotation, &str)] = &[
        (Rotation::Clockwise90, "90"),
        (Rotation::from_degrees_clockwise(180), "180"),
        (Rotation::CounterClockwise90, "270"),
    ];

    for &(rot, deg) in rotations {
        // For 90/270 rotations the output dimensions are transposed
        let (dst_w, dst_h) = match rot {
            Rotation::Clockwise90 | Rotation::CounterClockwise90 => (h, w),
            _ => (w, h),
        };
        let name = format!("rotate/{}deg/RGBA", deg);
        let throughput = (w * h * 4) as u64;

        let Ok(mut dst) = proc.create_image(dst_w, dst_h, RGBA) else {
            println!("  {:50} [skipped: allocation failed]", name);
            continue;
        };

        // Pre-flight
        if let Err(e) = proc.convert(&src, &mut dst, rot, Flip::None, Crop::no_crop()) {
            println!("  {:50} [unsupported: {}]", name, e);
            continue;
        }

        let result = run_bench(&name, WARMUP, ITERATIONS, || {
            proc.convert(&src, &mut dst, rot, Flip::None, Crop::no_crop())
                .unwrap();
        });
        result.print_summary_with_throughput(throughput);
        suite.record(&result);
    }
}

// =============================================================================
// 5. Hires Benchmarks — Large image resize/convert from 1080p and 4K
// =============================================================================

fn bench_hires(proc: &mut ImageProcessor, suite: &mut BenchSuite) {
    // --- 1080p source ---
    println!("\n== Hires: 1080p Source ==\n");

    let configs_1080p: &[BenchConfig] = &[
        BenchConfig::new(1920, 1080, 640, 360, YUYV, YUYV),
        BenchConfig::new(1920, 1080, 1280, 720, YUYV, YUYV),
        BenchConfig::new(1920, 1080, 1920, 1080, YUYV, YUYV),
        BenchConfig::new(1920, 1080, 640, 360, YUYV, RGBA),
        BenchConfig::new(1920, 1080, 1280, 720, YUYV, RGBA),
        BenchConfig::new(1920, 1080, 1920, 1080, YUYV, RGBA),
        BenchConfig::new(1920, 1080, 640, 360, YUYV, RGB),
        BenchConfig::new(1920, 1080, 1280, 720, YUYV, RGB),
        BenchConfig::new(1920, 1080, 1920, 1080, YUYV, RGB),
        BenchConfig::new(1920, 1080, 640, 360, RGBA, RGBA),
        BenchConfig::new(1920, 1080, 1280, 720, RGBA, RGBA),
        BenchConfig::new(1920, 1080, 1920, 1080, RGBA, RGBA),
        BenchConfig::new(1920, 1080, 640, 360, RGBA, RGB),
        BenchConfig::new(1920, 1080, 1280, 720, RGBA, RGB),
        BenchConfig::new(1920, 1080, 1920, 1080, RGBA, RGB),
        BenchConfig::new(1920, 1080, 640, 360, RGB, RGB),
        BenchConfig::new(1920, 1080, 1280, 720, RGB, RGB),
        BenchConfig::new(1920, 1080, 1920, 1080, RGB, RGB),
    ];

    run_hires_configs(proc, suite, &configs_1080p, "hires/1080p");

    // --- 4K source ---
    println!("\n== Hires: 4K Source ==\n");

    let configs_4k: &[BenchConfig] = &[
        BenchConfig::new(3840, 2160, 640, 360, YUYV, YUYV),
        BenchConfig::new(3840, 2160, 1280, 720, YUYV, YUYV),
        BenchConfig::new(3840, 2160, 1920, 1080, YUYV, YUYV),
        BenchConfig::new(3840, 2160, 3840, 2160, YUYV, YUYV),
        BenchConfig::new(3840, 2160, 640, 360, YUYV, RGBA),
        BenchConfig::new(3840, 2160, 1280, 720, YUYV, RGBA),
        BenchConfig::new(3840, 2160, 1920, 1080, YUYV, RGBA),
        BenchConfig::new(3840, 2160, 3840, 2160, YUYV, RGBA),
        BenchConfig::new(3840, 2160, 640, 360, YUYV, RGB),
        BenchConfig::new(3840, 2160, 1280, 720, YUYV, RGB),
        BenchConfig::new(3840, 2160, 1920, 1080, YUYV, RGB),
        BenchConfig::new(3840, 2160, 3840, 2160, YUYV, RGB),
        BenchConfig::new(3840, 2160, 640, 360, RGBA, RGBA),
        BenchConfig::new(3840, 2160, 1280, 720, RGBA, RGBA),
        BenchConfig::new(3840, 2160, 1920, 1080, RGBA, RGBA),
        BenchConfig::new(3840, 2160, 3840, 2160, RGBA, RGBA),
        BenchConfig::new(3840, 2160, 640, 360, RGBA, RGB),
        BenchConfig::new(3840, 2160, 1280, 720, RGBA, RGB),
        BenchConfig::new(3840, 2160, 1920, 1080, RGBA, RGB),
        BenchConfig::new(3840, 2160, 3840, 2160, RGBA, RGB),
        BenchConfig::new(3840, 2160, 640, 360, RGB, RGB),
        BenchConfig::new(3840, 2160, 1280, 720, RGB, RGB),
        BenchConfig::new(3840, 2160, 1920, 1080, RGB, RGB),
        BenchConfig::new(3840, 2160, 3840, 2160, RGB, RGB),
    ];

    run_hires_configs(proc, suite, &configs_4k, "hires/4k");
}

/// Run a set of hires benchmark configs using the ImageProcessor.
fn run_hires_configs(
    proc: &mut ImageProcessor,
    suite: &mut BenchSuite,
    configs: &[BenchConfig],
    prefix: &str,
) {
    // Pre-create source images for each input format used in these configs.
    // We need YUYV, RGBA, and RGB sources at the resolution of the first config
    // (all configs in a batch share the same input resolution).
    let (src_w, src_h) = (configs[0].in_w, configs[0].in_h);

    // YUYV source
    let yuyv_src = {
        let Ok(src) = proc.create_image(src_w, src_h, YUYV) else {
            println!("  [skipped: could not allocate YUYV source {}x{}]", src_w, src_h);
            return;
        };
        let data = get_test_data(src_w, src_h, YUYV);
        src.tensor().map().unwrap().as_mut_slice()[..data.len()].copy_from_slice(data);
        src
    };

    // RGBA source
    let rgba_src = {
        let Ok(src) = proc.create_image(src_w, src_h, RGBA) else {
            println!("  [skipped: could not allocate RGBA source {}x{}]", src_w, src_h);
            return;
        };
        let data: &[u8] = if src_w == 1920 && src_h == 1080 {
            CAMERA_1080P_RGBA
        } else {
            CAMERA_4K_RGBA
        };
        src.tensor().map().unwrap().as_mut_slice()[..data.len()].copy_from_slice(data);
        src
    };

    // RGB source — convert from RGBA
    let rgb_src = {
        let Ok(mut rgb) = proc.create_image(src_w, src_h, RGB) else {
            println!("  [skipped: could not allocate RGB source {}x{}]", src_w, src_h);
            return;
        };
        if let Err(e) = proc.convert(
            &rgba_src,
            &mut rgb,
            Rotation::None,
            Flip::None,
            Crop::no_crop(),
        ) {
            println!("  [skipped: RGBA->RGB conversion failed for source: {}]", e);
            return;
        }
        rgb
    };

    for config in configs {
        let name = format!("{}/{}", prefix, config.id());
        let throughput = config.throughput();

        let src = match config.in_fmt {
            f if f == YUYV => &yuyv_src,
            f if f == RGBA => &rgba_src,
            f if f == RGB => &rgb_src,
            _ => continue,
        };

        let Ok(mut dst) = proc.create_image(config.out_w, config.out_h, config.out_fmt) else {
            println!("  {:50} [skipped: allocation failed]", name);
            continue;
        };

        // Pre-flight
        if let Err(e) = proc.convert(src, &mut dst, Rotation::None, Flip::None, Crop::no_crop()) {
            println!("  {:50} [unsupported: {}]", name, e);
            continue;
        }

        let result = run_bench(&name, WARMUP, ITERATIONS, || {
            proc.convert(src, &mut dst, Rotation::None, Flip::None, Crop::no_crop())
                .unwrap();
        });
        result.print_summary_with_throughput(throughput);
        suite.record(&result);
    }
}

// =============================================================================
// Main
// =============================================================================

fn main() {
    let mut suite = BenchSuite::from_args();
    let mut proc = ImageProcessor::new().expect("Failed to create ImageProcessor");

    println!("Image Benchmark — edgefirst-bench harness");
    println!("  warmup={WARMUP}  iterations={ITERATIONS}");

    bench_load(&mut suite);
    bench_resize(&mut proc, &mut suite);
    bench_convert(&mut proc, &mut suite);
    bench_rotate(&mut proc, &mut suite);
    bench_hires(&mut proc, &mut suite);

    suite.finish();
    println!("\nDone.");
}
