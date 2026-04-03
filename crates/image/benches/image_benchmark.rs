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

use edgefirst_image::{Crop, Flip, ImageProcessor, ImageProcessorTrait, Rotation};
use edgefirst_tensor::{DType, PixelFormat, TensorMapTrait, TensorMemory, TensorTrait};

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

    // Load 3 test images to Mem backend in PixelFormat::Rgba
    for name in &["jaguar.jpg", "person.jpg", "zidane.jpg"] {
        let path = find_testdata_path(name);
        let bench_name = format!("load/mem/RGBA/{}", name.strip_suffix(".jpg").unwrap());
        let file = std::fs::read(&path).unwrap();
        let result = run_bench(&bench_name, WARMUP, ITERATIONS, || {
            let img = edgefirst_image::load_image(
                &file,
                Some(PixelFormat::Rgba),
                Some(TensorMemory::Mem),
            )
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
                let img = edgefirst_image::load_image(
                    zidane,
                    Some(PixelFormat::Rgba),
                    Some(TensorMemory::Shm),
                )
                .expect("Failed to load JPEG");
                std::hint::black_box(img);
            });
            result.print_summary();
            suite.record(&result);
        }

        if common::dma_available() {
            let name = "load/dma/RGBA/zidane";
            let result = run_bench(name, WARMUP, ITERATIONS, || {
                let img = edgefirst_image::load_image(
                    zidane,
                    Some(PixelFormat::Rgba),
                    Some(TensorMemory::Dma),
                )
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
    for (fmt, fmt_name) in &[
        (PixelFormat::Rgba, "RGBA"),
        (PixelFormat::Rgb, "RGB"),
        (PixelFormat::Grey, "GREY"),
    ] {
        let bench_name = format!("load/mem/{}/zidane", fmt_name);
        let result = run_bench(&bench_name, WARMUP, ITERATIONS, || {
            let img = edgefirst_image::load_image(zidane, Some(*fmt), Some(TensorMemory::Mem))
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
        let src = edgefirst_image::load_image(&file, Some(PixelFormat::Rgba), None).unwrap();
        let (sw, sh) = (src.width().unwrap(), src.height().unwrap());
        let src_dyn = src;

        for &(w, h) in target_sizes {
            let name = format!("resize/zidane/{}x{}/RGBA->RGBA", w, h);
            let throughput = (sw * sh * 4) as u64;
            let Ok(mut dst) = proc.create_image(w, h, PixelFormat::Rgba, DType::U8, None) else {
                println!("  {:50} [skipped: allocation failed]", name);
                continue;
            };

            // Pre-flight
            if let Err(e) = proc.convert(
                &src_dyn,
                &mut dst,
                Rotation::None,
                Flip::None,
                Crop::no_crop(),
            ) {
                println!("  {:50} [unsupported: {}]", name, e);
                continue;
            }

            let result = run_bench(&name, WARMUP, ITERATIONS, || {
                proc.convert(
                    &src_dyn,
                    &mut dst,
                    Rotation::None,
                    Flip::None,
                    Crop::no_crop(),
                )
                .unwrap();
            });
            result.print_summary_with_throughput(throughput);
            suite.record(&result);
        }
    }

    // RGB→RGB resize
    {
        let src = edgefirst_image::load_image(&file, Some(PixelFormat::Rgb), None).unwrap();
        let (sw, sh) = (src.width().unwrap(), src.height().unwrap());
        let src_dyn = src;

        for &(w, h) in target_sizes {
            let name = format!("resize/zidane/{}x{}/RGB->RGB", w, h);
            let throughput = (sw * sh * 3) as u64;
            let Ok(mut dst) = proc.create_image(w, h, PixelFormat::Rgb, DType::U8, None) else {
                println!("  {:50} [skipped: allocation failed]", name);
                continue;
            };

            // Pre-flight
            if let Err(e) = proc.convert(
                &src_dyn,
                &mut dst,
                Rotation::None,
                Flip::None,
                Crop::no_crop(),
            ) {
                println!("  {:50} [unsupported: {}]", name, e);
                continue;
            }

            let result = run_bench(&name, WARMUP, ITERATIONS, || {
                proc.convert(
                    &src_dyn,
                    &mut dst,
                    Rotation::None,
                    Flip::None,
                    Crop::no_crop(),
                )
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
        (
            PixelFormat::Yuyv,
            PixelFormat::Rgba,
            get_test_data(w, h, PixelFormat::Yuyv),
            2,
        ),
        (
            PixelFormat::Yuyv,
            PixelFormat::Rgb,
            get_test_data(w, h, PixelFormat::Yuyv),
            2,
        ),
        (
            PixelFormat::Yuyv,
            PixelFormat::Yuyv,
            get_test_data(w, h, PixelFormat::Yuyv),
            2,
        ),
        (PixelFormat::Rgba, PixelFormat::Yuyv, CAMERA_1080P_RGBA, 4),
        (PixelFormat::Rgba, PixelFormat::Rgb, CAMERA_1080P_RGBA, 4),
        (PixelFormat::Rgba, PixelFormat::Rgba, CAMERA_1080P_RGBA, 4),
        (
            PixelFormat::Rgba,
            PixelFormat::PlanarRgb,
            CAMERA_1080P_RGBA,
            4,
        ),
        (PixelFormat::Rgba, PixelFormat::Nv16, CAMERA_1080P_RGBA, 4),
    ];

    for &(in_fmt, out_fmt, data, bpp) in configs {
        let name = format!(
            "convert/1080p/{}->{}",
            format_name(in_fmt),
            format_name(out_fmt)
        );
        let throughput = w as u64 * h as u64 * bpp;

        let Ok(src) = proc.create_image(w, h, in_fmt, DType::U8, None) else {
            println!("  {:50} [skipped: allocation failed]", name);
            continue;
        };
        src.as_u8().unwrap().map().unwrap().as_mut_slice()[..data.len()].copy_from_slice(data);
        let Ok(mut dst) = proc.create_image(w, h, out_fmt, DType::U8, None) else {
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
    let src = edgefirst_image::load_image(zidane, Some(PixelFormat::Rgba), None)
        .expect("Failed to load zidane.jpg");
    let (w, h) = (src.width().unwrap(), src.height().unwrap());
    let src_dyn = src;

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
        let name = format!("rotate/{}deg/PixelFormat::Rgba", deg);
        let throughput = (w * h * 4) as u64;

        let Ok(mut dst) = proc.create_image(dst_w, dst_h, PixelFormat::Rgba, DType::U8, None)
        else {
            println!("  {:50} [skipped: allocation failed]", name);
            continue;
        };

        // Pre-flight
        if let Err(e) = proc.convert(&src_dyn, &mut dst, rot, Flip::None, Crop::no_crop()) {
            println!("  {:50} [unsupported: {}]", name, e);
            continue;
        }

        let result = run_bench(&name, WARMUP, ITERATIONS, || {
            proc.convert(&src_dyn, &mut dst, rot, Flip::None, Crop::no_crop())
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
        BenchConfig::new(1920, 1080, 640, 360, PixelFormat::Yuyv, PixelFormat::Yuyv),
        BenchConfig::new(1920, 1080, 1280, 720, PixelFormat::Yuyv, PixelFormat::Yuyv),
        BenchConfig::new(1920, 1080, 1920, 1080, PixelFormat::Yuyv, PixelFormat::Yuyv),
        BenchConfig::new(1920, 1080, 640, 360, PixelFormat::Yuyv, PixelFormat::Rgba),
        BenchConfig::new(1920, 1080, 1280, 720, PixelFormat::Yuyv, PixelFormat::Rgba),
        BenchConfig::new(1920, 1080, 1920, 1080, PixelFormat::Yuyv, PixelFormat::Rgba),
        BenchConfig::new(1920, 1080, 640, 360, PixelFormat::Yuyv, PixelFormat::Rgb),
        BenchConfig::new(1920, 1080, 1280, 720, PixelFormat::Yuyv, PixelFormat::Rgb),
        BenchConfig::new(1920, 1080, 1920, 1080, PixelFormat::Yuyv, PixelFormat::Rgb),
        BenchConfig::new(1920, 1080, 640, 360, PixelFormat::Rgba, PixelFormat::Rgba),
        BenchConfig::new(1920, 1080, 1280, 720, PixelFormat::Rgba, PixelFormat::Rgba),
        BenchConfig::new(1920, 1080, 1920, 1080, PixelFormat::Rgba, PixelFormat::Rgba),
        BenchConfig::new(1920, 1080, 640, 360, PixelFormat::Rgba, PixelFormat::Rgb),
        BenchConfig::new(1920, 1080, 1280, 720, PixelFormat::Rgba, PixelFormat::Rgb),
        BenchConfig::new(1920, 1080, 1920, 1080, PixelFormat::Rgba, PixelFormat::Rgb),
        BenchConfig::new(1920, 1080, 640, 360, PixelFormat::Rgb, PixelFormat::Rgb),
        BenchConfig::new(1920, 1080, 1280, 720, PixelFormat::Rgb, PixelFormat::Rgb),
        BenchConfig::new(1920, 1080, 1920, 1080, PixelFormat::Rgb, PixelFormat::Rgb),
    ];

    run_hires_configs(proc, suite, configs_1080p, "hires/1080p");

    // --- 4K source ---
    println!("\n== Hires: 4K Source ==\n");

    let configs_4k: &[BenchConfig] = &[
        BenchConfig::new(3840, 2160, 640, 360, PixelFormat::Yuyv, PixelFormat::Yuyv),
        BenchConfig::new(3840, 2160, 1280, 720, PixelFormat::Yuyv, PixelFormat::Yuyv),
        BenchConfig::new(3840, 2160, 1920, 1080, PixelFormat::Yuyv, PixelFormat::Yuyv),
        BenchConfig::new(3840, 2160, 3840, 2160, PixelFormat::Yuyv, PixelFormat::Yuyv),
        BenchConfig::new(3840, 2160, 640, 360, PixelFormat::Yuyv, PixelFormat::Rgba),
        BenchConfig::new(3840, 2160, 1280, 720, PixelFormat::Yuyv, PixelFormat::Rgba),
        BenchConfig::new(3840, 2160, 1920, 1080, PixelFormat::Yuyv, PixelFormat::Rgba),
        BenchConfig::new(3840, 2160, 3840, 2160, PixelFormat::Yuyv, PixelFormat::Rgba),
        BenchConfig::new(3840, 2160, 640, 360, PixelFormat::Yuyv, PixelFormat::Rgb),
        BenchConfig::new(3840, 2160, 1280, 720, PixelFormat::Yuyv, PixelFormat::Rgb),
        BenchConfig::new(3840, 2160, 1920, 1080, PixelFormat::Yuyv, PixelFormat::Rgb),
        BenchConfig::new(3840, 2160, 3840, 2160, PixelFormat::Yuyv, PixelFormat::Rgb),
        BenchConfig::new(3840, 2160, 640, 360, PixelFormat::Rgba, PixelFormat::Rgba),
        BenchConfig::new(3840, 2160, 1280, 720, PixelFormat::Rgba, PixelFormat::Rgba),
        BenchConfig::new(3840, 2160, 1920, 1080, PixelFormat::Rgba, PixelFormat::Rgba),
        BenchConfig::new(3840, 2160, 3840, 2160, PixelFormat::Rgba, PixelFormat::Rgba),
        BenchConfig::new(3840, 2160, 640, 360, PixelFormat::Rgba, PixelFormat::Rgb),
        BenchConfig::new(3840, 2160, 1280, 720, PixelFormat::Rgba, PixelFormat::Rgb),
        BenchConfig::new(3840, 2160, 1920, 1080, PixelFormat::Rgba, PixelFormat::Rgb),
        BenchConfig::new(3840, 2160, 3840, 2160, PixelFormat::Rgba, PixelFormat::Rgb),
        BenchConfig::new(3840, 2160, 640, 360, PixelFormat::Rgb, PixelFormat::Rgb),
        BenchConfig::new(3840, 2160, 1280, 720, PixelFormat::Rgb, PixelFormat::Rgb),
        BenchConfig::new(3840, 2160, 1920, 1080, PixelFormat::Rgb, PixelFormat::Rgb),
        BenchConfig::new(3840, 2160, 3840, 2160, PixelFormat::Rgb, PixelFormat::Rgb),
    ];

    run_hires_configs(proc, suite, configs_4k, "hires/4k");
}

/// Run a set of hires benchmark configs using the ImageProcessor.
fn run_hires_configs(
    proc: &mut ImageProcessor,
    suite: &mut BenchSuite,
    configs: &[BenchConfig],
    prefix: &str,
) {
    // Pre-create source images for each input format used in these configs.
    // We need PixelFormat::Yuyv, PixelFormat::Rgba, and PixelFormat::Rgb sources at the resolution of the first config
    // (all configs in a batch share the same input resolution).
    let (src_w, src_h) = (configs[0].in_w, configs[0].in_h);

    // PixelFormat::Yuyv source
    let yuyv_src = {
        let Ok(src) = proc.create_image(src_w, src_h, PixelFormat::Yuyv, DType::U8, None) else {
            println!(
                "  [skipped: could not allocate PixelFormat::Yuyv source {}x{}]",
                src_w, src_h
            );
            return;
        };
        let data = get_test_data(src_w, src_h, PixelFormat::Yuyv);
        src.as_u8().unwrap().map().unwrap().as_mut_slice()[..data.len()].copy_from_slice(data);
        src
    };

    // PixelFormat::Rgba source
    let rgba_src = {
        let Ok(src) = proc.create_image(src_w, src_h, PixelFormat::Rgba, DType::U8, None) else {
            println!(
                "  [skipped: could not allocate PixelFormat::Rgba source {}x{}]",
                src_w, src_h
            );
            return;
        };
        let data: &[u8] = if src_w == 1920 && src_h == 1080 {
            CAMERA_1080P_RGBA
        } else {
            CAMERA_4K_RGBA
        };
        src.as_u8().unwrap().map().unwrap().as_mut_slice()[..data.len()].copy_from_slice(data);
        src
    };

    // PixelFormat::Rgb source — convert from PixelFormat::Rgba
    let rgb_src = {
        let Ok(mut rgb) = proc.create_image(src_w, src_h, PixelFormat::Rgb, DType::U8, None) else {
            println!(
                "  [skipped: could not allocate PixelFormat::Rgb source {}x{}]",
                src_w, src_h
            );
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
            PixelFormat::Yuyv => &yuyv_src,
            PixelFormat::Rgba => &rgba_src,
            PixelFormat::Rgb => &rgb_src,
            _ => continue,
        };

        let Ok(mut dst) =
            proc.create_image(config.out_w, config.out_h, config.out_fmt, DType::U8, None)
        else {
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
// 6. Import Benchmarks — import_image from DMA-BUF file descriptors
// =============================================================================

fn bench_import(proc: &ImageProcessor, suite: &mut BenchSuite) {
    println!("\n== Import: import_image from DMA-BUF ==\n");

    #[cfg(not(target_os = "linux"))]
    {
        let _ = (proc, suite);
        println!("  Skipping import benchmarks: DMA-BUF is Linux-only");
    }

    #[cfg(target_os = "linux")]
    {
        use edgefirst_tensor::{PlaneDescriptor, TensorDyn};

        if !common::dma_available() {
            println!("  Skipping import benchmarks: DMA not available");
            return;
        }

        let configs: &[(usize, usize, PixelFormat, &str)] = &[
            (1920, 1080, PixelFormat::Rgba, "import/rgba/1080p"),
            (1920, 1080, PixelFormat::Nv12, "import/nv12/1080p"),
            (3840, 2160, PixelFormat::Rgba, "import/rgba/4k"),
        ];

        for &(w, h, fmt, name) in configs {
            // Allocate a DMA-backed tensor to obtain a valid DMA-BUF fd.
            let Ok(src_tensor) =
                TensorDyn::image(w, h, fmt, DType::U8, Some(TensorMemory::Dma))
            else {
                println!("  {:50} [skipped: DMA allocation failed]", name);
                continue;
            };

            // Pre-flight: verify import_image works with this configuration.
            let fd = src_tensor.dmabuf().expect("dmabuf fd");
            let pd = PlaneDescriptor::new(fd).expect("PlaneDescriptor");
            if let Err(e) = proc.import_image(pd, None, w, h, fmt, DType::U8) {
                println!("  {:50} [unsupported: {}]", name, e);
                continue;
            }

            let throughput = match fmt {
                PixelFormat::Nv12 => (w * h * 3 / 2) as u64,
                PixelFormat::Rgba => (w * h * 4) as u64,
                _ => (w * h) as u64,
            };

            let result = run_bench(name, WARMUP, ITERATIONS, || {
                let fd = src_tensor.dmabuf().expect("dmabuf fd");
                let pd = PlaneDescriptor::new(fd).expect("PlaneDescriptor");
                let img = proc
                    .import_image(pd, None, w, h, fmt, DType::U8)
                    .expect("import_image failed");
                std::hint::black_box(img);
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
    env_logger::init();
    let mut suite = BenchSuite::from_args();
    let mut proc = ImageProcessor::new().expect("Failed to create ImageProcessor");

    println!("Image Benchmark — edgefirst-bench harness");
    println!("  warmup={WARMUP}  iterations={ITERATIONS}");

    bench_load(&mut suite);
    bench_resize(&mut proc, &mut suite);
    bench_convert(&mut proc, &mut suite);
    bench_rotate(&mut proc, &mut suite);
    bench_hires(&mut proc, &mut suite);
    bench_import(&proc, &mut suite);

    suite.finish();
    println!("\nDone.");
}
