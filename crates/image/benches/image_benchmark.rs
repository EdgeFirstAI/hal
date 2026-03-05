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
    CPUProcessor, Crop, Flip, ImageProcessorTrait, Rotation, TensorImage, GREY, NV16, PLANAR_RGB,
    RGB, RGBA, YUYV,
};
#[cfg(target_os = "linux")]
use edgefirst_tensor::TensorMemory;
use edgefirst_tensor::{TensorMapTrait, TensorTrait};
#[cfg(target_os = "linux")]
use std::mem::ManuallyDrop;

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
// 2. Resize Benchmarks — CPU, G2D, OpenGL from zidane.jpg to target sizes
// =============================================================================

#[allow(unused_variables)]
fn bench_resize(suite: &mut BenchSuite) {
    println!("\n== Resize: JPEG Source to Target Sizes ==\n");

    let zidane_path = find_testdata_path("zidane.jpg");
    let file = std::fs::read(&zidane_path).unwrap();
    let target_sizes: &[(usize, usize)] = &[(640, 640), (1280, 720), (1920, 1080)];

    // CPU: RGBA→RGBA
    {
        let src = TensorImage::load_jpeg(&file, Some(RGBA), Some(TensorMemory::Mem)).unwrap();
        let mut proc = CPUProcessor::new();
        for &(w, h) in target_sizes {
            let name = format!("resize/cpu/zidane/{}x{}/RGBA->RGBA", w, h);
            let throughput = (src.width() * src.height() * 4) as u64;
            let mut dst = TensorImage::new(w, h, RGBA, None).unwrap();
            let result = run_bench(&name, WARMUP, ITERATIONS, || {
                proc.convert(&src, &mut dst, Rotation::None, Flip::None, Crop::no_crop())
                    .unwrap();
            });
            result.print_summary_with_throughput(throughput);
            suite.record(&result);
        }
    }

    // CPU: RGB→RGB
    {
        let src = TensorImage::load_jpeg(&file, Some(RGB), Some(TensorMemory::Mem)).unwrap();
        let mut proc = CPUProcessor::new();
        for &(w, h) in target_sizes {
            let name = format!("resize/cpu/zidane/{}x{}/RGB->RGB", w, h);
            let throughput = (src.width() * src.height() * 3) as u64;
            let mut dst = TensorImage::new(w, h, RGB, None).unwrap();
            let result = run_bench(&name, WARMUP, ITERATIONS, || {
                proc.convert(&src, &mut dst, Rotation::None, Flip::None, Crop::no_crop())
                    .unwrap();
            });
            result.print_summary_with_throughput(throughput);
            suite.record(&result);
        }
    }

    // G2D: RGBA→RGBA (DMA required)
    #[cfg(target_os = "linux")]
    {
        use edgefirst_image::G2DProcessor;

        for &(w, h) in target_sizes {
            let name = format!("resize/g2d/zidane/{}x{}/RGBA->RGBA", w, h);

            let Ok(src) = TensorImage::load_jpeg(&file, Some(RGBA), Some(TensorMemory::Dma)) else {
                println!("  {:50} [skipped: DMA unavailable]", name);
                continue;
            };
            let Ok(mut dst) = TensorImage::new(w, h, RGBA, Some(TensorMemory::Dma)) else {
                println!("  {:50} [skipped: DMA unavailable]", name);
                continue;
            };
            let Ok(proc) = G2DProcessor::new() else {
                println!("  {:50} [skipped: G2D unavailable]", name);
                continue;
            };
            let throughput = (src.width() * src.height() * 4) as u64;
            // ManuallyDrop: prevent g2d_close + dlclose("libg2d.so.2") which
            // triggers Vivante galcore atexit heap corruption on i.MX8.
            let mut proc = ManuallyDrop::new(proc);

            let result = run_bench(&name, WARMUP, ITERATIONS, || {
                proc.convert(&src, &mut dst, Rotation::None, Flip::None, Crop::no_crop())
                    .unwrap();
            });
            result.print_summary_with_throughput(throughput);
            suite.record(&result);
        }
    }

    // OpenGL: RGBA→RGBA with Mem and DMA sources
    #[cfg(all(target_os = "linux", feature = "opengl"))]
    {
        use edgefirst_image::GLProcessorThreaded;

        // Mem source
        let src_mem = TensorImage::load_jpeg(&file, Some(RGBA), Some(TensorMemory::Mem)).unwrap();
        if let Ok(mut proc) = GLProcessorThreaded::new(None) {
            for &(w, h) in target_sizes {
                let name = format!("resize/opengl-mem/zidane/{}x{}/RGBA->RGBA", w, h);
                let throughput = (src_mem.width() * src_mem.height() * 4) as u64;
                let mut dst = TensorImage::new(w, h, RGBA, Some(TensorMemory::Mem)).unwrap();

                // Pre-flight
                if let Err(e) = proc.convert(
                    &src_mem,
                    &mut dst,
                    Rotation::None,
                    Flip::None,
                    Crop::no_crop(),
                ) {
                    println!("  {:50} [skipped: {}]", name, e);
                    continue;
                }

                let result = run_bench(&name, WARMUP, ITERATIONS, || {
                    proc.convert(
                        &src_mem,
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
        } else {
            println!("  [skipped: OpenGL unavailable for mem resize]");
        }

        // DMA source
        if common::dma_available() {
            if let Ok(src_dma) = TensorImage::load_jpeg(&file, Some(RGBA), Some(TensorMemory::Dma))
            {
                if let Ok(mut proc) = GLProcessorThreaded::new(None) {
                    for &(w, h) in target_sizes {
                        let name = format!("resize/opengl-dma/zidane/{}x{}/RGBA->RGBA", w, h);
                        let throughput = (src_dma.width() * src_dma.height() * 4) as u64;
                        let Ok(mut dst) = TensorImage::new(w, h, RGBA, Some(TensorMemory::Dma))
                        else {
                            println!("  {:50} [skipped: DMA unavailable]", name);
                            continue;
                        };

                        // Pre-flight
                        if let Err(e) = proc.convert(
                            &src_dma,
                            &mut dst,
                            Rotation::None,
                            Flip::None,
                            Crop::no_crop(),
                        ) {
                            println!("  {:50} [skipped: {}]", name, e);
                            continue;
                        }

                        let result = run_bench(&name, WARMUP, ITERATIONS, || {
                            proc.convert(
                                &src_dma,
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
            } else {
                println!("  [skipped: could not load zidane.jpg to DMA for OpenGL resize]");
            }
        } else {
            println!("  [skipped: DMA unavailable for OpenGL DMA resize]");
        }
    }
}

// =============================================================================
// 3. Convert Benchmarks — Format conversion at 1080p
// =============================================================================

#[allow(unused_variables)]
fn bench_convert(suite: &mut BenchSuite) {
    println!("\n== Convert: Format Conversion at 1080p ==\n");

    let w = 1920usize;
    let h = 1080usize;

    // CPU conversions
    {
        // Configs: (in_fmt, out_fmt, data_fn, bytes_per_pixel_in)
        let cpu_configs: &[(_, _, &[u8], u64)] = &[
            (YUYV, RGBA, get_test_data(w, h, YUYV), 2),
            (YUYV, RGB, get_test_data(w, h, YUYV), 2),
            (YUYV, YUYV, get_test_data(w, h, YUYV), 2),
            (RGBA, YUYV, CAMERA_1080P_RGBA, 4),
            (RGBA, RGB, CAMERA_1080P_RGBA, 4),
            (RGBA, RGBA, CAMERA_1080P_RGBA, 4),
            (RGBA, PLANAR_RGB, CAMERA_1080P_RGBA, 4),
            (RGBA, NV16, CAMERA_1080P_RGBA, 4),
        ];

        let mut proc = CPUProcessor::new();
        for &(in_fmt, out_fmt, data, bpp) in cpu_configs {
            let name = format!(
                "convert/cpu/1080p/{}->{}",
                format_name(in_fmt),
                format_name(out_fmt)
            );
            let throughput = w as u64 * h as u64 * bpp;

            let src = TensorImage::new(w, h, in_fmt, None).unwrap();
            src.tensor().map().unwrap().as_mut_slice()[..data.len()].copy_from_slice(data);
            let mut dst = TensorImage::new(w, h, out_fmt, None).unwrap();

            // Pre-flight: skip unsupported conversions
            if let Err(e) =
                proc.convert(&src, &mut dst, Rotation::None, Flip::None, Crop::no_crop())
            {
                println!("  {:50} [skipped: {}]", name, e);
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

    // G2D conversions (DMA required)
    #[cfg(target_os = "linux")]
    {
        use edgefirst_image::G2DProcessor;

        let g2d_configs: &[(_, _, &[u8], u64)] = &[
            (YUYV, RGBA, get_test_data(w, h, YUYV), 2),
            (YUYV, RGB, get_test_data(w, h, YUYV), 2),
            (YUYV, YUYV, get_test_data(w, h, YUYV), 2),
            (RGBA, RGB, CAMERA_1080P_RGBA, 4),
            (RGBA, RGBA, CAMERA_1080P_RGBA, 4),
        ];

        for &(in_fmt, out_fmt, data, bpp) in g2d_configs {
            let name = format!(
                "convert/g2d/1080p/{}->{}",
                format_name(in_fmt),
                format_name(out_fmt)
            );
            let throughput = w as u64 * h as u64 * bpp;

            let Ok(src) = TensorImage::new(w, h, in_fmt, Some(TensorMemory::Dma)) else {
                println!("  {:50} [skipped: DMA unavailable]", name);
                continue;
            };
            src.tensor().map().unwrap().as_mut_slice()[..data.len()].copy_from_slice(data);
            let Ok(mut dst) = TensorImage::new(w, h, out_fmt, Some(TensorMemory::Dma)) else {
                println!("  {:50} [skipped: DMA unavailable]", name);
                continue;
            };
            let Ok(proc) = G2DProcessor::new() else {
                println!("  {:50} [skipped: G2D unavailable]", name);
                continue;
            };
            // ManuallyDrop: prevent g2d_close + dlclose("libg2d.so.2") which
            // triggers Vivante galcore atexit heap corruption on i.MX8.
            let mut proc = ManuallyDrop::new(proc);

            // Pre-flight
            if let Err(e) =
                proc.convert(&src, &mut dst, Rotation::None, Flip::None, Crop::no_crop())
            {
                println!("  {:50} [skipped: {}]", name, e);
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

    // OpenGL conversions (DMA required)
    #[cfg(all(target_os = "linux", feature = "opengl"))]
    {
        use edgefirst_image::GLProcessorThreaded;

        let gl_configs: &[(_, _, &[u8], u64)] = &[
            (YUYV, RGBA, get_test_data(w, h, YUYV), 2),
            (RGBA, RGBA, CAMERA_1080P_RGBA, 4),
            (RGBA, PLANAR_RGB, CAMERA_1080P_RGBA, 4),
        ];

        let Ok(mut proc) = GLProcessorThreaded::new(None) else {
            println!("  [skipped: OpenGL unavailable for convert section]");
            return;
        };

        for &(in_fmt, out_fmt, data, bpp) in gl_configs {
            let name = format!(
                "convert/opengl/1080p/{}->{}",
                format_name(in_fmt),
                format_name(out_fmt)
            );
            let throughput = w as u64 * h as u64 * bpp;

            let Ok(src) = TensorImage::new(w, h, in_fmt, Some(TensorMemory::Dma)) else {
                println!("  {:50} [skipped: DMA unavailable]", name);
                continue;
            };
            src.tensor().map().unwrap().as_mut_slice()[..data.len()].copy_from_slice(data);
            let Ok(mut dst) = TensorImage::new(w, h, out_fmt, Some(TensorMemory::Dma)) else {
                println!("  {:50} [skipped: DMA unavailable]", name);
                continue;
            };

            // Pre-flight
            if let Err(e) =
                proc.convert(&src, &mut dst, Rotation::None, Flip::None, Crop::no_crop())
            {
                println!("  {:50} [skipped: {}]", name, e);
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
// 4. Rotate Benchmarks — 90, 180, 270 degree rotations at 1080p
// =============================================================================

#[allow(unused_variables)]
fn bench_rotate(suite: &mut BenchSuite) {
    println!("\n== Rotate: CPU/G2D/OpenGL at 1080p ==\n");

    let zidane = include_bytes!("../../../testdata/zidane.jpg");
    let rotations: &[(Rotation, &str)] = &[
        (Rotation::Clockwise90, "90"),
        (Rotation::from_degrees_clockwise(180), "180"),
        (Rotation::CounterClockwise90, "270"),
    ];

    let target_w = 1920usize;
    let target_h = 1080usize;

    // CPU rotations
    {
        let src = TensorImage::load_jpeg(zidane, Some(RGBA), None).unwrap();
        let mut proc = CPUProcessor::new();

        for &(rot, deg) in rotations {
            // For 90/270 rotations the output dimensions are transposed
            let (dst_w, dst_h) = match rot {
                Rotation::Clockwise90 | Rotation::CounterClockwise90 => (target_h, target_w),
                _ => (target_w, target_h),
            };
            let name = format!("rotate/cpu/{}deg/1080p/RGBA->RGBA", deg);
            let throughput = (src.width() * src.height() * 4) as u64;
            let mut dst = TensorImage::new(dst_w, dst_h, RGBA, None).unwrap();

            let result = run_bench(&name, WARMUP, ITERATIONS, || {
                proc.convert(&src, &mut dst, rot, Flip::None, Crop::no_crop())
                    .unwrap();
            });
            result.print_summary_with_throughput(throughput);
            suite.record(&result);
        }
    }

    // G2D rotations (DMA required)
    #[cfg(target_os = "linux")]
    {
        use edgefirst_image::G2DProcessor;

        let Ok(src) = TensorImage::load_jpeg(zidane, Some(RGBA), Some(TensorMemory::Dma)) else {
            println!("  [skipped: DMA unavailable for G2D rotate]");
            return;
        };
        let Ok(proc) = G2DProcessor::new() else {
            println!("  [skipped: G2D unavailable for rotate]");
            return;
        };
        let mut proc = ManuallyDrop::new(proc);

        for &(rot, deg) in rotations {
            let (dst_w, dst_h) = match rot {
                Rotation::Clockwise90 | Rotation::CounterClockwise90 => (target_h, target_w),
                _ => (target_w, target_h),
            };
            let name = format!("rotate/g2d/{}deg/1080p/RGBA->RGBA", deg);
            let throughput = (src.width() * src.height() * 4) as u64;
            let Ok(mut dst) = TensorImage::new(dst_w, dst_h, RGBA, Some(TensorMemory::Dma)) else {
                println!("  {:50} [skipped: DMA unavailable]", name);
                continue;
            };

            // Pre-flight
            if let Err(e) = proc.convert(&src, &mut dst, rot, Flip::None, Crop::no_crop()) {
                println!("  {:50} [skipped: {}]", name, e);
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

    // OpenGL rotations (DMA required)
    #[cfg(all(target_os = "linux", feature = "opengl"))]
    {
        use edgefirst_image::GLProcessorThreaded;

        let Ok(src) = TensorImage::load_jpeg(zidane, Some(RGBA), Some(TensorMemory::Dma)) else {
            println!("  [skipped: DMA unavailable for OpenGL rotate]");
            return;
        };
        let Ok(mut proc) = GLProcessorThreaded::new(None) else {
            println!("  [skipped: OpenGL unavailable for rotate]");
            return;
        };

        for &(rot, deg) in rotations {
            let (dst_w, dst_h) = match rot {
                Rotation::Clockwise90 | Rotation::CounterClockwise90 => (target_h, target_w),
                _ => (target_w, target_h),
            };
            let name = format!("rotate/opengl/{}deg/1080p/RGBA->RGBA", deg);
            let throughput = (src.width() * src.height() * 4) as u64;
            let Ok(mut dst) = TensorImage::new(dst_w, dst_h, RGBA, Some(TensorMemory::Dma)) else {
                println!("  {:50} [skipped: DMA unavailable]", name);
                continue;
            };

            // Pre-flight
            if let Err(e) = proc.convert(&src, &mut dst, rot, Flip::None, Crop::no_crop()) {
                println!("  {:50} [skipped: {}]", name, e);
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
}

// =============================================================================
// 5. Hires Benchmarks — Large image resize/convert from 1080p and 4K
// =============================================================================

#[allow(unused_variables)]
fn bench_hires(suite: &mut BenchSuite) {
    println!("\n== Hires: CPU 1080p Source ==\n");

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

    // CPU 1080p
    {
        let yuyv_src = {
            let src = TensorImage::new(1920, 1080, YUYV, None).unwrap();
            src.tensor()
                .map()
                .unwrap()
                .as_mut_slice()
                .copy_from_slice(get_test_data(1920, 1080, YUYV));
            src
        };
        let rgba_src = {
            let src = TensorImage::new(1920, 1080, RGBA, None).unwrap();
            src.tensor()
                .map()
                .unwrap()
                .as_mut_slice()
                .copy_from_slice(CAMERA_1080P_RGBA);
            src
        };
        let rgb_src = {
            let mut rgb = TensorImage::new(1920, 1080, RGB, None).unwrap();
            let mut cpu = CPUProcessor::new();
            cpu.convert(
                &rgba_src,
                &mut rgb,
                Rotation::None,
                Flip::None,
                Crop::no_crop(),
            )
            .unwrap();
            rgb
        };

        let mut proc = CPUProcessor::new();
        for config in configs_1080p {
            let name = format!("hires/cpu/1080p/{}", config.id());
            let throughput = config.throughput();
            let src = match config.in_fmt {
                f if f == YUYV => &yuyv_src,
                f if f == RGBA => &rgba_src,
                f if f == RGB => &rgb_src,
                _ => continue,
            };
            let mut dst =
                TensorImage::new(config.out_w, config.out_h, config.out_fmt, None).unwrap();

            // Pre-flight
            if let Err(e) = proc.convert(src, &mut dst, Rotation::None, Flip::None, Crop::no_crop())
            {
                println!("  {:50} [skipped: {}]", name, e);
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

    // CPU 4K
    println!("\n== Hires: CPU 4K Source ==\n");

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

    {
        let yuyv_src = {
            let src = TensorImage::new(3840, 2160, YUYV, None).unwrap();
            src.tensor()
                .map()
                .unwrap()
                .as_mut_slice()
                .copy_from_slice(get_test_data(3840, 2160, YUYV));
            src
        };
        let rgba_src = {
            let src = TensorImage::new(3840, 2160, RGBA, None).unwrap();
            src.tensor()
                .map()
                .unwrap()
                .as_mut_slice()
                .copy_from_slice(CAMERA_4K_RGBA);
            src
        };
        let rgb_src = {
            let mut rgb = TensorImage::new(3840, 2160, RGB, None).unwrap();
            let mut cpu = CPUProcessor::new();
            cpu.convert(
                &rgba_src,
                &mut rgb,
                Rotation::None,
                Flip::None,
                Crop::no_crop(),
            )
            .unwrap();
            rgb
        };

        let mut proc = CPUProcessor::new();
        for config in configs_4k {
            let name = format!("hires/cpu/4k/{}", config.id());
            let throughput = config.throughput();
            let src = match config.in_fmt {
                f if f == YUYV => &yuyv_src,
                f if f == RGBA => &rgba_src,
                f if f == RGB => &rgb_src,
                _ => continue,
            };
            let mut dst =
                TensorImage::new(config.out_w, config.out_h, config.out_fmt, None).unwrap();

            // Pre-flight
            if let Err(e) = proc.convert(src, &mut dst, Rotation::None, Flip::None, Crop::no_crop())
            {
                println!("  {:50} [skipped: {}]", name, e);
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

    // G2D 1080p (DMA required)
    #[cfg(target_os = "linux")]
    {
        use edgefirst_image::G2DProcessor;

        println!("\n== Hires: G2D 1080p Source ==\n");

        let g2d_configs_1080p: &[BenchConfig] = &[
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
        ];

        for config in g2d_configs_1080p {
            let name = format!("hires/g2d/1080p/{}", config.id());
            let throughput = config.throughput();

            let data: &[u8] = if config.in_fmt == YUYV {
                get_test_data(1920, 1080, YUYV)
            } else {
                CAMERA_1080P_RGBA
            };

            let Ok(src) = TensorImage::new(
                config.in_w,
                config.in_h,
                config.in_fmt,
                Some(TensorMemory::Dma),
            ) else {
                println!("  {:50} [skipped: DMA unavailable]", name);
                continue;
            };
            src.tensor().map().unwrap().as_mut_slice()[..data.len()].copy_from_slice(data);
            let Ok(mut dst) = TensorImage::new(
                config.out_w,
                config.out_h,
                config.out_fmt,
                Some(TensorMemory::Dma),
            ) else {
                println!("  {:50} [skipped: DMA unavailable]", name);
                continue;
            };
            let Ok(proc) = G2DProcessor::new() else {
                println!("  {:50} [skipped: G2D unavailable]", name);
                continue;
            };
            // ManuallyDrop: prevent g2d_close + dlclose("libg2d.so.2") which
            // triggers Vivante galcore atexit heap corruption on i.MX8.
            let mut proc = ManuallyDrop::new(proc);

            // Pre-flight
            if let Err(e) =
                proc.convert(&src, &mut dst, Rotation::None, Flip::None, Crop::no_crop())
            {
                println!("  {:50} [skipped: {}]", name, e);
                continue;
            }

            let result = run_bench(&name, WARMUP, ITERATIONS, || {
                proc.convert(&src, &mut dst, Rotation::None, Flip::None, Crop::no_crop())
                    .unwrap();
            });
            result.print_summary_with_throughput(throughput);
            suite.record(&result);
        }

        // G2D 4K (DMA required)
        println!("\n== Hires: G2D 4K Source ==\n");

        let g2d_configs_4k: &[BenchConfig] = &[
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
        ];

        for config in g2d_configs_4k {
            let name = format!("hires/g2d/4k/{}", config.id());
            let throughput = config.throughput();

            let data: &[u8] = if config.in_fmt == YUYV {
                get_test_data(3840, 2160, YUYV)
            } else {
                CAMERA_4K_RGBA
            };

            let Ok(src) = TensorImage::new(
                config.in_w,
                config.in_h,
                config.in_fmt,
                Some(TensorMemory::Dma),
            ) else {
                println!("  {:50} [skipped: DMA unavailable]", name);
                continue;
            };
            src.tensor().map().unwrap().as_mut_slice()[..data.len()].copy_from_slice(data);
            let Ok(mut dst) = TensorImage::new(
                config.out_w,
                config.out_h,
                config.out_fmt,
                Some(TensorMemory::Dma),
            ) else {
                println!("  {:50} [skipped: DMA unavailable]", name);
                continue;
            };
            let Ok(proc) = G2DProcessor::new() else {
                println!("  {:50} [skipped: G2D unavailable]", name);
                continue;
            };
            let mut proc = ManuallyDrop::new(proc);

            // Pre-flight
            if let Err(e) =
                proc.convert(&src, &mut dst, Rotation::None, Flip::None, Crop::no_crop())
            {
                println!("  {:50} [skipped: {}]", name, e);
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

    // OpenGL 1080p and 4K (DMA required for YUYV input; Mem also tested for RGBA)
    #[cfg(all(target_os = "linux", feature = "opengl"))]
    {
        use edgefirst_image::GLProcessorThreaded;

        println!("\n== Hires: OpenGL 1080p Source ==\n");

        let Ok(mut proc) = GLProcessorThreaded::new(None) else {
            println!("  [skipped: OpenGL unavailable for hires section]");
            return;
        };

        // 1080p YUYV→RGBA (DMA)
        let gl_1080p_dma: &[BenchConfig] = &[
            BenchConfig::new(1920, 1080, 640, 360, YUYV, RGBA),
            BenchConfig::new(1920, 1080, 1280, 720, YUYV, RGBA),
            BenchConfig::new(1920, 1080, 1920, 1080, YUYV, RGBA),
            BenchConfig::new(1920, 1080, 640, 360, RGBA, RGBA),
            BenchConfig::new(1920, 1080, 1280, 720, RGBA, RGBA),
            BenchConfig::new(1920, 1080, 1920, 1080, RGBA, RGBA),
            BenchConfig::new(1920, 1080, 640, 360, RGBA, PLANAR_RGB),
            BenchConfig::new(1920, 1080, 1280, 720, RGBA, PLANAR_RGB),
            BenchConfig::new(1920, 1080, 1920, 1080, RGBA, PLANAR_RGB),
        ];

        for config in gl_1080p_dma {
            let name = format!("hires/opengl-dma/1080p/{}", config.id());
            let throughput = config.throughput();

            let data: &[u8] = if config.in_fmt == YUYV {
                get_test_data(1920, 1080, YUYV)
            } else {
                CAMERA_1080P_RGBA
            };

            let Ok(src) = TensorImage::new(
                config.in_w,
                config.in_h,
                config.in_fmt,
                Some(TensorMemory::Dma),
            ) else {
                println!("  {:50} [skipped: DMA unavailable]", name);
                continue;
            };
            src.tensor().map().unwrap().as_mut_slice()[..data.len()].copy_from_slice(data);
            let Ok(mut dst) = TensorImage::new(
                config.out_w,
                config.out_h,
                config.out_fmt,
                Some(TensorMemory::Dma),
            ) else {
                println!("  {:50} [skipped: DMA unavailable]", name);
                continue;
            };

            // Pre-flight
            if let Err(e) =
                proc.convert(&src, &mut dst, Rotation::None, Flip::None, Crop::no_crop())
            {
                println!("  {:50} [skipped: {}]", name, e);
                continue;
            }

            let result = run_bench(&name, WARMUP, ITERATIONS, || {
                proc.convert(&src, &mut dst, Rotation::None, Flip::None, Crop::no_crop())
                    .unwrap();
            });
            result.print_summary_with_throughput(throughput);
            suite.record(&result);
        }

        // 1080p RGBA→RGBA (Mem)
        let gl_1080p_mem: &[BenchConfig] = &[
            BenchConfig::new(1920, 1080, 640, 360, RGBA, RGBA),
            BenchConfig::new(1920, 1080, 1280, 720, RGBA, RGBA),
            BenchConfig::new(1920, 1080, 1920, 1080, RGBA, RGBA),
        ];

        for config in gl_1080p_mem {
            let name = format!("hires/opengl-mem/1080p/{}", config.id());
            let throughput = config.throughput();

            let src = TensorImage::new(
                config.in_w,
                config.in_h,
                config.in_fmt,
                Some(TensorMemory::Mem),
            )
            .unwrap();
            src.tensor()
                .map()
                .unwrap()
                .as_mut_slice()
                .copy_from_slice(CAMERA_1080P_RGBA);
            let mut dst = TensorImage::new(
                config.out_w,
                config.out_h,
                config.out_fmt,
                Some(TensorMemory::Mem),
            )
            .unwrap();

            // Pre-flight
            if let Err(e) =
                proc.convert(&src, &mut dst, Rotation::None, Flip::None, Crop::no_crop())
            {
                println!("  {:50} [skipped: {}]", name, e);
                continue;
            }

            let result = run_bench(&name, WARMUP, ITERATIONS, || {
                proc.convert(&src, &mut dst, Rotation::None, Flip::None, Crop::no_crop())
                    .unwrap();
            });
            result.print_summary_with_throughput(throughput);
            suite.record(&result);
        }

        println!("\n== Hires: OpenGL 4K Source ==\n");

        // 4K DMA
        let gl_4k_dma: &[BenchConfig] = &[
            BenchConfig::new(3840, 2160, 640, 360, YUYV, RGBA),
            BenchConfig::new(3840, 2160, 1280, 720, YUYV, RGBA),
            BenchConfig::new(3840, 2160, 1920, 1080, YUYV, RGBA),
            BenchConfig::new(3840, 2160, 3840, 2160, YUYV, RGBA),
            BenchConfig::new(3840, 2160, 640, 360, RGBA, RGBA),
            BenchConfig::new(3840, 2160, 1280, 720, RGBA, RGBA),
            BenchConfig::new(3840, 2160, 1920, 1080, RGBA, RGBA),
            BenchConfig::new(3840, 2160, 3840, 2160, RGBA, RGBA),
        ];

        for config in gl_4k_dma {
            let name = format!("hires/opengl-dma/4k/{}", config.id());
            let throughput = config.throughput();

            let data: &[u8] = if config.in_fmt == YUYV {
                get_test_data(3840, 2160, YUYV)
            } else {
                CAMERA_4K_RGBA
            };

            let Ok(src) = TensorImage::new(
                config.in_w,
                config.in_h,
                config.in_fmt,
                Some(TensorMemory::Dma),
            ) else {
                println!("  {:50} [skipped: DMA unavailable]", name);
                continue;
            };
            src.tensor().map().unwrap().as_mut_slice()[..data.len()].copy_from_slice(data);
            let Ok(mut dst) = TensorImage::new(
                config.out_w,
                config.out_h,
                config.out_fmt,
                Some(TensorMemory::Dma),
            ) else {
                println!("  {:50} [skipped: DMA unavailable]", name);
                continue;
            };

            // Pre-flight
            if let Err(e) =
                proc.convert(&src, &mut dst, Rotation::None, Flip::None, Crop::no_crop())
            {
                println!("  {:50} [skipped: {}]", name, e);
                continue;
            }

            let result = run_bench(&name, WARMUP, ITERATIONS, || {
                proc.convert(&src, &mut dst, Rotation::None, Flip::None, Crop::no_crop())
                    .unwrap();
            });
            result.print_summary_with_throughput(throughput);
            suite.record(&result);
        }

        // 4K Mem
        let gl_4k_mem: &[BenchConfig] = &[
            BenchConfig::new(3840, 2160, 640, 360, RGBA, RGBA),
            BenchConfig::new(3840, 2160, 1280, 720, RGBA, RGBA),
            BenchConfig::new(3840, 2160, 1920, 1080, RGBA, RGBA),
            BenchConfig::new(3840, 2160, 3840, 2160, RGBA, RGBA),
        ];

        for config in gl_4k_mem {
            let name = format!("hires/opengl-mem/4k/{}", config.id());
            let throughput = config.throughput();

            let src = TensorImage::new(
                config.in_w,
                config.in_h,
                config.in_fmt,
                Some(TensorMemory::Mem),
            )
            .unwrap();
            src.tensor()
                .map()
                .unwrap()
                .as_mut_slice()
                .copy_from_slice(CAMERA_4K_RGBA);
            let mut dst = TensorImage::new(
                config.out_w,
                config.out_h,
                config.out_fmt,
                Some(TensorMemory::Mem),
            )
            .unwrap();

            // Pre-flight
            if let Err(e) =
                proc.convert(&src, &mut dst, Rotation::None, Flip::None, Crop::no_crop())
            {
                println!("  {:50} [skipped: {}]", name, e);
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
// Main
// =============================================================================

fn main() {
    let mut suite = BenchSuite::from_args();

    println!("Image Benchmark — custom in-process harness (no fork)");
    println!("  warmup={WARMUP}  iterations={ITERATIONS}");

    bench_load(&mut suite);
    bench_resize(&mut suite);
    bench_convert(&mut suite);
    bench_rotate(&mut suite);
    bench_hires(&mut suite);

    suite.finish();
    println!("\nDone.");
}
