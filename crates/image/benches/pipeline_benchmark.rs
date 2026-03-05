// SPDX-FileCopyrightText: Copyright 2025 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! Vision pipeline benchmarks using a custom in-process harness.
//!
//! Criterion runs each benchmark in a forked subprocess. On i.MX8MP this
//! causes `corrupted double-linked list` crashes after GPU driver state is
//! poisoned across the fork boundary. On i.MX95 the Mali/Panfrost driver
//! exhausts EGL display connections by the 4th OpenGL benchmark. This is
//! fundamentally incompatible with GPU hardware initialization.
//!
//! This harness runs all benchmarks sequentially in a single process with
//! no forking, using the same `BenchResult`/`run_bench` pattern proven in
//! `crates/gpu-probe/src/bench.rs`.
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

use edgefirst_image::{CPUProcessor, Crop, Flip, ImageProcessorTrait, Rect, Rotation, TensorImage};
use edgefirst_image::{NV12, PLANAR_RGB_INT8, RGB, RGBA, RGB_INT8, VYUY, YUYV};
#[cfg(target_os = "linux")]
use edgefirst_tensor::TensorMemory;
use edgefirst_tensor::{TensorMapTrait, TensorTrait};
#[cfg(target_os = "linux")]
use std::mem::ManuallyDrop;

#[cfg(feature = "opencv")]
use opencv::{
    core::{set_num_threads, Mat, Scalar, Size, CV_8UC2, CV_8UC3, CV_8UC4},
    imgproc,
    prelude::*,
};

const WARMUP: usize = 10;
const ITERATIONS: usize = 100;

// =============================================================================
// Primary Benchmarks: Camera → Model (letterbox)
// =============================================================================

#[allow(unused_variables)]
fn bench_letterbox(configs: &[BenchConfig], suite: &mut BenchSuite) {
    println!("\n== Letterbox: Camera → Model ==\n");

    for config in configs {
        let (left, top, new_w, new_h) =
            calculate_letterbox(config.in_w, config.in_h, config.out_w, config.out_h);
        let crop = Crop::new()
            .with_dst_rect(Some(Rect::new(left, top, new_w, new_h)))
            .with_dst_color(Some([114, 114, 114, 255]));
        let throughput = config.throughput();

        // HAL CPU
        {
            let name = format!("letterbox/cpu/{}", config.id());
            let src = TensorImage::new(config.in_w, config.in_h, config.in_fmt, None).unwrap();
            let data = get_test_data(config.in_w, config.in_h, config.in_fmt);
            src.tensor().map().unwrap().as_mut_slice()[..data.len()].copy_from_slice(data);
            let mut dst =
                TensorImage::new(config.out_w, config.out_h, config.out_fmt, None).unwrap();
            let mut proc = CPUProcessor::new();

            let result = run_bench(&name, WARMUP, ITERATIONS, || {
                proc.convert(&src, &mut dst, Rotation::None, Flip::None, crop)
                    .unwrap();
            });
            result.print_summary_with_throughput(throughput);
            suite.record(&result);
        }

        // HAL G2D (for YUYV/VYUY/NV12 input → RGBA/RGB output only)
        #[cfg(target_os = "linux")]
        if (config.in_fmt == YUYV || config.in_fmt == VYUY || config.in_fmt == NV12)
            && (config.out_fmt == RGBA || config.out_fmt == RGB)
        {
            use edgefirst_image::G2DProcessor;

            let name = format!("letterbox/g2d/{}", config.id());
            let Ok(src) = TensorImage::new(
                config.in_w,
                config.in_h,
                config.in_fmt,
                Some(TensorMemory::Dma),
            ) else {
                println!("  {:50} [skipped: DMA unavailable]", name);
                continue;
            };
            let data = get_test_data(config.in_w, config.in_h, config.in_fmt);
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
            // ManuallyDrop: prevent g2d_close + dlclose("libg2d.so.2") which
            // triggers Vivante galcore atexit heap corruption on i.MX8.
            // Same rationale as the EGL Box::leak in opengl_headless.rs.
            let Ok(proc) = G2DProcessor::new() else {
                println!("  {:50} [skipped: G2D unavailable]", name);
                continue;
            };
            let mut proc = ManuallyDrop::new(proc);

            // Pre-flight: skip if format is not supported by G2D
            if let Err(e) = proc.convert(&src, &mut dst, Rotation::None, Flip::None, crop) {
                println!("  {:50} [skipped: {}]", name, e);
                continue;
            }

            let result = run_bench(&name, WARMUP, ITERATIONS, || {
                proc.convert(&src, &mut dst, Rotation::None, Flip::None, crop)
                    .unwrap();
            });
            result.print_summary_with_throughput(throughput);
            suite.record(&result);
        }

        // HAL OpenGL (for YUYV/VYUY/NV12 → RGBA/RGB/RGB_INT8/PLANAR_RGB_INT8 output)
        #[cfg(all(target_os = "linux", feature = "opengl"))]
        if (config.in_fmt == YUYV || config.in_fmt == VYUY || config.in_fmt == NV12)
            && (config.out_fmt == RGBA
                || config.out_fmt == RGB
                || config.out_fmt == RGB_INT8
                || config.out_fmt == PLANAR_RGB_INT8)
        {
            use edgefirst_image::GLProcessorThreaded;

            let name = format!("letterbox/opengl/{}", config.id());
            let Ok(src) = TensorImage::new(
                config.in_w,
                config.in_h,
                config.in_fmt,
                Some(TensorMemory::Dma),
            ) else {
                println!("  {:50} [skipped: DMA unavailable]", name);
                continue;
            };
            let data = get_test_data(config.in_w, config.in_h, config.in_fmt);
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
            let Ok(mut proc) = GLProcessorThreaded::new(None) else {
                println!("  {:50} [skipped: OpenGL unavailable]", name);
                continue;
            };

            // Pre-flight: skip if format/path is not supported
            if let Err(e) = proc.convert(&src, &mut dst, Rotation::None, Flip::None, crop) {
                println!("  {:50} [skipped: {}]", name, e);
                continue;
            }

            let result = run_bench(&name, WARMUP, ITERATIONS, || {
                proc.convert(&src, &mut dst, Rotation::None, Flip::None, crop)
                    .unwrap();
            });
            result.print_summary_with_throughput(throughput);
            suite.record(&result);
        }

        // OpenCV
        #[cfg(feature = "opencv")]
        bench_letterbox_opencv(config, left, top, new_w, new_h, throughput, suite);
    }
}

#[cfg(feature = "opencv")]
#[allow(clippy::too_many_arguments)]
fn bench_letterbox_opencv(
    config: &BenchConfig,
    left: usize,
    top: usize,
    new_w: usize,
    new_h: usize,
    throughput: u64,
    suite: &mut BenchSuite,
) {
    let color_code = if config.in_fmt == YUYV && config.out_fmt == RGBA {
        imgproc::COLOR_YUV2RGBA_YUYV
    } else if config.in_fmt == YUYV && config.out_fmt == RGB {
        imgproc::COLOR_YUV2RGB_YUYV
    } else {
        return; // Skip NV12 and unsupported combos
    };

    let data = get_test_data(config.in_w, config.in_h, config.in_fmt);
    let cv_type = if config.in_fmt == YUYV {
        CV_8UC2
    } else {
        CV_8UC3
    };
    let channels = if config.in_fmt == YUYV { 2 } else { 3 };

    let src_mat = unsafe {
        Mat::new_rows_cols_with_data_unsafe(
            config.in_h as i32,
            config.in_w as i32,
            cv_type,
            data.as_ptr() as *mut std::ffi::c_void,
            config.in_w * channels,
        )
        .unwrap()
    };

    let out_channels = if config.out_fmt == RGBA { 4 } else { 3 };
    let out_cv_type = if config.out_fmt == RGBA {
        CV_8UC4
    } else {
        CV_8UC3
    };
    let lb_size = Size::new(new_w as i32, new_h as i32);

    // Single-threaded
    {
        let name = format!("letterbox/opencv-1cpu/{}", config.id());
        let mut dst_data = vec![0u8; config.out_w * config.out_h * out_channels];
        let mut dst_mat = unsafe {
            Mat::new_rows_cols_with_data_unsafe(
                config.out_h as i32,
                config.out_w as i32,
                out_cv_type,
                dst_data.as_mut_ptr() as *mut std::ffi::c_void,
                config.out_w * out_channels,
            )
            .unwrap()
        };
        let mut converted = Mat::default();
        let mut resized = Mat::default();

        let result = run_bench(&name, WARMUP, ITERATIONS, || {
            set_num_threads(1).unwrap();
            imgproc::cvt_color_def(&src_mat, &mut converted, color_code).unwrap();
            imgproc::resize(
                &converted,
                &mut resized,
                lb_size,
                0.0,
                0.0,
                imgproc::INTER_LINEAR,
            )
            .unwrap();
            let no_mask = Mat::default();
            dst_mat.set_to(&Scalar::all(114.0), &no_mask).unwrap();
            let roi =
                opencv::core::Rect::new(left as i32, top as i32, lb_size.width, lb_size.height);
            let mut dst_roi = Mat::roi_mut(&mut dst_mat, roi).unwrap();
            resized.copy_to(&mut dst_roi).unwrap();
        });
        result.print_summary_with_throughput(throughput);
        suite.record(&result);
    }

    // Multi-threaded
    {
        let name = format!("letterbox/opencv-multi/{}", config.id());
        let mut dst_data = vec![0u8; config.out_w * config.out_h * out_channels];
        let mut dst_mat = unsafe {
            Mat::new_rows_cols_with_data_unsafe(
                config.out_h as i32,
                config.out_w as i32,
                out_cv_type,
                dst_data.as_mut_ptr() as *mut std::ffi::c_void,
                config.out_w * out_channels,
            )
            .unwrap()
        };
        let mut converted = Mat::default();
        let mut resized = Mat::default();

        let result = run_bench(&name, WARMUP, ITERATIONS, || {
            set_num_threads(0).unwrap();
            imgproc::cvt_color_def(&src_mat, &mut converted, color_code).unwrap();
            imgproc::resize(
                &converted,
                &mut resized,
                lb_size,
                0.0,
                0.0,
                imgproc::INTER_LINEAR,
            )
            .unwrap();
            let no_mask = Mat::default();
            dst_mat.set_to(&Scalar::all(114.0), &no_mask).unwrap();
            let roi =
                opencv::core::Rect::new(left as i32, top as i32, lb_size.width, lb_size.height);
            let mut dst_roi = Mat::roi_mut(&mut dst_mat, roi).unwrap();
            resized.copy_to(&mut dst_roi).unwrap();
        });
        result.print_summary_with_throughput(throughput);
        suite.record(&result);
    }
}

// =============================================================================
// Secondary Benchmarks: Format Conversion (no resize)
// =============================================================================

#[allow(unused_variables)]
fn bench_convert(configs: &[BenchConfig], suite: &mut BenchSuite) {
    println!("\n== Convert: Format Conversion ==\n");

    for config in configs {
        let throughput = config.throughput();

        // HAL CPU
        {
            let name = format!("convert/cpu/{}", config.id());
            let src = TensorImage::new(config.in_w, config.in_h, config.in_fmt, None).unwrap();
            let data = get_test_data(config.in_w, config.in_h, config.in_fmt);
            src.tensor().map().unwrap().as_mut_slice()[..data.len()].copy_from_slice(data);
            let mut dst =
                TensorImage::new(config.out_w, config.out_h, config.out_fmt, None).unwrap();
            let mut proc = CPUProcessor::new();

            let result = run_bench(&name, WARMUP, ITERATIONS, || {
                proc.convert(&src, &mut dst, Rotation::None, Flip::None, Crop::no_crop())
                    .unwrap();
            });
            result.print_summary_with_throughput(throughput);
            suite.record(&result);
        }

        // HAL G2D (for YUYV/VYUY/NV12 input → RGBA/RGB output only)
        #[cfg(target_os = "linux")]
        if (config.in_fmt == YUYV || config.in_fmt == VYUY || config.in_fmt == NV12)
            && (config.out_fmt == RGBA || config.out_fmt == RGB)
        {
            use edgefirst_image::G2DProcessor;

            let name = format!("convert/g2d/{}", config.id());
            let Ok(src) = TensorImage::new(
                config.in_w,
                config.in_h,
                config.in_fmt,
                Some(TensorMemory::Dma),
            ) else {
                println!("  {:50} [skipped: DMA unavailable]", name);
                continue;
            };
            let data = get_test_data(config.in_w, config.in_h, config.in_fmt);
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
            // ManuallyDrop: prevent g2d_close + dlclose("libg2d.so.2") which
            // triggers Vivante galcore atexit heap corruption on i.MX8.
            // Same rationale as the EGL Box::leak in opengl_headless.rs.
            let Ok(proc) = G2DProcessor::new() else {
                println!("  {:50} [skipped: G2D unavailable]", name);
                continue;
            };
            let mut proc = ManuallyDrop::new(proc);

            // Pre-flight: skip if format is not supported by G2D
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

        // HAL OpenGL (for YUYV/VYUY/NV12 → RGBA/RGB/RGB_INT8/PLANAR_RGB_INT8)
        #[cfg(all(target_os = "linux", feature = "opengl"))]
        if (config.in_fmt == YUYV || config.in_fmt == VYUY || config.in_fmt == NV12)
            && (config.out_fmt == RGBA
                || config.out_fmt == RGB
                || config.out_fmt == RGB_INT8
                || config.out_fmt == PLANAR_RGB_INT8)
        {
            use edgefirst_image::GLProcessorThreaded;

            let name = format!("convert/opengl/{}", config.id());
            let Ok(src) = TensorImage::new(
                config.in_w,
                config.in_h,
                config.in_fmt,
                Some(TensorMemory::Dma),
            ) else {
                println!("  {:50} [skipped: DMA unavailable]", name);
                continue;
            };
            let data = get_test_data(config.in_w, config.in_h, config.in_fmt);
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
            let Ok(mut proc) = GLProcessorThreaded::new(None) else {
                println!("  {:50} [skipped: OpenGL unavailable]", name);
                continue;
            };

            // Pre-flight: skip if format/path is not supported
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

        // OpenCV (skip NV12)
        #[cfg(feature = "opencv")]
        bench_convert_opencv(config, throughput, suite);
    }
}

#[cfg(feature = "opencv")]
fn bench_convert_opencv(config: &BenchConfig, throughput: u64, suite: &mut BenchSuite) {
    let color_code = match (config.in_fmt, config.out_fmt) {
        (f1, f2) if f1 == YUYV && f2 == RGBA => imgproc::COLOR_YUV2RGBA_YUYV,
        (f1, f2) if f1 == YUYV && f2 == RGB => imgproc::COLOR_YUV2RGB_YUYV,
        (f1, f2) if f1 == RGB && f2 == RGBA => imgproc::COLOR_RGB2RGBA,
        _ => return,
    };

    let data = get_test_data(config.in_w, config.in_h, config.in_fmt);
    let (cv_type, channels) = if config.in_fmt == YUYV {
        (CV_8UC2, 2)
    } else {
        (CV_8UC3, 3)
    };

    let src_mat = unsafe {
        Mat::new_rows_cols_with_data_unsafe(
            config.in_h as i32,
            config.in_w as i32,
            cv_type,
            data.as_ptr() as *mut std::ffi::c_void,
            config.in_w * channels,
        )
        .unwrap()
    };

    let out_channels = if config.out_fmt == RGBA { 4 } else { 3 };
    let out_cv_type = if config.out_fmt == RGBA {
        CV_8UC4
    } else {
        CV_8UC3
    };

    // Single-threaded
    {
        let name = format!("convert/opencv-1cpu/{}", config.id());
        let mut dst_data = vec![0u8; config.out_w * config.out_h * out_channels];
        let mut dst_mat = unsafe {
            Mat::new_rows_cols_with_data_unsafe(
                config.out_h as i32,
                config.out_w as i32,
                out_cv_type,
                dst_data.as_mut_ptr() as *mut std::ffi::c_void,
                config.out_w * out_channels,
            )
            .unwrap()
        };

        let result = run_bench(&name, WARMUP, ITERATIONS, || {
            set_num_threads(1).unwrap();
            imgproc::cvt_color_def(&src_mat, &mut dst_mat, color_code).unwrap();
        });
        result.print_summary_with_throughput(throughput);
        suite.record(&result);
    }

    // Multi-threaded
    {
        let name = format!("convert/opencv-multi/{}", config.id());
        let mut dst_data = vec![0u8; config.out_w * config.out_h * out_channels];
        let mut dst_mat = unsafe {
            Mat::new_rows_cols_with_data_unsafe(
                config.out_h as i32,
                config.out_w as i32,
                out_cv_type,
                dst_data.as_mut_ptr() as *mut std::ffi::c_void,
                config.out_w * out_channels,
            )
            .unwrap()
        };

        let result = run_bench(&name, WARMUP, ITERATIONS, || {
            set_num_threads(0).unwrap();
            imgproc::cvt_color_def(&src_mat, &mut dst_mat, color_code).unwrap();
        });
        result.print_summary_with_throughput(throughput);
        suite.record(&result);
    }
}

// =============================================================================
// Resize Benchmarks: Same aspect ratio (no letterbox)
// =============================================================================

#[allow(unused_variables)]
fn bench_resize(configs: &[BenchConfig], suite: &mut BenchSuite) {
    println!("\n== Resize: Same Aspect Ratio ==\n");

    for config in configs {
        let throughput = config.throughput();

        // HAL CPU
        {
            let name = format!("resize/cpu/{}", config.id());
            let src = TensorImage::new(config.in_w, config.in_h, config.in_fmt, None).unwrap();
            let data = get_test_data(config.in_w, config.in_h, config.in_fmt);
            src.tensor().map().unwrap().as_mut_slice()[..data.len()].copy_from_slice(data);
            let mut dst =
                TensorImage::new(config.out_w, config.out_h, config.out_fmt, None).unwrap();
            let mut proc = CPUProcessor::new();

            let result = run_bench(&name, WARMUP, ITERATIONS, || {
                proc.convert(&src, &mut dst, Rotation::None, Flip::None, Crop::no_crop())
                    .unwrap();
            });
            result.print_summary_with_throughput(throughput);
            suite.record(&result);
        }

        // HAL G2D
        #[cfg(target_os = "linux")]
        if config.in_fmt == YUYV {
            use edgefirst_image::G2DProcessor;

            let name = format!("resize/g2d/{}", config.id());
            let Ok(src) = TensorImage::new(
                config.in_w,
                config.in_h,
                config.in_fmt,
                Some(TensorMemory::Dma),
            ) else {
                println!("  {:50} [skipped: DMA unavailable]", name);
                continue;
            };
            let data = get_test_data(config.in_w, config.in_h, config.in_fmt);
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
            // ManuallyDrop: prevent g2d_close + dlclose("libg2d.so.2") which
            // triggers Vivante galcore atexit heap corruption on i.MX8.
            // Same rationale as the EGL Box::leak in opengl_headless.rs.
            let Ok(proc) = G2DProcessor::new() else {
                println!("  {:50} [skipped: G2D unavailable]", name);
                continue;
            };
            let mut proc = ManuallyDrop::new(proc);

            let result = run_bench(&name, WARMUP, ITERATIONS, || {
                proc.convert(&src, &mut dst, Rotation::None, Flip::None, Crop::no_crop())
                    .unwrap();
            });
            result.print_summary_with_throughput(throughput);
            suite.record(&result);
        }

        // HAL OpenGL
        #[cfg(all(target_os = "linux", feature = "opengl"))]
        if config.in_fmt == YUYV && config.out_fmt == RGBA {
            use edgefirst_image::GLProcessorThreaded;

            let name = format!("resize/opengl/{}", config.id());
            let Ok(src) = TensorImage::new(
                config.in_w,
                config.in_h,
                config.in_fmt,
                Some(TensorMemory::Dma),
            ) else {
                println!("  {:50} [skipped: DMA unavailable]", name);
                continue;
            };
            let data = get_test_data(config.in_w, config.in_h, config.in_fmt);
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
            let Ok(mut proc) = GLProcessorThreaded::new(None) else {
                println!("  {:50} [skipped: OpenGL unavailable]", name);
                continue;
            };

            // Pre-flight: skip if format/path is not supported
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

        // OpenCV
        #[cfg(feature = "opencv")]
        bench_resize_opencv(config, throughput, suite);
    }
}

#[cfg(feature = "opencv")]
fn bench_resize_opencv(config: &BenchConfig, throughput: u64, suite: &mut BenchSuite) {
    let data = get_test_data(config.in_w, config.in_h, config.in_fmt);
    let src_mat = unsafe {
        Mat::new_rows_cols_with_data_unsafe(
            config.in_h as i32,
            config.in_w as i32,
            CV_8UC2,
            data.as_ptr() as *mut std::ffi::c_void,
            config.in_w * 2,
        )
        .unwrap()
    };

    let out_channels = if config.out_fmt == RGBA { 4 } else { 3 };
    let out_cv_type = if config.out_fmt == RGBA {
        CV_8UC4
    } else {
        CV_8UC3
    };
    let color_code = if config.out_fmt == RGBA {
        imgproc::COLOR_YUV2RGBA_YUYV
    } else {
        imgproc::COLOR_YUV2RGB_YUYV
    };
    let target_size = Size::new(config.out_w as i32, config.out_h as i32);

    // Single-threaded
    {
        let name = format!("resize/opencv-1cpu/{}", config.id());
        let mut dst_data = vec![0u8; config.out_w * config.out_h * out_channels];
        let mut dst_mat = unsafe {
            Mat::new_rows_cols_with_data_unsafe(
                config.out_h as i32,
                config.out_w as i32,
                out_cv_type,
                dst_data.as_mut_ptr() as *mut std::ffi::c_void,
                config.out_w * out_channels,
            )
            .unwrap()
        };
        let mut converted = Mat::default();
        let mut resized = Mat::default();

        let result = run_bench(&name, WARMUP, ITERATIONS, || {
            set_num_threads(1).unwrap();
            imgproc::cvt_color_def(&src_mat, &mut converted, color_code).unwrap();
            imgproc::resize(
                &converted,
                &mut resized,
                target_size,
                0.0,
                0.0,
                imgproc::INTER_LINEAR,
            )
            .unwrap();
            resized.copy_to(&mut dst_mat).unwrap();
        });
        result.print_summary_with_throughput(throughput);
        suite.record(&result);
    }

    // Multi-threaded
    {
        let name = format!("resize/opencv-multi/{}", config.id());
        let mut dst_data = vec![0u8; config.out_w * config.out_h * out_channels];
        let mut dst_mat = unsafe {
            Mat::new_rows_cols_with_data_unsafe(
                config.out_h as i32,
                config.out_w as i32,
                out_cv_type,
                dst_data.as_mut_ptr() as *mut std::ffi::c_void,
                config.out_w * out_channels,
            )
            .unwrap()
        };
        let mut converted = Mat::default();
        let mut resized = Mat::default();

        let result = run_bench(&name, WARMUP, ITERATIONS, || {
            set_num_threads(0).unwrap();
            imgproc::cvt_color_def(&src_mat, &mut converted, color_code).unwrap();
            imgproc::resize(
                &converted,
                &mut resized,
                target_size,
                0.0,
                0.0,
                imgproc::INTER_LINEAR,
            )
            .unwrap();
            resized.copy_to(&mut dst_mat).unwrap();
        });
        result.print_summary_with_throughput(throughput);
        suite.record(&result);
    }
}

// =============================================================================
// Letterbox Pipeline Benchmarks: Realistic camera→model with clear+resize
// =============================================================================

#[allow(unused_variables)]
fn bench_letterbox_pipeline(suite: &mut BenchSuite) {
    println!("\n== Letterbox Pipeline: Realistic Camera→Model ==\n");

    let pipeline_configs: &[(usize, usize, usize, usize)] = &[
        (1920, 1080, 640, 640), // 1080p → YOLO standard
        (3840, 2160, 640, 640), // 4K → YOLO standard
    ];

    let src_formats: &[(&str, four_char_code::FourCharCode)] = &[("YUYV", YUYV), ("RGBA", RGBA)];

    for &(src_w, src_h, dst_w, dst_h) in pipeline_configs {
        let scale = f64::min(dst_w as f64 / src_w as f64, dst_h as f64 / src_h as f64);
        let content_w = (src_w as f64 * scale) as usize;
        let content_h = (src_h as f64 * scale) as usize;
        let pad_left = (dst_w - content_w) / 2;
        let pad_top = (dst_h - content_h) / 2;
        let dst_rect = Rect::new(pad_left, pad_top, content_w, content_h);
        let color: [u8; 4] = [114, 114, 114, 255];
        let crop = Crop::new()
            .with_dst_rect(Some(dst_rect))
            .with_dst_color(Some(color));

        // Throughput based on YUYV input (2 bytes/pixel)
        let throughput = (src_w * src_h * 2) as u64;

        for &(fmt_name, src_fmt) in src_formats {
            // CPU pipeline
            {
                let name = format!(
                    "pipeline/cpu/{}x{}->{}x{}/{}",
                    src_w, src_h, dst_w, dst_h, fmt_name
                );
                let src = TensorImage::new(src_w, src_h, src_fmt, None).unwrap();
                src.tensor().map().unwrap().as_mut_slice().fill(128);
                let mut dst = TensorImage::new(dst_w, dst_h, RGBA, None).unwrap();
                let mut cpu = CPUProcessor::new();

                let result = run_bench(&name, WARMUP, ITERATIONS, || {
                    cpu.convert(&src, &mut dst, Rotation::None, Flip::None, crop)
                        .unwrap();
                });
                result.print_summary_with_throughput(throughput);
                suite.record(&result);
            }

            // G2D pipeline
            #[cfg(target_os = "linux")]
            {
                use edgefirst_image::G2DProcessor;

                let name = format!(
                    "pipeline/g2d/{}x{}->{}x{}/{}",
                    src_w, src_h, dst_w, dst_h, fmt_name
                );
                let Ok(src) = TensorImage::new(src_w, src_h, src_fmt, Some(TensorMemory::Dma))
                else {
                    println!("  {:50} [skipped: DMA unavailable]", name);
                    continue;
                };
                src.tensor().map().unwrap().as_mut_slice().fill(128);
                let Ok(mut dst) = TensorImage::new(dst_w, dst_h, RGBA, Some(TensorMemory::Dma))
                else {
                    println!("  {:50} [skipped: DMA unavailable]", name);
                    continue;
                };
                // ManuallyDrop: prevent g2d_close + dlclose("libg2d.so.2") which
                // triggers Vivante galcore atexit heap corruption on i.MX8.
                // Same rationale as the EGL Box::leak in opengl_headless.rs.
                let Ok(proc) = G2DProcessor::new() else {
                    println!("  {:50} [skipped: G2D unavailable]", name);
                    continue;
                };
                let mut proc = ManuallyDrop::new(proc);

                let result = run_bench(&name, WARMUP, ITERATIONS, || {
                    proc.convert(&src, &mut dst, Rotation::None, Flip::None, crop)
                        .unwrap();
                });
                result.print_summary_with_throughput(throughput);
                suite.record(&result);
            }
        }
    }
}

// =============================================================================
// Main
// =============================================================================

fn main() {
    // Parse --bench flag (Criterion compat for `cargo bench`)
    // Just consume the flag silently — we always run benchmarks.
    let mut suite = BenchSuite::from_args();

    println!("Pipeline Benchmark — custom in-process harness (no fork)");
    println!("  warmup={WARMUP}  iterations={ITERATIONS}");

    // --- Letterbox configs ---
    let letterbox_configs = vec![
        // 1080p camera → YOLO standard (640x640)
        BenchConfig::new(1920, 1080, 640, 640, YUYV, RGBA),
        BenchConfig::new(1920, 1080, 640, 640, YUYV, RGB),
        BenchConfig::new(1920, 1080, 640, 640, YUYV, RGB_INT8),
        BenchConfig::new(1920, 1080, 640, 640, YUYV, PLANAR_RGB_INT8),
        BenchConfig::new(1920, 1080, 640, 640, VYUY, RGBA),
        BenchConfig::new(1920, 1080, 640, 640, VYUY, RGB),
        BenchConfig::new(1920, 1080, 640, 640, NV12, RGBA),
        // 4K camera → YOLO standard (640x640)
        BenchConfig::new(3840, 2160, 640, 640, YUYV, RGBA),
        BenchConfig::new(3840, 2160, 640, 640, YUYV, RGB),
        BenchConfig::new(3840, 2160, 640, 640, YUYV, RGB_INT8),
        BenchConfig::new(3840, 2160, 640, 640, NV12, RGBA),
        // 4K camera → YOLO hi-res (1280x1280)
        BenchConfig::new(3840, 2160, 1280, 1280, YUYV, RGBA),
        BenchConfig::new(3840, 2160, 1280, 1280, YUYV, RGB),
        BenchConfig::new(3840, 2160, 1280, 1280, YUYV, RGB_INT8),
        BenchConfig::new(3840, 2160, 1280, 1280, NV12, RGBA),
    ];
    bench_letterbox(&letterbox_configs, &mut suite);

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
    ];
    bench_convert(&convert_configs, &mut suite);

    // --- Resize configs ---
    let resize_configs = vec![
        // 4K → 1080p
        BenchConfig::new(3840, 2160, 1920, 1080, YUYV, RGBA),
        BenchConfig::new(3840, 2160, 1920, 1080, YUYV, RGB),
        // 4K → 720p
        BenchConfig::new(3840, 2160, 1280, 720, YUYV, RGBA),
        // 1080p → 720p
        BenchConfig::new(1920, 1080, 1280, 720, YUYV, RGBA),
    ];
    bench_resize(&resize_configs, &mut suite);

    // --- Letterbox pipeline ---
    bench_letterbox_pipeline(&mut suite);

    suite.finish();
    println!("\nDone.");
}
