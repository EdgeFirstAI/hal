// SPDX-FileCopyrightText: Copyright 2025 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! Vision pipeline benchmarks using Criterion.
//!
//! Focused benchmarks for realistic camera-to-model preprocessing scenarios:
//! - Primary: Camera (YUYV) → Model (RGBA/RGB) with letterbox
//! - Secondary: Format conversion (no resize)
//!
//! ## Run benchmarks (host)
//! ```bash
//! cargo bench -p edgefirst-image --bench pipeline_benchmark
//! ```
//!
//! ## Run benchmarks (on-target, cross-compiled)
//! ```bash
//! # The --bench flag is required for Criterion to run actual measurements
//! ./pipeline_benchmark --bench
//! ```
//!
//! ## View HTML report
//! Open `target/criterion/report/index.html` after running benchmarks.

mod common;

#[cfg(target_os = "linux")]
use common::g2d_available;
#[cfg(all(target_os = "linux", feature = "opengl"))]
use common::opengl_available;
use common::{calculate_letterbox, get_test_data, BenchConfig};
use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};

#[cfg(target_os = "linux")]
use edgefirst_image::G2DProcessor;
#[cfg(all(target_os = "linux", feature = "opengl"))]
use edgefirst_image::GLProcessorThreaded;
use edgefirst_image::{
    CPUProcessor, Crop, Flip, ImageProcessorTrait, Rect, Rotation, TensorImage, NV12, RGB, RGBA,
    YUYV,
};
#[cfg(target_os = "linux")]
use edgefirst_tensor::TensorMemory;
use edgefirst_tensor::{TensorMapTrait, TensorTrait};

#[cfg(feature = "opencv")]
use opencv::{
    core::{set_num_threads, Mat, Scalar, Size, CV_8UC2, CV_8UC3, CV_8UC4},
    imgproc,
    prelude::*,
};

// =============================================================================
// Primary Benchmarks: Camera → Model (letterbox)
// =============================================================================

/// Primary use case: Camera YUYV → Model RGBA with letterbox
#[allow(unused_variables)] // has_g2d/has_opengl used conditionally on Linux
fn bench_letterbox(c: &mut Criterion) {
    let mut group = c.benchmark_group("letterbox");
    group.sample_size(100);

    // Check hardware availability once at start (not per-iteration)
    #[cfg(target_os = "linux")]
    let has_g2d = g2d_available();
    #[cfg(not(target_os = "linux"))]
    let has_g2d = false;

    #[cfg(all(target_os = "linux", feature = "opengl"))]
    let has_opengl = opengl_available();
    #[cfg(not(all(target_os = "linux", feature = "opengl")))]
    let has_opengl = false;

    // Configurations: most valuable for YOLO model preprocessing
    let configs = vec![
        // 1080p camera → YOLO standard (640x640)
        BenchConfig {
            in_w: 1920,
            in_h: 1080,
            out_w: 640,
            out_h: 640,
            in_fmt: YUYV,
            out_fmt: RGBA,
        },
        BenchConfig {
            in_w: 1920,
            in_h: 1080,
            out_w: 640,
            out_h: 640,
            in_fmt: YUYV,
            out_fmt: RGB,
        },
        BenchConfig {
            in_w: 1920,
            in_h: 1080,
            out_w: 640,
            out_h: 640,
            in_fmt: NV12,
            out_fmt: RGBA,
        },
        // 4K camera → YOLO standard (640x640)
        BenchConfig {
            in_w: 3840,
            in_h: 2160,
            out_w: 640,
            out_h: 640,
            in_fmt: YUYV,
            out_fmt: RGBA,
        },
        BenchConfig {
            in_w: 3840,
            in_h: 2160,
            out_w: 640,
            out_h: 640,
            in_fmt: YUYV,
            out_fmt: RGB,
        },
        BenchConfig {
            in_w: 3840,
            in_h: 2160,
            out_w: 640,
            out_h: 640,
            in_fmt: NV12,
            out_fmt: RGBA,
        },
        // 4K camera → YOLO hi-res (1280x1280)
        BenchConfig {
            in_w: 3840,
            in_h: 2160,
            out_w: 1280,
            out_h: 1280,
            in_fmt: YUYV,
            out_fmt: RGBA,
        },
        BenchConfig {
            in_w: 3840,
            in_h: 2160,
            out_w: 1280,
            out_h: 1280,
            in_fmt: YUYV,
            out_fmt: RGB,
        },
        BenchConfig {
            in_w: 3840,
            in_h: 2160,
            out_w: 1280,
            out_h: 1280,
            in_fmt: NV12,
            out_fmt: RGBA,
        },
    ];

    for config in &configs {
        group.throughput(config.throughput());

        // HAL CPU
        {
            let src = TensorImage::new(config.in_w, config.in_h, config.in_fmt, None).unwrap();
            let data = get_test_data(config.in_w, config.in_h, config.in_fmt);
            src.tensor().map().unwrap().as_mut_slice()[..data.len()].copy_from_slice(data);
            let mut dst =
                TensorImage::new(config.out_w, config.out_h, config.out_fmt, None).unwrap();
            let mut proc = CPUProcessor::new();

            let (left, top, new_w, new_h) =
                calculate_letterbox(config.in_w, config.in_h, config.out_w, config.out_h);
            let crop = Crop::new()
                .with_dst_rect(Some(Rect::new(left, top, new_w, new_h)))
                .with_dst_color(Some([114, 114, 114, 255]));

            group.bench_with_input(BenchmarkId::new("cpu", config.id()), &config, |b, _| {
                b.iter(|| {
                    proc.convert(&src, &mut dst, Rotation::None, Flip::None, crop)
                        .unwrap();
                    black_box(&dst);
                });
            });
        }

        // HAL G2D (for YUYV and NV12 input)
        #[cfg(target_os = "linux")]
        if has_g2d && (config.in_fmt == YUYV || config.in_fmt == NV12) {
            let Ok(src) = TensorImage::new(
                config.in_w,
                config.in_h,
                config.in_fmt,
                Some(TensorMemory::Dma),
            ) else {
                eprintln!(
                    "G2D: DMA allocation failed for {}x{} - skipping",
                    config.in_w, config.in_h
                );
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
                continue;
            };
            let mut proc = G2DProcessor::new().unwrap();

            let (left, top, new_w, new_h) =
                calculate_letterbox(config.in_w, config.in_h, config.out_w, config.out_h);
            let crop = Crop::new()
                .with_dst_rect(Some(Rect::new(left, top, new_w, new_h)))
                .with_dst_color(Some([114, 114, 114, 255]));

            group.bench_with_input(BenchmarkId::new("g2d", config.id()), &config, |b, _| {
                b.iter(|| {
                    proc.convert(&src, &mut dst, Rotation::None, Flip::None, crop)
                        .unwrap();
                    black_box(&dst);
                });
            });
        }

        // HAL OpenGL (for YUYV/NV12 → RGBA output)
        // Resources created inside closure to survive Criterion's subprocess fork
        #[cfg(all(target_os = "linux", feature = "opengl"))]
        if has_opengl && (config.in_fmt == YUYV || config.in_fmt == NV12) && config.out_fmt == RGBA
        {
            let config = config.clone();
            group.bench_with_input(
                BenchmarkId::new("opengl", config.id()),
                &config,
                |b, config| {
                    // Create resources in the child subprocess after fork
                    let src = TensorImage::new(
                        config.in_w,
                        config.in_h,
                        config.in_fmt,
                        Some(TensorMemory::Dma),
                    )
                    .unwrap();
                    let data = get_test_data(config.in_w, config.in_h, config.in_fmt);
                    src.tensor().map().unwrap().as_mut_slice()[..data.len()].copy_from_slice(data);
                    let mut dst = TensorImage::new(
                        config.out_w,
                        config.out_h,
                        config.out_fmt,
                        Some(TensorMemory::Dma),
                    )
                    .unwrap();
                    let mut proc = GLProcessorThreaded::new().unwrap();

                    let (left, top, new_w, new_h) =
                        calculate_letterbox(config.in_w, config.in_h, config.out_w, config.out_h);
                    let crop = Crop::new()
                        .with_dst_rect(Some(Rect::new(left, top, new_w, new_h)))
                        .with_dst_color(Some([114, 114, 114, 255]));

                    // Warmup for OpenGL shader compilation (not measured)
                    let _ = proc.convert(&src, &mut dst, Rotation::None, Flip::None, crop);

                    b.iter(|| {
                        proc.convert(&src, &mut dst, Rotation::None, Flip::None, crop)
                            .unwrap();
                        black_box(&dst);
                    });
                },
            );
        }

        // OpenCV
        #[cfg(feature = "opencv")]
        {
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

            let color_code = if config.in_fmt == YUYV && config.out_fmt == RGBA {
                imgproc::COLOR_YUV2RGBA_YUYV
            } else if config.in_fmt == YUYV && config.out_fmt == RGB {
                imgproc::COLOR_YUV2RGB_YUYV
            } else {
                -1
            };

            let (left, top, new_w, new_h) =
                calculate_letterbox(config.in_w, config.in_h, config.out_w, config.out_h);
            let lb_size = Size::new(new_w as i32, new_h as i32);

            // Single-threaded OpenCV benchmark
            group.bench_with_input(
                BenchmarkId::new("opencv-1cpu", config.id()),
                &config,
                |b, _| {
                    set_num_threads(1).unwrap();
                    b.iter(|| {
                        // Color conversion
                        if color_code >= 0 {
                            imgproc::cvt_color_def(&src_mat, &mut converted, color_code).unwrap();
                        }
                        let working = if color_code >= 0 {
                            &converted
                        } else {
                            &src_mat
                        };

                        // Resize with letterbox
                        imgproc::resize(
                            working,
                            &mut resized,
                            lb_size,
                            0.0,
                            0.0,
                            imgproc::INTER_LINEAR,
                        )
                        .unwrap();

                        // Fill background and copy
                        let no_mask = Mat::default();
                        dst_mat.set_to(&Scalar::all(114.0), &no_mask).unwrap();
                        let roi = opencv::core::Rect::new(
                            left as i32,
                            top as i32,
                            lb_size.width,
                            lb_size.height,
                        );
                        let mut dst_roi = Mat::roi_mut(&mut dst_mat, roi).unwrap();
                        resized.copy_to(&mut dst_roi).unwrap();

                        black_box(&dst_data);
                    });
                },
            );

            // Multi-threaded OpenCV benchmark (auto thread count)
            group.bench_with_input(
                BenchmarkId::new("opencv-multi", config.id()),
                &config,
                |b, _| {
                    set_num_threads(0).unwrap(); // 0 = auto (use all available cores)
                    b.iter(|| {
                        // Color conversion
                        if color_code >= 0 {
                            imgproc::cvt_color_def(&src_mat, &mut converted, color_code).unwrap();
                        }
                        let working = if color_code >= 0 {
                            &converted
                        } else {
                            &src_mat
                        };

                        // Resize with letterbox
                        imgproc::resize(
                            working,
                            &mut resized,
                            lb_size,
                            0.0,
                            0.0,
                            imgproc::INTER_LINEAR,
                        )
                        .unwrap();

                        // Fill background and copy
                        let no_mask = Mat::default();
                        dst_mat.set_to(&Scalar::all(114.0), &no_mask).unwrap();
                        let roi = opencv::core::Rect::new(
                            left as i32,
                            top as i32,
                            lb_size.width,
                            lb_size.height,
                        );
                        let mut dst_roi = Mat::roi_mut(&mut dst_mat, roi).unwrap();
                        resized.copy_to(&mut dst_roi).unwrap();

                        black_box(&dst_data);
                    });
                },
            );
        }
    }

    group.finish();
}

// =============================================================================
// Secondary Benchmarks: Format Conversion (no resize)
// =============================================================================

/// Format conversion without resize (for ISP output scenarios)
#[allow(unused_variables)] // has_g2d/has_opengl used conditionally on Linux
fn bench_convert(c: &mut Criterion) {
    let mut group = c.benchmark_group("convert");
    group.sample_size(100);

    // Check hardware availability once at start
    #[cfg(target_os = "linux")]
    let has_g2d = g2d_available();
    #[cfg(not(target_os = "linux"))]
    let has_g2d = false;

    #[cfg(all(target_os = "linux", feature = "opengl"))]
    let has_opengl = opengl_available();
    #[cfg(not(all(target_os = "linux", feature = "opengl")))]
    let has_opengl = false;

    // Configurations: format conversion at camera resolution
    let configs = vec![
        // 1080p conversions
        BenchConfig {
            in_w: 1920,
            in_h: 1080,
            out_w: 1920,
            out_h: 1080,
            in_fmt: YUYV,
            out_fmt: RGBA,
        },
        BenchConfig {
            in_w: 1920,
            in_h: 1080,
            out_w: 1920,
            out_h: 1080,
            in_fmt: YUYV,
            out_fmt: RGB,
        },
        BenchConfig {
            in_w: 1920,
            in_h: 1080,
            out_w: 1920,
            out_h: 1080,
            in_fmt: NV12,
            out_fmt: RGBA,
        },
        BenchConfig {
            in_w: 1920,
            in_h: 1080,
            out_w: 1920,
            out_h: 1080,
            in_fmt: NV12,
            out_fmt: RGB,
        },
        BenchConfig {
            in_w: 1920,
            in_h: 1080,
            out_w: 1920,
            out_h: 1080,
            in_fmt: RGB,
            out_fmt: RGBA,
        },
        // 4K conversions
        BenchConfig {
            in_w: 3840,
            in_h: 2160,
            out_w: 3840,
            out_h: 2160,
            in_fmt: YUYV,
            out_fmt: RGBA,
        },
        BenchConfig {
            in_w: 3840,
            in_h: 2160,
            out_w: 3840,
            out_h: 2160,
            in_fmt: YUYV,
            out_fmt: RGB,
        },
        BenchConfig {
            in_w: 3840,
            in_h: 2160,
            out_w: 3840,
            out_h: 2160,
            in_fmt: NV12,
            out_fmt: RGBA,
        },
        BenchConfig {
            in_w: 3840,
            in_h: 2160,
            out_w: 3840,
            out_h: 2160,
            in_fmt: NV12,
            out_fmt: RGB,
        },
        BenchConfig {
            in_w: 3840,
            in_h: 2160,
            out_w: 3840,
            out_h: 2160,
            in_fmt: RGB,
            out_fmt: RGBA,
        },
    ];

    for config in &configs {
        group.throughput(config.throughput());

        // HAL CPU
        {
            let src = TensorImage::new(config.in_w, config.in_h, config.in_fmt, None).unwrap();
            let data = get_test_data(config.in_w, config.in_h, config.in_fmt);
            src.tensor().map().unwrap().as_mut_slice()[..data.len()].copy_from_slice(data);
            let mut dst =
                TensorImage::new(config.out_w, config.out_h, config.out_fmt, None).unwrap();
            let mut proc = CPUProcessor::new();

            group.bench_with_input(BenchmarkId::new("cpu", config.id()), &config, |b, _| {
                b.iter(|| {
                    proc.convert(&src, &mut dst, Rotation::None, Flip::None, Crop::no_crop())
                        .unwrap();
                    black_box(&dst);
                });
            });
        }

        // HAL G2D (for YUYV and NV12 input)
        #[cfg(target_os = "linux")]
        if has_g2d && (config.in_fmt == YUYV || config.in_fmt == NV12) {
            let Ok(src) = TensorImage::new(
                config.in_w,
                config.in_h,
                config.in_fmt,
                Some(TensorMemory::Dma),
            ) else {
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
                continue;
            };
            let mut proc = G2DProcessor::new().unwrap();

            group.bench_with_input(BenchmarkId::new("g2d", config.id()), &config, |b, _| {
                b.iter(|| {
                    proc.convert(&src, &mut dst, Rotation::None, Flip::None, Crop::no_crop())
                        .unwrap();
                    black_box(&dst);
                });
            });
        }

        // HAL OpenGL (for YUYV→RGBA and NV12→RGBA)
        // Resources created inside closure to survive Criterion's subprocess fork
        #[cfg(all(target_os = "linux", feature = "opengl"))]
        if has_opengl && (config.in_fmt == YUYV || config.in_fmt == NV12) && config.out_fmt == RGBA
        {
            let config = config.clone();
            group.bench_with_input(
                BenchmarkId::new("opengl", config.id()),
                &config,
                |b, config| {
                    let src = TensorImage::new(
                        config.in_w,
                        config.in_h,
                        config.in_fmt,
                        Some(TensorMemory::Dma),
                    )
                    .unwrap();
                    let data = get_test_data(config.in_w, config.in_h, config.in_fmt);
                    src.tensor().map().unwrap().as_mut_slice()[..data.len()].copy_from_slice(data);
                    let mut dst = TensorImage::new(
                        config.out_w,
                        config.out_h,
                        config.out_fmt,
                        Some(TensorMemory::Dma),
                    )
                    .unwrap();
                    let mut proc = GLProcessorThreaded::new().unwrap();

                    // Warmup (not measured)
                    let _ =
                        proc.convert(&src, &mut dst, Rotation::None, Flip::None, Crop::no_crop());

                    b.iter(|| {
                        proc.convert(&src, &mut dst, Rotation::None, Flip::None, Crop::no_crop())
                            .unwrap();
                        black_box(&dst);
                    });
                },
            );
        }

        // OpenCV (skip NV12 - complex handling)
        #[cfg(feature = "opencv")]
        if config.in_fmt != NV12 {
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

            let color_code = match (config.in_fmt, config.out_fmt) {
                (f1, f2) if f1 == YUYV && f2 == RGBA => imgproc::COLOR_YUV2RGBA_YUYV,
                (f1, f2) if f1 == YUYV && f2 == RGB => imgproc::COLOR_YUV2RGB_YUYV,
                (f1, f2) if f1 == RGB && f2 == RGBA => imgproc::COLOR_RGB2RGBA,
                _ => continue,
            };

            // Single-threaded OpenCV benchmark
            group.bench_with_input(
                BenchmarkId::new("opencv-1cpu", config.id()),
                &config,
                |b, _| {
                    set_num_threads(1).unwrap();
                    b.iter(|| {
                        imgproc::cvt_color_def(&src_mat, &mut dst_mat, color_code).unwrap();
                        black_box(&dst_data);
                    });
                },
            );

            // Multi-threaded OpenCV benchmark
            group.bench_with_input(
                BenchmarkId::new("opencv-multi", config.id()),
                &config,
                |b, _| {
                    set_num_threads(0).unwrap();
                    b.iter(|| {
                        imgproc::cvt_color_def(&src_mat, &mut dst_mat, color_code).unwrap();
                        black_box(&dst_data);
                    });
                },
            );
        }
    }

    group.finish();
}

// =============================================================================
// Resize Benchmarks: Same aspect ratio (no letterbox)
// =============================================================================

/// Downscale with same aspect ratio (16:9 → 16:9)
#[allow(unused_variables)] // has_g2d/has_opengl used conditionally on Linux
fn bench_resize(c: &mut Criterion) {
    let mut group = c.benchmark_group("resize");
    group.sample_size(100);

    // Check hardware availability once at start
    #[cfg(target_os = "linux")]
    let has_g2d = g2d_available();
    #[cfg(not(target_os = "linux"))]
    let has_g2d = false;

    #[cfg(all(target_os = "linux", feature = "opengl"))]
    let has_opengl = opengl_available();
    #[cfg(not(all(target_os = "linux", feature = "opengl")))]
    let has_opengl = false;

    // Configurations: 16:9 → 16:9 downscale
    let configs = vec![
        // 4K → 1080p (common display scaling)
        BenchConfig {
            in_w: 3840,
            in_h: 2160,
            out_w: 1920,
            out_h: 1080,
            in_fmt: YUYV,
            out_fmt: RGBA,
        },
        BenchConfig {
            in_w: 3840,
            in_h: 2160,
            out_w: 1920,
            out_h: 1080,
            in_fmt: YUYV,
            out_fmt: RGB,
        },
        // 4K → 720p
        BenchConfig {
            in_w: 3840,
            in_h: 2160,
            out_w: 1280,
            out_h: 720,
            in_fmt: YUYV,
            out_fmt: RGBA,
        },
        // 1080p → 720p
        BenchConfig {
            in_w: 1920,
            in_h: 1080,
            out_w: 1280,
            out_h: 720,
            in_fmt: YUYV,
            out_fmt: RGBA,
        },
    ];

    for config in &configs {
        group.throughput(config.throughput());

        // HAL CPU
        {
            let src = TensorImage::new(config.in_w, config.in_h, config.in_fmt, None).unwrap();
            let data = get_test_data(config.in_w, config.in_h, config.in_fmt);
            src.tensor().map().unwrap().as_mut_slice()[..data.len()].copy_from_slice(data);
            let mut dst =
                TensorImage::new(config.out_w, config.out_h, config.out_fmt, None).unwrap();
            let mut proc = CPUProcessor::new();

            group.bench_with_input(BenchmarkId::new("cpu", config.id()), &config, |b, _| {
                b.iter(|| {
                    proc.convert(&src, &mut dst, Rotation::None, Flip::None, Crop::no_crop())
                        .unwrap();
                    black_box(&dst);
                });
            });
        }

        // HAL G2D
        #[cfg(target_os = "linux")]
        if has_g2d && config.in_fmt == YUYV {
            let Ok(src) = TensorImage::new(
                config.in_w,
                config.in_h,
                config.in_fmt,
                Some(TensorMemory::Dma),
            ) else {
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
                continue;
            };
            let mut proc = G2DProcessor::new().unwrap();

            group.bench_with_input(BenchmarkId::new("g2d", config.id()), &config, |b, _| {
                b.iter(|| {
                    proc.convert(&src, &mut dst, Rotation::None, Flip::None, Crop::no_crop())
                        .unwrap();
                    black_box(&dst);
                });
            });
        }

        // HAL OpenGL
        // Resources created inside closure to survive Criterion's subprocess fork
        #[cfg(all(target_os = "linux", feature = "opengl"))]
        if has_opengl && config.in_fmt == YUYV && config.out_fmt == RGBA {
            let config = config.clone();
            group.bench_with_input(
                BenchmarkId::new("opengl", config.id()),
                &config,
                |b, config| {
                    let src = TensorImage::new(
                        config.in_w,
                        config.in_h,
                        config.in_fmt,
                        Some(TensorMemory::Dma),
                    )
                    .unwrap();
                    let data = get_test_data(config.in_w, config.in_h, config.in_fmt);
                    src.tensor().map().unwrap().as_mut_slice()[..data.len()].copy_from_slice(data);
                    let mut dst = TensorImage::new(
                        config.out_w,
                        config.out_h,
                        config.out_fmt,
                        Some(TensorMemory::Dma),
                    )
                    .unwrap();
                    let mut proc = GLProcessorThreaded::new().unwrap();

                    // Warmup (not measured)
                    let _ =
                        proc.convert(&src, &mut dst, Rotation::None, Flip::None, Crop::no_crop());

                    b.iter(|| {
                        proc.convert(&src, &mut dst, Rotation::None, Flip::None, Crop::no_crop())
                            .unwrap();
                        black_box(&dst);
                    });
                },
            );
        }

        // OpenCV
        #[cfg(feature = "opencv")]
        {
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
            let color_code = if config.out_fmt == RGBA {
                imgproc::COLOR_YUV2RGBA_YUYV
            } else {
                imgproc::COLOR_YUV2RGB_YUYV
            };
            let target_size = Size::new(config.out_w as i32, config.out_h as i32);

            // Single-threaded OpenCV benchmark
            group.bench_with_input(
                BenchmarkId::new("opencv-1cpu", config.id()),
                &config,
                |b, _| {
                    set_num_threads(1).unwrap();
                    b.iter(|| {
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
                        black_box(&dst_data);
                    });
                },
            );

            // Multi-threaded OpenCV benchmark
            group.bench_with_input(
                BenchmarkId::new("opencv-multi", config.id()),
                &config,
                |b, _| {
                    set_num_threads(0).unwrap();
                    b.iter(|| {
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
                        black_box(&dst_data);
                    });
                },
            );
        }
    }

    group.finish();
}

criterion_group!(benches, bench_letterbox, bench_convert, bench_resize);
criterion_main!(benches);
