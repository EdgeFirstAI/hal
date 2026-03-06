// SPDX-FileCopyrightText: Copyright 2025-2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! OpenCV comparison benchmarks using the edgefirst-bench harness.
//!
//! Tests the same letterbox, convert, and resize operations as
//! pipeline_benchmark but using OpenCV for cross-library comparison.
//!
//! Requires the `opencv` feature flag:
//! ```bash
//! cargo bench -p edgefirst-image --bench opencv_benchmark --features opencv -- --bench
//! ```

mod common;

use common::{calculate_letterbox, get_test_data, run_bench, BenchConfig, BenchSuite};
use edgefirst_image::{RGB, RGBA, YUYV};

use opencv::{
    core::{set_num_threads, Mat, Scalar, Size, CV_8UC2, CV_8UC3, CV_8UC4},
    imgproc,
    prelude::*,
};

const WARMUP: usize = 10;
const ITERATIONS: usize = 100;

// =============================================================================
// Letterbox: OpenCV
// =============================================================================

fn bench_letterbox_opencv(config: &BenchConfig, suite: &mut BenchSuite) {
    let color_code = if config.in_fmt == YUYV && config.out_fmt == RGBA {
        imgproc::COLOR_YUV2RGBA_YUYV
    } else if config.in_fmt == YUYV && config.out_fmt == RGB {
        imgproc::COLOR_YUV2RGB_YUYV
    } else {
        return; // Skip NV12 and unsupported combos
    };

    let (left, top, new_w, new_h) =
        calculate_letterbox(config.in_w, config.in_h, config.out_w, config.out_h);
    let throughput = config.throughput();
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
// Convert: OpenCV
// =============================================================================

fn bench_convert_opencv(config: &BenchConfig, suite: &mut BenchSuite) {
    let color_code = match (config.in_fmt, config.out_fmt) {
        (f1, f2) if f1 == YUYV && f2 == RGBA => imgproc::COLOR_YUV2RGBA_YUYV,
        (f1, f2) if f1 == YUYV && f2 == RGB => imgproc::COLOR_YUV2RGB_YUYV,
        (f1, f2) if f1 == RGB && f2 == RGBA => imgproc::COLOR_RGB2RGBA,
        _ => return,
    };

    let throughput = config.throughput();
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
// Resize: OpenCV
// =============================================================================

fn bench_resize_opencv(config: &BenchConfig, suite: &mut BenchSuite) {
    let throughput = config.throughput();
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
// Main
// =============================================================================

fn main() {
    let mut suite = BenchSuite::from_args();

    println!("OpenCV Benchmark — edgefirst-bench harness");
    println!("  warmup={WARMUP}  iterations={ITERATIONS}");

    let letterbox_configs = vec![
        BenchConfig::new(1920, 1080, 640, 640, YUYV, RGBA),
        BenchConfig::new(1920, 1080, 640, 640, YUYV, RGB),
        BenchConfig::new(3840, 2160, 640, 640, YUYV, RGBA),
        BenchConfig::new(3840, 2160, 640, 640, YUYV, RGB),
    ];
    println!("\n== Letterbox: OpenCV ==\n");
    for config in &letterbox_configs {
        bench_letterbox_opencv(config, &mut suite);
    }

    let convert_configs = vec![
        BenchConfig::new(1920, 1080, 1920, 1080, YUYV, RGBA),
        BenchConfig::new(1920, 1080, 1920, 1080, YUYV, RGB),
        BenchConfig::new(1920, 1080, 1920, 1080, RGB, RGBA),
        BenchConfig::new(3840, 2160, 3840, 2160, YUYV, RGBA),
        BenchConfig::new(3840, 2160, 3840, 2160, YUYV, RGB),
        BenchConfig::new(3840, 2160, 3840, 2160, RGB, RGBA),
    ];
    println!("\n== Convert: OpenCV ==\n");
    for config in &convert_configs {
        bench_convert_opencv(config, &mut suite);
    }

    let resize_configs = vec![
        BenchConfig::new(3840, 2160, 1920, 1080, YUYV, RGBA),
        BenchConfig::new(3840, 2160, 1920, 1080, YUYV, RGB),
        BenchConfig::new(3840, 2160, 1280, 720, YUYV, RGBA),
        BenchConfig::new(1920, 1080, 1280, 720, YUYV, RGBA),
    ];
    println!("\n== Resize: OpenCV ==\n");
    for config in &resize_configs {
        bench_resize_opencv(config, &mut suite);
    }

    suite.finish();
    println!("\nDone.");
}
