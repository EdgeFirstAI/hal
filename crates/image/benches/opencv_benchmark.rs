// SPDX-FileCopyrightText: Copyright 2025-2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! OpenCV comparison benchmarks using the edgefirst-bench harness.
//!
//! Tests the same letterbox, convert, and resize operations as
//! pipeline_benchmark but using OpenCV for cross-library comparison.
//!
//! Coverage:
//! - Inputs: YUYV, NV12, RGB
//! - Outputs: RGBA, RGB, GREY (single-channel)
//! - Operations: letterbox, convert, resize
//! - Resolutions: 1280×720, 1920×1080, 3840×2160 →
//!   640×640 / 1280×1280 (letterbox), same-size (convert),
//!   various (resize)
//!
//! Operations OpenCV cannot natively express (PlanarRgb output, i8
//! quantized outputs, VYUY/Nv16 input) are deliberately omitted — the
//! hal-side `pipeline_benchmark` covers those.
//!
//! Requires the `opencv` feature flag:
//! ```bash
//! cargo bench -p edgefirst-image --bench opencv_benchmark --features opencv -- --bench
//! ```

mod common;

use common::{calculate_letterbox, get_test_data, run_bench, BenchConfig, BenchSuite};
use edgefirst_tensor::PixelFormat;

const RGB: PixelFormat = PixelFormat::Rgb;
const RGBA: PixelFormat = PixelFormat::Rgba;
const YUYV: PixelFormat = PixelFormat::Yuyv;
const NV12: PixelFormat = PixelFormat::Nv12;
const GREY: PixelFormat = PixelFormat::Grey;

use opencv::{
    core::{set_num_threads, Mat, Scalar, Size, CV_8UC1, CV_8UC2, CV_8UC3, CV_8UC4},
    imgproc,
    prelude::*,
};

const WARMUP: usize = 10;
const ITERATIONS: usize = 100;

// =============================================================================
// Format helpers
// =============================================================================

/// Map a hal `PixelFormat` to (OpenCV Mat type, bytes-per-pixel) for a packed
/// or 2-channel layout. NV12 is handled separately — see `make_src_mat`.
fn cv_type_bpp(fmt: PixelFormat) -> Option<(i32, usize)> {
    match fmt {
        YUYV => Some((CV_8UC2, 2)),
        RGB => Some((CV_8UC3, 3)),
        RGBA => Some((CV_8UC4, 4)),
        GREY => Some((CV_8UC1, 1)),
        _ => None,
    }
}

/// Build the source OpenCV Mat over hal-format input data.
///
/// NV12 is laid out as a single-channel Mat with rows = h × 3/2: the first
/// `h` rows hold the Y plane, the remaining `h/2` rows hold interleaved UV
/// at half horizontal resolution. This is the standard OpenCV convention
/// for `COLOR_YUV2*_NV12`.
fn make_src_mat(data: &[u8], w: usize, h: usize, fmt: PixelFormat) -> Mat {
    match fmt {
        NV12 => unsafe {
            Mat::new_rows_cols_with_data_unsafe(
                (h * 3 / 2) as i32,
                w as i32,
                CV_8UC1,
                data.as_ptr() as *mut std::ffi::c_void,
                w,
            )
            .unwrap()
        },
        _ => {
            let (cv_type, bpp) = cv_type_bpp(fmt).expect("unsupported input fmt");
            unsafe {
                Mat::new_rows_cols_with_data_unsafe(
                    h as i32,
                    w as i32,
                    cv_type,
                    data.as_ptr() as *mut std::ffi::c_void,
                    w * bpp,
                )
                .unwrap()
            }
        }
    }
}

/// Build the destination OpenCV Mat over a preallocated output buffer.
fn make_dst_mat(buf: &mut [u8], w: usize, h: usize, fmt: PixelFormat) -> Mat {
    let (cv_type, bpp) = cv_type_bpp(fmt).expect("unsupported output fmt");
    unsafe {
        Mat::new_rows_cols_with_data_unsafe(
            h as i32,
            w as i32,
            cv_type,
            buf.as_mut_ptr() as *mut std::ffi::c_void,
            w * bpp,
        )
        .unwrap()
    }
}

/// Bytes per pixel in the destination buffer for sizing.
fn out_bpp(fmt: PixelFormat) -> usize {
    cv_type_bpp(fmt)
        .map(|(_, bpp)| bpp)
        .expect("unsupported output fmt")
}

/// Map a hal `(input, output)` format pair to the corresponding OpenCV
/// `cvtColor` code. Returns `None` for pairs OpenCV cannot express directly.
fn color_code(in_fmt: PixelFormat, out_fmt: PixelFormat) -> Option<i32> {
    Some(match (in_fmt, out_fmt) {
        (YUYV, RGBA) => imgproc::COLOR_YUV2RGBA_YUYV,
        (YUYV, RGB) => imgproc::COLOR_YUV2RGB_YUYV,
        (YUYV, GREY) => imgproc::COLOR_YUV2GRAY_YUYV,
        (NV12, RGBA) => imgproc::COLOR_YUV2RGBA_NV12,
        (NV12, RGB) => imgproc::COLOR_YUV2RGB_NV12,
        (NV12, GREY) => imgproc::COLOR_YUV2GRAY_NV12,
        (RGB, RGBA) => imgproc::COLOR_RGB2RGBA,
        (RGB, GREY) => imgproc::COLOR_RGB2GRAY,
        _ => return None,
    })
}

// =============================================================================
// Letterbox: OpenCV
// =============================================================================

fn bench_letterbox_opencv(config: &BenchConfig, suite: &mut BenchSuite) {
    let Some(code) = color_code(config.in_fmt, config.out_fmt) else {
        return;
    };

    let (left, top, new_w, new_h) =
        calculate_letterbox(config.in_w, config.in_h, config.out_w, config.out_h);
    let throughput = config.throughput();
    let data = get_test_data(config.in_w, config.in_h, config.in_fmt);
    let src_mat = make_src_mat(data, config.in_w, config.in_h, config.in_fmt);

    let out_bpp = out_bpp(config.out_fmt);
    let lb_size = Size::new(new_w as i32, new_h as i32);

    // Single-threaded
    {
        let name = format!("letterbox/opencv-1cpu/{}", config.id());
        let mut dst_data = vec![0u8; config.out_w * config.out_h * out_bpp];
        let mut dst_mat = make_dst_mat(&mut dst_data, config.out_w, config.out_h, config.out_fmt);
        let mut converted = Mat::default();
        let mut resized = Mat::default();

        let result = run_bench(&name, WARMUP, ITERATIONS, || {
            set_num_threads(1).unwrap();
            imgproc::cvt_color_def(&src_mat, &mut converted, code).unwrap();
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
        let mut dst_data = vec![0u8; config.out_w * config.out_h * out_bpp];
        let mut dst_mat = make_dst_mat(&mut dst_data, config.out_w, config.out_h, config.out_fmt);
        let mut converted = Mat::default();
        let mut resized = Mat::default();

        let result = run_bench(&name, WARMUP, ITERATIONS, || {
            set_num_threads(0).unwrap();
            imgproc::cvt_color_def(&src_mat, &mut converted, code).unwrap();
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
    let Some(code) = color_code(config.in_fmt, config.out_fmt) else {
        return;
    };

    let throughput = config.throughput();
    let data = get_test_data(config.in_w, config.in_h, config.in_fmt);
    let src_mat = make_src_mat(data, config.in_w, config.in_h, config.in_fmt);
    let out_bpp = out_bpp(config.out_fmt);

    // Single-threaded
    {
        let name = format!("convert/opencv-1cpu/{}", config.id());
        let mut dst_data = vec![0u8; config.out_w * config.out_h * out_bpp];
        let mut dst_mat = make_dst_mat(&mut dst_data, config.out_w, config.out_h, config.out_fmt);

        let result = run_bench(&name, WARMUP, ITERATIONS, || {
            set_num_threads(1).unwrap();
            imgproc::cvt_color_def(&src_mat, &mut dst_mat, code).unwrap();
        });
        result.print_summary_with_throughput(throughput);
        suite.record(&result);
    }

    // Multi-threaded
    {
        let name = format!("convert/opencv-multi/{}", config.id());
        let mut dst_data = vec![0u8; config.out_w * config.out_h * out_bpp];
        let mut dst_mat = make_dst_mat(&mut dst_data, config.out_w, config.out_h, config.out_fmt);

        let result = run_bench(&name, WARMUP, ITERATIONS, || {
            set_num_threads(0).unwrap();
            imgproc::cvt_color_def(&src_mat, &mut dst_mat, code).unwrap();
        });
        result.print_summary_with_throughput(throughput);
        suite.record(&result);
    }
}

// =============================================================================
// Resize: OpenCV
// =============================================================================

fn bench_resize_opencv(config: &BenchConfig, suite: &mut BenchSuite) {
    let Some(code) = color_code(config.in_fmt, config.out_fmt) else {
        return;
    };

    let throughput = config.throughput();
    let data = get_test_data(config.in_w, config.in_h, config.in_fmt);
    let src_mat = make_src_mat(data, config.in_w, config.in_h, config.in_fmt);
    let out_bpp = out_bpp(config.out_fmt);
    let target_size = Size::new(config.out_w as i32, config.out_h as i32);

    // Single-threaded
    {
        let name = format!("resize/opencv-1cpu/{}", config.id());
        let mut dst_data = vec![0u8; config.out_w * config.out_h * out_bpp];
        let mut dst_mat = make_dst_mat(&mut dst_data, config.out_w, config.out_h, config.out_fmt);
        let mut converted = Mat::default();
        let mut resized = Mat::default();

        let result = run_bench(&name, WARMUP, ITERATIONS, || {
            set_num_threads(1).unwrap();
            imgproc::cvt_color_def(&src_mat, &mut converted, code).unwrap();
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
        let mut dst_data = vec![0u8; config.out_w * config.out_h * out_bpp];
        let mut dst_mat = make_dst_mat(&mut dst_data, config.out_w, config.out_h, config.out_fmt);
        let mut converted = Mat::default();
        let mut resized = Mat::default();

        let result = run_bench(&name, WARMUP, ITERATIONS, || {
            set_num_threads(0).unwrap();
            imgproc::cvt_color_def(&src_mat, &mut converted, code).unwrap();
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

    // Letterbox: standard pipeline targets are 640×640 and 1280×1280.
    // Cover the three camera input resolutions × YUYV/NV12 × RGBA/RGB/GREY.
    let letterbox_configs = vec![
        // 720p inputs → 640×640
        BenchConfig::new(1280, 720, 640, 640, YUYV, RGBA),
        BenchConfig::new(1280, 720, 640, 640, YUYV, RGB),
        BenchConfig::new(1280, 720, 640, 640, YUYV, GREY),
        BenchConfig::new(1280, 720, 640, 640, NV12, RGBA),
        BenchConfig::new(1280, 720, 640, 640, NV12, RGB),
        BenchConfig::new(1280, 720, 640, 640, NV12, GREY),
        // 1080p inputs → 640×640
        BenchConfig::new(1920, 1080, 640, 640, YUYV, RGBA),
        BenchConfig::new(1920, 1080, 640, 640, YUYV, RGB),
        BenchConfig::new(1920, 1080, 640, 640, YUYV, GREY),
        BenchConfig::new(1920, 1080, 640, 640, NV12, RGBA),
        BenchConfig::new(1920, 1080, 640, 640, NV12, RGB),
        BenchConfig::new(1920, 1080, 640, 640, NV12, GREY),
        // 4K inputs → 640×640
        BenchConfig::new(3840, 2160, 640, 640, YUYV, RGBA),
        BenchConfig::new(3840, 2160, 640, 640, YUYV, RGB),
        BenchConfig::new(3840, 2160, 640, 640, YUYV, GREY),
        BenchConfig::new(3840, 2160, 640, 640, NV12, RGBA),
        BenchConfig::new(3840, 2160, 640, 640, NV12, RGB),
        BenchConfig::new(3840, 2160, 640, 640, NV12, GREY),
        // 4K → 1280×1280 (larger letterbox target for high-res models)
        BenchConfig::new(3840, 2160, 1280, 1280, YUYV, RGBA),
        BenchConfig::new(3840, 2160, 1280, 1280, NV12, RGBA),
    ];
    println!("\n== Letterbox: OpenCV ==\n");
    for config in &letterbox_configs {
        bench_letterbox_opencv(config, &mut suite);
    }

    // Convert (no resize): same-size YUYV/NV12 → RGBA/RGB/GREY plus RGB→RGBA, RGB→GREY.
    let convert_configs = vec![
        // 1080p same-size, YUYV input
        BenchConfig::new(1920, 1080, 1920, 1080, YUYV, RGBA),
        BenchConfig::new(1920, 1080, 1920, 1080, YUYV, RGB),
        BenchConfig::new(1920, 1080, 1920, 1080, YUYV, GREY),
        // 1080p same-size, NV12 input
        BenchConfig::new(1920, 1080, 1920, 1080, NV12, RGBA),
        BenchConfig::new(1920, 1080, 1920, 1080, NV12, RGB),
        BenchConfig::new(1920, 1080, 1920, 1080, NV12, GREY),
        // 1080p same-size, RGB input
        BenchConfig::new(1920, 1080, 1920, 1080, RGB, RGBA),
        BenchConfig::new(1920, 1080, 1920, 1080, RGB, GREY),
        // 4K same-size, YUYV input
        BenchConfig::new(3840, 2160, 3840, 2160, YUYV, RGBA),
        BenchConfig::new(3840, 2160, 3840, 2160, YUYV, RGB),
        BenchConfig::new(3840, 2160, 3840, 2160, YUYV, GREY),
        // 4K same-size, NV12 input
        BenchConfig::new(3840, 2160, 3840, 2160, NV12, RGBA),
        BenchConfig::new(3840, 2160, 3840, 2160, NV12, RGB),
        BenchConfig::new(3840, 2160, 3840, 2160, NV12, GREY),
        // 4K same-size, RGB input
        BenchConfig::new(3840, 2160, 3840, 2160, RGB, RGBA),
        BenchConfig::new(3840, 2160, 3840, 2160, RGB, GREY),
    ];
    println!("\n== Convert: OpenCV ==\n");
    for config in &convert_configs {
        bench_convert_opencv(config, &mut suite);
    }

    // Resize: convert + scale to common model and display targets.
    let resize_configs = vec![
        // 4K → 1080p
        BenchConfig::new(3840, 2160, 1920, 1080, YUYV, RGBA),
        BenchConfig::new(3840, 2160, 1920, 1080, YUYV, RGB),
        BenchConfig::new(3840, 2160, 1920, 1080, NV12, RGBA),
        BenchConfig::new(3840, 2160, 1920, 1080, NV12, RGB),
        // 4K → 720p
        BenchConfig::new(3840, 2160, 1280, 720, YUYV, RGBA),
        BenchConfig::new(3840, 2160, 1280, 720, NV12, RGBA),
        // 1080p → 720p
        BenchConfig::new(1920, 1080, 1280, 720, YUYV, RGBA),
        BenchConfig::new(1920, 1080, 1280, 720, NV12, RGBA),
    ];
    println!("\n== Resize: OpenCV ==\n");
    for config in &resize_configs {
        bench_resize_opencv(config, &mut suite);
    }

    suite.finish();
    println!("\nDone.");
}
