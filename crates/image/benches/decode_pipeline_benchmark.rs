// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! Decode → Letterbox pipeline benchmark measuring JPEG decode into a strided
//! input tensor followed by GPU/CPU letterbox convert to 640×640 output.
//!
//! This benchmark measures the full vision preprocessing pipeline:
//! 1. JPEG decode into an oversized (strided) input tensor
//! 2. Letterbox convert into model-sized output (640×640)
//!
//! Results are compared against the existing `pipeline_benchmark` baselines
//! which convert from non-strided (tight) inputs.
//!
//! ## Run
//! ```bash
//! cargo bench -p edgefirst-image --bench decode_pipeline_benchmark
//! ```

mod common;

use common::{calculate_letterbox, run_bench, BenchSuite};

use edgefirst_codec::{DecodeOptions, ImageDecoder, ImageLoad};
use edgefirst_image::{Crop, Flip, ImageProcessor, ImageProcessorTrait, Rect, Rotation};
use edgefirst_tensor::{DType, PixelFormat, TensorDyn};

const WARMUP: usize = 10;
const ITERATIONS: usize = 100;
const MODEL_W: usize = 640;
const MODEL_H: usize = 640;

struct TestImage {
    name: &'static str,
    filename: &'static str,
    width: usize,
    height: usize,
}

const TEST_IMAGES: &[TestImage] = &[
    TestImage {
        name: "zidane_1280x720",
        filename: "zidane.jpg",
        width: 1280,
        height: 720,
    },
    TestImage {
        name: "giraffe_640x640",
        filename: "giraffe.jpg",
        width: 640,
        height: 640,
    },
];

fn load_image_data(filename: &str) -> Vec<u8> {
    edgefirst_bench::testdata::read(filename)
}

/// Benchmark: JPEG decode into strided tensor (decode only).
fn bench_decode_strided(suite: &mut BenchSuite) {
    println!("\n== Decode: JPEG → strided RGB tensor ==\n");

    let mut decoder = ImageDecoder::new();
    let opts = DecodeOptions::default()
        .with_format(PixelFormat::Rgb)
        .with_exif(false);

    // Use an oversized tensor to demonstrate strided decode
    let max_w = TEST_IMAGES.iter().map(|i| i.width).max().unwrap();
    let max_h = TEST_IMAGES.iter().map(|i| i.height).max().unwrap();
    let mut input = TensorDyn::image(max_w, max_h, PixelFormat::Rgb, DType::U8, None).unwrap();
    let stride = input.effective_row_stride().unwrap_or(0);

    for img in TEST_IMAGES {
        let data = load_image_data(img.filename);
        let name = format!("decode_strided/{} (stride={})", img.name, stride);
        let throughput = (img.width * img.height * 3) as u64;

        // Warmup
        input.load_image(&mut decoder, &data, &opts).unwrap();

        let result = run_bench(&name, WARMUP, ITERATIONS, || {
            input.load_image(&mut decoder, &data, &opts).unwrap();
        });
        result.print_summary_with_throughput(throughput);
        suite.record(&result);
    }
}

/// Benchmark: full pipeline (decode + letterbox convert) with strided input.
fn bench_decode_convert(
    proc: &mut ImageProcessor,
    output_fmt: PixelFormat,
    layout_name: &str,
    suite: &mut BenchSuite,
) {
    println!(
        "\n== Pipeline: decode → letterbox {} (strided input) ==\n",
        layout_name
    );

    let mut decoder = ImageDecoder::new();
    let opts = DecodeOptions::default()
        .with_format(PixelFormat::Rgb)
        .with_exif(false);

    let max_w = TEST_IMAGES.iter().map(|i| i.width).max().unwrap();
    let max_h = TEST_IMAGES.iter().map(|i| i.height).max().unwrap();

    // Auto-select memory type: DMA if available, else Mem
    let mut input = proc
        .create_image(max_w, max_h, PixelFormat::Rgb, DType::U8, None)
        .expect("Failed to create input tensor");
    let mut output = proc
        .create_image(MODEL_W, MODEL_H, output_fmt, DType::U8, None)
        .expect("Failed to create output tensor");

    for img in TEST_IMAGES {
        let data = load_image_data(img.filename);
        let (left, top, new_w, new_h) =
            calculate_letterbox(img.width, img.height, MODEL_W, MODEL_H);
        let crop = Crop::new()
            .with_src_rect(Some(Rect::new(0, 0, img.width, img.height)))
            .with_dst_rect(Some(Rect::new(left, top, new_w, new_h)))
            .with_dst_color(Some([114, 114, 114, 255]));

        // Warmup all paths
        for _ in 0..WARMUP {
            input.load_image(&mut decoder, &data, &opts).unwrap();
            proc.convert(&input, &mut output, Rotation::None, Flip::None, crop)
                .unwrap();
        }

        // Decode only
        let decode_name = format!("pipeline_decode/{}/{}", layout_name, img.name);
        let decode_result = run_bench(&decode_name, 0, ITERATIONS, || {
            input.load_image(&mut decoder, &data, &opts).unwrap();
        });

        // Convert only (from last decoded image)
        input.load_image(&mut decoder, &data, &opts).unwrap();
        let convert_name = format!("pipeline_convert/{}/{}", layout_name, img.name);
        let convert_result = run_bench(&convert_name, 0, ITERATIONS, || {
            proc.convert(&input, &mut output, Rotation::None, Flip::None, crop)
                .unwrap();
        });

        // Full pipeline
        let pipeline_name = format!("pipeline_full/{}/{}", layout_name, img.name);
        let throughput = (img.width * img.height * 3) as u64;
        let pipeline_result = run_bench(&pipeline_name, 0, ITERATIONS, || {
            input.load_image(&mut decoder, &data, &opts).unwrap();
            proc.convert(&input, &mut output, Rotation::None, Flip::None, crop)
                .unwrap();
        });

        decode_result.print_summary();
        convert_result.print_summary();
        pipeline_result.print_summary_with_throughput(throughput);
        suite.record(&decode_result);
        suite.record(&convert_result);
        suite.record(&pipeline_result);
    }
}

fn main() {
    if std::env::var("RUST_LOG").is_err() {
        std::env::set_var("RUST_LOG", "warn");
    }
    env_logger::init();

    let args: Vec<String> = std::env::args().collect();
    let is_bench = args.iter().any(|a| a == "--bench");
    if !is_bench {
        println!("Usage: decode_pipeline_benchmark --bench");
        return;
    }

    println!("=== Decode → Letterbox Pipeline Benchmark ===");

    let mut suite = BenchSuite::from_args();

    // Decode-only benchmarks (no ImageProcessor needed)
    bench_decode_strided(&mut suite);

    // Full pipeline benchmarks
    let mut proc = ImageProcessor::new().expect("Failed to create ImageProcessor");

    bench_decode_convert(&mut proc, PixelFormat::Rgb, "HWC", &mut suite);
    bench_decode_convert(&mut proc, PixelFormat::PlanarRgb, "CHW", &mut suite);

    suite.finish();
}
