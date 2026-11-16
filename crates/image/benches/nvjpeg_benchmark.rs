// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! nvJPEG GPU decode benchmark — JPEG decode into a CUDA-registered PBO
//! (`ImageProcessor::create_image` destination) on Jetson/CUDA targets.
//!
//! Skips on a host without nvJPEG/CUDA.
//!
//! ## Run
//! ```bash
//! cargo bench -p edgefirst-image --bench nvjpeg_benchmark
//! ```

mod common;

use common::{run_bench, BenchSuite};

use edgefirst_codec::{ImageDecoder, ImageLoad};
use edgefirst_image::ImageProcessor;
use edgefirst_tensor::{DType, PixelFormat};
use std::sync::LazyLock;

const WARMUP: usize = 10;
const ITERATIONS: usize = 100;

static ZIDANE_JPG: LazyLock<Vec<u8>> =
    LazyLock::new(|| edgefirst_bench::testdata::read("zidane.jpg"));

static GIRAFFE_JPG: LazyLock<Vec<u8>> =
    LazyLock::new(|| edgefirst_bench::testdata::read("giraffe.jpg"));

// =============================================================================
// nvJPEG GPU decode (on-target only: Linux + CUDA + libnvjpeg)
//
// Decodes into a CUDA-registered PBO, emitting GPU-resident RGB.
//
//   codec/jpeg/nvjpeg/rgbi/<fixture>  — decode-only into the PBO device pointer
// =============================================================================

fn bench_nvjpeg(suite: &mut BenchSuite) {
    if !edgefirst_codec::nvjpeg_available() {
        println!("\n== nvjpeg: SKIP (set EDGEFIRST_ENABLE_NVJPEG=1; needs CUDA + libnvjpeg) ==\n");
        return;
    }
    println!("\n== edgefirst-image: nvjpeg decode into PBO (GPU-resident RGB) ==\n");

    let processor = match ImageProcessor::new() {
        Ok(p) => p,
        Err(e) => {
            println!("nvjpeg bench SKIP: ImageProcessor unavailable: {e}");
            return;
        }
    };

    // (width, height, jpeg bytes, cell name)
    let cases: [(usize, usize, &LazyLock<Vec<u8>>, &str); 2] = [
        (1280, 720, &ZIDANE_JPG, "codec/jpeg/nvjpeg/rgbi/zidane_720p"),
        (640, 640, &GIRAFFE_JPG, "codec/jpeg/nvjpeg/rgbi/giraffe_640"),
    ];

    for (w, h, jpeg, name) in cases {
        // RGB destination on the auto-selected backend.
        let mut tensor = match processor.create_image(
            w,
            h,
            PixelFormat::Rgb,
            DType::U8,
            None,
            edgefirst_tensor::CpuAccess::ReadWrite,
        ) {
            Ok(t) => t,
            Err(e) => {
                println!("{name} SKIP: create_image failed: {e}");
                continue;
            }
        };
        let mut decoder = ImageDecoder::new();
        // Warm once and confirm the decode produced RGB.
        let info = tensor.load_image(&mut decoder, jpeg).unwrap();
        if info.format != PixelFormat::Rgb {
            println!(
                "{name} SKIP: decoded to {:?}, not Rgb — nvJPEG did not engage \
                 (destination not CUDA-backed?)",
                info.format
            );
            continue;
        }

        let r = run_bench(name, WARMUP, ITERATIONS, || {
            tensor.load_image(&mut decoder, jpeg).unwrap();
            std::hint::black_box(&tensor);
        });
        r.print_summary();
        suite.record(&r);
    }
}

fn main() {
    let mut suite = BenchSuite::from_args();
    bench_nvjpeg(&mut suite);
    suite.finish();
}
