// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! nvJPEG GPU decode benchmark — JPEG decode into a CUDA-registered PBO
//! (`ImageProcessor::create_image` destination) on Jetson/CUDA targets.
//!
//! Lives in `edgefirst-image` rather than `edgefirst-codec` because it needs
//! `ImageProcessor` to allocate the CUDA-backed destination tensor, and `image`
//! already depends on `codec` — keeping it here avoids a circular dev-dep.
//!
//! Skips cleanly on any host without nvJPEG/CUDA; `bench_compare.py` treats
//! absent cells as one-sided and never gates on them.
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
// Decodes straight into a CUDA-registered PBO (what ImageProcessor::create_image
// yields on Jetson), emitting GPU-resident RGB. Skips cleanly on any host
// without nvJPEG/CUDA.
//
//   codec/jpeg/nvjpeg/rgbi/<fixture>  — decode-only into the PBO device pointer
// =============================================================================

fn bench_nvjpeg(suite: &mut BenchSuite) {
    if !edgefirst_codec::nvjpeg_available() {
        println!("\n== nvjpeg: SKIP (set EDGEFIRST_ENABLE_NVJPEG=1; needs CUDA + libnvjpeg) ==\n");
        return;
    }
    println!("\n== edgefirst-codec: nvjpeg decode into PBO (GPU-resident RGB) ==\n");

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
        // RGB PBO destination — auto-selected backend (PBO + CUDA on Jetson).
        let mut tensor = match processor.create_image(w, h, PixelFormat::Rgb, DType::U8, None) {
            Ok(t) => t,
            Err(e) => {
                println!("{name} SKIP: create_image failed: {e}");
                continue;
            }
        };
        let mut decoder = ImageDecoder::new();
        // Warm once and confirm nvJPEG actually fired (RGB) rather than the CPU
        // path silently producing NV12 into a non-CUDA tensor.
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
