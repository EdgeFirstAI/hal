// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! Codec decode benchmarks comparing `edgefirst-codec` (custom JPEG decoder
//! with NEON SIMD, strided decode into pre-allocated tensors) against the
//! `image` crate and raw `zune-png`.
//!
//! ## Run benchmarks
//! ```bash
//! cargo bench -p edgefirst-codec --bench codec_benchmark
//! ```
//!
//! ## JSON output for CI
//! ```bash
//! cargo bench -p edgefirst-codec --bench codec_benchmark -- --json results.json
//! ```

use edgefirst_bench::{run_bench, BenchSuite};
use edgefirst_codec::{ImageDecoder, ImageLoad};
use edgefirst_tensor::{PixelFormat, Tensor, TensorMemory};
use std::sync::LazyLock;

const WARMUP: usize = 10;
const ITERATIONS: usize = 100;

static ZIDANE_JPG: LazyLock<Vec<u8>> =
    LazyLock::new(|| edgefirst_bench::testdata::read("zidane.jpg"));

static GIRAFFE_JPG: LazyLock<Vec<u8>> =
    LazyLock::new(|| edgefirst_bench::testdata::read("giraffe.jpg"));

static ZIDANE_PNG: LazyLock<Vec<u8>> =
    LazyLock::new(|| edgefirst_bench::testdata::read("zidane.png"));

// =============================================================================
// 1. edgefirst-codec — decode into pre-allocated Tensor
// =============================================================================

fn bench_codec_decode(suite: &mut BenchSuite) {
    println!("\n== edgefirst-codec: decode into pre-allocated Tensor<u8> ==\n");

    // Colour JPEGs decode to NV12; PNG decodes to its native RGB.

    // JPEG: zidane.jpg (1280×720) → NV12
    {
        let mut tensor = Tensor::<u8>::image(
            1280,
            720,
            PixelFormat::Nv12,
            Some(TensorMemory::Mem),
            edgefirst_tensor::CpuAccess::ReadWrite,
        )
        .unwrap();
        let mut decoder = ImageDecoder::new();
        tensor.load_image(&mut decoder, &ZIDANE_JPG).unwrap();

        let r = run_bench("codec/jpeg/nv12/zidane_720p", WARMUP, ITERATIONS, || {
            tensor.load_image(&mut decoder, &ZIDANE_JPG).unwrap();
            std::hint::black_box(&tensor);
        });
        r.print_summary();
        suite.record(&r);
    }

    // JPEG: giraffe.jpg (640×640) — smaller baseline image → NV12
    {
        let mut tensor = Tensor::<u8>::image(
            640,
            640,
            PixelFormat::Nv12,
            Some(TensorMemory::Mem),
            edgefirst_tensor::CpuAccess::ReadWrite,
        )
        .unwrap();
        let mut decoder = ImageDecoder::new();
        tensor.load_image(&mut decoder, &GIRAFFE_JPG).unwrap();

        let r = run_bench("codec/jpeg/nv12/giraffe_640", WARMUP, ITERATIONS, || {
            tensor.load_image(&mut decoder, &GIRAFFE_JPG).unwrap();
            std::hint::black_box(&tensor);
        });
        r.print_summary();
        suite.record(&r);
    }

    // PNG: zidane.png (1280×720) → RGB
    {
        let mut tensor = Tensor::<u8>::image(
            1280,
            720,
            PixelFormat::Rgb,
            Some(TensorMemory::Mem),
            edgefirst_tensor::CpuAccess::ReadWrite,
        )
        .unwrap();
        let mut decoder = ImageDecoder::new();
        tensor.load_image(&mut decoder, &ZIDANE_PNG).unwrap();

        let r = run_bench("codec/png/rgb/zidane_720p", WARMUP, ITERATIONS, || {
            tensor.load_image(&mut decoder, &ZIDANE_PNG).unwrap();
            std::hint::black_box(&tensor);
        });
        r.print_summary();
        suite.record(&r);
    }

    // Strided decode: decode 1280×720 JPEG into a 1920×1080 NV12 tensor
    {
        let mut tensor = Tensor::<u8>::image(
            1920,
            1080,
            PixelFormat::Nv12,
            Some(TensorMemory::Mem),
            edgefirst_tensor::CpuAccess::ReadWrite,
        )
        .unwrap();
        let mut decoder = ImageDecoder::new();
        tensor.load_image(&mut decoder, &ZIDANE_JPG).unwrap();

        let r = run_bench(
            "codec/jpeg/nv12/zidane_into_1080p",
            WARMUP,
            ITERATIONS,
            || {
                tensor.load_image(&mut decoder, &ZIDANE_JPG).unwrap();
                std::hint::black_box(&tensor);
            },
        );
        r.print_summary();
        suite.record(&r);
    }
}

// =============================================================================
// 2. Raw zune-png — baseline (allocating decoder)
// =============================================================================

fn bench_zune_raw(suite: &mut BenchSuite) {
    println!("\n== zune-png: raw decode (allocating) ==\n");

    // zune-png: zidane.png → RGB
    {
        let r = run_bench("zune/png/rgb/zidane_720p", WARMUP, ITERATIONS, || {
            use zune_png::zune_core::bytestream::ZCursor;
            use zune_png::PngDecoder;
            let mut decoder = PngDecoder::new(ZCursor::new(&ZIDANE_PNG[..]));
            decoder.decode_headers().expect("png header decode failed");
            let pixels = decoder.decode_raw().expect("png decode failed");
            std::hint::black_box(pixels);
        });
        r.print_summary();
        suite.record(&r);
    }
}

// =============================================================================
// 3. image crate — comparison baseline
// =============================================================================

fn bench_image_crate(suite: &mut BenchSuite) {
    println!("\n== image crate: decode (allocating) ==\n");

    // image crate: zidane.jpg → RGB
    {
        let r = run_bench(
            "image_crate/jpeg/rgb/zidane_720p",
            WARMUP,
            ITERATIONS,
            || {
                let img =
                    image::load_from_memory_with_format(&ZIDANE_JPG, image::ImageFormat::Jpeg)
                        .expect("image crate decode failed");
                let rgb = img.to_rgb8();
                std::hint::black_box(rgb);
            },
        );
        r.print_summary();
        suite.record(&r);
    }

    // image crate: giraffe.jpg → RGB
    {
        let r = run_bench(
            "image_crate/jpeg/rgb/giraffe_640",
            WARMUP,
            ITERATIONS,
            || {
                let img =
                    image::load_from_memory_with_format(&GIRAFFE_JPG, image::ImageFormat::Jpeg)
                        .expect("image crate decode failed");
                let rgb = img.to_rgb8();
                std::hint::black_box(rgb);
            },
        );
        r.print_summary();
        suite.record(&r);
    }

    // image crate: zidane.png → RGB
    {
        let r = run_bench(
            "image_crate/png/rgb/zidane_720p",
            WARMUP,
            ITERATIONS,
            || {
                let img = image::load_from_memory_with_format(&ZIDANE_PNG, image::ImageFormat::Png)
                    .expect("image crate decode failed");
                let rgb = img.to_rgb8();
                std::hint::black_box(rgb);
            },
        );
        r.print_summary();
        suite.record(&r);
    }
}

// =============================================================================
// 4. EXIF orientation fixtures: decode throughput per EXIF orientation.
//
//    Reported names:
//      codec/exif/jpeg/orient_<N>  — NV12 decode time for orientation N
//      codec/exif/png /orient_<N>  — RGB  decode time for orientation N
// =============================================================================

fn bench_exif_overhead(suite: &mut BenchSuite) {
    println!("\n== edgefirst-codec: EXIF orientation fixtures (JPEG + PNG) ==\n");

    let exif_iters = ITERATIONS;
    let exif_warmup = WARMUP;

    // ---- JPEG (colour → NV12, 1280×720) ----
    for o in 1..=8u32 {
        let name = format!("codec/exif/jpeg/orient_{o}");
        let data = edgefirst_bench::testdata::read(format!("zidane_exif_{o}.jpg"));
        let mut tensor = Tensor::<u8>::image(
            1280,
            720,
            PixelFormat::Nv12,
            Some(TensorMemory::Mem),
            edgefirst_tensor::CpuAccess::ReadWrite,
        )
        .unwrap();
        let mut decoder = ImageDecoder::new();
        // Warm scratch with one decode before timing.
        tensor.load_image(&mut decoder, &data).unwrap();
        let r = run_bench(&name, exif_warmup, exif_iters, || {
            tensor.load_image(&mut decoder, &data).unwrap();
            std::hint::black_box(&tensor);
        });
        r.print_summary();
        suite.record(&r);
    }

    // ---- PNG (native RGB, 1280×720) ----
    for o in 1..=8u32 {
        let name = format!("codec/exif/png/orient_{o}");
        let data = edgefirst_bench::testdata::read(format!("zidane_exif_{o}.png"));
        let mut tensor = Tensor::<u8>::image(
            1280,
            720,
            PixelFormat::Rgb,
            Some(TensorMemory::Mem),
            edgefirst_tensor::CpuAccess::ReadWrite,
        )
        .unwrap();
        let mut decoder = ImageDecoder::new();
        tensor.load_image(&mut decoder, &data).unwrap();
        let r = run_bench(&name, exif_warmup, exif_iters, || {
            tensor.load_image(&mut decoder, &data).unwrap();
            std::hint::black_box(&tensor);
        });
        r.print_summary();
        suite.record(&r);
    }
}

// =============================================================================
// 5. V4L2 JPEG persistent-stream throughput (Linux with V4L2 hardware): one
//    decoder reused across all iterations. Skipped when no `/dev/video*`
//    device is present or `EDGEFIRST_DISABLE_V4L2` is set.
//
//   codec/jpeg/nv12/zidane_720p_persistent_stream
// =============================================================================

#[cfg(all(target_os = "linux", feature = "v4l2"))]
fn bench_v4l2_persistent_stream(suite: &mut BenchSuite) {
    // Skip when the backend is disabled or no V4L2 JPEG device is available.
    if std::env::var("EDGEFIRST_DISABLE_V4L2").is_ok() {
        println!("\n== v4l2 persistent-stream: SKIP (EDGEFIRST_DISABLE_V4L2 set) ==\n");
        return;
    }
    let has_video = std::fs::read_dir("/dev")
        .map(|entries| {
            entries.flatten().any(|e| {
                e.file_name()
                    .to_str()
                    .map(|n| n.starts_with("video"))
                    .unwrap_or(false)
            })
        })
        .unwrap_or(false);
    if !has_video {
        println!("\n== v4l2 persistent-stream: SKIP (no /dev/video* device) ==\n");
        return;
    }

    println!("\n== edgefirst-codec: V4L2 JPEG persistent-stream (steady-state reuse) ==\n");

    // One decoder reused across warmup and all measured iterations.
    let mut tensor = Tensor::<u8>::image(
        1280,
        720,
        PixelFormat::Nv12,
        Some(TensorMemory::Mem),
        edgefirst_tensor::CpuAccess::ReadWrite,
    )
    .unwrap();
    let mut decoder = ImageDecoder::new();
    // The first decode sets up the stream.
    tensor.load_image(&mut decoder, &ZIDANE_JPG).unwrap();

    let r = run_bench(
        "codec/jpeg/nv12/zidane_720p_persistent_stream",
        WARMUP,
        ITERATIONS,
        || {
            tensor.load_image(&mut decoder, &ZIDANE_JPG).unwrap();
            std::hint::black_box(&tensor);
        },
    );
    r.print_summary();
    suite.record(&r);
}

#[cfg(not(all(target_os = "linux", feature = "v4l2")))]
fn bench_v4l2_persistent_stream(suite: &mut BenchSuite) {
    // Not on Linux, or the v4l2 feature is disabled.
    let _ = suite;
}

// =============================================================================
// Main
// =============================================================================

fn main() {
    let mut suite = BenchSuite::from_args();

    bench_codec_decode(&mut suite);
    bench_zune_raw(&mut suite);
    bench_image_crate(&mut suite);
    bench_exif_overhead(&mut suite);
    bench_v4l2_persistent_stream(&mut suite);

    suite.finish();
}
