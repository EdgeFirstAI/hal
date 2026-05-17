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
use edgefirst_codec::{DecodeOptions, ImageDecoder, ImageLoad};
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

    let opts = DecodeOptions::default()
        .with_format(PixelFormat::Rgb)
        .with_exif(false);
    let opts_rgba = DecodeOptions::default()
        .with_format(PixelFormat::Rgba)
        .with_exif(false);

    // JPEG: zidane.jpg (1280×720) → RGB
    {
        let mut tensor =
            Tensor::<u8>::image(1280, 720, PixelFormat::Rgb, Some(TensorMemory::Mem)).unwrap();
        let mut decoder = ImageDecoder::new();
        tensor.load_image(&mut decoder, &ZIDANE_JPG, &opts).unwrap();

        let r = run_bench("codec/jpeg/rgb/zidane_720p", WARMUP, ITERATIONS, || {
            tensor.load_image(&mut decoder, &ZIDANE_JPG, &opts).unwrap();
            std::hint::black_box(&tensor);
        });
        r.print_summary();
        suite.record(&r);
    }

    // JPEG: zidane.jpg → RGBA
    {
        let mut tensor =
            Tensor::<u8>::image(1280, 720, PixelFormat::Rgba, Some(TensorMemory::Mem)).unwrap();
        let mut decoder = ImageDecoder::new();
        tensor
            .load_image(&mut decoder, &ZIDANE_JPG, &opts_rgba)
            .unwrap();

        let r = run_bench("codec/jpeg/rgba/zidane_720p", WARMUP, ITERATIONS, || {
            tensor
                .load_image(&mut decoder, &ZIDANE_JPG, &opts_rgba)
                .unwrap();
            std::hint::black_box(&tensor);
        });
        r.print_summary();
        suite.record(&r);
    }

    // JPEG: giraffe.jpg (640×640) — smaller baseline image
    {
        let mut tensor =
            Tensor::<u8>::image(640, 640, PixelFormat::Rgb, Some(TensorMemory::Mem)).unwrap();
        let mut decoder = ImageDecoder::new();
        tensor
            .load_image(&mut decoder, &GIRAFFE_JPG, &opts)
            .unwrap();

        let r = run_bench("codec/jpeg/rgb/giraffe_640", WARMUP, ITERATIONS, || {
            tensor
                .load_image(&mut decoder, &GIRAFFE_JPG, &opts)
                .unwrap();
            std::hint::black_box(&tensor);
        });
        r.print_summary();
        suite.record(&r);
    }

    // JPEG: f32 decode
    {
        let mut tensor =
            Tensor::<f32>::image(1280, 720, PixelFormat::Rgb, Some(TensorMemory::Mem)).unwrap();
        let mut decoder = ImageDecoder::new();
        tensor.load_image(&mut decoder, &ZIDANE_JPG, &opts).unwrap();

        let r = run_bench("codec/jpeg/rgb_f32/zidane_720p", WARMUP, ITERATIONS, || {
            tensor.load_image(&mut decoder, &ZIDANE_JPG, &opts).unwrap();
            std::hint::black_box(&tensor);
        });
        r.print_summary();
        suite.record(&r);
    }

    // PNG: zidane.png (1280×720) → RGB
    {
        let mut tensor =
            Tensor::<u8>::image(1280, 720, PixelFormat::Rgb, Some(TensorMemory::Mem)).unwrap();
        let mut decoder = ImageDecoder::new();
        tensor.load_image(&mut decoder, &ZIDANE_PNG, &opts).unwrap();

        let r = run_bench("codec/png/rgb/zidane_720p", WARMUP, ITERATIONS, || {
            tensor.load_image(&mut decoder, &ZIDANE_PNG, &opts).unwrap();
            std::hint::black_box(&tensor);
        });
        r.print_summary();
        suite.record(&r);
    }

    // Strided decode: decode 1280×720 into a 1920×1080 tensor
    {
        let mut tensor =
            Tensor::<u8>::image(1920, 1080, PixelFormat::Rgb, Some(TensorMemory::Mem)).unwrap();
        let mut decoder = ImageDecoder::new();
        tensor.load_image(&mut decoder, &ZIDANE_JPG, &opts).unwrap();

        let r = run_bench(
            "codec/jpeg/rgb/zidane_into_1080p",
            WARMUP,
            ITERATIONS,
            || {
                tensor.load_image(&mut decoder, &ZIDANE_JPG, &opts).unwrap();
                std::hint::black_box(&tensor);
            },
        );
        r.print_summary();
        suite.record(&r);
    }

    // JPEG: BGRA decode
    {
        let opts_bgra = DecodeOptions::default()
            .with_format(PixelFormat::Bgra)
            .with_exif(false);
        let mut tensor =
            Tensor::<u8>::image(1280, 720, PixelFormat::Bgra, Some(TensorMemory::Mem)).unwrap();
        let mut decoder = ImageDecoder::new();
        tensor
            .load_image(&mut decoder, &ZIDANE_JPG, &opts_bgra)
            .unwrap();

        let r = run_bench("codec/jpeg/bgra/zidane_720p", WARMUP, ITERATIONS, || {
            tensor
                .load_image(&mut decoder, &ZIDANE_JPG, &opts_bgra)
                .unwrap();
            std::hint::black_box(&tensor);
        });
        r.print_summary();
        suite.record(&r);
    }

    // JPEG: NV12 decode (skip color conversion)
    {
        let opts_nv12 = DecodeOptions::default()
            .with_format(PixelFormat::Nv12)
            .with_exif(false);
        let mut tensor = Tensor::<u8>::image(
            1280,
            720 * 3 / 2,
            PixelFormat::Grey,
            Some(TensorMemory::Mem),
        )
        .unwrap();
        let mut decoder = ImageDecoder::new();
        tensor
            .load_image(&mut decoder, &ZIDANE_JPG, &opts_nv12)
            .unwrap();

        let r = run_bench("codec/jpeg/nv12/zidane_720p", WARMUP, ITERATIONS, || {
            tensor
                .load_image(&mut decoder, &ZIDANE_JPG, &opts_nv12)
                .unwrap();
            std::hint::black_box(&tensor);
        });
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
// 4. EXIF orientation overhead — measure the cost of apply_exif=true on
//    each of the 8 spec-defined EXIF orientations. Uses fixtures generated
//    by `scripts/generate_exif_fixtures.py` which carry identical pixel
//    data and differ only in the EXIF/eXIf orientation tag.
//
//    Reported names:
//      exif/jpeg/orient_<N>/apply_<bool>  — decode time for orientation N
//      exif/png /orient_<N>/apply_<bool>
//
//    Compare:
//      - apply_false across all 8 → should be constant (sanity check;
//        all variants share scan/IDAT content).
//      - apply_true on orient_1   → "decided not to rotate" cost.
//      - apply_true on orient_3   → in-place 180° (no dim swap, no scratch).
//      - apply_true on orient_6/8 → 90°/270° rotation with scratch alloc + copy.
//      - apply_true on orient_5/7 → rotation + horizontal flip.
//    The 6/8 vs 1 delta is the headline number for "how much does EXIF
//    rotation cost on this platform".
// =============================================================================

fn bench_exif_overhead(suite: &mut BenchSuite) {
    println!("\n== edgefirst-codec: EXIF orientation overhead (JPEG + PNG) ==\n");

    // Source dims are 1280×720; orientations 5/6/7/8 produce 720×1280
    // output. Allocate at max(w,h) on both axes so a single tensor
    // serves every variant without reallocating between bench rounds.
    let max_dim = 1280;
    let exif_iters = ITERATIONS;
    let exif_warmup = WARMUP;

    // ---- JPEG ----
    for apply in [false, true] {
        let opts = DecodeOptions::default()
            .with_format(PixelFormat::Rgb)
            .with_exif(apply);
        for o in 1..=8u32 {
            let name = format!("codec/exif/jpeg/orient_{o}/apply_{apply}");
            let data = edgefirst_bench::testdata::read(format!("zidane_exif_{o}.jpg"));
            let mut tensor =
                Tensor::<u8>::image(max_dim, max_dim, PixelFormat::Rgb, Some(TensorMemory::Mem))
                    .unwrap();
            let mut decoder = ImageDecoder::new();
            // Warm scratch with one decode before timing.
            tensor.load_image(&mut decoder, &data, &opts).unwrap();
            let r = run_bench(&name, exif_warmup, exif_iters, || {
                tensor.load_image(&mut decoder, &data, &opts).unwrap();
                std::hint::black_box(&tensor);
            });
            r.print_summary();
            suite.record(&r);
        }
    }

    // ---- PNG ----
    for apply in [false, true] {
        let opts = DecodeOptions::default()
            .with_format(PixelFormat::Rgb)
            .with_exif(apply);
        for o in 1..=8u32 {
            let name = format!("codec/exif/png/orient_{o}/apply_{apply}");
            let data = edgefirst_bench::testdata::read(format!("zidane_exif_{o}.png"));
            let mut tensor =
                Tensor::<u8>::image(max_dim, max_dim, PixelFormat::Rgb, Some(TensorMemory::Mem))
                    .unwrap();
            let mut decoder = ImageDecoder::new();
            tensor.load_image(&mut decoder, &data, &opts).unwrap();
            let r = run_bench(&name, exif_warmup, exif_iters, || {
                tensor.load_image(&mut decoder, &data, &opts).unwrap();
                std::hint::black_box(&tensor);
            });
            r.print_summary();
            suite.record(&r);
        }
    }
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

    suite.finish();
}
