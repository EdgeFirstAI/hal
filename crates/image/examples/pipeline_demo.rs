// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! End-to-end decode → letterbox pipeline demonstrating zero-allocation reuse.
//!
//! This example shows the recommended pattern for real-time vision pipelines:
//!
//! 1. **Init**: Allocate input tensors large enough for the maximum
//!    expected image size, output tensors at model input resolution (640×640),
//!    and decoder state.
//! 2. **Hot loop**: Decode JPEG into the input tensor (strided), then
//!    `convert()` with letterbox resize into the output tensor.
//!
//! After warmup, the hot loop performs **zero heap allocations** — all buffers
//! (decoder scratch, tensor backing, GL resources) were sized during init.
//!
//! ## Memory modes
//!
//! By default, tensors are allocated with the best available backend:
//! - **DMA-BUF** (i.MX 8M Plus, i.MX 95, RPi 5): zero-copy EGL image import
//! - **PBO** (Jetson Orin Nano): use `EDGEFIRST_FORCE_TRANSFER=pbo`
//! - **Heap** (x86_64 desktop): fallback when no DMA-heap is available
//!
//! Override with `PIPELINE_MEM_TYPE=mem|dma|shm`.
//!
//! ## Run as demo
//! ```bash
//! cargo run --release -p edgefirst-image --example pipeline_demo
//! ```
//!
//! ## Run on embedded target
//! ```bash
//! # Cross-compile
//! cargo-zigbuild zigbuild --release --target aarch64-unknown-linux-gnu \
//!     -p edgefirst-image --example pipeline_demo
//! # Deploy and run
//! scp target/aarch64-unknown-linux-gnu/release/examples/pipeline_demo target:/tmp/
//! scp testdata/*.jpg target:/tmp/testdata/
//! ssh target 'EDGEFIRST_TESTDATA_DIR=/tmp/testdata /tmp/pipeline_demo'
//! ```
//!
//! ## Verify zero allocations with strace
//!
//! With CPU backend (guaranteed zero `brk`/`mmap`):
//! ```bash
//! cargo build --release -p edgefirst-image --example pipeline_demo
//! EDGEFIRST_FORCE_BACKEND=cpu strace -e brk,mmap -f \
//!     ./target/release/examples/pipeline_demo 2>&1 \
//!     | grep -A9999 'HOT LOOP START' | grep -B9999 'HOT LOOP END'
//! ```
//!
//! With OpenGL backend you may see a single GPU driver `mmap` (MAP_SHARED
//! on the DRM render device) the first time a new source dimension is
//! presented. This is a GPU-internal resource import, not a heap allocation.
//! On embedded targets with DMA-BUF tensors this does not occur.
//!
//! ## Verify zero allocations with perf
//! ```bash
//! perf stat -e page-faults ./target/release/examples/pipeline_demo
//! ```
//! Page faults during the hot loop indicate unexpected allocations.

use edgefirst_codec::{ImageDecoder, ImageLoad};
use edgefirst_image::{Crop, Flip, ImageProcessor, ImageProcessorTrait, Rotation};
use edgefirst_tensor::{DType, PixelFormat, TensorDyn, TensorMemory};
use std::time::Instant;

const MODEL_W: usize = 640;
const MODEL_H: usize = 640;
const ITERATIONS: usize = 100;

/// Letterbox geometry: scale to fit, centre in destination.
/// Retained for reference; placement is now computed by `Crop::letterbox`.
#[allow(dead_code)]
fn calculate_letterbox(
    src_w: usize,
    src_h: usize,
    dst_w: usize,
    dst_h: usize,
) -> (usize, usize, usize, usize) {
    let src_aspect = src_w as f64 / src_h as f64;
    let dst_aspect = dst_w as f64 / dst_h as f64;
    let (new_w, new_h) = if src_aspect > dst_aspect {
        (dst_w, (dst_w as f64 / src_aspect).round() as usize)
    } else {
        ((dst_h as f64 * src_aspect).round() as usize, dst_h)
    };
    let left = (dst_w - new_w) / 2;
    let top = (dst_h - new_h) / 2;
    (left, top, new_w, new_h)
}

/// Image entry: bytes + metadata for letterbox crop.
struct ImageEntry {
    name: &'static str,
    data: Vec<u8>,
    width: usize,
    height: usize,
}

fn load_test_images() -> Vec<ImageEntry> {
    let testdata = std::env::var("EDGEFIRST_TESTDATA_DIR")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|_| std::path::PathBuf::from("testdata"));

    let mut images = Vec::new();

    // Load available test images with known dimensions
    let candidates = [
        ("zidane.jpg", "zidane 1280×720", 1280usize, 720usize),
        ("giraffe.jpg", "giraffe 640×640", 640, 640),
    ];

    for (filename, name, w, h) in &candidates {
        let path = testdata.join(filename);
        match std::fs::read(&path) {
            Ok(data) => {
                eprintln!("  Loaded {name} ({} bytes)", data.len());
                images.push(ImageEntry {
                    name,
                    data,
                    width: *w,
                    height: *h,
                });
            }
            Err(e) => {
                eprintln!("  Skip {name}: {e}");
            }
        }
    }

    assert!(!images.is_empty(), "No test images found in {testdata:?}");
    images
}

/// Run the pipeline for one image, returning (decode_us, convert_us) medians.
///
/// Uses pre-allocated timing buffers to avoid allocations during measurement.
/// Assumes all warmup is done before calling this function.
#[allow(clippy::too_many_arguments)]
fn bench_pipeline(
    image: &ImageEntry,
    decoder: &mut ImageDecoder,
    input: &mut TensorDyn,
    output: &mut TensorDyn,
    proc: &mut ImageProcessor,
    decode_times: &mut Vec<u64>,
    convert_times: &mut Vec<u64>,
) -> (u64, u64) {
    decode_times.clear();
    convert_times.clear();

    for _ in 0..ITERATIONS {
        let t0 = Instant::now();
        let info = input.load_image(decoder, &image.data).unwrap();
        let t1 = Instant::now();

        let _ = (info.width, info.height);
        let crop = Crop::letterbox([114, 114, 114, 255]);

        proc.convert(input, output, Rotation::None, Flip::None, crop)
            .unwrap();
        let t2 = Instant::now();

        decode_times.push(t1.duration_since(t0).as_micros() as u64);
        convert_times.push(t2.duration_since(t1).as_micros() as u64);
    }

    decode_times.sort_unstable();
    convert_times.sort_unstable();

    let decode_med = decode_times[ITERATIONS / 2];
    let convert_med = convert_times[ITERATIONS / 2];

    (decode_med, convert_med)
}

fn main() {
    // Silence GL/image processor logs unless explicitly requested
    if std::env::var("RUST_LOG").is_err() {
        std::env::set_var("RUST_LOG", "warn");
    }
    env_logger::init();

    eprintln!("=== EdgeFirst Pipeline Demo: JPEG Decode → Letterbox Convert ===\n");

    // ── Init ───────────────────────────────────────────────────────────

    eprintln!("Initialization\n");

    let images = load_test_images();

    // Find the maximum image dimensions for input tensor sizing
    let max_w = images.iter().map(|i| i.width).max().unwrap();
    let max_h = images.iter().map(|i| i.height).max().unwrap();
    eprintln!("  Max input size: {max_w}×{max_h}");

    // Create ImageProcessor (probes GPU backends)
    let mut proc = ImageProcessor::new().expect("Failed to create ImageProcessor");

    // Memory type: auto-select (DMA on embedded, Mem on desktop).
    // Override with PIPELINE_MEM_TYPE=mem|dma|shm for testing.
    let mem_type = match std::env::var("PIPELINE_MEM_TYPE")
        .unwrap_or_default()
        .as_str()
    {
        "mem" => Some(TensorMemory::Mem),
        #[cfg(target_os = "linux")]
        "dma" => Some(TensorMemory::Dma),
        "shm" => Some(TensorMemory::Shm),
        _ => None, // auto-select: DMA if available, else Mem
    };
    let mem_label = mem_type.map_or("auto", |m| match m {
        TensorMemory::Mem => "Mem",
        #[cfg(target_os = "linux")]
        TensorMemory::Dma => "Dma",
        TensorMemory::Shm => "Shm",
        _ => "other",
    });
    eprintln!("  ImageProcessor created (memory: {mem_label})");

    // Allocate input tensor — larger than all test images.
    // With None (auto), DMA-BUF backed tensors are used on embedded Linux
    // for zero-copy GPU pipeline (decode → EGL image → convert).
    //
    // Colour JPEGs decode to their native NV12 layout; the codec configures
    // the tensor's dims+format during the decode and the convert step below
    // handles the NV12 → RGB colour conversion.
    let mut input = proc
        .create_image(
            max_w,
            max_h,
            PixelFormat::Nv12,
            DType::U8,
            mem_type,
            edgefirst_tensor::CpuAccess::ReadWrite,
        )
        .expect("Failed to create input tensor");
    let input_stride = input.effective_row_stride().unwrap_or(0);
    let input_memory = input.memory();
    eprintln!(
        "  Input tensor: {}×{}×{} ({} bytes, stride={}, {:?})",
        max_w,
        max_h,
        3,
        input.size(),
        input_stride,
        input_memory
    );

    // Allocate output tensors — 640×640 in both packed and planar layouts
    let mut output_packed = proc
        .create_image(
            MODEL_W,
            MODEL_H,
            PixelFormat::Rgb,
            DType::U8,
            mem_type,
            edgefirst_tensor::CpuAccess::ReadWrite,
        )
        .expect("Failed to create packed output tensor");
    let packed_stride = output_packed.effective_row_stride().unwrap_or(0);
    eprintln!(
        "  Output packed (HWC): {}×{}×{} ({} bytes, stride={}, {:?})",
        MODEL_W,
        MODEL_H,
        3,
        output_packed.size(),
        packed_stride,
        output_packed.memory()
    );

    let mut output_planar = proc
        .create_image(
            MODEL_W,
            MODEL_H,
            PixelFormat::PlanarRgb,
            DType::U8,
            mem_type,
            edgefirst_tensor::CpuAccess::ReadWrite,
        )
        .expect("Failed to create planar output tensor");
    eprintln!(
        "  Output planar (CHW): {}×{}×{} ({} bytes, {:?})",
        MODEL_W,
        MODEL_H,
        3,
        output_planar.size(),
        output_planar.memory()
    );

    // Create decoder
    let mut decoder = ImageDecoder::new();

    // Comprehensive warmup: exercise every (image × output) combination
    // multiple times so that all GL resources (PBOs, framebuffers),
    // thread pools, and scratch buffers are fully sized.
    // Warmup mirrors the hot loop pattern: N iterations per (image, output)
    // to ensure GPU caches stabilize for same-input repeated calls.
    eprintln!("\n  Warmup: exercising all code paths...");
    for image in &images {
        for _ in 0..10 {
            let info = input.load_image(&mut decoder, &image.data).unwrap();
            let _ = (info.width, info.height);
            let crop = Crop::letterbox([114, 114, 114, 255]);
            proc.convert(&input, &mut output_packed, Rotation::None, Flip::None, crop)
                .unwrap();
        }
    }
    for image in &images {
        for _ in 0..10 {
            let info = input.load_image(&mut decoder, &image.data).unwrap();
            let _ = (info.width, info.height);
            let crop = Crop::letterbox([114, 114, 114, 255]);
            proc.convert(&input, &mut output_planar, Rotation::None, Flip::None, crop)
                .unwrap();
        }
    }
    eprintln!("  Warmup complete — all buffers sized\n");

    // Pre-allocate timing vectors (reused across all benchmark runs)
    let mut decode_times = Vec::with_capacity(ITERATIONS);
    let mut convert_times = Vec::with_capacity(ITERATIONS);

    // ── Hot Loop ───────────────────────────────────────────────────────

    eprintln!("=== HOT LOOP START ===");

    eprintln!(
        "\n  {:<25} {:>10} {:>10} {:>10}",
        "Image", "Decode", "Convert", "Total"
    );
    eprintln!("  {:-<25} {:-<10} {:-<10} {:-<10}", "", "", "", "");

    // Packed RGB output (channels-last, HWC)
    eprintln!("\n  == Packed RGB output (HWC, stride={packed_stride}) ==\n");
    for image in &images {
        let (decode_us, convert_us) = bench_pipeline(
            image,
            &mut decoder,
            &mut input,
            &mut output_packed,
            &mut proc,
            &mut decode_times,
            &mut convert_times,
        );
        let total_us = decode_us + convert_us;
        eprintln!(
            "  {:<25} {:>7} µs {:>7} µs {:>7} µs",
            image.name, decode_us, convert_us, total_us
        );
    }

    // Planar RGB output (channels-first, CHW)
    eprintln!("\n  == Planar RGB output (CHW) ==\n");
    for image in &images {
        let (decode_us, convert_us) = bench_pipeline(
            image,
            &mut decoder,
            &mut input,
            &mut output_planar,
            &mut proc,
            &mut decode_times,
            &mut convert_times,
        );
        let total_us = decode_us + convert_us;
        eprintln!(
            "  {:<25} {:>7} µs {:>7} µs {:>7} µs",
            image.name, decode_us, convert_us, total_us
        );
    }

    eprintln!("\n=== HOT LOOP END ===");

    // ── Cleanup ────────────────────────────────────────────────────────

    drop(output_planar);
    drop(output_packed);
    drop(input);
    drop(proc);

    eprintln!("\nPipeline demo complete.");
}
