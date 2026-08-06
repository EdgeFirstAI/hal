// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! Tiled-convert proportionality benchmark (CPU backend).
//!
//! Demonstrates the crop-contract claim from the 2026-08 crop-contract work:
//! per-convert CPU cost scales with the *tile* (crop/destination) area, not
//! the *frame* (source) area. A single 4032x2268 NV12 source (a common
//! 12MP-sensor 4:3-ish capture size, deliberately not a "round" resolution)
//! is converted three ways:
//!
//!   (i)   full-frame  -> 640x640 PlanarRgb   (general resize path, scans
//!                                              the whole 4032x2268 frame)
//!   (ii)  640x640 crop -> 640x640 PlanarRgb  (scale-identity crop, fused
//!                                              NV->planar region path)
//!   (iii) 640x640 crop -> 320x320 PlanarRgb  (2x downscale crop, general
//!                                              resize path with a halo
//!                                              sized to the crop, not the
//!                                              frame)
//!
//! If (ii) and (iii) are proportional to tile area rather than frame area,
//! their per-convert times should sit far below (i) despite converting the
//! same pixel format pair. Self-contained: the source is synthesized (Y luma
//! gradient + neutral chroma), so NO testdata files are required on the
//! target.
//!
//! A fourth, purely informational `gl/` group repeats the same three cells
//! on the OpenGL backend (via `ImageProcessorConfig::backend`, independent
//! of `EDGEFIRST_FORCE_BACKEND`) now that GL renders `PlanarRgb` into heap
//! destinations directly. It is not part of the CPU proportionality claim
//! and is skipped silently if GL initialization fails on the host.
//!
//! ```bash
//! cargo bench -p edgefirst-image --bench tiled_convert_benchmark
//! # on-target:
//! EDGEFIRST_FORCE_BACKEND=cpu ./tiled_convert_benchmark --json tiled.json
//! ```

mod common;

use common::{run_bench, BenchSuite};

use edgefirst_image::{
    ComputeBackend, Crop, Flip, ImageProcessor, ImageProcessorConfig, ImageProcessorTrait, Region,
    Rotation,
};
use edgefirst_tensor::{CpuAccess, DType, PixelFormat, TensorDyn, TensorMapTrait, TensorTrait};

const WARMUP: usize = 10;
const ITERATIONS: usize = 100;

// A 12MP-class capture, deliberately not a "round" resolution, so the
// frame-area/tile-area contrast can't be mistaken for a coincidence of
// convenient numbers. Both dimensions are even (chroma-aligned for 4:2:0).
const SRC_W: usize = 4032;
const SRC_H: usize = 2268;

// The crop is centered and origin-aligned to an even pixel so it qualifies
// for the fused NV->planar region path (scale-identity, chroma-aligned).
const CROP_W: usize = 640;
const CROP_H: usize = 640;
const CROP_X: usize = (SRC_W - CROP_W) / 2;
const CROP_Y: usize = (SRC_H - CROP_H) / 2;

/// Combined semi-planar (NV12) buffer byte length for a `w x h` image.
fn nv12_len(w: usize, h: usize) -> usize {
    w * h * 3 / 2
}

/// Synthesize an NV12 source: Y plane = luma gradient, chroma = neutral 128.
fn fill_nv12(buf: &mut [u8], w: usize, h: usize) {
    for r in 0..h {
        for c in 0..w {
            buf[r * w + c] = ((r + c) * 255 / (w + h)) as u8;
        }
    }
    for b in buf.iter_mut().skip(w * h) {
        *b = 128;
    }
}

fn main() {
    // This bench characterises the CPU convert path specifically — pin the
    // backend before anything that could spawn a thread.
    // SAFETY: set_var runs at the very top of main(), before env_logger,
    // argument parsing, or any worker threads — the process is single-threaded.
    if std::env::var("EDGEFIRST_FORCE_BACKEND").is_err() {
        unsafe { std::env::set_var("EDGEFIRST_FORCE_BACKEND", "cpu") };
    }

    env_logger::init();
    let mut suite = BenchSuite::from_args();

    println!(
        "\n== Tiled-convert proportionality benchmark ({SRC_W}x{SRC_H} NV12 source) backend={} ==",
        std::env::var("EDGEFIRST_FORCE_BACKEND").unwrap_or_else(|_| "auto".into())
    );
    println!("   warmup={WARMUP} iterations={ITERATIONS}\n");

    let mut proc = match ImageProcessor::new() {
        Ok(p) => p,
        Err(e) => {
            println!("  [skipped: ImageProcessor init failed: {e}]");
            suite.finish();
            return;
        }
    };

    let src = match proc.create_image(
        SRC_W,
        SRC_H,
        PixelFormat::Nv12,
        DType::U8,
        None,
        edgefirst_tensor::CpuAccess::ReadWrite,
    ) {
        Ok(s) => s,
        Err(e) => {
            println!("  [skipped: source alloc failed: {e}]");
            suite.finish();
            return;
        }
    };
    {
        let mut map = src.as_u8().unwrap().map().unwrap();
        let n = nv12_len(SRC_W, SRC_H);
        fill_nv12(&mut map.as_mut_slice()[..n], SRC_W, SRC_H);
    }
    let src_bytes = nv12_len(SRC_W, SRC_H) as u64;
    let tile_bytes = nv12_len(CROP_W, CROP_H) as u64;

    let crop_identity = Crop::new().with_source(Some(Region::new(CROP_X, CROP_Y, CROP_W, CROP_H)));

    // Each cell: (name, out dims, crop, throughput bytes for the printed
    // summary). Throughput is reported against each cell's own input bytes
    // (frame for (i), tile for (ii)/(iii)) so the printed MiB/s columns stay
    // meaningful per cell; the proportionality claim itself is read off the
    // absolute per-convert times, not the throughput column.
    let cells: &[(&str, usize, usize, Crop, u64)] = &[
        (
            "full_frame_4032x2268_to_640x640",
            640,
            640,
            Crop::no_crop(),
            src_bytes,
        ),
        (
            "crop_640x640_to_640x640_fused",
            640,
            640,
            crop_identity,
            tile_bytes,
        ),
        (
            "crop_640x640_to_320x320_halo",
            320,
            320,
            crop_identity,
            tile_bytes,
        ),
    ];

    run_group(&mut proc, &src, cells, &mut suite);

    // Fourth, informational-only group: the same three cells on the OpenGL
    // backend, now that GL renders PlanarRgb into heap destinations directly
    // (rather than declining to CPU). Selected via `ImageProcessorConfig`
    // rather than `EDGEFIRST_FORCE_BACKEND` so it does not disturb the CPU
    // group above. Not part of the CPU proportionality claim.
    println!();
    // `..Default::default()` is a real update on Linux (extra GL/G2D fields)
    // but covers no remaining fields on macOS where `backend` is the only
    // field — allow needless_update for cross-platform parity.
    #[allow(clippy::needless_update)]
    let gl_config = ImageProcessorConfig {
        backend: ComputeBackend::OpenGl,
        ..Default::default()
    };
    match ImageProcessor::with_config(gl_config) {
        Ok(mut gl_proc) => {
            let gl_cells: Vec<(String, usize, usize, Crop, u64)> = cells
                .iter()
                .map(|&(name, ow, oh, crop, throughput)| {
                    (format!("gl/{name}"), ow, oh, crop, throughput)
                })
                .collect();
            let gl_cells: Vec<(&str, usize, usize, Crop, u64)> = gl_cells
                .iter()
                .map(|(name, ow, oh, crop, throughput)| {
                    (name.as_str(), *ow, *oh, *crop, *throughput)
                })
                .collect();
            run_group(&mut gl_proc, &src, &gl_cells, &mut suite);
        }
        Err(e) => println!("  gl/* [skipped: OpenGL ImageProcessor init failed: {e}]"),
    }

    suite.finish();
}

/// Run a group of (name, out_w, out_h, crop, throughput_bytes) cells against
/// `proc`, sharing the single `src` tensor across all of them.
fn run_group(
    proc: &mut ImageProcessor,
    src: &TensorDyn,
    cells: &[(&str, usize, usize, Crop, u64)],
    suite: &mut BenchSuite,
) {
    for &(name, ow, oh, crop, throughput) in cells {
        let mut dst = match proc.create_image(
            ow,
            oh,
            PixelFormat::PlanarRgb,
            DType::U8,
            None,
            CpuAccess::ReadWrite,
        ) {
            Ok(d) => d,
            Err(e) => {
                println!("  {name:50} [dst alloc failed: {e}]");
                continue;
            }
        };
        // Probe once: an unsupported cell is reported and skipped instead of
        // panicking inside the timing loop.
        if let Err(e) = proc.convert(src, &mut dst, Rotation::None, Flip::None, crop) {
            println!("  {name:50} [unsupported: {e}]");
            continue;
        }
        let r = run_bench(name, WARMUP, ITERATIONS, || {
            proc.convert(src, &mut dst, Rotation::None, Flip::None, crop)
                .unwrap();
        });
        r.print_summary_with_throughput(throughput);
        suite.record(&r);
    }
}
