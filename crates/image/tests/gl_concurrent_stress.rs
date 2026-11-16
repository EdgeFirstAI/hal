// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! Concurrency stress for the GL serialization policy
//! (`requires_full_serialization` in `gl/processor/mod.rs`).
//!
//! Several `ImageProcessor`s, each on its own thread with its own GL context,
//! issue deferred tile converts concurrently and verify every output pixel
//! against the analytic ramp they wrote into their own private source.
//!
//! The tests are `#[ignore]`d; run them with `cargo test -p edgefirst-image
//! --test gl_concurrent_stress -- --ignored --nocapture`. To run under the
//! `LifecycleOnly` policy:
//!
//! ```text
//! EDGEFIRST_GL_SERIALIZE=lifecycle cargo test -p edgefirst-image \
//!     --test gl_concurrent_stress -- --ignored --nocapture
//! ```
//!
//! Where `ImageProcessor` has no GL backend the converts fall back to CPU; the
//! test reports the fallback count.

use edgefirst_image::{ImageProcessor, ImageProcessorTrait, TilingConfig};
use edgefirst_tensor::{CpuAccess, DType, PixelFormat, TensorMapTrait, TensorMemory, TensorTrait};
use std::sync::{Arc, Barrier};

const SRC_W: usize = 128;
const SRC_H: usize = 128;
const TILE: usize = 64;
/// Per-channel tolerance.
const TOLERANCE: u8 = 2;

/// The analytic source ramp: a distinct value per pixel per channel.
fn want(x: u32, y: u32) -> [u8; 3] {
    [x as u8, y as u8, ((x + y) / 2) as u8]
}

/// One thread's work: private processor, private source and destination,
/// `tile_one` + `flush` per tile, verify against `want`. Returns
/// `(bad_tiles, tiles_checked, cpu_fallbacks)`.
fn stress_thread(iters: usize, img_w: usize, barrier: &Barrier) -> (usize, usize, u64) {
    let mut processor = ImageProcessor::new().expect("create ImageProcessor");

    let src = processor
        .create_image(
            SRC_W,
            SRC_H,
            PixelFormat::Rgba,
            DType::U8,
            Some(TensorMemory::Mem),
            CpuAccess::ReadWrite,
        )
        .expect("create source");
    {
        let t = src.as_u8().expect("u8 source");
        let mut map = t.map_mut().expect("map source");
        let s = map.as_mut_slice();
        let stride = s.len() / SRC_H;
        for y in 0..SRC_H {
            for x in 0..SRC_W {
                let o = y * stride + x * 4;
                s[o..o + 3].copy_from_slice(&want(x as u32, y as u32));
                s[o + 3] = 255;
            }
        }
    }

    let mut dst = processor
        .create_image(
            TILE,
            TILE,
            PixelFormat::Rgb,
            DType::U8,
            None,
            CpuAccess::ReadWrite,
        )
        .expect("create destination slot");

    let cfg = TilingConfig::new(TILE, TILE).with_overlap(0.2);
    let plan = processor
        .plan_tiles(img_w, SRC_H, &cfg)
        .expect("plan tiles");

    // Start every thread's GL work at once.
    barrier.wait();

    let mut bad_tiles = 0;
    let mut checked = 0;
    for _ in 0..iters {
        for placement in &plan {
            processor
                .tile_one(&src, &mut dst, placement, &cfg)
                .expect("tile_one");
            processor.flush().expect("flush");

            let t = dst.as_u8().expect("u8 destination");
            let map = t.map().expect("map destination");
            let got = map.as_slice();
            let stride = got.len() / TILE;
            let (ox, oy) = (placement.origin.0 as u32, placement.origin.1 as u32);
            let bad = (0..TILE)
                .flat_map(|y| (0..TILE).map(move |x| (x, y)))
                .flat_map(|(x, y)| {
                    let w = want(ox + x as u32, oy + y as u32);
                    (0..3).map(move |c| got[y * stride + x * 3 + c].abs_diff(w[c]))
                })
                .filter(|d| *d > TOLERANCE)
                .count();
            checked += 1;
            if bad > 0 {
                bad_tiles += 1;
            }
        }
    }
    (bad_tiles, checked, processor.convert_fallback_count())
}

fn run_stress(threads: usize, iters: usize, img_w: usize) {
    let policy = std::env::var("EDGEFIRST_GL_SERIALIZE").unwrap_or_else(|_| "<default>".into());
    let barrier = Arc::new(Barrier::new(threads));
    let handles: Vec<_> = (0..threads)
        .map(|_| {
            let barrier = Arc::clone(&barrier);
            std::thread::spawn(move || stress_thread(iters, img_w, &barrier))
        })
        .collect();

    let (mut bad, mut checked, mut fallbacks) = (0usize, 0usize, 0u64);
    for h in handles {
        let (b, c, f) = h.join().expect("stress thread panicked");
        bad += b;
        checked += c;
        fallbacks += f;
    }

    println!(
        "EDGEFIRST_GL_SERIALIZE={policy} threads={threads} iters={iters} img_w={img_w} \
         tiles_checked={checked} bad_tiles={bad} cpu_fallbacks={fallbacks}"
    );
    if fallbacks > 0 {
        println!(
            "  NOTE: {fallbacks} converts fell back to CPU — a clean result is not \
             evidence about the GL path on this host."
        );
    }
    assert_eq!(
        bad, 0,
        "{bad} of {checked} tiles diverged from their own source across {threads} \
         concurrent processors (EDGEFIRST_GL_SERIALIZE={policy}, cpu_fallbacks={fallbacks}). \
         Concurrent GL is losing per-draw state on this driver: it must take the Full \
         serialization policy — see `requires_full_serialization`."
    );
}

/// Iterations per thread on the concurrent legs.
const CONCURRENT_ITERS: usize = 250;

/// 8 concurrent processors, narrow source (tile crops are a strict sub-rect).
#[test]
#[ignore = "GL concurrency stress — a race, run manually with --ignored"]
fn concurrent_tile_converts_keep_their_source_region_8x() {
    run_stress(8, CONCURRENT_ITERS, 96);
}

/// 8 concurrent processors at the full source width, where the tile crop spans the frame.
#[test]
#[ignore = "GL concurrency stress — a race, run manually with --ignored"]
fn concurrent_tile_converts_keep_their_source_region_full_width() {
    run_stress(8, CONCURRENT_ITERS, SRC_W);
}

/// One processor, as the control.
#[test]
#[ignore = "GL concurrency stress — a race, run manually with --ignored"]
fn single_processor_tile_converts_are_clean() {
    run_stress(1, 100, 96);
}
