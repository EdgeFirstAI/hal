// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! Manual verification artifact for the GL serialization policy
//! (`requires_full_serialization` in `gl/processor/mod.rs`).
//!
//! Several `ImageProcessor`s, each on its own thread with its own GL context,
//! issue deferred tile converts concurrently and verify every output pixel
//! against the analytic ramp they wrote into their own private source. Nothing
//! is shared between threads: each allocates its source and destination once
//! and never hands them to anyone.
//!
//! **`#[ignore]`d because it is a race, not a deterministic assertion.** It
//! cannot gate CI: a clean run proves nothing on its own, and a scheduler that
//! never overlaps the processors would pass while broken. It is the artifact
//! you run by hand — `cargo test -p edgefirst-image --test gl_concurrent_stress
//! -- --ignored --nocapture` — when changing the serialization policy or
//! validating a new driver.
//!
//! What it caught: on macOS 27 / ANGLE Metal / Apple M2 Max under the old
//! `LifecycleOnly` default, a tile draw would execute with its **source-region
//! transform reset to identity** — the destination holding the thread's own
//! source stretched whole into the tile instead of the requested crop — or land
//! nothing at all. Rare and load-dependent: ~1-10 tiles per 10^4 at 8 threads
//! in this harness (a standalone binary saw ~3 per 10^3), never at 1 thread.
//! ANGLE now takes the `Full` policy and this reads clean — 0 of 150,000 tiles
//! over 5 runs, against 4 of 5 runs corrupting under `lifecycle`.
//!
//! To see the failure this pins, force the old policy:
//!
//! ```text
//! EDGEFIRST_GL_SERIALIZE=lifecycle cargo test -p edgefirst-image \
//!     --test gl_concurrent_stress -- --ignored --nocapture
//! ```
//!
//! A GPU is required. Where `ImageProcessor` has no GL backend the converts
//! fall back to CPU, which is immune — the test reports the fallback count so a
//! vacuously-clean run is visible rather than silent.

use edgefirst_image::{ImageProcessor, ImageProcessorTrait, TilingConfig};
use edgefirst_tensor::{CpuAccess, DType, PixelFormat, TensorMapTrait, TensorMemory, TensorTrait};
use std::sync::{Arc, Barrier};

const SRC_W: usize = 128;
const SRC_H: usize = 128;
const TILE: usize = 64;
/// Per-channel tolerance: the tile crop is scale-identity, so only filtering
/// round-off is expected. The failure this test exists for is whole-tile.
const TOLERANCE: u8 = 2;

/// The analytic source ramp — a distinct value per pixel per channel, so a
/// wrong source region cannot coincidentally match the right one.
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

    // Start every thread's GL work at once — the overlap is the whole point.
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

/// Iterations per thread on the concurrent legs. Sized for detection rate,
/// not runtime: under the old policy in this harness the divergence rate was
/// ~2 tiles in 10^4, so a short run reads clean more often than not. At this
/// count each concurrent leg checks ~2×10^4 tiles and
/// `EDGEFIRST_GL_SERIALIZE=lifecycle` fails most runs.
const CONCURRENT_ITERS: usize = 250;

/// 8 concurrent processors, narrow source (tile crops are a strict sub-rect).
#[test]
#[ignore = "GL concurrency stress — a race, run manually with --ignored"]
fn concurrent_tile_converts_keep_their_source_region_8x() {
    run_stress(8, CONCURRENT_ITERS, 96);
}

/// Same at the full source width, where the tile crop spans the frame
/// horizontally — this corrupted *more* than the narrow case under the old
/// policy, which is what ruled out a source-pitch explanation.
#[test]
#[ignore = "GL concurrency stress — a race, run manually with --ignored"]
fn concurrent_tile_converts_keep_their_source_region_full_width() {
    run_stress(8, CONCURRENT_ITERS, SRC_W);
}

/// The control: one processor never corrupted under any policy. If this fails,
/// the defect is not concurrency-related and the diagnosis above is wrong.
#[test]
#[ignore = "GL concurrency stress — a race, run manually with --ignored"]
fn single_processor_tile_converts_are_clean() {
    run_stress(1, 100, 96);
}
