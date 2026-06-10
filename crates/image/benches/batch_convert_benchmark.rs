// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! Batch preprocessing benchmark: deferred view-tile converts vs eager.
//!
//! Measures the `convert_deferred` + `flush` batch engine — N tiles rendered
//! into row-bands of ONE tall destination (on the Linux GL DMA path this is
//! one parent EGLImage import + per-tile `glViewport`/`glScissor` + a single
//! sync at `flush`) against N independent eager `convert` calls into N
//! standalone destinations (N imports + N syncs). The deferred path is the
//! seam the destination-engine refactor wires through `render.rs`, and this
//! bench is its before/after gate; it was previously unbenchmarked.
//!
//! Reports per-batch wall time for N ∈ {1, 4, 8, 16}; per-tile cost is
//! `median / N`. On backends without a deferred fast path (CPU/G2D fallback,
//! e.g. when DMA is unavailable) `convert_deferred` degrades to eager and the
//! two cells converge — the cell names record which backing actually ran.
//!
//! Self-contained: sources are synthesized NV12 (camera-shaped) and the
//! letterbox geometry matches the profiler (1280×720 → 640×640 tiles).
//!
//! ```bash
//! ./batch_convert_benchmark --json batch.json
//! ```

mod common;

use common::{run_bench, BenchSuite};

use edgefirst_image::{Crop, Flip, ImageProcessor, ImageProcessorTrait, Rotation};
use edgefirst_tensor::{
    DType, PixelFormat, Region, TensorDyn, TensorMapTrait, TensorMemory, TensorTrait,
};

const WARMUP: usize = 20;
const ITERATIONS: usize = 100;

const IN_W: usize = 1280;
const IN_H: usize = 720;
const TILE_W: usize = 640;
const TILE_H: usize = 640;

/// Synthesize an NV12 source: Y plane = luma gradient, chroma = neutral 128.
fn make_nv12_src(proc: &ImageProcessor) -> Option<TensorDyn> {
    let src = proc
        .create_image(IN_W, IN_H, PixelFormat::Nv12, DType::U8, None)
        .ok()?;
    {
        let t = src.as_u8().expect("u8 source");
        let mut map = t.map().expect("map source");
        let buf = map.as_mut_slice();
        for r in 0..IN_H {
            for c in 0..IN_W {
                buf[r * IN_W + c] = ((r + c) * 255 / (IN_W + IN_H)) as u8;
            }
        }
        let n = IN_W * IN_H * 3 / 2;
        for b in buf[IN_W * IN_H..n].iter_mut() {
            *b = 128;
        }
    }
    Some(src)
}

fn mem_name(m: TensorMemory) -> &'static str {
    match m {
        TensorMemory::Dma => "dma",
        TensorMemory::Mem => "mem",
        TensorMemory::Shm => "shm",
        TensorMemory::Pbo => "pbo",
    }
}

fn main() {
    env_logger::init();
    let mut suite = BenchSuite::from_args();

    // Pin the GL backend so the deferred path is measured, not G2D/CPU.
    // SAFETY: set_var is not thread-safe, but nothing else is running yet.
    unsafe { std::env::set_var("EDGEFIRST_FORCE_BACKEND", "opengl") };
    let proc = ImageProcessor::new();
    unsafe { std::env::remove_var("EDGEFIRST_FORCE_BACKEND") };

    let mut proc = match proc {
        Ok(p) => p,
        Err(e) => {
            println!("  [skipped: OpenGL ImageProcessor init failed: {e}]");
            suite.finish();
            return;
        }
    };

    let Some(src) = make_nv12_src(&proc) else {
        println!("  [skipped: NV12 source alloc failed]");
        suite.finish();
        return;
    };
    let lb_crop = Crop::letterbox([114, 114, 114, 255]);

    println!("\n== Batch convert: deferred view-tiles vs eager ==");
    println!("   src NV12 {IN_W}x{IN_H} -> {TILE_W}x{TILE_H} RGBA tiles, letterbox");
    println!("   warmup={WARMUP} iterations={ITERATIONS}\n");

    for n in [1usize, 4, 8, 16] {
        // Tall batched destination: N stacked row-bands of one buffer. DMA
        // preferred so the GL one-import band path runs where available.
        let parent = match TensorDyn::image(
            TILE_W,
            n * TILE_H,
            PixelFormat::Rgba,
            DType::U8,
            Some(TensorMemory::Dma),
        )
        .or_else(|_| TensorDyn::image(TILE_W, n * TILE_H, PixelFormat::Rgba, DType::U8, None))
        {
            Ok(p) => p,
            Err(e) => {
                println!("  n={n}: [skipped: batched destination alloc failed: {e}]");
                continue;
            }
        };
        let parent_mem = mem_name(parent.memory());

        // Deferred: N converts into sibling views + one flush.
        {
            let name = format!("batch/deferred@{parent_mem}/n{n}");
            // Probe once before timing so an unsupported path reports cleanly.
            let probe = (0..n)
                .try_for_each(|i| {
                    let mut tile = parent
                        .view(Region::new(0, i * TILE_H, TILE_W, TILE_H))
                        .unwrap();
                    proc.convert_deferred(&src, &mut tile, Rotation::None, Flip::None, lb_crop)
                })
                .and_then(|_| proc.flush());
            if let Err(e) = probe {
                println!("  {name:45} [unsupported: {e}]");
            } else {
                let r = run_bench(&name, WARMUP, ITERATIONS, || {
                    for i in 0..n {
                        let mut tile = parent
                            .view(Region::new(0, i * TILE_H, TILE_W, TILE_H))
                            .unwrap();
                        proc.convert_deferred(&src, &mut tile, Rotation::None, Flip::None, lb_crop)
                            .unwrap();
                    }
                    proc.flush().unwrap();
                });
                r.print_summary();
                println!(
                    "    per-tile median: {:.3} ms",
                    r.median().as_secs_f64() * 1000.0 / n as f64
                );
                suite.record(&r);
            }
        }

        // Eager baseline: N independent converts into N standalone buffers
        // (N imports + N syncs).
        {
            let mut dsts: Vec<TensorDyn> = Vec::with_capacity(n);
            let mut alloc_failed = false;
            for _ in 0..n {
                match proc.create_image(TILE_W, TILE_H, PixelFormat::Rgba, DType::U8, None) {
                    Ok(d) => dsts.push(d),
                    Err(e) => {
                        println!("  n={n}: [skipped: eager destination alloc failed: {e}]");
                        alloc_failed = true;
                        break;
                    }
                }
            }
            if alloc_failed {
                continue;
            }
            let eager_mem = mem_name(dsts[0].memory());
            let name = format!("batch/eager@{eager_mem}/n{n}");
            let probe = (0..n).try_for_each(|i| {
                proc.convert(&src, &mut dsts[i], Rotation::None, Flip::None, lb_crop)
            });
            if let Err(e) = probe {
                println!("  {name:45} [unsupported: {e}]");
            } else {
                let r = run_bench(&name, WARMUP, ITERATIONS, || {
                    for dst in dsts.iter_mut() {
                        proc.convert(&src, dst, Rotation::None, Flip::None, lb_crop)
                            .unwrap();
                    }
                });
                r.print_summary();
                println!(
                    "    per-tile median: {:.3} ms",
                    r.median().as_secs_f64() * 1000.0 / n as f64
                );
                suite.record(&r);
            }
        }
    }

    suite.finish();
}
