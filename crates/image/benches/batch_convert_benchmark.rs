// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! Batch preprocessing benchmark: deferred view-tile converts vs eager.
//!
//! Measures the `convert_deferred` + `flush` batch engine (N tiles rendered
//! into row-bands of one tall destination) against N independent eager
//! `convert` calls into N standalone destinations.
//!
//! Reports per-batch wall time for N ∈ {1, 4, 8, 16}; per-tile cost is
//! `median / N`. The cell names record which backing ran.
//!
//! Sources are synthesized NV12 and the letterbox geometry is 1280×720 →
//! 640×640 tiles.
//!
//! ```bash
//! ./batch_convert_benchmark --json batch.json
//! ```

mod common;

use common::{run_bench, BenchSuite};

use edgefirst_image::{Crop, ImageProcessor, ImageProcessorTrait};
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
        .create_image(
            IN_W,
            IN_H,
            PixelFormat::Nv12,
            DType::U8,
            None,
            edgefirst_tensor::CpuAccess::ReadWrite,
        )
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
        TensorMemory::DmaBuf => "dma",
        TensorMemory::Mem => "mem",
        TensorMemory::Shm => "shm",
        TensorMemory::Pbo => "pbo",
        // Not produced by this benchmark.
        TensorMemory::IoSurface => "iosurface",
        TensorMemory::Cuda => "cuda",
        _ => "unknown",
    }
}

fn main() {
    env_logger::init();
    let mut suite = BenchSuite::from_args();

    // Pin the GL backend.
    // SAFETY: nothing else is running yet.
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
        // Tall batched destination: N stacked row-bands of one buffer, DMA preferred.
        let parent = match TensorDyn::image(
            TILE_W,
            n * TILE_H,
            PixelFormat::Rgba,
            DType::U8,
            Some(TensorMemory::DmaBuf),
            edgefirst_tensor::CpuAccess::ReadWrite,
        )
        .or_else(|_| {
            TensorDyn::image(
                TILE_W,
                n * TILE_H,
                PixelFormat::Rgba,
                DType::U8,
                None,
                edgefirst_tensor::CpuAccess::ReadWrite,
            )
        }) {
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
            // Probe once before timing.
            let probe = (0..n)
                .try_for_each(|i| {
                    let mut tile = parent
                        .view(Region::new(0, i * TILE_H, TILE_W, TILE_H))
                        .unwrap();
                    proc.convert_deferred(
                        &src,
                        &mut tile,
                        Some(edgefirst_image::ConvertOptions::new().crop(lb_crop)),
                    )
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
                        proc.convert_deferred(
                            &src,
                            &mut tile,
                            Some(edgefirst_image::ConvertOptions::new().crop(lb_crop)),
                        )
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

        // Eager baseline: N independent converts into N standalone buffers.
        {
            let mut dsts: Vec<TensorDyn> = Vec::with_capacity(n);
            let mut alloc_failed = false;
            for _ in 0..n {
                match proc.create_image(
                    TILE_W,
                    TILE_H,
                    PixelFormat::Rgba,
                    DType::U8,
                    None,
                    edgefirst_tensor::CpuAccess::ReadWrite,
                ) {
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
                proc.convert(
                    &src,
                    &mut dsts[i],
                    Some(edgefirst_image::ConvertOptions::new().crop(lb_crop)),
                )
            });
            if let Err(e) = probe {
                println!("  {name:45} [unsupported: {e}]");
            } else {
                let r = run_bench(&name, WARMUP, ITERATIONS, || {
                    for dst in dsts.iter_mut() {
                        proc.convert(
                            &src,
                            dst,
                            Some(edgefirst_image::ConvertOptions::new().crop(lb_crop)),
                        )
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
