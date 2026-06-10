// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! GL convert memory×format matrix benchmark.
//!
//! Measures every (src memory) × (dst memory) × (dst format) × (dst dtype)
//! convert cell that the GL backend supports on the current target, using the
//! profiler's letterbox geometry (1280×720 → 640×640). This is the per-cell
//! before/after gate for the destination-path refactor: each cell maps onto
//! one of the GL destination lowerings (DMA EGLImage, PBO↔PBO, any→PBO,
//! PBO→Mem, texture readback), so a regression in any single lowering shows
//! up as a named cell, not an average.
//!
//! Memory backings are requested as {Auto, Dma, Mem} and the cell is named by
//! the memory the allocation actually **resolved** to (`tensor.memory()`),
//! because the interesting backing is target-dependent: Auto resolves to DMA
//! on i.MX/RPi boards and to PBO on Orin (where the GL transfer backend is
//! PBO and DMA import is unavailable). Cells whose resolved key collides
//! (e.g. Auto == Dma on i.MX) are deduplicated; unsupported cells are
//! reported and skipped.
//!
//! The GL backend is pinned via `EDGEFIRST_FORCE_BACKEND=opengl` so G2D/CPU
//! never absorb a cell. Self-contained: sources are synthesized.
//!
//! ```bash
//! ./convert_matrix_benchmark --json matrix.json
//! ```

mod common;

use common::{run_bench, BenchSuite};

use edgefirst_image::{Crop, Flip, ImageProcessor, ImageProcessorTrait, Rotation};
use edgefirst_tensor::{DType, PixelFormat, TensorDyn, TensorMapTrait, TensorMemory, TensorTrait};
use std::collections::HashSet;

const WARMUP: usize = 20;
const ITERATIONS: usize = 100;

const IN_W: usize = 1280;
const IN_H: usize = 720;
const OUT_W: usize = 640;
const OUT_H: usize = 640;

/// Memory request for one side of a cell. `Auto` lets `create_image` pick the
/// target's best backing (DMA on i.MX/RPi, PBO on Orin, IOSurface on macOS).
#[derive(Clone, Copy, Debug, PartialEq)]
enum MemReq {
    Auto,
    Dma,
    Mem,
}

impl MemReq {
    fn to_request(self) -> Option<TensorMemory> {
        match self {
            MemReq::Auto => None,
            MemReq::Dma => Some(TensorMemory::Dma),
            MemReq::Mem => Some(TensorMemory::Mem),
        }
    }
}

fn mem_name(m: TensorMemory) -> &'static str {
    match m {
        TensorMemory::Dma => "dma",
        TensorMemory::Mem => "mem",
        TensorMemory::Shm => "shm",
        TensorMemory::Pbo => "pbo",
    }
}

fn fmt_name(f: PixelFormat) -> &'static str {
    match f {
        PixelFormat::Rgba => "rgba",
        PixelFormat::Bgra => "bgra",
        PixelFormat::Rgb => "rgb",
        PixelFormat::PlanarRgb => "planar_rgb",
        PixelFormat::Grey => "grey",
        PixelFormat::Nv12 => "nv12",
        _ => "other",
    }
}

/// Source bytes for throughput reporting.
fn src_bytes(fmt: PixelFormat) -> u64 {
    (match fmt {
        PixelFormat::Nv12 => IN_W * IN_H * 3 / 2,
        PixelFormat::Rgba => IN_W * IN_H * 4,
        _ => IN_W * IN_H,
    }) as u64
}

/// Fill a source with a deterministic gradient (luma gradient + neutral
/// chroma for NV12; RGBA channel ramps for RGBA).
fn fill_src(t: &TensorDyn, fmt: PixelFormat) {
    let tensor = t.as_u8().expect("u8 source");
    let mut map = tensor.map().expect("map source");
    let buf = map.as_mut_slice();
    match fmt {
        PixelFormat::Nv12 => {
            let n = IN_W * IN_H * 3 / 2;
            for r in 0..IN_H {
                for c in 0..IN_W {
                    buf[r * IN_W + c] = ((r + c) * 255 / (IN_W + IN_H)) as u8;
                }
            }
            for b in buf[IN_W * IN_H..n].iter_mut() {
                *b = 128;
            }
        }
        _ => {
            for (i, px) in buf.chunks_exact_mut(4).take(IN_W * IN_H).enumerate() {
                px[0] = (i % 256) as u8;
                px[1] = ((i / 256) % 256) as u8;
                px[2] = ((i / 65536) % 256) as u8;
                px[3] = 255;
            }
        }
    }
}

fn main() {
    env_logger::init();
    let mut suite = BenchSuite::from_args();

    // Pin the GL backend so G2D/CPU never absorb a cell.
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

    println!("\n== GL convert matrix ({IN_W}x{IN_H} -> {OUT_W}x{OUT_H} letterbox) ==");
    println!("   warmup={WARMUP} iterations={ITERATIONS}\n");

    let lb_crop = Crop::letterbox([114, 114, 114, 255]);
    let mem_reqs = [MemReq::Auto, MemReq::Dma, MemReq::Mem];
    let dst_fmts = [
        PixelFormat::Rgba,
        PixelFormat::Bgra,
        PixelFormat::Rgb,
        PixelFormat::PlanarRgb,
        PixelFormat::Grey,
    ];
    let dtypes = [DType::U8, DType::I8];

    // Dedup cells by the memory the allocation actually resolved to, so a
    // board where Auto == Dma measures each real cell exactly once.
    let mut seen: HashSet<String> = HashSet::new();

    for src_fmt in [PixelFormat::Nv12, PixelFormat::Rgba] {
        for src_req in mem_reqs {
            let src = match proc.create_image(IN_W, IN_H, src_fmt, DType::U8, src_req.to_request())
            {
                Ok(s) => s,
                Err(_) => continue, // backing not available on this target
            };
            let src_mem = src.memory();
            fill_src(&src, src_fmt);

            for dst_req in mem_reqs {
                for dst_fmt in dst_fmts {
                    for dtype in dtypes {
                        // Probe the destination allocation first: its resolved
                        // memory is part of the cell key.
                        let Ok(mut dst) =
                            proc.create_image(OUT_W, OUT_H, dst_fmt, dtype, dst_req.to_request())
                        else {
                            continue;
                        };
                        let name = format!(
                            "matrix/{}@{}->{}.{}@{}",
                            fmt_name(src_fmt),
                            mem_name(src_mem),
                            fmt_name(dst_fmt),
                            if dtype == DType::I8 { "i8" } else { "u8" },
                            mem_name(dst.memory()),
                        );
                        if !seen.insert(name.clone()) {
                            continue; // resolved cell already measured
                        }

                        // Probe twice: some driver bugs only appear on the
                        // REPEAT convert of a cell (e.g. Mali GL_INVALID_VALUE
                        // on the second heap-RGBA -> DMA-RGB convert), and a
                        // single-shot probe would misclassify them as a panic
                        // inside the timing loop.
                        if let Err(e) = (0..2).try_for_each(|_| {
                            proc.convert(&src, &mut dst, Rotation::None, Flip::None, lb_crop)
                        }) {
                            println!("  {name:55} [unsupported: {e}]");
                            continue;
                        }
                        // Isolate the timing loop: a mid-loop convert failure
                        // marks THIS cell failed and the matrix continues —
                        // one broken cell must not cost the whole baseline.
                        let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                            run_bench(&name, WARMUP, ITERATIONS, || {
                                proc.convert(&src, &mut dst, Rotation::None, Flip::None, lb_crop)
                                    .unwrap();
                            })
                        }));
                        match r {
                            Ok(r) => {
                                r.print_summary_with_throughput(src_bytes(src_fmt));
                                suite.record(&r);
                            }
                            Err(_) => {
                                println!("  {name:55} [FAILED mid-loop — see stderr]");
                            }
                        }
                    }
                }
            }
        }
    }

    suite.finish();
}
