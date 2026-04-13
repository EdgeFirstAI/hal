// SPDX-FileCopyrightText: Copyright 2025-2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! Mask rendering benchmarks using a custom in-process harness.
//!
//! Benchmarks `draw_proto_masks` (fused proto→overlay) and `draw_decoded_masks`
//! (pre-decoded mask overlay) for both CPU and OpenGL backends.
//!
//! Uses the same fork-free `edgefirst-bench` harness as `pipeline_benchmark.rs`
//! to avoid GPU driver crashes on i.MX8/i.MX95 targets.
//!
//! ## Run benchmarks (host)
//! ```bash
//! cargo bench -p edgefirst-image --bench mask_benchmark
//! ```
//!
//! ## Run benchmarks (on-target, cross-compiled)
//! ```bash
//! ./mask_benchmark --bench
//! ```

mod common;

use common::{run_bench, BenchSuite};

use edgefirst_decoder::yolo::impl_yolo_segdet_quant_proto;
use edgefirst_decoder::{DetectBox, Nms, ProtoData, Quantization, Segmentation, XYWH};
use edgefirst_image::{ImageProcessor, ImageProcessorTrait};
use edgefirst_tensor::{DType, PixelFormat};
use ndarray::s;

const WARMUP: usize = 10;
const ITERATIONS: usize = 100;

// Quantization parameters from the YOLOv8 segmentation test model.
const QUANT_BOXES: Quantization = Quantization {
    scale: 0.019_484_945,
    zero_point: 20,
};
const QUANT_PROTOS: Quantization = Quantization {
    scale: 0.020_889_873,
    zero_point: -115,
};

const SCORE_THRESHOLD: f32 = 0.45;
const IOU_THRESHOLD: f32 = 0.45;

// Output image dimensions (typical YOLO input size).
const OUTPUT_W: usize = 640;
const OUTPUT_H: usize = 640;

/// Embedded test data: YOLOv8 segmentation model outputs.
const BOXES_RAW: &[u8] = include_bytes!("../../../testdata/yolov8_boxes_116x8400.bin");
const PROTOS_RAW: &[u8] = include_bytes!("../../../testdata/yolov8_protos_160x160x32.bin");

fn load_boxes_i8() -> ndarray::Array2<i8> {
    let bytes =
        unsafe { std::slice::from_raw_parts(BOXES_RAW.as_ptr() as *const i8, BOXES_RAW.len()) };
    ndarray::Array2::from_shape_vec((116, 8400), bytes.to_vec()).unwrap()
}

fn load_protos_i8() -> ndarray::Array3<i8> {
    let bytes =
        unsafe { std::slice::from_raw_parts(PROTOS_RAW.as_ptr() as *const i8, PROTOS_RAW.len()) };
    ndarray::Array3::from_shape_vec((160, 160, 32), bytes.to_vec()).unwrap()
}

/// Decode into ProtoData (for `draw_proto_masks` benchmarks).
fn decode_proto_data() -> (Vec<DetectBox>, ProtoData) {
    let boxes = load_boxes_i8();
    let protos = load_protos_i8();
    let mut output_boxes = Vec::with_capacity(50);
    let proto_data = impl_yolo_segdet_quant_proto::<XYWH, _, _>(
        (boxes.view(), QUANT_BOXES),
        (protos.view(), QUANT_PROTOS),
        SCORE_THRESHOLD,
        IOU_THRESHOLD,
        Some(Nms::ClassAgnostic),
        &mut output_boxes,
    );
    (output_boxes, proto_data)
}

/// Materialize `Segmentation` masks from `ProtoData` with clamped bounding
/// boxes. This bypasses the decoder's NORM_LIMIT check (which rejects boxes
/// whose coordinates exceed 1.01) so we always get valid pre-decoded masks
/// for benchmarking `draw_decoded_masks`.
fn materialize_segmentations(detect: &[DetectBox], proto_data: &ProtoData) -> Vec<Segmentation> {
    let protos_f32 = proto_data.protos.as_f32();
    let proto_h = protos_f32.shape()[0];
    let proto_w = protos_f32.shape()[1];
    let num_protos = protos_f32.shape()[2];

    detect
        .iter()
        .zip(proto_data.mask_coefficients.iter())
        .map(|(det, coeff)| {
            // Clamp bbox to [0, 1] to avoid NORM_LIMIT rejection
            let xmin = det.bbox.xmin.clamp(0.0, 1.0);
            let ymin = det.bbox.ymin.clamp(0.0, 1.0);
            let xmax = det.bbox.xmax.clamp(0.0, 1.0);
            let ymax = det.bbox.ymax.clamp(0.0, 1.0);

            // Map to proto-space pixel coordinates
            let x0 = (xmin * proto_w as f32) as usize;
            let y0 = (ymin * proto_h as f32) as usize;
            let x1 = ((xmax * proto_w as f32).ceil() as usize).min(proto_w);
            let y1 = ((ymax * proto_h as f32).ceil() as usize).min(proto_h);

            let roi_w = x1.saturating_sub(x0).max(1);
            let roi_h = y1.saturating_sub(y0).max(1);

            // Extract proto ROI and compute mask_coeff @ protos
            let roi = protos_f32.slice(s![y0..y0 + roi_h, x0..x0 + roi_w, ..]);
            let coeff_arr =
                ndarray::Array2::from_shape_vec((1, num_protos), coeff.clone()).unwrap();
            let protos_2d = roi
                .to_shape((roi_h * roi_w, num_protos))
                .unwrap()
                .reversed_axes();
            let mask = coeff_arr.dot(&protos_2d);
            let mask = mask
                .into_shape_with_order((roi_h, roi_w, 1))
                .unwrap()
                .mapv(|x: f32| {
                    let sigmoid = 1.0 / (1.0 + (-x).exp());
                    (sigmoid * 255.0).round() as u8
                });

            Segmentation {
                xmin: x0 as f32 / proto_w as f32,
                ymin: y0 as f32 / proto_h as f32,
                xmax: x1 as f32 / proto_w as f32,
                ymax: y1 as f32 / proto_h as f32,
                segmentation: mask,
            }
        })
        .collect()
}

// =============================================================================
// decode_masks: CPU decode cost (not rendering)
// =============================================================================

fn bench_decode_masks(suite: &mut BenchSuite) {
    println!("\n== decode_masks: CPU Decode Cost ==\n");

    let boxes = load_boxes_i8();
    let protos = load_protos_i8();

    // Proto-only decode: NMS + extract mask coefficients (no mask materialization).
    // This is the decode cost paid by the fused draw_proto_masks
    // paths which let the renderer compute mask_coeff @ protos.
    {
        let name = "decode_masks/proto";
        let result = run_bench(name, WARMUP, ITERATIONS, || {
            let mut output_boxes = Vec::with_capacity(50);
            let _proto_data = impl_yolo_segdet_quant_proto::<XYWH, _, _>(
                (boxes.view(), QUANT_BOXES),
                (protos.view(), QUANT_PROTOS),
                SCORE_THRESHOLD,
                IOU_THRESHOLD,
                Some(Nms::ClassAgnostic),
                &mut output_boxes,
            );
        });
        result.print_summary();
        suite.record(&result);
    }

    // Full decode: NMS + extract proto data + materialize pixel masks on CPU.
    // This is the decode cost paid by the draw_decoded_masks path.
    {
        let name = "decode_masks/materialize";
        let result = run_bench(name, WARMUP, ITERATIONS, || {
            let mut output_boxes = Vec::with_capacity(50);
            let proto_data = impl_yolo_segdet_quant_proto::<XYWH, _, _>(
                (boxes.view(), QUANT_BOXES),
                (protos.view(), QUANT_PROTOS),
                SCORE_THRESHOLD,
                IOU_THRESHOLD,
                Some(Nms::ClassAgnostic),
                &mut output_boxes,
            );
            let _seg = materialize_segmentations(&output_boxes, &proto_data);
        });
        result.print_summary();
        suite.record(&result);
    }
}

// =============================================================================
// proto_extraction: isolate extract_proto_data_quant cost
// =============================================================================

fn bench_proto_extraction(suite: &mut BenchSuite) {
    println!("\n== proto_extraction: Extract ProtoData Cost ==\n");

    let boxes = load_boxes_i8();
    let protos = load_protos_i8();

    // Run NMS once to get detection indices, then measure only the proto
    // extraction (the copy/dequant we plan to optimize).
    let mut warmup_boxes = Vec::with_capacity(50);
    let _ = impl_yolo_segdet_quant_proto::<XYWH, _, _>(
        (boxes.view(), QUANT_BOXES),
        (protos.view(), QUANT_PROTOS),
        SCORE_THRESHOLD,
        IOU_THRESHOLD,
        Some(Nms::ClassAgnostic),
        &mut warmup_boxes,
    );
    let n_detect = warmup_boxes.len();
    println!("  {n_detect} detections; measuring proto extraction (i8 copy + coeff dequant)\n");

    let name = "proto_extraction";
    let result = run_bench(name, WARMUP, ITERATIONS, || {
        let mut output_boxes = Vec::with_capacity(50);
        let _proto_data = impl_yolo_segdet_quant_proto::<XYWH, _, _>(
            (boxes.view(), QUANT_BOXES),
            (protos.view(), QUANT_PROTOS),
            SCORE_THRESHOLD,
            IOU_THRESHOLD,
            Some(Nms::ClassAgnostic),
            &mut output_boxes,
        );
    });
    result.print_summary();
    suite.record(&result);
}

// =============================================================================
// draw_proto_masks/forced_opengl: pure GL path (no hybrid)
// =============================================================================

fn bench_draw_proto_masks_forced_opengl(suite: &mut BenchSuite) {
    println!("\n== draw_proto_masks/forced_opengl: Pure GL Proto Path ==\n");

    // Create a processor forced to OpenGL only (no hybrid fallback).
    // SAFETY: set_var is not thread-safe, but benchmarks are single-threaded.
    unsafe { std::env::set_var("EDGEFIRST_FORCE_BACKEND", "opengl") };
    let proc = ImageProcessor::new();
    unsafe { std::env::remove_var("EDGEFIRST_FORCE_BACKEND") };

    let Ok(mut proc) = proc else {
        println!("  [skipped: OpenGL ImageProcessor creation failed]");
        return;
    };

    let (detect, proto_data) = decode_proto_data();
    let n_detect = detect.len();
    println!("  Decoded {n_detect} detections for benchmarking\n");

    let name = "draw_proto_masks/forced_opengl";
    let Ok(mut dst) = proc.create_image(OUTPUT_W, OUTPUT_H, PixelFormat::Rgba, DType::U8, None)
    else {
        println!("  {:50} [skipped: allocation failed]", name);
        return;
    };

    if let Err(e) = proc.draw_proto_masks(&mut dst, &detect, &proto_data, Default::default()) {
        println!("  {:50} [unsupported: {}]", name, e);
        return;
    }

    let result = run_bench(name, WARMUP, ITERATIONS, || {
        proc.draw_proto_masks(&mut dst, &detect, &proto_data, Default::default())
            .unwrap();
    });
    result.print_summary();
    suite.record(&result);
}

// =============================================================================
// draw_decoded_masks: pre-decoded mask overlay
// =============================================================================

fn bench_draw_decoded_masks(proc: &mut ImageProcessor, suite: &mut BenchSuite) {
    println!("\n== draw_decoded_masks: Pre-decoded Mask Overlay ==\n");

    let (detect, proto_data) = decode_proto_data();
    let segmentation = materialize_segmentations(&detect, &proto_data);
    let n_detect = detect.len();
    println!("  Materialized {n_detect} detection masks for benchmarking\n");

    let name = "draw_decoded_masks";
    let Ok(mut dst) = proc.create_image(OUTPUT_W, OUTPUT_H, PixelFormat::Rgba, DType::U8, None)
    else {
        println!("  {:50} [skipped: allocation failed]", name);
        return;
    };

    if let Err(e) = proc.draw_decoded_masks(&mut dst, &detect, &segmentation, Default::default()) {
        println!("  {:50} [unsupported: {}]", name, e);
        return;
    }

    let result = run_bench(name, WARMUP, ITERATIONS, || {
        proc.draw_decoded_masks(&mut dst, &detect, &segmentation, Default::default())
            .unwrap();
    });
    result.print_summary();
    suite.record(&result);
}

// =============================================================================
// draw_proto_masks: fused proto → overlay
// =============================================================================

fn bench_draw_proto_masks(proc: &mut ImageProcessor, suite: &mut BenchSuite) {
    println!("\n== draw_proto_masks: Fused Proto → Overlay ==\n");

    let (detect, proto_data) = decode_proto_data();
    let n_detect = detect.len();
    println!("  Decoded {n_detect} detections for benchmarking\n");

    let name = "draw_proto_masks";
    let Ok(mut dst) = proc.create_image(OUTPUT_W, OUTPUT_H, PixelFormat::Rgba, DType::U8, None)
    else {
        println!("  {:50} [skipped: allocation failed]", name);
        return;
    };

    if let Err(e) = proc.draw_proto_masks(&mut dst, &detect, &proto_data, Default::default()) {
        println!("  {:50} [unsupported: {}]", name, e);
        return;
    }

    let result = run_bench(name, WARMUP, ITERATIONS, || {
        proc.draw_proto_masks(&mut dst, &detect, &proto_data, Default::default())
            .unwrap();
    });
    result.print_summary();
    suite.record(&result);
}

// =============================================================================
// hybrid_materialize_and_draw: CPU decode + GPU/CPU overlay
// =============================================================================

fn bench_hybrid_materialize_and_draw(proc: &mut ImageProcessor, suite: &mut BenchSuite) {
    println!("\n== hybrid_materialize_and_draw: CPU Decode + Overlay ==\n");

    let (detect, proto_data) = decode_proto_data();
    let n_detect = detect.len();
    println!("  Decoded {n_detect} detections for benchmarking\n");

    let name = "hybrid_materialize_and_draw";
    let Ok(mut dst) = proc.create_image(OUTPUT_W, OUTPUT_H, PixelFormat::Rgba, DType::U8, None)
    else {
        println!("  {:50} [skipped: allocation failed]", name);
        return;
    };

    // Verify the hybrid path works before benchmarking.
    let segmentation = materialize_segmentations(&detect, &proto_data);
    if let Err(e) = proc.draw_decoded_masks(&mut dst, &detect, &segmentation, Default::default()) {
        println!("  {:50} [unsupported: {}]", name, e);
        return;
    }

    let result = run_bench(name, WARMUP, ITERATIONS, || {
        let segmentation = materialize_segmentations(&detect, &proto_data);
        proc.draw_decoded_masks(&mut dst, &detect, &segmentation, Default::default())
            .unwrap();
    });
    result.print_summary();
    suite.record(&result);
}

// =============================================================================
// Main
// =============================================================================

/// Diagnostic mode (enabled when `MASK_BENCH_DIAG=1`, `true`, or `yes`) —
/// bypass the timing loop, allocate a destination via
/// `processor.create_image`, log its `dst.memory()` (so we know whether DMA
/// actually worked), then call both `draw_decoded_masks` and
/// `draw_proto_masks` once each and dump the rendered output to
/// `/tmp/mask_decoded.rgba` and `/tmp/mask_proto.rgba` for visual
/// verification. Any other value of the env var (including `0` and `false`)
/// leaves the bench in normal timing-loop mode.
///
/// This is the diagnostic path for the "draw_decoded_masks doesn't modify
/// the destination on i.MX 95" bug. It uses embedded test data so it
/// doesn't need the NPU and runs on any target the bench binary
/// cross-compiles to.
fn run_diagnostic(proc: &mut ImageProcessor) {
    use edgefirst_tensor::{TensorMapTrait, TensorTrait};

    // Diagnostic dst dims are configurable via env var so we can repro
    // the imx95 Kinara example's setup which uses canvas = source image
    // dimensions (e.g. 3004×1688 for crowd.png).
    let dst_w: usize = std::env::var("MASK_BENCH_DST_W")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(OUTPUT_W);
    let dst_h: usize = std::env::var("MASK_BENCH_DST_H")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(OUTPUT_H);

    println!("\n== DIAGNOSTIC: draw_decoded_masks vs draw_proto_masks ==");
    println!("  dst dims: {}x{}", dst_w, dst_h);
    let (detect, proto_data) = decode_proto_data();
    let segmentation = materialize_segmentations(&detect, &proto_data);
    println!("  Detections: {}", detect.len());

    // ---- draw_decoded_masks ----
    println!("\n--- draw_decoded_masks ---");
    let mut dst = match proc.create_image(dst_w, dst_h, PixelFormat::Rgba, DType::U8, None) {
        Ok(t) => t,
        Err(e) => {
            println!("  create_image failed: {e:?}");
            return;
        }
    };
    println!("  dst.memory() = {:?}", dst.memory());
    println!("  dst dims = {}x{}", dst_w, dst_h);

    // Pre-fill dst with a known sentinel so we can detect "untouched" buffers.
    {
        let t = dst.as_u8_mut().expect("u8 dst");
        let mut m = t.map().expect("map dst");
        for px in m.as_mut_slice().chunks_exact_mut(4) {
            px[0] = 0x42;
            px[1] = 0x42;
            px[2] = 0x42;
            px[3] = 0x42;
        }
    }
    let pre_sum: u64 = {
        let t = dst.as_u8().expect("u8 dst");
        let m = t.map().expect("map dst");
        m.as_slice().iter().map(|b| *b as u64).sum()
    };

    let t0 = std::time::Instant::now();
    let result = proc.draw_decoded_masks(&mut dst, &detect, &segmentation, Default::default());
    let elapsed = t0.elapsed();
    println!("  call took {:.2} ms (DMA fast path < 10ms; non-DMA fallback ~95ms on Mali)", elapsed.as_secs_f64() * 1000.0);
    if let Err(e) = result {
        println!("  draw_decoded_masks ERROR: {e:?}");
    } else {
        let t = dst.as_u8().expect("u8 dst");
        let m = t.map().expect("map dst");
        let bytes = m.as_slice();
        let post_sum: u64 = bytes.iter().map(|b| *b as u64).sum();
        let unchanged = bytes.iter().all(|b| *b == 0x42);
        let nonzero_alpha = bytes.iter().skip(3).step_by(4).filter(|b| **b > 0 && **b != 0x42).count();
        println!("  pre_sum  = {pre_sum}");
        println!("  post_sum = {post_sum}");
        println!("  changed  = {}", !unchanged);
        println!("  non-sentinel alpha pixels = {nonzero_alpha}");
        if let Err(e) = std::fs::write("/tmp/mask_decoded.rgba", bytes) {
            println!("  failed to dump /tmp/mask_decoded.rgba: {e:?}");
        } else {
            println!("  wrote /tmp/mask_decoded.rgba ({} bytes)", bytes.len());
        }
    }

    // ---- draw_proto_masks (fused) ----
    println!("\n--- draw_proto_masks (fused) ---");
    let mut dst2 = match proc.create_image(dst_w, dst_h, PixelFormat::Rgba, DType::U8, None) {
        Ok(t) => t,
        Err(e) => {
            println!("  create_image failed: {e:?}");
            return;
        }
    };
    println!("  dst.memory() = {:?}", dst2.memory());

    {
        let t = dst2.as_u8_mut().expect("u8 dst2");
        let mut m = t.map().expect("map dst2");
        for px in m.as_mut_slice().chunks_exact_mut(4) {
            px[0] = 0x42;
            px[1] = 0x42;
            px[2] = 0x42;
            px[3] = 0x42;
        }
    }
    let t0 = std::time::Instant::now();
    let result = proc.draw_proto_masks(&mut dst2, &detect, &proto_data, Default::default());
    let elapsed = t0.elapsed();
    println!("  call took {:.2} ms", elapsed.as_secs_f64() * 1000.0);
    if let Err(e) = result {
        println!("  draw_proto_masks ERROR: {e:?}");
    } else {
        let t = dst2.as_u8().expect("u8 dst2");
        let m = t.map().expect("map dst2");
        let bytes = m.as_slice();
        let post_sum: u64 = bytes.iter().map(|b| *b as u64).sum();
        let unchanged = bytes.iter().all(|b| *b == 0x42);
        let nonzero_alpha = bytes.iter().skip(3).step_by(4).filter(|b| **b > 0 && **b != 0x42).count();
        println!("  post_sum = {post_sum}");
        println!("  changed  = {}", !unchanged);
        println!("  non-sentinel alpha pixels = {nonzero_alpha}");
        if let Err(e) = std::fs::write("/tmp/mask_proto.rgba", bytes) {
            println!("  failed to dump /tmp/mask_proto.rgba: {e:?}");
        } else {
            println!("  wrote /tmp/mask_proto.rgba ({} bytes)", bytes.len());
        }
    }
    println!();
}

fn main() {
    env_logger::init();
    let mut suite = BenchSuite::from_args();
    let mut proc = ImageProcessor::new().expect("Failed to create ImageProcessor");

    let diag_enabled = std::env::var("MASK_BENCH_DIAG")
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true") || v.eq_ignore_ascii_case("yes"))
        .unwrap_or(false);
    if diag_enabled {
        run_diagnostic(&mut proc);
        return;
    }

    println!("Mask Rendering Benchmark — edgefirst-bench harness");
    println!("  warmup={WARMUP}  iterations={ITERATIONS}");

    bench_decode_masks(&mut suite);
    bench_proto_extraction(&mut suite);
    bench_draw_decoded_masks(&mut proc, &mut suite);
    bench_draw_proto_masks(&mut proc, &mut suite);
    bench_draw_proto_masks_forced_opengl(&mut suite);
    bench_hybrid_materialize_and_draw(&mut proc, &mut suite);

    suite.finish();
    println!("\nDone.");
}
