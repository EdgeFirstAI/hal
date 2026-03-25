// SPDX-FileCopyrightText: Copyright 2025-2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! Mask rendering benchmarks using a custom in-process harness.
//!
//! Benchmarks `draw_masks_proto` (fused proto→overlay), `decode_masks_atlas`
//! (proto→pixel atlas), and `draw_masks` (pre-decoded mask overlay)
//! for both CPU and OpenGL backends.
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
const ITERATIONS: usize = 200;

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

/// Decode into ProtoData (for `draw_masks_proto` and `decode_masks_atlas` benchmarks).
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
/// for benchmarking `draw_masks`.
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
    // This is the decode cost paid by the fused draw_masks_proto / decode_masks_atlas
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
    // This is the decode cost paid by the draw_masks path.
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
// draw_masks_proto/forced_opengl: pure GL path (no hybrid)
// =============================================================================

fn bench_draw_masks_proto_forced_opengl(suite: &mut BenchSuite) {
    println!("\n== draw_masks_proto/forced_opengl: Pure GL Proto Path ==\n");

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

    let name = "draw_masks_proto/forced_opengl";
    let Ok(mut dst) = proc.create_image(OUTPUT_W, OUTPUT_H, PixelFormat::Rgba, DType::U8, None)
    else {
        println!("  {:50} [skipped: allocation failed]", name);
        return;
    };

    if let Err(e) = proc.draw_masks_proto(&mut dst, &detect, &proto_data, Default::default()) {
        println!("  {:50} [unsupported: {}]", name, e);
        return;
    }

    let result = run_bench(name, WARMUP, ITERATIONS, || {
        proc.draw_masks_proto(&mut dst, &detect, &proto_data, Default::default())
            .unwrap();
    });
    result.print_summary();
    suite.record(&result);
}

// =============================================================================
// draw_masks: pre-decoded mask overlay
// =============================================================================

fn bench_draw_masks(proc: &mut ImageProcessor, suite: &mut BenchSuite) {
    println!("\n== draw_masks: Pre-decoded Mask Overlay ==\n");

    let (detect, proto_data) = decode_proto_data();
    let segmentation = materialize_segmentations(&detect, &proto_data);
    let n_detect = detect.len();
    println!("  Materialized {n_detect} detection masks for benchmarking\n");

    let name = "draw_masks";
    let Ok(mut dst) = proc.create_image(OUTPUT_W, OUTPUT_H, PixelFormat::Rgba, DType::U8, None)
    else {
        println!("  {:50} [skipped: allocation failed]", name);
        return;
    };

    if let Err(e) = proc.draw_masks(&mut dst, &detect, &segmentation, Default::default()) {
        println!("  {:50} [unsupported: {}]", name, e);
        return;
    }

    let result = run_bench(name, WARMUP, ITERATIONS, || {
        proc.draw_masks(&mut dst, &detect, &segmentation, Default::default())
            .unwrap();
    });
    result.print_summary();
    suite.record(&result);
}

// =============================================================================
// draw_masks_proto: fused proto → overlay
// =============================================================================

fn bench_draw_masks_proto(proc: &mut ImageProcessor, suite: &mut BenchSuite) {
    println!("\n== draw_masks_proto: Fused Proto → Overlay ==\n");

    let (detect, proto_data) = decode_proto_data();
    let n_detect = detect.len();
    println!("  Decoded {n_detect} detections for benchmarking\n");

    let name = "draw_masks_proto";
    let Ok(mut dst) = proc.create_image(OUTPUT_W, OUTPUT_H, PixelFormat::Rgba, DType::U8, None)
    else {
        println!("  {:50} [skipped: allocation failed]", name);
        return;
    };

    if let Err(e) = proc.draw_masks_proto(&mut dst, &detect, &proto_data, Default::default()) {
        println!("  {:50} [unsupported: {}]", name, e);
        return;
    }

    let result = run_bench(name, WARMUP, ITERATIONS, || {
        proc.draw_masks_proto(&mut dst, &detect, &proto_data, Default::default())
            .unwrap();
    });
    result.print_summary();
    suite.record(&result);
}

// =============================================================================
// decode_masks_atlas: proto → pixel atlas
// =============================================================================

fn bench_decode_masks_atlas(proc: &mut ImageProcessor, suite: &mut BenchSuite) {
    println!("\n== decode_masks_atlas: Proto → Pixel Atlas ==\n");

    let (detect, proto_data) = decode_proto_data();
    let n_detect = detect.len();
    println!("  Decoded {n_detect} detections for benchmarking\n");

    let name = "decode_masks_atlas";
    // Note: proto_data.clone() is required because decode_masks_atlas consumes ProtoData by value.
    if let Err(e) = proc.decode_masks_atlas(&detect, proto_data.clone(), OUTPUT_W, OUTPUT_H) {
        println!("  {:50} [unsupported: {}]", name, e);
        return;
    }

    let result = run_bench(name, WARMUP, ITERATIONS, || {
        let _atlas = proc
            .decode_masks_atlas(&detect, proto_data.clone(), OUTPUT_W, OUTPUT_H)
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
    if let Err(e) = proc.draw_masks(&mut dst, &detect, &segmentation, Default::default()) {
        println!("  {:50} [unsupported: {}]", name, e);
        return;
    }

    let result = run_bench(name, WARMUP, ITERATIONS, || {
        let segmentation = materialize_segmentations(&detect, &proto_data);
        proc.draw_masks(&mut dst, &detect, &segmentation, Default::default())
            .unwrap();
    });
    result.print_summary();
    suite.record(&result);
}

// =============================================================================
// Main
// =============================================================================

fn main() {
    env_logger::init();
    let mut suite = BenchSuite::from_args();
    let mut proc = ImageProcessor::new().expect("Failed to create ImageProcessor");

    println!("Mask Rendering Benchmark — edgefirst-bench harness");
    println!("  warmup={WARMUP}  iterations={ITERATIONS}");

    bench_decode_masks(&mut suite);
    bench_proto_extraction(&mut suite);
    bench_draw_masks(&mut proc, &mut suite);
    bench_draw_masks_proto(&mut proc, &mut suite);
    bench_draw_masks_proto_forced_opengl(&mut suite);
    bench_decode_masks_atlas(&mut proc, &mut suite);
    bench_hybrid_materialize_and_draw(&mut proc, &mut suite);

    suite.finish();
    println!("\nDone.");
}
