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
use edgefirst_decoder::{DetectBox, Nms, ProtoData, ProtoLayout, Quantization, Segmentation, XYWH};
use edgefirst_image::{CPUProcessor, ImageProcessor, ImageProcessorTrait, MaskResolution};
use edgefirst_tensor::{DType, PixelFormat, Tensor, TensorDyn, TensorMapTrait, TensorTrait};
use half::f16;
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
        edgefirst_decoder::yolo::MAX_NMS_CANDIDATES,
        300,
        &mut output_boxes,
    );
    (output_boxes, proto_data)
}

/// Materialize `Segmentation` masks from `ProtoData` with clamped bounding
/// boxes. This bypasses the decoder's NORM_LIMIT check (which rejects boxes
/// whose coordinates exceed 1.01) so we always get valid pre-decoded masks
/// for benchmarking `draw_decoded_masks`.
fn materialize_segmentations(detect: &[DetectBox], proto_data: &ProtoData) -> Vec<Segmentation> {
    use edgefirst_tensor::{TensorDyn, TensorMapTrait, TensorTrait};
    // Dequantize protos once to f32 for this benchmark (production uses the
    // native per-dtype kernels in CPUProcessor::materialize_segmentations).
    let proto_shape = proto_data.protos.shape().to_vec();
    let (proto_h, proto_w, num_protos) = (proto_shape[0], proto_shape[1], proto_shape[2]);
    let proto_map = match &proto_data.protos {
        TensorDyn::I8(t) => {
            let m = t.map().unwrap();
            let (scale, zp) = match t.quantization() {
                Some(q) => (q.scale()[0], q.zero_point().map_or(0, |z| z[0])),
                None => (1.0, 0),
            };
            let data: Vec<f32> = m
                .as_slice()
                .iter()
                .map(|v| (*v as f32 - zp as f32) * scale)
                .collect();
            ndarray::Array3::from_shape_vec((proto_h, proto_w, num_protos), data).unwrap()
        }
        TensorDyn::F32(t) => {
            let m = t.map().unwrap();
            ndarray::Array3::from_shape_vec((proto_h, proto_w, num_protos), m.as_slice().to_vec())
                .unwrap()
        }
        other => panic!("unsupported proto dtype for benchmark: {:?}", other.dtype()),
    };

    // Flatten mask_coefficients: [N, num_protos] → one row per detection.
    let coeff_rows: Vec<Vec<f32>> = match &proto_data.mask_coefficients {
        TensorDyn::F32(t) => {
            let m = t.map().unwrap();
            let slice = m.as_slice();
            (0..detect.len())
                .map(|i| slice[i * num_protos..(i + 1) * num_protos].to_vec())
                .collect()
        }
        other => panic!(
            "unsupported mask_coeff dtype for benchmark: {:?}",
            other.dtype()
        ),
    };

    detect
        .iter()
        .zip(coeff_rows.iter())
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
            let roi = proto_map.slice(s![y0..y0 + roi_h, x0..x0 + roi_w, ..]);
            let coeff_arr: ndarray::Array2<f32> =
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
// dtype_dispatch: compare f32 vs f16 mask materialization paths
//
// Builds a `ProtoData` once from the i8 fixture (NMS produces ~30-50 detections
// at our threshold), widens both `protos` and `mask_coefficients` into f32 and
// f16 TensorDyn variants, and times `materialize_segmentations` (proto-res)
// and `materialize_scaled_segmentations` (640×640) through each dispatch.
//
// On builds with `-C target-cpu=cortex-a78ae` (or `-C target-feature=+fp16`),
// the f16 inner loop emits native ARMv8.2 `fcvt`/FMA instructions with single-
// precision accumulators; on baseline aarch64 builds the same loop falls back
// to soft-float `__extendhfsf2` calls per element. Comparing the two builds
// quantifies the fp16 codegen win on Tegra Orin / Cortex-A78AE-class cores.
// =============================================================================

fn build_proto_dtypes(detect: &[DetectBox], proto_data_i8: &ProtoData) -> (ProtoData, ProtoData) {
    // Widen the i8 protos and mask coefficients to f32 once (this matches what
    // the i8 path does internally per-pixel; we hoist it so the f32/f16 bench
    // measures only the materialize kernel, not the dequant).
    let proto_h: usize;
    let proto_w: usize;
    let num_protos: usize;
    let protos_f32_vec: Vec<f32> = match &proto_data_i8.protos {
        TensorDyn::I8(t) => {
            let shape = TensorTrait::shape(t);
            proto_h = shape[0];
            proto_w = shape[1];
            num_protos = shape[2];
            let q = t.quantization().expect("i8 protos must carry quant");
            let scale = q.scale()[0];
            let zp = q.zero_point().map(|z| z[0]).unwrap_or(0) as f32;
            let m = t.map().unwrap();
            m.as_slice()
                .iter()
                .map(|v| (*v as f32 - zp) * scale)
                .collect()
        }
        _ => panic!("expected i8 protos in fixture"),
    };

    let n_det = detect.len();
    let coeffs_f32_vec: Vec<f32> = match &proto_data_i8.mask_coefficients {
        TensorDyn::F32(t) => {
            let m = t.map().unwrap();
            m.as_slice().to_vec()
        }
        TensorDyn::F16(t) => {
            let m = t.map().unwrap();
            m.as_slice().iter().map(|v: &f16| v.to_f32()).collect()
        }
        _ => panic!("unexpected coefficient dtype"),
    };

    let proto_shape = [proto_h, proto_w, num_protos];
    let coeff_shape = [n_det, num_protos];

    let f32_protos = Tensor::<f32>::from_slice(&protos_f32_vec, &proto_shape).unwrap();
    let f32_coeffs = Tensor::<f32>::from_slice(&coeffs_f32_vec, &coeff_shape).unwrap();
    let proto_f32 = ProtoData {
        mask_coefficients: TensorDyn::F32(f32_coeffs),
        protos: TensorDyn::F32(f32_protos),
        layout: ProtoLayout::Nhwc,
    };

    let protos_f16_vec: Vec<f16> = protos_f32_vec.iter().map(|v| f16::from_f32(*v)).collect();
    let coeffs_f16_vec: Vec<f16> = coeffs_f32_vec.iter().map(|v| f16::from_f32(*v)).collect();
    let f16_protos = Tensor::<f16>::from_slice(&protos_f16_vec, &proto_shape).unwrap();
    let f16_coeffs = Tensor::<f16>::from_slice(&coeffs_f16_vec, &coeff_shape).unwrap();
    let proto_f16 = ProtoData {
        mask_coefficients: TensorDyn::F16(f16_coeffs),
        protos: TensorDyn::F16(f16_protos),
        layout: ProtoLayout::Nhwc,
    };

    (proto_f32, proto_f16)
}

fn bench_materialize_dtype(suite: &mut BenchSuite) {
    println!("\n== materialize_masks: dtype dispatch (f32 vs f16) ==\n");
    println!("  Same workload through fused_*_f32_slice and fused_*_f16_slice.");
    println!("  f16 codegen depends on RUSTFLAGS — `-C target-cpu=cortex-a78ae`");
    println!("  enables ARMv8.2-FP16 native instructions; baseline aarch64 emits");
    println!("  soft-float helpers per element.\n");

    let (detect, proto_data_i8) = decode_proto_data();
    if detect.is_empty() {
        println!("  [skipped: no detections from fixture]\n");
        return;
    }
    println!("  N = {} detections (post-NMS)\n", detect.len());

    let (proto_f32, proto_f16) = build_proto_dtypes(&detect, &proto_data_i8);
    let cpu = CPUProcessor::new();

    // Proto-resolution materialize (160×160 per detection).
    {
        let name = "materialize_masks/proto_res/i8";
        let result = run_bench(name, WARMUP, ITERATIONS, || {
            let _ = cpu
                .materialize_segmentations(&detect, &proto_data_i8, None)
                .unwrap();
        });
        result.print_summary();
        suite.record(&result);
    }
    {
        let name = "materialize_masks/proto_res/f32";
        let result = run_bench(name, WARMUP, ITERATIONS, || {
            let _ = cpu
                .materialize_segmentations(&detect, &proto_f32, None)
                .unwrap();
        });
        result.print_summary();
        suite.record(&result);
    }
    {
        let name = "materialize_masks/proto_res/f16";
        let result = run_bench(name, WARMUP, ITERATIONS, || {
            let _ = cpu
                .materialize_segmentations(&detect, &proto_f16, None)
                .unwrap();
        });
        result.print_summary();
        suite.record(&result);
    }

    // Scaled-resolution materialize (640×640 per detection — the COCO-validation
    // critical path; this is where the f16 codegen difference shows up most).
    {
        let name = "materialize_masks/scaled_640x640/i8";
        let result = run_bench(name, WARMUP, ITERATIONS, || {
            let _ = cpu
                .materialize_scaled_segmentations(&detect, &proto_data_i8, None, 640, 640)
                .unwrap();
        });
        result.print_summary();
        suite.record(&result);
    }
    {
        let name = "materialize_masks/scaled_640x640/f32";
        let result = run_bench(name, WARMUP, ITERATIONS, || {
            let _ = cpu
                .materialize_scaled_segmentations(&detect, &proto_f32, None, 640, 640)
                .unwrap();
        });
        result.print_summary();
        suite.record(&result);
    }
    {
        let name = "materialize_masks/scaled_640x640/f16";
        let result = run_bench(name, WARMUP, ITERATIONS, || {
            let _ = cpu
                .materialize_scaled_segmentations(&detect, &proto_f16, None, 640, 640)
                .unwrap();
        });
        result.print_summary();
        suite.record(&result);
    }

    // Reference the unused MaskResolution import to avoid a warning when
    // the helper above is rewritten — currently materialize_segmentations
    // implies Proto and materialize_scaled_segmentations implies Scaled.
    let _ = MaskResolution::Proto;
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
                edgefirst_decoder::yolo::MAX_NMS_CANDIDATES,
                300,
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
                edgefirst_decoder::yolo::MAX_NMS_CANDIDATES,
                300,
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
        edgefirst_decoder::yolo::MAX_NMS_CANDIDATES,
        300,
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
            edgefirst_decoder::yolo::MAX_NMS_CANDIDATES,
            300,
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
    // Log the actual tensor dimensions and stride, not just what we
    // requested — `create_image` silently pads the DMA row stride for
    // GPU pitch alignment, and the diagnostic is meant to surface that
    // difference so a mismatch is visible.
    let actual_w = dst.width().unwrap_or(0);
    let actual_h = dst.height().unwrap_or(0);
    let actual_stride = dst.effective_row_stride().unwrap_or(0);
    let natural_stride = actual_w * 4; // RGBA8
    println!("  dst.memory() = {:?}", dst.memory());
    println!(
        "  dst requested = {dst_w}x{dst_h}, actual = {actual_w}x{actual_h}, \
         stride = {actual_stride} bytes (natural {natural_stride}, padded = {})",
        if actual_stride > natural_stride {
            "yes"
        } else {
            "no"
        }
    );

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
    println!(
        "  call took {:.2} ms (DMA fast path < 10ms; non-DMA fallback ~95ms on Mali)",
        elapsed.as_secs_f64() * 1000.0
    );
    if let Err(e) = result {
        println!("  draw_decoded_masks ERROR: {e:?}");
    } else {
        let t = dst.as_u8().expect("u8 dst");
        let m = t.map().expect("map dst");
        let bytes = m.as_slice();
        let post_sum: u64 = bytes.iter().map(|b| *b as u64).sum();
        let unchanged = bytes.iter().all(|b| *b == 0x42);
        let nonzero_alpha = bytes
            .iter()
            .skip(3)
            .step_by(4)
            .filter(|b| **b > 0 && **b != 0x42)
            .count();
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
    let actual_w2 = dst2.width().unwrap_or(0);
    let actual_h2 = dst2.height().unwrap_or(0);
    let actual_stride2 = dst2.effective_row_stride().unwrap_or(0);
    let natural_stride2 = actual_w2 * 4; // RGBA8
    println!("  dst.memory() = {:?}", dst2.memory());
    println!(
        "  dst requested = {dst_w}x{dst_h}, actual = {actual_w2}x{actual_h2}, \
         stride = {actual_stride2} bytes (natural {natural_stride2}, padded = {})",
        if actual_stride2 > natural_stride2 {
            "yes"
        } else {
            "no"
        }
    );

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
        let nonzero_alpha = bytes
            .iter()
            .skip(3)
            .step_by(4)
            .filter(|b| **b > 0 && **b != 0x42)
            .count();
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

    bench_materialize_dtype(&mut suite);
    bench_decode_masks(&mut suite);
    bench_proto_extraction(&mut suite);
    bench_draw_decoded_masks(&mut proc, &mut suite);
    bench_draw_proto_masks(&mut proc, &mut suite);
    bench_draw_proto_masks_forced_opengl(&mut suite);
    bench_hybrid_materialize_and_draw(&mut proc, &mut suite);

    suite.finish();
    println!("\nDone.");
}
