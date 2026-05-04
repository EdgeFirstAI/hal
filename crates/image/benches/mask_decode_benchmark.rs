// SPDX-FileCopyrightText: Copyright 2025-2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! Focused mask decode benchmark for the RETINA (scaled) path.
//!
//! Benchmarks `materialize_scaled_segmentations` — the COCO-eval critical path
//! that upsamples i8 proto-plane logits to original-image resolution via
//! bilinear interpolation + binary threshold.
//!
//! ## Test data
//!
//! Uses `testdata/mask_decode_scaled.safetensors` containing post-NMS tensors
//! (protos, coefficients, boxes) captured from a real YOLOv8n-seg inference.
//! If the file doesn't exist, falls back to decoding from the raw `.bin`
//! fixtures and generates the safetensors file for future runs.
//!
//! ## Run benchmarks (host)
//! ```bash
//! cargo bench -p edgefirst-image --bench mask_decode_benchmark
//! ```
//!
//! ## Run benchmarks (on-target, cross-compiled)
//! ```bash
//! ./mask_decode_benchmark --bench
//! ```
//!
//! ## Run under perf (on-target)
//! ```bash
//! perf stat ./mask_decode_benchmark --bench
//! perf record -g ./mask_decode_benchmark --bench
//! ```

mod common;

use common::{run_bench, BenchSuite};

use edgefirst_decoder::yolo::impl_yolo_segdet_quant_proto;
use edgefirst_decoder::{DetectBox, Nms, ProtoData, Quantization, XYWH};
use edgefirst_image::CPUProcessor;
use edgefirst_tensor::{Tensor, TensorDyn, TensorMapTrait, TensorTrait};

use std::path::Path;

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

// Use a low threshold matching COCO evaluation (ara2-validator --with-masks).
// This produces many detections (up to ~100+ per image) — the worst-case that
// we need to optimize for. NMS IoU stays at 0.45 (standard).
const SCORE_THRESHOLD: f32 = 0.001;
const IOU_THRESHOLD: f32 = 0.45;

// Typical COCO image dimensions (non-square, letterboxed into 640×640 model input).
const COCO_W: u32 = 640;
const COCO_H: u32 = 480;

// Letterbox for a 640×480 image in a 640×640 model input:
// pad_x=0, pad_y=80, content occupies y=[80..560] in model space.
const LETTERBOX_640X480: [f32; 4] = [0.0, 80.0 / 640.0, 1.0, 560.0 / 640.0];

/// Path to safetensors test data.
fn safetensors_path() -> std::path::PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../testdata/mask_decode_scaled.safetensors")
}

/// Embedded test data: YOLOv8 segmentation model outputs (fallback).
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

/// Run NMS decode to get ProtoData + DetectBox from raw fixtures.
fn decode_from_fixtures() -> (Vec<DetectBox>, ProtoData) {
    let boxes = load_boxes_i8();
    let protos = load_protos_i8();
    let mut output_boxes = Vec::with_capacity(100);
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

/// Try to load pre-decoded data from safetensors. If the file doesn't exist,
/// falls back to decoding from raw fixtures and optionally saves the result.
fn load_test_data() -> (Vec<DetectBox>, ProtoData) {
    let st_path = safetensors_path();
    if st_path.exists() {
        match load_from_safetensors(&st_path) {
            Ok(data) => return data,
            Err(e) => {
                eprintln!("  [warn] Failed to load safetensors: {e}; falling back to .bin decode");
            }
        }
    }

    // Fall back: decode from raw .bin fixtures.
    let (detect, proto_data) = decode_from_fixtures();

    // Try to save for future runs (non-fatal if it fails).
    if let Err(e) = save_to_safetensors(&st_path, &detect, &proto_data) {
        eprintln!("  [warn] Could not save safetensors: {e}");
    }

    (detect, proto_data)
}

/// Load pre-decoded ProtoData + DetectBox from a safetensors file.
fn load_from_safetensors(path: &Path) -> Result<(Vec<DetectBox>, ProtoData), String> {
    let data = std::fs::read(path).map_err(|e| format!("read: {e}"))?;
    let st = safetensors::SafeTensors::deserialize(&data).map_err(|e| format!("parse: {e}"))?;

    // Load protos [160, 160, 32] i8
    let protos_view = st.tensor("protos").map_err(|e| format!("protos: {e}"))?;
    let protos_shape: Vec<usize> = protos_view.shape().to_vec();
    if protos_shape != [160, 160, 32] {
        return Err(format!("unexpected protos shape: {protos_shape:?}"));
    }
    let protos_bytes = protos_view.data();
    let protos_i8: Vec<i8> = protos_bytes.iter().map(|b| *b as i8).collect();
    let protos_tensor = Tensor::<i8>::from_slice(&protos_i8, &protos_shape)
        .map_err(|e| format!("protos tensor: {e}"))?;

    // Use the well-known quantization constants (stored in safetensors
    // metadata for documentation, but constants are authoritative).
    let quant =
        edgefirst_tensor::Quantization::per_tensor(QUANT_PROTOS.scale, QUANT_PROTOS.zero_point);
    let protos_tensor = protos_tensor
        .with_quantization(quant)
        .map_err(|e| format!("set quant: {e}"))?;

    // Load coefficients [N, 32] — may be i8 (new format) or f32 (legacy).
    let coeff_view = st
        .tensor("coefficients")
        .map_err(|e| format!("coefficients: {e}"))?;
    let coeff_shape: Vec<usize> = coeff_view.shape().to_vec();
    let n_det = coeff_shape[0];
    let coeff_bytes = coeff_view.data();

    let mask_coefficients: TensorDyn = if coeff_view.dtype() == safetensors::Dtype::I8 {
        // New format: raw i8 with quantization.
        let coeff_i8: Vec<i8> = coeff_bytes.iter().map(|b| *b as i8).collect();
        let t = Tensor::<i8>::from_slice(&coeff_i8, &coeff_shape)
            .map_err(|e| format!("coeff tensor: {e}"))?;
        let q =
            edgefirst_tensor::Quantization::per_tensor(QUANT_BOXES.scale, QUANT_BOXES.zero_point);
        let t = t
            .with_quantization(q)
            .map_err(|e| format!("coeff quant: {e}"))?;
        TensorDyn::I8(t)
    } else {
        // Legacy f32 format.
        let coeff_f32: Vec<f32> = coeff_bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        let t = Tensor::<f32>::from_slice(&coeff_f32, &coeff_shape)
            .map_err(|e| format!("coeff tensor: {e}"))?;
        TensorDyn::F32(t)
    };

    // Load boxes [N, 4] f32 (normalized xyxy)
    let boxes_view = st.tensor("boxes").map_err(|e| format!("boxes: {e}"))?;
    let boxes_bytes = boxes_view.data();
    let boxes_f32: Vec<f32> = boxes_bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();

    // Load scores [N] f32
    let scores_view = st.tensor("scores").map_err(|e| format!("scores: {e}"))?;
    let scores_bytes = scores_view.data();
    let scores_f32: Vec<f32> = scores_bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();

    // Reconstruct DetectBox array.
    let mut detect: Vec<DetectBox> = Vec::with_capacity(n_det);
    for i in 0..n_det {
        let xmin = boxes_f32[i * 4];
        let ymin = boxes_f32[i * 4 + 1];
        let xmax = boxes_f32[i * 4 + 2];
        let ymax = boxes_f32[i * 4 + 3];
        detect.push(DetectBox {
            bbox: edgefirst_decoder::BoundingBox::new(xmin, ymin, xmax, ymax),
            score: scores_f32[i],
            label: 0, // class doesn't affect mask decode
        });
    }

    let proto_data = ProtoData {
        mask_coefficients,
        protos: TensorDyn::I8(protos_tensor),
    };

    Ok((detect, proto_data))
}

/// Save post-NMS data to safetensors for future benchmark runs.
fn save_to_safetensors(
    path: &Path,
    detect: &[DetectBox],
    proto_data: &ProtoData,
) -> Result<(), String> {
    use std::collections::HashMap;

    let n_det = detect.len();

    // Extract protos as raw bytes.
    let protos_i8 = match &proto_data.protos {
        TensorDyn::I8(t) => {
            let m = t.map().map_err(|e| format!("map protos: {e}"))?;
            m.as_slice().to_vec()
        }
        _ => return Err("expected I8 protos".into()),
    };
    let proto_shape = proto_data.protos.shape().to_vec();

    // Extract coefficients (may be I8 or F32).
    let coeff_i8_data: Vec<i8>;
    let coeff_f32_data: Vec<f32>;
    let (coeff_raw_bytes, coeff_dtype): (Vec<u8>, safetensors::Dtype) =
        match &proto_data.mask_coefficients {
            TensorDyn::I8(t) => {
                let m = t.map().map_err(|e| format!("map coeffs: {e}"))?;
                coeff_i8_data = m.as_slice().to_vec();
                (
                    coeff_i8_data.iter().map(|v| *v as u8).collect(),
                    safetensors::Dtype::I8,
                )
            }
            TensorDyn::F32(t) => {
                let m = t.map().map_err(|e| format!("map coeffs: {e}"))?;
                coeff_f32_data = m.as_slice().to_vec();
                (
                    coeff_f32_data
                        .iter()
                        .flat_map(|v| v.to_le_bytes())
                        .collect(),
                    safetensors::Dtype::F32,
                )
            }
            _ => return Err("expected I8 or F32 coefficients".into()),
        };
    let coeff_shape = proto_data.mask_coefficients.shape().to_vec();
    let num_protos = coeff_shape[1];

    // Extract boxes and scores.
    let mut boxes_f32 = Vec::with_capacity(n_det * 4);
    let mut scores_f32 = Vec::with_capacity(n_det);
    for det in detect {
        let bbox = det.bbox.to_canonical();
        boxes_f32.extend_from_slice(&[bbox.xmin, bbox.ymin, bbox.xmax, bbox.ymax]);
        scores_f32.push(det.score);
    }

    // Build raw tensor data map.
    let mut tensors: Vec<(String, safetensors::tensor::TensorView<'_>)> = Vec::new();

    let protos_bytes: Vec<u8> = protos_i8.iter().map(|v| *v as u8).collect();
    tensors.push((
        "protos".to_string(),
        safetensors::tensor::TensorView::new(
            safetensors::Dtype::I8,
            proto_shape.clone(),
            &protos_bytes,
        )
        .map_err(|e| format!("protos view: {e}"))?,
    ));

    tensors.push((
        "coefficients".to_string(),
        safetensors::tensor::TensorView::new(coeff_dtype, coeff_shape.clone(), &coeff_raw_bytes)
            .map_err(|e| format!("coeff view: {e}"))?,
    ));

    let boxes_bytes: Vec<u8> = boxes_f32.iter().flat_map(|v| v.to_le_bytes()).collect();
    let boxes_shape = vec![n_det, 4];
    tensors.push((
        "boxes".to_string(),
        safetensors::tensor::TensorView::new(safetensors::Dtype::F32, boxes_shape, &boxes_bytes)
            .map_err(|e| format!("boxes view: {e}"))?,
    ));

    let scores_bytes: Vec<u8> = scores_f32.iter().flat_map(|v| v.to_le_bytes()).collect();
    let scores_shape = vec![n_det];
    tensors.push((
        "scores".to_string(),
        safetensors::tensor::TensorView::new(safetensors::Dtype::F32, scores_shape, &scores_bytes)
            .map_err(|e| format!("scores view: {e}"))?,
    ));

    // Metadata with quantization info.
    let mut metadata = HashMap::new();
    metadata.insert("proto_scale".to_string(), QUANT_PROTOS.scale.to_string());
    metadata.insert(
        "proto_zero_point".to_string(),
        QUANT_PROTOS.zero_point.to_string(),
    );
    metadata.insert("proto_height".to_string(), proto_shape[0].to_string());
    metadata.insert("proto_width".to_string(), proto_shape[1].to_string());
    metadata.insert("num_protos".to_string(), num_protos.to_string());
    metadata.insert("n_detections".to_string(), n_det.to_string());
    metadata.insert(
        "description".to_string(),
        "YOLOv8n-seg post-NMS test data for mask decode benchmark".to_string(),
    );

    let serialized = safetensors::tensor::serialize(tensors, &Some(metadata))
        .map_err(|e| format!("serialize: {e}"))?;

    std::fs::write(path, &serialized).map_err(|e| format!("write: {e}"))?;
    println!(
        "  Saved safetensors: {} ({} KB)",
        path.display(),
        serialized.len() / 1024
    );
    Ok(())
}

// =============================================================================
// Benchmarks
// =============================================================================

/// Benchmark the full scaled materialization path at typical COCO resolution.
fn bench_scaled_materialize(suite: &mut BenchSuite) {
    println!("\n== materialize_scaled: RETINA path (i8 protos, bilinear upsample) ==\n");

    let (detect, proto_data) = load_test_data();
    if detect.is_empty() {
        println!("  [skipped: no detections from fixture]\n");
        return;
    }
    let n = detect.len();
    println!("  N = {n} detections (post-NMS)");
    println!(
        "  Proto tensor: {:?} {:?}",
        proto_data.protos.shape(),
        proto_data.protos.dtype()
    );
    println!(
        "  Coeff tensor: {:?} {:?}",
        proto_data.mask_coefficients.shape(),
        proto_data.mask_coefficients.dtype()
    );

    let cpu = CPUProcessor::new();

    // 1. Scaled 640×480 (typical COCO, with letterbox)
    {
        let name = "scaled_640x480_letterbox/i8";
        println!("\n  Benchmark: {name} (N={n})");
        let result = run_bench(name, WARMUP, ITERATIONS, || {
            let _ = cpu
                .materialize_scaled_segmentations(
                    &detect,
                    &proto_data,
                    Some(LETTERBOX_640X480),
                    COCO_W,
                    COCO_H,
                )
                .unwrap();
        });
        result.print_summary();
        suite.record(&result);
    }

    // 2. Scaled 640×640 (square, no letterbox)
    {
        let name = "scaled_640x640/i8";
        println!("\n  Benchmark: {name} (N={n})");
        let result = run_bench(name, WARMUP, ITERATIONS, || {
            let _ = cpu
                .materialize_scaled_segmentations(&detect, &proto_data, None, 640, 640)
                .unwrap();
        });
        result.print_summary();
        suite.record(&result);
    }

    // 3. Scaled 1080×810 (high-res COCO variant)
    {
        let name = "scaled_1080x810_letterbox/i8";
        // Letterbox for 1080×810 in 640×640: scale=640/1080=0.593, pad_y=(640-810*0.593)/2=...
        // But we're materializing *output* at that res; letterbox is the same model-input transform.
        let result = run_bench(name, WARMUP, ITERATIONS, || {
            let _ = cpu
                .materialize_scaled_segmentations(
                    &detect,
                    &proto_data,
                    Some(LETTERBOX_640X480),
                    1080,
                    810,
                )
                .unwrap();
        });
        result.print_summary();
        suite.record(&result);
    }

    // 4. Proto-res (baseline for comparison — no bilinear upsample)
    {
        let name = "proto_res_160x160/i8";
        println!("\n  Benchmark: {name} (N={n}) — baseline without upsample");
        let result = run_bench(name, WARMUP, ITERATIONS, || {
            let _ = cpu
                .materialize_segmentations(&detect, &proto_data, None)
                .unwrap();
        });
        result.print_summary();
        suite.record(&result);
    }
}

/// Benchmark with varying detection counts to understand scaling behaviour.
fn bench_detection_count_scaling(suite: &mut BenchSuite) {
    println!("\n== detection_count_scaling: N vs latency ==\n");

    let (detect_all, proto_data_all) = load_test_data();
    if detect_all.is_empty() {
        println!("  [skipped: no detections]\n");
        return;
    }

    let cpu = CPUProcessor::new();
    let num_protos = proto_data_all.mask_coefficients.shape()[1];

    // Extract all coefficients once for subsetting.
    let all_coeff_i8: Vec<i8> = match &proto_data_all.mask_coefficients {
        TensorDyn::I8(t) => {
            let m = t.map().unwrap();
            m.as_slice().to_vec()
        }
        _ => panic!("expected I8 coefficients"),
    };
    let coeff_quant = match &proto_data_all.mask_coefficients {
        TensorDyn::I8(t) => t.quantization().unwrap().clone(),
        _ => panic!("expected I8 coefficients"),
    };

    // Extract protos once for reuse.
    let protos_i8: Vec<i8> = match &proto_data_all.protos {
        TensorDyn::I8(t) => {
            let m = t.map().unwrap();
            m.as_slice().to_vec()
        }
        _ => panic!("expected I8 protos"),
    };
    let proto_shape = proto_data_all.protos.shape().to_vec();
    let proto_quant = match &proto_data_all.protos {
        TensorDyn::I8(t) => t.quantization().unwrap().clone(),
        _ => panic!("expected I8 protos"),
    };

    // Test with subsets of detections.
    for &n in &[1, 5, 10, 20, 30, 50] {
        if n > detect_all.len() {
            break;
        }
        let detect = &detect_all[..n];

        // Build a ProtoData with only N rows of coefficients.
        let coeff_subset = &all_coeff_i8[..n * num_protos];
        let coeff_tensor = Tensor::<i8>::from_slice(coeff_subset, &[n, num_protos]).unwrap();
        let coeff_tensor = coeff_tensor.with_quantization(coeff_quant.clone()).unwrap();

        let protos_tensor = Tensor::<i8>::from_slice(&protos_i8, &proto_shape).unwrap();
        let protos_tensor = protos_tensor
            .with_quantization(proto_quant.clone())
            .unwrap();

        let proto_data = ProtoData {
            mask_coefficients: TensorDyn::I8(coeff_tensor),
            protos: TensorDyn::I8(protos_tensor),
        };

        let name = format!("scaled_640x480/N={n}");
        let result = run_bench(&name, WARMUP, ITERATIONS, || {
            let _ = cpu
                .materialize_scaled_segmentations(
                    detect,
                    &proto_data,
                    Some(LETTERBOX_640X480),
                    COCO_W,
                    COCO_H,
                )
                .unwrap();
        });
        result.print_summary();
        suite.record(&result);
    }
}

/// Isolate the logit precompute vs bilinear upsample costs.
/// Runs the full path but also measures proto-res (which is just the dot product)
/// to estimate the upsample overhead.
fn bench_phase_breakdown(suite: &mut BenchSuite) {
    println!("\n== phase_breakdown: logit precompute vs bilinear upsample ==\n");
    println!("  Proto-res = logit dot product only (no bilinear).");
    println!("  Scaled - Proto-res ≈ bilinear upsample + threshold cost.\n");

    let (detect, proto_data) = load_test_data();
    if detect.is_empty() {
        println!("  [skipped: no detections]\n");
        return;
    }

    let cpu = CPUProcessor::new();

    // Proto-res: just the dot product + sigmoid (at 160×160 native resolution)
    let name = "phase/dot_product_only";
    let result = run_bench(name, WARMUP, ITERATIONS, || {
        let _ = cpu
            .materialize_segmentations(&detect, &proto_data, None)
            .unwrap();
    });
    result.print_summary();
    suite.record(&result);

    // Scaled: dot product + bilinear upsample + threshold
    let name = "phase/dot_plus_bilinear_640x480";
    let result = run_bench(name, WARMUP, ITERATIONS, || {
        let _ = cpu
            .materialize_scaled_segmentations(
                &detect,
                &proto_data,
                Some(LETTERBOX_640X480),
                COCO_W,
                COCO_H,
            )
            .unwrap();
    });
    result.print_summary();
    suite.record(&result);
}

/// Benchmark the NMS decode phase (score filter + NMS + proto extraction).
/// This matches the `nms_decode` portion of the user's pipeline when called
/// with COCO-eval score_threshold=0.001.
fn bench_nms_decode(suite: &mut BenchSuite) {
    println!("\n== nms_decode: score filter + NMS + proto extraction ==\n");

    let boxes = load_boxes_i8();
    let protos = load_protos_i8();
    let n_proposals = boxes.dim().1;
    let n_classes = boxes.dim().0 - 4 - 32; // 80

    println!(
        "  Tensor: [{}, {n_proposals}] I8 ({n_classes} classes + 4 box + 32 mask)",
        boxes.dim().0
    );
    println!(
        "  Protos: [{}, {}, {}] I8",
        protos.dim().0,
        protos.dim().1,
        protos.dim().2
    );
    println!("  score_threshold={SCORE_THRESHOLD}, iou_threshold={IOU_THRESHOLD}");
    println!();

    // Full decode (argmax + NMS + proto extraction)
    let name = "nms_decode/full_0.001";
    let result = run_bench(name, WARMUP, ITERATIONS, || {
        let mut output_boxes = Vec::with_capacity(300);
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
        std::hint::black_box((&output_boxes, &proto_data));
    });
    let (detect, _) = decode_from_fixtures();
    println!("  Survivors after NMS: {} detections", detect.len());
    result.print_summary();
    suite.record(&result);

    // With typical inference threshold (0.25)
    let name = "nms_decode/full_0.25";
    let result = run_bench(name, WARMUP, ITERATIONS, || {
        let mut output_boxes = Vec::with_capacity(300);
        let proto_data = impl_yolo_segdet_quant_proto::<XYWH, _, _>(
            (boxes.view(), QUANT_BOXES),
            (protos.view(), QUANT_PROTOS),
            0.25,
            IOU_THRESHOLD,
            Some(Nms::ClassAgnostic),
            edgefirst_decoder::yolo::MAX_NMS_CANDIDATES,
            300,
            &mut output_boxes,
        );
        std::hint::black_box((&output_boxes, &proto_data));
    });
    result.print_summary();
    suite.record(&result);

    // Pre-NMS top-K=3000 (the key optimization: reduces NMS from O(8400²) to O(3000²))
    let name = "nms_decode/topk_3000";
    let result = run_bench(name, WARMUP, ITERATIONS, || {
        let mut output_boxes = Vec::with_capacity(300);
        let proto_data = impl_yolo_segdet_quant_proto::<XYWH, _, _>(
            (boxes.view(), QUANT_BOXES),
            (protos.view(), QUANT_PROTOS),
            SCORE_THRESHOLD,
            IOU_THRESHOLD,
            Some(Nms::ClassAgnostic),
            3000,
            300,
            &mut output_boxes,
        );
        std::hint::black_box((&output_boxes, &proto_data));
    });
    result.print_summary();
    suite.record(&result);

    // Isolate: just score filtering + box decode (no NMS)
    let name = "nms_decode/score_filter_only";
    let result = run_bench(name, WARMUP, ITERATIONS, || {
        let mut output_boxes = Vec::with_capacity(300);
        let proto_data = impl_yolo_segdet_quant_proto::<XYWH, _, _>(
            (boxes.view(), QUANT_BOXES),
            (protos.view(), QUANT_PROTOS),
            SCORE_THRESHOLD,
            IOU_THRESHOLD,
            None, // bypass NMS
            edgefirst_decoder::yolo::MAX_NMS_CANDIDATES,
            300,
            &mut output_boxes,
        );
        std::hint::black_box((&output_boxes, &proto_data));
    });
    result.print_summary();
    suite.record(&result);
}

fn main() {
    let mut suite = BenchSuite::from_args();

    if std::env::args().any(|a| a == "--generate") {
        // Just generate safetensors and exit.
        println!("Generating safetensors test data...");
        let (detect, proto_data) = decode_from_fixtures();
        println!("  {} detections from NMS", detect.len());
        if let Err(e) = save_to_safetensors(&safetensors_path(), &detect, &proto_data) {
            eprintln!("ERROR: {e}");
            std::process::exit(1);
        }
        return;
    }

    bench_nms_decode(&mut suite);
    bench_scaled_materialize(&mut suite);
    bench_detection_count_scaling(&mut suite);
    bench_phase_breakdown(&mut suite);

    suite.finish();
}
