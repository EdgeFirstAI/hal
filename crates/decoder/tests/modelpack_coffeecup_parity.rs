// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! Parity tests: HAL ModelPack anchor-grid decoder vs the canonical
//! ModelPack Python reference (``deepview/modelpack/demos/models/
//! inference_no_decoder.py``) on the coffeecup detection model.
//!
//! Two fixtures:
//! * ``coffeecup-mpk-det-relu-t-27d6_quant-u8-i8_smart.safetensors`` —
//!   TFLite INT8 outputs from the quantized split-decoder export.
//! * ``coffeecup-mpk-det-relu-t-27d6.safetensors`` — ONNX float32
//!   outputs from the pre-quantization export.
//!
//! Both fixtures embed the ``edgefirst.json`` schema, the raw model
//! outputs, and the post-NMS reference detections produced by running
//! ModelPack's reference decode formula in NumPy. The HAL decoder
//! must reproduce those detections within a small tolerance.
//!
//! Fixtures are regenerated with
//! ``scripts/decoder_generate_modelpack_fixture.py``.

mod common;

use std::path::PathBuf;

use edgefirst_decoder::{schema::SchemaV2, DecoderBuilder, DetectBox, Segmentation};
use edgefirst_tensor::TensorDyn;

fn workspace_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
}

fn fixture_path(name: &str) -> PathBuf {
    workspace_root().join("testdata/decoder").join(name)
}

fn iou_xyxy(a: &[f32; 4], b: &[f32; 4]) -> f32 {
    let x1 = a[0].max(b[0]);
    let y1 = a[1].max(b[1]);
    let x2 = a[2].min(b[2]);
    let y2 = a[3].min(b[3]);
    let inter = (x2 - x1).max(0.0) * (y2 - y1).max(0.0);
    let area_a = (a[2] - a[0]) * (a[3] - a[1]);
    let area_b = (b[2] - b[0]) * (b[3] - b[1]);
    inter / (area_a + area_b - inter + 1e-9)
}

/// Drive the HAL decoder on the fixture's raw tensors and verify the
/// post-NMS output matches the embedded Python reference one-for-one.
fn assert_coffeecup_parity(fixture_filename: &str, score_tol: f32, iou_floor: f32) {
    let path = fixture_path(fixture_filename);
    let fix = match common::per_scale_fixture::PerScaleFixture::load(&path) {
        Ok(f) => f,
        Err(common::per_scale_fixture::FixtureError::NotPresent(_)) => {
            eprintln!("skip: fixture {path:?} not present (run `git lfs pull`)");
            return;
        }
        Err(e) => panic!("fixture load failed: {e}"),
    };
    let schema: SchemaV2 =
        serde_json::from_str(fix.schema_json()).expect("schema_json must parse as SchemaV2");
    let nms = fix.nms_config();
    let decoder = DecoderBuilder::default()
        .with_schema(schema)
        .with_iou_threshold(nms.iou_threshold)
        .with_score_threshold(nms.score_threshold)
        .build()
        .expect("build decoder");

    let owned_tensors = fix
        .build_tensors_with_quant()
        .expect("build tensors from fixture");
    let inputs: Vec<&TensorDyn> = owned_tensors.iter().collect();

    let mut hal_boxes: Vec<DetectBox> = Vec::with_capacity(300);
    let mut hal_masks: Vec<Segmentation> = Vec::new();
    decoder
        .decode(&inputs, &mut hal_boxes, &mut hal_masks)
        .expect("HAL decode must succeed");

    let reference = fix.decoded().expect("fixture decoded() reference");
    let ref_n = reference.boxes_xyxy.shape()[0];

    assert_eq!(
        hal_boxes.len(),
        ref_n,
        "{fixture_filename}: HAL produced {} detections, reference {ref_n}",
        hal_boxes.len()
    );

    // Greedy IoU match each reference detection to a HAL detection.
    // With ref_n == 1 (single coffeecup) the match is unambiguous;
    // the loop generalises for future multi-object fixtures.
    let mut consumed = vec![false; hal_boxes.len()];
    for i in 0..ref_n {
        let ref_box: [f32; 4] = [
            reference.boxes_xyxy[[i, 0]],
            reference.boxes_xyxy[[i, 1]],
            reference.boxes_xyxy[[i, 2]],
            reference.boxes_xyxy[[i, 3]],
        ];
        let ref_score = reference.scores[i];
        let ref_class = reference.classes[i] as usize;

        let (best_idx, best_iou) = hal_boxes
            .iter()
            .enumerate()
            .filter(|(j, _)| !consumed[*j])
            .map(|(j, hal)| {
                let hal_box = [hal.bbox.xmin, hal.bbox.ymin, hal.bbox.xmax, hal.bbox.ymax];
                (j, iou_xyxy(&ref_box, &hal_box))
            })
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
            .expect("HAL must have an unconsumed detection to match against");

        assert!(
            best_iou >= iou_floor,
            "{fixture_filename}: ref det[{i}] xyxy={ref_box:?} has no HAL match \
             with IoU≥{iou_floor:.2} (best={best_iou:.4})"
        );

        let hal = &hal_boxes[best_idx];
        let score_diff = (hal.score - ref_score).abs();
        assert!(
            score_diff <= score_tol,
            "{fixture_filename}: ref det[{i}] score={ref_score:.4} vs HAL {:.4} \
             (|diff|={score_diff:.4} > tol={score_tol})",
            hal.score
        );
        assert_eq!(
            hal.label, ref_class,
            "{fixture_filename}: ref det[{i}] class={ref_class} vs HAL {}",
            hal.label
        );

        consumed[best_idx] = true;
    }
}

#[test]
fn coffeecup_modelpack_int8_smart_parity() {
    // Same fixture as the float ONNX path, just routed through the int8
    // dequant→sigmoid kernel. Tolerance is looser because the per-scale
    // quantization round-trip drifts a few percent at sigmoid boundaries.
    assert_coffeecup_parity(
        "coffeecup-mpk-det-relu-t-27d6_quant-u8-i8_smart.safetensors",
        /* score_tol = */ 0.02,
        /* iou_floor = */ 0.95,
    );
}

#[test]
fn coffeecup_modelpack_float_onnx_parity() {
    // Float-to-float: HAL and the NumPy reference should agree to within
    // floating-point round-off on a single coffeecup detection.
    assert_coffeecup_parity(
        "coffeecup-mpk-det-relu-t-27d6.safetensors",
        /* score_tol = */ 0.001,
        /* iou_floor = */ 0.99,
    );
}
