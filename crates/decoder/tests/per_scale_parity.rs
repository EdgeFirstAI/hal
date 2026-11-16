// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! Parity tests: HAL per-scale decoder vs. the Python reference at
//! `scripts/per_scale_decode_reference.py`.
//!
//! For each fixture, both decoders run on identical inputs and the outputs
//! are compared by:
//!   1. Per-anchor box / score / mc/proto agreement.
//!   2. Post-NMS DetectBox list agreement under IoU/class tolerances.
//!
//! Tests skip when `tflite_runtime` is not available or the fixture model
//! files are missing.

#[macro_use]
mod common;

use std::path::{Path, PathBuf};
use std::process::Command;

use common::testdata_root;

const REFERENCE_SCRIPT: &str = "scripts/per_scale_decode_reference.py";

/// Workspace root, two levels up from this crate's manifest dir.
fn workspace_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
}

/// Path to the Python reference script, resolved from the workspace root.
#[allow(dead_code)]
fn reference_script_path() -> PathBuf {
    workspace_root().join(REFERENCE_SCRIPT)
}

/// Path to the per-scale fixtures directory.
fn fixtures_dir_path() -> PathBuf {
    testdata_root().join("per_scale")
}

/// Returns true when the Python reference is runnable in this env.
#[allow(dead_code)]
fn python_reference_available() -> bool {
    // Prefer the project's venv if available.
    let pythons = ["./venv/bin/python", "venv/bin/python", "python", "python3"];
    for p in pythons {
        if let Ok(s) = Command::new(p)
            .args(["-c", "import tflite_runtime"])
            .status()
        {
            if s.success() {
                return true;
            }
        }
    }
    false
}

/// Locate the Python interpreter to use. Prefers ./venv/bin/python at the
/// workspace root.
#[allow(dead_code)]
fn python_cmd() -> Option<PathBuf> {
    let root = workspace_root();
    let candidates = [
        root.join("venv/bin/python"),
        PathBuf::from("venv/bin/python"),
    ];
    for c in candidates {
        if c.exists() {
            return Some(c);
        }
    }
    Some(PathBuf::from("python"))
}

/// Run the Python reference as a subprocess, dumping its outputs as JSON.
/// Returns the deserialized output structure.
#[allow(dead_code)]
fn run_python_reference(model: &Path, image: Option<&Path>) -> Option<serde_json::Value> {
    let py = python_cmd()?;
    let mut cmd = Command::new(py);
    cmd.current_dir(workspace_root());
    cmd.arg(REFERENCE_SCRIPT).arg(model).arg("--json-out");
    if let Some(img) = image {
        cmd.arg(img);
    }
    let out = cmd.output().ok()?;
    if !out.status.success() {
        eprintln!(
            "python reference exited non-zero: stderr={}",
            String::from_utf8_lossy(&out.stderr)
        );
        return None;
    }
    serde_json::from_slice(&out.stdout).ok()
}

/// Compare two f32 values within `ulps` ulps.
#[allow(dead_code)]
fn close_within_ulp(a: f32, b: f32, ulps: u32) -> bool {
    let diff = (a - b).abs();
    if diff < 1e-7 {
        return true;
    }
    let scale = a.abs().max(b.abs());
    diff < f32::EPSILON * scale * (ulps as f32)
}

/// Gives the test binary one test that always runs.
#[test]
fn per_scale_parity_harness_compiles() {
    assert!(close_within_ulp(1.0_f32, 1.0_f32, 1));
}

/// True only when the source repo is present at the compiled-in path.
fn repo_is_present() -> bool {
    workspace_root().join("Cargo.toml").exists()
}

/// Verifies the Python script exists at the expected path.
#[test]
fn python_reference_script_exists_in_repo() {
    if !repo_is_present() {
        skip_note!("source repo not present (deployed on-target run)");
        return;
    }
    let p = reference_script_path();
    assert!(p.exists(), "expected {p:?} to exist; check repo layout");
}

/// Verifies the fixtures dir exists.
#[test]
fn fixtures_dir_exists_in_repo() {
    if !repo_is_present() {
        skip_note!("source repo not present (deployed on-target run)");
        return;
    }
    let p = fixtures_dir_path();
    assert!(p.exists(), "expected {p:?} to exist");
}

// ────────────────────────────────────────────────────────────────────
// Synthetic-fixture HAL parity tests
// ────────────────────────────────────────────────────────────────────
//
// These tests run the full per-scale decoder pipeline (build → run → NMS →
// DetectBox) against hand-computed golden values for synthetic int8 inputs.

use edgefirst_decoder::per_scale::DecodeDtype;
use edgefirst_decoder::{schema::SchemaV2, DecoderBuilder, DetectBox, Nms};
use edgefirst_tensor::{
    Quantization as TQ, Tensor, TensorDyn, TensorMapTrait, TensorMemory, TensorTrait,
};

/// A minimal LTRB-detection-only per-scale schema with one FPN level
/// (stride=8, h=w=1, NC=2).
fn minimal_ltrb_schema_one_level() -> &'static str {
    r#"{
        "schema_version": 2,
        "nms": "class_agnostic",
        "input": {
            "shape": [1, 8, 8, 3],
            "dshape": [{"batch": 1}, {"height": 8}, {"width": 8}, {"num_features": 3}],
            "cameraadaptor": "rgb"
        },
        "outputs": [
            {
                "name": "boxes", "type": "boxes",
                "shape": [1, 4, 1],
                "dshape": [{"batch": 1}, {"box_coords": 4}, {"num_boxes": 1}],
                "encoding": "ltrb", "decoder": "ultralytics", "normalized": true,
                "outputs": [{
                    "name": "boxes_0", "type": "boxes",
                    "stride": 8, "scale_index": 0,
                    "shape": [1, 1, 1, 4],
                    "dshape": [{"batch": 1}, {"height": 1}, {"width": 1}, {"box_coords": 4}],
                    "dtype": "int8",
                    "quantization": {"scale": 0.1, "zero_point": 0, "dtype": "int8"}
                }]
            },
            {
                "name": "scores", "type": "scores",
                "shape": [1, 2, 1],
                "dshape": [{"batch": 1}, {"num_classes": 2}, {"num_boxes": 1}],
                "score_format": "per_class", "decoder": "ultralytics",
                "outputs": [{
                    "name": "scores_0", "type": "scores",
                    "stride": 8, "scale_index": 0,
                    "shape": [1, 1, 1, 2],
                    "dshape": [{"batch": 1}, {"height": 1}, {"width": 1}, {"num_classes": 2}],
                    "activation_required": "sigmoid",
                    "dtype": "int8",
                    "quantization": {"scale": 0.1, "zero_point": 0, "dtype": "int8"}
                }]
            }
        ]
    }"#
}

#[test]
fn synthetic_ltrb_one_anchor_zero_input_gives_zero_box_at_centre() {
    // 1x1 grid, stride=8, scale=0.1, zp=0.
    // Box int8 = [0, 0, 0, 0] → dequant = [0, 0, 0, 0] → ltrb_decode_at_centre.
    // Anchor centre at (0.5, 0.5).
    // dist2bbox: xc=(0.5+0)*8=4, yc=4, w=0, h=0.
    let schema_json = minimal_ltrb_schema_one_level();
    let schema: SchemaV2 = serde_json::from_str(schema_json).unwrap();

    let decoder = DecoderBuilder::default()
        .with_schema(schema)
        .with_decode_dtype(DecodeDtype::F32)
        .with_iou_threshold(0.7)
        .with_score_threshold(0.0001)
        .with_nms(Some(Nms::ClassAgnostic))
        .build()
        .expect("build");

    // Two zero-int8 input tensors with attached quantization.
    let mut box_t = Tensor::<i8>::new(&[1, 1, 1, 4], Some(TensorMemory::Mem), None).unwrap();
    box_t.set_quantization(TQ::per_tensor(0.1, 0)).unwrap();
    let mut score_t = Tensor::<i8>::new(&[1, 1, 1, 2], Some(TensorMemory::Mem), None).unwrap();
    score_t.set_quantization(TQ::per_tensor(0.1, 0)).unwrap();
    {
        let mut m = score_t.map().unwrap();
        m.as_mut_slice()[0] = 50; // dequant: 5.0 → sigmoid: ~0.99
    }

    let dyn_box = TensorDyn::I8(box_t);
    let dyn_score = TensorDyn::I8(score_t);
    let inputs: Vec<&TensorDyn> = vec![&dyn_box, &dyn_score];

    let mut output_boxes: Vec<DetectBox> = Vec::with_capacity(10);
    let mut masks = Vec::new();
    decoder
        .decode(&inputs, &mut output_boxes, &mut masks)
        .expect("decode");

    assert_eq!(
        output_boxes.len(),
        1,
        "expected 1 detection, got {}",
        output_boxes.len()
    );
    let det = &output_boxes[0];

    // Pixel-space box (4, 4, 4, 4), normalized by the 8x8 input: 0.5 at every
    // corner.
    assert!(
        (det.bbox.xmin - 0.5).abs() < 1e-3,
        "xmin = {}",
        det.bbox.xmin
    );
    assert!(
        (det.bbox.ymin - 0.5).abs() < 1e-3,
        "ymin = {}",
        det.bbox.ymin
    );
    assert!(
        (det.bbox.xmax - 0.5).abs() < 1e-3,
        "xmax = {}",
        det.bbox.xmax
    );
    assert!(
        (det.bbox.ymax - 0.5).abs() < 1e-3,
        "ymax = {}",
        det.bbox.ymax
    );
    // Class 0 had logit 5.0 (sigmoid≈0.993), class 1 had logit 0.0 (sigmoid=0.5).
    assert_eq!(det.label, 0, "expected class 0 winner");
    assert!(det.score > 0.99, "expected high score, got {}", det.score);
}

#[test]
fn synthetic_ltrb_distances_produce_expected_box() {
    // ltrb int8 = [10, 10, 30, 30] → dequant scale=0.1 → [1, 1, 3, 3]
    // dist2bbox at (0.5, 0.5) stride 8:
    //   xc = (0.5 + (3-1)/2) * 8 = (0.5 + 1) * 8 = 12
    //   yc = (0.5 + (3-1)/2) * 8 = 12
    //   w = (1+3) * 8 = 32
    //   h = 32
    // → XYXY: xmin = 12 - 16 = -4, xmax = 12 + 16 = 28, ymin = -4, ymax = 28
    let schema_json = minimal_ltrb_schema_one_level();
    let schema: SchemaV2 = serde_json::from_str(schema_json).unwrap();
    let decoder = DecoderBuilder::default()
        .with_schema(schema)
        .with_decode_dtype(DecodeDtype::F32)
        .with_iou_threshold(0.7)
        .with_score_threshold(0.0001)
        .with_nms(Some(Nms::ClassAgnostic))
        .build()
        .unwrap();

    let mut box_t = Tensor::<i8>::new(&[1, 1, 1, 4], Some(TensorMemory::Mem), None).unwrap();
    box_t.set_quantization(TQ::per_tensor(0.1, 0)).unwrap();
    {
        let mut m = box_t.map().unwrap();
        let s = m.as_mut_slice();
        s[0] = 10;
        s[1] = 10;
        s[2] = 30;
        s[3] = 30; // L=1, T=1, R=3, B=3
    }
    let mut score_t = Tensor::<i8>::new(&[1, 1, 1, 2], Some(TensorMemory::Mem), None).unwrap();
    score_t.set_quantization(TQ::per_tensor(0.1, 0)).unwrap();
    {
        let mut m = score_t.map().unwrap();
        m.as_mut_slice()[1] = 50; // class 1 wins
    }

    let dyn_box = TensorDyn::I8(box_t);
    let dyn_score = TensorDyn::I8(score_t);
    let inputs: Vec<&TensorDyn> = vec![&dyn_box, &dyn_score];

    let mut output_boxes: Vec<DetectBox> = Vec::with_capacity(10);
    let mut masks = Vec::new();
    decoder
        .decode(&inputs, &mut output_boxes, &mut masks)
        .expect("decode");

    assert_eq!(output_boxes.len(), 1);
    let det = &output_boxes[0];
    // Pixel-space (-4, -4, 28, 28) → normalized (-0.5, -0.5, 3.5, 3.5).
    assert!(
        (det.bbox.xmin - (-0.5)).abs() < 1e-2,
        "xmin = {}",
        det.bbox.xmin
    );
    assert!(
        (det.bbox.xmax - 3.5).abs() < 1e-2,
        "xmax = {}",
        det.bbox.xmax
    );
    assert!(
        (det.bbox.ymin - (-0.5)).abs() < 1e-2,
        "ymin = {}",
        det.bbox.ymin
    );
    assert!(
        (det.bbox.ymax - 3.5).abs() < 1e-2,
        "ymax = {}",
        det.bbox.ymax
    );
    assert_eq!(det.label, 1);
}

// ────────────────────────────────────────────────────────────────────
// NCHW layout coverage
// ────────────────────────────────────────────────────────────────────
//
// These tests build paired NHWC and NCHW schemas with the same logical
// content, feed equivalent tensor data in both layouts, and assert the
// decoded `DetectBox` lists match exactly.

/// LTRB schema with one stride-8 level, h=w=2, NC=2, NCHW children:
/// dshape `[batch, box_coords, height, width]`.
fn minimal_ltrb_schema_one_level_2x2_nchw() -> &'static str {
    r#"{
        "schema_version": 2,
        "nms": "class_agnostic",
        "input": {
            "shape": [1, 16, 16, 3],
            "dshape": [{"batch": 1}, {"height": 16}, {"width": 16}, {"num_features": 3}],
            "cameraadaptor": "rgb"
        },
        "outputs": [
            {
                "name": "boxes", "type": "boxes",
                "shape": [1, 4, 4],
                "dshape": [{"batch": 1}, {"box_coords": 4}, {"num_boxes": 4}],
                "encoding": "ltrb", "decoder": "ultralytics", "normalized": true,
                "outputs": [{
                    "name": "boxes_0", "type": "boxes",
                    "stride": 8, "scale_index": 0,
                    "shape": [1, 4, 2, 2],
                    "dshape": [{"batch": 1}, {"box_coords": 4}, {"height": 2}, {"width": 2}],
                    "dtype": "int8",
                    "quantization": {"scale": 0.1, "zero_point": 0, "dtype": "int8"}
                }]
            },
            {
                "name": "scores", "type": "scores",
                "shape": [1, 2, 4],
                "dshape": [{"batch": 1}, {"num_classes": 2}, {"num_boxes": 4}],
                "score_format": "per_class", "decoder": "ultralytics",
                "outputs": [{
                    "name": "scores_0", "type": "scores",
                    "stride": 8, "scale_index": 0,
                    "shape": [1, 2, 2, 2],
                    "dshape": [{"batch": 1}, {"num_classes": 2}, {"height": 2}, {"width": 2}],
                    "activation_required": "sigmoid",
                    "dtype": "int8",
                    "quantization": {"scale": 0.1, "zero_point": 0, "dtype": "int8"}
                }]
            }
        ]
    }"#
}

/// NHWC pair of the schema above: only the children's `shape` and `dshape`
/// ordering differ.
fn minimal_ltrb_schema_one_level_2x2_nhwc() -> &'static str {
    r#"{
        "schema_version": 2,
        "nms": "class_agnostic",
        "input": {
            "shape": [1, 16, 16, 3],
            "dshape": [{"batch": 1}, {"height": 16}, {"width": 16}, {"num_features": 3}],
            "cameraadaptor": "rgb"
        },
        "outputs": [
            {
                "name": "boxes", "type": "boxes",
                "shape": [1, 4, 4],
                "dshape": [{"batch": 1}, {"box_coords": 4}, {"num_boxes": 4}],
                "encoding": "ltrb", "decoder": "ultralytics", "normalized": true,
                "outputs": [{
                    "name": "boxes_0", "type": "boxes",
                    "stride": 8, "scale_index": 0,
                    "shape": [1, 2, 2, 4],
                    "dshape": [{"batch": 1}, {"height": 2}, {"width": 2}, {"box_coords": 4}],
                    "dtype": "int8",
                    "quantization": {"scale": 0.1, "zero_point": 0, "dtype": "int8"}
                }]
            },
            {
                "name": "scores", "type": "scores",
                "shape": [1, 2, 4],
                "dshape": [{"batch": 1}, {"num_classes": 2}, {"num_boxes": 4}],
                "score_format": "per_class", "decoder": "ultralytics",
                "outputs": [{
                    "name": "scores_0", "type": "scores",
                    "stride": 8, "scale_index": 0,
                    "shape": [1, 2, 2, 2],
                    "dshape": [{"batch": 1}, {"height": 2}, {"width": 2}, {"num_classes": 2}],
                    "activation_required": "sigmoid",
                    "dtype": "int8",
                    "quantization": {"scale": 0.1, "zero_point": 0, "dtype": "int8"}
                }]
            }
        ]
    }"#
}

/// Per-anchor logical box int8s (NHWC anchor-major, channel-contiguous):
/// 4 anchors × 4 LTRB channels, a distinct box per anchor.
const NCHW_BOX_DATA_BY_ANCHOR: [[i8; 4]; 4] = [
    [10, 10, 30, 30], // anchor 0: L=1, T=1, R=3, B=3
    [5, 5, 15, 15],   // anchor 1: L=0.5, T=0.5, R=1.5, B=1.5
    [20, 20, 40, 40], // anchor 2
    [15, 15, 25, 25], // anchor 3
];

/// Per-anchor score int8s (NHWC anchor-major, NC=2). Anchor 0 votes
/// class 1; the other anchors stay below threshold.
const NCHW_SCORE_DATA_BY_ANCHOR: [[i8; 2]; 4] = [
    [0, 50], // anchor 0: class 1 strongly wins
    [0, 0],  // anchor 1: 0.5/0.5
    [0, 0],  // anchor 2
    [0, 0],  // anchor 3
];

/// Pack the per-anchor data into NHWC byte order: `[a0c0, a0c1, ..., a0cN-1,
/// a1c0, ...]`.
fn pack_nhwc_box(per_anchor: &[[i8; 4]]) -> Vec<i8> {
    per_anchor.iter().flat_map(|a| a.iter().copied()).collect()
}
fn pack_nhwc_score(per_anchor: &[[i8; 2]]) -> Vec<i8> {
    per_anchor.iter().flat_map(|a| a.iter().copied()).collect()
}

/// Pack the per-anchor data into NCHW byte order: `[c0a0, c0a1, ..., c0aN-1,
/// c1a0, ...]`. For a 2x2 grid the anchor index is `hi*W + wi` row-major.
fn pack_nchw_box(per_anchor: &[[i8; 4]]) -> Vec<i8> {
    let n_anchors = per_anchor.len();
    let n_channels = 4;
    let mut out = vec![0i8; n_anchors * n_channels];
    for (ai, anchor) in per_anchor.iter().enumerate() {
        for (ci, &v) in anchor.iter().enumerate() {
            out[ci * n_anchors + ai] = v;
        }
    }
    out
}
fn pack_nchw_score(per_anchor: &[[i8; 2]]) -> Vec<i8> {
    let n_anchors = per_anchor.len();
    let n_channels = 2;
    let mut out = vec![0i8; n_anchors * n_channels];
    for (ai, anchor) in per_anchor.iter().enumerate() {
        for (ci, &v) in anchor.iter().enumerate() {
            out[ci * n_anchors + ai] = v;
        }
    }
    out
}

fn run_layout_decode(
    schema_json: &str,
    box_shape: &[usize],
    box_data: &[i8],
    score_shape: &[usize],
    score_data: &[i8],
) -> Vec<DetectBox> {
    let schema: SchemaV2 = serde_json::from_str(schema_json).unwrap();
    let decoder = DecoderBuilder::default()
        .with_schema(schema)
        .with_decode_dtype(DecodeDtype::F32)
        .with_iou_threshold(0.7)
        .with_score_threshold(0.5)
        .with_nms(Some(Nms::ClassAgnostic))
        .build()
        .expect("build");

    let mut box_t = Tensor::<i8>::new(box_shape, Some(TensorMemory::Mem), None).unwrap();
    box_t.set_quantization(TQ::per_tensor(0.1, 0)).unwrap();
    {
        let mut m = box_t.map().unwrap();
        m.as_mut_slice().copy_from_slice(box_data);
    }
    let mut score_t = Tensor::<i8>::new(score_shape, Some(TensorMemory::Mem), None).unwrap();
    score_t.set_quantization(TQ::per_tensor(0.1, 0)).unwrap();
    {
        let mut m = score_t.map().unwrap();
        m.as_mut_slice().copy_from_slice(score_data);
    }
    let dyn_box = TensorDyn::I8(box_t);
    let dyn_score = TensorDyn::I8(score_t);
    let inputs: Vec<&TensorDyn> = vec![&dyn_box, &dyn_score];

    let mut out_boxes: Vec<DetectBox> = Vec::with_capacity(10);
    let mut masks = Vec::new();
    decoder
        .decode(&inputs, &mut out_boxes, &mut masks)
        .expect("decode");
    out_boxes
}

#[test]
fn synthetic_ltrb_nchw_and_nhwc_produce_identical_detections() {
    // NHWC pass: data laid out anchor-major (channels contiguous per anchor).
    let nhwc_box = pack_nhwc_box(&NCHW_BOX_DATA_BY_ANCHOR);
    let nhwc_score = pack_nhwc_score(&NCHW_SCORE_DATA_BY_ANCHOR);
    let nhwc_dets = run_layout_decode(
        minimal_ltrb_schema_one_level_2x2_nhwc(),
        &[1, 2, 2, 4],
        &nhwc_box,
        &[1, 2, 2, 2],
        &nhwc_score,
    );

    // NCHW pass: same logical content, channel-major byte layout.
    let nchw_box = pack_nchw_box(&NCHW_BOX_DATA_BY_ANCHOR);
    let nchw_score = pack_nchw_score(&NCHW_SCORE_DATA_BY_ANCHOR);
    let nchw_dets = run_layout_decode(
        minimal_ltrb_schema_one_level_2x2_nchw(),
        &[1, 4, 2, 2],
        &nchw_box,
        &[1, 2, 2, 2],
        &nchw_score,
    );

    assert_eq!(
        nhwc_dets.len(),
        nchw_dets.len(),
        "detection count mismatch: nhwc={} nchw={}",
        nhwc_dets.len(),
        nchw_dets.len()
    );
    for (i, (a, b)) in nhwc_dets.iter().zip(nchw_dets.iter()).enumerate() {
        assert_eq!(a.label, b.label, "det[{i}] label mismatch");
        assert!(
            (a.score - b.score).abs() < 1e-5,
            "det[{i}] score mismatch: nhwc={} nchw={}",
            a.score,
            b.score
        );
        assert!(
            (a.bbox.xmin - b.bbox.xmin).abs() < 1e-3,
            "det[{i}] xmin mismatch"
        );
        assert!(
            (a.bbox.ymin - b.bbox.ymin).abs() < 1e-3,
            "det[{i}] ymin mismatch"
        );
        assert!(
            (a.bbox.xmax - b.bbox.xmax).abs() < 1e-3,
            "det[{i}] xmax mismatch"
        );
        assert!(
            (a.bbox.ymax - b.bbox.ymax).abs() < 1e-3,
            "det[{i}] ymax mismatch"
        );
    }
    assert!(!nhwc_dets.is_empty(), "expected at least one detection");
}

// ────────────────────────────────────────────────────────────────────
// Real-model Python smoke (skip-capable)
// ────────────────────────────────────────────────────────────────────
//
// These tests load TFLite fixtures when present at
// `testdata/per_scale/<model>.tflite` and run the Python reference on the
// model. They skip when the fixtures are absent.

#[test]
#[ignore = "external real-model fixture; this is Python-only smoke and does not exercise HAL decode"]
fn parity_yolov8n_seg_per_scale_int8() {
    let model = testdata_root()
        .join("per_scale")
        .join("yolov8n_seg_per_scale_int8.tflite");
    if !model.exists() {
        skip_note!(
            "real-model fixture not present at {model:?} — contact validator team for the binary"
        );
        return;
    }
    if !python_reference_available() {
        skip_note!("tflite_runtime not importable in Python env");
        return;
    }
    let py_result = run_python_reference(&model, None);
    let py_result = match py_result {
        Some(v) => v,
        None => {
            skip_note!("Python reference failed to run");
            return;
        }
    };
    let n = py_result
        .get("num_detections")
        .and_then(|v| v.as_u64())
        .unwrap_or(0);
    skip_note!("python reference detected {n} boxes on synthetic input");
    // TODO: run HAL on the same input and compare with the Python reference.
}

#[test]
#[ignore = "external real-model fixture; this is Python-only smoke and does not exercise HAL decode"]
fn parity_yolo26n_seg_per_scale_int8() {
    let model = testdata_root()
        .join("per_scale")
        .join("yolo26n_seg_per_scale_int8.tflite");
    if !model.exists() {
        skip_note!("yolo26n fixture not present");
        return;
    }
    if !python_reference_available() {
        skip_note!("tflite_runtime not importable");
        return;
    }
    let py_result = run_python_reference(&model, None);
    let py_result = match py_result {
        Some(v) => v,
        None => {
            skip_note!("Python reference failed to run");
            return;
        }
    };
    let n = py_result
        .get("num_detections")
        .and_then(|v| v.as_u64())
        .unwrap_or(0);
    skip_note!("python reference detected {n} boxes on synthetic input");
}

#[test]
fn fixture_loader_metadata_round_trip_synthetic() {
    use std::fs;
    use std::process::Command;

    let tmpdir = std::env::temp_dir().join("hal_test_fixture_loader_meta");
    let _ = fs::remove_dir_all(&tmpdir);
    fs::create_dir_all(&tmpdir).unwrap();
    let fixture_path = tmpdir.join("synth.safetensors");

    let venv_py = workspace_root().join("venv/bin/python");
    let py = if venv_py.exists() {
        venv_py.to_string_lossy().into_owned()
    } else {
        "python3".into()
    };
    let script = format!(
        r#"
import numpy as np, safetensors.numpy as stnp
stnp.save_file({{
    'input.image': np.zeros((1,4,4,3), dtype=np.uint8),
    'decoded.boxes_xyxy': np.zeros((0,4), dtype=np.float32),
    'decoded.scores': np.zeros((0,), dtype=np.float32),
    'decoded.classes': np.zeros((0,), dtype=np.uint32),
}}, '{p}', metadata={{
    'format_version': '1',
    'decoder_family': 'per_scale_yolo_seg',
    'model_basename': 'synthetic',
    'expected_count_min': '0',
    'documentation_md': '# synthetic',
    'schema_json': '{{}}',
    'quantization_json': '{{}}',
    'nms_config_json': '{{"iou_threshold":0.7,"score_threshold":0.001,"max_detections":300}}',
}})
"#,
        p = fixture_path.to_string_lossy()
    );

    let out = Command::new(&py).arg("-c").arg(&script).output();
    match out {
        Ok(o) if o.status.success() => {}
        Ok(o) => {
            skip_note!("python failed: {}", String::from_utf8_lossy(&o.stderr));
            return;
        }
        Err(e) => {
            skip_note!("python not available: {e}");
            return;
        }
    }

    let fix = common::per_scale_fixture::PerScaleFixture::load(&fixture_path)
        .expect("synthetic fixture must load");
    assert_eq!(fix.format_version, "1");
    assert_eq!(fix.decoder_family, "per_scale_yolo_seg");
    assert_eq!(fix.model_basename, "synthetic");
    assert_eq!(fix.expected_count_min, 0);
}

#[test]
fn fixture_loader_parses_schema_quant_nms() {
    use std::fs;
    use std::process::Command;

    let tmpdir = std::env::temp_dir().join("hal_test_fixture_loader_meta2");
    let _ = fs::remove_dir_all(&tmpdir);
    fs::create_dir_all(&tmpdir).unwrap();
    let fixture_path = tmpdir.join("synth.safetensors");

    let venv_py = workspace_root().join("venv/bin/python");
    let py = if venv_py.exists() {
        venv_py.to_string_lossy().into_owned()
    } else {
        "python3".into()
    };

    let script = format!(
        r#"
import numpy as np, safetensors.numpy as stnp, json
schema = {{"schema_version": 2, "outputs": []}}
stnp.save_file({{
    'input.image': np.zeros((1,4,4,3), dtype=np.uint8),
    'decoded.boxes_xyxy': np.zeros((0,4), dtype=np.float32),
    'decoded.scores': np.zeros((0,), dtype=np.float32),
    'decoded.classes': np.zeros((0,), dtype=np.uint32),
}}, '{p}', metadata={{
    'format_version': '1',
    'decoder_family': 'per_scale_yolo_seg',
    'model_basename': 'synthetic',
    'expected_count_min': '0',
    'documentation_md': '# synthetic',
    'schema_json': json.dumps(schema),
    'quantization_json': json.dumps({{
        'boxes_0': {{'scale': 0.157, 'zero_point': -42, 'dtype': 'int8'}}}}),
    'nms_config_json': json.dumps({{
        'iou_threshold': 0.7, 'score_threshold': 0.001, 'max_detections': 300}}),
}})
"#,
        p = fixture_path.to_string_lossy()
    );

    let out = Command::new(&py).arg("-c").arg(&script).output();
    match out {
        Ok(o) if o.status.success() => {}
        Ok(o) => {
            skip_note!("python: {}", String::from_utf8_lossy(&o.stderr));
            return;
        }
        Err(e) => {
            skip_note!("python: {e}");
            return;
        }
    }

    let fix = common::per_scale_fixture::PerScaleFixture::load(&fixture_path).expect("load");
    assert!(fix.schema_json().contains("schema_version"));
    let q = fix
        .quantization_for("boxes_0")
        .expect("boxes_0 quant present");
    assert!((q.scale - 0.157).abs() < 1e-6);
    assert_eq!(q.zero_point, -42);
    assert_eq!(q.dtype, "int8");
    let nms = fix.nms_config();
    assert!((nms.iou_threshold - 0.7).abs() < 1e-6);
    assert!((nms.score_threshold - 0.001).abs() < 1e-6);
    assert_eq!(nms.max_detections, 300);
}

#[test]
fn fixture_loader_parses_raw_tensors() {
    use std::fs;
    use std::process::Command;

    let tmpdir = std::env::temp_dir().join("hal_test_fixture_loader_raw");
    let _ = fs::remove_dir_all(&tmpdir);
    fs::create_dir_all(&tmpdir).unwrap();
    let fixture_path = tmpdir.join("synth.safetensors");

    let venv_py = workspace_root().join("venv/bin/python");
    let py = if venv_py.exists() {
        venv_py.to_string_lossy().into_owned()
    } else {
        "python3".into()
    };
    let script = format!(
        r#"
import numpy as np, safetensors.numpy as stnp
stnp.save_file({{
    'input.image':  np.full((1,4,4,3),  7, dtype=np.uint8),
    'raw.boxes_0':  np.full((1,2,2,4), -3, dtype=np.int8),
    'raw.scores_0': np.full((1,2,2,2), -2, dtype=np.int8),
    'raw.mc_0':     np.full((1,2,2,4), -1, dtype=np.int8),
    'raw.protos':   np.full((1,4,4,4),  5, dtype=np.int8),
    'decoded.boxes_xyxy': np.zeros((0,4), dtype=np.float32),
    'decoded.scores': np.zeros((0,), dtype=np.float32),
    'decoded.classes': np.zeros((0,), dtype=np.uint32),
}}, '{p}', metadata={{
    'format_version': '1', 'decoder_family': 'per_scale_yolo_seg',
    'model_basename': 'synth', 'expected_count_min': '0',
    'documentation_md': '#', 'schema_json': '{{}}',
    'quantization_json': '{{}}',
    'nms_config_json': '{{"iou_threshold":0.7,"score_threshold":0.001,"max_detections":300}}',
}})
"#,
        p = fixture_path.to_string_lossy()
    );

    let out = Command::new(&py).arg("-c").arg(&script).output();
    match out {
        Ok(o) if o.status.success() => {}
        _ => {
            skip_note!(
                "python3 with numpy/safetensors unavailable; cannot build the synthetic fixture"
            );
            return;
        }
    }

    let fix = common::per_scale_fixture::PerScaleFixture::load(&fixture_path).expect("load");
    let raw = fix.raw_tensor("raw.boxes_0").expect("boxes_0 present");
    assert_eq!(raw.shape, vec![1, 2, 2, 4]);
    assert_eq!(raw.dtype, common::per_scale_fixture::RawDtype::I8);
    assert_eq!(raw.bytes.len(), 16); // 1*2*2*4
    assert_eq!(raw.bytes[0] as i8, -3);

    let img = fix.input_image_uint8().expect("input image present");
    assert_eq!(img.shape(), &[1, 4, 4, 3]);
    assert!(img.iter().all(|&x| x == 7));
}

#[test]
fn fixture_loader_parses_decoded_and_intermediates() {
    use std::fs;
    use std::process::Command;
    let tmpdir = std::env::temp_dir().join("hal_test_fixture_loader_decoded");
    let _ = fs::remove_dir_all(&tmpdir);
    fs::create_dir_all(&tmpdir).unwrap();
    let fixture_path = tmpdir.join("synth.safetensors");

    let venv_py = workspace_root().join("venv/bin/python");
    let py = if venv_py.exists() {
        venv_py.to_string_lossy().into_owned()
    } else {
        "python3".into()
    };
    let script = format!(
        r#"
import numpy as np, safetensors.numpy as stnp
stnp.save_file({{
    'input.image': np.zeros((1,4,4,3), dtype=np.uint8),
    'decoded.boxes_xyxy': np.array([[10,20,30,40],[50,60,70,80]], dtype=np.float32),
    'decoded.scores':     np.array([0.9, 0.7], dtype=np.float32),
    'decoded.classes':    np.array([3, 7], dtype=np.uint32),
    'intermediate.boxes_0.dequant':  np.full((1,2,2,4), 0.5, dtype=np.float32),
    'intermediate.boxes_0.xywh':     np.full((4,4), 0.25, dtype=np.float32),
    'intermediate.scores_0.dequant': np.full((1,2,2,2), 0.1, dtype=np.float32),
    'intermediate.scores_0.activated': np.full((1,2,2,2), 0.6, dtype=np.float32),
}}, '{p}', metadata={{
    'format_version': '1', 'decoder_family': 'per_scale_yolo_seg',
    'model_basename': 'synth', 'expected_count_min': '0',
    'documentation_md': '#', 'schema_json': '{{}}',
    'quantization_json': '{{}}',
    'nms_config_json': '{{"iou_threshold":0.7,"score_threshold":0.001,"max_detections":300}}',
}})
"#,
        p = fixture_path.to_string_lossy()
    );
    let out = Command::new(&py).arg("-c").arg(&script).output();
    match out {
        Ok(o) if o.status.success() => {}
        _ => {
            skip_note!(
                "python3 with numpy/safetensors unavailable; cannot build the synthetic fixture"
            );
            return;
        }
    }

    let fix = common::per_scale_fixture::PerScaleFixture::load(&fixture_path).unwrap();
    let dec = fix.decoded().expect("decoded present");
    assert_eq!(dec.boxes_xyxy.shape(), &[2, 4]);
    assert_eq!(dec.scores.len(), 2);
    assert_eq!(dec.classes.len(), 2);
    assert_eq!(dec.classes[0], 3);
    assert_eq!(dec.classes[1], 7);

    let inter = fix.intermediates().expect("intermediates present");
    assert!(inter.boxes_dequant(0).is_some());
    assert!(inter.boxes_xywh(0).is_some());
    assert!(inter.boxes_ltrb(0).is_none());
    let xy = inter.boxes_xywh(0).unwrap();
    assert_eq!(xy.shape(), &[4, 4]);
}

#[test]
fn fixture_loader_build_tensors_with_quant_attaches_metadata() {
    use std::fs;
    use std::process::Command;
    let tmpdir = std::env::temp_dir().join("hal_test_fixture_build_tensors");
    let _ = fs::remove_dir_all(&tmpdir);
    fs::create_dir_all(&tmpdir).unwrap();
    let fixture_path = tmpdir.join("synth.safetensors");

    let venv_py = workspace_root().join("venv/bin/python");
    let py = if venv_py.exists() {
        venv_py.to_string_lossy().into_owned()
    } else {
        "python3".into()
    };
    let script = format!(
        r#"
import numpy as np, safetensors.numpy as stnp, json
stnp.save_file({{
    'input.image': np.zeros((1,4,4,3), dtype=np.uint8),
    'raw.boxes_0':  np.full((1,2,2,4), -3, dtype=np.int8),
    'raw.scores_0': np.full((1,2,2,2), -2, dtype=np.int8),
    'decoded.boxes_xyxy': np.zeros((0,4), dtype=np.float32),
    'decoded.scores': np.zeros((0,), dtype=np.float32),
    'decoded.classes': np.zeros((0,), dtype=np.uint32),
}}, '{p}', metadata={{
    'format_version': '1', 'decoder_family': 'per_scale_yolo_seg',
    'model_basename': 'synth', 'expected_count_min': '0',
    'documentation_md': '#', 'schema_json': '{{}}',
    'quantization_json': json.dumps({{
        'boxes_0':  {{'scale': 0.1, 'zero_point': 0, 'dtype': 'int8'}},
        'scores_0': {{'scale': 0.2, 'zero_point': 5, 'dtype': 'int8'}},
    }}),
    'nms_config_json': '{{"iou_threshold":0.7,"score_threshold":0.001,"max_detections":300}}',
}})
"#,
        p = fixture_path.to_string_lossy()
    );
    let out = Command::new(&py).arg("-c").arg(&script).output();
    match out {
        Ok(o) if o.status.success() => {}
        _ => {
            skip_note!(
                "python3 with numpy/safetensors unavailable; cannot build the synthetic fixture"
            );
            return;
        }
    }

    let fix = common::per_scale_fixture::PerScaleFixture::load(&fixture_path).unwrap();
    let tensors = fix.build_tensors_with_quant().expect("build_tensors");
    assert_eq!(tensors.len(), 2, "expected boxes_0 and scores_0");
    for t in &tensors {
        assert!(
            t.quantization().is_some(),
            "tensor must carry quantization (HAL would reject otherwise)"
        );
    }
}

#[test]
fn fixture_loader_returns_not_present_for_missing_path() {
    let nonexistent = testdata_root().join("decoder/this-fixture-does-not-exist.safetensors");
    let err = common::per_scale_fixture::PerScaleFixture::load(&nonexistent)
        .err()
        .expect("expected an error for a nonexistent path");
    assert!(matches!(
        err,
        common::per_scale_fixture::FixtureError::NotPresent(_)
    ));
}

#[test]
fn decoder_per_scale_pre_nms_capture_entry_point_returns_buffers() {
    use edgefirst_decoder::per_scale::DecodeDtype;
    use edgefirst_decoder::{schema::SchemaV2, DecoderBuilder};
    use edgefirst_tensor::{Quantization, Tensor, TensorDyn};

    let schema_str = minimal_ltrb_schema_one_level();
    let schema: SchemaV2 = serde_json::from_str(schema_str).unwrap();
    let decoder = DecoderBuilder::new()
        .with_schema(schema)
        .with_decode_dtype(DecodeDtype::F32)
        .build()
        .unwrap();

    // Synthesize zero-input tensors: boxes [1,1,1,4] i8, scores [1,1,1,2] i8.
    let boxes_vec = vec![0i8; 4];
    let mut boxes_t: TensorDyn = Tensor::<i8>::from_slice(&boxes_vec, &[1, 1, 1, 4])
        .unwrap()
        .into();
    boxes_t
        .set_quantization(Quantization::per_tensor(0.1, 0))
        .unwrap();

    let scores_vec = vec![0i8; 2];
    let mut scores_t: TensorDyn = Tensor::<i8>::from_slice(&scores_vec, &[1, 1, 1, 2])
        .unwrap()
        .into();
    scores_t
        .set_quantization(Quantization::per_tensor(0.1, 0))
        .unwrap();

    let inputs: Vec<&TensorDyn> = vec![&boxes_t, &scores_t];
    let pre = decoder
        ._testing_run_per_scale_pre_nms(&inputs)
        .expect("pre-NMS capture must succeed");

    // Single anchor at h=1,w=1, stride=8: xc=4, yc=4, w=0, h=0 (zero-input LTRB)
    assert_eq!(pre.boxes_xywh.shape(), &[1, 4]);
    assert!(
        (pre.boxes_xywh[[0, 0]] - 4.0).abs() < 1e-3,
        "xc={}",
        pre.boxes_xywh[[0, 0]]
    );
    assert!(
        (pre.boxes_xywh[[0, 1]] - 4.0).abs() < 1e-3,
        "yc={}",
        pre.boxes_xywh[[0, 1]]
    );
    assert_eq!(pre.scores.shape(), &[1, 2]);
    // Sigmoid(0) = 0.5
    let s0 = pre.scores[[0, 0]];
    assert!((s0 - 0.5).abs() < 1e-3, "score[0]={s0}");
}

/// IoU between two xyxy boxes.
fn box_iou(a: [f32; 4], b: [f32; 4]) -> f32 {
    let x1 = a[0].max(b[0]);
    let y1 = a[1].max(b[1]);
    let x2 = a[2].min(b[2]);
    let y2 = a[3].min(b[3]);
    let inter = (x2 - x1).max(0.0) * (y2 - y1).max(0.0);
    let area_a = (a[2] - a[0]) * (a[3] - a[1]);
    let area_b = (b[2] - b[0]) * (b[3] - b[1]);
    let union = area_a + area_b - inter;
    if union > 0.0 {
        inter / union
    } else {
        0.0
    }
}

/// Greedy IoU-based pairing of fixture detections to HAL detections.
/// Returns `(matched_pairs, mean_iou, class_agreement_count)`.
fn greedy_match_detections(
    fix_boxes: &ndarray::Array2<f32>,
    fix_classes: &ndarray::Array1<u32>,
    hal_boxes: &[DetectBox],
    min_iou: f32,
) -> (usize, f32, usize) {
    let n = fix_boxes.shape()[0];
    let m = hal_boxes.len();
    if n == 0 || m == 0 {
        return (0, 0.0, 0);
    }
    let mut iou = vec![vec![0.0f32; m]; n];
    for (py, iou_row) in fix_boxes.outer_iter().zip(iou.iter_mut()) {
        for (j, h) in hal_boxes.iter().enumerate() {
            iou_row[j] = box_iou(
                [py[0], py[1], py[2], py[3]],
                [h.bbox.xmin, h.bbox.ymin, h.bbox.xmax, h.bbox.ymax],
            );
        }
    }
    // Fixture detections in descending order of their best IoU.
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| {
        let ma = iou[a].iter().cloned().fold(0.0f32, f32::max);
        let mb = iou[b].iter().cloned().fold(0.0f32, f32::max);
        mb.partial_cmp(&ma).unwrap()
    });
    let mut used = vec![false; m];
    let mut matched = 0usize;
    let mut iou_sum = 0.0f32;
    let mut class_match = 0usize;
    for &i in &order {
        // Pick the unused HAL detection with highest IoU vs fixture[i].
        let mut best_j = None;
        let mut best_iou = min_iou;
        for j in 0..m {
            if used[j] {
                continue;
            }
            if iou[i][j] >= best_iou {
                best_iou = iou[i][j];
                best_j = Some(j);
            }
        }
        if let Some(j) = best_j {
            used[j] = true;
            matched += 1;
            iou_sum += best_iou;
            if hal_boxes[j].label as u32 == fix_classes[i] {
                class_match += 1;
            }
        }
    }
    let mean_iou = if matched > 0 {
        iou_sum / matched as f32
    } else {
        0.0
    };
    (matched, mean_iou, class_match)
}

/// Asserts that HAL `decode_proto` output agrees with the fixture's
/// reference detections:
/// - detection count within ±25% of the fixture's count;
/// - at least 80% of fixture detections have a HAL match at IoU ≥ 0.5
///   (greedy pairing);
/// - mean IoU on matched pairs ≥ 0.85;
/// - class agreement on matched pairs ≥ 90%.
fn assert_end_to_end_parity(fix: &common::per_scale_fixture::PerScaleFixture, model_label: &str) {
    let hal_boxes = decode_with_nms(fix, model_label, Nms::ClassAgnostic);

    let dec = fix.decoded().unwrap();
    let n_fix = dec.boxes_xyxy.shape()[0];
    let n_hal = hal_boxes.len();

    // Detection-count tolerance: ±25% of the fixture count, with a floor of 8.
    let lo = ((n_fix as f32) * 0.75).floor() as usize;
    let hi = (((n_fix as f32) * 1.25).ceil() as usize).max(n_fix + 8);
    assert!(
        n_hal >= lo && n_hal <= hi,
        "{model_label}: HAL produced {n_hal} detections, fixture has {n_fix} (allowed [{lo}, {hi}])"
    );

    let (matched, mean_iou, class_match) =
        greedy_match_detections(&dec.boxes_xyxy, &dec.classes, &hal_boxes, 0.5);
    let match_rate = matched as f32 / n_fix as f32;
    let class_agreement = if matched > 0 {
        class_match as f32 / matched as f32
    } else {
        0.0
    };

    assert!(
        match_rate >= 0.80,
        "{model_label}: only {matched}/{n_fix} fixture detections matched a HAL detection at IoU ≥ 0.5 ({:.1}%)",
        match_rate * 100.0
    );
    assert!(
        mean_iou >= 0.85,
        "{model_label}: mean IoU on matched pairs is {mean_iou:.3}, expected ≥ 0.85"
    );
    assert!(
        class_agreement >= 0.90,
        "{model_label}: class agreement {:.1}% on matched pairs, expected ≥ 90%",
        class_agreement * 100.0
    );
}

fn decode_with_nms(
    fix: &common::per_scale_fixture::PerScaleFixture,
    model_label: &str,
    nms_mode: Nms,
) -> Vec<DetectBox> {
    let schema: SchemaV2 = serde_json::from_str(fix.schema_json())
        .expect("fixture schema_json must parse as SchemaV2");
    let nms = fix.nms_config();
    // An effectively unbounded pre-NMS cap, and the fixture's max_detections
    // as `max_det`.
    let pre_nms_cap = 100_000_usize;
    let max_det = nms.max_detections as usize;
    let decoder = DecoderBuilder::default()
        .with_schema(schema)
        .with_decode_dtype(DecodeDtype::F32)
        .with_iou_threshold(nms.iou_threshold)
        .with_score_threshold(nms.score_threshold)
        .with_nms(Some(nms_mode))
        .with_pre_nms_top_k(pre_nms_cap)
        .with_max_det(max_det)
        .build()
        .unwrap_or_else(|e| panic!("{model_label}: build decoder: {e}"));
    let owned_tensors = fix
        .build_tensors_with_quant()
        .unwrap_or_else(|e| panic!("{model_label}: build tensors: {e}"));
    let inputs: Vec<&edgefirst_tensor::TensorDyn> = owned_tensors.iter().collect();
    let mut boxes: Vec<DetectBox> = Vec::with_capacity(nms.max_detections as usize);
    decoder
        .decode_proto(&inputs, &mut boxes)
        .unwrap_or_else(|e| panic!("{model_label}: decode_proto: {e}"));
    // Converts HAL's normalized boxes to pixel space, the space of the
    // fixture's reference coordinates.
    if let Some((w, h)) = decoder.input_dims() {
        let (w, h) = (w as f32, h as f32);
        for b in &mut boxes {
            b.bbox.xmin *= w;
            b.bbox.ymin *= h;
            b.bbox.xmax *= w;
            b.bbox.ymax *= h;
        }
    }
    boxes
}

fn assert_class_aware_superset(
    fix: &common::per_scale_fixture::PerScaleFixture,
    model_label: &str,
) {
    let agnostic = decode_with_nms(fix, model_label, Nms::ClassAgnostic);
    let aware = decode_with_nms(fix, model_label, Nms::ClassAware);
    assert!(
        aware.len() >= agnostic.len(),
        "{model_label}: class-aware NMS returned {} detections, class-agnostic returned {}; class-aware must keep at least as many survivors",
        aware.len(),
        agnostic.len()
    );
}

fn assert_pre_nms_parity(fix: &common::per_scale_fixture::PerScaleFixture, model_label: &str) {
    let schema: SchemaV2 =
        serde_json::from_str(fix.schema_json()).expect("fixture schema_json must parse");
    let decoder = DecoderBuilder::default()
        .with_schema(schema)
        .with_decode_dtype(DecodeDtype::F32)
        .build()
        .unwrap_or_else(|e| panic!("{model_label}: build decoder: {e}"));
    let owned_tensors = fix
        .build_tensors_with_quant()
        .unwrap_or_else(|e| panic!("{model_label}: build tensors: {e}"));
    let inputs: Vec<&edgefirst_tensor::TensorDyn> = owned_tensors.iter().collect();

    let pre = decoder
        ._testing_run_per_scale_pre_nms(&inputs)
        .unwrap_or_else(|e| panic!("{model_label}: pre-NMS: {e}"));
    let inter = fix
        .intermediates()
        .unwrap_or_else(|| panic!("{model_label}: fixture must include intermediates"));
    let num_classes = pre.scores.shape()[1];

    // Total anchors = sum of h*w across FPN levels (8400 for 640×640).
    let mut total = 0usize;
    let mut lvl = 0usize;
    while inter.boxes_xywh(lvl).is_some() {
        total += inter
            .boxes_xywh(lvl)
            .expect("boxes presence checked in while")
            .shape()[0];
        lvl += 1;
    }
    assert_eq!(
        pre.boxes_xywh.shape()[0],
        total,
        "{model_label}: HAL anchor count {} ≠ fixture anchor count {total}",
        pre.boxes_xywh.shape()[0]
    );
    assert_eq!(
        pre.scores.shape()[0],
        total,
        "{model_label}: HAL score rows {} ≠ fixture anchor count {total}",
        pre.scores.shape()[0]
    );

    if let Some(hal_mc) = pre.mask_coefs.as_ref() {
        assert_eq!(
            hal_mc.shape()[0],
            total,
            "{model_label}: HAL mask-coef rows {} ≠ fixture anchor count {total}",
            hal_mc.shape()[0]
        );
    }

    // Per-anchor parity by level.
    let mut offset = 0usize;
    let mut lvl = 0usize;
    while let Some(py_boxes) = inter.boxes_xywh(lvl) {
        let n = py_boxes.shape()[0];
        for i in 0..n {
            for axis in 0..4 {
                let h = pre.boxes_xywh[[offset + i, axis]];
                let p = py_boxes[[i, axis]];
                assert!(
                    (h - p).abs() < 1.0,
                    "{model_label}: lvl {lvl} anchor {i} box axis {axis}: HAL={h} ref={p}"
                );
            }
        }

        let py_scores = inter
            .scores_activated(lvl)
            .unwrap_or_else(|| panic!("{model_label}: missing intermediate.scores_{lvl}.activated"))
            .into_shape_with_order((n, num_classes))
            .unwrap_or_else(|e| {
                panic!(
                    "{model_label}: scores_{lvl}.activated reshape to ({n},{num_classes}) failed: {e}"
                )
            });
        for i in 0..n {
            for c in 0..num_classes {
                let h = pre.scores[[offset + i, c]];
                let p = py_scores[[i, c]];
                assert!(
                    (h - p).abs() < 1e-3,
                    "{model_label}: lvl {lvl} anchor {i} class {c} score: HAL={h} ref={p}"
                );
            }
        }

        if let Some(hal_mc) = pre.mask_coefs.as_ref() {
            let nm = hal_mc.shape()[1];
            let py_mc = inter
                .mc_dequant(lvl)
                .unwrap_or_else(|| panic!("{model_label}: missing intermediate.mc_{lvl}.dequant"))
                .into_shape_with_order((n, nm))
                .unwrap_or_else(|e| {
                    panic!("{model_label}: mc_{lvl}.dequant reshape to ({n},{nm}) failed: {e}")
                });
            for i in 0..n {
                for m in 0..nm {
                    let h = hal_mc[[offset + i, m]];
                    let p = py_mc[[i, m]];
                    assert!(
                        (h - p).abs() < 1e-3,
                        "{model_label}: lvl {lvl} anchor {i} mc {m}: HAL={h} ref={p}"
                    );
                }
            }
        }

        offset += n;
        lvl += 1;
    }

    if let Some(hal_protos) = pre.protos.as_ref() {
        let py_protos = inter
            .protos_dequant()
            .unwrap_or_else(|| panic!("{model_label}: missing intermediate.protos.dequant"));
        let hal_flat = hal_protos
            .as_slice()
            .unwrap_or_else(|| panic!("{model_label}: HAL protos are not contiguous"));
        let py_flat = py_protos
            .as_slice()
            .unwrap_or_else(|| panic!("{model_label}: fixture protos are not contiguous"));
        assert_eq!(
            hal_flat.len(),
            py_flat.len(),
            "{model_label}: proto length mismatch HAL={} fixture={}",
            hal_flat.len(),
            py_flat.len()
        );
        for (idx, (h, p)) in hal_flat.iter().zip(py_flat.iter()).enumerate() {
            assert!(
                (h - p).abs() < 1e-3,
                "{model_label}: proto idx {idx}: HAL={h} ref={p}"
            );
        }
    }
}

fn fixture_path(basename: &str) -> std::path::PathBuf {
    testdata_root().join("decoder").join(basename)
}

// ────────────────────────────────────────────────────────────────────
// yolov8n-seg per-scale parity (fixture-backed)
// ────────────────────────────────────────────────────────────────────

#[test]
fn yolov8n_seg_per_scale_smoke_detection_count() {
    use edgefirst_decoder::per_scale::DecodeDtype;
    use edgefirst_decoder::{schema::SchemaV2, DecoderBuilder, DetectBox, Nms};

    let path = fixture_path("yolov8n-seg.safetensors");
    let fix = match common::per_scale_fixture::PerScaleFixture::load(&path) {
        Ok(f) => f,
        Err(common::per_scale_fixture::FixtureError::NotPresent(_)) => {
            skip_note!("fixture {path:?} not present (run `git lfs pull`)");
            return;
        }
        Err(e) => panic!("fixture load failed: {e}"),
    };
    let schema: SchemaV2 = serde_json::from_str(fix.schema_json())
        .expect("fixture schema_json must parse as SchemaV2");
    let nms = fix.nms_config();
    let decoder = DecoderBuilder::default()
        .with_schema(schema)
        .with_decode_dtype(DecodeDtype::F32)
        .with_iou_threshold(nms.iou_threshold)
        .with_score_threshold(nms.score_threshold)
        .with_nms(Some(Nms::ClassAware))
        .build()
        .expect("build decoder");
    let owned_tensors = fix.build_tensors_with_quant().expect("build tensors");
    let inputs: Vec<&edgefirst_tensor::TensorDyn> = owned_tensors.iter().collect();

    let mut output_boxes: Vec<DetectBox> = Vec::with_capacity(300);
    let _proto = decoder
        .decode_proto(&inputs, &mut output_boxes)
        .expect("decode_proto");

    let n = output_boxes.len();
    assert!(
        n >= fix.expected_count_min as usize,
        "yolov8n-seg per-scale produced {n} detections, expected ≥ {}",
        fix.expected_count_min
    );
}

/// Asserts that per-scale `decode()` succeeds for a real-model fixture and
/// emits one `Segmentation` per `DetectBox`.
fn assert_per_scale_decode_with_masks_succeeds(model_label: &str, fixture_filename: &str) {
    use edgefirst_decoder::per_scale::DecodeDtype;
    use edgefirst_decoder::{schema::SchemaV2, DecoderBuilder, DetectBox, Nms, Segmentation};

    let path = fixture_path(fixture_filename);
    let fix = match common::per_scale_fixture::PerScaleFixture::load(&path) {
        Ok(f) => f,
        Err(common::per_scale_fixture::FixtureError::NotPresent(_)) => {
            skip_note!("fixture {path:?} not present (run `git lfs pull`)");
            return;
        }
        Err(e) => panic!("fixture load failed: {e}"),
    };
    let schema: SchemaV2 = serde_json::from_str(fix.schema_json())
        .expect("fixture schema_json must parse as SchemaV2");
    let nms = fix.nms_config();
    let decoder = DecoderBuilder::default()
        .with_schema(schema)
        .with_decode_dtype(DecodeDtype::F32)
        .with_iou_threshold(nms.iou_threshold)
        .with_score_threshold(nms.score_threshold)
        .with_nms(Some(Nms::ClassAware))
        .build()
        .expect("build decoder");

    assert_eq!(decoder.normalized_boxes(), Some(true));
    assert!(
        decoder.input_dims().is_some(),
        "{model_label} fixture schema must declare input dims so the \
         per-scale bridge can normalize pixel-space boxes"
    );

    let owned_tensors = fix.build_tensors_with_quant().expect("build tensors");
    let inputs: Vec<&edgefirst_tensor::TensorDyn> = owned_tensors.iter().collect();

    let mut output_boxes: Vec<DetectBox> = Vec::with_capacity(300);
    let mut output_masks: Vec<Segmentation> = Vec::with_capacity(300);
    decoder
        .decode(&inputs, &mut output_boxes, &mut output_masks)
        .expect("per-scale decode() must succeed end-to-end with masks");

    let n_boxes = output_boxes.len();
    let n_masks = output_masks.len();
    assert_eq!(
        n_boxes, n_masks,
        "decode() must emit one Segmentation per DetectBox; got {n_boxes} boxes, {n_masks} masks"
    );
    assert!(
        n_boxes >= fix.expected_count_min as usize,
        "{model_label} per-scale decode() produced {n_boxes} detections, expected ≥ {}",
        fix.expected_count_min
    );

    // Every box coordinate lies in the normalized range, within ±2.0.
    let in_norm_range = |v: f32| (-2.0..=2.0).contains(&v);
    for b in &output_boxes {
        assert!(
            in_norm_range(b.bbox.xmin)
                && in_norm_range(b.bbox.ymin)
                && in_norm_range(b.bbox.xmax)
                && in_norm_range(b.bbox.ymax),
            "{model_label} per-scale decode() bbox {:?} not in normalized range — \
             per_scale_to_masks did not normalize pixel-space coords",
            b.bbox
        );
    }
}

#[test]
fn yolov8n_seg_per_scale_decode_with_masks_succeeds() {
    assert_per_scale_decode_with_masks_succeeds("yolov8n-seg", "yolov8n-seg.safetensors");
}

#[test]
fn yolov8n_seg_per_scale_end_to_end_parity() {
    let path = fixture_path("yolov8n-seg.safetensors");
    let fix = match common::per_scale_fixture::PerScaleFixture::load(&path) {
        Ok(f) => f,
        Err(common::per_scale_fixture::FixtureError::NotPresent(_)) => {
            skip_note!("fixture {path:?} not present (run `git lfs pull`)");
            return;
        }
        Err(e) => panic!("{e}"),
    };
    assert_end_to_end_parity(&fix, "yolov8n-seg");
}

#[test]
fn yolov8n_seg_per_scale_class_aware_superset() {
    let path = fixture_path("yolov8n-seg.safetensors");
    let fix = match common::per_scale_fixture::PerScaleFixture::load(&path) {
        Ok(f) => f,
        Err(common::per_scale_fixture::FixtureError::NotPresent(_)) => {
            skip_note!("fixture {path:?} not present (run `git lfs pull`)");
            return;
        }
        Err(e) => panic!("{e}"),
    };
    assert_class_aware_superset(&fix, "yolov8n-seg");
}

#[test]
fn yolov8n_seg_per_scale_pre_nms_parity() {
    let path = fixture_path("yolov8n-seg.safetensors");
    let fix = match common::per_scale_fixture::PerScaleFixture::load(&path) {
        Ok(f) => f,
        Err(common::per_scale_fixture::FixtureError::NotPresent(_)) => {
            skip_note!("fixture not present");
            return;
        }
        Err(e) => panic!("{e}"),
    };
    assert_pre_nms_parity(&fix, "yolov8n-seg");
}

/// Detection counts at several `pre_nms_top_k` values: 0 and 8400 match the unbounded count, 3000 keeps ≥ 99%, 1000 keeps ≥ 50% and 300 keeps < 50%.
#[test]
fn yolov8n_seg_pre_nms_top_k_sensitivity() {
    use edgefirst_decoder::per_scale::DecodeDtype;
    use edgefirst_decoder::{schema::SchemaV2, DecoderBuilder, DetectBox, Nms};

    let path = fixture_path("yolov8n-seg.safetensors");
    let fix = match common::per_scale_fixture::PerScaleFixture::load(&path) {
        Ok(f) => f,
        Err(common::per_scale_fixture::FixtureError::NotPresent(_)) => {
            skip_note!("fixture not present");
            return;
        }
        Err(e) => panic!("{e}"),
    };

    let schema_str = fix.schema_json();
    let nms_config = fix.nms_config();

    let decode_with_top_k = |top_k: usize| -> usize {
        let schema: SchemaV2 =
            serde_json::from_str(schema_str).expect("fixture schema_json must parse");
        let decoder = DecoderBuilder::default()
            .with_schema(schema)
            .with_decode_dtype(DecodeDtype::F32)
            .with_iou_threshold(nms_config.iou_threshold)
            .with_score_threshold(nms_config.score_threshold)
            .with_nms(Some(Nms::ClassAgnostic))
            .with_pre_nms_top_k(top_k)
            .with_max_det(100_000) // uncapped
            .build()
            .unwrap_or_else(|e| panic!("build decoder (top_k={top_k}): {e}"));
        let owned_tensors = fix.build_tensors_with_quant().expect("build tensors");
        let inputs: Vec<&edgefirst_tensor::TensorDyn> = owned_tensors.iter().collect();
        let mut boxes: Vec<DetectBox> = Vec::with_capacity(300);
        decoder
            .decode_proto(&inputs, &mut boxes)
            .expect("decode_proto");
        boxes.len()
    };

    let n_unbounded = decode_with_top_k(100_000);
    let n_zero = decode_with_top_k(0);
    let n_8400 = decode_with_top_k(8400);
    let n_3000 = decode_with_top_k(3000);
    let n_1000 = decode_with_top_k(1000);
    let n_300 = decode_with_top_k(300);

    eprintln!(
        "pre_nms_top_k sensitivity (yolov8n-seg):\n  \
         unbounded={n_unbounded}, zero={n_zero}, 8400={n_8400}, 3000={n_3000}, \
         1000={n_1000}, 300={n_300}"
    );

    assert!(
        n_unbounded >= fix.expected_count_min as usize,
        "unbounded pre_nms_top_k produced only {n_unbounded} detections"
    );

    assert_eq!(
        n_zero, n_unbounded,
        "pre_nms_top_k=0 (no limit) should match unbounded ({n_unbounded}), got {n_zero}"
    );

    assert_eq!(
        n_8400, n_unbounded,
        "pre_nms_top_k=8400 should match unbounded ({n_unbounded}), got {n_8400}"
    );

    let ratio_3000 = n_3000 as f32 / n_unbounded as f32;
    let ratio_1000 = n_1000 as f32 / n_unbounded as f32;
    assert!(
        ratio_3000 >= 0.99,
        "pre_nms_top_k=3000 retained only {:.0}% of unbounded detections ({n_3000}/{n_unbounded})",
        ratio_3000 * 100.0
    );
    assert!(
        ratio_1000 >= 0.50,
        "pre_nms_top_k=1000 retained only {:.0}% of unbounded detections ({n_1000}/{n_unbounded})",
        ratio_1000 * 100.0
    );

    let ratio_300 = n_300 as f32 / n_unbounded as f32;
    assert!(
        ratio_300 < 0.50,
        "pre_nms_top_k=300 should retain <50% of unbounded detections, got {:.0}%",
        ratio_300 * 100.0
    );
    eprintln!(
        "  detection retention: 3000→{:.0}%, 1000→{:.0}%, 300→{:.0}%",
        ratio_3000 * 100.0,
        ratio_1000 * 100.0,
        ratio_300 * 100.0
    );
}

// ────────────────────────────────────────────────────────────────────
// yolo11n-seg per-scale parity (fixture-backed)
// ────────────────────────────────────────────────────────────────────

#[test]
fn yolo11n_seg_per_scale_smoke_detection_count() {
    use edgefirst_decoder::per_scale::DecodeDtype;
    use edgefirst_decoder::{schema::SchemaV2, DecoderBuilder, DetectBox, Nms};

    let path = fixture_path("yolo11n-seg.safetensors");
    let fix = match common::per_scale_fixture::PerScaleFixture::load(&path) {
        Ok(f) => f,
        Err(common::per_scale_fixture::FixtureError::NotPresent(_)) => {
            skip_note!("fixture {path:?} not present (run `git lfs pull`)");
            return;
        }
        Err(e) => panic!("fixture load failed: {e}"),
    };
    let schema: SchemaV2 = serde_json::from_str(fix.schema_json())
        .expect("fixture schema_json must parse as SchemaV2");
    let nms = fix.nms_config();
    let decoder = DecoderBuilder::default()
        .with_schema(schema)
        .with_decode_dtype(DecodeDtype::F32)
        .with_iou_threshold(nms.iou_threshold)
        .with_score_threshold(nms.score_threshold)
        .with_nms(Some(Nms::ClassAware))
        .build()
        .expect("build decoder");
    let owned_tensors = fix.build_tensors_with_quant().expect("build tensors");
    let inputs: Vec<&edgefirst_tensor::TensorDyn> = owned_tensors.iter().collect();

    let mut output_boxes: Vec<DetectBox> = Vec::with_capacity(300);
    let _proto = decoder
        .decode_proto(&inputs, &mut output_boxes)
        .expect("decode_proto");

    let n = output_boxes.len();
    assert!(
        n >= fix.expected_count_min as usize,
        "yolo11n-seg per-scale produced {n} detections, expected ≥ {}",
        fix.expected_count_min
    );
}

#[test]
fn yolo11n_seg_per_scale_end_to_end_parity() {
    let path = fixture_path("yolo11n-seg.safetensors");
    let fix = match common::per_scale_fixture::PerScaleFixture::load(&path) {
        Ok(f) => f,
        Err(common::per_scale_fixture::FixtureError::NotPresent(_)) => {
            skip_note!("fixture {path:?} not present (run `git lfs pull`)");
            return;
        }
        Err(e) => panic!("{e}"),
    };
    assert_end_to_end_parity(&fix, "yolo11n-seg");
}

#[test]
fn yolo11n_seg_per_scale_class_aware_superset() {
    let path = fixture_path("yolo11n-seg.safetensors");
    let fix = match common::per_scale_fixture::PerScaleFixture::load(&path) {
        Ok(f) => f,
        Err(common::per_scale_fixture::FixtureError::NotPresent(_)) => {
            skip_note!("fixture {path:?} not present (run `git lfs pull`)");
            return;
        }
        Err(e) => panic!("{e}"),
    };
    assert_class_aware_superset(&fix, "yolo11n-seg");
}

#[test]
fn yolo11n_seg_per_scale_pre_nms_parity() {
    let path = fixture_path("yolo11n-seg.safetensors");
    let fix = match common::per_scale_fixture::PerScaleFixture::load(&path) {
        Ok(f) => f,
        Err(common::per_scale_fixture::FixtureError::NotPresent(_)) => {
            skip_note!("fixture not present");
            return;
        }
        Err(e) => panic!("{e}"),
    };
    assert_pre_nms_parity(&fix, "yolo11n-seg");
}

#[test]
fn yolo11n_seg_per_scale_decode_with_masks_succeeds() {
    assert_per_scale_decode_with_masks_succeeds("yolo11n-seg", "yolo11n-seg.safetensors");
}

// ────────────────────────────────────────────────────────────────────
// yolo26n-seg per-scale parity (LTRB-encoded, fixture-backed)
// ────────────────────────────────────────────────────────────────────
//
// yolo26 uses LTRB box encoding (no DFL stage).

#[test]
fn yolo26n_seg_per_scale_smoke_detection_count() {
    use edgefirst_decoder::per_scale::DecodeDtype;
    use edgefirst_decoder::{schema::SchemaV2, DecoderBuilder, DetectBox, Nms};

    let path = fixture_path("yolo26n-seg.safetensors");
    let fix = match common::per_scale_fixture::PerScaleFixture::load(&path) {
        Ok(f) => f,
        Err(common::per_scale_fixture::FixtureError::NotPresent(_)) => {
            skip_note!("fixture {path:?} not present (run `git lfs pull`)");
            return;
        }
        Err(e) => panic!("fixture load failed: {e}"),
    };
    let schema: SchemaV2 = serde_json::from_str(fix.schema_json())
        .expect("fixture schema_json must parse as SchemaV2");
    let nms = fix.nms_config();
    let decoder = DecoderBuilder::default()
        .with_schema(schema)
        .with_decode_dtype(DecodeDtype::F32)
        .with_iou_threshold(nms.iou_threshold)
        .with_score_threshold(nms.score_threshold)
        .with_nms(Some(Nms::ClassAware))
        .build()
        .expect("build decoder");
    let owned_tensors = fix.build_tensors_with_quant().expect("build tensors");
    let inputs: Vec<&edgefirst_tensor::TensorDyn> = owned_tensors.iter().collect();

    let mut output_boxes: Vec<DetectBox> = Vec::with_capacity(300);
    let _proto = decoder
        .decode_proto(&inputs, &mut output_boxes)
        .expect("decode_proto");

    let n = output_boxes.len();
    assert!(
        n >= fix.expected_count_min as usize,
        "yolo26n-seg per-scale produced {n} detections, expected ≥ {}",
        fix.expected_count_min
    );
}

#[test]
fn yolo26n_seg_per_scale_end_to_end_parity() {
    let path = fixture_path("yolo26n-seg.safetensors");
    let fix = match common::per_scale_fixture::PerScaleFixture::load(&path) {
        Ok(f) => f,
        Err(common::per_scale_fixture::FixtureError::NotPresent(_)) => {
            skip_note!("fixture {path:?} not present (run `git lfs pull`)");
            return;
        }
        Err(e) => panic!("{e}"),
    };
    assert_end_to_end_parity(&fix, "yolo26n-seg");
}

#[test]
fn yolo26n_seg_per_scale_class_aware_superset() {
    let path = fixture_path("yolo26n-seg.safetensors");
    let fix = match common::per_scale_fixture::PerScaleFixture::load(&path) {
        Ok(f) => f,
        Err(common::per_scale_fixture::FixtureError::NotPresent(_)) => {
            skip_note!("fixture {path:?} not present (run `git lfs pull`)");
            return;
        }
        Err(e) => panic!("{e}"),
    };
    assert_class_aware_superset(&fix, "yolo26n-seg");
}

#[test]
fn yolo26n_seg_per_scale_pre_nms_parity() {
    let path = fixture_path("yolo26n-seg.safetensors");
    let fix = match common::per_scale_fixture::PerScaleFixture::load(&path) {
        Ok(f) => f,
        Err(common::per_scale_fixture::FixtureError::NotPresent(_)) => {
            skip_note!("fixture not present");
            return;
        }
        Err(e) => panic!("{e}"),
    };
    assert_pre_nms_parity(&fix, "yolo26n-seg");
}

/// `Decoder::decode` with masks succeeds for the LTRB-encoded model.
#[test]
fn yolo26n_seg_per_scale_decode_with_masks_succeeds() {
    assert_per_scale_decode_with_masks_succeeds("yolo26n-seg", "yolo26n-seg.safetensors");
}

// ────────────────────────────────────────────────────────────────────
// Multi-label decode tests
// ────────────────────────────────────────────────────────────────────
//
// These tests use the NHWC 2×2 fixture
// (`minimal_ltrb_schema_one_level_2x2_nhwc`).

/// Build an argmax or multi-label decoder from the 2×2 NHWC schema.
fn build_2x2_decoder(multi_label: bool) -> edgefirst_decoder::Decoder {
    use edgefirst_decoder::per_scale::DecodeDtype;
    let schema: SchemaV2 = serde_json::from_str(minimal_ltrb_schema_one_level_2x2_nhwc()).unwrap();
    DecoderBuilder::default()
        .with_schema(schema)
        .with_decode_dtype(DecodeDtype::F32)
        .with_iou_threshold(0.7)
        .with_score_threshold(0.01)
        .with_nms(Some(Nms::ClassAware))
        .with_pre_nms_top_k(0) // unbounded
        .with_multi_label(multi_label)
        .build()
        .expect("build")
}

/// Encode 4-anchor score data that puts two classes above threshold on
/// anchor 0 and one class above threshold on anchor 1.
///
/// Score layout NHWC [1,2,2,2] (anchors row-major, NC=2):
///   anchor 0: class 0 logit = 50 → sigmoid≈0.99, class 1 logit = 40 → sigmoid≈0.98
///   anchor 1: class 0 logit = 30 → sigmoid≈0.95, class 1 logit = 0  → sigmoid=0.5
///   anchor 2: all 0 (sigmoid=0.5)
///   anchor 3: all 0
fn make_multi_label_score_tensor() -> TensorDyn {
    // NHWC [1, 2, 2, 2]: row-major over h×w, then NC per cell
    let data: Vec<i8> = vec![
        50, 40, // anchor (0,0): class0 high, class1 also high
        30, 0, // anchor (0,1): class0 above threshold, class1 at 0.5
        0, 0, // anchor (1,0): both at 0.5
        0, 0, // anchor (1,1): both at 0.5
    ];
    let mut t = Tensor::<i8>::new(&[1, 2, 2, 2], Some(TensorMemory::Mem), None).unwrap();
    t.set_quantization(TQ::per_tensor(0.1, 0)).unwrap();
    {
        let mut m = t.map().unwrap();
        m.as_mut_slice().copy_from_slice(&data);
    }
    TensorDyn::I8(t)
}

/// Standard zero-box tensor for the 2×2 fixture (4 anchors, all LTRB=0).
fn make_zero_box_tensor() -> TensorDyn {
    let mut t = Tensor::<i8>::new(&[1, 2, 2, 4], Some(TensorMemory::Mem), None).unwrap();
    t.set_quantization(TQ::per_tensor(0.1, 0)).unwrap();
    TensorDyn::I8(t)
}

/// Multi-label emits more candidates than argmax when several classes of one anchor exceed the threshold.
#[test]
fn multi_label_count_exceeds_argmax() {
    let box_t = make_zero_box_tensor();
    let score_t = make_multi_label_score_tensor();
    let inputs: Vec<&TensorDyn> = vec![&box_t, &score_t];

    let decoder_argmax = build_2x2_decoder(false);
    let decoder_multilabel = build_2x2_decoder(true);

    let mut argmax_boxes: Vec<DetectBox> = Vec::with_capacity(10);
    let mut ml_boxes: Vec<DetectBox> = Vec::with_capacity(10);
    let mut masks = Vec::new();

    decoder_argmax
        .decode(&inputs, &mut argmax_boxes, &mut masks)
        .expect("argmax decode");
    decoder_multilabel
        .decode(&inputs, &mut ml_boxes, &mut masks)
        .expect("multilabel decode");

    assert!(
        ml_boxes.len() > argmax_boxes.len(),
        "multi-label ({}) should yield more candidates than argmax ({})",
        ml_boxes.len(),
        argmax_boxes.len()
    );
    // Argmax takes class 0 for anchor 0.
    let argmax_anchor0 = argmax_boxes
        .iter()
        .find(|b| b.score > 0.98)
        .expect("argmax should have a high-score box");
    assert_eq!(
        argmax_anchor0.label, 0,
        "argmax should pick class 0 (highest logit)"
    );
    // Multi-label includes both class 0 and class 1 for anchor 0.
    let has_class0 = ml_boxes.iter().any(|b| b.label == 0 && b.score > 0.98);
    let has_class1 = ml_boxes.iter().any(|b| b.label == 1 && b.score > 0.95);
    assert!(has_class0, "multi-label must include class 0 from anchor 0");
    assert!(has_class1, "multi-label must include class 1 from anchor 0");
}

/// Over 3 frames of one object, argmax detections give one track and multi-label detections give more.
#[cfg(feature = "tracker")]
#[test]
fn tracker_regression_argmax_one_track_multilabel_spawns_extra() {
    use edgefirst_tracker::{ByteTrackBuilder, MockDetection, Tracker};

    // A single object occupying [0.1, 0.1, 0.5, 0.5] across all 3 frames.
    let bbox = [0.1_f32, 0.1, 0.5, 0.5];

    // Argmax: one detection per frame, label 0.
    let argmax_dets: Vec<MockDetection> = vec![MockDetection::new(bbox, 0.9, 0)];

    // Multi-label: two detections per frame — same bbox, labels 0 and 1.
    let multilabel_dets: Vec<MockDetection> = vec![
        MockDetection::new(bbox, 0.9, 0),
        MockDetection::new(bbox, 0.85, 1), // second class, same anchor region
    ];

    // ── Argmax tracker ──────────────────────────────────────────────
    let mut argmax_tracker = ByteTrackBuilder::new()
        .track_high_conf(0.5)
        .track_iou(0.25)
        .build::<MockDetection>();

    for ts in [1_000_000_u64, 2_000_000, 3_000_000] {
        argmax_tracker.update(&argmax_dets, ts);
    }

    let argmax_tracks = argmax_tracker.get_active_tracks();
    assert_eq!(
        argmax_tracks.len(),
        1,
        "argmax should yield exactly 1 track for 1 object; got {}",
        argmax_tracks.len()
    );
    assert!(
        argmax_tracks[0].info.count >= 3,
        "track should have been updated 3 times; got count={}",
        argmax_tracks[0].info.count
    );

    // ── Multi-label tracker ─────────────────────────────────────────
    let mut ml_tracker = ByteTrackBuilder::new()
        .track_high_conf(0.5)
        .track_iou(0.25)
        .build::<MockDetection>();

    for ts in [1_000_000_u64, 2_000_000, 3_000_000] {
        ml_tracker.update(&multilabel_dets, ts);
    }

    let ml_tracks = ml_tracker.get_active_tracks();
    assert!(
        ml_tracks.len() > argmax_tracks.len(),
        "multi-label should produce more phantom tracks ({}) than argmax ({}); \
         documents why deployment must stay on argmax",
        ml_tracks.len(),
        argmax_tracks.len()
    );
}

/// Argmax decode with `with_multi_label(false)` is bit-identical to a decoder built without the call.
#[test]
fn argmax_regression_bit_identical_with_and_without_flag() {
    use edgefirst_decoder::per_scale::DecodeDtype;

    let make = |multi_label: Option<bool>| -> edgefirst_decoder::Decoder {
        let schema: SchemaV2 =
            serde_json::from_str(minimal_ltrb_schema_one_level_2x2_nhwc()).unwrap();
        let mut b = DecoderBuilder::default()
            .with_schema(schema)
            .with_decode_dtype(DecodeDtype::F32)
            .with_iou_threshold(0.7)
            .with_score_threshold(0.5)
            .with_nms(Some(Nms::ClassAgnostic));
        if let Some(ml) = multi_label {
            b = b.with_multi_label(ml);
        }
        b.build().expect("build")
    };

    let box_t = make_zero_box_tensor();
    let score_t = {
        // Only anchor 0 class 1 wins with score 0.99
        let data: Vec<i8> = vec![0, 50, 0, 0, 0, 0, 0, 0];
        let mut t = Tensor::<i8>::new(&[1, 2, 2, 2], Some(TensorMemory::Mem), None).unwrap();
        t.set_quantization(TQ::per_tensor(0.1, 0)).unwrap();
        {
            let mut m = t.map().unwrap();
            m.as_mut_slice().copy_from_slice(&data);
        }
        TensorDyn::I8(t)
    };
    let inputs: Vec<&TensorDyn> = vec![&box_t, &score_t];

    let default_dec = make(None); // never calls with_multi_label
    let explicit_argmax = make(Some(false)); // calls with_multi_label(false)

    let mut boxes_default: Vec<DetectBox> = Vec::with_capacity(10);
    let mut boxes_explicit: Vec<DetectBox> = Vec::with_capacity(10);
    let mut masks = Vec::new();

    default_dec
        .decode(&inputs, &mut boxes_default, &mut masks)
        .unwrap();
    explicit_argmax
        .decode(&inputs, &mut boxes_explicit, &mut masks)
        .unwrap();

    assert_eq!(
        boxes_default.len(),
        boxes_explicit.len(),
        "output length differs with vs. without explicit with_multi_label(false)"
    );
    for (a, b) in boxes_default.iter().zip(boxes_explicit.iter()) {
        assert_eq!(a.label, b.label);
        assert_eq!(a.score, b.score, "scores must be bit-identical");
        assert_eq!(a.bbox.xmin, b.bbox.xmin);
        assert_eq!(a.bbox.ymin, b.bbox.ymin);
        assert_eq!(a.bbox.xmax, b.bbox.xmax);
        assert_eq!(a.bbox.ymax, b.bbox.ymax);
    }
}

// ────────────────────────────────────────────────────────────────────
// Per-channel DFL dequantization tests
// ────────────────────────────────────────────────────────────────────

/// Minimal DFL schema: 1 level (stride=8, h=1, w=1, reg_max=4, NC=1).
/// Box channels = 4 * 4 = 16. Per-channel scales: scale[i] = 0.01*(i+1).
///
/// The parent logical uses `box_coords: 4` (the decoded coord count) and the
/// child uses `num_features: 16` (the raw DFL bin count).
fn minimal_dfl_schema_per_channel() -> String {
    // Build the 16-element scale array: [0.01, 0.02, ..., 0.16]
    let scales: Vec<String> = (1..=16)
        .map(|i| format!("{:.2}", i as f32 * 0.01))
        .collect();
    let scales_json = scales.join(",");
    format!(
        r#"{{
        "schema_version": 2,
        "nms": "class_agnostic",
        "input": {{
            "shape": [1, 8, 8, 3],
            "dshape": [{{"batch": 1}}, {{"height": 8}}, {{"width": 8}}, {{"num_features": 3}}],
            "cameraadaptor": "rgb"
        }},
        "outputs": [
            {{
                "name": "boxes", "type": "boxes",
                "shape": [1, 4, 1],
                "dshape": [{{"batch": 1}}, {{"box_coords": 4}}, {{"num_boxes": 1}}],
                "encoding": "dfl", "decoder": "ultralytics", "normalized": true,
                "outputs": [{{
                    "name": "boxes_0", "type": "boxes",
                    "stride": 8, "scale_index": 0,
                    "shape": [1, 1, 1, 16],
                    "dshape": [{{"batch": 1}}, {{"height": 1}}, {{"width": 1}}, {{"num_features": 16}}],
                    "dtype": "int8",
                    "quantization": {{"scale": [{scales_json}], "zero_point": 0, "dtype": "int8"}}
                }}]
            }},
            {{
                "name": "scores", "type": "scores",
                "shape": [1, 1, 1],
                "dshape": [{{"batch": 1}}, {{"num_classes": 1}}, {{"num_boxes": 1}}],
                "score_format": "per_class", "decoder": "ultralytics",
                "outputs": [{{
                    "name": "scores_0", "type": "scores",
                    "stride": 8, "scale_index": 0,
                    "shape": [1, 1, 1, 1],
                    "dshape": [{{"batch": 1}}, {{"height": 1}}, {{"width": 1}}, {{"num_classes": 1}}],
                    "activation_required": "sigmoid",
                    "dtype": "int8",
                    "quantization": {{"scale": 0.1, "zero_point": 0, "dtype": "int8"}}
                }}]
            }}
        ]
    }}"#,
        scales_json = scales_json,
    )
}

/// Compute the expected DFL box output for the per-channel case by hand.
/// reg_max=4, 1x1 anchor (gx=0.5, gy=0.5), stride=8.
/// input[i] = i as i8 (0,1,2,...,15); zp=0; scale[i] = 0.01*(i+1).
/// deq[i] = i * 0.01*(i+1).
fn compute_per_channel_golden(input: &[i8], scales: &[f32]) -> [f32; 4] {
    let reg_max = 4;
    let mut deq = [0.0_f32; 16];
    for i in 0..16 {
        deq[i] = input[i] as f32 * scales[i]; // zp=0
    }
    // softmax per side
    for side in 0..4 {
        let slice = &mut deq[side * reg_max..(side + 1) * reg_max];
        let max = slice.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let sum: f32 = slice.iter().map(|&x| (x - max).exp()).sum();
        for v in slice.iter_mut() {
            *v = (*v - max).exp() / sum;
        }
    }
    // weighted_sum per side (bins 0..reg_max weighted by distance value)
    let mut ltrb = [0.0_f32; 4];
    for side in 0..4 {
        for bin in 0..reg_max {
            ltrb[side] += deq[side * reg_max + bin] * bin as f32;
        }
    }
    // dist2bbox at anchor (0.5, 0.5) stride=8:
    // xc = (gx + (r - l)/2) * stride = (0.5 + (ltrb[2]-ltrb[0])/2) * 8
    // yc = (gy + (b - t)/2) * stride
    // w  = (l + r) * stride
    // h  = (t + b) * stride
    let (l, t, r, b) = (ltrb[0], ltrb[1], ltrb[2], ltrb[3]);
    let stride = 8.0_f32;
    let gx = 0.5_f32;
    let gy = 0.5_f32;
    let xc = (gx + (r - l) / 2.0) * stride;
    let yc = (gy + (b - t) / 2.0) * stride;
    let w = (l + r) * stride;
    let h = (t + b) * stride;
    [xc, yc, w, h]
}

/// Per-channel DFL decode matches a hand-computed golden, and per-tensor decode of the same data diverges.
#[test]
fn dfl_per_channel_dequant_matches_golden_and_per_tensor_diverges() {
    use edgefirst_tensor::Quantization as TQ;

    let schema_json = minimal_dfl_schema_per_channel();

    // Build the per-channel scales: scale[i] = 0.01*(i+1)
    let scales: Vec<f32> = (1..=16).map(|i| i as f32 * 0.01).collect();

    // Input int8 values: input[i] = i as i8 (0..15)
    let input_data: Vec<i8> = (0i8..16).collect();

    let golden = compute_per_channel_golden(&input_data, &scales);

    let schema_pc: SchemaV2 = serde_json::from_str(&schema_json).unwrap();
    let decoder_pc = DecoderBuilder::default()
        .with_schema(schema_pc)
        .with_decode_dtype(DecodeDtype::F32)
        .with_iou_threshold(0.7)
        .with_score_threshold(0.0001)
        .with_nms(Some(Nms::ClassAgnostic))
        .build()
        .expect("build per-channel decoder");

    // Build box tensor with per-channel quantization attached.
    let mut box_t_pc = Tensor::<i8>::new(&[1, 1, 1, 16], Some(TensorMemory::Mem), None).unwrap();
    {
        // Attach per-channel quantization: 16 distinct scales, zp=0 for all.
        let zps: Vec<i32> = vec![0i32; 16];
        box_t_pc
            .set_quantization(TQ::per_channel(scales.clone(), zps, 3).unwrap())
            .unwrap();
        let mut m = box_t_pc.map().unwrap();
        m.as_mut_slice().copy_from_slice(&input_data);
    }
    let mut score_t = Tensor::<i8>::new(&[1, 1, 1, 1], Some(TensorMemory::Mem), None).unwrap();
    score_t.set_quantization(TQ::per_tensor(0.1, 0)).unwrap();
    {
        let mut m = score_t.map().unwrap();
        m.as_mut_slice()[0] = 50i8; // sigmoid(5.0) ≈ 0.993
    }

    let dyn_box_pc = TensorDyn::I8(box_t_pc);
    let dyn_score_pc = TensorDyn::I8(score_t);
    let inputs_pc: Vec<&TensorDyn> = vec![&dyn_box_pc, &dyn_score_pc];

    let pre_pc = decoder_pc
        ._testing_run_per_scale_pre_nms(&inputs_pc)
        .expect("per-channel pre-NMS must succeed");

    // Box output is XYWH at anchor (0.5, 0.5)*8 = pixel (4, 4) plus corrections.
    let xc_pc = pre_pc.boxes_xywh[[0, 0]];
    let yc_pc = pre_pc.boxes_xywh[[0, 1]];
    let w_pc = pre_pc.boxes_xywh[[0, 2]];
    let h_pc = pre_pc.boxes_xywh[[0, 3]];

    assert!(
        (xc_pc - golden[0]).abs() < 1e-3,
        "per-channel xc={xc_pc:.4} vs golden={:.4}",
        golden[0]
    );
    assert!(
        (yc_pc - golden[1]).abs() < 1e-3,
        "per-channel yc={yc_pc:.4} vs golden={:.4}",
        golden[1]
    );
    assert!(
        (w_pc - golden[2]).abs() < 1e-3,
        "per-channel w={w_pc:.4} vs golden={:.4}",
        golden[2]
    );
    assert!(
        (h_pc - golden[3]).abs() < 1e-3,
        "per-channel h={h_pc:.4} vs golden={:.4}",
        golden[3]
    );

    // The same schema with a per-tensor-quantized box tensor (single scale 0.01).
    let schema_pt: SchemaV2 = serde_json::from_str(&schema_json).unwrap();
    let decoder_pt = DecoderBuilder::default()
        .with_schema(schema_pt)
        .with_decode_dtype(DecodeDtype::F32)
        .with_iou_threshold(0.7)
        .with_score_threshold(0.0001)
        .with_nms(Some(Nms::ClassAgnostic))
        .build()
        .expect("build per-tensor decoder");

    let mut box_t_pt = Tensor::<i8>::new(&[1, 1, 1, 16], Some(TensorMemory::Mem), None).unwrap();
    box_t_pt.set_quantization(TQ::per_tensor(0.01, 0)).unwrap();
    {
        let mut m = box_t_pt.map().unwrap();
        m.as_mut_slice().copy_from_slice(&input_data);
    }
    // The score tensor for the per-tensor run.
    let mut score_t2 = Tensor::<i8>::new(&[1, 1, 1, 1], Some(TensorMemory::Mem), None).unwrap();
    score_t2.set_quantization(TQ::per_tensor(0.1, 0)).unwrap();
    {
        let mut m = score_t2.map().unwrap();
        m.as_mut_slice()[0] = 50i8;
    }
    let dyn_box_pt = TensorDyn::I8(box_t_pt);
    let dyn_score2 = TensorDyn::I8(score_t2);
    let inputs_pt: Vec<&TensorDyn> = vec![&dyn_box_pt, &dyn_score2];

    let pre_pt = decoder_pt
        ._testing_run_per_scale_pre_nms(&inputs_pt)
        .expect("per-tensor pre-NMS must succeed");

    let xc_pt = pre_pt.boxes_xywh[[0, 0]];
    let yc_pt = pre_pt.boxes_xywh[[0, 1]];
    let w_pt = pre_pt.boxes_xywh[[0, 2]];
    let h_pt = pre_pt.boxes_xywh[[0, 3]];

    let max_diff = [
        (xc_pc - xc_pt).abs(),
        (yc_pc - yc_pt).abs(),
        (w_pc - w_pt).abs(),
        (h_pc - h_pt).abs(),
    ]
    .iter()
    .cloned()
    .fold(0.0f32, f32::max);

    assert!(
        max_diff > 0.01,
        "per-channel and per-tensor results should diverge when scales differ; \
         max_diff={max_diff:.4} (xc_pc={xc_pc:.4} vs xc_pt={xc_pt:.4}, \
         w_pc={w_pc:.4} vs w_pt={w_pt:.4})"
    );
}

/// A per-tensor-quantized box tensor decodes through the per-tensor path.
#[test]
fn dfl_per_tensor_path_uses_scalar_fast_path() {
    let schema_json = minimal_ltrb_schema_one_level();
    let schema: SchemaV2 = serde_json::from_str(schema_json).unwrap();

    let decoder = DecoderBuilder::default()
        .with_schema(schema)
        .with_decode_dtype(DecodeDtype::F32)
        .with_iou_threshold(0.7)
        .with_score_threshold(0.0001)
        .with_nms(Some(Nms::ClassAgnostic))
        .build()
        .expect("build");

    let mut box_t = Tensor::<i8>::new(&[1, 1, 1, 4], Some(TensorMemory::Mem), None).unwrap();
    box_t.set_quantization(TQ::per_tensor(0.1, 0)).unwrap();
    let mut score_t = Tensor::<i8>::new(&[1, 1, 1, 2], Some(TensorMemory::Mem), None).unwrap();
    score_t.set_quantization(TQ::per_tensor(0.1, 0)).unwrap();
    {
        let mut m = score_t.map().unwrap();
        m.as_mut_slice()[0] = 50;
    }

    let dyn_box = TensorDyn::I8(box_t);
    let dyn_score = TensorDyn::I8(score_t);
    let inputs: Vec<&TensorDyn> = vec![&dyn_box, &dyn_score];

    let mut output_boxes: Vec<DetectBox> = Vec::with_capacity(10);
    let mut masks = Vec::new();
    decoder
        .decode(&inputs, &mut output_boxes, &mut masks)
        .expect("decode");

    // One detection, centered at (0.5, 0.5) normalized.
    assert_eq!(output_boxes.len(), 1, "expected 1 detection");
    let det = &output_boxes[0];
    assert!(
        (det.bbox.xmin - 0.5).abs() < 1e-3,
        "xmin={} (expected 0.5)",
        det.bbox.xmin
    );
    assert_eq!(det.label, 0, "expected class 0");
    assert!(det.score > 0.99, "expected high score, got {}", det.score);
}

/// Tracked per-scale decode ignores multi-label and matches argmax.
#[cfg(feature = "tracker")]
#[test]
fn tracked_per_scale_decode_ignores_multi_label() {
    use edgefirst_tracker::ByteTrackBuilder;
    let box_t = make_zero_box_tensor();
    let score_t = make_multi_label_score_tensor();
    let inputs: Vec<&TensorDyn> = vec![&box_t, &score_t];
    let run = |multi_label: bool| {
        let d = build_2x2_decoder(multi_label);
        let mut tracker = ByteTrackBuilder::new().build::<DetectBox>();
        let (mut boxes, mut masks, mut tracks) = (Vec::with_capacity(10), Vec::new(), Vec::new());
        for ts in 0..3 {
            d.decode_tracked(
                &mut tracker,
                ts,
                &inputs,
                &mut boxes,
                &mut masks,
                &mut tracks,
            )
            .expect("decode_tracked");
        }
        (boxes, tracks.len())
    };
    assert_eq!(run(true), run(false));
}

/// Tracked per-scale proto decode ignores multi-label and matches argmax.
#[cfg(feature = "tracker")]
#[test]
fn tracked_per_scale_decode_proto_ignores_multi_label() {
    use edgefirst_tracker::ByteTrackBuilder;
    let box_t = make_zero_box_tensor();
    let score_t = make_multi_label_score_tensor();
    let inputs: Vec<&TensorDyn> = vec![&box_t, &score_t];
    let run = |multi_label: bool| {
        let d = build_2x2_decoder(multi_label);
        let mut tracker = ByteTrackBuilder::new().build::<DetectBox>();
        let (mut boxes, mut tracks) = (Vec::with_capacity(10), Vec::new());
        for ts in 0..3 {
            d.decode_proto_tracked(&mut tracker, ts, &inputs, &mut boxes, &mut tracks)
                .expect("decode_proto_tracked");
        }
        (boxes, tracks.len())
    };
    assert_eq!(run(true), run(false));
}

/// `decode_proto` detections of a captured fixture, class-aware at IoU 0.7 with at most 300
/// detections, in the builder's default dtype or `dtype`.
fn decode_fixture(
    fix: &common::per_scale_fixture::PerScaleFixture,
    dtype: Option<DecodeDtype>,
    score_threshold: f32,
    multi_label: bool,
) -> (Vec<DetectBox>, Option<DecodeDtype>) {
    let schema: SchemaV2 = serde_json::from_str(fix.schema_json()).expect("fixture schema");
    let mut builder = DecoderBuilder::default()
        .with_schema(schema)
        .with_iou_threshold(0.7)
        .with_score_threshold(score_threshold)
        .with_nms(Some(Nms::ClassAware))
        .with_multi_label(multi_label)
        .with_max_det(300);
    if multi_label {
        builder = builder.with_pre_nms_top_k(30_000);
    }
    if let Some(dtype) = dtype {
        builder = builder.with_decode_dtype(dtype);
    }
    let decoder = builder.build().expect("build decoder");
    let tensors = fix.build_tensors_with_quant().expect("fixture tensors");
    let inputs: Vec<&TensorDyn> = tensors.iter().collect();
    let mut boxes = Vec::with_capacity(300);
    decoder
        .decode_proto(&inputs, &mut boxes)
        .expect("decode_proto");
    (boxes, decoder.decode_dtype())
}

/// F16 decode against F32 on a captured frame, at a deployment threshold (argmax) and at the
/// validation threshold (multi-label): the same number of detections within 1%, and at least
/// 99% (deployment: all) and 97% (validation) of the F32 detections have an F16 detection of
/// the same class at IoU >= 0.9 within 1e-3 in score.
fn assert_f16_tracks_f32(model: &str) {
    let path = fixture_path(&format!("{model}.safetensors"));
    let fix = match common::per_scale_fixture::PerScaleFixture::load(&path) {
        Ok(f) => f,
        Err(common::per_scale_fixture::FixtureError::NotPresent(_)) => {
            skip_note!("fixture {path:?} not present (run `git lfs pull`)");
            return;
        }
        Err(e) => panic!("{e}"),
    };
    for (regime, threshold, multi_label, min_matched) in
        [("deploy", 0.25, false, 1.0), ("val", 0.001, true, 0.97)]
    {
        let (f32_boxes, f32_dtype) =
            decode_fixture(&fix, Some(DecodeDtype::F32), threshold, multi_label);
        let (f16_boxes, f16_dtype) =
            decode_fixture(&fix, Some(DecodeDtype::F16), threshold, multi_label);
        assert_eq!(f32_dtype, Some(DecodeDtype::F32));
        assert_eq!(f16_dtype, Some(DecodeDtype::F16));
        assert!(!f32_boxes.is_empty(), "{model} {regime}: no detections");
        let slack = f32_boxes.len() / 100 + usize::from(regime == "val");
        assert!(
            f32_boxes.len().abs_diff(f16_boxes.len()) <= slack,
            "{model} {regime}: {} F32 detections, {} F16",
            f32_boxes.len(),
            f16_boxes.len()
        );
        let mut used = vec![false; f16_boxes.len()];
        let mut matched = 0usize;
        for a in &f32_boxes {
            let best = f16_boxes
                .iter()
                .enumerate()
                .filter(|(j, b)| !used[*j] && b.label == a.label)
                .map(|(j, b)| {
                    let iou = box_iou(
                        [a.bbox.xmin, a.bbox.ymin, a.bbox.xmax, a.bbox.ymax],
                        [b.bbox.xmin, b.bbox.ymin, b.bbox.xmax, b.bbox.ymax],
                    );
                    (j, iou, (a.score - b.score).abs())
                })
                .filter(|&(_, iou, score)| iou >= 0.9 && score <= 1e-3)
                .max_by(|x, y| x.1.total_cmp(&y.1));
            if let Some((j, ..)) = best {
                used[j] = true;
                matched += 1;
            }
        }
        let fraction = matched as f32 / f32_boxes.len() as f32;
        assert!(
            fraction >= min_matched,
            "{model} {regime}: {matched} of {} F32 detections matched in F16",
            f32_boxes.len()
        );
    }
}

#[test]
fn yolov8n_f16_decode_tracks_f32() {
    assert_f16_tracks_f32("yolov8n-seg");
}

#[test]
fn yolo11n_f16_decode_tracks_f32() {
    assert_f16_tracks_f32("yolo11n-seg");
}

#[test]
fn yolo26n_f16_decode_tracks_f32() {
    assert_f16_tracks_f32("yolo26n-seg");
}

/// A builder given no dtype decodes in `DecodeDtype::native`; `with_decode_dtype` overrides it.
#[test]
fn decode_dtype_defaults_to_native_and_can_be_overridden() {
    let path = fixture_path("yolo26n-seg.safetensors");
    let fix = match common::per_scale_fixture::PerScaleFixture::load(&path) {
        Ok(f) => f,
        Err(common::per_scale_fixture::FixtureError::NotPresent(_)) => {
            skip_note!("fixture {path:?} not present (run `git lfs pull`)");
            return;
        }
        Err(e) => panic!("{e}"),
    };
    let native = DecodeDtype::native();
    #[cfg(not(target_arch = "aarch64"))]
    assert_eq!(native, DecodeDtype::F32);
    #[cfg(target_arch = "aarch64")]
    if std::env::var_os("EDGEFIRST_DECODER_FORCE_KERNEL").is_none() {
        let fp16 = std::arch::is_aarch64_feature_detected!("fp16");
        assert_eq!(native == DecodeDtype::F16, fp16);
    }
    let (default_boxes, default_dtype) = decode_fixture(&fix, None, 0.25, false);
    assert_eq!(default_dtype, Some(native));
    for dtype in [DecodeDtype::F32, DecodeDtype::F16] {
        let (boxes, used) = decode_fixture(&fix, Some(dtype), 0.25, false);
        assert_eq!(used, Some(dtype));
        if dtype == native {
            assert_eq!(boxes, default_boxes);
        }
    }
}
