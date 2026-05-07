// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! Parity tests: HAL per-scale decoder vs. the Python reference at
//! `scripts/per_scale_decode_reference.py`.
//!
//! Strategy: for each fixture, run both decoders on identical inputs
//! and compare outputs at three levels:
//!   1. Per-anchor box / score / mc/proto agreement.
//!   2. Post-NMS DetectBox list agreement under IoU/class tolerances.
//!
//! Tests gracefully skip when `tflite_runtime` is not available or the
//! fixture model files are missing.

mod common;

use std::path::{Path, PathBuf};
use std::process::Command;

const REFERENCE_SCRIPT: &str = "scripts/per_scale_decode_reference.py";
const FIXTURES_DIR: &str = "testdata/per_scale";

/// Workspace root, resolved from this crate's manifest dir.
/// `cargo test` sets CWD = `<workspace>/crates/decoder`, so we walk up two
/// levels to reach the repo root where `scripts/` and `testdata/` live.
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
#[allow(dead_code)]
fn fixtures_dir_path() -> PathBuf {
    workspace_root().join(FIXTURES_DIR)
}

/// Returns true when the Python reference is runnable in this env.
#[allow(dead_code)] // Used by Task 28 fixture tests.
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
    // Run from the workspace root so the script's relative paths resolve.
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

/// Sentinel test that documents the harness's purpose. Always passes.
#[test]
fn per_scale_parity_harness_compiles() {
    // The harness's real value is the helpers above, exercised by tests
    // added in Task 28 (synthetic fixtures) and Task 29 (real models).
    // This test exists so the parity test binary always has at least one
    // test to compile/link, even when `tflite_runtime` is unavailable.
    // Touch a helper so it isn't dead-code-flagged on this build.
    assert!(close_within_ulp(1.0_f32, 1.0_f32, 1));
}

/// Verifies the Python script exists at the expected path.
#[test]
fn python_reference_script_exists_in_repo() {
    let p = reference_script_path();
    assert!(p.exists(), "expected {p:?} to exist; check repo layout");
}

/// Verifies the fixtures dir exists.
#[test]
fn fixtures_dir_exists_in_repo() {
    let p = fixtures_dir_path();
    assert!(p.exists(), "expected {p:?} to exist; created in Task 13");
}

// ────────────────────────────────────────────────────────────────────
// Synthetic-fixture HAL parity tests
// ────────────────────────────────────────────────────────────────────
//
// These tests exercise the full per-scale decoder pipeline (build →
// run → NMS → DetectBox) against hand-computed golden values for
// synthetic int8 inputs. They are independent of the Python reference;
// the Python parity oracle covers real models in Task 29.

use edgefirst_decoder::per_scale::DecodeDtype;
use edgefirst_decoder::{schema::SchemaV2, DecoderBuilder, DetectBox, Nms};
use edgefirst_tensor::{
    Quantization as TQ, Tensor, TensorDyn, TensorMapTrait, TensorMemory, TensorTrait,
};

/// Construct a minimal LTRB-detection-only per-scale schema with one
/// FPN level (stride=8, h=w=1, NC=2). Used as a deterministic golden
/// test where we know exactly what the decoder should output for any
/// given input.
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
        .with_score_threshold(0.0001) // low so 0.5 sigmoid passes
        .with_nms(Some(Nms::ClassAgnostic))
        .build()
        .expect("build");

    // Two zero-int8 input tensors with attached quantization.
    let mut box_t = Tensor::<i8>::new(&[1, 1, 1, 4], Some(TensorMemory::Mem), None).unwrap();
    box_t.set_quantization(TQ::per_tensor(0.1, 0)).unwrap();
    let mut score_t = Tensor::<i8>::new(&[1, 1, 1, 2], Some(TensorMemory::Mem), None).unwrap();
    score_t.set_quantization(TQ::per_tensor(0.1, 0)).unwrap();
    // Set score logit slightly positive so post-sigmoid > 0.5 and survives NMS.
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

    // Expect exactly one detection (only one anchor, with score above threshold).
    assert_eq!(
        output_boxes.len(),
        1,
        "expected 1 detection, got {}",
        output_boxes.len()
    );
    let det = &output_boxes[0];

    // Box at centre (4, 4), zero-size after dist2bbox.
    // The bridge feeds XYWH (xc, yc, w, h) into impl_yolo_segdet_get_boxes which
    // converts to XYXY: xmin=xc-w/2, ymin=yc-h/2, xmax=xc+w/2, ymax=yc+h/2.
    // For zero w/h pixel-space: xmin = xmax = 4, ymin = ymax = 4.
    // EDGEAI-1303: per_scale path normalizes by input dims (8x8 here),
    // so the user-facing bbox is 4/8 = 0.5 across all corners.
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
    // EDGEAI-1303: per_scale path normalizes by input dims (8x8 here).
    // Pixel-space (-4, -4, 28, 28) → normalized (-0.5, -0.5, 3.5, 3.5).
    // YOLO can predict boxes outside the canvas for objects near the
    // border; the normalization preserves that out-of-range information.
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
// HAL Phase 1 added per-level NCHW support via a transpose-to-scratch
// path (NHWC kernels stay unmodified; the pipeline transposes NCHW
// children into a per-dtype scratch before dispatching). These tests
// build paired NHWC and NCHW schemas with the same logical content,
// feed bit-equivalent tensor data in both layouts, and assert the
// decoded `DetectBox` lists match exactly. A regression here points at
// either the transpose helper, the layout detection in `plan.rs`, or
// the scratch-buffer plumbing in `pipeline.rs`.

/// LTRB schema with one stride-8 level, h=w=2, NC=2, NCHW children.
///
/// Children dshape `[batch, box_coords, height, width]` (NCHW) instead of
/// the canonical `[batch, height, width, box_coords]` (NHWC). Same
/// quantization, same logical content; only the byte layout differs.
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

/// NHWC pair of the schema above. Identical logical content; only the
/// children's `shape` and `dshape` ordering differ.
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

/// Per-anchor logical box int8s (NHWC anchor-major, channel-contiguous).
/// 4 anchors × 4 LTRB channels = 16 elements. Each anchor has a distinct
/// box so we can confirm the binding stays intact under transpose.
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

    // Same number of detections, same labels, same scores, same boxes.
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
    // Sanity: anchor 0 had the only winning class-1 score and produces a box.
    assert!(!nhwc_dets.is_empty(), "expected at least one detection");
}

// ────────────────────────────────────────────────────────────────────
// Real-model Python smoke (skip-capable)
// ────────────────────────────────────────────────────────────────────
//
// These tests load real TFLite fixtures when present at
// `testdata/per_scale/<model>.tflite`. They run the Python reference
// on the model as an environment smoke check only.
//
// Fixtures are NOT committed (large binaries; see README in
// testdata/per_scale/). The validator team supplies them; tests
// skip cleanly when absent.
//
// NOTE for Phase 1: the HAL doesn't load TFLite models directly. This
// test invokes the Python reference (which DOES load TFLite) for both
// the inference and the decode steps; the HAL comparison piece is a
// TODO marker that requires the inference layer to extract raw int8
// outputs and hand them to the HAL via `apply_schema_quant`.

#[test]
#[ignore = "external real-model fixture; this is Python-only smoke and does not exercise HAL decode"]
fn parity_yolov8n_seg_per_scale_int8() {
    let model = workspace_root()
        .join(FIXTURES_DIR)
        .join("yolov8n_seg_per_scale_int8.tflite");
    if !model.exists() {
        eprintln!(
            "skipping: real-model fixture not present at {model:?} — \
                   contact validator team for the binary"
        );
        return;
    }
    if !python_reference_available() {
        eprintln!("skipping: tflite_runtime not importable in Python env");
        return;
    }
    let py_result = run_python_reference(&model, None);
    let py_result = match py_result {
        Some(v) => v,
        None => {
            eprintln!("skipping: Python reference failed to run");
            return;
        }
    };
    let n = py_result
        .get("num_detections")
        .and_then(|v| v.as_u64())
        .unwrap_or(0);
    eprintln!("python reference detected {n} boxes on synthetic input");
    // TODO HAL-Phase2: after the inference layer is integrated, run
    // HAL on the same synthetic input and compare detection counts +
    // per-detection scores within ~2 ulp f32. For Phase 1 we just
    // verify the Python side runs cleanly.
}

#[test]
#[ignore = "external real-model fixture; this is Python-only smoke and does not exercise HAL decode"]
fn parity_yolo26n_seg_per_scale_int8() {
    let model = workspace_root()
        .join(FIXTURES_DIR)
        .join("yolo26n_seg_per_scale_int8.tflite");
    if !model.exists() {
        eprintln!("skipping: yolo26n fixture not present");
        return;
    }
    if !python_reference_available() {
        eprintln!("skipping: tflite_runtime not importable");
        return;
    }
    let py_result = run_python_reference(&model, None);
    let py_result = match py_result {
        Some(v) => v,
        None => return,
    };
    let n = py_result
        .get("num_detections")
        .and_then(|v| v.as_u64())
        .unwrap_or(0);
    eprintln!("python reference detected {n} boxes on synthetic input");
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
            eprintln!(
                "skip: python failed: {}",
                String::from_utf8_lossy(&o.stderr)
            );
            return;
        }
        Err(e) => {
            eprintln!("skip: python not available: {e}");
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
            eprintln!("skip: python: {}", String::from_utf8_lossy(&o.stderr));
            return;
        }
        Err(e) => {
            eprintln!("skip: python: {e}");
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
            eprintln!("skip: python");
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
            eprintln!("skip");
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
            eprintln!("skip");
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
    let nonexistent =
        workspace_root().join("testdata/decoder/this-fixture-does-not-exist.safetensors");
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
    // Order fixture detections by their best IoU descending so confident
    // pairs claim their HAL counterpart first.
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

/// End-to-end parity assertion: HAL `decode_proto` output should agree
/// with the fixture's reference detections within tolerances that
/// account for ULP-level NMS-survivor differences between the Rust and
/// NumPy implementations.
///
/// Tolerances:
/// - Detection count within ±25% of the fixture's count (HAL may keep
///   slightly more or fewer survivors at the post-NMS cap).
/// - At least 80% of fixture detections must have a HAL match at IoU ≥
///   0.5 (greedy pairing).
/// - Mean IoU on matched pairs ≥ 0.85 (the per-scale kernels are
///   numerically equivalent; only NMS-survivor selection drifts).
/// - Class agreement on matched pairs ≥ 90%.
///
/// These tolerances are looser than the synthetic-fixture parity tests
/// because Rust's f32 NMS and NumPy's f32 NMS occasionally pick
/// different survivors at the score-tie boundary; the underlying
/// per-scale kernel output is verified strictly by
/// `*_pre_nms_parity` tests.
fn assert_end_to_end_parity(fix: &common::per_scale_fixture::PerScaleFixture, model_label: &str) {
    let hal_boxes = decode_with_nms(fix, model_label, Nms::ClassAgnostic);

    let dec = fix.decoded().unwrap();
    let n_fix = dec.boxes_xyxy.shape()[0];
    let n_hal = hal_boxes.len();

    // Detection-count tolerance: ±25% of the fixture count, with a
    // floor of 8 to handle small fixtures.
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
    // The fixtures were generated by the Python reference, which does
    // not apply a pre-NMS top-K cap — it runs NMS over the entire
    // candidate pool above `score_threshold`. PR #59/#60 added
    // `pre_nms_top_k` (default 300) to HAL's NMS path; replicating the
    // fixture's regime requires lifting that cap. Use a very large
    // value (effectively unbounded) and set `max_det` to the fixture's
    // declared max_detections so the post-NMS survivor count matches.
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
    // EDGEAI-1303: HAL now emits normalized [0, 1] bboxes from the
    // per-scale path (per `per_scale_to_proto_data`'s
    // `maybe_normalize_boxes_in_place` call). The existing
    // safetensors fixtures store *pixel-space* reference coords from
    // the pre-fix Python pipeline, so convert HAL output back to
    // pixel space here for the IoU comparison. The HAL output is
    // semantically correct; only the parity test needs unit
    // alignment.
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
    workspace_root().join("testdata/decoder").join(basename)
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
            eprintln!("skip: fixture {path:?} not present (run `git lfs pull`)");
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

/// Per-scale `decode()` (the fused mask-materialisation entrypoint)
/// must succeed end-to-end for a real-model fixture and emit one
/// `Segmentation` per `DetectBox`. Regression test for the missing
/// `maybe_normalize_boxes_in_place` step in `per_scale_to_masks`
/// flagged by Copilot's review of PR #63 — the per-scale path emits
/// pixel-space coords by design, and `protobox` would reject them
/// with `InvalidShape("…un-normalized…")` without normalization.
#[test]
fn yolov8n_seg_per_scale_decode_with_masks_succeeds() {
    use edgefirst_decoder::per_scale::DecodeDtype;
    use edgefirst_decoder::{schema::SchemaV2, DecoderBuilder, DetectBox, Nms, Segmentation};

    let path = fixture_path("yolov8n-seg.safetensors");
    let fix = match common::per_scale_fixture::PerScaleFixture::load(&path) {
        Ok(f) => f,
        Err(common::per_scale_fixture::FixtureError::NotPresent(_)) => {
            eprintln!("skip: fixture {path:?} not present (run `git lfs pull`)");
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

    // Sanity: schema-derived input_dims must be present for normalization
    // to take effect (per-scale builder forces normalized = Some(false)).
    assert_eq!(decoder.normalized_boxes(), Some(false));
    assert!(
        decoder.input_dims().is_some(),
        "yolov8n-seg fixture schema must declare input dims so the \
         per-scale bridge can normalize pixel-space boxes (EDGEAI-1303)"
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
        "yolov8n-seg per-scale decode() produced {n_boxes} detections, expected ≥ {}",
        fix.expected_count_min
    );

    // Boxes must be in roughly-normalized range — proves the bridge's
    // EDGEAI-1303 normalization step ran. YOLO can predict boxes that
    // extend slightly past the image edge for objects near the border,
    // so allow a small tolerance below 0 / above 1; the original
    // pixel-space coords would be in [0, ~640] which is far outside
    // this tolerance.
    let in_norm_range = |v: f32| (-0.05..=1.05).contains(&v);
    for b in &output_boxes {
        assert!(
            in_norm_range(b.bbox.xmin)
                && in_norm_range(b.bbox.ymin)
                && in_norm_range(b.bbox.xmax)
                && in_norm_range(b.bbox.ymax),
            "per-scale decode() bbox {:?} not in normalized range — \
             per_scale_to_masks did not normalize pixel-space coords",
            b.bbox
        );
    }
}

#[test]
fn yolov8n_seg_per_scale_end_to_end_parity() {
    let path = fixture_path("yolov8n-seg.safetensors");
    let fix = match common::per_scale_fixture::PerScaleFixture::load(&path) {
        Ok(f) => f,
        Err(common::per_scale_fixture::FixtureError::NotPresent(_)) => {
            eprintln!("skip: fixture {path:?} not present (run `git lfs pull`)");
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
            eprintln!("skip: fixture {path:?} not present (run `git lfs pull`)");
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
            eprintln!("skip: fixture not present");
            return;
        }
        Err(e) => panic!("{e}"),
    };
    assert_pre_nms_parity(&fix, "yolov8n-seg");
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
            eprintln!("skip: fixture {path:?} not present (run `git lfs pull`)");
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
            eprintln!("skip: fixture {path:?} not present (run `git lfs pull`)");
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
            eprintln!("skip: fixture {path:?} not present (run `git lfs pull`)");
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
            eprintln!("skip: fixture not present");
            return;
        }
        Err(e) => panic!("{e}"),
    };
    assert_pre_nms_parity(&fix, "yolo11n-seg");
}

// ────────────────────────────────────────────────────────────────────
// yolo26n-seg per-scale parity (LTRB-encoded, fixture-backed)
// ────────────────────────────────────────────────────────────────────
//
// yolo26 uses LTRB box encoding (no DFL stage). The fixture's
// intermediates omit the `boxes_<lvl>.ltrb` key — only `dequant` and
// `xywh` exist for LTRB. The tests below are encoding-agnostic.

#[test]
fn yolo26n_seg_per_scale_smoke_detection_count() {
    use edgefirst_decoder::per_scale::DecodeDtype;
    use edgefirst_decoder::{schema::SchemaV2, DecoderBuilder, DetectBox, Nms};

    let path = fixture_path("yolo26n-seg.safetensors");
    let fix = match common::per_scale_fixture::PerScaleFixture::load(&path) {
        Ok(f) => f,
        Err(common::per_scale_fixture::FixtureError::NotPresent(_)) => {
            eprintln!("skip: fixture {path:?} not present (run `git lfs pull`)");
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
            eprintln!("skip: fixture {path:?} not present (run `git lfs pull`)");
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
            eprintln!("skip: fixture {path:?} not present (run `git lfs pull`)");
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
            eprintln!("skip: fixture not present");
            return;
        }
        Err(e) => panic!("{e}"),
    };
    assert_pre_nms_parity(&fix, "yolo26n-seg");
}
