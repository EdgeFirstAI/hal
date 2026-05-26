// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! Regression test for EDGEAI-1303: when the schema declares
//! `Detection::normalized = false`, the legacy decode path must
//! divide bbox channels by the model input dimensions before
//! `protobox`. Without the fix, vanilla Ultralytics ONNX exports
//! (which emit pixel-space boxes at imgsz=640) failed end-to-end
//! decoding with `InvalidShape("…un-normalized…")` even though
//! the schema correctly described their coordinate space.
//!
//! Pins the public `Decoder::normalized_boxes()` per-path contract.
//! See the rustdoc on `Decoder::normalized_boxes` for the canonical
//! statement. In short:
//!
//! * **Per-scale path** (`per_scale.is_some()`): the accessor upgrades
//!   `normalized=false` + valid `input_dims` to `Some(true)` because
//!   every per-scale entry point unconditionally calls
//!   `maybe_normalize_boxes_in_place`. Pinned by `per_scale_parity.rs`
//!   and `decoder/tests.rs` (hailo fixture).
//!
//! * **Non-per-scale paths**: the accessor returns the raw schema flag
//!   (`self.normalized`) regardless of `input_dims`. Even when the
//!   in-place helper runs internally (e.g. `YoloSegDet`, `YoloSplitSegDet`
//!   on `decode`), the accessor under-claims because not every entry
//!   point on those `ModelType` variants is uniform — callers must handle
//!   pixel-space output explicitly when the flag says `Some(false)`.
//!
//! This file pins:
//! (a) non-per-scale raw-flag passthrough (`pixel_space_input_with_normalized_false_decodes`
//!     and `split_schema_pixel_space_with_normalized_false_decodes`),
//! (b) in-place normalization actually runs despite the accessor
//!     under-claiming (the `for b in &boxes` coordinate-range assertions),
//! (c) no-input-dims passthrough (`normalized_false_without_input_dims_reports_false`),
//! (d) `normalized=None` passthrough (`normalized_none_reports_none`),
//! (e) `normalized=true` direct passthrough (`normalized_true_reports_true`).

use edgefirst_decoder::{schema::SchemaV2, DecoderBuilder, DetectBox};
use edgefirst_tensor::{Tensor, TensorDyn, TensorMapTrait, TensorMemory, TensorTrait};

const NC: usize = 2;
const NM: usize = 32;
const N: usize = 256;
const FEAT: usize = 4 + NC + NM;
const PH: usize = 160;
const PW: usize = 160;
const IMG: usize = 640;

/// JSON v2 schema for a YOLOv8-seg-style decoder declaring its
/// boxes as pixel-space (`normalized: false`) at imgsz=640. The
/// decoder is expected to normalize by the input W/H pulled from
/// the input spec.
fn schema_json(normalized: bool) -> String {
    format!(
        r#"{{
            "schema_version": 2,
            "nms": "class_agnostic",
            "input": {{
                "shape": [1, {IMG}, {IMG}, 3],
                "dshape": [
                    {{"batch": 1}},
                    {{"height": {IMG}}},
                    {{"width": {IMG}}},
                    {{"num_features": 3}}
                ],
                "cameraadaptor": "rgb"
            }},
            "outputs": [
                {{
                    "name": "output0", "type": "detection",
                    "shape": [1, {FEAT}, {N}],
                    "dshape": [
                        {{"batch": 1}},
                        {{"num_features": {FEAT}}},
                        {{"num_boxes": {N}}}
                    ],
                    "decoder": "ultralytics",
                    "encoding": "direct",
                    "normalized": {normalized}
                }},
                {{
                    "name": "protos", "type": "protos",
                    "shape": [1, {NM}, {PH}, {PW}],
                    "dshape": [
                        {{"batch": 1}},
                        {{"num_protos": {NM}}},
                        {{"height": {PH}}},
                        {{"width": {PW}}}
                    ]
                }}
            ]
        }}"#,
    )
}

/// Build the synthetic detection + protos tensors. Plant 5 detections
/// at non-overlapping x positions in **pixel-space** at imgsz=640.
fn build_inputs() -> Vec<TensorDyn> {
    let n_targets = 5usize;
    let target_start = 10usize;
    let mut det_data = vec![0.0f32; FEAT * N];
    let set = |d: &mut [f32], r: usize, c: usize, v: f32| d[r * N + c] = v;
    for t in 0..n_targets {
        let anchor = target_start + t;
        // Pixel-space xc spread across the image at imgsz=640.
        let xc = 80.0 + 80.0 * t as f32; // 80, 160, 240, 320, 400
        set(&mut det_data, 0, anchor, xc);
        set(&mut det_data, 1, anchor, IMG as f32 * 0.5); // yc=320
        set(&mut det_data, 2, anchor, 25.6); // w ≈ 0.04*640
        set(&mut det_data, 3, anchor, 192.0); // h ≈ 0.3*640
        set(&mut det_data, 4, anchor, 0.9); // class-0 score
    }
    let det_tensor: TensorDyn = {
        let t = Tensor::<f32>::new(&[1, FEAT, N], Some(TensorMemory::Mem), None).unwrap();
        {
            let mut m = t.map().unwrap();
            m.as_mut_slice().copy_from_slice(&det_data);
        }
        TensorDyn::F32(t)
    };

    let proto_data = vec![0.0f32; NM * PH * PW];
    let protos_tensor: TensorDyn = {
        let t = Tensor::<f32>::new(&[1, NM, PH, PW], Some(TensorMemory::Mem), None).unwrap();
        {
            let mut m = t.map().unwrap();
            m.as_mut_slice().copy_from_slice(&proto_data);
        }
        TensorDyn::F32(t)
    };

    vec![det_tensor, protos_tensor]
}

#[test]
fn pixel_space_input_with_normalized_false_decodes() {
    let schema: SchemaV2 = serde_json::from_str(&schema_json(false)).unwrap();
    let decoder = DecoderBuilder::default()
        .with_score_threshold(0.5)
        .with_iou_threshold(0.99)
        .with_schema(schema)
        .build()
        .expect("schema-driven decoder must build");

    // Non-per-scale path: the accessor surfaces the raw schema flag.
    // The in-place normalizer still runs internally (confirmed below by
    // coordinate-range assertions), but `normalized_boxes()` under-claims
    // because not every YoloSegDet entry point is uniform across decode
    // variants — see `Decoder::normalized_boxes` rustdoc.
    assert_eq!(
        decoder.normalized_boxes(),
        Some(false),
        "non-per-scale YoloSegDet: accessor returns raw schema flag; \
         in-place normalization runs internally but is not reflected here",
    );
    assert_eq!(
        decoder.input_dims(),
        Some((IMG, IMG)),
        "decoder must capture the schema's model input dimensions",
    );

    let owned = build_inputs();
    let inputs: Vec<&TensorDyn> = owned.iter().collect();

    let mut boxes: Vec<DetectBox> = Vec::with_capacity(50);
    let mut masks: Vec<edgefirst_decoder::Segmentation> = Vec::with_capacity(50);
    decoder
        .decode(&inputs, &mut boxes, &mut masks)
        .expect("pixel-space decode must succeed when normalized=false (EDGEAI-1303)");

    assert!(
        !boxes.is_empty(),
        "expected detections to survive NMS; got 0",
    );
    for b in &boxes {
        assert!(
            (0.0..=1.0).contains(&b.bbox.xmin)
                && (0.0..=1.0).contains(&b.bbox.ymin)
                && (0.0..=1.0).contains(&b.bbox.xmax)
                && (0.0..=1.0).contains(&b.bbox.ymax),
            "decoded bbox {:?} not in normalized [0, 1]; \
             EDGEAI-1303 normalization did not run",
            b.bbox,
        );
    }
}

#[test]
fn pixel_space_input_with_normalized_true_still_rejects() {
    // Same pixel-space input, but the schema lies and says it's
    // already normalized. The decoder skips the normalization path
    // and `protobox`'s safety net rejects the un-normalized box.
    let schema: SchemaV2 = serde_json::from_str(&schema_json(true)).unwrap();
    let decoder = DecoderBuilder::default()
        .with_score_threshold(0.5)
        .with_iou_threshold(0.99)
        .with_schema(schema)
        .build()
        .expect("schema-driven decoder must build");

    let owned = build_inputs();
    let inputs: Vec<&TensorDyn> = owned.iter().collect();

    let mut boxes: Vec<DetectBox> = Vec::with_capacity(50);
    let mut masks: Vec<edgefirst_decoder::Segmentation> = Vec::with_capacity(50);
    let err = decoder
        .decode(&inputs, &mut boxes, &mut masks)
        .expect_err("incorrectly-declared normalized=true must trip protobox guard");

    let msg = err.to_string();
    assert!(
        msg.contains("un-normalized"),
        "expected protobox un-normalized rejection; got: {msg}",
    );
}

/// JSON v2 schema for a SplitSegDet (Boxes + Scores + MaskCoefficients
/// + Protos) decoder declaring its boxes as pixel-space at imgsz=640.
///
/// This is the schema flavour vanilla Ultralytics ONNX `--export-split`
/// produces and exercises the second (postprocess.rs:511) wiring point
/// for EDGEAI-1303 — distinct from the combined-Detection path above.
fn split_schema_json(normalized: bool) -> String {
    format!(
        r#"{{
            "schema_version": 2,
            "nms": "class_agnostic",
            "input": {{
                "shape": [1, {IMG}, {IMG}, 3],
                "dshape": [
                    {{"batch": 1}},
                    {{"height": {IMG}}},
                    {{"width": {IMG}}},
                    {{"num_features": 3}}
                ],
                "cameraadaptor": "rgb"
            }},
            "outputs": [
                {{
                    "name": "boxes", "type": "boxes",
                    "shape": [1, 4, {N}],
                    "dshape": [
                        {{"batch": 1}},
                        {{"box_coords": 4}},
                        {{"num_boxes": {N}}}
                    ],
                    "decoder": "ultralytics",
                    "encoding": "direct",
                    "normalized": {normalized}
                }},
                {{
                    "name": "scores", "type": "scores",
                    "shape": [1, {NC}, {N}],
                    "dshape": [
                        {{"batch": 1}},
                        {{"num_classes": {NC}}},
                        {{"num_boxes": {N}}}
                    ],
                    "decoder": "ultralytics",
                    "score_format": "per_class"
                }},
                {{
                    "name": "mask_coeff", "type": "mask_coefs",
                    "shape": [1, {NM}, {N}],
                    "dshape": [
                        {{"batch": 1}},
                        {{"num_protos": {NM}}},
                        {{"num_boxes": {N}}}
                    ]
                }},
                {{
                    "name": "protos", "type": "protos",
                    "shape": [1, {NM}, {PH}, {PW}],
                    "dshape": [
                        {{"batch": 1}},
                        {{"num_protos": {NM}}},
                        {{"height": {PH}}},
                        {{"width": {PW}}}
                    ]
                }}
            ]
        }}"#,
    )
}

/// Build the synthetic split tensors (boxes [1,4,N], scores [1,NC,N],
/// mask_coeff [1,NM,N], protos [1,NM,PH,PW]) with pixel-space boxes
/// planted at the same anchors as `build_inputs`.
fn build_split_inputs() -> Vec<TensorDyn> {
    let n_targets = 5usize;
    let target_start = 10usize;
    let mut box_data = vec![0.0f32; 4 * N];
    let mut score_data = vec![0.0f32; NC * N];
    let mask_data = vec![0.0f32; NM * N];
    let set = |d: &mut [f32], r: usize, c: usize, stride: usize, v: f32| d[r * stride + c] = v;
    for t in 0..n_targets {
        let anchor = target_start + t;
        let xc = 80.0 + 80.0 * t as f32;
        set(&mut box_data, 0, anchor, N, xc);
        set(&mut box_data, 1, anchor, N, IMG as f32 * 0.5);
        set(&mut box_data, 2, anchor, N, 25.6);
        set(&mut box_data, 3, anchor, N, 192.0);
        set(&mut score_data, 0, anchor, N, 0.9);
    }
    let to_dyn = |data: Vec<f32>, shape: &[usize]| -> TensorDyn {
        let t = Tensor::<f32>::new(shape, Some(TensorMemory::Mem), None).unwrap();
        {
            let mut m = t.map().unwrap();
            m.as_mut_slice().copy_from_slice(&data);
        }
        TensorDyn::F32(t)
    };
    vec![
        to_dyn(box_data, &[1, 4, N]),
        to_dyn(score_data, &[1, NC, N]),
        to_dyn(mask_data, &[1, NM, N]),
        to_dyn(vec![0.0f32; NM * PH * PW], &[1, NM, PH, PW]),
    ]
}

#[test]
fn split_schema_pixel_space_with_normalized_false_decodes() {
    let schema: SchemaV2 = serde_json::from_str(&split_schema_json(false)).unwrap();
    let decoder = DecoderBuilder::default()
        .with_score_threshold(0.5)
        .with_iou_threshold(0.99)
        .with_schema(schema)
        .build()
        .expect("split schema must build");

    // Non-per-scale path: accessor returns the raw schema flag.
    // YoloSplitSegDet calls maybe_normalize_boxes_in_place on decode()
    // but not uniformly across all entry points, so normalized_boxes()
    // under-claims — returns Some(false) (the raw schema flag), not
    // Some(true). Coordinate-range assertions below confirm the helper
    // still ran internally.
    assert_eq!(decoder.normalized_boxes(), Some(false));
    assert_eq!(decoder.input_dims(), Some((IMG, IMG)));

    let owned = build_split_inputs();
    let inputs: Vec<&TensorDyn> = owned.iter().collect();

    let mut boxes: Vec<DetectBox> = Vec::with_capacity(50);
    let mut masks: Vec<edgefirst_decoder::Segmentation> = Vec::with_capacity(50);
    decoder
        .decode(&inputs, &mut boxes, &mut masks)
        .expect("split-schema pixel-space decode must succeed (EDGEAI-1303)");

    assert!(!boxes.is_empty(), "expected detections to survive NMS");
    for b in &boxes {
        assert!(
            (0.0..=1.0).contains(&b.bbox.xmin)
                && (0.0..=1.0).contains(&b.bbox.ymin)
                && (0.0..=1.0).contains(&b.bbox.xmax)
                && (0.0..=1.0).contains(&b.bbox.ymax),
            "split-schema bbox {:?} not in [0, 1]; EDGEAI-1303 normalization \
             did not run on the SplitSegDet path",
            b.bbox,
        );
    }
}

/// Non-per-scale path, `normalized=false`, no `input_dims`: accessor
/// returns the raw schema flag `Some(false)`.
///
/// On non-per-scale paths the accessor always surfaces the raw
/// `self.normalized` field regardless of `input_dims`. The absence of
/// `input_dims` also means the in-place normalizer cannot fire, so this
/// is doubly correct: the flag reflects both the schema declaration and
/// the runtime behaviour.
///
/// Constructed programmatically via `add_output` to deliberately leave
/// `input_dims` unset (the v2 schema path would always populate them).
#[test]
fn normalized_false_without_input_dims_reports_false() {
    use edgefirst_decoder::{configs, ConfigOutput};

    let det = configs::Detection {
        anchors: None,
        decoder: configs::DecoderType::Ultralytics,
        quantization: None,
        shape: vec![1, FEAT, N],
        dshape: Vec::new(),
        normalized: Some(false),
    };
    let decoder = DecoderBuilder::default()
        .add_output(ConfigOutput::Detection(det))
        .build()
        .expect("programmatic decoder must build");

    assert_eq!(
        decoder.input_dims(),
        None,
        "programmatic build must leave input_dims unset",
    );
    // Non-per-scale: raw schema flag is surfaced unconditionally.
    assert_eq!(
        decoder.normalized_boxes(),
        Some(false),
        "non-per-scale: accessor returns raw schema flag; \
         input_dims absent so the helper cannot run either",
    );
}

/// Non-per-scale path, `normalized=None`: accessor returns `None`.
///
/// When the schema omits the `normalized` field the accessor propagates
/// the absence directly for non-per-scale decoders.
#[test]
fn normalized_none_reports_none() {
    use edgefirst_decoder::{configs, ConfigOutput};

    let det = configs::Detection {
        anchors: None,
        decoder: configs::DecoderType::Ultralytics,
        quantization: None,
        shape: vec![1, FEAT, N],
        dshape: Vec::new(),
        normalized: None,
    };
    let decoder = DecoderBuilder::default()
        .add_output(ConfigOutput::Detection(det))
        .build()
        .expect("programmatic decoder must build");

    assert_eq!(
        decoder.normalized_boxes(),
        None,
        "non-per-scale: absent normalized field must propagate as None",
    );
}

/// Non-per-scale path, `normalized=true`: accessor returns `Some(true)`.
///
/// When the schema explicitly declares boxes are already normalized the
/// accessor surfaces that flag directly on non-per-scale paths.
#[test]
fn normalized_true_reports_true() {
    use edgefirst_decoder::{configs, ConfigOutput};

    let det = configs::Detection {
        anchors: None,
        decoder: configs::DecoderType::Ultralytics,
        quantization: None,
        shape: vec![1, FEAT, N],
        dshape: Vec::new(),
        normalized: Some(true),
    };
    let decoder = DecoderBuilder::default()
        .add_output(ConfigOutput::Detection(det))
        .build()
        .expect("programmatic decoder must build");

    assert_eq!(
        decoder.normalized_boxes(),
        Some(true),
        "non-per-scale: normalized=true schema flag must propagate as Some(true)",
    );
}

/// Non-per-scale path, `normalized=false` with valid `input_dims`:
/// accessor returns `Some(false)` (raw schema flag), even though the
/// in-place normalizer runs during `decode()`.
///
/// This is the key distinction from the per-scale path: non-per-scale
/// decoders surface the raw `self.normalized` unconditionally because
/// not every `ModelType` entry point calls `maybe_normalize_boxes_in_place`
/// uniformly. Callers that need the post-decode coordinate state must
/// inspect the actual box values rather than relying solely on this flag.
///
/// Verified here: the accessor reports `Some(false)` AND the decoded
/// boxes are in `[0, 1]` (the helper did fire internally).
#[test]
fn non_per_scale_normalized_false_with_input_dims_reports_false_but_normalizes() {
    let schema: SchemaV2 = serde_json::from_str(&schema_json(false)).unwrap();
    let decoder = DecoderBuilder::default()
        .with_score_threshold(0.5)
        .with_iou_threshold(0.99)
        .with_schema(schema)
        .build()
        .expect("schema-driven decoder must build");

    assert_eq!(
        decoder.input_dims(),
        Some((IMG, IMG)),
        "v2 schema must populate input_dims from the input spec",
    );
    // Non-per-scale: accessor returns the raw schema flag, not the
    // post-decode state, even though input_dims is present.
    assert_eq!(
        decoder.normalized_boxes(),
        Some(false),
        "non-per-scale: accessor returns raw schema flag regardless of input_dims",
    );

    // But the in-place helper still fires during decode().
    let owned = build_inputs();
    let inputs: Vec<&TensorDyn> = owned.iter().collect();
    let mut boxes: Vec<DetectBox> = Vec::with_capacity(50);
    let mut masks: Vec<edgefirst_decoder::Segmentation> = Vec::with_capacity(50);
    decoder
        .decode(&inputs, &mut boxes, &mut masks)
        .expect("pixel-space decode must succeed (EDGEAI-1303)");

    assert!(!boxes.is_empty(), "expected detections to survive NMS");
    for b in &boxes {
        assert!(
            (0.0..=1.0).contains(&b.bbox.xmin)
                && (0.0..=1.0).contains(&b.bbox.ymin)
                && (0.0..=1.0).contains(&b.bbox.xmax)
                && (0.0..=1.0).contains(&b.bbox.ymax),
            "decoded bbox {:?} must be in [0, 1] despite accessor reporting Some(false)",
            b.bbox,
        );
    }
}
