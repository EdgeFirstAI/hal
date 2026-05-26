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
//! Also pins the public `Decoder::normalized_boxes()` contract: when
//! the schema declares pixel-space outputs AND `input_dims` is known,
//! the in-place normalizer runs internally, so the accessor reports
//! `Some(true)` (the post-decode state), not the raw schema flag.
//! Callers must not re-normalize when the accessor reports `Some(true)`
//! — doing so collapses coordinates to ~0.

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

    assert_eq!(
        decoder.normalized_boxes(),
        Some(true),
        "schema declared normalized: false but input_dims are known, so the \
         decoder internally normalizes; the accessor must report the \
         post-decode contract (Some(true)), not the raw schema flag",
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

    // Post-decode contract: schema says pixel-space, input_dims known,
    // so the SplitSegDet path internally normalizes — accessor reports
    // Some(true) to match what the caller actually receives.
    assert_eq!(decoder.normalized_boxes(), Some(true));
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

/// `normalized_boxes()` contract: when the schema declares pixel-space
/// outputs (`normalized: false`) but `input_dims` is unknown, the
/// in-place normalizer cannot run and the decoder leaks pixel-space
/// boxes — the accessor must report `Some(false)` in that case.
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
    assert_eq!(
        decoder.normalized_boxes(),
        Some(false),
        "without input_dims the in-place normalizer cannot run; \
         pixel-space leaks out, so the accessor must report Some(false)",
    );
}
