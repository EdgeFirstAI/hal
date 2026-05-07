// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! Regression test for EDGEAI-1303: when the schema declares
//! `Detection::normalized = false`, the legacy decode path must
//! divide bbox channels by the model input dimensions before
//! `protobox`. Without the fix, vanilla Ultralytics ONNX exports
//! (which emit pixel-space boxes at imgsz=640) failed end-to-end
//! decoding with `InvalidShape("…un-normalized…")` even though
//! the schema correctly described their coordinate space.

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
        Some(false),
        "schema declared normalized: false; decoder should report Some(false)",
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
