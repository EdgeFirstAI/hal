// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! A model tagged `decoder_version: yolo26` is end-to-end only when its head
//! is. An export made without the embedded NMS keeps the YOLOv8 head
//! (`[1, 4 + classes + protos, anchors]`), which the decoder has to run
//! through its own NMS instead of rejecting.

use edgefirst_decoder::{DecoderBuilder, DetectBox, Segmentation};
use edgefirst_tensor::{Tensor, TensorDyn, TensorMapTrait, TensorMemory, TensorTrait};

const NC: usize = 3;
const NM: usize = 32;
const N: usize = 256;
const FEAT: usize = 4 + NC + NM;
const PH: usize = 160;
const PW: usize = 160;
const TARGETS: usize = 5;

/// A yolo26 segmentation schema. `model` is the schema's `model` block, or
/// empty to leave it out; `head` is the detection output's shape and dshape.
fn schema_json(model: &str, head: &str) -> String {
    format!(
        r#"{{
            "schema_version": 2,
            "decoder_version": "yolo26",
            "nms": "class_agnostic",
            {model}
            "outputs": [
                {{
                    "name": "output0", "type": "detection",
                    {head},
                    "decoder": "ultralytics",
                    "encoding": "direct",
                    "normalized": true
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

/// The pre-NMS head: `[1, 4 + classes + protos, anchors]`.
fn pre_nms_head() -> String {
    format!(
        r#""shape": [1, {FEAT}, {N}],
           "dshape": [{{"batch": 1}}, {{"num_features": {FEAT}}}, {{"num_boxes": {N}}}]"#
    )
}

fn tensor(shape: &[usize], data: &[f32]) -> TensorDyn {
    let t = Tensor::<f32>::new(shape, Some(TensorMemory::Mem), None).unwrap();
    t.map().unwrap().as_mut_slice().copy_from_slice(data);
    TensorDyn::F32(t)
}

/// Pre-NMS outputs with `TARGETS` separate class-0 boxes.
fn pre_nms_outputs() -> Vec<TensorDyn> {
    let mut det = vec![0.0f32; FEAT * N];
    for t in 0..TARGETS {
        let anchor = 10 + t;
        det[anchor] = 0.1 + 0.15 * t as f32; // xc
        det[N + anchor] = 0.5; // yc
        det[2 * N + anchor] = 0.05; // w
        det[3 * N + anchor] = 0.3; // h
        det[4 * N + anchor] = 0.9; // class 0
    }
    vec![
        tensor(&[1, FEAT, N], &det),
        tensor(&[1, NM, PH, PW], &vec![0.0f32; NM * PH * PW]),
    ]
}

/// The boxes `pre_nms_outputs` plants, decoded from centre and size: a head
/// misread as end-to-end rows would return the raw values as corners.
fn assert_pre_nms_boxes(mut boxes: Vec<DetectBox>) {
    assert_eq!(boxes.len(), TARGETS);
    boxes.sort_by(|a, b| a.bbox.xmin.total_cmp(&b.bbox.xmin));
    for (t, b) in boxes.iter().enumerate() {
        let xc = 0.1 + 0.15 * t as f32;
        let want = [xc - 0.025, 0.35, xc + 0.025, 0.65];
        let got = [b.bbox.xmin, b.bbox.ymin, b.bbox.xmax, b.bbox.ymax];
        for (g, w) in got.iter().zip(want) {
            assert!((g - w).abs() < 1e-4, "box {t}: got {got:?}, want {want:?}");
        }
        assert_eq!(b.label, 0);
    }
}

fn decode(json: &str, outputs: &[TensorDyn]) -> Vec<DetectBox> {
    let decoder = DecoderBuilder::new()
        .with_config_json_str(json.to_string())
        .with_score_threshold(0.5)
        .with_iou_threshold(0.5)
        .build()
        .expect("the schema builds");
    let inputs: Vec<&TensorDyn> = outputs.iter().collect();
    let (mut boxes, mut masks) = (Vec::<DetectBox>::new(), Vec::<Segmentation>::new());
    decoder
        .decode(&inputs, &mut boxes, &mut masks)
        .expect("the outputs decode");
    boxes
}

#[test]
fn yolo26_with_end2end_false_decodes_the_pre_nms_head() {
    let json = schema_json(r#""model": {"end2end": false},"#, &pre_nms_head());
    assert_pre_nms_boxes(decode(&json, &pre_nms_outputs()));
}

/// Older exports carry the tag but no `end2end` flag. A head whose feature
/// count is not the end-to-end `6 + protos` cannot be end-to-end.
#[test]
fn yolo26_without_the_flag_decodes_a_head_that_cannot_be_end_to_end() {
    let json = schema_json("", &pre_nms_head());
    assert_pre_nms_boxes(decode(&json, &pre_nms_outputs()));
}

/// A schema that states `end2end: true` is held to it.
#[test]
fn yolo26_with_end2end_true_rejects_the_pre_nms_head() {
    let json = schema_json(r#""model": {"end2end": true},"#, &pre_nms_head());
    let err = DecoderBuilder::new()
        .with_config_json_str(json)
        .build()
        .expect_err("a pre-NMS head is not end-to-end");
    assert!(err.to_string().contains("6 + num_protos"), "{err}");
}

/// The embedded-NMS head, `[1, detections, 6 + protos]`, still takes the
/// end-to-end path, with the flag set or left out.
#[test]
fn yolo26_end_to_end_head_is_unchanged() {
    const DETS: usize = 100;
    const E2E: usize = 6 + NM;
    let head = format!(
        r#""shape": [1, {DETS}, {E2E}],
           "dshape": [{{"batch": 1}}, {{"num_boxes": {DETS}}}, {{"num_features": {E2E}}}]"#
    );
    let mut det = vec![0.0f32; DETS * E2E];
    for t in 0..TARGETS {
        let x1 = 0.1 + 0.15 * t as f32;
        det[t * E2E..t * E2E + 6].copy_from_slice(&[x1, 0.3, x1 + 0.05, 0.6, 0.9, 0.0]);
    }
    let outputs = vec![
        tensor(&[1, DETS, E2E], &det),
        tensor(&[1, NM, PH, PW], &vec![0.0f32; NM * PH * PW]),
    ];
    for model in ["", r#""model": {"end2end": true},"#] {
        let json = schema_json(model, &head);
        assert_eq!(
            decode(&json, &outputs).len(),
            TARGETS,
            "model block: {model:?}"
        );
    }
}
