// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! Outputs that share a shape are bound to their roles by name, else by
//! their configured position, and are refused when neither settles it.
//!
//! A split detector with four classes has scores the shape of its boxes
//! (`[1, 4, anchors]`); a segmentation model with as many classes as protos
//! has scores the shape of its mask coefficients. Binding those by shape
//! alone hands one role the other's tensor and decodes garbage without an
//! error.

use edgefirst_decoder::{
    configs::{self, DecoderType, DimName},
    ConfigOutput, DecoderBuilder, DecoderError, DetectBox, Segmentation,
};
use edgefirst_tensor::{Tensor, TensorDyn, TensorMapTrait, TensorMemory, TensorTrait};

const N: usize = 256;
const TARGETS: usize = 5;

fn tensor(name: Option<&str>, shape: &[usize], data: &[f32]) -> TensorDyn {
    let t = Tensor::<f32>::new(shape, Some(TensorMemory::Mem), name).unwrap();
    t.map().unwrap().as_mut_slice().copy_from_slice(data);
    TensorDyn::F32(t)
}

/// `[1, 4, N]` boxes with `TARGETS` separate boxes, as centre and size.
fn boxes_data() -> Vec<f32> {
    let mut boxes = vec![0.0f32; 4 * N];
    for t in 0..TARGETS {
        let anchor = 10 + t;
        boxes[anchor] = 0.1 + 0.15 * t as f32;
        boxes[N + anchor] = 0.5;
        boxes[2 * N + anchor] = 0.05;
        boxes[3 * N + anchor] = 0.3;
    }
    boxes
}

/// `[1, classes, N]` scores giving target `t` the class `t % classes`.
fn scores_data(classes: usize) -> Vec<f32> {
    let mut scores = vec![0.0f32; classes * N];
    for t in 0..TARGETS {
        scores[(t % classes) * N + 10 + t] = 0.9;
    }
    scores
}

/// The planted boxes, each with its own class. A swapped binding returns
/// the right count with the wrong corners and labels, so both are checked.
fn assert_planted(mut boxes: Vec<DetectBox>, classes: usize, tolerance: f32) {
    assert_eq!(boxes.len(), TARGETS, "{boxes:?}");
    boxes.sort_by(|a, b| a.bbox.xmin.total_cmp(&b.bbox.xmin));
    for (t, b) in boxes.iter().enumerate() {
        let xc = 0.1 + 0.15 * t as f32;
        let want = [xc - 0.025, 0.35, xc + 0.025, 0.65];
        let got = [b.bbox.xmin, b.bbox.ymin, b.bbox.xmax, b.bbox.ymax];
        for (g, w) in got.iter().zip(want) {
            assert!(
                (g - w).abs() < tolerance,
                "box {t}: got {got:?}, want {want:?}"
            );
        }
        assert_eq!(b.label, t % classes, "box {t}");
    }
}

fn boxes_config() -> ConfigOutput {
    ConfigOutput::Boxes(configs::Boxes {
        decoder: DecoderType::Ultralytics,
        quantization: None,
        shape: vec![1, 4, N],
        dshape: vec![
            (DimName::Batch, 1),
            (DimName::BoxCoords, 4),
            (DimName::NumBoxes, N),
        ],
        normalized: Some(true),
    })
}

fn scores_config(classes: usize) -> ConfigOutput {
    ConfigOutput::Scores(configs::Scores {
        decoder: DecoderType::Ultralytics,
        quantization: None,
        shape: vec![1, classes, N],
        dshape: vec![
            (DimName::Batch, 1),
            (DimName::NumClasses, classes),
            (DimName::NumBoxes, N),
        ],
    })
}

fn builder() -> DecoderBuilder {
    DecoderBuilder::new()
        .with_score_threshold(0.6)
        .with_iou_threshold(0.5)
        .with_nms(Some(configs::Nms::ClassAgnostic))
}

fn decode(
    decoder: &edgefirst_decoder::Decoder,
    outputs: &[&TensorDyn],
) -> Result<Vec<DetectBox>, DecoderError> {
    let (mut boxes, mut masks) = (Vec::<DetectBox>::new(), Vec::<Segmentation>::new());
    decoder.decode(outputs, &mut boxes, &mut masks)?;
    Ok(boxes)
}

/// Four classes: scores and boxes are both `[1, 4, N]`. Whichever the
/// configuration lists first, tensors passed in that order decode correctly.
#[test]
fn same_shape_outputs_bind_in_configured_order() {
    let boxes = tensor(None, &[1, 4, N], &boxes_data());
    let scores = tensor(None, &[1, 4, N], &scores_data(4));

    let boxes_first = builder()
        .add_output(boxes_config())
        .add_output(scores_config(4))
        .build()
        .unwrap();
    assert_planted(decode(&boxes_first, &[&boxes, &scores]).unwrap(), 4, 1e-4);

    let scores_first = builder()
        .add_output(scores_config(4))
        .add_output(boxes_config())
        .build()
        .unwrap();
    assert_planted(decode(&scores_first, &[&scores, &boxes]).unwrap(), 4, 1e-4);
}

fn named_schema() -> String {
    format!(
        r#"{{
            "schema_version": 2,
            "nms": "class_agnostic",
            "outputs": [
                {{
                    "name": "model/boxes", "type": "boxes",
                    "shape": [1, 4, {N}],
                    "dshape": [{{"batch": 1}}, {{"box_coords": 4}}, {{"num_boxes": {N}}}],
                    "decoder": "ultralytics", "encoding": "direct", "normalized": true
                }},
                {{
                    "name": "model/scores", "type": "scores",
                    "shape": [1, 4, {N}],
                    "dshape": [{{"batch": 1}}, {{"num_classes": 4}}, {{"num_boxes": {N}}}],
                    "decoder": "ultralytics", "score_format": "per_class"
                }}
            ]
        }}"#
    )
}

/// Tensors named like the schema's outputs are bound by name: in any order,
/// and with another tensor of the same shape among them.
#[test]
fn named_outputs_bind_by_name_in_any_order() {
    let decoder = builder()
        .with_config_json_str(named_schema())
        .build()
        .unwrap();
    let boxes = tensor(Some("model/boxes"), &[1, 4, N], &boxes_data());
    let scores = tensor(Some("model/scores"), &[1, 4, N], &scores_data(4));
    let other = tensor(Some("model/aux"), &[1, 4, N], &vec![0.7f32; 4 * N]);

    assert_planted(decode(&decoder, &[&scores, &boxes]).unwrap(), 4, 1e-4);
    assert_planted(
        decode(&decoder, &[&other, &scores, &boxes]).unwrap(),
        4,
        1e-4,
    );
}

/// With no names and a tensor count that is not the configured one, nothing
/// says which `[1, 4, N]` tensor is the boxes. That is an error naming the
/// roles and the shape, not a decode of whichever came first.
#[test]
fn same_shape_outputs_that_nothing_tells_apart_are_refused() {
    let decoder = builder()
        .add_output(boxes_config())
        .add_output(scores_config(4))
        .build()
        .unwrap();
    let boxes = tensor(None, &[1, 4, N], &boxes_data());
    let scores = tensor(None, &[1, 4, N], &scores_data(4));
    let other = tensor(None, &[1, 4, N], &vec![0.7f32; 4 * N]);

    let err = decode(&decoder, &[&other, &boxes, &scores]).expect_err("ambiguous outputs");
    let message = err.to_string();
    for part in ["boxes", "scores", "[1, 4, 256]"] {
        assert!(message.contains(part), "{part:?} missing from: {message}");
    }
}

/// A role whose tensor is missing is named in the error.
#[test]
fn a_missing_output_names_its_role() {
    let decoder = builder()
        .add_output(boxes_config())
        .add_output(scores_config(4))
        .build()
        .unwrap();
    let boxes = tensor(None, &[1, 4, N], &boxes_data());

    let message = decode(&decoder, &[&boxes])
        .expect_err("scores are missing")
        .to_string();
    assert!(message.contains("scores"), "{message}");
}

/// As many classes as protos: scores and mask coefficients are both
/// `[1, 32, N]`, and are told apart by their configured order.
#[test]
fn scores_and_mask_coefficients_of_one_shape_bind_in_configured_order() {
    const NM: usize = 32;
    const PROTO: usize = 160;
    let mask_coeff_config = ConfigOutput::MaskCoefficients(configs::MaskCoefficients {
        decoder: DecoderType::Ultralytics,
        quantization: None,
        shape: vec![1, NM, N],
        dshape: vec![
            (DimName::Batch, 1),
            (DimName::NumProtos, NM),
            (DimName::NumBoxes, N),
        ],
    });
    let protos_config = ConfigOutput::Protos(configs::Protos {
        decoder: DecoderType::Ultralytics,
        quantization: None,
        shape: vec![1, NM, PROTO, PROTO],
        dshape: vec![
            (DimName::Batch, 1),
            (DimName::NumProtos, NM),
            (DimName::Height, PROTO),
            (DimName::Width, PROTO),
        ],
    });
    let decoder = builder()
        .add_output(boxes_config())
        .add_output(mask_coeff_config)
        .add_output(scores_config(NM))
        .add_output(protos_config)
        .build()
        .unwrap();

    let boxes = tensor(None, &[1, 4, N], &boxes_data());
    let mask_coeff = tensor(None, &[1, NM, N], &vec![0.0f32; NM * N]);
    let scores = tensor(None, &[1, NM, N], &scores_data(NM));
    let protos = tensor(
        None,
        &[1, NM, PROTO, PROTO],
        &vec![0.0f32; NM * PROTO * PROTO],
    );
    assert_planted(
        decode(&decoder, &[&boxes, &mask_coeff, &scores, &protos]).unwrap(),
        NM,
        1e-4,
    );
}

/// The merge path dequantizes split outputs into one tensor per logical
/// output, in schema order, and those are bound to roles the same way: a
/// schema that lists the scores before boxes of the same shape still decodes.
#[test]
fn merged_outputs_of_one_shape_bind_in_schema_order() {
    let child = |name: &str, kind: &str| {
        format!(
            r#"{{
                "name": "{name}", "type": "{kind}",
                "shape": [1, 2, {N}],
                "dshape": [{{"batch": 1}}, {{"box_coords": 2}}, {{"num_boxes": {N}}}],
                "dtype": "uint8",
                "quantization": {{"scale": 0.00392156862745098, "zero_point": 0, "dtype": "uint8"}}
            }}"#
        )
    };
    let schema = format!(
        r#"{{
            "schema_version": 2,
            "nms": "class_agnostic",
            "outputs": [
                {{
                    "name": "scores", "type": "scores",
                    "shape": [1, 4, {N}],
                    "dshape": [{{"batch": 1}}, {{"num_classes": 4}}, {{"num_boxes": {N}}}],
                    "decoder": "ultralytics", "score_format": "per_class"
                }},
                {{
                    "name": "boxes", "type": "boxes",
                    "shape": [1, 4, {N}],
                    "dshape": [{{"batch": 1}}, {{"box_coords": 4}}, {{"num_boxes": {N}}}],
                    "decoder": "ultralytics", "encoding": "direct", "normalized": true,
                    "outputs": [{xy}, {wh}]
                }}
            ]
        }}"#,
        xy = child("boxes_xy", "boxes_xy"),
        wh = child("boxes_wh", "boxes_wh"),
    );
    let decoder = builder().with_config_json_str(schema).build().unwrap();

    let quantized = |rows: &[f32]| {
        let t = Tensor::<u8>::new(&[1, 2, N], Some(TensorMemory::Mem), None).unwrap();
        for (q, v) in t.map().unwrap().as_mut_slice().iter_mut().zip(rows) {
            *q = (v * 255.0).round() as u8;
        }
        TensorDyn::U8(t)
    };
    let all = boxes_data();
    let scores = tensor(None, &[1, 4, N], &scores_data(4));
    let (xy, wh) = (quantized(&all[..2 * N]), quantized(&all[2 * N..]));
    assert_planted(decode(&decoder, &[&scores, &xy, &wh]).unwrap(), 4, 0.01);
}

/// A tensor that carries an output's name belongs to that output, whichever
/// role is bound first: unnamed boxes passed after the named scores are not
/// given the scores tensor for standing at the boxes' configured position.
#[test]
fn a_named_tensor_is_not_taken_by_position_for_another_output() {
    let decoder = builder()
        .with_config_json_str(named_schema())
        .build()
        .unwrap();
    let boxes = tensor(None, &[1, 4, N], &boxes_data());
    let scores = tensor(Some("model/scores"), &[1, 4, N], &scores_data(4));

    assert_planted(decode(&decoder, &[&scores, &boxes]).unwrap(), 4, 1e-4);
}

/// ModelPack's `masks` head is not decoded, so a model whose other outputs
/// share a shape still decodes without it.
#[test]
fn an_unread_masks_head_is_not_required() {
    const SIDE: usize = 8;
    let schema = format!(
        r#"{{
            "schema_version": 2,
            "outputs": [
                {{
                    "name": "seg", "type": "segmentation", "decoder": "modelpack",
                    "shape": [1, {SIDE}, {SIDE}, 4],
                    "dshape": [{{"batch": 1}}, {{"height": {SIDE}}}, {{"width": {SIDE}}}, {{"num_classes": 4}}]
                }},
                {{
                    "name": "aux",
                    "shape": [1, {SIDE}, {SIDE}, 4]
                }},
                {{
                    "name": "mask", "type": "masks", "decoder": "modelpack",
                    "shape": [1, {SIDE}, {SIDE}],
                    "dshape": [{{"batch": 1}}, {{"height": {SIDE}}}, {{"width": {SIDE}}}]
                }}
            ]
        }}"#
    );
    let decoder = builder().with_config_json_str(schema).build().unwrap();
    let seg = tensor(
        Some("seg"),
        &[1, SIDE, SIDE, 4],
        &one_hot_rows(SIDE * SIDE, 4),
    );

    let (mut boxes, mut masks) = (Vec::<DetectBox>::new(), Vec::<Segmentation>::new());
    decoder.decode(&[&seg], &mut boxes, &mut masks).unwrap();
    assert_eq!(masks.len(), 1);
}

/// `rows` rows of `classes` values, row `r` scoring 1.0 for class
/// `r % classes`.
fn one_hot_rows(rows: usize, classes: usize) -> Vec<f32> {
    let mut data = vec![0.0f32; rows * classes];
    for r in 0..rows {
        data[r * classes + r % classes] = 1.0;
    }
    data
}

/// `data` in `[0, 1]` as an `f32` tensor, or as `u8` at a scale of 1/255.
fn unit_tensor(quantized: bool, shape: &[usize], data: &[f32]) -> TensorDyn {
    if !quantized {
        return tensor(None, shape, data);
    }
    let t = Tensor::<u8>::new(shape, Some(TensorMemory::Mem), None).unwrap();
    for (q, v) in t.map().unwrap().as_mut_slice().iter_mut().zip(data) {
        *q = (v * 255.0).round() as u8;
    }
    TensorDyn::U8(t)
}

fn mask_bytes(decoder: &edgefirst_decoder::Decoder, outputs: &[&TensorDyn]) -> Vec<u8> {
    let (mut boxes, mut masks) = (Vec::<DetectBox>::new(), Vec::<Segmentation>::new());
    decoder.decode(outputs, &mut boxes, &mut masks).unwrap();
    assert_eq!(masks.len(), 1);
    match &masks[0].segmentation {
        TensorDyn::U8(t) => t.map().unwrap().as_slice().to_vec(),
        other => panic!("mask is not u8: {other:?}"),
    }
}

/// A ModelPack segmentation head shaped like the boxes (`[1, rows, 1, 4]`)
/// is decoded from its own tensor, not from the boxes found first.
#[test]
fn modelpack_segmentation_shaped_like_the_boxes_reads_its_own_tensor() {
    const ROWS: usize = 16;
    for quantized in [false, true] {
        let quantization = quantized.then_some(configs::QuantTuple(1.0 / 255.0, 0));
        let boxes_config = ConfigOutput::Boxes(configs::Boxes {
            decoder: DecoderType::ModelPack,
            quantization,
            shape: vec![1, ROWS, 1, 4],
            dshape: vec![
                (DimName::Batch, 1),
                (DimName::NumBoxes, ROWS),
                (DimName::Padding, 1),
                (DimName::BoxCoords, 4),
            ],
            normalized: Some(true),
        });
        let scores_config = ConfigOutput::Scores(configs::Scores {
            decoder: DecoderType::ModelPack,
            quantization,
            shape: vec![1, ROWS, 3],
            dshape: vec![
                (DimName::Batch, 1),
                (DimName::NumBoxes, ROWS),
                (DimName::NumClasses, 3),
            ],
        });
        let segmentation_config = ConfigOutput::Segmentation(configs::Segmentation {
            decoder: DecoderType::ModelPack,
            quantization,
            shape: vec![1, ROWS, 1, 4],
            dshape: vec![
                (DimName::Batch, 1),
                (DimName::Height, ROWS),
                (DimName::Width, 1),
                (DimName::NumClasses, 4),
            ],
        });
        let segmentation_only = builder()
            .add_output(segmentation_config.clone())
            .build()
            .unwrap();
        let multitask = builder()
            .add_output(boxes_config)
            .add_output(scores_config)
            .add_output(segmentation_config)
            .build()
            .unwrap();

        let boxes = unit_tensor(quantized, &[1, ROWS, 1, 4], &[0.5f32; ROWS * 4]);
        let scores = unit_tensor(quantized, &[1, ROWS, 3], &[0.0f32; ROWS * 3]);
        let segmentation = unit_tensor(quantized, &[1, ROWS, 1, 4], &one_hot_rows(ROWS, 4));

        let want = mask_bytes(&segmentation_only, &[&segmentation]);
        assert_ne!(
            want,
            mask_bytes(&segmentation_only, &[&boxes]),
            "quantized={quantized}: the boxes must not decode to the same mask"
        );
        assert_eq!(
            mask_bytes(&multitask, &[&boxes, &scores, &segmentation]),
            want,
            "quantized={quantized}"
        );
    }
}

/// Two ModelPack detection heads of one shape are each decoded from their
/// own tensor, with their own anchors, in the configured order.
#[test]
fn modelpack_detection_heads_of_one_shape_bind_in_configured_order() {
    const GRID: usize = 4;
    const FEATURES: usize = 7;
    const ANCHORS: [f32; 2] = [0.1, 0.3];
    for quantized in [false, true] {
        let head = |anchor: f32| {
            ConfigOutput::Detection(configs::Detection {
                anchors: Some(vec![[anchor, anchor]]),
                decoder: DecoderType::ModelPack,
                quantization: quantized.then_some(configs::QuantTuple(0.2, 128)),
                shape: vec![1, GRID, GRID, FEATURES],
                dshape: vec![
                    (DimName::Batch, 1),
                    (DimName::Height, GRID),
                    (DimName::Width, GRID),
                    (DimName::NumAnchorsXFeatures, FEATURES),
                ],
                normalized: Some(true),
            })
        };
        let decoder = builder()
            .add_output(head(ANCHORS[0]))
            .add_output(head(ANCHORS[1]))
            .build()
            .unwrap();

        // Logits: -20 scores nothing, 0 is the anchor's own size, 10 is certain.
        let logits = |values: &[f32]| {
            let shape = [1, GRID, GRID, FEATURES];
            if !quantized {
                return tensor(None, &shape, values);
            }
            let t = Tensor::<u8>::new(&shape, Some(TensorMemory::Mem), None).unwrap();
            for (q, v) in t.map().unwrap().as_mut_slice().iter_mut().zip(values) {
                *q = (v / 0.2 + 128.0).round().clamp(0.0, 255.0) as u8;
            }
            TensorDyn::U8(t)
        };
        let empty = vec![-20.0f32; GRID * GRID * FEATURES];
        let mut planted = empty.clone();
        let cell = (GRID + 1) * FEATURES;
        planted[cell..cell + 6].copy_from_slice(&[0.0, 0.0, 0.0, 0.0, 10.0, 10.0]);
        let (empty, planted) = (logits(&empty), logits(&planted));

        for (outputs, anchor) in [
            ([&planted, &empty], ANCHORS[0]),
            ([&empty, &planted], ANCHORS[1]),
        ] {
            let boxes = decode(&decoder, &outputs).unwrap();
            assert_eq!(boxes.len(), 1, "quantized={quantized}: {boxes:?}");
            let width = boxes[0].bbox.xmax - boxes[0].bbox.xmin;
            assert!(
                (width - anchor).abs() < 0.02,
                "quantized={quantized}: width {width}, anchor {anchor}"
            );
        }
    }
}
