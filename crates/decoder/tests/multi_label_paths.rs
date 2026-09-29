// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! Multi-label decode across every NMS decode path.
//!
//! Every fixture shares one logical scene: anchor 0 clears the score
//! threshold for classes 0 and 1, anchor 1 only for class 2, and the other
//! anchors score zero. Argmax therefore yields two boxes and multi-label three,
//! with the extra box sharing anchor 0's bbox under a different label.

use edgefirst_decoder::{
    configs::{self, DecoderType, DimName, QuantTuple},
    ConfigOutput, Decoder, DecoderBuilder, DetectBox,
};
use edgefirst_tensor::{
    Quantization as TQ, Tensor, TensorDyn, TensorMapTrait, TensorMemory, TensorTrait,
};

const NC: usize = 3;
const N: usize = 4;
const NM: usize = 3;
const PH: usize = 8;
const PW: usize = 8;
const SCORE_THRESHOLD: f32 = 0.5;
const QSCALE: f32 = 0.01;

/// Per-anchor normalized XYWH boxes.
const BOXES_XYWH: [[f32; 4]; N] = [
    [0.3, 0.3, 0.2, 0.2],
    [0.7, 0.7, 0.2, 0.2],
    [0.5, 0.5, 0.1, 0.1],
    [0.2, 0.8, 0.1, 0.1],
];

/// Per-anchor class scores.
const SCORES: [[f32; NC]; N] = [
    [0.9, 0.8, 0.0],
    [0.0, 0.0, 0.9],
    [0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0],
];

fn mask_coef(anchor: usize, k: usize) -> f32 {
    0.1 * (anchor * NM + k + 1) as f32
}

fn xyxy(b: [f32; 4]) -> [f32; 4] {
    [
        b[0] - b[2] / 2.0,
        b[1] - b[3] / 2.0,
        b[0] + b[2] / 2.0,
        b[1] + b[3] / 2.0,
    ]
}

#[derive(Clone, Copy, Debug)]
enum Dtype {
    F16,
    F32,
    F64,
    I8,
    U8,
}

/// Build a tensor from logical f32 values, quantizing with `QSCALE` for
/// integer dtypes.
fn tensor(shape: &[usize], data: &[f32], dtype: Dtype) -> TensorDyn {
    assert_eq!(shape.iter().product::<usize>(), data.len());
    match dtype {
        Dtype::F16 => {
            let t = Tensor::<half::f16>::new(shape, Some(TensorMemory::Mem), None).unwrap();
            let h: Vec<half::f16> = data.iter().map(|&v| half::f16::from_f32(v)).collect();
            t.map().unwrap().as_mut_slice().copy_from_slice(&h);
            TensorDyn::F16(t)
        }
        Dtype::F32 => {
            let t = Tensor::<f32>::new(shape, Some(TensorMemory::Mem), None).unwrap();
            t.map().unwrap().as_mut_slice().copy_from_slice(data);
            TensorDyn::F32(t)
        }
        Dtype::F64 => {
            let t = Tensor::<f64>::new(shape, Some(TensorMemory::Mem), None).unwrap();
            let d: Vec<f64> = data.iter().map(|&v| f64::from(v)).collect();
            t.map().unwrap().as_mut_slice().copy_from_slice(&d);
            TensorDyn::F64(t)
        }
        Dtype::I8 => {
            let mut t = Tensor::<i8>::new(shape, Some(TensorMemory::Mem), None).unwrap();
            t.set_quantization(TQ::per_tensor(QSCALE, 0)).unwrap();
            let q: Vec<i8> = data.iter().map(|v| (v / QSCALE).round() as i8).collect();
            t.map().unwrap().as_mut_slice().copy_from_slice(&q);
            TensorDyn::I8(t)
        }
        Dtype::U8 => {
            let mut t = Tensor::<u8>::new(shape, Some(TensorMemory::Mem), None).unwrap();
            t.set_quantization(TQ::per_tensor(QSCALE, 0)).unwrap();
            let q: Vec<u8> = data.iter().map(|v| (v / QSCALE).round() as u8).collect();
            t.map().unwrap().as_mut_slice().copy_from_slice(&q);
            TensorDyn::U8(t)
        }
    }
}

fn quant(dtype: Dtype) -> Option<QuantTuple> {
    match dtype {
        Dtype::F16 | Dtype::F32 | Dtype::F64 => None,
        Dtype::I8 | Dtype::U8 => Some(QuantTuple(QSCALE, 0)),
    }
}

/// Feature-major `[rows, N]` data for Ultralytics outputs.
fn feature_major(rows: impl Fn(usize) -> Vec<f32>, n_rows: usize) -> Vec<f32> {
    let mut out = vec![0.0; n_rows * N];
    for a in 0..N {
        for (r, v) in rows(a).into_iter().enumerate() {
            out[r * N + a] = v;
        }
    }
    out
}

fn box_rows(a: usize) -> Vec<f32> {
    BOXES_XYWH[a].to_vec()
}

fn score_rows(a: usize) -> Vec<f32> {
    SCORES[a].to_vec()
}

fn coef_rows(a: usize) -> Vec<f32> {
    (0..NM).map(|k| mask_coef(a, k)).collect()
}

fn protos_tensor(dtype: Dtype) -> (ConfigOutput, TensorDyn) {
    let cfg = ConfigOutput::Protos(configs::Protos {
        decoder: DecoderType::Ultralytics,
        quantization: quant(dtype),
        shape: vec![1, PH, PW, NM],
        dshape: vec![
            (DimName::Batch, 1),
            (DimName::Height, PH),
            (DimName::Width, PW),
            (DimName::NumProtos, NM),
        ],
    });
    let data = vec![0.1f32; PH * PW * NM];
    (cfg, tensor(&[1, PH, PW, NM], &data, dtype))
}

fn detection_cfg(rows: usize, dtype: Dtype, decoder: DecoderType) -> ConfigOutput {
    ConfigOutput::Detection(configs::Detection {
        decoder,
        quantization: quant(dtype),
        shape: vec![1, rows, N],
        dshape: vec![
            (DimName::Batch, 1),
            (DimName::NumFeatures, rows),
            (DimName::NumBoxes, N),
        ],
        anchors: None,
        normalized: Some(true),
    })
}

fn boxes_cfg(dtype: Dtype) -> ConfigOutput {
    ConfigOutput::Boxes(configs::Boxes {
        decoder: DecoderType::Ultralytics,
        quantization: quant(dtype),
        shape: vec![1, 4, N],
        dshape: vec![
            (DimName::Batch, 1),
            (DimName::BoxCoords, 4),
            (DimName::NumBoxes, N),
        ],
        normalized: Some(true),
    })
}

fn scores_cfg(dtype: Dtype) -> ConfigOutput {
    ConfigOutput::Scores(configs::Scores {
        decoder: DecoderType::Ultralytics,
        quantization: quant(dtype),
        shape: vec![1, NC, N],
        dshape: vec![
            (DimName::Batch, 1),
            (DimName::NumClasses, NC),
            (DimName::NumBoxes, N),
        ],
    })
}

fn coefs_cfg(dtype: Dtype) -> ConfigOutput {
    ConfigOutput::MaskCoefficients(configs::MaskCoefficients {
        decoder: DecoderType::Ultralytics,
        quantization: quant(dtype),
        shape: vec![1, NM, N],
        dshape: vec![
            (DimName::Batch, 1),
            (DimName::NumProtos, NM),
            (DimName::NumBoxes, N),
        ],
    })
}

/// A model under test: its output configs and matching tensors.
struct Fixture {
    configs: Vec<ConfigOutput>,
    tensors: Vec<TensorDyn>,
    has_protos: bool,
}

fn yolo_det(dtype: Dtype) -> Fixture {
    let rows = 4 + NC;
    let data = feature_major(|a| [box_rows(a), score_rows(a)].concat(), rows);
    Fixture {
        configs: vec![detection_cfg(rows, dtype, DecoderType::Ultralytics)],
        tensors: vec![tensor(&[1, rows, N], &data, dtype)],
        has_protos: false,
    }
}

fn yolo_split_det(dtype: Dtype) -> Fixture {
    Fixture {
        configs: vec![boxes_cfg(dtype), scores_cfg(dtype)],
        tensors: vec![
            tensor(&[1, 4, N], &feature_major(box_rows, 4), dtype),
            tensor(&[1, NC, N], &feature_major(score_rows, NC), dtype),
        ],
        has_protos: false,
    }
}

fn yolo_segdet(dtype: Dtype) -> Fixture {
    let rows = 4 + NC + NM;
    let data = feature_major(
        |a| [box_rows(a), score_rows(a), coef_rows(a)].concat(),
        rows,
    );
    let (pcfg, pt) = protos_tensor(dtype);
    Fixture {
        configs: vec![detection_cfg(rows, dtype, DecoderType::Ultralytics), pcfg],
        tensors: vec![tensor(&[1, rows, N], &data, dtype), pt],
        has_protos: true,
    }
}

fn yolo_split_segdet(dtype: Dtype) -> Fixture {
    let (pcfg, pt) = protos_tensor(dtype);
    Fixture {
        configs: vec![boxes_cfg(dtype), scores_cfg(dtype), coefs_cfg(dtype), pcfg],
        tensors: vec![
            tensor(&[1, 4, N], &feature_major(box_rows, 4), dtype),
            tensor(&[1, NC, N], &feature_major(score_rows, NC), dtype),
            tensor(&[1, NM, N], &feature_major(coef_rows, NM), dtype),
            pt,
        ],
        has_protos: true,
    }
}

fn yolo_segdet_2way(dtype: Dtype) -> Fixture {
    let rows = 4 + NC;
    let data = feature_major(|a| [box_rows(a), score_rows(a)].concat(), rows);
    let (pcfg, pt) = protos_tensor(dtype);
    Fixture {
        configs: vec![
            detection_cfg(rows, dtype, DecoderType::Ultralytics),
            coefs_cfg(dtype),
            pcfg,
        ],
        tensors: vec![
            tensor(&[1, rows, N], &data, dtype),
            tensor(&[1, NM, N], &feature_major(coef_rows, NM), dtype),
            pt,
        ],
        has_protos: true,
    }
}

fn modelpack_det(dtype: Dtype) -> Fixture {
    let boxes: Vec<f32> = (0..N).flat_map(|a| xyxy(BOXES_XYWH[a])).collect();
    let scores: Vec<f32> = (0..N).flat_map(|a| SCORES[a]).collect();
    Fixture {
        configs: vec![
            ConfigOutput::Boxes(configs::Boxes {
                decoder: DecoderType::ModelPack,
                quantization: quant(dtype),
                shape: vec![1, N, 1, 4],
                dshape: vec![
                    (DimName::Batch, 1),
                    (DimName::NumBoxes, N),
                    (DimName::Padding, 1),
                    (DimName::BoxCoords, 4),
                ],
                normalized: Some(true),
            }),
            ConfigOutput::Scores(configs::Scores {
                decoder: DecoderType::ModelPack,
                quantization: quant(dtype),
                shape: vec![1, N, NC],
                dshape: vec![
                    (DimName::Batch, 1),
                    (DimName::NumBoxes, N),
                    (DimName::NumClasses, NC),
                ],
            }),
        ],
        tensors: vec![
            tensor(&[1, N, 1, 4], &boxes, dtype),
            tensor(&[1, N, NC], &scores, dtype),
        ],
        has_protos: false,
    }
}

fn build(fx: &Fixture, multi_label: Option<bool>) -> Decoder {
    let mut b = DecoderBuilder::default()
        .with_score_threshold(SCORE_THRESHOLD)
        .with_iou_threshold(0.5);
    for c in &fx.configs {
        b = b.add_output(c.clone());
    }
    if let Some(v) = multi_label {
        b = b.with_multi_label(v);
    }
    b.build().expect("fixture decoder must build")
}

fn decode(fx: &Fixture, multi_label: Option<bool>) -> Vec<DetectBox> {
    let d = build(fx, multi_label);
    let inputs: Vec<&TensorDyn> = fx.tensors.iter().collect();
    let mut boxes = Vec::with_capacity(16);
    let mut masks = Vec::with_capacity(16);
    d.decode(&inputs, &mut boxes, &mut masks).expect("decode");
    if fx.has_protos {
        assert_eq!(masks.len(), boxes.len(), "one mask per box");
    }
    boxes
}

fn decode_proto(fx: &Fixture, multi_label: bool) -> (Vec<DetectBox>, usize) {
    let d = build(fx, Some(multi_label));
    let inputs: Vec<&TensorDyn> = fx.tensors.iter().collect();
    let mut boxes = Vec::with_capacity(16);
    let proto = d
        .decode_proto(&inputs, &mut boxes)
        .expect("decode_proto")
        .expect("seg model returns proto data");
    (boxes, proto.mask_coefficients.shape()[0])
}

fn labels(boxes: &[DetectBox]) -> Vec<usize> {
    let mut l: Vec<usize> = boxes.iter().map(|b| b.label).collect();
    l.sort_unstable();
    l
}

/// The shared contract every path must meet.
fn check(name: &str, fx: &Fixture) {
    let default = decode(fx, None);
    let argmax = decode(fx, Some(false));
    let multi = decode(fx, Some(true));

    assert_eq!(default, argmax, "{name}: argmax must equal the default");
    assert_eq!(labels(&argmax), vec![0, 2], "{name}: argmax labels");
    assert_eq!(labels(&multi), vec![0, 1, 2], "{name}: multi-label labels");

    let a0 = argmax.iter().find(|b| b.label == 0).unwrap();
    let extra = multi.iter().find(|b| b.label == 1).unwrap();
    let tol = 2.0 * QSCALE;
    assert!(
        (a0.bbox.xmin - extra.bbox.xmin).abs() < tol
            && (a0.bbox.ymin - extra.bbox.ymin).abs() < tol
            && (a0.bbox.xmax - extra.bbox.xmax).abs() < tol
            && (a0.bbox.ymax - extra.bbox.ymax).abs() < tol,
        "{name}: extra class shares anchor 0's bbox ({a0:?} vs {extra:?})"
    );

    if fx.has_protos {
        let (pa, na) = decode_proto(fx, false);
        let (pm, nm) = decode_proto(fx, true);
        assert_eq!(labels(&pa), vec![0, 2], "{name}: proto argmax labels");
        assert_eq!(
            labels(&pm),
            vec![0, 1, 2],
            "{name}: proto multi-label labels"
        );
        assert_eq!(na, pa.len(), "{name}: one coefficient row per argmax box");
        assert_eq!(
            nm,
            pm.len(),
            "{name}: one coefficient row per multi-label box"
        );
    }
}

#[test]
fn yolo_det_f32() {
    check("yolo_det f32", &yolo_det(Dtype::F32));
}

#[test]
fn yolo_det_i8() {
    check("yolo_det i8", &yolo_det(Dtype::I8));
}

#[test]
fn yolo_det_f16() {
    check("yolo_det f16", &yolo_det(Dtype::F16));
}

#[test]
fn yolo_det_f64() {
    check("yolo_det f64", &yolo_det(Dtype::F64));
}

#[test]
fn yolo_split_segdet_f16() {
    check("yolo_split_segdet f16", &yolo_split_segdet(Dtype::F16));
}

#[test]
fn yolo_segdet_2way_f64() {
    check("yolo_segdet_2way f64", &yolo_segdet_2way(Dtype::F64));
}

#[test]
fn modelpack_det_f16() {
    check("modelpack_det f16", &modelpack_det(Dtype::F16));
}

#[test]
fn yolo_split_det_f32() {
    check("yolo_split_det f32", &yolo_split_det(Dtype::F32));
}

#[test]
fn yolo_split_det_i8() {
    check("yolo_split_det i8", &yolo_split_det(Dtype::I8));
}

#[test]
fn yolo_segdet_f32() {
    check("yolo_segdet f32", &yolo_segdet(Dtype::F32));
}

#[test]
fn yolo_segdet_i8() {
    check("yolo_segdet i8", &yolo_segdet(Dtype::I8));
}

#[test]
fn yolo_split_segdet_f32() {
    check("yolo_split_segdet f32", &yolo_split_segdet(Dtype::F32));
}

#[test]
fn yolo_split_segdet_i8() {
    check("yolo_split_segdet i8", &yolo_split_segdet(Dtype::I8));
}

#[test]
fn yolo_segdet_2way_f32() {
    check("yolo_segdet_2way f32", &yolo_segdet_2way(Dtype::F32));
}

#[test]
fn yolo_segdet_2way_i8() {
    check("yolo_segdet_2way i8", &yolo_segdet_2way(Dtype::I8));
}

#[test]
fn modelpack_det_f32() {
    check("modelpack_det f32", &modelpack_det(Dtype::F32));
}

#[test]
fn modelpack_det_u8() {
    check("modelpack_det u8", &modelpack_det(Dtype::U8));
}

/// The fused float path already supported multi-label; split float must agree.
#[test]
fn split_float_matches_fused_float_under_multi_label() {
    let mut fused = decode(&yolo_segdet(Dtype::F32), Some(true));
    let mut split = decode(&yolo_split_segdet(Dtype::F32), Some(true));
    let key = |b: &DetectBox| (b.label, (b.score * 1e6) as i64);
    fused.sort_by_key(key);
    split.sort_by_key(key);
    assert_eq!(fused, split);
}

/// Multi-label with NMS bypassed keeps every per-class candidate.
#[test]
fn multi_label_with_nms_bypassed_keeps_all_candidates() {
    let fx = yolo_det(Dtype::F32);
    let mut b = DecoderBuilder::default()
        .with_score_threshold(SCORE_THRESHOLD)
        .with_nms(None)
        .with_multi_label(true);
    for c in &fx.configs {
        b = b.add_output(c.clone());
    }
    let d = b.build().unwrap();
    let inputs: Vec<&TensorDyn> = fx.tensors.iter().collect();
    let (mut boxes, mut masks) = (Vec::with_capacity(16), Vec::new());
    d.decode(&inputs, &mut boxes, &mut masks).unwrap();
    assert_eq!(labels(&boxes), vec![0, 1, 2]);
}

/// ModelPack detection honours the configured NMS mode: class-aware keeps an
/// overlapping box of another class that class-agnostic suppresses.
#[test]
fn modelpack_det_honours_nms_mode() {
    let fx = modelpack_det_overlapping_classes();
    let run = |nms: configs::Nms| {
        let mut b = DecoderBuilder::default()
            .with_score_threshold(SCORE_THRESHOLD)
            .with_iou_threshold(0.5)
            .with_nms(Some(nms));
        for c in &fx.configs {
            b = b.add_output(c.clone());
        }
        let d = b.build().unwrap();
        let inputs: Vec<&TensorDyn> = fx.tensors.iter().collect();
        let (mut boxes, mut masks) = (Vec::with_capacity(16), Vec::new());
        d.decode(&inputs, &mut boxes, &mut masks).unwrap();
        labels(&boxes)
    };
    assert_eq!(run(configs::Nms::ClassAgnostic), vec![0]);
    assert_eq!(run(configs::Nms::ClassAware), vec![0, 1]);
}

/// Two anchors with the same bbox, argmax class 0 and class 1 respectively.
fn modelpack_det_overlapping_classes() -> Fixture {
    let b = xyxy(BOXES_XYWH[0]);
    let boxes: Vec<f32> = [b, b, [0.0; 4], [0.0; 4]].concat();
    let scores: Vec<f32> = [[0.9, 0.0, 0.0], [0.0, 0.8, 0.0], [0.0; NC], [0.0; NC]].concat();
    let mut fx = modelpack_det(Dtype::F32);
    fx.tensors = vec![
        tensor(&[1, N, 1, 4], &boxes, Dtype::F32),
        tensor(&[1, N, NC], &scores, Dtype::F32),
    ];
    fx
}

/// Tracked decode always uses one label per box, whatever the decoder's
/// multi-label setting, and never panics.
#[cfg(feature = "tracker")]
fn tracked_boxes(fx: &Fixture, multi_label: bool) -> (Vec<DetectBox>, usize) {
    use edgefirst_tracker::ByteTrackBuilder;
    let d = build(fx, Some(multi_label));
    let inputs: Vec<&TensorDyn> = fx.tensors.iter().collect();
    let mut tracker = ByteTrackBuilder::new().build::<DetectBox>();
    let (mut boxes, mut masks, mut tracks) = (Vec::with_capacity(16), Vec::new(), Vec::new());
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
}

#[cfg(feature = "tracker")]
#[test]
fn tracked_decode_ignores_multi_label_yolo_det_f32() {
    let fx = yolo_det(Dtype::F32);
    assert_eq!(tracked_boxes(&fx, true), tracked_boxes(&fx, false));
}

#[cfg(feature = "tracker")]
#[test]
fn tracked_decode_ignores_multi_label_yolo_det_i8() {
    let fx = yolo_det(Dtype::I8);
    assert_eq!(tracked_boxes(&fx, true), tracked_boxes(&fx, false));
}

#[cfg(feature = "tracker")]
#[test]
fn tracked_decode_ignores_multi_label_yolo_segdet_f32() {
    let fx = yolo_segdet(Dtype::F32);
    assert_eq!(tracked_boxes(&fx, true), tracked_boxes(&fx, false));
}

#[cfg(feature = "tracker")]
#[test]
fn tracked_decode_ignores_multi_label_modelpack_det_u8() {
    let fx = modelpack_det(Dtype::U8);
    assert_eq!(tracked_boxes(&fx, true), tracked_boxes(&fx, false));
}

/// Multi-label declared only by model metadata is also ignored when tracking.
#[cfg(feature = "tracker")]
#[test]
fn tracked_decode_ignores_metadata_multi_label() {
    use edgefirst_decoder::{schema::SchemaV2, ConfigOutputs};
    use edgefirst_tracker::ByteTrackBuilder;
    let fx = yolo_det(Dtype::F32);
    let run = |nms_multi_label: bool| {
        let mut schema = SchemaV2::from_v1(&ConfigOutputs {
            outputs: fx.configs.clone(),
            ..Default::default()
        })
        .unwrap();
        schema.nms_multi_label = Some(nms_multi_label);
        let d = DecoderBuilder::default()
            .with_schema(schema)
            .with_score_threshold(SCORE_THRESHOLD)
            .with_iou_threshold(0.5)
            .build()
            .unwrap();
        assert_eq!(d.multi_label(), nms_multi_label);
        let inputs: Vec<&TensorDyn> = fx.tensors.iter().collect();
        let mut tracker = ByteTrackBuilder::new().build::<DetectBox>();
        let (mut boxes, mut masks, mut tracks) = (Vec::with_capacity(16), Vec::new(), Vec::new());
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

#[cfg(feature = "tracker")]
#[test]
fn tracked_decode_proto_ignores_multi_label_yolo_segdet_f32() {
    use edgefirst_tracker::ByteTrackBuilder;
    let fx = yolo_segdet(Dtype::F32);
    let run = |multi_label: bool| {
        let d = build(&fx, Some(multi_label));
        let inputs: Vec<&TensorDyn> = fx.tensors.iter().collect();
        let mut tracker = ByteTrackBuilder::new().build::<DetectBox>();
        let (mut boxes, mut tracks) = (Vec::with_capacity(16), Vec::new());
        for ts in 0..3 {
            d.decode_proto_tracked(&mut tracker, ts, &inputs, &mut boxes, &mut tracks)
                .expect("decode_proto_tracked");
        }
        (boxes, tracks.len())
    };
    assert_eq!(run(true), run(false));
}

/// The single-label decode for external trackers ignores multi-label.
#[test]
fn decode_for_tracking_is_single_label() {
    let fx = yolo_det(Dtype::F32);
    let d = build(&fx, Some(true));
    let inputs: Vec<&TensorDyn> = fx.tensors.iter().collect();
    let (mut boxes, mut masks) = (Vec::with_capacity(16), Vec::new());
    d.decode_for_tracking(&inputs, &mut boxes, &mut masks)
        .unwrap();
    assert_eq!(labels(&boxes), vec![0, 2]);
}

/// `Decoder::max_det` and `pre_nms_top_k` bound the output of every NMS path,
/// and the caller's `Vec` capacity is only an allocation hint.
fn decode_with(fx: &Fixture, tune: impl Fn(DecoderBuilder) -> DecoderBuilder, cap: usize) -> usize {
    let mut b = DecoderBuilder::default()
        .with_score_threshold(SCORE_THRESHOLD)
        .with_iou_threshold(0.5)
        .with_multi_label(true);
    for c in &fx.configs {
        b = b.add_output(c.clone());
    }
    let d = tune(b).build().unwrap();
    let inputs: Vec<&TensorDyn> = fx.tensors.iter().collect();
    let (mut boxes, mut masks) = (Vec::with_capacity(cap), Vec::new());
    d.decode(&inputs, &mut boxes, &mut masks).unwrap();
    boxes.len()
}

fn check_caps(name: &str, fx: &Fixture) {
    assert_eq!(decode_with(fx, |b| b, 0), 3, "{name}: uncapped");
    assert_eq!(
        decode_with(fx, |b| b.with_max_det(1), 0),
        1,
        "{name}: max_det"
    );
    assert_eq!(
        decode_with(fx, |b| b.with_pre_nms_top_k(1), 0),
        1,
        "{name}: pre_nms_top_k"
    );
    assert_eq!(decode_with(fx, |b| b, 1), 3, "{name}: capacity is a hint");
}

#[test]
fn caps_yolo_det() {
    check_caps("yolo_det f32", &yolo_det(Dtype::F32));
    check_caps("yolo_det i8", &yolo_det(Dtype::I8));
}

#[test]
fn caps_yolo_split_det() {
    check_caps("yolo_split_det f32", &yolo_split_det(Dtype::F32));
    check_caps("yolo_split_det i8", &yolo_split_det(Dtype::I8));
}

#[test]
fn caps_modelpack_det() {
    check_caps("modelpack_det f32", &modelpack_det(Dtype::F32));
    check_caps("modelpack_det u8", &modelpack_det(Dtype::U8));
}

#[test]
fn caps_yolo_segdet() {
    check_caps("yolo_segdet f32", &yolo_segdet(Dtype::F32));
    check_caps("yolo_segdet i8", &yolo_segdet(Dtype::I8));
}

/// End-to-end (post-NMS) models honour `max_det`, not the `Vec` capacity.
#[test]
fn caps_yolo_end_to_end_det() {
    // Rows: [x1, y1, x2, y2, conf, class] for three detections.
    let rows = [
        [0.1f32, 0.1, 0.3, 0.3, 0.9, 0.0],
        [0.5, 0.5, 0.7, 0.7, 0.8, 1.0],
        [0.2, 0.6, 0.4, 0.8, 0.7, 2.0],
    ];
    let data: Vec<f32> = rows.iter().flatten().copied().collect();
    let cfg = ConfigOutput::Detection(configs::Detection {
        decoder: DecoderType::Ultralytics,
        quantization: None,
        shape: vec![1, 3, 6],
        dshape: vec![
            (DimName::Batch, 1),
            (DimName::NumBoxes, 3),
            (DimName::NumFeatures, 6),
        ],
        anchors: None,
        normalized: Some(true),
    });
    let t = tensor(&[1, 3, 6], &data, Dtype::F32);
    let run = |max_det: Option<usize>, cap: usize| {
        let mut b = DecoderBuilder::default()
            .with_score_threshold(SCORE_THRESHOLD)
            .with_decoder_version(configs::DecoderVersion::Yolo26)
            .add_output(cfg.clone());
        if let Some(m) = max_det {
            b = b.with_max_det(m);
        }
        let d = b.build().unwrap();
        let (mut boxes, mut masks) = (Vec::with_capacity(cap), Vec::new());
        d.decode(&[&t], &mut boxes, &mut masks).unwrap();
        boxes.len()
    };
    assert_eq!(run(None, 0), 3);
    assert_eq!(run(Some(1), 0), 1, "max_det");
    assert_eq!(run(None, 1), 3, "capacity is a hint");
}

/// Build a decoder with an explicit NMS mode, optionally declared by the
/// config (`nms` key) rather than the builder.
fn build_nms(fx: &Fixture, nms: configs::Nms, from_config: bool, multi_label: bool) -> Decoder {
    let mut b = DecoderBuilder::default()
        .with_score_threshold(SCORE_THRESHOLD)
        .with_iou_threshold(0.5)
        .with_multi_label(multi_label);
    if from_config {
        let cfg = edgefirst_decoder::ConfigOutputs {
            outputs: fx.configs.clone(),
            nms: Some(nms),
            ..Default::default()
        };
        b = b.with_config(cfg).with_nms(Some(configs::Nms::Auto));
    } else {
        for c in &fx.configs {
            b = b.add_output(c.clone());
        }
        b = b.with_nms(Some(nms));
    }
    b.build().expect("fixture decoder must build")
}

/// Multi-label candidates go through the NMS mode the caller or config chose.
/// Anchor 0's class-0 and class-1 candidates share one bbox, so class-agnostic
/// NMS keeps only the higher-scoring class 0, as Ultralytics
/// `non_max_suppression(multi_label=True, agnostic=True)` does, while
/// class-aware NMS keeps both.
fn check_nms_mode_under_multi_label(name: &str, fx: &Fixture) {
    for from_config in [false, true] {
        let run = |nms: configs::Nms| {
            let d = build_nms(fx, nms, from_config, true);
            assert_eq!(d.nms, Some(nms), "{name}: resolved NMS mode");
            let inputs: Vec<&TensorDyn> = fx.tensors.iter().collect();
            let (mut boxes, mut masks) = (Vec::with_capacity(16), Vec::with_capacity(16));
            d.decode(&inputs, &mut boxes, &mut masks).expect("decode");
            let decoded = labels(&boxes);
            if fx.has_protos {
                let mut pboxes = Vec::with_capacity(16);
                d.decode_proto(&inputs, &mut pboxes)
                    .expect("decode_proto")
                    .expect("seg model returns proto data");
                assert_eq!(labels(&pboxes), decoded, "{name}: decode_proto agrees");
            }
            decoded
        };
        assert_eq!(
            run(configs::Nms::ClassAgnostic),
            vec![0, 2],
            "{name} (from_config={from_config}): class-agnostic suppresses across classes"
        );
        assert_eq!(
            run(configs::Nms::ClassAware),
            vec![0, 1, 2],
            "{name} (from_config={from_config}): class-aware keeps every class"
        );
    }
}

#[test]
fn nms_mode_under_multi_label_yolo_det() {
    check_nms_mode_under_multi_label("yolo_det f32", &yolo_det(Dtype::F32));
    check_nms_mode_under_multi_label("yolo_det i8", &yolo_det(Dtype::I8));
}

#[test]
fn nms_mode_under_multi_label_yolo_split_det() {
    check_nms_mode_under_multi_label("yolo_split_det f32", &yolo_split_det(Dtype::F32));
    check_nms_mode_under_multi_label("yolo_split_det i8", &yolo_split_det(Dtype::I8));
}

#[test]
fn nms_mode_under_multi_label_yolo_segdet() {
    check_nms_mode_under_multi_label("yolo_segdet f32", &yolo_segdet(Dtype::F32));
    check_nms_mode_under_multi_label("yolo_segdet i8", &yolo_segdet(Dtype::I8));
}

#[test]
fn nms_mode_under_multi_label_yolo_split_segdet() {
    check_nms_mode_under_multi_label("yolo_split_segdet f32", &yolo_split_segdet(Dtype::F32));
    check_nms_mode_under_multi_label("yolo_split_segdet i8", &yolo_split_segdet(Dtype::I8));
}

#[test]
fn nms_mode_under_multi_label_yolo_segdet_2way() {
    check_nms_mode_under_multi_label("yolo_segdet_2way f32", &yolo_segdet_2way(Dtype::F32));
    check_nms_mode_under_multi_label("yolo_segdet_2way i8", &yolo_segdet_2way(Dtype::I8));
}

#[test]
fn nms_mode_under_multi_label_modelpack_det() {
    check_nms_mode_under_multi_label("modelpack_det f32", &modelpack_det(Dtype::F32));
    check_nms_mode_under_multi_label("modelpack_det u8", &modelpack_det(Dtype::U8));
}
