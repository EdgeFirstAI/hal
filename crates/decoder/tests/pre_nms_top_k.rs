// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! The pre-NMS candidate cap on a detection-only model with more
//! candidates than the default cap.
//!
//! The scene is a 20 × 20 grid of 400 anchors with disjoint boxes, each
//! scoring 0.9 for class 0 and 0.8 for class 1, so NMS suppresses nothing
//! and every candidate that reaches NMS is returned (with `max_det` raised
//! out of the way). Argmax yields 400 candidates, multi-label 800.

use edgefirst_decoder::{
    configs::{DecoderType, DimName, Nms},
    ConfigOutput, ConfigOutputs, Decoder, DecoderBuilder, DEFAULT_PRE_NMS_TOP_K,
    MULTI_LABEL_PRE_NMS_TOP_K,
};
use edgefirst_tensor::{Tensor, TensorDyn, TensorMapTrait, TensorMemory, TensorTrait};

const GRID: usize = 20;
const N: usize = GRID * GRID;
const NC: usize = 2;
const ROWS: usize = 4 + NC;

fn detection_cfg() -> ConfigOutput {
    ConfigOutput::Detection(edgefirst_decoder::configs::Detection {
        decoder: DecoderType::Ultralytics,
        quantization: None,
        shape: vec![1, ROWS, N],
        dshape: vec![
            (DimName::Batch, 1),
            (DimName::NumFeatures, ROWS),
            (DimName::NumBoxes, N),
        ],
        anchors: None,
        normalized: Some(true),
    })
}

fn detection_tensor() -> TensorDyn {
    let mut data = vec![0.0f32; ROWS * N];
    let cell = 1.0 / GRID as f32;
    for a in 0..N {
        let (gx, gy) = ((a % GRID) as f32, (a / GRID) as f32);
        let row = [
            (gx + 0.5) * cell,
            (gy + 0.5) * cell,
            0.4 * cell,
            0.4 * cell,
            0.9,
            0.8,
        ];
        for (r, v) in row.into_iter().enumerate() {
            data[r * N + a] = v;
        }
    }
    let t = Tensor::<f32>::new(&[1, ROWS, N], Some(TensorMemory::Mem), None).unwrap();
    t.map().unwrap().as_mut_slice().copy_from_slice(&data);
    TensorDyn::F32(t)
}

fn builder() -> DecoderBuilder {
    DecoderBuilder::default()
        .with_score_threshold(0.5)
        .with_iou_threshold(0.5)
        .with_nms(Some(Nms::ClassAware))
        .with_max_det(10_000)
}

fn from_metadata(multi_label: bool) -> DecoderBuilder {
    let mut cfg = ConfigOutputs::new(vec![detection_cfg()]);
    cfg.nms_multi_label = Some(multi_label);
    builder().with_config(cfg)
}

fn count(d: &Decoder) -> usize {
    let t = detection_tensor();
    let (mut boxes, mut masks) = (Vec::new(), Vec::new());
    d.decode(&[&t], &mut boxes, &mut masks).unwrap();
    boxes.len()
}

#[test]
fn default_cap_limits_detection_only_candidates() {
    let d = builder().add_output(detection_cfg()).build().unwrap();
    assert_eq!(d.pre_nms_top_k, DEFAULT_PRE_NMS_TOP_K);
    assert_eq!(DEFAULT_PRE_NMS_TOP_K, 300);
    assert_eq!(count(&d), 300, "400 candidates, capped to 300 before NMS");
}

#[test]
fn zero_is_unbounded() {
    let d = builder()
        .add_output(detection_cfg())
        .with_pre_nms_top_k(0)
        .build()
        .unwrap();
    assert_eq!(d.pre_nms_top_k, 0);
    assert_eq!(count(&d), N);

    let d = builder()
        .add_output(detection_cfg())
        .with_multi_label(true)
        .with_pre_nms_top_k(0)
        .build()
        .unwrap();
    assert_eq!(d.pre_nms_top_k, 0);
    assert_eq!(count(&d), N * NC);
}

#[test]
fn multi_label_defaults_to_ultralytics_max_nms() {
    assert_eq!(MULTI_LABEL_PRE_NMS_TOP_K, 30_000);
    let d = builder()
        .add_output(detection_cfg())
        .with_multi_label(true)
        .build()
        .unwrap();
    assert_eq!(d.pre_nms_top_k, MULTI_LABEL_PRE_NMS_TOP_K);
    assert_eq!(count(&d), N * NC, "every multi-label candidate reaches NMS");
}

#[test]
fn metadata_multi_label_defaults_to_ultralytics_max_nms() {
    let d = from_metadata(true).build().unwrap();
    assert!(d.multi_label());
    assert_eq!(d.pre_nms_top_k, MULTI_LABEL_PRE_NMS_TOP_K);
    assert_eq!(count(&d), N * NC);

    let d = from_metadata(false).build().unwrap();
    assert_eq!(d.pre_nms_top_k, DEFAULT_PRE_NMS_TOP_K);
}

#[test]
fn explicit_cap_wins_over_the_multi_label_default() {
    let d = builder()
        .add_output(detection_cfg())
        .with_pre_nms_top_k(300)
        .with_multi_label(true)
        .build()
        .unwrap();
    assert_eq!(d.pre_nms_top_k, 300);
    assert_eq!(count(&d), 300);

    // An explicit multi_label(false) over metadata `true` keeps the argmax default.
    let d = from_metadata(true).with_multi_label(false).build().unwrap();
    assert_eq!(d.pre_nms_top_k, DEFAULT_PRE_NMS_TOP_K);
}
