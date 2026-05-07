// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! Regression test for EDGEAI-1302: `Decoder::decode()` and
//! `Decoder::decode_proto()` must respect the decoder's own `max_det`
//! regardless of the caller's `Vec::capacity()` — capacity is purely
//! an allocation hint.
//!
//! Before the fix, both entrypoints computed the post-NMS truncate as
//! `self.max_det.min(output_boxes.capacity())`, which silently
//! truncated everything to zero when callers passed `Vec::new()`.

use edgefirst_decoder::{
    configs::{self, DecoderType, DimName, QuantTuple},
    ConfigOutput, DecoderBuilder, DetectBox,
};
use edgefirst_tensor::{Tensor, TensorDyn, TensorMapTrait, TensorMemory, TensorTrait};

/// Build a YOLOv8-seg-shaped synthetic decoder with N=256 anchors of
/// which 10 (anchors 10..20) carry plausible class-0 detections that
/// should survive score-filter and NMS. NMS is class-agnostic with a
/// near-1.0 IoU threshold so all 10 survive separately.
fn build_synthetic_segdet_decoder() -> (
    edgefirst_decoder::Decoder,
    Vec<TensorDyn>,
    usize, // expected detection count after NMS
) {
    const NC: usize = 2;
    const NM: usize = 32;
    const N: usize = 256;
    const FEAT: usize = 4 + NC + NM;
    const PH: usize = 160;
    const PW: usize = 160;

    let detection_cfg = configs::Detection {
        decoder: DecoderType::Ultralytics,
        quantization: Some(QuantTuple(1.0, 0)),
        shape: vec![1, FEAT, N],
        dshape: vec![],
        anchors: None,
        normalized: Some(true),
    };
    let protos_cfg = configs::Protos {
        decoder: DecoderType::Ultralytics,
        quantization: Some(QuantTuple(1.0, 0)),
        shape: vec![1, NM, PH, PW],
        dshape: vec![
            (DimName::Batch, 1),
            (DimName::NumProtos, NM),
            (DimName::Height, PH),
            (DimName::Width, PW),
        ],
    };

    let decoder = DecoderBuilder::default()
        .with_score_threshold(0.5)
        .with_iou_threshold(0.99)
        .with_nms(Some(configs::Nms::ClassAgnostic))
        .add_output(ConfigOutput::Detection(detection_cfg))
        .add_output(ConfigOutput::Protos(protos_cfg))
        .build()
        .expect("synthetic segdet decoder must build");

    // Detection tensor (1, FEAT, N): plant 10 detections at anchors 10..20.
    let n_targets = 10usize;
    let target_start = 10usize;
    let mut det_data = vec![0.0f32; FEAT * N];
    let set = |d: &mut [f32], r: usize, c: usize, v: f32| d[r * N + c] = v;
    for t in 0..n_targets {
        let anchor = target_start + t;
        // Non-overlapping boxes along x-axis so NMS doesn't suppress.
        let xc = 0.05 + 0.08 * t as f32;
        set(&mut det_data, 0, anchor, xc);
        set(&mut det_data, 1, anchor, 0.5);
        set(&mut det_data, 2, anchor, 0.06);
        set(&mut det_data, 3, anchor, 0.4);
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

    // Protos tensor (1, NM, PH, PW): zero-initialised, doesn't affect bbox path.
    let proto_data = vec![0.0f32; NM * PH * PW];
    let protos_tensor: TensorDyn = {
        let t = Tensor::<f32>::new(&[1, NM, PH, PW], Some(TensorMemory::Mem), None).unwrap();
        {
            let mut m = t.map().unwrap();
            m.as_mut_slice().copy_from_slice(&proto_data);
        }
        TensorDyn::F32(t)
    };

    (decoder, vec![det_tensor, protos_tensor], n_targets)
}

#[test]
fn decode_with_empty_vec_returns_detections() {
    let (decoder, owned, expected) = build_synthetic_segdet_decoder();
    let inputs: Vec<&TensorDyn> = owned.iter().collect();

    let mut output_boxes: Vec<DetectBox> = Vec::new();
    let mut output_masks: Vec<edgefirst_decoder::Segmentation> = Vec::new();
    decoder
        .decode(&inputs, &mut output_boxes, &mut output_masks)
        .expect("decode must succeed with Vec::new() outputs");

    assert!(
        !output_boxes.is_empty(),
        "decode with capacity-0 output Vec dropped all detections (EDGEAI-1302); \
         expected {expected}, got {}",
        output_boxes.len()
    );
    assert!(
        output_boxes.len() <= decoder.max_det,
        "decode produced {} boxes, exceeds max_det={}",
        output_boxes.len(),
        decoder.max_det
    );
    assert_eq!(
        output_boxes.len(),
        expected,
        "expected {expected} surviving detections; got {}",
        output_boxes.len()
    );
}

#[test]
fn decode_proto_with_empty_vec_returns_detections() {
    let (decoder, owned, expected) = build_synthetic_segdet_decoder();
    let inputs: Vec<&TensorDyn> = owned.iter().collect();

    let mut output_boxes: Vec<DetectBox> = Vec::new();
    let proto = decoder
        .decode_proto(&inputs, &mut output_boxes)
        .expect("decode_proto must succeed with Vec::new() output");
    assert!(
        proto.is_some(),
        "segmentation schema should produce ProtoData"
    );
    assert!(
        !output_boxes.is_empty(),
        "decode_proto with capacity-0 output Vec dropped all detections (EDGEAI-1302); \
         expected {expected}, got {}",
        output_boxes.len()
    );
    assert_eq!(
        output_boxes.len(),
        expected,
        "expected {expected} surviving detections; got {}",
        output_boxes.len()
    );
}
