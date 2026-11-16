// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! `Decoder::decode()` (the inline mask-materialising entrypoint) returns
//! bbox coordinates that are bit-identical to `Decoder::decode_proto()`
//! for the same input: the post-NMS bbox is not replaced by the
//! proto-grid-snapped roi that `protobox` returns.

use edgefirst_decoder::{
    configs::{self, DecoderType, DimName, QuantTuple},
    ConfigOutput, DecoderBuilder, DetectBox,
};
use edgefirst_tensor::{Tensor, TensorDyn, TensorMapTrait, TensorMemory, TensorTrait};

/// Build a YOLOv8-seg-shaped synthetic decoder with 5 detections at
/// non-overlapping x positions whose post-NMS bboxes fall on off-grid
/// sub-pixel coordinates.
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

    let n_targets = 5usize;
    let target_start = 10usize;
    let mut det_data = vec![0.0f32; FEAT * N];
    let set = |d: &mut [f32], r: usize, c: usize, v: f32| d[r * N + c] = v;
    for t in 0..n_targets {
        let anchor = target_start + t;
        // Off-grid bbox: no component is a multiple of 1/160.
        let xc = 0.0625 * t as f32 + 0.123_456;
        let yc = 0.503_137; // off-grid (≠ k/160)
        let w = 0.041_257; // off-grid; also keeps xc±w/2 off-grid
        let h = 0.297_413;
        set(&mut det_data, 0, anchor, xc);
        set(&mut det_data, 1, anchor, yc);
        set(&mut det_data, 2, anchor, w);
        set(&mut det_data, 3, anchor, h);
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

    (decoder, vec![det_tensor, protos_tensor], n_targets)
}

#[test]
fn decode_returns_bit_identical_bboxes_to_decode_proto() {
    let (decoder, owned, expected) = build_synthetic_segdet_decoder();
    let inputs: Vec<&TensorDyn> = owned.iter().collect();

    let mut decode_boxes: Vec<DetectBox> = Vec::with_capacity(50);
    let mut decode_masks: Vec<edgefirst_decoder::Segmentation> = Vec::with_capacity(50);
    decoder
        .decode(&inputs, &mut decode_boxes, &mut decode_masks)
        .expect("decode must succeed");

    let mut proto_boxes: Vec<DetectBox> = Vec::with_capacity(50);
    decoder
        .decode_proto(&inputs, &mut proto_boxes)
        .expect("decode_proto must succeed");

    assert_eq!(
        decode_boxes.len(),
        expected,
        "decode produced {} boxes, expected {expected}",
        decode_boxes.len()
    );
    assert_eq!(
        proto_boxes.len(),
        expected,
        "decode_proto produced {} boxes, expected {expected}",
        proto_boxes.len()
    );

    for (i, (a, b)) in decode_boxes.iter().zip(proto_boxes.iter()).enumerate() {
        assert_eq!(
            a.label, b.label,
            "det[{i}] label mismatch: decode={} decode_proto={}",
            a.label, b.label
        );
        assert_eq!(
            a.score.to_bits(),
            b.score.to_bits(),
            "det[{i}] score mismatch: decode={} decode_proto={}",
            a.score,
            b.score
        );
        assert_eq!(
            a.bbox.xmin.to_bits(),
            b.bbox.xmin.to_bits(),
            "det[{i}] xmin: decode={} decode_proto={}",
            a.bbox.xmin,
            b.bbox.xmin,
        );
        assert_eq!(
            a.bbox.ymin.to_bits(),
            b.bbox.ymin.to_bits(),
            "det[{i}] ymin: decode={} decode_proto={}",
            a.bbox.ymin,
            b.bbox.ymin,
        );
        assert_eq!(
            a.bbox.xmax.to_bits(),
            b.bbox.xmax.to_bits(),
            "det[{i}] xmax: decode={} decode_proto={}",
            a.bbox.xmax,
            b.bbox.xmax,
        );
        assert_eq!(
            a.bbox.ymax.to_bits(),
            b.bbox.ymax.to_bits(),
            "det[{i}] ymax: decode={} decode_proto={}",
            a.bbox.ymax,
            b.bbox.ymax,
        );
    }
}

#[test]
fn decode_does_not_snap_bboxes_to_proto_grid() {
    let (decoder, owned, _expected) = build_synthetic_segdet_decoder();
    let inputs: Vec<&TensorDyn> = owned.iter().collect();

    let mut boxes: Vec<DetectBox> = Vec::with_capacity(50);
    let mut masks: Vec<edgefirst_decoder::Segmentation> = Vec::with_capacity(50);
    decoder
        .decode(&inputs, &mut boxes, &mut masks)
        .expect("decode must succeed");

    let proto_step: f32 = 1.0 / 160.0;
    let on_grid = |v: f32| (v / proto_step - (v / proto_step).round()).abs() < 1e-5;

    for (i, b) in boxes.iter().enumerate() {
        let any_coord_on_grid = on_grid(b.bbox.xmin)
            || on_grid(b.bbox.ymin)
            || on_grid(b.bbox.xmax)
            || on_grid(b.bbox.ymax);
        assert!(
            !any_coord_on_grid,
            "det[{i}] ({:?}) has at least one coord snapped to the proto \
             grid (1/160); decode() should not call protobox-style \
             quantization on the output bbox",
            b.bbox
        );
    }
}
