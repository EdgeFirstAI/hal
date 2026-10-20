// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! Input-handling contract of the legacy decode path: model outputs are
//! mapped read-only, and a padded row pitch is honoured rather than read as
//! data.
//!
//! The static tensor backend tolerates a read-write map of a buffer declared
//! `CpuAccess::Read` but counts it in `unplanned_cpu_access_count()`; this
//! file runs as its own process, so the counter moves only when a test here
//! maps beyond a declaration. Only the read-only test fills a `Read` tensor,
//! and it snapshots the counter after doing so.

use edgefirst_decoder::{configs, ConfigOutput, DecoderBuilder, DetectBox, Nms};
use edgefirst_tensor::{
    unplanned_cpu_access_count, CpuAccess, PixelFormat, Region, Tensor, TensorDyn, TensorMapTrait,
    TensorMemory, TensorTrait,
};

const NC: usize = 2;
const FEAT: usize = 4 + NC;
const N: usize = 8;
const SCALE: f32 = 1.0 / 255.0;

/// `[FEAT, N]` u8 YOLO detection features (xywh normalized, then class
/// scores) at scale 1/255, zero point 0.
fn features() -> Vec<u8> {
    let mut f = vec![0u8; FEAT * N];
    for i in 0..N {
        f[i] = (20 + 25 * i) as u8;
        f[N + i] = 128;
        f[2 * N + i] = 20;
        f[3 * N + i] = 20;
        f[(4 + i % NC) * N + i] = if i % 3 == 0 { 40 } else { 230 };
    }
    f
}

fn detection_config(shape: Vec<usize>, dshape: Vec<(configs::DimName, usize)>) -> ConfigOutput {
    ConfigOutput::Detection(configs::Detection {
        decoder: configs::DecoderType::Ultralytics,
        quantization: Some(configs::QuantTuple(SCALE, 0)),
        shape,
        dshape,
        anchors: None,
        normalized: Some(true),
    })
}

fn decode(output: ConfigOutput, input: &TensorDyn) -> Vec<DetectBox> {
    let decoder = DecoderBuilder::default()
        .with_score_threshold(0.5)
        .with_iou_threshold(0.7)
        .with_nms(Some(Nms::ClassAgnostic))
        .add_output(output)
        .build()
        .expect("detection decoder builds");
    let mut boxes = Vec::with_capacity(N);
    decoder
        .decode(&[input], &mut boxes, &mut Vec::new())
        .expect("decode succeeds");
    boxes
}

fn packed_reference() -> Vec<DetectBox> {
    let packed = Tensor::<u8>::from_slice(&features(), &[1, FEAT, N]).expect("packed tensor");
    let boxes = decode(detection_config(vec![1, FEAT, N], vec![]), &packed.into());
    assert!(!boxes.is_empty(), "fixture must yield detections");
    boxes
}

fn assert_same_boxes(got: &[DetectBox], want: &[DetectBox]) {
    assert_eq!(got, want);
}

#[test]
fn read_only_declared_input_decodes_without_a_write_map() {
    let t = Tensor::<u8>::image(
        N,
        FEAT,
        PixelFormat::Grey,
        Some(TensorMemory::Mem),
        CpuAccess::Read,
    )
    .expect("Read-declared Grey image");
    {
        let mut m = t.map_write().expect("fill map");
        m.as_mut_slice()[..FEAT * N].copy_from_slice(&features());
    }
    let mut t = t;
    t.reshape(&[1, FEAT, N])
        .expect("reshape to detection layout");
    assert_eq!(t.cpu_access(), CpuAccess::Read);
    let input: TensorDyn = t.into();

    let before = unplanned_cpu_access_count();
    let boxes = decode(detection_config(vec![1, FEAT, N], vec![]), &input);
    assert_eq!(
        unplanned_cpu_access_count(),
        before,
        "decoding a CpuAccess::Read input must not map it for writing"
    );
    assert_same_boxes(&boxes, &packed_reference());
}

#[test]
fn padded_row_input_decodes_like_the_packed_input() {
    const PITCH: usize = N + 5;
    let parent = Tensor::<u8>::image(
        PITCH,
        FEAT,
        PixelFormat::Grey,
        Some(TensorMemory::Mem),
        CpuAccess::ReadWrite,
    )
    .expect("parent Grey image");
    {
        let feats = features();
        let mut m = parent.map().expect("parent map");
        let dst = m.as_mut_slice();
        dst.fill(0xFF);
        for (row, src) in feats.chunks_exact(N).enumerate() {
            dst[row * PITCH..row * PITCH + N].copy_from_slice(src);
        }
    }
    let view = parent
        .view(Region::new(0, 0, N, FEAT))
        .expect("region view");
    assert_eq!(view.shape(), &[FEAT, N, 1]);
    assert_eq!(view.row_stride(), Some(PITCH));
    let input: TensorDyn = view.into();

    use configs::DimName::{Batch, NumBoxes, NumFeatures};
    let boxes = decode(
        detection_config(
            vec![FEAT, N, 1],
            vec![(NumFeatures, FEAT), (NumBoxes, N), (Batch, 1)],
        ),
        &input,
    );
    assert_same_boxes(&boxes, &packed_reference());
}

#[test]
fn input_with_pitch_shorter_than_a_row_is_rejected() {
    let mut t = Tensor::<u8>::image(
        N,
        FEAT,
        PixelFormat::Grey,
        Some(TensorMemory::Mem),
        CpuAccess::ReadWrite,
    )
    .expect("Grey image");
    t.set_row_stride_unchecked(N - 1);
    let input: TensorDyn = t.into();

    use configs::DimName::{Batch, NumBoxes, NumFeatures};
    let decoder = DecoderBuilder::default()
        .with_score_threshold(0.5)
        .add_output(detection_config(
            vec![FEAT, N, 1],
            vec![(NumFeatures, FEAT), (NumBoxes, N), (Batch, 1)],
        ))
        .build()
        .expect("detection decoder builds");
    let err = decoder
        .decode(&[&input], &mut Vec::new(), &mut Vec::new())
        .expect_err("a pitch shorter than a row cannot address the data");
    assert!(
        matches!(err, edgefirst_decoder::DecoderError::InvalidShape(_)),
        "unexpected error {err:?}"
    );
}
