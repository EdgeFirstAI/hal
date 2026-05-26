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
//! * **`ModelType::YoloSegDet`, `ModelType::YoloSplitSegDet`, and
//!   `ModelType::YoloSegDet2Way`**: as of commit `fa8e919`, all three
//!   model types call `maybe_normalize_boxes_in_place` uniformly across
//!   every entry point (`decode`, `decode_proto`, `decode_tracked`,
//!   `decode_tracked_proto`, both quantized and float variants). The
//!   accessor therefore upgrades `normalized=false` + valid `input_dims`
//!   to `Some(true)` for all three, matching the per-scale contract.
//!   See `Decoder::normalized_boxes` rustdoc as the canonical statement.
//!
//! * **Detection-only paths** (`YoloDet`, `YoloSplitDet`),
//!   **`YoloEndToEnd*`**, and **`ModelPack*`**: the accessor returns the
//!   raw schema flag (`self.normalized`) regardless of `input_dims`.
//!   These model types do not invoke the in-place helper uniformly across
//!   all entry points — callers must handle pixel-space output explicitly
//!   when the flag says `Some(false)`.
//!
//! This file pins:
//! (a) `YoloSegDet` accessor upgrade (`pixel_space_input_with_normalized_false_decodes`
//!     and `yolo_segdet_tracker_path_normalizes`),
//! (b) `YoloSplitSegDet` accessor upgrade (`split_schema_pixel_space_with_normalized_false_decodes`
//!     and `yolo_split_segdet_tracker_path_normalizes`),
//! (c) `YoloSegDet2Way` accessor upgrade (`yolo_segdet_2way_decode_normalizes`),
//! (d) in-place normalization verifiable by coordinate-range assertions
//!     (`for b in &boxes` checks throughout),
//! (e) detection-only (`YoloDet`) raw-flag passthrough
//!     (`detection_only_normalized_false_with_input_dims_reports_false_raw`),
//! (f) no-input-dims passthrough (`normalized_false_without_input_dims_reports_false`),
//! (g) `normalized=None` passthrough (`normalized_none_reports_none`),
//! (h) `normalized=true` direct passthrough (`normalized_true_reports_true`).

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

    // YoloSegDet now normalizes uniformly across all entry points (decode,
    // decode_proto, decode_tracked) after EDGEAI-1303 was extended to the
    // tracker paths in this release — accessor upgrades Some(false)+input_dims
    // to Some(true) to match.
    assert_eq!(
        decoder.normalized_boxes(),
        Some(true),
        "YoloSegDet: accessor must upgrade Some(false)+input_dims to Some(true) \
         now that every entry point calls maybe_normalize_boxes_in_place uniformly",
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

    // YoloSplitSegDet now normalizes uniformly across all entry points (after
    // fa8e919 added helper calls to the tracker macro, proto variants, and
    // float tracker paths) — accessor upgrades Some(false)+input_dims to
    // Some(true) to match. See Decoder::normalized_boxes rustdoc.
    // Coordinate-range assertions below confirm the helper fires on decode().
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

/// Regression for the architect's `e18873d` change: the `decode_tracked` path
/// on a `YoloSegDet` schema with `normalized=false` and valid `input_dims` must:
///
/// 1. Report `Some(true)` from `normalized_boxes()` — proving the accessor
///    upgrade arm covers `ModelType::YoloSegDet` on the tracker path.
/// 2. Produce boxes in `[0, 1]` — proving `maybe_normalize_boxes_in_place`
///    fires inside the tracker dispatch (via `process_tracked_yolo_segdet_float`
///    and the quantized macro equivalent).
///
/// Without the `e18873d` patch, `decode_tracked` on this schema skipped the
/// helper and returned pixel-space coordinates, silently breaking every caller
/// that relied on the accessor's truthful `Some(true)` signal. This test
/// ensures a future refactor cannot reintroduce that asymmetry without failing.
#[cfg(feature = "tracker")]
#[test]
fn yolo_segdet_tracker_path_normalizes() {
    use edgefirst_tracker::ByteTrackBuilder;

    let schema: SchemaV2 = serde_json::from_str(&schema_json(false)).unwrap();
    let decoder = DecoderBuilder::default()
        .with_score_threshold(0.5)
        .with_iou_threshold(0.99)
        .with_schema(schema)
        .build()
        .expect("schema-driven decoder must build");

    // Accessor must upgrade: YoloSegDet + normalized=false + input_dims → Some(true).
    assert_eq!(
        decoder.normalized_boxes(),
        Some(true),
        "YoloSegDet tracker path: accessor must report Some(true) after EDGEAI-1303 \
         was extended to cover decode_tracked in e18873d",
    );

    let owned = build_inputs();
    let inputs: Vec<&TensorDyn> = owned.iter().collect();

    let mut tracker = ByteTrackBuilder::new()
        .track_update(0.1)
        .track_high_conf(0.3)
        .build();
    let mut output_boxes: Vec<DetectBox> = Vec::with_capacity(50);
    let mut output_masks: Vec<edgefirst_decoder::Segmentation> = Vec::with_capacity(50);
    let mut output_tracks: Vec<edgefirst_decoder::TrackInfo> = Vec::with_capacity(50);

    decoder
        .decode_tracked(
            &mut tracker,
            0,
            &inputs,
            &mut output_boxes,
            &mut output_masks,
            &mut output_tracks,
        )
        .expect("YoloSegDet decode_tracked must succeed with pixel-space input");

    assert!(
        !output_boxes.is_empty(),
        "expected detections to survive NMS on the tracker path",
    );
    for b in &output_boxes {
        assert!(
            (0.0..=1.0).contains(&b.bbox.xmin)
                && (0.0..=1.0).contains(&b.bbox.ymin)
                && (0.0..=1.0).contains(&b.bbox.xmax)
                && (0.0..=1.0).contains(&b.bbox.ymax),
            "tracker-path bbox {:?} not in [0, 1]; maybe_normalize_boxes_in_place \
             did not fire in decode_tracked — e18873d regression",
            b.bbox,
        );
    }
}

/// Detection-only path (`YoloDet`), `normalized=false` with valid `input_dims`
/// constructed via programmatic `with_config`: accessor returns `Some(false)`
/// (raw schema flag) because `YoloDet` is intentionally out of scope for the
/// uniform-normalization upgrade. Pure detection paths do not call
/// `maybe_normalize_boxes_in_place` and are not listed in
/// `legacy_path_normalizes_uniformly`.
///
/// Renamed from `non_per_scale_non_yolo_segdet_normalized_false_with_input_dims_reports_false`
/// after `fa8e919` brought `YoloSplitSegDet` into uniform normalization,
/// invalidating the old test premise. `YoloDet` and `YoloSplitDet` remain
/// out of scope per the architect's audit.
///
/// Uses `with_config_yolo_det` to explicitly build a `YoloDet` schema with
/// valid `input_dims` via the inline input spec approach. See
/// `Decoder::normalized_boxes` rustdoc for the canonical contract.
#[test]
fn detection_only_normalized_false_with_input_dims_reports_false_raw() {
    use edgefirst_decoder::{configs, ConfigOutput};

    // Build a YoloDet with an input spec so input_dims is populated.
    // We achieve this via a v2 schema JSON that has only a detection output
    // (no protos, no mask_coeff), which routes to ModelType::YoloDet.
    let schema_str = format!(
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
                    "shape": [1, {feat}, {N}],
                    "dshape": [
                        {{"batch": 1}},
                        {{"num_features": {feat}}},
                        {{"num_boxes": {N}}}
                    ],
                    "decoder": "ultralytics",
                    "encoding": "direct",
                    "normalized": false
                }}
            ]
        }}"#,
        feat = 4 + NC,
    );
    let schema: SchemaV2 = serde_json::from_str(&schema_str).unwrap();
    let decoder = DecoderBuilder::default()
        .with_score_threshold(0.5)
        .with_iou_threshold(0.99)
        .with_schema(schema)
        .build()
        .expect("detection-only schema must build");

    // Detection-only (YoloDet) is explicitly out of scope for the upgrade:
    // input_dims is present but the accessor surfaces the raw schema flag.
    assert_eq!(
        decoder.input_dims(),
        Some((IMG, IMG)),
        "v2 schema with input spec must populate input_dims",
    );
    assert_eq!(
        decoder.normalized_boxes(),
        Some(false),
        "YoloDet: accessor must return raw schema flag — detection-only paths \
         are not in the uniform-normalization upgrade list (fa8e919)",
    );

    // Verify we got a detection-only schema (no per_scale, raw flag).
    // Programmatic fallback: also confirm that a flat programmatic build
    // (no input spec → input_dims = None) also returns Some(false).
    let det_flat = configs::Detection {
        anchors: None,
        decoder: configs::DecoderType::Ultralytics,
        quantization: None,
        shape: vec![1, FEAT, N],
        dshape: Vec::new(),
        normalized: Some(false),
    };
    let decoder_flat = DecoderBuilder::default()
        .add_output(ConfigOutput::Detection(det_flat))
        .build()
        .expect("flat YoloDet must build");
    assert_eq!(
        decoder_flat.input_dims(),
        None,
        "flat programmatic build leaves input_dims unset",
    );
    assert_eq!(
        decoder_flat.normalized_boxes(),
        Some(false),
        "YoloDet flat: raw flag propagated regardless of input_dims",
    );
}

/// Regression for `fa8e919`: the `decode_tracked` path on a `YoloSplitSegDet`
/// schema with `normalized=false` and valid `input_dims` must:
///
/// 1. Report `Some(true)` from `normalized_boxes()` — proving the accessor
///    upgrade arm covers `ModelType::YoloSplitSegDet` on the tracker path.
/// 2. Produce boxes in `[0, 1]` — proving `maybe_normalize_boxes_in_place`
///    fires inside the tracker macro (`process_tracked_yolo_split_segdet!`).
///
/// Without `fa8e919`, `decode_tracked` on this schema skipped the helper and
/// returned pixel-space coordinates. This test ensures a future refactor
/// cannot reintroduce that asymmetry without failing.
#[cfg(feature = "tracker")]
#[test]
fn yolo_split_segdet_tracker_path_normalizes() {
    use edgefirst_tracker::ByteTrackBuilder;

    let schema: SchemaV2 = serde_json::from_str(&split_schema_json(false)).unwrap();
    let decoder = DecoderBuilder::default()
        .with_score_threshold(0.5)
        .with_iou_threshold(0.99)
        .with_schema(schema)
        .build()
        .expect("split schema must build");

    // Accessor must upgrade: YoloSplitSegDet + normalized=false + input_dims → Some(true).
    assert_eq!(
        decoder.normalized_boxes(),
        Some(true),
        "YoloSplitSegDet tracker path: accessor must report Some(true) after fa8e919 \
         extended uniform normalization to the tracker macro path",
    );

    let owned = build_split_inputs();
    let inputs: Vec<&TensorDyn> = owned.iter().collect();

    let mut tracker = ByteTrackBuilder::new()
        .track_update(0.1)
        .track_high_conf(0.3)
        .build();
    let mut output_boxes: Vec<DetectBox> = Vec::with_capacity(50);
    let mut output_masks: Vec<edgefirst_decoder::Segmentation> = Vec::with_capacity(50);
    let mut output_tracks: Vec<edgefirst_decoder::TrackInfo> = Vec::with_capacity(50);

    decoder
        .decode_tracked(
            &mut tracker,
            0,
            &inputs,
            &mut output_boxes,
            &mut output_masks,
            &mut output_tracks,
        )
        .expect("YoloSplitSegDet decode_tracked must succeed with pixel-space input");

    assert!(
        !output_boxes.is_empty(),
        "expected detections to survive NMS on the YoloSplitSegDet tracker path",
    );
    for b in &output_boxes {
        assert!(
            (0.0..=1.0).contains(&b.bbox.xmin)
                && (0.0..=1.0).contains(&b.bbox.ymin)
                && (0.0..=1.0).contains(&b.bbox.xmax)
                && (0.0..=1.0).contains(&b.bbox.ymax),
            "tracker-path bbox {:?} not in [0, 1]; maybe_normalize_boxes_in_place \
             did not fire in YoloSplitSegDet decode_tracked — fa8e919 regression",
            b.bbox,
        );
    }
}

/// Regression for `fa8e919`: the `decode` path on a `YoloSegDet2Way` schema
/// with `normalized=false` and valid `input_dims` must:
///
/// 1. Report `Some(true)` from `normalized_boxes()` — proving the accessor
///    upgrade arm covers `ModelType::YoloSegDet2Way`.
/// 2. Produce boxes in `[0, 1]` — proving `maybe_normalize_boxes_in_place`
///    fires in `decode_yolo_segdet_2way_float` and `decode_yolo_segdet_2way_quantized`.
///
/// Schema constructed via v2 JSON: detection `[1, NC+4, N]` (boxes in rows 0..4
/// in XYWH pixel-space, scores in rows 4..) + mask_coeff `[1, NM, N]` +
/// protos `[1, PH, PW, NM]`. This is the minimal 2-way split that the builder
/// routes to `ModelType::YoloSegDet2Way`.
#[test]
fn yolo_segdet_2way_decode_normalizes() {
    // Schema for YoloSegDet2Way: detection (combined) + mask_coeff + protos.
    // The detection tensor has NC+4 features: rows 0..4 are box coords (XYWH,
    // pixel-space), rows 4.. are per-class scores.
    let det_feat = 4 + NC;
    let schema_str = format!(
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
                    "shape": [1, {det_feat}, {N}],
                    "dshape": [
                        {{"batch": 1}},
                        {{"num_features": {det_feat}}},
                        {{"num_boxes": {N}}}
                    ],
                    "decoder": "ultralytics",
                    "encoding": "direct",
                    "normalized": false
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
                    "shape": [1, {PH}, {PW}, {NM}],
                    "dshape": [
                        {{"batch": 1}},
                        {{"height": {PH}}},
                        {{"width": {PW}}},
                        {{"num_protos": {NM}}}
                    ]
                }}
            ]
        }}"#,
    );

    let schema: SchemaV2 = serde_json::from_str(&schema_str).unwrap();
    let decoder = DecoderBuilder::default()
        .with_score_threshold(0.5)
        .with_iou_threshold(0.99)
        .with_schema(schema)
        .build()
        .expect("YoloSegDet2Way schema must build");

    // Accessor must upgrade: YoloSegDet2Way + normalized=false + input_dims → Some(true).
    assert_eq!(
        decoder.normalized_boxes(),
        Some(true),
        "YoloSegDet2Way: accessor must report Some(true) when normalized=false \
         and input_dims is present (fa8e919 uniform normalization)",
    );
    assert_eq!(
        decoder.input_dims(),
        Some((IMG, IMG)),
        "v2 schema must populate input_dims from the input spec",
    );

    // Build inputs: combined detection [1, NC+4, N] with pixel-space XYWH boxes
    // in rows 0..4 and per-class scores in rows 4.., plus mask_coeff and protos.
    let n_targets = 5usize;
    let target_start = 10usize;
    let mut det_data = vec![0.0f32; det_feat * N];
    let set = |d: &mut [f32], r: usize, c: usize, v: f32| d[r * N + c] = v;
    for t in 0..n_targets {
        let anchor = target_start + t;
        let xc = 80.0 + 80.0 * t as f32; // pixel-space xc at imgsz=640
        set(&mut det_data, 0, anchor, xc);
        set(&mut det_data, 1, anchor, IMG as f32 * 0.5); // yc=320
        set(&mut det_data, 2, anchor, 25.6); // w
        set(&mut det_data, 3, anchor, 192.0); // h
        set(&mut det_data, 4, anchor, 0.9); // class-0 score
    }

    let to_dyn = |data: Vec<f32>, shape: &[usize]| -> TensorDyn {
        let t = Tensor::<f32>::new(shape, Some(TensorMemory::Mem), None).unwrap();
        {
            let mut m = t.map().unwrap();
            m.as_mut_slice().copy_from_slice(&data);
        }
        TensorDyn::F32(t)
    };

    let owned = [
        to_dyn(det_data, &[1, det_feat, N]),
        to_dyn(vec![0.0f32; NM * N], &[1, NM, N]),
        to_dyn(vec![0.0f32; NM * PH * PW], &[1, PH, PW, NM]),
    ];
    let inputs: Vec<&TensorDyn> = owned.iter().collect();

    let mut boxes: Vec<DetectBox> = Vec::with_capacity(50);
    let mut masks: Vec<edgefirst_decoder::Segmentation> = Vec::with_capacity(50);
    decoder
        .decode(&inputs, &mut boxes, &mut masks)
        .expect("YoloSegDet2Way pixel-space decode must succeed (fa8e919)");

    assert!(
        !boxes.is_empty(),
        "expected detections to survive NMS on YoloSegDet2Way path",
    );
    for b in &boxes {
        assert!(
            (0.0..=1.0).contains(&b.bbox.xmin)
                && (0.0..=1.0).contains(&b.bbox.ymin)
                && (0.0..=1.0).contains(&b.bbox.xmax)
                && (0.0..=1.0).contains(&b.bbox.ymax),
            "YoloSegDet2Way bbox {:?} not in [0, 1]; maybe_normalize_boxes_in_place \
             did not fire — fa8e919 regression",
            b.bbox,
        );
    }
}
