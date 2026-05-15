# edgefirst-decoder

[![Crates.io](https://img.shields.io/crates/v/edgefirst-decoder.svg)](https://crates.io/crates/edgefirst-decoder)
[![Documentation](https://docs.rs/edgefirst-decoder/badge.svg)](https://docs.rs/edgefirst-decoder)
[![License](https://img.shields.io/crates/l/edgefirst-decoder.svg)](LICENSE)

**High-performance ML model output decoding for object detection and segmentation.**

This crate provides efficient post-processing for YOLO and ModelPack model outputs, supporting both floating-point and quantized inference results.

## Role in edgefirst-hal

`edgefirst-decoder` sits between the inference engine and the image-rendering
side of the EdgeFirst HAL workspace:

- Depends on [`edgefirst-tensor`](https://github.com/EdgeFirstAI/hal/blob/main/crates/tensor/) for reading model output buffers.
- Optionally depends on [`edgefirst-tracker`](https://github.com/EdgeFirstAI/hal/blob/main/crates/tracker/) (feature `tracker`) for `decode_tracked()`.
- Consumed by [`edgefirst-image`](https://github.com/EdgeFirstAI/hal/blob/main/crates/image/) — its `DetectBox`, `Segmentation`, and proto-data outputs feed the `draw_decoded_masks` / `draw_proto_masks` / `draw_masks_tracked` rendering APIs.
- Re-exported from [`edgefirst-hal`](https://github.com/EdgeFirstAI/hal/blob/main/crates/hal/) as `edgefirst_hal::decoder`.
- Bridged to C via [`edgefirst-hal-capi`](https://github.com/EdgeFirstAI/hal/blob/main/crates/capi/) (cbindgen-generated C ABI).
- Bridged to Python via [`crates/python`](https://github.com/EdgeFirstAI/hal/blob/main/crates/python/) (PyO3 binding over the Rust umbrella crate; does not go through the C ABI).

## Supported Models

| Family | Detection | Segmentation | Formats |
|--------|-----------|--------------|---------|
| **YOLO** | YOLOv5, v8, v11, v26 | Instance seg | float32, int8, uint8 |
| **ModelPack** | SSD-style | Semantic seg | float32, int8, uint8 |

## Features

- **Quantized decoding** - Direct int8/uint8 processing without dequantization overhead
- **Configurable NMS** - Class-agnostic or class-aware non-maximum suppression
- **Batch processing** - Efficient handling of batched model outputs
- **Builder pattern** - Flexible configuration with sensible defaults

## Quick Start

```rust,ignore
use edgefirst_decoder::{DecoderBuilder, DetectBox, Segmentation};

// Build decoder from model config
let decoder = DecoderBuilder::new()
    .with_score_threshold(0.25)
    .with_iou_threshold(0.7)
    .with_config_json_str(model_config_json)
    .build()?;

// Decode quantized model output
let mut detections: Vec<DetectBox> = Vec::with_capacity(100);
let mut masks: Vec<Segmentation> = Vec::with_capacity(100);

decoder.decode_quantized(
    &[model_output.view().into()],
    &mut detections,
    &mut masks,
)?;

// Process results
for det in &detections {
    println!("Class {} at [{:.1}, {:.1}, {:.1}, {:.1}] score={:.2}",
        det.label, det.bbox.xmin, det.bbox.ymin, det.bbox.xmax, det.bbox.ymax, det.score);
}
```

## Low-Level API

For known model types, use the direct decoding functions:

```rust,ignore
use edgefirst_decoder::yolo::decode_yolo_det;
use edgefirst_decoder::Quantization;

let mut detections = Vec::with_capacity(100);
decode_yolo_det(
    (output_array.view(), Quantization::new(0.012345, 26)),
    0.25,  // score threshold
    0.7,   // IOU threshold
    Some(edgefirst_decoder::configs::Nms::ClassAgnostic),
    &mut detections,
);
```

## Configuration

Decoders can be configured via JSON/YAML matching the model's output specification:

```json
{
  "decoder": "ultralytics",
  "shape": [1, 84, 8400],
  "quantization": [0.012345, 26],
  "normalized": true
}
```

## NMS Modes

- `ClassAgnostic` - Suppress overlapping boxes regardless of class (default)
- `ClassAware` - Only suppress boxes with the same class label
- `None` - Bypass NMS (for models with built-in NMS)

## Pre-NMS Top-K: Validation vs Deployment

The decoder's `pre_nms_top_k` parameter caps how many score-passing candidates
enter NMS, bounding its O(N²) cost via an O(N) partial sort. The default of
**300** is tuned for deployment — but it **must** be raised (or set to `0` for
no limit) for mAP evaluation.

### Why it matters

| Scenario | `score_threshold` | Anchors passing filter | Effect of `pre_nms_top_k = 300` |
|----------|------------------:|-----------------------:|--------------------------------|
| Deployment | ≥ 0.25 | Tens | No effect — fewer candidates than the cap |
| COCO mAP eval | 0.001 | Thousands | **Discards ~74 % of valid candidates before NMS** |

With COCO's low threshold, most of the 8 400 YOLO anchors pass the score
filter. The default top-K of 300 silently truncates the candidate pool,
causing **~9 pp box mAP loss** — a measurement artifact, not a model quality
issue. The decoder math is correct in both cases.

### Recommended settings

```rust,ignore
// Deployment (real-time inference)
let decoder = DecoderBuilder::new()
    .with_config_json_str(config)
    .with_score_threshold(0.25)
    // pre_nms_top_k = 300 (default) — appropriate
    .build()?;

// COCO mAP evaluation
let decoder = DecoderBuilder::new()
    .with_config_json_str(config)
    .with_score_threshold(0.001)
    .with_pre_nms_top_k(8400)   // pass all anchors to NMS (or 0 = no limit)
    .with_max_det(300)           // COCO detection cap applied post-NMS
    .build()?;
```

### Performance trade-off

Post-processing latency scales with the number of candidates entering NMS.
At deployment thresholds (`≥ 0.25`), the candidate count is already small
regardless of the top-K setting, so raising it has negligible cost. At
validation thresholds (`0.001`), the increase is measurable — but necessary
for correct recall across the full precision-recall curve.

## End-to-End Models (YOLO26)

YOLO26 models embed NMS directly in the model architecture (one-to-one matching heads), eliminating the need for external NMS post-processing.

Configure via the `decoder_version` field in the model config:

```json
{
  "decoder": "ultralytics",
  "decoder_version": "yolo26",
  "shape": [1, 300, 6],
  "quantization": [0.012345, 26],
  "normalized": true
}
```

When `decoder_version` is `"yolo26"`, the decoder:
- Bypasses NMS entirely (the `nms` config field is ignored)
- Expects post-NMS output format: `[batch, N, 6+]` where columns are `[x1, y1, x2, y2, conf, class, ...]`
- Supports both detection-only and detection+segmentation variants

For non-end-to-end YOLO26 exports (`end2end=false`), use `decoder_version: "yolov8"` with explicit NMS configuration.

### Non-End-to-End Mode

Models exported with `end2end=false` require external NMS, configurable via the `nms` field:

```json
{
  "decoder": "ultralytics",
  "decoder_version": "yolov8",
  "nms": "class_agnostic",
  "shape": [1, 84, 8400]
}
```

## Proto Mask API

For segmentation models, the decoder provides two APIs for accessing mask prototype data:

- `decode_quantized_proto()` — returns raw quantized proto data and mask coefficients without materializing pixel masks
- `decode_float_proto()` — returns float proto data and mask coefficients

These are preferred when passing mask data to GPU rendering pipelines (e.g., `ImageProcessor::draw_proto_masks()`), as they avoid the CPU cost of materializing full-resolution masks.

```rust,ignore
// GPU rendering path: decode proto data, pass to GL for fused rendering
let (detections, proto_data) = decoder.decode_quantized_proto(
    &[model_output.view().into()],
)?;

// Pass proto_data directly to GPU for fused mask overlay
processor.draw_proto_masks(&mut frame, &detections, &proto_data)?;
```

## Model Type Variants

The decoder automatically selects the appropriate model type from the
output schema, supporting YOLO (detection / segmentation, with or
without end-to-end NMS and split outputs) and ModelPack (detection,
segmentation, split variants). The full variant matrix, output-shape
disambiguation rules, the 2-way split format used by TFLite INT8
segmentation models, and the `nc=28` edge case are documented in
[ARCHITECTURE.md § Model-type selection](https://github.com/EdgeFirstAI/hal/blob/main/crates/decoder/ARCHITECTURE.md#model-type-selection).

## Tracked Decoding

The `tracker` feature adds `decode_tracked` to integrate object tracking directly into the decode step. Each decoded detection box is matched to a persistent track and assigned a stable UUID for the lifetime of the track.

Enable the feature in `Cargo.toml`:

```toml
edgefirst-decoder = { version = "0.13", features = ["tracker"] }
```

### Usage

```rust,ignore
use edgefirst_decoder::{DecoderBuilder, DetectBox, Segmentation, TrackInfo};
use edgefirst_decoder::Tracker; // re-exported from edgefirst-tracker
use edgefirst_tracker::ByteTrackBuilder;

let decoder = DecoderBuilder::new()
    .with_score_threshold(0.25)
    .with_iou_threshold(0.7)
    .with_config_json_str(model_config_json)
    .build()?;

let mut tracker = ByteTrackBuilder::new()
    .track_high_conf(0.5)
    .track_update(0.1)
    .build();

let mut detections: Vec<DetectBox> = Vec::with_capacity(100);
let mut masks: Vec<Segmentation> = Vec::with_capacity(100);
let mut tracks: Vec<TrackInfo> = Vec::with_capacity(100);

decoder.decode_tracked(
    &mut tracker,
    timestamp,          // u64 frame timestamp
    &model_outputs,     // &[&TensorDyn]
    &mut detections,
    &mut masks,
    &mut tracks,
)?;

// detections[i] and tracks[i] correspond to the same object
for (det, track) in detections.iter().zip(tracks.iter()) {
    println!(
        "Track {} class {} score={:.2} at [{:.1}, {:.1}, {:.1}, {:.1}]",
        track.uuid, det.label, det.score,
        det.bbox.xmin, det.bbox.ymin, det.bbox.xmax, det.bbox.ymax,
    );
}
```

### TrackInfo Fields

| Field | Type | Description |
|-------|------|-------------|
| `uuid` | `Uuid` | Stable unique identifier for the track |
| `tracked_location` | `[f32; 4]` | Kalman-smoothed location in XYXY format |
| `count` | `i32` | Number of times the track has been updated |
| `created` | `u64` | Timestamp when the track was first created |
| `last_updated` | `u64` | Timestamp of the most recent update |

The `tracked_location` reflects the Kalman-filter prediction and may differ slightly from `det.bbox`, which is the raw decoded box before smoothing.

## Documentation

- Architecture overview: [ARCHITECTURE.md](https://github.com/EdgeFirstAI/hal/blob/main/crates/decoder/ARCHITECTURE.md)
- Testing guide: [TESTING.md](https://github.com/EdgeFirstAI/hal/blob/main/crates/decoder/TESTING.md)
- Full API reference: [docs.rs/edgefirst-decoder](https://docs.rs/edgefirst-decoder)
- Project README: [../../README.md](https://github.com/EdgeFirstAI/hal/blob/main/README.md)

## License

Licensed under the Apache License, Version 2.0. See [LICENSE](https://github.com/EdgeFirstAI/hal/blob/main/LICENSE) for details.
