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
- Bridged to C via [`edgefirst-decoder-capi`](https://github.com/EdgeFirstAI/hal/blob/main/crates/decoder-capi/) (`libedgefirst_decoder`, `edgefirst/decoder.h`).
- Bridged to Python via [`crates/python-decoder`](https://github.com/EdgeFirstAI/hal/blob/main/crates/python-decoder/).

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
use edgefirst_tensor::TensorDyn;

// Build the decoder once, from the model's config document.
let decoder = DecoderBuilder::new()
    .with_score_threshold(0.25)
    .with_iou_threshold(0.7)
    .with_config_json_str(model_config_json)   // String
    .build()?;

// Then decode once per inference frame. `decode` dispatches to the
// quantized or float kernel from the tensor dtype — there is no separate
// entry point per dtype.
let mut detections: Vec<DetectBox> = Vec::with_capacity(100);
let mut masks: Vec<Segmentation> = Vec::with_capacity(100);

let outputs: Vec<&TensorDyn> = model_outputs.iter().collect();
decoder.decode(&outputs, &mut detections, &mut masks)?;

// Process results
for det in &detections {
    println!("Class {} at [{:.1}, {:.1}, {:.1}, {:.1}] score={:.2}",
        det.label, det.bbox.xmin, det.bbox.ymin, det.bbox.xmax, det.bbox.ymax, det.score);
}
```

Both `Vec`s are cleared on entry, and their capacity is an allocation hint
rather than a cap — the detection count is bounded by `max_det` (default 300),
not by what you pre-allocated.

> **Note:** every model decode kernel in the `yolo` and `modelpack` modules is
> crate-private, so `Decoder` is the entire public decoding surface — there is
> no supported way to decode a model output without going through it. The
> `float` and `byte` modules do export reusable primitives (`nms_float`,
> `nms_class_aware_int`, `iou_value`, `ios_value`, `box_area`, and friends) if
> you need NMS or box geometry on your own candidates.

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

For segmentation models, `decode_proto()` returns the mask prototypes and
per-detection coefficients instead of materialized pixel masks. Prefer it when
the masks are headed for a GPU rendering pipeline (e.g.
`ImageProcessor::draw_proto_masks()`): the GPU evaluates
`sigmoid(coeffs @ protos)` per output pixel, so the CPU never pays for
full-resolution masks.

It returns `Ok(None)` for detection-only and ModelPack models, and
`Ok(Some(ProtoData))` for YOLO segmentation models. Like `decode`, it picks the
quantized or float kernel from the tensor dtype.

```rust,ignore
// GPU rendering path: decode proto data, pass to GL for fused rendering
let mut detections: Vec<DetectBox> = Vec::with_capacity(100);
let proto_data = decoder.decode_proto(&outputs, &mut detections)?;

// Detection-only models return None — there is nothing to overlay.
if let Some(proto_data) = proto_data {
    processor.draw_proto_masks(&mut frame, &detections, &proto_data)?;
}
```

## Model Type Variants

The decoder automatically selects the appropriate model type from the
output schema, supporting YOLO (detection / segmentation, with or
without end-to-end NMS and split outputs) and ModelPack (detection,
segmentation, split variants). The full variant matrix, output-shape
disambiguation rules, the 2-way split format used by TFLite INT8
segmentation models, and the `nc=28` edge case are documented in
[ARCHITECTURE.md § Model-type selection](https://github.com/EdgeFirstAI/hal/blob/main/crates/decoder/ARCHITECTURE.md#model-type-selection).

## Tiled Inference (SAHI)

When a frame is preprocessed into an overlapping tile grid (see
`edgefirst-image`'s tiled preprocessing), each tile is decoded independently and
its detections are lifted back to full-frame coordinates and merged. A single
object split across a tile seam has low IoU but high **IoS**
(intersection-over-smaller), so the default merge metric is `Ios` with a `0.5`
threshold via a GREEDYNMM pass — canonical SAHI postprocessing.

The input and output sides share one `TilePlacement` (produced by
`ImageProcessor::plan_tiles`/`tile_into`) describing how each tile was cut.

```rust,ignore
use edgefirst_decoder::{DecoderBuilder, DetectBox, Nms, Segmentation};
use edgefirst_decoder::tiling::{MergeConfig, TiledFrameAccumulator};

// The per-tile decoder is deliberately permissive: a fragment clipped at a
// seam scores low, and a high per-tile threshold would discard it before the
// merge could rebuild the object. Gate the final scores in MergeConfig instead.
let decoder = DecoderBuilder::new()
    .with_config_json_str(model_config_json)
    .with_score_threshold(0.05)
    .with_nms(Some(Nms::ClassAware))
    .build()?;

// One accumulator per in-flight frame; collect tiles in any order.
let mut acc = TiledFrameAccumulator::new(
    (frame_w as f32, frame_h as f32),
    placements.len(),   // tiles_total — the fan-in fence
    MergeConfig::default(),
    16,                 // estimated detections per tile (capacity hint)
);

for (tile_outputs, placement) in tiles {
    let mut boxes: Vec<DetectBox> = Vec::new();
    let mut masks: Vec<Segmentation> = Vec::new();
    decoder.decode(&tile_outputs, &mut boxes, &mut masks)?;
    // Lifts to full-frame pixels and appends. Returns false for a tile this
    // accumulator has already seen, which is how retries stay harmless.
    acc.push_tile(boxes, &placement);
}

// Once every tile has arrived, merge and normalize to [0,1] for the tracker.
assert!(acc.is_complete());
let detections = acc.finalize_normalized(); // full-frame, deduplicated
```

`push_tile` is idempotent per `placement.index`, so out-of-order **and**
at-least-once tile delivery both converge to the same result — the merge runs once
at `finalize`, never per push. This is the "collect after the final tile" contract
a pipelined runtime needs: `plan_tiles` sizes the ring up front, `tile_one` streams
tiles through inference, and `is_complete()`/`remaining()` fence the frame.

The free functions `lift_tile_boxes` and `merge_tiled_detections` expose the two
stages directly if you need to merge without the accumulator. `MergeConfig` tunes
the metric (`Ios`, default, or `Iou`), the match `threshold` (0.5),
`class_agnostic` (false), `max_det` (300), and a final `score_threshold`. That
last one defaults to `0.0` on purpose: per-tile decode is the real flood
control, and this is where you put the score gate once fragments have been
joined. The same machinery supports standard SAHI with Ultralytics YOLO models.
Merged boxes can be fed straight to the tracker (next section), which expects
normalized coordinates, hence `finalize_normalized`.

> **Note:** IoS merge reconstructs the *enclosing union* of fragments and cannot
> recover an object larger than a single tile; add an optional full-frame
> downscaled pass (another `push_tile` at `origin=(0,0)`, `crop_size=frame_dims`)
> for mixed-scale datasets.

## Tracked Decoding

The `tracker` feature adds `decode_tracked` to integrate object tracking directly into the decode step. Each decoded detection box is matched to a persistent track and assigned a stable UUID for the lifetime of the track.

Enable the feature in `Cargo.toml`:

```toml
edgefirst-decoder = { version = "0.28", features = ["tracker"] }
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

### What `decode_tracked` does to `detections`

`decode_tracked` does not simply append track metadata to the decoded boxes. It
rewrites `detections` to be the set of **active tracks**, which changes two
things worth planning around:

- Every `det.bbox` is overwritten with that track's Kalman-smoothed
  `tracked_location`. After `decode_tracked`, `det.bbox` and
  `track.tracked_location` hold the same coordinates; the raw pre-smoothing box
  is not returned. Use `decode` if you need the unsmoothed detections.
- Tracks still alive but unmatched this frame are included, so `detections` can
  contain entries that no detection in this frame produced. They carry a
  coasting Kalman prediction, and on the segmentation path they have no mask.
  `track.last_updated` (older than the current timestamp) identifies them.

`detections[i]` and `tracks[i]` describe the same track in both cases, so the
zip above stays valid.

## Documentation

- Architecture overview: [ARCHITECTURE.md](https://github.com/EdgeFirstAI/hal/blob/main/crates/decoder/ARCHITECTURE.md)
- Testing guide: [TESTING.md](https://github.com/EdgeFirstAI/hal/blob/main/crates/decoder/TESTING.md)
- Full API reference: [docs.rs/edgefirst-decoder](https://docs.rs/edgefirst-decoder)
- Project README: [../../README.md](https://github.com/EdgeFirstAI/hal/blob/main/README.md)

## License

Licensed under the Apache License, Version 2.0. See [LICENSE](https://github.com/EdgeFirstAI/hal/blob/main/LICENSE) for details.
