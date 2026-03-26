# edgefirst-decoder

[![Crates.io](https://img.shields.io/crates/v/edgefirst-decoder.svg)](https://crates.io/crates/edgefirst-decoder)
[![Documentation](https://docs.rs/edgefirst-decoder/badge.svg)](https://docs.rs/edgefirst-decoder)
[![License](https://img.shields.io/crates/l/edgefirst-decoder.svg)](LICENSE)

**High-performance ML model output decoding for object detection and segmentation.**

This crate provides efficient post-processing for YOLO and ModelPack model outputs, supporting both floating-point and quantized inference results.

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

The decoder automatically selects the appropriate model type based on the config:

| Variant | Tensors | Description |
|---------|---------|-------------|
| `YoloDet` | 1 (detection) | Standard YOLO detection |
| `YoloSegDet` | 2 (detection + protos) | YOLO detection + segmentation |
| `YoloSplitDet` | 2 (boxes + scores) | Split-output detection |
| `YoloSplitSegDet` | 4 (boxes + scores + mask_coeff + protos) | Split-output segmentation |
| `YoloEndToEndDet` | 1 (detection) | End-to-end detection (post-NMS) |
| `YoloEndToEndSegDet` | 2 (detection + protos) | End-to-end segmentation |
| `YoloSplitEndToEndDet` | 3 (boxes + scores + classes) | Split end-to-end detection |
| `YoloSplitEndToEndSegDet` | 5 (boxes + scores + classes + mask_coeff + protos) | Split end-to-end segmentation |
| `ModelPackDet` | 2 (boxes + scores) | ModelPack detection |
| `ModelPackSegDet` | 3 (boxes + scores + segmentation) | ModelPack segmentation |
| `ModelPackDetSplit` | N (detection layers) | ModelPack split detection |
| `ModelPackSegDetSplit` | N+1 (detection layers + segmentation) | ModelPack split segmentation |
| `ModelPackSeg` | 1 (segmentation) | ModelPack semantic segmentation |

## License

Licensed under the Apache License, Version 2.0. See [LICENSE](https://github.com/EdgeFirstAI/hal/blob/main/LICENSE) for details.
