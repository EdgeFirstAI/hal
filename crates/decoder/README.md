# edgefirst-decoder

[![Crates.io](https://img.shields.io/crates/v/edgefirst-decoder.svg)](https://crates.io/crates/edgefirst-decoder)
[![Documentation](https://docs.rs/edgefirst-decoder/badge.svg)](https://docs.rs/edgefirst-decoder)
[![License](https://img.shields.io/crates/l/edgefirst-decoder.svg)](LICENSE)

**High-performance ML model output decoding for object detection and segmentation.**

This crate provides efficient post-processing for YOLO and ModelPack model outputs, supporting both floating-point and quantized inference results.

## Supported Models

| Family | Detection | Segmentation | Formats |
|--------|-----------|--------------|---------|
| **YOLO** | YOLOv5, v8, v11 | Instance seg | float32, int8, uint8 |
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

## License

Licensed under the Apache License, Version 2.0. See [LICENSE](https://github.com/EdgeFirstAI/hal/blob/main/LICENSE) for details.
