# edgefirst-hal

[![Crates.io](https://img.shields.io/crates/v/edgefirst-hal.svg)](https://crates.io/crates/edgefirst-hal)
[![Documentation](https://docs.rs/edgefirst-hal/badge.svg)](https://docs.rs/edgefirst-hal)
[![License](https://img.shields.io/crates/l/edgefirst-hal.svg)](LICENSE)

**EdgeFirst Hardware Abstraction Layer** - A unified Rust library for edge AI inference pipelines.

This is the umbrella crate that re-exports the core EdgeFirst HAL components:

- [`edgefirst-tensor`](https://crates.io/crates/edgefirst-tensor) - Zero-copy tensor memory management (DMA, SHM, system memory)
- [`edgefirst-image`](https://crates.io/crates/edgefirst-image) - High-performance image processing and format conversion
- [`edgefirst-decoder`](https://crates.io/crates/edgefirst-decoder) - ML model output decoding (YOLO, ModelPack)

## Features

- **Zero-copy memory management** with DMA-BUF and POSIX shared memory support
- **Hardware-accelerated image processing** via G2D (NXP i.MX) and OpenGL
- **Efficient ML post-processing** for object detection and segmentation models
- **Cross-platform** - Linux (with hardware acceleration), macOS, and other Unix systems

## Quick Start

```rust,ignore
use edgefirst_hal::{tensor, image, decoder};

// Create a tensor with automatic memory selection
let tensor = tensor::Tensor::<f32>::new(&[1, 3, 640, 640], None, None)?;

// Load and process an image
let img = image::TensorImage::load(&image_bytes, Some(image::RGBA), None)?;
let mut processor = image::ImageProcessor::new()?;
processor.convert(&img, &mut dst, image::Rotation::None, image::Flip::None, image::Crop::default())?;

// Decode YOLO model outputs
let decoder = decoder::DecoderBuilder::new()
    .with_score_threshold(0.25)
    .with_iou_threshold(0.7)
    .build()?;
```

## Platform Support

| Platform | Memory Types | Image Acceleration |
|----------|--------------|-------------------|
| Linux (i.MX8) | DMA, SHM, Mem | G2D, OpenGL, CPU |
| Linux (other) | SHM, Mem | OpenGL, CPU |
| macOS | SHM, Mem | CPU |
| Other Unix | SHM, Mem | CPU |

## License

Licensed under the Apache License, Version 2.0. See [LICENSE](https://github.com/EdgeFirstAI/hal/blob/main/LICENSE) for details.

## Links

- [EdgeFirst AI](https://edgefirst.ai)
- [GitHub Repository](https://github.com/EdgeFirstAI/hal)
- [Documentation](https://docs.rs/edgefirst-hal)
