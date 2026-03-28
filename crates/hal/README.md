# edgefirst-hal

[![Crates.io](https://img.shields.io/crates/v/edgefirst-hal.svg)](https://crates.io/crates/edgefirst-hal)
[![Documentation](https://docs.rs/edgefirst-hal/badge.svg)](https://docs.rs/edgefirst-hal)
[![License](https://img.shields.io/crates/l/edgefirst-hal.svg)](LICENSE)

**EdgeFirst Hardware Abstraction Layer** — a unified Rust library for edge AI inference pipelines.

This is the umbrella crate that re-exports the core EdgeFirst HAL components:

- [`edgefirst-tensor`](https://crates.io/crates/edgefirst-tensor) — Zero-copy tensor memory management (DMA, SHM, PBO, system memory)
- [`edgefirst-image`](https://crates.io/crates/edgefirst-image) — Hardware-accelerated image processing and format conversion
- [`edgefirst-decoder`](https://crates.io/crates/edgefirst-decoder) — ML model output decoding (YOLOv5/v8/v11/v26, ModelPack)
- [`edgefirst-tracker`](https://crates.io/crates/edgefirst-tracker) — Multi-object tracking (ByteTrack)

## Features

- **Zero-copy memory management** with DMA-BUF, POSIX shared memory, and PBO support
- **Hardware-accelerated image processing** via OpenGL, G2D (NXP i.MX), and optimized CPU
- **Efficient ML post-processing** for object detection and segmentation models
- **Int8 GPU shaders** for direct signed int8 output without CPU post-processing
- **Cross-platform** — Linux (with hardware acceleration), macOS, and other Unix systems

## Quick Start

```rust,ignore
use edgefirst_image::{load_image, ImageProcessor, ImageProcessorTrait, Rotation, Flip, Crop};
use edgefirst_tensor::{PixelFormat, DType};

// Load a source image
let bytes = std::fs::read("image.jpg")?;
let input = load_image(&bytes, Some(PixelFormat::Rgb), None)?;

// Create an image processor (auto-selects best backend)
let mut processor = ImageProcessor::new()?;

// Allocate a GPU-optimal output buffer — always use create_image() for
// destinations passed to convert(). This selects the best memory type
// (DMA-buf, PBO, or system memory) for zero-copy GPU paths.
let mut output = processor.create_image(640, 640, PixelFormat::Rgb, DType::U8, None)?;

// Convert with letterbox resize
processor.convert(&input, &mut output, Rotation::None, Flip::None, Crop::default())?;
```

> **Why `create_image()`?** Creating tensors directly with `Tensor::new()` or
> `TensorDyn::image()` bypasses GPU memory negotiation. The processor cannot
> allocate PBO-backed buffers without knowing the GL context. Use `create_image()`
> for any tensor that will be passed to `convert()`.

## Platform Support

| Platform | Memory Types | Image Acceleration |
|----------|--------------|-------------------|
| Linux (NXP i.MX8/i.MX95) | DMA, SHM, PBO, Mem | OpenGL, G2D, CPU |
| Linux (other) | SHM, PBO, Mem | OpenGL, CPU |
| macOS | Mem | CPU |
| Other Unix | SHM, Mem | CPU |

## Feature Flags

The following Cargo feature flags are available for `edgefirst-hal`:

- `ndarray` (default) — Enable ndarray integration in the tensor crate. Allows converting tensors to/from `ndarray::Array`.
- `opengl` (default) — Enable the OpenGL backend for hardware-accelerated image processing on Linux.
- `tracker` (optional, not default) — Enable multi-object tracking support via ByteTrack. Enables `draw_masks_tracked()` in the image crate and `decode_tracked()` in the decoder crate. Requires explicit opt-in:

  ```toml
  [dependencies]
  edgefirst-hal = { version = "...", features = ["tracker"] }
  ```

## Python Bindings

This library is also available as a Python package:

```bash
pip install edgefirst-hal
```

See [`edgefirst-hal` on PyPI](https://pypi.org/project/edgefirst-hal/) for
Python-specific documentation.

## License

Licensed under the Apache License, Version 2.0. See [LICENSE](https://github.com/EdgeFirstAI/hal/blob/main/LICENSE) for details.

## Links

- [EdgeFirst AI](https://edgefirst.ai)
- [GitHub Repository](https://github.com/EdgeFirstAI/hal)
- [API Documentation](https://docs.rs/edgefirst-hal)
- [Python Package](https://pypi.org/project/edgefirst-hal/)
