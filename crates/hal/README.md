# edgefirst-hal

[![Crates.io](https://img.shields.io/crates/v/edgefirst-hal.svg)](https://crates.io/crates/edgefirst-hal)
[![Documentation](https://docs.rs/edgefirst-hal/badge.svg)](https://docs.rs/edgefirst-hal)
[![License](https://img.shields.io/crates/l/edgefirst-hal.svg)](LICENSE)

**EdgeFirst Hardware Abstraction Layer** — a unified Rust library for edge AI inference pipelines.

This is the umbrella crate that re-exports the core EdgeFirst HAL components:

- [`edgefirst-tensor`](https://crates.io/crates/edgefirst-tensor) — Zero-copy tensor memory management (platform GPU buffer, SHM, PBO, system memory)
- [`edgefirst-codec`](https://crates.io/crates/edgefirst-codec) — JPEG/PNG decode into pre-allocated tensors
- [`edgefirst-image`](https://crates.io/crates/edgefirst-image) — Hardware-accelerated image processing and format conversion
- [`edgefirst-decoder`](https://crates.io/crates/edgefirst-decoder) — ML model output decoding (YOLOv5/v8/v11/v26, ModelPack)
- [`edgefirst-tracker`](https://crates.io/crates/edgefirst-tracker) — Multi-object tracking (ByteTrack)

`codec` and `decoder` sit at opposite ends of the pipeline: `codec` turns image
bytes into tensors, `decoder` turns model output tensors into detections.

## Features

- **Zero-copy memory management** with DMA-BUF, IOSurface, AHardwareBuffer, POSIX shared memory, and PBO support
- **Hardware-accelerated image processing** via OpenGL, G2D (NXP i.MX), and optimized CPU
- **Hardware JPEG decode** via V4L2 mem2mem on Linux SoCs and nvJPEG on CUDA GPUs, each falling back to the built-in CPU decoder
- **Efficient ML post-processing** for object detection and segmentation models
- **Int8 GPU shaders** for direct signed int8 output without CPU post-processing
- **Cross-platform** — Linux, macOS/iOS, and Android with hardware acceleration; other Unix on CPU

## Quick Start

Decode a JPEG into a tensor, then letterbox it into the shape a model wants.
Both buffers are allocated once, outside the loop.

```rust,ignore
use edgefirst_hal::codec::{ImageDecoder, ImageLoad};
use edgefirst_hal::image::{Crop, Flip, ImageProcessor, ImageProcessorTrait, Rotation};
use edgefirst_hal::tensor::{CpuAccess, DType, PixelFormat};

// Create an image processor (auto-selects the best backend).
let mut processor = ImageProcessor::new()?;

// Allocate both buffers with create_image() — see the note below. The source
// holds the codec's native NV12 and is CPU-written by the decoder; the
// destination is the RGB the model consumes.
let mut src =
    processor.create_image(1920, 1080, PixelFormat::Nv12, DType::U8, None, CpuAccess::Write)?;
let mut dst =
    processor.create_image(640, 640, PixelFormat::Rgb, DType::U8, None, CpuAccess::None)?;

let mut decoder = ImageDecoder::new();

// Hot loop: decode, then convert (colour + resize). The codec reports EXIF
// orientation in `info` but does not apply it — pass it to convert().
let bytes = std::fs::read("image.jpg")?;
let info = src.load_image(&mut decoder, &bytes)?;
let rotation = Rotation::from_degrees_clockwise(info.rotation_degrees as usize);
let flip = if info.flip_horizontal { Flip::Horizontal } else { Flip::None };
processor.convert(&src, &mut dst, rotation, flip, Crop::new())?;
```

> **Why `create_image()`?** Creating tensors directly with `Tensor::new()` or
> `TensorDyn::image()` bypasses GPU memory negotiation. The processor cannot
> allocate PBO-backed buffers without knowing the GL context. Use `create_image()`
> for any tensor that will be passed to `convert()`.

## Platform Support

| Platform | Memory Types | Image Acceleration |
|----------|--------------|-------------------|
| Linux (NXP i.MX8/i.MX95) | DMA-BUF, SHM, PBO, Mem | OpenGL, G2D, CPU |
| Linux (other) | DMA-BUF, SHM, PBO, Mem | OpenGL, CPU |
| macOS / iOS | IOSurface, SHM, Mem | OpenGL (ANGLE), CPU |
| Android | AHardwareBuffer, SHM, Mem | OpenGL, CPU |
| Other Unix | SHM, Mem | CPU |
| Windows | Mem | CPU |

DMA-BUF on Linux needs a mountable dma-heap and permission to use it; without
that the allocator falls back and everything still works, just with a copy.
`TensorMemory::Dma` names the platform's native GPU buffer on all three of
Linux, Apple, and Android, so portable code never branches on the mechanism.

## Feature Flags

The following Cargo feature flags are available for `edgefirst-hal`:

- `ndarray` (default) — Enable ndarray integration in the tensor crate. Allows converting tensors to/from `ndarray::Array`.
- `opengl` (default) — Enable the OpenGL backend for hardware-accelerated image processing. Compiled on Linux, macOS, iOS, and Android.
- `tracing` (default) — Enable the `edgefirst_hal::trace` module, which installs the process-wide subscriber that turns the sub-crates' spans into a Chrome/Perfetto trace file. Pulls in `tracing-subscriber` and `tracing-chrome`.
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

## Documentation

- Architecture overview: [ARCHITECTURE.md](https://github.com/EdgeFirstAI/hal/blob/main/crates/hal/ARCHITECTURE.md)
- Testing guide: [TESTING.md](https://github.com/EdgeFirstAI/hal/blob/main/crates/hal/TESTING.md)
- Full API reference: [docs.rs/edgefirst-hal](https://docs.rs/edgefirst-hal)
- Project README: [README.md](https://github.com/EdgeFirstAI/hal/blob/main/README.md)
- Python package: [pypi.org/project/edgefirst-hal](https://pypi.org/project/edgefirst-hal/)
- [EdgeFirst AI](https://edgefirst.ai)

## License

Licensed under the Apache License, Version 2.0. See [LICENSE](https://github.com/EdgeFirstAI/hal/blob/main/LICENSE) for details.
