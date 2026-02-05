# edgefirst-image

[![Crates.io](https://img.shields.io/crates/v/edgefirst-image.svg)](https://crates.io/crates/edgefirst-image)
[![Documentation](https://docs.rs/edgefirst-image/badge.svg)](https://docs.rs/edgefirst-image)
[![License](https://img.shields.io/crates/l/edgefirst-image.svg)](LICENSE)

**High-performance image processing for edge AI inference pipelines.**

This crate provides hardware-accelerated image loading, format conversion, resizing, rotation, and cropping operations optimized for ML preprocessing workflows.

## Features

- **Multiple backends** - Automatic selection of fastest available: G2D → OpenGL → CPU
- **Format conversion** - RGBA, RGB, NV12, NV16, YUYV, GREY, planar formats
- **Geometric transforms** - Resize, rotate (90° increments), flip, crop
- **Zero-copy integration** - Works with `edgefirst-tensor` DMA/SHM buffers
- **JPEG/PNG support** - Load and save with EXIF orientation handling

## Quick Start

```rust
use edgefirst_image::{TensorImage, ImageProcessor, Rotation, Flip, Crop, RGBA};

// Load an image
let bytes = std::fs::read("input.jpg")?;
let src = TensorImage::load(&bytes, Some(RGBA), None)?;

// Create processor (auto-selects best backend)
let mut processor = ImageProcessor::new()?;

// Create destination with desired size
let mut dst = TensorImage::new(640, 640, RGBA, None)?;

// Convert with resize, rotation, letterboxing
processor.convert(
    &src,
    &mut dst,
    Rotation::None,
    Flip::None,
    Crop::letterbox(),  // Preserve aspect ratio
)?;

// Save result
dst.save_jpeg("output.jpg", 90)?;
```

## Backends

| Backend | Platform | Hardware | Notes |
|---------|----------|----------|-------|
| G2D | Linux (i.MX8) | 2D GPU | Fastest for NXP platforms |
| OpenGL | Linux | GPU | EGL/GBM headless rendering |
| CPU | All | SIMD | Portable fallback |

## Supported Formats

| FourCC | Description | Channels |
|--------|-------------|----------|
| RGBA | 32-bit RGBA | 4 |
| RGB | 24-bit RGB | 3 |
| NV12 | YUV 4:2:0 semi-planar | 1.5 |
| NV16 | YUV 4:2:2 semi-planar | 2 |
| YUYV | YUV 4:2:2 packed | 2 |
| GREY | 8-bit grayscale | 1 |
| 8BPS | Planar RGB | 3 |

## Feature Flags

- `opengl` (default) - Enable OpenGL backend on Linux
- `decoder` (default) - Enable detection box rendering

## Environment Variables

- `EDGEFIRST_DISABLE_G2D` - Disable G2D backend
- `EDGEFIRST_DISABLE_GL` - Disable OpenGL backend
- `EDGEFIRST_DISABLE_CPU` - Disable CPU backend

## License

Licensed under the Apache License, Version 2.0. See [LICENSE](LICENSE) for details.
