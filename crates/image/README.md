# edgefirst-image

[![Crates.io](https://img.shields.io/crates/v/edgefirst-image.svg)](https://crates.io/crates/edgefirst-image)
[![Documentation](https://docs.rs/edgefirst-image/badge.svg)](https://docs.rs/edgefirst-image)
[![License](https://img.shields.io/crates/l/edgefirst-image.svg)](LICENSE)

**High-performance image processing for edge AI inference pipelines.**

This crate provides hardware-accelerated image loading, format conversion, resizing, rotation, and cropping operations optimized for ML preprocessing workflows.

## Features

- **Multiple backends** — Automatic selection: G2D (if format supported) → CPU (same-size simple copies) → OpenGL (GPU) → CPU (general fallback)
- **Format conversion** - RGBA, RGB, NV12, NV16, YUYV, GREY, planar formats
- **Geometric transforms** - Resize, rotate (90° increments), flip, crop
- **Zero-copy integration** - Works with `edgefirst-tensor` DMA/SHM buffers
- **JPEG/PNG support** - Load and save with EXIF orientation handling

## Quick Start

```rust
use edgefirst_image::{load_image, save_jpeg, ImageProcessor, ImageProcessorTrait, Rotation, Flip, Crop};
use edgefirst_tensor::{PixelFormat, DType, TensorDyn};

// Load an image
let bytes = std::fs::read("input.jpg")?;
let src = load_image(&bytes, Some(PixelFormat::Rgba), None)?;

// Create processor (auto-selects best backend)
let mut processor = ImageProcessor::new()?;

// Create destination with desired size
let mut dst = processor.create_image(640, 640, PixelFormat::Rgba, DType::U8, None)?;

// Convert with resize, rotation, letterboxing
processor.convert(
    &src,
    &mut dst,
    Rotation::None,
    Flip::None,
    Crop::letterbox(),  // Preserve aspect ratio
)?;

// Save result
save_jpeg(&dst, "output.jpg", 90)?;
```

## Backends

| Backend | Platform | Hardware | Notes |
|---------|----------|----------|-------|
| G2D | Linux (i.MX8) | 2D GPU | Fastest for NXP platforms |
| OpenGL | Linux | GPU | EGL/GBM headless rendering |
| CPU | All | SIMD | Portable fallback |

## Supported Formats

| Format | Description | Channels |
|--------|-------------|----------|
| `PixelFormat::Rgba` | 32-bit RGBA | 4 |
| `PixelFormat::Rgb` | 24-bit RGB | 3 |
| `PixelFormat::Nv12` | YUV 4:2:0 semi-planar | 1.5 |
| `PixelFormat::Nv16` | YUV 4:2:2 semi-planar | 2 |
| `PixelFormat::Yuyv` | YUV 4:2:2 packed | 2 |
| `PixelFormat::Grey` | 8-bit grayscale | 1 |
| `PixelFormat::PlanarRgb` | Planar RGB | 3 |
| `PixelFormat::Vyuy` | YUV 4:2:2 packed (VYUY order) | 2 |
| `PixelFormat::Bgra` | 32-bit BGRA | 4 |
| `PixelFormat::PlanarRgba` | Planar RGBA | 4 |

Note: Int8 variants (e.g. packed RGB int8, planar RGB int8) use `DType::I8` with the corresponding `PixelFormat` rather than separate format constants.

## Feature Flags

- `opengl` (default) - Enable OpenGL backend on Linux
- `decoder` (default) - Enable detection box rendering

## Environment Variables

- `EDGEFIRST_DISABLE_G2D` - Disable G2D backend
- `EDGEFIRST_DISABLE_GL` - Disable OpenGL backend
- `EDGEFIRST_DISABLE_CPU` - Disable CPU backend
- `EDGEFIRST_FORCE_BACKEND` — Force a single backend: `cpu`, `g2d`, or `opengl`. Disables fallback chain.
- `EDGEFIRST_FORCE_TRANSFER` — Force GPU transfer method: `pbo` or `dmabuf`
- `EDGEFIRST_TENSOR_FORCE_MEM` — Set to `1` to force heap memory (disables DMA/SHM)

## Segmentation Mask Rendering

Three rendering pipelines for YOLO instance segmentation masks:

### Fused GPU Proto Path (`draw_masks_proto`)

Computes `sigmoid(coefficients @ protos)` per-pixel in a fragment shader — no intermediate mask materialization. Preferred for real-time overlay.

```rust,ignore
let (detections, proto_data) = decoder.decode_quantized_proto(&outputs)?;
processor.draw_masks_proto(&mut frame, &detections, &proto_data)?;
```

### Hybrid CPU+GPU Path

CPU materializes binary masks (`materialize_segmentations()`), then OpenGL overlays them. Auto-selected when both CPU and GL backends are available.

### Atlas Decode Path (`decode_masks_atlas`)

Renders all detection masks into a compact vertical strip atlas via GPU, reads back as uint8 arrays. Use when you need per-instance mask pixels for downstream processing.

```rust,ignore
let masks = processor.decode_masks_atlas(&detections, &proto_data, 640, 640)?;
```

### Shader Variants

| Variant | Proto Format | Interpolation |
|---------|-------------|---------------|
| int8-nearest | R8I quantized | Nearest neighbor |
| int8-bilinear | R8I quantized | Manual 4-tap bilinear |
| f32 | R32F float | Hardware GL_LINEAR |
| f16 | R16F half | Hardware GL_LINEAR |

### Int8 Interpolation Mode

Control quantized proto interpolation quality:

```rust,ignore
processor.set_int8_interpolation_mode(Int8InterpolationMode::Bilinear);
```

See [BENCHMARKS.md](../../BENCHMARKS.md) for per-platform performance numbers.

## Zero-Copy Model Input

Use `create_image()` to allocate the destination tensor with the processor's
optimal memory backend (DMA-buf, PBO, or system memory). This enables
zero-copy GPU paths that direct `Tensor::new()` allocation cannot achieve:

```rust,ignore
let mut dst = processor.create_image(640, 640, PixelFormat::Rgb, DType::U8, None)?;
processor.convert(&src, &mut dst, Rotation::None, Flip::None, Crop::letterbox())?;
```

If you need to write into a pre-allocated buffer with a specific memory type
(e.g. an NPU-bound tensor), you can still use direct allocation:

```rust,ignore
let mut model_input = Tensor::<u8>::new(&[640, 640, 3], None, None)?;
model_input.set_format(PixelFormat::Rgb)?;
let mut dst = TensorDyn::from(model_input);
processor.convert(&src, &mut dst, Rotation::None, Flip::None, Crop::letterbox())?;
```

## Multiplane NV12/NV16

For V4L2 multi-planar DMA-BUF buffers (separate Y and UV file descriptors):

```rust,ignore
let img = Tensor::from_planes(y_tensor, uv_tensor, PixelFormat::Nv12)?;
let src = TensorDyn::from(img);
processor.convert(&src, &mut dst, Rotation::None, Flip::None, Crop::default())?;
```

The OpenGL backend imports each plane's DMA-BUF fd separately for zero-copy GPU access.

## License

Licensed under the Apache License, Version 2.0. See [LICENSE](https://github.com/EdgeFirstAI/hal/blob/main/LICENSE) for details.
