# edgefirst-image

[![Crates.io](https://img.shields.io/crates/v/edgefirst-image.svg)](https://crates.io/crates/edgefirst-image)
[![Documentation](https://docs.rs/edgefirst-image/badge.svg)](https://docs.rs/edgefirst-image)
[![License](https://img.shields.io/crates/l/edgefirst-image.svg)](LICENSE)

**High-performance image processing for edge AI inference pipelines.**

This crate provides hardware-accelerated image loading, format conversion, resizing, rotation, and cropping operations optimized for ML preprocessing workflows.

## Features

- **Multiple backends** — Automatic selection: OpenGL (GPU) → G2D (NXP i.MX) → CPU (fallback)
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
- `tracker` (optional) - Enable multi-object tracking support in `draw_masks_tracked()`. Requires `features = ["tracker"]` in your dependency declaration.

## Environment Variables

- `EDGEFIRST_DISABLE_G2D` - Disable G2D backend
- `EDGEFIRST_DISABLE_GL` - Disable OpenGL backend
- `EDGEFIRST_DISABLE_CPU` - Disable CPU backend
- `EDGEFIRST_FORCE_BACKEND` — Force a single backend: `cpu`, `g2d`, or `opengl`. Disables fallback chain.
- `EDGEFIRST_FORCE_TRANSFER` — Force GPU transfer method: `pbo` or `dmabuf`
- `EDGEFIRST_TENSOR_FORCE_MEM` — Set to `1` to force heap memory (disables DMA/SHM)
- `EDGEFIRST_OPENGL_RENDERSURFACE` — Set to `1` to use renderbuffer-backed EGLImages for DMA destinations. Required on i.MX 95 / Mali-G310 with Neutron NPU DMA-BUF destinations. Defaults to `0` (texture path).
- `EDGEFIRST_PROTO_COMPUTE` — Set to `1` to enable the experimental GLES 3.1 compute shader path for proto repack. Requires GLES 3.1 hardware support.

## Segmentation Mask Rendering

Three rendering pipelines for YOLO instance segmentation masks:

### MaskOverlay

`MaskOverlay` controls how segmentation masks are composited onto the destination image:

```rust,ignore
use edgefirst_image::MaskOverlay;

// Default: no background replacement, full opacity
let overlay = MaskOverlay::default();

// With a background image and 50% transparent masks
let overlay = MaskOverlay { background: Some(&bg_tensor), opacity: 0.5 };
```

Fields:
- `background: Option<&TensorDyn>` — Optional tensor to blit into `dst` before drawing masks. Must match `dst`'s shape. `None` keeps the existing `dst` content.
- `opacity: f32` — Scales mask alpha in the range `0.0` (invisible) to `1.0` (fully opaque, default).

### draw_masks()

Convenience method that decodes model outputs, runs NMS, and draws segmentation masks in a single call:

```rust,ignore
let boxes = processor.draw_masks(&decoder, &outputs, &mut frame, MaskOverlay::default())?;
```

### draw_masks_tracked()

Like `draw_masks()` but integrates a `Tracker` for maintaining object identities across frames. The tracker runs after NMS but before mask extraction. Requires the `tracker` feature flag.

```rust,ignore
#[cfg(feature = "tracker")]
let (boxes, tracks) = processor.draw_masks_tracked(
    &decoder,
    &mut tracker,
    timestamp_ns,
    &outputs,
    &mut frame,
    MaskOverlay::default(),
)?;
```

Returns `(Vec<DetectBox>, Vec<TrackInfo>)`.

### Fused GPU Proto Path (`draw_proto_masks`)

Computes `sigmoid(coefficients @ protos)` per-pixel in a fragment shader — no intermediate mask materialization. Preferred for real-time overlay.

```rust,ignore
let (detections, proto_data) = decoder.decode_quantized_proto(&outputs)?;
processor.draw_proto_masks(&mut frame, &detections, &proto_data)?;
```

### Hybrid CPU+GPU Path

CPU materializes binary masks (`materialize_segmentations()`), then OpenGL overlays them. Auto-selected when both CPU and GL backends are available.

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

## Zero-Copy External Buffer (Linux)

When integrating with an NPU delegate (e.g. VxDelegate) that owns its own
DMA-BUF buffers, use `import_image()` to render directly into the
delegate's buffer — eliminating the `memcpy` between HAL's buffer and the
delegate's buffer:

```rust,ignore
use edgefirst_tensor::PlaneDescriptor;

// UC1: Render into VxDelegate's DMA-BUF — zero copies
let pd = PlaneDescriptor::new(vx_fd.as_fd())?;  // dups fd — caller keeps ownership
let mut dst = processor.import_image(pd, None, 640, 640, PixelFormat::Rgb, DType::U8)?;
processor.convert(&src, &mut dst, Rotation::None, Flip::None, Crop::letterbox())?;
// dst's backing memory IS vx_fd — no memcpy needed
```

For the reverse direction (HAL allocates, consumer imports):

```rust,ignore
let hal_dst = processor.create_image(640, 640, PixelFormat::Rgb, DType::U8, None)?;
let fd = hal_dst.dmabuf_clone()?;  // Error if not DMA-backed
vxdelegate.register_buffer(fd)?;
```

**Performance tip:** When rotating through a pool of DMA-BUFs (e.g. 2-3
from VxDelegate), create the `TensorDyn` wrappers once at init and reuse
them across frames. This avoids EGL image cache misses (~100-300us each).

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
