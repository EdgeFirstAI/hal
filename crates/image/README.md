# edgefirst-image

[![Crates.io](https://img.shields.io/crates/v/edgefirst-image.svg)](https://crates.io/crates/edgefirst-image)
[![Documentation](https://docs.rs/edgefirst-image/badge.svg)](https://docs.rs/edgefirst-image)
[![License](https://img.shields.io/crates/l/edgefirst-image.svg)](LICENSE)

**High-performance image processing for edge AI inference pipelines.**

This crate provides hardware-accelerated image loading, format conversion, resizing, rotation, and cropping operations optimized for ML preprocessing workflows.

## Role in edgefirst-hal

`edgefirst-image` sits at the centre of the EdgeFirst HAL workspace, owning
the GPU/G2D/CPU dispatch and segmentation-mask rendering. Its dependency
neighbours:

- Depends on [`edgefirst-tensor`](https://github.com/EdgeFirstAI/hal/blob/main/crates/tensor/) for `TensorDyn`, `BufferIdentity`, and the `PboOps` trait it implements for the GL backend.
- Depends unconditionally on [`edgefirst-decoder`](https://github.com/EdgeFirstAI/hal/blob/main/crates/decoder/) for `DetectBox`, `Segmentation`, and the proto-mask data feeding `draw_proto_masks` (there is no opt-out feature flag).
- Optionally depends on [`edgefirst-tracker`](https://github.com/EdgeFirstAI/hal/blob/main/crates/tracker/) (feature `tracker`) for `draw_masks_tracked`.
- Bridged to C via [`edgefirst-image-capi`](https://github.com/EdgeFirstAI/hal/blob/main/crates/image-capi/) (`libedgefirst_image`, `edgefirst/image.h`).
- Bridged to Python via [`crates/python-image`](https://github.com/EdgeFirstAI/hal/blob/main/crates/python-image/).

## Features

- **Multiple backends** — Automatic selection: OpenGL (GPU) → G2D (NXP i.MX) → CPU (fallback)
- **Format conversion** — RGB/RGBA/BGRA/GREY, planar RGB(A), semi-planar NV12/NV16/NV24, packed YUYV/VYUY
- **Geometric transforms** — Source crop, resize, letterbox, rotate (90° increments), flip
- **Zero-copy integration** — Works with `edgefirst-tensor` DMA-BUF, IOSurface, AHardwareBuffer, PBO, and SHM buffers
- **Tiled preprocessing** — SAHI grids for small-object detection in 4K frames
- **JPEG/PNG support** — Load and save with EXIF orientation handling

## Quick Start

```rust
use edgefirst_image::{save_jpeg, ImageProcessor, ImageProcessorTrait, Rotation, Flip, Crop};
use edgefirst_codec::{peek_info, ImageDecoder, ImageLoad};
use edgefirst_tensor::{CpuAccess, PixelFormat, DType, Tensor, TensorDyn, TensorMemory};

// Decode an image into its native format. The codec reports the source's
// native pixel format (JPEG -> NV12/GREY, PNG -> RGB/RGBA/GREY) and sizes,
// then configures the destination tensor during the decode.
let bytes = std::fs::read("input.jpg")?;
let info = peek_info(&bytes)?;
let mut decoder = ImageDecoder::new();
let mut src = Tensor::<u8>::image(info.width, info.height, info.format, Some(TensorMemory::Mem),
    CpuAccess::ReadWrite)?;
src.load_image(&mut decoder, &bytes)?;
let src = TensorDyn::from(src);

// Create processor (auto-selects best backend)
let mut processor = ImageProcessor::new()?;

// Create destination with desired size and format (the convert below
// handles NV12 -> RGBA colour conversion, resize, and letterboxing)
let mut dst =
    processor.create_image(640, 640, PixelFormat::Rgba, DType::U8, None, CpuAccess::ReadWrite)?;

// Convert with resize, rotation, letterboxing
processor.convert(
    &src,
    &mut dst,
    Rotation::None,
    Flip::None,
    Crop::letterbox([114, 114, 114, 255]),  // preserve aspect ratio, grey pad
)?;

// Save result
save_jpeg(&dst, "output.jpg", 90)?;
```

## Backends

| Backend | Platform | Hardware | Notes |
|---------|----------|----------|-------|
| G2D | Linux (NXP i.MX 8M Plus / 8M Mini) | 2D blit engine | Fastest for NXP platforms; no mask rendering |
| OpenGL | Linux, macOS/iOS, Android, Windows | GPU | One engine everywhere behind the `GlPlatform` seam: EGL/GBM + DMA-BUF on Linux, ANGLE→Metal + IOSurface on Apple, native EGL + AHardwareBuffer on Android, ANGLE→Direct3D 11 + D3D11 texture transfers on Windows, PBO the fallback where a format has no texture layout (`EDGEFIRST_ANGLE_PATH` for the DLLs; `EDGEFIRST_D3D11_ADAPTER`, alias `EDGEFIRST_ANGLE_ADAPTER`, for the adapter — see the root README § Windows GPU Acceleration) |
| CPU | All | SIMD (NEON / AVX2, rayon) | Portable fallback, always available |

## Supported Formats

| Format | Description | Channels |
|--------|-------------|----------|
| `PixelFormat::Rgba` | 32-bit RGBA | 4 |
| `PixelFormat::Rgb` | 24-bit RGB | 3 |
| `PixelFormat::Nv12` | YUV 4:2:0 semi-planar | 1.5 |
| `PixelFormat::Nv16` | YUV 4:2:2 semi-planar | 2 |
| `PixelFormat::Nv24` | YUV 4:4:4 semi-planar | 3 |
| `PixelFormat::Yuyv` | YUV 4:2:2 packed | 2 |
| `PixelFormat::Grey` | 8-bit grayscale | 1 |
| `PixelFormat::PlanarRgb` | Planar RGB | 3 |
| `PixelFormat::Vyuy` | YUV 4:2:2 packed (VYUY order) | 2 |
| `PixelFormat::Bgra` | 32-bit BGRA | 4 |
| `PixelFormat::PlanarRgba` | Planar RGBA | 4 |

Note: Int8 variants (e.g. packed RGB int8, planar RGB int8) use `DType::I8` with the corresponding `PixelFormat` rather than separate format constants.

## Feature Flags

- `opengl` (default) — Enable the OpenGL backend.
- `tracker` — Enable multi-object tracking in `draw_masks_tracked()`. Requires `features = ["tracker"]` in your dependency declaration.
- `opencv` — Build the OpenCV comparison benchmark. Needs a system OpenCV install.
- `dma_test_formats`, `g2d_test_formats` — Test-only: unlock the zero-copy and G2D fixture tiers. See [TESTING.md](https://github.com/EdgeFirstAI/hal/blob/main/crates/image/TESTING.md).

The decoder dependency is unconditional — there is no feature flag to opt out
of detection-box and mask rendering.

## Environment Variables

Backend selection:

- `EDGEFIRST_FORCE_BACKEND` — Force a single backend: `cpu`, `g2d`, or `opengl`. Disables the fallback chain, and makes the `EDGEFIRST_DISABLE_*` variables inert.
- `EDGEFIRST_DISABLE_GL` / `EDGEFIRST_DISABLE_G2D` / `EDGEFIRST_DISABLE_CPU` — Set to `1` to drop that backend from the chain.

Memory and transfer:

- `EDGEFIRST_TENSOR_FORCE_MEM` — Set to `1` to force heap memory (disables DMA/SHM).
- `EDGEFIRST_FORCE_TRANSFER` — Force the GPU transfer method: `dmabuf`, `pbo`, or `sync`.
- `EDGEFIRST_EGL_CACHE_CAPACITY` — Per-cache EGLImage capacity (default 64).
- `EDGEFIRST_OPENGL_RENDERSURFACE` — Set to `1` to use renderbuffer-backed EGLImages for DMA destinations. Required on i.MX 95 / Mali-G310 with Neutron NPU DMA-BUF destinations. Defaults to `0` (texture path).

Conversion behaviour:

- `EDGEFIRST_COLORIMETRY` — `fast` (default) or `exact`. `fast` keeps single-plane NV12 on the driver's YUV sampler even when its colorimetry does not match; `exact` forces the in-shader matrix. See [ARCHITECTURE.md § Colorimetry](https://github.com/EdgeFirstAI/hal/blob/main/crates/image/ARCHITECTURE.md#colorimetry-1).
- `EDGEFIRST_NV_CONVERT_PATH` — `auto` (default), `sampler`, or `shader`. Pins the NV12 GPU conversion path for A/B measurement.
- `EDGEFIRST_GL_SERIALIZE` — `full` or `lifecycle`. Overrides the per-driver GL serialization policy.
- `EDGEFIRST_PROTO_COMPUTE` — Set to `1` to enable the experimental GLES 3.1 compute shader path for proto repack. Requires GLES 3.1 hardware.

## Segmentation Mask Rendering

Three rendering pipelines for YOLO instance segmentation masks:

### MaskOverlay

`MaskOverlay` controls how segmentation masks are composited onto the destination image:

```rust,ignore
use edgefirst_image::MaskOverlay;

// Default: no background replacement, full opacity
let overlay = MaskOverlay::default();

// With a background image and 50% transparent masks
let overlay = MaskOverlay {
    background: Some(&bg_tensor),
    opacity: 0.5,
    ..MaskOverlay::default()
};
```

Fields:
- `background: Option<&TensorDyn>` — Optional tensor to blit into `dst` before drawing masks. Must match `dst`'s shape and format, and must not alias `dst` (an aliased pair returns `Error::AliasedBuffers`). `None` clears `dst` instead.
- `opacity: f32` — Scales mask alpha in the range `0.0` (invisible) to `1.0` (fully opaque, default).
- `letterbox: Option<[f32; 4]>` — `[xmin, ymin, xmax, ymax]` in model-input normalized space. When set, decoder output is mapped back to the original image's coordinates. Build it from the same `Crop` you gave `convert()` with `MaskOverlay::with_letterbox_crop`.
- `color_mode: ColorMode` — `Class` (default, colour per class label), `Instance` (colour per detection index), or `Track` (reserved for track IDs; behaves like `Instance` today).

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
let mut detections = Vec::new();
if let Some(proto_data) = decoder.decode_proto(&outputs, &mut detections)? {
    processor.draw_proto_masks(&mut frame, &detections, &proto_data, MaskOverlay::default())?;
}
```

### Hybrid CPU+GPU Path

The CPU materializes binary masks with `materialize_masks()`, then OpenGL blits
them via `draw_decoded_masks()`. This is the recommended pattern on Vivante
GC7000UL (i.MX 8M Plus), where the fused fragment shader falls off a
performance cliff at high detection counts:

```rust,ignore
use edgefirst_image::MaskResolution;

let masks = processor.materialize_masks(
    &detections,
    &proto_data,
    overlay.letterbox,
    MaskResolution::Scaled { width: dst_w, height: dst_h },
)?;
drop(proto_data); // free the proto tensor immediately

processor.draw_decoded_masks(&mut frame, &detections, &masks, MaskOverlay::default())?;
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

See [BENCHMARKS.md](https://github.com/EdgeFirstAI/hal/blob/main/BENCHMARKS.md) for per-platform performance numbers.

## Zero-Copy Model Input

Use `create_image()` to allocate the destination tensor with the processor's
optimal memory backend (DMA-buf, PBO, or system memory). This enables
zero-copy GPU paths that direct `Tensor::new()` allocation cannot achieve:

```rust,ignore
let mut dst =
    processor.create_image(640, 640, PixelFormat::Rgb, DType::U8, None, CpuAccess::ReadWrite)?;
processor.convert(&src, &mut dst, Rotation::None, Flip::None, Crop::letterbox([114, 114, 114, 255]))?;
```

If you need to write into a pre-allocated buffer with a specific memory type
(e.g. an NPU-bound tensor), you can still use direct allocation:

```rust,ignore
let mut model_input = Tensor::<u8>::new(&[640, 640, 3], None, None)?;
model_input.set_format(PixelFormat::Rgb)?;
let mut dst = TensorDyn::from(model_input);
processor.convert(&src, &mut dst, Rotation::None, Flip::None, Crop::letterbox([114, 114, 114, 255]))?;
```

## Tiled Preprocessing (SAHI)

For small-object detection in high-resolution frames (e.g. 4K), run the model at
its native tile resolution over an overlapping grid of native-resolution crops —
**SAHI** (Slicing Aided Hyper Inference). `ImageProcessor` renders every tile into
a single tall packed batch tensor with **one** GL import and **one** flush, so the
N tiles cost roughly one GPU sync rather than N.

```rust,ignore
use edgefirst_image::TilingConfig;
use edgefirst_tensor::{CpuAccess, DType, PixelFormat};

// 640x640 tiles, >=20% overlap (the minimum; actual overlap is redistributed
// evenly so every tile is full-size and the last tile lands flush at frame-tile).
let cfg = TilingConfig::new(640, 640).with_overlap(0.2);

// plan_tiles is pure geometry (no GPU): use its length to size the batch.
let placements = processor.plan_tiles(src_w, src_h, &cfg)?;

// One tall destination: tile_w wide, N*tile_h tall — N tiles stacked
// vertically. Allocate once, reuse per frame.
let mut batch = processor.alloc_tile_batch(
    placements.len(), &cfg,
    PixelFormat::Rgb, DType::U8, None, CpuAccess::None,
)?;

// Render all tiles (deferred convert per tile + single flush). Returns the same
// placements mapping each tile band back to full-frame coordinates.
let placements = processor.tile_into(&src, &mut batch, &cfg)?;
```

Each tile selects its source crop by **sampling** the whole-frame tensor (texture
coordinates), never by viewing it — a viewed source would mint one EGLImage import
per tile and defeat the zero-copy property. The destination tiles are sibling views
of one parent buffer rendered as `glViewport`/`glScissor` bands.

> **Batch-format limit.** The GL band lowering handles single-pass geometry
> only. A zero-copy (DMA) batch destination in packed `Rgb` or any planar
> layout reinterprets the render-target shape (`W*3/4 × H`, `H*3` bands), which
> the band path does not yet compute, so GL declines those tiles and each one
> falls back to CPU. Batch in `Rgba`, `Bgra`, or `Grey` to stay on the batched
> GPU path today, or use `tile_one` — it writes whole slots rather than bands,
> so it is not subject to this limit at any format.

For pipelined I/O (overlapping preprocessing, inference, and collection), use
`tile_one` to render into a caller-owned model-input slot (e.g. one slot of a ring)
instead of the batched `tile_into`:

```rust,ignore
let placements = processor.plan_tiles(src_w, src_h, &cfg)?; // pure, no GPU
let mut slot = processor.create_image(
    cfg.tile_w, cfg.tile_h, PixelFormat::Rgb, DType::U8, None, CpuAccess::None,
)?;
for p in &placements {
    processor.tile_one(&src, &mut slot, p, &cfg)?; // deferred: caller owns sync
    processor.flush()?;                            // flush per-slot or per-ring cadence
    // ... submit slot for inference, collect output keyed by p.index ...
}
```

`plan_tiles` is pure geometry (no GPU work) so a profiler can size pools up front;
`tile_one` issues one deferred convert into a caller-owned slot so tiles overlap
with inference. The returned `TilePlacement` values are consumed by the decoder's
`lift_tile_boxes` / `TiledFrameAccumulator` to merge per-tile detections back into
full-frame results (see the `edgefirst-decoder` README).

## Zero-Copy External Buffer (Linux)

When integrating with an NPU delegate (e.g. VxDelegate) that owns its own
DMA-BUF buffers, use `import_image()` to render directly into the
delegate's buffer — eliminating the `memcpy` between HAL's buffer and the
delegate's buffer:

```rust,ignore
use edgefirst_tensor::PlaneDescriptor;

// UC1: Render into VxDelegate's DMA-BUF — zero copies
let pd = PlaneDescriptor::new(vx_fd.as_fd())?;  // dups fd — caller keeps ownership
let mut dst = processor.import_image(
    pd,
    None,               // chroma plane, for multiplane NV12/NV16
    640, 640,
    PixelFormat::Rgb,
    DType::U8,
    None,               // colorimetry: supply the producer's signalling for YUV sources
)?;
processor.convert(&src, &mut dst, Rotation::None, Flip::None, Crop::letterbox([114, 114, 114, 255]))?;
// dst's backing memory IS vx_fd — no memcpy needed
```

For the reverse direction (HAL allocates, consumer imports):

```rust,ignore
let hal_dst =
    processor.create_image(640, 640, PixelFormat::Rgb, DType::U8, None, CpuAccess::ReadWrite)?;
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

## Documentation

- Architecture overview: [ARCHITECTURE.md](https://github.com/EdgeFirstAI/hal/blob/main/crates/image/ARCHITECTURE.md)
- Testing guide: [TESTING.md](https://github.com/EdgeFirstAI/hal/blob/main/crates/image/TESTING.md)
- Full API reference: [docs.rs/edgefirst-image](https://docs.rs/edgefirst-image)
- Project README: [../../README.md](https://github.com/EdgeFirstAI/hal/blob/main/README.md)

## License

Licensed under the Apache License, Version 2.0. See [LICENSE](https://github.com/EdgeFirstAI/hal/blob/main/LICENSE) for details.
