# EdgeFirst Codec

Zero-allocation image decoding framework for pre-allocated tensor buffers
in real-time vision pipelines.

## Overview

`edgefirst-codec` decodes JPEG and PNG images directly into pre-allocated
`Tensor<T>` or `TensorDyn` buffers, supporting strided memory layouts
(GPU pitch-aligned DMA-BUF, PBO). This eliminates per-frame allocations
in the hot loop — the primary design goal.

The decoder emits each image in its **native pixel format** and never colour-
converts, rotates, or resizes. Colour conversion and geometry are the job of
[`ImageProcessor::convert()`](../image) downstream:

| Input | Native output format(s)                        |
|-------|------------------------------------------------|
| JPEG  | `Nv12` (3-component colour) / `Grey` (1 component) |
| PNG   | `Rgb` / `Rgba` / `Grey`                        |

The decoder configures the destination tensor's dimensions and pixel format to
match the decoded image (within the tensor's existing allocation), so a single
tensor sized for the largest expected frame can receive smaller images without
reallocating.

JPEG decoding uses a custom from-scratch baseline decoder with reusable state,
achieving zero heap allocations after the first decode at each resolution.
SIMD-optimized IDCT kernels (NEON on AArch64, SSE4.1/SSE2 on x86-64) are
selected automatically at init via dynamic dispatch. On Linux, an optional
[V4L2 hardware backend](#hardware-acceleration-v4l2) offloads JPEG decode to a
SoC accelerator when one is present. PNG decoding uses `zune-png`.

## Quick Start

```rust
use edgefirst_codec::{ImageDecoder, ImageLoad};
use edgefirst_tensor::{CpuAccess, Tensor, PixelFormat, TensorMemory};

// Allocate once at init (prefer ImageProcessor::create_image() for DMA/PBO).
// A colour JPEG decodes to NV12, so allocate an NV12 tensor.
let mut tensor = Tensor::<u8>::image(1920, 1080, PixelFormat::Nv12,
    Some(TensorMemory::Mem), CpuAccess::ReadWrite).unwrap();
let mut decoder = ImageDecoder::new();

// Decode in the hot loop — zero allocations after warmup for JPEG.
let jpeg_bytes = std::fs::read("frame.jpg").unwrap();
let info = tensor.load_image(&mut decoder, &jpeg_bytes).unwrap();
println!("Decoded {}x{} {:?}", info.width, info.height, info.format);
// info.rotation_degrees / info.flip_horizontal carry the EXIF orientation
// the caller should apply downstream (the codec never rotates — see below).
```

## Recommended Pattern

For maximum performance, use tensors allocated by
`ImageProcessor::create_image()` and convert the native decode into the format
your pipeline needs:

```rust,ignore
use edgefirst_image::{ImageProcessor, ImageProcessorTrait, Crop, Rotation, Flip};
use edgefirst_codec::{ImageDecoder, ImageLoad};

let mut processor = ImageProcessor::new()?;
// Source tensor holds the codec's native NV12; destination is the RGB the
// model consumes.
let mut src =
    processor.create_image(1920, 1080, PixelFormat::Nv12, DType::U8, None, CpuAccess::ReadWrite)?;
let mut dst =
    processor.create_image(640, 640, PixelFormat::Rgb, DType::U8, None, CpuAccess::ReadWrite)?;
let mut decoder = ImageDecoder::new();

loop {
    let bytes = capture_frame();
    let info = src.load_image(&mut decoder, &bytes)?;
    // convert() performs colour conversion (NV12 → RGB), resize, and any EXIF
    // rotation/flip the decode reported.
    processor.convert(&src, &mut dst, Rotation::None, Flip::None,
        Crop::new(0, 0, info.width, info.height))?;
}
```

Benefits of `ImageProcessor::create_image()` tensors:
- **DMA-BUF backing** for zero-copy GPU import (and the V4L2 zero-copy decode
  path — see below)
- **PBO backing** when OpenGL is the active transfer path
- **GPU pitch alignment** (64-byte for Mali compatibility)

Free-standing tensors work but cannot use PBO and may lack GPU-aligned pitch.

## EXIF Orientation: Reported, Never Applied

The decoder **reports** the source's EXIF orientation in `ImageInfo` but writes
the pixels and dimensions exactly as stored — it never rotates or flips. This
keeps the decode path branch-free and lets the GPU `convert()` apply orientation
for free alongside colour conversion and resize.

`ImageInfo` carries the transform the caller should apply:
- `rotation_degrees`: clockwise rotation in degrees (`0` / `90` / `180` / `270`)
- `flip_horizontal`: whether to also mirror horizontally

Apply the rotation by passing it to `ImageProcessor::convert()` (mapping
`rotation_degrees` to `Rotation` and `flip_horizontal` to `Flip`). When the
image has no EXIF orientation both fields are `0` / `false`.

## Hardware Acceleration (V4L2)

On Linux, the `v4l2` feature (enabled by default) adds a hardware JPEG-decode
backend that drives any device exposing a JPEG decoder through the standard
V4L2 mem2mem (M2M) API — the lead target is i.MX `mxc-jpeg`, but discovery is
purely capability-based, so no device node, driver name, or output format is
hardcoded.

The backend is probed lazily on the first JPEG decode and is tried **before**
the software decoder. Anything it cannot drive transparently — no JPEG M2M
device present, an unsupported capture format, a per-image hardware failure —
falls back to the from-scratch CPU decoder, producing identical native output.
After repeated failures a circuit breaker demotes the device to CPU for the
rest of the session.

The deps (`nix`, `libc`) are pulled in only on Linux targets and all backend
code is gated `#[cfg(all(target_os = "linux", feature = "v4l2"))]`, so off Linux
the feature compiles to nothing.

| Environment variable          | Effect                                        |
|-------------------------------|-----------------------------------------------|
| `EDGEFIRST_DISABLE_V4L2=1`    | Skip the probe entirely; always use the CPU decoder |
| `EDGEFIRST_CODEC_V4L2_DEVICE` | Probe only this device node (e.g. `/dev/video11`) instead of enumerating `/dev/video*` |

When the destination is a DMA-backed tensor with MCU(16)-aligned dimensions and
the driver accepts a single-plane contiguous capture at the tensor pitch, the
hardware decodes straight into the tensor's dmabuf — a true zero-copy path.
Otherwise the driver buffers are mapped and the decoded planes are copied
(cropped to the logical image) into the destination.

## Supported Formats

| Format | Input  | Native output                              |
|--------|--------|--------------------------------------------|
| JPEG   | `&[u8]`| `Nv12` (colour) / `Grey` (greyscale), `u8` only |
| PNG    | `&[u8]`| `Rgb` / `Rgba` / `Grey`                    |

Need `Rgb`/`Rgba`/`Bgra` from a JPEG, or a resized/rotated result? Decode to the
native format, then call `ImageProcessor::convert()`.

## Decoder Limitations

The codec decodes a strict subset of the JPEG / PNG specs. Inputs that fall
outside the supported subset surface a typed `CodecError::Unsupported(...)`
variant so callers can pattern-match programmatically (no string parsing
required).

### JPEG

| JPEG feature                                     | Status        |
|--------------------------------------------------|---------------|
| Baseline DCT (SOF0)                              | Supported     |
| 8-bit sample precision                           | Supported     |
| 1 component (greyscale → `Grey`) or 3 components (YCbCr → `Nv12`) | Supported |
| Chroma subsampling 4:4:4 / 4:2:2 / 4:2:0 / 4:4:0 | Supported (downsampled to 4:2:0 for `Nv12`) |
| Non-`u8` destination tensor                      | **Unsupported** — `UnsupportedDtype` (NV12/GREY are `u8`) |
| Progressive DCT (SOF2)                           | **Unsupported** — `Unsupported(ProgressiveJpeg)` |
| Extended sequential DCT (SOF1)                   | **Unsupported** |
| Lossless predictive (SOF3)                       | **Unsupported** — `Unsupported(LosslessJpeg)` |
| Hierarchical (SOF5/6/7)                          | **Unsupported** — `Unsupported(HierarchicalJpeg)` |
| Arithmetic coding (SOF9/10/11/13/14/15)          | **Unsupported** — `Unsupported(ArithmeticCodedJpeg)` |
| Sample precision other than 8-bit                | **Unsupported** — `Unsupported(JpegPrecision { bits })` |
| CMYK / YCCK / >3 components                      | **Unsupported** — `Unsupported(JpegComponentCount { components })` |
| Chroma sampling that exceeds luma                | **Unsupported** — `Unsupported(JpegChromaSubsampling)` |
| Thumbnails (JFIF / APP markers)                  | Ignored       |
| EXIF orientation                                 | Reported in `ImageInfo`, never applied (see above) |

### PNG

PNG decoding goes through `zune-png`; the codec writes the native colorspace
(Luma/LumaA → `Grey`, RGB → `Rgb`, RGBA → `Rgba`) into the tensor with
stride-aware row copies and optional bit-depth/dtype conversion.

| PNG feature                                      | Status        |
|--------------------------------------------------|---------------|
| 8-bit colorspace: Luma / LumaA / RGB / RGBA      | Supported     |
| 16-bit colorspace: RGB / RGBA / Luma → `u16` / `i16` / `f32` tensors | Supported |
| `eXIf` chunk orientation                         | Reported in `ImageInfo`, never applied |
| Palette (indexed-color) PNG                      | Per zune-png (expanded to RGB/RGBA by the decoder) |
| APNG (animated)                                  | Not exercised (decoder set to `png_set_decode_animated(false)`) |
| Interlaced (Adam7)                               | Per zune-png |

## Data Types

JPEG decodes to `u8` only (its native `Nv12`/`Grey` are byte layouts). PNG
supports the full set of tensor element types:

| Type  | PNG support | Notes                              |
|-------|-------------|------------------------------------|
| `u8`  | ✓           | Direct copy (identity)             |
| `u16` | ✓           | Scaled `* 257` from 8-bit; native from 16-bit PNG |
| `i8`  | ✓           | XOR 0x80 sign-bit flip             |
| `i16` | ✓           | XOR 0x8000 sign-bit flip           |
| `f32` | ✓           | Normalized to [0.0, 1.0]           |

## API Reference

### `ImageDecoder`

Reusable decoder with internal scratch buffers (and, on Linux, the lazily-probed
V4L2 backend state). Create once, reuse across frames — scratch buffers and the
hardware streaming session amortize after the first decode.

### `ImageLoad` Extension Trait

Implemented for both `Tensor<T>` and `TensorDyn`:

- `load_image(&mut self, decoder, data)` — decode from `&[u8]`
- `load_image_read(&mut self, decoder, reader)` — decode from `Read`
- `load_image_file(&mut self, decoder, path)` — decode from file path

Each configures the tensor's dimensions and format to the decoded native format
and returns an `ImageInfo`. Returns `CodecError::InsufficientCapacity` if the
decoded image is larger than the tensor's allocation.

### `ImageInfo`

Returned by all decode methods:
- `width`, `height`: decoded image size (the source's true, unrotated dimensions)
- `format`: native pixel format written to the tensor
- `row_stride`: row stride in bytes used when writing into the tensor
- `rotation_degrees`: EXIF clockwise rotation the caller should apply (`0`/`90`/`180`/`270`)
- `flip_horizontal`: whether the caller should also flip horizontally

`peek_image_info(data)` returns the same metadata without decoding pixels.

## License

Apache-2.0
