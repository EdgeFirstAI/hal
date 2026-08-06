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
| JPEG  | `Nv12` (4:2:0) / `Nv16` (4:2:2) / `Nv24` (4:4:4) / `Grey` (1 component) |
| PNG   | `Rgb` / `Rgba` / `Grey`                        |

A colour JPEG lands on the NV format that matches its own chroma sampling, so
nothing is resampled on the way out. Only exotic subsamplings (4:1:1, 4:1:0,
mismatched Cb/Cr) get downsampled, and they fall back to `Nv12`.

The decoder configures the destination tensor's dimensions and pixel format to
match the decoded image (within the tensor's existing allocation), so a single
tensor sized for the largest expected frame can receive smaller images without
reallocating.

JPEG decoding uses a custom from-scratch baseline decoder with reusable state,
achieving zero heap allocations after the first decode at each resolution.
SIMD-optimized IDCT kernels (NEON on AArch64, SSE4.1/SSE2 on x86-64) are
selected automatically at init via dynamic dispatch. On Linux two optional
hardware backends sit in front of it — [V4L2](#hardware-acceleration-v4l2) for
SoC JPEG accelerators and [nvJPEG](#gpu-acceleration-nvjpeg) for CUDA GPUs —
and either falls back to the CPU decoder transparently. PNG decoding uses
`zune-png`.

## Feature Flags

| Feature | Default | Effect |
|---|---|---|
| `v4l2` | on | Linux V4L2 mem2mem hardware JPEG decode. Opt-out at runtime with `EDGEFIRST_DISABLE_V4L2=1` |
| `nvjpeg` | on | Linux CUDA nvJPEG GPU decode. Opt-**in** at runtime with `EDGEFIRST_ENABLE_NVJPEG=1` |
| `opencv` | off | OpenCV-backed decode, for comparison and parity testing |
| `turbojpeg` | off | libjpeg-turbo, for benchmark comparison |

Both default features are fully `#[cfg]`-gated to Linux, so off Linux they
compile to nothing. `nvjpeg` is pure `dlopen` and adds no link-time CUDA
dependency, so enabling it does not constrain where the binary runs.

## Quick Start

```rust,no_run
use edgefirst_codec::{ImageDecoder, ImageLoad};
use edgefirst_tensor::{CpuAccess, PixelFormat, Tensor, TensorMemory};

// Allocate once at init (prefer ImageProcessor::create_image() for DMA/PBO).
// A 4:2:0 colour JPEG decodes to NV12, so allocate an NV12 tensor. The
// decoder CPU-writes the pixels: declare CpuAccess::Write.
let mut tensor = Tensor::<u8>::image(1920, 1080, PixelFormat::Nv12,
    Some(TensorMemory::Mem), CpuAccess::Write).unwrap();
let mut decoder = ImageDecoder::new();

// Decode in the hot loop — zero allocations after warmup for JPEG.
let jpeg_bytes = std::fs::read("frame.jpg").unwrap();
let info = tensor.load_image(&mut decoder, &jpeg_bytes).unwrap();
println!("Decoded {}x{} {:?}", info.width, info.height, info.format);
// info.rotation_degrees / info.flip_horizontal carry the EXIF orientation
// the caller should apply downstream (the codec never rotates — see below).
```

Don't know the image's dimensions or chroma sampling up front? `peek_info`
parses the headers without touching pixels, so you can size the tensor to the
format the decode will actually produce:

```rust,no_run
use edgefirst_codec::{peek_info, ImageDecoder, ImageLoad};
use edgefirst_tensor::{CpuAccess, Tensor, TensorMemory};

let data = std::fs::read("frame.jpg").unwrap();
let peek = peek_info(&data).unwrap();
let mut tensor = Tensor::<u8>::image(peek.width, peek.height, peek.format,
    Some(TensorMemory::Mem), CpuAccess::Write).unwrap();
let mut decoder = ImageDecoder::new();
let info = tensor.load_image(&mut decoder, &data).unwrap();
```

## Recommended Pattern

For maximum performance, use tensors allocated by
`ImageProcessor::create_image()` and convert the native decode into the format
your pipeline needs:

```rust,ignore
use edgefirst_codec::{ImageDecoder, ImageLoad};
use edgefirst_image::{Crop, Flip, ImageProcessor, ImageProcessorTrait, Rotation};
use edgefirst_tensor::{CpuAccess, DType, PixelFormat};

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

## GPU Acceleration (nvJPEG)

On Linux with CUDA, the `nvjpeg` feature (compiled in by default) adds a GPU
JPEG-decode backend for NVIDIA platforms, with Jetson Orin as the lead target.
It sits ahead of V4L2 and CPU in the dispatch order, but it is **off unless you
opt in**:

| Environment variable | Effect |
|---|---|
| `EDGEFIRST_ENABLE_NVJPEG=1` | Turn the backend on (`true` / `yes` also work). Off by default |

The default is off because nvJPEG decodes on the same GPU your inference runs
on. Sharing the device can cost a concurrent TensorRT engine more than the
decode speedup returns, so it stays opt-in for decode-bound workloads or
machines with no concurrent GPU compute. V4L2 is the mirror image — a separate
hardware block that contends with nothing, so it is opt-*out* via
`EDGEFIRST_DISABLE_V4L2`.

Two things to know before enabling it:

- **It emits `Rgb`, not an NV format** — a deliberate exception to the
  native-format contract. nvJPEG does YCbCr→RGB on the GPU at near-zero
  marginal cost and the result is GPU-resident, which removes the downstream
  NV12→RGB step. (The L4T nvJPEG 12.3.3 build has no NV12 output mode anyway.)
- **It only engages for a CUDA-backed destination** — in practice a
  `TensorMemory::Pbo` tensor from `ImageProcessor::create_image()` on Jetson.
  Anything else is left untouched and falls through to V4L2 or CPU.

The backend is pure `dlopen` with no link-time CUDA dependency, so one binary
runs on Jetson (nvJPEG), i.MX (V4L2), and a laptop (CPU). `nvjpeg_available()`
reports whether it is live — useful for benchmarks and for consumers that
branch on backend availability. Transient nvJPEG failures restore the native
format and fall through to the CPU decoder, which also covers the
progressive and non-baseline JPEGs nvJPEG rejects.

## Supported Formats

| Format | Input  | Native output                              |
|--------|--------|--------------------------------------------|
| JPEG   | `&[u8]`| `Nv12` / `Nv16` / `Nv24` (colour, matching the source's chroma sampling) or `Grey` (greyscale), `u8` only |
| PNG    | `&[u8]`| `Rgb` / `Rgba` / `Grey`                    |

Need `Rgb`/`Rgba`/`Bgra` from a JPEG, or a resized/rotated result? Decode to the
native format, then call `ImageProcessor::convert()`.

The one exception is the nvJPEG GPU backend, which emits packed `Rgb` — see
below.

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
| 1 component (greyscale → `Grey`) or 3 components (YCbCr → an NV format) | Supported |
| Chroma subsampling 4:4:4 → `Nv24`, 4:2:2 → `Nv16`, 4:2:0 → `Nv12` | Supported, no resampling |
| Other subsampling (4:1:1, 4:1:0, mismatched Cb/Cr) | Supported — block-averaged down to `Nv12` |
| Non-`u8` destination tensor                      | **Unsupported** — `UnsupportedDtype` (the NV formats and GREY are `u8`) |
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

JPEG decodes to `u8` only (its native NV formats and `Grey` are byte
layouts). PNG
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

### `peek_info`

`peek_info(data)` returns the same `ImageInfo` from the headers alone, without
decoding pixels. Use it in one-shot flows to size a tensor to the image (see
Quick Start). Its `row_stride` is the natural pitch for the format, since no
destination tensor is involved yet.

### Errors

Every decode entry point returns `Result<ImageInfo, CodecError>`:

| Variant | Cause |
|---|---|
| `CodecError::InsufficientCapacity` | The decoded image does not fit the tensor's allocation. Allocate for the largest expected frame |
| `CodecError::UnsupportedDtype` | Non-`u8` destination for a JPEG (`Nv12`/`Nv16`/`Nv24`/`Grey` are byte layouts) |
| `CodecError::UnsupportedFormat` | The destination's pixel format cannot hold this decode |
| `CodecError::Unsupported(UnsupportedFeature)` | Input outside the supported JPEG/PNG subset — match on the inner variant rather than parsing a string. See [Decoder Limitations](#decoder-limitations) |
| `CodecError::InvalidData` | Magic bytes match neither JPEG nor PNG, or the bitstream is malformed |
| `CodecError::Io` | The `load_image_file` / `load_image_read` source failed |
| `CodecError::Tensor` | The destination tensor rejected the reconfigure or map |

## License

Apache-2.0
