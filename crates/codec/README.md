# EdgeFirst Codec

Zero-allocation image decoding framework for pre-allocated tensor buffers
in real-time vision pipelines.

## Overview

`edgefirst-codec` encodes and decodes video with the platform's hardware H.264 codecs
(see [Video Encoding](#video-encoding-h264) and [Video Decoding](#video-decoding-h264)) and decodes JPEG and PNG images directly into pre-allocated
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
use edgefirst_image::{ConvertOptions, Flip, ImageProcessor, ImageProcessorTrait, Rotation};
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
    // convert() performs colour conversion (NV12 → RGB) and resize; the codec
    // reports EXIF orientation in `info` but does not apply it, so pass it on.
    let rot = Rotation::from_degrees_clockwise(info.rotation_degrees as usize);
    let flip = if info.flip_horizontal { Flip::Horizontal } else { Flip::None };
    processor.convert(&src, &mut dst, Some(ConvertOptions::new().rotation(rot).flip(flip)))?;
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

## Engines and Capabilities

Every codec runs on an `Engine`: `V4l2` (Linux), `MediaFoundation` (Windows), `VideoToolbox` (Apple), `NvJpeg` (Linux with CUDA) or `Cpu` (still images only). `Codec` names the formats: `Jpeg`, `Png`, `H264`, `Mjpeg`, and `Hevc`, `Vp8`, `Vp9`, `Av1`, which no engine runs yet.

| Codec | Linux (V4L2) | Windows (Media Foundation) | CPU |
|---|---|---|---|
| H.264 encode / decode | yes / yes | yes / yes | no |
| MJPEG encode / decode | JPEG encoder / JPEG decoder (i.MX 95 and others with `mxc-jpeg`) | no / yes (GPU vendor's decoder, else Microsoft's) | no |
| JPEG still encode (`ImageEncoder`) | JPEG encoder | no | yes |
| JPEG / PNG still decode (`ImageDecoder`) | JPEG decoder | no | yes / yes |

`Engine::Auto`, the default everywhere, picks the first engine that can do the request: the platform's own for video, and nvJPEG (when enabled), V4L2, then the CPU for JPEG. Naming an engine (`EncoderConfig::with_engine`, `DecoderConfig::with_engine`, `ImageDecoder::set_engine`) uses that engine alone, with no fallback. There is no CPU video codec in HAL: a video codec the platform cannot run fails with `CodecError::Unsupported(UnsupportedFeature::Codec { .. })`, and an engine this platform or build does not have with `CodecError::NoDevice`. A platform engine may itself use software, such as Microsoft's H.264 encoder on a GPU without one; `EncoderInfo::hardware` and `Capability::hardware` say which.

`capabilities()` lists what the engines on this machine can do, without opening a stream: engine, device, codec, direction, whether it is hardware, the pixel formats, and how frames move at best (`Transfer::ZeroCopy`, `GpuCopy` or `CpuCopy`). An open stream reports what it actually does: `VideoEncoder::info()` returns an `EncoderInfo`, and `StreamInfo` carries the decoder's `engine` and `transfer`.

```rust,no_run
use edgefirst_codec::{capabilities, Codec, Direction};

for c in capabilities() {
    println!("{} {} {} on {} ({}, {:?}, {})", c.engine, c.direction, c.codec, c.device,
             if c.hardware { "hardware" } else { "software" }, c.formats, c.transfer);
}
let h264 = capabilities()
    .into_iter()
    .any(|c| c.codec == Codec::H264 && c.direction == Direction::Encode);
```

| Environment variable     | Effect                                     |
|--------------------------|--------------------------------------------|
| `EDGEFIRST_CODEC_ENGINE` | The engines `Engine::Auto` may use, in order, separated by commas: for example `cpu` decodes JPEG on the CPU only, and `media-foundation` keeps video on Media Foundation. Codec objects that name an engine ignore it |

## Encoding Still Images (JPEG)

`ImageEncoder` writes a tensor as a JPEG file. It holds no per-image state, so create one and reuse it; `encode_into` reuses the output buffer.

```rust,no_run
use edgefirst_codec::{Codec, ImageEncoder};
# let image: edgefirst_tensor::TensorDyn = unimplemented!();
let mut encoder = ImageEncoder::new(Codec::Jpeg)?;
encoder.set_quality(90)?;
let mut jpeg = Vec::new();
encoder.encode_into(&image, &mut jpeg)?;
# Ok::<(), Box<dyn std::error::Error>>(())
```

- **Engines:** under `Engine::Auto` a V4L2 JPEG encoder takes DMA-BUF sources it can read, kept open for the last geometry, and the CPU takes the rest. `set_engine(Engine::Cpu)` or `Engine::V4l2` uses one alone.
- **CPU sources:** `u8` NV12 (written 4:2:0), NV16 and YUYV (4:2:2), Grey, RGB and RGBA, at any row stride. Limited-range YUV is stretched to the full range JPEG readers expect; the matrix is kept, so BT.709 sources decode with BT.601 colours.
- **Quality** is 1 to 100, 85 by default.

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

## Video Encoding (H.264, MJPEG)

`edgefirst_codec::video::VideoEncoder` encodes image tensors to an H.264 Annex B
stream with the platform's hardware encoder. On Linux that is any V4L2 stateful
memory-to-memory encoder, found by capability: i.MX 8M Plus `vsi_v4l2`, i.MX 95
Wave6, and others. It needs the `v4l2` feature (on by default). On Windows it is a Media Foundation encoder transform: the GPU's (NVIDIA, Intel, AMD) where the driver registers one, otherwise Microsoft's software encoder. It needs the `media-foundation` feature (on by default).

```rust,no_run
use edgefirst_codec::video::{Bitrate, EncoderConfig, FrameOptions, VideoEncoder};
use edgefirst_tensor::PixelFormat;

let mut config = EncoderConfig::h264(1920, 1080, PixelFormat::Nv12, 30.0);
config.bitrate = Bitrate::Bps(8_000_000);
let mut encoder = VideoEncoder::new(config)?;
# let frame: edgefirst_tensor::TensorDyn = unimplemented!();
let opts = FrameOptions { pts: 0, keyframe: true, ..FrameOptions::default() };
if let Some(au) = encoder.encode(&frame, &opts)? {
    // au.data is one access unit; au.keyframe, au.pts
}
let rest = encoder.flush()?; // every access unit still in the encoder
# Ok::<(), Box<dyn std::error::Error>>(())
```

- **Sources** must be DMA-BUF tensors in the configured format. They are imported without a copy, and `encode()` returns once the device has read them, so the caller can reuse the buffer straight away. Every source of a stream must have the same row stride.
- **Crops:** `FrameOptions::crop` encodes a region of a larger source. The region must be the configured size and start on a chroma sample. NV12 crops need a device that takes the two-plane `NV12M` layout (both i.MX encoders do).
- **Headers:** with `EncoderConfig::repeat_headers` (the default), every IDR, forced or not, starts with the SPS and PPS, so a receiver can join at any key frame.
- **PTS:** `FrameOptions::pts` comes back unchanged on the matching `EncodedFrame`, in any unit.
- **Input formats** are whatever the device lists for HAL's V4L2 code of the format (`PixelFormat::to_fourcc`); NV12 and YUYV on both i.MX encoders.
- **Windows:** sources are NV12 or YUYV tensors in any memory. On a hardware encoder a native NV12 texture of the configured size is read in place, without a copy; crops, host tensors, and sources for an encoder that holds frames are copied into a texture of the encoder's own, on the GPU where the source is a texture. Either way the caller can reuse the source as soon as `encode()` returns. B-frames are off and the transform runs in low-latency mode, so `encode()` normally returns the access unit of the frame it was given. Microsoft's software encoder, used where there is no hardware encoder, reads system memory.
- **MJPEG:** `EncoderConfig::new(Codec::Mjpeg, ..)` encodes each frame as a JPEG picture, every one a key frame, at `EncoderConfig::quality` (85 by default). Only V4L2 JPEG encoders run it; elsewhere it is `CodecError::Unsupported`.
- **`info()`** names the engine and device and says how sources reach the encoder (`EncoderInfo::transfer`): zero-copy on V4L2; on Media Foundation zero-copy for an encoder that reads textures in place, a GPU copy for one that holds frames (decided on the first frame), and a CPU copy for the software encoder.

| Environment variable           | Effect                                     |
|--------------------------------|--------------------------------------------|
| `EDGEFIRST_CODEC_V4L2_ENCODER` | Use only this encoder node (e.g. `/dev/video0`) instead of searching `/dev/video*` |
| `EDGEFIRST_CODEC_MF_ENCODER`   | Windows: use only the encoder transforms whose name contains this text (e.g. `NVIDIA`), or only the `hardware` or `software` ones |

Run `cargo run --release -p edgefirst-codec --example video-encode -- out.h264 --frames 300` to encode a test pattern (or `--input frames.nv12`) and print the per-frame encode time.

## Video Decoding (H.264, MJPEG)

`edgefirst_codec::video::VideoDecoder` decodes an H.264 Annex B stream into NV12 image tensors with the platform's hardware decoder: on Linux any V4L2 stateful memory-to-memory decoder, found by capability (i.MX 8M Plus `vsi_v4l2`, i.MX 95 Wave6). It needs the `v4l2` feature (on by default). On Windows it is Microsoft's Media Foundation H.264 decoder, with the `media-foundation` feature (on by default).

```rust,no_run
use edgefirst_codec::video::{access_units, DecodeEvent, DecoderConfig, OutputPool, VideoDecoder};

let stream = std::fs::read("clip.h264")?;
let config = DecoderConfig::h264().with_output(OutputPool::Allocate { count: 4 });
let mut decoder = VideoDecoder::new(config)?;
for (pts, au) in access_units(&stream).enumerate() {
    for event in decoder.decode(au, pts as u64)? {
        match event {
            DecodeEvent::Frame(frame) => { /* frame.tensor(), frame.pts() */ }
            DecodeEvent::Reconfigured(info) => { /* info.width, info.height, info.crop */ }
        }
    }
}
let rest = decoder.flush()?; // every event still in the decoder
# Ok::<(), Box<dyn std::error::Error>>(())
```

- **Input** is one access unit per call, in decode order; `access_units()` splits a byte stream. `decode()` returns everything that is ready, which may be nothing, and frames come back in presentation order with the `pts` passed for their access unit.
- **Frames are lent, not copied.** Each `DecodedFrame` is an NV12 DMA-BUF tensor of the visible picture over the device's buffer (whose coded height may be larger: 1088 rows for 1080p). The buffer goes back to the device when the frame is dropped, so the tensor is only reachable through the frame; share a picture by sharing the frame (`Arc<DecodedFrame>`). Frames import into the image crate's GPU paths without a copy.
- **The pool:** `OutputPool::Allocate { count }` lets the caller hold `count` frames at once; the decoder adds the device's minimum and one buffer for it to decode into. `OutputPool::Provided(tensors)` decodes into the caller's DMA-BUFs. When the caller holds every buffer, `decode()` fails with `CodecError::PoolExhausted` without taking the access unit: drop frames and pass it again. `StreamInfo::memory` says whether the device imports the pool or, where it will not, writes its own buffers exported as DMA-BUFs, and `StreamInfo::transfer` how each picture reaches the pool (zero-copy on V4L2, a GPU copy on Media Foundation with DXVA, a CPU copy without).
- **MJPEG:** `DecoderConfig::new(Codec::Mjpeg)` takes one JPEG picture per `decode()` call and returns its frame at once. Pictures may change size from one to the next; each change is a `DecodeEvent::Reconfigured`. On Windows the decoder is the GPU vendor's MJPEG transform where there is one (NVIDIA's decodes on the GPU), otherwise Microsoft's software one; 4:2:2 pictures, as cameras send, come back as NV12. On Linux it is the V4L2 JPEG decoder, which writes NV12 or Grey; a picture it cannot decode is an error, since HAL has no CPU video decoder.
- **Size changes** mid-stream (a new SPS) drain the old pictures and rebuild the pool; `DecodeEvent::Reconfigured` carries the new `StreamInfo`. Frames already returned keep their memory.
- **`flush()`** returns the remaining pictures. The decoder then starts a new sequence, so the next access unit must be an IDR with its SPS and PPS, as after a seek.
- **Windows:** the decoder runs on the GPU (DXVA), and each picture is copied on the GPU into a pool tensor of the visible size, a native NV12 D3D11 texture whose recorded fence value orders consumers behind the copy; without DXVA (no D3D11 device, or an adapter without video decoding) it decodes into system memory, copied into the pool on the CPU. A provided pool must hold NV12 images of exactly the stream's visible size, so it cannot follow a size change. The decoder runs in low-latency mode: a picture comes back as soon as the stream's reordering allows.

| Environment variable           | Effect                                     |
|--------------------------------|--------------------------------------------|
| `EDGEFIRST_CODEC_V4L2_DECODER` | Use only this decoder node (e.g. `/dev/video1`) instead of searching `/dev/video*` |
| `EDGEFIRST_CODEC_MF_DECODER`   | Windows: use only the decoder transforms whose name contains this text, or only the `hardware` or `software` ones |

Run `cargo run --release -p edgefirst-codec --example video-decode -- clip.h264 --out frames.nv12` to decode a file, write the visible pictures and print the per-access-unit decode time.

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
| `CodecError::Unsupported(UnsupportedFeature)` | Input outside the supported JPEG/PNG subset — match on the inner variant rather than parsing a string. See [Decoder Limitations](#decoder-limitations). `UnsupportedFeature::Codec` when the engine set with `set_engine` cannot decode the image |
| `CodecError::NoDevice` | The engine set with `set_engine` is not on this platform or in this build, or has no device |
| `CodecError::InvalidData` | Magic bytes match neither JPEG nor PNG, or the bitstream is malformed |
| `CodecError::Io` | The `load_image_file` / `load_image_read` source failed |
| `CodecError::Tensor` | The destination tensor rejected the reconfigure or map |

## Footprint

`edgefirst-codec` is a lightweight dependency. Installing the Python package (`edgefirst-codec` plus its `edgefirst-tensor` dependency) costs roughly **1.5 MB to download and ~5 MB installed**, excluding NumPy.

Approximate marginal cost of adding JPEG decode to a Python project, over and above the NumPy every option requires:

| Package | Approx. download | Notes |
|---|---|---|
| PyTurboJPEG | ~0.5 MB | `ctypes` wrapper; the wheel is tiny but `libturbojpeg` must be installed separately as a system package |
| simplejpeg | ~0.4 MB | JPEG only; bundles libjpeg-turbo |
| **edgefirst-codec** | **~1.5 MB** | JPEG + PNG + EXIF, V4L2 and nvJPEG hardware backends, zero-copy decode into DMA tensors |
| Pillow | ~6–7 MB | Bundles AVIF, WebP, TIFF, OpenJPEG, freetype and harfbuzz alongside libjpeg |
| opencv-python-headless | ~40–60 MB | Full computer-vision library |
| opencv-python | ~50–75 MB | Adds GUI backends |

Two things to read from this. Against Pillow and OpenCV — the libraries most projects actually reach for when they need to decode a JPEG — we are smaller by roughly an order of magnitude, because neither is a codec: their bulk is formats and subsystems a decode-only pipeline never touches. Against the dedicated JPEG wrappers we are a few times larger, but in absolute terms the gap is around a megabyte, and it buys PNG, EXIF, the hardware backends, and a self-contained wheel with no system libraries to install.

For scale, NumPy itself is ~16 MB to download and dominates every row above, so the choice between any of these has little effect on total install size.

These figures are approximate and vary by platform and version; they are here to convey magnitude, not to be exact.

## License

Apache-2.0

<img referrerpolicy="no-referrer" src="https://px.edgefirst.ai/a.png?x-pxid=37aa7de2-6935-484e-b31d-9fd9a57fcd63" alt="" width="1" height="1" style="position:absolute; width:1px; height:1px; opacity:0; pointer-events:none;" />
