# EdgeFirst Codec Architecture

## Overview

The `edgefirst-codec` crate provides image decoding into pre-allocated
tensor buffers. It is designed for real-time vision pipelines where the
anti-pattern of allocating new output buffers on every frame must be
avoided.

The core principle: **allocate once at init, decode in the hot loop**.

A second principle drives the data path: the decoder emits each image in its
**native pixel format** and does nothing else — no colour conversion, no
resize, no rotation. JPEG decodes to `Nv12` (colour) or `Grey` (greyscale);
PNG decodes to `Rgb` / `Rgba` / `Grey`. Everything beyond raw decode —
colour-space conversion, EXIF orientation, resize, crop — belongs to
[`ImageProcessor::convert()`](../image), which runs on the GPU where available.
This keeps the decode path branch-free and lets a single `convert()` fold all
the geometry/colour work into one pass.

## Crate Position in the Workspace

```
edgefirst-tensor ← edgefirst-codec ← edgefirst-image (re-export)
                                    ← edgefirst-hal (re-export)
                                    ← crates/python (bindings)
```

`edgefirst-codec` depends only on `edgefirst-tensor` plus `zune-png`
(for PNG decoding) and `kamadak-exif` (for reading EXIF orientation). JPEG
decoding uses a custom from-scratch decoder with no external dependencies.
On Linux, the default-on `v4l2` feature adds `nix` and `libc` (Linux-target
only) for the hardware backend. The crate has no dependency on
`edgefirst-image` or any GPU libraries, keeping the dependency graph clean.

## Module Map

| Module       | Purpose                                         |
|--------------|-------------------------------------------------|
| `lib.rs`     | Crate root, public re-exports                   |
| `error.rs`   | `CodecError` enum (capacity / dtype / unsupported / IO / V4L2) |
| `pixel.rs`   | `ImagePixel` trait (u8, u16, i8, i16, f32)      |
| `options.rs` | `ImageInfo` struct (decoded metadata + reported EXIF orientation) |
| `exif.rs`    | EXIF orientation parsing → `(rotation_degrees, flip_horizontal)`; never applied |
| `decoder.rs` | `ImageDecoder` struct, magic-byte format detection, decode dispatch |
| `traits.rs`  | `ImageLoad` extension trait for Tensor/TensorDyn|
| `jpeg/`      | Custom baseline JPEG decoder + V4L2 backend (see below) |
| `png.rs`     | Native-format PNG decode with 8-bit and native 16-bit paths |

### JPEG Module Map (`jpeg/`)

| Module           | Purpose                                              |
|------------------|------------------------------------------------------|
| `mod.rs`         | `JpegDecoderState`, `decode_jpeg_into<T>()`, `native_format()`, V4L2 dispatch seam, EXIF reporting |
| `types.rs`       | `Component`, `SamplingFactor`, `ImageHeader`, `QuantTable`, `ZIGZAG` |
| `markers.rs`     | SOF/SOS/DQT/DHT/DRI/APP marker parsing               |
| `bitstream.rs`   | 64-bit bit buffer with FF/00 byte-stuffing, bulk refill |
| `huffman.rs`     | 11-bit lookahead Huffman LUT, `decode_block()` with dequant fusion |
| `idct/mod.rs`    | IDCT dispatcher (scalar/NEON/SSE4.1/SSE2 selection via function pointers) |
| `idct/scalar.rs` | Two-pass Loeffler 8×8 IDCT with DC-only fast path    |
| `idct/neon.rs`   | NEON 8×8 IDCT: 4-wide Loeffler butterfly, 4×4 transpose, DC-only fill |
| `idct/sse2.rs`   | SSE2 8×8 IDCT: 4-wide Loeffler butterfly, emulated mullo_epi32 |
| `idct/sse41.rs`  | SSE4.1 8×8 IDCT: native mullo_epi32, min/max clamping |
| `mcu.rs`         | MCU decode loop, `McuScratch`, native `Grey`/`Nv12` row writes, 4:2:0 chroma downsample (`avg_block`) |
| `v4l2/`          | Optional Linux hardware JPEG backend (see below)     |

### V4L2 Backend Module Map (`jpeg/v4l2/`, Linux + `v4l2` feature)

| Module        | Purpose                                                |
|---------------|--------------------------------------------------------|
| `mod.rs`      | `V4l2Probe` lifecycle, persistent streaming session, `try_decode()` orchestration, zero-copy + MMAP capture paths |
| `device.rs`   | Capability-based probe: env overrides, enumerate `/dev/video*`, `QUERYCAP` + `ENUM_FMT` (require JPEG on OUTPUT) |
| `ioctl.rs`    | All raw `#[repr(C)]` UAPI structs, FourCC + buffer-type/memory constants, `nix` ioctl macro defs |
| `buffers.rs`  | `REQBUFS`/`QUERYBUF`/mmap/`QBUF`/`DQBUF` for OUTPUT & CAPTURE; MMAP and DMABUF memory |
| `format.rs`   | `classify()` the driver-chosen CAPTURE FourCC → `CapKind` (`Nv12` / `Grey` / 4:4:4-packed) |

## Key Design Decisions

### Native-Format Output

The decoder writes the image's native pixel format and configures the
destination tensor to match (`Tensor::configure_image(w, h, format)`), within
the tensor's existing allocation:

- **JPEG**, 3-component colour → `Nv12` (Y plane + interleaved Cb/Cr at 4:2:0).
- **JPEG**, 1-component → `Grey`.
- **PNG** → `Rgb` / `Rgba` / `Grey` per the source colorspace.

JPEG output is `u8` only — `Nv12`/`Grey` are byte layouts — so a non-`u8`
destination is rejected with `CodecError::UnsupportedDtype`. The PNG path
supports the full set of tensor element types (see below).

No colour conversion, resize, or rotation happens here. Callers that need
`Rgb`/`Rgba`/`Bgra`, a resize, or EXIF orientation applied run
`ImageProcessor::convert()` on the native decode.

### EXIF Orientation Is Reported, Not Applied

`markers.rs` (JPEG APP1) and `zune-png` (`eXIf` chunk) surface the EXIF
orientation tag, which `exif.rs` maps to `(rotation_degrees, flip_horizontal)`
and returns in `ImageInfo`. The decoder **never** rotates or flips: it writes
the source's native, unrotated pixels and dimensions. The caller applies the
reported transform downstream (typically via the same `convert()` call that
does colour conversion and resize). This avoids a redundant in-place rotation
pass in the decode hot loop.

### Hardware Decode Tried Before CPU (Linux)

On Linux with the `v4l2` feature, `decode_jpeg_into` dispatches to the V4L2
hardware backend first; the from-scratch CPU decoder is the fallback. See
[V4L2 Hardware Backend](#v4l2-hardware-backend) below. On non-Linux targets, or
with the feature disabled, the seam compiles to nothing and the CPU decoder
always runs.

### Standalone `ImageDecoder` Struct

The decoder is a standalone struct rather than being embedded in
`ImageProcessor` or stored in thread-local state. This gives callers explicit
ownership and composability — one decoder per pipeline stage, no hidden global
state. Its `JpegDecoderState` holds the reusable MCU scratch and (on Linux) the
lazily-probed V4L2 backend, both amortised across frames.

```rust
let mut decoder = ImageDecoder::new();
// Scratch buffers and the hardware session amortize across calls
loop {
    let info = tensor.load_image(&mut decoder, &bytes)?;
}
```

### `ImageLoad` Extension Trait

The primary user-facing API is the `ImageLoad` trait, implemented for both
`Tensor<T>` (where `T: ImagePixel`) and `TensorDyn`. This keeps the tensor
types in `edgefirst-tensor` unaware of codec internals.

### `&[u8]` as the Hot Path

The decode pipeline takes `&[u8]` as input — the most common case (memory-
mapped files, network buffers, camera frames). `Read`-based wrappers buffer
into `ImageDecoder.input_buffer` before delegating to the `&[u8]` path.

### Strided Output

Decoders write row-by-row using the tensor's `effective_row_stride()`. This
supports tensors with GPU pitch alignment padding (e.g., 64-byte alignment
for Mali DMA-BUF import). The stride gap bytes are untouched.

```
Tensor buffer layout (1280×720 Grey, 64-byte aligned stride = 1280):
┌──────────────────────────────┬────┐
│ row 0: 1280×1 = 1280 bytes  │ 0  │  ← no padding (1280 % 64 == 0)
├──────────────────────────────┼────┤
│ row 1: 1280×1 = 1280 bytes  │ 0  │
├──────────────────────────────┼────┤
│ ...                          │    │
└──────────────────────────────┴────┘
```

For misaligned widths (e.g., 641 pixels Grey = 641 bytes, padded to 704):
```
┌────────────────────────┬──────────┐
│ row 0: 641 bytes       │ 63 pad   │  ← stride = 704
├────────────────────────┼──────────┤
│ row 1: 641 bytes       │ 63 pad   │
└────────────────────────┴──────────┘
```

### Works Best with `ImageProcessor::create_image()`

While `ImageLoad` works with any `Tensor<T>` or `TensorDyn`, optimal
performance requires tensors allocated by `ImageProcessor::create_image()`:

- **DMA-BUF backing**: zero-copy path to GPU for `convert()`, and the V4L2
  zero-copy decode path
- **PBO backing**: when GL is the active transfer path
- **GPU pitch alignment**: row stride padded for Mali DMA-BUF import

Free-standing `Tensor::new()` or `Tensor::image()` works but:
- Cannot produce PBO tensors (requires GL context)
- May not have GPU-aligned pitch (works, but `convert()` may use CPU path)
- Is never eligible for V4L2 zero-copy (which requires a DMA-backed tensor)

### Tensor Dimensions After Decode

When a smaller image (e.g., 640×480) is decoded into a larger tensor
(e.g., 1920×1080), the decoder reconfigures the tensor's logical shape to the
decoded dimensions within the same physical allocation. `ImageInfo` reports the
decoded dimensions. Callers use `Crop` with `ImageProcessor::convert()` to
process the decoded region and apply any reported EXIF orientation:

```rust
let info = tensor.load_image(&mut decoder, &bytes)?;
let rot = Rotation::from_degrees_clockwise(info.rotation_degrees as usize);
let flip = if info.flip_horizontal { Flip::Horizontal } else { Flip::None };
processor.convert(&tensor, &mut dst, rot, flip,
    Crop::new(0, 0, info.width, info.height))?;
```

## Decode Pipeline

### JPEG Decode Flow

The custom baseline JPEG decoder processes images through these stages:

1. **Marker parsing** (`markers.rs`): parse SOF0, DQT, DHT, DRI, SOS, APP1
   segments. Build Huffman tables, quantisation tables, and read the EXIF
   orientation tag (reported only).
2. **Native format + capacity** (`mod.rs`): derive the native format
   (`native_format()`: 3 components → `Nv12`, 1 → `Grey`) and reconfigure the
   destination tensor via `configure_image()`, which errors with
   `InsufficientCapacity` if the image exceeds the tensor's allocation.
3. **Hardware attempt** (Linux + `v4l2`): try `V4l2Probe::try_decode()`. On
   success it returns the `ImageInfo`; on `Fallback` (no device / unsupported
   capture format / transient hardware failure) the CPU path below runs.
4. **MCU decode loop** (`mcu.rs`): for each MCU row:
   a. **Huffman decode** (`huffman.rs`): 11-bit lookahead LUT decodes DC/AC
      coefficients with dequantisation fused into the decode step.
   b. **IDCT** (`idct/`): two-pass Loeffler 8×8 IDCT with a DC-only fast path
      converts frequency coefficients → spatial component samples.
   c. **Native write** (`mcu.rs`): `write_grey_rows` copies the luma plane for
      `Grey`; `write_nv12_rows` writes the Y plane and the interleaved Cb/Cr
      plane, downsampling chroma to 4:2:0 via `avg_block` (passthrough for an
      already-4:2:0 source, block-average for 4:2:2 / 4:4:4 / 4:4:0). Both write
      at the tensor's `effective_row_stride()`.
5. **Return** `ImageInfo` with decoded dimensions, native format, row stride,
   and the reported EXIF orientation.

**Key optimisations:**
- `JpegDecoderState` persists across frames — `McuScratch` buffers grow to the
  high-water mark and are reused. After the first decode at a given resolution,
  the JPEG decoder performs zero heap allocations.
- Dequantisation is fused into Huffman decode: `decode_block()` multiplies each
  coefficient by the quant table entry during decode, not as a separate pass.
- DC-only IDCT fast path: when all 63 AC coefficients are zero, the IDCT reduces
  to a constant fill (single multiply + shift).
- Function pointer dispatch for the IDCT: selected once at init based on CPU
  feature detection (NEON on AArch64, SSE4.1 > SSE2 on x86-64, scalar fallback).

### NV12 Output Path

For a colour JPEG the decoder produces `Nv12` directly, skipping any
YCbCr→RGB conversion:

- The Y plane is copied from the IDCT luma output at the tensor's row stride.
- The Cb/Cr components are downsampled to 4:2:0 (`avg_block`) and interleaved
  pair-wise into the UV plane that follows the Y plane.

This is the codec's only colour output. The chroma carries the source's JFIF
(BT.601 full-range) colorimetry; mapping that to RGB is the downstream
`convert()` step's responsibility, not the codec's.

### Chroma Subsampling Support

The Cb/Cr components are decoded at their native sampling and downsampled to
4:2:0 when written into `Nv12`:

| Source sampling | Description              | Downsample to 4:2:0      |
|-----------------|--------------------------|--------------------------|
| 4:2:0           | Horizontal + vertical 2× | Passthrough (`avg_block` 1×1) |
| 4:2:2           | Horizontal 2×            | Vertical 2× block-average |
| 4:4:0           | Vertical 2×              | Horizontal 2× block-average |
| 4:4:4           | No subsampling           | 2×2 block-average        |
| Greyscale       | Single component         | No chroma (`Grey` output) |

### IDCT SIMD Kernels

The IDCT is the only SIMD-dispatched kernel remaining in the CPU path (colour
conversion and chroma upsampling were removed when the codec moved to native
output — colour now belongs to `convert()`, and chroma is a simple block-average
into 4:2:0).

**NEON (AArch64)** — selected via
`std::arch::is_aarch64_feature_detected!("neon")`:

| Kernel   | Strategy                                          | Throughput              |
|----------|---------------------------------------------------|-------------------------|
| **IDCT** | 4-wide Loeffler butterfly with `int32x4_t`, 4×4 transpose via `vzip`, DC-only fills 8 bytes via `vdup`/`vst1` | 4 cols/rows per iteration |

**x86-64** — tiered dispatch SSE4.1 > SSE2 > scalar via
`is_x86_feature_detected!()`:

| Kernel   | Tier   | Strategy                                          | Throughput              |
|----------|--------|---------------------------------------------------|-------------------------|
| **IDCT** | SSE4.1 | 4-wide Loeffler with native `_mm_mullo_epi32`, `_mm_min_epi32`/`_mm_max_epi32` clamp | 4 cols/rows per iteration |
| **IDCT** | SSE2   | 4-wide Loeffler with emulated `mullo_epi32` (4 instructions), comparison-based clamp | 4 cols/rows per iteration |

SSE4.1 improvements over SSE2: native `_mm_mullo_epi32` replaces 4-instruction
emulation (~30% fewer IDCT instructions); `_mm_min_epi32`/`_mm_max_epi32`
replaces the 5-instruction comparison-based clamp with a 2-instruction
branchless clamp.

### PNG Decode Flow

1. Parse PNG headers via `zune-png` → dimensions, colorspace, bit depth, and the
   `eXIf` orientation tag (reported only).
2. Map the colorspace to the native `PixelFormat` (Luma/LumaA → `Grey`, RGB →
   `Rgb`, RGBA → `Rgba`) and reconfigure the tensor via `configure_image()`.
3. Choose a decode strategy based on the target type and source bit depth:
   - **u8/i8 targets**: `decode_into(&mut [u8])` — fast u8 path with optional
     XOR for i8 and LumaA→Grey alpha stripping.
   - **u16/i16/f32 targets**: `decode()` → `DecodingResult`, preserving native
     16-bit data from 16-bit PNGs.
4. Row-copy from decoded data → tensor buffer at stride offsets, with pixel-type
   conversion via `from_u8()` / `from_u16()` per source depth.
5. Return `ImageInfo` with decoded dimensions and reported EXIF orientation.

### Format Auto-Detection

The decoder inspects magic bytes:
- `FF D8 FF` → JPEG
- `89 50 4E 47` → PNG
- Otherwise → `CodecError::InvalidData`

## V4L2 Hardware Backend

On Linux, the `v4l2` feature (default-on) adds a mem2mem hardware JPEG-decode
backend in `jpeg/v4l2/`. It drives any device that exposes a JPEG decoder
through the standard V4L2 M2M API; the lead target is i.MX `mxc-jpeg`
(`/dev/video11`) but nothing about the node, driver, or output format is
hardcoded.

### Discovery (capability-based, portable)

`V4l2Probe` is probed lazily on the first JPEG decode and at most once per
`ImageDecoder`. `device.rs`:

1. Honours `EDGEFIRST_DISABLE_V4L2=1` (skip → CPU) and
   `EDGEFIRST_CODEC_V4L2_DEVICE=<path>` (probe only that node); otherwise
   enumerates `/dev/video*`.
2. `VIDIOC_QUERYCAP` requires `V4L2_CAP_STREAMING` and a multi-planar M2M
   capability. (Single-planar-only M2M devices currently fall back to CPU.)
3. `VIDIOC_ENUM_FMT` on the OUTPUT (coded) queue must advertise
   `V4L2_PIX_FMT_JPEG` — this, not the device name, is the "is this a JPEG
   decoder" test. Nodes without JPEG are skipped (so camera, HEVC/H264, and
   ISI-M2M nodes on the same board are correctly rejected).

### Persistent streaming session

After the first decode at a given geometry, both OUTPUT and CAPTURE queues stay
`STREAMON` with their buffers set up, so subsequent same-geometry decodes pay
only the per-frame `QBUF`/`DQBUF` cost. A geometry change rebuilds the stream.

Per-image lifecycle: copy/queue the JPEG on OUTPUT → bounded `DQEVENT` drain
after `SOURCE_CHANGE` (an unbounded drain hangs — capped at `MAX_EVENTS`) →
`G_FMT(CAPTURE)` for the driver-chosen format and MCU(16)-rounded dimensions →
queue CAPTURE → poll/`DQBUF` → copy out (cropped to the logical image) → recycle
the OUTPUT buffer.

### Capture paths

`format.rs::classify()` maps the driver's CAPTURE FourCC to a `CapKind`:
- `NV12`/`NV12M` → `Nv12`, `GREY` → `Grey`: copied straight into the tensor.
- `YUV24` (packed 4:4:4) → reported but the native-NV12 write for it is left to
  the CPU decoder (`Fallback`).

Two ways the decoded planes reach the tensor:
- **Zero-copy (DMABUF):** when the destination is a DMA tensor with
  MCU(16)-aligned dimensions and the driver accepts a single-plane contiguous
  CAPTURE (`V4L2_PIX_FMT_NV12`/`GREY`) at the tensor pitch, the tensor's dmabuf
  fd is imported as the CAPTURE buffer (`V4L2_MEMORY_DMABUF`) and the hardware
  decodes straight in — no copy. (The driver's *default* `NV12M` 2-plane layout
  does **not** compose into one tensor buffer, so a single-plane format is
  requested via `S_FMT`.)
- **MMAP + copy:** otherwise the driver buffers are mapped and the `Nv12`/`Grey`
  planes are copied, cropped to the logical dimensions, into the destination.

### Fallback & recovery

`try_decode()` returns `Ok(Some(info))` on a hardware decode, `Ok(None)` when no
device is available, or `Err(V4l2Decode::Fallback(reason))` on a transient
failure (the device is reset and the caller transparently runs the CPU decoder
on the same tensor). `Fatal` is reserved for deterministic input errors. After
`MAX_CONSECUTIVE_FAILURES` (8) the device is demoted to CPU for the rest of the
session (circuit breaker).

### ABI note

The raw UAPI structs in `ioctl.rs` must match the kernel's `sizeof`, which a
compile-time `size_of` assert checks against our arithmetic, **not** the
kernel's. `v4l2_format` is **208 bytes** (its union contains `v4l2_window`,
whose pointers force 8-byte alignment and 4 bytes of padding after `type`); a
wrong size yields the wrong ioctl request number → `ENOTTY` → a silent CPU
fallback that makes parity tests pass trivially. On-target `strace` is the only
reliable verification of the raw ioctl ABI — see `TESTING.md`.

## Tracing Spans

`ImageDecoder::decode_into` (and the trait-method `Tensor::load_image`) emits a
[`tracing::trace_span!`] tree describing each phase of the JPEG/PNG decode.
Spans are captured by
[`edgefirst_hal::trace::start_tracing`](https://github.com/EdgeFirstAI/hal/blob/main/crates/hal/src/trace.rs)
into Chrome JSON for Perfetto. The cost when no subscriber is active is a single
relaxed atomic load per call site.

### Naming convention

Span names follow `<crate>.<function>[.<operation>]`:

- **`<crate>.<function>`** — top-level span: the public function the user
  invoked. The codec exposes format-specific entry points (`codec.decode_jpeg`,
  `codec.decode_png`) selected automatically from the magic bytes.
- **`<crate>.<function>.<operation>`** — meaningful internal work
  (`codec.decode_jpeg.parse_markers`, `codec.decode_jpeg.mcu_loop`).

### Span tree

```text
codec.decode_jpeg                                       [user-facing fn]
│ fields: dtype = "u8", n_bytes
│
├── codec.decode_jpeg.parse_markers                     ← parse SOF0/DQT/DHT/DRI/SOS/APP1, read EXIF orientation
└── codec.decode_jpeg.mcu_loop                          ← Huffman + IDCT + native Grey/NV12 write (CPU path only)

codec.decode_png                                        [user-facing fn]
│ fields: dtype, n_bytes
│
└── codec.decode_png.zune_decode                        ← delegate to zune-png
    field: path = "u8" | "native_u16"
```

When the V4L2 hardware backend handles a JPEG, the `mcu_loop` span is absent —
the decode happens in the driver, not the CPU loop.

### What each span measures

| Span                              | What is happening inside | Reference equivalent |
|-----------------------------------|--------------------------|----------------------|
| `codec.decode_jpeg`               | Full JPEG decode: marker parsing then either the V4L2 hardware decode or the CPU MCU loop, writing native `Nv12`/`Grey`. | Baseline JPEG decode per ITU T.81. |
| `codec.decode_jpeg.parse_markers` | Walk the JPEG byte stream once: parse SOF0, DQT, DHT, DRI, SOS, and APP1 (EXIF) segments; build Huffman LUTs and quant tables; read the EXIF orientation tag. | Equivalent to libjpeg's `jpeg_read_header` + DHT/DQT table builds. |
| `codec.decode_jpeg.mcu_loop`      | The CPU core loop: per MCU row, Huffman-decode + dequant-fuse → two-pass Loeffler IDCT (scalar / NEON / SSE4.1 / SSE2) → native `Grey`/`Nv12` write with 4:2:0 chroma downsample, strided into the tensor. Allocation-free after warmup. Absent when hardware decodes. | Equivalent to libjpeg's `jpeg_read_scanlines` loop, IDCT handwritten and SIMD-dispatched. |
| `codec.decode_png`                | Full PNG decode: header parse, native or 8-bit `zune-png` decode, native-format row write. | PNG decode per ISO/IEC 15948 (RFC 2083). |
| `codec.decode_png.zune_decode`    | The bulk of PNG cost: zlib inflate + PNG filter reversal inside [`zune-png`](https://docs.rs/zune-png). `path = "u8"` is the strided-output fast path; `path = "native_u16"` preserves 16-bit-per-channel PNGs for u16/i16/f32 tensor targets. | Equivalent to libpng's `png_read_image`. |

[`tracing::trace_span!`]: https://docs.rs/tracing/latest/tracing/macro.trace_span.html

## Supported Pixel Formats

| Output Format | JPEG | PNG  | Notes                              |
|---------------|------|------|------------------------------------|
| Nv12          | ✓    | —    | Colour JPEG: Y + interleaved UV (4:2:0) |
| Grey          | ✓    | ✓    | Single luma component              |
| Rgb           | —    | ✓    | Native PNG RGB                     |
| Rgba          | —    | ✓    | Native PNG RGBA                    |

`Rgb`/`Rgba`/`Bgra` from a JPEG, and any resize/rotation, come from
`ImageProcessor::convert()` applied to the native decode — they are deliberately
not codec responsibilities.

## Supported Source Features

The codec implements a **strict subset** of the JPEG and PNG specifications.
Inputs that fall outside the subset surface a typed
`CodecError::Unsupported(UnsupportedFeature)`. See the per-feature table in
[`README.md`](README.md#decoder-limitations) for the full matrix and the typed
error variant that each rejected case carries.

The codec does **not** transparently fall back to another *software* decoder for
unsupported inputs and does **not** transcode them. (The V4L2→CPU fallback is a
different mechanism: both decoders implement the same strict subset and produce
identical native output; the fallback only changes *where* the supported decode
runs.) The contract is "accept this strict subset; reject everything else with a
precise typed error."

## Data Type Support

JPEG decodes to `u8` only (native `Nv12`/`Grey`); a non-`u8` destination is
rejected with `CodecError::UnsupportedDtype`. PNG supports the full set:

| Type  | PNG (8-bit source)   | PNG (16-bit source) |
|-------|----------------------|---------------------|
| `u8`  | Direct copy          | `>> 8`              |
| `u16` | `* 257` scaling      | Direct copy         |
| `i8`  | XOR 0x80             | `(>> 8) XOR 0x80`  |
| `i16` | `* 257` then XOR     | XOR 0x8000          |
| `f32` | `/ 255.0`            | `/ 65535.0`         |

### XOR Trick for Signed Types

Signed integer decoding uses a bit-flip to convert unsigned pixel data into the
signed range, which is the standard approach for ML quantization:

- **i8**: `(u8_value ^ 0x80) as i8` — maps `0→-128`, `128→0`, `255→127`
- **i16**: `(u16_value ^ 0x8000) as i16` — maps `0→-32768`, `32768→0`, `65535→32767`

### u16 Scaling from u8

When 8-bit PNG data is decoded into `u16`, each byte is scaled to the full
16-bit range: `u8_value as u16 * 257`. This maps `0→0`, `128→32896`, `255→65535`
exactly (257 = 0x0101).

## Scratch Buffer Strategy

### JPEG (`JpegDecoderState`)

The custom JPEG decoder uses `JpegDecoderState`, which persists across frames:

- `mcu_scratch` (`McuScratch`): per-component IDCT output for one MCU row band,
  grown to the high-water mark and reused. After the first decode at a given
  resolution, subsequent CPU JPEG decodes perform **zero heap allocations**.
- `v4l2` (Linux + feature): the lazily-probed hardware backend, holding the
  persistent streaming session and mapped buffers across frames.

There is no EXIF rotation scratch — the codec reads the orientation tag and
reports it without allocating a rotation workspace.

### PNG (`zune-png`)

PNG decoding uses `zune-png`, which allocates internal decoder state on each
call. The edgefirst-codec PNG layer reuses `ImageDecoder.input_buffer` for
`Read`-based input, but the zune-png library itself allocates per-frame.

### Allocation Sources by Layer

| Layer                    | After Warmup     | Notes                        |
|--------------------------|------------------|------------------------------|
| JPEG `McuScratch`        | No allocations   | Grows to high-water mark     |
| JPEG Huffman/quant tables| No allocations   | Rebuilt from marker data     |
| JPEG IDCT workspace      | No allocations   | Stack-allocated `[i32; 64]`  |
| JPEG native row write    | No allocations   | Strided into pre-allocated tensor |
| V4L2 streaming session   | No allocations   | Buffers set up once per geometry |
| zune-png `decode()`      | 1 `Vec` / call   | Returns owned `Vec<u16/u8>`  |
| zune-png `decode_into()` | ~3 `brk` / call  | Internal filter state        |
