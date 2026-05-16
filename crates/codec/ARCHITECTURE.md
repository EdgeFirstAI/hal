# EdgeFirst Codec Architecture

## Overview

The `edgefirst-codec` crate provides image decoding into pre-allocated
tensor buffers. It is designed for real-time vision pipelines where the
anti-pattern of allocating new output buffers on every frame must be
avoided.

The core principle: **allocate once at init, decode in the hot loop**.

## Crate Position in the Workspace

```
edgefirst-tensor ← edgefirst-codec ← edgefirst-image (re-export)
                                    ← edgefirst-hal (re-export)
                                    ← crates/python (bindings)
```

`edgefirst-codec` depends only on `edgefirst-tensor` plus format-specific
decode libraries (`zune-jpeg`, `zune-png`). It has no dependency on
`edgefirst-image` or any GPU libraries, keeping the dependency graph clean.

## Module Map

| Module       | Purpose                                         |
|--------------|-------------------------------------------------|
| `lib.rs`     | Crate root, public re-exports                   |
| `error.rs`   | `CodecError` enum with capacity/dtype/format/IO |
| `pixel.rs`   | `ImagePixel` trait (u8, u16, i8, i16, f32)      |
| `options.rs` | `DecodeOptions` and `ImageInfo` structs         |
| `decoder.rs` | `ImageDecoder` struct with scratch buffers      |
| `traits.rs`  | `ImageLoad` extension trait for Tensor/TensorDyn|
| `jpeg.rs`    | JPEG decode with EXIF orientation handling       |
| `png.rs`     | PNG decode with format conversion and native 16-bit support |

## Key Design Decisions

### Standalone `ImageDecoder` Struct

The decoder is a standalone struct rather than being embedded in
`ImageProcessor` or stored in thread-local state. This gives callers
explicit ownership and composability — one decoder per pipeline stage,
no hidden global state.

```rust
let mut decoder = ImageDecoder::new();
// Scratch buffers amortize across calls
loop {
    let info = tensor.load_image(&mut decoder, &bytes, &opts)?;
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
Tensor buffer layout (1280×720 RGB, 64-byte aligned stride = 3840):
┌──────────────────────────────┬────┐
│ row 0: 1280×3 = 3840 bytes  │ 0  │  ← no padding (3840 % 64 == 0)
├──────────────────────────────┼────┤
│ row 1: 1280×3 = 3840 bytes  │ 0  │
├──────────────────────────────┼────┤
│ ...                          │    │
└──────────────────────────────┴────┘
```

For misaligned widths (e.g., 641 pixels × 3 = 1923 bytes, padded to 1984):
```
┌────────────────────────┬──────────┐
│ row 0: 641×3 = 1923    │ 61 pad   │  ← stride = 1984
├────────────────────────┼──────────┤
│ row 1: 641×3 = 1923    │ 61 pad   │
└────────────────────────┴──────────┘
```

### Works Best with `ImageProcessor::create_image()`

While `ImageLoad` works with any `Tensor<T>` or `TensorDyn`, optimal
performance requires tensors allocated by `ImageProcessor::create_image()`:

- **DMA-BUF backing**: Zero-copy path to GPU for `convert()`
- **PBO backing**: When GL is the active transfer path
- **GPU pitch alignment**: Row stride padded for Mali DMA-BUF import

Free-standing `Tensor::new()` or `Tensor::image()` works but:
- Cannot produce PBO tensors (requires GL context)
- May not have GPU-aligned pitch (works, but `convert()` may use CPU path)

### Tensor Dimensions After Decode

When a smaller image (e.g., 640×480) is decoded into a larger tensor
(e.g., 1920×1080), the tensor's physical buffer and shape are unchanged.
`ImageInfo` reports the actual decoded dimensions. Callers use `Crop` with
`ImageProcessor::convert()` to process only the decoded region:

```rust
let info = tensor.load_image(&mut decoder, &bytes, &opts)?;
processor.convert(&tensor, &mut dst, rot, flip,
    Crop::new(0, 0, info.width, info.height))?;
```

## Decode Pipeline

### JPEG Decode Flow

1. Parse JPEG headers via `zune-jpeg` → get dimensions, colorspace
2. Validate tensor capacity ≥ decoded dimensions
3. Optionally read EXIF orientation tag
4. Decode into `ImageDecoder.scratch` (contiguous buffer, reused across frames)
5. Apply EXIF rotation/flip in-place on scratch (if enabled)
6. Row-copy from scratch → tensor buffer at stride offsets, with pixel type
   conversion via `ImagePixel::from_u8()` (JPEG is always 8-bit source)
7. Return `ImageInfo` with decoded dimensions

**Fast paths:** `u8` uses `copy_from_slice`; `i8` uses copy + XOR 0x80 in-place.
Generic path (u16, i16, f32) converts element-by-element via `from_u8()`.

### PNG Decode Flow

1. Parse PNG headers via `zune-png` → get dimensions, colorspace, bit depth
2. Validate tensor capacity ≥ decoded dimensions
3. Choose decode strategy based on target type and source bit depth:
   - **u8/i8 targets**: Use `decode_into(&mut [u8])` — fast u8 path with
     optional XOR for i8
   - **u16/i16/f32 targets**: Use `decode()` → `DecodingResult` which
     preserves native 16-bit data from 16-bit PNGs
4. Convert pixel format if needed (e.g., RGBA→RGB, RGB→Grey)
5. Row-copy from decoded data → tensor buffer at stride offsets with pixel
   type conversion via `from_u8()` or `from_u16()` depending on source depth
6. Return `ImageInfo` with decoded dimensions

### Format Auto-Detection

The decoder inspects magic bytes:
- `FF D8 FF` → JPEG
- `89 50 4E 47` → PNG
- Otherwise → `CodecError::InvalidData`

## Supported Pixel Formats

| Output Format | JPEG | PNG  | Notes                           |
|---------------|------|------|---------------------------------|
| RGB           | ✓    | ✓    | Native JPEG output              |
| RGBA          | ✓    | ✓    | Alpha = 255 for JPEG            |
| Grey          | ✓    | ✓    | Luminance only                  |
| BGRA          | ✓    | ✓    | B/R channel swap from RGB/RGBA  |

## Data Type Support

| Type  | JPEG               | PNG (8-bit source)   | PNG (16-bit source) |
|-------|--------------------|----------------------|---------------------|
| `u8`  | Direct copy        | Direct copy          | `>> 8`              |
| `u16` | `* 257` scaling    | `* 257` scaling      | Direct copy         |
| `i8`  | XOR 0x80           | XOR 0x80             | `(>> 8) XOR 0x80`  |
| `i16` | `* 257` then XOR   | `* 257` then XOR     | XOR 0x8000          |
| `f32` | `/ 255.0`          | `/ 255.0`            | `/ 65535.0`         |

### XOR Trick for Signed Types

Signed integer decoding uses a bit-flip to convert unsigned pixel data into
the signed range, which is the standard approach for ML quantization:

- **i8**: `(u8_value ^ 0x80) as i8` — maps `0→-128`, `128→0`, `255→127`
- **i16**: `(u16_value ^ 0x8000) as i16` — maps `0→-32768`, `32768→0`, `65535→32767`

### u16 Scaling from u8

When JPEG (8-bit) data is decoded into `u16`, each byte is scaled to the full
16-bit range: `u8_value as u16 * 257`. This maps `0→0`, `128→32896`, `255→65535`
exactly (257 = 0x0101).

## Scratch Buffer Strategy

`ImageDecoder` contains a `Vec<u8>` scratch buffer that grows to the
high-water mark and is reused. The edgefirst-codec layer itself does not
perform heap allocations in the decode hot path after warmup — all row-copy
and pixel conversion logic operates on the pre-allocated scratch and tensor
buffers.

However, the underlying decode libraries (zune-jpeg, zune-png) allocate
internal state (Huffman tables, filter state) on each call because they
do not support decoder state reuse. This means the overall decode path
does involve allocations per call, originating in the format libraries.

**Allocation-free layers (after warmup):**
- Scratch buffer management (`Vec::resize` is a no-op at high-water mark)
- Row-copy and stride padding logic
- Pixel type conversion (u8→u16, u8→i8 XOR, u8→f32)
- Format conversion (RGB↔RGBA, Grey↔RGB)

**Allocating layers (per call):**
- `zune-jpeg::JpegDecoder::new()` — internal Huffman/quantization tables
- `zune-png::PngDecoder::decode()` — returns owned `Vec<u16>` for 16-bit PNGs
- `exif::Reader::read_raw()` — when EXIF orientation is enabled

For the PNG native-depth path (`u16`/`f32` targets), zune-png's `decode()`
returns an owned `Vec<u16>` or `Vec<u8>` — this is an additional allocation
per call beyond the decoder state. The u8/i8 targets use `decode_into()` which
writes directly into the reusable scratch buffer, avoiding this extra allocation.
