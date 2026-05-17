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

`edgefirst-codec` depends only on `edgefirst-tensor` plus `zune-png`
(for PNG decoding) and `kamadak-exif` (for EXIF orientation). JPEG
decoding uses a custom from-scratch decoder with no external dependencies.
The crate has no dependency on `edgefirst-image` or any GPU libraries,
keeping the dependency graph clean.

## Module Map

| Module       | Purpose                                         |
|--------------|-------------------------------------------------|
| `lib.rs`     | Crate root, public re-exports                   |
| `error.rs`   | `CodecError` enum with capacity/dtype/format/IO |
| `pixel.rs`   | `ImagePixel` trait (u8, u16, i8, i16, f32)      |
| `options.rs` | `DecodeOptions` and `ImageInfo` structs         |
| `decoder.rs` | `ImageDecoder` struct with `JpegDecoderState`   |
| `traits.rs`  | `ImageLoad` extension trait for Tensor/TensorDyn|
| `jpeg/`      | Custom baseline JPEG decoder (see below)        |
| `png.rs`     | PNG decode with format conversion and native 16-bit support |

### JPEG Module Map (`jpeg/`)

| Module           | Purpose                                              |
|------------------|------------------------------------------------------|
| `mod.rs`         | `JpegDecoderState`, `decode_jpeg_into<T>()`, EXIF    |
| `types.rs`       | `Component`, `SamplingFactor`, `ImageHeader`, `QuantTable`, `ZIGZAG` |
| `markers.rs`     | SOF/SOS/DQT/DHT/DRI/APP marker parsing               |
| `bitstream.rs`   | 64-bit bit buffer with FF/00 byte-stuffing, bulk refill |
| `huffman.rs`     | 9-bit lookahead Huffman LUT, `decode_block()` with dequant fusion |
| `idct/mod.rs`    | IDCT dispatcher (scalar/NEON/SSE2 selection via function pointers) |
| `idct/scalar.rs` | Two-pass Loeffler 8×8 IDCT with DC-only fast path    |
| `idct/neon.rs`   | NEON 8×8 IDCT: 4-wide Loeffler butterfly, 4×4 transpose, DC-only fill |
| `idct/sse2.rs`   | SSE2 8×8 IDCT: 4-wide Loeffler butterfly, comparison-based clamp |
| `color/mod.rs`   | Color conversion dispatcher                           |
| `color/scalar.rs`| BT.601 full-range YCbCr→RGB/RGBA/BGRA/Grey           |
| `color/neon.rs`  | NEON YCbCr→RGB/RGBA/BGRA: 8-pixel SIMD with vst3/vst4 |
| `color/sse2.rs`  | SSE2 YCbCr→RGB/RGBA/BGRA: 8-pixel SIMD with unpack interleave |
| `upsample/mod.rs`| Chroma upsample dispatcher                            |
| `upsample/scalar.rs` | Bilinear 3:1 blend for horizontal 2× upsampling |
| `upsample/neon.rs`   | NEON horizontal 2× upsample: widening multiply-accumulate |
| `upsample/sse2.rs`   | SSE2 horizontal 2× upsample: 16-bit multiply with pack |
| `mcu.rs`         | MCU decode loop, `McuScratch`, strided output, NV12 path |

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

The custom baseline JPEG decoder processes images through these stages:

1. **Marker parsing** (`markers.rs`): Parse SOF0, DQT, DHT, DRI, SOS, APP1
   segments. Build Huffman tables, quantisation tables, and extract EXIF data.
2. **Capacity validation**: Verify tensor dimensions ≥ decoded image size
   (accounting for EXIF rotation if enabled).
3. **MCU decode loop** (`mcu.rs`): For each MCU row:
   a. **Huffman decode** (`huffman.rs`): 9-bit lookahead LUT decodes DC/AC
      coefficients with dequantisation fused into the decode step.
   b. **IDCT** (`idct/`): Two-pass Loeffler 8×8 IDCT with DC-only fast
      path converts frequency coefficients → spatial pixel values.
   c. **Chroma upsample** (`upsample/`): Bilinear 3:1 blend expands
      subsampled Cb/Cr channels to full resolution.
   d. **Color conversion** (`color/`): BT.601 full-range YCbCr→RGB/RGBA/
      BGRA/Grey conversion with clamping.
   e. **Strided output**: Write converted pixels to tensor buffer at
      `effective_row_stride()` offsets.
4. **EXIF rotation/flip**: Apply orientation transform in-place (if enabled).
5. **Type conversion**: For non-u8 targets, convert pixel data via
   `ImagePixel::from_u8()` with fast paths for i8 (XOR 0x80).
6. **Return** `ImageInfo` with decoded dimensions.

**Key optimisations:**
- `JpegDecoderState` persists across frames — `McuScratch` buffers grow
  to the high-water mark and are reused. After the first decode at a given
  resolution, the JPEG decoder performs zero heap allocations.
- Dequantisation is fused into Huffman decode: `decode_block()` multiplies
  each coefficient by the quant table entry during decode, not as a
  separate pass.
- DC-only IDCT fast path: when all 63 AC coefficients are zero, the IDCT
  reduces to a constant fill (single multiply + shift).
- Function pointer dispatch for IDCT/color/upsample: selected once at init
  based on CPU feature detection (NEON on AArch64, SSE2 on x86-64, scalar fallback).

### NEON SIMD Kernels (AArch64)

On AArch64, the decoder uses NEON intrinsics for the three hot-path kernels.
Each kernel is selected via `std::arch::is_aarch64_feature_detected!("neon")`
at init time.

| Kernel       | Strategy                                          | Throughput    |
|--------------|---------------------------------------------------|---------------|
| **IDCT**     | 4-wide Loeffler butterfly with int32x4_t, 4×4 transpose via vzip, DC-only fills 8 bytes via vdup/vst1 | 4 cols/rows per iteration |
| **Color**    | 7-bit fixed-point YCbCr→RGB/RGBA/BGRA, vmovl widening, vrshrq rounding shift, vqmovun saturation, vst3/vst4 interleaved store | 8 pixels per iteration |
| **Upsample** | Widening bilinear 3:1 blend via vmulq_n_u16, interleaved output via vst2 | 8→16 samples per iteration |

### SSE2 SIMD Kernels (x86-64)

On x86-64, the decoder uses SSE2 intrinsics for the same three kernels.
SSE2 is guaranteed on all x86-64 CPUs but dispatch still uses
`is_x86_feature_detected!("sse2")` for pattern consistency.

| Kernel       | Strategy                                          | Throughput    |
|--------------|---------------------------------------------------|---------------|
| **IDCT**     | 4-wide Loeffler butterfly with `__m128i`, emulated `mullo_epi32` (SSE2 lacks native i32 multiply), comparison-based clamp to [0,255] | 4 cols/rows per iteration |
| **Color**    | 7-bit fixed-point YCbCr→RGB/RGBA/BGRA, `_mm_unpacklo_epi8` widening, `_mm_srai_epi16` shift, `_mm_packus_epi16` saturation, unpack-interleave for RGBA/BGRA, temp-buffer scatter for RGB | 8 pixels per iteration |
| **Upsample** | 16-bit bilinear 3:1 blend via `_mm_mullo_epi16`, `_mm_packus_epi16` narrow, `_mm_unpacklo_epi8` interleave | 16→32 samples per iteration |

SSE2 notes:
- `_mm_mullo_epi32` requires SSE4.1; the IDCT emulates it via two `_mm_mul_epu32` +
  shuffle/unpack (low 32 bits of unsigned multiply match signed multiply).
- RGB 3-channel interleave uses a temp buffer + scalar scatter (SSE2 has no byte
  shuffle like SSSE3's `_mm_shuffle_epi8`); RGBA/BGRA use native `_mm_unpacklo_epi8`
  interleaving which is actually faster than RGB.

All kernels have scalar tails for remainder elements when the width is not
a multiple of the SIMD width.

### NV12 Output Path

For NV12 output, the decoder skips YCbCr→RGB color conversion entirely:
- Y plane is copied directly from the IDCT output buffer
- Cb and Cr planes are interleaved pair-wise into the UV plane

This path is faster than RGB/RGBA because it avoids the fixed-point color
conversion entirely. It is intended for hardware video encoders and GPU
pipelines that consume NV12 natively. EXIF rotation is not supported for
NV12 output.

### JPEG Decoder Architecture

```
JpegDecoderState
├── McuScratch (reusable across frames)
│   ├── component_bufs: Vec<Vec<u8>>   — per-component IDCT output
│   ├── cb_row / cr_row: Vec<u8>       — upsampled chroma rows
│   └── output_row: Vec<u8>            — color-converted output row
└── exif_scratch: Vec<u8>              — EXIF rotation workspace
```

The MCU loop processes one MCU row at a time:
1. Decode all blocks (Y, Cb, Cr) into `component_bufs`
2. For each pixel row in the MCU row:
   - Upsample chroma into `cb_row`/`cr_row`
   - Color-convert Y+Cb+Cr → `output_row`
   - Copy `output_row` → tensor at strided offset

### Chroma Subsampling Support

| Sampling | Description     | H/V Ratios | Upsample Path         |
|----------|-----------------|------------|------------------------|
| 4:4:4    | No subsampling  | 1:1 / 1:1  | Direct (no upsample)  |
| 4:2:2    | Horizontal 2×   | 2:1 / 1:1  | `upsample_h2()`       |
| 4:2:0    | Horizontal + Vertical 2× | 2:1 / 2:1 | `upsample_h2()` + row duplication |
| Greyscale| Single component| N/A        | `grey_copy()`          |

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
| NV12          | ✓    | —    | Y plane + interleaved UV (4:2:0)|

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

### JPEG (`JpegDecoderState`)

The custom JPEG decoder uses `JpegDecoderState` which persists across frames.
The internal `McuScratch` buffers grow to the high-water mark and are reused.
After the first decode at a given resolution, subsequent JPEG decodes perform
**zero heap allocations** in the entire decode path.

**Allocation-free after warmup:**
- `McuScratch` component buffers, chroma rows, output row
- Huffman table lookups (tables are rebuilt from marker data each frame using
  pre-allocated `Vec` storage)
- IDCT workspace (stack-allocated `[i32; 64]`)
- Bitstream reader (borrows input `&[u8]`)
- Row-copy and stride padding logic
- Pixel type conversion (u8→u16, u8→i8 XOR, u8→f32)

**EXIF rotation** (`exif_scratch`) uses a reusable `Vec<u8>` that grows to
the high-water mark. `kamadak-exif::Reader::read_raw()` allocates on each
call — disable with `DecodeOptions::with_exif(false)` in the hot loop if
the application handles orientation separately.

### PNG (`zune-png`)

PNG decoding uses `zune-png` which allocates internal decoder state on each
call. The edgefirst-codec PNG layer reuses `ImageDecoder.input_buffer` for
`Read`-based input but the zune-png library itself allocates per-frame.

### Allocation Sources by Layer

| Layer                    | After Warmup     | Notes                        |
|--------------------------|------------------|------------------------------|
| JPEG `McuScratch`        | No allocations   | Grows to high-water mark     |
| JPEG Huffman/quant tables| No allocations   | Rebuilt from marker data     |
| JPEG IDCT workspace      | No allocations   | Stack-allocated `[i32; 64]`  |
| Row-copy / stride        | No allocations   | Operates on pre-allocated buffers |
| Pixel conversion         | No allocations   | In-place or element-wise     |
| EXIF reader              | 1 `Vec` / call   | `to_vec()` on EXIF data; skip with `apply_exif(false)` |
| zune-png `decode()`      | 1 `Vec` / call   | Returns owned `Vec<u16/u8>`  |
| zune-png `decode_into()` | ~3 `brk` / call  | Internal filter state        |
