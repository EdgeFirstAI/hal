# EdgeFirst Codec Architecture

## Overview

The `edgefirst-codec` crate provides zero-allocation image decoding into
pre-allocated tensor buffers. It is designed for real-time vision pipelines
where the anti-pattern of allocating new memory on every frame must be
avoided.

The core principle: **allocate once at init, decode in the hot loop**.

## Crate Position in the Workspace

```
edgefirst-tensor ← edgefirst-codec ← edgefirst-image (re-export)
                                    ← edgefirst-hal (re-export)
                                    ← crates/python (bindings)
```

`edgefirst-codec` depends only on `edgefirst-tensor` plus format-specific
decode libraries (`zune-jpeg`, `zune-png` in Phase 1). It has no dependency
on `edgefirst-image` or any GPU libraries, keeping the dependency graph clean.

## Module Map

| Module       | Purpose                                        |
|--------------|------------------------------------------------|
| `lib.rs`     | Crate root, public re-exports                  |
| `error.rs`   | `CodecError` enum with capacity/dtype/format/IO |
| `pixel.rs`   | `ImagePixel` trait (u8, f32 support)            |
| `options.rs` | `DecodeOptions` and `ImageInfo` structs         |
| `decoder.rs` | `ImageDecoder` struct with scratch buffers      |
| `traits.rs`  | `ImageLoad` extension trait for Tensor/TensorDyn|
| `jpeg.rs`    | JPEG decode with EXIF orientation handling       |
| `png.rs`     | PNG decode with format conversion               |

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

### JPEG Decode Flow (Phase 1 — Zune Shim)

1. Parse JPEG headers via `zune-jpeg` → get dimensions, colorspace
2. Validate tensor capacity ≥ decoded dimensions
3. Optionally read EXIF orientation tag
4. Decode into `ImageDecoder.scratch` (contiguous buffer)
5. Apply EXIF rotation/flip in-place on scratch (if enabled)
6. Row-copy from scratch → tensor buffer at stride offsets
7. For `f32` tensors: row-copy with `u8 → f32` conversion (`/ 255.0`)
8. Return `ImageInfo` with decoded dimensions

### PNG Decode Flow (Phase 1 — Zune Shim)

1. Parse PNG headers via `zune-png` → get dimensions, colorspace, bit depth
2. Validate tensor capacity ≥ decoded dimensions
3. Decode into `ImageDecoder.scratch`
4. Convert pixel format if needed (e.g., RGBA→RGB, RGB→Grey)
5. Row-copy from scratch → tensor buffer at stride offsets
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

| Type  | Status | Conversion              |
|-------|--------|-------------------------|
| `u8`  | ✓      | Direct copy (identity)  |
| `f32` | ✓      | `value / 255.0`         |
| `i8`  | ✗      | Returns `UnsupportedDtype` |
| `f16` | Future | Behind feature gate     |

## Phase 2+ Roadmap

### Phase 2: Custom JPEG Decoder

Replace `zune-jpeg` with a custom Rust JPEG decoder that writes MCU tiles
directly into strided output buffers. Key optimizations:

- NEON-optimized 8×8 IDCT
- NEON-optimized YCbCr→RGB color conversion
- Fused u8→f32 path with `vcvtq_f32_u32`
- Direct YUV/Grey output (skip chroma upsampling)

### Phase 3: Custom PNG Decoder

- Streaming inflate into strided row buffer
- NEON-optimized PNG filter reconstruction

### Phase 4: OpenGL Decode Path

- Feature-gated `opengl` support
- Compute shader IDCT + color conversion on GPU
- Direct decode into PBO/DMA-BUF tensors
