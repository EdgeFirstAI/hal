# EdgeFirst Codec

Zero-allocation image decoding framework for pre-allocated tensor buffers
in real-time vision pipelines.

## Overview

`edgefirst-codec` decodes JPEG and PNG images directly into pre-allocated
`Tensor<T>` or `TensorDyn` buffers, supporting strided memory layouts
(GPU pitch-aligned DMA-BUF, PBO). This eliminates per-frame allocations
in the hot loop — the primary design goal.

JPEG decoding uses a custom from-scratch baseline decoder with reusable
state, achieving zero heap allocations after the first decode at each
resolution. SIMD-optimized kernels (NEON on AArch64, SSE2 on x86-64) are
selected automatically at init via dynamic dispatch. PNG decoding uses
`zune-png`.

## Quick Start

```rust
use edgefirst_codec::{ImageDecoder, DecodeOptions, ImageLoad};
use edgefirst_tensor::{Tensor, PixelFormat, TensorMemory};

// Allocate once at init (prefer ImageProcessor::create_image() for DMA/PBO)
let mut tensor = Tensor::<u8>::image(1920, 1080, PixelFormat::Rgb,
    Some(TensorMemory::Mem)).unwrap();
let mut decoder = ImageDecoder::new();

// Decode in the hot loop — zero allocations after warmup for JPEG
let jpeg_bytes = std::fs::read("frame.jpg").unwrap();
let info = tensor.load_image(&mut decoder, &jpeg_bytes,
    &DecodeOptions::default()).unwrap();
println!("Decoded {}x{} {:?}", info.width, info.height, info.format);
```

## Recommended Pattern

For maximum performance, use tensors allocated by
`ImageProcessor::create_image()`:

```rust,ignore
use edgefirst_image::{ImageProcessor, ImageProcessorTrait, Crop};
use edgefirst_codec::{ImageDecoder, DecodeOptions, ImageLoad};

let mut processor = ImageProcessor::new()?;
let mut src = processor.create_image(1920, 1080, PixelFormat::Rgb,
    DType::U8, None)?;
let mut dst = processor.create_image(640, 640, PixelFormat::Rgb,
    DType::U8, None)?;
let mut decoder = ImageDecoder::new();

loop {
    let bytes = capture_frame();
    let info = src.load_image(&mut decoder, &bytes,
        &DecodeOptions::default())?;
    processor.convert(&src, &mut dst, Rotation::None, Flip::None,
        Crop::new(0, 0, info.width, info.height))?;
}
```

Benefits of `ImageProcessor::create_image()` tensors:
- **DMA-BUF backing** for zero-copy GPU import
- **PBO backing** when OpenGL is the active transfer path
- **GPU pitch alignment** (64-byte for Mali compatibility)

Free-standing tensors work but cannot use PBO and may lack GPU-aligned pitch.

## Supported Formats

| Format | Input  | Output Formats           |
|--------|--------|--------------------------|
| JPEG   | `&[u8]`| RGB, RGBA, Grey, BGRA    |
| PNG    | `&[u8]`| RGB, RGBA, Grey, BGRA    |

## Data Types

| Type  | Support | Notes                              |
|-------|---------|------------------------------------|
| `u8`  | ✓       | Direct copy (identity)             |
| `u16` | ✓       | Scaled `* 257` from 8-bit; native from 16-bit PNG |
| `i8`  | ✓       | XOR 0x80 sign-bit flip             |
| `i16` | ✓       | XOR 0x8000 sign-bit flip           |
| `f32` | ✓       | Normalized to [0.0, 1.0]           |

## API Reference

### `ImageDecoder`

Reusable decoder with internal scratch buffers. Create once, reuse across
frames — scratch buffers amortize after the first decode.

### `ImageLoad` Extension Trait

- `load_image(&mut self, decoder, data, opts)` — decode from `&[u8]`
- `load_image_read(&mut self, decoder, reader, opts)` — decode from `Read`
- `load_image_file(&mut self, decoder, path, opts)` — decode from file path

### `DecodeOptions`

- `format`: Output pixel format (`None` = native from file)
- `scale_denom`: JPEG IDCT downscale (1/2/4/8, default 1)
- `apply_exif`: Apply EXIF orientation (default true)

### `ImageInfo`

Returned by all decode methods with actual decoded dimensions:
- `width`, `height`: Decoded image size
- `format`: Output pixel format
- `row_stride`: Row stride in bytes

## License

Apache-2.0
