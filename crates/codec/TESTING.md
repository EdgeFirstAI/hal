# EdgeFirst Codec Testing

## Test Organization

### Unit Tests

Inline `#[cfg(test)]` modules in each source file:

| Module                | Tests                                               |
|-----------------------|-----------------------------------------------------|
| `decoder.rs`          | Format detection, capacity validation, error paths  |
| `exif.rs`             | EXIF orientation tag → `(rotation_degrees, flip_horizontal)` mapping; default for no-EXIF |
| `pixel.rs`            | `ImagePixel::from_u8`/`from_u16` for all 5 types     |
| `png.rs`              | Colorspace → `PixelFormat` mapping, u8/u16 correctness |
| `jpeg/bitstream.rs`   | Bit read, byte-stuffing, sign extension             |
| `jpeg/huffman.rs`     | Table build, symbol decode, block decode            |
| `jpeg/markers.rs`     | Minimal JPEG parse, invalid data rejection          |
| `jpeg/mcu.rs`         | `avg_block` chroma downsampling (passthrough + 2×2 average) |
| `jpeg/idct/scalar.rs` | DC-only shortcut, known coefficient IDCT            |
| `jpeg/idct/{neon,sse2,sse41}.rs` | Scalar↔SIMD parity per kernel            |
| `jpeg/v4l2/format.rs` | `classify()` CAPTURE FourCC → `CapKind`             |
| `jpeg/v4l2/mod.rs`    | Backend helpers (geometry/format negotiation logic) |

### Integration Tests

Located in `crates/codec/tests/`:

| File                    | What It Tests                                      |
|-------------------------|----------------------------------------------------|
| `decode_jpeg.rs`        | JPEG → native `Nv12`/`Grey`, pixel accuracy vs the `image` crate, strided decode, capacity/error handling, reuse patterns, non-`u8` rejection |
| `decode_png.rs`         | PNG → Tensor<u8/u16/i8/i16/f32>, native colorspace |
| `decode_tensordyn.rs`   | TensorDyn decode for all dtypes, file path API     |
| `exif_orientations.rs`  | End-to-end EXIF orientation reporting (JPEG + PNG): native dims preserved, `(rotation_degrees, flip_horizontal)` per tag |
| `v4l2_jpeg.rs`          | Device-gated hardware decode: NV12 parity vs CPU, persistent streaming loop, geometry change, DMA-BUF zero-copy (skips cleanly when no V4L2 JPEG device / dma_heap) |

### Test Images

Tests use images from the workspace `testdata/` directory:

| Image               | Format | Size      | Notes                       |
|---------------------|--------|-----------|-----------------------------|
| `zidane.jpg`        | JPEG   | 1280×720  | Standard colour test image (decodes to NV12) |
| `person.jpg`        | JPEG   | 4256×2532 | Large image, capacity test  |
| `giraffe.jpg`       | JPEG   | 640×427   | Smaller image               |
| `jaguar.jpg`        | JPEG   | varies    | Reuse pattern test          |
| `grey.jpg`          | JPEG   | 1024×681  | Greyscale image (decodes to Grey) |
| `zidane.png`        | PNG    | 1280×720  | PNG version of zidane       |
| `zidane_exif_<1-8>.jpg`/`.png` | JPEG/PNG | 1280×720 | Same pixels, varying only the EXIF orientation tag (1–8) |

## Running Tests

```bash
# All codec tests (single-threaded — required for GPU resource safety)
cargo test -p edgefirst-codec -- --test-threads=1

# Or with nextest (process-isolated; preferred in CI)
cargo nextest run -p edgefirst-codec -j 1

# Unit tests only
cargo test -p edgefirst-codec --lib -- --test-threads=1

# Specific test
cargo test -p edgefirst-codec decode_zidane_nv12 -- --test-threads=1
```

**Important:** Always use `--test-threads=1` (or `nextest -j 1`). See the root
[TESTING.md](../../TESTING.md) for the rationale (GPU resource contention).

The `v4l2_jpeg.rs` tests are device-gated: they skip cleanly unless a V4L2 JPEG
M2M device is present, so they pass on any host and exercise hardware only on a
target that has one (e.g. imx95-frdm `/dev/video11`).

## Test Patterns

### Native Format

The decoder emits the source's native format; tests assert the tensor is
configured accordingly (colour JPEG → `Nv12`, greyscale JPEG → `Grey`):

```rust
let info = tensor.load_image(&mut decoder, &jpeg)?;
assert_eq!(info.format, PixelFormat::Nv12);
assert_eq!((info.width, info.height), (1280, 720));
```

### Capacity Validation

Tests verify that decoding into a too-small tensor returns
`CodecError::InsufficientCapacity` with correct image and tensor dimensions:

```rust
let result = tensor.load_image(&mut decoder, &jpeg);
assert!(matches!(result, Err(CodecError::InsufficientCapacity { .. })));
```

### Non-`u8` JPEG Rejection

JPEG decodes to `u8`-only native formats; a non-`u8` destination is rejected:

```rust
let mut tensor = Tensor::<f32>::image(w, h, PixelFormat::Nv12, Some(Mem))?;
let result = tensor.load_image(&mut decoder, &jpeg);
assert!(matches!(result, Err(CodecError::UnsupportedDtype(_))));
```

The same holds for an unsupported `TensorDyn` dtype:

```rust
let mut tensor = TensorDyn::image(w, h, fmt, DType::I32, Some(Mem)).unwrap();
let result = tensor.load_image(&mut decoder, &data);
assert!(result.is_err());
```

### EXIF Orientation Reporting

Tests decode images that differ only in EXIF orientation tag and assert the
dimensions stay native (unrotated) while `ImageInfo` carries the expected
transform:

```rust
let info = tensor.load_image(&mut decoder, &testdata("zidane_exif_3.jpg"))?;
assert_eq!((info.width, info.height), (1280, 720)); // never rotated
assert_eq!(info.rotation_degrees, 180);
assert_eq!(info.flip_horizontal, false);
```

### Hot-Loop Reuse Pattern

Tests decode multiple different images into the same pre-allocated tensor,
verifying each decode succeeds and reconfigures the tensor:

```rust
let mut tensor = Tensor::<u8>::image(4256, 2532, PixelFormat::Nv12, Some(Mem)).unwrap();
for name in &["zidane.jpg", "giraffe.jpg", "jaguar.jpg"] {
    let info = tensor.load_image(&mut decoder, &testdata(name)).unwrap();
    assert!(info.width > 0);
}
```

### PNG Type-Conversion Validation

The PNG path supports all tensor element types. Tests verify the conversions:

- **f32 normalization**: `Tensor<f32>` decode produces values in `[0.0, 1.0]`.
- **u16 scaling**: 8-bit PNG into `u16` produces exact multiples of 257 (proving
  the `u8 * 257` scaling).
- **i8/i16 XOR consistency**: decode the same PNG into `u8` and `i8` and verify
  `i8_pixel == (u8_pixel ^ 0x80) as i8`, confirming the fast XOR path matches the
  generic `ImagePixel::from_u8()` conversion.

## Allocation Profiling

A key design goal of the codec is zero heap allocations in the JPEG decode hot
path after warmup. The custom JPEG decoder (`JpegDecoderState`) achieves this by
reusing all scratch buffers across frames; on Linux the V4L2 backend keeps its
streaming session and buffers set up across frames.

### Profiling Methodology

Use the included `zero_alloc_check` example to measure allocation behaviour
under `strace`:

```bash
cargo build --release -p edgefirst-codec --example zero_alloc_check
strace -e brk,mmap -f ./target/release/examples/zero_alloc_check 2>&1 \
    | grep -A9999 'HOT LOOP START' | grep -B9999 'HOT LOOP END'
```

This traces `brk` and `mmap` syscalls (proxies for heap allocations) during 100
decode iterations of the same JPEG into a pre-allocated `Tensor<u8>`.

### Expected Results

With the custom CPU JPEG decoder, the hot loop should show **zero** `brk`/`mmap`
syscalls after the warmup iteration. All buffers (`McuScratch`, tensor buffer)
are pre-allocated during warmup and reused.

**Note:** the EXIF orientation read (`kamadak-exif`) allocates when an APP1/EXIF
segment is present. The codec always reads orientation (it reports it in
`ImageInfo`); there is no opt-out, so a JPEG carrying EXIF contributes one
allocation per decode. Strip EXIF upstream if a JPEG source must be fully
allocation-free in the hot loop.

### Allocation Sources by Layer

| Layer                    | After Warmup     | Notes                        |
|--------------------------|------------------|------------------------------|
| JPEG `McuScratch`        | No allocations   | Grows to high-water mark     |
| JPEG Huffman/quant tables| No allocations   | Rebuilt from marker data     |
| JPEG IDCT workspace      | No allocations   | Stack-allocated `[i32; 64]`  |
| JPEG native row write    | No allocations   | Strided into pre-allocated tensor |
| V4L2 streaming session   | No allocations   | Buffers set up once per geometry |
| EXIF reader              | 1 `Vec` / call   | Only when APP1/`eXIf` present |
| zune-png `decode()`      | 1 `Vec` / call   | Returns owned `Vec<u16/u8>`  |
| zune-png `decode_into()` | ~3 `brk` / call  | Internal filter state        |

### Maintaining Zero-Allocation Discipline

When modifying the JPEG decode path:
- **Do not** add `Vec::new()`, `vec![]`, `String::new()`, `Box::new()`, or any
  allocating operation inside the per-frame decode logic
- **Do** reuse `McuScratch` buffers (they grow via `resize()` only if the new
  image is larger than any previously seen)
- **Do** pre-size any new buffers at init time and add them to `McuScratch` or
  `JpegDecoderState`
- If a new allocation is unavoidable, document it in ARCHITECTURE.md
- Run the `zero_alloc_check` example under `strace` to verify changes do not
  introduce new allocation sources

For full pipeline validation (JPEG decode → ImageProcessor letterbox convert),
see the `pipeline_demo` example in `crates/image/examples/` and the
`decode_pipeline_benchmark` in `crates/image/benches/`. See
[`TESTING.md`](../../TESTING.md#zero-allocation-pipeline-verification) at the
workspace root for strace verification instructions.

## V4L2 Hardware Backend Verification

The raw V4L2 ioctl ABI cannot be verified by host unit tests — a struct whose
size differs from the kernel's yields the wrong ioctl request number and a
silent CPU fallback that makes parity tests pass trivially. The only reliable
verification is on-target `strace`:

```bash
# Cross-compile the device-gated test
cargo zigbuild --test v4l2_jpeg --target aarch64-unknown-linux-gnu

# On the target (e.g. imx95-frdm), run under strace and confirm the real
# lifecycle: S_FMT(JPEG) … REQBUFS … STREAMON … QBUF … DQEVENT … G_FMT …
# CAPTURE QBUF … DQBUF=0 (decode complete).
strace -f -e trace=ioctl ./v4l2_jpeg-* --test-threads=1
```

Persistence is proven by a single CAPTURE `REQBUFS` versus N `DQBUF`s across a
multi-frame loop; zero-copy by `QBUF` with `V4L2_MEMORY_DMABUF` and a
byte-match against the MMAP path.

## Benchmarks

See `crates/codec/benches/codec_benchmark.rs` for performance benchmarks.

Benchmark comparisons:
- `edgefirst-codec` custom JPEG decoder vs `image` crate
- `edgefirst-codec` vs raw `zune-png` (PNG overhead of strided copy)
- Strided vs tight decode (measures stride padding overhead)

The nvJPEG GPU decode benchmark cells (`codec/jpeg/nvjpeg/rgbi/*`) that measure
JPEG decode into a CUDA-backed PBO live in `crates/image/benches/nvjpeg_benchmark.rs`
(run with `cargo bench -p edgefirst-image --bench nvjpeg_benchmark`). They require
`ImageProcessor` to allocate the CUDA destination, so they belong in
`edgefirst-image` which owns that type.

```bash
# Run codec benchmarks
cargo bench -p edgefirst-codec

# Cross-compile for target hardware
cargo-zigbuild test --target aarch64-unknown-linux-gnu --release --no-run \
    -p edgefirst-codec --bench codec_benchmark
```

Results are collected in `benchmarks/imx8mp-frdm/codec_benchmark.json` and
`benchmarks/imx95-frdm/codec_benchmark.json`. See
[BENCHMARKS.md](../../BENCHMARKS.md) for analysis.

## Cross-Compiled Test Execution

Integration tests use the `EDGEFIRST_TESTDATA_DIR` environment variable to
locate test images when running on target hardware (where `CARGO_MANIFEST_DIR`
does not exist):

```bash
# On target device
export EDGEFIRST_TESTDATA_DIR=/tmp/testdata
scp -r testdata/ root@target:/tmp/testdata
./decode_jpeg-* --test-threads=1
```
