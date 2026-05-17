# EdgeFirst Codec Testing

## Test Organization

### Unit Tests

Inline `#[cfg(test)]` modules in each source file:

| Module                | Tests                                               |
|-----------------------|-----------------------------------------------------|
| `jpeg/mod.rs`         | EXIF orientation default                            |
| `jpeg/bitstream.rs`   | Bit read, byte-stuffing, sign extension             |
| `jpeg/huffman.rs`     | Table build, symbol decode, block decode            |
| `jpeg/idct/scalar.rs` | DC-only shortcut, known coefficient IDCT            |
| `jpeg/color/scalar.rs`| YCbCr→RGB accuracy, grey copy, neutral grey         |
| `jpeg/upsample/scalar.rs` | H2 single/two/uniform upsampling               |
| `jpeg/markers.rs`     | Minimal JPEG parse, invalid data rejection          |
| `png.rs`              | Format conversion (u8 and u16), correctness checks  |
| `decoder.rs`          | Format detection, capacity validation, error paths  |
| `pixel.rs`            | `ImagePixel::from_u8`/`from_u16` for all 5 types   |
| `options.rs`          | Builder API, default values                         |

### Integration Tests

Located in `crates/codec/tests/`:

| File                    | What It Tests                                      |
|-------------------------|----------------------------------------------------|
| `decode_jpeg.rs`        | JPEG → Tensor<u8/u16/i8/i16/f32>, pixel accuracy, strided decode, NV12 output, error handling, reuse patterns |
| `decode_png.rs`         | PNG → Tensor<u8/u16/i8/i16/f32>, format conversion|
| `decode_tensordyn.rs`   | TensorDyn decode for all dtypes, file path API     |

### Test Images

Tests use images from the workspace `testdata/` directory:

| Image         | Format | Size      | Notes                     |
|---------------|--------|-----------|---------------------------|
| `zidane.jpg`  | JPEG   | 1280×720  | Standard RGB test image   |
| `person.jpg`  | JPEG   | 4256×2532 | Large image, capacity test|
| `giraffe.jpg` | JPEG   | 640×427   | Smaller image             |
| `jaguar.jpg`  | JPEG   | varies    | Reuse pattern test        |
| `grey.jpg`    | JPEG   | 1024×681  | Greyscale image           |
| `zidane.png`  | PNG    | 1280×720  | PNG version of zidane     |

## Running Tests

```bash
# All codec tests (single-threaded — required for GPU resource safety)
cargo test -p edgefirst-codec -- --test-threads=1

# Unit tests only
cargo test -p edgefirst-codec --lib -- --test-threads=1

# Integration tests only
cargo test -p edgefirst-codec --tests -- --test-threads=1

# Specific test
cargo test -p edgefirst-codec decode_zidane_rgb -- --test-threads=1
```

**Important:** Always use `--test-threads=1`. See the root
[TESTING.md](../../TESTING.md) for the rationale (GPU resource contention).

## Test Patterns

### Capacity Validation

Tests verify that decoding into a too-small tensor returns
`CodecError::InsufficientCapacity` with correct image and tensor dimensions:

```rust
let result = tensor.load_image(&mut decoder, &jpeg, &opts);
assert!(matches!(result, Err(CodecError::InsufficientCapacity { .. })));
```

### Hot-Loop Reuse Pattern

Tests decode multiple different images into the same pre-allocated tensor,
verifying each decode succeeds and produces correct dimensions:

```rust
let mut tensor = Tensor::<u8>::image(4256, 2532, Rgb, Some(Mem)).unwrap();
for name in &["zidane.jpg", "giraffe.jpg", "jaguar.jpg"] {
    let info = tensor.load_image(&mut decoder, &testdata(name), &opts).unwrap();
    assert!(info.width > 0);
}
```

### f32 Normalization

Tests verify that `Tensor<f32>` decode produces pixel values in `[0.0, 1.0]`:

```rust
let map = tensor.map().unwrap();
for &v in &pixels[..width * channels] {
    assert!((0.0..=1.0).contains(&v));
}
```

### XOR Consistency (Signed Types)

Tests decode the same image into both `u8` and `i8` tensors, then verify
every pixel satisfies `i8_pixel == (u8_pixel ^ 0x80) as i8`:

```rust
for i in 0..1000 {
    let expected = (u8_pixels[i] ^ 0x80) as i8;
    assert_eq!(i8_pixels[i], expected);
}
```

This confirms the fast i8 XOR path produces identical results to the generic
`ImagePixel::from_u8()` conversion path.

### u16 Scaling Validation

Tests verify that JPEG decode into `u16` produces values that are exact
multiples of 257 (proving the `u8 * 257` scaling is applied correctly):

```rust
for &v in &pixels[..sample_count] {
    assert_eq!(v % 257, 0);
}
```

### Unsupported Dtype

Tests verify that decoding into an unsupported `TensorDyn` dtype (e.g.,
`I32`) returns `CodecError::UnsupportedDtype`:

```rust
let mut tensor = TensorDyn::image(w, h, fmt, DType::I32, Some(Mem)).unwrap();
let result = tensor.load_image(&mut decoder, &data, &opts);
assert!(result.is_err());
```

## Allocation Profiling

A key design goal of the codec is zero heap allocations in the JPEG decode
hot path after warmup. The custom JPEG decoder (`JpegDecoderState`) achieves
this by reusing all scratch buffers across frames.

### Profiling Methodology

Use the included `zero_alloc_check` example to measure allocation behaviour
under `strace`:

```bash
cargo build --release -p edgefirst-codec --example zero_alloc_check
strace -e brk,mmap -f ./target/release/examples/zero_alloc_check 2>&1 \
    | grep -A9999 'HOT LOOP START' | grep -B9999 'HOT LOOP END'
```

This traces `brk` and `mmap` syscalls (proxies for heap allocations) during
100 decode iterations of the same JPEG into a pre-allocated `Tensor<u8>`.

### Expected Results

With the custom JPEG decoder, the hot loop should show **zero** `brk`/`mmap`
syscalls after the warmup iteration. All buffers (`McuScratch`, `exif_scratch`,
tensor buffer) are pre-allocated during warmup and reused.

**Note:** If EXIF orientation is enabled (`apply_exif(true)`, the default),
`kamadak-exif::Reader::read_raw()` allocates on each call. Use
`DecodeOptions::with_exif(false)` to eliminate this allocation source in
latency-critical loops.

### Allocation Sources by Layer

| Layer                    | After Warmup     | Notes                        |
|--------------------------|------------------|------------------------------|
| JPEG `McuScratch`        | No allocations   | Grows to high-water mark     |
| JPEG Huffman/quant tables| No allocations   | Rebuilt from marker data     |
| JPEG IDCT workspace      | No allocations   | Stack-allocated `[i32; 64]`  |
| Row-copy / stride        | No allocations   | Operates on pre-allocated buffers |
| Pixel conversion         | No allocations   | In-place or element-wise     |
| EXIF reader              | 1 `Vec` / call   | Skip with `apply_exif(false)` |
| zune-png `decode()`      | 1 `Vec` / call   | Returns owned `Vec<u16/u8>`  |
| zune-png `decode_into()` | ~3 `brk` / call  | Internal filter state        |

### Maintaining Zero-Allocation Discipline

When modifying the JPEG decode path:
- **Do not** add `Vec::new()`, `vec![]`, `String::new()`, `Box::new()`, or
  any allocating operation inside the per-frame decode logic
- **Do** reuse `McuScratch` buffers (they grow via `resize()` only if the
  new image is larger than any previously seen)
- **Do** pre-size any new buffers at init time and add them to `McuScratch`
  or `JpegDecoderState`
- If a new allocation is unavoidable, document it in ARCHITECTURE.md
- Run the `zero_alloc_check` example under `strace` to verify changes do
  not introduce new allocation sources

For full pipeline validation (JPEG decode → ImageProcessor letterbox
convert), see the `pipeline_demo` example in `crates/image/examples/`
and the `decode_pipeline_benchmark` in `crates/image/benches/`. See
[`TESTING.md`](../../TESTING.md#zero-allocation-pipeline-verification)
at the workspace root for strace verification instructions.

## Benchmarks

See `crates/codec/benches/codec_benchmark.rs` for performance benchmarks.

Benchmark comparisons:
- `edgefirst-codec` custom JPEG decoder vs `image` crate
- `edgefirst-codec` vs raw `zune-png` (PNG overhead of strided copy)
- Strided vs tight decode (measures stride padding overhead)
- `u8` vs `f32` decode (measures conversion overhead)

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
