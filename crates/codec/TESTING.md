# EdgeFirst Codec Testing

## Test Organization

### Unit Tests

Inline `#[cfg(test)]` modules in each source file:

| Module       | Tests                                               |
|--------------|-----------------------------------------------------|
| `jpeg.rs`    | Decode test JPEGs, verify dimensions + pixel data   |
| `png.rs`     | Format conversion (u8 and u16), correctness checks  |
| `decoder.rs` | Format detection, capacity validation, error paths  |
| `pixel.rs`   | `ImagePixel::from_u8`/`from_u16` for all 5 types   |
| `options.rs` | Builder API, default values                         |

### Integration Tests

Located in `crates/codec/tests/`:

| File                    | What It Tests                                      |
|-------------------------|----------------------------------------------------|
| `decode_jpeg.rs`        | JPEG → Tensor<u8/u16/i8/i16/f32>, reuse patterns  |
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

A key design goal of the codec is to minimise heap allocations in the decode
hot path. The edgefirst-codec layer (scratch buffer, row-copy, pixel
conversion) is allocation-free after warmup. However, the underlying format
libraries (zune-jpeg, zune-png) create internal decoder state on every call
because they do not support state reuse.

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

### Current Results

With zune-jpeg as the decode backend, approximately 3 `brk` syscalls occur
per JPEG decode iteration, originating from zune-jpeg's internal Huffman table
and quantization matrix allocation. These are **not** from edgefirst-codec's
own scratch buffer or row-copy logic.

### Allocation Sources by Layer

| Layer                | After Warmup     | Notes                        |
|----------------------|------------------|------------------------------|
| `ImageDecoder.scratch` | No allocations | `Vec::resize` is no-op at high-water mark |
| Row-copy / stride    | No allocations   | Operates on pre-allocated buffers |
| Pixel conversion     | No allocations   | In-place or element-wise     |
| Format conversion    | 1 `Vec` if needed | Only when src_fmt ≠ dest_fmt |
| zune-jpeg decoder    | ~3 `brk` / call  | Internal Huffman/quant tables |
| zune-png `decode()`  | 1 `Vec` / call   | Returns owned `Vec<u16/u8>`  |
| zune-png `decode_into()` | ~3 `brk` / call | Internal filter state    |
| EXIF reader          | 1 `Vec` / call   | `to_vec()` on EXIF data; skip with `apply_exif(false)` |

### Maintaining Low-Allocation Discipline

When modifying the JPEG or PNG decode paths:
- **Do not** add `Vec::new()`, `vec![]`, `String::new()`, `Box::new()`, or
  any allocating operation inside the per-frame decode logic in edgefirst-codec
- **Do** reuse the `ImageDecoder.scratch` buffer (it grows via `resize()`
  only if the new image is larger than any previously seen)
- **Do** pre-size any buffers at init time
- If a new allocation is unavoidable (e.g., a library returns an owned Vec),
  document it in ARCHITECTURE.md
- Run the `zero_alloc_check` example under `strace` to verify any changes
  do not introduce new allocation sources in the codec layer

## Benchmarks

See `crates/codec/benches/codec_benchmark.rs` for performance benchmarks.

Benchmark comparisons:
- `edgefirst-codec` vs raw `zune-jpeg`/`zune-png` (overhead of strided copy)
- `edgefirst-codec` vs `image` crate (allocation-heavy reference)
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
