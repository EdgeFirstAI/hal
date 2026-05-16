# EdgeFirst Codec Testing

## Test Organization

### Unit Tests

Inline `#[cfg(test)]` modules in each source file:

| Module       | Tests                                              |
|--------------|---------------------------------------------------|
| `jpeg.rs`    | Decode test JPEGs, verify dimensions + pixel data  |
| `png.rs`     | Decode test PNGs, format conversion correctness    |
| `decoder.rs` | Format detection, capacity validation, error paths |
| `pixel.rs`   | `ImagePixel::from_u8` conversion correctness       |
| `options.rs` | Builder API, default values                        |

### Integration Tests

Located in `crates/codec/tests/`:

| File                    | What It Tests                                    |
|-------------------------|--------------------------------------------------|
| `decode_jpeg.rs`        | JPEG → Tensor<u8>/f32, various formats, reuse    |
| `decode_png.rs`         | PNG → Tensor<u8>/f32, format conversion           |
| `decode_tensordyn.rs`   | TensorDyn decode, dtype validation, file path API |

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

## Benchmarks

See `crates/codec/benches/codec_benchmark.rs` for performance benchmarks.

Planned benchmark comparisons:
- `edgefirst-codec` vs raw `zune-jpeg`/`zune-png` (overhead of strided copy)
- `edgefirst-codec` vs legacy `edgefirst_image::load_image()` (allocation savings)
- `edgefirst-codec` vs `image` crate, OpenCV, turbojpeg (feature-gated)

```bash
# Run codec benchmarks
cargo bench -p edgefirst-codec

# With OpenCV comparison (requires opencv feature)
cargo bench -p edgefirst-codec --features opencv
```
