# edgefirst-hal-capi Testing

## Test Layout

```
crates/capi/
├── src/
│   ├── tensor.rs       # Doc-tests on hal_tensor_* functions
│   ├── image.rs        # Doc-tests on hal_image_processor_* functions
│   ├── decoder.rs      # Doc-tests on hal_decoder_* functions
│   ├── tracker.rs      # Doc-tests on hal_bytetrack_* functions
│   ├── delegate.rs     # Doc-tests on delegate ABI types
│   └── ...
├── tests/              # C-side integration suite (built via Makefile)
│   ├── test_all.c
│   ├── test_tensor.c
│   ├── test_image.c
│   ├── test_decoder.c
│   ├── test_tracker.c
│   ├── test_neutron_dmabuf.c
│   ├── test_neutron_dmabuf_infer.c
│   ├── bench_preproc.c
│   ├── test_common.h
│   └── Makefile
└── include/edgefirst/
    └── hal.h           # cbindgen-generated; CI verifies header parity
```

The Rust side has thin per-function `#[test]` modules and doc-tests on the
public FFI surface. The bulk of the coverage lives in the C-side test
suite under [`crates/capi/tests/`](https://github.com/EdgeFirstAI/hal/tree/main/crates/capi/tests).

## Running Tests

### Rust side

```bash
# Rust unit + doc-tests
cargo test -p edgefirst-hal-capi
```

### C side

```bash
# 1. Build the C library (release artifacts the Makefile expects)
cargo build --release -p edgefirst-hal-capi

# 2. Build and run the full C test suite
cd crates/capi/tests
make test

# 3. Individual test programs
make test_tensor
make test_image
make test_decoder
make test_tracker

# 4. Memory check (valgrind on Linux)
make valgrind
```

The C suite links against `target/release/libedgefirst_hal.{so,a}`; the
Makefile resolves include paths relative to the workspace root, so you
must `cargo build` first.

| Source file | Coverage |
|-------------|----------|
| [`test_tensor.c`](https://github.com/EdgeFirstAI/hal/blob/main/crates/capi/tests/test_tensor.c) | Tensor allocation, DMA-BUF, memory types, fd ownership |
| [`test_image.c`](https://github.com/EdgeFirstAI/hal/blob/main/crates/capi/tests/test_image.c) | Image loading, format conversion, draw masks |
| [`test_decoder.c`](https://github.com/EdgeFirstAI/hal/blob/main/crates/capi/tests/test_decoder.c) | YOLO/ModelPack decoder, post-processing parity |
| [`test_tracker.c`](https://github.com/EdgeFirstAI/hal/blob/main/crates/capi/tests/test_tracker.c) | ByteTrack create/update/get_active_tracks |
| [`test_neutron_dmabuf.c`](https://github.com/EdgeFirstAI/hal/blob/main/crates/capi/tests/test_neutron_dmabuf.c) | On-target Neutron NPU DMA-BUF regression |
| [`test_neutron_dmabuf_infer.c`](https://github.com/EdgeFirstAI/hal/blob/main/crates/capi/tests/test_neutron_dmabuf_infer.c) | End-to-end zero-copy inference path |
| [`bench_preproc.c`](https://github.com/EdgeFirstAI/hal/blob/main/crates/capi/tests/bench_preproc.c) | Reference C preprocessing benchmark — also serves as the canonical example for the [Performance Recommendations](https://github.com/EdgeFirstAI/hal/blob/main/crates/capi/ARCHITECTURE.md#performance-recommendations-dma-buf--egl-path) in `ARCHITECTURE.md` |

## Special Requirements

- **Single-threaded execution** — when the C tests exercise the GL
  backend, they inherit the
  [project-wide single-threaded rule](https://github.com/EdgeFirstAI/hal/blob/main/TESTING.md#single-threaded-execution).
  The C test Makefile runs each `test_*` binary serially.
- **LFS testdata** — image/decoder C tests load fixtures from `testdata/`
  (LFS-tracked). The Makefile resolves `EDGEFIRST_TESTDATA_DIR` from the
  environment and falls back to `<workspace>/testdata`.
- **Hardware gates** — `test_neutron_dmabuf*` requires a live TFLite
  delegate, model, and the i.MX 95 NPU device tree. Run on target with
  `NEUTRON_ENABLE_ZERO_COPY=1`.
- **No special features required** for the Rust-side tests; default
  features build the full FFI surface.
- **`opencv` feature** — when enabled, additional comparison test paths
  are exercised. CI runs both `--features default,opencv` and the default
  feature set.

## Coverage Notes

- The Rust side participates in workspace `cargo llvm-cov`; lcov output
  appears under `crates/capi/`.
- C-side coverage is collected via `gcov` instrumentation when the
  Makefile is invoked with `make test COV=1`. Combined Rust + C coverage
  is uploaded to SonarCloud by the
  [`Process Hardware Coverage`](https://github.com/EdgeFirstAI/hal/blob/main/.github/workflows/test.yml)
  CI job.
- The C tests are the primary correctness gate for the public ABI: any
  signature drift between the Rust source, the cbindgen-generated header,
  and the C call sites is caught here at compile time.

## Header Parity Check

`crates/capi/build.rs` runs `cbindgen` during `cargo build` and writes
the result to `include/edgefirst/hal.h`. The committed copy must match
the generated copy. CI catches drift via the
[`Generate SBOM & Check License Compliance`](https://github.com/EdgeFirstAI/hal/blob/main/.github/workflows/sbom.yml)
workflow's `make sbom` step, which fails the build if the header changed
without being committed. Locally, run:

```bash
cargo build -p edgefirst-hal-capi
git diff -- crates/capi/include/edgefirst/hal.h    # should be empty
```

## Cross-References

- Project testing patterns: [../../TESTING.md](https://github.com/EdgeFirstAI/hal/blob/main/TESTING.md)
- Validating optimizations: [TESTING.md#validating-optimizations](https://github.com/EdgeFirstAI/hal/blob/main/TESTING.md#validating-optimizations)
- C API performance patterns: [ARCHITECTURE.md#performance-recommendations-dma-buf--egl-path](https://github.com/EdgeFirstAI/hal/blob/main/crates/capi/ARCHITECTURE.md#performance-recommendations-dma-buf--egl-path)
- Image-side test gating (GL/G2D probes): [../image/TESTING.md](https://github.com/EdgeFirstAI/hal/blob/main/crates/image/TESTING.md)
- Tensor-side DMA tests: [../tensor/TESTING.md](https://github.com/EdgeFirstAI/hal/blob/main/crates/tensor/TESTING.md)
