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
    └── hal.h           # cbindgen-generated; committed in-tree (review is the parity gate)
```

The Rust side has thin per-function `#[test]` modules and doc-tests on the
public FFI surface. The bulk of the coverage lives in the C-side test
suite under [`crates/capi/tests/`](https://github.com/EdgeFirstAI/hal/tree/main/crates/capi/tests).

## Running Tests

### Rust side

`cargo test` runs with `-- --test-threads=1` per the workspace
single-threaded rule
([root TESTING.md § Single-threaded execution](https://github.com/EdgeFirstAI/hal/blob/main/TESTING.md#single-threaded-execution)).

```bash
# Rust unit + doc-tests
cargo test -p edgefirst-hal-capi -- --test-threads=1
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
- **No LFS testdata for the standard C suite.** The C tests
  (`test_tensor`, `test_image`, `test_decoder`, `test_tracker`,
  `bench_preproc`) synthesize all inputs in-process — they do not
  read from `testdata/` and the Makefile does not resolve
  `EDGEFIRST_TESTDATA_DIR`. The `test_neutron_dmabuf*` programs do
  consume external data (a TFLite model and an input frame), but the
  paths are passed as command-line arguments per the per-binary
  `--help` output, not via an env var.
- **Hardware gates** — `test_neutron_dmabuf*` requires a live TFLite
  delegate, model, and the i.MX 95 NPU device tree. Run on target with
  `NEUTRON_ENABLE_ZERO_COPY=1`.
- **No special features required** for the Rust-side tests; default
  features build the full FFI surface.

## Coverage Notes

- The Rust side participates in workspace `cargo llvm-cov`; lcov output
  appears under `crates/capi/`.
- The Rust llvm-cov coverage is merged with Python slipcover coverage by
  the [`Process Hardware Coverage`](https://github.com/EdgeFirstAI/hal/blob/main/.github/workflows/test.yml)
  CI job and uploaded to SonarCloud. The C test binaries are not
  instrumented with `gcov` today; they validate behaviour and provide a
  compile-time link-check of the cbindgen header, but C-side line
  coverage is not currently reported.
- The C tests are the primary correctness gate for the public ABI: any
  signature drift between the Rust source, the cbindgen-generated header,
  and the C call sites is caught here at compile time.

## Header Parity Check

`crates/capi/build.rs` runs `cbindgen` during `cargo build` and writes
the result to `include/edgefirst/hal.h`. The committed copy must stay
in sync with the generated copy — contributors are expected to commit
the regenerated header alongside any Rust FFI change. CI does not
currently enforce this with an explicit `git diff --exit-code` step, so
review is the gate. Locally:

```bash
cargo build -p edgefirst-hal-capi
git diff -- crates/capi/include/edgefirst/hal.h    # should be empty
```

If the diff is non-empty, commit the regenerated header.

## Cross-References

- Project testing patterns: [../../TESTING.md](https://github.com/EdgeFirstAI/hal/blob/main/TESTING.md)
- Validating optimizations: [TESTING.md#validating-optimizations](https://github.com/EdgeFirstAI/hal/blob/main/TESTING.md#validating-optimizations)
- C API performance patterns: [ARCHITECTURE.md#performance-recommendations-dma-buf--egl-path](https://github.com/EdgeFirstAI/hal/blob/main/crates/capi/ARCHITECTURE.md#performance-recommendations-dma-buf--egl-path)
- Image-side test gating (GL/G2D probes): [../image/TESTING.md](https://github.com/EdgeFirstAI/hal/blob/main/crates/image/TESTING.md)
- Tensor-side DMA tests: [../tensor/TESTING.md](https://github.com/EdgeFirstAI/hal/blob/main/crates/tensor/TESTING.md)
