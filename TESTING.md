# EdgeFirst HAL - Testing Guide

**Version:** 1.0
**Last Updated:** March 2026
**Status:** Production
**Audience:** Developers contributing to EdgeFirst HAL or running tests on target hardware

---

## Overview

This guide consolidates all testing information for the EdgeFirst HAL project. Testing spans five categories: Rust unit tests, Python integration tests, C API integration tests, GL hardware-gated tests, and on-target tests. All categories share a single constraint: tests must run single-threaded.

---

## Quick Reference

| Makefile Target  | What It Does                                              |
|------------------|-----------------------------------------------------------|
| `make test`        | Run all tests (Rust + Python + C API) with coverage       |
| `make test-rust`   | Run Rust tests only with `cargo-llvm-cov nextest`         |
| `make test-python` | Run Python tests only with pytest (and slipcover if available) |
| `make test-capi`   | Build the C library and run C API integration tests       |
| `make bench`       | Run Rust criterion benchmarks                             |
| `make build`       | Build with coverage instrumentation (profiling profile)   |

C API tests are also available individually from `crates/capi/tests/` using their own Makefile.

---

## Required Toolchain

The following tools must be installed before running tests:

| Tool | Required | Purpose |
|------|----------|---------|
| `cargo-nextest` | Yes | Fast Rust test runner; used by `make test-rust` |
| `cargo-llvm-cov` | Yes | Coverage instrumentation; required by `make test-rust` |
| `maturin` | Yes | Build Python bindings before running Python tests |
| `psutil` | Optional | Python dependency for some test helpers |
| `slipcover` | Optional | Python branch coverage; `make test-python` falls back to plain pytest when absent |

Install the mandatory tools:

```bash
cargo install cargo-nextest
cargo install cargo-llvm-cov --locked
pip install maturin
```

Install `slipcover` into the local virtual environment for coverage reports:

```bash
source venv/bin/activate
pip install slipcover
```

---

## Test Categories

### Rust Unit Tests

Rust tests are co-located with source code in `#[cfg(test)]` modules:

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_process_data_valid_input() {
        // test implementation
    }
}
```

Tests live in every crate under `crates/`: `tensor`, `image`, `decoder`, `tracker`, `hal`, `capi`, `gpu-probe`. The `edgefirst_hal` Python crate is excluded from most test runs because it requires a live Python interpreter.

### Python Integration Tests

Python tests are located in `tests/` and use pytest:

```
tests/
├── test_tensor.py
├── image/
│   ├── test_image.py
│   └── test_image_perf.py
└── decoder/
    └── test_decoder.py
```

Python tests require the bindings to be built with `maturin develop` before running. Some tests use `pytest.mark.skipif` to gate platform-specific paths (e.g., DMA-BUF tests require `sys.platform == "linux"`). Benchmark tests use `pytest.mark.benchmark` with `warmup_iterations=3`.

### C API Integration Tests

C integration tests live in `crates/capi/tests/` and are built by their own Makefile. The test suite covers tensor, image, decoder, and tracker APIs:

| Source File | Coverage Area |
|-------------|---------------|
| `test_tensor.c` | Tensor allocation, DMA-BUF, memory types |
| `test_image.c` | Image loading, format conversion |
| `test_decoder.c` | YOLO decoder, post-processing |
| `test_tracker.c` | Object tracking API |
| `test_neutron_dmabuf.c` | On-target Neutron NPU DMA-BUF regression |
| `bench_preproc.c` | C preprocessing benchmark |

Build and run:

```bash
# First build the C library
cargo build --release -p edgefirst-hal-capi

# Then build and run the C tests
cd crates/capi/tests
make test

# Individual test suites
make test_tensor
make test_image
make test_decoder
make test_tracker

# Memory check
make valgrind
```

The Neutron DMA-BUF regression test (`test_neutron_dmabuf`) requires a live TFLite delegate and model; it must be run on target hardware with `NEUTRON_ENABLE_ZERO_COPY=1`.

### GL Hardware-Gated Tests

OpenGL and G2D tests inside `crates/image/src/gl/tests.rs` use `OnceLock<bool>` to probe hardware availability once and skip if the hardware is absent:

```rust
static GL_AVAILABLE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();

fn is_opengl_available() -> bool {
    *GL_AVAILABLE.get_or_init(|| GLProcessorThreaded::new(None).is_ok())
}
```

A test that requires OpenGL calls `is_opengl_available()` at the start and returns early when it returns `false`. This keeps the test suite green on developer machines without GPU hardware, while exercising the full hardware path on target boards and CI runners with GPU access.

The Neutron-scenario tests gate on `/dev/neutron0` as a platform discriminator for i.MX 95 with Mali GPU. This device node indicates that large-offset DMA-BUF EGLImage imports are supported — these fail with `EGL(BadAccess)` on Vivante (i.MX 8MP) even without the NPU driver:

```rust
fn is_neutron_available() -> bool {
    *NEUTRON_AVAILABLE.get_or_init(|| std::path::Path::new("/dev/neutron0").exists())
}
```

### On-Target Tests

Tests that exercise DMA-heap or G2D acceleration require target hardware. They are automatically skipped on development machines when the required devices are absent.

| Device Node | What It Gates |
|-------------|---------------|
| `/dev/dma_heap/linux,cma` or `/dev/dma_heap/system` | DMA-BUF tensor allocation tests |
| `/dev/galcore` | G2D hardware acceleration tests |
| `/dev/neutron0` | Neutron NPU DMA-BUF EGLImage tests |
| `/dev/dri/renderD128` | DRM render node; required for DMA-buf backend |

---

## Single-Threaded Execution

**All tests must run single-threaded.** This is a hard requirement, not a recommendation.

Use `--test-threads=1` with `cargo test` and `-j 1` with `cargo nextest`. The Makefile enforces this automatically via `cargo llvm-cov nextest ... -j 1`.

### Why Single-Threaded

Three independent constraints each independently require single-threaded execution:

1. **EGL display sharing**: When multiple tests run in parallel threads within one process, `eglTerminate` on one thread can tear down a shared EGL display while other threads still reference it. This causes intermittent failures that are difficult to reproduce and platform-dependent.

2. **G2D driver state**: The `galcore` kernel driver maintains per-process state that is not safe to access from concurrent threads creating and destroying G2D contexts.

3. **DMA-heap CMA pool exhaustion**: Concurrent DMA-heap allocations from multiple test threads can exhaust the CMA pool on memory-constrained embedded targets, causing allocation failures that mask the real test failures.

`cargo nextest` already provides per-test process isolation, but `-j 1` is still required to prevent DMA and GPU contention across test processes.

This constraint applies to CI (`test.yml`), the Makefile, and local development. See ARCHITECTURE.md section "Testing with GPU Resources" for the full technical rationale.

---

## Running Tests Locally

### Native Development (x86_64)

```bash
# 1. Run the full test suite (Rust + Python + C API) with coverage
make test

# 2. Run Rust tests only
make test-rust

# 3. Run Python tests only
make test-python

# 4. Run Rust tests manually with nextest
cargo nextest run --workspace --exclude edgefirst_hal -j 1

# 5. Run tests for a specific crate
cargo test -p edgefirst_image -- --test-threads=1
cargo test -p edgefirst_decoder -- --test-threads=1
cargo test -p edgefirst_tensor -- --test-threads=1
```

### Python Test Setup

Python tests require the native bindings to be installed into the virtual environment:

```bash
# 1. Activate the virtual environment
source venv/bin/activate

# 2. Build and install the Python bindings in development mode
maturin develop -m crates/python/Cargo.toml

# 3. Run tests
python -m pytest tests/

# 4. Run a specific test module
python -m pytest tests/image/test_image.py
python -m pytest tests/decoder/test_decoder.py

# 5. Run with coverage (if slipcover is installed)
python -m slipcover --xml --out target/python-coverage.xml -m pytest tests/
```

### Disabling Hardware Backends

Use environment variables to isolate tests to specific backends:

| Variable | Effect |
|----------|--------|
| `EDGEFIRST_TENSOR_FORCE_MEM=1` | Force heap (`MemTensor`) allocation; skips DMA-heap and shared memory |
| `EDGEFIRST_FORCE_BACKEND=cpu` | Force CPU-only image processing |
| `EDGEFIRST_FORCE_BACKEND=opengl` | Force OpenGL backend |
| `EDGEFIRST_FORCE_BACKEND=g2d` | Force G2D backend |
| `EDGEFIRST_DISABLE_GL=1` | Disable OpenGL even when hardware is present |
| `EDGEFIRST_DISABLE_G2D=1` | Disable G2D even when hardware is present |

Example — run tests without any GPU backend:

```bash
EDGEFIRST_DISABLE_GL=1 EDGEFIRST_DISABLE_G2D=1 cargo test --workspace -- --test-threads=1
```

### Benchmarking native fp16 paths (local only)

The default build does **not** enable `+fp16` (aarch64) or `+f16c`
(x86_64) — distributed HAL binaries stay on each target triple's
baseline ISA so they run on older CPUs within the same triple.
Benchmarks that need to exercise the native f16 mask kernel
(`fused_dot_sigmoid_f16_slice`) must set `RUSTFLAGS` explicitly on
the host that supports the extension.

```bash
# Orin Nano (Cortex-A78AE):
RUSTFLAGS="-C target-cpu=cortex-a78ae" \
    cargo bench -p edgefirst-image --bench mask_benchmark

# Generic aarch64 with FEAT_FP16 (verify the host first —
# imx8mp / Cortex-A53 will SIGILL):
RUSTFLAGS="-C target-feature=+fp16" \
    cargo bench -p edgefirst-image --bench mask_benchmark

# x86_64 Haswell+ (enables the explicit _mm256_cvtph_ps intrinsic
# kernel in addition to the scalar path):
RUSTFLAGS="-C target-feature=+f16c,+fma,+avx2" \
    cargo bench -p edgefirst-image --bench mask_benchmark
```

Verify the release build actually compiled to native instructions
(not the soft-float `__extendhfsf2` helper) via
`scripts/audit_f16_codegen.sh <target-triple>`. Requires
`cargo install cargo-show-asm`.

---

## Cross-Compilation and On-Target Testing

The project primarily targets ARM64 embedded Linux (NXP i.MX 8M Plus, i.MX 95). Tests that exercise DMA or GPU acceleration must run on physical hardware.

### Step 1: Cross-Compile Tests

Use `cargo-zigbuild` to produce test binaries without running them. Exclude the Python crate because PyO3 requires a target Python installation:

```bash
cargo-zigbuild test --target aarch64-unknown-linux-gnu --release --no-run \
    --workspace --exclude edgefirst_hal
```

Test binaries are written to `target/aarch64-unknown-linux-gnu/release/deps/`.

### Step 2: Copy Binaries to Target

```bash
# Copy test binaries (the hash suffix varies per build)
scp target/aarch64-unknown-linux-gnu/release/deps/edgefirst_image-* user@target:/tmp/hal-tests/
scp target/aarch64-unknown-linux-gnu/release/deps/edgefirst_tensor-* user@target:/tmp/hal-tests/
scp target/aarch64-unknown-linux-gnu/release/deps/edgefirst_decoder-* user@target:/tmp/hal-tests/

# Also copy test data
scp -r testdata/ user@target:/tmp/hal-tests/
```

### Step 3: Run Tests on Target

```bash
ssh user@target 'cd /tmp/hal-tests && ./edgefirst_image-<hash> --test-threads=1'
```

The `--test-threads=1` flag is mandatory on target hardware for the same reasons as local development — the CMA pool exhaustion risk is higher on embedded boards with limited memory.

---

## Validating Optimizations

This section is the **testing-level reference** for the
[Optimization Guide in README.md](README.md#optimization-guide). Each rule
in that guide has a corresponding way to verify your integration follows
it; this section documents those verification techniques.

### Verifying EGL Image Cache Hits

The OpenGL backend's EGL image cache emits a summary log line at process
shutdown. Run any test or benchmark with `RUST_LOG=edgefirst_image=debug`:

```bash
RUST_LOG=edgefirst_image=debug \
    cargo test -p edgefirst-image --test ... -- --test-threads=1 2>&1 | tee cache.log
```

Look for the final line:

```text
EglImageCache stats: 999 hits, 1 misses, 1 entries remaining
```

A correctly-tuned camera pipeline reaches steady state with **misses ≈ pool
depth** (one miss per unique physical buffer) and **hits ≈ frame count**.
If `misses` grows linearly with frame count, the calling code is creating
new tensor objects per frame — review the cache key (Rule 3 in the
Optimization Guide).

The cache also logs `EglImageCache: swept N dead entries` when tensors are
dropped while their entries are still in the cache. Frequent sweeps in a
steady-state pipeline are a sign of leaked or churned tensor objects.

### Verifying Backend Selection

The active backend is logged at `info` level when `ImageProcessor::new()`
succeeds:

```text
Shared EGL display initialized: kind=Wayland, transfer=DmaBuf
```

The `transfer=` field reports `DmaBuf`, `Pbo`, or `Sync` — these correspond
to the three rows in the README *Platform GPU Support* table. To confirm a
specific backend is being exercised, set the corresponding environment
variable from the table below and re-run with `RUST_LOG=info`.

| Variable | Effect | Use to verify |
|----------|--------|---------------|
| `EDGEFIRST_FORCE_BACKEND=cpu` | CPU only — no GPU code path | Code paths that should never crash without a GPU |
| `EDGEFIRST_FORCE_BACKEND=opengl` | OpenGL only — no G2D, no CPU fallback | GL backend correctness |
| `EDGEFIRST_FORCE_BACKEND=g2d` | G2D only — fails on non-i.MX | G2D-specific format pairs |
| `EDGEFIRST_FORCE_TRANSFER=pbo` | PBO transfers even when DMA-buf is available | PBO path on Linux with DMA-buf hardware |
| `EDGEFIRST_FORCE_TRANSFER=dmabuf` | DMA-buf transfers; fails if EGLImage import fails | DMA-buf path on systems where it normally falls back |
| `EDGEFIRST_FORCE_TRANSFER=sync` | Memcpy upload/readback via `glTexImage2D` / `glReadnPixels` | Non-zero-copy baseline; useful for measuring the cost of the fast paths |
| `EDGEFIRST_TENSOR_FORCE_MEM=1` | Forces heap tensors; disables DMA / SHM | Pure-CPU regression baseline |

### Regression Testing for Tensor Reuse

To catch regressions where a caller starts allocating a tensor per frame,
gate the test on the cache miss count reported in the `EglImageCache stats`
log line. The pattern below is illustrative — the `alloc_count()` accessor
is **not** a stable public API. In practice, capture the log output and
parse the `H hits, M misses` summary, or use the `bench_preproc` reuse
variant (see below) as the regression baseline.

```rust
// Illustrative — alloc_count() is shown for clarity, not as a real API.
// In a real test, parse the EglImageCache stats log line instead.
#[test]
fn test_steady_state_no_realloc() {
    let _ = env_logger::builder().is_test(true).try_init();
    let mut proc = ImageProcessor::new().unwrap();
    let mut dst = proc.create_image(640, 640, PixelFormat::Rgb,
                                    DType::U8, None).unwrap();
    let src = load_image(&fixture_bytes(), Some(PixelFormat::Rgb), None).unwrap();

    // Warm-up
    proc.convert(&src, &mut dst,
                 Rotation::None, Flip::None, Crop::default()).unwrap();

    // Steady state — every iteration must hit the EGL image cache.
    for _ in 0..100 {
        proc.convert(&src, &mut dst,
                     Rotation::None, Flip::None, Crop::default()).unwrap();
    }
    // Drop `proc` to flush the EglImageCache stats log line. Assert in the
    // test harness that misses == 1 (the warm-up) by parsing the log.
}
```

For the C API, `bench_preproc` (in `crates/capi/tests/bench_preproc.c`) is
the reference reproduction of the allocate-once / reuse-every-frame pattern
and serves as both a benchmark and a regression test.

### Verifying the Inode-Keyed Cache (V4L2 / libcamera Integrations)

When integrating the HAL into a pipeline that consumes V4L2 or libcamera
buffers, write an integration test that simulates fd recycling:

1. Open a single `dma_buf` and obtain two distinct fd numbers via `dup()`.
2. Import the first fd into HAL, run a `convert()`, and capture the EGL
   cache miss count.
3. Close the first fd, then import the *second* fd (which refers to the
   same physical buffer).
4. Run `convert()` again. If the calling code uses an inode-keyed cache,
   it returns the existing tensor and the miss count does not increase.
   If it uses an fd-keyed cache or skips caching entirely, the miss count
   grows by one.

The Au-Zone GStreamer elements ship with this regression test in their
own test suite; downstream integrators should write the equivalent for
their pipeline.

### Verifying the `from_numpy` Strided Fast-Path

Rule 7 in the [Optimization Guide](README.md#rule-7--numpy-interop-pass-arrays-straight-to-from_numpy)
asserts that callers should not pre-apply `np.ascontiguousarray()` —
HAL handles strided sources internally. Two regression tests in
`tests/test_tensor.py` pin this:

| Test | What it verifies |
|---|---|
| `test_from_numpy_hailort_shape` | Correctness on the HailoRT-shaped `(1, 116, 8400)` f32 transposed view (Path 3 of `copy_numpy_to_tensor_dyn`). The destination tensor matches the source values element-by-element. |
| `test_from_numpy_hailort_shape_perf_sanity` | Perf bound: the automatic fast path runs no more than 1.5× the manual `np.ascontiguousarray + from_numpy(contig)` workaround. Prior to PR #58 the ratio was ≈ 10×. |

Both tests are in the standard Python suite and run in CI. If the
perf-sanity test fails, suspect a regression in `numpy.ascontiguousarray`
behaviour or in the Path-3 dispatch logic in `crates/python/src/tensor.rs`.

### Verifying `MaskResolution` Selection in COCO Validators

Rule 8 in the [Optimization Guide](README.md#rule-8--choose-the-correct-maskresolution-for-your-evaluation)
asserts that COCO / IoU evaluation must use `MaskResolution::Scaled`,
not the default `Proto`. Downstream validators should pin this with a
small assertion at the materialize call site:

```python
masks = proc.materialize_masks(
    boxes, scores, classes, proto_data,
    letterbox=letterbox_norm,
    resolution=hal.MaskResolution.Scaled(orig_w, orig_h),
)
# Smoke-check the contract: tile shape (h, w, 1) uint8 at
# orig-image resolution, binary {0, 255} after upsample-then-threshold.
assert masks[0].shape[2] == 1
assert masks[0].dtype == np.uint8
```

If a validator's mask-mAP regresses against a numpy / ONNX reference,
inspect this call first — the most common cause is calling
`materialize_masks` with no `resolution` argument (defaults to `Proto`)
and then thresholding the proto-resolution sigmoid in caller code
before resizing, which produces blocky binary edges.

### Quantifying Native CPU Feature Builds

When benchmarking with `RUSTFLAGS="-C target-feature=+fp16"` (or `+f16c` on
x86_64), confirm the kernel actually compiled to native instructions
rather than the soft-float helper:

```bash
scripts/audit_f16_codegen.sh aarch64-unknown-linux-gnu
```

The script disassembles a release build and fails if `__extendhfsf2` is
present (soft-float fallback) or if `fcvt` / `vcvtph2ps` is absent. See
[§ Benchmarking native fp16 paths](#benchmarking-native-fp16-paths-local-only)
above for the full workflow.

---

## Coverage Thresholds

| Scope | Minimum Threshold |
|-------|-------------------|
| Overall workspace | 70% line coverage |
| Critical paths (tensor ops, decoder, image processing hot paths) | 90% line coverage |

Coverage is collected using `cargo-llvm-cov` for Rust and `slipcover` for Python. Both generate LCOV reports that are merged and reported to SonarCloud.

Coverage reports:
- Rust: `target/rust-coverage.lcov`
- Python: `target/python-coverage.xml`

CI enforces coverage gating on pull requests. A failing coverage check blocks the merge.

---

## CI/CD Test Matrix

Tests run across multiple runner types:

| Job | Runner | Architecture | Hardware |
|-----|--------|--------------|----------|
| Build & Test (x86_64) | ubuntu-22.04 | x86_64 | No GPU |
| Build & Test (macOS) | macos-latest | x86_64/arm64 | No GPU |
| Build (aarch64) | ubuntu-22.04-arm-xlarge | aarch64 | No GPU (compile only) |
| Test (aarch64) | ubuntu-22.04-arm | aarch64 | No GPU |
| Hardware Test (imx8mp) | nxp-imx8mp-latest | aarch64 | G2D, DMA-heap |

The hardware runner (`nxp-imx8mp-latest`) is the only environment where G2D and DMA-BUF tests are fully exercised. Hardware-gated tests that return early on the x86 and arm runners are counted as passed (not skipped) because the gate is an explicit probe, not a `#[ignore]` attribute.

---

## Common Failures and Remedies

| Symptom | Likely Cause | Remedy |
|---------|--------------|--------|
| Intermittent EGL segfault | Tests running in parallel | Ensure `--test-threads=1` / `-j 1` |
| `DMA-heap allocation failed` on target | CMA pool exhaustion from parallel tests | Reduce parallelism; use `-j 1` |
| `cargo-nextest not found` | Tool not installed | `cargo install cargo-nextest` |
| `cargo-llvm-cov not found` | Tool not installed | `cargo install cargo-llvm-cov --locked` |
| Python tests fail with `ModuleNotFoundError` | Bindings not built | `maturin develop -m crates/python/Cargo.toml` |
| C API tests fail with `libedgefirst_hal.so not found` | C library not built | `cargo build --release -p edgefirst-hal-capi` |
| GL tests all skip | No EGL display available | Expected on headless CI; set `DISPLAY` or use a virtual framebuffer |
