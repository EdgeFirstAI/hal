# EdgeFirst HAL — Testing Guide

**Audience:** developers contributing to EdgeFirst HAL or running tests on
target hardware.

This guide is the **cross-crate** testing reference. It covers the rules
and patterns that apply across the whole workspace — single-threaded
execution, on-target hardware gating, cross-compilation, optimization
validation, coverage, and the CI matrix. Crate-specific testing detail
lives in each sub-crate's `TESTING.md`:

| Crate | Per-crate testing guide |
|-------|------------------------|
| `tensor` | [crates/tensor/TESTING.md](https://github.com/EdgeFirstAI/hal/blob/main/crates/tensor/TESTING.md) |
| `image` | [crates/image/TESTING.md](https://github.com/EdgeFirstAI/hal/blob/main/crates/image/TESTING.md) — GL gating, fp16 benches |
| `decoder` | [crates/decoder/TESTING.md](https://github.com/EdgeFirstAI/hal/blob/main/crates/decoder/TESTING.md) |
| `tracker` | [crates/tracker/TESTING.md](https://github.com/EdgeFirstAI/hal/blob/main/crates/tracker/TESTING.md) |
| `hal` (umbrella) | [crates/hal/TESTING.md](https://github.com/EdgeFirstAI/hal/blob/main/crates/hal/TESTING.md) |
| `capi` (C API) | [crates/capi/TESTING.md](https://github.com/EdgeFirstAI/hal/blob/main/crates/capi/TESTING.md) — C suite, header parity check |
| `python` | [crates/python/TESTING.md](https://github.com/EdgeFirstAI/hal/blob/main/crates/python/TESTING.md) — maturin develop, slipcover |

---

## Quick Reference

| Makefile target | What it does |
|-----------------|--------------|
| `make test` | Run all tests (Rust + Python + C API) with coverage |
| `make test-rust` | Run Rust tests only with `cargo-llvm-cov nextest` |
| `make test-python` | Run Python tests with pytest (and slipcover if available) |
| `make test-capi` | Build the C library and run C API integration tests |
| `make bench` | Run the workspace Rust benchmarks (custom `harness = false` binaries backed by [`crates/bench`](https://github.com/EdgeFirstAI/hal/tree/main/crates/bench); not Criterion) |
| `make build` | Build with coverage instrumentation (profiling profile) |
| `make format lint check` | Pre-commit gate — required before every commit |

C API tests are also available individually from
[`crates/capi/tests/`](https://github.com/EdgeFirstAI/hal/tree/main/crates/capi/tests)
using their own Makefile.

---

## Required Toolchain

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

Install `slipcover` into the local virtual environment for coverage:

```bash
source venv/bin/activate
pip install slipcover
```

---

## Rust Unit Tests

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

Most crates under `crates/` carry their own Rust tests: `tensor`, `image`,
`decoder`, `tracker`, `hal`, `capi`, and `bench`. The `gpu-probe` crate
ships no `#[test]` modules — it is a runtime probe binary and is exercised
indirectly through the tensor/image suites it informs. The `edgefirst_hal`
Python crate is excluded from most workspace runs because it requires a
live Python interpreter — see
[`crates/python/TESTING.md`](https://github.com/EdgeFirstAI/hal/blob/main/crates/python/TESTING.md).

Cross-crate integration suites live in each crate's `tests/` directory;
see the relevant per-crate `TESTING.md` for what each suite covers.

---

## On-Target Tests

Tests that exercise DMA-heap or G2D acceleration require target hardware.
They skip themselves on development machines when the required device
nodes are absent.

| Device node | What it gates |
|-------------|---------------|
| `/dev/dma_heap/linux,cma` or `/dev/dma_heap/system` | DMA-BUF tensor allocation tests |
| `/dev/galcore` | G2D hardware acceleration tests |
| `/dev/neutron0` | Neutron NPU DMA-BUF EGLImage tests |
| `/dev/dri/renderD128` (preferred) or `/dev/dri/card0` / `/dev/dri/card1` | DRM render/card node; the GL context opens the first that exists, so any one is sufficient |

The hardware-gated test pattern (a single `OnceLock<bool>` probe that
short-circuits on missing devices) is documented in
[`crates/image/TESTING.md`](https://github.com/EdgeFirstAI/hal/blob/main/crates/image/TESTING.md#gl-hardware-gated-tests).

---

## Single-Threaded Execution

**All tests must run single-threaded.** This is a hard requirement, not a
recommendation.

Use `--test-threads=1` with `cargo test` and `-j 1` with `cargo nextest`.
The [`Makefile`](https://github.com/EdgeFirstAI/hal/blob/main/Makefile)
enforces this automatically via `cargo llvm-cov nextest ... -j 1`.

### Why single-threaded

Three independent constraints each require it:

1. **GL driver concurrency bugs** — the HAL uses a process-global EGL
   display (`SharedEglDisplay`) that is intentionally never terminated,
   but the embedded GPU drivers it sits on top of still corrupt
   driver-internal state under concurrent operations on that shared
   display. Vivante `galcore` (i.MX 8M Plus) hits SIGSEGV at offset
   `0x18` in its ioctl path and futex deadlocks when context creation,
   DMA-BUF import, or draw commands overlap across threads; Broadcom
   V3D 7.1.10.2 (Raspberry Pi 5) drops affected contexts into
   `EGL(NotInitialized)` under similar pressure. The HAL's `GL_MUTEX`
   serializes every GL/EGL call at runtime to keep this safe;
   `--test-threads=1` extends the same discipline to test harness
   processes, since `cargo nextest` runs each test in its own process
   but multiple test processes still share the GPU driver state.
2. **G2D driver state** — the `galcore` kernel driver maintains
   per-process state that is unsafe to access from concurrent threads
   creating and destroying G2D contexts.
3. **DMA-heap CMA pool exhaustion** — concurrent DMA-heap allocations
   from multiple test threads can exhaust the CMA pool on
   memory-constrained embedded targets, masking the real failures.

`cargo nextest` already provides per-test process isolation, but `-j 1`
is still required to prevent DMA and GPU contention across test
processes.

This constraint applies to CI, the Makefile, and local development.
[`crates/image/ARCHITECTURE.md`](https://github.com/EdgeFirstAI/hal/blob/main/crates/image/ARCHITECTURE.md#gl-command-serialization-gl_mutex)
documents the GL_MUTEX implementation that makes the single-threaded rule
load-bearing.

---

## Running Tests Locally

### Native development (x86_64)

```bash
# 1. Run the full test suite (Rust + Python + C API) with coverage
make test

# 2. Rust tests only
make test-rust

# 3. Python tests only
make test-python

# 4. Rust tests manually with nextest
cargo nextest run --workspace --exclude edgefirst_hal -j 1

# 5. Tests for a specific crate
cargo test -p edgefirst-image -- --test-threads=1
cargo test -p edgefirst-decoder -- --test-threads=1
cargo test -p edgefirst-tensor -- --test-threads=1
```

### Per-language workflows

Python: see
[`crates/python/TESTING.md`](https://github.com/EdgeFirstAI/hal/blob/main/crates/python/TESTING.md)
for the `maturin develop` + `pytest` + `slipcover` workflow.

C API: see
[`crates/capi/TESTING.md`](https://github.com/EdgeFirstAI/hal/blob/main/crates/capi/TESTING.md)
for the Makefile-based C test suite, valgrind integration, and the
cbindgen header parity check.

### Disabling hardware backends

Use environment variables to isolate tests to specific backends:

| Variable | Effect |
|----------|--------|
| `EDGEFIRST_TENSOR_FORCE_MEM=1` | Force heap (`MemTensor`) allocation; skips DMA-heap and shared memory |
| `EDGEFIRST_FORCE_BACKEND=cpu` | Force CPU-only image processing |
| `EDGEFIRST_FORCE_BACKEND=opengl` | Force OpenGL backend |
| `EDGEFIRST_FORCE_BACKEND=g2d` | Force G2D backend |
| `EDGEFIRST_DISABLE_GL=1` | Disable OpenGL even when hardware is present |
| `EDGEFIRST_DISABLE_G2D=1` | Disable G2D even when hardware is present |
| `EDGEFIRST_FORCE_TRANSFER=pbo` / `=dmabuf` / `=sync` | Force GL transfer backend |

Example — run tests without any GPU backend:

```bash
EDGEFIRST_DISABLE_GL=1 EDGEFIRST_DISABLE_G2D=1 \
  cargo test --workspace -- --test-threads=1
```

---

## Cross-Compilation and On-Target Testing

The project primarily targets ARM64 embedded Linux (NXP i.MX 8M Plus,
i.MX 95). Tests that exercise DMA or GPU acceleration must run on
physical hardware.

### Step 1 — Cross-compile tests

Use `cargo-zigbuild` to produce test binaries without running them.
Exclude the Python crate (PyO3 requires a target Python installation):

```bash
cargo-zigbuild test --target aarch64-unknown-linux-gnu --release --no-run \
  --workspace --exclude edgefirst_hal
```

Test binaries land in `target/aarch64-unknown-linux-gnu/release/deps/`.

### Step 2 — Copy binaries to target

```bash
scp target/aarch64-unknown-linux-gnu/release/deps/edgefirst_image-* user@target:/tmp/hal-tests/
scp target/aarch64-unknown-linux-gnu/release/deps/edgefirst_tensor-* user@target:/tmp/hal-tests/
scp target/aarch64-unknown-linux-gnu/release/deps/edgefirst_decoder-* user@target:/tmp/hal-tests/

# Also copy testdata
scp -r testdata/ user@target:/tmp/hal-tests/
```

### Step 3 — Run on target

```bash
ssh user@target 'cd /tmp/hal-tests && EDGEFIRST_TESTDATA_DIR=$(pwd)/testdata ./edgefirst_image-<hash> --test-threads=1'
```

The `--test-threads=1` flag is mandatory on target for the same reasons
as local development — CMA pool exhaustion risk is higher on embedded
boards with limited memory.

CI automates this flow in
[`.github/workflows/test.yml`](https://github.com/EdgeFirstAI/hal/blob/main/.github/workflows/test.yml).
Binaries are stripped on the build host (split debuginfo preserved for
coverage attribution) and uploaded as the `hardware-test-binaries`
artifact for the hardware runner to download.

---

## Validating Optimizations

This section is the **testing-level reference** for the
[Optimization Guide in README.md](https://github.com/EdgeFirstAI/hal/blob/main/README.md#optimization-guide).
Each rule in that guide has a corresponding way to verify your
integration follows it; this section documents those techniques.

### Verifying EGL image cache hits

The OpenGL backend's EGL image cache emits a summary log line at process
shutdown. Run any test or benchmark with
`RUST_LOG=edgefirst_image=debug`:

```bash
RUST_LOG=edgefirst_image=debug \
  cargo test -p edgefirst-image --test ... -- --test-threads=1 2>&1 | tee cache.log
```

Look for the final line:

```text
EglImageCache stats: 999 hits, 1 misses, 1 entries remaining
```

A correctly-tuned camera pipeline reaches steady state with **misses ≈
pool depth** (one miss per unique physical buffer) and **hits ≈ frame
count**. If `misses` grows linearly with frame count, the calling code
is creating new tensor objects per frame — review the cache key (Rule 3
in the Optimization Guide).

The cache also logs `EglImageCache: swept N dead entries` when tensors
are dropped while their entries are still in the cache. Frequent sweeps
in a steady-state pipeline are a sign of leaked or churned tensor
objects.

### Verifying backend selection

The active backend is logged at `info` level when
`ImageProcessor::new()` succeeds:

```text
Shared EGL display initialized: kind=Wayland, transfer=DmaBuf
```

This `transfer=` field is the **initial EGL display capability probe**
and reports only `DmaBuf` or `Sync`. The effective transfer backend used
for conversions (after `Sync -> Pbo` upgrade and any
`EDGEFIRST_FORCE_TRANSFER` override) is logged when the GL converter is
created:

```text
GLConverter created (transfer=Pbo)
```

To confirm the final backend being exercised, set the corresponding
environment variable from the table below and re-run with
`RUST_LOG=edgefirst_image=debug`.

| Variable | Effect | Use to verify |
|----------|--------|---------------|
| `EDGEFIRST_FORCE_BACKEND=cpu` | CPU only — no GPU code path | Code paths that should never crash without a GPU |
| `EDGEFIRST_FORCE_BACKEND=opengl` | OpenGL only — no G2D, no CPU fallback | GL backend correctness |
| `EDGEFIRST_FORCE_BACKEND=g2d` | G2D only — fails on non-i.MX | G2D-specific format pairs |
| `EDGEFIRST_FORCE_TRANSFER=pbo` | PBO transfers even when DMA-buf is available | PBO path on Linux with DMA-buf hardware |
| `EDGEFIRST_FORCE_TRANSFER=dmabuf` | DMA-buf transfers; fails if EGLImage import fails | DMA-buf path on systems where it normally falls back |
| `EDGEFIRST_FORCE_TRANSFER=sync` | Memcpy upload/readback via `glTexImage2D` / `glReadnPixels` | Non-zero-copy baseline; useful for measuring fast-path cost |
| `EDGEFIRST_TENSOR_FORCE_MEM=1` | Forces heap tensors; disables DMA / SHM | Pure-CPU regression baseline |

### Warm up before measuring

The first few frames through a pipeline pay one-time costs that do not
recur in steady state: the EGL image cache populates from cold, GPU
shaders are JIT-compiled and cached by the driver, and any DMA-BUF
import that succeeds at the API level still pays its first
`eglCreateImageKHR` round trip. If a benchmark starts timing on frame
0, the first measurement is biased upward by these one-time costs and
the distribution's tail is dominated by them.

The canonical pattern is to run a small warm-up loop on the same
fixtures used by the measurement loop, with the **same tensors**, the
**same decoder**, and the **same `ImageProcessor`** that the measurement
will use:

```rust
// Warm-up — same objects, results discarded
for path in fixture_paths.iter().take(WARMUP_FRAMES) {
    backend.process_one_image(path)?;
}

// Measurement — caches are now warm, GL shaders JIT-cached
for path in fixture_paths {
    let t0 = Instant::now();
    backend.process_one_image(path)?;
    record_latency(t0.elapsed());
}
```

A typical warm-up depth is 3–5 frames for the cache itself plus enough
frames to span one full buffer-pool rotation (e.g. 9 for the i.MX 8M
Plus VPU's 1080p pool). Allocate any new tensor objects **before** the
warm-up phase, not between warm-up and measurement, or the measurement
will see a fresh `BufferIdentity` and miss its own warmed cache.

### Regression testing for tensor reuse

To catch regressions where a caller starts allocating a tensor per
frame, gate the test on the cache miss count reported in the
`EglImageCache stats` log line. The pattern below is illustrative — the
`alloc_count()` accessor is **not** a stable public API. In practice,
capture the log output and parse the `H hits, M misses` summary, or use
the `bench_preproc` reuse variant as the regression baseline.

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
    proc.convert(&src, &mut dst, Rotation::None, Flip::None, Crop::default()).unwrap();

    // Steady state — every iteration must hit the EGL image cache.
    for _ in 0..100 {
        proc.convert(&src, &mut dst, Rotation::None, Flip::None, Crop::default()).unwrap();
    }
    // Drop `proc` to flush the EglImageCache stats log line. Assert in
    // the test harness that misses == 1 (the warm-up) by parsing the log.
}
```

For the C API,
[`bench_preproc`](https://github.com/EdgeFirstAI/hal/blob/main/crates/capi/tests/bench_preproc.c)
is the reference reproduction of the allocate-once / reuse-every-frame
pattern and serves as both a benchmark and a regression test.

### Verifying the inode-keyed cache (V4L2 / libcamera integrations)

When integrating the HAL into a pipeline that consumes V4L2 or libcamera
buffers, write an integration test that simulates fd recycling:

1. Open a single `dma_buf` and obtain two distinct fd numbers via
   `dup()`.
2. Import the first fd into HAL, run a `convert()`, and capture the EGL
   cache miss count.
3. Close the first fd, then import the *second* fd (which refers to the
   same physical buffer).
4. Run `convert()` again. If the calling code uses an inode-keyed
   cache, it returns the existing tensor and the miss count does not
   increase. If it uses an fd-keyed cache or skips caching entirely,
   the miss count grows by one.

The Au-Zone GStreamer elements ship with this regression test in their
own test suite; downstream integrators should write the equivalent for
their pipeline. See
[Appendix C in ARCHITECTURE.md](https://github.com/EdgeFirstAI/hal/blob/main/ARCHITECTURE.md#appendix-c-dma-buf-identity-and-tensor-caching)
for the full identity-and-caching story.

### Verifying the `from_numpy` strided fast-path

Rule 7 in the [Optimization Guide](https://github.com/EdgeFirstAI/hal/blob/main/README.md#optimization-guide)
asserts that callers should not pre-apply `np.ascontiguousarray()` — HAL
handles strided sources internally. Two regression tests in
[`tests/test_tensor.py`](https://github.com/EdgeFirstAI/hal/blob/main/tests/test_tensor.py)
pin this:

| Test | What it verifies |
|------|------------------|
| `test_from_numpy_hailort_shape` | Correctness on the HailoRT-shaped `(1, 116, 8400)` f32 transposed view (Path 3 of `copy_numpy_to_tensor_dyn`). Destination matches source element-by-element. |
| `test_from_numpy_hailort_shape_perf_sanity` | Perf bound: automatic fast path runs no more than 1.5× the manual `np.ascontiguousarray + from_numpy(contig)` workaround. Prior to PR #58 the ratio was ≈ 10×. |

Both run in CI. If the perf-sanity test fails, suspect a regression in
`numpy.ascontiguousarray` behaviour or in the Path-3 dispatch logic in
[`crates/python/src/tensor.rs`](https://github.com/EdgeFirstAI/hal/blob/main/crates/python/src/tensor.rs).

### Verifying `MaskResolution` selection in COCO validators

Rule 8 in the [Optimization Guide](https://github.com/EdgeFirstAI/hal/blob/main/README.md#optimization-guide)
asserts that COCO / IoU evaluation must use `MaskResolution::Scaled`,
not the default `Proto`. Downstream validators should pin this with a
small assertion at the materialize call site:

```python
import edgefirst_hal as hal

masks = proc.materialize_masks(
    boxes, scores, classes, proto_data,
    letterbox=letterbox_norm,
    resolution=hal.MaskResolution.Scaled(orig_w, orig_h),
)
# Smoke-check the contract: tile shape (h, w, 1) uint8 at orig-image
# resolution, binary {0, 255} after upsample-then-threshold.
assert masks[0].shape[2] == 1
assert masks[0].dtype == np.uint8
```

If a validator's mask-mAP regresses against a numpy / ONNX reference,
inspect this call first — the most common cause is calling
`materialize_masks` with no `resolution` argument (defaults to `Proto`)
and then thresholding the proto-resolution sigmoid in caller code
before resizing, which produces blocky binary edges.

### Quantifying native CPU feature builds

When benchmarking with `RUSTFLAGS="-C target-feature=+fp16"` (or `+f16c`
on x86_64), confirm the kernel actually compiled to native instructions
rather than the soft-float helper:

```bash
scripts/audit_f16_codegen.sh aarch64-unknown-linux-gnu
```

The script disassembles a release build and fails if `__extendhfsf2` is
present (soft-float fallback) or if `fcvt` / `vcvtph2ps` is absent. See
[`crates/image/TESTING.md`](https://github.com/EdgeFirstAI/hal/blob/main/crates/image/TESTING.md#native-fp16--avx2-build-overrides-for-local-benchmarking)
for the full f16 benchmarking workflow.

---

## Coverage Thresholds

| Scope | Minimum threshold |
|-------|-------------------|
| Overall workspace | 70% line coverage |
| Critical paths (tensor ops, decoder, image processing hot paths) | 90% line coverage |

Coverage is collected using `cargo-llvm-cov` for Rust and `slipcover` for
Python. Both generate LCOV reports that are merged and reported to
SonarCloud.

Coverage reports:
- Rust: `target/rust-coverage.lcov`
- Python: `target/python-coverage.xml`

CI enforces coverage gating on pull requests. A failing coverage check
blocks the merge. See
[`.github/workflows/test.yml`](https://github.com/EdgeFirstAI/hal/blob/main/.github/workflows/test.yml)
for the merge logic between host-side and hardware-runner-side coverage.

---

## CI/CD Test Matrix

Tests run across multiple runner types:

| Job | Runner | Architecture | Hardware |
|-----|--------|--------------|----------|
| Build & Test (x86_64) | `ubuntu-22.04-xlarge` | x86_64 | No GPU |
| Doc Tests | `ubuntu-22.04-xlarge` | x86_64 | No GPU |
| Build & Test (macOS) | `macos-latest` | arm64 (Apple Silicon) | No GPU |
| Build (aarch64) | `ubuntu-22.04-arm-xlarge` | aarch64 | No GPU (compile only) |
| Test (aarch64) | `ubuntu-22.04-arm` | aarch64 | No GPU |
| Hardware Test (imx8mp) | `nxp-imx8mp-latest` | aarch64 | G2D, DMA-heap |
| Process Hardware Coverage | `ubuntu-22.04-arm` | aarch64 | (post-processing host) |

The hardware runner (`nxp-imx8mp-latest`) is the only environment where
G2D and DMA-BUF tests are fully exercised. Hardware-gated tests that
return early on x86 and arm runners are counted as passed (not skipped)
because the gate is an explicit probe, not a `#[ignore]` attribute.

The x86_64 build steps moved to `ubuntu-22.04-xlarge` (16 vCPU) in the
v0.22 → v0.23 cycle, cutting build time roughly 30 min → ~12 min.
Hardware-test binaries are stripped on the build host into a
`hardware-test-binaries-stripped/` directory and uploaded under the
artifact name `hardware-test-binaries`; unstripped originals are
preserved as the `coverage-binaries-aarch64` artifact so source-line
attribution still works during the post-target coverage merge.

---

## Common Failures and Remedies

| Symptom | Likely cause | Remedy |
|---------|--------------|--------|
| Intermittent EGL segfault | Tests running in parallel | Ensure `--test-threads=1` / `-j 1` |
| `DMA-heap allocation failed` on target | CMA pool exhaustion from parallel tests | Reduce parallelism; use `-j 1` |
| `cargo-nextest not found` | Tool not installed | `cargo install cargo-nextest` |
| `cargo-llvm-cov not found` | Tool not installed | `cargo install cargo-llvm-cov --locked` |
| Python tests fail with `ModuleNotFoundError` | Bindings not built | `maturin develop -m crates/python/Cargo.toml` |
| C API tests fail with `libedgefirst_hal.so not found` | C library not built | `cargo build --release -p edgefirst-hal-capi` |
| GL tests all skip | No EGL display available | Expected on headless CI; set `DISPLAY` or use a virtual framebuffer |
| `EDGEFIRST_TESTDATA_DIR not set` panic | Running cross-compiled bench without env | Export `EDGEFIRST_TESTDATA_DIR=$(pwd)/testdata` on target |
| CI fmt / clippy failure | Local gate skipped | Run `make format lint check` before committing |
