# edgefirst-image Testing

## Test Layout

```
crates/image/
├── src/
│   ├── lib.rs              # Doc-tests on the public API surface
│   ├── cpu/
│   │   ├── tests.rs        # CPU backend conversion + mask tests
│   │   ├── convert.rs      # fast_image_resize-based format conversion
│   │   ├── resize.rs       # Resize kernels
│   │   └── masks.rs        # CPU mask materialization (f16 hot path)
│   ├── g2d.rs              # G2D backend tests (gated on /dev/galcore)
│   └── gl/
│       └── tests.rs        # OpenGL backend tests (gated via OnceLock probe)
└── benches/
    ├── image_benchmark.rs       # Convert/resize/rotate throughput
    ├── mask_benchmark.rs        # Mask decode + draw paths
    ├── mask_decode_benchmark.rs # Mask materialization vs proto-only
    ├── pipeline_benchmark.rs    # End-to-end pipeline measurements
    ├── decode_pipeline_benchmark.rs  # JPEG decode → letterbox convert pipeline
    ├── opencv_benchmark.rs      # Comparison vs OpenCV reference (optional)
    ├── sanity_check.rs          # Quick smoke tests for bench harness setup
    └── common.rs                # Shared bench fixture helpers
```

## Running Tests

```bash
# Full image suite — single-threaded is mandatory
cargo test -p edgefirst-image -- --test-threads=1

# Single backend (skips others)
EDGEFIRST_DISABLE_GL=1 EDGEFIRST_DISABLE_G2D=1 \
  cargo test -p edgefirst-image -- --test-threads=1

# Just the GL tests (skipped automatically without GPU)
cargo test -p edgefirst-image --lib gl::tests -- --test-threads=1

# Doc-tests
cargo test -p edgefirst-image --doc -- --test-threads=1
```

## Special Requirements

### Single-threaded execution is mandatory

All `edgefirst-image` tests must run with `--test-threads=1`. The image
crate is the load-bearing reason for the
[project-wide single-threaded rule](https://github.com/EdgeFirstAI/hal/blob/main/TESTING.md#single-threaded-execution)
— concurrent EGL, G2D, and DMA-heap operations crash on Vivante
hardware and risk CMA pool exhaustion on memory-constrained targets.
The
[`Makefile`](https://github.com/EdgeFirstAI/hal/blob/main/Makefile)
enforces `-j 1` automatically. See
[`ARCHITECTURE.md`](https://github.com/EdgeFirstAI/hal/blob/main/crates/image/ARCHITECTURE.md#gl-command-serialization-gl_mutex)
for the `GL_MUTEX` implementation that makes this rule load-bearing.

### GL hardware-gated tests

OpenGL tests in
[`crates/image/src/gl/tests.rs`](https://github.com/EdgeFirstAI/hal/blob/main/crates/image/src/gl/tests.rs)
use `OnceLock<bool>` to probe hardware availability once and skip if absent:

```rust
static GL_AVAILABLE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
fn is_opengl_available() -> bool {
    *GL_AVAILABLE.get_or_init(|| GLProcessorThreaded::new(None).is_ok())
}
```

A test that requires OpenGL calls `is_opengl_available()` at the start and
returns early if it returns `false`. This keeps the suite green on developer
machines without GPU hardware while still exercising the full hardware path
on target boards and CI runners.

The Neutron-scenario tests gate on `/dev/neutron0` as a platform
discriminator for i.MX 95 with Mali GPU. This device node indicates that
large-offset DMA-BUF EGLImage imports are supported — these fail with
`EGL(BadAccess)` on Vivante (i.MX 8M Plus) even without the NPU driver.

G2D tests live in
[`crates/image/src/g2d.rs`](https://github.com/EdgeFirstAI/hal/blob/main/crates/image/src/g2d.rs)
behind the `g2d_test_formats` Cargo feature and use their own DMA / G2D
availability checks (probing `/dev/galcore` and the DMA-heap nodes). They
are run by CI on the i.MX 8M Plus hardware runner. To exercise them
locally on i.MX hardware:

```bash
cargo test -p edgefirst-image --features g2d_test_formats -- --test-threads=1
```

### Device-node gates

| Device node | Tests gated |
|-------------|-------------|
| `/dev/dma_heap/linux,cma` or `/dev/dma_heap/system` | DMA-BUF tensor allocation |
| `/dev/galcore` | G2D hardware acceleration |
| `/dev/neutron0` | Neutron NPU DMA-BUF EGLImage |
| `/dev/dri/renderD128` (preferred) or `/dev/dri/card0` / `/dev/dri/card1` | DRM render/card node; the GL context opens the first that exists |

### Disabling backends for isolation

| Variable | Effect |
|----------|--------|
| `EDGEFIRST_TENSOR_FORCE_MEM=1` | Force heap allocation; skip DMA / SHM |
| `EDGEFIRST_FORCE_BACKEND=cpu` | CPU-only image processing |
| `EDGEFIRST_FORCE_BACKEND=opengl` | OpenGL-only |
| `EDGEFIRST_FORCE_BACKEND=g2d` | G2D-only |
| `EDGEFIRST_DISABLE_GL=1` | Disable OpenGL even if available |
| `EDGEFIRST_DISABLE_G2D=1` | Disable G2D even if available |
| `EDGEFIRST_FORCE_TRANSFER=dmabuf` / `=pbo` / `=sync` | Force GL transfer backend (the `sync` value pins the non-zero-copy `glReadPixels` baseline used for benchmarking) |

Example — run tests without any GPU backend:

```bash
EDGEFIRST_DISABLE_GL=1 EDGEFIRST_DISABLE_G2D=1 \
  cargo test -p edgefirst-image -- --test-threads=1
```

### LFS testdata

JPEG/PNG fixtures and proto-mask test vectors live under `testdata/` and
are tracked via Git LFS. Tests load them through
`edgefirst_bench::testdata::read*`, which honors `EDGEFIRST_TESTDATA_DIR`.
CI sets this to `${{ github.workspace }}/testdata`. Locally, the helper
falls back to `<workspace>/testdata` so no env var is needed.

## Benchmarks

The mask rendering benchmarks use a custom in-process harness
([`edgefirst-bench`](https://github.com/EdgeFirstAI/hal/tree/main/crates/bench))
instead of Criterion to avoid GPU driver crashes from fork-based benchmarking
on i.MX 8/i.MX 95 targets.

| Benchmark | Source | What it measures |
|-----------|--------|------------------|
| `image_benchmark` | [`benches/image_benchmark.rs`](https://github.com/EdgeFirstAI/hal/blob/main/crates/image/benches/image_benchmark.rs) | Format conversion + resize + rotation throughput per backend |
| `mask_benchmark` | [`benches/mask_benchmark.rs`](https://github.com/EdgeFirstAI/hal/blob/main/crates/image/benches/mask_benchmark.rs) | `decode_masks/{proto,materialize}`, `draw_masks/{cpu,opengl}`, `draw_proto_masks/{cpu,opengl}` |
| `mask_decode_benchmark` | [`benches/mask_decode_benchmark.rs`](https://github.com/EdgeFirstAI/hal/blob/main/crates/image/benches/mask_decode_benchmark.rs) | NMS + mask coefficient extraction (proto-only path) vs. full materialization |
| `pipeline_benchmark` | [`benches/pipeline_benchmark.rs`](https://github.com/EdgeFirstAI/hal/blob/main/crates/image/benches/pipeline_benchmark.rs) | End-to-end inference + decode + draw pipeline |
| `opencv_benchmark` | [`benches/opencv_benchmark.rs`](https://github.com/EdgeFirstAI/hal/blob/main/crates/image/benches/opencv_benchmark.rs) | Comparison vs OpenCV reference (requires `opencv` feature) |

Run on host:

```bash
cargo bench -p edgefirst-image --bench mask_benchmark
```

Cross-compile for a target (see
[`BENCHMARKS.md`](https://github.com/EdgeFirstAI/hal/blob/main/BENCHMARKS.md)
for the full deploy workflow):

```bash
cargo-zigbuild zigbuild --target aarch64-unknown-linux-gnu --release \
  -p edgefirst-image --features opengl --bench mask_benchmark
```

### Native fp16 / AVX2 build overrides for local benchmarking

The default build does **not** enable `+fp16` (aarch64) or `+f16c`
(x86_64) — distributed HAL binaries stay on each target triple's baseline
ISA so they run on older CPUs within the same triple. Benchmarks that need
to exercise the native f16 mask kernel
(`fused_dot_sign_f16_slice` in
[`crates/image/src/cpu/masks.rs`](https://github.com/EdgeFirstAI/hal/blob/main/crates/image/src/cpu/masks.rs))
must set `RUSTFLAGS` explicitly on the host that supports the extension.
The relevant kernel symbol is `fused_dot_sign_f16_slice` (see
[`crates/image/src/cpu/masks.rs`](https://github.com/EdgeFirstAI/hal/blob/main/crates/image/src/cpu/masks.rs)
— search for `fused_dot_sign_*`); the `_f16c` variant is the AVX2
intrinsic path enabled by `+f16c,+fma`.

```bash
# Orin Nano (Cortex-A78AE)
RUSTFLAGS="-C target-cpu=cortex-a78ae" \
  cargo bench -p edgefirst-image --bench mask_benchmark

# Generic aarch64 with FEAT_FP16 (verify the host first —
# imx8mp / Cortex-A53 will SIGILL!)
RUSTFLAGS="-C target-feature=+fp16" \
  cargo bench -p edgefirst-image --bench mask_benchmark

# x86_64 Haswell+ (enables the explicit _mm256_cvtph_ps intrinsic kernel
# in addition to the scalar path)
RUSTFLAGS="-C target-feature=+f16c,+fma,+avx2" \
  cargo bench -p edgefirst-image --bench mask_benchmark
```

Verify the release build actually compiled to native instructions (not the
soft-float `__extendhfsf2` helper) via
[`scripts/audit_f16_codegen.sh`](https://github.com/EdgeFirstAI/hal/blob/main/scripts/audit_f16_codegen.sh)
`<target-triple>`. Requires `cargo install cargo-show-asm`.

## Coverage Notes

- Participates in workspace `cargo llvm-cov`; lcov output appears under
  `crates/image/`.
- GPU-gated tests are excluded from coverage on hosts without hardware,
  which is why per-target coverage numbers vary between developer
  workstations and CI hardware runners. The hardware-test job on i.MX 8M
  Plus is the source of truth for backend-coverage parity.
- To exercise the CPU-fallback paths on a host that would otherwise
  pick OpenGL, set `EDGEFIRST_DISABLE_GL=1` locally before invoking
  `cargo test`. CI does not currently include this configuration in the
  matrix; downstream contributors covering specific fallback regressions
  may want to enable it on a per-PR basis.

## Cross-References

- Project testing patterns: [../../TESTING.md](https://github.com/EdgeFirstAI/hal/blob/main/TESTING.md)
- Validating optimizations: [TESTING.md#validating-optimizations](https://github.com/EdgeFirstAI/hal/blob/main/TESTING.md#validating-optimizations)
- GL_MUTEX rationale and resource-cleanup layers: [ARCHITECTURE.md](https://github.com/EdgeFirstAI/hal/blob/main/crates/image/ARCHITECTURE.md#gl-command-serialization-gl_mutex)
- Decoder testing (proto API consumer): [../decoder/TESTING.md](https://github.com/EdgeFirstAI/hal/blob/main/crates/decoder/TESTING.md)
- Tensor backend tests: [../tensor/TESTING.md](https://github.com/EdgeFirstAI/hal/blob/main/crates/tensor/TESTING.md)
