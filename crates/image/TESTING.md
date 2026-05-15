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
cargo test -p edgefirst-image --doc
```

## Single-Threaded Execution Is Required

All `edgefirst-image` tests must run with `--test-threads=1`. Three
independent constraints each require it:

1. **EGL display sharing** — concurrent `eglTerminate` from multiple test
   threads can tear down a display while other threads still reference it.
2. **G2D driver state** — Vivante `galcore` maintains per-process state that
   is unsafe to access from concurrent threads creating/destroying G2D
   contexts.
3. **DMA-heap CMA pool exhaustion** — concurrent DMA-heap allocations from
   multiple test threads can exhaust the CMA pool on memory-constrained
   targets, masking the real failures.

`cargo nextest` provides per-test process isolation, but `-j 1` is still
required to prevent DMA/GPU contention across processes. The project
[`Makefile`](https://github.com/EdgeFirstAI/hal/blob/main/Makefile)
enforces this automatically.

See [`ARCHITECTURE.md`](https://github.com/EdgeFirstAI/hal/blob/main/crates/image/ARCHITECTURE.md#gl-command-serialization-gl_mutex)
for the full technical rationale and the GL_MUTEX implementation.

## Special Requirements

### GL hardware-gated tests

OpenGL and G2D tests in
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

### Device-node gates

| Device node | Tests gated |
|-------------|-------------|
| `/dev/dma_heap/linux,cma` or `/dev/dma_heap/system` | DMA-BUF tensor allocation |
| `/dev/galcore` | G2D hardware acceleration |
| `/dev/neutron0` | Neutron NPU DMA-BUF EGLImage |
| `/dev/dri/renderD128` | DRM render node; required for DMA-buf backend |

### Disabling backends for isolation

| Variable | Effect |
|----------|--------|
| `EDGEFIRST_TENSOR_FORCE_MEM=1` | Force heap allocation; skip DMA / SHM |
| `EDGEFIRST_FORCE_BACKEND=cpu` | CPU-only image processing |
| `EDGEFIRST_FORCE_BACKEND=opengl` | OpenGL-only |
| `EDGEFIRST_FORCE_BACKEND=g2d` | G2D-only |
| `EDGEFIRST_DISABLE_GL=1` | Disable OpenGL even if available |
| `EDGEFIRST_DISABLE_G2D=1` | Disable G2D even if available |
| `EDGEFIRST_FORCE_TRANSFER=pbo` / `=dmabuf` | Force GL transfer backend |

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

Cross-compile + deploy to target:

```bash
make bench-arm
```

### Native fp16 / AVX2 build overrides for local benchmarking

The default build does **not** enable `+fp16` (aarch64) or `+f16c`
(x86_64) — distributed HAL binaries stay on each target triple's baseline
ISA so they run on older CPUs within the same triple. Benchmarks that need
to exercise the native f16 mask kernel
(`fused_dot_sigmoid_f16_slice` in
[`crates/image/src/cpu/masks.rs`](https://github.com/EdgeFirstAI/hal/blob/main/crates/image/src/cpu/masks.rs))
must set `RUSTFLAGS` explicitly on the host that supports the extension:

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
- `EDGEFIRST_DISABLE_GL=1` runs are added to the matrix in
  [`.github/workflows/test.yml`](https://github.com/EdgeFirstAI/hal/blob/main/.github/workflows/test.yml)
  to keep the CPU-fallback paths covered even on machines where GL would
  succeed.

## Cross-References

- Project testing patterns: [../../TESTING.md](https://github.com/EdgeFirstAI/hal/blob/main/TESTING.md)
- Validating optimizations: [TESTING.md#validating-optimizations](https://github.com/EdgeFirstAI/hal/blob/main/TESTING.md#validating-optimizations)
- GL_MUTEX rationale and resource-cleanup layers: [ARCHITECTURE.md](https://github.com/EdgeFirstAI/hal/blob/main/crates/image/ARCHITECTURE.md#gl-command-serialization-gl_mutex)
- Decoder testing (proto API consumer): [../decoder/TESTING.md](https://github.com/EdgeFirstAI/hal/blob/main/crates/decoder/TESTING.md)
- Tensor backend tests: [../tensor/TESTING.md](https://github.com/EdgeFirstAI/hal/blob/main/crates/tensor/TESTING.md)
