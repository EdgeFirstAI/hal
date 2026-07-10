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
    ├── nv_path_benchmark.rs     # NV12/16/24 sampler-vs-shader A/B (EDGEFIRST_NV_CONVERT_PATH)
    ├── decode_pipeline_benchmark.rs  # JPEG decode → letterbox convert pipeline
    ├── nvjpeg_benchmark.rs      # nvJPEG GPU decode into CUDA-backed PBO (on-target: Jetson)
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

The module runs on **both Linux and macOS/ANGLE** in three tiers, selected
by per-item `cfg` gates:

| Tier | Gate | Contents |
|------|------|----------|
| Portable | none | mask/segmentation/box render, proto suite, src_rect crops, decision tables, F16 zero-copy roundtrip |
| Zero-copy | `feature = "dma_test_formats"` | `@Dma` fixtures that allocate on both platforms (RGBA/BGRA/GREY/NV*/YUYV — DMA-BUF on Linux, IOSurface on macOS): pool/recycle/steady-state, NV12/YUYV references, subview no-aliasing, NV16/NV24 Path-B oracles, odd-geometry `g01–g06`/grey/64×64 GPU-vs-CPU oracles |
| Linux-only | `cfg(target_os = "linux")` (± the feature) | display probing, PBO/CUDA destinations, packed-RGB `@Dma` incl. the int8 odd-geometry oracles (no IOSurface FourCC for 3-byte RGB), DMA stride guards, multi-plane fd imports, Neutron scenarios, NV path-selection asserts and divergence probes |

The zero-copy tier probes `is_gpu_image_buffer_available()`
(`edgefirst_tensor::is_gpu_buffer_available`) instead of the Linux-flavored
`is_dma_available()`.

**Android** deliberately has no `#[cfg(target_os = "android")]` test tier
in this module: no CI runner can execute Android GL (the `build-android`
lane is compile + clippy + link-validation only), so on-device correctness
and performance are gated by the internal hal-mobile Device Farm harness
driving the `edgefirst-android-validation` C-ABI entry points (GL-vs-CPU
F16 oracle + convert benchmark with a steady-state import-cache-miss
gate). The portable tiers above still cover the engine code Android shares
with Linux/macOS. **Acceptance bar for newly ported tests:** green under
`EDGEFIRST_GL_SERIALIZE=full` — GitHub's macOS runners expose a
paravirtualized Metal GPU that takes the Full serialization policy.
`scripts/test-macos.sh` enables `dma_test_formats` so the macOS lane runs
the first two tiers; the feature set must stay identical across its
coverage passes (see the script comment).

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
| `EDGEFIRST_NV_CONVERT_PATH=sampler` / `=shader` / `=auto` | Force the NV12 GPU conversion path (`auto` default; see BENCHMARKS.md) |
| `EDGEFIRST_EGL_CACHE_CAPACITY=<n>` | Override the per-cache EGLImage capacity (default 64) |

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
| `nvjpeg_benchmark` | [`benches/nvjpeg_benchmark.rs`](https://github.com/EdgeFirstAI/hal/blob/main/crates/image/benches/nvjpeg_benchmark.rs) | nvJPEG GPU decode into CUDA-backed PBO (`codec/jpeg/nvjpeg/rgbi/*` cells; on-target only, skips cleanly without CUDA) |
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

## Float Render Tests

GPU float preprocessing (F16/F32) is covered by gated integration and unit
tests. All skip gracefully when the required hardware or capability is absent.

| Test name | Gate | What it covers |
|-----------|------|----------------|
| `convert_f16_nchw_pbo_roundtrip` | GL available + `supported_render_dtypes().f16` | F16 `PlanarRgb` NCHW PBO: encode, render, readback, value check |
| `convert_f32_nhwc_pbo_roundtrip` | GL available + `supported_render_dtypes().f32` | F32 `Rgb` NHWC PBO: render, readback, value check |
| `convert_f32_nhwc_pbo_resize_bilinear` | GL available + `supported_render_dtypes().f32` | F32 PBO readback after bilinear resize |
| `convert_f16_nchw_dma_roundtrip` | V3D/Mali DMA-BUF + `supported_render_dtypes().f16` | F16 zero-copy DMA-BUF render path |
| `cpu_convert_u8_to_f16_planar_rgb` | none (CPU) | CPU widen: u8 resize then f16 NCHW planar layout |
| `cpu_convert_u8_to_f32_rgb` | none (CPU) | CPU widen: u8 resize then f32 NHWC layout |
| `convert_rgba_to_f16_cpu` / `*_falls_back` | GL absent or Vivante | CPU fallback engaged when GPU float unavailable |
| `supported_render_dtypes_linux_smoke` | GL available (Linux) | `supported_render_dtypes()` reports real values on Linux |
| `create_image_f32_dma_rejected` | Linux | `create_image(F32, Some(Dma))` → `NotSupported` |

**Capability probe tool:** `crates/gpu-probe` (`cargo run -p gpu-probe --
--probe-only`) measures per-target float render capability (texture FBO
renderability, F16 DMA-BUF render, PBO readback timing) and prints a
`RenderDtypeSupport` summary. Run it on each board before enabling float
paths in production pipelines.

**Vivante skip:** tests that require GPU float are automatically skipped on
Vivante GC7000UL targets where `supported_render_dtypes()` returns
`{ f16: false, f32: false }`. The CPU fallback tests (`*_cpu`, `*_falls_back`)
verify the fallback produces correct `[0, 1]` normalized output on those hosts.

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

### Coverage lanes and the float-render surface

SonarCloud aggregates LCOV from five lanes: **x86_64**, **aarch64**,
**macOS (Apple Silicon)**, **software-gl (Mesa llvmpipe)**, and
**i.MX 8M Plus**. A new-code line counts as covered if *any* lane hits it.
The float (F16/F32) preprocessing surface splits across lanes by where its
code can actually execute:

| Surface | Where it runs in CI coverage | Lane |
|---------|------------------------------|------|
| `classify_float_render`, `packed_rgba16f_layout`, `float_render_support`, `dma_f16_packed_layout`, `pbo_elem_count`/`pbo_shape`, `float_crop_uniforms`, `float_pbo_buffer_id` | Pure logic — runs everywhere | all |
| CPU U8→F16/F32 widen + `[0,1]` normalize fallback (`cpu/mod.rs`) | CPU — runs everywhere | all |
| **IOSurface F16 zero-copy GL render** (engine + `platform/angle.rs`, `iosurface_import.rs`) | ANGLE → Metal on the macOS runner's GPU | **macOS** |
| **Linux GL float PBO render** (`gl/processor/float.rs` `upload_float_src` / `draw_float_quad` / `convert_float_to_pbo`, `threaded.rs` PBO send/recv) | Mesa **llvmpipe** — software GL, no GPU needed, same GLSL | **software-gl** |
| **Linux GL float DMA-BUF render** (`gl/processor/float.rs` `convert_float_to_dma`) | Needs a real **V3D/Mali** GPU | *none — see ceiling below* |

The **macOS coverage lane** ([`scripts/test-macos.sh`](https://github.com/EdgeFirstAI/hal/blob/main/scripts/test-macos.sh)
`COVERAGE=1`) is what makes the IOSurface F16 render path count: macOS runs
OpenGL ES on ANGLE-over-Metal, so `test_macos_gl_pbuffer_cache_reuses_surfaces`
and the F16 IOSurface convert tests exercise the real GPU. Because the test
binaries must be codesigned with the library-validation entitlement before
they can `dlopen` ANGLE, the coverage build is **two-pass**: build+run
instrumented (GL tests self-skip), codesign, then re-run with
`HAL_TEST_ALLOW_DLOPEN_ANGLE=1` so the GL path executes under instrumentation.

The **software-gl coverage lane** runs the Linux GL backend on **Mesa
llvmpipe**. HAL normally *rejects* software renderers (they are slower than
the native CPU backend — see `GLProcessorST::new`), so this lane sets
`EDGEFIRST_ALLOW_SOFTWARE_GL=1` to opt past that rejection plus
`LIBGL_ALWAYS_SOFTWARE=1` / `EGL_PLATFORM=surfaceless` to bring up a headless
llvmpipe EGL display. llvmpipe executes the *same* RGBA16F/R32F fragment
shader the GPU would, so the `convert_f16_nchw_pbo_roundtrip` /
`convert_f32_nhwc_pbo_roundtrip` value-checks are numerically valid — this
lane genuinely covers the PBO render path (`upload_float_src`,
`draw_float_quad`, `convert_float_to_pbo`) without any GPU. The lane is
**bonus-coverage only** (best-effort steps); the x86/macOS/hardware lanes are
the authoritative test gates. To reproduce locally on a Linux host with Mesa:

```bash
LIBGL_ALWAYS_SOFTWARE=1 EGL_PLATFORM=surfaceless EDGEFIRST_ALLOW_SOFTWARE_GL=1 \
  cargo test -p edgefirst-image --lib -- --test-threads=1
```

### Known ceiling: Linux GL float DMA-BUF render path

After the software-gl lane, the only float-render code with no CI coverage is
the **DMA-BUF zero-copy** path — `gl/processor/float.rs::convert_float_to_dma`
and its EGLImage/DMA-BUF import. It binds a `DRM_FORMAT_ABGR16161616F`
dma-buf as the render target, which only works on a real **V3D** (Raspberry
Pi 5) or **Mali** (i.MX 95) GPU. The CI coverage matrix has no such runner
(x86_64/aarch64 are headless, the i.MX 8M Plus lane is **Vivante** where float
GL is *disabled*, `supported_render_dtypes()` → `{ f16: false, f32: false }`).
That path is therefore covered **on-target, manually** — the
`convert_f16_nchw_dma_roundtrip` test passes on rpi5-hailo and imx95-frdm —
but does not contribute to the CI coverage number until a V3D/Mali board joins
the coverage matrix. It is the only remaining float-render gap; everything
else (pure logic, CPU fallback, IOSurface F16, and the Linux PBO render path)
is covered in CI.

## Cross-References

- Project testing patterns: [../../TESTING.md](https://github.com/EdgeFirstAI/hal/blob/main/TESTING.md)
- Validating optimizations: [TESTING.md#validating-optimizations](https://github.com/EdgeFirstAI/hal/blob/main/TESTING.md#validating-optimizations)
- GL_MUTEX rationale and resource-cleanup layers: [ARCHITECTURE.md](https://github.com/EdgeFirstAI/hal/blob/main/crates/image/ARCHITECTURE.md#gl-command-serialization-gl_mutex)
- Decoder testing (proto API consumer): [../decoder/TESTING.md](https://github.com/EdgeFirstAI/hal/blob/main/crates/decoder/TESTING.md)
- Tensor backend tests: [../tensor/TESTING.md](https://github.com/EdgeFirstAI/hal/blob/main/crates/tensor/TESTING.md)
