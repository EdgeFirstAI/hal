# AI Assistant Development Guidelines (HAL)

Process, CI tiers, release chain, and runner policy live in the canonical
[EdgeFirstAI/.github copilot-instructions](https://github.com/EdgeFirstAI/.github/blob/main/.github/copilot-instructions.md).
This file keeps only HAL-specific workspace, testing, and platform notes.

**Version:** 1.7
**Last Updated:** September 2026

---

## EdgeFirst HAL Project-Specific Guidelines

### Project Overview

EdgeFirst HAL is a Rust-based hardware abstraction layer with Python bindings, providing zero-copy memory management, hardware-accelerated image processing, and ML model post-processing for embedded Linux, macOS, iOS, and Android platforms.

### Technology Stack

- **Language:** Rust 1.70+ with Python 3.8+ bindings
- **Build system:** Cargo workspace, maturin for Python
- **Key dependencies:** ndarray, rayon, pyo3, enum_dispatch
- **Target platforms:** Linux x86_64, ARM64 (NXP i.MX, generic embedded); macOS Apple Silicon (ANGLE + IOSurface); iOS 16+ (build + link validated in CI); Android API 26+ (native GLES + AHardwareBuffer via cargo-ndk)

### Workspace Structure

```
crates/
├── tensor/         # Zero-copy tensor abstractions (DMA-BUF, IOSurface,
│                   # AHardwareBuffer, SHM, heap, PBO backends)
├── codec/          # Image decode into tensors (JPEG, PNG)
├── image/          # Hardware-accelerated image processing (GL / G2D / CPU)
├── decoder/        # YOLO and model output decoding
├── tracker/        # Object tracking algorithms
├── tensor-abi/     # Layout-only C types for tensors
├── decoder-abi/    # Layout-only C types for detections (detect.h)
├── tensor-ffi/     # Declarations-only extern "C" to libedgefirst_tensor.so
├── tensor-capi/    # cdylib: libedgefirst_tensor.so (workspace-excluded)
├── codec-capi/     # cdylib: libedgefirst_codec.so (workspace-excluded)
├── image-capi/     # cdylib: libedgefirst_image.so (workspace-excluded)
├── decoder-capi/   # cdylib: libedgefirst_decoder.so (workspace-excluded)
├── tracker-capi/   # cdylib: libedgefirst_tracker.so (workspace-excluded)
├── python-common/  # Shared PyO3 binding code (rlib)
├── python-tensor/  # edgefirst.tensor wheel
├── python-codec/   # edgefirst.codec wheel
├── python-image/   # edgefirst.image wheel
├── python-decoder/ # edgefirst.decoder wheel
├── python-tracker/ # edgefirst.tracker wheel
├── egl/            # Vendored khronos-egl fork (dynamic loading)
├── gl/             # Vendored trimmed GLES bindings
├── bench/          # Shared benchmark harness + testdata helpers
└── gpu-probe/      # Standalone Linux GPU capability probe
```

### Naming Conventions

**Rust Types:**
- Core types: `Tensor<T>`, `TensorDyn`, `PixelFormat`, `DType`, `ImageProcessor`, `Decoder`
- Trait names: `TensorTrait<T>`, `ImageProcessorTrait`
- Enum variants: PascalCase (e.g., `DmaTensor`, `ShmTensor`, `MemTensor`)

**Python Wrapper Types:**
- Use `Py` prefix: `PyTensor`, `PyPixelFormat`, `PyImageProcessor`
- Located in `crates/python-common/src/`

### Memory Management Pattern

The HAL uses a fallback chain for memory allocation:

```rust
// Automatic fallback: DMA → Shared Memory → Heap
let tensor = Tensor::<u8>::new(&[height, width, channels], None, None)?;

// Explicit backend selection
let tensor = Tensor::<u8>::new(&[height, width, channels], Some(TensorMemory::DmaBuf), None)?;
```

- `TensorMemory::DmaBuf` is the zero-copy hardware slot on every OS:
  `DmaTensor<T>` (Linux DMA-heap), `IoSurfaceTensor<T>` (macOS/iOS
  IOSurface), `AHardwareBufferTensor<T>` (Android gralloc)
- `ShmTensor<T>`: POSIX shared memory for IPC
- `MemTensor<T>`: Standard heap allocation as fallback
- `PboTensor<T>`: GL pixel-buffer objects (allocated via `create_image`)
- Environment variable `EDGEFIRST_TENSOR_FORCE_MEM=1` forces heap allocation

**Image tensors declare CPU access.** `Tensor::image(...)` and
`ImageProcessor::create_image(...)` take a required `CpuAccess`
(`None`/`Read`/`Write`/`ReadWrite`) — hardware (GPU/NPU) access is always
implied, CPU mapping is the opt-in. `None` keeps Android allocations
eligible for vendor tile compression (UBWC/AFBC/…); precise declarations
buy cheaper mappings everywhere (read-only IOSurface locks, dma-buf sync
direction, write-combined maps). Mapping beyond the declaration is
best-effort: warn-once + `unplanned_cpu_access_count()`. The
`ImageDesc` builder (`Tensor::image_desc`/`create_image_desc`) is the
full-featured path and can request `Compression::Any`. See
`crates/tensor/ARCHITECTURE.md` § CPU access declaration.

### Error Handling

All public APIs return `Result<T, E>` with specific error types:
- `TensorError` for tensor operations
- `ImageError` for image processing
- `DecoderError` for model decoding

### Hardware Acceleration Pattern

Image processing uses a fallback chain: OpenGL → G2D → CPU

```rust
impl ImageProcessorTrait for ImageProcessor {
    fn convert(&mut self, src: &TensorDyn, dst: &mut TensorDyn,
               rotation: Rotation, flip: Flip, crop: Crop) -> Result<()> {
        if let Some(opengl) = &mut self.opengl {
            match opengl.convert(src, dst, rotation, flip, crop) {
                Ok(_) => return Ok(()),
                Err(_) => { /* fall through */ }
            }
        }
        if let Some(g2d) = &mut self.g2d {
            match g2d.convert(src, dst, rotation, flip, crop) {
                Ok(_) => return Ok(()),
                Err(_) => { /* fall through */ }
            }
        }
        self.cpu.convert(src, dst, rotation, flip, crop)
    }
}
```

The GL backend is a single engine behind the compile-time `GlPlatform`
seam (`crates/image/src/gl/platform/`): native EGL + DMA-BUF import on
Linux, ANGLE (EGL→Metal) + IOSurface on macOS/iOS, native EGL +
AHardwareBuffer EGLImage import on Android.

### Cross-Compilation with zigbuild

**Always cross-compile for aarch64 using `cargo-zigbuild` during development.**
The project targets ARM64 embedded Linux (NXP i.MX 8M Plus). Do not use a
`.cargo/config.toml` with Yocto SDK linker settings — zigbuild provides its own
linker and sysroot via Zig.

```bash
# Build all crates for aarch64
cargo-zigbuild zigbuild --target aarch64-unknown-linux-gnu --release --workspace

# Cross-compile unit tests (without running them)
cargo-zigbuild test --target aarch64-unknown-linux-gnu --release --no-run \
    --workspace --exclude edgefirst-python-tensor --exclude edgefirst-python-codec \
    --exclude edgefirst-python-image --exclude edgefirst-python-decoder \
    --exclude edgefirst-python-tracker

# Run tests on target hardware (after scp to the device)
ssh <target> 'cd /tmp/hal-tests && ./edgefirst_image-<hash> --test-threads=1'
```

The Python crates (`edgefirst-python-*`) must be excluded from zigbuild cross-compilation
because PyO3 requires `PYO3_CROSS_PYTHON_VERSION` or a target Python installation.
Python wheels are built separately via `maturin` in CI.

**Mobile targets use their native toolchains, not zigbuild, and this repo's
mobile responsibility stops at the native Rust API compiling and linting
clean** — not a C artifact, not bindings, not packaging. `mobile-sdk`
binds to `edgefirst-tensor`/`-image`/`-codec`/`-decoder`/`-tracker`
directly via boltffi and owns everything above that line. Android builds
via `cargo ndk -t arm64-v8a -t x86_64 -P 26 build --release -p
edgefirst-tensor -p edgefirst-image -p edgefirst-codec -p
edgefirst-decoder -p edgefirst-tracker`; iOS via `cargo build --target
aarch64-apple-ios[-sim] --release` with the same `-p` list. No link
closure is validated here — that (against the NDK system libraries /
ANGLE + Apple frameworks respectively) is `mobile-sdk`'s concern.

### Pre-Commit Verification (MANDATORY)

**You MUST run `make format lint` before EVERY commit.** No exceptions.
**You MUST run `make sbom` before EVERY pull request.** No exceptions.

These are non-negotiable gates. If either fails, fix the issue before
proceeding. Do not skip, defer, or rationalize skipping these steps.

> **Quick CI hard-fails `cargo fmt --check`.** Keep running `make format`
> locally so the gate is not the first place drift shows up.

> **`ruff` MUST be installed in the local `./venv/`.** The `format-python` and
> `lint-python` Make targets run ruff only `if [ -f "venv/bin/ruff" ]` —
> otherwise they print a warning and **silently pass without touching Python
> code**. Without ruff in the venv, `make format` formats Rust only and Python
> drift slips through undetected. One-time setup:
>
> ```bash
> # Install Python tooling into the LOCAL venv (never the global environment)
> python3 -m venv venv          # if ./venv does not exist yet (it is gitignored)
> venv/bin/pip install ruff
> venv/bin/ruff --version        # confirm make format/lint will run ruff
> ```

```bash
# Before EVERY commit:
make format lint

# Before EVERY PR (in addition to format + lint):
make sbom
```

For a full pre-commit verification (recommended but not always required):

```bash
make format lint check test sbom
```

This runs, in order:
1. **format** — `cargo fmt --all` and `ruff format`
2. **lint** — `cargo clippy -- -D warnings` and `ruff check`
3. **check** — `cargo check --features opengl,ndarray --workspace` plus an
   `--exclude` for every `edgefirst-python-*` crate, as the `check` target
   spells it:

   ```bash
   cargo check --features opengl,ndarray --workspace \
       --exclude edgefirst-python-common --exclude edgefirst-python-tensor \
       --exclude edgefirst-python-codec --exclude edgefirst-python-image \
       --exclude edgefirst-python-decoder --exclude edgefirst-python-tracker
   ```

   **The excludes are load-bearing — do not "simplify" them away.** The
   `python-*` crates default to `edgefirst-tensor/dynamic` while
   decoder/image/codec/tensor default to `static`, and cargo unifies features
   across everything one invocation selects. A single `--workspace` run
   therefore turns both backends on at once for `edgefirst-tensor`, which is
   permanently illegal for that crate (see below) — on a clean tree the bare
   command fails with hundreds of errors rather than checking anything. The
   same applies to `cargo build`, `cargo test` and `cargo clippy`: run them
   through `make`, which carries the excludes, rather than by hand.

   Also never `--all-features`: `edgefirst-tensor`'s `static`/`dynamic`
   backend features are mutually exclusive by design, so enabling every
   feature at once is permanently illegal for this crate, not merely unusual.
   The excludes above exist to avoid reaching that same state by accident.
4. **test** — Rust tests with coverage (`cargo llvm-cov nextest`) and Python tests
5. **sbom** — SBOM generation and license policy validation

If any step fails, fix the issue before committing. Do not skip or ignore
failures. This mirrors what CI/CD runs and prevents broken pipelines.

### Build and Test Commands

```bash
# MANDATORY before committing (see above)
make format lint check test sbom

# Build all crates (native, for local development). Go through make: a bare
# `cargo build --workspace` unifies the static and dynamic tensor backends
# and cannot compile (see the check step above).
make build

# Test all Rust code (native)
make test-rust

# Cross-compile for aarch64 (preferred for development)
cargo-zigbuild zigbuild --target aarch64-unknown-linux-gnu --release --workspace

# Build Python bindings
maturin develop -m crates/python-tensor/Cargo.toml

# Test Python bindings
python -m pytest tests/

# Format code
cargo fmt --all

# Lint (make carries the --excludes a bare --workspace clippy needs)
make lint

# Run benchmarks
cargo bench -p edgefirst_image
```

### Common Tasks

**Adding a New Tensor Operation:**
1. Add method to `TensorTrait<T>` in `crates/tensor/src/lib.rs`
2. Implement for `DmaTensor`, `ShmTensor`, `MemTensor`
3. Add Python binding in `crates/python-common/src/tensor.rs`
4. Add test in `tests/test_tensor.py`
5. Update type stubs in the owning package's `python/edgefirst/<pkg>/__init__.pyi`

**Adding a New Image Format:**
1. Add variant to `PixelFormat` enum in `crates/tensor/src/format.rs`
2. Update `channels()`, `layout()`, `is_yuv()`, `has_alpha()`, `to_fourcc()`/`from_fourcc()` methods
3. Update format conversion logic in `crates/image/src/cpu/` (convert.rs, mod.rs dispatch tables)
4. Add G2D support (if hardware supports it) — update `pixelfmt_to_fourcc()` in g2d.rs
5. Add CPU fallback and Python tests

### Code Style

**Rust:**
- Use `rustfmt` configuration in `rustfmt.toml`
- Prefer `?` over `unwrap()` in production code
- Use `Result<T, E>` for all fallible operations
- Document public APIs with `///` doc comments

**Python:**
- Follow PEP 8 style guide
- Use type hints in `.pyi` stub files
- Match Python naming (snake_case functions, PascalCase classes)

### Platform Considerations

- **Linux (NXP i.MX)**: Full hardware acceleration via G2D + native EGL/DMA-BUF
- **Linux (Generic)**: DMA-heap and OpenGL (native EGL; PBO transfer on NVIDIA)
- **macOS (Apple Silicon)**: full GL acceleration via ANGLE (EGL→Metal)
  with zero-copy IOSurface tensors — NOT CPU-only
- **iOS 16+**: same ANGLE + IOSurface architecture as macOS; CI validates
  build + link closure; the runtime app shell is a future Swift effort
- **Android (API 26+)**: native EGL/GLES backend with zero-copy
  AHardwareBuffer tensors; CI validates clippy + build + link closure;
  on-device correctness/performance gates run in the internal hal-mobile
  Device Farm harness (see TESTING.md § Android On-Device Validation)
- **Windows**: compile check only (CPU fallback paths)

When adding features:
- Always provide CPU fallback
- Use feature flags for optional hardware support
- Test on multiple platforms if possible

### GPU and Hardware Resource Cleanup

**EGL/OpenGL cleanup** uses a defense-in-depth strategy (see
[`ARCHITECTURE.md`](../ARCHITECTURE.md) and
[`crates/image/ARCHITECTURE.md`](../crates/image/ARCHITECTURE.md) for full
details). Key rules:

1. **Always call `eglTerminate`** in `GlContext::drop` — EGL display connections
   are ref-counted. Omitting `eglTerminate` causes display connection exhaustion.
2. **Never call `eglReleaseThread`** — Mesa's atexit handler may have already
   freed the per-thread state, causing a use-after-free.
3. **Leak the EGL library handle** via `Box::leak` — prevents `dlclose` from
   unmapping driver code that atexit handlers still reference.
4. **Wrap all EGL cleanup in `catch_unwind`** — absorbs panics from drivers
   that misbehave during teardown.

**G2D cleanup** rules:

1. **Normal code**: Let `G2DProcessor::drop` run naturally (calls `g2d_close`).
2. **Benchmark code**: Use `ManuallyDrop<G2DProcessor>` to avoid repeated
   `g2d_close` + `dlclose` cycles that can corrupt shared `galcore` state.
3. **Never `dlclose` the G2D library** while EGL is also loaded — they share
   the Vivante `galcore` kernel driver.

**Resource leak prevention**: All GPU resources (EGL displays, contexts,
surfaces) and DMA buffers must be explicitly released. Only the EGL library
handle and `Rc<Egl>` wrapper are intentionally leaked (they are lightweight
Rust objects; actual GPU resources are freed by explicit cleanup calls).

### Testing with GPU Resources

**All tests must run single-threaded.** Use `--test-threads=1` for `cargo test`
and `-j 1` for `cargo nextest`. This prevents:
- EGL display races (concurrent `eglTerminate` tearing down shared state)
- G2D driver contention (galcore is not thread-safe for context creation)
- DMA-heap CMA pool exhaustion on memory-constrained embedded targets

This applies to CI, the Makefile, and local development. See
[`TESTING.md`](../TESTING.md) and [`ARCHITECTURE.md`](../ARCHITECTURE.md)
for details.

### Common Pitfalls

1. **Don't mix tensor types unnecessarily** - Use `None` for automatic selection
2. **Don't forget CPU fallbacks** - Hardware acceleration may not be available
3. **Don't unwrap in library code** - Always return `Result` for errors
4. **Don't forget Python type stubs** - Update `.pyi` when adding Python APIs
5. **Don't skip tests** - Add both Rust and Python tests for new features
6. **Don't forget documentation** - Public APIs need doc comments
7. **Don't nest `with_quantized!` macros** - Each nesting level multiplies monomorphized paths by 6 (6^N explosion). Dequantize tensors sequentially with `dequant_3d!`/`dequant_4d!` helpers, or split into independent phases that each nest at most 2 levels
8. **Don't run tests in parallel** - Always use `--test-threads=1` or `-j 1` (see above)
9. **Don't skip `eglTerminate`** - Omitting it leaks EGL display connections
10. **Don't `dlclose` GPU libraries** - Use `Box::leak` to keep driver code mapped
11. **Don't over-declare CPU access** - `CpuAccess::ReadWrite` on image constructors pins a linear layout on Android and forfeits vendor tile compression; use `None` for GPU/NPU-only buffers, `Write` for decode targets, `Read` for readback paths

---

## Documentation Standards

All public APIs must include:
```rust
/// Brief one-line description.
///
/// More detailed explanation if needed.
///
/// # Arguments
///
/// * `arg1` - Description
/// * `arg2` - Description
///
/// # Returns
///
/// Description of return value
///
/// # Errors
///
/// When this function returns an error and why
///
/// # Examples
///
/// ```rust
/// use edgefirst_tensor::Tensor;
///
/// let tensor = Tensor::<u8>::new(&[640, 480, 3], None, None)?;
/// ```
pub fn function_name(arg1: Type1, arg2: Type2) -> Result<ReturnType> {
    // Implementation
}
```

---

## Architecture & Testing Documentation

### MANDATORY: Read Before Major Changes

This repository maintains detailed architecture and testing documentation at
both the workspace level and within each sub-crate. **Before making any major
API change, architectural modification, or design pattern update, you MUST
read the relevant documents listed below.** After making such changes, you
MUST update the affected documents to stay in sync with the implementation.

### Document Map

#### Architecture Documents

| Document | Scope |
|----------|-------|
| [`ARCHITECTURE.md`](../ARCHITECTURE.md) | **Cross-crate** architecture: shared design patterns, platform support matrix (Linux / macOS / iOS / Android), performance-tracing infrastructure, DMA-BUF + AHardwareBuffer identity, tensor caching, source organization |
| [`crates/tensor/ARCHITECTURE.md`](../crates/tensor/ARCHITECTURE.md) | Backend dispatch (`Tensor<T>` → DMA-BUF/IOSurface/AHardwareBuffer/SHM/Mem/PBO), multi-plane DMA-BUF, `BufferIdentity` cache key + Android getId interning, CPU access declaration (`CpuAccess`), tile-compression metadata |
| [`crates/codec/ARCHITECTURE.md`](../crates/codec/ARCHITECTURE.md) | Image decode pipeline, strided output strategy, `ImageLoad` trait, multi-dtype support, allocation profiling |
| [`crates/image/ARCHITECTURE.md`](../crates/image/ARCHITECTURE.md) | GL/G2D/CPU backend chain, `GlPlatform` porting seam (Linux / macOS-ANGLE / Android), EGL image cache, zero-copy telemetry counters, `GL_MUTEX`, Vivante workarounds, shutdown safety |
| [`crates/decoder/ARCHITECTURE.md`](../crates/decoder/ARCHITECTURE.md) | Model-type selection, `dshape` contract, per-scale framework, fused proto path, NMS modes |
| [`crates/tracker/ARCHITECTURE.md`](../crates/tracker/ARCHITECTURE.md) | ByteTrack two-pass association, Kalman state, `DetectionBox` trait |
| [`crates/tensor-capi/ARCHITECTURE.md`](../crates/tensor-capi/ARCHITECTURE.md) | Opaque-handle C ABI; siblings link `libedgefirst_tensor.so` |
| [`crates/python-common/ARCHITECTURE.md`](../crates/python-common/ARCHITECTURE.md) | PyO3 bindings; extensions link one `libedgefirst_tensor.so` |
| [`crates/python-common/ARCHITECTURE.md`](../crates/python-common/ARCHITECTURE.md) | PyO3 bindings, numpy 3-path copy strategy, abi3 wheels |

#### Testing Documents

| Document | Scope |
|----------|-------|
| [`TESTING.md`](../TESTING.md) | **Cross-crate** testing: single-threaded execution rules, on-target gating, cross-compilation, coverage, CI matrix, Android on-device validation (Device Farm) |
| [`crates/tensor/TESTING.md`](../crates/tensor/TESTING.md) | Per-backend unit tests, DMA/SHM/IOSurface gating, host-tested Android layout + intern-policy tables, allocation benchmarks |
| [`crates/codec/TESTING.md`](../crates/codec/TESTING.md) | JPEG/PNG decode tests, strided output validation, hot-loop reuse pattern, allocation profiling, benchmark methodology |
| [`crates/image/TESTING.md`](../crates/image/TESTING.md) | GL gating via OnceLock probe, G2D gating on `/dev/galcore`, fp16 benchmarks |
| [`crates/decoder/TESTING.md`](../crates/decoder/TESTING.md) | Builder/segmentation/parity tests, per-scale NEON kernel tests, integration suites |
| [`crates/tracker/TESTING.md`](../crates/tracker/TESTING.md) | ByteTrack association tests, Kalman filter math |
| [`crates/tensor-capi/`](../crates/tensor-capi/) | C-side integration suite per modular library |
| [`crates/python-common/TESTING.md`](../crates/python-common/TESTING.md) | maturin develop workflow, pytest/slipcover coverage |

### When You MUST Reference These Documents

You must read and consult the relevant architecture and testing documents when:

1. **Adding or modifying public API surfaces** — check the crate's ARCHITECTURE.md for type contracts, invariants, and design rationale before changing signatures
2. **Changing memory management patterns** — consult `crates/tensor/ARCHITECTURE.md` and the root `ARCHITECTURE.md` for the DMA-BUF identity, fallback chain, and zero-copy contracts
3. **Modifying GPU/hardware backend code** — consult `crates/image/ARCHITECTURE.md` for EGL lifecycle, `GL_MUTEX` rules, G2D cleanup, and shutdown safety
4. **Changing decoder logic or model support** — consult `crates/decoder/ARCHITECTURE.md` for the `dshape` physical-memory-order contract, per-scale framework, and NMS pipeline
5. **Changing image decode or codec APIs** — consult `crates/codec/ARCHITECTURE.md` for the strided decode pipeline, `ImageLoad` trait, and allocation discipline
6. **Updating the C ABI** — consult `crates/tensor-capi/ARCHITECTURE.md` for the opaque-handle contract; each sibling leaf has its own header under `include/edgefirst/`
7. **Adding or changing tests** — consult the relevant crate's TESTING.md for test gating patterns, fixture conventions, and the single-threaded execution rule
8. **Cross-crate refactoring** — consult both the root `ARCHITECTURE.md` and all affected crate-level documents

### When You MUST Update These Documents

After making changes, you MUST update the corresponding documentation if:

1. **A public type, trait, or function signature changes** — update the crate's ARCHITECTURE.md module map and key types sections
2. **A new module or file is added** — add it to the crate's ARCHITECTURE.md module map table
3. **A design pattern or invariant changes** — update the architectural rationale in the affected document(s)
4. **A new test pattern or gating mechanism is introduced** — update the crate's TESTING.md test layout and running instructions
5. **A new benchmark binary is added** — update both the crate's TESTING.md and the Benchmarks section of this file
6. **Cross-crate contracts change** (e.g., tensor shape conventions, `BufferIdentity`, `PboOps` trait) — update the root `ARCHITECTURE.md` and all affected crate documents

### On-Demand Discrepancy Detection

When asked to review or audit architecture/testing documentation:

1. **Read the relevant ARCHITECTURE.md and TESTING.md documents**
2. **Cross-reference against the actual source code** — check module maps, type names, trait signatures, and documented invariants against the implementation
3. **Report any discrepancies** — identify where documentation describes behavior that no longer matches the code, missing modules/types, stale file paths, or outdated design rationale
4. **Propose specific fixes** — provide concrete edits to bring documentation back in sync with the implementation

Common discrepancy patterns to check for:
- Module map tables listing files that no longer exist or missing new files
- Documented type/trait names that have been renamed
- Described design patterns that have been superseded
- Test layout sections that don't match the current directory structure
- Benchmark tables missing new benchmark binaries
- Stale cross-references between documents

---

## Working with AI Assistants

### Best Practices

- Verify suggestions match project conventions
- Run linters after accepting suggestions
- Ensure meaningful test assertions
- Follow security best practices
- **Reference ARCHITECTURE.md and TESTING.md** documents before proposing changes to core design patterns
- **Update documentation** when implementation changes affect documented contracts

### Common Pitfalls

- **Hallucinated APIs:** Verify library functions exist
- **Outdated patterns:** Check current best practices
- **Over-engineering:** Prefer simple solutions
- **Missing edge cases:** Explicitly test boundaries
- **License violations:** AI may suggest incompatible code
- **Stale documentation:** Always check if your changes require ARCHITECTURE.md or TESTING.md updates

---

## Benchmarks

See [README.md § Benchmarking](../README.md#benchmarking) for full instructions on running, cross-compiling, and deploying benchmarks. See [BENCHMARKS.md](../BENCHMARKS.md) for collected results and analysis.

### Quick Reference

| Binary | Crate | What It Measures |
|--------|-------|-----------------|
| `tensor_benchmark` | `edgefirst-tensor` | Allocation and map/unmap latency (Heap, SHM, DMA) |
| `pipeline_benchmark` | `edgefirst-image` | Letterbox pipeline and format conversion |
| `nv_path_benchmark` | `edgefirst-image` | NV12/16/24 `ExternalSampler` vs `ShaderR8` A/B (`EDGEFIRST_NV_CONVERT_PATH=sampler\|shader`) |
| `decode_pipeline_benchmark` | `edgefirst-image` | JPEG decode → letterbox convert (strided input, HWC/CHW) |
| `nvjpeg_benchmark` | `edgefirst-image` | nvJPEG GPU decode into CUDA-backed PBO (on-target only, skips without CUDA) |
| `mask_benchmark` | `edgefirst-image` | Mask rendering paths (GL, CPU, hybrid) |
| `image_benchmark` | `edgefirst-image` | JPEG loading, convert, and resize operations |
| `decoder_benchmark` | `edgefirst-decoder` | YOLO post-processing, NMS, dequantization |
| `opencv_benchmark` | `edgefirst-image` | Cross-library comparison (requires `--features opencv`) |

The standalone `benchmarks/` workspace (excluded from the root workspace and CI
triggers) carries the JPEG decoder A/B harness: `hal_cpu`, `rust_jpeg`
(zune/image), `turbojpeg_bench`, `stb_bench`, `wuffs_bench`, plus the hardware
arms `hal_v4l2_gl`, `hal_v4l2_cpu`, `hal_nvjpeg` and the `hal_gl`/`hal_g2d`
pipeline arms — see `benchmarks/README.md` for the module map, protocol rules,
and corpus tooling.

**Key env vars:** `EDGEFIRST_FORCE_BACKEND={cpu,opengl,g2d}`, `EDGEFIRST_FORCE_TRANSFER=pbo`

**JSON convention:** `benchmarks/<platform>/<name>.json` (platforms: `imx8mp-frdm`, `imx95-frdm`, `rpi5-hailo`, `x86-desktop`)

**Updating tables:** `python3 .github/scripts/generate_benchmark_tables.py --data-dir benchmarks/`

---

## Getting Help

**Development questions:**
- Check `CONTRIBUTING.md` for setup instructions
- Review existing code for patterns
- Search GitHub Issues
- Ask in GitHub Discussions

**Security concerns:**
- Email: `support@au-zone.com` (subject: "Security Vulnerability")
- Do not disclose publicly

---

## Quick Reference

| Item | Format |
|------|--------|
| Branch | `feature/JIRA-123-description` |
| Commit (feature) | `JIRA-123: Brief description` |
| Commit (housekeeping) | `Release v0.6.0` or descriptive message |
| PR approvals | 2 for main, 1 for develop |
| Allowed licenses | MIT, Apache-2.0, BSD |
| Disallowed licenses | GPL, AGPL |
| Test coverage | 70% minimum |
| Security contact | `support@au-zone.com` |

---

*This document helps AI assistants contribute effectively while maintaining quality, security, and consistency.*
