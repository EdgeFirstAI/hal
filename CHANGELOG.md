# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed

- **BREAKING: image constructors require a `CpuAccess` declaration.**
  `Tensor::image`/`image_with_stride`/`image_with_capacity`,
  `TensorDyn::image`/`image_with_stride`, `ImageProcessor::create_image`,
  capi `hal_tensor_new_image` + `hal_image_processor_create_image` (new
  `HalCpuAccess` enum), and Python `create_image`/`Tensor.image`
  (keyword `access`, strict `"none"` default) all take the declared CPU
  involvement. Hardware (GPU/NPU/ISP/codec) access is always implied;
  CPU access is the opt-in that selects the mapping mode (write-combined
  for `Write`, cached/read-only locks and dma-buf sync direction for
  `Read`) and, on Android, the gralloc CPU usage bits — a hardware-only
  (`None`) buffer is eligible for vendor tile compression. Mapping
  beyond the declaration stays best-effort but warns once per buffer and
  counts in the new `unplanned_cpu_access_count()`
  (`hal_unplanned_cpu_access_count`); Android hardware-only buffers
  refuse CPU maps deterministically.

  Migration: append the access argument to every image-constructor call —
  `CpuAccess::ReadWrite` reproduces the previous implicit behavior
  byte-for-byte; declare precisely (`Write` for decode targets, `Read`
  for verification readers, `None` for hardware-only destinations) to
  pick up the cheaper mappings. `map()` itself is unchanged (ReadWrite
  semantics); the new `map_with`/`map_read`/`map_write`/`map_mut` are
  opt-in. Python scripts that call `map()`/`numpy()` must pass
  `access="readwrite"` (the default is strict `"none"`).

### Added

- **Tile-compression image metadata and the `ImageDesc` creation path.**
  `Compression::{Any, Scheme(..)}` requests and the recorded
  `CompressionScheme::{Ubwc, Afbc, Pvric, Dcc}` become optional image
  metadata (sibling to colorimetry; preserved by `configure_image`,
  inherited by views). `ImageDesc` + `Tensor::image_desc` /
  `TensorDyn::image_desc` / `ImageProcessor::create_image_desc` carry
  the request (capi: opaque `hal_image_desc_*` handle,
  `hal_tensor_new_image_desc`, `hal_image_processor_create_image_desc`,
  `hal_tensor_compression`, `hal_platform_compression_support`; Python:
  `compression=` keyword + `Tensor.compression`). `Any` resolves linear
  with a counted fallback (`compression_fallback_count()`); `Scheme`
  errors on mismatch. The device scheme is classified from
  `ro.hardware.egl` (host-tested table).
- **Android `BufferIdentity` interning on `AHardwareBuffer_getId`**
  (API 31+, dlsym-resolved): every re-wrap of the same buffer shares one
  identity, so CameraX/ImageReader re-wrap pipelines hit the EGLImage
  import cache in steady state; API 26–30 keeps fresh-per-wrap
  identities (correct, uncached, visible in miss counters).
- **Device Farm validation cells** for the access/compression/identity
  surface: CPU-usage sweep, write-combined mapping bench, hardware-only
  map contract, getId-interning window, and compressed-render
  round-trip (`edgefirst-android-validation`).
- **Android support: AHardwareBuffer zero-copy tensor storage and a native
  OpenGL ES backend** (min API 26, `aarch64-linux-android` +
  `x86_64-linux-android`). Android becomes the third `GlPlatform`
  implementation, following the architecture validated on-device by the
  Phase-1 `hal-mobile` probe (Galaxy S26 Ultra):
  - **`AHardwareBufferTensor`** fills the `TensorMemory::Dma` slot (the role
    DMA-BUF plays on Linux and IOSurface on macOS/iOS): raw
    `libnativewindow` FFI (stable NDK ABI since API 26), lock/unlock CPU
    maps carrying the GPU↔CPU coherency contract, gralloc-stride-aware
    geometry, BLOB byte-bags for generic tensors, and RGBA8/RGBA16F image
    buffers sharing the planar-F16 RGBA16F packing with the macOS path. New
    public API: `Tensor::from_hardware_buffer`, `hardware_buffer_ptr`,
    `hardware_buffer_physical_dims` (+ `TensorDyn` equivalents),
    `image_ahardwarebuffer_layout`, `is_ahardwarebuffer_available`.
  - **`AndroidEgl` GL backend**: native default-display EGL (no ANGLE, no
    GBM) with per-processor contexts; zero-copy import via
    `EGL_ANDROID_image_native_buffer` with **persistent** texture bindings
    (like Linux DMA-BUF, so the binding-skip cache applies). New
    `TransferBackend::AHardwareBuffer`.
  - **`edgefirst-android-validation`** crate: link-closure staticlib plus
    C-ABI entry points that run the real `ImageProcessor` on-device (GL-vs-
    CPU F16 oracle, convert benchmark with a steady-state import-cache miss
    gate) for the hal-mobile AWS Device Farm harness. New
    `scripts/build-android.sh` and `scripts/validate-android-link.sh`, and a
    `build-android` CI lane (clippy + build + link validation for both ABIs,
    pinned NDK r27c).
  - Deferred follow-ups (fall back to CPU today): YUV camera buffers
    (external-OES sampling), Grey/NV single-plane zero-copy (`R8_UNORM`
    needs API 29), and shared-memory *allocation* (bionic has no
    `shm_open`; importing an existing segment via `from_fd` works).
- **True end-to-end zero-copy on the float convert paths + NPU-direct
  output surface** (all platforms; measured on macOS/ANGLE and on-device):
  - **Float-path zero-copy source import**: the F16/F32 render family
    previously CPU-uploaded its RGBA source every frame on every platform
    (the largest silent zero-copy degrade found by a full engine audit —
    ~3.2 ms/frame at 1080p on-device). `feed_float_src` now imports
    Dma-backed sources as EGLImages with a per-driver fallback chain
    (escape hatch: `EDGEFIRST_GL_NO_FLOAT_SRC_IMPORT=1`). macOS medians:
    720p RGBA→PlanarF16 762→363 µs (2.1×), 1080p 1422→351 µs (4.1×).
  - **`ConvertStats` telemetry**: per-feed counters (`src_imports`,
    `src_pbo_uploads`, `src_uploads`, `zero_copy_declines`) via
    `GLProcessorThreaded::convert_stats()`, a `src_feed` field on the
    `image.convert.gl` span, `ImageProcessor::convert_fallback_count()`,
    and warn-once (per buffer) NV import failures — every silent fallback
    is now loud and countable.
  - **Explicit-Dma contract**: `Some(TensorMemory::Dma)` image requests
    with no zero-copy mapping now return `InvalidArgument` on
    macOS/Android instead of silently allocating a byte-bag no GL import
    can bind (semi-planar u8 on macOS keeps its designed `'L008'` R8
    mapping).
  - **Packed RGB u8/i8 zero-copy on Android AND macOS** (the INT8 NPU
    input layout): new `packed_rgb888_layout` shared by allocation and
    render ((W·3/4, H) RGBA8888 texels), wired into both the
    AHardwareBuffer format table and the IOSurface FourCC table (on
    macOS this replaces the accidental `'L008'` byte-bag the two-pass
    shader used to bind); I8 rides the same layout on both platforms.
    The packed-RGB GL correctness tests now run on macOS too.
  - **NPU-direct C API**: `hal_tensor_from_hardware_buffer`,
    `hal_tensor_hardware_buffer_ptr`/`_physical_dims`,
    `hal_tensor_recorded_row_stride`, `hal_tensor_effective_row_stride`,
    and `hal_tensor_copy_to_flat` (+ `Tensor/TensorDyn::copy_to_flat`) —
    consume the convert output with zero CPU readback, with the padded-
    pitch flatness contract documented (README § NPU-direct output).
  - **GL→NPU fence export**: `convert_with_fence` /
    `hal_image_processor_convert_fence` return an
    `EGL_ANDROID_native_fence_sync` fd instead of blocking in `glFinish`,
    so the NPU waits on the GPU directly
    (`ANeuralNetworksExecution_startComputeWithDependencies`); platforms
    without native fences keep the blocking contract and return no fd.
  - Float-draw micro-optimizations: uniform locations cached per program
    and the constant full-screen quad moved to static VBOs.

## [0.25.3] - 2026-06-24

### Changed

- **Dependency hygiene: abandoned crates vendored or replaced, dependency tree
  unified.** No behavioral change — decoder outputs are bit-identical and the
  GPU paths are unchanged; this release only refreshes and tidies what the
  workspace carries and statically analyses.
  - **`khronos-egl` 6.0.0 → vendored `edgefirst-egl`** (`crates/egl`): the
    upstream crate is dormant (last release 2023). Vendored as a maintained,
    dynamic-only fork (imported as `edgefirst_egl`) with the static-linking
    path, `pkg-config` build dependency, and `build.rs` removed. The public API
    is unchanged, so consumers only needed the import identifier renamed.
    glutin/glow were ruled out — neither provides the EGLImage/dmabuf import the
    HAL relies on, and glutin is itself still on `libloading` 0.8.
  - **`gls` 0.1.6 → vendored `edgefirst-gl`** (`crates/gl`): the upstream crate
    is abandoned (last release 2021). Vendored as a trimmed fork (imported as
    `edgefirst_gl`) carrying only the surface the HAL uses — the raw GLES 3.2 +
    EGLImage bindings, 12 safe wrappers, and the `Error` type — which drops its
    `nalgebra`, `serde`, `libc`, and `winapi` dependencies. The result has no
    runtime dependencies. A full glow migration was rejected (~2730 call sites,
    paradigm shift, GPU re-validation, and it would not remove the raw EGLImage
    extension code). Both vendored crates have no path-only deps and are ready
    to publish to crates.io standalone.
  - **`fast-math` dropped** (`edgefirst-decoder`): upstream abandoned (last
    release 2019). Its sole use — an approximate sigmoid in the ModelPack
    decoder — now calls a bit-for-bit vendored Schraudolph `exp_raw`
    (MIT/Apache-2.0, Huon Wilson), so decoder outputs stay byte-identical to
    prior releases (verified by the existing golden-value and cross-path
    decoder tests).
  - **`serde_yaml` → `serde_yaml_ng`**: the original (dtolnay) is archived and
    deprecated; the API-identical maintained continuation is aliased via a
    package rename so all `serde_yaml::` paths are unchanged.
  - **`libloading` unified on 0.9** and **`g2d-sys` 1.3.1 / `edgefirst-gbm`
    0.19** bumped: the updated Au-Zone crates drop their stale transitive pins,
    collapsing the duplicate dependency tree — single `libloading` 0.9 (0.8.9
    gone), `nix` 0.31.3 (0.29 gone), `rustix` 1.x (0.38 gone), `ndarray` 0.17.2
    (0.16 gone) — and removing the stale `linux-raw-sys` 0.4.15 and entire
    `windows-*` 0.52 set that drm/gbm previously dragged in.
  - **Comprehensive `cargo update`** across the tree (`cbindgen` 0.29.2 →
    0.29.4 regenerates the C API header, plus ~15 in-semver bumps). `jiff` is
    pinned to 0.2.24 to keep an unmaintained `proc-macro-error2`
    (RUSTSEC-2026-0173) out of the lockfile as a phantom; `cargo audit` remains
    at the single pre-existing `paste` phantom.

## [0.25.2] - 2026-06-23

### Added

- **Validation-accurate multi-label box decode** (`edgefirst-decoder`): an
  opt-in decode mode (`DecoderBuilder::with_multi_label`, default `false`) that
  recovers a systematic accuracy delta versus the Ultralytics validator. It
  emits one candidate per class above threshold per anchor (matching Ultralytics
  `val multi_label=True`) instead of an argmax single class, reusing the shared
  anchor index so the mask-coefficient row is not duplicated. The shared f32
  seam forces class-aware NMS so per-class duplicates do not cross-suppress.
  Builder-only (never embedded in `edgefirst.json`); a `debug_assert` keeps the
  tracked decode paths on argmax so ByteTrack's IoU-only matching does not spawn
  phantom tracks. Default **off**, so deployment and tracking output is
  byte-for-byte unchanged. Measured +0.56 pp box / +0.45 pp mask mAP on COCO
  val2017 (yolov8n-seg).

### Fixed

- **Per-channel DFL quantization is now always honored** (`edgefirst-decoder`):
  the per-scale box DFL head reads the box tensor's quantization metadata each
  frame and dequantizes per-channel whenever the model is per-channel quantized,
  rather than collapsing to a single per-tensor scale (which flattened the wide
  anchor-free distance distribution — worst on yolo26, ~1.6–4.7 pp INT8 mAP
  loss). The per-tensor scalar fast path is still selected automatically for
  genuinely per-tensor (or float) tensors; per-channel kernels cover
  `i8`/`u8`/`i16`/`u16` (f32 and f16) and are allocation-free on the decode hot
  path. This supersedes the short-lived opt-in `with_dfl_per_channel` builder
  flag introduced in PR #122 — honoring per-channel quantization is never
  optional.

### Changed

- **Dependencies upgraded to latest** across the workspace. Notable major
  upgrades: `pyo3`, `pyo3-build-config`, `numpy`, and `pythonize` 0.28 → 0.29
  (in lockstep); `nalgebra` 0.34 → 0.35; `opencv` 0.98 → 0.99; `zip` 2 → 8;
  `safetensors` 0.7 → 0.8; `ctor` 0.4 → 1.0; and `turbojpeg` 0.5 → 1.4. Patch
  and minor bumps were applied to `ctor`, `libc`, `log`, `nix`, `rand`,
  `rayon`, `serde_json`, `tokio`, `uuid`, and `yuv`. Source changes were
  limited to the `ctor` 1.0 `#[ctor(unsafe)]` requirement and a `libloading`
  deref; no behavioural changes.
  - `libloading` is held at `0.8.x`: `khronos-egl` 6.0.0 (the latest release)
    depends on `libloading ^0.8` and the GL backend shares a
    `libloading::Library` handle with its `Dynamic` EGL instance, so a 0.9 bump
    would split the `Library` type across the crate boundary. To be revisited
    when `khronos-egl` ships a `libloading` 0.9 build.
  - `numpy` 0.29 is pinned to `ndarray` 0.17.2 (the workspace version) so its
    `IntoPyArray`/`ToPyArray` conversions resolve against the same `ndarray` the
    rest of the workspace uses; `lapjv` continues to use `ndarray` 0.16.

### Documentation

- **Corrected the GL serialization (`GL_MUTEX`) docs** in `README.md`,
  `BENCHMARKS.md`, and the umbrella `ARCHITECTURE.md`, which still described
  serialization as universal (the pre-2026-06 behavior). GL command
  serialization is now a per-driver policy — `Full` (global `GL_MUTEX`) only on
  Vivante `galcore` (i.MX 8M Plus) and virtualized/paravirtual GPUs, and
  `LifecycleOnly` (concurrent execution) on Mali/Panfrost, V3D, Tegra,
  llvmpipe, and real Apple GPUs. The summaries now match the canonical
  description in `crates/image/ARCHITECTURE.md § GL Concurrency Model` and the
  stale `#gl-command-serialization-gl_mutex` cross-doc anchors were repointed.

## [0.25.1] - 2026-06-17

### Fixed

- **Circular dev-dependency between `edgefirst-codec` and `edgefirst-image`**:
  the nvJPEG GPU decode benchmark (`bench_nvjpeg`) was moved from
  `edgefirst-codec`'s bench suite to a new `nvjpeg_benchmark` in
  `edgefirst-image`, where it naturally belongs (it needs `ImageProcessor` to
  allocate the CUDA-backed destination tensor, and `image` already depends on
  `codec`). This removes the dev-dep cycle that prevented `cargo publish` from
  packaging the crates without a `sed`-based workaround in the release workflow.

## [0.25.0] - 2026-06-16

### Removed

- **Dead legacy per-scale merge path** (`edgefirst-decoder`): removed the
  superseded `LogicalMerge::PerScale` schema-v2 merge arm and the now-orphaned
  `decoder/dfl.rs` module (−1187 LOC, no behaviour change). Per-scale FPN/DFL
  decoding is unaffected — it runs through the live `per_scale/` subsystem,
  which claims those schemas at build time; the `boxes` `reg_max` value is now
  sourced from `PerScalePlan`.

### Documentation

- **Tracker API documentation** (`edgefirst-tracker`): full rustdoc for the
  public tracking API (`DetectionBox`/`Tracker` traits, `TrackInfo`,
  `ByteTrack`, the Kalman motion model) plus a tracing span catalog in its
  `ARCHITECTURE.md`.
- The converged OpenGL engine (a single engine on Linux and macOS) is now
  described in `ARCHITECTURE.md`, and was revalidated on-target across Mali-G310
  (i.MX 95), V3D (RPi 5), NVIDIA Tegra Orin, and Vivante GC7000UL (i.MX 8M Plus)
  — 359 GL tests, 0 failures on each.

### Changed

- **Unified tensor backend `view()`** (`edgefirst-tensor`): the per-backend
  `view`/`subview` implementations are unified onto the `TensorTrait` contract
  (`subview` collapses five backend match arms into one), preserving the
  `BufferIdentity`-sharing invariant that GL EGLImage import reuse relies on.
- **Cached CPU convert intermediates** (`edgefirst-image`): the multi-step
  CPU convert pipeline (pre-resize format-convert and the resized-RGB
  scratch used by letterbox/resize) now reuses two `CPUProcessor`-held
  buffers across frames instead of allocating — and `alloc_zeroed`-clearing
  — a fresh full-frame buffer per call. Each consumer fully overwrites the
  region it reads, so reused (non-zeroed) contents are never observed, and
  output is unchanged. Measured on Orin Nano (interleaved A/B, NV12 1280×720
  → 640×640 letterbox): median −8% (2.7→2.5 ms) and, more importantly, the
  tail flattens (p95 3.3→2.6 ms, max 5.2→2.8 ms) by removing the per-frame
  mmap/page-fault spikes.
- **Strip-fused CPU NV→PlanarRgb conversion** (`edgefirst-image`): the
  no-resize NV12/NV16/NV24 → `PlanarRgb`/`PlanarRgba` conversion now decodes
  the YUV source into packed RGB one cache-resident row strip at a time
  (into a `CPUProcessor`-cached scratch) and NEON-deinterleaves each strip
  straight into the destination planes. The full-size packed-RGB
  intermediate no longer round-trips through DRAM and is no longer
  reallocated per frame, so the strip stays hot in L2 between the YUV decode
  and the deinterleave. Output is byte-for-byte identical to the previous
  two-pass path (NV12 4:2:0 replicates one chroma row across two luma rows,
  so strip boundaries do not change the per-row chroma pairing — verified
  across NV12/16/24, planar/planar-RGBA, and odd dimensions). Measured on
  Orin Nano (CPU backend, 1280×720, no resize): NV12→PlanarRgb a further
  −21% on top of the NEON deinterleave (≈ −46% vs the original scalar
  path). The letterbox path is unchanged (its resize step requires the
  full packed buffer).
- **NEON-accelerated CPU packed↔planar conversion and float widen**
  (`edgefirst-image`): the CPU `pack_to_planar` scatter (RGB/RGBA →
  `PlanarRgb`/`PlanarRgba` — the JPEG→NV→planar model-input path on the
  Jetson Orin CPU pipeline) now uses a single-pass NEON `vld3`/`vld4`
  deinterleave (one hardware-deinterleaving load reads the packed source
  once and splits the channels) instead of a per-plane scalar gather that
  re-read the source once per plane. The U8→F32 `/255` normalisation widen
  uses NEON `vmovl`/`vcvtq`/`vdivq` (bit-identical to the scalar form), and
  the U8→F16 widen gains a native half-precision kernel
  (`ucvtf.8h`+`fdiv.8h`) selected at runtime on FEAT_FP16 CPUs (e.g. the
  Orin's Cortex-A78AE), with a scalar fallback elsewhere (forceable via
  `EDGEFIRST_IMAGE_NO_FP16`). The deinterleave is memory-bandwidth-bound,
  so it now runs serially (the previous per-plane rayon fan-out added
  scheduling overhead without a throughput gain); the rare arbitrary
  channel-mapping path keeps its parallel fallback. Measured on Orin Nano
  (CPU backend, 1280×720 → 640×640 letterbox): NV12→PlanarRgb −25%,
  letterbox→PlanarRgb −9%, letterbox→PlanarRgb F16 −24% (F16 is now faster
  than F32 for model input). Output is unchanged for the integer/F32 paths
  and within ~1 f16 ULP for the native F16 path. NEON is baseline on
  aarch64; non-aarch64 targets keep the scalar path.
- **BREAKING — unified GL processor on macOS** (`edgefirst-image`):
  `ImageProcessor` on macOS now drives the same `GLProcessorThreaded`
  engine as Linux, with a dedicated worker thread and a private ANGLE
  context per processor on the shared Metal display. `MacosGlProcessor`
  is removed from the public API. Multiple `ImageProcessor` instances
  run GL work in parallel on macOS (the previous backend serialized all
  instances on one shared context), `ImageProcessorConfig::egl_display`
  exists on both platforms (ignored with a debug log on macOS), and
  `set_colorimetry_mode` / `set_int8_interpolation_mode` /
  `CacheStats` / `GlCacheStats` / `EglDisplayKind` are available on
  macOS. The IOSurface conversion paths run zero-copy through the
  engine (sources attach as `TEXTURE_2D`; YUYV via a portable
  `sampler2D` shader), and macOS inherits the engine's full conversion
  matrix — resize/letterbox for every format pair, rotation/flip, int8,
  masks — where the legacy backend was same-size YUYV/NV→RGBA only.
  Measured parallel scaling on Apple Silicon: NV12→RGBA zero-copy
  S(2)=1.62 / S(4)=3.16, F16 model-input path S(2)=1.52 / S(4)=3.13.

### Added

- **`cpu_preprocess_benchmark`** (`edgefirst-image`): a self-contained
  on-target benchmark for the CPU JPEG-preprocessing path (NV12/NV16/NV24
  → RGB / PlanarRgb / PlanarRgb-F32 / PlanarRgb-F16, same-size and
  letterbox), used to characterise and gate the Orin CPU pipeline. Sources
  are synthesised (no testdata), and it honours `EDGEFIRST_FORCE_BACKEND`,
  `EDGEFIRST_BENCH_ITERS`, and `EDGEFIRST_BENCH_ONLY` for `perf` profiling
  of individual cells.
- **Fused NV12/NV16/NV24 → PlanarRgb F16 GL convert** (`edgefirst-image`):
  the model-input conversion (YUV source → letterboxed planar F16
  tensor) now runs as two GL passes inside one `convert()` call on every
  platform with F16 render support — macOS IOSurface and Linux DMA-BUF
  alike (previously macOS-only via the legacy backend, CPU on Linux).
  The intermediate between the passes is a GPU-resident texture: the
  pixel path is zero-copy source → GPU → texture → GPU → zero-copy
  destination, with no host round-trip and no synchronization between
  the passes (same-context command ordering). Guarded by a
  cross-platform oracle against the CPU reference and a backend
  assertion that proves the GL engine did the work.

### Fixed

- **Virtualized GPUs get the Full GL serialization policy**
  (`edgefirst-image`): on paravirtual Metal (e.g. GitHub's macOS CI
  runners, macOS VMs) concurrent GL across per-processor contexts
  mis-renders — parallel converts produced 60–86% diverged output
  bytes against each processor's own sequential oracle. Renderers
  matching `Paravirtual`/`virtio` in `GL_RENDERER` now serialize
  per-message like Vivante (`EDGEFIRST_GL_SERIALIZE=lifecycle`
  overrides); real Apple GPUs keep full parallelism. Surfaced by the
  parallel-correctness gate once the `glReadnPixels` fix below made
  its converts genuinely GPU-executed on macOS.

- **GL readbacks on macOS used an ES 3.2-only entry point**
  (`edgefirst-image`): every heap-destination readback called
  `glReadnPixels`, which ANGLE/Metal (ES 3.0) rejects with
  `GL_INVALID_OPERATION` — the engine errored on any Mem-destination
  convert, mask draw, or PBO readback on macOS and silently fell back
  to CPU. All readbacks now use plain `glReadPixels` (core since ES
  2.0, identical semantics — the destination size is validated by
  construction), so heap-destination paths genuinely run on the GPU
  on macOS. Guarded by a new direct-engine regression test, and the
  fused-convert oracle / backend-assertion tests are strengthened so
  a GL failure can no longer hide behind the CPU fallback.

- **GL proto-mask texture lifecycle** (`edgefirst-image`): two latent bugs
  with one root cause. (1) The experimental GLES 3.1 compute-shader proto
  repack (`EDGEFIRST_PROTO_COMPUTE=1`) rendered garbage masks on every
  conformant driver — `glBindImageTexture` requires immutable-format
  textures in ES 3.1, so its `imageStore` writes never landed. (2)
  Alternating proto dtypes (int8 ↔ float) on one processor raised
  `GL_INVALID_VALUE`: the float paths re-allocated the shared proto
  texture without updating the dims gate, so the next int8 upload wrote
  into an incompatible allocation. Proto textures are now immutable
  `TexStorage3D` allocations recreated on any dims/format change, which
  also gives the f32/f16 paths the re-upload fast path they lacked (they
  re-ran `glTexImage3D` on every call) and fixes an undefined
  `texelFetch` past the array depth in the int8 two-pass dequant shader
  when `num_protos % 4 != 0`.

### Changed

- **Issue #106 closed — imx8mp GL drift reclaimed and documented**:
  `image_benchmark` regains the same-size 1080p RGBA→BGRA / RGBA→GREY
  cells as permanent drift sentinels, verified back at the v0.15.0
  numbers on imx8mp (7.7 / 7.9 ms, from 13.8 / 14.9 ms at the
  2026-06-09 drift capture). BENCHMARKS.md's GL convert, letterbox, and
  mask tables are re-collected on the convergence engine with per-table
  capture notes (imx95 YUYV→RGB 1080p halved to 5.7 ms; the mask hybrid
  path is ~3× faster than the stale v0.15.0 figures).
- **GL proto-segmentation render internals** (`edgefirst-image`): the
  per-dtype render variants collapsed into one plan-driven path selected
  by a pure, host-tested decision table (`proto_dispatch::plan_proto`).
  The int8 two-pass dequant target and FBO are now persistent and
  dims-gated instead of re-created per call, and per-detection /
  per-dispatch uniform locations are cached instead of looked up by
  string each draw. New `image.draw.gl.proto` tracing span records the
  chosen upload strategy and program. No public API change; outputs are
  unchanged (pinned by new cross-dtype, fallback, compute, churn, and
  interpolation-mode regression tests).

### Added

- **Parallel `ImageProcessor` execution on non-Vivante GPUs**
  (`edgefirst-image`): the process-global GL lock is now a per-driver
  serialization policy. On Vivante (i.MX 8M Plus) every operation still
  serializes globally (the galcore driver is not thread-safe across
  contexts); on Mali, V3D, Tegra, llvmpipe — and any platform without a
  known-bad driver — multiple `ImageProcessor` instances now genuinely
  execute GPU work in parallel, each on its own dedicated thread and GL
  context. Measured aggregate scaling for a 720p NV12→RGB letterbox:
  Mali G310 S(4)=2.05, V3D S(4)=1.16, Tegra S(4)=1.17 (Vivante flat by
  design). `EDGEFIRST_GL_SERIALIZE=full|lifecycle` overrides the policy
  (`full` also mitigates a measured Tegra context-switch collapse on
  degenerate tiny-convert multi-processor workloads). New
  `parallel_processors_benchmark` and a cross-wire-sensitive parallel
  correctness test demonstrate and pin the behavior; processor
  creation/teardown and EGLImage create/destroy remain briefly
  serialized on every platform.
- **`ColorimetryMode` (`Fast` | `Exact`) + `EDGEFIRST_COLORIMETRY`**
  (`edgefirst-image`): the colorimetry/performance trade-off is now an
  explicit knob. `ImageProcessorConfig::colorimetry` (default
  `ColorimetryMode::Fast`, issue #106 policy) or the
  `EDGEFIRST_COLORIMETRY=fast|exact` environment variable (which takes
  precedence and pins the mode for the processor's lifetime). **Behavior
  change on Vivante (i.MX 8M Plus):** under `auto`, single-plane 4-aligned
  NV12 sources now take the hardware external sampler for *every*
  colorimetry by default (~12× faster: 2.5 ms vs 29 ms at 720p; the
  driver applies its fixed BT.601-limited matrix, approximate for
  non-BT.601-limited sources). `ColorimetryMode::Exact` restores the
  previous behavior (sampler only when the driver matrix matches the
  source exactly, colorimetry-exact in-shader matrix otherwise). All other
  GPUs are unaffected — the exact in-shader path is already the fast path
  there.
- **DMABUF V4L2 hardware JPEG decode** (`edgefirst-codec`): the CAPTURE queue
  now imports DMA buffers instead of allocating MMAP buffers, so a geometry
  change costs ~1 ms of ioctls instead of a ~110 ms kernel buffer
  reallocation. Two decode targets, tried in order: zero-copy into the
  destination DMA tensor (MCU-aligned NV12/GREY), or a persistent codec-owned
  DMA scratch with copy-out (NEON YUV24→NV24 deinterleave for 4:4:4). The
  OUTPUT (coded) buffer is allocated once with headroom and survives geometry
  changes. Hardware decode requires DMA buffer allocation (dma_heap), in line
  with the HAL's DMABUF-centric design; platforms without it use the CPU
  decoder. i.MX95, COCO val 5K (~1000 distinct resolutions): decode mean
  128.7 ms → 7.0 ms, 2.5× faster than the CPU decoder; raw hardware
  throughput ~350 FPS at 720p (`probe_decode_throughput`). New `v4l2_scan`
  example flags images that decode slowly through the hardware path.
- **Batched preprocessing — `convert_deferred()` + `flush()`** (`edgefirst-image`,
  C API, Python): render `N` model inputs into row-band views of one batched
  destination as **one GPU import + one sync**. Loop
  `convert_deferred(src, &mut dst.batch(n), …)` then call `flush()` once. A
  `view()`/`batch()` destination resolves its **parent** (the new
  `Tensor::view_origin()` snapshot), so the OpenGL backend keys the destination
  EGLImage on the parent identity+geometry — every sibling tile shares a single
  import and renders to its band via `glViewport`/`glScissor` (the tile offset is
  render state, never a cache key). `convert_deferred` skips the per-tile
  `glFinish`; `flush` issues one `finish_via_fence`; the CUDA `cuda_map` path
  auto-flushes a pending batch before mapping, so device reads never see an
  in-flight render. Non-GL backends (CPU/G2D) run an eager per-tile convert and
  treat `flush` as a no-op. C API: `hal_image_processor_convert_deferred()` /
  `hal_image_processor_flush()`; Python: `ImageProcessor.convert_deferred()` /
  `.flush()`. **v1 scope:** the single-pass `Rgba`/`Bgra`/`Grey` u8/i8 DMA path;
  two-pass packed-`Rgb`, planar, non-DMA/PBO/float GL, G2D, and macOS GL decline a
  view destination and fall back to the CPU backend (correct via offset + parent
  stride). Validated on-target (rpi5/V3D, orin-nano/CUDA, i.MX95, i.MX8MP/Vivante).
- **GPU NV12/NV16/NV24 conversion on non-DMA (heap/PBO) backends**
  (`edgefirst-image`): the in-shader R8 (`ShaderR8`) path now also runs when
  DMA-BUF EGLImage import is unavailable (e.g. NVIDIA Jetson Orin) by uploading
  the combined semi-planar buffer as an R8 texture, replacing the CPU fallback.
- **`EDGEFIRST_NV_CONVERT_PATH`** (`sampler` | `shader` | `auto`,
  `edgefirst-image`): selects the NV12 GPU conversion path. `auto` (default)
  prefers the portable, colorimetry-exact in-shader `ShaderR8`, except
  BT.601-limited single-plane NV12 on Vivante (where the hardware sampler is
  ~12× faster and colorimetry-correct). `sampler`/`shader` force a path for
  benchmarking and platform bring-up.
- **`EDGEFIRST_EGL_CACHE_CAPACITY`** (`edgefirst-image`): overrides the per-cache
  EGLImage capacity (default 64) for high-cardinality varied-geometry streams.
- `nv_path_benchmark` (`edgefirst-image`): cross-platform `sampler`-vs-`shader`
  NV12/16/24 A/B benchmark (synthesized sources, no testdata required).
- `Tensor::subview()` / `TensorDyn::subview()` (`edgefirst-tensor`): zero-copy
  sub-region views that share the parent's allocation (`Mem` heap `Arc` or `Dma`
  fd) and map at a byte offset, for assembling a batch into one buffer. N views
  into one parent share the buffer (no copy) and write independently;
  disjointness of simultaneously-mapped mutable windows is the caller's contract.
  Wired through the Python bindings (`Tensor.subview()`, the `plane_offset`
  property, `set_plane_offset()` — a view exposes the full NumPy buffer-protocol
  surface via `map()`/`from_numpy()` like any tensor) and the C API
  (`hal_tensor_subview`, `hal_tensor_plane_offset`, `hal_tensor_set_plane_offset`).
- V4L2 hardware JPEG decode backend (`edgefirst-codec`, Linux, default-on
  `v4l2` feature). Capability-based probe drives any device exposing a JPEG
  decoder through the standard V4L2 mem2mem API (lead target i.MX `mxc-jpeg`),
  emitting the codec's native `NV12`/`GREY`. It is tried **before** the
  software decoder and falls back transparently when no device is present, the
  capture format is unsupported, or a per-image hardware failure occurs; a
  circuit breaker demotes a repeatedly-failing device to CPU. A persistent
  streaming session is kept across frames, and DMA-backed tensors with
  MCU(16)-aligned dimensions get a zero-copy path (the hardware decodes straight
  into the tensor's dmabuf). Env vars: `EDGEFIRST_DISABLE_V4L2=1` to force CPU,
  `EDGEFIRST_CODEC_V4L2_DEVICE=<node>` to probe a specific device.
- `Tensor::configure_image()` / `Tensor::image_with_capacity()` /
  `capacity_bytes()` / `set_logical_shape()` and `TensorDyn::configure_image()`
  (`edgefirst-tensor`): capacity-aware reconfiguration of an existing allocation
  to a decoded image's dimensions and pixel format, erroring with
  `InsufficientCapacity` when the image exceeds the allocation.
- `PixelFormat::Nv24` (semi-planar YUV 4:4:4) is now exposed in the **C** and
  **Python** bindings (`HAL_PIXEL_FORMAT_NV24` / `PixelFormat.Nv24`), reaching
  parity with the core enum. Previously a 4:4:4 JPEG decoded to the core's
  `Nv24` but the Python binding raised `unsupported pixel format: Nv24` on
  peek/decode and the C binding silently mis-reported it as `RGB`. Clients can
  now peek, decode, and convert 4:4:4 (and 4:2:2 `Nv16`) sources through both
  APIs. A C-API round-trip test now asserts every core `PixelFormat` has a
  distinct `HalPixelFormat` mapping (no silent fallback).
- EXIF orientation reporting in `ImageInfo` (`rotation_degrees`,
  `flip_horizontal`) for both JPEG and PNG, plus `peek_info()` to read it
  without decoding pixels.
- C decode API EXIF out-params: `hal_tensor_decode_image()` and
  `hal_tensor_decode_image_file()` gained `out_rotation_degrees` /
  `out_flip_horizontal` (both accept `NULL`).
- **Zero-copy CUDA tensor mapping** — `ImageProcessor::convert()` float PBO
  output is now directly consumable by CUDA/TensorRT consumers with no host
  round-trip on CUDA-capable devices (Jetson Orin-series, desktop CUDA GPUs).
  The PBO produced by the GL render path is registered with CUDA via
  `cudaGraphicsGLRegisterBuffer` on the GL worker thread; the resulting device
  pointer is valid from any thread via the per-device CUDA primary context.
  A DMA-BUF import path (`cudaImportExternalMemory(OpaqueFd)`) is also
  provided for externally allocated DMA-BUF-backed tensors.
  CUDA support is loaded at runtime via `dlopen("libcudart.so")` — no
  link-time dependency, no compile-time feature gate.
  - **Rust**: `Tensor::cuda_map() -> Option<CudaMap>`, `TensorDyn::cuda_map()`,
    `is_cuda_available() -> bool`, `Tensor::cuda()`. `CudaMap` is a scoped
    RAII guard exposing `device_ptr()` / `len()`; dropping it releases the
    PBO for the next `convert()`.
  - **C API**: `hal_is_cuda_available()`, `hal_tensor_cuda_map()`,
    `hal_tensor_cuda_device_ptr()` (size via `out_size` out-param),
    `hal_tensor_cuda_unmap()`.
  - **Python**: `edgefirst_hal.is_cuda_available()`,
    `Tensor.cuda_map() -> CudaMap | None` — a `with`-block context manager
    exposing `.device_ptr` and `.size`.
  Validated on Jetson Orin-nano (O5/O6/O8) on L4T R36.4 / CUDA 12.6 / TRT 10.3
  (numeric max\_err 0.00024 vs CPU reference) and desktop RTX 3090.
- **imx8mp coverage flush-guard** (`edgefirst_tensor::covguard`) — a
  `SIGABRT` handler compiled in under `-Cinstrument-coverage` that flushes
  the LLVM profile to disk before re-raising the signal. Prevents coverage
  loss on the i.MX 8M Plus CI lane where the Vivante EGL driver calls
  `abort()` during process shutdown.
- **F16 zero-copy preprocessing on macOS via ANGLE + RGBA16F-packed
  IOSurface.** RGBA8 input → PlanarRgb F16 output runs end-to-end on
  the GPU and writes directly into an IOSurface that ORT consumes as
  `&[f16]` over the locked base address — no CPU normalize pass, no
  HWC→CHW transpose, no copy. Validated end-to-end through the
  EdgeFirst profiler against 18 Studio sessions across YOLOv5 / YOLOv8
  / YOLO11 / YOLO26 (n / s / m sizes) on CoreML ANE and Metal GPU
  execution providers on Apple Silicon (M-series). AP50 delta vs the
  ONNX-CPU FP32 baseline is in the ±0.07 noise floor on every
  configuration; inference latency ranges from 2.9 ms (YOLOv5n on
  Metal GPU, ~340 FPS) to 19.5 ms (YOLO11m on ANE) without any CPU
  preprocessing overhead. Drives the four pinned-compute-unit
  `--provider coreml-{cpu,gpu,ane}` modes that ship in the matching
  EdgeFirst profiler release.
- `ImageProcessor::supported_render_dtypes()` introspection returning
  a `RenderDtypeSupport { f32, f16 }` struct that reports the GPU's
  `GL_EXT_color_buffer_float` and `GL_EXT_color_buffer_half_float`
  probe results. Consumers query this once at startup to decide
  whether to engage the IOSurface fast path or fall back to CPU
  staging. The `f16` flag is the operative one for the IOSurface
  render path; see the per-field doc on `RenderDtypeSupport` for the
  ANGLE-specific caveat on `f32`.
- `Tensor::<half::f16>::image(W, H, PixelFormat::PlanarRgb,
  Some(TensorMemory::Dma))` allocates an RGBA16F-packed IOSurface
  sized `(W/4, 3*H)` — four contiguous f16 planar elements packed
  per RGBA16F pixel. The contiguous byte layout matches the tensor's
  `[3, H, W]` shape exactly so ORT (or any NCHW consumer) sees the
  data with no transpose. `W % 4 == 0` is required and validated at
  allocation; non-multiples produce `InvalidShape` with the
  alignment requirement spelled out.
- `MacosGlProcessor::convert_rgba8_to_planar_float` — a one-pass
  fragment shader that resizes, normalizes, and transposes RGBA8
  IOSurface input to a PlanarRgb F16 IOSurface destination with
  full letterbox crop support (`src_rect_uv`, `dst_rect_px`,
  `pad_color` uniforms). Wired into the standard
  `ImageProcessor::convert` dispatch on
  `(Rgba, PlanarRgb, F16)`.
- `image_iosurface_layout(format, dtype)` re-export for the GL
  backend — single source of truth for the
  `(PixelFormat, DType) → (FourCC, bytes_per_element)` mapping.
- Re-export of `half::f16` from the tensor crate root so downstream
  crates can write `Tensor::<edgefirst_tensor::f16>::image(...)`
  without taking a direct `half` dependency.
- Linux GPU float preprocessing: F16 NCHW (`PlanarRgb`) and F32 NHWC
  (`Rgb`) via PBO readback, plus F16 zero-copy DMA-BUF render on
  V3D/Mali GPUs. Enables low-copy `convert()` output for HailoRT (F32)
  and TensorRT (F16) consumers.
- `gpu-probe`: F16/F32 float render-target capability probe (texture-FBO
  renderability, F16 dma-buf render, PBO readback timing).
- `gpu-probe`: `probe_nv_dmabuf` module — imports a DMA-BUF as a combined R8
  EGLImage (NV12/NV16/NV24), runs the Path B `texelFetch` shader, and reports
  whether semi-planar YUV DMA-BUF import succeeds on the platform.
- `ImageProcessor::supported_render_dtypes()` now reports real GPU
  float-render capability on Linux (previously always reported none).
- 4-axis `Colorimetry` (`space`/`transfer`/`encoding`/`range`, each `Option`)
  carried on `Tensor`/`TensorDyn` (`edgefirst-tensor`). Set automatically by the
  codec (JPEG → JFIF/BT.601-full, PNG Rgb/Rgba → sRGB+full, PNG Grey → full) and
  by `ImageProcessor::import_image` (caller supplies colorimetry from the
  producer's signalling). C surface: `hal_colorimetry` struct,
  `hal_tensor_set_colorimetry` / `hal_tensor_colorimetry`,
  `hal_colorimetry_from_v4l2`, `hal_import_image` colorimetry param. Python
  surface: `Colorimetry` class + enums, `tensor.colorimetry` property,
  `import_image(colorimetry=...)`.
- **macOS NV12/NV16/NV24 GPU conversion.** `MacosGlProcessor` now converts
  NV12/NV16/NV24 → RGBA and NV12/NV16/NV24 → PlanarRgb F16 entirely on the
  GPU via the R8 IOSurface `texelFetch` shader. Previously semi-planar sources
  fell back to the CPU YUV→RGB path on macOS. The two-pass NV* → PlanarRgb F16
  path runs under a single GL session
  (`image.convert.gl.macos.nv_to_planar` span).
- **Python `TensorMap` strided buffer protocol.** `TensorMap.__getbuffer__`
  now exposes the physical `effective_row_stride()` as the NumPy buffer's outer
  stride for image tensors with row padding (DMA / GPU tensors). Previously the
  tight logical stride was always reported, causing `np.asarray(memoryview(m))`
  to produce a sheared array for padded NV12/NV16/NV24 or RGBA tensors.

### Changed

- **Single source for the YUV↔RGB matrix/range constants** (`edgefirst-tensor`,
  `edgefirst-image`). The BT.601/709/2020 luma weights `(kr, kb)` and the
  full/limited-range luma/chroma swings are now defined once, as
  `ColorEncoding::luma_weights()` (→ `MatrixWeights`) and
  `ColorRange::scaling()` (→ `RangeScaling`) on the core colorimetry types. The
  in-shader GL coefficients (`yuv_to_rgb_coeffs`) and the hand-rolled
  fixed-point CPU YUYV encoders both derive from this source instead of carrying
  private duplicate tables, so a coefficient change is made in exactly one
  place. Output is byte-identical (verified by the existing encode goldens and a
  new absolute-value coefficient test).
- **Documented colorimetry transfer-function limitation.** The `ColorTransfer`
  (gamma / TRC) axis is stored, propagated, and round-tripped but is **not
  applied** by any conversion backend; all YUV↔RGB paths operate on the matrix
  and range only and assume the platform-native transfer curve. HDR curves
  (PQ / HLG) are therefore not tone-mapped. This is now called out on the type.
- `convert()` now honors each tensor's `Colorimetry`, resolving unknown axes at
  use-time via an SD/HD height heuristic (height ≥ 720 → BT.709, else BT.601;
  limited range when unknown) without mutating the tensor. CPU and OpenGL backends
  are fully colorimetry-correct (matrix + range). G2D applies the matrix
  (`set_bt601/709_colorspace`) but cannot express full-range or BT.2020 via
  `g2d-sys`; G2D declines those combinations and the dispatch falls back to
  GL/CPU.
- **Breaking:** `edgefirst-codec` now decodes to the image's native format only
  — JPEG → `Nv12` (4:2:0 chroma subsampling) / `Nv16` (4:2:2) / `Nv24` (4:4:4)
  or `Grey` (greyscale); PNG → `Rgb`/`Rgba`/`Grey`. The decoder configures the
  destination tensor's dimensions and format to match and never colour-converts,
  resizes, or rotates. Use `ImageProcessor::convert()` for `Rgb`/`Rgba`/`Bgra`,
  resize, and applying the reported EXIF orientation.
  JPEG decodes to `u8` only (non-`u8` destinations return `UnsupportedDtype`).
- **Breaking:** EXIF orientation is reported, never applied by the codec; the
  decoded pixels and dimensions are the source's native, unrotated values.
- **Breaking:** the C and Python decode APIs drop the output-`format`
  parameter; the decode configures the tensor to the native format, which the
  caller inspects (e.g. `hal_tensor_pixel_format()`) and converts as needed.
- **(Potentially breaking)** `Tensor::<T>::image()` is now bounded
  `T: 'static` so the runtime `DType` can be derived from `T` via the
  new internal `dtype_of::<T>()` helper (`TypeId`-based, hence the
  `'static` requirement). Every numeric type HAL ships, and every
  primitive numeric type, already satisfies `'static`, so in-tree and
  typical external callers are unaffected. The bound does, however,
  exclude any user-defined element type that borrows non-`'static`
  references even while satisfying `Num + Clone + Debug + Send +
  Sync` — such a type would no longer compile through `image()`. We
  judge this combination to be vanishingly rare for pixel element
  types, but it is a real API constraint, not a no-op.
- `Tensor::<T>::image(.., Some(TensorMemory::Dma))` now **rejects**
  combinations whose natural row pitch isn't 64-byte aligned with a
  clear `InvalidArgument` error that names the alignment requirement
  and suggests the next-aligned width. The previous behaviour
  silently fell through to a generic `L008` byte-bag IOSurface that
  any GL importer later rejected with the opaque `EGL_BAD_ATTRIBUTE`
  — a debugging poison-trap that surfaced during the profiler
  IOSurface bring-up. `memory=None` (auto-select) keeps the
  graceful SHM/Mem fallback; only **explicit** `Dma` requests fail
  loudly.
- `IoSurfaceTensor::new_image` takes a `dtype: DType` argument and
  routes planar F16 destinations through the RGBA16F-packed geometry
  (`width/4 × channels*height`). Internal change — public callers go
  through `Tensor::<T>::image()` which threads the dtype
  automatically.
- `PbufferCacheKey` in the macOS GL backend now includes a
  `dtype_disc` field so a tensor handed to the processor as
  `Tensor<u8>` and the same surface seen later as
  `Tensor<half::f16>` no longer alias under the previous
  `(id, format)` key.
- `eglCreatePbufferFromClientBuffer` failure now reports the full
  EGL attribute set (FourCC, internal format, GL type,
  bytes-per-element, surface dimensions) alongside the raw EGL
  error code. The opaque `EGL_BAD_ATTRIBUTE` message that surfaced
  during bring-up gave no path to the actual root cause; the
  enriched message names every input that the kernel rejected.
- **Tracing span rename** — the Vivante two-pass span pair
  `image.convert.gl.nv12_to_planar.{pass1_rgba,pass2_deinterleave}` is renamed
  to `image.convert.gl.nv_to_planar.{pass1_rgba,pass2_deinterleave}` to reflect
  that the two-pass workaround now covers NV12/NV16/NV24, not just NV12. Saved
  Perfetto queries targeting the old names will need updating.

### Removed

- **Breaking:** `DecodeOptions` and the `opts` parameter on all
  decode/load/peek APIs.
- **Breaking:** the `edgefirst-image` `load_image()` free function. Decode via
  the re-exported `codec::{ImageDecoder, ImageLoad}` into a pre-allocated tensor
  instead.
- The in-codec YCbCr→RGB/RGBA/BGRA colour kernels, chroma-upsample kernels, and
  SIMD type-conversion path (these moved to `ImageProcessor::convert()`); the
  JPEG CPU path now writes native `NV12`/`NV16`/`NV24`/`GREY` directly.
- **Breaking (internal types):** the per-memory backing types
  `MemTensor`/`MemMap`, `DmaTensor`/`DmaMap`, `ShmTensor`/`ShmMap`, and
  `IoSurfaceTensor`/`IoSurfaceMap` are no longer re-exported from
  `edgefirst-tensor` — they are implementation details. Allocate `Tensor<T>` /
  `TensorDyn` and `map()` them instead. The PBO extension point
  (`PboTensor`/`PboMap`/`PboMapping`/`PboOps`) and `image_iosurface_layout`
  remain public.

### Fixed

- **V4L2: JPEGs with embedded-thumbnail metadata no longer wedge `mxc-jpeg`**
  (`edgefirst-codec`). The i.MX bitstream parser does not length-skip APPn
  segments, so an APP13 (Photoshop IRB) carrying a nested thumbnail JPEG
  stalled the hardware until the 2 s decode timeout (deterministic on COCO val
  `000000122046.jpg`). Metadata segments — except JFIF APP0 and Adobe APP14 —
  are now stripped while staging the bitstream into the OUTPUT buffer.

- **macOS YUYV→RGBA now honors per-tensor colorimetry** (`edgefirst-image`).
  The macOS ANGLE YUYV shader baked BT.601 full-range coefficients into GLSL;
  it now derives its YUV→RGB matrix + range from the source tensor's resolved
  colorimetry via six uniforms, like the NV path. An untagged 720p source
  resolves to BT.709 limited, raising camera-fixture similarity from 0.9733 to
  0.9973 on ANGLE. The relaxed `feature/colorimetry WIP` test tolerances are
  re-justified against measured data (CPU/GL/macOS tightened; G2D reclassified
  as a structural fixed-point bound, not a WIP delta), and new BT.2020 (CPU) and
  BT.601-vs-BT.709 (Python) pixel-difference tests guard the per-source matrix.

### Changed

- **GL backend portability refactor** (`edgefirst-image`, internal). Decoupled
  the shader/format code from `gbm` by taking `DrmFourcc` from the portable
  `drm-fourcc` crate — `gbm` is now referenced only by the GBM EGL-display
  backend (`gl/context.rs`) and two error variants. Unified the Linux and macOS
  NV12/16/24→RGBA shaders onto one divide-free body in the new portable
  `gl::shaders_common` (faster-or-neutral on Apple ANGLE; the divide-free form
  avoids per-fragment integer modulo of a variable divisor), guarded by a
  byte-identity golden test that runs on every platform. Added portable
  `gl::core` (shared `float_crop_uniforms`) and `gl::fourcc`, and removed the
  macOS-only `IoSurfaceF16Nchw` variant from the shared float-path enum. No
  user-visible behavior change beyond the YUYV fix above.

- **Heap (`Mem`) tensor allocation no longer eagerly memsets every element**
  (`edgefirst-tensor`). `MemBacking::zeroed` had regressed to
  `(0..n).map(UnsafeCell::new).collect()`, which allocates uninitialised then
  writes every element — committing every page up front. It now allocates via
  `alloc_zeroed` (calloc) when `T::zero()` is the all-zeros bit pattern (every
  primitive numeric type), so the kernel returns lazily-zeroed pages. This backs
  every `Tensor::image(.., Mem)` and the per-call image intermediates the CPU
  converter allocates, restoring the pre-regression allocation latency.
- **CPU packed-YUV → GREY ignored the source row stride** (`edgefirst-image`).
  `Yuyv`/`Vyuy` → `Grey` chunked the flat mapped buffer with a fixed 16-byte
  stride, so a padded source (DMA-BUF, V4L2 hardware decode, or pitch-aligned
  `create_image`) read luma from per-row padding and produced misaligned/garbage
  output. Both paths now iterate rows honouring `effective_row_stride`, matching
  the other `*_to_grey` converters.
- **CPU planar → packed read-back ignored stride** (`edgefirst-image`).
  `PlanarRgb`/`PlanarRgba` → `Rgb`/`Rgba` derived plane boundaries from
  `mapped_len / channels` and walked each plane as if tightly packed, corrupting
  output for strided/padded planar sources. The four converters now share a
  single stride-aware `planar_to_packed` helper that derives plane offsets from
  `height × row_stride` and reads each row at its true offset.
- **CPU NV24 → GREY truncated multiplane sources** (`edgefirst-image`).
  `convert_nv24_to_grey` computed the luma height as `shape[0] / 3` before the
  multiplane check, so a true-multiplane NV24 tensor (separate luma/chroma
  allocations) produced only the top third of its rows. The height is now
  resolved inside the multiplane branch, matching `convert_nv24_to_rgb`.
- **CUDA external-memory handle drop called `cudaFree` on a mapped pointer**
  (`edgefirst-tensor`). `CudaHandle::Drop` for an imported DMA-BUF (`ExternalMem`)
  called `cudaFree` on the pointer returned by `cudaExternalMemoryGetMappedBuffer`
  before destroying the external-memory object. CUDA frees that buffer together
  with the external-memory object, so the extra `cudaFree` is disallowed and can
  corrupt driver bookkeeping / double-free. Drop now only calls
  `cudaDestroyExternalMemory`.
- **GPU EGLImage source cache stale-geometry read (recycled DMA-BUF pools).**
  The OpenGL source EGLImage cache keyed only on buffer identity + plane offset.
  A pooled DMA-BUF tensor reconfigured between converts (`configure_image`, e.g.
  a decode pool reused at a new width/format/stride per frame) reused the cached
  EGLImage imported at the *previous* geometry, so the GPU sampled at the wrong
  pitch — producing wrong/garbage output (deterministic single-threaded,
  nondeterministic under parallel decode). The cache key now carries
  `width`/`height`/`row_stride`/`format`, so a content/geometry change is a
  distinct entry. Constant-geometry reuse still hits the cache (no perf change).
- **GPU NV16/NV24 zero-copy convert on Linux/embedded GLES.** NV16 (4:2:2) and
  NV24 (4:4:4) DMA-BUF sources previously had no GPU path and silently fell back
  to the CPU YUV→RGB converter (the dominant cost in preprocessing on embedded
  GPUs). They now convert on the GPU zero-copy via "Path B": the combined
  semi-planar buffer is imported as a single-plane R8 EGLImage and a hand-written
  `texelFetch` shader (core GL ES 3.0, no `GL_OES_EGL_image_external` extension)
  does the YUV→RGB. The shader uses direct 2D addressing (no per-pixel integer
  divide/modulo), which is essential on Vivante (≈3.3× over the naive form).
  NV12 continues to use the proven Path A (`samplerExternalOES` hardware-YUV),
  because `samplerExternalOES` does not correctly sample 4:2:2/4:4:4 on the
  embedded drivers. The output dtype is target-appropriate (u8/i8 RGB for the
  quantized NPU targets, F16 PlanarRgb on Tegra/macOS). Validated on Vivante
  GC7000UL, Mali-G310, V3D, and Tegra. BT.601 full-range (interim colorimetry).
- PBO-backed tensors now implement a capacity-based `set_logical_shape` (and
  `capacity_bytes`), matching `Mem`/`Shm`/`DMA`/`IOSurface`. Previously PBO fell
  back to the strict-`reshape` default (exact element match), so an oversized
  reusable pool could not be `configure_image`d to a smaller image. This broke
  the native-chroma decode pool on PBO-backed hosts (e.g. Linux PCs / Jetson
  without `/dev/dma_heap`), where reconfiguring the `3·H` GREY pool to a smaller
  `NV12`/`NV16`/`NV24` shape failed with a `ShapeMismatch`. DMA-backed pools were
  unaffected.
- GPU and heap sub-region views (`view()`/`batch()`) render to their windows
  without aliasing. The OpenGL EGLImage import keys on the view's **parent**
  identity+geometry, so all sibling views of one DMA-BUF share a single import and
  each renders to its sub-rect via `glViewport`/`glScissor` — the tile offset is
  render state, not part of the cache key — which is what enables one-import
  batched render-to-DMA-BUF (see `convert_deferred`/`flush` above). Heap (`Mem`)
  tensors map correctly at non-zero offsets, and the shared backing uses
  interior-mutable cells so disjoint sub-views carry correct write provenance.
- Odd-height NV12 from the V4L2 hardware JPEG decoder no longer drops its last
  chroma row (the MMAP copy used `final_h / 2`; now `ceil(final_h / 2)`).
- **NV12 odd-dimension support.** The JPEG decoder previously rejected any
  colour JPEG whose width *or* height was odd with `NV12 requires even
  dimensions`, breaking common photo/COCO images (e.g. 640×483, 375×500).
  `PixelFormat::Nv12` now handles odd dimensions:
  - **Odd height** is represented directly: the combined-plane shape is
    `H + ceil(H/2)` rows (luma + chroma), which equals the classic `3H/2` for
    even heights and stays exact for odd ones (e.g. 483 → 725 rows).
  - **Odd width** rounds the buffer width up to even (a chroma-interleaving
    requirement — one `(U, V)` pair per two luma columns has no whole-byte form
    at odd width). The true odd width is reported by the decoder in `ImageInfo`
    and trimmed by a `convert()` crop; `width()` reports the even buffer width.
    This matches the standard even-width NV12 representation used by V4L2,
    cameras, and most codecs, and avoids a row stride (strided NV12 buffers
    cannot be CPU-mapped on non-Linux platforms).

  The contiguous NV12 `convert()` CPU kernels (`Nv12`→`Rgb`/`Rgba`/`Grey`) are
  now stride- and logical-height-aware, so externally-allocated padded NV12
  buffers convert correctly too.

- **Strided CPU mapping for Mem/Shm/IOSurface tensors.** `Tensor::map()`
  previously rejected any strided tensor on non-Linux with "CPU mapping of
  strided tensors is not supported on this platform (DMA backing is
  Linux-only)". That was an unimplemented path, not a platform limit. `map()`
  now exposes the full row-padded buffer for row-stride iteration across every
  HAL-owned storage, mirroring the self-allocated Linux DMA path:
  - **Mem / Shm** (all platforms): validates `row_stride × rows` against the
    allocation capacity.
  - **IOSurface** (macOS): plumbs `IOSurfaceGetBytesPerRow`; the
    `IOSurfaceLock` already yields the surface base address, so the strided
    view is genuinely zero-copy (no staging buffer). Image-formatted surfaces
    carry their 64-aligned `bytes_per_row` as the tensor's row stride, and
    `Tensor::image()` now allocates a **padded zero-copy IOSurface** for packed
    formats whose width isn't 64-aligned (RGBA/BGRA/YUYV and packed F16) instead
    of falling back to SHM — GL imports via the surface pitch and the CPU maps
    via the strided path. Formats without an IOSurface FourCC (Rgb/Grey u8) and
    the flat-consumed planar-F16 packing still require an aligned pitch.

  Imported DMA-BUFs (external allocator layout) and PBO storages remain
  GPU-path only.

- **Linux reports real capability.** `ImageProcessor::supported_render_dtypes()`
  returns the GPU's actual `GL_EXT_color_buffer_half_float` /
  `GL_EXT_color_buffer_float` probe results on Linux. Vivante GC7000UL
  always returns `{ f16: false, f32: false }` (float readback 170–320 ms —
  disabled); V3D/Mali and Tegra report the extensions they expose.
  `convert()` to F16/F32 destinations never errors — it falls back to CPU
  when the GPU float path is unavailable.
- **No F32 IOSurface path (macOS).** ANGLE's
  `EGL_ANGLE_iosurface_client_buffer` extension specifies exactly
  one float `(type, internal_format)` pair —
  `(GL_HALF_FLOAT, GL_RGBA)` = RGBA16F. Every `(GL_FLOAT, *)`
  combination is rejected with `EGL_BAD_ATTRIBUTE` even when
  `GL_EXT_color_buffer_float` is present. `image_iosurface_layout`
  therefore returns `None` for any F32 combination; callers that
  request F32 IOSurface allocation get an explicit `NotImplemented`
  error and fall back to SHM/Mem + CPU staging.
- **F32 DMA-BUF not supported (all platforms).** No 32-bit-float DRM
  fourcc exists; `create_image(memory: Some(Dma), dtype: F32)` returns
  `NotSupported`. F32 uses PBO transfer instead.
- **PlanarRgb only on the convert dispatch.** The RGBA16F-packed
  fragment shader and viewport are sized for 3 channel planes.
  `MacosGlProcessor::supports()` and the convert dispatch currently
  reject `(Rgba, PlanarRgba)` to avoid mis-rendering the alpha plane.
- **CPU float fallback is always present.** When the GL and G2D backends
  cannot handle a float destination (unsupported combo, rotation set, or
  unavailable hardware), `ImageProcessor::convert()` falls back to CPU
  for any pixel-format pair. Resize is performed at u8 precision before
  widening to F16/F32; normalization is `[0, 1]`.

### Fixed

- **ModelPack segmentation decoder rejected valid models** (DE-2651, DE-2628):
  `Decoder` construction failed with `InvalidConfig("Segmentation dshape missing
  required dimension NumClasses")` — and, when the segmentation output was
  consequently dropped, `InvalidConfig("Invalid ModelPack model outputs")` — for
  ModelPack models whose segmentation `dshape` tagged the canonical NHWC channel
  axis with something other than `num_classes` (e.g. `num_protos`, the cloud
  validator's shape-inference fallback for `.keras` exports). `verify_modelpack_seg`
  now requires only the spatial dshape axes by name and infers the class count
  from the canonical NHWC channel axis (`shape[3]`) when `num_classes` is absent,
  preferring an explicit `num_classes` tag when present. Structural dshape
  integrity remains enforced by `validate_output_layout`. (Back-ported to the
  0.22.x line as 0.22.2.)
- **Unknown dshape axis names no longer fail metadata parsing** (DE-2651):
  `DimName` gained a catch-all `Unknown` variant, so an unrecognised axis name
  (e.g. older ModelPack exports tagging the input dshape channel axis
  `channels`) is preserved as `Unknown` — keeping the dshape length aligned with
  the shape and sorting the axis to the canonical tail — instead of rejecting the
  entire decoder build.

## [0.24.2] - 2026-05-28

### Fixed

- GL overlay channel-permuted background on i.MX targets:
  `GLProcessorST::draw_camera_texture_to_rgb_planar` and its `_2d`
  intermediate variant set `GL_TEXTURE_SWIZZLE_R` to `GL_RED`,
  `GL_GREEN`, `GL_BLUE` across three loop iterations so the single-
  channel R8 output of each pass picks up the matching source
  component, but returned with the swizzle parked at `GL_BLUE` (the
  last iteration's value). On NXP Vivante GC7000 and Mali Valhall
  drivers — both of which honour `GL_TEXTURE_SWIZZLE_*` on
  `GL_TEXTURE_EXTERNAL_OES` despite the GLES spec leaving it
  undefined — the residual swizzle persisted into the next sampler
  read of the long-lived `camera_eglimage_texture`, including the
  bg blit inside `draw_decoded_masks`. Every overlay pixel landed
  with `canvas.R := src.B` while G/B/A stayed correct, producing
  the characteristic green-on-warm / magenta-on-cool tint on the
  overlay background in the saved JPEG. Mesa ignored the swizzle on
  external textures, so the CI golden image
  (`testdata/output_render_gl.jpg`) was clean and the unit test
  using `giraffe.jpg` did not surface the bug.

  Both paths now restore `GL_TEXTURE_SWIZZLE_R` to `GL_RED` before
  returning, keeping the texture's per-channel swizzle at the GLES
  identity. `TEXTURE_SWIZZLE_G/B/A` are not touched inside either
  function, so a single `glTexParameteri` call per path is enough
  to restore identity. Verified end-to-end through the
  `ara2-rs` `yolov8` example on `imx8mp-frdm` (Vivante GC7000Lite)
  and `imx95-frdm` (Mali Valhall): both produced the expected
  natural-colour overlay with clean green bounding boxes after the
  fix, and both reproduced the `canvas.R := src.B` corruption
  without it.

## [0.24.1] - 2026-05-26

### Fixed

- GStreamer bridge double-normalized detections: `Decoder::normalized_boxes()`
  returned `Some(false)` ("pixel-space") even for paths that had already
  divided bbox channels by `(W, H)` via `yolo::maybe_normalize_boxes_in_place`,
  causing callers reading `Some(false)` to divide a second time and
  collapse detections to ~0.

  **Root cause.** Before this release, `maybe_normalize_boxes_in_place`
  was absent from the tracker entry points (`decode_tracked`,
  `decode_tracked_proto`) and from the proto entry points for the
  seg/mask/proto model families. The helper was also missing entirely
  from `YoloSplitSegDet` and `YoloSegDet2Way` paths. The accessor had
  no way to report the actual post-decode coordinate space.

  **Implementation fix.** `maybe_normalize_boxes_in_place` is now called
  uniformly across every entry point for four paths:

  1. **Per-scale decoders** (schema-v2 per-scale layout, DFL/LTRB
     dist2bbox path): unchanged from 0.24.0; already normalized
     uniformly.
  2. **`ModelType::YoloSegDet`**: helper added to
     `process_tracked_yolo_segmentation!` and
     `process_tracked_yolo_segdet_float`.
  3. **`ModelType::YoloSplitSegDet`**: helper added to
     `process_tracked_yolo_segmentation_split!` and
     `process_tracked_yolo_segdet_split_float`; `_proto` variants
     updated; `impl_yolo_split_segdet_float_proto` signature extended
     to accept `normalized` and `input_dims`.
  4. **`ModelType::YoloSegDet2Way`**: helper added to
     `process_tracked_yolo_segmentation_2way!` and the inline
     tracked-2way float helpers; `_proto` variants updated.

  Total new call sites: 8 (7 in `postprocess.rs`, 1 in `yolo.rs`).

  **Accessor fix.** `Decoder::normalized_boxes()` now reports the
  post-decode coordinate space for all four paths. The
  `legacy_path_normalizes_uniformly()` predicate was extended to cover
  `YoloSegDet`, `YoloSplitSegDet`, and `YoloSegDet2Way` (previously
  only `YoloSegDet`). When the schema declares `normalized: false` and
  `input_dims` is a valid non-zero pair, the accessor returns
  `Some(true)` rather than the raw `Some(false)`.

  **Intentional raw-flag paths.** Detection-only (`YoloDet`,
  `YoloSplitDet`), end-to-end YOLO (`YoloEndToEnd*`), `ModelPack*`, and
  the schema-v2 merge program return the raw schema annotation unchanged.
  These paths have distinct coordinate conventions or lack the protobox
  safety-net coupling. Callers receiving `Some(false)` from these model
  types must call `input_dims()` and divide themselves.

  Callers must not re-normalize when the accessor reports `Some(true)`.

## [0.24.0] - 2026-05-25

### Added

- `hal_tensor_set_quantization(tensor, scale, zero_point)` — C API to
  attach per-tensor affine quantization metadata to an integer tensor.
  Required when wrapping framework buffers (e.g. NNStreamer `GstMemory`
  outputs) that carry no quant of their own before feeding them to a
  schema-driven `hal_decoder_decode_proto()` call. The C boundary
  actively validates inputs that the Rust `Quantization::per_tensor`
  constructor does not: `scale` must be finite and strictly positive
  (NaN, ±∞, zero, negative all reject with `EINVAL`); `zero_point`
  must fit the tensor's integer dtype range (e.g. `0..=255` for `u8`,
  `-128..=127` for `i8`); float tensors reject with `EINVAL`.

### Changed

- **C decoder errno mapping rewritten.** `hal_decoder_decode`,
  `hal_decoder_decode_proto`, and `hal_decoder_decode_tracked`
  previously collapsed every `DecoderError` variant to `EIO`, masking
  caller-side preconditions as "internal" faults. A new
  `errno_for_decoder_error()` mapper now splits the variants:
  - **`EINVAL` (caller-side input is malformed or incomplete)** —
    `InvalidShape`, `InvalidConfig`, `NoConfig`, `DtypeMismatch`,
    `QuantMissing`, `Yaml`, `Json`, `NotSupported`, `NDArrayShape`.
  - **`EIO` (internal/library fault the caller cannot fix)** —
    `Internal`, `KernelDispatchUnreachable`, `ForcedKernelUnavailable`.

  In particular, **missing per-tensor quantization on integer decoder
  inputs now surfaces as `EINVAL` instead of `EIO`**, so operators
  triaging GStreamer / NNStreamer pipelines can read the upstream
  wiring bug directly rather than as a hardware/library failure. The
  full `DecoderError` variant is still logged via `log::error!` for
  diagnostics.

- Doxygen `@par Errors` blocks on the three decoder entry points
  reworded to describe the new caller-vs-internal split.

## [0.23.2] - 2026-05-21

### Changed

- **Tracing span rename — observable surface, not API.** All
  `tracing::trace_span!` call sites across the workspace adopt a uniform
  `<crate>.<function>[.<operation>[.<sub-operation>]]` naming convention
  (snake_case, dotted). The `<function>` component is the public user-facing
  function the trace was rooted at (`decoder.decode`, `image.convert`,
  `codec.decode_jpeg`, …); inner spans nest under it. Shared internal
  building blocks reached from multiple top-level functions
  (`decoder.per_scale_run`, `decoder.nms_get_boxes`) omit the function prefix
  because a single source location can only carry one static name — the
  parent span in the recorded trace identifies the caller. New spans should
  follow the rough guideline of >500 µs on Cortex-A53 to justify the
  instrumentation.

  Span names appear as labels in Perfetto / Chrome traces; any saved query,
  dashboard, or filter that targets the old names will need updating. See
  the per-crate ARCHITECTURE.md "Tracing Spans" sections for the
  authoritative mapping.

  Notable replacements (full list in the per-crate docs):

  | Old name (collision) | New name |
  |---|---|
  | `decode` (4 YOLO entry points + 2 segdet inner kernels) | `decoder.decode.yolo_{quant,float}_{flat,split}` and `decoder.nms_get_boxes` |
  | `score_filter` / `top_k` / `nms` / `box_dequant` | `decoder.nms_get_boxes.{score_filter,top_k,suppress,dequant_boxes}` |
  | `process_masks` | `decoder.decode.process_masks` |
  | `extract_proto` | `decoder.decode_proto.extract_proto_data` |
  | `per_scale_decode` and `box` / `score` / `mc` | `decoder.per_scale_run` and `decoder.per_scale_run.level.{boxes,scores,mask_coefs}` |
  | `per_scale_bridge::*` | `decoder.{decode.per_scale_to_masks,decode_proto.per_scale_to_proto_data,per_scale_run.widen_f32}` |
  | `Decoder::decode` / `Decoder::decode_proto` | `decoder.{decode,decode_proto}` |
  | `image_convert` / `draw_masks` | `image.{convert,draw_decoded_masks}` |
  | `g2d_convert` / `gl_convert` | `image.convert.{g2d,gl}` |
  | `gl_pass1_to_rgba` / `gl_pass2_pack_rgb` / `gl_pass2_to_planar` | `image.convert.gl.{pack_rgb.pass1_rgba,pack_rgb.pass2_pack,nv12_to_planar.pass1_rgba,nv12_to_planar.pass2_deinterleave}` |
  | `cpu_format_convert` / `cpu_resize` | `image.convert.cpu.{format_convert,resize_flip_rotate}` |
  | `materialize_masks` / `mask_i8_fastpath` / `mask_i16_i8_fastpath` | `image.materialize_masks` and `image.materialize_masks.{kernel_i8,kernel_i16xi8,kernel_i8_scaled,kernel_i16xi8_scaled}` |
  | `tracker_update` / `predict` / `match_*` | `tracker.update` and `tracker.update.{predict,match_high_conf,match_low_conf}` |
  | `tensor_alloc` / `tensor_map` | `tensor.{alloc,map}` |
  | `py_*` | `python.*` |

### Added

- Tracing spans in the `edgefirst-codec` crate (previously had none):
  `codec.decode_jpeg`, `codec.decode_png`, plus inner spans
  `codec.decode_jpeg.{parse_markers,mcu_loop,apply_exif,type_convert}` and
  `codec.decode_png.zune_decode`. The JPEG marker-parse, MCU decode loop,
  and EXIF rotation phases are now individually visible in Perfetto.
- `tracing` workspace dependency added to `edgefirst-codec`.
- "Tracing Spans" sections in `crates/decoder/ARCHITECTURE.md`,
  `crates/image/ARCHITECTURE.md`, and `crates/codec/ARCHITECTURE.md` —
  authoritative span tree + per-span operation description, satisfying the
  cross-reference promise made in `README.md#performance-tracing`.

## [0.23.1] - 2026-05-19

### Fixed

- GL texture sampling now applies a half-texel inset to `src_rect` texture
  coordinates, preventing `GL_LINEAR` bilinear interpolation from bleeding
  into padding pixels when decoding into reused strided buffers via
  `load_image`. All non-DMA upload paths (`draw_src_texture`,
  `draw_src_texture_from_pbo`) and the EGLImage path now also set
  `GL_CLAMP_TO_EDGE` as secondary defense.

## [0.23.0] - 2026-05-17

### Added

- `edgefirst_codec::peek_info()` — parse JPEG/PNG headers and return image
  dimensions/format without decoding pixels. Used to allocate a tensor sized
  to the image before calling `decode_into`/`load_image`. Returns
  post-rotation dimensions when `apply_exif=true` and the image carries a
  90°/270° orientation tag.
- Python `Tensor.peek_image_info(data)` and `Tensor.peek_image_info_file(path)`
  staticmethods, mirroring the Rust `peek_info` API.
- EXIF orientation handling for PNG images on the u8 decode path (uses the
  `eXIf` chunk surfaced by zune-png and the shared `apply_exif_u8` helper).
- `Error::Codec(CodecError)` variant on `edgefirst_image::Error` with a
  `From<CodecError>` conversion.
- `CodecError::Unsupported(UnsupportedFeature)` variant + `UnsupportedFeature`
  enum exposing typed variants for features the codec doesn't implement:
  `ProgressiveJpeg`, `ArithmeticCodedJpeg`, `LosslessJpeg`,
  `HierarchicalJpeg`, `JpegPrecision { bits }`, `JpegComponentCount`,
  `JpegChromaSubsampling`. Callers can pattern-match instead of grepping
  error strings.
- SOF marker recognition for all JPEG variants (SOF1/3/5/6/7/9/10/11/13/14/15)
  — previously these fell through to the "unknown marker" path and were
  silently skipped; now they return a precise `Unsupported(...)` error.
- JPEG chroma subsampling sanity check: rejects images whose chroma rate
  exceeds luma (a division-by-zero hazard in the chroma upsampler / NV12
  writer with adversarial input).
- NV12 output with odd width or odd height is now rejected up front
  (`CodecError::InvalidData(...)`). The NV12 chroma plane is defined
  only for even dimensions; the prior path silently truncated the
  right/bottom chroma column.
- Scalar↔SIMD parity tests for all 8 JPEG SIMD kernels (IDCT
  NEON/SSE2/SSE4.1, color NEON/SSE2/SSSE3, chroma upsample NEON/SSE2).
  26 inline tests with deterministic synthetic inputs and ±1 LSB
  tolerance, gated on the right `target_arch` + runtime feature
  detection. Catches arithmetic divergence that end-to-end image tests
  (which tolerate ±12 LSB) silently masked.
- Codec integration tests for typed `Unsupported(ProgressiveJpeg)`,
  odd-width NV12 rejection, and partial-MCU bottom-edge correctness.
- EXIF orientation fixtures (`testdata/zidane_exif_<N>.{jpg,png}` for
  N=1..=8) generated by `scripts/generate_exif_fixtures.py`. All 8 JPEG
  variants share identical scan data and all 8 PNG variants share
  identical IDAT data — only the EXIF/eXIf orientation tag differs.
- `crates/codec/tests/exif_orientations.rs`: 22 end-to-end tests
  covering both JPEG and PNG paths. Each orientation is checked against
  an in-test oracle (explicit per-pixel `(nx, ny)` mapping) with SAD=0
  required, plus `apply_false`-invariance and `peek_info` dim-swap
  assertions for the 90°/270° cases.
- `bench_exif_overhead` group in `codec_benchmark` measuring decode time
  for all 8 orientations × {apply_false, apply_true} × {JPEG, PNG} on
  every supported target. Results for PC + imx8mp-frdm + imx95-frdm +
  rpi5-hailo + orin-nano are tabulated in `BENCHMARKS.md` under
  "EXIF Orientation Overhead".
- GL PBO regression tests (`test_gl_convert_any_to_pbo_no_deadlock`,
  `test_gl_convert_pbo_to_pbo_no_deadlock`) — explicit liveness coverage
  for the deadlock fixed in commit c494fae.
- `codec` rows added to the root `ARCHITECTURE.md` and `TESTING.md`
  per-crate index tables.
- `crates/codec/README.md` "Decoder Limitations" section with a complete
  table of unsupported JPEG/PNG features, each mapped to the typed
  `CodecError::Unsupported(...)` variant that the rejection carries.
- `crates/codec/ARCHITECTURE.md` "Supported Source Features" section
  cross-referencing the limitations table and stating the codec's
  contract ("accept this strict subset; reject everything else with a
  precise typed error").

### Changed

- **EXIF rotation in the codec u8 fast path is now correct on rotated
  destination tensors.** Previously the decoder wrote native-dimension rows
  into a post-rotation-sized tensor with the post-rotation stride, causing
  silent row overlap. The decode now lands in a native-stride scratch buffer,
  is rotated in place, then stride-copied into the destination — including
  pitch-padded DMA tensors.
- `edgefirst_codec::exif::read_exif_orientation()` now strips the JPEG APP1
  `"Exif\0\0"` identifier before handing the TIFF block to kamadak-exif. The
  prior parse silently returned "no rotation" for every real-world JPEG.
- Color JPEGs can now decode to `PixelFormat::Grey` (Y-plane copy, chroma
  ignored). Previously the codec returned `UnsupportedFormat(Grey)` for
  multi-component input.
- EXIF helpers (`read_exif_orientation`, `apply_exif_u8`) moved from the
  JPEG submodule into a shared `crate::exif` module so JPEG and PNG paths
  share one implementation. `apply_exif_u8` parameter renamed `channels` →
  `bytes_per_pixel` to reflect that the routine is byte-width agnostic.

### Removed

- **Breaking change.** Removed the legacy `edgefirst_image::load_image`,
  `load_jpeg`, and `load_png` functions, along with the C
  `hal_tensor_load_image`, `hal_tensor_load_image_file`, `hal_tensor_load_jpeg`,
  `hal_tensor_load_png` symbols and the Python `Tensor.load`,
  `Tensor.load_from_bytes` staticmethods. Decode now goes through the
  `edgefirst_codec` API: `peek_info` → allocate tensor → `decode_into` /
  `Tensor.decode_image`. This makes per-frame allocations explicit and lets
  the decoder write directly into pitch-padded DMA tensors.
- `zune-jpeg` is no longer a runtime dependency of `edgefirst-image`; it is
  retained only as a `[dev-dependencies]` entry in `edgefirst-codec` for
  oracle parity tests/benchmarks.
- `zune-png` is no longer a direct dependency of `edgefirst-image`
  (`edgefirst-codec` continues to depend on it for PNG decoding).
- Removed unused `edgefirst-image` internal helpers `rotate_flip_to_dyn`,
  `copy_packed_to_padded_dma`, and `read_exif_orientation` (their only
  callers were the now-removed `load_image` family).
- Removed `Error::JpegDecoding(zune_jpeg::errors::DecodeErrors)` and
  `Error::PngDecoding(zune_png::error::PngDecodeErrors)` variants on
  `edgefirst_image::Error`; codec errors now surface via `Error::Codec`.
- `DecodeOptions::scale_denom` and `DecodeOptions::with_scale()`. The field
  was declared and documented but never read by any decode path — calling
  `with_scale(2)` silently produced a full-resolution decode. Removed
  rather than leave a public-API trap.

### Fixed

- Codec EXIF orientation parsing now actually fires on real-world JPEGs (it
  was a silent no-op for every JPEG whose APP1 segment carried the standard
  `"Exif\0\0"` prefix).
- NV12 odd-dimension safety: NV12 requires even width and height by definition
  (one Cb/Cr pair per 2×2 luma block). Previously `chroma_w = img_w / 2`
  silently truncated the right-most chroma column on odd-width images,
  producing a magenta/green edge artefact. The decoder now rejects odd
  NV12 dimensions up front with `InvalidData("NV12 requires even dimensions
  …")` rather than silently mis-decoding.
- AC Huffman coefficient size is bounded to the spec maximum of 10 bits.
  Crafted Huffman tables can produce symbols up to size 15, which previously
  fed `read_bits(15)` and `extend` past the spec range, then multiplied by
  the quant table to overflow i32 silently.
- JPEG with component count ∉ {1, 3} is rejected up front with
  `Unsupported(JpegComponentCount { components })`. Previously a 4-component
  CMYK JPEG would index `hdr.components[0..3]` and silently misinterpret
  the 4th component as Cr.
- `ImageDecoder::decode_from_reader{,_dyn}` no longer clones the input
  buffer on every call (split-borrow via a new private `decode_into_inner`
  free function). Restores the documented "allocate once, decode many"
  invariant on the file/`Read` path used by Python/C surfaces.

## [0.22.1] - 2026-05-15

### Added

- **NEON SIMD optimizations for decoder post-processing pipeline** (DE-2557):
  tiered dispatch with runtime CPU feature detection (NeonBase, FP16, DotProd,
  I8MM). Key kernels:
  - Fused softmax+weighted_sum: eliminates intermediate memory traffic by
    computing in-register (2 passes instead of 5, saves 256 memory ops per
    anchor).
  - NEON weighted_sum with DotProd (`udot`) and I8MM (`usmmla`) fast paths.
  - Vectorized dist2bbox (LTRB to XYXY) processing 4 boxes per iteration.
  - Fused dequant+sigmoid with FP16 narrowing variant for A55+.
  - NMS `jaccard_batch4`: vectorized IoU for 4 candidates simultaneously.
  - Benchmark gains vs v0.22.0: **1.60–1.67×** on i.MX 8M Plus (A53),
    **1.08–1.10×** on i.MX 95 (A55).

- Apple Silicon NEON optimizations enabled for macOS CI (`+dotprod`, `+i8mm`
  target features for `aarch64-apple-darwin`).

- `BoundingBox` is now `#[repr(C)]` with compile-time size/alignment asserts,
  guaranteeing 4 contiguous f32 fields for SIMD `vld1q_f32` loads.

### Changed

- Test fixtures migrated from `include_bytes!`/`include_str!` to runtime
  reads via `edgefirst_bench::testdata` helper. Reduces binary sizes and
  allows testdata to live outside the compiled artifact.

- CI: upgraded GitHub Actions to latest upstream versions, x86_64 runners
  to `ubuntu-22.04-xlarge`, and added binary stripping for hardware
  deployment artifacts.

### Fixed

- `logical_to_legacy_config_output` squeezing the padding dimension for
  ModelPack "boxes" output shape, causing `RuntimeError "Invalid ModelPack
  Boxes shape [1, 1935, 4]"`. Added a condition to skip squeeze when the
  decoder type is ModelPack (DE-2490).

## [0.22.0] - 2026-05-11

### Breaking Changes

- `DecoderBuilder` default NMS mode changed from `Nms::ClassAgnostic` to
  `Nms::Auto`. `Nms::Auto` resolves from the model config (e.g.
  `edgefirst.json`) and falls back to `ClassAgnostic` when no config
  specifies a mode. Runtime behaviour is identical for callers that did
  not previously set NMS via config — but downstream code with exhaustive
  matches on `Nms` must add the new `Auto` variant.

- Python `Decoder` constructors (`__init__`, `from_outputs`,
  `from_json_str`, `from_yaml_str`) now default the `nms` parameter to
  `Nms.Auto` instead of `Nms.ClassAgnostic`.

### Added

- **`Nms::Auto` variant** for the NMS configuration enum. Lets the
  decoder resolve NMS mode from the model config at build time, falling
  back to `ClassAgnostic`. Exposed across all bindings:
  - Rust: `configs::Nms::Auto`
  - Python: `Nms.Auto`
  - C: `HAL_NMS_AUTO`

- **`hal_decoder_params_set_pre_nms_top_k()`** and
  **`hal_decoder_params_set_max_det()`** — new C API functions to
  configure the pre-NMS top-K and post-NMS detection cap. Previously
  only accessible from Rust/Python.

- **Native i16 mask coefficient support** with optimized i16×i8 integer
  dot-product kernel. Avoids f32 widening for models that emit 16-bit
  quantized mask coefficients (e.g. INT16-calibrated segmentation heads).

- **NCHW proto layout** in the mask decode f32-fallback path for i8
  protos. Previously only the integer fast-path handled NCHW protos.

- Layout-coverage decoder test fixtures (`testdata/decoder/`) and
  improved `parse_edgefirst` example with NCHW/logical/smart schema
  variants.

- Extended `pre_nms_top_k` documentation across Rust, Python, and C
  bindings explaining deployment vs COCO mAP evaluation trade-offs.

### Fixed

- `DecoderBuilder::with_nms()` now properly overrides the config-derived
  NMS default. Previously an explicit `with_nms(ClassAware)` could be
  silently replaced by the config's NMS setting.

- i16 quantization fallback in mask decode: when i16 coefficients lack
  quantization metadata, the decoder now correctly falls through to the
  f32 dequant path instead of panicking.

## [0.21.0] - 2026-05-08

### Breaking Changes

- The eight public NMS functions in `crate::float` and `crate::byte`
  (`nms_float`, `nms_extra_float`, `nms_class_aware_float`,
  `nms_extra_class_aware_float`, `nms_int`, `nms_extra_int`,
  `nms_class_aware_int`, `nms_extra_class_aware_int`) gained a
  `max_det: Option<usize>` parameter immediately after `iou`. Pass
  `None` to preserve pre-0.20.1 behaviour (full O(N²) suppression).
  Pass `Some(n)` to early-terminate the greedy loop after `n` survivors
  are confirmed; survivors are guaranteed to match the top-`n`-by-score
  result of full NMS because the sorted input puts higher scores first.
  All in-tree `Decoder` paths and the `decode_yolo_*` wrappers thread
  the post-NMS cap through automatically; the breaking change only
  affects callers that import these functions directly.

- The four ModelPack decode functions in `crate::modelpack`
  (`decode_modelpack_det`, `decode_modelpack_float`,
  `decode_modelpack_split_quant`, `decode_modelpack_split_float`) and the
  `ModelPackDetectionConfig` struct are now `pub(crate)`. Use the
  `Decoder` struct API instead. Previously these functions also used
  `output_boxes.capacity()` as a post-NMS detection cap, which caused
  `Vec::new()` callers to silently receive zero detections.
  `Decoder::decode` now threads `Decoder::max_det` (default 300) into
  all ModelPack paths, consistent with the YOLO decode paths.

### Changed

- `Decoder::decode` / `decode_proto` / `decode_yolo_*` now skip the
  full O(N²) NMS suppression once `max_det` survivors are confirmed.
  No mAP impact (early-term survivors are identical to the top-`max_det`
  of full NMS), measurable latency win on dense scenes (large
  pre-NMS candidate sets).

- Workspace dependency refresh. No public API changes.
  - `ndarray` 0.16.1 → 0.17.2 (new `ArrayRef`/`LayoutRef`/`RawRef`
    types available; existing concrete `Array{1..4}` / `ArrayView*`
    usage unchanged).
  - `ndarray-stats` 0.6.0 → 0.7.0 (tracks ndarray 0.17).
  - `pyo3` / `pyo3-build-config` 0.26 → 0.28.3, `numpy` 0.26 → 0.28.0,
    `pythonize` 0.26 → 0.28.0. Forced by ndarray bump (numpy 0.28 is
    the first release supporting ndarray 0.17). All PyO3 0.27/0.28
    migration items addressed: `#[pyclass(from_py_object)]` opt-in
    added for `Clone` types, `Bound::from_owned_ptr_or_err` replaces
    `Py::from_owned_ptr_or_err`, `.cast()` replaces `.downcast()`.
    No deprecation warnings remain.
  - `nalgebra` 0.32.6 → 0.34.2. `Allocator<R, U8, U8>` /
    `Allocator<R, U8>` trait bounds in `tracker::kalman` migrated
    to the 2-generic form (`Allocator<U8, U8>` / `Allocator<U8>`)
    after the Scalar generic was removed in nalgebra 0.33.
  - `nix` 0.30.1 → 0.31.2.
  - `opencv` 0.95 → 0.98.2.
  - `safetensors` 0.4 → 0.7.0 (already pinned to 0.7 in
    `crates/decoder`; workspace alignment).
  - `fast_image_resize` 5.5.0 → 6.0.0.
  - `image-compare` 0.4.2 → 0.5.0.
  - `jpeg-encoder` 0.6.1 → 0.7.0.
  - `lapjv` 0.2.1 → 0.3.0.
  - `yuv` 0.8.12 → 0.8.14.
  - `zune-jpeg` 0.4.21 → 0.5.15, `zune-png` 0.4.10 → 0.5.2,
    `zune-core` 0.4 → 0.5. JPEG/PNG load paths in
    `crates/image/src/lib.rs` migrated to wrap the input slice in
    `ZCursor::new(...)` (the new `ZByteReaderTrait` no longer impls
    on `&[u8]` directly) and drop the `get_` prefix on
    `output_colorspace` / `info` / `colorspace` accessors.
  - `cargo update` lock refresh: `rayon` 1.11→1.12, `tokio` 1.50→1.52,
    `uuid` 1.22→1.23, plus assorted patch bumps.

### Deferred

- `libloading` 0.8 → 0.9: blocked by `khronos-egl 6.0.0` pinning
  `libloading ^0.8`. Will revisit when khronos-egl ships a release
  that supports libloading 0.9.
- `ctor` 0.4 → 1.0.x: 1.0 was tagged 4 days before this release after
  hopping through 0.5–0.13 in the prior week. Holding off until the
  surface stabilises.

## [0.20.0] - 2026-05-06

### Breaking Changes

- **`crate::yolo` is now a private module.** All `decode_yolo_*` and
  `impl_yolo_*` free functions plus the `MAX_NMS_CANDIDATES` /
  `DEFAULT_MAX_DETECTIONS` constants are `pub(crate)`. The
  `yolo_segmentation_to_mask` helper is also private. The supported
  decoder API surface is `Decoder` + `DecoderBuilder`; external
  callers that reached into `crate::yolo::*` (e.g. the
  `impl_yolo_segdet_quant_proto` direct entry point) must migrate to
  `Decoder::decode` / `Decoder::decode_proto`. Building a `Decoder`
  with `DecoderBuilder::with_config_yolo_segdet` (or `with_schema`)
  reaches the same kernels with the schema-derived quant, normalize,
  and input-dim plumbing wired in.
- `crate::yolo::decode_segdet_f32` and `crate::yolo::decode_segdet_quant`
  (private helpers consumed by the segmentation decode pipeline)
  changed return type from `Vec<(DetectBox, Array3<u8>)>` to
  `Vec<(DetectBox, BoundingBox, Array3<u8>)>`. The new middle element
  is the proto-grid-aligned roi reported back so `Segmentation`
  bounds can describe the cropped tensor independently of the bbox
  (EDGEAI-1304 follow-up). Module is now `pub(crate)` so this affects
  the in-tree `decoder/postprocess.rs` callers only.
- Behavioural change for callers that exploited the pre-EDGEAI-1302
  `output_boxes.capacity()` cap as a per-call detection limit:
  `Decoder::decode()` and `Decoder::decode_proto()` now ignore
  capacity entirely and bound the output by `Decoder::max_det` (set
  via `DecoderBuilder::with_max_det`, default 300). Migrate by setting
  the cap on the builder; the previous workaround silently dropped
  detections to zero when `Vec::new()` was passed.
- Behavioural change for callers that worked around EDGEAI-1303 by
  patching pixel-space boxes to `[0, 1]` in-graph (e.g. a final
  `Mul([1/W, 1/W, 1/W, 1/W])` node): if the schema also declares
  `Detection::normalized = false` and the input spec carries known
  W/H, the decoder will now divide a second time, producing
  `[0, 1/W²]`-scale boxes that fail to decode. Migrate by removing
  the in-graph workaround OR setting `normalized: true` in the
  schema.

### Added

- `DecoderBuilder::with_input_dims(width, height)` setter, plus
  `Decoder::input_dims()` accessor, mirroring the existing
  `with_*` / `*_boxes()` pair for the `normalized` flag. Lets
  callers building from `add_output(...)` (programmatic) or
  config files without an `input` block opt into EDGEAI-1303
  normalization without rewriting their schema. Schema-derived
  dims are still picked up automatically; the explicit setter
  takes precedence so callers can correct misdeclared schemas.
  Mirrored in the Python and C bindings:
  - Python: every `PyDecoder` constructor (`__init__`,
    `Decoder.from_outputs`, `Decoder.from_json_str`,
    `Decoder.from_yaml_str`) gained an `input_dims=None` keyword
    argument; `Decoder.input_dims` is exposed as a read-only
    property next to `Decoder.normalized_boxes`.
  - C: `hal_decoder_params_set_input_dims(params, width, height)`
    and `hal_decoder_input_dims(decoder, *width, *height)` for
    the setter and accessor respectively.
- New `per_scale` decoder subsystem for schema-v2 per-scale models.
  Supports DFL (yolov8 / yolo11) and LTRB (yolo26) box encodings, all
  combinations of i8/u8/i16/u16/f16/f32 inputs and f32/f16 outputs.
  Selected automatically when the schema declares per-scale children;
  legacy `merge::PerScale` path is deprecated and errors loudly.
- Per-scale subsystem now supports both **NHWC and NCHW children** via
  named-axis dshape lookup. NCHW children are transposed to a per-dtype
  scratch buffer (`LayoutScratch`) before the existing NHWC kernel
  dispatch, keeping kernel coverage uniform for NEON optimization.
- **NEON 16x16 byte tile transpose** for the NCHW → NHWC scratch
  step, replacing the scalar walk for `u8` / `i8` inputs when
  `c >= 16` and `h*w >= 16`. Uses the canonical 4-stage TRN1/TRN2
  pattern (`.16b` → `.8h` → `.4s` → `.2d`) to permute a 16-byte tile
  in registers. Targets the Ara-240 NCHW path; NHWC inputs (TFLite,
  the canonical fixtures) skip the transpose entirely and are
  unaffected.
- **NEON (Tier 1) kernels** for the per-scale subsystem on
  aarch64. Adds `kernels::neon_baseline` with NEON i8/u8/i16/u16 → f32
  dequant primitives (FMA-fused affine), NEON sigmoid f32 (vbslq
  blend), and NEON softmax f32 (3-pass max/exp/normalize). All
  `BoxLevelDispatch` / `ScoreLevelDispatch` / `MaskCoefDispatch` /
  `ProtoDispatch` enums carry `*NeonBase` variants; `select()` picks
  NEON when `CpuFeatures::neon_baseline` is true. Bit-correct against
  scalar oracle within 1-2 ulp envelope.
- **Polynomial NEON `expf` (f32, 4-lane)** replaces lane-extract
  scalar libm `expf` in NEON sigmoid + softmax. Cephes 5-degree
  minimax with split-`ln2` range reduction and IEEE-754 exponent-bit
  injection of `2^k`. Profiling revealed libm `expf` was 58% of
  decode time on Cortex-A53 — the polynomial drops decode 74 ms →
  37 ms (50%) on imx8mp-evk with mAP unchanged at 0.417.
- **Polynomial NEON `expf` (f16, 8-lane)** for ARMv8.2-A
  Cortex-A55+ targets (i.MX95, RPi5, Jetson Orin). Uses
  `.arch_extension fp16` inline-asm escape hatch since Rust 1.94
  stable does not expose the FP16 NEON intrinsics. New
  `*NeonFp16` dispatch tier sits above `NeonBase`; selected when
  `CpuFeatures::neon_fp16` is detected. Inputs clamped to ±10 to
  keep `2^k` injection inside f16's 5-bit-exponent range. Drops
  imx95-evk decode 29.5 ms → 23.5 ms (-20%) on top of the f32
  polynomial.
- **Per-level rayon parallelization** for the per-scale pipeline.
  Each FPN level's box / score / mc work runs on a separate thread;
  the heaviest level (index 0, 80x80 grid) runs on the calling
  thread to avoid one round-trip barrier. Level-scoped disjoint
  slices are pre-split via iterative `split_at_mut`. Activates only
  when all levels are NHWC (NCHW would need per-level scratch).
  Drops imx8mp 36.9 ms → 30.8 ms (Cortex-A53, 4 cores) and
  imx95 23.5 ms → 20.5 ms (Cortex-A55, 6 cores).
- **End-to-end NEON speedups** vs original libm scalar `expf`:
  imx8mp-evk 74.4 ms → 30.8 ms (2.42×); imx95-evk ~60 ms →
  20.5 ms (~2.93×). coco128 box mAP@[0.5:0.95] holds at 0.417
  across all variants (within rounding of the FP32 reference).
- **Perfetto / Chrome-trace spans** throughout the per-scale
  subsystem: `per_scale_decode`, `resolve_bindings`, per-level
  `level`, and per-role `box` / `score` / `mc` / `protos`. Drove
  the hotspot analysis.
- `DecoderBuilder::with_decode_dtype` for choosing f32 or f16 outputs
  through the per-scale pipeline.
- `per_scale::apply_schema_quant` helper that walks a schema-v2
  document and attaches per-tensor `Quantization` to integer input
  tensors via shape match.  Use when the upstream inference layer
  hasn't already attached quantization metadata.
- `EDGEFIRST_DECODER_FORCE_KERNEL` env var for forcing the decoder's
  kernel tier (`scalar`, `neon`/`neon_baseline`; `neon_fp16` and
  `neon_dotprod` are recognised but not yet wired to kernels).
- New `BoxEncoding::Direct` serde alias `"ltrb"` so yolo26 schemas
  (which declare `"encoding": "ltrb"`) deserialize correctly.
- Per-scale baseline benchmarks (`decoder/per_scale/dfl_i8_to_f32`,
  `dfl_i8_to_f16`, `ltrb_i8_to_f32`) in `decoder_benchmark`.
- `testdata/decoder/<model>.safetensors` reference fixtures backed by
  git-lfs (`yolov8n-seg`, `yolo11n-seg`, `yolo26n-seg`), with
  `scripts/decoder_generate_fixture.py` for regeneration. Each fixture
  bundles raw int8 outputs, per-stage NumPy reference intermediates,
  post-NMS detections, the patched schema (with `dshape` synthesized
  on logical parents), and a Markdown narrative under
  `__metadata__["documentation_md"]`. New parity tests in
  `crates/decoder/tests/per_scale_parity.rs` (smoke / end-to-end /
  pre-NMS) consume the fixtures via `tests/common/per_scale_fixture.rs`
  and the `Decoder::_testing_run_per_scale_pre_nms` capture entry
  point.

### Changed

- Per-scale schemas now apply sigmoid activation on score logits when
  the schema declares `activation_required: sigmoid` on the per-scale
  children (previously omitted by `merge.rs::execute_per_scale`).
- Per-scale → legacy bridge (`per_scale_bridge::widen_to_f32`) is now
  hybrid: zero-copy borrow when the per-scale buffer is already `f32`
  (the default for `DecodeDtype::F32`), and one owned allocation only
  when widening from `f16`. Previously each decode unconditionally
  copied boxes/scores/mask_coefs/protos to fresh `Vec`s — for a
  yolov8-seg model that's ~7 MB/decode of avoidable allocation. Both
  `Decoder::decode` and `Decoder::decode_proto` and the per-scale
  bridge now also emit `tracing::trace_span!` spans
  (`Decoder::decode`, `per_scale_bridge::widen_to_f32` with a
  `kind=f32_borrow|f16_widen` attribute, `nms_get_boxes`,
  `process_masks`, `extract_proto_data`) so callers can profile each
  stage independently.

### Fixed

- yolo26 LTRB-encoded per-scale boxes now decode correctly. Previously
  `merge.rs::plan_dfl` only handled DFL encoding and yolo26 schemas
  failed at plan time.
- Per-scale TFLite schemas no longer require validator NumPy-NMS
  fallback; the HAL handles them natively.
- `Decoder::decode()` and `Decoder::decode_proto()` no longer treat
  `output_boxes.capacity()` as a `max_det` cap on **any** of the
  legacy YOLO paths (`YoloSegDet` combined-output, `YoloSplitSegDet`
  three-output, `YoloSegDet2Way` split-detection). The post-NMS
  detection count is now bounded only by `Decoder::max_det` (default
  300, set via `DecoderBuilder::with_max_det`); capacity is purely an
  allocation hint. Previously, callers passing `Vec::new()` (capacity
  0) silently received zero detections, and callers passing
  `Vec::with_capacity(N)` got an implicit per-call cap of `N`
  (EDGEAI-1302). See **Breaking Changes** above for the workaround-migration
  note.
- `Decoder::decode()` no longer overwrites post-NMS bbox coordinates
  with the proto-grid-quantized roi from `protobox`. Returned bboxes
  are now bit-identical to those `Decoder::decode_proto()` produces
  for the same input. Previously, `decode()` snapped every coordinate
  to a `1/160` grid, producing bit-identical duplicates on cluttered
  scenes and corrupting same-class IoU evaluation (EDGEAI-1304). The
  accompanying `Segmentation` bounds (`xmin/ymin/xmax/ymax`) now
  describe the proto-grid-aligned crop region of the returned mask
  tensor, so the bbox and the mask region are independently
  consistent with their respective data.
- HAL decoder now honours `Detection::normalized: false` across the
  combined-`Detection` path **and** the three-output split path
  (`Boxes` + `Scores` + `MaskCoefficients` + `Protos`) used by
  vanilla Ultralytics ONNX `--export-split` exports. When the
  schema declares pixel-space boxes and the model input dimensions
  are known (extracted from `input.shape` / `input.dshape` in v2
  schemas), the decoder divides bbox channels by `(W, H)` before
  NMS so `protobox` and IoU semantics see normalized coords. The
  fallback handles partially-named dshapes (e.g. only `Width`
  declared) by NHWC-positional inference of the missing axis instead
  of silently disabling normalization (EDGEAI-1303). The `protobox`
  rejection message now distinguishes "schema missing input spec"
  from "schema declares normalized incorrectly" so the recommended
  fix is actionable. See **Breaking Changes** above for the
  in-graph-workaround migration note.

## [0.19.0] - 2026-05-05

### Breaking Changes

- **`MaskResolution::Proto` and `MaskResolution::Scaled` now return binary
  `{0, 255}` masks** instead of continuous sigmoid `[0, 255]` values. The
  sign-threshold optimization (`dot > 0 → 255`) gives the same segmentation
  result as `sigmoid(dot) > 0.5` but eliminates the sigmoid computation
  entirely. Callers that relied on intermediate confidence values should apply
  custom sigmoid to the raw `ProtoData` tensor (via `decode_proto()`).
- **`impl_yolo_segdet_quant_proto` and `impl_yolo_split_segdet_quant_get_boxes`**
  gained `pre_nms_top_k` and `max_det` parameters. Downstream Rust callers
  using these internal APIs must update their call sites.
- **`ProtoData.protos` shape depends on layout.** When
  `ProtoData.layout == ProtoLayout::Nchw`, the tensor shape is
  `[num_protos, H, W]` instead of `[H, W, num_protos]`. Python callers should
  check the new `ProtoData.layout` property; C callers should use
  `hal_proto_data_layout()`.

### Added

- **Comprehensive tracing instrumentation** across all HAL crates for
  Perfetto/Chrome JSON profiling. Spans cover decode pipeline, mask
  materialization, image conversion (per-pass), tracker update, tensor
  allocation, and Python/C entry points. Enabled by default; near-zero
  overhead (single atomic load per span site) when no subscriber is active.
- `hal.Tracing(path)` Python context manager and `hal_start_tracing()` /
  `hal_stop_tracing()` C API for capturing trace sessions viewable at
  <https://ui.perfetto.dev/>.
- `TracingError::SessionExhausted` variant to distinguish "session already
  used" from "currently active" and "user-installed subscriber" errors.
- `ProtoData.layout` property in Python and `hal_proto_data_layout()` in C API
  to query proto tensor memory layout (NHWC vs NCHW).
- Pre-NMS top-K filtering (`Decoder.pre_nms_top_k`) to reduce O(N²) NMS cost.
  Default: 300. Skipped when `nms=None` (bypass mode).
- `Decoder.max_det` to cap post-NMS output count (default: 300, matches
  Ultralytics convention).
- NEON-accelerated column-major argmax for quantized score tensors on aarch64,
  with automatic fallback to row-major for models with > 255 classes.
- NCHW proto layout detection — skips the 3.1 ms NCHW→NHWC transpose on
  Cortex-A53/A55 when the model outputs protos in channel-first order.
- x86_64 F16C+FMA vectorized kernel for f16 proto mask materialization
  (sign-threshold variant).

### Changed

- `tracing` Cargo feature is now enabled by default in `edgefirst-hal`,
  `edgefirst-hal-capi`, and the Python crate. Trace capture remains opt-in
  at runtime (call `start_tracing()` to begin recording).
- C API tracing functions (`hal_start_tracing`, `hal_stop_tracing`,
  `hal_is_tracing_active`) are always exported regardless of feature flags;
  return `ENOSYS` when compiled without `tracing` feature.
- Python `Tracing` class is always available; raises `RuntimeError` when
  tracing feature is not compiled in.
- `pre_nms_top_k` is now ignored when `nms=None` (NMS bypass mode preserves
  all above-threshold candidates).
- Detection output is always sorted by descending score after NMS for
  deterministic ordering.
- Python `max_boxes` parameter now enforces a hard output limit (truncates
  boxes, masks, and tracks to `max_boxes` after decode).
- `decode_proto()` allocates proto tensors with the correct shape and layout
  even when zero detections survive NMS (previously the shape did not reflect
  the model's physical layout for NCHW protos).
- Column-major argmax gracefully falls back to row-major when models have
  > 255 classes (previously panicked with an assertion failure).

### Fixed

- NCHW stride detection now uses exact stride matching `[W, 1, H*W]` instead
  of the overly broad `strides()[2] > 1` check that could misclassify
  arbitrary strided views as contiguous NCHW buffers.
- Mutex poisoning in tracing API no longer causes process-wide panics
  (uses `unwrap_or_else(|e| e.into_inner())` recovery pattern).

## [0.18.1] - 2026-05-01

### Performance

- **Restore `materialize_masks` parallelism + logit-precompute (regressed
  by PR #51).** PR #54 originally added rayon `par_iter` across
  detections plus a batched-GEMM-style precompute that hoisted the
  K-wide dot-product out of the per-output-pixel inner loop. PR #51
  (EDGEAI-1244 f16 dispatch refactor) inadvertently rewrote both
  `materialize_segmentations` (Proto) and `materialize_scaled_segmentations`
  (Scaled) to use serial `.iter()` and per-output-pixel bilinear-sample-
  then-dot, silently dropping both optimisations. Restoring them:
  - `materialize_segmentations`: rayon parallel across detections, with
    the proto-tensor `map()` guard acquired once per call instead of
    once per detection.
  - `materialize_scaled_segmentations`: rayon parallel across detections
    plus a per-detection logit-precompute pass — for each detection,
    compute f32 logits at every proto-plane ROI pixel via a single
    K-wide dot, then bilinear-interpolate the scalar logit at every
    output pixel before applying sigmoid + threshold. Crucially the
    bilinear upsample still operates on continuous logits, not on the
    sigmoid-quantised u8, so accuracy is preserved.
  Measured on imx8mp-frdm with the corrected ara2-validator on
  coco128-seg (N=119 avg detections per image, max_det=300,
  score_threshold=0.001):
    `materialize_hal` mean **569 ms → 33 ms (17× faster)**;
    mask mAP unchanged at 0.3220 (vs the dvapi/numpy reference's 0.3221
    — 0.0001 absolute drift).

### Changed

- **`Tensor.from_numpy()` fast path for fully-strided sources.** A numpy
  view that is non-contiguous *and* has no contiguous trailing
  dimension (e.g. `(1, 116, 8400)` produced by `arr.transpose(0, 2, 1)`
  on a `(1, 8400, 116)` backing buffer — the layout HailoRT returns
  natively) previously fell back to per-element strided iteration in
  Path 3. On rpi5-hailo this measured ~27 ms/call versus ~6.5 ms when
  the caller pre-applied `np.ascontiguousarray()`. Path 3 now performs
  that materialization internally via `np.ascontiguousarray()`, whose
  vectorized C strided→contig pass is dramatically faster than
  ndarray's stride-respecting iterator. Callers that maintained a
  manual `np.ascontiguousarray(...)` workaround can now drop it; the
  fast path is automatic.

### Documentation

- **Optimization Guide refresh: NumPy interop and `MaskResolution`
  doctrine.** The cross-document Optimization Guide gains two new
  rules and matching cost-table rows:
  - **Rule 7 — NumPy interop:** pass numpy arrays straight to
    `Tensor.from_numpy()`; do not pre-`ascontiguousarray()`. Documents
    the three internal paths (`copy_numpy_to_tensor_dyn`), the narrow
    case where pre-materialising still helps (a single strided view
    fed many times), and the rpi5-hailo numbers (≈ 6.5 ms automatic
    fast path vs ≈ 27 ms legacy element-wise loop).
  - **Rule 8 — `MaskResolution` selection:** for COCO / IoU evaluation
    use `MaskResolution::Scaled(orig_w, orig_h)`. The default
    `MaskResolution::Proto` returns continuous sigmoid values at proto
    resolution; if a downstream caller thresholds those *before*
    upsampling (the natural-looking but wrong order), edges become
    blocky and mask mAP regresses by 0.04–0.05 absolute on YOLOv8-seg
    / `coco128-seg`. `Scaled` performs the upsample inside HAL's
    batched-GEMM kernel and thresholds afterwards.
  Touches `README.md` (rule table + Rule 7/8 sections),
  `BENCHMARKS.md` (cost table + NumPy Interop Fast-Path section),
  `ARCHITECTURE.md` (three-path strategy under Python Bindings), and
  `TESTING.md` (validation pointers to `test_from_numpy_hailort_shape`
  and `MaskResolution.Scaled` smoke checks).

## [0.18.0] - 2026-04-25

### Added

- **Native f16 support for decoder output tensors.** The YOLO
  segmentation decoder now accepts `Tensor<half::f16>` for proto
  masks and mask coefficients natively, alongside the existing `f32`
  and `i8+quantization` paths. Targets the Orin Nano TensorRT-fp16
  engine (`yolo26x-seg-t-2592-fp16.engine`) and any future f16-output
  model. On ARMv8.2-FP16 (Orin Cortex-A78AE) and x86_64+F16C
  (Haswell+) the scalar kernel lowers `half::f16::to_f32()` to a
  single `fcvt` / `vcvtph2ps`; x86_64 additionally gets an explicit
  `_mm256_cvtph_ps` + `_mm256_fmadd_ps` intrinsic kernel for
  deterministic 8-lane codegen. On Cortex-A53 (imx8mp, no FP16
  hardware) the fallback path calls the soft-float `__extendhfsf2`
  helper per proto load — correctness-preserving at reduced
  throughput. Resolves EDGEAI-1244.

- **`edgefirst_tensor::Quantization` as unified tensor-level
  metadata.** Covers all four modes defined by the edgefirst.json
  spec: per-tensor symmetric, per-tensor asymmetric, per-channel
  symmetric, per-channel asymmetric. Fallible constructors
  (`per_tensor`, `per_tensor_symmetric`, `per_channel`,
  `per_channel_symmetric`) reject length-mismatched / empty inputs
  at construction with structured `Error::QuantizationInvalid`.

- **Type-gated quantization accessors on `Tensor<T>`.** The sealed
  `IntegerType` trait restricts `Tensor::<T>::quantization()`,
  `set_quantization()`, `with_quantization()`, `clear_quantization()`
  to integer `T` (`u8/i8/u16/i16/u32/i32/u64/i64`). Float `T`
  (`f16/f32/f64`) has no accessor — `Tensor::<f32>::quantization()`
  fails to compile. A compile-fail doctest anchors the invariant.

- **`QuantMode` dispatch enum** for kernel entry. Matches once at the
  kernel boundary to pick a monomorphized inner kernel
  (per-tensor vs. per-channel), avoiding `&Quantization` polymorphism
  in hot loops.

- **CPU mask kernel per-channel quantization support.** The fused
  dequant + dot + sigmoid kernel handles per-channel scales on axis 2
  (the channel dimension) in addition to per-tensor. Other
  per-channel axes return `Error::NotSupported`.

- **TensorDyn proto-data access through the C API.**
  `hal_proto_data_take_protos()` and
  `hal_proto_data_take_mask_coefficients()` transfer ownership of
  each tensor out of a `HalProtoData*`. Subsequent calls for the
  same field return NULL. Design motivated by `TensorDyn` not
  implementing `Clone`.

- **`HalTensorQuant` opaque handle** (C API) with accessors
  `hal_quantization_{kind, scale_len, scale_at, zero_point_at,
  is_symmetric, axis}`. Obtained via `hal_tensor_quantization()`;
  returns NULL for float or unquantized integer tensors. Named
  distinctly from the pre-existing flat `HalQuantization` struct
  (legacy decoder-side scalar quantization plumbing).

- **Python `Quantization` pyclass** with staticmethod constructors
  and property accessors for all four modes.

- **Python `Tensor.quantization` property and
  `set_quantization_{per_tensor,per_tensor_symmetric,per_channel,
  per_channel_symmetric}` / `clear_quantization` methods.** Float
  tensors raise on set.

- **Python `ProtoData.take_protos()` and `take_mask_coefficients()`**
  consuming accessors returning `Optional[Tensor]`.

- **Build configuration:** `.cargo/config.toml` enables `+fp16` on
  `aarch64-apple-darwin` only (every Apple Silicon M-series chip has
  FEAT_FP16; Intel Macs build under `x86_64-apple-darwin` which is
  unaffected). All other targets keep their baseline ISA so a single
  distributed binary stays portable across older CPUs within the same
  triple. Developers benchmarking on hosts with richer ISAs (Orin fp16,
  x86_64 F16C) opt in via `RUSTFLAGS` — see `README.md` and
  `TESTING.md`. Runtime CPU-feature detection + dynamic dispatch for
  broader enablement is deferred to a future release.

- **`scripts/audit_f16_codegen.sh`** validates that the f16 kernel's
  release disassembly contains `fcvt` / `vcvtph2ps` (not
  `__extendhfsf2`) on the target, for developers verifying a local
  benchmarking build.

- **Batched-GEMM `materialize_masks` path** for COCO-style validation
  workloads (high `max_det`, low score threshold). A single batched
  GEMM at proto resolution
  (`coeffs (N, K) · protos.T (K, H·W)`) replaces the per-detection
  scalar kernel via `ndarray::linalg::general_mat_mul`
  (`matrixmultiply` backend, no new deps), with rayon-parallel
  per-detection finalisation and a `MaskScratch` pool on
  `CPUProcessor` to amortise the ~3 MB f32 dequant across frames.
  Per-detection fused kernels remain the small-N fallback (Proto
  N ≥ 16, Scaled N ≥ 2). Cross-platform A/B at N=100 scaled
  640×640: imx8mp 1.4 s → 153 ms (9.1×), imx95 1.4 s → 114 ms
  (12.3×), rpi5-hailo 466 ms → 42 ms (10.9×), x86 desktop 461 ms
  → 10 ms (44.7×). Full table in `BENCHMARKS.md`.

- **NEON kernels for the scaled-resolution mask path on aarch64.**
  `fill_scaled_tile_neon` processes 4 output pixels per iteration
  along the bbox row (4-wide `vld1q_f32` corner gather, FMA
  bilinear, `u32x4 → u16x4 → u8x4` pack); `dequant_i8_to_f32_neon`
  converts 16 i8 → 16 f32 per iteration with the standard
  `sxtl/scvtf/fsub/fmul` cascade. Dispatch fast-path skips ndarray
  indexing via `Array::as_slice()`; the float branch collapses to a
  single `copy_from_slice`. Non-aarch64 keeps the scalar path.
  Stacked on the batched-GEMM win, this brings i.MX 95 N=100 scaled
  640×640 from 1400 ms → 44 ms — a cumulative **31×** speedup.

- **Schema v2 logical *and* physical outputs may omit `type`.**
  `LogicalOutput.type_` and `PhysicalOutput.type_` are now
  `Option<*Type>` with `#[serde(default, skip_serializing_if =
  "Option::is_none")]`. Typeless outputs are carried in the schema
  for round-trip but filtered out of the legacy `ConfigOutputs`
  produced by `to_legacy_config_outputs`, and skipped by
  `DecodeProgram::try_from_schema` so the merge path stays aligned
  with the legacy outputs the decoder consumes. Encodes the design
  principle "decoders find the outputs they know about; they
  shouldn't care about outputs they don't, as long as they can
  capture the ones they require." Models that ship auxiliary
  / diagnostic tensors alongside the YOLO heads now parse without
  pre-emptive serde errors. Five regression tests cover the parse,
  round-trip, filter, and physical-uniqueness-skip behaviours.

- **`load_png` now accepts any converter-reachable destination
  format.** Greyscale PNGs (Luma) load directly as
  `PixelFormat::Grey`; LumaA is normalised inline (alpha stripped);
  RGB ↔ Grey, Grey → Rgba/Bgra, etc. dispatch through
  `CPUProcessor::convert_format_pf`. Previously `load_png` rejected
  anything other than Rgb/Rgba with `NotImplemented`, forcing
  callers off PNG for greyscale inputs. JPEG already supported Grey.

- **`make wheel` target builds release Python wheels via maturin
  with `--zig` and `--compatibility manylinux2014`.** `TARGET=<triple>`
  selects cross-compile targets; `PYABI=py38|py311` produces an
  abi3 stable-ABI wheel using the existing features in
  `crates/python/Cargo.toml`.

- **`hal_image_processor_draw_proto_masks` C API.** Fused GPU-
  accelerated proto-mask rendering: takes pre-decoded boxes and raw
  `HalProtoData*` from the decoder and computes
  `mask_coeff @ protos` at full output resolution via the GLES 3.1
  compute-shader path with bilinear sampling. Lets
  `edgefirstoverlay`'s fused rendering path skip the intermediate
  160×160 materialize + upscale.

### Changed

- **`ProtoData` now holds two `TensorDyn` fields** (`protos` and
  `mask_coefficients`) instead of a custom `ProtoTensor` enum + flat
  `Vec<Vec<f32>>` coefficient list. Deleted the `ProtoTensor` enum
  and `IntoProtoTensor` trait — a single generic `FloatProtoElem`
  impl chain now routes `f16 / f32 / f64` inputs through
  `Tensor::from_slice` / `Tensor::from_arrayview3` without unsafe
  TypeId casts. The quantized `i8` proto path attaches its per-tensor
  quantization metadata via `Tensor::<i8>::with_quantization()`
  directly.

- **Mem-backed tensors accept zero-element shapes** (`[0, N]`). The
  tracker path's genuine "no detections this frame" sentinel now
  produces a zero-row coefficient tensor rather than a placeholder
  1-row tensor. DMA-backed storage continues to reject zero-size.

- **CPU mask kernels hoist tensor maps out of the per-detection
  loop** (fixes the ~3-13 MB per-detection allocation flagged in
  code review). `materialize_segmentations`,
  `materialize_scaled_segmentations`, and `draw_proto_masks` dispatch
  on `TensorDyn::dtype()` once and call slice-native kernels
  (`fused_dequant_dot_sigmoid_i8_slice`, `fused_dot_sigmoid_f32_slice`,
  `fused_dot_sigmoid_f16_slice`) for the inner work.

- **GL renderer dispatches on `TensorDyn::dtype()`** and returns
  `Error::NotSupported` for per-channel quantization (GL shader-side
  per-channel is deferred). CPU path handles per-channel; callers
  should fall back to CPU when the GL renderer rejects.

- **Schema v2 `Quantization` now converts cleanly to
  `edgefirst_tensor::Quantization`** via `TryFrom<&schema::Quantization>`,
  covering all four modes.

- **`ConfigOutput::MaskCoefficients` serde tag renamed to `mask_coefs`
  with `mask_coefficients` kept as a backward-compatible alias.** The v2
  spec vocabulary uses `mask_coefs`; the legacy v1 spelling
  (`mask_coefficients`) continues to parse unchanged so existing models
  and fixtures are not affected.

- **`ImageProcessor::materialize_masks` and
  `CPUProcessor::materialize_{segmentations,scaled_segmentations}`
  now take `&mut self`** so the batched-GEMM path can hold its
  `MaskScratch` borrow across the call. **Source-breaking for
  Rust callers that hold `ImageProcessor` / `CPUProcessor` by shared
  reference.** Python and C API call sites are unchanged. The
  `draw_proto_masks` hybrid path was updated so the CPU borrow is
  scoped before the GL borrow.

- **`load_jpeg`, `load_png`, and `rotate_flip_to_dyn` pad row stride
  to GPU alignment when materialising into DMA storage on Linux.**
  Mali G310 (i.MX 95) rejects `eglCreateImage` from DMA-BUFs whose
  `PLANE0_PITCH_EXT` is not a multiple of 64 bytes, surfacing as
  `EGL_BAD_ALLOC` and silently falling back to CPU texture upload
  (~2× slowdown on COCO validation). The decoder now stages the
  image into a tightly-packed Mem buffer at natural pitch then
  row-copies into a pitch-padded DMA tensor allocated via
  `Tensor::image_with_stride`. Shared helpers
  `padded_dma_pitch_for` and `copy_packed_to_padded_dma` cover both
  load paths plus the EXIF rotate/flip fixup. Non-Linux targets and
  callers requesting non-DMA memory take the natural-pitch fast
  path unchanged.

- **`DmaImportAttrs::from_tensor` accepts non-4-aligned widths for
  4-bpp packed formats (`Rgba`, `Bgra`).** The pre-existing
  `width % 4 == 0` check was rejecting dataset-loader widths like
  375 / 427 / 443 even though their natural pitch (`width × 4`)
  is trivially 4-byte aligned and the DRM `ABGR8888`/`ARGB8888`
  fourcc spec imposes no pixel-alignment requirement. The check
  remains in force for `Rgb` (3 bpp) and `Grey` (1 bpp) where the
  natural pitch only word-aligns at multiples of 4. Verified
  end-to-end on Vivante GC7000UL (imx8mp), Mali G310 (imx95), V3D
  7.1 (rpi5-hailo), and Tegra (orin-nano).

- **Scaled-resolution mask kernel skips redundant work.**
  `scaled_segmentations_*` thresholds at `sigmoid > 0.5`, which is
  monotonically equivalent to `acc > 0` — the
  `fast_sigmoid` call (and the i8 path's positive-`scale` multiply)
  is dropped. `MaskScratch::compute_logits` now grows its dequant /
  logits buffers on demand instead of `Vec::resize(n, 0.0)` every
  frame, since the buffers are immediately overwritten by the
  dequant loop and the GEMM (`beta=0.0`). On i.MX 95 these two
  changes amount to a further 12–17 % win on the scaled path.

- **GL and CPU mask kernels reclassify per-channel-on-non-channel-axis
  rejections from `InvalidShape` to `NotSupported`,** and the
  `num_protos > 64` path in `fused_dequant_dot_sigmoid_i8_slice` now
  allocates a heap scratch buffer instead of erroring. Production
  models stay within the stack-scratch fast path; larger models work
  correctly at slightly higher per-call cost.

- **GL `render_proto_segmentation` no longer rebuilds
  `Vec<Vec<f32>>` mask coefficients or `to_owned()` clones the
  proto tensor on every frame.** Sub-helpers
  (`render_proto_detection_quads`,
  `render_proto_segmentation_{int8,f32,f16,f16_native}`,
  `render_proto_int8_two_pass`, and the two `repack_protos_*`
  helpers) take flat `&[f32]` coefficients with a `num_protos`
  stride and `ndarray::ArrayView3<'_, T>` for protos. F32 coeffs +
  F32 protos: zero hot-path heap allocations. F16 coeffs widen
  once into a flat `Vec<f32>`.

- **Schema v2 → legacy lowering rejects malformed
  `Quantization::Deserialize` shapes.** `Quantization::validate()`
  now defends against constructors bypassed via `Deserialize`:
  empty `scale`, per-tensor with multi-element `zero_point`, and
  per-channel with `zero_point.len() != scale.len()` are all
  rejected with structured `Error::QuantizationInvalid`.

### Fixed

- **Physical-order contract for `shape` / `dshape` in output configuration
  (EDGEAI-1288).** Two production bugs — TFLite YOLOv8-seg vertical-stripe
  mask corruption on i.MX 8M Plus and Ara-2 anchor-first split-tensor
  mis-decode — were both caused by ambiguity in how `shape` and `dshape`
  were interpreted relative to the producer's physical buffer layout. HAL
  now enforces a single, explicit rule: **`shape` and `dshape` passed to
  `hal_decoder_params_add_output` (or their YAML/JSON equivalents) must
  be in physical memory order, outermost axis first.** HAL derives
  C-contiguous strides from `shape` and wraps the buffer with those
  strides directly; when `dshape` is present HAL uses it to permute the
  stride tuple into the decoder's canonical logical order — no bytes are
  moved. Callers who omit `dshape` assert that `shape` is already
  canonical for the output role. The size-heuristic in `protos_to_hwc`
  (which tried to infer NCHW vs NHWC from channel-count comparisons) has
  been removed; producers with a non-canonical physical layout must now
  declare it explicitly via `dshape`. Validation at
  `DecoderBuilder::build` time catches size mismatches between `shape`
  and `dshape` entries with a clear error. **Non-breaking for all
  in-tree callers**, which already declared physical order; callers that
  relied on the removed heuristic must add an explicit `dshape`.
- **Python `Decoder(dict)` constructor now accepts schema v2 metadata.**
  `PyDecoder::new` previously routed every dict through the legacy
  `ConfigOutputs` deserialiser, producing the misleading *"invalid type:
  map, expected tuple struct QuantTuple"* error when given v2 documents
  produced by `tflite-converter` ≥ v0.3.0. A smart constructor now
  discriminates on the authoritative `schema_version` field: `>= 2`
  routes to `DecoderBuilder::with_schema(SchemaV2)`, which supports
  object-form quantization, per-channel quantization, split logical
  outputs, and the full v2 type vocabulary. Legacy documents (no
  `schema_version`, or `schema_version: 1`) continue through the
  unchanged legacy path. The string constructors
  (`new_from_json_str` / `new_from_yaml_str`) already went through the
  v2 path; only the dict constructor was broken. Reference: EDGEAI-1081.

- **Schema v2 `shape` no longer truncates to `[]` when `dshape` is
  omitted.** `squeeze_padding_dims` zipped `shape` against a `keep`
  mask derived from `dshape`. With `dshape` defaulting to empty
  (its `#[serde(default)]` representation for logical outputs that
  omit named dims), `zip` returned an empty iterator and silently
  produced `shape = []`, which then failed `verify_yolo_split_*`
  rank checks with the misleading `Invalid Yolo Split Boxes shape
  []`. The bug only surfaced via `Decoder({...})` and the JSON/YAML
  constructors that lower v2 → legacy; `Decoder.new_from_outputs`
  bypassed the lowering and worked. Two regression tests anchor the
  helper directly and the integration path through
  `to_legacy_config_outputs`.

- **`materialize_segmentations` / `materialize_scaled_segmentations`
  now return `Error::InvalidShape` when `detect.len() !=
  mask_coefficients.len()`,** replacing the fused path's silent
  `zip` truncation and pre-empting the `scratch.logits(N, HW)`
  slice panic in the batched path.

- **`copy_packed_to_padded_dma` validates `src` / `dst` width /
  height / format up front and uses `saturating_mul` consistently
  in error format strings.** The bounds check itself was already
  saturating, but the error path re-multiplied with `*` and
  panicked in debug builds on overflow, turning an intended `Err`
  return into a panic during error reporting.

- **`padded_dma_pitch_for(memory: None)` now respects
  `is_dma_available()`.** Previously the helper treated `None` as
  "caller wants DMA" and routed `load_jpeg` / `load_png` /
  `rotate_flip_to_dyn` into the pitch-padded DMA path
  unconditionally. On Linux systems where DMA isn't available
  (sandboxed CI, missing `/dev/dma_heap`, permission-denied
  containers), this forced `image_with_stride(..., None)` to fail
  allocating DMA instead of falling back to SHM/Mem like
  `image(..., None)` does.

- **`draw_proto_masks_impl` no longer silently returns `Ok(())`
  on shape mismatch.** Rank or `num_protos` mismatch on
  `mask_coefficients` now surfaces `Error::InvalidShape`; only
  the genuine "no detections this frame" case
  (`coeff_shape[0] == 0`) short-circuits to `Ok(())`. Matches the
  GL renderer's behaviour.

- **`PyDecoder::new` schema_version parsing surfaces overflow,
  negatives, and non-numeric values as a Python `RuntimeError:
  invalid schema_version: ...`** instead of silently truncating
  via `as_u64()` + `as u32` or being mapped to `None`.
  `serde_json::from_value::<u32>` does the typed parse directly.

- **Ara-2 CHW dimension ordering in v2 schema merge program.** The
  Ara-2 `tensor_filter` reports physical-tensor dimensions in native
  CHW order rather than NNStreamer's innermost-first convention,
  which caused exact-shape matching in the merge program to miss
  matches and surface as `EIO` errors on v2 schema models with
  protos. Two fixes: (1) `decode_proto()` now also runs the merge
  program when one is compiled (previously only `decode()` did);
  (2) `find_and_dequantize` falls back to element-count matching
  with axis-permutation detection (`find_axis_permutation`) when
  exact-shape lookup fails, transposing the matched tensor into
  the expected layout before dequantization. `execute_merge`,
  `execute_channel_concat`, and `execute_per_scale` use the
  permutation-aware lookup. All 286 existing decoder tests still
  pass; new unit tests cover identity / NCHW↔NHWC / stripped-unit-
  dim / already-used-tensor / no-match cases.

### Not in this release

- GL shader-side per-channel dequantization (CPU handles it; GL
  returns `NotSupported`). Deferred pending a future ticket.
- Continuous perf validation in CI. No FP16-capable CI runner is
  available; local benchmarks are documented in `TESTING.md`.
  Tracked as EDGEAI-1247.
- Runtime CPU-feature detection + dynamic dispatch for richer ISAs
  (aarch64 FEAT_FP16 on non-Apple, x86_64 F16C / FMA / AVX2). The
  default build stays on each target's baseline ISA so a single
  distributed binary remains portable; local benchmarking uses
  `RUSTFLAGS` opt-in per `README.md` / `TESTING.md`. Dynamic dispatch
  is the prerequisite for enabling these ISAs on shipped binaries.

## [0.17.0] - 2026-04-21

### Added

- **`MaskResolution` enum and `resolution` parameter on
  `ImageProcessor.materialize_masks`.** When set to
  `MaskResolution.Scaled(width, height)`, HAL upsamples the full
  proto plane once (correct edge-clamp bilinear) and returns
  per-detection binary `uint8 {0, 255}` masks at the target
  resolution — interchangeable with the existing continuous-sigmoid
  Proto masks via the same `> 127` threshold. Letterbox-aware: when
  `letterbox` is `Some`, `(width, height)` are interpreted as
  original-content pixel dims and the inverse letterbox transform is
  applied during upsample. Resolves the YOLOv8-seg mAP collapse
  observed on Hailo+HAL when per-tile `ImageProcessor.convert()`
  upsampling leaks edge activations across tile boundaries; callers
  currently relying on `decode() + ImageProcessor.convert()` per-tile
  resize should migrate to
  `decode_proto() + materialize_masks(MaskResolution.Scaled)`.
  Reference: EDGEAI-759.

- **HailoRT split-output DFL decode support.** The schema-v2 `boxes`
  logical output with `encoding: dfl` and per-scale physical children
  is now accepted end-to-end. At compile time the decoder extracts
  `reg_max` from the first child (validated uniform across children)
  and pre-computes per-FPN-level anchor grids with the Ultralytics
  `+0.5` centre offset; at decode time the per-scale merge runs a
  numerically stable softmax + weighted-sum + `dist2bbox` per child,
  collapsing the `4 × reg_max` feature axis to 4 xcycwh pixel-
  coordinate channels before concat. The merged `(1, 4, total_anchors)`
  tensor flows through the existing Ultralytics split-decoder
  unchanged. Reference: `HAILORT_DECODER.md`, EDGEAI-759.

- **DFL primitives module (`edgefirst_decoder::decoder::dfl`).**
  Pure-function building blocks — `make_anchor_grid`, `dfl_bins`,
  `softmax_inplace`, `decode_dfl_level` — operating on `&[f32]` so
  they're trivially unit-testable and SIMD-ready. Seven unit tests
  mirror validator-side pytest vectors (uniform/concentrated DFL
  distributions, row-major anchor ordering, large-logit numerical
  stability).

- **Hailo YOLOv8-seg reference fixture
  (`testdata/hailo_yolov8seg_edgefirst.json`).** Canonical schema-v2
  example with three FPN scales (strides 8/16/32), DFL-encoded boxes,
  sigmoid-pre-applied per-class scores, per-scale mask coefficients,
  and NHWC protos. Used by end-to-end numerical-parity tests that
  mirror `scripts/decode_hailo_split.py::test_synthetic` from
  `edgefirst-validator @ feature/DE-823-hailort`.

- **Schema v2 metadata parser (`edgefirst_decoder::schema`).** New
  data-only module implementing the two-layer logical/physical output
  model from the updated `edgefirst.json` specification. Logical outputs
  describe semantics (`encoding`, `score_format`, `normalized`,
  `decoder`); physical children describe quantization mechanics
  (`dtype`, per-channel `quantization`, `stride`, `scale_index`,
  `activation_applied` / `activation_required`). At most one level of
  nesting permitted. Supports all converter decompositions in the spec
  (flat TFLite, ARA-2 `boxes_xy`/`boxes_wh`, Hailo per-scale per-FPN).

- **Parse-time v1 compatibility shim.** `SchemaV2::parse_json` and
  `SchemaV2::parse_yaml` auto-detect legacy metadata (missing
  `schema_version` or `schema_version: 1`) and convert to v2 in memory,
  so the internal decoder only ever sees v2. Files declaring a
  `schema_version` higher than the supported maximum are rejected with
  `DecoderError::NotSupported` instead of being silently parsed against
  the wrong grammar. Rust callers can invoke `SchemaV2::from_v1`
  directly when they already hold a deserialized v1 `ConfigOutputs`.

- **Per-channel quantization support.** `schema::Quantization`
  represents both per-tensor (scalar `scale`) and per-channel (array
  `scale` with `axis`) quantization, matching the ONNX / TFLite
  conventions. Deserializer accepts either scalar or array inline.

- **Explicit `DType`, `Activation`, `BoxEncoding`, `ScoreFormat`
  enums.** Replaces v1's inference-by-shape with explicit metadata so
  that future schema changes (heterogeneous `reg_max`, per-level DFL
  bins, new activations) do not require decoder dispatch changes.

- **`DecoderBuilder::with_schema`.** New builder entry point that
  accepts a `schema::SchemaV2` (parsed from file / JSON / YAML, or
  constructed programmatically) and compiles it into a decoder.
  Validates the schema, derives a `DecodeProgram` for any logical
  outputs with physical children, and downconverts the logical-level
  metadata to the legacy dispatch representation. Flat v2 schemas run
  through the existing decode kernels unchanged; schemas with
  `encoding: dfl` combined with per-scale children are decoded by the
  merge path (see the HailoRT DFL entry above). Flat DFL (no
  children) remains unsupported — the HAL's DFL kernel only runs
  inside the per-scale merge.

- **Physical → logical merge path at decode time.** `Decoder::decode`
  now executes the compiled `DecodeProgram` when present,
  dequantizing physical children with per-tensor `(scale, zp)` and
  producing one `ArrayD<f32>` per logical output. Two merge
  strategies implemented:
  - `ChannelConcat` for channel sub-splits (ARA-2 `boxes_xy` /
    `boxes_wh`): concat along the logical channel axis.
  - `PerScale` for FPN per-scale splits: sort children stride-
    ascending, permute each from its declared layout (NHWC or NCHW
    per its own `dshape`) to canonical `(batch, features, H×W)`,
    concat along the anchor axis, reshape/transpose into the
    logical output shape.

  Duplicate-shape sibling children (the ARA-2 case where `boxes_xy`
  and `boxes_wh` share `[1, 2, 8400]`) are bound in declared child
  order via a used-index list — callers pass tensors positionally in
  schema traversal order until name-keyed decode lands.

- **Schema-driven decoding rejects unsupported features early.**
  `NotSupported`/`InvalidConfig` errors flagged at build time for:
  flat DFL-encoded boxes (no children — the HAL's DFL kernel only
  runs inside the per-scale merge), heterogeneous `reg_max` across
  FPN children, per-channel quant on a split child, missing
  `dshape` on per-scale children, mixed per-scale + channel sub-
  split decompositions on the same logical output.

- **ARA-2 DVM `padding` dim support.** Schema-v2 metadata emitted by
  the ARA-2 converter declares a trailing `padding: 1` axis on most
  logical outputs (e.g. `[1, 4, 8400, 1]` for boxes, `[1, 80, 8400,
  1]` for scores). Both the legacy-config downconversion and the
  merge-program execution now squeeze these axes deterministically
  from `dshape`, so the legacy rank-3 dispatch sees the shapes it
  expects. Verified against real `model.dvm` metadata from
  `edgefirst-studio-ara2` INT8 and INT16 builds (round-trip parse,
  validate, build, decode-smoke all pass).

- **`DecoderBuilder::build` auto-detects schema v2 JSON / YAML.**
  `with_config_json_str` and `with_config_yaml_str` (and therefore the
  Python `Decoder.new_from_json_str` / `new_from_yaml_str` static
  constructors) now route through `SchemaV2::parse_json` /
  `parse_yaml`, so v2 metadata reaches the native `DecodeProgram` merge
  path without a Python-side flattening pass. v1 metadata still builds
  unchanged via the in-memory `from_v1` shim. Validators that previously
  unpacked v2 `boxes_xy` / `boxes_wh` and re-quantized to a shared
  scale on every frame can drop that work and pass the raw
  `edgefirst.json` directly to the HAL.

### Performance

- **`MAX_NMS_CANDIDATES` pre-NMS top-K cap (Ultralytics-aligned, 30 000).**
  At very low score thresholds (e.g. `t=0.01` on YOLOv8 with
  8400 anchors × 80 classes), nearly the full 672 000-entry score grid
  used to feed O(n²) NMS and a per-survivor 32×25 600 mask matmul,
  producing minutes-per-frame decode times on dense images. Both the
  float and quantized split-segdet paths now sort the above-threshold
  candidate list by score and truncate to 30 000 before NMS. Default
  thresholds (≥0.25) almost never reach the cap, so this is a
  silent worst-case improvement, not a behaviour change.

### Fixed

- **`Tensor.from_numpy` panic on pitch-aligned destinations
  (`STRIDES_BUG`).** When `ImageProcessor.create_image` allocates a
  DMA-BUF or PBO buffer whose row stride is rounded up to a 64-byte
  GPU alignment boundary, the mapped backing slice is longer than
  `Image.size` (the logical `width × height × bpp`). The old
  `from_numpy` did a single `copy_from_slice` from a numpy array
  sized off `Image.size` and panicked on the length mismatch — which
  triggered every time `edgefirst-validator` resized YOLOv8-seg
  per-instance masks (mask widths come from box coordinates, almost
  never 64-byte-aligned). `from_numpy` now detects pitch-aligned
  destinations via `Tensor.row_stride`, copies row-by-row into the
  padded buffer, and only raises on genuine shape/dtype mismatches.

### Added

- **`Tensor.row_stride` Python property.** Reports the actual row
  pitch in bytes for image tensors. For images allocated via
  `create_image`, this reflects any DMA pitch-alignment padding
  applied to the row. Lets downstream code distinguish padded from
  unpadded allocations without guessing.

## [0.16.4] - 2026-04-17

### Fixed

- **`draw_decoded_masks` / `draw_proto_masks` now always produce the full
  output frame.** Previously these functions returned `Ok(())` without
  touching `dst` when the detection list was empty, silently relying on
  the caller having pre-cleared the destination — an invariant that is
  nowhere documented and false for recycled buffers in triple-buffered
  display pipelines. The new contract writes `dst` in every case:

  | detections | background | output                               |
  |------------|------------|--------------------------------------|
  | none       | none       | dst cleared to `0x00000000`          |
  | none       | set        | dst ← background                     |
  | set        | none       | masks drawn over cleared dst         |
  | set        | set        | masks drawn over background          |

- **Backend-native base-layer rendering.** Each backend now owns base
  layer production with its appropriate hardware primitive:

  - G2D: `g2d_clear` for empty-frame no-bg, `g2d_blit` for empty-frame
    with bg. The prior release returned `NotImplemented` for every G2D
    call, breaking forced-G2D entirely. G2D still returns
    `NotImplemented` when detections/segmentations are present (no
    rasterizer).
  - OpenGL: `glClear(0)` when no bg; GPU DMA-BUF EGLImage blit when bg
    is DMA-backed; direct `glTexImage2D` upload of bg into the render
    texture when bg is non-DMA memory (previously the non-DMA path
    memcpy'd bg into `dst`, which was then overwritten by the
    subsequent framebuffer readback — a pre-existing latent bug).
  - CPU: buffer fill / memcpy as the terminal fallback.

- **Removed CPU `copy_from_slice` from the accelerated dispatch paths.**
  The old `MaskOverlay::apply_background` helper memcpy'd the background
  through a mapped DMA buffer before delegating to GL/CPU, defeating the
  zero-copy architecture and forcing cache-maintenance ops on both
  surfaces every zero-detection frame. The helper is deleted; GL and G2D
  handle bg with hardware primitives, CPU handles it as a terminal
  fallback only.

- **Auto dispatch prefers G2D for empty frames.** A zero-detection frame
  on a platform with G2D available now completes in a single `g2d_clear`
  or `g2d_blit` op, avoiding GL renderbuffer setup + readback each frame
  in a triple-buffered display loop.

### Tests

- Added a 4-scenario × 4-backend (auto, cpu, opengl, g2d) pixel-verified
  test matrix. Every test pre-fills `dst` with a distinctive dirty
  pattern so any lingering "caller must clear" assumption fails loudly.
  Full suite passes on imx8mp-frdm (Vivante GC7000UL + G2D) and
  imx95-frdm (Mali-G310 + G2D via DPU).

- Updated `test_segmentation_yolo` (CPU + GL) to use
  `MaskOverlay::with_background` instead of pre-loading the camera frame
  into `dst`, matching the new output contract and the pattern used by
  the C API / Python callers.

## [0.16.3] - 2026-04-13

### Fixed

- **Mali Valhall DMA-BUF pitch alignment (i.MX 95).** `eglCreateImageKHR`
  on Mali Valhall (e.g. Mali-G310 on i.MX 95) rejects every DMA-BUF whose
  row pitch is not a multiple of 64 bytes, returning `EGL_BAD_ALLOC`.
  The HAL was gracefully falling through to the non-DMA CPU readback
  path, which is 10–20× slower and indistinguishable from "DMA worked
  but rendered nothing." This affected `draw_decoded_masks` and
  `draw_proto_masks` on i.MX 95 for any canvas whose
  `width × bytes_per_pixel` was not 64-aligned (e.g. crowd-scene canvases
  at 3004×1688 RGBA8 → pitch 12016, 16-aligned, fail). Convert paths
  were unaffected because their source buffers come from
  libcamera/v4l2, which already align to the GPU's preferred pitch.

  `ImageProcessor::create_image` now silently pads the **row stride**
  (not the logical width) so the resulting DMA-BUF satisfies
  `GPU_DMA_BUF_PITCH_ALIGNMENT_BYTES` (64). Following the V4L2 / DRM
  / GStreamer convention, the tensor's logical `width()` / `height()`
  continue to report the user-requested values and the padding is
  carried by `row_stride()` / `effective_row_stride()`. Callers that
  iterate pixel rows must use the stride (not `width × bpp`) for row
  offsets; the CPU mapping spans the full `stride × height` bytes.
  Verified on i.MX 95: crowd canvas (3004×1688 RGBA8) draw drops from
  91 ms → 9.26 ms (~10× faster), with `tensor.width() == 3004` and
  `tensor.effective_row_stride() == 12032`.

  New tensor primitives power this:
  - `Tensor::image_with_stride(width, height, format, row_stride, memory)`
    and `TensorDyn::image_with_stride(...)` allocate a DMA-backed
    image with an explicit row stride that may exceed the natural
    `width × channels × sizeof(T)` pitch. Currently DMA-only and
    packed-formats-only.
  - `DmaTensor::new_with_byte_size` decouples the allocation byte
    count from `shape.product()` so the backing DMA-BUF can be sized
    to `row_stride × height` bytes while the logical shape remains
    `[height, width, channels]`.
  - `DmaMap::new_with_byte_size` exposes the full padded mmap via
    `as_slice()` / `as_mut_slice()` so CPU iteration respects the
    stride without going past the end of the returned slice.
  - `Tensor::map()` now allows CPU mapping of self-allocated strided
    DMA tensors (previously rejected). Foreign strided imports
    (`from_fd` + `set_row_stride`, the V4L2/GStreamer case) continue
    to be GPU-only as before — the external allocator owns the layout
    and HAL cannot validate CPU access expectations.

### Performance

- **`render_yolo_segmentation` per-instance overhead.** The hot inner
  loop of `draw_decoded_masks` paid for four `glTexParameteri` calls,
  a `Vec<u8>` heap allocation + 1 px CPU zero-pad copy, and a
  `glFinish` after every draw. All four are now eliminated or hoisted
  out of the per-instance loop:
  - `setup_yolo_segmentation_pass` runs once before the batch and sets
    program/texture/parameters.
  - The CPU pad is gone; texel-centre UVs (`0.5/w` to `(w-0.5)/w`)
    keep bilinear sampling strictly inside the uploaded mask region.
    The fragment shader's existing `smoothstep(0.5, 0.65)` provides
    the edge antialiasing the pad used to proxy for.
  - Internal format switched from unsized `GL_RED` to sized `GL_R8`.
  - `glFinish` per instance is now branched on `is_vivante`. Vivante's
    immediate-mode driver regresses ~2× without the periodic drain;
    Mali Valhall's TBDR pipeline regresses ~30% *with* it (every
    flush forces a tile store-unload of the framebuffer). The split
    lets each driver win.

  Combined with the alignment fix above, the Kinara `yolov8.rs` example
  on imx95 with `crowd.png` (39 detections) drops Draw stage from
  135.89 ms → ~10 ms end-to-end. imx8mp Vivante crowd Draw is unchanged
  at ~38 ms in this release; further improvement requires the Tier 2-a
  mask pool work (separate effort).

### Added

- **One-time slow-path warning.** `draw_decoded_masks` and
  `draw_proto_masks` now emit a single warn-level log the first time
  they fall back from the DMA fast path to the CPU readback path,
  identifying the call site, the failing setup function, and the
  underlying error. The message specifically calls out Mali's 64-byte
  pitch requirement so the next regression has a direct pointer at the
  cause. Subsequent fallbacks are demoted to debug-level.

- **Trace-level draw-path diagnostics.** Both draw entry points log at
  `log::trace!` level when entering, and report which path was taken
  (DMA fast path / PBO / non-DMA fallback). Gated on
  `log::log_enabled!(Trace)` so the formatting cost is a single
  integer compare when trace logging is disabled.

- **GPU pitch alignment helpers exposed to all language bindings** so
  callers that allocate their own DMA-BUFs (GStreamer plugins, V4L2 ring
  buffers, custom video pipelines) can size them to satisfy Mali's
  64-byte requirement without having to hard-code the constant. The
  helpers all delegate to a single overflow-safe Rust implementation
  that returns the original width unchanged (with a warn-level log) if
  the rounded value would overflow `usize`, so callers can rely on the
  returned width being **at least** the requested width.

  | Language | Symbols |
  | --- | --- |
  | Rust | `edgefirst_image::align_width_for_gpu_pitch`, `edgefirst_image::primary_plane_bpp`, `edgefirst_image::GPU_DMA_BUF_PITCH_ALIGNMENT_BYTES` |
  | C | `hal_align_width_for_gpu_pitch`, `hal_align_width_for_pixel_format`, `hal_gpu_dma_buf_pitch_alignment_bytes` |
  | Python | `edgefirst_hal.align_width_for_gpu_pitch`, `edgefirst_hal.align_width_for_pixel_format`, `edgefirst_hal.gpu_dma_buf_pitch_alignment_bytes` |

  The `*_for_pixel_format` variants in C and Python derive bytes-per-pixel
  from a `HalPixelFormat`/`PyPixelFormat` + dtype so callers don't have
  to remember per-format BPPs. Unit tests cover RGBA8/BGRA8, RGB888,
  Grey/u8, NV12 luma, zero-input edge cases, and overflow-extreme widths
  (up to `usize::MAX`).

- **`mask_benchmark` diagnostic mode.** `MASK_BENCH_DIAG=1` bypasses the
  timing loop and runs a single `draw_decoded_masks` + `draw_proto_masks`
  pair against a real DMA-allocated destination, dumps the rendered
  bytes to `/tmp/mask_decoded.rgba` / `/tmp/mask_proto.rgba`, and reports
  call latency. `MASK_BENCH_DST_W` and `MASK_BENCH_DST_H` env vars
  override the canvas dimensions for reproducing arbitrary caller
  sizing.

- **`bench_mask_pool` POC bench in `gpu-probe`.** Standalone GL-only
  proof-of-concept that compares the per-instance baseline against an
  instanced `GL_TEXTURE_2D_ARRAY` + `glDrawElementsInstanced` path at
  N ∈ {2, 5, 10, 20, 40, 80}. Includes pixel-level cross-validation and
  a separate `readback_640x640_rgba8` measurement. Used for the Tier 2-a
  mask pool design validation.

## [0.16.2] - 2026-04-10

### Fixed

- **C API release packaging — proper SOVERSION symlink chain.** The release
  pipeline now ships the C shared library as a standard GNU/Linux versioned
  symlink chain:

  ```
  libedgefirst_hal.so          → libedgefirst_hal.so.0
  libedgefirst_hal.so.0        → libedgefirst_hal.so.0.16    (name matches DT_SONAME)
  libedgefirst_hal.so.0.16     → libedgefirst_hal.so.0.16.2
  libedgefirst_hal.so.0.16.2   (real file)
  ```

  Previously the release archive contained flat symlinks that all pointed
  directly at the realname file instead of chaining through the MAJOR and
  MAJOR.MINOR levels. `DT_SONAME` inside the ELF remains
  `libedgefirst_hal.so.0` (major only), matching convention used by glibc,
  OpenSSL, zlib, etc., so the dynamic linker resolves dependents through the
  chain at runtime. The SOVERSION rationale is now documented in
  `crates/capi/build.rs`.

## [0.16.1] - 2026-04-08

### Added

- **NV12 DMA-BUF plane configuration tests** — comprehensive test coverage
  for all three NV12 EGL import scenarios: true multiplane (separate
  DMA-BUFs, libcamera style), same-fd multiplane (dup'd fds with offset,
  V4L2/GStreamer style), and contiguous single-fd (UV computed from luma
  geometry). Unit tests verify `DmaImportAttrs` plane resolution and EGL
  attribute serialization; integration tests exercise the full
  `import_image` → `eglCreateImage` → `convert()` render path for each
  scenario and compare pixel output against a contiguous NV12 reference.
  Validated on Mali G310 (i.MX 95), Vivante GC7000UL (i.MX 8MP), and
  V3D 7.1 (RPi5).

### Changed

- **Extracted `DmaImportAttrs` from EGL import path** — the plane
  resolution and EGL attribute-building logic in `create_image_from_dma2`
  is now a standalone `DmaImportAttrs` struct with `from_tensor()` and
  `to_egl_attribs()` methods. This separates the testable attribute
  construction from the actual `eglCreateImage` call. No public API
  change; the struct is module-internal (`pub(super)`).

## [0.16.0] - 2026-04-02

### Changed

- **Test parametrization overhaul (EDGEAI-1219)** — consolidated
  repetitive test code via macros and parametrize decorators while
  expanding coverage:
  - Decoder: 16 tracked segdet tests collapsed into two macro families
    (`real_data_tracked_test!`, `e2e_tracked_test!`), ~950 line reduction.
  - Python: 10 dtype tests replaced with single `@pytest.mark.parametrize`
    `test_dtype` function, ~250 line reduction.
  - C API tests integrated into `make test` via new `test-capi` target.

### Added

- **`Tensor.from_numpy()` method** — type-safe copy from any numpy array
  into a HAL tensor. Accepts all numeric dtypes (uint8 through float64
  and float16). Supports contiguous, strided, and negative-stride arrays
  with three optimized copy paths:
  1. Fully contiguous → `copy_from_slice` (memcpy).
  2. Strided outer with contiguous inner rows → row-by-row memcpy with
     O(n_rows) stride-computed offsets (no element iteration).
  3. Fully strided → per-element copy via ndarray iterator.
  All paths release the GIL and parallelize via rayon above 256 KiB.

- **Tracker benchmark** — `crates/tracker/benches/tracker_benchmark.rs`
  measuring ByteTrack update latency (1/10/50/100 detections),
  100-frame warm-state scenario, and isolated Kalman predict/update.

- **import\_image benchmark** — DMA-BUF import at 1080p/4K resolutions
  for RGBA and NV12 formats. Skips gracefully on non-DMA platforms.

- **PlaneDescriptor stride/offset tests** — import\_image with padded
  stride, aligned stride, NV12 multiplane with offset, stride-too-small
  error case. All DMA-gated.

- **Python mask rendering tests** — `draw_decoded_masks` with empty
  input, multiple boxes, `ColorMode.Instance`, opacity scaling, and
  `draw_masks` fused decode+render path.

- **Error Display tests** — `TensorError`, `ImageError`, `DecoderError`
  Display formatting verified for all constructible variants.

- **Tracker test expansion** — 14 → 38 unit tests. New coverage for
  ByteTrack two-stage matching, builder method effects, degenerate
  inputs, 100-detection scaling. Kalman filter prediction accuracy,
  numerical stability (1000 cycles), gating distance edge cases.

- **MemTensor and ShmTensor unit tests** — both had zero tests. Added
  new/map/reshape/error-path tests for MemTensor and
  new/map/from_fd-roundtrip/reshape tests for ShmTensor.

- **VYUY and NV16 format conversion tests** — added VYUY→Grey,
  VYUY→PlanarRgb, VYUY→NV16, NV16→RGB, NV16→RGBA via existing
  `generate_conversion_tests!` macro.

- **Dequantization ground truth test** — `test_dequant_ground_truth`
  verifies `dequantize_cpu` and `dequantize_cpu_chunked` against
  hand-computed expected values for four (scale, zero_point) combinations
  including i8 boundary values.

### Fixed

- **Python `Tensor.from_fd()` double-close bug** — `from_fd` used
  `OwnedFd::from_raw_fd()` which took ownership of the caller's fd. If
  Python code called `os.close(fd)` afterward (the natural pattern), it
  triggered a double-close; if the fd number was reused before the tensor
  was dropped, the tensor would close an unrelated fd. Now dups the fd at
  the FFI boundary via `BorrowedFd::borrow_raw()` +
  `try_clone_to_owned()`, matching the contract already used by
  `import_image` (Python) and `hal_tensor_from_fd` (C API). Updated
  `.pyi` stub to document the caller-retains-ownership contract.

- **Unstable zero-copy perf assertions** — `test_dma_zero_copy_perf`
  and `test_shm_zero_copy_perf` increased to 50 iterations with 1ms
  noise floor and 1.5× margin to prevent flaky failures on fast
  hardware.

- **Python `.pyi` stub missing from PyPI wheels** — the type stub file
  was excluded from wheels built with `maturin --sdist` because the
  sdist did not include it. Added explicit `[tool.maturin] include` for
  the `.pyi` file. Also fixed a README conflict in the sdist by
  overriding the workspace readme with the crate-local one.

- **fd leak in `from_fd` test** — `tensor_from_fd_func` now closes the
  dup returned by `.fd` after passing it to `from_fd()`, matching the
  caller-retains-ownership contract.

### Removed

- **`Tensor.copy_from_numpy()`** — removed in favor of `from_numpy()`,
  which is a strict superset supporting all numeric dtypes, arbitrary
  tensor shapes, strided arrays, and GIL-released parallel copies.

## [0.15.2] - 2026-04-01

### Fixed

- **Vivante galcore kernel deadlock on multi-thread EGL init** — creating
  multiple `ImageProcessor` instances with the OpenGL backend on separate
  threads caused a kernel-level deadlock on Vivante GC7000UL (i.MX8M Plus).
  Each `GlContext` opened a fresh DRM fd and created an independent EGL
  display; galcore deadlocked when a second DRM fd was opened while the
  first display was active. Fix: share a single EGL display (and its
  backing GBM device / DRM fd) across all `GlContext` instances via a
  process-wide `OnceLock<SharedEglDisplay>` singleton. The display is
  intentionally never terminated (same leak pattern as `EGL_LIB` and
  `SHARED_DRM_FD`). `GL_MUTEX` is retained for serializing GL operations.

- **`probe_egl_displays()` galcore safety** — when a shared EGL display
  already exists, `probe_egl_displays()` now returns only the cached
  display kind instead of opening additional DRM fds (which would trigger
  the galcore deadlock).

### Changed

- `GlContext` no longer owns the EGL display — the `display` field changed
  from `EglDisplayType` (owned enum with GBM device) to `egl::Display`
  (copyable handle from the shared singleton). `GlContext::drop` no longer
  calls `eglTerminate`.

### Added

- `test_probe_then_create_gl_context` — validates that `probe_egl_displays()`
  populates the shared display and subsequent `GLProcessorThreaded::new()`
  reuses it.

- Un-ignored `test_opengl_10_threads` — the shared display fix eliminates
  the Vivante galcore deadlock that caused the hang.

## [0.15.1] - 2026-03-31

### Fixed

- **Multi-thread EGL/GL deadlock** — creating multiple `ImageProcessor`
  instances on separate threads caused SIGSEGV and deadlocks on Vivante
  `galcore` (i.MX8M Plus) and `EGL(NotInitialized)` errors on Broadcom V3D
  (Raspberry Pi 5). Root cause: these GPU drivers are not thread-safe for
  concurrent EGL/GL operations even with independent displays and contexts.
  Added a global `GL_MUTEX` that serializes all OpenGL initialization,
  command dispatch, and teardown across `GLProcessorST` instances. Mali-G310
  (i.MX95) was unaffected but benefits from the safety guarantee.

### Added

- Three multi-thread GPU integration tests:
  `test_multiple_image_processors_same_thread`,
  `test_multiple_image_processors_separate_threads`, and
  `test_image_processors_concurrent_operations` (barrier-synchronized
  concurrent resize across 4 threads). Validated on Vivante GC7000UL,
  Mali-G310, and Broadcom V3D 7.1.10.2.

- `GL_MUTEX` documentation in ARCHITECTURE.md covering problem, solution,
  per-driver behavior, and performance implications.

## [0.15.0] - 2026-03-30

### Added

- **Three-stage segmentation pipeline API** — new `materialize_masks()` on
  `ImageProcessor` exposes CPU mask materialization as a first-class operation,
  enabling `decode_proto` → `materialize_masks` → `draw_decoded_masks` with
  user access to intermediate masks for analytics, IoU computation, or export.
  Mask values are continuous sigmoid confidence (u8 0-255), not binary
  thresholded. Exposed in Rust, Python (`ProtoData`, `ColorMode`,
  `Decoder.decode_proto`, `ImageProcessor.materialize_masks`), and C
  (`hal_decoder_decode_proto`, `hal_proto_data_free`,
  `hal_image_processor_materialize_masks`, `hal_color_mode` enum) APIs.

- **`ColorMode` enum** — controls whether mask colors are assigned by class
  label (`Class`, default), detection index (`Instance`), or track ID
  (`Track`). Added as parameter to `draw_decoded_masks`, `draw_masks`, and
  `draw_masks_tracked` across all API layers.

- **`letterbox` parameter** on `draw_decoded_masks`, `draw_masks`, and
  `draw_masks_tracked` — enables letterbox coordinate unmapping directly in
  the rendering step. Propagated through the GL threaded message channel.

- **Jetson Orin Nano platform** — benchmark data collected and added to
  BENCHMARKS.md; `benchmark_common.py` updated with platform configs.

### Changed

- **Per-texture EGL binding state tracking** — moved EGL binding cache keys
  from three processor-level fields (`last_bound_dst_egl`,
  `last_bound_draw_dst_egl`, `last_bound_src_egl`) onto the `Texture` struct
  itself (`bound_egl_key`). Eliminates cross-path cache invalidation bugs,
  reduces maintenance from 6 manual invalidation sites to 2 helper methods
  (`invalidate_dst_textures`, `invalidate_src_textures`). Fixed latent bug
  where `setup_renderbuffer_non_dma` / `setup_renderbuffer_from_pbo` could
  skip necessary `EGLImageTargetTexture2DOES` calls after non-DMA → DMA
  transition.

- **`EglImageCache::sweep()` returns `bool`**, `evict_lru()` returns `bool`
  and simplified (removed redundant nested `if let`).

- **C API breaking changes** — `hal_image_processor_draw_decoded_masks`,
  `hal_image_processor_draw_masks`, and `hal_image_processor_draw_masks_tracked`
  gain `letterbox` and `color_mode` parameters. Callers must update call sites.

- **BENCHMARKS.md v3.0** — refreshed all benchmarks across 5 platforms with
  v0.15.0 code; hybrid path 1.4-14.2× faster than fused GPU on all platforms.

### Fixed

- **`decode_proto` for detection-only models** — `decode_quantized_proto`,
  `decode_float_proto`, and both tracked variants now properly decode
  detection boxes instead of returning empty `output_boxes` with `Ok(None)`.

- **`Segmentation` doc comment** — corrected from "binary per-instance mask"
  to "continuous sigmoid confidence values quantized to u8".

- **`GLProcessorThreaded` letterbox propagation** — `DrawDecodedMasks` and
  `DrawProtoMasks` message variants now carry the `letterbox` field instead
  of hardcoding `None`.

- **`test_disable_env_var` race condition** — saves/restores
  `EDGEFIRST_FORCE_BACKEND` to prevent interference with parallel tests.

- **C API `materialize_masks` error mapping** — `NoConverter` maps to
  `ENOTSUP` instead of `EIO`.

## [0.14.1] - 2026-03-29

### Added

- **On-target Neutron DMA-BUF inference regression test** (`test_neutron_dmabuf_infer`) —
  new C test that exercises the full HAL write → `hal_invoke()` → output
  validation path over a live Neutron DMA-BUF, with a corresponding
  `make test_neutron_dmabuf_infer` target. Requires
  `NEUTRON_ENABLE_ZERO_COPY=1` and a real delegate + model on target.

- **ARCHITECTURE.md Appendix C: DMA-BUF Identity and Tensor Caching** —
  documents why DMA-BUF fd numbers are unreliable as cache keys (fd numbers
  recycle across buffer pool cycles), `fstat(fd).st_ino` as the correct
  stable identity, `(inode, offset)` keying for multi-plane buffers,
  cache warm-up / steady-state behaviour table (i.MX 95 figures), and
  the reference implementation pattern from `edgefirstcameraadaptor`.
  Bumps ARCHITECTURE.md version 3.0 → 3.1.

### Fixed

- **`DmaTensor::from_fd()` skips DRM attachment for foreign (imported) fds** —
  previously every `from_fd()` call attempted `DRM_IOCTL_PRIME_FD_TO_HANDLE`
  on the imported fd. For fds exported by other kernel drivers (e.g. the
  Neutron NPU), this ioctl fails because the fd is not owned by the DRM
  device; the resulting error was silently ignored but added unnecessary ioctl
  overhead on every import. Foreign fd cache coherency is the responsibility
  of the exporting kernel driver. (`crates/tensor/src/dma.rs`)

- **`DmaTensor::try_clone()` preserves imported/owned distinction** —
  cloning an imported `DmaTensor` now consistently produces another imported
  tensor (no DRM attachment), matching the semantics of `from_fd()`.

## [0.14.0] - 2026-03-28

### Added

- **`HalCameraAdaptorFormatInfo`** (EDGEAI-1189) — new struct added to the
  camera adaptor delegate ABI that describes a camera adaptor's channel
  mapping and V4L2 FourCC code: `input_channels` (`int`),
  `output_channels` (`int`), and `fourcc` (`char[8]`, NUL-terminated V4L2
  FourCC string, at most 4 bytes + NUL). Populated by the delegate's
  `hal_camera_adaptor_get_format_info()` callback.

- **YOLO-SEG 2-way split decoder** (`ModelType::YoloSegDet2Way`) — supports
  TFLite INT8 segmentation models with 3 outputs: a combined detection tensor
  `[1, nc+4, N]` (boxes + scores), a separate mask-coefficient tensor
  `[1, 32, N]`, and a prototype-mask tensor `[1, H/4, W/4, 32]`.  The
  builder auto-selects this variant when `Detection` + `MaskCoefficients` +
  `Protos` outputs are provided.  Supported in all decode paths (float,
  quantized, and proto variants), including all tracked decode paths
  (`decode_tracked_quantized`, `decode_tracked_float`,
  `decode_tracked_quantized_proto`, `decode_tracked_float_proto`, and the
  public `decode_tracked` / `decode_proto_tracked` wrappers).

- **`MaskOverlay` compositing options** for `draw_decoded_masks()` and
  `draw_proto_masks()` — new `overlay` parameter with two controls:
  - `background: Option<&TensorDyn>` — when set, copies the background
    image into `dst` before compositing masks, allowing masks to be
    rendered over a separate frame.
  - `opacity: f32` (default `1.0`) — scales the alpha of every rendered
    mask and bounding box color; `0.5` produces semi-transparent overlays.

  Available in Rust, Python (`background=None, opacity=1.0` keyword args
  on `ImageProcessor.draw_masks()` and `ImageProcessor.draw_decoded_masks()`),
  and C (`hal_image_processor_draw_masks()` gains `background` and
  `opacity` params). All 7 GL fragment shaders updated with `uniform float
  opacity`.

- **GLES 3.1 context upgrade** — EGL context creation now requests
  GLES 3.1 (compute shaders) with automatic fallback to 3.0.

### Changed

- **`ImageProcessorTrait::draw_decoded_masks()` and `draw_proto_masks()`** gain a
  new `overlay: MaskOverlay<'_>` parameter. Implementors must update their
  trait impls. Pass `MaskOverlay::default()` for backward-compatible
  behaviour.

- **Fused `draw_masks` moved from Decoder to ImageProcessor** — the fused
  decode+render path is now owned by ImageProcessor (which owns the GPU).
  - **Python**: `Decoder.draw_masks()` removed. Use
    `ImageProcessor.draw_masks(decoder, model_output, dst, ...)`.
    The method now accepts an optional `tracker=` keyword argument for
    tracked inference. `decoder.decode()` now accepts `List[Tensor]`
    (previously `List[np.ndarray]`).
  - **C API**: `hal_decoder_draw_masks()` and `hal_decoder_decode_tracked_draw_masks()`
    removed. Use `hal_image_processor_draw_masks()` and
    `hal_image_processor_draw_masks_tracked()` (processor is now the
    first parameter).
  - **Rust**: Trait methods were renamed (`draw_masks` → `draw_decoded_masks`,
    `draw_masks_proto` → `draw_proto_masks`) and gained an `overlay` parameter.

- **C API `hal_image_processor_draw_masks()`** gains `background`
  (`const hal_tensor*`, pass `NULL` for none) and `opacity` (`float`, pass
  `1.0` for none) parameters. Existing C callers must update.

### Performance

- **Fused dequant+matmul kernel** for `materialize_segmentations()` —
  computes `mask_coeff @ protos` directly from the i8 proto tensor without
  allocating a 3.1 MB f32 copy. Includes fast sigmoid approximation
  (~10× faster than libm `expf`) and 4-way loop unrolling.
  Hybrid path speedup: 1.25–1.34× across all targets.

- **Zero-copy i8 proto extraction** — `extract_proto_data_quant()` uses
  `TypeId` specialization to avoid per-element `as_()` conversion when
  the proto tensor is already `i8`; uses flat `to_owned()` memcpy instead.

- **`SendablePtr` for `ProtoData`** in the GL threaded path — eliminates
  an 819 KB–3.3 MB deep clone per frame.

- **`glTexSubImage3D` fast path** — proto texture dimensions are tracked;
  reuses the existing GL texture object when dimensions match (every frame
  after the first), avoiding driver-side reallocation.

- **Cached opacity uniform** — skips 28 redundant GL state changes per
  frame on the default `opacity=1.0` path.

- **GLES 3.1 compute shader for HWC→CHW proto repack** — opt-in via
  `EDGEFIRST_PROTO_COMPUTE=1`. Uploads HWC i8 data to an SSBO and
  transposes to `GL_TEXTURE_2D_ARRAY` via compute dispatch. 2.2–2.4×
  speedup on the fused GL proto path (imx8mp, imx95).

### Removed

- **`decode_masks` / `decode_masks_atlas` atlas-based mask readback path** —
  removed the entire atlas decode pipeline which was too slow on embedded
  targets (435 ms on imx8mp Vivante GC7000UL). This removes:
  - `hal_decoder_decode_masks()` C API function
  - `Decoder.decode_masks()` Python method and type stub
  - `ImageProcessorTrait::decode_masks_atlas()` Rust trait method
  - `MaskRegion` and `MaskResult` types
  - Three GL logit-threshold shaders (`proto_mask_logit_*`)
  - Mask FBO, PBO, and atlas rendering infrastructure in the GL backend
  - `bench_decode_masks_atlas` benchmark

  Use `draw_proto_masks()` for GPU-accelerated mask overlay, or
  `materialize_segmentations()` + `draw_decoded_masks()` for the hybrid
  CPU decode path.

- **Python static helper methods on `Decoder`** — the following static/class
  methods have been removed. Use the `Decoder` instance API instead:
  - `Decoder.decode_yolo_det()`
  - `Decoder.decode_yolo_segdet()`
  - `Decoder.decode_modelpack_det()`
  - `Decoder.decode_modelpack_det_split()`
  - `Decoder.dequantize()`
  - `Decoder.segmentation_to_mask()`

- **`ArrayViewDQuantized` made private** (`pub(crate)`) — use the `TensorDyn`
  API for all decode inputs. Downstream crates that referenced
  `ArrayViewDQuantized` directly must migrate to `TensorDyn`.

### Migration Guide

This section summarises all breaking changes introduced in this release and
shows before/after code for each one.

#### Python

```python
# --- decode() input type changed ---
# Before: decoder.decode([np_array0, np_array1])
# After:  decoder.decode([hal_tensor0, hal_tensor1])   # List[Tensor]

# --- draw_masks moved from Decoder to ImageProcessor ---
# Before: decoder.draw_masks(outputs, processor, dst)
# After:  processor.draw_masks(decoder, outputs, dst)
# With tracking:
#         processor.draw_masks(decoder, outputs, dst, tracker=tracker)

# --- decode_masks() removed ---
# Before: decoder.decode_masks(outputs, processor)
# After:  decoder.decode(outputs)  # mask data is part of the decode() result

# --- Static methods removed ---
# Before: Decoder.decode_yolo_det(outputs, ...)
#         Decoder.decode_yolo_segdet(outputs, ...)
#         Decoder.decode_modelpack_det(outputs, ...)
#         Decoder.decode_modelpack_det_split(outputs, ...)
#         Decoder.dequantize(tensor)
#         Decoder.segmentation_to_mask(seg)
# After:  Use the Decoder instance API:
#         decoder = Decoder(config, score_threshold, iou_threshold)
#         boxes, segs = decoder.decode(outputs)
```

#### C

```c
/* --- draw_masks moved to ImageProcessor --- */
/* Before: hal_decoder_draw_masks(decoder, processor, outputs, n, dst, &boxes); */
/* After:  hal_image_processor_draw_masks(processor, decoder, outputs, n, dst,
                                          NULL, 1.0, &boxes);
   (new background + opacity params; pass NULL and 1.0 for old behaviour)      */

/* --- Tracked draw_masks also moved --- */
/* Before: hal_decoder_decode_tracked_draw_masks(decoder, tracker, ts, processor, ...); */
/* After:  hal_image_processor_draw_masks_tracked(processor, decoder, tracker, ts, ...); */

/* --- decode_masks removed --- */
/* Before: hal_decoder_decode_masks(decoder, processor, outputs, n, w, h,
                                    &boxes, &masks);                            */
/* After:  hal_decoder_decode(decoder, outputs, n, &boxes, &segs);             */
```

#### Rust

```rust
// --- Trait method renames (ImageProcessorTrait implementors) ---
// draw_masks       → draw_decoded_masks
// draw_masks_proto → draw_proto_masks
// Both gain an `overlay: MaskOverlay<'_>` parameter.
// Pass MaskOverlay::default() for backward-compatible behaviour.

// --- New primary API on ImageProcessor ---
// processor.draw_masks(&decoder, &outputs, &mut dst, overlay)

// --- Decoder now accepts TensorDyn ---
// Before: decoder.decode(&[&array_view_dquantized, ...], &mut boxes, &mut masks)
// After:  decoder.decode(&[&tensor0, &tensor1], &mut boxes, &mut masks)
// (previously required ArrayViewDQuantized / ArrayViewD<T>)

// --- ArrayViewDQuantized is now pub(crate) ---
// Use TensorDyn API instead. ArrayViewDQuantized is no longer part of the
// public API surface.
```

## [0.13.2] - 2026-03-26

### Fixed

- GL destination EGLImage renderbuffer path added to fix `GL_OUT_OF_MEMORY` (0x505) on non-dma_heap DMA-BUF buffers (e.g. Neutron NPU on i.MX 95). In 0.13.x the default remains the texture path for compatibility; set `EDGEFIRST_OPENGL_RENDERSURFACE=1` to opt in. This will become the automatic default on supported platforms in a future release after broader testing.
- Allow CPU mapping of offset DMA tensors for G2D I8 post-processing at non-zero buffer offsets
- DrmAttachment failure log level: warn for self-allocated buffers, debug for imported (external) buffers

## [0.13.1] - 2026-03-26

### Fixed

- DMA-BUF `plane_offset` not applied in G2D and OpenGL destination paths
- Validate row stride is a multiple of bytes-per-pixel in G2D

## [0.13.0] - 2026-03-25

### Added

- **Integrated object tracking in decoder** (`decode_tracked`) — fused
  decode+track pipeline that applies ByteTrack multi-object tracking to
  decoded detections, maintaining persistent track IDs across frames with
  Kalman-filtered location smoothing. Available across all language bindings:

  | Layer | New APIs |
  |-------|----------|
  | Rust | `decode_tracked_quantized()`, `decode_tracked_float()`, `decode_tracked_quantized_proto()`, `decode_tracked_float_proto()` |
  | C | `hal_decoder_decode_tracked()`, `hal_decoder_decode_tracked_draw_masks()` |
  | Python | `Decoder.decode_tracked()` |

- **ByteTrack tracker API** — standalone tracker with configurable confidence
  thresholds, IOU matching, and track lifespan. New types `TrackInfo` (UUID,
  smoothed location, timestamps, update count) and `ActiveTrackInfo` (track
  info + raw detection). Available in Rust, C (`hal_bytetrack_*` functions),
  and Python (`ByteTrack` class).

- **Delegate DMA-BUF API type definitions** (EDGEAI-1189) — C ABI contract
  for zero-copy NPU integration. Defines `hal_dmabuf_tensor_info` struct
  (96 bytes, no padding on LP64) and function signatures for delegate
  implementors: `hal_dmabuf_is_supported()`, `hal_dmabuf_get_tensor_info()`,
  `hal_dmabuf_sync_for_device()`, `hal_dmabuf_sync_for_cpu()`. Forward-
  compatible via `info_size` parameter. Types only — function implementations
  are provided by delegate libraries (e.g. `edgefirst-tflite`).

### Fixed

- Panics, null dereference, and unsafe implementations identified during
  code review of the tracker integration (EDGEAI-1022)

## [0.12.0] - 2026-03-24

### Added

- **Row stride and plane offset for DMA-BUF tensors** — tensors from
  `import_image` now carry per-plane stride and offset metadata, propagated
  end-to-end into EGL `DMA_BUF_PLANE*_PITCH` / `PLANE*_OFFSET` attributes.
  This enables correct GPU rendering of V4L2 buffers with row padding
  (e.g. `bytesperline=2048` on a 1920-wide frame). New APIs:

  | Layer | New APIs |
  |-------|----------|
  | Rust | `set_row_stride()`, `with_row_stride()`, `effective_row_stride()`, `set_plane_offset()`, `plane_offset()` |
  | C | `hal_plane_descriptor_set_stride()`, `hal_plane_descriptor_set_offset()`, `hal_tensor_row_stride()` |
  | Python | `import_image(..., stride=, offset=, chroma_stride=, chroma_offset=)` |

  Changing the pixel format to a different value via `set_format()` clears
  stored stride and offset; re-setting the same format preserves both.
  `Tensor::map()` rejects strided or offset tensors to prevent incorrect
  CPU access on padded buffers.

- **VYUY pixel format** (`PixelFormat::Vyuy`) — packed YUV 4:2:2 with VYUY
  byte order. Available in Rust, C (`HAL_PIXEL_FORMAT_VYUY`), and Python.
  Note: EGL DMA-BUF import produces incorrect output on Vivante (~0.28
  SSIM); the GL backend auto-falls back to the CPU/G2D path for this format.

- **Software OpenGL renderer rejection** — the GL backend detects
  llvmpipe/softpipe/swrast at init and returns `Error::NotSupported`,
  causing the auto-detection fallback chain to select CPU or G2D instead.

- **GL thread panic safety** — the OpenGL worker thread now wraps all
  message handlers in `catch_unwind`. A caught panic sets a `poisoned` flag;
  all subsequent GL calls return `Error::Internal` instead of hanging.

- **C API preprocessing benchmark** (`bench_preproc.c`) — reference for
  GStreamer/V4L2 integrators covering seven DMA-BUF import patterns, a
  format conversion matrix, and a two-stage chained pipeline.

### Removed

- **`hal_tensor_from_planes()`** — use `hal_import_image()` with separate
  `hal_plane_descriptor_new()` descriptors instead. The new API supports
  per-plane stride/offset and is processor-aware (EGL cache compatible).

### Changed

- **Contiguous NV12 UV offset is stride-aware** — the UV plane byte offset
  for single-fd NV12 is now `effective_row_stride() * height` instead of
  `width * height`, fixing corrupted chroma on padded buffers.

- **Multiplane `import_image` rejects unsupported dtypes** — only `U8` and
  `I8` are valid for multiplane NV12 imports; other dtypes return
  `Error::NotSupported` instead of silently producing a wrong-type tensor.

### Fixed

- **NV12 UV sampling incorrect for padded buffers** — V4L2 buffers with
  stride > width caused corrupted chroma because the UV plane offset passed
  to EGL was computed from width, not from the actual stride.

- **`hal_import_image` double-free** — passing the same pointer as both
  `image` and `chroma` caused UB. Now returns `EINVAL`.

## [0.11.0] - 2026-03-22

### Added

- **Zero-copy external buffer import** (`import_image`) across Rust, C,
  and Python APIs. Enables GPU rendering directly into a caller-owned
  DMA-BUF — eliminating the `memcpy` between HAL's output buffer and an
  NPU delegate's input buffer. Supports per-plane stride and offset for
  padded allocators (V4L2, GStreamer). The caller retains ownership of
  the fd; HAL borrows it via `dup(fd)`.

  **Rust:**
  ```rust,ignore
  use edgefirst_tensor::PlaneDescriptor;
  let pd = PlaneDescriptor::new(vx_fd.as_fd())?.with_stride(bytesperline);
  let mut dst = processor.import_image(pd, None, 640, 640, PixelFormat::Rgb, DType::U8)?;
  processor.convert(&src, &mut dst, Rotation::None, Flip::None, Crop::letterbox())?;
  ```

  **Python:**
  ```python
  dst = processor.import_image(vx_fd, 640, 640, ef.PixelFormat.Rgb, stride=bytesperline)
  processor.convert(src, dst)
  ```

  **C:**
  ```c
  struct hal_plane_descriptor *pd = hal_plane_descriptor_new(vx_fd);
  hal_plane_descriptor_set_stride(pd, bytesperline);
  struct hal_tensor *dst = hal_import_image(proc, pd, NULL,
                                             640, 640,
                                             HAL_PIXEL_FORMAT_RGB, HAL_DTYPE_U8);
  ```

  Returns `Error::NotSupported` if the fd is not DMA-backed (e.g. POSIX
  shm fd), ensuring the zero-copy contract is enforced at the API boundary.

- **Tensor pixel-format metadata** — `TensorDyn::set_format()` and
  `TensorDyn::with_format()` attach a `PixelFormat` to any tensor,
  enabling `convert()` to use raw tensors created via `from_fd()` or
  `Tensor::new()` as image destinations without going through
  `create_image()`. Shape validation ensures the format matches the
  tensor dimensions. Available in Python (`Tensor.set_format()`) and
  C (`hal_tensor_set_format()`).

- **DMA-BUF fd accessors** — `TensorDyn::dmabuf()` (borrows the fd) and
  `TensorDyn::dmabuf_clone()` (returns an owned duplicate) for exporting
  HAL-allocated DMA-BUF buffers to external consumers.
  `Tensor::<T>::dmabuf()` provides the same on typed tensors. Available
  in Python (`Tensor.dmabuf_clone()`) and C (`hal_tensor_dmabuf_clone()`).

  Use case: HAL allocates via `create_image()`, consumer imports the fd:
  ```rust,ignore
  let hal_dst = processor.create_image(640, 640, PixelFormat::Rgb, DType::U8, None)?;
  let fd = hal_dst.dmabuf_clone()?;  // Error if not DMA-backed
  delegate.register_buffer(fd)?;
  ```

- PyPI README (`crates/python/README.md`) with quick-start, zero-copy
  external buffer examples, and platform support matrix

### Removed

- **`hal_image_processor_convert_ref`** — removed from Rust, C header, and
  ARCHITECTURE.md. This function was redundant with the `hal_tensor_set_format()`
  + `hal_image_processor_convert()` two-step and had been a recurring source of
  safety bugs (use-after-free in v0.9, unsound `ptr::read`/`ManuallyDrop` in
  v0.10.0). Callers should use:
  ```c
  hal_tensor_set_format(dst, format);
  hal_image_processor_convert(proc, src, dst, rotation, flip, crop);
  ```

### Fixed

- Corrected errno mapping for `hal_import_image`: `InvalidShape` / `NotAnImage`
  and `Tensor(InvalidArgument/InvalidShape/ShapeMismatch)` map to `EINVAL`
  (not `EIO`); `UnsupportedFormat` / `NotSupported` map to `ENOTSUP`

## [0.10.0] - 2026-03-20

### Changed

- **BREAKING: Unified Tensor API** — `TensorImage` removed from all APIs.
  The `Tensor` type is now the single tensor type across Rust, C, and Python.
  - **Python**: `TensorImage(w, h, fmt)` → `Tensor.image(w, h, fmt)`;
    `TensorImage.load()` → `Tensor.load()`; `FourCC` → `PixelFormat`
  - **C**: `hal_tensor_image_*` functions replaced by `hal_tensor_*` equivalents
    (e.g., `hal_tensor_new_image()`, `hal_tensor_load_image()`,
    `hal_tensor_width()`, `hal_tensor_pixel_format()`);
    `HalTensorImage` removed — use `HalTensor` for all tensors
  - **Rust**: `TensorDyn` is the unified type-erased tensor;
    `PyTensor` and `HalTensor` now wrap `TensorDyn` directly

### Added

- `TensorDyn::new()`, `TensorDyn::from_fd()`, `TensorDyn::size()`,
  `TensorDyn::memory()`, `TensorDyn::reshape()`, `TensorDyn::clone_fd()`
  methods for complete type-erased tensor operations
- `HalDtype::F16` variant in C API for float16 tensor support
- NV12 even-height validation in `Tensor::image()`
- Semi-planar shape constraint validation in `Tensor::set_format()`
- **`dtype` parameter on `create_image()`** across Rust, C, and Python APIs.
  Supports `DType::I8` / `"int8"` for direct signed int8 tensor output;
  PBO-backed images support `u8` and `i8` byte-sized types, other dtypes
  prefer DMA-buf when available and fall back to system memory
- **Int8 GPU shaders** for all GL render paths (DMA, non-DMA, non-DMA
  planar, PBO-to-PBO). The fragment shader applies XOR 0x80 bias on the
  GPU, eliminating the CPU readback + XOR post-pass for ~2× faster int8
  letterbox pipeline on GPU-capable platforms
- Comprehensive `trace`-level logging in the convert dispatch path for
  profiling backend selection and per-stage timing
- **Dual Python wheel builds** (`abi3-py311` and `abi3-py38`). Python 3.11+
  users get zero-copy buffer protocol support; Python 3.8–3.10 users get a
  compatible fallback. Pip automatically selects the best wheel

- **C logging API** (`hal_log_init_file`, `hal_log_init_callback`) —
  two new C functions for initialising HAL logging from C/C++ callers.
  `hal_log_init_file(FILE*, HalLogLevel)` writes formatted log lines to a
  `FILE*` (typically `stderr`).
  `hal_log_init_callback(HalLogCallback, void*, HalLogLevel)` routes
  each log record to a user-supplied callback, enabling integration with
  GStreamer, syslog, or other logging frameworks. Only the first call per
  process takes effect; subsequent calls return `-1` with `errno = EALREADY`

### Fixed

- EGL image cache evict-before-create causing unnecessary cache churn
  and potential fd accumulation on GREY format fallback
- Use-after-free in `hal_image_processor_convert_ref` (replaced
  `Box::from_raw` with `ptr::read` + `ManuallyDrop`)
- Missing `return Ok(())` in stable `normalize_to_float_16` RGBA path
  that caused double-processing
- Clippy `iflet_redundant_pattern_matching` and `collapsible_match` lints
- Broken `FourCC` imports and variant casing in Python tests
- Stale `.display()` calls in G2D tests and duplicate `.as_u8()` in GL tests
- Int8 letterbox bias on all four GL render paths — `glClear` bypasses the
  fragment shader, so `crop.dst_color` must be XOR'd on the CPU side before
  calling `glClearColor`
- Alpha-channel preservation in CPU and G2D int8 XOR bias — now skips
  alpha bytes for Rgba/Bgra formats, matching GL shader behavior
- GL auto-backend priority: OpenGL is now preferred over G2D when both
  are available
- Pinned Rust toolchain versions in CI to prevent profraw format corruption
  across build/coverage-processing jobs; `--failure-mode=all` for
  `llvm-profdata merge` to skip corrupted profraw files from GPU driver
  crashes

## [0.9.1] - 2026-03-10

### Added

- **BGRA destination format**: `create_image()` and `convert()` now accept
  BGRA as a destination format for Cairo/Wayland compositing (ARGB32 on
  little-endian). Supported natively on OpenGL (via `GL_BGRA`) and G2D
  (`G2D_BGRA8888`); CPU backend uses R/B channel swizzle after RGBA
  conversion. Available in Rust (`PixelFormat::Bgra`), Python (`PixelFormat.Bgra`),
  and C (fourcc `"BGRA"`). `draw_masks()` and `draw_masks_proto()` support
  BGRA destination images on the OpenGL backend; CPU mask rendering
  accepts only RGBA/RGB destinations.

- **`EDGEFIRST_FORCE_BACKEND` environment variable**: forces `ImageProcessor`
  to initialize only the specified backend (`cpu`, `g2d`, or `opengl`) with
  no fallback chain. Useful for benchmarking individual backends in isolation.

- **Hybrid CPU+GL mask rendering path**: `draw_masks_proto()` now materializes
  segmentations on CPU via `materialize_segmentations()` then composites via
  OpenGL, which is 2.5–27× faster than the previous full-GPU path across
  tested platforms (imx8mp, imx95, rpi5, x86).

- **`CPUProcessor::materialize_segmentations()`**: new public method that
  computes per-detection segmentation masks from raw prototype data without
  rendering them onto an image.

- **Platform benchmark infrastructure**: migrated all benchmarks from Divan to
  `edgefirst-bench` with JSON output support (`BenchSuite`), added dedicated
  benchmark binaries (`tensor_benchmark`, `pipeline_benchmark`,
  `mask_benchmark`, `decoder_benchmark`, `opencv_benchmark`), and added
  benchmark data collection scripts and `BENCHMARKS.md` results document.

- **Test coverage for new APIs**: 21 new tests covering NORM_LIMIT regression
  (protobox), `materialize_segmentations`, `EDGEFIRST_FORCE_BACKEND` env var,
  hybrid mask path error handling, `decode_*_proto` functions, and GL smoke
  tests.

### Fixed

- **YOLO NORM_LIMIT too restrictive**: relaxed `NORM_LIMIT` from 1.01 to 2.0
  in `protobox()` to allow YOLO models that legitimately predict bounding box
  coordinates slightly > 1.0 for objects near frame edges. Coordinates > 2.0
  still return `InvalidShape` to catch un-normalized pixel-space boxes.

### Changed

- ARCHITECTURE.md updated to v2.6: documents hybrid mask path, benchmark
  infrastructure, and `EDGEFIRST_FORCE_BACKEND` env var
- README.md updated with benchmarking section and quick-reference commands

## [0.9.0] - 2026-03-04

### Added

- **`hal_decoder_draw_masks()` C API function**: fused decode+render path
  that mirrors the Python `Decoder.draw_masks()` binding — takes decoder,
  processor, raw model outputs, and destination image; returns rendered overlay
  and detection boxes in a single call. Internally selects proto-based GPU
  matmul path for seg models, with automatic fallback to decoded-mask rendering
  for detection-only models.

- **`hal_decoder_decode_masks()` C API function**: decodes segmentation model
  outputs and returns per-detection pixel masks at the specified output
  resolution. Mirrors the Python `Decoder.decode_masks()` binding.

- **PBO (Pixel Buffer Object) tensor backend** (`edgefirst-tensor`):
  `PboTensor<T>` provides GPU-native buffer storage for platforms where DMA-buf
  is unavailable (e.g. NVIDIA desktop GPUs). PBO tensors are managed by the
  OpenGL thread via a `WeakSender` channel that allows clean shutdown without
  blocking on orphaned tensors.

- **`ImageProcessor::create_image()`** factory method that probes GPU
  capabilities at initialization and selects the optimal memory backend:
  DMA-buf > PBO > heap memory. This is now the preferred way to allocate
  images for use with `convert()`. Available in Rust, Python
  (`ImageProcessor.create_image()`), and C
  (`hal_image_processor_create_image()`).

- **PBO convert paths** for all source/destination combinations:
  `convert_pbo_to_pbo()` (both PBO), `convert_any_to_pbo()` (Mem/DMA source
  to PBO destination), and `convert_pbo_to_mem()` (PBO source to Mem
  destination). All paths use direct GL buffer bindings
  (`GL_PIXEL_UNPACK_BUFFER` / `GL_PIXEL_PACK_BUFFER`) to avoid the deadlock
  that would occur if the GL thread called `tensor.map()` on a PBO tensor.

- **`hal_image_processor_create_image()`** C API function with documentation,
  null-safety checks, and errno-based error reporting

- **Python `ImageProcessor.create_image()`** method with format parameter
  defaulting to RGBA

- Python tests: `test_create_image`, `test_create_image_formats`,
  `test_create_image_convert`, `test_create_image_roundtrip`

- C API tests: `test_image_processor_create_image`,
  `test_image_processor_create_image_null_params`,
  `test_image_processor_create_image_convert`

- Rust integration test: `test_convert_pbo_to_pbo` exercising PBO-to-PBO
  conversion with SSIM comparison against CPU reference

### Fixed

- **GL thread shutdown hang**: `GlPboOps` used a strong `Sender` clone that
  kept the GL thread's message channel alive after `GLProcessorThreaded` was
  dropped, causing `handle.join()` to block indefinitely. Changed to
  `WeakSender` so the channel closes when the last `ImageProcessor` reference
  is dropped.

- **PBO-to-PBO deadlock**: `convert_pbo_to_pbo()` called `draw_src_texture()`
  which invoked `tensor.map()` on the GL thread, sending a message back to
  itself. Added `draw_src_texture_from_pbo()` that binds the source PBO as
  `GL_PIXEL_UNPACK_BUFFER` with a NULL `glTexImage2D` pointer for zero-copy
  upload.

- **Mixed PBO/Mem deadlock**: `convert_dest_non_dma()` called
  `dst.tensor().map()` on the GL thread for PBO destinations. Added dedicated
  `convert_any_to_pbo()` and `convert_pbo_to_mem()` methods that use GL buffer
  bindings instead of tensor mapping.

### Changed

- **Mask API renamed for clarity** — API names now use consistent verbs:
  `decode` = get mask data back, `draw` = overlay onto image. The compound
  names that accumulated during iterative optimization have been replaced
  with shorter, intent-revealing names across all language bindings:

  **Renamed methods:**

  | Layer | Old Name | New Name |
  |-------|----------|----------|
  | Rust trait | `render_from_protos()` | `draw_masks_proto()` |
  | Rust trait | `render_to_image()` | `draw_masks()` |
  | Rust trait | `render_masks_from_protos()` | `decode_masks_atlas()` |
  | Rust enum | `ImageRenderProtos` | `DrawMasksProto` |
  | Rust enum | `ImageRender` | `DrawMasks` |
  | Rust enum | `RenderMasksFromProtos` | `DecodeMasksAtlas` |
  | Python `Decoder` | `decode_and_render()` | `draw_masks()` |
  | Python `ImageProcessor` | `render_to_image()` | `draw_masks()` |
  | C API | `hal_image_processor_render_to_image()` | `hal_image_processor_draw_masks()` |

  **New methods (no previous equivalent):**

  | Layer | New Name |
  |-------|----------|
  | Python `Decoder` | `decode_masks()` |
  | C API | `hal_decoder_draw_masks()` |
  | C API | `hal_decoder_decode_masks()` |

  **Migration guide:**
  - **Python users**: rename `decoder.decode_and_render(...)` →
    `decoder.draw_masks(...)` and `processor.render_to_image(...)` →
    `processor.draw_masks(...)`; `decoder.decode_masks(...)` is new
  - **C users**: rename `hal_image_processor_render_to_image()` →
    `hal_image_processor_draw_masks()`; `hal_decoder_draw_masks()` and
    `hal_decoder_decode_masks()` are new
  - **Rust users**: rename `render_from_protos()` → `draw_masks_proto()`,
    `render_to_image()` → `draw_masks()`, and
    `render_masks_from_protos()` → `decode_masks_atlas()` on `ImageProcessorTrait`
    implementors

- **`Decoder.decode_masks()` return type changed** — now returns individual
  per-detection masks as `List[ndarray]` (each shape `(H, W)`, uint8) instead
  of the raw atlas tuple. The atlas packing is now an internal optimization.

- ARCHITECTURE.md updated to v2.4: documents PBO tensor architecture,
  `create_image()` backend selection, PBO convert dispatch table, and
  `WeakSender` shutdown design

- README.md updated: `create_image()` presented as preferred image allocation
  method with usage guidance for Rust, Python, and C; DMA-buf permission
  requirements; platform GPU support table

## [0.8.0] - 2026-02-24

### Added

- GPU segmentation mask atlas rendering: `decode_masks()` Python binding and
  `render_mask_atlas()` on `ImageProcessorTrait` with GL, CPU, and G2D (stub)
  backends — renders all masks in a single GPU pass with one PBO readback,
  eliminating CPU mask computation and per-mask GL resize roundtrips
- EGL display probe and override API: `probe_egl_displays()`, `EglDisplayKind` enum,
  `EglDisplayInfo` struct, and `ImageProcessor::with_config()` constructor in Rust;
  `EglDisplayKind`, `EglDisplayInfo`, `probe_egl_displays()`, and
  `ImageProcessor(egl_display=...)` in Python — enables selecting or avoiding specific
  EGL display types on problematic hardware (e.g. Vivante GBM on i.MX8)
- Python API for programmatic decoder configuration (EDGEAI-774): `DecoderType`,
  `DecoderVersion`, `DimName` enums, `Output` class with factory methods
  (`detection`, `boxes`, `scores`, `protos`, `segmentation`, `mask_coefficients`,
  `mask`), and `Decoder.new_from_outputs()` constructor
- Decoder builder API with programmatic output configuration and C API refactoring
  to opaque `hal_decoder_params` pattern with setter functions
- `yolov8` serde alias for `DecoderType::Ultralytics` so old model metadata
  with `decoder: yolov8` deserializes without migration
- Dict-format `dshape` deserialization in decoder configs: accepts both
  array-of-single-key-dicts (`[{"batch": 1}, ...]`) and array-of-tuples formats
- Segmentation pipeline comparison script (`example_seg_pipeline.py`): end-to-end
  OpenCV vs HAL decode vs HAL fused benchmark for YOLOv8-seg INT8 TFLite

### Fixed

- NV16 handling in `TensorImage::new()`, `width()`, `height()`, `channels()`:
  constructor created wrong 3D shape instead of 2D `[H*2, W]`, and accessors
  only special-cased NV12, causing index-out-of-bounds panics
- NV12/NV16 in `TensorImageRef::from_borrowed_tensor`: expected all formats to
  have 3D tensor shapes, causing index-out-of-bounds panic for semi-planar
  formats via the borrowed reference path (`hal_image_processor_convert_ref`)

## [0.7.0] - 2026-02-17

### Added

- NV12/NV16 2D tensor support: `TensorImage::from_tensor()` now accepts 2D
  tensors with shapes [H*3/2, W] for NV12 and [H*2, W] for NV16, with
  shape validation and format-specific documentation
- C API: `hal_tensor_image_from_tensor()` with dtype/ndim validation,
  `hal_tensor_image_map_create()`, and `HalTensor::ndim()` helper
- C API: pkg-config (`edgefirst-hal.pc`) and SOVERSION support with standard
  GNU/Linux versioned shared library symlinks in release archives
- Shared DRM render node file descriptor (via `OnceLock`) across all DMA-buf
  PRIME imports, avoiding deadlocks on Vivante with concurrent V4L2/VPU usage
- EGL/GL shutdown cleanup architecture documentation in ARCHITECTURE.md with
  root cause analysis and industry cross-references

### Fixed

- EGL/GL cleanup crashes during process shutdown: defense-in-depth strategy
  using `Box::leak` (prevent `dlclose`), `ManuallyDrop` (skip
  `eglReleaseThread`), `catch_unwind` (catch panics from invalidated function
  pointers), and omitted `eglTerminate` (avoid Vivante double-free)
- OpenGL EGL initialization: reorder GBM/DRM probe before default display to
  avoid blocking when Wayland compositor is not running
- Test suite: DMA permission guards for rootless CI, Python test runner
  migration to pytest, pre-commit hook enforcement
- C API test failures after SOVERSION change
- Decoder: remove incorrect `#[serde(flatten)]` on `Protos.quantization`

## [0.6.2] - 2026-02-13

### Added

- `TensorImage.from_fd()` Python method to create tensor images from
  file descriptors (Linux only), enabling zero-copy DMA-buf sharing
  with external processes
- Linux aarch64 (ARM64) Python wheels built natively on `ubuntu-22.04-arm`
  with `manylinux2014` compatibility, published to PyPI and GitHub Releases

### Changed

- PyO3 stable ABI is now opt-in via `abi3-py311` and `abi3-py38` Cargo
  features instead of being the default; CI workflows pass
  `--features abi3-py311` to build portable wheels for Python 3.11+
  while local development builds target the exact installed Python version

## [0.6.1] - 2026-02-13

### Added

- Linux aarch64 (ARM64) Python wheels built natively on `ubuntu-22.04-arm`
  with `manylinux2014` compatibility, published to PyPI and GitHub Releases

### Fixed

- Enable PyO3 stable ABI (`abi3-py311`) so wheels are recognized as compatible
  binary distributions by pip on Python 3.11+ instead of falling back to
  source builds

## [0.6.0] - 2026-02-13

### Changed

- Migrate `g2d-sys` from local crate to published `g2d-sys 1.2.0` from crates.io
  (now maintained at [github.com/EdgeFirstAI/g2d-rs](https://github.com/EdgeFirstAI/g2d-rs))
- G2D `convert` now calls `g2d.finish()` once after queuing all operations
  (clear + blit), reducing GPU synchronization barriers from 2 to 1 per frame
- Letterbox clear is now controlled by the caller via `Crop::dst_color`:
  `Some(color)` clears the destination with the specified color before blit,
  `None` skips the clear (preserving whatever is in the destination buffer,
  enabling custom backgrounds or caller-managed clearing)
- Always use hardware `g2d_clear` for supported pixel formats (2-byte and
  4-byte formats); CPU fallback only for RGB888 which G2D does not support
- Move benchmarks from `test.yml` to dedicated `benchmark.yml` workflow
  triggered on-demand via `workflow_dispatch` with QuickChart result charts

### Fixed

- Implement DRM PRIME attachment for DMA-buf cache coherency — `DMA_BUF_IOCTL_SYNC`
  was previously a no-op on cached CMA heaps because no `dma_buf_attach` existed.
  Each `DmaTensor` now creates a persistent DRM attachment via
  `DRM_IOCTL_PRIME_FD_TO_HANDLE` to enable proper cache invalidation and flushing

### Added

- `DrmAttachment` struct in `edgefirst-tensor` for persistent DRM PRIME imports
- Realistic letterbox pipeline benchmarks (`bench_letterbox_pipeline`) comparing
  CPU vs G2D vs OpenGL for 1080p/4K YUYV/NV12 → 640×640 RGBA letterbox resize
- `CPUProcessor::fill_image_outside_crop` is now public

## [0.5.2] - 2026-02-12

### Changed

- Consolidate all scripts to `.github/scripts/` for CI/CD automation
- Migrate `AGENTS.md` to `.github/copilot-instructions.md` (GitHub Copilot standard location)
- Add detailed release process documentation with version verification steps
- Update commit message guidelines: JIRA keys not required for housekeeping tasks

## [0.5.1] - 2026-02-11

### Changed

- GitHub release workflow improvements for trusted publishing

## [0.5.0] - 2026-02-09

First published release of EdgeFirst HAL to [crates.io](https://crates.io) and [PyPI](https://pypi.org).

### Core Crates

- **`edgefirst-hal`** - Main HAL crate re-exporting tensor, image, and decoder functionality
- **`edgefirst-tensor`** - Zero-copy tensor memory management with DMA-heap, shared memory,
  and heap allocation backends
- **`edgefirst-image`** - High-performance image processing with hardware-accelerated conversion
  and resizing (G2D, OpenGL ES, CPU fallback)
- **`edgefirst-decoder`** - ML model output decoder for YOLO (v5/v8/v11/v26) and ModelPack
  object detection and instance segmentation with NMS
- **`edgefirst-tracker`** - ByteTrack multi-object tracking with Kalman filtering
- **`g2d-sys`** - Low-level FFI bindings for NXP i.MX G2D 2D graphics acceleration

### Added

- **Tensor Memory Management** (`edgefirst-tensor`):
  - Zero-copy memory buffers with DMA-heap, shared memory, and heap backends
  - `TensorImage` for loading JPEG/PNG images into tensor-backed memory
  - DMA buffer synchronization controls and file descriptor management
  - `is_dma_available()` public API for runtime capability detection
  - ndarray integration (optional, enabled by default)
  - Cross-platform support: Linux DMA/SHM, macOS/Windows heap fallback

- **Image Processing** (`edgefirst-image`):
  - Format conversion: YUYV, NV12, NV16, RGB, RGBA, GREY, Planar RGB/RGBA (8BPS)
  - Hardware-accelerated resize via NXP G2D and OpenGL ES 3.0
  - CPU fallback converter for all platforms
  - Source and destination crop with letterboxing and fill color
  - Rotation (0/90/180/270) and horizontal/vertical flip
  - EXIF-aware automatic image orientation
  - Normalization modes: signed (-1..1), unsigned (0..1), raw, with f16 support
  - Multi-threaded normalization with rayon
  - Automatic converter selection: tries all available backends (G2D, OpenGL, CPU)
  - OpenGL renders to texture with dedicated thread for stability
  - GBM dynamic loading via `edgefirst-gbm` for EGL surface management
  - Segmentation mask and bounding box overlay rendering (OpenGL, G2D, CPU)

- **Model Decoder** (`edgefirst-decoder`):
  - YOLO v5/v8/v11 object detection decoding with class-aware and class-agnostic NMS
  - YOLO v26 detection support with architecture version disambiguation
  - YOLO instance segmentation with quantized mask decoding (f32, i8, i32)
  - ModelPack detection and segmentation decoding
  - Split decoder for models with separate box and class outputs
  - End-to-end model support (models with built-in NMS)
  - Configuration via YAML/JSON files and EdgeFirst Model Metadata format
  - Configurable score threshold, IoU threshold, and NMS mode
  - Dequantization support for int8/uint8/int32 quantized model outputs
  - Generic decode API with automatic model type and architecture detection

- **Multi-Object Tracker** (`edgefirst-tracker`):
  - ByteTrack algorithm with Kalman filtering
  - Configurable track birth/death thresholds
  - IoU-based association with LAPJV assignment

- **Python Bindings** (`edgefirst-hal` on PyPI):
  - PyO3-based Python API with numpy and f16 integration
  - `TensorImage` for image loading, format query, and tensor memory control
  - `ImageProcessor` for format conversion, resize, crop, and normalization
  - `Decoder` for YOLO/ModelPack model output post-processing
  - Config creation from Python dictionaries via pythonize
  - `PyArrayLike` inputs (accepts both numpy arrays and Python lists)
  - File descriptor management (`from_fd`, `fd` property)
  - Destination crop and fill color controls
  - Python type stubs (`.pyi`) for IDE autocompletion
  - Support for Python 3.8+ (ABI3) and 3.11+ (ABI3-311)

- **NXP G2D Bindings** (`g2d-sys`):
  - FFI bindings for NXP i.MX G2D hardware 2D acceleration
  - Dynamic loading via `libloading` for runtime availability detection
  - Version-aware API support (legacy and modern G2D, imx95)

- **Platform Support**:
  - Linux x86_64 and aarch64 with full hardware acceleration
  - macOS (Apple Silicon and x86_64) with CPU-only processing
  - Windows with heap memory tensors and CPU processing
  - NXP i.MX8/i.MX9 with G2D and OpenGL ES hardware acceleration

- **CI/CD Infrastructure**:
  - GitHub Actions workflows for testing, coverage, and releases
  - Multi-platform CI: x86_64, aarch64, macOS ARM
  - On-target hardware testing with NXP i.MX boards
  - Multi-platform Python wheel builds (Linux manylinux2014, macOS, Windows)
  - PyPI trusted publishing with OIDC
  - SBOM generation and license compliance checking
  - Comprehensive Criterion and Divan benchmark suites
  - Shared benchmark module with common test utilities

- **Publishing**:
  - All workspace crates published to crates.io
  - `edgefirst-gbm` / `edgefirst-gbm-sys` published as standalone crates for GBM
    dynamic loading support (fork of Smithay/gbm.rs)
  - Comprehensive crate documentation with README, keywords, and categories
  - API documentation with doc-tests for all public interfaces

### Notes

- Python bindings are distributed via PyPI only (`publish = false` for the Python crate)
- `g2d-sys` maintains its own version (1.0.1) independent of the workspace
- `edgefirst-gbm` 0.18.1 and `edgefirst-gbm-sys` 0.4.1 are published separately
  from the [EdgeFirstAI/gbm.rs](https://github.com/EdgeFirstAI/gbm.rs) repository
- Apache-2.0 license with NOTICE file for third-party attributions
