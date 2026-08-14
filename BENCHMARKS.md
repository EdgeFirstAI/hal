# EdgeFirst HAL - Benchmarks

**Version:** 3.12
**Last Updated:** August 14, 2026
**Status:** two capture generations coexist in this file — read the scope note
below before quoting any number.

**GL / preprocessing matrix (Buffer Infrastructure, Image Preprocessing,
Format Conversion, Mask Rendering, `bench_preproc`, `decode_pipeline_benchmark`
sections): 0.25.0 release refresh.** The full bench matrix was re-collected on
the 0.25.0 converged-GL-engine code across imx8mp-frdm (Vivante GC7000UL +
G2D), imx95-frdm (Mali-G310 + DPU-G2D), rpi5-hailo (V3D 7.1), and
jetson-orin-nano (NVIDIA Tegra Orin, PBO/CUDA path), plus the existing
`mbp-m2-max` (ANGLE + IOSurface). The re-collected matrix confirms the recent
GL-convergence captures within measurement noise (e.g. imx95 GL 1080p
YUYV→RGBA letterbox 1.2 ms → 957 µs, NV12→RGBA 1.2 ms → 830 µs); no
regressions were observed on any GPU class. The Allocation table is updated
for the notable imx8mp DMA-alloc improvement (38 ms → 1.8 ms at 720p); other
tables are unchanged within noise. Raw per-board JSON lives in the
(git-ignored) `benchmarks/<platform>/` working tree and is regenerated into
these tables via `.github/scripts/generate_benchmark_tables.py`. The macOS GL
rows here are still the pre-convergence capture — 0.25.0 also moved macOS onto
the shared GL engine, which lifted the old YUYV→RGBA-only limit, but the
macOS matrix has not been re-collected since (Known Gap #17). YUYV→RGBA
same-size convert vs the Apple Silicon CPU path: **1.32×** at 1080p, **4.76×**
at 4K.

> **The GL/preprocessing matrix above is two minor releases behind the
> workspace.** It was collected on 0.25.0; the workspace is now on 0.28.x.
> Treat it as the last known-good baseline rather than a description of
> current performance, and re-collect before using it to argue a regression
> either way. **This does not apply to the JPEG Decode, AWS cloud baseline,
> or Hardware decoders sections below** — those are current-code captures
> (see their own capture dates) and supersede the older, narrower `Image
> Codec Decode` section further down this file (marked superseded in place).

---

## Overview

This document tracks EdgeFirst HAL performance across target platforms. It serves as a regression baseline: results are updated with each release to detect performance improvements or regressions introduced by code changes.

The benchmarking strategy tests **all compute backends** (CPU, OpenGL, G2D) with **all applicable buffer strategies** (DMA-buf, PBO, Sync) on every platform, including forcing non-default buffer paths on platforms that would normally prefer a different strategy. This ensures the full fallback chain is exercised and performance characteristics are understood for every deployment scenario.

## Optimization Performance Reference

This section is the **benchmark-level reference** for the
[Optimization Guide in README.md](README.md#optimization-guide). The README
states the rules, [ARCHITECTURE.md] explains the mechanism behind each rule,
and the table below quantifies the cost of breaking it on each platform.

| Rule (from README) | Benchmark | Cost when broken |
|--------------------|-----------|------------------|
| Reuse tensors across frames | [§ Tensor Reuse Impact](#tensor-reuse-impact) | i.MX 95 (Mali): **3.3×**; i.MX 8MP (Vivante): **1.7×**; x86 PBO: 1.0× |
| Cache imported camera tensors by inode | [§ Tensor Reuse Impact](#tensor-reuse-impact) (recreate variant); see also [ARCHITECTURE.md § Appendix C][arch-appendix-c] | Equivalent to recreating the source tensor every frame: 3.5 ms penalty per `convert()` on i.MX 95; 2.2 ms on i.MX 8MP |
| Allocate via `ImageProcessor::create_image()` | [§ Image Preprocessing: Letterbox Pipeline](#image-preprocessing-letterbox-pipeline-camera--model-input) | Forced wrong-backend transfer adds the cost of a `glTexSubImage2D` upload (≈full conversion time) on every frame |
| Build the decoder once | [§ Decoder Post-Processing](#decoder-post-processing) | Decoder construction parses the model schema and allocates working buffers — cost depends on output schema complexity |
| One `ImageProcessor` per pipeline thread | [image/ARCHITECTURE.md § GL Concurrency Model][gl-mutex] | Driver-dependent: on Vivante (i.MX 8MP) and paravirtual GPUs concurrent `convert()` calls serialize through the global `GL_MUTEX` (throughput drops toward single-threaded regardless of core count); on Mali, V3D, Tegra, and real Apple GPUs they run concurrently. One processor per thread is the portable rule. |
| Native CPU feature builds (Rule 6) | [§ materialize_masks Batched-GEMM Optimisation](#materialize_masks-batched-gemm-optimisation) | Soft-float f16 helpers (`__extendhfsf2`) are measurably slower than native `fcvt` / `vcvtph2ps` on the mask kernel hot path; the exact factor depends on vector width and CPU. Verify with `scripts/audit_f16_codegen.sh`. |
| Pass numpy arrays straight to `from_numpy()` (Rule 7) | [§ NumPy Interop Fast-Path](#numpy-interop-fast-path) | A redundant `np.ascontiguousarray` pre-copy on every call. Sized example: `(1, 116, 8400)` f32 transposed view on rpi5-hailo runs ≈ 6.5 ms in HAL's automatic fast path vs ≈ 27 ms in the legacy element-wise loop (4× faster); pre-applying `ascontiguousarray` above HAL adds a redundant copy of the same magnitude. |
| Use `MaskResolution::Scaled` for COCO eval (Rule 8) | [§ materialize_masks Batched-GEMM Optimisation](#materialize_masks-batched-gemm-optimisation) | Threshold-then-upsample (`Proto` followed by binary `cv2.resize`) regresses mask mAP by 0.04–0.05 absolute on YOLOv8-seg / `coco128-seg`. The `Scaled` path is also faster at N ≥ 16 because the batched GEMM amortises across detections instead of being repeated per-detection in caller code. |

[ARCHITECTURE.md]: ARCHITECTURE.md
[gl-mutex]: crates/image/ARCHITECTURE.md#gl-concurrency-model-serialization-policy
[arch-appendix-c]: ARCHITECTURE.md#appendix-c-dma-buf-identity-and-tensor-caching

### How to Reproduce the Numbers

The empirical penalties above all come from `bench_preproc` (the C
preprocessing benchmark). It deliberately measures three variants of the
same pipeline:

| Variant | What it does | Maps to README rule |
|---------|--------------|--------------------|
| `reuse` | Single source tensor held alive for all 100 frames | Rule 1 followed |
| `recreate` | Source tensor freed and reallocated every frame | Rule 1 broken (or Rule 3 broken with fd recycling) |
| `pool` | Round-robin through 4 pre-allocated source tensors | Rule 1 followed with multiple in-flight buffers (V4L2 pool simulation) |

`pool` matches `reuse` to within 4% on every embedded platform, confirming
that the EGL image cache scales correctly with pool depth. `recreate` is
the failure mode that an inode-keyed cache (Rule 3) prevents.

See [§ Running `bench_preproc`](#running-bench_preproc) below for the
build and deployment commands. See [TESTING.md § Validating Optimizations][test-opt]
for how to verify your own integration follows each rule.

[test-opt]: TESTING.md#validating-optimizations

---

## Benchmarking Strategy

### Compute Backends

Each benchmark category runs across all available **compute backends**:

| Compute Backend | Description | Platforms |
|----------------|-------------|-----------|
| **CPU** | Pure software using vectorized operations + Rayon parallelism | All |
| **OpenGL** | GPU-accelerated via OpenGL ES shader pipeline | Linux with EGL |
| **G2D** | NXP 2D hardware blitter (Vivante) | NXP i.MX Family |

Future backends may include OpenCL, Vulkan, and other vendor-specific 2D accelerators.

### Buffer Strategies

Orthogonally, each compute backend operates on buffers using different memory and transfer strategies:

| Buffer Strategy | Tensor Type | GPU Transfer Method | When Used |
|----------------|-------------|-------------------|-----------|
| **DMA-buf** | `DmaTensor` | `EGL_EXT_image_dma_buf_import` (zero-copy) | Linux with DMA-heap + compatible GPU driver |
| **PBO** | `PboTensor` | `GL_PIXEL_UNPACK/PACK_BUFFER` (zero-copy GL binding) | OpenGL ES 3.0 when DMA-buf roundtrip fails |
| **Sync** | `MemTensor` | `glTexImage2D` / `glReadnPixels` (memcpy) | Fallback when PBO unavailable |
| **Heap** | `MemTensor` | N/A (CPU-only, no GPU transfer) | CPU backend, or non-Linux platforms |
| **SHM** | `ShmTensor` | N/A (IPC sharing) | Cross-process sharing |

**Backend × buffer combinations benchmarked:**

| Compute Backend | DMA-buf | PBO | Sync | Heap |
|----------------|---------|-----|------|------|
| **OpenGL** | Yes (preferred) | Yes (fallback) | When PBO unavailable | — |
| **G2D** | Yes (required) | — | — | — |
| **CPU** | — | — | — | Yes |

Typically we benchmark DMA-buf and PBO for GPU backends. The Sync (upload/readpixels) path is only benchmarked when PBO is not supported on a platform.

### NV12 / NV16 / NV24 Conversion Paths (sampler vs shader)

Semi-planar YUV (NV12/NV16/NV24) → RGB on the OpenGL backend has two GPU paths,
selectable via the **`EDGEFIRST_NV_CONVERT_PATH`** environment variable
(`sampler` | `shader` | `auto`, default `auto`):

| Path | `NvConvertPath` | Mechanism | YUV→RGB by | Notes |
|------|-----------------|-----------|-----------|-------|
| **Sampler** | `ExternalSampler` | `samplerExternalOES` EGLImage | the **GPU driver** | NV12 only (incl. multiplane). Colorimetry & chroma upsampling are the driver's. |
| **Shader** | `ShaderR8` | R8 `texelFetch` + in-shader matrix | **HAL** (exact, per-tensor) | NV12/NV16/NV24, single-plane (combined buffer). Portable & identical across GPUs. |

On a **non-DMA** backend (PBO/Sync, e.g. orin) `ShaderR8` uploads the combined
buffer as an R8 texture (`glTexImage2D`) and runs the same shader — so the GPU
NV path is available even without DMA-buf EGLImage import. True-multiplane NV12
(separate Y/UV fds) has no single-buffer R8 view, so it always uses `Sampler`.

**Correctness (per-pixel Δ vs CPU reference, on-target):** the matrix matches on
every GPU (solid frames ≤1); the divergence is **chroma upsampling**. `Sampler`
uses the driver's *bilinear* chroma (≤55 at chroma edges on V3D & Mali);
`ShaderR8` uses nearest/replicate and matches CPU (≤2). Vivante's sampler is
nearest-like and also matches CPU.

**Latency (NV12 720p convert, median, on-target A/B via `nv_path_benchmark`):**

| Platform | GPU | Sampler | Shader | Selected by `auto` |
|----------|-----|---------|--------|--------------------|
| rpi5-hailo | V3D | 2.1 ms | 2.4 ms | **Shader** (correct, ~equal speed) |
| imx95-frdm | Mali-G310 | 1.5 ms | 2.3 ms | **Shader** (chroma correctness) |
| imx8mp-frdm | Vivante GC7000UL | **2.5 ms** | **29.2 ms** | **Sampler** (shader ~12× slower; sampler also correct here) |
| jetson-orin-nano | Tegra (NVIDIA) | 2.4 ms | 2.1 ms | **Shader** (R8 upload; no DMA-buf import) |

**`auto` policy (HIGH-PERFORMANCE default, issue #106):** prefer `ShaderR8`
(portable, colorimetry-exact) wherever it is also the fast path — every GPU
above except Vivante. On **Vivante**, single-plane 4-aligned NV12 takes
`ExternalSampler` for **every** colorimetry in the default
`ColorimetryMode::Fast`: the driver applies its fixed BT.601-limited matrix,
which is exact for BT.601-limited sources and approximate for the rest — the
12× speed gap (2.5 ms vs 29 ms) is the trade. Opt in to exactness with
`ImageProcessorConfig::colorimetry = ColorimetryMode::Exact` or
`EDGEFIRST_COLORIMETRY=exact`: the sampler is then used only when the driver
matrix matches the source's resolved (encoding, range) exactly.
`EDGEFIRST_NV_CONVERT_PATH` still force-overrides the path for benchmarking
and platform bring-up.

Source-width constraint for the `Sampler` (NV12 EGLImage) path: **even width**
(4:2:0 chroma is W/2). The import uses the 64-byte-aligned row pitch, so widths
that are even but not 4-aligned (e.g. 1282) take the zero-copy sampler path; a
driver that still rejects falls back automatically.

### Buffer Infrastructure Benchmarks

In addition to compute benchmarks, we separately measure:
- **Allocation latency** — `Tensor::new()` for each buffer type (DMA, SHM, Mem, PBO)
- **Map/unmap latency** — `tensor.map()` for each buffer type
- **Memcpy throughput** — read/write bandwidth for mapped buffers

These infrastructure benchmarks isolate the memory subsystem overhead from the compute backend performance.

### Benchmark Categories

1. **Buffer Infrastructure** — Allocation, mapping, and memcpy latency per buffer type
2. **Image Preprocessing** — Camera-to-model pipeline (format conversion + resize + letterbox)
3. **Format Conversion** — Same-size format conversion (no geometric transform)
4. **Resize** — Geometric resize with optional rotation/flip
5. **Post-processing** — Model output decoding (detection, segmentation, NMS)
6. **Mask Rendering** — Segmentation mask materialization and overlay
7. **End-to-End Pipeline** — Full camera → preprocess → decode → render cycle

### Standard Test Configurations

**Input resolutions:**
- 720p (1280×720) — lower-resolution cameras
- 1080p (1920×1080) — standard cameras
- 4K (3840×2160) — high-resolution cameras

**Model input sizes:**
- 640×640 — standard resolution models
- 1280×1280 — high-resolution models

**Source formats:** YUYV, VYUY, NV12, NV16, RGBA, RGB, GREY
**Destination formats:** RGBA, BGRA, RGB, GREY, PlanarRgb (8BPS)
**Output dtypes:** u8 (default), i8 (int8 quantized model input — XOR 0x80 bias)

### Format Abbreviations

| Benchmark Name | PixelFormat | DType | Description |
|---------------|-------------|-------|-------------|
| **RGBA** | `PixelFormat::Rgba` | `U8` | 4-channel packed RGBA |
| **RGB** | `PixelFormat::Rgb` | `U8` | 3-channel packed RGB |
| **8BPS** | `PixelFormat::PlanarRgb` | `U8` | 3× separate u8 planes (R, G, B) |
| **RGB_i8** | `PixelFormat::Rgb` | `I8` | Packed RGB with XOR 0x80 bias |
| **8BPS_i8** | `PixelFormat::PlanarRgb` | `I8` | Planar RGB with XOR 0x80 bias |

### Measurement Methodology

All benchmarks use the `edgefirst-bench` custom harness which:
- Runs in-process (no fork) to avoid GPU driver crashes
- Executes warmup iterations (unmeasured) followed by measured iterations
- Reports: median, mean, min, max, p95, p99
- Reports throughput in MiB/s where applicable

**Standard parameters:** 10 warmup iterations, 100 measured iterations (adjustable per benchmark).

**Table notation:** **bold** = fastest backend for this conversion; `—` = data not collected; `N/A` = not supported by this backend; `BLOCKED` = actively disabled due to hardware bug (see Known Issues).

> **Tip:** Use the HAL's built-in [Performance Tracing](README.md#performance-tracing)
> to capture per-call timing in your actual pipeline. Benchmarks measure
> isolated operations; traces reveal how those operations compose and where
> time is spent in real workloads. See
> [ARCHITECTURE.md § Performance Tracing Architecture](ARCHITECTURE.md#performance-tracing-architecture)
> for the recommended perf + tracing workflow.

---

## Running Benchmarks

See [README.md § Benchmarking](README.md#benchmarking) for full instructions on running benchmarks locally, cross-compiling for aarch64, and deploying to target platforms.

### Benchmark Binaries

| Binary | Crate | What It Measures |
|--------|-------|-----------------|
| `tensor_benchmark` | `edgefirst-tensor` | Tensor allocation and map/unmap latency across buffer types (Heap, SHM, DMA) |
| `image_benchmark` | `edgefirst-image` | JPEG loading, format convert, resize operations across buffer backends |
| `pipeline_benchmark` | `edgefirst-image` | Letterbox pipeline and format conversion (camera→model input) |
| `nv_path_benchmark` | `edgefirst-image` | NV12/16/24 sampler-vs-shader A/B (set `EDGEFIRST_NV_CONVERT_PATH=sampler\|shader`); synthesized sources, no testdata |
| `decode_pipeline_benchmark` | `edgefirst-image` | JPEG decode → letterbox convert end-to-end (strided input, HWC/CHW output) |
| `nvjpeg_benchmark` | `edgefirst-image` | nvJPEG GPU decode into CUDA-backed PBO (`codec/jpeg/nvjpeg/rgbi/*` cells); on-target only, skips cleanly without CUDA |
| `mask_benchmark` | `edgefirst-image` | Mask rendering: draw_decoded_masks, draw_proto_masks, hybrid path |
| `mask_decode_benchmark` | `edgefirst-image` | Focused mask decode on the RETINA (scaled) path |
| `convert_matrix_benchmark` | `edgefirst-image` | GL convert across the memory × format matrix |
| `batch_convert_benchmark` | `edgefirst-image` | Deferred view-tile converts vs. eager, one tile at a time |
| `tiled_convert_benchmark` | `edgefirst-image` | Confirms CPU convert cost scales with the requested crop area, not the source frame area |
| `cpu_preprocess_benchmark` | `edgefirst-image` | CPU JPEG-preprocessing path, profiled against the Jetson Orin Nano |
| `parallel_processors_benchmark` | `edgefirst-image` | Multi-processor GL throughput scaling; shows where the serialization policy binds |
| `opencv_benchmark` | `edgefirst-image` | OpenCV baseline comparison for same operations |
| `decoder_benchmark` | `edgefirst-decoder` | YOLO detection/segmentation post-processing, NMS, dequantization |
| `codec_benchmark` | `edgefirst-codec` | JPEG/PNG decode into pre-allocated tensors vs. image crate and zune-png; NEON tiers on AArch64, `IntelTier` (SSE2/SSE4.1/AVX2) on x86-64, vectorised type conversion |
| `tracker_benchmark` | `edgefirst-tracker` | ByteTrack association and track lifecycle |

`crates/image` also carries a `sanity_check` binary. It is an adaptive smoke check
over letterbox format combinations, not a benchmark, and feeds no results table.

JSON files are collected in `benchmarks/<platform>/` and processed by `.github/scripts/generate_benchmark_tables.py` to produce the tables in this document.

---

## Target Platforms

### maivin (Torizon 7)

| Property | Value |
|----------|-------|
| **Platform ID** | `maivin` |
| **SoC** | NXP i.MX 8M Plus Industrial Variant |
| **CPU** | 4× Cortex-A53 @ 1.6 GHz |
| **GPU** | Vivante GC7000UL (OpenGL ES 3.1) |
| **RAM** | 4 GB LPDDR4 |
| **OS** | Torizon OS 7 (Linux 6.6) |
| **G2D** | Yes (Vivante) |
| **DMA-buf** | Yes (CMA) |
| **Notes** | Primary production target; G2D + OpenGL + DMA-buf all available |

### imx8mp-frdm

| Property | Value |
|----------|-------|
| **Platform ID** | `imx8mp-frdm` |
| **SoC** | NXP i.MX 8M Plus |
| **CPU** | 4× Cortex-A53 @ 1.6 GHz |
| **GPU** | Vivante GC7000UL (OpenGL ES 3.1) |
| **RAM** | 2 GB LPDDR4 |
| **OS** | NXP BSP (Linux 6.12) |
| **G2D** | Yes (Vivante) |
| **DMA-buf** | Yes (CMA) |
| **Notes** | NXP evaluation board; same SoC as maivin, latest NXP BSP. **NV12→planar blocked on GL** (GPU hang, see Known Issues §9). |

### imx95-frdm

| Property | Value |
|----------|-------|
| **Platform ID** | `imx95-frdm` |
| **SoC** | NXP i.MX 95 |
| **CPU** | 6× Cortex-A55 @ 1.8 GHz |
| **GPU** | Mali G310 (Panfrost, OpenGL ES 3.1) |
| **RAM** | 8 GB LPDDR5 |
| **OS** | NXP BSP (Linux 6.12) |
| **G2D** | Yes (NXP PXP backend) |
| **DMA-buf** | Yes (CMA) |
| **Notes** | Next-gen NXP platform; Mali GPU replaces Vivante |

### jetson-orin-nano

| Property | Value |
|----------|-------|
| **Platform ID** | `jetson-orin-nano` |
| **SoC** | NVIDIA Jetson Orin Nano |
| **CPU** | 6× Cortex-A78AE @ 1.5 GHz |
| **GPU** | NVIDIA Ampere (1024 CUDA cores) |
| **RAM** | 8 GB LPDDR5 |
| **OS** | JetPack / L4T |
| **G2D** | No |
| **DMA-buf** | Yes (system heap, DMA roundtrip may fail — PBO path likely) |
| **Notes** | NVIDIA EGL may not import DMA-buf from system heap; PBO path expected |

### rpi5-hailo

| Property | Value |
|----------|-------|
| **Platform ID** | `rpi5-hailo` |
| **SoC** | Broadcom BCM2712 |
| **CPU** | 4× Cortex-A76 @ 2.4 GHz |
| **GPU** | VideoCore VII (OpenGL ES 3.1 via Mesa V3D) |
| **RAM** | 8 GB LPDDR4X |
| **OS** | Raspberry Pi OS (Debian 12) |
| **G2D** | No |
| **DMA-buf** | Yes (system heap) |
| **Notes** | Mesa V3D driver; RGB/RGB_i8 packed GL via two-pass packing shader |

### x86-desktop

| Property | Value |
|----------|-------|
| **Platform ID** | `x86-desktop` |
| **SoC** | — |
| **CPU** | (varies — document specific CPU at collection time) |
| **GPU** | NVIDIA (desktop, proprietary driver) |
| **RAM** | (varies) |
| **OS** | Ubuntu 24.04+ |
| **G2D** | No |
| **DMA-buf** | DMA allocation works but NVIDIA EGL cannot import — PBO path |
| **Notes** | Development platform; PBO backend primary, DMA-buf roundtrip fails |

### mbp-m2-max

| Property | Value |
|----------|-------|
| **Platform ID** | `mbp-m2-max` |
| **Model** | MacBook Pro Mac14,5 |
| **SoC** | Apple M2 Max |
| **CPU** | 12 cores (8 Performance + 4 Efficiency, ARMv8.6-A, NEON + dotprod + i8mm + FP16) |
| **GPU** | Apple integrated 38-core (M2 Max), driven via ANGLE → Metal |
| **RAM** | 32 GB unified memory |
| **OS** | macOS 26+ (`aarch64-apple-darwin`) |
| **G2D** | No (Linux-only) |
| **DMA-buf** | No (Linux-only); HAL maps `TensorMemory::Dma` onto IOSurface for zero-copy bind |
| **GL Transfer Backend** | IOSurface (zero-copy via `EGL_ANGLE_iosurface_client_buffer`) |
| **Notes** | Apple Silicon developer platform. ANGLE supplies `libEGL.dylib` and `libGLESv2.dylib` — either the signed EdgeFirst release via `scripts/fetch-angle.sh` or the Homebrew tap — and translates GLES 3.0 → Metal, so the same shader source used on Linux GPUs runs unchanged. Since 0.25.0 macOS runs the shared GL engine and inherits its full conversion matrix; the GL rows in the tables below predate that and still reflect the old YUYV→RGBA-only backend (Known Gap #17). |

---

## JPEG Decode: EdgeFirst vs libjpeg-turbo

**Scope:** JPEG bytes → decoded raster in memory. Letterbox / model-input
preprocessing (`ImageProcessor::convert()`) is a separate measurement and is not
included here. Eight arms:

| Arm | Command |
|-----|---------|
| **EdgeFirst** | `hal_cpu --decode-only` — accurate `islow`-class IDCT, the default |
| **EdgeFirst `fast`** | same, `EDGEFIRST_CODEC_DCT=fast` — opt-in AAN `ifast`-class IDCT (`DctMethod::Fast`) |
| **Turbo `islow`** | `turbojpeg_bench --dct accurate` — libjpeg-turbo's default IDCT |
| **Turbo `ifast`** | `turbojpeg_bench --dct fast` |
| **zune-jpeg** | `rust_jpeg --engine zune` — YCbCr out on the YUV arm, RGB on the RGB arm; built with its `x86` **and `neon`** SIMD features (both default-on in zune-jpeg 0.5.15, pinned explicitly in the harness), so its NEON IDCT / colour-convert / upsampling kernels are active on the ARM boards — confirmed by a direct neon-on/off A/B (1.16–1.20× on rpi5-hailo), not just the features table; see § JPEG Decode's "Against the Rust ecosystem" bullet |
| **image crate** | `rust_jpeg --engine image` — **zune-jpeg behind the `image` 0.25 wrapper**, not an independent decoder: `load_from_memory` + `to_rgb8` (allocates per call; RGB only, its API exposes no raw-YUV output). Its row measures the wrapper's per-call allocation and colour handling on top of zune |
| **stb_image** | `stb_bench` — v2.30 single-header baseline (public domain/MIT); fixed accurate-class IDCT with SSE2 (auto) / NEON (enabled) SIMD; RGB only, allocates per call (its API has no decode-into). Pixel parity vs djpeg accurate: max abs diff 3, mean ≤0.07 |
| **Wuffs** | `wuffs_bench` — Google's memory-safe decoder, v0.4.0-alpha.10 transpiled C (Apache-2.0); single fixed IDCT, **bit-exact** vs djpeg accurate (max abs diff 0 on every parity sample); RGB only (3-byte, via its swizzler — measurably its slow path, see the "floor" bullet below), high-water buffer reuse |

The YUV arm stops after decode into a YUV layout with no RGB colour step
(EdgeFirst `--decode-fmt native` NV12/16/24, turbo `tjDecompressToYUV2`
planar, zune interleaved YCbCr); the RGB arm decodes to interleaved RGB
(EdgeFirst's fused MCU write, `TJPF_RGB`, zune/image RGB). The YUV output
layouts are the right conceptual comparison — every arm stops in YCbCr —
but they are not byte-identical: turbo writes fully-planar, row-padded
Y/Cb/Cr planes while EdgeFirst writes semi-planar NV formats, so the write
patterns differ even though neither pays a colour conversion. COCO `val2017`
is 4:4:4, so the fused RGB write engages. zune's YUV arm skips COCO's few
greyscale images (its Luma→YCbCr mapping is unimplemented); the other arms
decode them. Quantified rather than assumed: val2017 has exactly 10
greyscale files out of 5000, and the harness's evenly-spaced `n=200` sampling
(`list_jpegs`: stride `len/n` across the sorted directory) selects **zero**
of them — verified by replicating the sampling against the full corpus. Every
arm's n=200 YUV sample is the same 200 images; the skip has no effect on any
number in this document.

All arms are native binaries — `hal_cpu` / `rust_jpeg` (Rust) and the C
arms `modules/turbojpeg/bench.c`, `stb_bench`, and `wuffs_bench` (the
latter two built on the shared `modules/cbench/cbench.h` harness header) —
measured by harnesses that agree on image selection, preload before timing,
`CLOCK_MONOTONIC` around decode alone, percentile index, median-CI ranks,
and MP/s. A Python driver on one side only would put interpreter dispatch
and FFI marshalling inside the timed region.

Buffer policy is matched wherever each API allows it: EdgeFirst decodes into
a reused tensor pool, turbo into a reused high-water output buffer, and zune
via `decode_into` into a reused buffer — no arm pays a steady-state
allocation per frame. The `image` arm allocates per call by design; that is
the wrapper overhead its row exists to show. Every arm decodes
single-threaded on one pinned core (zune-jpeg itself is single-threaded —
the multi-core post-processing zune-image advertises is not in play),
matching libjpeg-turbo's one-thread-per-image model, and timing stops at the
decoded raster, the same decode-only convention `tjbench` uses. mozjpeg has
no row because its decoder *is* libjpeg-turbo's (it is an encoder-focused
fork); Intel IPP has no row because Intel discontinued its JPEG codec sample
(UIC), leaving no supported IPP JPEG decoder to benchmark short of
hand-assembling one from primitives.

**On the IDCT accuracy classes.** libjpeg-turbo decompresses with either the
accurate `islow` IDCT or the faster, lower-accuracy `ifast` one; **`islow` is
its default**, and it is the kernel EdgeFirst's default implements, so
EdgeFirst↔`islow` is the like-for-like comparison and the one claims should
quote. EdgeFirst's opt-in `fast` mode is the same trade turbo's `ifast`
makes, so `fast`↔`ifast` is the like-for-like comparison within the fast
class. The measured `islow`↔`ifast` gaps (5–11%) sit inside libjpeg-turbo's
own documented range — its `doc/libjpeg.txt` puts `ifast` "generally about
5-15% faster" than `islow` on non-AVX2 CPUs and "similar" on AVX2, and its
3.0 ChangeLog calls the fast algorithms "a legacy feature" — so a reader
remembering a larger historical `ifast` advantage is remembering older
hardware. Measured on 1000 COCO images (`dct_compare`), EdgeFirst `fast` vs
the default kernel: cosine similarity mean 0.99998 / worst 0.99985, PSNR
mean 51.4 dB / worst 42.1 dB, max pixel delta 24. mAP impact is not yet
measured; `fast` stays opt-in.

Re-captured 2026-08-13 under the updated protocol (`decode-ab-sweep.sh`,
`CARGO_PROFILE=release`, no perf/trace): three interleaved rounds per host
with all arms alternating inside each round, pinned to one core, n=200 drawn
evenly spaced across the corpus by **every** arm; the quoted number is the
**median across rounds' p50s**, with each round's p50, the min–max spread,
the mean, and a nonparametric 95% CI on the median recorded in the run
outputs, and per-host build/run `provenance.txt` (toolchains, comparator
library path/version, governor, clocks) written next to them. Decoder state:
register-cached bit cursor, fast-AC terminator baking, entropy-derived IDCT
tiers, paired-coefficient probe (High tier), dedicated 4:4:4 MCU loop,
SSE4.1 fused-RGB block kernel. The orin-nano row was captured on the
fallback unit (`adis-uav1`, Orin Nano Super devkit; the prior capture
established its turbo baseline matches orin-nano within 0.1%). x86-desktop
is `sebstation` (same i9-11900K host as prior x86 captures). mbp-m2-max is
this Apple Silicon MacBook itself, run in-process rather than over SSH (the
board label has no separate host); macOS has no `taskset`-equivalent used
here, so its row runs unpinned — the likely reason its Max spread column
runs a bit higher than the pinned Linux boards. No run failed. Round-to-round
spread is mostly sub-1% but not uniformly — see the "Max spread" column
below (worst observed: 7.7%, mbp-m2-max RGB `image`, unpinned; worst pinned:
5.5%, rpi5-hailo RGB `tj_ifast`). The
headline `EdgeFirst`/`Turbo islow` cells are not immune: `hal`'s own worst
spread across every board/format cell in this table is 7.3% (mbp-m2-max RGB,
unpinned — no `taskset`-equivalent restrains the scheduler on macOS, so a
round can land on a different core mix); on the pinned Linux boards `hal`'s
worst is 4.5% (rpi5-hailo YUV, a single high first round — 2.464 ms vs
2.357/2.372 ms on rounds 2–3, most likely cold-cache/first-touch rather than
genuine measurement noise, since every other arm's round 1 on that same
host/corpus is unremarkable); `tj_islow`'s worst is 7.0% (mbp-m2-max RGB,
same unpinned cause) / 2.3% pinned (imx8mp-frdm YUV). None of this changes
which arm wins a cell — the gaps between arms are far larger than any
observed spread — but it does mean single-round numbers would occasionally
overstate a lead; the published figures are always the 3-round median for
exactly this reason.

All numbers are p50 ms; lower is better. **Bold** marks the fastest arm in
each accuracy class (accurate: EdgeFirst vs turbo `islow`; fast: EdgeFirst
`fast` vs turbo `ifast`).

Max round-to-round spread (`(max−min)/median` across all 3 rounds, worst arm
in that row) is reported per row rather than per cell, to keep the table
readable — see the per-cell `spread=[lo,hi]` values in each host's
`decode-ab-sweep/val2017/summary.txt` for the full breakdown.

| Board | Core | Arm | EdgeFirst | EF `fast` | Turbo `islow` | Turbo `ifast` | zune-jpeg | image | stb | Wuffs | Max spread |
|-------|------|-----|-----------|-----------|---------------|---------------|-----------|-------|-----|-------|------------|
| rpi5-hailo | A76 | YUV | **2.372** | **2.072** | 3.173 | 2.900 | 4.115 | — | — | — | 4.8% (tj_ifast) |
| | | RGB | **2.574** | **2.263** | 3.356 | 3.110 | 4.411 | 4.939 | 5.423 | 7.312 | 5.5% (tj_ifast) |
| orin-nano | A78AE | YUV | **3.719** | **3.178** | 5.144 | 4.627 | 6.404 | — | — | — | 1.0% (hal) |
| | | RGB | **4.012** | **3.461** | 5.491 | 4.984 | 6.834 | 8.385 | 7.859 | 9.884 | 1.2% (zune) |
| x86-desktop | Rocket Lake i9-11900K | YUV | **1.222** | 1.221 | 1.433 | 1.364 | 1.911 | — | — | — | 1.2% (tj_ifast) |
| | | RGB | **1.323** | 1.321 | 1.504 | 1.432 | 1.932 | 2.041 | 2.511 | 2.726 | 4.3% (stb) |
| imx95-pro | A55 | YUV | **6.032** | **5.270** | 7.044 | 6.576 | 11.457 | — | — | — | 1.3% (hal_fast) |
| | | RGB | **6.340** | **5.601** | 7.558 | 7.057 | 12.006 | 12.986 | 13.206 | 21.389 | 0.9% (image) |
| imx8mp-frdm | A53 | YUV | **6.745** | **5.941** | 7.747 | 6.954 | 13.504 | — | — | — | 2.3% (tj_islow) |
| | | RGB | **7.073** | **6.359** | 7.934 | 7.305 | 13.878 | 14.607 | 14.890 | 23.266 | 2.3% (zune) |
| mbp-m2-max | Apple M2 Max | YUV | **1.152** | 1.066 | 1.623 | 1.560 | 2.141 | — | — | — | 4.6% (zune) |
| | | RGB | **1.220** | 1.110 | 1.705 | 1.557 | 2.231 | 2.534 | 3.169 | 3.419 | 7.7% (image) |

- **EdgeFirst's accurate default beats turbo's `islow` everywhere** (ratio =
  turbo `islow` ÷ EdgeFirst, YUV / RGB) — **1.15× / 1.12×** on the A53,
  **1.17× / 1.19×** on the A55, **1.34× / 1.30×** on the A76, **1.38× /
  1.37×** on the A78AE, **1.17× / 1.14×** on Rocket Lake, and **1.41× /
  1.40×** on the M2 Max — EdgeFirst's largest accurate-class lead on any
  board in the eight-arm sweep. It also beats turbo's **`ifast`** on every
  platform (1.03× on the A53 … 1.35× on the M2 Max, YUV) while producing
  `islow`-class pixels.
- **The opt-in `fast` mode extends the lead within the fast class** (ratio =
  turbo `ifast` ÷ EdgeFirst `fast`, YUV / RGB): **1.17× / 1.15×** on the A53,
  **1.25× / 1.26×** on the A55, **1.40× / 1.37×** on the A76, **1.46× /
  1.44×** on the A78AE, **1.46× / 1.40×** on the M2 Max. On x86 there is no
  fast kernel yet — the option is advisory and runs the accurate path (the
  `fast` and default rows measure the same code there); the M2 Max NEON fast
  kernel is fully active, unlike Rocket Lake's no-op.
- **Against the Rust ecosystem**, EdgeFirst is 1.6× faster than zune-jpeg
  on Rocket Lake, 1.7× on the A76/A78AE, 1.8× on the M2 Max, and 1.9–2.0× on
  the in-order A55/A53 — and the ARM rows are measured against zune's **NEON
  build**.
  This is now a measured claim, not a features-table inference: zune-jpeg's
  own docs describe disabling SIMD via *"disable the `x86` feature"*, with no
  equivalent sentence for `neon`, so a `neon`-vs-no-`neon` A/B was run
  directly (a standalone harness linking only `zune-jpeg` — not through the
  `image` crate's wrapper, whose own `zune-jpeg` dependency requests default
  features and would otherwise unify `neon` back in regardless of what a
  same-binary comparison built) on rpi5-hailo (A76), same
  `DecoderOptions::default()` call the `zune` arm itself makes (so the
  harness is not incidentally tripping zune's `set_use_unsafe(false)` path
  either), n=200, median-of-3: **YUV 4.871 ms → 4.211 ms (neon 1.16× faster,
  −13.6%)**, **RGB 5.232 ms → 4.345 ms (neon 1.20× faster, −16.9%)**. NEON is
  demonstrably engaged and buys a real, moderate speedup — not the ~2%
  "maybe it's a no-op" floor, not a 25–40% ceiling either — so the ARM
  comparison is against a genuine SIMD build, and the widening in-order gap
  against EdgeFirst is scheduling and tuning, not a scalar-vs-SIMD artifact.
  The image crate (zune-jpeg internally, plus a per-call allocation and RGB
  conversion) trails further.
- **stb_image and Wuffs are the floor, as expected**: stb runs 1.9–2.1×
  behind EdgeFirst's RGB arm across hosts, and Wuffs 2.1× (x86) to 3.4×
  (A55), its memory-safety cost growing with core simplicity and image
  size. Both decode correctly — Wuffs bit-exactly matches the accurate
  reference and stb sits within ±3 LSB (see the arm table) — so their rows
  are like-for-like within the accurate class. **Wuffs' RGB row specifically
  is measured on its slow path, not a broken benchmark**: the default probe
  order prefers 3-byte RGB (matching the other RGB arms) via Wuffs' own
  swizzler, but Wuffs is built around 4-byte pixel buffers — the 3-byte
  path is not its fast path. Forcing the native 4-byte destination
  (`EDGEFIRST_WUFFS_FORCE_4BPP=1`, added to `wuffs_bench` for this
  measurement) confirms it: **1.52×** faster on rpi5-hailo/A76 (7.310 ms →
  4.816 ms, n=200) and **1.59×** faster on imx8mp-frdm/A53 (23.201 ms →
  14.573 ms, n=200). The 3-byte row is kept as the headline number because
  it is the like-for-like comparison against every other RGB arm's output
  layout, but a caller free to take 4-byte RGBA/BGRA from Wuffs should
  expect roughly a third to a half off the numbers in this table.
- **The DRI (restart-marker) claim is now measured, not asserted.** COCO
  val2017 itself has zero DRI files (all 5000 checked with
  `corpus_stats.py`: 4990 baseline 4:4:4, 10 greyscale, none progressive),
  so `make-dri-corpus.sh` re-serialised it **losslessly** (`jpegtran
  -restart 1` — identical DCT coefficients, only restart markers added). This
  penalty (own-DRI-time ÷ own-non-DRI-time, a self-relative overhead — not a
  competitor ratio, so kept as a percentage) is what enabling restart markers
  costs each decoder on its own baseline: turbo `islow` pays +5.4% (x86),
  +10.4% (A76), +11.7% (A78AE), +15.6% (A55), +15.7% (A53) — approaching the
  ≤20% its own README documents for the disabled fast-Huffman path — while
  EdgeFirst pays ≤1.2% on every host (x86 is the worst case at 1.1–1.2%; the
  other four hosts are ≤1.0%). EdgeFirst's accurate-class lead therefore
  grows on restart-carrying camera streams (ratio = turbo `islow` ÷
  EdgeFirst, both on the DRI corpus, YUV): **1.22×** on x86, **1.46×** on the
  A76, **1.53×** on the A78AE, **1.35×** on the A55, **1.33×** on the A53.

### Control corpora

The same eight-arm sweep ran over five control corpora (see
`benchmarks/README.md` § Corpora for recipes and licensing; per-run
summaries, CIs, and provenance live under
`benchmarks/results/<board>/decode-ab-sweep/<corpus>/` from the capture):
`val2017-yuv420` (the same 5000 images transcoded to 4:2:0), `val2017-dri`
(lossless restart-marker isolate, discussed above), and the CLIC 2025
large-image set (62 files, 1.8–4.2 MP, p50 2.8 MP) encoded at 4:2:0 and
4:4:4 from identical pixels plus a 4:2:0 DRI variant.

**The headline gap is not corpus-driven.** The specific objection these
controls answer: zune-jpeg's deficit against turbo on x86 could be an
artifact of COCO being 4:4:4 and ~0.27 MP. Measured on the i9-11900K (YUV
arm, p50 ms):

| Corpus | Turbo `islow` | zune-jpeg | zune deficit |
|--------|---------------|-----------|--------------|
| val2017 (4:4:4, 0.27 MP) | 1.433 | 1.911 | 1.33× |
| val2017-yuv420 (same images, 4:2:0) | 0.748 | 1.137 | **1.52×** |
| CLIC 4:2:0 (2.8 MP) | 5.555 | 8.697 | **1.57×** |
| CLIC 4:4:4 (2.8 MP) | 8.824 | 11.709 | 1.33× |

zune never closes to within 10% of turbo on any control; moving to 4:2:0 —
the subsampling where turbo's SIMD upsampling engages — *widens* its
deficit. The COCO-measured gap generalises.

EdgeFirst's accurate-class lead across every corpus (turbo `islow` p50 ÷
EdgeFirst p50, YUV arm):

| Board | val2017 | val2017-yuv420 | val2017-dri | CLIC 4:2:0 | CLIC 4:4:4 | CLIC 4:2:0-dri |
|-------|---------|----------------|-------------|-----------|-----------|----------------|
| x86-desktop | 1.17× | 1.15× | 1.22× | 1.12× | 1.22× | 1.16× |
| rpi5-hailo (A76) | 1.34× | 1.27× | 1.46× | 1.20× | 1.31× | 1.30× |
| orin-nano (A78AE) | 1.38× | 1.31× | 1.53× | 1.28× | 1.51× | 1.39× |
| imx95-pro (A55) | 1.17× | 1.05× | 1.34× | 1.01× | 1.27× | 1.14× |
| imx8mp-frdm (A53) | 1.15× | 1.00× | 1.33× | 1.00× | 1.31× | 1.16× |
| mbp-m2-max | 1.41× | 1.37× | 1.49× | 1.32× | 1.40× | 1.41× |

(val2017-yuv420, CLIC 4:2:0, and CLIC 4:2:0-dri re-captured for this revision
alongside the new RGB arm below; the shift from the previous capture is
within normal round-to-round variance — see the Max spread discussion
above.)

EdgeFirst leads or ties every cell. The two 1.00× cells — both on the
in-order A53, on the 4:2:0 corpora where turbo is strongest — are the
honest edge: the accurate class is a dead heat there (3.774 vs 3.786 ms on
the small set, 30.401 vs 30.517 ms on CLIC), and at large 4:2:0 turbo
`ifast` edges `hal_fast` by 0.8% (27.169 vs 27.378 ms, YUV) — the only
fast-class YUV cell EdgeFirst does not win across six hosts and six
corpora. DRI widens the lead everywhere, most on the cores cameras
actually ship.

**The fused 4:2:0→RGB path (new in this revision) closes the gap this
document previously had to leave unmeasured.** Previously an `rgb` request
on a 4:2:0 source silently resolved to native NV12 — the RGB comparison
did not exist on the subsampling that dominates real-world JPEG, precisely
where a libjpeg-turbo-literate reader would expect EdgeFirst's lead to
narrow or reverse (turbo's SIMD chroma upsampling is strongest there). It
is now measured, using a 2×2 nearest-neighbour (box) chroma upsample fused
into the write — not libjpeg's fancy/triangle filter, a deliberate
speed/accuracy tradeoff (44–50 dB PSNR vs a reference decode, see the JPEG
Decode arm table). EdgeFirst's accurate-class RGB lead (turbo `islow` p50 ÷
EdgeFirst p50) on the two native-4:2:0 corpora:

| Board | val2017-yuv420 RGB | CLIC 4:2:0 RGB | CLIC 4:2:0-dri RGB |
|-------|---------------------|-----------------|---------------------|
| x86-desktop | 1.17× | 1.17× | 1.20× |
| rpi5-hailo (A76) | 1.36× | 1.34× | 1.42× |
| orin-nano (A78AE) | 1.46× | 1.45× | 1.56× |
| imx95-pro (A55) | 1.17× | 1.18× | 1.29× |
| imx8mp-frdm (A53) | 1.11× | 1.16× | 1.28× |
| mbp-m2-max | 1.39× | 1.36× | 1.44× |

EdgeFirst wins the RGB arm on every host, including the A53 — the box
upsample's extra work (versus the YUV arm's straight NV12 passthrough) costs
less than turbo's fancy upsampling costs it, even where the YUV arm was a
dead heat. The fast-class RGB comparison (turbo `ifast` ÷ EdgeFirst `fast`)
tells the same story and, unlike the YUV arm, has **no loss cell**: 1.10×
(x86) to 1.65× (A78AE, CLIC 4:2:0-dri). DRI widens the RGB lead too, the
same shape as the YUV arm above.

### AWS cloud baselines

The same decode-only matrix ran on AWS Batch (2026-08-13) across the five
reproducible instance classes reviewers publish against — one job per
corpus per queue, each on an exclusively-owned on-demand instance (the job
requests the full 8 vCPUs), ECS AL2023 AMIs, the multi-arch
`edgefirst-hal-jpeg-bench` container built from this tree, three
interleaved rounds inside one session per instance, arms pinned to core 0,
n=200. The container matrix carries six arms (EdgeFirst accurate, turbo
`islow`, zune, image, stb, Wuffs — no fast-class arms). COCO val2017,
YUV/RGB p50 ms, median of rounds; **bold** = fastest:

| Queue | CPU | Arm | EdgeFirst | Turbo `islow` | zune-jpeg | image | stb | Wuffs |
|-------|-----|-----|-----------|---------------|-----------|-------|-----|-------|
| aws-m8g | Graviton4 (Neoverse-V2) | YUV | **1.376** | 2.117 | 2.547 | — | — | — |
| | | RGB | **1.457** | 2.200 | 2.663 | 3.379 | 3.794 | 3.793 |
| aws-c7g | Graviton3 (Neoverse-V1) | YUV | **1.665** | 2.529 | 2.984 | — | — | — |
| | | RGB | **1.781** | 2.642 | 3.126 | 3.850 | 4.353 | 4.753 |
| aws-m6g | Graviton2 (Neoverse-N1) | YUV | **2.292** | 3.271 | 3.968 | — | — | — |
| | | RGB | **2.489** | 3.460 | 4.201 | 4.823 | 6.043 | 6.927 |
| aws-m7i | Sapphire Rapids | YUV | **1.760** | 2.070 | 2.637 | — | — | — |
| | | RGB | **1.882** | 2.184 | 2.667 | 2.826 | 3.417 | 3.272 |
| aws-c7a | Genoa (Zen 4) | YUV | **1.455** | 1.667 | 2.131 | — | — | — |
| | | RGB | **1.556** | 1.735 | 2.084 | 2.283 | 3.192 | 2.756 |

- **EdgeFirst's largest leads anywhere are on the Graviton parts** — the
  Arm baselines other published codec numbers use: 1.54× over turbo
  `islow` on Graviton4, 1.52× on Graviton3, 1.43× on Graviton2 (COCO,
  YUV), holding 1.32–1.63× across every control corpus. On the modern x86
  servers the lead is 1.15–1.18× (1.07–1.24× across corpora) — EdgeFirst
  is fastest in **every cloud cell**.
- **The zune verdict replicates on the exact instances others benchmark
  on**: zune trails turbo by 1.18–1.28× on COCO and the gap *widens* to
  1.32–1.56× on the 4:2:0 corpora, on every queue — same shape as the
  workstation and SBC results.
- **Cross-corpus caveat**: unlike the board sweeps, each corpus ran on its
  own freshly-launched instance, so within-corpus arm ratios are
  same-silicon and tight (round spreads <1%) but cross-corpus deltas
  carry instance-to-instance variance — the m7i pair shows it (its
  val2017 and val2017-dri jobs landed on different-turbo-bin instances,
  producing a nonphysical negative DRI delta). DRI penalties are therefore
  quoted from the board captures, where the corpus pairs shared one
  session; the cloud runs agree in direction on the other four queues
  (turbo +8.8–10.0%, EdgeFirst +0.9–2.8%).
- **Instance-bin variance is now bounded, not inferred.** A second
  independently-launched on-demand instance per queue re-ran the val2017
  headline corpus (same job definition, same container image, same
  protocol). Four of five queues landed within **≤0.6%** of the first
  instance on every arm (m8g, c7g, m6g, c7a — both absolute p50s and the
  EdgeFirst÷turbo ratio). **m7i is the outlier**: its second instance ran
  **6–7% faster** on every arm (hal 1.760 ms → 1.638 ms, turbo `islow`
  2.070 ms → 1.941 ms) — confirming the m7i cross-corpus anomaly above is a
  real per-instance effect (turbo bin / CPU stepping / noisy neighbour),
  not a one-off. The reassuring part: **the ratio barely moves even when
  the absolute time does** — turbo ÷ EdgeFirst on m7i YUV is 1.176× on
  instance 1 and 1.185× on instance 2, a 0.8-point spread against a 6.9%
  swing in the underlying numbers, because whatever changed the instance's
  speed changed both arms together (same session, same silicon). That is
  what licenses quoting the cloud ratios directly rather than only the
  board captures: absolute latency varies instance-to-instance on at least
  one queue, but the competitive ratio this document actually claims does
  not.

### Hardware decoders

The arms above are deliberately CPU-only. SoC hardware decode paths are
measured separately — same corpora, same harness, published as their own
table rather than mixed into the CPU comparison, because the resource being
spent differs:

- **i.MX 95 (V4L2 mxc-jpeg)** performs no colour-space conversion. Verified
  on-target via `VIDIOC_ENUM_FMT`: the decode node's RGB capture formats
  exist only for RGB-*encoded* JPEGs (the mirror of the encoder node's RGB
  inputs); standard YCbCr streams decode to their native sampling —
  4:2:0→NV12, 4:2:2→YUYV, 4:4:4→YUV3, greyscale→GREY. The table therefore
  carries **two rows**: `hal_v4l2_* --decode-only` (JPEG → NV* in DMA — the
  stopping point for consumers that take NV12 directly, e.g. GL-sampled
  model preprocessing) and `--full-res-convert` (adds the full-resolution
  NV*→RGB second pass, via GPU (`hal_v4l2_gl`) or CPU (`hal_v4l2_cpu`) — the
  full cost for RGB consumers), with the per-frame decode/convert split
  reported so both audiences can read their number from one run. Both passes
  follow the pool-reuse discipline: `ImageProcessor::create_image()`
  allocates the decode source and the RGB destination once, sized to the
  dataset's largest frame, before the timed loop (warmup then absorbs
  first-touch faults, EGL import, and V4L2 queue setup); per-frame work only
  reconfigures logical dimensions. Buffer creation is never inside the timed
  region, and the harness verifies the pools actually held — a
  `BufferIdentity` change during the hot loop counts as `identity_churn` and
  warns loudly (0 on the captured runs).
- **Jetson Orin (nvJPEG)** is **not** a dedicated JPEG engine on this SoC:
  only the GPU_HYBRID backend exists (Huffman on CPU, IDCT/postprocess on
  the **shared CUDA cores** — the NVJPG ASIC is unreachable through CUDA
  nvJPEG). Decoding therefore competes with anything else on the GPU, most
  importantly AI inference; that contention is real, is the reason the
  backend is opt-in (`EDGEFIRST_ENABLE_NVJPEG`), and is **out of scope for
  these benchmarks** — concurrent decode-plus-inference impact will be
  measured separately. Its fixed output is interleaved RGB (no NV12 in
  CUDA 12.3.3), so it has no NV-native row.

Captured 2026-08-13 with the same protocol as the CPU sweep (median of 3
interleaved rounds, n=200 / all 62 CLIC files, pinned harness core; raw
logs under `benchmarks/results/<board>/hw-decode/<corpus>/`). The EdgeFirst
CPU column repeats the same board's CPU decode (YUV arm) for reference.

**i.MX 95 V4L2 (mxc-jpeg), p50 ms.** decode-only stops at native NV*-in-DMA
with near-zero CPU; the full-res rows add the NV*→RGB second pass
(decode + convert split in parentheses):

| Corpus | decode-only | + RGB, CPU convert | + RGB, GL convert | EdgeFirst CPU |
|--------|-------------|--------------------|-------------------|---------------|
| val2017 | 4.54 | 6.77 (4.47 + 2.35) | 9.11 (5.87 + 2.97) | 6.03 |
| val2017-yuv420 | 2.53 | 4.44 (2.38 + 2.05) | 7.00 (3.78 + 2.98) | 3.37 |
| val2017-dri | 4.39 | 6.65 (4.33 + 2.34) | 8.98 (5.77 + 2.99) | 6.05 |
| CLIC 4:2:0 | 21.12 | 44.0 (21.6 + 23.2) | 47.6 (28.6 + 18.7) | 27.68 |
| CLIC 4:4:4 | 42.15 | 62.7 (42.3 + 19.2) | 71.6 (51.2 + 18.0) | 40.70 |
| CLIC 4:2:0-dri | 21.24 | 44.6 (21.8 + 23.5) | 47.5 (28.9 + 18.7) | 27.95 |

What the splits show: the hardware block is **~25% faster than CPU decode
on 4:2:0 and COCO** while freeing the core, and — unlike every CPU decoder
— is **indifferent to restart markers** (val2017-dri matches val2017 within
noise); at large 4:4:4 the tuned CPU decoder edges it. The two converters
cross over exactly where predicted: at ~0.27 MP the CPU convert wins (GL's
per-frame import/sync overhead dominates small frames) while at 2.8 MP the
GL convert leg is cheaper (18.7 vs 23.2 ms) — though GL work visibly
contends with the V4L2 decode leg (28.6 vs 21.6 ms decode under GL), so at
these sizes the CPU-convert total still wins. One cost the full-res GL leg
carries by design: reconfiguring the pooled RGB destination to each frame's
dimensions requires a fresh EGLImage import per *distinct* geometry
(imports are cached per size; reusing a stale-geometry image would sample
at the wrong pitch), so varying-size corpora pay real import cost inside
the convert leg that the fixed-size letterbox path does not — a genuine
cost of GL output at native sizes, not harness overhead. NV12-native consumers should
read the decode-only column; RGB consumers the full-cost columns.

**Jetson Orin nvJPEG (GPU_HYBRID, RGB out), p50 ms** — captured on the
orin-nano stand-in (`adis-uav1`), per-frame engagement verified via
`codec.decode_jpeg.nvjpeg_*` tracing spans:

| Corpus | nvJPEG | EdgeFirst CPU |
|--------|--------|---------------|
| val2017 | 4.72 | 3.72 |
| val2017-yuv420 | 2.47 | 2.03 |
| val2017-dri | 4.40 | 3.75 |
| CLIC 4:2:0 | 16.46 | 15.73 |
| CLIC 4:4:4 | 22.95 | 22.78 |
| CLIC 4:2:0-dri | 16.69 | 15.69 |

nvJPEG's GPU_HYBRID never beats the tuned CPU decoder outright at any size
here, and its throughput tracks bitstream entropy — the Huffman leg stays
on the CPU, so the large high-entropy CLIC files bottleneck exactly where a
CPU decoder does. Its value on this SoC is GPU-resident RGB output and
partial CPU offload (it is, however, restart-marker-indifferent like the
i.MX block), not latency — which is why it stays opt-in and why the
decode-plus-inference contention measurement is scoped separately.

### SIMD tier value (x86-desktop)

Same host and workload, `EDGEFIRST_CODEC_FORCE_INTEL=…`, n=800 best-of-3.

| Tier | YUV | vs scalar | RGB | vs scalar |
|------|-----|-----------|-----|-----------|
| `scalar` | 2.297 ms | 1.00× | 2.971 ms | 1.00× |
| `sse2` | **1.272 ms** | **1.81×** | 1.968 ms | 1.51× |
| `sse41` | 1.272 ms | 1.81× | **1.339 ms** | **2.22×** |
| `avx2` / auto | 1.270 ms | 1.81× | 1.330 ms | 2.23× |

The two arms earn their tiers in different places. The `islow` IDCT needs only
SSE2, and it is the whole YUV win — SSE4.1 and AVX2 add nothing there. RGB adds
the colour kernel, which needs `pshufb`, so SSE4.1 is worth a further 32% on
that arm. AVX2 is within noise of SSE4.1 on both: at 8×8, wider registers buy
little beyond the VEX encoding.

Per-block IDCT cost (in-tree microbenchmark `idct_kernel_cost`): **21.5 ns**
on Rocket Lake, **81.1 ns** on the A76, **212.6 ns** on the A55, **220.8 ns** on
the A53.

### Build & run provenance

Every sweep writes `results/<board>/decode-ab-*/provenance.txt` recording
what a reader needs to reproduce (or falsify) that host's numbers: rustc and
cargo-zigbuild versions, build profile and run parameters, the zune-jpeg /
image crate versions from `Cargo.lock` (with zune's `x86`+`neon` SIMD
features pinned explicitly in the harness manifest), the host kernel and CPU
model, the pinned core's cpufreq governor and current/max clocks, and the
loader-resolved libturbojpeg path with its owning package. The C arm
additionally resolves the library it actually loaded via `dladdr` and embeds
that path in its console output and CSV notes. The turbo provenance matters
doubly on ARM: libjpeg-turbo 3.2 removed its GNU-assembler NEON
implementation, so ARM builds need GCC 12+ or Clang for full performance —
the recorded library version is what makes the ARM baselines checkable.
Harness compilers are mixed by construction and recorded as such: the
turbojpeg harness is compiled remotely with the host gcc on x86 targets
(its timed region runs inside the dlopened system libturbojpeg either
way), while the stb/wuffs arms — whose decoders compile into the binary —
are cross-built with zig cc (clang) for both architectures; the provenance
records both compilers.

Per-board libjpeg-turbo package and compiler, from each host's
`provenance.txt` plus its package manager (`dpkg -l` / `rpm -qa`):

| Board | Core | libjpeg-turbo | Package source | Compiler |
|-------|------|---------------|-----------------|----------|
| imx8mp-frdm | A53 | 3.0.1-r0 | Yocto/OE recipe (`libturbojpeg0`) | GCC 14.3.0 |
| imx95-pro | A55 | 1:3.1.2-r0 | Yocto/OE recipe (`libturbojpeg0`) | GCC 15.2.0 |
| rpi5-hailo | A76 | 1:2.1.5-4 | Raspberry Pi OS / Debian apt (`libturbojpeg0`) | GCC 14.2.0 |
| orin-nano (adis-uav1 stand-in) | A78AE | 2.1.2-0ubuntu1 | Ubuntu apt (`libturbojpeg`) | GCC 11.4.0 |
| x86-desktop (sebstation) | Rocket Lake | 1:2.1.5-4ubuntu4 | Ubuntu apt (`libturbojpeg0`) | GCC 15.2.0 |

None of the five hosts actually runs libjpeg-turbo 3.2+ — the newest are
3.0.1/3.1.2 (imx8mp/imx95-pro, both pre-dating the GNU-assembler NEON
removal), the rest are the older 2.1.x series. The 3.2 NEON-assembler-removal
risk named above is a real thing to check on *future* re-captures (especially
if a board image updates its `libturbojpeg0` package), but it is not in play
for any number in this document today — every ARM host here has its classic
GNU-assembler NEON kernels intact.

### Reproduce

```bash
# Full eight-arm decoder sweep, aarch64 boards + x86_64 hosts over SSH
# (release profile, interleaved rounds, median-of-3 reported with per-round
# p50s and spread; EdgeFirst accurate/fast, turbo islow/ifast, zune-jpeg,
# image crate, stb_image, Wuffs; provenance.txt written per host; results
# land under results/<host>/decode-ab-sweep/<corpus>/)
./benchmarks/scripts/decode-ab-sweep.sh imx8mp-frdm imx95-pro rpi5-hailo adis-uav1 sebstation

# Same sweep against a control corpus (staged by scripts/sync-corpus.sh)
EDGEFIRST_BENCH_COCO_REMOTE=/data/corpora/val2017-yuv420 \
  ./benchmarks/scripts/decode-ab-sweep.sh rpi5-hailo

# Hardware-decode rows (i.MX 95 V4L2 / Jetson nvJPEG; see the hardware table)
# hal_v4l2_gl --decode-only | hal_v4l2_cpu --full-res-convert |
# hal_v4l2_gl --full-res-convert | hal_nvjpeg --decode-only --decode-fmt native

# Three-arm publish subset (EdgeFirst accurate vs turbo islow/ifast only)
CARGO_PROFILE=release EDGEFIRST_BENCH_ORIN_FALLBACK=adis-uav1 \
  ./benchmarks/scripts/decode-ab-publish.sh imx8mp-frdm imx95-pro rpi5-hailo orin-nano

# Fast-vs-accurate DCT pixel similarity (cosine/PSNR/max delta) over a corpus
cargo run --release -p dct_compare -- --dir /path/to/coco/val2017

# Smoke / investigation (profiling profile, sequential, islow only)
./benchmarks/scripts/decode-ab-matrix.sh imx95-pro rpi5-hailo

# Decoder A/B, x86 / AWS Batch
docker build -f benchmarks/docker/Dockerfile -t edgefirst-hal-jpeg-bench .
docker run --rm -v /path/to/coco:/data/coco:ro -v "$PWD/results:/results" \
  -e BOARD=x86-desktop -e FORMATS=yuv,rgb edgefirst-hal-jpeg-bench

# Single arm on the host (DCT=fast for the ifast columns)
make -C benchmarks/modules/turbojpeg
./benchmarks/modules/turbojpeg/build/turbojpeg_bench \
  --limit 800 --warmup 40 --decode-only --format yuv --dct accurate

# IDCT kernel microbenchmark
cargo test -p edgefirst-codec --release -- --ignored --nocapture idct_kernel_cost

# mbp-m2-max: no SSH target (this machine IS the board) — build natively and
# run each arm directly against a local corpus copy, unpinned (no
# taskset-equivalent on macOS). See benchmarks/results/mbp-m2-max/decode-ab-sweep/
# for the captured summaries; there is no committed local-runner script
# (the boards above are the reproducible path) — mirror decode-ab-sweep.sh's
# per-arm commands with EDGEFIRST_BENCH_COCO pointed at a local corpus dir.
cargo build --release -p hal_cpu -p rust_jpeg
make -C benchmarks/modules/turbojpeg && make -C benchmarks/modules/stb && make -C benchmarks/modules/wuffs

# Wuffs 3-byte-swizzle-vs-4-byte-native A/B (see the JPEG Decode "floor" bullet)
EDGEFIRST_WUFFS_FORCE_4BPP=1 EDGEFIRST_BENCH_COCO=/path/to/coco/val2017 \
  ./benchmarks/modules/wuffs/build/wuffs_bench --limit 200 --warmup 20 --format rgb

# Fused native-4:2:0→RGB accuracy check (dumps a PPM for comparison against
# a reference decoder, e.g. PIL) — see the ARCHITECTURE.md § Fused Decode Output
cargo run --release -p edgefirst-codec --example dump_rgb420 -- in.jpg out.ppm
```

## Benchmark Results

> **The result tables below are generated. Do not hand-edit them.** They are
> rendered from the per-board JSON in `benchmarks/<platform>/` by
> `.github/scripts/generate_benchmark_tables.py`. A number typed in by hand will
> be silently overwritten on the next regeneration, and in the meantime it is a
> claim with no measurement behind it. To change a number, re-collect and re-run
> the generator. Surrounding prose is hand-written and safe to edit.

**Data collected:** March 30, 2026 (v0.15.0, per-texture EGL binding
optimization) unless a section states otherwise — the GL convert,
letterbox, and mask-rendering tables were re-collected June 2026 after
the GL convergence engine (PR #109/#110, issue #106); each carries its
own capture note.

### Buffer Infrastructure

#### Allocation Latency

Measures `Tensor::new()` latency for each buffer type and resolution.

| Platform | Buffer | 720p (3.5 MB) | 1080p (7.9 MB) | 4K (31.6 MB) |
|----------|--------|---------------|-----------------|---------------|
| imx8mp-frdm | MEM | 311 us | 698 us | 2.8 ms |
| imx8mp-frdm | SHM | 26 us | 26 us | 26 us |
| imx8mp-frdm | DMA | 1.8 ms | 2.8 ms | 10.3 ms |
| imx95-frdm | MEM | 263 us | 594 us | 2.4 ms |
| imx95-frdm | SHM | 31 us | 31 us | 31 us |
| imx95-frdm | DMA | 983 us | 2.1 ms | 8.4 ms |
| rpi5-hailo | MEM | 285 us | 790 us | 3.5 ms |
| rpi5-hailo | SHM | 5.0 us | 5.0 us | 6.0 us |
| rpi5-hailo | DMA | 714 us | 1.6 ms | 6.2 ms |
| jetson-orin-nano | MEM | 140 us | 330 us | 1.3 ms |
| jetson-orin-nano | SHM | 12 us | 12 us | 12 us |
| mbp-m2-max | MEM | 28 us | 65 us | 323 us |
| mbp-m2-max | SHM | 2.0 us | 2.0 us | 2.0 us |
| mbp-m2-max | DMA | 16 us | 16 us | 16 us |

Apple Silicon DMA-row is IOSurface: allocation cost is dominated by the
`IOSurfaceCreate` round-trip into the kernel, not the buffer size, so it
stays at ~16 µs from 720p to 4K. The Linux DMA-buf path scales linearly
with size because the kernel zeros the buffer pages on allocation; macOS
defers initialization to first touch.

#### Map/Unmap Latency

Measures `tensor.map()` round-trip latency.

| Platform | Buffer | 720p | 1080p | 4K |
|----------|--------|------|-------|-----|
| imx8mp-frdm | SHM | 13 us | 14 us | 13 us |
| imx8mp-frdm | DMA | 349 us | 767 us | 3.0 ms |
| imx95-frdm | SHM | 12 us | 12 us | 12 us |
| imx95-frdm | DMA | 278 us | 625 us | 2.5 ms |
| rpi5-hailo | SHM | 2.0 us | 2.0 us | 3.0 us |
| rpi5-hailo | DMA | 99 us | 220 us | 869 us |
| jetson-orin-nano | SHM | 3.0 us | 3.0 us | 3.0 us |
| x86-desktop | SHM | 1.0 us | 1.0 us | 1.0 us |
| mbp-m2-max | SHM | 0.5 us | 0.5 us | 0.5 us |
| mbp-m2-max | DMA | 0.5 us | 0.5 us | 0.5 us |

IOSurface map on macOS is a `IOSurfaceLock` call on a buffer the kernel
already owns; it stays sub-microsecond across all sizes. SHM is a single
`fstat` + cached `mmap` of an already-open file descriptor.

#### Memcpy Throughput

Measures `tensor.map(); copy_from_slice(src)` on a single CPU thread,
filling the full image. Captures the cost of touching the backing memory
through the chosen buffer kind.

| Platform | Buffer | 720p (3.5 MB) | 1080p (7.9 MB) | 4K (31.6 MB) |
|----------|--------|---------------|----------------|---------------|
| mbp-m2-max | MEM | 89 us — 39.6 GiB/s | 255 us — 30.3 GiB/s | 688 us — 45.0 GiB/s |
| mbp-m2-max | SHM | 211 us — 16.7 GiB/s | 502 us — 15.4 GiB/s | 1.7 ms — 18.3 GiB/s |
| mbp-m2-max | DMA | 82 us — 41.4 GiB/s | 247 us — 31.2 GiB/s | 614 us — 50.3 GiB/s |

IOSurface (Dma) and heap (Mem) deliver comparable bandwidth because both
hit cached unified memory. SHM is 2–2.7× slower at every resolution: the
shared-memory file lives behind a `mmap` that doesn't get the same
prefetcher treatment as anonymous heap. **Verdict: prefer Dma over Shm on
macOS when you need a backing tensor that the GL backend can also import.**

### Image Preprocessing: Letterbox Pipeline (Camera → Model Input)

**GL rows re-collected:** June 2026 (GL convergence engine, PR #109/#110;
median of 3 runs × n=100, `pipeline_benchmark`). G2D/CPU rows are the
v0.15.0 capture (those backends were not changed).

**1080p → 640×640:**

| Platform | Compute | Buffer | YUYV→RGBA | YUYV→RGB | YUYV→8BPi | NV12→RGBA | VYUY→RGBA |
|----------|---------|--------|-----------|----------|-----------|-----------|-----------|
| imx8mp-frdm | G2D | DMA | 2.7 ms | 4.0 ms | — | 4.1 ms | — |
| imx8mp-frdm | GL | DMA | 1.8 ms | 11.3 ms | — | 3.2 ms | — |
| imx8mp-frdm | CPU | Heap | 17.4 ms | 17.6 ms | — | 33.9 ms | 17.5 ms |
| imx95-frdm | G2D | DMA | 3.9 ms | 4.6 ms | — | 3.7 ms | — |
| imx95-frdm | GL | DMA | 1.2 ms | 2.6 ms | — | 1.2 ms | — |
| imx95-frdm | CPU | Heap | 14.5 ms | 14.9 ms | — | 16.6 ms | 14.5 ms |
| rpi5-hailo | GL | DMA | 3.3 ms | 4.0 ms | — | 1.2 ms | — |
| rpi5-hailo | CPU | Heap | 7.6 ms | 7.2 ms | — | 8.1 ms | 7.6 ms |
| jetson-orin-nano | GL | PBO | 4.3 ms | 4.6 ms | — | 4.3 ms | — |
| jetson-orin-nano | CPU | Heap | 6.1 ms | 5.9 ms | — | 5.3 ms | 6.2 ms |
| x86-desktop | CPU | Heap | 3.0 ms | 1.5 ms | — | 1.8 ms | 5.4 ms |
| mbp-m2-max | CPU | Heap | 1.5 ms | 1.7 ms | 2.0 ms | 1.3 ms | — |

**4K → 640×640:**

| Platform | Compute | Buffer | YUYV→RGBA | YUYV→RGB | NV12→RGBA |
|----------|---------|--------|-----------|----------|-----------|
| imx8mp-frdm | G2D | DMA | 4.0 ms | 5.3 ms | 5.8 ms |
| imx8mp-frdm | GL | DMA | 2.3 ms | 12.4 ms | 9.4 ms |
| imx8mp-frdm | CPU | Heap | 59.9 ms | 50.6 ms | 125 ms |
| imx95-frdm | G2D | DMA | 15.8 ms | 16.5 ms | 13.3 ms |
| imx95-frdm | GL | DMA | 1.6 ms | 3.6 ms | 4.8 ms |
| imx95-frdm | CPU | Heap | 46.5 ms | 41.9 ms | 55.3 ms |
| rpi5-hailo | GL | DMA | 18.5 ms | 19.3 ms | 5.0 ms |
| rpi5-hailo | CPU | Heap | 24.0 ms | 19.7 ms | 24.4 ms |
| jetson-orin-nano | GL | DMA | — | — | — |
| jetson-orin-nano | CPU | Heap | 18.4 ms | 20.0 ms | 14.9 ms |
| x86-desktop | CPU | Heap | 9.5 ms | 6.7 ms | 9.0 ms |
| mbp-m2-max | CPU | Heap | 4.6 ms | 4.4 ms | 3.7 ms |

### Format Conversion (Same Size, No Resize)

**imx8mp/imx95 GL rows re-collected:** June 11, 2026
(`image_benchmark`, PR #110 head, n=100). This closes the issue #106
drift: the 2026-06-09 capture had imx8mp RGBA→BGRA at 13.8 ms (+79%)
and RGBA→GREY at 14.9 ms (+91%) versus this table; the GL convergence
engine (PR #109) reclaimed both to the v0.15.0 numbers, and the
RGBA→BGRA / RGBA→GREY cells are now permanent `image_benchmark`
sentinels so same-size drift cannot go unmeasured again. Orin YUYV GL
cells are the 2026-06-10 PR #109 capture; remaining GL cells and all
G2D/CPU rows are the v0.15.0 capture.

**1080p → 1080p:**

| Platform | Compute | Buffer | YUYV→RGBA | YUYV→RGB | NV12→RGBA | RGB→RGBA | RGBA→BGRA | RGBA→GREY |
|----------|---------|--------|-----------|----------|-----------|----------|-----------|-----------|
| imx8mp-frdm | G2D | DMA | 6.0 ms | 10.5 ms | 6.3 ms | — | — | — |
| imx8mp-frdm | GL | DMA | 7.3 ms | 49.0 ms | 6.1 ms | — | 7.7 ms | 7.9 ms |
| imx8mp-frdm | CPU | Heap | 13.5 ms | 11.8 ms | 25.7 ms | 13.7 ms | 30.2 ms | 10.0 ms |
| imx95-frdm | G2D | DMA | 4.6 ms | 4.3 ms | 4.5 ms | — | — | — |
| imx95-frdm | GL | DMA | 3.0 ms | 5.7 ms | 3.2 ms | — | 3.1 ms | 2.2 ms |
| imx95-frdm | CPU | Heap | 12.4 ms | 11.0 ms | 16.1 ms | 11.1 ms | 24.8 ms | 9.0 ms |
| rpi5-hailo | GL | DMA | 7.2 ms | 10.4 ms | 5.4 ms | — | 8.4 ms | 6.2 ms |
| rpi5-hailo | CPU | Heap | 6.8 ms | 5.4 ms | 8.0 ms | 6.6 ms | 12.2 ms | 2.5 ms |
| jetson-orin-nano | GL | PBO | 2.3 ms | 2.2 ms | — | 1.5 ms | 4.2 ms | 1.5 ms |
| jetson-orin-nano | CPU | Heap | 3.0 ms | 2.8 ms | 2.1 ms | 789 us | 3.2 ms | 1.4 ms |
| x86-desktop | CPU | Heap | 516 us | 559 us | 256 us | 261 us | 758 us | 219 us |
| mbp-m2-max | GL | IOSurface | 409 us | — | — | — | — | — |
| mbp-m2-max | CPU | Heap | 541 us | 499 us | 329 us | 141 us | 784 us | 314 us |

The imx8mp YUYV→RGB / RGBA→RGB cells (~49 ms) are the Vivante two-pass
packed-RGB path (GL has no 3-byte render format); imx95 YUYV→RGB halved
versus v0.15.0 (11.8 → 5.7 ms) from the engine's single-finish two-pass.

The macOS GL row only covers YUYV→RGBA today; other format pairs fall
through to CPU. Even with that single working pair the speedup is **1.3×**
at 1080p, and (see the 4K convert below) **4.8×** at 3840×2160 because the
GPU path is essentially bandwidth-bound while CPU scales with pixel count.

**3840×2160 → 3840×2160 (4K convert):**

| Platform | Compute | Buffer | YUYV→RGBA | YUYV→RGB | NV12→RGBA |
|----------|---------|--------|-----------|----------|-----------|
| mbp-m2-max | GL | IOSurface | 458 us | — | — |
| mbp-m2-max | CPU | Heap | 2.2 ms | 2.0 ms | 1.4 ms |

### Decoder Post-Processing

All CPU-only (decoder is not GPU-accelerated).

**YOLOv8 Detection (84×8400, 80 classes):**

| Platform | Data Type | Decode + NMS | Decode Only | NMS Only | Dequantize |
|----------|-----------|-------------|-------------|----------|------------|
| imx8mp-frdm | i8 (quant) | 1.0 ms | 998 us | 20 us | 3.7 ms |
| imx8mp-frdm | f32 | 6.1 ms | — | — | — |
| imx95-frdm | i8 (quant) | 847 us | 778 us | 19 us | 2.9 ms |
| imx95-frdm | f32 | 6.0 ms | — | — | — |
| rpi5-hailo | i8 (quant) | 243 us | 257 us | 4.0 us | 2.1 ms |
| rpi5-hailo | f32 | 2.9 ms | — | — | — |
| jetson-orin-nano | i8 (quant) | 343 us | 331 us | 7.0 us | 2.0 ms |
| jetson-orin-nano | f32 | 2.2 ms | — | — | — |
| x86-desktop | i8 (quant) | 82 us | 189 us | 4.0 us | 383 us |
| x86-desktop | f32 | 460 us | — | — | — |
| mbp-m2-max | i8 (quant) | 29 us | 25 us | 2.0 us | 376 us |
| mbp-m2-max | f32 | 221 us | — | — | — |

**YOLOv8 Segmentation (mask coefficient → pixel decode):**

| Platform | Data Type | Masks Decode |
|----------|-----------|-------------|
| imx8mp-frdm | i8 (quant) | 3.1 ms |
| imx8mp-frdm | f32 | 5.9 ms |
| imx95-frdm | i8 (quant) | 3.4 ms |
| imx95-frdm | f32 | 6.6 ms |
| rpi5-hailo | i8 (quant) | 974 us |
| rpi5-hailo | f32 | 2.5 ms |
| jetson-orin-nano | i8 (quant) | 1.1 ms |
| jetson-orin-nano | f32 | 2.2 ms |
| x86-desktop | i8 (quant) | 352 us |
| x86-desktop | f32 | 663 us |
| mbp-m2-max | i8 (quant) | 246 us |
| mbp-m2-max | f32 | 413 us |

### Image Codec Decode (`edgefirst-codec`) — JPEG rows superseded

> [!IMPORTANT]
> **The JPEG tables and claims in this section are superseded by [§ JPEG
> Decode: EdgeFirst vs libjpeg-turbo](#jpeg-decode-edgefirst-vs-libjpeg-turbo)**,
> captured 2026-08-13 against the current (post-rewrite) decoder. This
> section's JPEG numbers were captured May 18, 2026 (v0.22.1) against a
> decoder whose entropy/IDCT/MCU-write internals have since been rewritten
> (see the JPEG Decode section's provenance and the 0.28.x CHANGELOG); its
> "within 6% of the image crate" (x86) and "20–22% faster" (ARM) claims no
> longer hold — the current decoder measures ~1.5–1.6× faster than the
> `image` crate on the same hosts. Kept here for historical continuity
> (RGBA/BGRA, f32, NV12-skip, and strided-decode overhead breakdowns this
> project has not re-collected since) rather than deleted outright — treat
> every JPEG number below as v0.22.1 archive, not current performance. The
> **PNG Decode** table at the end of this section is unaffected (the JPEG
> rewrite didn't touch PNG) but is likewise not re-verified since May 2026.

**Data collected:** May 18, 2026 (v0.22.1, custom JPEG decoder with NEON/SSE4.1/SSSE3 kernels + vectorised type conversion, Mem tensors)

Compares decode paths:
- **edgefirst-codec** — `Tensor::load_image()` strided decode into pre-allocated tensor (zero-allocation hot path; custom baseline JPEG decoder with NEON SIMD on AArch64, SSE4.1/SSSE3 SIMD on x86-64, vectorised u8→f32/u16/i16 conversion)
- **image crate** — `image::load_from_memory_with_format()` + `to_rgb8()` (allocates per call; uses zune-jpeg internally with SSE2/AVX2 SIMD)
- **zune-png** — raw `zune_png::PngDecoder::decode_raw()` (PNG only; allocates per call)

All JPEG measurements use the custom decoder (not zune-jpeg). All measurements are Mem (heap) tensors. DMA-buf and PBO-backed tensors will add map/unmap overhead per the Buffer Infrastructure table above.

**JPEG Decode — RGB u8:**

| Platform | Image | edgefirst-codec | image crate | Speedup |
|----------|-------|-----------------|-------------|---------|
| imx8mp-frdm (A53) | zidane 720p (1280×720) | 14.4 ms | 17.9 ms | **20% faster** |
| imx8mp-frdm (A53) | giraffe 640 (640×640) | 12.1 ms | 13.8 ms | **12% faster** |
| imx95-frdm (A55) | zidane 720p (1280×720) | 13.7 ms | 17.5 ms | **22% faster** |
| imx95-frdm (A55) | giraffe 640 (640×640) | 11.5 ms | 12.7 ms | **9% faster** |
| x86-desktop | zidane 720p (1280×720) | 1.7 ms | 1.6 ms | 6% slower |
| x86-desktop | giraffe 640 (640×640) | 1.7 ms | 1.9 ms | **12% faster** |
| mbp-m2-max (M2 Max) | zidane 720p (1280×720) | 1.4 ms | 2.0 ms | **30% faster** |
| mbp-m2-max (M2 Max) | giraffe 640 (640×640) | 1.8 ms | — | — |

**JPEG Decode — RGBA / BGRA u8:**

| Platform | Format | edgefirst-codec | vs RGB | Notes |
|----------|--------|-----------------|--------|-------|
| imx8mp-frdm | RGBA | 14.4 ms | 0% | NEON vst4 interleaved store |
| imx8mp-frdm | BGRA | 14.3 ms | −0.7% | NEON vst4 with swapped R/B |
| imx95-frdm | RGBA | 13.8 ms | +0.7% | |
| imx95-frdm | BGRA | 14.0 ms | +2.2% | |
| x86-desktop | RGBA | 1.6 ms | −6% | SSE2 unpack interleave |
| x86-desktop | BGRA | 1.7 ms | 0% | SSE2 unpack with swapped R/B |
| mbp-m2-max | RGBA | 1.5 ms | +7% | NEON vst4 interleaved store |
| mbp-m2-max | BGRA | 1.5 ms | +7% | NEON vst4 with swapped R/B |

**JPEG Decode — NV12 (skip color conversion):**

| Platform | edgefirst-codec | vs RGB | Notes |
|----------|-----------------|--------|-------|
| imx8mp-frdm | 11.0 ms | **−24%** | Direct Y copy + Cb/Cr interleave, no YCbCr→RGB |
| imx95-frdm | 10.4 ms | **−24%** | |
| x86-desktop | 1.3 ms | **−24%** | |
| mbp-m2-max | 1.2 ms | **−17%** | |

**JPEG Decode — RGB f32:**

| Platform | edgefirst-codec | vs u8 | Notes |
|----------|-----------------|-------|-------|
| imx8mp-frdm | 16.8 ms | 1.17× | u8 decode + NEON vectorised f32 normalization |
| imx95-frdm | 16.2 ms | 1.18× | |
| x86-desktop | 2.0 ms | 1.18× | SSE2 vectorised f32 normalization |
| mbp-m2-max | 1.7 ms | 1.21× | NEON vectorised f32 normalization |

**JPEG Strided Decode (720p image → 1080p tensor):**

| Platform | edgefirst-codec | vs tight decode | Notes |
|----------|-----------------|-----------------|-------|
| imx8mp-frdm | 14.3 ms | 0% | Zero overhead — MCU loop writes directly at stride |
| imx95-frdm | 13.8 ms | 0% | |
| x86-desktop | 1.6 ms | 0% | |
| mbp-m2-max | 1.4 ms | 0% | |

**PNG Decode — RGB u8:**

| Platform | edgefirst-codec | zune raw | image crate |
|----------|-----------------|----------|-------------|
| imx8mp-frdm | 29.6 ms | 28.8 ms | 33.9 ms |
| imx95-frdm | 26.5 ms | 25.4 ms | 29.3 ms |
| x86-desktop | 4.8 ms | 4.8 ms | 4.8 ms |
| mbp-m2-max | 5.3 ms | 5.3 ms | 5.2 ms |

**Key Observations:**
- On AArch64, the custom JPEG decoder with NEON SIMD is **20–22% faster** than the `image` crate (which uses zune-jpeg internally). The NEON kernels optimize IDCT, YCbCr→RGB color conversion, and chroma upsampling.
- On x86-64, SSE4.1 IDCT and SSSE3 color conversion bring performance to **within 6% of the image crate** for 720p and **12% faster** for smaller images. The remaining gap is due to zune-jpeg's AVX2 kernels. SIMD dispatch selects the highest tier automatically: SSE4.1 > SSE2 > scalar.
- **f32 decode is only 1.17–1.18× slower than u8** thanks to SIMD-vectorised u8→f32 normalization (NEON: `vcvtq_f32_u32` + `vmulq_f32`; SSE2: `_mm_cvtepi32_ps` + `_mm_mul_ps`). Previous scalar path was 4.0× slower.
- **NV12 output is 24% faster** than RGB because it skips color conversion entirely: Y plane is copied directly from IDCT output, Cb/Cr are interleaved without YCbCr→RGB math.
- **Strided decode has zero overhead** — the MCU decode loop writes directly into the tensor at the tensor's row stride, so decoding a 720p image into a 1080p tensor costs the same as into an exact-size tensor.
- RGBA/BGRA add <2% overhead vs RGB on ARM (NEON `vst4_u8`); on x86, RGBA is **6% faster** than RGB due to SSE2's native 4-channel interleave vs RGB's 3-channel SSSE3 shuffle.
- PNG decode uses zune-png internally; edgefirst-codec adds 2–5% overhead for strided row-copy into the pre-allocated tensor.
- imx95-frdm (Cortex-A55 @ 1.8 GHz) is ~4–5% faster than imx8mp-frdm (Cortex-A53 @ 1.6 GHz) across JPEG decode paths.

#### nvJPEG GPU Decode (Jetson Orin) — on-target only, superseded conclusion

> [!IMPORTANT]
> **This subsection's headline (nvJPEG 2.70×/1.67× faster than CPU, captured
> 2026-06-15) reads as a flat contradiction of [§ Hardware
> decoders](#hardware-decoders)'s 2026-08-13 finding that nvJPEG never beats
> the tuned CPU decoder (4.72 vs 3.72 ms on val2017). It is not a
> contradiction — both numbers are correct for what they measured, and the
> gap is real, not noise. Three compounding differences: (1) the June capture
> ran against the **pre-rewrite** CPU decoder — the August section's CPU
> column is ~1.6× faster on its own; (2) June measured **two single
> fixtures** (zidane, giraffe); August measures a 5000-image corpus median;
> (3) June's nvJPEG cell is the **full `load_image`** (`cuda_map` +
> `nvjpegDecode` + stream sync + unmap), while August's CPU/nvJPEG comparison
> is **decode-only** on both sides. In short: nvJPEG's GPU-resident,
> zero-copy *output* still has real value (see the "GPU-resident output + a
> freed CPU" point below), but the CPU decoder closed and reversed the raw
> **decode-latency** gap through the same rewrite documented in the JPEG
> Decode section — a genuine before/after story, not a measurement error.
> Trust the August § Hardware decoders numbers for current decode-only
> latency; this subsection's tables remain a valid record of the full
> `load_image` cost as of 0.25.0-era code.

On NVIDIA platforms the codec prefers the nvJPEG GPU backend
(`nvjpeg → v4l2 → cpu`). It decodes interleaved **RGB** straight into a
CUDA-registered PBO (what `ImageProcessor::create_image` yields on Jetson), so
the decoded image is born GPU-resident and is consumed zero-copy by
`convert()`. Benchmark cells `codec/jpeg/nvjpeg/rgbi/<fixture>` run only when
CUDA + libnvjpeg are present (`edgefirst_codec::nvjpeg_available()`) and skip
cleanly elsewhere — `bench_compare.py` treats absent cells as one-sided and
never gates on them. nvJPEG is **opt-in** (off by default so it never contends
with CUDA inference), so the cells need `EDGEFIRST_ENABLE_NVJPEG=1`. Build/run
on-target with the CUDA library path:

```bash
EDGEFIRST_ENABLE_NVJPEG=1 \
  LD_LIBRARY_PATH=/usr/local/cuda/targets/aarch64-linux/lib \
  cargo bench -p edgefirst-codec --bench codec_benchmark -- --json out.json
```

**In-tree `cargo bench` cells** (Orin Nano, L4T R36.4.7 / CUDA 12.6 / nvJPEG
12.3.3; captured 2026-06-15, `codec_benchmark` cross-built for
`aarch64-unknown-linux-gnu`, n=100). The nvJPEG cell is the full `load_image` =
`cuda_map` + `nvjpegDecode` (RGBI) + stream sync + unmap into a `create_image`
PBO; the CPU `nv12` cell is the **same binary's** CPU decoder, so this is a
like-for-like decode comparison on one run:

| Cell | nvJPEG RGB (GPU-resident) | CPU NV12 | Speedup (median) |
|------|--------------------------|----------|------------------|
| zidane 720p (1280×720) | **2.21 ms** (p95 2.24) | 5.97 ms (p95 5.98) | **2.70×** |
| giraffe 640 (640×640)  | **3.24 ms** (p95 3.49) | 5.39 ms (p95 5.41) | **1.67×** |

These confirm the earlier `crates/image/examples/nvjpeg_decode` smoke-test figures (2.04 ms /
3.31 ms). And nvJPEG output is already RGB and GPU-resident, so it also saves the
NV12→RGB convert and the host→GPU upload the CPU path still owes.

**Decode-only RGBI into a device pointer** (lower-level C++ probe data, same Orin,
full per-call stream sync):

| Image | Subsampling | Bitstream | nvJPEG (GPU_HYBRID) |
|-------|-------------|-----------|---------------------|
| 1280×800 | 4:4:4 | 107 KB | ~7.0 ms |
| 3840×2160 | 4:4:4 | 153 KB | ~17.8 ms |
| 3840×2160 | 4:4:4 | 3.28 MB | ~73.6 ms |

**Key points:**
- On Orin, `NVJPEG_BACKEND_HARDWARE` is unsupported (`nvjpegCreateEx` → status
  7); nvJPEG runs on **CUDA cores** (`GPU_HYBRID`/`DEFAULT`), not the NVJPG ASIC.
  To use the ASIC you would need the Jetson multimedia `NvJPEGDecoder` API — out
  of scope.
- The win is **GPU-resident output + a freed CPU**, not raw decode speed.
  Decode time tracks resolution and bitstream entropy (4K spans ~18–74 ms). The
  honest comparison is end-to-end **nvJPEG-RGB-decode + `convert()`** vs
  **CPU-NV12-decode + GPU-upload + NV12→RGB `convert()`** — to be captured
  on-target as `codec/jpeg/nvjpeg/e2e/*` once deployed.
- Each frame pays two GL-thread `cuda_map`/unmap round-trips (the PBO must be
  unmapped before `convert()` samples it); the `nvjpeg_map`/`nvjpeg_unmap`
  tracing spans isolate that cost. Keep the decode-source pool on a GL thread
  off the `convert()` hot path to avoid contention.
- _Status:_ the end-to-end codec path and the in-tree `codec/jpeg/nvjpeg/rgbi/*`
  bench cells are validated on-target (tables above, captured 2026-06-15). The
  multi-worker e2e/overlap comparison still needs a profiler-level deployment to
  drive concurrent decoders, and is to be captured here then.

### EXIF Orientation Overhead

**Data collected:** 2026-05-17 (codec at b77df09..4e04dc4 + EXIF coverage). Each
fixture in `testdata/zidane_exif_<N>.{jpg,png}` carries identical pixel data
for `zidane.jpg` (1280×720) with only the EXIF orientation tag varying
(N = 1..=8, per the EXIF/TIFF spec). Apply-false rows verify the fixtures
share scan/IDAT content; apply-true rows measure the cost of the in-place
byte rearrangement performed by `codec/src/exif.rs::apply_exif_u8`.

Orientation reference: **1**=identity, **2**=mirror-H, **3**=180°, **4**=mirror-V,
**5**=90° CW + mirror-H, **6**=90° CW, **7**=90° CCW + mirror-H, **8**=90° CCW.

#### JPEG decode (`zidane.jpg` 1280×720 → RGB u8, median over n=100)

| Platform | apply | o=1 | o=2 (flip-H) | o=3 (180°) | o=4 (mirror-V) | o=5 (rot+flip) | o=6 (90°) | o=7 (rot+flip) | o=8 (270°) |
|----------|-------|------|------|------|------|------|------|------|------|
| PC (x86_64, host) | false | 1.6 ms | 1.6 ms | 1.6 ms | 1.6 ms | 1.6 ms | 1.6 ms | 1.7 ms | 1.6 ms |
| PC (x86_64, host) | true  | 1.7 ms | 3.4 ms | 3.6 ms | 5.3 ms | 5.4 ms | 3.4 ms | 5.3 ms | 3.6 ms |
| imx95-frdm (A55) | false | 13.8 ms | 13.8 ms | 13.8 ms | 13.8 ms | 13.8 ms | 13.8 ms | 13.9 ms | 13.8 ms |
| imx95-frdm (A55) | true  | 13.9 ms | 31.1 ms | 31.5 ms | 48.2 ms | 47.1 ms | 30.5 ms | 47.1 ms | 30.5 ms |
| imx8mp-frdm (A53) | false | 14.4 ms | 14.3 ms | 14.3 ms | 14.4 ms | 14.4 ms | 14.4 ms | 14.3 ms | 14.3 ms |
| imx8mp-frdm (A53) | true  | 14.3 ms | 30.0 ms | 35.5 ms | 49.7 ms | 55.7 ms | 41.3 ms | 55.6 ms | 41.3 ms |
| orin-nano (A78AE) | false | 6.4 ms | 6.4 ms | 6.4 ms | 6.4 ms | 6.4 ms | 6.4 ms | 6.4 ms | 6.4 ms |
| orin-nano (A78AE) | true  | 6.4 ms | 12.0 ms | 13.8 ms | 19.1 ms | 17.5 ms | 12.2 ms | 17.5 ms | 12.3 ms |
| rpi5-hailo (A76) | false | 4.1 ms | 4.1 ms | 4.2 ms | 4.1 ms | 4.2 ms | 4.1 ms | 4.2 ms | 4.2 ms |
| rpi5-hailo (A76) | true  | 4.2 ms | 8.7 ms | 10.3 ms | 14.4 ms | 15.6 ms | 11.4 ms | 15.6 ms | 11.4 ms |

#### PNG decode (`zidane_exif_<N>.png` 1280×720 → RGB u8, median over n=100)

| Platform | apply | o=1 | o=2 | o=3 | o=4 | o=5 | o=6 | o=7 | o=8 |
|----------|-------|------|------|------|------|------|------|------|------|
| PC (x86_64, host) | false | 5.6 ms | 5.6 ms | 5.6 ms | 5.5 ms | 5.6 ms | 5.6 ms | 5.6 ms | 5.6 ms |
| PC (x86_64, host) | true  | 5.6 ms | 8.1 ms | 7.6 ms | 9.9 ms | 9.9 ms | 7.4 ms | 9.7 ms | 7.5 ms |
| imx95-frdm (A55) | false | 38.1 ms | 38.1 ms | 38.1 ms | 38.0 ms | 38.0 ms | 38.0 ms | 38.0 ms | 38.1 ms |
| imx95-frdm (A55) | true  | 38.1 ms | 55.4 ms | 55.8 ms | 72.3 ms | 71.1 ms | 54.5 ms | 71.1 ms | 54.4 ms |
| imx8mp-frdm (A53) | false | 41.5 ms | 41.6 ms | 41.6 ms | 41.6 ms | 41.5 ms | 41.5 ms | 41.6 ms | 41.6 ms |
| imx8mp-frdm (A53) | true  | 41.6 ms | 56.8 ms | 62.4 ms | 76.6 ms | 82.8 ms | 68.4 ms | 82.6 ms | 68.3 ms |
| orin-nano (A78AE) | false | 19.4 ms | 19.4 ms | 19.4 ms | 19.4 ms | 19.4 ms | 19.4 ms | 19.4 ms | 19.4 ms |
| orin-nano (A78AE) | true  | 19.4 ms | 25.1 ms | 26.8 ms | 32.2 ms | 30.6 ms | 25.2 ms | 30.6 ms | 25.2 ms |
| rpi5-hailo (A76) | false | 14.5 ms | 14.5 ms | 14.5 ms | 14.5 ms | 14.5 ms | 14.5 ms | 14.5 ms | 14.5 ms |
| rpi5-hailo (A76) | true  | 14.5 ms | 19.2 ms | 20.7 ms | 24.8 ms | 25.9 ms | 21.8 ms | 25.9 ms | 21.8 ms |

**Key Observations:**
- **`apply_false` is flat across all 8 orientations on every platform** — the
  fixtures truly share scan/IDAT content, and the codec doesn't waste cycles
  on the EXIF tag in the no-rotation path.
- **Orientation 1 with `apply_true` matches `apply_false` exactly** — the codec
  reads the EXIF tag, sees identity, and skips `apply_exif_u8` entirely. No
  hidden overhead for callers that pass `apply_exif=true` defensively.
- **In-place transforms (o=3 = 180°) cost roughly +1 byte-rearrangement per
  pixel.** On the imx8mp Cortex-A53 the delta is +21 ms for JPEG (+147% over
  the baseline 14.4 ms) which approximates the DDR write-bandwidth-limited
  cost of touching 2.7 MB (1280×720×3 RGB bytes) once.
- **90°/270° rotations (o=6, o=8) cost the same as 180°** despite needing a
  scratch buffer — the allocation is negligible vs the byte-rearrangement
  itself, and the codec reuses the rotation scratch across calls (see
  `state.exif_scratch` in `crates/codec/src/jpeg/mod.rs`).
- **Combined rotate+flip (o=4, o=5, o=7) costs ~2× the rotation alone** —
  the codec applies flip-H as a separate pass after rotation, so each
  transform is paid for in full DDR bandwidth.
- **Cortex-A55 (imx95-frdm) is faster than A53 (imx8mp-frdm) on the
  combined-transform paths** despite the A55 being only marginally faster on
  the JPEG decode itself (~14 ms vs 14.4 ms baseline). The A55's wider
  load/store pipeline accelerates the byte rearrangement (apply_exif_u8 is
  pure memcpy-shaped work).
- **A76 (rpi5-hailo) and A78AE (orin-nano) are the fastest by a wide margin**
  — A76 decode at 4.2 ms vs A55 at 14 ms, and EXIF rotation overhead scales
  proportionally. EXIF on these platforms is essentially free at frame
  cadences ≥ 30 Hz.
- **PNG EXIF overhead is roughly the same absolute cost as JPEG EXIF** — the
  transform operates on the post-decode pixel buffer, not on the source
  bytes. The PNG baseline is just higher because zune-png decode itself is
  slower than the custom JPEG decoder.

**Reproduce:**
```bash
# Host
source venv/bin/activate
python scripts/generate_exif_fixtures.py        # one-shot; commit fixtures
EDGEFIRST_TESTDATA_DIR=$(pwd)/testdata cargo bench -p edgefirst-codec --bench codec_benchmark

# Cross-compile and deploy to embedded target
cargo zigbuild -p edgefirst-codec --bench codec_benchmark --release --target aarch64-unknown-linux-gnu
BIN=$(ls -t target/aarch64-unknown-linux-gnu/release/deps/codec_benchmark-* | grep -v "\.d$" | head -1)
for host in imx8mp-frdm imx95-frdm rpi5-hailo orin-nano; do
    ssh "$host" "mkdir -p ~/bench/testdata"
    scp "$BIN" "$host:~/bench/codec_benchmark"
    scp testdata/zidane*.jpg testdata/zidane*.png "$host:~/bench/testdata/"
    ssh "$host" "chmod +x ~/bench/codec_benchmark"
    ssh "$host" "EDGEFIRST_TESTDATA_DIR=~/bench/testdata ~/bench/codec_benchmark" \
        | tee /tmp/exif_bench_$host.log
done
```

### Mask Rendering

**Data re-collected:** June 11, 2026 (GL proto-dispatch convergence;
median of 3 runs × n=100; imx8mp/imx95/rpi5 at PR #110 head,
jetson-orin-nano at PR #109 head). CPU / x86 / macOS rows retain the
v0.15.0 capture — those paths are measured by the batched-GEMM tables
below.

**640×640 RGBA destination, ~2 detections (YOLOv8n-seg):**

| Platform | Compute | Buffer | draw_decoded_masks (pre-decoded) | draw_proto_masks (fused) | hybrid_materialize_and_draw |
|----------|---------|--------|-------------------------------|------------------------|---------------------------|
| imx8mp-frdm | GL | DMA | 2.7 ms | 274 ms | 6.1 ms |
| imx8mp-frdm | CPU | Heap | 5.3 ms | 77.8 ms | 8.3 ms |
| imx95-frdm | GL | DMA | 1.2 ms | 24.4 ms | 4.8 ms |
| imx95-frdm | CPU | Heap | 5.3 ms | 76.2 ms | 8.4 ms |
| rpi5-hailo | GL | DMA | 1.3 ms | 7.7 ms | 2.4 ms |
| rpi5-hailo | CPU | Heap | 885 us | 14.5 ms | 1.7 ms |
| jetson-orin-nano | GL | DMA | 518 us | 3.0 ms | 1.4 ms |
| jetson-orin-nano | CPU | Heap | 873 us | 22.1 ms | 1.9 ms |
| x86-desktop | CPU | Heap | 648 us | 5.1 ms | 635 us |
| mbp-m2-max | CPU | Heap | 215 us | 6.9 ms | 419 us |

**Hybrid Path Comparison (CPU materialize + GL overlay vs fused GPU):**

The hybrid path decodes masks on CPU (`materialize_segmentations`) then overlays via GL (`draw_decoded_masks`). This is faster than fused GPU `draw_proto_masks` on all tested platforms — dramatically so on Vivante, whose fragment scheduling falls off a cliff on the per-quad sigmoid work (see the eager-materialize guidance in `crates/image/ARCHITECTURE.md`). The auto-selection in `ImageProcessor::draw_proto_masks()` prefers the hybrid path when both CPU and OpenGL backends are available. The hybrid column is ~3× faster than the v0.15.0 capture across the boards thanks to the v0.22 batched-GEMM materialise.

**New in v0.15.0:** The `materialize_masks()` API exposes the CPU materialization step as a first-class operation, enabling a three-stage pipeline (`decode_proto` → `materialize_masks` → `draw_decoded_masks`) where users can inspect, export, or fork the intermediate masks for analytics before rendering. Mask values are continuous sigmoid confidence (u8 0-255), not binary thresholded.

| Platform | Full GPU (GL draw_proto_masks) | Hybrid (GL) | Speedup | Auto draw_proto_masks |
|----------|-------------------------------|-------------|---------|----------------------|
| imx8mp-frdm | 274 ms | 6.1 ms | **44.9×** | 4.0 ms |
| imx95-frdm | 24.4 ms | 4.8 ms | **5.1×** | 3.7 ms |
| rpi5-hailo | 7.7 ms | 2.4 ms | **3.2×** | 1.6 ms |
| jetson-orin-nano | 3.0 ms | 1.4 ms | **2.1×** | 1.2 ms |

**Mask Decode Cost (CPU-only, measured in mask_benchmark):**

| Platform | Proto Decode (NMS+coefficients) | Full Materialize (NMS+coefficients+pixels) |
|----------|-------------------------------|-------------------------------------------|
| imx8mp-frdm | 916 us | 4.0 ms |
| imx95-frdm | 712 us | 3.6 ms |
| rpi5-hailo | 186 us | 1.2 ms |
| jetson-orin-nano | 194 us | 1.1 ms |
| x86-desktop | 381 us | 903 us |
| mbp-m2-max | 222 us | 862 us |

### materialize_masks Batched-GEMM Optimisation

`ImageProcessor::materialize_masks` previously ran a per-detection scalar
kernel (per-pixel bilinear sample + K-wide dot + sigmoid). The validation
workload — COCO-style with `max_det=100` at low score thresholds — degraded
linearly with the detection count, dominating the HAL output stage.

The new path:

- **Single batched GEMM** at proto resolution: `coeffs (N, K) · protos.T (K, H·W)`
  via `ndarray::linalg::general_mat_mul` (backed by `matrixmultiply` —
  pure-Rust SIMD, no new deps). Runs once per frame regardless of N.
- **Rayon-parallel per-detection finalisation**: each worker reads its row
  of the logits buffer, applies `fast_sigmoid` (Proto resolution) or
  `fast_sigmoid` + bilinear upsample (Scaled resolution), and emits the
  final `Segmentation`.
- **Pooled scratch**: `MaskScratch` on `CPUProcessor` reuses the
  dequantised-protos and logits buffers across calls — validation loops
  amortise allocations over all frames.
- **Fused fallback** retained for small N where the batched up-front cost
  outweighs the per-detection savings:
  - `MaskResolution::Proto`: batched at `N >= 16`
  - `MaskResolution::Scaled`: batched at `N >= 2`

Measured A/B in `mask_benchmark` (`materialize_masks/{proto_res,scaled_640x640}`)
with the env-gated `EDGEFIRST_LEGACY_MATERIALIZE=1` toggle.

**MaskResolution::Proto (median, ms; legacy → batched):**

| Platform | N=8 | N=16 | N=32 | N=64 | N=100 |
|----------|-----|------|------|------|-------|
| imx8mp-frdm  (4× A53)   | 5.9→5.9   (1.00×) | 11.7→13.2 (0.89×) | 23.3→17.7 (1.32×) | 46.6→27.4 (1.70×) | 72.7→38.8 (1.87×) |
| imx95-frdm   (6× A55)   | 6.0→5.9   (1.02×) | 11.8→11.5 (1.03×) | 23.5→16.3 (1.44×) | 46.9→25.7 (1.83×) | 73.2→36.8 (1.99×) |
| rpi5-hailo   (4× A76)   | 1.5→1.9   (0.79×) | 3.0→2.7   (1.11×) | 6.0→4.0   (1.50×) | 11.9→6.7  (1.78×) | 18.6→9.7  (1.92×) |
| x86-desktop  (20-core)  | 0.56→0.59 (0.95×) | 1.1→1.1   (1.00×) | 2.3→1.9   (1.21×) | 4.5→1.8   (2.50×) | 7.0→2.6   (2.69×) |

**MaskResolution::Scaled 640×640 (median, ms; legacy → batched):**

| Platform | N=2 | N=8 | N=16 | N=32 | N=64 | N=100 |
|----------|-----|-----|------|------|------|-------|
| imx8mp-frdm  (4× A53)   | 29.8→18.0 (1.66×) | 115.5→22.1 (5.23×)  | 229.7→33.1 (6.94×)  | 458.0→55.8 (8.21×)  | 914.6→101.5 (9.01×)  | **1400→153** (**9.13×**)  |
| imx95-frdm   (6× A55)   | 29.8→17.3 (1.72×) | 115.5→18.2 (6.35×)  | 229.7→27.9 (8.23×)  | 458.2→43.6 (10.51×) | 915.0→77.0  (11.88×) | **1400→114** (**12.28×**) |
| rpi5-hailo   (4× A76)   | 9.7→3.8   (2.55×) | 37.6→5.2   (7.23×)  | 74.8→8.1   (9.23×)  | 149.2→14.7 (10.15×) | 298.0→27.3 (10.92×) | **466→42**   (**10.95×**) |
| x86-desktop  (20-core)  | 9.6→3.5   (2.74×) | 37.2→2.2   (16.91×) | 74.0→2.5   (29.60×) | 147.9→4.0  (36.98×) | 295.0→6.9  (42.75×) | **461→10**   (**44.74×**) |

**Notes:**

- The Proto path gains less than the Scaled path because its per-detection
  ROI kernel only touches `bbox_area × K` pixels — small at any N. The
  batched path always pays a full-plane `H × W × K` dequant + GEMM, so it
  only wins once aggregate ROI work exceeds that fixed cost.
- The Scaled path gains massively because the legacy kernel did
  `bbox_area × K × 4` ops per detection at output resolution (the ×4 from
  bilinear). The batched path does the heavy K-wide dot at proto resolution
  (160×160 = 25,600 vs 640×640 = 409,600 sample points → 16× fewer
  dot-product ops) and reduces the per-detection work to a cheap
  `bbox_area` bilinear upsample on the flat logit plane.
- The Proto regression at N=8 on rpi5-hailo (0.79×) and N=16 on
  imx8mp-frdm (0.89×) sit just above each platform's crossover. The
  threshold of 16 is a conservative cross-platform compromise; A76 and x86
  benefit from a lower threshold, A53 prefers a higher one. Tunable via
  the `BATCHED_GEMM_MIN_N_PROTO` constant.
- The Scaled path is a clear win on every tested platform from N=2
  upward, scaling cleanly to ~9–45× at N=100 depending on cache hierarchy
  and SIMD width.

### NumPy Interop Fast-Path

`Tensor.from_numpy()` (and the implicit numpy → HAL conversions used by
`Decoder.decode_proto()` and friends) selects one of three paths in
`copy_numpy_to_tensor_dyn` (`crates/python/src/tensor.rs:339`) based on
the source array's strides:

| Path | Source layout | Strategy |
|---|---|---|
| 1 | Fully contiguous | Single `copy_from_slice` (memcpy), rayon-parallel ≥ 256 KiB |
| 2 | Strided with contiguous inner rows | Per-row memcpy iterating outer dimensions |
| 3 | Fully strided (no contiguous inner row) | Internal `np.ascontiguousarray()` materialisation, then Path 1 memcpy |

The Path 3 pattern matches the layout HailoRT returns natively: a
`(1, channels, anchors)` view obtained by `arr.transpose(0, 2, 1)`
over a `(1, anchors, channels)` backing buffer. Prior to PR #58, the
Path 3 branch iterated element-by-element over the strided ndarray
view, which broke vectorisation and incurred stride arithmetic per
load. The fix calls `np.ascontiguousarray()` internally, which uses
numpy's vectorized C strided→contig pass, then falls back to the
Path 1 memcpy.

**rpi5-hailo, `(1, 116, 8400)` f32 transposed view:**

| Variant | Time per call | Ratio vs fast path |
|---|---|---|
| Manual `np.ascontiguousarray + from_numpy(contig)` (legacy workaround) | ≈ 6.5 ms | 1.00× (baseline) |
| `from_numpy(strided)` automatic fast path (PR #58) | ≈ 6.5 ms | 1.0–1.5× (perf-sanity test bound) |
| `from_numpy(strided)` legacy element-wise loop | ≈ 27 ms | ≈ 4× slower |

**Implication for callers:** drop manual `np.ascontiguousarray()`
workarounds — the fast path is automatic. Pre-applying it above HAL
adds a redundant copy.

The behaviour is pinned by `test_from_numpy_hailort_shape` (correctness)
and `test_from_numpy_hailort_shape_perf_sanity` (≤ 1.5× slower than the
manual workaround) in `tests/test_tensor.py`.

---

## C API Preprocessing Benchmark (`bench_preproc`)

This section documents results from the C API preprocessing benchmark, which measures end-to-end `hal_image_processor_convert()` latency as seen by a C caller — including EGL/DMA-buf import, GPU dispatch, readback, and any tensor lifecycle overhead. The benchmark is the primary evidence base for the tensor reuse recommendations in ARCHITECTURE.md.

**Source:** `crates/capi/tests/bench_preproc.c`

**Reference:** ARCHITECTURE.md § "C API Performance Recommendations (DMA-BUF / EGL Path)"

### Test Configuration

| Parameter | Value |
|-----------|-------|
| Input | 1920×1080 NV12 or YUYV (DMA-buf) |
| Output | 640×640 letterbox |
| Warmup | 5 iterations (unmeasured) |
| Measured | 100 iterations |
| Reported | Avg, Min, Max (ms) |

The benchmark exercises six format paths (NV12/YUYV × RGBA/RGB/PlanarRgb, each in u8 and i8 variants), then adds three lifecycle scenarios: recreating the output tensor per frame, chaining two convert calls, and rotating through a four-buffer pool.

### Cross-Platform Summary

Key averages for the most common format paths (1080p → 640×640 letterbox):

| Conversion | i.MX 95 (Mali) | i.MX 8MP (Vivante) | x86 (GTX 1080 PBO) |
|------------|---------------:|-------------------:|-------------------:|
| NV12→RGBA | 1.52 ms | 3.39 ms | 1.22 ms |
| NV12→RGB | 3.68 ms | 14.40 ms | 1.03 ms |
| NV12→PlanarRgb | 3.67 ms | 17.51 ms | 1.21 ms |
| YUYV→RGBA | 1.12 ms | 1.72 ms | 1.51 ms |
| YUYV→RGB | 3.32 ms | 11.95 ms | 1.44 ms |
| YUYV→PlanarRgb | 2.29 ms | 5.62 ms | 1.58 ms |
| **Recreate tensor/frame** | **5.00 ms** | **5.61 ms** | **1.23 ms** |
| **Buffer pool (4 bufs)** | **1.58 ms** | **3.44 ms** | **1.27 ms** |

> **Key insight:** NV12→RGB and NV12→PlanarRgb are 14–20 ms on i.MX 8MP because these paths trigger CPU fallback on Vivante GC7000UL (NV12→planar is blocked due to GPU hang, packed RGB is 3–4× slower than G2D). On i.MX 95 (Mali) and x86 (PBO), all paths stay under 5 ms.

### Per-Platform Detail

#### i.MX 95-EVK (Mali G310, single-pass GL, DMA-buf)

| Benchmark | Avg (ms) | Min (ms) | Max (ms) |
|-----------|----------|----------|----------|
| NV12→RGBA | 1.52 | 1.43 | 1.83 |
| NV12→RGBA I8 | 1.54 | 1.43 | 2.93 |
| NV12→RGB | 3.68 | 3.50 | 4.00 |
| NV12→RGB I8 | 4.95 | 4.72 | 5.78 |
| NV12→PlanarRgb | 3.67 | 3.39 | 4.22 |
| NV12→PlanarRgb I8 | 3.65 | 3.38 | 4.09 |
| YUYV→RGBA | 1.12 | 1.05 | 1.17 |
| YUYV→RGBA I8 | 1.23 | 1.15 | 1.32 |
| YUYV→RGB | 3.32 | 3.13 | 3.61 |
| YUYV→RGB I8 | 4.68 | 4.39 | 5.30 |
| YUYV→PlanarRgb | 2.29 | 2.21 | 2.48 |
| YUYV→PlanarRgb I8 | 2.60 | 2.55 | 2.75 |
| Recreate tensor per frame | 5.00 | 4.64 | 5.43 |
| Chained (NV12→RGBA→PlanarRgb) | 4.12 | 4.00 | 4.54 |
| Buffer pool (4 bufs rotating) | 1.58 | 1.48 | 1.70 |

#### i.MX 8M Plus EVK-06 (Vivante GC7000UL, DMA-buf)

| Benchmark | Avg (ms) | Min (ms) | Max (ms) |
|-----------|----------|----------|----------|
| NV12→RGBA | 3.39 | 3.09 | 3.79 |
| NV12→RGBA I8 | 3.29 | 3.13 | 3.81 |
| NV12→RGB | 14.40 | 13.06 | 15.86 |
| NV12→RGB I8 | 18.00 | 16.64 | 18.89 |
| NV12→PlanarRgb | 17.51 | 16.84 | 25.29 |
| NV12→PlanarRgb I8 | 19.75 | 18.64 | 26.45 |
| YUYV→RGBA | 1.72 | 1.66 | 1.91 |
| YUYV→RGBA I8 | 1.70 | 1.63 | 1.87 |
| YUYV→RGB | 11.95 | 10.68 | 12.69 |
| YUYV→RGB I8 | 15.01 | 13.85 | 16.20 |
| YUYV→PlanarRgb | 5.62 | 5.24 | 6.32 |
| YUYV→PlanarRgb I8 | 5.82 | 5.31 | 6.68 |
| Recreate tensor per frame | 5.61 | 5.01 | 6.70 |
| Chained (NV12→RGBA→PlanarRgb) | 8.53 | 8.03 | 9.98 |
| Buffer pool (4 bufs rotating) | 3.44 | 3.15 | 4.11 |

> **Note:** NV12→RGB and NV12→PlanarRgb are 14–20 ms because these paths hit CPU fallback on Vivante (NV12→planar is blocked at the GL layer; packed RGB uses G2D which is slower than on Mali). For latency-sensitive pipelines on i.MX 8MP, prefer NV12→RGBA (3.4 ms) and rely on the VX Delegate CameraAdaptor for the final layout conversion inside the NPU graph.

#### x86 Desktop (NVIDIA GTX 1080, PBO path)

| Benchmark | Avg (ms) | Min (ms) | Max (ms) |
|-----------|----------|----------|----------|
| NV12→RGBA | 1.22 | 1.07 | 1.92 |
| NV12→RGBA I8 | 1.51 | 1.36 | 2.51 |
| NV12→RGB | 1.03 | 0.94 | 2.45 |
| NV12→RGB I8 | 1.12 | 1.02 | 1.57 |
| NV12→PlanarRgb | 1.21 | 1.08 | 1.73 |
| NV12→PlanarRgb I8 | 1.25 | 1.16 | 3.65 |
| YUYV→RGBA | 1.51 | 1.41 | 2.15 |
| YUYV→RGBA I8 | 1.97 | 1.69 | 2.65 |
| YUYV→RGB | 1.44 | 1.33 | 3.66 |
| YUYV→RGB I8 | 1.49 | 1.37 | 2.88 |
| YUYV→PlanarRgb | 1.58 | 1.45 | 2.13 |
| YUYV→PlanarRgb I8 | 1.67 | 1.51 | 4.26 |
| Recreate tensor per frame | 1.23 | 1.10 | 2.09 |
| Chained (NV12→RGBA→PlanarRgb) | 1.47 | 1.34 | 2.14 |
| Buffer pool (4 bufs rotating) | 1.27 | 1.12 | 3.01 |

> **Note:** All format paths are 1.0–2.0 ms on this platform. The recreate-tensor penalty is negligible (1.0×) because the PBO path does not use `EGLImage` — output tensors are bound directly as PBO destinations so there is no EGL image cache involved.

### Tensor Reuse Impact

Recreating the output tensor on every frame forces a new DMA-buf allocation, a new `EGLImage` import, and a new `GL_TEXTURE_EXTERNAL_OES` binding for that buffer. On EGLImage-based platforms (DMA-buf path), this cache miss dominates — the raw GPU work for the conversion itself is not the bottleneck.

| Platform | Reuse avg | Recreate avg | Penalty | Buffer pool avg | Pool vs. reuse |
|----------|----------:|-------------:|--------:|----------------:|---------------:|
| i.MX 95 (Mali) | 1.52 ms | 5.00 ms | **3.3×** | 1.58 ms | 1.04× |
| i.MX 8MP (Vivante) | 3.39 ms | 5.61 ms | **1.7×** | 3.44 ms | 1.01× |
| x86 (GTX 1080 PBO) | 1.22 ms | 1.23 ms | **1.0×** | 1.27 ms | 1.04× |

The reuse baseline uses a single source tensor held alive across all 100 frames. The recreate variant calls `hal_tensor_free` and `hal_image_processor_create_image` on the **source** tensor every frame before converting (the destination tensor is reused). The buffer pool variant rotates through four pre-allocated source tensors in round-robin order (simulating a V4L2 buffer pool with multiple frames in flight).

**Buffer pool matches single-tensor reuse on both embedded platforms** (1.01–1.04×). This confirms that the EGL image cache works correctly as long as the same buffer objects are reused — the pool size does not matter as long as each buffer is seen again before its cache entry is evicted. The recreate penalty is entirely attributable to EGL import overhead, not to DMA-buf allocation itself.

**The penalty is zero on PBO** (x86 desktop) because `PboTensor` uses `glBindBuffer` on a pre-allocated PBO, with no `EGLImage` lifecycle. Recreating a PBO tensor is still cheaper than an EGL import on Mali/Vivante.

#### Why This Matters for Embedded Pipelines

A 30 fps camera pipeline has a 33 ms per-frame budget. On i.MX 95:

- Single `convert()` with tensor reuse: **1.5 ms** (4.5% of budget)
- Single `convert()` with recreated tensor: **5.0 ms** (15% of budget) — a 3.5 ms waste
- Chained two-step pipeline (NV12→RGBA→PlanarRgb) with reuse: **4.1 ms** (12% of budget)
- Same chained pipeline if both output tensors are recreated: ~**10 ms** (30% of budget)

On i.MX 8MP, where the per-convert budget is already tighter due to Vivante driver characteristics, the same two-step chain with recreated tensors consumes ~**11 ms** — one third of the entire 33 ms frame budget before inference even begins.

**Conclusion: tensor reuse is not optional on embedded. Allocate output tensors once at pipeline startup and reuse them every frame. Use a buffer pool when multiple frames are in flight concurrently.**

### Running `bench_preproc`

```bash
# Cross-compile for aarch64
cargo-zigbuild zigbuild --target aarch64-unknown-linux-gnu --release -p edgefirst-hal-capi

# The C benchmark is built by the capi crate's build.rs; the binary is at:
#   target/aarch64-unknown-linux-gnu/release/bench_preproc

# Deploy and run on target
scp target/aarch64-unknown-linux-gnu/release/bench_preproc user@target:/tmp/
ssh user@target '/tmp/bench_preproc'
```

The binary requires a DMA-heap device (`/dev/dma_heap/linux,cma` or `/dev/dma_heap/system`) and an EGL display. On x86 with NVIDIA, it automatically falls back to the PBO path.

> **CI environments:** Set `EDGEFIRST_FORCE_BACKEND=cpu` to skip software GL
> detection overhead. Without this, the GL backend will attempt EGL init,
> detect llvmpipe/swrast, and fall back to CPU — adding ~200ms to startup.

---

## Decode → Letterbox Pipeline Benchmark (`decode_pipeline_benchmark`)

This section documents the end-to-end JPEG decode → GPU letterbox convert
pipeline using the zero-allocation, strided-buffer pattern. The input tensor
is allocated larger than all test images so that the JPEG decoder writes into
a strided sub-region; the `ImageProcessor::convert()` then performs a
letterbox resize into a 640×640 model-input tensor.

**Source:** `crates/image/examples/pipeline_demo.rs`,
`crates/image/benches/decode_pipeline_benchmark.rs`

**Key design:** All tensors are allocated once during init. After warmup, the
hot loop performs **zero heap allocations** — verified via `strace` filtering
for `brk` and `MAP_ANONYMOUS` mmap calls during the `HOT LOOP START/END`
markers.

**Memory modes:**
- **DMA-BUF** (imx8mp, imx95, rpi5): tensors backed by Linux DMA-heap for
  zero-copy EGL image import. Verified zero heap allocations on all three.
- **CPU/Heap** (x86, orin-nano): tensors backed by standard heap allocation
  with CPU-only convert path. Verified zero allocations.

### Results (collected 2026-05-17)

All times are median over 100 iterations after 10× warmup per combination.

#### imx8mp-frdm (Cortex-A53, Vivante GC7000UL, DMA-BUF)

| Image | Output | Decode | Convert | Total |
|-------|--------|-------:|--------:|------:|
| zidane 1280×720 | HWC (stride=1920) | 16,723 µs | 6,465 µs | 23,188 µs |
| giraffe 640×640 | HWC (stride=1920) | 14,220 µs | 3,468 µs | 17,688 µs |
| zidane 1280×720 | CHW (planar) | 16,680 µs | 7,435 µs | 24,115 µs |
| giraffe 640×640 | CHW (planar) | 14,223 µs | 4,264 µs | 18,487 µs |

#### imx95-frdm (Cortex-A55, Mali GPU, DMA-BUF)

| Image | Output | Decode | Convert | Total |
|-------|--------|-------:|--------:|------:|
| zidane 1280×720 | HWC (stride=1920) | 16,130 µs | 5,598 µs | 21,728 µs |
| giraffe 640×640 | HWC (stride=1920) | 13,624 µs | 3,308 µs | 16,932 µs |
| zidane 1280×720 | CHW (planar) | 16,137 µs | 6,344 µs | 22,481 µs |
| giraffe 640×640 | CHW (planar) | 13,692 µs | 3,649 µs | 17,341 µs |

#### rpi5-hailo (Cortex-A76, VideoCore V3D, DMA-BUF)

| Image | Output | Decode | Convert | Total |
|-------|--------|-------:|--------:|------:|
| zidane 1280×720 | HWC (stride=1920) | 4,620 µs | 2,283 µs | 6,903 µs |
| giraffe 640×640 | HWC (stride=1920) | 4,307 µs | 848 µs | 5,155 µs |
| zidane 1280×720 | CHW (planar) | 4,599 µs | 3,235 µs | 7,834 µs |
| giraffe 640×640 | CHW (planar) | 4,332 µs | 1,302 µs | 5,634 µs |

#### orin-nano (Cortex-A78AE, GL/PBO)

| Image | Output | Decode | Convert | Total |
|-------|--------|-------:|--------:|------:|
| zidane 1280×720 | HWC (stride=1920) | 6,438 µs | 1,008 µs | 7,446 µs |
| giraffe 640×640 | HWC (stride=1920) | 6,108 µs | 630 µs | 6,738 µs |
| zidane 1280×720 | CHW (planar) | 6,478 µs | 1,576 µs | 8,054 µs |
| giraffe 640×640 | CHW (planar) | 6,112 µs | 447 µs | 6,559 µs |

GL/PBO path now works after fixing a PBO deadlock in `setup_renderbuffer_non_dma`
(the GL thread called `dst.map()` which re-entered the GL thread channel). Convert
times improved ~36% vs CPU-only (1,008 µs vs 1,578 µs for zidane HWC).

#### x86-desktop (Ryzen, CPU-only)

| Image | Output | Decode | Convert | Total |
|-------|--------|-------:|--------:|------:|
| zidane 1280×720 | HWC (stride=1920) | 1,922 µs | 546 µs | 2,468 µs |
| giraffe 640×640 | HWC (stride=1920) | 1,704 µs | 39 µs | 1,743 µs |
| zidane 1280×720 | CHW (planar) | 1,876 µs | 696 µs | 2,572 µs |
| giraffe 640×640 | CHW (planar) | 1,766 µs | 231 µs | 1,997 µs |

### Zero-Allocation Verification

| Platform | Memory | Heap allocs in hot loop | Notes |
|----------|--------|------------------------:|-------|
| imx8mp-frdm | DMA-BUF | 0 | 1,400 MAP_SHARED mmap (DMA-BUF map/unmap for GPU, expected) |
| imx95-frdm | DMA-BUF | 0¹ | 1 `PROT_NONE` 64MB reservation (GPU address space, not heap) |
| rpi5-hailo | DMA-BUF | 0 | 1,400 MAP_SHARED mmap (DMA-BUF map/unmap for GPU, expected) |
| x86-desktop | CPU/Heap | 0 | Verified with `EDGEFIRST_FORCE_BACKEND=cpu` |

¹ The single `mmap(PROT_NONE, 64MB)` on imx95 is a GPU driver virtual
address space reservation with no read/write permissions — not a heap
allocation.

### Cross-Platform Analysis

- **Decode performance scales with CPU**: A76 (rpi5) is ~3.5× faster than
  A53 (imx8mp), matching the expected IPC and clock frequency difference.
  Orin A78AE falls between. x86 SSE2/SSE4.1 is fastest at ~1.9ms for 720p.
- **DMA-BUF convert benefits**: On DMA-BUF platforms the convert step uses
  zero-copy EGL image import — the GPU reads directly from the DMA-BUF
  without any CPU-side copy. This is most visible on rpi5 where HWC convert
  is only 848µs for 640×640.
- **Strided input overhead**: The strided decode (1280-wide tensor for
  640-wide images) adds no measurable overhead to convert — the GPU shader
  reads only the valid region via `src_rect`.

---

## Known Benchmark Gaps

### Missing Platforms

1. **maivin** — Primary production target (Torizon 7, same SoC as imx8mp-frdm).
   Pending Torizon image with benchmark tooling.

2. **jetson-orin-nano** — CPU and GL (RGBA/BGRA/Grey) benchmarks collected. YUV EGL import not supported (YUYV/NV12 GL pipeline rows show "—"). DMA-buf allocation benchmarks show anomalous scaling (720p slower than 4K) — likely CMA fragmentation during collection, needs re-run.

### Missing Buffer Strategy Coverage

3. **No forced Sync (memcpy) benchmarks** — No benchmark of the Sync fallback (`glTexImage2D`/`glReadnPixels` memcpy) to quantify the overhead of non-zero-copy GPU upload/readback.

### Known Performance Issues

4. **rpi5-hailo 4K DMA-buf allocation fails** — Mesa V3D driver cannot allocate DMA-buf textures at 3840×2160 for same-size conversion. OpenGL convert benchmarks at 4K produce GL errors on this platform.

5. **x86-desktop OpenGL cannot import YUV textures** — NVIDIA PBO path does not support YUYV/NV12/VYUY source textures. OpenGL letterbox and convert benchmarks show "—" for YUV source formats on this platform.

6. **imx95-frdm GL DMA-buf slower than PBO for letterbox** — v1.2 benchmarks labelled imx95-frdm GL as "DMA" but were actually running on PBO (EGL extension query bug caused DMA-buf roundtrip probe to fail). After fixing the extension query (v1.3), GL now uses true DMA-buf import. DMA-buf letterbox 1080p→640 YUYV→RGBA is 3.4ms vs 1.4ms on PBO — the DMA-buf import/export overhead exceeds PBO zero-copy bind. G2D improved (3.5ms from 3.9ms). Fused mask rendering (`draw_proto_masks`) dramatically improved: 5.2ms from 25.2ms (**4.8× faster**).

7. **BGRA framebuffer CPU byte-swap overhead** — BGRA textures as framebuffer attachments have GPU-dependent swizzle behavior (some implementations don't swizzle fragment shader output). Workaround uses RGBA format internally with CPU-side R↔B byte swaps on upload and readback. RGBA→BGRA conversion on imx95-frdm GL went from 3.4ms (v1.2 PBO, no swap needed) to 26.5ms (v1.3 DMA + CPU swap). CPU backend RGBA→BGRA is 24.5ms for reference.

8. **NV12→planar GPU hang on Vivante GC7000UL** — Rendering from an NV12 source texture (via `EGL_LINUX_DMA_BUF_EXT`) to a planar RGB framebuffer (MRT with 3× color attachments) causes an **unrecoverable GPU hang** on the Vivante GC7000UL (i.MX 8M Plus, galcore 6.4.11). The GPU command processor stalls permanently, the calling process enters kernel uninterruptible sleep (Ds state), cannot be killed even with SIGKILL, and the galcore driver state is corrupted system-wide — all subsequent GPU operations from any process hang until a full board reboot. YUYV→planar and NV12→packed work fine; the bug is specific to NV12 multi-plane texture + MRT output. The HAL never issues the single-pass shader: NV\* → planar always renders NV → RGBA intermediate → planar, which is now the plan on every GPU rather than a Vivante carve-out. See [crates/image/ARCHITECTURE.md § NV12/NV16/NV24 → PlanarRgb two-pass render](crates/image/ARCHITECTURE.md#nv12nv16nv24--planarrgb-two-pass-render) (EDGEAI-1180).

9. **rpi5-hailo GL planar at 4K is slow** — YUYV→8BPS/8BPS_i8 at 4K takes ~102ms on Mesa V3D GL, while CPU handles it in ~24ms. NV12→planar at 4K is ~26ms on GL. The bottleneck appears to be in Mesa V3D's MRT path when combined with high-resolution YUYV texture sampling.

10. **imx8mp-frdm GL packed RGB uses two-pass approach** — Vivante GC7000UL OpenGL does not support packed RGB output natively; the two-pass packed RGB packing shader renders to an RGBA intermediate then packs to RGB using a dedicated shader. This two-pass approach is now enabled but is 3-4× slower than G2D's hardware blitter for packed RGB output on Vivante (see footnote ¹ in 720p tables).

11. **rpi5-hailo GL packed RGB uses two-pass approach** — Same as imx8mp-frdm: Mesa V3D uses the two-pass packed RGB packing shader (RGBA intermediate then dedicated RGB packing shader). Now enabled but may be slower than CPU for some conversions on VideoCore.

### Missing Format Coverage

12. **No same-size convert benchmark for desktop Linux + Mesa x86_64 GL path** —
    The matrix has GL letterbox for that platform but no same-size convert
    column; the few rows are absorbed into the i.MX comparison.

13. **No NV16 benchmarks** — NV16 (4:2:2 semi-planar) CPU conversion exists but G2D/GL paths and benchmarks are missing.

### Missing Scenarios

14. **No PBO tensor allocation benchmarks** — Tensor allocation benchmarks cover Mem, SHM, and DMA but not PBO (which requires GL context).

15. ~~**No end-to-end pipeline benchmark**~~ — Resolved: `decode_pipeline_benchmark` and `pipeline_demo` cover the decode → letterbox convert pipeline. Full camera → inference → mask render cycle benchmark still pending.

16. ~~**Orin Nano GL/PBO pipeline hangs during warmup**~~ — Resolved: PBO deadlock in `setup_renderbuffer_non_dma` fixed by routing PBO destinations through `setup_renderbuffer_from_pbo` which avoids re-entering the GL thread channel. GL/PBO results now collected.

17. **mbp-m2-max GL rows are a stale capture, not a code limit** — This gap used
    to describe a real backend limitation: the standalone `MacosGlProcessor`
    shipped one fragment shader (BT.709 YUYV→RGBA, limited range) and returned
    `NotImplemented`/`NotSupported` for everything else. That backend is gone.
    Since 0.25.0 macOS drives the same `GLProcessorThreaded` engine as Linux
    through the ANGLE platform seam (`crates/image/src/gl/platform/macos.rs`)
    and inherits the engine's full conversion matrix — every format pair,
    resize, letterbox, rotation, flip, int8, masks.

    What has *not* happened is a re-collection: the macOS GL tables below still
    show only `convert/1920x1080/YUYV->RGBA` and `convert/3840x2160/YUYV->RGBA`,
    with the rest of the pipeline table identical to the CPU-only column. The
    numbers that are there are valid; the blanks now mean "not measured", not
    "not supported". Re-running the mbp-m2-max matrix would close this.

    One trap survives from the old text and is worth keeping: a new IOSurface
    format needs a FourCC + bytes-per-element mapping in
    `tensor::iosurface::image_fourcc_and_bpe` *and* a matching
    `EGL_TEXTURE_INTERNAL_FORMAT_ANGLE` entry in
    `image::gl::iosurface_import::ImageLayout::gl_internal_format`. Get those
    out of step and ANGLE validates the pairing at
    `eglCreatePbufferFromClientBuffer` time and returns a bare
    `EGL_BAD_ATTRIBUTE` that says nothing about which side is wrong.

---

## Changelog

| Version | Date | Changes |
|---------|------|---------|
| 3.12 | 2026-08-14 | Response to the v3.11 pre-publication review (`docs/BENCHMARKS_JPEG_REVIEW_v3.11.md`). Blocking: v0.22.1 `Image Codec Decode` JPEG rows marked superseded in place; nvJPEG June-vs-August conclusions reconciled (pre-optimization CPU baseline / single fixture vs corpus / full `load_image` vs decode-only); the "two releases behind" banner scoped to the GL/preprocessing matrix only. High: implemented the fused native-4:2:0→RGB decode path (`write_rgb_rows_420`, box chroma upsample) that was the one previously-unmeasured cell a libjpeg-turbo-literate reader would expect EdgeFirst to lose on — it wins everywhere instead, including a fast-class sweep with no loss cell; measured zune's `neon` feature is genuinely engaged (1.16–1.20× A/B on rpi5-hailo, not inferred from a features table); measured Wuffs' RGB row is on its 3-byte swizzle slow path (1.52–1.59× behind its native 4-byte output); published per-board libjpeg-turbo version/package-source/compiler table (none of the five original hosts actually run turbo 3.2+). Medium: added a Max-spread column to the headline table (and corrected the "sub-1%" prose it was previously not measuring); unified ratio (not percentage) framing throughout with YUV/RGB labels; quantified the zune greyscale-skip asymmetry at exactly zero images in the n=200 sample; ran a second independently-launched AWS instance per queue (bounds instance-bin variance at ≤0.6% on 4/5 queues, confirms a real ~7% effect on the m7i queue that barely moves the published ratio); added `mbp-m2-max` to the eight-arm sweep (largest accurate-class lead of any board, 1.41×). |
| 3.11 | 2026-08-13 | JPEG decode section re-captured and expanded for publication: eight arms (stb_image and Wuffs added, both pixel-parity-verified against djpeg accurate — Wuffs bit-exact, stb ±3 LSB), median-of-3-interleaved-rounds protocol with per-round spread, mean, and nonparametric 95% median CIs, identical strided image selection for every arm, and per-host build/run provenance. New sections: control corpora (val2017-yuv420 / lossless val2017-dri / CLIC 2025 4:2:0+4:4:4+DRI — the zune-vs-turbo gap is shown to be general, not corpus-driven, and the DRI claim is measured), AWS cloud baselines (Graviton2/3/4, Sapphire Rapids, Genoa — EdgeFirst fastest in every cell, largest leads on Graviton), hardware decoders (i.MX 95 V4L2 decode-only + full-res NV*→RGB second pass via CPU/GL with decode/convert split; Orin nvJPEG scoped as shared-CUDA-core GPU_HYBRID), and build & run provenance. Headline: accurate beats turbo `islow` everywhere (+12.9% A53 … +27.7% A78AE; 1.43–1.54× on Graviton). |
| 3.10 | 2026-08-13 | JPEG decode section rebuilt as the six-arm sweep (`decode-ab-sweep.sh`): EdgeFirst accurate + opt-in `fast` (`DctMethod::Fast`) vs turbo `islow`/`ifast`, zune-jpeg, and the image crate, across A53/A55/A76/A78AE/Rocket Lake. Accurate beats both turbo kernels everywhere (+11.7% A53 … +28.7% A78AE vs `islow`); `fast` beats `ifast` by 14–32% with its accuracy envelope stated (cosine ≥ 0.99985, PSNR ≥ 42 dB, max Δ 24 over 1000 COCO images). x86 re-captured after the SSE4.1 fused-RGB block-kernel fix. orin-nano row captured on the `adis-uav1` fallback (turbo baseline within 0.1% of the prior capture). |
| 3.9 | 2026-06-16 | 0.25.0 release refresh: full bench matrix re-collected on the converged-GL-engine code across imx8mp-frdm, imx95-frdm, rpi5-hailo, and jetson-orin-nano, plus the existing mbp-m2-max rows. Confirms the GL-convergence captures within measurement noise (imx95 GL 1080p YUYV→RGBA letterbox 1.2 ms → 957 µs, NV12→RGBA 1.2 ms → 830 µs); no GPU regressions. Allocation table updated for the imx8mp DMA-alloc improvement (38 ms → 1.8 ms at 720p). macOS GL rows remain the pre-convergence capture (Known Gap #17). |
| 3.8 | 2026-05-24 | macOS GL backend lands via ANGLE + IOSurface. `TensorMemory::Dma` extended to back IOSurface on macOS, with `is_gpu_buffer_available()` as the portable probe. Capture buffer-infrastructure numbers on mbp-m2-max for Mem/Shm/Dma (alloc 16 µs constant for IOSurface, memcpy 2–2.7× faster than SHM at every resolution). YUYV→RGBA same-size convert: 1.3× at 1080p, 4.8× at 4K vs CPU. Add mbp-m2-max **CPU-only** rows to letterbox / decoder / mask-decode / codec tables; add mbp-m2-max **GL** rows (YUYV→RGBA only) to the same-size format-conversion and 4K-convert tables. Letterbox GL rows pending Gap #17 closure. |
| 3.7 | 2026-05-22 | Add macOS platform (Apple M2 Max, `mbp-m2-max`) with CPU baseline benchmarks. |
| 3.6 | 2026-05-17 | Add decode→letterbox pipeline benchmark (`decode_pipeline_benchmark`, `pipeline_demo`): cross-platform results on imx8mp-frdm, imx95-frdm, rpi5-hailo, orin-nano, x86-desktop. Zero heap allocations verified on all DMA-BUF platforms via strace. Auto-detect DMA/PBO/Mem memory type. |
| 3.5 | 2026-05-18 | Perf-driven optimizations: 11-bit Huffman LUT (was 9-bit); batch byte-stuffing in bitstream refill; SSE4.1 IDCT with native `mullo_epi32` and `min/max` clamping; SSSE3 RGB shuffle store; NEON+SSE2 vectorised u8→f32/u16/i16 conversion. f32 decode now only 1.17× slower than u8 (was 4.0×); x86 RGB within 6% of image crate (was 25%); all results updated on 3 platforms. |
| 3.4 | 2026-05-17 | Add SSE2 SIMD kernels for x86-64: IDCT, YCbCr→RGB/RGBA/BGRA color conversion, and horizontal chroma upsample. x86 JPEG decode now 1.75× faster than scalar; within 25% of image crate for 720p and matches/beats it for 640×640. Update all x86-desktop results. |
| 3.3 | 2026-05-17 | Custom JPEG decoder with NEON SIMD: replace zune-jpeg wrapper with from-scratch baseline decoder; 17–23% faster than image crate on ARM; add NV12/BGRA/giraffe benchmarks; add x86-desktop baselines; collect on imx8mp, imx95, x86. |
| 3.2 | 2026-05-15 | Add `edgefirst-codec` image decode baselines on imx8mp-frdm and imx95-frdm: JPEG (720p, 4K, RGBA, f32, strided) and PNG (720p) vs image crate. |
| 3.1 | 2026-04-23 | `materialize_masks` batched-GEMM path: single GEMM at proto resolution + rayon-parallel per-detection finalisation + pooled `MaskScratch` buffers. Scaled 640×640 wins 1.7–45× across N=2–100; Proto wins 1.0–2.7× at N≥32. Cross-platform A/B measured on imx8mp-frdm, imx95-frdm, rpi5-hailo, x86-desktop |
| 3.0 | 2026-03-30 | v0.15.0 release: add jetson-orin-nano platform; refresh all benchmarks across 5 platforms; per-texture EGL binding optimization eliminates redundant EGLImageTargetTexture2DOES calls; add materialize_masks API with three-stage pipeline benchmarks; hybrid path 1.4–14.2× faster than fused GPU on all platforms |
| 2.2 | 2026-03-27 | Add collection date stamps to all benchmark result sections; add image_benchmark to benchmark binary table; note pending YoloSegDet2Way benchmark data in decoder section; note pending mask rendering optimization updates |
| 2.1 | 2026-03-23 | Add C API preprocessing benchmark (`bench_preproc`) results for i.MX 95-EVK (Mali), i.MX 8MP EVK-06 (Vivante), and x86 desktop (GTX 1080 PBO); add tensor reuse impact analysis (3.3× penalty on i.MX 95, 1.7× on i.MX 8MP, negligible on PBO); document buffer pool validation |
| 2.0 | 2026-03-20 | TensorDyn unification: auto-backend priority changed to OpenGL→G2D→CPU; always use two-pass packed RGB (rgb_direct removed); added per-platform forced-backend comparison tables at 720p; added u8/i8 DType benchmark variants; replaced 8BPi with 8BPS_i8 naming |
| 1.5 | 2026-03-18 | Remove stale Known Issue #3 (EDGEFIRST_FORCE_TRANSFER=pbo now implemented); documentation accuracy updates |
| 1.4 | 2026-03-13 | Add planar RGB (8BPS/8BPi) format benchmarks; document NV12→planar GPU hang on Vivante GC7000UL (blocked, CPU fallback); split letterbox tables into packed/planar; update mask rendering (imx8mp fused GPU improved 275ms→5.9ms); add rpi5 GL planar performance notes; refresh all platforms |
| 1.3 | 2026-03-12 | Update imx95-frdm after DMA-buf fix (GL now uses true DMA-buf, was PBO); BGRA CPU byte-swap workaround; fused mask rendering 4.8× faster |
| 1.2 | 2026-03-09 | Add hybrid mask benchmark and comparison table; auto-selection now prefers hybrid path |
| 1.1 | 2026-03-08 | Baseline results for imx8mp-frdm, imx95-frdm, rpi5-hailo, x86-desktop |
| 1.0 | 2026-03-04 | Initial document with strategy, platforms, and gap analysis |
