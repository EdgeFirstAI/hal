# EdgeFirst HAL - Benchmarks

**Version:** 1.5
**Last Updated:** March 18, 2026
**Status:** Removed stale known issues, updated documentation accuracy

---

## Overview

This document tracks EdgeFirst HAL performance across target platforms. It serves as a regression baseline: results are updated with each release to detect performance improvements or regressions introduced by code changes.

The benchmarking strategy tests **all compute backends** (CPU, OpenGL, G2D) with **all applicable buffer strategies** (DMA-buf, PBO, Sync) on every platform, including forcing non-default buffer paths on platforms that would normally prefer a different strategy. This ensures the full fallback chain is exercised and performance characteristics are understood for every deployment scenario.

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
**Destination formats:** RGBA, BGRA, RGB, RGB_INT8, GREY, PLANAR_RGB (8BPS), PLANAR_RGB_INT8 (8BPi)

### Format Abbreviations

| Abbreviation | Format | Description |
|-------------|--------|-------------|
| **8BPS** | PLANAR_RGB | 3× separate uint8 planes (R, G, B) |
| **8BPi** | PLANAR_RGB_INT8 | 3× separate uint8 planes (R, G, B) |
| **RGBi** | RGB_INT8 | Packed RGB, uint8 per channel |

### Measurement Methodology

All benchmarks use the `edgefirst-bench` custom harness which:
- Runs in-process (no fork) to avoid GPU driver crashes
- Executes warmup iterations (unmeasured) followed by measured iterations
- Reports: median, mean, min, max, p95, p99
- Reports throughput in MiB/s where applicable

**Standard parameters:** 10 warmup iterations, 100 measured iterations (adjustable per benchmark).

---

## Running Benchmarks

See [README.md § Benchmarking](README.md#benchmarking) for full instructions on running benchmarks locally, cross-compiling for aarch64, and deploying to target platforms.

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
| **Notes** | Mesa V3D driver; no RGB/RGBi packed GL support (two-pass disabled) |

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

---

## Benchmark Results

### Buffer Infrastructure

#### Allocation Latency

Measures `Tensor::new()` latency for each buffer type and resolution.

| Platform | Buffer | 720p (3.5 MB) | 1080p (7.9 MB) | 4K (31.6 MB) |
|----------|--------|---------------|-----------------|---------------|
| imx8mp-frdm | MEM | 309 us | 695 us | 2.8 ms |
| imx8mp-frdm | SHM | 26 us | 26 us | 26 us |
| imx8mp-frdm | DMA | 1.5 ms | 2.7 ms | 10.2 ms |
| imx95-frdm | MEM | 263 us | 596 us | 2.4 ms |
| imx95-frdm | SHM | 31 us | 31 us | 31 us |
| imx95-frdm | DMA | 988 us | 2.1 ms | 8.4 ms |
| rpi5-hailo | MEM | 281 us | 740 us | 3.6 ms |
| rpi5-hailo | SHM | 6.0 us | 6.0 us | 6.0 us |
| rpi5-hailo | DMA | 714 us | 1.6 ms | 6.3 ms |
| x86-desktop | MEM | 71 us | 239 us | 1.2 ms |
| x86-desktop | SHM | 4.0 us | 4.0 us | 4.0 us |

#### Map/Unmap Latency

Measures `tensor.map()` round-trip latency. MEM buffers have zero map overhead (direct pointer access). SHM and DMA buffers require kernel calls.

| Platform | Buffer | 720p | 1080p | 4K |
|----------|--------|------|-------|-----|
| imx8mp-frdm | SHM | 13 us | 13 us | 13 us |
| imx8mp-frdm | DMA | 348 us | 766 us | 3.0 ms |
| imx95-frdm | SHM | 12 us | 12 us | 12 us |
| imx95-frdm | DMA | 277 us | 624 us | 2.5 ms |
| rpi5-hailo | SHM | 2.0 us | 2.0 us | 2.0 us |
| rpi5-hailo | DMA | 99 us | 220 us | 868 us |
| x86-desktop | SHM | 1.0 us | 1.0 us | 1.0 us |

### Image Preprocessing: Letterbox Pipeline (Camera → Model Input)

The most critical benchmark: simulates a real camera-to-model preprocessing pipeline with format conversion, resize, and letterbox padding.

#### Packed Formats (1080p → 640×640)

| Platform | Compute | Buffer | YUYV→RGBA | YUYV→RGB | YUYV→RGBi | NV12→RGBA | NV12→RGBi | VYUY→RGBA |
|----------|---------|--------|-----------|----------|-----------|-----------|-----------|-----------|
| imx8mp-frdm | G2D | DMA | 3.0 ms | 4.2 ms | — | 4.1 ms | — | — |
| imx8mp-frdm | GL | DMA | 1.8 ms | — | — | 3.5 ms | — | — |
| imx8mp-frdm | CPU | Heap | 17.7 ms | 17.5 ms | 19.4 ms | 20.6 ms | 18.5 ms | 17.5 ms |
| imx95-frdm | G2D | DMA | 3.9 ms | 4.0 ms | — | 3.8 ms | — | — |
| imx95-frdm | GL | DMA | 1.2 ms | 1.3 ms | 1.3 ms | 1.5 ms | 1.6 ms | — |
| imx95-frdm | CPU | Heap | 14.4 ms | 14.9 ms | 16.4 ms | 19.0 ms | 17.0 ms | 14.4 ms |
| rpi5-hailo | GL | DMA | 3.3 ms | — | — | 1.2 ms | — | — |
| rpi5-hailo | CPU | Heap | 7.7 ms | 7.2 ms | 6.2 ms | 7.4 ms | 5.7 ms | 7.6 ms |
| x86-desktop | GL | PBO | — | — | — | — | — | — |
| x86-desktop | CPU | Heap | 1.4 ms | 1.4 ms | 1.4 ms | 1.1 ms | 1.0 ms | 1.5 ms |

> **Note:** imx8mp-frdm GL "—" for RGB/RGBi: Vivante GL lacks packed RGB support (two-pass disabled). rpi5-hailo GL same limitation on Mesa V3D. x86-desktop GL cannot import YUV textures via PBO.

#### Planar Formats (1080p → 640×640)

Planar formats (8BPS = PLANAR_RGB, 8BPi = PLANAR_RGB_INT8) use separate memory planes for each color channel — required by some ML inference frameworks.

| Platform | Compute | Buffer | YUYV→8BPS | YUYV→8BPi | NV12→8BPS | NV12→8BPi |
|----------|---------|--------|-----------|-----------|-----------|-----------|
| imx8mp-frdm | G2D | DMA | — | — | — | — |
| imx8mp-frdm | GL | DMA | 5.5 ms | 5.5 ms | **BLOCKED** | **BLOCKED** |
| imx8mp-frdm | CPU | Heap | 18.2 ms | 20.1 ms | 17.7 ms | 19.3 ms |
| imx95-frdm | G2D | DMA | — | — | — | — |
| imx95-frdm | GL | DMA | 2.4 ms | 2.8 ms | 3.6 ms | 3.6 ms |
| imx95-frdm | CPU | Heap | 16.4 ms | 17.0 ms | 15.5 ms | 17.0 ms |
| rpi5-hailo | GL | DMA | 16.7 ms | 16.7 ms | 5.0 ms | 5.1 ms |
| rpi5-hailo | CPU | Heap | 8.6 ms | 8.8 ms | 8.0 ms | 8.0 ms |
| x86-desktop | GL | PBO | — | — | — | — |
| x86-desktop | CPU | Heap | 1.5 ms | 1.5 ms | 1.1 ms | 1.2 ms |

> **BLOCKED:** NV12→planar on Vivante GC7000UL causes an unrecoverable GPU hang (kernel Ds state, requires reboot). The HAL explicitly blocks this combination on Vivante GPUs and falls back to CPU in auto mode. See Known Issues §9 and `VSI_GPU_NV12_BUG.md`.
>
> G2D does not support planar output formats.

#### Packed Formats (4K → 640×640)

| Platform | Compute | Buffer | YUYV→RGBA | YUYV→RGB | NV12→RGBA |
|----------|---------|--------|-----------|----------|-----------|
| imx8mp-frdm | G2D | DMA | 4.2 ms | 5.7 ms | 6.8 ms |
| imx8mp-frdm | GL | DMA | 2.4 ms | — | 9.7 ms |
| imx8mp-frdm | CPU | Heap | 59.5 ms | 50.0 ms | 75.7 ms |
| imx95-frdm | G2D | DMA | 13.9 ms | 14.6 ms | 13.3 ms |
| imx95-frdm | GL | DMA | 1.6 ms | 1.7 ms | 4.7 ms |
| imx95-frdm | CPU | Heap | 46.2 ms | 41.2 ms | 64.5 ms |
| rpi5-hailo | GL | DMA | 18.5 ms | — | 5.0 ms |
| rpi5-hailo | CPU | Heap | 23.9 ms | 19.8 ms | 22.2 ms |
| x86-desktop | GL | PBO | — | — | — |
| x86-desktop | CPU | Heap | 6.8 ms | 5.5 ms | 6.3 ms |

#### Planar Formats (4K → 640×640)

| Platform | Compute | Buffer | YUYV→8BPS | YUYV→8BPi | NV12→8BPS | NV12→8BPi |
|----------|---------|--------|-----------|-----------|-----------|-----------|
| imx8mp-frdm | GL | DMA | 8.1 ms | 8.2 ms | **BLOCKED** | **BLOCKED** |
| imx8mp-frdm | CPU | Heap | 50.9 ms | 52.5 ms | — | — |
| imx95-frdm | GL | DMA | 3.7 ms | 3.6 ms | 8.4 ms | 9.1 ms |
| imx95-frdm | CPU | Heap | 46.2 ms | 46.2 ms | — | — |
| rpi5-hailo | GL | DMA | 102.1 ms | 102.2 ms | 25.8 ms | 25.8 ms |
| rpi5-hailo | CPU | Heap | 23.9 ms | 23.9 ms | 22.2 ms | 22.2 ms |
| x86-desktop | CPU | Heap | 5.8 ms | 5.8 ms | 5.1 ms | 5.2 ms |

> **Note:** rpi5-hailo GL YUYV→planar at 4K is ~102ms (very slow) — CPU is 4× faster at 24ms. This appears to be a Mesa V3D bottleneck with MRT at high input resolution. NV12→planar is much faster (26ms) because V3D handles NV12 DMA-buf import more efficiently.

### Format Conversion (Same Size, No Resize)

**1080p → 1080p:**

| Platform | Compute | Buffer | YUYV→RGBA | YUYV→RGB | NV12→RGBA | RGB→RGBA | RGBA→BGRA | RGBA→GREY |
|----------|---------|--------|-----------|----------|-----------|----------|-----------|-----------|
| imx8mp-frdm | G2D | DMA | 6.4 ms | 11.1 ms | 6.3 ms | — | — | — |
| imx8mp-frdm | GL | DMA | 7.1 ms | — | 6.5 ms | — | 7.9 ms | 7.7 ms |
| imx8mp-frdm | CPU | Heap | 6.4 ms | 11.1 ms | 6.5 ms | 13.8 ms | 32.2 ms | 10.1 ms |
| imx95-frdm | G2D | DMA | 3.6 ms | 3.4 ms | 3.5 ms | — | — | — |
| imx95-frdm | GL | DMA | 2.4 ms | 2.5 ms | 3.2 ms | — | — | — |
| imx95-frdm | CPU | Heap | 4.6 ms | 4.3 ms | 4.5 ms | 10.7 ms | 25.5 ms | 8.6 ms |
| rpi5-hailo | GL | DMA | 7.3 ms | — | 5.4 ms | — | 17.2 ms | 6.2 ms |
| rpi5-hailo | CPU | Heap | 6.9 ms | 5.5 ms | 8.1 ms | 6.4 ms | 12.1 ms | 2.5 ms |
| x86-desktop | GL | PBO | — | — | — | 1.2 ms | 1.5 ms | 1.5 ms |
| x86-desktop | CPU | Heap | 574 us | 585 us | 233 us | 305 us | 1.0 ms | 302 us |

### Decoder Post-Processing

All CPU-only (decoder is not GPU-accelerated).

**YOLOv8 Detection (84×8400, 80 classes):**

| Platform | Data Type | Decode + NMS | Decode Only | NMS Only | Dequantize |
|----------|-----------|-------------|-------------|----------|------------|
| imx8mp-frdm | i8 (quant) | 1.0 ms | 996 us | 20 us | 3.8 ms |
| imx8mp-frdm | f32 | 6.5 ms | — | — | — |
| imx95-frdm | i8 (quant) | 833 us | 770 us | 19 us | 2.9 ms |
| imx95-frdm | f32 | 6.3 ms | — | — | — |
| rpi5-hailo | i8 (quant) | 247 us | 251 us | 4.0 us | 2.1 ms |
| rpi5-hailo | f32 | 3.0 ms | — | — | — |
| x86-desktop | i8 (quant) | 70 us | 69 us | 2.0 us | 418 us |
| x86-desktop | f32 | 582 us | — | — | — |

**YOLOv8 Segmentation (mask coefficient → pixel decode):**

| Platform | Data Type | Masks Decode |
|----------|-----------|-------------|
| imx8mp-frdm | i8 (quant) | 4.8 ms |
| imx8mp-frdm | f32 | 6.4 ms |
| imx95-frdm | i8 (quant) | 3.0 ms |
| imx95-frdm | f32 | 6.6 ms |
| rpi5-hailo | i8 (quant) | 1.2 ms |
| rpi5-hailo | f32 | 2.6 ms |
| x86-desktop | i8 (quant) | 391 us |
| x86-desktop | f32 | 850 us |

### Mask Rendering

**640×640 RGBA destination, ~2 detections (YOLOv8n-seg):**

| Platform | Compute | Buffer | draw_masks (pre-decoded) | draw_masks_proto (fused) | decode_masks_atlas | hybrid_materialize_and_draw |
|----------|---------|--------|------------------------|------------------------|--------------------|---------------------------|
| imx8mp-frdm | GL | DMA | 2.6 ms | 5.9 ms | 408 ms | 5.9 ms |
| imx8mp-frdm | CPU | Heap | 5.9 ms | 80.2 ms | 77.8 ms | 9.0 ms |
| imx95-frdm | GL | DMA | 1.9 ms | 5.5 ms | 28.2 ms | 5.3 ms |
| imx95-frdm | CPU | Heap | 7.3 ms | 78.8 ms | 76.1 ms | 10.3 ms |
| rpi5-hailo | GL | DMA | 1.5 ms | 2.4 ms | 7.7 ms | 2.4 ms |
| rpi5-hailo | CPU | Heap | 1.5 ms | 15.0 ms | 14.7 ms | 1.9 ms |
| x86-desktop | GL | PBO | 102 us | 375 us | 847 us | 394 us |
| x86-desktop | CPU | Heap | 102 us | 5.9 ms | 5.9 ms | 786 us |

> **Note:** imx8mp-frdm GL `decode_masks_atlas` degraded from 432ms to 408ms — still anomalously slow compared to other platforms. This is a Vivante GC7000UL driver inefficiency with the atlas shader path.

**Hybrid Path Comparison (CPU materialize + GL overlay vs fused GPU):**

The hybrid path decodes masks on CPU (`materialize_segmentations`) then overlays via GL (`draw_masks`). This is faster than fused GPU `draw_masks_proto` on all tested platforms. The auto-selection in `ImageProcessor::draw_masks_proto()` now prefers the hybrid path when both CPU and OpenGL backends are available.

| Platform | Full GPU (GL draw_masks_proto) | Hybrid (GL) | Speedup | Auto draw_masks_proto |
|----------|-------------------------------|-------------|---------|----------------------|
| imx8mp-frdm | 5.9 ms | 5.9 ms | **1.0×** | 5.9 ms |
| imx95-frdm | 5.5 ms | 5.3 ms | **1.0×** | 5.5 ms |
| rpi5-hailo | 2.4 ms | 2.4 ms | **1.0×** | 2.4 ms |
| x86-desktop | 375 us | 394 us | **0.9×** | 375 us |

> **Note:** The fused GPU path has improved significantly on imx8mp-frdm (from 275ms → 5.9ms) due to shader optimizations. The hybrid and fused paths are now comparable on all platforms.

**Mask Decode Cost (CPU-only, measured in mask_benchmark):**

| Platform | Proto Decode (NMS+coefficients) | Full Materialize (NMS+coefficients+pixels) |
|----------|-------------------------------|-------------------------------------------|
| imx8mp-frdm | 1.5 ms | 4.9 ms |
| imx95-frdm | 1.3 ms | 4.4 ms |
| rpi5-hailo | 419 us | 1.5 ms |
| x86-desktop | 117 us | 423 us |

---

## Known Benchmark Gaps

### Missing Platforms

1. **maivin** — Primary production target (Torizon 7, same SoC as imx8mp-frdm).
   Pending Torizon image with benchmark tooling.

2. **jetson-orin-nano** — NVIDIA Jetson platform. Pending JetPack environment setup and DMA-buf/PBO compatibility testing.

### Missing Buffer Strategy Coverage

3. **No forced Sync (memcpy) benchmarks** — No benchmark of the Sync fallback (`glTexImage2D`/`glReadnPixels` memcpy) to quantify the overhead of non-zero-copy GPU upload/readback.

### Known Performance Issues

4. **imx8mp-frdm GL mask atlas anomalously slow** — `decode_masks_atlas` takes 408ms on Vivante GC7000UL vs 28ms on Mali G310 (imx95-frdm). The Vivante driver appears to have inefficiencies with the atlas shader path.

5. **rpi5-hailo 4K DMA-buf allocation fails** — Mesa V3D driver cannot allocate DMA-buf textures at 3840×2160 for same-size conversion. OpenGL convert benchmarks at 4K produce GL errors on this platform.

6. **x86-desktop OpenGL cannot import YUV textures** — NVIDIA PBO path does not support YUYV/NV12/VYUY source textures. OpenGL letterbox and convert benchmarks show "—" for YUV source formats on this platform.

7. **imx95-frdm GL DMA-buf slower than PBO for letterbox** — v1.2 benchmarks labelled imx95-frdm GL as "DMA" but were actually running on PBO (EGL extension query bug caused DMA-buf roundtrip probe to fail). After fixing the extension query (v1.3), GL now uses true DMA-buf import. DMA-buf letterbox 1080p→640 YUYV→RGBA is 3.4ms vs 1.4ms on PBO — the DMA-buf import/export overhead exceeds PBO zero-copy bind. G2D improved (3.5ms from 3.9ms). Fused mask rendering (`draw_masks_proto`) dramatically improved: 5.2ms from 25.2ms (**4.8× faster**).

8. **BGRA framebuffer CPU byte-swap overhead** — BGRA textures as framebuffer attachments have GPU-dependent swizzle behavior (some implementations don't swizzle fragment shader output). Workaround uses RGBA format internally with CPU-side R↔B byte swaps on upload and readback. RGBA→BGRA conversion on imx95-frdm GL went from 3.4ms (v1.2 PBO, no swap needed) to 26.5ms (v1.3 DMA + CPU swap). CPU backend RGBA→BGRA is 24.5ms for reference.

9. **NV12→planar GPU hang on Vivante GC7000UL** — Rendering from an NV12 source texture (via `EGL_LINUX_DMA_BUF_EXT`) to a planar RGB framebuffer (MRT with 3× color attachments) causes an **unrecoverable GPU hang** on the Vivante GC7000UL (i.MX 8M Plus, galcore 6.4.11). The GPU command processor stalls permanently, the calling process enters kernel uninterruptible sleep (Ds state), cannot be killed even with SIGKILL, and the galcore driver state is corrupted system-wide — all subsequent GPU operations from any process hang until a full board reboot. YUYV→planar and NV12→packed work fine; the bug is specific to NV12 multi-plane texture + MRT output. The HAL explicitly blocks this combination on Vivante GPUs and falls back to CPU in auto mode. See `VSI_GPU_NV12_BUG.md` for the full vendor bug report.

10. **rpi5-hailo GL planar at 4K is slow** — YUYV→8BPS/8BPi at 4K takes ~102ms on Mesa V3D GL, while CPU handles it in ~24ms. NV12→planar at 4K is ~26ms on GL. The bottleneck appears to be in Mesa V3D's MRT path when combined with high-resolution YUYV texture sampling.

11. **imx8mp-frdm GL lacks RGB/RGBi packed support** — Vivante GC7000UL OpenGL does not support packed RGB output without a two-pass approach (RGBA render then strip alpha), which is currently disabled. All RGB/RGBi results on this platform use CPU or G2D.

12. **rpi5-hailo GL lacks RGB/RGBi packed support** — Same limitation as imx8mp-frdm: Mesa V3D two-pass packed RGB is disabled.

### Missing Format Coverage

13. **No NV16 benchmarks** — NV16 (4:2:2 semi-planar) CPU conversion exists but G2D/GL paths and benchmarks are missing.

### Missing Scenarios

14. **No PBO tensor allocation benchmarks** — Tensor allocation benchmarks cover Mem, SHM, and DMA but not PBO (which requires GL context).

15. **No end-to-end pipeline benchmark** — No benchmark covers the full camera → preprocess → decode → mask render cycle in a single measurement.

---

## Changelog

| Version | Date | Changes |
|---------|------|---------|
| 1.5 | 2026-03-18 | Remove stale Known Issue #3 (EDGEFIRST_FORCE_TRANSFER=pbo now implemented); documentation accuracy updates |
| 1.4 | 2026-03-13 | Add planar RGB (8BPS/8BPi) format benchmarks; document NV12→planar GPU hang on Vivante GC7000UL (blocked, CPU fallback); split letterbox tables into packed/planar; update mask rendering (imx8mp fused GPU improved 275ms→5.9ms); add rpi5 GL planar performance notes; refresh all platforms |
| 1.3 | 2026-03-12 | Update imx95-frdm after DMA-buf fix (GL now uses true DMA-buf, was PBO); BGRA CPU byte-swap workaround; fused mask rendering 4.8× faster |
| 1.2 | 2026-03-09 | Add hybrid mask benchmark and comparison table; auto-selection now prefers hybrid path |
| 1.1 | 2026-03-08 | Baseline results for imx8mp-frdm, imx95-frdm, rpi5-hailo, x86-desktop |
| 1.0 | 2026-03-04 | Initial document with strategy, platforms, and gap analysis |
