# EdgeFirst HAL - Benchmarks

**Version:** 1.2
**Last Updated:** March 9, 2026
**Status:** Baseline results collected for imx8mp-frdm, imx95-frdm, rpi5-hailo, x86-desktop

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
**Destination formats:** RGBA, BGRA, RGB, GREY, PLANAR_RGB, and INT8 variants

### Measurement Methodology

All benchmarks use the `edgefirst-bench` custom harness which:
- Runs in-process (no fork) to avoid GPU driver crashes
- Executes warmup iterations (unmeasured) followed by measured iterations
- Reports: median, mean, min, max, p95, p99
- Reports throughput in MiB/s where applicable

**Standard parameters:** 10 warmup iterations, 200 measured iterations (adjustable per benchmark).

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
| **Notes** | NXP evaluation board; same SoC as maivin, latest NXP BSP |

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
| **Notes** | Mesa V3D driver; DMA-buf roundtrip status TBD |

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

**1080p → 640×640:**

| Platform | Compute | Buffer | YUYV→RGBA | YUYV→RGB | YUYV→8BPi | NV12→RGBA | VYUY→RGBA |
|----------|---------|--------|-----------|----------|-----------|-----------|-----------|
| imx8mp-frdm | G2D | DMA | 3.3 ms | 4.2 ms | — | 4.1 ms | — |
| imx8mp-frdm | GL | DMA | 1.8 ms | — | 5.6 ms | 3.5 ms | — |
| imx8mp-frdm | CPU | Heap | 17.4 ms | 17.6 ms | 19.9 ms | 21.5 ms | 17.5 ms |
| imx95-frdm | G2D | DMA | 3.9 ms | 4.6 ms | — | 3.8 ms | — |
| imx95-frdm | GL | DMA | 1.4 ms | 1.4 ms | 2.8 ms | 1.7 ms | — |
| imx95-frdm | CPU | Heap | 14.6 ms | 14.9 ms | 17.0 ms | 19.1 ms | 14.6 ms |
| rpi5-hailo | GL | DMA | 3.3 ms | — | 16.9 ms | 1.2 ms | — |
| rpi5-hailo | CPU | Heap | 7.6 ms | 7.2 ms | 7.4 ms | 7.4 ms | 7.6 ms |
| x86-desktop | GL | PBO | — | — | — | — | — |
| x86-desktop | CPU | Heap | 1.7 ms | 1.5 ms | 1.9 ms | 1.4 ms | 1.8 ms |

> **Note:** x86-desktop GL shows "—" because NVIDIA PBO cannot import YUYV/NV12 textures directly. On x86, OpenGL is only effective for RGBA↔RGBA operations.

**4K → 640×640:**

| Platform | Compute | Buffer | YUYV→RGBA | YUYV→RGB | NV12→RGBA |
|----------|---------|--------|-----------|----------|-----------|
| imx8mp-frdm | G2D | DMA | 4.2 ms | 5.7 ms | 6.8 ms |
| imx8mp-frdm | GL | DMA | 2.4 ms | — | 9.3 ms |
| imx8mp-frdm | CPU | Heap | 58.8 ms | 49.5 ms | 75.7 ms |
| imx95-frdm | G2D | DMA | 13.9 ms | 16.6 ms | 13.2 ms |
| imx95-frdm | GL | DMA | 1.8 ms | 1.7 ms | 5.0 ms |
| imx95-frdm | CPU | Heap | 46.2 ms | 41.7 ms | 66.5 ms |
| rpi5-hailo | GL | DMA | 18.6 ms | — | — |
| rpi5-hailo | CPU | Heap | 23.9 ms | 19.8 ms | 22.2 ms |
| x86-desktop | GL | PBO | — | — | — |
| x86-desktop | CPU | Heap | 7.7 ms | 6.4 ms | 9.5 ms |

> **Note:** rpi5-hailo GL shows "—" for 4K NV12→RGBA because Mesa V3D cannot allocate DMA-buf textures at 4K resolution. Only 1080p and below are supported.

### Format Conversion (Same Size, No Resize)

**1080p → 1080p:**

| Platform | Compute | Buffer | YUYV→RGBA | YUYV→RGB | NV12→RGBA | RGB→RGBA | RGBA→BGRA | RGBA→GREY |
|----------|---------|--------|-----------|----------|-----------|----------|-----------|-----------|
| imx8mp-frdm | G2D | DMA | 6.4 ms | 11.1 ms | 6.3 ms | — | — | — |
| imx8mp-frdm | GL | DMA | 7.3 ms | — | 8.0 ms | — | 7.9 ms | 7.7 ms |
| imx8mp-frdm | CPU | Heap | 13.7 ms | 11.8 ms | 13.0 ms | 13.8 ms | 30.5 ms | 10.1 ms |
| imx95-frdm | G2D | DMA | 4.7 ms | 4.3 ms | 4.5 ms | — | — | — |
| imx95-frdm | GL | DMA | 3.3 ms | 3.1 ms | 3.2 ms | — | 3.4 ms | 3.1 ms |
| imx95-frdm | CPU | Heap | 12.3 ms | 10.8 ms | 15.8 ms | 10.9 ms | 24.2 ms | 8.8 ms |
| rpi5-hailo | GL | DMA | 16.4 ms | — | 5.5 ms | — | 17.2 ms | 6.2 ms |
| rpi5-hailo | CPU | Heap | 6.8 ms | 5.4 ms | 8.0 ms | 6.5 ms | 12.1 ms | 2.5 ms |
| x86-desktop | GL | PBO | — | — | — | 1.2 ms | 1.5 ms | 1.5 ms |
| x86-desktop | CPU | Heap | 616 us | 619 us | 255 us | 323 us | 1.1 ms | 295 us |

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
| imx8mp-frdm | GL | DMA | 2.6 ms | 275 ms | 432 ms | 5.9 ms |
| imx8mp-frdm | CPU | Heap | 5.9 ms | 80.2 ms | 77.8 ms | 9.0 ms |
| imx95-frdm | GL | DMA | 1.9 ms | 25.2 ms | 28.0 ms | 5.7 ms |
| imx95-frdm | CPU | Heap | 6.0 ms | 78.1 ms | 75.7 ms | 9.0 ms |
| rpi5-hailo | GL | DMA | 1.5 ms | 7.6 ms | 7.9 ms | 2.4 ms |
| rpi5-hailo | CPU | Heap | 1.1 ms | 15.0 ms | 14.7 ms | 1.9 ms |
| x86-desktop | GL | PBO | 107 us | 802 us | 923 us | 394 us |
| x86-desktop | CPU | Heap | 519 us | 5.9 ms | 5.9 ms | 786 us |

> **Note:** imx8mp-frdm GL mask performance is anomalously slow (275ms for `draw_masks_proto` vs 25.2ms on imx95-frdm). This appears to be a Vivante GC7000UL driver inefficiency with the mask shader — investigation is tracked.

**Hybrid Path Comparison (CPU materialize + GL overlay vs fused GPU):**

The hybrid path decodes masks on CPU (`materialize_segmentations`) then overlays via GL (`draw_masks`). This is faster than fused GPU `draw_masks_proto` on all tested platforms. The auto-selection in `ImageProcessor::draw_masks_proto()` now prefers the hybrid path when both CPU and OpenGL backends are available.

| Platform | Full GPU (GL draw_masks_proto) | Hybrid (GL) | Speedup | Auto draw_masks_proto |
|----------|-------------------------------|-------------|---------|----------------------|
| imx8mp-frdm | 275 ms | 5.9 ms | **47×** | 6.0 ms |
| imx95-frdm | 25.2 ms | 5.7 ms | **4.4×** | 5.6 ms |
| rpi5-hailo | 7.6 ms | 2.4 ms | **3.2×** | 2.4 ms |
| x86-desktop | 802 us | 394 us | **2.0×** | 386 us |

> **Note:** "Auto draw_masks_proto" confirms the auto path now selects the hybrid path — timings match `hybrid_materialize_and_draw`.

**Mask Decode Cost (CPU-only, measured in mask_benchmark):**

| Platform | Proto Decode (NMS+coefficients) | Full Materialize (NMS+coefficients+pixels) |
|----------|-------------------------------|-------------------------------------------|
| imx8mp-frdm | 1.6 ms | 4.9 ms |
| imx95-frdm | 1.3 ms | 4.3 ms |
| rpi5-hailo | 527 us | 1.5 ms |
| x86-desktop | 122 us | 429 us |

---

## Known Benchmark Gaps

### Missing Platforms

1. **maivin** — Primary production target (Torizon 7, same SoC as imx8mp-frdm).
   Pending Torizon image with benchmark tooling.

2. **jetson-orin-nano** — NVIDIA Jetson platform. Pending JetPack environment setup and DMA-buf/PBO compatibility testing.

### Missing Buffer Strategy Coverage

3. **No mechanism to force PBO on DMA-capable platforms** — Benchmarks on i.MX platforms only test the DMA-buf buffer path for OpenGL. There is no way to force the PBO buffer strategy to compare PBO vs DMA-buf transfer performance on the same hardware. *Requires: `EDGEFIRST_FORCE_TRANSFER=pbo` env var support.*

4. **No forced Sync (memcpy) benchmarks** — No benchmark of the Sync fallback (`glTexImage2D`/`glReadnPixels` memcpy) to quantify the overhead of non-zero-copy GPU upload/readback.

### Known Performance Issues

5. **imx8mp-frdm GL mask rendering anomalously slow** — draw_masks_proto takes 275ms on Vivante GC7000UL vs 25.2ms on Mali G310 (imx95-frdm). The Vivante driver appears to have inefficiencies with the mask shader.

6. **rpi5-hailo 4K DMA-buf allocation fails** — Mesa V3D driver cannot allocate DMA-buf textures at 3840×2160. OpenGL benchmarks at 4K are skipped on this platform.

7. **x86-desktop OpenGL cannot import YUV textures** — NVIDIA PBO path does not support YUYV/NV12/VYUY source textures. OpenGL letterbox and convert benchmarks show "—" for YUV source formats on this platform.

### Missing Format Coverage

8. **No NV16 benchmarks** — Only NV12 is tested for semi-planar formats. NV16 (4:2:2 semi-planar) has different memory layout characteristics.

9. **No planar RGBA benchmarks** — Only planar RGB is tested.

### Missing Scenarios

10. **No PBO tensor allocation benchmarks** — Tensor allocation benchmarks cover Mem, SHM, and DMA but not PBO (which requires GL context).

11. **No end-to-end pipeline benchmark** — No benchmark covers the full camera → preprocess → decode → mask render cycle in a single measurement.

---

## Changelog

| Version | Date | Changes |
|---------|------|---------|
| 1.2 | 2026-03-09 | Add hybrid mask benchmark and comparison table; auto-selection now prefers hybrid path |
| 1.1 | 2026-03-08 | Baseline results for imx8mp-frdm, imx95-frdm, rpi5-hailo, x86-desktop |
| 1.0 | 2026-03-04 | Initial document with strategy, platforms, and gap analysis |
