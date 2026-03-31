# EdgeFirst HAL - Benchmarks

**Version:** 3.0
**Last Updated:** March 30, 2026
**Status:** v0.15.0 release — added jetson-orin-nano platform; refreshed all benchmarks across 5 platforms; added materialize_masks API and hybrid path comparison; per-texture EGL binding optimization

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

---

## Running Benchmarks

See [README.md § Benchmarking](README.md#benchmarking) for full instructions on running benchmarks locally, cross-compiling for aarch64, and deploying to target platforms.

### Benchmark Binaries

| Binary | Crate | What It Measures |
|--------|-------|-----------------|
| `tensor_benchmark` | `edgefirst-tensor` | Tensor allocation and map/unmap latency across buffer types (Heap, SHM, DMA) |
| `image_benchmark` | `edgefirst-image` | JPEG loading, format convert, resize operations across buffer backends |
| `pipeline_benchmark` | `edgefirst-image` | Letterbox pipeline and format conversion (camera→model input) |
| `mask_benchmark` | `edgefirst-image` | Mask rendering: draw_decoded_masks, draw_proto_masks, hybrid path |
| `opencv_benchmark` | `edgefirst-image` | OpenCV baseline comparison for same operations |
| `decoder_benchmark` | `edgefirst-decoder` | YOLO detection/segmentation post-processing, NMS, dequantization |

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

---

## Benchmark Results

**Data collected:** March 30, 2026 (v0.15.0, per-texture EGL binding optimization)

### Buffer Infrastructure

#### Allocation Latency

Measures `Tensor::new()` latency for each buffer type and resolution.

| Platform | Buffer | 720p (3.5 MB) | 1080p (7.9 MB) | 4K (31.6 MB) |
|----------|--------|---------------|-----------------|---------------|
| imx8mp-frdm | MEM | 310 us | 698 us | 2.8 ms |
| imx8mp-frdm | SHM | 26 us | 26 us | 26 us |
| imx8mp-frdm | DMA | 38.2 ms | 29.9 ms | 10.0 ms |
| imx95-frdm | MEM | 266 us | 596 us | 2.4 ms |
| imx95-frdm | SHM | 31 us | 31 us | 31 us |
| imx95-frdm | DMA | 983 us | 2.1 ms | 8.4 ms |
| rpi5-hailo | MEM | 249 us | 736 us | 3.5 ms |
| rpi5-hailo | SHM | 6.0 us | 6.0 us | 6.0 us |
| rpi5-hailo | DMA | 713 us | 1.6 ms | 6.2 ms |
| jetson-orin-nano | MEM | 171 us | 386 us | 1.5 ms |
| jetson-orin-nano | SHM | 14 us | 14 us | 14 us |
| x86-desktop | MEM | 105 us | 268 us | 807 us |
| x86-desktop | SHM | 2.0 us | 2.0 us | 2.0 us |

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

### Image Preprocessing: Letterbox Pipeline (Camera → Model Input)

**1080p → 640×640:**

| Platform | Compute | Buffer | YUYV→RGBA | YUYV→RGB | YUYV→8BPi | NV12→RGBA | VYUY→RGBA |
|----------|---------|--------|-----------|----------|-----------|-----------|-----------|
| imx8mp-frdm | G2D | DMA | 2.7 ms | 4.0 ms | — | 4.1 ms | — |
| imx8mp-frdm | GL | DMA | 1.7 ms | 11.8 ms | — | 3.5 ms | — |
| imx8mp-frdm | CPU | Heap | 17.4 ms | 17.6 ms | — | 33.9 ms | 17.5 ms |
| imx95-frdm | G2D | DMA | 3.9 ms | 4.6 ms | — | 3.7 ms | — |
| imx95-frdm | GL | DMA | 1.2 ms | 3.3 ms | — | 1.5 ms | — |
| imx95-frdm | CPU | Heap | 14.5 ms | 14.9 ms | — | 16.6 ms | 14.5 ms |
| rpi5-hailo | GL | DMA | 3.3 ms | 4.1 ms | — | 1.2 ms | — |
| rpi5-hailo | CPU | Heap | 7.6 ms | 7.2 ms | — | 8.1 ms | 7.6 ms |
| jetson-orin-nano | GL | DMA | — | — | — | — | — |
| jetson-orin-nano | CPU | Heap | 6.1 ms | 5.9 ms | — | 5.3 ms | 6.2 ms |
| x86-desktop | CPU | Heap | 3.0 ms | 1.5 ms | — | 1.8 ms | 5.4 ms |

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

### Format Conversion (Same Size, No Resize)

**1080p → 1080p:**

| Platform | Compute | Buffer | YUYV→RGBA | YUYV→RGB | NV12→RGBA | RGB→RGBA | RGBA→BGRA | RGBA→GREY |
|----------|---------|--------|-----------|----------|-----------|----------|-----------|-----------|
| imx8mp-frdm | G2D | DMA | 6.0 ms | 10.5 ms | 6.3 ms | — | — | — |
| imx8mp-frdm | GL | DMA | 7.2 ms | 49.9 ms | 6.1 ms | — | 7.7 ms | 7.8 ms |
| imx8mp-frdm | CPU | Heap | 13.5 ms | 11.8 ms | 25.7 ms | 13.7 ms | 30.2 ms | 10.0 ms |
| imx95-frdm | G2D | DMA | 4.6 ms | 4.3 ms | 4.5 ms | — | — | — |
| imx95-frdm | GL | DMA | 3.1 ms | 11.8 ms | 3.2 ms | — | 3.1 ms | 2.9 ms |
| imx95-frdm | CPU | Heap | 12.4 ms | 11.0 ms | 16.1 ms | 11.1 ms | 24.8 ms | 9.0 ms |
| rpi5-hailo | GL | DMA | 7.2 ms | 10.4 ms | 5.4 ms | — | 8.4 ms | 6.2 ms |
| rpi5-hailo | CPU | Heap | 6.8 ms | 5.4 ms | 8.0 ms | 6.6 ms | 12.2 ms | 2.5 ms |
| jetson-orin-nano | GL | DMA | — | — | — | 1.5 ms | 4.2 ms | 1.5 ms |
| jetson-orin-nano | CPU | Heap | 3.0 ms | 2.8 ms | 2.1 ms | 789 us | 3.2 ms | 1.4 ms |
| x86-desktop | CPU | Heap | 516 us | 559 us | 256 us | 261 us | 758 us | 219 us |

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

### Mask Rendering

**640×640 RGBA destination, ~2 detections (YOLOv8n-seg):**

| Platform | Compute | Buffer | draw_decoded_masks (pre-decoded) | draw_proto_masks (fused) | hybrid_materialize_and_draw |
|----------|---------|--------|-------------------------------|------------------------|---------------------------|
| imx8mp-frdm | GL | DMA | 2.4 ms | 276 ms | 19.5 ms |
| imx8mp-frdm | CPU | Heap | 5.3 ms | 77.8 ms | 8.3 ms |
| imx95-frdm | GL | DMA | 1.9 ms | 26.0 ms | 5.6 ms |
| imx95-frdm | CPU | Heap | 5.3 ms | 76.2 ms | 8.4 ms |
| rpi5-hailo | GL | DMA | 1.5 ms | 8.0 ms | 5.6 ms |
| rpi5-hailo | CPU | Heap | 885 us | 14.5 ms | 1.7 ms |
| jetson-orin-nano | GL | DMA | 556 us | 3.0 ms | 1.7 ms |
| jetson-orin-nano | CPU | Heap | 873 us | 22.1 ms | 1.9 ms |
| x86-desktop | CPU | Heap | 648 us | 5.1 ms | 635 us |

**Hybrid Path Comparison (CPU materialize + GL overlay vs fused GPU):**

The hybrid path decodes masks on CPU (`materialize_segmentations`) then overlays via GL (`draw_decoded_masks`). This is faster than fused GPU `draw_proto_masks` on all tested platforms. The auto-selection in `ImageProcessor::draw_proto_masks()` prefers the hybrid path when both CPU and OpenGL backends are available.

**New in v0.15.0:** The `materialize_masks()` API exposes the CPU materialization step as a first-class operation, enabling a three-stage pipeline (`decode_proto` → `materialize_masks` → `draw_decoded_masks`) where users can inspect, export, or fork the intermediate masks for analytics before rendering. Mask values are continuous sigmoid confidence (u8 0-255), not binary thresholded.

| Platform | Full GPU (GL draw_proto_masks) | Hybrid (GL) | Speedup | Auto draw_proto_masks |
|----------|-------------------------------|-------------|---------|----------------------|
| imx8mp-frdm | 276 ms | 19.5 ms | **14.2×** | 4.2 ms |
| imx95-frdm | 26.0 ms | 5.6 ms | **4.6×** | 4.2 ms |
| rpi5-hailo | 8.0 ms | 5.6 ms | **1.4×** | 2.0 ms |
| jetson-orin-nano | 3.0 ms | 1.7 ms | **1.8×** | 1.1 ms |

**Mask Decode Cost (CPU-only, measured in mask_benchmark):**

| Platform | Proto Decode (NMS+coefficients) | Full Materialize (NMS+coefficients+pixels) |
|----------|-------------------------------|-------------------------------------------|
| imx8mp-frdm | 1.5 ms | 4.9 ms |
| imx95-frdm | 1.2 ms | 4.3 ms |
| rpi5-hailo | 376 us | 1.4 ms |
| jetson-orin-nano | 440 us | 1.6 ms |
| x86-desktop | 381 us | 903 us |

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
cargo-zigbuild zigbuild --target aarch64-unknown-linux-gnu --release -p edgefirst-capi

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

8. **NV12→planar GPU hang on Vivante GC7000UL** — Rendering from an NV12 source texture (via `EGL_LINUX_DMA_BUF_EXT`) to a planar RGB framebuffer (MRT with 3× color attachments) causes an **unrecoverable GPU hang** on the Vivante GC7000UL (i.MX 8M Plus, galcore 6.4.11). The GPU command processor stalls permanently, the calling process enters kernel uninterruptible sleep (Ds state), cannot be killed even with SIGKILL, and the galcore driver state is corrupted system-wide — all subsequent GPU operations from any process hang until a full board reboot. YUYV→planar and NV12→packed work fine; the bug is specific to NV12 multi-plane texture + MRT output. The HAL explicitly blocks this combination on Vivante GPUs and falls back to CPU in auto mode. See `VSI_GPU_NV12_BUG.md` for the full vendor bug report.

9. **rpi5-hailo GL planar at 4K is slow** — YUYV→8BPS/8BPS_i8 at 4K takes ~102ms on Mesa V3D GL, while CPU handles it in ~24ms. NV12→planar at 4K is ~26ms on GL. The bottleneck appears to be in Mesa V3D's MRT path when combined with high-resolution YUYV texture sampling.

10. **imx8mp-frdm GL packed RGB uses two-pass approach** — Vivante GC7000UL OpenGL does not support packed RGB output natively; the two-pass packed RGB packing shader renders to an RGBA intermediate then packs to RGB using a dedicated shader. This two-pass approach is now enabled but is 3-4× slower than G2D's hardware blitter for packed RGB output on Vivante (see footnote ¹ in 720p tables).

11. **rpi5-hailo GL packed RGB uses two-pass approach** — Same as imx8mp-frdm: Mesa V3D uses the two-pass packed RGB packing shader (RGBA intermediate then dedicated RGB packing shader). Now enabled but may be slower than CPU for some conversions on VideoCore.

### Missing Format Coverage

13. **No NV16 benchmarks** — NV16 (4:2:2 semi-planar) CPU conversion exists but G2D/GL paths and benchmarks are missing.

### Missing Scenarios

14. **No PBO tensor allocation benchmarks** — Tensor allocation benchmarks cover Mem, SHM, and DMA but not PBO (which requires GL context).

15. **No end-to-end pipeline benchmark** — No benchmark covers the full camera → preprocess → decode → mask render cycle in a single measurement.

---

## Changelog

| Version | Date | Changes |
|---------|------|---------|
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
