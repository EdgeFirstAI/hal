# EdgeFirst HAL - Benchmarks

**Version:** 1.0
**Last Updated:** March 4, 2026
**Status:** Initial — result tables pending first collection run

---

## Overview

This document tracks EdgeFirst HAL performance across target platforms.
It serves as a regression baseline: results are updated with each release to
detect performance improvements or regressions introduced by code changes.

The benchmarking strategy tests **all compute backends** (CPU, OpenGL, G2D)
with **all applicable buffer strategies** (DMA-buf, PBO, Sync) on every
platform, including forcing non-default buffer paths on platforms that would
normally prefer a different strategy. This ensures the full fallback chain
is exercised and performance characteristics are understood for every
deployment scenario.

## Benchmarking Strategy

### Compute Backends

Each benchmark category runs across all available **compute backends**:

| Compute Backend | Description | Platforms |
|----------------|-------------|-----------|
| **CPU** | Pure software using vectorized operations + Rayon parallelism | All |
| **OpenGL** | GPU-accelerated via OpenGL ES shader pipeline | Linux with EGL |
| **G2D** | NXP 2D hardware blitter (Vivante) | NXP i.MX Familly |

Future backends may include OpenCL, Vulkan, and other vendor-specific 2D
accelerators.

### Buffer Strategies

Orthogonally, each compute backend operates on buffers using different
memory and transfer strategies:

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

Typically we benchmark DMA-buf and PBO for GPU backends. The Sync (upload/
readpixels) path is only benchmarked when PBO is not supported on a platform.

### Buffer Infrastructure Benchmarks

In addition to compute benchmarks, we separately measure:
- **Allocation latency** — `Tensor::new()` for each buffer type (DMA, SHM, Mem, PBO)
- **Map/unmap latency** — `tensor.map()` for each buffer type
- **Memcpy throughput** — read/write bandwidth for mapped buffers

These infrastructure benchmarks isolate the memory subsystem overhead from
the compute backend performance.

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

### Quick Start

```bash
# Run all Rust benchmarks
make bench

# Run specific benchmark suites
cargo bench -p edgefirst-image --bench pipeline_benchmark -- --bench
cargo bench -p edgefirst-image --bench image_benchmark
cargo bench -p edgefirst-image --bench mask_benchmark -- --bench
cargo bench -p edgefirst-decoder --bench decoder_benchmark
cargo bench -p edgefirst-tensor --bench tensor_benchmark

# Run Python benchmarks
source venv/bin/activate
python tests/bench_decode_render.py --iterations 200 --json results.json
```

### Cross-Compile and Run on Target

```bash
# Build benchmarks for aarch64
cargo-zigbuild test --target aarch64-unknown-linux-gnu --release --no-run \
    --workspace --exclude edgefirst_hal

# Copy to target
scp target/aarch64-unknown-linux-gnu/release/deps/pipeline_benchmark-* target:/tmp/

# Run on target
ssh target '/tmp/pipeline_benchmark-* --bench'
```

### Controlling Compute Backends

Use environment variables to isolate individual compute backends:

```bash
# CPU-only (disable all hardware acceleration)
EDGEFIRST_DISABLE_GL=1 EDGEFIRST_DISABLE_G2D=1 cargo bench ...

# OpenGL-only (disable G2D and CPU)
EDGEFIRST_DISABLE_G2D=1 EDGEFIRST_DISABLE_CPU=1 cargo bench ...

# G2D-only (disable GL and CPU, NXP i.MX platforms only)
EDGEFIRST_DISABLE_GL=1 EDGEFIRST_DISABLE_CPU=1 cargo bench ...
```

### Controlling Buffer Strategies

```bash
# Force heap memory for tensor allocation (no DMA-buf)
# On platforms with OpenGL, this triggers PBO buffer strategy
EDGEFIRST_TENSOR_FORCE_MEM=1 cargo bench ...

# Force PBO buffer strategy for OpenGL (even when DMA-buf is available)
# TODO: requires new EDGEFIRST_FORCE_TRANSFER_BACKEND=pbo env var (Phase 0)
```

### Output Format

Benchmark results are printed as:
```
  benchmark/name                                     median=   1.23ms  mean=   1.45ms  min=   1.10ms  max=   2.30ms  p95=   1.89ms  (n=200)
```

Python benchmarks with `--json` produce machine-readable output for automated collection.

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

> **Status:** Tables below are templates. Results will be populated during the
> first comprehensive benchmark collection run (Phase 0 of the refactoring plan).

### Buffer Infrastructure

#### Allocation Latency

Measures `Tensor::new()` latency for each buffer type and resolution.

| Platform | Buffer | 480p (1.3 MB) | 720p (3.5 MB) | 1080p (7.9 MB) | 4K (31.6 MB) |
|----------|--------|---------------|---------------|-----------------|---------------|
| maivin | DMA | — | — | — | — |
| maivin | SHM | — | — | — | — |
| maivin | Mem | — | — | — | — |
| maivin | PBO | — | — | — | — |
| imx8mp-frdm | DMA | — | — | — | — |
| imx8mp-frdm | SHM | — | — | — | — |
| imx8mp-frdm | Mem | — | — | — | — |
| imx8mp-frdm | PBO | — | — | — | — |
| imx95-frdm | DMA | — | — | — | — |
| imx95-frdm | SHM | — | — | — | — |
| imx95-frdm | Mem | — | — | — | — |
| imx95-frdm | PBO | — | — | — | — |
| jetson-orin-nano | DMA | — | — | — | — |
| jetson-orin-nano | SHM | — | — | — | — |
| jetson-orin-nano | Mem | — | — | — | — |
| jetson-orin-nano | PBO | — | — | — | — |
| rpi5-hailo | DMA | — | — | — | — |
| rpi5-hailo | SHM | — | — | — | — |
| rpi5-hailo | Mem | — | — | — | — |
| rpi5-hailo | PBO | — | — | — | — |
| x86-desktop | DMA | — | — | — | — |
| x86-desktop | SHM | — | — | — | — |
| x86-desktop | Mem | — | — | — | — |
| x86-desktop | PBO | — | — | — | — |

#### Map/Unmap Latency

Measures `tensor.map()` round-trip latency.

| Platform | Buffer | 480p | 720p | 1080p | 4K |
|----------|--------|------|------|-------|-----|
| maivin | DMA | — | — | — | — |
| maivin | SHM | — | — | — | — |
| maivin | Mem | — | — | — | — |
| maivin | PBO | — | — | — | — |
| imx8mp-frdm | DMA | — | — | — | — |
| imx95-frdm | DMA | — | — | — | — |
| x86-desktop | PBO | — | — | — | — |

### Image Preprocessing: Letterbox Pipeline (Camera → Model Input)

The most critical benchmark: simulates a real camera-to-model preprocessing
pipeline with format conversion, resize, and letterbox padding.

**1080p → 640×640:**

| Platform | Compute | Buffer | YUYV→RGBA | YUYV→RGB | YUYV→PLANAR_RGB | NV12→RGBA | NV12→RGB | VYUY→RGBA |
|----------|---------|--------|-----------|----------|-----------------|-----------|----------|-----------|
| maivin | G2D | DMA | — | — | — | — | — | — |
| maivin | GL | DMA | — | — | — | — | — | — |
| maivin | GL | PBO | — | — | — | — | — | — |
| maivin | CPU | Heap | — | — | — | — | — | — |
| imx8mp-frdm | G2D | DMA | — | — | — | — | — | — |
| imx8mp-frdm | GL | DMA | — | — | — | — | — | — |
| imx8mp-frdm | CPU | Heap | — | — | — | — | — | — |
| imx95-frdm | G2D | DMA | — | — | — | — | — | — |
| imx95-frdm | GL | DMA | — | — | — | — | — | — |
| imx95-frdm | CPU | Heap | — | — | — | — | — | — |
| jetson-orin-nano | GL | PBO | — | — | — | — | — | — |
| jetson-orin-nano | CPU | Heap | — | — | — | — | — | — |
| rpi5-hailo | GL | DMA | — | — | — | — | — | — |
| rpi5-hailo | CPU | Heap | — | — | — | — | — | — |
| x86-desktop | GL | PBO | — | — | — | — | — | — |
| x86-desktop | CPU | Heap | — | — | — | — | — | — |

**4K → 640×640:**

| Platform | Compute | Buffer | YUYV→RGBA | YUYV→RGB | NV12→RGBA | NV12→RGB |
|----------|---------|--------|-----------|----------|-----------|----------|
| maivin | G2D | DMA | — | — | — | — |
| maivin | GL | DMA | — | — | — | — |
| maivin | CPU | Heap | — | — | — | — |
| imx8mp-frdm | G2D | DMA | — | — | — | — |
| imx8mp-frdm | GL | DMA | — | — | — | — |
| imx8mp-frdm | CPU | Heap | — | — | — | — |
| imx95-frdm | G2D | DMA | — | — | — | — |
| imx95-frdm | GL | DMA | — | — | — | — |
| imx95-frdm | CPU | Heap | — | — | — | — |
| jetson-orin-nano | GL | PBO | — | — | — | — |
| jetson-orin-nano | CPU | Heap | — | — | — | — |
| rpi5-hailo | GL | DMA | — | — | — | — |
| rpi5-hailo | CPU | Heap | — | — | — | — |
| x86-desktop | GL | PBO | — | — | — | — |
| x86-desktop | CPU | Heap | — | — | — | — |

### Format Conversion (Same Size, No Resize)

**1080p → 1080p:**

| Platform | Compute | Buffer | YUYV→RGBA | YUYV→RGB | NV12→RGBA | RGB→RGBA | RGBA→BGRA | RGBA→GREY |
|----------|---------|--------|-----------|----------|-----------|----------|-----------|-----------|
| maivin | G2D | DMA | — | — | — | — | — | — |
| maivin | GL | DMA | — | — | — | — | — | — |
| maivin | CPU | Heap | — | — | — | — | — | — |
| imx95-frdm | G2D | DMA | — | — | — | — | — | — |
| imx95-frdm | GL | DMA | — | — | — | — | — | — |
| imx95-frdm | CPU | Heap | — | — | — | — | — | — |
| x86-desktop | GL | PBO | — | — | — | — | — | — |
| x86-desktop | CPU | Heap | — | — | — | — | — | — |

### Decoder Post-Processing

All CPU-only (decoder is not GPU-accelerated).

**YOLOv8 Detection (84×8400, 80 classes):**

| Platform | Data Type | Decode + NMS | Decode Only | NMS Only | Dequantize |
|----------|-----------|-------------|-------------|----------|------------|
| maivin | i8 (quant) | — | — | — | — |
| maivin | f32 | — | — | — | — |
| imx8mp-frdm | i8 (quant) | — | — | — | — |
| imx95-frdm | i8 (quant) | — | — | — | — |
| jetson-orin-nano | i8 (quant) | — | — | — | — |
| rpi5-hailo | i8 (quant) | — | — | — | — |
| x86-desktop | i8 (quant) | — | — | — | — |
| x86-desktop | f32 | — | — | — | — |

**YOLOv8 Segmentation (detection + mask coefficients):**

| Platform | Data Type | Full Segdet | Masks Only |
|----------|-----------|------------|------------|
| maivin | i8 (quant) | — | — |
| maivin | f32 | — | — |
| imx8mp-frdm | i8 (quant) | — | — |
| imx95-frdm | i8 (quant) | — | — |
| x86-desktop | i8 (quant) | — | — |

### Mask Rendering

**640×640 RGBA destination, ~5 detections (YOLOv8n-seg):**

| Platform | Compute | Buffer | draw_masks (pre-decoded) | draw_masks_proto (fused) | decode_masks_atlas |
|----------|---------|--------|------------------------|------------------------|--------------------|
| maivin | GL | DMA | — | — | — |
| maivin | CPU | Heap | — | — | — |
| imx8mp-frdm | GL | DMA | — | — | — |
| imx8mp-frdm | CPU | Heap | — | — | — |
| imx95-frdm | GL | DMA | — | — | — |
| imx95-frdm | CPU | Heap | — | — | — |
| jetson-orin-nano | GL | PBO | — | — | — |
| jetson-orin-nano | CPU | Heap | — | — | — |
| rpi5-hailo | GL | DMA | — | — | — |
| rpi5-hailo | CPU | Heap | — | — | — |
| x86-desktop | GL | PBO | — | — | — |
| x86-desktop | CPU | Heap | — | — | — |

---

## Known Benchmark Gaps

The following gaps have been identified and are tracked for resolution in
the Phase 0 benchmark migration:

### Missing Buffer Strategy Coverage

1. **No mechanism to force PBO on DMA-capable platforms** — Benchmarks on
   i.MX platforms only test the DMA-buf buffer path for OpenGL. There is no
   way to force the PBO buffer strategy to compare PBO vs DMA-buf transfer
   performance on the same hardware.
   *Requires: new env var or API to override buffer strategy selection.*

2. **No forced Sync (memcpy) benchmarks** — When OpenGL is available with
   either DMA-buf or PBO, there's no benchmark of the Sync fallback
   (`glTexImage2D`/`glReadnPixels` memcpy) to quantify the overhead of
   non-zero-copy GPU upload/readback.

### Missing Format Coverage

3. **No BGRA destination benchmarks** — BGRA was recently added for
   Cairo/Wayland compositing but has no benchmark coverage.

4. **No NV16 benchmarks** — Only NV12 is tested for semi-planar formats.
   NV16 (4:2:2 semi-planar) has different memory layout characteristics.

5. **No GREY conversion benchmarks** — Grayscale conversion is only tested
   in one small upscale test. No pipeline or hires benchmarks exist.

6. **No VYUY benchmarks for OpenGL/G2D** — VYUY is only benchmarked on CPU.

7. **No planar RGBA benchmarks** — Only planar RGB is tested.

### Missing Scenarios

8. **No PBO tensor allocation benchmarks** — Tensor allocation benchmarks
   cover Mem, SHM, and DMA but not PBO (which requires GL context).

9. **No end-to-end pipeline benchmark** — No benchmark covers the full
   camera → preprocess → decode → mask render cycle in a single measurement.

10. **No rotation benchmarks for CPU at hires** — Rotation is only benchmarked
    at standard sizes, not at 1080p/4K.

### Harness Migration

11. **Mixed harness usage** — `image_benchmark.rs` and `decoder_benchmark.rs`
    use Divan (fork-based), while `pipeline_benchmark.rs` and
    `mask_benchmark.rs` use the custom `edgefirst-bench` harness.
    Divan's forking model can crash on GPU targets.
    *All benchmarks should migrate to `edgefirst-bench`.*

---

## Changelog

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | 2026-03-04 | Initial document with strategy, platforms, and gap analysis |
