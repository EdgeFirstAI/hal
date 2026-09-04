# EdgeFirst Hardware Abstraction Layer

[![Build Status](https://github.com/EdgeFirstAI/hal/workflows/CI/badge.svg)](https://github.com/EdgeFirstAI/hal/actions)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Crates.io](https://img.shields.io/crates/v/edgefirst-tensor.svg)](https://crates.io/crates/edgefirst-tensor)
[![PyPI](https://img.shields.io/pypi/v/edgefirst-tensor.svg)](https://pypi.org/project/edgefirst-tensor/)

The EdgeFirst Hardware Abstraction Layer (HAL) is a Rust workspace providing
hardware-accelerated tensor management, image processing, ML model output
decoding, and multi-object tracking for edge AI inference pipelines. It ships
as a Rust crate, a Python package, and a C library, all built from the same
code. A single OpenGL ES engine runs on Linux (native EGL + DMA-BUF), macOS
and iOS (ANGLE over Metal + IOSurface), and Android (native EGL +
AHardwareBuffer), alongside NXP G2D on i.MX and a portable CPU fallback
everywhere else.

## Features

- **Zero-copy memory management** — DMA-BUF, IOSurface, AHardwareBuffer, POSIX shared memory, OpenGL PBO, and heap, with automatic backend selection
- **Zero-copy CUDA tensor mapping** — `convert()` PBO output mapped directly to a CUDA device pointer for TensorRT and other CUDA consumers; no host round-trip on Jetson (Orin-series). See [Zero-copy CUDA (TensorRT) input](#zero-copy-cuda-tensorrt-input).
- **Hardware-accelerated image processing** — OpenGL → G2D → CPU dispatch with shared cache infrastructure
- **Tiled inference (SAHI)** — overlapping tile grid rendered in one GPU pass, with IoS-based merge of per-tile detections back to full-frame coordinates. See [Tiled inference (SAHI)](#tiled-inference-sahi).
- **YOLO + ModelPack decoding** — YOLOv5 / v8 / v11 / v26 (incl. end-to-end) and ModelPack post-processing
- **Multi-object tracking** — ByteTrack with Kalman filtering and stable per-track UUIDs
- **Cross-platform** — Linux (i.MX 8M Plus / i.MX 95 / RPi 5 / Jetson / desktop), macOS, iOS, and Android, over CPU / GPU / zero-copy-buffer tiers

## Quick Start

### Installation

Python — install only the pieces you need:

```bash
pip install edgefirst-tensor    # the zero-copy tensor core
pip install edgefirst-codec     # JPEG/PNG decode (needs tensor)
pip install edgefirst-image     # convert/resize/draw (needs tensor, not decoder)
pip install edgefirst-decoder   # YOLO/ModelPack post-processing
pip install edgefirst-tracker   # ByteTrack (standalone)
```

Each is a self-contained wheel under the `edgefirst.` namespace; they
compose without any one pulling in the others.

Rust:

```sh
cargo add edgefirst-codec
cargo add edgefirst-image   # add only what you use
```

C: download `edgefirst-hal-<version>-<target>.tar.gz` (Linux) or `.zip`
(Windows, macOS) from
[GitHub Releases](https://github.com/EdgeFirstAI/hal/releases). The archive is
relocatable — see [`packaging/c/INSTALL.txt`](packaging/c/INSTALL.txt) for
pkg-config, runtime search path, and a JPEG→tensor example. Link
`libedgefirst_tensor` plus any of codec / image / decoder / tracker. Headers
live under `include/edgefirst/` (`tensor.h`, `codec.h`, `image.h`, `decoder.h`,
`tracker.h`, `detect.h`). Windows ships `bin/*.dll` and `lib/*.lib` import
libraries. **Before 1.0 the C ABI is stable across patch releases but may break across minors** — any `0.N.z` is drop-in for any other `0.N.z`, and `0.N` to `0.N+1` may not be. Mix the five libraries only within one minor. See [§ C ABI Stability and Versioning](https://github.com/EdgeFirstAI/hal/blob/main/ARCHITECTURE.md#c-abi-stability-and-versioning).

### Basic usage

**Python:**

```python
from edgefirst.codec import Tensor, decode_file_into
from edgefirst.decoder import Decoder
from edgefirst.image import ImageProcessor, PixelFormat

processor = ImageProcessor()

# Decode straight into a DMA/PBO-backed tensor for zero-copy convert(). A
# real pipeline allocates once and reuses the tensor every frame; JPEG
# decodes to its native Nv12, PNG to Rgb/Rgba/Grey. decode_file_into() is
# edgefirst.codec's free function for decoding into a tensor from *another*
# edgefirst.* package -- create_image() tensors can't be a decode_image()
# method's `self`, so this is the entry point that makes them decodable.
info = Tensor.peek_image_info_file("image.jpg")
# info.format is edgefirst.codec's own PixelFormat, and create_image() below
# is edgefirst.image's -- each package has its own type objects, but
# PixelFormat is a value type: it compares/hashes by value and is accepted
# in argument position across packages, so info.format works directly here.
src = processor.create_image(info.width, info.height, info.format, "uint8", "readwrite")
decode_file_into(src, "image.jpg")

model_input = processor.create_image(640, 640, PixelFormat.Rgb, "uint8", "readwrite")

# convert() handles the colour conversion and the resize in one call.
# Omit letterbox= to stretch to fill instead of preserving aspect ratio.
processor.convert(src, model_input, letterbox=[114, 114, 114, 255])

# outputs is the list of Tensors your inference engine produced from model_input.
decoder = Decoder(model_config, score_threshold=0.5, iou_threshold=0.45)
boxes, scores, classes, masks = decoder.decode(outputs)

# Fused decode + draw lives on the decoder so image-only installs stay decoder-free.
decoder.draw_onto(processor, outputs, model_input)
```

`edgefirst.codec.set_output_format(PixelFormat.Rgb)` (4:4:4 sources) or
`set_output_format(PixelFormat.Nv12)` (any colour source) opts a
JPEG decode into fusing that colour conversion into the decode itself,
in place of decode-then-`convert()`; `set_dct_method(DctMethod.Fast)`
trades a small, bounded accuracy loss for a faster IDCT. See
[`crates/python-codec/README.md`](https://github.com/EdgeFirstAI/hal/blob/main/crates/python-codec/README.md#quick-start).

**Rust:**

Depend on the crates you use. There is no umbrella `edgefirst-hal` crate.

```rust
use edgefirst_codec::{ImageDecoder, ImageLoad};
use edgefirst_image::{Crop, Flip, ImageProcessor, ImageProcessorTrait, Rotation};
use edgefirst_tensor::{CpuAccess, DType, PixelFormat};

let bytes = std::fs::read("image.jpg")?;
let mut processor = ImageProcessor::new()?;
let mut decoder = ImageDecoder::new();

// JPEG decodes to its native NV12 (colour); decode into an NV12 source tensor.
// load_image() reconfigures the tensor's shape and format to the decoded
// content, so allocate at or above the largest frame you expect.
let mut input =
    processor.create_image(1920, 1080, PixelFormat::Nv12, DType::U8, None, CpuAccess::Write)?;
let _info = input.load_image(&mut decoder, &bytes)?;

// convert() handles NV12 -> RGB and the letterbox resize in one call. The
// decode never rotates; pass the EXIF rotation here if you want it applied.
let mut output =
    processor.create_image(640, 640, PixelFormat::Rgb, DType::U8, None, CpuAccess::Read)?;
processor.convert(&input, &mut output, Rotation::None, Flip::None,
    Crop::letterbox([114, 114, 114, 255]))?;
```

`decoder.set_output_format(Some(PixelFormat::Rgb))` (4:4:4 sources) or
`Some(PixelFormat::Nv12)` (any colour source) fuses that colour
conversion into the decode itself, in place of decode-then-`convert()`;
`decoder.set_dct_method(DctMethod::Fast)` trades a small, bounded
accuracy loss for a faster IDCT. See
[`crates/codec/ARCHITECTURE.md#fused-decode-output-opt-in-pure-cpu`](https://github.com/EdgeFirstAI/hal/blob/main/crates/codec/ARCHITECTURE.md#fused-decode-output-opt-in-pure-cpu).

If you prefer to depend on the sub-crates directly (e.g. to opt out of
features or to track them at independent versions), add the relevant
`edgefirst-image`, `edgefirst-tensor`, `edgefirst-decoder`, and
`edgefirst-tracker` entries to your `Cargo.toml` and use the
unprefixed `edgefirst_image::*` / `edgefirst_tensor::*` paths above.

**C:**

```c
#include <edgefirst/image.h>
#include <edgefirst/tensor.h>

ef_image_processor *proc = ef_image_processor_new();
/* `src` is decoded with libedgefirst_codec (`ef_image_decoder_decode_file_into`)
 * or wrapped from a dma-buf with `ef_tensor_builder_wrap`. */
ef_tensor *src = /* ... */;
ef_tensor *dst = ef_image_processor_create_image(
    proc, 640, 640, /* format/dtype/access: see image.h */);
ef_image_processor_convert(proc, src, dst, /* rotation, flip, crop */);
```

### Zero-copy CUDA (TensorRT) input

On CUDA-capable devices (e.g. Jetson Orin-series) the float PBO produced
by `convert()` can be mapped directly to a CUDA device pointer with no
host round-trip. The recommended pattern is to try `cuda_map()` first and
fall back to the host `map()` when CUDA is unavailable:

**Rust:**

```rust
use edgefirst_tensor::{is_cuda_available, TensorTrait};

// At pipeline startup — check once
if is_cuda_available() {
    println!("CUDA present; will use zero-copy PBO→CUDA path");
}

// Per frame — try CUDA, fall back to host
if let Some(cuda) = dst.cuda_map() {
    // cuda.device_ptr() is a raw device pointer valid until `cuda` is dropped.
    // Drop `cuda` before the next convert() so the PBO is free to be reused.
    trt_enqueue(cuda.device_ptr(), cuda.len());
    // `cuda` drops here → PBO released
} else {
    let host = dst.map()?;
    trt_enqueue_host(host.as_slice());
}
```

**Python:**

```python
from edgefirst.image import ImageProcessor, PixelFormat

proc = ImageProcessor()
dst = proc.create_image(640, 640, PixelFormat.PlanarRgb, "float16")

for frame in camera_frames:
    proc.convert(frame, dst)
    cuda = dst.cuda_map()          # CudaMap | None
    if cuda is not None:
        with cuda:
            # cuda.device_ptr is a CUDA device pointer (int)
            trt_context.execute(cuda.device_ptr)
    else:
        host = dst.map()
        trt_context.execute_host(bytes(host))
```

`cuda_map()` fast-fails to `None` when `libcudart` is not present at
runtime — no compile-time feature gate, no link-time dependency. CUDA
register/map runs on the GL worker thread; the returned device pointer
is usable from any thread. Drop the `CudaMap` guard before the next
`convert()` call to release the PBO back to the GL pipeline.

For the full mechanism, aliasing rules, DMA-BUF import path, and
per-language API reference, see
[crates/tensor/README.md § CUDA tensor mapping](https://github.com/EdgeFirstAI/hal/blob/main/crates/tensor/README.md#cuda-tensor-mapping)
and
[crates/tensor/ARCHITECTURE.md § Zero-copy CUDA tensor mapping](https://github.com/EdgeFirstAI/hal/blob/main/crates/tensor/ARCHITECTURE.md#zero-copy-cuda-tensor-mapping).

### Tiled inference (SAHI)

Small objects in a high-resolution frame disappear when the whole frame is
squeezed down to a 640×640 model input. SAHI (Slicing Aided Hyper Inference)
runs the same model at its native resolution over an overlapping grid of
native-resolution crops instead, then stitches the per-tile detections back
together. HAL covers both halves: `edgefirst-image` cuts and renders the grid,
`edgefirst-decoder` lifts and merges the results.

The input side renders every tile into one tall packed batch tensor with a
single GL import and a single flush, so N tiles cost roughly one GPU sync
rather than N. The output side merges with a greedy, class-aware pass under the
**IoS** (intersection-over-smaller) metric, because an object split across a
tile seam has low IoU with its own fragments but high IoS; by default it keeps
the highest-scoring box of each matched group unchanged
(`MergeMode::KeepBest`). A `TilePlacement` produced by
`plan_tiles` / `tile_into` is the shared record of how each tile was cut, and it
is what the merge uses to lift boxes back to full-frame coordinates.

**Rust:**

```rust
use edgefirst_image::{ImageProcessor, ImageProcessorTrait, TilingConfig};
use edgefirst_decoder::{DecoderBuilder, DetectBox, Nms, Segmentation};
use edgefirst_decoder::tiling::{MergeConfig, TiledFrameAccumulator};
use edgefirst_tensor::{CpuAccess, DType, PixelFormat};

// 640x640 tiles with at least 20% overlap. The realized overlap is
// redistributed evenly so every tile is full-size and the last one lands flush.
let cfg = TilingConfig::new(640, 640).with_overlap(0.2);

// plan_tiles is pure geometry (no GPU work), so its length sizes the batch.
let placements = processor.plan_tiles(src_w, src_h, &cfg)?;

// One tall [tile_w, N * tile_h] destination. Allocate once, reuse per frame.
let mut batch = processor.alloc_tile_batch(
    placements.len(), &cfg, PixelFormat::Rgb, DType::U8, None, CpuAccess::None)?;

// Render every tile: deferred convert per tile, one flush at the end.
let placements = processor.tile_into(&src, &mut batch, &cfg)?;

// Per-tile decoding is deliberately permissive. A fragment clipped at a seam
// scores low, and a high per-tile threshold discards it before the merge can
// rebuild the object. Gate the final scores in MergeConfig instead.
let decoder = DecoderBuilder::new()
    .with_config_yaml_str(model_config_yaml)
    .with_score_threshold(0.05)
    .with_nms(Some(Nms::ClassAware))
    .build()?;

let mut acc = TiledFrameAccumulator::new(
    (src_w as f32, src_h as f32),
    placements.len(),       // tiles_total — the fan-in fence
    MergeConfig::default(), // Ios metric, 0.5 threshold, keep-best, max_det 300
    16,                     // estimated detections per tile (capacity hint)
);

// HAL does not run inference. `tile_results` pairs each placement with the
// output tensors your engine produced for that tile; tiles may arrive in any
// order, so pair them explicitly rather than relying on loop position.
for (tile_outputs, placement) in tile_results {
    let mut boxes: Vec<DetectBox> = Vec::new();
    let mut masks: Vec<Segmentation> = Vec::new();
    decoder.decode(&tile_outputs, &mut boxes, &mut masks)?;
    acc.push_tile(boxes, &placement);
}

// Merged, deduplicated, normalized to [0, 1] for the tracker.
let detections = acc.finalize_normalized();
```

**Python:**

```python
from edgefirst.image import TilingConfig, PixelFormat
from edgefirst.decoder import TiledFrameAccumulator, MergeConfig

cfg = TilingConfig(640, 640, overlap=0.2)
placements = processor.plan_tiles(src.width, src.height, cfg)

batch = processor.alloc_tile_batch(len(placements), cfg, PixelFormat.Rgb)
placements = processor.tile_into(src, batch, cfg)

acc = TiledFrameAccumulator(
    (float(src.width), float(src.height)), len(placements), MergeConfig())

for tile_outputs, placement in tile_results:
    boxes, scores, classes, _masks = decoder.decode(tile_outputs)
    acc.push_tile(boxes, scores, classes, placement)

boxes, scores, classes = acc.finalize_normalized()
```

`push_tile` is idempotent per `placement.index`, so tiles can arrive in any
order and an at-least-once delivery retry stays harmless. The merge runs once
at `finalize`, never per push, which is what a pipelined runtime needs:
`plan_tiles` sizes the ring up front, `tile_one` streams individual tiles
through inference into a caller-owned slot, and `is_complete()` / `remaining()`
fence the frame.

`MergeConfig` tunes the metric (`Ios` by default, or `Iou`), the match
`threshold` (0.5), `class_agnostic` (false), `max_det` (300), a final
`score_threshold` (0.0), and the `mode` (`KeepBest` by default, or `Union`).
The `score_threshold` default is deliberate: per-tile decoding is the real
flood control, and the merged score gate belongs after fragments have been
joined.

> [!NOTE]
> `MergeMode::KeepBest` suppresses: the best box of each matched group comes
> back unchanged, never grown, so an object larger than a single tile is reported
> as its best fragment. `MergeMode::Union` (the original GREEDYNMM behaviour)
> grows that box to the group's enclosing union instead, but it measured about
> 0.05 AP50 worse on every frame of the Ocean Cleanup ADIS 4K validation
> (whole frame: 0.491 AP50 with plain NMS, 0.437 after the union, 0.490 with
> keep-best), so it is opt-in. For mixed-scale datasets, add a full-frame
> downscaled pass as one more `push_tile` at `origin=(0, 0)`,
> `crop_size=frame_dims`.

The C API mirrors the same split (`ef_image_processor_plan_tiles` /
`ef_image_processor_tile_into` / `ef_image_processor_tile_one` on the input side, `ef_tiled_frame_accumulator_*`
on the output side). Full per-language detail lives in
[image/README.md § Tiled Preprocessing](https://github.com/EdgeFirstAI/hal/blob/main/crates/image/README.md#tiled-preprocessing-sahi)
and
[decoder/README.md § Tiled Inference](https://github.com/EdgeFirstAI/hal/blob/main/crates/decoder/README.md#tiled-inference-sahi).

Per-language quick-starts and richer examples live in each crate's README:
[Rust](https://github.com/EdgeFirstAI/hal/blob/main/crates/tensor/README.md),
[C API](https://github.com/EdgeFirstAI/hal/blob/main/crates/tensor-capi/README.md),
[Python](https://github.com/EdgeFirstAI/hal/blob/main/crates/python-tensor/README.md).

## System Architecture

```mermaid
graph TB
    subgraph "EdgeFirst HAL Ecosystem"
        PyT["edgefirst-tensor<br/>wheel"]
        PyC["edgefirst-codec<br/>wheel"]
        PyI["edgefirst-image<br/>wheel"]
        PyD["edgefirst-decoder<br/>wheel"]
        PyK["edgefirst-tracker<br/>wheel"]

        TensorSo["libedgefirst_tensor.so"]
        CodecSo["libedgefirst_codec.so"]
        ImageSo["libedgefirst_image.so"]
        DecoderSo["libedgefirst_decoder.so"]
        TrackerSo["libedgefirst_tracker.so"]

        PyT --> TensorSo
        PyC --> CodecSo
        PyC --> TensorSo
        PyI --> ImageSo
        PyI --> TensorSo
        PyD --> DecoderSo
        PyD --> TensorSo
        PyK --> TrackerSo

        CodecSo --> TensorSo
        ImageSo --> TensorSo
        DecoderSo --> TensorSo
    end
```

## Core Components

| Crate | Role | Architecture | Testing |
|-------|------|--------------|---------|
| [`edgefirst-tensor`](https://github.com/EdgeFirstAI/hal/blob/main/crates/tensor/) | Zero-copy multi-dim buffers (DMA / SHM / Mem / PBO) | [ARCH](https://github.com/EdgeFirstAI/hal/blob/main/crates/tensor/ARCHITECTURE.md) | [TEST](https://github.com/EdgeFirstAI/hal/blob/main/crates/tensor/TESTING.md) |
| [`edgefirst-codec`](https://github.com/EdgeFirstAI/hal/blob/main/crates/codec/) | JPEG/PNG decode into pre-allocated tensors (strided, multi-dtype) | [ARCH](https://github.com/EdgeFirstAI/hal/blob/main/crates/codec/ARCHITECTURE.md) | [TEST](https://github.com/EdgeFirstAI/hal/blob/main/crates/codec/TESTING.md) |
| [`edgefirst-image`](https://github.com/EdgeFirstAI/hal/blob/main/crates/image/) | OpenGL / G2D / CPU image processor + mask rendering | [ARCH](https://github.com/EdgeFirstAI/hal/blob/main/crates/image/ARCHITECTURE.md) | [TEST](https://github.com/EdgeFirstAI/hal/blob/main/crates/image/TESTING.md) |
| [`edgefirst-decoder`](https://github.com/EdgeFirstAI/hal/blob/main/crates/decoder/) | YOLO + ModelPack post-processing, NMS, proto-mask APIs | [ARCH](https://github.com/EdgeFirstAI/hal/blob/main/crates/decoder/ARCHITECTURE.md) | [TEST](https://github.com/EdgeFirstAI/hal/blob/main/crates/decoder/TESTING.md) |
| [`edgefirst-tracker`](https://github.com/EdgeFirstAI/hal/blob/main/crates/tracker/) | ByteTrack multi-object tracking | [ARCH](https://github.com/EdgeFirstAI/hal/blob/main/crates/tracker/ARCHITECTURE.md) | [TEST](https://github.com/EdgeFirstAI/hal/blob/main/crates/tracker/TESTING.md) |
| [`edgefirst-tensor-capi`](https://github.com/EdgeFirstAI/hal/blob/main/crates/tensor-capi/) | C ABI: `libedgefirst_tensor.so` + `edgefirst/tensor.h` | [ARCH](https://github.com/EdgeFirstAI/hal/blob/main/crates/tensor-capi/ARCHITECTURE.md) | [TEST](https://github.com/EdgeFirstAI/hal/blob/main/crates/tensor-capi/tests/c/) |
| [`edgefirst-codec-capi`](https://github.com/EdgeFirstAI/hal/blob/main/crates/codec-capi/) | C ABI: `libedgefirst_codec.so` + `edgefirst/codec.h` | [ARCH](https://github.com/EdgeFirstAI/hal/blob/main/crates/codec-capi/ARCHITECTURE.md) | [TEST](https://github.com/EdgeFirstAI/hal/blob/main/crates/codec-capi/tests/c/) |
| [`edgefirst-image-capi`](https://github.com/EdgeFirstAI/hal/blob/main/crates/image-capi/) | C ABI: `libedgefirst_image.so` + `edgefirst/image.h` | [ARCH](https://github.com/EdgeFirstAI/hal/blob/main/crates/image-capi/ARCHITECTURE.md) | [TEST](https://github.com/EdgeFirstAI/hal/blob/main/crates/image-capi/tests/c/) |
| [`edgefirst-decoder-capi`](https://github.com/EdgeFirstAI/hal/blob/main/crates/decoder-capi/) | C ABI: `libedgefirst_decoder.so` + `edgefirst/decoder.h` | [ARCH](https://github.com/EdgeFirstAI/hal/blob/main/crates/decoder-capi/ARCHITECTURE.md) | [TEST](https://github.com/EdgeFirstAI/hal/blob/main/crates/decoder-capi/tests/c/) |
| [`edgefirst-tracker-capi`](https://github.com/EdgeFirstAI/hal/blob/main/crates/tracker-capi/) | C ABI: `libedgefirst_tracker.so` + `edgefirst/tracker.h` | [ARCH](https://github.com/EdgeFirstAI/hal/blob/main/crates/tracker-capi/ARCHITECTURE.md) | [TEST](https://github.com/EdgeFirstAI/hal/blob/main/crates/tracker-capi/tests/c/) |
| `crates/python-*` (five PyPI wheels) | PyO3 bindings; each links `libedgefirst_tensor.so` | [ARCH](https://github.com/EdgeFirstAI/hal/blob/main/crates/python-common/ARCHITECTURE.md) | [TEST](https://github.com/EdgeFirstAI/hal/blob/main/crates/python-common/TESTING.md) |

The deep dive on each component (class diagrams, supported operations,
backend dispatch, performance considerations) lives in the per-crate
`ARCHITECTURE.md`. The cross-cutting story (DMA-BUF identity, performance
tracing internals, design patterns) lives in the project
[ARCHITECTURE.md](https://github.com/EdgeFirstAI/hal/blob/main/ARCHITECTURE.md).

## Optimization Guide

This section is the **rules** part of the cross-language performance
contract. Each rule has a measurable cost when broken; see
[BENCHMARKS.md](https://github.com/EdgeFirstAI/hal/blob/main/BENCHMARKS.md)
for empirical penalties per platform,
[ARCHITECTURE.md](https://github.com/EdgeFirstAI/hal/blob/main/ARCHITECTURE.md)
for *why* the rule exists, and
[TESTING.md](https://github.com/EdgeFirstAI/hal/blob/main/TESTING.md#validating-optimizations)
for how to verify your integration follows it.

| Rule | Why it matters | Measured penalty when broken |
|------|----------------|------------------------------|
| Reuse tensors across frames | Each new tensor mints a fresh `BufferIdentity`; the EGL image cache misses every frame | 1.7–3.3× slower preprocessing on Vivante / Mali |
| Allocate via `ImageProcessor::create_image()` | Auto-selects DMA-buf / PBO / heap based on the active GPU; bypassing forces a slow transfer path | Forced `glTexSubImage2D` upload or full CPU readback |
| Cache imported camera tensors by **inode**, not by fd | V4L2 / libcamera recycle fd numbers across a small buffer pool; an fd-keyed cache misses on every frame even when the physical buffer is the same | Full EGL re-import per frame (≈0.5–1.5 ms on Vivante, doubled with chroma planes) |
| Build `Decoder` once, decode many | Decoder construction parses model metadata and allocates working buffers | Parse + alloc cost per frame |
| One `ImageProcessor` per pipeline | Each instance owns its own GL context, EGL display, and per-thread caches | On Vivante / paravirtual GPUs multiple contexts serialize on the global `GL_MUTEX`; on Mali / V3D / Tegra / Apple they run concurrently (one per thread is the portable rule) |
| Use native fp16 / AVX build overrides only on supporting CPUs | These flags unlock native widening / vector paths for local perf testing | Unsupported targets may SIGILL or fail to build; portability loss |
| Pass numpy arrays straight to `Tensor.from_numpy()` — do not pre-`ascontiguousarray()` | HAL detects strided sources and materializes via numpy's vectorized C strided→contig pass; a manual workaround above HAL adds a redundant copy | Redundant pre-copy on every call (≈ 1.5 ms on a `(1, 116, 8400)` f32 view, rpi5-hailo) |
| For COCO/IoU evaluation use `MaskResolution::Scaled { width, height }`, not `Proto` | `Scaled` upsamples the proto plane *before* thresholding (clean sub-pixel edges); `Proto` thresholds at proto resolution and callers typically nearest-upsample (blocky) | Mask mAP regression of up to 0.04–0.05 absolute when `Proto` is nearest-upsampled |

> [!IMPORTANT]
> The single most common performance bug is calling `Tensor::from_fd()`
> (or `import_image()`) on every frame from a V4L2 / libcamera buffer
> pool. The HAL's internal EGL image cache cannot rescue you — the cache
> key includes a per-tensor monotonic ID that is fresh on every import.
> The fix lives in the **calling code**, not in HAL.

### Rule 1 — Reuse tensors across frames

Allocate input and output tensors once at pipeline startup; reuse the same
objects on every frame. The DMA memory backing a tensor is live: when an
upstream producer (V4L2 DQBUF, codec output, ISP) writes new pixels into
it, the existing tensor and its cached EGLImage remain valid. No
re-import, no re-allocation.

```rust
let mut proc = ImageProcessor::new()?;
let mut dst = proc.create_image(640, 640, PixelFormat::Rgb, DType::U8, None, CpuAccess::ReadWrite)?;

for frame in camera_frames {
    proc.convert(&frame, &mut dst, Rotation::None, Flip::None, Crop::default())?;
    run_inference(&dst)?;
}
```

```python
from edgefirst.image import ImageProcessor, PixelFormat

proc = ImageProcessor()
dst = proc.create_image(640, 640, PixelFormat.Rgb)
for frame in camera_frames:
    proc.convert(frame, dst)
    run_inference(dst)
```

### Rule 2 — Allocate via `ImageProcessor::create_image()`

`create_image()` selects the fastest memory backend for the active GPU at
construction time:

| Priority | Backend | Transfer | Platforms |
|----------|---------|----------|-----------|
| 1st | **DMA-buf** | Zero-copy EGLImage import | NXP i.MX 8M Plus, i.MX 95 |
| 2nd | **PBO** | Zero-copy GL buffer binding | NVIDIA desktop |
| 3rd | **Mem** (heap) | CPU memcpy fallback | All platforms |

The probe runs once at `ImageProcessor::new()` time. All subsequent
`create_image()` calls reuse the same backend. Use `create_image()` for
every destination passed to `convert()`; direct `Tensor::new(memory=...)`
bypasses the probe.

**Declare CPU access.** Every image constructor takes a required
`CpuAccess` parameter: hardware (GPU/NPU/ISP/codec) access is always
implied, CPU access is the opt-in. Declare `Write` for decode targets,
`Read` for buffers you verify/consume on the CPU, `ReadWrite` when both,
and `CpuAccess::None` for pure hardware pipelines — on Android a
hardware-only buffer is eligible for gralloc's vendor tile compression
(UBWC/AFBC/PVRIC/DCC), and on every platform the declaration selects the
cheapest mapping mode (write-combined for `Write`, read-only IOSurface
locks / dma-buf sync direction for `Read`). Mapping beyond the
declaration still works best-effort but warns once per buffer and counts
in `unplanned_cpu_access_count()`; on Android hardware-only buffers
refuse CPU maps deterministically.

For DMA-buf access, the process needs `/dev/dma_heap/{linux,cma|system}`
and a DRM render/card node — the GL backend probes
`/dev/dri/renderD128`, then `/dev/dri/card0`, then `/dev/dri/card1` and
uses the first one that opens. On embedded Linux, add the user to
`video` and `render` groups, or set udev rules. If DMA-buf fails,
`create_image()` transparently falls back to PBO or heap.

### Rule 3 — Cache imported camera tensors by inode, not by fd

V4L2, libcamera, and codec output all surface frames as DMA-BUF file
descriptors drawn from a small fixed pool (typically 4–16 buffers). The fd
**number** is recycled: the same fd can refer to a different physical
buffer between frames, and the same physical buffer can be exported with a
different fd over time. **A cache keyed by fd will produce false hits or
false misses.**

The kernel assigns each `dma_buf` object a unique inode in the anonymous
inode filesystem. The inode is constant for the buffer's lifetime
regardless of how many times it is exported. Cache imported HAL tensors
by `(inode, plane_offset)`:

```c
#include <sys/stat.h>

typedef struct { ino_t inode; size_t offset; } BufferKey;

struct stat st;
if (fstat(fd, &st) != 0) continue;
BufferKey key = { .inode = st.st_ino, .offset = plane_offset };

struct ef_tensor *tensor = lookup_tensor(cache, &key);
if (!tensor) {
    ef_tensor_builder *b = ef_tensor_builder_new();
    if (!b) { perror("ef_tensor_builder_new"); continue; }
    ef_tensor_builder_dtype(b, EF_DTYPE_U8);
    ef_tensor_builder_format(b, "NV12");
    ef_tensor_builder_add_plane(b, fd, plane_offset, 0, 0, 0, 0);
    tensor = ef_tensor_builder_wrap(b);
    ef_tensor_builder_free(b);
    if (!tensor) { perror("ef_tensor_builder_wrap"); continue; }
    insert_tensor(cache, &key, tensor);
}
ef_image_processor_convert(proc, tensor, dst, /* rotation, flip, crop */);
```

```python
import os
buffer_cache: dict[tuple[int, int], ef.Tensor] = {}

def get_or_import(proc, fd, offset, width, height, fmt):
    key = (os.fstat(fd).st_ino, offset)
    t = buffer_cache.get(key)
    if t is None:
        t = proc.import_image(fd, width, height, fmt, "uint8", offset=offset)
        buffer_cache[key] = t
    return t
```

EdgeFirst's GStreamer elements implement this as a reference. For other
pipelines (libcamera direct, custom V4L2, RTSP decoder) you are
responsible for the equivalent layer above HAL. See
[ARCHITECTURE.md § Appendix C](https://github.com/EdgeFirstAI/hal/blob/main/ARCHITECTURE.md#appendix-c-dma-buf-identity-and-tensor-caching)
for the full identity-and-caching story.

### Rule 4 — Build the decoder once

`Decoder` parses the model output schema, resolves quantization, and
allocates working buffers at construction time. Build it once outside the
loop; the decoder clears its output vectors per call:

```rust
let decoder = DecoderBuilder::default()
    .with_config_yaml_str(config_yaml)
    .with_score_threshold(0.5)
    .with_iou_threshold(0.45)
    .build()?;

for frame in frames {
    let outputs = run_inference(frame)?;
    let refs: Vec<&TensorDyn> = outputs.iter().collect();
    decoder.decode(&refs, &mut boxes, &mut masks)?;
}
```

The same applies to `ByteTrack`: construct once, call `update()` per
frame.

### Rule 5 — One `ImageProcessor` per pipeline

`ImageProcessor` owns its OpenGL context, dedicated GL thread, and EGL
image cache. The EGL **display** itself is process-global (a shared
`SharedEglDisplay` initialized once and never terminated), so additional
processors don't pay the display-creation cost — but each one still
creates a fresh context and per-instance caches. Whether GL operations
across processors run in parallel is a per-driver policy: on Vivante
(i.MX 8M Plus) and virtualized/paravirtual GPUs every command serializes
on a global `GL_MUTEX`; on Mali, V3D, Tegra, llvmpipe, and real Apple
GPUs they execute concurrently (override with `EDGEFIRST_GL_SERIALIZE`).
Construct one per pipeline (or one per worker thread for parallel
pipelines) and share it across all `convert()`, `draw_*()`, and
`create_image()` calls.

`ImageProcessor` is `Send + Sync`, so it can be moved or shared across
threads. On serializing drivers, concurrent use of a single shared
instance funnels through `GL_MUTEX`; per-worker ownership runs in
parallel wherever the driver allows and gives more predictable cache
behaviour everywhere.

### Rule 6 — Local fp16 / AVX build overrides

The default HAL binary is built to the target triple's guaranteed
baseline ISA so a single distributed binary runs on every CPU within that
triple. Richer ISAs (ARMv8.2-FP16, x86_64 F16C / FMA / AVX2) are **not**
enabled by default; until HAL gains runtime CPU-feature detection with
dynamic dispatch, baking them in would SIGILL on older CPUs.

For local benchmarking on supporting hosts, enable them via `RUSTFLAGS`:

```bash
# Orin Nano (Cortex-A78AE) — exclude the PyO3 binding (cross-Python toolchain not configured)
RUSTFLAGS="-C target-cpu=cortex-a78ae" cargo build --release \
  --target aarch64-unknown-linux-gnu --workspace --exclude edgefirst-python-tensor \
  --exclude edgefirst-python-codec --exclude edgefirst-python-image \
  --exclude edgefirst-python-decoder --exclude edgefirst-python-tracker

# Generic aarch64 with FEAT_FP16 (do NOT use on Cortex-A53 / imx8mp)
RUSTFLAGS="-C target-feature=+fp16" cargo build --release \
  --target aarch64-unknown-linux-gnu -p edgefirst-image

# x86_64 Haswell+ (F16C + FMA + AVX2)
RUSTFLAGS="-C target-feature=+f16c,+fma,+avx2" cargo build --release \
  -p edgefirst-image
```

When active, the f16 mask kernel at
[`crates/image/src/cpu/masks.rs`](https://github.com/EdgeFirstAI/hal/blob/main/crates/image/src/cpu/masks.rs)
compiles to native widening (`fcvt` on aarch64, `vcvtph2ps` on x86_64),
and on x86_64 with `+f16c,+fma` an explicit 8-lane `_mm256_cvtph_ps +
_mm256_fmadd_ps` intrinsic path is enabled via cfg gate. Verify with
[`scripts/audit_f16_codegen.sh`](https://github.com/EdgeFirstAI/hal/blob/main/scripts/audit_f16_codegen.sh).

### Rule 7 — NumPy interop: pass arrays straight to `from_numpy()`

`Tensor.from_numpy()` (and the implicit copy from numpy arrays passed to
`Decoder.decode_proto()`) handles strided / non-contiguous sources
internally. Do **not** maintain a manual `np.ascontiguousarray()`
workaround — it wastes a copy.

The Python binding's `copy_numpy_to_tensor_dyn` selects one of three
paths based on the source array's layout:

| Source layout | Path | Cost |
|---|---|---|
| Fully contiguous | Single `copy_from_slice` (memcpy), rayon-parallel ≥ 256 KiB | Lower bound |
| Strided with contiguous inner rows (column slice, sub-volume, negative stride) | Per-row memcpy iterating outer dimensions | ≈ same as contiguous |
| Fully strided (transposed view, every-other-element) | Internal `np.ascontiguousarray()` materialization, then Path 1 memcpy | ≈ 4× contiguous |

The fully-strided case is the one that bites users in practice: HailoRT's
natural output is `arr.transpose(0, 2, 1)` over a `(1, anchors,
channels)` buffer. PR #58 replaced the legacy element-wise loop with
internal `np.ascontiguousarray` materialization (≈ 4× faster than the
legacy loop, within ≈ 1.5× of the manual workaround).

```python
# Wrong (post-PR #58): adds an extra copy above HAL.
tensor.from_numpy(np.ascontiguousarray(arr_strided))

# Right: HAL detects the strided layout and materializes internally.
tensor.from_numpy(arr_strided)
```

The regression tests in
[`tests/test_tensor.py`](https://github.com/EdgeFirstAI/hal/blob/main/tests/test_tensor.py)
(`test_from_numpy_hailort_shape`,
`test_from_numpy_hailort_shape_perf_sanity`) pin the behaviour and the
≤ 1.5× perf bound.

### Rule 8 — Choose the correct `MaskResolution`

`ImageProcessor.materialize_masks()` accepts a `MaskResolution`
parameter:

| Mode | Output | Pipeline | When to use |
|------|--------|----------|-------------|
| `MaskResolution::Proto` (default) | `(roi_h, roi_w, 1)` u8 binary at 160×160 proto resolution | dot → sign threshold → emit | Real-time visualization, when proto-resolution binary suffices |
| `MaskResolution::Scaled { width, height }` | `(roi_h, roi_w, 1)` u8 binary at requested resolution | dot → sigmoid → upsample to `(W, H)` → threshold (`>127`) | All COCO / IoU / mAP evaluation |

```python
from edgefirst.image import MaskResolution

# Wrong: threshold then upsample → blocky edges, mAP regression.
tiles = proc.materialize_masks(boxes, scores, classes, proto_data, letterbox=lb)
for tile, box in zip(tiles, boxes):
    binary = (tile[:, :, 0] > 127).astype(np.uint8)
    canvas[y:y+h, x:x+w] = cv2.resize(binary, (W, H), cv2.INTER_NEAREST)

# Right: HAL upsamples-then-thresholds inside its batched-GEMM kernel.
tiles = proc.materialize_masks(boxes, scores, classes, proto_data,
                               letterbox=lb,
                               resolution=MaskResolution.Scaled(W, H))
for tile, box in zip(tiles, boxes):
    canvas[y:y+h, x:x+w] = (tile[:, :, 0] > 127).astype(np.uint8)
```

The `Scaled` path uses the batched-GEMM materializer (PR #54). At N ≥ 16
detections it amortizes a single GEMM at proto resolution and upsamples
per-detection in rayon-parallel — both more accurate than
threshold-then-resize *and* faster than per-detection scalar work in
caller code.

> [!TIP]
> If you see a mask-mAP gap between your HAL validator and a reference
> (ONNX / numpy) implementation, this rule is almost always the first
> thing to check.

### Where to go next

| Document | Level | Use it for |
|----------|-------|------------|
| [ARCHITECTURE.md § Appendix C: DMA-BUF Identity and Tensor Caching](https://github.com/EdgeFirstAI/hal/blob/main/ARCHITECTURE.md#appendix-c-dma-buf-identity-and-tensor-caching) | Architecture | Why the rules exist: `BufferIdentity`, EGL image cache, the v4l2 / GStreamer fd-recycling story, and the inode-keyed downstream cache pattern |
| [image/ARCHITECTURE.md § Performance Considerations](https://github.com/EdgeFirstAI/hal/blob/main/crates/image/ARCHITECTURE.md#performance-considerations) | Architecture | Backend dispatch and per-instance caches; see also [§ GL Concurrency Model](https://github.com/EdgeFirstAI/hal/blob/main/crates/image/ARCHITECTURE.md#gl-concurrency-model-serialization-policy) for the per-driver `GL_MUTEX` policy |
| [TESTING.md § Validating Optimizations](https://github.com/EdgeFirstAI/hal/blob/main/TESTING.md#validating-optimizations) | Testing | Confirming your integration follows the rules |
| [BENCHMARKS.md](https://github.com/EdgeFirstAI/hal/blob/main/BENCHMARKS.md) | Benchmarks | Empirical cost of breaking each rule, per platform |

## Toolchain and Platform Floors

Two version floors are project policy. Both are chosen, not derived from
whatever hardware happens to be on hand, and both are deliberately
conservative — raising either one drops users.

| Floor | Value | Applies to |
|-------|-------|-----------|
| Rust (MSRV) | **1.94.0** | Everything. Declared as `rust-version`, so cargo refuses an older toolchain with a clear error rather than a wall of syntax failures. CI gates on the same version. |
| glibc | **2.35** | Cross-compiled Linux binaries (Ubuntu 22.04's glibc, matching the CI runners). |

The MSRV is set once in `[workspace.package]` and inherited by every
crate. The workspace uses `resolver = "3"`, the MSRV-aware resolver, so
`rust-version` also constrains **dependency selection**: `cargo update`
will not pull a dependency release that requires a newer compiler.
Raising the MSRV is therefore a user-visible change, not just a build
detail — bump it here, in the CI `RUST_STABLE_VERSION`, and in the
CHANGELOG together.

No source change was needed to support 1.94.0; it records the floor the
code already met. Verified against a real 1.94.0 toolchain across the
combinations that compile genuinely different code:

| Checked on 1.94.0 | Covers |
|---|---|
| macOS host, all crates, `--all-targets` | IOSurface, ANGLE/GL, the PyO3 bindings |
| `aarch64-unknown-linux-gnu`, `--all-targets` | DMA-BUF, SHM, the `dmabuf` ioctls, G2D |
| `edgefirst-tensor --features tracing` | `trace.rs`, which is **off** by default |
| `edgefirst-tensor --no-default-features --features static` | the `ndarray`-free build |
| `edgefirst-codec --no-default-features --features static` | PNG/V4L2/nvJPEG all gated out |
| `edgefirst-image --features tracker` | the optional tracker path |

`--no-default-features` always needs `static` (or `dynamic`) named alongside
it: `edgefirst-tensor`'s two backend features are mutually exclusive and one
is required, so `--no-default-features` with neither is a compile error, not
a fallback to a default.

The `python-*` crates cannot be cross-compiled without
`PYO3_CROSS_PYTHON_VERSION` or an `abi3-py3*` feature, so they are covered
by the host build rather than the Linux one.

The glibc floor is what
[`scripts/on-target-test.sh`](https://github.com/EdgeFirstAI/hal/blob/main/scripts/on-target-test.sh)
builds against (`cargo-zigbuild --target
<arch>-unknown-linux-gnu.2.35`), so one binary set runs on every supported
target. Building against a newer glibc links symbols an older loader cannot
resolve, and the failure appears only at run time on the oldest device you
own. Override for a one-off with `GLIBC=<version>`; change the default in
the script and here together.

Published Python wheels use a **separate, lower** floor — manylinux2014
(glibc 2.17), via `maturin --zig --compatibility manylinux2014` — because
they are built for the wider PyPI audience rather than for our targets.
`make wheel` applies it.

## Platform Support

| Feature | Linux (i.MX) | Linux (other) | macOS | iOS | Android | Windows |
|---------|--------------|---------------|-------|-----|---------|---------|
| DMA tensors | Yes | Yes | No | No | No | No |
| PBO tensors (GPU) | Yes | Yes | No | No | No | Yes (with ANGLE) |
| IOSurface tensors (zero-copy) | No | No | Yes (with ANGLE) | Yes (with ANGLE) | No | No |
| AHardwareBuffer tensors (zero-copy) | No | No | No | No | Yes | No |
| Shared memory tensors | Yes | Yes | Yes | Yes | Import-only¹ | No |
| Heap tensors | Yes | Yes | Yes | Yes | Yes | Yes |
| G2D acceleration | Yes | No | No | No | No | No |
| OpenGL acceleration | Yes (optional) | Yes (optional) | Yes (with ANGLE) | Yes (with ANGLE) | Yes (native EGL) | Yes (with ANGLE, Direct3D 11) |
| CPU fallback | Yes | Yes | Yes | Yes | Yes | Yes |

¹ Android's bionic libc has no POSIX `shm_open`, so shared-memory tensor
*allocation* reports `NotImplemented`; *importing* an existing segment
received as a file descriptor (`from_fd`) works.

On macOS the OpenGL backend is enabled when [ANGLE](https://github.com/google/angle)
is installed — see [macOS GPU Acceleration](#macos-gpu-acceleration) below
for setup. If ANGLE is not present the HAL falls back to the CPU backend.
On iOS the OpenGL backend uses the same ANGLE-over-Metal path — see
[iOS](#ios) below. On Android the OpenGL backend uses the platform's
native GLES driver directly (no translation layer) — see
[Android](#android) below. On Windows the OpenGL backend runs the same
engine on ANGLE over Direct3D 11 with PBO transfers — see
[Windows GPU Acceleration](#windows-gpu-acceleration) below.

## macOS GPU Acceleration

The HAL uses [Google's ANGLE](https://github.com/google/angle) to translate
the same OpenGL ES 3.0 calls used on Linux to Metal, and Apple's
[IOSurface](https://developer.apple.com/documentation/iosurface) for
zero-copy buffer interchange (the role DMA-BUF plays on Linux). ANGLE is
not part of macOS and must be installed separately. If it is not present
at runtime the HAL logs a warning and falls back to the CPU backend.

> **ANGLE access:** ANGLE itself is an open-source Google project, and our
> pre-built, signed + notarized xcframework integration is published from
> the **public** repository
> ([`EdgeFirstAI/angle-package`](https://github.com/EdgeFirstAI/angle-package)).
> Anyone can fetch it — no credentials or organization membership required.
> Two ways to get ANGLE:
>
> - **Recommended (macOS + iOS)** — fetch the pre-built release with
>   `scripts/fetch-angle.sh` (see
>   [Option A](#option-a--edgefirst-pre-built-release-recommended) below).
>   This is exactly what CI uses.
> - **macOS alternative** — install ANGLE via the public Homebrew tap:
>   `brew install startergo/angle/angle` (then re-sign the dylibs — see
>   [Option B — Homebrew tap](#option-b--homebrew-tap-macos-alternative)
>   below). The HAL finds it automatically.
> - **Build without macOS/iOS GL** — the HAL's default features include
>   `opengl`, but you can disable it (`--no-default-features --features
>   static,ndarray,tracing`) to build the CPU-only path, which needs no
>   ANGLE at all. `static` is required alongside `--no-default-features`:
>   `edgefirst-tensor`'s two backend features are mutually exclusive and one
>   is required, so naming neither is a compile error, not a fallback.

### Installing ANGLE (macOS)

The HAL looks for `libEGL.dylib` / `libGLESv2.dylib` via the `EDGEFIRST_ANGLE_PATH`
env var, then standard search paths (Homebrew, `@loader_path`,
`@executable_path`). There are two ways to satisfy this:

#### Option A — EdgeFirst pre-built release (recommended)

Our pre-built, **signed + notarized** xcframeworks (built from a pinned
ANGLE revision) are published in the
[`EdgeFirstAI/angle-package`](https://github.com/EdgeFirstAI/angle-package/releases)
releases. This repo is **public** — anyone can fetch the release with no
credentials. A single helper downloads, sha256-verifies, and extracts them
(into both the xcframework layout for iOS app embedding and a flat-lib
layout for the macOS runtime `dlopen` path):

```bash
scripts/fetch-angle.sh                       # → target/angle/ (default tag v2.1.28252)
EDGEFIRST_ANGLE_PATH=target/angle/macos-flat-lib \
  cargo run --release --example pipeline_demo
```

Because the release is public, `scripts/fetch-angle.sh` needs **no
authentication** — it works out of the box both locally and in CI. (It
still honors `gh auth login` / `GH_TOKEN` / `GITHUB_TOKEN` if present,
which raises GitHub's API rate limit, but none are required.)

> **Why a flat-lib dir for macOS?** ANGLE's `libEGL` internally `dlopen`s
> `libGLESv2.dylib` from its own directory (located via `dladdr`) to
> resolve GL entry points, so the two must be flat siblings. The signed
> framework bundles do not satisfy this, so the helper stages
> `libEGL.dylib` + `libGLESv2.dylib` siblings copied out of the framework
> binaries. Pulling a binary out of its framework invalidates the
> Developer-ID signature (it is scoped to the bundle's `Info.plist`, so
> `dlopen` then fails with *"code signature invalid"*), so
> `scripts/fetch-angle.sh` **ad-hoc re-signs the two flat dylibs for you** —
> you never re-sign manually (unlike the Homebrew path below).

#### Option B — Homebrew tap (macOS alternative)

ANGLE is also available via a public third-party Homebrew tap — an
alternative to Option A if you prefer a package manager on macOS.
Homebrew's `install_name_tool` step invalidates the bundled code signatures
and macOS 26 (Tahoe) refuses to load dylibs with broken signatures at
`dlopen` time (immediate `SIGKILL (Code Signature Invalid)` with no
stdout), so an ad-hoc re-sign is mandatory after each install/upgrade:

```bash
brew install startergo/angle/angle
codesign --force --sign - $(brew --prefix)/opt/angle/lib/libEGL.dylib
codesign --force --sign - $(brew --prefix)/opt/angle/lib/libGLESv2.dylib
```

See [Homebrew/brew#19144](https://github.com/Homebrew/brew/issues/19144)
for the upstream tracking issue. The release path above avoids this
problem entirely.

### Verifying the GPU backend is active

```bash
RUST_LOG=edgefirst_image=debug cargo run --release --example pipeline_demo
```

Look for `ANGLE (Apple, ANGLE Metal Renderer: ...)` in the bring-up log.
If ANGLE is missing or signatures are still broken you will see a
warning and the CPU backend is selected.

### Custom ANGLE locations

If your ANGLE install is not on the default search path, set
`EDGEFIRST_ANGLE_PATH` to the directory containing `libEGL.dylib` and
`libGLESv2.dylib` (flat siblings — see the note above):

```bash
EDGEFIRST_ANGLE_PATH=/path/to/angle/lib cargo run --release ...
```

The lookup order is: `EDGEFIRST_ANGLE_PATH` → Homebrew → `@loader_path`
(alongside the binary) → `@executable_path` → unqualified `libEGL.dylib`
on the dyld search path. For bundled distributions, drop the re-signed
ANGLE dylibs next to the executable (or into `<App>.app/Contents/Frameworks/`)
and no env var is needed.

### When you don't need this setup

- **EdgeFirst-signed binary distribution** — official binary releases
  bundle ANGLE re-signed under the EdgeFirst Apple Developer ID. Install
  and run with no additional setup.
- **Windows wheels and C archives** bundle ANGLE (see
  [Windows GPU Acceleration](#windows-gpu-acceleration)); the macOS wheels
  do not yet — set `EDGEFIRST_ANGLE_PATH` there as above.

These channels exist precisely so end users do not need to deal with the
Homebrew install or re-signing step.

## Windows GPU Acceleration

On Windows the HAL runs the same OpenGL ES engine on
[Google's ANGLE](https://github.com/google/angle) translating to
**Direct3D 11**. There is no zero-copy buffer kind on Windows yet, so GPU
destinations are **PBO tensors** (`TensorMemory::Pbo`, the same path desktop
Linux uses on NVIDIA where DMA-BUF import is unavailable): `Mem` sources are
uploaded by GL and results are read back through `GL_PIXEL_PACK_BUFFER`
and `map()`ped on the GL thread. D3D11 shared-texture tensors (and CUDA via
D3D11 interop) are a planned follow-on. If ANGLE cannot be loaded the HAL
logs a warning and falls back to the CPU backend.

### Installing ANGLE (Windows)

Pre-built `libEGL.dll` + `libGLESv2.dll` (Direct3D 11 backend, static CRT,
no VC++ redistributable needed) are published from the same public
[`EdgeFirstAI/angle-package`](https://github.com/EdgeFirstAI/angle-package/releases)
release tag as the Apple xcframeworks, as `angle-windows-x64-<tag>.zip`.
Fetch, sha256-verify and extract them with the same helper, from Git Bash
(the `--windows` flag is implied on a Windows host):

```bash
bash scripts/fetch-angle.sh                # → target/angle/windows-x64/{bin,lib,include}
```

The HAL looks for `libEGL.dll` in this order, loading it by absolute path
with `LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR` so ANGLE's `libEGL.dll` can find its
sibling `libGLESv2.dll` (the two must stay in the same directory):

1. `%EDGEFIRST_ANGLE_PATH%\libEGL.dll`
2. next to the module that contains the HAL (`edgefirst_image.dll` from the
   C archive, or `_image.pyd` from the wheel) — this is how the bundled
   distributions work with no configuration;
3. next to the executable;
4. `libEGL.dll` on the default DLL search path.

Everything else (`d3d11.dll`, `dxgi.dll`, `d3dcompiler_47.dll`) comes from
System32 on Windows 10/11.

```powershell
$env:EDGEFIRST_ANGLE_PATH = "$PWD\target\angle\windows-x64\bin"
$env:RUST_LOG = 'edgefirst_image=debug'
cargo run --release -p edgefirst-image --example pipeline_demo
```

Look for `ANGLE D3D11 adapter: NVIDIA GeForce RTX 3070 ...` and
`ANGLE (NVIDIA, NVIDIA GeForce RTX 3070 ... Direct3D11 ...)` in the bring-up
log, and `GLConverter created (transfer=Pbo)`. `create_image()` then returns
`TensorMemory::Pbo` destinations.

### Choosing the adapter

`EDGEFIRST_ANGLE_ADAPTER` selects the Direct3D 11 adapter ANGLE creates its
device on (the process-global display is created once, so set it before the
first `ImageProcessor`):

| Value | Meaning |
|-------|---------|
| unset / `hardware` | ANGLE's default hardware adapter (DXGI adapter 0) |
| `warp` | Microsoft Basic Render Driver — software. Classified as a software renderer, so it also needs `EDGEFIRST_ALLOW_SOFTWARE_GL=1` (CI, machines without a GPU) |
| `discrete` | the hardware adapter with the most dedicated video memory (hybrid laptops) |
| `<high>:<low>` | an explicit adapter LUID (decimal or `0x` hex) |
| any other text | case-insensitive substring of the adapter description, e.g. `RTX 3070` or `Intel` |

`RUST_LOG=edgefirst_image=debug` lists every DXGI adapter with its LUID.
Under a Remote Desktop session the hardware adapter is normally still
enumerated; if only the Basic Render Driver is, the HAL warns up front and
falls back to CPU unless `EDGEFIRST_ALLOW_SOFTWARE_GL=1`.

### When you don't need this setup (Windows)

- **`pip install edgefirst-image`** — the Windows wheel bundles `libEGL.dll`
  + `libGLESv2.dll` next to the extension module.
- **The Windows C archive** (`edgefirst-hal-<version>-x86_64-windows.zip`)
  ships them in `bin/` next to `edgefirst_image.dll`, with ANGLE's BSD
  licence under `share/licenses/angle/`.

Both are built by CI with `EDGEFIRST_ANGLE_PATH` set at build time; a local
`maturin build` / `scripts/package-capi.sh` bundles ANGLE the same way when
the variable is set, and produces a CPU-only artifact when it is not.

## iOS

The HAL Rust library closure builds for iOS (arm64 device +
arm64 simulator) with the default features (including `opengl`), reusing
the same ANGLE-over-Metal GL backend as macOS. The supported targets are:

- `aarch64-apple-ios` — iOS devices (arm64)
- `aarch64-apple-ios-sim` — iOS Simulator on Apple-Silicon Macs (arm64)

> **ANGLE note:** iOS GL requires ANGLE xcframeworks. There is **no
> public Homebrew equivalent for iOS** (unlike macOS), so fetch them from
> the **public**
> [`angle-package`](https://github.com/EdgeFirstAI/angle-package) release
> with `scripts/fetch-angle.sh` (no credentials needed). If you would
> rather not fetch ANGLE at all, you can still build the Rust library
> for iOS with the `opengl` feature disabled:
> `cargo build --target aarch64-apple-ios --no-default-features --features static,ndarray,tracing`
> (`static` is required alongside `--no-default-features` — `edgefirst-tensor`'s
> two backend features are mutually exclusive and one is required).
> The Rust `cargo build` itself (with `opengl`) succeeds without ANGLE
> present — see [How the GL backend resolves ANGLE on iOS](#how-the-gl-backend-resolves-angle-on-ios).

> Intel-simulator (`x86_64-apple-ios`) is **not** supported — the
> `angle-package` distribution ships arm64-only slices (see below).

### Prerequisites

Xcode + the iOS SDKs (`xcode-select --install` or a full Xcode), plus the
Rust iOS targets:

```bash
rustup target add aarch64-apple-ios aarch64-apple-ios-sim
```

### Build

This repo's mobile responsibility is the native Rust API compiling (and
linting clean) on the iOS targets — not a C artifact, not Swift bindings,
not app packaging. `mobile-sdk` binds to these crates directly via
[boltffi](https://github.com/EdgeFirstAI) and owns everything above that
line. Build the sibling crates directly:

```bash
cargo build --target aarch64-apple-ios     --release \
  -p edgefirst-tensor -p edgefirst-image -p edgefirst-codec -p edgefirst-decoder -p edgefirst-tracker
cargo build --target aarch64-apple-ios-sim --release \
  -p edgefirst-tensor -p edgefirst-image -p edgefirst-codec -p edgefirst-decoder -p edgefirst-tracker
```

### How the GL backend resolves ANGLE on iOS

ANGLE's EGL/GLES symbols are resolved at **runtime** via `libloading`, not
at link time. On macOS the HAL `dlopen`s `libEGL.dylib` from the release
flat-lib (or Homebrew); on iOS the symbols are already in the process image
(the ANGLE xcframeworks are embedded in the app bundle), so the loader
resolves them via `Library::this()` (equivalent to `dlopen(NULL)`).

Consequence: a standalone `cargo build` for an iOS target succeeds
**without** the ANGLE frameworks present — the Rust staticlib has no
link-time references to `eglInitialize` etc. The frameworks are only
needed at app-link/runtime. The `.cargo/config.toml` iOS entries therefore
carry no rustflags or linker overrides.

### The ANGLE xcframeworks

iOS GL requires shipping [ANGLE](https://github.com/google/angle) as
embedded dynamic frameworks in the app bundle. Our integration uses the
**signed + notarized** xcframeworks from the public
[`EdgeFirstAI/angle-package`](https://github.com/EdgeFirstAI/angle-package/releases)
release (`EGL.xcframework` + `GLESv2.xcframework`, each with `ios-arm64`,
`ios-arm64-simulator`, `macos-arm64`). `scripts/fetch-angle.sh` downloads
and verifies them (default tag `v2.1.28252`, matching the ANGLE
`GL_VERSION` string):

```bash
scripts/fetch-angle.sh       # → target/angle/{EGL,GLESv2}.xcframework
```

A consuming iOS app target embeds them (Xcode "Embed & Sign", or XcodeGen
`embed: true`):

```yaml
dependencies:
  - { framework: ../hal/target/angle/EGL.xcframework,   embed: true }
  - { framework: ../hal/target/angle/GLESv2.xcframework, embed: true }
```

### What is validated vs. deferred

CI's `build-ios` job proves the Rust API — including the iOS-only
`IOSurface` backend and the GL platform seam — compiles and lints clean for
`aarch64-apple-ios` and `aarch64-apple-ios-sim`. It links nothing and
produces no artifact for shipping; that this repo's mobile responsibility
stops there is deliberate, not an oversight.

What is **not** covered here, because it belongs to `mobile-sdk`:

- **boltffi bindings and Swift/Kotlin packaging** — whether the result is
  a monolith `.xcframework` or a modular one is `mobile-sdk`'s call, not
  this repo's.
- **Link-closure validation against the ANGLE xcframeworks + Apple system
  frameworks** (`IOSurface`, `CoreFoundation`, `Metal`) — that is a
  property of whatever `mobile-sdk` ships, not of this crate's `.rlib`.
- **Runtime validation** — actual EGL initialization on a device or
  simulator requires the app shell. The internal `hal-mobile` assessment
  already proved the ANGLE-over-Metal + IOSurface path works on iPhone 17
  Pro (`GL_EXT_color_buffer_half_float` present).

### fp16 / target features

Unlike `aarch64-apple-darwin` (where `+fp16,+dotprod,+i8mm` are baked in
— every M-series chip is ARMv8.6-A+), the iOS targets carry **no**
target-feature rustflags. The iOS 16 deployment floor still includes A11
(iPhone 8, ARMv8.1-A, no fp16/dotprod/i8mm), so enabling them would
SIGILL on older devices. The deployment target matches the
`angle-package` build (`IPHONEOS_DEPLOYMENT_TARGET = 16.0`).

## Android

The HAL builds for Android with the default features (including
`opengl`), using the platform's **native OpenGL ES driver** directly —
unlike macOS/iOS there is no ANGLE translation layer to install, because
Android ships a first-class GLES implementation (Adreno, Mali, etc.).
Zero-copy buffer interchange uses
[AHardwareBuffer](https://developer.android.com/ndk/reference/group/a-hardware-buffer)
(the role DMA-BUF plays on Linux and IOSurface on Apple platforms),
imported into GL via `EGL_ANDROID_image_native_buffer`. The supported
targets are:

- `aarch64-linux-android` — Android devices (arm64-v8a)
- `x86_64-linux-android` — the Android emulator on x86_64 hosts

The minimum supported API level is **26** (Android 8.0) — the floor of
the stable AHardwareBuffer NDK ABI.

### Prerequisites

```bash
rustup target add aarch64-linux-android x86_64-linux-android
cargo install cargo-ndk
# Android NDK r26+ (r27c LTS recommended); set ANDROID_NDK_HOME or let
# cargo-ndk auto-detect it under your Android SDK.
```

### Building

This repo's mobile responsibility is the native Rust API compiling (and
linting clean) on the Android targets — not a C artifact, not Kotlin
bindings, not app packaging. `mobile-sdk` binds to these crates directly
via [boltffi](https://github.com/EdgeFirstAI) and owns everything above
that line. Build the sibling crates directly, both ABIs at API 26:

```bash
cargo ndk -t arm64-v8a -t x86_64 -P 26 build --release \
  -p edgefirst-tensor -p edgefirst-image -p edgefirst-codec -p edgefirst-decoder -p edgefirst-tracker
```

### What is validated vs. deferred

CI's `build-android` job proves the Rust API — including the
Android-only `AHardwareBuffer` backend and the GL platform seam —
compiles and lints clean for `aarch64-linux-android` and
`x86_64-linux-android`. It links nothing and produces no artifact for
shipping.

What is **not** covered here, because it belongs elsewhere:

- **boltffi bindings and Kotlin packaging** — `mobile-sdk`'s
  responsibility, not this repo's.
- **Link-closure validation against the NDK system libraries** — a
  property of whatever `mobile-sdk` ships, not of this crate's `.rlib`.
- **Runtime validation** — on-device GL correctness and performance run
  via the internal `hal-mobile` AWS Device Farm harness, which drives
  the real `ImageProcessor` through JNI (see TESTING.md § Android
  On-Device Validation);
  the Phase-1 assessment already proved the native-GLES + AHardwareBuffer
  path on a Galaxy S26 Ultra (`GL_EXT_color_buffer_half_float` present,
  letterbox 720p→640×640 F16 in 741 µs).
- **Deferred zero-copy paths** — YUV camera buffers (external-OES
  sampling) and single-channel Grey/NV imports (`R8_UNORM` needs
  API 29); these fall back to CPU conversion today.

### NPU-direct output (zero CPU readback)

The convert destination is a real AHardwareBuffer, so an NPU runtime can
consume it directly — no `map()`, no CPU readback:

```c
// Allocate once, reuse every frame (Rule 1). F16 NCHW model input;
// auto-select yields an AHardwareBuffer when the GL backend is active —
// assert ef_tensor_storage_kind(dst) == EF_STORAGE_KIND_DMA_BUF at startup.
ef_tensor *dst = ef_image_processor_create_image(
    proc, 640, 640, "rgb8_planar", EF_DTYPE_F16, EF_STORAGE_KIND_DMA_BUF, EF_CPU_ACCESS_NONE);

// One-time: hand the SAME buffer to the NPU runtime.
AHardwareBuffer* ahb = ef_tensor_hardware_buffer_ptr(dst);
ANeuralNetworksMemory* mem;
ANeuralNetworksMemory_createFromAHardwareBuffer(ahb, &mem);   // NNAPI
// (LiteRT: wrap `ahb` via TfLiteAHardwareBufferAttachment instead.)

// Per frame: when convert() returns, the GPU has finished writing and
// the handle contents are safe to execute against.
ef_image_processor_convert(proc, src, dst, 0 /* rotation none */,
                            0 /* flip none */, &letterbox);
// ... ANeuralNetworksExecution_setInputFromMemory(exec, 0, NULL, mem, 0, bytes);
```

For a pipelined handoff that skips the blocking GPU sync entirely, use
`ef_image_processor_convert_fence()`: it returns a sync-fence fd
(`EGL_ANDROID_native_fence_sync`) the NPU runtime waits on instead
(`ANeuralNetworksExecution_startComputeWithDependencies`), or `-1` with
the work already synced on drivers without fence support.

**Flatness**: gralloc chooses the row pitch and may pad it (observed on
the S26 Ultra: 640-px planar F16 → 1536-byte rows, natural 1280). Check
`ef_tensor_row_stride(dst)`:

- `0` — the buffer IS the flat `[1, C, H, W]` stream; hand it off as-is.
- nonzero — describe the pitch to the runtime, pick a width whose pitch
  the device does not pad, or fall back to
  `ef_tensor_copy_to(dst, buf, len)` (~0.3 ms at 2.4 MB — still
  cheaper than a full CPU convert, but no longer zero-copy; profile).

**INT8 NPUs**: allocate the destination as `"rgb8"` /
`"rgba8"` with `EF_DTYPE_U8` or `EF_DTYPE_I8` (NHWC,
zero-copy on Android via the RGBA8888 texel packing) and attach the
model's quantization so consumers agree on the scale:

```c
float scale = 1.0f / 255.0f;
int32_t zp = 0;
ef_tensor_quantization_set(dst, /*axis=*/-1, &scale, &zp, 1);
```

The I8 path applies the `^0x80` bias in-shader during the convert — the
buffer bytes are already signed model input.

**Tile compression** (bandwidth): a hardware-only destination can
additionally request the device's vendor tile layout through the
image-descriptor path — the GPU renders into it and Qualcomm's QNN can
consume it natively (UBWC data formats declared at context-binary
preparation); other NPU stacks take the linear default:

```c
ef_tensor_image_desc *desc = ef_tensor_image_desc_new(640, 640, "rgba8", EF_DTYPE_U8);
ef_tensor_image_desc_set_compression(desc, 1);   // 1 = any scheme; linear fallback is counted
ef_tensor *dst = ef_image_processor_create_image_desc(proc, desc);
ef_tensor_image_desc_free(desc);
// ef_tensor_compression(dst) records the scheme actually allocated
// (EF_COMPRESSION_UBWC on Adreno, EF_COMPRESSION_NONE = linear fallback — see
// ef_compression_fallback_count()).
```

Compression requires `EF_CPU_ACCESS_NONE` (CPU mapping pins the layout
linear) and a compressed tensor has no meaningful linear row stride —
it is a hardware-to-hardware handle only.

**macOS parity**: the same pattern works with
`ef_tensor_iosurface_ref()` — wrap the IOSurface in a `CVPixelBuffer`
(`CVPixelBufferCreateWithIOSurface`) for CoreML/ANE input; `convert()`
returning likewise guarantees GPU completion.

### fp16 / target features

Like iOS, the Android targets carry **no** target-feature rustflags: the
API-26 device floor spans ARMv8.0-A cores (Cortex-A53 class, no
fp16/dotprod/i8mm), so baking those features in would SIGILL on real
older hardware (see `.cargo/config.toml`).

## Build System

The workspace builds with standard `cargo`. The
[`Makefile`](https://github.com/EdgeFirstAI/hal/blob/main/Makefile) wraps
the common workflows (`make test`, `make bench`, `make build`,
`make format lint check`) with the right flags and gates.

For Python wheels, see
[`crates/python-tensor/README.md`](https://github.com/EdgeFirstAI/hal/blob/main/crates/python-tensor/README.md)
and
[`crates/python-common/TESTING.md`](https://github.com/EdgeFirstAI/hal/blob/main/crates/python-common/TESTING.md).
For the C libraries and consumer linking, see
[`crates/tensor-capi/README.md`](https://github.com/EdgeFirstAI/hal/blob/main/crates/tensor-capi/README.md)
and the sibling `*-capi` READMEs.

## Environment Variables

| Variable | Description |
|----------|-------------|
| `EDGEFIRST_TENSOR_FORCE_MEM` | `1` forces heap memory (disables DMA / SHM) |
| `EDGEFIRST_DISABLE_G2D` | Disable G2D backend |
| `EDGEFIRST_DISABLE_GL` | Disable OpenGL backend |
| `EDGEFIRST_DISABLE_CPU` | Disable CPU backend |
| `EDGEFIRST_FORCE_BACKEND` | Force one backend: `cpu`, `g2d`, or `opengl` (disables fallback) |
| `EDGEFIRST_FORCE_TRANSFER` | Force GL transfer: `pbo`, `dmabuf`, or `sync` |
| `EDGEFIRST_NV_CONVERT_PATH` | NV12/16/24 GPU conversion path: `sampler`, `shader`, or `auto` (default). `auto` prefers the portable, colorimetry-exact in-shader `ShaderR8`, except BT.601-limited single-plane NV12 on Vivante (hardware sampler is ~12× faster and correct). `sampler`/`shader` force a path for benchmarking/bring-up |
| `EDGEFIRST_COLORIMETRY` | `fast` (default) or `exact`. High-performance colour conversion is the default; `exact` opts into the colorimetry-exact path where it costs more. Takes precedence over the per-processor setting |
| `EDGEFIRST_GL_SERIALIZE` | `full` or `lifecycle` — pin the GL command serialization policy instead of using the per-driver default (see [Rule 5](#rule-5--one-imageprocessor-per-pipeline)) |
| `EDGEFIRST_ENABLE_NVJPEG` | `1` opts into the nvJPEG GPU JPEG decoder on CUDA hosts (off by default so it never silently contends with the inference engine) |
| `EDGEFIRST_EGL_CACHE_CAPACITY` | Override the per-cache EGLImage capacity (default 64) for high-cardinality varied-geometry streams |
| `EDGEFIRST_ALLOW_SOFTWARE_GL` | `1` opts in to running the GL backend on a software renderer (otherwise rejected); for CI / headless bring-up |
| `EDGEFIRST_OPENGL_RENDERSURFACE` | `1` enables EGL renderbuffer path for non-`dma_heap` DMA-BUF (i.MX 95 Neutron NPU) |
| `EDGEFIRST_PROTO_COMPUTE` | `1` enables GLES 3.1 compute shader for HWC→CHW proto repack |
| `EDGEFIRST_DISABLE_V4L2` | `1` forces the software JPEG decoder, bypassing the V4L2 hardware JPEG backend (Linux) |
| `EDGEFIRST_CODEC_V4L2_DEVICE` | Probe a specific V4L2 device node for hardware JPEG decode instead of auto-discovery |
| `EDGEFIRST_ANGLE_PATH` | macOS and Windows: directory containing `libEGL.dylib` / `libGLESv2.dylib` (macOS) or `libEGL.dll` / `libGLESv2.dll` (Windows). Overrides the default search (macOS: Homebrew → `@loader_path` → `@executable_path` → `libEGL.dylib` on dyld; Windows: next to the loading module → next to the executable → the default DLL search path). Set this when deploying a bundled or custom-signed ANGLE alongside the binary. |
| `EDGEFIRST_ANGLE_ADAPTER` | Windows only: which Direct3D 11 adapter ANGLE uses — `hardware` (default), `warp` (software; needs `EDGEFIRST_ALLOW_SOFTWARE_GL=1`), `discrete`, an adapter LUID `<high>:<low>`, or a substring of the adapter description (see [Windows GPU Acceleration](#windows-gpu-acceleration)) |
| `EDGEFIRST_TESTDATA_DIR` | Override testdata location (used by benches and CI) |
| `RUST_LOG` | Standard `env_logger` filter — `RUST_LOG=edgefirst_image=debug` for backend dispatch + cache stats |

Per-crate variables and additional detail live in each crate's README.

## Testing

See [TESTING.md](https://github.com/EdgeFirstAI/hal/blob/main/TESTING.md)
for the cross-cutting testing guide (single-threaded rule, on-target
gating, cross-compilation, CI matrix, optimization validation). Per-crate
testing detail lives in each crate's `TESTING.md` — links in the
[Core Components](#core-components) table.

## Benchmarking

| Binary | Crate | What it measures |
|--------|-------|------------------|
| `tensor_benchmark` | `edgefirst-tensor` | Tensor allocation and map/unmap latency across buffer types |
| `codec_benchmark` | `edgefirst-codec` | Strided decode into pre-allocated tensors vs. the `image` crate and raw `zune-png` |
| `image_benchmark` | `edgefirst-image` | Crop, flip, rotate, resize, draw |
| `pipeline_benchmark` | `edgefirst-image` | Letterbox pipeline + format conversion |
| `convert_matrix_benchmark` | `edgefirst-image` | Full src/dst memory × format × dtype GL convert matrix |
| `batch_convert_benchmark` | `edgefirst-image` | Batched `convert_deferred` + `flush` vs. eager per-tile convert |
| `tiled_convert_benchmark` | `edgefirst-image` | Crop contract: per-convert CPU cost scales with tile area, not source area |
| `decode_pipeline_benchmark` | `edgefirst-image` | JPEG decode → letterbox convert (strided, HWC/CHW) |
| `nv_path_benchmark` | `edgefirst-image` | NV12/16/24 `ExternalSampler` vs. `ShaderR8` conversion paths |
| `cpu_preprocess_benchmark` | `edgefirst-image` | CPU-only JPEG decode + preprocess path (for targets that reserve the GPU for inference) |
| `parallel_processors_benchmark` | `edgefirst-image` | Aggregate convert throughput with 1 / 2 / 4 concurrent `ImageProcessor` instances |
| `mask_benchmark` | `edgefirst-image` | `draw_decoded_masks`, `draw_proto_masks`, hybrid path |
| `mask_decode_benchmark` | `edgefirst-image` | `materialize_scaled_segmentations` — the COCO-eval scaled-mask path |
| `nvjpeg_benchmark` | `edgefirst-image` | nvJPEG GPU decode into a CUDA-registered PBO (Jetson / CUDA targets) |
| `opencv_benchmark` | `edgefirst-image` | OpenCV baseline comparison |
| `decoder_benchmark` | `edgefirst-decoder` | YOLO post-processing, NMS, dequant |
| `tracker_benchmark` | `edgefirst-tracker` | ByteTrack throughput vs. simultaneous tracks |

Run on host:

```bash
cargo bench -p edgefirst-image --bench pipeline_benchmark -- --bench

# Force a backend
EDGEFIRST_FORCE_BACKEND=cpu cargo bench -p edgefirst-image --bench pipeline_benchmark -- --bench
```

Cross-compile + deploy to a target (SSH hostnames in `~/.ssh/config`:
`imx8mp-frdm`, `imx95-frdm`, `rpi5-hailo`, `jetson-orin-nano`,
`maivin`):

```bash
cargo-zigbuild zigbuild --target aarch64-unknown-linux-gnu --release \
  -p edgefirst-image --features opengl --bench pipeline_benchmark

scp target/aarch64-unknown-linux-gnu/release/deps/pipeline_benchmark-* imx8mp-frdm:/tmp/
ssh imx8mp-frdm '/tmp/pipeline_benchmark-* --bench --json /tmp/pipeline.json'
```

All benchmarks accept `--bench --json <path>` for structured output.
Store results under `benchmarks/<platform>/<name>.json`. Update
[BENCHMARKS.md](https://github.com/EdgeFirstAI/hal/blob/main/BENCHMARKS.md)
via:

```bash
python3 .github/scripts/generate_benchmark_tables.py --data-dir benchmarks/
```

## Performance Tracing

The HAL captures performance traces across every processing stage. Traces
are written in the Chrome JSON format and open directly in the
[Perfetto UI](https://ui.perfetto.dev/).

### How it works

Every HAL library crate emits `tracing` spans on hot paths. Those spans cost
close to nothing when no subscriber is active: each site compiles to a single
relaxed atomic load, with no heap allocations, no string formatting, and no
function calls on the hot path.

When a session is started via the API, a Chrome JSON subscriber records
all span enter/exit events with high-resolution timestamps and structured
metadata (detection counts, proto dimensions, format conversions, memory
types, etc.) to a file.

### Span coverage

The tracing surface covers decode, image conversion, GL multi-pass, mask
materialization, tensor lifecycle, tracker association, and the Python
entry points. Each span carries structured fields — see the per-crate
ARCHITECTURE.md files for the authoritative list of spans and fields per
component.

### Enabling tracing

Python:

```python
from edgefirst.tensor import Tracing
with Tracing("/tmp/trace.json"):
    # ... run inference pipeline ...
    pass
```

Rust:

```rust
use edgefirst_tensor::trace::{start_tracing, stop_tracing};

start_tracing("/tmp/trace.json").expect("start tracing");
// ... inference pipeline ...
stop_tracing(); // flushes and closes the trace file
```

C:

```c
#include <edgefirst/tensor.h>
ef_start_tracing("/tmp/trace.json");
/* ... inference pipeline ... */
ef_stop_tracing();
```

### Viewing traces

1. Open <https://ui.perfetto.dev/>
2. Drag the generated `.json` file onto the page
3. Click slices to see structured fields in the *Current Selection* panel

### Using traces for optimization

The tracing infrastructure complements the rules in the
[Optimization Guide](#optimization-guide) and the data in
[BENCHMARKS.md](https://github.com/EdgeFirstAI/hal/blob/main/BENCHMARKS.md):

1. **Identify bottlenecks** — common findings:
   - `decoder.decode_proto.extract_proto_data > 3 ms` → model emits NCHW protos but HAL is transposing (check the `layout` field)
   - `image.convert.cpu.format_convert` appearing twice → intermediate format conversion (consider matching src/dst formats)
   - `tensor.alloc` per-frame → tensors not being reused (Rule 1)
   - `image.convert.gl.egl_import` on every frame → camera tensors re-imported instead of cached (Rule 3)
2. **Validate rules** — re-run with tracing after applying a rule to confirm the expected spans disappear or shrink.
3. **Cross-reference with `perf`** — for CPU-bound spans, combine trace data with `perf record` for instruction-level hotspots.

### Limitations

- Only one trace session per process lifetime (Rust global subscriber model).
- Rayon worker spans are not automatically parented to the calling span.
- The `log::*` output (via `env_logger` / C callback logger) operates independently from trace capture; both can be active simultaneously.

## Dependencies

### Key external dependencies

- [PyO3](https://pyo3.rs) — Python bindings
- [ndarray](https://docs.rs/ndarray) — N-dimensional arrays
- [rayon](https://docs.rs/rayon) — Data parallelism
- [fast_image_resize](https://docs.rs/fast_image_resize) — CPU image operations
- [zune-png](https://docs.rs/zune-png) — PNG image decoding (JPEG uses custom decoder)
- [dma-heap](https://docs.rs/dma-heap) — Linux DMA allocation
- [nix](https://docs.rs/nix) — Unix system calls

### Internal dependency graph

```mermaid
graph TD
    Tensor[edgefirst-tensor]
    Codec[edgefirst-codec]
    Image[edgefirst-image]
    Decoder[edgefirst-decoder]
    Tracker[edgefirst-tracker<br/>optional]
    G2D[g2d-sys<br/>optional]

    Image --> Tensor
    Image --> Decoder
    Image --> Codec
    Image -.optional.-> G2D
    Image -.->|tracker feature| Tracker
    Decoder --> Tensor
    Decoder -.->|tracker feature| Tracker
    Codec --> Tensor

    Python[edgefirst.{tensor,codec,image,decoder,tracker}<br/>PyO3]
    TensorC[libedgefirst_tensor]
    ImageC[libedgefirst_image]
    CodecC[libedgefirst_codec]
    DecoderC[libedgefirst_decoder]
    TrackerC[libedgefirst_tracker]

    Python --> Tensor
    Python --> Image
    Python --> Decoder
    TensorC --> Tensor
    ImageC --> Image
    CodecC --> Codec
    DecoderC --> Decoder
    TrackerC --> Tracker

    style Python fill:#e1f5ff
    style TensorC fill:#e1f5ff
    style ImageC fill:#e1f5ff
    style CodecC fill:#e1f5ff
    style DecoderC fill:#e1f5ff
    style TrackerC fill:#e1f5ff
    style Tracker fill:#e8f5e9
```

## Future Considerations

1. **Model HAL** — planned abstraction for inference engines (ONNX, TFLite, Kinara)
2. **VPI integration** — support for NVIDIA Vision Programming Interface
3. **Additional trackers** — SORT, Deep SORT
4. **Async I/O** — non-blocking image loading and processing

## Support

### Community resources

- [GitHub Discussions](https://github.com/EdgeFirstAI/hal/discussions) — questions and ideas
- [Issue Tracker](https://github.com/EdgeFirstAI/hal/issues) — bug reports and feature requests

### EdgeFirst ecosystem

This project is part of the EdgeFirst Perception stack:

- [**EdgeFirst Studio**](https://edgefirst.studio?utm_source=github&utm_medium=readme&utm_campaign=hal) — complete MLOps platform
- [**EdgeFirst Hardware Platforms**](https://au-zone.com/hardware?utm_source=github&utm_medium=readme&utm_campaign=hal) — NPU/GPU acceleration on NXP i.MX

### Professional services

Au-Zone Technologies supports production deployments with training and
workshops, custom development, integration services, enterprise SLAs, and
hardware reference designs.

Contact: <support@au-zone.com> · [au-zone.com](https://au-zone.com?utm_source=github&utm_medium=readme&utm_campaign=hal)

## Contributing

Contributions are welcome. See
[CONTRIBUTING.md](https://github.com/EdgeFirstAI/hal/blob/main/CONTRIBUTING.md)
for development setup and guidelines. This project follows our
[Code of Conduct](https://github.com/EdgeFirstAI/hal/blob/main/CODE_OF_CONDUCT.md).

## Security

For security vulnerabilities, see
[SECURITY.md](https://github.com/EdgeFirstAI/hal/blob/main/SECURITY.md)
or email <support@au-zone.com> with subject "Security Vulnerability".

## Documentation

- [ARCHITECTURE.md](https://github.com/EdgeFirstAI/hal/blob/main/ARCHITECTURE.md) — cross-crate architecture story
- [TESTING.md](https://github.com/EdgeFirstAI/hal/blob/main/TESTING.md) — workspace testing rules and CI matrix
- [BENCHMARKS.md](https://github.com/EdgeFirstAI/hal/blob/main/BENCHMARKS.md) — empirical performance reference
- [CHANGELOG.md](https://github.com/EdgeFirstAI/hal/blob/main/CHANGELOG.md) — release history
- Per-crate docs (README + ARCHITECTURE + TESTING) — see [Core Components](#core-components) table

## License

Apache License 2.0 — see [LICENSE](https://github.com/EdgeFirstAI/hal/blob/main/LICENSE) for details.

Copyright 2025-2026 Au-Zone Technologies
