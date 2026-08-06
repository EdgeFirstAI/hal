# edgefirst-hal

[![PyPI](https://img.shields.io/pypi/v/edgefirst-hal.svg)](https://pypi.org/project/edgefirst-hal/)
[![Python](https://img.shields.io/pypi/pyversions/edgefirst-hal.svg)](https://pypi.org/project/edgefirst-hal/)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

Hardware-accelerated image processing, zero-copy tensors, and YOLO decoding
for edge AI inference pipelines. Built in Rust with Python bindings via PyO3.

## Installation

```bash
pip install edgefirst-hal
```

Pre-built wheels are available for Linux (x86_64, aarch64), macOS, and Windows.
No Rust toolchain required.

> **Python 3.11+** wheels use the improved stable ABI for zero-copy buffer
> protocol support. Python 3.8–3.10 wheels use a compatible fallback.
> Pip selects the best wheel automatically.

## Quick Start

```python
import numpy as np
import edgefirst_hal as ef

# Read the JPEG header, size a tensor to match, then decode into it.
# Decoding is always into a tensor you already own, so a real pipeline
# allocates once and reuses the tensor every frame.
info = ef.Tensor.peek_image_info_file("photo.jpg")
src = ef.Tensor.image(info.width, info.height, info.format, access="readwrite")
src.decode_image_file("photo.jpg")

# Create an image processor (auto-selects best backend: GPU > G2D > CPU)
processor = ef.ImageProcessor()

# Allocate a GPU-optimal output buffer — always use create_image() for
# destinations passed to convert(), so the processor can select the best
# memory type (DMA-buf, IOSurface, PBO, or system memory) for zero-copy
# GPU paths. access declares CPU involvement (hardware access is
# implied): this script reads the pixels below, so declare "readwrite".
# Hardware-only pipelines keep the strict default access="none".
dst = processor.create_image(640, 640, ef.PixelFormat.Rgb, access="readwrite")

# Convert with a letterbox resize (preserves aspect ratio, pads with grey).
# Omit `letterbox=` to stretch-to-fill instead.
processor.convert(src, dst, letterbox=[114, 114, 114, 255])

# Access pixel data as a numpy array. Use the context manager + .numpy()
# form — this is the portable pattern that works on both wheel variants.
# `shape` is a property, not a method.
with dst.map() as m:
    pixels = np.frombuffer(m.numpy(), dtype=np.uint8).reshape(dst.shape)

# The shorter `np.frombuffer(dst.map(), ...)` form only works on the
# abi3-py311 wheel, where `TensorMap` exposes Python's buffer protocol
# directly. The abi3-py38 compatibility wheel disables `__getbuffer__`,
# so use `.numpy()` if your code needs to run on Python 3.8–3.10.
```

JPEG decodes to `Nv12` and PNG to `Rgb`/`Rgba`/`Grey`, so `info.format`
above is the source's native format, not RGB. The `convert()` call is what
gets you to the format your model wants.

## Role in edgefirst-hal

The `edgefirst-hal` package on PyPI is the Python face of the EdgeFirst
HAL Rust workspace:

- Built from [`crates/python`](https://github.com/EdgeFirstAI/hal/tree/main/crates/python),
  which is a PyO3 binding over the `edgefirst-hal` Rust umbrella crate.
- Does **not** consume the C API ([`edgefirst-hal-capi`](https://github.com/EdgeFirstAI/hal/blob/main/crates/capi/));
  the binding goes directly through Rust.
- Exposes the same `Tensor`, `ImageProcessor`, `Decoder`, and `ByteTrack`
  surfaces as the Rust crate, with numpy-friendly conversions and the
  buffer protocol for zero-copy interop.
- Wheels are distributed as two stable-ABI variants per platform —
  `abi3-py311` (preferred, supports buffer protocol features added in
  3.11) and `abi3-py38` (compatibility fallback for 3.8–3.10).
  Pip selects the best wheel automatically.

## Key Features

- **Zero-copy tensors** — DMA-BUF (Linux), IOSurface (macOS), POSIX shared
  memory, and PBO-backed buffers with automatic fallback to system memory
- **Hardware-accelerated image processing** — OpenGL, NXP G2D, and
  optimized CPU backends with automatic selection
- **Letterbox resize** — aspect-ratio-preserving resize with configurable
  padding color, rotation, and flip
- **Int8 output** — `create_image(..., dtype="int8")` for direct signed
  int8 tensor output with GPU-accelerated XOR bias
- **Declared CPU access** — `create_image(..., access="readwrite")` for
  scripts that touch pixels (map/numpy); the strict `"none"` default
  keeps hardware pipelines eligible for vendor tile compression
  (`compression="any"` on Android, outcome on `Tensor.compression`)
- **YOLO decoding** — YOLOv5, YOLOv8, YOLO11, and YOLO26 detection and
  instance segmentation (including end-to-end models)
- **Object tracking** — ByteTrack multi-object tracker with Kalman filtering
- **Tiled inference** — SAHI-style overlapping tile grids with GPU cropping
  and an intersection-over-smaller merge for objects split across seams
- **Fully typed** — ships with `.pyi` stubs for IDE autocompletion and
  type checking with mypy / pyright

## Image Processing

```python
import edgefirst_hal as ef

processor = ef.ImageProcessor()

info = ef.Tensor.peek_image_info_file("frame.jpg")
src = ef.Tensor.image(info.width, info.height, info.format, access="readwrite")
src.decode_image_file("frame.jpg")

# Stretch to model input size
dst = processor.create_image(640, 640, ef.PixelFormat.Rgb)
processor.convert(src, dst)

# Letterbox instead: preserve aspect ratio, pad with the given RGBA colour
processor.convert(src, dst, letterbox=[114, 114, 114, 255])

# With rotation and horizontal flip
processor.convert(
    src, dst, rotation=ef.Rotation.Clockwise90, flip=ef.Flip.Horizontal
)

# Crop a source region — Region(x, y, width, height) in source pixels
processor.convert(src, dst, source=ef.Region(100, 100, 400, 400))

# Int8 output for quantized models
dst_i8 = processor.create_image(640, 640, ef.PixelFormat.Rgb, dtype="int8")
processor.convert(src, dst_i8)
```

## Zero-Copy External Buffer (Linux)

When integrating with an NPU delegate that owns DMA-BUF buffers, render
directly into the delegate's buffer to eliminate a `memcpy`:

```python
import edgefirst_hal as ef

processor = ef.ImageProcessor()

info = ef.Tensor.peek_image_info_file("frame.jpg")
src = ef.Tensor.image(info.width, info.height, info.format, access="readwrite")
src.decode_image_file("frame.jpg")

# Render directly into the delegate's DMA-BUF — zero copies
dst = processor.import_image(fd=vx_fd, width=640, height=640, format=ef.PixelFormat.Rgb)
processor.convert(src, dst)

# Reverse: HAL allocates, consumer imports the fd
hal_dst = processor.create_image(640, 640, ef.PixelFormat.Rgb)
fd = hal_dst.dmabuf_clone()  # Raises if not DMA-backed
delegate.register(fd)
```

You can also attach format metadata to any raw tensor created via `from_fd()`:

```python
t = ef.Tensor.from_fd(some_fd, [480, 640, 3])
t.set_format(ef.PixelFormat.Rgb)
processor.convert(src, t)
```

**Performance tip:** When rotating through a pool of DMA-BUFs (e.g. 2-3
from an NPU delegate), create the `Tensor` wrappers once at init and
reuse them across frames. This avoids EGL image cache misses (~100-300us
each on Vivante GPUs).

## CUDA Zero-Copy (TensorRT)

When running inference with TensorRT or cupy, `Tensor.cuda_map()` exposes a
raw CUDA device pointer to a tensor that has been registered with CUDA (e.g.
via the GL-CUDA interop path). The mapping is scoped by a context manager so
the GPU buffer is released automatically for the next `convert()` call.

Check availability first, then try `cuda_map()` and fall back to `map()` for
CPU paths:

```python
import edgefirst_hal as ef

# One-time check — cached after first call
if not ef.is_cuda_available():
    print("libcudart not found; falling back to CPU tensors")

tensor = ef.ImageProcessor().create_image(640, 640, ef.PixelFormat.Rgb)

cm = tensor.cuda_map()
if cm is not None:
    with cm as m:
        # m.device_ptr is the raw CUDA device pointer (int)
        # m.size is the buffer size in bytes
        trt_context.set_input_tensor_address("input", m.device_ptr)
        trt_context.execute_async_v3(stream)
else:
    # No CUDA handle on this tensor — use the CPU path
    with tensor.map() as host:
        run_cpu_inference(host)
```

`CudaMap` exposes:
- `device_ptr` (`int`) — raw CUDA device pointer, suitable for
  `cupy.ndarray.from_dlpack`, `pycuda.gpuarray`, or TensorRT
  `set_input_tensor_address`.
- `size` (`int`) — buffer size in bytes.
- `release()` — explicitly release before the `with` block ends (idempotent).

## NumPy Interop

Reading a tensor goes through `map()`, which returns a context manager. The
mapping holds driver state for DMA and PBO buffers, so it has to be released
deterministically. Keep it inside the `with` block:

```python
import numpy as np

with tensor.map() as m:
    arr = np.frombuffer(m.numpy(), dtype=np.uint8).reshape(tensor.shape)
    # `arr` is a view into the tensor for MEM and DMA backends; PBO does a
    # glMapBufferRange round-trip on the GL thread.
```

Writing goes the other way through `from_numpy()`. The element count and
dtype must match the tensor exactly; a mismatch raises `RuntimeError` rather
than silently truncating or converting.

Pass your array in as-is. `from_numpy()` inspects the source strides and
picks one of three copy strategies, so a manual `np.ascontiguousarray()`
above HAL just duplicates work the binding already does:

| Source layout | What happens | Cost |
|---------------|--------------|------|
| Fully contiguous | One `memcpy`, parallelized above 256 KiB | Lower bound |
| Strided outer, contiguous inner rows (column slice, sub-volume, negative stride) | Per-row `memcpy` over the outer dimensions | Within about 5% of the contiguous case |
| Fully strided (transposed view, every-other-element) | `np.ascontiguousarray()` internally, then the contiguous `memcpy` | Roughly 4x the contiguous case |

That third row is the one that used to bite people. A HailoRT output arrives
as `arr.transpose(0, 2, 1)` over a `(1, anchors, channels)` buffer, which has
no contiguous inner row at all. The old element-wise loop measured 27 ms per
call on a `(1, 116, 8400)` float32 view on rpi5-hailo, against 6.5 ms once
the copy was materialized in vectorized C.

There is a fourth case worth knowing about: when the destination came from
`create_image()` on a DMA or PBO backend, its rows are padded up to the GPU's
pitch alignment, so the mapped buffer is larger than the logical element
count. `from_numpy()` detects that and copies row by row, skipping the
padding. You don't have to do anything, but it explains why the mapped
`memoryview` can be bigger than `shape` suggests.

## Tiled Inference (SAHI)

Small objects in a large frame survive better if you run the model over
overlapping tiles and merge the results. The tiling API covers both halves:
cutting the frame up on the GPU, and stitching the detections back together.

```python
import edgefirst_hal as ef

processor = ef.ImageProcessor()
cfg = ef.TilingConfig(640, 640, overlap=0.2)

# Plan once per frame size — this does not touch the GPU
placements = processor.plan_tiles(frame_w, frame_h, cfg)

# One model-input sized slot, reused per tile
slot = processor.create_image(640, 640, ef.PixelFormat.Rgb, access="readwrite")

acc = ef.TiledFrameAccumulator(
    (float(frame_w), float(frame_h)),
    len(placements),
    ef.MergeConfig(metric=ef.MatchMetric.Ios, threshold=0.5),
)

for placement in placements:
    processor.tile_one(frame, slot, placement, cfg)
    processor.flush()  # tile_one is deferred; flush on your own cadence
    boxes, scores, classes, _ = decoder.decode([run_inference(slot)])
    acc.push_tile(boxes, scores, classes, placement)

boxes, scores, classes = acc.finalize_normalized()
```

`push_tile` expects tile-local boxes normalized to `[0, 1]` over the model
input, which is what `decode()` gives you when `decoder.normalized_boxes` is
`True`. If your decoder emits pixel coordinates, divide by the model input
size before pushing.

`tile_one` is deferred so you can overlap GPU cropping with inference: issue
several tiles, then `flush()` once, rather than flushing per tile as the loop
above does for clarity. If you'd rather render every tile in one
shot, `alloc_tile_batch(n, cfg)` gives you a tall `tile_w x n*tile_h` parent
and `tile_into(src, batch, cfg)` fills it with a single flush; address the
individual slots with `batch.view(ef.Region(0, i * 640, 640, 640))`.

The merge defaults to intersection-over-smaller rather than IoU. An object
split across a tile seam produces one fragment and one near-whole box; their
IoU is low enough that NMS keeps both as duplicates, while IoS is high
enough to merge them. Pass `metric=ef.MatchMetric.Iou` for plain NMS
semantics.

`push_tile` is idempotent per `placement.index`, so a retried tile won't
double-count, and it returns `False` when it ignores one. `finalize()`
returns full-frame pixel boxes, `finalize_normalized()` returns `[0, 1]`
boxes matching the non-tiled decode contract, and both consume the
accumulator, so calling either twice raises `RuntimeError`. To drive the
pieces yourself, `tile_grid()`, `lift_tile_boxes()`, and
`merge_tiled_detections()` are the same steps as free functions.

One porting trap: Python's `tile_grid(frame_w, frame_h, tile_w, tile_h, ...)`
is width-first, but the Rust and C equivalents are height-first
(`tile_grid(frame_h, frame_w, tile_h, tile_w, ...)`). Double-check the order
when translating code between bindings.

## YOLO Decoding

Describe the model's outputs, build a decoder, and feed it the raw output
tensors:

```python
import edgefirst_hal as ef

# One combined detection output, e.g. YOLOv8n at 640x640 with 80 classes
decoder = ef.Decoder.new_from_outputs(
    [ef.Output.detection(shape=[1, 84, 8400], decoder=ef.DecoderType.Ultralytics)],
    score_threshold=0.25,
    iou_threshold=0.45,
)

# decode() always returns four values. `masks` is an empty list for
# detection-only models.
boxes, scores, classes, masks = decoder.decode([output_tensor])
```

`boxes` is an `(N, 4)` float32 array of `[xmin, ymin, xmax, ymax]`, `scores`
is `(N,)` float32, and `classes` is `(N,)` unsigned integer indices. Check
`decoder.normalized_boxes` before scaling: `True` means the boxes are
already in `[0, 1]`, `False` means pixel coordinates relative to
`decoder.input_dims`, and `None` means the schema didn't say.

If your model ships a metadata file, pass it straight through instead of
listing outputs by hand:

```python
import json
import edgefirst_hal as ef

with open("model.json") as f:
    decoder = ef.Decoder(json.load(f), score_threshold=0.25, iou_threshold=0.45)
```

`Decoder.new_from_json_str` and `Decoder.new_from_yaml_str` take the same
metadata as an unparsed string.

## Object Tracking

`ByteTrack` is a multi-object tracker based on ByteTrack with Kalman filtering.
It assigns consistent track IDs across frames.

```python
import edgefirst_hal as ef

tracker = ef.ByteTrack(
    high_conf=0.7,         # High-confidence detection threshold
    iou=0.25,              # IoU threshold for association
    update=0.25,           # Update/low-confidence threshold
    lifespan_ns=500_000_000,  # Track lifespan without detection (nanoseconds)
)

# Decode and track in one call (returns boxes, scores, classes, masks, track_infos)
boxes, scores, classes, masks, tracks = decoder.decode_tracked(
    tracker, timestamp_ns, [output_tensor]
)
# masks is empty for detection-only models

# Or query currently active tracks
active = tracker.get_active_tracks()
```

## Segmentation Mask Rendering

### draw_decoded_masks()

Draw pre-decoded masks onto a destination image:

```python
processor.draw_decoded_masks(
    dst,
    bbox,           # numpy array [N, 4]
    scores,         # numpy array [N]
    classes,        # numpy array [N]
    seg=[],         # list of segmentation arrays (optional)
    background=None,  # optional background tensor to blit before drawing
    opacity=1.0,    # mask alpha scale (0.0 – 1.0)
)
```

### draw_masks()

Decode model outputs and draw segmentation masks in a single call. Masks never
leave Rust, eliminating the Python round-trip overhead of `decode()` +
`draw_decoded_masks()`.

Without a tracker, returns `(boxes, scores, classes)`. With a tracker, returns
`(boxes, scores, classes, track_infos)`.

```python
import edgefirst_hal as ef

processor = ef.ImageProcessor()
tracker = ef.ByteTrack()

# Without tracking
boxes, scores, classes = processor.draw_masks(decoder, outputs, dst)

# With overlay parameters
boxes, scores, classes = processor.draw_masks(
    decoder, outputs, dst,
    background=bg_tensor,  # blit bg_tensor into dst before masks
    opacity=0.7,           # semi-transparent masks
)

# With tracking (requires tracker= and timestamp=)
import time
ts = time.monotonic_ns()
boxes, scores, classes, tracks = processor.draw_masks(
    decoder, outputs, dst,
    tracker=tracker,
    timestamp=ts,
)
```

## Platform Support

| Platform | GPU acceleration | `TensorMemory` kinds |
|----------|------------------|----------------------|
| Linux (NXP i.MX 8M Plus, i.MX 95) | OpenGL ES + G2D | DMA, SHM, PBO, MEM |
| Linux (x86_64, other ARM) | OpenGL ES | DMA, SHM, PBO, MEM |
| macOS | OpenGL ES via ANGLE (Metal) | DMA (IOSurface-backed), SHM, MEM |
| Windows | CPU only | MEM |

`TensorMemory.DMA` names the platform's zero-copy GPU buffer: DMA-BUF on
Linux, IOSurface on macOS. `is_gpu_buffer_available()` tells you whether it
will succeed without you having to care which one you got;
`is_dma_available()` and `is_iosurface_available()` answer the narrower,
platform-specific question.

DMA on Linux needs a usable DMA-BUF heap, so `create_image()` falls back to
PBO and then MEM where there isn't one. Hardware acceleration is selected
automatically at `ImageProcessor()` construction; every platform has a
working CPU fallback.

## Part of the EdgeFirst Ecosystem

`edgefirst-hal` is the runtime inference library in the
[EdgeFirst](https://edgefirst.ai) platform for deploying AI at the edge.

- **[EdgeFirst Studio](https://edgefirst.studio)** — label, train, and
  deploy models for edge devices
- **[Rust crates](https://crates.io/crates/edgefirst-hal)** — use the
  same library directly from Rust or C
- **[GitHub](https://github.com/EdgeFirstAI/hal)** — source code,
  architecture docs, benchmarks, and contribution guide

## Documentation

- Architecture overview: [ARCHITECTURE.md](https://github.com/EdgeFirstAI/hal/blob/main/crates/python/ARCHITECTURE.md)
- Testing guide: [TESTING.md](https://github.com/EdgeFirstAI/hal/blob/main/crates/python/TESTING.md)
- Project README (cross-language overview): [../../README.md](https://github.com/EdgeFirstAI/hal/blob/main/README.md)
- Optimization guide (cross-language user rules): [README.md#optimization-guide](https://github.com/EdgeFirstAI/hal/blob/main/README.md#optimization-guide)

## License

Apache-2.0 — see [LICENSE](https://github.com/EdgeFirstAI/hal/blob/main/LICENSE).
