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
import edgefirst_hal as ef

# Load a source image
src = ef.Tensor.load("photo.jpg", ef.PixelFormat.Rgb)

# Create an image processor (auto-selects best backend: GPU > G2D > CPU)
processor = ef.ImageProcessor()

# Allocate a GPU-optimal output buffer — always use create_image() for
# destinations passed to convert(), so the processor can select the best
# memory type (DMA-buf, PBO, or system memory) for zero-copy GPU paths.
dst = processor.create_image(640, 640, ef.PixelFormat.Rgb)

# Convert with letterbox resize (preserves aspect ratio)
processor.convert(src, dst)

# Access pixel data as a numpy array
import numpy as np
pixels = np.frombuffer(dst.map(), dtype=np.uint8).reshape(dst.shape())
```

## Key Features

- **Zero-copy tensors** — DMA-BUF, POSIX shared memory, and PBO-backed
  buffers with automatic fallback to system memory
- **Hardware-accelerated image processing** — OpenGL, NXP G2D, and
  optimized CPU backends with automatic selection
- **Letterbox resize** — aspect-ratio-preserving resize with configurable
  padding color, rotation, and flip
- **Int8 output** — `create_image(..., dtype="int8")` for direct signed
  int8 tensor output with GPU-accelerated XOR bias
- **YOLO decoding** — YOLOv5, YOLOv8, YOLO11, and YOLO26 detection and
  instance segmentation (including end-to-end models)
- **Object tracking** — ByteTrack multi-object tracker with Kalman filtering
- **Fully typed** — ships with `.pyi` stubs for IDE autocompletion and
  type checking with mypy / pyright

## Image Processing

```python
import edgefirst_hal as ef

processor = ef.ImageProcessor()
src = ef.Tensor.load("frame.jpg", ef.PixelFormat.Rgb)

# Letterbox resize to model input size
dst = processor.create_image(640, 640, ef.PixelFormat.Rgb)
processor.convert(src, dst)

# With rotation and horizontal flip
processor.convert(src, dst, rotation=ef.Rotation.Rotate90, flip=ef.Flip.Horizontal)

# Crop source region
processor.convert(src, dst, src_crop=ef.Rect(100, 100, 400, 400))

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
src = ef.Tensor.load("frame.jpg", ef.PixelFormat.Rgb)

# Render directly into the delegate's DMA-BUF — zero copies
dst = processor.create_image_from_fd(vx_fd, 640, 640, ef.PixelFormat.Rgb)
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

## YOLO Decoding

```python
import edgefirst_hal as ef

# Configure decoder from model metadata
decoder = ef.Decoder(
    {"detection": {"shape": [1, 84, 8400], "dtype": "float32"}},
    score_threshold=0.5,
    iou_threshold=0.45,
)

# Decode model outputs → (boxes, scores, class_ids)
boxes, scores, classes = decoder.decode([output_tensor])
```

## Platform Support

| Platform | GPU Acceleration | Memory Types |
|----------|-----------------|-------------|
| Linux (NXP i.MX8/i.MX95) | OpenGL + G2D | DMA-buf, SHM, PBO, Mem |
| Linux (x86_64, other ARM) | OpenGL | SHM, PBO, Mem |
| macOS / Windows | CPU only | Mem |

Hardware acceleration is used automatically when available. All platforms
fall back to CPU.

## Part of the EdgeFirst Ecosystem

`edgefirst-hal` is the runtime inference library in the
[EdgeFirst](https://edgefirst.ai) platform for deploying AI at the edge.

- **[EdgeFirst Studio](https://edgefirst.studio)** — label, train, and
  deploy models for edge devices
- **[Rust crates](https://crates.io/crates/edgefirst-hal)** — use the
  same library directly from Rust or C
- **[GitHub](https://github.com/EdgeFirstAI/hal)** — source code,
  architecture docs, benchmarks, and contribution guide

## License

Apache-2.0 — see [LICENSE](https://github.com/EdgeFirstAI/hal/blob/main/LICENSE).
