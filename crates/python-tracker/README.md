# edgefirst-tracker

ByteTrack multi-object tracking. Installs alone — it does not require the tensor, image, or decoder wheels.

[![PyPI](https://img.shields.io/pypi/v/edgefirst-tracker.svg)](https://pypi.org/project/edgefirst-tracker/)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

## Part of the EdgeFirst HAL

`edgefirst-tracker` is one of five Python packages built from the [EdgeFirst Hardware Abstraction Layer](https://github.com/EdgeFirstAI/hal).

**The [`EdgeFirstAI/hal`](https://github.com/EdgeFirstAI/hal) repository is the home for all of them** — source, issue tracker, architecture documentation and release notes.

| Package | Provides |
|---|---|
| [`edgefirst-tensor`](https://pypi.org/project/edgefirst-tensor/) | Zero-copy tensor allocation and host/GPU/CUDA mapping |
| [`edgefirst-codec`](https://pypi.org/project/edgefirst-codec/) | JPEG and PNG decoding directly into pre-allocated tensors |
| [`edgefirst-image`](https://pypi.org/project/edgefirst-image/) | GPU-accelerated colour conversion, resize, letterbox, tiling and drawing |
| [`edgefirst-decoder`](https://pypi.org/project/edgefirst-decoder/) | YOLO and ModelPack output decoding |
| [`edgefirst-tracker`](https://pypi.org/project/edgefirst-tracker/) | ByteTrack (this package) |

## Installation

```bash
pip install edgefirst-tracker
```

Requires Python 3.8 or newer and NumPy. Wheels are published for Linux (x86_64, aarch64), macOS (arm64), and Windows (x86_64). This package does **not** depend on `edgefirst-tensor`.

Packages install under the [PEP 420](https://peps.python.org/pep-0420/) `edgefirst.*` namespace, so the import is `edgefirst.tracker`.

## Quick start

`ByteTrack.update` takes detections for one timestamp and returns a list of `TrackInfo` (or `None` for unmatched rows) of the same length as the input:

```python
import numpy as np
from edgefirst.tracker import ByteTrack

tracker = ByteTrack()

# boxes is (N, 4) XYXY; scores and labels are length N.
boxes = np.array([[0.1, 0.2, 0.3, 0.4]], dtype=np.float32)
scores = np.array([0.9], dtype=np.float32)
labels = np.array([0], dtype=np.intp)
timestamp_ns = 0

tracks = tracker.update(boxes, scores, labels, timestamp_ns)
for t in tracks:
    if t is None:
        continue
    print(t.uuid, t.tracked_location, t.count)
```

`Decoder.decode_tracked` on the [`edgefirst-decoder`](https://pypi.org/project/edgefirst-decoder/) wheel accepts this `ByteTrack` (or any object with an `update` method).

## Parameters

| Argument | Default | Meaning |
|---|---|---|
| `high_conf` | `0.7` | Detections above this score take the first association pass |
| `iou` | `0.25` | IoU gate for matching a detection to a track |
| `update` | `0.25` | Kalman update blending factor |
| `lifespan_ns` | `500_000_000` | Drop a track that has not matched for this many nanoseconds (500 ms) |

## Errors

`update` raises if `boxes`, `scores`, and `labels` disagree on `N`, or if `boxes` is not shape `(N, 4)`. Empty detections (`N == 0`) are valid — unmatched tracks age and expire according to `lifespan_ns`.

## Performance

Association is CPU-only (Kalman + IoU). Typical cost is microseconds per frame at tens of detections; it does not allocate GPU tensors and does not link `libedgefirst_tensor`. Build the tracker once and call `update` every frame.

## Versioning and changelog

All five `edgefirst-*` packages are versioned and released together with the HAL itself. Release notes live in the single [**CHANGELOG.md**](https://github.com/EdgeFirstAI/hal/blob/main/CHANGELOG.md).

## Links

- [Changelog](https://github.com/EdgeFirstAI/hal/blob/main/CHANGELOG.md) — release notes for all packages
- [Source](https://github.com/EdgeFirstAI/hal) — the `hal` monorepo
- [Issue tracker](https://github.com/EdgeFirstAI/hal/issues)
- [Package documentation](https://github.com/EdgeFirstAI/hal/blob/main/crates/tracker/README.md) — the underlying Rust crate
- [EdgeFirst](https://edgefirst.ai)

## License

Apache-2.0
