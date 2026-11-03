# edgefirst-kernels

Quantized and floating-point compute kernels (softmax, sigmoid, DFL, dist2bbox) for edge AI inference. Installs alone — it does not require the tensor, image, or decoder wheels.

[![PyPI](https://img.shields.io/pypi/v/edgefirst-kernels.svg)](https://pypi.org/project/edgefirst-kernels/)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

## Part of the EdgeFirst HAL

`edgefirst-kernels` is one of the Python packages built from the [EdgeFirst Hardware Abstraction Layer](https://github.com/EdgeFirstAI/hal).

**The [`EdgeFirstAI/hal`](https://github.com/EdgeFirstAI/hal) repository is the home for all of them** — source, issue tracker, architecture documentation and release notes.

## Installation

```bash
pip install edgefirst-kernels
```

Requires Python 3.8 or newer and NumPy. This package does **not** depend on `edgefirst-tensor`. The import is `edgefirst.kernels`.

## Quick start

Build a plan once, then run it on numpy arrays. Arrays are read in place with any non-negative strides (a transposed view works without a copy), and `run` releases the GIL.

```python
import numpy as np
from edgefirst.kernels import SoftmaxI8Plan

plan = SoftmaxI8Plan(
    input_scale=0.1,
    input_zero_point=0,
    output_scale=1 / 256,
    output_zero_point=-128,
)
x = np.random.default_rng(0).integers(-128, 128, size=(8400, 16), dtype=np.int8)
out = np.empty_like(x)
plan.run(x, out, threads=4)
print(plan.tier)
```

Decoding YOLOv8 boxes from int8 logits into float32 pixels:

```python
from edgefirst.kernels import DflBoxPlan

levels = [
    (6400, 80, 8.0),
    (1600, 40, 16.0),
    (400, 20, 32.0),
]  # (anchors, grid_w, stride)
plan = DflBoxPlan(np.int8, (0.08, -10), np.float32, None, 16, levels, "xywh")
logits = np.zeros((8400, 64), dtype=np.int8)
boxes = np.empty((8400, 4), dtype=np.float32)
plan.run(logits, boxes)
```

## Plans

| Plan | Operation |
|---|---|
| `SoftmaxI8Plan`, `SoftmaxI16Plan` | Softmax over the last axis, matching TFLite's int8 and int16 `SOFTMAX` |
| `SigmoidPlan` | Elementwise sigmoid for int8, uint8, int16, uint16, float16 and float32 |
| `LutI8Plan` | 256-entry lookup table (`from_table`) or the TFLite `LOGISTIC` table (`sigmoid`) |
| `DflPlan` | Distribution focal loss: `[anchors, 4 * reg_max]` to `[anchors, 4]` distances |
| `Dist2BboxPlan` | Distances to boxes over a multi-level anchor axis |
| `DflBoxPlan` | Fused DFL and dist2bbox |

A quantization of `None` means identity, and float tensors ignore theirs. `run` raises `ValueError` when the input and output overlap in memory, interleaved views of one array included. Every plan has `run(input, output, *, threads=None)` and a read-only `tier` property (`scalar`, `neon`, `neon_fp16` or `neon_dotprod`). Set `EDGEFIRST_KERNELS_FORCE_TIER` to force a tier.

## Errors

| Exception | Raised for |
|---|---|
| `ValueError` | Bad parameters, mismatched dtype or shape, negative or unaligned strides, `threads < 1`, input and output sharing memory |
| `NotImplementedError` | Valid input the plan does not handle, or a forced tier the CPU lacks |

## License

Apache-2.0. See [LICENSE](LICENSE).
