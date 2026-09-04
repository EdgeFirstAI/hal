# edgefirst-decoder

YOLO and ModelPack output decoding for edge AI — bounding boxes, segmentation masks and multi-object tracking, without leaving native code.

[![PyPI](https://img.shields.io/pypi/v/edgefirst-decoder.svg)](https://pypi.org/project/edgefirst-decoder/)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

## Part of the EdgeFirst HAL

`edgefirst-decoder` is one of five Python packages built from the [EdgeFirst Hardware Abstraction Layer](https://github.com/EdgeFirstAI/hal).

**The [`EdgeFirstAI/hal`](https://github.com/EdgeFirstAI/hal) repository is the home for all of them** — source, issue tracker, architecture documentation and release notes.

| Package | Provides |
|---|---|
| [`edgefirst-tensor`](https://pypi.org/project/edgefirst-tensor/) | Zero-copy tensor allocation and host/GPU/CUDA mapping |
| [`edgefirst-codec`](https://pypi.org/project/edgefirst-codec/) | JPEG and PNG decoding directly into pre-allocated tensors |
| [`edgefirst-image`](https://pypi.org/project/edgefirst-image/) | GPU-accelerated colour conversion, resize, letterbox, tiling and drawing |
| [`edgefirst-decoder`](https://pypi.org/project/edgefirst-decoder/) | YOLO and ModelPack output decoding (this package) |
| [`edgefirst-tracker`](https://pypi.org/project/edgefirst-tracker/) | ByteTrack multi-object tracking |

## Installation

```bash
pip install edgefirst-decoder
```

Requires Python 3.8 or newer; `edgefirst-tensor` and NumPy are installed automatically. Wheels are published for Linux (x86_64, aarch64), macOS (arm64), and Windows (x86_64).

Packages install under the [PEP 420](https://peps.python.org/pep-0420/) `edgefirst.*` namespace, so the import is `edgefirst.decoder`.

## Quick start

Describe your model's outputs, then decode inference results into detections:

```python
import numpy as np
from edgefirst.decoder import Decoder, Output, Tensor

decoder = Decoder.new_from_outputs(
    outputs=[Output.detection(shape=[1, 84, 8400])],
    score_threshold=0.25,
    iou_threshold=0.45,
)

# `raw` is what your inference runtime produced. Copy it into a HAL tensor;
# in a real pipeline the runtime writes into the tensor directly instead.
raw = np.zeros((1, 84, 8400), dtype=np.float32)
raw[0, 0:4, 0] = [0.5, 0.5, 0.2, 0.2]  # cx, cy, w, h (normalized)
raw[0, 4, 0] = 0.9  # class 0 score

output = Tensor(raw.shape, "float32")
output.from_numpy(raw)

boxes, scores, classes, masks = decoder.decode([output])
print(np.asarray(boxes), np.asarray(scores), np.asarray(classes))
```

Boxes come back normalized. NMS runs by default in class-agnostic mode; pass `nms=Nms.ClassAware` or `nms=None` to change or bypass it.

For a quantized model, attach the quantization parameters to the output description so the decoder dequantizes as it reads:

```python
Output.detection(shape=[1, 84, 8400]).with_quantization(scale=0.004, zero_point=-123)
```

Configurations can also be supplied as a dictionary, or as JSON or YAML, with `Decoder(config_dict)`, `Decoder.new_from_json_str()` and `Decoder.new_from_yaml_str()`. Set `decoder_version=DecoderVersion.Yolo26` for end-to-end Ultralytics models.

`decode()` and friends accept a model-output tensor from any `edgefirst.*` package, not just this one — they cross packages through the capsule protocol, not by type. Value types such as `PixelFormat` and `TensorMemory` are accepted from any package too, comparing and hashing equal across packages by value. Importing `Tensor` from `edgefirst.decoder` when calling into this package is still good style for readability, not a requirement. See the Interoperability section below for the one thing that does not cross: `isinstance` against a concrete class.

## Segmentation masks

`decode()` returns masks at prototype resolution as arrays of shape `(H, W, C)`:

- **Instance segmentation** (YOLO): `C == 1`, a binary per-instance mask — threshold at 128.
- **Semantic segmentation** (ModelPack): `C == num_classes`, per-pixel class scores — take `argmax` over the last axis.

## Multi-object tracking

Tracking lives in the standalone [`edgefirst-tracker`](https://pypi.org/project/edgefirst-tracker/) wheel. `Decoder.decode_tracked` accepts any object with an `update` method, including `edgefirst.tracker.ByteTrack`:

```python
from edgefirst.tracker import ByteTrack

tracker = ByteTrack()
boxes, scores, classes, masks, tracks = decoder.decode_tracked(
    tracker, timestamp_ns, outputs
)
```

## Schema inference for Ultralytics exports

A vanilla Ultralytics YOLOv8/11/26 export carries no `edgefirst.json`, but its
own metadata and tensor shapes are enough to derive one.
`infer_ultralytics_schema` reads what your runtime reports and returns a
schema you can hand straight to `Decoder()`:

```python
import onnxruntime as ort
from edgefirst.decoder import Decoder, infer_ultralytics_schema

sess = ort.InferenceSession("yolov8n.onnx", providers=["CPUExecutionProvider"])

# Ultralytics ONNX exports are float32 throughout and unquantized. A TFLite
# interpreter reports dtype and quantization per tensor instead; pass those as
# a 4th element on each output: (name, shape, dtype, (scales, zero_points)).
inputs = [(t.name, list(t.shape), "float32") for t in sess.get_inputs()]
outputs = [(t.name, list(t.shape), "float32") for t in sess.get_outputs()]
metadata = dict(sess.get_modelmeta().custom_metadata_map)

inferred = infer_ultralytics_schema("onnx", inputs, outputs, metadata)
print(inferred.description)  # "Ultralytics YOLOv8/11 detect, 80 classes"

decoder = Decoder(inferred.schema, score_threshold=0.25, iou_threshold=0.45)

boxes, scores, classes, masks = decoder.decode(model_outputs)
print(inferred.labels[classes[0]])  # class name for the first detection
```

The result is a named tuple, so `schema, labels, description = inferred`
works too — but two of the three fields are strings, and naming them is
cheaper than remembering the order.

`source` decides the box convention: Ultralytics ONNX exports report
pixel-space coordinates, TFLite exports report `[0, 1]`. `"other"` is
accepted but refused by inference rather than defaulted — that convention
follows the exporter, is not derivable from shapes, and guessing it scales
every box by the input size. Supported dtype strings are `"int8"`,
`"uint8"`, `"int16"`, `"uint16"`, `"int32"`, `"uint32"`, `"float16"` and
`"float32"`.

`schema` is a plain dict, ready for `Decoder(schema)` — there is no JSON
string to parse back. `labels` is the class names in index order, which is
what maps `decode()`'s class indices back to names.

Shapes must be concrete. A model exported with `dynamic=True` reports a
symbolic axis (`'batch'` from ONNX, `-1` from TFLite); those are refused
with a `ValueError` naming the tensor and axis, because the layout rules
need real sizes.

The schema pins the NMS *mode* and leaves the *thresholds* to you.
Ultralytics runs NMS class-aware (`agnostic=False`), so an inferred pre-NMS
schema says so rather than inheriting `Decoder`'s class-agnostic default,
which would suppress a box against an overlapping box of a different class.
Passing `nms=` still overrides. Thresholds are not inferable, and
`Decoder`'s defaults (`score_threshold=0.1`, `iou_threshold=0.7`) are not
Ultralytics' (`0.25`/`0.45`) — pass them as shown above. YOLO26 end-to-end
exports apply NMS in-graph and carry no mode at all.

`ValueError` is raised for anything that is not a recognizable Ultralytics
export — missing or unparsable metadata, an unsupported task (only `detect`
and `segment`; pose, OBB and classify are refused), or a class count that does
not fit the output width. Metadata and shapes are cross-checked against each
other, so a disagreement is reported rather than resolved by preference.

## What this package provides

| API | Purpose |
|---|---|
| `Decoder` | Model output decoding to boxes, scores, classes and masks |
| `Decoder.new_from_outputs()` | Programmatic configuration from `Output` descriptions |
| `Decoder.new_from_json_str()` / `new_from_yaml_str()` | Configuration from JSON or YAML |
| `infer_ultralytics_schema()` | Schema derived from a vanilla Ultralytics export's own metadata and shapes |
| `InferredSchema` | Named tuple returned by `infer_ultralytics_schema()` |
| `Output`, `DimName` | Output shape and semantics description |
| `Nms`, `DecoderType`, `DecoderVersion` | NMS mode and model family selection |
| `ProtoData` | Mask prototypes and coefficients |
| `Decoder.draw_onto` | Fused decode + draw onto an `ImageProcessor` |
| `MatchMetric`, `MergeMode`, `MergeConfig`, `TiledFrameAccumulator` | SAHI tile-merge (keep-best by default; `MergeMode.Union` for the enclosing union) |

## Interoperability

`decode()` / `decode_proto()` / `decode_tracked()` accept model-output tensors from *any* `edgefirst.*` package, and `Decoder` / `ProtoData` instances produced here hand off to `edgefirst.image`'s `ImageProcessor` the same way. Each extension module registers its own type objects ([PyO3 issue #1444](https://github.com/PyO3/pyo3/issues/1444) — `isinstance` across packages is always `False`, even for two objects wrapping the same Rust type), so acceptance goes through duck-typed capsule protocols instead:

```python
# CORRECT — works regardless of which edgefirst.* package produced obj
if hasattr(obj, "__edgefirst_tensor__"):
    ...

# WRONG — always False for a tensor from a sibling package
if isinstance(obj, edgefirst.image.Tensor):
    ...
```

`edgefirst.decoder.EdgeFirstTensorExportable` (re-exported from `edgefirst.tensor`), `EdgeFirstDecoderExportable` and `EdgeFirstProtoDataExportable` are `typing.Protocol` classes you can annotate a cross-package parameter with. See [`crates/python-common/INTEROP.md`](https://github.com/EdgeFirstAI/hal/blob/main/crates/python-common/INTEROP.md) for the full protocol.

## Versioning and changelog

All four `edgefirst-*` packages are versioned and released together with the HAL itself, so a given version number refers to the same source tree in every language. Because of that there is no per-package changelog: release notes for every version live in the single [**CHANGELOG.md**](https://github.com/EdgeFirstAI/hal/blob/main/CHANGELOG.md) in the `hal` repository, which follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/) and [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## Links

- [Changelog](https://github.com/EdgeFirstAI/hal/blob/main/CHANGELOG.md) — release notes for all packages
- [Source](https://github.com/EdgeFirstAI/hal) — the `hal` monorepo
- [Issue tracker](https://github.com/EdgeFirstAI/hal/issues)
- [Package documentation](https://github.com/EdgeFirstAI/hal/blob/main/crates/decoder/README.md) — the underlying Rust crate
- [EdgeFirst](https://edgefirst.ai)

## License

Apache-2.0
