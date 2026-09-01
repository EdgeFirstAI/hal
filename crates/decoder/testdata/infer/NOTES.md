# Ultralytics export signal fixtures

Captured from real exports on 2026-08-31 with
`scripts/capture_infer_fixtures.py`, from models exported using a developer's
own Ultralytics installation (ultralytics 8.4.137, torch 2.13.0,
onnxruntime 1.29.0, ai-edge-litert 2.2.0). This repository ships no
Ultralytics code and installs none: these fixtures record the *shapes and
metadata* such an export produces, so the decoder — an independent Rust
implementation — can be tested against them offline and with no
network. 11 fixtures, one per supported export:

- ONNX (pixel-space, no quantization metadata): `yolov8n`, `yolo11n`,
  `yolo26n`, `yolov8n-seg`, `yolo11n-seg`, `yolo26n-seg`
- TFLite (normalized, embeds `metadata.json`): `yolov8n_float32`,
  `yolov8n_int8`, `yolov8n-seg_float32`, `yolov8n-seg_int8`,
  `yolo26n_float32`

`ai-edge-litert` installed and imported cleanly on this host (Python 3.11
venv) — no fallback to `tensorflow.lite.Interpreter` was needed.

Toolchain note: as of ultralytics 8.4.137, `format=tflite` is deprecated
and silently redirected to the unified `format=litert` exporter. That
exporter writes the plain (non-int8) output as `<model>.tflite` with **no**
`_float32` suffix — it only appends a disambiguating suffix when
`int8=True` is passed (confirmed: `yolov8n_int8.tflite`,
`yolov8n-seg_int8.tflite` were named correctly by the exporter itself).
The plain export must therefore be renamed to `<model>_float32.tflite` by
hand to match the fixture names below (see TESTING.md).

## Answers

**(1) ONNX `metadata_props` keys and sample values** (identical key set
across all 6 ONNX exports; `dict(sess.get_modelmeta().custom_metadata_map)`,
all values are strings):

```
end2end     "False"                                  # "True" for yolo26n/-seg (e2e NMS-free head)
description "Ultralytics YOLOv8n model trained on coco.yaml"
channels    "3"
author      "Ultralytics"
docs        "https://docs.ultralytics.com"
head        "Detect"
version     "8.4.137"
batch       "1"
license     ""                                       # blanked at capture, see below
stride      "32"
task        "detect"
date        "2026-08-31T12:57:21.834115-06:00"
imgsz       "[640, 640]"
names       "{0: 'person', 1: 'bicycle', ... }"        # stringified Python dict
args        "{'data': None, 'batch': 1, ... }"         # stringified Python dict
```

`end2end` is the one field that varies meaningfully by family: `"False"`
for yolov8n/yolo11n (NMS-required heads), `"True"` for yolo26n (built-in
NMS-free end-to-end head) — confirmed across both det and seg variants of
each family.

**(2) TFLite metadata zip entry name/format**: every TFLite export (float32
and int8, det and seg) carries exactly one zip-associated file named
**`metadata.json`** — not a YAML as the brief's working assumption guessed.
It is a JSON object with the *same field set* as the ONNX metadata above,
but properly typed (e.g. `stride` is a JSON number, `names` is a real JSON
object keyed by string index) rather than all-strings:

```json
{"description": "Ultralytics YOLOv8n model trained on coco.yaml",
 "author": "Ultralytics", "date": "...", "version": "8.4.137",
 "license": "", "docs": "...", "stride": 32, "task": "detect",
 "head": "Detect", "batch": 1, "imgsz": [640, 640],
 "names": {"0": "person", "1": "bicycle", ...}}
```

**(3) YOLO26 det AND seg ONNX output shapes**:
- `yolo26n.onnx`: input `images` `[1, 3, 640, 640]`; output `output0`
  `[1, 300, 6]` (end-to-end: 300 fixed detections × [x1,y1,x2,y2,conf,cls]).
- `yolo26n-seg.onnx`: input `images` `[1, 3, 640, 640]`; outputs `output0`
  `[1, 300, 38]` (300 × [x1,y1,x2,y2,conf,cls,32 mask coeffs] — confirms the
  spec's `[1, N, 38]` assumption) and `output1` `[1, 32, 160, 160]` (mask
  prototypes, same shape convention as v8/11 seg).
- For contrast, `yolov8n.onnx` output0 is `[1, 84, 8400]` (anchor-free grid,
  NMS not baked in) and `yolov8n-seg.onnx` outputs are `[1, 116, 8400]` +
  `[1, 32, 160, 160]`.

**(4) Is int8 TFLite output quantization per-tensor?** The question doesn't
quite apply the way it was posed: the int8 exports' *externally exposed*
input/output tensors are **float32 with no quantization at all**
(`quantization == (0.0, 0)`, empty `scales`/`zero_points` arrays) — the
LiteRT exporter wraps the fully-int8 internal graph with
quantize/dequantize boundary ops, so `get_input_details()` /
`get_output_details()` never show int8 dtype or scale/zero-point for
`yolov8n_int8.tflite` or `yolov8n-seg_int8.tflite`. This is a real behavior
change from older Ultralytics TFLite exports that exposed uint8/int8 I/O
directly — **schema inference cannot distinguish an int8 export from a
float32 one by inspecting boundary tensor dtype/quantization; it needs
another signal** (e.g. the embedded `metadata.json`, filename, or internal
tensor inspection).

Going one level in with `interpreter.get_tensor_details()` (all 448
tensors, not just I/O): the internal int8 tensor immediately before the
final dequantize op has a single scale/zero-point pair (e.g.
`(0.003905248362571001, -128)`, `len(scales) == 1`) — i.e. genuinely
**per-tensor** quantization, not per-channel, confirmed on both
`yolov8n_int8.tflite` and `yolov8n-seg_int8.tflite`.

**(5) Normalization check** (real image: `ultralytics/assets/bus.jpg`,
letterboxed to 640×640, NCHW float32 in [0,1], fed identically to both
models):
- `yolov8n.onnx`: max box-coordinate magnitude = **637.25** → pixel-space
  (consistent with the ~640 input size).
- `yolov8n_float32.tflite`: max box-coordinate magnitude = **0.9957** →
  normalized [0,1].

This confirms the spec's assumption (a): ONNX boxes are pixel-space, TFLite
boxes are normalized [0,1].

## Fixture format

Each `<export-name>.signals.json` is `{"source": "onnx"|"tflite", "inputs":
[...], "outputs": [...], "metadata": {...}}` per
`tests/decoder/ultralytics_signals.py::signals_for`. All 11 fixtures parse
as JSON and have non-empty `inputs`/`outputs`; ONNX fixtures have 15
metadata keys each, TFLite fixtures have exactly 1 (`metadata.json`).

One capture-time transform is applied: Ultralytics stamps the absolute path
of the dataset YAML from whichever machine trained the released weights into
the `description` metadata field (`/usr/src/ultralytics/...` from their
Docker image, `/home/<user>/...` from a developer checkout). Nothing in the
inference path reads `description`, so `capture_infer_fixtures.py::scrub`
reduces each such path to its basename (`coco.yaml`) before writing — the
fixtures stay portable and diff cleanly across recaptures. Every other field
is captured verbatim, with one exception: the exporter's own `license`
string is blanked by `capture_infer_fixtures.py::drop_upstream_license`.
Nothing in the inference path reads it, and this repository contains no
upstream model code, so carrying another project's licence declaration in
our test data would only misstate what this code is. `author` and `docs`
are kept deliberately — `infer_ultralytics_schema` reads them to refuse a
model whose metadata names a different vendor.
