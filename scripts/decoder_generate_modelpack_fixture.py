#!/usr/bin/env python3
"""ModelPack decoder reference fixture generator.

PURPOSE
-------
Capture a ModelPack anchor-grid model's raw per-FPN-scale outputs plus the
reference post-NMS detections into a single ``.safetensors`` file for use as
an oracle in HAL Rust unit tests (``crates/decoder/tests/``).

Supports both TFLite (int8 quantized, zip-wrapped with embedded
``edgefirst.json``) and ONNX (float32, ``edgefirst`` key in
``metadata_props``) models produced by ModelPack 4.0+ exports with
``decoder: modelpack`` and ``encoding: anchor`` per output.

The fixture lets HAL Rust tests load the raw tensors and the embedded
``edgefirst.json`` schema, run the HAL ModelPack decoder, and assert
parity against the reference detections — without pulling any inference
runtime into the Rust test.

REFERENCE DECODE FORMULA
------------------------
Per FPN scale with shape ``[1, H, W, num_anchors * (5 + nc)]``::

    g    = sigmoid(raw_or_dequant).reshape(1, H, W, na, 5 + nc)
    grid = meshgrid(x=0..W, y=0..H)                            # [1, H, W, 1, 2]
    xy   = (g[..., 0:2] * 2 + grid - 0.5) / [W, H]             # normalized [0, 1]
    wh   = (g[..., 2:4] * 2) ** 2 * anchors * 0.5              # normalized [0, 1]
    obj  = g[..., 4:5]
    cls  = g[..., 5:]
    scores = obj                  if nc == 1 else obj * cls
    boxes_xyxy = concat([xy - wh, xy + wh], axis=-1)

Mirrors ``deepview/modelpack/demos/models/inference_no_decoder.py``.

USAGE
-----
::

    python scripts/decoder_generate_modelpack_fixture.py \\
        coffeecup-mpk-det-relu-t-27d6_quant-u8-i8_smart.tflite \\
        testdata/coffeecup.jpg \\
        --output testdata/decoder/coffeecup-mpk-det-relu-t-27d6_quant-u8-i8_smart.safetensors

    python scripts/decoder_generate_modelpack_fixture.py \\
        coffeecup-mpk-det-relu-t-27d6.onnx \\
        testdata/coffeecup.jpg \\
        --output testdata/decoder/coffeecup-mpk-det-relu-t-27d6.safetensors
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import subprocess
import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class FixtureConfig:
    model_path: Path
    image_path: Path
    output_path: Path
    score_threshold: float
    iou_threshold: float
    max_output: int


def parse_args(argv: list[str] | None = None) -> FixtureConfig:
    p = argparse.ArgumentParser(
        prog="decoder_generate_modelpack_fixture",
        description="Generate a HAL ModelPack decoder reference fixture (.safetensors).",
    )
    p.add_argument("model", type=Path, help="Path to .tflite or .onnx model.")
    p.add_argument("image", type=Path, help="Representative input image.")
    p.add_argument("--output", required=True, type=Path, help="Output .safetensors path.")
    p.add_argument("--score-threshold", type=float, default=0.25)
    p.add_argument("--iou-threshold", type=float, default=0.5)
    p.add_argument("--max-output", type=int, default=300)
    args = p.parse_args(argv)
    return FixtureConfig(
        model_path=args.model,
        image_path=args.image,
        output_path=args.output,
        score_threshold=args.score_threshold,
        iou_threshold=args.iou_threshold,
        max_output=args.max_output,
    )


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fp:
        for chunk in iter(lambda: fp.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


# ──────────────────────────────────────────────────────────────────────
# Schema extraction
# ──────────────────────────────────────────────────────────────────────

def extract_schema_tflite(model_path: Path) -> dict:
    """ModelPack TFLite is a zip with edgefirst.json inside."""
    with zipfile.ZipFile(model_path) as zf:
        with zf.open("edgefirst.json") as fp:
            return json.loads(fp.read().decode("utf-8"))


def extract_schema_onnx(model_path: Path) -> dict:
    """ModelPack ONNX stores schema in metadata_props['edgefirst']."""
    import onnxruntime as ort
    sess = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    meta = sess.get_modelmeta()
    js = meta.custom_metadata_map.get("edgefirst")
    if js is None:
        raise SystemExit(
            f"{model_path}: ONNX model has no 'edgefirst' custom metadata "
            "key — not a ModelPack-exported model?"
        )
    return json.loads(js)


# ──────────────────────────────────────────────────────────────────────
# Inference
# ──────────────────────────────────────────────────────────────────────

def detect_input_hw(input_shape: list[int]) -> tuple[int, int, str]:
    """Return (in_h, in_w, layout) for a 4D model input. ``layout`` is 'NHWC' or 'NCHW'.

    Detection rule: channel axis is whichever of indices 1 or 3 equals 3.
    """
    if len(input_shape) != 4 or input_shape[0] != 1:
        raise SystemExit(f"unexpected input shape {input_shape}; need [1, ...]")
    if input_shape[3] == 3:
        return int(input_shape[1]), int(input_shape[2]), "NHWC"
    if input_shape[1] == 3:
        return int(input_shape[2]), int(input_shape[3]), "NCHW"
    raise SystemExit(f"could not locate channel=3 axis in {input_shape}")


def preprocess_image(image_path: Path, in_h: int, in_w: int) -> np.ndarray:
    """Resize to model input, return uint8 NHWC [1, H, W, 3]."""
    from PIL import Image
    img = Image.open(image_path).convert("RGB").resize((in_w, in_h))
    return np.array(img, dtype=np.uint8)[np.newaxis]  # [1, H, W, 3]


def infer_tflite(model_path: Path, image_uint8_nhwc: np.ndarray) -> dict[tuple[int, ...], np.ndarray]:
    """Run TFLite inference; return {shape: raw_int8_tensor}.

    Binds outputs by shape (TFLite output names like ``PartitionedCall:0``
    don't map to schema names like ``output_0``). Per-tensor quantization
    parameters travel via the embedded ``edgefirst.json`` instead of being
    threaded out of this function.
    """
    try:
        from tflite_runtime.interpreter import Interpreter
    except ImportError:
        from tensorflow.lite.python.interpreter import Interpreter
    interp = Interpreter(model_path=str(model_path))
    interp.allocate_tensors()
    in_det = interp.get_input_details()[0]
    interp.set_tensor(in_det["index"], image_uint8_nhwc)
    interp.invoke()
    raw: dict[tuple[int, ...], np.ndarray] = {}
    for od in interp.get_output_details():
        t = interp.get_tensor(od["index"])
        raw[tuple(int(x) for x in t.shape)] = t
    return raw


def infer_onnx(model_path: Path, image_uint8_nhwc: np.ndarray) -> dict[tuple[int, ...], np.ndarray]:
    """Run ONNX inference; return {shape: float32_tensor} in NHWC layout.

    ModelPack ONNX takes NCHW float32 input but the heads emit NHWC outputs.
    """
    import onnxruntime as ort
    sess = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    in_meta = sess.get_inputs()[0]
    # NCHW float32 [1, 3, H, W]; normalize uint8/255.0
    img_nchw = (image_uint8_nhwc.astype(np.float32) / 255.0).transpose(0, 3, 1, 2)
    outs = sess.run(None, {in_meta.name: img_nchw})
    return {tuple(int(x) for x in t.shape): t for t in outs}


# ──────────────────────────────────────────────────────────────────────
# Reference ModelPack decoder (pure numpy)
# ──────────────────────────────────────────────────────────────────────

def _sigmoid(x: np.ndarray) -> np.ndarray:
    # Numerically stable sigmoid
    out = np.empty_like(x, dtype=np.float32)
    pos = x >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    e = np.exp(x[~pos])
    out[~pos] = e / (1.0 + e)
    return out


def decode_modelpack_outputs(
    outputs_by_name: dict[str, np.ndarray],
    schema: dict,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply ModelPack reference decode to all detection outputs.

    Returns (boxes_xyxy [N, 4], scores [N, num_classes]) — flattened
    across FPN scales, pre-NMS, normalized [0, 1] coordinates.
    """
    num_classes = len(schema.get("dataset", {}).get("classes", [])) or 1
    all_boxes, all_scores = [], []
    for spec in schema["outputs"]:
        if spec.get("type") != "detection":
            continue
        name = spec["name"]
        if name not in outputs_by_name:
            raise SystemExit(f"output {name!r} not found in model outputs")
        t = outputs_by_name[name].astype(np.float32)
        anchors = np.asarray(spec["anchors"], dtype=np.float32)
        na = len(anchors)
        _, gh, gw, c = t.shape
        nc = c // na - 5
        if nc != num_classes:
            raise SystemExit(
                f"output {name!r}: derived nc={nc} from shape, "
                f"but schema says {num_classes} classes"
            )
        g = _sigmoid(t).reshape(1, gh, gw, na, nc + 5)
        gx, gy = np.meshgrid(np.arange(gw), np.arange(gh))
        grid = np.stack([gx, gy], axis=-1).astype(np.float32)
        grid = grid[None, ..., None, :]  # [1, H, W, 1, 2]
        xy = (g[..., 0:2] * 2.0 + grid - 0.5) / np.array([gw, gh], dtype=np.float32)
        wh = (g[..., 2:4] * 2.0) ** 2 * anchors * 0.5
        obj = g[..., 4:5]
        cls = g[..., 5:]
        scores = obj if num_classes == 1 else obj * cls
        boxes = np.concatenate([xy - wh, xy + wh], axis=-1)
        all_boxes.append(boxes.reshape(-1, 4))
        all_scores.append(scores.reshape(-1, num_classes))
    boxes = np.concatenate(all_boxes, axis=0)
    scores = np.concatenate(all_scores, axis=0)
    return boxes, scores


def nms_class_agnostic(
    boxes: np.ndarray,
    scores: np.ndarray,
    score_threshold: float,
    iou_threshold: float,
    max_output: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Greedy class-agnostic NMS. Returns (boxes [K,4], scores [K], classes [K])."""
    # Reduce per-anchor multi-class to (max_score, argmax_class)
    max_scores = scores.max(axis=1)
    classes = scores.argmax(axis=1)
    keep_mask = max_scores >= score_threshold
    boxes = boxes[keep_mask]
    max_scores = max_scores[keep_mask]
    classes = classes[keep_mask]

    order = np.argsort(-max_scores)
    keep: list[int] = []
    while len(order) > 0 and len(keep) < max_output:
        i = int(order[0])
        keep.append(i)
        if len(order) == 1:
            break
        rest = order[1:]
        x1 = np.maximum(boxes[i, 0], boxes[rest, 0])
        y1 = np.maximum(boxes[i, 1], boxes[rest, 1])
        x2 = np.minimum(boxes[i, 2], boxes[rest, 2])
        y2 = np.minimum(boxes[i, 3], boxes[rest, 3])
        inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
        area_i = (boxes[i, 2] - boxes[i, 0]) * (boxes[i, 3] - boxes[i, 1])
        area_r = (boxes[rest, 2] - boxes[rest, 0]) * (boxes[rest, 3] - boxes[rest, 1])
        iou = inter / (area_i + area_r - inter + 1e-9)
        order = rest[iou < iou_threshold]
    keep_idx = np.array(keep, dtype=np.int64)
    return boxes[keep_idx], max_scores[keep_idx], classes[keep_idx]


# ──────────────────────────────────────────────────────────────────────
# Dequantize helpers
# ──────────────────────────────────────────────────────────────────────

def dequantize_tensors(
    raw_by_shape: dict[tuple[int, ...], np.ndarray],
    schema: dict,
) -> dict[str, np.ndarray]:
    """Bind raw tensors to schema outputs by shape and dequantize each.

    Returns ``{schema_output_name: float32_ndarray}``.
    """
    dequant = {}
    for spec in schema["outputs"]:
        shape = tuple(spec["shape"])
        if shape not in raw_by_shape:
            raise SystemExit(f"output {spec['name']!r}: shape {shape} not produced by model")
        t = raw_by_shape[shape]
        q = spec.get("quantization")
        if q is not None and t.dtype != np.float32:
            scale = float(q["scale"])
            zp = int(q["zero_point"])
            dequant[spec["name"]] = (t.astype(np.float32) - zp) * scale
        else:
            dequant[spec["name"]] = t.astype(np.float32)
    return dequant


def raw_tensors_by_name(
    raw_by_shape: dict[tuple[int, ...], np.ndarray],
    schema: dict,
) -> dict[str, np.ndarray]:
    """Bind raw (un-dequantized) tensors to schema names by shape."""
    out = {}
    for spec in schema["outputs"]:
        shape = tuple(spec["shape"])
        if shape in raw_by_shape:
            out[spec["name"]] = raw_by_shape[shape]
    return out


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────

DOCUMENTATION_TEMPLATE = """\
# {basename} — HAL ModelPack decoder fixture

Generated by ``scripts/decoder_generate_modelpack_fixture.py``.

## Source
- Model: ``{model_path}`` (sha256: ``{model_sha}``)
- Image: ``{image_path}``
- Generated: ``{generated_at}``
- Generator git sha: ``{git_sha}``

## Tensor keys
- ``input.image`` — preprocessed uint8 NHWC [1, {in_h}, {in_w}, 3].
- ``raw.<output_name>`` — raw model output per FPN scale (dtype matches model).
- ``intermediate.<output_name>.dequant`` — float32 dequantized output.
- ``decoded.boxes_xyxy`` — post-NMS reference boxes [N, 4], normalized [0, 1].
- ``decoded.scores`` — post-NMS reference scores [N].
- ``decoded.classes`` — post-NMS reference class indices [N].

## Decode formula (per scale)
```
g    = sigmoid(dequant).reshape(1, H, W, na, 5 + nc)
xy   = (g[..., 0:2] * 2 + meshgrid - 0.5) / [W, H]
wh   = (g[..., 2:4] * 2) ** 2 * anchors * 0.5
obj  = g[..., 4:5]
cls  = g[..., 5:]
score = obj if nc == 1 else obj * cls
xyxy  = concat([xy - wh, xy + wh])
```
NMS: class-agnostic, score≥{score_threshold}, IoU<{iou_threshold},
max={max_output} detections.

## HAL kernel mapping
- ``raw.*`` → input to ``DecoderBuilder::with_schema(schema).build()``.
- ``intermediate.<name>.dequant`` → expected output of
  ``modelpack::postprocess_modelpack_split_quant`` dequant pass.
- ``decoded.*`` → expected post-``Decoder::decode`` outputs.

## Consumer tests
- ``crates/decoder/tests/modelpack_coffeecup_parity.rs``
"""


def main() -> None:
    cfg = parse_args()
    model_path = cfg.model_path
    image_path = cfg.image_path

    # 1. Schema
    suffix = model_path.suffix.lower()
    if suffix == ".tflite":
        schema = extract_schema_tflite(model_path)
        decoder_family = "modelpack_split_int8"
    elif suffix == ".onnx":
        schema = extract_schema_onnx(model_path)
        decoder_family = "modelpack_split_float"
    else:
        raise SystemExit(f"unsupported model suffix {suffix!r}; expect .tflite or .onnx")

    # 2. Preprocess (always feed inference an NHWC uint8 tensor — infer_onnx
    # transposes to NCHW where needed; in_h/in_w come from whichever layout
    # the schema declares).
    in_h, in_w, layout = detect_input_hw(schema["input"]["shape"])
    image_uint8 = preprocess_image(image_path, in_h, in_w)

    # 3. Inference
    if suffix == ".tflite":
        raw_by_shape = infer_tflite(model_path, image_uint8)
    else:
        raw_by_shape = infer_onnx(model_path, image_uint8)

    raw_by_name = raw_tensors_by_name(raw_by_shape, schema)
    dequant_by_name = dequantize_tensors(raw_by_shape, schema)

    # 4. Reference decode + NMS
    pre_nms_boxes, pre_nms_scores = decode_modelpack_outputs(dequant_by_name, schema)
    boxes_xyxy, scores, classes = nms_class_agnostic(
        pre_nms_boxes, pre_nms_scores,
        cfg.score_threshold, cfg.iou_threshold, cfg.max_output,
    )
    print(f"reference pre-NMS detections: {len(pre_nms_boxes)}", file=sys.stderr)
    print(f"reference post-NMS detections: {len(boxes_xyxy)} (score≥{cfg.score_threshold})",
          file=sys.stderr)

    # 5. Pack tensors
    tensors: dict[str, np.ndarray] = {
        "input.image": image_uint8.squeeze(0),
    }
    for name, t in raw_by_name.items():
        tensors[f"raw.{name}"] = np.ascontiguousarray(t)
    for name, t in dequant_by_name.items():
        tensors[f"intermediate.{name}.dequant"] = np.ascontiguousarray(t)
    tensors["decoded.boxes_xyxy"] = np.ascontiguousarray(boxes_xyxy.astype(np.float32))
    tensors["decoded.scores"] = np.ascontiguousarray(scores.astype(np.float32))
    # Use uint32 so PerScaleFixture::decoded() reads it as the u32 it expects.
    tensors["decoded.classes"] = np.ascontiguousarray(classes.astype(np.uint32))

    # 6. Metadata — keys match testdata/decoder/common/per_scale_fixture.rs
    # so the existing PerScaleFixture loader can consume modelpack fixtures
    # without code duplication. For the float ONNX path we emit an empty
    # ``quantization_json`` map so the loader skips quant attachment.
    quant_meta = {
        spec["name"]: spec["quantization"]
        for spec in schema["outputs"]
        if spec.get("quantization") is not None
    }
    # Key name ``max_detections`` matches PerScaleFixture::load() so the
    # Rust test reads the actual fixture-time cap instead of falling back
    # to the loader's default.
    nms_cfg = {
        "mode": "class_agnostic",
        "score_threshold": cfg.score_threshold,
        "iou_threshold": cfg.iou_threshold,
        "max_detections": cfg.max_output,
    }
    metadata = {
        "format_version": "1",
        "decoder_family": decoder_family,
        "model_path": str(model_path),
        "model_basename": model_path.name,
        "model_sha256": sha256_file(model_path),
        "image_path": str(image_path),
        "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "generator_git_sha": git_sha(),
        "reference_script_sha256": sha256_file(Path(__file__).resolve()),
        "schema_json": json.dumps(schema, separators=(",", ":")),
        "edgefirst_json": json.dumps(schema, separators=(",", ":")),  # alias for layout-fixture parity
        "quantization_json": json.dumps(quant_meta, separators=(",", ":")),
        "nms_config_json": json.dumps(nms_cfg, separators=(",", ":")),
        "expected_count_min": str(len(boxes_xyxy)),
        "documentation_md": DOCUMENTATION_TEMPLATE.format(
            basename=model_path.name,
            model_path=model_path,
            model_sha=sha256_file(model_path),
            image_path=image_path,
            generated_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            git_sha=git_sha(),
            in_h=in_h,
            in_w=in_w,
            score_threshold=cfg.score_threshold,
            iou_threshold=cfg.iou_threshold,
            max_output=cfg.max_output,
        ),
    }

    # 7. Save
    from safetensors.numpy import save_file
    cfg.output_path.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(cfg.output_path), metadata=metadata)
    print(f"wrote {cfg.output_path} ({cfg.output_path.stat().st_size} bytes)", file=sys.stderr)


if __name__ == "__main__":
    main()
