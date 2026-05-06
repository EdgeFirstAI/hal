#!/usr/bin/env python3
"""Decoder reference fixture generator.

PURPOSE
-------
Capture a real TFLite model's raw outputs and the corresponding NumPy
reference decoder's expected outputs into a single ``.safetensors`` file
for use as an oracle in HAL Rust unit tests.

The first decoder family supported is "per-scale YOLO segmentation":
models produced by the EdgeFirst tflite-converter with
``quantization_split`` enabled.  These models have their accuracy-
sensitive tail layers (DFL / sigmoid / dist2bbox / per-scale concat)
stripped from the graph so each FPN level keeps independent
quantization parameters.  The HAL re-implements those stripped layers
in higher precision on the CPU.  This generator captures every stage
so HAL parity can be checked stage-by-stage.

PHYSICAL OUTPUT DECOMPOSITION — WHY
------------------------------------
Standard YOLO TFLite models concatenate per-scale outputs and run DFL
internally.  When the whole graph is quantized to int8, the concat and
DFL steps amplify quantization error because they mix logits whose
distributions span very different ranges.  The converter's
``quantization_split`` mode keeps the per-FPN-level outputs separate,
each with its own ``scale``/``zero_point``, and the runtime performs
the DFL + concat + dist2bbox + sigmoid + NMS on the CPU in fp32 (or
fp16).  This recovers most of the accuracy lost to monolithic
quantization (yolov8n-seg: 47.16 → 52.38 mAP@0.5 on coco128).

For a deep dive see ``scripts/per_scale_decode_reference.py``.

ARCHITECTURE STAGES → FIXTURE KEYS → HAL KERNELS
-------------------------------------------------
========================  ============================  ============================
NumPy reference stage     Fixture key                   HAL kernel
========================  ============================  ============================
raw int8 box logits        ``raw.boxes_<lvl>``           kernels/level_box.rs entry
dequantize box logits      ``intermediate.boxes_<lvl>.dequant``  level_box.rs dequant
DFL: softmax + weighted    ``intermediate.boxes_<lvl>.ltrb``  (DFL only)  level_box.rs DFL
dist2bbox + stride scale   ``intermediate.boxes_<lvl>.xywh``  level_box.rs dist2bbox
raw int8 score logits      ``raw.scores_<lvl>``          kernels/level_score.rs entry
dequantize scores          ``intermediate.scores_<lvl>.dequant``  level_score.rs dequant
sigmoid                    ``intermediate.scores_<lvl>.activated``  level_score.rs sigmoid
raw int8 mask coefs        ``raw.mc_<lvl>``              kernels/level_mc.rs entry
dequantize mask coefs      ``intermediate.mc_<lvl>.dequant``  level_mc.rs
raw int8 protos            ``raw.protos``                pipeline.rs proto path
dequantize protos          ``intermediate.protos.dequant``  pipeline.rs
class-agnostic NMS         ``decoded.boxes_xyxy``        decoder/per_scale_bridge.rs
                           ``decoded.scores``
                           ``decoded.classes``
mask cropping & threshold  ``decoded.masks``             decoder/per_scale_bridge.rs
========================  ============================  ============================

USAGE
-----
::

    python scripts/decoder_generate_fixture.py \\
        /tmp/e2e-t2681/yolov8n-seg-t-2681_per-scale.tflite \\
        testdata/zidane.jpg \\
        --output testdata/decoder/yolov8n-seg.safetensors

REPRODUCIBILITY
---------------
Pin numpy and safetensors versions::

    pip install 'numpy>=1.26' 'safetensors>=0.4' 'pillow>=10' 'tflite_runtime>=2.14'

The generator records ``reference_script_sha256`` and
``generator_git_sha`` in the fixture metadata so a future regenerator
can detect drift.
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class FixtureConfig:
    """Validated CLI configuration for fixture generation."""

    model_path: Path
    image_path: Path
    output_path: Path
    include_stages: bool
    score_threshold: float
    iou_threshold: float
    expected_count_min: int


def parse_args(argv: list[str] | None = None) -> FixtureConfig:
    """Parse argv into a :class:`FixtureConfig` or ``SystemExit`` on error."""
    p = argparse.ArgumentParser(
        prog="decoder_generate_fixture",
        description="Generate a HAL decoder reference fixture (.safetensors).",
    )
    p.add_argument("model", type=Path, help="Path to the TFLite model.")
    p.add_argument("image", type=Path, help="Path to a representative input image.")
    p.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Output .safetensors path.",
    )
    p.add_argument(
        "--no-stages",
        dest="include_stages",
        action="store_false",
        default=True,
        help="Omit per-stage intermediate tensors (saves ~70%% size).",
    )
    p.add_argument(
        "--score-threshold",
        type=float,
        default=0.001,
        help="NMS score threshold (default: 0.001 for high recall).",
    )
    p.add_argument(
        "--iou-threshold",
        type=float,
        default=0.7,
        help="NMS IoU threshold (default: 0.7).",
    )
    p.add_argument(
        "--expected-count-min",
        type=int,
        default=10,
        help="Recorded in fixture; smoke test asserts decode "
        "produces at least this many detections.",
    )
    args = p.parse_args(argv)
    return FixtureConfig(
        model_path=args.model,
        image_path=args.image,
        output_path=args.output,
        include_stages=args.include_stages,
        score_threshold=args.score_threshold,
        iou_threshold=args.iou_threshold,
        expected_count_min=args.expected_count_min,
    )


def collect_raw_tensors_by_role(layout, raw_outputs_by_shape: dict) -> dict:
    """Map per-scale children (and protos) to fixture-key -> ndarray.

    Parameters
    ----------
    layout : per_scale_decode_reference.PerScaleLayout
        Parsed model metadata.
    raw_outputs_by_shape : dict[tuple[int, ...], np.ndarray]
        Lookup table of TFLite output tensors indexed by their shape
        tuple — the same primary key the HAL ``resolve_bindings`` uses.

    Returns
    -------
    dict[str, np.ndarray]
        Keys are fixture tensor keys: ``raw.boxes_<lvl>``,
        ``raw.scores_<lvl>``, ``raw.mc_<lvl>``, ``raw.protos``.
        ``raw.mc_<lvl>`` is omitted for detection-only models;
        ``raw.protos`` is omitted when no protos tensor exists.
    """
    table: dict[str, np.ndarray] = {}
    for lvl_idx, lvl in enumerate(layout.fpn_levels):
        bx_shape = tuple(lvl["box_shape"])
        sc_shape = tuple(lvl["score_shape"])
        if bx_shape in raw_outputs_by_shape:
            table[f"raw.boxes_{lvl_idx}"] = raw_outputs_by_shape[bx_shape]
        if sc_shape in raw_outputs_by_shape:
            table[f"raw.scores_{lvl_idx}"] = raw_outputs_by_shape[sc_shape]
        mc_shape = lvl.get("mc_shape")
        if mc_shape is not None:
            mc_shape = tuple(mc_shape)
            if mc_shape in raw_outputs_by_shape:
                table[f"raw.mc_{lvl_idx}"] = raw_outputs_by_shape[mc_shape]
    if layout.protos_shape is not None:
        ps = tuple(layout.protos_shape)
        if ps in raw_outputs_by_shape:
            table["raw.protos"] = raw_outputs_by_shape[ps]
    return table


def run_inference_collect_outputs(model_path: Path, image_path: Path):
    """Run TFLite on one image and return ``(layout, raw_outputs_by_shape, image_uint8, metadata)``.

    Parameters
    ----------
    model_path : Path
    image_path : Path

    Returns
    -------
    tuple
        - ``layout`` : PerScaleLayout
        - ``raw_outputs_by_shape`` : dict[tuple[int,...], np.ndarray]
        - ``image_uint8`` : np.ndarray of shape (H, W, 3) — the
          letterbox-free resize fed to the model
        - ``metadata`` : dict — the raw edgefirst.json, kept separately
          since PerScaleLayout doesn't store it
    """
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    try:
        import per_scale_decode_reference as ref
    finally:
        sys.path.pop(0)
    try:
        from tflite_runtime.interpreter import Interpreter  # type: ignore
    except ImportError:
        from tensorflow.lite.python.interpreter import Interpreter  # type: ignore
    from PIL import Image

    metadata = ref.load_metadata(str(model_path))
    layout = ref.PerScaleLayout(metadata)
    interp = Interpreter(model_path=str(model_path))
    interp.allocate_tensors()
    in_det = interp.get_input_details()[0]
    _, in_h, in_w, _ = in_det["shape"]
    img = Image.open(image_path).convert("RGB").resize((in_w, in_h))
    img_np = np.array(img, dtype=np.uint8)
    interp.set_tensor(in_det["index"], img_np[np.newaxis])
    interp.invoke()
    raw_outputs_by_shape: dict[tuple[int, ...], np.ndarray] = {}
    for od in interp.get_output_details():
        raw_outputs_by_shape[tuple(od["shape"].tolist())] = interp.get_tensor(od["index"])
    return layout, raw_outputs_by_shape, img_np, metadata


def capture_box_score_mc_intermediates(
    *,
    level_index: int,
    encoding: str,         # "dfl" or "ltrb"
    reg_max: int | None,   # required for dfl, ignored for ltrb
    h: int,
    w: int,
    stride: int,
    box_q: np.ndarray, box_q_scale: float, box_q_zp: int,
    sco_q: np.ndarray, sco_q_scale: float, sco_q_zp: int,
    mc_q:  np.ndarray | None, mc_q_scale: float | None, mc_q_zp: int | None,
) -> dict[str, np.ndarray]:
    """Capture per-stage intermediates for one FPN level.

    For DFL encoding the per-stage tensors are
    ``boxes_<lvl>.{dequant, ltrb, xywh}``; for LTRB the ``ltrb`` key is
    skipped (the post-DFL stage doesn't exist in that path).

    Score intermediates are always ``{dequant, activated}``; the
    activation is sigmoid for the per-scale path.

    Mask-coef intermediates are dequant-only; ``mc_q=None`` produces no
    mc keys (detection-only models).
    """
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    try:
        import per_scale_decode_reference as ref
    finally:
        sys.path.pop(0)

    inter: dict[str, np.ndarray] = {}

    # ── boxes ────────────────────────────────────────────────────────
    if encoding == "dfl":
        if reg_max is None:
            raise ValueError("reg_max is required for DFL encoding")
        deq, ltrb, xywh = ref.dfl_decode_level_with_intermediates(
            box_q, box_q_scale, box_q_zp, h, w, stride, reg_max,
        )
        inter[f"intermediate.boxes_{level_index}.dequant"] = deq
        inter[f"intermediate.boxes_{level_index}.ltrb"] = ltrb
        inter[f"intermediate.boxes_{level_index}.xywh"] = xywh
    elif encoding == "ltrb":
        deq, xywh = ref.ltrb_decode_level_with_intermediates(
            box_q, box_q_scale, box_q_zp, h, w, stride,
        )
        inter[f"intermediate.boxes_{level_index}.dequant"] = deq
        inter[f"intermediate.boxes_{level_index}.xywh"] = xywh
    else:
        raise ValueError(f"unknown encoding {encoding!r}")

    # ── scores ───────────────────────────────────────────────────────
    sco_deq = (sco_q.astype(np.float32) - float(sco_q_zp)) * float(sco_q_scale)
    sco_act = ref.sigmoid(sco_deq).astype(np.float32)
    inter[f"intermediate.scores_{level_index}.dequant"] = sco_deq
    inter[f"intermediate.scores_{level_index}.activated"] = sco_act

    # ── mask coefs ───────────────────────────────────────────────────
    if mc_q is not None:
        if mc_q_scale is None or mc_q_zp is None:
            raise ValueError("mc_q_scale/mc_q_zp required when mc_q is provided")
        mc_deq = (mc_q.astype(np.float32) - float(mc_q_zp)) * float(mc_q_scale)
        inter[f"intermediate.mc_{level_index}.dequant"] = mc_deq

    return inter


def capture_protos_intermediate(
    proto_q: np.ndarray, q_scale: float, q_zp: int,
) -> dict[str, np.ndarray]:
    """Capture the proto dequant intermediate (single tensor)."""
    deq = (proto_q.astype(np.float32) - float(q_zp)) * float(q_scale)
    return {"intermediate.protos.dequant": deq}


def build_documentation_md(
    *,
    decoder_family: str,
    model_basename: str,
    encoding: str,
    keys: list[str],
    nms_config: dict,
    expected_count_min: int,
) -> str:
    """Build the embedded ``documentation_md`` metadata payload.

    The result is a self-contained Markdown document explaining what
    the fixture is for, what was decomposed from the model, and how
    each tensor key maps to a HAL kernel. It is included in the
    safetensors ``__metadata__`` and can be extracted with
    ``safetensors.safe_open(path).metadata()["documentation_md"]``.
    """
    key_table = "\n".join(f"| `{k}` | {_describe_key(k)} |" for k in sorted(keys))
    nms_lines = "\n".join(f"- `{k}`: {v}" for k, v in nms_config.items())
    test_basename = model_basename.replace("-", "_")
    return f"""# Decoder Reference Fixture — {model_basename}

**Decoder family:** `{decoder_family}`
**Box encoding:** `{encoding}`
**Smoke-test detection floor:** ≥ {expected_count_min}

## Physical output decomposition — what and why

This fixture captures the input/output behaviour of a TFLite model
whose accuracy-sensitive tail layers (DFL, sigmoid, dist2bbox, per-scale
concat) have been stripped from the graph by the EdgeFirst tflite-
converter (`quantization_split` mode). Stripping those layers preserves
each FPN level's quantization parameters separately, recovering most of
the int8 accuracy lost to monolithic quantization. The runtime then
re-implements the stripped layers in higher precision on the CPU. In
HAL, that re-implementation lives in `crates/decoder/src/per_scale/`.

This fixture is the oracle for that CPU re-implementation: it bundles
the model's raw quantized outputs together with every stage of the
NumPy reference decode and the final post-NMS detections. Rust unit
tests load the fixture, run the HAL decoder on the raw tensors, and
compare against the bundled goldens.

## Tensor keys

| Key | Meaning |
|-----|---------|
{key_table}

## NMS configuration used at fixture-generation time

{nms_lines}

## How this fixture is consumed

- End-to-end parity test:
  `crates/decoder/tests/per_scale_parity.rs::{test_basename}_per_scale_end_to_end_parity`
- Stage parity test (pre-NMS):
  `crates/decoder/tests/per_scale_parity.rs::{test_basename}_per_scale_pre_nms_parity`
- Smoke detection-count test:
  `crates/decoder/tests/per_scale_parity.rs::{test_basename}_per_scale_smoke_detection_count`

## Regenerating

```
python scripts/decoder_generate_fixture.py <model.tflite> <image.jpg> \\
    --output testdata/decoder/{model_basename}.safetensors
```
"""


def _describe_key(key: str) -> str:
    """One-line description of a fixture tensor key."""
    if key == "input.image":
        return "Pre-letterboxed input image fed to TFLite (uint8, NHWC)."
    if key == "decoded.boxes_xyxy":
        return "Post-NMS boxes in pixel coordinates (xyxy, fp32)."
    if key == "decoded.scores":
        return "Post-NMS class confidences (fp32)."
    if key == "decoded.classes":
        return "Post-NMS class indices (uint32)."
    if key == "decoded.masks":
        return "Per-detection binary segmentation mask (uint8 0/255)."
    if key.startswith("raw.boxes_"):
        return "Quantized box-channel tensor for one FPN level (NHWC)."
    if key.startswith("raw.scores_"):
        return "Quantized class-logit tensor for one FPN level (NHWC)."
    if key.startswith("raw.mc_"):
        return "Quantized mask-coefficient tensor for one FPN level (NHWC)."
    if key == "raw.protos":
        return "Quantized prototype-mask tensor (NHWC, single output)."
    if key.endswith(".dequant"):
        return "Stage 1: dequantized to fp32 (zero-point subtract + scale multiply)."
    if key.endswith(".ltrb"):
        return "Stage 2 (DFL only): per-side softmax + weighted sum, one row per anchor."
    if key.endswith(".xywh"):
        return "Stage 3: dist2bbox + stride scaling — pixel-coord boxes, one row per anchor."
    if key.endswith(".activated"):
        return "Sigmoid applied to score logits."
    return "Per-stage reference output (fp32)."


def assemble_and_write(
    tensors: dict[str, np.ndarray],
    metadata: dict[str, str],
    output_path: Path,
) -> None:
    """Write the fixture using safetensors.numpy.save_file.

    The metadata dict is stored as the safetensors ``__metadata__``
    record. All values must be ``str``. Tensor keys may use ``.``
    namespacing.
    """
    import safetensors.numpy as stnp
    output_path.parent.mkdir(parents=True, exist_ok=True)
    stnp.save_file(tensors, str(output_path), metadata=metadata)


import copy

_LOGICAL_DSHAPE_FEATURE_NAMES = {
    "boxes": ("box_coords", "num_boxes"),
    "scores": ("num_classes", "num_boxes"),
    "mask_coefs": ("num_protos", "num_boxes"),
}


def _patch_schema_dshape(metadata: dict) -> dict:
    """Synthesize missing ``dshape`` entries on logical parent outputs.

    The EdgeFirst tflite-converter emits ``dshape`` on physical
    per-scale children but not on the logical parents (``boxes``,
    ``scores``, ``mask_coefs``). HAL's ``DecoderBuilder::build`` rejects
    schemas where a logical with per-scale children lacks ``dshape``,
    so the validator patches schemas in memory before passing them to
    HAL. We do the same here at fixture-generation time so the bundled
    schema is HAL-ready out of the box.

    The patch is idempotent and non-destructive: the input metadata is
    deep-copied; entries that already carry ``dshape`` or that don't
    match the logical-with-children shape are left alone.

    Returns a new ``dict``; the input is not mutated.
    """
    patched = copy.deepcopy(metadata)
    for output in patched.get("outputs", []):
        t = output.get("type")
        shape = output.get("shape", [])
        has_children = bool(output.get("outputs"))
        if (
            t in _LOGICAL_DSHAPE_FEATURE_NAMES
            and "dshape" not in output
            and has_children
            and len(shape) == 3
        ):
            feat_name, box_name = _LOGICAL_DSHAPE_FEATURE_NAMES[t]
            output["dshape"] = [
                {"batch": shape[0]},
                {feat_name: shape[1]},
                {box_name: shape[2]},
            ]
    return patched


def _git_sha() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, check=True,
        )
        sha = out.stdout.strip()
        dirty = subprocess.run(
            ["git", "diff", "--quiet"], check=False,
        ).returncode != 0
        return f"{sha}{'-dirty' if dirty else ''}"
    except Exception:
        return "unknown"


def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def main(argv: list[str] | None = None) -> int:
    cfg = parse_args(argv)

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    try:
        import per_scale_decode_reference as ref
    finally:
        sys.path.pop(0)

    layout, raw_outputs_by_shape, image_uint8, edgefirst_metadata = (
        run_inference_collect_outputs(cfg.model_path, cfg.image_path)
    )

    raw = collect_raw_tensors_by_role(layout, raw_outputs_by_shape)
    tensors: dict[str, np.ndarray] = {"input.image": image_uint8[np.newaxis]}
    tensors.update(raw)

    # Per-stage intermediates
    if cfg.include_stages:
        for lvl_idx, lvl in enumerate(layout.fpn_levels):
            box_q_t = raw[f"raw.boxes_{lvl_idx}"]
            sco_q_t = raw[f"raw.scores_{lvl_idx}"]
            mc_q_t  = raw.get(f"raw.mc_{lvl_idx}")
            box_scale, box_zp = lvl["box_quant"]
            sco_scale, sco_zp = lvl["score_quant"]
            mc_scale = mc_zp = None
            if mc_q_t is not None and lvl.get("mc_quant") is not None:
                mc_scale, mc_zp = lvl["mc_quant"]
            inter = capture_box_score_mc_intermediates(
                level_index=lvl_idx,
                encoding=layout.box_encoding,
                reg_max=lvl.get("reg_max"),
                h=lvl["h"], w=lvl["w"], stride=int(lvl["stride"]),
                box_q=box_q_t, box_q_scale=float(box_scale), box_q_zp=int(box_zp),
                sco_q=sco_q_t, sco_q_scale=float(sco_scale), sco_q_zp=int(sco_zp),
                mc_q=mc_q_t,
                mc_q_scale=float(mc_scale) if mc_scale is not None else None,
                mc_q_zp=int(mc_zp) if mc_zp is not None else None,
            )
            tensors.update(inter)
        if "raw.protos" in raw and layout.protos_quant is not None:
            p_scale, p_zp = layout.protos_quant
            tensors.update(capture_protos_intermediate(
                raw["raw.protos"], float(p_scale), int(p_zp),
            ))

    # Build shape_map directly (avoid bind_outputs' ndarray dance)
    outputs_in_order = list(raw_outputs_by_shape.values())
    shape_to_idx = {sh: i for i, sh in enumerate(raw_outputs_by_shape.keys())}

    decoded = ref.decode_per_scale(
        outputs_in_order, layout, shape_to_idx,
        img_w=image_uint8.shape[1], img_h=image_uint8.shape[0],
        iou_threshold=cfg.iou_threshold,
        score_threshold=cfg.score_threshold,
        max_detections=300,
    )
    tensors["decoded.boxes_xyxy"] = decoded["boxes_xyxy"].astype(np.float32)
    tensors["decoded.scores"] = decoded["scores"].astype(np.float32)
    tensors["decoded.classes"] = decoded["classes"].astype(np.uint32)
    if decoded.get("masks") is not None and decoded["masks"].size > 0:
        tensors["decoded.masks"] = (decoded["masks"] > 0).astype(np.uint8) * 255

    # Build quantization map keyed by lookup name (matches loader's key shape)
    quant_map: dict[str, dict] = {}
    for lvl_idx, lvl in enumerate(layout.fpn_levels):
        if lvl.get("box_quant") is not None:
            scale, zp = lvl["box_quant"]
            quant_map[f"boxes_{lvl_idx}"] = {
                "scale": float(scale), "zero_point": int(zp), "dtype": "int8",
            }
        if lvl.get("score_quant") is not None:
            scale, zp = lvl["score_quant"]
            quant_map[f"scores_{lvl_idx}"] = {
                "scale": float(scale), "zero_point": int(zp), "dtype": "int8",
            }
        if lvl.get("mc_quant") is not None:
            scale, zp = lvl["mc_quant"]
            quant_map[f"mc_{lvl_idx}"] = {
                "scale": float(scale), "zero_point": int(zp), "dtype": "int8",
            }
    if layout.protos_quant is not None:
        scale, zp = layout.protos_quant
        quant_map["protos"] = {
            "scale": float(scale), "zero_point": int(zp), "dtype": "int8",
        }

    nms_config = {
        "iou_threshold": cfg.iou_threshold,
        "score_threshold": cfg.score_threshold,
        "max_detections": 300,
    }

    documentation_md = build_documentation_md(
        decoder_family="per_scale_yolo_seg",
        model_basename=cfg.output_path.stem,
        encoding=layout.box_encoding,
        keys=list(tensors.keys()),
        nms_config=nms_config,
        expected_count_min=cfg.expected_count_min,
    )

    metadata: dict[str, str] = {
        "format_version": "1",
        "decoder_family": "per_scale_yolo_seg",
        "model_basename": cfg.output_path.stem,
        "model_path": str(cfg.model_path),
        "image_path": str(cfg.image_path),
        "schema_json": json.dumps(_patch_schema_dshape(edgefirst_metadata)),
        "quantization_json": json.dumps(quant_map),
        "nms_config_json": json.dumps(nms_config),
        "expected_count_min": str(cfg.expected_count_min),
        "generated_at": datetime.datetime.utcnow().isoformat() + "Z",
        "reference_script_sha256": _file_sha256(
            Path(__file__).resolve().parent / "per_scale_decode_reference.py"),
        "generator_git_sha": _git_sha(),
        "documentation_md": documentation_md,
    }

    n_det = len(tensors["decoded.boxes_xyxy"])
    if n_det < cfg.expected_count_min:
        print(
            f"WARNING: only {n_det} detections, "
            f"below expected_count_min={cfg.expected_count_min}",
            file=sys.stderr,
        )

    assemble_and_write(tensors, metadata, cfg.output_path)
    size = cfg.output_path.stat().st_size
    print(
        f"wrote {cfg.output_path} ({size} bytes, {len(tensors)} tensors, "
        f"{n_det} detections)",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
