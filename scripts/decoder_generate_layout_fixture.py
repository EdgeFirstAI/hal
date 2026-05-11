#!/usr/bin/env python3
"""Layout-agnostic decoder reference fixture generator.

PURPOSE
-------
Capture a TFLite model's raw outputs + embedded schema into a single
``.safetensors`` file for HAL decoder unit tests. Unlike the
``decoder_generate_fixture.py`` companion script, this one supports
*all three* converter output layouts produced by the EdgeFirst
tflite-converter:

- **combined** (single fused detection tensor + protos)
- **logical** (separate boxes / scores / mask_coefs / protos)
- **smart** / per-scale (per-stride FPN children with DFL)

The companion ``decoder_generate_fixture.py`` is per-scale-only and
captures per-stage intermediates (DFL / sigmoid / dist2bbox) for
parity testing. This script is layout-agnostic and captures **only
the model outputs** plus the embedded schema, so HAL tests can drive
the decoder against bit-exact real-world outputs without running an
inference runtime.

FIXTURE LAYOUT
--------------
- ``raw.<output_name>`` — one tensor per TFLite output, named after
  the ``LogicalOutput.name`` field of the embedded schema (or the
  TFLite tensor name when no schema match is found).
- ``image.uint8`` — the (1, H, W, 3) NHWC uint8 image fed to the
  model after a plain `Image.resize` (no letterbox), shape
  matches the model's declared input shape.
- ``__metadata__["edgefirst_json"]`` — the exact JSON string
  extracted from the model's ZIP trailer.
- ``__metadata__["model_sha256"]`` — content hash of the input
  model file.
- ``__metadata__["image_sha256"]`` — content hash of the input
  image file.
- ``__metadata__["model_input_shape"]`` — JSON-encoded
  ``[N, H, W, C]``.
- ``__metadata__["output_quant.<name>"]`` — JSON-encoded
  ``{"scale": float, "zero_point": int, "dtype": str, "shape": list}``
  per output.
- ``__metadata__["timestamp_utc"]`` — ISO 8601 generation time.

USAGE
-----
::

    python scripts/decoder_generate_layout_fixture.py \\
        path/to/yolov8n-seg-t-2681_quant-u8-i8_combined.tflite \\
        testdata/zidane.jpg \\
        --output testdata/decoder/yolov8n-seg-t-2681_combined.safetensors

REPRODUCIBILITY
---------------
Pin numpy and safetensors versions::

    pip install 'numpy>=1.26' 'safetensors>=0.4' 'pillow>=10' 'tflite_runtime>=2.14'

Falls back to ``tensorflow.lite`` when ``tflite_runtime`` is unavailable.
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import io
import json
import sys
import zipfile
from pathlib import Path

import numpy as np


def sha256_hex(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def extract_edgefirst_json(model_bytes: bytes) -> str:
    """Read ``edgefirst.json`` from the ZIP archive appended to the model."""
    try:
        with zipfile.ZipFile(io.BytesIO(model_bytes)) as zf:
            with zf.open("edgefirst.json") as fp:
                return fp.read().decode("utf-8")
    except (zipfile.BadZipFile, KeyError) as e:
        raise SystemExit(
            f"model has no embedded edgefirst.json archive: {e}"
        ) from e


def build_shape_to_logical_name(schema: dict) -> dict[tuple[int, ...], str]:
    """Map physical-tensor shapes -> the schema name they realise.

    For flat outputs (no `outputs` children) the logical's name applies
    directly. For per-scale outputs (with children), each child carries
    its own name and shape; we register the children, not the parent.
    """
    table: dict[tuple[int, ...], str] = {}
    for entry in schema.get("outputs", []):
        children = entry.get("outputs") or []
        if children:
            for child in children:
                shape = tuple(child.get("shape", []))
                name = child.get("name") or entry.get("name") or "unknown"
                if shape:
                    table[shape] = name
        else:
            shape = tuple(entry.get("shape", []))
            name = entry.get("name") or "unknown"
            if shape:
                table[shape] = name
    return table


def main() -> None:
    p = argparse.ArgumentParser(
        prog="decoder_generate_layout_fixture",
        description="Capture raw TFLite outputs + schema for HAL decoder fixtures.",
    )
    p.add_argument("model", type=Path, help="Path to .tflite with embedded edgefirst.json.")
    p.add_argument("image", type=Path, help="Representative input image.")
    p.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Output .safetensors path.",
    )
    args = p.parse_args()

    if not args.model.exists():
        raise SystemExit(f"model not found: {args.model}")
    if not args.image.exists():
        raise SystemExit(f"image not found: {args.image}")

    model_bytes = args.model.read_bytes()
    edgefirst_json_str = extract_edgefirst_json(model_bytes)
    schema = json.loads(edgefirst_json_str)

    # Lazy import so the script's --help doesn't require TF.
    try:
        from tflite_runtime.interpreter import Interpreter  # type: ignore
    except ImportError:
        from tensorflow.lite.python.interpreter import Interpreter  # type: ignore
    from PIL import Image
    import safetensors.numpy

    interp = Interpreter(model_path=str(args.model))
    interp.allocate_tensors()
    in_det = interp.get_input_details()[0]
    # interp returns numpy ints; cast to Python int so json serialises.
    in_shape = [int(d) for d in in_det["shape"]]  # [N, H, W, C]
    n, h, w, c = in_shape
    if n != 1 or c not in (1, 3, 4):
        raise SystemExit(f"unexpected input shape {in_shape}")

    # Plain resize (no letterbox) — keeps the fixture independent of
    # any host-side preprocessing choice. Tests that need letterbox
    # outputs should generate their own image array.
    img = Image.open(args.image).convert("RGB").resize((w, h))
    img_np = np.array(img, dtype=np.uint8)
    interp.set_tensor(in_det["index"], img_np[np.newaxis])
    interp.invoke()

    shape_to_name = build_shape_to_logical_name(schema)
    raw: dict[str, np.ndarray] = {
        "image.uint8": img_np[np.newaxis].copy(),
    }
    output_quant: dict[str, dict] = {}
    used_names: set[str] = set()
    for od in interp.get_output_details():
        out_shape = tuple(od["shape"].tolist())
        # Prefer the schema's own naming so HAL tests can match by
        # `LogicalOutput.name` (or `PhysicalOutput.name` for per-scale
        # children). Fall back to the TFLite tensor name with `:` and
        # `/` sanitised, then to a positional fallback.
        name = shape_to_name.get(out_shape)
        if name is None:
            tflite_name = od.get("name") or f"output_{od['index']}"
            name = tflite_name.replace("/", "_").replace(":", "_")
        if name in used_names:
            # Two outputs with the same shape: disambiguate by index.
            name = f"{name}_{od['index']}"
        used_names.add(name)

        tensor = interp.get_tensor(od["index"]).copy()
        raw[f"raw.{name}"] = tensor
        scale, zp = od["quantization"]
        output_quant[name] = {
            "scale": float(scale),
            "zero_point": int(zp),
            "dtype": str(tensor.dtype),
            "shape": list(out_shape),
        }

    metadata = {
        "edgefirst_json": edgefirst_json_str,
        "model_sha256": sha256_hex(args.model),
        "image_sha256": sha256_hex(args.image),
        "model_input_shape": json.dumps(in_shape),
        "decoder_version": str(schema.get("decoder_version", "")),
        "schema_version": str(schema.get("schema_version", 1)),
        "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "documentation_md": (
            "## Layout-agnostic raw-outputs fixture\n\n"
            "Generated by ``scripts/decoder_generate_layout_fixture.py``. "
            "Captures the raw TFLite outputs of an EdgeFirst-converted "
            "model run on a representative image. Tests load this "
            "fixture, attach quantization to each tensor, and drive the "
            "HAL ``Decoder`` against the raw outputs without an "
            "inference runtime.\n\n"
            "Use ``__metadata__['edgefirst_json']`` to reconstruct the "
            "schema. Use ``__metadata__['output_quant.<name>']`` for "
            "each output's ``(scale, zero_point, dtype, shape)``.\n"
        ),
    }
    for name, info in output_quant.items():
        metadata[f"output_quant.{name}"] = json.dumps(info)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    safetensors.numpy.save_file(raw, str(args.output), metadata=metadata)
    rel_keys = ", ".join(sorted(k for k in raw if k.startswith("raw.")))
    size_mb = args.output.stat().st_size / (1 << 20)
    print(f"WROTE {args.output} ({size_mb:.2f} MB)")
    print(f"  outputs: {rel_keys}")
    print(f"  schema_version={schema.get('schema_version', 1)} "
          f"decoder={schema.get('decoder_version')} "
          f"image={img_np.shape}")


if __name__ == "__main__":
    main()
