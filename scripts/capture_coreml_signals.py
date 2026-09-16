#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright 2025-2026 Au-Zone Technologies
# SPDX-License-Identifier: Apache-2.0
"""Capture a CoreML model's I/O signals into a decoder test fixture.

Mirrors the ONNX/TFLite capture used for the existing
``crates/decoder/testdata/infer/*.signals.json`` fixtures: shapes, dtypes
and the flat string metadata map exactly as an inference runtime would
report them to ``infer_ultralytics_schema``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import coremltools as ct

# CoreML ArrayFeatureType.ArrayDataType -> decoder schema dtype names.
_ARRAY_DTYPE = {
    65552: "float32",  # FLOAT32
    65568: "float64",  # DOUBLE
    131104: "int32",   # INT32
    65550: "float16",  # FLOAT16
}


def _tensor(name: str, feature) -> dict:
    kind = feature.WhichOneof("Type")
    if kind == "multiArrayType":
        ma = feature.multiArrayType
        dtype = _ARRAY_DTYPE.get(ma.dataType)
        if dtype is None:
            raise SystemExit(f"unmapped CoreML array dataType {ma.dataType} on '{name}'")
        return {
            "name": name,
            "shape": list(ma.shape),
            "dtype": dtype,
            "quantization": None,
        }
    if kind == "imageType":
        im = feature.imageType
        raise SystemExit(
            f"'{name}' is an imageType ({im.width}x{im.height}); the profiler's "
            "native CoreML engine consumes a float16 multiArray. Re-export with "
            "tools/export_fp16.py before capturing a fixture."
        )
    raise SystemExit(f"unsupported CoreML feature type '{kind}' on '{name}'")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("model", type=Path, help="path to a .mlpackage or .mlmodel")
    ap.add_argument("-o", "--output", type=Path, required=True)
    args = ap.parse_args()

    spec = ct.models.MLModel(str(args.model), skip_model_load=True).get_spec()
    desc = spec.description

    signals = {
        "source": "coreml",
        "inputs": [_tensor(i.name, i.type) for i in desc.input],
        "outputs": [_tensor(o.name, o.type) for o in desc.output],
        "metadata": {
            k: str(v) for k, v in spec.description.metadata.userDefined.items()
        },
    }
    args.output.write_text(json.dumps(signals, indent=1) + "\n")
    print(f"wrote {args.output}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
