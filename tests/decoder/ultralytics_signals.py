# SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
# SPDX-License-Identifier: Apache-2.0

"""Extract model I/O signals + metadata from Ultralytics exports.

Shared by `scripts/capture_infer_fixtures.py` and `test_ultralytics_discovery.py`.
"""

import zipfile
from pathlib import Path


def onnx_signals(path):
    """Reads I/O signals from an ONNX export via onnxruntime.

    Every tensor is reported as float32 with no quantization.
    """
    import onnxruntime as ort

    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    meta = sess.get_modelmeta().custom_metadata_map

    def t(io):
        return {
            "name": io.name,
            "shape": io.shape,
            "dtype": "float32",
            "quantization": None,
        }

    return {
        "source": "onnx",
        "inputs": [t(i) for i in sess.get_inputs()],
        "outputs": [t(o) for o in sess.get_outputs()],
        "metadata": dict(meta),
    }


def tflite_signals(path):
    """Reads I/O signals from a TFLite/LiteRT export via the interpreter."""
    try:
        from ai_edge_litert.interpreter import Interpreter
    except ImportError:
        from tensorflow.lite import Interpreter

    interp = Interpreter(model_path=str(path))
    interp.allocate_tensors()

    def t(d):
        scale, zp = d["quantization"]
        q = None if scale == 0.0 else {"scale": [scale], "zero_point": [int(zp)]}
        return {
            "name": d["name"],
            "shape": [int(x) for x in d["shape"]],
            "dtype": str(d["dtype"].__name__),
            "quantization": q,
        }

    # TFLite metadata associated files ride in a zip appended to the
    # flatbuffer; an export without one yields an empty map.
    meta = {}
    try:
        with zipfile.ZipFile(path) as z:
            for n in z.namelist():
                meta[n] = z.read(n).decode("utf-8", errors="replace")
    except zipfile.BadZipFile:
        pass

    return {
        "source": "tflite",
        "inputs": [t(d) for d in interp.get_input_details()],
        "outputs": [t(d) for d in interp.get_output_details()],
        "metadata": meta,
    }


def signals_for(path: Path) -> dict:
    """Dispatches to the ONNX or TFLite reader by file extension."""
    return onnx_signals(path) if path.suffix == ".onnx" else tflite_signals(path)
