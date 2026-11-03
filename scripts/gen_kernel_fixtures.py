#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
# SPDX-License-Identifier: Apache-2.0
"""Generate single-op TFLite reference fixtures for the edgefirst-kernels parity tests.

Each fixture is one quantized SOFTMAX or LOGISTIC op built with the TFLite
schema object API and run through the LiteRT interpreter. The seeded input and
the interpreter's output are stored in a ``.safetensors`` file whose metadata
carries the op, dtype and quantization parameters.

Usage: python -I scripts/gen_kernel_fixtures.py <output-dir>
"""

from __future__ import annotations

import importlib.metadata
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import flatbuffers
import numpy as np
from ai_edge_litert import schema_py_generated as schema
from ai_edge_litert.interpreter import Interpreter
from safetensors.numpy import save_file

SEED = 2348
TFLITE_VERSION = 3

TENSOR_TYPE = {
    "int8": schema.TensorType.INT8,
    "uint8": schema.TensorType.UINT8,
    "int16": schema.TensorType.INT16,
}
OP_VERSION = {("SOFTMAX", "int8"): 2, ("SOFTMAX", "int16"): 3}


@dataclass(frozen=True)
class Spec:
    name: str
    op: str
    dtype: str
    shape: tuple[int, ...]
    in_scale: float
    in_zp: int
    out_scale: float
    out_zp: int
    beta: float = 1.0


def _softmax(name, dtype, shape, in_scale, in_zp, out_scale, out_zp):
    return Spec(name, "SOFTMAX", dtype, shape, in_scale, in_zp, out_scale, out_zp)


def _logistic(name, dtype, in_scale, in_zp, out_scale, out_zp):
    shape = (1, 80, 8400)
    return Spec(name, "LOGISTIC", dtype, shape, in_scale, in_zp, out_scale, out_zp)


I8_OUT = (1 / 256, -128)
I16_OUT = (1 / 32768, 0)

SPECS = [
    _softmax("softmax_i8_yolo11_attn", "int8", (1, 2, 400, 400), 0.128, -15, *I8_OUT),
    _softmax("softmax_i8_yolo26_attn", "int8", (1, 2, 1008, 1008), 0.128, -15, *I8_OUT),
    _softmax("softmax_i8_odd_rows_1009", "int8", (3, 1009), 0.05, 3, *I8_OUT),
    _softmax("softmax_i8_odd_rows_17", "int8", (5, 17), 0.05, 3, *I8_OUT),
    _softmax("softmax_i8_odd_rows_1", "int8", (7, 1), 0.05, 3, *I8_OUT),
    _softmax("softmax_i8_dfl", "int8", (1, 4, 8400, 16), 0.11, -20, *I8_OUT),
    _softmax("softmax_i16_attn", "int16", (1, 2, 400, 400), 1 / 2048, 0, *I16_OUT),
    _softmax("softmax_i16_odd_rows_1009", "int16", (3, 1009), 1 / 2048, 0, *I16_OUT),
    _softmax("softmax_i16_odd_rows_1", "int16", (7, 1), 1 / 2048, 0, *I16_OUT),
    _softmax("softmax_i16_scale_small", "int16", (6, 257), 1e-5, 0, *I16_OUT),
    _softmax("softmax_i16_scale_large", "int16", (6, 257), 4.5, 0, *I16_OUT),
    _logistic("logistic_i8_scores", "int8", 0.1, -30, *I8_OUT),
    _logistic("logistic_u8_scores", "uint8", 0.1, 100, 1 / 256, 0),
    _logistic("logistic_i16_nonpot", "int16", 0.0005, 0, *I16_OUT),
    _logistic("logistic_i16_pot", "int16", 2.0**-12, 0, *I16_OUT),
]


def _tensor(name, spec, scale, zero_point, buffer):
    q = schema.QuantizationParametersT()
    q.scale = [scale]
    q.zeroPoint = [zero_point]
    t = schema.TensorT()
    t.name = name
    t.shape = list(spec.shape)
    t.type = TENSOR_TYPE[spec.dtype]
    t.buffer = buffer
    t.quantization = q
    return t


def build_model(spec: Spec) -> bytes:
    """Serialize a one-operator quantized model for ``spec``."""
    code = schema.OperatorCodeT()
    builtin = getattr(schema.BuiltinOperator, spec.op)
    code.builtinCode = builtin
    code.deprecatedBuiltinCode = builtin
    code.version = OP_VERSION.get((spec.op, spec.dtype), 2)

    op = schema.OperatorT()
    op.opcodeIndex = 0
    op.inputs = [0]
    op.outputs = [1]
    if spec.op == "SOFTMAX":
        op.builtinOptionsType = schema.BuiltinOptions.SoftmaxOptions
        opts = schema.SoftmaxOptionsT()
        opts.beta = spec.beta
        op.builtinOptions = opts

    sub = schema.SubGraphT()
    sub.tensors = [
        _tensor("input", spec, spec.in_scale, spec.in_zp, 1),
        _tensor("output", spec, spec.out_scale, spec.out_zp, 2),
    ]
    sub.inputs = [0]
    sub.outputs = [1]
    sub.operators = [op]

    model = schema.ModelT()
    model.version = TFLITE_VERSION
    model.operatorCodes = [code]
    model.subgraphs = [sub]
    model.buffers = [schema.BufferT() for _ in range(3)]
    model.description = spec.name

    builder = flatbuffers.Builder(1024)
    builder.Finish(model.Pack(builder), file_identifier=b"TFL3")
    return bytes(builder.Output())


def make_input(spec: Spec) -> np.ndarray:
    """Seeded uniform input over the full integer range, with two forced rows.

    Row 0 (last axis) is all-equal and row 1 holds both type extremes.
    """
    info = np.iinfo(spec.dtype)
    seed = [SEED, sum(spec.name.encode())]
    rng = np.random.default_rng(seed)
    x = rng.integers(info.min, int(info.max) + 1, size=spec.shape, dtype=spec.dtype)
    rows = x.reshape(-1, spec.shape[-1])
    rows[0, :] = rows[0, 0]
    if rows.shape[0] > 1 and rows.shape[1] > 1:
        rows[1, 0] = info.min
        rows[1, 1] = info.max
    return x


def run_model(model: bytes, x: np.ndarray) -> np.ndarray:
    interp = Interpreter(model_content=model)
    interp.allocate_tensors()
    interp.set_tensor(interp.get_input_details()[0]["index"], x)
    interp.invoke()
    return interp.get_tensor(interp.get_output_details()[0]["index"]).copy()


def write_fixture(spec: Spec, out_dir: Path, litert_version: str) -> Path:
    x = make_input(spec)
    y = run_model(build_model(spec), x)
    meta = {
        "op": spec.op,
        "dtype": spec.dtype,
        "input_scale": repr(float(np.float32(spec.in_scale))),
        "input_zero_point": str(spec.in_zp),
        "output_scale": repr(float(np.float32(spec.out_scale))),
        "output_zero_point": str(spec.out_zp),
        "shape": json.dumps(list(spec.shape)),
        "litert_version": litert_version,
    }
    if spec.op == "SOFTMAX":
        meta["beta"] = repr(spec.beta)
    path = out_dir / f"{spec.name}.safetensors"
    save_file({"input": x, "output": y}, str(path), metadata=meta)
    return path


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print(__doc__, file=sys.stderr)
        return 2
    out_dir = Path(argv[1])
    out_dir.mkdir(parents=True, exist_ok=True)
    version = importlib.metadata.version("ai-edge-litert")
    for spec in SPECS:
        path = write_fixture(spec, out_dir, version)
        print(f"{path} ({path.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
