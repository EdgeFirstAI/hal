#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
# SPDX-License-Identifier: Apache-2.0
"""Generate TFLite reference fixtures for the edgefirst-kernels parity tests.

Each fixture is a small quantized graph built with the TFLite schema object
API and run through the LiteRT interpreter: one SOFTMAX or LOGISTIC op, one
int8 ADD, SUB, MUL, BATCH_MATMUL or 1x1 CONV_2D, an attention chain
(MUL -> BATCH_MATMUL -> SOFTMAX -> BATCH_MATMUL) or a DFL head
(SOFTMAX -> CONV_2D). The seeded inputs, constant operands and the
interpreter's output are stored in a ``.safetensors`` file whose metadata
carries the op, the quantization parameters and the op resolver.

The single-op SOFTMAX and LOGISTIC fixtures run on the default resolver. The
int8 arithmetic and matrix fixtures run without the default delegates, so the
output is TFLite's own builtin kernels and not XNNPACK's: ``kernels`` is
``optimized`` (``BUILTIN_WITHOUT_DEFAULT_DELEGATES``) or ``reference``
(``BUILTIN_REF``, the ``_ref`` fixtures).

Usage: python -I scripts/gen_kernel_fixtures.py <output-dir>
"""

from __future__ import annotations

import importlib.metadata
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

import flatbuffers
import numpy as np
from ai_edge_litert import schema_py_generated as schema
from ai_edge_litert.interpreter import Interpreter, OpResolverType
from safetensors.numpy import save_file

SEED = 2348
TFLITE_VERSION = 3

TENSOR_TYPE = {
    "int8": schema.TensorType.INT8,
    "uint8": schema.TensorType.UINT8,
    "int16": schema.TensorType.INT16,
    "int32": schema.TensorType.INT32,
}
OP_VERSION = {
    ("SOFTMAX", "int8"): 2,
    ("SOFTMAX", "int16"): 3,
    ("ADD", "int8"): 2,
    ("SUB", "int8"): 2,
    ("MUL", "int8"): 2,
    ("BATCH_MATMUL", "int8"): 2,
    ("CONV_2D", "int8"): 3,
}
ACTIVATION = {
    "none": schema.ActivationFunctionType.NONE,
    "relu": schema.ActivationFunctionType.RELU,
    "relu_n1_to_1": schema.ActivationFunctionType.RELU_N1_TO_1,
    "relu6": schema.ActivationFunctionType.RELU6,
}
RESOLVER = {
    "optimized": OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES,
    "reference": OpResolverType.BUILTIN_REF,
}


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


def _quant(scale, zero_point, axis=0):
    q = schema.QuantizationParametersT()
    q.scale = [float(s) for s in np.atleast_1d(scale)]
    q.zeroPoint = [int(z) for z in np.atleast_1d(zero_point)]
    q.quantizedDimension = axis
    return q


def _tensor(name, spec, scale, zero_point, buffer):
    t = schema.TensorT()
    t.name = name
    t.shape = list(spec.shape)
    t.type = TENSOR_TYPE[spec.dtype]
    t.buffer = buffer
    t.quantization = _quant(scale, zero_point)
    return t


def _finish(model) -> bytes:
    builder = flatbuffers.Builder(1024)
    builder.Finish(model.Pack(builder), file_identifier=b"TFL3")
    return bytes(builder.Output())


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
    return _finish(model)


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


# Multi-tensor graphs: int8 arithmetic, matrix products and fused chains.


def _f32(x: float) -> float:
    return float(np.float32(x))


@dataclass
class Graph:
    """A one-subgraph TFLite model under construction."""

    tensors: list = field(default_factory=list)
    buffers: list = field(default_factory=lambda: [schema.BufferT()])
    codes: list = field(default_factory=list)
    operators: list = field(default_factory=list)

    def tensor(self, name, shape, dtype, scale, zero_point, data=None, axis=0) -> int:
        t = schema.TensorT()
        t.name = name
        t.shape = list(shape)
        t.type = TENSOR_TYPE[dtype]
        t.quantization = _quant(scale, zero_point, axis)
        b = schema.BufferT()
        if data is not None:
            b.data = np.ascontiguousarray(data, dtype=dtype).view(np.uint8).reshape(-1)
        t.buffer = len(self.buffers)
        self.buffers.append(b)
        self.tensors.append(t)
        return len(self.tensors) - 1

    def op(self, name, dtype, inputs, outputs, options_type=None, options=None):
        builtin = getattr(schema.BuiltinOperator, name)
        key = (builtin, OP_VERSION[(name, dtype)])
        if key not in self.codes:
            self.codes.append(key)
        o = schema.OperatorT()
        o.opcodeIndex = self.codes.index(key)
        o.inputs = list(inputs)
        o.outputs = list(outputs)
        if options_type is not None:
            o.builtinOptionsType = options_type
            o.builtinOptions = options
        self.operators.append(o)

    def build(self, inputs, outputs, description) -> bytes:
        sub = schema.SubGraphT()
        sub.tensors = self.tensors
        sub.inputs = list(inputs)
        sub.outputs = list(outputs)
        sub.operators = self.operators
        model = schema.ModelT()
        model.version = TFLITE_VERSION
        codes = []
        for builtin, version in self.codes:
            c = schema.OperatorCodeT()
            c.builtinCode = builtin
            c.deprecatedBuiltinCode = builtin
            c.version = version
            codes.append(c)
        model.operatorCodes = codes
        model.subgraphs = [sub]
        model.buffers = self.buffers
        model.description = description
        return _finish(model)


def _rng(name: str) -> np.random.Generator:
    return np.random.default_rng([SEED, sum(name.encode())])


def _uniform_i8(rng, shape) -> np.ndarray:
    return rng.integers(-128, 128, size=shape, dtype=np.int8)


def _layer_i8(rng, shape, sigma=30.0) -> np.ndarray:
    """int8 values spread like a layer output: mostly small, a few saturated."""
    return np.clip(np.rint(rng.normal(0.0, sigma, size=shape)), -128, 127).astype(
        np.int8
    )


def _activation_options(cls, activation):
    o = cls()
    o.fusedActivationFunction = ACTIVATION[activation]
    if hasattr(o, "potScaleInt16"):
        o.potScaleInt16 = False
    return o


def _run(model: bytes, inputs: list[np.ndarray], kernels: str) -> np.ndarray:
    interp = Interpreter(
        model_content=model, experimental_op_resolver_type=RESOLVER[kernels]
    )
    interp.allocate_tensors()
    for detail, x in zip(interp.get_input_details(), inputs):
        interp.set_tensor(detail["index"], x)
    interp.invoke()
    return interp.get_tensor(interp.get_output_details()[0]["index"]).copy()


def _qmeta(meta, role, q):
    meta[f"{role}_scale"] = repr(_f32(q[0]))
    meta[f"{role}_zero_point"] = str(int(q[1]))


@dataclass(frozen=True)
class Binary:
    """An int8 ADD, SUB or MUL of ``lhs`` and ``rhs`` (``rhs`` broadcast when its shape differs)."""

    name: str
    op: str
    lhs_shape: tuple[int, ...]
    rhs_shape: tuple[int, ...]
    lhs: tuple[float, int]
    rhs: tuple[float, int]
    output: tuple[float, int]
    activation: str = "none"
    kernels: str = "optimized"
    every_pair: bool = False

    def write(self, out_dir: Path, litert_version: str) -> Path:
        rng = _rng(self.name)
        if self.every_pair:
            v = np.arange(-128, 128, dtype=np.int8)
            x = np.repeat(v, 256).reshape(self.lhs_shape)
            y = np.tile(v, 256).reshape(self.rhs_shape)
        else:
            x = _uniform_i8(rng, self.lhs_shape)
            y = _uniform_i8(rng, self.rhs_shape)
            if x.size > 1:
                x.reshape(-1)[:2] = [-128, 127]
        g = Graph()
        a = g.tensor("lhs", self.lhs_shape, "int8", *self.lhs)
        b = g.tensor("rhs", self.rhs_shape, "int8", *self.rhs)
        o = g.tensor("output", self.lhs_shape, "int8", *self.output)
        opts = {
            "ADD": (schema.BuiltinOptions.AddOptions, schema.AddOptionsT),
            "SUB": (schema.BuiltinOptions.SubOptions, schema.SubOptionsT),
            "MUL": (schema.BuiltinOptions.MulOptions, schema.MulOptionsT),
        }[self.op]
        g.op(
            self.op,
            "int8",
            [a, b],
            [o],
            opts[0],
            _activation_options(opts[1], self.activation),
        )
        out = _run(g.build([a, b], [o], self.name), [x, y], self.kernels)
        meta = {
            "op": self.op,
            "kernels": self.kernels,
            "activation": self.activation,
            "litert_version": litert_version,
        }
        for role, q in (("lhs", self.lhs), ("rhs", self.rhs), ("output", self.output)):
            _qmeta(meta, role, q)
        path = out_dir / f"{self.name}.safetensors"
        save_file({"lhs": x, "rhs": y, "output": out}, str(path), metadata=meta)
        return path


@dataclass(frozen=True)
class BatchMatMul:
    """An int8 BATCH_MATMUL; ``lhs`` and ``rhs`` are stored as TFLite reads them (before adj)."""

    name: str
    lhs_shape: tuple[int, ...]
    rhs_shape: tuple[int, ...]
    lhs: tuple[float, int]
    rhs: tuple[float, int]
    output: tuple[float, int]
    adj_x: bool = False
    adj_y: bool = False
    kernels: str = "optimized"

    def write(self, out_dir: Path, litert_version: str) -> Path:
        rng = _rng(self.name)
        x = _layer_i8(rng, self.lhs_shape, 40.0)
        y = _layer_i8(rng, self.rhs_shape, 40.0)
        m = self.lhs_shape[-1] if self.adj_x else self.lhs_shape[-2]
        n = self.rhs_shape[-2] if self.adj_y else self.rhs_shape[-1]
        out_shape = (*self.lhs_shape[:-2], m, n)
        g = Graph()
        a = g.tensor("lhs", self.lhs_shape, "int8", *self.lhs)
        b = g.tensor("rhs", self.rhs_shape, "int8", *self.rhs)
        o = g.tensor("output", out_shape, "int8", *self.output)
        opts = schema.BatchMatMulOptionsT()
        opts.adjX, opts.adjY, opts.asymmetricQuantizeInputs = (
            self.adj_x,
            self.adj_y,
            False,
        )
        g.op(
            "BATCH_MATMUL",
            "int8",
            [a, b],
            [o],
            schema.BuiltinOptions.BatchMatMulOptions,
            opts,
        )
        out = _run(g.build([a, b], [o], self.name), [x, y], self.kernels)
        meta = {
            "op": "BATCH_MATMUL",
            "kernels": self.kernels,
            "adj_x": str(self.adj_x).lower(),
            "adj_y": str(self.adj_y).lower(),
            "litert_version": litert_version,
        }
        for role, q in (("lhs", self.lhs), ("rhs", self.rhs), ("output", self.output)):
            _qmeta(meta, role, q)
        path = out_dir / f"{self.name}.safetensors"
        save_file({"lhs": x, "rhs": y, "output": out}, str(path), metadata=meta)
        return path


def _conv_1x1(
    g, name, inp, in_q, cin, cout, out_shape, out_q, filt, filter_scales, bias, act
):
    """Adds a 1x1, stride-1, VALID per-channel int8 CONV_2D reading tensor ``inp``."""
    f = g.tensor(
        f"{name}_filter",
        (cout, 1, 1, cin),
        "int8",
        filter_scales,
        np.zeros(cout, dtype=np.int64),
        data=filt.reshape(cout, 1, 1, cin),
    )
    bias_scales = [_f32(in_q[0]) * _f32(s) for s in filter_scales]
    bq = g.tensor(
        f"{name}_bias", (cout,), "int32", bias_scales, np.zeros(cout), data=bias
    )
    o = g.tensor(f"{name}_output", out_shape, "int8", *out_q)
    opts = schema.Conv2DOptionsT()
    opts.padding = schema.Padding.VALID
    opts.strideW = opts.strideH = 1
    opts.dilationWFactor = opts.dilationHFactor = 1
    opts.fusedActivationFunction = ACTIVATION[act]
    g.op(
        "CONV_2D", "int8", [inp, f, bq], [o], schema.BuiltinOptions.Conv2DOptions, opts
    )
    return o


@dataclass(frozen=True)
class Conv1x1:
    """An int8 1x1 CONV_2D (NHWC input, per-channel filter scales, int32 bias, fused activation)."""

    name: str
    input_shape: tuple[int, int, int, int]
    cout: int
    input: tuple[float, int]
    output: tuple[float, int]
    activation: str
    kernels: str = "optimized"

    def write(self, out_dir: Path, litert_version: str) -> Path:
        rng = _rng(self.name)
        cin = self.input_shape[-1]
        x = _layer_i8(rng, self.input_shape, 45.0)
        filt = _layer_i8(rng, (self.cout, cin), 50.0)
        filter_scales = [_f32(s) for s in rng.uniform(0.002, 0.02, size=self.cout)]
        bias = rng.integers(-20000, 20000, size=self.cout, dtype=np.int32)
        out_shape = (*self.input_shape[:-1], self.cout)
        g = Graph()
        i = g.tensor("input", self.input_shape, "int8", *self.input)
        o = _conv_1x1(
            g,
            "conv",
            i,
            self.input,
            cin,
            self.cout,
            out_shape,
            self.output,
            filt,
            filter_scales,
            bias,
            self.activation,
        )
        out = _run(g.build([i], [o], self.name), [x], self.kernels)
        meta = {
            "op": "CONV_2D",
            "kernels": self.kernels,
            "activation": self.activation,
            "filter_scales": json.dumps(filter_scales),
            "litert_version": litert_version,
        }
        _qmeta(meta, "input", self.input)
        _qmeta(meta, "output", self.output)
        path = out_dir / f"{self.name}.safetensors"
        tensors = {"input": x, "filter": filt, "bias": bias, "output": out}
        save_file(tensors, str(path), metadata=meta)
        return path


@dataclass(frozen=True)
class Attention:
    """MUL (Q by a constant scalar) -> BATCH_MATMUL (adj_y) -> SOFTMAX -> BATCH_MATMUL.

    Q, K and V are ``[1, heads, tokens, dim]``; the output ``[1, heads, tokens, dv]``.
    """

    name: str
    heads: int
    tokens: int
    dk: int
    dv: int
    kernels: str = "optimized"
    x: tuple[float, int] = (0.1251, -15)
    mul_rhs: tuple[float, int] = (0.000_693_2, -128)
    mul_rhs_value: int = 127
    q_scaled: tuple[float, int] = (0.010_18, 6)
    scores: tuple[float, int] = (0.1279, -9)
    probs: tuple[float, int] = I8_OUT
    output: tuple[float, int] = (0.063_33, 2)

    def write(self, out_dir: Path, litert_version: str) -> Path:
        rng = _rng(self.name)
        h, n = self.heads, self.tokens
        q = _layer_i8(rng, (1, h, n, self.dk))
        k = _layer_i8(rng, (1, h, n, self.dk))
        v = _layer_i8(rng, (1, h, n, self.dv))
        g = Graph()
        tq = g.tensor("q", q.shape, "int8", *self.x)
        tk = g.tensor("k", k.shape, "int8", *self.x)
        tv = g.tensor("v", v.shape, "int8", *self.x)
        c = g.tensor(
            "mul_rhs", (1,), "int8", *self.mul_rhs, data=np.array([self.mul_rhs_value])
        )
        q2 = g.tensor("q_scaled", q.shape, "int8", *self.q_scaled)
        g.op(
            "MUL",
            "int8",
            [tq, c],
            [q2],
            schema.BuiltinOptions.MulOptions,
            _activation_options(schema.MulOptionsT, "none"),
        )
        s = g.tensor("scores", (1, h, n, n), "int8", *self.scores)
        bmm = schema.BatchMatMulOptionsT()
        bmm.adjX, bmm.adjY, bmm.asymmetricQuantizeInputs = False, True, False
        g.op(
            "BATCH_MATMUL",
            "int8",
            [q2, tk],
            [s],
            schema.BuiltinOptions.BatchMatMulOptions,
            bmm,
        )
        p = g.tensor("probs", (1, h, n, n), "int8", *self.probs)
        sm = schema.SoftmaxOptionsT()
        sm.beta = 1.0
        g.op("SOFTMAX", "int8", [s], [p], schema.BuiltinOptions.SoftmaxOptions, sm)
        o = g.tensor("output", (1, h, n, self.dv), "int8", *self.output)
        bmm2 = schema.BatchMatMulOptionsT()
        bmm2.adjX, bmm2.adjY, bmm2.asymmetricQuantizeInputs = False, False, False
        g.op(
            "BATCH_MATMUL",
            "int8",
            [p, tv],
            [o],
            schema.BuiltinOptions.BatchMatMulOptions,
            bmm2,
        )
        out = _run(g.build([tq, tk, tv], [o], self.name), [q, k, v], self.kernels)
        meta = {
            "op": "ATTENTION",
            "kernels": self.kernels,
            "mul_rhs_value": str(self.mul_rhs_value),
            "beta": "1.0",
            "litert_version": litert_version,
        }
        for role, qp in (
            ("q", self.x),
            ("k", self.x),
            ("v", self.x),
            ("mul_rhs", self.mul_rhs),
            ("q_scaled", self.q_scaled),
            ("scores", self.scores),
            ("probs", self.probs),
            ("output", self.output),
        ):
            _qmeta(meta, role, qp)
        path = out_dir / f"{self.name}.safetensors"
        save_file({"q": q, "k": k, "v": v, "output": out}, str(path), metadata=meta)
        return path


@dataclass(frozen=True)
class DflHead:
    """SOFTMAX over 16 bins -> 1x1 CONV_2D onto one channel with weights ``round(i*127/15)``.

    The input is NHWC ``[1, 4, anchors, 16]``; the output ``[1, 4, anchors, 1]``.
    """

    name: str
    anchors: int
    input: tuple[float, int]
    output: tuple[float, int] = (0.0579, -128)
    filter_scale: float = 15 / 127
    kernels: str = "optimized"

    def write(self, out_dir: Path, litert_version: str) -> Path:
        rng = _rng(self.name)
        shape = (1, 4, self.anchors, 16)
        x = rng.integers(-60, 61, size=shape, dtype=np.int8)
        rows = x.reshape(-1, 16)
        rows[0, :] = rows[0, 0]
        rows[1, 0], rows[1, 1] = -128, 127
        rows[2, :] = -128
        rows[2, 15] = 127
        weights = np.array([(i * 127 + 7) // 15 for i in range(16)], dtype=np.int8)
        g = Graph()
        i = g.tensor("input", shape, "int8", *self.input)
        p = g.tensor("probs", shape, "int8", *I8_OUT)
        sm = schema.SoftmaxOptionsT()
        sm.beta = 1.0
        g.op("SOFTMAX", "int8", [i], [p], schema.BuiltinOptions.SoftmaxOptions, sm)
        fs = [_f32(self.filter_scale)]
        o = _conv_1x1(
            g,
            "conv",
            p,
            I8_OUT,
            16,
            1,
            (*shape[:-1], 1),
            self.output,
            weights.reshape(1, 16),
            fs,
            np.zeros(1, dtype=np.int32),
            "none",
        )
        out = _run(g.build([i], [o], self.name), [x], self.kernels)
        meta = {
            "op": "SOFTMAX_CONV_2D",
            "kernels": self.kernels,
            "beta": "1.0",
            "filter_scales": json.dumps(fs),
            "litert_version": litert_version,
        }
        _qmeta(meta, "input", self.input)
        _qmeta(meta, "probs", I8_OUT)
        _qmeta(meta, "output", self.output)
        path = out_dir / f"{self.name}.safetensors"
        save_file(
            {"input": x, "filter": weights, "output": out}, str(path), metadata=meta
        )
        return path


BIN_LHS = (0.0579, -128)
BIN_RHS = (0.0391, 5)
BIN_OUT = (0.0811, -3)
ODD = (3, 1001)  # 3003 elements: 187 blocks of 16 and a tail of 11
# Every (lhs, rhs) pair. TFLite's NEON-block ADD (one shift and one rounding per input) and
# its scalar ADD almost never differ: at this quantization, one pair of the 65536 does. A
# MUL multiplier above one, where the NEON blocks saturate the product to int16 and add the
# output zero point with wrap-around, separates the two MUL kernels on many elements.
PAIRS = (256, 256)
ADD_SPLIT = ((0.002_458_421, -125), (0.001_481_796_5, -48), (0.007_060_605_6, 73))

GRAPHS = [
    Binary("add_i8", "ADD", (4, 1024), (4, 1024), BIN_LHS, BIN_RHS, BIN_OUT),
    Binary("add_i8_odd", "ADD", ODD, ODD, BIN_LHS, BIN_RHS, BIN_OUT, "relu"),
    Binary("add_i8_pairs", "ADD", PAIRS, PAIRS, *ADD_SPLIT, every_pair=True),
    Binary(
        "add_i8_pairs_ref", "ADD", PAIRS, PAIRS, *ADD_SPLIT, "none", "reference", True
    ),
    Binary("sub_i8_odd", "SUB", ODD, ODD, BIN_LHS, BIN_RHS, BIN_OUT),
    Binary("mul_i8_odd", "MUL", ODD, ODD, BIN_LHS, BIN_RHS, (0.25, 10)),
    Binary(
        "mul_i8_odd_relu6", "MUL", ODD, ODD, BIN_LHS, BIN_RHS, (0.0393, -128), "relu6"
    ),
    Binary("mul_i8_odd_wide", "MUL", ODD, ODD, BIN_LHS, BIN_RHS, (0.0015, -100)),
    Binary(
        "mul_i8_odd_wide_ref",
        "MUL",
        ODD,
        ODD,
        BIN_LHS,
        BIN_RHS,
        (0.0015, -100),
        "none",
        "reference",
    ),
    Binary(
        "mul_i8_scalar",
        "MUL",
        (1, 2, 400, 32),
        (1,),
        (0.1251, -15),
        (0.000_693_2, -128),
        (0.010_18, 6),
    ),
    Binary(
        "mul_i8_scalar_odd", "MUL", (1, 3, 1001), (1,), BIN_LHS, BIN_RHS, (0.25, 10)
    ),
    BatchMatMul(
        "batch_matmul_i8",
        (2, 64, 48),
        (2, 48, 37),
        (0.0412, -7),
        (0.0377, 3),
        (0.71, 4),
    ),
    BatchMatMul(
        "batch_matmul_i8_adj",
        (2, 48, 64),
        (2, 37, 48),
        (0.0412, -7),
        (0.0377, 3),
        (0.71, 4),
        adj_x=True,
        adj_y=True,
    ),
    BatchMatMul(
        "batch_matmul_i8_ref",
        (2, 64, 48),
        (2, 48, 37),
        (0.0412, -7),
        (0.0377, 3),
        (0.71, 4),
        kernels="reference",
    ),
    Conv1x1("conv1x1_i8", (1, 10, 13, 24), 40, (0.0523, -11), (0.0441, -128), "relu6"),
    Conv1x1(
        "conv1x1_i8_ref",
        (1, 10, 13, 24),
        40,
        (0.0523, -11),
        (0.0441, -128),
        "relu6",
        "reference",
    ),
    Attention("attention_i8_yolo26", 2, 400, 32, 64),
    Attention("attention_i8_odd", 3, 37, 12, 9),
    DflHead("dfl_head_i8", 2100, (0.0947, 3)),
]


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
    for graph in GRAPHS:
        path = graph.write(out_dir, version)
        print(f"{path} ({path.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
