# SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
# SPDX-License-Identifier: Apache-2.0

import glob
import json
import math
import statistics
import threading
import time
from pathlib import Path

import numpy as np
import pytest

kernels = pytest.importorskip(
    "edgefirst.kernels",
    reason="WARNING: SKIPPED: edgefirst.kernels is not installed (make install-python)",
)

TESTDATA = Path(__file__).resolve().parents[2] / "testdata" / "kernels"

YOLOV8N_LEVELS = [(6400, 80, 8.0), (1600, 40, 16.0), (400, 20, 32.0)]


def load_fixture(name):
    safetensors_numpy = pytest.importorskip(
        "safetensors.numpy",
        reason="WARNING: SKIPPED: safetensors is not installed",
    )
    from safetensors import safe_open

    path = TESTDATA / f"{name}.safetensors"
    tensors = safetensors_numpy.load_file(str(path))
    with safe_open(str(path), "np") as f:
        meta = f.metadata()
    return tensors["input"], tensors["output"], meta


def quant(meta, prefix):
    return float(meta[f"{prefix}_scale"]), int(meta[f"{prefix}_zero_point"])


@pytest.mark.parametrize(
    "name",
    [
        "softmax_i8_dfl",
        "softmax_i8_yolo11_attn",
        "softmax_i8_odd_rows_1",
        "softmax_i8_odd_rows_17",
        "softmax_i8_odd_rows_1009",
    ],
)
def test_softmax_i8_matches_fixture(name):
    x, expected, meta = load_fixture(name)
    in_s, in_z = quant(meta, "input")
    out_s, out_z = quant(meta, "output")
    plan = kernels.SoftmaxI8Plan(in_s, in_z, out_s, out_z, float(meta["beta"]))
    out = np.zeros_like(expected)
    plan.run(x, out)
    # The NEON tiers sum a row in a different order, so they may differ by one LSB.
    diff = np.abs(out.astype(np.int32) - expected.astype(np.int32))
    assert diff.max() <= (0 if plan.tier == "scalar" else 1)


@pytest.mark.parametrize(
    "name",
    [
        "softmax_i16_attn",
        "softmax_i16_odd_rows_1",
        "softmax_i16_odd_rows_1009",
        "softmax_i16_scale_small",
        "softmax_i16_scale_large",
    ],
)
def test_softmax_i16_matches_fixture(name):
    x, expected, meta = load_fixture(name)
    in_s, in_z = quant(meta, "input")
    out_s, out_z = quant(meta, "output")
    plan = kernels.SoftmaxI16Plan(in_s, in_z, out_s, out_z, float(meta["beta"]))
    out = np.zeros_like(expected)
    plan.run(x, out)
    diff = np.abs(out.astype(np.int32) - expected.astype(np.int32))
    assert diff.max() <= (0 if plan.tier == "scalar" else 1)


@pytest.mark.parametrize(
    "name,dtype",
    [("logistic_i8_scores", np.int8), ("logistic_u8_scores", np.uint8)],
)
def test_sigmoid_lut_matches_fixture(name, dtype):
    x, expected, meta = load_fixture(name)
    plan = kernels.LutI8Plan.sigmoid(
        np.dtype(dtype), quant(meta, "input"), quant(meta, "output")
    )
    out = np.zeros_like(expected)
    plan.run(x, out)
    np.testing.assert_array_equal(out, expected)


def test_sigmoid_plan_matches_lut_plan():
    x, expected, meta = load_fixture("logistic_i8_scores")
    plan = kernels.SigmoidPlan(
        np.int8, quant(meta, "input"), np.int8, quant(meta, "output")
    )
    out = np.zeros_like(expected)
    plan.run(x, out)
    np.testing.assert_array_equal(out, expected)


def test_lut_from_table_identity():
    table = np.arange(256, dtype=np.uint8)
    plan = kernels.LutI8Plan.from_table(np.uint8, table)
    x = np.arange(256, dtype=np.uint8)[::-1].copy()
    out = np.zeros_like(x)
    plan.run(x, out)
    np.testing.assert_array_equal(out, x)
    with pytest.raises(ValueError):
        kernels.LutI8Plan.from_table(np.uint8, np.arange(255, dtype=np.uint8))


def dfl_box_reference(logits, scale, zp, levels, reg_max, fmt):
    deq = scale * (logits.astype(np.float64) - zp)
    deq = deq.reshape(-1, 4, reg_max)
    e = np.exp(deq - deq.max(axis=2, keepdims=True))
    dist = (e / e.sum(axis=2, keepdims=True)) @ np.arange(reg_max, dtype=np.float64)
    out = np.empty((logits.shape[0], 4), dtype=np.float64)
    start = 0
    for anchors, grid_w, stride in levels:
        idx = np.arange(anchors)
        cx = (idx % grid_w) + 0.5
        cy = (idx // grid_w) + 0.5
        d = dist[start : start + anchors]
        x1, y1 = cx - d[:, 0], cy - d[:, 1]
        x2, y2 = cx + d[:, 2], cy + d[:, 3]
        if fmt == "xywh":
            box = np.stack([(x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1], axis=1)
        else:
            box = np.stack([x1, y1, x2, y2], axis=1)
        out[start : start + anchors] = box * stride
        start += anchors
    return out


@pytest.mark.parametrize("fmt", ["xywh", "xyxy"])
def test_dfl_box_against_numpy_reference(fmt):
    rng = np.random.default_rng(7)
    reg_max = 16
    total = sum(a for a, _, _ in YOLOV8N_LEVELS)
    logits = rng.integers(-128, 128, size=(total, 4 * reg_max), dtype=np.int8)
    scale, zp = 0.08, -10
    plan = kernels.DflBoxPlan(
        np.int8, (scale, zp), np.float32, None, reg_max, YOLOV8N_LEVELS, fmt
    )
    out = np.zeros((total, 4), dtype=np.float32)
    plan.run(logits, out)
    ref = dfl_box_reference(logits, scale, zp, YOLOV8N_LEVELS, reg_max, fmt)
    np.testing.assert_allclose(out, ref, atol=1e-3)


def test_dfl_and_dist2bbox_compose_to_dfl_box():
    rng = np.random.default_rng(3)
    reg_max = 8
    levels = [(64, 8, 8.0), (16, 4, 16.0)]
    total = 80
    logits = rng.integers(-128, 128, size=(total, 4 * reg_max), dtype=np.int8)
    q = (0.1, 0)
    dfl = kernels.DflPlan(np.int8, q, np.float32, None, reg_max)
    d2b = kernels.Dist2BboxPlan(np.float32, None, np.float32, None, levels)
    fused = kernels.DflBoxPlan(np.int8, q, np.float32, None, reg_max, levels)
    dist = np.zeros((total, 4), dtype=np.float32)
    dfl.run(logits, dist)
    two_step = np.zeros((total, 4), dtype=np.float32)
    d2b.run(dist, two_step)
    one_step = np.zeros((total, 4), dtype=np.float32)
    fused.run(logits, one_step)
    np.testing.assert_allclose(one_step, two_step, atol=1e-3)


def test_float16_io():
    plan = kernels.SigmoidPlan(np.float16, None, np.float16, None)
    x = np.linspace(-6, 6, 33, dtype=np.float16)
    out = np.zeros_like(x)
    plan.run(x, out)
    ref = 1.0 / (1.0 + np.exp(-x.astype(np.float64)))
    np.testing.assert_allclose(out.astype(np.float64), ref, atol=2e-3)


def test_transposed_input_view():
    x, _, meta = load_fixture("softmax_i8_odd_rows_17")
    in_s, in_z = quant(meta, "input")
    out_s, out_z = quant(meta, "output")
    plan = kernels.SoftmaxI8Plan(in_s, in_z, out_s, out_z)
    transposed = np.ascontiguousarray(x.T)
    assert transposed.T.strides != x.strides
    out = np.zeros_like(x)
    plan.run(transposed.T, out)
    ref = np.zeros_like(x)
    plan.run(x, ref)
    np.testing.assert_array_equal(out, ref)
    # A strided output works too.
    wide = np.zeros((x.shape[0], x.shape[1] * 2), dtype=np.int8)
    plan.run(x, wide[:, ::2])
    np.testing.assert_array_equal(wide[:, ::2], ref)


def test_negative_stride_raises_value_error():
    plan = kernels.SoftmaxI8Plan(0.1, 0, 1 / 256, -128)
    x = np.zeros((4, 8), dtype=np.int8)
    with pytest.raises(ValueError):
        plan.run(x[:, ::-1], np.zeros_like(x))


def test_dtype_mismatch_raises_value_error():
    plan = kernels.SoftmaxI8Plan(0.1, 0, 1 / 256, -128)
    x = np.zeros((4, 8), dtype=np.int16)
    with pytest.raises(ValueError):
        plan.run(x, np.zeros((4, 8), dtype=np.int8))


def test_same_array_as_input_and_output_is_refused():
    plan = kernels.SoftmaxI8Plan(0.1, 0, 1 / 256, -128)
    x = np.zeros((4, 8), dtype=np.int8)
    with pytest.raises(ValueError):
        plan.run(x, x)


def test_interleaved_views_of_one_array_are_refused():
    plan = kernels.SigmoidPlan(np.int8, (0.1, 0), np.int8, (1 / 256, -128))
    a = np.full((4, 16), 7, dtype=np.int8)
    with pytest.raises(ValueError, match="overlap"):
        plan.run(a[..., ::2], a[..., 1::2])
    with pytest.raises(ValueError, match="overlap"):
        plan.run(a[..., 1::2], a[..., ::2])
    assert (a == 7).all()
    # Disjoint halves of one array do not overlap.
    plan.run(a[:2], a[2:])
    assert (a[:2] == 7).all() and (a[2:] != 7).all()


def test_float_tensors_ignore_their_quant():
    x = np.linspace(-6, 6, 33, dtype=np.float32)
    want = np.zeros_like(x)
    kernels.SigmoidPlan(np.float32, None, np.float32, None).run(x, want)
    got = np.zeros_like(x)
    kernels.SigmoidPlan(np.float32, (0.0, 0), np.float32, (0.0, 0)).run(x, got)
    np.testing.assert_array_equal(got, want)
    xh = x.astype(np.float16)
    want_h = np.zeros_like(xh)
    kernels.SigmoidPlan(np.float16, None, np.float16, None).run(xh, want_h)
    got_h = np.zeros_like(xh)
    kernels.SigmoidPlan(np.float16, (0.0, 0), np.float16, (0.0, 0)).run(xh, got_h)
    np.testing.assert_array_equal(got_h, want_h)


def test_readonly_output_raises_value_error():
    plan = kernels.SoftmaxI8Plan(0.1, 0, 1 / 256, -128)
    x = np.zeros((4, 8), dtype=np.int8)
    out = np.zeros((4, 8), dtype=np.int8)
    out.flags.writeable = False
    with pytest.raises(ValueError):
        plan.run(x, out)


def test_empty_input_is_a_no_op():
    plan = kernels.SoftmaxI8Plan(0.1, 0, 1 / 256, -128)
    plan.run(np.zeros((0, 8), dtype=np.int8), np.zeros((0, 8), dtype=np.int8))


def test_bad_quant_raises_not_implemented():
    # Softmax int8 needs output scale 1/256 and zero point -128.
    with pytest.raises(NotImplementedError):
        kernels.SoftmaxI8Plan(0.1, 0, 0.5, 0)


def test_bad_params_raise_value_error():
    with pytest.raises(ValueError):
        kernels.SoftmaxI8Plan(0.0, 0, 1 / 256, -128)
    with pytest.raises(ValueError):
        kernels.DflPlan(np.int8, (0.1, 0), np.float32, None, 0)
    with pytest.raises(ValueError):
        kernels.Dist2BboxPlan(np.float32, None, np.float32, None, [(10, 0, 8.0)])
    with pytest.raises(ValueError):
        kernels.SigmoidPlan(np.float64, None, np.float32, None)
    with pytest.raises(ValueError):
        kernels.Dist2BboxPlan(np.float32, None, np.float32, None, [(4, 2, 8.0)], "ltrb")


def test_shape_mismatch_raises_value_error():
    plan = kernels.SoftmaxI8Plan(0.1, 0, 1 / 256, -128)
    x = np.zeros((4, 8), dtype=np.int8)
    out = np.full((4, 7), 5, dtype=np.int8)
    with pytest.raises(ValueError):
        plan.run(x, out)
    assert (out == 5).all()


def test_threads_matches_single_thread():
    rng = np.random.default_rng(1)
    x = rng.integers(-128, 128, size=(513, 1009), dtype=np.int8)
    plan = kernels.SoftmaxI8Plan(0.05, 3, 1 / 256, -128)
    single = np.zeros_like(x)
    plan.run(x, single)
    for threads in (1, 2, 4, 64):
        out = np.zeros_like(x)
        plan.run(x, out, threads=threads)
        np.testing.assert_array_equal(out, single)
    with pytest.raises(ValueError):
        plan.run(x, np.zeros_like(x), threads=0)
    with pytest.raises(ValueError):
        plan.run(x, np.zeros_like(x), threads=-2)


def test_tier_is_a_known_string():
    plan = kernels.SoftmaxI8Plan(0.1, 0, 1 / 256, -128)
    assert plan.tier in {"scalar", "neon", "neon_fp16", "neon_dotprod"}
    with pytest.raises(AttributeError):
        plan.tier = "scalar"


# int8 arithmetic, matrix products and fused chains.
#
# The fixtures come from the x86 LiteRT wheel. Its optimized int8 ADD and MUL
# run TFLite's NEON blocks (through NEON_2_SSE), so a fixture from the
# optimized resolver is reproduced by ``kernels="neon_gemmlowp"``, whose
# BATCH_MATMUL and CONV_2D round as the reference kernels do, so the reference
# resolver's matrix fixtures are reproduced by ``kernels="reference"`` too.

I8_PROBS = (1 / 256, -128)


def load_graph(name):
    safetensors_numpy = pytest.importorskip(
        "safetensors.numpy",
        reason="WARNING: SKIPPED: safetensors is not installed",
    )
    from safetensors import safe_open

    path = TESTDATA / f"{name}.safetensors"
    tensors = safetensors_numpy.load_file(str(path))
    with safe_open(str(path), "np") as f:
        meta = f.metadata()
    return tensors, meta


def fixtures_with_op(*ops):
    """Fixture names whose ``op`` metadata is one of ``ops``, read at collection time."""
    try:
        from safetensors import safe_open
    except ImportError:
        return []
    names = []
    for path in sorted(glob.glob(str(TESTDATA / "*.safetensors"))):
        with safe_open(path, "np") as f:
            if f.metadata().get("op") in ops:
                names.append(Path(path).stem)
    return names


BINARY_FIXTURES = fixtures_with_op("ADD", "SUB", "MUL")
MATMUL_FIXTURES = fixtures_with_op("BATCH_MATMUL")
CONV_FIXTURES = fixtures_with_op("CONV_2D")


def binary_kernels(meta):
    return "neon_gemmlowp" if meta["kernels"] == "optimized" else "reference"


def quantize_multiplier(real):
    """TFLite's ``QuantizeMultiplier``: ``(multiplier, shift)`` with ``real = m * 2^(shift - 31)``."""
    if real == 0.0:
        return 0, 0
    m, shift = math.frexp(real)
    q = math.floor(m * (1 << 31) + 0.5)
    if q == 1 << 31:
        q //= 2
        shift += 1
    return q, shift


def lsb_diff(a, b):
    return np.abs(a.astype(np.int32) - b.astype(np.int32))


def test_new_op_fixtures_are_present():
    assert BINARY_FIXTURES and MATMUL_FIXTURES and CONV_FIXTURES


@pytest.mark.parametrize("name", BINARY_FIXTURES)
def test_binary_matches_fixture(name):
    t, meta = load_graph(name)
    plan = kernels.BinaryI8Plan(
        meta["op"].lower(),
        quant(meta, "lhs"),
        quant(meta, "rhs"),
        quant(meta, "output"),
        meta["activation"],
        binary_kernels(meta),
    )
    lhs = t["lhs"]
    rhs = np.broadcast_to(t["rhs"], lhs.shape)
    out = np.full_like(lhs, 0x55)
    plan.run(lhs, rhs, out)
    np.testing.assert_array_equal(out, t["output"])


def test_binary_neon_blocks_are_distinguished():
    # A MUL whose multiplier exceeds 1 saturates differently in TFLite's NEON
    # blocks: the optimized fixture takes them, the reference fixture does not.
    t, meta = load_graph("mul_i8_odd_wide")
    args = ("mul", quant(meta, "lhs"), quant(meta, "rhs"), quant(meta, "output"))
    neon, ref = (np.zeros_like(t["lhs"]) for _ in range(2))
    kernels.BinaryI8Plan(*args, kernels="neon_ruy").run(t["lhs"], t["rhs"], neon)
    kernels.BinaryI8Plan(*args, kernels="reference").run(t["lhs"], t["rhs"], ref)
    np.testing.assert_array_equal(neon, t["output"])
    assert (neon != ref).any()
    plain = np.zeros_like(t["lhs"])
    kernels.BinaryI8Plan(*args, kernels="ruy").run(t["lhs"], t["rhs"], plain)
    np.testing.assert_array_equal(plain, ref)


@pytest.mark.parametrize("name", ["mul_i8_scalar", "mul_i8_scalar_odd"])
def test_binary_table_for_rhs_matches_the_plan(name):
    t, meta = load_graph(name)
    plan = kernels.BinaryI8Plan(
        "mul",
        quant(meta, "lhs"),
        quant(meta, "rhs"),
        quant(meta, "output"),
        kernels=binary_kernels(meta),
    )
    lhs = t["lhs"]
    table = plan.table_for_rhs(int(t["rhs"][0]), lhs.size)
    assert table is not None and table.dtype == np.uint8 and table.shape == (256,)
    out = np.zeros_like(lhs)
    kernels.LutI8Plan.from_table(np.int8, table).run(lhs, out)
    np.testing.assert_array_equal(out, t["output"])


def test_binary_table_for_rhs_lengths():
    plan = kernels.BinaryI8Plan(
        "add", (0.002894, -128), (0.003897, -128), (0.004491, -128)
    )
    whole = plan.table_for_rhs(5, 32)
    scalar = plan.table_for_rhs(5, 0)
    assert whole is not None and scalar is not None
    mixed = plan.table_for_rhs(5, 33)
    assert mixed is None or (
        np.array_equal(whole, scalar) and np.array_equal(mixed, whole)
    )
    ref = kernels.BinaryI8Plan(
        "add", (0.002894, -128), (0.003897, -128), (0.004491, -128), kernels="reference"
    )
    np.testing.assert_array_equal(ref.table_for_rhs(5, 33), scalar)


def test_binary_strided_views_match_contiguous():
    rng = np.random.default_rng(11)
    plan = kernels.BinaryI8Plan("add", (0.05, -3), (0.04, 2), (0.08, 1), "relu")
    lhs = rng.integers(-128, 128, size=(37, 29), dtype=np.int8)
    rhs = rng.integers(-128, 128, size=(37, 29), dtype=np.int8)
    want = np.zeros_like(lhs)
    plan.run(lhs, rhs, want)
    lhs_t = np.ascontiguousarray(lhs.T).T
    wide = np.full((37, 58), 0x55, dtype=np.int8)
    plan.run(lhs_t, rhs, wide[:, ::2])
    np.testing.assert_array_equal(wide[:, ::2], want)
    assert (wide[:, 1::2] == 0x55).all()


def test_binary_inputs_may_share_memory_but_not_the_output():
    plan = kernels.BinaryI8Plan("mul", (0.05, 0), (0.05, 0), (0.1, 0))
    x = np.arange(-64, 64, dtype=np.int8).reshape(8, 16)
    out = np.zeros_like(x)
    plan.run(x, x, out)
    want = np.zeros_like(x)
    plan.run(x, x.copy(), want)
    np.testing.assert_array_equal(out, want)
    for args in ((x, x.copy(), x), (x.copy(), x, x)):
        with pytest.raises(ValueError, match="overlap"):
            plan.run(*args)
    a = np.full((8, 32), 7, dtype=np.int8)
    with pytest.raises(ValueError, match="overlap"):
        plan.run(a[:, ::2], np.zeros((8, 16), np.int8), a[:, 1::2])
    assert (a == 7).all()


def test_binary_rejects_bad_arguments():
    good = ((0.05, 0), (0.05, 0), (0.1, 0))
    with pytest.raises(ValueError):
        kernels.BinaryI8Plan("div", *good)
    with pytest.raises(ValueError):
        kernels.BinaryI8Plan("add", *good, activation="tanh")
    with pytest.raises(ValueError):
        kernels.BinaryI8Plan("add", *good, kernels="fast")
    with pytest.raises(ValueError):
        kernels.BinaryI8Plan("add", (0.0, 0), (0.05, 0), (0.1, 0))
    with pytest.raises(ValueError):
        kernels.BinaryI8Plan("add", (0.05, 200), (0.05, 0), (0.1, 0))
    # ADD's output rescale must stay below one; SUB has TFLite's larger-than-one path.
    with pytest.raises(NotImplementedError):
        kernels.BinaryI8Plan("add", (1.0, 0), (1.0, 0), (1e-9, 0))
    kernels.BinaryI8Plan("sub", (1.0, 0), (1.0, 0), (1e-9, 0))
    plan = kernels.BinaryI8Plan("add", *good)
    x = np.zeros((4, 8), dtype=np.int8)
    out = np.full((4, 8), 5, dtype=np.int8)
    with pytest.raises(ValueError):
        plan.run(x, np.zeros((4, 7), np.int8), out)
    with pytest.raises(ValueError):
        plan.run(x, x.astype(np.int16), out)
    assert (out == 5).all()


def test_binary_threads_match_single_thread():
    rng = np.random.default_rng(2)
    lhs = rng.integers(-128, 128, size=(64, 1009), dtype=np.int8)
    rhs = rng.integers(-128, 128, size=(64, 1009), dtype=np.int8)
    plan = kernels.BinaryI8Plan("add", (0.05, -3), (0.04, 2), (0.08, 1))
    single = np.zeros_like(lhs)
    plan.run(lhs, rhs, single)
    for threads in (2, 4, 64):
        out = np.zeros_like(lhs)
        plan.run(lhs, rhs, out, threads=threads)
        np.testing.assert_array_equal(out, single)


def matmul_operands(t, meta):
    lhs, rhs = t["lhs"], t["rhs"]
    if meta["adj_x"] == "true":
        lhs = lhs.swapaxes(-1, -2)
    if meta["adj_y"] == "true":
        rhs = rhs.swapaxes(-1, -2)
    return lhs, rhs


@pytest.mark.parametrize("name", MATMUL_FIXTURES)
def test_batch_matmul_matches_fixture(name):
    t, meta = load_graph(name)
    lhs, rhs = matmul_operands(t, meta)
    plan = kernels.MatMulI8Plan.tflite_batch_matmul(
        quant(meta, "lhs"), quant(meta, "rhs"), quant(meta, "output"), "reference"
    )
    out = np.full_like(t["output"], 0x55)
    plan.run(lhs, rhs, out)
    np.testing.assert_array_equal(out, t["output"])
    # ruy's single rounding moves a few outputs by one LSB.
    ruy = kernels.MatMulI8Plan.tflite_batch_matmul(
        quant(meta, "lhs"), quant(meta, "rhs"), quant(meta, "output"), "neon_ruy"
    )
    got = np.zeros_like(out)
    ruy.run(lhs, rhs, got)
    diff = lsb_diff(got, t["output"])
    assert 0 < np.count_nonzero(diff) and diff.max() == 1


def test_matmul_explicit_requant_equals_batch_matmul():
    t, meta = load_graph("batch_matmul_i8")
    (ls, lz), (rs, rz), (os_, oz) = (quant(meta, r) for r in ("lhs", "rhs", "output"))
    real = float(np.float32(np.float32(ls) * np.float32(rs))) / float(np.float32(os_))
    plan = kernels.MatMulI8Plan(
        lz, rz, (os_, oz), [quantize_multiplier(real)], kernels="reference"
    )
    out = np.zeros_like(t["output"])
    plan.run(t["lhs"], t["rhs"], out)
    np.testing.assert_array_equal(out, t["output"])


@pytest.mark.parametrize("name", CONV_FIXTURES)
def test_conv1x1_matches_fixture(name):
    t, meta = load_graph(name)
    x, filt = t["input"], t["filter"]
    cin, cout = x.shape[-1], filt.shape[0]
    plan = kernels.MatMulI8Plan.tflite_conv(
        quant(meta, "input"),
        json.loads(meta["filter_scales"]),
        quant(meta, "output"),
        bias=t["bias"].tolist(),
        activation=meta["activation"],
        kernels="reference",
    )
    out = np.full((x.size // cin, cout), 0x55, dtype=np.int8)
    plan.run(x.reshape(-1, cin), filt.reshape(cout, cin).T, out)
    np.testing.assert_array_equal(out, t["output"].reshape(-1, cout))


def test_matmul_rejects_bad_arguments():
    kernels.MatMulI8Plan.check_depth(4096)
    with pytest.raises(NotImplementedError):
        kernels.MatMulI8Plan.check_depth(4097)
    with pytest.raises(ValueError):
        kernels.MatMulI8Plan(200, 0, (0.1, 0), [(1 << 30, -3)])
    with pytest.raises(ValueError):
        kernels.MatMulI8Plan(0, 0, (0.1, 0), [])
    with pytest.raises(ValueError):
        kernels.MatMulI8Plan(0, 0, (0.0, 0), [(1 << 30, -3)])
    with pytest.raises(ValueError):
        kernels.MatMulI8Plan.tflite_batch_matmul((0.0, 0), (0.1, 0), (0.1, 0))
    # ruy's rounding is exact only for multipliers below one half.
    with pytest.raises(NotImplementedError):
        kernels.MatMulI8Plan.tflite_batch_matmul((1.0, 0), (1.0, 0), (1.0, 0))
    kernels.MatMulI8Plan.tflite_batch_matmul((1.0, 0), (1.0, 0), (1.0, 0), "reference")
    plan = kernels.MatMulI8Plan.tflite_conv(
        (0.05, 0), [0.01, 0.02], (0.1, 0), bias=[1, 2]
    )
    lhs = np.ones((3, 5), dtype=np.int8)
    out = np.full((3, 2), 5, dtype=np.int8)
    with pytest.raises(ValueError):
        plan.run(lhs, np.ones((4, 2), np.int8), out)
    with pytest.raises(ValueError):
        plan.run(lhs, np.ones((5, 3), np.int8), np.zeros((3, 3), np.int8))
    with pytest.raises(NotImplementedError):
        big = np.zeros((1, 4097), np.int8)
        plan.run(big, np.zeros((4097, 2), np.int8), out[:1])
    assert (out == 5).all()
    rhs = np.ones((5, 2), dtype=np.int8)
    with pytest.raises(ValueError, match="overlap"):
        buf = np.zeros(16, dtype=np.int8)
        plan.run(lhs, buf[:10].reshape(5, 2), buf[4:10].reshape(3, 2))
    plan.run(lhs, rhs, out)


def test_matmul_threads_match_single_thread():
    rng = np.random.default_rng(4)
    lhs = rng.integers(-128, 128, size=(3, 37, 300), dtype=np.int8)
    rhs = rng.integers(-128, 128, size=(3, 300, 29), dtype=np.int8)
    plan = kernels.MatMulI8Plan.tflite_batch_matmul((0.05, 3), (0.04, -2), (2.0, 1))
    single = np.zeros((3, 37, 29), dtype=np.int8)
    plan.run(lhs, rhs, single)
    for threads in (2, 4, 64):
        out = np.zeros_like(single)
        plan.run(lhs, rhs.swapaxes(1, 2).copy().swapaxes(1, 2), out, threads=threads)
        np.testing.assert_array_equal(out, single)


def dfl_head(meta, weights, kernels_name="reference"):
    projection = kernels.MatMulI8Plan.tflite_conv(
        quant(meta, "probs"),
        json.loads(meta["filter_scales"]),
        quant(meta, "output"),
        bias=[0],
        kernels=kernels_name,
    )
    return kernels.SoftmaxDotI8Plan(
        quant(meta, "input"),
        weights,
        projection,
        float(meta["beta"]),
        quant(meta, "probs"),
    )


def dfl_head_float(x, meta, weights):
    """The output before rounding: ``zo + E[w] / probs_scale * multiplier``, clamped."""
    in_s, _ = quant(meta, "input")
    probs_s, _ = quant(meta, "probs")
    out_s, out_z = quant(meta, "output")
    (fs,) = json.loads(meta["filter_scales"])
    m, shift = quantize_multiplier(
        float(np.float32(probs_s)) * float(np.float32(fs)) / float(np.float32(out_s))
    )
    rows = x.reshape(-1, x.shape[-1]).astype(np.float64)
    e = np.exp(float(np.float32(in_s)) * (rows - rows.max(axis=1, keepdims=True)))
    expect = (e * weights.astype(np.float64)).sum(axis=1) / e.sum(axis=1)
    acc = expect / float(np.float32(probs_s))
    return np.clip(out_z + acc * m * 2.0 ** (shift - 31), -128, 127)


def test_softmax_dot_dfl_head_against_float_and_tflite():
    t, meta = load_graph("dfl_head_i8")
    x, weights = t["input"], t["filter"].reshape(-1)
    plan = dfl_head(meta, weights)
    out = np.full(x.shape[:-1] + (1,), 0x55, dtype=np.int8)
    plan.run(x, out)
    got = out.reshape(-1).astype(np.float64)
    tflite = t["output"].reshape(-1).astype(np.float64)
    want = dfl_head_float(x, meta, weights)
    assert np.abs(got - want).max() <= 1.0
    assert np.abs(got - want).mean() < np.abs(tflite - want).mean()
    assert (np.abs(got - tflite) <= np.ceil(np.abs(tflite - want)) + 1).all()


def test_softmax_dot_weights_as_a_sequence_and_strided_rows():
    t, meta = load_graph("dfl_head_i8")
    x = t["input"].reshape(-1, 16)[:300]
    weights = t["filter"].reshape(-1)
    want = np.zeros((300, 1), dtype=np.int8)
    dfl_head(meta, weights).run(x, want)
    padded = np.zeros((300, 24), dtype=np.int8)
    padded[:, :16] = x
    out = np.full((300, 3), 0x55, dtype=np.int8)
    dfl_head(meta, weights.tolist()).run(padded[:, :16], out[:, 1:2])
    np.testing.assert_array_equal(out[:, 1:2], want)
    assert (out[:, ::2] == 0x55).all()
    with pytest.raises(NotImplementedError):
        dfl_head(meta, weights).run(
            np.zeros((4, 32), np.int8)[:, ::2], np.zeros((4, 1), np.int8)
        )


def test_softmax_dot_rejects_bad_arguments():
    _, meta = load_graph("dfl_head_i8")
    weights = np.arange(16, dtype=np.int8)
    with pytest.raises(ValueError):
        dfl_head(meta, np.zeros(0, dtype=np.int8))
    zero_lhs = kernels.MatMulI8Plan.tflite_conv((1 / 256, 0), [0.1], (0.05, 0))
    with pytest.raises(ValueError):
        kernels.SoftmaxDotI8Plan((0.1, 0), weights, zero_lhs)
    two_columns = kernels.MatMulI8Plan.tflite_conv(
        (1 / 256, -128), [0.1, 0.1], (0.05, 0)
    )
    with pytest.raises(ValueError):
        kernels.SoftmaxDotI8Plan((0.1, 0), weights, two_columns)
    one = kernels.MatMulI8Plan.tflite_conv((1 / 256, -128), [0.1], (0.05, 0))
    with pytest.raises(NotImplementedError):
        kernels.SoftmaxDotI8Plan((0.1, 0), weights, one, probs_quant=(1 / 128, -128))
    plan = kernels.SoftmaxDotI8Plan((0.1, 0), weights, one)
    out = np.full((4, 2), 5, dtype=np.int8)
    with pytest.raises(ValueError):
        plan.run(np.zeros((4, 16), np.int8), out)
    with pytest.raises(ValueError):
        plan.run(np.zeros((4, 15), np.int8), out[:, :1])
    assert (out == 5).all()


def test_softmax_dot_threads_match_single_thread():
    t, meta = load_graph("dfl_head_i8")
    plan = dfl_head(meta, t["filter"].reshape(-1))
    x = t["input"]
    single = np.zeros(x.shape[:-1] + (1,), dtype=np.int8)
    plan.run(x, single)
    for threads in (2, 4, 64):
        out = np.zeros_like(single)
        plan.run(x, out, threads=threads)
        np.testing.assert_array_equal(out, single)


def attention_plan(meta, n_q_elements, kernels_name="reference"):
    """The fixture chain's plan: the MUL table as the optimized fixture computed it."""
    mul = kernels.BinaryI8Plan(
        "mul",
        quant(meta, "q"),
        quant(meta, "mul_rhs"),
        quant(meta, "q_scaled"),
        kernels="neon_gemmlowp" if meta["kernels"] == "optimized" else "reference",
    )
    table = mul.table_for_rhs(int(meta["mul_rhs_value"]), n_q_elements)
    assert table is not None
    return kernels.AttentionI8Plan(
        quant(meta, "q"),
        quant(meta, "k"),
        quant(meta, "v"),
        quant(meta, "scores"),
        quant(meta, "probs"),
        quant(meta, "output"),
        float(meta["beta"]),
        q_mul=(table, quant(meta, "q_scaled")),
        kernels=kernels_name,
    ), mul


def attention_chain(meta, mul, q, k, v):
    """MUL -> BATCH_MATMUL -> SOFTMAX -> BATCH_MATMUL with this module's single-op plans."""
    q2 = np.zeros_like(q)
    rhs = np.full(q.shape, int(meta["mul_rhs_value"]), dtype=np.int8)
    mul.run(q, rhs, q2)
    h, n, m = q.shape[0], q.shape[1], k.shape[1]
    bmm1 = kernels.MatMulI8Plan.tflite_batch_matmul(
        quant(meta, "q_scaled"), quant(meta, "k"), quant(meta, "scores"), "reference"
    )
    s = np.zeros((h, n, m), dtype=np.int8)
    bmm1.run(q2, k.swapaxes(1, 2), s)
    sm = kernels.SoftmaxI8Plan(
        *quant(meta, "scores"), *quant(meta, "probs"), float(meta["beta"])
    )
    p = np.zeros_like(s)
    sm.run(s, p)
    bmm2 = kernels.MatMulI8Plan.tflite_batch_matmul(
        quant(meta, "probs"), quant(meta, "v"), quant(meta, "output"), "reference"
    )
    out = np.zeros((h, n, v.shape[2]), dtype=np.int8)
    bmm2.run(p, v, out)
    return out


# NEON tiers run TFLite's softmax within one LSB per probability, so the fused output
# may move by one LSB; the Rust suite (crates/kernels/tests/attention.rs) holds the
# same bar, and on the Cortex-A53 and Cortex-A55 boards no output of these fixtures
# moves at all.
ATTENTION_NEON_MAX_LSB = 1
ATTENTION_NEON_MAX_SHARE = 0.02


@pytest.mark.parametrize("name", ["attention_i8_yolo26", "attention_i8_odd"])
def test_attention_matches_fixture(name):
    t, meta = load_graph(name)
    q, k, v = t["q"][0], t["k"][0], t["v"][0]
    plan, mul = attention_plan(meta, q.size)
    out = np.full(t["output"][0].shape, 0x55, dtype=np.int8)
    plan.run(q, k, v, out)
    np.testing.assert_array_equal(out, attention_chain(meta, mul, q, k, v))
    diff = lsb_diff(out, t["output"][0])
    if plan.tier == "scalar":
        assert diff.max() == 0
    else:
        assert diff.max() <= ATTENTION_NEON_MAX_LSB
        assert np.count_nonzero(diff) <= ATTENTION_NEON_MAX_SHARE * diff.size


def test_attention_strided_views_of_one_array():
    t, meta = load_graph("attention_i8_odd")
    q, k, v = t["q"][0], t["k"][0], t["v"][0]
    h, n, dk = q.shape
    dv = v.shape[2]
    plan, _ = attention_plan(meta, q.size)
    want = np.zeros((h, n, dv), dtype=np.int8)
    plan.run(q, k, v, want)
    # One token-major [n, h, dk + dk + dv] array; q, k and v are head slices of it.
    x = np.concatenate([q, k, v], axis=2).transpose(1, 0, 2).copy()
    tq, tk, tv = (
        x[..., a:b].transpose(1, 0, 2)
        for a, b in ((0, dk), (dk, 2 * dk), (2 * dk, 2 * dk + dv))
    )
    out_tm = np.full((n, h, dv + 3), 0x55, dtype=np.int8)
    plan.run(tq, tk, tv, out_tm[..., :dv].transpose(1, 0, 2))
    np.testing.assert_array_equal(out_tm[..., :dv].transpose(1, 0, 2), want)
    assert (out_tm[..., dv:] == 0x55).all()
    # Outputs over the q columns and over the v columns of the same array.
    for lo in (0, 2 * dk):
        with pytest.raises(ValueError, match="overlap"):
            plan.run(tq, tk, tv, x[..., lo : lo + dv].transpose(1, 0, 2))


def test_attention_scratch_follows_the_shapes():
    rng = np.random.default_rng(5)
    t, meta = load_graph("attention_i8_odd")
    plan, _ = attention_plan(meta, t["q"][0].size)

    def case(h, n, m, dk, dv):
        q = rng.integers(-40, 40, size=(h, n, dk), dtype=np.int8)
        k = rng.integers(-40, 40, size=(h, m, dk), dtype=np.int8)
        v = rng.integers(-40, 40, size=(h, m, dv), dtype=np.int8)
        fresh, _ = attention_plan(meta, t["q"][0].size)
        want = np.zeros((h, n, dv), dtype=np.int8)
        fresh.run(q, k, v, want)
        return q, k, v, want

    cases = [case(3, 37, 37, 12, 9), case(1, 5, 70, 32, 16), case(2, 9, 9, 4, 4)]
    for q, k, v, want in cases + cases[::-1]:
        out = np.zeros_like(want)
        plan.run(q, k, v, out)
        np.testing.assert_array_equal(out, want)
    errors = []

    def worker(i):
        q, k, v, want = cases[i % len(cases)]
        for _ in range(5):
            out = np.zeros_like(want)
            plan.run(q, k, v, out, threads=2)
            if not np.array_equal(out, want):
                errors.append(i)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(6)]
    for th in threads:
        th.start()
    for th in threads:
        th.join()
    assert not errors


def test_attention_limits_and_bad_arguments():
    kernels.AttentionI8Plan.check_shape(2048, 256, 256)
    for shape in ((2049, 8, 8), (8, 257, 8), (8, 8, 257)):
        with pytest.raises(NotImplementedError):
            kernels.AttentionI8Plan.check_shape(*shape)
    x = (0.1, 0)
    with pytest.raises(NotImplementedError):
        kernels.AttentionI8Plan(x, x, x, x, (1 / 128, -128), x)
    with pytest.raises(ValueError):
        kernels.AttentionI8Plan((0.0, 0), x, x, x, I8_PROBS, x)
    with pytest.raises(ValueError):
        kernels.AttentionI8Plan(
            x, x, x, x, I8_PROBS, x, q_mul=(np.zeros(255, np.uint8), x)
        )
    with pytest.raises(ValueError):
        kernels.AttentionI8Plan(x, x, x, x, I8_PROBS, x, kernels="fast")
    plan = kernels.AttentionI8Plan(x, x, x, (0.5, 0), I8_PROBS, (0.2, 0))
    q = np.zeros((2, 4, 8), dtype=np.int8)
    out = np.full((2, 4, 8), 5, dtype=np.int8)
    with pytest.raises(ValueError):
        plan.run(q[0], q[0], q[0], out[0])
    with pytest.raises(ValueError):
        plan.run(q, np.zeros((2, 4, 7), np.int8), q, out)
    with pytest.raises(ValueError):
        plan.run(q, q, np.zeros((2, 5, 8), np.int8), out)
    with pytest.raises(NotImplementedError):
        long = np.zeros((1, 2049, 8), dtype=np.int8)
        plan.run(q[:1], long, long, out[:1])
    assert (out == 5).all()
    plan.run(q, q, q, out)
    assert (out != 5).all()


def test_attention_threads_match_single_thread():
    t, meta = load_graph("attention_i8_yolo26")
    q, k, v = t["q"][0], t["k"][0], t["v"][0]
    plan, _ = attention_plan(meta, q.size)
    single = np.zeros(t["output"][0].shape, dtype=np.int8)
    plan.run(q, k, v, single)
    for threads in (2, 4, 64):
        out = np.zeros_like(single)
        plan.run(q, k, v, out, threads=threads)
        np.testing.assert_array_equal(out, single)


def test_new_plans_report_a_tier():
    plans = [
        kernels.BinaryI8Plan("add", (0.1, 0), (0.1, 0), (0.2, 0)),
        kernels.MatMulI8Plan.tflite_batch_matmul((0.1, 0), (0.1, 0), (0.5, 0)),
        kernels.SoftmaxDotI8Plan(
            (0.1, 0),
            [1, 2],
            kernels.MatMulI8Plan.tflite_batch_matmul(I8_PROBS, (1.0, 0), (0.25, 0)),
        ),
        kernels.AttentionI8Plan(
            (0.1, 0), (0.1, 0), (0.1, 0), (0.5, 0), I8_PROBS, (0.2, 0)
        ),
    ]
    for plan in plans:
        assert plan.tier in {"scalar", "neon", "neon_fp16", "neon_dotprod"}
        with pytest.raises(AttributeError):
            plan.tier = "scalar"


def _counter_rate(duration, work=None):
    stop = threading.Event()
    counted = []

    def count():
        n = 0
        while not stop.is_set():
            n += 1
        counted.append(n)

    t = threading.Thread(target=count)
    t.start()
    t0 = time.perf_counter()
    if work is None:
        time.sleep(duration)
    else:
        while time.perf_counter() - t0 < duration:
            work()
    elapsed = time.perf_counter() - t0
    stop.set()
    t.join()
    return counted[0] / elapsed


def test_gil_released():
    plan = kernels.SoftmaxI8Plan(0.05, 0, 1 / 256, -128)
    x = np.random.default_rng(0).integers(-128, 128, size=(4096, 4096), dtype=np.int8)
    out = np.zeros_like(x)

    def real():
        plan.run(x, out)

    def control():
        t0 = time.perf_counter()
        n = 0
        while time.perf_counter() - t0 < 0.05:
            n += 1

    def gap():
        reals, controls = [], []
        for _ in range(6):
            base = _counter_rate(0.05)
            reals.append(_counter_rate(0.05, real) / base)
            controls.append(_counter_rate(0.05, control) / base)
        return statistics.median(reals) - statistics.median(controls)

    best = max(gap() for _ in range(3))
    assert best > 0.1, f"kernel run does not release the GIL (gap {best:.2f})"
