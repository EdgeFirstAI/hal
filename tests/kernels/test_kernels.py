# SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
# SPDX-License-Identifier: Apache-2.0

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
