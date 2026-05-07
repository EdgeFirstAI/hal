"""Pin the public surface of scripts/per_scale_decode_reference.py.

The decoder fixture generator imports these symbols. If any disappears,
the generator breaks silently — this test fails loudly instead.
"""

import importlib
import sys
from pathlib import Path


def test_reference_script_exports_expected_symbols():
    repo_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo_root / "scripts"))
    try:
        mod = importlib.import_module("per_scale_decode_reference")
    finally:
        sys.path.pop(0)

    required = [
        "PerScaleLayout",
        "bind_outputs",
        "dequantize",
        "dfl_decode_level",
        "ltrb_decode_level",
        "sigmoid",
        "decode_per_scale",
        "load_metadata",
    ]
    missing = [s for s in required if not hasattr(mod, s)]
    assert not missing, f"reference script missing required symbols: {missing}"


def test_dfl_decode_level_with_intermediates_returns_three_stages():
    import numpy as np

    repo_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo_root / "scripts"))
    try:
        mod = importlib.import_module("per_scale_decode_reference")
    finally:
        sys.path.pop(0)

    h, w, reg_max, stride = 2, 2, 16, 8
    box_q = np.zeros((1, h, w, 4 * reg_max), dtype=np.int8)
    box_q[0, 0, 0, 0] = 64
    q_scale, q_zp = 0.1, 0

    dequant, ltrb, xywh = mod.dfl_decode_level_with_intermediates(
        box_q, q_scale, q_zp, h, w, stride, reg_max
    )
    assert dequant.shape == (1, h, w, 4 * reg_max)
    assert dequant.dtype == np.float32
    assert ltrb.shape == (h * w, 4)
    assert ltrb.dtype == np.float32
    assert xywh.shape == (h * w, 4)
    assert xywh.dtype == np.float32
    assert ltrb[0, 0] > 0.0


def test_ltrb_decode_level_with_intermediates_returns_two_stages():
    import numpy as np

    repo_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo_root / "scripts"))
    try:
        mod = importlib.import_module("per_scale_decode_reference")
    finally:
        sys.path.pop(0)

    h, w, stride = 2, 2, 8
    box_q = np.zeros((1, h, w, 4), dtype=np.int8)
    q_scale, q_zp = 0.1, 0

    dequant, xywh = mod.ltrb_decode_level_with_intermediates(
        box_q, q_scale, q_zp, h, w, stride
    )
    assert dequant.shape == (1, h, w, 4)
    assert dequant.dtype == np.float32
    assert xywh.shape == (h * w, 4)
    assert xywh.dtype == np.float32


def test_with_intermediates_xywh_matches_original_dfl():
    import numpy as np

    repo_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo_root / "scripts"))
    try:
        mod = importlib.import_module("per_scale_decode_reference")
    finally:
        sys.path.pop(0)
    h, w, reg_max, stride = 4, 4, 16, 8
    rng = np.random.default_rng(0)
    box_q = rng.integers(-128, 127, size=(1, h, w, 4 * reg_max), dtype=np.int8)
    q_scale, q_zp = 0.123, 7
    # Dequantize and reshape to the shape the original function expects: (H, W, 4*reg_max)
    bbox_float = (box_q.astype(np.float32) - float(q_zp)) * float(q_scale)
    xywh_orig = mod.dfl_decode_level(
        bbox_float.reshape(h, w, 4 * reg_max), h, w, stride, reg_max
    )
    _, _, xywh_new = mod.dfl_decode_level_with_intermediates(
        box_q, q_scale, q_zp, h, w, stride, reg_max
    )
    np.testing.assert_allclose(xywh_orig, xywh_new, rtol=0, atol=1e-6)


def test_with_intermediates_xywh_matches_original_ltrb():
    import numpy as np

    repo_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo_root / "scripts"))
    try:
        mod = importlib.import_module("per_scale_decode_reference")
    finally:
        sys.path.pop(0)
    h, w, stride = 4, 4, 8
    rng = np.random.default_rng(1)
    box_q = rng.integers(-128, 127, size=(1, h, w, 4), dtype=np.int8)
    q_scale, q_zp = 0.05, -3
    # Dequantize and reshape to the shape the original function expects: (H, W, 4)
    bbox_float = (box_q.astype(np.float32) - float(q_zp)) * float(q_scale)
    xywh_orig = mod.ltrb_decode_level(bbox_float.reshape(h, w, 4), h, w, stride)
    _, xywh_new = mod.ltrb_decode_level_with_intermediates(
        box_q, q_scale, q_zp, h, w, stride
    )
    np.testing.assert_allclose(xywh_orig, xywh_new, rtol=0, atol=1e-6)
