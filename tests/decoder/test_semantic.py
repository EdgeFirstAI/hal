# SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
# SPDX-License-Identifier: Apache-2.0

import json

import edgefirst.decoder as ef
import numpy as np
import pytest

SCHEMA = {
    "schema_version": 2,
    "outputs": [
        {
            "name": "logits",
            "type": "segmentation",
            "decoder": "semantic",
            "shape": [1, 3, 4, 6],
            "dshape": [{"batch": 1}, {"num_classes": 3}, {"height": 4}, {"width": 6}],
            "dtype": "float32",
            "upsample": {"method": "bilinear", "align_corners": False},
            "decode_mode": "accurate",
        }
    ],
}


def tensor(arr):
    t = ef.Tensor(list(arr.shape), dtype=str(arr.dtype))
    t.from_numpy(np.ascontiguousarray(arr))
    return t


def read(t, dtype):
    with t.map() as m:
        return np.frombuffer(m, dtype=dtype).reshape(t.shape).copy()


def stripes():
    """Class k wins columns [2k, 2k + 2)."""
    logits = np.zeros((1, 3, 4, 6), dtype=np.float32)
    for k in range(3):
        logits[0, k, :, 2 * k : 2 * k + 2] = 1.0
    return logits


def test_semantic_properties():
    decoder = ef.Decoder.new_from_json_str(json.dumps(SCHEMA))
    assert decoder.semantic_classes == 3
    assert decoder.semantic_mode == ef.DecodeMode.Accurate


@pytest.mark.parametrize("mode", [ef.DecodeMode.Accurate, ef.DecodeMode.Fast])
def test_decode_semantic_upsamples(mode):
    decoder = ef.Decoder.new_from_json_str(json.dumps(SCHEMA))
    dst = ef.Tensor([8, 12], dtype="uint8")
    decoder.decode_semantic([tensor(stripes())], dst, mode=mode)
    out = read(dst, np.uint8)
    assert out.shape == (8, 12)
    assert (out[:, 0] == 0).all() and (out[:, 6] == 1).all() and (out[:, 11] == 2).all()


def test_decode_semantic_letterbox_and_confidence():
    decoder = ef.Decoder.new_from_json_str(json.dumps(SCHEMA))
    dst = ef.Tensor([4, 4], dtype="uint16")
    conf = ef.Tensor([4, 4], dtype="float32")
    # Keep the left two thirds: classes 0 and 1 only.
    decoder.decode_semantic(
        [tensor(stripes())],
        dst,
        conf,
        mode=ef.DecodeMode.Fast,
        letterbox=[0.0, 0.0, 4 / 6, 1.0],
    )
    out = read(dst, np.uint16)
    assert out.tolist() == [[0, 0, 1, 1]] * 4
    p = read(conf, np.float32)
    expected = np.e / (np.e + 2)
    assert np.allclose(p, expected, atol=1e-6)


def test_decode_semantic_errors():
    decoder = ef.Decoder.new_from_json_str(json.dumps(SCHEMA))
    with pytest.raises(RuntimeError):
        decoder.decode_semantic([tensor(stripes())], ef.Tensor([4, 4], dtype="float32"))
    with pytest.raises(RuntimeError):
        decoder.decode_semantic(
            [tensor(stripes())],
            ef.Tensor([4, 4], dtype="uint8"),
            letterbox=[0.5, 0.0, 0.5, 1.0],
        )
    with pytest.raises(RuntimeError):
        boxes = decoder.decode([tensor(stripes())])
        del boxes


def test_detection_decoder_has_no_semantic_output():
    schema = {
        "schema_version": 2,
        "outputs": [
            {
                "name": "out",
                "type": "detection",
                "decoder": "ultralytics",
                "shape": [1, 84, 8400],
                "dshape": [{"batch": 1}, {"num_features": 84}, {"num_boxes": 8400}],
                "dtype": "float32",
            }
        ],
    }
    decoder = ef.Decoder.new_from_json_str(json.dumps(schema))
    assert decoder.semantic_classes is None
    assert decoder.semantic_mode is None
    with pytest.raises(RuntimeError):
        decoder.decode_semantic(
            [tensor(np.zeros((1, 84, 8400), dtype=np.float32))],
            ef.Tensor([4, 4], dtype="uint8"),
        )
