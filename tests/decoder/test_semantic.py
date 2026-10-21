# SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
# SPDX-License-Identifier: Apache-2.0

import json
import struct
from pathlib import Path

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
    assert (out[:, 0] == 0).all()
    assert (out[:, 6] == 1).all()
    assert (out[:, 11] == 2).all()


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
    scores = [tensor(stripes())]
    float_dst = ef.Tensor([4, 4], dtype="float32")
    with pytest.raises(RuntimeError):
        decoder.decode_semantic(scores, float_dst)
    dst = ef.Tensor([4, 4], dtype="uint8")
    with pytest.raises(RuntimeError):
        decoder.decode_semantic(scores, dst, letterbox=[0.5, 0.0, 0.5, 1.0])
    with pytest.raises(RuntimeError):
        decoder.decode(scores)


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
    outputs = [tensor(np.zeros((1, 84, 8400), dtype=np.float32))]
    dst = ef.Tensor([4, 4], dtype="uint8")
    with pytest.raises(RuntimeError):
        decoder.decode_semantic(outputs, dst)


SEGFORMER_FIXTURES = [
    "segformer-b1-ade-512x672_zidane.safetensors",
    "segformer-b1-ade-512x512_jaguar.safetensors",
    "segformer-b1-ade-512x512_coco_420_odd.safetensors",
]
SAFETENSORS_DTYPES = {
    "F32": np.float32,
    "F16": np.float16,
    "I8": np.int8,
    "U8": np.uint8,
}


def load_safetensors(path):
    """Tensors and metadata of a safetensors file, without the safetensors package."""
    data = path.read_bytes()
    if data.startswith(b"version https://git-lfs"):
        pytest.skip(f"{path} is an LFS pointer (run `git lfs pull`)")
    (size,) = struct.unpack("<Q", data[:8])
    header = json.loads(data[8 : 8 + size])
    metadata = header.pop("__metadata__", {})
    body = data[8 + size :]
    tensors = {
        key: np.frombuffer(body[b:e], dtype=SAFETENSORS_DTYPES[info["dtype"]]).reshape(
            info["shape"]
        )
        for key, info in header.items()
        for b, e in [info["data_offsets"]]
    }
    return tensors, metadata


@pytest.mark.parametrize("name", SEGFORMER_FIXTURES)
@pytest.mark.parametrize("suffix", ["", "_i8"])
def test_segformer_outputs_match_the_reference_decode(name, suffix):
    path = Path(__file__).resolve().parents[2] / "testdata" / "decoder" / name
    if not path.exists():
        pytest.skip(f"{path} not present (run `git lfs pull`)")
    tensors, metadata = load_safetensors(path)
    decoder = ef.Decoder.new_from_json_str(metadata["edgefirst_json" + suffix])
    logits = tensor(tensors["logits" + suffix])
    letterbox = json.loads(metadata["letterbox"])
    for key, mode, floor in (
        ("accurate", ef.DecodeMode.Accurate, 0.998),
        ("fast", ef.DecodeMode.Fast, 0.97),
    ):
        reference = tensors[f"ref.{key}{suffix}"]
        dst = ef.Tensor(list(reference.shape), dtype="uint8")
        decoder.decode_semantic([logits], dst, mode=mode, letterbox=letterbox)
        agreement = (read(dst, np.uint8) == reference).mean()
        assert agreement >= floor, f"{key}: {agreement:.4%}"


def test_programmatic_semantic_output_decodes_like_json():
    dshape = [
        (ef.DimName.Batch, 1),
        (ef.DimName.NumClasses, 3),
        (ef.DimName.Height, 4),
        (ef.DimName.Width, 6),
    ]
    output = ef.Output.segmentation(dshape=dshape, decoder=ef.DecoderType.Semantic)
    decoder = ef.Decoder.new_from_outputs([output])
    assert decoder.semantic_classes == 3
    reference = ef.Decoder.new_from_json_str(json.dumps(SCHEMA))
    outputs = [tensor(stripes())]
    for mode in (ef.DecodeMode.Accurate, ef.DecodeMode.Fast):
        dst = ef.Tensor([8, 12], dtype="uint8")
        expected = ef.Tensor([8, 12], dtype="uint8")
        decoder.decode_semantic(outputs, dst, mode=mode)
        reference.decode_semantic(outputs, expected, mode=mode)
        assert read(dst, np.uint8).tolist() == read(expected, np.uint8).tolist()


@pytest.mark.parametrize("access", ["read", "none"])
def test_decode_semantic_into_a_destination_declared_without_write(access):
    decoder = ef.Decoder.new_from_json_str(json.dumps(SCHEMA))
    dst = ef.Tensor.image(4, 4, ef.PixelFormat.Grey, access=access)
    dst.reshape([4, 4])
    outputs = [tensor(stripes())]
    with pytest.raises(PermissionError, match="CPU access denied"):
        decoder.decode_semantic(outputs, dst)
