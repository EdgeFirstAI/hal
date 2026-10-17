# SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
# SPDX-License-Identifier: Apache-2.0

"""ConvertOptions: normalization, quantized output, letterbox placement and
output-domain padding, against a NumPy reference."""

import numpy as np
import pytest
from edgefirst.image import (
    ConvertOptions,
    ImageProcessor,
    LetterboxPlacement,
    NormalizationPreset,
    PixelFormat,
    Tensor,
)

IMAGENET_MEAN = np.array([123.675, 116.28, 103.53])
IMAGENET_STD = np.array([58.395, 57.12, 57.375])


def source(w=32, h=16):
    rng = np.random.default_rng(7)
    rgba = rng.integers(0, 256, size=(h, w, 4), dtype=np.uint8)
    rgba[..., 3] = 255
    src = Tensor.image(w, h, format=PixelFormat.Rgba, access="readwrite")
    with src.map() as m:
        np.asarray(m.numpy())[:] = rgba
    return src, rgba


def read(t):
    with t.map() as m:
        return np.array(m.numpy())


def test_imagenet_planar_f32_matches_numpy():
    proc = ImageProcessor()
    src, rgba = source()
    dst = proc.create_image(
        32, 16, PixelFormat.PlanarRgb, dtype="float32", access="readwrite"
    )
    proc.convert(src, dst, normalization=NormalizationPreset.Imagenet)
    want = ((rgba[..., :3] - IMAGENET_MEAN) / IMAGENET_STD).transpose(2, 0, 1)
    np.testing.assert_allclose(read(dst).reshape(want.shape), want, atol=0.02)


def test_normalization_spellings_agree():
    proc = ImageProcessor()
    src, _ = source()
    outs = []
    for norm in (
        NormalizationPreset.Signed,
        "signed",
        ([127.5], [127.5]),
        ([127.5, 127.5, 127.5], [127.5, 127.5, 127.5]),
    ):
        dst = proc.create_image(
            32, 16, PixelFormat.Rgb, dtype="float32", access="readwrite"
        )
        proc.convert(src, dst, options=ConvertOptions(normalization=norm))
        outs.append(read(dst))
    for o in outs[1:]:
        np.testing.assert_array_equal(o, outs[0])
    assert outs[0].min() >= -1.0 and outs[0].max() <= 1.0


def test_top_left_letterbox_with_output_pad():
    proc = ImageProcessor()
    src, _ = source(32, 16)
    dst = proc.create_image(
        32, 32, PixelFormat.PlanarRgb, dtype="float32", access="readwrite"
    )
    opts = (
        ConvertOptions()
        .letterbox((0, 0, 0, 255))
        .placement(LetterboxPlacement.TopLeft)
        .pad_output((0.0, 0.0, 0.0, 0.0))
        .normalization("imagenet")
    )
    proc.convert(src, dst, options=opts)
    planes = read(dst).reshape(3, 32, 32)
    assert np.all(planes[:, 16:, :] == 0.0)
    assert np.any(planes[:, :16, :] != 0.0)


def test_int8_needs_quantization_and_uses_it():
    proc = ImageProcessor()
    src, rgba = source()
    dst = proc.create_image(32, 16, PixelFormat.Rgb, dtype="int8", access="readwrite")
    with pytest.raises(RuntimeError, match="quantization"):
        proc.convert(src, dst, normalization="imagenet")
    dst.set_quantization_per_tensor(0.0187, -14)
    proc.convert(src, dst, normalization="imagenet")
    y = (rgba[..., :3] - IMAGENET_MEAN) / IMAGENET_STD
    want = np.clip(np.floor(y / 0.0187 + 0.5) - 14, -128, 127)
    got = read(dst).reshape(want.shape).astype(np.float64)
    assert np.max(np.abs(got - want)) <= 1


def test_unknown_normalization_name_is_refused():
    with pytest.raises(RuntimeError, match="unknown normalization"):
        ConvertOptions(normalization="bogus")


def test_enum_codes_match_the_c_vocabulary():
    assert [
        int(p) for p in (LetterboxPlacement.Center, LetterboxPlacement.TopLeft)
    ] == [0, 1]
    presets = (
        NormalizationPreset.Unsigned,
        NormalizationPreset.Signed,
        NormalizationPreset.Raw,
        NormalizationPreset.Imagenet,
    )
    assert [int(p) for p in presets] == [0, 1, 2, 3]
