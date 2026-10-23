# SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
# SPDX-License-Identifier: Apache-2.0

"""Applying a model's preprocessing spec, and the content region a convert
paints: ``ConvertOptions.from_spec``, per-convert interpolation and
``ConvertOptions.content_region``."""

from __future__ import annotations

import shutil
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from edgefirst.image import (
    ContentRegion,
    ConvertOptions,
    Fit,
    ImageProcessor,
    Interpolation,
    LetterboxPlacement,
    PadSpace,
    PixelFormat,
    Region,
    Tensor,
    TilingConfig,
)

ROOT = Path(__file__).resolve().parents[2]
PAD = (0, 0, 0, 255)


def cpu_processor(monkeypatch, **kwargs) -> ImageProcessor:
    monkeypatch.setenv("EDGEFIRST_DISABLE_GL", "1")
    monkeypatch.setenv("EDGEFIRST_DISABLE_G2D", "1")
    monkeypatch.delenv("EDGEFIRST_FORCE_BACKEND", raising=False)
    monkeypatch.delenv("EDGEFIRST_DISABLE_CPU", raising=False)
    monkeypatch.delenv("EDGEFIRST_INTERPOLATION", raising=False)
    return ImageProcessor(**kwargs)


def image(w: int, h: int, fill=None) -> Tensor:
    t = Tensor.image(w, h, format=PixelFormat.Rgba, access="readwrite")
    with t.map() as m:
        a = np.asarray(m.numpy())
        if fill is None:
            rng = np.random.default_rng(w * 1000 + h)
            a[:] = rng.integers(0, 256, size=a.shape, dtype=np.uint8)
            a[..., 3] = 255
        else:
            a[:] = fill
    return t


def read(t: Tensor) -> np.ndarray:
    with t.map() as m:
        return np.array(m.numpy())


def painted_rect(img: np.ndarray) -> tuple[int, int, int, int]:
    """The bounding box of the pixels that are not the black pad."""
    ys, xs = np.nonzero(img[..., :3].any(axis=-1))
    return (
        int(xs.min()),
        int(ys.min()),
        int(xs.max() - xs.min() + 1),
        int(ys.max() - ys.min() + 1),
    )


# --- content region ----------------------------------------------------------


@pytest.mark.parametrize("src", [(641, 479), (479, 641), (1, 3), (3, 1)])
@pytest.mark.parametrize("dst", [(640, 640), (320, 256)])
@pytest.mark.parametrize("placement", ["center", "topleft"])
def test_content_region_matches_the_painted_rect(monkeypatch, src, dst, placement):
    proc = cpu_processor(monkeypatch)
    options = ConvertOptions(letterbox=PAD, placement=placement)
    region = options.content_region(*src, *dst)
    assert isinstance(region, ContentRegion)

    out = Tensor.image(*dst, format=PixelFormat.Rgba, access="readwrite")
    proc.convert(image(*src, fill=255), out, options=options)
    assert painted_rect(read(out)) == (region.x, region.y, region.width, region.height)

    x0, y0, x1, y1 = region.normalized
    assert x0 == pytest.approx(region.x / dst[0])
    assert y0 == pytest.approx(region.y / dst[1])
    assert x1 == pytest.approx((region.x + region.width) / dst[0])
    assert y1 == pytest.approx((region.y + region.height) / dst[1])
    if placement == "topleft":
        assert (region.x, region.y) == (0, 0)


def test_content_region_values():
    region = ConvertOptions(letterbox=PAD).content_region(1280, 720, 640, 640)
    assert (region.x, region.y, region.width, region.height) == (0, 140, 640, 360)
    assert region.normalized == pytest.approx((0.0, 0.21875, 1.0, 0.78125))
    px = region.pixels
    assert isinstance(px, Region)
    assert (px.x, px.y, px.width, px.height) == (0, 140, 640, 360)
    assert not region.is_full
    assert "ContentRegion(" in repr(region)
    assert region == ConvertOptions(letterbox=PAD).content_region(1280, 720, 640, 640)

    stretch = ConvertOptions().content_region(1280, 720, 640, 640)
    assert stretch.is_full
    assert stretch.normalized == (0.0, 0.0, 1.0, 1.0)


def test_content_region_includes_the_source_crop():
    options = ConvertOptions(letterbox=PAD, source=Region(0, 0, 640, 320))
    region = options.content_region(1920, 1080, 640, 640)
    assert (region.y, region.height) == (160, 320)


def test_content_region_rejects_a_crop_outside_the_source():
    options = ConvertOptions(letterbox=PAD, source=Region(100, 0, 640, 640))
    with pytest.raises(ValueError):
        options.content_region(640, 640, 320, 320)


@pytest.mark.parametrize(
    "sizes",
    [(0, 480, 640, 640), (640, 0, 640, 640), (640, 480, 0, 640), (640, 480, 640, 0)],
)
def test_content_region_rejects_a_zero_size(sizes):
    for options in (ConvertOptions(), ConvertOptions(letterbox=PAD)):
        with pytest.raises(ValueError, match="non-zero"):
            options.content_region(*sizes)


# --- placement spellings -------------------------------------------------------


@pytest.mark.parametrize("spelling", [LetterboxPlacement.TopLeft, "topleft", 1])
def test_top_left_placement_spellings_agree(monkeypatch, spelling):
    proc = cpu_processor(monkeypatch)
    src = image(64, 32)
    want = Tensor.image(48, 48, format=PixelFormat.Rgba, access="readwrite")
    proc.convert(
        src, want, letterbox=PAD, letterbox_placement=LetterboxPlacement.TopLeft
    )
    for options in (
        ConvertOptions(letterbox=PAD, placement=spelling),
        ConvertOptions(letterbox=PAD).placement(spelling),
    ):
        got = Tensor.image(48, 48, format=PixelFormat.Rgba, access="readwrite")
        proc.convert(src, got, options=options)
        np.testing.assert_array_equal(read(got), read(want))
    got = Tensor.image(48, 48, format=PixelFormat.Rgba, access="readwrite")
    proc.convert(src, got, letterbox=PAD, letterbox_placement=spelling)
    np.testing.assert_array_equal(read(got), read(want))
    assert ConvertOptions(letterbox=PAD, placement=spelling).content_region(
        64, 32, 48, 48
    ) == ConvertOptions(letterbox=PAD, placement="topleft").content_region(
        64, 32, 48, 48
    )


def test_bad_placement_is_rejected():
    with pytest.raises(ValueError, match="center, topleft"):
        ConvertOptions(placement="bottom")
    with pytest.raises(ValueError):
        ConvertOptions().placement(7)
    with pytest.raises(TypeError):
        ConvertOptions().placement(1.5j)


# --- interpolation -----------------------------------------------------------


@pytest.mark.parametrize(
    "mode", ["bicubic", "area", Interpolation.Bicubic, Interpolation.Area]
)
def test_bicubic_and_area_run_as_bilinear(monkeypatch, mode):
    src = image(40, 30)

    def resize(proc):
        dst = Tensor.image(56, 42, format=PixelFormat.Rgba, access="readwrite")
        proc.convert(src, dst)
        return read(dst)

    bilinear = resize(cpu_processor(monkeypatch, interpolation="bilinear"))
    np.testing.assert_array_equal(
        resize(cpu_processor(monkeypatch, interpolation=mode)), bilinear
    )
    proc = cpu_processor(monkeypatch, interpolation="nearest")
    proc.set_interpolation(mode)
    np.testing.assert_array_equal(resize(proc), bilinear)


def nearest_and_bilinear(monkeypatch, src):
    """The 56x42 resize of ``src`` by processors set to nearest and bilinear."""
    out = []
    for mode in ("nearest", "bilinear"):
        proc = cpu_processor(monkeypatch, interpolation=mode)
        out.append(
            converted(proc, ConvertOptions(), src, 56, 42, PixelFormat.Rgba, "uint8")
        )
    assert not np.array_equal(out[0], out[1]), "the source must tell the filters apart"
    return out


@pytest.mark.parametrize("spelling", [Interpolation.Nearest, "nearest", 1])
def test_per_convert_interpolation_overrides_the_processor(monkeypatch, spelling):
    src = image(40, 30)
    nearest, bilinear = nearest_and_bilinear(monkeypatch, src)
    proc = cpu_processor(monkeypatch)
    for options in (
        ConvertOptions(interpolation=spelling),
        ConvertOptions().interpolation(spelling),
    ):
        got = converted(proc, options, src, 56, 42, PixelFormat.Rgba, "uint8")
        np.testing.assert_array_equal(got, nearest)
    plain = converted(proc, ConvertOptions(), src, 56, 42, PixelFormat.Rgba, "uint8")
    np.testing.assert_array_equal(plain, bilinear)


def test_two_option_sets_on_one_processor_do_not_interfere(monkeypatch):
    src = image(40, 30)
    nearest, bilinear = nearest_and_bilinear(monkeypatch, src)
    proc = cpu_processor(monkeypatch, interpolation="nearest")
    model_a = ConvertOptions.from_spec(spec(fit="stretch", interpolation="nearest"))
    model_b = ConvertOptions.from_spec(spec(fit="stretch", interpolation="bicubic"))
    for _ in range(2):
        a = converted(proc, model_a, src, 56, 42, PixelFormat.Rgba, "uint8")
        b = converted(proc, model_b, src, 56, 42, PixelFormat.Rgba, "uint8")
        np.testing.assert_array_equal(a, nearest)
        np.testing.assert_array_equal(b, bilinear)
    plain = converted(proc, ConvertOptions(), src, 56, 42, PixelFormat.Rgba, "uint8")
    np.testing.assert_array_equal(plain, nearest)


def test_env_pin_overrides_per_convert_interpolation(monkeypatch):
    src = image(40, 30)
    _, bilinear = nearest_and_bilinear(monkeypatch, src)
    monkeypatch.setenv("EDGEFIRST_INTERPOLATION", "bilinear")
    pinned = ImageProcessor()
    options = ConvertOptions(interpolation="nearest")
    got = converted(pinned, options, src, 56, 42, PixelFormat.Rgba, "uint8")
    np.testing.assert_array_equal(got, bilinear)


def test_bad_interpolation_is_rejected():
    with pytest.raises(ValueError, match="unknown interpolation"):
        ConvertOptions(interpolation="lanczos")
    with pytest.raises(ValueError, match="unknown interpolation"):
        ConvertOptions().interpolation(4)
    with pytest.raises(TypeError):
        ConvertOptions().interpolation(1.5j)


def test_interpolation_codes():
    assert [int(m) for m in (Interpolation.Bilinear, Interpolation.Nearest)] == [0, 1]
    assert [int(m) for m in (Interpolation.Bicubic, Interpolation.Area)] == [2, 3]
    assert [int(PadSpace.Pixel), int(PadSpace.Normalized)] == [0, 1]


# --- from_spec -----------------------------------------------------------------


def spec(**kw):
    base = {
        "normalization": ((0.0, 0.0, 0.0), (255.0, 255.0, 255.0)),
        "fit": "letterbox",
        "placement": "center",
        "pad": (114.0, 114.0, 114.0),
        "pad_space": "pixel",
        "interpolation": "bilinear",
        "size": None,
    }
    base.update(kw)
    return SimpleNamespace(**base)


def converted(proc, options, src, w=48, h=48, fmt=PixelFormat.Rgb, dtype="float32"):
    dst = proc.create_image(w, h, fmt, dtype=dtype, access="readwrite")
    proc.convert(src, dst, options=options)
    return read(dst)


def test_from_spec_matches_manual_options(monkeypatch):
    proc = cpu_processor(monkeypatch)
    src = image(64, 32)
    imagenet = ((123.675, 116.28, 103.53), (58.395, 57.12, 57.375))
    s = spec(
        normalization=imagenet,
        placement="topleft",
        pad=(0.0, 0.0, 0.0),
        pad_space="normalized",
    )
    manual = (
        ConvertOptions()
        .letterbox((0, 0, 0, 255))
        .placement(LetterboxPlacement.TopLeft)
        .pad_output((0.0, 0.0, 0.0, 1.0))
        .normalization("imagenet")
    )
    np.testing.assert_array_equal(
        converted(proc, ConvertOptions.from_spec(s), src),
        converted(proc, manual, src),
    )


def test_from_spec_accepts_enums_codes_and_mean_std(monkeypatch):
    proc = cpu_processor(monkeypatch)
    src = image(64, 32)
    by_enum = spec(
        fit=Fit.Letterbox,
        placement=LetterboxPlacement.TopLeft,
        pad_space=PadSpace.Pixel,
        interpolation=Interpolation.Nearest,
    )
    by_code = spec(fit=1, placement=1, pad_space=0, interpolation=1)
    by_name = spec(placement="topleft", interpolation="nearest")
    want = converted(proc, ConvertOptions.from_spec(by_name), src)
    for s in (by_enum, by_code):
        np.testing.assert_array_equal(
            converted(proc, ConvertOptions.from_spec(s), src), want
        )
    mean_std = spec(placement="topleft", interpolation="nearest")
    del mean_std.normalization
    mean_std.mean = (0.0, 0.0, 0.0)
    mean_std.std = (255.0, 255.0, 255.0)
    np.testing.assert_array_equal(
        converted(proc, ConvertOptions.from_spec(mean_std), src), want
    )
    stretch = ConvertOptions.from_spec(spec(fit="stretch"))
    assert stretch.content_region(64, 32, 48, 48).is_full


def test_from_spec_carries_the_interpolation(monkeypatch):
    src = image(40, 30)
    nearest, _ = nearest_and_bilinear(monkeypatch, src)
    proc = cpu_processor(monkeypatch)
    options = ConvertOptions.from_spec(spec(fit="stretch", interpolation="nearest"))
    got = converted(proc, options, src, 56, 42, PixelFormat.Rgba, "uint8")
    np.testing.assert_array_equal(got, nearest)
    kept = ConvertOptions(interpolation="nearest").with_spec(spec(fit="stretch"))
    got = converted(proc, kept, src, 56, 42, PixelFormat.Rgba, "uint8")
    assert not np.array_equal(got, nearest), "with_spec takes the spec's interpolation"


def missing(name):
    s = spec()
    delattr(s, name)
    return s


@pytest.mark.parametrize(
    "bad",
    [
        None,
        {"fit": "letterbox"},
        object(),
        *(
            missing(n)
            for n in ("fit", "placement", "pad", "pad_space", "interpolation", "size")
        ),
        missing("normalization"),
        spec(normalization=None),
        spec(fit=None),
        spec(pad=None),
        spec(size=(640, "a")),
    ],
)
def test_from_spec_rejects_what_is_not_a_spec(bad):
    with pytest.raises(TypeError):
        ConvertOptions.from_spec(bad)
    with pytest.raises(TypeError):
        ConvertOptions().with_spec(bad)


def test_with_spec_keeps_the_source_crop():
    crop = ConvertOptions(source=Region(0, 0, 32, 32)).with_spec(spec())
    assert crop.content_region(64, 32, 48, 48).is_full


@pytest.mark.parametrize(
    "bad",
    [
        {"pad": (300.0, 0.0, 0.0)},
        {"pad": (float("nan"),) * 3},
        {"pad": (1.0, 2.0)},
        {"fit": "fill"},
        {"pad_space": "linear"},
        {"interpolation": "lanczos"},
        {"normalization": "imagenet2"},
        {"size": (-1, 640)},
        {"size": (640, -640)},
        {"size": (0, 0)},
        {"size": (640, 0)},
    ],
)
def test_bad_spec_is_rejected(bad):
    with pytest.raises(ValueError):
        ConvertOptions.from_spec(spec(**bad))


def test_from_spec_from_a_resolved_config(monkeypatch, tmp_path):
    dec = pytest.importorskip("edgefirst.decoder")
    for name in ("config.json", "preprocessor_config.json"):
        shutil.copy(ROOT / "testdata/hf/segformer-512" / name, tmp_path / name)
    model = tmp_path / "model.onnx"
    model.write_bytes(b"")
    signals = dec.ModelSignals(
        "onnx",
        [("pixel_values", [1, 3, 512, 512], "f32")],
        [("logits", [1, 150, 128, 128], "f32")],
    )
    plan = dec.resolve_model_config(model, signals).preprocessing

    proc = cpu_processor(monkeypatch)
    options = ConvertOptions.from_spec(plan)
    src = image(64, 32)
    np.testing.assert_array_equal(
        converted(proc, options, src, fmt=PixelFormat.PlanarRgb),
        converted(
            proc,
            ConvertOptions(normalization="imagenet"),
            src,
            fmt=PixelFormat.PlanarRgb,
        ),
    )
    assert options.content_region(640, 480, 512, 512).is_full

    fields = ("normalization", "fit", "pad", "pad_space", "interpolation", "size")
    topleft = SimpleNamespace(
        **{f: getattr(plan, f) for f in fields}, placement="topleft"
    )
    topleft.fit = "letterbox"
    region = ConvertOptions.from_spec(topleft).content_region(640, 480, 512, 512)
    assert (region.x, region.y, region.width, region.height) == (0, 0, 512, 384)


# --- tiling options ------------------------------------------------------------


def test_tiling_config_takes_convert_options():
    options = ConvertOptions(letterbox=PAD, placement="topleft")
    cfg = TilingConfig(640, 640, options=options)
    assert repr(cfg.options) == repr(options)
    assert repr(TilingConfig(640, 640).with_options(options).options) == repr(options)
    plan = ImageProcessor().plan_tiles(1500, 400, cfg)
    assert len(plan) >= 3
    for p in plan:
        assert p.content_region == pytest.approx((0.0, 0.0, 1.0, 0.625))


def test_tiling_options_refuse_rotation():
    from edgefirst.image import Rotation

    cfg = TilingConfig(640, 640, options=ConvertOptions(rotation=Rotation.Clockwise90))
    with pytest.raises(RuntimeError, match="rotate or flip"):
        ImageProcessor().plan_tiles(1920, 1080, cfg)
