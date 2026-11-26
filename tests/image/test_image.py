# SPDX-FileCopyrightText: Copyright 2025 Au-Zone Technologies
# SPDX-License-Identifier: Apache-2.0

import math
import os
import sys

import numpy as np
import pytest
from edgefirst.codec import ImageInfo, decode_file_into, decode_into
from edgefirst.decoder import Decoder
from edgefirst.image import (
    ColorMode,
    Flip,
    ImageProcessor,
    NormalizationPreset,
    PixelFormat,
    Rotation,
    Tensor,
)
from edgefirst.tensor import ColorEncoding, Colorimetry, ColorRange, TensorMemory
from PIL import Image

from tests.dma_skip import dma_unavailable, skip_dma
from tests.gpu_policy import skip_unless_gpu_backed


def load_image(image, format="RGBA", resize=None):
    im = Image.open(image).convert(format)
    if resize is not None:
        im = im.resize(resize)
    return np.array(im)


def calculate_similarity_rms_u8(imageA, imageB) -> float:
    imgA = np.asarray(imageA)
    imgB = np.asarray(imageB)

    imgA = imgA.astype("float") / 255.0
    imgB = imgB.astype("float") / 255.0

    squared_diff = (imgA - imgB) ** 2

    rms = math.sqrt(float(np.mean(squared_diff)))
    return 1.0 - rms


# Maximum source tensor dimensions for the decode tests.
MAX_SRC_W, MAX_SRC_H = 1920, 1080


def _image_size(path):
    """Read image dimensions from file header without decoding pixels."""
    with Image.open(path) as im:
        return im.size  # (width, height)


def test_flip():
    w, h = _image_size("testdata/zidane.jpg")
    src = Tensor.image(w, h, format=PixelFormat.Nv12, access="readwrite")
    decode_file_into(src, "testdata/zidane.jpg")
    converter = ImageProcessor()
    dst = converter.create_image(1280, 720, PixelFormat.Rgba, access="readwrite")
    converter.convert(src, dst, flip=Flip.Horizontal)

    n = np.zeros((720, 1280, 3), dtype=np.uint8)
    dst.normalize_to_numpy(n)

    expected = load_image("testdata/zidane.jpg", "RGB")
    expected = expected[:, ::-1, :]

    assert calculate_similarity_rms_u8(n, expected) > 0.98


def test_grey_load():
    w, h = _image_size("testdata/grey.jpg")
    converter = ImageProcessor()

    grey_ = Tensor.image(w, h, format=PixelFormat.Grey, access="readwrite")
    info = decode_file_into(grey_, "testdata/grey.jpg")
    assert info.format == PixelFormat.Grey
    grey = np.zeros((h, w, 1), dtype=np.uint8)
    grey_.normalize_to_numpy(grey)

    rgba_ = converter.create_image(w, h, PixelFormat.Rgba, access="readwrite")
    converter.convert(grey_, rgba_)
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    rgba_.normalize_to_numpy(rgba)

    default_ = converter.create_image(w, h, PixelFormat.Rgb, access="readwrite")
    converter.convert(grey_, default_)
    default = np.zeros((h, w, 3), dtype=np.uint8)
    default_.normalize_to_numpy(default)

    assert calculate_similarity_rms_u8(rgba[:, :, 0:3], default) > 0.99
    assert calculate_similarity_rms_u8(rgba[:, :, 0], grey[:, :, 0]) > 0.99
    assert calculate_similarity_rms_u8(default[:, :, 0], grey[:, :, 0]) > 0.99


def test_normalize():
    w, h = _image_size("testdata/zidane.jpg")
    src = Tensor.image(w, h, format=PixelFormat.Nv12, access="readwrite")
    decode_file_into(src, "testdata/zidane.jpg")
    converter = ImageProcessor()
    dst = converter.create_image(640, 640, PixelFormat.Rgba, access="readwrite")
    converter.convert(src, dst)
    n = np.zeros((640, 640, 3), dtype=np.int8)
    dst.normalize_to_numpy(n, normalization=NormalizationPreset.Signed)
    unique_vals = np.unique(n)
    assert (unique_vals == np.array(range(-128, 128))).all()

    dst.normalize_to_numpy(n, normalization=NormalizationPreset.Signed, zero_point=127)
    unique_vals0 = np.unique(n)
    assert (unique_vals0 == np.array(range(-127, 128))).all()


def test_render():
    w, h = _image_size("testdata/giraffe.jpg")
    converter = ImageProcessor()
    bg_native = Tensor.image(w, h, format=PixelFormat.Nv12, access="readwrite")
    decode_file_into(bg_native, "testdata/giraffe.jpg")
    bg = Tensor.image(w, h, format=PixelFormat.Rgba, access="readwrite")
    converter.convert(bg_native, bg)
    seg = np.fromfile("testdata/yolov8_seg_crop_76x55.bin", dtype=np.uint8).reshape(
        (76, 55, 1)
    )
    converter.set_class_colors([[255, 255, 0, 233], [128, 128, 255, 100]])
    dst = converter.create_image(w, h, PixelFormat.Rgba, access="readwrite")
    converter.draw_decoded_masks(
        dst,
        bbox=np.array([[0.59375, 0.25, 0.9375, 0.725]], dtype=np.float32),
        scores=np.array([0.9], dtype=np.float32),
        classes=np.array([1], dtype=np.uintp),
        seg=[seg],
        background=bg,
    )
    expected_gl = load_image("testdata/output_render_gl.jpg", "RGBA")
    expected_cpu = load_image("testdata/output_render_cpu.jpg", "RGBA")
    with dst.map() as m:
        img = np.asarray(m.numpy())
        assert (
            calculate_similarity_rms_u8(img, expected_gl) > 0.95
            or calculate_similarity_rms_u8(img, expected_cpu) > 0.95
        )


def _draw_box(converter, color, opacity=1.0):
    bg_px = np.array([10, 20, 30, 200], dtype=np.uint8)
    bg = Tensor.image(32, 32, format=PixelFormat.Rgba, access="readwrite")
    with bg.map() as m:
        np.asarray(m.numpy())[:] = bg_px
    converter.set_class_colors([color])
    dst = converter.create_image(32, 32, PixelFormat.Rgba, access="readwrite")
    converter.draw_decoded_masks(
        dst,
        bbox=np.array([[0.25, 0.25, 0.75, 0.75]], dtype=np.float32),
        scores=np.array([0.9], dtype=np.float32),
        classes=np.array([0], dtype=np.uintp),
        background=bg,
        opacity=opacity,
    )
    with dst.map() as m:
        img = np.asarray(m.numpy()).copy()
    drawn = np.flatnonzero((img[16] != bg_px).any(axis=-1))
    return img, drawn


def test_box_line_width():
    converter = ImageProcessor()
    _, drawn = _draw_box(converter, [200, 100, 50, 255])
    assert drawn.tolist() == [5, 6, 7, 24, 25, 26]
    converter.set_box_line_width(5)
    _, drawn = _draw_box(converter, [200, 100, 50, 255])
    assert drawn.tolist() == [3, 4, 5, 6, 7, 24, 25, 26, 27, 28]
    with pytest.raises(ValueError):
        converter.set_box_line_width(0)
    with pytest.raises(OverflowError):
        converter.set_box_line_width(-1)


def test_box_colour_alpha():
    converter = ImageProcessor()
    img, _ = _draw_box(converter, [200, 100, 50, 255])
    assert img[16, 7].tolist() == [200, 100, 50, 200]
    img, _ = _draw_box(converter, [200, 100, 50, 128])
    # 50% coverage over (10, 20, 30); the destination alpha is untouched.
    expected = np.array([105, 60, 40, 200])
    assert np.abs(img[16, 7].astype(int) - expected).max() <= 2
    assert img[16, 16].tolist() == [10, 20, 30, 200]


def test_rgb_resize():
    w, h = _image_size("testdata/zidane.jpg")
    src = Tensor.image(w, h, format=PixelFormat.Nv12, access="readwrite")
    decode_file_into(src, "testdata/zidane.jpg")
    converter = ImageProcessor()
    dst = converter.create_image(640, 640, PixelFormat.Rgba, access="readwrite")
    converter.convert(src, dst)
    with dst.map() as m:
        n = np.asarray(m.numpy())
        expected = load_image("testdata/zidane.jpg", "RGBA", resize=(640, 640))
        assert calculate_similarity_rms_u8(n, expected) > 0.98


def test_interpolation(monkeypatch):
    # A resized checkerboard: "nearest" copies source pixels, "bilinear"
    # blends them, and EDGEFIRST_INTERPOLATION overrides the constructor.
    monkeypatch.delenv("EDGEFIRST_INTERPOLATION", raising=False)
    checker = np.indices((30, 40)).sum(axis=0) % 2 * 255
    rgba = np.stack([checker] * 3 + [np.full_like(checker, 255)], axis=-1).astype(
        np.uint8
    )

    def resize(processor):
        src = Tensor.image(40, 30, format=PixelFormat.Rgba, access="readwrite")
        with src.map() as m:
            np.asarray(m.numpy())[:] = rgba
        dst = processor.create_image(56, 42, PixelFormat.Rgba, access="readwrite")
        processor.convert(src, dst)
        with dst.map() as m:
            return np.array(m.numpy())

    blended = lambda img: np.count_nonzero((img[..., 0] > 0) & (img[..., 0] < 255))
    assert blended(resize(ImageProcessor(interpolation="nearest"))) == 0
    assert blended(resize(ImageProcessor(interpolation="bilinear"))) > 0
    processor = ImageProcessor()
    processor.set_interpolation("nearest")
    assert blended(resize(processor)) == 0
    with pytest.raises(ValueError, match="unknown interpolation"):
        ImageProcessor(interpolation="lanczos")

    monkeypatch.setenv("EDGEFIRST_INTERPOLATION", "nearest")
    assert blended(resize(ImageProcessor(interpolation="bilinear"))) == 0


def test_rgba_to_rgb():
    w, h = _image_size("testdata/zidane.jpg")
    src = Tensor.image(w, h, format=PixelFormat.Nv12, access="readwrite")
    decode_file_into(src, "testdata/zidane.jpg")
    converter = ImageProcessor()
    dst = converter.create_image(w, h, PixelFormat.Rgb, access="readwrite")
    converter.convert(src, dst, Rotation.Rotate0, Flip.NoFlip)

    with dst.map() as m:
        n = np.asarray(m.numpy())
        expected = load_image("testdata/zidane.jpg", "RGB")
        assert calculate_similarity_rms_u8(n, expected) > 0.98


@pytest.mark.parametrize(
    "fixture,fmt",
    [
        ("testdata/zidane_422.jpg", PixelFormat.Nv16),
        ("testdata/zidane_444.jpg", PixelFormat.Nv24),
    ],
    ids=["nv16_422", "nv24_444"],
)
def test_decode_native_nv16_nv24(fixture, fmt):
    """The decoder selects NV16 or NV24 from the JPEG subsampling."""
    if not os.path.exists(fixture):
        pytest.skip(f"{fixture} fixture missing")
    w, h = _image_size(fixture)
    src = Tensor.image(w, h, format=fmt, access="readwrite")
    assert src.format == fmt
    info = decode_file_into(src, fixture)
    assert info.format == fmt
    assert (info.width, info.height) == (w, h)

    converter = ImageProcessor()
    dst = converter.create_image(w, h, PixelFormat.Rgb, access="readwrite")
    converter.convert(src, dst)
    n = np.zeros((h, w, 3), dtype=np.uint8)
    dst.normalize_to_numpy(n)
    expected = load_image(fixture, "RGB")
    assert calculate_similarity_rms_u8(n, expected) > 0.95


@pytest.mark.parametrize(
    "fixture,fmt",
    [
        ("testdata/coco_420_odd.jpg", PixelFormat.Nv12),
        ("testdata/coco_422_odd.jpg", PixelFormat.Nv16),
        ("testdata/coco_444_odd.jpg", PixelFormat.Nv24),
        ("testdata/coco_grey_odd.jpg", PixelFormat.Grey),
    ],
    ids=["nv12_420", "nv16_422", "nv24_444", "grey"],
)
def test_decode_native_odd_dimensions(fixture, fmt):
    """Odd-dimension JPEGs decode to their native format and convert to RGB."""
    if not os.path.exists(fixture):
        pytest.skip(f"{fixture} fixture missing")
    w, h = _image_size(fixture)
    assert (w % 2 == 1) or (h % 2 == 1), f"{fixture}: expected an odd dimension"

    src = Tensor.image(w, h, format=fmt, access="readwrite")
    assert src.format == fmt
    info = decode_file_into(src, fixture)
    assert info.format == fmt
    assert (info.width, info.height) == (w, h)

    converter = ImageProcessor()
    dst = converter.create_image(w, h, PixelFormat.Rgb, access="readwrite")
    converter.convert(src, dst)
    n = np.zeros((h, w, 3), dtype=np.uint8)
    dst.normalize_to_numpy(n)
    expected = load_image(fixture, "RGB")
    threshold = 0.98 if fmt == PixelFormat.Grey else 0.95
    assert calculate_similarity_rms_u8(n, expected) > threshold


def test_set_output_format_fused_rgb():
    """set_output_format(PixelFormat.Rgb) makes a 4:4:4 JPEG decode report Rgb."""
    import edgefirst.codec as hal

    fixture = "testdata/zidane_444.jpg"
    if not os.path.exists(fixture):
        pytest.skip(f"{fixture} fixture missing")
    w, h = _image_size(fixture)
    hal.set_output_format(PixelFormat.Rgb)
    try:
        src = Tensor.image(w, h, format=PixelFormat.Rgb, access="readwrite")
        info = decode_file_into(src, fixture)
        assert info.format == PixelFormat.Rgb
        assert (info.width, info.height) == (w, h)
        n = np.zeros((h, w, 3), dtype=np.uint8)
        src.normalize_to_numpy(n)
        expected = load_image(fixture, "RGB")
        assert calculate_similarity_rms_u8(n, expected) > 0.95
    finally:
        hal.set_output_format(None)


def test_set_dct_method_fast_decodes():
    """set_dct_method(DctMethod.Fast) decodes to the same format and dimensions."""
    import edgefirst.codec as hal

    w, h = _image_size("testdata/zidane.jpg")
    hal.set_dct_method(hal.DctMethod.Fast)
    try:
        src = Tensor.image(w, h, format=PixelFormat.Nv12, access="readwrite")
        info = decode_file_into(src, "testdata/zidane.jpg")
    finally:
        hal.set_dct_method(hal.DctMethod.Accurate)
    assert info.format == PixelFormat.Nv12
    assert (info.width, info.height) == (w, h)


def test_is_v4l2_available_returns_bool():
    """is_v4l2_available() returns a bool."""
    import edgefirst.codec as hal

    assert isinstance(hal.is_v4l2_available(), bool)


# ---------------------------------------------------------------------------
# Strided zero-copy and DMA buffer-protocol tests.
# ---------------------------------------------------------------------------


def _skip_unless_pitch_is_padded_and_aligned(t, tight_row_bytes):
    """Skip unless this adapter gave the tensor a 64-byte-aligned padded row."""
    if t.row_stride == tight_row_bytes or t.row_stride % 64 != 0:
        pytest.skip(
            f"this adapter returns a {t.row_stride}-byte pitch for a "
            f"{tight_row_bytes}-byte row; the padded path is exercised on "
            "adapters that pad to a 64-byte multiple"
        )


def _dma_image_or_skip(w, h, fmt):
    """Allocate a DMA-backed image tensor, or skip when DMA is unavailable."""
    try:
        t = Tensor.image(w, h, format=fmt, mem=TensorMemory.DMABUF, access="readwrite")
    except (AttributeError, RuntimeError):
        skip_dma("DMA memory not supported on this platform")
    if t.memory != TensorMemory.DMABUF:
        skip_dma("DMA requested but backend substituted another memory type")
    return t


@pytest.mark.parametrize(
    "fmt,channels",
    [
        (PixelFormat.Rgb, 3),
        (PixelFormat.Rgba, 4),
        (PixelFormat.Grey, 1),
    ],
    ids=["rgb", "rgba", "grey"],
)
def test_dma_padded_buffer_protocol_zero_copy(fmt, channels):
    """A padded-pitch DMA image tensor exposes a strided, zero-copy buffer."""
    w, h = 595, 438  # odd width
    t = _dma_image_or_skip(w, h, fmt)
    _skip_unless_pitch_is_padded_and_aligned(t, w * channels)

    ref = (np.arange(h * w * channels, dtype=np.uint8) % 251).reshape(h, w, channels)
    t.from_numpy(ref)

    with t.map() as m:
        mv = memoryview(m)
        assert mv.shape == (h, w, channels)
        assert mv.strides[0] == t.row_stride
        arr = np.asarray(mv)
        assert np.array_equal(arr, ref), "strided buffer-protocol read sheared the data"
        poked = (int(ref[0, 0, 0]) ^ 0xFF) & 0xFF
        memoryview(m)[0, 0, 0] = poked

    with t.map() as m2:
        arr2 = np.asarray(memoryview(m2))
        assert int(arr2[0, 0, 0]) == poked, "mutation did not reach the DMA-BUF"

    if channels == 3:
        t.from_numpy(ref)
        out = np.zeros((h, w, 3), dtype=np.uint8)
        t.normalize_to_numpy(out, NormalizationPreset.Raw, None)
        assert np.array_equal(out, ref), "normalize_to_numpy sheared the padded buffer"


@pytest.mark.parametrize(
    "fmt",
    [PixelFormat.Nv12, PixelFormat.Nv16, PixelFormat.Nv24],
    ids=["nv12", "nv16", "nv24"],
)
def test_dma_semiplanar_buffer_protocol_strided(fmt):
    """A semi-planar DMA tensor exposes its Y+UV plane at the physical pitch."""
    w, h = 595, 438
    t = _dma_image_or_skip(w, h, fmt)
    _skip_unless_pitch_is_padded_and_aligned(t, w)

    with t.map() as m:
        mv = memoryview(m)
        assert mv.ndim == 2 and mv.shape[1] == w
        assert mv.strides[0] == t.row_stride
        rows = mv.shape[0]
        for r in range(rows):
            mv[r, 0] = r % 256  # per-row marker

    with t.map() as m2:
        a2 = np.asarray(memoryview(m2))
        assert all(int(a2[r, 0]) == r % 256 for r in range(a2.shape[0])), (
            "per-row zero-copy round-trip failed on padded semi-planar buffer"
        )


@pytest.mark.parametrize(
    "fmt",
    [PixelFormat.Nv12, PixelFormat.Nv16, PixelFormat.Nv24],
    ids=["nv12", "nv16", "nv24"],
)
def test_dma_semiplanar_from_numpy_writes_every_combined_row(fmt):
    """``from_numpy`` writes every row of a padded semi-planar tensor at its pitch."""
    w, h = 595, 438
    t = _dma_image_or_skip(w, h, fmt)
    _skip_unless_pitch_is_padded_and_aligned(t, w)

    rows = t.shape[0]
    ref = (np.arange(rows * w, dtype=np.uint8) % 251).reshape(rows, w)
    t.from_numpy(ref)

    with t.map() as m:
        assert np.array_equal(np.asarray(memoryview(m)), ref), (
            "semi-planar write sheared"
        )


def test_dma_planar_buffer_protocol_pads_rows_and_planes():
    """A padded planar DMA tensor pads rows and spaces planes ``pitch * H`` apart."""
    w, h = 595, 438
    t = _dma_image_or_skip(w, h, PixelFormat.PlanarRgb)
    _skip_unless_pitch_is_padded_and_aligned(t, w)

    ref = (np.arange(3 * h * w, dtype=np.uint8) % 251).reshape(3, h, w)
    t.from_numpy(ref)

    with t.map() as m:
        mv = memoryview(m)
        assert mv.shape == (3, h, w)
        assert mv.strides == (t.row_stride * h, t.row_stride, 1)
        assert np.array_equal(np.asarray(mv), ref), (
            "planar buffer-protocol read sheared"
        )


def test_dma_one_row_planar_buffer_keeps_its_pitch():
    """A one-row planar image keeps its planes one padded row apart."""
    t = _dma_image_or_skip(2, 1, PixelFormat.PlanarRgb)
    if t.row_stride == 2:
        pytest.skip("this heap returned a tight pitch; nothing to pad")
    with t.map() as m:
        mv = memoryview(m)
        assert mv.shape == (3, 1, 2)
        assert mv.strides == (t.row_stride, t.row_stride, 1)


def test_mem_tight_buffer_protocol_contiguous():
    """A tight image tensor exposes a C-contiguous buffer."""
    w, h, c = 600, 400, 3
    t = Tensor.image(
        w, h, format=PixelFormat.Rgb, mem=TensorMemory.MEM, access="readwrite"
    )
    assert t.row_stride == w * c, "Mem image must be tightly packed"
    ref = (np.arange(h * w * c, dtype=np.uint8) % 251).reshape(h, w, c)
    t.from_numpy(ref)
    with t.map() as m:
        mv = memoryview(m)
        assert mv.shape == (h, w, c)
        assert mv.strides == (w * c, c, 1), "tight buffer must be C-contiguous"
        assert np.array_equal(np.asarray(mv), ref)


def test_enum_cmp():
    dst = Tensor.image(640, 640, format=PixelFormat.Rgba, access="readwrite")
    assert dst.format == PixelFormat.Rgba


@pytest.mark.skipif(
    sys.platform == "darwin",
    reason="macOS DMA tensors are IOSurface-backed, which share cross-process via "
    "surface_id() (a Mach port), not a POSIX fd — `tensor.fd` raises NotImplemented. "
    "The IOSurface cross-process passing test is tracked as future work.",
)
def test_from_fd_dma():
    try:
        tensor = Tensor([100, 100, 3], dtype="uint8", mem=TensorMemory.DMABUF)
    except (AttributeError, RuntimeError):
        skip_dma("DMA memory not supported on this platform")

    tensor = Tensor([720, 1280, 4], dtype="uint8", mem=TensorMemory.DMABUF)
    with tensor.map() as m:
        np.asarray(m.numpy()).fill(233)

    fd = tensor.fd
    try:
        img = Tensor.from_fd(fd, [720, 1280, 4], dtype="uint8")
        with img.map() as m:
            data = np.asarray(m.numpy())
            assert (data == 233).all()
    finally:
        os.close(fd)


def test_from_fd_shm():
    try:
        tensor = Tensor([100, 100, 3], dtype="uint8", mem=TensorMemory.SHM)
    except (AttributeError, RuntimeError):
        pytest.skip("SHM memory not supported on this platform")

    tensor = Tensor([720, 1280, 4], dtype="uint8", mem=TensorMemory.SHM)
    with tensor.map() as m:
        np.asarray(m.numpy()).fill(233)

    fd = tensor.fd
    try:
        img = Tensor.from_fd(fd, [720, 1280, 4], dtype="uint8")
        with img.map() as m:
            data = np.asarray(m.numpy())
            assert (data == 233).all()
    finally:
        os.close(fd)


def test_create_image():
    """Test ImageProcessor.create_image() returns a valid, mappable image."""
    converter = ImageProcessor()
    img = converter.create_image(320, 240, PixelFormat.Rgba, access="readwrite")
    assert img.width == 320
    assert img.height == 240
    assert img.format == PixelFormat.Rgba

    with img.map() as m:
        data = np.asarray(m.numpy())
        assert data.size == 320 * 240 * 4


def _cpu_only_processor(monkeypatch):
    monkeypatch.setenv("EDGEFIRST_DISABLE_GL", "1")
    monkeypatch.setenv("EDGEFIRST_DISABLE_G2D", "1")
    monkeypatch.delenv("EDGEFIRST_FORCE_BACKEND", raising=False)
    monkeypatch.delenv("EDGEFIRST_DISABLE_CPU", raising=False)
    return ImageProcessor()


@pytest.mark.parametrize("access", ["read", "none", None])
def test_cpu_convert_into_create_image(monkeypatch, access):
    """The CPU backend fills a ``create_image`` destination of any requested access."""
    converter = _cpu_only_processor(monkeypatch)
    w, h = _image_size("testdata/zidane.jpg")
    src = Tensor.image(w, h, format=PixelFormat.Nv12)
    decode_file_into(src, "testdata/zidane.jpg")
    kwargs = {} if access is None else {"access": access}
    dst = converter.create_image(320, 240, PixelFormat.Rgb, **kwargs)
    converter.convert(src, dst)
    if access == "none":
        assert dst.declared_cpu_access == "write"
        with pytest.raises(PermissionError, match="CPU access denied"):
            dst.map("read")
    else:
        with dst.map("read") as m:
            assert np.asarray(m.numpy()).any()


def _dma_or_skip(cls, *args, **kwargs):
    """``cls.image`` on DMA-BUF, skipping with a warning on hosts without a heap."""
    try:
        return cls.image(*args, mem=TensorMemory.DMABUF, **kwargs)
    except Exception as exc:
        if not dma_unavailable(exc):
            raise
        reason = f"no usable DMA heap: {exc}"
        print(f"WARNING: SKIPPED: {reason}", file=sys.stderr)
        skip_dma(reason)


def test_gpu_declined_convert_raises_permission_error():
    """A destination without CPU access that the GPU declines raises PermissionError."""
    converter = ImageProcessor()
    dst = converter.create_image(64, 48, PixelFormat.Rgba, access="none")
    skip_unless_gpu_backed(dst)
    if dst.declared_cpu_access != "none":
        reason = f"{dst.memory} destinations always declare CPU access"
        print(f"WARNING: SKIPPED: {reason}", file=sys.stderr)
        pytest.skip(reason)
    src = Tensor.image(64, 48, PixelFormat.Vyuy, mem=TensorMemory.MEM)
    with pytest.raises(PermissionError, match="CPU access denied") as info:
        converter.convert(src, dst)
    assert "opengl" in str(info.value)


def test_decode_image_method_that_needs_a_cpu_write_raises_on_a_read_tensor():
    """``Tensor.decode_image`` into a read-only tensor raises PermissionError
    whenever the decode writes it on the CPU: always into host memory, and
    into a DMA tensor for an image whose size no hardware decoder writes in
    place (500x375, not on a 16-pixel macroblock grid)."""
    from edgefirst.codec import Tensor as CodecTensor

    with open("testdata/zidane.jpg", "rb") as fh:
        data = fh.read()
    w, h = _image_size("testdata/zidane.jpg")
    host = CodecTensor.image(
        w, h, PixelFormat.Nv12, mem=TensorMemory.MEM, access="read"
    )
    with pytest.raises(PermissionError, match="CPU access denied"):
        host.decode_image(data)
    with open("testdata/coco_420_odd.jpg", "rb") as fh:
        odd = fh.read()
    w, h = _image_size("testdata/coco_420_odd.jpg")
    dma = _dma_or_skip(CodecTensor, w, h, PixelFormat.Nv12, access="read")
    with pytest.raises(PermissionError, match="CPU access denied"):
        dma.decode_image(odd)


def test_a_hardware_jpeg_decoder_writes_a_read_dma_tensor_in_place():
    """Under ``Engine.Auto`` a V4L2 JPEG decoder writes a read-only DMA tensor
    by DMA: an aligned image decodes without a CPU access."""
    from edgefirst import codec
    from edgefirst.codec import Tensor as CodecTensor

    if not codec.is_v4l2_available():
        reason = "no V4L2 JPEG decoder here"
        print(f"WARNING: SKIPPED: {reason}", file=sys.stderr)
        pytest.skip(reason)
    with open("testdata/zidane.jpg", "rb") as fh:
        data = fh.read()
    w, h = _image_size("testdata/zidane.jpg")
    dma = _dma_or_skip(CodecTensor, w, h, PixelFormat.Nv12, access="read")
    info = dma.decode_image(data)
    assert (info.width, info.height) == (w, h)
    with dma.map("read") as m:
        spread = float(np.asarray(m).reshape(-1)[: w * h].std())
    assert spread > 10, "the decoded picture is flat"


@pytest.mark.parametrize("mem", ["mem", "dma"])
def test_imported_alias_keeps_declaration(monkeypatch, mem):
    """A tensor imported from another EdgeFirst package keeps its access declaration."""
    from edgefirst.tensor import PixelFormat as CorePixelFormat
    from edgefirst.tensor import Tensor as CoreTensor
    from edgefirst.tensor import TensorMemory as CoreMemory

    converter = _cpu_only_processor(monkeypatch)
    src = Tensor.image(64, 48, PixelFormat.Rgba, mem=TensorMemory.MEM)
    memory = CoreMemory.MEM if mem == "mem" else CoreMemory.DMABUF
    try:
        dst = CoreTensor.image(64, 48, CorePixelFormat.Rgb, mem=memory, access="read")
    except Exception as exc:
        if mem != "dma" or not dma_unavailable(exc):
            raise
        reason = f"no usable DMA heap: {exc}"
        print(f"WARNING: SKIPPED: {reason}", file=sys.stderr)
        skip_dma(reason)
    assert dst.declared_cpu_access == "read"
    with pytest.raises(PermissionError, match="CPU access denied"):
        converter.convert(src, dst)
    assert dst.declared_cpu_access == "read"


@pytest.mark.parametrize("origin", [(0, 0), (16, 8)])
@pytest.mark.parametrize("mem", ["mem", "dma"])
@pytest.mark.parametrize("package", ["image", "tensor"])
def test_cpu_convert_refuses_a_write_only_view(monkeypatch, package, mem, origin):
    """The CPU convert refuses a write-only view from this package or another."""
    from edgefirst.tensor import PixelFormat as CorePixelFormat
    from edgefirst.tensor import Region
    from edgefirst.tensor import Tensor as CoreTensor

    converter = _cpu_only_processor(monkeypatch)
    cls, fmt = (
        (Tensor, PixelFormat.Rgb)
        if package == "image"
        else (CoreTensor, CorePixelFormat.Rgb)
    )
    if mem == "mem":
        parent = cls.image(64, 48, fmt, mem=TensorMemory.MEM, access="write")
    else:
        parent = _dma_or_skip(cls, 64, 48, fmt, access="write")
    src = Tensor.image(64, 48, PixelFormat.Rgba, mem=TensorMemory.MEM)
    view = parent.view(Region(*origin, 32, 24))
    with pytest.raises(PermissionError, match="CPU access denied"):
        converter.convert(src, view)
    converter.convert(src, parent)


@pytest.mark.parametrize("dtype", [np.uint8, np.float32])
def test_normalize_to_numpy_reads_a_read_only_tensor(dtype):
    """normalize_to_numpy reads a tensor declared read-only."""
    src = Tensor.image(8, 4, PixelFormat.Rgb, mem=TensorMemory.MEM, access="read")
    out = np.full((4, 8, 3), 7, dtype=dtype)
    src.normalize_to_numpy(out)
    assert (out != 7).all()


def test_writes_into_a_tensor_declared_without_write_raise(monkeypatch):
    """The CPU convert and the codec refuse a tensor declared without write access."""
    converter = _cpu_only_processor(monkeypatch)
    w, h = _image_size("testdata/zidane.jpg")
    src = Tensor.image(w, h, format=PixelFormat.Nv12)
    decode_file_into(src, "testdata/zidane.jpg")
    dst = Tensor.image(320, 240, PixelFormat.Rgb, mem=TensorMemory.MEM, access="read")
    with pytest.raises(PermissionError, match="CPU access denied"):
        converter.convert(src, dst)

    with open("testdata/zidane.jpg", "rb") as fh:
        data = fh.read()
    target = Tensor.image(w, h, PixelFormat.Nv12, mem=TensorMemory.MEM, access="read")
    with pytest.raises(PermissionError, match="CPU access denied"):
        decode_into(target, data)
    write_only = Tensor.image(
        w, h, PixelFormat.Nv12, mem=TensorMemory.MEM, access="write"
    )
    decode_into(write_only, data)


def test_create_image_formats():
    """Test create_image with different pixel formats."""
    converter = ImageProcessor()
    for fmt, channels in [
        (PixelFormat.Rgb, 3),
        (PixelFormat.Rgba, 4),
        (PixelFormat.Grey, 1),
    ]:
        img = converter.create_image(160, 120, fmt, access="readwrite")
        assert img.width == 160
        assert img.height == 120
        assert img.format == fmt
        with img.map() as m:
            data = np.asarray(m.numpy())
            assert data.size == 160 * 120 * channels


def test_create_image_dtype_i8():
    """Test create_image with dtype='int8'."""
    converter = ImageProcessor()
    img = converter.create_image(
        320, 240, PixelFormat.Rgb, dtype="int8", access="readwrite"
    )
    assert img.width == 320
    assert img.height == 240
    assert img.format == PixelFormat.Rgb
    assert img.dtype == "int8"

    w, h = _image_size("testdata/zidane.jpg")
    src = Tensor.image(w, h, format=PixelFormat.Nv12, access="readwrite")
    decode_file_into(src, "testdata/zidane.jpg")
    dst = converter.create_image(
        640, 640, PixelFormat.Rgb, dtype="int8", access="readwrite"
    )
    converter.convert(src, dst)


def test_create_image_convert():
    """Test that images from create_image() work with convert()."""
    converter = ImageProcessor()

    w, h = _image_size("testdata/zidane.jpg")
    src = Tensor.image(w, h, format=PixelFormat.Nv12, access="readwrite")
    decode_file_into(src, "testdata/zidane.jpg")

    dst = converter.create_image(640, 640, PixelFormat.Rgba, access="readwrite")

    converter.convert(src, dst)

    n = np.zeros((640, 640, 3), dtype=np.uint8)
    dst.normalize_to_numpy(n)
    assert n.any(), "Destination image is all zeros after convert"

    dst_mem = Tensor.image(640, 640, PixelFormat.Rgba, access="readwrite")
    converter.convert(src, dst_mem)
    n_mem = np.zeros((640, 640, 3), dtype=np.uint8)
    dst_mem.normalize_to_numpy(n_mem)

    assert calculate_similarity_rms_u8(n, n_mem) > 0.95


def test_create_image_roundtrip():
    """Test create_image as both src and dst for convert()."""
    converter = ImageProcessor()

    src = converter.create_image(640, 480, PixelFormat.Rgba, access="readwrite")
    with src.map() as m:
        arr = np.asarray(m.numpy())
        pattern = np.tile(np.arange(256, dtype=np.uint8), arr.size // 256 + 1)[
            : arr.size
        ]
        arr.flat[:] = pattern

    dst = converter.create_image(320, 240, PixelFormat.Rgba, access="readwrite")

    converter.convert(src, dst)

    with dst.map() as m:
        result = np.asarray(m.numpy())
        assert result.any(), "Destination is all zeros after roundtrip convert"


@pytest.mark.skipif(sys.platform != "linux", reason="import_image is Linux-only")
def test_import_image_negative_fd():
    """import_image raises RuntimeError for invalid fd."""
    processor = ImageProcessor()
    with pytest.raises(RuntimeError):
        processor.import_image(fd=-1, width=640, height=640, format=PixelFormat.Rgb)


@pytest.mark.skipif(sys.platform != "linux", reason="import_image is Linux-only")
def test_import_image_negative_chroma_fd():
    """import_image raises RuntimeError for invalid chroma fd."""
    processor = ImageProcessor()
    with pytest.raises(RuntimeError):
        processor.import_image(
            fd=0, width=640, height=640, format=PixelFormat.Nv12, chroma_fd=-1
        )


@pytest.mark.skipif(sys.platform != "linux", reason="import_image is Linux-only")
def test_import_image_zero_dimensions():
    """import_image raises RuntimeError for zero width or height."""
    processor = ImageProcessor()
    with pytest.raises(RuntimeError):
        processor.import_image(fd=0, width=0, height=640, format=PixelFormat.Rgb)
    with pytest.raises(RuntimeError):
        processor.import_image(fd=0, width=640, height=0, format=PixelFormat.Rgb)


@pytest.mark.skipif(sys.platform != "linux", reason="import_image is Linux-only")
def test_import_image_chroma_with_packed_format():
    """import_image raises RuntimeError when chroma_fd given for non-semi-planar format."""
    processor = ImageProcessor()
    with pytest.raises(RuntimeError):
        processor.import_image(
            fd=0, width=640, height=480, format=PixelFormat.Rgba, chroma_fd=1
        )


@pytest.mark.skipif(sys.platform != "linux", reason="import_image is Linux-only")
def test_import_image_dma_success():
    """import_image succeeds with a real DMA-BUF fd (skipped if DMA unavailable)."""
    try:
        src = Tensor([480, 640, 4], dtype="uint8", mem=TensorMemory.DMABUF)
    except RuntimeError:
        skip_dma("DMA allocation not available on this platform")
    fd = src.fd
    processor = ImageProcessor()
    imported = processor.import_image(
        fd=fd, width=640, height=480, format=PixelFormat.Rgba
    )
    assert imported.width == 640
    assert imported.height == 480


# ─── Mask rendering tests ────────────────────────────────────────


def test_draw_decoded_masks_empty():
    """draw_decoded_masks with zero detections should produce background only."""
    w, h = _image_size("testdata/giraffe.jpg")
    converter = ImageProcessor()
    bg_native = Tensor.image(w, h, format=PixelFormat.Nv12, access="readwrite")
    decode_file_into(bg_native, "testdata/giraffe.jpg")
    bg = Tensor.image(w, h, format=PixelFormat.Rgba, access="readwrite")
    converter.convert(bg_native, bg)
    dst = converter.create_image(w, h, PixelFormat.Rgba, access="readwrite")
    converter.draw_decoded_masks(
        dst,
        bbox=np.zeros((0, 4), dtype=np.float32),
        scores=np.zeros((0,), dtype=np.float32),
        classes=np.zeros((0,), dtype=np.uintp),
        background=bg,
    )
    # Compare against the converted RGBA background.
    expected = np.zeros((h, w, 4), dtype=np.uint8)
    bg.normalize_to_numpy(expected)
    with dst.map() as m:
        img = np.asarray(m.numpy())
        assert calculate_similarity_rms_u8(img, expected) > 0.99


def test_draw_decoded_masks_multiple_boxes():
    """draw_decoded_masks with multiple detections (boxes only, no seg)."""
    w, h = _image_size("testdata/giraffe.jpg")
    converter = ImageProcessor()
    src = Tensor.image(w, h, format=PixelFormat.Nv12, access="readwrite")
    decode_file_into(src, "testdata/giraffe.jpg")
    dst = converter.create_image(w, h, PixelFormat.Rgba, access="readwrite")
    converter.convert(src, dst)
    converter.draw_decoded_masks(
        dst,
        bbox=np.array(
            [
                [0.1, 0.1, 0.3, 0.3],
                [0.5, 0.5, 0.8, 0.8],
                [0.2, 0.6, 0.4, 0.9],
            ],
            dtype=np.float32,
        ),
        scores=np.array([0.9, 0.8, 0.7], dtype=np.float32),
        classes=np.array([0, 1, 2], dtype=np.uintp),
    )
    original = load_image("testdata/giraffe.jpg", "RGBA")
    with dst.map() as m:
        img = np.asarray(m.numpy())
        assert calculate_similarity_rms_u8(img, original) < 0.999


def test_draw_decoded_masks_instance_color_mode():
    """draw_decoded_masks with ColorMode.Instance assigns per-detection colors."""
    w, h = _image_size("testdata/giraffe.jpg")
    converter = ImageProcessor()
    src = Tensor.image(w, h, format=PixelFormat.Nv12, access="readwrite")
    decode_file_into(src, "testdata/giraffe.jpg")
    dst = converter.create_image(w, h, PixelFormat.Rgba, access="readwrite")
    converter.convert(src, dst)
    # Two detections with same class but Instance coloring
    converter.draw_decoded_masks(
        dst,
        bbox=np.array(
            [[0.1, 0.1, 0.4, 0.4], [0.5, 0.5, 0.9, 0.9]],
            dtype=np.float32,
        ),
        scores=np.array([0.9, 0.8], dtype=np.float32),
        classes=np.array([0, 0], dtype=np.uintp),
        color_mode=ColorMode.Instance,
    )
    original = load_image("testdata/giraffe.jpg", "RGBA")
    with dst.map() as m:
        img = np.asarray(m.numpy())
        assert calculate_similarity_rms_u8(img, original) < 0.999


def test_draw_decoded_masks_with_opacity():
    """draw_decoded_masks with reduced opacity makes overlays more transparent."""
    w, h = _image_size("testdata/giraffe.jpg")
    converter = ImageProcessor()
    src = Tensor.image(w, h, format=PixelFormat.Nv12, access="readwrite")
    decode_file_into(src, "testdata/giraffe.jpg")
    dst_full = converter.create_image(w, h, PixelFormat.Rgba, access="readwrite")
    converter.convert(src, dst_full)
    dst_half = converter.create_image(w, h, PixelFormat.Rgba, access="readwrite")
    converter.convert(src, dst_half)
    bbox = np.array([[0.1, 0.1, 0.9, 0.9]], dtype=np.float32)
    scores = np.array([0.9], dtype=np.float32)
    classes = np.array([0], dtype=np.uintp)

    converter.draw_decoded_masks(
        dst_full, bbox=bbox, scores=scores, classes=classes, opacity=1.0
    )
    converter.draw_decoded_masks(
        dst_half, bbox=bbox, scores=scores, classes=classes, opacity=0.3
    )

    original = load_image("testdata/giraffe.jpg", "RGBA")
    with dst_full.map() as m1, dst_half.map() as m2:
        img_full = np.array(m1.numpy()).reshape((dst_full.height, dst_full.width, 4))
        img_half = np.array(m2.numpy()).reshape((dst_half.height, dst_half.width, 4))
        sim_full = calculate_similarity_rms_u8(img_full, original)
        sim_half = calculate_similarity_rms_u8(img_half, original)
        assert sim_half >= sim_full


def test_draw_masks_fused():
    """draw_masks fused decode+render path with synthetic detection model output."""
    config = """
decoder_version: yolo26
outputs:
 - type: detection
   decoder: ultralytics
   shape: [1, 10, 6]
   dshape:
    - [batch, 1]
    - [num_boxes, 10]
    - [num_features, 6]
   normalized: true
"""
    decoder = Decoder.new_from_yaml_str(config, score_threshold=0.1, iou_threshold=0.7)

    data = np.zeros((1, 10, 6), dtype=np.float32)
    data[0, 0, :] = [0.1, 0.1, 0.3, 0.3, 0.9, 2.0]  # box + score + class
    model_output_tensor = Tensor([1, 10, 6], dtype="float32")
    with model_output_tensor.map() as m:
        buf = np.asarray(m.numpy())
        buf[:] = data

    w, h = _image_size("testdata/giraffe.jpg")
    converter = ImageProcessor()
    src = Tensor.image(w, h, format=PixelFormat.Nv12, access="readwrite")
    decode_file_into(src, "testdata/giraffe.jpg")
    dst = converter.create_image(w, h, PixelFormat.Rgba, access="readwrite")
    converter.convert(src, dst)
    result = decoder.draw_onto(converter, [model_output_tensor], dst)

    assert len(result) == 3
    boxes, scores, classes = result
    assert boxes.shape[1] == 4
    assert len(scores) == len(boxes)
    assert len(classes) == len(boxes)


def test_decode_image_from_bytes():
    """Test decode_image with raw bytes into oversized tensor (strided)."""
    with open("testdata/zidane.jpg", "rb") as fh:
        data = fh.read()
    tensor = Tensor.image(
        MAX_SRC_W, MAX_SRC_H, format=PixelFormat.Nv12, access="readwrite"
    )
    info: ImageInfo = decode_into(tensor, data)
    assert info.width == 1280
    assert info.height == 720
    assert info.format == PixelFormat.Nv12
    assert info.row_stride == MAX_SRC_W
    assert info.row_stride >= info.width
    assert info.row_stride % 64 == 0


def test_decode_image_reuse():
    """Decoding different images into one oversized tensor reconfigures its format."""
    tensor = Tensor.image(
        MAX_SRC_W, MAX_SRC_H, format=PixelFormat.Nv12, access="readwrite"
    )

    info1 = decode_file_into(tensor, "testdata/zidane.jpg")
    assert info1.width == 1280
    assert info1.height == 720
    assert info1.format == PixelFormat.Nv12
    assert tensor.format == PixelFormat.Nv12
    assert (tensor.width, tensor.height) == (1280, 720)

    info2 = decode_file_into(tensor, "testdata/grey.jpg")
    assert info2.width == 1024
    assert info2.height == 681
    assert info2.format == PixelFormat.Grey
    assert tensor.format == PixelFormat.Grey
    assert (tensor.width, tensor.height) == (1024, 681)


def test_decode_image_native_format():
    """Test decode_image emits the source's native format (NV12 for color JPEG)."""
    tensor = Tensor.image(
        MAX_SRC_W, MAX_SRC_H, format=PixelFormat.Nv12, access="readwrite"
    )
    info = decode_file_into(tensor, "testdata/zidane.jpg")
    assert info.width == 1280
    assert info.height == 720
    assert info.format == PixelFormat.Nv12


def test_decode_image_grey():
    """Test decoding into Grey format with oversized tensor."""
    tensor = Tensor.image(
        MAX_SRC_W, MAX_SRC_H, format=PixelFormat.Grey, access="readwrite"
    )
    info = decode_file_into(tensor, "testdata/grey.jpg")
    assert info.width == 1024
    assert info.height == 681
    assert info.format == PixelFormat.Grey


def test_decode_image_pipeline():
    """Test the full recommended pipeline: create_image -> decode -> convert."""
    converter = ImageProcessor()

    w, h = _image_size("testdata/zidane.jpg")
    src = converter.create_image(w, h, PixelFormat.Nv12, access="readwrite")
    decode_file_into(src, "testdata/zidane.jpg")

    dst = converter.create_image(640, 640, PixelFormat.Rgba, access="readwrite")
    converter.convert(src, dst)

    n = np.zeros((640, 640, 3), dtype=np.uint8)
    dst.normalize_to_numpy(n)
    assert n.any(), "Destination image is all zeros after pipeline"


def test_decode_image_exact_size():
    """Allocate tensor at exact decoded image size by reading the JPEG header first."""
    with Image.open("testdata/zidane.jpg") as im:
        w, h = im.size

    tensor = Tensor.image(w, h, format=PixelFormat.Nv12, access="readwrite")
    info = decode_file_into(tensor, "testdata/zidane.jpg")
    assert info.width == w
    assert info.height == h
    assert info.format == PixelFormat.Nv12
    assert info.row_stride == w


def test_decode_image_exact_size_png():
    """Exact-size decode for PNG format (native RGB for this fixture)."""
    with Image.open("testdata/zidane.png") as im:
        w, h = im.size

    tensor = Tensor.image(w, h, format=PixelFormat.Rgb, access="readwrite")
    info = decode_file_into(tensor, "testdata/zidane.png")
    assert info.width == w
    assert info.height == h
    assert info.format == PixelFormat.Rgb
    assert info.row_stride == w * 3


def test_decode_image_tensor_too_small_width():
    """Decoding into a tensor narrower than the image should raise an error."""
    tensor = Tensor.image(320, 720, format=PixelFormat.Nv12, access="readwrite")
    with pytest.raises(RuntimeError, match="[Cc]apacity|[Ss]ize|[Ww]idth|[Ss]tride"):
        decode_file_into(tensor, "testdata/zidane.jpg")


def test_decode_image_tensor_too_small_height():
    """Decoding into a tensor shorter than the image should raise an error."""
    tensor = Tensor.image(1280, 360, format=PixelFormat.Nv12, access="readwrite")
    with pytest.raises(RuntimeError, match="[Cc]apacity|[Ss]ize|[Hh]eight|[Rr]ow"):
        decode_file_into(tensor, "testdata/zidane.jpg")


def test_decode_image_tensor_too_small_both():
    """Decoding into a tensor smaller in both dimensions should raise an error."""
    tensor = Tensor.image(100, 100, format=PixelFormat.Nv12, access="readwrite")
    with pytest.raises(RuntimeError):
        decode_file_into(tensor, "testdata/zidane.jpg")


def test_decode_image_strided_pixel_correctness():
    """Strided decode produces the same luma pixels as an exact-size decode."""
    w, h = _image_size("testdata/zidane.jpg")
    exact = Tensor.image(w, h, format=PixelFormat.Nv12, access="readwrite")
    decode_file_into(exact, "testdata/zidane.jpg")
    with exact.map() as m:
        arr = np.asarray(memoryview(m))
        exact_y = np.array(arr[:h, :w], copy=True)

    big = Tensor.image(
        MAX_SRC_W, MAX_SRC_H, format=PixelFormat.Nv12, access="readwrite"
    )
    info = decode_file_into(big, "testdata/zidane.jpg")

    # Extract the decoded luma sub-region using the reported row stride.
    with big.map() as m:
        arr = np.asarray(memoryview(m))
        assert arr.strides[0] == info.row_stride
        big_y = np.array(arr[:h, :w], copy=True)

    assert np.array_equal(exact_y, big_y), (
        "Strided decode pixels differ from exact-size decode"
    )


@pytest.mark.parametrize(
    "width,height",
    [
        (105, 124),
        (100, 124),
        (160, 80),
        (200, 64),
    ],
)
def test_from_numpy_grey_unaligned_width(width, height):
    """`from_numpy` accepts a width not aligned to the GPU pitch."""
    converter = ImageProcessor()
    img = converter.create_image(width, height, PixelFormat.Grey, access="readwrite")

    assert img.size == width * height, (
        f"Image.size ({img.size}) must remain logical (w*h={width * height})"
    )

    rs = img.row_stride
    assert rs is not None and rs >= width, f"row_stride {rs} must be >= width {width}"

    src = np.arange(width * height, dtype=np.uint8).reshape(height, width)
    img.from_numpy(src)

    with img.map() as m:
        raw = np.asarray(m.numpy())
        assert raw.size == width * height, (
            f"mapped view size {raw.size} != Image.size {width * height}"
        )


def test_convert_honors_tagged_colorimetry():
    """convert() honors the source tensor's tagged colorimetry."""
    w, h = 1280, 720
    nv12 = np.fromfile("testdata/zidane.nv12", dtype=np.uint8).reshape(h * 3 // 2, w)
    converter = ImageProcessor()

    def convert_as(encoding, color_range):
        src = Tensor.image(w, h, format=PixelFormat.Nv12, access="readwrite")
        src.from_numpy(nv12)
        src.colorimetry = Colorimetry(encoding=encoding, range=color_range)
        dst = converter.create_image(w, h, PixelFormat.Rgb, access="readwrite")
        converter.convert(src, dst)
        out = np.zeros((h, w, 3), dtype=np.uint8)
        dst.normalize_to_numpy(out)
        return out

    bt601_full = convert_as(ColorEncoding.Bt601, ColorRange.Full)
    bt709_limited = convert_as(ColorEncoding.Bt709, ColorRange.Limited)

    assert not np.array_equal(bt601_full, bt709_limited), (
        "BT.601-full and BT.709-limited decoded to identical pixels — "
        "convert() ignored the tagged colorimetry"
    )
    mean_diff = np.abs(bt601_full.astype(int) - bt709_limited.astype(int)).mean()
    assert mean_diff > 1.0, f"mean pixel delta {mean_diff:.3f} too small to be real"


@pytest.mark.parametrize(
    "method", ["convert", "convert_deferred", "convert_with_fence"]
)
def test_convert_tags_an_untagged_yuv_destination(method):
    """Every convert entry point tags an untagged YUV destination with the
    colorimetry its height implies, and keeps a tagged destination's tag."""
    p = ImageProcessor()
    src = Tensor.image(640, 360, PixelFormat.Rgba, None, "readwrite")
    bt709_full = Colorimetry(encoding=ColorEncoding.Bt709, range=ColorRange.Full)

    def run(dst):
        fence = getattr(p, method)(src, dst)
        if method == "convert_with_fence" and fence is not None:
            if sys.platform == "win32":
                import ctypes

                ctypes.windll.kernel32.CloseHandle(ctypes.c_void_p(fence))
            else:
                os.close(fence)
        if method == "convert_deferred":
            p.flush()

    for fmt in (PixelFormat.Nv12, PixelFormat.Nv16, PixelFormat.Yuyv):
        dst = Tensor.image(640, 360, fmt, None, "readwrite")
        assert dst.colorimetry is None
        run(dst)
        tag = dst.colorimetry
        assert tag is not None, f"{fmt}: the destination was left untagged"
        assert tag.encoding == ColorEncoding.Bt601, f"{fmt}: {tag}"
        assert tag.range == ColorRange.Limited, f"{fmt}: {tag}"

        tagged = Tensor.image(640, 360, fmt, None, "readwrite")
        tagged.colorimetry = bt709_full
        run(tagged)
        assert tagged.colorimetry.encoding == ColorEncoding.Bt709, f"{fmt}"
        assert tagged.colorimetry.range == ColorRange.Full, f"{fmt}"


@pytest.mark.gpu
def test_convert_with_fence_returns_a_waitable_or_none():
    """convert_with_fence() returns a waitable completion handle, or None."""
    p = ImageProcessor()
    src = Tensor.image(64, 48, PixelFormat.Rgba, None, "readwrite")
    dst = p.create_image(64, 48, PixelFormat.Rgb, "uint8", "read")
    skip_unless_gpu_backed(dst)

    fence = p.convert_with_fence(src, dst, Rotation.Rotate0, Flip.NoFlip)
    if fence is None:
        return  # no native fence on this platform

    if sys.platform == "win32":
        import ctypes

        try:
            waited = ctypes.windll.kernel32.WaitForSingleObject(
                ctypes.c_void_p(fence), 5000
            )
        finally:
            ctypes.windll.kernel32.CloseHandle(ctypes.c_void_p(fence))
        assert waited == 0, "the fence was not set within 5 seconds"
    else:
        os.close(fence)


@pytest.mark.gpu
def test_convert_publishes_its_completion_on_the_destination():
    """A convert records a fence value that the caller's destination tensor reports."""
    if sys.platform != "win32":
        pytest.skip("only D3D11 textures record completions")
    import ctypes

    p = ImageProcessor()
    src = Tensor.image(64, 48, PixelFormat.Rgba, None, "readwrite")
    dst = p.create_image(64, 48, PixelFormat.Rgb, "uint8", "read")
    skip_unless_gpu_backed(dst)
    assert dst.gpu_completion() is None, "nothing has written this texture yet"

    p.convert(src, dst, Rotation.Rotate0, Flip.NoFlip)

    completion = dst.gpu_completion()
    assert completion is not None, "the convert published no completion"
    fence, value = completion
    try:
        assert value > 0
    finally:
        ctypes.windll.kernel32.CloseHandle(ctypes.c_void_p(fence))


@pytest.mark.skipif(sys.platform != "linux", reason="CMA DMA heap is Linux-only")
def test_create_image_contiguous_is_cma_dma_or_an_error():
    proc = ImageProcessor()
    # Without a CMA heap, or with its pool full, the request fails naming it.
    try:
        t = proc.create_image(64, 64, PixelFormat.Rgba, contiguous=True)
    except RuntimeError as exc:
        assert "linux,cma" in str(exc)
        return
    assert os.path.exists("/dev/dma_heap/linux,cma")
    assert t.memory == TensorMemory.DMABUF
    assert t.contiguity == "contiguous"
