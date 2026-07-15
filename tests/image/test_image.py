# SPDX-FileCopyrightText: Copyright 2025 Au-Zone Technologies
# SPDX-License-Identifier: Apache-2.0

from edgefirst_hal import (
    ColorMode,
    Colorimetry,
    ColorEncoding,
    ColorRange,
    Decoder,
    ImageInfo,
    ImageProcessor,
    Flip,
    PixelFormat,
    Normalization,
    Rotation,
    Tensor,
    TensorMemory,
)
import numpy as np
from PIL import Image
import math
import os
import sys
import pytest


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


# Maximum source tensor dimensions — oversized to exercise strided decode.
# Used by decode-specific tests where pixel comparison goes through
# normalize_to_numpy at the decoded sub-region size.
MAX_SRC_W, MAX_SRC_H = 1920, 1080


def _image_size(path):
    """Read image dimensions from file header without decoding pixels."""
    with Image.open(path) as im:
        return im.size  # (width, height)


def test_flip(monkeypatch):
    # This test's oracle is PIL's exact JPEG decode, so it opts into
    # colorimetry-exact conversion. The default ColorimetryMode is Fast
    # (issue #106): on Vivante it routes every NV12 colorimetry through the
    # external sampler's fixed conversion (~12x faster), which lands ~0.965
    # against this oracle.
    monkeypatch.setenv("EDGEFIRST_COLORIMETRY", "exact")
    w, h = _image_size("testdata/zidane.jpg")
    # JPEG decodes to its native NV12; convert downstream produces RGBA.
    src = Tensor.image(w, h, format=PixelFormat.Nv12, access="readwrite")
    src.decode_image_file("testdata/zidane.jpg")
    converter = ImageProcessor()
    dst = converter.create_image(1280, 720, PixelFormat.Rgba, access="readwrite")
    converter.convert(src, dst, flip=Flip.Horizontal)

    n = np.zeros((720, 1280, 3), dtype=np.uint8)
    dst.normalize_to_numpy(n)

    expected = load_image("testdata/zidane.jpg", "RGB")
    expected = expected[:, ::-1, :]

    # Decode produces native NV12 tagged JFIF (BT.601 full-range); convert()
    # honors it, matching PIL's direct JPEG RGB decode. Only 4:2:0 chroma
    # downsampling differs, so 0.98 holds.
    assert calculate_similarity_rms_u8(n, expected) > 0.98


def test_grey_load():
    w, h = _image_size("testdata/grey.jpg")
    converter = ImageProcessor()

    # A greyscale JPEG decodes to its native Grey format.
    grey_ = Tensor.image(w, h, format=PixelFormat.Grey, access="readwrite")
    info = grey_.decode_image_file("testdata/grey.jpg")
    assert info.format == PixelFormat.Grey
    grey = np.zeros((h, w, 1), dtype=np.uint8)
    grey_.normalize_to_numpy(grey)

    # RGBA / RGB variants are obtained by converting the native Grey image.
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
    src.decode_image_file("testdata/zidane.jpg")
    converter = ImageProcessor()
    dst = converter.create_image(640, 640, PixelFormat.Rgba, access="readwrite")
    converter.convert(src, dst)
    n = np.zeros((640, 640, 3), dtype=np.int8)
    dst.normalize_to_numpy(n, normalization=Normalization.SIGNED)
    unique_vals = np.unique(n)
    assert (unique_vals == np.array(range(-128, 128))).all()

    dst.normalize_to_numpy(n, normalization=Normalization.SIGNED, zero_point=127)
    unique_vals0 = np.unique(n)
    assert (unique_vals0 == np.array(range(-127, 128))).all()


def test_render():
    w, h = _image_size("testdata/giraffe.jpg")
    converter = ImageProcessor()
    # Decode the JPEG to its native NV12, then convert to an RGBA background.
    bg_native = Tensor.image(w, h, format=PixelFormat.Nv12, access="readwrite")
    bg_native.decode_image_file("testdata/giraffe.jpg")
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
        img = np.array(m.numpy()).reshape((dst.height, dst.width, 4))
        # Threshold 0.95: GPU smoothstep anti-aliasing at mask edges produces
        # small differences across platforms (x86 Mesa vs Vivante GC7000UL),
        # and the background now arrives via an NV12->RGBA conversion that
        # differs slightly from the reference renders' direct-RGBA decode.
        assert (
            calculate_similarity_rms_u8(img, expected_gl) > 0.95
            or calculate_similarity_rms_u8(img, expected_cpu) > 0.95
        )


def test_rgb_resize(monkeypatch):
    # PIL-exact oracle → opt into colorimetry-exact (see test_flip).
    monkeypatch.setenv("EDGEFIRST_COLORIMETRY", "exact")
    w, h = _image_size("testdata/zidane.jpg")
    src = Tensor.image(w, h, format=PixelFormat.Nv12, access="readwrite")
    src.decode_image_file("testdata/zidane.jpg")
    converter = ImageProcessor()
    dst = converter.create_image(640, 640, PixelFormat.Rgba, access="readwrite")
    converter.convert(src, dst)
    with dst.map() as m:
        n = np.array(m.numpy()).reshape((dst.height, dst.width, 4))
        expected = load_image("testdata/zidane.jpg", "RGBA", resize=(640, 640))
        # JFIF (BT.601 full) NV12->RGB matches PIL's JPEG decode; 0.98 holds.
        assert calculate_similarity_rms_u8(n, expected) > 0.98


def test_rgba_to_rgb(monkeypatch):
    # PIL-exact oracle → opt into colorimetry-exact (see test_flip).
    monkeypatch.setenv("EDGEFIRST_COLORIMETRY", "exact")
    w, h = _image_size("testdata/zidane.jpg")
    # Decode the JPEG to native NV12, then convert to RGB (3-channel) output.
    src = Tensor.image(w, h, format=PixelFormat.Nv12, access="readwrite")
    src.decode_image_file("testdata/zidane.jpg")
    converter = ImageProcessor()
    dst = converter.create_image(w, h, PixelFormat.Rgb, access="readwrite")
    # Whole-image convert (the former Rect(0,0,w,h) was a no-op crop; the
    # destination shape is the placement and `source=None` samples the whole src).
    converter.convert(src, dst, Rotation.Rotate0, Flip.NoFlip)

    with dst.map() as m:
        n = np.array(m.numpy()).reshape((dst.height, dst.width, 3))
        expected = load_image("testdata/zidane.jpg", "RGB")
        # JFIF (BT.601 full) NV12->RGB matches PIL's JPEG decode; 0.98 holds.
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
    """The Python surface exposes NV16 (4:2:2) and NV24 (4:4:4): the decoder
    selects the matching native format from the JPEG subsampling, and the
    semi-planar source converts to RGB through the same path as NV12."""
    if not os.path.exists(fixture):
        pytest.skip(f"{fixture} fixture missing")
    w, h = _image_size(fixture)
    src = Tensor.image(w, h, format=fmt, access="readwrite")
    assert src.format == fmt
    info = src.decode_image_file(fixture)
    # The decoder reports (and configures) the native semi-planar format.
    assert info.format == fmt
    assert (info.width, info.height) == (w, h)

    converter = ImageProcessor()
    dst = converter.create_image(w, h, PixelFormat.Rgb, access="readwrite")
    converter.convert(src, dst)
    n = np.zeros((h, w, 3), dtype=np.uint8)
    dst.normalize_to_numpy(n)
    expected = load_image(fixture, "RGB")
    # 0.95: same BT.601-full-vs-PIL tolerance as the NV12 conversions.
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
    """Real-world COCO JPEGs with ODD dimensions, one per native decode output:
    4:2:0->NV12, 4:2:2->NV16, 4:4:4->NV24, greyscale->Grey. Each is decoded to
    its native format and converted to RGB on whichever backend the lane
    provides — CPU on headless x86_64, G2D/GL on i.MX8MP, ANGLE GL on macOS — so
    the odd-stride chroma layout is exercised end-to-end on both CPU and GPU.

    Odd dimensions stress the stride math: an odd width rounds the buffer up to
    even(width) then 64-byte alignment (so chroma columns stay byte-aligned and
    the GPU DMA pitch is valid), while NV12/NV24 chroma row counts use ceil(H/2)
    and 2*H of an odd height.

    The 4:2:2 fixture is the only synthesized one: COCO has no native 4:2:2
    JPEGs, so it is re-encoded from a real COCO photo at its native odd size.
    """
    if not os.path.exists(fixture):
        pytest.skip(f"{fixture} fixture missing")
    w, h = _image_size(fixture)
    assert (w % 2 == 1) or (h % 2 == 1), f"{fixture}: expected an odd dimension"

    src = Tensor.image(w, h, format=fmt, access="readwrite")
    assert src.format == fmt
    info = src.decode_image_file(fixture)
    # The decoder selects + configures the native format and keeps the true
    # (odd) logical dimensions — no even-rounding leaks into the reported size.
    assert info.format == fmt
    assert (info.width, info.height) == (w, h)

    converter = ImageProcessor()
    dst = converter.create_image(w, h, PixelFormat.Rgb, access="readwrite")
    converter.convert(src, dst)
    # normalize_to_numpy honours the destination's (possibly padded) row stride,
    # which a flat reshape would not for odd widths.
    n = np.zeros((h, w, 3), dtype=np.uint8)
    dst.normalize_to_numpy(n)
    expected = load_image(fixture, "RGB")
    # GREY has no YUV colorimetry ambiguity (Grey→RGB = replicate luma), so a
    # tighter tolerance is appropriate. All other formats use 0.95 to cover
    # BT.601-full-vs-PIL divergence across CPU / ANGLE GL / Vivante GL backends.
    threshold = 0.98 if fmt == PixelFormat.Grey else 0.95
    assert calculate_similarity_rms_u8(n, expected) > threshold


# ---------------------------------------------------------------------------
# Strided zero-copy + DMA buffer-protocol tests.
#
# A DMA / GPU image tensor is allocated with a 64-byte-aligned row pitch, so
# whenever `width * channels` is not already 64-aligned the backing buffer is
# ROW-PADDED: `row_stride > width * channels`. `map()` exposes the full padded
# buffer, so any consumer that assumes a tight `width*channels`-per-row layout
# reads every row after the first at the wrong offset (a progressive shear).
# These cells prove both the read (buffer protocol + normalize_to_numpy) and
# write (from_numpy) paths honour the physical pitch, zero-copy onto the DMA-BUF.
#
# On platforms without a DMA heap (headless x86 CI) the allocation falls back /
# is unavailable and the cells skip — the padded pitch only arises on DMA.
# ---------------------------------------------------------------------------


def _dma_image_or_skip(w, h, fmt):
    """Allocate a DMA-backed image tensor, or skip when DMA is unavailable."""
    try:
        t = Tensor.image(w, h, format=fmt, mem=TensorMemory.DMA, access="readwrite")
    except (AttributeError, RuntimeError):
        pytest.skip("DMA memory not supported on this platform")
    if t.memory != TensorMemory.DMA:
        pytest.skip("DMA requested but backend substituted another memory type")
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
    """A padded-pitch DMA image tensor exposes a *strided* (not sheared
    contiguous) buffer via the Python buffer protocol.

    End-to-end: write a known gradient with `from_numpy` (stride-aware write),
    read it back through `memoryview` / `np.asarray` (stride-aware read), then
    mutate through the live mapping and confirm the change is visible on a fresh
    map — i.e. the view is zero-copy onto the DMA-BUF, not a private copy.
    """
    w, h = 595, 438  # odd width → width*channels is not 64-aligned → padded
    t = _dma_image_or_skip(w, h, fmt)
    assert t.row_stride > w * channels, "expected a padded pitch for this width"
    assert t.row_stride % 64 == 0, "DMA pitch must be 64-byte aligned"

    ref = (np.arange(h * w * channels, dtype=np.uint8) % 251).reshape(h, w, channels)
    t.from_numpy(ref)

    with t.map() as m:
        mv = memoryview(m)
        # Logical shape, physical pitch as the outermost stride.
        assert mv.shape == (h, w, channels)
        assert mv.strides[0] == t.row_stride
        arr = np.asarray(mv)
        assert np.array_equal(arr, ref), "strided buffer-protocol read sheared the data"
        # Poke the live mapping (zero-copy write-through onto the DMA-BUF).
        poked = (int(ref[0, 0, 0]) ^ 0xFF) & 0xFF
        memoryview(m)[0, 0, 0] = poked

    with t.map() as m2:
        arr2 = np.asarray(memoryview(m2))
        assert int(arr2[0, 0, 0]) == poked, "mutation did not reach the DMA-BUF"

    # normalize_to_numpy reads the same padded backing and must agree.
    if channels == 3:
        t.from_numpy(ref)
        out = np.zeros((h, w, 3), dtype=np.uint8)
        t.normalize_to_numpy(out, Normalization.RAW, None)
        assert np.array_equal(out, ref), "normalize_to_numpy sheared the padded buffer"


@pytest.mark.parametrize(
    "fmt",
    [PixelFormat.Nv12, PixelFormat.Nv16, PixelFormat.Nv24],
    ids=["nv12", "nv16", "nv24"],
)
def test_dma_semiplanar_buffer_protocol_strided(fmt):
    """Semi-planar DMA tensors expose the combined Y+UV plane as a 2-D
    ``[rows, width]`` buffer at the physical pitch. Verify the pitch is the
    outer stride and that per-row writes through the mapping round-trip
    zero-copy (the chroma rows share the same padded pitch as luma)."""
    w, h = 595, 438
    t = _dma_image_or_skip(w, h, fmt)
    assert t.row_stride > w and t.row_stride % 64 == 0

    with t.map() as m:
        mv = memoryview(m)
        assert mv.ndim == 2 and mv.shape[1] == w
        assert mv.strides[0] == t.row_stride
        rows = mv.shape[0]
        for r in range(rows):
            mv[r, 0] = r % 256  # per-row marker through the live mapping

    with t.map() as m2:
        a2 = np.asarray(memoryview(m2))
        assert all(int(a2[r, 0]) == r % 256 for r in range(a2.shape[0])), (
            "per-row zero-copy round-trip failed on padded semi-planar buffer"
        )


def test_mem_tight_buffer_protocol_contiguous():
    """A tight (non-padded) image tensor still exposes a plain C-contiguous
    buffer — the strided exposure must only engage for padded backings."""
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
        tensor = Tensor([100, 100, 3], dtype="uint8", mem=TensorMemory.DMA)
    except (AttributeError, RuntimeError):
        pytest.skip("DMA memory not supported on this platform")

    tensor = Tensor([720, 1280, 4], dtype="uint8", mem=TensorMemory.DMA)
    with tensor.map() as m:
        np.frombuffer(m.numpy(), dtype=np.uint8).fill(233)

    fd = tensor.fd
    try:
        img = Tensor.from_fd(fd, [720, 1280, 4], dtype="uint8")
        with img.map() as m:
            data = np.frombuffer(m.numpy(), dtype=np.uint8).reshape(720, 1280, 4)
            assert (data == 233).all()
    except Exception:
        os.close(fd)


def test_from_fd_shm():
    try:
        tensor = Tensor([100, 100, 3], dtype="uint8", mem=TensorMemory.SHM)
    except (AttributeError, RuntimeError):
        pytest.skip("SHM memory not supported on this platform")

    tensor = Tensor([720, 1280, 4], dtype="uint8", mem=TensorMemory.SHM)
    with tensor.map() as m:
        np.frombuffer(m.numpy(), dtype=np.uint8).fill(233)

    fd = tensor.fd
    try:
        img = Tensor.from_fd(fd, [720, 1280, 4], dtype="uint8")
        with img.map() as m:
            data = np.frombuffer(m.numpy(), dtype=np.uint8).reshape(720, 1280, 4)
            assert (data == 233).all()
    except Exception:
        os.close(fd)


def test_create_image():
    """Test ImageProcessor.create_image() returns a valid, mappable image."""
    converter = ImageProcessor()
    img = converter.create_image(320, 240, PixelFormat.Rgba, access="readwrite")
    assert img.width == 320
    assert img.height == 240
    assert img.format == PixelFormat.Rgba

    # Image should be mappable
    with img.map() as m:
        data = np.frombuffer(m.numpy(), dtype=np.uint8)
        assert len(data) == 320 * 240 * 4


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
            data = np.frombuffer(m.numpy(), dtype=np.uint8)
            assert len(data) == 160 * 120 * channels


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

    # Convert into i8 destination should succeed
    w, h = _image_size("testdata/zidane.jpg")
    src = Tensor.image(w, h, format=PixelFormat.Nv12, access="readwrite")
    src.decode_image_file("testdata/zidane.jpg")
    dst = converter.create_image(
        640, 640, PixelFormat.Rgb, dtype="int8", access="readwrite"
    )
    converter.convert(src, dst)


def test_create_image_convert():
    """Test that images from create_image() work with convert()."""
    converter = ImageProcessor()

    # Load a source image normally (native NV12 from the JPEG)
    w, h = _image_size("testdata/zidane.jpg")
    src = Tensor.image(w, h, format=PixelFormat.Nv12, access="readwrite")
    src.decode_image_file("testdata/zidane.jpg")

    # Create destination via create_image (may use PBO, DMA, or Mem)
    dst = converter.create_image(640, 640, PixelFormat.Rgba, access="readwrite")

    # Convert should succeed regardless of backing type
    converter.convert(src, dst)

    # Verify the result has actual pixel data (not all zeros)
    n = np.zeros((640, 640, 3), dtype=np.uint8)
    dst.normalize_to_numpy(n)
    assert n.any(), "Destination image is all zeros after convert"

    # Compare with standard Tensor destination
    dst_mem = Tensor.image(640, 640, PixelFormat.Rgba, access="readwrite")
    converter.convert(src, dst_mem)
    n_mem = np.zeros((640, 640, 3), dtype=np.uint8)
    dst_mem.normalize_to_numpy(n_mem)

    assert calculate_similarity_rms_u8(n, n_mem) > 0.95


def test_create_image_roundtrip():
    """Test create_image as both src and dst for convert()."""
    converter = ImageProcessor()

    # Create source via create_image and fill it
    src = converter.create_image(640, 480, PixelFormat.Rgba, access="readwrite")
    with src.map() as m:
        data = np.frombuffer(m.numpy(), dtype=np.uint8).copy()
        # Fill with a gradient pattern
        data[:] = np.tile(np.arange(256, dtype=np.uint8), len(data) // 256 + 1)[
            : len(data)
        ]
        m.numpy()[:] = data

    # Create destination via create_image
    dst = converter.create_image(320, 240, PixelFormat.Rgba, access="readwrite")

    # Convert using both create_image tensors
    converter.convert(src, dst)

    # Verify destination has data (not zeros)
    with dst.map() as m:
        result = np.frombuffer(m.numpy(), dtype=np.uint8)
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
        src = Tensor([480, 640, 4], dtype="uint8", mem=TensorMemory.DMA)
    except RuntimeError:
        pytest.skip("DMA allocation not available on this platform")
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
    bg_native.decode_image_file("testdata/giraffe.jpg")
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
    # Verify dst matches the background (no detections → bg only). Compare
    # against the converted RGBA background itself (not PIL's RGBA decode),
    # since the background now arrives via an NV12->RGBA conversion.
    expected = np.zeros((h, w, 4), dtype=np.uint8)
    bg.normalize_to_numpy(expected)
    with dst.map() as m:
        img = np.array(m.numpy()).reshape((dst.height, dst.width, 4))
        assert calculate_similarity_rms_u8(img, expected) > 0.99


def test_draw_decoded_masks_multiple_boxes():
    """draw_decoded_masks with multiple detections (boxes only, no seg)."""
    w, h = _image_size("testdata/giraffe.jpg")
    converter = ImageProcessor()
    src = Tensor.image(w, h, format=PixelFormat.Nv12, access="readwrite")
    src.decode_image_file("testdata/giraffe.jpg")
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
    # Should render 3 boxes. We can't pixel-match without a reference,
    # but verify it didn't crash and image was modified.
    original = load_image("testdata/giraffe.jpg", "RGBA")
    with dst.map() as m:
        img = np.array(m.numpy()).reshape((dst.height, dst.width, 4))
        # The image should be different from original (boxes drawn on it)
        assert calculate_similarity_rms_u8(img, original) < 0.999


def test_draw_decoded_masks_instance_color_mode():
    """draw_decoded_masks with ColorMode.Instance assigns per-detection colors."""
    w, h = _image_size("testdata/giraffe.jpg")
    converter = ImageProcessor()
    src = Tensor.image(w, h, format=PixelFormat.Nv12, access="readwrite")
    src.decode_image_file("testdata/giraffe.jpg")
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
    # Verify it didn't crash and image was modified
    original = load_image("testdata/giraffe.jpg", "RGBA")
    with dst.map() as m:
        img = np.array(m.numpy()).reshape((dst.height, dst.width, 4))
        assert calculate_similarity_rms_u8(img, original) < 0.999


def test_draw_decoded_masks_with_opacity():
    """draw_decoded_masks with reduced opacity makes overlays more transparent."""
    w, h = _image_size("testdata/giraffe.jpg")
    converter = ImageProcessor()
    src = Tensor.image(w, h, format=PixelFormat.Nv12, access="readwrite")
    src.decode_image_file("testdata/giraffe.jpg")
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
        # Half-opacity result should be closer to the original than full-opacity
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

    # Create a synthetic float detection tensor: 1 detection with box + score + class
    data = np.zeros((1, 10, 6), dtype=np.float32)
    data[0, 0, :] = [0.1, 0.1, 0.3, 0.3, 0.9, 2.0]  # box + score + class
    model_output_tensor = Tensor([1, 10, 6], dtype="float32")
    with model_output_tensor.map() as m:
        buf = np.frombuffer(m.numpy(), dtype=np.float32).reshape((1, 10, 6))
        buf[:] = data

    w, h = _image_size("testdata/giraffe.jpg")
    converter = ImageProcessor()
    src = Tensor.image(w, h, format=PixelFormat.Nv12, access="readwrite")
    src.decode_image_file("testdata/giraffe.jpg")
    dst = converter.create_image(w, h, PixelFormat.Rgba, access="readwrite")
    converter.convert(src, dst)
    result = converter.draw_masks(decoder, [model_output_tensor], dst)

    # Should return (boxes, scores, classes)
    assert len(result) == 3
    boxes, scores, classes = result
    assert boxes.shape[1] == 4
    assert len(scores) == len(boxes)
    assert len(classes) == len(boxes)


def test_decode_image_from_bytes():
    """Test decode_image with raw bytes into oversized tensor (strided)."""
    data = open("testdata/zidane.jpg", "rb").read()
    # Color JPEG decodes to its native NV12; the tensor is reconfigured to match.
    tensor = Tensor.image(
        MAX_SRC_W, MAX_SRC_H, format=PixelFormat.Nv12, access="readwrite"
    )
    info: ImageInfo = tensor.decode_image(data)
    assert info.width == 1280
    assert info.height == 720
    assert info.format == PixelFormat.Nv12
    # The decoder reconfigures the tensor's LOGICAL width/height to the decoded
    # image (1280x720) but PRESERVES the oversized buffer's allocated row stride
    # so a consumer can still index rows of the reused buffer correctly. The
    # tensor was allocated MAX_SRC_W (1920) wide — already 64-aligned — so the
    # reported stride stays 1920, wider than the 1280 logical width. (Reporting
    # the logical width here would make the buffer unreadable: rows would be
    # indexed at the wrong byte offset.)
    assert info.row_stride == MAX_SRC_W
    assert info.row_stride >= info.width
    assert info.row_stride % 64 == 0


def test_decode_image_reuse():
    """Test decoding multiple different images into the same oversized tensor.

    The decoder configures the tensor's format per image, so the same buffer
    is reused across a color JPEG (NV12) and a greyscale JPEG (Grey).
    """
    tensor = Tensor.image(
        MAX_SRC_W, MAX_SRC_H, format=PixelFormat.Nv12, access="readwrite"
    )

    # First decode: zidane.jpg (1280x720) — native NV12, strided into 1920x1080
    info1 = tensor.decode_image_file("testdata/zidane.jpg")
    assert info1.width == 1280
    assert info1.height == 720
    assert info1.format == PixelFormat.Nv12

    # Second decode: grey.jpg (1024x681) — native Grey, reconfigured in place
    info2 = tensor.decode_image_file("testdata/grey.jpg")
    assert info2.width == 1024
    assert info2.height == 681
    assert info2.format == PixelFormat.Grey


def test_decode_image_native_format():
    """Test decode_image emits the source's native format (NV12 for color JPEG)."""
    tensor = Tensor.image(
        MAX_SRC_W, MAX_SRC_H, format=PixelFormat.Nv12, access="readwrite"
    )
    info = tensor.decode_image_file("testdata/zidane.jpg")
    assert info.width == 1280
    assert info.height == 720
    assert info.format == PixelFormat.Nv12


def test_decode_image_grey():
    """Test decoding into Grey format with oversized tensor."""
    tensor = Tensor.image(
        MAX_SRC_W, MAX_SRC_H, format=PixelFormat.Grey, access="readwrite"
    )
    info = tensor.decode_image_file("testdata/grey.jpg")
    assert info.width == 1024
    assert info.height == 681
    assert info.format == PixelFormat.Grey


def test_decode_image_pipeline():
    """Test the full recommended pipeline: create_image -> decode -> convert."""
    converter = ImageProcessor()

    # Allocate via ImageProcessor for the optimal memory backend, sized to the
    # image. Decode emits native NV12; convert produces the RGBA result.
    w, h = _image_size("testdata/zidane.jpg")
    src = converter.create_image(w, h, PixelFormat.Nv12, access="readwrite")
    src.decode_image_file("testdata/zidane.jpg")

    dst = converter.create_image(640, 640, PixelFormat.Rgba, access="readwrite")
    converter.convert(src, dst)

    # Verify result has pixel data
    n = np.zeros((640, 640, 3), dtype=np.uint8)
    dst.normalize_to_numpy(n)
    assert n.any(), "Destination image is all zeros after pipeline"


def test_decode_image_exact_size():
    """Allocate tensor at exact decoded image size by reading the JPEG header first."""
    # Use PIL to read just the header (no pixel decode)
    with Image.open("testdata/zidane.jpg") as im:
        w, h = im.size

    # Allocate tensor at exact dimensions. Color JPEG decodes to native NV12;
    # the NV12 luma plane is one byte per pixel, so an exact-size tensor has
    # row_stride == width (no padding).
    tensor = Tensor.image(w, h, format=PixelFormat.Nv12, access="readwrite")
    info = tensor.decode_image_file("testdata/zidane.jpg")
    assert info.width == w
    assert info.height == h
    assert info.format == PixelFormat.Nv12
    assert info.row_stride == w


def test_decode_image_exact_size_png():
    """Exact-size decode for PNG format (native RGB for this fixture)."""
    with Image.open("testdata/zidane.png") as im:
        w, h = im.size

    # zidane.png is a 3-channel RGB PNG; decode emits its native RGB format.
    tensor = Tensor.image(w, h, format=PixelFormat.Rgb, access="readwrite")
    info = tensor.decode_image_file("testdata/zidane.png")
    assert info.width == w
    assert info.height == h
    assert info.format == PixelFormat.Rgb
    assert info.row_stride == w * 3


def test_decode_image_tensor_too_small_width():
    """Decoding into a tensor narrower than the image should raise an error."""
    tensor = Tensor.image(320, 720, format=PixelFormat.Nv12, access="readwrite")
    with pytest.raises(RuntimeError, match="[Cc]apacity|[Ss]ize|[Ww]idth|[Ss]tride"):
        tensor.decode_image_file("testdata/zidane.jpg")


def test_decode_image_tensor_too_small_height():
    """Decoding into a tensor shorter than the image should raise an error."""
    tensor = Tensor.image(1280, 360, format=PixelFormat.Nv12, access="readwrite")
    with pytest.raises(RuntimeError, match="[Cc]apacity|[Ss]ize|[Hh]eight|[Rr]ow"):
        tensor.decode_image_file("testdata/zidane.jpg")


def test_decode_image_tensor_too_small_both():
    """Decoding into a tensor smaller in both dimensions should raise an error."""
    tensor = Tensor.image(100, 100, format=PixelFormat.Nv12, access="readwrite")
    with pytest.raises(RuntimeError):
        tensor.decode_image_file("testdata/zidane.jpg")


def test_decode_image_strided_pixel_correctness():
    """Verify strided decode produces identical pixels to exact-size decode.

    Color JPEGs decode to native NV12; this compares the NV12 luma (Y) plane
    of an exact-size decode against the strided sub-region of an oversized
    decode, exercising the per-row stride handling.
    """
    # Exact-size decode (no stride padding) — luma stride == width.
    w, h = _image_size("testdata/zidane.jpg")
    exact = Tensor.image(w, h, format=PixelFormat.Nv12, access="readwrite")
    exact.decode_image_file("testdata/zidane.jpg")
    with exact.map() as m:
        exact_y = np.frombuffer(m.numpy(), dtype=np.uint8)[: w * h].reshape(h, w).copy()

    # Oversized decode (strided — the common real-world path).
    big = Tensor.image(
        MAX_SRC_W, MAX_SRC_H, format=PixelFormat.Nv12, access="readwrite"
    )
    info = big.decode_image_file("testdata/zidane.jpg")

    # Extract the decoded luma sub-region via map() and the reported row stride.
    with big.map() as m:
        raw = np.frombuffer(m.numpy(), dtype=np.uint8)
        stride = info.row_stride
        big_y = np.zeros((h, w), dtype=np.uint8)
        for row in range(h):
            start = row * stride
            big_y[row] = raw[start : start + w]

    # Pixels must match exactly — strided decode must not corrupt data.
    assert np.array_equal(exact_y, big_y), (
        "Strided decode pixels differ from exact-size decode"
    )


@pytest.mark.parametrize(
    "width,height",
    [
        (105, 124),  # the original STRIDES_BUG.md repro
        (100, 124),
        (160, 80),
        (200, 64),
    ],
)
def test_from_numpy_grey_unaligned_width_stride_bug(width, height):
    """Regression for STRIDES_BUG.md.

    The original bug: `create_image(w, h, Grey)` on backends that
    apply a GPU pitch alignment rounds the underlying buffer up
    (e.g. 105 → 128-byte pitch on Mali Valhall), but `Image.size`
    still reports the *logical* `w × h`. A numpy array sized from
    `Image.size` was the natural caller choice, so `from_numpy`
    panicked in `copy_from_slice` on the length mismatch with the
    padded destination. The fix (`tensor.rs`) detects the padded
    destination via `dst_len > tensor_len` and switches to a
    row-by-row copy using `effective_row_stride()`.

    This test locks in that `from_numpy` no longer panics for a
    non-64-aligned width, and that `Image.size` / `Image.row_stride`
    satisfy the documented contract (size is logical, row_stride is
    reported and at least the logical row width). Byte-level
    correctness of the mapped write is backend-sensitive — PBO
    staging paths reinterpret the buffer with their own stride
    between write and read — and is covered by higher-level
    end-to-end tests rather than a regression test whose concern is
    the panic.
    """
    converter = ImageProcessor()
    img = converter.create_image(width, height, PixelFormat.Grey, access="readwrite")

    assert img.size == width * height, (
        f"Image.size ({img.size}) must remain logical (w*h={width * height})"
    )

    rs = img.row_stride
    assert rs is not None and rs >= width, f"row_stride {rs} must be >= width {width}"

    # The regression: this call used to panic in copy_from_slice for
    # widths not aligned to the GPU pitch. Must now succeed.
    src = np.arange(width * height, dtype=np.uint8).reshape(height, width)
    img.from_numpy(src)

    # The mapped buffer must be addressable at the logical size. All
    # reads must stay inside the `with` block — `m.numpy()` returns a
    # zero-copy numpy view that dangles once the TensorMap context
    # exits; touching it afterwards (including via pytest saferepr on
    # assertion failure) segfaults.
    with img.map() as m:
        raw = np.frombuffer(m.numpy(), dtype=np.uint8)
        assert len(raw) == width * height, (
            f"mapped view length {len(raw)} != Image.size {width * height}"
        )


def test_convert_honors_tagged_colorimetry(monkeypatch):
    """convert() must honor the source tensor's tagged colorimetry (QA F11).

    The same NV12 bytes decoded as BT.601 full-range vs BT.709 limited-range
    must produce visibly different RGB pixels — proving the HAL threads the
    per-source colorimetry tag through the conversion instead of applying one
    hardcoded YUV matrix/range.

    Exact tag honoring is the ColorimetryMode::Exact contract, opted into
    here via EDGEFIRST_COLORIMETRY. The Fast default (issue #106)
    deliberately collapses colorimetries into the external sampler's fixed
    conversion on Vivante (~12x faster); non-Vivante GPUs honor tags in both
    modes.
    """
    monkeypatch.setenv("EDGEFIRST_COLORIMETRY", "exact")
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
    # A whole-matrix/range change shifts pixels well beyond a stray rounding
    # difference; require a clearly visible mean delta.
    mean_diff = np.abs(bt601_full.astype(int) - bt709_limited.astype(int)).mean()
    assert mean_diff > 1.0, f"mean pixel delta {mean_diff:.3f} too small to be real"
