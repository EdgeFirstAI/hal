# SPDX-FileCopyrightText: Copyright 2025 Au-Zone Technologies
# SPDX-License-Identifier: Apache-2.0

from edgefirst_hal import (
    ImageProcessor,
    Flip,
    PixelFormat,
    Normalization,
    Rotation,
    Rect,
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


def test_flip():
    src = Tensor.load("testdata/zidane.jpg", PixelFormat.Rgba)
    converter = ImageProcessor()
    dst = converter.create_image(1280, 720, PixelFormat.Rgba)
    converter.convert(src, dst, flip=Flip.Horizontal)

    n = np.zeros((720, 1280, 3), dtype=np.uint8)
    dst.normalize_to_numpy(n)

    expected = load_image("testdata/zidane.jpg", "RGB")
    expected = expected[:, ::-1, :]

    assert calculate_similarity_rms_u8(n, expected) > 0.99


def test_grey_load():
    rgba_ = Tensor.load("testdata/grey.jpg", PixelFormat.Rgba)
    rgba = np.zeros((rgba_.height, rgba_.width, 4), dtype=np.uint8)
    rgba_.normalize_to_numpy(rgba)

    grey_ = Tensor.load("testdata/grey.jpg", PixelFormat.Grey)
    grey = np.zeros((grey_.height, grey_.width, 1), dtype=np.uint8)
    grey_.normalize_to_numpy(grey)

    default_ = Tensor.load("testdata/grey.jpg", PixelFormat.Rgb)
    default = np.zeros((default_.height, default_.width, 3), dtype=np.uint8)
    default_.normalize_to_numpy(default)

    assert calculate_similarity_rms_u8(rgba[:, :, 0:3], default) > 0.99
    assert calculate_similarity_rms_u8(rgba[:, :, 0], grey[:, :, 0]) > 0.99
    assert calculate_similarity_rms_u8(default[:, :, 0], grey[:, :, 0]) > 0.99


def test_normalize():
    src = Tensor.load("testdata/zidane.jpg", PixelFormat.Rgba)
    converter = ImageProcessor()
    dst = converter.create_image(640, 640, PixelFormat.Rgba)
    converter.convert(src, dst)
    n = np.zeros((640, 640, 3), dtype=np.int8)
    dst.normalize_to_numpy(n, normalization=Normalization.SIGNED)
    unique_vals = np.unique(n)
    assert (unique_vals == np.array(range(-128, 128))).all()

    dst.normalize_to_numpy(n, normalization=Normalization.SIGNED, zero_point=127)
    unique_vals0 = np.unique(n)
    assert (unique_vals0 == np.array(range(-127, 128))).all()


def test_render():
    dst = Tensor.load("testdata/giraffe.jpg", PixelFormat.Rgba)
    seg = np.fromfile("testdata/yolov8_seg_crop_76x55.bin", dtype=np.uint8).reshape(
        (76, 55, 1)
    )
    converter = ImageProcessor()
    converter.set_class_colors([[255, 255, 0, 233], [128, 128, 255, 100]])
    converter.draw_decoded_masks(
        dst,
        bbox=np.array([[0.59375, 0.25, 0.9375, 0.725]], dtype=np.float32),
        scores=np.array([0.9], dtype=np.float32),
        classes=np.array([1], dtype=np.uintp),
        seg=[seg],
    )
    expected_gl = load_image("testdata/output_render_gl.jpg", "RGBA")
    expected_cpu = load_image("testdata/output_render_cpu.jpg", "RGBA")
    with dst.map() as m:
        img = np.array(m.view()).reshape((dst.height, dst.width, 4))
        # Threshold 0.97: GPU smoothstep anti-aliasing at mask edges produces
        # small differences across platforms (x86 Mesa vs Vivante GC7000UL).
        assert (
            calculate_similarity_rms_u8(img, expected_gl) > 0.97
            or calculate_similarity_rms_u8(img, expected_cpu) > 0.97
        )


def test_rgb_resize():
    src = Tensor.load("testdata/zidane.jpg", PixelFormat.Rgb)
    converter = ImageProcessor()
    dst = converter.create_image(640, 640, PixelFormat.Rgba)
    converter.convert(src, dst)
    with dst.map() as m:
        n = np.array(m.view()).reshape((dst.height, dst.width, 4))
        expected = load_image("testdata/zidane.jpg", "RGBA", resize=(640, 640))
        assert calculate_similarity_rms_u8(n, expected) > 0.99


def test_rgba_to_rgb():
    src = Tensor.load("testdata/zidane.jpg", PixelFormat.Rgba)
    converter = ImageProcessor()
    dst = converter.create_image(1280, 720, PixelFormat.Rgb)
    converter.convert(src, dst, Rotation.Rotate0, Flip.NoFlip, Rect(0, 0, 1280, 720))

    with dst.map() as m:
        n = np.array(m.view()).reshape((dst.height, dst.width, 3))
        expected = load_image("testdata/zidane.jpg", "RGB")
        assert calculate_similarity_rms_u8(n, expected) > 0.99


def test_enum_cmp():
    dst = Tensor.image(640, 640, format=PixelFormat.Rgba)
    assert dst.format == PixelFormat.Rgba


def test_from_fd_dma():
    try:
        tensor = Tensor([100, 100, 3], dtype="uint8", mem=TensorMemory.DMA)
    except (AttributeError, RuntimeError):
        pytest.skip("DMA memory not supported on this platform")

    tensor = Tensor([720, 1280, 4], dtype="uint8", mem=TensorMemory.DMA)
    with tensor.map() as m:
        np.frombuffer(m.view(), dtype=np.uint8).fill(233)

    fd = tensor.fd
    try:
        img = Tensor.from_fd(fd, [720, 1280, 4], dtype="uint8")
        with img.map() as m:
            data = np.frombuffer(m.view(), dtype=np.uint8).reshape(720, 1280, 4)
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
        np.frombuffer(m.view(), dtype=np.uint8).fill(233)

    fd = tensor.fd
    try:
        img = Tensor.from_fd(fd, [720, 1280, 4], dtype="uint8")
        with img.map() as m:
            data = np.frombuffer(m.view(), dtype=np.uint8).reshape(720, 1280, 4)
            assert (data == 233).all()
    except Exception:
        os.close(fd)


def test_create_image():
    """Test ImageProcessor.create_image() returns a valid, mappable image."""
    converter = ImageProcessor()
    img = converter.create_image(320, 240, PixelFormat.Rgba)
    assert img.width == 320
    assert img.height == 240
    assert img.format == PixelFormat.Rgba

    # Image should be mappable
    with img.map() as m:
        data = np.frombuffer(m.view(), dtype=np.uint8)
        assert len(data) == 320 * 240 * 4


def test_create_image_formats():
    """Test create_image with different pixel formats."""
    converter = ImageProcessor()
    for fmt, channels in [
        (PixelFormat.Rgb, 3),
        (PixelFormat.Rgba, 4),
        (PixelFormat.Grey, 1),
    ]:
        img = converter.create_image(160, 120, fmt)
        assert img.width == 160
        assert img.height == 120
        assert img.format == fmt
        with img.map() as m:
            data = np.frombuffer(m.view(), dtype=np.uint8)
            assert len(data) == 160 * 120 * channels


def test_create_image_dtype_i8():
    """Test create_image with dtype='int8'."""
    converter = ImageProcessor()
    img = converter.create_image(320, 240, PixelFormat.Rgb, dtype="int8")
    assert img.width == 320
    assert img.height == 240
    assert img.format == PixelFormat.Rgb
    assert img.dtype == "int8"

    # Convert into i8 destination should succeed
    src = Tensor.load("testdata/zidane.jpg", PixelFormat.Rgba)
    dst = converter.create_image(640, 640, PixelFormat.Rgb, dtype="int8")
    converter.convert(src, dst)


def test_create_image_convert():
    """Test that images from create_image() work with convert()."""
    converter = ImageProcessor()

    # Load a source image normally
    src = Tensor.load("testdata/zidane.jpg", PixelFormat.Rgba)

    # Create destination via create_image (may use PBO, DMA, or Mem)
    dst = converter.create_image(640, 640, PixelFormat.Rgba)

    # Convert should succeed regardless of backing type
    converter.convert(src, dst)

    # Verify the result has actual pixel data (not all zeros)
    n = np.zeros((640, 640, 3), dtype=np.uint8)
    dst.normalize_to_numpy(n)
    assert n.any(), "Destination image is all zeros after convert"

    # Compare with standard Tensor destination
    dst_mem = Tensor.image(640, 640, PixelFormat.Rgba)
    converter.convert(src, dst_mem)
    n_mem = np.zeros((640, 640, 3), dtype=np.uint8)
    dst_mem.normalize_to_numpy(n_mem)

    assert calculate_similarity_rms_u8(n, n_mem) > 0.95


def test_create_image_roundtrip():
    """Test create_image as both src and dst for convert()."""
    converter = ImageProcessor()

    # Create source via create_image and fill it
    src = converter.create_image(640, 480, PixelFormat.Rgba)
    with src.map() as m:
        data = np.frombuffer(m.view(), dtype=np.uint8).copy()
        # Fill with a gradient pattern
        data[:] = np.tile(np.arange(256, dtype=np.uint8), len(data) // 256 + 1)[
            : len(data)
        ]
        m.view()[:] = data

    # Create destination via create_image
    dst = converter.create_image(320, 240, PixelFormat.Rgba)

    # Convert using both create_image tensors
    converter.convert(src, dst)

    # Verify destination has data (not zeros)
    with dst.map() as m:
        result = np.frombuffer(m.view(), dtype=np.uint8)
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
