# SPDX-FileCopyrightText: Copyright 2025 Au-Zone Technologies
# SPDX-License-Identifier: Apache-2.0

from edgefirst_hal import (
    ColorMode,
    Decoder,
    ImageInfo,
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


# Maximum source tensor dimensions — oversized to exercise strided decode.
# Used by decode-specific tests where pixel comparison goes through
# normalize_to_numpy at the decoded sub-region size.
MAX_SRC_W, MAX_SRC_H = 1920, 1080


def _image_size(path):
    """Read image dimensions from file header without decoding pixels."""
    with Image.open(path) as im:
        return im.size  # (width, height)


def test_flip():
    w, h = _image_size("testdata/zidane.jpg")
    # JPEG decodes to its native NV12; convert downstream produces RGBA.
    src = Tensor.image(w, h, format=PixelFormat.Nv12)
    src.decode_image_file("testdata/zidane.jpg")
    converter = ImageProcessor()
    dst = converter.create_image(1280, 720, PixelFormat.Rgba)
    converter.convert(src, dst, flip=Flip.Horizontal)

    n = np.zeros((720, 1280, 3), dtype=np.uint8)
    dst.normalize_to_numpy(n)

    expected = load_image("testdata/zidane.jpg", "RGB")
    expected = expected[:, ::-1, :]

    # Decode produces native NV12; convert() applies a BT.601 full-range (JFIF)
    # YUV->RGB transform that matches PIL's direct JPEG RGB decode used to build
    # the reference. Only the codec's 4:2:0 chroma downsampling differs, so 0.98
    # holds. (Interim 601-full hardcode; see crates/image/ARCHITECTURE.md.)
    assert calculate_similarity_rms_u8(n, expected) > 0.98


def test_grey_load():
    w, h = _image_size("testdata/grey.jpg")
    converter = ImageProcessor()

    # A greyscale JPEG decodes to its native Grey format.
    grey_ = Tensor.image(w, h, format=PixelFormat.Grey)
    info = grey_.decode_image_file("testdata/grey.jpg")
    assert info.format == PixelFormat.Grey
    grey = np.zeros((h, w, 1), dtype=np.uint8)
    grey_.normalize_to_numpy(grey)

    # RGBA / RGB variants are obtained by converting the native Grey image.
    rgba_ = converter.create_image(w, h, PixelFormat.Rgba)
    converter.convert(grey_, rgba_)
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    rgba_.normalize_to_numpy(rgba)

    default_ = converter.create_image(w, h, PixelFormat.Rgb)
    converter.convert(grey_, default_)
    default = np.zeros((h, w, 3), dtype=np.uint8)
    default_.normalize_to_numpy(default)

    assert calculate_similarity_rms_u8(rgba[:, :, 0:3], default) > 0.99
    assert calculate_similarity_rms_u8(rgba[:, :, 0], grey[:, :, 0]) > 0.99
    assert calculate_similarity_rms_u8(default[:, :, 0], grey[:, :, 0]) > 0.99


def test_normalize():
    w, h = _image_size("testdata/zidane.jpg")
    src = Tensor.image(w, h, format=PixelFormat.Nv12)
    src.decode_image_file("testdata/zidane.jpg")
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
    w, h = _image_size("testdata/giraffe.jpg")
    converter = ImageProcessor()
    # Decode the JPEG to its native NV12, then convert to an RGBA background.
    bg_native = Tensor.image(w, h, format=PixelFormat.Nv12)
    bg_native.decode_image_file("testdata/giraffe.jpg")
    bg = Tensor.image(w, h, format=PixelFormat.Rgba)
    converter.convert(bg_native, bg)
    seg = np.fromfile("testdata/yolov8_seg_crop_76x55.bin", dtype=np.uint8).reshape(
        (76, 55, 1)
    )
    converter.set_class_colors([[255, 255, 0, 233], [128, 128, 255, 100]])
    dst = converter.create_image(w, h, PixelFormat.Rgba)
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
        img = np.array(m.view()).reshape((dst.height, dst.width, 4))
        # Threshold 0.95: GPU smoothstep anti-aliasing at mask edges produces
        # small differences across platforms (x86 Mesa vs Vivante GC7000UL),
        # and the background now arrives via an NV12->RGBA conversion that
        # differs slightly from the reference renders' direct-RGBA decode.
        assert (
            calculate_similarity_rms_u8(img, expected_gl) > 0.95
            or calculate_similarity_rms_u8(img, expected_cpu) > 0.95
        )


def test_rgb_resize():
    w, h = _image_size("testdata/zidane.jpg")
    src = Tensor.image(w, h, format=PixelFormat.Nv12)
    src.decode_image_file("testdata/zidane.jpg")
    converter = ImageProcessor()
    dst = converter.create_image(640, 640, PixelFormat.Rgba)
    converter.convert(src, dst)
    with dst.map() as m:
        n = np.array(m.view()).reshape((dst.height, dst.width, 4))
        expected = load_image("testdata/zidane.jpg", "RGBA", resize=(640, 640))
        # 0.95: NV12->RGB BT.709 conversion differs slightly from PIL's RGB.
        assert calculate_similarity_rms_u8(n, expected) > 0.95


def test_rgba_to_rgb():
    w, h = _image_size("testdata/zidane.jpg")
    # Decode the JPEG to native NV12, then convert to RGB (3-channel) output.
    src = Tensor.image(w, h, format=PixelFormat.Nv12)
    src.decode_image_file("testdata/zidane.jpg")
    converter = ImageProcessor()
    dst = converter.create_image(w, h, PixelFormat.Rgb)
    converter.convert(src, dst, Rotation.Rotate0, Flip.NoFlip, Rect(0, 0, w, h))

    with dst.map() as m:
        n = np.array(m.view()).reshape((dst.height, dst.width, 3))
        expected = load_image("testdata/zidane.jpg", "RGB")
        # 0.95: NV12->RGB BT.709 conversion differs slightly from PIL's RGB.
        assert calculate_similarity_rms_u8(n, expected) > 0.95


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
    w, h = _image_size("testdata/zidane.jpg")
    src = Tensor.image(w, h, format=PixelFormat.Nv12)
    src.decode_image_file("testdata/zidane.jpg")
    dst = converter.create_image(640, 640, PixelFormat.Rgb, dtype="int8")
    converter.convert(src, dst)


def test_create_image_convert():
    """Test that images from create_image() work with convert()."""
    converter = ImageProcessor()

    # Load a source image normally (native NV12 from the JPEG)
    w, h = _image_size("testdata/zidane.jpg")
    src = Tensor.image(w, h, format=PixelFormat.Nv12)
    src.decode_image_file("testdata/zidane.jpg")

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


# ─── Mask rendering tests ────────────────────────────────────────


def test_draw_decoded_masks_empty():
    """draw_decoded_masks with zero detections should produce background only."""
    w, h = _image_size("testdata/giraffe.jpg")
    converter = ImageProcessor()
    bg_native = Tensor.image(w, h, format=PixelFormat.Nv12)
    bg_native.decode_image_file("testdata/giraffe.jpg")
    bg = Tensor.image(w, h, format=PixelFormat.Rgba)
    converter.convert(bg_native, bg)
    dst = converter.create_image(w, h, PixelFormat.Rgba)
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
        img = np.array(m.view()).reshape((dst.height, dst.width, 4))
        assert calculate_similarity_rms_u8(img, expected) > 0.99


def test_draw_decoded_masks_multiple_boxes():
    """draw_decoded_masks with multiple detections (boxes only, no seg)."""
    w, h = _image_size("testdata/giraffe.jpg")
    converter = ImageProcessor()
    src = Tensor.image(w, h, format=PixelFormat.Nv12)
    src.decode_image_file("testdata/giraffe.jpg")
    dst = converter.create_image(w, h, PixelFormat.Rgba)
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
        img = np.array(m.view()).reshape((dst.height, dst.width, 4))
        # The image should be different from original (boxes drawn on it)
        assert calculate_similarity_rms_u8(img, original) < 0.999


def test_draw_decoded_masks_instance_color_mode():
    """draw_decoded_masks with ColorMode.Instance assigns per-detection colors."""
    w, h = _image_size("testdata/giraffe.jpg")
    converter = ImageProcessor()
    src = Tensor.image(w, h, format=PixelFormat.Nv12)
    src.decode_image_file("testdata/giraffe.jpg")
    dst = converter.create_image(w, h, PixelFormat.Rgba)
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
        img = np.array(m.view()).reshape((dst.height, dst.width, 4))
        assert calculate_similarity_rms_u8(img, original) < 0.999


def test_draw_decoded_masks_with_opacity():
    """draw_decoded_masks with reduced opacity makes overlays more transparent."""
    w, h = _image_size("testdata/giraffe.jpg")
    converter = ImageProcessor()
    src = Tensor.image(w, h, format=PixelFormat.Nv12)
    src.decode_image_file("testdata/giraffe.jpg")
    dst_full = converter.create_image(w, h, PixelFormat.Rgba)
    converter.convert(src, dst_full)
    dst_half = converter.create_image(w, h, PixelFormat.Rgba)
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
        img_full = np.array(m1.view()).reshape((dst_full.height, dst_full.width, 4))
        img_half = np.array(m2.view()).reshape((dst_half.height, dst_half.width, 4))
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
        buf = np.frombuffer(m.view(), dtype=np.float32).reshape((1, 10, 6))
        buf[:] = data

    w, h = _image_size("testdata/giraffe.jpg")
    converter = ImageProcessor()
    src = Tensor.image(w, h, format=PixelFormat.Nv12)
    src.decode_image_file("testdata/giraffe.jpg")
    dst = converter.create_image(w, h, PixelFormat.Rgba)
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
    tensor = Tensor.image(MAX_SRC_W, MAX_SRC_H, format=PixelFormat.Nv12)
    info: ImageInfo = tensor.decode_image(data)
    assert info.width == 1280
    assert info.height == 720
    assert info.format == PixelFormat.Nv12
    # NV12 luma plane is one byte per pixel; the reported stride is the decoded
    # row width (the decoder reconfigures the tensor to the image dimensions).
    assert info.row_stride == info.width


def test_decode_image_reuse():
    """Test decoding multiple different images into the same oversized tensor.

    The decoder configures the tensor's format per image, so the same buffer
    is reused across a color JPEG (NV12) and a greyscale JPEG (Grey).
    """
    tensor = Tensor.image(MAX_SRC_W, MAX_SRC_H, format=PixelFormat.Nv12)

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
    tensor = Tensor.image(MAX_SRC_W, MAX_SRC_H, format=PixelFormat.Nv12)
    info = tensor.decode_image_file("testdata/zidane.jpg")
    assert info.width == 1280
    assert info.height == 720
    assert info.format == PixelFormat.Nv12


def test_decode_image_grey():
    """Test decoding into Grey format with oversized tensor."""
    tensor = Tensor.image(MAX_SRC_W, MAX_SRC_H, format=PixelFormat.Grey)
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
    src = converter.create_image(w, h, PixelFormat.Nv12)
    src.decode_image_file("testdata/zidane.jpg")

    dst = converter.create_image(640, 640, PixelFormat.Rgba)
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
    tensor = Tensor.image(w, h, format=PixelFormat.Nv12)
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
    tensor = Tensor.image(w, h, format=PixelFormat.Rgb)
    info = tensor.decode_image_file("testdata/zidane.png")
    assert info.width == w
    assert info.height == h
    assert info.format == PixelFormat.Rgb
    assert info.row_stride == w * 3


def test_decode_image_tensor_too_small_width():
    """Decoding into a tensor narrower than the image should raise an error."""
    tensor = Tensor.image(320, 720, format=PixelFormat.Nv12)
    with pytest.raises(RuntimeError, match="[Cc]apacity|[Ss]ize|[Ww]idth|[Ss]tride"):
        tensor.decode_image_file("testdata/zidane.jpg")


def test_decode_image_tensor_too_small_height():
    """Decoding into a tensor shorter than the image should raise an error."""
    tensor = Tensor.image(1280, 360, format=PixelFormat.Nv12)
    with pytest.raises(RuntimeError, match="[Cc]apacity|[Ss]ize|[Hh]eight|[Rr]ow"):
        tensor.decode_image_file("testdata/zidane.jpg")


def test_decode_image_tensor_too_small_both():
    """Decoding into a tensor smaller in both dimensions should raise an error."""
    tensor = Tensor.image(100, 100, format=PixelFormat.Nv12)
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
    exact = Tensor.image(w, h, format=PixelFormat.Nv12)
    exact.decode_image_file("testdata/zidane.jpg")
    with exact.map() as m:
        exact_y = np.frombuffer(m.view(), dtype=np.uint8)[: w * h].reshape(h, w).copy()

    # Oversized decode (strided — the common real-world path).
    big = Tensor.image(MAX_SRC_W, MAX_SRC_H, format=PixelFormat.Nv12)
    info = big.decode_image_file("testdata/zidane.jpg")

    # Extract the decoded luma sub-region via map() and the reported row stride.
    with big.map() as m:
        raw = np.frombuffer(m.view(), dtype=np.uint8)
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
    img = converter.create_image(width, height, PixelFormat.Grey)

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
    # reads must stay inside the `with` block — `m.view()` returns a
    # zero-copy numpy view that dangles once the TensorMap context
    # exits; touching it afterwards (including via pytest saferepr on
    # assertion failure) segfaults.
    with img.map() as m:
        raw = np.frombuffer(m.view(), dtype=np.uint8)
        assert len(raw) == width * height, (
            f"mapped view length {len(raw)} != Image.size {width * height}"
        )
