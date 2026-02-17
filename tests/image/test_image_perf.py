# SPDX-FileCopyrightText: Copyright 2025 Au-Zone Technologies
# SPDX-License-Identifier: Apache-2.0

import pytest
import os
import math

from edgefirst_hal import TensorImage, ImageProcessor, Flip, FourCC, Rotation
import numpy as np

pytestmark = pytest.mark.benchmark

Image = pytest.importorskip("PIL.Image")
dst_size = (512, 512)


# PIL image rotation is counter clockwise
def load_image(image, format="RGBA", resize=None, rotate=None):
    im = Image.open(image).convert(format)
    if resize is not None:
        im = im.resize(resize)
    if rotate is not None:
        im = im.transpose(rotate)
    return np.array(im)


# PIL image rotation is counter clockwise
def load_bytes_to_image(bytes, format, size, resize=None, rotate=None):
    im = Image.frombytes(format, size, bytes)
    if resize is not None:
        im = im.resize(resize)
    if rotate is not None:
        im = im.transpose(rotate)

    return np.array(im)


def calculate_similarity_rms_u8(imageA, imageB) -> float:
    imgA = np.asarray(imageA)
    imgB = np.asarray(imageB)

    imgA = imgA.astype("float") / 255.0
    imgB = imgB.astype("float") / 255.0

    squared_diff = (imgA - imgB) ** 2

    rms = math.sqrt(float(np.mean(squared_diff)))
    return 1.0 - rms


original_env = os.environ.copy()
try:
    gl = os.environ.get("EDGEFIRST_DISABLE_GL", "0")
    g2d = os.environ.get("EDGEFIRST_DISABLE_G2D", "0")
    cpu = os.environ.get("EDGEFIRST_DISABLE_CPU", "0")

    os.environ["EDGEFIRST_DISABLE_GL"] = "1"
    os.environ["EDGEFIRST_DISABLE_G2D"] = "1"
    os.environ["EDGEFIRST_DISABLE_CPU"] = cpu
    cpu_processor = ImageProcessor()

    os.environ["EDGEFIRST_DISABLE_GL"] = gl
    os.environ["EDGEFIRST_DISABLE_G2D"] = "1"
    os.environ["EDGEFIRST_DISABLE_CPU"] = "1"
    gl_processor = ImageProcessor()

    os.environ["EDGEFIRST_DISABLE_GL"] = "1"
    os.environ["EDGEFIRST_DISABLE_CPU"] = "1"
    os.environ["EDGEFIRST_DISABLE_G2D"] = g2d
    g2d_processor = ImageProcessor()

finally:
    os.environ.clear()
    os.environ.update(original_env)


@pytest.mark.benchmark(group="rgba_to_rgba", warmup_iterations=3)
def test_resize_cpu_rgba_to_rgba(benchmark):
    """Benchmark CPU RGBA to RGBA resize."""

    src = TensorImage.load("testdata/zidane.jpg", FourCC.RGBA)
    dst = TensorImage(*dst_size, FourCC.RGBA)

    benchmark(cpu_processor.convert, src, dst, Rotation.Rotate0, Flip.NoFlip)

    with dst.map() as d:
        arr_dst = np.frombuffer(d.view(), dtype=np.uint8).reshape(
            (dst.height, dst.width, 4)
        )
        expected = load_image("testdata/zidane.jpg", "RGBA", resize=dst_size)
        assert calculate_similarity_rms_u8(arr_dst, expected) > 0.98


@pytest.mark.benchmark(group="rgba_to_rgba", warmup_iterations=3)
def test_resize_cv2_rgba_to_rgba(benchmark):
    """Benchmark CV2 RGBA to RGBA resize."""
    cv2 = pytest.importorskip("cv2")

    src = TensorImage.load("testdata/zidane.jpg", FourCC.RGBA)
    dst_cv2 = TensorImage(*dst_size, FourCC.RGBA)
    dst = TensorImage(*dst_size, FourCC.RGBA)

    def resize(arr, size, dst):
        cv2.resize(arr, size, dst=dst)
        _ = dst[0, 0]

    with src.map() as m, dst_cv2.map() as d:
        arr = np.frombuffer(m.view(), dtype=np.uint8).reshape(
            (src.height, src.width, 4)
        )
        d = np.frombuffer(d.view(), dtype=np.uint8).reshape(
            (dst_cv2.height, dst_cv2.width, 4)
        )
        benchmark(resize, arr, dst_size, dst=d)

    cpu_processor.convert(src, dst, Rotation.Rotate0, Flip.NoFlip)

    with dst.map() as d, dst_cv2.map() as d_cv2:
        arr_dst = np.frombuffer(d.view(), dtype=np.uint8).reshape(
            (dst.height, dst.width, 4)
        )
        arr_cv2 = np.frombuffer(d_cv2.view(), dtype=np.uint8).reshape(
            (dst_cv2.height, dst_cv2.width, 4)
        )
        assert calculate_similarity_rms_u8(arr_dst, arr_cv2) > 0.98


@pytest.mark.benchmark(group="rgba_to_rgba", warmup_iterations=3)
def test_resize_gl_rgba_to_rgba(benchmark):
    """Benchmark GL RGBA to RGBA resize."""
    src = TensorImage.load("testdata/zidane.jpg", FourCC.RGBA)
    dst = TensorImage(*dst_size, FourCC.RGBA)

    try:
        gl_processor.convert(src, dst, Rotation.Rotate0, Flip.NoFlip)
    except RuntimeError:
        pytest.skip("OpenGL not available")

    benchmark(gl_processor.convert, src, dst, Rotation.Rotate0, Flip.NoFlip)

    with dst.map() as d:
        arr_dst = np.frombuffer(d.view(), dtype=np.uint8).reshape(
            (dst.height, dst.width, 4)
        )
        expected = load_image("testdata/zidane.jpg", "RGBA", resize=dst_size)
        assert calculate_similarity_rms_u8(arr_dst, expected) > 0.98


@pytest.mark.benchmark(group="rgba_to_rgba", warmup_iterations=3)
def test_resize_g2d_rgba_to_rgba(benchmark):
    """Benchmark G2D RGBA to RGBA resize."""

    src = TensorImage.load("testdata/zidane.jpg", FourCC.RGBA)
    dst = TensorImage(*dst_size, FourCC.RGBA)

    try:
        g2d_processor.convert(src, dst, Rotation.Rotate0, Flip.NoFlip)
    except RuntimeError:
        pytest.skip("G2D not available")

    benchmark(g2d_processor.convert, src, dst, Rotation.Rotate0, Flip.NoFlip)

    with dst.map() as d:
        arr_dst = np.frombuffer(d.view(), dtype=np.uint8).reshape(
            (dst.height, dst.width, 4)
        )
        expected = load_image("testdata/zidane.jpg", "RGBA", resize=dst_size)
        assert calculate_similarity_rms_u8(arr_dst, expected) > 0.98


@pytest.mark.benchmark(group="rgba_to_rgb", warmup_iterations=3)
def test_resize_cpu_rgba_to_rgb(benchmark):
    """Benchmark CPU RGBA to RGB resize."""

    src = TensorImage.load("testdata/zidane.jpg", FourCC.RGBA)
    dst = TensorImage(*dst_size, FourCC.RGB)

    benchmark(cpu_processor.convert, src, dst, Rotation.Rotate0, Flip.NoFlip)
    with dst.map() as d:
        arr_dst = np.frombuffer(d.view(), dtype=np.uint8).reshape(
            (dst.height, dst.width, 3)
        )
        expected = load_image("testdata/zidane.jpg", "RGB", resize=dst_size)
        assert calculate_similarity_rms_u8(arr_dst, expected) > 0.98


@pytest.mark.benchmark(group="rgba_to_rgb", warmup_iterations=3)
def test_resize_cv2_rgba_to_rgb(benchmark):
    """Benchmark CV2 RGBA to RGB resize."""
    cv2 = pytest.importorskip("cv2")

    src = TensorImage.load("testdata/zidane.jpg", FourCC.RGBA)
    dst_cv2 = TensorImage(*dst_size, FourCC.RGB)
    dst = TensorImage(*dst_size, FourCC.RGB)

    tmp = np.zeros((src.height, src.width, 3), dtype=np.uint8)

    def resize(arr, size, dst):
        cv2.cvtColor(arr, cv2.COLOR_RGBA2RGB, dst=tmp)
        cv2.resize(tmp, size, dst=dst)
        _ = dst[0, 0]

    with src.map() as m, dst_cv2.map() as d:
        arr = np.frombuffer(m.view(), dtype=np.uint8).reshape(
            (src.height, src.width, 4)
        )
        d = np.frombuffer(d.view(), dtype=np.uint8).reshape(
            (dst_cv2.height, dst_cv2.width, 3)
        )
        benchmark(resize, arr, dst_size, dst=d)

    cpu_processor.convert(src, dst, Rotation.Rotate0, Flip.NoFlip)

    with dst.map() as d, dst_cv2.map() as d_cv2:
        arr_dst = np.frombuffer(d.view(), dtype=np.uint8).reshape(
            (dst.height, dst.width, 3)
        )
        arr_cv2 = np.frombuffer(d_cv2.view(), dtype=np.uint8).reshape(
            (dst_cv2.height, dst_cv2.width, 3)
        )
        assert calculate_similarity_rms_u8(arr_dst, arr_cv2) > 0.98


@pytest.mark.benchmark(group="rgba_to_rgb", warmup_iterations=3)
def test_resize_g2d_rgba_to_rgb(benchmark):
    """Benchmark G2D RGBA to RGB resize."""
    src = TensorImage.load("testdata/zidane.jpg", FourCC.RGBA)
    dst = TensorImage(*dst_size, FourCC.RGB)

    try:
        g2d_processor.convert(src, dst, Rotation.Rotate0, Flip.NoFlip)
    except RuntimeError:
        pytest.skip("G2D not available")

    benchmark(g2d_processor.convert, src, dst, Rotation.Rotate0, Flip.NoFlip)
    with dst.map() as d:
        arr_dst = np.frombuffer(d.view(), dtype=np.uint8).reshape(
            (dst.height, dst.width, 3)
        )
        expected = load_image("testdata/zidane.jpg", "RGB", resize=dst_size)
        assert calculate_similarity_rms_u8(arr_dst, expected) > 0.98


@pytest.mark.benchmark(group="yuyv_to_rgba", warmup_iterations=3)
def test_resize_cpu_yuyv_to_rgba(benchmark):
    """Benchmark CPU YUYV to RGBA resize."""
    with open("testdata/camera720p.yuyv", "rb") as f:
        data = f.read()

    with open("testdata/camera720p.rgba", "rb") as f:
        expected_bytes = f.read()

    src = TensorImage(1280, 720, FourCC("YUYV"))
    src_array = np.frombuffer(data, dtype=np.uint8)
    with src.map() as m:
        np.frombuffer(m.view(), dtype=np.uint8)[:] = src_array

    dst = TensorImage(*dst_size, FourCC.RGBA)

    benchmark(cpu_processor.convert, src, dst, Rotation.Rotate0, Flip.NoFlip)
    with dst.map() as d:
        arr_dst = np.frombuffer(d.view(), dtype=np.uint8).reshape(
            (dst.height, dst.width, 4)
        )
        expected = load_bytes_to_image(expected_bytes, "RGBA", (1280, 720), dst_size)
        assert calculate_similarity_rms_u8(arr_dst, expected) > 0.98


@pytest.mark.benchmark(group="yuyv_to_rgba", warmup_iterations=3)
def test_resize_cv2_yuyv_to_rgba(benchmark):
    """Benchmark CV2 YUYV to RGBA resize."""
    cv2 = pytest.importorskip("cv2")

    with open("testdata/camera720p.yuyv", "rb") as f:
        data = f.read()

    src = TensorImage(1280, 720, FourCC("YUYV"))
    src_array = np.frombuffer(data, dtype=np.uint8)
    with src.map() as m:
        np.frombuffer(m.view(), dtype=np.uint8)[:] = src_array

    dst_cv2 = TensorImage(*dst_size, FourCC.RGBA)
    dst = TensorImage(*dst_size, FourCC.RGBA)

    tmp = np.zeros((src.height, src.width, 4), dtype=np.uint8)

    def resize(arr, size, dst):
        cv2.cvtColor(arr, cv2.COLOR_YUV2RGBA_YUY2, dst=tmp)
        cv2.resize(tmp, size, dst=dst)
        _ = dst[0, 0]

    with src.map() as m, dst_cv2.map() as d:
        arr = np.frombuffer(m.view(), dtype=np.uint8).reshape(
            (src.height, src.width, 2)
        )
        d = np.frombuffer(d.view(), dtype=np.uint8).reshape(
            (dst_cv2.height, dst_cv2.width, 4)
        )
        benchmark(resize, arr, dst_size, dst=d)

    cpu_processor.convert(src, dst, Rotation.Rotate0, Flip.NoFlip)

    with dst.map() as d, dst_cv2.map() as d_cv2:
        arr_dst = np.frombuffer(d.view(), dtype=np.uint8).reshape(
            (dst.height, dst.width, 4)
        )
        arr_cv2 = np.frombuffer(d_cv2.view(), dtype=np.uint8).reshape(
            (dst_cv2.height, dst_cv2.width, 4)
        )
        assert calculate_similarity_rms_u8(arr_dst, arr_cv2) > 0.98


@pytest.mark.benchmark(group="yuyv_to_rgba", warmup_iterations=3)
def test_resize_gl_yuyv_to_rgba(benchmark):
    """Benchmark GL YUYV to RGBA resize."""
    with open("testdata/camera720p.yuyv", "rb") as f:
        data = f.read()

    with open("testdata/camera720p.rgba", "rb") as f:
        expected_bytes = f.read()

    src = TensorImage(1280, 720, FourCC("YUYV"))
    src_array = np.frombuffer(data, dtype=np.uint8)
    with src.map() as m:
        np.frombuffer(m.view(), dtype=np.uint8)[:] = src_array

    dst = TensorImage(*dst_size, FourCC.RGBA)

    try:
        gl_processor.convert(src, dst, Rotation.Rotate0, Flip.NoFlip)
    except RuntimeError:
        pytest.skip("OpenGL not available")

    benchmark(gl_processor.convert, src, dst, Rotation.Rotate0, Flip.NoFlip)
    with dst.map() as d:
        arr_dst = np.frombuffer(d.view(), dtype=np.uint8).reshape(
            (dst.height, dst.width, 4)
        )
        expected = load_bytes_to_image(expected_bytes, "RGBA", (1280, 720), dst_size)
        assert calculate_similarity_rms_u8(arr_dst, expected) > 0.98


@pytest.mark.benchmark(group="yuyv_to_rgba", warmup_iterations=3)
def test_resize_g2d_yuyv_to_rgba(benchmark):
    """Benchmark G2D YUYV to RGBA resize."""
    with open("testdata/camera720p.yuyv", "rb") as f:
        data = f.read()

    with open("testdata/camera720p.rgba", "rb") as f:
        expected_bytes = f.read()

    src = TensorImage(1280, 720, FourCC("YUYV"))
    src_array = np.frombuffer(data, dtype=np.uint8)
    with src.map() as m:
        np.frombuffer(m.view(), dtype=np.uint8)[:] = src_array

    dst = TensorImage(*dst_size, FourCC.RGBA)

    try:
        g2d_processor.convert(src, dst, Rotation.Rotate0, Flip.NoFlip)
    except RuntimeError:
        pytest.skip("G2D not available")

    benchmark(g2d_processor.convert, src, dst, Rotation.Rotate0, Flip.NoFlip)
    with dst.map() as d:
        arr_dst = np.frombuffer(d.view(), dtype=np.uint8).reshape(
            (dst.height, dst.width, 4)
        )
        expected = load_bytes_to_image(expected_bytes, "RGBA", (1280, 720), dst_size)
        assert calculate_similarity_rms_u8(arr_dst, expected) > 0.98


@pytest.mark.benchmark(group="yuyv_to_rgb", warmup_iterations=3)
def test_resize_cpu_yuyv_to_rgb(benchmark):
    """Benchmark CPU YUYV to RGB resize."""
    with open("testdata/camera720p.yuyv", "rb") as f:
        data = f.read()

    with open("testdata/camera720p.rgb", "rb") as f:
        expected_bytes = f.read()

    src = TensorImage(1280, 720, FourCC("YUYV"))
    src_array = np.frombuffer(data, dtype=np.uint8)
    with src.map() as m:
        np.frombuffer(m.view(), dtype=np.uint8)[:] = src_array

    dst = TensorImage(*dst_size, FourCC.RGB)

    benchmark(cpu_processor.convert, src, dst, Rotation.Rotate0, Flip.NoFlip)
    with dst.map() as d:
        arr_dst = np.frombuffer(d.view(), dtype=np.uint8).reshape(
            (dst.height, dst.width, 3)
        )
        expected = load_bytes_to_image(expected_bytes, "RGB", (1280, 720), dst_size)
        assert calculate_similarity_rms_u8(arr_dst, expected) > 0.98


@pytest.mark.benchmark(group="yuyv_to_rgb", warmup_iterations=3)
def test_resize_cv2_yuyv_to_rgb(benchmark):
    """Benchmark CV2 YUYV to RGB resize."""
    cv2 = pytest.importorskip("cv2")

    with open("testdata/camera720p.yuyv", "rb") as f:
        data = f.read()

    src = TensorImage(1280, 720, FourCC("YUYV"))
    src_array = np.frombuffer(data, dtype=np.uint8)
    with src.map() as m:
        np.frombuffer(m.view(), dtype=np.uint8)[:] = src_array

    dst_cv2 = TensorImage(*dst_size, FourCC.RGB)
    dst = TensorImage(*dst_size, FourCC.RGB)

    tmp = np.zeros((src.height, src.width, 3), dtype=np.uint8)

    def resize(arr, size, dst):
        cv2.cvtColor(arr, cv2.COLOR_YUV2RGB_YUY2, dst=tmp)
        cv2.resize(tmp, size, dst=dst)
        _ = dst[0, 0]

    with src.map() as m, dst_cv2.map() as d:
        arr = np.frombuffer(m.view(), dtype=np.uint8).reshape(
            (src.height, src.width, 2)
        )
        d = np.frombuffer(d.view(), dtype=np.uint8).reshape(
            (dst_cv2.height, dst_cv2.width, 3)
        )
        benchmark(resize, arr, dst_size, dst=d)

    cpu_processor.convert(src, dst, Rotation.Rotate0, Flip.NoFlip)

    with dst.map() as d, dst_cv2.map() as d_cv2:
        arr_dst = np.frombuffer(d.view(), dtype=np.uint8).reshape(
            (dst.height, dst.width, 3)
        )
        arr_cv2 = np.frombuffer(d_cv2.view(), dtype=np.uint8).reshape(
            (dst_cv2.height, dst_cv2.width, 3)
        )
        assert calculate_similarity_rms_u8(arr_dst, arr_cv2) > 0.98


@pytest.mark.benchmark(group="yuyv_to_rgb", warmup_iterations=3)
def test_resize_g2d_yuyv_to_rgb(benchmark):
    """Benchmark G2D YUYV to RGB resize."""
    with open("testdata/camera720p.yuyv", "rb") as f:
        data = f.read()

    with open("testdata/camera720p.rgb", "rb") as f:
        expected_bytes = f.read()

    src = TensorImage(1280, 720, FourCC("YUYV"))
    src_array = np.frombuffer(data, dtype=np.uint8)
    with src.map() as m:
        np.frombuffer(m.view(), dtype=np.uint8)[:] = src_array

    dst = TensorImage(*dst_size, FourCC.RGB)

    try:
        g2d_processor.convert(src, dst, Rotation.Rotate0, Flip.NoFlip)
    except RuntimeError:
        pytest.skip("G2D not available")

    benchmark(g2d_processor.convert, src, dst, Rotation.Rotate0, Flip.NoFlip)
    with dst.map() as d:
        arr_dst = np.frombuffer(d.view(), dtype=np.uint8).reshape(
            (dst.height, dst.width, 3)
        )
        expected = load_bytes_to_image(expected_bytes, "RGB", (1280, 720), dst_size)
        assert calculate_similarity_rms_u8(arr_dst, expected) > 0.98


def test_load_jpeg_person(benchmark):
    """Benchmark JPEG loading for person.jpg."""
    with open("testdata/person.jpg", "rb") as f:
        data = f.read()

    benchmark(TensorImage.load_from_bytes, data, FourCC.RGBA)


def test_load_jpeg_zidane(benchmark):
    """Benchmark JPEG loading for zidane.jpg."""
    with open("testdata/zidane.jpg", "rb") as f:
        data = f.read()
    benchmark(TensorImage.load_from_bytes, data, FourCC.RGBA)


@pytest.mark.benchmark(group="rotate_90", warmup_iterations=3)
def test_cpu_rotate_90(benchmark):
    """Benchmark CPU 90 degree rotation."""
    with open("testdata/zidane.jpg", "rb") as f:
        data = f.read()

    src = TensorImage.load_from_bytes(data, FourCC.RGBA)
    dst = TensorImage(src.height, src.width, FourCC.RGBA)

    benchmark(cpu_processor.convert, src, dst, Rotation.Clockwise90, Flip.NoFlip)
    expected = load_image(
        "testdata/zidane.jpg",
        "RGBA",
        resize=(src.width, src.height),
        rotate=Image.ROTATE_270,
    )
    with dst.map() as d:
        arr_dst = np.frombuffer(d.view(), dtype=np.uint8).reshape(
            (dst.height, dst.width, 4)
        )
        assert calculate_similarity_rms_u8(arr_dst, expected) > 0.98


@pytest.mark.benchmark(group="rotate_90", warmup_iterations=3)
def test_cv2_rotate_90(benchmark):
    """Benchmark OpenCV 90 degree rotation."""
    cv2 = pytest.importorskip("cv2")

    src = TensorImage.load("testdata/zidane.jpg", FourCC.RGBA)
    dst = TensorImage(src.height, src.width, FourCC.RGBA)

    with src.map() as m, dst.map() as d:
        arr_src = np.frombuffer(m.view(), dtype=np.uint8).reshape(
            (src.height, src.width, 4)
        )
        arr_dst = np.frombuffer(d.view(), dtype=np.uint8).reshape(
            (dst.height, dst.width, 4)
        )

        def rotate_90(arr_src, arr_dst):
            cv2.rotate(arr_src, cv2.ROTATE_90_CLOCKWISE, dst=arr_dst)
            _ = arr_dst[0, 0]

        benchmark(rotate_90, arr_src, arr_dst)
    expected = load_image(
        "testdata/zidane.jpg",
        "RGBA",
        resize=(src.width, src.height),
        rotate=Image.ROTATE_270,
    )
    with dst.map() as d:
        arr_dst = np.frombuffer(d.view(), dtype=np.uint8).reshape(
            (dst.height, dst.width, 4)
        )
        assert calculate_similarity_rms_u8(arr_dst, expected) > 0.98


@pytest.mark.benchmark(group="rotate_90", warmup_iterations=3)
def test_gl_rotate_90(benchmark):
    """Benchmark CPU 90 degree rotation."""
    with open("testdata/zidane.jpg", "rb") as f:
        data = f.read()

    src = TensorImage.load_from_bytes(data, FourCC.RGBA)
    dst = TensorImage(src.height, src.width, FourCC.RGBA)

    try:
        gl_processor.convert(src, dst, Rotation.Clockwise90, Flip.NoFlip)
    except RuntimeError:
        pytest.skip("OpenGL not available")

    benchmark(gl_processor.convert, src, dst, Rotation.Clockwise90, Flip.NoFlip)
    expected = load_image(
        "testdata/zidane.jpg",
        "RGBA",
        resize=(src.width, src.height),
        rotate=Image.ROTATE_270,
    )
    with dst.map() as d:
        arr_dst = np.frombuffer(d.view(), dtype=np.uint8).reshape(
            (dst.height, dst.width, 4)
        )
        assert calculate_similarity_rms_u8(arr_dst, expected) > 0.98


@pytest.mark.benchmark(group="rotate_90", warmup_iterations=3)
def test_g2d_rotate_90(benchmark):
    """Benchmark CPU 90 degree rotation."""
    with open("testdata/zidane.jpg", "rb") as f:
        data = f.read()

    src = TensorImage.load_from_bytes(data, FourCC.RGBA)
    dst = TensorImage(src.height, src.width, FourCC.RGBA)

    try:
        g2d_processor.convert(src, dst, Rotation.Clockwise90, Flip.NoFlip)
    except RuntimeError:
        pytest.skip("G2D not available")

    benchmark(g2d_processor.convert, src, dst, Rotation.Clockwise90, Flip.NoFlip)
    expected = load_image(
        "testdata/zidane.jpg",
        "RGBA",
        resize=(src.width, src.height),
        rotate=Image.ROTATE_270,
    )
    with dst.map() as d:
        arr_dst = np.frombuffer(d.view(), dtype=np.uint8).reshape(
            (dst.height, dst.width, 4)
        )
        assert calculate_similarity_rms_u8(arr_dst, expected) > 0.98


@pytest.mark.benchmark(group="rotate_180", warmup_iterations=3)
def test_cpu_rotate_180(benchmark):
    """Benchmark CPU 180 degree rotation."""
    with open("testdata/zidane.jpg", "rb") as f:
        data = f.read()

    src = TensorImage.load_from_bytes(data, FourCC.RGBA)
    dst = TensorImage(src.width, src.height, FourCC.RGBA)

    benchmark(cpu_processor.convert, src, dst, Rotation.Rotate180, Flip.NoFlip)

    expected = load_image(
        "testdata/zidane.jpg",
        "RGBA",
        resize=(src.width, src.height),
        rotate=Image.ROTATE_180,
    )
    with dst.map() as d:
        arr_dst = np.frombuffer(d.view(), dtype=np.uint8).reshape(
            (dst.height, dst.width, 4)
        )
        assert calculate_similarity_rms_u8(arr_dst, expected) > 0.98


@pytest.mark.benchmark(group="rotate_180", warmup_iterations=3)
def test_cv2_rotate_180(benchmark):
    """Benchmark OpenCV 180 degree rotation."""
    cv2 = pytest.importorskip("cv2")

    src = TensorImage.load("testdata/zidane.jpg", FourCC.RGBA)
    dst = TensorImage(src.width, src.height, FourCC.RGBA)

    with src.map() as m, dst.map() as d:
        arr_src = np.frombuffer(m.view(), dtype=np.uint8).reshape(
            (src.height, src.width, 4)
        )
        arr_dst = np.frombuffer(d.view(), dtype=np.uint8).reshape(
            (dst.height, dst.width, 4)
        )

        def rotate_180(arr_src, arr_dst):
            cv2.rotate(arr_src, cv2.ROTATE_180, dst=arr_dst)
            _ = arr_dst[0, 0]

        benchmark(rotate_180, arr_src, arr_dst)
    expected = load_image(
        "testdata/zidane.jpg",
        "RGBA",
        resize=(src.width, src.height),
        rotate=Image.ROTATE_180,
    )
    with dst.map() as d:
        arr_dst = np.frombuffer(d.view(), dtype=np.uint8).reshape(
            (dst.height, dst.width, 4)
        )
        assert calculate_similarity_rms_u8(arr_dst, expected) > 0.98


@pytest.mark.benchmark(group="rotate_180", warmup_iterations=3)
def test_gl_rotate_180(benchmark):
    """Benchmark OpenGL 180 degree rotation."""
    with open("testdata/zidane.jpg", "rb") as f:
        data = f.read()

    src = TensorImage.load_from_bytes(data, FourCC.RGBA)
    dst = TensorImage(src.width, src.height, FourCC.RGBA)

    try:
        gl_processor.convert(src, dst, Rotation.Rotate180, Flip.NoFlip)
    except RuntimeError:
        pytest.skip("OpenGL not available")

    benchmark(gl_processor.convert, src, dst, Rotation.Rotate180, Flip.NoFlip)
    expected = load_image(
        "testdata/zidane.jpg",
        "RGBA",
        resize=(src.width, src.height),
        rotate=Image.ROTATE_180,
    )
    with dst.map() as d:
        arr_dst = np.frombuffer(d.view(), dtype=np.uint8).reshape(
            (dst.height, dst.width, 4)
        )
        assert calculate_similarity_rms_u8(arr_dst, expected) > 0.98


@pytest.mark.benchmark(group="rotate_180", warmup_iterations=3)
def test_g2d_rotate_180(benchmark):
    """Benchmark G2D 180 degree rotation."""

    src = TensorImage.load("testdata/zidane.jpg", FourCC.RGBA)
    dst = TensorImage(src.width, src.height, FourCC.RGBA)

    try:
        g2d_processor.convert(src, dst, Rotation.Rotate180, Flip.NoFlip)
    except RuntimeError:
        pytest.skip("OpenGL not available")

    benchmark(g2d_processor.convert, src, dst, Rotation.Rotate180, Flip.NoFlip)
    expected = load_image(
        "testdata/zidane.jpg",
        "RGBA",
        resize=(src.width, src.height),
        rotate=Image.ROTATE_180,
    )
    with dst.map() as d:
        arr_dst = np.frombuffer(d.view(), dtype=np.uint8).reshape(
            (dst.height, dst.width, 4)
        )
        assert calculate_similarity_rms_u8(arr_dst, expected) > 0.98
