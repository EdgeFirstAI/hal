from edgefirst_hal import TensorImage, ImageProcessor, Flip, FourCC, Normalization, Rotation, Rect
import numpy as np
from PIL import Image
import math


def load_image(image, format="RGBA", resize=None):
    im = Image.open(image).convert(format)
    if resize is not None:
        im = im.resize(resize)
    return np.array(im)


def calculate_similarity_rms_u8(imageA, imageB) -> float:
    imgA = np.asarray(imageA)
    imgB = np.asarray(imageB)

    imgA = imgA.astype("float")/255.0
    imgB = imgB.astype("float")/255.0

    squared_diff = (imgA - imgB) ** 2

    rms = math.sqrt(float(np.mean(squared_diff)))
    return 1.0-rms


def test_flip():
    src = TensorImage.load("testdata/zidane.jpg", FourCC.RGBA)
    dst = TensorImage(1280, 720)
    converter = ImageProcessor()
    converter.convert(src, dst, flip=Flip.Horizontal)

    n = np.zeros((720, 1280, 3), dtype=np.uint8)
    dst.normalize_to_numpy(n)

    expected = load_image("testdata/zidane.jpg", "RGB")
    expected = expected[:, ::-1, :]

    assert calculate_similarity_rms_u8(n, expected) > 0.99


def test_grey_load():
    rgba_ = TensorImage.load("testdata/grey.jpg", FourCC.RGBA)
    rgba = np.zeros((rgba_.height, rgba_.width, 4), dtype=np.uint8)
    rgba_.normalize_to_numpy(rgba)

    grey_ = TensorImage.load("testdata/grey.jpg", FourCC.GREY)
    grey = np.zeros((grey_.height, grey_.width, 1), dtype=np.uint8)
    grey_.normalize_to_numpy(grey)

    default_ = TensorImage.load("testdata/grey.jpg")
    default = np.zeros((default_.height, default_.width, 3), dtype=np.uint8)
    default_.normalize_to_numpy(default)

    assert calculate_similarity_rms_u8(rgba[:, :, 0:3], default) > 0.99
    assert calculate_similarity_rms_u8(rgba[:, :, 0], grey[:, :, 0]) > 0.99
    assert calculate_similarity_rms_u8(default[:, :, 0], grey[:, :, 0]) > 0.99


def test_normalize():
    src = TensorImage.load("testdata/zidane.jpg", FourCC.RGBA)
    dst = TensorImage(640, 640, fourcc=FourCC.RGBA)
    converter = ImageProcessor()
    converter.convert(src, dst)
    n = np.zeros((640, 640, 3), dtype=np.int8)
    dst.normalize_to_numpy(n, normalization=Normalization.SIGNED)
    unique_vals = np.unique(n)
    assert (unique_vals == np.array(range(-128, 128))).all()

    dst.normalize_to_numpy(
        n, normalization=Normalization.SIGNED, zero_point=127)
    unique_vals0 = np.unique(n)
    assert (unique_vals0 == np.array(range(-127, 128))).all()


def test_render():
    dst = TensorImage.load("testdata/giraffe.jpg", FourCC.RGBA)
    seg = np.fromfile('testdata/yolov8_seg_crop_76x55.bin',
                      dtype=np.uint8).reshape((76, 55, 1))
    converter = ImageProcessor()
    converter.set_class_colors([[255, 255, 0, 233], [128, 128, 255, 100]])
    converter.render_to_image(dst,
                              bbox=np.array(
                                  [[0.59375, 0.25, 0.9375, 0.725]], dtype=np.float32),
                              scores=np.array([0.9], dtype=np.float32),
                              classes=np.array([1], dtype=np.uintp),
                              seg=[seg]
                              )
    expected_gl = load_image("testdata/output_render_gl.jpg", "RGBA")
    expected_cpu = load_image("testdata/output_render_cpu.jpg", "RGBA")
    dst.save_jpeg("output_render.jpeg", 95)
    with dst.map() as m:
        img = np.array(m.view()).reshape((dst.height, dst.width, 4))
        assert calculate_similarity_rms_u8(
            img, expected_gl) > 0.99 or calculate_similarity_rms_u8(img, expected_cpu) > 0.99


def test_rgb_resize():
    src = TensorImage.load("testdata/zidane.jpg", FourCC.RGB)
    dst = TensorImage(640, 640, FourCC.RGBA)
    converter = ImageProcessor()
    converter.convert(src, dst)
    with dst.map() as m:
        n = np.array(m.view()).reshape((dst.height, dst.width, 4))
        expected = load_image("testdata/zidane.jpg", "RGBA", resize=(640, 640))
        assert calculate_similarity_rms_u8(n, expected) > 0.99


def test_rgba_to_rgb():
    src = TensorImage.load("testdata/zidane.jpg", FourCC.RGBA)
    dst = TensorImage(1280, 720, FourCC.RGB)
    converter = ImageProcessor()
    converter.convert(src, dst, Rotation.Rotate0,
                      Flip.NoFlip, Rect(0, 0, 1280, 720))
    n = np.zeros((720, 1280, 3), dtype=np.uint8)

    with dst.map() as m:
        n = np.array(m.view()).reshape((dst.height, dst.width, 3))
        expected = load_image("testdata/zidane.jpg", "RGB")
        assert calculate_similarity_rms_u8(n, expected) > 0.99


def test_enum_cmp():
    dst = TensorImage(640, 640, fourcc=FourCC.RGBA)
    formats_equal = dst.format == FourCC.RGBA
