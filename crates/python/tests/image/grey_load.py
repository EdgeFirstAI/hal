from edgefirst_python import TensorImage, ImageConverter, Rotation, Rect, Flip, FourCC
rgba = TensorImage.load(
    "../../testdata/grey.jpg", FourCC.RGBA)
rgba = rgba.to_numpy()

grey = TensorImage.load(
    "../../testdata/grey.jpg", FourCC.GREY)
grey = grey.to_numpy()


default = TensorImage.load(
    "../../testdata/grey.jpg")
default = default.to_numpy()
