from edgefirst_python import TensorImage, ImageConverter, Rotation, Rect, Flip, FourCC
src = TensorImage.load(
    "../../testdata/zidane.jpg", FourCC.RGBA)
dst = TensorImage(1280, 720)
converter = ImageConverter()
converter.convert(src, dst, Rotation.Rotate0, Flip.NoFlip,
                  Rect(0, 0, 1280, 720))

n = dst.to_numpy()
