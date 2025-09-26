from edgefirst_python import TensorImage, ImageConverter, Flip, FourCC
src = TensorImage.load(
    "../../testdata/zidane.jpg", FourCC.RGBA)
dst = TensorImage(1280, 720)
converter = ImageConverter()
converter.convert(src, dst, flip=Flip.Horizontal)

n = dst.to_numpy()
