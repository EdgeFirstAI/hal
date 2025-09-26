from edgefirst_python import TensorImage, ImageConverter,  FourCC
src = TensorImage.load(
    "../../testdata/zidane.jpg", FourCC.RGB)
dst = TensorImage(640, 640)
converter = ImageConverter()
converter.convert(src, dst)
n = dst.to_numpy()
