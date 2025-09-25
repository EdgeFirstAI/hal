import edgefirst_python
src = edgefirst_python.TensorImage.load(
    "../../testdata/zidane.jpg", edgefirst_python.FourCC.RGBA)
dst = edgefirst_python.TensorImage(1280, 720)
converter = edgefirst_python.ImageConverter()
converter.convert(src, dst, edgefirst_python.Rotation.Rotate0,
                  edgefirst_python.Rect(0, 0, 1280, 720))

n = dst.to_numpy()
