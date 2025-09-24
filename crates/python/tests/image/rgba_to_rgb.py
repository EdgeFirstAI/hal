import edgefirst_python
src = edgefirst_python.TensorImage.load(
    "../../testdata/zidane.jpg", edgefirst_python.FourCC.RGB)
dst = edgefirst_python.TensorImage(640, 640)
converter = edgefirst_python.ImageConverter()
converter.convert(src, dst, edgefirst_python.Rotation.Rotate0,
                  edgefirst_python.Rect(0, 0, 640, 640))
dst.save_jpeg("test.jpeg")
