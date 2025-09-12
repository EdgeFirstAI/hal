import edgefirst_python
import numpy as np
boxes, scores, classes = edgefirst_python.Decoder.decode_i8(
    np.fromfile('../../testdata/yolov8s_80_classes.bin',
                dtype=np.int8).reshape(84, 8400),
    0.0040811873,
    -123,
    0.25,
    0.7,
    50,
    edgefirst_python.BBoxType.XYWH)
