import edgefirst_hal
import numpy as np

output0 = np.fromfile('testdata/modelpack_split_17x30x18.bin',
                      dtype=np.uint8).reshape(17, 30, 18)
anchors0 = [
    [0.13750000298023224, 0.2074074000120163],
    [0.2541666626930237, 0.21481481194496155],
    [0.23125000298023224, 0.35185185074806213],
]
quant0 = (0.09929127991199493, 183)

output1 = np.fromfile('testdata/modelpack_split_9x15x18.bin',
                      dtype=np.uint8).reshape(9, 15, 18)
anchors1 = [
    [0.36666667461395264, 0.31481480598449707],
    [0.38749998807907104, 0.4740740656852722],
    [0.5333333611488342, 0.644444465637207],
]
quant1 = (0.08547406643629074, 174)

boxes, scores, classes = edgefirst_hal.Decoder.decode_modelpack_det_split([
    output0, output1], [anchors0, anchors1], [quant0, quant1], 0.45, 0.45)
