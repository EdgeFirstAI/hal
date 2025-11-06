import edgefirst_hal
import numpy as np

output0 = np.fromfile('testdata/modelpack_split_17x30x18.bin',
                      dtype=np.uint8).reshape(1, 17, 30, 18)
output1 = np.fromfile('testdata/modelpack_split_9x15x18.bin',
                      dtype=np.uint8).reshape(1, 9, 15, 18)
int32_arr = np.zeros((100, 100), np.int32)
config = open("testdata/modelpack_split.yaml").read()
decoder = edgefirst_hal.Decoder.new_from_yaml_str(config, 0.45, 0.45)
boxes, scores, classes, masks = decoder.decode([output0, output1, int32_arr])
