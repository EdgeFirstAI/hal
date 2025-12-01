# SPDX-FileCopyrightText: Copyright 2025 Au-Zone Technologies
# SPDX-License-Identifier: Apache-2.0

import edgefirst_hal
import numpy as np
import yaml

output0 = np.fromfile("testdata/modelpack_split_17x30x18.bin", dtype=np.uint8).reshape(
    1, 17, 30, 18
)
output1 = np.fromfile("testdata/modelpack_split_9x15x18.bin", dtype=np.uint8).reshape(
    1, 9, 15, 18
)

config = open("testdata/modelpack_split.yaml").read()
config = yaml.safe_load(config)
decoder = edgefirst_hal.Decoder(config, 0.45, 0.45)
boxes, scores, classes, masks = decoder.decode([output0, output1])
