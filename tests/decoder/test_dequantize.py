# SPDX-FileCopyrightText: Copyright 2025 Au-Zone Technologies
# SPDX-License-Identifier: Apache-2.0

import edgefirst_hal
import numpy as np

output = np.empty((84, 8400), dtype=np.float32)
edgefirst_hal.Decoder.dequantize(
    np.fromfile("testdata/yolov8s_80_classes.bin", dtype=np.int8).reshape(84, 8400),
    (0.0040811873, -123),
    output,
)
