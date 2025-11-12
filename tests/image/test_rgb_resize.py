# SPDX-FileCopyrightText: Copyright 2025 Au-Zone Technologies
# SPDX-License-Identifier: Apache-2.0

import numpy as np
from edgefirst_hal import TensorImage, ImageConverter,  FourCC
src = TensorImage.load(
    "testdata/zidane.jpg", FourCC.RGB)
dst = TensorImage(640, 640)
converter = ImageConverter()
converter.convert(src, dst)
n = np.zeros((640, 640, 3), dtype=np.uint8)
dst.normalize_to_numpy(n)
