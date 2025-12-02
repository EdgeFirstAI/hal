# SPDX-FileCopyrightText: Copyright 2025 Au-Zone Technologies
# SPDX-License-Identifier: Apache-2.0

import numpy as np
from edgefirst_hal import TensorImage, ImageConverter, FourCC, Normalization

src = TensorImage.load("testdata/zidane.jpg", FourCC.RGBA)
dst = TensorImage(640, 640, fourcc=FourCC.RGBA)
converter = ImageConverter()
converter.convert(src, dst)
n = np.zeros((640, 640, 3), dtype=np.int8)
dst.normalize_to_numpy(n, normalization=Normalization.SIGNED)
unique_vals = list(np.unique(n))

dst.normalize_to_numpy(n, normalization=Normalization.SIGNED, zero_point=127)
unique_vals0 = list(np.unique(n))
