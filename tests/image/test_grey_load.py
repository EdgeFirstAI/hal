# SPDX-FileCopyrightText: Copyright 2025 Au-Zone Technologies
# SPDX-License-Identifier: Apache-2.0

import numpy as np
from edgefirst_hal import TensorImage, ImageConverter, Rotation, Rect, Flip, FourCC
rgba_ = TensorImage.load(
    "testdata/grey.jpg", FourCC.RGBA)
rgba = np.zeros((rgba_.height, rgba_.width, 4), dtype=np.uint8)
rgba_.normalize_to_numpy(rgba)

grey_ = TensorImage.load(
    "testdata/grey.jpg", FourCC.GREY)
grey = np.zeros((grey_.height, grey_.width, 1), dtype=np.uint8)
grey_.normalize_to_numpy(grey)


default_ = TensorImage.load(
    "testdata/grey.jpg")
default = np.zeros((default_.height, default_.width, 3), dtype=np.uint8)
default_.normalize_to_numpy(default)
