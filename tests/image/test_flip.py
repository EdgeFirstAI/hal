# SPDX-FileCopyrightText: Copyright 2025 Au-Zone Technologies
# SPDX-License-Identifier: Apache-2.0

from edgefirst_hal import TensorImage, ImageConverter, Flip, FourCC
import numpy as np

src = TensorImage.load("testdata/zidane.jpg", FourCC.RGBA)
dst = TensorImage(1280, 720)
converter = ImageConverter()
converter.convert(src, dst, flip=Flip.Horizontal)

n = np.zeros((720, 1280, 3), dtype=np.uint8)
dst.normalize_to_numpy(n)
