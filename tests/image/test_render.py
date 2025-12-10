# SPDX-FileCopyrightText: Copyright 2025 Au-Zone Technologies
# SPDX-License-Identifier: Apache-2.0

import numpy as np
from edgefirst_hal import TensorImage, ImageProcessor, FourCC

dst = TensorImage.load("testdata/giraffe.jpg", FourCC.RGBA)
seg = np.fromfile('testdata/yolov8_seg_crop_76x55.bin',
                  dtype=np.uint8).reshape((76, 55, 1))

converter = ImageProcessor()
converter.render_to_image(dst,
                          bbox=np.array(
                              [[0.59375, 0.25, 0.9375, 0.725]], dtype=np.float32),
                          scores=np.array([0.9], dtype=np.float32),
                          classes=np.array([0], dtype=np.uintp),
                          seg=[seg]
                          )
