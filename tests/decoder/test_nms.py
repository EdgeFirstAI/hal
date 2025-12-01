# SPDX-FileCopyrightText: Copyright 2025 Au-Zone Technologies
# SPDX-License-Identifier: Apache-2.0


import numpy as np
import edgefirst_hal
import tensorflow as tf

output0 = np.fromfile("testdata/modelpack_split_17x30x18.bin", dtype=np.uint8).reshape(
    1, 17, 30, 18
)
output1 = np.fromfile("testdata/modelpack_split_9x15x18.bin", dtype=np.uint8).reshape(
    1, 9, 15, 18
)

config = open("testdata/modelpack_split.yaml").read()
decoder = edgefirst_hal.Decoder.new_from_yaml_str(config, 0.0, 1.0)
boxes, scores, classes, masks = decoder.decode([output0, output1], 100000)

for iou in range(0, 100):
    for score in range(0, 100):
        iou_threshold = iou / 100
        score_threshold = score / 100
        indices = tf.image.non_max_suppression(
            boxes,
            scores,
            max_output_size=100000,
            iou_threshold=iou_threshold,
            score_threshold=score_threshold,
        )
        tf_boxes, tf_scores, tf_classes = (
            boxes[indices],
            scores[indices],
            classes[indices],
        )
        decoder.score_threshold = score_threshold
        decoder.iou_threshold = iou_threshold
        hal_boxes, hal_scores, hal_classes, _ = decoder.decode(
            [output0, output1], 100000
        )

        assert np.all(np.abs(hal_boxes - tf_boxes) < 1e-6)
        assert np.all(np.abs(hal_scores - tf_scores) < 1e-6)
        assert np.all(hal_classes == tf_classes)
