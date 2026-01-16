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
tracker = edgefirst_hal.ByteTrack(
    high_conf=0.96)  # default settings for tracker, but with 0.96 score threshold
decoder = edgefirst_hal.Decoder(config, 0.45, 0.45, tracker=tracker)
boxes, scores, classes, masks, tracks = decoder.decode_tracked(
    [output0, output1], 1000_000_000)
tracks_from_tracker = tracker.get_active_tracks()
assert tracks == tracks_from_tracker

tracks = tracker.update(
    np.array([[0.0, 0.0, 1.0, 1.0]], dtype=np.float32),
    np.array([0.0], dtype=np.float32),
    np.array([0], dtype=np.uintp),
    timestamp_ns=2000_000_000,
)
assert tracks == [None]
