# SPDX-FileCopyrightText: Copyright 2025 Au-Zone Technologies
# SPDX-License-Identifier: Apache-2.0


import edgefirst_hal
import numpy as np
import pytest


def test_tracker():
    output0 = np.fromfile("testdata/modelpack_split_17x30x18.bin", dtype=np.uint8).reshape(
        1, 17, 30, 18
    )
    output1 = np.fromfile("testdata/modelpack_split_9x15x18.bin", dtype=np.uint8).reshape(
        1, 9, 15, 18
    )

    config = open("testdata/modelpack_split.yaml").read()
    # default settings for tracker, but with 0.96 score threshold
    tracker = edgefirst_hal.ByteTrack(high_conf=0.96)
    decoder = edgefirst_hal.Decoder.new_from_yaml_str(
        config, 0.45, 0.45)
    boxes, scores, classes, masks, tracks = decoder.decode_tracked(tracker, 1000_000_000, [output0, output1])
    tracks_from_tracker = tracker.get_active_tracks()
    tracks_from_tracker = [track.info for track in tracks_from_tracker]
    assert len(tracks) > 0
    assert tracks == tracks_from_tracker


def test_external_tracker_update():
    output0 = np.fromfile("testdata/modelpack_split_17x30x18.bin", dtype=np.uint8).reshape(
        1, 17, 30, 18
    )
    output1 = np.fromfile("testdata/modelpack_split_9x15x18.bin", dtype=np.uint8).reshape(
        1, 9, 15, 18
    )

    config = open("testdata/modelpack_split.yaml").read()
    # default settings for tracker, but with 0.96 score threshold
    tracker = edgefirst_hal.ByteTrack(high_conf=0.96)
    decoder = edgefirst_hal.Decoder.new_from_yaml_str(
        config, 0.45, 0.45)
    _, _, _, _, tracks = decoder.decode_tracked(
        tracker, 1000_000_000,
        [output0, output1])

    tracks = tracker.update(
        np.array([[0.0, 0.0, 1.0, 1.0]], dtype=np.float32),
        np.array([0.0], dtype=np.float32),
        np.array([0], dtype=np.uintp),
        timestamp_ns=2000_000_000,
    )

    assert tracks == [None]


def test_yolo_seg():
    output0 = np.fromfile("testdata/yolov8_boxes_116x8400.bin", dtype=np.uint8).reshape(
        1, 116, 8400
    )
    output1 = np.fromfile("testdata/yolov8_protos_160x160x32.bin", dtype=np.uint8).reshape(
        1, 160, 160, 32
    )
    config = open("testdata/yolov8_seg.yaml").read()
    tracker = edgefirst_hal.ByteTrack(high_conf=0.2)
    decoder = edgefirst_hal.Decoder.new_from_yaml_str(
        config, 0.45, 0.45)
    boxes, scores, classes, masks, tracks = decoder.decode_tracked(tracker, 100_000_000, [output0, output1])
    assert np.allclose(boxes, [[0.08125, 0.7125, 0.3, 0.825], [0.59375, 0.25, 0.9375, 0.725]], atol=1e-5)
    last_updates = [track.last_updated for track in tracks]
    assert last_updates == [100_000_000, 100_000_000]
    assert len(masks) == 2

    # clear boxes
    output0_cleared = np.zeros_like(output0)

    # confirm no boxes when decoded normally
    boxes, scores, classes, masks = decoder.decode([output0_cleared, output1])
    assert boxes.shape == (0, 4)
    assert scores.shape == (0,)
    assert classes.shape == (0,)
    assert masks == []

    boxes, scores, classes, masks, tracks = decoder.decode_tracked(tracker, 200_000_000, [output0_cleared, output1])

    # we should have two boxes from the tracker, but no masks since the tracker doesn't know how to update masks coefficients
    assert np.allclose(boxes, [[0.08125, 0.7125, 0.3, 0.825], [0.59375, 0.25, 0.9375, 0.725]], atol=1e-2)
    last_updates = [track.last_updated for track in tracks]
    assert last_updates == [100_000_000, 100_000_000]
    assert masks == []

    boxes, scores, classes, masks, tracks = decoder.decode_tracked(tracker, 300_000_000, [output0, output1])
    assert np.allclose(boxes, [[0.08125, 0.7125, 0.3, 0.825], [0.59375, 0.25, 0.9375, 0.725]], atol=1e-5)
    last_updates = [track.last_updated for track in tracks]
    assert last_updates == [300_000_000, 300_000_000]
    assert len(masks) == 2


def test_new_track_info():
    track_info = edgefirst_hal.TrackInfo(
        uuid="123e4567e89b12d3a456426614174000",
        tracked_location=(0.1, 0.2, 0.3, 0.4),
        count=1,
        created=100_000_000,
        last_updated=100_000_000,
    )
    assert track_info.uuid == "123e4567-e89b-12d3-a456-426614174000"
    assert np.allclose(track_info.tracked_location, [0.1, 0.2, 0.3, 0.4], atol=1e-5)
    assert track_info.count == 1
    assert track_info.created == 100_000_000
    assert track_info.last_updated == 100_000_000


def test_new_active_track_info():
    track_info = edgefirst_hal.TrackInfo(
        uuid="123e4567e89b12d3a456426614174000",
        tracked_location=(0.1, 0.2, 0.3, 0.4),
        count=1,
        created=100_000_000,
        last_updated=100_000_000,
    )
    active_track_info = edgefirst_hal.ActiveTrackInfo(
        track_info, (0.1, 0.2, 0.3, 0.4), 0.3, 2
    )
    assert active_track_info.info.uuid == "123e4567-e89b-12d3-a456-426614174000"
    assert np.allclose(active_track_info.last_box[0], [0.1, 0.2, 0.3, 0.4], atol=1e-5)
    assert np.isclose(active_track_info.last_box[1], 0.3, atol=1e-5)
    assert active_track_info.last_box[2] == 2
