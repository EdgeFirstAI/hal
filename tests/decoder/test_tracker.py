# SPDX-FileCopyrightText: Copyright 2025 Au-Zone Technologies
# SPDX-License-Identifier: Apache-2.0


import edgefirst_hal
import numpy as np


def test_tracker():
    """
    Test basic tracking functionality.
    """
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
    """
    Test that external updates to the tracker correctly influence the track states.
    """
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
    """
    Test end-to-end tracked decoding with YOLO segmentation data. 
    Verifies that masks are not generated for boxes that originate from tracker predictions
    """
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
    """
    Test creating a new TrackInfo object and verify its fields are set correctly.
    """
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


def test_tracker_random_jitter():
    """
    Test tracking stability with random jitter on XY coordinates.
    """
    output = np.fromfile("testdata/yolov8s_80_classes.bin", dtype=np.int8).reshape(
        1, 84, 8400
    )
    config = """
decoder_version: yolov8
outputs:
 - type: detection
   decoder: ultralytics
   quantization: [0.0040811873, -123]
   shape: [1, 84, 8400]
   dshape:
    - [batch, 1]
    - [num_features, 84]
    - [num_boxes, 8400]
   normalized: true
"""

    decoder = edgefirst_hal.Decoder.new_from_yaml_str(config, 0.25, 0.1)
    tracker = edgefirst_hal.ByteTrack(update=0.1, high_conf=0.3)

    # First frame - establish tracks
    boxes, scores, classes, masks, tracks = decoder.decode_tracked(
        tracker, 0, [output]
    )
    assert len(boxes) == 2

    expected_box0 = [0.5285137, 0.05305544, 0.87541467, 0.9998909]
    expected_box1 = [0.130598, 0.43260583, 0.35098213, 0.9958097]
    assert np.allclose(boxes[0], expected_box0, atol=1e-2)
    assert np.allclose(boxes[1], expected_box1, atol=1e-2)

    last_boxes = boxes.copy()

    rng = np.random.default_rng(seed=0xABBEEF)

    for i in range(1, 101):
        out = output.copy()
        # introduce jitter into the XY coordinates
        x_jitter = rng.standard_normal(8400).astype(np.float32)
        out[0, 0, :] += np.astype(np.clip(x_jitter, -2.0, 2.0) / 2.0 * 1e-2 / 0.0040811873, np.int8)

        y_jitter = rng.standard_normal(8400).astype(np.float32)
        y_jitter = np.clip(y_jitter, -2.0, 2.0) / 2.0
        out[0, 1, :] += np.astype(np.clip(y_jitter, -2.0, 2.0) / 2.0 * 1e-2 / 0.0040811873, np.int8)

        timestamp = 100_000_000 * i // 3
        boxes, scores, classes, masks, tracks = decoder.decode_tracked(
            tracker, timestamp, [out]
        )
        assert len(boxes) == 2, f"Frame {i}: expected 2 boxes, got {len(boxes)}"
        assert np.allclose(boxes[0], expected_box0, atol=5e-3), \
            f"Frame {i}: box0 {boxes[0]} not close to expected {expected_box0}"
        assert np.allclose(boxes[1], expected_box1, atol=5e-3), \
            f"Frame {i}: box1 {boxes[1]} not close to expected {expected_box1}"

        # Check that boxes didn't jump too far from last frame
        assert np.allclose(boxes[0], last_boxes[0], atol=2e-3)
        assert np.allclose(boxes[1], last_boxes[1], atol=2e-3)
        last_boxes = boxes.copy()


def test_tracker_segdet_no_detection_prediction():
    """
    Test that tracker predicts forward when no detections are present in segdet.
    """
    output0 = np.fromfile("testdata/yolov8_boxes_116x8400.bin", dtype=np.uint8).reshape(
        1, 116, 8400
    )
    output1 = np.fromfile("testdata/yolov8_protos_160x160x32.bin", dtype=np.uint8).reshape(
        1, 160, 160, 32
    )
    config = open("testdata/yolov8_seg.yaml").read()
    tracker = edgefirst_hal.ByteTrack(update=0.1, high_conf=0.7)
    decoder = edgefirst_hal.Decoder.new_from_yaml_str(config, 0.45, 0.45)

    # First frame - establish tracks
    boxes, scores, classes, masks, tracks = decoder.decode_tracked(
        tracker, 0, [output0, output1]
    )
    assert len(boxes) == 2
    assert len(masks) == 2

    expected_box0 = [0.08515105, 0.7131401, 0.29802868, 0.8195788]
    expected_box1 = [0.59605736, 0.25545314, 0.93666154, 0.72378385]
    assert np.allclose(boxes[0], expected_box0, atol=1.0 / 160.0)
    assert np.allclose(boxes[1], expected_box1, atol=1.0 / 160.0)

    # Clear scores to indicate no detections.
    output0_cleared = output0.copy()
    output0_cleared[0, 4:84, :] = 0  # zero out scores

    boxes, scores, classes, masks, tracks = decoder.decode_tracked(
        tracker, 100_000_000 // 3, [output0_cleared, output1]
    )

    # Tracker should predict forward with the last known boxes
    assert len(boxes) == 2
    assert np.allclose(boxes[0], expected_box0, atol=1e-2)
    assert np.allclose(boxes[1], expected_box1, atol=1e-2)

    # No masks when boxes come from tracker prediction without matching detection
    assert masks == []


def test_tracker_linear_motion():
    """
    Test that tracker correctly follows linear motion and predicts forward.
    """
    output = np.fromfile("testdata/yolov8s_80_classes.bin", dtype=np.int8).reshape(
        1, 84, 8400
    )
    config = """
decoder_version: yolov8
outputs:
 - type: detection
   decoder: ultralytics
   quantization: [0.0040811873, -123]
   shape: [1, 84, 8400]
   dshape:
    - [batch, 1]
    - [num_features, 84]
    - [num_boxes, 8400]
   normalized: true
"""
    quant_scale = 0.0040811873

    decoder = edgefirst_hal.Decoder.new_from_yaml_str(config, 0.25, 0.1)
    tracker = edgefirst_hal.ByteTrack(update=0.1, high_conf=0.3)

    # First frame
    boxes, scores, classes, masks, tracks = decoder.decode_tracked(
        tracker, 0, [output]
    )
    assert len(boxes) == 2

    expected_box0 = [0.5285137, 0.05305544, 0.87541467, 0.9998909]
    expected_box1 = [0.130598, 0.43260583, 0.35098213, 0.9958097]
    assert np.allclose(boxes[0], expected_box0, atol=1e-2)
    assert np.allclose(boxes[1], expected_box1, atol=1e-2)

    # Apply linear motion for 100 frames
    for i in range(1, 101):
        out = output.copy()
        # Introduce linear movement into X coordinates
        out[0, 0, :] += int(round(i * 1e-3 / quant_scale))

        timestamp = 100_000_000 * i // 3
        boxes, scores, classes, masks, tracks = decoder.decode_tracked(
            tracker, timestamp, [out]
        )
        assert len(boxes) == 2, f"Frame {i}: expected 2 boxes, got {len(boxes)}"
    # After 100 frames of linear X motion (0.001 per frame = 0.1 total)
    expected_box0[0] += 0.1
    expected_box0[2] += 0.1
    expected_box1[0] += 0.1
    expected_box1[2] += 0.1

    active_tracks = tracker.get_active_tracks()
    assert len(active_tracks) == 2

    assert np.allclose(active_tracks[0].info.tracked_location, expected_box0, atol=1e-2)
    assert np.allclose(active_tracks[1].info.tracked_location, expected_box1, atol=1e-2)

    # Clear scores to simulate no detections
    output_cleared = output.copy()
    output_cleared[0, 4:, :] = -128  # minimum value for int8

    boxes, scores, classes, masks, tracks = decoder.decode_tracked(
        tracker, 100_000_000 * 101 // 3, [output_cleared]
    )
    # Tracker should still produce boxes from prediction
    assert len(boxes) == 2


def test_tracker_end_to_end_float_detection():
    """
    Test end-to-end tracked decoding with float detection data.
    """
    config = """
decoder_version: yolo26
outputs:
 - type: detection
   decoder: ultralytics
   quantization: [0.00784313725490196, 0]
   shape: [1, 10, 6]
   dshape:
    - [batch, 1]
    - [num_boxes, 10]
    - [num_features, 6]
   normalized: true
"""
    # Build a detection tensor: [1, 10, 6] with one valid detection
    detect = np.zeros((1, 10, 6), dtype=np.float32)
    detect[0, 0, 0] = 0.1234  # xmin
    detect[0, 0, 1] = 0.1234  # ymin
    detect[0, 0, 2] = 0.2345  # xmax
    detect[0, 0, 3] = 0.2345  # ymax
    detect[0, 0, 4] = 0.9876  # score
    detect[0, 0, 5] = 2.0     # class

    decoder = edgefirst_hal.Decoder.new_from_yaml_str(config, 0.45, 0.45)
    tracker = edgefirst_hal.ByteTrack(update=0.1, high_conf=0.7)

    # First frame with detection
    boxes, scores, classes, masks, tracks = decoder.decode_tracked(
        tracker, 0, [detect]
    )
    assert len(boxes) == 1
    assert np.allclose(boxes[0], [0.1234, 0.1234, 0.2345, 0.2345], atol=1e-3)
    assert np.isclose(scores[0], 0.9876, atol=1e-3)
    assert classes[0] == 2

    # Clear scores to simulate no detections
    detect_cleared = detect.copy()
    detect_cleared[0, :, 4] = 0.0  # zero all scores

    boxes, scores, classes, masks, tracks = decoder.decode_tracked(
        tracker, 100_000_000 // 3, [detect_cleared]
    )
    # Tracker should predict forward
    assert len(boxes) == 1
    assert np.allclose(boxes[0], [0.1234, 0.1234, 0.2345, 0.2345], atol=1e-3)


def test_tracker_end_to_end_segdet():
    """
    Test end-to-end tracked decoding with segdet data (quantized path).
    """
    config = """
decoder_version: yolo26
outputs:
 - type: detection
   decoder: ultralytics
   quantization: [0.00784313725490196, 0]
   shape: [1, 10, 38]
   dshape:
    - [batch, 1]
    - [num_boxes, 10]
    - [num_features, 38]
   normalized: true
 - type: protos
   decoder: ultralytics
   quantization: [0.0039215686274509803921568627451, 128]
   shape: [1, 160, 160, 32]
   dshape:
    - [batch, 1]
    - [height, 160]
    - [width, 160]
    - [num_protos, 32]
"""
    # Build detection tensor with one valid box
    # Quantization: scale=2.0/255.0, zp=0
    # For float values: boxes=[0.1234, 0.1234, 0.2345, 0.2345], score=0.9876, class=2
    # Quantized: val = round(float_val / scale)
    scale = 2.0 / 255.0
    detect = np.zeros((1, 10, 38), dtype=np.uint8)
    detect[0, 0, 0] = int(round(0.1234 / scale))  # xmin
    detect[0, 0, 1] = int(round(0.1234 / scale))  # ymin
    detect[0, 0, 2] = int(round(0.2345 / scale))  # xmax
    detect[0, 0, 3] = int(round(0.2345 / scale))  # ymax
    detect[0, 0, 4] = int(round(0.9876 / scale))  # score
    detect[0, 0, 5] = int(round(2.0 / scale))     # class

    protos = np.zeros((1, 160, 160, 32), dtype=np.uint8)

    decoder = edgefirst_hal.Decoder.new_from_yaml_str(config, 0.45, 0.45)
    tracker = edgefirst_hal.ByteTrack(update=0.1, high_conf=0.7)

    # First frame
    boxes, scores, classes, masks, tracks = decoder.decode_tracked(
        tracker, 0, [detect, protos]
    )
    assert len(boxes) == 1

    # Expected values after quantization roundtrip
    expected_xmin = int(round(0.1234 / scale)) * scale
    expected_ymin = int(round(0.1234 / scale)) * scale
    expected_xmax = int(round(0.2345 / scale)) * scale
    expected_ymax = int(round(0.2345 / scale)) * scale
    assert np.allclose(boxes[0], [expected_xmin, expected_ymin, expected_xmax, expected_ymax],
                       atol=1.0 / 160.0)

    # Second frame with no detections
    detect_cleared = detect.copy()
    detect_cleared[0, :, 4] = 0

    boxes, scores, classes, masks, tracks = decoder.decode_tracked(
        tracker, 100_000_000 // 3, [detect_cleared, protos]
    )
    assert len(boxes) == 1
    # No masks from tracker prediction
    assert masks == []


def test_tracker_end_to_end_segdet_float():
    """
    Test end-to-end tracked decoding with segdet float data.
    """
    config = """
decoder_version: yolo26
outputs:
 - type: detection
   decoder: ultralytics
   quantization: [0.00784313725490196, 0]
   shape: [1, 10, 38]
   dshape:
    - [batch, 1]
    - [num_boxes, 10]
    - [num_features, 38]
   normalized: true
 - type: protos
   decoder: ultralytics
   quantization: [0.0039215686274509803921568627451, 128]
   shape: [1, 160, 160, 32]
   dshape:
    - [batch, 1]
    - [height, 160]
    - [width, 160]
    - [num_protos, 32]
"""
    detect = np.zeros((1, 10, 38), dtype=np.float32)
    detect[0, 0, 0] = 0.1234
    detect[0, 0, 1] = 0.1234
    detect[0, 0, 2] = 0.2345
    detect[0, 0, 3] = 0.2345
    detect[0, 0, 4] = 0.9876
    detect[0, 0, 5] = 2.0

    protos = np.zeros((1, 160, 160, 32), dtype=np.float32)

    decoder = edgefirst_hal.Decoder.new_from_yaml_str(config, 0.45, 0.45)
    tracker = edgefirst_hal.ByteTrack(update=0.1, high_conf=0.7)

    boxes, scores, classes, masks, tracks = decoder.decode_tracked(
        tracker, 0,
        [detect, protos]
    )
    assert len(boxes) == 1
    assert np.allclose(boxes[0], [0.1234, 0.1234, 0.2345, 0.2345], atol=1.0 / 160.0)

    # Clear scores to simulate no detections
    detect_cleared = detect.copy()
    detect_cleared[0, :, 4] = 0.0

    boxes, scores, classes, masks, tracks = decoder.decode_tracked(
        tracker, 100_000_000 // 3,
        [detect_cleared, protos]
    )
    assert len(boxes) == 1
    assert np.allclose(boxes[0], [0.1234, 0.1234, 0.2345, 0.2345], atol=1e-3)
    assert masks == []


def test_tracker_end_to_end_split_segdet():
    """
    Test end-to-end tracked decoding with split segdet outputs.
    """
    config = """
decoder_version: yolo26
outputs:
 - type: boxes
   decoder: ultralytics
   quantization: [0.00784313725490196, 0]
   shape: [1, 10, 4]
   dshape:
    - [batch, 1]
    - [num_boxes, 10]
    - [box_coords, 4]
   normalized: true
 - type: scores
   decoder: ultralytics
   quantization: [0.00784313725490196, 0]
   shape: [1, 10, 1]
   dshape:
    - [batch, 1]
    - [num_boxes, 10]
    - [num_classes, 1]
 - type: classes
   decoder: ultralytics
   quantization: [0.00784313725490196, 0]
   shape: [1, 10, 1]
   dshape:
    - [batch, 1]
    - [num_boxes, 10]
    - [num_classes, 1]
 - type: mask_coefficients
   decoder: ultralytics
   quantization: [0.00784313725490196, 0]
   shape: [1, 10, 32]
   dshape:
    - [batch, 1]
    - [num_boxes, 10]
    - [num_protos, 32]
 - type: protos
   decoder: ultralytics
   quantization: [0.0039215686274509803921568627451, 128]
   shape: [1, 160, 160, 32]
   dshape:
    - [batch, 1]
    - [height, 160]
    - [width, 160]
    - [num_protos, 32]
"""
    scale = 2.0 / 255.0

    boxes_arr = np.zeros((1, 10, 4), dtype=np.uint8)
    boxes_arr[0, 0, 0] = int(round(0.1234 / scale))
    boxes_arr[0, 0, 1] = int(round(0.1234 / scale))
    boxes_arr[0, 0, 2] = int(round(0.2345 / scale))
    boxes_arr[0, 0, 3] = int(round(0.2345 / scale))

    scores_arr = np.zeros((1, 10, 1), dtype=np.uint8)
    scores_arr[0, 0, 0] = int(round(0.9876 / scale))

    classes_arr = np.zeros((1, 10, 1), dtype=np.uint8)
    classes_arr[0, 0, 0] = int(round(2.0 / scale))

    mask_arr = np.zeros((1, 10, 32), dtype=np.uint8)
    protos_arr = np.zeros((1, 160, 160, 32), dtype=np.uint8)

    decoder = edgefirst_hal.Decoder.new_from_yaml_str(config, 0.45, 0.45)
    tracker = edgefirst_hal.ByteTrack(update=0.1, high_conf=0.7)

    boxes, scores, classes, masks, tracks = decoder.decode_tracked(
        tracker, 0,
        [boxes_arr, scores_arr, classes_arr, mask_arr, protos_arr]
    )
    assert len(boxes) == 1

    expected_xmin = int(round(0.1234 / scale)) * scale
    expected_ymin = int(round(0.1234 / scale)) * scale
    expected_xmax = int(round(0.2345 / scale)) * scale
    expected_ymax = int(round(0.2345 / scale)) * scale
    assert np.allclose(boxes[0], [expected_xmin, expected_ymin, expected_xmax, expected_ymax],
                       atol=1.0 / 160.0)

    # Clear scores to simulate no detections
    scores_cleared = np.zeros_like(scores_arr)

    boxes, scores, classes, masks, tracks = decoder.decode_tracked(
        tracker, 100_000_000 // 3,
        [boxes_arr, scores_cleared, classes_arr, mask_arr, protos_arr]
    )
    assert len(boxes) == 1
    assert masks == []


def test_tracker_end_to_end_split_segdet_float():
    """
    Test end-to-end tracked decoding with split segdet float outputs.
    """
    config = """
decoder_version: yolo26
outputs:
 - type: boxes
   decoder: ultralytics
   quantization: [0.00784313725490196, 0]
   shape: [1, 10, 4]
   dshape:
    - [batch, 1]
    - [num_boxes, 10]
    - [box_coords, 4]
   normalized: true
 - type: scores
   decoder: ultralytics
   quantization: [0.00784313725490196, 0]
   shape: [1, 10, 1]
   dshape:
    - [batch, 1]
    - [num_boxes, 10]
    - [num_classes, 1]
 - type: classes
   decoder: ultralytics
   quantization: [0.00784313725490196, 0]
   shape: [1, 10, 1]
   dshape:
    - [batch, 1]
    - [num_boxes, 10]
    - [num_classes, 1]
 - type: mask_coefficients
   decoder: ultralytics
   quantization: [0.00784313725490196, 0]
   shape: [1, 10, 32]
   dshape:
    - [batch, 1]
    - [num_boxes, 10]
    - [num_protos, 32]
 - type: protos
   decoder: ultralytics
   quantization: [0.0039215686274509803921568627451, 128]
   shape: [1, 160, 160, 32]
   dshape:
    - [batch, 1]
    - [height, 160]
    - [width, 160]
    - [num_protos, 32]
"""
    boxes_arr = np.zeros((1, 10, 4), dtype=np.float64)
    boxes_arr[0, 0] = [0.1234, 0.1234, 0.2345, 0.2345]

    scores_arr = np.zeros((1, 10, 1), dtype=np.float64)
    scores_arr[0, 0, 0] = 0.9876

    classes_arr = np.zeros((1, 10, 1), dtype=np.float64)
    classes_arr[0, 0, 0] = 2.0

    mask_arr = np.zeros((1, 10, 32), dtype=np.float64)
    protos_arr = np.zeros((1, 160, 160, 32), dtype=np.float64)

    decoder = edgefirst_hal.Decoder.new_from_yaml_str(config, 0.45, 0.45)
    tracker = edgefirst_hal.ByteTrack(update=0.1, high_conf=0.7)

    boxes, scores, classes, masks, tracks = decoder.decode_tracked(
        tracker, 0,
        [boxes_arr, scores_arr, classes_arr, mask_arr, protos_arr]
    )
    assert len(boxes) == 1
    assert np.allclose(boxes[0], [0.1234, 0.1234, 0.2345, 0.2345], atol=1.0 / 160.0)

    # Clear scores to simulate no detections
    scores_cleared = np.zeros_like(scores_arr)

    boxes, scores, classes, masks, tracks = decoder.decode_tracked(
        tracker, 100_000_000 // 3,
        [boxes_arr, scores_cleared, classes_arr, mask_arr, protos_arr]
    )
    assert len(boxes) == 1
    assert np.allclose(boxes[0], [0.1234, 0.1234, 0.2345, 0.2345], atol=1e-3)
    assert masks == []


def test_tracker_segdet_split_with_testdata():
    """Test tracked decoding with split segdet using real test data files.

    Corresponds to Rust test: test_decoder_tracked_segdet_split
    """
    raw_boxes = np.fromfile("testdata/yolov8_boxes_116x8400.bin", dtype=np.uint8).reshape(
        1, 116, 8400
    )
    protos = np.fromfile("testdata/yolov8_protos_160x160x32.bin", dtype=np.uint8).reshape(
        1, 160, 160, 32
    )

    # Split the combined tensor into boxes, scores, mask coefficients
    boxes_data = raw_boxes[:, :4, :]       # [1, 4, 8400]
    scores_data = raw_boxes[:, 4:84, :]    # [1, 80, 8400]
    mask_data = raw_boxes[:, 84:, :]       # [1, 32, 8400]

    config = """
decoder_version: yolov8
outputs:
 - type: boxes
   decoder: ultralytics
   quantization: [0.021287762, 31]
   shape: [1, 4, 8400]
   dshape:
    - [batch, 1]
    - [box_coords, 4]
    - [num_boxes, 8400]
   normalized: true
 - type: scores
   decoder: ultralytics
   quantization: [0.021287762, 31]
   shape: [1, 80, 8400]
   dshape:
    - [batch, 1]
    - [num_classes, 80]
    - [num_boxes, 8400]
 - type: mask_coefficients
   decoder: ultralytics
   quantization: [0.021287762, 31]
   shape: [1, 32, 8400]
   dshape:
    - [batch, 1]
    - [num_protos, 32]
    - [num_boxes, 8400]
 - type: protos
   decoder: ultralytics
   quantization: [0.02491162, 11]
   shape: [1, 160, 160, 32]
   dshape:
    - [batch, 1]
    - [height, 160]
    - [width, 160]
    - [num_protos, 32]
"""
    decoder = edgefirst_hal.Decoder.new_from_yaml_str(config, 0.45, 0.45)
    tracker = edgefirst_hal.ByteTrack(update=0.1, high_conf=0.7)

    boxes, scores, classes, masks, tracks = decoder.decode_tracked(
        tracker, 0,
        [boxes_data, scores_data, mask_data, protos]
    )
    assert len(boxes) == 2

    expected_box0 = [0.08515105, 0.7131401, 0.29802868, 0.8195788]
    expected_box1 = [0.59605736, 0.25545314, 0.93666154, 0.72378385]
    assert np.allclose(boxes[0], expected_box0, atol=1.0 / 160.0)
    assert np.allclose(boxes[1], expected_box1, atol=1.0 / 160.0)

    # Clear scores to simulate no detections
    scores_cleared = np.zeros_like(scores_data)

    boxes, scores, classes, masks, tracks = decoder.decode_tracked(
        tracker, 100_000_000 // 3,
        [boxes_data, scores_cleared, mask_data, protos]
    )

    # Tracker predicts forward
    assert len(boxes) == 2
    assert np.allclose(boxes[0], expected_box0, atol=1e-2)
    assert np.allclose(boxes[1], expected_box1, atol=1e-2)
    assert masks == []
