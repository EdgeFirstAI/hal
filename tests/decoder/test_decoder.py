# SPDX-FileCopyrightText: Copyright 2025 Au-Zone Technologies
# SPDX-License-Identifier: Apache-2.0

import edgefirst_hal
import numpy as np
import pytest


def test_decoder():
    output = np.empty((84, 8400), dtype=np.float32)
    quant = np.fromfile("testdata/yolov8s_80_classes.bin", dtype=np.int8).reshape(
        84, 8400
    )
    edgefirst_hal.Decoder.dequantize(
        quant,
        (0.0040811873, -123),
        output,
    )

    dequantized = (quant.astype(np.float32) - (-123.0)) * 0.0040811873
    assert np.allclose(output, dequantized)


def test_from_json():
    output0 = np.fromfile(
        "testdata/modelpack_split_17x30x18.bin", dtype=np.uint8
    ).reshape(1, 17, 30, 18)
    output1 = np.fromfile(
        "testdata/modelpack_split_9x15x18.bin", dtype=np.uint8
    ).reshape(1, 9, 15, 18)

    with open("testdata/modelpack_split.json") as f:
        config = f.read()

    decoder = edgefirst_hal.Decoder.new_from_json_str(config, 0.45, 0.45)
    boxes, scores, classes, masks = decoder.decode([output0, output1])
    assert np.allclose(boxes, [[0.43171933, 0.68243736, 0.5626645, 0.808863]])
    assert np.allclose(scores, [0.99240804])
    assert np.allclose(classes, [0])
    assert len(masks) == 0


def test_from_yaml():
    output0 = np.fromfile(
        "testdata/modelpack_split_17x30x18.bin", dtype=np.uint8
    ).reshape(1, 17, 30, 18)
    output1 = np.fromfile(
        "testdata/modelpack_split_9x15x18.bin", dtype=np.uint8
    ).reshape(1, 9, 15, 18)
    with open("testdata/modelpack_split.yaml") as f:
        config = f.read()

    decoder = edgefirst_hal.Decoder.new_from_yaml_str(config, 0.45, 0.45)
    boxes, scores, classes, masks = decoder.decode([output0, output1])
    assert np.allclose(boxes, [[0.43171933, 0.68243736, 0.5626645, 0.808863]])
    assert np.allclose(scores, [0.99240804])
    assert np.allclose(classes, [0])
    assert len(masks) == 0


def test_from_dict_json():
    import json

    output0 = np.fromfile(
        "testdata/modelpack_split_17x30x18.bin", dtype=np.uint8
    ).reshape(1, 17, 30, 18)
    output1 = np.fromfile(
        "testdata/modelpack_split_9x15x18.bin", dtype=np.uint8
    ).reshape(1, 9, 15, 18)

    with open("testdata/modelpack_split.json") as f:
        config = f.read()

    config = json.loads(config)
    decoder = edgefirst_hal.Decoder(config, 0.45, 0.45)
    boxes, scores, classes, masks = decoder.decode([output0, output1])
    assert np.allclose(boxes, [[0.43171933, 0.68243736, 0.5626645, 0.808863]])
    assert np.allclose(scores, [0.99240804])
    assert np.allclose(classes, [0])
    assert len(masks) == 0


def test_from_dict_yaml():
    yaml = pytest.importorskip("yaml")

    output0 = np.fromfile(
        "testdata/modelpack_split_17x30x18.bin", dtype=np.uint8
    ).reshape(1, 17, 30, 18)
    output1 = np.fromfile(
        "testdata/modelpack_split_9x15x18.bin", dtype=np.uint8
    ).reshape(1, 9, 15, 18)

    with open("testdata/modelpack_split.yaml") as f:
        config = f.read()

    config = yaml.safe_load(config)
    decoder = edgefirst_hal.Decoder(config, 0.45, 0.45)
    boxes, scores, classes, masks = decoder.decode([output0, output1])
    assert np.allclose(boxes, [[0.43171933, 0.68243736, 0.5626645, 0.808863]])
    assert np.allclose(scores, [0.99240804])
    assert np.allclose(classes, [0])
    assert len(masks) == 0


def test_modelpack_direct():
    output0 = np.fromfile(
        "testdata/modelpack_split_17x30x18.bin", dtype=np.uint8
    ).reshape(17, 30, 18)
    anchors0 = [
        [0.13750000298023224, 0.2074074000120163],
        [0.2541666626930237, 0.21481481194496155],
        [0.23125000298023224, 0.35185185074806213],
    ]
    quant0 = (0.09929127991199493, 183)

    output1 = np.fromfile(
        "testdata/modelpack_split_9x15x18.bin", dtype=np.uint8
    ).reshape(9, 15, 18)
    anchors1 = [
        [0.36666667461395264, 0.31481480598449707],
        [0.38749998807907104, 0.4740740656852722],
        [0.5333333611488342, 0.644444465637207],
    ]
    quant1 = (0.08547406643629074, 174)

    boxes, scores, classes = edgefirst_hal.Decoder.decode_modelpack_det_split(
        [output0, output1], [anchors0, anchors1], [quant0, quant1], 0.45, 0.45
    )

    assert np.allclose(boxes, [[0.43171933, 0.68243736, 0.5626645, 0.808863]])
    assert np.allclose(scores, [0.99240804])
    assert np.allclose(classes, [0])


def test_filter_int32():
    output0 = np.fromfile(
        "testdata/modelpack_split_17x30x18.bin", dtype=np.uint8
    ).reshape(1, 17, 30, 18)
    output1 = np.fromfile(
        "testdata/modelpack_split_9x15x18.bin", dtype=np.uint8
    ).reshape(1, 9, 15, 18)
    # this int32 array should be ignored by the decoder
    int32_arr = np.zeros((100, 100), np.int32)

    with open("testdata/modelpack_split.yaml") as f:
        config = f.read()

    decoder = edgefirst_hal.Decoder.new_from_yaml_str(config, 0.45, 0.45)
    boxes, scores, classes, masks = decoder.decode([output0, output1, int32_arr])

    assert np.allclose(boxes, [[0.43171933, 0.68243736, 0.5626645, 0.808863]])
    assert np.allclose(scores, [0.99240804])
    assert np.allclose(classes, [0])
    assert len(masks) == 0


def test_nms():
    tf = pytest.importorskip("tensorflow")

    output0 = np.fromfile(
        "testdata/modelpack_split_17x30x18.bin", dtype=np.uint8
    ).reshape(1, 17, 30, 18)
    output1 = np.fromfile(
        "testdata/modelpack_split_9x15x18.bin", dtype=np.uint8
    ).reshape(1, 9, 15, 18)

    with open("testdata/modelpack_split.yaml") as f:
        config = f.read()

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


def test_yolo_det():
    config = {
        "outputs": [
            {
                "decoder": "ultralytics",
                "quantization": [0.0040811873, -123],
                "shape": [
                    1,
                    84,
                    8400,
                ],
                "type": "detection",
                "dshape": [
                    ["batch", 1],
                    ["num_features", 84],
                    ["num_boxes", 8400],
                ],
            },
        ],
    }
    decoder = edgefirst_hal.Decoder(config, 0.2, 0.7)
    output = np.fromfile("testdata/yolov8s_80_classes.bin", dtype=np.int8).reshape(
        1, 84, 8400
    )
    boxes, scores, classes, masks = decoder.decode([output])

    assert np.allclose(
        boxes,
        [
            [0.5285137, 0.05305544, 0.87541467, 0.9998909],
            [0.130598, 0.43260583, 0.35098213, 0.9958097],
        ],
    )
    assert np.allclose(scores, [0.5591227, 0.33057618])
    assert np.allclose(classes, [0, 75])
    assert len(masks) == 0


def test_context_switch():
    """Test running multiple decoders concurrently"""
    import threading

    def yolo_det():
        config = {
            "outputs": [
                {
                    "decoder": "ultralytics",
                    "quantization": [0.0040811873, -123],
                    "shape": [1, 84, 8400],
                    "type": "detection",
                    "dshape": [
                        ["batch", 1],
                        ["num_features", 84],
                        ["num_boxes", 8400],
                    ],
                },
            ],
        }
        decoder = edgefirst_hal.Decoder(config, 0.25, 0.7)
        output = np.fromfile("testdata/yolov8s_80_classes.bin", dtype=np.int8).reshape(
            1, 84, 8400
        )

        for _ in range(100):
            boxes, scores, classes, masks = decoder.decode([output])

            assert np.allclose(
                boxes[0], [0.5285137, 0.05305544, 0.87541467, 0.9998909], atol=1e-6
            )
            assert np.allclose(scores[0], 0.5591227, atol=1e-6)
            assert classes[0] == 0

            assert np.allclose(
                boxes[1], [0.130598, 0.43260583, 0.35098213, 0.9958097], atol=1e-6
            )
            assert np.allclose(scores[1], 0.33057618, atol=1e-6)
            assert classes[1] == 75

            assert len(masks) == 0

    def modelpack_det_split():
        output0 = np.fromfile(
            "testdata/modelpack_split_17x30x18.bin", dtype=np.uint8
        ).reshape(1, 17, 30, 18)
        output1 = np.fromfile(
            "testdata/modelpack_split_9x15x18.bin", dtype=np.uint8
        ).reshape(1, 9, 15, 18)

        with open("testdata/modelpack_split.json") as f:
            config = f.read()

        decoder = edgefirst_hal.Decoder.new_from_json_str(config, 0.8, 0.5)

        for _ in range(100):
            boxes, scores, classes, masks = decoder.decode([output0, output1])

            assert np.allclose(
                boxes, [[0.43171933, 0.68243736, 0.5626645, 0.808863]], atol=1e-6
            )
            assert np.allclose(scores, [0.99240804], atol=1e-6)
            assert np.allclose(classes, [0])
            assert len(masks) == 0

    threads = [
        threading.Thread(target=yolo_det),
        threading.Thread(target=modelpack_det_split),
        threading.Thread(target=yolo_det),
        threading.Thread(target=modelpack_det_split),
        threading.Thread(target=yolo_det),
        threading.Thread(target=modelpack_det_split),
        threading.Thread(target=yolo_det),
        threading.Thread(target=modelpack_det_split),
    ]

    for thread in threads:
        thread.start()

    for thread in threads:
        thread.join()


@pytest.mark.benchmark(group="yolo_det", warmup_iterations=3)
def test_yolo_det_int8_hal(benchmark):
    config = {
        "outputs": [
            {
                "decoder": "ultralytics",
                "quantization": [0.0040811873, -123],
                "shape": [1, 84, 8400],
                "type": "detection",
                "dshape": [
                    ["batch", 1],
                    ["num_features", 84],
                    ["num_boxes", 8400],
                ],
            },
        ],
    }
    decoder = edgefirst_hal.Decoder(config, 0.25, 0.7)
    output = np.fromfile("testdata/yolov8s_80_classes.bin", dtype=np.int8).reshape(
        1, 84, 8400
    )

    def run_decode():
        return decoder.decode([output])

    boxes, scores, classes, masks = benchmark(run_decode)

    assert np.allclose(
        boxes,
        [
            [0.5285137, 0.05305544, 0.87541467, 0.9998909],
            [0.130598, 0.43260583, 0.35098213, 0.9958097],
        ],
    )
    assert np.allclose(scores, [0.5591227, 0.33057618])
    assert np.allclose(classes, [0, 75])
    assert len(masks) == 0


@pytest.mark.benchmark(group="yolo_det", warmup_iterations=3)
def test_yolo_det_int8_numpy_decode_tf_nms(benchmark):
    tf = pytest.importorskip("tensorflow")
    output = np.fromfile("testdata/yolov8s_80_classes.bin", dtype=np.int8).reshape(
        1, 84, 8400
    )

    iou_t = 0.7
    score_t = 0.25

    def run_decode(output):
        # Dequantize
        scale, zero_point = 0.0040811873, -123
        dequantized = (output.astype(np.float32) - zero_point) * scale

        # Extract box coordinates and class scores
        box_data = dequantized[0, :4, :]  # [4, 8400]
        class_scores = dequantized[0, 4:, :]  # [80, 8400]

        # Get max class score and class index for each box
        max_scores = np.max(class_scores, axis=0)  # [8400]

        # Filter by score threshold
        score_mask = max_scores >= score_t
        filtered_boxes = box_data[:, score_mask]  # [4, N]
        filtered_scores = max_scores[score_mask]  # [N]
        filtered_classes = np.argmax(class_scores[:, score_mask], axis=0)  # [N]

        # Convert from center format to corner format
        cx, cy, w, h = (
            filtered_boxes[0],
            filtered_boxes[1],
            filtered_boxes[2],
            filtered_boxes[3],
        )
        x1 = cx - w / 2
        y1 = cy - h / 2
        x2 = cx + w / 2
        y2 = cy + h / 2
        boxes = np.stack([x1, y1, x2, y2], axis=1)  # [N, 4]

        # Use TensorFlow NMS

        keep_indices = tf.image.non_max_suppression(
            boxes,
            filtered_scores,
            max_output_size=len(boxes),
            iou_threshold=iou_t,
            score_threshold=0.0,
        )
        boxes = boxes[keep_indices]
        scores = filtered_scores[keep_indices]
        classes = filtered_classes[keep_indices]

        return boxes, scores, classes, []

    boxes, scores, classes, masks = benchmark(run_decode, output)

    assert np.allclose(
        boxes,
        [
            [0.5285137, 0.05305544, 0.87541467, 0.9998909],
            [0.130598, 0.43260583, 0.35098213, 0.9958097],
        ],
    )
    assert np.allclose(scores, [0.5591227, 0.33057618])
    assert np.allclose(classes, [0, 75])
    assert len(masks) == 0


@pytest.mark.benchmark(group="yolo_det", warmup_iterations=3)
def test_yolo_det_int8_cv2(benchmark):
    cv2 = pytest.importorskip("cv2")

    output = np.fromfile("testdata/yolov8s_80_classes.bin", dtype=np.int8).reshape(
        1, 84, 8400
    )
    iou_t = 0.7
    score_t = 0.25

    def run_decode(output):
        # Dequantize
        scale, zero_point = 0.0040811873, -123
        dequantized = (output.astype(np.float32) - zero_point) * scale

        # Extract box coordinates and class scores
        box_data = dequantized[0, :4, :]  # [4, 8400]
        class_scores = dequantized[0, 4:, :]  # [80, 8400]

        # Get max class score and class index for each box
        max_scores = np.max(class_scores, axis=0)  # [8400]

        # Filter by score threshold
        score_mask = max_scores >= score_t
        filtered_boxes = box_data[:, score_mask]  # [4, N]
        filtered_scores = max_scores[score_mask]  # [N]
        filtered_classes = np.argmax(class_scores[:, score_mask], axis=0)  # [N]

        # Convert from center format to corner format
        cx, cy, w, h = (
            filtered_boxes[0],
            filtered_boxes[1],
            filtered_boxes[2],
            filtered_boxes[3],
        )
        x1 = cx - w / 2
        y1 = cy - h / 2
        x2 = cx + w / 2
        y2 = cy + h / 2
        boxes = np.stack([x1, y1, x2, y2], axis=1)  # [N, 4]

        keep_indices = cv2.dnn.NMSBoxes(boxes, filtered_scores, score_t, iou_t)

        boxes = boxes[keep_indices]
        scores = filtered_scores[keep_indices]
        classes = filtered_classes[keep_indices]

        return boxes, scores, classes, []

    boxes, scores, classes, masks = benchmark(run_decode, output)

    assert np.allclose(
        boxes,
        [
            [0.5285137, 0.05305544, 0.87541467, 0.9998909],
            [0.130598, 0.43260583, 0.35098213, 0.9958097],
        ],
    )
    assert np.allclose(scores, [0.5591227, 0.33057618])
    assert np.allclose(classes, [0, 75])
    assert len(masks) == 0


@pytest.mark.benchmark(group="yolo_det", warmup_iterations=3)
def test_yolo_det_int8_no_nms(benchmark):
    output = np.fromfile("testdata/yolov8s_80_classes.bin", dtype=np.int8).reshape(
        1, 84, 8400
    )
    score_t = 0.25

    def run_decode(output):
        # Dequantize
        scale, zero_point = 0.0040811873, -123
        dequantized = (output.astype(np.float32) - zero_point) * scale

        # Extract box coordinates and class scores
        box_data = dequantized[0, :4, :]  # [4, 8400]
        class_scores = dequantized[0, 4:, :]  # [80, 8400]

        # Get max class score and class index for each box
        max_scores = np.max(class_scores, axis=0)  # [8400]

        # Filter by score threshold
        score_mask = max_scores >= score_t
        filtered_boxes = box_data[:, score_mask]  # [4, N]
        filtered_scores = max_scores[score_mask]  # [N]
        filtered_classes = np.argmax(class_scores[:, score_mask], axis=0)  # [N]

        # Convert from center format to corner format
        cx, cy, w, h = (
            filtered_boxes[0],
            filtered_boxes[1],
            filtered_boxes[2],
            filtered_boxes[3],
        )
        x1 = cx - w / 2
        y1 = cy - h / 2
        x2 = cx + w / 2
        y2 = cy + h / 2
        boxes = np.stack([x1, y1, x2, y2], axis=1)  # [N, 4]

        return boxes, filtered_scores, filtered_classes, []

    benchmark(run_decode, output)


def numpy_nms(
    boxes: np.ndarray,
    scores: np.ndarray,
    iou_threshold: float = 0.70,
) -> np.ndarray:
    """
    Single class NMS implemented in NumPy.
    Method taken from:: https://github.com/Megvii-BaseDetection/YOLOX/blob/main/yolox/utils/demo_utils.py#L57
    Original source from:: https://github.com/amusi/Non-Maximum-Suppression/blob/master/nms.py

    Parameters
    ----------
    boxes: np.ndarray
        Normalized input boxes to the NMS with shape (n, 4)
        in [xmin, ymin, xmax, ymax] format.
    scores: np.ndarray
        Input scores to the NMS (n, ).
    iou_threshold: float
        This is the IoU threshold for the NMS. Higher values
        are less strict in filtering overlapping detections.

    Returns
    -------
    np.ndarray
        This contains the indices of the boxes to keep.
    """
    if len(boxes) == 0:
        return np.array([], dtype=np.int32)

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    # Calculate areas (remove the +1 for normalized coordinates)
    areas = (x2 - x1) * (y2 - y1)

    # Sort by scores in descending order
    order = (-scores).argsort(kind="stable")

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)

        # Calculate intersection coordinates
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        # Calculate intersection area (remove +1 for normalized coords)
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h

        # Calculate IoU
        union = areas[i] + areas[order[1:]] - inter
        iou = np.where(union > 0, inter / union, 0.0)

        # Keep boxes with IoU less than threshold
        inds = np.where(iou < iou_threshold)[0]
        order = order[inds + 1]

    return np.array(keep, dtype=np.int32)


@pytest.mark.benchmark(group="yolo_det", warmup_iterations=3)
def test_yolo_det_int8_numpy(benchmark):
    output = np.fromfile("testdata/yolov8s_80_classes.bin", dtype=np.int8).reshape(
        1, 84, 8400
    )

    iou_t = 0.7
    score_t = 0.25

    def run_decode(output):
        # Dequantize
        scale, zero_point = 0.0040811873, -123
        dequantized = (output.astype(np.float32) - zero_point) * scale

        # Extract box coordinates and class scores
        box_data = dequantized[0, :4, :]  # [4, 8400]
        class_scores = dequantized[0, 4:, :]  # [80, 8400]

        # Get max class score and class index for each box
        max_scores = np.max(class_scores, axis=0)  # [8400]

        # Filter by score threshold
        score_mask = max_scores >= score_t
        filtered_boxes = box_data[:, score_mask]  # [4, N]
        filtered_scores = max_scores[score_mask]  # [N]
        filtered_classes = np.argmax(class_scores[:, score_mask], axis=0)  # [N]

        # Convert from center format to corner format
        cx, cy, w, h = (
            filtered_boxes[0],
            filtered_boxes[1],
            filtered_boxes[2],
            filtered_boxes[3],
        )
        x1 = cx - w / 2
        y1 = cy - h / 2
        x2 = cx + w / 2
        y2 = cy + h / 2
        boxes = np.stack([x1, y1, x2, y2], axis=1)  # [N, 4]

        keep_indices = numpy_nms(boxes, filtered_scores, iou_t)

        boxes = boxes[keep_indices]
        scores = filtered_scores[keep_indices]
        classes = filtered_classes[keep_indices]

        return boxes, scores, classes, []

    boxes, scores, classes, masks = benchmark(run_decode, output)

    assert np.allclose(
        boxes,
        [
            [0.5285137, 0.05305544, 0.87541467, 0.9998909],
            [0.130598, 0.43260583, 0.35098213, 0.9958097],
        ],
    )
    assert np.allclose(scores, [0.5591227, 0.33057618])
    assert np.allclose(classes, [0, 75])
    assert len(masks) == 0
