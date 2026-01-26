# SPDX-FileCopyrightText: Copyright 2025 Au-Zone Technologies
# SPDX-License-Identifier: Apache-2.0

import edgefirst_hal
import numpy as np
import pytest


def test_decoder():
    output = np.empty((84, 8400), dtype=np.float32)
    quant = np.fromfile("testdata/yolov8s_80_classes.bin",
                        dtype=np.int8).reshape(84, 8400)
    edgefirst_hal.Decoder.dequantize(
        quant,
        (0.0040811873, -123),
        output,
    )

    dequantized = (quant.astype(np.float32) - (-123.0)) * 0.0040811873
    assert np.allclose(output, dequantized)


def test_from_json():
    output0 = np.fromfile("testdata/modelpack_split_17x30x18.bin", dtype=np.uint8).reshape(
        1, 17, 30, 18
    )
    output1 = np.fromfile("testdata/modelpack_split_9x15x18.bin", dtype=np.uint8).reshape(
        1, 9, 15, 18
    )

    with open("testdata/modelpack_split.json") as f:
        config = f.read()

    decoder = edgefirst_hal.Decoder.new_from_json_str(config, 0.45, 0.45)
    boxes, scores, classes, masks = decoder.decode([output0, output1])
    assert np.allclose(boxes, [[0.43171933, 0.68243736, 0.5626645, 0.808863]])
    assert np.allclose(scores, [0.99240804])
    assert np.allclose(classes, [0])
    assert len(masks) == 0


def test_from_yaml():
    output0 = np.fromfile("testdata/modelpack_split_17x30x18.bin", dtype=np.uint8).reshape(
        1, 17, 30, 18
    )
    output1 = np.fromfile("testdata/modelpack_split_9x15x18.bin", dtype=np.uint8).reshape(
        1, 9, 15, 18
    )
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

    output0 = np.fromfile("testdata/modelpack_split_17x30x18.bin", dtype=np.uint8).reshape(
        1, 17, 30, 18
    )
    output1 = np.fromfile("testdata/modelpack_split_9x15x18.bin", dtype=np.uint8).reshape(
        1, 9, 15, 18
    )

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

    output0 = np.fromfile("testdata/modelpack_split_17x30x18.bin", dtype=np.uint8).reshape(
        1, 17, 30, 18
    )
    output1 = np.fromfile("testdata/modelpack_split_9x15x18.bin", dtype=np.uint8).reshape(
        1, 9, 15, 18
    )

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
    output0 = np.fromfile("testdata/modelpack_split_17x30x18.bin", dtype=np.uint8).reshape(
        17, 30, 18
    )
    anchors0 = [
        [0.13750000298023224, 0.2074074000120163],
        [0.2541666626930237, 0.21481481194496155],
        [0.23125000298023224, 0.35185185074806213],
    ]
    quant0 = (0.09929127991199493, 183)

    output1 = np.fromfile("testdata/modelpack_split_9x15x18.bin", dtype=np.uint8).reshape(
        9, 15, 18
    )
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
    output0 = np.fromfile("testdata/modelpack_split_17x30x18.bin", dtype=np.uint8).reshape(
        1, 17, 30, 18
    )
    output1 = np.fromfile("testdata/modelpack_split_9x15x18.bin", dtype=np.uint8).reshape(
        1, 9, 15, 18
    )
    # this int32 array should be ignored by the decoder
    int32_arr = np.zeros((100, 100), np.int32)

    with open("testdata/modelpack_split.yaml") as f:
        config = f.read()

    decoder = edgefirst_hal.Decoder.new_from_yaml_str(config, 0.45, 0.45)
    boxes, scores, classes, masks = decoder.decode(
        [output0, output1, int32_arr])

    assert np.allclose(boxes, [[0.43171933, 0.68243736, 0.5626645, 0.808863]])
    assert np.allclose(scores, [0.99240804])
    assert np.allclose(classes, [0])
    assert len(masks) == 0


def test_nms():
    tf = pytest.importorskip("tensorflow")

    output0 = np.fromfile("testdata/modelpack_split_17x30x18.bin", dtype=np.uint8).reshape(
        1, 17, 30, 18
    )
    output1 = np.fromfile("testdata/modelpack_split_9x15x18.bin", dtype=np.uint8).reshape(
        1, 9, 15, 18
    )

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
                "quantization": [
                    0.0040811873,
                    -123
                ],
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
                ]
            },
        ],

    }
    decoder = edgefirst_hal.Decoder(config, 0.2, 0.7)
    output = np.fromfile("testdata/yolov8s_80_classes.bin",
                         dtype=np.int8).reshape(1, 84, 8400)
    boxes, scores, classes, masks = decoder.decode([output])

    assert np.allclose(boxes, [[0.5285137, 0.05305544, 0.87541467, 0.9998909], [
                       0.130598, 0.43260583, 0.35098213, 0.9958097]])
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
                    ]
                },
            ],
        }
        decoder = edgefirst_hal.Decoder(config, 0.25, 0.7)
        output = np.fromfile("testdata/yolov8s_80_classes.bin", dtype=np.int8).reshape(1, 84, 8400)

        for _ in range(100):
            boxes, scores, classes, masks = decoder.decode([output])

            assert np.allclose(boxes[0], [0.5285137, 0.05305544, 0.87541467, 0.9998909], atol=1e-6)
            assert np.allclose(scores[0], 0.5591227, atol=1e-6)
            assert classes[0] == 0

            assert np.allclose(boxes[1], [0.130598, 0.43260583, 0.35098213, 0.9958097], atol=1e-6)
            assert np.allclose(scores[1], 0.33057618, atol=1e-6)
            assert classes[1] == 75

            assert len(masks) == 0

    def modelpack_det_split():
        output0 = np.fromfile("testdata/modelpack_split_17x30x18.bin", dtype=np.uint8).reshape(1, 17, 30, 18)
        output1 = np.fromfile("testdata/modelpack_split_9x15x18.bin", dtype=np.uint8).reshape(1, 9, 15, 18)

        with open("testdata/modelpack_split.json") as f:
            config = f.read()

        decoder = edgefirst_hal.Decoder.new_from_json_str(config, 0.8, 0.5)

        for _ in range(100):
            boxes, scores, classes, masks = decoder.decode([output0, output1])

            assert np.allclose(boxes, [[0.43171933, 0.68243736, 0.5626645, 0.808863]], atol=1e-6)
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
