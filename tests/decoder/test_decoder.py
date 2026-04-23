# SPDX-FileCopyrightText: Copyright 2025 Au-Zone Technologies
# SPDX-License-Identifier: Apache-2.0

import edgefirst_hal
import numpy as np
import pytest


def numpy_to_tensor(arr, mem=None):
    import edgefirst_hal

    dtype_map = {
        "int8": "int8",
        "uint8": "uint8",
        "int16": "int16",
        "uint16": "uint16",
        "int32": "int32",
        "uint32": "uint32",
        "int64": "int64",
        "uint64": "uint64",
        "float32": "float32",
        "float64": "float64",
    }
    import numpy as np

    hal_dtype = dtype_map.get(str(arr.dtype))
    if hal_dtype is None:
        raise ValueError(f"Unsupported numpy dtype: {arr.dtype}")
    arr = np.ascontiguousarray(arr)
    tensor = edgefirst_hal.Tensor(list(arr.shape), dtype=hal_dtype, mem=mem)
    with tensor.map() as m:
        dst = np.frombuffer(m, dtype=arr.dtype).reshape(arr.shape)
        np.copyto(dst, arr)
    return tensor


def test_from_json():
    output0 = numpy_to_tensor(
        np.fromfile("testdata/modelpack_split_17x30x18.bin", dtype=np.uint8).reshape(
            1, 17, 30, 18
        )
    )
    output1 = numpy_to_tensor(
        np.fromfile("testdata/modelpack_split_9x15x18.bin", dtype=np.uint8).reshape(
            1, 9, 15, 18
        )
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
    output0 = numpy_to_tensor(
        np.fromfile("testdata/modelpack_split_17x30x18.bin", dtype=np.uint8).reshape(
            1, 17, 30, 18
        )
    )
    output1 = numpy_to_tensor(
        np.fromfile("testdata/modelpack_split_9x15x18.bin", dtype=np.uint8).reshape(
            1, 9, 15, 18
        )
    )
    with open("testdata/modelpack_split.yaml") as f:
        config = f.read()

    decoder = edgefirst_hal.Decoder.new_from_yaml_str(config, 0.45, 0.45)
    boxes, scores, classes, masks = decoder.decode([output0, output1])
    assert np.allclose(boxes, [[0.43171933, 0.68243736, 0.5626645, 0.808863]])
    assert np.allclose(scores, [0.99240804])
    assert np.allclose(classes, [0])
    assert len(masks) == 0


def test_from_json_v2_ara2_int8():
    # Schema v2 metadata for an ARA-2 YOLOv8-seg model. The boxes
    # logical splits into xy + wh physical children with independent
    # int8 quantization, so the HAL must compile a DecodeProgram and
    # merge them at decode time. Zero-filled tensors dequantize to
    # near-zero scores (well under 0.25) so the smoke test asserts
    # the full parse → validate → plan → dequant → merge → dispatch
    # pipeline runs without panicking and yields no detections.
    with open("testdata/ara2_int8_edgefirst.json") as f:
        config = f.read()

    decoder = edgefirst_hal.Decoder.new_from_json_str(config, 0.25, 0.5)

    xy = numpy_to_tensor(np.zeros((1, 2, 8400, 1), dtype=np.int8))
    wh = numpy_to_tensor(np.zeros((1, 2, 8400, 1), dtype=np.int8))
    scores = numpy_to_tensor(np.zeros((1, 80, 8400, 1), dtype=np.uint8))
    mask_coefs = numpy_to_tensor(np.zeros((1, 32, 8400, 1), dtype=np.int8))
    protos = numpy_to_tensor(np.zeros((1, 32, 160, 160), dtype=np.int8))

    boxes, scores_out, classes, masks = decoder.decode(
        [xy, wh, scores, mask_coefs, protos]
    )
    assert len(boxes) == 0
    assert len(scores_out) == 0
    assert len(classes) == 0
    assert len(masks) == 0


def test_from_dict_json():
    import json

    output0 = numpy_to_tensor(
        np.fromfile("testdata/modelpack_split_17x30x18.bin", dtype=np.uint8).reshape(
            1, 17, 30, 18
        )
    )
    output1 = numpy_to_tensor(
        np.fromfile("testdata/modelpack_split_9x15x18.bin", dtype=np.uint8).reshape(
            1, 9, 15, 18
        )
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

    output0 = numpy_to_tensor(
        np.fromfile("testdata/modelpack_split_17x30x18.bin", dtype=np.uint8).reshape(
            1, 17, 30, 18
        )
    )
    output1 = numpy_to_tensor(
        np.fromfile("testdata/modelpack_split_9x15x18.bin", dtype=np.uint8).reshape(
            1, 9, 15, 18
        )
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


def test_from_dict_v2_minimal():
    """EDGEAI-1081: Schema v2 dict with object-style quantization must succeed.

    Regression test for the blocker where `Decoder(dict)` routed v2 dicts
    through the legacy `ConfigOutputs` deserializer, producing the
    misleading "invalid type: map, expected tuple struct QuantTuple" error
    because the legacy `QuantTuple(f32, i32)` expects a JSON array while
    v2 prescribes `{"scale": ..., "zero_point": ..., "dtype": ...}`.
    """
    cfg = {
        "schema_version": 2,
        "decoder_version": "yolov8",
        "nms": "class_agnostic",
        "outputs": [
            {
                "name": "boxes",
                "type": "boxes",
                "shape": [1, 4, 8400],
                "dshape": [{"batch": 1}, {"box_coords": 4}, {"num_boxes": 8400}],
                "dtype": "int8",
                "quantization": {
                    "scale": 0.0262,
                    "zero_point": 0,
                    "dtype": "int8",
                },
                "decoder": "ultralytics",
                "encoding": "direct",
                "normalized": True,
            },
            {
                "name": "scores",
                "type": "scores",
                "shape": [1, 80, 8400],
                "dshape": [
                    {"batch": 1},
                    {"num_classes": 80},
                    {"num_boxes": 8400},
                ],
                "dtype": "int8",
                "quantization": {
                    "scale": 0.00392,
                    "zero_point": -128,
                    "dtype": "int8",
                },
                "decoder": "ultralytics",
                "score_format": "per_class",
            },
        ],
    }
    decoder = edgefirst_hal.Decoder(cfg)
    assert decoder is not None


def test_from_dict_v2_ara2_full_surface():
    """EDGEAI-1081: Full v2 surface (per-channel quant, split xy/wh, mask_coefs).

    Feeds the canonical ARA-2 fixture through the dict constructor. This
    exercises the same DecodeProgram compilation path `new_from_json_str`
    already covers via string, but via dict so the Python binding's smart
    constructor is the one doing the heavy lifting.
    """
    import json

    with open("testdata/ara2_int8_edgefirst.json") as f:
        cfg = json.load(f)

    decoder = edgefirst_hal.Decoder(cfg, 0.25, 0.5)

    xy = numpy_to_tensor(np.zeros((1, 2, 8400, 1), dtype=np.int8))
    wh = numpy_to_tensor(np.zeros((1, 2, 8400, 1), dtype=np.int8))
    scores = numpy_to_tensor(np.zeros((1, 80, 8400, 1), dtype=np.uint8))
    mask_coefs = numpy_to_tensor(np.zeros((1, 32, 8400, 1), dtype=np.int8))
    protos = numpy_to_tensor(np.zeros((1, 32, 160, 160), dtype=np.int8))

    boxes, scores_out, classes, masks = decoder.decode(
        [xy, wh, scores, mask_coefs, protos]
    )
    assert len(boxes) == 0
    assert len(scores_out) == 0
    assert len(classes) == 0
    assert len(masks) == 0


def test_from_dict_v2_rejects_unsupported_schema_version():
    """EDGEAI-1081: `schema_version` above MAX_SUPPORTED must be rejected.

    Without version gating the document would silently fall through to the
    legacy deserialiser and produce a confusing QuantTuple error. The
    smart constructor must surface a NotSupported-shaped error instead.
    """
    cfg = {
        "schema_version": 99,
        "outputs": [
            {"type": "boxes", "shape": [1, 4, 8400], "dtype": "int8"},
        ],
    }
    with pytest.raises(Exception) as exc_info:
        edgefirst_hal.Decoder(cfg)
    msg = str(exc_info.value)
    assert "QuantTuple" not in msg, f"v1 error leaked into v2 path: {msg}"
    assert "99" in msg or "not supported" in msg.lower() or "NotSupported" in msg


def test_from_dict_v2_mask_coefs_type_tag():
    """EDGEAI-1081: v2 vocabulary 'mask_coefs' is accepted by the dict path.

    The v2 spec defines the mask coefficients type as `mask_coefs`; the
    legacy enum spelled it `mask_coefficients`. Both must continue to
    parse — v2 as the primary tag, legacy as an alias.
    """
    cfg = {
        "schema_version": 2,
        "nms": "class_agnostic",
        "decoder_version": "yolov8",
        "outputs": [
            {
                "type": "boxes",
                "shape": [1, 4, 8400],
                "dshape": [
                    {"batch": 1},
                    {"box_coords": 4},
                    {"num_boxes": 8400},
                ],
                "dtype": "int8",
                "quantization": {"scale": 0.01, "zero_point": 0, "dtype": "int8"},
                "decoder": "ultralytics",
                "encoding": "direct",
            },
            {
                "type": "scores",
                "shape": [1, 80, 8400],
                "dshape": [
                    {"batch": 1},
                    {"num_classes": 80},
                    {"num_boxes": 8400},
                ],
                "dtype": "int8",
                "quantization": {"scale": 0.01, "zero_point": 0, "dtype": "int8"},
                "decoder": "ultralytics",
                "score_format": "per_class",
            },
            {
                "type": "mask_coefs",
                "shape": [1, 32, 8400],
                "dshape": [
                    {"batch": 1},
                    {"num_protos": 32},
                    {"num_boxes": 8400},
                ],
                "dtype": "int8",
                "quantization": {"scale": 0.01, "zero_point": 0, "dtype": "int8"},
                "decoder": "ultralytics",
            },
            {
                "type": "protos",
                "shape": [1, 160, 160, 32],
                "dshape": [
                    {"batch": 1},
                    {"height": 160},
                    {"width": 160},
                    {"num_protos": 32},
                ],
                "dtype": "int8",
                "quantization": {"scale": 0.01, "zero_point": 0, "dtype": "int8"},
            },
        ],
    }
    decoder = edgefirst_hal.Decoder(cfg)
    assert decoder is not None


def test_nms():
    tf = pytest.importorskip("tensorflow")

    output0 = numpy_to_tensor(
        np.fromfile("testdata/modelpack_split_17x30x18.bin", dtype=np.uint8).reshape(
            1, 17, 30, 18
        )
    )
    output1 = numpy_to_tensor(
        np.fromfile("testdata/modelpack_split_9x15x18.bin", dtype=np.uint8).reshape(
            1, 9, 15, 18
        )
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
                [output0, output1], 100_000
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
    output = numpy_to_tensor(
        np.fromfile("testdata/yolov8s_80_classes.bin", dtype=np.int8).reshape(
            1, 84, 8400
        )
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
        output = numpy_to_tensor(
            np.fromfile("testdata/yolov8s_80_classes.bin", dtype=np.int8).reshape(
                1, 84, 8400
            )
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
        output0 = numpy_to_tensor(
            np.fromfile(
                "testdata/modelpack_split_17x30x18.bin", dtype=np.uint8
            ).reshape(1, 17, 30, 18)
        )
        output1 = numpy_to_tensor(
            np.fromfile("testdata/modelpack_split_9x15x18.bin", dtype=np.uint8).reshape(
                1, 9, 15, 18
            )
        )

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


def test_new_from_outputs_yolov8():
    """Test Decoder.new_from_outputs with a single detection output (YOLOv8 style)."""
    output = edgefirst_hal.Output.detection(shape=[1, 84, 8400]).with_quantization(
        scale=0.0040811873, zero_point=-123
    )
    decoder = edgefirst_hal.Decoder.new_from_outputs(
        outputs=[output],
        score_threshold=0.2,
        iou_threshold=0.7,
    )
    model_output = numpy_to_tensor(
        np.fromfile("testdata/yolov8s_80_classes.bin", dtype=np.int8).reshape(
            1, 84, 8400
        )
    )
    boxes, scores, classes, masks = decoder.decode([model_output])

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


def test_new_from_outputs_yolov8_split():
    """Test Decoder.new_from_outputs with split boxes + scores outputs."""
    boxes_output = edgefirst_hal.Output.boxes(
        dshape=[
            (edgefirst_hal.DimName.Batch, 1),
            (edgefirst_hal.DimName.BoxCoords, 4),
            (edgefirst_hal.DimName.NumBoxes, 8400),
        ]
    )
    scores_output = edgefirst_hal.Output.scores(
        dshape=[
            (edgefirst_hal.DimName.Batch, 1),
            (edgefirst_hal.DimName.NumClasses, 80),
            (edgefirst_hal.DimName.NumBoxes, 8400),
        ]
    )
    decoder = edgefirst_hal.Decoder.new_from_outputs(
        outputs=[boxes_output, scores_output],
        score_threshold=0.2,
        iou_threshold=0.7,
    )
    # Generate split tensors from the combined detection data
    raw = np.fromfile("testdata/yolov8s_80_classes.bin", dtype=np.int8).reshape(
        1, 84, 8400
    )
    dequantized = (raw.astype(np.float32) - (-123.0)) * 0.0040811873
    boxes_arr = np.ascontiguousarray(dequantized[:, :4, :])
    scores_arr = np.ascontiguousarray(dequantized[:, 4:, :])

    boxes, scores, classes, masks = decoder.decode(
        [numpy_to_tensor(boxes_arr), numpy_to_tensor(scores_arr)]
    )
    assert len(boxes) > 0
    assert len(scores) > 0
    assert len(masks) == 0


def test_output_with_quantization():
    """Test that with_quantization setter works correctly."""
    output = edgefirst_hal.Output.detection(shape=[1, 84, 8400]).with_quantization(
        scale=0.0040811873, zero_point=-123
    )
    decoder = edgefirst_hal.Decoder.new_from_outputs(
        outputs=[output],
        score_threshold=0.2,
        iou_threshold=0.7,
    )
    model_output = numpy_to_tensor(
        np.fromfile("testdata/yolov8s_80_classes.bin", dtype=np.int8).reshape(
            1, 84, 8400
        )
    )
    boxes, scores, classes, masks = decoder.decode([model_output])
    # Should produce the same results as the dict-based config
    assert np.allclose(
        boxes,
        [
            [0.5285137, 0.05305544, 0.87541467, 0.9998909],
            [0.130598, 0.43260583, 0.35098213, 0.9958097],
        ],
    )


def test_output_with_dshape():
    """Test Output creation with named dimensions (dshape)."""
    output = edgefirst_hal.Output.detection(
        dshape=[
            (edgefirst_hal.DimName.Batch, 1),
            (edgefirst_hal.DimName.NumFeatures, 84),
            (edgefirst_hal.DimName.NumBoxes, 8400),
        ]
    ).with_quantization(scale=0.0040811873, zero_point=-123)

    decoder = edgefirst_hal.Decoder.new_from_outputs(
        outputs=[output],
        score_threshold=0.2,
        iou_threshold=0.7,
    )
    model_output = numpy_to_tensor(
        np.fromfile("testdata/yolov8s_80_classes.bin", dtype=np.int8).reshape(
            1, 84, 8400
        )
    )
    boxes, scores, classes, masks = decoder.decode([model_output])

    assert np.allclose(
        boxes,
        [
            [0.5285137, 0.05305544, 0.87541467, 0.9998909],
            [0.130598, 0.43260583, 0.35098213, 0.9958097],
        ],
    )


def test_output_shape_or_dshape_required():
    """Test that providing neither shape nor dshape raises an error."""
    with pytest.raises(ValueError, match="Either 'shape' or 'dshape' must be provided"):
        edgefirst_hal.Output.detection()


def test_output_shape_and_dshape_exclusive():
    """Test that providing both shape and dshape raises an error."""
    with pytest.raises(
        ValueError, match="Provide either 'shape' or 'dshape', not both"
    ):
        edgefirst_hal.Output.detection(
            shape=[1, 84, 8400],
            dshape=[
                (edgefirst_hal.DimName.Batch, 1),
                (edgefirst_hal.DimName.NumFeatures, 84),
                (edgefirst_hal.DimName.NumBoxes, 8400),
            ],
        )


def test_decoder_version():
    """Test that DecoderVersion enum variants are accessible."""
    assert edgefirst_hal.DecoderVersion.Yolov5 is not None
    assert edgefirst_hal.DecoderVersion.Yolov8 is not None
    assert edgefirst_hal.DecoderVersion.Yolo11 is not None
    assert edgefirst_hal.DecoderVersion.Yolo26 is not None


def test_output_chaining():
    """Test that Output setter methods can be chained."""
    output = (
        edgefirst_hal.Output.detection(shape=[1, 84, 8400])
        .with_quantization(scale=0.004, zero_point=-123)
        .with_normalized(True)
    )
    decoder = edgefirst_hal.Decoder.new_from_outputs(
        outputs=[output],
        score_threshold=0.25,
        iou_threshold=0.7,
    )
    assert decoder.normalized_boxes is True


def test_with_anchors_on_non_detection_raises():
    """Test that with_anchors() raises ValueError on non-detection outputs."""
    with pytest.raises(ValueError, match="with_anchors.*only valid for detection"):
        edgefirst_hal.Output.boxes(shape=[1, 4, 8400]).with_anchors([(0.5, 0.5)])


def test_with_normalized_on_unsupported_raises():
    """Test that with_normalized() raises ValueError on unsupported output types."""
    with pytest.raises(
        ValueError, match="with_normalized.*only valid for detection or boxes"
    ):
        edgefirst_hal.Output.scores(shape=[1, 80, 8400]).with_normalized(True)


def test_factory_methods_protos():
    """Test that Output.protos() and mask_coefficients() factory methods work."""
    # 80 classes + 4 box coords + 32 mask coefficients = 116 num_features
    # protos: [batch, num_protos, height, width]
    detection = edgefirst_hal.Output.detection(
        dshape=[
            (edgefirst_hal.DimName.Batch, 1),
            (edgefirst_hal.DimName.NumFeatures, 116),
            (edgefirst_hal.DimName.NumBoxes, 8400),
        ]
    )
    protos = edgefirst_hal.Output.protos(
        dshape=[
            (edgefirst_hal.DimName.Batch, 1),
            (edgefirst_hal.DimName.NumProtos, 32),
            (edgefirst_hal.DimName.Height, 160),
            (edgefirst_hal.DimName.Width, 160),
        ]
    )
    decoder = edgefirst_hal.Decoder.new_from_outputs(
        outputs=[detection, protos],
        score_threshold=0.25,
    )
    assert decoder is not None


def test_factory_methods_modelpack():
    """Test that DecoderType.ModelPack can be passed to factory methods."""
    output = edgefirst_hal.Output.detection(
        shape=[1, 17, 30, 18],
        decoder=edgefirst_hal.DecoderType.ModelPack,
    )
    assert output is not None


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
    output = numpy_to_tensor(
        np.fromfile("testdata/yolov8s_80_classes.bin", dtype=np.int8).reshape(
            1, 84, 8400
        )
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
