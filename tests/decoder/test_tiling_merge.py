# SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
# SPDX-License-Identifier: Apache-2.0

"""Pure-numpy tests for the tiled-detection postprocessing bindings.

No GPU is touched: these exercise ``lift_tile_boxes``, ``merge_tiled_detections``
and ``TiledFrameAccumulator`` against the canonical IOS-vs-IOU cases ported from
ModelPack ``tests/test_tiled_merge.py``.
"""

import numpy as np
import pytest

from edgefirst_hal import (
    MatchMetric,
    MergeConfig,
    TiledFrameAccumulator,
    TilePlacement,
    lift_tile_boxes,
    merge_tiled_detections,
)


def _triple(boxes, scores, classes):
    """Pack lists into the (bbox, scores, classes) numpy triple."""
    bbox = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
    sc = np.asarray(scores, dtype=np.float32)
    cl = np.asarray(classes, dtype=np.uintp)
    return bbox, sc, cl


# --- merge: the canonical IOS-vs-IOU case --------------------------------


def test_ios_merges_fragment_iou_does_not():
    # B = [350, 100, 400, 300] is fully inside A = [100, 100, 400, 300]:
    # IoS = 1.0, IoU ~= 0.167.
    bbox, sc, cl = _triple(
        [[100, 100, 400, 300], [350, 100, 400, 300]], [0.9, 0.7], [0, 0]
    )

    ios_b, ios_s, ios_c = merge_tiled_detections(
        bbox, sc, cl, MergeConfig(metric=MatchMetric.Ios, threshold=0.5)
    )
    assert ios_b.shape == (1, 4)
    np.testing.assert_allclose(ios_b[0], [100, 100, 400, 300])
    assert ios_s[0] == pytest.approx(0.9)
    assert ios_c[0] == 0

    iou_b, _, _ = merge_tiled_detections(
        bbox, sc, cl, MergeConfig(metric=MatchMetric.Iou, threshold=0.5)
    )
    assert iou_b.shape == (2, 4)


def test_merge_output_dtypes():
    bbox, sc, cl = _triple([[0, 0, 10, 10]], [0.9], [3])
    b, s, c = merge_tiled_detections(bbox, sc, cl, MergeConfig())
    assert b.dtype == np.float32
    assert s.dtype == np.float32
    assert c.dtype == np.uintp
    assert c[0] == 3


def test_merge_respects_class_unless_agnostic():
    # Same geometry as the IOS case but B is a different class.
    bbox, sc, cl = _triple(
        [[100, 100, 400, 300], [350, 100, 400, 300]], [0.9, 0.7], [0, 1]
    )

    aware_b, _, _ = merge_tiled_detections(bbox, sc, cl, MergeConfig())
    assert aware_b.shape == (2, 4)

    agn_b, _, agn_c = merge_tiled_detections(
        bbox, sc, cl, MergeConfig(class_agnostic=True)
    )
    assert agn_b.shape == (1, 4)
    # Merged box keeps the base (highest-score) label.
    assert agn_c[0] == 0
    np.testing.assert_allclose(agn_b[0], [100, 100, 400, 300])


def test_merge_enclosing_union_for_partial_overlap():
    # Two boxes overlapping exactly IoS == 0.5 merge to their enclosing union.
    bbox, sc, cl = _triple([[0, 0, 100, 100], [50, 0, 150, 100]], [0.9, 0.8], [0, 0])
    b, s, _ = merge_tiled_detections(
        bbox, sc, cl, MergeConfig(metric=MatchMetric.Ios, threshold=0.5)
    )
    assert b.shape == (1, 4)
    np.testing.assert_allclose(b[0], [0, 0, 150, 100])
    assert s[0] == pytest.approx(0.9)


def test_merge_empty_is_empty():
    bbox, sc, cl = _triple([], [], [])
    b, s, c = merge_tiled_detections(bbox, sc, cl, MergeConfig())
    assert b.shape == (0, 4)
    assert s.shape == (0,)
    assert c.shape == (0,)


def test_merge_score_threshold_and_max_det():
    bbox, sc, cl = _triple([[0, 0, 10, 10], [100, 100, 110, 110]], [0.3, 0.8], [0, 0])
    b, s, _ = merge_tiled_detections(bbox, sc, cl, MergeConfig(score_threshold=0.5))
    assert b.shape == (1, 4)
    assert s[0] == pytest.approx(0.8)

    # max_det caps to the highest-scoring groups.
    boxes = [[i * 50, 0, i * 50 + 10, 10] for i in range(10)]
    scores = [1.0 - i * 0.05 for i in range(10)]
    bbox, sc, cl = _triple(boxes, scores, [0] * 10)
    b, s, _ = merge_tiled_detections(bbox, sc, cl, MergeConfig(max_det=3))
    assert b.shape == (3, 4)
    assert s[0] >= s[1] >= s[2]


# --- lift ----------------------------------------------------------------


def test_lift_tile_boxes_origin_plus_scale():
    p = TilePlacement(
        index=0,
        count=1,
        origin=(100.0, 200.0),
        crop_size=(640.0, 640.0),
        frame_dims=(3840.0, 2160.0),
    )
    bbox, sc, cl = _triple(
        [[0.0, 0.0, 1.0, 1.0], [0.25, 0.5, 0.75, 1.0]], [0.9, 0.8], [0, 0]
    )
    b, _, _ = lift_tile_boxes(bbox, sc, cl, p)
    np.testing.assert_allclose(b[0], [100, 200, 740, 840])
    np.testing.assert_allclose(b[1], [260, 520, 580, 840])


def test_lift_tile_boxes_inverts_letterbox():
    p = TilePlacement(
        index=0,
        count=1,
        origin=(0.0, 0.0),
        crop_size=(640.0, 640.0),
        frame_dims=(640.0, 640.0),
        letterbox=(0.1, 0.1, 0.9, 0.9),
    )
    bbox, sc, cl = _triple([[0.1, 0.1, 0.9, 0.9]], [0.9], [0])
    b, _, _ = lift_tile_boxes(bbox, sc, cl, p)
    np.testing.assert_allclose(b[0], [0, 0, 640, 640], atol=1e-2)


def test_tile_placement_getters():
    p = TilePlacement(
        index=2,
        count=4,
        origin=(10.0, 20.0),
        crop_size=(640.0, 480.0),
        frame_dims=(1920.0, 1080.0),
        letterbox=(0.0, 0.1, 1.0, 0.9),
    )
    assert p.index == 2
    assert p.count == 4
    assert p.origin == (10.0, 20.0)
    assert p.crop_size == (640.0, 480.0)
    assert p.frame_dims == (1920.0, 1080.0)
    assert p.letterbox == pytest.approx((0.0, 0.1, 1.0, 0.9))

    no_lb = TilePlacement(0, 1, (0.0, 0.0), (1.0, 1.0), (1.0, 1.0))
    assert no_lb.letterbox is None


# --- accumulator fan-in --------------------------------------------------


def _seam_tiles():
    """Two side-by-side tiles of a 1280x640 frame, a box near each seam."""
    frame = (1280.0, 640.0)
    p0 = TilePlacement(0, 2, (0.0, 0.0), (640.0, 640.0), frame)
    p1 = TilePlacement(1, 2, (640.0, 0.0), (640.0, 640.0), frame)
    t0 = _triple([[0.9, 0.4, 1.0, 0.6]], [0.8], [0])
    t1 = _triple([[0.0, 0.4, 0.1, 0.6]], [0.9], [0])
    return frame, (p0, t0), (p1, t1)


def test_accumulator_fan_in_fence():
    frame, (p0, t0), (p1, t1) = _seam_tiles()
    acc = TiledFrameAccumulator(frame, 2, MergeConfig())
    assert not acc.is_complete()
    assert acc.remaining() == 2

    assert acc.push_tile(*t0, p0) is True
    assert acc.remaining() == 1
    assert not acc.is_complete()

    assert acc.push_tile(*t1, p1) is True
    assert acc.is_complete()
    assert acc.remaining() == 0

    b, _, _ = acc.finalize()
    assert b.shape[0] >= 1

    # Finalized accumulators raise on reuse.
    with pytest.raises(RuntimeError):
        acc.finalize()


def test_accumulator_out_of_order_equals_in_order():
    frame, (p0, t0), (p1, t1) = _seam_tiles()

    a = TiledFrameAccumulator(frame, 2, MergeConfig())
    a.push_tile(*t0, p0)
    a.push_tile(*t1, p1)
    a_b, a_s, a_c = a.finalize()

    b = TiledFrameAccumulator(frame, 2, MergeConfig())
    b.push_tile(*t1, p1)
    b.push_tile(*t0, p0)
    b_b, b_s, b_c = b.finalize()

    np.testing.assert_allclose(a_b, b_b)
    np.testing.assert_allclose(a_s, b_s)
    np.testing.assert_array_equal(a_c, b_c)


def test_accumulator_dedups_duplicate_push():
    frame, (p0, t0), (p1, t1) = _seam_tiles()
    acc = TiledFrameAccumulator(frame, 2, MergeConfig())
    assert acc.push_tile(*t0, p0) is True
    # Duplicate index is ignored (idempotent under at-least-once delivery).
    assert acc.push_tile(*t0, p0) is False
    assert acc.remaining() == 1


def test_accumulator_finalize_normalized():
    frame = (1280.0, 640.0)
    p = TilePlacement(0, 1, (100.0, 50.0), (640.0, 640.0), frame)
    boxes = _triple([[0.1, 0.1, 0.4, 0.4]], [0.9], [0])

    a = TiledFrameAccumulator(frame, 1, MergeConfig())
    a.push_tile(*boxes, p)
    px, _, _ = a.finalize()

    b = TiledFrameAccumulator(frame, 1, MergeConfig())
    b.push_tile(*boxes, p)
    nb, _, _ = b.finalize_normalized()

    fw, fh = frame
    expected = px[0] / np.array([fw, fh, fw, fh], dtype=np.float32)
    np.testing.assert_allclose(nb[0], expected, atol=1e-4)
