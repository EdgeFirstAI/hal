# SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
# SPDX-License-Identifier: Apache-2.0

import statistics
import threading
import time

import edgefirst.decoder as ed
import numpy as np
import pytest
from edgefirst.decoder import MergeConfig, TiledFrameAccumulator, merge_tiled_detections
from edgefirst.image import (
    ImageProcessor,
    MaskResolution,
    PixelFormat,
    Tensor,
    TilePlacement,
)
from edgefirst.tensor import TensorMemory


def _python_throughput(duration_s, run_alongside=None):
    """Measure the rate, in iterations per second, of a pure-Python counting loop on a
    background thread.

    Runs for about `duration_s` seconds while this thread sleeps or, when given, calls
    `run_alongside` (a zero-arg callable) in a tight loop. The count is divided by the
    measured length of the run.
    """
    stop = threading.Event()
    counted = []

    def count():
        n = 0
        while not stop.is_set():
            n += 1
        counted.append(n)

    t = threading.Thread(target=count)
    t.start()
    t0 = time.perf_counter()
    if run_alongside is None:
        time.sleep(duration_s)
    else:
        while time.perf_counter() - t0 < duration_s:
            run_alongside()
    t1 = time.perf_counter()
    stop.set()
    t.join()
    elapsed = t1 - t0
    return counted[0] / elapsed


def _gil_holding_control(duration_s):
    """Return a `run_alongside` callable that holds the GIL for `duration_s` seconds."""

    def control():
        t0 = time.perf_counter()
        x = 0
        while time.perf_counter() - t0 < duration_s:
            x += 1

    return control


def _assert_releases_gil(
    op_name, run_alongside, duration_s=0.4, min_gap=0.1, attempts=3, n_rounds=6
):
    """Assert that an unrelated Python thread keeps making progress while
    `run_alongside` runs.

    Each attempt alternates `n_rounds` rounds of `duration_s / n_rounds` seconds between
    `run_alongside` and the GIL-holding control, each against a fresh idle baseline, and
    passes when `median(real) - median(control) > min_gap`. A failure reports the best
    of `attempts`.
    """
    slice_s = duration_s / n_rounds
    control_op = _gil_holding_control(slice_s)
    observed = []
    for _ in range(attempts):
        real_ratios = []
        control_ratios = []
        for round_idx in range(n_rounds):
            baseline = _python_throughput(slice_s)
            if round_idx % 2 == 0:
                concurrent = _python_throughput(slice_s, run_alongside=run_alongside)
                real_ratios.append(concurrent / baseline)
            else:
                concurrent = _python_throughput(slice_s, run_alongside=control_op)
                control_ratios.append(concurrent / baseline)
        median_real = statistics.median(real_ratios)
        median_control = statistics.median(control_ratios)
        gap = median_real - median_control
        observed.append((gap, median_real, median_control, real_ratios, control_ratios))
        if gap > min_gap:
            return

    best_gap, best_real, best_control, best_real_ratios, best_control_ratios = max(
        observed, key=lambda o: o[0]
    )
    assert best_gap > min_gap, (
        f"{op_name} did not clearly beat a same-attempt GIL-holding control -- "
        f"best gap {best_gap:.2%} (real median {best_real:.2%} vs. control median "
        f"{best_control:.2%}) is not > {min_gap:.0%}; {op_name} appears to hold the "
        f"GIL (real per-round ratios={[f'{r:.2%}' for r in best_real_ratios]}, "
        f"control per-round ratios={[f'{r:.2%}' for r in best_control_ratios]}; "
        f"best of {attempts} attempts, gaps of "
        f"{', '.join(f'{o[0]:.2%}' for o in observed)})"
    )


def _numpy_to_tensor(arr):
    """Copy a numpy array into a HAL `Tensor`."""
    tensor = ed.Tensor(list(arr.shape), dtype=str(arr.dtype))
    with tensor.map() as m:
        dst = np.frombuffer(m, dtype=arr.dtype).reshape(arr.shape)
        np.copyto(dst, arr)
    return tensor


def _ara2_model_output():
    """Model-output tensors for the ARA-2 int8 config that decode to a detection."""
    rng = np.random.default_rng(0)
    xy = _numpy_to_tensor(np.zeros((1, 2, 8400, 1), dtype=np.int8))
    wh = _numpy_to_tensor(np.zeros((1, 2, 8400, 1), dtype=np.int8))
    scores = _numpy_to_tensor(np.full((1, 80, 8400, 1), 127, dtype=np.uint8))
    mask_coefs = _numpy_to_tensor(np.full((1, 32, 8400, 1), 20, dtype=np.int8))
    protos = _numpy_to_tensor(rng.integers(-127, 127, (1, 32, 160, 160), dtype=np.int8))
    return [xy, wh, scores, mask_coefs, protos]


def _ara2_decoder():
    with open("testdata/ara2_int8_edgefirst.json") as f:
        config = f.read()
    return ed.Decoder.new_from_json_str(config, 0.25, 0.5)


def _modelpack_split_model_output():
    """Model-output tensors for the ModelPack split fixture."""
    output0 = _numpy_to_tensor(
        np.fromfile("testdata/modelpack_split_17x30x18.bin", dtype=np.uint8).reshape(
            1, 17, 30, 18
        )
    )
    output1 = _numpy_to_tensor(
        np.fromfile("testdata/modelpack_split_9x15x18.bin", dtype=np.uint8).reshape(
            1, 9, 15, 18
        )
    )
    return [output0, output1]


def _modelpack_split_decoder():
    with open("testdata/modelpack_split.yaml") as f:
        config = f.read()
    return ed.Decoder.new_from_yaml_str(config, 0.45, 0.45)


def _synthetic_detections(n):
    """`n` synthetic detections with random positions and sizes."""
    rng = np.random.default_rng(0)
    bbox = rng.uniform(0, 4000, size=(n, 4)).astype(np.float32)
    bbox[:, 2] = bbox[:, 0] + rng.uniform(10, 200, n)
    bbox[:, 3] = bbox[:, 1] + rng.uniform(10, 200, n)
    scores = rng.uniform(0, 1, n).astype(np.float32)
    classes = rng.integers(0, 10, n).astype(np.uintp)
    return bbox, scores, classes


def test_convert_releases_the_gil():
    """An unrelated Python thread keeps making progress during `convert()`."""
    p = ImageProcessor()
    w, h = 3840, 2160
    src = Tensor.image(w, h, PixelFormat.Rgb, mem=TensorMemory.MEM, access="readwrite")
    src.from_numpy(np.zeros((h, w, 3), np.uint8))
    dst = Tensor.image(
        w // 2, h // 2, PixelFormat.Rgb, mem=TensorMemory.MEM, access="readwrite"
    )

    duration_s = 0.4
    baseline = _python_throughput(duration_s)
    concurrent = _python_throughput(
        duration_s, run_alongside=lambda: p.convert(src, dst)
    )

    ratio = concurrent / baseline
    assert ratio > 0.5, (
        f"an unrelated Python thread only made {ratio:.2%} of its normal progress "
        f"while convert() was running -- convert() appears to hold the GIL "
        f"(baseline={baseline}, concurrent={concurrent})"
    )


def test_convert_with_pbo_backed_source_releases_the_gil():
    """A `convert()` with a PBO-backed source releases the GIL."""
    p = ImageProcessor()
    w, h = 3840, 2160
    src = p.create_image(w, h, PixelFormat.Rgb, "uint8", "readwrite")
    if src.memory != TensorMemory.PBO:
        pytest.skip(
            f"this environment did not fall back to PBO for create_image() (got "
            f"{src.memory!r}); nothing PBO-specific to exercise"
        )
    dst = Tensor.image(
        w // 2, h // 2, PixelFormat.Rgb, mem=TensorMemory.MEM, access="readwrite"
    )

    duration_s = 0.4
    baseline = _python_throughput(duration_s)
    concurrent = _python_throughput(
        duration_s, run_alongside=lambda: p.convert(src, dst)
    )

    ratio = concurrent / baseline
    assert ratio > 0.5, (
        f"an unrelated Python thread only made {ratio:.2%} of its normal progress "
        f"while convert() (PBO-backed source) was running -- convert() appears to hold "
        f"the GIL, meaning the PBO exclusion in can_detach()/reconstructible() is still "
        f"in effect (baseline={baseline}, concurrent={concurrent})"
    )


# ---------------------------------------------------------------------------
# The decoder, mask and tiling-postprocessing entry points.
# ---------------------------------------------------------------------------


def test_decode_tracked_releases_the_gil():
    """`Decoder.decode_tracked` does not stall an unrelated Python thread."""
    decoder = _modelpack_split_decoder()
    from edgefirst.tracker import ByteTrack

    tracker = ByteTrack()
    model_output = _modelpack_split_model_output()

    def run():
        run.i += 1
        decoder.decode_tracked(tracker, run.i, model_output)

    run.i = 0
    _assert_releases_gil("decode_tracked()", run)


def test_materialize_masks_releases_the_gil():
    """`ImageProcessor.materialize_masks` does not stall an unrelated Python thread."""
    decoder = _ara2_decoder()
    model_output = _ara2_model_output()
    boxes, scores, classes, proto_data = decoder.decode_proto(model_output)
    assert len(boxes) > 0, "fixture produced no detections to materialize masks for"

    processor = ImageProcessor()

    def run():
        processor.materialize_masks(
            boxes,
            scores,
            classes,
            proto_data,
            resolution=MaskResolution.Scaled(1920, 1080),
        )

    _assert_releases_gil("materialize_masks()", run)


def test_draw_onto_crosses_image_and_decoder():
    """Decoder.draw_onto draws through an ImageProcessor."""
    decoder = _ara2_decoder()
    model_output = _ara2_model_output()
    processor = ImageProcessor()
    dst = Tensor.image(
        1920, 1080, PixelFormat.Rgba, mem=TensorMemory.MEM, access="readwrite"
    )
    decoder.draw_onto(processor, model_output, dst)


def test_push_tile_releases_the_gil():
    """`TiledFrameAccumulator.push_tile` does not stall an unrelated Python thread."""
    bbox, scores, classes = _synthetic_detections(500)
    acc = TiledFrameAccumulator(
        frame_dims=(4000.0, 4000.0), tiles_total=1_000_000, cfg=MergeConfig()
    )

    def run():
        run.i += 1
        placement = TilePlacement(
            index=run.i,
            count=1_000_000,
            origin=(0.0, 0.0),
            crop_size=(4000.0, 4000.0),
            frame_dims=(4000.0, 4000.0),
        )
        acc.push_tile(bbox, scores, classes, placement)

    run.i = 0
    _assert_releases_gil("TiledFrameAccumulator.push_tile()", run)


def _populated_accumulator(n_boxes=3000):
    bbox, scores, classes = _synthetic_detections(n_boxes)
    placement = TilePlacement(
        index=0,
        count=1,
        origin=(0.0, 0.0),
        crop_size=(4000.0, 4000.0),
        frame_dims=(4000.0, 4000.0),
    )
    acc = TiledFrameAccumulator(
        frame_dims=(4000.0, 4000.0), tiles_total=1, cfg=MergeConfig()
    )
    acc.push_tile(bbox, scores, classes, placement)
    return acc


def test_finalize_releases_the_gil():
    """`TiledFrameAccumulator.finalize` does not stall an unrelated Python thread."""
    accs = iter([_populated_accumulator() for _ in range(300)])

    def run():
        next(accs).finalize()

    _assert_releases_gil("TiledFrameAccumulator.finalize()", run)


def test_finalize_normalized_releases_the_gil():
    """`TiledFrameAccumulator.finalize_normalized` does not stall another thread."""
    accs = iter([_populated_accumulator() for _ in range(300)])

    def run():
        next(accs).finalize_normalized()

    _assert_releases_gil("TiledFrameAccumulator.finalize_normalized()", run)


def test_merge_tiled_detections_releases_the_gil():
    """`merge_tiled_detections` does not stall an unrelated Python thread."""
    bbox, scores, classes = _synthetic_detections(3000)
    cfg = MergeConfig()

    def run():
        merge_tiled_detections(bbox, scores, classes, cfg)

    _assert_releases_gil("merge_tiled_detections()", run)
