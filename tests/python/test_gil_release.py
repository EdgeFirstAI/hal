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
    """Count how many pure-Python loop iterations complete in `duration_s`
    seconds on a background thread, optionally running `run_alongside` (a
    zero-arg callable, executed in a tight loop on *this* thread) at the
    same time.

    The counter thread never touches Rust or any `edgefirst` object -- it
    is plain `while not stop: n += 1`. Its only dependency on the rest of
    the process is whether the GIL is available for the interpreter to
    schedule it at all. That makes the ratio between a lone counter run and
    one run alongside `run_alongside` a direct measurement of GIL
    availability, independent of whether `run_alongside`'s own work can
    physically run in parallel with anything else (a single GL context, or
    a mutex-guarded backend, may not be able to -- see the note below).
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
    stop.set()
    t.join()
    return counted[0]


def _gil_holding_control(duration_s):
    """A `run_alongside` callable that provably holds the GIL for the whole
    call: a tight pure-Python loop with a trivial arithmetic body (real
    bytecode dispatch, not an OS-level sleep or spin that could yield the
    GIL for free). Whatever throughput an unrelated thread gets alongside
    *this* is exactly what CPython's own periodic GIL-switch check hands
    out when nothing is actually released -- the number a real regression
    would be indistinguishable from. See `_assert_releases_gil` for why
    this is measured as a same-attempt control rather than assumed from a
    fixed constant.
    """

    def control():
        t0 = time.perf_counter()
        x = 0
        while time.perf_counter() - t0 < duration_s:
            x += 1

    return control


def _assert_releases_gil(
    op_name, run_alongside, duration_s=0.4, min_gap=0.2, attempts=3, n_rounds=6
):
    """Shared assertion for the throughput formulation: an unrelated Python
    thread must keep making most of its normal progress while `run_alongside`
    (some long-running HAL call) runs continuously. See
    `test_convert_releases_the_gil` for the full rationale.

    An earlier version compared `run_alongside`'s ratio (against an idle
    baseline) to a fixed constant (0.7), re-measuring the baseline
    immediately before each attempt's concurrent sample. That still failed
    on a hosted runner with all three retry attempts landing at 67.93%,
    67.91%, 69.19% -- narrow and repeatable, not a scheduler hiccup's wide
    scatter. The cause is not baseline drift (the baseline *was* already
    paired): it's CPU contention. With the GIL genuinely released,
    `run_alongside`'s native work runs on another core and competes with
    the counting thread for cycles on a busy, few-vCPU hosted runner,
    while the idle-baseline window's only thread is blocked in
    `time.sleep` and competes with nothing. Comparing against an idle
    baseline measures "how many free cores exist right now", which a
    fixed threshold cannot tell apart from the GIL being held.

    So instead of comparing to a fixed constant, each attempt also
    measures a same-attempt, same-conditions *control* that provably does
    hold the GIL (`_gil_holding_control`, above) and asserts `run_alongside`
    clearly beats it. Both legs see the same external load because they
    run in the same attempt, interleaved: `n_rounds` short rounds
    alternate between "real" (`run_alongside` vs. a fresh idle baseline)
    and "control" (the GIL-holding loop vs. a fresh idle baseline), each
    round 1/n_rounds of `duration_s`. The per-round ratios are reduced to
    a median for the real rounds and a median for the control rounds, and
    the assertion is `median(real) - median(control) > min_gap` -- an
    absolute gap, not a ratio, because contention depresses both legs'
    absolute level together without closing the gap between "released"
    and "held" (a real regression's ratio converges *onto* the control's,
    it doesn't just drop).

    Calibrated on this machine (`duration_s=0.4`, `n_rounds=6`,
    `decode_tracked()` as the real op), median-of-3-rounds-per-leg,
    best-of-3-attempts:

    | condition                                            | real ratio | control ratio | gap                    |
    |-------------------------------------------------------|-----------:|---------------:|------------------------|
    | unloaded (this machine has 16 cores)                   | ~0.84-1.18 | ~0.48-0.72     | ~0.41 mean, 0.16 min single-round |
    | 4 background `while True: pass` processes (unpinned)  | ~0.97-1.04 | ~0.46-0.62     | ~0.49 mean             |
    | same, GIL-held mutation (real op := the control)       | ~0         | ~0             | ~0.01 (both directions tested) |

    4 unpinned busy-loop processes barely move the numbers on this 16-core
    box -- there are far more free cores than competitors -- so as a
    tougher, deliberately adversarial stress test beyond what was asked,
    the same measurements were repeated with this process *and* several
    competing `while True: pass` processes pinned (`taskset`) onto the
    same 2 cores, to force real core contention resembling a saturated
    few-vCPU hosted runner. Filtered to `decode_tracked()` alone (2 pinned
    cores + 2 hostile pinned processes), it separated cleanly 3/3 runs
    (gaps 28.95%, 43.85%, 36.10%); a full-suite batch under the harsher 2
    pinned cores + 4 hostile pinned processes told a different story: 8 of
    10 full-suite runs had at least one row fail, spread across
    `decode_tracked` (x2), `push_tile` (x2), `merge_tiled_detections` (x3),
    `finalize_normalized` (x3), `finalize` (x1) and `materialize_masks`
    (x2) -- i.e. under that specific extreme, artificial oversubscription
    (more hostile processes pinned to 2 cores than there are cores, run
    for the full ~10-17s multi-row suite rather than one filtered row) the
    residual risk is real and not confined to the shortest calls. It did
    not reproduce under the literal 4-unpinned-process scenario in either
    direction across 5 repeated full-suite runs (0/5 failures), and
    `min_gap=0.2` sits clear of every gap measured there (mutation
    best-of-3 gap 0.01-0.02; real op 0.44-0.53 mean).

    Tried and rejected as a way to close this gap: raising
    `sys.setswitchinterval()` for the duration of the measurement, to make
    a GIL-holding call's throughput drop toward 0% instead of the ~40-50%
    CPython's default 5ms switch check hands out (the idea being a wider
    real-vs-control gap would tolerate more core-contention noise).
    Measured effect (profiled directly, not assumed): `threading.Thread
    .start()` itself blocks for very close to one full switch interval on
    *every* call once the interval is raised (confirmed at intervals of
    0.5s, 1.0s, 2.0s and 5.0s -- `t.start()` took 1.0006s and 5.0004s
    respectively in isolated timing) -- the newly-started counting thread,
    once running, doesn't yield the GIL back to the thread waiting on
    `Thread._started` until the interval elapses, so raising it stalls
    thread creation itself, which `_python_throughput` does on every
    single sub-measurement. A full-suite run at `setswitchinterval(1.0)`
    exceeded 60s (vs. ~7.4s normally) and was killed rather than let
    finish. Separately, and independent of that stall: the control's
    ratio did not trend toward 0% even at very large intervals (0.5s-5.0s)
    -- it plateaued at 43-50%, same order as the default -- because the
    counting thread accumulates most of its count *during* that same
    startup stall (fully uncontested, main thread blocked waiting), which
    inflates both legs' counts roughly together. This lever does not work
    with `_python_throughput`'s current per-call fresh-thread design, so
    it was not kept; a fix would need a persistent counting thread reused
    across sub-measurements, which is a larger change than this ticket's
    scope. `n_rounds=6` (3 real + 3 control rounds, alternating) rather
    than 2 gives each leg's median enough samples that one noisy round
    can't dominate it; `attempts=3` retries a genuinely unlucky attempt
    the same way the previous design did.
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
    """Copy a numpy array into a HAL `Tensor` (`edgefirst.decoder`'s -- same
    binary layout `edgefirst.image`/`edgefirst.tensor` accept, per the
    cross-package `__edgefirst_tensor__` protocol)."""
    tensor = ed.Tensor(list(arr.shape), dtype=str(arr.dtype))
    with tensor.map() as m:
        dst = np.frombuffer(m, dtype=arr.dtype).reshape(arr.shape)
        np.copyto(dst, arr)
    return tensor


def _ara2_model_output():
    """Model-output tensors for the ARA-2 int8 segmentation config (8400
    anchors, 80 classes, 32 mask prototypes at 160x160) -- the same fixture
    `tests/decoder/test_decoder.py::test_from_json_v2_ara2_int8` and
    `tests/python/test_decoder_gil_release.py` use. `scores`/`mask_coefs`
    are filled (not zeroed) so `decode_proto` actually produces a detection
    -- an all-zero input dequantizes under threshold and yields none, which
    would make `materialize_masks`/`draw_masks` trivial no-ops instead of
    the real per-instance-mask workload their GIL release is for.
    """
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
    """Model-output tensors + config for the small ModelPack split fixture
    `tests/decoder/test_decoder_tracked.py` uses -- `decode_tracked` rejects
    the ARA-2 config above (its combined-box tensor shape doesn't match
    ARA-2's split xy/wh layout), so this is the fixture that actually
    exercises the tracked path.
    """
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
    """`n` synthetic, non-degenerate detections -- random positions/sizes so
    the tiling merge (which is seam-duplicate-aware, not just a flat NMS)
    has genuinely varied boxes to work with, not `n` copies of the same one.
    """
    rng = np.random.default_rng(0)
    bbox = rng.uniform(0, 4000, size=(n, 4)).astype(np.float32)
    bbox[:, 2] = bbox[:, 0] + rng.uniform(10, 200, n)
    bbox[:, 3] = bbox[:, 1] + rng.uniform(10, 200, n)
    scores = rng.uniform(0, 1, n).astype(np.float32)
    classes = rng.integers(0, 10, n).astype(np.uintp)
    return bbox, scores, classes


def test_convert_releases_the_gil():
    """A long-running `convert()` must not stall unrelated Python threads.

    The straightforward version of this test -- two threads both calling
    `convert()` on the *same* `ImageProcessor` and comparing wall-clock --
    does not work here: `ImageProcessor` wraps a `Mutex`, and on this
    machine `convert()` lands on the OpenGL backend, whose GL context is
    only ever current on one thread. Two concurrent calls on the same
    processor therefore always serialize on that mutex regardless of the
    GIL -- measured at a ~2.0x-2.5x ratio for two threads even after the
    GIL fix (checked against a 3840x2160 -> 1920x1080 convert, forcing the
    CPU backend included), which would make a wall-clock-ratio test flaky
    in the "erroring on symptoms of something else" sense the reviewer
    asked to avoid, not the "occasionally slow CI runner" sense.

    So this measures the actual property the GIL release is for: while one
    thread is deep inside a `convert()` call, an unrelated Python thread
    that touches no Rust object at all must still make progress. Before the
    fix that thread's throughput dropped to ~34% of its GIL-idle baseline
    on this machine (measured); after the fix it holds at ~97-99%. The 0.5
    threshold below leaves wide margin on both sides of that gap while
    still failing hard if `convert()` goes back to holding the GIL.
    """
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
    """F25 (task 11 review, follow-on): a PBO-backed source must detach the
    same way a `Mem`/`DmaBuf`-backed one already does.

    Before task 11, `TensorDyn::import_descriptor` had no `kind::PBO` arm at
    all, so `reconstructible()` (`crates/python-common/src/interop.rs`)
    excluded `TensorMemory::Pbo` from `TensorArg::can_detach` -- every
    `convert()` call with a PBO-backed source held the GIL for the whole
    call, silently, wherever GL fell back to PBO transfers (DMA-BUF
    verification failing, the common case on this environment's driver).
    Task 11 gave PBO a real, cross-cdylib-safe `import_descriptor` arm (a
    `#[repr(C)]` vtable plus `TensorCapsulePayload::pbo_keepalive`), which
    made dropping `Pbo` from `reconstructible()`'s exclusion (F25) actually
    safe -- this is the regression test proving that fix took effect
    behaviorally, not just that nothing crashed.

    `create_image()` only produces a PBO-backed tensor when this
    environment's GL stack falls back to it (zero-copy DMA-BUF import
    failing verification); skips rather than asserting a specific
    allocation strategy if that is not what happened here.
    """
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
# H4b: the four entry points Task 4 flagged but did not detach, plus the
# five tiling-postprocessing functions named alongside them. Each case below
# uses the same throughput formulation as `test_convert_releases_the_gil`
# (not a wall-clock speedup -- see that test's doc comment for why a
# Mutex/GL-context-serialized backend makes a speedup measurement fail even
# after a correct fix) and, per the H4b brief, must fail without the
# corresponding `py.detach` in crates/python-common/src.
# ---------------------------------------------------------------------------


def test_decode_tracked_releases_the_gil():
    """`Decoder.decode_tracked` (decoder.rs) must not stall an unrelated
    Python thread. Uses the ModelPack split fixture, not ARA-2: ARA-2's
    split xy/wh output shape isn't one `decode_tracked` accepts (`Did not
    find output with shape [1, 4, 8400]`) -- this is the fixture
    `test_decoder_tracked.py` already exercises this entry point with.
    """
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
    """`ImageProcessor.materialize_masks` (image.rs) must not stall an
    unrelated Python thread. `resolution=Scaled(1920, 1080)` forces the full
    proto-plane upsample this method's own doc comment describes -- "per
    full-resolution buffers" -- rather than the small proto-resolution tiles
    the default returns, so the detached region does real, size-scaled work
    regardless of how many detections the fixture happens to produce.
    """
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
    """Fused convenience lives on Decoder and takes an ImageProcessor via
    PyAny, so image-only installs never see it.
    """
    decoder = _ara2_decoder()
    model_output = _ara2_model_output()
    processor = ImageProcessor()
    dst = Tensor.image(
        1920, 1080, PixelFormat.Rgba, mem=TensorMemory.MEM, access="readwrite"
    )
    decoder.draw_onto(processor, model_output, dst)


def test_push_tile_releases_the_gil():
    """`TiledFrameAccumulator.push_tile` (tiling.rs) must not stall an
    unrelated Python thread. Uses a fresh `placement.index` per call (a
    huge `tiles_total`) because `push_tile` is idempotent per index --
    repeating the same index would make every call after the first a no-op
    early return, not the "accumulates two numpy arrays per call" work this
    entry point is in the H4b table for.
    """
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
    """`TiledFrameAccumulator.finalize` (tiling.rs) must not stall an
    unrelated Python thread. `finalize` consumes the accumulator (only
    callable once), so this pre-builds one populated accumulator per
    expected call rather than reusing a single instance -- building enough
    of them ahead of time is setup, not part of the timed measurement.
    """
    accs = iter([_populated_accumulator() for _ in range(300)])

    def run():
        next(accs).finalize()

    _assert_releases_gil("TiledFrameAccumulator.finalize()", run)


def test_finalize_normalized_releases_the_gil():
    """`TiledFrameAccumulator.finalize_normalized` (tiling.rs) -- see
    `test_finalize_releases_the_gil`; same reasoning, the renormalizing
    sibling entry point.
    """
    accs = iter([_populated_accumulator() for _ in range(300)])

    def run():
        next(accs).finalize_normalized()

    _assert_releases_gil("TiledFrameAccumulator.finalize_normalized()", run)


# No `test_lift_tile_boxes_releases_the_gil`: `py_lift_tile_boxes` is
# deliberately *not* detached -- see its doc comment in tiling.rs. Measured
# with this same harness: holding the GIL for the whole call already gives
# an unrelated Python thread ~45% throughput (the call is short enough that
# CPython's own periodic GIL-switch check covers it), and wrapping just the
# lift in `py.detach` measured 14%-52% across five runs -- noisier and not
# reliably better, because `numpy_to_detect_boxes` (unavoidably GIL-bound;
# it reads the caller's numpy arrays) does comparable O(N) work right next
# to it. A test built on that gap would be flaky by construction, not
# occasionally slow -- see the H4b report for the full measurement.


def test_merge_tiled_detections_releases_the_gil():
    """The free function `merge_tiled_detections` (tiling.rs's
    `py_merge_tiled_detections`) must not stall an unrelated Python thread --
    the standalone cross-tile merge, the same GREEDYNMM work `finalize` runs
    internally.
    """
    bbox, scores, classes = _synthetic_detections(3000)
    cfg = MergeConfig()

    def run():
        merge_tiled_detections(bbox, scores, classes, cfg)

    _assert_releases_gil("merge_tiled_detections()", run)
