# SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
# SPDX-License-Identifier: Apache-2.0

import threading

import edgefirst.decoder as ef
import numpy as np


def _numpy_to_tensor(arr):
    tensor = ef.Tensor(list(arr.shape), dtype=str(arr.dtype))
    with tensor.map() as m:
        dst = np.frombuffer(m, dtype=arr.dtype).reshape(arr.shape)
        np.copyto(dst, arr)
    return tensor


def test_concurrent_setter_during_decode_raises_not_corrupts():
    """A `decode()` in flight must reject a concurrent threshold write with
    PyO3's borrow error, not race on `Decoder`'s plain fields.

    Regression test for the native-path aliasing UB fixed in `Decoder.decode`
    / `Decoder.decode_proto` (crates/python-common/src/decoder.rs): an
    earlier revision dropped PyO3's runtime borrow flag before entering
    `py.detach`, so a second thread's plain field write (`decoder.
    score_threshold = x`, via `set_score_threshold`) could materialize an
    aliasing `&mut Decoder` while the detached decode still held `&Decoder`
    -- undefined behaviour, not just a race. `decode`/`decode_proto` now
    keep the `PyRef` guard alive on their own stack frame for the whole
    detached region, so the same write must now fail cleanly with pyo3's
    "already borrowed" `RuntimeError` instead.

    One thread hammers `decode()` in a tight loop on a real (if trivial)
    8400-anchor ARA-2 workload -- large enough that each call spends a
    measurable, GIL-released stretch inside the detached region -- while
    the main thread hammers the `score_threshold` setter concurrently.
    With the borrow flag held for the whole call, a collision is expected
    within a handful of iterations, not a rare event to get lucky on; the
    bound below (2000 setter attempts) is generous headroom, not a tight
    deadline.
    """
    with open("testdata/ara2_int8_edgefirst.json") as f:
        config = f.read()
    decoder = ef.Decoder.new_from_json_str(config, 0.25, 0.5)

    model_output = [
        _numpy_to_tensor(np.zeros((1, 2, 8400, 1), dtype=np.int8)),  # xy
        _numpy_to_tensor(np.zeros((1, 2, 8400, 1), dtype=np.int8)),  # wh
        _numpy_to_tensor(np.zeros((1, 80, 8400, 1), dtype=np.uint8)),  # scores
        _numpy_to_tensor(np.zeros((1, 32, 8400, 1), dtype=np.int8)),  # mask_coefs
        _numpy_to_tensor(np.zeros((1, 32, 160, 160), dtype=np.int8)),  # protos
    ]

    stop = threading.Event()
    decode_errors = []

    def decode_loop():
        try:
            while not stop.is_set():
                decoder.decode(model_output)
        except Exception as e:  # pragma: no cover - diagnostic only
            decode_errors.append(e)
            stop.set()

    t = threading.Thread(target=decode_loop)
    t.start()

    borrow_errors = []
    attempts = 0
    try:
        while not stop.is_set() and attempts < 2000:
            attempts += 1
            try:
                decoder.score_threshold = 0.3
            except RuntimeError as e:
                borrow_errors.append(str(e))
                break
    finally:
        stop.set()
        t.join(timeout=10)

    assert not t.is_alive(), "decode thread did not stop"
    assert not decode_errors, f"decode() raised unexpectedly: {decode_errors}"
    assert borrow_errors, (
        f"expected a concurrent score_threshold write to hit pyo3's \"already "
        f'borrowed" error while decode() was in flight, but {attempts} attempts '
        f"all succeeded -- decode() may be leaving the object unguarded again"
    )
    assert "borrow" in borrow_errors[0].lower(), borrow_errors[0]
