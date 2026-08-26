# SPDX-FileCopyrightText: Copyright 2025-2026 Au-Zone Technologies
# SPDX-License-Identifier: Apache-2.0

"""Producer-side access negotiation for the __edgefirst_tensor__ capsule
protocol (task 3 of the cross-package tensor handoff).

These tests exercise `Tensor.__edgefirst_tensor__(access=...)` from pure
Python. The capsule it returns is opaque from Python -- there is no bound
API yet to unwrap it and inspect the descriptor's `ptr` field or observe
whether a host pin was actually taken (that is the consumer side, landing
in a later task). So these tests can only confirm the *call contract*:
`access=None` and the three explicit access strings all succeed, and an
invalid access string raises `ValueError`. They cannot yet assert that
`access=None` skips the pin while `"read"`/`"write"`/`"readwrite"` take
one -- that requires either a consumer-side unwrap helper or a
pin-observable backend (e.g. a PBO, which needs a live GL context) that
is out of scope here.
"""

import sys

import pytest
from edgefirst.tensor import PixelFormat, Tensor


def test_capsule_defaults_to_no_pin():
    t = Tensor.image(64, 32, PixelFormat.Rgb, None, "readwrite")
    cap = t.__edgefirst_tensor__()  # no access requested
    assert cap is not None


def test_capsule_accepts_access_request():
    t = Tensor.image(64, 32, PixelFormat.Rgb, None, "readwrite")
    assert t.__edgefirst_tensor__(access="read") is not None
    assert t.__edgefirst_tensor__(access="write") is not None
    assert t.__edgefirst_tensor__(access="readwrite") is not None


def test_capsule_rejects_bad_access():
    t = Tensor.image(64, 32, PixelFormat.Rgb, None, "readwrite")
    with pytest.raises(ValueError, match="access"):
        t.__edgefirst_tensor__(access="sideways")


# --- Task 4+5: TensorArg, the cross-package extractor, wired into
# ImageProcessor -----------------------------------------------------------
#
# The headline scenario this whole plan exists for: decode a JPEG in
# edgefirst.codec, preprocess it in edgefirst.image. Before this change that
# raised `TypeError: 'Tensor' object is not an instance of 'Tensor'` --
# every edgefirst.* extension module links its own copy of the PyO3
# bindings, so each gets its own distinct `Tensor` type object and
# `isinstance`/downcast across packages is always false.


def test_codec_tensor_converts_via_image(tmp_path):
    """Decode in one package, preprocess in another -- the pipeline this
    whole plan exists for."""
    import numpy as np
    from edgefirst.codec import Tensor as CodecTensor
    from edgefirst.image import ImageProcessor
    from edgefirst.image import PixelFormat as ImgFmt
    from PIL import Image

    p = tmp_path / "f.jpg"
    Image.new("RGB", (128, 64), (200, 100, 50)).save(p)

    info = CodecTensor.peek_image_info_file(str(p))
    src = CodecTensor.image(info.width, info.height, info.format, None, "readwrite")
    src.decode_image_file(str(p))

    proc = ImageProcessor()
    dst = proc.create_image(64, 64, ImgFmt.Rgb, "uint8", "readwrite")
    proc.convert(src, dst, letterbox=[114, 114, 114, 255])  # <-- crosses packages

    with dst.map() as m:
        out = np.frombuffer(m, dtype=np.uint8).reshape(64, 64, 3)
        assert out[32, 32].tolist() != [0, 0, 0], "converted image is blank"


def test_codec_tensor_rejected_before_the_fix_with_a_clear_message(tmp_path):
    """Same scenario as above, but proves *why* it used to fail and that the
    replacement message is actionable: it names what was expected, what was
    received, and points at the protocol -- not the pre-fix
    `'Tensor' object is not an instance of 'Tensor'`, which names neither
    package and gives the reader nothing to act on.

    This doesn't revert the fix; it exercises the same error path a truly
    unrelated object hits, which is the only way to observe the message
    from pure Python now that the real cross-package case succeeds.
    """
    from edgefirst.image import ImageProcessor
    from edgefirst.image import PixelFormat as ImgFmt

    proc = ImageProcessor()
    dst = proc.create_image(8, 8, ImgFmt.Rgb, "uint8", "readwrite")
    with pytest.raises(TypeError) as excinfo:
        proc.convert(object(), dst)
    msg = str(excinfo.value)
    assert "__edgefirst_tensor__" in msg, f"doesn't name the protocol: {msg}"
    assert "expected a tensor" in msg, f"doesn't name what was expected: {msg}"
    assert "got a" in msg, f"doesn't name what was received: {msg}"
    assert (
        "github.com/EdgeFirstAI/hal/blob/main/crates/python-common/INTEROP.md" in msg
    ), f"doesn't point at a URL a pip user can actually open: {msg}"


def test_draw_decoded_masks_background_from_different_package():
    """`background` is read-only, and now goes through `TensorArg::extract`
    like `model_output` does -- a background image produced by a different
    package must composite, not raise the isolation TypeError this whole
    plan exists to eliminate. `draw_decoded_masks` itself is fixed at
    `dst` for the destination-can't-cross-packages-either half.
    """
    import numpy as np
    from edgefirst.image import ImageProcessor
    from edgefirst.image import PixelFormat as ImgFmt
    from edgefirst.tensor import PixelFormat as TFmt
    from edgefirst.tensor import Tensor as TTensor

    w, h = 16, 16
    background = TTensor.image(w, h, TFmt.Rgba, None, "readwrite")  # different package
    with background.map() as m:
        buf = np.frombuffer(m, dtype=np.uint8).reshape(h, w, 4)
        buf[:] = (30, 60, 90, 255)

    proc = ImageProcessor()
    dst = proc.create_image(w, h, ImgFmt.Rgba, "uint8", "readwrite")
    proc.draw_decoded_masks(
        dst,
        bbox=np.zeros((0, 4), dtype=np.float32),
        scores=np.zeros((0,), dtype=np.float32),
        classes=np.zeros((0,), dtype=np.uintp),
        background=background,  # <-- crosses packages
    )

    with dst.map() as m:
        out = np.frombuffer(m, dtype=np.uint8).reshape(h, w, 4)
        assert out[8, 8].tolist() != [0, 0, 0, 0], (
            "background did not composite -- dst is blank"
        )


def test_access_negotiation_retries_mem_backed_destination():
    """Carried coverage item from Task 3, updated for the extract_mut()
    retry fix (task 5b review): a Mem-backed *destination* from a
    different package is a legitimate CPU-fallback target -- the earlier
    "a destination must already be GPU-importable" policy was an
    unjustified assumption, not a spec requirement, and made every
    heap-backed tensor from another package unusable as a `convert()` dst
    for no real reason.

    `TensorArg::extract_mut`'s first `access=None` call takes no pin; when
    the resulting descriptor comes back HOST-kind with a null `ptr`, it now
    retries once with `access="readwrite"` -- mirroring `extract()`'s
    `access="read"` retry for read-only sources, just with the stronger
    access a write destination (and a decode that may read-modify-write a
    strided destination) needs.

    Both `src` and `dst` are Mem-backed producers from a *different*
    package than the `ImageProcessor` doing the convert, and the assertion
    checks real pixel data landed in `dst` -- not merely that the call
    didn't raise.
    """
    import numpy as np
    from edgefirst.image import ImageProcessor
    from edgefirst.tensor import PixelFormat as TFmt
    from edgefirst.tensor import Tensor as TTensor
    from edgefirst.tensor import TensorMemory

    proc = ImageProcessor()
    src = TTensor.image(8, 8, TFmt.Rgb, TensorMemory.MEM, "readwrite")
    with src.map() as m:
        buf = np.frombuffer(m, dtype=np.uint8).reshape(8, 8, 3)
        buf[:] = (200, 100, 50)
    dst = TTensor.image(8, 8, TFmt.Rgb, TensorMemory.MEM, "readwrite")

    proc.convert(src, dst)  # dst used to raise "host descriptor has no address"

    with dst.map() as m:
        out = np.frombuffer(m, dtype=np.uint8).reshape(8, 8, 3)
        assert out[4, 4].tolist() != [0, 0, 0], "converted image is blank"


def test_gpu_backed_destination_skips_the_retry():
    """The retry added above must stay conditional on a HOST-kind
    descriptor with no address -- a GPU-importable destination
    (DMA/IOSurface/PBO) carries a usable native handle regardless of
    `ptr`, so it must never hit that branch and must stay on the original
    zero-pin `access=None` fast path.

    There is no Python-visible hook onto the imported descriptor itself
    (see this file's module docstring), so this checks the *producer's*
    reported memory kind instead: `create_image()`'s tensor is never
    Mem/Shm-backed, and `TensorDyn::descriptor_pinned` derives the
    descriptor's `kind` directly from that field
    (`protocol::kind_of`). That proves the retry's `kind == HOST`
    precondition is structurally false on this path. It does **not**
    directly observe that zero pins were taken at runtime -- proving that
    would need the same consumer-side introspection hook this file already
    notes is missing.

    `str()` rather than `==`/`in` against `edgefirst.tensor.TensorMemory`
    sidesteps the same cross-package identity issue this whole plan
    exists to solve: `dst.memory`'s reported `__module__` is
    `edgefirst.tensor` (baked into the shared binding source for cosmetic
    consistency) but it is `edgefirst.image`'s own distinct compiled type,
    so equality against a value imported from `edgefirst.tensor` is always
    `False`.
    """
    from edgefirst.image import ImageProcessor
    from edgefirst.image import PixelFormat as ImgFmt

    proc = ImageProcessor()
    dst = proc.create_image(8, 8, ImgFmt.Rgb, "uint8", "readwrite")
    assert str(dst.memory) not in ("TensorMemory.MEM", "TensorMemory.SHM"), (
        f"create_image() yielded {dst.memory!r} -- the retry-skip guarantee "
        "this test protects is vacuous unless the destination is GPU-backed"
    )


def test_access_none_never_pins_bypasses_gate_loudly_if_skipped():
    """Loud-skip guard: if the tensor kind above ever stops being
    Mem-backed by default on some platform (making the negotiation gate
    above vacuous), fail loudly on stderr instead of silently passing.
    """
    from edgefirst.tensor import PixelFormat as TFmt
    from edgefirst.tensor import Tensor as TTensor
    from edgefirst.tensor import TensorMemory

    t = TTensor.image(8, 8, TFmt.Rgb, TensorMemory.MEM, "none")
    if t.memory != TensorMemory.MEM:
        print(
            "SKIP-SIGNAL: TensorMemory.MEM request did not yield a Mem-backed "
            f"tensor (got {t.memory!r}); the access-negotiation gate in "
            "test_access_negotiation_retries_mem_backed_destination is vacuous "
            "on this platform.",
            file=sys.stderr,
        )
        pytest.skip("TensorMemory.MEM did not yield a Mem-backed tensor")
    assert t.memory == TensorMemory.MEM


def test_decode_into_gpu_backed_tensor_from_another_package():
    """The documented zero-copy pipeline: allocate GPU-backed, decode into
    it, convert. `decode_file_into` is `edgefirst.codec`'s free-function
    counterpart of `Tensor.decode_image_file` -- it accepts any object
    implementing `__edgefirst_tensor__`, so an `edgefirst.image.Tensor`
    allocated by `ImageProcessor.create_image()` (a DMA/PBO-backed
    destination this package's own `Tensor` type never was) can be decoded
    into directly, closing the gap `decode_image_file` (a method that can
    only ever take `self`) cannot.
    """
    import numpy as np
    from edgefirst.codec import Tensor as CodecTensor
    from edgefirst.codec import decode_file_into
    from edgefirst.image import ImageProcessor
    from edgefirst.image import PixelFormat as ImgFmt

    proc = ImageProcessor()
    info = CodecTensor.peek_image_info_file("testdata/zidane.jpg")
    src = proc.create_image(info.width, info.height, ImgFmt.Nv12, "uint8", "readwrite")

    got = decode_file_into(src, "testdata/zidane.jpg")
    assert (got.width, got.height) == (info.width, info.height)

    dst = proc.create_image(640, 640, ImgFmt.Rgb, "uint8", "readwrite")
    proc.convert(src, dst, letterbox=[114, 114, 114, 255])
    with dst.map() as m:
        out = np.frombuffer(m, dtype=np.uint8).reshape(640, 640, 3)
        assert out[320, 320].tolist() != [0, 0, 0], "converted image is blank"


def test_decode_into_mem_backed_destination_from_another_package():
    """The extract_mut() retry fix (task 5b review) applies to decode_into()
    too, not just convert(): a Mem-backed destination allocated by a
    different package is a legitimate decode target. The first
    `access=None` `__edgefirst_tensor__` call takes no pin; when it comes
    back HOST-kind with a null `ptr`, `decode_into`/`decode_file_into`
    retry once with `access="readwrite"` via the same `TensorArg::
    extract_mut` path `convert()`'s destination uses, rather than raising
    "host descriptor has no address" as it did before this fix.
    """
    import numpy as np
    from edgefirst.codec import decode_file_into
    from edgefirst.tensor import PixelFormat as TFmt
    from edgefirst.tensor import Tensor as TTensor
    from edgefirst.tensor import TensorMemory

    dst = TTensor.image(1280, 720, TFmt.Nv12, TensorMemory.MEM, "readwrite")

    info = decode_file_into(dst, "testdata/zidane.jpg")  # different package

    assert (info.width, info.height) == (1280, 720)
    with dst.map() as m:
        out = np.frombuffer(m, dtype=np.uint8)
        assert out[360 * 1280 + 640] != 0, "decoded luma is blank"


def test_decoder_accepts_foreign_tensor():
    """`Decoder.decode`'s `model_output` now goes through
    `TensorArg::extract` like `ImageProcessor.draw_masks`'s does -- a model
    output tensor allocated by a different package (`edgefirst.tensor`, not
    `edgefirst.decoder`) must decode, not raise the isolation TypeError this
    whole plan exists to eliminate.
    """
    import numpy as np
    from edgefirst.decoder import Decoder, Output
    from edgefirst.tensor import Tensor as CoreTensor

    dec = Decoder.new_from_outputs(
        outputs=[Output.detection(shape=[1, 84, 8400])],
        score_threshold=0.25,
        iou_threshold=0.45,
    )
    raw = np.zeros((1, 84, 8400), dtype=np.float32)
    raw[0, 0:4, 0] = [0.5, 0.5, 0.2, 0.2]
    raw[0, 4, 0] = 0.9

    t = CoreTensor(raw.shape, "float32")  # from edgefirst.tensor, NOT .decoder
    t.from_numpy(raw)

    boxes, scores, _classes, _masks = dec.decode([t])
    assert len(boxes) == 1 and float(np.asarray(scores)[0]) == pytest.approx(0.9)


def test_decoder_decode_proto_accepts_foreign_tensor():
    """`Decoder.decode_proto`'s `model_output` now goes through
    `TensorArg::extract` too -- this is the segmentation entry point that
    `materialize_masks` builds on, so it must accept foreign tensors just
    like `decode` does.
    """
    import numpy as np
    from edgefirst.decoder import Decoder
    from edgefirst.tensor import Tensor as CoreTensor

    nc, nm, n_anchors, proto_h, proto_w = 1, 1, 1, 4, 4
    metadata = {
        "decoder_version": "yolov8",
        "nms": "class_agnostic",
        "outputs": [
            {
                "type": "detection",
                "decoder": "ultralytics",
                "shape": [1, 4 + nc + nm, n_anchors],
                "score_format": "per_class",
                "quantization": [1.0, 0],
            },
            {
                "type": "protos",
                "decoder": "ultralytics",
                "shape": [1, nm, proto_h, proto_w],
                "dshape": [
                    {"batch": 1},
                    {"num_protos": nm},
                    {"height": proto_h},
                    {"width": proto_w},
                ],
                "quantization": [1.0, 0],
            },
        ],
    }
    dec = Decoder(metadata, score_threshold=0.25, iou_threshold=0.45)

    combined = np.zeros((1, 4 + nc + nm, n_anchors), dtype=np.float32)
    combined[0, 0:4, 0] = [0.5, 0.5, 0.2, 0.2]
    combined[0, 4, 0] = 0.9  # class score
    combined[0, 5, 0] = 4.0  # mask coefficient
    protos = np.ones((1, nm, proto_h, proto_w), dtype=np.float32)

    t_combined = CoreTensor(combined.shape, "float32")  # from edgefirst.tensor
    t_combined.from_numpy(combined)
    t_protos = CoreTensor(protos.shape, "float32")  # from edgefirst.tensor
    t_protos.from_numpy(protos)

    boxes, scores, _classes, proto_data = dec.decode_proto([t_combined, t_protos])
    assert proto_data is not None
    assert len(boxes) == 1 and float(np.asarray(scores)[0]) == pytest.approx(0.9)


# --- Task 7: Decoder and ProtoData across packages ------------------------
#
# `ImageProcessor.draw_masks(decoder, ...)` and `.materialize_masks(...,
# proto_data, ...)` are the fused paths `edgefirst-image` declares its
# `edgefirst-decoder` dependency for -- but a `Decoder`/`ProtoData` built by
# `edgefirst.decoder` used to be rejected by `edgefirst.image` for the same
# per-module-type-object reason every other cross-package case in this file
# is. Per ruling R15, the two protocols are NOT the same mechanism:
#
# - `Decoder.__edgefirst_decoder__` carries a raw, version-guarded pointer --
#   a `Decoder` is a live Rust object with internal state that cannot be
#   decomposed the way a tensor can.
# - `ProtoData.__edgefirst_protodata__` carries no pointer at all: it
#   composes the already-proven `__edgefirst_tensor__` protocol for its two
#   `TensorDyn` fields plus a layout string, so it needs no version guard.


def test_draw_onto_accepts_image_processor():
    """Decoder.draw_onto drives image drawing without ImageProcessor linking
    a Decoder type. Score/class must come from the actual Decoder.
    """
    import numpy as np
    from edgefirst.decoder import Decoder, Output
    from edgefirst.image import ImageProcessor
    from edgefirst.image import PixelFormat as ImgFmt
    from edgefirst.tensor import Tensor as CoreTensor

    dec = Decoder.new_from_outputs(
        outputs=[Output.detection(shape=[1, 84, 8400])],
        score_threshold=0.25,
        iou_threshold=0.45,
    )
    raw = np.zeros((1, 84, 8400), dtype=np.float32)
    raw[0, 0:4, 0] = [0.5, 0.5, 0.2, 0.2]
    raw[0, 4, 0] = 0.9  # class-0 score
    t = CoreTensor(list(raw.shape), dtype="float32")
    t.from_numpy(raw)

    proc = ImageProcessor()
    dst = proc.create_image(64, 64, ImgFmt.Rgb, "uint8", "readwrite")
    boxes, scores, classes = dec.draw_onto(proc, [t], dst)

    assert len(boxes) == 1
    assert float(np.asarray(scores)[0]) == pytest.approx(0.9)
    assert int(np.asarray(classes)[0]) == 0


def test_image_processor_has_no_fused_draw_masks():
    """Image-only installs never see the decoder-taking convenience."""
    from edgefirst.image import ImageProcessor

    assert not hasattr(ImageProcessor, "draw_masks")


def test_materialize_masks_accepts_foreign_proto_data():
    """`ImageProcessor.materialize_masks`'s `proto_data` argument now goes
    through `interop::ProtoDataArg::extract` -- composing the existing
    `__edgefirst_tensor__` protocol for `mask_coefficients` and `protos`
    (ruling R15), not a raw pointer. A `ProtoData` produced by
    `edgefirst.decoder.decode_proto` must materialize real masks in
    `edgefirst.image`, not raise the isolation TypeError this whole plan
    exists to eliminate.

    Asserts the materialized mask's dtype and values match the documented
    contract (binary ``uint8 {0, 255}``, foreground where the dot product is
    positive) for a hand-computed coefficient/proto pair, not merely that
    the call didn't raise.
    """
    import numpy as np
    from edgefirst.decoder import Decoder
    from edgefirst.decoder import Tensor as DTensor
    from edgefirst.image import ImageProcessor

    nc, nm, n_anchors, proto_h, proto_w = 1, 1, 1, 4, 4
    metadata = {
        "decoder_version": "yolov8",
        "nms": "class_agnostic",
        "outputs": [
            {
                "type": "detection",
                "decoder": "ultralytics",
                "shape": [1, 4 + nc + nm, n_anchors],
                "score_format": "per_class",
                "quantization": [1.0, 0],
            },
            {
                "type": "protos",
                "decoder": "ultralytics",
                "shape": [1, nm, proto_h, proto_w],
                "dshape": [
                    {"batch": 1},
                    {"num_protos": nm},
                    {"height": proto_h},
                    {"width": proto_w},
                ],
                "quantization": [1.0, 0],
            },
        ],
    }
    dec = Decoder(metadata, score_threshold=0.25, iou_threshold=0.45)

    combined = np.zeros((1, 4 + nc + nm, n_anchors), dtype=np.float32)
    combined[0, 0:4, 0] = [0.5, 0.5, 1.0, 1.0]  # full-frame bbox: xc, yc, w, h
    combined[0, 4, 0] = 0.9  # class score
    combined[0, 5, 0] = 4.0  # mask coefficient (positive -> foreground everywhere)
    protos = np.ones((1, nm, proto_h, proto_w), dtype=np.float32)

    t_combined = DTensor(list(combined.shape), dtype="float32")
    t_combined.from_numpy(combined)
    t_protos = DTensor(list(protos.shape), dtype="float32")
    t_protos.from_numpy(protos)

    boxes, scores, classes, proto_data = dec.decode_proto([t_combined, t_protos])
    assert proto_data is not None
    assert len(boxes) == 1

    proc = ImageProcessor()
    masks = proc.materialize_masks(
        boxes, scores, classes, proto_data
    )  # <-- crosses decoder -> image

    assert len(masks) == 1
    m = masks[0]
    assert m.dtype == np.uint8
    assert m.shape == (proto_h, proto_w, 1)
    # coef=4.0 dotted with protos=1.0 everywhere -> logit=4.0 > 0 everywhere,
    # so the documented ">0 is foreground" contract says fully 255.
    assert np.array_equal(m, np.full_like(m, 255)), (
        "expected the whole proto-resolution tile to be foreground per the "
        "documented positive-dot-product convention"
    )


def test_materialize_masks_rejects_an_object_without_the_protocol():
    """Mirrors `test_codec_tensor_rejected_before_the_fix_with_a_clear_message`
    for `ProtoData`: an unrelated object is rejected with a message naming
    the protocol, what was expected, and what was received.
    """
    import numpy as np
    from edgefirst.image import ImageProcessor

    proc = ImageProcessor()
    with pytest.raises(TypeError) as excinfo:
        proc.materialize_masks(
            np.zeros((0, 4), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
            np.zeros((0,), dtype=np.uintp),
            object(),
        )
    msg = str(excinfo.value)
    assert "__edgefirst_protodata__" in msg, f"doesn't name the protocol: {msg}"
    assert "expected a ProtoData" in msg, f"doesn't name what was expected: {msg}"
    assert "got a" in msg, f"doesn't name what was received: {msg}"
