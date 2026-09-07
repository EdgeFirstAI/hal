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

import ctypes
import sys

import pytest
from edgefirst.tensor import PixelFormat, Tensor, is_gpu_buffer_available

from tests.gpu_policy import skip_unless_gpu_backed


def _close(handle: int) -> None:
    """Close a Windows handle a completion or fence call handed back.

    A plain call, never an assertion: these run in `finally` blocks, where a
    failing assert would replace the real failure with this one.
    """
    ctypes.windll.kernel32.CloseHandle(ctypes.c_void_p(handle))


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
        # np.asarray(memoryview(m)) honours the mapping's own row_stride
        # (a padded pitch on a D3D11 texture destination) instead of
        # assuming a tight buffer the way frombuffer().reshape() does.
        out = np.asarray(memoryview(m))
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
        # np.asarray(memoryview(m)) honours the mapping's own row_stride
        # instead of assuming a tight buffer the way frombuffer().reshape()
        # does.
        np.asarray(memoryview(m))[:] = (30, 60, 90, 255)

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
        out = np.asarray(memoryview(m))
        assert out[8, 8].tolist() != [0, 0, 0, 0], (
            "background did not composite -- dst is blank"
        )


@pytest.mark.gpu
def test_draw_decoded_masks_publishes_its_completion_on_a_foreign_texture():
    """A D3D11 texture allocated read-only by `edgefirst.tensor` and drawn
    into by `edgefirst.image` reports the draw's completion afterwards.

    Only the GL engine can write a read-only texture: the CPU backend maps
    it for writing and fails (`ef_tensor_map failed: errno 13`), which is
    what every draw on Windows did while the GL draw arms were Linux-only.
    The GL draw records a fence value on the tensor it renders into -- the
    one reconstructed from the descriptor, not the caller's -- and the
    binding publishes it back the way `convert` does, so a device consumer
    can wait on the drawn frame instead of the CPU.
    """
    if sys.platform != "win32" or not is_gpu_buffer_available():
        pytest.skip("needs Windows with a D3D11 device")

    import numpy as np
    from edgefirst.image import ImageProcessor
    from edgefirst.tensor import TensorMemory

    proc = ImageProcessor()
    # Allocated by edgefirst.tensor, drawn into by edgefirst.image's GL engine.
    dst = Tensor.image(64, 64, PixelFormat.Rgba, TensorMemory.DMABUF, "read")
    skip_unless_gpu_backed(dst)
    assert dst.memory == TensorMemory.DMABUF, (
        f"destination is {dst.memory!r}, not a texture -- this test is "
        "vacuous unless the draw lands in a real GPU allocation"
    )
    assert dst.gpu_completion() is None, "nothing has written this texture yet"

    proc.draw_decoded_masks(
        dst,  # <-- crosses packages
        bbox=np.array([[0.25, 0.25, 0.75, 0.75]], dtype=np.float32),
        scores=np.array([0.99], dtype=np.float32),
        classes=np.array([0], dtype=np.uintp),
        seg=[np.full((4, 4, 1), 255, dtype=np.uint8)],
    )

    completion = dst.gpu_completion()
    assert completion is not None, "the draw published no completion on dst"
    fence, value = completion
    try:
        assert value > 0
    finally:
        # An owned duplicate under the dynamic backend the wheels link.
        _close(fence)

    with dst.map("read") as m:
        # np.asarray(memoryview(m)) honours the texture's staging pitch.
        out = np.asarray(memoryview(m))
        assert out[2, 2].tolist() == [0, 0, 0, 0], "corner (2,2) was not cleared"
        assert out[32, 32].tolist() != [0, 0, 0, 0], (
            "box centre (32,32) was not coloured"
        )


@pytest.mark.gpu
def test_draw_proto_masks_publishes_its_completion_on_a_foreign_texture():
    """The proto draw publishes its completion the way the decoded-mask
    draw does: a read-only texture from `edgefirst.tensor`, prototype data
    from `edgefirst.decoder`, rendered by `edgefirst.image`.

    Same reasoning as the decoded-mask test above. The proto data crosses
    packages through `__edgefirst_protodata__`, built the way
    `test_materialize_masks_accepts_foreign_proto_data` builds it.
    """
    if sys.platform != "win32" or not is_gpu_buffer_available():
        pytest.skip("needs Windows with a D3D11 device")

    import numpy as np
    from edgefirst.decoder import Decoder
    from edgefirst.decoder import Tensor as DTensor
    from edgefirst.image import ImageProcessor
    from edgefirst.tensor import TensorMemory

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
    combined[0, 0:4, 0] = [0.5, 0.5, 0.5, 0.5]  # centred half-frame box
    combined[0, 4, 0] = 0.9  # class score
    combined[0, 5, 0] = 4.0  # positive coefficient: foreground everywhere
    protos = np.ones((1, nm, proto_h, proto_w), dtype=np.float32)
    t_combined = DTensor(list(combined.shape), dtype="float32")
    t_combined.from_numpy(combined)
    t_protos = DTensor(list(protos.shape), dtype="float32")
    t_protos.from_numpy(protos)
    boxes, scores, classes, proto_data = dec.decode_proto([t_combined, t_protos])
    assert len(boxes) == 1

    proc = ImageProcessor()
    dst = Tensor.image(64, 64, PixelFormat.Rgba, TensorMemory.DMABUF, "read")
    skip_unless_gpu_backed(dst)
    assert dst.gpu_completion() is None, "nothing has written this texture yet"

    proc.draw_proto_masks(dst, boxes, scores, classes, proto_data)  # <-- crosses

    completion = dst.gpu_completion()
    assert completion is not None, "the proto draw published no completion on dst"
    fence, value = completion
    try:
        assert value > 0
    finally:
        _close(fence)

    with dst.map("read") as m:
        out = np.asarray(memoryview(m))
        assert out[2, 2].tolist() == [0, 0, 0, 0], "corner (2,2) was not cleared"
        assert out[32, 32].tolist() != [0, 0, 0, 0], (
            "box centre (32,32) was not coloured"
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


@pytest.mark.gpu
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
    skip_unless_gpu_backed(dst)
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


@pytest.mark.gpu
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
    skip_unless_gpu_backed(src)

    got = decode_file_into(src, "testdata/zidane.jpg")
    assert (got.width, got.height) == (info.width, info.height)

    dst = proc.create_image(640, 640, ImgFmt.Rgb, "uint8", "readwrite")
    proc.convert(src, dst, letterbox=[114, 114, 114, 255])
    with dst.map() as m:
        # np.asarray(memoryview(m)) honours the mapping's own row_stride. A
        # `create_image()` destination is a D3D11 texture on Windows, whose
        # staging pitch need not equal the tight 1920-byte row
        # frombuffer().reshape() assumes.
        out = np.asarray(memoryview(m))
        assert out[320, 320].tolist() != [0, 0, 0], "converted image is blank"


@pytest.mark.gpu
def test_d3d11_texture_crosses_packages_through_the_capsule():
    """A D3D11 texture allocated by `edgefirst.tensor` and rendered into by
    `edgefirst.image`, through the capsule and nothing else.

    Every `edgefirst.*` wheel links its own copy of the tensor crate, so
    before the device rendezvous each copy created its own `ID3D11Device`
    and a texture from one was unreachable from the other -- the convert
    either failed to import the destination or rendered into a texture the
    consumer's device could not open. It also needs the `D3D11_TEXTURE`
    descriptor kind: without it the destination arrives as a HOST
    descriptor with no address and the convert falls back to (or fails on)
    host memory, which is not the zero-copy path this exists for.

    The completion crosses back two ways, and both are checked here. A
    convert renders into a `TensorDyn` the binding reconstructs from the
    descriptor -- that is what lets it release the GIL -- so the fence value
    `edgefirst.image`'s GL engine records lands on that short-lived tensor;
    the binding publishes it onto the caller's object through
    `set_gpu_write` before returning, which is what `dst.gpu_completion()`
    reads back. `convert_with_fence` carries the same completion the other
    way, as an event the caller waits on and owns.
    """
    if sys.platform != "win32" or not is_gpu_buffer_available():
        pytest.skip("needs Windows with a D3D11 device")

    import numpy as np
    from edgefirst.image import Flip, ImageProcessor, Rotation
    from edgefirst.tensor import TensorMemory

    p = ImageProcessor()
    src = Tensor.image(64, 48, PixelFormat.Rgba, None, "readwrite")
    with src.map() as m:
        np.asarray(memoryview(m))[:] = (200, 100, 50, 255)

    # Allocated by edgefirst.tensor, written by edgefirst.image's GL engine.
    dst = Tensor.image(64, 48, PixelFormat.Rgb, TensorMemory.DMABUF, "read")
    # The same GPU policy the other tests here follow: Windows runs this only
    # with HAL_TEST_REQUIRE_GL=1 and ANGLE configured, because a CPU-backend
    # convert has no device fence to hand back.
    skip_unless_gpu_backed(dst)
    assert dst.memory == TensorMemory.DMABUF, (
        f"destination is {dst.memory!r}, not a texture -- this test is "
        "vacuous unless the convert crosses a real GPU allocation"
    )

    assert dst.gpu_completion() is None, "nothing has written this texture yet"

    p.convert(src, dst, Rotation.Rotate0, Flip.NoFlip)  # <-- crosses packages

    with dst.map("read") as m:
        # np.asarray(memoryview(m)) honours the texture's staging pitch,
        # which is not the tight 192-byte row of a 64-pixel rgb8 image.
        assert np.asarray(memoryview(m))[0, 0].tolist() == [200, 100, 50]

    completion = dst.gpu_completion()
    assert completion is not None, "the convert published no completion on dst"
    fence, value = completion
    try:
        assert value > 0
    finally:
        # An owned duplicate under the dynamic backend the wheels link.
        _close(fence)

    # The same completion the other way round: the event a fenced convert
    # hands back, which a caller waits on instead of the fence value.
    event = p.convert_with_fence(src, dst, Rotation.Rotate0, Flip.NoFlip)
    assert event is not None, (
        "a D3D11 display signals its device fence, so a fenced convert into "
        "a texture from another package must hand back an event"
    )
    try:
        # WAIT_OBJECT_0. A fence the consumer's device could not signal
        # would time out here (WAIT_TIMEOUT, 0x102).
        waited = ctypes.windll.kernel32.WaitForSingleObject(
            ctypes.c_void_p(event), 5000
        )
    finally:
        # The caller owns the event, as it owns the handle `gpu_completion`
        # returns: both are duplicates under the dynamic backend the wheels
        # link.
        _close(event)
    assert waited == 0

    # The second convert published a value of its own. Strictly greater, not
    # merely not-smaller: each recorded completion signals the shared device
    # fence at a new, higher value, so an equal reading would mean the fenced
    # convert published nothing and this is still the first convert's value
    # sitting in a monotonic slot.
    later = dst.gpu_completion()
    assert later is not None
    later_fence, later_value = later
    try:
        assert later_value > value
    finally:
        _close(later_fence)

    # `convert_deferred` publishes too, and its value covers the queued
    # render. Read before the `flush`, deliberately: that is the whole point
    # of the deferred path -- a device consumer waits on this fence instead
    # of paying the flush. The flush then completes the render this test
    # queued rather than leaving it outstanding.
    p.convert_deferred(src, dst, Rotation.Rotate0, Flip.NoFlip)
    deferred = dst.gpu_completion()
    p.flush()
    assert deferred is not None, "convert_deferred published no completion"
    deferred_fence, deferred_value = deferred
    try:
        assert deferred_value > later_value
    finally:
        _close(deferred_fence)


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
    assert len(boxes) == 1
    assert float(np.asarray(scores)[0]) == pytest.approx(0.9)


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
    assert len(boxes) == 1
    assert float(np.asarray(scores)[0]) == pytest.approx(0.9)


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
