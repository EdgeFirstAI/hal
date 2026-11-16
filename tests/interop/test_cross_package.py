# SPDX-FileCopyrightText: Copyright 2025-2026 Au-Zone Technologies
# SPDX-License-Identifier: Apache-2.0

"""Tests of tensors, decoders and proto data crossing between EdgeFirst packages."""

import ctypes
import sys

import pytest
from edgefirst.tensor import PixelFormat, Tensor, is_gpu_buffer_available

from tests.dma_skip import skip_dma
from tests.gpu_policy import skip_unless_gpu_backed


def _close(handle: int) -> None:
    """Close a Windows handle a completion or fence call handed back."""
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


# --- TensorArg, the cross-package extractor, wired into ImageProcessor ------


def test_codec_tensor_converts_via_image(tmp_path):
    """Decode in one package, preprocess in another."""
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
        out = np.asarray(memoryview(m))
        assert out[32, 32].tolist() != [0, 0, 0], "converted image is blank"


def test_object_without_the_protocol_is_rejected_with_a_clear_message(tmp_path):
    """An object without the protocol is rejected with a message naming the protocol."""
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
    """A background image produced by a different package composites."""
    import numpy as np
    from edgefirst.image import ImageProcessor
    from edgefirst.image import PixelFormat as ImgFmt
    from edgefirst.tensor import PixelFormat as TFmt
    from edgefirst.tensor import Tensor as TTensor

    w, h = 16, 16
    background = TTensor.image(w, h, TFmt.Rgba, None, "readwrite")  # different package
    with background.map() as m:
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
    """A read-only texture from another package reports the draw's completion."""
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
        _close(fence)

    with dst.map("read") as m:
        out = np.asarray(memoryview(m))
        assert out[2, 2].tolist() == [0, 0, 0, 0], "corner (2,2) was not cleared"
        assert out[32, 32].tolist() != [0, 0, 0, 0], (
            "box centre (32,32) was not coloured"
        )


@pytest.mark.gpu
def test_draw_proto_masks_publishes_its_completion_on_a_foreign_texture():
    """The proto draw reports its completion on a foreign read-only texture."""
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
    """A Mem-backed destination from a different package is a valid convert() target."""
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

    proc.convert(src, dst)

    with dst.map() as m:
        out = np.frombuffer(m, dtype=np.uint8).reshape(8, 8, 3)
        assert out[4, 4].tolist() != [0, 0, 0], "converted image is blank"


@pytest.mark.gpu
def test_gpu_backed_destination_skips_the_retry():
    """A `create_image()` destination on a GPU-backed host is not Mem- or Shm-backed."""
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
    """Skip, reporting on stderr, when a MEM request is not Mem-backed."""
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
    """`decode_file_into` decodes into a GPU-backed tensor from another package."""
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
        out = np.asarray(memoryview(m))
        assert out[320, 320].tolist() != [0, 0, 0], "converted image is blank"


@pytest.mark.gpu
def test_d3d11_texture_crosses_packages_through_the_capsule():
    """A D3D11 texture from `edgefirst.tensor` is rendered into by `edgefirst.image`."""
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
    skip_unless_gpu_backed(dst)
    assert dst.memory == TensorMemory.DMABUF, (
        f"destination is {dst.memory!r}, not a texture -- this test is "
        "vacuous unless the convert crosses a real GPU allocation"
    )

    assert dst.gpu_completion() is None, "nothing has written this texture yet"

    p.convert(src, dst, Rotation.Rotate0, Flip.NoFlip)  # <-- crosses packages

    with dst.map("read") as m:
        assert np.asarray(memoryview(m))[0, 0].tolist() == [200, 100, 50]

    completion = dst.gpu_completion()
    assert completion is not None, "the convert published no completion on dst"
    fence, value = completion
    try:
        assert value > 0
    finally:
        _close(fence)

    # A fenced convert hands back an event to wait on.
    event = p.convert_with_fence(src, dst, Rotation.Rotate0, Flip.NoFlip)
    assert event is not None, (
        "a D3D11 display signals its device fence, so a fenced convert into "
        "a texture from another package must hand back an event"
    )
    try:
        waited = ctypes.windll.kernel32.WaitForSingleObject(
            ctypes.c_void_p(event), 5000
        )
    finally:
        _close(event)
    assert waited == 0

    # The second convert published a greater value.
    later = dst.gpu_completion()
    assert later is not None
    later_fence, later_value = later
    try:
        assert later_value > value
    finally:
        _close(later_fence)

    # `convert_deferred` publishes a value before the flush.
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
    """A Mem-backed destination from another package is a valid decode target."""
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
    """`Decoder.decode` accepts a model output tensor allocated by another package."""
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

    t = CoreTensor(raw.shape, "float32")  # from edgefirst.tensor
    t.from_numpy(raw)

    boxes, scores, _classes, _masks = dec.decode([t])
    assert len(boxes) == 1
    assert float(np.asarray(scores)[0]) == pytest.approx(0.9)


def test_decoder_decode_proto_accepts_foreign_tensor():
    """`Decoder.decode_proto` accepts a model output tensor from another package."""
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


# --- Decoder and ProtoData across packages ----------------------------------


def test_draw_onto_accepts_image_processor():
    """Decoder.draw_onto draws its scores and classes through an ImageProcessor."""
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
    """`materialize_masks` accepts a `ProtoData` produced by `edgefirst.decoder`."""
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
    # coef 4.0 dotted with protos 1.0 -> logit +4.0 everywhere, foreground.
    assert np.array_equal(m, np.full_like(m, 255)), (
        "expected the whole proto-resolution tile to be foreground per the "
        "documented positive-dot-product convention"
    )


def test_materialize_masks_carries_quantization_across_packages():
    """An int8 `ProtoData` keeps its quantization crossing into `edgefirst.image`."""
    import numpy as np
    from edgefirst.decoder import Decoder
    from edgefirst.decoder import Tensor as DTensor
    from edgefirst.image import ImageProcessor

    nc, nm, n_anchors, proto_h, proto_w = 1, 1, 1, 4, 4
    scale, zero_point = 0.5, 0
    metadata = {
        "decoder_version": "yolov8",
        "nms": "class_agnostic",
        "outputs": [
            {
                "type": "detection",
                "decoder": "ultralytics",
                "shape": [1, 4 + nc + nm, n_anchors],
                "score_format": "per_class",
                "quantization": [scale, zero_point],
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
                "quantization": [scale, zero_point],
            },
        ],
    }
    dec = Decoder(metadata, score_threshold=0.25, iou_threshold=0.45)

    # Quantized ints: score 1.0, coefficient 2.0 and protos 1.0 once dequantized.
    combined = np.zeros((1, 4 + nc + nm, n_anchors), dtype=np.int8)
    combined[0, 0:4, 0] = [1, 1, 2, 2]  # xc, yc, w, h -> 0.5, 0.5, 1.0, 1.0
    combined[0, 4, 0] = 2  # class score -> 1.0
    combined[0, 5, 0] = 4  # mask coefficient -> 2.0
    protos = np.full((1, nm, proto_h, proto_w), 2, dtype=np.int8)  # -> 1.0

    t_combined = DTensor(list(combined.shape), dtype="int8")
    t_combined.from_numpy(combined)
    t_combined.set_quantization_per_tensor(scale, zero_point)
    t_protos = DTensor(list(protos.shape), dtype="int8")
    t_protos.from_numpy(protos)
    t_protos.set_quantization_per_tensor(scale, zero_point)

    boxes, scores, classes, proto_data = dec.decode_proto([t_combined, t_protos])
    assert proto_data is not None
    assert len(boxes) == 1

    proc = ImageProcessor()
    masks = proc.materialize_masks(
        boxes, scores, classes, proto_data
    )  # <-- crosses decoder -> image, carrying quantization

    assert len(masks) == 1
    m = masks[0]
    assert m.dtype == np.uint8
    assert m.shape == (proto_h, proto_w, 1)
    # coef 2.0 dotted with protos 1.0 -> logit +2.0 everywhere, foreground.
    assert np.array_equal(m, np.full_like(m, 255)), (
        "expected the whole proto-resolution tile to be foreground; a mask "
        "of zeros would mean the carried scales were not applied"
    )


def test_tensor_capsule_is_v2_for_a_per_channel_producer():
    """A per-channel-quantized producer's capsule is named `edgefirst_tensor_v2`."""
    import numpy as np
    from edgefirst.decoder import Decoder  # noqa: F401
    from edgefirst.image import ImageProcessor  # noqa: F401
    from edgefirst.tensor import Tensor as CoreTensor

    src = CoreTensor([3, 2], dtype="int8")
    src.from_numpy(np.zeros((3, 2), dtype=np.int8))
    src.set_quantization_per_channel([0.5, 0.25, 0.125], [1, 2, 3], 0)

    cap = src.__edgefirst_tensor__()
    assert "edgefirst_tensor_v2" in repr(cap), (
        f"capsule name must move with any payload layout change, got {cap!r}"
    )

    q = src.quantization
    assert q is not None
    assert q.scale == pytest.approx([0.5, 0.25, 0.125])
    assert q.zero_point == [1, 2, 3]
    assert q.axis == 0
    assert q.is_per_channel


def _view_source_or_skip(cls, mem, shape, fmt):
    """Allocate a whole image in `mem`, skipping only if DMA-BUF is absent."""
    from edgefirst.tensor import TensorMemory

    if mem == TensorMemory.DMABUF:
        try:
            t = cls(shape, "uint8", mem)
        except (
            RuntimeError,
            OSError,
        ) as e:  # pragma: no cover
            skip_dma(f"DMA-BUF allocation unavailable here: {e}")
        if t.memory != TensorMemory.DMABUF:
            skip_dma(f"DMA-BUF request fell back to {t.memory!r}")
    else:
        t = cls(shape, "uint8", mem)
    t.set_format(fmt)
    return t


@pytest.mark.parametrize("mem_name", ["DMABUF", "MEM"])
@pytest.mark.parametrize("source", ["foreign", "native"])
def test_view_converts_its_own_sub_region_not_the_parents_origin(source, mem_name):
    """A ``view()`` handed to ``convert()`` reads its own sub-region."""
    import numpy as np
    from edgefirst.image import ImageProcessor
    from edgefirst.image import Tensor as ImageTensor
    from edgefirst.tensor import PixelFormat, Region, TensorMemory
    from edgefirst.tensor import Tensor as CoreTensor

    # Distinct per-row and per-column values.
    w = h = 64
    x0 = y0 = 8
    side = 16
    img = np.zeros((h, w, 3), np.uint8)
    img[..., 0] = (np.arange(h)[:, None] * 4) % 256
    img[..., 1] = (np.arange(w)[None, :] * 4) % 256
    img[..., 2] = 255

    cls = CoreTensor if source == "foreign" else ImageTensor
    mem = getattr(TensorMemory, mem_name)
    src = _view_source_or_skip(cls, mem, [h, w, 3], PixelFormat.Rgb)
    src.from_numpy(img)
    view = src.view(Region(x0, y0, side, side))

    dst = cls([side, side, 3], "uint8", TensorMemory.MEM)
    dst.set_format(PixelFormat.Rgb)
    ImageProcessor().convert(view, dst)
    with dst.map() as m:
        out = np.frombuffer(m, np.uint8).reshape(side, side, 3).copy()

    expected = img[y0 : y0 + side, x0 : x0 + side]
    origin = img[0:side, 0:side]
    assert np.array_equal(out, expected), (
        "convert() read the wrong region of the parent buffer; "
        + (
            "it read the parent's origin, i.e. the plane offset was lost"
            if np.array_equal(out, origin)
            else f"out[0,0]={out[0, 0]} expected={expected[0, 0]}"
        )
    )


def _pbo_processor_and_source(monkeypatch, make_proc, w, h, fmt):
    """A PBO-backed image and the processor that owns it, or a skip.

    Sets ``EDGEFIRST_FORCE_TRANSFER=pbo`` before building the processor.
    """
    from edgefirst.tensor import TensorMemory

    monkeypatch.setenv("EDGEFIRST_FORCE_TRANSFER", "pbo")
    try:
        proc = make_proc()
        src = proc.create_image(w, h, fmt)
    except (RuntimeError, OSError) as e:  # pragma: no cover
        pytest.skip(f"no GL image allocator here: {e}")
    if src.memory != TensorMemory.PBO:  # pragma: no cover
        pytest.skip(
            f"create_image returned {src.memory!r} even under "
            "EDGEFIRST_FORCE_TRANSFER=pbo"
        )
    return proc, src


def test_pbo_view_converts_its_own_sub_region_not_the_parents_origin(monkeypatch):
    """A ``view()`` of a PBO-backed image stays PBO-backed and reads its own region."""
    import numpy as np
    from edgefirst.image import ImageProcessor
    from edgefirst.tensor import PixelFormat, Region, TensorMemory
    from edgefirst.tensor import Tensor as CoreTensor

    w, h = 320, 240
    x0, y0 = 40, 32
    side = 96
    # Each row and each column carries its own index value.
    img = np.zeros((h, w, 3), np.uint8)
    img[..., 0] = np.arange(h)[:, None] % 256
    img[..., 1] = np.arange(w)[None, :] % 256
    img[..., 2] = 255

    proc, src = _pbo_processor_and_source(
        monkeypatch, ImageProcessor, w, h, PixelFormat.Rgb
    )

    src.from_numpy(img)
    view = src.view(Region(x0, y0, side, side))
    assert view.memory == TensorMemory.PBO, (
        f"a view of a PBO must stay PBO-backed, got {view.memory!r}; "
        "it demoted to a host placeholder"
    )

    dst = CoreTensor([side, side, 3], "uint8", TensorMemory.MEM)
    dst.set_format(PixelFormat.Rgb)
    proc.convert(view, dst)
    with dst.map() as m:
        out = np.frombuffer(m, np.uint8).reshape(side, side, 3).copy()

    if np.array_equal(out, np.zeros_like(out)):
        pytest.fail(
            "convert() read all zeros: the view was a window onto the host "
            "placeholder, not the GL buffer"
        )
    expected = img[y0 : y0 + side, x0 : x0 + side]
    origin = img[0:side, 0:side]
    assert np.array_equal(out, expected), (
        "convert() read the wrong region of the parent buffer; "
        + (
            "it read the parent's origin, i.e. the plane offset was lost"
            if np.array_equal(out, origin)
            else f"out[0,0]={out[0, 0]} expected={expected[0, 0]}"
        )
    )


def test_pbo_capsule_refuses_an_explicit_access_and_allows_the_unpinned_form(
    monkeypatch,
):
    """A PBO-backed tensor's capsule succeeds unpinned and refuses explicit access."""
    from edgefirst.image import ImageProcessor
    from edgefirst.tensor import PixelFormat

    _proc, src = _pbo_processor_and_source(
        monkeypatch, ImageProcessor, 320, 240, PixelFormat.Rgb
    )

    capsule = src.__edgefirst_tensor__()
    assert capsule is not None, (
        "the unpinned capsule is what convert() takes; refusing it would "
        "disable the zero-copy PBO path entirely"
    )

    for access in ("read", "write", "readwrite"):
        with pytest.raises(RuntimeError, match="PBO"):
            src.__edgefirst_tensor__(access=access)


def test_materialize_masks_rejects_an_object_without_the_protocol():
    """An object without the `ProtoData` protocol is rejected, naming the protocol."""
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
