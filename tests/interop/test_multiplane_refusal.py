# SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
# SPDX-License-Identifier: Apache-2.0

"""A tensor whose chroma plane is a separate tensor cannot be passed to
another edgefirst package: its capsule export raises NotImplementedError,
and so does every call of another package that takes it."""

import sys
from pathlib import Path

import pytest

from tests.dma_skip import image_or_skip_dma

REFUSED = "cannot be passed to another edgefirst package"
TESTDATA = Path(__file__).resolve().parents[2] / "testdata"

W, H = 64, 48

linux_only = pytest.mark.skipif(
    sys.platform != "linux",
    reason="two-plane tensors come from DMA-BUF imports and V4L2 frames",
)


def two_plane_images():
    """An NV12M image and one with its chroma at an offset, both from
    ``edgefirst.image``, with the buffers they import."""
    from edgefirst.image import ImageProcessor, PixelFormat, TensorMemory

    p = ImageProcessor()
    luma = image_or_skip_dma(W, H, PixelFormat.Grey, mem=TensorMemory.DMABUF)
    chroma = image_or_skip_dma(W, H // 2, PixelFormat.Grey, mem=TensorMemory.DMABUF)
    one = image_or_skip_dma(W, H * 2, PixelFormat.Grey, mem=TensorMemory.DMABUF)
    two_buffers = p.import_image(luma.fd, W, H, PixelFormat.Nv12, chroma_fd=chroma.fd)
    offset = p.import_image(
        one.fd, W, H, PixelFormat.Nv12, chroma_fd=one.fd, chroma_offset=W * (H + 8)
    )
    return {"two_buffers": two_buffers, "chroma_offset": offset}, (luma, chroma, one)


@linux_only
@pytest.mark.parametrize("layout", ["two_buffers", "chroma_offset"])
@pytest.mark.parametrize("access", [None, "read"])
def test_a_two_plane_tensor_refuses_its_capsule(layout, access):
    images, _keep = two_plane_images()
    with pytest.raises(NotImplementedError, match=REFUSED):
        images[layout].__edgefirst_tensor__(access=access)


@linux_only
@pytest.mark.parametrize("layout", ["two_buffers", "chroma_offset"])
def test_another_package_refuses_a_two_plane_tensor(layout):
    from edgefirst import codec

    images, _keep = two_plane_images()
    src = images[layout]
    with pytest.raises(NotImplementedError, match=REFUSED):
        codec.ImageEncoder(codec.Codec.Jpeg).encode(src)
    jpeg = (TESTDATA / "zidane.jpg").read_bytes()
    with pytest.raises(NotImplementedError, match=REFUSED):
        codec.decode_into(src, jpeg)


def test_a_one_plane_tensor_still_crosses():
    from edgefirst import codec
    from edgefirst.image import PixelFormat, Tensor

    src = Tensor.image(W, H, PixelFormat.Nv12, None, "readwrite")
    assert src.__edgefirst_tensor__() is not None
    assert len(codec.ImageEncoder(codec.Codec.Jpeg).encode(src)) > 0


@linux_only
@pytest.mark.codec_sessions(1)
def test_a_decoded_video_frame_is_refused_by_the_image_package():
    from edgefirst import codec
    from edgefirst.image import ImageProcessor
    from edgefirst.image import PixelFormat as ImagePixelFormat

    try:
        decoder = codec.VideoDecoder(codec.Codec.H264)
    except (codec.NoDeviceError, codec.UnsupportedError) as e:
        pytest.skip(f"no hardware H.264 decoder here: {e}")
    stream = (TESTDATA / "h264_resolution_change.h264").read_bytes()
    frame = None
    for pts, unit in enumerate(codec.access_units(stream)):
        frames = [e for e in decoder.decode(unit, pts) if isinstance(e, codec.Frame)]
        if frames:
            frame = frames[0]
            break
    if frame is None:
        frame = next(e for e in decoder.flush() if isinstance(e, codec.Frame))
    p = ImageProcessor()
    dst = p.create_image(frame.width, frame.height, ImagePixelFormat.Rgba)
    with pytest.raises(NotImplementedError, match=REFUSED):
        p.convert(frame, dst)
