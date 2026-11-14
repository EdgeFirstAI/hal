# SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
# SPDX-License-Identifier: Apache-2.0

"""edgefirst.codec: engines, capabilities, JPEG encoding and video.

Video tests run where this machine has a hardware codec and skip with the
reason elsewhere; there is no software video codec to fall back to.
"""

import gc
from pathlib import Path

import numpy as np
import pytest
from edgefirst import codec
from edgefirst.codec import (
    Codec,
    CodecError,
    Engine,
    Frame,
    ImageEncoder,
    PixelFormat,
    StreamInfo,
    Tensor,
    VideoDecoder,
    VideoEncoder,
)

TESTDATA = Path(__file__).resolve().parents[2] / "testdata"


def grey(width, height, access="readwrite", value=None):
    t = Tensor.image(
        width, height, PixelFormat.Grey, mem=codec.TensorMemory.MEM, access=access
    )
    if value is not None:
        with t.map("write") as m:
            np.asarray(m)[...] = value
    return t


def test_every_error_is_a_codec_error_and_a_runtime_error():
    kinds = [
        codec.NoDeviceError,
        codec.UnsupportedError,
        codec.PoolExhaustedError,
        codec.CodecTimeoutError,
        codec.InvalidConfigError,
        codec.InvalidInputError,
        codec.InvalidDataError,
    ]
    for kind in kinds:
        assert issubclass(kind, CodecError)
        assert kind.__module__ == "edgefirst.codec"
    assert issubclass(CodecError, RuntimeError)


def test_codec_and_engine_names():
    assert str(Codec.H264) == "H.264"
    assert str(Engine.MediaFoundation) == "media-foundation"
    assert Engine.from_name("cpu") == Engine.Cpu
    assert Engine.from_name(str(Engine.V4l2)) == Engine.V4l2
    with pytest.raises(ValueError):
        Engine.from_name("no-such-engine")


def test_capabilities_list_the_cpu_jpeg_encoder():
    caps = codec.capabilities()
    cpu = [
        c
        for c in caps
        if c.engine == Engine.Cpu
        and c.codec == Codec.Jpeg
        and c.direction == codec.Direction.Encode
    ]
    assert len(cpu) == 1, caps
    assert not cpu[0].hardware
    assert PixelFormat.Nv12 in cpu[0].formats
    assert isinstance(cpu[0].transfer, codec.Transfer)


def test_a_grey_image_encodes_to_a_jpeg_and_decodes_back():
    enc = ImageEncoder(quality=90, engine=Engine.Cpu)
    assert enc.quality == 90
    assert enc.engine == Engine.Cpu
    jpeg = enc.encode(grey(64, 48, value=200))
    assert jpeg[:2] == b"\xff\xd8"
    info = Tensor.peek_image_info(jpeg)
    assert (info.width, info.height) == (64, 48)
    dst = grey(64, 48)
    codec.decode_into(dst, jpeg)
    with dst.map("read") as m:
        pixels = np.asarray(m)
    assert abs(int(pixels.mean()) - 200) <= 2


def test_image_encoder_refusals():
    with pytest.raises(codec.UnsupportedError):
        ImageEncoder(Codec.Png)
    for bad in (0, 101, 300, -1):
        with pytest.raises(codec.InvalidConfigError):
            ImageEncoder(quality=bad)
    enc = ImageEncoder(engine=Engine.Cpu)
    with pytest.raises(codec.InvalidConfigError):
        enc.quality = 0
    with pytest.raises(PermissionError):
        enc.encode(grey(16, 16, access="write"))


def test_undecodable_data_is_invalid_data():
    with pytest.raises(codec.InvalidDataError):
        codec.decode_into(grey(16, 16), b"\xff\xd8\xff\x00 not a jpeg")


def test_access_units_split_an_annex_b_stream():
    stream = (TESTDATA / "h264_1080p_main_bframes.h264").read_bytes()
    units = codec.access_units(stream)
    assert len(units) > 1
    assert all(u[:3] == b"\x00\x00\x01" or u[:4] == b"\x00\x00\x00\x01" for u in units)
    assert sum(len(u) for u in units) <= len(stream)


def open_or_skip(factory):
    try:
        return factory()
    except (codec.NoDeviceError, codec.UnsupportedError) as e:
        pytest.skip(f"no hardware codec here: {e}")


def decode_all(decoder, units):
    events = []
    for pts, unit in enumerate(units):
        events += decoder.decode(unit, pts)
    events += decoder.flush()
    return events


def test_a_stream_decodes_into_tensor_frames():
    units = codec.access_units((TESTDATA / "h264_1080p_main_bframes.h264").read_bytes())
    decoder = open_or_skip(lambda: VideoDecoder(Codec.H264, pool=len(units) + 2))
    events = decode_all(decoder, units)
    assert isinstance(events[0], StreamInfo), events[:2]
    info = events[0]
    assert (info.width, info.height) == (1920, 1080)
    assert info.format == PixelFormat.Nv12
    assert decoder.stream_info.width == 1920
    frames = [e for e in events if isinstance(e, Frame)]
    assert sorted(f.pts for f in frames) == list(range(len(units)))
    f = frames[0]
    assert isinstance(f, Tensor)
    assert (f.width, f.height, f.format) == (1920, 1080, PixelFormat.Nv12)
    with f.map("read") as m:
        assert np.asarray(m).size > 0


def test_frames_hold_the_pool_until_released():
    units = codec.access_units((TESTDATA / "h264_1080p_main_bframes.h264").read_bytes())
    decoder = open_or_skip(lambda: VideoDecoder(Codec.H264, pool=1))
    held = []
    with pytest.raises(codec.PoolExhaustedError):
        for pts, unit in enumerate(units):
            held += [e for e in decoder.decode(unit, pts) if isinstance(e, Frame)]
    stuck = pts
    held.clear()
    gc.collect()
    decoder.decode(units[stuck], stuck)


def test_a_pool_holding_one_buffer_twice_is_refused():
    t = Tensor.image(64, 64, PixelFormat.Grey, mem=codec.TensorMemory.MEM)
    with pytest.raises(codec.InvalidConfigError, match="same buffer"):
        VideoDecoder(Codec.H264, pool=[t, t])


def nv12(width, height, luma):
    t = Tensor.image(width, height, PixelFormat.Nv12)
    with t.map("write") as m:
        a = np.asarray(m)
        a[:height] = luma
        a[height:] = 128
    return t


def test_an_h264_stream_round_trips_through_the_hardware():
    w, h, count = 320, 240, 8
    encoder = open_or_skip(
        lambda: VideoEncoder(Codec.H264, w, h, PixelFormat.Nv12, 30.0, gop=count)
    )
    assert encoder.info.hardware
    packets = []
    for pts in range(count):
        src = nv12(w, h, 16 + pts * 24)
        crop = codec.Region(0, 0, w, h) if pts == 1 else None
        p = encoder.encode(src, pts, keyframe=pts == 0, crop=crop)
        if p is not None:
            packets.append(p)
    packets += encoder.flush()
    assert [p.pts for p in packets] == list(range(count))
    assert packets[0].keyframe
    assert all(len(p) == len(p.data) > 0 for p in packets)

    decoder = open_or_skip(lambda: VideoDecoder(Codec.H264, pool=count + 2))
    events = []
    for p in packets:
        events += decoder.decode(p.data, p.pts)
    events += decoder.flush()
    frames = [e for e in events if isinstance(e, Frame)]
    assert [f.pts for f in frames] == list(range(count))
    assert (frames[0].width, frames[0].height) == (w, h)
    assert not any(f.corrupted for f in frames)
    for f in frames:
        with f.map("read") as m:
            luma = np.asarray(m)[:h, :w]
        assert abs(float(luma.mean()) - (16 + f.pts * 24)) < 4, f.pts
    with pytest.raises(RuntimeError):
        frames[0].map("write")
