# SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
# SPDX-License-Identifier: Apache-2.0

"""NV12 sources whose chroma plane is a tensor of its own.

A two-buffer source (NV12M) and a one-buffer source with its chroma at an
offset must convert like the same picture in one tight NV12 buffer, through
every entry point. Linux only: the sources are imported DMA-BUFs.
"""

import os
import select
import sys

import numpy as np
import pytest

if sys.platform != "linux":
    pytest.skip("imported DMA-BUF sources are Linux only", allow_module_level=True)

from edgefirst.image import (
    ImageProcessor,
    PixelFormat,
    TensorMemory,
    TilingConfig,
)

from tests.dma_skip import image_or_skip_dma

W, H = 128, 64
GAP = 8


def dma(width, height):
    return image_or_skip_dma(width, height, PixelFormat.Grey, mem=TensorMemory.DMABUF)


def paint(luma, chroma):
    """A gradient within the limited range, which every convert path keeps."""
    yy, xx = np.mgrid[0:H, 0:W]
    luma[:H, :W] = 16 + (xx * 2 + yy * 3) % 220
    chroma[: H // 2, 0:W:2] = 64 + (np.arange(W // 2)[None, :] % 128)
    chroma[: H // 2, 1:W:2] = 200 - (np.arange(H // 2)[:, None] % 64)


def rows(m, height):
    return np.asarray(m).reshape(height, -1)


def make_sources():
    """A tight NV12 reference, an NV12M source and a chroma-offset source of
    the same picture, with the buffers they import."""
    p = ImageProcessor()
    ref = dma(W, H * 3 // 2)
    with ref.map("write") as m:
        a = rows(m, H * 3 // 2)
        paint(a[:H], a[H:])
    tight = p.import_image(ref.fd, W, H, PixelFormat.Nv12)

    luma, chroma = dma(W, H), dma(W, H // 2)
    with luma.map("write") as m, chroma.map("write") as n:
        paint(rows(m, H), rows(n, H // 2))
    two_buffers = p.import_image(luma.fd, W, H, PixelFormat.Nv12, chroma_fd=chroma.fd)

    one = dma(W, H + GAP + H // 2)
    with one.map("write") as m:
        a = rows(m, H + GAP + H // 2)
        paint(a[:H], a[H + GAP :])
    offset = p.import_image(
        one.fd,
        W,
        H,
        PixelFormat.Nv12,
        chroma_fd=one.fd,
        chroma_offset=W * (H + GAP),
    )
    keep = (ref, luma, chroma, one)
    return tight, {"two_buffers": two_buffers, "chroma_offset": offset}, keep


@pytest.fixture(scope="module")
def sources():
    return make_sources()


def wait(fence):
    if fence is None:
        return
    try:
        ready, _, _ = select.select([fence], [], [], 5.0)
    finally:
        os.close(fence)
    assert ready, "the fence did not signal within 5 seconds"


def read(t):
    with t.map("read") as m:
        return np.asarray(m).astype(int)


def convert(p, src, method):
    dst = p.create_image(W, H, PixelFormat.Rgba)
    if method == "convert":
        p.convert(src, dst)
    elif method == "convert_with_fence":
        wait(p.convert_with_fence(src, dst))
    else:
        p.convert_deferred(src, dst)
        p.flush()
    return read(dst)


def tile(p, src, method):
    cfg = TilingConfig(64, 64, overlap=0.0)
    n = len(p.plan_tiles(W, H, cfg))
    dst = p.alloc_tile_batch(n, cfg, PixelFormat.Rgba, access="readwrite")
    if method == "tile_into":
        p.tile_into(src, dst, cfg)
    else:
        _, fence = p.tile_into_with_fence(src, dst, cfg)
        wait(fence)
    return read(dst)


@pytest.mark.parametrize("layout", ["two_buffers", "chroma_offset"])
@pytest.mark.parametrize(
    "method", ["convert", "convert_with_fence", "convert_deferred"]
)
def test_a_multiplane_source_converts_like_a_tight_one(sources, layout, method):
    tight, multiplane, _ = sources
    src = multiplane[layout]
    assert (src.width, src.height, src.format) == (W, H, PixelFormat.Nv12)
    p = ImageProcessor()
    want = convert(p, tight, method)
    got = convert(p, src, method)
    assert np.abs(got - want).max() <= 2


@pytest.mark.parametrize("layout", ["two_buffers", "chroma_offset"])
@pytest.mark.parametrize("method", ["tile_into", "tile_into_with_fence"])
def test_a_multiplane_source_tiles_like_a_tight_one(sources, layout, method):
    tight, multiplane, _ = sources
    p = ImageProcessor()
    want = tile(p, tight, method)
    got = tile(p, multiplane[layout], method)
    assert np.abs(got - want).max() <= 2
