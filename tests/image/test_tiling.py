# SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
# SPDX-License-Identifier: Apache-2.0

"""Tests for SAHI-style input tiling bindings.

The grid geometry and config validation are pure (no GPU). The
``tile_into`` round-trip is GPU-gated and skips when no backend can render.
"""

import numpy as np
import pytest

from edgefirst_hal import (
    Fit,
    ImageProcessor,
    PixelFormat,
    TilingConfig,
    tile_grid,
)


# --- grid geometry (pure, no GPU) ----------------------------------------


def test_tile_grid_axis_worked_example():
    # adis-uav-model sahi(1920, 640, 0.1) origins along one axis => 0..1280.
    # tile_grid is width-first; use a 1920x640 frame with a 640x640 tile so
    # the y axis collapses to a single row and we read column origins.
    specs = tile_grid(1920, 640, 640, 640, 0.1)
    xs = [s.source.x for s in specs]
    assert xs[0] == 0
    assert xs[-1] == 1920 - 640  # 1280
    assert xs == sorted(xs)
    # adis worked example: [0, 427, 853, 1280]
    assert xs == [0, 427, 853, 1280]


def test_tile_grid_4k_is_32_tiles():
    specs = tile_grid(3840, 2160, 640, 640, 0.2)
    assert len(specs) == 8 * 4  # 32 tiles
    for s in specs:
        assert s.source.width == 640
        assert s.source.height == 640
        assert s.source.x + 640 <= 3840
        assert s.source.y + 640 <= 2160
    # Row-major: index 0 at origin, index 8 starts row 1.
    assert (specs[0].source.x, specs[0].source.y) == (0, 0)
    assert specs[0].index == 0 and specs[0].row == 0 and specs[0].col == 0
    assert specs[8].row == 1 and specs[8].col == 0


def test_tile_grid_frame_smaller_than_tile_single():
    specs = tile_grid(500, 400, 640, 640, 0.2)
    assert len(specs) == 1
    s = specs[0].source
    assert (s.x, s.y, s.width, s.height) == (0, 0, 500, 400)


def test_tile_grid_rejects_bad_config():
    # Same validation as TilingConfig / the C API: a degenerate overlap or tile
    # size raises instead of generating an enormous grid.
    with pytest.raises(Exception):
        tile_grid(1920, 1080, 640, 640, 1.0)
    with pytest.raises(Exception):
        tile_grid(1920, 1080, 640, 640, -0.1)
    with pytest.raises(Exception):
        tile_grid(1920, 1080, 0, 640, 0.2)


# --- config ---------------------------------------------------------------


def test_tiling_config_getters_and_chaining():
    cfg = TilingConfig(640, 480, overlap=0.3)
    assert cfg.tile_w == 640
    assert cfg.tile_h == 480
    assert cfg.overlap == pytest.approx(0.3)
    assert cfg.fit == Fit.Stretch
    assert cfg.pad == (114, 114, 114, 255)

    cfg2 = cfg.with_overlap(0.1).with_fit(Fit.Letterbox)
    assert cfg2.overlap == pytest.approx(0.1)
    assert cfg2.fit == Fit.Letterbox
    # Original is unchanged (chainable returns copies).
    assert cfg.overlap == pytest.approx(0.3)
    assert cfg.fit == Fit.Stretch


def test_tiling_config_custom_pad():
    cfg = TilingConfig(640, 640, fit=Fit.Letterbox, pad=(0, 1, 2, 3))
    assert cfg.fit == Fit.Letterbox
    assert cfg.pad == (0, 1, 2, 3)


def test_plan_tiles_counts_match_grid():
    proc = ImageProcessor()
    cfg = TilingConfig(640, 640, overlap=0.2)
    placements = proc.plan_tiles(3840, 2160, cfg)
    assert len(placements) == 32
    counts = {p.count for p in placements}
    assert counts == {32}
    indices = sorted(p.index for p in placements)
    assert indices == list(range(32))
    # Every placement records the full-frame dims.
    for p in placements:
        assert p.frame_dims == (3840.0, 2160.0)
        assert p.crop_size == (640.0, 640.0)


def test_plan_tiles_rejects_bad_overlap():
    proc = ImageProcessor()
    with pytest.raises(Exception):
        proc.plan_tiles(1920, 1080, TilingConfig(640, 640, overlap=1.0))
    with pytest.raises(Exception):
        proc.plan_tiles(1920, 1080, TilingConfig(640, 640, overlap=-0.1))


def test_alloc_tile_batch_rejects_zero_tile_size():
    proc = ImageProcessor()
    with pytest.raises(Exception):
        proc.alloc_tile_batch(4, TilingConfig(0, 640))


# --- GPU round-trip (gated) ----------------------------------------------


def test_tile_into_round_trip():
    """tile_into binding smoke test: it returns one placement per grid tile and
    the packed parent is shaped to stack the tiles vertically.

    This exercises the Python wiring (config -> grid -> render -> placements),
    not pixel parity — band-vs-standalone content parity is covered by the Rust
    ``tile_into_auto_dma_parity`` / ``tile_into_cpu_distinct_content_parity``
    tests, which can assert exact/structural pixel equality directly.
    """
    proc = ImageProcessor()
    cfg = TilingConfig(64, 64, overlap=0.0)

    # 128x64 frame, 64x64 tile, no overlap => 2 tiles side by side.
    try:
        src = proc.create_image(128, 64, PixelFormat.Rgba)
        placements = proc.plan_tiles(128, 64, cfg)
        n = len(placements)
        dst = proc.alloc_tile_batch(n, cfg, PixelFormat.Rgba)
    except (RuntimeError, AttributeError) as e:
        pytest.skip(f"GPU/image backend unavailable: {e}")

    assert n == 2

    # Confirm the source is CPU-mappable and at least its logical size (the
    # buffer may be row-padded); otherwise this backend can't run the render.
    try:
        with src.map() as m:
            buf = np.frombuffer(m.numpy(), dtype=np.uint8)
            assert len(buf) >= 128 * 64 * 4
    except (RuntimeError, NotImplementedError) as e:
        pytest.skip(f"source not CPU-mappable: {e}")

    try:
        out = proc.tile_into(src, dst, cfg)
    except RuntimeError as e:
        pytest.skip(f"tile_into render unavailable: {e}")

    assert len(out) == 2
    assert out[0].index == 0
    assert out[1].index == 1
    # The packed parent stacks n tiles of tile_h vertically.
    assert dst.width == 64
    assert dst.height == n * 64
