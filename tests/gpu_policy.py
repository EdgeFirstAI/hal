# SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
# SPDX-License-Identifier: Apache-2.0

"""GPU test policy for the 0.29 modular ABI.

Linux CI skips (no real GPU; software GL is opted out). Windows skips
(no GPU support yet). macOS must run and pass via ANGLE.
"""

from __future__ import annotations

import sys

import pytest


def skip_unless_gpu_backed(dst) -> None:
    """Enforce the GPU test policy on a ``create_image()`` result.

    - Windows: skip (no GPU support yet).
    - Linux: skip when the destination is host memory (GitHub-hosted
      runners have no real GPU; software GL is opted out).
    - macOS: require a GPU-backed tensor. A vacuous skip is a product bug.
    """
    mem = str(dst.memory)
    host = mem in ("TensorMemory.MEM", "TensorMemory.SHM")
    if sys.platform == "win32":
        pytest.skip("Windows has no GPU support yet")
    if sys.platform.startswith("linux"):
        if host:
            pytest.skip(
                "Linux CI has no real GPU; create_image() fell back to host memory"
            )
        return
    if sys.platform == "darwin":
        assert not host, (
            f"create_image() yielded {dst.memory!r} on macOS; GPU-backed "
            "(IOSurface/PBO) is required when ANGLE is configured"
        )
        return
    pytest.skip(f"GPU tests are not defined for platform {sys.platform!r}")
