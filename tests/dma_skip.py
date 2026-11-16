# SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
# SPDX-License-Identifier: Apache-2.0

"""Skip when DMABUF allocation fails because the host has no usable dma-heap.

A permission-denied message and, on Windows, a not-implemented message count as
unavailable; any other error fails the test.
"""

from __future__ import annotations

import os
import sys

import pytest
from edgefirst.tensor import Tensor, TensorMemory


def dma_unavailable(exc: BaseException) -> bool:
    msg = str(exc)
    if sys.platform == "win32" and (
        "NotImplemented" in msg or "only available on" in msg
    ):
        return True
    return (
        "Permission denied" in msg
        or "PermissionDenied" in msg
        or "errno 13" in msg
        or "Errno 13" in msg
    )


def skip_dma(reason: str) -> None:
    """Skip a DMA-BUF test, or fail it under ``HAL_TEST_REQUIRE_DMA=1``."""
    if os.environ.get("HAL_TEST_REQUIRE_DMA") == "1":
        pytest.fail(
            f"HAL_TEST_REQUIRE_DMA=1 but a DMA test would have skipped: {reason}"
        )
    pytest.skip(reason)


def skip_if_dma_unavailable(exc: BaseException) -> None:
    if dma_unavailable(exc):
        skip_dma(f"DMA-BUF unavailable on this host: {exc}")
    raise exc


def image_or_skip_dma(*args, **kwargs):
    """``Tensor.image(...)`` that skips only wrapped dma-heap denial."""
    try:
        return Tensor.image(*args, **kwargs)
    except Exception as exc:
        mem = kwargs.get("mem")
        if mem is TensorMemory.DMABUF or (
            len(args) >= 4 and args[3] is TensorMemory.DMABUF
        ):
            skip_if_dma_unavailable(exc)
        raise


def tensor_or_skip_dma(*args, **kwargs):
    """``Tensor(...)`` constructor that skips only wrapped dma-heap denial."""
    try:
        return Tensor(*args, **kwargs)
    except Exception as exc:
        mem = kwargs.get("mem")
        if mem is TensorMemory.DMABUF:
            skip_if_dma_unavailable(exc)
        raise
