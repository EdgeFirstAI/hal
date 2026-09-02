# SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
# SPDX-License-Identifier: Apache-2.0

"""Skip when DMABUF allocation fails because the host has no usable dma-heap.

The C API wraps ``PermissionDenied`` as ``IoError(Custom { kind: Other })``,
so the Python message is ``image_alloc: … Permission denied`` rather than a
bare ``PermissionDenied`` kind. Matching only ``errno`` / ``e.kind()``
misses that wrap. Other I/O errors must still fail the test.

Windows has no ``TensorMemory::DmaBuf`` backing at all (the GPU path there
is PBO via ANGLE), so the allocator reports ``NotImplemented`` rather than a
permission problem; that is the same "this host cannot allocate DMA" fact
and skips too.
"""

from __future__ import annotations

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


def skip_if_dma_unavailable(exc: BaseException) -> None:
    if dma_unavailable(exc):
        pytest.skip(f"DMA-BUF unavailable on this host: {exc}")
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
