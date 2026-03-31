# SPDX-FileCopyrightText: Copyright 2025 Au-Zone Technologies
# SPDX-License-Identifier: Apache-2.0

import numpy as np
from edgefirst_hal import Tensor, TensorMemory, PixelFormat
import os
import sys
import pytest
import gc


# fmt: off
DTYPE_PARAMS = [
    # (dtype,     size, fmt, itemsize, val1,  val2)
    ("int8",      120, "b",  1,       10,    -120),
    ("uint8",     120, "B",  1,       10,    250),
    ("int16",     240, "h",  2,       10,    -30000),
    ("uint16",    240, "H",  2,       10,    60000),
    ("int32",     480, "i",  4,       10,    -2000000000),
    ("uint32",    480, "I",  4,       10,    4000000000),
    ("int64",     960, "q",  8,       10,    -9000000000000000000),
    ("uint64",    960, "Q",  8,       10,    18000000000000000000),
    ("float32",   480, "f",  4,       10.0,  -2000000000.0),
    ("float64",   960, "d",  8,       10.0,  -9000000000000000000.0),
]
# fmt: on


@pytest.mark.parametrize("dtype,size,fmt,itemsize,val1,val2", DTYPE_PARAMS,
                         ids=[p[0] for p in DTYPE_PARAMS])
def test_dtype(dtype, size, fmt, itemsize, val1, val2):
    tensor = Tensor([1, 2, 3, 4, 5], dtype=dtype)
    assert tensor.size == size
    assert tensor.shape == [1, 2, 3, 4, 5]
    assert tensor.dtype == dtype

    with tensor.map() as m:
        m[0] = val1
        assert m[0] == val1
        m[1] = val2
        assert m[1] == val2

        if hasattr(m, "__getbuffer__"):
            with memoryview(m) as v:
                assert v.format == fmt
                assert v.ndim == 5
                assert v.shape == (1, 2, 3, 4, 5)
                assert v.itemsize == itemsize
                assert v[0, 0, 0, 0, 0] == val1
                assert v[0, 0, 0, 0, 1] == val2
            with pytest.raises(ValueError):
                _ = v[0, 0, 0, 0, 0]
        else:
            n = np.frombuffer(m.view(), dtype=tensor.dtype).reshape(tensor.shape)
            assert n[0, 0, 0, 0, 0] == val1
            assert n[0, 0, 0, 0, 1] == val2
    with pytest.raises(BufferError):
        _ = m[0]


def test_from_fd_dma():
    try:
        tensor = Tensor([100, 100, 3], dtype="uint8", mem=TensorMemory.DMA)
    except (AttributeError, RuntimeError):
        pytest.skip("DMA memory not supported on this platform")

    assert tensor.memory == TensorMemory.DMA

    tensor_fd = Tensor.from_fd(tensor.fd, tensor.shape, tensor.dtype)

    assert tensor_fd.size == tensor.size
    assert tensor_fd.shape == tensor.shape
    assert tensor_fd.dtype == tensor.dtype
    assert tensor_fd.memory == tensor.memory

    with tensor.map() as m:
        for i in range(tensor.size):
            m[i] = 233

    with tensor_fd.map() as m:
        for i in range(tensor_fd.size):
            assert m[i] == 233


def test_dma_zero_copy_perf():
    try:
        tensor = Tensor([100, 100, 3], dtype="uint8", mem=TensorMemory.DMA)
    except (AttributeError, RuntimeError):
        pytest.skip("DMA memory not supported on this platform")

    assert tensor.memory == TensorMemory.DMA

    import time

    iterations = 50
    elapsed = 0
    elapsed_copy = 0

    for _ in range(iterations):
        start = time.perf_counter()
        tensor_fd = Tensor.from_fd(tensor.fd, tensor.shape, tensor.dtype)
        elapsed += time.perf_counter() - start

        assert tensor_fd.size == tensor.size
        assert tensor_fd.shape == tensor.shape
        assert tensor_fd.dtype == tensor.dtype
        assert tensor_fd.memory == tensor.memory

        with tensor.map() as m:
            for i in range(tensor.size):
                m[i] = 233

        with tensor_fd.map() as m:
            for i in range(tensor_fd.size):
                assert m[i] == 233

        tensor_copy = Tensor(tensor.shape, dtype=tensor.dtype, mem=tensor.memory)
        start = time.perf_counter()
        with tensor.map() as src, tensor_copy.map() as dst:
            for i in range(tensor.size):
                dst[i] = src[i]
        elapsed_copy += time.perf_counter() - start

    # Skip timing assertion if both are under 1ms (measurement noise dominates)
    if elapsed > 0.001 or elapsed_copy > 0.001:
        assert elapsed < elapsed_copy * 1.5, (
            f"zero-copy ({elapsed:.4f}s) not faster than copy ({elapsed_copy:.4f}s)"
        )


def test_from_fd_shm():
    try:
        tensor = Tensor([100, 100, 3], dtype="uint8", mem=TensorMemory.SHM)
    except (AttributeError, RuntimeError):
        pytest.skip("SHM memory not supported on this platform")

    assert tensor.memory == TensorMemory.SHM

    tensor_fd = Tensor.from_fd(tensor.fd, tensor.shape, tensor.dtype)

    assert tensor_fd.size == tensor.size
    assert tensor_fd.shape == tensor.shape
    assert tensor_fd.dtype == tensor.dtype
    assert tensor_fd.memory == tensor.memory

    with tensor.map() as m:
        for i in range(tensor.size):
            m[i] = 233

    with tensor_fd.map() as m:
        for i in range(tensor_fd.size):
            assert m[i] == 233


def test_shm_zero_copy_perf():
    try:
        tensor = Tensor([100, 100, 3], dtype="uint8", mem=TensorMemory.SHM)
    except (AttributeError, RuntimeError):
        pytest.skip("SHM memory not supported on this platform")

    assert tensor.memory == TensorMemory.SHM

    import time

    iterations = 50
    elapsed = 0
    elapsed_copy = 0

    for _ in range(iterations):
        start = time.perf_counter()
        tensor_fd = Tensor.from_fd(tensor.fd, tensor.shape, tensor.dtype)
        elapsed += time.perf_counter() - start

        assert tensor_fd.size == tensor.size
        assert tensor_fd.shape == tensor.shape
        assert tensor_fd.dtype == tensor.dtype
        assert tensor_fd.memory == tensor.memory

        with tensor.map() as m:
            for i in range(tensor.size):
                m[i] = 233

        with tensor_fd.map() as m:
            for i in range(tensor_fd.size):
                assert m[i] == 233

        tensor_copy = Tensor(tensor.shape, dtype=tensor.dtype, mem=tensor.memory)
        start = time.perf_counter()
        with tensor.map() as src, tensor_copy.map() as dst:
            for i in range(tensor.size):
                dst[i] = src[i]
        elapsed_copy += time.perf_counter() - start

    # Skip timing assertion if both are under 1ms (measurement noise dominates)
    if elapsed > 0.001 or elapsed_copy > 0.001:
        assert elapsed < elapsed_copy * 1.5, (
            f"zero-copy ({elapsed:.4f}s) not faster than copy ({elapsed_copy:.4f}s)"
        )


def tensor_fd_func(mem_type: TensorMemory):
    try:
        original = Tensor([100, 100, 3], dtype="uint8", mem=mem_type)
        del original
    except (AttributeError, RuntimeError):
        pytest.skip(f"{mem_type} memory not supported on this platform")

    for _ in range(100):
        tensor = Tensor([100, 100], dtype="uint8", mem=mem_type)
        with tensor.map() as m:
            m[0] = 3
        del tensor


def test_dma_no_fd_leaks():
    """Test that DMA tensors don't leak file descriptors"""
    psutil = pytest.importorskip("psutil")
    proc = psutil.Process(os.getpid())
    gc.collect()
    try:
        start_fds = proc.num_fds()
    except AttributeError:
        pytest.skip("num_fds not supported on this platform")

    tensor_fd_func(TensorMemory.DMA)

    gc.collect()
    end_fds = proc.num_fds()
    assert start_fds == end_fds, (
        f"File descriptor leak detected: {start_fds} -> {end_fds}"
    )


def test_shm_no_fd_leaks():
    """Test that SHM tensors don't leak file descriptors"""
    psutil = pytest.importorskip("psutil")
    proc = psutil.Process(os.getpid())
    gc.collect()
    try:
        start_fds = proc.num_fds()
    except AttributeError:
        pytest.skip("num_fds not supported on this platform")

    tensor_fd_func(TensorMemory.SHM)

    gc.collect()
    end_fds = proc.num_fds()
    assert start_fds == end_fds, (
        f"File descriptor leak detected: {start_fds} -> {end_fds}"
    )


def tensor_from_fd_func(mem_type: TensorMemory):
    try:
        original = Tensor([100, 100, 3], dtype="uint8", mem=mem_type)
    except (AttributeError, RuntimeError):
        pytest.skip(f"{mem_type} memory not supported on this platform")

    for _ in range(100):
        tensor_fd = Tensor.from_fd(original.fd, original.shape, original.dtype)
        with tensor_fd.map() as m:
            m[0] = 3
        del tensor_fd
    del original


def test_dma_fd_leak_with_from_fd():
    """Test that creating tensors from FDs doesn't leak descriptors"""
    psutil = pytest.importorskip("psutil")
    proc = psutil.Process(os.getpid())
    gc.collect()
    try:
        start_fds = proc.num_fds()
    except AttributeError:
        pytest.skip("num_fds not supported on this platform")

    tensor_from_fd_func(TensorMemory.DMA)
    gc.collect()
    end_fds = proc.num_fds()
    assert start_fds == end_fds, (
        f"File descriptor leak detected: {start_fds} -> {end_fds}"
    )


def test_shm_fd_leak_with_from_fd():
    """Test that creating tensors from SHM FDs doesn't leak descriptors"""
    psutil = pytest.importorskip("psutil")
    proc = psutil.Process(os.getpid())
    gc.collect()
    try:
        start_fds = proc.num_fds()
    except AttributeError:
        pytest.skip("num_fds not supported on this platform")

    tensor_from_fd_func(TensorMemory.SHM)

    gc.collect()
    end_fds = proc.num_fds()
    assert start_fds == end_fds, (
        f"File descriptor leak detected: {start_fds} -> {end_fds}"
    )


def test_set_format_packed():
    """set_format attaches pixel format metadata to a raw tensor."""
    t = Tensor([480, 640, 3], dtype="uint8", mem=TensorMemory.MEM)
    assert t.format is None
    t.set_format(PixelFormat.Rgb)
    assert t.format == PixelFormat.Rgb
    assert t.width == 640
    assert t.height == 480


def test_set_format_rejects_wrong_shape():
    """set_format raises RuntimeError when shape doesn't match format."""
    t = Tensor([480, 640, 4], dtype="uint8", mem=TensorMemory.MEM)
    with pytest.raises(RuntimeError):
        t.set_format(PixelFormat.Rgb)  # expects 3 channels, got 4


@pytest.mark.skipif(sys.platform != "linux", reason="DMA-BUF is Linux-only")
def test_dmabuf_clone_mem_raises():
    """dmabuf_clone raises RuntimeError on non-DMA tensors."""
    t = Tensor([4, 4], dtype="float32", mem=TensorMemory.MEM)
    with pytest.raises(RuntimeError):
        t.dmabuf_clone()
