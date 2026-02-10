# SPDX-FileCopyrightText: Copyright 2025 Au-Zone Technologies
# SPDX-License-Identifier: Apache-2.0

import numpy as np
from edgefirst_hal import Tensor, TensorMemory
import os
import pytest
import gc


def test_int8():
    tensor = Tensor([1, 2, 3, 4, 5], dtype="int8")
    assert tensor.size == 120
    assert tensor.shape == [1, 2, 3, 4, 5]
    assert tensor.dtype == "int8"

    with tensor.map() as m:
        m[0] = 10
        assert m[0] == 10
        m[1] = -120
        assert m[1] == -120

        if hasattr(m, "__getbuffer__"):
            with memoryview(m) as v:
                assert v.format == "b"
                assert v.ndim == 5
                assert v.shape == (1, 2, 3, 4, 5)
                assert v.itemsize == 1
                assert v[0, 0, 0, 0, 0] == 10
                assert v[0, 0, 0, 0, 1] == -120
            with pytest.raises(ValueError):
                _ = v[0, 0, 0, 0, 0]
        else:
            n = np.frombuffer(m.view(), dtype=tensor.dtype).reshape(tensor.shape)
            assert n[0, 0, 0, 0, 0] == 10
            assert n[0, 0, 0, 0, 1] == -120
    with pytest.raises(BufferError):
        _ = m[0]


def test_u8():
    tensor = Tensor([1, 2, 3, 4, 5], dtype="uint8")
    assert tensor.size == 120
    assert tensor.shape == [1, 2, 3, 4, 5]
    assert tensor.dtype == "uint8"

    with tensor.map() as m:
        m[0] = 10
        assert m[0] == 10
        m[1] = 250
        assert m[1] == 250

        if hasattr(m, "__getbuffer__"):
            with memoryview(m) as v:
                assert v.format == "B"
                assert v.ndim == 5
                assert v.shape == (1, 2, 3, 4, 5)
                assert v.itemsize == 1
                assert v[0, 0, 0, 0, 0] == 10
                assert v[0, 0, 0, 0, 1] == 250
            with pytest.raises(ValueError):
                _ = v[0, 0, 0, 0, 0]
        else:
            n = np.frombuffer(m.view(), dtype=tensor.dtype).reshape(tensor.shape)
            assert n[0, 0, 0, 0, 0] == 10
            assert n[0, 0, 0, 0, 1] == 250
    with pytest.raises(BufferError):
        _ = m[0]


def test_i16():
    tensor = Tensor([1, 2, 3, 4, 5], dtype="int16")
    assert tensor.size == 240
    assert tensor.shape == [1, 2, 3, 4, 5]
    assert tensor.dtype == "int16"

    with tensor.map() as m:
        m[0] = 10
        assert m[0] == 10
        m[1] = -30000
        assert m[1] == -30000

        if hasattr(m, "__getbuffer__"):
            with memoryview(m) as v:
                assert v.format == "h"
                assert v.ndim == 5
                assert v.shape == (1, 2, 3, 4, 5)
                assert v.itemsize == 2
                assert v[0, 0, 0, 0, 0] == 10
                assert v[0, 0, 0, 0, 1] == -30000
            with pytest.raises(ValueError):
                _ = v[0, 0, 0, 0, 0]
        else:
            n = np.frombuffer(m.view(), dtype=tensor.dtype).reshape(tensor.shape)
            assert n[0, 0, 0, 0, 0] == 10
            assert n[0, 0, 0, 0, 1] == -30000
    with pytest.raises(BufferError):
        _ = m[0]


def test_u16():
    tensor = Tensor([1, 2, 3, 4, 5], dtype="uint16")
    assert tensor.size == 240
    assert tensor.shape == [1, 2, 3, 4, 5]
    assert tensor.dtype == "uint16"

    with tensor.map() as m:
        m[0] = 10
        assert m[0] == 10
        m[1] = 60000
        assert m[1] == 60000

        if hasattr(m, "__getbuffer__"):
            with memoryview(m) as v:
                assert v.format == "H"
                assert v.ndim == 5
                assert v.shape == (1, 2, 3, 4, 5)
                assert v.itemsize == 2
                assert v[0, 0, 0, 0, 0] == 10
                assert v[0, 0, 0, 0, 1] == 60000
            with pytest.raises(ValueError):
                _ = v[0, 0, 0, 0, 0]
        else:
            n = np.frombuffer(m.view(), dtype=tensor.dtype).reshape(tensor.shape)
            assert n[0, 0, 0, 0, 0] == 10
            assert n[0, 0, 0, 0, 1] == 60000
    with pytest.raises(BufferError):
        _ = m[0]


def test_i32():
    tensor = Tensor([1, 2, 3, 4, 5], dtype="int32")
    assert tensor.size == 480
    assert tensor.shape == [1, 2, 3, 4, 5]
    assert tensor.dtype == "int32"

    with tensor.map() as m:
        m[0] = 10
        assert m[0] == 10
        m[1] = -2000000000
        assert m[1] == -2000000000

        if hasattr(m, "__getbuffer__"):
            with memoryview(m) as v:
                assert v.format == "i"
                assert v.ndim == 5
                assert v.shape == (1, 2, 3, 4, 5)
                assert v.itemsize == 4
                assert v[0, 0, 0, 0, 0] == 10
                assert v[0, 0, 0, 0, 1] == -2000000000
            with pytest.raises(ValueError):
                _ = v[0, 0, 0, 0, 0]
        else:
            n = np.frombuffer(m.view(), dtype=tensor.dtype).reshape(tensor.shape)
            assert n[0, 0, 0, 0, 0] == 10
            assert n[0, 0, 0, 0, 1] == -2000000000
    with pytest.raises(BufferError):
        _ = m[0]


def test_u32():
    tensor = Tensor([1, 2, 3, 4, 5], dtype="uint32")
    assert tensor.size == 480
    assert tensor.shape == [1, 2, 3, 4, 5]
    assert tensor.dtype == "uint32"

    with tensor.map() as m:
        m[0] = 10
        assert m[0] == 10
        m[1] = 4000000000
        assert m[1] == 4000000000

        if hasattr(m, "__getbuffer__"):
            with memoryview(m) as v:
                assert v.format == "I"
                assert v.ndim == 5
                assert v.shape == (1, 2, 3, 4, 5)
                assert v.itemsize == 4
                assert v[0, 0, 0, 0, 0] == 10
                assert v[0, 0, 0, 0, 1] == 4000000000
            with pytest.raises(ValueError):
                _ = v[0, 0, 0, 0, 0]
        else:
            n = np.frombuffer(m.view(), dtype=tensor.dtype).reshape(tensor.shape)
            assert n[0, 0, 0, 0, 0] == 10
            assert n[0, 0, 0, 0, 1] == 4000000000
    with pytest.raises(BufferError):
        _ = m[0]


def test_i64():
    tensor = Tensor([1, 2, 3, 4, 5], dtype="int64")
    assert tensor.size == 960
    assert tensor.shape == [1, 2, 3, 4, 5]
    assert tensor.dtype == "int64"

    with tensor.map() as m:
        m[0] = 10
        assert m[0] == 10
        m[1] = -9000000000000000000
        assert m[1] == -9000000000000000000

        if hasattr(m, "__getbuffer__"):
            with memoryview(m) as v:
                assert v.format == "q"
                assert v.ndim == 5
                assert v.shape == (1, 2, 3, 4, 5)
                assert v.itemsize == 8
                assert v[0, 0, 0, 0, 0] == 10
                assert v[0, 0, 0, 0, 1] == -9000000000000000000
            with pytest.raises(ValueError):
                _ = v[0, 0, 0, 0, 0]
        else:
            n = np.frombuffer(m.view(), dtype=tensor.dtype).reshape(tensor.shape)
            assert n[0, 0, 0, 0, 0] == 10
            assert n[0, 0, 0, 0, 1] == -9000000000000000000
    with pytest.raises(BufferError):
        _ = m[0]


def test_u64():
    tensor = Tensor([1, 2, 3, 4, 5], dtype="uint64")
    assert tensor.size == 960
    assert tensor.shape == [1, 2, 3, 4, 5]
    assert tensor.dtype == "uint64"

    with tensor.map() as m:
        m[0] = 10
        assert m[0] == 10
        m[1] = 18000000000000000000
        assert m[1] == 18000000000000000000

        if hasattr(m, "__getbuffer__"):
            with memoryview(m) as v:
                assert v.format == "Q"
                assert v.ndim == 5
                assert v.shape == (1, 2, 3, 4, 5)
                assert v.itemsize == 8
                assert v[0, 0, 0, 0, 0] == 10
                assert v[0, 0, 0, 0, 1] == 18000000000000000000
            with pytest.raises(ValueError):
                _ = v[0, 0, 0, 0, 0]
        else:
            n = np.frombuffer(m.view(), dtype=tensor.dtype).reshape(tensor.shape)
            assert n[0, 0, 0, 0, 0] == 10
            assert n[0, 0, 0, 0, 1] == 18000000000000000000
    with pytest.raises(BufferError):
        _ = m[0]


# TODO: Enable once f16 support is added.
# def test_f16():
#     tensor = Tensor([1.0, 2.0, 3.0, 4.0, 5.0], dtype='f16')
#     assert tensor.size == 240
#     assert tensor.shape == [1, 2, 3, 4, 5]
#     assert tensor.dtype == 'f16'


def test_f32():
    tensor = Tensor([1, 2, 3, 4, 5], dtype="float32")
    assert tensor.size == 480
    assert tensor.shape == [1, 2, 3, 4, 5]
    assert tensor.dtype == "float32"

    with tensor.map() as m:
        m[0] = 10.0
        assert m[0] == 10.0
        m[1] = -2000000000.0
        assert m[1] == -2000000000.0

        if hasattr(m, "__getbuffer__"):
            with memoryview(m) as v:
                assert v.format == "f"
                assert v.ndim == 5
                assert v.shape == (1, 2, 3, 4, 5)
                assert v.itemsize == 4
                assert v[0, 0, 0, 0, 0] == 10.0
                assert v[0, 0, 0, 0, 1] == -2000000000.0
            with pytest.raises(ValueError):
                _ = v[0, 0, 0, 0, 0]
        else:
            n = np.frombuffer(m.view(), dtype=tensor.dtype).reshape(tensor.shape)
            assert n[0, 0, 0, 0, 0] == 10.0
            assert n[0, 0, 0, 0, 1] == -2000000000.0
    with pytest.raises(BufferError):
        _ = m[0]


def test_f64():
    tensor = Tensor([1, 2, 3, 4, 5], dtype="float64")
    assert tensor.size == 960
    assert tensor.shape == [1, 2, 3, 4, 5]
    assert tensor.dtype == "float64"

    with tensor.map() as m:
        m[0] = 10.0
        assert m[0] == 10.0
        m[1] = -9000000000000000000.0
        assert m[1] == -9000000000000000000.0

        if hasattr(m, "__getbuffer__"):
            with memoryview(m) as v:
                assert v.format == "d"
                assert v.ndim == 5
                assert v.shape == (1, 2, 3, 4, 5)
                assert v.itemsize == 8
                assert v[0, 0, 0, 0, 0] == 10.0
                assert v[0, 0, 0, 0, 1] == -9000000000000000000.0
            with pytest.raises(ValueError):
                _ = v[0, 0, 0, 0, 0]
        else:
            n = np.frombuffer(m.view(), dtype=tensor.dtype).reshape(tensor.shape)
            assert n[0, 0, 0, 0, 0] == 10.0
            assert n[0, 0, 0, 0, 1] == -9000000000000000000.0
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

    elapsed = 0
    elapsed_copy = 0

    for _ in range(10):  # Run multiple times to get a better measurement
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

    assert elapsed < elapsed_copy


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

    elapsed = 0
    elapsed_copy = 0

    for _ in range(10):  # Run multiple times to get a better measurement
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

    assert elapsed < elapsed_copy


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
