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


@pytest.mark.parametrize(
    "dtype,size,fmt,itemsize,val1,val2", DTYPE_PARAMS, ids=[p[0] for p in DTYPE_PARAMS]
)
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
        fd = original.fd  # .fd returns a dup — caller must close
        tensor_fd = Tensor.from_fd(fd, original.shape, original.dtype)
        os.close(fd)  # close the dup from .fd (from_fd dups again internally)
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


# ---------------------------------------------------------------------------
# from_numpy tests
# ---------------------------------------------------------------------------

# fmt: off
FROM_NUMPY_DTYPE_PARAMS = [
    "uint8", "int8", "uint16", "int16",
    "uint32", "int32", "uint64", "int64",
    "float32", "float64",
]
# fmt: on


def _read_tensor(tensor, dtype):
    """Read all tensor data back via map() as a shaped numpy array."""
    with tensor.map() as m:
        return np.frombuffer(m.view(), dtype=dtype).reshape(tensor.shape).copy()


@pytest.mark.parametrize("dtype", FROM_NUMPY_DTYPE_PARAMS)
def test_from_numpy_contiguous_dtypes(dtype):
    """Path 1 (contiguous memcpy): round-trip for all 10 supported dtypes."""
    arr = np.arange(1, 25, dtype=dtype).reshape(4, 6)
    t = Tensor(list(arr.shape), dtype=dtype)
    t.from_numpy(arr)
    result = _read_tensor(t, dtype)
    assert np.array_equal(result, arr)


def test_from_numpy_contiguous_large():
    """Path 1 parallel (>=256 KiB): contiguous float32 above the parallelism threshold."""
    # 256 KiB = 262144 bytes; float32 = 4 bytes → 65536 elements minimum
    rows, cols = 256, 512  # 131072 float32 elements = 524288 bytes
    arr = np.arange(rows * cols, dtype="float32").reshape(rows, cols)
    t = Tensor([rows, cols], dtype="float32")
    t.from_numpy(arr)
    result = _read_tensor(t, "float32")
    assert np.array_equal(result, arr)


def test_from_numpy_strided_column_slice():
    """Path 2 (strided outer, contiguous inner rows): column slice big[:, :N]."""
    big = np.arange(1, 10 * 20 + 1, dtype="uint16").reshape(10, 20)
    src = big[:, :12]  # non-contiguous outer stride, contiguous rows of 12
    t = Tensor(list(src.shape), dtype="uint16")
    t.from_numpy(src)
    result = _read_tensor(t, "uint16")
    assert np.array_equal(result, src)


def test_from_numpy_strided_3d_subvolume():
    """Path 2 extended: 3D sub-volume big[:A, :B, :C] with strided outer dims."""
    big = np.arange(1, 8 * 16 * 32 + 1, dtype="int32").reshape(8, 16, 32)
    src = big[:5, :10, :24]  # strided first two dims, contiguous innermost
    t = Tensor(list(src.shape), dtype="int32")
    t.from_numpy(src)
    result = _read_tensor(t, "int32")
    assert np.array_equal(result, src)


def test_from_numpy_strided_large():
    """Path 2 parallel (>=256 KiB): large column slice exercises parallel row-copy."""
    # 512 rows x 400 uint16 columns = 400 KiB
    big = np.arange(512 * 512, dtype="uint16").reshape(512, 512)
    src = big[:, :400]  # 512 * 400 * 2 = 409600 bytes > 256 KiB
    t = Tensor(list(src.shape), dtype="uint16")
    t.from_numpy(src)
    result = _read_tensor(t, "uint16")
    assert np.array_equal(result, src)


def test_from_numpy_negative_stride():
    """Negative outer stride (arr[::-1]): Path 2 with reversed row order."""
    arr = np.arange(1, 8 * 10 + 1, dtype="float32").reshape(8, 10)
    src = arr[::-1]  # reversed rows; inner dim is still contiguous
    t = Tensor(list(src.shape), dtype="float32")
    t.from_numpy(src)
    result = _read_tensor(t, "float32")
    assert np.array_equal(result, src)


def test_from_numpy_negative_stride_both():
    """Negative strides on both axes (arr[::-1, ::-1]): Path 3 per-element."""
    arr = np.arange(1, 7 * 9 + 1, dtype="int16").reshape(7, 9)
    src = arr[::-1, ::-1]
    t = Tensor(list(src.shape), dtype="int16")
    t.from_numpy(src)
    result = _read_tensor(t, "int16")
    assert np.array_equal(result, src)


def test_from_numpy_transposed():
    """Transpose (.T) produces fully strided layout: Path 3."""
    arr = np.arange(1, 6 * 8 + 1, dtype="float64").reshape(6, 8)
    src = arr.T  # shape (8, 6), non-contiguous in both dims
    t = Tensor(list(src.shape), dtype="float64")
    t.from_numpy(src)
    result = _read_tensor(t, "float64")
    assert np.array_equal(result, src)


def test_from_numpy_every_other():
    """Row stride of 2 (arr[::2]): Path 3 because inner dim has non-unit outer step."""
    arr = np.arange(1, 20 * 10 + 1, dtype="int8").reshape(20, 10)
    src = arr[::2]  # 10 rows selected from 20
    t = Tensor(list(src.shape), dtype="int8")
    t.from_numpy(src)
    result = _read_tensor(t, "int8")
    assert np.array_equal(result, src)


def test_from_numpy_step_2d():
    """Step > 1 on both axes (arr[::2, ::3]): Path 3 fully strided."""
    arr = np.arange(1, 30 * 15 + 1, dtype="uint32").reshape(30, 15)
    src = arr[::2, ::3]  # shape (15, 5)
    t = Tensor(list(src.shape), dtype="uint32")
    t.from_numpy(src)
    result = _read_tensor(t, "uint32")
    assert np.array_equal(result, src)


def test_from_numpy_large_transposed():
    """Path 3 parallel (>=256 KiB): large transposed float32 array."""
    # 512 x 512 float32 transposed = 1 MiB
    arr = np.arange(512 * 512, dtype="float32").reshape(512, 512)
    src = arr.T  # (512, 512) fully strided
    t = Tensor(list(src.shape), dtype="float32")
    t.from_numpy(src)
    result = _read_tensor(t, "float32")
    assert np.array_equal(result, src)


def test_from_numpy_fortran_order():
    """Fortran-order (column-major) array: non-contiguous in C layout."""
    arr = np.asfortranarray(np.arange(1, 6 * 8 + 1, dtype="int32").reshape(6, 8))
    assert not arr.flags["C_CONTIGUOUS"]
    t = Tensor(list(arr.shape), dtype="int32")
    t.from_numpy(arr)
    result = _read_tensor(t, "int32")
    assert np.array_equal(result, arr)


def test_from_numpy_single_element():
    """Degenerate case: 1-element array (scalar tensor)."""
    arr = np.array([[42]], dtype="uint8")
    t = Tensor([1, 1], dtype="uint8")
    t.from_numpy(arr)
    result = _read_tensor(t, "uint8")
    assert np.array_equal(result, arr)


def test_from_numpy_1d():
    """1D contiguous array: simplest Path 1 case."""
    arr = np.arange(1, 65, dtype="int64")
    t = Tensor([64], dtype="int64")
    t.from_numpy(arr)
    result = _read_tensor(t, "int64")
    assert np.array_equal(result, arr)


def test_from_numpy_size1_dim():
    """Shape with a size-1 dimension does not confuse stride classification."""
    arr = np.arange(1, 13, dtype="float32").reshape(1, 12)
    t = Tensor([1, 12], dtype="float32")
    t.from_numpy(arr)
    result = _read_tensor(t, "float32")
    assert np.array_equal(result, arr)


def test_from_numpy_dtype_mismatch():
    """Passing a float32 array into an int32 tensor must raise RuntimeError."""
    arr = np.arange(1, 13, dtype="float32").reshape(3, 4)
    t = Tensor([3, 4], dtype="int32")
    with pytest.raises(RuntimeError):
        t.from_numpy(arr)


def test_from_numpy_size_mismatch():
    """Wrong element count (tensor has more elements than array) raises RuntimeError."""
    arr = np.arange(1, 13, dtype="uint8").reshape(3, 4)  # 12 elements
    t = Tensor([4, 4], dtype="uint8")  # 16 elements
    with pytest.raises(RuntimeError):
        t.from_numpy(arr)


def test_from_numpy_non_array():
    """Passing a plain Python list instead of a numpy array raises RuntimeError."""
    t = Tensor([3, 4], dtype="uint8")
    with pytest.raises((RuntimeError, TypeError)):
        t.from_numpy([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
