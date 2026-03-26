# SPDX-FileCopyrightText: Copyright 2025 Au-Zone Technologies
# SPDX-License-Identifier: Apache-2.0

"""Shared test utilities for decoder tests."""

import numpy as np

import edgefirst_hal


def numpy_to_tensor(
    arr: np.ndarray, mem: edgefirst_hal.TensorMemory | None = None
) -> edgefirst_hal.Tensor:
    """Create a HAL Tensor from a numpy array, copying the data.

    Supports any dtype/shape that edgefirst_hal.Tensor supports.
    """
    dtype_map = {
        np.dtype("int8"): "int8",
        np.dtype("uint8"): "uint8",
        np.dtype("int16"): "int16",
        np.dtype("uint16"): "uint16",
        np.dtype("int32"): "int32",
        np.dtype("uint32"): "uint32",
        np.dtype("int64"): "int64",
        np.dtype("uint64"): "uint64",
        np.dtype("float32"): "float32",
        np.dtype("float64"): "float64",
    }
    hal_dtype = dtype_map.get(arr.dtype)
    if hal_dtype is None:
        raise ValueError(f"Unsupported numpy dtype: {arr.dtype}")

    # Ensure contiguous C-order array
    arr = np.ascontiguousarray(arr)

    tensor = edgefirst_hal.Tensor(list(arr.shape), dtype=hal_dtype, mem=mem)
    with tensor.map() as m:
        dst = np.frombuffer(m, dtype=arr.dtype).reshape(arr.shape)
        np.copyto(dst, arr)
    return tensor
