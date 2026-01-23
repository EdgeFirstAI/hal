# SPDX-FileCopyrightText: Copyright 2025 Au-Zone Technologies
# SPDX-License-Identifier: Apache-2.0

import numpy as np
from edgefirst_hal import Tensor, TensorMemory
from unittest import TestCase


class TestTensor(TestCase):
    def test_int8(self):
        tensor = Tensor([1, 2, 3, 4, 5], dtype="int8")
        self.assertEqual(tensor.size, 120)
        self.assertEqual(tensor.shape, [1, 2, 3, 4, 5])
        self.assertEqual(tensor.dtype, "int8")

        with tensor.map() as m:
            m[0] = 10
            self.assertEqual(m[0], 10)
            m[1] = -120
            self.assertEqual(m[1], -120)

            if hasattr(m, "__getbuffer__"):
                with memoryview(m) as v:
                    self.assertEqual(v.format, "b")
                    self.assertEqual(v.ndim, 5)
                    self.assertEqual(v.shape, (1, 2, 3, 4, 5))
                    self.assertEqual(v.itemsize, 1)
                    self.assertEqual(v[0, 0, 0, 0, 0], 10)
                    self.assertEqual(v[0, 0, 0, 0, 1], -120)
                self.assertRaises(ValueError, lambda: v[0, 0, 0, 0, 0])
            else:
                n = np.frombuffer(m.view(), dtype=tensor.dtype).reshape(tensor.shape)
                self.assertEqual(n[0, 0, 0, 0, 0], 10)
                self.assertEqual(n[0, 0, 0, 0, 1], -120)
        self.assertRaises(BufferError, lambda: m[0])

    def test_u8(self):
        tensor = Tensor([1, 2, 3, 4, 5], dtype="uint8")
        self.assertEqual(tensor.size, 120)
        self.assertEqual(tensor.shape, [1, 2, 3, 4, 5])
        self.assertEqual(tensor.dtype, "uint8")

        with tensor.map() as m:
            m[0] = 10
            self.assertEqual(m[0], 10)
            m[1] = 250
            self.assertEqual(m[1], 250)

            if hasattr(m, "__getbuffer__"):
                with memoryview(m) as v:
                    self.assertEqual(v.format, "B")
                    self.assertEqual(v.ndim, 5)
                    self.assertEqual(v.shape, (1, 2, 3, 4, 5))
                    self.assertEqual(v.itemsize, 1)
                    self.assertEqual(v[0, 0, 0, 0, 0], 10)
                    self.assertEqual(v[0, 0, 0, 0, 1], 250)
                self.assertRaises(ValueError, lambda: v[0, 0, 0, 0, 0])
            else:
                n = np.frombuffer(m.view(), dtype=tensor.dtype).reshape(tensor.shape)
                self.assertEqual(n[0, 0, 0, 0, 0], 10)
                self.assertEqual(n[0, 0, 0, 0, 1], 250)
        self.assertRaises(BufferError, lambda: m[0])

    def test_i16(self):
        tensor = Tensor([1, 2, 3, 4, 5], dtype="int16")
        self.assertEqual(tensor.size, 240)
        self.assertEqual(tensor.shape, [1, 2, 3, 4, 5])
        self.assertEqual(tensor.dtype, "int16")

        with tensor.map() as m:
            m[0] = 10
            self.assertEqual(m[0], 10)
            m[1] = -30000
            self.assertEqual(m[1], -30000)

            if hasattr(m, "__getbuffer__"):
                with memoryview(m) as v:
                    self.assertEqual(v.format, "h")
                    self.assertEqual(v.ndim, 5)
                    self.assertEqual(v.shape, (1, 2, 3, 4, 5))
                    self.assertEqual(v.itemsize, 2)
                    self.assertEqual(v[0, 0, 0, 0, 0], 10)
                    self.assertEqual(v[0, 0, 0, 0, 1], -30000)
                self.assertRaises(ValueError, lambda: v[0, 0, 0, 0, 0])
            else:
                n = np.frombuffer(m.view(), dtype=tensor.dtype).reshape(tensor.shape)
                self.assertEqual(n[0, 0, 0, 0, 0], 10)
                self.assertEqual(n[0, 0, 0, 0, 1], -30000)
        self.assertRaises(BufferError, lambda: m[0])

    def test_u16(self):
        tensor = Tensor([1, 2, 3, 4, 5], dtype="uint16")
        self.assertEqual(tensor.size, 240)
        self.assertEqual(tensor.shape, [1, 2, 3, 4, 5])
        self.assertEqual(tensor.dtype, "uint16")

        with tensor.map() as m:
            m[0] = 10
            self.assertEqual(m[0], 10)
            m[1] = 60000
            self.assertEqual(m[1], 60000)

            if hasattr(m, "__getbuffer__"):
                with memoryview(m) as v:
                    self.assertEqual(v.format, "H")
                    self.assertEqual(v.ndim, 5)
                    self.assertEqual(v.shape, (1, 2, 3, 4, 5))
                    self.assertEqual(v.itemsize, 2)
                    self.assertEqual(v[0, 0, 0, 0, 0], 10)
                    self.assertEqual(v[0, 0, 0, 0, 1], 60000)
                self.assertRaises(ValueError, lambda: v[0, 0, 0, 0, 0])
            else:
                n = np.frombuffer(m.view(), dtype=tensor.dtype).reshape(tensor.shape)
                self.assertEqual(n[0, 0, 0, 0, 0], 10)
                self.assertEqual(n[0, 0, 0, 0, 1], 60000)
        self.assertRaises(BufferError, lambda: m[0])

    def test_i32(self):
        tensor = Tensor([1, 2, 3, 4, 5], dtype="int32")
        self.assertEqual(tensor.size, 480)
        self.assertEqual(tensor.shape, [1, 2, 3, 4, 5])
        self.assertEqual(tensor.dtype, "int32")

        with tensor.map() as m:
            m[0] = 10
            self.assertEqual(m[0], 10)
            m[1] = -2000000000
            self.assertEqual(m[1], -2000000000)

            if hasattr(m, "__getbuffer__"):
                with memoryview(m) as v:
                    self.assertEqual(v.format, "i")
                    self.assertEqual(v.ndim, 5)
                    self.assertEqual(v.shape, (1, 2, 3, 4, 5))
                    self.assertEqual(v.itemsize, 4)
                    self.assertEqual(v[0, 0, 0, 0, 0], 10)
                    self.assertEqual(v[0, 0, 0, 0, 1], -2000000000)
                self.assertRaises(ValueError, lambda: v[0, 0, 0, 0, 0])
            else:
                n = np.frombuffer(m.view(), dtype=tensor.dtype).reshape(tensor.shape)
                self.assertEqual(n[0, 0, 0, 0, 0], 10)
                self.assertEqual(n[0, 0, 0, 0, 1], -2000000000)
        self.assertRaises(BufferError, lambda: m[0])

    def test_u32(self):
        tensor = Tensor([1, 2, 3, 4, 5], dtype="uint32")
        self.assertEqual(tensor.size, 480)
        self.assertEqual(tensor.shape, [1, 2, 3, 4, 5])
        self.assertEqual(tensor.dtype, "uint32")

        with tensor.map() as m:
            m[0] = 10
            self.assertEqual(m[0], 10)
            m[1] = 4000000000
            self.assertEqual(m[1], 4000000000)

            if hasattr(m, "__getbuffer__"):
                with memoryview(m) as v:
                    self.assertEqual(v.format, "I")
                    self.assertEqual(v.ndim, 5)
                    self.assertEqual(v.shape, (1, 2, 3, 4, 5))
                    self.assertEqual(v.itemsize, 4)
                    self.assertEqual(v[0, 0, 0, 0, 0], 10)
                    self.assertEqual(v[0, 0, 0, 0, 1], 4000000000)
                self.assertRaises(ValueError, lambda: v[0, 0, 0, 0, 0])
            else:
                n = np.frombuffer(m.view(), dtype=tensor.dtype).reshape(tensor.shape)
                self.assertEqual(n[0, 0, 0, 0, 0], 10)
                self.assertEqual(n[0, 0, 0, 0, 1], 4000000000)
        self.assertRaises(BufferError, lambda: m[0])

    def test_i64(self):
        tensor = Tensor([1, 2, 3, 4, 5], dtype="int64")
        self.assertEqual(tensor.size, 960)
        self.assertEqual(tensor.shape, [1, 2, 3, 4, 5])
        self.assertEqual(tensor.dtype, "int64")

        with tensor.map() as m:
            m[0] = 10
            self.assertEqual(m[0], 10)
            m[1] = -9000000000000000000
            self.assertEqual(m[1], -9000000000000000000)

            if hasattr(m, "__getbuffer__"):
                with memoryview(m) as v:
                    self.assertEqual(v.format, "q")
                    self.assertEqual(v.ndim, 5)
                    self.assertEqual(v.shape, (1, 2, 3, 4, 5))
                    self.assertEqual(v.itemsize, 8)
                    self.assertEqual(v[0, 0, 0, 0, 0], 10)
                    self.assertEqual(v[0, 0, 0, 0, 1], -9000000000000000000)
                self.assertRaises(ValueError, lambda: v[0, 0, 0, 0, 0])
            else:
                n = np.frombuffer(m.view(), dtype=tensor.dtype).reshape(tensor.shape)
                self.assertEqual(n[0, 0, 0, 0, 0], 10)
                self.assertEqual(n[0, 0, 0, 0, 1], -9000000000000000000)
        self.assertRaises(BufferError, lambda: m[0])

    def test_u64(self):
        tensor = Tensor([1, 2, 3, 4, 5], dtype="uint64")
        self.assertEqual(tensor.size, 960)
        self.assertEqual(tensor.shape, [1, 2, 3, 4, 5])
        self.assertEqual(tensor.dtype, "uint64")

        with tensor.map() as m:
            m[0] = 10
            self.assertEqual(m[0], 10)
            m[1] = 18000000000000000000
            self.assertEqual(m[1], 18000000000000000000)

            if hasattr(m, "__getbuffer__"):
                with memoryview(m) as v:
                    self.assertEqual(v.format, "Q")
                    self.assertEqual(v.ndim, 5)
                    self.assertEqual(v.shape, (1, 2, 3, 4, 5))
                    self.assertEqual(v.itemsize, 8)
                    self.assertEqual(v[0, 0, 0, 0, 0], 10)
                    self.assertEqual(v[0, 0, 0, 0, 1], 18000000000000000000)
                self.assertRaises(ValueError, lambda: v[0, 0, 0, 0, 0])
            else:
                n = np.frombuffer(m.view(), dtype=tensor.dtype).reshape(tensor.shape)
                self.assertEqual(n[0, 0, 0, 0, 0], 10)
                self.assertEqual(n[0, 0, 0, 0, 1], 18000000000000000000)
        self.assertRaises(BufferError, lambda: m[0])

    # TODO: Enable once f16 support is added.
    # def test_f16(self):
    #     tensor = Tensor([1.0, 2.0, 3.0, 4.0, 5.0], dtype='f16')
    #     self.assertEqual(tensor.size, 240)
    #     self.assertEqual(tensor.shape, [1, 2, 3, 4, 5])
    #     self.assertEqual(tensor.dtype, 'f16')

    def test_f32(self):
        tensor = Tensor([1, 2, 3, 4, 5], dtype="float32")
        self.assertEqual(tensor.size, 480)
        self.assertEqual(tensor.shape, [1, 2, 3, 4, 5])
        self.assertEqual(tensor.dtype, "float32")

        with tensor.map() as m:
            m[0] = 10.0
            self.assertEqual(m[0], 10.0)
            m[1] = -2000000000.0
            self.assertEqual(m[1], -2000000000.0)

            if hasattr(m, "__getbuffer__"):
                with memoryview(m) as v:
                    self.assertEqual(v.format, "f")
                    self.assertEqual(v.ndim, 5)
                    self.assertEqual(v.shape, (1, 2, 3, 4, 5))
                    self.assertEqual(v.itemsize, 4)
                    self.assertEqual(v[0, 0, 0, 0, 0], 10.0)
                    self.assertEqual(v[0, 0, 0, 0, 1], -2000000000.0)
                self.assertRaises(ValueError, lambda: v[0, 0, 0, 0, 0])
            else:
                n = np.frombuffer(m.view(), dtype=tensor.dtype).reshape(tensor.shape)
                self.assertEqual(n[0, 0, 0, 0, 0], 10.0)
                self.assertEqual(n[0, 0, 0, 0, 1], -2000000000.0)
        self.assertRaises(BufferError, lambda: m[0])

    def test_f64(self):
        tensor = Tensor([1, 2, 3, 4, 5], dtype="float64")
        self.assertEqual(tensor.size, 960)
        self.assertEqual(tensor.shape, [1, 2, 3, 4, 5])
        self.assertEqual(tensor.dtype, "float64")

        with tensor.map() as m:
            m[0] = 10.0
            self.assertEqual(m[0], 10.0)
            m[1] = -9000000000000000000.0
            self.assertEqual(m[1], -9000000000000000000.0)

            if hasattr(m, "__getbuffer__"):
                with memoryview(m) as v:
                    self.assertEqual(v.format, "d")
                    self.assertEqual(v.ndim, 5)
                    self.assertEqual(v.shape, (1, 2, 3, 4, 5))
                    self.assertEqual(v.itemsize, 8)
                    self.assertEqual(v[0, 0, 0, 0, 0], 10.0)
                    self.assertEqual(v[0, 0, 0, 0, 1], -9000000000000000000.0)
                self.assertRaises(ValueError, lambda: v[0, 0, 0, 0, 0])
            else:
                n = np.frombuffer(m.view(), dtype=tensor.dtype).reshape(tensor.shape)
                self.assertEqual(n[0, 0, 0, 0, 0], 10.0)
                self.assertEqual(n[0, 0, 0, 0, 1], -9000000000000000000.0)
        self.assertRaises(BufferError, lambda: m[0])

    def test_from_fd_dma(self):
        try:
            tensor = Tensor([100, 100, 3], dtype="uint8", mem=TensorMemory.DMA)
        except (AttributeError, RuntimeError):
            self.skipTest("DMA memory not supported on this platform")

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

    def test_dma_zero_copy_perf(self):

        try:
            tensor = Tensor([100, 100, 3], dtype="uint8", mem=TensorMemory.DMA)
        except (AttributeError, RuntimeError):
            self.skipTest("DMA memory not supported on this platform")

        assert tensor.memory == TensorMemory.DMA

        import time
        start = time.perf_counter()
        tensor_fd = Tensor.from_fd(tensor.fd, tensor.shape, tensor.dtype)
        elapsed = time.perf_counter() - start

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

        tensor_copy = Tensor(
            tensor.shape, dtype=tensor.dtype, mem=tensor.memory)
        start = time.perf_counter()
        with tensor.map() as src, tensor_copy.map() as dst:
            for i in range(tensor.size):
                dst[i] = src[i]
        elapsed_copy = time.perf_counter() - start

        assert elapsed < elapsed_copy

    def test_from_fd_shm(self):
        try:
            tensor = Tensor([100, 100, 3], dtype="uint8", mem=TensorMemory.SHM)
        except (AttributeError, RuntimeError):
            self.skipTest("SHM memory not supported on this platform")

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

    def test_shm_zero_copy_perf(self):
        try:
            tensor = Tensor([100, 100, 3], dtype="uint8", mem=TensorMemory.SHM)
        except (AttributeError, RuntimeError):
            self.skipTest("SHM memory not supported on this platform")

        assert tensor.memory == TensorMemory.SHM

        import time
        start = time.perf_counter()
        tensor_fd = Tensor.from_fd(tensor.fd, tensor.shape, tensor.dtype)
        elapsed = time.perf_counter() - start

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

        tensor_copy = Tensor(
            tensor.shape, dtype=tensor.dtype, mem=tensor.memory)
        start = time.perf_counter()
        with tensor.map() as src, tensor_copy.map() as dst:
            for i in range(tensor.size):
                dst[i] = src[i]
        elapsed_copy = time.perf_counter() - start

        assert elapsed < elapsed_copy
