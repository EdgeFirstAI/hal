# SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
# SPDX-License-Identifier: Apache-2.0
"""Windows D3D11 texture tensors from Python."""

import ctypes
import os
import sys
import time

import numpy as np
import pytest
from edgefirst.tensor import (
    PixelFormat,
    Tensor,
    TensorMemory,
    is_cuda_available,
    is_gpu_buffer_available,
)

windows_only = pytest.mark.skipif(
    sys.platform != "win32", reason="D3D11 textures are Windows-only"
)


def _warp_adapter_requested() -> bool:
    """True when this process was pointed at the WARP software adapter.

    WARP has no CUDA device behind it, so a texture allocated on it carries
    no CUDA handle and maps to ``None``.

    Follows the HAL's own precedence (``d3d11/adapter.rs``): the primary
    name wins, and the alias is read only when the primary is unset or
    blank. Reading either would skip a genuine regression on the hardware
    adapter whenever a stale ``EDGEFIRST_ANGLE_ADAPTER=warp`` is left in the
    shell -- the exact configuration TESTING.md warns about.
    """
    primary = os.environ.get("EDGEFIRST_D3D11_ADAPTER", "").strip()
    chosen = primary or os.environ.get("EDGEFIRST_ANGLE_ADAPTER", "").strip()
    return chosen.lower() == "warp"


def _close(handle: int) -> None:
    """Close a Windows handle a completion or handle call handed back.

    A plain call, never an assertion: these run in `finally` blocks, where a
    failing assert would replace the real failure with this one.
    """
    ctypes.windll.kernel32.CloseHandle(ctypes.c_void_p(handle))


def _write_first_pixel(t, texels):
    """Write the first texel of row 0 through the map, honouring the pitch."""
    with t.map() as m:
        np.asarray(memoryview(m))[0, 0] = texels


def _read_first_pixel(t):
    with t.map("read") as m:
        return list(np.asarray(memoryview(m))[0, 0])


@windows_only
@pytest.mark.gpu
def test_texture_tensor_round_trip_and_handles():
    if not is_gpu_buffer_available():
        pytest.skip("no D3D11 device")
    t = Tensor.image(64, 32, PixelFormat.Rgba, TensorMemory.DMABUF, "readwrite")
    assert t.memory == TensorMemory.DMABUF
    assert t.d3d11_texture()
    lay = t.d3d11_layout()
    assert (lay.texture_width, lay.texture_height, lay.bytes_per_texel) == (64, 32, 4)
    _write_first_pixel(t, [1, 2, 3, 4])
    assert t.gpu_completion() is None
    t.set_gpu_write(7)
    completion = t.gpu_completion()
    assert completion is not None, "a recorded write must surface as a completion"
    fence, value = completion
    try:
        assert value == 7
    finally:
        _close(fence)
    h = t.d3d11_shared_handle()
    try:
        again = Tensor.from_d3d11_shared_handle(
            h, [32, 64, 4], "uint8", PixelFormat.Rgba, "read"
        )
        assert _read_first_pixel(again) == [1, 2, 3, 4]
    finally:
        _close(h)


@windows_only
@pytest.mark.gpu
def test_texture_import_and_module_device():
    if not is_gpu_buffer_available():
        pytest.skip("no D3D11 device")
    from edgefirst.tensor import d3d11_device, d3d11_use_external_device

    device = d3d11_device()
    assert device
    # The process device exists by now, so adopting one is refused.
    with pytest.raises(RuntimeError):
        d3d11_use_external_device(device)

    t = Tensor.image(64, 32, PixelFormat.Rgba, TensorMemory.DMABUF, "readwrite")
    _write_first_pixel(t, [9, 8, 7, 6])
    again = Tensor.from_d3d11_texture(
        t.d3d11_texture(), [32, 64, 4], "uint8", PixelFormat.Rgba, "read"
    )
    assert _read_first_pixel(again) == [9, 8, 7, 6]
    # The shape must describe the texture the constructor measured; [8, 8, 4]
    # is a well-formed Rgba shape belonging to a different image.
    for wrong in ([32, 64], [8, 8, 4]):
        with pytest.raises(ValueError):
            Tensor.from_d3d11_texture(
                t.d3d11_texture(), wrong, "uint8", PixelFormat.Rgba, "read"
            )


@windows_only
@pytest.mark.gpu
def test_semi_planar_accepts_either_shape_spelling():
    """NV12 640x480 allocates 720 combined rows but addresses a 480-row grid.

    Nothing in a rank-2 shape distinguishes the two, so the constructors read
    the dimensions off the texture and accept either spelling of them. A
    third rank-2 shape that is neither is an argument error.
    """
    if not is_gpu_buffer_available():
        pytest.skip("no D3D11 device")
    t = Tensor.image(640, 480, PixelFormat.Nv12, TensorMemory.DMABUF, "readwrite")
    assert t.d3d11_texture()
    assert t.d3d11_layout().texture_height == 720
    with t.map() as m:
        np.asarray(memoryview(m))[0, :4] = [11, 22, 33, 44]

    h = t.d3d11_shared_handle()
    try:
        for shape in ([720, 640], [480, 640]):
            again = Tensor.from_d3d11_shared_handle(
                h, shape, "uint8", PixelFormat.Nv12, "read"
            )
            # Both spellings recover the image the texture holds, not the
            # 320-row one a shape-derived reading of [480, 640] would infer.
            assert (again.width, again.height) == (640, 480), f"shape {shape}"
            with again.map("read") as m:
                first = list(np.asarray(memoryview(m))[0, :4])
            assert first == [11, 22, 33, 44], f"shape {shape}"
        # 479 rows is neither the combined plane height nor the luma height.
        with pytest.raises(ValueError):
            Tensor.from_d3d11_shared_handle(
                h, [479, 640], "uint8", PixelFormat.Nv12, "read"
            )
    finally:
        _close(h)


@windows_only
@pytest.mark.gpu
def test_constructors_accept_the_documented_default_access():
    """``access="none"`` is the documented default of both constructors.

    It is a declaration ("provision no CPU staging"), not a map direction,
    and the dynamic-backend wheel used to refuse it outright.
    """
    if not is_gpu_buffer_available():
        pytest.skip("no D3D11 device")
    t = Tensor.image(64, 32, PixelFormat.Rgba, TensorMemory.DMABUF, "readwrite")
    wrapped = Tensor.from_d3d11_texture(
        t.d3d11_texture(), [32, 64, 4], "uint8", PixelFormat.Rgba
    )
    assert wrapped.d3d11_texture() == t.d3d11_texture()

    h = t.d3d11_shared_handle()
    try:
        opened = Tensor.from_d3d11_shared_handle(
            h, [32, 64, 4], "uint8", PixelFormat.Rgba
        )
        assert opened.d3d11_texture()
    finally:
        _close(h)


@windows_only
@pytest.mark.gpu
def test_shared_handle_hands_out_a_fresh_duplicate_each_call():
    if not is_gpu_buffer_available():
        pytest.skip("no D3D11 device")
    t = Tensor.image(32, 16, PixelFormat.Rgba, TensorMemory.DMABUF, "read")
    a = t.d3d11_shared_handle()
    b = t.d3d11_shared_handle()
    try:
        assert a and b
        assert a != b, "each call owns its own duplicate"
    finally:
        _close(a)
        _close(b)
    # The tensor keeps its own handle, so closing both duplicates leaves it
    # shareable.
    c = t.d3d11_shared_handle()
    _close(c)


@windows_only
@pytest.mark.gpu
def test_set_gpu_write_is_a_monotonic_maximum():
    if not is_gpu_buffer_available():
        pytest.skip("no D3D11 device")
    t = Tensor.image(32, 16, PixelFormat.Rgba, TensorMemory.DMABUF, "read")
    t.set_gpu_write(42)
    t.set_gpu_write(7)
    completion = t.gpu_completion()
    assert completion is not None, "a recorded write must surface"
    fence, value = completion
    try:
        assert value == 42, "an older value never displaces a newer one"
    finally:
        _close(fence)


@windows_only
@pytest.mark.gpu
def test_gpu_write_value_is_the_completion_value_without_a_handle():
    if not is_gpu_buffer_available():
        pytest.skip("no D3D11 device")
    t = Tensor.image(64, 32, PixelFormat.Rgba, TensorMemory.DMABUF, "readwrite")
    assert t.gpu_write_value == 0
    t.set_gpu_write(9)
    fence, value = t.gpu_completion()
    try:
        assert value == 9
        assert t.gpu_write_value == value
    finally:
        _close(fence)


@windows_only
@pytest.mark.gpu
def test_module_device_is_the_device_the_library_allocates_on():
    """The module functions are backend-routed.

    On the shipped wheel the tensors are allocated inside
    ``libedgefirst_tensor``, so ``d3d11_device()`` must report *its* device
    rather than a second one in the wheel's own linked copy of the crate.
    """
    if not is_gpu_buffer_available():
        pytest.skip("no D3D11 device")
    from edgefirst.tensor import d3d11_device, d3d11_use_external_device

    try:
        lib = ctypes.CDLL("edgefirst_tensor.dll")
        entry = lib.ef_d3d11_device
    except (OSError, AttributeError):
        pytest.skip("edgefirst_tensor.dll is not reachable by name from this build")
    entry.restype = ctypes.c_void_p
    entry.argtypes = []
    assert d3d11_device() == entry()

    # And a pointer that is not a device is refused rather than adopted.
    with pytest.raises(RuntimeError):
        d3d11_use_external_device(1)


@windows_only
def test_host_memory_tensors_have_no_texture():
    t = Tensor.image(16, 16, PixelFormat.Rgba, TensorMemory.MEM, "readwrite")
    assert t.memory == TensorMemory.MEM
    assert t.d3d11_texture() is None
    assert t.d3d11_layout() is None
    assert t.gpu_write_value == 0


def test_try_map_is_generic():
    t = Tensor.image(16, 16, PixelFormat.Rgb, None, "readwrite")
    with t.try_map("read"):
        pass


@windows_only
@pytest.mark.gpu
def test_try_map_on_a_texture_sees_the_written_bytes():
    """The binding's non-blocking map on the one backing that can block.

    ``try_map`` translates the backend's ``WouldBlock`` into
    ``BlockingIOError``, and the retry must yield: on WARP the threads that
    finish the staging copy are the CPU. Both are exercised here -- the loop
    is what makes the map succeed, and the bytes prove the copy it waited
    for is the one the CPU write produced.

    A deterministic ``BlockingIOError`` is not asserted: forcing one needs a
    write map held from another thread on the same tensor, and the Python
    map guard is not shareable across threads, so this covers the success
    path and the yield rule only.
    """
    if not is_gpu_buffer_available():
        pytest.skip("no D3D11 device")
    t = Tensor.image(64, 64, PixelFormat.Rgba, TensorMemory.DMABUF, "readwrite")
    _write_first_pixel(t, [4, 3, 2, 1])

    deadline = time.monotonic() + 5.0
    blocked = 0
    while True:
        try:
            with t.try_map("read") as m:
                assert list(np.asarray(memoryview(m))[0, 0]) == [4, 3, 2, 1]
            break
        except BlockingIOError:
            blocked += 1
            assert time.monotonic() < deadline, "try_map never made progress"
            # The documented rule: a retry loop must yield, or on WARP it
            # starves the copy it is waiting for.
            time.sleep(0)
    print(f"try_map blocked {blocked} time(s) before succeeding")


def test_cuda_map_mut_on_host_memory_is_none():
    """``cuda_map_mut`` is callable on every platform and every backing, and
    answers ``None`` where the backing carries no CUDA registration.

    Host memory never carries one -- there is nothing for CUDA to import --
    so ``None`` here is the documented answer, not a skip. The mapping case
    is covered by ``test_cuda_map_mut_when_cuda_is_present``, which uses a
    D3D11 texture.
    """
    t = Tensor.image(16, 16, PixelFormat.Rgb, None, "readwrite")
    assert t.cuda_map_mut() is None


def test_cuda_map_mut_when_cuda_is_present():
    if (
        sys.platform != "win32"
        or not is_gpu_buffer_available()
        or not is_cuda_available()
    ):
        pytest.skip("needs Windows, a D3D11 device and CUDA")
    t = Tensor.image(64, 32, PixelFormat.Rgba, TensorMemory.DMABUF, "readwrite")
    cm = t.cuda_map_mut()
    if cm is None and _warp_adapter_requested():
        pytest.skip("WARP adapter: no CUDA device behind the texture")
    assert cm is not None
    with cm as m:
        assert m.device_ptr != 0
        # Tight rows, not the staging pitch row_stride reports.
        assert m.size == 64 * 32 * 4
        assert len(m) == m.size


def test_platform_specific_methods_are_undefined_elsewhere():
    if sys.platform == "win32":
        assert hasattr(Tensor, "d3d11_shared_handle")
        assert hasattr(Tensor, "gpu_write_value")
    else:
        assert not hasattr(Tensor, "d3d11_shared_handle")
        assert not hasattr(Tensor, "gpu_write_value")
