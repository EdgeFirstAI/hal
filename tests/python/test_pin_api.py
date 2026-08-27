"""Persistent host pinning from Python (issue #134)."""

import edgefirst.tensor as t
import pytest


def test_pin_survives_mutation_and_brackets_are_portable():
    """Pin once, mutate the tensor, keep the address.

    A guard that borrowed the tensor could not coexist with the ``&mut`` that
    ``ImageProcessor.convert`` takes on its destination every frame -- that is
    the conflict #134 reports. The pin borrows nothing.
    """
    tensor = t.Tensor.image(64, 64, t.PixelFormat.Rgb, t.TensorMemory.MEM, "readwrite")
    pin = tensor.pin_host("readwrite")

    addr, length = pin.ptr, pin.len
    assert addr != 0
    assert length == 64 * 64 * 3, length
    assert pin.alignment >= 1

    # The point of #134: a mutating call while pinned, address unchanged.
    tensor.reshape([64 * 64 * 3])
    assert pin.ptr == addr, "re-layout moved a pinned address"
    assert pin.len == length, "pin length must be a snapshot, not a tracker"

    # Portable bracket: a no-op on Mem, must not raise.
    with tensor.cpu_access("write"):
        pass
    with tensor.cpu_access("read"):
        pass

    pin.release()
    with pytest.raises(RuntimeError):
        _ = pin.ptr
