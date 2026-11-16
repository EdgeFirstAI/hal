"""Persistent host pinning from Python."""

import edgefirst.tensor as t
import pytest


def test_pin_survives_mutation_and_brackets_are_portable():
    """Pin once, mutate the tensor, keep the address."""
    tensor = t.Tensor.image(64, 64, t.PixelFormat.Rgb, t.TensorMemory.MEM, "readwrite")
    pin = tensor.pin_host("readwrite")

    addr, length = pin.ptr, pin.len
    assert addr != 0
    assert length == 64 * 64 * 3, length
    assert pin.alignment >= 1

    # A mutating call while pinned leaves the address unchanged.
    tensor.reshape([64 * 64 * 3])
    assert pin.ptr == addr, "re-layout moved a pinned address"
    assert pin.len == length, "pin length must be a snapshot, not a tracker"

    # `cpu_access` on Mem is a no-op and does not raise.
    with tensor.cpu_access("write"):
        pass
    with tensor.cpu_access("read"):
        pass

    pin.release()
    with pytest.raises(RuntimeError):
        _ = pin.ptr
