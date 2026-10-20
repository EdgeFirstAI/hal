"""The four tensor-linked extensions report which edgefirst-tensor backend they were built against."""

import importlib
import os
import pathlib

TENSOR_LINKED = [
    "edgefirst.tensor",
    "edgefirst.codec",
    "edgefirst.image",
    "edgefirst.decoder",
]


def _backends():
    return {
        name: importlib.import_module(name).__tensor_backend__ for name in TENSOR_LINKED
    }


def test_backend_is_a_known_value():
    for name, backend in _backends().items():
        assert backend in ("static", "dynamic"), (
            f"{name}.__tensor_backend__ = {backend!r}"
        )


def test_backend_is_the_shipped_one():
    """Released wheels are dynamic; only the G13 comparison venv is built static on purpose."""
    expected = (
        "static" if os.environ.get("EDGEFIRST_TENSOR_STATIC_COMPARISON") else "dynamic"
    )
    for name, backend in _backends().items():
        assert backend == expected, (
            f"{name}.__tensor_backend__ = {backend!r}, expected {expected!r}"
        )


def test_extensions_agree_on_one_backend():
    """A mixed install would hold more than one tensor implementation in the process."""
    backends = _backends()
    assert len(set(backends.values())) == 1, backends


def test_dynamic_tensor_wheel_bundles_the_shared_library():
    """The dynamic backend is only usable if edgefirst.tensor carries libedgefirst_tensor."""
    import edgefirst.tensor

    package_dir = pathlib.Path(edgefirst.tensor.__file__).parent
    bundled = [
        p.name
        for p in package_dir.iterdir()
        if p.name.startswith("libedgefirst_tensor.") or p.name == "edgefirst_tensor.dll"
    ]
    if edgefirst.tensor.__tensor_backend__ == "dynamic":
        assert bundled, f"dynamic build but no libedgefirst_tensor in {package_dir}"
    else:
        assert not bundled, f"static build but {bundled} bundled in {package_dir}"
