"""The A3 packaging contract, verified against installed modules."""

import importlib
import pathlib
import subprocess
import sys

import pytest

MODULES = [
    "edgefirst.tensor",
    "edgefirst.codec",
    "edgefirst.image",
    "edgefirst.decoder",
    "edgefirst.tracker",
]


@pytest.mark.parametrize("name", MODULES)
def test_module_imports(name):
    importlib.import_module(name)


def test_no_namespace_init_anywhere():
    """One edgefirst/__init__.py would shadow the namespace for every sibling."""
    import edgefirst

    for path in edgefirst.__path__:
        assert not (pathlib.Path(path) / "__init__.py").exists(), (
            f"{path}/__init__.py shadows the PEP 420 namespace"
        )


def test_importing_all_modules_does_not_panic():
    """env_logger::Builder::init() panics on a second logger. With four
    pymodule_init functions in one process, anything but try_init crashes
    every user on their second import."""
    code = "import edgefirst.tensor, edgefirst.codec, edgefirst.image, edgefirst.decoder, edgefirst.tracker"
    r = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=False
    )
    assert r.returncode == 0, f"multi-import failed:\n{r.stderr}"
    assert "PanicException" not in r.stderr


@pytest.mark.parametrize("name", MODULES)
def test_pyclass_modules_are_named(name):
    """PyO3 defaults __module__ to 'builtins', which breaks pickling and repr."""
    mod = importlib.import_module(name)
    for attr in vars(mod).values():
        if isinstance(attr, type) and attr.__module__ != "builtins":
            assert attr.__module__.startswith("edgefirst."), (
                f"{attr.__name__} reports {attr.__module__}"
            )


def test_packages_do_not_share_python_types():
    """The premise of A3: each extension statically links its own bindings, so
    the same Rust type surfaces as DIFFERENT Python classes per module. Any
    cross-package handoff that relied on isinstance would be broken."""
    from edgefirst import image, tensor

    assert tensor.Tensor is not image.Tensor


def test_cross_package_handoff_uses_the_capsule():
    """...which is why the contract is a capsule, read by duck typing."""
    from edgefirst import tensor

    t = tensor.Tensor([16, 16, 3], "uint8")
    assert hasattr(t, "__edgefirst_tensor__")
    cap = t.__edgefirst_tensor__()
    assert "edgefirst_tensor_v1" in repr(cap)


@pytest.mark.parametrize("name", ["tensor", "codec", "image", "decoder", "tracker"])
def test_each_package_ships_stubs_and_marker(name):
    """The pre-0.29 wheel shipped no stubs at all (the .pyi was sdist-only);
    the split must not repeat that."""
    mod = importlib.import_module(f"edgefirst.{name}")
    pkg = pathlib.Path(mod.__file__).parent
    assert (pkg / "py.typed").exists(), f"edgefirst.{name} has no py.typed"
    assert (pkg / "__init__.pyi").exists(), f"edgefirst.{name} has no stubs"


def test_no_symbol_was_lost_in_the_split():
    """Every top-level name the old edgefirst_hal exported must still exist
    somewhere in the namespace."""
    from edgefirst import codec, decoder, image, tensor, tracker

    available = set()
    for m in (tensor, codec, image, decoder, tracker):
        available |= {n for n in vars(m) if not n.startswith("_")}
    for name in (
        "Tensor",
        "TensorMemory",
        "ImageProcessor",
        "Decoder",
        "ByteTrack",
        "Colorimetry",
        "Tracing",
        "build_info",
        "version",
        "is_dma_available",
        "is_v4l2_available",
    ):
        assert name in available, f"{name} lost in the split"
