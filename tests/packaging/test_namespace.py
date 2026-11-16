"""The packaging contract of the edgefirst namespace, checked on installed modules."""

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
    "edgefirst.kernels",
]


@pytest.mark.parametrize("name", MODULES)
def test_module_imports(name):
    importlib.import_module(name)


def test_no_namespace_init_anywhere():
    """No path of the edgefirst namespace holds an __init__.py."""
    import edgefirst

    for path in edgefirst.__path__:
        assert not (pathlib.Path(path) / "__init__.py").exists(), (
            f"{path}/__init__.py shadows the PEP 420 namespace"
        )


def test_importing_all_modules_does_not_panic():
    """Importing every module in one process does not panic."""
    code = "import edgefirst.tensor, edgefirst.codec, edgefirst.image, edgefirst.decoder, edgefirst.tracker, edgefirst.kernels"
    r = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=False
    )
    assert r.returncode == 0, f"multi-import failed:\n{r.stderr}"
    assert "PanicException" not in r.stderr


@pytest.mark.parametrize("name", MODULES)
def test_pyclass_modules_are_named(name):
    """Every class of a module reports an edgefirst.* __module__ or 'builtins'."""
    mod = importlib.import_module(name)
    for attr in vars(mod).values():
        if isinstance(attr, type) and attr.__module__ != "builtins":
            assert attr.__module__.startswith("edgefirst."), (
                f"{attr.__name__} reports {attr.__module__}"
            )


def test_packages_do_not_share_python_types():
    """The same Rust type surfaces as a different Python class in each package."""
    from edgefirst import image, tensor

    assert tensor.Tensor is not image.Tensor


def test_cross_package_handoff_uses_the_capsule():
    """A tensor hands itself across packages as an `edgefirst_tensor_v2` capsule."""
    from edgefirst import tensor

    t = tensor.Tensor([16, 16, 3], "uint8")
    assert hasattr(t, "__edgefirst_tensor__")
    cap = t.__edgefirst_tensor__()
    assert "edgefirst_tensor_v2" in repr(cap)


@pytest.mark.parametrize(
    "name", ["tensor", "codec", "image", "decoder", "tracker", "kernels"]
)
def test_each_package_ships_stubs_and_marker(name):
    """Each package ships a py.typed marker and stubs."""
    mod = importlib.import_module(f"edgefirst.{name}")
    pkg = pathlib.Path(mod.__file__).parent
    assert (pkg / "py.typed").exists(), f"edgefirst.{name} has no py.typed"
    assert (pkg / "__init__.pyi").exists(), f"edgefirst.{name} has no stubs"


def test_every_expected_symbol_is_available():
    """Every top-level name edgefirst_hal exported exists in the namespace."""
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
        assert name in available, f"{name} is missing"
