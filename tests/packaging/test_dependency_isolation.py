"""Isolation of the libraries: Rust crates via cargo tree, Python
wheels via the DT_NEEDED / otool of the installed extension.
"""

from __future__ import annotations

import importlib
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

needs_cargo = pytest.mark.skipif(
    shutil.which("cargo") is None,
    reason="cargo is not on PATH; crate-graph checks run on hosted CI",
)

# Internal crate names, matched exactly.
INTERNAL = (
    "edgefirst-tensor",
    "edgefirst-codec",
    "edgefirst-image",
    "edgefirst-decoder",
    "edgefirst-tracker",
    "edgefirst-kernels",
)

CARGO_TREE = [
    "cargo",
    "tree",
    "-e",
    "normal",
    "--prefix",
    "none",
    "--format",
    "{p}",
]


def _crate_names(tree_output: str) -> set[str]:
    """The set of crate names `cargo tree --prefix none --format {p}` printed.

    Each line is ``<name> v<version>``; the first whitespace-delimited token is the
    crate name.
    """
    names = set()
    for line in tree_output.splitlines():
        name = line.split(" ", 1)[0]
        if name:
            names.add(name)
    return names


def links(package: str, extra: list[str] | None = None) -> set[str]:
    cmd = [*CARGO_TREE, "-p", package]
    if extra:
        cmd.extend(extra)
    out = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    return _crate_names(out) & set(INTERNAL)


# Captured `cargo tree --prefix none --format {p}` output with one
# dependency whose name is a different INTERNAL crate's name plus a suffix.
_DECODER_ABI_ONLY_FIXTURE = (
    "edgefirst-python-image v0.29.0\n"
    "edgefirst-image v0.29.0\n"
    "edgefirst-codec v0.29.0\n"
    "edgefirst-tensor v0.29.0\n"
    "edgefirst-decoder-abi v0.29.0\n"
    "edgefirst-tensor v0.29.0\n"
)


def test_links_reports_a_crate_only_when_it_is_actually_in_the_graph(monkeypatch):
    """`links()` matches whole crate names, not prefixes."""

    class _FakeCompletedProcess:
        stdout = _DECODER_ABI_ONLY_FIXTURE

    monkeypatch.setattr(
        subprocess, "run", lambda *args, **kwargs: _FakeCompletedProcess()
    )
    linked = links("edgefirst-python-image")
    assert linked == {"edgefirst-image", "edgefirst-codec", "edgefirst-tensor"}
    assert "edgefirst-decoder" not in linked


SIBLING_C_LIBS = (
    "libedgefirst_codec",
    "libedgefirst_image",
    "libedgefirst_decoder",
    "libedgefirst_tracker",
    "libedgefirst_kernels",
)


def _wheel_extension(leaf: str) -> Path:
    name = f"edgefirst.{leaf}._{leaf}"
    mod = importlib.import_module(name)
    path = Path(mod.__file__)
    if not path.is_file():
        pytest.fail(f"{name}.__file__ is not a file: {path}")
    return path


def _pe_imports(path: Path) -> list[str]:
    """DLL names in a PE image's import directory (the DT_NEEDED analogue)."""
    import struct

    data = path.read_bytes()
    (e_lfanew,) = struct.unpack_from("<I", data, 0x3C)
    if data[e_lfanew : e_lfanew + 4] != b"PE\0\0":
        raise ValueError(f"{path} is not a PE image")
    coff = e_lfanew + 4
    n_sections, opt_size = (
        struct.unpack_from("<H", data, coff + 2)[0],
        struct.unpack_from("<H", data, coff + 16)[0],
    )
    opt = coff + 20
    (magic,) = struct.unpack_from("<H", data, opt)
    dir_off = opt + (112 if magic == 0x20B else 96)  # PE32+ vs PE32
    import_rva, import_size = struct.unpack_from(
        "<II", data, dir_off + 8
    )  # directory 1
    if import_rva == 0 or import_size == 0:
        return []
    sections = []
    sec = opt + opt_size
    for i in range(n_sections):
        off = sec + i * 40
        vsize, vaddr, rsize, raddr = struct.unpack_from("<IIII", data, off + 8)
        sections.append((vaddr, max(vsize, rsize), raddr))

    def rva_to_off(rva: int) -> int:
        for vaddr, size, raddr in sections:
            if vaddr <= rva < vaddr + size:
                return rva - vaddr + raddr
        raise ValueError(f"RVA {rva:#x} outside every section of {path}")

    names = []
    desc = rva_to_off(import_rva)
    while True:
        _oft, _ts, _fwd, name_rva, first_thunk = struct.unpack_from(
            "<IIIII", data, desc
        )
        if name_rva == 0 and first_thunk == 0:
            break
        if name_rva:
            off = rva_to_off(name_rva)
            end = data.index(b"\0", off)
            names.append(data[off:end].decode("ascii", "replace"))
        desc += 20
    return names


def _needed_text(so: Path) -> str:
    if sys.platform == "darwin":
        r = subprocess.run(
            ["otool", "-L", str(so)], capture_output=True, text=True, check=True
        )
        return r.stdout
    if sys.platform == "win32":
        # `.pyd` extension modules are PE DLLs.
        return "\n".join(_pe_imports(so))
    r = subprocess.run(
        ["readelf", "-d", str(so)], capture_output=True, text=True, check=True
    )
    return r.stdout


def _library_named(text: str, lib: str) -> bool:
    # Linux/macOS: `libedgefirst_tensor.so.0` / `.dylib`; Windows: `edgefirst_tensor.dll`.
    return lib in text or lib.removeprefix("lib") in text.lower()


def _links_libedgefirst_tensor(so: Path) -> bool:
    return _library_named(_needed_text(so), "libedgefirst_tensor")


def _embedded_sibling_c_libs(so: Path) -> list[str]:
    text = _needed_text(so)
    return [lib for lib in SIBLING_C_LIBS if _library_named(text, lib)]


def _tensor_backend_is_dynamic() -> bool:
    """Whether the installed edgefirst-tensor declares the dynamic backend."""
    import edgefirst.tensor

    return edgefirst.tensor.__tensor_backend__ == "dynamic"


@pytest.mark.parametrize("leaf", ["tensor", "codec", "image", "decoder"])
def test_python_wheel_dt_needed_libedgefirst_tensor(leaf):
    """The four wheels link libedgefirst_tensor only under the `dynamic` backend."""
    so = _wheel_extension(leaf)
    dynamic = _tensor_backend_is_dynamic()
    if dynamic:
        assert _links_libedgefirst_tensor(so), (
            f"{so} has no DT_NEEDED/otool entry for libedgefirst_tensor"
        )
    else:
        assert not _links_libedgefirst_tensor(so), (
            f"{so} links libedgefirst_tensor under a static tensor backend; "
            "a static build must embed the implementation, not DT_NEEDED it"
        )
    siblings = _embedded_sibling_c_libs(so)
    assert not siblings, (
        f"{so} dynamically links sibling C libraries {siblings}; "
        "python wheels must not DT_NEEDED those implementations"
    )


@pytest.mark.parametrize("leaf", ["tracker", "kernels"])
def test_tensor_free_wheel_does_not_link_tensor(leaf):
    """The tracker and kernels extensions do not link libedgefirst_tensor."""
    so = _wheel_extension(leaf)
    assert not _links_libedgefirst_tensor(so), (
        f"{so} unexpectedly links libedgefirst_tensor"
    )
    siblings = _embedded_sibling_c_libs(so)
    assert not siblings, f"{so} dynamically links sibling C libraries {siblings}"


@needs_cargo
def test_kernels_crate_links_no_other_hal_crate():
    """edgefirst-kernels links no other HAL crate."""
    assert links("edgefirst-kernels") == {"edgefirst-kernels"}


# `edgefirst-tensor`'s optional features.
FEATURE_SETS = ["", "ndarray", "tracing", "ndarray,tracing"]

# The backend is a build cfg: static unless the dynamic config is passed.
TENSOR_DYNAMIC = Path(__file__).resolve().parents[2] / ".cargo" / "tensor-dynamic.toml"
BACKENDS = {"static": [], "dynamic": ["--config", str(TENSOR_DYNAMIC)]}


def _cargo_env():
    """The caller's environment minus RUSTFLAGS."""
    return {
        k: v
        for k, v in os.environ.items()
        if k not in ("RUSTFLAGS", "CARGO_ENCODED_RUSTFLAGS")
    }


@needs_cargo
def test_dynamic_config_selects_the_ffi_backend():
    """Only the dynamic backend pulls in edgefirst-tensor-ffi."""
    trees = {
        backend: subprocess.run(
            ["cargo", "tree", *args, "-p", "edgefirst-tensor", "-e", "normal"],
            capture_output=True,
            text=True,
            check=True,
            env=_cargo_env(),
        ).stdout
        for backend, args in BACKENDS.items()
    }
    assert "edgefirst-tensor-ffi" in trees["dynamic"]
    assert "edgefirst-tensor-ffi" not in trees["static"]


@needs_cargo
@pytest.mark.parametrize("backend", sorted(BACKENDS))
@pytest.mark.parametrize("features", FEATURE_SETS, ids=lambda f: f or "none")
def test_tensor_compiles_across_feature_sets(features, backend):
    """edgefirst-tensor builds under every feature combination on both backends."""
    cmd = [
        "cargo",
        "check",
        *BACKENDS[backend],
        "-p",
        "edgefirst-tensor",
        "--no-default-features",
    ]
    if features:
        cmd += ["--features", features]
    r = subprocess.run(
        cmd, capture_output=True, text=True, check=False, env=_cargo_env()
    )
    assert r.returncode == 0, r.stderr[-2000:]


# --- edgefirst-image must not link a model decoder to draw ------------------

# Crates a default build of edgefirst-image must not link.
NOT_LINKED_BY_DEFAULT = (
    "edgefirst-decoder",
    "serde_json",
    "serde_yaml_ng",
    "unsafe-libyaml",
    "ndarray-stats",
    "argminmax",
    "rand",
    "indexmap",
    "hashbrown",
    "itertools",
)


@needs_cargo
def test_image_does_not_link_a_decoder_by_default():
    """A default build of edgefirst-image does not link a decoder crate."""
    linked = links("edgefirst-image")
    regressions = sorted(d for d in NOT_LINKED_BY_DEFAULT if d in linked)
    assert not regressions, (
        f"edgefirst-image links {regressions}. "
        f"If a new caller needs the decoder, put it behind --features decode."
    )


@needs_cargo
def test_decode_feature_restores_the_decoder():
    """The `decode` feature links edgefirst-decoder into edgefirst-image."""
    r = subprocess.run(
        [
            "cargo",
            "tree",
            "-p",
            "edgefirst-image",
            "-e",
            "normal",
            "--prefix",
            "none",
            "--format",
            "{p}",
            "--features",
            "decode",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert r.returncode == 0, r.stderr[-2000:]
    assert "edgefirst-decoder" in _crate_names(r.stdout), (
        "--features decode did not pull in edgefirst-decoder; the wrappers it "
        "gates cannot compile"
    )
