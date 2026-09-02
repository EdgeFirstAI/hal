"""Isolation of the five libraries: Rust crates via cargo tree, Python
wheels via the DT_NEEDED / otool of the installed extension.

A coupling can be re-added by one unconditional reference and shrink to
nothing under dead-code elimination, so it is invisible in a size baseline
while still being wrong. `cargo tree --prefix none --format {p}` is the
honest check for Rust. Python wheels do not cargo-depend on the sibling
Rust crates (they are python-common + a C library from build.rs), so
isolation for those is measured on the built ``.so`` / ``.dylib``.
"""

# `set[str]` in a runtime annotation is evaluated at def time; the project
# floor is Python 3.8, where the builtin is not subscriptable.
from __future__ import annotations

import importlib
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

# Hardware runners (imx8mp) install wheels and run pytest without a
# Rust toolchain. Crate-graph isolation is the hosted-CI job.
needs_cargo = pytest.mark.skipif(
    shutil.which("cargo") is None,
    reason="cargo is not on PATH; crate-graph checks run on hosted CI",
)

# Exact crate names only. This list will keep growing as more leaves are
# split out (it already did once: `edgefirst-decoder-abi`, `edgefirst-tensor-
# abi`, `edgefirst-tensor-ffi`), and every `-abi`/`-ffi` split creates a real
# dependency whose name has an existing INTERNAL crate's full name as a
# prefix. `links()` below compares crate-name tokens for equality precisely
# because of that: a substring/`in` test against the raw `cargo tree` text
# would report the shorter crate "linked" whenever only its `-abi`/`-ffi`
# leaf is (`edgefirst-decoder-abi` in the tree made `edgefirst-decoder` look
# linked when it wasn't -- see the dependency-isolation hygiene track).
# Restoring `c in out` here as a "simplification" reintroduces that bug.
INTERNAL = (
    "edgefirst-tensor",
    "edgefirst-codec",
    "edgefirst-image",
    "edgefirst-decoder",
    "edgefirst-tracker",
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

    Each line is ``<name> v<version>`` (stable; no box-drawing prefix). The
    first whitespace-delimited token is the crate name. A substring scan of
    the whole blob would report ``edgefirst-decoder`` whenever only
    ``edgefirst-decoder-abi`` is present.
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
# dependency whose name is a different INTERNAL crate's name plus a suffix
# -- exactly the `edgefirst-decoder` / `edgefirst-decoder-abi` shape that
# broke the substring version of `links()`.
_DECODER_ABI_ONLY_FIXTURE = (
    "edgefirst-python-image v0.29.0\n"
    "edgefirst-image v0.29.0\n"
    "edgefirst-codec v0.29.0\n"
    "edgefirst-tensor v0.29.0\n"
    "edgefirst-decoder-abi v0.29.0\n"
    "edgefirst-tensor v0.29.0\n"
)


def test_links_reports_a_crate_only_when_it_is_actually_in_the_graph(monkeypatch):
    """`links()` must not report `edgefirst-decoder` on a tree that contains
    only `edgefirst-decoder-abi` -- a real, intentional dependency whose name
    happens to start with another INTERNAL crate's full name. A substring
    test against the raw `cargo tree` blob gets this wrong; per-line crate-
    name parsing does not.
    """

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
)


def _wheel_extension(leaf: str) -> Path:
    name = f"edgefirst.{leaf}._{leaf}"
    mod = importlib.import_module(name)
    path = Path(mod.__file__)
    if not path.is_file():
        pytest.fail(f"{name}.__file__ is not a file: {path}")
    return path


def _pe_imports(path: Path) -> list[str]:
    """DLL names in a PE image's import directory (the DT_NEEDED analogue).

    Pure Python so the check does not need dumpbin / a VS developer shell.
    """
    import struct

    data = path.read_bytes()
    (e_lfanew,) = struct.unpack_from("<I", data, 0x3C)
    if data[e_lfanew : e_lfanew + 4] != b"PE\0\0":
        raise ValueError(f"{path} is not a PE image")
    coff = e_lfanew + 4
    n_sections, opt_size = struct.unpack_from("<H", data, coff + 2)[0], struct.unpack_from("<H", data, coff + 16)[0]
    opt = coff + 20
    (magic,) = struct.unpack_from("<H", data, opt)
    dir_off = opt + (112 if magic == 0x20B else 96)  # PE32+ vs PE32
    import_rva, import_size = struct.unpack_from("<II", data, dir_off + 8)  # directory 1
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
        _oft, _ts, _fwd, name_rva, first_thunk = struct.unpack_from("<IIIII", data, desc)
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
        # `.pyd` extension modules are PE DLLs; their import directory names
        # `edgefirst_tensor.dll` (no `lib` prefix on Windows).
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


@pytest.mark.parametrize("leaf", ["tensor", "codec", "image", "decoder"])
def test_python_wheel_dt_needed_libedgefirst_tensor(leaf):
    """Four wheels dynamically link libedgefirst_tensor; they must not
    embed a sibling C library (G12).
    """
    so = _wheel_extension(leaf)
    assert _links_libedgefirst_tensor(so), (
        f"{so} has no DT_NEEDED/otool entry for libedgefirst_tensor"
    )
    siblings = _embedded_sibling_c_libs(so)
    assert not siblings, (
        f"{so} dynamically links sibling C libraries {siblings}; "
        "python wheels must not DT_NEEDED those implementations"
    )


def test_tracker_wheel_does_not_link_tensor():
    """Tracker detections cross as a plain array; _tracker.so must not
    DT_NEEDED libedgefirst_tensor.
    """
    so = _wheel_extension("tracker")
    assert not _links_libedgefirst_tensor(so), (
        f"{so} unexpectedly links libedgefirst_tensor"
    )
    siblings = _embedded_sibling_c_libs(so)
    assert not siblings, f"{so} dynamically links sibling C libraries {siblings}"


# `edgefirst-tensor`'s feature set, from its own Cargo.toml: `static` and
# `dynamic` are the backend choice -- mutually exclusive and one of them is
# REQUIRED (see the `compile_error!` guards in `crates/tensor/src/lib.rs`,
# gated on `not(any(static, dynamic))` and on `all(static, dynamic)`). Every
# entry below therefore names exactly one backend. `--all-features` is never
# legal against this crate for the same reason: it would enable both
# backends at once and hit the second guard. `dynamic-test-link` is
# deliberately left out of every entry too -- per its own doc comment in
# Cargo.toml it is "opt-in only, never enabled by a production consumer",
# and it links `libedgefirst_tensor.so` at build-script time, which requires
# `edgefirst-tensor-capi` to already be built; that precondition is exercised
# by `make test-capi-modular`, not by a general compile-across-features check.
#
# `["--features", "static,ndarray,tracing"]` and
# `["--features", "dynamic,ndarray,tracing"]` are what "every feature except
# the other backend" becomes once `--all-features` is off the table -- one
# maximal combination per backend, since a single `--all-features` run can no
# longer stand in for exhausting the legal combinations.
FEATURE_SETS = [
    ["--features", "static"],
    ["--features", "static,ndarray"],
    ["--features", "static,tracing"],
    ["--features", "static,ndarray,tracing"],
    ["--features", "dynamic,ndarray,tracing"],
]


@needs_cargo
@pytest.mark.parametrize("extra", FEATURE_SETS, ids=lambda e: e[1])
def test_tensor_compiles_across_feature_sets(extra):
    """edgefirst-tensor must build under any *legal* combination of its
    features.

    A crate that only compiles with its defaults cannot be trimmed by a
    consumer, and the breakage is invisible until someone tries. Checked here
    rather than left to CI's default-features-only build.
    """
    cmd = ["cargo", "check", "-p", "edgefirst-tensor", "--no-default-features", *extra]
    # check=False: a non-zero exit IS the finding here, and the assertion below
    # reports cargo's own diagnostics. CalledProcessError would hide them.
    r = subprocess.run(cmd, capture_output=True, text=True, check=False)
    assert r.returncode == 0, r.stderr[-2000:]


# --- edgefirst-image must not link a model decoder to draw ------------------

# Everything the `image -> decoder` edge used to drag in. A YAML parser, a
# JSON parser, a stats crate and an RNG, because drawing a rectangle needed
# `DetectBox`. The vocabulary now lives in edgefirst-tensor and the two
# decoder-taking convenience wrappers sit behind the non-default `decode`
# feature, so a default build of edgefirst-image links none of this.
SHED_BY_DECOUPLING = (
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
    """Drawing masks must not require a model-postprocessing crate.

    `draw_decoded_masks` and `draw_proto_masks` take plain data, so the
    default build has no reason to link `edgefirst-decoder`. Asserted on the
    dependency *graph* rather than on wheel size: static linking plus DCE can
    hide a re-added edge that costs no bytes but is still wrong.
    """
    linked = links("edgefirst-image")
    regressions = sorted(d for d in SHED_BY_DECOUPLING if d in linked)
    assert not regressions, (
        f"edgefirst-image links {regressions} again; the decoupling regressed. "
        f"If a new caller needs the decoder, put it behind --features decode."
    )


@needs_cargo
def test_decode_feature_restores_the_decoder():
    """The gate must be a gate, not a deletion.

    Without this, dropping the wrappers entirely would also pass the test
    above — and the convenience API is meant to remain available on request.

    Parsed with `_crate_names`, not a raw substring check on `r.stdout`:
    `edgefirst-image` links `edgefirst-decoder-abi` unconditionally (see
    `test_image_does_not_link_a_decoder_by_default`'s docstring), so
    `"edgefirst-decoder" in r.stdout` would already be true with the feature
    off and prove nothing about what `--features decode` actually changed.
    """
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
