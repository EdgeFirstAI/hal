"""Each wheel must link only what it needs.

These assert the *dependency graph*, not sizes: a coupling can be re-added by
one unconditional reference and shrink to nothing under dead-code elimination,
so it is invisible in a size baseline while still being wrong. `cargo tree` is
the honest check.
"""

# `set[str]` in a runtime annotation is evaluated at def time; the project
# floor is Python 3.8, where the builtin is not subscriptable.
from __future__ import annotations

import subprocess

import pytest

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


def _crate_names(tree_output: str) -> set[str]:
    """The set of crate names `cargo tree` printed, one per line.

    Real per-line parsing, not a substring scan of the whole blob: each line
    is `<tree-drawing prefix><name> v<version> (<path or "*">)`, e.g.
    `│   └── edgefirst-decoder-abi v0.29.0 (/repo/crates/decoder-abi)`. The
    box-drawing prefix (`├── `, `│   `, `└── `, plain indentation) is made up
    entirely of `│├└─` and spaces, so stripping those from the left leaves
    the name as the first whitespace-delimited token; the version, path, and
    any `(*)` dedup marker that follow are irrelevant here and ignored.
    """
    names = set()
    for line in tree_output.splitlines():
        name = line.lstrip(" │├└─").split(" ", 1)[0]
        if name:
            names.add(name)
    return names


def links(package: str) -> set[str]:
    out = subprocess.run(
        ["cargo", "tree", "-p", package, "-e", "normal"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    return _crate_names(out) & set(INTERNAL)


# Captured, trimmed `cargo tree` output shaped like a real python wheel's
# tree (tree-drawing prefixes, nesting, versions, absolute paths, a `(*)`
# dedup marker) with one dependency whose name is a different INTERNAL
# crate's name plus a suffix -- exactly the `edgefirst-decoder`/
# `edgefirst-decoder-abi` shape that broke the substring version of `links()`.
_DECODER_ABI_ONLY_FIXTURE = (
    "edgefirst-python-image v0.29.0 (/repo/crates/python-image)\n"
    "├── edgefirst-image v0.29.0 (/repo/crates/image)\n"
    "│   ├── edgefirst-codec v0.29.0 (/repo/crates/codec)\n"
    "│   │   └── edgefirst-tensor v0.29.0 (/repo/crates/tensor)\n"
    "│   └── edgefirst-decoder-abi v0.29.0 (/repo/crates/decoder-abi)\n"
    "└── edgefirst-tensor v0.29.0 (/repo/crates/tensor) (*)\n"
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


def test_tensor_wheel_links_only_the_tensor_crate():
    """The smallest wheel must not carry a decoder.

    It did until 0.29: python-common's tensor.rs referenced edgefirst_codec
    unconditionally, so `tensor = ["dep:edgefirst-codec"]` was load-bearing and
    every edgefirst-tensor install shipped a JPEG/PNG decoder it never exposed.
    """
    assert links("edgefirst-python-tensor") == {"edgefirst-tensor"}


def test_codec_wheel_does_not_pull_the_image_stack():
    assert links("edgefirst-python-codec") == {"edgefirst-tensor", "edgefirst-codec"}


def test_decoder_wheel_does_not_pull_the_image_stack():
    assert "edgefirst-image" not in links("edgefirst-python-decoder")


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
