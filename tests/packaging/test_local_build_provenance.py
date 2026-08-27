"""Every installed `edgefirst.*` package must be the local build.

The five packages pin each other with `~=` compatible-release specifiers. Those
pins describe what a *published* wheel needs from PyPI; they are wrong for a
from-source install of the whole workspace, where the correct sibling is by
definition the one built alongside. If pip is allowed to resolve them it either
fails outright (any unreleased version is unpublished) or -- far worse --
quietly installs a *released* sibling over the locally built one, and the whole
suite then measures shipped code while reporting green. The Makefile and the CI
wheel installs pass `--no-deps` to prevent that; these tests notice if someone
removes it.

Provenance comes from PEP 610 `direct_url.json`, which pip writes only for
installs from a local path or URL (a wheel installed from `target/wheels/*.whl`
gets one too, so this works under the CI layout as well as a source install). A
distribution resolved from an index has no such file, so its absence is the
signal.

Scope, deliberately: only the four workspace packages are checked, by exact
name. A prefix match would sweep in unrelated published distributions such as
`edgefirst-schemas` and report a false red that re-running `make test-python`
could never fix -- and a gate whose whole value is being believed cannot afford
that. Parametrising over this constant tuple, rather than over whatever happens
to be installed, is also what stops the tests passing vacuously: there are
always exactly four cases, and a missing package fails its own case.
"""

from __future__ import annotations

import importlib.metadata as md
import json
import pathlib
import re

import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]

#: The workspace's own distributions -- exact names, never a prefix match.
WORKSPACE_PACKAGES = (
    "edgefirst-tensor",
    "edgefirst-codec",
    "edgefirst-image",
    "edgefirst-decoder",
)


def distribution(name: str) -> md.Distribution:
    return md.distribution(name)


def origin_url(dist: md.Distribution) -> str | None:
    """The PEP 610 origin, or None when the dist came from an index."""
    try:
        raw = dist.read_text("direct_url.json")
    except (OSError, KeyError):
        # No such metadata file: resolved from an index, not a local path.
        return None
    if not raw:
        return None
    return json.loads(raw).get("url")


def workspace_version() -> str:
    cargo = (REPO_ROOT / "Cargo.toml").read_text()
    block = cargo.split("[workspace.package]", 1)[1]
    return re.search(r'^version = "([^"]+)"', block, re.MULTILINE).group(1)


@pytest.mark.parametrize("name", WORKSPACE_PACKAGES)
def test_edgefirst_package_came_from_a_local_build(name):
    """This is the assertion that would have caught the original defect."""
    dist = distribution(name)
    url = origin_url(dist)
    assert url is not None, (
        f"{name} {dist.version} has no PEP 610 direct_url.json, which means pip "
        f"resolved it from an index (PyPI) rather than this working tree. The "
        f"suite would be exercising released code, not this branch. Install "
        f"with `--no-deps` from the local path."
    )
    assert url.startswith("file://"), (
        f"{name} {dist.version} was installed from {url!r}, not a local path. "
        f"Only locally built modules may be used by the test suite."
    )


@pytest.mark.parametrize("name", WORKSPACE_PACKAGES)
def test_edgefirst_package_matches_the_workspace_version(name):
    """Catches an install left behind by an earlier version of the tree.

    Reach is limited and deliberately so: it cannot see a rebuilt extension that
    was never reinstalled (the recorded version does not change), and while the
    workspace sits at the same version as `main` a wrongly-resolved sibling at
    that same version would pass here. The `direct_url` assertion above carries
    the weight; this one only rules out a stale *version*.
    """
    dist = distribution(name)
    expected = workspace_version()
    assert dist.version == expected, (
        f"{name} is installed at {dist.version} but the workspace is at "
        f"{expected}; the installed extension predates the current tree. "
        f"Re-run `make test-python`."
    )
