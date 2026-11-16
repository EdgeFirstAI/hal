"""Every installed `edgefirst.*` workspace package must be the local build.

Provenance comes from PEP 610 `direct_url.json`, which pip writes only for installs from
a local path or URL; a distribution resolved from an index has none. Only the workspace
packages are checked, by exact name.
"""

from __future__ import annotations

import importlib.metadata as md
import json
import pathlib
import re

import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]

#: The workspace's own distributions.
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
    """The distribution has a PEP 610 `direct_url.json` naming a local path."""
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
    """The installed version equals the workspace version."""
    dist = distribution(name)
    expected = workspace_version()
    assert dist.version == expected, (
        f"{name} is installed at {dist.version} but the workspace is at "
        f"{expected}; the installed extension predates the current tree. "
        f"Re-run `make test-python`."
    )
