"""The shipped type stubs must describe the module that ships with them.

Nothing compared `__init__.pyi` to the extension module it annotates, so the
0.29 packaging split left four stubs quietly wrong: `edgefirst.tensor`'s stub
declared a `TensorMap` class that no longer exists and a `normalize_to_numpy` /
`save_jpeg` pair that live in other packages, while `HostPin`, `HostView` and
`CpuAccessGuard` had no stubs at all. A stub is API documentation the type
checker enforces on *users*, so a wrong one is worse than a missing one — it
makes correct code fail to type-check and incorrect code pass.

Two directions, because each catches a different rot:

* **coverage** — every public runtime name has a stub declaration. Catches new
  API that nobody stubbed.
* **ghosts** — every unconditionally-stubbed class and method exists at
  runtime. Catches API that was removed or moved to another package.

Names bound inside an `if` in the stub (`if sys.platform == "linux":`) are
counted for coverage but exempt from the ghost check: they are deliberately
absent on other platforms.
"""

from __future__ import annotations

import ast
import importlib
import pathlib

import pytest

PACKAGES = ("tensor", "codec", "image", "decoder", "tracker")


def stub_path(pkg: str) -> pathlib.Path:
    return pathlib.Path(f"crates/python-{pkg}/python/edgefirst/{pkg}/__init__.pyi")


def _bound_names(nodes) -> set[str]:
    """Top-level names a stub binds: classes, functions, imports, aliases."""
    names: set[str] = set()
    for n in nodes:
        if isinstance(n, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            names.add(n.name)
        elif isinstance(n, ast.ImportFrom):
            names |= {a.asname or a.name for a in n.names}
        elif isinstance(n, ast.AnnAssign) and isinstance(n.target, ast.Name):
            names.add(n.target.id)
        elif isinstance(n, ast.Assign):
            names |= {t.id for t in n.targets if isinstance(t, ast.Name)}
    return names


@pytest.mark.parametrize("pkg", PACKAGES)
def test_every_runtime_name_is_stubbed(pkg: str):
    tree = ast.parse(stub_path(pkg).read_text())
    # Conditional bodies count here: a name gated on sys.platform is still
    # stubbed, and this test may run on the platform that has it.
    conditional = [
        c for n in tree.body if isinstance(n, ast.If) for c in (*n.body, *n.orelse)
    ]
    stubbed = _bound_names(tree.body) | _bound_names(conditional)

    mod = importlib.import_module(f"edgefirst.{pkg}")
    runtime = {n for n in dir(mod) if not n.startswith("_")}

    missing = sorted(runtime - stubbed)
    assert not missing, (
        f"edgefirst.{pkg} exports {missing} but {stub_path(pkg)} does not "
        f"declare them; a user's type checker will reject correct code"
    )


@pytest.mark.parametrize("pkg", PACKAGES)
def test_no_stubbed_name_is_a_ghost(pkg: str):
    tree = ast.parse(stub_path(pkg).read_text())
    mod = importlib.import_module(f"edgefirst.{pkg}")

    ghosts: list[str] = []
    for node in tree.body:
        if not isinstance(node, ast.ClassDef):
            continue
        runtime_cls = getattr(mod, node.name, None)
        if runtime_cls is None:
            # An imported alias re-exported under another name is not a class
            # this module defines; only flag names the stub declares with
            # `class`, which this branch is.
            ghosts.append(f"class {node.name}")
            continue
        have = {m for m in dir(runtime_cls)}
        for member in node.body:
            if not isinstance(member, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            if member.name.startswith("__"):
                continue  # dunders come from the metaclass/protocol, not dir()
            if member.name not in have:
                ghosts.append(f"{node.name}.{member.name}")

    assert not ghosts, (
        f"{stub_path(pkg)} declares {ghosts}, absent from edgefirst.{pkg} at "
        f"runtime; a user calling them gets a clean type check and an "
        f"AttributeError"
    )
