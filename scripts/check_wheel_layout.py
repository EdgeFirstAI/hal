#!/usr/bin/env python3
"""Assert the wheel layout.

The shared tensor library (`libedgefirst_tensor.so.<major>` on Linux,
`libedgefirst_tensor.<major>.dylib` on macOS, `edgefirst_tensor.dll` on
Windows) must be in the `tensor` wheel and in no other; the other wheels
link to it. Also checks for namespace-shadowing `__init__.py` and that stubs
are present. Exits non-zero on any violation.
"""

from __future__ import annotations

import pathlib
import sys
import zipfile
from collections import Counter

EXPECTED = {"tensor", "codec", "image", "decoder", "tracker", "kernels"}


def _pkg_from_wheel_name(name: str) -> str:
    return name.split("-")[0].replace("edgefirst_", "")


def _is_tensor_library(name: str) -> bool:
    """Whether a wheel entry is the shared tensor library itself.

    Linux ships ``libedgefirst_tensor.so.<major>``, macOS
    ``libedgefirst_tensor.<major>.dylib``, Windows ``edgefirst_tensor.dll``.
    """
    base = name.rsplit("/", 1)[-1]
    return base.startswith("libedgefirst_tensor") or base == "edgefirst_tensor.dll"


def _check_tensor_home(wheel_name: str, pkg: str, names: list[str]) -> list[str]:
    # The shared tensor library must be in the tensor wheel and in no other.
    tensor_lib = [n for n in names if _is_tensor_library(n)]
    if pkg == "tensor":
        if not tensor_lib:
            return [
                (
                    f"{wheel_name} is the tensor wheel but does not ship the shared "
                    "tensor library (libedgefirst_tensor.so.<major>, "
                    "libedgefirst_tensor.<major>.dylib or edgefirst_tensor.dll)"
                )
            ]
        if len(tensor_lib) > 1:
            return [
                (
                    f"{wheel_name} ships the shared tensor library {len(tensor_lib)} "
                    f"times ({tensor_lib}); should be exactly one (the SONAME/"
                    "install-name file on Linux/macOS, edgefirst_tensor.dll on "
                    "Windows)"
                )
            ]
        return []
    if tensor_lib:
        return [
            (
                f"{wheel_name} vendors a shared core ({tensor_lib}); "
                "it must link, not vendor, the shared tensor library"
            )
        ]
    return []


def _check_wheel_contents(wheel: pathlib.Path) -> list[str]:
    names = zipfile.ZipFile(wheel).namelist()
    pkg = _pkg_from_wheel_name(wheel.name)
    errors: list[str] = []

    if "edgefirst/__init__.py" in names:
        errors.append(
            f"{wheel.name} ships edgefirst/__init__.py, shadowing the namespace"
        )

    errors.extend(_check_tensor_home(wheel.name, pkg, names))

    if not any(n.endswith("py.typed") for n in names):
        errors.append(f"{wheel.name} has no py.typed marker")
    if not any(n.endswith(".pyi") for n in names):
        errors.append(f"{wheel.name} ships no type stubs")
    if not any(n.endswith((".so", ".pyd", ".dylib")) for n in names):
        errors.append(f"{wheel.name} contains no extension module")

    # The licence file is under *.dist-info/licenses/.
    has_license = any(
        "/licenses/" in n or n.endswith(("/LICENSE", "/LICENSE.txt")) for n in names
    )
    if not has_license:
        errors.append(f"{wheel.name} ships no LICENSE (PEP 639 license-files)")
    return errors


def _check_tag_uniformity(wheels: list[pathlib.Path]) -> list[str]:
    # Tag-set uniformity across the distributions.
    tags: dict[str, set[tuple[str, str, str]]] = {}
    for wheel in wheels:
        parts = wheel.stem.split("-")  # name-version-python-abi-platform
        if len(parts) >= 5:
            pkg = parts[0].replace("edgefirst_", "")
            # Compares the python, abi and platform tags.
            tags.setdefault(pkg, set()).add((parts[2], parts[3], parts[4]))
    if len(tags) <= 1:
        return []

    # The majority tag set is the expected one.
    counts = Counter(frozenset(v) for v in tags.values())
    reference = set(counts.most_common(1)[0][0])
    errors: list[str] = []
    for pkg, got in sorted(tags.items()):
        if got != reference:
            errors.append(
                f"edgefirst-{pkg} builds {sorted(t[0] for t in got)} while the "
                f"others build {sorted(t[0] for t in reference)}; a mixed abi3 "
                f"install forks the documented API (py38 lacks the zero-copy "
                f"buffer protocol)"
            )
    return errors


def main(wheel_dir: str = "target/wheels") -> int:
    wheels = sorted(pathlib.Path(wheel_dir).glob("*.whl"))
    if not wheels:
        print(f"ERROR: no wheels in {wheel_dir}")
        return 1

    errors: list[str] = []
    seen: set[str] = set()
    for wheel in wheels:
        seen.add(_pkg_from_wheel_name(wheel.name))
        errors.extend(_check_wheel_contents(wheel))

    missing = EXPECTED - seen
    if missing:
        errors.append(f"missing wheels for: {sorted(missing)}")

    errors.extend(_check_tag_uniformity(wheels))

    for error in errors:
        print(f"ERROR: {error}")
    if not errors:
        print(f"OK: {len(wheels)} wheels: single tensor home, typed, namespace intact")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:]))
