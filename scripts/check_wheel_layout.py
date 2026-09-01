#!/usr/bin/env python3
"""Assert the A3 wheel layout.

Single-tensor-home (task P2): the shared tensor library
(`libedgefirst_tensor.so.<major>` on Linux, `libedgefirst_tensor.<major>.dylib`
on macOS, `edgefirst_tensor.dll` on Windows) lives in exactly the `tensor`
wheel -- every other wheel links to it via RPATH (or, on Windows, an
`os.add_dll_directory()` call in its `__init__.py`) instead of embedding or
vendoring a copy. Older revisions of this check predated that
architecture and required every wheel to be fully self-contained, which is
now exactly backwards for the `tensor` wheel: it is SUPPOSED to carry the
shared core, precisely because nothing else should have to. Also checks for
namespace-shadowing `__init__.py` and that stubs are present. Exits non-zero
on any violation.
"""

from __future__ import annotations

import pathlib
import sys
import zipfile
from collections import Counter

EXPECTED = {"tensor", "codec", "image", "decoder", "tracker"}


def _pkg_from_wheel_name(name: str) -> str:
    return name.split("-")[0].replace("edgefirst_", "")


def _is_tensor_library(name: str) -> bool:
    """Whether a wheel entry is the shared tensor library itself.

    Linux ships ``libedgefirst_tensor.so.<major>``, macOS
    ``libedgefirst_tensor.<major>.dylib``, Windows ``edgefirst_tensor.dll``
    (no ``lib`` prefix and no SONAME/version suffix: a DLL is always named by
    its bare file name).
    """
    base = name.rsplit("/", 1)[-1]
    return base.startswith("libedgefirst_tensor") or base == "edgefirst_tensor.dll"


def _check_tensor_home(wheel_name: str, pkg: str, names: list[str]) -> list[str]:
    # A3: single-tensor-home. The shared tensor library lives in exactly the
    # tensor wheel -- its own home -- and every other wheel links to it
    # (RPATH $ORIGIN/../tensor on Linux/macOS; os.add_dll_directory() on
    # edgefirst/tensor/ at import time on Windows) rather than vendoring a
    # copy. A copy anywhere else is exactly the duplication this
    # architecture removes, reintroduced invisibly: auditwheel/delocate/
    # delvewheel vendor external shared libraries by default, which is
    # precisely how this was found missing here (task P2's report, and
    # again from a differential run afterward).
    tensor_lib = [n for n in names if _is_tensor_library(n)]
    if pkg == "tensor":
        if not tensor_lib:
            return [
                f"{wheel_name} is the tensor wheel but does not ship the shared "
                "tensor library (libedgefirst_tensor.so.<major>, "
                "libedgefirst_tensor.<major>.dylib or edgefirst_tensor.dll)"
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

    # Apache-2.0 §4(a): the licence must travel with the redistributed work.
    # maturin + PEP 639 `license-files` puts it under *.dist-info/licenses/.
    has_license = any(
        "/licenses/" in n or n.endswith(("/LICENSE", "/LICENSE.txt")) for n in names
    )
    if not has_license:
        errors.append(f"{wheel.name} ships no LICENSE (PEP 639 license-files)")
    return errors


def _check_tag_uniformity(wheels: list[pathlib.Path]) -> list[str]:
    # Tag-set uniformity across the five distributions.
    #
    # The release builds both abi3-py311 and abi3-py38. If one package fails to
    # produce a cp311-abi3 wheel for some platform while the others succeed, pip
    # silently installs a cp38-abi3 build of that package alongside cp311-abi3
    # siblings. They are independent CPython modules so nothing crashes -- but
    # the py38 build lacks the zero-copy buffer protocol, so one package behaves
    # differently from the rest and the documented API quietly forks per user.
    tags: dict[str, set[tuple[str, str, str]]] = {}
    for wheel in wheels:
        parts = wheel.stem.split("-")  # name-version-python-abi-platform
        if len(parts) >= 5:
            pkg = parts[0].replace("edgefirst_", "")
            # The python tag (index 2) is what distinguishes cp38-abi3 from
            # cp311-abi3 -- the abi tag is "abi3" for both, so comparing only
            # (abi, platform) would never see the difference that matters.
            tags.setdefault(pkg, set()).add((parts[2], parts[3], parts[4]))
    if len(tags) <= 1:
        return []

    # Majority wins, so the message names the odd one out rather than an
    # arbitrary pick.
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
        # ASCII on purpose: a Windows subprocess pipe is cp1252, where a
        # U+2713 check mark raises UnicodeEncodeError and turns a passing
        # check into exit status 1 (tests/packaging/test_wheel_layout.py).
        print(f"OK: {len(wheels)} wheels: single tensor home, typed, namespace intact")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:]))
