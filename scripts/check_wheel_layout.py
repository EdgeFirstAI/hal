#!/usr/bin/env python3
"""Assert the A3 wheel layout.

Single-tensor-home (task P2): `libedgefirst_tensor.so` lives in exactly the
`tensor` wheel -- every other wheel links to it via RPATH instead of
embedding or vendoring a copy. Older revisions of this check predated that
architecture and required every wheel to be fully self-contained, which is
now exactly backwards for the `tensor` wheel: it is SUPPOSED to carry the
shared core, precisely because nothing else should have to. Also checks for
namespace-shadowing `__init__.py` and that stubs are present. Exits non-zero
on any violation.
"""
import pathlib
import sys
import zipfile

EXPECTED = {"tensor", "codec", "image", "decoder", "tracker"}


def main(wheel_dir: str = "target/wheels") -> int:
    wheels = sorted(pathlib.Path(wheel_dir).glob("*.whl"))
    if not wheels:
        print(f"ERROR: no wheels in {wheel_dir}")
        return 1

    errors: list[str] = []
    seen: set[str] = set()
    for w in wheels:
        names = zipfile.ZipFile(w).namelist()
        pkg = w.name.split("-")[0].replace("edgefirst_", "")
        seen.add(pkg)

        if "edgefirst/__init__.py" in names:
            errors.append(f"{w.name} ships edgefirst/__init__.py, shadowing the namespace")

        # A3: single-tensor-home. libedgefirst_tensor.so lives in exactly the
        # tensor wheel -- its own home -- and every other wheel links to it
        # (RPATH $ORIGIN/../tensor) rather than vendoring a copy. A copy
        # anywhere else is exactly the duplication this architecture
        # removes, reintroduced invisibly: auditwheel/delocate vendor
        # external shared libraries by default, which is precisely how this
        # was found missing here (task P2's report, and again from a
        # differential run afterward).
        tensor_so = [n for n in names if "libedgefirst_tensor" in n]
        if pkg == "tensor":
            if not tensor_so:
                errors.append(f"{w.name} is the tensor wheel but does not ship libedgefirst_tensor.so")
            elif len(tensor_so) > 1:
                errors.append(
                    f"{w.name} ships libedgefirst_tensor.so {len(tensor_so)} times "
                    f"({tensor_so}); should be exactly one (the .so.<major> SONAME file)"
                )
        elif tensor_so:
            errors.append(f"{w.name} vendors a shared core ({tensor_so}); it must link, not vendor, libedgefirst_tensor.so")

        if not any(n.endswith("py.typed") for n in names):
            errors.append(f"{w.name} has no py.typed marker")
        if not any(n.endswith(".pyi") for n in names):
            errors.append(f"{w.name} ships no type stubs")
        if not any(n.endswith((".so", ".pyd", ".dylib")) for n in names):
            errors.append(f"{w.name} contains no extension module")

        # Apache-2.0 §4(a): the licence must travel with the redistributed work.
        # maturin + PEP 639 `license-files` puts it under *.dist-info/licenses/.
        has_license = any(
            "/licenses/" in n or n.endswith("/LICENSE") or n.endswith("/LICENSE.txt")
            for n in names
        )
        if not has_license:
            errors.append(f"{w.name} ships no LICENSE (PEP 639 license-files)")

    missing = EXPECTED - seen
    if missing:
        errors.append(f"missing wheels for: {sorted(missing)}")

    # Tag-set uniformity across the five distributions.
    #
    # The release builds both abi3-py311 and abi3-py38. If one package fails to
    # produce a cp311-abi3 wheel for some platform while the others succeed, pip
    # silently installs a cp38-abi3 build of that package alongside cp311-abi3
    # siblings. They are independent CPython modules so nothing crashes -- but
    # the py38 build lacks the zero-copy buffer protocol, so one package behaves
    # differently from the rest and the documented API quietly forks per user.
    tags: dict[str, set[tuple[str, str, str]]] = {}
    for w in wheels:
        parts = w.stem.split("-")          # name-version-python-abi-platform
        if len(parts) >= 5:
            pkg = parts[0].replace("edgefirst_", "")
            # The python tag (index 2) is what distinguishes cp38-abi3 from
            # cp311-abi3 -- the abi tag is "abi3" for both, so comparing only
            # (abi, platform) would never see the difference that matters.
            tags.setdefault(pkg, set()).add((parts[2], parts[3], parts[4]))
    if len(tags) > 1:
        from collections import Counter
        # Majority wins, so the message names the odd one out rather than an
        # arbitrary pick.
        counts = Counter(frozenset(v) for v in tags.values())
        reference = set(counts.most_common(1)[0][0])
        for pkg, got in sorted(tags.items()):
            if got != reference:
                errors.append(
                    f"edgefirst-{pkg} builds {sorted(t[0] for t in got)} while the "
                    f"others build {sorted(t[0] for t in reference)}; a mixed abi3 "
                    f"install forks the documented API (py38 lacks the zero-copy "
                    f"buffer protocol)"
                )

    for e in errors:
        print(f"ERROR: {e}")
    if not errors:
        print(f"✓ {len(wheels)} wheels: single tensor home, typed, namespace intact")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:]))
