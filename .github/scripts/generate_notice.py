#!/usr/bin/env python3
"""
Generate NOTICE file from SBOM (CycloneDX format)
Extracts packages requiring attribution based on their licenses
"""

import base64
import json
import sys
from typing import Set, List, Dict, Tuple

# Licences whose terms are satisfied by naming the component in NOTICE.
ATTRIBUTION_REQUIRED_LICENSES: Set[str] = {
    "Apache-2.0",
    "BSD-2-Clause",
    "BSD-3-Clause",
    "BSD-4-Clause",
    "ISC",
    "MIT",
    # Listed so that an MPL component is named rather than absent for an
    # unstated reason. The licence's source-availability obligation attaches to
    # modification and distribution, neither of which applies here.
    "MPL-2.0",
    "Zlib",
    # Assigned to a component whose licence could not be mapped to an SPDX id,
    # so that it surfaces in NOTICE rather than being dropped.
    "Unclassified",
}

# Substrings identifying a licence from its full text, for a component whose
# SBOM entry carries no SPDX id or expression. A crate setting `license-file`
# rather than `license` lands here: cargo-cyclonedx emits
# `{"license": {"name": "Unknown", "text": {...}}}` for it.
#
# Order matters: BSD-3-Clause's text contains BSD-2-Clause's, so it is tested
# first.
LICENSE_TEXT_SIGNATURES: List[Tuple[str, Tuple[str, ...]]] = [
    ("Apache-2.0", ("apache license", "version 2.0")),
    ("ISC", ("permission to use, copy, modify, and/or distribute this software",)),
    ("MIT", ("permission is hereby granted, free of charge",)),
    ("Zlib", ("altered source versions must be plainly marked",)),
    ("BSD-3-Clause", ("redistributions in binary form must reproduce", "neither the name of")),
    ("BSD-2-Clause", ("redistributions in binary form must reproduce",)),
]

# Licences that are real, recognised, and ask for nothing. They are listed so a
# component carrying one is KNOWN to need no attribution, rather than merely
# failing to match the set above -- the difference between a deliberate omission
# and a silent loss, which is the whole subject of this file.
NO_ATTRIBUTION_LICENSES: Set[str] = {
    "0BSD",
    "CC0-1.0",
    "MIT-0",
    "Unlicense",
    "WTFPL",
}

# Every licence identifier this script can reason about. A `name` is accepted as
# identification only if it appears here: names are free-form, so treating an
# unrecognised one ("MIT License", a bespoke corporate licence) as identification
# would mark the component classified, skip the licence text that could have
# placed it, and then drop it for not matching the attribution set -- reinstating
# exactly the silent loss this script exists to prevent. Anything else falls
# through to the text, and failing that to Unclassified, which is attributed.
KNOWN_LICENSES: Set[str] = (
    ATTRIBUTION_REQUIRED_LICENSES | NO_ATTRIBUTION_LICENSES) - {"Unclassified"}


def identify_license_text(text_entry: Dict) -> Set[str]:
    """Identify a licence from its embedded full text.

    Returns the matching SPDX ids, or {"Unclassified"} when text is present but
    unrecognised. Never an empty set: that would drop the component.
    """
    content = text_entry.get("content")
    if not content:
        return set()

    if text_entry.get("encoding") == "base64":
        try:
            content = base64.b64decode(content).decode("utf-8", errors="replace")
        except ValueError:  # binascii.Error subclasses this
            return {"Unclassified"}

    lowered = content.lower()
    for spdx_id, needles in LICENSE_TEXT_SIGNATURES:
        if all(needle in lowered for needle in needles):
            return {spdx_id}
    return {"Unclassified"}


def extract_license_from_component(component: Dict) -> Set[str]:
    """Extract all license identifiers from a component."""
    licenses: Set[str] = set()

    if "licenses" not in component:
        return licenses

    for lic_entry in component["licenses"]:
        license_obj = lic_entry.get("license") or {}
        identified = False

        # Check for direct license ID
        if "id" in license_obj:
            licenses.add(license_obj["id"])
            identified = True

        # A `name` is free-form, so it identifies a licence only when it is a
        # recognised identifier. "Unknown" and "MIT License" alike fall through
        # to the licence text below.
        name = (license_obj.get("name") or "").strip()
        if name in KNOWN_LICENSES:
            licenses.add(name)
            identified = True

        # Last resort: the full licence text. Only consulted when neither an id
        # nor a recognised name was found, so a component is never dropped for
        # expressing its licence as prose.
        if not identified and "text" in license_obj:
            identified_text = identify_license_text(license_obj["text"])
            licenses |= identified_text
            identified = identified or bool(identified_text)

        # The entry carried a name or text but nothing above could identify it.
        # Unclassified keeps the component in NOTICE, where an unreadable
        # licence is visible; dropping it would not be.
        if not identified and (name or "text" in license_obj):
            licenses.add("Unclassified")

        # Check for SPDX expression
        if "expression" in lic_entry:
            expr = lic_entry["expression"]
            # Parse expression (simplified - splits on OR/AND/WITH)
            parts = expr.replace("(", "").replace(")", "")
            parts = parts.replace(" OR ", " ").replace(" AND ", " ").replace(" WITH ", " ")
            for part in parts.split():
                if part and not part.isspace():
                    licenses.add(part)

    return licenses


def requires_attribution(licenses: Set[str]) -> bool:
    """Check if any of the licenses require attribution."""
    return bool(licenses.intersection(ATTRIBUTION_REQUIRED_LICENSES))


def generate_notice(sbom_path: str) -> str:
    """Generate NOTICE content from SBOM."""

    try:
        with open(sbom_path, 'r') as f:
            sbom = json.load(f)
    except Exception as e:
        print(f"Error reading SBOM: {e}", file=sys.stderr)
        sys.exit(1)

    components = sbom.get("components", [])

    # Extract components requiring attribution, deduplicating by (name, version)
    seen: Dict[tuple[str, str], Set[str]] = {}

    for component in components:
        name = component.get("name", "unknown")
        version = component.get("version", "unknown")
        licenses = extract_license_from_component(component)

        if licenses and requires_attribution(licenses):
            key = (name, version)
            if key in seen:
                seen[key].update(licenses)
            else:
                seen[key] = set(licenses)

    # Prefer versioned entries over "unknown" — if we have both, drop "unknown"
    versioned_names = {name for (name, ver) in seen if ver != "unknown"}
    attribution_components: List[tuple[str, str, Set[str]]] = [
        (name, version, lics)
        for (name, version), lics in seen.items()
        if version != "unknown" or name not in versioned_names
    ]

    # Sort by name, then version
    attribution_components.sort(key=lambda x: (x[0].lower(), x[1]))

    # Generate NOTICE content
    notice = []
    notice.append("EdgeFirst Hardware Abstraction Layer (HAL)")
    notice.append("Copyright © 2025 Au-Zone Technologies. All Rights Reserved.")
    notice.append("")
    notice.append("This product includes software developed at Au-Zone Technologies")
    notice.append("(https://au-zone.com/).")
    notice.append("")
    notice.append("This software contains components from the following third-party projects")
    notice.append("that require attribution:")
    notice.append("")
    notice.append("This list is derived from the resolved dependency graph, so it names every")
    notice.append("component under an attribution licence -- including ones reached only by a")
    notice.append("build script or only by the test suite, alongside those linked into the")
    notice.append("distributed artifacts. That breadth is deliberate: naming a component that")
    notice.append("is not shipped costs nothing, while omitting one that is shipped is the")
    notice.append("attribution gap this file exists to prevent. Entries are therefore not")
    notice.append("pruned by whether a component reaches a released binary.")
    notice.append("")

    if attribution_components:
        for name, version, licenses in attribution_components:
            license_str = ", ".join(sorted(licenses))
            notice.append(f"  * {name} {version} ({license_str})")
    else:
        notice.append("  (No third-party components requiring attribution)")

    notice.append("")
    notice.append("For a complete Software Content Register (SBOM) including all dependencies,")
    notice.append("licenses, and version information, see the sbom.json file generated via")
    notice.append("GitHub Actions in this repository or included in release artifacts.")
    notice.append("")

    return "\n".join(notice)


def attributable_entries(sbom_path: str) -> List[Tuple[str, str]]:
    """Every (name, version) in the SBOM whose licence requires attribution."""
    with open(sbom_path, "r") as handle:
        sbom = json.load(handle)

    # Deduplicated the way generate_notice() renders them: the merged SBOM can
    # carry the same (name, version) from several per-target scans, and the
    # count is meant to describe NOTICE's lines, not the merge's row count.
    entries = set()
    for component in sbom.get("components", []):
        licenses = extract_license_from_component(component)
        if licenses and requires_attribution(licenses):
            entries.add((component.get("name", "unknown"),
                         component.get("version", "unknown")))
    return sorted(entries)


def check_notice(sbom_path: str, notice_path: str) -> int:
    """Fail when NOTICE omits a component whose licence requires attribution.

    Scoped to what this script would write, not to every component in the SBOM.
    The generator omits components under licences asking for no attribution, so
    gating on the full list would fail on a component NOTICE is correct to omit,
    and regenerating could not clear it.

    Entries match on name and version, so a version bump leaving NOTICE stale is
    caught as well.
    """
    with open(notice_path, "r", encoding="utf-8") as handle:
        notice = handle.read()

    missing = [
        (name, version)
        for name, version in attributable_entries(sbom_path)
        if f"  * {name} {version} (" not in notice
    ]

    if missing:
        print(
            f"error: {len(missing)} component(s) requiring attribution are not "
            f"named in {notice_path}:",
            file=sys.stderr,
        )
        for name, version in sorted(missing):
            print(f"  - {name} {version}", file=sys.stderr)
        print(
            f"Run `make notice` to regenerate {notice_path}, then review the diff.",
            file=sys.stderr,
        )
        return 1

    print(f"{notice_path}: names all {len(attributable_entries(sbom_path))} components requiring attribution")
    return 0


def main():
    if len(sys.argv) == 4 and sys.argv[1] == "--check":
        sys.exit(check_notice(sys.argv[2], sys.argv[3]))

    if len(sys.argv) != 2:
        print(
            "Usage: generate_notice.py <sbom.json>\n"
            "       generate_notice.py --check <sbom.json> <NOTICE>",
            file=sys.stderr,
        )
        sys.exit(1)

    sbom_path = sys.argv[1]
    notice_content = generate_notice(sbom_path)
    print(notice_content)


if __name__ == "__main__":
    main()
