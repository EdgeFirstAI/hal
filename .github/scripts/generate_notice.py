#!/usr/bin/env python3
"""
Generate NOTICE file from SBOM (CycloneDX format)
Extracts packages requiring attribution based on their licenses
"""

import json
import sys
from typing import Set, List, Dict

# Licenses that require attribution in NOTICE file
ATTRIBUTION_REQUIRED_LICENSES: Set[str] = {
    "Apache-2.0",
    "BSD-2-Clause",
    "BSD-3-Clause",
    "BSD-4-Clause",
    "MIT",
}


def extract_license_from_component(component: Dict) -> Set[str]:
    """Extract all license identifiers from a component."""
    licenses: Set[str] = set()

    if "licenses" not in component:
        return licenses

    for lic_entry in component["licenses"]:
        # Check for direct license ID
        if "license" in lic_entry and "id" in lic_entry["license"]:
            licenses.add(lic_entry["license"]["id"])

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


def main():
    if len(sys.argv) != 2:
        print("Usage: generate_notice.py <sbom.json>", file=sys.stderr)
        sys.exit(1)

    sbom_path = sys.argv[1]
    notice_content = generate_notice(sbom_path)
    print(notice_content)


if __name__ == "__main__":
    main()
