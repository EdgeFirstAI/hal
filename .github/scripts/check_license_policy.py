#!/usr/bin/env python3
"""
Check license policy compliance for SBOM (CycloneDX format)
Validates that all dependencies comply with license policy

Note: sublime_fuzzy 0.7.0 may show as unlicensed in SBOM scans,
but is confirmed Apache-2.0 at https://github.com/Schlechtwetterfront/fuzzy-rs
(transitive dependency via Rerun, optional feature only)
"""

import json
import sys
from typing import Set, List, Dict, Tuple

# License overrides for dependencies with missing or incorrect metadata
LICENSE_OVERRIDES = {
    "sublime_fuzzy@0.7.0": "Apache-2.0",  # Confirmed from upstream repo
    "dma-heap@0.4.1": "MIT",  # Confirmed from upstream: https://github.com/mripard/dma-heap/
    # Internal crates - all Apache-2.0 (workspace.package.license)
    # Versioned entries (from cargo-cyclonedx)
    "edgefirst-hal@0.5.0": "Apache-2.0",
    "edgefirst-hal-capi@0.5.0": "Apache-2.0",
    "edgefirst-decoder@0.5.0": "Apache-2.0",
    "edgefirst-image@0.5.0": "Apache-2.0",
    "edgefirst-tensor@0.5.0": "Apache-2.0",
    "edgefirst-tracker@0.5.0": "Apache-2.0",
    "edgefirst_hal@0.5.0": "Apache-2.0",  # Python package name
    "g2d-sys@1.0.1": "Apache-2.0",
    # Unversioned entries (from scancode source scan)
    "edgefirst-hal@unknown": "Apache-2.0",
    "edgefirst-hal-capi@unknown": "Apache-2.0",
    "edgefirst-decoder@unknown": "Apache-2.0",
    "edgefirst-image@unknown": "Apache-2.0",
    "edgefirst-tensor@unknown": "Apache-2.0",
    "edgefirst-tracker@unknown": "Apache-2.0",
    "edgefirst_hal@unknown": "Apache-2.0",
}

# License policy - CUSTOMIZE THESE FOR YOUR ORGANIZATION
ALLOWED_LICENSES: Set[str] = {
    "MIT",
    "MIT-0",  # MIT No Attribution (public domain equivalent)
    "Apache-2.0",
    "BSD-2-Clause",
    "BSD-3-Clause",
    "BSD-4-Clause",
    "ISC",
    "0BSD",
    "Unlicense",
    "Zlib",
    "BSL-1.0",  # Boost Software License
    "EPL-2.0",  # Eclipse Public License 2.0
    "Unicode-3.0",  # Unicode License
    "LLVM-exception",  # Apache-2.0 WITH LLVM-exception
    "CDLA-Permissive-2.0",  # Community Data License Agreement
    "CC0-1.0",  # Creative Commons Zero (public domain dedication)
    "OFL-1.1",  # SIL Open Font License (for fonts)
    "Ubuntu-font-1.0",  # Ubuntu Font License
    "MPL-2.0",  # Mozilla Public License 2.0 - file-level copyleft, safe as dependency
    "IJG",  # Independent JPEG Group License (permissive)
}

# Weak copyleft licenses requiring manual review
# Note: MPL-2.0 moved to ALLOWED - file-level copyleft, safe for dependencies
REVIEW_REQUIRED_LICENSES: Set[str] = {
    "LGPL-2.0",
    "LGPL-2.1",
    "LGPL-3.0",
}

DISALLOWED_LICENSES: Set[str] = {
    "GPL-1.0",
    "GPL-2.0",
    "GPL-3.0",
    "AGPL-1.0",
    "AGPL-3.0",
    "CC-BY-NC",
    "CC-BY-NC-SA",
    "CC-BY-ND",
    "SSPL",
    "SSPL-1.0",
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


def check_license_policy(sbom_path: str) -> Tuple[bool, List[str], List[str], List[str]]:
    """
    Check license policy compliance.

    Returns:
        (passed, disallowed_components, review_required_components, unknown_components)
    """
    try:
        with open(sbom_path, 'r') as f:
            sbom = json.load(f)
    except Exception as e:
        print(f"Error reading SBOM: {e}", file=sys.stderr)
        sys.exit(1)

    components = sbom.get("components", [])

    disallowed_components: List[str] = []
    review_required_components: List[str] = []
    unknown_components: List[str] = []

    for component in components:
        name = component.get("name", "unknown")
        version = component.get("version", "unknown")
        component_id = f"{name}@{version}"
        
        # Check for license override first
        if component_id in LICENSE_OVERRIDES:
            # Use overridden license
            continue
        
        licenses = extract_license_from_component(component)

        if not licenses:
            # No license information found
            unknown_components.append(f"{name} {version} (no license information)")
            continue

        # Check for disallowed licenses
        disallowed = licenses.intersection(DISALLOWED_LICENSES)
        if disallowed:
            disallowed_components.append(
                f"{name} {version} - DISALLOWED: {', '.join(sorted(disallowed))}"
            )
            continue

        # Check for licenses requiring review
        review_required = licenses.intersection(REVIEW_REQUIRED_LICENSES)
        if review_required:
            review_required_components.append(
                f"{name} {version} - REVIEW REQUIRED: {', '.join(sorted(review_required))}"
            )
            continue

        # Check for unknown licenses (not in allowed, review, or disallowed lists)
        known_licenses = ALLOWED_LICENSES | REVIEW_REQUIRED_LICENSES | DISALLOWED_LICENSES
        unknown = licenses - known_licenses
        if unknown:
            unknown_components.append(
                f"{name} {version} - UNKNOWN: {', '.join(sorted(unknown))}"
            )

    passed = len(disallowed_components) == 0
    return passed, disallowed_components, review_required_components, unknown_components


def main():
    if len(sys.argv) != 2:
        print("Usage: check_license_policy.py <sbom.json>", file=sys.stderr)
        sys.exit(1)

    sbom_path = sys.argv[1]
    passed, disallowed, review_required, unknown = check_license_policy(sbom_path)

    print("=" * 80)
    print("License Policy Check")
    print("=" * 80)
    print()

    if disallowed:
        print("DISALLOWED LICENSES FOUND:")
        print("-" * 80)
        for component in disallowed:
            print(f"  ❌ {component}")
        print()

    if review_required:
        print("LICENSES REQUIRING MANUAL REVIEW:")
        print("-" * 80)
        for component in review_required:
            print(f"  ⚠️  {component}")
        print()

    if unknown:
        print("UNKNOWN LICENSES FOUND:")
        print("-" * 80)
        for component in unknown:
            print(f"  ❓ {component}")
        print()

    if passed and not review_required and not unknown:
        print("✅ All dependencies comply with license policy!")
        print()
        sys.exit(0)
    elif passed and (review_required or unknown):
        print("⚠️  License policy check passed, but manual review required:")
        if review_required:
            print(f"   - {len(review_required)} component(s) with licenses requiring review")
        if unknown:
            print(f"   - {len(unknown)} component(s) with unknown licenses")
        print()
        sys.exit(0)
    else:
        print("❌ License policy check FAILED!")
        print(f"   - {len(disallowed)} component(s) with disallowed licenses")
        print()
        sys.exit(1)


if __name__ == "__main__":
    main()
