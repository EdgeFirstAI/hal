#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Copyright © 2025 Au-Zone Technologies. All Rights Reserved.
#
# Generate complete SBOM by merging cargo-cyclonedx (dependencies) and scancode (source)
# Fast execution (~5-6 seconds) using parent directory scanning to avoid scancode path bugs

set -e  # Exit on error

echo "=================================================="
echo "Generating Complete SBOM for EdgeFirst HAL"
echo "=================================================="
echo

# Step 1: Generate dependency SBOM with cargo-cyclonedx (~1 second per target)
#
# Without an explicit --target, cargo-cyclonedx resolves only the
# dependency graph for the HOST target (`rustc -vV`), silently dropping
# every platform-gated dependency that isn't reachable from the machine
# generating the SBOM (e.g. running this on macOS drops the Linux-only
# g2d-sys/linux-raw-sys/safe_arch attributions NOTICE needs). NOTICE must
# cover every platform this HAL ships to regardless of which host runs
# this script.
#
# `--target all` would cover every platform Cargo.lock could resolve for
# ANY Rust target (wasm32, UEFI, ...), pulling in attributions for code
# this HAL never ships (e.g. getrandom's UEFI backend drags in r-efi)
# and drowning the license-policy report in irrelevant unknown-license
# warnings. Instead, enumerate exactly the targets this HAL ships/tests
# for (kept in sync with the CI build matrix in .github/workflows/) and
# merge their per-target SBOMs — a precise union, not an ecosystem-wide
# one. Target resolution only needs rustc's built-in target spec, not an
# installed toolchain (`rustup target add` not required).
HAL_TARGETS=(
    x86_64-unknown-linux-gnu    # Desktop Linux; also stands in for aarch64-unknown-linux-gnu (embedded i.MX/RPi5), which resolves the same target_os="linux" dependency graph
    aarch64-apple-darwin        # macOS
    aarch64-apple-ios           # iOS device
    aarch64-apple-ios-sim       # iOS simulator
    aarch64-linux-android       # Android device
    x86_64-linux-android        # Android emulator
    x86_64-pc-windows-msvc      # Windows
)
echo "[1/5] Generating dependency SBOM with cargo-cyclonedx (${#HAL_TARGETS[@]} targets)..."
# Drop any leftover *.cdx.json before regenerating. Older cargo-cyclonedx
# invocations wrote host-only files without --target-in-filename
# (e.g. crates/decoder/edgefirst-decoder.cdx.json); those keep the
# workspace version from the run that produced them and would otherwise
# pollute the merge with stale internal-crate versions (breaking
# verify_version.py / NOTICE). Always start from a clean slate.
find crates -maxdepth 2 -type f -name '*.cdx.json' -delete
for t in "${HAL_TARGETS[@]}"; do
    echo "  target: ${t}"
    cargo cyclonedx --format json --all --target "${t}" --target-in-filename
done

# Merge all crate SBOMs into a single dependency SBOM
if ! command -v cyclonedx &> /dev/null; then
    if ! command -v ~/.local/bin/cyclonedx &> /dev/null; then
        echo "Error: cyclonedx CLI not found. Please install from https://github.com/CycloneDX/cyclonedx-cli"
        exit 1
    fi
    CYCLONEDX=~/.local/bin/cyclonedx
else
    CYCLONEDX=cyclonedx
fi

# Find all generated .cdx.json files and merge them
CRATE_SBOMS=$(find crates -name "*.cdx.json" | tr '\n' ' ')
if [ -n "$CRATE_SBOMS" ]; then
    $CYCLONEDX merge --input-files $CRATE_SBOMS --output-file edgefirst_hal.cdx.json
    echo "✓ Generated edgefirst_hal.cdx.json (Rust dependencies from all crates)"
else
    echo "Warning: No crate SBOMs found"
fi
echo

# Step 2: Generate source code SBOM with scancode (~5 seconds)
echo "[2/5] Generating source code SBOM with scancode..."
if [ ! -f "venv/bin/scancode" ]; then
    echo "Error: scancode not found. Please install: python3 -m venv venv && venv/bin/pip install scancode-toolkit"
    exit 1
fi

# Workaround for scancode path resolution bugs (GitHub issues #2966, #4459):
# - Scanning from project directory causes recursive traversal into target/ and venv/
# Solution: Run scancode from parent directory
ORIGINAL_DIR=$(pwd)
PROJECT_NAME=$(basename "$ORIGINAL_DIR")
cd ..

# Scan Rust crates directory
echo "  Scanning Rust crates..."
"$ORIGINAL_DIR/venv/bin/scancode" -clpieu \
    --cyclonedx "$ORIGINAL_DIR/source-crates-raw.json" \
    --only-findings \
    "$PROJECT_NAME/crates/"

echo "  Scanning Cargo.toml..."
"$ORIGINAL_DIR/venv/bin/scancode" -clpieu \
    --cyclonedx "$ORIGINAL_DIR/source-cargo-toml-raw.json" \
    --only-findings \
    "$PROJECT_NAME/Cargo.toml"

echo "  Scanning Cargo.lock..."
"$ORIGINAL_DIR/venv/bin/scancode" -clpieu \
    --cyclonedx "$ORIGINAL_DIR/source-cargo-lock-raw.json" \
    --only-findings \
    "$PROJECT_NAME/Cargo.lock"

echo "  Scanning Python tests..."
"$ORIGINAL_DIR/venv/bin/scancode" -clpieu \
    --cyclonedx "$ORIGINAL_DIR/source-tests-raw.json" \
    --only-findings \
    "$PROJECT_NAME/tests/"

# Return to original directory
cd "$ORIGINAL_DIR"

echo "✓ Generated raw source SBOM files"
echo

# Step 3: Clean and merge scancode outputs
echo "[3/5] Cleaning and merging scancode outputs..."
python3 << 'PYCLEAN'
import json
import sys

# Clean each scancode JSON
files_to_clean = [
    'source-crates-raw.json',
    'source-cargo-toml-raw.json',
    'source-cargo-lock-raw.json',
    'source-tests-raw.json'
]

for raw_file in files_to_clean:
    try:
        with open(raw_file) as f:
            data = json.load(f)
        
        output_file = raw_file.replace('-raw.json', '.json')
        
        # Clean the metadata properties
        if 'metadata' in data and 'properties' in data['metadata']:
            data['metadata']['properties'] = [
                p for p in data['metadata']['properties']
                if isinstance(p.get('value'), str)
            ]
        
        with open(output_file, 'w') as f:
            json.dump(data, f, indent=2)
    except Exception as e:
        print(f"Warning: Could not process {raw_file}: {e}", file=sys.stderr)
        continue

sys.exit(0)
PYCLEAN

echo "✓ Cleaned scancode outputs"
echo

# Step 4: Merge source SBOMs using cyclonedx-cli
echo "[4/5] Merging source SBOMs..."
if ! command -v cyclonedx &> /dev/null; then
    if ! command -v ~/.local/bin/cyclonedx &> /dev/null; then
        echo "Error: cyclonedx CLI not found. Please install from https://github.com/CycloneDX/cyclonedx-cli"
        exit 1
    fi
    CYCLONEDX=~/.local/bin/cyclonedx
else
    CYCLONEDX=cyclonedx
fi

$CYCLONEDX merge \
    --input-files source-crates.json source-cargo-toml.json source-cargo-lock.json source-tests.json \
    --output-file source-sbom.json

# Clean up intermediate files
rm -f source-*-raw.json source-crates.json source-cargo-toml.json source-cargo-lock.json source-tests.json

echo "✓ Merged source SBOMs into source-sbom.json"
echo

# Step 5: Merge dependency and source SBOMs
echo "[5/5] Merging dependency and source SBOMs..."
$CYCLONEDX merge \
    --input-files edgefirst_hal.cdx.json source-sbom.json \
    --output-file sbom.json

echo "✓ Generated sbom.json (merged: dependencies + source)"
echo

# Check license policy
echo "Checking license policy compliance..."
python3 .github/scripts/check_license_policy.py sbom.json
POLICY_EXIT=$?
echo

# Generate NOTICE file
echo "Generating NOTICE file..."
python3 .github/scripts/generate_notice.py sbom.json > NOTICE
echo "✓ Generated NOTICE (third-party attributions)"
echo

# Cleanup temporary files
rm -f source-sbom.json edgefirst_hal.cdx.json

echo "=================================================="
echo "SBOM Generation Complete"
echo "=================================================="
echo "Files generated:"
echo "  - sbom.json (merged SBOM)"
echo "  - NOTICE (third-party attributions)"
echo

# Exit with license policy check result
exit $POLICY_EXIT
