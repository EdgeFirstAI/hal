#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Copyright © 2025 Au-Zone Technologies. All Rights Reserved.
#
# Setup coverage environment for cargo-llvm-cov instrumented builds.
#
# This script configures environment variables so that:
# 1. cargo-llvm-cov builds instrumented binaries
# 2. maturin uses the same target directory as cargo-llvm-cov
# 3. profraw files are written to the llvm-cov-target directory
#
# Usage:
#   source .github/scripts/setup-coverage-env.sh
#
# The script writes environment exports to /tmp/coverage-env.sh which
# should be sourced before running maturin or tests that exercise
# the Python bindings.

set -e

# Clean previous coverage data
cargo llvm-cov clean --workspace

# Export coverage environment (filter out info/warning lines)
cargo llvm-cov show-env --export-prefix 2>&1 | grep '^export ' > /tmp/coverage-env.sh

# Renames an exported RUSTFLAGS to the host target's CARGO_TARGET_<TRIPLE>_RUSTFLAGS.
host_rustflags_var="CARGO_TARGET_$(rustc -vV | sed -n 's/^host: //p' | tr 'a-z-' 'A-Z_')_RUSTFLAGS"
sed -i "s/^export RUSTFLAGS=/export ${host_rustflags_var}=/" /tmp/coverage-env.sh

# Configure maturin to use the same target directory as cargo-llvm-cov
echo 'export CARGO_TARGET_DIR="$CARGO_LLVM_COV_TARGET_DIR/llvm-cov-target"' >> /tmp/coverage-env.sh

# Write profraw files to the llvm-cov-target directory
echo 'export LLVM_PROFILE_FILE="$CARGO_LLVM_COV_TARGET_DIR/llvm-cov-target/hal-%p-%16m.profraw"' >> /tmp/coverage-env.sh

echo "Coverage environment configured in /tmp/coverage-env.sh"
