#!/bin/bash
# Cross-compile benchmarks with OpenCV for ARM64 using Yocto SDK
#
# This script is ONLY needed for cross-compilation from x86_64.
# For native ARM64 builds (e.g., GitHub Actions ubuntu-22.04-arm), just run:
#   cargo build --examples -p edgefirst-image --features opencv --release
#
# Usage:
#   ./scripts/build-opencv-benchmark.sh <path-to-yocto-sdk>
#
# Example:
#   ./scripts/build-opencv-benchmark.sh /opt/yocto-sdk-imx8mp-frdm-6.12.49-2.2.0

set -e

SDK_ROOT="${1:-}"

if [ -z "$SDK_ROOT" ] || [ ! -d "$SDK_ROOT" ]; then
    echo "Usage: $0 <path-to-yocto-sdk>"
    echo ""
    echo "Cross-compiles benchmarks with OpenCV using a Yocto SDK."
    echo ""
    echo "Example:"
    echo "  $0 /opt/yocto-sdk-imx8mp-frdm-6.12.49-2.2.0"
    echo "  $0 /opt/yocto-sdk-imx95-frdm-6.12.49-2.2.0"
    echo ""
    echo "For native ARM64 builds, just use cargo directly:"
    echo "  cargo build --examples -p edgefirst-image --features opencv --release"
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT"

# Find the environment setup script
ENV_SCRIPT=$(ls "$SDK_ROOT"/environment-setup-* 2>/dev/null | head -1)
if [ -z "$ENV_SCRIPT" ]; then
    echo "ERROR: No environment-setup-* script found in $SDK_ROOT"
    exit 1
fi

echo "Using SDK: $SDK_ROOT"
echo "Environment script: $ENV_SCRIPT"

# Source the SDK environment
source "$ENV_SCRIPT"

# Extract compiler binary (first word) and flags (rest)
read -r CC_BIN CC_FLAGS <<< "$CC"
read -r CXX_BIN CXX_FLAGS <<< "$CXX"

# Find full path to compiler binary
CC_PATH=$(which "$CC_BIN" 2>/dev/null || echo "$CC_BIN")
CXX_PATH=$(which "$CXX_BIN" 2>/dev/null || echo "$CXX_BIN")

echo ""
echo "Compiler configuration:"
echo "  CC:     $CC_PATH"
echo "  CFLAGS: $CC_FLAGS"
echo "  CXX:    $CXX_PATH"
echo "  CXXFLAGS: $CXX_FLAGS"
echo "  Sysroot: $SDKTARGETSYSROOT"

# Set up environment for Rust/OpenCV build
export CC="$CC_PATH"
export CXX="$CXX_PATH"
export CFLAGS="$CC_FLAGS"
export CXXFLAGS="$CXX_FLAGS"
export AR="${AR:-${CC_BIN%gcc}ar}"

# Cargo linker configuration
export CARGO_TARGET_AARCH64_UNKNOWN_LINUX_GNU_LINKER="$CC_PATH"
export CARGO_TARGET_AARCH64_UNKNOWN_LINUX_GNU_RUSTFLAGS="-C link-arg=--sysroot=$SDKTARGETSYSROOT"

# OpenCV manual linking (avoids pkg-config issues with missing optional modules)
export OPENCV_LINK_LIBS="opencv_imgproc,opencv_core"
export OPENCV_LINK_PATHS="$SDKTARGETSYSROOT/usr/lib"
export OPENCV_INCLUDE_PATHS="$SDKTARGETSYSROOT/usr/include/opencv4"

echo ""
echo "Building benchmarks..."

# Build the Criterion benchmark
cargo build --bench pipeline_benchmark -p edgefirst-image \
    --target aarch64-unknown-linux-gnu \
    --features opencv \
    --release

echo ""
echo "Build complete!"
echo "Criterion benchmark: target/aarch64-unknown-linux-gnu/release/deps/pipeline_benchmark-*"
