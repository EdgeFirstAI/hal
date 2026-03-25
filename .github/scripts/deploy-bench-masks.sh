#!/usr/bin/env bash
# Cross-compile mask_benchmark, deploy to ARM targets, run, collect JSON results.
#
# Usage:
#   .github/scripts/deploy-bench-masks.sh [target...]
#
# Targets default to: imx8mpevk-06 imx95-evk
# Each target must be reachable via SSH (ssh <target>).
#
# Results are saved to benchmarks/<target>/mask-<backend>-baseline.json

set -euo pipefail

BENCH_BIN="mask_benchmark"
CARGO_TARGET="aarch64-unknown-linux-gnu"
BUILD_DIR="target/${CARGO_TARGET}/release/deps"
LOCAL_RESULTS="benchmarks"
WARMUP=10
ITERATIONS=200

# Default targets (override with CLI args)
if [ $# -gt 0 ]; then
    TARGETS=("$@")
else
    TARGETS=("imx8mpevk-06" "imx95-evk")
fi

echo "=== Mask Benchmark Deploy & Collect ==="
echo "Targets: ${TARGETS[*]}"
echo ""

# Step 1: Cross-compile
echo "==> Cross-compiling ${BENCH_BIN} for ${CARGO_TARGET}..."
cargo-zigbuild build --target "${CARGO_TARGET}" --release \
    --features opengl -p edgefirst-image --bench "${BENCH_BIN}"

# Step 2: Find the benchmark binary
BIN_PATH=$(find "${BUILD_DIR}" -maxdepth 1 -type f -executable -name "${BENCH_BIN}-*" \
    ! -name "*.d" -newer Cargo.toml | head -1)
if [ -z "${BIN_PATH}" ]; then
    echo "ERROR: Could not find ${BENCH_BIN} binary in ${BUILD_DIR}"
    exit 1
fi
echo "==> Binary: ${BIN_PATH}"
echo ""

# Step 3: Deploy and run on each target
for target in "${TARGETS[@]}"; do
    echo "============================================"
    echo "==> Target: ${target}"
    echo "============================================"

    mkdir -p "${LOCAL_RESULTS}/${target}"

    echo "==> Deploying binary..."
    scp -q "${BIN_PATH}" "${target}:/tmp/${BENCH_BIN}"
    ssh "${target}" "chmod +x /tmp/${BENCH_BIN}"

    for backend in auto opengl cpu; do
        echo ""
        echo "--- ${target} / ${backend} ---"
        JSON_FILE="mask-${backend}-baseline.json"
        REMOTE_JSON="/tmp/${JSON_FILE}"

        ENV_PREFIX=""
        if [ "${backend}" != "auto" ]; then
            ENV_PREFIX="EDGEFIRST_FORCE_BACKEND=${backend}"
        fi

        if ssh "${target}" "${ENV_PREFIX} /tmp/${BENCH_BIN} --bench --json ${REMOTE_JSON}" 2>&1 | tee "${LOCAL_RESULTS}/${target}/mask-${backend}-baseline.log"; then
            scp -q "${target}:${REMOTE_JSON}" "${LOCAL_RESULTS}/${target}/${JSON_FILE}"
            echo "  -> Saved: ${LOCAL_RESULTS}/${target}/${JSON_FILE}"
        else
            echo "  -> FAILED (backend may not be available)"
        fi
    done

    # Cleanup remote
    ssh "${target}" "rm -f /tmp/${BENCH_BIN} /tmp/mask-*.json" 2>/dev/null || true
    echo ""
done

# Step 4: Local x86 benchmarks
echo "============================================"
echo "==> Local x86 benchmarks"
echo "============================================"
mkdir -p "${LOCAL_RESULTS}/x86-desktop"

cargo bench -p edgefirst-image --bench "${BENCH_BIN}" -- --bench --json "${LOCAL_RESULTS}/x86-desktop/mask-auto-baseline.json" 2>&1 | tee "${LOCAL_RESULTS}/x86-desktop/mask-auto-baseline.log"

echo ""
echo "=== All baselines collected ==="
echo "Results in: ${LOCAL_RESULTS}/*/mask-*-baseline.json"
