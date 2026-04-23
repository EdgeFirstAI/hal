#!/usr/bin/env bash
# EDGEAI-1244 native-f16 codegen auditor.
#
# Dumps the disassembly of the HAL's f16 mask kernel and checks for
# native-FP16 instructions (fcvt / vcvtph2ps), failing if the compiler
# emitted the soft-float helper (__extendhfsf2) instead.
#
# Usage:
#   RUSTFLAGS="-C target-cpu=cortex-a78ae" \
#       scripts/audit_f16_codegen.sh aarch64-unknown-linux-gnu
#
#   RUSTFLAGS="-C target-feature=+f16c" \
#       scripts/audit_f16_codegen.sh x86_64-unknown-linux-gnu
#
# Requires cargo-asm:
#   cargo install cargo-show-asm

set -euo pipefail

TARGET="${1:-}"
if [[ -z "$TARGET" ]]; then
    echo "Usage: $0 <target-triple>" >&2
    exit 2
fi

SYMBOL='fused_dot_sigmoid_f16_slice'

case "$TARGET" in
    aarch64-*)
        EXPECTED_INSTRS='(fcvt|fcvtl|fcvtl2|fcvtn|fcvtn2)'
        ;;
    x86_64-*)
        EXPECTED_INSTRS='(vcvtph2ps|vcvtps2ph)'
        ;;
    *)
        echo "Unsupported target: $TARGET" >&2
        exit 2
        ;;
esac

echo "==> Auditing $SYMBOL on $TARGET"
echo "==> Expected instructions: $EXPECTED_INSTRS"

TMP_ASM=$(mktemp --suffix=.asm)
trap "rm -f $TMP_ASM" EXIT

cargo asm --release --target "$TARGET" -p edgefirst-image \
    --lib "$SYMBOL" > "$TMP_ASM" 2>/dev/null || {
    echo "!! cargo asm failed; is cargo-show-asm installed?" >&2
    echo "!! install: cargo install cargo-show-asm" >&2
    exit 1
}

if grep -q '__extendhfsf2\|__truncsfhf2' "$TMP_ASM"; then
    echo "FAIL: soft-float helper found in $SYMBOL disassembly"
    grep -n '__extendhfsf2\|__truncsfhf2' "$TMP_ASM" | head -5
    echo ""
    echo "Check that the correct target-cpu or target-feature is set in RUSTFLAGS."
    exit 1
fi

if ! grep -qE "$EXPECTED_INSTRS" "$TMP_ASM"; then
    echo "FAIL: no native-FP16 instructions ($EXPECTED_INSTRS) found"
    echo "Top 20 non-empty disassembly lines:"
    grep -v '^$' "$TMP_ASM" | head -20
    exit 1
fi

MATCH_COUNT=$(grep -cE "$EXPECTED_INSTRS" "$TMP_ASM" || true)
echo "PASS: $MATCH_COUNT native-FP16 instructions found in $SYMBOL"
