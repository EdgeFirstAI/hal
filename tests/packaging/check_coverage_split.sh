#!/usr/bin/env bash
# Checks that the coverage report attributes records to every crate.
set -euo pipefail
REPORT="${1:-target/rust-coverage.lcov}"

[ -f "$REPORT" ] || { echo "ERROR: $REPORT not found; run 'make test-rust'"; exit 1; }

fail=0
for crate in tensor codec image decoder tracker kernels; do
    if grep -q "crates/$crate/src" "$REPORT"; then
        n=$(grep -c "crates/$crate/src" "$REPORT")
        printf '  %-9s %s records\n' "$crate" "$n"
    else
        echo "  ERROR: no coverage recorded for crates/$crate/src"
        fail=1
    fi
done

# The extension cdylibs are not instrumented; their bindings are measured
# through python-common.
if grep -q "crates/python-common/src" "$REPORT"; then
    n=$(grep -c "crates/python-common/src" "$REPORT")
    printf '  %-9s %s records\n' "python-common" "$n"
else
    echo "  ERROR: no coverage recorded for crates/python-common/src"
    fail=1
fi

[ "$fail" -eq 0 ] && echo "✓ coverage attributes to every crate after the split"
exit "$fail"
