#!/usr/bin/env bash
# Coverage must still attribute to every crate after the packaging split.
#
# The failure mode here is silent UNDER-reporting, not an error: splitting into
# five dynamic objects changes where .profraw files originate, and the
# .init_array constructor that each cdylib used to carry is gone -- covguard is
# now installed from each pymodule_init instead. If aggregation breaks, the
# report simply shrinks, which no test would otherwise notice.
set -euo pipefail
REPORT="${1:-target/rust-coverage.lcov}"

[ -f "$REPORT" ] || { echo "ERROR: $REPORT not found; run 'make test-rust'"; exit 1; }

fail=0
for crate in tensor codec image decoder tracker; do
    if grep -q "crates/$crate/src" "$REPORT"; then
        n=$(grep -c "crates/$crate/src" "$REPORT")
        printf '  %-9s %s records\n' "$crate" "$n"
    else
        echo "  ERROR: no coverage recorded for crates/$crate/src"
        fail=1
    fi
done

# The five extension cdylibs are excluded from instrumentation on purpose --
# they are registration shims with no logic worth measuring, and llvm-cov
# cannot usefully instrument a cdylib. Their bindings live in python-common,
# which IS measured, so assert that rather than the shims.
if grep -q "crates/python-common/src" "$REPORT"; then
    echo "  python-common covered (the shims are excluded by design)"
else
    echo "  WARNING: no coverage for crates/python-common/src"
fi

[ "$fail" -eq 0 ] && echo "✓ coverage attributes to all five crates after the split"
exit "$fail"
