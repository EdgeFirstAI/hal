#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
# SPDX-License-Identifier: Apache-2.0
#
# Runs crates/tensor/tests/scenarios.rs under Miri with Stacked Borrows and
# with Tree Borrows, then runs the ignored diagnostic test under Stacked
# Borrows and checks that it fails with its retag-invalidation signature.
#
# Exits 0 when all three checks pass, 1 when any fails, and 2 when the Miri
# component is not installed.
set -uo pipefail

cd "$(dirname "$0")/.." || exit 1

# The nightly toolchain named by RUST_NIGHTLY_VERSION, or `nightly`.
NIGHTLY="${RUST_NIGHTLY_VERSION:-nightly}"

if ! rustup "+${NIGHTLY}" component list --installed 2>/dev/null | grep -q '^miri-'; then
  echo "miri.sh: no Miri component installed for the ${NIGHTLY} toolchain." >&2
  echo "  Install with: rustup +${NIGHTLY} component add miri rust-src" >&2
  exit 2
fi

fails=0

run_model() {
  local label="$1"
  shift
  echo "== Miri (${label}): crates/tensor/tests/scenarios.rs =="
  if "$@"; then
    echo "  PASS  ${label}"
  else
    echo "  FAIL  ${label}"
    fails=$((fails + 1))
  fi
  return 0
}

run_model "Stacked Borrows (default)" \
  cargo "+${NIGHTLY}" miri test -p edgefirst-tensor --test scenarios

run_model "Tree Borrows (-Zmiri-tree-borrows)" \
  env MIRIFLAGS="-Zmiri-tree-borrows" cargo "+${NIGHTLY}" miri test -p edgefirst-tensor --test scenarios

# Runs the ignored diagnostic under Stacked Borrows. It must fail with
# "trying to retag from" and name the three source lines that create,
# invalidate and read the aliasing pair, which are located by their text.
scenarios_rs="crates/tensor/tests/scenarios.rs"
diag_line_a=$(grep -n '^        let a = &mut \*(raw as \*mut TensorDyn);' "${scenarios_rs}" | head -1 | cut -d: -f1)
diag_line_b=$(grep -n '^        let b = &mut \*(raw as \*mut TensorDyn);' "${scenarios_rs}" | head -1 | cut -d: -f1)
diag_line_read=$(grep -n '^        let _ = a\.dtype();' "${scenarios_rs}" | head -1 | cut -d: -f1)
diag_log="${TMPDIR:-/tmp}/miri-sh-diagnostic.$$.log"
echo "== Miri (Stacked Borrows): the diagnostic itself (--ignored) =="
if [[ -z "${diag_line_a}" ]] || [[ -z "${diag_line_b}" ]] || [[ -z "${diag_line_read}" ]]; then
  echo "  FAIL  could not locate the diagnostic's three anchor lines in ${scenarios_rs} -- has it been rewritten? update this script's anchors"
  fails=$((fails + 1))
else
cargo "+${NIGHTLY}" miri test -p edgefirst-tensor --test scenarios \
  -- --ignored unwrap_then_use_aliases_the_same_tensor >"${diag_log}" 2>&1
diag_rc=$?
if [[ "${diag_rc}" -ne 0 ]] \
  && grep -q 'trying to retag from' "${diag_log}" \
  && grep -q "scenarios.rs:${diag_line_a}" "${diag_log}" \
  && grep -q "scenarios.rs:${diag_line_b}" "${diag_log}" \
  && grep -q "scenarios.rs:${diag_line_read}" "${diag_log}"; then
  echo "  PASS  diagnostic still demonstrates the retag-invalidation hazard"
elif [[ "${diag_rc}" -ne 0 ]]; then
  # Failed without the expected signature.
  echo "  FAIL  diagnostic failed, but NOT with the expected retag-invalidation signature -- see ${diag_log}"
  fails=$((fails + 1))
else
  # The diagnostic passed.
  echo "  MODEL CHANGED  the diagnostic passes under Stacked Borrows -- see ${diag_log}"
  fails=$((fails + 1))
fi
rm -f "${diag_log}"
fi

echo
if [[ "${fails}" -eq 0 ]]; then
  echo "PASS: scenarios.rs is clean under both aliasing models, and the diagnostic still demonstrates the hazard it exists to record"
  exit 0
fi
echo "FAIL: ${fails} check(s) failed (see above for which)"
exit 1
