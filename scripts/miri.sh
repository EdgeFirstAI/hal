#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
# SPDX-License-Identifier: Apache-2.0
#
# Runs the dynamic-backend aliasing battery
# (crates/tensor/tests/scenarios.rs) under Miri, under BOTH aliasing
# models: Stacked Borrows (Miri's default, the older and stricter model)
# and Tree Borrows (`-Zmiri-tree-borrows`, newer and more permissive). This
# is what G7 (scripts/check-single-home.sh) gates on. The two models can
# and do disagree on some shapes -- see task-11-report.md for a concrete
# example this file's own diagnostic test surfaced. A disagreement is a
# finding to report, not a reason to tune either run until it goes quiet.
#
# A third check, after the two models above, actually runs that diagnostic
# (`#[ignore]`d, so the two runs above never touch it) under Stacked
# Borrows and asserts its specific failure signature -- not just "it
# failed", since a build error or an OOM would satisfy that too and read
# as confirmation when it isn't one. See that check's own comments below
# for what it does when the diagnostic unexpectedly passes.
#
# No special feature flags needed: `dma-heap 0.4.1` hardcodes its ioctl
# opcode constant as `u32` (src/ioctl.rs), matching rustix's `linux_raw`
# backend's `Opcode` type on real Linux. rustix's own build.rs deliberately
# forces the *libc* backend under Miri ("Miri doesn't support inline asm
# ... so if we're running under miri, use the libc backend"), and on a
# glibc target the libc backend's Opcode is `u64` -- so dma-heap fails a
# hard type-check (error[E0308]) before Miri ever interprets anything.
# crates/tensor/Cargo.toml's dma-heap dependency is gated
# `cfg(all(target_os = "linux", not(miri)))`, so it simply drops out of the
# dependency graph for this specific `cargo miri` invocation -- Cargo does
# respect `cfg(miri)` inside a `[target.'cfg(...)'.dependencies]` table
# (verified empirically while building this script; it is not universally
# documented). Every normal (non-Miri) build keeps dma-heap, completely
# unaffected -- no other crate's Cargo.toml needed to change. See that
# dependency's neighboring comment in Cargo.toml, crates/tensor/src/dma.rs's
# matching `not(miri)` cfg split, and task-11-report.md for the full
# account of how this was established (not guessed) before choosing the
# fix.
#
# scenarios.rs itself never needed dma-heap in the first place: its
# raw-handle borrow shapes (TensorDyn::into_raw/from_raw/with_raw) are the
# `static` backend's own Box-based implementation, pure Rust, and
# Tensor::new's auto-select path falls back to a plain heap (Mem) tensor
# whenever DMA construction is unavailable -- so these tests run unmodified
# whether dma-heap is compiled in or not.
#
# Miri cannot execute FFI at all -- not "executes it differently", refuses
# outright ("can't call foreign function `dlopen`" is a real error this
# effort hit, from an unrelated CUDA availability probe every Tensor::new
# triggers on Linux; see crates/tensor/src/cuda.rs's `#[cfg(miri)]` split
# and task-11-report.md). That means this script can only ever meaningfully
# exercise the `static` backend. The `dynamic` backend's methods are thin
# wrappers around real `libedgefirst_tensor.so` calls; there is no way to
# run those under Miri. `make test-two-library-user` and the `test-capi-*`
# targets are what actually exercise the FFI boundary -- a green run here
# is real evidence about the tensor implementation's own borrow shapes and
# no evidence at all about that boundary.
set -uo pipefail

cd "$(dirname "$0")/.." || exit 1

# Prefer the pinned nightly CI already installed (RUST_NIGHTLY_VERSION).
# Bare `+nightly` is a floating toolchain the x86 job does not set up.
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
    # Miri is slow and memory-hungry. A run killed by the OOM killer or a
    # resource limit looks like any other nonzero exit here -- if the
    # output above shows no "test result:" line and no Miri UB diagnostic,
    # that is a resource death, not a real failure, and should be reported
    # and retried rather than counted as a finding.
    echo "  FAIL  ${label}"
    fails=$((fails + 1))
  fi
}

run_model "Stacked Borrows (default)" \
  cargo "+${NIGHTLY}" miri test -p edgefirst-tensor --test scenarios

run_model "Tree Borrows (-Zmiri-tree-borrows)" \
  env MIRIFLAGS="-Zmiri-tree-borrows" cargo "+${NIGHTLY}" miri test -p edgefirst-tensor --test scenarios

# `unwrap_then_use_aliases_the_same_tensor` is `#[ignore]`d, so neither
# `run_model` call above ever executes it -- its own doc comment used to
# claim "if a future Miri accepts it, this test fails and says so", which
# was not true of anything that actually ran. This is what makes that claim
# true: run the diagnostic itself, under Stacked Borrows only (Tree Borrows
# is already known and documented to accept this shape -- see the test's
# own doc comment -- so re-running it there would only reconfirm a known
# result, not check anything).
#
# Asserting only that the run FAILS is not enough -- a Miri version change,
# an OOM, a build error, or an unrelated bit of UB would all satisfy that
# and read as "the disagreement still holds" when it might not. So this
# checks for the SPECIFIC retag-invalidation signature scenarios.rs's own
# diagnostic is documented to produce: the phrase "trying to retag from",
# plus provenance references to the three lines that create/invalidate/use
# the aliasing pair. All three must appear, or this is not confirmed to be
# the same failure the diagnostic was written to demonstrate.
#
# The three line numbers are read from the source at run time, by exact
# text, rather than hardcoded -- a hardcoded number is exactly the anchor
# that broke once already while this check was being written (a doc-comment
# edit above the test body shifted every line below it), and it is the same
# lesson check-single-home.sh's G6 drift probe already learned the same way
# (it anchors its own source patch by exact text, explicitly not by line
# number, for this reason).
scenarios_rs="crates/tensor/tests/scenarios.rs"
diag_line_a=$(grep -n '^        let a = &mut \*(raw as \*mut TensorDyn);' "${scenarios_rs}" | head -1 | cut -d: -f1)
diag_line_b=$(grep -n '^        let b = &mut \*(raw as \*mut TensorDyn);' "${scenarios_rs}" | head -1 | cut -d: -f1)
diag_line_read=$(grep -n '^        let _ = a\.dtype();' "${scenarios_rs}" | head -1 | cut -d: -f1)
diag_log="${TMPDIR:-/tmp}/miri-sh-diagnostic.$$.log"
echo "== Miri (Stacked Borrows): the diagnostic itself (--ignored) =="
if [ -z "${diag_line_a}" ] || [ -z "${diag_line_b}" ] || [ -z "${diag_line_read}" ]; then
  echo "  FAIL  could not locate the diagnostic's three anchor lines in ${scenarios_rs} -- has it been rewritten? update this script's anchors"
  fails=$((fails + 1))
else
cargo "+${NIGHTLY}" miri test -p edgefirst-tensor --test scenarios \
  -- --ignored unwrap_then_use_aliases_the_same_tensor >"${diag_log}" 2>&1
diag_rc=$?
if [ "${diag_rc}" -ne 0 ] \
  && grep -q 'trying to retag from' "${diag_log}" \
  && grep -q "scenarios.rs:${diag_line_a}" "${diag_log}" \
  && grep -q "scenarios.rs:${diag_line_b}" "${diag_log}" \
  && grep -q "scenarios.rs:${diag_line_read}" "${diag_log}"; then
  echo "  PASS  diagnostic still demonstrates the retag-invalidation hazard"
elif [ "${diag_rc}" -ne 0 ]; then
  # Failed, but not with the signature this diagnostic exists to produce --
  # could be a genuinely different bug, a Miri internal error, a build
  # failure, or a resource death (Miri is slow and memory-hungry; an OOM
  # kill looks like any other nonzero exit here). Whatever it is, it is not
  # confirmed to be the aliasing hazard, so it is reported as its own,
  # different result rather than folded into "the disagreement still
  # holds" -- see ${diag_log}.
  echo "  FAIL  diagnostic failed, but NOT with the expected retag-invalidation signature -- see ${diag_log}"
  fails=$((fails + 1))
else
  # The diagnostic PASSED under Stacked Borrows -- Miri no longer rejects
  # the unsound shape it is built to demonstrate. This is not a defect in
  # this codebase (nothing here changed); it is news about Miri's own
  # model. `cannot_measure` would be wrong -- a measurement was taken, and
  # it came back with an answer. A plain FAIL would read as "the code
  # regressed", which is not what happened either. So: a distinct, loud,
  # unambiguous message, counted toward this script's exit status (so it
  # is not silently swallowed by CI) but never described the same way as
  # an aliasing-model failure above.
  echo "  MODEL CHANGED  the aliasing-model disagreement no longer holds: the diagnostic that used to fail under Stacked Borrows now passes. This is not a code regression -- Miri's own behavior on this shape changed. Re-derive crates/tensor/tests/scenarios.rs's diagnostic and task-11-report.md's account of the disagreement -- see ${diag_log}"
  fails=$((fails + 1))
fi
rm -f "${diag_log}"
fi

echo
if [ "${fails}" -eq 0 ]; then
  echo "G7 PASS: scenarios.rs is clean under both aliasing models, and the diagnostic still demonstrates the hazard it exists to record"
  exit 0
fi
echo "G7 FAIL: ${fails} check(s) failed (see above for which)"
exit 1
