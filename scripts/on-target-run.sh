#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
# SPDX-License-Identifier: Apache-2.0
#
# Run HAL's full test suite on the board this runs on, from inside a tests
# bundle built by scripts/on-target-bundle.sh. scripts/on-target-test.sh runs
# it over ssh; the hardware-test workflow runs it on a self-hosted runner.
#
# Usage (from anywhere; the bundle is this script's parent directory):
#   <bundle>/scripts/on-target-run.sh <results-dir>
#
# Env:
#   FILTER                      test-name filter passed to every binary. It
#                               matches the test FUNCTION name; a filter that
#                               matches nothing makes every binary run zero
#                               tests, reported as NO-TESTS.
#   HAL_BOARD_ENTRY             the board entry this run is for, recorded in
#                               summary.json for the fleet roll-up
#   HAL_ONTARGET_REQUIRE_GPU    1 to fail, as NO-GPU, on a board without a
#                               DRM render node, where every GPU test would
#                               otherwise skip and the run would pass
#
# Gates are armed from what the board has, as the CI board lane arms them:
# HAL_TEST_REQUIRE_GL with a render node, HAL_TEST_REQUIRE_G2D with libg2d.so.2
# (HAL_TEST_SKIP_G2D instead on i.MX 95), EDGEFIRST_SKIP_VIVANTE_KNOWN_BUGS with
# /dev/galcore, and HAL_TEST_REQUIRE_DMA when dma-heap-setup.sh's probe
# allocation succeeds.
#
# Writes one log per binary, check-single-home.log, test_two_library_user.log,
# test_codec_api.log, python-codec.log,
# capabilities.txt, summary.txt (`RESULT|detail`, one line) and summary.json
# into <results-dir>. Exits 0 on PASS, 1 on FAIL, NO-TESTS or NO-GPU.
#
# The board needs bash and the POSIX tools (awk, grep, sed, sort) only.
# python3 runs the Python codec tests and nothing else, so a board without it
# skips those and still reports its result.

set -uo pipefail

BUNDLE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RESULTS="${1:?usage: $0 <results-dir>}"
FILTER="${FILTER:-}"
REQUIRE_GPU="${HAL_ONTARGET_REQUIRE_GPU:-0}"

rm -rf "${RESULTS}"
mkdir -p "${RESULTS}"
RESULTS="$(cd "${RESULTS}" && pwd)"
cd "${BUNDLE}"

arch="$(uname -m)"
heaps="$(ls /dev/dma_heap 2>/dev/null | tr '\n' ',')"
render="$(ls /dev/dri 2>/dev/null | grep -c render)"
galcore="$([[ -e /dev/galcore ]] && echo yes || echo no)"
neutron="$([[ -e /dev/neutron0 ]] && echo yes || echo no)"
caps="dma_heap=${heaps:-none} render=${render} galcore=${galcore} neutron=${neutron}"
echo "${caps}" > "${RESULTS}/capabilities.txt"
echo "    ${caps}"

extra_env=()
# The Vivante driver has an intermittent double-free that otherwise reads as a
# regression in whatever you just changed.
[[ "${galcore}" == yes ]] && extra_env+=(EDGEFIRST_SKIP_VIVANTE_KNOWN_BUGS=1)
# A board with a DRM render node has a GPU, so "the GL backend did not come
# up" is a defect there and not a fact of the machine. Without this every
# require-gated test -- the `gl_backend_available_canary` most visibly --
# skips on every board and the run stays green through a broken GL stack.
[[ "${render}" -gt 0 ]] && extra_env+=(HAL_TEST_REQUIRE_GL=1)
# A board that ships libg2d must be able to use it, except an i.MX 95: its
# libg2d loads, but G2D is not supported there, so its tests skip.
if [[ "$(cat /sys/devices/soc0/soc_id 2>/dev/null)" == "i.MX95" ]]; then
  extra_env+=(HAL_TEST_SKIP_G2D=1)
elif compgen -G "/usr/lib/libg2d.so.2*" >/dev/null || compgen -G "/usr/lib/*/libg2d.so.2*" >/dev/null; then
  extra_env+=(HAL_TEST_REQUIRE_G2D=1)
fi

# dma-heap-setup.sh exports through $GITHUB_ENV when it is set, so point it at
# a file of our own and read the gate back. It never changes permissions on a
# self-hosted machine and never fails.
dma_env="$(mktemp)"
GITHUB_ENV="${dma_env}" GITHUB_STEP_SUMMARY="" bash scripts/dma-heap-setup.sh \
  > "${RESULTS}/dma-heap-setup.log" 2>&1
dma_state="$(sed -n 's/^HAL_CI_DMA_STATE=//p' "${dma_env}" | tail -n 1)"
grep -q '^HAL_TEST_REQUIRE_DMA=1$' "${dma_env}" && extra_env+=(HAL_TEST_REQUIRE_DMA=1)
rm -f "${dma_env}"
echo "    gates: ${extra_env[*]:-none} (dma ${dma_state:-unknown})"

# summary.json is written with awk, not an interpreter: the Python tests are
# optional on a board, so the result the fleet roll-up reads must not depend
# on python3.
write_summary() {
  local result="$1" detail="$2"
  echo "${result}|${detail}" > "${RESULTS}/summary.txt"
  local logs=("${RESULTS}"/*.log)
  [[ -e "${logs[0]}" ]] || logs=()
  # The values reach awk through the environment; `-v` would interpret
  # backslash escapes in them.
  SUMMARY_RESULT="${result}" SUMMARY_DETAIL="${detail}" SUMMARY_ARCH="${arch}" \
    SUMMARY_CAPS="${caps}" SUMMARY_DMA="${dma_state:-unknown}" \
    SUMMARY_HOST="$(uname -n)" SUMMARY_MANIFEST="${BUNDLE}/MANIFEST" \
    awk '
      function json(s,    out, i, c) {
        out = ""
        for (i = 1; i <= length(s); i++) {
          c = substr(s, i, 1)
          if (c == "\\" || c == "\"") out = out "\\" c
          else if (c in control) out = out control[c]
          else out = out c
        }
        return "\"" out "\""
      }
      BEGIN {
        for (i = 1; i < 32; i++) control[sprintf("%c", i)] = sprintf("\\u%04x", i)
        manifest = ENVIRON["SUMMARY_MANIFEST"]
        while ((getline line < manifest) > 0) {
          if (line == "") continue
          eq = index(line, "=")
          key = eq ? substr(line, 1, eq - 1) : line
          if (!(key in bundle)) keys[++nkeys] = key
          bundle[key] = eq ? substr(line, eq + 1) : ""
        }
      }
      /^test result: [A-Za-z]+\. [0-9]+ passed; [0-9]+ failed; [0-9]+ ignored/ {
        passed += $4; failed += $6; ignored += $8
      }
      /SKIPPED/ { skipped++ }
      END {
        printf "{\n"
        printf "  \"result\": %s,\n", json(ENVIRON["SUMMARY_RESULT"])
        printf "  \"detail\": %s,\n", json(ENVIRON["SUMMARY_DETAIL"])
        printf "  \"arch\": %s,\n", json(ENVIRON["SUMMARY_ARCH"])
        printf "  \"capabilities\": %s,\n", json(ENVIRON["SUMMARY_CAPS"])
        printf "  \"dma\": %s,\n", json(ENVIRON["SUMMARY_DMA"])
        printf "  \"host\": %s,\n", json(ENVIRON["SUMMARY_HOST"])
        printf "  \"entry\": %s,\n", json(ENVIRON["HAL_BOARD_ENTRY"])
        printf "  \"runner\": %s,\n", json(ENVIRON["RUNNER_NAME"])
        printf "  \"tests\": {\"passed\": %d, \"failed\": %d, \"ignored\": %d, \"skipped\": %d},\n",
          passed, failed, ignored, skipped
        printf "  \"bundle\": {"
        for (k = 1; k <= nkeys; k++)
          printf "%s\n    %s: %s", (k > 1 ? "," : ""), json(keys[k]), json(bundle[keys[k]])
        printf "%s}\n}\n", (nkeys ? "\n  " : "")
      }
    ' /dev/null ${logs[@]+"${logs[@]}"} > "${RESULTS}/summary.json"
  return 0
}

if [[ "${REQUIRE_GPU}" == "1" && "${render}" -eq 0 ]]; then
  echo "     NO-GPU: no DRM render node, so every GPU test would skip"
  write_summary NO-GPU "no DRM render node (/dev/dri/renderD*); GPU tests cannot run"
  exit 1
fi

bins=()
for bin in bin/*; do
  [[ -f "${bin}" && -x "${bin}" ]] && bins+=("${bin}")
done
if [[ ${#bins[@]} -eq 0 ]]; then
  write_summary FAIL "bundle has no test binaries"
  exit 1
fi

pass=0; fail=0; failed_bins=(); ran_nothing=0
for bin in "${bins[@]}"; do
  name="$(basename "${bin}")"
  echo "  -- ${name}"
  log="${RESULTS}/${name}.log"
  # --test-threads=1 is a hard invariant on target: GL driver concurrency
  # bugs, per-process G2D state, and CMA pool exhaustion each require it.
  # shellcheck disable=SC2086  # FILTER is deliberately word-split, as before
  if env EDGEFIRST_TESTDATA_DIR="${BUNDLE}/testdata" ${extra_env[@]+"${extra_env[@]}"} \
      "./${bin}" --test-threads=1 ${FILTER} > "${log}" 2>&1; then
    pass=$((pass + 1))
    # A hardware-gated test that returns early still reports "ok", so the
    # count alone cannot tell you whether the path ran.
    if grep -q "SKIPPED" "${log}"; then
      echo "     ok, but $(grep -c "SKIPPED" "${log}") skipped"
    fi
    # A binary that ran no test also exits 0, and the skip check above cannot
    # see it: there was no test to skip.
    if grep -q "^running 0 tests" "${log}"; then
      ran_nothing=$((ran_nothing + 1))
      echo "     ok, but ran 0 tests${FILTER:+ (FILTER=\"${FILTER}\" matched no test name)}"
    fi
  else
    fail=$((fail + 1)); failed_bins+=("${name}")
    echo "     FAILED (see ${log})"
    grep -E "^(test .* FAILED|failures:|error)" "${log}" | head -8 | sed 's/^/     /'
  fi
done

# G1/G2/G2t/G4/G9/G11 via check-single-home.sh. G5 (footprint) and G7 (Miri) read
# `cannot_measure` on a board: no target/release build and no scripts/miri.sh
# are bundled, on purpose. That is an attributable gap for those two gates
# only, so it is not a failure; a FAIL on any of G1/G1b/G2/G2t/G4/G9/G11 is.
echo "  -- check-single-home.sh (G1/G2/G2t/G4/G9/G11)"
ghlog="${RESULTS}/check-single-home.log"
./scripts/check-single-home.sh > "${ghlog}" 2>&1
gh_real_fail="$(grep -cE '^\s*(G1|G1b|G2|G2t|G4|G9|G11)\s+FAIL' "${ghlog}")"
gh_expected_gap="$(grep -cE '^\s*(G5|G7)\s+FAIL' "${ghlog}")"
if [[ "${gh_real_fail}" -eq 0 ]]; then
  pass=$((pass + 1))
  echo "     ok -- G1/G1b/G2/G2t/G4/G9/G11 clean; ${gh_expected_gap} expected gap(s) (G5/G7 need the build host)"
else
  fail=$((fail + 1)); failed_bins+=("check-single-home.sh")
  echo "     FAILED: ${gh_real_fail} real gate failure(s)"
  grep -E 'FAIL' "${ghlog}" | sed 's/^/     /'
fi

# G3: the minimal two-library user, the binary `make test-two-library-user`
# runs locally.
echo "  -- test_two_library_user (G3)"
g3log="${RESULTS}/test_two_library_user.log"
if env ${extra_env[@]+"${extra_env[@]}"} ./target/debug/test_two_library_user \
    > "${g3log}" 2>&1; then
  pass=$((pass + 1))
else
  fail=$((fail + 1)); failed_bins+=("test_two_library_user")
  echo "     FAILED"
  sed 's/^/     /' "${g3log}"
fi

# The codec C API end to end on this board's engines. A part the board
# cannot run prints "SKIP: <reason>", rewritten into the WARNING form the
# skip count and warnings.txt collect.
echo "  -- test_codec_api"
codeclog="${RESULTS}/test_codec_api.log"
env ${extra_env[@]+"${extra_env[@]}"} ./target/debug/test_codec_api 2>&1 \
  | sed 's/^SKIP: /WARNING: SKIPPED: test_codec_api - /' > "${codeclog}"
if [[ "${PIPESTATUS[0]}" -eq 0 ]]; then
  pass=$((pass + 1))
  grep -q "SKIPPED" "${codeclog}" && echo "     ok, but $(grep -c "SKIPPED" "${codeclog}") skipped"
else
  fail=$((fail + 1)); failed_bins+=("test_codec_api")
  echo "     FAILED"
  sed 's/^/     /' "${codeclog}"
fi

# The Python codec tests, with the bundle's own wheels and pytest on
# PYTHONPATH. They need a python3 of 3.11 or newer with numpy; a board
# without one, or a bundle built without wheels, skips them with the reason.
# pytest reports the skips that share a location and reason on one
# `SKIPPED [N]` line. The skip count and warnings.txt count lines, so that
# line is written N times.
echo "  -- Python codec tests"
pylog="${RESULTS}/python-codec.log"
py_skip=""
if [[ ! -d python ]]; then
  py_skip="the bundle has no wheels (built without PYTHON_WHEELS)"
elif ! python3 -c 'import sys, numpy; sys.exit(sys.version_info < (3, 11))' > /dev/null 2>&1; then
  py_skip="no python3 3.11+ with numpy on this board"
fi
if [[ -n "${py_skip}" ]]; then
  echo "WARNING: SKIPPED: Python codec tests - ${py_skip}" > "${pylog}"
  echo "     skipped: ${py_skip}"
else
  env PYTHONPATH="${BUNDLE}/python" ${extra_env[@]+"${extra_env[@]}"} \
    python3 -m pytest tests/codec -q -rs -p no:cacheprovider 2>&1 \
    | awk '
        match($0, /^SKIPPED \[[0-9]+\] /) {
          n = substr($0, 10, RLENGTH - 11) + 0
          for (i = 0; i < n; i++) print "WARNING: SKIPPED: " substr($0, RLENGTH + 1)
          next
        }
        { print }' > "${pylog}"
  if [[ "${PIPESTATUS[0]}" -eq 0 ]]; then
    pass=$((pass + 1))
    echo "     ok -- $(tail -n 1 "${pylog}")"
  else
    fail=$((fail + 1)); failed_bins+=("python-codec")
    echo "     FAILED (see ${pylog})"
    grep -E "^(FAILED|ERROR)|Error" "${pylog}" | head -8 | sed 's/^/     /'
  fi
fi

skipped="$(cat "${RESULTS}"/*.log 2>/dev/null | grep -c "SKIPPED")"
# Every distinct test warning, most frequent first, so skips are audited rather than only counted.
grep -h "WARNING: " "${RESULTS}"/*.log 2>/dev/null | sed 's/.*WARNING: //' | sort | uniq -c | sort -rn > "${RESULTS}/warnings.txt" || true
if [[ -s "${RESULTS}/warnings.txt" ]]; then
  echo "  -- test warnings"
  sed 's/^/     /' "${RESULTS}/warnings.txt"
fi
if [[ ${fail} -gt 0 ]]; then
  write_summary FAIL "${pass} ok, ${fail} failed: ${failed_bins[*]}"
  exit 1
elif [[ ${ran_nothing} -eq ${#bins[@]} ]]; then
  # Every binary ran zero tests, so this board exercised nothing.
  write_summary NO-TESTS "all ${ran_nothing} binaries ran 0 tests${FILTER:+ -- FILTER=\"${FILTER}\" matched no test name}"
  exit 1
fi
detail="${pass} binaries, ${skipped} tests skipped"
[[ ${ran_nothing} -gt 0 ]] && detail="${detail}, ${ran_nothing} ran 0 tests"
write_summary PASS "${detail}, G1/G2/G2t/G3/G4/G9/G11 clean"
exit 0
