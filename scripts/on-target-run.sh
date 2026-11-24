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
#                               matches the test function name; a filter that
#                               matches nothing makes every binary run zero
#                               tests, reported as NO-TESTS.
#   HAL_BOARD_ENTRY             the board entry this run is for, recorded in
#                               summary.json for the fleet roll-up
#   HAL_ONTARGET_REQUIRE_GPU    1 to fail, as NO-GPU, on a board without a
#                               DRM render node
#   HAL_ONTARGET_TEST_THREADS   test threads per binary; unset uses libtest's
#                               default, one per CPU
#   HAL_ONTARGET_GDB            1 to run each test binary under `gdb -batch`,
#                               which prints every thread's backtrace into the
#                               binary's log if the binary is killed by a signal
#
# Sets the test gates from what the board has: HAL_TEST_REQUIRE_GL with a
# render node, HAL_TEST_REQUIRE_G2D with libg2d.so.2 (HAL_TEST_SKIP_G2D
# instead on i.MX 95), EDGEFIRST_SKIP_VIVANTE_KNOWN_BUGS with /dev/galcore,
# and HAL_TEST_REQUIRE_DMA when dma-heap-setup.sh's probe allocation
# succeeds.
#
# Writes one log per binary, check-single-home.log,
# test_two_library_user.log, test_codec_api.log, python-codec.log,
# capabilities.txt, diagnostics.txt (core pattern, core size limit, gdb,
# coredumpctl), summary.txt (`RESULT|detail`, one line) and summary.json into
# <results-dir>. A test binary killed by a signal also gets
# <binary>.crash.txt, with the signal and whatever backtrace coredumpctl or a
# core file and gdb can give. Exits 0 on PASS, 1 on FAIL, NO-TESTS or NO-GPU.
#
# Needs bash and the POSIX tools (awk, grep, sed, sort). python3 is used for
# the Python codec tests only.

set -uo pipefail

BUNDLE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RESULTS="${1:?usage: $0 <results-dir>}"
FILTER="${FILTER:-}"
REQUIRE_GPU="${HAL_ONTARGET_REQUIRE_GPU:-0}"
THREADS_ARG=()
[[ -n "${HAL_ONTARGET_TEST_THREADS:-}" ]] && THREADS_ARG=(--test-threads="${HAL_ONTARGET_TEST_THREADS}")

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

# Lets test binaries dump core, and records what the board offers for crash
# backtraces.
ulimit -c unlimited 2>/dev/null || true
have_gdb="$(command -v gdb >/dev/null 2>&1 && echo yes || echo no)"
have_coredumpctl="$(command -v coredumpctl >/dev/null 2>&1 && echo yes || echo no)"
diag="core_pattern=$(cat /proc/sys/kernel/core_pattern 2>/dev/null || echo unknown) core_limit=$(ulimit -c) gdb=${have_gdb} coredumpctl=${have_coredumpctl}"
echo "${diag}" > "${RESULTS}/diagnostics.txt"
echo "    ${diag}"
DEBUGGER=()
if [[ "${HAL_ONTARGET_GDB:-0}" == "1" ]]; then
  if [[ "${have_gdb}" == yes ]]; then
    DEBUGGER=(gdb -q -batch -return-child-result -ex run -ex "thread apply all bt" --args)
  else
    echo "    HAL_ONTARGET_GDB=1 but the board has no gdb"
  fi
fi

# Writes <results>/<name>.crash.txt for a test binary `bin` (pid `pid`) that
# exited with status `rc` > 128.
collect_crash() {
  local bin="$1" pid="$2" rc="$3"
  local name out core
  name="$(basename "${bin}")"
  out="${RESULTS}/${name}.crash.txt"
  echo "${name} (pid ${pid}) was killed by signal $((rc - 128))" > "${out}"
  if [[ "${have_coredumpctl}" == yes ]]; then
    # Retries for up to 10 s while systemd-coredump records the dump.
    for _ in 1 2 3 4 5 6 7 8 9 10; do
      if coredumpctl --no-pager info "${pid}" >> "${out}" 2>&1; then
        break
      fi
      sleep 1
    done
  fi
  for core in "core.${pid}" core; do
    [[ -f "${core}" ]] || continue
    if [[ "${have_gdb}" == yes ]]; then
      gdb -q -batch -ex "thread apply all bt" "${bin}" "${core}" >> "${out}" 2>&1
    else
      echo "core file ${core} left on the board; no gdb to read it" >> "${out}"
    fi
    rm -f "${core}"
    break
  done
}

extra_env=()
[[ "${galcore}" == yes ]] && extra_env+=(EDGEFIRST_SKIP_VIVANTE_KNOWN_BUGS=1)
[[ "${render}" -gt 0 ]] && extra_env+=(HAL_TEST_REQUIRE_GL=1)
if [[ "$(cat /sys/devices/soc0/soc_id 2>/dev/null)" == "i.MX95" ]]; then
  extra_env+=(HAL_TEST_SKIP_G2D=1)
elif compgen -G "/usr/lib/libg2d.so.2*" >/dev/null || compgen -G "/usr/lib/*/libg2d.so.2*" >/dev/null; then
  extra_env+=(HAL_TEST_REQUIRE_G2D=1)
fi

# Runs dma-heap-setup.sh with GITHUB_ENV pointed at a temporary file and
# reads the DMA gate back from it.
dma_env="$(mktemp)"
GITHUB_ENV="${dma_env}" GITHUB_STEP_SUMMARY="" bash scripts/dma-heap-setup.sh \
  > "${RESULTS}/dma-heap-setup.log" 2>&1
dma_state="$(sed -n 's/^HAL_CI_DMA_STATE=//p' "${dma_env}" | tail -n 1)"
grep -q '^HAL_TEST_REQUIRE_DMA=1$' "${dma_env}" && extra_env+=(HAL_TEST_REQUIRE_DMA=1)
rm -f "${dma_env}"
echo "    gates: ${extra_env[*]:-none} (dma ${dma_state:-unknown})"

# Writes summary.txt and summary.json.
write_summary() {
  local result="$1" detail="$2"
  echo "${result}|${detail}" > "${RESULTS}/summary.txt"
  local logs=("${RESULTS}"/*.log)
  [[ -e "${logs[0]}" ]] || logs=()
  # Passes the values to awk through the environment.
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
  # shellcheck disable=SC2086
  env EDGEFIRST_TESTDATA_DIR="${BUNDLE}/testdata" ${extra_env[@]+"${extra_env[@]}"} \
    ${DEBUGGER[@]+"${DEBUGGER[@]}"} "./${bin}" ${THREADS_ARG[@]+"${THREADS_ARG[@]}"} ${FILTER} \
    > "${log}" 2>&1 &
  pid=$!
  wait "${pid}"
  rc=$?
  if [[ ${rc} -eq 0 ]]; then
    pass=$((pass + 1))
    # Reports the skips of a binary that passed.
    if grep -q "SKIPPED" "${log}"; then
      echo "     ok, but $(grep -c "SKIPPED" "${log}") skipped"
    fi
    # Counts a binary that ran no test.
    if grep -q "^running 0 tests" "${log}"; then
      ran_nothing=$((ran_nothing + 1))
      echo "     ok, but ran 0 tests${FILTER:+ (FILTER=\"${FILTER}\" matched no test name)}"
    fi
  else
    fail=$((fail + 1)); failed_bins+=("${name}")
    echo "     FAILED (see ${log})"
    grep -E "^(test .* FAILED|failures:|error)" "${log}" | head -8 | sed 's/^/     /'
    if [[ ${rc} -gt 128 ]]; then
      collect_crash "${bin}" "${pid}" "${rc}"
      echo "     killed by signal $((rc - 128)); see ${RESULTS}/${name}.crash.txt"
    fi
  fi
done

# Runs the checks of check-single-home.sh that apply to a deployed bundle.
echo "  -- check-single-home.sh"
single_home_log="${RESULTS}/check-single-home.log"
if ./scripts/check-single-home.sh --on-target > "${single_home_log}" 2>&1; then
  pass=$((pass + 1))
else
  fail=$((fail + 1)); failed_bins+=("check-single-home.sh")
  echo "     FAILED"
  grep -E 'FAIL' "${single_home_log}" | sed 's/^/     /'
fi

# Runs the two-library C program.
echo "  -- test_two_library_user"
two_lib_log="${RESULTS}/test_two_library_user.log"
if env ${extra_env[@]+"${extra_env[@]}"} ./target/debug/test_two_library_user \
    > "${two_lib_log}" 2>&1; then
  pass=$((pass + 1))
else
  fail=$((fail + 1)); failed_bins+=("test_two_library_user")
  echo "     FAILED"
  sed 's/^/     /' "${two_lib_log}"
fi

# Runs test_codec_api and rewrites its `SKIP: <reason>` lines as
# `WARNING: SKIPPED:` lines.
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

# Runs the Python codec tests with the bundle's wheels and pytest on
# PYTHONPATH. Skips them, with the reason, on a board without a python3 of
# 3.11 or newer with numpy and in a bundle built without wheels. Writes
# each of pytest's `SKIPPED [N]` lines N times as `WARNING: SKIPPED:` lines.
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
# Writes every distinct test warning to warnings.txt, most frequent first.
grep -h "WARNING: " "${RESULTS}"/*.log 2>/dev/null | sed 's/.*WARNING: //' | sort | uniq -c | sort -rn > "${RESULTS}/warnings.txt" || true
if [[ -s "${RESULTS}/warnings.txt" ]]; then
  echo "  -- test warnings"
  sed 's/^/     /' "${RESULTS}/warnings.txt"
fi
if [[ ${fail} -gt 0 ]]; then
  write_summary FAIL "${pass} ok, ${fail} failed: ${failed_bins[*]}"
  exit 1
elif [[ ${ran_nothing} -eq ${#bins[@]} ]]; then
  write_summary NO-TESTS "all ${ran_nothing} binaries ran 0 tests${FILTER:+ -- FILTER=\"${FILTER}\" matched no test name}"
  exit 1
fi
detail="${pass} binaries, ${skipped} tests skipped"
[[ ${ran_nothing} -gt 0 ]] && detail="${detail}, ${ran_nothing} ran 0 tests"
write_summary PASS "${detail}"
exit 0
