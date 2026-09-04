#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
# SPDX-License-Identifier: Apache-2.0
#
# The Definition of Done for the single-tensor-home plan, as commands.
#
# Every gate here is RED as of 2026-08-24 and can only go green if the
# requirement is met. Nothing in this file measures a proxy. In particular
# there is deliberately NO pairwise-nm-no-shared-symbol check: that gate went
# green on the PREVIOUS, WRONG architecture, because per-crate vtable copies
# were designed to avoid symbol collision while the code was duplicated.
#
# The rule that shapes every gate below: INABILITY TO MEASURE IS NOT A PASS.
# A missing library, a missing header, an unexpanded glob, or a stripped
# binary with no symbol table must fail loudly (cannot_measure), never
# silently satisfy a "want 0" or "want >= N" comparison by accident. This is
# the same class of defect a proxy gate is -- it lets "done" go green without
# the requirement being met -- so it gets the same zero tolerance.
#
# The precise rule: ASSERT A POSITIVE MEASUREMENT WAS OBTAINED, never assert
# an input merely exists. `[ -f "$x" ]` answers "is there a file"; it says
# nothing about whether reading it produced anything. A 0-byte header passes
# `[ -f ]`, greps zero declarations, and a naive gate reads that as "0 bad
# entries, want 0, PASS" -- exactly backwards. Where a file's byte content is
# what gets measured, the precondition is `[ -s "$f" ] && [ -r "$f" ]`
# (exists, non-empty, readable).
#
# `-s`/`-r` is necessary and NOT sufficient. It proves a file has bytes and
# can be read; it proves nothing about whether those bytes are the file the
# gate thinks it is measuring. A comment-only, truncated, or wrongly
# generated header is non-empty and readable, and every "want 0" count taken
# from it comes back 0 -- simultaneously "genuinely correct" and "measured
# nothing", with nothing in the output to tell the two apart.
#
# So wherever a gate's TARGET value is legitimately 0, the count it gates on
# can never double as proof that the measurement ran, and a separate LIVENESS
# signal is required: an independent count that any real input makes >= 1,
# whether the gate passes or fails. G9 and G11 both read headers whose gated
# count is driven to 0 by this plan, and both use the same liveness signal --
# the total number of `ef_*` function declarations the header yields
# (`header_decls` below). A real header always declares at least one, no
# matter how many of them mention detections; 0 can only mean the file is not
# a header this script can parse. The liveness count is checked separately
# from, and never conflated with, the value the gate compares to its target.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "${ROOT}"

# Two profiles, not one, because the two things this script measures come
# from two different builds. `make capi-libs` (a prerequisite of this
# script's Makefile target) only ever produces target/debug, and debug is
# never stripped, so it is what G1 needs to reliably see embedded symbols.
# G5's BASELINE_BYTES is a release measurement, so a debug comparison would
# be meaningless -- it needs target/release, which `make capi-libs-release`
# provides. (This split also sidesteps a latent problem: the hygiene track
# plans to add `strip = true` to the leaves' own [profile.release], which
# would blind G1 forever if G1 read release.)
#
# PROFILE, when set, overrides BOTH directories to the same target/<PROFILE>
# -- this is what lets `PROFILE=nonexistent ./scripts/check-single-home.sh`
# exercise every gate's cannot-measure path at once for verification.
PROFILE="${PROFILE:-}"
if [[ -n "${PROFILE}" ]]; then
  DEBUG_LIBDIR="target/${PROFILE}"
  RELEASE_LIBDIR="target/${PROFILE}"
else
  DEBUG_LIBDIR="target/debug"
  RELEASE_LIBDIR="target/release"
fi

SIBS="image codec decoder tracker"
# The five modular Python extensions, by their cdylib stem (`[lib] name` in
# each crates/python-*/Cargo.toml, so `_tensor` -> `lib_tensor.so`).
PY_EXTS="tensor image codec decoder tracker"

# Hash a shared object's LOADABLE content -- everything the dynamic linker
# maps and the CPU executes -- with debug info and the build-id note removed.
#
# Whole-file `sha256sum` is the right measure for the four C siblings, which
# G6 never rebuilds. It is the WRONG measure for the four Python extensions,
# which G6 *does* rebuild: they legitimately compile `edgefirst-tensor`'s
# dynamic backend, so inserting a field into `crates/tensor/src/lib.rs`
# shifts every line after it and DWARF's line tables move with them. That is
# a debug-info change, not an implementation leaking in, and a whole-file
# hash cannot tell the two apart.
#
# Established by enumerating every section of `lib_tensor.so` across the
# probe: exactly three differ -- `.debug_info`, `.debug_line`, and
# `.note.gnu.build-id`. `.text`, `.rodata`, `.data`, `.data.rel.ro`,
# `.rela.dyn`, `.dynsym` and the rest are byte-identical. The build-id is
# excluded because it is by construction a digest of the build inputs, so it
# moves whenever *anything* does -- including the debug info this measure
# deliberately ignores.
g6_code_hash() { # g6_code_hash <shared object> -> sha256 of its loadable content
  local so="$1"
  local tmp
  tmp=$(mktemp) || return 1
  if objcopy --strip-debug --remove-section=.note.gnu.build-id "${so}" "${tmp}" 2>/dev/null; then
    sha256sum <"${tmp}"
  fi
  rm -f "${tmp}"
}
fails=0
gate() { # gate <id> <description> <0|1 pass>
  local id="$1" desc="$2" status="$3"
  if [[ "${status}" -eq 0 ]]; then printf '  %-4s PASS  %s\n' "${id}" "${desc}"
  else printf '  %-4s FAIL  %s\n' "${id}" "${desc}"; fails=$((fails+1)); fi
  return 0
}
cannot_measure() { # cannot_measure <gate> <reason>
  local id="$1" reason="$2"
  printf '  %-4s FAIL  cannot verify: %s\n' "${id}" "${reason}"; fails=$((fails+1))
  return 0
}
header_decls() { # header_decls <header>: print the ef_* function names it declares
  local header="$1"
  # Drop doc-comment lines (a `@brief`/prose line may legitimately mention
  # another function by name, e.g. tensor.h's own header comment cites
  # ef_image_processor_create_image()) and typedef lines (struct/enum type
  # definitions, whose tag -- e.g. `ef_tensor_builder` in `typedef struct
  # ef_tensor_builder ef_tensor_builder;` -- is not a function and is never
  # exported). What is left is real declaration lines, including ones whose
  # return type is itself `struct ef_x *` (e.g. `struct ef_tensor_builder
  # *ef_tensor_builder_new(void);`), so the token extracted must be the one
  # immediately followed by `(` -- the function name -- not any earlier
  # ef_-prefixed type name on the same line.
  #
  # Shared by G9 (as its liveness signal) and G11 (as the set it checks
  # against the library's exports) so the two can never disagree about what
  # counts as a declaration.
  grep -vE '^\s*(\*|/\*|//)' "${header}" \
    | grep -v '^typedef' \
    | grep -oE 'ef_[a-z0-9_]+\(' \
    | grep -oE 'ef_[a-z0-9_]+' \
    | sort -u
  return 0
}

# --drift: G6, the drift test. Off by default -- unlike every other gate
# here it REBUILDS `edgefirst-tensor` twice and patches a source file, so it
# is opt-in (`./scripts/check-single-home.sh --drift`) rather than part of
# the default sweep. `--drift-only` runs G6 alone and skips the rest (useful
# when only this gate is being iterated on).
#
# Different evidence for the same claim G1 makes, not a restatement of it:
# G1 inspects a build that already happened and finds no `static_backend`
# symbol; G6 causes a change in `edgefirst-tensor`'s own private layout,
# rebuilds ONLY `libedgefirst_tensor.so`, and confirms the four siblings'
# bytes cannot move -- because nothing recompiles them. G1 is a point-in-time
# inspection; G6 is a causal test. Task 12's own report has the full
# rationale and a manually-verified run of exactly this procedure.
#
# NOT idempotent back-to-back: this gate reverts the SOURCE after each run,
# but never rebuilds target/debug/libedgefirst_tensor.so back to a
# pre-probe state -- that .so is left reflecting the probed source. Run
# `--drift` (or `--drift-only`) a second time without an intervening `make
# capi-libs` in between, and the gate re-inserts the identical probe text,
# finds a target/debug that already matches it byte for byte, and `cargo
# build` correctly does nothing (no recompile needed) -- so
# libedgefirst_tensor.so's hash cannot move and this reads `cannot_measure`,
# not a repeat PASS. That is the gate working correctly on a stale target,
# not a bug in the gate; `make capi-libs` between runs avoids it.
DRIFT_MODE=0
DRIFT_ONLY=0
DIFFERENTIAL_MODE=0
DIFFERENTIAL_ONLY=0
for _arg in "$@"; do
  case "${_arg}" in
    --drift) DRIFT_MODE=1 ;;
    --drift-only) DRIFT_MODE=1; DRIFT_ONLY=1 ;;
    --differential) DIFFERENTIAL_MODE=1 ;;
    --differential-only) DIFFERENTIAL_MODE=1; DIFFERENTIAL_ONLY=1 ;;
    *) echo "FAIL: unknown argument ${_arg}" >&2; exit 1 ;;
  esac
done

if [[ "${DRIFT_MODE}" -eq 1 ]]; then
  echo "== G6: siblings do not drift when edgefirst-tensor's own layout changes =="
  g6_lib_rs="crates/tensor/src/lib.rs"
  g6_patched=0
  # The probe is inserted and removed by exact, previously-verified anchor
  # strings (task 12's own manual run confirmed each one is unique and
  # round-trips cleanly), not a line number -- lines move; this text does
  # not, short of someone editing these exact fields.
  g6_revert() {
    if [[ "${g6_patched}" -eq 1 ]]; then
      # Check python3's own exit status before declaring the probe gone --
      # an interrupted or failed revert must not be reported as complete.
      # This is the one gate that mutates source, so a false "reverted" here
      # is the worst failure mode this script has: a probe field silently
      # left behind in a tree someone is about to commit from.
      if python3 - "${g6_lib_rs}" <<'PY'
import sys
path = sys.argv[1]
with open(path) as f:
    src = f.read()

# Same assert-then-replace discipline the patch side uses (see below): a
# `.replace(..., 1)` that matches nothing is not an error to plain Python,
# it just returns the string unchanged -- and unlike the patch side, a
# revert that silently no-ops has no downstream safety net. A missed
# insertion on the patch side fails to compile (a struct literal missing a
# newly-required field is a hard Rust error), so `cargo build` catches it
# for free. A missed REMOVAL on the revert side leaves valid, compiling
# Rust with a leftover `_drift_probe` field sitting in the tree -- nothing
# forces that to surface. So every anchor here is asserted present first.

anchor = (
    "    view_origin: Option<ViewOrigin>,\n"
    "    /// G6 drift probe (single-tensor-home task 12): inserted and removed by\n"
    "    /// scripts/check-single-home.sh --drift. Must never survive a commit.\n"
    "    _drift_probe: u8,\n"
    "}\n"
)
assert src.count(anchor) == 1, "Tensor<T> struct's probe field anchor not found -- revert incomplete"
src = src.replace(anchor, "    view_origin: Option<ViewOrigin>,\n}\n", 1)

anchor = (
    "            cpu_access: CpuAccess::ReadWrite,\n"
    "            compression: None,\n"
    "            view_origin: None,\n"
    "            _drift_probe: 0,\n"
    "        }\n    }"
)
assert src.count(anchor) == 1, "wrap()'s probe-field initializer not found -- revert incomplete"
src = src.replace(
    anchor,
    "            cpu_access: CpuAccess::ReadWrite,\n"
    "            compression: None,\n"
    "            view_origin: None,\n"
    "        }\n    }",
    1,
)

anchor = (
    "            cpu_access: CpuAccess::ReadWrite,\n"
    "            compression: None,\n"
    "            view_origin: None,\n"
    "            _drift_probe: 0,\n"
    "        })\n    }"
)
assert src.count(anchor) == 1, "from_pbo()'s probe-field initializer not found -- revert incomplete"
src = src.replace(
    anchor,
    "            cpu_access: CpuAccess::ReadWrite,\n"
    "            compression: None,\n"
    "            view_origin: None,\n"
    "        })\n    }",
    1,
)

anchor = (
    "            compression: luma.compression,\n"
    "            // A composed multiplane tensor is a whole image, not a sub-view.\n"
    "            view_origin: None,\n"
    "            _drift_probe: 0,\n"
    "        })\n    }"
)
assert src.count(anchor) == 1, "from_planes()'s probe-field initializer not found -- revert incomplete"
src = src.replace(
    anchor,
    "            compression: luma.compression,\n"
    "            // A composed multiplane tensor is a whole image, not a sub-view.\n"
    "            view_origin: None,\n"
    "        })\n    }",
    1,
)

with open(path, "w") as f:
    f.write(src)
PY
      then
        g6_patched=0
      else
        echo "  G6   WARN  revert of ${g6_lib_rs} failed -- inspect it by hand for a leftover _drift_probe field before committing anything" >&2
      fi
    fi
    return 0
  }
  # Revert on ANY exit from this point on -- a failed build, an interrupted
  # run, or a bug in this script must not leave the probe field behind.
  trap g6_revert EXIT

  if [[ ! -s "${g6_lib_rs}" ]] || [[ ! -r "${g6_lib_rs}" ]]; then
    cannot_measure "G6" "${g6_lib_rs} is missing, empty, or unreadable"
  elif command -v git >/dev/null 2>&1 && ! git diff --quiet -- "${g6_lib_rs}" 2>/dev/null; then
    # Refuses to patch (and later revert) a file that already has someone
    # else's uncommitted edit in it -- the exact failure shape that put one
    # agent's work inside another's commit earlier on this branch. Silently
    # reverting over a real edit would be worse than refusing to run.
    cannot_measure "G6" "${g6_lib_rs} already has uncommitted changes -- refusing to patch and revert over them"
  elif [[ ! -s "${DEBUG_LIBDIR}/libedgefirst_image.so" ]]; then
    cannot_measure "G6" "${DEBUG_LIBDIR}/libedgefirst_image.so is missing -- run \`make capi-libs\` first"
  elif [[ ! -s "${DEBUG_LIBDIR}/lib_tensor.so" ]]; then
    cannot_measure "G6" "${DEBUG_LIBDIR}/lib_tensor.so is missing -- run \`cargo build -p edgefirst-python-tensor -p edgefirst-python-image -p edgefirst-python-codec -p edgefirst-python-decoder\` first"
  else
    # Hash edgefirst-tensor's own library too, not just the four siblings --
    # see the rebuild step below for why: a causal test has to prove the
    # cause fired, not just check for the absence of an effect.
    g6_tensor_before=$(sha256sum "${DEBUG_LIBDIR}/libedgefirst_tensor.so" 2>/dev/null)
    g6_before=$(for l in $SIBS; do sha256sum "${DEBUG_LIBDIR}/libedgefirst_${l}.so" 2>/dev/null; done)
    # The four Python extensions (task P4). Their baseline is BUILT here,
    # from the known-clean source, rather than read off disk.
    #
    # Reading it off disk is what the `.so` half of this gate already warns
    # about, one level up: a previous `--drift` run rebuilds the extensions
    # WITH the probe and then reverts only the source, so the artifacts left
    # behind belong to a tree that no longer exists. Hashing those as the
    # "before" compares a patched build against a patched build, or worse, a
    # patched one against a clean one -- and either way the number is not
    # about the probe. (Observed while writing this: the first red this gate
    # produced was exactly that, not a real leak.)
    #
    # The `git diff --quiet` guard above has already established the source
    # is clean at this point, so this build IS the clean baseline.
    g6_py_baseline_log="${TMPDIR:-/tmp}/check-single-home-g6-baseline.$$.log"
    if ! cargo build $(for l in $PY_EXTS; do printf -- "-p edgefirst-python-%s " "$l"; done) \
         >"${g6_py_baseline_log}" 2>&1; then
      cannot_measure "G6" "could not build the Python extensions' clean baseline -- see ${g6_py_baseline_log}"
      g6_py_before=""
    else
      g6_py_before=$(for l in $PY_EXTS; do g6_code_hash "${DEBUG_LIBDIR}/lib_${l}.so"; done)
    fi
    rm -f "${g6_py_baseline_log}"
    python3 - "${g6_lib_rs}" <<'PY'
import sys
path = sys.argv[1]
with open(path) as f:
    src = f.read()
assert src.count("    view_origin: Option<ViewOrigin>,\n}\n") == 1, "struct anchor not unique"
src = src.replace(
    "    view_origin: Option<ViewOrigin>,\n}\n",
    "    view_origin: Option<ViewOrigin>,\n"
    "    /// G6 drift probe (single-tensor-home task 12): inserted and removed by\n"
    "    /// scripts/check-single-home.sh --drift. Must never survive a commit.\n"
    "    _drift_probe: u8,\n"
    "}\n",
    1,
)
src = src.replace(
    "            cpu_access: CpuAccess::ReadWrite,\n"
    "            compression: None,\n"
    "            view_origin: None,\n"
    "        }\n    }",
    "            cpu_access: CpuAccess::ReadWrite,\n"
    "            compression: None,\n"
    "            view_origin: None,\n"
    "            _drift_probe: 0,\n"
    "        }\n    }",
    1,
)
src = src.replace(
    "            cpu_access: CpuAccess::ReadWrite,\n"
    "            compression: None,\n"
    "            view_origin: None,\n"
    "        })\n    }",
    "            cpu_access: CpuAccess::ReadWrite,\n"
    "            compression: None,\n"
    "            view_origin: None,\n"
    "            _drift_probe: 0,\n"
    "        })\n    }",
    1,
)
src = src.replace(
    "            compression: luma.compression,\n"
    "            // A composed multiplane tensor is a whole image, not a sub-view.\n"
    "            view_origin: None,\n"
    "        })\n    }",
    "            compression: luma.compression,\n"
    "            // A composed multiplane tensor is a whole image, not a sub-view.\n"
    "            view_origin: None,\n"
    "            _drift_probe: 0,\n"
    "        })\n    }",
    1,
)
with open(path, "w") as f:
    f.write(src)
PY
    if [[ $? -ne 0 ]]; then
      cannot_measure "G6" "could not locate Tensor<T>'s struct/constructors in ${g6_lib_rs} to patch (anchors moved -- update this script)"
    else
      g6_patched=1
      g6_build_log="${TMPDIR:-/tmp}/check-single-home-g6-build.$$.log"
      # Plain `cargo build`, not `--release`: DEBUG_LIBDIR (what both hash
      # steps read) is target/debug by default, the dev profile's output
      # directory. `--release` writes to target/release instead -- a
      # rebuild that lands somewhere neither hash step ever reads, so the
      # "before" and "after" hashes of the FOUR SIBLINGS would be identical
      # unconditionally, whether or not anything actually leaked into them.
      # (PROFILE, when set, points DEBUG_LIBDIR at target/<PROFILE> -- but
      # this build always writes target/debug regardless, since it never
      # reads PROFILE itself. A PROFILE whose libdir doesn't exist yet is
      # caught by the earlier `cannot_measure` branches, which return
      # before this line runs -- but PROFILE=release with an *existing*
      # release build passes those preconditions and reaches here. It
      # still fails safe: the hashes read target/release while this build
      # writes target/debug, so libedgefirst_tensor.so's hash can't move,
      # and the tensor-changed check below reports `cannot_measure` rather
      # than a false PASS. The tensor-changed check is what makes any
      # PROFILE override safe here, not an assumption that this line is
      # unreachable under one.)
      # Two rebuilds, and the second is the point of P4's half of this gate.
      #
      # The C siblings are NOT rebuilt here (this command builds tensor-capi
      # alone), so their byte-identity says "nothing asked them to change".
      # That is real evidence -- a monolithic build would have had to rebuild
      # them for the new layout -- but it is weaker than it looks.
      #
      # The four Python extensions ARE rebuilt, deliberately. They depend on
      # `edgefirst-tensor` as a Rust crate, so a source change to it
      # recompiles the crate for them; if the probe were reaching their code
      # their bytes would move. `_drift_probe` goes into `Tensor<T>`, which
      # is `#[cfg(feature = "static")]` -- the extensions build `dynamic`, so
      # they compile the changed file and must still emit identical bytes.
      # That is a causal test rather than an absence-of-effect one.
      if ! cargo build --manifest-path crates/tensor-capi/Cargo.toml --target-dir target >"${g6_build_log}" 2>&1; then
        cannot_measure "G6" "rebuilding libedgefirst_tensor.so with the probe field failed -- see ${g6_build_log}"
      # Skipped when the baseline never produced one: a second
      # `cannot_measure` for the same cause is noise, and the last line a
      # gate prints is the one that gets quoted.
      elif [[ -z "${g6_py_before}" ]]; then
        : # the baseline already reported why
      elif ! cargo build $(for l in $PY_EXTS; do printf -- "-p edgefirst-python-%s " "$l"; done) \
             >>"${g6_build_log}" 2>&1; then
        cannot_measure "G6" "rebuilding the four Python extensions with the probe field failed -- see ${g6_build_log}"
      else
        g6_tensor_after=$(sha256sum "${DEBUG_LIBDIR}/libedgefirst_tensor.so" 2>/dev/null)
        g6_after=$(for l in $SIBS; do sha256sum "${DEBUG_LIBDIR}/libedgefirst_${l}.so" 2>/dev/null; done)
        g6_py_after=$(for l in $PY_EXTS; do g6_code_hash "${DEBUG_LIBDIR}/lib_${l}.so"; done)
        # A causal test has to prove the cause fired. Byte-identical
        # siblings prove nothing on their own -- that is also what a
        # rebuild that touched nothing looks like. Check that first:
        # `_drift_probe` changes Tensor<T>'s own layout, so
        # libedgefirst_tensor.so itself MUST differ across this rebuild;
        # if it didn't, the rebuild that just ran is not evidence of
        # anything and this is `cannot_measure`, not a silent PASS.
        if [[ "${g6_tensor_before}" = "${g6_tensor_after}" ]]; then
          cannot_measure "G6" "libedgefirst_tensor.so did not change after the rebuild -- the probe never took effect, so byte-identical siblings would prove nothing (see ${g6_build_log})"
        elif [[ "${g6_before}" != "${g6_after}" ]]; then
          gate "G6" "a sibling's bytes CHANGED when only edgefirst-tensor's private layout changed -- see ${g6_build_log}" 1
        # The Python half needs its own liveness signal, and byte-identity
        # cannot be it: an extension that was never recompiled is also
        # byte-identical. `edgefirst-tensor` must appear as recompiled in the
        # build log, or this proves nothing -- the same reason the tensor
        # hash-moved check above exists for the C half.
        elif ! grep -q "Compiling edgefirst-tensor " "${g6_build_log}"; then
          cannot_measure "G6" "the Python extension rebuild did not recompile edgefirst-tensor, so byte-identical extensions would prove nothing (see ${g6_build_log})"
        elif [[ -z "${g6_py_before}" ]] || [[ -z "${g6_py_after}" ]]; then
          cannot_measure "G6" "could not hash the Python extensions' loadable content (is objcopy available?)"
        elif [[ "${g6_py_before}" != "${g6_py_after}" ]]; then
          gate "G6" "a Python extension's LOADABLE CONTENT changed when only edgefirst-tensor's private layout changed -- something is still being baked in (see ${g6_build_log})" 1
        else
          gate "G6" "libedgefirst_tensor.so changed (rebuild confirmed); the four C siblings are byte-identical and the four Python extensions' loadable content is identical ACROSS A REAL RECOMPILE of edgefirst-tensor" 0
        fi
      fi
      rm -f "${g6_build_log}"
    fi
    g6_revert
    # Only disarm the EXIT trap once the revert actually succeeded
    # (g6_revert clears g6_patched itself on success, leaves it set on
    # failure) -- otherwise a failed revert here would go both unwarned-of
    # a second time AND unretried, since the trap that would have caught it
    # on exit is gone.
    if [[ "${g6_patched}" -eq 0 ]]; then
      trap - EXIT
    fi
  fi
  echo
  if [[ "${DRIFT_ONLY}" -eq 1 ]]; then
    if [[ "${fails}" -eq 0 ]]; then echo "ALL GATES GREEN"; exit 0; fi
    echo "${fails} GATE(S) RED"; exit 1
  fi
fi

# `--differential-only` skips every gate below and runs G13 alone, which is
# what its own comment always claimed. It did not: G13 sits at the END of
# this script, so the early-exit after it came too late and every other gate
# ran first -- burying G13's verdict among unrelated failures and, worse,
# leaving the process exit code carrying no information about the
# differential at all. A lane that shells out and checks `$?` learned
# nothing.
#
# The guarded body below is deliberately NOT re-indented. Bash does not care,
# and a two-line diff is reviewable where a 600-line whitespace change is
# not. The matching `fi` is immediately before the G13 block.
if [[ "${DIFFERENTIAL_ONLY}" -eq 0 ]]; then

echo "== G1: no sibling embeds edgefirst-tensor's implementation =="
# Plain `nm`, not `nm -D`: the embedded symbols this gate looks for are
# static/internal copies, not dynamic exports, so `-D` sees none of them
# even on a library that plainly carries hundreds. Switching to `-D` would
# make G1 a permanent false PASS, which is worse than the bug this round is
# fixing. This was proposed once already on this branch and caught only by
# measuring it.
#
# NOT a plain `edgefirst_tensor` substring match (task 9's original
# implementation, corrected here after a real dynamic flip made it read
# false). A correctly `dynamic`-flipped leaf still STATICALLY links the
# `edgefirst-tensor` crate -- only the *implementation* moved to the .so;
# the crate's own thin `dynamic` wrapper code, and any generic std-library
# method the leaf's own source instantiates over an `edgefirst_tensor`
# type (`Option<edgefirst_tensor::X>::map`, `mem::drop::<edgefirst_tensor
# ::Y>`, ...), still compile into the leaf and still carry
# `edgefirst_tensor` in their mangled names. Measured by hand across the
# three leaves task 9 flipped: 807/7616/177 such symbols respectively, ALL
# either monomorphized std-library generics or `dynamic`'s own wrapper
# code -- zero from the real implementation. A plain substring match
# cannot tell those apart from genuine embedding, so it can never reach 0
# for any real consumer of this crate's public types, flipped or not --
# an unreachable target is exactly as dishonest as a target that is
# trivially reachable.
#
# `static_backend` is the actual, measured discriminator for the
# `TensorStorage`-dispatched implementations: `dma.rs`/`mem.rs`/`shm.rs`/
# `pbo.rs`/`iosurface.rs`/`ahardwarebuffer.rs` are all reached only through
# `TensorStorage`'s enum -- nothing else calls into them, on either backend.
# Verified against all four siblings' actual debug builds before adopting
# this: `static_backend` symbol count was 88 in the one leaf still
# statically embedding (`libedgefirst_image.so`, unflipped) and exactly 0
# in each of the three genuinely-flipped leaves (`codec`/`decoder`/
# `tracker`) -- a clean binary split, proving this marker discriminates
# rather than reading green regardless of flip state. (Rejected candidates,
# for the same reason: `capacity_bytes` and bare `cuda`/`mem` module-path
# substrings all produced false positives -- `dynamic`'s own `TensorTrait`
# impl and its `cuda()`/`as_dma()` accessors legitimately reference those
# names too.)
#
# `cuda.rs` is deliberately NOT covered by this marker, and this is a real
# gap G1 does not close -- not an oversight in the marker's design. Unlike
# the modules above, `mod cuda;` in `lib.rs` carries no `#[cfg(feature =
# "static")]` gate: it is a standalone `dlopen(libcudart)` interop layer
# with no `TensorStorage` dependency, `pub use`d at the crate root on
# *both* backends. `codec-capi` calls `edgefirst_tensor::is_cuda_available
# ()`/`stream_create()`/`stream_synchronize()` directly (its nvjpeg path),
# never through the tensor ABI, so gating `cuda.rs` behind `static` would
# break `codec` under `dynamic` -- it is not a bug this gate can fix by
# tightening the marker. Each leaf that links the crate and reaches any of
# `cuda.rs`'s functions compiles its own copy of `cuda::load()`'s
# dlopen+dlsym sequence, genuinely duplicated across leaves the same way
# `dynamic`'s own wrapper code is (see above) -- measured and reported
# separately as G1b, immediately below, so this real (if modest) exception
# is visible rather than silently passing inside G1's PASS.
#
# Because plain nm needs the static symbol table, it is blind on a stripped
# binary -- that must read as cannot_measure, not as "0 embedded symbols,
# want 0, PASS".
for l in $SIBS; do
  so="${DEBUG_LIBDIR}/libedgefirst_${l}.so"
  if [[ ! -s "${so}" ]] || [[ ! -r "${so}" ]]; then
    cannot_measure "G1" "${so} is missing, empty, or unreadable"
    continue
  fi
  syms=$(nm "${so}" 2>/dev/null)
  if [[ -z "${syms}" ]]; then
    cannot_measure "G1" "${so} has no symbol table (stripped, or unreadable)"
    continue
  fi
  n=$(printf '%s\n' "${syms}" | grep -c 'static_backend' || true)
  gate "G1" "libedgefirst_${l}.so carries ${n} static_backend (embedded implementation) symbols (want 0)" \
       "$([[ "${n}" -eq 0 ]] && echo 0 || echo 1)"
done

echo "== G1b: cuda.rs's ungated dlopen surface (accepted exception, measured not hidden) =="
# `static_backend` (G1, above) does not and cannot cover this -- see the
# comment on G1 for why `cuda.rs` is excluded from that marker by design,
# not by omission. This is a SEPARATE, INFORMATIONAL measurement: it always
# reports PASS (there is no target to drive toward -- gating `cuda.rs`
# behind `static` would break `codec-capi`'s direct, non-ABI calls to it
# under `dynamic`), but it must never silently read as "0 bytes duplicated"
# just because nobody looked. The byte figure is the point of this check,
# not the PASS.
#
# Positive allowlist, not a substring exclusion list, for the same reason
# G1 rejected a bare `cuda` substring: `edgefirst_tensor::cuda::*` also
# appears in generic std-library scaffolding this crate's own callers
# instantiate (`Option<CudaHandle>::as_ref`, `Box<CudaHandle>`'s drop glue,
# `Arc<dyn CudaGlOps>`'s vtable plumbing, a `Iterator::find_map` monomorphized
# over `load`'s own closure type, ...) and in each leaf's own trait impls of
# `CudaGlOps` -- none of that is `cuda.rs`'s implementation duplicating
# itself, so counting it would overstate this exception's real cost. Every
# alternative below is a real, non-generic item `cuda.rs` itself defines: the
# dlopen/dlsym sequence (`load` and its closure, `table`, the `TABLE`
# static), the public functions built on it (`is_cuda_available`,
# `memcpy_device_to_host`, `stream_create`/`stream_destroy`/
# `stream_synchronize`, `gl_map_resource`/`gl_register_buffer`/
# `gl_unmap_resource`/`gl_unregister_resource`, `import_dma_fd`), and --
# missed by an earlier version of this marker, which counted only the free
# functions and so undercounted the real cost -- `CudaHandle`'s and
# `CudaMap`'s own inherent methods and `Drop`/`Debug` impls (both types are
# defined in `cuda.rs` itself, so these are exactly as much "cuda.rs's own
# implementation" as `load()` is). Deliberately not stated as a count in
# prose: a number here rots the next time an item is added or removed, and
# the regex itself is the source of truth for what it matches.
#
# Cross-checked against `lib.rs`'s own `pub use cuda::{...}` re-export block
# (every free function `cuda.rs` makes part of this crate's public API):
# `memcpy_device_to_host` was missing from an earlier version of this list
# too (found in review, F33) -- present now. Nothing else in that block is
# missing: `CudaGlOps` is a trait with no implementation of its own in
# `cuda.rs` (only external types implement it, already excluded by this
# marker's `edgefirst_tensor::cuda::` prefix requirement) and `CudaStream`
# is a bare type alias with no methods to duplicate.
#
# The match is anchored to the start of `nm -S`'s own symbol column (after
# the address/size/type fields), not a bare substring search: a symbol like
# `<... as Iterator>::find_map::<..., edgefirst_tensor::cuda::load::
# {closure#0}>` contains the same text as the real, standalone
# `cuda::load::{closure#0}` symbol, but only as an embedded generic
# argument, not as the symbol's own identity -- anchoring at start-of-name
# is what tells the two apart. A leaf carries only the subset its own call
# graph actually reaches (verified: `tracker-capi` reaches none of it and
# correctly measures 0; `codec-capi`'s `CudaMap::device_ptr`/`::len` show up
# because its nvjpeg path calls them, `image-capi`'s `CudaHandle::new_gl`
# shows up because its GL PBO path calls it, neither appears in the other).
CUDA_MARKER='^[0-9a-f]+ +[0-9a-f]+ +[a-zA-Z] +<?edgefirst_tensor\[[0-9a-f]+\]::cuda::(load(::\{closure#0\})?|is_cuda_available|memcpy_device_to_host|stream_create|stream_destroy|stream_synchronize|gl_map_resource|gl_register_buffer|gl_unmap_resource|gl_unregister_resource|import_dma_fd|table|TABLE|CudaHandle(>::(new_gl|new_external|map)| as core\[[0-9a-f]+\]::(ops::drop::Drop>::drop|fmt::Debug>::fmt))|CudaMap(>::(device_ptr|len|is_empty)| as core\[[0-9a-f]+\]::ops::drop::Drop>::drop))$'
for l in tensor $SIBS; do
  so="${DEBUG_LIBDIR}/libedgefirst_${l}.so"
  if [[ ! -s "${so}" ]] || [[ ! -r "${so}" ]]; then
    cannot_measure "G1b" "${so} is missing, empty, or unreadable"
    continue
  fi
  syms=$(nm -S "${so}" 2>/dev/null | c++filt 2>/dev/null)
  if [[ -z "${syms}" ]]; then
    cannot_measure "G1b" "${so} has no symbol table (stripped, or unreadable)"
    continue
  fi
  hits=$(printf '%s\n' "${syms}" | grep -cE "${CUDA_MARKER}" || true)
  bytes=$(printf '%s\n' "${syms}" | grep -E "${CUDA_MARKER}" | awk '{sum+=strtonum("0x"$2)} END{print sum+0}')
  gate "G1b" "libedgefirst_${l}.so duplicates ${bytes} B (${hits} symbols) of cuda.rs's own dlopen implementation -- accepted exception, see G1's comment" 0
done

echo "== G2: each sibling really links libedgefirst_tensor.so =="
# `-D` is the right tool here (unlike G1): DT_NEEDED and undefined dynamic
# symbols are exactly the dynamic-linking metadata `-D`/`readelf -d` read,
# and both survive `strip`, so this gate is not blinded by a stripped
# release build the way plain `nm` would be.
#
# A leaf can legitimately need ZERO `ef_tensor_*` symbols: `tracker-capi`
# touches only plain `edgefirst_tensor` value types (`DetectBox`/
# `BoundingBox`), never calls a tensor primitive, and the default
# `--as-needed` linker behavior correctly drops even an explicit
# `-ledgefirst_tensor` when nothing references it -- there is genuinely
# nothing to link. That is the designed outcome for a library that needs
# no tensor implementation, not a failure to flip, so "DT_NEEDED=0
# undefined ef_tensor_*=0" must not always read as FAIL. But an
# unmeasurable/missing library reads the same way by accident (`readelf`/
# `nm -D` on a nonexistent file also print nothing), so a bare "0 and 0"
# cannot be trusted as this legitimate state on its own -- it is
# cross-checked against G1's own `static_backend` measurement (plain
# `nm`, read fresh here rather than passed from G1, so this gate stays
# self-contained): only a leaf that ALSO carries zero embedded
# implementation gets read as "correctly needs nothing". A leaf that
# still embeds the implementation (never flipped, e.g. `image-capi`
# today) and shows 0/0 is still a real FAIL -- it never attempted to
# link at all, which is a different fact than "did not need to".
for l in $SIBS; do
  so="${DEBUG_LIBDIR}/libedgefirst_${l}.so"
  if [[ ! -s "${so}" ]] || [[ ! -r "${so}" ]]; then
    cannot_measure "G2" "${so} is missing, empty, or unreadable"
    continue
  fi
  syms=$(nm "${so}" 2>/dev/null)
  if [[ -z "${syms}" ]]; then
    cannot_measure "G2" "${so} has no symbol table (stripped, or unreadable)"
    continue
  fi
  embeds_static=$(printf '%s\n' "${syms}" | grep -c 'static_backend' || true)
  need=$(readelf -d "${so}" 2>/dev/null | grep -c 'libedgefirst_tensor.so' || true)
  und=$(nm -D -u "${so}" 2>/dev/null | grep -c ' ef_tensor_' || true)
  if [[ "${need}" -ge 1 ]] && [[ "${und}" -ge 1 ]]; then
    gate "G2" "libedgefirst_${l}.so: DT_NEEDED=${need} undefined ef_tensor_*=${und}" 0
  elif [[ "${embeds_static}" -eq 0 ]] && [[ "${need}" -eq 0 ]] && [[ "${und}" -eq 0 ]]; then
    gate "G2" "libedgefirst_${l}.so: no embedded implementation (static_backend=0) and no ef_tensor_* reference -- this leaf needs no tensor primitive" 0
  else
    gate "G2" "libedgefirst_${l}.so: DT_NEEDED=${need} undefined ef_tensor_*=${und}, static_backend=${embeds_static}" 1
  fi
done

echo "== G4: the transition vtable is gone =="
# Repo-wide, not scoped to crates/*-capi/src -- task 10's own report caught
# the earlier, narrower version of this gate: its message read "0 files
# still reference EfTensorVtable/is_own_mint" with no mention of where it
# looked, which is a claim about the whole repository when the search was
# actually four directories. A grep of the same two strings over the full
# tree (source only, build output and .git excluded) is the honest version
# of that claim, and it is a stronger gate besides: it would also catch a
# stray EfTensorVtable reappearing anywhere outside crates/*-capi.
#
# One named exception. `crates/tensor/src/pbo.rs` declares `PboOpsVtable`,
# the *different* function-pointer table (from the cross-cdylib PBO work,
# 7979285c/e449ae66) that carries PBO callbacks across .so boundaries so
# `import_descriptor` can rebuild a PBO-backed tensor. Its own doc comments
# name-check `tensor-capi`'s (now-deleted) `EfTensorVtable` once, by way of
# analogy to the technique -- a correct, historical mention, not a survival
# of the thing this gate exists to catch. Excluded by path, not by softening
# the pattern, so the exemption is visible and named rather than silently
# narrowing what the grep can ever find.
#
# `find` over the whole tree, not a bare glob: unlike the old crates/*-capi/
# src form, `.` always expands (it is the working directory, not a pattern),
# so there is no unexpanded-glob trap here -- but the empty-result trap
# below it still applies (a partial checkout with zero .rs files would also
# grep clean), so a positive file count is still asserted before trusting a
# 0.
src_file_count=$(find . -path ./target -prune -o -path ./.git -prune -o \
  -type f -name '*.rs' -print 2>/dev/null | wc -l)
if [[ "${src_file_count}" -eq 0 ]]; then
  cannot_measure "G4" "repo-wide search found no .rs files"
else
  v=$(find . -path ./target -prune -o -path ./.git -prune -o \
    -type f -name '*.rs' -print 2>/dev/null \
    | grep -v '^\./crates/tensor/src/pbo\.rs$' \
    | xargs -r grep -l 'EfTensorVtable\|is_own_mint' 2>/dev/null | wc -l)
  gate "G4" "${v} files repo-wide (excluding crates/tensor/src/pbo.rs's unrelated PboOpsVtable) still reference EfTensorVtable/is_own_mint (want 0)" \
       "$([[ "${v}" -eq 0 ]] && echo 0 || echo 1)"
fi

echo "== G5: footprint =="
# WHAT THIS GATE MEASURES, AND WHAT IT DELIBERATELY NO LONGER CLAIMS.
#
# G5 reports the five libraries' FINAL SIZES and fails only if their total
# exceeds a recorded ceiling. It does not compute, report, or assert a
# "saved" delta against a historical baseline. That is a deliberate
# narrowing, for two independent reasons, both established by measurement
# rather than argued:
#
# 1. THE OLD BASELINE WAS NOT A MEASUREMENT. The previous BASELINE_BYTES
#    (v2, 18,412,784 B) was recorded in commit `be7589e7` as a re-measurement
#    of `1f71595d` in a scratch worktree. That commit CANNOT COMPILE THE TREE
#    IT CLAIMS TO MEASURE: `be7589e7` deletes `crates/image-capi/src/vtable.rs`
#    while leaving `pub mod vtable;` at `crates/image-capi/src/lib.rs:40`, so
#    `edgefirst-image-capi` fails with E0583 (verified in a scratch worktree
#    at that commit; fixed later by `f8b90af1`). The 8.9 MB
#    `libedgefirst_image.so` folded into its total was therefore a leftover
#    artifact from an earlier build of a different tree -- roughly half the
#    recorded figure, contributed by a library that did not build. Every G5
#    verdict since has been calibrated against that number. It is not
#    recoverable by re-deriving a target from it; the input itself is void.
#
# 2. A SAVINGS DELTA MEASURES THE WRONG THING ANYWAY. Subtracting a current
#    total from a frozen historical one charges every feature added since
#    against a number labelled "de-duplication saved". Concretely, this
#    branch added 79,528 B to `libedgefirst_tensor.so` -- nine dynamic-backend
#    methods, an error class on every NULL-returning C entry, and the exported
#    surface going 56 -> 69 functions -- which is new capability, and which a
#    savings delta reports as a loss. Worse, growth in the SHARED library is
#    exactly the shape this branch exists to produce: one copy getting bigger
#    while four siblings do not.
#
# DE-DUPLICATION IS G1'S JOB, NOT THIS GATE'S. G1 measures it directly and
# exactly -- 0 `static_backend` symbols in each of the four siblings -- and
# can actually fail if a leaf re-embeds. Two gates claiming the same property
# when only one of them can measure it is how a wrong number acquires a
# second, agreeing witness. G5's remaining job is narrower and honest: notice
# if the five libraries get big.
CEILING_BYTES=23420000
# CEILING HISTORY -- a number with an arithmetic derivation, not a feeling.
#
# v3 (2026-09-05): re-measured at HEAD by this gate in CI (ubuntu-22.04,
# x86_64) on the Windows zero-copy tensor round (hal#152), which the v2
# ceiling tripped by 15,656 B:
#   tensor 1,422,088 + image 9,721,704 + codec 1,208,336 + decoder 10,456,288
#   + tracker 407,240 = 23,215,656 B.
# Delta over v2 is 215,512 B: tensor +34,400 (try_map, cuda_map_mut,
# gpu_write_value, the D3D11_TEXTURE descriptor kind and the reference-mode
# blob record, all of which exist on every platform; the D3D11 device and
# texture code itself is cfg'd out of these builds), decoder +240,248 and
# codec +20,696 (main's 0.29.x releases between v2 and this measurement),
# image -87,088 (the stride-honouring readbacks replaced three open-coded
# copies with one), tracker +7,256. G1 still reports 0 sibling embeds.
# Headroom above the measurement: 204,344 B (0.88%), the same role as v1
# and v2 (build-to-build variance, not a de-duplication budget).
#
# v2 (2026-08-26): re-measured at HEAD after the modular C/Python split and
# the 0.29.0 release-readiness work (ordered crates.io publish, one C archive,
# tracker wheels). Built by this gate itself:
#   tensor 1,387,688 + image 9,808,792 + codec 1,187,640 + decoder 10,216,040
#   + tracker 399,984 = 23,000,144 B.
# Headroom above that: 199,856 B (0.87%), same role as v1 (build-to-build
# variance, not a de-duplication budget). Decoder is the bulk of the v1→v2
# delta (~3.4 MB); G1 still reports 0 sibling embeds.
#
# v1 (2026-08-25): measured at HEAD, honestly built by this gate itself:
#   tensor 1,084,344 + image 8,975,320 + codec 747,960 + decoder 6,803,296
#   + tracker 400,200 = 18,011,120 B. Ceiling was 18,100,000 B.
#
# Headroom above that: 88,880 B (0.49%), justified against BUILD-TO-BUILD
# VARIANCE measured here rather than against any inherited figure. The
# largest variance observed on this branch from a change with no size intent
# was 1,288 B (swapping ~30 `chunks_exact` sites to `as_chunks` across
# `crates/image`, commit `cb580cef`, which moved the five-library total by
# that much -- downward, as it happens). The headroom is ~69x that, so noise
# cannot trip this ceiling and a real 88 KB of growth can.
#
# The earlier revision of this comment justified its number differently, by
# citing per-leaf "cost of re-embedding the tensor implementation" figures
# (image 325,999 B, decoder 145,012 B, codec 127,732 B). Those are NOT used
# here, for two reasons. They come from the same comment whose baseline is
# discredited above, and re-using unverified numbers from a discredited
# source is the exact mistake this revision exists to correct. And they
# describe a regression that turns out not to be silently reachable: flipping
# `crates/codec-capi/Cargo.toml`'s `edgefirst-tensor` dependency from
# `dynamic` to `static` -- the literal re-embed -- does not compile (E0308,
# two errors; the leaf's own source calls dynamic-backend-shaped APIs).
# A leaf cannot quietly re-acquire a private copy; it fails to build, and if
# it somehow did, G1 measures it directly. This ceiling is not that guard and
# should not be sized as though it were.
#
# What this ceiling DOES catch is ordinary, quiet growth: an added dependency
# that pulls in more than expected, a profile setting drifting (losing
# `strip = true` or `lto = "thin"` would dwarf this headroom), or ABI surface
# expanding past the point where someone should look at it deliberately.
#
# WHEN THIS CEILING IS HIT BY LEGITIMATE GROWTH, RE-RECORD IT -- deliberately,
# with the new measurement and the reason written here. That is the intended
# workflow: the ceiling is a decision point, not a law of physics. What it
# must never become again is a number nobody can trace to a build that
# actually happened.

# G5 BUILDS WHAT IT MEASURES. It does not sum `target/release/*.so` off disk.
#
# Reading artifacts off disk is the single most productive defect class on
# this branch, and it has now been found in four places: G6's automated mode
# hashed `target/debug` while rebuilding `--release`; G6's Python baseline
# read artifacts a previous `--drift` run had left from a tree that no longer
# existed; a differential comparison ran against a venv built from a
# different commit; and this gate. When last run off disk, the four siblings
# in `target/release` were dated 09:13 and `libedgefirst_tensor.so` 13:55 --
# two vintages, neither at HEAD -- and G5 reported a total for a tree that
# never existed, missing its threshold by 1,184 B. Built at HEAD the same
# moment, the real shortfall was 98,336 B. The artifact on disk is not what
# its name implies, and a gate that trusts the name is measuring history.
g5_unmeasurable=""
g5_provenance=""
if [[ -n "${PROFILE}" ]]; then
  # PROFILE redirects RELEASE_LIBDIR to target/<PROFILE>, but this gate can
  # only build `--release`. Measuring target/<PROFILE> while building
  # target/release is precisely G6's original defect, so refuse rather than
  # reproduce it. `PROFILE=nonexistent` exists to exercise the cannot-measure
  # paths, and this IS G5's cannot-measure path.
  g5_unmeasurable="PROFILE=${PROFILE} is set, but G5 measures only what it builds and can only build --release -- unset PROFILE to run G5"
else
  # The same five-crate release build the Makefile's `capi-libs-release`
  # target performs, inlined rather than shelled out to `make` so the gate
  # does not depend on the Makefile staying in sync with it. The leaves are
  # workspace-EXCLUDED standalone packages (the static/dynamic feature switch
  # is mutually exclusive and cargo unifies features across one invocation),
  # so each is built via its own --manifest-path, one invocation apiece.
  g5_build_log="${TMPDIR:-/tmp}/check-single-home-g5-build.$$.log"
  : >"${g5_build_log}"
  for l in tensor $SIBS; do
    if ! cargo build --release --manifest-path "crates/${l}-capi/Cargo.toml" \
         --target-dir target >>"${g5_build_log}" 2>&1; then
      g5_unmeasurable="crates/${l}-capi failed to build at HEAD -- see ${g5_build_log} (G5 measures what it builds; it will not fall back to whatever is on disk)"
      break
    fi
  done
  [[ -z "${g5_unmeasurable}" ]] && rm -f "${g5_build_log}"
  g5_provenance="built at HEAD"
fi

g5_sizes=""
tot=0
if [[ -z "${g5_unmeasurable}" ]]; then
  for l in tensor $SIBS; do
    f="${RELEASE_LIBDIR}/libedgefirst_${l}.so"
    # `[ -f ]` alone is not enough: a 0-byte `.so` (a truncated or interrupted
    # build) passes it, and `stat -c%s` happily returns "0" for it -- not
    # empty, not an error, just a real zero that silently folds into the sum
    # and moves the total AWAY from the ceiling instead of failing. `[ -s ]`
    # requires a positive size, so it rejects both "missing" and "present but
    # empty" in one test. This still matters even though the build above
    # succeeded: a build can succeed and write an artifact elsewhere (a
    # different --target-dir, a renamed cdylib), which would leave a stale or
    # absent file here with nothing having failed.
    if [[ ! -s "${f}" ]]; then
      g5_unmeasurable="${f} is missing or empty even though crates/${l}-capi built successfully -- artifact path mismatch, not a build failure"
      break
    fi
    s=$(stat -c%s "${f}" 2>/dev/null)
    if [[ -z "${s}" ]]; then
      g5_unmeasurable="${f} size could not be read"
      break
    fi
    tot=$((tot+s))
    g5_sizes="${g5_sizes}  G5   ---   ${l} ${s} B"$'\n'
  done
fi

if [[ -n "${g5_unmeasurable}" ]]; then
  # This is the case a missing (or empty) target/release directory used to
  # hit: every `stat` fell back to 0 or a 0-byte file measured as 0, `tot`
  # stayed 0, and a total of 0 sits comfortably under any ceiling -- five
  # unbuilt (or truncated) libraries reporting PASS.
  cannot_measure "G5" "${g5_unmeasurable}"
else
  printf '%s' "${g5_sizes}"
  # The result line states its PROVENANCE, not just its number. A size is
  # meaningless without knowing which tree produced it, and this gate spent a
  # day reporting numbers for a tree that never existed. Anyone reading a G5
  # line should be able to tell, from the line alone, whether a build happened.
  gate "G5" "five-library total ${tot} B (${g5_provenance}; ceiling ${CEILING_BYTES} B). De-duplication is G1's measure, not this gate's." \
       "$([[ "${tot}" -le "${CEILING_BYTES}" ]] && echo 0 || echo 1)"
fi

echo "== G7: dynamic backend aliasing shapes are clean under Miri (both models) =="
# `scripts/miri.sh` runs `crates/tensor/tests/scenarios.rs` under Stacked
# Borrows (Miri's default) and Tree Borrows (`-Zmiri-tree-borrows`). Miri
# cannot execute FFI at all, so this can only ever exercise the `static`
# backend's raw-handle borrow shapes -- see that script's own header
# comment and task-11-report.md for exactly what this gate does and does
# not cover, and for a real, established disagreement between the two
# models that scenarios.rs's own diagnostic test surfaced.
#
# Toolchain absence is `cannot_measure`, not a silent pass: `miri.sh` exits
# 2 specifically when the nightly Miri component isn't installed, distinct
# from exit 1 (a real aliasing failure) and exit 0 (both models clean).
if [[ ! -x scripts/miri.sh ]]; then
  cannot_measure "G7" "scripts/miri.sh is missing or not executable"
else
  g7_log="${TMPDIR:-/tmp}/check-single-home-g7.$$.log"
  ./scripts/miri.sh >"${g7_log}" 2>&1
  miri_rc=$?
  if [[ "${miri_rc}" -eq 2 ]]; then
    cannot_measure "G7" "Miri toolchain unavailable -- $(tail -n1 "${g7_log}")"
  else
    gate "G7" "scripts/miri.sh: both aliasing models (rc=${miri_rc}; re-run ./scripts/miri.sh for detail)" \
         "$([[ "${miri_rc}" -eq 0 ]] && echo 0 || echo 1)"
  fi
  rm -f "${g7_log}"
fi

echo "== G9: detections are owned by decoder =="
tensor_so="${DEBUG_LIBDIR}/libedgefirst_tensor.so"
decoder_so="${DEBUG_LIBDIR}/libedgefirst_decoder.so"
tensor_hdr="crates/tensor-capi/include/edgefirst/tensor.h"
g9_unmeasurable=""
# tensor.h needs BOTH preconditions. `-s`/`-r`, not `-f`, for the same reason
# as G1/G5: a 0-byte tensor.h would pass `-f`, `grep -c` it for zero matches,
# and read as "0 mentions, want 0, PASS" even though nothing was read. But
# `-s`/`-r` only proves the file has readable bytes, not that they are
# tensor.h -- a comment-only or truncated header clears it and still greps 0
# mentions. `h` itself cannot close that hole: its target is legitimately 0
# once the detection vocabulary moves out, so `h == 0` is exactly the state
# the plan is driving toward and can never be evidence the measurement ran.
# The liveness signal is therefore a different quantity: the total number of
# ef_* declarations tensor.h yields, which a real tensor.h always makes >= 1
# no matter how many of them mention detections. Same signal G11 uses, same
# helper, checked separately from `h`.
if [[ ! -s "${tensor_so}" ]] || [[ ! -r "${tensor_so}" ]]; then
  g9_unmeasurable="${tensor_so} is missing, empty, or unreadable"
elif [[ -z "$(nm -D --defined-only "${tensor_so}" 2>/dev/null)" ]]; then
  g9_unmeasurable="${tensor_so} exports no dynamic symbols at all"
elif [[ ! -s "${decoder_so}" ]] || [[ ! -r "${decoder_so}" ]]; then
  g9_unmeasurable="${decoder_so} is missing, empty, or unreadable"
elif [[ -z "$(nm -D --defined-only "${decoder_so}" 2>/dev/null)" ]]; then
  g9_unmeasurable="${decoder_so} exports no dynamic symbols at all"
elif [[ ! -s "${tensor_hdr}" ]] || [[ ! -r "${tensor_hdr}" ]]; then
  g9_unmeasurable="${tensor_hdr} is missing, empty, or unreadable"
elif [[ "$(header_decls "${tensor_hdr}" | wc -l)" -eq 0 ]]; then
  g9_unmeasurable="${tensor_hdr}: extraction found 0 declarations (expected >= 1)"
fi
if [[ -n "${g9_unmeasurable}" ]]; then
  cannot_measure "G9" "${g9_unmeasurable}"
else
  t=$(nm -D --defined-only "${tensor_so}" 2>/dev/null | grep -c 'ef_detect_box_list' || true)
  d=$(nm -D --defined-only "${decoder_so}" 2>/dev/null | grep -c 'ef_detect_box_list' || true)
  # `|| true` for the same reason as `t`/`d` above: `grep -c` exits 1 when the
  # count is 0, which here is the SUCCESS state this gate is driving toward.
  # Harmless today (no `set -e`), but without it the script would abort at the
  # exact moment G9's header requirement is finally met if `-e` were ever added.
  h=$(grep -c 'ef_detect_box' "${tensor_hdr}" || true)
  gate "G9" "tensor.so exports ${t} (want 0), decoder.so exports ${d} (want 6), tensor.h mentions ${h} (want 0)" \
       "$([[ "${t}" -eq 0 ]] && [[ "${d}" -eq 6 ]] && [[ "${h}" -eq 0 ]] && echo 0 || echo 1)"
fi

echo "== G11: header/library alignment =="
for l in tensor image codec decoder tracker; do
  hdr="crates/${l}-capi/include/edgefirst/${l}.h"
  so="${DEBUG_LIBDIR}/libedgefirst_${l}.so"
  if [[ ! -s "${hdr}" ]] || [[ ! -r "${hdr}" ]]; then
    cannot_measure "G11" "${hdr} is missing, empty, or unreadable"
    continue
  fi
  if [[ ! -s "${so}" ]] || [[ ! -r "${so}" ]]; then
    cannot_measure "G11" "${so} is missing, empty, or unreadable"
    continue
  fi
  bad=0
  decl_count=0
  exports=$(nm -D --defined-only "${so}" 2>/dev/null)
  while read -r fn; do
    [[ -z "${fn}" ]] && continue
    decl_count=$((decl_count+1))
    printf '%s\n' "${exports}" | grep -q " ${fn}\$" || bad=$((bad+1))
  done < <(header_decls "${hdr}")
  # `bad` (the value gated on) is legitimately 0 in the passing state, so it
  # cannot double as its own "did this actually run" signal -- an empty
  # while-loop body (zero declarations extracted, e.g. a header whose
  # content doesn't match the extraction pipeline's shape at all) leaves
  # `bad` at its initial 0 too, indistinguishable from "checked N
  # declarations, 0 mismatches". `decl_count` is the total extracted,
  # independent of match outcome, and every one of these five headers
  # genuinely declares functions -- 0 extracted can only mean the header
  # was empty, unreadable, or not the file the extraction rule expects.
  if [[ "${decl_count}" -eq 0 ]]; then
    cannot_measure "G11" "${hdr}: extraction found 0 declarations (expected >= 1)"
    continue
  fi
  gate "G11" "${l}.h declares ${bad} function(s) libedgefirst_${l}.so does not export (want 0)" \
       "$([[ "${bad}" -eq 0 ]] && echo 0 || echo 1)"
done

echo "== G12: Python extensions really link libedgefirst_tensor.so =="
# G2's claim, on the Python side: each installed extension must show a
# DT_NEEDED entry for libedgefirst_tensor.so AND at least one undefined
# ef_tensor_* symbol -- both, for the reason G2 needs both. A DT_NEEDED
# entry with no undefined reference to back it up means the default
# `--as-needed` linker behavior already dropped the link at build time;
# the entry alone is not proof anything is actually used from it.
#
# Do NOT fall back to an `nm`/embedded-symbol count here, the way G1 does
# on the C side -- every shipped Python extension is stripped, with no
# unstripped build of these anywhere to read instead (unlike the C
# leaves' debug .so's the rest of this script uses). `nm`'s plain symbol
# table is gone on a stripped binary, so counting embedded
# `static_backend` symbols would read 0 for every extension regardless of
# whether anything is actually embedded -- a `cannot_measure` condition
# wearing a clean PASS. This exact mistake was made once already, scoping
# this very gate: an early `nm | grep -c static_backend` sweep over these
# five extensions returned 0 across the board and looked like a perfect
# result. It was reading a stripped binary's absent symbol table, not the
# code's actual shape. So G12 checks only what strip cannot remove:
# DT_NEEDED and the dynamic symbol table, exactly like G2.
#
# The venv, never a global/system Python -- these are the actually
# installed, actually shipped extensions, not a fresh build.
PY_VENV="${PY_VENV:-venv}"
if [[ ! -x "${PY_VENV}/bin/python" ]]; then
  cannot_measure "G12" "${PY_VENV}/bin/python not found -- build/install the Python extensions into the venv first (never a system Python)"
else
  # Same technique the Makefile's own `build-python`/`test-python` targets
  # use to find site-packages: ask the venv's own interpreter, rather than
  # hardcoding a `python3.NN` version that will silently go stale.
  py_tag=$("${PY_VENV}/bin/python" -c 'import sys; print(f"python{sys.version_info.major}.{sys.version_info.minor}")' 2>/dev/null)
  py_site="${PY_VENV}/lib/${py_tag}/site-packages"
  if [[ -z "${py_tag}" ]] || [[ ! -d "${py_site}" ]]; then
    cannot_measure "G12" "could not resolve ${PY_VENV}'s site-packages directory (expected ${PY_VENV}/lib/python3.X/site-packages)"
  else
    # SCOPE: the four modular extensions, deliberately, and stated here
    # rather than left to whichever venv someone happens to point at.
    #
    # This gate used to carry a fifth target, `edgefirst_hal`, described as
    # "the pre-0.29 monolith (crates/capi)". That description was wrong in a
    # way worth recording, because it sent two people chasing a product
    # decision that did not exist.
    #
    # TWO DIFFERENT ARTIFACTS SHARE THE NAME `edgefirst_hal`:
    #
    #   * The PYTHON monolith -- `crates/python`, built as
    #     `edgefirst_hal/edgefirst_hal.cpython-*.so`, published to PyPI as
    #     the single `edgefirst-hal` project through v0.28.3. THIS is what
    #     the fifth target named, and `crates/python` was DELETED from this
    #     branch at commit `013a59c9`, replaced by the four `crates/python-*`
    #     packages measured below. No source here can rebuild it. The target
    #     resolved only in a stale dev venv, against a file predating the
    #     package split -- so its result depended entirely on which venv the
    #     gate was pointed at, reporting DT_NEEDED=0 against one and
    #     "cannot verify" against another, for an artifact this branch does
    #     not produce at all.
    #
    #   * The C monolith -- `crates/capi`, built as `libedgefirst_hal.so`
    #     plus `libedgefirst_hal.a` and `hal.h`. DELETED: the shipped C API
    #     is the five modular leaves (`libedgefirst_{tensor,codec,image,
    #     decoder,tracker}`). G12 still measures Python extensions only.
    #
    # A GREEN G12 THEREFORE MEANS "codec/image/decoder share one tensor;
    # tracker does not link tensor". The C libraries are outside this
    # gate's reach -- they are not Python extensions -- and are measured
    # separately (G1/G2/G5).
    #
    # label:glob, relative to site-packages. codec/image/decoder live under
    # the shared `edgefirst/` namespace package this plan's RPATH mechanism
    # relies on (`$ORIGIN/../tensor`). Tracker is standalone.
    #
    # CI builds with `--features abi3-py311`, which installs `_foo.abi3.so`.
    # A local `maturin develop` without abi3 installs `_foo.cpython-*.so`.
    # Accept either, never both (that is a stale venv).
    py_targets="image tensor codec decoder tracker"
    # A leftover `edgefirst_hal` install is not a failure of this gate's
    # claim -- it is not one of the four, and nothing on this branch builds
    # it -- but it IS a live hazard for whoever owns this venv: it is
    # importable, it predates `013a59c9`, and it carries its own embedded
    # copy of the tensor implementation. Silence about it would be the same
    # mistake in the other direction, so it is reported and does not fail.
    if [[ -d "${py_site}/edgefirst_hal" ]]; then
      printf '  %-4s WARN  %s\n' "G12" "${py_site}/edgefirst_hal is a stale install of the Python monolith removed at 013a59c9 -- nothing on this branch builds it; \`pip uninstall edgefirst-hal\` in this venv"
    fi
    for label in ${py_targets}; do
      abi3="${py_site}/edgefirst/${label}/_${label}.abi3.so"
      # shellcheck disable=SC2206
      cpython=(${py_site}/edgefirst/${label}/_${label}.cpython-*.so)
      matches=()
      if [[ -e "${abi3}" ]]; then
        matches+=("${abi3}")
      fi
      if [[ -e "${cpython[0]}" ]]; then
        matches+=("${cpython[@]}")
      fi
      if [[ "${#matches[@]}" -eq 0 ]]; then
        cannot_measure "G12" "no installed extension matching ${py_site}/edgefirst/${label}/_${label}.abi3.so or _${label}.cpython-*.so -- build/install it into the venv first"
        continue
      fi
      if [[ "${#matches[@]}" -gt 1 ]]; then
        cannot_measure "G12" "${#matches[@]} files matched edgefirst/${label}/_${label}.{abi3,cpython-*}.so, expected exactly one -- stale build in the venv?"
        continue
      fi
      so="${matches[0]}"
      if [[ ! -s "${so}" ]] || [[ ! -r "${so}" ]]; then
        cannot_measure "G12" "${so} is missing, empty, or unreadable"
        continue
      fi
      need=$(readelf -d "${so}" 2>/dev/null | grep -c 'libedgefirst_tensor.so' || true)
      und=$(nm -D -u "${so}" 2>/dev/null | grep -c ' ef_tensor_' || true)
      if [[ "${label}" = "tracker" ]]; then
        gate "G12" "${label} (${so#"${py_site}"/}): DT_NEEDED=${need} undefined ef_tensor_*=${und} (tracker must not link tensor)" \
             "$([[ "${need}" -eq 0 ]] && [[ "${und}" -eq 0 ]] && echo 0 || echo 1)"
      else
        gate "G12" "${label} (${so#"${py_site}"/}): DT_NEEDED=${need} undefined ef_tensor_*=${und}" \
             "$([[ "${need}" -ge 1 ]] && [[ "${und}" -ge 1 ]] && echo 0 || echo 1)"
      fi
    done
  fi
fi

# --------------------------------------------------------------------------
# G13: the two backends AGREE, test for test -- not merely both compile
# --------------------------------------------------------------------------
#
# Every other gate here inspects an artifact: which symbols a library
# exports, what a header declares, what an extension's DT_NEEDED says. Not
# one of them can see a method that links fine and then returns the wrong
# answer at runtime, and that is exactly what the dynamic backend shipped:
# 35 Python tests failed against dynamic-linked wheels that pass against
# static ones (task P2's report). Four distinct root causes -- a missing
# reshape primitive, `map` refusing PBO-backed tensors, `clone_fd` refusing
# SHM-backed ones, and `create_image(dtype="int8")` reporting `uint8`.
#
# A method-parity check -- does every `pub fn` on the static backend have a
# dynamic counterpart? -- was the obvious candidate and would have caught
# NONE of the four: all four methods exist on both sides. The difference is
# behavioural, so the check has to be behavioural. This runs the same Python
# suite against both backends and requires identical per-test outcomes.
#
# WHAT THIS GATE DOES NOT SEE. It compares per-test OUTCOMES, not code paths.
# Two backends can take entirely different routes on identical input and agree
# here, provided both routes reach the same verdict. That is not hypothetical:
# on this branch, static reached `setup_renderbuffer_dma` and hit a driver
# error while dynamic never attempted the EGLImage import at all (`as_dma`
# returning None) -- both then fell back to PBO transfers, both suites went
# green, and this gate reported 0 divergences. So a green G13 is evidence about
# refusals and wrong answers that REACH AN ASSERTION. It is not evidence of
# behavioural equivalence, and it must not be cited as such.
#
# Opt-in, like G6: it runs the whole Python suite twice. `--differential`
# adds it to the sweep, `--differential-only` runs it alone.
#
# Both sides are supplied by the caller, because building them is the
# packaging lane's job, not this script's:
#   PY_STATIC_VENV   (default: venv)        -- static-linked extensions
#   PY_DYNAMIC_VENV  (no default)           -- dynamic-linked extensions
# Missing or unverifiable inputs are `cannot_measure`, which FAILS. A gate
# that silently passes because it could not find a venv is the "gate that
# cannot fail" this branch has produced seven of.
fi # end of the --differential-only skip that began before G1

if [[ "${DIFFERENTIAL_MODE}" -eq 1 ]]; then
  echo "== G13: static and dynamic backends agree, test for test =="
  g13_static="${PY_STATIC_VENV:-venv}"
  g13_dynamic="${PY_DYNAMIC_VENV:-}"
  g13_tests="${PY_TEST_PATHS:-tests}"
  g13_ok=1

  # Record one line per test as `nodeid<TAB>outcome`, via `--junitxml`.
  #
  # Not parsed out of pytest's `-rA` short summary, which was the first
  # attempt: `PASSED`/`FAILED` lines carry the nodeid but `SKIPPED` lines
  # carry `[count] path:line: reason` instead, so every skip parsed as a
  # nodeid of `[1]` and each one showed up as a spurious divergence. The
  # XML keys every outcome the same way. `junitxml` is built into pytest,
  # so this still needs nothing extra installed in either venv.
  g13_run() { # g13_run <venv> <outfile>
    local venv="$1"
    local outfile="$2"
    # pytest's exit code is the ONLY signal that separates "ran the suite and
    # everything passed" from "collected nothing" (5) or "usage error, e.g. the
    # path does not exist" (4). Both of the latter still write a VALID junitxml
    # with tests="0", so the XML cannot tell them apart. Discarding this with
    # `|| true` is what let a nonexistent PY_TEST_PATHS report a clean pass.
    if "${venv}/bin/python" -m pytest "${g13_tests}" --tb=no -q -p no:cacheprovider \
      --junitxml="${outfile}.xml" >"${outfile}.raw" 2>&1; then echo 0 >"${outfile}.rc"; else echo "$?" >"${outfile}.rc"; fi
    "${venv}/bin/python" - "${outfile}.xml" >"${outfile}" <<'G13_PARSE' || true
import sys, xml.etree.ElementTree as ET
try:
    root = ET.parse(sys.argv[1]).getroot()
except Exception:
    sys.exit(0)  # empty output -> the caller's "collected 0 tests" guard fires
import re
rows, missing = [], set()
for case in root.iter("testcase"):
    node = "{}::{}".format(case.get("classname", ""), case.get("name", ""))
    outcome = "PASSED"
    for child in case:
        tag = child.tag
        if tag in ("failure", "error", "skipped"):
            outcome = tag.upper()
            if tag == "skipped":
                # "could not import 'yaml': No module named 'yaml'" -- a
                # skip caused by the VENV, not by the backend. Recorded
                # separately so the caller can tell the two apart.
                m = re.search(r"could not import '([^']+)'", child.get("message", "") or "")
                if m:
                    missing.add(m.group(1))
            break
    rows.append("{}\t{}".format(node, outcome))
# An empty `rows` must produce an EMPTY file, not one newline. The caller's
# liveness guard counts lines; `print("")` writes 1 byte and `wc -l` reads 1,
# so a run that collected nothing read as "1 test" and compared clean.
if rows:
    print("\n".join(sorted(set(rows))))
with open(sys.argv[1] + ".missing", "w") as fh:
    fh.write("\n".join(sorted(missing)))
G13_PARSE
    return 0
  }

  g13_check_venv() { # g13_check_venv <label> <path>
    local label="$1"
    local path="$2"
    if [[ -z "${path}" ]]; then
      cannot_measure "G13" "${label} venv not supplied -- set PY_DYNAMIC_VENV to a venv with dynamic-linked extensions installed (never a system Python)"
      return 1
    fi
    if [[ ! -x "${path}/bin/python" ]]; then
      cannot_measure "G13" "${label} venv ${path}/bin/python not found -- build/install the extensions into it first"
      return 1
    fi
    if ! "${path}/bin/python" -c 'import pytest' >/dev/null 2>&1; then
      cannot_measure "G13" "${label} venv ${path} has no pytest installed"
      return 1
    fi
    return 0
  }

  # Prove the two venvs really carry DIFFERENT backends before comparing
  # them. Without this the gate compares a venv with itself and reports a
  # confident PASS having measured nothing -- the exact shape of vacuous
  # pass this script's own history is full of. DT_NEEDED, not `nm`: every
  # shipped extension is stripped, so `nm` on one reads empty and would
  # make this check pass for the wrong reason.
  # All FOUR extensions, not just the tensor one. A venv whose `_tensor.so`
  # is dynamic while `_image.so` is still statically linked would have
  # reported "dynamic" and then compared static against static for the bulk
  # of the suite -- and `tests/image` is where three of P2b's four root
  # causes surfaced. Mixed-provenance wheels in one venv is a demonstrated
  # event on this lane, not a hypothetical: task P2 left a wheel that
  # vendored an ARM `libedgefirst_tensor.so.0` inside an x86_64 wheel.
  # G12 already walks all four; this reuses the same set.
  g13_links_tensor_so() { # g13_links_tensor_so <venv> -> static|dynamic|missing|mixed(...)
    local venv="$1"
    local site tag ext kinds pattern label
    tag=$("${venv}/bin/python" -c 'import sys; print(f"python{sys.version_info.major}.{sys.version_info.minor}")' 2>/dev/null)
    site="${venv}/lib/${tag}/site-packages"
    kinds=""
    for label in tensor image codec decoder; do
      pattern="${site}/edgefirst/${label}/_${label}.cpython-*.so"
      # shellcheck disable=SC2206
      local matches=(${pattern})
      ext="${matches[0]}"
      if [[ ! -s "${ext}" ]]; then echo "missing(${label})"; return; fi
      if readelf -d "${ext}" 2>/dev/null | grep -q 'libedgefirst_tensor.so'; then
        kinds="${kinds}${label}=dynamic "
      else
        kinds="${kinds}${label}=static "
      fi
    done
    # Every extension must agree, or the comparison is partly
    # static-against-static and the number means less than it says.
    case "${kinds}" in
      "tensor=dynamic image=dynamic codec=dynamic decoder=dynamic ") echo "dynamic" ;;
      "tensor=static image=static codec=static decoder=static ") echo "static" ;;
      *) echo "mixed(${kinds%% })" ;;
    esac
    return 0
  }

  # Both venvs must have been built from the CURRENT source, not merely be one
  # of each backend. A venv left over from a previous day still reports the
  # right DT_NEEDED and still runs the suite, so every other check here passes
  # -- while the comparison silently acquires a second variable: elapsed source
  # changes. That is not hypothetical either. The first green this gate ever
  # produced ("0 divergences over 369 tests") was measured with a static venv
  # whose extension predated the dynamic side by sixteen hours of commits to
  # crates/, and nothing said so. A differential gate whose two sides differ in
  # more than the one variable under test is measuring something it cannot name.
  g13_ext_mtime() { # g13_ext_mtime <venv> -> epoch seconds, or empty
    local venv="$1"
    local tag site matches
    tag=$("${venv}/bin/python" -c 'import sys; print(f"python{sys.version_info.major}.{sys.version_info.minor}")' 2>/dev/null)
    site="${venv}/lib/${tag}/site-packages"
    # shellcheck disable=SC2206
    matches=(${site}/edgefirst/tensor/_tensor.cpython-*.so)
    [[ -s "${matches[0]}" ]] && stat -c %Y "${matches[0]}" 2>/dev/null
    return 0
  }

  if g13_check_venv "static" "${g13_static}" && g13_check_venv "dynamic" "${g13_dynamic}"; then
    g13_kind_static=$(g13_links_tensor_so "${g13_static}")
    g13_kind_dynamic=$(g13_links_tensor_so "${g13_dynamic}")
    if [[ "${g13_kind_static}" != "static" ]] || [[ "${g13_kind_dynamic}" != "dynamic" ]]; then
      cannot_measure "G13" "the two venvs are not one of each backend (${g13_static}=${g13_kind_static}, ${g13_dynamic}=${g13_kind_dynamic}) -- comparing a venv with itself would pass having measured nothing"
    elif [[ -n "${g13_src_epoch:=$(git log -1 --format=%ct -- crates/ 2>/dev/null)}" ]] \
      && { g13_stale=""; \
           for g13_v in "${g13_static}" "${g13_dynamic}"; do \
             g13_m=$(g13_ext_mtime "${g13_v}"); \
             [[ -n "${g13_m}" ]] && [[ "${g13_m}" -lt "${g13_src_epoch}" ]] \
               && g13_stale="${g13_stale}${g13_v} ($(date -d "@${g13_m}" '+%b %d %H:%M')) "; \
           done; [[ -n "${g13_stale}" ]]; }; then
      cannot_measure "G13" "venv(s) built before the current source: ${g13_stale}-- newest commit touching crates/ is $(date -d "@${g13_src_epoch}" '+%b %d %H:%M'). Rebuild and reinstall both, or the comparison carries a second variable (elapsed source changes) alongside the backend it means to test"
    else
      # `mktemp -d` failing (a full /tmp -- this host runs against a quota
      # ceiling) left g13_tmp empty, `wc -l` failed, and the count became the
      # empty string. `[ "" -eq 0 ]` does not evaluate false -- it ERRORS with
      # "integer expression expected" and returns 2, so the liveness `if` was
      # false and the comparison ran anyway, printing a pass "over  tests".
      if ! g13_tmp=$(mktemp -d) || [[ ! -d "${g13_tmp}" ]]; then
        cannot_measure "G13" "mktemp -d failed -- cannot stage the two runs (is TMPDIR full?)"
      else
      g13_run "${g13_static}" "${g13_tmp}/static"
      g13_run "${g13_dynamic}" "${g13_tmp}/dynamic"
      g13_rc_static=$(cat "${g13_tmp}/static.rc" 2>/dev/null || echo "")
      g13_rc_dynamic=$(cat "${g13_tmp}/dynamic.rc" 2>/dev/null || echo "")
      g13_n_static=$(wc -l <"${g13_tmp}/static" 2>/dev/null || echo "")
      g13_n_dynamic=$(wc -l <"${g13_tmp}/dynamic" 2>/dev/null || echo "")
      # Exit 0 = all passed, 1 = tests failed (both mean the suite RAN).
      # 2 interrupted, 3 internal error, 4 usage error, 5 nothing collected --
      # none of those is a measurement, and all of them still leave a parseable
      # junitxml behind.
      if [[ "${g13_rc_static}" != "0" ]] && [[ "${g13_rc_static}" != "1" ]]; then
        cannot_measure "G13" "static run did not execute the suite (pytest exit ${g13_rc_static:-unknown}; 4=usage error/bad path, 5=nothing collected) -- see ${g13_tmp}/static.raw"
      elif [[ "${g13_rc_dynamic}" != "0" ]] && [[ "${g13_rc_dynamic}" != "1" ]]; then
        cannot_measure "G13" "dynamic run did not execute the suite (pytest exit ${g13_rc_dynamic:-unknown}; 4=usage error/bad path, 5=nothing collected) -- see ${g13_tmp}/dynamic.raw"
      # Validate as NUMBERS before comparing as numbers. An empty or non-numeric
      # count must not reach `[ -eq ]`, which errors rather than returning false.
      elif ! printf '%s' "${g13_n_static}" | grep -qE '^[0-9]+$' \
        || ! printf '%s' "${g13_n_dynamic}" | grep -qE '^[0-9]+$'; then
        cannot_measure "G13" "could not count outcomes (static='${g13_n_static}' dynamic='${g13_n_dynamic}') -- see ${g13_tmp}/*.raw"
      # A FLOOR, not merely >0. A conftest breakage or a package missing from
      # BOTH venvs collapses the suite into a handful of identical per-FILE
      # ERROR rows, which compare equal and report a confident zero. The
      # missing-package guard cannot see that: it keys on pytest's *skip*
      # message, which a hard collection ImportError never emits.
      elif [[ "${g13_n_static}" -lt "${G13_MIN_TESTS:-200}" ]] || [[ "${g13_n_dynamic}" -lt "${G13_MIN_TESTS:-200}" ]]; then
        cannot_measure "G13" "only ${g13_n_static}/${g13_n_dynamic} outcomes, below the floor of ${G13_MIN_TESTS:-200} -- a collection error collapses the suite into a few per-file ERROR rows that compare equal. Set G13_MIN_TESTS to run a deliberate subset. See ${g13_tmp}/*.raw"
      else
        # One line per DIVERGING TEST, not two (a `comm` of the two outcome
        # lists prints the same nodeid on both sides and doubles the count,
        # which makes the number meaningless as "how many tests disagree").
        # `-a1 -a2 -e MISSING` keeps a test present on only one side: a
        # collection error that drops a whole file is a divergence, not an
        # excuse for it to vanish from the comparison.
        g13_diff=$(LC_ALL=C join -t "$(printf '\t')" -a1 -a2 -e MISSING -o 0,1.2,2.2 \
                     "${g13_tmp}/static" "${g13_tmp}/dynamic" \
                   | awk -F"\t" '$2 != $3 { printf "%s  static=%s dynamic=%s\n", $1, $2, $3 }')
        g13_ndiff=$(printf '%s' "${g13_diff}" | grep -c . || true)
        # An import-skip difference is the two VENVS disagreeing about which
        # optional packages are installed, not the two BACKENDS disagreeing
        # about anything. Reporting it as a divergence is how a gate earns a
        # reputation for crying wolf and stops being read; reporting it as
        # `cannot_measure` names the actual fix. Found by running this gate:
        # six of its first seven "divergences" were a fresh venv missing
        # yaml, safetensors and psutil.
        g13_missing=$(LC_ALL=C comm -3 "${g13_tmp}/static.xml.missing" \
                        "${g13_tmp}/dynamic.xml.missing" 2>/dev/null \
                      | tr -d '\t' | tr '\n' ' ' | sed 's/ *$//')
        # `cannot_measure` and then STOP. This used to fall through to the
        # `gate` call below, so when the package sets differed but no outcome
        # happened to diverge, the last line G13 printed was
        # `PASS 0 outcome(s) differ` -- under a `FAIL cannot verify` line it
        # had already printed. `fails` was right and the sweep was red, but
        # the number that gets quoted is the last one, and it said the
        # opposite of the verdict. Fourth instance on this gate of a figure
        # outliving the caveat that qualified it.
        if [[ -n "${g13_missing}" ]]; then
          cannot_measure "G13" "the two venvs have different optional packages installed (${g13_missing}) -- install them in both, or the import-skips they cause will read as backend divergences"
        else
        # CONFIRM a divergence before reporting it. Timing-sensitive tests --
        # the GIL-release ones measure that a background thread makes progress
        # while a long call runs -- fail under the load of running the whole
        # suite twice, and did so on exactly one of the two sides here,
        # reporting two "divergences" that reproduce on neither backend when
        # re-run. A gate whose green is non-deterministic gets re-run until it
        # is green, which is the same as not having it. Only divergences that
        # survive a second full pass on BOTH sides are real; the rest are
        # reported as flakes so they are visible without being fatal.
        if [[ "${g13_ndiff}" -ne 0 ]]; then
          echo "         ${g13_ndiff} candidate divergence(s); confirming with a second pass"
          g13_run "${g13_static}" "${g13_tmp}/static2"
          g13_run "${g13_dynamic}" "${g13_tmp}/dynamic2"
          g13_diff2=$(LC_ALL=C join -t"$(printf '\t')" -j1 \
                        <(LC_ALL=C sort -t"$(printf '\t')" -k1,1 "${g13_tmp}/static2") \
                        <(LC_ALL=C sort -t"$(printf '\t')" -k1,1 "${g13_tmp}/dynamic2") \
                     | awk -F"\t" '$2 != $3 { printf "%s  static=%s dynamic=%s\n", $1, $2, $3 }')
          g13_confirmed=$(LC_ALL=C comm -12 \
                            <(printf '%s\n' "${g13_diff}" | LC_ALL=C sort) \
                            <(printf '%s\n' "${g13_diff2}" | LC_ALL=C sort) | grep -v '^$' || true)
          g13_flaky=$((g13_ndiff - $(printf '%s' "${g13_confirmed}" | grep -c . || true)))
          [[ "${g13_flaky}" -gt 0 ]] && echo "         ${g13_flaky} did not reproduce (flaky under load, not a backend difference)"
          g13_diff="${g13_confirmed}"
          g13_ndiff=$(printf '%s' "${g13_diff}" | grep -c . || true)
        fi
        if [[ "${g13_ndiff}" -ne 0 ]]; then
          g13_ok=0
          printf '%s\n' "${g13_diff}" | head -40 | sed 's/^/         /'
          [[ "${g13_ndiff}" -gt 40 ]] && echo "         ... and $((g13_ndiff - 40)) more"
        fi
        # The venvs are built from the WORKING TREE, not from HEAD. With
        # uncommitted changes under crates/ the resulting number belongs to no
        # commit and cannot be cited for the branch -- and on a tree with more
        # than one agent in it, the extra source may not even be the author's
        # own. Carried in the RESULT LINE rather than logged separately,
        # because a caveat printed anywhere else travels separately from the
        # number it qualifies, and it is the number that gets quoted.
        g13_dirty=$(git status --porcelain -- crates/ 2>/dev/null | grep -c . || true)
        g13_note=""
        [[ "${g13_dirty}" -gt 0 ]] && g13_note="  [${g13_dirty} uncommitted file(s) under crates/ -- this number belongs to no commit]"
        gate "G13" "${g13_ndiff} outcome(s) differ between backends over ${g13_n_static} tests (want 0)${g13_note}" \
             "$([[ "${g13_ok}" -eq 1 ]] && echo 0 || echo 1)"
        echo "         raw logs: ${g13_tmp}/{static,dynamic}.raw"
        fi # end of the package-mismatch short-circuit
      fi
      fi
    fi
  fi

  if [[ "${DIFFERENTIAL_ONLY}" -eq 1 ]]; then
    echo
    if [[ "${fails}" -eq 0 ]]; then echo "ALL GATES GREEN"; exit 0; fi
    echo "${fails} GATE(S) RED"; exit 1
  fi
fi

echo
if [[ "${fails}" -eq 0 ]]; then echo "ALL GATES GREEN"; exit 0; fi
echo "${fails} GATE(S) RED"; exit 1
