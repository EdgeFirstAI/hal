#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
# SPDX-License-Identifier: Apache-2.0
#
# Checks that the C libraries and the Python extensions share one copy of the
# tensor implementation. Prints one PASS or FAIL line per check and exits 1
# when any check failed. A check whose input is missing, empty or unreadable
# fails.
#
# Options:
#   --drift               also run the layout-drift test, which patches
#                         crates/tensor/src/lib.rs, rebuilds, and reverts it
#   --drift-only          run only that test
#   --differential        also run the Python suite against a static and a
#                         dynamic venv and compare the outcomes
#   --differential-only   run only that comparison
#   --on-target           skip the footprint, Miri and Python-extension checks
#
# Env:
#   PROFILE           read the libraries from target/<PROFILE>
#   PY_VENV           venv holding the installed extensions (default: venv)
#   PY_STATIC_VENV    venv with static-linked extensions (default: venv)
#   PY_DYNAMIC_VENV   venv with dynamic-linked extensions
#   PY_TEST_PATHS     test paths for the comparison (default: tests)
#   DIFFERENTIAL_MIN_TESTS  fewest outcomes the comparison accepts (default: 200)
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "${ROOT}"

# The symbol and link checks read target/debug and the footprint check reads
# target/release. PROFILE points both at target/<PROFILE>.
PROFILE="${PROFILE:-}"
if [[ -n "${PROFILE}" ]]; then
  DEBUG_LIBDIR="target/${PROFILE}"
  RELEASE_LIBDIR="target/${PROFILE}"
else
  DEBUG_LIBDIR="target/debug"
  RELEASE_LIBDIR="target/release"
fi

SIBS="image codec decoder tracker"
# C libraries with no tensor dependency.
TENSOR_FREE="kernels"
# The Python extensions, by cdylib stem (`_tensor` builds `lib_tensor.so`).
PY_EXTS="tensor image codec decoder tracker"
# Selects the dynamic tensor backend.
TENSOR_DYNAMIC=(--config "${ROOT}/.cargo/tensor-dynamic.toml")

# Prints the sha256 of a shared object's loadable content: the file without
# its debug sections and build-id note.
code_hash() { # code_hash <shared object> -> sha256 of its loadable content
  local so="$1"
  local tmp
  tmp=$(mktemp) || return 1
  if objcopy --strip-debug --remove-section=.note.gnu.build-id "${so}" "${tmp}" 2>/dev/null; then
    sha256sum <"${tmp}"
  fi
  rm -f "${tmp}"
}
fails=0
gate() { # gate <description> <0|1 pass>
  local desc="$1" status="$2"
  if [[ "${status}" -eq 0 ]]; then printf '  PASS  %s\n' "${desc}"
  else printf '  FAIL  %s\n' "${desc}"; fails=$((fails+1)); fi
  return 0
}
cannot_measure() { # cannot_measure <reason>
  local reason="$1"
  printf '  FAIL  cannot verify: %s\n' "${reason}"; fails=$((fails+1))
  return 0
}
header_decls() { # header_decls <header>: print the ef_* function names it declares
  local header="$1"
  # Prints the function names on declaration lines. Comment and typedef
  # lines are dropped; a name is the `ef_*` token followed by `(`.
  grep -vE '^\s*(\*|/\*|//)' "${header}" \
    | grep -v '^typedef' \
    | grep -oE 'ef_[a-z0-9_]+\(' \
    | grep -oE 'ef_[a-z0-9_]+' \
    | sort -u
  return 0
}

DRIFT_MODE=0
DRIFT_ONLY=0
DIFFERENTIAL_MODE=0
DIFFERENTIAL_ONLY=0
ON_TARGET=0
for _arg in "$@"; do
  case "${_arg}" in
    --drift) DRIFT_MODE=1 ;;
    --drift-only) DRIFT_MODE=1; DRIFT_ONLY=1 ;;
    --differential) DIFFERENTIAL_MODE=1 ;;
    --differential-only) DIFFERENTIAL_MODE=1; DIFFERENTIAL_ONLY=1 ;;
    --on-target) ON_TARGET=1 ;;
    *) echo "FAIL: unknown argument ${_arg}" >&2; exit 1 ;;
  esac
done

if [[ "${DRIFT_MODE}" -eq 1 ]]; then
  echo "== siblings do not drift when edgefirst-tensor's own layout changes =="
  drift_lib_rs="crates/tensor/src/lib.rs"
  drift_patched=0
  # Removes the probe field from the source, by exact anchor text.
  drift_revert() {
    if [[ "${drift_patched}" -eq 1 ]]; then
      if python3 - "${drift_lib_rs}" <<'PY'
import sys
path = sys.argv[1]
with open(path) as f:
    src = f.read()

# Each anchor is asserted present before it is replaced.

anchor = (
    "    view_origin: Option<ViewOrigin>,\n"
    "    /// Drift probe, inserted and removed by\n"
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
        drift_patched=0
      else
        echo "  WARN  revert of ${drift_lib_rs} failed -- inspect it by hand for a leftover _drift_probe field before committing anything" >&2
      fi
    fi
    return 0
  }
  # Reverts the source on any exit.
  trap drift_revert EXIT

  if [[ ! -s "${drift_lib_rs}" ]] || [[ ! -r "${drift_lib_rs}" ]]; then
    cannot_measure "${drift_lib_rs} is missing, empty, or unreadable"
  elif command -v git >/dev/null 2>&1 && ! git diff --quiet -- "${drift_lib_rs}" 2>/dev/null; then
    # Refuses to patch a file that has uncommitted changes.
    cannot_measure "${drift_lib_rs} already has uncommitted changes -- refusing to patch and revert over them"
  elif [[ ! -s "${DEBUG_LIBDIR}/libedgefirst_image.so" ]]; then
    cannot_measure "${DEBUG_LIBDIR}/libedgefirst_image.so is missing -- run \`make capi-libs\` first"
  elif [[ ! -s "${DEBUG_LIBDIR}/lib_tensor.so" ]]; then
    cannot_measure "${DEBUG_LIBDIR}/lib_tensor.so is missing -- run \`cargo build --config .cargo/tensor-dynamic.toml -p edgefirst-python-tensor -p edgefirst-python-image -p edgefirst-python-codec -p edgefirst-python-decoder\` first"
  else
    # Hashes the tensor library and the siblings before the probe.
    drift_tensor_before=$(sha256sum "${DEBUG_LIBDIR}/libedgefirst_tensor.so" 2>/dev/null)
    drift_before=$(for l in $SIBS; do sha256sum "${DEBUG_LIBDIR}/libedgefirst_${l}.so" 2>/dev/null; done)
    # Builds the Python extensions from the unpatched source and hashes their
    # loadable content.
    drift_py_baseline_log="${TMPDIR:-/tmp}/check-single-home-drift-baseline.$$.log"
    if ! cargo build "${TENSOR_DYNAMIC[@]}" $(for l in $PY_EXTS; do printf -- "-p edgefirst-python-%s " "$l"; done) \
         >"${drift_py_baseline_log}" 2>&1; then
      cannot_measure "could not build the Python extensions' clean baseline -- see ${drift_py_baseline_log}"
      drift_py_before=""
    else
      drift_py_before=$(for l in $PY_EXTS; do code_hash "${DEBUG_LIBDIR}/lib_${l}.so"; done)
    fi
    rm -f "${drift_py_baseline_log}"
    python3 - "${drift_lib_rs}" <<'PY'
import sys
path = sys.argv[1]
with open(path) as f:
    src = f.read()
assert src.count("    view_origin: Option<ViewOrigin>,\n}\n") == 1, "struct anchor not unique"
src = src.replace(
    "    view_origin: Option<ViewOrigin>,\n}\n",
    "    view_origin: Option<ViewOrigin>,\n"
    "    /// Drift probe, inserted and removed by\n"
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
      cannot_measure "could not locate Tensor<T>'s struct/constructors in ${drift_lib_rs} to patch (anchors moved -- update this script)"
    else
      drift_patched=1
      drift_build_log="${TMPDIR:-/tmp}/check-single-home-drift-build.$$.log"
      # Rebuilds libedgefirst_tensor.so with the probe, then the Python
      # extensions. The C siblings are not rebuilt.
      if ! cargo build --manifest-path crates/tensor-capi/Cargo.toml --target-dir target >"${drift_build_log}" 2>&1; then
        cannot_measure "rebuilding libedgefirst_tensor.so with the probe field failed -- see ${drift_build_log}"
      elif [[ -z "${drift_py_before}" ]]; then
        :
      elif ! cargo build "${TENSOR_DYNAMIC[@]}" $(for l in $PY_EXTS; do printf -- "-p edgefirst-python-%s " "$l"; done) \
             >>"${drift_build_log}" 2>&1; then
        cannot_measure "rebuilding the four Python extensions with the probe field failed -- see ${drift_build_log}"
      else
        drift_tensor_after=$(sha256sum "${DEBUG_LIBDIR}/libedgefirst_tensor.so" 2>/dev/null)
        drift_after=$(for l in $SIBS; do sha256sum "${DEBUG_LIBDIR}/libedgefirst_${l}.so" 2>/dev/null; done)
        drift_py_after=$(for l in $PY_EXTS; do code_hash "${DEBUG_LIBDIR}/lib_${l}.so"; done)
        # The tensor library must differ across the rebuild.
        if [[ "${drift_tensor_before}" = "${drift_tensor_after}" ]]; then
          cannot_measure "libedgefirst_tensor.so did not change after the rebuild -- the probe never took effect, so byte-identical siblings would prove nothing (see ${drift_build_log})"
        elif [[ "${drift_before}" != "${drift_after}" ]]; then
          gate "a sibling's bytes changed when only edgefirst-tensor's private layout changed -- see ${drift_build_log}" 1
        # The build log must show edgefirst-tensor recompiled for the
        # extensions.
        elif ! grep -q "Compiling edgefirst-tensor " "${drift_build_log}"; then
          cannot_measure "the Python extension rebuild did not recompile edgefirst-tensor, so byte-identical extensions would prove nothing (see ${drift_build_log})"
        elif [[ -z "${drift_py_before}" ]] || [[ -z "${drift_py_after}" ]]; then
          cannot_measure "could not hash the Python extensions' loadable content (is objcopy available?)"
        elif [[ "${drift_py_before}" != "${drift_py_after}" ]]; then
          gate "a Python extension's loadable content changed when only edgefirst-tensor's private layout changed (see ${drift_build_log})" 1
        else
          gate "libedgefirst_tensor.so changed (rebuild confirmed); the four C siblings are byte-identical and the four Python extensions' loadable content is identical across a recompile of edgefirst-tensor" 0
        fi
      fi
      rm -f "${drift_build_log}"
    fi
    drift_revert
    # Disarms the EXIT trap once the revert succeeded.
    if [[ "${drift_patched}" -eq 0 ]]; then
      trap - EXIT
    fi
  fi
  echo
  if [[ "${DRIFT_ONLY}" -eq 1 ]]; then
    if [[ "${fails}" -eq 0 ]]; then echo "ALL GATES GREEN"; exit 0; fi
    echo "${fails} GATE(S) RED"; exit 1
  fi
fi

# --differential-only skips every check from here to the comparison. The
# guarded body is not indented.
if [[ "${DIFFERENTIAL_ONLY}" -eq 0 ]]; then

echo "== no sibling embeds edgefirst-tensor's implementation =="
# Counts `static_backend` symbols, the tensor implementation's own, in each
# sibling's static symbol table (plain `nm`). cuda.rs is not among them and
# is measured by the next check. A library without a symbol table fails.
for l in $SIBS; do
  so="${DEBUG_LIBDIR}/libedgefirst_${l}.so"
  if [[ ! -s "${so}" ]] || [[ ! -r "${so}" ]]; then
    cannot_measure "${so} is missing, empty, or unreadable"
    continue
  fi
  syms=$(nm "${so}" 2>/dev/null)
  if [[ -z "${syms}" ]]; then
    cannot_measure "${so} has no symbol table (stripped, or unreadable)"
    continue
  fi
  n=$(printf '%s\n' "${syms}" | grep -c 'static_backend' || true)
  gate "libedgefirst_${l}.so carries ${n} static_backend (embedded implementation) symbols (want 0)" \
       "$([[ "${n}" -eq 0 ]] && echo 0 || echo 1)"
done

echo "== size of cuda.rs's code in each library =="
# Sums the size of cuda.rs's own functions in each library. Always passes;
# the byte figure is the result. CUDA_MARKER matches a symbol by the start
# of its name, so a generic instantiated over a cuda type does not count.
CUDA_MARKER='^[0-9a-f]+ +[0-9a-f]+ +[a-zA-Z] +<?edgefirst_tensor\[[0-9a-f]+\]::cuda::(load(::\{closure#0\})?|is_cuda_available|memcpy_device_to_host|stream_create|stream_destroy|stream_synchronize|gl_map_resource|gl_register_buffer|gl_unmap_resource|gl_unregister_resource|import_dma_fd|table|TABLE|CudaHandle(>::(new_gl|new_external|map)| as core\[[0-9a-f]+\]::(ops::drop::Drop>::drop|fmt::Debug>::fmt))|CudaMap(>::(device_ptr|len|is_empty)| as core\[[0-9a-f]+\]::ops::drop::Drop>::drop))$'
for l in tensor $SIBS; do
  so="${DEBUG_LIBDIR}/libedgefirst_${l}.so"
  if [[ ! -s "${so}" ]] || [[ ! -r "${so}" ]]; then
    cannot_measure "${so} is missing, empty, or unreadable"
    continue
  fi
  syms=$(nm -S "${so}" 2>/dev/null | c++filt 2>/dev/null)
  if [[ -z "${syms}" ]]; then
    cannot_measure "${so} has no symbol table (stripped, or unreadable)"
    continue
  fi
  hits=$(printf '%s\n' "${syms}" | grep -cE "${CUDA_MARKER}" || true)
  bytes=$(printf '%s\n' "${syms}" | grep -E "${CUDA_MARKER}" | awk '{sum+=strtonum("0x"$2)} END{print sum+0}')
  gate "libedgefirst_${l}.so duplicates ${bytes} B (${hits} symbols) of cuda.rs's own dlopen implementation" 0
done

echo "== each sibling links libedgefirst_tensor.so =="
# A sibling passes with a DT_NEEDED entry for libedgefirst_tensor.so and at
# least one undefined ef_tensor_* symbol, or with neither and no
# `static_backend` symbol.
for l in $SIBS; do
  so="${DEBUG_LIBDIR}/libedgefirst_${l}.so"
  if [[ ! -s "${so}" ]] || [[ ! -r "${so}" ]]; then
    cannot_measure "${so} is missing, empty, or unreadable"
    continue
  fi
  syms=$(nm "${so}" 2>/dev/null)
  if [[ -z "${syms}" ]]; then
    cannot_measure "${so} has no symbol table (stripped, or unreadable)"
    continue
  fi
  embeds_static=$(printf '%s\n' "${syms}" | grep -c 'static_backend' || true)
  need=$(readelf -d "${so}" 2>/dev/null | grep -c 'libedgefirst_tensor.so' || true)
  und=$(nm -D -u "${so}" 2>/dev/null | grep -c ' ef_tensor_' || true)
  if [[ "${need}" -ge 1 ]] && [[ "${und}" -ge 1 ]]; then
    gate "libedgefirst_${l}.so: DT_NEEDED=${need} undefined ef_tensor_*=${und}" 0
  elif [[ "${embeds_static}" -eq 0 ]] && [[ "${need}" -eq 0 ]] && [[ "${und}" -eq 0 ]]; then
    gate "libedgefirst_${l}.so: no embedded implementation (static_backend=0) and no ef_tensor_* reference -- this leaf needs no tensor primitive" 0
  else
    gate "libedgefirst_${l}.so: DT_NEEDED=${need} undefined ef_tensor_*=${und}, static_backend=${embeds_static}" 1
  fi
done

echo "== tensor-free libraries link no sibling library and no ef_tensor_* symbol =="
# A library with no DT_NEEDED entry at all fails as unreadable.
for l in ${TENSOR_FREE}; do
  so="${DEBUG_LIBDIR}/libedgefirst_${l}.so"
  if [[ ! -s "${so}" ]] || [[ ! -r "${so}" ]]; then
    cannot_measure "${so} is missing, empty, or unreadable"
    continue
  fi
  all_need=$(readelf -d "${so}" 2>/dev/null | grep -c '(NEEDED)' || true)
  if [[ "${all_need}" -eq 0 ]]; then
    cannot_measure "${so}: readelf found no DT_NEEDED entries at all"
    continue
  fi
  need=$(readelf -d "${so}" 2>/dev/null | grep '(NEEDED)' | grep -c 'libedgefirst_' || true)
  und=$(nm -D -u "${so}" 2>/dev/null | grep -c ' ef_tensor_' || true)
  gate "libedgefirst_${l}.so: DT_NEEDED on libedgefirst_*=${need} undefined ef_tensor_*=${und} (want 0 and 0)" \
       "$([[ "${need}" -eq 0 ]] && [[ "${und}" -eq 0 ]] && echo 0 || echo 1)"
done

echo "== the transition vtable is gone =="
# Counts the .rs files outside target/ and .git that mention EfTensorVtable
# or is_own_mint. crates/tensor/src/pbo.rs, which declares the unrelated
# PboOpsVtable, is excluded. A tree with no .rs files fails.
src_file_count=$(find . -path ./target -prune -o -path ./.git -prune -o \
  -type f -name '*.rs' -print 2>/dev/null | wc -l)
if [[ "${src_file_count}" -eq 0 ]]; then
  cannot_measure "repo-wide search found no .rs files"
else
  v=$(find . -path ./target -prune -o -path ./.git -prune -o \
    -type f -name '*.rs' -print 2>/dev/null \
    | grep -v '^\./crates/tensor/src/pbo\.rs$' \
    | xargs -r grep -l 'EfTensorVtable\|is_own_mint' 2>/dev/null | wc -l)
  gate "${v} files repo-wide (excluding crates/tensor/src/pbo.rs's unrelated PboOpsVtable) still reference EfTensorVtable/is_own_mint (want 0)" \
       "$([[ "${v}" -eq 0 ]] && echo 0 || echo 1)"
fi

# Skipped with --on-target.
if [[ "${ON_TARGET}" -eq 0 ]]; then
echo "== footprint =="
# Builds the five libraries in release and fails when their total size
# exceeds CEILING_BYTES.
CEILING_BYTES=23420000

footprint_unmeasurable=""
footprint_provenance=""
if [[ -n "${PROFILE}" ]]; then
  # The check builds --release only, so it does not run with PROFILE set.
  footprint_unmeasurable="PROFILE=${PROFILE} is set, and the footprint check builds --release only -- unset PROFILE to run it"
else
  # Builds each leaf by its own manifest; the dynamic ones take the
  # tensor-dynamic config.
  footprint_build_log="${TMPDIR:-/tmp}/check-single-home-footprint-build.$$.log"
  : >"${footprint_build_log}"
  for l in tensor $SIBS; do
    leaf_cfg=()
    case "${l}" in image|codec|decoder) leaf_cfg=("${TENSOR_DYNAMIC[@]}") ;; esac
    if ! cargo build --release ${leaf_cfg[@]+"${leaf_cfg[@]}"} --manifest-path "crates/${l}-capi/Cargo.toml" \
         --target-dir target >>"${footprint_build_log}" 2>&1; then
      footprint_unmeasurable="crates/${l}-capi failed to build at HEAD -- see ${footprint_build_log}"
      break
    fi
  done
  [[ -z "${footprint_unmeasurable}" ]] && rm -f "${footprint_build_log}"
  footprint_provenance="built at HEAD"
fi

footprint_sizes=""
tot=0
if [[ -z "${footprint_unmeasurable}" ]]; then
  for l in tensor $SIBS; do
    f="${RELEASE_LIBDIR}/libedgefirst_${l}.so"
    # A missing or empty library fails the check.
    if [[ ! -s "${f}" ]]; then
      footprint_unmeasurable="${f} is missing or empty even though crates/${l}-capi built successfully -- artifact path mismatch, not a build failure"
      break
    fi
    s=$(stat -c%s "${f}" 2>/dev/null)
    if [[ -z "${s}" ]]; then
      footprint_unmeasurable="${f} size could not be read"
      break
    fi
    tot=$((tot+s))
    footprint_sizes="${footprint_sizes}        ${l} ${s} B"$'\n'
  done
fi

if [[ -n "${footprint_unmeasurable}" ]]; then
  cannot_measure "${footprint_unmeasurable}"
else
  printf '%s' "${footprint_sizes}"
  gate "five-library total ${tot} B (${footprint_provenance}; ceiling ${CEILING_BYTES} B)" \
       "$([[ "${tot}" -le "${CEILING_BYTES}" ]] && echo 0 || echo 1)"
fi

echo "== dynamic backend aliasing shapes are clean under Miri (both models) =="
# Runs scripts/miri.sh, which checks crates/tensor/tests/scenarios.rs under
# Stacked Borrows and Tree Borrows. Exit 2 from it means Miri is not
# installed, which fails as unverifiable.
if [[ ! -x scripts/miri.sh ]]; then
  cannot_measure "scripts/miri.sh is missing or not executable"
else
  miri_log="${TMPDIR:-/tmp}/check-single-home-miri.$$.log"
  ./scripts/miri.sh >"${miri_log}" 2>&1
  miri_rc=$?
  if [[ "${miri_rc}" -eq 2 ]]; then
    cannot_measure "Miri toolchain unavailable -- $(tail -n1 "${miri_log}")"
  else
    gate "scripts/miri.sh: both aliasing models (rc=${miri_rc}; re-run ./scripts/miri.sh for detail)" \
         "$([[ "${miri_rc}" -eq 0 ]] && echo 0 || echo 1)"
  fi
  rm -f "${miri_log}"
fi

fi # end of the --on-target skip

echo "== detections are owned by decoder =="
tensor_so="${DEBUG_LIBDIR}/libedgefirst_tensor.so"
decoder_so="${DEBUG_LIBDIR}/libedgefirst_decoder.so"
tensor_hdr="crates/tensor-capi/include/edgefirst/tensor.h"
detect_unmeasurable=""
# Fails as unverifiable when either library exports no dynamic symbols or
# tensor.h yields no ef_* declaration.
if [[ ! -s "${tensor_so}" ]] || [[ ! -r "${tensor_so}" ]]; then
  detect_unmeasurable="${tensor_so} is missing, empty, or unreadable"
elif [[ -z "$(nm -D --defined-only "${tensor_so}" 2>/dev/null)" ]]; then
  detect_unmeasurable="${tensor_so} exports no dynamic symbols at all"
elif [[ ! -s "${decoder_so}" ]] || [[ ! -r "${decoder_so}" ]]; then
  detect_unmeasurable="${decoder_so} is missing, empty, or unreadable"
elif [[ -z "$(nm -D --defined-only "${decoder_so}" 2>/dev/null)" ]]; then
  detect_unmeasurable="${decoder_so} exports no dynamic symbols at all"
elif [[ ! -s "${tensor_hdr}" ]] || [[ ! -r "${tensor_hdr}" ]]; then
  detect_unmeasurable="${tensor_hdr} is missing, empty, or unreadable"
elif [[ "$(header_decls "${tensor_hdr}" | wc -l)" -eq 0 ]]; then
  detect_unmeasurable="${tensor_hdr}: extraction found 0 declarations (expected >= 1)"
fi
if [[ -n "${detect_unmeasurable}" ]]; then
  cannot_measure "${detect_unmeasurable}"
else
  t=$(nm -D --defined-only "${tensor_so}" 2>/dev/null | grep -c 'ef_detect_box_list' || true)
  d=$(nm -D --defined-only "${decoder_so}" 2>/dev/null | grep -c 'ef_detect_box_list' || true)
  h=$(grep -c 'ef_detect_box' "${tensor_hdr}" || true)
  gate "tensor.so exports ${t} (want 0), decoder.so exports ${d} (want 6), tensor.h mentions ${h} (want 0)" \
       "$([[ "${t}" -eq 0 ]] && [[ "${d}" -eq 6 ]] && [[ "${h}" -eq 0 ]] && echo 0 || echo 1)"
fi

echo "== header/library alignment =="
for l in tensor image codec decoder tracker kernels; do
  hdr="crates/${l}-capi/include/edgefirst/${l}.h"
  so="${DEBUG_LIBDIR}/libedgefirst_${l}.so"
  if [[ ! -s "${hdr}" ]] || [[ ! -r "${hdr}" ]]; then
    cannot_measure "${hdr} is missing, empty, or unreadable"
    continue
  fi
  if [[ ! -s "${so}" ]] || [[ ! -r "${so}" ]]; then
    cannot_measure "${so} is missing, empty, or unreadable"
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
  # A header that yields no declaration fails as unverifiable.
  if [[ "${decl_count}" -eq 0 ]]; then
    cannot_measure "${hdr}: extraction found 0 declarations (expected >= 1)"
    continue
  fi
  gate "${l}.h declares ${bad} function(s) libedgefirst_${l}.so does not export (want 0)" \
       "$([[ "${bad}" -eq 0 ]] && echo 0 || echo 1)"
done

# Skipped with --on-target.
if [[ "${ON_TARGET}" -eq 0 ]]; then
echo "== Python extensions link libedgefirst_tensor.so =="
# Each extension installed in PY_VENV must have a DT_NEEDED entry for
# libedgefirst_tensor.so and at least one undefined ef_tensor_* symbol; the
# tracker extension must have neither.
PY_VENV="${PY_VENV:-venv}"
if [[ ! -x "${PY_VENV}/bin/python" ]]; then
  cannot_measure "${PY_VENV}/bin/python not found -- build/install the Python extensions into the venv first (never a system Python)"
else
  # Asks the venv's interpreter for its site-packages directory.
  py_tag=$("${PY_VENV}/bin/python" -c 'import sys; print(f"python{sys.version_info.major}.{sys.version_info.minor}")' 2>/dev/null)
  py_site="${PY_VENV}/lib/${py_tag}/site-packages"
  if [[ -z "${py_tag}" ]] || [[ ! -d "${py_site}" ]]; then
    cannot_measure "could not resolve ${PY_VENV}'s site-packages directory (expected ${PY_VENV}/lib/python3.X/site-packages)"
  else
    # An extension is installed as `_<name>.abi3.so` or
    # `_<name>.cpython-*.so` under edgefirst/<name>/; exactly one must exist.
    py_targets="image tensor codec decoder tracker"
    # Reports an installed `edgefirst_hal` package without failing.
    if [[ -d "${py_site}/edgefirst_hal" ]]; then
      printf '  WARN  %s\n' "${py_site}/edgefirst_hal is an install of the retired edgefirst-hal package; \`pip uninstall edgefirst-hal\` in this venv"
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
        cannot_measure "no installed extension matching ${py_site}/edgefirst/${label}/_${label}.abi3.so or _${label}.cpython-*.so -- build/install it into the venv first"
        continue
      fi
      if [[ "${#matches[@]}" -gt 1 ]]; then
        cannot_measure "${#matches[@]} files matched edgefirst/${label}/_${label}.{abi3,cpython-*}.so, expected exactly one -- stale build in the venv?"
        continue
      fi
      so="${matches[0]}"
      if [[ ! -s "${so}" ]] || [[ ! -r "${so}" ]]; then
        cannot_measure "${so} is missing, empty, or unreadable"
        continue
      fi
      need=$(readelf -d "${so}" 2>/dev/null | grep -c 'libedgefirst_tensor.so' || true)
      und=$(nm -D -u "${so}" 2>/dev/null | grep -c ' ef_tensor_' || true)
      if [[ "${label}" = "tracker" ]]; then
        gate "${label} (${so#"${py_site}"/}): DT_NEEDED=${need} undefined ef_tensor_*=${und} (tracker must not link tensor)" \
             "$([[ "${need}" -eq 0 ]] && [[ "${und}" -eq 0 ]] && echo 0 || echo 1)"
      else
        gate "${label} (${so#"${py_site}"/}): DT_NEEDED=${need} undefined ef_tensor_*=${und}" \
             "$([[ "${need}" -ge 1 ]] && [[ "${und}" -ge 1 ]] && echo 0 || echo 1)"
      fi
    done
  fi
fi

fi # end of the --on-target skip

# Runs the Python suite against a static and a dynamic venv and requires
# identical per-test outcomes. A missing or unusable venv fails.
fi # end of the --differential-only skip

if [[ "${DIFFERENTIAL_MODE}" -eq 1 ]]; then
  echo "== static and dynamic backends agree, test for test =="
  cmp_static="${PY_STATIC_VENV:-venv}"
  cmp_dynamic="${PY_DYNAMIC_VENV:-}"
  cmp_tests="${PY_TEST_PATHS:-tests}"
  cmp_ok=1

  # Runs the suite in a venv and writes one `nodeid<TAB>outcome` line per
  # test, read from pytest's junitxml.
  run_suite() { # run_suite <venv> <outfile>
    local venv="$1"
    local outfile="$2"
    # Records pytest's exit code in <outfile>.rc.
    local static_comparison=""
    [[ "${venv}" == "${cmp_static}" ]] && static_comparison=1
    if EDGEFIRST_TENSOR_STATIC_COMPARISON="${static_comparison}" "${venv}/bin/python" -m pytest "${cmp_tests}" --tb=no -q -p no:cacheprovider \
      --junitxml="${outfile}.xml" >"${outfile}.raw" 2>&1; then echo 0 >"${outfile}.rc"; else echo "$?" >"${outfile}.rc"; fi
    "${venv}/bin/python" - "${outfile}.xml" >"${outfile}" <<'PARSE_JUNIT' || true
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
                # A skip caused by a missing package is recorded in
                # <xml>.missing.
                m = re.search(r"could not import '([^']+)'", child.get("message", "") or "")
                if m:
                    missing.add(m.group(1))
            break
    rows.append("{}\t{}".format(node, outcome))
# Writes nothing when there are no rows.
if rows:
    print("\n".join(sorted(set(rows))))
with open(sys.argv[1] + ".missing", "w") as fh:
    fh.write("\n".join(sorted(missing)))
PARSE_JUNIT
    return 0
  }

  check_venv() { # check_venv <label> <path>
    local label="$1"
    local path="$2"
    if [[ -z "${path}" ]]; then
      cannot_measure "${label} venv not supplied -- set PY_DYNAMIC_VENV to a venv with dynamic-linked extensions installed (never a system Python)"
      return 1
    fi
    if [[ ! -x "${path}/bin/python" ]]; then
      cannot_measure "${label} venv ${path}/bin/python not found -- build/install the extensions into it first"
      return 1
    fi
    if ! "${path}/bin/python" -c 'import pytest' >/dev/null 2>&1; then
      cannot_measure "${label} venv ${path} has no pytest installed"
      return 1
    fi
    return 0
  }

  # Prints `static` or `dynamic` when all four extensions of a venv agree on
  # having a DT_NEEDED entry for libedgefirst_tensor.so, else `mixed(...)`
  # or `missing(...)`.
  venv_backend() { # venv_backend <venv> -> static|dynamic|missing|mixed(...)
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
    case "${kinds}" in
      "tensor=dynamic image=dynamic codec=dynamic decoder=dynamic ") echo "dynamic" ;;
      "tensor=static image=static codec=static decoder=static ") echo "static" ;;
      *) echo "mixed(${kinds%% })" ;;
    esac
    return 0
  }

  # Prints the modification time of a venv's tensor extension.
  venv_ext_mtime() { # venv_ext_mtime <venv> -> epoch seconds, or empty
    local venv="$1"
    local tag site matches
    tag=$("${venv}/bin/python" -c 'import sys; print(f"python{sys.version_info.major}.{sys.version_info.minor}")' 2>/dev/null)
    site="${venv}/lib/${tag}/site-packages"
    # shellcheck disable=SC2206
    matches=(${site}/edgefirst/tensor/_tensor.cpython-*.so)
    [[ -s "${matches[0]}" ]] && stat -c %Y "${matches[0]}" 2>/dev/null
    return 0
  }

  if check_venv "static" "${cmp_static}" && check_venv "dynamic" "${cmp_dynamic}"; then
    cmp_kind_static=$(venv_backend "${cmp_static}")
    cmp_kind_dynamic=$(venv_backend "${cmp_dynamic}")
    if [[ "${cmp_kind_static}" != "static" ]] || [[ "${cmp_kind_dynamic}" != "dynamic" ]]; then
      cannot_measure "the two venvs are not one of each backend (${cmp_static}=${cmp_kind_static}, ${cmp_dynamic}=${cmp_kind_dynamic}) -- comparing a venv with itself would pass having measured nothing"
    elif [[ -n "${cmp_src_epoch:=$(git log -1 --format=%ct -- crates/ 2>/dev/null)}" ]] \
      && { cmp_stale=""; \
           for cmp_v in "${cmp_static}" "${cmp_dynamic}"; do \
             cmp_m=$(venv_ext_mtime "${cmp_v}"); \
             [[ -n "${cmp_m}" ]] && [[ "${cmp_m}" -lt "${cmp_src_epoch}" ]] \
               && cmp_stale="${cmp_stale}${cmp_v} ($(date -d "@${cmp_m}" '+%b %d %H:%M')) "; \
           done; [[ -n "${cmp_stale}" ]]; }; then
      cannot_measure "venv(s) built before the current source: ${cmp_stale}-- newest commit touching crates/ is $(date -d "@${cmp_src_epoch}" '+%b %d %H:%M'). Rebuild and reinstall both, or the comparison carries a second variable (elapsed source changes) alongside the backend it means to test"
    else
      if ! cmp_tmp=$(mktemp -d) || [[ ! -d "${cmp_tmp}" ]]; then
        cannot_measure "mktemp -d failed -- cannot stage the two runs (is TMPDIR full?)"
      else
      run_suite "${cmp_static}" "${cmp_tmp}/static"
      run_suite "${cmp_dynamic}" "${cmp_tmp}/dynamic"
      cmp_rc_static=$(cat "${cmp_tmp}/static.rc" 2>/dev/null || echo "")
      cmp_rc_dynamic=$(cat "${cmp_tmp}/dynamic.rc" 2>/dev/null || echo "")
      cmp_n_static=$(wc -l <"${cmp_tmp}/static" 2>/dev/null || echo "")
      cmp_n_dynamic=$(wc -l <"${cmp_tmp}/dynamic" 2>/dev/null || echo "")
      # pytest exit codes 0 and 1 mean the suite ran; any other fails as
      # unverifiable.
      if [[ "${cmp_rc_static}" != "0" ]] && [[ "${cmp_rc_static}" != "1" ]]; then
        cannot_measure "static run did not execute the suite (pytest exit ${cmp_rc_static:-unknown}; 4=usage error/bad path, 5=nothing collected) -- see ${cmp_tmp}/static.raw"
      elif [[ "${cmp_rc_dynamic}" != "0" ]] && [[ "${cmp_rc_dynamic}" != "1" ]]; then
        cannot_measure "dynamic run did not execute the suite (pytest exit ${cmp_rc_dynamic:-unknown}; 4=usage error/bad path, 5=nothing collected) -- see ${cmp_tmp}/dynamic.raw"
      # The counts must be numbers.
      elif ! printf '%s' "${cmp_n_static}" | grep -qE '^[0-9]+$' \
        || ! printf '%s' "${cmp_n_dynamic}" | grep -qE '^[0-9]+$'; then
        cannot_measure "could not count outcomes (static='${cmp_n_static}' dynamic='${cmp_n_dynamic}') -- see ${cmp_tmp}/*.raw"
      # Fewer outcomes than DIFFERENTIAL_MIN_TESTS on either side fails as unverifiable.
      elif [[ "${cmp_n_static}" -lt "${DIFFERENTIAL_MIN_TESTS:-200}" ]] || [[ "${cmp_n_dynamic}" -lt "${DIFFERENTIAL_MIN_TESTS:-200}" ]]; then
        cannot_measure "only ${cmp_n_static}/${cmp_n_dynamic} outcomes, below the floor of ${DIFFERENTIAL_MIN_TESTS:-200} -- a collection error collapses the suite into a few per-file ERROR rows that compare equal. Set DIFFERENTIAL_MIN_TESTS to run a deliberate subset. See ${cmp_tmp}/*.raw"
      else
        # One line per test whose outcomes differ. A test present on one side
        # only is reported as MISSING on the other.
        cmp_diff=$(LC_ALL=C join -t "$(printf '\t')" -a1 -a2 -e MISSING -o 0,1.2,2.2 \
                     "${cmp_tmp}/static" "${cmp_tmp}/dynamic" \
                   | awk -F"\t" '$2 != $3 { printf "%s  static=%s dynamic=%s\n", $1, $2, $3 }')
        cmp_ndiff=$(printf '%s' "${cmp_diff}" | grep -c . || true)
        # Packages missing from one venv only.
        cmp_missing=$(LC_ALL=C comm -3 "${cmp_tmp}/static.xml.missing" \
                        "${cmp_tmp}/dynamic.xml.missing" 2>/dev/null \
                      | tr -d '\t' | tr '\n' ' ' | sed 's/ *$//')
        if [[ -n "${cmp_missing}" ]]; then
          cannot_measure "the two venvs have different optional packages installed (${cmp_missing}) -- install them in both, or the import-skips they cause will read as backend divergences"
        else
        # Re-runs both sides when outcomes differ and keeps the differences
        # that occur in both passes; the rest are reported as flaky.
        if [[ "${cmp_ndiff}" -ne 0 ]]; then
          echo "         ${cmp_ndiff} candidate divergence(s); confirming with a second pass"
          run_suite "${cmp_static}" "${cmp_tmp}/static2"
          run_suite "${cmp_dynamic}" "${cmp_tmp}/dynamic2"
          cmp_diff2=$(LC_ALL=C join -t"$(printf '\t')" -j1 \
                        <(LC_ALL=C sort -t"$(printf '\t')" -k1,1 "${cmp_tmp}/static2") \
                        <(LC_ALL=C sort -t"$(printf '\t')" -k1,1 "${cmp_tmp}/dynamic2") \
                     | awk -F"\t" '$2 != $3 { printf "%s  static=%s dynamic=%s\n", $1, $2, $3 }')
          cmp_confirmed=$(LC_ALL=C comm -12 \
                            <(printf '%s\n' "${cmp_diff}" | LC_ALL=C sort) \
                            <(printf '%s\n' "${cmp_diff2}" | LC_ALL=C sort) | grep -v '^$' || true)
          cmp_flaky=$((cmp_ndiff - $(printf '%s' "${cmp_confirmed}" | grep -c . || true)))
          [[ "${cmp_flaky}" -gt 0 ]] && echo "         ${cmp_flaky} did not reproduce (flaky under load, not a backend difference)"
          cmp_diff="${cmp_confirmed}"
          cmp_ndiff=$(printf '%s' "${cmp_diff}" | grep -c . || true)
        fi
        if [[ "${cmp_ndiff}" -ne 0 ]]; then
          cmp_ok=0
          printf '%s\n' "${cmp_diff}" | head -40 | sed 's/^/         /'
          [[ "${cmp_ndiff}" -gt 40 ]] && echo "         ... and $((cmp_ndiff - 40)) more"
        fi
        # The result line notes uncommitted changes under crates/.
        cmp_dirty=$(git status --porcelain -- crates/ 2>/dev/null | grep -c . || true)
        cmp_note=""
        [[ "${cmp_dirty}" -gt 0 ]] && cmp_note="  [${cmp_dirty} uncommitted file(s) under crates/ -- this number belongs to no commit]"
        gate "${cmp_ndiff} outcome(s) differ between backends over ${cmp_n_static} tests (want 0)${cmp_note}" \
             "$([[ "${cmp_ok}" -eq 1 ]] && echo 0 || echo 1)"
        echo "         raw logs: ${cmp_tmp}/{static,dynamic}.raw"
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
