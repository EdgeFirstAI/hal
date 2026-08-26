// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0
//
// The test this whole design exists for: two libraries in one process, and a
// tensor from either is fully usable through the other.
//
// A single library cannot demonstrate this. The hazard — the dynamic linker
// binding every caller to one library's `ef_tensor_free` — only exists once
// there are two.
//
// There used to be a per-library dispatch table and a debug free-counter
// here (`ef_image_debug_free_count`) proving a tensor minted by
// `libedgefirst-image` was freed by `libedgefirst-image`'s own copy of the
// implementation, not `libedgefirst-tensor`'s. Both are gone: all four
// sibling `-capi` leaves now link `libedgefirst_tensor.so` dynamically
// instead of embedding their own copy, so there is exactly one
// implementation of `ef_tensor_free` in the process, and "the wrong
// implementation" is no longer a thing that can happen -- not merely
// something this test happened not to trigger. The equivalent guarantee is
// proven at the symbol table, not at runtime: `edgefirst-image-capi`'s own
// `the_two_libraries_export_no_symbol_in_common` test (`src/lib.rs`) asserts
// `libedgefirst_image.so` exports no `ef_tensor_*` symbol at all, so there is
// nothing here for `ef_tensor_free` to have been misrouted to even in
// principle. What this test still proves, and the only thing left worth
// proving: minting through `libedgefirst-image` and reading/freeing through
// `libedgefirst-tensor`'s real exported accessors actually links and runs.

#include <stdio.h>
#include <string.h>

#include "edgefirst/image.h"

int main(void) {
  int failures = 0;

  ef_image_processor *p = ef_image_processor_new();
  if (p == NULL) {
    fprintf(stderr, "SKIP: no image processor on this host\n");
    return 0;
  }

  // Minted by libedgefirst-image...
  ef_tensor *t = ef_image_processor_create_image(p, 64, 48, "NV12",
                                                 /* U8 */ 0, /* mem */ 0,
                                                 /* ReadWrite */ 3);
  if (t == NULL) {
    fprintf(stderr, "create_image returned NULL\n");
    ef_image_processor_free(p);
    return 1;
  }

  // ...read through the accessors declared in tensor.h. These are ordinary
  // exported functions of libedgefirst_tensor.so -- the single implementation
  // home -- so a tensor minted by any library reads exactly the same way.
  if (ef_tensor_ndim(t) != 2) {
    fprintf(stderr, "ndim: expected 2, got %u\n", ef_tensor_ndim(t));
    failures++;
  }
  if (strcmp(ef_tensor_format(t), "NV12") != 0) {
    fprintf(stderr, "format: expected NV12, got %s\n", ef_tensor_format(t));
    failures++;
  }
  if (ef_tensor_plane_count(t) != 2) {
    fprintf(stderr, "NV12 must report 2 planes, got %u\n",
            ef_tensor_plane_count(t));
    failures++;
  }

  // Reaching another library's planes is the route by which one library
  // consumes a tensor it did not mint.
  ef_tensor_plane pl;
  if (ef_tensor_plane_at(t, 1, &pl) != 0) {
    fprintf(stderr, "plane_at(1) failed on a 2-plane tensor\n");
    failures++;
  } else if (pl.offset == 0) {
    fprintf(stderr, "chroma plane must not start at offset 0\n");
    failures++;
  }
  if (ef_tensor_plane_at(t, 99, &pl) == 0) {
    fprintf(stderr, "an out-of-range plane index must fail\n");
    failures++;
  }

  // The load-bearing part: `ef_tensor_free` lives entirely in
  // libedgefirst-tensor, and so does this tensor's real allocation --
  // `ef_image_processor_create_image` mints through
  // `edgefirst_tensor::TensorDyn`, which under the `dynamic` backend is
  // itself a thin wrapper over the same `ef_tensor_*` calls. There is no
  // second implementation this free could be misrouted to (see this file's
  // header comment); proving that stays a build-time check
  // (`the_two_libraries_export_no_symbol_in_common` in
  // `edgefirst-image-capi/src/lib.rs`), not a runtime one. What this test
  // proves is simpler and just as real: the free actually runs, on a tensor
  // that genuinely crossed the library boundary.
  ef_tensor_free(t);

  // And a tensor from `ef_tensor_new` directly, freed the same way, proving
  // both minting paths land in the one real implementation.
  uint64_t dims[2] = {3, 8};
  ef_tensor *own = ef_tensor_new(/* U8 */ 0, dims, 2);
  if (own == NULL) {
    fprintf(stderr, "ef_tensor_new returned NULL\n");
    failures++;
  } else {
    if (ef_tensor_ndim(own) != 2) {
      fprintf(stderr, "tensor-minted ndim wrong\n");
      failures++;
    }
    ef_tensor_free(own);
  }

  ef_image_processor_free(p);
  return failures == 0 ? 0 : 1;
}
