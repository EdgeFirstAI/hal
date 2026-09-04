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

#include <errno.h>
#include <stdio.h>
#include <string.h>
#ifdef _WIN32
#include <windows.h>
#endif

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

  // ef_image_processor_convert_fence_handle is declared on every platform
  // (see image.h's "Platform-specific entry points" paragraph): Windows
  // hands back an owned event, everywhere else it refuses with ENOTSUP --
  // the same declared-everywhere rule test_d3d11.c pins for the tensor
  // leaf's D3D11 family.
  //
  // `mem` has no dedicated "auto" enumerator: code 0 is EF_STORAGE_KIND_MEM,
  // a real request for host memory, not "unspecified". UINT32_MAX is
  // outside the 0-5 vocabulary, so create_image() treats it as "let the
  // allocator choose" -- the Dma-first arm that hands back a D3D11 texture
  // on Windows, which is what this fenced convert needs to be meaningful.
  ef_tensor *fsrc = ef_image_processor_create_image(p, 64, 48, "rgba8",
                                                    /* U8 */ 0,
                                                    /* mem: auto */ UINT32_MAX,
                                                    /* ReadWrite */ 3);
  ef_tensor *fdst = ef_image_processor_create_image(p, 64, 48, "rgb8",
                                                    /* U8 */ 0,
                                                    /* mem: auto */ UINT32_MAX,
                                                    /* ReadWrite */ 3);
  if (fsrc == NULL || fdst == NULL) {
    fprintf(stderr, "convert_fence_handle: create_image returned NULL\n");
    failures++;
  } else {
    void *fence = NULL;
    int rc = ef_image_processor_convert_fence_handle(p, fsrc, fdst, 0, 0,
                                                      NULL, &fence);
#ifdef _WIN32
    if (rc != 0) {
      fprintf(stderr, "convert_fence_handle: expected 0 on Windows, got %d\n",
              rc);
      failures++;
    } else if (fence == NULL) {
      fprintf(stderr, "convert_fence_handle: expected an owned event on Windows\n");
      failures++;
    } else if (WaitForSingleObject(fence, 5000) != WAIT_OBJECT_0) {
      fprintf(stderr, "convert_fence_handle: WaitForSingleObject timed out\n");
      failures++;
      CloseHandle(fence);
    } else {
      CloseHandle(fence);
    }
#else
    if (rc != ENOTSUP) {
      fprintf(stderr,
              "convert_fence_handle: expected ENOTSUP off Windows, got %d\n",
              rc);
      failures++;
    }
#endif
    ef_tensor_free(fsrc);
    ef_tensor_free(fdst);
  }

#ifdef _WIN32
  // The cross-process route, run inside one process: a texture minted through
  // libedgefirst-image, serialized by libedgefirst-tensor into a
  // self-describing blob, and imported back as an independent tensor naming
  // the same texture and the same completion. That is the other half of the
  // cross-package story the capsule descriptor covers -- a consumer that
  // cannot share an address space still gets the texture.
  //
  // The out-of-band handle table is empty here: a D3D11 texture is named by
  // the NT handle values carried inside the blob, not by descriptors passed
  // beside it, so `NULL, 0` is the whole of it and `ef_tensor_export` must
  // accept that rather than report ENOSPC for a table with nothing to write.
  ef_tensor *bsrc = ef_image_processor_create_image(p, 64, 48, "rgba8",
                                                    /* U8 */ 0,
                                                    /* mem: auto */ UINT32_MAX,
                                                    /* ReadWrite */ 3);
  ef_tensor *bdst = ef_image_processor_create_image(p, 64, 48, "rgb8",
                                                    /* U8 */ 0,
                                                    /* mem: auto */ UINT32_MAX,
                                                    /* ReadWrite */ 3);
  if (bsrc == NULL || bdst == NULL) {
    fprintf(stderr, "export/import: create_image returned NULL\n");
    failures++;
    // Whichever one was allocated still has to be released; the shared
    // cleanup at the end of the else arm is not reached from here.
    ef_tensor_free(bsrc);
    ef_tensor_free(bdst);
  } else {
    // One byte value everywhere, padding included, so the destination's row 0
    // is comparable without knowing either row pitch.
    ef_tensor_view sv;
    if (ef_tensor_map(bsrc, EF_CPU_ACCESS_WRITE, &sv) != 0) {
      fprintf(stderr, "export/import: mapping the source failed\n");
      failures++;
    } else {
      memset(sv.ptr, 0x2A, sv.len);
      ef_tensor_unmap(bsrc);
    }

    // convert_fence_handle rather than convert: the row read below is a CPU
    // read of GPU-written bytes, and this event is the only waitable object
    // the C surface hands back for it -- the same wait the fenced section
    // above makes. The `dfence` from ef_tensor_gpu_completion is a *fence*
    // NT handle, not a synchronization object: WaitForSingleObject on it
    // returns WAIT_OBJECT_0 at once (measured) without ever seeing the
    // value, so waiting on it would order nothing while looking like it did.
    //
    // This whole block is already inside the file's `#ifdef _WIN32`, so the
    // off-Windows arm of the plain convert belongs to the fenced section
    // above, not here.
    void *cevent = NULL;
    if (ef_image_processor_convert_fence_handle(p, bsrc, bdst, 0, 0, NULL,
                                                &cevent) != 0) {
      fprintf(stderr, "export/import: convert failed\n");
      failures++;
    } else if (cevent == NULL) {
      fprintf(stderr, "export/import: the convert handed back no event\n");
      failures++;
    } else {
      if (WaitForSingleObject(cevent, 5000) != WAIT_OBJECT_0) {
        fprintf(stderr, "export/import: the convert never completed\n");
        failures++;
      }
      CloseHandle(cevent);
    }

    // The value the convert recorded, which the imported tensor must report
    // too -- a texture without its completion is a texture a consumer cannot
    // safely read. The handle is a fence, for a GPU consumer to open and wait
    // on `dvalue` with; it is closed here rather than waited on.
    void *dfence = NULL;
    uint64_t dvalue = 0;
    if (ef_tensor_gpu_completion(bdst, &dfence, &dvalue) != 0) {
      fprintf(stderr, "export/import: gpu_completion on the destination failed\n");
      failures++;
    } else if (dvalue == 0) {
      fprintf(stderr, "export/import: the convert recorded no completion\n");
      failures++;
    }
    if (dfence != NULL) {
      CloseHandle(dfence);
    }

    // Row 0 of the destination, copied out rather than compared under two
    // simultaneous maps: 64 rgb8 pixels are 192 bytes, inside any pitch this
    // texture can have.
    uint8_t row[192];
    uintptr_t dst_len = 0;
    int have_row = 0;
    ef_tensor_view dv;
    if (ef_tensor_map(bdst, EF_CPU_ACCESS_READ, &dv) != 0) {
      fprintf(stderr, "export/import: mapping the destination failed\n");
      failures++;
    } else {
      if (dv.len < sizeof row) {
        fprintf(stderr, "export/import: destination map is %zu bytes, too short\n",
                dv.len);
        failures++;
      } else {
        memcpy(row, dv.ptr, sizeof row);
        dst_len = dv.len;
        have_row = 1;
      }
      ef_tensor_unmap(bdst);
    }
    if (have_row) {
      uintptr_t i;
      int blank = 1;
      for (i = 0; i < sizeof row; i++) {
        if (row[i] != 0) {
          blank = 0;
          break;
        }
      }
      if (blank) {
        fprintf(stderr, "export/import: the convert left the destination blank\n");
        failures++;
      }
    }

    uint8_t blob[4096];
    uintptr_t blob_len = 0;
    uintptr_t fds_len = 0;
    int rc = ef_tensor_export(bdst, blob, sizeof blob, &blob_len, NULL, 0,
                              &fds_len);
    if (rc != 0) {
      fprintf(stderr, "export: expected 0, got %d (blob needs %zu bytes)\n", rc,
              blob_len);
      failures++;
    } else if (fds_len != 0) {
      fprintf(stderr, "export: a Windows blob carries no fds, got %zu\n",
              fds_len);
      failures++;
    } else {
      ef_tensor *back = ef_tensor_import(blob, blob_len, NULL, 0);
      if (back == NULL) {
        fprintf(stderr, "import: a blob this process just exported must import\n");
        failures++;
      } else {
        if (ef_tensor_storage_kind(back) != EF_STORAGE_KIND_DMA_BUF) {
          fprintf(stderr, "import: expected storage kind %u, got %u\n",
                  (unsigned)EF_STORAGE_KIND_DMA_BUF,
                  ef_tensor_storage_kind(back));
          failures++;
        }
        ef_tensor_view iv;
        if (ef_tensor_map(back, EF_CPU_ACCESS_READ, &iv) != 0) {
          fprintf(stderr, "import: mapping the imported tensor failed\n");
          failures++;
        } else {
          if (have_row && iv.len != dst_len) {
            fprintf(stderr, "import: map is %zu bytes, exported tensor's is %zu\n",
                    iv.len, dst_len);
            failures++;
          } else if (have_row && memcmp(iv.ptr, row, sizeof row) != 0) {
            fprintf(stderr, "import: the same texture must read the same bytes\n");
            failures++;
          }
          ef_tensor_unmap(back);
        }
        /* The import waits on the exporter's fence value and then signals the
           importing copy's own fence behind that wait, so what it reports is a
           point on the timeline its own gpu_completion fence names -- newer
           than the exported value, never equal to it. Asserting equality would
           pin a value meaningful only on the exporter's timeline, which across
           processes is a fence the consumer never reaches. */
        void *ifence = NULL;
        uint64_t ivalue = 0;
        if (ef_tensor_gpu_completion(back, &ifence, &ivalue) != 0) {
          fprintf(stderr, "import: gpu_completion on the imported tensor failed\n");
          failures++;
        } else if (ivalue <= dvalue) {
          fprintf(stderr, "import: completion value %llu is not past the exported %llu\n",
                  (unsigned long long)ivalue, (unsigned long long)dvalue);
          failures++;
        }
        if (ifence != NULL) {
          CloseHandle(ifence);
        }
        ef_tensor_free(back);
      }
    }
    ef_tensor_free(bsrc);
    ef_tensor_free(bdst);
  }
#endif

  ef_image_processor_free(p);
  return failures == 0 ? 0 : 1;
}
