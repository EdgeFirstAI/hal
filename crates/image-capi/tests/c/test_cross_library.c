// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0
//
// Two libraries in one process: a tensor created through
// `libedgefirst-image` is read and freed through the accessors
// `libedgefirst-tensor` exports.

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

  // Created by libedgefirst-image.
  ef_tensor *t = ef_image_processor_create_image(p, 64, 48, "NV12",
                                                 /* U8 */ 0, /* mem */ 0,
                                                 /* ReadWrite */ 3);
  if (t == NULL) {
    fprintf(stderr, "create_image returned NULL\n");
    ef_image_processor_free(p);
    return 1;
  }

  // Read through the accessors declared in tensor.h.
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

  // Freed through libedgefirst-tensor.
  ef_tensor_free(t);

  // A tensor from `ef_tensor_new`, freed the same way.
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

  // ef_image_processor_convert_fence_handle returns an owned event on Windows
  // and ENOTSUP elsewhere. UINT32_MAX is outside the storage vocabulary, so
  // create_image() lets the allocator choose.
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
    int rc = ef_image_processor_convert_fence_handle(p, fsrc, fdst, NULL,
                                                      &fence);
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
  // A texture created through libedgefirst-image is exported by
  // libedgefirst-tensor into a blob and imported back as an independent tensor
  // naming the same texture. The handle table is empty.
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
    ef_tensor_free(bsrc);
    ef_tensor_free(bdst);
  } else {
    // Fills the source, padding included, with one byte value.
    ef_tensor_view sv;
    if (ef_tensor_map(bsrc, EF_CPU_ACCESS_WRITE, &sv) != 0) {
      fprintf(stderr, "export/import: mapping the source failed\n");
      failures++;
    } else {
      memset(sv.ptr, 0x2A, sv.len);
      ef_tensor_unmap(bsrc);
    }

    // Waits on the convert's event before the CPU reads the destination.
    void *cevent = NULL;
    if (ef_image_processor_convert_fence_handle(p, bsrc, bdst, NULL,
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

    // The completion value the convert recorded on the destination.
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

    // Copies row 0 of the destination: 64 rgb8 pixels are 192 bytes.
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
        /* The imported tensor reports a completion value past the exported
           one. */
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
