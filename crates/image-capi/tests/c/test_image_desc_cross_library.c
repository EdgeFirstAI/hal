// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0
//
// `ef_tensor_image_desc` is minted and read entirely by libedgefirst-tensor;
// `ef_image_processor_create_image_desc` (libedgefirst-image) only ever sees
// it through `ef_tensor_image_desc_get`'s scalar view -- never a dereference
// of tensor-capi's private layout. A single library cannot demonstrate that
// the handle stays opaque across the boundary; this links both.

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

  // Minted entirely by libedgefirst-tensor.
  ef_tensor_image_desc *d =
      ef_tensor_image_desc_new(64, 48, "NV12", /* U8 */ 0);
  if (d == NULL) {
    fprintf(stderr, "ef_tensor_image_desc_new returned NULL\n");
    ef_image_processor_free(p);
    return 1;
  }
  if (ef_tensor_image_desc_set_memory(d, /* Mem */ 0) != 0) {
    fprintf(stderr, "set_memory(Mem) unexpectedly failed\n");
    failures++;
  }
  // Access stays at its default, `None`: `ImageDesc` itself makes a
  // compression request invalid once CPU access is declared, so ReadWrite +
  // Any is a genuinely invalid combination, not something this crossing
  // should paper over. `access` is exercised on its own in
  // `test_cross_library.c`'s `create_image` call.
  if (ef_tensor_image_desc_set_compression(d, /* Any */ 1) != 0) {
    fprintf(stderr, "set_compression(Any) unexpectedly failed\n");
    failures++;
  }

  // The load-bearing crossing: image reads the request through
  // `ef_tensor_image_desc_get`'s view, never through the pointer itself.
  ef_tensor *t = ef_image_processor_create_image_desc(p, d);
  if (t == NULL) {
    fprintf(stderr, "create_image_desc returned NULL\n");
    failures++;
  } else {
    if (strcmp(ef_tensor_format(t), "NV12") != 0) {
      fprintf(stderr, "format: expected NV12, got %s\n", ef_tensor_format(t));
      failures++;
    }
    if (ef_tensor_plane_count(t) != 2) {
      fprintf(stderr, "NV12 must report 2 planes, got %u\n",
              ef_tensor_plane_count(t));
      failures++;
    }
    // There is one real implementation (`libedgefirst_tensor.so`); freeing
    // this tensor just needs to link and run, exactly as
    // `test_cross_library.c`'s header comment explains for `create_image`.
    ef_tensor_free(t);
  }

  // Not consumed by allocation: the same request fills a pool.
  for (int i = 0; i < 3; i++) {
    ef_tensor *pooled = ef_image_processor_create_image_desc(p, d);
    if (pooled == NULL) {
      fprintf(stderr, "create_image_desc[%d] (pool) returned NULL\n", i);
      failures++;
      break;
    }
    ef_tensor_free(pooled);
  }

  // NULL handling on both sides of the boundary.
  if (ef_image_processor_create_image_desc(NULL, d) != NULL) {
    fprintf(stderr, "a NULL processor must be refused\n");
    failures++;
  }
  if (ef_image_processor_create_image_desc(p, NULL) != NULL) {
    fprintf(stderr, "a NULL request must be refused\n");
    failures++;
  }

  // A rejected setter leaves the request at tensor-capi's 1x1 grey
  // placeholder (see `desc::tests::a_rejected_setter_leaves_the_request_usable`
  // there) -- not the caller's original fields, but still a request image can
  // mint from, since "usable" is the property this library actually depends
  // on.
  if (ef_tensor_image_desc_set_memory(d, 9999) == 0) {
    fprintf(stderr, "an unknown memory code must be refused\n");
    failures++;
  }
  ef_tensor *placeholder = ef_image_processor_create_image_desc(p, d);
  if (placeholder == NULL) {
    fprintf(stderr,
            "the request must survive a rejected setter (placeholder)\n");
    failures++;
  } else {
    ef_tensor_free(placeholder);
  }

  ef_tensor_image_desc_free(d);
  ef_image_processor_free(p);
  return failures == 0 ? 0 : 1;
}
