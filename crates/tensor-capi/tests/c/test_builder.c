// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0
//
// The builder as a C caller actually uses it: chained, with a single check at
// the end. That pattern only works because errors are sticky — this file is
// the executable form of that claim.

#include <stdio.h>

#include "edgefirst/tensor.h"

int main(void) {
  int failures = 0;

  // The happy path, written the way a C++ wrapper would chain it: no check
  // per line, one check at the end.
  {
    ef_tensor_builder *b = ef_tensor_builder_new();
    uint64_t dims[2] = {4, 4};
    int64_t strides[2] = {4, 1};
    ef_tensor_builder_dtype(b, 0 /* U8 */);
    ef_tensor_builder_shape(b, dims, 2);
    ef_tensor_builder_strides(b, strides, 2);
    ef_tensor_builder_storage(b, 0 /* Mem */);
    ef_tensor_builder_format(b, "");
    if (ef_tensor_builder_error(b) != 0) {
      fprintf(stderr, "valid chain reported error %d\n", ef_tensor_builder_error(b));
      failures++;
    }
    ef_tensor_builder_free(b);
  }

  // The whole point: a fault in the middle of a chain is still reported at
  // the end, and is still the FIRST fault rather than the last call's status.
  {
    ef_tensor_builder *b = ef_tensor_builder_new();
    uint64_t dims[2] = {4, 4};
    ef_tensor_builder_dtype(b, 9999); // invalid
    ef_tensor_builder_shape(b, dims, 2);
    ef_tensor_builder_storage(b, 0);
    if (ef_tensor_builder_error(b) == 0) {
      fprintf(stderr, "a mid-chain fault was lost\n");
      failures++;
    }
    ef_tensor_builder_free(b);
  }

  ef_tensor_builder_free(NULL); // no-op, like free(3)

  return failures == 0 ? 0 : 1;
}
