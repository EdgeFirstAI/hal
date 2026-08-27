/* SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
 * SPDX-License-Identifier: Apache-2.0
 *
 * Links and RUNS against libedgefirst_tensor.
 * A header that compiles is not the same as a library that loads. */
#include <edgefirst/tensor.h>
#include <stdint.h>
#include <stdio.h>

int main(void) {
    uint64_t dims[1] = {4};
    ef_tensor *t = ef_tensor_new(EF_DTYPE_U8, dims, 1);
    if (!t) { fprintf(stderr, "FAIL: ef_tensor_new returned NULL\n"); return 1; }
    const uint64_t *shape = ef_tensor_shape(t);
    if (!shape) { fprintf(stderr, "FAIL: ef_tensor_shape returned NULL\n"); return 2; }
    printf("PASS: tensor links and runs (shape[0]=%llu)\n",
           (unsigned long long)shape[0]);
    ef_tensor_free(t);
    return 0;
}
