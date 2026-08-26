// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

#include <edgefirst/tensor.h>
#include <assert.h>
#include <errno.h>

int main(void) {
    uint64_t dims[2] = {2, 3};
    ef_tensor *t = ef_tensor_new(0, dims, 2);
    assert(t != NULL);

    ef_tensor_view v = {0};
    assert(ef_tensor_map(t, EF_CPU_ACCESS_READ_WRITE, &v) == 0);
    assert(v.len == 6);
    for (size_t i = 0; i < v.len; i++) v.ptr[i] = (uint8_t)(10 + i);
    assert(ef_tensor_unmap(t) == 0);

    uint8_t out[6] = {0};
    assert(ef_tensor_copy_to(t, out, sizeof out) == 6);
    for (size_t i = 0; i < 6; i++) assert(out[i] == 10 + i);

    assert(ef_tensor_map(t, EF_CPU_ACCESS_NONE, &v) == EINVAL);
    assert(ef_tensor_retain(t) == 0);
    ef_tensor_free(t);
    assert(ef_tensor_ndim(t) == 2); /* retained: still alive */
    ef_tensor_free(t);
    return 0;
}
