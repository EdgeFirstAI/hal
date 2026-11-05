<!--
SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
SPDX-License-Identifier: Apache-2.0
-->

# edgefirst-kernels (C API)

Quantized compute kernels for edge inference: int8/int16 softmax, sigmoid, 8-bit table lookup, DFL and dist2bbox box decoding, TFLite-exact int8 `ADD`/`SUB`/`MUL` and matrix multiplication (`BATCH_MATMUL`, 1×1 `CONV_2D`), a fused DFL softmax-and-projection head, and fused int8 attention.
Ships as `libedgefirst_kernels` with `edgefirst/kernels.h`.

```sh
cc my_app.c $(pkg-config --cflags --libs edgefirst-kernels) -o my_app
```

The library has no `DT_NEEDED` entry for any sibling EdgeFirst library: tensors are caller-owned `ef_kernel_view`s (pointer, dtype, shape, optional element strides).

## Usage

```c
#include <edgefirst/kernels.h>

ef_softmax_i8_params p = {{0.1f, 0}, {1.0f / 256.0f, -128}, 1.0f};
ef_softmax_i8_plan *plan = NULL;
if (ef_softmax_i8_plan_new(&p, &plan) != 0) {
    /* errno == ENOTSUP: valid but not handled; EINVAL: bad arguments */
    fprintf(stderr, "%s\n", ef_kernels_last_error());
    return -1;
}

size_t shape[2] = {1, 4};
ef_kernel_view in = {in_data, EF_KERNEL_DTYPE_I8, 2, shape, NULL};
ef_kernel_view out = {out_data, EF_KERNEL_DTYPE_I8, 2, shape, NULL};
ef_softmax_i8_plan_run(plan, &in, &out, NULL); /* NULL pool: calling thread */
ef_softmax_i8_plan_free(plan);
```

Plans are immutable once built; `ef_<op>_plan_run` may be called on one plan from several threads at once.
A plan builder returning `ENOTSUP` is how a caller asks whether a node can run here.
Pass an `ef_kernel_pool` to split a run across the caller's own worker threads; give them about 64 KiB of stack headroom for the kernels' staging buffers.
A view's `data` must be aligned to its element type, and the output must not overlap any input in memory (interleaved views of one buffer included); `plan_run` rejects either with `EINVAL`. Inputs may share memory, so attention's Q, K and V can be head slices of one fused buffer.
`ef_attention_i8_plan_run` takes a caller-owned `ef_attention_scratch` from `ef_attention_scratch_new` for its call shape; reuse it across calls, but give each concurrent call its own.
