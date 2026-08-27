<!--
SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
SPDX-License-Identifier: Apache-2.0
-->

# edgefirst-tensor (C API)

The C API for EdgeFirst's zero-copy tensor. Ships as `libedgefirst_tensor`
with a single header, `edgefirst/tensor.h`.

```sh
cc my_app.c $(pkg-config --cflags --libs edgefirst-tensor) -o my_app
```

This is the base library. It depends on nothing else in the HAL, so an
application doing tensor IPC or file serialization links this alone. Every other
EdgeFirst library builds on it: their headers include `edgefirst/tensor.h`, and
they depend on `edgefirst-tensor`.

## Three surfaces

A tensor's layout is defined in exactly one of them, which is what keeps the
other two free to change.

| Surface | Use it for | Why |
|---|---|---|
| Opaque handle | in-process access | accessors are functions `libedgefirst_tensor` exports; the handle itself declares no layout |
| Builder | construction | per-field errors, and it survives its terminal call |
| `(blob, handles)` | IPC and serialization | handles travel out of band; a raw fd is meaningless across a process |

### Tensors work the same wherever they came from

`ef_image_processor_create_image()` returns an `ef_tensor`. So does
`ef_tensor_new()`. Both are read, and freed, with the same functions:

```c
ef_tensor_shape(t);
ef_tensor_free(t);
```

You do not need to know which library minted a tensor, and there is no separate
"image tensor" type to convert.

<details>
<summary>How that works, if you are curious</summary>

Rust has no stable ABI, so a shared library normally cannot safely hand another
library a value built from its own copy of the Rust code. `libedgefirst_tensor`
sidesteps that by being the *only* library that exports `ef_tensor_shape` and
the rest of the accessors — there is nothing else to interpose, no dynamic-linker
ambiguity to resolve.

`libedgefirst-image` still mints its own tensors internally rather than linking
against this library for construction, so under the hood each exported accessor
dispatches through a small internal table the handle carries rather than
assuming its own layout — `free` goes through the same table, since the
allocation belongs to whichever library made it. That dispatch is a transition
detail, not API: it is never declared in the header, and it goes away once every
EdgeFirst library mints tensors through this one instead of its own copy.

This is written down because it explains why `ef_tensor_free` works uniformly on
a handle from either library, not because a caller has to think about it.
</details>

## Allocating

```c
uint64_t dims[2] = {480, 640};
ef_tensor *t = ef_tensor_new(EF_DTYPE_U8, dims, 2);
/* ... */
ef_tensor_free(t);
```

For anything beyond host memory, use the builder. Errors are **sticky**: after
one setter fails the rest no-op, and the terminal call reports the *first*
failure. Check once, at the end.

```c
ef_tensor_builder *b = ef_tensor_builder_new();
ef_tensor_builder_dtype(b, EF_DTYPE_U8);
ef_tensor_builder_shape(b, dims, 2);
ef_tensor_builder_storage(b, EF_STORAGE_KIND_DMA_BUF);
ef_tensor_builder_format(b, "NV12");
ef_tensor *t = ef_tensor_builder_alloc(b);
if (!t) fprintf(stderr, "build failed: %s\n", strerror(ef_tensor_builder_error(b)));
ef_tensor_builder_free(b);   /* the builder survives alloc; reuse it for a pool */
```

`alloc` derives storage itself and takes **no** planes. `wrap` adopts external
handles and requires **at least one**. Using the wrong one is an error, not a
convention — so a V4L2 or libcamera buffer is wrapped, never accidentally
reallocated.

## What needs `ImageProcessor`

Only PBO creation, because a PBO is a GL object and only the context owner can
make one.

| Operation | This library alone | Needs `ImageProcessor` |
|---|---|---|
| Allocate `mem` / `shm` / `dmabuf` | yes | — |
| Wrap external V4L2 / libcamera fds | yes | — |
| Wrap an IOSurface | yes | — |
| Allocate a **PBO** | no | **yes** |

Once a tensor exists, its origin stops mattering: a wrapped camera frame and an
`ImageProcessor`-allocated buffer are indistinguishable to every consumer,
because identity is derived from the buffer rather than from provenance.

## Sending a tensor to another process

Export is a pair. The blob is self-describing bytes; the handles travel out of
band (`SCM_RIGHTS`, `pidfd_getfd`), because the receiver is handed different
descriptor numbers than the sender used. Inside the blob, a plane's handle is an
**index** into that table.

```c
size_t blob_len = 0, fds_len = 0;
ef_tensor_export(t, NULL, 0, &blob_len, NULL, 0, &fds_len);   /* ask */
uint8_t *blob = malloc(blob_len);
int *fds = malloc(fds_len * sizeof(int));
ef_tensor_export(t, blob, blob_len, &blob_len, fds, fds_len, &fds_len);
```

Import **dups** every handle it keeps, so you may close your copies as soon as
it returns. There is no keepalive protocol.

A tensor with no shareable handle (`mem`, `pbo`) is exported **inline** — its
bytes travel in the blob — because there is nothing to refer to. Reference mode
round-trips to the same buffer; inline mode round-trips content and metadata
into a new allocation, so handle identity is not preserved and must not be
assumed.
