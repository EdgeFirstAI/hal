<!--
SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
SPDX-License-Identifier: Apache-2.0
-->

# edgefirst-image (C API)

Hardware-accelerated image conversion, scaling and colour handling across GL,
G2D and CPU backends. Ships as `libedgefirst_image` with `edgefirst/image.h`.

```sh
cc my_app.c $(pkg-config --cflags --libs edgefirst-image) -o my_app
```

`edgefirst-image` requires `edgefirst-tensor`: using this library means handling
tensors, so `image.h` includes `edgefirst/tensor.h` and the pkg-config pulls the
tensor library in.

## Usage

```c
ef_image_processor *p = ef_image_processor_new();

ef_tensor *src = ef_image_processor_create_image(p, 1920, 1080, "NV12",
                                                 EF_DTYPE_U8,
                                                 EF_STORAGE_KIND_DMA_BUF,
                                                 /* CpuAccess::None */ 0);
ef_tensor *dst = ef_image_processor_create_image(p, 640, 480, "rgb8",
                                                 EF_DTYPE_U8,
                                                 EF_STORAGE_KIND_DMA_BUF, 0);

ef_image_processor_convert(p, src, dst);

ef_tensor_free(src);
ef_tensor_free(dst);
ef_image_processor_free(p);
```

Tensors from here are ordinary tensors — read them with `ef_tensor_shape`,
release them with `ef_tensor_free`. There is no separate image-tensor type, and
one from `ef_tensor_new()` works here just as well.

## What actually needs a processor

Only PBO allocation, because a PBO is a GL object and only the context owner can
create one. `mem`, `shm` and `dmabuf` are all available from
`libedgefirst-tensor` alone, and a V4L2 or libcamera buffer is wrapped there
too. Once a tensor exists its origin stops mattering.

## Converting a tensor from somewhere else

`convert` accepts a source or destination minted by any EdgeFirst library, backed
any way it likes — host memory, dma-buf or PBO. There is no re-import step and no
case this library has to refuse, because every EdgeFirst library reaches the same
tensor implementation in `libedgefirst_tensor.so`: a handle minted by one is the
same object to all of them, not a compatible copy that has to be translated.

Earlier versions of this library held their own private copy of the tensor code,
which made a tensor from elsewhere genuinely foreign — it had to be re-imported
from its planes, a host-memory source had no shareable handle to import from, and
a foreign destination would have been written through the re-import rather than
into the caller's buffer. Both of those returned `EOPNOTSUPP`. Neither case
exists any more; there is nothing left to special-case.

## Asynchronous convert

`ef_image_processor_convert_fence` returns a sync-fence fd instead of blocking,
for the GL to NPU handoff. The caller owns the descriptor and must close it.
`-1` means the platform has no native fence and the convert already completed —
not an error.
