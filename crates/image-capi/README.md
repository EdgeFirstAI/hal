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

ef_image_processor_convert(p, src, dst, NULL); /* NULL: default options */

ef_tensor_free(src);
ef_tensor_free(dst);
ef_image_processor_free(p);
```

Tensors from here are ordinary tensors — read them with `ef_tensor_shape`,
release them with `ef_tensor_free`. There is no separate image-tensor type, and
one from `ef_tensor_new()` works here just as well.

## Convert options

`NULL` options convert the whole source, stretched, with float outputs mapped as `x / 255`. An `ef_convert_options` handle adds rotation, flip, a source rectangle, a letterbox with its placement and pad, and normalization:

```c
ef_convert_options *o = ef_convert_options_new();
const uint8_t pad[4] = {0, 0, 0, 255};
const float pad_out[4] = {0.0f, 0.0f, 0.0f, 0.0f};
ef_convert_options_set_letterbox(o, pad);
ef_convert_options_set_letterbox_placement(o, EF_LETTERBOX_PLACEMENT_TOP_LEFT);
ef_convert_options_set_pad_output(o, pad_out);
ef_convert_options_set_normalization_preset(o, EF_NORMALIZATION_IMAGENET);
ef_image_processor_convert(p, src, dst, o);
ef_convert_options_free(o);
```

Normalization is `(x - mean) / std` on 0-255 values (`ef_convert_options_set_normalization` takes explicit coefficients). A u8 or i8 destination with a non-default normalization must carry quantization (`ef_tensor_quantization_set`); the result is quantized with it, and the convert returns `EINVAL` without it. The setters return `EINVAL` for an unknown code and leave the options unchanged.

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

<img referrerpolicy="no-referrer" src="https://px.edgefirst.ai/a.png?x-pxid=37aa7de2-6935-484e-b31d-9fd9a57fcd63" alt="" width="1" height="1" style="position:absolute; width:1px; height:1px; opacity:0; pointer-events:none;" />
