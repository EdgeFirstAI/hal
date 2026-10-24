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

## Example: a display pipeline

[`examples/display_pipeline.c`](examples/display_pipeline.c) renders a camera frame and detection boxes into a hardware-only (`EF_CPU_ACCESS_NONE`) DMA-BUF canvas, ready to hand to a Wayland compositor by fd, keeps a CPU-written overlay in an `EF_CPU_ACCESS_WRITE` image, decodes captured model outputs with the decoder library, and checks that the library refuses CPU reads of both images. It prints each check and the decode and draw timings. Build it against the tensor, image and decoder libraries, for example cross-compiled for an aarch64 board with the libraries from `cargo-zigbuild build --release --target aarch64-unknown-linux-gnu` (see `scripts/on-target-bundle.sh`):

```sh
zig cc -std=c11 -O2 -target aarch64-linux-gnu.2.35 -o display_pipeline \
    crates/image-capi/examples/display_pipeline.c \
    -Icrates/image-capi/include -Icrates/tensor-capi/include \
    -Icrates/decoder-capi/include -Icrates/decoder-abi/include \
    -L$LIBDIR -ledgefirst_image -ledgefirst_decoder -ledgefirst_tensor \
    -Wl,-rpath,'$ORIGIN'
```

`crates/python-image/examples/display_pipeline.py` is the same walk-through in Python.

## Convert options

`NULL` options convert the whole source, stretched, with float outputs mapped as `x / 255`. An `ef_convert_options` handle adds rotation, flip, a source rectangle, a letterbox with its placement and pad, normalization, and interpolation:

```c
ef_convert_options *o = ef_convert_options_new();
const uint8_t pad[4] = {0, 0, 0, 255};
const float pad_out[4] = {0.0f, 0.0f, 0.0f, 0.0f};
ef_convert_options_set_letterbox(o, pad);
ef_convert_options_set_letterbox_placement(o, EF_LETTERBOX_PLACEMENT_TOP_LEFT);
ef_convert_options_set_pad_output(o, pad_out);
ef_convert_options_set_normalization_preset(o, EF_NORMALIZATION_IMAGENET);
ef_convert_options_set_interpolation(o, EF_INTERPOLATION_NEAREST);
ef_image_processor_convert(p, src, dst, o);
ef_convert_options_free(o);
```

`ef_convert_options_content_region(o, src_w, src_h, dst_w, dst_h, pixels, normalized)` reports where a convert with `o` puts the image: `[x, y, width, height]` in destination pixels and `[xmin, ymin, xmax, ymax]` normalized, the exact rectangle the convert paints. The normalized form is what the draw functions' `content_region` argument and the decoder's `ef_semantic_options_set_content_region` take.

Options without an interpolation resample with the processor's (`ef_image_processor_set_interpolation`); `EF_INTERPOLATION_BICUBIC` and `EF_INTERPOLATION_AREA` run as bilinear, logged once per mode, and `EDGEFIRST_INTERPOLATION` overrides both. Because the filter travels with the options, one processor can serve several models without reconfiguring it between them.

Normalization is `(x - mean) / std` on 0-255 values (`ef_convert_options_set_normalization` takes explicit coefficients). A u8 or i8 destination with a non-default normalization must carry quantization (`ef_tensor_quantization_set`); the result is quantized with it, and the convert returns `EINVAL` without it. The setters return `EINVAL` for an unknown code and leave the options unchanged.

## Model preprocessing

A model's preprocessing arrives as an `ef_preprocess_spec` — fit, letterbox placement and pad, normalization mean/std on 0-255 values, interpolation, input size — declared in header-only `edgefirst/detect.h`. `libedgefirst_decoder`'s `ef_model_config_preprocessing` fills one from the model's own configuration (see that library's README); a caller may also start from `ef_preprocess_spec_default()` (unsigned normalization, centred letterbox padded with pixel 114, bilinear) and set fields by hand.

```c
ef_preprocess_spec spec;
ef_model_config_preprocessing(cfg, &spec);        /* libedgefirst_decoder */

ef_convert_options *o = ef_convert_options_from_spec(&spec);
if (o) {
    ef_image_processor_convert(p, frame, input, o);
    ef_convert_options_free(o);
}
```

`ef_convert_options_from_spec` builds the options — normalization, fit, placement, pad and interpolation. It returns `NULL` with `errno` `EINVAL` for an unknown code, a zero `std` or a pixel pad outside 0-255.

## Tiling

`ef_tiling_config` holds the tile geometry. How each tile is converted is the `const ef_convert_options *` that `ef_image_processor_plan_tiles`, `_tile_into` and `_tile_one` take beside it (`NULL` stretches each tile); pass the same options to the plan and the render. With a letterbox, every `ef_tile_placement` carries its `content_region`. Options that rotate or flip are `EINVAL`.

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
