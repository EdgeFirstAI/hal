<!--
SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
SPDX-License-Identifier: Apache-2.0
-->

# edgefirst-codec (C API)

Hardware-accelerated image decoding — JPEG via V4L2 or nvJPEG, plus PNG. Ships as
`libedgefirst_codec` with `edgefirst/codec.h`.

```sh
cc my_app.c $(pkg-config --cflags --libs edgefirst-codec) -o my_app
```

`edgefirst-codec` requires `edgefirst-tensor`: decoding means producing tensors,
so `codec.h` includes `edgefirst/tensor.h` and the pkg-config pulls the tensor
library in. It does **not** require `edgefirst-image` or `edgefirst-decoder` — an
application that only decodes JPEGs links two libraries and no more. That is the
whole reason these ship separately.

## Usage

```c
#include <edgefirst/codec.h>

ef_image_decoder *d = ef_image_decoder_new();

ef_tensor *dst = ef_tensor_image_alloc(1920, 1080, "NV12", EF_DTYPE_U8,
                                       /* has_memory */ 1,
                                       EF_TENSOR_MEMORY_DMA_BUF,
                                       /* access */ 0);

if (ef_image_decoder_decode_into(d, jpeg_bytes, jpeg_len, dst) != 0) {
    fprintf(stderr, "decode failed: %s\n", ef_tensor_last_error_message());
    return 1;
}

ef_tensor_free(dst);
ef_image_decoder_free(d);
```

`ef_image_decoder_decode_file_into` takes a path instead of a buffer.

## Decoding writes into a tensor you already have

`decode_into` does not allocate. The caller supplies the destination, which is
what lets the decode target be a dma-buf the GPU reads next with no copy in
between — the most important property for a camera-to-inference pipeline.

The destination may be minted by any EdgeFirst library and backed any way it
likes: host memory, dma-buf or PBO. Every library reaches the same tensor
implementation in `libedgefirst_tensor.so`, so there is no "foreign" destination
to re-import and none this library has to refuse.

## Sizing the destination

`InsufficientCapacity` maps to `ENOSPC` specifically rather than to a generic
failure, so a caller who sized the destination from a JPEG header can tell "your
buffer is too small" apart from "this file is not decodable", and act on it.

## Backend availability

`ef_codec_v4l2_available` and `ef_codec_nvjpeg_available` report at runtime
whether the accelerated paths are present on this machine, so an application can
choose rather than discover it from a decode failure.
