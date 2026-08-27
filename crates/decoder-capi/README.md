<!--
SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
SPDX-License-Identifier: Apache-2.0
-->

# edgefirst-decoder (C API)

Turns raw model output tensors into detections and segmentation masks. Ships as
`libedgefirst_decoder` with `edgefirst/decoder.h`.

```sh
cc my_app.c $(pkg-config --cflags --libs edgefirst-decoder) -o my_app
```

`edgefirst-decoder` requires `edgefirst-tensor`: it consumes output tensors, so
`decoder.h` includes `edgefirst/tensor.h`. It does not require `edgefirst-image`
or `edgefirst-codec` — decoding model output has nothing to do with how the input
frame was produced.

## Usage

```c
#include <edgefirst/decoder.h>

ef_decoder_params *p = ef_decoder_params_new();
ef_decoder_params_set_config_file(p, "model.yaml");
ef_decoder_params_set_score_threshold(p, 0.25f);

ef_decoder *d = ef_decoder_new(p);
ef_decoder_params_free(p);

const ef_tensor *outputs[] = { out0, out1 };
struct ef_detect_box_list *boxes = NULL;
ef_segmentation_list *masks = NULL;

if (ef_decoder_decode(d, outputs, 2, &boxes, &masks) != 0) {
    fprintf(stderr, "decode failed: %s\n", ef_tensor_last_error_message());
    return 1;
}

for (uintptr_t i = 0; i < ef_detect_box_list_len(boxes); i++) {
    ef_detect_box b = ef_detect_box_list_get(boxes, i);
    /* ... */
}

ef_detect_box_list_free(boxes);
ef_segmentation_list_free(masks);
ef_decoder_free(d);
```

## Exactly one configuration source

JSON, YAML or a file — supplying none, or more than one, is an error rather than
a precedence rule nobody remembers. Two sources that disagree have no defined
resolution, so the API refuses instead of picking.

## Detections cross as plain values

`ef_detect_box` is a plain `#[repr(C)]` value declared exactly once, in the
dependency-free `edgefirst-decoder-abi` crate, and emitted into `decoder.h`.
`ef_detect_box_list_data` borrows the list as a C array, which is what lets
`libedgefirst_tracker` consume detections **without linking this library** — only
implementations force a link, plain values are shared by declaration.

## Tiled inference

`ef_tiled_frame_accumulator` merges detections across tiles of one frame.

`push_tile` is **idempotent per tile index**: a retried tile, an out-of-range
one, or a placement from a different plan is ignored and its detections dropped,
so out-of-order *and* at-least-once delivery converge on the same frame.

The default merge metric is Intersection-over-Smaller, not IoU. An object split
across a tile seam has *low* IoU with its own fragment, so an IoU default would
keep both halves as separate detections.

`finalize` is destructive and returns `NULL` on a second call, rather than an
empty list that would be indistinguishable from a frame which genuinely found
nothing.
