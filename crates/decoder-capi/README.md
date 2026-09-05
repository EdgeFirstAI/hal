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

The default merge **mode** is keep-best: the highest-scoring box of each matched
group is kept and the rest are dropped. The enclosing-union merge measured about
0.05 AP50 worse on every frame of the Ocean Cleanup ADIS 4K validation
(TOP2-836), so it is opt-in: fill an `ef_merge_config` with
`ef_merge_config_default`, set `mode = 1`, and pass it to
`ef_tiled_frame_accumulator_new` or `ef_merge_tiled_detections`.

`mode` was added to `ef_merge_config` in the 4-byte tail pad it already had, so
the struct is still 32 bytes and no other field moved. `ef_decoder_abi_version`
is `2`: the layout did not change, but the default merge did, so the same call
with the same struct returns different box geometry than version 1 -- gate on
the probe rather than on a link succeeding.

**This is a minor-version ABI break.** `ef_merge_config` shipped without `mode`
in 0.29.x, and a caller built against that header never initialised the tail
pad this release now reads. Rebuild against the 0.30 header; do not mix a 0.29
consumer with a 0.30 `libedgefirst_decoder`.

`finalize` is destructive and returns `NULL` on a second call, rather than an
empty list that would be indistinguishable from a frame which genuinely found
nothing.

## Schema inference for Ultralytics exports

A vanilla Ultralytics YOLOv8/11/26 export carries no `edgefirst.json`, but its
own metadata and tensor shapes are enough to derive one. Accumulate what your
inference runtime reports into `ef_infer_signals`, and
`ef_infer_ultralytics_schema` gives back a schema ready for
`ef_decoder_params_set_config_json`.

```c
#include <edgefirst/decoder.h>
#include <stdint.h>
#include <stdio.h>

ef_infer_signals *s = ef_infer_signals_new(0); /* 0 onnx, 1 tflite */

const uintptr_t in_shape[4] = { 1, 3, 640, 640 };
ef_infer_signals_add_input(s, "images", in_shape, 4, EF_INFER_DTYPE_FLOAT32);

const uintptr_t out_shape[3] = { 1, 6, 8400 }; /* 4 box + 2 classes */
ef_infer_signals_add_output(s, "output0", out_shape, 3, EF_INFER_DTYPE_FLOAT32,
                            NULL, NULL, 0); /* quant_len 0 = unquantized */

/* Verbatim from the model: ONNX metadata_props, or TFLite metadata.json. */
ef_infer_signals_add_metadata(s, "names", "{0: 'person', 1: 'bicycle'}");
ef_infer_signals_add_metadata(s, "task", "detect");
ef_infer_signals_add_metadata(s, "end2end", "False");

char *err = NULL;
ef_inferred_schema *inferred = ef_infer_ultralytics_schema(s, &err);
ef_infer_signals_free(s);
if (!inferred) {
    fprintf(stderr, "inference failed: %s\n", err ? err : "(no message)");
    ef_decoder_string_free(err);
    return 1;
}

char *schema_json = ef_inferred_schema_json(inferred);
char *labels_json = ef_inferred_schema_labels_json(inferred);
char *description = ef_inferred_schema_description(inferred);
printf("%s\n", description); /* "Ultralytics YOLOv8/11 detect, 2 classes" */

ef_decoder_params *p = ef_decoder_params_new();
ef_decoder_params_set_config_json(p, schema_json, 0); /* len 0 = NUL-terminated */
ef_decoder_params_set_score_threshold(p, 0.25f);
ef_decoder *d = ef_decoder_new(p);
ef_decoder_params_free(p);

ef_decoder_string_free(schema_json);
ef_decoder_string_free(labels_json);
ef_decoder_string_free(description);
ef_inferred_schema_free(inferred);
```

`labels_json` is a JSON array of class names in index order — the decoder
reports class *indices*, and this is what maps them back to names.

Every string the API returns is an independent allocation freed with
`ef_decoder_string_free`, including `err`. Initialise your `char *` to `NULL`
and detect failure from the returned handle, not from `err`: success leaves
`*err_out` untouched. Every failure path writes a message — a panic caught
at the boundary included — but the write itself can fail if the message
cannot be allocated or holds an embedded NUL, so test `err` before printing
it. Freeing `NULL` is a no-op, so the error path above is correct in every
case.

The dtype codes are `EF_INFER_DTYPE_*`, mirroring `schema::DType`. They are a
separate vocabulary from `edgefirst/tensor.h`'s `EF_DTYPE_*` — schema dtype is
the narrower quantized/float set a model's *logical* I/O carries — and they
deliberately start at `0x100` so the two ranges are disjoint. Both cross as
bare `uint32_t`, so overlapping ranges would have made every `EF_DTYPE_*`
value a valid code here meaning something else; disjoint ranges turn passing
the wrong one into `EINVAL`. `source` is a plain integer with no macros:
`0` onnx, `1` tflite, `2` other. `2` is accepted by `ef_infer_signals_new`
but refused by inference — whether boxes are pixel-space or `[0, 1]` follows
the exporter, is not derivable from shapes, and guessing scales every box by
the input size.

An inferred schema pins the NMS *mode* and leaves the *thresholds* to you.
Ultralytics runs NMS class-aware (`agnostic=False`), so a pre-NMS YOLOv8/11
schema says so — leaving it unset is not neutral, because
`ef_decoder_params_new` defaults to mode `1` (automatic), which resolves an
unset config to class-agnostic and would suppress a box against an
overlapping box of a *different* class. `ef_decoder_params_set_nms` still
overrides. Thresholds are not inferable from shapes, and the library's
defaults (`0.5`/`0.5`) are not Ultralytics' (`0.25`/`0.45`), so set them
explicitly as above. YOLO26 end-to-end exports apply their own NMS in-graph
and carry no mode at all.

Inference never guesses: metadata and shapes are cross-checked, and a
disagreement — a class count that does not fit the output width, a `segment`
task with no prototype tensor, an unsupported task such as pose or OBB — is
reported through `err_out` rather than resolved by preference.
