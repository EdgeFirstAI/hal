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
the struct is still 32 bytes and no other field moved. `ef_decoder_abi_version` moved to `2` for it: `sizeof` did not change, but a caller built against version 1 never writes `mode`, so gate on the probe rather than on a link succeeding.

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

ef_infer_signals *s = ef_infer_signals_new(0); /* 0 onnx, 1 tflite; see below */

const uintptr_t in_shape[4] = { 1, 3, 640, 640 };
ef_infer_signals_add_input(s, "images", in_shape, 4, EF_INFER_DTYPE_FLOAT32);

const uintptr_t out_shape[3] = { 1, 6, 8400 }; /* 4 box + 2 classes */
ef_infer_signals_add_output(s, "output0", out_shape, 3, EF_INFER_DTYPE_FLOAT32,
                            NULL, NULL, 0); /* quant_len 0 = unquantized */

/* Verbatim from the model: ONNX metadata_props, TFLite metadata.json,
 * or CoreML's own metadata alongside Ultralytics' props. */
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

The dtype codes are `edgefirst/tensor.h`'s `EF_DTYPE_*`: dtype has one numbering across the stack, so pass the code a tensor already reports. `EF_INFER_DTYPE_*` remain as aliases of them (`EF_INFER_DTYPE_FLOAT32` is `EF_DTYPE_F32`). Before `ef_decoder_abi_version` 3 these were `0x100`-based codes; a caller compiled against that header must be rebuilt. A model's logical I/O carries no `u64` or `f64`, so `EF_DTYPE_U64` and `EF_DTYPE_F64` are `EINVAL`. `source` is an `EF_MODEL_SOURCE_*` code. `EF_MODEL_SOURCE_OTHER` is accepted by `ef_infer_signals_new` but refused by inference — whether boxes are pixel-space or `[0, 1]` follows the exporter, is not derivable from shapes, and guessing scales every box by the input size.

An inferred schema pins the NMS *mode* and leaves the *thresholds* to you. Ultralytics runs NMS class-aware (`agnostic=False`), so a pre-NMS YOLOv8/11 schema says so explicitly. `ef_decoder_params_new` defaults to `EF_NMS_AUTO`, which takes the config's mode and otherwise resolves to class-aware; `ef_decoder_params_set_nms` still overrides. Thresholds are not inferable from shapes, and the library's defaults (`0.5`/`0.5`) are not Ultralytics' (`0.25`/`0.45`), so set them explicitly as above. YOLO26 end-to-end exports apply their own NMS in-graph and carry no mode at all.

For validation and mAP runs, `ef_decoder_params_set_multi_label(p, 1)` emits one box per class above the score threshold instead of one per anchor. It overrides the model config's `nms_multi_label`; tracked decode ignores it and always keeps one label per box. `ef_decoder_multi_label(d)` reports the setting a built decoder ended up with, including one taken from metadata. Multi-label candidates go through the NMS mode unchanged, so `EF_NMS_CLASS_AGNOSTIC` suppresses across the classes of one anchor.

`ef_decoder_params_set_pre_nms_top_k` caps how many candidates reach NMS; `0` means no limit. Left unset, the cap is 300, or 30 000 (Ultralytics' `max_nms`) when multi-label decode is on; for validation at a low score threshold, pass `0`.

Inference never guesses: metadata and shapes are cross-checked, and a
disagreement — a class count that does not fit the output width, a `segment`
task with no prototype tensor, an unsupported task such as pose or OBB — is
reported through `err_out` rather than resolved by preference.

## Model configuration

`ef_model_config_resolve` finds what a model needs to be run and decoded — output schema, labels, input preprocessing — from what it carries: an `edgefirst.json` embedded in the model or beside it, Hugging Face `config.json`/`preprocessor_config.json` embedded or beside it, Ultralytics metadata, Hugging Face input/output names, and the documented defaults, in that order, field by field. String overrides apply last, and every value records its source.

The preprocessing crosses to `libedgefirst_image` as an `ef_preprocess_spec`, declared in `edgefirst/detect.h` which both headers include, so the two libraries never link each other:

```c
#include <edgefirst/decoder.h>
#include <edgefirst/image.h>

/* Signals as for schema inference; a Hugging Face config needs them. */
ef_infer_signals *s = ef_infer_signals_new(EF_MODEL_SOURCE_ONNX);
const uintptr_t in_shape[4] = { 1, 3, 512, 512 };
const uintptr_t out_shape[4] = { 1, 150, 128, 128 };
ef_infer_signals_add_input(s, "pixel_values", in_shape, 4, EF_DTYPE_F32);
ef_infer_signals_add_output(s, "logits", out_shape, 4, EF_DTYPE_F32, NULL, NULL, 0);

ef_model_overrides *o = ef_model_overrides_new();
char *err = NULL;
if (ef_model_overrides_set(o, "resize", "letterbox-topleft", &err) != 0) { /* print err */ }

ef_model_config *cfg = ef_model_config_resolve("segformer/model.onnx", s, o, &err);
ef_model_overrides_free(o);
ef_infer_signals_free(s);
if (!cfg) {
    fprintf(stderr, "%s\n", err ? err : "(no message)"); /* errno says why */
    ef_decoder_string_free(err);
    return 1;
}

char *lines = ef_model_config_describe(cfg); /* "normalize=imagenet [preprocessor_config.json]" ... */
puts(lines);
ef_decoder_string_free(lines);

/* Decoder: the resolved schema. */
char *schema = ef_model_config_schema_json(cfg);
ef_decoder_params *p = ef_decoder_params_new();
ef_decoder_params_set_config_json(p, schema, 0);
ef_decoder *d = ef_decoder_new(p);
ef_decoder_params_free(p);
ef_decoder_string_free(schema);

/* Image: the preprocessing, as plain values. */
ef_preprocess_spec spec = ef_preprocess_spec_default();
ef_model_config_preprocessing(cfg, &spec);
ef_convert_options *conv = ef_convert_options_from_spec(&spec); /* interpolation too */
ef_image_processor_convert(proc, frame, input, conv);

/* Semantic decode over the content the convert painted. */
float region[4];
ef_convert_options_content_region(conv, frame_w, frame_h, spec.width, spec.height, NULL, region);
ef_semantic_options *so = ef_semantic_options_new();
ef_semantic_options_set_content_region(so, region);
ef_decoder_decode_semantic(d, outputs, 1, class_map, NULL, so);
```

`ef_model_config_source(cfg, EF_CONFIG_FIELD_*, &source, &detail)` reports where a value came from as an `EF_CONFIG_SOURCE_*` code and a detail string; `_family`, `_task` and `_decode_mode` report codes, and `_labels_json`, `_description` and `_notes` strings, all freed with `ef_decoder_string_free`. Override keys are `task`, `normalize`, `resize`, `decode` and `labels` (comma-separated names, e.g. `ef_model_overrides_set(o, "labels", "background,person", &err)`); the grammar is documented on `ef_model_overrides_set`, and `ef_model_config_describe` prints those keys in it. A bad key or value is `EINVAL` with the accepted forms in `err`.

`ef_model_config_preprocessing` gives the model's preprocessing as its configuration states it. The image library runs bicubic and area interpolation as bilinear, and `ef_model_config_notes` says so.

`ef_model_config_resolve` returns `NULL` and sets `errno`: `ENOENT` for a missing file, `ENODATA` for a configuration nothing supplies (the message names the override that would), `EBADMSG` for a malformed container or config file, `ENOTSUP` for one the library cannot run, `EISDIR`, `EFBIG` or `EILSEQ` for a path that is a directory or a sidecar that is too large or not UTF-8, `EINVAL` for a bad argument or an override that does not apply.

`ef_tile_placement.content_region` and `ef_semantic_options_set_content_region` name the normalized `[xmin, ymin, xmax, ymax]` rectangle the image occupies in the model input.

<img referrerpolicy="no-referrer" src="https://px.edgefirst.ai/a.png?x-pxid=37aa7de2-6935-484e-b31d-9fd9a57fcd63" alt="" width="1" height="1" style="position:absolute; width:1px; height:1px; opacity:0; pointer-events:none;" />
