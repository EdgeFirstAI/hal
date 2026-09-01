#ifndef EDGEFIRST_DECODER_H
#define EDGEFIRST_DECODER_H

/**
 * @file decoder.h
 * @brief EdgeFirst model output decoding C API
 *
 * SPDX-License-Identifier: Apache-2.0
 * Copyright (c) 2026 Au-Zone Technologies. All Rights Reserved.
 *
 * Turns raw model output tensors into detections and segmentation masks.
 *
 * Configuration is a builder: set fields on an `ef_decoder_params`, then build
 * a decoder from it. Exactly one configuration source must be supplied — an
 * output list, JSON, YAML, or a file — because two sources disagreeing has no
 * defined resolution.
 *
 * Detections come back as `ef_detect_box_list`. The box / mask / tile
 * layouts themselves live in header-only `edgefirst/detect.h`.
 */

#include <stdarg.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>

#include "edgefirst/detect.h"
#include "edgefirst/tensor.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief An opaque decoder.
 *
 * Forward-declared so callers write `ef_decoder *` without the `struct`
 * keyword. Its definition is private: decoder state is not part of the ABI.
 */
typedef struct ef_decoder ef_decoder;

/** @brief Opaque decoder configuration, built up field by field. */
typedef struct ef_decoder_params ef_decoder_params;

/** @brief An opaque list of segmentation masks. */
typedef struct ef_segmentation_list ef_segmentation_list;

/** @brief Prototype tensors from `ef_decoder_decode_proto`. */
typedef struct ef_proto_data ef_proto_data;

/** @brief Decoder-local ByteTrack handle. */
typedef struct ef_decoder_tracker ef_decoder_tracker;

/** @brief Track list from `ef_decoder_decode_tracked`. */
typedef struct ef_decoder_track_list ef_decoder_track_list;

/**
 * @name EF_INFER_DTYPE_*
 * @brief Numeric dtype codes for `ef_infer_signals_add_input`/
 * `ef_infer_signals_add_output`.
 *
 * Mirrors `edgefirst_decoder::schema::DType`'s declaration order. This is a
 * distinct vocabulary from `edgefirst/tensor.h`'s `EF_DTYPE_*` (a wider,
 * differently-ordered enum for physical tensor storage) -- schema dtype is
 * the narrower quantized/float set a model's logical I/O carries. As with
 * `EF_DTYPE_*`, the functions that take one keep a plain `uint32_t`
 * parameter rather than this enum: an out-of-range code is rejected with
 * `EINVAL` rather than transmuted into a Rust enum, which would be
 * undefined behaviour for a value no variant names.
 *
 * The codes start at `0x100` so the two vocabularies cannot overlap. Both
 * cross as bare `uint32_t`, so had these numbered from `0` every value
 * would have been a valid code in BOTH enums meaning something different
 * -- `7` is FLOAT32 here and `EF_DTYPE_I64` there, and `0`/`1` invert
 * signedness. A caller passing the tensor dtype code it already holds
 * would have been silently misread rather than rejected. Disjoint ranges
 * make that mistake an `EINVAL`.
 */
#define EF_INFER_DTYPE_INT8 0x100
#define EF_INFER_DTYPE_UINT8 0x101
#define EF_INFER_DTYPE_INT16 0x102
#define EF_INFER_DTYPE_UINT16 0x103
#define EF_INFER_DTYPE_INT32 0x104
#define EF_INFER_DTYPE_UINT32 0x105
#define EF_INFER_DTYPE_FLOAT16 0x106
#define EF_INFER_DTYPE_FLOAT32 0x107


/* Generated with cbindgen:0.29.4 */

/* WARNING: The generated portion of this file is produced by cbindgen. Do not modify it directly. */

#include <stdarg.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>

/**
 * An owned list of detections.
 *
 * Opaque and heap-owned rather than a `(ptr, len)` pair, so a consumer cannot
 * outlive the producer's allocation by holding the pointer. This is the one
 * implementation of `ef_detect_box_list`: the tracker and any other consumer
 * read a list via [`ef_detect_box_list_data`]/[`ef_detect_box_list_len`]
 * without linking this library.
 */
typedef struct ef_detect_box_list ef_detect_box_list;

/**
 * Raw model I/O signals, accumulated field by field.
 */
typedef struct ef_infer_signals ef_infer_signals;

/**
 * An inferred Ultralytics schema. Its JSON views are rendered on demand by
 * [`ef_inferred_schema_json`]/[`ef_inferred_schema_labels_json`].
 */
typedef struct ef_inferred_schema ef_inferred_schema;

/**
 * Accumulates detections from every tile of one frame.
 */
typedef struct ef_tiled_frame_accumulator ef_tiled_frame_accumulator;

/**
 * One track written by [`ef_decoder_decode_tracked`].
 */
typedef struct ef_decoder_track {
  uint8_t uuid[16];
  float xmin;
  float ymin;
  float xmax;
  float ymax;
  int32_t count;
  uint64_t created;
  uint64_t last_updated;
} ef_decoder_track;

/**
 * ABI version of this library's C surface.
 */
uint32_t ef_decoder_abi_version(void);

/**
 * Create an empty detection list. `NULL` on allocation failure.
 */
struct ef_detect_box_list *ef_detect_box_list_new(void);

/**
 * Append a detection.
 *
 * @return 0 on success, `EINVAL` for a null list or box.
 *
 * # Safety
 * `l` and `b` must be `NULL` or valid.
 */
int ef_detect_box_list_push(struct ef_detect_box_list *l, const ef_detect_box *b);

/**
 * Create decoder parameters with the library's defaults.
 */
ef_decoder_params *ef_decoder_params_new(void);

/**
 * Free decoder parameters. Freeing `NULL` is a no-op.
 *
 * # Safety
 * `p` must be `NULL` or have come from this library.
 */
void ef_decoder_params_free(ef_decoder_params *p);

/**
 * Minimum confidence for a detection to be kept.
 *
 * Written out rather than macro-generated: **cbindgen does not expand macros**,
 * so a macro-defined `extern "C"` fn is exported by the library and absent
 * from the header — present in `nm`, undeclarable by any C caller. That is a
 * silent break, caught by the leaf's header-parity tests.
 *
 * # Safety
 * `p` must be `NULL` or a live parameter set.
 */
int ef_decoder_params_set_score_threshold(ef_decoder_params *p, float v);

/**
 * IoU above which NMS suppresses the lower-scoring box.
 *
 * # Safety
 * `p` must be `NULL` or a live parameter set.
 */
int ef_decoder_params_set_iou_threshold(ef_decoder_params *p, float v);

/**
 * How many candidates survive into NMS. Bounds the worst case.
 *
 * # Safety
 * `p` must be `NULL` or a live parameter set.
 */
int ef_decoder_params_set_pre_nms_top_k(ef_decoder_params *p, uintptr_t v);

/**
 * Maximum detections returned per frame.
 *
 * # Safety
 * `p` must be `NULL` or a live parameter set.
 */
int ef_decoder_params_set_max_det(ef_decoder_params *p, uintptr_t v);

/**
 * Set the model's input dimensions, when the config does not carry them.
 *
 * # Safety
 * `p` must be `NULL` or a live parameter set.
 */
int ef_decoder_params_set_input_dims(ef_decoder_params *p, uintptr_t width, uintptr_t height);

/**
 * NMS mode: 0 = off, 1 = automatic, 2 = class-aware, 3 = class-agnostic.
 *
 * # Safety
 * `p` must be `NULL` or a live parameter set.
 */
int ef_decoder_params_set_nms(ef_decoder_params *p, uint32_t nms);

/**
 * Configure from a JSON string. `len` may be 0 for NUL-terminated.
 *
 * # Safety
 * `json` must be readable for `len` bytes, or NUL-terminated.
 */
int ef_decoder_params_set_config_json(ef_decoder_params *p, const char *json, uintptr_t len);

/**
 * Configure from a YAML string. `len` may be 0 for NUL-terminated.
 *
 * # Safety
 * `yaml` must be readable for `len` bytes, or NUL-terminated.
 */
int ef_decoder_params_set_config_yaml(ef_decoder_params *p, const char *yaml, uintptr_t len);

/**
 * Configure from a file, JSON or YAML detected by extension and content.
 *
 * # Safety
 * `path` must be a NUL-terminated string.
 */
int ef_decoder_params_set_config_file(ef_decoder_params *p, const char *path);

/**
 * Build a decoder. `NULL` on failure.
 *
 * **Exactly one** configuration source must be set — JSON, YAML, or a file.
 * Two sources disagreeing has no defined resolution, so supplying none or more
 * than one is an error rather than a precedence rule nobody remembers.
 *
 * The parameters are not consumed and may be reused.
 *
 * # Safety
 * `p` must be a live parameter set.
 */
ef_decoder *ef_decoder_new(const ef_decoder_params *p);

/**
 * Free a decoder. Freeing `NULL` is a no-op.
 *
 * # Safety
 * `d` must be `NULL` or have come from this library.
 */
void ef_decoder_free(ef_decoder *d);

/**
 * The model's input dimensions, when known.
 *
 * @return 0 on success, `ENODATA` when the configuration did not declare them.
 *
 * # Safety
 * `width` and `height` must be writable.
 */
int ef_decoder_input_dims(const ef_decoder *d, uintptr_t *width, uintptr_t *height);

/**
 * Whether the model emits normalized box coordinates.
 *
 * @return 1 yes, 0 no, -1 when the configuration does not say.
 *
 * # Safety
 * `d` must be `NULL` or live.
 */
int ef_decoder_normalized_boxes(const ef_decoder *d);

/**
 * The model type as a NUL-terminated string the caller must free with
 * [`ef_decoder_string_free`].
 *
 * # Safety
 * `d` must be `NULL` or live.
 */
char *ef_decoder_model_type(const ef_decoder *d);

/**
 * Free a string this library returned. Freeing `NULL` is a no-op.
 *
 * Its own entry point rather than `free(3)`: the allocation came from Rust,
 * and on some platforms the two allocators are not the same.
 *
 * # Safety
 * `s` must be `NULL` or have come from this library.
 */
void ef_decoder_string_free(char *s);

/**
 * Decode model outputs into detections and, for segmentation models, masks.
 *
 * `out_masks` may be `NULL` when masks are not wanted.
 *
 * @return 0 on success. Both out-parameters are written only on success, so a
 *         caller never has to free a partially-populated result.
 *
 * # Safety
 * `outputs` must point to `count` live tensor handles.
 */
int ef_decoder_decode(const ef_decoder *d,
                      const ef_tensor *const *outputs,
                      uintptr_t count,
                      struct ef_detect_box_list **out_boxes,
                      ef_segmentation_list **out_masks);

/**
 * Number of detections. Zero for a `NULL` list.
 *
 * # Safety
 * `l` must be `NULL` or valid.
 */
uintptr_t ef_detect_box_list_len(const struct ef_detect_box_list *l);

/**
 * Copy detection `index` into `out`.
 *
 * Copies rather than lending a pointer: a borrowed element would dangle the
 * moment the list is freed or grown, and C gives the caller no way to notice.
 *
 * @return 0 on success, `EINVAL` for a null argument or out-of-range index.
 *
 * # Safety
 * `out` must be writable.
 */
int ef_detect_box_list_get(const struct ef_detect_box_list *l, uintptr_t index, ef_detect_box *out);

/**
 * Borrow the detections as a C array, for `ef_bytetrack_update`.
 *
 * A plain array rather than an opaque handle, so the tracker reads it without
 * linking this library.
 *
 * # Safety
 * `l` must be `NULL` or valid.
 */
const ef_detect_box *ef_detect_box_list_data(const struct ef_detect_box_list *l);

/**
 * Free a detection list. Freeing `NULL` is a no-op.
 *
 * There is exactly one implementation of this type, in
 * `libedgefirst_decoder`, so any `ef_detect_box_list *` from any EdgeFirst
 * entry point is freed with this function.
 *
 * # Safety
 * `l` must be `NULL` or have come from this library.
 */
void ef_detect_box_list_free(struct ef_detect_box_list *l);

/**
 * Number of masks. Zero for a `NULL` list.
 *
 * # Safety
 * `l` must be `NULL` or valid.
 */
uintptr_t ef_segmentation_list_len(const ef_segmentation_list *l);

/**
 * The mask region for entry `index`, in normalized coordinates.
 *
 * These bound the **mask region**, which is snapped to the proto grid and so
 * encloses — rather than equals — the companion detection's box.
 *
 * @return 0 on success, `EINVAL` for a null argument or bad index.
 *
 * # Safety
 * All out-parameters must be writable.
 */
int ef_segmentation_list_get_bbox(const ef_segmentation_list *l,
                                  uintptr_t index,
                                  float *xmin,
                                  float *ymin,
                                  float *xmax,
                                  float *ymax);

/**
 * Borrow the masks as a C array.
 *
 * Materialises the borrowed views on first call and caches them, so repeated
 * calls are free and every returned pointer stays valid for the list's life.
 *
 * This is what lets `libedgefirst-image` draw decoder output without linking
 * this library: it reads plain values and a borrowed byte pointer, needing no
 * shared allocator and no symbol to resolve.
 *
 * @return the first element, or `NULL` for a null or empty list. Pair with
 *         [`ef_segmentation_list_len`].
 *
 * # Safety
 * `l` must be `NULL` or valid, and must outlive any use of the result.
 */
const ef_segmentation *ef_segmentation_list_data(ef_segmentation_list *l);

/**
 * Append a programmatic output spec. Returns the new index, or `-1`.
 *
 * `type_`: 0 detection, 1 boxes, 2 scores, 3 protos, 4 segmentation,
 * 5 mask coefficients, 6 mask, 7 classes.
 * `decoder`: 0 ultralytics, 1 modelpack.
 *
 * # Safety
 * `shape` must point to `ndim` sizes; `dims` may be NULL.
 */
int ef_decoder_params_add_output(ef_decoder_params *p,
                                 uint32_t type,
                                 uint32_t decoder,
                                 const uintptr_t *shape,
                                 const uint32_t *dims,
                                 uintptr_t ndim);

/**
 * Set quantization on output `index`.
 *
 * # Safety
 * `p` must be `NULL` or a live handle from this library.
 */
int ef_decoder_params_output_set_quantization(ef_decoder_params *p,
                                              int index,
                                              float scale,
                                              int zero_point);

/**
 * Set anchors on a detection output. `anchors` is `num_anchors` pairs.
 *
 * # Safety
 * `p` must be live; `anchors` must point to `num_anchors` pairs.
 */
int ef_decoder_params_output_set_anchors(ef_decoder_params *p,
                                         int index,
                                         const float (*anchors)[2],
                                         uintptr_t num_anchors);

/**
 * Mark a detection/boxes output as normalized (`1`) or pixel (`0`).
 *
 * # Safety
 * `p` must be `NULL` or a live handle from this library.
 */
int ef_decoder_params_output_set_normalized(ef_decoder_params *p, int index, int normalized);

/**
 * Decoder version: 0 Yolov5, 1 Yolov8, 2 Yolo11, 3 Yolo26.
 *
 * # Safety
 * `p` must be `NULL` or a live handle from this library.
 */
int ef_decoder_params_set_decoder_version(ef_decoder_params *p, uint32_t version);

/**
 * Decode detections and, for segmentation models, proto tensors.
 *
 * # Safety
 * `outputs` must point to `count` live handles.
 */
ef_proto_data *ef_decoder_decode_proto(const ef_decoder *d,
                                       const ef_tensor *const *outputs,
                                       uintptr_t count,
                                       struct ef_detect_box_list **out_boxes);

/**
 * Free proto data. NULL is a no-op.
 *
 * # Safety
 * `proto` must be `NULL` or have come from this library.
 */
void ef_proto_data_free(ef_proto_data *proto);

/**
 * Proto layout: 0 NHWC, 1 NCHW. `-1` for NULL.
 *
 * # Safety
 * `proto` must be `NULL` or a live handle from this library.
 */
int32_t ef_proto_data_layout(const ef_proto_data *proto);

/**
 * Take ownership of the proto tensor. NULL if already taken.
 *
 * # Safety
 * `proto` must be `NULL` or a live handle from this library.
 */
ef_tensor *ef_proto_data_take_protos(ef_proto_data *proto);

/**
 * Take ownership of the mask-coefficient tensor. NULL if already taken.
 *
 * # Safety
 * `proto` must be `NULL` or a live handle from this library.
 */
ef_tensor *ef_proto_data_take_mask_coefficients(ef_proto_data *proto);

/**
 * Dequantize an integer tensor into a pre-allocated f32 tensor.
 *
 * # Safety
 * `input` and `output` must be live handles.
 */
int ef_dequantize(const ef_tensor *input, float scale, int zero_point, ef_tensor *output);

/**
 * Convert segmentation `index` to a new `[H, W]` u8 tensor.
 *
 * # Safety
 * `list` must be a live handle from this library.
 */
ef_tensor *ef_segmentation_to_mask(const ef_segmentation_list *list, uintptr_t index);

ef_decoder_tracker *ef_decoder_tracker_new(void);

/**
 * Free a decoder-local tracker. NULL is a no-op.
 *
 * # Safety
 * `t` must be `NULL` or have come from [`ef_decoder_tracker_new`].
 */
void ef_decoder_tracker_free(ef_decoder_tracker *t);

/**
 * Number of tracks. Zero for NULL.
 *
 * # Safety
 * `l` must be `NULL` or a live handle from this library.
 */
uintptr_t ef_decoder_track_list_len(const ef_decoder_track_list *l);

/**
 * Copy track `index` into `out`. Returns 0 on success.
 *
 * # Safety
 * `l` and `out` must be live or NULL as documented.
 */
int ef_decoder_track_list_get(const ef_decoder_track_list *l,
                              uintptr_t index,
                              struct ef_decoder_track *out);

/**
 * Free a decoder track list. NULL is a no-op.
 *
 * # Safety
 * `l` must be `NULL` or have come from this library.
 */
void ef_decoder_track_list_free(ef_decoder_track_list *l);

/**
 * Decode and update a decoder-owned tracker.
 *
 * # Safety
 * `outputs` must point to `count` live handles.
 */
int ef_decoder_decode_tracked(const ef_decoder *d,
                              ef_decoder_tracker *tracker,
                              uint64_t timestamp,
                              const ef_tensor *const *outputs,
                              uintptr_t count,
                              struct ef_detect_box_list **out_boxes,
                              ef_segmentation_list **out_masks,
                              ef_decoder_track_list **out_tracks);

/**
 * Free a mask list. Freeing `NULL` is a no-op.
 *
 * # Safety
 * `l` must be `NULL` or have come from this library.
 */
void ef_segmentation_list_free(ef_segmentation_list *l);

/**
 * Create empty signals for a model read from `source` (`0` onnx, `1`
 * tflite, `2` other). `NULL` for an unrecognized source or allocation
 * failure.
 */
struct ef_infer_signals *ef_infer_signals_new(uint32_t source);

/**
 * Free signals. Freeing `NULL` is a no-op.
 *
 * # Safety
 * `s` must be `NULL` or have come from this library.
 */
void ef_infer_signals_free(struct ef_infer_signals *s);

/**
 * Append an input tensor. `dtype` is an `EF_INFER_DTYPE_*` code.
 *
 * @return 0 on success, `EINVAL` for a null/invalid argument.
 *
 * # Safety
 * `name` must be NUL-terminated; `shape` must point to `rank` sizes.
 */
int ef_infer_signals_add_input(struct ef_infer_signals *s,
                               const char *name,
                               const uintptr_t *shape,
                               uintptr_t rank,
                               uint32_t dtype);

/**
 * Append an output tensor, with optional quantization.
 *
 * `quant_len` is `0` for an unquantized tensor or `1` for per-tensor
 * quantization; `scale` and `zero_point` (when non-NULL) each carry
 * `quant_len` entries. `zero_point` may be `NULL` for symmetric
 * quantization.
 *
 * A `quant_len` above `1` describes per-channel quantization, which this
 * setter accepts but [`ef_infer_ultralytics_schema`] then refuses: the
 * decoder consumes per-tensor quantization only, so such a schema would
 * build a decoder that fails. The refusal is deferred to inference so the
 * error arrives on the call that reports errors, with a message naming
 * the offending tensor.
 *
 * @return 0 on success, `EINVAL` for a null/invalid argument (including a
 *         nonzero `quant_len` with a `NULL` `scale`).
 *
 * # Safety
 * `name` must be NUL-terminated; `shape` must point to `rank` sizes;
 * `scale`/`zero_point` must point to `quant_len` elements when non-NULL.
 */
int ef_infer_signals_add_output(struct ef_infer_signals *s,
                                const char *name,
                                const uintptr_t *shape,
                                uintptr_t rank,
                                uint32_t dtype,
                                const float *scale,
                                const int32_t *zero_point,
                                uintptr_t quant_len);

/**
 * Insert a metadata key/value pair, as captured verbatim from the model's
 * container format (ONNX `metadata_props`, TFLite `metadata.json`).
 *
 * @return 0 on success, `EINVAL` for a null handle or unreadable string.
 *
 * # Safety
 * `key` and `value` must be NUL-terminated.
 */
int ef_infer_signals_add_metadata(struct ef_infer_signals *s, const char *key, const char *value);

/**
 * Infer an Ultralytics schema from accumulated signals.
 *
 * `NULL` on failure. When `err_out` is non-NULL, `*err_out` is set to a
 * message the caller frees with `ef_decoder_string_free`.
 *
 * **Initialize your `char *` to `NULL` before calling.** `*err_out` is
 * left untouched on success, and while every failure path writes a
 * message, the write itself can still fail (a message carrying an
 * embedded NUL, or the allocation behind it). Detect failure from the
 * returned handle, and test `*err_out` separately before reading it:
 *
 * ```c
 * char *err = NULL;
 * ef_inferred_schema *r = ef_infer_ultralytics_schema(s, &err);
 * if (!r) {
 *     fprintf(stderr, "%s\n", err ? err : "(no message)");
 *     ef_decoder_string_free(err); // freeing NULL is a no-op
 * }
 * ```
 *
 * # Safety
 * `s` must be `NULL` or a live handle from this library; `err_out` must be
 * `NULL` or writable.
 */
struct ef_inferred_schema *ef_infer_ultralytics_schema(const struct ef_infer_signals *s,
                                                       char **err_out);

/**
 * The inferred schema as `edgefirst.json` schema v2 JSON. The caller frees
 * the result with `ef_decoder_string_free`. `NULL` for a `NULL` handle or
 * on serialization failure.
 *
 * # Safety
 * `r` must be `NULL` or a live handle from this library.
 */
char *ef_inferred_schema_json(const struct ef_inferred_schema *r);

/**
 * The inferred class labels as a JSON array of strings, in index order.
 * The caller frees the result with `ef_decoder_string_free`. `NULL` for a
 * `NULL` handle or on serialization failure.
 *
 * # Safety
 * `r` must be `NULL` or a live handle from this library.
 */
char *ef_inferred_schema_labels_json(const struct ef_inferred_schema *r);

/**
 * A human-readable summary, e.g. "Ultralytics YOLO26 segment, 80 classes".
 * The caller frees the result with `ef_decoder_string_free`. `NULL` for a
 * `NULL` handle.
 *
 * # Safety
 * `r` must be `NULL` or a live handle from this library.
 */
char *ef_inferred_schema_description(const struct ef_inferred_schema *r);

/**
 * Free an inferred schema. Freeing `NULL` is a no-op.
 *
 * # Safety
 * `r` must be `NULL` or have come from this library.
 */
void ef_inferred_schema_free(struct ef_inferred_schema *r);

/**
 * Fill `out` with the library's default merge configuration.
 *
 * # Safety
 * `out` must be writable.
 */
int ef_merge_config_default(ef_merge_config *out);

/**
 * Create an accumulator for a frame of `tiles_total` tiles.
 *
 * # Safety
 * `cfg` must be valid.
 */
struct ef_tiled_frame_accumulator *ef_tiled_frame_accumulator_new(float frame_width,
                                                                  float frame_height,
                                                                  uintptr_t tiles_total,
                                                                  const ef_merge_config *cfg,
                                                                  uintptr_t est_per_tile);

/**
 * Free an accumulator. Freeing `NULL` is a no-op.
 *
 * # Safety
 * `a` must be `NULL` or have come from this library.
 */
void ef_tiled_frame_accumulator_free(struct ef_tiled_frame_accumulator *a);

/**
 * Add one tile's detections.
 *
 * **Idempotent per tile index.** A duplicate index, an out-of-range one, or a
 * placement from a different plan (its `count` disagreeing with this
 * accumulator's tile total) is ignored and its detections dropped. That is
 * what makes out-of-order *and* at-least-once delivery converge to the same
 * frame — a retried tile does not double-count, and a tile from another
 * frame cannot corrupt this one's fan-in.
 *
 * Use [`ef_tiled_frame_accumulator_is_complete`] to test for completion; this
 * return value answers a different question.
 *
 * @return 1 when the tile was newly accepted, 0 when it was ignored as a
 *         duplicate or foreign placement, `-1` on a bad argument or after
 *         finalize.
 *
 * # Safety
 * `boxes` must point to `count` elements; `placement` must be valid.
 */
int ef_tiled_frame_accumulator_push_tile(struct ef_tiled_frame_accumulator *a,
                                         const ef_detect_box *boxes,
                                         uintptr_t count,
                                         const ef_tile_placement *placement);

/**
 * Whether every tile has been seen.
 *
 * # Safety
 * `a` must be `NULL` or valid.
 */
int ef_tiled_frame_accumulator_is_complete(const struct ef_tiled_frame_accumulator *a);

/**
 * How many tiles are still outstanding.
 *
 * # Safety
 * `a` must be `NULL` or valid.
 */
uintptr_t ef_tiled_frame_accumulator_remaining(const struct ef_tiled_frame_accumulator *a);

/**
 * Merge every pushed tile into one detection list.
 *
 * Consumes the accumulator's contents: a second call returns `NULL`, because
 * merging is destructive and returning an empty list would be
 * indistinguishable from a frame that genuinely found nothing.
 *
 * `normalized` non-zero returns frame-normalized coordinates.
 *
 * @return a list the caller frees with `ef_detect_box_list_free`, or `NULL`.
 *
 * # Safety
 * `a` must be valid.
 */
struct ef_detect_box_list *ef_tiled_frame_accumulator_finalize(struct ef_tiled_frame_accumulator *a,
                                                               int normalized);

/**
 * Lift a tile's detections into frame coordinates.
 *
 * @return a list the caller frees with `ef_detect_box_list_free`, or `NULL`.
 *
 * # Safety
 * `boxes` must point to `count` elements; `placement` must be valid.
 */
struct ef_detect_box_list *ef_lift_tile_boxes(const ef_detect_box *boxes,
                                              uintptr_t count,
                                              const ef_tile_placement *placement);

/**
 * Merge overlapping detections that already share one coordinate space.
 *
 * @return a list the caller frees with `ef_detect_box_list_free`, or `NULL`.
 *
 * # Safety
 * `boxes` must point to `count` elements; `cfg` must be valid.
 */
struct ef_detect_box_list *ef_merge_tiled_detections(const ef_detect_box *boxes,
                                                     uintptr_t count,
                                                     const ef_merge_config *cfg);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* EDGEFIRST_DECODER_H */
