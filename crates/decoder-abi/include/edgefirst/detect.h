#ifndef EDGEFIRST_DETECT_H
#define EDGEFIRST_DETECT_H

/**
 * @file detect.h
 * @brief Header-only detection vocabulary (boxes, masks, tile placement,
 * input preprocessing).
 *
 * SPDX-License-Identifier: Apache-2.0
 * Copyright (c) 2026 Au-Zone Technologies. All Rights Reserved.
 *
 * Plain values shared by decoder, image, and tracker. There is no
 * libedgefirst_detect.so — include this header; do not link a detect library.
 * Layout matches crates/decoder-abi.
 */

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * A detection: a normalized box, a score, and a label index.
 *
 * Coordinates are normalized to `[0, 1]` against the *model input*, not the
 * source image.
 */
typedef struct ef_detect_box {
    float xmin;
    float ymin;
    float xmax;
    float ymax;
    /** Model-specific confidence; higher is more confident. */
    float score;
    /** Label index into the model's class list. */
    uint32_t label;
} ef_detect_box;

/**
 * One segmentation result, as plain values.
 *
 * `mask` borrows the producing list's buffer and is valid only until that
 * list is freed.
 */
typedef struct ef_segmentation {
    float xmin;
    float ymin;
    float xmax;
    float ymax;
    const uint8_t *mask;
    uint32_t width;
    uint32_t height;
} ef_segmentation;

/**
 * How overlapping detections from neighbouring tiles are merged.
 *
 * metric: 0 = IoU, 1 = Intersection-over-Smaller (default for seam splits).
 * mode:   0 = keep-best (default): the highest-scoring box of each matched
 *         group is kept and the boxes it matched are dropped.
 *         1 = union: the group becomes its enclosing union carrying the max
 *         score (the original GREEDYNMM merge; about 0.05 AP50 worse on
 *         every frame of the Ocean Cleanup ADIS 4K validation, TOP2-836).
 *
 * A zero-initialised struct is NOT the library default: only `mode`'s
 * default is 0. `metric` defaults to 1 (IoS), so an all-zero struct selects
 * IoU -- the metric that leaves a seam-split object as two detections, which
 * is the one thing tiled merging exists to avoid. Always fill the struct with
 * `ef_merge_config_default` and then override what you need.
 */
typedef struct ef_merge_config {
    uint32_t metric;
    float threshold;
    int class_agnostic;
    uintptr_t max_det;
    float score_threshold;
    uint32_t mode;
} ef_merge_config;

/**
 * Where one tile sits within the frame.
 */
typedef struct ef_tile_placement {
    uintptr_t index;
    uintptr_t count;
    float origin_x;
    float origin_y;
    float crop_width;
    float crop_height;
    int has_content_region;
    float content_region[4];
    float frame_width;
    float frame_height;
} ef_tile_placement;

/*
 * Option codes are fixed-width: a C23 or C++ compiler sees a `uint32_t`
 * enum, an older C compiler a `uint32_t` typedef beside the enumerators.
 */

/** How a source is fit into the model input. */
enum ef_fit
#if defined(__cplusplus) || __STDC_VERSION__ >= 202311L
  : uint32_t
#endif
{
  /** Stretch the source to fill the model input. */
  EF_FIT_STRETCH = 0,
  /** Keep the aspect ratio and pad the rest (a letterbox). */
  EF_FIT_LETTERBOX = 1,
};
#ifndef __cplusplus
#if __STDC_VERSION__ >= 202311L
typedef enum ef_fit ef_fit;
#else
typedef uint32_t ef_fit;
#endif
#endif

/** Where a letterboxed image sits. */
enum ef_letterbox_placement
#if defined(__cplusplus) || __STDC_VERSION__ >= 202311L
  : uint32_t
#endif
{
  /** Centred, padding split between both sides. */
  EF_LETTERBOX_PLACEMENT_CENTER = 0,
  /** Anchored at the top-left corner, padding on the right and bottom. */
  EF_LETTERBOX_PLACEMENT_TOP_LEFT = 1,
};
#ifndef __cplusplus
#if __STDC_VERSION__ >= 202311L
typedef enum ef_letterbox_placement ef_letterbox_placement;
#else
typedef uint32_t ef_letterbox_placement;
#endif
#endif

/** The space a letterbox pad value is expressed in. */
enum ef_pad_space
#if defined(__cplusplus) || __STDC_VERSION__ >= 202311L
  : uint32_t
#endif
{
  /** A 0-255 pixel value, normalized like the image. */
  EF_PAD_SPACE_PIXEL = 0,
  /** A value written as-is into the normalized destination. */
  EF_PAD_SPACE_NORMALIZED = 1,
};
#ifndef __cplusplus
#if __STDC_VERSION__ >= 202311L
typedef enum ef_pad_space ef_pad_space;
#else
typedef uint32_t ef_pad_space;
#endif
#endif

/**
 * A resampling filter. The image library runs bilinear and nearest;
 * bicubic and area name reference pipelines and run as bilinear.
 */
enum ef_interpolation
#if defined(__cplusplus) || __STDC_VERSION__ >= 202311L
  : uint32_t
#endif
{
  /** Bilinear, half-pixel centres (the default). */
  EF_INTERPOLATION_BILINEAR = 0,
  /** Nearest neighbour. */
  EF_INTERPOLATION_NEAREST = 1,
  /** Bicubic. */
  EF_INTERPOLATION_BICUBIC = 2,
  /** Pixel-area averaging. */
  EF_INTERPOLATION_AREA = 3,
};
#ifndef __cplusplus
#if __STDC_VERSION__ >= 202311L
typedef enum ef_interpolation ef_interpolation;
#else
typedef uint32_t ef_interpolation;
#endif
#endif

/**
 * A model's input preprocessing as plain values: produced by
 * `ef_model_config_preprocessing` (libedgefirst_decoder) and consumed by
 * `ef_convert_options_from_spec` (libedgefirst_image), so neither library
 * links the other. Start a hand-built one from `ef_preprocess_spec_default()`:
 * a zeroed struct has a zero `std` and is refused.
 *
 * `mean` and `std` normalize as `(x - mean) / std` on 0-255 values (ImageNet
 * is mean 123.675,116.28,103.53, std 58.395,57.12,57.375). `pad` is in the
 * space `pad_space` names.
 */
typedef struct ef_preprocess_spec {
    /** Model input width in pixels, 0 when unknown. */
    uint32_t width;
    /** Model input height in pixels, 0 when unknown. */
    uint32_t height;
    /** An `EF_FIT_*` code. */
    uint32_t fit;
    /** An `EF_LETTERBOX_PLACEMENT_*` code; ignored by a stretch. */
    uint32_t placement;
    /** An `EF_PAD_SPACE_*` code for `pad`. */
    uint32_t pad_space;
    /** Letterbox pad per RGB channel. */
    float pad[3];
    /** Normalization mean per RGB channel, 0-255 domain. */
    float mean[3];
    /** Normalization standard deviation per RGB channel, 0-255 domain. */
    float std[3];
    /** An `EF_INTERPOLATION_*` code. */
    uint32_t interpolation;
} ef_preprocess_spec;

/**
 * The default preprocessing: unsigned normalization (`x / 255`), a centred
 * letterbox padded with pixel value 114, bilinear, size unknown.
 */
static inline ef_preprocess_spec ef_preprocess_spec_default(void) {
    ef_preprocess_spec s;
    int i;
    s.width = 0;
    s.height = 0;
    s.fit = EF_FIT_LETTERBOX;
    s.placement = EF_LETTERBOX_PLACEMENT_CENTER;
    s.pad_space = EF_PAD_SPACE_PIXEL;
    for (i = 0; i < 3; i++) {
        s.pad[i] = 114.0f;
        s.mean[i] = 0.0f;
        s.std[i] = 255.0f;
    }
    s.interpolation = EF_INTERPOLATION_BILINEAR;
    return s;
}

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* EDGEFIRST_DETECT_H */
