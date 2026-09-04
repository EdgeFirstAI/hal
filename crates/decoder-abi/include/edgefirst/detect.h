#ifndef EDGEFIRST_DETECT_H
#define EDGEFIRST_DETECT_H

/**
 * @file detect.h
 * @brief Header-only detection vocabulary (boxes, masks, tile placement).
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
    int has_letterbox;
    float letterbox[4];
    float frame_width;
    float frame_height;
} ef_tile_placement;

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* EDGEFIRST_DETECT_H */
