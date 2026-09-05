// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0
//
// Golden sizes and offsets for every public by-value struct this library
// declares. This is the drift class no name-level check can see; identical on
// all LP64 targets. A failure here means a layout moved: before 1.0 that is
// allowed (see the ABI note in the root README), but it must be deliberate,
// and it must come with a bump to `ef_decoder_abi_version`.

#include "edgefirst/decoder.h"

#include <stddef.h>

_Static_assert(sizeof(ef_detect_box) == 24, "detect box frozen at 24");
_Static_assert(offsetof(ef_detect_box, xmin) == 0, "");
_Static_assert(offsetof(ef_detect_box, ymin) == 4, "");
_Static_assert(offsetof(ef_detect_box, xmax) == 8, "");
_Static_assert(offsetof(ef_detect_box, ymax) == 12, "");
_Static_assert(offsetof(ef_detect_box, score) == 16, "");
_Static_assert(offsetof(ef_detect_box, label) == 20, "");

/* Four f32s, then a pointer (align 8), then two u32s. */
_Static_assert(sizeof(ef_segmentation) == 32, "segmentation frozen at 32");
_Static_assert(offsetof(ef_segmentation, xmin) == 0, "");
_Static_assert(offsetof(ef_segmentation, ymin) == 4, "");
_Static_assert(offsetof(ef_segmentation, xmax) == 8, "");
_Static_assert(offsetof(ef_segmentation, ymax) == 12, "");
_Static_assert(offsetof(ef_segmentation, mask) == 16, "");
_Static_assert(offsetof(ef_segmentation, width) == 24, "");
_Static_assert(offsetof(ef_segmentation, height) == 28, "");

/* max_det is uintptr_t, so 4 bytes of pad after class_agnostic. */
_Static_assert(sizeof(ef_merge_config) == 32, "merge config is 32 bytes");
_Static_assert(offsetof(ef_merge_config, metric) == 0, "");
_Static_assert(offsetof(ef_merge_config, threshold) == 4, "");
_Static_assert(offsetof(ef_merge_config, class_agnostic) == 8, "");
_Static_assert(offsetof(ef_merge_config, max_det) == 16, "");
_Static_assert(offsetof(ef_merge_config, score_threshold) == 24, "");
/* `mode` fills what was the tail pad, so the struct is 32 bytes with or
   without it and every earlier offset is unmoved. */
_Static_assert(offsetof(ef_merge_config, mode) == 28, "");

/* Two uintptr_t, four f32, c_int, letterbox[4], two f32: 60, align 8 -> 64. */
_Static_assert(sizeof(ef_tile_placement) == 64, "tile placement frozen at 64");
_Static_assert(offsetof(ef_tile_placement, index) == 0, "");
_Static_assert(offsetof(ef_tile_placement, count) == 8, "");
_Static_assert(offsetof(ef_tile_placement, origin_x) == 16, "");
_Static_assert(offsetof(ef_tile_placement, origin_y) == 20, "");
_Static_assert(offsetof(ef_tile_placement, crop_width) == 24, "");
_Static_assert(offsetof(ef_tile_placement, crop_height) == 28, "");
_Static_assert(offsetof(ef_tile_placement, has_letterbox) == 32, "");
_Static_assert(offsetof(ef_tile_placement, letterbox) == 36, "");
_Static_assert(offsetof(ef_tile_placement, frame_width) == 52, "");
_Static_assert(offsetof(ef_tile_placement, frame_height) == 56, "");

int main(void) { return 0; }
