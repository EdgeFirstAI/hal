// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0
//
// Golden sizes and offsets for every public by-value struct. These are the
// drift class no name-level check can see; identical on all LP64 targets.
// A failure here means the frozen-forever rule was violated -- the fix is a
// suffixed successor struct, never an in-place edit.

#include "edgefirst/tensor.h"

#include <stddef.h>

_Static_assert(sizeof(ef_tensor_plane) == 48, "plane frozen at 48");
_Static_assert(offsetof(ef_tensor_plane, handle) == 0, "");
_Static_assert(offsetof(ef_tensor_plane, offset) == 8, "");
_Static_assert(offsetof(ef_tensor_plane, stride) == 16, "");
_Static_assert(offsetof(ef_tensor_plane, size) == 24, "");
_Static_assert(offsetof(ef_tensor_plane, used) == 32, "");
_Static_assert(offsetof(ef_tensor_plane, modifier) == 40, "");

_Static_assert(sizeof(ef_tensor_view) == 16, "view frozen at 16");
_Static_assert(offsetof(ef_tensor_view, ptr) == 0, "");
_Static_assert(offsetof(ef_tensor_view, len) == 8, "");

/* Two u64s force align 8, so the seven trailing u32s pad 44 -> 48. */
_Static_assert(sizeof(ef_image_desc_view) == 48, "image desc view frozen at 48");
_Static_assert(offsetof(ef_image_desc_view, width) == 0, "");
_Static_assert(offsetof(ef_image_desc_view, height) == 8, "");
_Static_assert(offsetof(ef_image_desc_view, format) == 16, "");
_Static_assert(offsetof(ef_image_desc_view, dtype) == 20, "");
_Static_assert(offsetof(ef_image_desc_view, access) == 24, "");
_Static_assert(offsetof(ef_image_desc_view, memory) == 28, "");
_Static_assert(offsetof(ef_image_desc_view, has_memory) == 32, "");
_Static_assert(offsetof(ef_image_desc_view, compression) == 36, "");
_Static_assert(offsetof(ef_image_desc_view, has_compression) == 40, "");

/* Five u64s + u32, align 8, pad 44 -> 48. */
_Static_assert(sizeof(EfViewOrigin) == 48, "view origin frozen at 48");
_Static_assert(offsetof(EfViewOrigin, parent_width) == 0, "");
_Static_assert(offsetof(EfViewOrigin, parent_height) == 8, "");
_Static_assert(offsetof(EfViewOrigin, parent_row_stride) == 16, "");
_Static_assert(offsetof(EfViewOrigin, x) == 24, "");
_Static_assert(offsetof(EfViewOrigin, y) == 32, "");
_Static_assert(offsetof(EfViewOrigin, has_origin) == 40, "");

/* Five u32s, no wider field to force alignment: 20 bytes at align 4. */
_Static_assert(sizeof(ef_d3d11_layout) == 20, "d3d11 layout frozen at 20");
_Static_assert(offsetof(ef_d3d11_layout, dxgi_format) == 0, "");
_Static_assert(offsetof(ef_d3d11_layout, texture_width) == 4, "");
_Static_assert(offsetof(ef_d3d11_layout, texture_height) == 8, "");
_Static_assert(offsetof(ef_d3d11_layout, bytes_per_texel) == 12, "");
_Static_assert(offsetof(ef_d3d11_layout, gl_internal_format) == 16, "");

_Static_assert(sizeof(ef_quantization_info) == 12, "quantization info frozen at 12");
_Static_assert(offsetof(ef_quantization_info, axis) == 0, "");
_Static_assert(offsetof(ef_quantization_info, count) == 4, "");
_Static_assert(offsetof(ef_quantization_info, has_quantization) == 8, "");

int main(void) { return 0; }
