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

int main(void) { return 0; }
