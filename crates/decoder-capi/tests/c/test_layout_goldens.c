// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0
//
// Golden sizes and offsets for every public by-value struct this library
// declares. This is the drift class no name-level check can see; identical on
// all LP64 targets. A failure here means the frozen-forever rule was
// violated -- the fix is a suffixed successor struct, never an in-place edit.

#include "edgefirst/decoder.h"

#include <stddef.h>

_Static_assert(sizeof(ef_detect_box) == 24, "detect box frozen at 24");
_Static_assert(offsetof(ef_detect_box, xmin) == 0, "");
_Static_assert(offsetof(ef_detect_box, ymin) == 4, "");
_Static_assert(offsetof(ef_detect_box, xmax) == 8, "");
_Static_assert(offsetof(ef_detect_box, ymax) == 12, "");
_Static_assert(offsetof(ef_detect_box, score) == 16, "");
_Static_assert(offsetof(ef_detect_box, label) == 20, "");

int main(void) { return 0; }
