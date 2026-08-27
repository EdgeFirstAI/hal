/* SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
 * SPDX-License-Identifier: Apache-2.0
 *
 * Links and RUNS against libedgefirst_image + libedgefirst_tensor.
 * A header that compiles is not the same as a library that loads. */
#include <edgefirst/image.h>
#include <stdio.h>

int main(void) {
    struct ef_image_processor *p = ef_image_processor_new();
    if (!p) { fprintf(stderr, "FAIL: ef_image_processor_new returned NULL\n"); return 1; }
    printf("PASS: image links and runs\n");
    ef_image_processor_free(p);
    return 0;
}
