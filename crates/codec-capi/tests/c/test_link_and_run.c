/* SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
 * SPDX-License-Identifier: Apache-2.0
 *
 * Links and RUNS against libedgefirst_codec + libedgefirst_tensor.
 * A header that compiles is not the same as a library that loads. */
#include <edgefirst/codec.h>
#include <stdio.h>

int main(void) {
    struct ef_image_decoder *d = ef_image_decoder_new();
    if (!d) { fprintf(stderr, "FAIL: ef_image_decoder_new returned NULL\n"); return 1; }
    printf("PASS: codec links and runs\n");
    ef_image_decoder_free(d);
    return 0;
}
