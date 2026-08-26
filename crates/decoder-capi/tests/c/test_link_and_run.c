/* SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
 * SPDX-License-Identifier: Apache-2.0
 *
 * Links and RUNS against libedgefirst_decoder + libedgefirst_tensor.
 * A header that compiles is not the same as a library that loads. */
#include <edgefirst/decoder.h>
#include <stdio.h>

int main(void) {
    struct ef_decoder_params *p = ef_decoder_params_new();
    if (!p) { fprintf(stderr, "FAIL: ef_decoder_params_new returned NULL\n"); return 1; }
    printf("PASS: decoder links and runs\n");
    ef_decoder_params_free(p);
    return 0;
}
