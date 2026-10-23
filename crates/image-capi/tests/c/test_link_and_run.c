/* SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
 * SPDX-License-Identifier: Apache-2.0
 *
 * Links and RUNS against libedgefirst_image + libedgefirst_tensor.
 * A header that compiles is not the same as a library that loads. */
#include <edgefirst/image.h>
#include <errno.h>
#include <stdio.h>
#include <string.h>

/* A preprocessing spec as ef_model_config_preprocessing (libedgefirst_decoder)
 * fills it: a centred letterbox, pad 114, ImageNet normalization. */
static ef_preprocess_spec letterbox_spec(void) {
    ef_preprocess_spec spec = ef_preprocess_spec_default();
    spec.width = 640;
    spec.height = 640;
    const float mean[3] = { 123.675f, 116.28f, 103.53f };
    const float std[3] = { 58.395f, 57.12f, 57.375f };
    memcpy(spec.mean, mean, sizeof mean);
    memcpy(spec.std, std, sizeof std);
    spec.interpolation = EF_INTERPOLATION_BICUBIC; /* runs as bilinear */
    return spec;
}

static int spec_to_content_region(ef_image_processor *p) {
    ef_preprocess_spec spec = letterbox_spec();
    ef_convert_options *o = ef_convert_options_from_spec(&spec);
    if (!o) {
        fprintf(stderr, "FAIL: ef_convert_options_from_spec\n");
        return 1;
    }
    uint32_t px[4] = { 0 };
    float norm[4] = { 0 };
    int rc = 0;
    if (ef_convert_options_set_interpolation(o, EF_INTERPOLATION_NEAREST) != 0
        || ef_convert_options_set_interpolation(o, 4) != EINVAL) {
        fprintf(stderr, "FAIL: ef_convert_options_set_interpolation\n"); rc = 1;
    }
    if (ef_convert_options_content_region(o, 1920, 1080, 640, 640, px, norm) != 0) {
        fprintf(stderr, "FAIL: ef_convert_options_content_region\n"); rc = 1;
    } else if (px[0] != 0 || px[1] != 140 || px[2] != 640 || px[3] != 360
               || norm[1] != 0.21875f || norm[3] != 0.78125f) {
        fprintf(stderr, "FAIL: content region %u,%u %ux%u\n", px[0], px[1], px[2], px[3]); rc = 1;
    }

    /* Every tile of a letterboxed plan carries the content region. */
    ef_tiling_config cfg = ef_tiling_config_default(640, 640);
    ef_tile_placement_list *plan = ef_image_processor_plan_tiles(p, 1500, 400, &cfg, o);
    if (!plan || ef_tile_placement_list_len(plan) < 3) {
        fprintf(stderr, "FAIL: plan_tiles with options\n"); rc = 1;
    } else {
        ef_tile_placement t;
        if (ef_tile_placement_list_get(plan, 1, &t) != 0 || !t.has_content_region) {
            fprintf(stderr, "FAIL: a letterboxed tile has no content region\n"); rc = 1;
        }
    }
    ef_tile_placement_list_free(plan);
    ef_convert_options_free(o);

    spec.fit = 7;
    errno = 0;
    if (ef_convert_options_from_spec(&spec) != NULL || errno != EINVAL) {
        fprintf(stderr, "FAIL: an unknown fit code was accepted\n"); rc = 1;
    }
    return rc;
}

int main(void) {
    if (ef_image_abi_version() != 4) {
        fprintf(stderr, "FAIL: ef_image_abi_version() = %u, expected 4\n",
                (unsigned) ef_image_abi_version());
        return 1;
    }

    struct ef_image_processor *p = ef_image_processor_new_with_backend(EF_COMPUTE_BACKEND_CPU);
    if (!p) { fprintf(stderr, "FAIL: ef_image_processor_new_with_backend returned NULL\n"); return 1; }
    int rc = spec_to_content_region(p);
    ef_image_processor_free(p);
    if (rc != 0) return 1;

    p = ef_image_processor_new();
    if (!p) { fprintf(stderr, "FAIL: ef_image_processor_new returned NULL\n"); return 1; }
    printf("PASS: image links and runs\n");
    ef_image_processor_free(p);
    return 0;
}
