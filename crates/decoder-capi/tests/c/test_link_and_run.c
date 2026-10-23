/* SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
 * SPDX-License-Identifier: Apache-2.0
 *
 * Links and RUNS against libedgefirst_decoder + libedgefirst_tensor.
 * A header that compiles is not the same as a library that loads.
 *
 * scripts/check-headers.sh compiles and runs exactly this file per modular
 * library, so it is the only lane that proves the exported symbols resolve
 * at runtime from a real C consumer -- the Rust `#[cfg(test)]` tests in
 * src/infer.rs call the same functions, but statically, within the crate.
 *
 * Run from the repository root: the model-config case reads
 * testdata/hf/segformer-512. */
#define _POSIX_C_SOURCE 200809L
#define _DARWIN_C_SOURCE
#include <edgefirst/decoder.h>
#include <errno.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

/* Ultralytics schema inference, end to end: accumulate the signals an
 * inference runtime would report, infer a schema, and build a real decoder
 * from the JSON it produces. Two classes and a 6-wide feature dim rather
 * than COCO's 80/84, so the `names` dict stays readable while still being
 * internally consistent -- a mismatch is a hard error, not a warning. */
static int infer_round_trip(void) {
    ef_infer_signals *s = ef_infer_signals_new(0); /* 0 = onnx */
    if (!s) { fprintf(stderr, "FAIL: ef_infer_signals_new returned NULL\n"); return 1; }

    const uintptr_t in_shape[4] = { 1, 3, 640, 640 };
    if (ef_infer_signals_add_input(s, "images", in_shape, 4, EF_INFER_DTYPE_FLOAT32) != 0) {
        fprintf(stderr, "FAIL: add_input\n"); ef_infer_signals_free(s); return 1;
    }
    const uintptr_t out_shape[3] = { 1, 6, 8400 }; /* 4 box + 2 classes */
    if (ef_infer_signals_add_output(s, "output0", out_shape, 3, EF_INFER_DTYPE_FLOAT32,
                                    NULL, NULL, 0) != 0) {
        fprintf(stderr, "FAIL: add_output\n"); ef_infer_signals_free(s); return 1;
    }
    ef_infer_signals_add_metadata(s, "names", "{0: 'person', 1: 'bicycle'}");
    ef_infer_signals_add_metadata(s, "task", "detect");
    ef_infer_signals_add_metadata(s, "end2end", "False");

    char *err = NULL;
    ef_inferred_schema *inferred = ef_infer_ultralytics_schema(s, &err);
    ef_infer_signals_free(s);
    if (!inferred) {
        fprintf(stderr, "FAIL: inference: %s\n", err ? err : "(no message)");
        ef_decoder_string_free(err);
        return 1;
    }
    if (err) { /* err_out must stay untouched on success */
        fprintf(stderr, "FAIL: err_out written on a successful inference\n");
        ef_decoder_string_free(err);
        ef_inferred_schema_free(inferred);
        return 1;
    }

    char *schema_json = ef_inferred_schema_json(inferred);
    char *labels_json = ef_inferred_schema_labels_json(inferred);
    char *description = ef_inferred_schema_description(inferred);
    int rc = 0;
    if (!schema_json || !labels_json || !description) {
        fprintf(stderr, "FAIL: a JSON view returned NULL\n"); rc = 1;
    } else if (!strstr(labels_json, "person") || !strstr(labels_json, "bicycle")) {
        fprintf(stderr, "FAIL: labels lost the class names: %s\n", labels_json); rc = 1;
    } else if (!strstr(schema_json, "\"decoder_version\":\"yolov8\"")) {
        fprintf(stderr, "FAIL: schema did not pin decoder_version\n"); rc = 1;
    }

    if (rc == 0) {
        /* The point of the round trip: the inferred JSON must be something
         * this same library will accept back as a decoder configuration. */
        ef_decoder_params *p = ef_decoder_params_new();
        ef_decoder_params_set_config_json(p, schema_json, 0); /* 0 = NUL-terminated */
        ef_decoder_params_set_score_threshold(p, 0.25f);
        ef_decoder *d = ef_decoder_new(p);
        ef_decoder_params_free(p);
        if (!d) {
            fprintf(stderr, "FAIL: inferred schema did not build a decoder\n"); rc = 1;
        } else {
            ef_decoder_free(d);
        }
    }

    ef_decoder_string_free(schema_json);
    ef_decoder_string_free(labels_json);
    ef_decoder_string_free(description);
    ef_inferred_schema_free(inferred);
    return rc;
}

/* A failed inference must report through err_out rather than crashing or
 * returning a handle -- and the caller must be able to free that message
 * with the same string free as any other returned string. */
static int infer_reports_failure(void) {
    ef_infer_signals *s = ef_infer_signals_new(0);
    if (!s) { fprintf(stderr, "FAIL: ef_infer_signals_new returned NULL\n"); return 1; }
    char *err = NULL;
    ef_inferred_schema *inferred = ef_infer_ultralytics_schema(s, &err); /* no metadata */
    ef_infer_signals_free(s);
    if (inferred) {
        fprintf(stderr, "FAIL: empty metadata produced a schema\n");
        ef_inferred_schema_free(inferred);
        ef_decoder_string_free(err);
        return 1;
    }
    if (!err || err[0] == '\0') {
        fprintf(stderr, "FAIL: a failed inference left no message\n");
        ef_decoder_string_free(err);
        return 1;
    }
    ef_decoder_string_free(err);
    return 0;
}

/* Copy `name` from the SegFormer fixture into `dir`. */
static int copy_fixture(const char *dir, const char *name) {
    char src[512], dst[512];
    snprintf(src, sizeof src, "testdata/hf/segformer-512/%s", name);
    snprintf(dst, sizeof dst, "%s/%s", dir, name);
    FILE *in = fopen(src, "rb");
    if (!in) { fprintf(stderr, "FAIL: cannot open %s (run from the repo root)\n", src); return 1; }
    FILE *out = fopen(dst, "wb");
    if (!out) { fclose(in); fprintf(stderr, "FAIL: cannot create %s\n", dst); return 1; }
    char buf[4096];
    size_t n;
    while ((n = fread(buf, 1, sizeof buf, in)) > 0) fwrite(buf, 1, n, out);
    fclose(in);
    return fclose(out) != 0;
}

/* Model configuration discovery: a model with Hugging Face configs beside it
 * resolves to a semantic schema and a preprocessing spec the image library
 * takes as-is (ef_convert_options_from_spec). */
static int model_config_round_trip(void) {
    const char *tmp = getenv("TMPDIR");
    char dir[512];
    snprintf(dir, sizeof dir, "%s/ef_model_config_XXXXXX", tmp && *tmp ? tmp : "/tmp");
    if (!mkdtemp(dir)) { fprintf(stderr, "FAIL: mkdtemp\n"); return 1; }
    char model[600];
    snprintf(model, sizeof model, "%s/model.bin", dir);
    int rc = copy_fixture(dir, "config.json") || copy_fixture(dir, "preprocessor_config.json");
    FILE *f = fopen(model, "wb");
    if (f) { fputs("model bytes", f); fclose(f); } else { rc = 1; }

    ef_infer_signals *s = ef_infer_signals_new(EF_MODEL_SOURCE_ONNX);
    const uintptr_t in_shape[4] = { 1, 3, 512, 512 };
    const uintptr_t out_shape[4] = { 1, 150, 128, 128 };
    ef_infer_signals_add_input(s, "pixel_values", in_shape, 4, EF_DTYPE_F32);
    ef_infer_signals_add_output(s, "logits", out_shape, 4, EF_DTYPE_F32, NULL, NULL, 0);

    ef_model_overrides *o = ef_model_overrides_new();
    char *err = NULL;
    if (ef_model_overrides_set(o, "resize", "fill", &err) != EINVAL || !err) {
        fprintf(stderr, "FAIL: a bad override was accepted\n"); rc = 1;
    }
    ef_decoder_string_free(err);
    err = NULL;

    ef_model_config *c = rc ? NULL : ef_model_config_resolve(model, s, o, &err);
    ef_model_overrides_free(o);
    ef_infer_signals_free(s);
    if (!rc && !c) {
        fprintf(stderr, "FAIL: resolve: %s\n", err ? err : "(no message)"); rc = 1;
    }
    ef_decoder_string_free(err);

    if (c) {
        uint32_t task = 0, family = 0, source = 0;
        char *detail = NULL;
        ef_preprocess_spec spec;
        memset(&spec, 0, sizeof spec);
        if (ef_model_config_task(c, &task) != 0 || task != EF_MODEL_TASK_SEMANTIC) {
            fprintf(stderr, "FAIL: task %u\n", (unsigned) task); rc = 1;
        } else if (ef_model_config_family(c, &family) != 0 || family != EF_MODEL_FAMILY_HUGGING_FACE) {
            fprintf(stderr, "FAIL: family %u\n", (unsigned) family); rc = 1;
        } else if (ef_model_config_source(c, EF_CONFIG_FIELD_NORMALIZATION, &source, &detail) != 0
                   || source != EF_CONFIG_SOURCE_SIDECAR_HF) {
            fprintf(stderr, "FAIL: normalization source %u\n", (unsigned) source); rc = 1;
        } else if (ef_model_config_preprocessing(c, &spec) != 0 || spec.width != 512
                   || spec.fit != EF_FIT_STRETCH || spec.interpolation != EF_INTERPOLATION_BILINEAR
                   || spec.mean[0] < 123.0f || spec.mean[0] > 124.0f) {
            fprintf(stderr, "FAIL: preprocess spec\n"); rc = 1;
        }
        ef_decoder_string_free(detail);

        char *describe = ef_model_config_describe(c);
        if (!describe || !strstr(describe, "normalize=")) {
            fprintf(stderr, "FAIL: describe: %s\n", describe ? describe : "(null)"); rc = 1;
        }
        ef_decoder_string_free(describe);

        /* The resolved schema builds a decoder for the semantic output. */
        char *schema = ef_model_config_schema_json(c);
        ef_decoder_params *p = ef_decoder_params_new();
        ef_decoder_params_set_config_json(p, schema, 0);
        ef_decoder *d = ef_decoder_new(p);
        ef_decoder_params_free(p);
        ef_decoder_string_free(schema);
        uintptr_t classes = 0;
        if (!d || ef_decoder_semantic_classes(d, &classes) != 0 || classes != 150) {
            fprintf(stderr, "FAIL: schema did not build a 150-class semantic decoder\n"); rc = 1;
        }
        ef_decoder_free(d);
        ef_model_config_free(c);
    }

    char path[600];
    snprintf(path, sizeof path, "%s/config.json", dir); remove(path);
    snprintf(path, sizeof path, "%s/preprocessor_config.json", dir); remove(path);
    remove(model);
    rmdir(dir);
    return rc;
}

int main(void) {
    if (ef_decoder_abi_version() != 4) {
        fprintf(stderr, "FAIL: ef_decoder_abi_version() = %u, expected 4\n",
                (unsigned) ef_decoder_abi_version());
        return 1;
    }

    struct ef_decoder_params *p = ef_decoder_params_new();
    if (!p) { fprintf(stderr, "FAIL: ef_decoder_params_new returned NULL\n"); return 1; }
    ef_decoder_params_free(p);

    if (infer_round_trip() != 0) return 1;
    if (infer_reports_failure() != 0) return 1;
    if (model_config_round_trip() != 0) return 1;

    printf("PASS: decoder links and runs\n");
    return 0;
}
