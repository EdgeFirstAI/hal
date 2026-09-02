/* SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
 * SPDX-License-Identifier: Apache-2.0
 *
 * Links and RUNS against libedgefirst_decoder + libedgefirst_tensor.
 * A header that compiles is not the same as a library that loads.
 *
 * scripts/check-headers.sh compiles and runs exactly this file per modular
 * library, so it is the only lane that proves the exported symbols resolve
 * at runtime from a real C consumer -- the Rust `#[cfg(test)]` tests in
 * src/infer.rs call the same functions, but statically, within the crate. */
#include <edgefirst/decoder.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

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

int main(void) {
    struct ef_decoder_params *p = ef_decoder_params_new();
    if (!p) { fprintf(stderr, "FAIL: ef_decoder_params_new returned NULL\n"); return 1; }
    ef_decoder_params_free(p);

    if (infer_round_trip() != 0) return 1;
    if (infer_reports_failure() != 0) return 1;

    printf("PASS: decoder links and runs\n");
    return 0;
}
