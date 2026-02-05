// SPDX-FileCopyrightText: Copyright 2025 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0
//
// Decoder module tests for EdgeFirst HAL C API

#include "test_common.h"

// =============================================================================
// Decoder Builder Tests
// =============================================================================

static void test_decoder_builder_new(void) {
    TEST("decoder_builder_new");

    struct hal_decoder_builder* builder = hal_decoder_builder_new();
    ASSERT_NOT_NULL(builder);

    hal_decoder_builder_free(builder);
    TEST_PASS();
}

static void test_decoder_builder_null_handling(void) {
    TEST("decoder_builder_null_handling");

    // Free NULL should be no-op
    hal_decoder_builder_free(NULL);

    TEST_PASS();
}

static void test_decoder_builder_with_thresholds(void) {
    TEST("decoder_builder_with_thresholds");

    struct hal_decoder_builder* builder = hal_decoder_builder_new();
    ASSERT_NOT_NULL(builder);

    // Set score threshold
    int result = hal_decoder_builder_with_score_threshold(builder, 0.3f);
    ASSERT_EQ(0, result);

    // Set IOU threshold
    result = hal_decoder_builder_with_iou_threshold(builder, 0.45f);
    ASSERT_EQ(0, result);

    // Set NMS mode
    result = hal_decoder_builder_with_nms(builder, HAL_NMS_CLASS_AWARE);
    ASSERT_EQ(0, result);

    hal_decoder_builder_free(builder);
    TEST_PASS();
}

static void test_decoder_builder_with_thresholds_null(void) {
    TEST("decoder_builder_with_thresholds_null");

    errno = 0;
    int result = hal_decoder_builder_with_score_threshold(NULL, 0.5f);
    ASSERT_EQ(-1, result);
    ASSERT_ERRNO(EINVAL);

    errno = 0;
    result = hal_decoder_builder_with_iou_threshold(NULL, 0.5f);
    ASSERT_EQ(-1, result);
    ASSERT_ERRNO(EINVAL);

    errno = 0;
    result = hal_decoder_builder_with_nms(NULL, HAL_NMS_CLASS_AGNOSTIC);
    ASSERT_EQ(-1, result);
    ASSERT_ERRNO(EINVAL);

    TEST_PASS();
}

static void test_decoder_builder_with_config_json(void) {
    TEST("decoder_builder_with_config_json");

    struct hal_decoder_builder* builder = hal_decoder_builder_new();
    ASSERT_NOT_NULL(builder);

    // Valid YOLO detection config
    const char* json = "{"
        "\"outputs\": [{"
            "\"decoder\": \"ultralytics\","
            "\"type\": \"detection\","
            "\"shape\": [1, 84, 8400],"
            "\"dshape\": [[\"batch\", 1], [\"num_features\", 84], [\"num_boxes\", 8400]]"
        "}],"
        "\"nms\": \"class_aware\""
    "}";

    int result = hal_decoder_builder_with_config_json(builder, json);
    ASSERT_EQ(0, result);

    hal_decoder_builder_free(builder);
    TEST_PASS();
}

static void test_decoder_builder_with_config_yaml(void) {
    TEST("decoder_builder_with_config_yaml");

    struct hal_decoder_builder* builder = hal_decoder_builder_new();
    ASSERT_NOT_NULL(builder);

    // Valid YOLO detection config in YAML
    const char* yaml =
        "outputs:\n"
        "  - decoder: ultralytics\n"
        "    type: detection\n"
        "    shape: [1, 84, 8400]\n"
        "    dshape: [[batch, 1], [num_features, 84], [num_boxes, 8400]]\n"
        "nms: class_aware\n";

    int result = hal_decoder_builder_with_config_yaml(builder, yaml);
    ASSERT_EQ(0, result);

    hal_decoder_builder_free(builder);
    TEST_PASS();
}

static void test_decoder_builder_with_config_null(void) {
    TEST("decoder_builder_with_config_null");

    struct hal_decoder_builder* builder = hal_decoder_builder_new();
    ASSERT_NOT_NULL(builder);

    // NULL JSON string
    errno = 0;
    int result = hal_decoder_builder_with_config_json(builder, NULL);
    ASSERT_EQ(-1, result);
    ASSERT_ERRNO(EINVAL);

    // NULL YAML string
    errno = 0;
    result = hal_decoder_builder_with_config_yaml(builder, NULL);
    ASSERT_EQ(-1, result);
    ASSERT_ERRNO(EINVAL);

    // NULL builder
    errno = 0;
    result = hal_decoder_builder_with_config_json(NULL, "{}");
    ASSERT_EQ(-1, result);
    ASSERT_ERRNO(EINVAL);

    hal_decoder_builder_free(builder);
    TEST_PASS();
}

static void test_decoder_builder_build_without_config(void) {
    TEST("decoder_builder_build_without_config");

    struct hal_decoder_builder* builder = hal_decoder_builder_new();
    ASSERT_NOT_NULL(builder);

    // Building without config should fail
    errno = 0;
    struct hal_decoder* decoder = hal_decoder_builder_build(builder);
    ASSERT_NULL(decoder);
    // errno should be EBADMSG

    // Builder is consumed even on failure, no need to free

    TEST_PASS();
}

static void test_decoder_builder_build_with_config(void) {
    TEST("decoder_builder_build_with_config");

    struct hal_decoder_builder* builder = hal_decoder_builder_new();
    ASSERT_NOT_NULL(builder);

    // Valid YOLO detection config
    const char* json = "{"
        "\"outputs\": [{"
            "\"decoder\": \"ultralytics\","
            "\"type\": \"detection\","
            "\"shape\": [1, 84, 8400],"
            "\"dshape\": [[\"batch\", 1], [\"num_features\", 84], [\"num_boxes\", 8400]]"
        "}],"
        "\"nms\": \"class_aware\""
    "}";

    int result = hal_decoder_builder_with_config_json(builder, json);
    ASSERT_EQ(0, result);

    result = hal_decoder_builder_with_score_threshold(builder, 0.25f);
    ASSERT_EQ(0, result);

    struct hal_decoder* decoder = hal_decoder_builder_build(builder);
    ASSERT_NOT_NULL(decoder);

    hal_decoder_free(decoder);

    // Builder is consumed by build, no need to free

    TEST_PASS();
}

// =============================================================================
// Detection Box List Tests
// =============================================================================

static void test_detect_box_list_null_handling(void) {
    TEST("detect_box_list_null_handling");

    ASSERT_EQ(0, hal_detect_box_list_len(NULL));

    struct hal_detect_box box;
    errno = 0;
    int result = hal_detect_box_list_get(NULL, 0, &box);
    ASSERT_EQ(-1, result);
    ASSERT_ERRNO(EINVAL);

    // Free NULL should be no-op
    hal_detect_box_list_free(NULL);

    TEST_PASS();
}

static void test_detect_box_list_get_invalid(void) {
    TEST("detect_box_list_get_invalid");

    // We can't easily create a real detect_box_list without running decoder
    // Test NULL out_box parameter
    // (Would need a real list to test index out of bounds)

    TEST_PASS();
}

// =============================================================================
// Segmentation List Tests
// =============================================================================

static void test_segmentation_list_null_handling(void) {
    TEST("segmentation_list_null_handling");

    ASSERT_EQ(0, hal_segmentation_list_len(NULL));

    float xmin, ymin, xmax, ymax;
    errno = 0;
    int result = hal_segmentation_list_get_bbox(NULL, 0, &xmin, &ymin, &xmax, &ymax);
    ASSERT_EQ(-1, result);
    ASSERT_ERRNO(EINVAL);

    errno = 0;
    size_t height, width;
    const uint8_t* mask = hal_segmentation_list_get_mask(NULL, 0, &height, &width);
    ASSERT_NULL(mask);
    ASSERT_ERRNO(EINVAL);

    // Free NULL should be no-op
    hal_segmentation_list_free(NULL);

    TEST_PASS();
}

// =============================================================================
// Decoder Tests
// =============================================================================

static void test_decoder_null_handling(void) {
    TEST("decoder_null_handling");

    // Free NULL should be no-op
    hal_decoder_free(NULL);

    // Decode with NULL decoder
    size_t shape[] = {1, 85, 8400};
    struct hal_tensor* tensor = hal_tensor_new(HAL_DTYPE_F32, shape, 3, HAL_TENSOR_MEMORY_MEM, NULL);
    ASSERT_NOT_NULL(tensor);

    const struct hal_tensor* outputs[] = {tensor};
    struct hal_detect_box_list* boxes = NULL;

    errno = 0;
    int result = hal_decoder_decode_float(NULL, outputs, 1, &boxes);
    ASSERT_EQ(-1, result);
    ASSERT_ERRNO(EINVAL);

    hal_tensor_free(tensor);
    TEST_PASS();
}

static void test_decoder_decode_invalid_params(void) {
    TEST("decoder_decode_invalid_params");

    // Build a decoder with valid Ultralytics YOLO config
    struct hal_decoder_builder* builder = hal_decoder_builder_new();
    ASSERT_NOT_NULL(builder);

    // Valid YOLO detection config (84 features = 4 bbox + 80 classes)
    const char* json = "{"
        "\"outputs\": [{"
            "\"decoder\": \"ultralytics\","
            "\"type\": \"detection\","
            "\"shape\": [1, 84, 8400],"
            "\"dshape\": [[\"batch\", 1], [\"num_features\", 84], [\"num_boxes\", 8400]]"
        "}],"
        "\"nms\": \"class_aware\""
    "}";

    int result = hal_decoder_builder_with_config_json(builder, json);
    ASSERT_EQ(0, result);

    struct hal_decoder* decoder = hal_decoder_builder_build(builder);
    ASSERT_NOT_NULL(decoder);

    struct hal_detect_box_list* boxes = NULL;

    // NULL outputs
    errno = 0;
    result = hal_decoder_decode_float(decoder, NULL, 1, &boxes);
    ASSERT_EQ(-1, result);
    ASSERT_ERRNO(EINVAL);

    // NULL out_boxes
    size_t shape[] = {1, 85, 8400};
    struct hal_tensor* tensor = hal_tensor_new(HAL_DTYPE_F32, shape, 3, HAL_TENSOR_MEMORY_MEM, NULL);
    ASSERT_NOT_NULL(tensor);
    const struct hal_tensor* outputs[] = {tensor};

    errno = 0;
    result = hal_decoder_decode_float(decoder, outputs, 1, NULL);
    ASSERT_EQ(-1, result);
    ASSERT_ERRNO(EINVAL);

    // Zero outputs
    errno = 0;
    result = hal_decoder_decode_float(decoder, outputs, 0, &boxes);
    ASSERT_EQ(-1, result);
    ASSERT_ERRNO(EINVAL);

    hal_tensor_free(tensor);
    hal_decoder_free(decoder);
    TEST_PASS();
}

// =============================================================================
// Main Test Runner
// =============================================================================

void run_decoder_tests(void) {
    TEST_SUITE("Decoder");

    // Builder tests
    test_decoder_builder_new();
    test_decoder_builder_null_handling();
    test_decoder_builder_with_thresholds();
    test_decoder_builder_with_thresholds_null();
    test_decoder_builder_with_config_json();
    test_decoder_builder_with_config_yaml();
    test_decoder_builder_with_config_null();
    test_decoder_builder_build_without_config();
    test_decoder_builder_build_with_config();

    // Detection box list tests
    test_detect_box_list_null_handling();
    test_detect_box_list_get_invalid();

    // Segmentation list tests
    test_segmentation_list_null_handling();

    // Decoder tests
    test_decoder_null_handling();
    test_decoder_decode_invalid_params();
}

#ifdef TEST_DECODER_STANDALONE
// Define test result tracking variables for standalone mode
int tests_run = 0;
int tests_passed = 0;
int tests_failed = 0;
const char* current_test_name = NULL;
const char* current_suite_name = NULL;

int main(void) {
    run_decoder_tests();
    print_test_summary();
    return get_test_exit_code();
}
#endif
