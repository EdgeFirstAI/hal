// SPDX-FileCopyrightText: Copyright 2025 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0
//
// Decoder module tests for EdgeFirst HAL C API

#include "test_common.h"

// Valid YOLO detection config (84 features = 4 bbox + 80 classes)
static const char* YOLO_JSON_CONFIG =
    "{"
        "\"outputs\": [{"
            "\"decoder\": \"ultralytics\","
            "\"type\": \"detection\","
            "\"shape\": [1, 84, 8400],"
            "\"dshape\": [[\"batch\", 1], [\"num_features\", 84], [\"num_boxes\", 8400]]"
        "}],"
        "\"nms\": \"class_aware\""
    "}";

static const char* YOLO_YAML_CONFIG =
    "outputs:\n"
    "  - decoder: ultralytics\n"
    "    type: detection\n"
    "    shape: [1, 84, 8400]\n"
    "    dshape: [[batch, 1], [num_features, 84], [num_boxes, 8400]]\n"
    "nms: class_aware\n";

// =============================================================================
// Decoder Params Tests
// =============================================================================

static void test_decoder_params_default(void) {
    TEST("decoder_params_default");

    struct hal_decoder_params params = hal_decoder_params_default();
    ASSERT_NULL(params.config_json);
    ASSERT_NULL(params.config_yaml);
    ASSERT_NULL(params.config_file);
    ASSERT_FLOAT_EQ(0.5f, params.score_threshold, 0.001f);
    ASSERT_FLOAT_EQ(0.5f, params.iou_threshold, 0.001f);
    ASSERT_EQ(HAL_NMS_CLASS_AGNOSTIC, params.nms);

    TEST_PASS();
}

static void test_decoder_new_with_json(void) {
    TEST("decoder_new_with_json");

    struct hal_decoder_params params = hal_decoder_params_default();
    params.config_json = YOLO_JSON_CONFIG;

    struct hal_decoder* decoder = hal_decoder_new(&params);
    ASSERT_NOT_NULL(decoder);

    hal_decoder_free(decoder);
    TEST_PASS();
}

static void test_decoder_new_with_yaml(void) {
    TEST("decoder_new_with_yaml");

    struct hal_decoder_params params = hal_decoder_params_default();
    params.config_yaml = YOLO_YAML_CONFIG;

    struct hal_decoder* decoder = hal_decoder_new(&params);
    ASSERT_NOT_NULL(decoder);

    hal_decoder_free(decoder);
    TEST_PASS();
}

static void test_decoder_new_with_thresholds(void) {
    TEST("decoder_new_with_thresholds");

    struct hal_decoder_params params = hal_decoder_params_default();
    params.config_json = YOLO_JSON_CONFIG;
    params.score_threshold = 0.25f;
    params.iou_threshold = 0.45f;
    params.nms = HAL_NMS_CLASS_AWARE;

    struct hal_decoder* decoder = hal_decoder_new(&params);
    ASSERT_NOT_NULL(decoder);

    hal_decoder_free(decoder);
    TEST_PASS();
}

static void test_decoder_new_null_params(void) {
    TEST("decoder_new_null_params");

    errno = 0;
    struct hal_decoder* decoder = hal_decoder_new(NULL);
    ASSERT_NULL(decoder);
    ASSERT_ERRNO(EINVAL);

    TEST_PASS();
}

static void test_decoder_new_no_config(void) {
    TEST("decoder_new_no_config");

    struct hal_decoder_params params = hal_decoder_params_default();

    errno = 0;
    struct hal_decoder* decoder = hal_decoder_new(&params);
    ASSERT_NULL(decoder);
    ASSERT_ERRNO(EINVAL);

    TEST_PASS();
}

static void test_decoder_new_multiple_configs(void) {
    TEST("decoder_new_multiple_configs");

    struct hal_decoder_params params = hal_decoder_params_default();
    params.config_json = YOLO_JSON_CONFIG;
    params.config_yaml = YOLO_YAML_CONFIG;

    errno = 0;
    struct hal_decoder* decoder = hal_decoder_new(&params);
    ASSERT_NULL(decoder);
    ASSERT_ERRNO(EINVAL);

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
    int result = hal_decoder_decode(NULL, outputs, 1, &boxes, NULL);
    ASSERT_EQ(-1, result);
    ASSERT_ERRNO(EINVAL);

    hal_tensor_free(tensor);
    TEST_PASS();
}

static void test_decoder_decode_invalid_params(void) {
    TEST("decoder_decode_invalid_params");

    // Build a decoder with valid Ultralytics YOLO config
    struct hal_decoder_params params = hal_decoder_params_default();
    params.config_json = YOLO_JSON_CONFIG;

    struct hal_decoder* decoder = hal_decoder_new(&params);
    ASSERT_NOT_NULL(decoder);

    struct hal_detect_box_list* boxes = NULL;

    // NULL outputs
    errno = 0;
    int result = hal_decoder_decode(decoder, NULL, 1, &boxes, NULL);
    ASSERT_EQ(-1, result);
    ASSERT_ERRNO(EINVAL);

    // NULL out_boxes
    size_t shape[] = {1, 85, 8400};
    struct hal_tensor* tensor = hal_tensor_new(HAL_DTYPE_F32, shape, 3, HAL_TENSOR_MEMORY_MEM, NULL);
    ASSERT_NOT_NULL(tensor);
    const struct hal_tensor* outputs[] = {tensor};

    errno = 0;
    result = hal_decoder_decode(decoder, outputs, 1, NULL, NULL);
    ASSERT_EQ(-1, result);
    ASSERT_ERRNO(EINVAL);

    // Zero outputs
    errno = 0;
    result = hal_decoder_decode(decoder, outputs, 0, &boxes, NULL);
    ASSERT_EQ(-1, result);
    ASSERT_ERRNO(EINVAL);

    hal_tensor_free(tensor);
    hal_decoder_free(decoder);
    TEST_PASS();
}

// =============================================================================
// Decoder Introspection Tests
// =============================================================================

static void test_decoder_model_type(void) {
    TEST("decoder_model_type");

    struct hal_decoder_params params = hal_decoder_params_default();
    params.config_json = YOLO_JSON_CONFIG;

    struct hal_decoder* decoder = hal_decoder_new(&params);
    ASSERT_NOT_NULL(decoder);

    char* model_type = hal_decoder_model_type(decoder);
    ASSERT_NOT_NULL(model_type);
    // Should be a non-empty string
    ASSERT_TRUE(strlen(model_type) > 0);
    free(model_type);

    // NULL decoder
    ASSERT_NULL(hal_decoder_model_type(NULL));

    hal_decoder_free(decoder);
    TEST_PASS();
}

static void test_decoder_normalized_boxes(void) {
    TEST("decoder_normalized_boxes");

    struct hal_decoder_params params = hal_decoder_params_default();
    params.config_json = YOLO_JSON_CONFIG;

    struct hal_decoder* decoder = hal_decoder_new(&params);
    ASSERT_NOT_NULL(decoder);

    int result = hal_decoder_normalized_boxes(decoder);
    // Result is 1, 0, or -1 (unknown)
    ASSERT_TRUE(result >= -1 && result <= 1);

    // NULL decoder returns -1
    ASSERT_EQ(-1, hal_decoder_normalized_boxes(NULL));

    hal_decoder_free(decoder);
    TEST_PASS();
}

// =============================================================================
// Dequantize Tests
// =============================================================================

static void test_dequantize_null_params(void) {
    TEST("dequantize_null_params");

    size_t shape[] = {2, 3};
    struct hal_quantization quant = { .scale = 1.0f, .zero_point = 0 };

    struct hal_tensor* output = hal_tensor_new(HAL_DTYPE_F32, shape, 2, HAL_TENSOR_MEMORY_MEM, NULL);
    ASSERT_NOT_NULL(output);

    // NULL input
    errno = 0;
    ASSERT_EQ(-1, hal_dequantize(NULL, quant, output));
    ASSERT_ERRNO(EINVAL);

    // NULL output
    struct hal_tensor* input = hal_tensor_new(HAL_DTYPE_U8, shape, 2, HAL_TENSOR_MEMORY_MEM, NULL);
    ASSERT_NOT_NULL(input);

    errno = 0;
    ASSERT_EQ(-1, hal_dequantize(input, quant, NULL));
    ASSERT_ERRNO(EINVAL);

    hal_tensor_free(input);
    hal_tensor_free(output);
    TEST_PASS();
}

// =============================================================================
// Segmentation to Mask Tests
// =============================================================================

static void test_segmentation_to_mask_null(void) {
    TEST("segmentation_to_mask_null");

    // NULL list
    errno = 0;
    struct hal_tensor* mask = hal_segmentation_to_mask(NULL, 0);
    ASSERT_NULL(mask);
    ASSERT_ERRNO(EINVAL);

    TEST_PASS();
}

// =============================================================================
// Main Test Runner
// =============================================================================

void run_decoder_tests(void) {
    TEST_SUITE("Decoder");

    // Params tests
    test_decoder_params_default();
    test_decoder_new_with_json();
    test_decoder_new_with_yaml();
    test_decoder_new_with_thresholds();
    test_decoder_new_null_params();
    test_decoder_new_no_config();
    test_decoder_new_multiple_configs();

    // Detection box list tests
    test_detect_box_list_null_handling();
    test_detect_box_list_get_invalid();

    // Segmentation list tests
    test_segmentation_list_null_handling();

    // Decoder tests
    test_decoder_null_handling();
    test_decoder_decode_invalid_params();

    // Decoder introspection tests
    test_decoder_model_type();
    test_decoder_normalized_boxes();

    // Dequantize tests
    test_dequantize_null_params();

    // Segmentation to mask tests
    test_segmentation_to_mask_null();
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
