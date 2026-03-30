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

static void test_decoder_params_new(void) {
    TEST("decoder_params_new");

    struct hal_decoder_params* params = hal_decoder_params_new();
    ASSERT_NOT_NULL(params);

    hal_decoder_params_free(params);
    TEST_PASS();
}

static void test_decoder_new_with_json(void) {
    TEST("decoder_new_with_json");

    struct hal_decoder_params* params = hal_decoder_params_new();
    ASSERT_NOT_NULL(params);

    int rc = hal_decoder_params_set_config_json(params, YOLO_JSON_CONFIG, 0);
    ASSERT_EQ(0, rc);

    struct hal_decoder* decoder = hal_decoder_new(params);
    ASSERT_NOT_NULL(decoder);

    hal_decoder_free(decoder);
    hal_decoder_params_free(params);
    TEST_PASS();
}

static void test_decoder_new_with_yaml(void) {
    TEST("decoder_new_with_yaml");

    struct hal_decoder_params* params = hal_decoder_params_new();
    ASSERT_NOT_NULL(params);

    int rc = hal_decoder_params_set_config_yaml(params, YOLO_YAML_CONFIG, 0);
    ASSERT_EQ(0, rc);

    struct hal_decoder* decoder = hal_decoder_new(params);
    ASSERT_NOT_NULL(decoder);

    hal_decoder_free(decoder);
    hal_decoder_params_free(params);
    TEST_PASS();
}

static void test_decoder_new_with_thresholds(void) {
    TEST("decoder_new_with_thresholds");

    struct hal_decoder_params* params = hal_decoder_params_new();
    ASSERT_NOT_NULL(params);

    hal_decoder_params_set_config_json(params, YOLO_JSON_CONFIG, 0);
    hal_decoder_params_set_score_threshold(params, 0.25f);
    hal_decoder_params_set_iou_threshold(params, 0.45f);
    hal_decoder_params_set_nms(params, HAL_NMS_CLASS_AWARE);

    struct hal_decoder* decoder = hal_decoder_new(params);
    ASSERT_NOT_NULL(decoder);

    hal_decoder_free(decoder);
    hal_decoder_params_free(params);
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

    struct hal_decoder_params* params = hal_decoder_params_new();
    ASSERT_NOT_NULL(params);

    errno = 0;
    struct hal_decoder* decoder = hal_decoder_new(params);
    ASSERT_NULL(decoder);
    ASSERT_ERRNO(EINVAL);

    hal_decoder_params_free(params);
    TEST_PASS();
}

static void test_decoder_params_null_handling(void) {
    TEST("decoder_params_null_handling");

    // All setter functions should return -1 with EINVAL on NULL params
    errno = 0;
    ASSERT_EQ(-1, hal_decoder_params_set_config_json(NULL, "{}", 0));
    ASSERT_ERRNO(EINVAL);

    errno = 0;
    ASSERT_EQ(-1, hal_decoder_params_set_config_yaml(NULL, "---", 0));
    ASSERT_ERRNO(EINVAL);

    errno = 0;
    ASSERT_EQ(-1, hal_decoder_params_set_config_file(NULL, "test.yaml"));
    ASSERT_ERRNO(EINVAL);

    errno = 0;
    ASSERT_EQ(-1, hal_decoder_params_set_score_threshold(NULL, 0.5f));
    ASSERT_ERRNO(EINVAL);

    errno = 0;
    ASSERT_EQ(-1, hal_decoder_params_set_iou_threshold(NULL, 0.5f));
    ASSERT_ERRNO(EINVAL);

    // Free NULL should be no-op
    hal_decoder_params_free(NULL);

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
    struct hal_decoder_params* params = hal_decoder_params_new();
    ASSERT_NOT_NULL(params);
    hal_decoder_params_set_config_json(params, YOLO_JSON_CONFIG, 0);

    struct hal_decoder* decoder = hal_decoder_new(params);
    ASSERT_NOT_NULL(decoder);
    hal_decoder_params_free(params);

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

    struct hal_decoder_params* params = hal_decoder_params_new();
    ASSERT_NOT_NULL(params);
    hal_decoder_params_set_config_json(params, YOLO_JSON_CONFIG, 0);

    struct hal_decoder* decoder = hal_decoder_new(params);
    ASSERT_NOT_NULL(decoder);
    hal_decoder_params_free(params);

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

    struct hal_decoder_params* params = hal_decoder_params_new();
    ASSERT_NOT_NULL(params);
    hal_decoder_params_set_config_json(params, YOLO_JSON_CONFIG, 0);

    struct hal_decoder* decoder = hal_decoder_new(params);
    ASSERT_NOT_NULL(decoder);
    hal_decoder_params_free(params);

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
// Tracked Decoder Tests
// =============================================================================

static void test_decoder_tracked_linear_motion(void) {
    TEST("decoder_tracked_linear_motion");

    // Load quantized test data: yolov8s_80_classes.bin as i8 [1, 8400, 84]
    size_t det_shape[] = {1, 8400, 84};

    struct hal_tensor* det_tensor = hal_tensor_new(HAL_DTYPE_I8, det_shape, 3, HAL_TENSOR_MEMORY_MEM, NULL);
    ASSERT_NOT_NULL(det_tensor);

    hal_tensor_map* det_map = hal_tensor_map_create(det_tensor);
    ASSERT_NOT_NULL(det_map);

    int8_t* det_data = hal_tensor_map_data(det_map);
    ASSERT_NOT_NULL(det_data);

    memset(det_data, -128, hal_tensor_len(det_tensor) * hal_tensor_dtype_size(det_tensor));
    det_data[0] = 49;
    det_data[1] = 6;
    det_data[2] = -38;
    det_data[3] = 109;
    det_data[4] = 14;

    det_data[84 + 0] = -64;
    det_data[84 + 1] = 52;
    det_data[84 + 2] = -69;
    det_data[84 + 3] = 15;
    det_data[84 + 4 + 75] = -42;

    hal_tensor_map_unmap(det_map);

    // Build decoder via YAML config
    static const char* CONFIG =
        "decoder_version: yolov8\n"
        "outputs:\n"
        "  - type: detection\n"
        "    decoder: ultralytics\n"
        "    quantization: [0.0040811873, -123]\n"
        "    shape: [1, 8400, 84]\n"
        "    dshape:\n"
        "      - [batch, 1]\n"
        "      - [num_boxes, 8400]\n"
        "      - [num_features, 84]\n"
        "    normalized: true\n"
        "nms: class_agnostic\n";

    struct hal_decoder_params* params = hal_decoder_params_new();
    ASSERT_NOT_NULL(params);
    ASSERT_EQ(0, hal_decoder_params_set_config_yaml(params, CONFIG, 0));
    hal_decoder_params_set_score_threshold(params, 0.25f);
    hal_decoder_params_set_iou_threshold(params, 0.1f);

    struct hal_decoder* decoder = hal_decoder_new(params);
    ASSERT_NOT_NULL(decoder);
    hal_decoder_params_free(params);

    // Create tracker with matching parameters
    struct hal_bytetrack* tracker = hal_bytetrack_new(
        0.1f,   // track_update
        0.3f,   // high_thresh
        0.2f,   // match_thresh
        30,     // frame_rate
        60      // track_buffer
    );
    ASSERT_NOT_NULL(tracker);

    // Frame 0: establish tracks
    const struct hal_tensor* outputs[] = {det_tensor};
    struct hal_detect_box_list* boxes = NULL;
    struct hal_segmentation_list* segs = NULL;
    struct hal_track_info_list* tracks = NULL;

    int rc = hal_decoder_decode_tracked(
        decoder, tracker, 0, outputs, 1, &boxes, &segs, &tracks
    );
    ASSERT_EQ(0, rc);
    ASSERT_NOT_NULL(boxes);
    ASSERT_EQ(2, hal_detect_box_list_len(boxes));

    // Verify initial box positions
    struct hal_detect_box box0, box1;
    hal_detect_box_list_get(boxes, 0, &box0);
    hal_detect_box_list_get(boxes, 1, &box1);

    ASSERT_FLOAT_EQ(0.5285137f, box0.xmin, 1e-3f);
    ASSERT_FLOAT_EQ(0.05305544f, box0.ymin, 1e-3f);
    ASSERT_FLOAT_EQ(0.87541467f, box0.xmax, 1e-3f);
    ASSERT_FLOAT_EQ(0.9998909f, box0.ymax, 1e-3f);

    ASSERT_FLOAT_EQ(0.130598f, box1.xmin, 1e-3f);
    ASSERT_FLOAT_EQ(0.43260583f, box1.ymin, 1e-3f);
    ASSERT_FLOAT_EQ(0.35098213f, box1.xmax, 1e-3f);
    ASSERT_FLOAT_EQ(0.9958097f, box1.ymax, 1e-3f);

    hal_detect_box_list_free(boxes);
    if (segs) hal_segmentation_list_free(segs);
    if (tracks) hal_track_info_list_free(tracks);

    // Frames 1-100: apply linear motion to X coordinates
    // offset per frame = round(i * 1e-3 / 0.0040811873) as i8 added to each x value
    float quant_scale = 0.0040811873f;

    for (int i = 1; i <= 100; i++) {
        // Clone the original tensor for each frame
        struct hal_tensor* frame = hal_tensor_new(HAL_DTYPE_I8, det_shape, 3, HAL_TENSOR_MEMORY_MEM, NULL);
        ASSERT_NOT_NULL(frame);    

        // Add linear X offset: saturating add to x row (row 0 of 84)
        struct hal_tensor_map* map = hal_tensor_map_create(frame);
        ASSERT_NOT_NULL(map);
        int8_t* data = (int8_t*)hal_tensor_map_data(map);
        ASSERT_NOT_NULL(data);

        det_map = hal_tensor_map_create(det_tensor);
        ASSERT_NOT_NULL(det_map);
        det_data = hal_tensor_map_data(det_map);
        ASSERT_NOT_NULL(det_data);
        memcpy(data, det_data, 84*8400); // copy original data
        hal_tensor_map_unmap(det_map);

        int8_t offset = (int8_t)roundf((float)i * 1e-3f / quant_scale);
        for (size_t j = 0; j < 8400; j++) {
            int16_t val = (int16_t)data[84*j] + (int16_t)offset;
            if (val > 127) val = 127;
            if (val < -128) val = -128;
            data[84*j] = (int8_t)val;
        }
        hal_tensor_map_unmap(map);

        const struct hal_tensor* frame_outputs[] = {frame};
        uint64_t ts = 100000000l * i / 3l;

        rc = hal_decoder_decode_tracked(
            decoder, tracker, ts, frame_outputs, 1, &boxes, &segs, &tracks
        );
        ASSERT_EQ(0, rc);
        ASSERT_EQ(2, hal_detect_box_list_len(boxes));

        hal_detect_box_list_free(boxes);
        if (segs) hal_segmentation_list_free(segs);
        if (tracks) hal_track_info_list_free(tracks);
        hal_tensor_free(frame);
    }

    // Verify active tracks reflect linear motion (total X offset ~0.1)
    struct hal_track_info_list* active = hal_bytetrack_get_active_tracks(tracker);
    ASSERT_NOT_NULL(active);
    ASSERT_EQ(2, hal_track_info_list_len(active));

    struct hal_track_info info0, info1;
    hal_track_info_list_get(active, 0, &info0);
    hal_track_info_list_get(active, 1, &info1);

    // Tracked locations should reflect 0.1 X offset
    ASSERT_FLOAT_EQ(0.5285137f + 0.1f, info0.location[0], 1e-3f);
    ASSERT_FLOAT_EQ(0.87541467f + 0.1f, info0.location[2], 1e-3f);
    ASSERT_FLOAT_EQ(0.130598f + 0.1f, info1.location[0], 1e-3f);
    ASSERT_FLOAT_EQ(0.35098213f + 0.1f, info1.location[2], 1e-3f);

    // Y coordinates should be unchanged
    ASSERT_FLOAT_EQ(0.05305544f, info0.location[1], 1e-3f);
    ASSERT_FLOAT_EQ(0.9998909f, info0.location[3], 1e-3f);
    ASSERT_FLOAT_EQ(0.43260583f, info1.location[1], 1e-3f);
    ASSERT_FLOAT_EQ(0.9958097f, info1.location[3], 1e-3f);

    hal_track_info_list_free(active);

    // Frame 101: no detections — tracker should predict forward using
    // learned linear velocity, resulting in ~0.001 additional X offset (total ~0.101)

    // Zero out all scores (rows 4..84) with i8::MIN (-128)
    det_map = hal_tensor_map_create(det_tensor);
    ASSERT_NOT_NULL(det_map);
    det_data = (int8_t*)hal_tensor_map_data(det_map);
    ASSERT_NOT_NULL(det_data);
    memset(det_data, -128, 84*8400); // zero all scores and features to prevent any detections
    hal_tensor_map_unmap(det_map);

    const struct hal_tensor* empty_outputs[] = {det_tensor};
    uint64_t ts_101 = 100000000l * 101l / 3l;

    rc = hal_decoder_decode_tracked(
        decoder, tracker, ts_101, empty_outputs, 1, &boxes, &segs, &tracks
    );
    ASSERT_EQ(0, rc);
    ASSERT_EQ(2, hal_detect_box_list_len(boxes));

    // Predicted boxes should have ~0.101 X offset (0.1 + one more step of 0.001)
    hal_detect_box_list_get(boxes, 0, &box0);
    hal_detect_box_list_get(boxes, 1, &box1);

    ASSERT_FLOAT_EQ(0.5285137f + 0.101f, box0.xmin, 1e-3f);
    ASSERT_FLOAT_EQ(0.87541467f + 0.101f, box0.xmax, 1e-3f);
    ASSERT_FLOAT_EQ(0.130598f + 0.101f, box1.xmin, 1e-3f);
    ASSERT_FLOAT_EQ(0.35098213f + 0.101f, box1.xmax, 1e-3f);

    hal_detect_box_list_free(boxes);
    if (segs) hal_segmentation_list_free(segs);
    if (tracks) hal_track_info_list_free(tracks);
    hal_tensor_free(det_tensor);
    hal_bytetrack_free(tracker);
    hal_decoder_free(decoder);
    TEST_PASS();
}

static void test_decoder_tracked_end_to_end_segdet_split_proto(void) {
    TEST("decoder_tracked_end_to_end_segdet_split_proto");

    float quant_scale = 2.0f/255.0f;
    size_t boxes_shape[] = {1, 10, 4};
    struct hal_tensor* boxes_tensor = hal_tensor_new(
        HAL_DTYPE_U8, boxes_shape, 3, HAL_TENSOR_MEMORY_MEM, NULL
    );

    struct hal_image_processor* processor = hal_image_processor_new();
    ASSERT_NOT_NULL(processor);


    ASSERT_NOT_NULL(boxes_tensor);
    {
        struct hal_tensor_map* map = hal_tensor_map_create(boxes_tensor);
        ASSERT_NOT_NULL(map);
        uint8_t* data = (uint8_t*)hal_tensor_map_data(map);
        ASSERT_NOT_NULL(data);
        memset(data, 0, 10 * 4);
        data[0] = (uint8_t)roundf(0.1234f / quant_scale);
        data[1] = (uint8_t)roundf(0.1234f / quant_scale);
        data[2] = (uint8_t)roundf(0.2345f / quant_scale);
        data[3] = (uint8_t)roundf(0.2345f / quant_scale);
        hal_tensor_map_unmap(map);
    }

    // Scores tensor [1, 10, 1] as u8
    size_t scores_shape[] = {1, 10, 1};
    struct hal_tensor* scores_tensor = hal_tensor_new(
        HAL_DTYPE_U8, scores_shape, 3, HAL_TENSOR_MEMORY_MEM, NULL
    );
    ASSERT_NOT_NULL(scores_tensor);
    {
        struct hal_tensor_map* map = hal_tensor_map_create(scores_tensor);
        ASSERT_NOT_NULL(map);
        uint8_t* data = (uint8_t*)hal_tensor_map_data(map);
        ASSERT_NOT_NULL(data);
        memset(data, 0, 10);
        data[0] = (uint8_t)roundf(0.9876f / quant_scale); // 126
        hal_tensor_map_unmap(map);
    }

    // Classes tensor [1, 10, 1] as u8
    size_t classes_shape[] = {1, 10, 1};
    struct hal_tensor* classes_tensor = hal_tensor_new(
        HAL_DTYPE_U8, classes_shape, 3, HAL_TENSOR_MEMORY_MEM, NULL
    );
    ASSERT_NOT_NULL(classes_tensor);
    {
        struct hal_tensor_map* map = hal_tensor_map_create(classes_tensor);
        ASSERT_NOT_NULL(map);
        uint8_t* data = (uint8_t*)hal_tensor_map_data(map);
        ASSERT_NOT_NULL(data);
        memset(data, 0, 10);
        data[0] = (uint8_t)fminf(roundf(2.0f / quant_scale), 255.0f); // 255
        hal_tensor_map_unmap(map);
    }

    // Mask coefficients tensor [1, 10, 32] as u8 — all zeros
    size_t mask_shape[] = {1, 10, 32};
    struct hal_tensor* mask_tensor = hal_tensor_new(
        HAL_DTYPE_U8, mask_shape, 3, HAL_TENSOR_MEMORY_MEM, NULL
    );
    ASSERT_NOT_NULL(mask_tensor);

    size_t protos_shape[] = {1, 160, 160, 32};
    struct hal_tensor* protos_tensor = hal_tensor_new(
        HAL_DTYPE_U8, protos_shape, 4, HAL_TENSOR_MEMORY_MEM, NULL
    );
    ASSERT_NOT_NULL(protos_tensor);

    // Build decoder via YAML config (matching the Rust test config)
    static const char* CONFIG =
        "decoder_version: yolo26\n"
        "outputs:\n"
        "  - type: boxes\n"
        "    decoder: ultralytics\n"
        "    quantization: [0.00784313725490196, 0]\n"
        "    shape: [1, 10, 4]\n"
        "    dshape:\n"
        "      - [batch, 1]\n"
        "      - [num_boxes, 10]\n"
        "      - [box_coords, 4]\n"
        "    normalized: true\n"
        "  - type: scores\n"
        "    decoder: ultralytics\n"
        "    quantization: [0.00784313725490196, 0]\n"
        "    shape: [1, 10, 1]\n"
        "    dshape:\n"
        "      - [batch, 1]\n"
        "      - [num_boxes, 10]\n"
        "      - [num_classes, 1]\n"
        "  - type: classes\n"
        "    decoder: ultralytics\n"
        "    quantization: [0.00784313725490196, 0]\n"
        "    shape: [1, 10, 1]\n"
        "    dshape:\n"
        "      - [batch, 1]\n"
        "      - [num_boxes, 10]\n"
        "      - [num_classes, 1]\n"
        "  - type: mask_coefficients\n"
        "    decoder: ultralytics\n"
        "    quantization: [0.00784313725490196, 0]\n"
        "    shape: [1, 10, 32]\n"
        "    dshape:\n"
        "      - [batch, 1]\n"
        "      - [num_boxes, 10]\n"
        "      - [num_protos, 32]\n"
        "  - type: protos\n"
        "    decoder: ultralytics\n"
        "    quantization: [0.0039215686274509803921568627451, 128]\n"
        "    shape: [1, 160, 160, 32]\n"
        "    dshape:\n"
        "      - [batch, 1]\n"
        "      - [height, 160]\n"
        "      - [width, 160]\n"
        "      - [num_protos, 32]\n";

    struct hal_decoder_params* params = hal_decoder_params_new();
    ASSERT_NOT_NULL(params);
    ASSERT_EQ(0, hal_decoder_params_set_config_yaml(params, CONFIG, 0));
    hal_decoder_params_set_score_threshold(params, 0.45f);
    hal_decoder_params_set_iou_threshold(params, 0.45f);

    struct hal_decoder* decoder = hal_decoder_new(params);
    ASSERT_NOT_NULL(decoder);
    hal_decoder_params_free(params);

    // Create tracker
    struct hal_bytetrack* tracker = hal_bytetrack_new(
        0.1f,   // track_update
        0.7f,   // high_thresh
        0.5f,   // match_thresh
        30,     // frame_rate
        30      // track_buffer
    );
    ASSERT_NOT_NULL(tracker);

    // Frame 0: decode tracked
    const struct hal_tensor* outputs[] = {
        boxes_tensor, scores_tensor, classes_tensor, mask_tensor, protos_tensor
    };
    struct hal_detect_box_list* box_list = NULL;
    struct hal_segmentation_list* seg_list = NULL;
    struct hal_track_info_list* track_list = NULL;

    struct hal_tensor* image = hal_tensor_new_image(400, 400, HAL_PIXEL_FORMAT_RGBA, HAL_DTYPE_U8, HAL_TENSOR_MEMORY_MEM);
    ASSERT_NOT_NULL(image);

    int rc = hal_image_processor_draw_masks_tracked(
        processor, decoder, tracker, 0, outputs, 5, image, NULL, 1.0f, NULL, HAL_COLOR_MODE_CLASS, &box_list, &track_list
    );
    
    ASSERT_EQ(0, rc);
    ASSERT_NOT_NULL(box_list);
    ASSERT_EQ(1, hal_detect_box_list_len(box_list));

    // Verify detection: expected bbox ~(0.125, 0.125, 0.235, 0.235), score ~0.988, label 2
    // Difference in expected is due to quantization error from u8
    struct hal_detect_box box0;
    hal_detect_box_list_get(box_list, 0, &box0);
    ASSERT_FLOAT_EQ(0.12549022f, box0.xmin, 1.0f / 160.0f);
    ASSERT_FLOAT_EQ(0.12549022f, box0.ymin, 1.0f / 160.0f);
    ASSERT_FLOAT_EQ(0.23529413f, box0.xmax, 1.0f / 160.0f);
    ASSERT_FLOAT_EQ(0.23529413f, box0.ymax, 1.0f / 160.0f);
    ASSERT_EQ(2, (int)box0.label);

    hal_detect_box_list_free(box_list);
    if (seg_list) hal_segmentation_list_free(seg_list);
    if (track_list) hal_track_info_list_free(track_list);

    // Frame 1: zero all scores to simulate no detections, verify tracker prediction
    {
        struct hal_tensor_map* map = hal_tensor_map_create(scores_tensor);
        ASSERT_NOT_NULL(map);
        uint8_t* data = (uint8_t*)hal_tensor_map_data(map);
        ASSERT_NOT_NULL(data);
        memset(data, 0, 10); // u8::MIN = 0
        hal_tensor_map_unmap(map);
    }

    box_list = NULL;
    seg_list = NULL;
    track_list = NULL;

    rc = hal_image_processor_draw_masks_tracked(
        processor, decoder, tracker, 100000000 / 3, outputs, 5, image, NULL, 1.0f, NULL, HAL_COLOR_MODE_CLASS, &box_list, &track_list
    );

    ASSERT_EQ(0, rc);
    ASSERT_NOT_NULL(box_list);

    // Tracker should predict the box forward (same location since no motion)
    ASSERT_EQ(1, hal_detect_box_list_len(box_list));
    hal_detect_box_list_get(box_list, 0, &box0);
    ASSERT_FLOAT_EQ(0.12549022f, box0.xmin, 1e-3f);
    ASSERT_FLOAT_EQ(0.12549022f, box0.ymin, 1e-3f);
    ASSERT_FLOAT_EQ(0.23529413f, box0.xmax, 1e-3f);
    ASSERT_FLOAT_EQ(0.23529413f, box0.ymax, 1e-3f);

    // No segmentation masks when boxes come from tracker prediction
    ASSERT_EQ(0, hal_segmentation_list_len(seg_list));

    hal_detect_box_list_free(box_list);
    if (seg_list) hal_segmentation_list_free(seg_list);
    if (track_list) hal_track_info_list_free(track_list);

    hal_tensor_free(boxes_tensor);
    hal_tensor_free(scores_tensor);
    hal_tensor_free(classes_tensor);
    hal_tensor_free(mask_tensor);
    hal_tensor_free(protos_tensor);
    hal_tensor_free(image);
    hal_bytetrack_free(tracker);
    hal_decoder_free(decoder);
    hal_image_processor_free(processor);
    TEST_PASS();
}


// =============================================================================
// Main Test Runner
// =============================================================================

void run_decoder_tests(void) {
    TEST_SUITE("Decoder");

    // Params tests
    test_decoder_params_new();
    test_decoder_new_with_json();
    test_decoder_new_with_yaml();
    test_decoder_new_with_thresholds();
    test_decoder_new_null_params();
    test_decoder_new_no_config();
    test_decoder_params_null_handling();

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

    // Tracked decoder tests
    test_decoder_tracked_linear_motion();
    test_decoder_tracked_end_to_end_segdet_split_proto();
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
