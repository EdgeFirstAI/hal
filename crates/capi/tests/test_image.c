// SPDX-FileCopyrightText: Copyright 2025 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0
//
// Image module tests for EdgeFirst HAL C API

#include "test_common.h"
#include <unistd.h>

// =============================================================================
// TensorImage Creation Tests
// =============================================================================

static void test_tensor_image_new_rgb(void) {
    TEST("tensor_image_new_rgb");

    struct hal_tensor_image* img = hal_tensor_image_new(640, 480, HAL_FOURCC_RGB, HAL_TENSOR_MEMORY_MEM);
    ASSERT_NOT_NULL(img);

    ASSERT_EQ(640, hal_tensor_image_width(img));
    ASSERT_EQ(480, hal_tensor_image_height(img));
    ASSERT_EQ(HAL_FOURCC_RGB, hal_tensor_image_fourcc(img));

    hal_tensor_image_free(img);
    TEST_PASS();
}

static void test_tensor_image_new_rgba(void) {
    TEST("tensor_image_new_rgba");

    struct hal_tensor_image* img = hal_tensor_image_new(1920, 1080, HAL_FOURCC_RGBA, HAL_TENSOR_MEMORY_MEM);
    ASSERT_NOT_NULL(img);

    ASSERT_EQ(1920, hal_tensor_image_width(img));
    ASSERT_EQ(1080, hal_tensor_image_height(img));
    ASSERT_EQ(HAL_FOURCC_RGBA, hal_tensor_image_fourcc(img));

    hal_tensor_image_free(img);
    TEST_PASS();
}

static void test_tensor_image_new_nv12(void) {
    TEST("tensor_image_new_nv12");

    struct hal_tensor_image* img = hal_tensor_image_new(640, 480, HAL_FOURCC_NV12, HAL_TENSOR_MEMORY_MEM);
    ASSERT_NOT_NULL(img);

    ASSERT_EQ(640, hal_tensor_image_width(img));
    ASSERT_EQ(480, hal_tensor_image_height(img));
    ASSERT_EQ(HAL_FOURCC_NV12, hal_tensor_image_fourcc(img));

    hal_tensor_image_free(img);
    TEST_PASS();
}

static void test_tensor_image_new_grey(void) {
    TEST("tensor_image_new_grey");

    struct hal_tensor_image* img = hal_tensor_image_new(256, 256, HAL_FOURCC_GREY, HAL_TENSOR_MEMORY_MEM);
    ASSERT_NOT_NULL(img);

    ASSERT_EQ(256, hal_tensor_image_width(img));
    ASSERT_EQ(256, hal_tensor_image_height(img));
    ASSERT_EQ(HAL_FOURCC_GREY, hal_tensor_image_fourcc(img));

    hal_tensor_image_free(img);
    TEST_PASS();
}

static void test_tensor_image_new_invalid(void) {
    TEST("tensor_image_new_invalid");

    // Zero width
    errno = 0;
    struct hal_tensor_image* img = hal_tensor_image_new(0, 480, HAL_FOURCC_RGB, HAL_TENSOR_MEMORY_MEM);
    ASSERT_NULL(img);
    ASSERT_ERRNO(EINVAL);

    // Zero height
    errno = 0;
    img = hal_tensor_image_new(640, 0, HAL_FOURCC_RGB, HAL_TENSOR_MEMORY_MEM);
    ASSERT_NULL(img);
    ASSERT_ERRNO(EINVAL);

    TEST_PASS();
}

static void test_tensor_image_null_handling(void) {
    TEST("tensor_image_null_handling");

    ASSERT_EQ(0, hal_tensor_image_width(NULL));
    ASSERT_EQ(0, hal_tensor_image_height(NULL));
    ASSERT_EQ(HAL_FOURCC_RGB, hal_tensor_image_fourcc(NULL)); // Default to RGB
    ASSERT_NULL(hal_tensor_image_tensor(NULL));

    // Free NULL should be no-op
    hal_tensor_image_free(NULL);

    TEST_PASS();
}

// =============================================================================
// Rect and Crop Tests
// =============================================================================

static void test_rect_new(void) {
    TEST("rect_new");

    struct hal_rect rect = hal_rect_new(10, 20, 100, 200);

    ASSERT_EQ(10, rect.left);
    ASSERT_EQ(20, rect.top);
    ASSERT_EQ(100, rect.width);
    ASSERT_EQ(200, rect.height);

    TEST_PASS();
}

static void test_crop_new(void) {
    TEST("crop_new");

    struct hal_crop crop = hal_crop_new();

    ASSERT_FALSE(crop.has_src_rect);
    ASSERT_FALSE(crop.has_dst_rect);
    ASSERT_FALSE(crop.has_dst_color);

    TEST_PASS();
}

static void test_crop_set_src_rect(void) {
    TEST("crop_set_src_rect");

    struct hal_crop crop = hal_crop_new();
    struct hal_rect rect = hal_rect_new(10, 20, 100, 200);

    hal_crop_set_src_rect(&crop, &rect);

    ASSERT_TRUE(crop.has_src_rect);
    ASSERT_EQ(10, crop.src_rect.left);
    ASSERT_EQ(20, crop.src_rect.top);
    ASSERT_EQ(100, crop.src_rect.width);
    ASSERT_EQ(200, crop.src_rect.height);

    // Clear src_rect
    hal_crop_set_src_rect(&crop, NULL);
    ASSERT_FALSE(crop.has_src_rect);

    TEST_PASS();
}

static void test_crop_set_dst_rect(void) {
    TEST("crop_set_dst_rect");

    struct hal_crop crop = hal_crop_new();
    struct hal_rect rect = hal_rect_new(0, 0, 320, 240);

    hal_crop_set_dst_rect(&crop, &rect);

    ASSERT_TRUE(crop.has_dst_rect);
    ASSERT_EQ(0, crop.dst_rect.left);
    ASSERT_EQ(0, crop.dst_rect.top);
    ASSERT_EQ(320, crop.dst_rect.width);
    ASSERT_EQ(240, crop.dst_rect.height);

    TEST_PASS();
}

static void test_crop_set_dst_color(void) {
    TEST("crop_set_dst_color");

    struct hal_crop crop = hal_crop_new();

    hal_crop_set_dst_color(&crop, 255, 128, 64, 200);

    ASSERT_TRUE(crop.has_dst_color);
    ASSERT_EQ(255, crop.dst_color[0]);
    ASSERT_EQ(128, crop.dst_color[1]);
    ASSERT_EQ(64, crop.dst_color[2]);
    ASSERT_EQ(200, crop.dst_color[3]);

    TEST_PASS();
}

// =============================================================================
// Image Processor Tests
// =============================================================================

static void test_image_processor_new(void) {
    TEST("image_processor_new");

    struct hal_image_processor* proc = hal_image_processor_new();
    // May be NULL if no backend available
    if (proc != NULL) {
        hal_image_processor_free(proc);
    } else {
        TEST_SKIP("No image processing backend available");
        return;
    }

    TEST_PASS();
}

static void test_image_processor_null_handling(void) {
    TEST("image_processor_null_handling");

    // Free NULL should be no-op
    hal_image_processor_free(NULL);

    TEST_PASS();
}

static void test_image_processor_convert_rgb_to_rgb(void) {
    TEST("image_processor_convert_rgb_to_rgb");

    struct hal_image_processor* proc = hal_image_processor_new();
    if (proc == NULL) {
        TEST_SKIP("No image processing backend available");
        return;
    }

    // Create source image
    struct hal_tensor_image* src = hal_tensor_image_new(320, 240, HAL_FOURCC_RGB, HAL_TENSOR_MEMORY_MEM);
    ASSERT_NOT_NULL(src);

    // Create destination image (same size)
    struct hal_tensor_image* dst = hal_tensor_image_new(320, 240, HAL_FOURCC_RGB, HAL_TENSOR_MEMORY_MEM);
    ASSERT_NOT_NULL(dst);

    // Convert with no rotation/flip/crop
    int result = hal_image_processor_convert(proc, src, dst,
                                              HAL_ROTATION_NONE, HAL_FLIP_NONE, NULL);
    ASSERT_EQ(0, result);

    hal_tensor_image_free(src);
    hal_tensor_image_free(dst);
    hal_image_processor_free(proc);
    TEST_PASS();
}

static void test_image_processor_convert_scale(void) {
    TEST("image_processor_convert_scale");

    struct hal_image_processor* proc = hal_image_processor_new();
    if (proc == NULL) {
        TEST_SKIP("No image processing backend available");
        return;
    }

    // Create source image
    struct hal_tensor_image* src = hal_tensor_image_new(640, 480, HAL_FOURCC_RGB, HAL_TENSOR_MEMORY_MEM);
    ASSERT_NOT_NULL(src);

    // Create destination image (different size - scaling)
    struct hal_tensor_image* dst = hal_tensor_image_new(320, 240, HAL_FOURCC_RGB, HAL_TENSOR_MEMORY_MEM);
    ASSERT_NOT_NULL(dst);

    // Scale down
    int result = hal_image_processor_convert(proc, src, dst,
                                              HAL_ROTATION_NONE, HAL_FLIP_NONE, NULL);
    ASSERT_EQ(0, result);

    hal_tensor_image_free(src);
    hal_tensor_image_free(dst);
    hal_image_processor_free(proc);
    TEST_PASS();
}

static void test_image_processor_convert_with_rotation(void) {
    TEST("image_processor_convert_with_rotation");

    struct hal_image_processor* proc = hal_image_processor_new();
    if (proc == NULL) {
        TEST_SKIP("No image processing backend available");
        return;
    }

    struct hal_tensor_image* src = hal_tensor_image_new(640, 480, HAL_FOURCC_RGB, HAL_TENSOR_MEMORY_MEM);
    ASSERT_NOT_NULL(src);

    // For 90-degree rotation, dst should be 480x640
    struct hal_tensor_image* dst = hal_tensor_image_new(480, 640, HAL_FOURCC_RGB, HAL_TENSOR_MEMORY_MEM);
    ASSERT_NOT_NULL(dst);

    int result = hal_image_processor_convert(proc, src, dst,
                                              HAL_ROTATION_ROTATE90, HAL_FLIP_NONE, NULL);
    ASSERT_EQ(0, result);

    hal_tensor_image_free(src);
    hal_tensor_image_free(dst);
    hal_image_processor_free(proc);
    TEST_PASS();
}

static void test_image_processor_convert_invalid(void) {
    TEST("image_processor_convert_invalid");

    struct hal_image_processor* proc = hal_image_processor_new();
    if (proc == NULL) {
        TEST_SKIP("No image processing backend available");
        return;
    }

    struct hal_tensor_image* img = hal_tensor_image_new(320, 240, HAL_FOURCC_RGB, HAL_TENSOR_MEMORY_MEM);
    ASSERT_NOT_NULL(img);

    // NULL processor
    errno = 0;
    int result = hal_image_processor_convert(NULL, img, img,
                                              HAL_ROTATION_NONE, HAL_FLIP_NONE, NULL);
    ASSERT_EQ(-1, result);
    ASSERT_ERRNO(EINVAL);

    // NULL src
    errno = 0;
    result = hal_image_processor_convert(proc, NULL, img,
                                          HAL_ROTATION_NONE, HAL_FLIP_NONE, NULL);
    ASSERT_EQ(-1, result);
    ASSERT_ERRNO(EINVAL);

    // NULL dst
    errno = 0;
    result = hal_image_processor_convert(proc, img, NULL,
                                          HAL_ROTATION_NONE, HAL_FLIP_NONE, NULL);
    ASSERT_EQ(-1, result);
    ASSERT_ERRNO(EINVAL);

    hal_tensor_image_free(img);
    hal_image_processor_free(proc);
    TEST_PASS();
}

// =============================================================================
// DMA Image Tests (Linux/on-target specific)
// =============================================================================

static void test_tensor_image_dma(void) {
    TEST("tensor_image_dma");

    if (!is_dma_available()) {
        TEST_SKIP("DMA memory not available");
        return;
    }

    struct hal_tensor_image* img = hal_tensor_image_new(640, 480, HAL_FOURCC_RGB, HAL_TENSOR_MEMORY_DMA);
    ASSERT_NOT_NULL(img);

    ASSERT_EQ(640, hal_tensor_image_width(img));
    ASSERT_EQ(480, hal_tensor_image_height(img));

    // Clone FD should work for DMA images
    int fd = hal_tensor_image_clone_fd(img);
    ASSERT_TRUE(fd >= 0);
    close(fd);

    hal_tensor_image_free(img);
    TEST_PASS();
}

// =============================================================================
// Main Test Runner
// =============================================================================

void run_image_tests(void) {
    TEST_SUITE("Image");

    // Creation tests
    test_tensor_image_new_rgb();
    test_tensor_image_new_rgba();
    test_tensor_image_new_nv12();
    test_tensor_image_new_grey();
    test_tensor_image_new_invalid();
    test_tensor_image_null_handling();

    // Rect and Crop tests
    test_rect_new();
    test_crop_new();
    test_crop_set_src_rect();
    test_crop_set_dst_rect();
    test_crop_set_dst_color();

    // Image processor tests
    test_image_processor_new();
    test_image_processor_null_handling();
    test_image_processor_convert_rgb_to_rgb();
    test_image_processor_convert_scale();
    test_image_processor_convert_with_rotation();
    test_image_processor_convert_invalid();

    // DMA tests
    test_tensor_image_dma();
}

#ifdef TEST_IMAGE_STANDALONE
// Define test result tracking variables for standalone mode
int tests_run = 0;
int tests_passed = 0;
int tests_failed = 0;
const char* current_test_name = NULL;
const char* current_suite_name = NULL;

int main(void) {
    run_image_tests();
    print_test_summary();
    return get_test_exit_code();
}
#endif
