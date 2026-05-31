// SPDX-FileCopyrightText: Copyright 2025 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0
//
// Image module tests for EdgeFirst HAL C API

#include "test_common.h"
#include <unistd.h>

// =============================================================================
// Image Tensor Creation Tests
// =============================================================================

static void test_tensor_image_new_rgb(void) {
    TEST("tensor_image_new_rgb");

    struct hal_tensor* img = hal_tensor_new_image(640, 480, HAL_PIXEL_FORMAT_RGB, HAL_DTYPE_U8, HAL_TENSOR_MEMORY_MEM);
    ASSERT_NOT_NULL(img);

    ASSERT_EQ(640, hal_tensor_width(img));
    ASSERT_EQ(480, hal_tensor_height(img));
    ASSERT_EQ(HAL_PIXEL_FORMAT_RGB, hal_tensor_pixel_format(img));

    hal_tensor_free(img);
    TEST_PASS();
}

static void test_tensor_image_new_rgba(void) {
    TEST("tensor_image_new_rgba");

    struct hal_tensor* img = hal_tensor_new_image(1920, 1080, HAL_PIXEL_FORMAT_RGBA, HAL_DTYPE_U8, HAL_TENSOR_MEMORY_MEM);
    ASSERT_NOT_NULL(img);

    ASSERT_EQ(1920, hal_tensor_width(img));
    ASSERT_EQ(1080, hal_tensor_height(img));
    ASSERT_EQ(HAL_PIXEL_FORMAT_RGBA, hal_tensor_pixel_format(img));

    hal_tensor_free(img);
    TEST_PASS();
}

static void test_tensor_image_new_nv12(void) {
    TEST("tensor_image_new_nv12");

    struct hal_tensor* img = hal_tensor_new_image(640, 480, HAL_PIXEL_FORMAT_NV12, HAL_DTYPE_U8, HAL_TENSOR_MEMORY_MEM);
    ASSERT_NOT_NULL(img);

    ASSERT_EQ(640, hal_tensor_width(img));
    ASSERT_EQ(480, hal_tensor_height(img));
    ASSERT_EQ(HAL_PIXEL_FORMAT_NV12, hal_tensor_pixel_format(img));

    hal_tensor_free(img);
    TEST_PASS();
}

static void test_tensor_image_new_grey(void) {
    TEST("tensor_image_new_grey");

    struct hal_tensor* img = hal_tensor_new_image(256, 256, HAL_PIXEL_FORMAT_GREY, HAL_DTYPE_U8, HAL_TENSOR_MEMORY_MEM);
    ASSERT_NOT_NULL(img);

    ASSERT_EQ(256, hal_tensor_width(img));
    ASSERT_EQ(256, hal_tensor_height(img));
    ASSERT_EQ(HAL_PIXEL_FORMAT_GREY, hal_tensor_pixel_format(img));

    hal_tensor_free(img);
    TEST_PASS();
}

static void test_tensor_image_new_invalid(void) {
    TEST("tensor_image_new_invalid");

    // Zero width
    errno = 0;
    struct hal_tensor* img = hal_tensor_new_image(0, 480, HAL_PIXEL_FORMAT_RGB, HAL_DTYPE_U8, HAL_TENSOR_MEMORY_MEM);
    ASSERT_NULL(img);
    ASSERT_ERRNO(EINVAL);

    // Zero height
    errno = 0;
    img = hal_tensor_new_image(640, 0, HAL_PIXEL_FORMAT_RGB, HAL_DTYPE_U8, HAL_TENSOR_MEMORY_MEM);
    ASSERT_NULL(img);
    ASSERT_ERRNO(EINVAL);

    TEST_PASS();
}

static void test_tensor_image_null_handling(void) {
    TEST("tensor_image_null_handling");

    ASSERT_EQ(0, hal_tensor_width(NULL));
    ASSERT_EQ(0, hal_tensor_height(NULL));
    ASSERT_EQ(HAL_PIXEL_FORMAT_RGB, hal_tensor_pixel_format(NULL)); // Default to RGB
    ASSERT_NULL(hal_tensor_shape(NULL, NULL));

    // Free NULL should be no-op
    hal_tensor_free(NULL);

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
    struct hal_tensor* src = hal_tensor_new_image(320, 240, HAL_PIXEL_FORMAT_RGB, HAL_DTYPE_U8, HAL_TENSOR_MEMORY_MEM);
    ASSERT_NOT_NULL(src);

    // Create destination image (same size)
    struct hal_tensor* dst = hal_tensor_new_image(320, 240, HAL_PIXEL_FORMAT_RGB, HAL_DTYPE_U8, HAL_TENSOR_MEMORY_MEM);
    ASSERT_NOT_NULL(dst);

    // Convert with no rotation/flip/crop
    int result = hal_image_processor_convert(proc, src, dst,
                                              HAL_ROTATION_NONE, HAL_FLIP_NONE, NULL);
    ASSERT_EQ(0, result);

    hal_tensor_free(src);
    hal_tensor_free(dst);
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
    struct hal_tensor* src = hal_tensor_new_image(640, 480, HAL_PIXEL_FORMAT_RGB, HAL_DTYPE_U8, HAL_TENSOR_MEMORY_MEM);
    ASSERT_NOT_NULL(src);

    // Create destination image (different size - scaling)
    struct hal_tensor* dst = hal_tensor_new_image(320, 240, HAL_PIXEL_FORMAT_RGB, HAL_DTYPE_U8, HAL_TENSOR_MEMORY_MEM);
    ASSERT_NOT_NULL(dst);

    // Scale down
    int result = hal_image_processor_convert(proc, src, dst,
                                              HAL_ROTATION_NONE, HAL_FLIP_NONE, NULL);
    ASSERT_EQ(0, result);

    hal_tensor_free(src);
    hal_tensor_free(dst);
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

    struct hal_tensor* src = hal_tensor_new_image(640, 480, HAL_PIXEL_FORMAT_RGB, HAL_DTYPE_U8, HAL_TENSOR_MEMORY_MEM);
    ASSERT_NOT_NULL(src);

    // For 90-degree rotation, dst should be 480x640
    struct hal_tensor* dst = hal_tensor_new_image(480, 640, HAL_PIXEL_FORMAT_RGB, HAL_DTYPE_U8, HAL_TENSOR_MEMORY_MEM);
    ASSERT_NOT_NULL(dst);

    int result = hal_image_processor_convert(proc, src, dst,
                                              HAL_ROTATION_ROTATE90, HAL_FLIP_NONE, NULL);
    ASSERT_EQ(0, result);

    hal_tensor_free(src);
    hal_tensor_free(dst);
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

    struct hal_tensor* img = hal_tensor_new_image(320, 240, HAL_PIXEL_FORMAT_RGB, HAL_DTYPE_U8, HAL_TENSOR_MEMORY_MEM);
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

    hal_tensor_free(img);
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

    // macOS IOSurface allocation requires a format with a defined
    // FourCC (Yuyv / Rgba / Bgra today); RGB falls through to SHM and
    // there's no fd or IOSurface to inspect. Use RGBA so the test
    // exercises the Dma path on both Linux DMA-buf and macOS IOSurface.
    struct hal_tensor* img = hal_tensor_new_image(640, 480, HAL_PIXEL_FORMAT_RGBA, HAL_DTYPE_U8, HAL_TENSOR_MEMORY_DMA);
    ASSERT_NOT_NULL(img);

    ASSERT_EQ(640, hal_tensor_width(img));
    ASSERT_EQ(480, hal_tensor_height(img));

    // Platform-specific export path: fd on Linux, IOSurfaceID on macOS.
#ifdef __linux__
    int fd = hal_tensor_clone_fd(img);
    ASSERT_TRUE(fd >= 0);
    close(fd);
#elif defined(__APPLE__)
    ASSERT_EQ(HAL_TENSOR_MEMORY_DMA, hal_tensor_memory_type(img));
    uint32_t id = hal_tensor_iosurface_id(img);
    ASSERT_TRUE(id != 0);
#endif

    hal_tensor_free(img);
    TEST_PASS();
}

// =============================================================================
// Image Decode Tests (into pre-allocated tensors)
// =============================================================================

static const char* test_image_jpeg_path(void) {
    if (access("testdata/zidane.jpg", R_OK) == 0) {
        return "testdata/zidane.jpg";
    }
    if (access("../../../testdata/zidane.jpg", R_OK) == 0) {
        return "../../../testdata/zidane.jpg";
    }
    return NULL;
}

static void test_tensor_decode_image_jpeg(void) {
    TEST("tensor_decode_image_jpeg");

    // Create a pre-allocated tensor large enough for zidane.jpg (1280x720).
    // JPEG decodes to its native NV12 format, so allocate NV12.
    struct hal_tensor* tensor = hal_tensor_new_image(1280, 720, HAL_PIXEL_FORMAT_NV12, HAL_DTYPE_U8, HAL_TENSOR_MEMORY_MEM);
    ASSERT_NOT_NULL(tensor);

    const char* path = test_image_jpeg_path();
    if (!path) {
        hal_tensor_free(tensor);
        fprintf(stderr, "    SKIP: testdata/zidane.jpg not found\n");
        tests_run--;
        return;
    }

    // Read the JPEG file
    FILE* f = fopen(path, "rb");
    if (!f) {
        hal_tensor_free(tensor);
        fprintf(stderr, "    SKIP: testdata/zidane.jpg not found\n");
        tests_run--;
        return;
    }
    fseek(f, 0, SEEK_END);
    long fsize = ftell(f);
    fseek(f, 0, SEEK_SET);
    uint8_t* data = (uint8_t*)malloc((size_t)fsize);
    ASSERT_NOT_NULL(data);
    size_t bytes_read = fread(data, 1, (size_t)fsize, f);
    fclose(f);
    if (bytes_read != (size_t)fsize) {
        free(data);
        hal_tensor_free(tensor);
        fprintf(stderr, "    SKIP: failed to read testdata/zidane.jpg\n");
        tests_run--;
        return;
    }

    // Decode into the pre-allocated tensor. The decoder emits the native
    // format (NV12 for a colour JPEG) and configures the tensor accordingly.
    size_t width = 0, height = 0;
    uint16_t rotation = 0xFFFF;
    bool flip = true;
    int ret = hal_tensor_decode_image(tensor, data, (size_t)fsize, &width, &height,
                                      &rotation, &flip);
    ASSERT_EQ(0, ret);
    ASSERT_EQ(1280, (int)width);
    ASSERT_EQ(720, (int)height);
    ASSERT_EQ(HAL_PIXEL_FORMAT_NV12, hal_tensor_pixel_format(tensor));
    // zidane.jpg carries no EXIF orientation → identity transform reported.
    ASSERT_EQ(0, (int)rotation);
    ASSERT_EQ(0, (int)flip);

    free(data);
    hal_tensor_free(tensor);
    TEST_PASS();
}

static void test_tensor_decode_image_file_jpeg(void) {
    TEST("tensor_decode_image_file_jpeg");

    // JPEG decodes to its native NV12 format.
    struct hal_tensor* tensor = hal_tensor_new_image(1280, 720, HAL_PIXEL_FORMAT_NV12, HAL_DTYPE_U8, HAL_TENSOR_MEMORY_MEM);
    ASSERT_NOT_NULL(tensor);

    const char* path = test_image_jpeg_path();
    if (!path) {
        hal_tensor_free(tensor);
        fprintf(stderr, "    SKIP: testdata/zidane.jpg not found or decode failed\n");
        tests_run--;
        return;
    }

    size_t width = 0, height = 0;
    uint16_t rotation = 0xFFFF;
    bool flip = true;
    int ret = hal_tensor_decode_image_file(tensor, path, &width, &height, &rotation, &flip);
    if (ret != 0) {
        // File might not exist in test environment
        hal_tensor_free(tensor);
        fprintf(stderr, "    SKIP: testdata/zidane.jpg not found or decode failed\n");
        tests_run--;
        return;
    }
    ASSERT_EQ(0, ret);
    ASSERT_EQ(1280, (int)width);
    ASSERT_EQ(720, (int)height);
    ASSERT_EQ(HAL_PIXEL_FORMAT_NV12, hal_tensor_pixel_format(tensor));
    ASSERT_EQ(0, (int)rotation);
    ASSERT_EQ(0, (int)flip);

    // Consumers that need RGBA convert the native result themselves.
    struct hal_image_processor* proc = hal_image_processor_new();
    if (proc) {
        struct hal_tensor* rgba = hal_tensor_new_image(1280, 720, HAL_PIXEL_FORMAT_RGBA, HAL_DTYPE_U8, HAL_TENSOR_MEMORY_MEM);
        ASSERT_NOT_NULL(rgba);
        int cret = hal_image_processor_convert(proc, tensor, rgba, HAL_ROTATION_NONE, HAL_FLIP_NONE, NULL);
        ASSERT_EQ(0, cret);
        ASSERT_EQ(HAL_PIXEL_FORMAT_RGBA, hal_tensor_pixel_format(rgba));
        hal_tensor_free(rgba);
        hal_image_processor_free(proc);
    }

    hal_tensor_free(tensor);
    TEST_PASS();
}

static void test_tensor_decode_image_native_format(void) {
    TEST("tensor_decode_image_native_format");

    // The decoder always emits the source's native format; for a colour JPEG
    // that is NV12. The tensor is configured to that format on decode.
    struct hal_tensor* tensor = hal_tensor_new_image(1280, 720, HAL_PIXEL_FORMAT_NV12, HAL_DTYPE_U8, HAL_TENSOR_MEMORY_MEM);
    ASSERT_NOT_NULL(tensor);

    const char* path = test_image_jpeg_path();
    if (!path) {
        hal_tensor_free(tensor);
        fprintf(stderr, "    SKIP: testdata/zidane.jpg not found or decode failed\n");
        tests_run--;
        return;
    }

    size_t width = 0, height = 0;
    int ret = hal_tensor_decode_image_file(tensor, path, &width, &height, NULL, NULL);
    if (ret != 0) {
        hal_tensor_free(tensor);
        fprintf(stderr, "    SKIP: testdata/zidane.jpg not found or decode failed\n");
        tests_run--;
        return;
    }
    ASSERT_EQ(0, ret);
    ASSERT_EQ(1280, (int)width);
    ASSERT_EQ(720, (int)height);
    ASSERT_EQ(HAL_PIXEL_FORMAT_NV12, hal_tensor_pixel_format(tensor));

    hal_tensor_free(tensor);
    TEST_PASS();
}

// Resolve an EXIF-oriented test JPEG (zidane_exif_<tag>.jpg) from either the
// repo root or the crate-relative testdata directory. Returns NULL if absent.
static const char* test_image_exif_jpeg_path(const char* name) {
    static char buf[256];
    snprintf(buf, sizeof(buf), "testdata/%s", name);
    if (access(buf, R_OK) == 0) {
        return buf;
    }
    snprintf(buf, sizeof(buf), "../../../testdata/%s", name);
    if (access(buf, R_OK) == 0) {
        return buf;
    }
    return NULL;
}

// The decoder reports EXIF orientation in the out-params but never rotates the
// pixels: dimensions stay at the source's native (unrotated) 1280x720, and the
// reported (rotation, flip) is the transform the caller should apply.
static void test_tensor_decode_image_exif_orientation(void) {
    TEST("tensor_decode_image_exif_orientation");

    // EXIF tag 3 → 180° clockwise, no flip; tag 2 → 0°, horizontal flip.
    // (See crates/codec/src/exif.rs for the full tag→transform mapping.)
    struct {
        const char* name;
        int rotation;
        int flip;
    } cases[] = {
        {"zidane_exif_3.jpg", 180, 0},
        {"zidane_exif_2.jpg", 0, 1},
    };

    int checked = 0;
    for (size_t i = 0; i < sizeof(cases) / sizeof(cases[0]); i++) {
        const char* path = test_image_exif_jpeg_path(cases[i].name);
        if (!path) {
            continue;
        }
        struct hal_tensor* tensor = hal_tensor_new_image(
            1280, 720, HAL_PIXEL_FORMAT_NV12, HAL_DTYPE_U8, HAL_TENSOR_MEMORY_MEM);
        ASSERT_NOT_NULL(tensor);

        size_t width = 0, height = 0;
        uint16_t rotation = 0xFFFF;
        bool flip = !cases[i].flip;
        int ret = hal_tensor_decode_image_file(tensor, path, &width, &height,
                                               &rotation, &flip);
        ASSERT_EQ(0, ret);
        // Codec never rotates → native dimensions regardless of orientation.
        ASSERT_EQ(1280, (int)width);
        ASSERT_EQ(720, (int)height);
        ASSERT_EQ(cases[i].rotation, (int)rotation);
        ASSERT_EQ(cases[i].flip, (int)flip);

        hal_tensor_free(tensor);
        checked++;
    }

    if (checked == 0) {
        fprintf(stderr, "    SKIP: no testdata/zidane_exif_*.jpg found\n");
        tests_run--;
        return;
    }
    TEST_PASS();
}

static void test_tensor_decode_image_null_handling(void) {
    TEST("tensor_decode_image_null_handling");

    // NULL tensor
    int ret = hal_tensor_decode_image(NULL, (const uint8_t*)"data", 4, NULL, NULL, NULL, NULL);
    ASSERT_EQ(-1, ret);

    // NULL data
    struct hal_tensor* tensor = hal_tensor_new_image(640, 480, HAL_PIXEL_FORMAT_RGB, HAL_DTYPE_U8, HAL_TENSOR_MEMORY_MEM);
    ASSERT_NOT_NULL(tensor);
    ret = hal_tensor_decode_image(tensor, NULL, 100, NULL, NULL, NULL, NULL);
    ASSERT_EQ(-1, ret);

    // Zero length
    ret = hal_tensor_decode_image(tensor, (const uint8_t*)"data", 0, NULL, NULL, NULL, NULL);
    ASSERT_EQ(-1, ret);

    hal_tensor_free(tensor);
    TEST_PASS();
}

static void test_tensor_decode_image_file_null_handling(void) {
    TEST("tensor_decode_image_file_null_handling");

    int ret = hal_tensor_decode_image_file(NULL, "testdata/zidane.jpg", NULL, NULL, NULL, NULL);
    ASSERT_EQ(-1, ret);

    struct hal_tensor* tensor = hal_tensor_new_image(640, 480, HAL_PIXEL_FORMAT_RGB, HAL_DTYPE_U8, HAL_TENSOR_MEMORY_MEM);
    ASSERT_NOT_NULL(tensor);
    ret = hal_tensor_decode_image_file(tensor, NULL, NULL, NULL, NULL, NULL);
    ASSERT_EQ(-1, ret);

    hal_tensor_free(tensor);
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

    // Decode tests (new codec API)
    test_tensor_decode_image_jpeg();
    test_tensor_decode_image_file_jpeg();
    test_tensor_decode_image_native_format();
    test_tensor_decode_image_exif_orientation();
    test_tensor_decode_image_null_handling();
    test_tensor_decode_image_file_null_handling();
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
