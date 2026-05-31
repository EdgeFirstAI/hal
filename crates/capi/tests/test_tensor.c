// SPDX-FileCopyrightText: Copyright 2025 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0
//
// Tensor module tests for EdgeFirst HAL C API

#include "test_common.h"
#include <unistd.h>

// =============================================================================
// Tensor Creation Tests
// =============================================================================

static void test_tensor_new_f32(void) {
    TEST("tensor_new_f32");

    size_t shape[] = {2, 3, 4};
    struct hal_tensor* t = hal_tensor_new(HAL_DTYPE_F32, shape, 3, HAL_TENSOR_MEMORY_MEM, "test_f32");
    ASSERT_NOT_NULL(t);

    ASSERT_EQ(HAL_DTYPE_F32, hal_tensor_dtype(t));
    ASSERT_EQ(4, hal_tensor_dtype_size(t));
    ASSERT_EQ(HAL_TENSOR_MEMORY_MEM, hal_tensor_memory_type(t));
    ASSERT_EQ(24, hal_tensor_len(t));  // 2*3*4 = 24
    ASSERT_EQ(96, hal_tensor_size(t)); // 24 * 4 = 96 bytes

    size_t ndim;
    const size_t* s = hal_tensor_shape(t, &ndim);
    ASSERT_NOT_NULL(s);
    ASSERT_EQ(3, ndim);
    ASSERT_EQ(2, s[0]);
    ASSERT_EQ(3, s[1]);
    ASSERT_EQ(4, s[2]);

    char* name = hal_tensor_name(t);
    ASSERT_NOT_NULL(name);
    ASSERT_EQ_STR("test_f32", name);
    free(name);

    hal_tensor_free(t);
    TEST_PASS();
}

static void test_tensor_new_all_dtypes(void) {
    TEST("tensor_new_all_dtypes");

    struct {
        enum hal_dtype dtype;
        size_t expected_size;
        const char* name;
    } dtypes[] = {
        {HAL_DTYPE_U8, 1, "u8"},
        {HAL_DTYPE_I8, 1, "i8"},
        {HAL_DTYPE_U16, 2, "u16"},
        {HAL_DTYPE_I16, 2, "i16"},
        {HAL_DTYPE_U32, 4, "u32"},
        {HAL_DTYPE_I32, 4, "i32"},
        {HAL_DTYPE_U64, 8, "u64"},
        {HAL_DTYPE_I64, 8, "i64"},
        {HAL_DTYPE_F32, 4, "f32"},
        {HAL_DTYPE_F64, 8, "f64"},
    };

    size_t shape[] = {10};

    for (size_t i = 0; i < sizeof(dtypes) / sizeof(dtypes[0]); i++) {
        struct hal_tensor* t = hal_tensor_new(dtypes[i].dtype, shape, 1, HAL_TENSOR_MEMORY_MEM, dtypes[i].name);
        ASSERT_NOT_NULL(t);
        ASSERT_EQ(dtypes[i].dtype, hal_tensor_dtype(t));
        ASSERT_EQ(dtypes[i].expected_size, hal_tensor_dtype_size(t));
        hal_tensor_free(t);
    }

    TEST_PASS();
}

static void test_tensor_new_invalid(void) {
    TEST("tensor_new_invalid");

    size_t shape[] = {10};

    // NULL shape
    errno = 0;
    struct hal_tensor* t = hal_tensor_new(HAL_DTYPE_F32, NULL, 1, HAL_TENSOR_MEMORY_MEM, NULL);
    ASSERT_NULL(t);
    ASSERT_ERRNO(EINVAL);

    // Zero ndim
    errno = 0;
    t = hal_tensor_new(HAL_DTYPE_F32, shape, 0, HAL_TENSOR_MEMORY_MEM, NULL);
    ASSERT_NULL(t);
    ASSERT_ERRNO(EINVAL);

    TEST_PASS();
}

static void test_tensor_null_handling(void) {
    TEST("tensor_null_handling");

    // All these should handle NULL gracefully
    ASSERT_EQ(HAL_DTYPE_U8, hal_tensor_dtype(NULL));
    ASSERT_EQ(0, hal_tensor_dtype_size(NULL));
    ASSERT_EQ(HAL_TENSOR_MEMORY_MEM, hal_tensor_memory_type(NULL));
    ASSERT_NULL(hal_tensor_shape(NULL, NULL));
    ASSERT_EQ(0, hal_tensor_len(NULL));
    ASSERT_EQ(0, hal_tensor_size(NULL));

    // Free NULL should be no-op
    hal_tensor_free(NULL);

    TEST_PASS();
}

// =============================================================================
// Tensor Map Tests
// =============================================================================

static void test_tensor_map_write_read(void) {
    TEST("tensor_map_write_read");

    size_t shape[] = {4};
    struct hal_tensor* t = hal_tensor_new(HAL_DTYPE_F32, shape, 1, HAL_TENSOR_MEMORY_MEM, NULL);
    ASSERT_NOT_NULL(t);

    // Map for write
    struct hal_tensor_map* map = hal_tensor_map_create(t);
    ASSERT_NOT_NULL(map);

    float* data = (float*)hal_tensor_map_data(map);
    ASSERT_NOT_NULL(data);

    // Write data
    data[0] = 1.0f;
    data[1] = 2.0f;
    data[2] = 3.0f;
    data[3] = 4.0f;

    // Verify map properties
    ASSERT_EQ(4, hal_tensor_map_len(map));
    ASSERT_EQ(16, hal_tensor_map_size(map)); // 4 floats * 4 bytes

    size_t ndim;
    const size_t* s = hal_tensor_map_shape(map, &ndim);
    ASSERT_NOT_NULL(s);
    ASSERT_EQ(1, ndim);
    ASSERT_EQ(4, s[0]);

    hal_tensor_map_unmap(map);

    // Map again to verify data persists
    map = hal_tensor_map_create(t);
    ASSERT_NOT_NULL(map);

    const float* data_const = (const float*)hal_tensor_map_data_const(map);
    ASSERT_NOT_NULL(data_const);
    ASSERT_FLOAT_EQ(1.0f, data_const[0], 0.001f);
    ASSERT_FLOAT_EQ(2.0f, data_const[1], 0.001f);
    ASSERT_FLOAT_EQ(3.0f, data_const[2], 0.001f);
    ASSERT_FLOAT_EQ(4.0f, data_const[3], 0.001f);

    hal_tensor_map_unmap(map);
    hal_tensor_free(t);
    TEST_PASS();
}

static void test_tensor_map_null_handling(void) {
    TEST("tensor_map_null_handling");

    ASSERT_NULL(hal_tensor_map_create(NULL));
    ASSERT_NULL(hal_tensor_map_data(NULL));
    ASSERT_NULL(hal_tensor_map_data_const(NULL));
    ASSERT_NULL(hal_tensor_map_shape(NULL, NULL));
    ASSERT_EQ(0, hal_tensor_map_len(NULL));
    ASSERT_EQ(0, hal_tensor_map_size(NULL));

    // Unmap NULL should be no-op
    hal_tensor_map_unmap(NULL);

    TEST_PASS();
}

// =============================================================================
// Tensor Reshape Tests
// =============================================================================

static void test_tensor_reshape(void) {
    TEST("tensor_reshape");

    size_t shape[] = {2, 3, 4};
    struct hal_tensor* t = hal_tensor_new(HAL_DTYPE_F32, shape, 3, HAL_TENSOR_MEMORY_MEM, NULL);
    ASSERT_NOT_NULL(t);
    ASSERT_EQ(24, hal_tensor_len(t));

    // Reshape to 24x1
    size_t new_shape[] = {24};
    int result = hal_tensor_reshape(t, new_shape, 1);
    ASSERT_EQ(0, result);

    size_t ndim;
    const size_t* s = hal_tensor_shape(t, &ndim);
    ASSERT_EQ(1, ndim);
    ASSERT_EQ(24, s[0]);
    ASSERT_EQ(24, hal_tensor_len(t));

    // Reshape to 4x6
    size_t shape2[] = {4, 6};
    result = hal_tensor_reshape(t, shape2, 2);
    ASSERT_EQ(0, result);

    s = hal_tensor_shape(t, &ndim);
    ASSERT_EQ(2, ndim);
    ASSERT_EQ(4, s[0]);
    ASSERT_EQ(6, s[1]);

    hal_tensor_free(t);
    TEST_PASS();
}

static void test_tensor_reshape_invalid(void) {
    TEST("tensor_reshape_invalid");

    size_t shape[] = {10};
    struct hal_tensor* t = hal_tensor_new(HAL_DTYPE_F32, shape, 1, HAL_TENSOR_MEMORY_MEM, NULL);
    ASSERT_NOT_NULL(t);

    // Wrong element count
    size_t bad_shape[] = {5};
    errno = 0;
    int result = hal_tensor_reshape(t, bad_shape, 1);
    ASSERT_EQ(-1, result);
    ASSERT_ERRNO(EINVAL);

    // NULL shape
    errno = 0;
    result = hal_tensor_reshape(t, NULL, 1);
    ASSERT_EQ(-1, result);
    ASSERT_ERRNO(EINVAL);

    // Zero ndim
    errno = 0;
    result = hal_tensor_reshape(t, shape, 0);
    ASSERT_EQ(-1, result);
    ASSERT_ERRNO(EINVAL);

    // NULL tensor
    errno = 0;
    result = hal_tensor_reshape(NULL, shape, 1);
    ASSERT_EQ(-1, result);
    ASSERT_ERRNO(EINVAL);

    hal_tensor_free(t);
    TEST_PASS();
}

// =============================================================================
// DMA Memory Tests (Linux-specific)
// =============================================================================

static void test_tensor_dma_memory(void) {
    TEST("tensor_dma_memory");

    if (!is_dma_available()) {
        TEST_SKIP("DMA memory not available");
        return;
    }

    size_t shape[] = {1920, 1080, 3};
    struct hal_tensor* t = hal_tensor_new(HAL_DTYPE_U8, shape, 3, HAL_TENSOR_MEMORY_DMA, "dma_test");
    ASSERT_NOT_NULL(t);
    ASSERT_EQ(HAL_TENSOR_MEMORY_DMA, hal_tensor_memory_type(t));

    // Platform-specific export path: fd on Linux, IOSurfaceID on macOS.
    // The Dma tensor must expose whichever handle its platform uses.
#ifdef __linux__
    int fd = hal_tensor_clone_fd(t);
    ASSERT_TRUE(fd >= 0);
    close(fd);
#elif defined(__APPLE__)
    uint32_t id = hal_tensor_iosurface_id(t);
    ASSERT_TRUE(id != 0);
    void* surf = hal_tensor_iosurface_ref(t);
    ASSERT_NOT_NULL(surf);
#endif

    // Map and verify we can write data
    struct hal_tensor_map* map = hal_tensor_map_create(t);
    ASSERT_NOT_NULL(map);

    uint8_t* data = (uint8_t*)hal_tensor_map_data(map);
    ASSERT_NOT_NULL(data);

    // Write a pattern
    data[0] = 0xAB;
    data[1] = 0xCD;

    hal_tensor_map_unmap(map);

    // Verify data persists
    map = hal_tensor_map_create(t);
    const uint8_t* data_const = (const uint8_t*)hal_tensor_map_data_const(map);
    ASSERT_EQ(0xAB, data_const[0]);
    ASSERT_EQ(0xCD, data_const[1]);

    hal_tensor_map_unmap(map);
    hal_tensor_free(t);
    TEST_PASS();
}

#ifdef __APPLE__
// IOSurface round-trip: hal_tensor_from_iosurface must recover a
// tensor that shares the same IOSurfaceID as the original.
static void test_tensor_iosurface_roundtrip(void) {
    TEST("tensor_iosurface_roundtrip");

    if (!hal_is_iosurface_available()) {
        TEST_SKIP("IOSurface not available");
        return;
    }

    size_t shape[] = {720, 1280, 4};
    struct hal_tensor* t =
        hal_tensor_new(HAL_DTYPE_U8, shape, 3, HAL_TENSOR_MEMORY_DMA, "iosurface_rt");
    ASSERT_NOT_NULL(t);

    uint32_t id = hal_tensor_iosurface_id(t);
    ASSERT_TRUE(id != 0);

    void* surf = hal_tensor_iosurface_ref(t);
    ASSERT_NOT_NULL(surf);

    struct hal_tensor* imported =
        hal_tensor_from_iosurface(HAL_DTYPE_U8, surf, shape, 3, NULL);
    ASSERT_NOT_NULL(imported);
    ASSERT_EQ(HAL_TENSOR_MEMORY_DMA, hal_tensor_memory_type(imported));
    ASSERT_EQ(id, hal_tensor_iosurface_id(imported));

    hal_tensor_free(imported);
    hal_tensor_free(t);
    TEST_PASS();
}
#endif

static void test_tensor_clone_fd_mem_fails(void) {
    TEST("tensor_clone_fd_mem_fails");

    size_t shape[] = {10};
    struct hal_tensor* t = hal_tensor_new(HAL_DTYPE_F32, shape, 1, HAL_TENSOR_MEMORY_MEM, NULL);
    ASSERT_NOT_NULL(t);

    // Clone FD should fail for regular memory tensors
    errno = 0;
    int fd = hal_tensor_clone_fd(t);
    ASSERT_EQ(-1, fd);
    // errno should be ENOTSUP or EIO

    hal_tensor_free(t);
    TEST_PASS();
}

// =============================================================================
// Quantization Metadata Tests
// =============================================================================

static void test_tensor_quantization_float_returns_null(void) {
    TEST("tensor_quantization_float_returns_null");

    size_t shape[] = {2, 2};
    struct hal_tensor* t = hal_tensor_new(HAL_DTYPE_F32, shape, 2, HAL_TENSOR_MEMORY_MEM, NULL);
    ASSERT_NOT_NULL(t);
    // Float tensors never carry quantization.
    ASSERT_EQ(NULL, hal_tensor_quantization(t));
    hal_tensor_free(t);
    TEST_PASS();
}

static void test_tensor_quantization_null_input(void) {
    TEST("tensor_quantization_null_input");

    ASSERT_EQ(NULL, hal_tensor_quantization(NULL));
    ASSERT_EQ(0, hal_quantization_scale_len(NULL));
    ASSERT_EQ(0.0f, hal_quantization_scale_at(NULL, 0));
    ASSERT_EQ(0, hal_quantization_zero_point_at(NULL, 0));
    ASSERT_EQ(false, hal_quantization_is_symmetric(NULL));
    size_t axis = 999;
    ASSERT_EQ(false, hal_quantization_axis(NULL, &axis));
    // Must be safe to free NULL.
    hal_quantization_free(NULL);
    TEST_PASS();
}

// =============================================================================
// CUDA Capability and Fallback Tests
// =============================================================================

static void test_cuda_availability_and_fallback(void) {
    TEST("cuda_availability_and_fallback");

    // hal_is_cuda_available() must be callable and return a bool (no crash).
    (void)hal_is_cuda_available();

    // A Mem tensor has no registered CUDA handle, so cuda_map always returns
    // NULL regardless of whether libcudart is present.  Callers must fall back
    // to the host map in this case — verify that contract here.
    size_t shape[] = {4, 4};
    struct hal_tensor* t = hal_tensor_new(HAL_DTYPE_F32, shape, 2, HAL_TENSOR_MEMORY_MEM, NULL);
    ASSERT_NOT_NULL(t);

    void* cm = hal_tensor_cuda_map(t);
    ASSERT_NULL(cm); // Mem tensor never has a CUDA mapping

    if (cm) {
        // On a real CUDA host this branch exercises the zero-copy path.
        size_t sz = 0;
        void* dptr = hal_tensor_cuda_device_ptr(cm, &sz);
        ASSERT_NOT_NULL(dptr);
        hal_tensor_cuda_unmap(cm);
    } else {
        // Fallback: host map must succeed for a Mem tensor.
        struct hal_tensor_map* m = hal_tensor_map_create(t);
        ASSERT_NOT_NULL(m);
        void* data = hal_tensor_map_data(m);
        ASSERT_NOT_NULL(data);
        hal_tensor_map_unmap(m);
    }

    hal_tensor_free(t);
    TEST_PASS();
}

// =============================================================================
// Colorimetry Tests
// =============================================================================

static void test_tensor_colorimetry_roundtrip(void) {
    TEST("tensor_colorimetry_roundtrip");
    struct hal_tensor* t = hal_tensor_new_image(1280, 720, HAL_PIXEL_FORMAT_NV12, HAL_DTYPE_U8, HAL_TENSOR_MEMORY_MEM);
    ASSERT_NOT_NULL(t);
    struct hal_colorimetry c;
    ASSERT_EQ(0, hal_colorimetry_from_v4l2(3, 1, 2, 1, &c)); // REC709, XFER709, ENC709, FULL
    ASSERT_EQ(2, c.encoding); // bt709
    ASSERT_EQ(1, c.range);    // full
    hal_tensor_set_colorimetry(t, &c);
    struct hal_colorimetry got;
    ASSERT_EQ(0, hal_tensor_colorimetry(t, &got));
    ASSERT_EQ(c.encoding, got.encoding);
    ASSERT_EQ(c.range, got.range);
    ASSERT_EQ(c.space, got.space);
    ASSERT_EQ(c.transfer, got.transfer);
    hal_tensor_free(t);
    TEST_PASS();
}

static void test_tensor_colorimetry_clear_and_unset(void) {
    TEST("tensor_colorimetry_clear_and_unset");
    struct hal_tensor* t = hal_tensor_new_image(64, 64, HAL_PIXEL_FORMAT_NV12, HAL_DTYPE_U8, HAL_TENSOR_MEMORY_MEM);
    ASSERT_NOT_NULL(t);

    // Freshly created tensor has no colorimetry: all axes read back as 0.
    struct hal_colorimetry got;
    ASSERT_EQ(0, hal_tensor_colorimetry(t, &got));
    ASSERT_EQ(0, got.space);
    ASSERT_EQ(0, got.transfer);
    ASSERT_EQ(0, got.encoding);
    ASSERT_EQ(0, got.range);

    // Set something, then clear with NULL.
    struct hal_colorimetry c = {1, 1, 1, 1};
    hal_tensor_set_colorimetry(t, &c);
    ASSERT_EQ(0, hal_tensor_colorimetry(t, &got));
    ASSERT_EQ(1, got.space);
    hal_tensor_set_colorimetry(t, NULL);
    ASSERT_EQ(0, hal_tensor_colorimetry(t, &got));
    ASSERT_EQ(0, got.space);
    ASSERT_EQ(0, got.range);

    // NULL-arg error handling.
    ASSERT_EQ(-1, hal_tensor_colorimetry(NULL, &got));
    ASSERT_EQ(-1, hal_tensor_colorimetry(t, NULL));
    ASSERT_EQ(-1, hal_colorimetry_from_v4l2(3, 1, 2, 1, NULL));

    hal_tensor_free(t);
    TEST_PASS();
}

// =============================================================================
// Main Test Runner
// =============================================================================

void run_tensor_tests(void) {
    TEST_SUITE("Tensor");

    // Creation tests
    test_tensor_new_f32();
    test_tensor_new_all_dtypes();
    test_tensor_new_invalid();
    test_tensor_null_handling();

    // Map tests
    test_tensor_map_write_read();
    test_tensor_map_null_handling();

    // Reshape tests
    test_tensor_reshape();
    test_tensor_reshape_invalid();

    // DMA tests (Linux DMA-BUF + macOS IOSurface)
    test_tensor_dma_memory();
    test_tensor_clone_fd_mem_fails();
#ifdef __APPLE__
    test_tensor_iosurface_roundtrip();
#endif

    // Quantization accessor tests
    test_tensor_quantization_float_returns_null();
    test_tensor_quantization_null_input();

    // CUDA capability query and try-cuda_map / fallback pattern
    test_cuda_availability_and_fallback();

    // Colorimetry tests
    test_tensor_colorimetry_roundtrip();
    test_tensor_colorimetry_clear_and_unset();
}

#ifdef TEST_TENSOR_STANDALONE
// Define test result tracking variables for standalone mode
int tests_run = 0;
int tests_passed = 0;
int tests_failed = 0;
const char* current_test_name = NULL;
const char* current_suite_name = NULL;

int main(void) {
    run_tensor_tests();
    print_test_summary();
    return get_test_exit_code();
}
#endif
