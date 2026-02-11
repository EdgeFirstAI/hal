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

    // Clone FD should work for DMA tensors
    int fd = hal_tensor_clone_fd(t);
    ASSERT_TRUE(fd >= 0);

    // Close the cloned fd
    close(fd);

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

    // DMA tests (Linux-specific)
    test_tensor_dma_memory();
    test_tensor_clone_fd_mem_fails();
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
