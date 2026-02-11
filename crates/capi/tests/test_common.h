// SPDX-FileCopyrightText: Copyright 2025 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0
//
// Common test utilities for EdgeFirst HAL C API tests
//
// Simple test framework for cross-platform compatibility (x86_64, aarch64, on-target)

#ifndef HAL_TEST_COMMON_H
#define HAL_TEST_COMMON_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <math.h>

// Include the HAL header
#include "../include/edgefirst/hal.h"

// Test result tracking - extern declarations, defined in test_all.c or standalone main
extern int tests_run;
extern int tests_passed;
extern int tests_failed;
extern const char* current_test_name;
extern const char* current_suite_name;

// ANSI color codes (disabled on non-TTY or embedded targets)
#ifdef HAL_TEST_NO_COLOR
#define COLOR_RED ""
#define COLOR_GREEN ""
#define COLOR_YELLOW ""
#define COLOR_RESET ""
#else
#define COLOR_RED "\033[31m"
#define COLOR_GREEN "\033[32m"
#define COLOR_YELLOW "\033[33m"
#define COLOR_RESET "\033[0m"
#endif

// Test macros
#define TEST_SUITE(name) \
    do { \
        current_suite_name = name; \
        printf("\n=== Test Suite: %s ===\n", name); \
    } while (0)

#define TEST(name) \
    do { \
        current_test_name = name; \
        tests_run++; \
    } while (0)

#define TEST_PASS() \
    do { \
        tests_passed++; \
        printf(COLOR_GREEN "[PASS]" COLOR_RESET " %s\n", current_test_name); \
    } while (0)

#define TEST_FAIL(msg) \
    do { \
        tests_failed++; \
        printf(COLOR_RED "[FAIL]" COLOR_RESET " %s: %s\n", current_test_name, msg); \
    } while (0)

#define TEST_SKIP(reason) \
    do { \
        printf(COLOR_YELLOW "[SKIP]" COLOR_RESET " %s: %s\n", current_test_name, reason); \
    } while (0)

// Assertion macros
#define ASSERT_TRUE(cond) \
    do { \
        if (!(cond)) { \
            char buf[256]; \
            snprintf(buf, sizeof(buf), "Expected true: %s (line %d)", #cond, __LINE__); \
            TEST_FAIL(buf); \
            return; \
        } \
    } while (0)

#define ASSERT_FALSE(cond) \
    do { \
        if (cond) { \
            char buf[256]; \
            snprintf(buf, sizeof(buf), "Expected false: %s (line %d)", #cond, __LINE__); \
            TEST_FAIL(buf); \
            return; \
        } \
    } while (0)

#define ASSERT_NULL(ptr) \
    do { \
        if ((ptr) != NULL) { \
            char buf[256]; \
            snprintf(buf, sizeof(buf), "Expected NULL: %s (line %d)", #ptr, __LINE__); \
            TEST_FAIL(buf); \
            return; \
        } \
    } while (0)

#define ASSERT_NOT_NULL(ptr) \
    do { \
        if ((ptr) == NULL) { \
            char buf[256]; \
            snprintf(buf, sizeof(buf), "Expected non-NULL: %s (line %d)", #ptr, __LINE__); \
            TEST_FAIL(buf); \
            return; \
        } \
    } while (0)

#define ASSERT_EQ(expected, actual) \
    do { \
        if ((expected) != (actual)) { \
            char buf[256]; \
            snprintf(buf, sizeof(buf), "Expected %ld, got %ld (line %d)", \
                     (long)(expected), (long)(actual), __LINE__); \
            TEST_FAIL(buf); \
            return; \
        } \
    } while (0)

#define ASSERT_EQ_STR(expected, actual) \
    do { \
        if (strcmp((expected), (actual)) != 0) { \
            char buf[256]; \
            snprintf(buf, sizeof(buf), "Expected '%s', got '%s' (line %d)", \
                     (expected), (actual), __LINE__); \
            TEST_FAIL(buf); \
            return; \
        } \
    } while (0)

#define ASSERT_FLOAT_EQ(expected, actual, epsilon) \
    do { \
        if (fabs((expected) - (actual)) > (epsilon)) { \
            char buf[256]; \
            snprintf(buf, sizeof(buf), "Expected %f, got %f (line %d)", \
                     (double)(expected), (double)(actual), __LINE__); \
            TEST_FAIL(buf); \
            return; \
        } \
    } while (0)

#define ASSERT_ERRNO(expected_errno) \
    do { \
        if (errno != (expected_errno)) { \
            char buf[256]; \
            snprintf(buf, sizeof(buf), "Expected errno %d (%s), got %d (%s) (line %d)", \
                     (expected_errno), strerror(expected_errno), errno, strerror(errno), __LINE__); \
            TEST_FAIL(buf); \
            return; \
        } \
    } while (0)

// Print test summary
static inline void print_test_summary(void) {
    printf("\n=== Test Summary ===\n");
    printf("Total:  %d\n", tests_run);
    printf(COLOR_GREEN "Passed: %d" COLOR_RESET "\n", tests_passed);
    if (tests_failed > 0) {
        printf(COLOR_RED "Failed: %d" COLOR_RESET "\n", tests_failed);
    } else {
        printf("Failed: %d\n", tests_failed);
    }
    printf("\n");
}

// Return exit code based on test results
static inline int get_test_exit_code(void) {
    return tests_failed > 0 ? 1 : 0;
}

// Check if DMA is available (Linux-specific)
static inline int is_dma_available(void) {
#ifdef __linux__
    // Try to create a small DMA tensor
    size_t shape[] = {16};
    struct hal_tensor* t = hal_tensor_new(HAL_DTYPE_U8, shape, 1, HAL_TENSOR_MEMORY_DMA, NULL);
    if (t != NULL) {
        // Check if it actually got DMA memory
        int has_dma = (hal_tensor_memory_type(t) == HAL_TENSOR_MEMORY_DMA);
        hal_tensor_free(t);
        return has_dma;
    }
    return 0;
#else
    return 0;
#endif
}

// Check if hardware accelerated image processing is available
static inline int is_hw_image_processing_available(void) {
    struct hal_image_processor* proc = hal_image_processor_new();
    if (proc != NULL) {
        hal_image_processor_free(proc);
        return 1;
    }
    return 0;
}

#endif // HAL_TEST_COMMON_H
