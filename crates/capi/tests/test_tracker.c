// SPDX-FileCopyrightText: Copyright 2025 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0
//
// Tracker module tests for EdgeFirst HAL C API

#include "test_common.h"

// =============================================================================
// ByteTrack Creation Tests
// =============================================================================

static void test_bytetrack_new_default(void) {
    TEST("bytetrack_new_default");

    struct hal_bytetrack* tracker = hal_bytetrack_new_default();
    ASSERT_NOT_NULL(tracker);

    hal_bytetrack_free(tracker);
    TEST_PASS();
}

static void test_bytetrack_new_with_params(void) {
    TEST("bytetrack_new_with_params");

    struct hal_bytetrack* tracker = hal_bytetrack_new(
        0.3f,   // track_thresh
        0.8f,   // high_thresh
        0.2f,   // match_thresh
        30,     // frame_rate
        60      // track_buffer (frames)
    );
    ASSERT_NOT_NULL(tracker);

    hal_bytetrack_free(tracker);
    TEST_PASS();
}

static void test_bytetrack_null_handling(void) {
    TEST("bytetrack_null_handling");

    // Free NULL should be no-op
    hal_bytetrack_free(NULL);

    // Get active tracks from NULL
    errno = 0;
    struct hal_track_info_list* list = hal_bytetrack_get_active_tracks(NULL);
    ASSERT_NULL(list);
    ASSERT_ERRNO(EINVAL);

    // Update with NULL tracker
    errno = 0;
    list = hal_bytetrack_update(NULL, NULL, 0);
    ASSERT_NULL(list);
    ASSERT_ERRNO(EINVAL);

    TEST_PASS();
}

// =============================================================================
// Track Info List Tests
// =============================================================================

static void test_track_info_list_null_handling(void) {
    TEST("track_info_list_null_handling");

    ASSERT_EQ(0, hal_track_info_list_len(NULL));

    struct hal_track_info info;
    errno = 0;
    int result = hal_track_info_list_get(NULL, 0, &info);
    ASSERT_EQ(-1, result);
    ASSERT_ERRNO(EINVAL);

    // Free NULL should be no-op
    hal_track_info_list_free(NULL);

    TEST_PASS();
}

static void test_track_info_list_get_out_of_bounds(void) {
    TEST("track_info_list_get_out_of_bounds");

    struct hal_bytetrack* tracker = hal_bytetrack_new_default();
    ASSERT_NOT_NULL(tracker);

    // Get active tracks (should be empty initially)
    struct hal_track_info_list* list = hal_bytetrack_get_active_tracks(tracker);
    ASSERT_NOT_NULL(list);

    size_t len = hal_track_info_list_len(list);
    ASSERT_EQ(0, len);

    // Try to get from empty list
    struct hal_track_info info;
    errno = 0;
    int result = hal_track_info_list_get(list, 0, &info);
    ASSERT_EQ(-1, result);
    ASSERT_ERRNO(EINVAL);

    hal_track_info_list_free(list);
    hal_bytetrack_free(tracker);
    TEST_PASS();
}

// =============================================================================
// UUID Utility Tests
// =============================================================================

static void test_uuid_to_string_zeros(void) {
    TEST("uuid_to_string_zeros");

    uint8_t uuid[16] = {0};
    char buffer[37];

    int result = hal_uuid_to_string(&uuid, buffer, sizeof(buffer));
    ASSERT_EQ(0, result);
    ASSERT_EQ_STR("00000000-0000-0000-0000-000000000000", buffer);

    TEST_PASS();
}

static void test_uuid_to_string_nonzero(void) {
    TEST("uuid_to_string_nonzero");

    // UUID: 12345678-1234-5678-1234-567812345678
    uint8_t uuid[16] = {
        0x12, 0x34, 0x56, 0x78,
        0x12, 0x34,
        0x56, 0x78,
        0x12, 0x34,
        0x56, 0x78, 0x12, 0x34, 0x56, 0x78
    };
    char buffer[37];

    int result = hal_uuid_to_string(&uuid, buffer, sizeof(buffer));
    ASSERT_EQ(0, result);
    ASSERT_EQ_STR("12345678-1234-5678-1234-567812345678", buffer);

    TEST_PASS();
}

static void test_uuid_to_string_buffer_too_small(void) {
    TEST("uuid_to_string_buffer_too_small");

    uint8_t uuid[16] = {0};
    char buffer[36]; // Too small (need 37)

    errno = 0;
    int result = hal_uuid_to_string(&uuid, buffer, sizeof(buffer));
    ASSERT_EQ(-1, result);
    ASSERT_ERRNO(EINVAL);

    TEST_PASS();
}

static void test_uuid_to_string_null_params(void) {
    TEST("uuid_to_string_null_params");

    uint8_t uuid[16] = {0};
    char buffer[37];

    // NULL uuid
    errno = 0;
    int result = hal_uuid_to_string(NULL, buffer, sizeof(buffer));
    ASSERT_EQ(-1, result);
    ASSERT_ERRNO(EINVAL);

    // NULL buffer
    errno = 0;
    result = hal_uuid_to_string(&uuid, NULL, sizeof(buffer));
    ASSERT_EQ(-1, result);
    ASSERT_ERRNO(EINVAL);

    TEST_PASS();
}

// =============================================================================
// ByteTrack Update/Track Tests
// =============================================================================

static void test_bytetrack_update_null_detections(void) {
    TEST("bytetrack_update_null_detections");

    struct hal_bytetrack* tracker = hal_bytetrack_new_default();
    ASSERT_NOT_NULL(tracker);

    // Update with NULL detections
    errno = 0;
    struct hal_track_info_list* list = hal_bytetrack_update(tracker, NULL, 0);
    ASSERT_NULL(list);
    ASSERT_ERRNO(EINVAL);

    hal_bytetrack_free(tracker);
    TEST_PASS();
}

static void test_bytetrack_get_active_tracks_empty(void) {
    TEST("bytetrack_get_active_tracks_empty");

    struct hal_bytetrack* tracker = hal_bytetrack_new_default();
    ASSERT_NOT_NULL(tracker);

    // Get active tracks without any updates
    struct hal_track_info_list* list = hal_bytetrack_get_active_tracks(tracker);
    ASSERT_NOT_NULL(list);

    // Should be empty
    ASSERT_EQ(0, hal_track_info_list_len(list));

    hal_track_info_list_free(list);
    hal_bytetrack_free(tracker);
    TEST_PASS();
}

static void test_track_info_struct(void) {
    TEST("track_info_struct");

    // Test track_info struct layout
    struct hal_track_info info = {0};

    // UUID should be 16 bytes
    ASSERT_EQ(16, sizeof(info.uuid));

    // Location should be 4 floats
    ASSERT_EQ(16, sizeof(info.location)); // 4 * 4 bytes

    // Count should be 32-bit
    ASSERT_EQ(4, sizeof(info.count));

    // Timestamps should be 64-bit
    ASSERT_EQ(8, sizeof(info.created));
    ASSERT_EQ(8, sizeof(info.last_updated));

    TEST_PASS();
}

// =============================================================================
// Main Test Runner
// =============================================================================

void run_tracker_tests(void) {
    TEST_SUITE("Tracker");

    // ByteTrack creation tests
    test_bytetrack_new_default();
    test_bytetrack_new_with_params();
    test_bytetrack_null_handling();

    // Track info list tests
    test_track_info_list_null_handling();
    test_track_info_list_get_out_of_bounds();

    // UUID utility tests
    test_uuid_to_string_zeros();
    test_uuid_to_string_nonzero();
    test_uuid_to_string_buffer_too_small();
    test_uuid_to_string_null_params();

    // ByteTrack update/track tests
    test_bytetrack_update_null_detections();
    test_bytetrack_get_active_tracks_empty();
    test_track_info_struct();
}

#ifdef TEST_TRACKER_STANDALONE
// Define test result tracking variables for standalone mode
int tests_run = 0;
int tests_passed = 0;
int tests_failed = 0;
const char* current_test_name = NULL;
const char* current_suite_name = NULL;

int main(void) {
    run_tracker_tests();
    print_test_summary();
    return get_test_exit_code();
}
#endif
