// SPDX-FileCopyrightText: Copyright 2025 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0
//
// Main test runner for EdgeFirst HAL C API
//
// Combines all module tests into a single executable

#include <stddef.h>  // For NULL

// Define test result tracking variables (extern declared in test_common.h)
int tests_run = 0;
int tests_passed = 0;
int tests_failed = 0;
const char* current_test_name = NULL;
const char* current_suite_name = NULL;

#include "test_common.h"

// External test suite functions
extern void run_tensor_tests(void);
extern void run_image_tests(void);
extern void run_decoder_tests(void);
extern void run_tracker_tests(void);

int main(int argc, char* argv[]) {
    printf("EdgeFirst HAL C API Test Suite\n");
    printf("================================\n");

    // Check for specific test suite selection
    int run_all = (argc == 1);
    int run_tensor = run_all;
    int run_image = run_all;
    int run_decoder = run_all;
    int run_tracker = run_all;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--tensor") == 0 || strcmp(argv[i], "-t") == 0) {
            run_tensor = 1;
        } else if (strcmp(argv[i], "--image") == 0 || strcmp(argv[i], "-i") == 0) {
            run_image = 1;
        } else if (strcmp(argv[i], "--decoder") == 0 || strcmp(argv[i], "-d") == 0) {
            run_decoder = 1;
        } else if (strcmp(argv[i], "--tracker") == 0 || strcmp(argv[i], "-r") == 0) {
            run_tracker = 1;
        } else if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            printf("\nUsage: %s [OPTIONS]\n\n", argv[0]);
            printf("Options:\n");
            printf("  -t, --tensor   Run tensor tests only\n");
            printf("  -i, --image    Run image tests only\n");
            printf("  -d, --decoder  Run decoder tests only\n");
            printf("  -r, --tracker  Run tracker tests only\n");
            printf("  -h, --help     Show this help message\n");
            printf("\nWith no options, all test suites are run.\n");
            return 0;
        } else {
            fprintf(stderr, "Unknown option: %s\n", argv[i]);
            return 1;
        }
    }

    // Run selected test suites
    if (run_tensor) {
        run_tensor_tests();
    }
    if (run_image) {
        run_image_tests();
    }
    if (run_decoder) {
        run_decoder_tests();
    }
    if (run_tracker) {
        run_tracker_tests();
    }

    // Print summary and return exit code
    print_test_summary();
    return get_test_exit_code();
}
