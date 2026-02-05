# SPDX-FileCopyrightText: Copyright 2025 Au-Zone Technologies
# SPDX-License-Identifier: Apache-2.0
#
# Makefile for EdgeFirst HAL C API Tests
#
# Usage:
#   make              - Build all tests (release)
#   make debug        - Build all tests (debug)
#   make test         - Build and run all tests
#   make test_tensor  - Build and run tensor tests only
#   make clean        - Clean build artifacts
#   make valgrind     - Run tests with valgrind (memory check)

# Directories
CAPI_DIR := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))..
HAL_ROOT := $(CAPI_DIR)/../..
TARGET_DIR := $(HAL_ROOT)/target
BUILD_DIR := $(CAPI_DIR)/build

# Library paths - prefer release, fallback to debug
RELEASE_LIB_DIR := $(TARGET_DIR)/release
DEBUG_LIB_DIR := $(TARGET_DIR)/debug

# Detect available library
ifeq ($(wildcard $(RELEASE_LIB_DIR)/libedgefirst_hal.so),)
  ifeq ($(wildcard $(RELEASE_LIB_DIR)/libedgefirst_hal.a),)
    LIB_DIR := $(DEBUG_LIB_DIR)
  else
    LIB_DIR := $(RELEASE_LIB_DIR)
  endif
else
  LIB_DIR := $(RELEASE_LIB_DIR)
endif

# Include path
INCLUDE_DIR := $(CAPI_DIR)/include

# Compiler settings
CC := gcc
CFLAGS := -Wall -Wextra -Werror -std=c11 -D_GNU_SOURCE
CFLAGS += -I$(INCLUDE_DIR)

# Debug/Release flags
ifdef DEBUG
  CFLAGS += -g -O0 -DDEBUG
else
  CFLAGS += -O2 -DNDEBUG
endif

# Platform detection
UNAME_S := $(shell uname -s)
UNAME_M := $(shell uname -m)

ifeq ($(UNAME_S),Linux)
  LDFLAGS := -L$(LIB_DIR) -Wl,-rpath,$(LIB_DIR)
  LIBS := -ledgefirst_hal -lm -lpthread -ldl
else ifeq ($(UNAME_S),Darwin)
  LDFLAGS := -L$(LIB_DIR)
  LIBS := -ledgefirst_hal -lm -lpthread
else
  $(error Unsupported platform: $(UNAME_S))
endif

# Source files
TEST_SRCS := test_tensor.c test_image.c test_decoder.c test_tracker.c
TEST_MAIN := test_all.c

# Object files
TEST_OBJS := $(patsubst %.c,$(BUILD_DIR)/%.o,$(TEST_SRCS))
MAIN_OBJ := $(BUILD_DIR)/test_all.o

# Executables
TEST_ALL := $(BUILD_DIR)/test_all

# Individual test executables (for standalone testing)
TEST_TENSOR := $(BUILD_DIR)/test_tensor
TEST_IMAGE := $(BUILD_DIR)/test_image
TEST_DECODER := $(BUILD_DIR)/test_decoder
TEST_TRACKER := $(BUILD_DIR)/test_tracker

# Default target
.PHONY: all
all: $(TEST_ALL)

# Debug build
.PHONY: debug
debug:
	$(MAKE) DEBUG=1 all

# Create build directory
$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

# Compile test source files
$(BUILD_DIR)/%.o: %.c test_common.h | $(BUILD_DIR)
	$(CC) $(CFLAGS) -c $< -o $@

# Compile standalone test objects (with define)
$(BUILD_DIR)/%_standalone.o: %.c test_common.h | $(BUILD_DIR)
	$(CC) $(CFLAGS) -DTEST_$(shell echo $* | tr '[:lower:]' '[:upper:]')_STANDALONE -c $< -o $@

# Link main test executable
$(TEST_ALL): $(MAIN_OBJ) $(TEST_OBJS)
	$(CC) $(CFLAGS) $^ -o $@ $(LDFLAGS) $(LIBS)

# Standalone test executables
$(TEST_TENSOR): $(BUILD_DIR)/test_tensor_standalone.o
	$(CC) $(CFLAGS) $^ -o $@ $(LDFLAGS) $(LIBS)

$(TEST_IMAGE): $(BUILD_DIR)/test_image_standalone.o
	$(CC) $(CFLAGS) $^ -o $@ $(LDFLAGS) $(LIBS)

$(TEST_DECODER): $(BUILD_DIR)/test_decoder_standalone.o
	$(CC) $(CFLAGS) $^ -o $@ $(LDFLAGS) $(LIBS)

$(TEST_TRACKER): $(BUILD_DIR)/test_tracker_standalone.o
	$(CC) $(CFLAGS) $^ -o $@ $(LDFLAGS) $(LIBS)

# Run all tests
.PHONY: test
test: $(TEST_ALL)
	@echo "Running EdgeFirst HAL C API tests..."
	@echo "Library: $(LIB_DIR)"
	@echo "Platform: $(UNAME_S) $(UNAME_M)"
	@echo ""
	$(TEST_ALL)

# Run individual test suites
.PHONY: test_tensor
test_tensor: $(TEST_TENSOR)
	$(TEST_TENSOR)

.PHONY: test_image
test_image: $(TEST_IMAGE)
	$(TEST_IMAGE)

.PHONY: test_decoder
test_decoder: $(TEST_DECODER)
	$(TEST_DECODER)

.PHONY: test_tracker
test_tracker: $(TEST_TRACKER)
	$(TEST_TRACKER)

# Memory check with valgrind
.PHONY: valgrind
valgrind: $(TEST_ALL)
	valgrind --leak-check=full --show-leak-kinds=all --track-origins=yes \
		--error-exitcode=1 $(TEST_ALL)

# Clean build artifacts
.PHONY: clean
clean:
	rm -rf $(BUILD_DIR)

# Print configuration (for debugging)
.PHONY: info
info:
	@echo "CAPI_DIR:    $(CAPI_DIR)"
	@echo "HAL_ROOT:    $(HAL_ROOT)"
	@echo "LIB_DIR:     $(LIB_DIR)"
	@echo "INCLUDE_DIR: $(INCLUDE_DIR)"
	@echo "CC:          $(CC)"
	@echo "CFLAGS:      $(CFLAGS)"
	@echo "LDFLAGS:     $(LDFLAGS)"
	@echo "LIBS:        $(LIBS)"
	@echo "UNAME_S:     $(UNAME_S)"
	@echo "UNAME_M:     $(UNAME_M)"

# Check that library exists
.PHONY: check-lib
check-lib:
	@if [ ! -f "$(LIB_DIR)/libedgefirst_hal.so" ] && [ ! -f "$(LIB_DIR)/libedgefirst_hal.a" ]; then \
		echo "Error: libedgefirst_hal not found in $(LIB_DIR)"; \
		echo "Run 'cargo build --release -p edgefirst-hal-capi' first"; \
		exit 1; \
	fi

# Ensure library is built before compiling tests
$(TEST_OBJS) $(MAIN_OBJ): | check-lib
