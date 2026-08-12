# SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
# SPDX-License-Identifier: Apache-2.0
#
# Native TurboJPEG decode benchmark.
#
#   make                  host build            -> build/turbojpeg_bench
#   make aarch64          cross build for boards -> build/turbojpeg_bench.aarch64
#
# libturbojpeg is dlopen'd at run time, so the cross build needs no sysroot.

CC      ?= cc
CFLAGS  ?= -O2 -g -Wall -Wextra -std=c11
LDLIBS  := -ldl
BUILD   := build

AARCH64_CC ?= aarch64-linux-gnu-gcc

.PHONY: all aarch64 clean
all: $(BUILD)/turbojpeg_bench

$(BUILD):
	@mkdir -p $(BUILD)

$(BUILD)/turbojpeg_bench: bench.c | $(BUILD)
	$(CC) $(CFLAGS) -o $@ $< $(LDLIBS)

aarch64: $(BUILD)/turbojpeg_bench.aarch64

$(BUILD)/turbojpeg_bench.aarch64: bench.c | $(BUILD)
	$(AARCH64_CC) $(CFLAGS) -static-libgcc -o $@ $< $(LDLIBS)

clean:
	rm -rf $(BUILD)
