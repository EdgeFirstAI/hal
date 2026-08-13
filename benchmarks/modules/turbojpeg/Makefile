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
UNAME_S := $(shell uname -s)
# libdl is a separate library on Linux; on Darwin dlopen lives in libSystem.
ifeq ($(UNAME_S),Darwin)
LDLIBS  :=
else
LDLIBS  := -ldl
endif
BUILD   := build

# Linux aarch64 boards. Prefer a GNU cross gcc when present; otherwise zig
# (same toolchain cargo-zigbuild uses). Always link libdl — the target is
# Linux even when this Makefile runs on Darwin.
ifeq ($(shell command -v aarch64-linux-gnu-gcc 2>/dev/null),)
AARCH64_CC ?= zig cc -target aarch64-linux-gnu
AARCH64_LDLIBS := -ldl
else
AARCH64_CC ?= aarch64-linux-gnu-gcc
AARCH64_LDLIBS := -ldl -static-libgcc
endif

.PHONY: all aarch64 clean
all: $(BUILD)/turbojpeg_bench

$(BUILD):
	@mkdir -p $(BUILD)

$(BUILD)/turbojpeg_bench: bench.c | $(BUILD)
	$(CC) $(CFLAGS) -o $@ $< $(LDLIBS)

aarch64: $(BUILD)/turbojpeg_bench.aarch64

$(BUILD)/turbojpeg_bench.aarch64: bench.c | $(BUILD)
	$(AARCH64_CC) $(CFLAGS) -o $@ $< $(AARCH64_LDLIBS)

clean:
	rm -rf $(BUILD)
