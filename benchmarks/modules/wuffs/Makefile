# SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
# SPDX-License-Identifier: Apache-2.0
#
# Wuffs JPEG decode benchmark.
#
#   make                  host build             -> build/wuffs_bench
#   make aarch64          cross build for boards -> build/wuffs_bench.aarch64
#   make x86_64           cross build (zig)      -> build/wuffs_bench.x86_64
#
# The 3.6 MB transpiled wuffs-v0.4.c is not vendored: `make deps` (run
# automatically) fetches it at a pinned release tag and verifies its sha256,
# keeping it out of git while keeping the build reproducible. Wuffs is
# Apache-2.0.

CC      ?= cc
CFLAGS  ?= -O2 -g -Wall -Wextra -std=c11
UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Darwin)
LDLIBS  :=
else
LDLIBS  := -lm
endif
BUILD   := build

INCLUDES := -I../cbench -I$(BUILD)

# Wuffs v0.4.0-alpha.10 (the v0.4 line is the first with the JPEG decoder).
WUFFS_URL    := https://raw.githubusercontent.com/google/wuffs/v0.4.0-alpha.10/release/c/wuffs-v0.4.c
WUFFS_SHA256 := 1f8039ef82911604c063f6ac2ed57254bdb17d742aebdeae06356530d4a0fde7

SHA256_CHECK := $(shell command -v sha256sum >/dev/null 2>&1 && echo "sha256sum -c -" || echo "shasum -a 256 -c -")

ifeq ($(shell command -v aarch64-linux-gnu-gcc 2>/dev/null),)
AARCH64_CC ?= zig cc -target aarch64-linux-gnu
AARCH64_LDLIBS := -lm
else
AARCH64_CC ?= aarch64-linux-gnu-gcc
AARCH64_LDLIBS := -lm -static-libgcc
endif
X86_64_CC ?= zig cc -target x86_64-linux-gnu

.PHONY: all aarch64 x86_64 deps clean
all: $(BUILD)/wuffs_bench

$(BUILD):
	@mkdir -p $(BUILD)

deps: $(BUILD)/wuffs-v0.4.c

$(BUILD)/wuffs-v0.4.c: | $(BUILD)
	curl -fsSL -o $@.tmp $(WUFFS_URL)
	echo "$(WUFFS_SHA256)  $@.tmp" | $(SHA256_CHECK)
	mv $@.tmp $@

$(BUILD)/wuffs_bench: bench.c ../cbench/cbench.h $(BUILD)/wuffs-v0.4.c | $(BUILD)
	$(CC) $(CFLAGS) $(INCLUDES) -o $@ bench.c $(LDLIBS)

aarch64: $(BUILD)/wuffs_bench.aarch64

$(BUILD)/wuffs_bench.aarch64: bench.c ../cbench/cbench.h $(BUILD)/wuffs-v0.4.c | $(BUILD)
	$(AARCH64_CC) $(CFLAGS) $(INCLUDES) -o $@ bench.c $(AARCH64_LDLIBS)

x86_64: $(BUILD)/wuffs_bench.x86_64

$(BUILD)/wuffs_bench.x86_64: bench.c ../cbench/cbench.h $(BUILD)/wuffs-v0.4.c | $(BUILD)
	$(X86_64_CC) $(CFLAGS) $(INCLUDES) -o $@ bench.c -lm

clean:
	rm -rf $(BUILD)
