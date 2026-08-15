# SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
# SPDX-License-Identifier: Apache-2.0
#
# stb_image decode benchmark.
#
#   make                  host build             -> build/stb_bench
#   make aarch64          cross build for boards -> build/stb_bench.aarch64
#   make x86_64           cross build (zig)      -> build/stb_bench.x86_64
#
# stb_image.h is not vendored: `make deps` (run automatically) fetches it at a
# pinned commit and verifies its sha256, keeping the 280 KB header out of git
# while keeping the build reproducible. stb_image is public domain / MIT.

CC      ?= cc
CFLAGS  ?= -O2 -g -Wall -Wextra -std=c11
UNAME_S := $(shell uname -s)
# libdl is separate from libm on Linux (both live in libSystem on Darwin);
# cbench.h's --parity mode dlopens libturbojpeg.
ifeq ($(UNAME_S),Darwin)
LDLIBS  :=
else
LDLIBS  := -lm -ldl
endif
BUILD   := build

INCLUDES := -I../cbench -I$(BUILD)

# stb_image v2.30, pinned to the last commit touching stb_image.h (2024-07).
STB_URL    := https://raw.githubusercontent.com/nothings/stb/013ac3beddff3dbffafd5177e7972067cd2b5083/stb_image.h
STB_SHA256 := 594c2fe35d49488b4382dbfaec8f98366defca819d916ac95becf3e75f4200b3

SHA256_CHECK := $(shell command -v sha256sum >/dev/null 2>&1 && echo "sha256sum -c -" || echo "shasum -a 256 -c -")

ifeq ($(shell command -v aarch64-linux-gnu-gcc 2>/dev/null),)
AARCH64_CC ?= zig cc -target aarch64-linux-gnu
AARCH64_LDLIBS := -lm -ldl
else
AARCH64_CC ?= aarch64-linux-gnu-gcc
AARCH64_LDLIBS := -lm -ldl -static-libgcc
endif
X86_64_CC ?= zig cc -target x86_64-linux-gnu

.PHONY: all aarch64 x86_64 deps clean
all: $(BUILD)/stb_bench

$(BUILD):
	@mkdir -p $(BUILD)

deps: $(BUILD)/stb_image.h

$(BUILD)/stb_image.h: | $(BUILD)
	curl -fsSL -o $@.tmp $(STB_URL)
	echo "$(STB_SHA256)  $@.tmp" | $(SHA256_CHECK)
	mv $@.tmp $@

$(BUILD)/stb_bench: bench.c ../cbench/cbench.h $(BUILD)/stb_image.h | $(BUILD)
	$(CC) $(CFLAGS) $(INCLUDES) -o $@ bench.c $(LDLIBS)

aarch64: $(BUILD)/stb_bench.aarch64

$(BUILD)/stb_bench.aarch64: bench.c ../cbench/cbench.h $(BUILD)/stb_image.h | $(BUILD)
	$(AARCH64_CC) $(CFLAGS) $(INCLUDES) -o $@ bench.c $(AARCH64_LDLIBS)

x86_64: $(BUILD)/stb_bench.x86_64

$(BUILD)/stb_bench.x86_64: bench.c ../cbench/cbench.h $(BUILD)/stb_image.h | $(BUILD)
	$(X86_64_CC) $(CFLAGS) $(INCLUDES) -o $@ bench.c -lm -ldl

clean:
	rm -rf $(BUILD)
