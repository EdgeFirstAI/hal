// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0
//
// cbindgen's own `include_guard` wraps only the generated body, so anything
// hand-written in header/trailer would be redefined on the second include.

#include "edgefirst/codec.h"
#include "edgefirst/codec.h"

int main(void) { return ef_codec_abi_version() == 0; }
