// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0
//
// Includes codec.h twice.

#include "edgefirst/codec.h"
#include "edgefirst/codec.h"

int main(void) { return ef_codec_abi_version() == 0; }
