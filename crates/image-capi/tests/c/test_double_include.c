// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0
//
// The header must survive being included twice. cbindgen's `include_guard`
// wraps only the generated body, so anything in header/trailer -- which is
// where hand-written declarations must live, since cbindgen cannot emit a
// `static inline` body -- would be redefined on the second include.

#include "edgefirst/image.h"
#include "edgefirst/image.h"

int main(void) { return ef_image_abi_version() == 0; }
