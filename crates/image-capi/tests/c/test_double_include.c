// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0
//
// Includes the header twice.

#include "edgefirst/image.h"
#include "edgefirst/image.h"

int main(void) { return ef_image_abi_version() == 0; }
