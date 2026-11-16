// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0
//
// The header compiles when included twice.

#include "edgefirst/tensor.h"
#include "edgefirst/tensor.h"

int main(void) { return ef_tensor_abi_version() == 0; }
