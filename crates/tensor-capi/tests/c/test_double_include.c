// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0
//
// The header must survive being included twice.
//
// This is not hypothetical bookkeeping: the vtable struct and every
// `static inline` accessor live in cbindgen's `trailer`, which sits OUTSIDE
// the guard cbindgen's own `include_guard` option emits. With that option the
// second include redefines all of them. Compiling this file is the gate.

#include "edgefirst/tensor.h"
#include "edgefirst/tensor.h"

int main(void) { return ef_tensor_abi_version() == 0; }
