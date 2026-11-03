// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

fn main() {
    pyo3_build_config::use_pyo3_cfgs();
    // Kernels does not link libedgefirst_tensor: the plans operate on borrowed numpy buffers.
    edgefirst_build_support::python_extension();
}
