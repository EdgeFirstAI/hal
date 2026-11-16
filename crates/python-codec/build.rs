// SPDX-FileCopyrightText: Copyright 2025 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

fn main() {
    pyo3_build_config::use_pyo3_cfgs();

    // Emit the Python extension-module link arguments.
    edgefirst_build_support::python_extension();

    // `_codec` calls into libedgefirst_tensor, which the edgefirst-tensor wheel ships in the sibling
    // `edgefirst/tensor/` directory (on Windows, `__init__.py` registers it with os.add_dll_directory).
    edgefirst_build_support::link_python_extension_to_tensor(
        edgefirst_build_support::Rpath::Sibling("tensor"),
    );
}
