// SPDX-FileCopyrightText: Copyright 2025 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

fn main() {
    pyo3_build_config::use_pyo3_cfgs();

    // PyO3's `extension-module` link args do not reach a cdylib whose PyO3 surface comes from an rlib.
    edgefirst_build_support::python_extension();

    // `_decoder` calls into libedgefirst_tensor, which the edgefirst-tensor wheel ships in the sibling
    // `edgefirst/tensor/` directory (on Windows, `__init__.py` registers it with os.add_dll_directory).
    edgefirst_build_support::link_python_extension_to_tensor(
        edgefirst_build_support::Rpath::Sibling("tensor"),
    );
}
