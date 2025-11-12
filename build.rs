// SPDX-FileCopyrightText: Copyright 2025 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

fn main() {
    #[cfg(feature = "python")]
    pyo3_build_config::use_pyo3_cfgs();
}
