// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

fn main() {
    pyo3_build_config::use_pyo3_cfgs();
    // Tracker does not link libedgefirst_tensor: ByteTrack consumes plain DetectBox values.
    edgefirst_build_support::python_extension();
}
