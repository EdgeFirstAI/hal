// SPDX-FileCopyrightText: Copyright 2025 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

fn main() {
    pyo3_build_config::use_pyo3_cfgs();

    // Declare the custom cfg to avoid warnings
    println!("cargo::rustc-check-cfg=cfg(nightly)");

    // Detect if we're using nightly Rust
    let is_nightly = rustc_version::version_meta()
        .map(|meta| meta.channel == rustc_version::Channel::Nightly)
        .unwrap_or(false);

    if is_nightly {
        println!("cargo:rustc-cfg=nightly");
    }
}
