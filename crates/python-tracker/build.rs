// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

use std::env;

fn main() {
    pyo3_build_config::use_pyo3_cfgs();

    println!("cargo::rustc-check-cfg=cfg(nightly)");
    let is_nightly = rustc_version::version_meta()
        .map(|meta| meta.channel == rustc_version::Channel::Nightly)
        .unwrap_or(false);
    if is_nightly {
        println!("cargo:rustc-cfg=nightly");
    }

    println!("cargo::rustc-check-cfg=cfg(coverage)");
    let rustflags = std::env::var("CARGO_ENCODED_RUSTFLAGS").unwrap_or_default();
    if rustflags.contains("instrument-coverage") {
        println!("cargo:rustc-cfg=coverage");
    }
    println!("cargo::rerun-if-env-changed=CARGO_ENCODED_RUSTFLAGS");

    // Tracker does not DT_NEEDED libedgefirst_tensor.so: ByteTrack consumes
    // plain DetectBox values. PyO3 still needs dynamic_lookup on macOS.
    if env::var("CARGO_CFG_TARGET_OS").unwrap_or_default() == "macos" {
        println!("cargo:rustc-link-arg=-Wl,-undefined,dynamic_lookup");
    }
}
