// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

use std::env;
use std::path::PathBuf;

fn main() {
    let crate_dir = env::var("CARGO_MANIFEST_DIR").unwrap();
    let output_dir = PathBuf::from(&crate_dir).join("include").join("edgefirst");

    let config = cbindgen::Config::from_file(PathBuf::from(&crate_dir).join("cbindgen.toml"))
        .expect("Unable to find cbindgen.toml configuration file");

    cbindgen::Builder::new()
        .with_crate(&crate_dir)
        .with_config(config)
        .generate()
        .expect("Unable to generate C bindings")
        .write_to_file(output_dir.join("tracker.h"));

    // DT_SONAME carries only the major version, matching the convention used
    // by glibc/OpenSSL/zlib: it is copied verbatim into every dependent's
    // DT_NEEDED, so embedding MINOR or PATCH would force downstream re-links
    // on every release and defeat ABI versioning.
    let target_os = env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    if target_os == "linux" {
        let major = env::var("CARGO_PKG_VERSION_MAJOR").unwrap();
        println!("cargo:rustc-cdylib-link-arg=-Wl,-soname,libedgefirst_tracker.so.{major}");
    }

    // Tracker does not link libedgefirst_tensor: detections cross as a
    // plain C array. No DT_NEEDED, no rpath on tensor.

    // The header is regenerated only when a source file or the cbindgen
    // config changes -- NOT when the header itself is edited or deleted. Any
    // test that reads the header must therefore depend on the source, or it
    // is validating a stale artifact.
    println!("cargo:rerun-if-changed=src/");
    println!("cargo:rerun-if-changed=cbindgen.toml");
}
