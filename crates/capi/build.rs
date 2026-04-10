// SPDX-FileCopyrightText: Copyright 2025 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

use std::env;
use std::path::PathBuf;

fn main() {
    let crate_dir = env::var("CARGO_MANIFEST_DIR").unwrap();
    let output_dir = PathBuf::from(&crate_dir).join("include").join("edgefirst");

    // Generate the C header using cbindgen
    let config = cbindgen::Config::from_file(PathBuf::from(&crate_dir).join("cbindgen.toml"))
        .expect("Unable to find cbindgen.toml configuration file");

    cbindgen::Builder::new()
        .with_crate(&crate_dir)
        .with_config(config)
        .generate()
        .expect("Unable to generate C bindings")
        .write_to_file(output_dir.join("hal.h"));

    // Set DT_SONAME for the shared library on Linux.
    //
    // Convention: DT_SONAME contains only the major version (e.g.
    // `libedgefirst_hal.so.0`), not the full crate version. This is the
    // standard GNU/Linux convention used by glibc, OpenSSL, zlib, etc.
    //
    // Rationale: DT_SONAME is copied verbatim into every dependent binary's
    // DT_NEEDED entry. Embedding MINOR or PATCH there would force every
    // downstream consumer to re-link on every release, defeating ABI
    // versioning. The dynamic linker resolves DT_NEEDED by looking for a
    // file whose name matches that string, which is why the release
    // pipeline ships the chain
    //   libedgefirst_hal.so
    //     → libedgefirst_hal.so.MAJOR
    //     → libedgefirst_hal.so.MAJOR.MINOR
    //     → libedgefirst_hal.so.MAJOR.MINOR.PATCH  (the real file)
    //
    // Bump the crate MAJOR version only when the C ABI breaks (removed or
    // changed symbols, struct layout changes, etc.); that will propagate to
    // DT_SONAME automatically via CARGO_PKG_VERSION_MAJOR.
    let target_os = env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    if target_os == "linux" {
        let major = env::var("CARGO_PKG_VERSION_MAJOR").unwrap();
        println!("cargo:rustc-cdylib-link-arg=-Wl,-soname,libedgefirst_hal.so.{major}");
    }

    // Tell cargo to re-run if source files change
    println!("cargo:rerun-if-changed=src/");
    println!("cargo:rerun-if-changed=cbindgen.toml");
}
