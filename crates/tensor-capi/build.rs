// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

use std::env;
use std::path::PathBuf;

fn coverage_instrumentation_active() -> bool {
    fn has_cov(s: &str) -> bool {
        s.contains("instrument-coverage")
    }
    env::var_os("EF_SKIP_VERSION_SCRIPT").is_some()
        || has_cov(&env::var("CARGO_ENCODED_RUSTFLAGS").unwrap_or_default())
        || has_cov(&env::var("RUSTFLAGS").unwrap_or_default())
}

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
        .write_to_file(output_dir.join("tensor.h"));

    // DT_SONAME carries only the major version, matching the convention used
    // by glibc/OpenSSL/zlib: it is copied verbatim into every dependent's
    // DT_NEEDED, so embedding MINOR or PATCH would force downstream re-links
    // on every release and defeat ABI versioning.
    let target_os = env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    if target_os == "linux" {
        let major = env::var("CARGO_PKG_VERSION_MAJOR").unwrap();
        println!("cargo:rustc-cdylib-link-arg=-Wl,-soname,libedgefirst_tensor.so.{major}");
        // Coverage instrumentation injects its own `--version-script=.../list`.
        // Combining that with `exports.map` is `ld: anonymous version tag
        // cannot be combined with other version tags` on aarch64.
        // Nested `cargo` from python-tensor's build.rs strips
        // CARGO_ENCODED_RUSTFLAGS but still inherits shell RUSTFLAGS from
        // cargo-llvm-cov show-env -- so both env vars (and an explicit
        // skip from the parent build.rs) must be consulted.
        if !coverage_instrumentation_active() {
            println!("cargo:rustc-cdylib-link-arg=-Wl,--version-script={crate_dir}/exports.map");
        }
        println!("cargo:rerun-if-env-changed=CARGO_ENCODED_RUSTFLAGS");
        println!("cargo:rerun-if-env-changed=RUSTFLAGS");
        println!("cargo:rerun-if-env-changed=EF_SKIP_VERSION_SCRIPT");
    } else if target_os == "macos" {
        let major = env::var("CARGO_PKG_VERSION_MAJOR").unwrap();
        println!(
            "cargo:rustc-cdylib-link-arg=-Wl,-install_name,@rpath/libedgefirst_tensor.{major}.dylib"
        );
        println!("cargo:rustc-cdylib-link-arg=-Wl,-exported_symbols_list,{crate_dir}/exports.syms");
    }
    println!("cargo:rerun-if-changed=exports.map");
    println!("cargo:rerun-if-changed=exports.syms");

    // The header is regenerated only when a source file or the cbindgen
    // config changes -- NOT when the header itself is edited or deleted. Any
    // test that reads the header must therefore depend on the source, or it
    // is validating a stale artifact. (Learned the hard way in Plan 2.)
    println!("cargo:rerun-if-changed=src/");
    println!("cargo:rerun-if-changed=cbindgen.toml");
}
