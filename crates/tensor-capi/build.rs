// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

use std::env;
use std::path::PathBuf;
use std::process::Command;

fn rustflags() -> String {
    format!(
        "{} {}",
        env::var("CARGO_ENCODED_RUSTFLAGS").unwrap_or_default(),
        env::var("RUSTFLAGS").unwrap_or_default()
    )
}

fn skip_exports_map() -> bool {
    let flags = rustflags();
    // rustc 1.94 injects its own `--version-script=.../list` for Linux
    // cdylibs (`--no-undefined-version`). A second script is
    // `ld: anonymous version tag cannot be combined with other version tags`
    // on aarch64 bfd. Coverage rustflags carry the same collision.
    env::var_os("EF_SKIP_VERSION_SCRIPT").is_some()
        || flags.contains("instrument-coverage")
        || flags.contains("version-script")
}

fn rust_lld() -> Option<PathBuf> {
    let rustc = env::var("RUSTC").unwrap_or_else(|_| "rustc".to_string());
    let output = Command::new(rustc)
        .args(["--print", "sysroot"])
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    let sysroot = String::from_utf8(output.stdout).ok()?;
    let sysroot = sysroot.trim();
    let target = env::var("TARGET").ok()?;
    let lld = PathBuf::from(format!("{sysroot}/lib/rustlib/{target}/bin/gcc-ld/ld.lld"));
    lld.is_file().then_some(lld)
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
        // aarch64 GNU bfd rejects rustc's injected version script combined
        // with exports.map. lld concatenates them, which keeps the ef_*
        // export set G1 measures.
        if env::var("CARGO_CFG_TARGET_ARCH").ok().as_deref() == Some("aarch64") {
            if let Some(lld) = rust_lld() {
                println!("cargo:rustc-cdylib-link-arg=-fuse-ld={}", lld.display());
            } else {
                println!("cargo:rustc-cdylib-link-arg=-fuse-ld=lld");
            }
        }
        if !skip_exports_map() {
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
