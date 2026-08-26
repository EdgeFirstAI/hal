// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

use std::env;
use std::path::{Path, PathBuf};

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
        .write_to_file(output_dir.join("codec.h"));

    // DT_SONAME carries only the major version, matching the convention used
    // by glibc/OpenSSL/zlib: it is copied verbatim into every dependent's
    // DT_NEEDED, so embedding MINOR or PATCH would force downstream re-links
    // on every release and defeat ABI versioning.
    let target_os = env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    if target_os == "linux" {
        let major = env::var("CARGO_PKG_VERSION_MAJOR").unwrap();
        println!("cargo:rustc-cdylib-link-arg=-Wl,-soname,libedgefirst_codec.so.{major}");
    }

    // This library calls into libedgefirst_tensor.so rather than embedding a copy
    // of the tensor implementation. The undefined ef_tensor_* symbols are
    // resolved at load time, which is the whole point of the split.
    //
    // The search path is derived from PROFILE (not OUT_DIR): OUT_DIR is this
    // build script's own scratch directory under
    // target/<profile>/build/<pkg>-<hash>/out, which is not where cargo
    // writes libedgefirst_tensor.so. `../../target/<profile>`, relative to
    // this crate's manifest dir, is where crates/tensor-capi lands when
    // every -capi leaf is built with the shared `--target-dir target` from
    // the workspace root (see `make capi-libs`/`capi-libs-release`) --
    // target/debug for a debug build, target/release for a release build.
    //
    // Cross-compiling (TARGET != HOST, e.g. scripts/on-target-test.sh's
    // aarch64 builds) changes where that landed: cargo nests cross output
    // under target/<TARGET-triple>/<profile>, never bare target/<profile>
    // (that path stays the HOST's own native build). Found the hard way --
    // task 12's first on-target cross-build linked against the
    // wrong-architecture libedgefirst_tensor.so already sitting in
    // target/debug from an unrelated local build, and failed with
    // "incompatible with aarch64linux", not a missing-file error, which is
    // what made it non-obvious.
    let profile = env::var("PROFILE").unwrap_or_else(|_| "debug".to_string());
    let target = env::var("TARGET").unwrap_or_default();
    let host = env::var("HOST").unwrap_or_default();
    let mut tensor_lib_dir = PathBuf::from(&crate_dir).join("../../target");
    if !target.is_empty() && target != host {
        tensor_lib_dir = tensor_lib_dir.join(&target);
    }
    let tensor_lib_dir = tensor_lib_dir.join(&profile);
    println!(
        "cargo:rustc-link-search=native={}",
        tensor_lib_dir.display()
    );
    link_tensor_cdylib(&target_os, &tensor_lib_dir);

    // A consumer of THIS library that never itself calls an ef_tensor_*
    // function directly (the common case: a caller of ef_codec_* functions
    // has no reason to) links this .so without also linking
    // libedgefirst_tensor.so, and the default `--as-needed` linker behavior
    // then drops even an explicit `-ledgefirst_tensor` from the consumer's
    // own executable when nothing in it references that library's symbols.
    // RUNPATH is not transitive, so the consumer's own rpath cannot resolve
    // THIS library's undefined ef_tensor_* references either -- this
    // library must carry its own rpath. `$ORIGIN`/`@loader_path` (the
    // directory this .so itself is loaded from) is correct because every
    // shipped release places all five libraries side by side; a build-time
    // absolute path would be wrong for an installed copy.
    if target_os == "linux" {
        println!("cargo:rustc-cdylib-link-arg=-Wl,-rpath,$ORIGIN");
    } else if target_os == "macos" {
        println!("cargo:rustc-cdylib-link-arg=-Wl,-rpath,@loader_path");
    }

    // The header is regenerated only when a source file or the cbindgen
    // config changes -- NOT when the header itself is edited or deleted. Any
    // test that reads the header must therefore depend on the source, or it
    // is validating a stale artifact. (Learned the hard way in Plan 2.)
    println!("cargo:rerun-if-changed=src/");
    println!("cargo:rerun-if-changed=cbindgen.toml");
}

/// Link `libedgefirst_tensor` as a DLL/so/dylib, never as the Rust staticlib.
///
/// On MSVC, `dylib=edgefirst_tensor` still resolves `edgefirst_tensor.lib`,
/// which is the staticlib cargo writes next to the DLL. Linking that
/// staticlib into this Rust cdylib duplicates rust std (LNK2005:
/// `rust_panic`, alloc hooks, …). The import library is
/// `edgefirst_tensor.dll.lib`.
fn link_tensor_cdylib(target_os: &str, tensor_lib_dir: &Path) {
    if target_os == "windows" {
        println!(
            "cargo:rustc-link-search=native={}",
            tensor_lib_dir.join("deps").display()
        );
        println!("cargo:rustc-link-lib=dylib:+verbatim=edgefirst_tensor.dll.lib");
    } else {
        println!("cargo:rustc-link-lib=dylib=edgefirst_tensor");
    }
}
