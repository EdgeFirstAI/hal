// SPDX-FileCopyrightText: Copyright 2025 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

use std::env;
use std::fs;
use std::path::{Path, PathBuf};

fn main() {
    pyo3_build_config::use_pyo3_cfgs();

    // Declare the custom cfg.
    println!("cargo::rustc-check-cfg=cfg(nightly)");

    // Detect a nightly toolchain.
    let is_nightly = rustc_version::version_meta()
        .map(|meta| meta.channel == rustc_version::Channel::Nightly)
        .unwrap_or(false);

    if is_nightly {
        println!("cargo:rustc-cfg=nightly");
    }

    let target_os = env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    // Emit the Python extension-module link arguments.
    edgefirst_build_support::python_extension();

    // Link `_tensor` to libedgefirst_tensor and bundle the library next to it.
    if let Some(built_dir) = edgefirst_build_support::link_python_extension_to_tensor(
        edgefirst_build_support::Rpath::SameDir,
    ) {
        bundle_tensor(&built_dir, &target_os);
    } else {
        // A static build removes any bundled libedgefirst_tensor from the python-source tree.
        let crate_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
        if let Ok(entries) = fs::read_dir(crate_dir.join("python/edgefirst/tensor")) {
            for entry in entries.flatten() {
                let name = entry.file_name();
                let name = name.to_string_lossy();
                if name.starts_with("libedgefirst_tensor.") || name == "edgefirst_tensor.dll" {
                    let _ = fs::remove_file(entry.path());
                }
            }
        }
    }
}

/// Copies the built libedgefirst_tensor into `python/edgefirst/tensor/`, which maturin's
/// `python-source` packages verbatim.
fn bundle_tensor(built_dir: &Path, target_os: &str) {
    let crate_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    // The SONAME major is the package's major version.
    let major = env::var("CARGO_PKG_VERSION_MAJOR").unwrap();
    let (built_name, shipped_name) =
        edgefirst_build_support::shared_tensor_file_names(target_os, &major);
    let built_lib = built_dir.join(&built_name);
    assert!(
        built_lib.is_file(),
        "expected {} after building tensor-capi",
        built_lib.display()
    );

    // Ship only the SONAME (or DLL) name.
    let dest = crate_dir
        .join("python/edgefirst/tensor")
        .join(&shipped_name);
    fs::copy(&built_lib, &dest).unwrap_or_else(|e| {
        panic!(
            "failed to copy {} to {}: {e}",
            built_lib.display(),
            dest.display()
        )
    });
}
