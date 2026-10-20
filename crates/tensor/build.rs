// SPDX-FileCopyrightText: Copyright 2025 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

// Under the dynamic backend this crate's own test binaries need libedgefirst_tensor, which
// `edgefirst-tensor-ffi` declares without a `#[link]`; `-tests` link args never reach dependents.
// Mirrors build-support's `tensor_lib_dir`, which this published crate cannot depend on.
fn main() {
    let rustflags = std::env::var("CARGO_ENCODED_RUSTFLAGS").unwrap_or_default();
    let wrapper = std::env::var("__CARGO_LLVM_COV_RUSTC_WRAPPER_RUSTFLAGS").unwrap_or_default();
    println!("cargo::rerun-if-env-changed=__CARGO_LLVM_COV_RUSTC_WRAPPER_RUSTFLAGS");
    if rustflags.contains("instrument-coverage") || wrapper.contains("instrument-coverage") {
        println!("cargo::rustc-cfg=coverage");
    }

    if std::env::var_os("CARGO_CFG_EDGEFIRST_TENSOR_DYNAMIC").is_none() {
        return;
    }
    println!("cargo::rerun-if-env-changed=EDGEFIRST_TENSOR_LIB_DIR");
    let lib_dir = match std::env::var_os("EDGEFIRST_TENSOR_LIB_DIR").filter(|d| !d.is_empty()) {
        Some(dir) => std::path::PathBuf::from(dir),
        None => {
            let out_dir = std::env::var("OUT_DIR").expect("OUT_DIR is set for build scripts");
            // OUT_DIR is `<profile-dir>/build/<pkg>-<hash>/out`; tensor-capi's library sits in <profile-dir>.
            match std::path::Path::new(&out_dir).ancestors().nth(3) {
                Some(dir) => dir.to_path_buf(),
                None => return,
            }
        }
    };
    if std::env::var("CARGO_CFG_TARGET_ENV").as_deref() == Ok("msvc") {
        println!("cargo::rustc-link-arg-tests=/LIBPATH:{}", lib_dir.display());
        println!("cargo::rustc-link-arg-tests=edgefirst_tensor.dll.lib");
    } else {
        println!("cargo::rustc-link-arg-tests=-L{}", lib_dir.display());
        println!("cargo::rustc-link-arg-tests=-ledgefirst_tensor");
    }
}
