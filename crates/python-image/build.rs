// SPDX-FileCopyrightText: Copyright 2025 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

use std::env;
use std::path::{Path, PathBuf};
use std::process::Command;

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

    // Coverage-capture resilience: detect -Cinstrument-coverage and set
    // the `coverage` cfg so the SIGABRT handler ctor is compiled in.
    println!("cargo::rustc-check-cfg=cfg(coverage)");
    let rustflags = std::env::var("CARGO_ENCODED_RUSTFLAGS").unwrap_or_default();
    if rustflags.contains("instrument-coverage") {
        println!("cargo::rustc-cfg=coverage");
    }
    println!("cargo::rerun-if-env-changed=CARGO_ENCODED_RUSTFLAGS");

    let target_os = env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();

    // PyO3's `extension-module` normally emits this, but it does not reach a
    // cdylib whose PyO3 surface arrives via an rlib dependency. Extension
    // modules resolve CPython symbols from the interpreter at load time, so
    // the undefined references are expected and must be allowed through.
    if target_os == "macos" {
        println!("cargo:rustc-link-arg=-Wl,-undefined,dynamic_lookup");
    }

    // Only when "dynamic" is active -- under "static" (G13's comparison
    // build, see this crate's own Cargo.toml) this crate embeds
    // edgefirst-tensor's implementation directly, and there is nothing
    // external to link against. Cargo sets CARGO_FEATURE_<NAME>=1 for a
    // crate's own active features, visible to its build script.
    if env::var_os("CARGO_FEATURE_DYNAMIC").is_some() {
        link_tensor(&target_os);
    }

    if target_os == "windows" {
        bundle_angle(&PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap()));
    }
}

/// Windows: bundle ANGLE (GLES over Direct3D 11) into the wheel.
///
/// The HAL's Windows GL backend looks for `libEGL.dll` next to the module
/// that loaded it (`_image.pyd`) before anything else, so copying ANGLE's two
/// DLLs into `python/edgefirst/image/` -- which maturin's `python-source`
/// packages verbatim -- gives `pip install edgefirst-image` the GPU backend
/// with no environment setup. The DLLs come from `EDGEFIRST_ANGLE_PATH`, the
/// directory `scripts/fetch-angle.sh --windows` produces
/// (`target/angle/windows-x64/bin`); CI sets it for the Windows wheel rows.
/// Unset: the wheel ships CPU-only and the loader falls back to
/// `EDGEFIRST_ANGLE_PATH` / next-to-the-executable / the DLL search path at
/// runtime, exactly as before. A stale bundled copy from an earlier build is
/// removed in that case so the wheel content follows the environment.
fn bundle_angle(crate_dir: &Path) {
    println!("cargo:rerun-if-env-changed=EDGEFIRST_ANGLE_PATH");
    let dest_dir = crate_dir.join("python/edgefirst/image");
    const DLLS: [&str; 2] = ["libEGL.dll", "libGLESv2.dll"];
    let src_dir = env::var_os("EDGEFIRST_ANGLE_PATH")
        .filter(|p| !p.is_empty())
        .map(PathBuf::from);
    match src_dir {
        Some(src_dir) if DLLS.iter().all(|d| src_dir.join(d).is_file()) => {
            for dll in DLLS {
                let src = src_dir.join(dll);
                println!("cargo:rerun-if-changed={}", src.display());
                let dest = dest_dir.join(dll);
                std::fs::copy(&src, &dest).unwrap_or_else(|e| {
                    panic!(
                        "failed to copy {} to {}: {e}",
                        src.display(),
                        dest.display()
                    )
                });
            }
            println!(
                "cargo:warning=bundling ANGLE (libEGL.dll, libGLESv2.dll) from {} into the edgefirst-image wheel",
                src_dir.display()
            );
        }
        Some(src_dir) => {
            println!(
                "cargo:warning=EDGEFIRST_ANGLE_PATH={} does not contain libEGL.dll + libGLESv2.dll; the wheel ships without ANGLE",
                src_dir.display()
            );
            for dll in DLLS {
                let _ = std::fs::remove_file(dest_dir.join(dll));
            }
        }
        None => {
            for dll in DLLS {
                let _ = std::fs::remove_file(dest_dir.join(dll));
            }
        }
    }
}

/// Single-tensor-home, Python side (task P2): `_image` links
/// `libedgefirst_tensor.so` (via python-common's own `ef_tensor_*` FFI
/// calls) instead of embedding `edgefirst-tensor`'s implementation.
/// Unlike `crates/python-tensor/build.rs`, this crate does NOT bundle a
/// copy into its own wheel -- `crates/python-tensor`'s wheel is the sole
/// carrier of `libedgefirst_tensor.so`; this one only needs to find it at
/// load time via RPATH `$ORIGIN/../tensor`, reaching across the shared
/// `edgefirst/` namespace package the way the four C leaves' `.so` files
/// reach `libedgefirst_tensor.so` in the same directory as themselves.
/// (Windows: the library is `edgefirst_tensor.dll` and there is no rpath;
/// the cross-directory step happens at import time instead -- see below.)
fn link_tensor(target_os: &str) {
    let crate_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let tensor_capi_manifest = crate_dir.join("../tensor-capi/Cargo.toml");
    println!("cargo:rerun-if-changed={}", tensor_capi_manifest.display());
    println!(
        "cargo:rerun-if-changed={}",
        crate_dir.join("../tensor-capi/src").display()
    );
    println!(
        "cargo:rerun-if-changed={}",
        crate_dir.join("../tensor/src").display()
    );
    println!(
        "cargo:rerun-if-changed={}",
        crate_dir.join("../tensor/Cargo.toml").display()
    );

    // Same dedicated target-dir crates/python-tensor/build.rs uses, and
    // deliberately the SAME one (not a private one of this crate's own):
    // a `cargo build` against an up-to-date target-dir is a fast no-op, so
    // building all four Python extensions in one session compiles
    // edgefirst-tensor into this directory once, not four times. Still not
    // the shared root `target/` the C leaves use -- this build script runs
    // inside an outer cargo/maturin invocation already holding a lock on
    // that directory; see crates/python-tensor/build.rs's own comment for
    // why a nested build needs a different one.
    let profile = env::var("PROFILE").unwrap_or_else(|_| "debug".to_string());
    let target = env::var("TARGET").unwrap_or_default();
    let host = env::var("HOST").unwrap_or_default();
    let nested_target_dir = crate_dir.join("../../target/python-tensor-capi");

    let mut cmd = Command::new(env::var("CARGO").unwrap_or_else(|_| "cargo".to_string()));
    cmd.arg("build")
        .arg("--manifest-path")
        .arg(&tensor_capi_manifest)
        .arg("--target-dir")
        .arg(&nested_target_dir);
    if profile == "release" {
        cmd.arg("--release");
    }
    if !target.is_empty() && target != host {
        cmd.arg("--target").arg(&target);
    }
    let status = cmd
        .status()
        .unwrap_or_else(|e| panic!("failed to spawn `cargo build` for tensor-capi: {e}"));
    if !status.success() {
        panic!(
            "`cargo build --manifest-path {}` failed (status {status}) -- the shared tensor library could not be built",
            tensor_capi_manifest.display()
        );
    }

    let mut built_dir = nested_target_dir;
    if !target.is_empty() && target != host {
        built_dir = built_dir.join(&target);
    }
    let built_dir = built_dir.join(&profile);

    println!("cargo:rustc-link-search=native={}", built_dir.display());
    if target_os == "windows" {
        // MSVC: a plain `dylib=edgefirst_tensor` resolves
        // `edgefirst_tensor.lib`, the Rust staticlib cargo writes next to
        // the DLL, and linking that into this cdylib duplicates rust std
        // (LNK2005). The DLL's import library is `edgefirst_tensor.dll.lib`,
        // so name that file verbatim -- same pattern as
        // crates/python-tensor/build.rs and crates/image-capi/build.rs.
        let import_lib_dir = windows_import_lib_dir(&built_dir);
        if import_lib_dir != built_dir {
            println!(
                "cargo:rustc-link-search=native={}",
                import_lib_dir.display()
            );
        }
        println!("cargo:rustc-link-lib=dylib:+verbatim=edgefirst_tensor.dll.lib");
    } else {
        println!("cargo:rustc-link-lib=dylib=edgefirst_tensor");
    }

    // `$ORIGIN/../tensor` (Linux) / `@loader_path/../tensor` (macOS):
    // unlike the four C leaves (siblings in the same directory), this
    // extension and `_tensor.cpython-*.so` install into DIFFERENT
    // directories within the shared `edgefirst/` namespace package
    // (`edgefirst/image/_image...` vs. `edgefirst/tensor/_tensor...`),
    // so the relative step has to cross into the sibling package
    // directory rather than stay in the same one. RUNPATH is not
    // transitive, so this extension needs its own rpath regardless of
    // what `edgefirst-python-common` (an rlib, no rpath of its own) or
    // anything else in the process might carry.
    //
    // Windows has no rpath. Python 3.8+ loads extension modules with
    // `LoadLibraryExW(LOAD_LIBRARY_SEARCH_DEFAULT_DIRS |
    // LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR)`: the `.pyd`'s own directory plus
    // whatever `os.add_dll_directory()` registered, never `PATH`. `_image.pyd`
    // lives in `edgefirst/image/` while `edgefirst_tensor.dll` ships in
    // `edgefirst/tensor/`, so the cross-directory step happens at import
    // time instead: `python/edgefirst/image/__init__.py` registers
    // `edgefirst/tensor/` before importing `._image`.
    if target_os == "macos" {
        println!("cargo:rustc-cdylib-link-arg=-Wl,-rpath,@loader_path/../tensor");
    } else if target_os != "windows" {
        println!("cargo:rustc-cdylib-link-arg=-Wl,-rpath,$ORIGIN/../tensor");
    }
}

/// Directory holding `edgefirst_tensor.dll.lib`, the MSVC import library the
/// nested tensor-capi build produced alongside `edgefirst_tensor.dll`. Cargo
/// writes it in `deps/` and, depending on version, also uplifts a copy next
/// to the DLL in `built_dir` itself; take whichever exists. Windows only.
fn windows_import_lib_dir(built_dir: &Path) -> PathBuf {
    let candidates = [built_dir.to_path_buf(), built_dir.join("deps")];
    candidates
        .iter()
        .find(|dir| dir.join("edgefirst_tensor.dll.lib").is_file())
        .cloned()
        .unwrap_or_else(|| {
            panic!(
                "expected edgefirst_tensor.dll.lib in {} or {} after building tensor-capi, but it is in neither",
                candidates[0].display(),
                candidates[1].display()
            )
        })
}
