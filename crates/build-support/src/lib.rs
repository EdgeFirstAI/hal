// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! Build-script helpers for the shared-library leaves.
//!
//! The C libraries (`image`, `codec`, `decoder`) and the tensor-linked Python
//! extensions do not embed the tensor implementation: they are built with
//! `--cfg edgefirst_tensor_dynamic` (see `.cargo/tensor-dynamic.toml`) and
//! call into `libedgefirst_tensor`, which `edgefirst-tensor-capi` builds from
//! the static backend. This crate holds the build-script half of that
//! arrangement in one place:
//!
//! - [`tensor_dynamic`] / [`require_dynamic_tensor`]: read and enforce the backend cfg.
//! - [`link_capi_to_tensor`] / [`tensor_lib_dir`]: a C leaf links the producer built beside it.
//! - [`link_python_extension_to_tensor`] / [`build_tensor_producer`]: a Python leaf builds the
//!   producer in a nested cargo invocation and links it.
//! - [`link_shared_tensor`]: link `libedgefirst_tensor` and set the rpath.
//! - [`soname`], [`python_extension`], [`coverage_instrumented`]: per-leaf link boilerplate.
//!
//! Everything that inspects the environment is a thin wrapper over a pure
//! function, so the decisions are unit-tested without a cargo build.

use std::env;
use std::ffi::OsString;
use std::path::{Path, PathBuf};
use std::process::Command;

/// The cfg that selects `edgefirst-tensor`'s dynamic backend.
pub const DYNAMIC_CFG: &str = "edgefirst_tensor_dynamic";

/// Opt-in for a static build of a Python leaf, used only by the G13 differential gate.
pub const STATIC_COMPARISON_ENV: &str = "EDGEFIRST_TENSOR_STATIC_COMPARISON";

/// Overrides the directory a C leaf links `libedgefirst_tensor` from.
pub const TENSOR_LIB_DIR_ENV: &str = "EDGEFIRST_TENSOR_LIB_DIR";

/// Whether this build script's package is being compiled with the dynamic tensor backend.
pub fn tensor_dynamic() -> bool {
    env::var_os("CARGO_CFG_EDGEFIRST_TENSOR_DYNAMIC").is_some()
}

/// Enforces the dynamic backend for a leaf that links `libedgefirst_tensor`.
///
/// Returns `true` when the cfg is set. A static build fails the build unless
/// [`STATIC_COMPARISON_ENV`] is `1`, in which case this returns `false` and
/// the caller must not link the shared library.
pub fn require_dynamic_tensor(package: &str) -> bool {
    println!("cargo::rerun-if-env-changed={STATIC_COMPARISON_ENV}");
    let opted_out = env::var(STATIC_COMPARISON_ENV).as_deref() == Ok("1");
    match backend_decision(tensor_dynamic(), opted_out) {
        Ok(dynamic) => dynamic,
        Err(()) => panic!(
            "{package} links libedgefirst_tensor and must be built with the dynamic tensor \
             backend: pass `--config .cargo/tensor-dynamic.toml` to cargo/maturin (or use \
             `make wheel`). Set {STATIC_COMPARISON_ENV}=1 only for the G13 static comparison build."
        ),
    }
}

fn backend_decision(dynamic: bool, static_opt_in: bool) -> Result<bool, ()> {
    match (dynamic, static_opt_in) {
        (true, _) => Ok(true),
        (false, true) => Ok(false),
        (false, false) => Err(()),
    }
}

/// Directory a C leaf links `libedgefirst_tensor` from.
///
/// [`TENSOR_LIB_DIR_ENV`] wins when set; otherwise this is the profile
/// directory of the current build (`<target-dir>[/<triple>]/<profile>`),
/// where `make capi-libs` builds `tensor-capi` beforehand. Derived from
/// `OUT_DIR` so custom target dirs, cross targets and custom profiles all
/// resolve correctly.
pub fn tensor_lib_dir() -> PathBuf {
    println!("cargo::rerun-if-env-changed={TENSOR_LIB_DIR_ENV}");
    if let Some(dir) = env::var_os(TENSOR_LIB_DIR_ENV).filter(|d| !d.is_empty()) {
        return PathBuf::from(dir);
    }
    let out_dir = env::var_os("OUT_DIR").expect("OUT_DIR is set for build scripts");
    profile_dir_from_out_dir(Path::new(&out_dir))
}

/// Links a C library leaf against the `libedgefirst_tensor` in [`tensor_lib_dir`].
pub fn link_capi_to_tensor() {
    link_shared_tensor(&tensor_lib_dir(), Rpath::SameDir);
}

/// Links a tensor-linked Python extension against a nested build of `libedgefirst_tensor`.
///
/// Returns the directory holding the built library, or `None` for the G13 static
/// comparison build (see [`require_dynamic_tensor`]).
pub fn link_python_extension_to_tensor(rpath: Rpath) -> Option<PathBuf> {
    let package = env::var("CARGO_PKG_NAME").expect("CARGO_PKG_NAME is set for build scripts");
    if !require_dynamic_tensor(&package) {
        return None;
    }
    let workspace = crates_dir()
        .parent()
        .expect("crates/ sits in the workspace root")
        .to_path_buf();
    let lib_dir = build_tensor_producer(&workspace.join("target/python-tensor-capi"));
    link_shared_tensor(&lib_dir, rpath);
    Some(lib_dir)
}

/// Lets a PyO3 extension cdylib leave CPython symbols for the interpreter to resolve on macOS.
pub fn python_extension() {
    if env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("macos") {
        println!("cargo:rustc-link-arg=-Wl,-undefined,dynamic_lookup");
    }
}

/// Sets an ELF cdylib's DT_SONAME to `lib<name>.so.<major>`.
///
/// Only the major version: dependents copy the SONAME into DT_NEEDED, so MINOR or PATCH
/// would force a re-link on every release.
pub fn soname(name: &str) {
    if env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("linux") {
        let major = env::var("CARGO_PKG_VERSION_MAJOR").expect("CARGO_PKG_VERSION_MAJOR is set");
        println!("cargo:rustc-cdylib-link-arg=-Wl,-soname,lib{name}.so.{major}");
    }
}

fn crates_dir() -> PathBuf {
    let manifest_dir =
        PathBuf::from(env::var_os("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR is set"));
    manifest_dir
        .parent()
        .expect("leaf crates live under crates/")
        .to_path_buf()
}

fn profile_dir_from_out_dir(out_dir: &Path) -> PathBuf {
    // OUT_DIR is always `<profile-dir>/build/<pkg>-<hash>/out`.
    out_dir
        .ancestors()
        .nth(3)
        .expect("OUT_DIR is nested three levels under the profile directory")
        .to_path_buf()
}

/// Where the leaf sits relative to the installed `libedgefirst_tensor`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Rpath {
    /// Same directory, as for the C libraries and `edgefirst.tensor`'s extension.
    SameDir,
    /// A sibling directory of the leaf's own, as for `edgefirst/image` reaching `edgefirst/tensor`.
    Sibling(&'static str),
}

impl Rpath {
    fn origin_relative(self, target_os: &str) -> Option<String> {
        let origin = match target_os {
            "macos" | "ios" => "@loader_path",
            "windows" => return None,
            _ => "$ORIGIN",
        };
        Some(match self {
            Rpath::SameDir => origin.to_string(),
            Rpath::Sibling(dir) => format!("{origin}/../{dir}"),
        })
    }
}

/// Links the current cdylib against the shared `libedgefirst_tensor` in `lib_dir` and sets its rpath.
pub fn link_shared_tensor(lib_dir: &Path, rpath: Rpath) {
    let target_os = env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    println!("cargo:rustc-link-search=native={}", lib_dir.display());
    match target_os.as_str() {
        "windows" => {
            // A plain `dylib=edgefirst_tensor` resolves the Rust staticlib cargo writes next to the
            // DLL, duplicating std at link time (LNK2005); name the DLL's import library verbatim.
            let import_dir = windows_import_lib_dir(lib_dir);
            if import_dir != lib_dir {
                println!("cargo:rustc-link-search=native={}", import_dir.display());
            }
            println!("cargo:rustc-link-lib=dylib:+verbatim=edgefirst_tensor.dll.lib");
        }
        "linux" => {
            // Trailing link args land after the crate's rlibs, and --no-as-needed keeps the
            // DT_NEEDED even when the binary's own objects never name an ef_tensor_* symbol
            // (aarch64 bfd otherwise drops it).
            println!("cargo:rustc-link-arg=-Wl,--no-as-needed");
            println!("cargo:rustc-link-arg=-ledgefirst_tensor");
            println!("cargo:rustc-link-arg=-Wl,--as-needed");
        }
        _ => println!("cargo:rustc-link-lib=dylib=edgefirst_tensor"),
    }
    // RUNPATH is not transitive, so each leaf carries its own.
    if let Some(path) = rpath.origin_relative(&target_os) {
        println!("cargo:rustc-cdylib-link-arg=-Wl,-rpath,{path}");
    }
}

fn windows_import_lib_dir(lib_dir: &Path) -> PathBuf {
    // Cargo writes the import library under deps/ and may also uplift it next to the DLL.
    let candidates = [lib_dir.to_path_buf(), lib_dir.join("deps")];
    candidates
        .iter()
        .find(|dir| dir.join("edgefirst_tensor.dll.lib").is_file())
        .cloned()
        .unwrap_or_else(|| {
            panic!(
                "expected edgefirst_tensor.dll.lib in {} or {}; build edgefirst-tensor-capi first",
                candidates[0].display(),
                candidates[1].display()
            )
        })
}

/// Whether this build is instrumented for coverage, through rustflags or cargo-llvm-cov's rustc wrapper.
pub fn coverage_instrumented() -> bool {
    println!("cargo:rerun-if-env-changed=__CARGO_LLVM_COV_RUSTC_WRAPPER_RUSTFLAGS");
    let encoded = env::var("CARGO_ENCODED_RUSTFLAGS").unwrap_or_default();
    let wrapper = env::var("__CARGO_LLVM_COV_RUSTC_WRAPPER_RUSTFLAGS").unwrap_or_default();
    flags_instrument_coverage(&encoded) || flags_instrument_coverage(&wrapper)
}

fn flags_instrument_coverage(flags: &str) -> bool {
    let mut codegen = false;
    flags
        .split(|c: char| c == '\x1f' || c.is_whitespace())
        .filter(|arg| !arg.is_empty())
        .any(|arg| {
            let option = match arg.strip_prefix("-C") {
                Some("") => {
                    codegen = true;
                    return false;
                }
                Some(rest) => rest,
                None if std::mem::take(&mut codegen) => arg,
                None => return false,
            };
            codegen = false;
            let (name, value) = option.split_once('=').unwrap_or((option, "yes"));
            name.replace('_', "-") == "instrument-coverage"
                && !matches!(value, "no" | "n" | "off" | "false")
        })
}

/// Builds `edgefirst-tensor-capi` (the static producer of `libedgefirst_tensor`) in a nested
/// cargo invocation for a Python leaf, and returns the directory the library landed in.
///
/// `nested_target_dir` must differ from the outer build's target dir, whose lock the outer
/// cargo holds. The nested build receives the outer rustflags minus the dynamic cfg, so
/// instrumentation reaches the producer while the producer stays static.
pub fn build_tensor_producer(nested_target_dir: &Path) -> PathBuf {
    let crates_dir = crates_dir();
    let producer_manifest = crates_dir.join("tensor-capi/Cargo.toml");
    for watched in [
        "tensor-capi/Cargo.toml",
        "tensor-capi/Cargo.lock",
        "tensor-capi/build.rs",
        "tensor-capi/exports.map",
        "tensor-capi/exports.syms",
        "tensor-capi/src",
        "tensor/Cargo.toml",
        "tensor/build.rs",
        "tensor/src",
        "tensor-abi",
        "tensor-ffi",
    ] {
        println!(
            "cargo:rerun-if-changed={}",
            crates_dir.join(watched).display()
        );
    }

    let profile = env::var("PROFILE").unwrap_or_else(|_| "debug".to_string());
    let target = env::var("TARGET").unwrap_or_default();
    let host = env::var("HOST").unwrap_or_default();
    let cross = !target.is_empty() && target != host;
    let encoded = env::var("CARGO_ENCODED_RUSTFLAGS").unwrap_or_default();

    let mut cmd = Command::new(env::var_os("CARGO").unwrap_or_else(|| OsString::from("cargo")));
    cmd.arg("build")
        .arg("--locked")
        .arg("--manifest-path")
        .arg(&producer_manifest)
        .arg("--target-dir")
        .arg(nested_target_dir)
        .env("CARGO_ENCODED_RUSTFLAGS", strip_cfg(&encoded, DYNAMIC_CFG))
        .env_remove("RUSTC_WORKSPACE_WRAPPER");
    if profile == "release" {
        cmd.arg("--release");
    }
    if cross {
        cmd.arg("--target").arg(&target);
    }
    if coverage_instrumented() {
        // tensor-capi's exports map collides with coverage's own version script.
        cmd.env("EF_SKIP_VERSION_SCRIPT", "1");
    }
    let status = cmd
        .status()
        .unwrap_or_else(|e| panic!("failed to spawn the nested tensor-capi build: {e}"));
    assert!(
        status.success(),
        "nested `cargo build --manifest-path {}` failed ({status}); libedgefirst_tensor could not be built",
        producer_manifest.display()
    );

    let mut built = nested_target_dir.to_path_buf();
    if cross {
        built.push(&target);
    }
    built.join(profile)
}

/// Removes every `--cfg <name>` pair (and `--cfg=<name>`) from a `CARGO_ENCODED_RUSTFLAGS` value.
pub fn strip_cfg(encoded: &str, name: &str) -> String {
    if encoded.is_empty() {
        return String::new();
    }
    let mut kept: Vec<&str> = Vec::new();
    let mut args = encoded.split('\x1f').peekable();
    while let Some(arg) = args.next() {
        if arg == "--cfg" && args.peek() == Some(&name) {
            args.next();
            continue;
        }
        if arg.strip_prefix("--cfg=") == Some(name) {
            continue;
        }
        kept.push(arg);
    }
    kept.join("\x1f")
}

/// Built and shipped file names of `libedgefirst_tensor` for a target OS.
///
/// The shipped name carries the SONAME's major version on ELF and Mach-O; Windows ships the DLL as built.
pub fn shared_tensor_file_names(target_os: &str, major: &str) -> (String, String) {
    match target_os {
        "windows" => ("edgefirst_tensor.dll".into(), "edgefirst_tensor.dll".into()),
        "macos" | "ios" => (
            "libedgefirst_tensor.dylib".into(),
            format!("libedgefirst_tensor.{major}.dylib"),
        ),
        _ => (
            "libedgefirst_tensor.so".into(),
            format!("libedgefirst_tensor.so.{major}"),
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn backend_decision_rules() {
        assert_eq!(backend_decision(true, false), Ok(true));
        assert_eq!(backend_decision(true, true), Ok(true));
        assert_eq!(backend_decision(false, true), Ok(false));
        assert_eq!(backend_decision(false, false), Err(()));
    }

    #[test]
    fn coverage_flag_spellings() {
        assert!(flags_instrument_coverage("-Cinstrument-coverage"));
        assert!(flags_instrument_coverage("-C\x1finstrument-coverage"));
        assert!(flags_instrument_coverage(
            "-C instrument_coverage=all --cfg=coverage"
        ));
        assert!(!flags_instrument_coverage("-Cinstrument-coverage=off"));
        assert!(!flags_instrument_coverage("--cfg\x1finstrument-coverage"));
        assert!(!flags_instrument_coverage(""));
    }

    #[test]
    fn strip_cfg_removes_only_the_named_cfg() {
        let flags = [
            "-Cinstrument-coverage",
            "--cfg",
            "edgefirst_tensor_dynamic",
            "--cfg",
            "coverage",
            "--cfg=edgefirst_tensor_dynamic",
            "-Cstrip=none",
        ]
        .join("\x1f");
        assert_eq!(
            strip_cfg(&flags, DYNAMIC_CFG),
            ["-Cinstrument-coverage", "--cfg", "coverage", "-Cstrip=none"].join("\x1f")
        );
    }

    #[test]
    fn strip_cfg_keeps_lookalikes_and_handles_edges() {
        assert_eq!(strip_cfg("", DYNAMIC_CFG), "");
        assert_eq!(
            strip_cfg("--cfg\x1fedgefirst_tensor_dynamic", DYNAMIC_CFG),
            ""
        );
        let lookalike = "--cfg\x1fedgefirst_tensor_dynamic_extra";
        assert_eq!(strip_cfg(lookalike, DYNAMIC_CFG), lookalike);
        assert_eq!(strip_cfg("--cfg", DYNAMIC_CFG), "--cfg");
    }

    #[test]
    fn profile_dir_handles_native_and_cross_layouts() {
        assert_eq!(
            profile_dir_from_out_dir(Path::new("/w/target/debug/build/x-1/out")),
            Path::new("/w/target/debug")
        );
        assert_eq!(
            profile_dir_from_out_dir(Path::new(
                "/w/target/aarch64-unknown-linux-gnu/profiling/build/x-1/out"
            )),
            Path::new("/w/target/aarch64-unknown-linux-gnu/profiling")
        );
    }

    #[test]
    fn rpath_per_platform() {
        assert_eq!(
            Rpath::SameDir.origin_relative("linux").as_deref(),
            Some("$ORIGIN")
        );
        assert_eq!(
            Rpath::Sibling("tensor").origin_relative("macos").as_deref(),
            Some("@loader_path/../tensor")
        );
        assert_eq!(
            Rpath::Sibling("tensor")
                .origin_relative("android")
                .as_deref(),
            Some("$ORIGIN/../tensor")
        );
        assert_eq!(Rpath::SameDir.origin_relative("windows"), None);
    }

    #[test]
    fn file_names_per_platform() {
        assert_eq!(
            shared_tensor_file_names("linux", "0"),
            (
                "libedgefirst_tensor.so".into(),
                "libedgefirst_tensor.so.0".into()
            )
        );
        assert_eq!(
            shared_tensor_file_names("macos", "1").1,
            "libedgefirst_tensor.1.dylib"
        );
        assert_eq!(
            shared_tensor_file_names("windows", "0").1,
            "edgefirst_tensor.dll"
        );
    }
}
