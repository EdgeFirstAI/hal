// SPDX-FileCopyrightText: Copyright 2025 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

use std::env;
use std::fs;
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

    // Only when "dynamic" is active (Cargo sets CARGO_FEATURE_<NAME>=1 for
    // a crate's own active features, visible to its build script). Under
    // "static" -- G13's comparison build, see this crate's own Cargo.toml
    // -- this crate embeds edgefirst-tensor's implementation directly, no
    // external .so to build, link, or bundle; running this unconditionally
    // would link an external libedgefirst_tensor.so INTO a binary that
    // also embeds the implementation, redundant at best, a symbol
    // collision at worst.
    if env::var_os("CARGO_FEATURE_DYNAMIC").is_some() {
        link_and_bundle_tensor(&target_os);
    } else {
        // Proven necessary, not assumed: a static build run in a tree
        // that previously built dynamic (same working directory, e.g.
        // switching --features locally) silently bundled the STALE
        // libedgefirst_tensor.so* (edgefirst_tensor.dll on Windows) left
        // behind by that earlier run into the "static" wheel -- maturin's
        // python-source packaging picks up whatever is sitting in this
        // directory, stale or not, same failure shape as the
        // aarch64-in-an-x86_64-wheel bug this task is also chasing. Must
        // actively remove, not just skip writing.
        let crate_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
        let py_tensor_dir = crate_dir.join("python/edgefirst/tensor");
        if let Ok(entries) = fs::read_dir(&py_tensor_dir) {
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

/// Single-tensor-home, Python side (task P2): `_tensor` links
/// `libedgefirst_tensor.so` (`libedgefirst_tensor.dylib` on macOS,
/// `edgefirst_tensor.dll` on Windows) instead of embedding
/// `edgefirst-tensor`'s implementation, and ships a copy of it in the wheel
/// so the extension has something to resolve at load time.
/// `crates/tensor-capi` is the crate that
/// produces that library; it is deliberately workspace-excluded (see the
/// root `Cargo.toml`'s `[workspace.exclude]` comment -- the static/dynamic
/// feature switch is mutually exclusive and cargo unifies features across
/// one invocation's package set), so it cannot be a normal path dependency
/// here. This function builds it via `--manifest-path`, the same way `make
/// capi-libs`/`capi-libs-release` does for the four C leaves.
fn link_and_bundle_tensor(target_os: &str) {
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

    // A DEDICATED target-dir, not the shared root `target/` the four C
    // leaves use (`make capi-libs`'s `--target-dir target`). This build
    // script itself runs as part of building `edgefirst-python-tensor`,
    // which -- unlike `tensor-capi` -- IS a normal member of the root
    // workspace and therefore already has an outer `cargo`/`maturin`
    // invocation holding that same `target/` directory. Pointing this
    // nested `cargo build` at a different `--target-dir` gives it its own
    // lock file, so there is no nested-lock contention with the build this
    // script is itself running inside of. The cost is a second, independent
    // compilation of `edgefirst-tensor` from the same source -- acceptable
    // here: this is a build-time artifact for wheel packaging, not the
    // "single .so on the machine" claim G6/G12 measure, which is about one
    // library within `site-packages`'s own `edgefirst/` namespace, not
    // across every build system that ever touches this source.
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
    // Cross-compiling (TARGET != HOST): a bare `--target-dir` build still
    // lands in `<dir>/<profile>` for the HOST unless `--target` is passed
    // explicitly -- found the hard way once already, in the C leaves'
    // build.rs (task 12), where the omission silently linked the wrong
    // architecture's .so instead of failing outright.
    if !target.is_empty() && target != host {
        cmd.arg("--target").arg(&target);
    }
    // Nested cargo strips CARGO_ENCODED_RUSTFLAGS. Tell tensor-capi's
    // build.rs to skip exports.map whenever this parent is itself under
    // coverage instrumentation, so llvm-cov's version-script is the only
    // one the linker sees.
    let encoded = env::var("CARGO_ENCODED_RUSTFLAGS").unwrap_or_default();
    let rustflags = env::var("RUSTFLAGS").unwrap_or_default();
    if encoded.contains("instrument-coverage") || rustflags.contains("instrument-coverage") {
        cmd.env("EF_SKIP_VERSION_SCRIPT", "1");
    }
    let status = cmd
        .status()
        .unwrap_or_else(|e| panic!("failed to spawn `cargo build` for tensor-capi: {e}"));
    if !status.success() {
        panic!("`cargo build --manifest-path {}` failed (status {status}) -- the shared tensor library could not be built", tensor_capi_manifest.display());
    }

    let mut built_dir = nested_target_dir.clone();
    if !target.is_empty() && target != host {
        built_dir = built_dir.join(&target);
    }
    let built_dir = built_dir.join(&profile);

    // Three naming schemes for the same artifact. Linux/macOS: cargo writes
    // the bare `libedgefirst_tensor.{so,dylib}` and the wheel ships it under
    // its SONAME/install_name, `libedgefirst_tensor.so.<major>` /
    // `libedgefirst_tensor.<major>.dylib` (the bundling comment below says
    // why ONLY that name). Windows: cargo writes `edgefirst_tensor.dll` --
    // no `lib` prefix -- and there is no SONAME concept at all: the import
    // library and every dependent's import table name the DLL by its bare
    // file name, so the shipped copy keeps that exact name and carries no
    // version suffix.
    let (built_name, shipped_name) = if target_os == "windows" {
        (
            "edgefirst_tensor.dll".to_string(),
            "edgefirst_tensor.dll".to_string(),
        )
    } else if target_os == "macos" {
        (
            "libedgefirst_tensor.dylib".to_string(),
            format!("libedgefirst_tensor.{}.dylib", tensor_lib_version_major()),
        )
    } else {
        (
            "libedgefirst_tensor.so".to_string(),
            format!("libedgefirst_tensor.so.{}", tensor_lib_version_major()),
        )
    };
    let built_lib = built_dir.join(&built_name);
    if !built_lib.is_file() {
        panic!(
            "expected {} after building tensor-capi, but it is not there",
            built_lib.display()
        );
    }

    // Link `_tensor` against it, exactly the way the four C leaves'
    // build.rs files link against this same library (see e.g.
    // crates/codec-capi/build.rs) -- `-L` the directory it landed in, `-l`
    // it by name.
    println!("cargo:rustc-link-search=native={}", built_dir.display());
    if target_os == "windows" {
        // MSVC: a plain `dylib=edgefirst_tensor` resolves
        // `edgefirst_tensor.lib`, which is the Rust STATICLIB cargo writes
        // next to the DLL (tensor-capi's crate-type is `["staticlib",
        // "cdylib"]`), and linking that into this cdylib duplicates rust
        // std (LNK2005: `rust_panic`, alloc hooks, ...). The DLL's import
        // library is `edgefirst_tensor.dll.lib`, so name that file
        // verbatim -- the same pattern crates/image-capi/build.rs's
        // `link_tensor_cdylib` uses. Cargo writes the import library in
        // `deps/` and may also uplift it beside the DLL; whichever is
        // present gets the `-L`.
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

    // Same non-transitivity reasoning as the C leaves: RUNPATH is not
    // transitive, so whatever eventually loads `_tensor.cpython-*.so` (the
    // Python interpreter's own dynamic loader, via `dlopen`) cannot resolve
    // `_tensor`'s own undefined `ef_tensor_*` references through some other
    // rpath -- `_tensor` must carry its own. `$ORIGIN`/`@loader_path` (the
    // directory `_tensor.cpython-*.so` itself is loaded from) is correct
    // because the library this function bundles below ships as its sibling
    // in the same wheel, in the same `edgefirst/tensor/` directory.
    //
    // Windows has no rpath, and needs none HERE: Python 3.8+ loads
    // extension modules with `LoadLibraryExW(LOAD_LIBRARY_SEARCH_DEFAULT_DIRS
    // | LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR)`, so `_tensor.pyd`'s import-table
    // entry for `edgefirst_tensor.dll` resolves from `_tensor.pyd`'s OWN
    // directory -- exactly where the copy bundled below lands. `PATH` is
    // deliberately NOT searched under those flags. The sibling extensions
    // in OTHER `edgefirst/*` directories cannot lean on that rule; their
    // `__init__.py` registers `edgefirst/tensor/` via
    // `os.add_dll_directory()` before importing their `.pyd` (see e.g.
    // crates/python-image/python/edgefirst/image/__init__.py).
    if target_os == "macos" {
        println!("cargo:rustc-cdylib-link-arg=-Wl,-rpath,@loader_path");
    } else if target_os != "windows" {
        println!("cargo:rustc-cdylib-link-arg=-Wl,-rpath,$ORIGIN");
    }

    // Bundle it into the wheel. maturin's `python-source = "python"`
    // (pyproject.toml) packages that whole directory tree verbatim into the
    // wheel alongside the compiled extension -- placing a copy here, next
    // to `__init__.py`, is what lands it at `edgefirst/tensor/` in the
    // built wheel, a sibling of `_tensor.cpython-*.so`. (`.gitignore`
    // excludes the generated file from the tracked source tree, the same
    // way it already excludes the compiled `_tensor.*.so` maturin itself
    // drops there.)
    //
    // Ship ONLY the versioned (SONAME) name, not also the bare
    // `libedgefirst_tensor.so` -- this was a real, measured mistake in an
    // earlier version of this function: it wrote the bare name as a real
    // file, then a `.so.<major>` SYMLINK pointing at it. maturin's
    // `python-source` packaging dereferences symlinks when it builds the
    // wheel, so that symlink became a second full copy in the actual
    // wheel, not a lightweight alias -- confirmed via `zipfile.ZipInfo`
    // (`is_symlink=False`, both entries byte-identical, ~1 MB doubled for
    // nothing). The bare name was never needed there in the first place:
    // `ldd`/the dynamic loader only ever resolves the SONAME embedded by
    // tensor-capi's own build.rs (`libedgefirst_tensor.so.<major>`, what
    // DT_NEEDED actually names) -- confirmed empirically on the installed
    // extension (`ldd` resolves `libedgefirst_tensor.so.0`, never the bare
    // name). Nothing in this wheel's own runtime path ever opens the bare
    // name; the two references to it elsewhere in this repo
    // (`crates/tensor-capi/src/lib.rs`, `.../tests/check_abi.rs`) are
    // tensor-capi's own C-side self-tests reading directly from the
    // shared `target/debug`, unrelated to this wheel. The bare name is
    // still needed at LINK time (`-ledgefirst_tensor` above), but that
    // reads from `built_dir` (this crate's own build directory), never
    // from this bundled copy.
    //
    // Windows: the shipped name IS the built name, `edgefirst_tensor.dll`
    // -- no SONAME, no version suffix -- and, as above, ONLY the DLL: the
    // `edgefirst_tensor.dll.lib` import library and the
    // `edgefirst_tensor.lib` staticlib cargo writes next to it are
    // link-time inputs read from `built_dir`, never opened at load time,
    // and bundling either would only ship dead weight.
    let py_tensor_dir = crate_dir.join("python/edgefirst/tensor");
    let dest_shipped = py_tensor_dir.join(&shipped_name);
    fs::copy(&built_lib, &dest_shipped).unwrap_or_else(|e| {
        panic!(
            "failed to copy {} to {}: {e}",
            built_lib.display(),
            dest_shipped.display()
        )
    });
}

/// Directory holding `edgefirst_tensor.dll.lib`, the MSVC import library the
/// nested tensor-capi build produced alongside `edgefirst_tensor.dll`. Cargo
/// writes it in `deps/` and, depending on version, also uplifts a copy next
/// to the DLL in `built_dir` itself; either location is fine for `-L`, so
/// take whichever exists rather than hardcoding one. Windows only.
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

/// `edgefirst-tensor`'s own major version, read from the workspace root's
/// `[workspace.package] version`, not from `crates/tensor/Cargo.toml`
/// directly -- that crate's own `version` field is `version.workspace =
/// true` (inherited), not a literal string. Not this crate's own
/// `CARGO_PKG_VERSION_MAJOR` either, which is `edgefirst-python-tensor`'s
/// version, a different package that happens to share the workspace
/// version today but has no reason to stay in lockstep. Not hardcoded,
/// which would silently go stale the day `edgefirst-tensor` crosses 1.0.
fn tensor_lib_version_major() -> String {
    let crate_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let manifest = crate_dir.join("../../Cargo.toml");
    println!("cargo:rerun-if-changed={}", manifest.display());
    let src = fs::read_to_string(&manifest)
        .unwrap_or_else(|e| panic!("failed to read {}: {e}", manifest.display()));
    // Scoped to the `[workspace.package]` table specifically: the root
    // manifest's own `[package]`/`[workspace.dependencies]` tables contain
    // other `version = "..."` lines (e.g. `edgefirst-codec`'s pinned
    // version) that are not this one.
    let mut in_workspace_package = false;
    for line in src.lines() {
        let trimmed = line.trim();
        if let Some(section) = trimmed.strip_prefix('[').and_then(|s| s.strip_suffix(']')) {
            in_workspace_package = section == "workspace.package";
            continue;
        }
        if in_workspace_package && trimmed.starts_with("version") {
            let version = trimmed
                .split('"')
                .nth(1)
                .unwrap_or_else(|| panic!("could not parse a quoted version out of: {trimmed}"));
            return version
                .split('.')
                .next()
                .unwrap_or_else(|| panic!("version {version} has no major component"))
                .to_string();
        }
    }
    panic!(
        "no `version = ...` line found under [workspace.package] in {}",
        manifest.display()
    );
}
