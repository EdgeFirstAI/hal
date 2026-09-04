// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0
//! Finding, freshness-checking and reading the built C libraries.
//!
//! The five modular C-API leaves are workspace-excluded standalone packages,
//! so where cargo puts their artifacts depends on how they were built, and
//! every leaf's tests need the same three answers: which directory holds the
//! library, is it newer than the sources it is supposed to prove something
//! about, and what does it export. This module is the one place those live.
//!
//! Test support only, `#[path]`-included by each leaf's test module. `env!`
//! expands during the *including* crate's compilation, so
//! `CARGO_MANIFEST_DIR` here is that leaf's own directory; nothing needs to
//! be passed in.

use std::path::{Path, PathBuf};

/// Report a skipped check on stderr.
///
/// Straight to the handle, not `eprintln!`: libtest captures a passing
/// test's `eprintln!`, and a skip nobody sees is indistinguishable from a
/// pass.
pub fn skip(reason: &str) {
    use std::io::Write;
    let _ = writeln!(std::io::stderr(), "SKIP: {reason}");
}

/// Directories that may hold this leaf's built C libraries, in the order
/// they are searched.
///
/// The target directory depends on how the leaf was built:
/// `CARGO_TARGET_DIR`, an explicit `--target-dir` (what the Makefile and CI
/// pass), or the leaf's own `crates/<leaf>/target`. This test binary itself
/// sits in `<target dir>/<profile>/deps`, so its own path names the
/// directory and profile cargo actually used; the rest are fallbacks for a
/// `cargo build` and a `cargo test` that were given different ones.
pub fn artifact_dirs() -> Vec<PathBuf> {
    let mut dirs: Vec<PathBuf> = Vec::new();
    let mut add = |dir: PathBuf| {
        if !dirs.contains(&dir) {
            dirs.push(dir);
        }
    };
    if let Ok(exe) = std::env::current_exe() {
        if let Some(profile) = exe.parent().and_then(|deps| deps.parent()) {
            add(profile.to_path_buf());
        }
    }
    if let Some(target) = std::env::var_os("CARGO_TARGET_DIR") {
        add(PathBuf::from(target).join("debug"));
    }
    let mut workspace = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    workspace.pop();
    workspace.pop();
    add(workspace.join("target").join("debug"));
    add(PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("target")
        .join("debug"));
    dirs
}

/// The platform's file name for the shared library `stem` builds to.
pub fn cdylib_name(stem: &str) -> String {
    #[cfg(target_os = "macos")]
    {
        format!("lib{stem}.dylib")
    }
    #[cfg(all(not(windows), not(target_os = "macos")))]
    {
        format!("lib{stem}.so")
    }
    #[cfg(windows)]
    {
        format!("{stem}.dll")
    }
}

/// Is `artifact` newer than everything in `src_dir`?
///
/// Every check that reads a compiled artifact needs this, because neither
/// `cargo test` nor `cargo test --all-targets` builds the `staticlib` /
/// `cdylib` targets -- only `cargo build` does. A check that skips it
/// silently validates a library built *before* the change under test, which
/// is exactly how the "no accessor is exported" check was found to be
/// vacuous.
pub fn freshness(artifact: &Path, src_dir: &Path) -> Result<(), String> {
    let built = std::fs::metadata(artifact)
        .and_then(|m| m.modified())
        .map_err(|e| format!("{}: {e}", artifact.display()))?;
    let newest_src = std::fs::read_dir(src_dir)
        .ok()
        .and_then(|rd| {
            rd.filter_map(|e| e.ok())
                .filter_map(|e| e.metadata().ok())
                .filter_map(|m| m.modified().ok())
                .max()
        })
        .ok_or_else(|| format!("nothing readable in {}", src_dir.display()))?;
    if built >= newest_src {
        return Ok(());
    }
    Err(format!(
        "{} is older than {} -- this check did NOT run. `cargo test` does \
         not build staticlib/cdylib targets; `cargo build` the owning leaf \
         first",
        artifact.display(),
        src_dir.display()
    ))
}

/// The first directory in [`artifact_dirs`] holding a `name` newer than
/// `src_dir`.
///
/// A stale hit is passed over rather than returned: a leaf built into the
/// workspace `target/` and a `cargo test` run with a different target
/// directory both leave a copy behind, and the older one proves nothing. A
/// stale copy is only reported when no candidate is fresh, so the reason a
/// caller prints names the real problem.
///
/// Default is a **loud skip**, not a panic: panicking would make plain
/// `cargo test` fail for everyone after any edit, and a clearly-reported
/// skip does not carry the failure mode being guarded -- nobody reads "NOT
/// CHECKED" as a pass. Set `EF_REQUIRE_FRESH_ARTIFACTS=1` where build order
/// is guaranteed (the Makefile's capi lane and CI) to turn a stale-only
/// result into a hard failure.
pub fn find_artifact(name: &str, src_dir: &Path) -> Result<PathBuf, String> {
    let dirs = artifact_dirs();
    let mut stale = Vec::new();
    for dir in &dirs {
        let path = dir.join(name);
        if !path.exists() {
            continue;
        }
        match freshness(&path, src_dir) {
            Ok(()) => return Ok(path),
            Err(reason) => stale.push(reason),
        }
    }
    if stale.is_empty() {
        return Err(format!(
            "{name} is in none of {}; run `cargo build --manifest-path \
             crates/<leaf>-capi/Cargo.toml --target-dir target` for the leaf \
             that builds it",
            dirs.iter()
                .map(|d| d.display().to_string())
                .collect::<Vec<_>>()
                .join(", ")
        ));
    }
    let reason = stale.join("; ");
    if std::env::var("EF_REQUIRE_FRESH_ARTIFACTS").is_ok() {
        panic!("{reason}");
    }
    Err(reason)
}

/// [`find_artifact`] for the shared library `stem` builds to.
pub fn find_cdylib(stem: &str, src_dir: &Path) -> Result<PathBuf, String> {
    find_artifact(&cdylib_name(stem), src_dir)
}

/// File names starting with `prefix` in **every** candidate directory.
///
/// The `_capi` check reads this: a stale `libedgefirst_tensor_capi.so` in a
/// target directory this run did not build into is exactly the file an
/// install or link step can still pick up, so scanning only the directory
/// that happened to hold the fresh library would miss it.
pub fn artifacts_named(prefix: &str) -> Vec<String> {
    let mut found: Vec<String> = Vec::new();
    for dir in artifact_dirs() {
        let Ok(entries) = std::fs::read_dir(&dir) else {
            continue;
        };
        for entry in entries.filter_map(|e| e.ok()) {
            let name = entry.file_name().to_string_lossy().to_string();
            if name.starts_with(prefix) && !found.contains(&name) {
                found.push(name);
            }
        }
    }
    found
}

/// Every symbol `lib` exports, read with `nm`.
#[cfg(not(windows))]
pub fn symbols_of(lib: &Path) -> Result<Vec<String>, String> {
    // -g: external symbols only. -U/--defined-only: skip undefined ones,
    // which would otherwise include every libc symbol we merely call.
    let args: &[&str] = if cfg!(target_os = "macos") {
        &["-gU"]
    } else {
        &["-D", "--defined-only"]
    };
    let out = std::process::Command::new("nm")
        .args(args)
        .arg(lib)
        .output()
        .map_err(|e| format!("nm: {e}"))?;
    if !out.status.success() {
        return Err(format!(
            "nm {} exited {}:\n{}",
            lib.display(),
            out.status,
            String::from_utf8_lossy(&out.stderr)
        ));
    }
    let syms: Vec<String> = String::from_utf8_lossy(&out.stdout)
        .lines()
        .filter_map(|l| l.split_whitespace().last())
        .map(|s| s.trim_start_matches('_').to_string())
        .collect();
    if syms.is_empty() {
        // An empty list would make every "X is absent" assertion pass.
        return Err(format!("nm listed no symbol in {}", lib.display()));
    }
    Ok(syms)
}

/// Every symbol `lib` exports, read with `dumpbin /exports`.
///
/// `super::msvc` is the sibling support module each leaf includes beside
/// this one, by the same `#[path]`.
#[cfg(windows)]
pub fn symbols_of(lib: &Path) -> Result<Vec<String>, String> {
    super::msvc::toolchain()?.exports(lib)
}
