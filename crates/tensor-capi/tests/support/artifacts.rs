// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0
//! Finding, freshness-checking and reading the built C libraries.
//!
//! Test support only, `#[path]`-included by each modular C-API leaf's test
//! module. `env!` expands during the including crate's compilation, so
//! `CARGO_MANIFEST_DIR` here is that leaf's own directory.

use std::path::{Path, PathBuf};

/// Report a skipped check by writing `WARNING: SKIPPED: {reason}` directly
/// to stderr.
pub fn skip(reason: &str) {
    use std::io::Write;
    let _ = writeln!(std::io::stderr(), "WARNING: SKIPPED: {reason}");
}

/// Forwards each `SKIP: ...` line a C test program printed as a [`skip`] warning.
pub fn relay_skips(test: &str, out: &std::process::Output) {
    for stream in [&out.stdout, &out.stderr] {
        for line in String::from_utf8_lossy(stream).lines() {
            if let Some(why) = line.trim().strip_prefix("SKIP:") {
                skip(&format!("{test} - {}", why.trim()));
            }
        }
    }
}

/// Directories that may hold this leaf's built C libraries, in the order
/// they are searched: the profile directory of this test binary,
/// `CARGO_TARGET_DIR/debug`, the workspace `target/debug`, and the leaf's
/// own `target/debug`.
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

/// Checks that `artifact` is at least as new as every file in `src_dir`.
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
/// A stale copy is passed over, and reported only when no candidate is
/// fresh. A stale-only result returns `Err`, or panics when
/// `EF_REQUIRE_FRESH_ARTIFACTS` is set.
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

/// File names starting with `prefix` in every candidate directory.
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
    // -g: external symbols only. -U/--defined-only: skip undefined ones.
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
        return Err(format!("nm listed no symbol in {}", lib.display()));
    }
    Ok(syms)
}

/// Every symbol `lib` exports, read with `dumpbin /exports`.
#[cfg(windows)]
pub fn symbols_of(lib: &Path) -> Result<Vec<String>, String> {
    super::msvc::toolchain()?.exports(lib)
}
