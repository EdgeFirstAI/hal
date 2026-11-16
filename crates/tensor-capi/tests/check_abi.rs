// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! check-abi: every symbol `edgefirst-tensor-ffi` declares is an export of
//! the built libedgefirst_tensor, and every `ef_` export is declared.

/// Whether this build artifact is at least as new as the crate's sources.
///
/// A stale artifact prints a `SKIPPED` warning and returns `false`, or
/// panics when `EF_REQUIRE_FRESH_ARTIFACTS` is set.
fn artifact_is_fresh(artifact: &std::path::Path) -> bool {
    let art = match std::fs::metadata(artifact).and_then(|m| m.modified()) {
        Ok(t) => t,
        Err(_) => return false,
    };
    let src_dir = std::path::Path::new(concat!(env!("CARGO_MANIFEST_DIR"), "/src"));
    let newest_src = std::fs::read_dir(src_dir).ok().and_then(|rd| {
        rd.filter_map(Result::ok)
            .filter_map(|e| e.metadata().ok())
            .filter_map(|m| m.modified().ok())
            .max()
    });
    let Some(newest_src) = newest_src else {
        return false;
    };
    if art >= newest_src {
        return true;
    }
    let msg = format!(
        "{} is older than src/ -- this check did NOT run. \
         `cargo test` does not build staticlib/cdylib targets; run \
         `cargo build -p edgefirst-tensor-capi` first.",
        artifact.display()
    );
    if std::env::var("EF_REQUIRE_FRESH_ARTIFACTS").is_ok() {
        panic!("{msg}");
    }
    // Written directly to stderr.
    use std::io::Write;
    let _ = writeln!(std::io::stderr(), "WARNING: SKIPPED: {msg}");
    false
}

/// Symbols the built cdylib exports.
///
/// Returns `None` when the library is missing or stale, the platform's
/// symbol tool is unavailable, or it lists no symbol.
fn exported_symbols_fresh() -> Option<Vec<String>> {
    let mut dir = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    dir.pop();
    dir.pop();
    dir.push("target");
    dir.push("debug");
    let lib = ["libedgefirst_tensor.dylib", "libedgefirst_tensor.so"]
        .iter()
        .map(|n| dir.join(n))
        .find(|p| p.exists())?;

    if !artifact_is_fresh(&lib) {
        return None;
    }
    // -g: external symbols only. -U/--defined-only: skip undefined ones.
    let args: &[&str] = if cfg!(target_os = "macos") {
        &["-gU"]
    } else {
        &["-D", "--defined-only"]
    };
    let out = std::process::Command::new("nm")
        .args(args)
        .arg(&lib)
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    let text = String::from_utf8_lossy(&out.stdout);
    let syms: Vec<String> = text
        .lines()
        .filter_map(|l| l.split_whitespace().last())
        .map(|s| s.trim_start_matches('_').to_string())
        .collect();
    if syms.is_empty() {
        return None;
    }
    Some(syms)
}

#[test]
fn every_ffi_declaration_is_a_real_export() {
    let Some(symbols) = exported_symbols_fresh() else {
        return;
    };
    let missing: Vec<_> = edgefirst_tensor_ffi::DECLARED
        .iter()
        .filter(|name| !symbols.iter().any(|s| s == *name))
        .collect();
    assert!(
        missing.is_empty(),
        "declared in tensor-ffi but not exported by libedgefirst_tensor: {missing:?}"
    );
}

#[test]
fn every_exported_ef_symbol_is_declared() {
    let Some(symbols) = exported_symbols_fresh() else {
        return;
    };
    let undeclared: Vec<_> = symbols
        .iter()
        .filter(|s| s.starts_with("ef_"))
        .filter(|s| !edgefirst_tensor_ffi::DECLARED.iter().any(|d| d == *s))
        .collect();
    assert!(
        undeclared.is_empty(),
        "exported by libedgefirst_tensor but not declared in tensor-ffi: {undeclared:?}"
    );
}

#[test]
fn the_ffi_declaration_count_matches_the_header() {
    let header = std::fs::read_to_string(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/include/edgefirst/tensor.h"
    ))
    .expect("header exists");

    // A declaration starts on a non-indented line that names an `ef_`
    // symbol and opens a parameter list.
    let declared_in_header = header
        .lines()
        .filter(|l| {
            !l.starts_with(char::is_whitespace)
                && !l.starts_with("typedef")
                && l.starts_with(|c: char| c.is_ascii_alphabetic())
                && l.contains("ef_")
                && l.contains('(')
        })
        .count();

    assert_eq!(edgefirst_tensor_ffi::DECLARED.len(), declared_in_header);
}
