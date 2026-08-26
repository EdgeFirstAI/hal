// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! check-abi: every symbol `edgefirst-tensor-ffi` declares is a real export
//! of the built libedgefirst_tensor. A declaration without an export is a
//! load-time failure waiting in every consumer of the dynamic backend.

// The nm/freshness helpers below are copied from `src/lib.rs`'s `#[cfg(test)]`
// module rather than shared: integration tests are a separate compilation
// unit and cannot see `#[cfg(test)]` items from the library crate, and this
// crate has no non-test module to hold them without exposing test-only
// plumbing as part of the public API. Keep any behavioral fix in step with
// the other copy.

/// Is this build artifact newer than the crate's sources?
///
/// Every test here that reads a compiled artifact needs this check, because
/// neither `cargo test` nor `cargo test --all-targets` builds the
/// `staticlib`/`cdylib` targets — only `cargo build` does. A test that skips
/// it silently validates a library built *before* the change under test.
///
/// Default is a **loud skip**, not a panic: panicking would make plain
/// `cargo test` fail for everyone after any edit, and a clearly-reported
/// skip does not carry the failure mode being guarded — nobody reads
/// "NOT CHECKED" as a pass. Set `EF_REQUIRE_FRESH_ARTIFACTS=1` where build
/// order is guaranteed (the Makefile's `test-capi-modular` lane and CI) to
/// turn the skip into a hard failure.
fn artifact_is_fresh(artifact: &std::path::Path) -> bool {
    let art = match std::fs::metadata(artifact).and_then(|m| m.modified()) {
        Ok(t) => t,
        Err(_) => return false,
    };
    let src_dir = std::path::Path::new(concat!(env!("CARGO_MANIFEST_DIR"), "/src"));
    let newest_src = std::fs::read_dir(src_dir).ok().and_then(|rd| {
        rd.filter_map(|e| e.ok())
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
    // Straight to stderr: libtest swallows `eprintln!` for passing tests,
    // and a skip nobody sees is indistinguishable from a pass.
    use std::io::Write;
    let _ = writeln!(std::io::stderr(), "SKIP: {msg}");
    false
}

/// Symbols the built cdylib actually exports.
///
/// Returns `None` when the platform's symbol tool is unavailable, or when
/// the artifact is stale, so the check skips loudly rather than passing on
/// an empty list -- a test that asserts "X is absent" against an empty
/// vector always passes.
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

    // `cargo test` builds the rlib/test harness for this crate but does NOT
    // relink the cdylib, so this can easily read a library built before the
    // change under test. A stale read is worse than no read, so staleness
    // is checked explicitly rather than trusted.
    if !artifact_is_fresh(&lib) {
        return None;
    }
    // -g: external symbols only. -U/--defined-only: skip undefined ones,
    // which would otherwise include every libc symbol we merely call.
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
        return; // helper already panics/prints if the artifact is stale/missing
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
    // The other half of set equality: `every_ffi_declaration_is_a_real_export`
    // (above) checks DECLARED subset-of exports; this checks the exported
    // ef_-prefixed symbols subset-of DECLARED. Together they pin DECLARED to
    // be exactly the library's `ef_` export surface, so a new export cannot
    // land without a matching `edgefirst-tensor-ffi` declaration. Non-`ef_`
    // symbols are out of scope here -- `the_dynamic_symbol_table_carries_
    // only_ef_symbols` in `src/lib.rs` owns that hygiene check.
    let Some(symbols) = exported_symbols_fresh() else {
        return; // helper already panics/prints if the artifact is stale/missing
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
    // Both directions: DECLARED subset-of exports (above) and |DECLARED| ==
    // the number of function declarations in the shipped header, so a new
    // export cannot land without its Rust declaration.
    let header = std::fs::read_to_string(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/include/edgefirst/tensor.h"
    ))
    .expect("header exists");

    // A naive `ends_with(';') && contains('(')` line filter miscounts: several
    // declarations here wrap across lines (e.g. `ef_tensor_builder_add_plane`,
    // `ef_tensor_export`), so only their *first* line ends in `,` rather than
    // `;`, and every continuation line also contains `(` or `)` without being
    // a declaration of its own. What is uniform across every declaration --
    // wrapped or not -- is that the line where it *starts* is flush left (no
    // leading whitespace): every doc-comment line in this header (including
    // the ones that quote a call like `ef_tensor_new()` in prose) is indented
    // with a leading ` * `, and every continuation line of a wrapped
    // declaration is indented to align with the parameter list. So: a
    // top-level, non-indented line naming an `ef_` symbol and opening a
    // parameter list is exactly one declaration's start, regardless of where
    // it ends.
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
