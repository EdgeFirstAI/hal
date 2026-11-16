// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! One opaque C type must have one implementation.
//!
//! This test scans every `struct` declaration in the five `-capi` crates and
//! fails on any name declared by more than one, outside a named allow-list.
//! `tensor-ffi`, which holds declarations only, is not scanned.

use std::collections::BTreeMap;

/// Struct names that may appear in more than one `-capi` crate.
const ALLOWED_DUPLICATES: &[&str] = &[];

/// `struct Name` declarations in one file's text, at any visibility.
fn struct_names_in(text: &str) -> Vec<String> {
    let mut out = Vec::new();
    for line in text.lines() {
        let line = line.trim_start();
        let rest = ["pub(crate) ", "pub(super) ", "pub "]
            .iter()
            .find_map(|p| line.strip_prefix(p))
            .unwrap_or(line);
        let Some(rest) = rest.strip_prefix("struct ") else {
            continue;
        };
        let name: String = rest
            .chars()
            .take_while(|c| c.is_ascii_alphanumeric() || *c == '_')
            .collect();
        if !name.is_empty() {
            out.push(name);
        }
    }
    out
}

/// struct name -> the `-capi` crates (by directory name) that declare it.
fn struct_owners() -> BTreeMap<String, Vec<String>> {
    let root = std::path::Path::new(concat!(env!("CARGO_MANIFEST_DIR"), "/.."));
    let mut owners: BTreeMap<String, Vec<String>> = BTreeMap::new();
    let Ok(entries) = std::fs::read_dir(root) else {
        return owners;
    };
    for e in entries.filter_map(Result::ok) {
        let crate_name = e.file_name().to_string_lossy().to_string();
        if !crate_name.ends_with("-capi") {
            continue;
        }
        let src = e.path().join("src");
        let Ok(files) = std::fs::read_dir(&src) else {
            continue;
        };
        for f in files.filter_map(Result::ok) {
            if f.path().extension().and_then(std::ffi::OsStr::to_str) != Some("rs") {
                continue;
            }
            let Ok(text) = std::fs::read_to_string(f.path()) else {
                continue;
            };
            for s in struct_names_in(&text) {
                let owners_of_s = owners.entry(s).or_default();
                if !owners_of_s.contains(&crate_name) {
                    owners_of_s.push(crate_name.clone());
                }
            }
        }
    }
    owners
}

#[test]
fn no_struct_name_is_declared_by_two_capi_crates_outside_the_allow_list() {
    let owners = struct_owners();
    assert!(
        !owners.is_empty(),
        "scanned zero struct declarations across crates/*-capi/src -- the \
         glob matched nothing, which is a broken test, not a clean result"
    );
    let mut violations: Vec<(String, Vec<String>)> = owners
        .into_iter()
        .filter(|(name, crates)| crates.len() > 1 && !ALLOWED_DUPLICATES.contains(&name.as_str()))
        .collect();
    violations.sort();
    assert!(
        violations.is_empty(),
        "these struct names are declared by more than one -capi crate, which \
         is exactly the two-implementations-of-one-opaque-type hazard this \
         test exists to catch. Add to ALLOWED_DUPLICATES only if it is \
         transition vtable machinery; never to silence a real duplicate --\
         give the real type a single home instead, the way EfTensorImageDesc \
         was moved to tensor-capi: {violations:#?}"
    );
}

#[test]
fn the_allow_list_only_names_things_actually_present() {
    let owners = struct_owners();
    for name in ALLOWED_DUPLICATES {
        assert!(
            owners.get(*name).is_some_and(|c| c.len() > 1),
            "{name} is in ALLOWED_DUPLICATES but is no longer declared by \
             more than one -capi crate; remove it -- the list may only shrink"
        );
    }
}
