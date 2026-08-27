// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! One opaque C type must have one implementation. `EfTensorImageDesc` was
//! F10 in review round 1 of the single-tensor-home work: `image-capi`
//! carried a `#[repr(transparent)]` copy of a struct `tensor-capi` also
//! defined, with no `repr` at all on tensor-capi's side, so the two were not
//! even formally guaranteed to agree -- exactly the `ef_detect_box_list`
//! situation Task 3 undid, minus the vtable that would have made it safe.
//!
//! This test makes "no other struct name is duplicated this way" mechanical
//! rather than a claim re-verified by hand every review round: it scans
//! every `struct` declaration in the five `-capi` crates and fails on any
//! name declared by more than one, outside a named allow-list.
//!
//! Scoped to the five `-capi` crates only. `EfTensorBuilder` pairs with a
//! matching `_opaque: [u8; 0]` marker in `tensor-ffi`, and that is the
//! correct pattern, not a violation: `tensor-ffi` is declarations-only (no
//! `#[no_mangle]`, nothing to implement, so nothing that can drift out of
//! sync with the real definition), which is exactly why it is deliberately
//! out of scope here.

use std::collections::BTreeMap;

/// Struct names that legitimately appear in more than one `-capi` crate.
///
/// Used to hold the transition vtable's dispatch machinery (`EfTensor`, its
/// per-library dispatch-table struct, `EfTensorImpl`), present once per
/// sibling so each could dispatch to whichever library minted a handle.
/// Task 10 (single-tensor-home) deleted that machinery: every sibling now
/// links `libedgefirst_tensor.so`
/// dynamically instead of embedding its own copy, so `EfTensor` and friends
/// are declared exactly once, in `tensor-capi` itself, and the siblings only
/// `use edgefirst_tensor_ffi::EfTensor` -- an import, not a declaration. This
/// list may only ever shrink, never grow -- a new entry here is a new
/// duplicate, not something to silence.
const ALLOWED_DUPLICATES: &[&str] = &[];

/// `struct Name` declarations in one file's text, `pub`/`pub(crate)`/private
/// alike -- visibility does not change whether two crates defining the same
/// name is a hazard.
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
    // A name sitting in the list after the code it described is deleted
    // would silently stop testing anything -- this keeps the list honest.
    let owners = struct_owners();
    for name in ALLOWED_DUPLICATES {
        assert!(
            owners.get(*name).is_some_and(|c| c.len() > 1),
            "{name} is in ALLOWED_DUPLICATES but is no longer declared by \
             more than one -capi crate; remove it -- the list may only shrink"
        );
    }
}
