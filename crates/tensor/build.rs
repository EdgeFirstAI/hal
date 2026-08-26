// SPDX-FileCopyrightText: Copyright 2025 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0
fn main() {
    println!("cargo::rustc-check-cfg=cfg(coverage)");
    let rustflags = std::env::var("CARGO_ENCODED_RUSTFLAGS").unwrap_or_default();
    if rustflags.contains("instrument-coverage") {
        println!("cargo::rustc-cfg=coverage");
    }
    println!("cargo::rerun-if-env-changed=CARGO_ENCODED_RUSTFLAGS");

    // Opt-in only (the `dynamic-test-link` feature, never enabled by a
    // production consumer -- see its doc comment in Cargo.toml): links
    // `edgefirst-tensor`'s own `--features dynamic` test/doctest binaries
    // against `libedgefirst_tensor.so`, which `edgefirst-tensor-ffi`
    // deliberately declares with no `#[link]` attribute of its own
    // (linking is normally the *consumer's* decision, and this crate has no
    // other consumer to make that decision on its behalf for its own
    // tests). `../../target/debug`, relative to this crate's manifest dir,
    // is where `crates/tensor-capi` lands when built with `--target-dir
    // target` from the workspace root -- the same directory `cargo
    // test -p edgefirst-tensor` itself builds into, so no extra
    // search-path configuration is needed beyond building `tensor-capi`
    // first. A stale or missing `.so` here is a link error, not a silent
    // pass, the same "build the producer first" precondition
    // `tensor-capi`'s own `EF_REQUIRE_FRESH_ARTIFACTS`-gated tests document.
    if std::env::var_os("CARGO_FEATURE_DYNAMIC_TEST_LINK").is_some() {
        let manifest_dir =
            std::env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR is always set");
        let lib_dir = std::path::Path::new(&manifest_dir).join("../../target/debug");
        println!("cargo::rustc-link-search=native={}", lib_dir.display());
        println!("cargo::rustc-link-lib=dylib=edgefirst_tensor");
    }
}
