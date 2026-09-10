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
    // tests). A stale or missing `.so` here is a link error, not a silent
    // pass, the same "build the producer first" precondition
    // `tensor-capi`'s own `EF_REQUIRE_FRESH_ARTIFACTS`-gated tests document.
    if std::env::var_os("CARGO_FEATURE_DYNAMIC_TEST_LINK").is_some() {
        // `EDGEFIRST_TENSOR_LIB_DIR` wins when set -- cross-compiling this
        // lane for a board (`cargo zigbuild ... --target
        // aarch64-unknown-linux-gnu.2.35`) needs to point at a cross-built
        // `libedgefirst_tensor.so` under `target/aarch64-unknown-linux-gnu/
        // debug/`, which nothing below can derive on its own: there is no
        // env var that names the *producer* crate's (`tensor-capi`) output
        // directory, only this crate's own.
        println!("cargo::rerun-if-env-changed=EDGEFIRST_TENSOR_LIB_DIR");
        let lib_dir =
            match std::env::var_os("EDGEFIRST_TENSOR_LIB_DIR").filter(|dir| !dir.is_empty()) {
                Some(dir) => std::path::PathBuf::from(dir),
                None => {
                    // Cargo always lays `OUT_DIR` out as
                    // `<profile-dir>/build/<pkg>-<hash>/out`, where
                    // `<profile-dir>` is `target/<profile>` for a native build
                    // and `target/<target-triple>/<profile>` for any `--target`
                    // build (cross or not) -- exactly the directory sibling
                    // crates' compiled dylibs land in, including `tensor-capi`'s
                    // when built with `--target-dir target` from the workspace
                    // root. Walking up from `OUT_DIR` finds it without assuming
                    // a fixed `../../target/debug` relative to this crate's
                    // manifest dir (wrong for any `--target` build, cross or
                    // not) or a fixed `--target-dir` (wrong for a custom one --
                    // see `scripts/on-target-test.sh`'s own comment on exactly
                    // this class of bug in `build_libs_for`).
                    let out_dir = std::env::var("OUT_DIR").expect("OUT_DIR is always set");
                    std::path::Path::new(&out_dir)
                        .ancestors()
                        .nth(3)
                        .expect("OUT_DIR is nested at least 3 levels under target/<profile>")
                        .to_path_buf()
                }
            };
        println!("cargo::rustc-link-search=native={}", lib_dir.display());
        println!("cargo::rustc-link-lib=dylib=edgefirst_tensor");
    }
}
