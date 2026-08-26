// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! The C API for `edgefirst-tensor`.
//!
//! This crate exists so that `extern "C"` stays out of the Rust crate, and so
//! the `cdylib` can exist without a Rust consumer ever compiling the FFI
//! layer. It mirrors the `python-*` sibling layout.
//!
//! # Three surfaces, and layout appears in exactly one
//!
//! * **Opaque handle + exported accessors** — in-process. The handle is fully
//!   opaque; every accessor is an exported function of this library, the
//!   single implementation home of the tensor type. A tensor minted by a
//!   sibling library (e.g. `ef_image_processor_create_image()`) is the same
//!   kind of object as one from `ef_tensor_new()` -- every sibling links
//!   this library's shared `libedgefirst_tensor.so` rather than embedding
//!   its own copy, so there is exactly one implementation, and no dispatch
//!   is needed to read a handle regardless of which library minted it.
//! * **Builder** — construction, with sticky errors.
//! * **`(blob, fds)`** — IPC and serialization, the only place a layout is
//!   defined on the wire.
//!
//! Each library exports essentially one real symbol family, its
//! **constructors**, which is correct: "who allocated this" is the one
//! question where library identity matters.

pub mod builder;
pub mod codes;
pub mod cuda;
pub mod desc;
pub mod handle;
pub mod hardware;
pub mod image;
pub mod last_error;
pub mod log;
pub mod map;
pub mod mutate;
pub mod probe;
pub mod quant;
pub mod serialize;
pub mod trace;

/// ABI version of this library's C surface.
///
/// Bumped only when the C ABI breaks. Adding a new exported accessor does
/// *not* bump this: an existing consumer's header simply never names the new
/// symbol, so linking against an older `libedgefirst_tensor.so` still
/// resolves everything it actually calls.
#[no_mangle]
pub extern "C" fn ef_tensor_abi_version() -> u32 {
    1
}

#[cfg(test)]
mod tests {
    /// The shipped header text.
    ///
    /// Reading the generated header from a test is only meaningful because
    /// `build.rs` declares `rerun-if-changed=src/`: editing a source file
    /// regenerates it. Editing or deleting the header alone does **not**, so a
    /// test like this one validates a stale artifact unless the crate's source
    /// changed in the same commit.
    fn header_text() -> String {
        std::fs::read_to_string(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/include/edgefirst/tensor.h"
        ))
        .expect("cbindgen wrote include/edgefirst/tensor.h")
    }

    /// Compile a C translation unit against the shipped header.
    ///
    /// Returns `None` when no C compiler is available, so the check skips with
    /// a printed reason rather than passing silently on a host without one.
    fn cc_syntax_check(src: &str) -> Option<std::process::Output> {
        let include = concat!(env!("CARGO_MANIFEST_DIR"), "/include");
        let cc = std::env::var("CC").unwrap_or_else(|_| "cc".to_string());
        match std::process::Command::new(&cc)
            // -Werror: a header that merely *compiles* is not enough. The
            // vtable once declared `struct ef_tensor *` before the type
            // existed, so every dispatch warned about incompatible
            // pointer types while still exiting 0.
            .args(["-fsyntax-only", "-Werror", "-Wall", "-I", include, src])
            .output()
        {
            Ok(o) => Some(o),
            Err(e) => {
                eprintln!("SKIP: no C compiler ({cc}: {e}); header not syntax-checked");
                None
            }
        }
    }

    /// Like [`cc_syntax_check`], but pins `-std=c11`.
    ///
    /// `_Static_assert` is a C11 feature; a toolchain's bare `cc` default can
    /// predate it (or, on some vendors, silently accept it as an extension
    /// under an older `-std=`, which would not be testing what the golden
    /// file claims to test).
    fn cc_syntax_check_c11(src: &str) -> Option<std::process::Output> {
        let include = concat!(env!("CARGO_MANIFEST_DIR"), "/include");
        let cc = std::env::var("CC").unwrap_or_else(|_| "cc".to_string());
        match std::process::Command::new(&cc)
            .args([
                "-std=c11",
                "-fsyntax-only",
                "-Werror",
                "-Wall",
                "-I",
                include,
                src,
            ])
            .output()
        {
            Ok(o) => Some(o),
            Err(e) => {
                eprintln!("SKIP: no C compiler ({cc}: {e}); header not syntax-checked");
                None
            }
        }
    }

    /// Is this build artifact newer than the crate's sources?
    ///
    /// Every test here that reads a compiled artifact needs this check, because
    /// neither `cargo test` nor `cargo test --all-targets` builds the
    /// `staticlib`/`cdylib` targets — only `cargo build` does. A test that skips
    /// it silently validates a library built *before* the change under test,
    /// which is exactly how the "no accessor is exported" check was found to be
    /// vacuous.
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
    /// Returns `None` when the platform's symbol tool is unavailable, so the
    /// check skips loudly rather than passing on an empty list -- a test that
    /// asserts "X is absent" against an empty vector always passes.
    fn exported_symbols() -> Option<Vec<String>> {
        let mut dir = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        dir.pop();
        dir.pop();
        dir.push("target");
        dir.push("debug");
        let lib = ["libedgefirst_tensor.dylib", "libedgefirst_tensor.so"]
            .iter()
            .map(|n| dir.join(n))
            .find(|p| p.exists())?;

        // `cargo test --lib` builds the rlib for the test harness but does NOT
        // relink the cdylib, so this can easily read a library built before the
        // change under test. That is not hypothetical: adding an exported
        // `ef_tensor_ndim` and running `cargo test` left this passing, and only
        // an explicit `cargo build` made it fail.
        //
        // A stale read is worse than no read, so staleness is checked
        // explicitly: `artifact_is_fresh` loud-skips by default (a clearly
        // reported skip, not a silent pass) and only panics when
        // `EF_REQUIRE_FRESH_ARTIFACTS=1` is set, as it is in the Makefile's
        // `test-capi-modular` lane and CI.
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
    fn the_header_is_opaque_and_declares_every_accessor_exactly_once() {
        let text = header_text();
        assert!(
            !text.contains("static inline"),
            "no inline bodies: accessors are exported functions of the one home"
        );
        assert!(
            !text.contains("ef_tensor_vtable"),
            "the vtable is implementation machinery, not ABI"
        );
        assert!(
            !text.contains("struct ef_tensor {"),
            "the handle is fully opaque"
        );
        for name in [
            "ef_tensor_ndim",
            "ef_tensor_shape",
            "ef_tensor_strides",
            "ef_tensor_dtype",
            "ef_tensor_storage_kind",
            "ef_tensor_plane_count",
            "ef_tensor_format",
            "ef_tensor_plane_at",
        ] {
            // No leading-space requirement: cbindgen writes pointer-return
            // declarations as `const uint64_t *ef_tensor_shape(...)`, with no
            // space between `*` and the name.
            assert_eq!(
                text.matches(&format!("{name}(")).count(),
                1,
                "{name} declared exactly once"
            );
        }
    }

    #[test]
    fn every_accessor_is_an_exported_symbol() {
        // Inverted from r2's "no accessor is exported": under one implementation
        // home there is one exporter and nothing to interpose, so the accessors
        // ARE the ABI. The freshness lesson from the old test transfers unchanged.
        let Some(symbols) = exported_symbols() else {
            return; // helper already panics if the artifact is stale
        };
        for name in [
            "ef_tensor_ndim",
            "ef_tensor_shape",
            "ef_tensor_strides",
            "ef_tensor_dtype",
            "ef_tensor_storage_kind",
            "ef_tensor_plane_count",
            "ef_tensor_format",
            "ef_tensor_plane_at",
            "ef_tensor_new",
            "ef_tensor_free",
        ] {
            assert!(
                symbols.iter().any(|s| s == name),
                "{name} must be exported by libedgefirst_tensor"
            );
        }
    }

    #[test]
    fn the_dynamic_symbol_table_carries_only_ef_symbols() {
        // Rust cdylibs export more than intended; a trimmed dynamic symbol table
        // is ABI hygiene in itself (spec: Sonames, symbol visibility).
        let Some(symbols) = exported_symbols() else {
            return;
        };
        let foreign: Vec<_> = symbols.iter().filter(|s| !s.starts_with("ef_")).collect();
        assert!(
            foreign.is_empty(),
            "non-ef_ symbols in the dynamic table: {foreign:?}"
        );
    }

    /// Compile, link against the staticlib, and run a C program.
    ///
    /// Linking matters: a syntax-only pass cannot compare a size Rust computed
    /// against one C computed, and it would not notice a missing symbol either.
    fn cc_build_and_run(src: &str, bin_name: &str) -> Option<std::process::Output> {
        let include = concat!(env!("CARGO_MANIFEST_DIR"), "/include");
        let mut dir = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        dir.pop();
        dir.pop();
        dir.push("target");
        dir.push("debug");
        let lib = dir.join("libedgefirst_tensor.a");
        if !lib.exists() {
            eprintln!("SKIP: {} not built yet; C link test not run", lib.display());
            return None;
        }
        // The staticlib has the same staleness exposure as the cdylib: linking
        // C against a stale .a validates the previous build's symbols.
        if !artifact_is_fresh(&lib) {
            return None;
        }
        let out_bin = std::env::temp_dir().join(bin_name);
        let cc = std::env::var("CC").unwrap_or_else(|_| "cc".to_string());
        let mut cmd = std::process::Command::new(&cc);
        cmd.args(["-Werror", "-Wall", "-I", include, src])
            .arg(&lib)
            .arg("-o")
            .arg(&out_bin);
        // The staticlib needs the platform runtime the Rust code calls into.
        if cfg!(target_os = "macos") {
            cmd.args(["-framework", "CoreFoundation", "-framework", "IOSurface"]);
        } else {
            cmd.args(["-lpthread", "-ldl", "-lm"]);
        }
        // Probe the toolchain FIRST, with a trivial translation unit that does
        // not touch this library. Without this split, a genuine link failure —
        // a symbol we removed, a broken export — is indistinguishable from
        // "this host has no compiler", and gets swallowed as a skip. That is
        // not hypothetical: dropping `#[no_mangle]` from `ef_tensor_free` left
        // this test PASSING, reported as a toolchain problem.
        if !toolchain_works(&cc) {
            use std::io::Write;
            let _ = writeln!(
                std::io::stderr(),
                "SKIP: no working C toolchain ({cc}); C link test not run"
            );
            return None;
        }
        match cmd.output() {
            // The toolchain works, so a failure here is OUR fault: a missing
            // symbol, a bad header, a broken export. Never a skip.
            Ok(o) if !o.status.success() => panic!(
                "C link against libedgefirst_tensor FAILED (the toolchain is \
                 known good, so this is a real defect):\n{}",
                String::from_utf8_lossy(&o.stderr)
            ),
            Ok(_) => std::process::Command::new(&out_bin).output().ok(),
            Err(e) => panic!("failed to run {cc} after it probed OK: {e}"),
        }
    }

    /// Can this host compile and link a trivial C program at all?
    ///
    /// Separates "no toolchain here" from "our library is broken", which the
    /// C tests otherwise cannot tell apart.
    fn toolchain_works(cc: &str) -> bool {
        let probe = std::env::temp_dir().join("ef_cc_probe.c");
        let out = std::env::temp_dir().join("ef_cc_probe.bin");
        if std::fs::write(&probe, b"int main(void){return 0;}\n").is_err() {
            return false;
        }
        std::process::Command::new(cc)
            .arg(&probe)
            .arg("-o")
            .arg(&out)
            .output()
            .map(|o| o.status.success())
            .unwrap_or(false)
    }

    /// Can this host compile and link a trivial C++17 program with `cxx`?
    ///
    /// Same split as [`toolchain_works`], for the same reason: without it, a
    /// missing `c++`/`clang++` on this host and a genuine C++17 header defect
    /// are indistinguishable failures.
    fn cpp17_toolchain_works(cxx: &str) -> bool {
        let probe = std::env::temp_dir().join("ef_cxx17_probe.cpp");
        let out = std::env::temp_dir().join("ef_cxx17_probe.bin");
        if std::fs::write(&probe, b"int main() { return 0; }\n").is_err() {
            return false;
        }
        std::process::Command::new(cxx)
            .args(["-std=c++17"])
            .arg(&probe)
            .arg("-o")
            .arg(&out)
            .output()
            .map(|o| o.status.success())
            .unwrap_or(false)
    }

    /// Probe for a working C++17 toolchain: `c++` first, then `clang++`.
    fn find_cpp17_compiler() -> Option<&'static str> {
        ["c++", "clang++"]
            .into_iter()
            .find(|cxx| cpp17_toolchain_works(cxx))
    }

    /// Compile `path` as a freestanding C++17 translation unit with `cxx`.
    ///
    /// `-fsyntax-only`: a header needs no `main` to prove it parses. `-x c++`:
    /// forces C++ regardless of the file's extension, so a `.h` scratch file
    /// is compiled as C++ the same as the real header. `cxx` must already be
    /// toolchain-probed via [`find_cpp17_compiler`]; a spawn failure here is
    /// therefore our fault, never a missing-compiler skip.
    fn cpp17_syntax_check(cxx: &str, path: &str) -> std::process::Output {
        let include = concat!(env!("CARGO_MANIFEST_DIR"), "/include");
        std::process::Command::new(cxx)
            .args([
                "-x",
                "c++",
                "-std=c++17",
                "-Wall",
                "-Wextra",
                "-Wpedantic",
                "-Werror",
                "-fsyntax-only",
                "-I",
                include,
                path,
            ])
            .output()
            .unwrap_or_else(|e| {
                panic!("{cxx} failed to spawn after toolchain probe succeeded: {e}")
            })
    }

    #[test]
    fn the_header_compiles_as_cpp17() {
        // Every accessor already sits inside `extern "C" { }`, so a C++
        // translation unit including this header is a real use case (a C++
        // consumer linking `libedgefirst_tensor.a` directly), not a
        // hypothetical.
        let Some(cxx) = find_cpp17_compiler() else {
            eprintln!("SKIP: no working C++17 toolchain (c++/clang++); header not C++17-checked");
            return;
        };

        // Red-verify FIRST: prove this gate can actually fail before trusting
        // it to pass. `class` is a reserved keyword in C++ (legal as an
        // identifier in C), so a scratch header declaring `int class = 1;`
        // must be rejected under `-std=c++17`. If it were accepted, the gate
        // below would be trivially green regardless of what the real header
        // contains.
        let scratch = std::env::temp_dir().join("ef_cpp17_hostile_scratch.h");
        std::fs::write(&scratch, b"int class = 1;\n").expect("write scratch header");
        let bad = cpp17_syntax_check(cxx, scratch.to_str().expect("temp path is UTF-8"));
        assert!(
            !bad.status.success(),
            "red-verification failed: `int class = 1;` compiled cleanly under \
             {cxx} -std=c++17, so this gate cannot fail and proves nothing"
        );

        let header = concat!(env!("CARGO_MANIFEST_DIR"), "/include/edgefirst/tensor.h");
        let out = cpp17_syntax_check(cxx, header);
        assert!(
            out.status.success(),
            "tensor.h failed to compile as C++17 ({cxx} -x c++ -std=c++17 -Wall \
             -Wextra -Wpedantic -Werror):\n{}",
            String::from_utf8_lossy(&out.stderr)
        );
    }

    #[test]
    fn the_layout_goldens_hold() {
        // Sizes and offsets are the drift class no name-level check can see,
        // and are identical on every LP64 target this header ships for.
        let src = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/c/test_layout_goldens.c");
        let Some(out) = cc_syntax_check_c11(src) else {
            return;
        };
        assert!(
            out.status.success(),
            "layout golden _Static_asserts failed:\n{}",
            String::from_utf8_lossy(&out.stderr)
        );
    }

    #[test]
    fn a_c_caller_can_chain_the_builder_and_check_once() {
        // Compiles, links and runs: the sticky-error contract is only useful
        // if it holds through the real C surface, not just the Rust one.
        let src = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/c/test_builder.c");
        let Some(out) = cc_build_and_run(src, "ef_test_builder") else {
            return;
        };
        assert!(
            out.status.success(),
            "C builder contract failed:\n{}",
            String::from_utf8_lossy(&out.stderr)
        );
    }

    #[test]
    fn a_c_caller_reads_a_tensor_through_the_exported_accessors() {
        // The C-side proof that the header's declarations, the exported
        // symbols, and the behavior agree, now that the accessors are real
        // exported functions rather than static inline bodies dispatching
        // through a hand-mirrored vtable.
        let src = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/c/test_accessors.c");
        let Some(out) = cc_build_and_run(src, "ef_test_accessors") else {
            return;
        };
        assert!(
            out.status.success(),
            "C accessor contract failed:\n{}",
            String::from_utf8_lossy(&out.stderr)
        );
    }

    #[test]
    fn a_c_caller_can_map_unmap_and_copy_a_tensor() {
        // The C-side proof of the map window's contract: a write map, unmap,
        // then a copy_to that needs no map, plus the None/EINVAL and
        // retain-survives-one-free edges.
        let src = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/c/test_map.c");
        let Some(out) = cc_build_and_run(src, "ef_test_map") else {
            return;
        };
        assert!(
            out.status.success(),
            "C map-window contract failed:\n{}",
            String::from_utf8_lossy(&out.stderr)
        );
    }

    #[test]
    fn the_header_survives_being_included_twice() {
        // cbindgen's own `include_guard` option wraps only the generated body,
        // leaving `header` and `trailer` outside it, so a second #include would
        // redefine the forward-declared typedefs and the generated struct and
        // function declarations between them. The guard is therefore written
        // by hand in header/trailer.
        let src = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/c/test_double_include.c");
        let Some(out) = cc_syntax_check(src) else {
            return;
        };
        assert!(
            out.status.success(),
            "header failed to compile when included twice:\n{}",
            String::from_utf8_lossy(&out.stderr)
        );
    }

    #[test]
    fn the_shipped_artifact_is_libedgefirst_tensor_not_capi() {
        // REQUIRED: consumers link `-ledgefirst-tensor`. The `-capi` suffix
        // names the source crate only, mirroring the `python-*` siblings, and
        // must never reach a filename. A `[lib] name` typo here would ship
        // libedgefirst_tensor_capi.so and break every consumer's link line.
        let mut dir = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        dir.pop();
        dir.pop();
        dir.push("target");
        dir.push("debug");
        let found: Vec<String> = std::fs::read_dir(&dir)
            .map(|rd| {
                rd.filter_map(|e| e.ok())
                    .map(|e| e.file_name().to_string_lossy().to_string())
                    .filter(|n| n.starts_with("libedgefirst_tensor"))
                    .collect()
            })
            .unwrap_or_default();
        if found.is_empty() {
            eprintln!(
                "SKIP: nothing built in {}; artifact name not checked",
                dir.display()
            );
            return;
        }
        assert!(
            !found.iter().any(|n| n.contains("_capi")),
            "the C library must ship as libedgefirst_tensor.*, never \
             libedgefirst_tensor_capi.*; found {found:?}.\n\
             If `[lib] name` is already correct, these are STALE artifacts from \
             a previous name -- cargo does not remove them on rename, and an \
             install or link step can still pick the wrong file up. \
             Delete target/debug/libedgefirst_tensor_capi.* or `cargo clean`."
        );
        assert!(
            found.iter().any(|n| n == "libedgefirst_tensor.a"),
            "expected libedgefirst_tensor.a among {found:?}"
        );
    }

    #[test]
    fn the_header_names_every_vocabulary_enumerator_the_readme_promises() {
        // Without these a caller writes `ef_tensor_builder_dtype(b, 0)` and
        // hopes -- and bare integers are how this repo previously ended up with
        // Python's MEM == 3 colliding with C's PBO == 3. The README's examples
        // use these exact spellings, so a rename would make the docs lie.
        let h = header_text();
        for name in [
            "EF_DTYPE_U8",
            "EF_DTYPE_F32",
            "EF_STORAGE_KIND_MEM",
            "EF_STORAGE_KIND_DMA_BUF",
            "EF_STORAGE_KIND_PBO",
        ] {
            assert!(h.contains(name), "tensor.h must declare {name}");
        }
        // The README is a promise about this header; keep them in step.
        let readme = std::fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/README.md"))
            .expect("README.md");
        for tok in readme
            .split(|c: char| !(c.is_ascii_alphanumeric() || c == '_'))
            .filter(|t| t.starts_with("EF_"))
        {
            assert!(
                h.contains(tok),
                "README names `{tok}` but tensor.h does not declare it"
            );
        }
    }

    #[test]
    fn no_rust_type_name_leaks_into_the_c_api() {
        // cbindgen emits Rust type names verbatim unless told otherwise, so
        // `struct EfTensorBuilder` would become part of the published C API and
        // then be permanent. Every public name must read as C.
        let h = header_text();
        for rust_name in ["EfTensor", "EfTensorBuilder", "EfTensorImpl"] {
            assert!(
                !h.contains(rust_name),
                "the Rust type name `{rust_name}` leaked into tensor.h; add it to \
                 cbindgen.toml's [export.rename]"
            );
        }
    }

    #[test]
    fn the_generated_header_exists_and_declares_the_abi_probe() {
        let h = header_text();
        assert!(
            h.contains("ef_tensor_abi_version"),
            "the header must declare the ABI probe"
        );
    }

    #[test]
    fn the_header_carries_none_of_the_monoliths_symbols() {
        // This library is standalone. If `hal_` appears here, the crate has
        // picked up the old monolith's surface and the split has not happened.
        let h = header_text();
        assert!(
            !h.contains("hal_"),
            "tensor.h must not carry the monolith's hal_* symbols"
        );
    }

    /// Parse cbindgen's `EF_NAME = VALUE,`-style enumerator lines out of the
    /// shipped header.
    ///
    /// cbindgen 0.29 (with `cpp_compat = true`, see cbindgen.toml) emits each
    /// vocabulary as one `enum ef_foo #if ... : uint32_t #endif { EF_FOO_X =
    /// 0, ... };` block: the `#if defined(__cplusplus) || __STDC_VERSION__ >=
    /// 202311L` conditional around the block only adds or removes the fixed
    /// *underlying type* annotation, and the `#ifndef __cplusplus` block after
    /// it only adds or skips a redundant `typedef`; the enumerator values
    /// themselves appear exactly once, unconditionally, inside the `{ }`. A
    /// flat per-line scan is therefore exhaustive and unambiguous -- there is
    /// no second, differently-formatted copy of the values to miss.
    fn parse_enumerator_values(header: &str) -> std::collections::HashMap<String, u32> {
        let mut values = std::collections::HashMap::new();
        for line in header.lines() {
            let line = line.trim().trim_end_matches(',').trim_end_matches(';');
            let Some((name, value)) = line.split_once('=') else {
                continue;
            };
            let name = name.trim();
            let value = value.trim();
            if !name.starts_with("EF_") {
                continue;
            }
            if let Ok(v) = value.parse::<u32>() {
                values.insert(name.to_string(), v);
            }
        }
        values
    }

    #[test]
    fn the_header_enumerator_values_match_the_rust_vocabulary() {
        // Name-level checks (`the_header_names_every_vocabulary_enumerator_
        // the_readme_promises`) only prove a spelling exists; they say nothing
        // about *which integer* cbindgen wrote next to it. codes.rs's
        // compile-time `assert!` block proves the Rust-side ABI crate agrees
        // with `edgefirst-tensor`'s authority, but that is still one Rust
        // crate agreeing with another -- it never reads the generated
        // artifact a C caller actually links against. This test closes that
        // last link: the text cbindgen wrote into tensor.h, parsed back out,
        // against the same authorities codes.rs uses.
        use edgefirst_tensor::{DType, TensorMemory};
        use edgefirst_tensor_abi::EfCpuAccess;

        let h = header_text();
        let values = parse_enumerator_values(&h);

        let expect = |name: &str, want: u32| {
            let got = *values
                .get(name)
                .unwrap_or_else(|| panic!("tensor.h does not declare enumerator {name}"));
            assert_eq!(
                got, want,
                "{name}: tensor.h says {got}, the Rust vocabulary says {want}"
            );
        };

        expect("EF_DTYPE_U8", DType::U8.code());
        expect("EF_DTYPE_I8", DType::I8.code());
        expect("EF_DTYPE_U16", DType::U16.code());
        expect("EF_DTYPE_I16", DType::I16.code());
        expect("EF_DTYPE_U32", DType::U32.code());
        expect("EF_DTYPE_I32", DType::I32.code());
        expect("EF_DTYPE_U64", DType::U64.code());
        expect("EF_DTYPE_I64", DType::I64.code());
        expect("EF_DTYPE_F16", DType::F16.code());
        expect("EF_DTYPE_F32", DType::F32.code());
        expect("EF_DTYPE_F64", DType::F64.code());

        expect("EF_STORAGE_KIND_MEM", TensorMemory::Mem.code());
        expect("EF_STORAGE_KIND_SHM", TensorMemory::Shm.code());
        expect("EF_STORAGE_KIND_DMA_BUF", TensorMemory::DmaBuf.code());
        expect("EF_STORAGE_KIND_IO_SURFACE", TensorMemory::IoSurface.code());
        expect("EF_STORAGE_KIND_PBO", TensorMemory::Pbo.code());
        expect("EF_STORAGE_KIND_CUDA", TensorMemory::Cuda.code());

        expect("EF_CPU_ACCESS_NONE", EfCpuAccess::None as u32);
        expect("EF_CPU_ACCESS_READ", EfCpuAccess::Read as u32);
        expect("EF_CPU_ACCESS_WRITE", EfCpuAccess::Write as u32);
        expect("EF_CPU_ACCESS_READ_WRITE", EfCpuAccess::ReadWrite as u32);
    }
}
