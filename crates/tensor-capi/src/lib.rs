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
pub mod d3d11;
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

// Test-support modules shared with the other modular C-API leaves. They are
// included here, relative to `src/`, rather than inside `mod tests`: a
// `#[path]` inside an inline module resolves relative to `src/tests/`, and
// Linux and macOS refuse to walk `..` through a directory that does not exist.
#[cfg(test)]
#[path = "../tests/support/artifacts.rs"]
mod artifacts;
#[cfg(all(test, windows))]
#[path = "../tests/support/msvc.rs"]
mod msvc;

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
    #[cfg(not(windows))]
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
    #[cfg(not(windows))]
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

    // Artifact discovery, freshness and symbol reading, shared with the
    // other modular C-API leaves; included at the crate root (see the
    // `#[path]` above `mod tests`) because a path relative to this inline
    // module would traverse `src/tests/`, a directory that does not exist,
    // which Linux and macOS refuse to resolve.
    use crate::artifacts;
    use artifacts::skip;

    /// This leaf's sources, which every artifact read here is checked
    /// against: a library older than them proves nothing about the change
    /// under test.
    const SRC_DIR: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/src");

    /// The staticlib and the shared library this platform builds, plus the
    /// prefix every artifact of this library shares.
    ///
    /// MSVC drops the `lib` prefix and names the staticlib `.lib`, so these
    /// are the platform's spellings, not POSIX's: looking for the POSIX
    /// names on Windows found nothing and skipped every run.
    #[cfg(not(windows))]
    const ARTIFACT_PREFIX: &str = "libedgefirst_tensor";
    #[cfg(windows)]
    const ARTIFACT_PREFIX: &str = "edgefirst_tensor";
    #[cfg(not(windows))]
    const STATICLIB: &str = "libedgefirst_tensor.a";
    #[cfg(windows)]
    const STATICLIB: &str = "edgefirst_tensor.lib";

    /// A built artifact of this leaf, fresh, or the reason there is none.
    fn find_artifact(name: &str) -> Result<std::path::PathBuf, String> {
        artifacts::find_artifact(name, std::path::Path::new(SRC_DIR))
    }

    /// Symbols the built cdylib actually exports.
    ///
    /// Returns `None` when the platform's symbol tool is unavailable, so the
    /// check skips loudly rather than passing on an empty list -- a test that
    /// asserts "X is absent" against an empty vector always passes.
    fn exported_symbols() -> Option<Vec<String>> {
        // `cargo test --lib` builds the rlib for the test harness but does NOT
        // relink the cdylib, so this can easily read a library built before the
        // change under test. That is not hypothetical: adding an exported
        // `ef_tensor_ndim` and running `cargo test` left this passing, and only
        // an explicit `cargo build` made it fail.
        //
        // A stale read is worse than no read, so `find_cdylib` passes over a
        // stale copy and reports one only when no candidate directory holds a
        // fresh one: a loud skip by default, a panic under
        // `EF_REQUIRE_FRESH_ARTIFACTS=1` as the Makefile's
        // `test-capi-modular` lane and CI set it.
        let lib = match artifacts::find_cdylib("edgefirst_tensor", std::path::Path::new(SRC_DIR)) {
            Ok(lib) => lib,
            Err(reason) => {
                skip(&format!("exports not read: {reason}"));
                return None;
            }
        };
        match artifacts::symbols_of(&lib) {
            Ok(syms) => Some(syms),
            Err(reason) => {
                skip(&format!("exports of {} not read: {reason}", lib.display()));
                None
            }
        }
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
    #[cfg(not(windows))]
    fn cc_build_and_run(src: &str, bin_name: &str) -> Option<std::process::Output> {
        let include = concat!(env!("CARGO_MANIFEST_DIR"), "/include");
        let lib = match find_artifact(STATICLIB) {
            Ok(lib) => lib,
            Err(reason) => {
                skip(&format!("{src} not linked: {reason}"));
                return None;
            }
        };
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
            skip(&format!("{src} not linked: no working C toolchain ({cc})"));
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
            Ok(_) => run_program(&out_bin, std::process::Command::new(&out_bin)),
            Err(e) => panic!("failed to run {cc} after it probed OK: {e}"),
        }
    }

    /// Can this host compile and link a trivial C program at all?
    ///
    /// Separates "no toolchain here" from "our library is broken", which the
    /// C tests otherwise cannot tell apart.
    #[cfg(not(windows))]
    fn toolchain_works(cc: &str) -> bool {
        // Per-process names: two leaves' test binaries, or two `cargo test`
        // runs, would otherwise compile over each other's probe.
        let pid = std::process::id();
        let probe = std::env::temp_dir().join(format!("ef_cc_probe-{pid}.c"));
        let out = std::env::temp_dir().join(format!("ef_cc_probe-{pid}.bin"));
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

    // MSVC discovery shared with the other modular C-API leaves; included at
    // the crate root for the same reason as `artifacts`.
    #[cfg(windows)]
    use crate::msvc;

    /// The Windows twin of `cc_syntax_check`: `cl.exe /Zs` parses without
    /// generating code, and `/W4 /WX` stands in for `-Wall -Werror`.
    #[cfg(windows)]
    fn cc_syntax_check(src: &str) -> Option<std::process::Output> {
        let include = concat!(env!("CARGO_MANIFEST_DIR"), "/include");
        Some(msvc::require(src)?.syntax_check(&[include], &[], src))
    }

    /// The Windows twin of `cc_syntax_check_c11`: `/std:c11` pins the C11
    /// mode the goldens' `_Static_assert` needs, as `-std=c11` does for `cc`.
    #[cfg(windows)]
    fn cc_syntax_check_c11(src: &str) -> Option<std::process::Output> {
        let include = concat!(env!("CARGO_MANIFEST_DIR"), "/include");
        Some(msvc::require(src)?.syntax_check(&[include], &["/std:c11"], src))
    }

    /// The Windows twin of `cc_build_and_run`: `cl.exe` against the staticlib
    /// cargo names `edgefirst_tensor.lib` on MSVC.
    #[cfg(windows)]
    fn cc_build_and_run(src: &str, bin_name: &str) -> Option<std::process::Output> {
        let include = concat!(env!("CARGO_MANIFEST_DIR"), "/include");
        let lib = match find_artifact(STATICLIB) {
            Ok(lib) => lib,
            Err(reason) => {
                skip(&format!("{src} not linked: {reason}"));
                return None;
            }
        };
        // Same split as the POSIX branch: a toolchain that cannot build a
        // trivial program is a skip naming what failed; any failure after
        // that is our defect.
        let toolchain = msvc::require(src)?;
        // Under the toolchain's own scratch directory, one per process: the
        // libtest threads driving these programs would otherwise write one
        // another's object files.
        let exe = toolchain.scratch().join(format!("{bin_name}.exe"));
        // The staticlib alone: the `windows` crate binds Direct3D 11 and
        // DXGI as raw-dylib imports that travel inside it, and `cargo rustc
        // -- --print native-static-libs` names neither d3d11.lib nor
        // dxgi.lib, so listing them added two libraries the link resolves
        // nothing from. The image leaf never listed them either.
        let libs = [lib.as_os_str()];
        let out = toolchain.build(&[include], src, &exe, &libs);
        // The command line and cl's own stdout diagnostics are folded into
        // stderr by the harness, so this message carries both.
        assert!(
            out.status.success(),
            "C link against edgefirst_tensor.lib FAILED (the toolchain is \
             known good, so this is a real defect):\n{}",
            String::from_utf8_lossy(&out.stderr)
        );
        run_program(&exe, std::process::Command::new(&exe))
    }

    /// Runs a built C program, reporting a spawn failure rather than
    /// swallowing it.
    ///
    /// A program that cannot even start is not a skip: the harness compiled
    /// and linked it a moment ago, so the reason is a missing DLL or a
    /// broken image, and `None` alone would leave the test reporting ok.
    fn run_program(
        exe: &std::path::Path,
        mut cmd: std::process::Command,
    ) -> Option<std::process::Output> {
        match cmd.output() {
            Ok(out) => Some(out),
            Err(e) => {
                skip(&format!(
                    "{} was built but could not be run ({cmd:?}): {e}",
                    exe.display()
                ));
                None
            }
        }
    }

    /// Can this host compile and link a trivial C++17 program with `cxx`?
    ///
    /// Same split as [`toolchain_works`], for the same reason: without it, a
    /// missing `c++`/`clang++` on this host and a genuine C++17 header defect
    /// are indistinguishable failures.
    fn cpp17_toolchain_works(cxx: &str) -> bool {
        // Per-process names, as `toolchain_works` uses.
        let pid = std::process::id();
        let probe = std::env::temp_dir().join(format!("ef_cxx17_probe-{pid}.cpp"));
        let out = std::env::temp_dir().join(format!("ef_cxx17_probe-{pid}.bin"));
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
        let scratch =
            std::env::temp_dir().join(format!("ef_cpp17_hostile_scratch-{}.h", std::process::id()));
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
    fn d3d11_exports_are_usable_or_refuse_off_windows() {
        // The C-side proof that the D3D11 family is declared on every
        // platform: on Windows it allocates a texture tensor and drives the
        // layout, share, completion and re-wrap path; elsewhere it asserts
        // each export refuses at run time instead of failing to link.
        let src = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/c/test_d3d11.c");
        let Some(out) = cc_build_and_run(src, "ef_test_d3d11") else {
            return;
        };
        // The program reports a missing device on stdout, so a --nocapture
        // run shows whether the Windows arm ran or skipped.
        eprint!("{}", String::from_utf8_lossy(&out.stdout));
        assert!(
            out.status.success(),
            "C D3D11 contract failed:\n{}",
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
        //
        // The staticlib has to exist for the check to mean anything, and
        // finding it is what says the leaf was built at all.
        if let Err(reason) = find_artifact(STATICLIB) {
            skip(&format!("artifact name not checked: {reason}"));
            return;
        }
        // Every candidate directory, not just the one that held the fresh
        // staticlib: a `_capi` file left in a target directory this run did
        // not build into is exactly the one an install or link step can
        // still pick up.
        let found = artifacts::artifacts_named(ARTIFACT_PREFIX);
        let wrong: Vec<&String> = found.iter().filter(|n| n.contains("_capi")).collect();
        assert!(
            wrong.is_empty(),
            "the C library must ship as {ARTIFACT_PREFIX}.*, never \
             {ARTIFACT_PREFIX}_capi.*; found {wrong:?} among {found:?} in {dirs:?}.\n\
             If `[lib] name` is already correct, these are STALE artifacts from \
             a previous name -- cargo does not remove them on rename, and an \
             install or link step can still pick the wrong file up. \
             Delete them or `cargo clean`.",
            dirs = artifacts::artifact_dirs()
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
