// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! The C API for `edgefirst-image`.
//!
//! Sibling crate to `edgefirst-image`, keeping `extern "C"` out of the Rust
//! crate. Ships as `libedgefirst_image` with `edgefirst/image.h`.
//!
//! # This library mints tensors
//!
//! `ef_image_processor_create_image` returns an `ef_tensor`, and allocating a
//! PBO is the one operation that genuinely requires a processor — only the GL
//! context owner can create one.
//!
//! A handle from here is an ordinary `ef_tensor`: callers read it with
//! `ef_tensor_shape` and release it with `ef_tensor_free`, exactly as they would
//! one from `ef_tensor_new`. `edgefirst/image.h` includes `edgefirst/tensor.h`
//! and the pkg-config declares `Requires: edgefirst-tensor`, because using this
//! library means using the tensor API.
//!
//! This crate links `libedgefirst_tensor.so` dynamically rather than
//! embedding a private copy of the tensor implementation, so a tensor from
//! here is not merely "compatible" with one from `ef_tensor_new` -- it *is*
//! one, the same `EfTensorImpl` layout, allocated by the same allocator. That
//! is what lets a caller read and free it with the ordinary exported
//! `ef_tensor_*` accessors regardless of which library minted it.
//!
//! This crate does not link `edgefirst-tensor-capi` (the source crate that
//! builds `libedgefirst_tensor.so`) as a Rust dependency, for one concrete
//! reason: that crate's `#[no_mangle]` constructors would then be compiled
//! into this `.so` as well, and two libraries exporting `ef_tensor_new` is
//! the interposition hazard task 9's dynamic-linking switch exists to avoid
//! -- the dynamic linker would bind every caller to one copy of it.
//!
//! `edgefirst-tensor-ffi` is not that crate, despite the name similarity: it
//! is declarations-only (no `#[no_mangle]`, no `#[link]`), so depending on it
//! adds no symbol here, only an undefined reference resolved when something
//! later links `libedgefirst_tensor.so` alongside this one. `processor.rs`
//! uses it both to mint tensors (`TensorDyn::into_raw`) and to read one it
//! did not mint (`TensorDyn::with_raw`), and to read an `ef_tensor_image_desc`
//! handle through `ef_tensor_image_desc_get`'s scalar view rather than a
//! dereference of tensor-capi's private layout.

pub mod draw;
pub mod processor;
pub mod tiling;

/// ABI version of this library's C surface.
#[no_mangle]
pub extern "C" fn ef_image_abi_version() -> u32 {
    1
}

// Test-support modules that live in the tensor leaf. They are included here,
// relative to `src/`, rather than inside `mod tests`: a `#[path]` inside an
// inline module resolves relative to `src/tests/`, and Linux and macOS refuse
// to walk `..` through a directory that does not exist.
#[cfg(test)]
#[path = "../../tensor-capi/tests/support/artifacts.rs"]
mod artifacts;
#[cfg(all(test, windows))]
#[path = "../../tensor-capi/tests/support/msvc.rs"]
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
            "/include/edgefirst/image.h"
        ))
        .expect("cbindgen wrote include/edgefirst/image.h")
    }

    /// Compile a C translation unit against the shipped header.
    ///
    /// Returns `None` when no C compiler is available, so the check skips with
    /// a printed reason rather than passing silently on a host without one.
    #[cfg(not(windows))]
    fn cc_syntax_check(src: &str) -> Option<std::process::Output> {
        let include = concat!(env!("CARGO_MANIFEST_DIR"), "/include");
        // The shared tensor ABI header lives in tensor-capi; detect.h lives
        // in decoder-abi. Both install into ${includedir}/edgefirst/, so a
        // consumer needs one -I; in-tree they are three directories.
        let abi_include = concat!(env!("CARGO_MANIFEST_DIR"), "/../tensor-capi/include");
        let detect_include = concat!(env!("CARGO_MANIFEST_DIR"), "/../decoder-abi/include");
        let cc = std::env::var("CC").unwrap_or_else(|_| "cc".to_string());
        match std::process::Command::new(&cc)
            // -Werror: a header that merely *compiles* is not enough. The
            // vtable once declared `struct ef_tensor *` before the type
            // existed, so every dispatch warned about incompatible
            // pointer types while still exiting 0.
            .args([
                "-fsyntax-only",
                "-Werror",
                "-Wall",
                "-I",
                include,
                "-I",
                abi_include,
                "-I",
                detect_include,
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

    /// This leaf's sources, and the tensor leaf's, which the artifacts read
    /// here are checked against. Each library is checked against ITS OWN
    /// crate's sources: comparing libedgefirst_tensor against image-capi/src
    /// would report staleness every time this crate is edited, which is
    /// noise, not a signal.
    const SRC_DIR: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/src");
    const TENSOR_SRC_DIR: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../tensor-capi/src");

    /// The staticlib this platform builds and the prefix every artifact of
    /// this library shares.
    ///
    /// MSVC drops the `lib` prefix and names the staticlib `.lib`, so these
    /// are the platform's spellings, not POSIX's: looking for the POSIX
    /// names on Windows found nothing and skipped every run.
    #[cfg(not(windows))]
    const ARTIFACT_PREFIX: &str = "libedgefirst_image";
    #[cfg(windows)]
    const ARTIFACT_PREFIX: &str = "edgefirst_image";
    #[cfg(not(windows))]
    const STATICLIB: &str = "libedgefirst_image.a";
    #[cfg(windows)]
    const STATICLIB: &str = "edgefirst_image.lib";

    /// A built artifact of this leaf, fresh, or the reason there is none.
    fn find_artifact(name: &str) -> Result<std::path::PathBuf, String> {
        artifacts::find_artifact(name, std::path::Path::new(SRC_DIR))
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

    #[test]
    fn the_shipped_artifact_is_libedgefirst_image_not_capi() {
        // REQUIRED: consumers link `-ledgefirst-image`. The `-capi` suffix
        // names the source crate only, mirroring the `python-*` siblings, and
        // must never reach a filename. A `[lib] name` typo here would ship
        // libedgefirst_image_capi.so and break every consumer's link line.
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

    /// `ef_*` symbols the built shared library `stem` exports, checked
    /// against `src_dir` for freshness.
    fn ef_symbols_of(stem: &str, src_dir: &str) -> Result<Vec<String>, String> {
        let lib = artifacts::find_cdylib(stem, std::path::Path::new(src_dir))?;
        let mut v: Vec<String> = artifacts::symbols_of(&lib)?
            .into_iter()
            .filter(|s| s.starts_with("ef_"))
            .collect();
        v.sort();
        v.dedup();
        Ok(v)
    }

    #[test]
    fn the_two_libraries_export_no_symbol_in_common() {
        // The interposition hazard task 9's dynamic-linking switch avoids: if
        // both libraries exported `ef_tensor_free`, the dynamic linker would
        // bind every caller to one of them, and a tensor minted by the other
        // would be freed by the wrong implementation. One library alone
        // could never demonstrate this -- it takes both linked into one
        // binary to prove there is nothing left for the linker to interpose.
        let img = ef_symbols_of("edgefirst_image", SRC_DIR);
        let tsr = ef_symbols_of("edgefirst_tensor", TENSOR_SRC_DIR);
        let (Ok(img), Ok(tsr)) = (&img, &tsr) else {
            // Which library, and why: the old message blamed the build for
            // every cause, including a Windows host where it looked for
            // libedgefirst_*.so that MSVC never produces.
            let reason = [img.as_ref().err(), tsr.as_ref().err()]
                .into_iter()
                .flatten()
                .cloned()
                .collect::<Vec<_>>()
                .join("; ");
            skip(&format!(
                "the two libraries' exports not compared: {reason}"
            ));
            return;
        };
        assert!(
            !img.is_empty() && !tsr.is_empty(),
            "both must export something"
        );

        let shared: Vec<&String> = img.iter().filter(|s| tsr.contains(s)).collect();
        assert!(
            shared.is_empty(),
            "the two libraries export {} symbol(s) in common, which the dynamic \
             linker would resolve to one of them for every caller: {shared:?}",
            shared.len()
        );

        // Sharper: image must export no tensor constructor at all. It mints
        // tensors (via `TensorDyn::into_raw`), but the implementation, and
        // every exported `ef_tensor_*` accessor including `ef_tensor_free`,
        // lives solely in `libedgefirst_tensor.so`.
        assert!(
            !img.iter().any(|s| s.starts_with("ef_tensor_")),
            "image must not export any ef_tensor_* symbol; found {:?}",
            img.iter()
                .filter(|s| s.starts_with("ef_tensor_"))
                .collect::<Vec<_>>()
        );
    }
    /// Compile, link against the staticlib, and run a C program.
    ///
    /// Linking matters: a syntax-only pass cannot compare a size Rust computed
    /// against one C computed, and it would not notice a missing symbol either.
    #[cfg(not(windows))]
    fn cc_build_and_run(src: &str, bin_name: &str) -> Option<std::process::Output> {
        let include = concat!(env!("CARGO_MANIFEST_DIR"), "/include");
        let (lib, tensor_lib) = match (
            find_artifact(STATICLIB),
            artifacts::find_artifact(
                "libedgefirst_tensor.a",
                std::path::Path::new(TENSOR_SRC_DIR),
            ),
        ) {
            (Ok(lib), Ok(tensor_lib)) => (lib, tensor_lib),
            (image, tensor) => {
                let reason = [image.err(), tensor.err()]
                    .into_iter()
                    .flatten()
                    .collect::<Vec<_>>()
                    .join("; ");
                skip(&format!("{src} not linked: {reason}"));
                return None;
            }
        };
        let out_bin = std::env::temp_dir().join(bin_name);
        let cc = std::env::var("CC").unwrap_or_else(|_| "cc".to_string());
        let mut cmd = std::process::Command::new(&cc);
        cmd.args([
            "-Werror",
            "-Wall",
            "-I",
            include,
            "-I",
            concat!(env!("CARGO_MANIFEST_DIR"), "/../tensor-capi/include"),
            "-I",
            concat!(env!("CARGO_MANIFEST_DIR"), "/../decoder-abi/include"),
            src,
        ])
        .arg(&lib)
        .arg(&tensor_lib)
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
                "C link against libedgefirst_image FAILED (the toolchain is \
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

    // MSVC discovery, shared with tensor-capi where it lives; included at the
    // crate root for the same reason as `artifacts`.
    #[cfg(windows)]
    use crate::msvc;

    /// The three include directories the POSIX helpers pass with `-I`.
    #[cfg(windows)]
    const INCLUDES: [&str; 3] = [
        concat!(env!("CARGO_MANIFEST_DIR"), "/include"),
        concat!(env!("CARGO_MANIFEST_DIR"), "/../tensor-capi/include"),
        concat!(env!("CARGO_MANIFEST_DIR"), "/../decoder-abi/include"),
    ];

    /// The Windows twin of `cc_syntax_check`: `cl.exe /Zs` parses without
    /// generating code, and `/W4 /WX` stands in for `-Wall -Werror`.
    #[cfg(windows)]
    fn cc_syntax_check(src: &str) -> Option<std::process::Output> {
        Some(msvc::require(src)?.syntax_check(&INCLUDES, &[], src))
    }

    /// The Windows twin of `cc_build_and_run`.
    ///
    /// Links `edgefirst_image.lib` with the tensor DLL's import library and
    /// never the tensor staticlib, as `build.rs` does for this library
    /// itself: a second copy of the Rust runtime fails the link with
    /// duplicate symbols. The binary therefore loads `edgefirst_tensor.dll`
    /// at start, so `target/debug` is put on its PATH.
    #[cfg(windows)]
    fn cc_build_and_run(src: &str, bin_name: &str) -> Option<std::process::Output> {
        let (lib, tensor_import_lib) = match (
            find_artifact(STATICLIB),
            artifacts::find_artifact(
                "edgefirst_tensor.dll.lib",
                std::path::Path::new(TENSOR_SRC_DIR),
            ),
        ) {
            (Ok(lib), Ok(tensor_import_lib)) => (lib, tensor_import_lib),
            (image, tensor) => {
                let reason = [image.err(), tensor.err()]
                    .into_iter()
                    .flatten()
                    .collect::<Vec<_>>()
                    .join("; ");
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
        let libs = [lib.as_os_str(), tensor_import_lib.as_os_str()];
        let out = toolchain.build(&INCLUDES, src, &exe, &libs);
        // The command line and cl's own stdout diagnostics are folded into
        // stderr by the harness, so this message carries both.
        assert!(
            out.status.success(),
            "C link against edgefirst_image.lib FAILED (the toolchain is \
             known good, so this is a real defect):\n{}",
            String::from_utf8_lossy(&out.stderr)
        );
        // edgefirst_tensor.dll sits beside the import library the binary was
        // linked against, so the loader finds it there.
        let dll_dir = tensor_import_lib
            .parent()
            .expect("find_artifact returns a path under a directory")
            .to_path_buf();
        let inherited = std::env::var_os("PATH").unwrap_or_default();
        let path =
            std::env::join_paths(std::iter::once(dll_dir).chain(std::env::split_paths(&inherited)))
                .expect("PATH entries split on the separator contain none");
        let mut cmd = std::process::Command::new(&exe);
        cmd.env("PATH", path);
        run_program(&exe, cmd)
    }

    #[test]
    fn a_tensor_crosses_the_library_boundary_intact() {
        // Links BOTH libraries into one binary: mint with image, read and free
        // through libedgefirst-tensor, and assert the free landed back in image.
        // A wrong free is silent heap corruption, so the test checks a counter
        // rather than merely surviving the call.
        let src = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/c/test_cross_library.c");
        let Some(out) = cc_build_and_run(src, "ef_test_cross_library") else {
            return;
        };
        assert!(
            out.status.success(),
            "cross-library contract failed:\n{}{}",
            String::from_utf8_lossy(&out.stdout),
            String::from_utf8_lossy(&out.stderr)
        );
    }

    #[test]
    fn an_image_desc_request_crosses_the_library_boundary_intact() {
        // The other half of the boundary crossing `a_tensor_crosses_the_library_
        // boundary_intact` checks: an `ef_tensor_image_desc` is minted and read
        // entirely by libedgefirst-tensor, and `ef_image_processor_create_image_
        // desc` (libedgefirst-image) only ever sees it through
        // `ef_tensor_image_desc_get`'s scalar view. A single library cannot
        // demonstrate that the handle stays opaque; this links both.
        let src = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/c/test_image_desc_cross_library.c"
        );
        let Some(out) = cc_build_and_run(src, "ef_test_image_desc_cross_library") else {
            return;
        };
        assert!(
            out.status.success(),
            "image-desc cross-library contract failed:\n{}{}",
            String::from_utf8_lossy(&out.stdout),
            String::from_utf8_lossy(&out.stderr)
        );
    }

    #[test]
    fn no_rust_type_name_leaks_into_the_c_api() {
        // cbindgen emits Rust type names verbatim unless told otherwise, so
        // `EfImageProcessor` would become part of the published C API and then
        // be permanent. It also breaks the header outright when the type is
        // excluded (as `EfTensor` is, since it comes from tensor.h).
        let h = header_text();
        for rust_name in [
            "EfTensor",
            "EfImageProcessor",
            "EfTensorImpl",
            "EfMaskList",
            "EfTileSpecList",
            "EfTilePlacementList",
        ] {
            assert!(
                !h.contains(rust_name),
                "the Rust type name `{rust_name}` leaked into image.h; add it to \
                 cbindgen.toml's [export.rename]"
            );
        }
    }

    #[test]
    fn image_h_includes_the_tensor_header() {
        // Callers of this library handle tensors, so image.h must bring the
        // tensor API with it rather than leaving a caller to discover the
        // missing include.
        assert!(
            header_text().contains(r#"#include "edgefirst/tensor.h""#),
            "image.h must include edgefirst/tensor.h"
        );
    }

    #[test]
    fn the_generated_header_exists_and_declares_the_abi_probe() {
        assert!(header_text().contains("ef_image_abi_version"));
    }

    #[test]
    fn the_header_declares_the_ported_surface() {
        let h = header_text();
        for sym in [
            "ef_image_processor_convert_deferred",
            "ef_image_processor_draw_decoded_masks",
            "ef_image_processor_draw_proto_masks",
            "ef_image_processor_materialize_masks",
            "ef_image_processor_plan_tiles",
            "ef_image_processor_tile_into",
            "ef_image_processor_tile_one",
            "ef_image_processor_alloc_tile_batch",
            "ef_align_width_for_gpu_pitch",
            "ef_align_width_for_pixel_format",
            "ef_gpu_dma_buf_pitch_alignment_bytes",
            "ef_tile_grid",
            "ef_tile_spec_list_len",
            "ef_tile_placement_list_len",
        ] {
            assert!(
                h.contains(sym),
                "image.h must declare `{sym}` (scan this crate's header, not the old monolith)"
            );
        }
    }

    #[test]
    fn the_header_carries_none_of_the_monoliths_symbols() {
        // If `hal_` appears here the crate picked up the old monolith's
        // surface and the split has not actually happened.
        assert!(
            !header_text().contains("hal_"),
            "image.h must not carry the monolith's hal_* symbols"
        );
    }

    #[test]
    fn the_header_survives_being_included_twice() {
        // cbindgen's own `include_guard` wraps only the generated body, leaving
        // header/trailer exposed to redefinition. The guard here is hand-written
        // for that reason; this is the gate.
        let src = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/c/test_double_include.c");
        let Some(out) = cc_syntax_check(src) else {
            return;
        };
        assert!(
            out.status.success(),
            "image.h failed to compile when included twice:\n{}",
            String::from_utf8_lossy(&out.stderr)
        );
    }
}
