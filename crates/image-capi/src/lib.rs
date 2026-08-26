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

    #[test]
    fn the_shipped_artifact_is_libedgefirst_image_not_capi() {
        // REQUIRED: consumers link `-ledgefirst-image`. The `-capi` suffix
        // names the source crate only, mirroring the `python-*` siblings, and
        // must never reach a filename. A `[lib] name` typo here would ship
        // libedgefirst_image_capi.so and break every consumer's link line.
        let mut dir = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        dir.pop();
        dir.pop();
        dir.push("target");
        dir.push("debug");
        let found: Vec<String> = std::fs::read_dir(&dir)
            .map(|rd| {
                rd.filter_map(|e| e.ok())
                    .map(|e| e.file_name().to_string_lossy().to_string())
                    .filter(|n| n.starts_with("libedgefirst_image"))
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
            "the C library must ship as libedgefirst_image.*, never \
             libedgefirst_image_capi.*; found {found:?}.\n\
             If `[lib] name` is already correct, these are STALE artifacts from \
             a previous name -- cargo does not remove them on rename, and an \
             install or link step can still pick the wrong file up. \
             Delete target/debug/libedgefirst_image_capi.* or `cargo clean`."
        );
        assert!(
            found.iter().any(|n| n == "libedgefirst_image.a"),
            "expected libedgefirst_image.a among {found:?}"
        );
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
    /// order is guaranteed (the Makefile's C-library target, CI) to turn the
    /// skip into a hard failure.
    fn artifact_is_fresh_vs(artifact: &std::path::Path, src_dir: &std::path::Path) -> bool {
        let art = match std::fs::metadata(artifact).and_then(|m| m.modified()) {
            Ok(t) => t,
            Err(_) => return false,
        };
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
             `cargo build` the owning crate first.",
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

    /// `ef_*` symbols a built library exports.
    fn ef_symbols_of(lib: &std::path::Path, src_dir: &std::path::Path) -> Option<Vec<String>> {
        // Each library is checked against ITS OWN crate's sources. Comparing
        // libedgefirst_tensor against image-capi/src would report staleness
        // every time this crate is edited, which is noise, not a signal.
        if !lib.exists() || !artifact_is_fresh_vs(lib, src_dir) {
            return None;
        }
        let args: &[&str] = if cfg!(target_os = "macos") {
            &["-gU"]
        } else {
            &["-D", "--defined-only"]
        };
        let out = std::process::Command::new("nm")
            .args(args)
            .arg(lib)
            .output()
            .ok()?;
        if !out.status.success() {
            return None;
        }
        let mut v: Vec<String> = String::from_utf8_lossy(&out.stdout)
            .lines()
            .filter_map(|l| l.split_whitespace().last())
            .map(|s| s.trim_start_matches('_').to_string())
            .filter(|s| s.starts_with("ef_"))
            .collect();
        v.sort();
        v.dedup();
        Some(v)
    }

    #[test]
    fn the_two_libraries_export_no_symbol_in_common() {
        // The interposition hazard task 9's dynamic-linking switch avoids: if
        // both libraries exported `ef_tensor_free`, the dynamic linker would
        // bind every caller to one of them, and a tensor minted by the other
        // would be freed by the wrong implementation. One library alone
        // could never demonstrate this -- it takes both linked into one
        // binary to prove there is nothing left for the linker to interpose.
        let mut dir = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        dir.pop();
        dir.pop();
        dir.push("target");
        dir.push("debug");
        let ext = if cfg!(target_os = "macos") {
            "dylib"
        } else {
            "so"
        };
        let (Some(img), Some(tsr)) = (
            ef_symbols_of(
                &dir.join(format!("libedgefirst_image.{ext}")),
                std::path::Path::new(concat!(env!("CARGO_MANIFEST_DIR"), "/src")),
            ),
            ef_symbols_of(
                &dir.join(format!("libedgefirst_tensor.{ext}")),
                std::path::Path::new(concat!(env!("CARGO_MANIFEST_DIR"), "/../tensor-capi/src")),
            ),
        ) else {
            use std::io::Write;
            let _ = writeln!(
                std::io::stderr(),
                "SKIP: both libraries must be built and current; \
                 run `cargo build -p edgefirst-tensor-capi -p edgefirst-image-capi`"
            );
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
    fn cc_build_and_run(src: &str, bin_name: &str) -> Option<std::process::Output> {
        let include = concat!(env!("CARGO_MANIFEST_DIR"), "/include");
        let mut dir = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        dir.pop();
        dir.pop();
        dir.push("target");
        dir.push("debug");
        let lib = dir.join("libedgefirst_image.a");
        let tensor_lib = dir.join("libedgefirst_tensor.a");
        if !lib.exists() {
            eprintln!("SKIP: {} not built yet; C link test not run", lib.display());
            return None;
        }
        // The staticlib has the same staleness exposure as the cdylib: linking
        // C against a stale .a validates the previous build's symbols.
        if !artifact_is_fresh_vs(
            &lib,
            std::path::Path::new(concat!(env!("CARGO_MANIFEST_DIR"), "/src")),
        ) {
            return None;
        }
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
                "C link against libedgefirst_image FAILED (the toolchain is \
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
