// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! The C API for `edgefirst-image`.
//!
//! Ships as `libedgefirst_image` with `edgefirst/image.h`.
//!
//! `ef_image_processor_create_image` returns an `ef_tensor`; callers read it
//! with `ef_tensor_shape` and release it with `ef_tensor_free`.
//! `edgefirst/image.h` includes `edgefirst/tensor.h` and the pkg-config
//! declares `Requires: edgefirst-tensor`.
//!
//! This crate links `libedgefirst_tensor.so` dynamically and uses the
//! declarations in `edgefirst-tensor-ffi` to create and read tensors.

#[cfg(not(edgefirst_tensor_dynamic))]
compile_error!(
    "edgefirst-image-capi links libedgefirst_tensor and must be built with the dynamic tensor backend: \
     build it from crates/image-capi, whose .cargo/config.toml selects it (as `make capi-libs` does)."
);

pub mod codes;
pub mod draw;
pub mod processor;
pub mod tiling;

/// ABI version of this library's C surface.
#[no_mangle]
pub extern "C" fn ef_image_abi_version() -> u32 {
    5
}

// Test-support modules shared with the tensor leaf.
#[cfg(test)]
#[path = "../../tensor-capi/tests/support/artifacts.rs"]
mod artifacts;
#[cfg(all(test, windows))]
#[path = "../../tensor-capi/tests/support/msvc.rs"]
mod msvc;

#[cfg(test)]
mod tests {
    /// The shipped header text.
    fn header_text() -> String {
        std::fs::read_to_string(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/include/edgefirst/image.h"
        ))
        .expect("cbindgen wrote include/edgefirst/image.h")
    }

    /// Compile a C translation unit against the shipped header. `None` when
    /// no C compiler is available.
    #[cfg(not(windows))]
    fn cc_syntax_check(src: &str) -> Option<std::process::Output> {
        let include = concat!(env!("CARGO_MANIFEST_DIR"), "/include");
        // The tensor header is in tensor-capi and detect.h in decoder-abi.
        let abi_include = concat!(env!("CARGO_MANIFEST_DIR"), "/../tensor-capi/include");
        let detect_include = concat!(env!("CARGO_MANIFEST_DIR"), "/../decoder-abi/include");
        let cc = std::env::var("CC").unwrap_or_else(|_| "cc".to_string());
        match std::process::Command::new(&cc)
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
                skip(&format!(
                    "no C compiler ({cc}: {e}); header not syntax-checked"
                ));
                None
            }
        }
    }

    // Artifact discovery, freshness and symbol reading.
    use crate::artifacts;
    use artifacts::skip;

    /// This leaf's sources and the tensor leaf's, which the artifacts are checked against.
    const SRC_DIR: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/src");
    const TENSOR_SRC_DIR: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../tensor-capi/src");

    /// The staticlib this platform builds and the prefix every artifact of
    /// this library shares.
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

    /// Runs a built C program. A spawn failure is reported as a skip and returns `None`.
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
        if let Err(reason) = find_artifact(STATICLIB) {
            skip(&format!("artifact name not checked: {reason}"));
            return;
        }
        // Artifacts in every candidate directory.
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
        let img = ef_symbols_of("edgefirst_image", SRC_DIR);
        let tsr = ef_symbols_of("edgefirst_tensor", TENSOR_SRC_DIR);
        let (Ok(img), Ok(tsr)) = (&img, &tsr) else {
            // Reports which library was not found and why.
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

        assert!(
            !img.iter().any(|s| s.starts_with("ef_tensor_")),
            "image must not export any ef_tensor_* symbol; found {:?}",
            img.iter()
                .filter(|s| s.starts_with("ef_tensor_"))
                .collect::<Vec<_>>()
        );
    }
    /// Compile, link against the staticlib, and run a C program.
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
        // Platform libraries the staticlib needs.
        if cfg!(target_os = "macos") {
            cmd.args(["-framework", "CoreFoundation", "-framework", "IOSurface"]);
        } else {
            cmd.args(["-lpthread", "-ldl", "-lm"]);
        }
        // Probes the toolchain with a trivial translation unit that does not touch this library.
        if !toolchain_works(&cc) {
            skip(&format!("{src} not linked: no working C toolchain ({cc})"));
            return None;
        }
        match cmd.output() {
            Ok(o) if !o.status.success() => panic!(
                "C link against libedgefirst_image FAILED (the toolchain is \
                 known good, so this is a real defect):\n{}",
                String::from_utf8_lossy(&o.stderr)
            ),
            Ok(_) => run_program(&out_bin, std::process::Command::new(&out_bin)),
            Err(e) => panic!("failed to run {cc} after it probed OK: {e}"),
        }
    }

    /// Whether this host can compile and link a trivial C program.
    #[cfg(not(windows))]
    fn toolchain_works(cc: &str) -> bool {
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

    // MSVC discovery, shared with tensor-capi.
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
    /// generating code, with `/W4 /WX`.
    #[cfg(windows)]
    fn cc_syntax_check(src: &str) -> Option<std::process::Output> {
        Some(msvc::require(src)?.syntax_check(&INCLUDES, &[], src))
    }

    /// The Windows twin of `cc_build_and_run`.
    ///
    /// Links `edgefirst_image.lib` with the tensor DLL's import library and
    /// puts the DLL's directory on the program's PATH.
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
        // `None` when the toolchain cannot build a trivial program.
        let toolchain = msvc::require(src)?;
        let exe = toolchain.scratch().join(format!("{bin_name}.exe"));
        let libs = [lib.as_os_str(), tensor_import_lib.as_os_str()];
        let out = toolchain.build(&INCLUDES, src, &exe, &libs);
        assert!(
            out.status.success(),
            "C link against edgefirst_image.lib FAILED (the toolchain is \
             known good, so this is a real defect):\n{}",
            String::from_utf8_lossy(&out.stderr)
        );
        // edgefirst_tensor.dll sits beside the import library.
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
        let src = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/c/test_cross_library.c");
        let Some(out) = cc_build_and_run(src, "ef_test_cross_library") else {
            return;
        };
        artifacts::relay_skips("a_tensor_crosses_the_library_boundary_intact", &out);
        assert!(
            out.status.success(),
            "cross-library contract failed:\n{}{}",
            String::from_utf8_lossy(&out.stdout),
            String::from_utf8_lossy(&out.stderr)
        );
    }

    #[test]
    fn an_image_desc_request_crosses_the_library_boundary_intact() {
        let src = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/c/test_image_desc_cross_library.c"
        );
        let Some(out) = cc_build_and_run(src, "ef_test_image_desc_cross_library") else {
            return;
        };
        artifacts::relay_skips(
            "an_image_desc_request_crosses_the_library_boundary_intact",
            &out,
        );
        assert!(
            out.status.success(),
            "image-desc cross-library contract failed:\n{}{}",
            String::from_utf8_lossy(&out.stdout),
            String::from_utf8_lossy(&out.stderr)
        );
    }

    #[test]
    fn no_rust_type_name_leaks_into_the_c_api() {
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
    fn the_abi_probe_reports_the_current_version() {
        assert_eq!(super::ef_image_abi_version(), 5);
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
            "ef_convert_options_from_spec",
            "ef_convert_options_set_interpolation",
            "ef_convert_options_content_region",
        ] {
            assert!(h.contains(sym), "image.h must declare `{sym}`");
        }
    }

    #[test]
    fn the_header_carries_no_hal_symbols() {
        assert!(
            !header_text().contains("hal_"),
            "image.h must not carry hal_* symbols"
        );
    }

    #[test]
    fn the_header_survives_being_included_twice() {
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
