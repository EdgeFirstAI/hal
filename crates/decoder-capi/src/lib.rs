// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! The C API for `edgefirst-decoder`: model output to detections and masks.
//!
//! Produces `ef_detect_box_list`. `ef_detect_box_list_data()`/`_len()` give a
//! plain `(ef_detect_box *, size_t)` view of a list.

#[cfg(not(edgefirst_tensor_dynamic))]
compile_error!(
    "edgefirst-decoder-capi links libedgefirst_tensor and must be built with the dynamic tensor backend: \
     build it from crates/decoder-capi, whose .cargo/config.toml selects it (as `make capi-libs` does)."
);

pub mod codes;
pub mod decode;
pub mod infer;
pub mod model_config;
pub mod semantic;
pub mod tiling;

#[cfg(test)]
mod test_alloc;

/// ABI version of this library's C surface.
#[no_mangle]
pub extern "C" fn ef_decoder_abi_version() -> u32 {
    4
}

#[cfg(test)]
mod tests {
    fn header_text() -> String {
        std::fs::read_to_string(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/include/edgefirst/decoder.h"
        ))
        .expect("cbindgen wrote include/edgefirst/decoder.h")
    }

    /// Syntax-checks `src` against the headers with `-Werror -Wall`. `None`
    /// when no C compiler is available.
    fn cc_syntax_check(src: &str) -> Option<std::process::Output> {
        let cc = std::env::var("CC").unwrap_or_else(|_| "cc".to_string());
        match std::process::Command::new(&cc)
            .args([
                "-fsyntax-only",
                "-Werror",
                "-Wall",
                "-I",
                concat!(env!("CARGO_MANIFEST_DIR"), "/include"),
                "-I",
                concat!(env!("CARGO_MANIFEST_DIR"), "/../tensor-capi/include"),
                "-I",
                concat!(env!("CARGO_MANIFEST_DIR"), "/../decoder-abi/include"),
                src,
            ])
            .output()
        {
            Ok(o) => Some(o),
            Err(e) => {
                use std::io::Write;
                let _ = writeln!(
                    std::io::stderr(),
                    "WARNING: SKIPPED: no C compiler ({cc}: {e})"
                );
                None
            }
        }
    }

    /// Like [`cc_syntax_check`], but pins `-std=c11`.
    fn cc_syntax_check_c11(src: &str) -> Option<std::process::Output> {
        let cc = std::env::var("CC").unwrap_or_else(|_| "cc".to_string());
        match std::process::Command::new(&cc)
            .args([
                "-std=c11",
                "-fsyntax-only",
                "-Werror",
                "-Wall",
                "-I",
                concat!(env!("CARGO_MANIFEST_DIR"), "/include"),
                "-I",
                concat!(env!("CARGO_MANIFEST_DIR"), "/../tensor-capi/include"),
                "-I",
                concat!(env!("CARGO_MANIFEST_DIR"), "/../decoder-abi/include"),
                src,
            ])
            .output()
        {
            Ok(o) => Some(o),
            Err(e) => {
                use std::io::Write;
                let _ = writeln!(
                    std::io::stderr(),
                    "WARNING: SKIPPED: no C compiler ({cc}: {e})"
                );
                None
            }
        }
    }

    #[test]
    fn the_generated_header_exists_and_declares_the_abi_probe() {
        assert!(header_text().contains("ef_decoder_abi_version"));
    }

    #[test]
    fn the_abi_probe_reports_the_current_version() {
        assert_eq!(super::ef_decoder_abi_version(), 4);
    }

    #[test]
    fn the_header_carries_no_hal_symbols() {
        assert!(
            !header_text().contains("hal_"),
            "decoder.h must not carry hal_* symbols"
        );
    }

    #[test]
    fn decoder_h_includes_the_tensor_header() {
        assert!(
            header_text().contains(r#"#include "edgefirst/tensor.h""#),
            "decoder.h must include edgefirst/tensor.h"
        );
    }

    #[test]
    fn no_rust_type_name_leaks_into_the_c_api() {
        let h = header_text();
        for rust_name in [
            "EfByteTrack",
            "EfTrackInfoList",
            "EfTrackInfo",
            "EfDetectBoxList",
            "EfProtoData",
            "EfDecoderTracker",
            "EfDecoderTrackList",
            "EfSemanticOptions",
            "EfInferSignals",
            "EfModelOverrides",
            "EfModelConfig",
            "EfPreprocessSpec",
            "EfConfigSource",
            "EfModelFamily",
            "EfModelTask",
            "EfConfigField",
        ] {
            assert!(
                !h.contains(rust_name),
                "the Rust type name `{rust_name}` leaked into decoder.h; add it to \
                 cbindgen.toml's [export.rename]"
            );
        }
    }

    #[test]
    fn the_preprocess_vocabulary_is_declared_only_in_detect_h() {
        let h = header_text();
        for ty in [
            "enum ef_fit",
            "enum ef_letterbox_placement",
            "enum ef_pad_space",
            "enum ef_interpolation",
            "struct ef_preprocess_spec {",
        ] {
            assert!(
                !h.contains(ty),
                "`{ty}` belongs to detect.h; decoder.h must not redeclare it"
            );
        }
    }

    #[test]
    fn the_layout_goldens_hold() {
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
    fn the_header_survives_being_included_twice() {
        let src = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/c/test_double_include.c");
        let Some(out) = cc_syntax_check(src) else {
            return;
        };
        assert!(
            out.status.success(),
            "decoder.h failed to compile when included twice:\n{}",
            String::from_utf8_lossy(&out.stderr)
        );
    }

    #[test]
    fn the_shipped_artifact_is_libedgefirst_decoder_not_capi() {
        let mut dir = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        dir.pop();
        dir.pop();
        dir.push("target");
        dir.push("debug");
        let found: Vec<String> = std::fs::read_dir(&dir)
            .map(|rd| {
                rd.filter_map(|e| e.ok())
                    .map(|e| e.file_name().to_string_lossy().to_string())
                    .filter(|n| n.starts_with("libedgefirst_decoder"))
                    .collect()
            })
            .unwrap_or_default();
        if found.is_empty() {
            use std::io::Write;
            let _ = writeln!(
                std::io::stderr(),
                "WARNING: SKIPPED: nothing built; name not checked"
            );
            return;
        }
        assert!(
            !found.iter().any(|n| n.contains("_capi")),
            "must ship as libedgefirst_decoder.*, never ..._capi; found {found:?}. \
             If `[lib] name` is right these are STALE artifacts from a previous \
             name — cargo does not remove them on rename."
        );
    }
}
