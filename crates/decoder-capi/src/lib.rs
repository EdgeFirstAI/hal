// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! The C API for `edgefirst-decoder`: model output to detections and masks.
//!
//! Produces `ef_detect_box_list`, declared here — this is its one
//! implementation home. A consumer such as the tracker reads a list through
//! the exported `ef_detect_box_list_data()`/`_len()` without linking this
//! library: those give back a plain `(ef_detect_box *, size_t)` view.

pub mod decode;
pub mod infer;
pub mod tiling;

/// ABI version of this library's C surface.
#[no_mangle]
pub extern "C" fn ef_decoder_abi_version() -> u32 {
    1
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

    /// `-Werror`: a header that merely parses is not enough. The vtable once
    /// declared a type before it existed and warned on every dispatch while
    /// still exiting 0.
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
                let _ = writeln!(std::io::stderr(), "SKIP: no C compiler ({cc}: {e})");
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
                let _ = writeln!(std::io::stderr(), "SKIP: no C compiler ({cc}: {e})");
                None
            }
        }
    }

    #[test]
    fn the_generated_header_exists_and_declares_the_abi_probe() {
        assert!(header_text().contains("ef_decoder_abi_version"));
    }

    #[test]
    fn the_header_carries_none_of_the_monoliths_symbols() {
        assert!(
            !header_text().contains("hal_"),
            "decoder.h must not carry the monolith's hal_* symbols"
        );
    }

    #[test]
    fn decoder_h_includes_the_tensor_header() {
        // The detection types it takes are declared there.
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
        ] {
            assert!(
                !h.contains(rust_name),
                "the Rust type name `{rust_name}` leaked into decoder.h; add it to \
                 cbindgen.toml's [export.rename]"
            );
        }
    }

    #[test]
    fn the_layout_goldens_hold() {
        // By-value structs in `detect.h` (`ef_detect_box`, `ef_segmentation`,
        // `ef_merge_config`, `ef_tile_placement`) are the drift class no
        // name-level check can see, and are identical on every LP64 target
        // this header ships for.
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
            let _ = writeln!(std::io::stderr(), "SKIP: nothing built; name not checked");
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
