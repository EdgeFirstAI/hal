// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! The C API for `edgefirst-codec`: JPEG and PNG decode, and encode.
//!
//! Decoding writes **into** a tensor the caller already has rather than
//! allocating one. That is what lets a decode target be a DMA buffer the GPU
//! will read next, with no copy in between — and it is why the tensor may have
//! been minted by any EdgeFirst library, not only this one.

pub mod decoder;

/// ABI version of this library's C surface.
#[no_mangle]
pub extern "C" fn ef_codec_abi_version() -> u32 {
    1
}

#[cfg(test)]
mod tests {
    fn header_text() -> String {
        std::fs::read_to_string(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/include/edgefirst/codec.h"
        ))
        .expect("cbindgen wrote include/edgefirst/codec.h")
    }

    /// Syntax-check a C translation unit against the shipped headers.
    ///
    /// `-Werror`: a header that merely parses is not enough — the vtable once
    /// declared `struct ef_tensor *` before the type existed and warned on
    /// every dispatch while still exiting 0.
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

    /// `ef_*` symbols a built library exports, or `None` if it is absent.
    fn ef_symbols_of(lib: &std::path::Path) -> Option<Vec<String>> {
        if !lib.exists() {
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
    fn no_two_libraries_export_the_same_symbol() {
        // With five libraries the interposition hazard is at its sharpest: any
        // symbol two of them export would be resolved to one for every caller,
        // and a tensor from the other would then be handled by the wrong code.
        // Checked pairwise across all three rather than against tensor alone.
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
        let libs = ["tensor", "image", "codec", "tracker", "decoder"];
        let mut sets = Vec::new();
        for name in libs {
            match ef_symbols_of(&dir.join(format!("libedgefirst_{name}.{ext}"))) {
                Some(v) => sets.push((name, v)),
                None => {
                    use std::io::Write;
                    let _ = writeln!(
                        std::io::stderr(),
                        "SKIP: libedgefirst_{name} not built; collision check not run"
                    );
                    return;
                }
            }
        }
        for i in 0..sets.len() {
            for j in (i + 1)..sets.len() {
                let (an, a) = &sets[i];
                let (bn, b) = &sets[j];
                let shared: Vec<&String> = a.iter().filter(|s| b.contains(s)).collect();
                assert!(
                    shared.is_empty(),
                    "libedgefirst_{an} and libedgefirst_{bn} both export {shared:?}"
                );
            }
        }
        // Sharper still: only the tensor library defines the tensor API. The
        // others mint and consume tensors but export none of its entry points.
        for (name, syms) in &sets {
            if *name == "tensor" {
                continue;
            }
            assert!(
                !syms.iter().any(|s| s.starts_with("ef_tensor_")),
                "libedgefirst_{name} must export no ef_tensor_* symbol; found {:?}",
                syms.iter()
                    .filter(|s| s.starts_with("ef_tensor_"))
                    .collect::<Vec<_>>()
            );
        }
    }

    #[test]
    fn the_generated_header_exists_and_declares_the_abi_probe() {
        assert!(header_text().contains("ef_codec_abi_version"));
    }

    #[test]
    fn the_header_carries_none_of_the_monoliths_symbols() {
        assert!(
            !header_text().contains("hal_"),
            "codec.h must not carry the monolith's hal_* symbols"
        );
    }

    #[test]
    fn codec_h_includes_the_tensor_header() {
        // Decoding targets a tensor, so codec.h must bring the tensor API with
        // it rather than leaving a caller to find the missing include.
        assert!(
            header_text().contains(r#"#include "edgefirst/tensor.h""#),
            "codec.h must include edgefirst/tensor.h"
        );
    }

    #[test]
    fn no_rust_type_name_leaks_into_the_c_api() {
        let h = header_text();
        for rust_name in ["EfTensor", "EfImageDecoder", "EfTensorImpl"] {
            assert!(
                !h.contains(rust_name),
                "the Rust type name `{rust_name}` leaked into codec.h; add it to \
                 cbindgen.toml's [export.rename]"
            );
        }
    }

    #[test]
    fn the_header_survives_being_included_twice() {
        let src = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/c/test_double_include.c");
        let Some(out) = cc_syntax_check(src) else {
            return;
        };
        assert!(
            out.status.success(),
            "codec.h failed to compile when included twice:\n{}",
            String::from_utf8_lossy(&out.stderr)
        );
    }

    #[test]
    fn the_shipped_artifact_is_libedgefirst_codec_not_capi() {
        // Consumers link `-ledgefirst-codec`. The `-capi` suffix names the
        // source crate only and must never reach a filename.
        let mut dir = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        dir.pop();
        dir.pop();
        dir.push("target");
        dir.push("debug");
        let found: Vec<String> = std::fs::read_dir(&dir)
            .map(|rd| {
                rd.filter_map(|e| e.ok())
                    .map(|e| e.file_name().to_string_lossy().to_string())
                    .filter(|n| n.starts_with("libedgefirst_codec"))
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
            "must ship as libedgefirst_codec.*, never libedgefirst_codec_capi.*; \
             found {found:?}. If `[lib] name` is right these are STALE artifacts \
             from a previous name — cargo does not remove them on rename."
        );
    }
}
