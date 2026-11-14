// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! The C API for `edgefirst-codec`: still images (JPEG and PNG decode, JPEG
//! encode) and video (H.264 and MJPEG encode and decode), with tensors as the
//! pictures throughout.
//!
//! Image decoding writes **into** a tensor the caller already has rather than
//! allocating one. That is what lets a decode target be a DMA buffer the GPU
//! will read next, with no copy in between — and it is why the tensor may have
//! been minted by any EdgeFirst library, not only this one. Video decoders
//! hand out frames whose tensors stay in the decoder's pool until the frame
//! is freed.

#[cfg(not(edgefirst_tensor_dynamic))]
compile_error!(
    "edgefirst-codec-capi links libedgefirst_tensor and must be built with the dynamic tensor backend: \
     build it from crates/codec-capi, whose .cargo/config.toml selects it (as `make capi-libs` does)."
);

pub mod decoder;
pub mod encoder;
pub mod engine;
pub mod error;
pub mod video;

/// ABI version of this library's C surface.
///
/// 2: `ef_codec_colorimetry_from_v4l2` takes an `is_rgb_or_hsv` argument
/// before `out`.
///
/// 3: engines, capabilities, `ef_image_encoder`, `ef_video_encoder` and
/// `ef_video_decoder`; `ef_codec_last_error`; image decoding answers
/// `EBADMSG` for undecodable data and `ENOTSUP` for unsupported features,
/// which were `EIO`.
#[no_mangle]
pub extern "C" fn ef_codec_abi_version() -> u32 {
    3
}

// Test support shared with the other C-API leaves, which lives in the tensor
// leaf. Included here, relative to `src/`: a `#[path]` inside an inline
// module resolves relative to `src/tests/`, which Linux and macOS refuse to
// walk `..` through.
// This leaf uses part of each; the other leaves use the rest.
#[cfg(test)]
#[path = "../../tensor-capi/tests/support/artifacts.rs"]
#[allow(dead_code)]
mod artifacts;
#[cfg(all(test, windows))]
#[path = "../../tensor-capi/tests/support/msvc.rs"]
#[allow(dead_code)]
mod msvc;

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
                let _ = writeln!(
                    std::io::stderr(),
                    "WARNING: SKIPPED: no C compiler ({cc}: {e})"
                );
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
                        "WARNING: SKIPPED: libedgefirst_{name} not built; collision check not run"
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
    fn the_abi_probe_is_generation_3() {
        assert_eq!(super::ef_codec_abi_version(), 3);
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
        for rust_name in [
            "EfTensor",
            "EfImageDecoder",
            "EfTensorImpl",
            "EfImageEncoder",
            "EfCodec",
            "EfVideo",
            "EfH264",
        ] {
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
            let _ = writeln!(
                std::io::stderr(),
                "WARNING: SKIPPED: nothing built; name not checked"
            );
            return;
        }
        assert!(
            !found.iter().any(|n| n.contains("_capi")),
            "must ship as libedgefirst_codec.*, never libedgefirst_codec_capi.*; \
             found {found:?}. If `[lib] name` is right these are STALE artifacts \
             from a previous name — cargo does not remove them on rename."
        );
    }

    use crate::artifacts;
    use artifacts::skip;

    /// This leaf's sources and the tensor leaf's, which each library's
    /// artifacts are checked against for freshness.
    const SRC_DIR: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/src");
    const TENSOR_SRC_DIR: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../tensor-capi/src");

    #[cfg(not(windows))]
    const STATICLIB: &str = "libedgefirst_codec.a";
    #[cfg(windows)]
    const STATICLIB: &str = "edgefirst_codec.lib";

    /// Runs a built C program. Failing to start one linked a moment ago is
    /// a missing library or a broken image, not a skip.
    fn run_program(
        exe: &std::path::Path,
        mut cmd: std::process::Command,
    ) -> Option<std::process::Output> {
        match cmd.output() {
            Ok(out) => Some(out),
            Err(e) => panic!("{} was built but could not be run: {e}", exe.display()),
        }
    }

    /// This library's staticlib and the tensor library to link with it, or
    /// `None` after reporting why they are not there.
    fn libraries(src: &str, tensor_lib: &str) -> Option<(std::path::PathBuf, std::path::PathBuf)> {
        match (
            artifacts::find_artifact(STATICLIB, std::path::Path::new(SRC_DIR)),
            artifacts::find_artifact(tensor_lib, std::path::Path::new(TENSOR_SRC_DIR)),
        ) {
            (Ok(lib), Ok(tensor)) => Some((lib, tensor)),
            (codec, tensor) => {
                let reason = [codec.err(), tensor.err()]
                    .into_iter()
                    .flatten()
                    .collect::<Vec<_>>()
                    .join("; ");
                skip(&format!("{src} not linked: {reason}"));
                None
            }
        }
    }

    /// Compiles `src`, links it against the staticlibs, and runs it.
    ///
    /// The toolchain is probed with a trivial program first, so a link
    /// failure of this library is a failure and never mistaken for a host
    /// without a compiler.
    #[cfg(not(windows))]
    fn cc_build_and_run(src: &str, bin_name: &str) -> Option<std::process::Output> {
        let (lib, tensor_lib) = libraries(src, "libedgefirst_tensor.a")?;
        let cc = std::env::var("CC").unwrap_or_else(|_| "cc".to_string());
        let pid = std::process::id();
        let probe = std::env::temp_dir().join(format!("ef_codec_cc_probe-{pid}.c"));
        let probe_bin = std::env::temp_dir().join(format!("ef_codec_cc_probe-{pid}.bin"));
        let works = std::fs::write(&probe, b"int main(void){return 0;}\n").is_ok()
            && std::process::Command::new(&cc)
                .arg(&probe)
                .arg("-o")
                .arg(&probe_bin)
                .output()
                .is_ok_and(|o| o.status.success());
        if !works {
            skip(&format!("{src} not linked: no working C toolchain ({cc})"));
            return None;
        }
        let exe = std::env::temp_dir().join(format!("{bin_name}-{pid}"));
        let mut cmd = std::process::Command::new(&cc);
        cmd.args([
            "-std=c11",
            "-Werror",
            "-Wall",
            "-Wextra",
            "-I",
            concat!(env!("CARGO_MANIFEST_DIR"), "/include"),
            "-I",
            concat!(env!("CARGO_MANIFEST_DIR"), "/../tensor-capi/include"),
            src,
        ])
        .arg(&lib)
        .arg(&tensor_lib)
        .arg("-o")
        .arg(&exe);
        if cfg!(target_os = "macos") {
            cmd.args(["-framework", "CoreFoundation", "-framework", "IOSurface"]);
        } else {
            cmd.args(["-lpthread", "-ldl", "-lm"]);
        }
        let out = cmd.output().expect("the C compiler ran a moment ago");
        assert!(
            out.status.success(),
            "C link against {STATICLIB} FAILED (the toolchain is known good):\n{}",
            String::from_utf8_lossy(&out.stderr)
        );
        run_program(&exe, std::process::Command::new(&exe))
    }

    /// The Windows twin of `cc_build_and_run`: links `edgefirst_codec.lib`
    /// with the tensor DLL's import library, and puts the DLL's directory on
    /// the program's PATH.
    #[cfg(windows)]
    fn cc_build_and_run(src: &str, bin_name: &str) -> Option<std::process::Output> {
        let (lib, tensor_import_lib) = libraries(src, "edgefirst_tensor.dll.lib")?;
        let toolchain = crate::msvc::require(src)?;
        let exe = toolchain.scratch().join(format!("{bin_name}.exe"));
        let includes = [
            concat!(env!("CARGO_MANIFEST_DIR"), "/include"),
            concat!(env!("CARGO_MANIFEST_DIR"), "/../tensor-capi/include"),
        ];
        let libs = [lib.as_os_str(), tensor_import_lib.as_os_str()];
        let out = toolchain.build(&includes, src, &exe, &libs);
        assert!(
            out.status.success(),
            "C link against {STATICLIB} FAILED (the toolchain is known good):\n{}",
            String::from_utf8_lossy(&out.stderr)
        );
        let dll_dir = tensor_import_lib
            .parent()
            .expect("an artifact is in a directory")
            .to_path_buf();
        let inherited = std::env::var_os("PATH").unwrap_or_default();
        let path =
            std::env::join_paths(std::iter::once(dll_dir).chain(std::env::split_paths(&inherited)))
                .expect("PATH entries split on the separator contain none");
        let mut run = std::process::Command::new(&exe);
        run.env("PATH", path);
        run_program(&exe, run)
    }

    /// Parses `src` as C++17 against the shipped headers, or `None` after
    /// reporting that no C++ compiler works here.
    #[cfg(not(windows))]
    fn cxx_syntax_check(src: &str) -> Option<std::process::Output> {
        let cxx = std::env::var("CXX").unwrap_or_else(|_| "c++".to_string());
        let pid = std::process::id();
        let probe = std::env::temp_dir().join(format!("ef_codec_cxx_probe-{pid}.cpp"));
        let works = std::fs::write(&probe, b"int main() { return 0; }\n").is_ok()
            && std::process::Command::new(&cxx)
                .args(["-std=c++17", "-fsyntax-only"])
                .arg(&probe)
                .output()
                .is_ok_and(|o| o.status.success());
        if !works {
            skip(&format!(
                "{src} not compiled: no working C++17 toolchain ({cxx})"
            ));
            return None;
        }
        let out = std::process::Command::new(&cxx)
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
                concat!(env!("CARGO_MANIFEST_DIR"), "/include"),
                "-I",
                concat!(env!("CARGO_MANIFEST_DIR"), "/../tensor-capi/include"),
                src,
            ])
            .output()
            .expect("the C++ compiler ran a moment ago");
        Some(out)
    }

    #[cfg(windows)]
    fn cxx_syntax_check(src: &str) -> Option<std::process::Output> {
        let includes = [
            concat!(env!("CARGO_MANIFEST_DIR"), "/include"),
            concat!(env!("CARGO_MANIFEST_DIR"), "/../tensor-capi/include"),
        ];
        Some(crate::msvc::require(src)?.syntax_check(&includes, &["/TP", "/std:c++17"], src))
    }

    /// C++ callers include the same header: each enum's name must be one
    /// type in C++, where the enum tag is already a type name.
    #[test]
    fn the_header_compiles_as_cpp17() {
        let src = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/c/test_cpp_include.cpp");
        let Some(out) = cxx_syntax_check(src) else {
            return;
        };
        assert!(
            out.status.success(),
            "codec.h failed to compile as C++17:\n{}{}",
            String::from_utf8_lossy(&out.stdout),
            String::from_utf8_lossy(&out.stderr)
        );
    }

    /// The whole C surface from a C program: capabilities, a JPEG encoded
    /// and decoded back, refusals, and an H.264 stream encoded from tensors
    /// and decoded into frames where this machine has hardware for it.
    #[test]
    fn the_codec_api_works_from_c() {
        let src = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/c/test_codec_api.c");
        let Some(out) = cc_build_and_run(src, "ef_test_codec_api") else {
            return;
        };
        artifacts::relay_skips("the_codec_api_works_from_c", &out);
        assert!(
            out.status.success(),
            "the C program failed:\n{}{}",
            String::from_utf8_lossy(&out.stdout),
            String::from_utf8_lossy(&out.stderr)
        );
        print!("{}", String::from_utf8_lossy(&out.stdout));
    }
}
