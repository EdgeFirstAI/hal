// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! The C API for `edgefirst-tensor`.
//!
//! # Surfaces
//!
//! * **Opaque handle + exported accessors** — in-process. The handle is
//!   opaque; every accessor is an exported function of this library. A tensor
//!   minted by a sibling library (e.g. `ef_image_processor_create_image()`)
//!   is the same kind of object as one from `ef_tensor_new()`.
//! * **Builder** — construction, with sticky errors.
//! * **`(blob, fds)`** — IPC and serialization, which defines the wire
//!   layout.

#[cfg(edgefirst_tensor_dynamic)]
compile_error!(
    "edgefirst-tensor-capi is the tensor implementation behind libedgefirst_tensor and \
     cannot be built with `--cfg edgefirst_tensor_dynamic`; build it without .cargo/tensor-dynamic.toml."
);

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
pub mod pbo;
pub mod probe;
pub mod quant;
pub mod serialize;
pub mod trace;

/// ABI version of this library's C surface.
///
/// Incremented when the C ABI breaks; adding an exported function does not
/// change it.
#[no_mangle]
pub extern "C" fn ef_tensor_abi_version() -> u32 {
    2
}

// Test-support modules shared with the other modular C-API leaves.
#[cfg(test)]
#[path = "../tests/support/artifacts.rs"]
mod artifacts;
#[cfg(test)]
#[path = "../../tensor/tests/support/dma_require.rs"]
mod dma_require;
#[cfg(all(test, windows))]
#[path = "../tests/support/msvc.rs"]
mod msvc;

#[cfg(test)]
mod tests {
    /// The shipped header text.
    fn header_text() -> String {
        std::fs::read_to_string(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/include/edgefirst/tensor.h"
        ))
        .expect("cbindgen wrote include/edgefirst/tensor.h")
    }

    /// Compile a C translation unit against the shipped header.
    ///
    /// Returns `None`, with a printed reason, when no C compiler is
    /// available.
    #[cfg(not(windows))]
    fn cc_syntax_check(src: &str) -> Option<std::process::Output> {
        let include = concat!(env!("CARGO_MANIFEST_DIR"), "/include");
        let cc = std::env::var("CC").unwrap_or_else(|_| "cc".to_string());
        match std::process::Command::new(&cc)
            .args(["-fsyntax-only", "-Werror", "-Wall", "-I", include, src])
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

    /// Like [`cc_syntax_check`], with `-std=c11`.
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
                skip(&format!(
                    "no C compiler ({cc}: {e}); header not syntax-checked"
                ));
                None
            }
        }
    }

    // Artifact discovery, freshness and symbol reading, shared with the
    // other modular C-API leaves.
    use crate::artifacts;
    use artifacts::skip;

    /// This leaf's sources, which every artifact read here is checked
    /// against for freshness.
    const SRC_DIR: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/src");

    /// The prefix every artifact of this library shares, and the staticlib's
    /// name, in this platform's spelling.
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

    /// Symbols the built cdylib exports.
    ///
    /// Returns `None`, with a printed reason, when no fresh cdylib is found
    /// or the platform's symbol tool is unavailable.
    fn exported_symbols() -> Option<Vec<String>> {
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
            assert_eq!(
                text.matches(&format!("{name}(")).count(),
                1,
                "{name} declared exactly once"
            );
        }
    }

    #[test]
    fn every_accessor_is_an_exported_symbol() {
        let Some(symbols) = exported_symbols() else {
            return;
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
        // Link the platform runtime libraries the staticlib needs.
        if cfg!(target_os = "macos") {
            cmd.args(["-framework", "CoreFoundation", "-framework", "IOSurface"]);
        } else {
            cmd.args(["-lpthread", "-ldl", "-lm"]);
        }
        // Probe the toolchain with a trivial translation unit that does not
        // touch this library.
        if !toolchain_works(&cc) {
            skip(&format!("{src} not linked: no working C toolchain ({cc})"));
            return None;
        }
        match cmd.output() {
            // A link failure with a working toolchain panics.
            Ok(o) if !o.status.success() => panic!(
                "C link against libedgefirst_tensor FAILED (the toolchain is \
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
        // Per-process file names.
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

    // MSVC discovery shared with the other modular C-API leaves.
    #[cfg(windows)]
    use crate::msvc;

    /// The Windows twin of `cc_syntax_check`: `cl.exe /Zs` parses without
    /// generating code, and `/W4 /WX` stands in for `-Wall -Werror`.
    #[cfg(windows)]
    fn cc_syntax_check(src: &str) -> Option<std::process::Output> {
        let include = concat!(env!("CARGO_MANIFEST_DIR"), "/include");
        Some(msvc::require(src)?.syntax_check(&[include], &[], src))
    }

    /// The Windows twin of `cc_syntax_check_c11`, with `/std:c11`.
    #[cfg(windows)]
    fn cc_syntax_check_c11(src: &str) -> Option<std::process::Output> {
        let include = concat!(env!("CARGO_MANIFEST_DIR"), "/include");
        Some(msvc::require(src)?.syntax_check(&[include], &["/std:c11"], src))
    }

    /// The Windows twin of `cc_build_and_run`: `cl.exe` against
    /// `edgefirst_tensor.lib`.
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
        // Skips when the toolchain cannot build a trivial program.
        let toolchain = msvc::require(src)?;
        // The executable goes under the toolchain's per-process scratch
        // directory.
        let exe = toolchain.scratch().join(format!("{bin_name}.exe"));
        let libs = [lib.as_os_str()];
        let out = toolchain.build(&[include], src, &exe, &libs);
        // The harness folds the command line and cl's stdout into stderr.
        assert!(
            out.status.success(),
            "C link against edgefirst_tensor.lib FAILED (the toolchain is \
             known good, so this is a real defect):\n{}",
            String::from_utf8_lossy(&out.stderr)
        );
        run_program(&exe, std::process::Command::new(&exe))
    }

    /// Runs a built C program, returning `None` with a printed reason when
    /// it cannot be spawned.
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

    /// Whether this host can compile and link a trivial C++17 program with
    /// `cxx`.
    fn cpp17_toolchain_works(cxx: &str) -> bool {
        // Per-process file names.
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
    /// `-fsyntax-only` parses without linking, and `-x c++` forces C++
    /// regardless of the file's extension. Panics when `cxx` cannot be
    /// spawned.
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
        let Some(cxx) = find_cpp17_compiler() else {
            skip("no working C++17 toolchain (c++/clang++); header not C++17-checked");
            return;
        };

        // A scratch header declaring `int class = 1;` must be rejected under
        // `-std=c++17`.
        let scratch =
            std::env::temp_dir().join(format!("ef_cpp17_hostile_scratch-{}.h", std::process::id()));
        std::fs::write(&scratch, b"int class = 1;\n").expect("write scratch header");
        let bad = cpp17_syntax_check(cxx, scratch.to_str().expect("temp path is UTF-8"));
        assert!(
            !bad.status.success(),
            "`int class = 1;` compiled under \
             {cxx} -std=c++17, so the C++ check cannot fail"
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
        // On Windows the C program allocates a texture tensor and drives the
        // layout, share, completion and re-wrap path; elsewhere it asserts
        // each export refuses at run time.
        let src = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/c/test_d3d11.c");
        let Some(out) = cc_build_and_run(src, "ef_test_d3d11") else {
            return;
        };
        artifacts::relay_skips("d3d11_exports_are_usable_or_refuse_off_windows", &out);
        assert!(
            out.status.success(),
            "C D3D11 contract failed:\n{}",
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
            "header failed to compile when included twice:\n{}",
            String::from_utf8_lossy(&out.stderr)
        );
    }

    #[test]
    fn the_shipped_artifact_is_libedgefirst_tensor_not_capi() {
        // Skip when the staticlib has not been built.
        if let Err(reason) = find_artifact(STATICLIB) {
            skip(&format!("artifact name not checked: {reason}"));
            return;
        }
        // Search every candidate directory.
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
        // Every `EF_` token in the README is declared in the header.
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
    fn the_abi_probe_reports_the_current_version() {
        assert_eq!(super::ef_tensor_abi_version(), 2);
    }

    #[test]
    fn the_header_carries_no_hal_symbols() {
        let h = header_text();
        assert!(
            !h.contains("hal_"),
            "tensor.h must not carry hal_* symbols"
        );
    }

    /// Parse cbindgen's `EF_NAME = VALUE,`-style enumerator lines out of the
    /// shipped header.
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
        use edgefirst_tensor::{CpuAccess, DType, TensorMemory};
        use edgefirst_tensor_abi::EfCompressionRequest;

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
        expect(
            "EF_STORAGE_KIND_D3D11_TEXTURE",
            TensorMemory::D3d11Texture.code(),
        );

        expect("EF_CPU_ACCESS_NONE", CpuAccess::None.code());
        expect("EF_CPU_ACCESS_READ", CpuAccess::Read.code());
        expect("EF_CPU_ACCESS_WRITE", CpuAccess::Write.code());
        expect("EF_CPU_ACCESS_READ_WRITE", CpuAccess::ReadWrite.code());

        expect(
            "EF_COMPRESSION_REQUEST_NONE",
            EfCompressionRequest::None as u32,
        );
        expect(
            "EF_COMPRESSION_REQUEST_ANY",
            EfCompressionRequest::Any as u32,
        );
        expect(
            "EF_COMPRESSION_REQUEST_SCHEME",
            EfCompressionRequest::Scheme as u32,
        );
    }
}
