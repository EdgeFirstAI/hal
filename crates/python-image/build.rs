// SPDX-FileCopyrightText: Copyright 2025 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

use std::env;
use std::path::{Path, PathBuf};

fn main() {
    pyo3_build_config::use_pyo3_cfgs();

    // Emit the Python extension-module link arguments.
    edgefirst_build_support::python_extension();

    // `_image` calls into libedgefirst_tensor, which the edgefirst-tensor wheel ships in the sibling
    // `edgefirst/tensor/` directory (on Windows, `__init__.py` registers it with os.add_dll_directory).
    edgefirst_build_support::link_python_extension_to_tensor(
        edgefirst_build_support::Rpath::Sibling("tensor"),
    );

    if env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("windows") {
        bundle_angle(&PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap()));
    }
}

/// Windows: bundle ANGLE (GLES over Direct3D 11) into the wheel.
///
/// Copies ANGLE's two DLLs from `EDGEFIRST_ANGLE_PATH` into
/// `python/edgefirst/image/`, which maturin's `python-source` packages
/// verbatim. When the variable is unset or the directory lacks the DLLs, a
/// bundled copy from an earlier build is removed.
fn bundle_angle(crate_dir: &Path) {
    println!("cargo:rerun-if-env-changed=EDGEFIRST_ANGLE_PATH");
    let dest_dir = crate_dir.join("python/edgefirst/image");
    const DLLS: [&str; 2] = ["libEGL.dll", "libGLESv2.dll"];
    let src_dir = env::var_os("EDGEFIRST_ANGLE_PATH")
        .filter(|p| !p.is_empty())
        .map(PathBuf::from);
    match src_dir {
        Some(src_dir) if DLLS.iter().all(|d| src_dir.join(d).is_file()) => {
            for dll in DLLS {
                let src = src_dir.join(dll);
                println!("cargo:rerun-if-changed={}", src.display());
                let dest = dest_dir.join(dll);
                std::fs::copy(&src, &dest).unwrap_or_else(|e| {
                    panic!(
                        "failed to copy {} to {}: {e}",
                        src.display(),
                        dest.display()
                    )
                });
            }
            println!(
                "cargo:warning=bundling ANGLE (libEGL.dll, libGLESv2.dll) from {} into the edgefirst-image wheel",
                src_dir.display()
            );
        }
        Some(src_dir) => {
            println!(
                "cargo:warning=EDGEFIRST_ANGLE_PATH={} does not contain libEGL.dll + libGLESv2.dll; the wheel ships without ANGLE",
                src_dir.display()
            );
            for dll in DLLS {
                let _ = std::fs::remove_file(dest_dir.join(dll));
            }
        }
        None => {
            for dll in DLLS {
                let _ = std::fs::remove_file(dest_dir.join(dll));
            }
        }
    }
}
