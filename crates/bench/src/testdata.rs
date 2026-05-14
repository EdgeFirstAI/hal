// SPDX-FileCopyrightText: Copyright 2025-2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! Testdata path resolution for tests, benchmarks, and doctests.
//!
//! Replaces compile-time [`include_bytes!`] / [`include_str!`] calls with
//! runtime filesystem reads. Keeps test and bench binaries small by avoiding
//! testdata embedding — critical for cross-compiled binaries deployed to
//! target hardware, where embedded testdata bloats artifact downloads.
//!
//! # Resolution order
//!
//! 1. `EDGEFIRST_TESTDATA_DIR` environment variable. Set this in CI or
//!    when running tests from an unusual location. The value is used
//!    verbatim with no further validation.
//! 2. `<workspace>/testdata`, derived from this crate's compile-time
//!    `CARGO_MANIFEST_DIR` (always `<workspace>/crates/bench`).
//!
//! If neither resolution yields a real directory, [`root`] panics with a
//! diagnostic message — testdata absence is a developer / CI setup error,
//! not a runtime condition that callers should handle.
//!
//! # Example
//!
//! ```no_run
//! use edgefirst_bench::testdata;
//!
//! // Read a JSON schema fixture
//! let schema = testdata::read_to_string("per_scale/synthetic_yolov8n_schema.json");
//!
//! // Read a binary tensor dump
//! let bytes = testdata::read("yolov8s_80_classes.bin");
//!
//! // Get just the path (useful when handing to another reader)
//! let path = testdata::path("decoder/yolo11n-seg.safetensors");
//! ```

use std::path::{Path, PathBuf};

/// Environment variable name that overrides the default testdata location.
pub const ENV_VAR: &str = "EDGEFIRST_TESTDATA_DIR";

/// Returns the root testdata directory.
///
/// See the [module-level documentation](self) for the resolution order.
///
/// # Panics
///
/// Panics if neither `EDGEFIRST_TESTDATA_DIR` is set nor the
/// workspace-relative `testdata/` directory exists.
pub fn root() -> PathBuf {
    if let Ok(dir) = std::env::var(ENV_VAR) {
        return PathBuf::from(dir);
    }
    // CARGO_MANIFEST_DIR for this crate is <workspace>/crates/bench at
    // compile time, so `testdata/` is two levels up.
    let candidate = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .join("testdata");
    if candidate.is_dir() {
        return candidate;
    }
    panic!(
        "testdata directory not found: ${ENV_VAR} is unset and {} does not exist. \
         Set ${ENV_VAR} to the testdata directory or run from a workspace checkout.",
        candidate.display(),
    );
}

/// Returns the absolute path to a file or directory under `testdata/`.
///
/// The `rel` argument is joined onto [`root`] verbatim and is not
/// validated to exist — callers will get a clear filesystem error when
/// they attempt to open the result. Use [`read`] or [`read_to_string`]
/// for one-shot loads that should panic on missing files.
pub fn path(rel: impl AsRef<Path>) -> PathBuf {
    root().join(rel)
}

/// Reads the testdata file at `rel` as raw bytes.
///
/// # Panics
///
/// Panics if the file cannot be read. The panic message includes the
/// resolved absolute path to aid debugging.
pub fn read(rel: impl AsRef<Path>) -> Vec<u8> {
    let p = path(rel.as_ref());
    std::fs::read(&p)
        .unwrap_or_else(|e| panic!("read testdata {}: {}", p.display(), e))
}

/// Reads the testdata file at `rel` as a UTF-8 string.
///
/// # Panics
///
/// Panics if the file cannot be read or is not valid UTF-8.
pub fn read_to_string(rel: impl AsRef<Path>) -> String {
    let p = path(rel.as_ref());
    std::fs::read_to_string(&p)
        .unwrap_or_else(|e| panic!("read testdata {}: {}", p.display(), e))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `root()` resolves to a real directory under default workspace layout.
    #[test]
    fn root_resolves_to_workspace_testdata() {
        // SAFETY: tests run with the workspace's testdata/ in place.
        // Unset any user-side override to test the workspace-fallback path.
        let saved = std::env::var(ENV_VAR).ok();
        // SAFETY: tests in this module are run single-threaded via -j 1.
        unsafe {
            std::env::remove_var(ENV_VAR);
        }

        let r = root();
        assert!(r.is_dir(), "expected testdata dir at {}", r.display());

        // Restore prior value if any.
        if let Some(v) = saved {
            // SAFETY: see above.
            unsafe {
                std::env::set_var(ENV_VAR, v);
            }
        }
    }

    /// `EDGEFIRST_TESTDATA_DIR` override is honored.
    #[test]
    fn env_var_override_is_respected() {
        let tmp = std::env::temp_dir();
        let saved = std::env::var(ENV_VAR).ok();
        // SAFETY: tests in this module are run single-threaded via -j 1.
        unsafe {
            std::env::set_var(ENV_VAR, &tmp);
        }

        let r = root();
        assert_eq!(r, tmp);

        // Restore.
        unsafe {
            match saved {
                Some(v) => std::env::set_var(ENV_VAR, v),
                None => std::env::remove_var(ENV_VAR),
            }
        }
    }

    /// `path()` joins relative segments onto the root.
    #[test]
    fn path_joins_relative_segments() {
        let saved = std::env::var(ENV_VAR).ok();
        // SAFETY: see above.
        unsafe {
            std::env::set_var(ENV_VAR, "/fake/root");
        }

        let p = path("per_scale/foo.json");
        assert_eq!(p, PathBuf::from("/fake/root/per_scale/foo.json"));

        unsafe {
            match saved {
                Some(v) => std::env::set_var(ENV_VAR, v),
                None => std::env::remove_var(ENV_VAR),
            }
        }
    }
}
