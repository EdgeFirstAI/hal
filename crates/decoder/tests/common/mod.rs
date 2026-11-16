//! Shared helpers for decoder integration tests.
#![allow(dead_code)]

/// Emit a skip notice that survives libtest's output capture.
macro_rules! skip_note {
    ($($arg:tt)*) => {{
        use std::io::Write;
        let _ = write!(&mut std::io::stderr(), "WARNING: SKIPPED: ");
        let _ = writeln!(&mut std::io::stderr(), $($arg)*);
    }};
}

pub mod per_scale_fixture;

use std::path::{Path, PathBuf};

/// Root of the `testdata/` tree: `EDGEFIRST_TESTDATA_DIR` when set, else the
/// manifest-relative path baked in at compile time.
pub fn testdata_root() -> PathBuf {
    std::env::var("EDGEFIRST_TESTDATA_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| {
            PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .join("..")
                .join("..")
                .join("testdata")
        })
}

/// Bytes of a fixture, or `None` after a skip notice when it is missing or
/// still a Git LFS pointer.
pub fn read_fixture(path: &Path) -> Option<Vec<u8>> {
    let Ok(bytes) = std::fs::read(path) else {
        skip_note!("fixture {path:?} not present (run `git lfs pull`)");
        return None;
    };
    if bytes.starts_with(b"version https://git-lfs") {
        skip_note!("fixture {path:?} is an LFS pointer (run `git lfs pull`)");
        return None;
    }
    Some(bytes)
}
