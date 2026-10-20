//! Shared helpers for decoder integration tests.
#![allow(dead_code)] // Helpers may not be used by every test binary.

/// Emit a skip notice that survives libtest's output capture.
///
/// libtest captures `println!`/`eprintln!` and replays it only for **failing**
/// tests, so a test that skips and returns prints `... ok` with its reason
/// discarded — indistinguishable from one that did the work. Writing to
/// `std::io::stderr()` directly bypasses that. See TESTING.md.
macro_rules! skip_note {
    ($($arg:tt)*) => {{
        use std::io::Write;
        let _ = write!(&mut std::io::stderr(), "WARNING: SKIPPED: ");
        let _ = writeln!(&mut std::io::stderr(), $($arg)*);
    }};
}

pub mod per_scale_fixture;

use std::path::{Path, PathBuf};

/// Root of the `testdata/` tree. `EDGEFIRST_TESTDATA_DIR` wins when set; the
/// manifest-relative fallback is baked in at compile time and so is useless
/// for a cross-compiled on-target run.
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
