// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! `dlopen` loader for the CUDA nvJPEG library.
//!
//! Mirrors the discipline of `edgefirst_tensor`'s `cuda.rs`: the library is
//! opened at runtime and every entry point resolved into a function-pointer
//! table cached in a [`OnceLock`]. Absent or incomplete ⇒ the whole table is
//! `None` and the codec degrades to the V4L2/CPU decoders.
//!
//! Two target-specific facts shape the candidate list:
//! - The real `libnvjpeg.so.12` is NOT on the default loader path on JetPack 6;
//!   it lives under `/usr/local/cuda/targets/<arch>/lib/`.
//! - The `libnvjpeg.so` on the default path (`/usr/lib/.../nvidia/`) is a
//!   libjpeg-turbo *decoy* that exports `jpeg_*`, not `nvjpeg*`.
//!
//! So the explicit CUDA paths are tried first, and `nvjpegCreateEx` is required
//! to resolve before a library is accepted — the decoy is rejected because it
//! lacks that symbol.

use super::ffi::*;
use libloading::Library;
use std::sync::OnceLock;

/// Environment variable forcing the V4L2/CPU decoders (skip nvJPEG probing).
const ENV_DISABLE: &str = "EDGEFIRST_DISABLE_NVJPEG";

/// Candidate library names/paths, in priority order. Explicit CUDA install
/// locations come first (the soname is not on the default loader path); the
/// bare soname is last so a correctly-set `LD_LIBRARY_PATH` still works.
const CANDIDATES: &[&str] = &[
    "/usr/local/cuda/targets/aarch64-linux/lib/libnvjpeg.so.12",
    "/usr/local/cuda/targets/aarch64-linux/lib/libnvjpeg.so",
    "/usr/local/cuda/targets/x86_64-linux/lib/libnvjpeg.so.12",
    "/usr/local/cuda/lib64/libnvjpeg.so.12",
    "/usr/local/cuda/lib64/libnvjpeg.so",
    "libnvjpeg.so.12",
];

/// Resolved nvJPEG entry points. The leaked `Library` keeps the symbols valid
/// for the process lifetime (matching `cuda.rs`).
pub(crate) struct NvjpegLib {
    _lib: &'static Library,
    pub create_ex: FnCreateEx,
    pub jpeg_state_create: FnJpegStateCreate,
    pub get_image_info: FnGetImageInfo,
    pub decode: FnDecode,
    pub destroy: FnDestroy,
    pub jpeg_state_destroy: FnJpegStateDestroy,
}

static LIB: OnceLock<Option<NvjpegLib>> = OnceLock::new();

fn load() -> Option<NvjpegLib> {
    if std::env::var_os(ENV_DISABLE).is_some() {
        log::debug!("nvjpeg disabled via {ENV_DISABLE}");
        return None;
    }
    for name in CANDIDATES {
        // SAFETY: dlopen of a system library; failures are expected and skipped.
        let Ok(lib) = (unsafe { Library::new(name) }) else {
            continue;
        };
        // Require the real nvjpeg symbol set; this rejects the libjpeg-turbo
        // decoy (which has no `nvjpegCreateEx`). A library that opens but is
        // missing any symbol is skipped in favour of the next candidate.
        match resolve(&lib) {
            Some((
                create_ex,
                jpeg_state_create,
                get_image_info,
                decode,
                destroy,
                jpeg_state_destroy,
            )) => {
                let lib: &'static Library = Box::leak(Box::new(lib));
                log::info!("nvjpeg loaded from {name}");
                return Some(NvjpegLib {
                    _lib: lib,
                    create_ex,
                    jpeg_state_create,
                    get_image_info,
                    decode,
                    destroy,
                    jpeg_state_destroy,
                });
            }
            None => continue,
        }
    }
    log::debug!("nvjpeg unavailable (no libnvjpeg.so.12 with nvjpeg* symbols found)");
    None
}

/// Resolve all required symbols from `lib`, returning `None` if any is missing.
#[allow(clippy::type_complexity)]
fn resolve(
    lib: &Library,
) -> Option<(
    FnCreateEx,
    FnJpegStateCreate,
    FnGetImageInfo,
    FnDecode,
    FnDestroy,
    FnJpegStateDestroy,
)> {
    macro_rules! sym {
        ($n:literal) => {
            // SAFETY: the symbol type matches the declared `nvjpeg.h` ABI.
            *unsafe { lib.get(concat!($n, "\0").as_bytes()) }.ok()?
        };
    }
    Some((
        sym!("nvjpegCreateEx"),
        sym!("nvjpegJpegStateCreate"),
        sym!("nvjpegGetImageInfo"),
        sym!("nvjpegDecode"),
        sym!("nvjpegDestroy"),
        sym!("nvjpegJpegStateDestroy"),
    ))
}

/// The resolved nvJPEG table, or `None` when the library is unavailable. Cached.
pub(crate) fn lib() -> Option<&'static NvjpegLib> {
    LIB.get_or_init(load).as_ref()
}

/// True iff both libnvjpeg and libcudart loaded with all required symbols.
/// Cheap (cached) — intended as a fast pre-check for benches/consumers.
pub fn is_nvjpeg_available() -> bool {
    lib().is_some() && edgefirst_tensor::is_cuda_available()
}
