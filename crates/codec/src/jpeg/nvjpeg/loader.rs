// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! `dlopen` loader for the CUDA nvJPEG library.
//!
//! The library is opened at runtime and every entry point resolved into a
//! function-pointer table cached in a [`OnceLock`]. When the library is
//! absent or incomplete the table is `None` and the codec uses the V4L2/CPU
//! decoders.
//!
//! The explicit CUDA paths are tried first, and `nvjpegCreateEx` must
//! resolve before a library is accepted.

use super::ffi::*;
use libloading::Library;
use std::sync::OnceLock;

/// Environment variable enabling the nvJPEG backend, which is off by
/// default. `1`/`true`/`yes` enable it.
const ENV_ENABLE: &str = "EDGEFIRST_ENABLE_NVJPEG";

/// Candidate library names/paths, in priority order: explicit CUDA install
/// locations, then the bare soname.
const CANDIDATES: &[&str] = &[
    "/usr/local/cuda/targets/aarch64-linux/lib/libnvjpeg.so.12",
    "/usr/local/cuda/targets/aarch64-linux/lib/libnvjpeg.so",
    "/usr/local/cuda/targets/x86_64-linux/lib/libnvjpeg.so.12",
    "/usr/local/cuda/lib64/libnvjpeg.so.12",
    "/usr/local/cuda/lib64/libnvjpeg.so",
    "libnvjpeg.so.12",
];

/// Resolved nvJPEG entry points. The leaked `Library` keeps the symbols valid
/// for the process lifetime.
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
    if !crate::jpeg::env_flag(ENV_ENABLE) {
        log::debug!("nvjpeg off by default; set {ENV_ENABLE}=1 to enable");
        return None;
    }
    for name in CANDIDATES {
        // SAFETY: dlopen of a system library.
        let Ok(lib) = (unsafe { Library::new(*name) }) else {
            continue;
        };
        // A library missing any required symbol is skipped in favour of the
        // next candidate.
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

/// True iff nvJPEG is opted in (`EDGEFIRST_ENABLE_NVJPEG`) and both
/// libnvjpeg and libcudart loaded with all required symbols. Cached.
pub fn is_nvjpeg_available() -> bool {
    lib().is_some() && edgefirst_tensor::is_cuda_available()
}
