// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! iOS link-validation shim.
//!
//! This crate exists ONLY to drive `scripts/validate-ios-link.sh`: it builds
//! the full HAL dependency closure (`edgefirst-hal` + `tensor` + `codec` +
//! `image` + `decoder` + `tracker` + `egl` + `gl`) into a single `staticlib`
//! (`.a`) and forces the GL/IOSurface symbol references into the archive so
//! the link step cannot eliminate them. The resulting `.a` is then linked
//! against the ANGLE xcframeworks + Apple system frameworks to prove the
//! native symbol closure is complete (see README.md § iOS).
//!
//! This crate is `publish = false` and intentionally NOT in the workspace's
//! `default-members` — it is built only on demand by the validation script.

#![allow(dead_code)]

use edgefirst_hal as hal;

/// Force the Apple framework closure into the staticlib so the link step in
/// `scripts/validate-ios-link.sh` genuinely validates it.
///
/// The body *calls* real constructors (not `TypeId::of`, which is a
/// compile-time constant that emits no code and references nothing): the
/// concrete monomorphizations are codegen'd into this crate's object file,
/// so the archive carries actual undefined references to `IOSurfaceCreate`,
/// `CFDictionaryCreateMutable`, etc. That is what makes the `-framework
/// IOSurface` / `-framework CoreFoundation` flags at the link step
/// load-bearing — drop either and the link fails with undefined symbols.
///
/// This function is never called at runtime by the validation; `main.c` in
/// the link script calls it only so ld pulls this archive member in (an
/// archive member is linked only to resolve an undefined symbol — without
/// this call the whole `.a` is dropped and the check is a false positive).
/// It is `#[no_mangle]` + `pub` so the staticlib exports it, and a companion
/// `#[used]` static roots it through this crate's LTO pass.
#[no_mangle]
pub extern "C" fn edgefirst_ios_validation_force_closure() {
    // (1) tensor IOSurface path — `TensorMemory::Dma` routes to
    // `IoSurfaceTensor::<u8>::new` → `IOSurfaceCreate` +
    // `CFDictionaryCreateMutable`/`CFNumberCreate` (see
    // crates/tensor/src/iosurface.rs), pulling that crate's
    // `#[link(name = "IOSurface"/"CoreFoundation", kind = "framework")]`
    // extern block into the archive.
    if let Ok(t) = hal::tensor::Tensor::<u8>::new(&[64], Some(hal::tensor::TensorMemory::Dma), None)
    {
        std::hint::black_box(&t);
    }

    // (2) image GL closure — constructing the ANGLE-backed `ImageProcessor`
    // (default features include `opengl`) pulls `GLProcessorThreaded` and the
    // whole `gl/` module — including the IOSurface GL-import path that carries
    // its own `#[link(kind = "framework")]` refs — into the link, proving the
    // full iOS GL Rust closure is self-consistent and links. The EGL/GLES
    // entry points themselves are resolved at runtime via `libloading`
    // (`Library::this()`), so they carry no link-time reference here and are
    // validated separately by the `nm` export check in the link script.
    if let Ok(p) = hal::image::ImageProcessor::new() {
        std::hint::black_box(&p);
    }
}

/// Keep `edgefirst_ios_validation_force_closure` rooted through LTO. The
/// `#[used]` attribute forces the compiler to emit this static into the
/// object file, and its initializer takes the function's address, so the
/// linker sees the reference and retains the whole transitive closure.
#[used]
#[no_mangle]
static EDGEFIRST_IOS_VALIDATION_ROOT: extern "C" fn() = edgefirst_ios_validation_force_closure;
