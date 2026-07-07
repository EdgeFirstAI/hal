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

/// Force the HAL closure into the staticlib's symbol table. The body touches
/// enough of the public API (tensor construction, image/codec/decoder type
/// surface) that the linker cannot dead-strip the transitive dependency
/// graph — including the `#[link(name = "IOSurface"/"CoreFoundation",
/// kind = "framework")]` references in the tensor and image crates.
///
/// This function is never called at runtime by the validation; its mere
/// presence in the archive is what matters. It is `#[no_mangle]` + `pub`
/// so the staticlib exports it (exported symbols survive archive dead-
/// stripping), and a companion `#[used]` static holds its address so the
/// LTO pass inside this crate cannot drop it either.
#[no_mangle]
pub extern "C" fn edgefirst_ios_validation_force_closure() {
    // Reference the facade modules so the whole closure is linked.
    let _ = std::any::TypeId::of::<hal::tensor::Tensor<u8>>();
    let _ = std::any::TypeId::of::<hal::image::ImageProcessor>();
    let _ = std::any::TypeId::of::<hal::codec::ImageDecoder>();
    let _ = std::any::TypeId::of::<hal::decoder::BoundingBox>();
}

/// Keep `edgefirst_ios_validation_force_closure` rooted through LTO. The
/// `#[used]` attribute forces the compiler to emit this static into the
/// object file, and its initializer takes the function's address, so the
/// linker sees the reference and retains the whole transitive closure.
#[used]
#[no_mangle]
static EDGEFIRST_IOS_VALIDATION_ROOT: extern "C" fn() = edgefirst_ios_validation_force_closure;
