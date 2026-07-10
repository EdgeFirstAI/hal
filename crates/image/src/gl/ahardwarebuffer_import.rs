// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! AHardwareBuffer → EGLImage import for the Android GL backend.
//!
//! The Android counterpart of `dma_import.rs` (Linux DMA-BUF) and
//! `iosurface_import.rs` (macOS IOSurface) — and by far the thinnest of
//! the three: an AHardwareBuffer is **self-describing** (it carries its
//! own format, dimensions, and stride from allocation), so the import
//! needs no per-format attribute assembly. The whole path is:
//!
//! ```text
//! AHardwareBuffer* ── eglGetNativeClientBufferANDROID ──▶ EGLClientBuffer
//!                  ── eglCreateImageKHR(EGL_NATIVE_BUFFER_ANDROID,
//!                                       {EGL_IMAGE_PRESERVED=TRUE}) ──▶ EGLImage
//! ```
//!
//! `EGL_IMAGE_PRESERVED = TRUE` is REQUIRED: without it the buffer's
//! existing contents are undefined after import — a silent-garbage
//! correctness bug, not an error.
//!
//! The three entry points are EGL *extension* functions
//! (`EGL_ANDROID_image_native_buffer` + `EGL_KHR_image_base`), not core
//! EGL 1.4, so they are resolved once per process via `eglGetProcAddress`
//! and cached — the same pattern `context.rs` uses for the Linux KHR
//! image functions and `macos.rs` for `eglGetPlatformDisplayEXT`.
//!
//! ## Locking
//!
//! No process-wide lifecycle lock is taken around create/destroy (unlike
//! the Linux funnel's `image_lifecycle_guard`): that guard exists for
//! embedded Linux drivers (Vivante) whose display-level EGL entry points
//! race across threads. Android's system EGL is specification-conformant
//! thread-safe (same reasoning as the ANGLE path, which relies on ANGLE's
//! internal synchronization). If a vendor driver ever proves otherwise
//! on-device, a dedicated guard can be added here — every create/destroy
//! funnels through this module.

#![cfg(target_os = "android")]

use super::Egl;
use crate::{Error, Result};
use edgefirst_egl as egl;
use std::ffi::c_void;
use std::sync::OnceLock;

/// `EGL_NATIVE_BUFFER_ANDROID` — the `eglCreateImageKHR` target for
/// AHardwareBuffer import (from `EGL_ANDROID_image_native_buffer`).
const EGL_NATIVE_BUFFER_ANDROID: u32 = 0x3140;

type FnGetNativeClientBuffer = unsafe extern "C" fn(buffer: *mut c_void) -> egl::EGLClientBuffer;
type FnCreateImageKHR = unsafe extern "C" fn(
    dpy: egl::EGLDisplay,
    ctx: egl::EGLContext,
    target: u32,
    buffer: egl::EGLClientBuffer,
    attribs: *const egl::Attrib,
) -> egl::EGLImage;
type FnDestroyImageKHR = unsafe extern "C" fn(dpy: egl::EGLDisplay, image: egl::EGLImage) -> u32;

/// The resolved extension entry points. Function pointers obtained from
/// `eglGetProcAddress` are process-global on Android (one system EGL), so
/// a single resolution serves every display/context.
struct AhbEglFns {
    get_native_client_buffer: FnGetNativeClientBuffer,
    create_image_khr: FnCreateImageKHR,
    destroy_image_khr: FnDestroyImageKHR,
}

static AHB_EGL_FNS: OnceLock<std::result::Result<AhbEglFns, String>> = OnceLock::new();

/// Resolve (once per process) the AHardwareBuffer-import extension
/// functions, or report which extension is missing.
fn fns(egl: &Egl) -> Result<&'static AhbEglFns> {
    AHB_EGL_FNS
        .get_or_init(|| {
            let get_native_client_buffer = egl
                .get_proc_address("eglGetNativeClientBufferANDROID")
                .ok_or_else(|| {
                    "eglGetNativeClientBufferANDROID not exported \
                     (EGL_ANDROID_image_native_buffer missing)"
                        .to_string()
                })?;
            let create_image_khr = egl.get_proc_address("eglCreateImageKHR").ok_or_else(|| {
                "eglCreateImageKHR not exported (EGL_KHR_image_base missing)".to_string()
            })?;
            let destroy_image_khr = egl.get_proc_address("eglDestroyImageKHR").ok_or_else(|| {
                "eglDestroyImageKHR not exported (EGL_KHR_image_base missing)".to_string()
            })?;
            // SAFETY: the pointers come from EGL's own dispatch table and
            // match the documented C signatures of the extension functions
            // they were queried by name for.
            Ok(unsafe {
                AhbEglFns {
                    get_native_client_buffer: std::mem::transmute::<
                        extern "system" fn(),
                        FnGetNativeClientBuffer,
                    >(get_native_client_buffer),
                    create_image_khr: std::mem::transmute::<extern "system" fn(), FnCreateImageKHR>(
                        create_image_khr,
                    ),
                    destroy_image_khr: std::mem::transmute::<
                        extern "system" fn(),
                        FnDestroyImageKHR,
                    >(destroy_image_khr),
                }
            })
        })
        .as_ref()
        .map_err(|s| Error::NotSupported(s.clone()))
}

/// Create an EGLImage over an AHardwareBuffer.
///
/// Every Android EGLImage creation funnels through here, carrying the
/// same `image.convert.gl.egl_import` tracing span as the Linux funnel
/// (`platform/linux.rs::new_egl_image_owned`) — one span = one actual
/// `eglCreateImageKHR`. Steady-state frame loops must show ZERO of these
/// after warmup; the span count is the observable for cache-behavior
/// equality gates.
///
/// # Safety
///
/// `buffer_ptr` must be a valid live AHardwareBuffer pointer (borrowed
/// from a live tensor — the import cache entry's `guard` ties the
/// EGLImage's lifetime to it).
pub(super) unsafe fn create_ahardwarebuffer_eglimage(
    egl: &Egl,
    display: egl::Display,
    buffer_ptr: *mut c_void,
) -> Result<egl::Image> {
    let _span =
        tracing::trace_span!("image.convert.gl.egl_import", target = "ahardwarebuffer").entered();
    let fns = fns(egl)?;

    let client_buffer_raw = (fns.get_native_client_buffer)(buffer_ptr);
    if client_buffer_raw.is_null() {
        return Err(Error::Io(std::io::Error::other(format!(
            "eglGetNativeClientBufferANDROID returned NULL (EGL error {:?})",
            egl.get_error()
        ))));
    }

    // EGL_IMAGE_PRESERVED = TRUE: the buffer's existing contents MUST
    // survive the import (see module docs — absence is silent garbage).
    let attribs: [egl::Attrib; 3] = [
        egl::IMAGE_PRESERVED as egl::Attrib,
        egl::TRUE as egl::Attrib,
        egl::NONE as egl::Attrib,
    ];
    let image_raw = (fns.create_image_khr)(
        display.as_ptr(),
        egl::NO_CONTEXT,
        EGL_NATIVE_BUFFER_ANDROID,
        client_buffer_raw,
        attribs.as_ptr(),
    );
    if image_raw == egl::NO_IMAGE {
        return Err(Error::Io(std::io::Error::other(format!(
            "eglCreateImageKHR(EGL_NATIVE_BUFFER_ANDROID) returned NO_IMAGE (EGL error {:?})",
            egl.get_error()
        ))));
    }
    // SAFETY: image_raw is a valid, non-NO_IMAGE EGLImage per the check above.
    Ok(egl::Image::from_ptr(image_raw))
}

/// Destroy an EGLImage created by [`create_ahardwarebuffer_eglimage`].
/// Failures are logged, not propagated — this runs on the owned import's
/// Drop path.
pub(super) fn destroy_ahardwarebuffer_eglimage(
    egl: &Egl,
    display: egl::Display,
    image: egl::Image,
) {
    if image.as_ptr() == egl::NO_IMAGE {
        return;
    }
    match fns(egl) {
        // SAFETY: the image was created on this display by this module and
        // is destroyed at most once (the owning wrapper's Drop).
        Ok(f) => {
            let ok = unsafe { (f.destroy_image_khr)(display.as_ptr(), image.as_ptr()) };
            if ok == 0 {
                log::error!(
                    "eglDestroyImageKHR failed (EGL error {:?})",
                    egl.get_error()
                );
            }
        }
        Err(e) => log::error!("destroy_ahardwarebuffer_eglimage: {e:?}"),
    }
}
