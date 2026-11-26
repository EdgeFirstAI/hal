// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! `EGL_ANDROID_native_fence_sync` export.
//!
//! Turns "all GL commands submitted so far on this context" into a kernel
//! sync-fence fd:
//!
//! ```text
//! eglCreateSyncKHR(EGL_SYNC_NATIVE_FENCE_ANDROID)   // insert fence
//!   → glFlush()                                     // submit (spec-required
//!                                                   //  before the dup)
//!   → eglDupNativeFenceFDANDROID(sync)              // own the fd
//!   → eglDestroySyncKHR(sync)                       // fd outlives the sync
//! ```
//!
//! The entry points are EGL extension functions (`EGL_KHR_fence_sync` +
//! `EGL_ANDROID_native_fence_sync`), resolved once per process via
//! `eglGetProcAddress`. Callers gate on the display's extension probe
//! ([`display_supports_native_fence`]) on Linux and Android.

#![cfg(any(target_os = "linux", target_os = "android"))]

use super::Egl;
use crate::{Error, Result};
use edgefirst_egl as egl;
use std::ffi::c_void;
use std::os::fd::{FromRawFd, OwnedFd};
use std::sync::OnceLock;

/// `EGL_SYNC_NATIVE_FENCE_ANDROID` (EGL_ANDROID_native_fence_sync).
const EGL_SYNC_NATIVE_FENCE_ANDROID: u32 = 0x3144;

type EglSyncKhr = *mut c_void;

type FnCreateSyncKhr =
    unsafe extern "C" fn(dpy: egl::EGLDisplay, r#type: u32, attribs: *const egl::Int) -> EglSyncKhr;
type FnDestroySyncKhr = unsafe extern "C" fn(dpy: egl::EGLDisplay, sync: EglSyncKhr) -> u32;
type FnDupNativeFenceFd = unsafe extern "C" fn(dpy: egl::EGLDisplay, sync: EglSyncKhr) -> egl::Int;

struct FenceEglFns {
    create_sync_khr: FnCreateSyncKhr,
    destroy_sync_khr: FnDestroySyncKhr,
    dup_native_fence_fd: FnDupNativeFenceFd,
}

static FENCE_EGL_FNS: OnceLock<std::result::Result<FenceEglFns, String>> = OnceLock::new();

fn fns(egl: &Egl) -> Result<&'static FenceEglFns> {
    FENCE_EGL_FNS
        .get_or_init(|| {
            let create_sync_khr = egl.get_proc_address("eglCreateSyncKHR").ok_or_else(|| {
                "eglCreateSyncKHR not exported (EGL_KHR_fence_sync missing)".to_string()
            })?;
            let destroy_sync_khr = egl.get_proc_address("eglDestroySyncKHR").ok_or_else(|| {
                "eglDestroySyncKHR not exported (EGL_KHR_fence_sync missing)".to_string()
            })?;
            let dup_native_fence_fd = egl
                .get_proc_address("eglDupNativeFenceFDANDROID")
                .ok_or_else(|| {
                    "eglDupNativeFenceFDANDROID not exported \
                     (EGL_ANDROID_native_fence_sync missing)"
                        .to_string()
                })?;
            // SAFETY: pointers come from EGL's own dispatch table and match
            // the documented C signatures they were queried by name for.
            Ok(unsafe {
                FenceEglFns {
                    create_sync_khr: std::mem::transmute::<extern "system" fn(), FnCreateSyncKhr>(
                        create_sync_khr,
                    ),
                    destroy_sync_khr: std::mem::transmute::<extern "system" fn(), FnDestroySyncKhr>(
                        destroy_sync_khr,
                    ),
                    dup_native_fence_fd: std::mem::transmute::<
                        extern "system" fn(),
                        FnDupNativeFenceFd,
                    >(dup_native_fence_fd),
                }
            })
        })
        .as_ref()
        .map_err(|s| Error::NotSupported(s.clone()))
}

/// Whether an EGL display extension string names both
/// `EGL_ANDROID_native_fence_sync` and `EGL_KHR_fence_sync`.
pub(super) fn display_supports_native_fence(extensions: &str) -> bool {
    let has = |name: &str| extensions.split_ascii_whitespace().any(|e| e == name);
    has("EGL_ANDROID_native_fence_sync") && has("EGL_KHR_fence_sync")
}

/// Export a native sync-fence fd guarding every GL command submitted so
/// far on the current context. Must be called on the GL worker thread
/// with the convert's context current. The returned fd is owned by the
/// caller; the EGL sync object is destroyed before returning.
pub(super) fn export_native_fence_fd(egl: &Egl, display: egl::Display) -> Result<OwnedFd> {
    let fns = fns(egl)?;

    // Insert the fence with the default (pending) native fd attribute.
    let attribs: [egl::Int; 1] = [egl::NONE];
    // SAFETY: display is a live initialized EGLDisplay; attribs is
    // NONE-terminated per the KHR signature.
    let sync = unsafe {
        (fns.create_sync_khr)(
            display.as_ptr(),
            EGL_SYNC_NATIVE_FENCE_ANDROID,
            attribs.as_ptr(),
        )
    };
    if sync.is_null() {
        return Err(Error::OpenGl(format!(
            "eglCreateSyncKHR(EGL_SYNC_NATIVE_FENCE_ANDROID) failed (EGL error {:?})",
            egl.get_error()
        )));
    }

    // Flush between create and dup, as the extension requires.
    unsafe { edgefirst_gl::gl::Flush() };

    // SAFETY: sync is the live object created above on this display.
    let fd = unsafe { (fns.dup_native_fence_fd)(display.as_ptr(), sync) };
    let dup_error = (fd < 0).then(|| egl.get_error());
    let destroy_ok = unsafe { (fns.destroy_sync_khr)(display.as_ptr(), sync) } != 0;
    if !destroy_ok {
        log::warn!("eglDestroySyncKHR failed after fence dup (leaking the sync object)");
    }
    if let Some(error) = dup_error {
        return Err(Error::OpenGl(format!(
            "eglDupNativeFenceFDANDROID returned {fd} (EGL error {error:?})"
        )));
    }
    // SAFETY: fd is a freshly dup'd, caller-owned kernel sync-fence fd.
    Ok(unsafe { OwnedFd::from_raw_fd(fd) })
}

#[cfg(test)]
mod tests {
    use super::display_supports_native_fence;

    /// Both extensions are required, matched as whole tokens.
    #[test]
    fn a_native_fence_needs_both_extensions_as_whole_tokens() {
        assert!(display_supports_native_fence(
            "EGL_KHR_image EGL_KHR_fence_sync EGL_ANDROID_native_fence_sync"
        ));
        assert!(!display_supports_native_fence(
            "EGL_ANDROID_native_fence_sync"
        ));
        assert!(!display_supports_native_fence("EGL_KHR_fence_sync"));
        assert!(!display_supports_native_fence(
            "EGL_KHR_fence_sync EGL_ANDROID_native_fence_sync_extra"
        ));
        assert!(!display_supports_native_fence(""));
    }
}
