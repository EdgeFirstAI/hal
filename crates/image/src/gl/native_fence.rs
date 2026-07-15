// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! `EGL_ANDROID_native_fence_sync` export — the GL→NPU handoff fence.
//!
//! A convert whose destination is a zero-copy AHardwareBuffer does not
//! need the CPU to wait for the GPU at all: the NPU runtime can wait on
//! a kernel sync-fence fd instead
//! (`ANeuralNetworksExecution_startComputeWithDependencies`). This module
//! turns "all GL commands submitted so far on this context" into such an
//! fd:
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
//! `eglGetProcAddress` — the same pattern as `ahardwarebuffer_import.rs`.
//! Callers gate on the display's extension probe
//! (`SharedAndroidDisplay::supports_native_fence`) rather than on
//! resolution: `eglGetProcAddress` may return a non-NULL stub for
//! unsupported extensions, so the extension string is the authority.

#![cfg(target_os = "android")]

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

/// Export a native sync-fence fd guarding every GL command submitted so
/// far on the CURRENT context. Must be called on the GL worker thread
/// with the convert's context current. The returned fd is owned by the
/// caller; the EGL sync object itself is destroyed before returning (the
/// dup'd fd is independent per the extension spec).
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

    // The native fence object is not created until the command stream is
    // flushed — the spec REQUIRES a flush between create and dup, or dup
    // returns EGL_NO_NATIVE_FENCE_FD_ANDROID.
    unsafe { edgefirst_gl::gl::Flush() };

    // SAFETY: sync is the live object created above on this display.
    let fd = unsafe { (fns.dup_native_fence_fd)(display.as_ptr(), sync) };
    let destroy_ok = unsafe { (fns.destroy_sync_khr)(display.as_ptr(), sync) } != 0;
    if !destroy_ok {
        log::warn!("eglDestroySyncKHR failed after fence dup (leaking the sync object)");
    }
    if fd < 0 {
        return Err(Error::OpenGl(format!(
            "eglDupNativeFenceFDANDROID returned {fd} (EGL error {:?})",
            egl.get_error()
        )));
    }
    // SAFETY: fd is a freshly dup'd, caller-owned kernel sync-fence fd.
    Ok(unsafe { OwnedFd::from_raw_fd(fd) })
}
