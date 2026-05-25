// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! IOSurface allocation and EGL pbuffer import for the macOS GL backend.
//!
//! Mirrors the role of `dma_import.rs` on Linux: takes a tensor that
//! carries platform-native zero-copy GPU buffer (IOSurface on macOS,
//! DMA-BUF on Linux) and produces the EGL handle the GL backend can
//! sample from or render to.
//!
//! The EGL flow is structurally different from Linux. Linux uses
//! `eglCreateImageKHR` with `EGL_LINUX_DMA_BUF_EXT` to produce an
//! `EGLImage` that is bound to a texture via
//! `glEGLImageTargetTexture2DOES`. macOS uses
//! `eglCreatePbufferFromClientBuffer` with `EGL_IOSURFACE_ANGLE` to
//! produce an `EGLSurface` (pbuffer) that is bound to a texture via
//! `eglBindTexImage`. The macOS path is invoked from
//! [`super::macos_processor::MacosGlProcessor::convert_yuyv_to_rgba`];
//! Linux callers do not go through this module.
//!
//! See `spikes/angle_iosurface/` (local, gitignored) for the proof-of-
//! concept that validates each constant + attribute combination.

#![cfg(target_os = "macos")]

use crate::Error;
use edgefirst_tensor::{PixelFormat, Tensor, TensorTrait};
use khronos_egl as egl;

// ---------------------------------------------------------------------------
// ANGLE EGL constants for the IOSurface client-buffer path.
//
// These are from `include/EGL/eglext_angle.h` in the ANGLE source tree.
// IMPORTANT: `EGL_TEXTURE_TYPE_ANGLE` is `0x345C`, not `0x345B`. The
// `0x345B` slot is `EGL_TEXTURE_RECTANGLE_ANGLE` (a target value, not
// an attribute key). Sending `0x345B` as the type-attribute key causes
// `eglCreatePbufferFromClientBuffer` to reject the call with
// `EGL_BAD_ATTRIBUTE`. The spike at `spikes/angle_iosurface/` hit this
// bug during validation; the comment is preserved here to prevent
// future repeats.
// ---------------------------------------------------------------------------

const EGL_IOSURFACE_ANGLE: u32 = 0x3454;
const EGL_IOSURFACE_PLANE_ANGLE: i32 = 0x345A;
#[allow(dead_code)] // referenced by the macOS texture target attribute below
const EGL_TEXTURE_RECTANGLE_ANGLE: i32 = 0x345B;
const EGL_TEXTURE_TYPE_ANGLE: i32 = 0x345C;
const EGL_TEXTURE_INTERNAL_FORMAT_ANGLE: i32 = 0x345D;
pub(super) const EGL_BIND_TO_TEXTURE_TARGET_ANGLE: i32 = 0x348D;
const EGL_TEXTURE_TARGET: i32 = 0x3081;
const EGL_TEXTURE_FORMAT: i32 = 0x3080;
const EGL_TEXTURE_RGBA: i32 = 0x305E;
const EGL_TEXTURE_2D: i32 = 0x305F;

// GL constants used in attribute lists (GL_RG, GL_BGRA_EXT, etc.) — the
// `gles` crate isn't directly available here, so hardcode the values
// from the spike. These have been part of OpenGL ES since 3.0 / the
// GL_EXT_texture_format_BGRA8888 extension.
const GL_RG: i32 = 0x8227;
const GL_RGBA: i32 = 0x1908;
const GL_BGRA_EXT: i32 = 0x80E1;
const GL_UNSIGNED_BYTE: i32 = 0x1401;

// IOSurface FOURCC pixel-format codes recognized by ANGLE's Metal
// backend. The spike validated these against `EGL_BAD_ATTRIBUTE`-style
// failures.
const FOURCC_2C08: u32 = u32::from_be_bytes(*b"2C08"); // 2-channel 8-bit (YUYV-as-GL_RG)
const FOURCC_RGBA: u32 = u32::from_be_bytes(*b"RGBA"); // 32-bit RGBA8888
const FOURCC_BGRA: u32 = u32::from_be_bytes(*b"BGRA"); // 32-bit BGRA8888

// ---------------------------------------------------------------------------
// Raw IOSurface allocation helpers.
//
// Image tensors on macOS need IOSurfaces with the proper image FourCC
// (YUYV/NV12/BGRA) and per-pixel byte count — different from the
// generic byte-bag IOSurface that `crates/tensor/src/iosurface.rs`
// allocates for arbitrary tensor shapes. This module owns the image-
// specific layout logic; the tensor crate owns the generic case.
// ---------------------------------------------------------------------------

// CoreFoundation is also linked from crates/tensor/src/iosurface.rs;
// duplicate `kind = "framework"` attribute is harmless and required.
#[allow(clippy::duplicated_attributes)]
#[link(name = "IOSurface", kind = "framework")]
#[link(name = "CoreFoundation", kind = "framework")]
extern "C" {
    fn IOSurfaceCreate(properties: *mut std::ffi::c_void) -> *mut std::ffi::c_void;
    fn CFRelease(cf: *const std::ffi::c_void);

    fn CFDictionaryCreateMutable(
        allocator: *const std::ffi::c_void,
        capacity: isize,
        key_callbacks: *const std::ffi::c_void,
        value_callbacks: *const std::ffi::c_void,
    ) -> *mut std::ffi::c_void;
    fn CFDictionarySetValue(
        dict: *mut std::ffi::c_void,
        key: *const std::ffi::c_void,
        value: *const std::ffi::c_void,
    );
    fn CFStringCreateWithCString(
        allocator: *const std::ffi::c_void,
        cstr: *const i8,
        encoding: u32,
    ) -> *mut std::ffi::c_void;
    fn CFNumberCreate(
        allocator: *const std::ffi::c_void,
        ty: i32,
        value_ptr: *const std::ffi::c_void,
    ) -> *mut std::ffi::c_void;

    static kCFTypeDictionaryKeyCallBacks: std::ffi::c_void;
    static kCFTypeDictionaryValueCallBacks: std::ffi::c_void;
}

const K_CF_NUMBER_LONG_TYPE: i32 = 10;
const K_CF_STRING_ENCODING_UTF8: u32 = 0x08000100;

/// IOSurface layout parameters for image-backed surfaces.
///
/// `fourcc` and `bytes_per_element` come from
/// [`edgefirst_tensor::image_iosurface_layout`] — the single source of
/// truth for the `PixelFormat → (FourCC, bpe)` mapping. The image crate
/// only owns the FourCC → GL-internal-format map below, since the GL
/// constants are an image-side concern.
struct ImageLayout {
    fourcc: u32,
    bytes_per_element: usize,
    width: usize,
    height: usize,
}

impl ImageLayout {
    fn for_format(fmt: PixelFormat, width: usize, height: usize) -> Result<Self, Error> {
        let (fourcc, bytes_per_element) = edgefirst_tensor::image_iosurface_layout(fmt)
            .ok_or_else(|| {
                Error::NotImplemented(format!(
                    "IOSurface allocation for PixelFormat::{fmt:?} not yet supported \
                     (no FourCC mapping in edgefirst_tensor::image_iosurface_layout — \
                     multi-plane formats need separate property dictionary setup)"
                ))
            })?;
        Ok(Self {
            fourcc,
            bytes_per_element,
            width,
            height,
        })
    }

    fn gl_type(&self) -> i32 {
        GL_UNSIGNED_BYTE
    }

    fn gl_internal_format(&self) -> i32 {
        // The FourCC ↔ GL-internal-format mapping is image-side: the
        // tensor crate owns the FourCC choice (via `image_iosurface_layout`)
        // and this side owns the GL pairing. Adding a new shader requires
        // both sides to agree.
        match self.fourcc {
            FOURCC_2C08 => GL_RG,
            FOURCC_RGBA => GL_RGBA,
            FOURCC_BGRA => GL_BGRA_EXT,
            // Defensive fallback — should be unreachable given
            // `image_iosurface_layout` only returns the three above.
            _ => GL_BGRA_EXT,
        }
    }
}

/// Build the CFDictionary describing an image-backed IOSurface.
///
/// # Safety
///
/// The returned `CFDictionaryRef` must be released by the caller with
/// `CFRelease` after passing to `IOSurfaceCreate`.
unsafe fn build_image_props(layout: &ImageLayout) -> Result<*mut std::ffi::c_void, Error> {
    let bpr = (layout.width * layout.bytes_per_element + 63) & !63;
    let alloc_size = bpr * layout.height;

    let dict = CFDictionaryCreateMutable(
        std::ptr::null(),
        0,
        &kCFTypeDictionaryKeyCallBacks,
        &kCFTypeDictionaryValueCallBacks,
    );
    if dict.is_null() {
        return Err(Error::Io(std::io::Error::other(
            "CFDictionaryCreateMutable returned null",
        )));
    }

    let set_num = |key: &str, value: i64| -> Result<(), Error> {
        let key_c = std::ffi::CString::new(key)
            .map_err(|e| Error::Internal(format!("CString: {e}")))?;
        let key_cf =
            CFStringCreateWithCString(std::ptr::null(), key_c.as_ptr(), K_CF_STRING_ENCODING_UTF8);
        if key_cf.is_null() {
            return Err(Error::Io(std::io::Error::other(
                "CFStringCreateWithCString returned null",
            )));
        }
        let value_cf = CFNumberCreate(
            std::ptr::null(),
            K_CF_NUMBER_LONG_TYPE,
            &value as *const i64 as *const std::ffi::c_void,
        );
        if value_cf.is_null() {
            CFRelease(key_cf);
            return Err(Error::Io(std::io::Error::other(
                "CFNumberCreate returned null",
            )));
        }
        CFDictionarySetValue(dict, key_cf, value_cf);
        CFRelease(key_cf);
        CFRelease(value_cf);
        Ok(())
    };

    let result = (|| -> Result<(), Error> {
        set_num("IOSurfaceWidth", layout.width as i64)?;
        set_num("IOSurfaceHeight", layout.height as i64)?;
        set_num("IOSurfaceBytesPerElement", layout.bytes_per_element as i64)?;
        set_num("IOSurfacePixelFormat", layout.fourcc as i64)?;
        set_num("IOSurfaceBytesPerRow", bpr as i64)?;
        set_num("IOSurfaceAllocSize", alloc_size as i64)?;
        Ok(())
    })();

    if let Err(e) = result {
        CFRelease(dict);
        return Err(e);
    }
    Ok(dict)
}

/// Create an image-backed IOSurface for the given pixel format and
/// dimensions. Returns a raw `IOSurfaceRef` whose ownership is
/// transferred to the caller (release with `CFRelease`).
///
/// # Safety
///
/// The returned pointer must be released with `CFRelease` exactly once.
pub(super) unsafe fn create_image_iosurface(
    fmt: PixelFormat,
    width: usize,
    height: usize,
) -> Result<*mut std::ffi::c_void, Error> {
    let layout = ImageLayout::for_format(fmt, width, height)?;
    let dict = build_image_props(&layout)?;
    let surface = IOSurfaceCreate(dict);
    CFRelease(dict);
    if surface.is_null() {
        return Err(Error::Io(std::io::Error::other(
            "IOSurfaceCreate returned null — likely memory pressure or invalid layout",
        )));
    }
    Ok(surface)
}

// ---------------------------------------------------------------------------
// EGL pbuffer import via EGL_ANGLE_iosurface_client_buffer
// ---------------------------------------------------------------------------

/// Function-pointer type for `eglCreatePbufferFromClientBuffer` —
/// looked up via `eglGetProcAddress` at runtime.
type FnCreatePbufferFromClientBuffer = unsafe extern "C" fn(
    dpy: egl::EGLDisplay,
    buftype: u32,
    buffer: egl::EGLClientBuffer,
    config: egl::EGLConfig,
    attrib_list: *const i32,
) -> egl::EGLSurface;

/// Bind an IOSurface to an EGL pbuffer via
/// `EGL_ANGLE_iosurface_client_buffer`. The pbuffer can then be bound
/// as a texture via `eglBindTexImage` for sampling or as a renderbuffer
/// attachment for drawing.
///
/// # Safety
///
/// `surface_ref` must be a valid IOSurfaceRef live for the duration of
/// the returned pbuffer's lifetime. `cfg` must be an EGL config with
/// `EGL_BIND_TO_TEXTURE_TARGET_ANGLE = EGL_TEXTURE_2D` selected.
pub(super) unsafe fn create_iosurface_pbuffer(
    egl: &super::Egl,
    display: egl::Display,
    config: egl::Config,
    surface_ref: *mut std::ffi::c_void,
    fmt: PixelFormat,
    width: usize,
    height: usize,
) -> Result<egl::Surface, Error> {
    let layout = ImageLayout::for_format(fmt, width, height)?;

    let create_pbuffer_ptr = egl
        .get_proc_address("eglCreatePbufferFromClientBuffer")
        .ok_or_else(|| {
            Error::Io(std::io::Error::other(
                "eglCreatePbufferFromClientBuffer not exported by ANGLE libEGL",
            ))
        })?;
    let create_pbuffer: FnCreatePbufferFromClientBuffer =
        std::mem::transmute(create_pbuffer_ptr);

    let attribs = [
        egl::WIDTH,
        width as i32,
        egl::HEIGHT,
        height as i32,
        EGL_IOSURFACE_PLANE_ANGLE,
        0,
        EGL_TEXTURE_TARGET,
        EGL_TEXTURE_2D,
        EGL_TEXTURE_INTERNAL_FORMAT_ANGLE,
        layout.gl_internal_format(),
        EGL_TEXTURE_FORMAT,
        EGL_TEXTURE_RGBA,
        EGL_TEXTURE_TYPE_ANGLE,
        layout.gl_type(),
        egl::NONE,
    ];

    let raw = create_pbuffer(
        display.as_ptr(),
        EGL_IOSURFACE_ANGLE,
        surface_ref as egl::EGLClientBuffer,
        config.as_ptr(),
        attribs.as_ptr(),
    );
    if raw.is_null() {
        let egl_err = egl.get_error();
        return Err(Error::Io(std::io::Error::other(format!(
            "eglCreatePbufferFromClientBuffer(EGL_IOSURFACE_ANGLE) failed: {egl_err:?}"
        ))));
    }
    Ok(egl::Surface::from_ptr(raw))
}

/// Extract the IOSurface backing a tensor (macOS only).
///
/// Returns `None` if the tensor isn't IOSurface-backed (e.g. SHM/Mem).
/// The returned pointer is borrowed — its lifetime is tied to the
/// underlying tensor.
pub(super) fn tensor_iosurface_ref(tensor: &Tensor<u8>) -> Option<*mut std::ffi::c_void> {
    // Inspect the tensor's memory backend; only TensorMemory::Dma (which
    // is IOSurface-backed on macOS) carries the right inner type.
    if !matches!(
        tensor.memory(),
        edgefirst_tensor::TensorMemory::Dma
    ) {
        return None;
    }
    tensor.iosurface_ref()
}
