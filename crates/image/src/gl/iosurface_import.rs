// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! IOSurface allocation and EGL pbuffer import for the macOS/iOS GL backend.
//!
//! Mirrors the role of `dma_import.rs` on Linux: takes a tensor that
//! carries platform-native zero-copy GPU buffer (IOSurface on macOS/iOS,
//! DMA-BUF on Linux) and produces the EGL handle the GL backend can
//! sample from or render to.
//!
//! The EGL flow is structurally different from Linux. Linux uses
//! `eglCreateImageKHR` with `EGL_LINUX_DMA_BUF_EXT` to produce an
//! `EGLImage` that is bound to a texture via
//! `glEGLImageTargetTexture2DOES`. macOS/iOS uses
//! `eglCreatePbufferFromClientBuffer` with `EGL_IOSURFACE_ANGLE` to
//! produce an `EGLSurface` (pbuffer) that is bound to a texture via
//! `eglBindTexImage`. The macOS/iOS path is invoked from
//! the engine's zero-copy source attach (`draw_src_texture`);
//! Linux callers do not go through this module.
//!
//! See `spikes/angle_iosurface/` (local, gitignored) for the proof-of-
//! concept that validates each constant + attribute combination.

#![cfg(any(target_os = "macos", target_os = "ios"))]

use crate::Error;
use edgefirst_egl as egl;
use edgefirst_tensor::{packed_rgba16f_layout, DType, PixelFormat, Tensor, TensorTrait};

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
const GL_RED: i32 = 0x1903;
const GL_RG: i32 = 0x8227;
const GL_RGBA: i32 = 0x1908;
const GL_BGRA_EXT: i32 = 0x80E1;
const GL_UNSIGNED_BYTE: i32 = 0x1401;
const GL_HALF_FLOAT: i32 = 0x140B;

// IOSurface FOURCC pixel-format codes recognized by ANGLE's Metal
// backend. Per `EGL_ANGLE_iosurface_client_buffer.txt`, the only
// supported float `(type, internal_format)` pair is `(GL_HALF_FLOAT,
// GL_RGBA)` = RGBA16F, mapped from FourCC `'RGhA'`
// (kCVPixelFormatType_64RGBAHalf). No R32F, R16F, or RGBA32F binding
// is accepted by the spec — calls return `EGL_BAD_ATTRIBUTE`.
const FOURCC_L008: u32 = u32::from_be_bytes(*b"L008"); // 1-channel 8-bit (GREY / semi-planar YUV byte plane, GL_RED)
const FOURCC_2C08: u32 = u32::from_be_bytes(*b"2C08"); // 2-channel 8-bit (YUYV-as-GL_RG)
const FOURCC_RGBA: u32 = u32::from_be_bytes(*b"RGBA"); // 32-bit RGBA8888
const FOURCC_BGRA: u32 = u32::from_be_bytes(*b"BGRA"); // 32-bit BGRA8888
const FOURCC_RGHA: u32 = u32::from_be_bytes(*b"RGhA"); // kCVPixelFormatType_64RGBAHalf

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
/// truth for the `(PixelFormat, DType) → (FourCC, bpe)` mapping. The
/// image crate only owns the FourCC → GL-internal-format map below,
/// since the GL constants are an image-side concern.
///
/// Surface dimensions:
///   * Packed formats: `(width, height)`.
///   * PlanarRgb / PlanarRgba F16: `(width / 4, channels * height)` —
///     4 contiguous half-floats packed per RGBA16F pixel because
///     ANGLE only supports `GL_HALF_FLOAT + GL_RGBA` (no
///     single-channel float). The tensor crate enforces
///     `width % 4 == 0`.
#[cfg_attr(test, derive(Debug))]
struct ImageLayout {
    fourcc: u32,
    bytes_per_element: usize,
    /// Logical image width (consumer-visible).
    width: usize,
    /// Logical image height (without channel-plane multiplication).
    height: usize,
    /// Physical IOSurface width (= width / 4 for packed-planar F16).
    surface_width: usize,
    /// Physical IOSurface height (= height * channels for planar formats).
    surface_height: usize,
    dtype: DType,
    fmt: PixelFormat,
}

impl ImageLayout {
    fn for_format(
        fmt: PixelFormat,
        dtype: DType,
        width: usize,
        height: usize,
    ) -> Result<Self, Error> {
        let (fourcc, bytes_per_element) = edgefirst_tensor::image_iosurface_layout(fmt, dtype)
            .ok_or_else(|| {
                Error::NotImplemented(format!(
                    "IOSurface allocation for ({fmt:?}, {dtype:?}) not yet supported \
                     (no FourCC mapping in edgefirst_tensor::image_iosurface_layout — \
                     multi-plane formats and unsupported dtypes need separate handling)"
                ))
            })?;
        let (surface_width, surface_height) = match (fmt, dtype) {
            (PixelFormat::PlanarRgb | PixelFormat::PlanarRgba, DType::F16) => {
                // Single source of truth for the RGBA16F-packed `(W/4, C*H)`
                // geometry — see `edgefirst_tensor::packed_rgba16f_layout`.
                // It returns `None` on width%4 != 0 or `usize` overflow.
                let layout = packed_rgba16f_layout(fmt, dtype, width, height).ok_or_else(|| {
                    Error::Internal(format!(
                        "{fmt:?} F16 RGBA16F packing requires width%4==0 and \
                         non-overflowing geometry (got {width}x{height})"
                    ))
                })?;
                (layout.surface_w, layout.surface_h)
            }
            (PixelFormat::PlanarRgb, _) => {
                let sh = height.checked_mul(3).ok_or_else(|| {
                    Error::Internal(format!("PlanarRgb surface height overflow (h={height})"))
                })?;
                (width, sh)
            }
            (PixelFormat::PlanarRgba, _) => {
                let sh = height.checked_mul(4).ok_or_else(|| {
                    Error::Internal(format!("PlanarRgba surface height overflow (h={height})"))
                })?;
                (width, sh)
            }
            // Semi-planar YUV bound as a single R8 plane: the combined-plane
            // surface sized to the 64-aligned row pitch (matches the tensor
            // allocation in `iosurface::new_image`; see
            // `PixelFormat::semi_planar_surface_dims` for the ANGLE width==pitch
            // rationale — the single source of truth for both allocators).
            (PixelFormat::Nv12 | PixelFormat::Nv16 | PixelFormat::Nv24, _) => fmt
                .semi_planar_surface_dims(width, height, bytes_per_element)
                .ok_or_else(|| {
                    Error::Internal(format!(
                        "{fmt:?} has no semi-planar surface dims for {width}x{height}"
                    ))
                })?,
            _ => (width, height),
        };
        Ok(Self {
            fourcc,
            bytes_per_element,
            width,
            height,
            surface_width,
            surface_height,
            dtype,
            fmt,
        })
    }

    // gl_type / gl_internal_format return `Result` rather than `unreachable!`:
    // a `(dtype, FourCC)` that `image_iosurface_layout` accepts but these maps
    // don't handle is a table-drift developer error (adding a format requires
    // updating both the tensor-crate FourCC table and these GL pairings), not a
    // runtime condition — but a clean error beats a panic if the two ever drift.
    fn gl_type(&self) -> Result<i32, Error> {
        match self.dtype {
            DType::U8 | DType::I8 => Ok(GL_UNSIGNED_BYTE),
            // Per `EGL_ANGLE_iosurface_client_buffer.txt`, F16 is the
            // only ANGLE-supported float; we always use the packed
            // `GL_HALF_FLOAT + GL_RGBA` binding regardless of whether
            // the destination is packed Rgba or planar (planar uses
            // 4-element pixel packing).
            DType::F16 => Ok(GL_HALF_FLOAT),
            other => Err(Error::NotSupported(format!(
                "ImageLayout::gl_type: dtype {other:?} has no GL/IOSurface type mapping \
                 (table drift vs image_iosurface_layout)"
            ))),
        }
    }

    fn gl_internal_format(&self) -> Result<i32, Error> {
        // The FourCC ↔ GL-internal-format mapping is image-side: the
        // tensor crate owns the FourCC choice (via `image_iosurface_layout`)
        // and this side owns the GL pairing. Adding a new shader requires
        // both sides to agree.
        match self.fourcc {
            FOURCC_L008 => Ok(GL_RED),
            FOURCC_2C08 => Ok(GL_RG),
            FOURCC_RGBA => Ok(GL_RGBA),
            FOURCC_BGRA => Ok(GL_BGRA_EXT),
            // RGBA16F: ANGLE's own
            // `EGLIOSurfaceClientBufferTest::RenderToRGBA16FIOSurface`
            // test uses the UNSIZED `GL_RGBA` (paired with
            // `GL_HALF_FLOAT`) in the EGL pbuffer attribs. ANGLE
            // internally promotes that pair to the sized RGBA16F
            // framebuffer; passing the sized `GL_RGBA16F` directly
            // is rejected with `EGL_BAD_ATTRIBUTE`. See ANGLE source
            // `src/tests/egl_tests/EGLIOSurfaceClientBufferTest.cpp`.
            FOURCC_RGHA => Ok(GL_RGBA),
            other => Err(Error::NotSupported(format!(
                "unsupported IOSurface FourCC 0x{other:08X} in GL import \
                 (table drift vs image_iosurface_layout)"
            ))),
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
    // Checked arithmetic: an overflowing bytes-per-row or allocation size
    // would describe an under-sized IOSurface, which the GL import then
    // treats as a valid render target / mapped buffer — a memory-safety
    // hazard. Fail loudly instead of wrapping.
    let bpr = layout
        .surface_width
        .checked_mul(layout.bytes_per_element)
        .and_then(|b| b.checked_add(63))
        .map(|b| b & !63)
        .ok_or_else(|| {
            Error::Internal(format!(
                "IOSurface bytes-per-row overflow (surface_width={}, bpe={})",
                layout.surface_width, layout.bytes_per_element
            ))
        })?;
    let alloc_size = bpr.checked_mul(layout.surface_height).ok_or_else(|| {
        Error::Internal(format!(
            "IOSurface allocation size overflow (bpr={bpr}, surface_height={})",
            layout.surface_height
        ))
    })?;

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
        let key_c =
            std::ffi::CString::new(key).map_err(|e| Error::Internal(format!("CString: {e}")))?;
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
        set_num("IOSurfaceWidth", layout.surface_width as i64)?;
        set_num("IOSurfaceHeight", layout.surface_height as i64)?;
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
    dtype: DType,
    width: usize,
    height: usize,
) -> Result<*mut std::ffi::c_void, Error> {
    let layout = ImageLayout::for_format(fmt, dtype, width, height)?;
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
//
// Eight args is one over clippy's default; every one is needed for the
// `eglCreatePbufferFromClientBuffer` call (EGL display + config,
// IOSurface ref, plus the image shape used by `ImageLayout`). The
// caller already groups these naturally; bundling into a struct would
// just push the verbosity to the call site.
#[allow(clippy::too_many_arguments)]
pub(super) unsafe fn create_iosurface_pbuffer(
    egl: &super::Egl,
    display: egl::Display,
    config: egl::Config,
    surface_ref: *mut std::ffi::c_void,
    fmt: PixelFormat,
    dtype: DType,
    width: usize,
    height: usize,
) -> Result<egl::Surface, Error> {
    let layout = ImageLayout::for_format(fmt, dtype, width, height)?;

    let create_pbuffer_ptr = egl
        .get_proc_address("eglCreatePbufferFromClientBuffer")
        .ok_or_else(|| {
            Error::Io(std::io::Error::other(
                "eglCreatePbufferFromClientBuffer not exported by ANGLE libEGL",
            ))
        })?;
    let create_pbuffer: FnCreatePbufferFromClientBuffer = std::mem::transmute(create_pbuffer_ptr);

    let gl_internal_format = layout.gl_internal_format()?;
    let gl_type = layout.gl_type()?;
    let attribs = [
        egl::WIDTH,
        layout.surface_width as i32,
        egl::HEIGHT,
        layout.surface_height as i32,
        EGL_IOSURFACE_PLANE_ANGLE,
        0,
        EGL_TEXTURE_TARGET,
        EGL_TEXTURE_2D,
        EGL_TEXTURE_INTERNAL_FORMAT_ANGLE,
        gl_internal_format,
        EGL_TEXTURE_FORMAT,
        EGL_TEXTURE_RGBA,
        EGL_TEXTURE_TYPE_ANGLE,
        gl_type,
        egl::NONE,
    ];

    // DIAGNOSTIC: trace every input to eglCreatePbufferFromClientBuffer
    // so we can correlate failures with the actual arguments. Hot loop
    // safe — only logged at trace level.
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
            "eglCreatePbufferFromClientBuffer(EGL_IOSURFACE_ANGLE) failed: \
             {egl_err:?} (surface_ref={surface_ref:?}, \
             surface={surface_w}x{surface_h}, fourcc=0x{fc:08x}, \
             internal_format=0x{ifmt:04x}, type=0x{ty:04x}, bpe={bpe})",
            surface_w = layout.surface_width,
            surface_h = layout.surface_height,
            fc = layout.fourcc,
            ifmt = gl_internal_format,
            ty = gl_type,
            bpe = layout.bytes_per_element,
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
    if !matches!(tensor.memory(), edgefirst_tensor::TensorMemory::Dma) {
        return None;
    }
    tensor.iosurface_ref()
}

// `ImageLayout::for_format` is pure geometry/validation (no GL/IOSurface
// allocation), so it is unit-testable on the macOS coverage lane without a
// GPU. These cover the surface-dimension computation and every error arm.
#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use super::ImageLayout;
    use edgefirst_tensor::{DType, PixelFormat};

    #[test]
    fn planar_rgb_f16_packs_to_quarter_width_triple_height() {
        // 16x16 PlanarRgb F16 → RGBA16F-packed (W/4, 3*H) = (4, 48).
        let layout = ImageLayout::for_format(PixelFormat::PlanarRgb, DType::F16, 16, 16).unwrap();
        assert_eq!(layout.surface_width, 4);
        assert_eq!(layout.surface_height, 48);
        assert_eq!((layout.width, layout.height), (16, 16));
    }

    #[test]
    fn planar_rgba_f16_packs_to_quarter_width_quadruple_height() {
        let layout = ImageLayout::for_format(PixelFormat::PlanarRgba, DType::F16, 16, 16).unwrap();
        assert_eq!(layout.surface_width, 4);
        assert_eq!(layout.surface_height, 64);
    }

    #[test]
    fn planar_rgb_f16_misaligned_width_errors() {
        // width % 4 != 0 cannot be RGBA16F-packed.
        let err = ImageLayout::for_format(PixelFormat::PlanarRgb, DType::F16, 15, 16).unwrap_err();
        assert!(
            matches!(err, crate::Error::Internal(_)),
            "expected Internal, got {err:?}"
        );
    }

    #[test]
    fn unmapped_format_dtype_is_not_implemented() {
        // A (format, dtype) with no FourCC mapping in image_iosurface_layout
        // returns NotImplemented rather than a bogus layout.
        let err = ImageLayout::for_format(PixelFormat::Nv12, DType::F32, 16, 16).unwrap_err();
        assert!(
            matches!(err, crate::Error::NotImplemented(_)),
            "expected NotImplemented, got {err:?}"
        );
    }

    #[test]
    fn packed_u8_format_uses_logical_dimensions() {
        // Non-planar packed RGBA8 keeps logical (W, H) as the surface size.
        let layout = ImageLayout::for_format(PixelFormat::Rgba, DType::U8, 16, 16).unwrap();
        assert_eq!((layout.surface_width, layout.surface_height), (16, 16));
    }
}
