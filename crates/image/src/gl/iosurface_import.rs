// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! IOSurface allocation and EGL pbuffer import for the macOS/iOS GL backend.
//!
//! Takes a tensor that carries an IOSurface and produces the EGL handle the
//! GL backend samples from or renders to:
//! `eglCreatePbufferFromClientBuffer` with `EGL_IOSURFACE_ANGLE` produces an
//! `EGLSurface` (pbuffer) that is bound to a texture via `eglBindTexImage`.

#![cfg(any(target_os = "macos", target_os = "ios"))]

use crate::Error;
use edgefirst_egl as egl;
use edgefirst_tensor::{packed_rgba16f_layout, DType, PixelFormat, Tensor, TensorTrait};

// ---------------------------------------------------------------------------
// ANGLE EGL constants for the IOSurface client-buffer path, from
// `include/EGL/eglext_angle.h` in the ANGLE source tree.
// ---------------------------------------------------------------------------

const EGL_IOSURFACE_ANGLE: u32 = 0x3454;
const EGL_IOSURFACE_PLANE_ANGLE: i32 = 0x345A;
#[allow(dead_code)]
const EGL_TEXTURE_RECTANGLE_ANGLE: i32 = 0x345B;
const EGL_TEXTURE_TYPE_ANGLE: i32 = 0x345C;
const EGL_TEXTURE_INTERNAL_FORMAT_ANGLE: i32 = 0x345D;
pub(super) const EGL_BIND_TO_TEXTURE_TARGET_ANGLE: i32 = 0x348D;
const EGL_TEXTURE_TARGET: i32 = 0x3081;
const EGL_TEXTURE_FORMAT: i32 = 0x3080;
const EGL_TEXTURE_RGBA: i32 = 0x305E;
const EGL_TEXTURE_2D: i32 = 0x305F;

// GL constants used in attribute lists (OpenGL ES 3.0 and
// GL_EXT_texture_format_BGRA8888).
pub(super) const GL_RED: i32 = 0x1903;
pub(super) const GL_RG: i32 = 0x8227;
const GL_RGBA: i32 = 0x1908;
const GL_BGRA_EXT: i32 = 0x80E1;
const GL_UNSIGNED_BYTE: i32 = 0x1401;
const GL_HALF_FLOAT: i32 = 0x140B;

// IOSurface FOURCC pixel-format codes recognized by ANGLE's Metal backend.
const FOURCC_L008: u32 = u32::from_be_bytes(*b"L008"); // 1-channel 8-bit (GREY / semi-planar YUV byte plane, GL_RED)
const FOURCC_2C08: u32 = u32::from_be_bytes(*b"2C08"); // 2-channel 8-bit (YUYV-as-GL_RG)
const FOURCC_RGBA: u32 = u32::from_be_bytes(*b"RGBA"); // 32-bit RGBA8888
const FOURCC_BGRA: u32 = u32::from_be_bytes(*b"BGRA"); // 32-bit BGRA8888
const FOURCC_RGHA: u32 = u32::from_be_bytes(*b"RGhA"); // kCVPixelFormatType_64RGBAHalf

// ---------------------------------------------------------------------------
// Raw IOSurface allocation helpers for image tensors.
// ---------------------------------------------------------------------------

#[allow(clippy::duplicated_attributes)]
#[link(name = "IOSurface", kind = "framework")]
#[link(name = "CoreFoundation", kind = "framework")]
extern "C" {
    fn IOSurfaceCreate(properties: *mut std::ffi::c_void) -> *mut std::ffi::c_void;
    fn IOSurfaceGetID(surface: *mut std::ffi::c_void) -> u32;
    fn IOSurfaceGetPlaneCount(surface: *mut std::ffi::c_void) -> usize;
    fn IOSurfaceGetBaseAddress(surface: *mut std::ffi::c_void) -> *mut std::ffi::c_void;
    fn IOSurfaceGetBaseAddressOfPlane(
        surface: *mut std::ffi::c_void,
        plane: usize,
    ) -> *mut std::ffi::c_void;
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
/// [`edgefirst_tensor::image_iosurface_layout`].
///
/// Surface dimensions:
///   * Packed formats: `(width, height)`.
///   * PlanarRgb / PlanarRgba F16: `(width / 4, channels * height)` —
///     4 contiguous half-floats packed per RGBA16F pixel. Requires
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
                // The RGBA16F-packed `(W/4, C*H)` geometry from
                // `edgefirst_tensor::packed_rgba16f_layout`: `None` on
                // width%4 != 0 or `usize` overflow.
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
            // surface sized to the 64-aligned row pitch.
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

    fn gl_type(&self) -> Result<i32, Error> {
        match self.dtype {
            DType::U8 | DType::I8 => Ok(GL_UNSIGNED_BYTE),
            // F16 uses the packed `GL_HALF_FLOAT + GL_RGBA` binding for
            // packed Rgba and planar destinations alike.
            DType::F16 => Ok(GL_HALF_FLOAT),
            other => Err(Error::NotSupported(format!(
                "ImageLayout::gl_type: dtype {other:?} has no GL/IOSurface type mapping \
                 (table drift vs image_iosurface_layout)"
            ))),
        }
    }

    fn gl_internal_format(&self) -> Result<i32, Error> {
        match self.fourcc {
            FOURCC_L008 => Ok(GL_RED),
            FOURCC_2C08 => Ok(GL_RG),
            FOURCC_RGBA => Ok(GL_RGBA),
            FOURCC_BGRA => Ok(GL_BGRA_EXT),
            // RGBA16F: the unsized `GL_RGBA`, paired with `GL_HALF_FLOAT`.
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
    // Checked arithmetic: overflow is an error.
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

    let dict = unsafe {
        CFDictionaryCreateMutable(
            std::ptr::null(),
            0,
            &kCFTypeDictionaryKeyCallBacks,
            &kCFTypeDictionaryValueCallBacks,
        )
    };
    if dict.is_null() {
        return Err(Error::Io(std::io::Error::other(
            "CFDictionaryCreateMutable returned null",
        )));
    }

    let set_num = |key: &str, value: i64| -> Result<(), Error> {
        let key_c =
            std::ffi::CString::new(key).map_err(|e| Error::Internal(format!("CString: {e}")))?;
        let key_cf = unsafe {
            CFStringCreateWithCString(std::ptr::null(), key_c.as_ptr(), K_CF_STRING_ENCODING_UTF8)
        };
        if key_cf.is_null() {
            return Err(Error::Io(std::io::Error::other(
                "CFStringCreateWithCString returned null",
            )));
        }
        let value_cf = unsafe {
            CFNumberCreate(
                std::ptr::null(),
                K_CF_NUMBER_LONG_TYPE,
                &value as *const i64 as *const std::ffi::c_void,
            )
        };
        if value_cf.is_null() {
            unsafe { CFRelease(key_cf) };
            return Err(Error::Io(std::io::Error::other(
                "CFNumberCreate returned null",
            )));
        }
        unsafe {
            CFDictionarySetValue(dict, key_cf, value_cf);
            CFRelease(key_cf);
            CFRelease(value_cf);
        }
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
        unsafe { CFRelease(dict) };
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
    let dict = unsafe { build_image_props(&layout)? };
    let surface = unsafe { IOSurfaceCreate(dict) };
    unsafe { CFRelease(dict) };
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

/// Looks up `eglCreatePbufferFromClientBuffer` in ANGLE's libEGL.
fn create_pbuffer_fn(egl: &super::Egl) -> Result<FnCreatePbufferFromClientBuffer, Error> {
    let ptr = egl
        .get_proc_address("eglCreatePbufferFromClientBuffer")
        .ok_or_else(|| {
            Error::Io(std::io::Error::other(
                "eglCreatePbufferFromClientBuffer not exported by ANGLE libEGL",
            ))
        })?;
    // SAFETY: the symbol has the `eglCreatePbufferFromClientBuffer` signature.
    Ok(
        unsafe {
            std::mem::transmute::<extern "system" fn(), FnCreatePbufferFromClientBuffer>(ptr)
        },
    )
}

/// The `EGL_ANGLE_iosurface_client_buffer` attribute list for a
/// `width`x`height` binding of IOSurface plane `plane`.
fn pbuffer_attribs(
    width: usize,
    height: usize,
    plane: i32,
    internal_format: i32,
    gl_type: i32,
) -> [i32; 15] {
    [
        egl::WIDTH,
        width as i32,
        egl::HEIGHT,
        height as i32,
        EGL_IOSURFACE_PLANE_ANGLE,
        plane,
        EGL_TEXTURE_TARGET,
        EGL_TEXTURE_2D,
        EGL_TEXTURE_INTERNAL_FORMAT_ANGLE,
        internal_format,
        EGL_TEXTURE_FORMAT,
        EGL_TEXTURE_RGBA,
        EGL_TEXTURE_TYPE_ANGLE,
        gl_type,
        egl::NONE,
    ]
}

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
    let create_pbuffer = create_pbuffer_fn(egl)?;

    let gl_internal_format = layout.gl_internal_format()?;
    let gl_type = layout.gl_type()?;
    let attribs = pbuffer_attribs(
        layout.surface_width,
        layout.surface_height,
        0,
        gl_internal_format,
        gl_type,
    );

    let raw = unsafe {
        create_pbuffer(
            display.as_ptr(),
            EGL_IOSURFACE_ANGLE,
            surface_ref as egl::EGLClientBuffer,
            config.as_ptr(),
            attribs.as_ptr(),
        )
    };
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
    Ok(unsafe { egl::Surface::from_ptr(raw) })
}

/// Bind plane `plane` of a planar IOSurface to an EGL pbuffer as a
/// `width`x`height` texture of `internal_format` (`GL_RED` or `GL_RG`) with
/// `GL_UNSIGNED_BYTE` components.
///
/// # Errors
///
/// `Error::Io` when ANGLE does not export `eglCreatePbufferFromClientBuffer`
/// or refuses the binding.
///
/// # Safety
///
/// As [`create_iosurface_pbuffer`]. `width` and `height` must not exceed the
/// plane's own extent.
#[allow(clippy::too_many_arguments)]
pub(super) unsafe fn create_iosurface_plane_pbuffer(
    egl: &super::Egl,
    display: egl::Display,
    config: egl::Config,
    surface_ref: *mut std::ffi::c_void,
    plane: i32,
    internal_format: i32,
    width: usize,
    height: usize,
) -> Result<egl::Surface, Error> {
    let create_pbuffer = create_pbuffer_fn(egl)?;
    let attribs = pbuffer_attribs(width, height, plane, internal_format, GL_UNSIGNED_BYTE);
    let raw = unsafe {
        create_pbuffer(
            display.as_ptr(),
            EGL_IOSURFACE_ANGLE,
            surface_ref as egl::EGLClientBuffer,
            config.as_ptr(),
            attribs.as_ptr(),
        )
    };
    if raw.is_null() {
        let egl_err = egl.get_error();
        return Err(Error::Io(std::io::Error::other(format!(
            "eglCreatePbufferFromClientBuffer(EGL_IOSURFACE_ANGLE) failed: \
             {egl_err:?} (surface_ref={surface_ref:?}, plane={plane}, \
             {width}x{height}, internal_format=0x{internal_format:04x})"
        ))));
    }
    Ok(unsafe { egl::Surface::from_ptr(raw) })
}

/// The index of the plane of `surface` that starts `offset` bytes from the
/// surface's base address; `None` when no plane does or the surface is not
/// planar, or the tensor is not IOSurface-backed.
pub(super) fn plane_index<T>(tensor: &Tensor<T>, offset: usize) -> Option<i32>
where
    T: num_traits::Num + Clone + std::fmt::Debug + Send + Sync + edgefirst_tensor::Element,
{
    let surface = tensor.iosurface_ref()?;
    // SAFETY: `surface` is borrowed from `tensor`, which keeps it alive; these
    // calls only read its plane layout.
    unsafe {
        let base = IOSurfaceGetBaseAddress(surface) as usize;
        (0..IOSurfaceGetPlaneCount(surface))
            .find(|&p| {
                (IOSurfaceGetBaseAddressOfPlane(surface, p) as usize).wrapping_sub(base) == offset
            })
            .and_then(|p| i32::try_from(p).ok())
    }
}

/// The `IOSurfaceID` of the tensor's surface; `None` when it is 0 or the
/// tensor is not IOSurface-backed.
pub(super) fn surface_id<T>(tensor: &Tensor<T>) -> Option<u32>
where
    T: num_traits::Num + Clone + std::fmt::Debug + Send + Sync + edgefirst_tensor::Element,
{
    let surface = tensor.iosurface_ref()?;
    // SAFETY: `surface` is borrowed from `tensor`, which keeps it alive.
    Some(unsafe { IOSurfaceGetID(surface) }).filter(|&id| id != 0)
}

/// Extract the IOSurface backing a tensor (macOS only).
///
/// Returns `None` if the tensor isn't IOSurface-backed (e.g. SHM/Mem).
/// The returned pointer is borrowed — its lifetime is tied to the
/// underlying tensor.
pub(super) fn tensor_iosurface_ref(tensor: &Tensor<u8>) -> Option<*mut std::ffi::c_void> {
    // Only TensorMemory::DmaBuf tensors are IOSurface-backed.
    if !matches!(tensor.memory(), edgefirst_tensor::TensorMemory::DmaBuf) {
        return None;
    }
    tensor.iosurface_ref()
}

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use super::ImageLayout;
    use edgefirst_tensor::{DType, PixelFormat};

    #[test]
    fn planar_rgb_f16_packs_to_quarter_width_triple_height() {
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
        let err = ImageLayout::for_format(PixelFormat::PlanarRgb, DType::F16, 15, 16).unwrap_err();
        assert!(
            matches!(err, crate::Error::Internal(_)),
            "expected Internal, got {err:?}"
        );
    }

    #[test]
    fn unmapped_format_dtype_is_not_implemented() {
        let err = ImageLayout::for_format(PixelFormat::Nv12, DType::F32, 16, 16).unwrap_err();
        assert!(
            matches!(err, crate::Error::NotImplemented(_)),
            "expected NotImplemented, got {err:?}"
        );
    }

    #[test]
    fn packed_u8_format_uses_logical_dimensions() {
        let layout = ImageLayout::for_format(PixelFormat::Rgba, DType::U8, 16, 16).unwrap();
        assert_eq!((layout.surface_width, layout.surface_height), (16, 16));
    }
}
