// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! The image processor: lifecycle, and the tensors it mints.
//!
//! Allocating a PBO is the one operation that genuinely requires a processor,
//! because only the owner of the GL context can create one. Everything else a
//! caller might allocate — `mem`, `shm`, `dmabuf` — is available from
//! `libedgefirst-tensor` alone.

use std::ffi::{c_char, c_int, c_void, CStr};
use std::panic::{catch_unwind, AssertUnwindSafe};

use edgefirst_image::ImageProcessor;
use edgefirst_tensor::{
    Compression, CpuAccess, DType, ImageDesc, PixelFormat, TensorDyn, TensorMemory,
};
use edgefirst_tensor_ffi::EfTensor;
use edgefirst_tensor_ffi::{ef_tensor_image_desc_get, EfImageDescView, EfTensorImageDesc};

/// An opaque image processor.
pub struct EfImageProcessor {
    pub(crate) inner: ImageProcessor,
}

/// Create a processor, probing the platform's converters. `NULL` on failure.
#[no_mangle]
pub extern "C" fn ef_image_processor_new() -> *mut EfImageProcessor {
    catch_unwind(|| match ImageProcessor::new() {
        Ok(inner) => Box::into_raw(Box::new(EfImageProcessor { inner })),
        Err(_) => std::ptr::null_mut(),
    })
    .unwrap_or(std::ptr::null_mut())
}

/// Free a processor. Freeing `NULL` is a no-op, matching `free(3)`.
///
/// # Safety
/// `p` must be `NULL` or have come from [`ef_image_processor_new`].
#[no_mangle]
pub unsafe extern "C" fn ef_image_processor_free(p: *mut EfImageProcessor) {
    unsafe {
        if p.is_null() {
            return;
        }
        let _ = catch_unwind(AssertUnwindSafe(|| drop(Box::from_raw(p))));
    }
}

/// Read a C string, rejecting `NULL` and invalid UTF-8.
unsafe fn cstr<'a>(p: *const c_char) -> Option<&'a str> {
    unsafe {
        if p.is_null() {
            return None;
        }
        CStr::from_ptr(p).to_str().ok()
    }
}

/// Allocate an image, returning a tensor.
///
/// `format` is the wire descriptor (`"NV12"`, `"rgb8"`), matching
/// `ef_tensor_builder_format` rather than introducing a second vocabulary.
/// `dtype` and `access` are the shared integer codes.
///
/// `storage` selects the backing; pass `EF_STORAGE_KIND_PBO` for the case that
/// actually needs a processor. The result is an ordinary `ef_tensor` — read it
/// with `ef_tensor_shape`, release it with `ef_tensor_free`.
///
/// @return a tensor the caller owns, or `NULL`.
///
/// # Safety
/// `format` must be `NULL` or a NUL-terminated string.
#[no_mangle]
pub unsafe extern "C" fn ef_image_processor_create_image(
    p: *mut EfImageProcessor,
    width: usize,
    height: usize,
    format: *const c_char,
    dtype: u32,
    storage: u32,
    access: u32,
) -> *mut EfTensor {
    unsafe {
        catch_unwind(AssertUnwindSafe(|| {
            if p.is_null() || width == 0 || height == 0 {
                return std::ptr::null_mut();
            }
            let (Some(f), Some(fmt)) = (
                cstr(format),
                cstr(format).and_then(PixelFormat::from_str_code),
            ) else {
                return std::ptr::null_mut();
            };
            let _ = f;
            let Some(dt) = DType::from_code(dtype) else {
                return std::ptr::null_mut();
            };
            let Some(acc) = cpu_access_from_code(access) else {
                return std::ptr::null_mut();
            };
            // `storage` is a hint the allocator may decline; `None` lets it choose.
            let mem = TensorMemory::from_code(storage);
            match (*p).inner.create_image(width, height, fmt, dt, mem, acc) {
                Ok(t) => t.into_raw(),
                Err(_) => std::ptr::null_mut(),
            }
        }))
        .unwrap_or(std::ptr::null_mut())
    }
}

/// Rebuild an `ImageDesc` from the scalar view `ef_tensor_image_desc_get`
/// fills -- the single reconstruction site for this direction, paired with
/// tensor-capi's `desc::view_of` for the other. `the_view_round_trips_...`
/// tests there and `a_view_built_by_hand_reconstructs_correctly` below are
/// the proof the two agree.
///
/// `None` for an unrecognized code: a stale vocabulary between two
/// independently-versioned leaves is a real possibility (see `lib.rs`), so
/// this is a request this library declines, never a panic.
fn image_desc_from_view(v: &EfImageDescView) -> Option<ImageDesc> {
    let fmt = PixelFormat::from_code(v.format)?;
    let dt = DType::from_code(v.dtype)?;
    let access = cpu_access_from_code(v.access)?;
    let mut desc = ImageDesc::new(v.width as usize, v.height as usize, fmt, dt).with_access(access);
    if v.has_memory != 0 {
        desc = desc.with_memory(Some(TensorMemory::from_code(v.memory)?));
    }
    if v.has_compression != 0 {
        // 1 = `Any`, the only compression request any `ef_tensor_image_desc_set_*`
        // entry point can create; 2 ("a specific vendor scheme") has no
        // decodable detail behind it here, so it is refused rather than
        // silently downgraded to `Any` or dropped.
        match v.compression {
            1 => desc = desc.with_compression(Compression::Any),
            _ => return None,
        }
    }
    Some(desc)
}

/// Allocate the image an `ef_tensor_image_desc` request describes.
///
/// The request comes from `libedgefirst-tensor`, the type's single
/// implementation home (see `edgefirst_tensor_capi::desc`); this library
/// never dereferences the handle itself, only the scalar view
/// `ef_tensor_image_desc_get` fills. The request is not consumed and may be
/// reused, so one description can fill a pool.
///
/// @return a tensor the caller owns, or `NULL`.
///
/// # Safety
/// `p` and `d` must be live.
#[no_mangle]
pub unsafe extern "C" fn ef_image_processor_create_image_desc(
    p: *mut EfImageProcessor,
    d: *const EfTensorImageDesc,
) -> *mut EfTensor {
    unsafe {
        catch_unwind(AssertUnwindSafe(|| {
            if p.is_null() || d.is_null() {
                return std::ptr::null_mut();
            }
            let mut view = EfImageDescView::default();
            if ef_tensor_image_desc_get(d, &mut view) != 0 {
                return std::ptr::null_mut();
            }
            let Some(desc) = image_desc_from_view(&view) else {
                return std::ptr::null_mut();
            };
            match (*p).inner.create_image_desc(&desc) {
                Ok(t) => t.into_raw(),
                Err(_) => std::ptr::null_mut(),
            }
        }))
        .unwrap_or(std::ptr::null_mut())
    }
}

/// Map a `CpuAccess` code.
///
/// Hand-mapped rather than derived: `CpuAccess` has no shared `code()` in the
/// vocabulary macro, so this is the one place the numbering is asserted. The
/// test below pins it.
pub(crate) fn cpu_access_from_code(code: u32) -> Option<CpuAccess> {
    match code {
        0 => Some(CpuAccess::None),
        1 => Some(CpuAccess::Read),
        2 => Some(CpuAccess::Write),
        3 => Some(CpuAccess::ReadWrite),
        _ => None,
    }
}

/// Borrow the `TensorDyn` behind a handle without taking ownership of it.
///
/// There is now exactly one implementation of `ef_tensor` (all four sibling
/// `-capi` leaves link this crate's shared `libedgefirst_tensor.so` rather
/// than embedding their own copy), so every handle is read the same way
/// regardless of which library minted it -- no per-library dispatch table,
/// no re-import from a foreign platform handle. `TensorDyn::with_raw` wraps
/// the handle in `ManuallyDrop` so the closure's `&mut TensorDyn` never runs
/// `Drop` (which would call `ef_tensor_free`); the caller still owns the
/// handle's lifetime exactly as before.
///
/// # Safety
/// `t` must be `NULL` or a live handle from an EdgeFirst library.
pub(crate) unsafe fn with_tensor<R>(
    t: *const EfTensor,
    f: impl FnOnce(&TensorDyn) -> R,
) -> Result<R, c_int> {
    unsafe {
        if t.is_null() {
            return Err(libc::EINVAL);
        }
        Ok(TensorDyn::with_raw(t as *mut EfTensor, |td| f(td)))
    }
}

/// Borrow the `TensorDyn` behind a destination handle for mutation.
///
/// # Safety
/// `t` must be `NULL` or a live handle from an EdgeFirst library.
pub(crate) unsafe fn with_tensor_mut<R>(
    t: *mut EfTensor,
    f: impl FnOnce(&mut TensorDyn) -> R,
) -> Result<R, c_int> {
    unsafe {
        if t.is_null() {
            return Err(libc::EINVAL);
        }
        Ok(TensorDyn::with_raw(t, f))
    }
}

/// Geometry for a convert: source rectangle and letterbox padding.
///
/// `NULL` anywhere one is accepted means the whole source, stretched to fill.
#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct EfCrop {
    /// Source rectangle in pixels. A zero `width` or `height` means the whole
    /// source, so a zeroed struct is the same as passing `NULL`.
    pub x: u32,
    pub y: u32,
    pub width: u32,
    pub height: u32,
    /// Non-zero to preserve aspect ratio, padding the remainder.
    pub letterbox: c_int,
    /// Letterbox fill colour, RGBA. Ignored unless `letterbox` is set.
    pub pad: [u8; 4],
}

/// Build a `Crop` from the C description.
fn crop_from(c: *const EfCrop) -> edgefirst_image::Crop {
    if c.is_null() {
        return edgefirst_image::Crop::default();
    }
    // SAFETY: checked non-null; `EfCrop` is a plain value.
    let c = unsafe { *c };
    let mut out = if c.letterbox != 0 {
        edgefirst_image::Crop::letterbox(c.pad)
    } else {
        edgefirst_image::Crop::no_crop()
    };
    if c.width != 0 && c.height != 0 {
        out = out.with_source(Some(edgefirst_tensor::Region::new(
            c.x as usize,
            c.y as usize,
            c.width as usize,
            c.height as usize,
        )));
    }
    out
}

/// Map the rotation code, rejecting anything not a quarter turn.
fn rotation_from(code: u32) -> Option<edgefirst_image::Rotation> {
    match code {
        0 => Some(edgefirst_image::Rotation::None),
        1 => Some(edgefirst_image::Rotation::Clockwise90),
        2 => Some(edgefirst_image::Rotation::Rotate180),
        3 => Some(edgefirst_image::Rotation::CounterClockwise90),
        _ => None,
    }
}

/// Map the flip code.
fn flip_from(code: u32) -> Option<edgefirst_image::Flip> {
    match code {
        0 => Some(edgefirst_image::Flip::None),
        1 => Some(edgefirst_image::Flip::Vertical),
        2 => Some(edgefirst_image::Flip::Horizontal),
        _ => None,
    }
}

/// Convert `src` into `dst`, scaling, converting colour and rotating as needed.
///
/// `src`/`dst` may have been minted by any EdgeFirst library -- every library
/// links the same shared tensor implementation, so both are read the same
/// way regardless of which one minted them. `crop` may be `NULL` for the
/// whole source.
///
/// @return 0 on success, otherwise an errno.
///
/// # Safety
/// `p`, `src` and `dst` must be live handles.
#[no_mangle]
pub unsafe extern "C" fn ef_image_processor_convert(
    p: *mut EfImageProcessor,
    src: *const EfTensor,
    dst: *mut EfTensor,
    rotation: u32,
    flip: u32,
    crop: *const EfCrop,
) -> c_int {
    unsafe {
        catch_unwind(AssertUnwindSafe(|| {
            if p.is_null() {
                return libc::EINVAL;
            }
            // Validated before any work: an unknown code silently treated as
            // "none" would rotate nothing and look like a backend failure.
            let (Some(rot), Some(fl)) = (rotation_from(rotation), flip_from(flip)) else {
                return libc::EINVAL;
            };
            use edgefirst_image::ImageProcessorTrait as _;
            let result = with_tensor(src, |s| {
                with_tensor_mut(dst, |d| (*p).inner.convert(s, d, rot, fl, crop_from(crop)))
            });
            match result {
                Ok(Ok(Ok(()))) => 0,
                Ok(Ok(Err(_))) => libc::EIO,
                Ok(Err(e)) | Err(e) => e,
            }
        }))
        .unwrap_or(libc::EINVAL)
    }
}

/// Like [`ef_image_processor_convert`], but does not wait for the GPU.
///
/// # Safety
/// `p`, `src` and `dst` must be live handles.
#[no_mangle]
pub unsafe extern "C" fn ef_image_processor_convert_deferred(
    p: *mut EfImageProcessor,
    src: *const EfTensor,
    dst: *mut EfTensor,
    rotation: u32,
    flip: u32,
    crop: *const EfCrop,
) -> c_int {
    unsafe {
        catch_unwind(AssertUnwindSafe(|| {
            if p.is_null() {
                return libc::EINVAL;
            }
            let (Some(rot), Some(fl)) = (rotation_from(rotation), flip_from(flip)) else {
                return libc::EINVAL;
            };
            use edgefirst_image::ImageProcessorTrait as _;
            let result = with_tensor(src, |s| {
                with_tensor_mut(dst, |d| {
                    (*p).inner.convert_deferred(s, d, rot, fl, crop_from(crop))
                })
            });
            match result {
                Ok(Ok(Ok(()))) => 0,
                Ok(Ok(Err(_))) => libc::EIO,
                Ok(Err(e)) | Err(e) => e,
            }
        }))
        .unwrap_or(libc::EINVAL)
    }
}

/// DMA-BUF row pitch alignment the GL backend requires, in bytes.
#[no_mangle]
pub extern "C" fn ef_gpu_dma_buf_pitch_alignment_bytes() -> usize {
    edgefirst_image::GPU_DMA_BUF_PITCH_ALIGNMENT_BYTES
}

/// Round `width` up so `width * bpp` meets the GPU pitch alignment.
#[no_mangle]
pub extern "C" fn ef_align_width_for_gpu_pitch(width: usize, bpp: usize) -> usize {
    edgefirst_image::align_width_for_gpu_pitch(width, bpp)
}

/// Align `width` for `format` (`"NV12"`, `"rgba8"`, …) and `dtype`.
///
/// # Safety
/// `format` must be a NUL-terminated string.
#[no_mangle]
pub unsafe extern "C" fn ef_align_width_for_pixel_format(
    width: usize,
    format: *const c_char,
    dtype: u32,
) -> usize {
    unsafe {
        catch_unwind(AssertUnwindSafe(|| {
            let Some(s) = cstr(format) else {
                return width;
            };
            let Some(fmt) = PixelFormat::from_str_code(s) else {
                return width;
            };
            let Some(dt) = DType::from_code(dtype) else {
                return width;
            };
            match edgefirst_image::primary_plane_bpp(fmt, dt.size()) {
                Some(bpp) => edgefirst_image::align_width_for_gpu_pitch(width, bpp),
                None => width,
            }
        }))
        .unwrap_or(width)
    }
}

/// Create a processor forced to one backend.
///
/// `backend`: 0 = auto, 1 = CPU, 2 = G2D, 3 = OpenGL. A forced backend
/// disables the fallback chain entirely — if it is unavailable the call fails
/// rather than quietly using another, which is the point of forcing one.
///
/// @return `NULL` when the backend is unknown or unavailable here.
#[no_mangle]
pub extern "C" fn ef_image_processor_new_with_backend(backend: u32) -> *mut EfImageProcessor {
    catch_unwind(|| {
        let b = match backend {
            0 => edgefirst_image::ComputeBackend::Auto,
            1 => edgefirst_image::ComputeBackend::Cpu,
            2 => edgefirst_image::ComputeBackend::G2d,
            3 => edgefirst_image::ComputeBackend::OpenGl,
            _ => return std::ptr::null_mut(),
        };
        // `needless_update` fires on macOS, where the config has only
        // `backend`; on Linux it also has a cfg-gated `egl_display`.
        #[allow(clippy::needless_update)]
        let config = edgefirst_image::ImageProcessorConfig {
            backend: b,
            ..Default::default()
        };
        match edgefirst_image::ImageProcessor::with_config(config) {
            Ok(inner) => Box::into_raw(Box::new(EfImageProcessor { inner })),
            Err(_) => std::ptr::null_mut(),
        }
    })
    .unwrap_or(std::ptr::null_mut())
}

/// Set the RGBA palette used when drawing class masks.
///
/// Copied, not borrowed, so the caller may free its array immediately.
///
/// @return 0 on success, `EINVAL` for a null argument or zero colours.
///
/// # Safety
/// `colors` must point to `count` RGBA quads.
#[no_mangle]
pub unsafe extern "C" fn ef_image_processor_set_class_colors(
    p: *mut EfImageProcessor,
    colors: *const [u8; 4],
    count: usize,
) -> c_int {
    unsafe {
        catch_unwind(AssertUnwindSafe(|| {
            if p.is_null() || colors.is_null() || count == 0 {
                return libc::EINVAL;
            }
            let slice = std::slice::from_raw_parts(colors, count);
            use edgefirst_image::ImageProcessorTrait as _;
            match (*p).inner.set_class_colors(slice) {
                Ok(()) => 0,
                Err(_) => libc::EINVAL,
            }
        }))
        .unwrap_or(libc::EINVAL)
    }
}

/// Flush any queued GPU work and wait for it.
///
/// @return 0 on success, otherwise an errno.
///
/// # Safety
/// `p` must be a live processor.
#[no_mangle]
pub unsafe extern "C" fn ef_image_processor_flush(p: *mut EfImageProcessor) -> c_int {
    unsafe {
        catch_unwind(AssertUnwindSafe(|| {
            if p.is_null() {
                return libc::EINVAL;
            }
            use edgefirst_image::ImageProcessorTrait as _;
            match (*p).inner.flush() {
                Ok(()) => 0,
                Err(_) => libc::EIO,
            }
        }))
        .unwrap_or(libc::EINVAL)
    }
}

/// Platforms: Linux, macOS, iOS, Android.
///
/// Convert, returning a sync-fence fd instead of blocking on the GPU.
///
/// The GL to NPU handoff. `*fence_fd` receives a descriptor the caller owns and
/// must close, or `-1` when the platform has no native fence and the convert
/// therefore completed synchronously — in which case the destination is already
/// safe to read.
///
/// @return 0 on success, `ENOTSUP` off Unix, otherwise an errno.
///
/// # Safety
/// `p`, `src`, `dst` must be live handles; `fence_fd` must be writable.
#[no_mangle]
pub unsafe extern "C" fn ef_image_processor_convert_fence(
    p: *mut EfImageProcessor,
    src: *const EfTensor,
    dst: *mut EfTensor,
    rotation: u32,
    flip: u32,
    crop: *const EfCrop,
    fence_fd: *mut c_int,
) -> c_int {
    unsafe {
        catch_unwind(AssertUnwindSafe(|| {
            if p.is_null() || fence_fd.is_null() {
                return libc::EINVAL;
            }
            // Written before any early return, so a caller never reads an
            // uninitialised descriptor after a failure -- a bad rotation or
            // flip code included, which is where this used to leave `*fence_fd`
            // untouched while the handle sibling below cleared it.
            *fence_fd = -1;
            let (Some(rot), Some(fl)) = (rotation_from(rotation), flip_from(flip)) else {
                return libc::EINVAL;
            };
            #[cfg(not(unix))]
            {
                let _ = (p, src, dst, rot, fl, crop);
                libc::ENOTSUP
            }
            #[cfg(unix)]
            {
                let result = with_tensor(src, |s| {
                    with_tensor_mut(dst, |d| {
                        (*p).inner
                            .convert_with_fence(s, d, rot, fl, crop_from(crop))
                    })
                });
                match result {
                    Ok(Ok(Ok(Some(owned)))) => {
                        use std::os::fd::IntoRawFd;
                        *fence_fd = owned.into_raw_fd();
                        0
                    }
                    // No native fence here: the convert already completed, which the
                    // -1 reports. Not an error.
                    Ok(Ok(Ok(None))) => 0,
                    Ok(Ok(Err(_))) => libc::EIO,
                    Ok(Err(e)) | Err(e) => e,
                }
            }
        }))
        .unwrap_or(libc::EINVAL)
    }
}

/// Platforms: Windows.
///
/// Convert, returning an event handle instead of blocking on the GPU. The
/// event is set when the destination is complete; the caller owns it and
/// closes it with `CloseHandle`. `*fence` is `NULL` when the convert
/// completed synchronously (no fence on this display).
///
/// @return 0 on success, `ENOTSUP` off Windows, otherwise an errno.
///
/// # Safety
/// `p`, `src`, `dst` must be live handles; `fence` must be writable.
#[no_mangle]
pub unsafe extern "C" fn ef_image_processor_convert_fence_handle(
    p: *mut EfImageProcessor,
    src: *const EfTensor,
    dst: *mut EfTensor,
    rotation: u32,
    flip: u32,
    crop: *const EfCrop,
    fence: *mut *mut c_void,
) -> c_int {
    unsafe {
        catch_unwind(AssertUnwindSafe(|| {
            if p.is_null() || fence.is_null() {
                return libc::EINVAL;
            }
            // Written before the rotation/flip check below, as the fd
            // sibling above does, so a caller reading `*fence` after any
            // failure -- a bad code included -- never sees an uninitialised
            // pointer.
            *fence = std::ptr::null_mut();
            let (Some(rot), Some(fl)) = (rotation_from(rotation), flip_from(flip)) else {
                return libc::EINVAL;
            };
            #[cfg(not(target_os = "windows"))]
            {
                let _ = (src, dst, rot, fl, crop);
                libc::ENOTSUP
            }
            #[cfg(target_os = "windows")]
            {
                use std::os::windows::io::IntoRawHandle;
                let result = with_tensor(src, |s| {
                    with_tensor_mut(dst, |d| {
                        (*p).inner
                            .convert_with_fence(s, d, rot, fl, crop_from(crop))
                    })
                });
                match result {
                    Ok(Ok(Ok(Some(owned)))) => {
                        *fence = owned.into_raw_handle();
                        0
                    }
                    // No fence on this display: the convert already
                    // completed, which the NULL `*fence` reports. Not an
                    // error.
                    Ok(Ok(Ok(None))) => 0,
                    Ok(Ok(Err(_))) => libc::EIO,
                    Ok(Err(e)) | Err(e) => e,
                }
            }
        }))
        .unwrap_or(libc::EINVAL)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn processor() -> *mut EfImageProcessor {
        let p = ef_image_processor_new();
        assert!(!p.is_null(), "the platform must provide some converter");
        p
    }

    #[test]
    fn convert_accepts_two_tensors_this_library_minted() {
        let p = processor();
        let rgb = std::ffi::CString::new("rgb8").unwrap();
        let nv12 = std::ffi::CString::new("NV12").unwrap();
        unsafe {
            let src = ef_image_processor_create_image(p, 64, 48, nv12.as_ptr(), 0, 0, 3);
            let dst = ef_image_processor_create_image(p, 64, 48, rgb.as_ptr(), 0, 0, 3);
            assert!(!src.is_null() && !dst.is_null());
            let rc = ef_image_processor_convert(p, src, dst, 0, 0, std::ptr::null());
            assert_eq!(rc, 0, "a same-library convert must succeed");
            edgefirst_tensor_ffi::ef_tensor_free(src);
            edgefirst_tensor_ffi::ef_tensor_free(dst);
            ef_image_processor_free(p);
        }
    }

    #[test]
    fn an_unknown_backend_is_refused_rather_than_falling_back() {
        // Forcing a backend exists to disable the fallback chain; quietly
        // substituting another would defeat the only reason to call this.
        assert!(ef_image_processor_new_with_backend(99).is_null());
        // Auto and CPU are available on every host this builds for.
        let p = ef_image_processor_new_with_backend(0);
        assert!(!p.is_null());
        unsafe { ef_image_processor_free(p) };
    }

    #[test]
    fn class_colors_are_copied_not_borrowed() {
        let p = processor();
        unsafe {
            {
                // Heap-allocated and scoped on purpose: it is freed before
                // the palette is used, so a borrow rather than a copy would
                // read freed memory. A stack array would not prove that -- the
                // bytes would still be readable after it went out of scope.
                let colors = [[255u8, 0, 0, 255], [0, 255, 0, 255]].to_vec();
                assert_eq!(
                    ef_image_processor_set_class_colors(p, colors.as_ptr(), colors.len()),
                    0
                );
            }
            assert_eq!(
                ef_image_processor_set_class_colors(p, std::ptr::null(), 2),
                libc::EINVAL
            );
            let one = [[1u8, 2, 3, 4]];
            assert_eq!(
                ef_image_processor_set_class_colors(p, one.as_ptr(), 0),
                libc::EINVAL,
                "an empty palette is a caller bug, not a reset"
            );
            ef_image_processor_free(p);
        }
    }

    #[test]
    fn an_unknown_rotation_or_flip_code_is_refused() {
        // Treating an unknown code as "none" would rotate nothing and read as
        // a backend failure rather than a caller mistake.
        let p = processor();
        let nv12 = std::ffi::CString::new("NV12").unwrap();
        unsafe {
            let src = ef_image_processor_create_image(p, 64, 48, nv12.as_ptr(), 0, 0, 3);
            let dst = ef_image_processor_create_image(p, 64, 48, nv12.as_ptr(), 0, 0, 3);
            assert_eq!(
                ef_image_processor_convert(p, src, dst, 99, 0, std::ptr::null()),
                libc::EINVAL
            );
            assert_eq!(
                ef_image_processor_convert(p, src, dst, 0, 99, std::ptr::null()),
                libc::EINVAL
            );
            edgefirst_tensor_ffi::ef_tensor_free(src);
            edgefirst_tensor_ffi::ef_tensor_free(dst);
            ef_image_processor_free(p);
        }
    }

    #[test]
    fn a_zeroed_crop_means_the_whole_source() {
        // A caller who memsets the struct must get what NULL gives, or
        // zero-initialisation becomes a trap.
        let whole = format!("{:?}", crop_from(std::ptr::null()));
        let zeroed = format!("{:?}", crop_from(&EfCrop::default()));
        assert_eq!(whole, zeroed, "a zeroed crop must equal no crop");
    }

    #[test]
    fn every_rotation_and_flip_code_maps_and_the_rest_do_not() {
        for (c, r) in [
            (0, edgefirst_image::Rotation::None),
            (1, edgefirst_image::Rotation::Clockwise90),
            (2, edgefirst_image::Rotation::Rotate180),
            (3, edgefirst_image::Rotation::CounterClockwise90),
        ] {
            assert_eq!(rotation_from(c), Some(r), "rotation code {c}");
        }
        assert_eq!(rotation_from(4), None);
        for (c, f) in [
            (0, edgefirst_image::Flip::None),
            (1, edgefirst_image::Flip::Vertical),
            (2, edgefirst_image::Flip::Horizontal),
        ] {
            assert_eq!(flip_from(c), Some(f), "flip code {c}");
        }
        assert_eq!(flip_from(3), None);
    }

    #[test]
    fn a_processor_can_be_created_and_freed() {
        let p = processor();
        unsafe { ef_image_processor_free(p) };
        unsafe { ef_image_processor_free(std::ptr::null_mut()) };
    }

    #[test]
    fn create_image_mints_a_real_usable_tensor() {
        // A tensor from here is an ordinary `ef_tensor`, minted inside
        // `libedgefirst_tensor.so` (this library's own allocation strategy
        // chose PBO/mem/etc, but the implementation is the shared one) --
        // read with the real exported accessors, freed with the real
        // exported `ef_tensor_free`, exactly like one from `ef_tensor_new`.
        let p = processor();
        let nv12 = std::ffi::CString::new("NV12").unwrap();
        let t = unsafe { ef_image_processor_create_image(p, 64, 48, nv12.as_ptr(), 0, 0, 3) };
        assert!(!t.is_null());
        let fmt = unsafe {
            std::ffi::CStr::from_ptr(edgefirst_tensor_ffi::ef_tensor_format(t))
                .to_str()
                .unwrap()
                .to_string()
        };
        assert_eq!(PixelFormat::from_str_code(&fmt), Some(PixelFormat::Nv12));
        unsafe { edgefirst_tensor_ffi::ef_tensor_free(t) };
        unsafe { ef_image_processor_free(p) };
    }

    #[test]
    fn a_null_processor_or_bad_argument_is_an_error_not_a_crash() {
        let nv12 = std::ffi::CString::new("NV12").unwrap();
        unsafe {
            assert!(ef_image_processor_create_image(
                std::ptr::null_mut(),
                64,
                48,
                nv12.as_ptr(),
                0,
                0,
                3
            )
            .is_null());
            let p = processor();
            // Zero dimensions, unknown format, unknown dtype, unknown access.
            assert!(ef_image_processor_create_image(p, 0, 48, nv12.as_ptr(), 0, 0, 3).is_null());
            let junk = std::ffi::CString::new("not-a-format").unwrap();
            assert!(ef_image_processor_create_image(p, 64, 48, junk.as_ptr(), 0, 0, 3).is_null());
            assert!(
                ef_image_processor_create_image(p, 64, 48, std::ptr::null(), 0, 0, 3).is_null()
            );
            assert!(
                ef_image_processor_create_image(p, 64, 48, nv12.as_ptr(), 9999, 0, 3).is_null()
            );
            assert!(
                ef_image_processor_create_image(p, 64, 48, nv12.as_ptr(), 0, 0, 9999).is_null()
            );
            ef_image_processor_free(p);
        }
    }

    #[test]
    fn the_cpu_access_codes_match_the_rust_enum() {
        // The one vocabulary here without a shared `code()`, so it is the one
        // that can silently drift. Ordering is the enum's declaration order.
        assert_eq!(cpu_access_from_code(0), Some(CpuAccess::None));
        assert_eq!(cpu_access_from_code(1), Some(CpuAccess::Read));
        assert_eq!(cpu_access_from_code(2), Some(CpuAccess::Write));
        assert_eq!(cpu_access_from_code(3), Some(CpuAccess::ReadWrite));
        assert_eq!(cpu_access_from_code(4), None);
    }

    /// A view with every field away from its default -- 64x48 NV12 U8,
    /// `Shm`, `ReadWrite`, `Any` compression.
    fn full_view() -> EfImageDescView {
        EfImageDescView {
            width: 64,
            height: 48,
            format: PixelFormat::Nv12.code(),
            dtype: DType::U8.code(),
            access: 3, // ReadWrite
            memory: 1, // Shm
            has_memory: 1,
            compression: 1, // Any
            has_compression: 1,
        }
    }

    #[test]
    fn a_view_built_by_hand_reconstructs_correctly() {
        // `image_desc_from_view` is the one place this crate turns a scalar
        // view back into an `ImageDesc`, paired with tensor-capi's
        // `desc::view_of` for the forward direction (see that crate's
        // `the_view_round_trips_every_field`, which builds a view through
        // the real setters and checks the same fields the other way).
        // Cross-library linking to run both directions through the real
        // FFI is `an_image_desc_request_crosses_the_library_boundary_intact`
        // in `lib.rs`; this is the reconstruction logic in isolation.
        let d = image_desc_from_view(&full_view()).expect("a full view must decode");
        assert_eq!(d.width(), 64);
        assert_eq!(d.height(), 48);
        assert_eq!(d.format(), PixelFormat::Nv12);
        assert_eq!(d.dtype(), DType::U8);
        assert_eq!(d.access(), CpuAccess::ReadWrite);
        assert_eq!(d.memory(), Some(TensorMemory::Shm));
        assert_eq!(d.compression(), Some(Compression::Any));
    }

    #[test]
    fn an_empty_view_decodes_to_no_memory_or_compression_request() {
        let mut v = full_view();
        v.has_memory = 0;
        v.has_compression = 0;
        let d = image_desc_from_view(&v).expect("must still decode");
        assert_eq!(d.memory(), None);
        assert_eq!(d.compression(), None);
    }

    #[test]
    fn an_unrecognized_code_in_any_field_is_refused_not_guessed_at() {
        // A stale wire vocabulary between two independently-versioned leaves
        // (see `lib.rs`) must be a declined request, never a panic or a
        // silent default.
        let mut v = full_view();
        v.format = 9999;
        assert!(image_desc_from_view(&v).is_none(), "bad format");

        let mut v = full_view();
        v.dtype = 9999;
        assert!(image_desc_from_view(&v).is_none(), "bad dtype");

        let mut v = full_view();
        v.access = 9999;
        assert!(image_desc_from_view(&v).is_none(), "bad access");

        let mut v = full_view();
        v.memory = 9999;
        assert!(image_desc_from_view(&v).is_none(), "bad memory code");

        let mut v = full_view();
        v.compression = 9999;
        assert!(image_desc_from_view(&v).is_none(), "bad compression code");
    }

    #[test]
    fn a_specific_compression_scheme_is_refused_here_too() {
        // Code 2 ("a specific vendor scheme") is a real state
        // `ef_tensor_image_desc_get` can report, but no C setter can ever
        // request one, so there is nothing to reconstruct -- refused, same
        // as an unrecognized code, not silently treated as `Any`.
        let mut v = full_view();
        v.compression = 2;
        assert!(image_desc_from_view(&v).is_none());
    }

    /// `p`/`fence` are checked before any platform arm runs, so a `NULL`
    /// here is `EINVAL` on every platform this family is declared on --
    /// unlike a bad `src`/`dst`, whose errno differs by platform (see the
    /// Windows-only round trip below).
    #[test]
    fn convert_fence_handle_refuses_a_null_processor_or_fence() {
        let mut fence: *mut c_void = std::ptr::null_mut();
        unsafe {
            assert_eq!(
                ef_image_processor_convert_fence_handle(
                    std::ptr::null_mut(),
                    std::ptr::null(),
                    std::ptr::null_mut(),
                    0,
                    0,
                    std::ptr::null(),
                    &mut fence,
                ),
                libc::EINVAL
            );
            let p = processor();
            assert_eq!(
                ef_image_processor_convert_fence_handle(
                    p,
                    std::ptr::null(),
                    std::ptr::null_mut(),
                    0,
                    0,
                    std::ptr::null(),
                    std::ptr::null_mut(),
                ),
                libc::EINVAL,
                "a null fence out-param must be refused before it is written"
            );
            ef_image_processor_free(p);
        }
    }

    /// The Windows path end to end, on the real GPU: mint a texture-backed
    /// src/dst through this same library, convert without blocking, and wait
    /// on the returned event.
    ///
    /// The C leaf harness does not execute C tests on Windows (POSIX `cc`
    /// and a `.a` -- see `lib.rs`'s `cc_build_and_run`), so this is the one
    /// place in this crate that actually drives
    /// `ef_image_processor_convert_fence_handle` on real hardware, the same
    /// role `a_texture_tensor_round_trips_through_the_exports` plays for the
    /// tensor leaf's D3D11 family (`tensor-capi/src/d3d11.rs`).
    #[cfg(target_os = "windows")]
    #[test]
    fn convert_fence_handle_returns_an_owned_event_set_on_completion() {
        if !edgefirst_tensor::is_gpu_buffer_available() {
            eprintln!("SKIP: no D3D11 device on this host");
            return;
        }
        let p = processor();
        let rgba = std::ffi::CString::new("rgba8").unwrap();
        let rgb = std::ffi::CString::new("rgb8").unwrap();
        // `storage` has no dedicated "auto" enumerator (see
        // `TensorMemory::from_code`): any code outside the 0-5 vocabulary
        // maps to `None`, which is what lets `create_image` reach its
        // Dma-first arm and hand back a D3D11 texture. Code 0 is `Mem` --
        // a real, valid request, not "unspecified" -- so it would allocate
        // a host tensor here and make this test vacuous.
        const AUTO: u32 = u32::MAX;
        unsafe {
            let src = ef_image_processor_create_image(p, 64, 48, rgba.as_ptr(), 0, AUTO, 3);
            let dst = ef_image_processor_create_image(p, 64, 48, rgb.as_ptr(), 0, AUTO, 3);
            assert!(!src.is_null() && !dst.is_null());

            let mut fence: *mut c_void = std::ptr::dangling_mut();
            let rc = ef_image_processor_convert_fence_handle(
                p,
                src,
                dst,
                0,
                0,
                std::ptr::null(),
                &mut fence,
            );
            assert_eq!(rc, 0, "a same-library fenced convert must succeed");
            assert!(
                !fence.is_null(),
                "a texture destination on Windows must hand back a fence"
            );
            assert_eq!(
                wait_for_single_object(fence, 5000),
                0,
                "fence was not set within 5 seconds"
            );
            close_handle(fence);

            // The event and the destination's own completion name one value:
            // a fenced convert into a texture flushes and signals once, and
            // records that value on `dst`. A consumer may wait on either.
            let mut dvalue: u64 = 0;
            let mut dfence: *mut c_void = std::ptr::null_mut();
            assert_eq!(
                edgefirst_tensor_ffi::ef_tensor_gpu_completion(dst, &mut dfence, &mut dvalue),
                0
            );
            assert_ne!(dvalue, 0, "the fenced convert recorded a completion");
            if !dfence.is_null() {
                close_handle(dfence);
            }

            edgefirst_tensor_ffi::ef_tensor_free(src);
            edgefirst_tensor_ffi::ef_tensor_free(dst);
            ef_image_processor_free(p);
        }
    }

    /// `WaitForSingleObject`/`CloseHandle` without pulling the `windows`
    /// crate into this leaf: the handle crossing this ABI is a plain
    /// `void *`, and the test needs exactly two Win32 calls to wait on and
    /// give back the owned one. Mirrors `tensor-capi/src/d3d11.rs`'s
    /// `close_handle` test helper.
    #[cfg(target_os = "windows")]
    fn wait_for_single_object(h: *mut c_void, millis: u32) -> u32 {
        extern "system" {
            fn WaitForSingleObject(h: *mut c_void, millis: u32) -> u32;
        }
        // SAFETY: `h` is a live NT event handle this test received from an
        // export documented to transfer ownership.
        unsafe { WaitForSingleObject(h, millis) }
    }

    #[cfg(target_os = "windows")]
    fn close_handle(h: *mut c_void) {
        extern "system" {
            fn CloseHandle(h: *mut c_void) -> i32;
        }
        // SAFETY: `h` is an owned NT handle this test received from an
        // export documented to transfer ownership.
        assert_ne!(unsafe { CloseHandle(h) }, 0);
    }
}
