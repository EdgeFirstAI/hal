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

use edgefirst_decoder_abi::EfPreprocessSpec;
use edgefirst_image::{
    ConvertOptions, Crop, Fit, FitMode, Flip, ImageProcessor, Interpolation, LetterboxPlacement,
    Normalization, NormalizationPreset, PadSpace, PreprocessSpec, Rotation,
};
use edgefirst_tensor::{
    Compression, CpuAccess, DType, ImageDesc, PixelFormat, TensorDyn, TensorMemory,
};
use edgefirst_tensor_ffi::EfTensor;
use edgefirst_tensor_ffi::{
    ef_tensor_image_desc_contiguous, ef_tensor_image_desc_get, EfCompressionRequest,
    EfImageDescView, EfTensorImageDesc,
};

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
/// The returned tensor's `ef_tensor_cpu_access` may be wider than `access`:
/// what this processor's own convert may do to it as a destination on the
/// CPU is added, `EF_CPU_ACCESS_WRITE` when it may fill the whole image and
/// `EF_CPU_ACCESS_READ_WRITE` for passes that read back or write part of it
/// (`EF_DTYPE_I8`, a CPU `bgra8` swizzle, a G2D `rgb8` letterbox border). A
/// PBO image is always `EF_CPU_ACCESS_READ_WRITE`. Nothing is added for use
/// as a convert source or for a CPU mask draw; declare those yourself.
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
        // `Any` is the only compression request any `ef_tensor_image_desc_set_*`
        // entry point can create; `Scheme` ("a specific vendor scheme") has no
        // decodable detail behind it here, so it is refused rather than
        // silently downgraded to `Any` or dropped.
        if v.compression != EfCompressionRequest::Any as u32 {
            return None;
        }
        desc = desc.with_compression(Compression::Any);
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
/// The declared access is widened as for `ef_image_processor_create_image`,
/// except for a compression request, which keeps `EF_CPU_ACCESS_NONE`.
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
            let Some(mut desc) = image_desc_from_view(&view) else {
                return std::ptr::null_mut();
            };
            // Not part of the fixed-size view; read through its own getter.
            if ef_tensor_image_desc_contiguous(d) == 1 {
                desc = desc.with_contiguous(true);
            }
            match (*p).inner.create_image_desc(&desc) {
                Ok(t) => t.into_raw(),
                Err(_) => std::ptr::null_mut(),
            }
        }))
        .unwrap_or(std::ptr::null_mut())
    }
}

/// Map a `CpuAccess` code, `None` (0) included.
pub(crate) fn cpu_access_from_code(code: u32) -> Option<CpuAccess> {
    CpuAccess::from_code(code)
}

/// Borrow the `TensorDyn` behind a handle without taking ownership of it.
///
/// There is now exactly one implementation of `ef_tensor` (all four sibling
/// `-capi` leaves link this crate's shared `libedgefirst_tensor.so` rather
/// than embedding their own copy), so every handle is read the same way
/// regardless of which library minted it -- no per-library dispatch table,
/// no re-import from a foreign platform handle. `TensorDyn::with_raw` borrows
/// the handle without freeing it, so the caller still owns its lifetime.
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

/// Options for a convert, built with the `ef_convert_options_*` setters.
/// `NULL` anywhere one is accepted means the defaults: the whole source,
/// stretched, no rotation or flip, and float outputs mapped as `x / 255`.
pub struct EfConvertOptions {
    inner: ConvertOptions,
}

/// The options to use for a convert: `o`'s, or the defaults for `NULL`.
///
/// # Safety
/// `o` must be `NULL` or a live handle.
pub(crate) unsafe fn options_from(o: *const EfConvertOptions) -> Option<ConvertOptions> {
    // SAFETY: the caller's contract.
    unsafe { o.as_ref() }.map(|o| o.inner)
}

/// Apply `f` to a live options handle; `EINVAL` for `NULL`.
///
/// # Safety
/// `o` must be `NULL` or a live handle.
unsafe fn update(
    o: *mut EfConvertOptions,
    f: impl FnOnce(ConvertOptions) -> Option<ConvertOptions>,
) -> c_int {
    catch_unwind(AssertUnwindSafe(|| {
        // SAFETY: the caller's contract.
        let Some(o) = (unsafe { o.as_mut() }) else {
            return libc::EINVAL;
        };
        match f(o.inner) {
            Some(next) => {
                o.inner = next;
                0
            }
            None => libc::EINVAL,
        }
    }))
    .unwrap_or(libc::EINVAL)
}

/// New default convert options. `NULL` on allocation failure. Free with
/// [`ef_convert_options_free`].
#[no_mangle]
pub extern "C" fn ef_convert_options_new() -> *mut EfConvertOptions {
    catch_unwind(|| {
        Box::into_raw(Box::new(EfConvertOptions {
            inner: ConvertOptions::new(),
        }))
    })
    .unwrap_or(std::ptr::null_mut())
}

/// Free options from [`ef_convert_options_new`]. `NULL` is a no-op.
///
/// # Safety
/// `o` must be `NULL` or a live handle, not used afterwards.
#[no_mangle]
pub unsafe extern "C" fn ef_convert_options_free(o: *mut EfConvertOptions) {
    if !o.is_null() {
        // SAFETY: the caller's contract; the handle came from `Box::into_raw`.
        drop(unsafe { Box::from_raw(o) });
    }
}

/// Rotate the output: an `EF_ROTATION_*` code.
///
/// @return 0, or `EINVAL` for an unknown code or `NULL`.
///
/// # Safety
/// `o` must be `NULL` or a live handle.
#[no_mangle]
pub unsafe extern "C" fn ef_convert_options_set_rotation(
    o: *mut EfConvertOptions,
    rotation: u32,
) -> c_int {
    unsafe { update(o, |c| Some(c.rotation(Rotation::from_code(rotation)?))) }
}

/// Flip the output: an `EF_FLIP_*` code.
///
/// @return 0, or `EINVAL` for an unknown code or `NULL`.
///
/// # Safety
/// `o` must be `NULL` or a live handle.
#[no_mangle]
pub unsafe extern "C" fn ef_convert_options_set_flip(o: *mut EfConvertOptions, flip: u32) -> c_int {
    unsafe { update(o, |c| Some(c.flip(Flip::from_code(flip)?))) }
}

/// Sample only this rectangle of the source, in pixels. A zero `width` or
/// `height` samples the whole source.
///
/// @return 0, or `EINVAL` for `NULL`.
///
/// # Safety
/// `o` must be `NULL` or a live handle.
#[no_mangle]
pub unsafe extern "C" fn ef_convert_options_set_source(
    o: *mut EfConvertOptions,
    x: u32,
    y: u32,
    width: u32,
    height: u32,
) -> c_int {
    let region = (width != 0 && height != 0).then(|| {
        edgefirst_tensor::Region::new(x as usize, y as usize, width as usize, height as usize)
    });
    unsafe { update(o, |c| Some(c.source(region))) }
}

/// Letterbox: keep the source aspect ratio and pad the rest with the RGBA
/// pixel `pad`, mapped like the image. `NULL` stretches instead.
///
/// @return 0, or `EINVAL` for a `NULL` handle.
///
/// # Safety
/// `o` must be `NULL` or a live handle; `pad` `NULL` or four bytes.
#[no_mangle]
pub unsafe extern "C" fn ef_convert_options_set_letterbox(
    o: *mut EfConvertOptions,
    pad: *const u8,
) -> c_int {
    // SAFETY: the caller's contract.
    let pad = unsafe { pad.cast::<[u8; 4]>().as_ref() }.copied();
    unsafe {
        update(o, |c| {
            Some(match pad {
                Some(pad) => c.letterbox(pad),
                None => c.crop(Crop {
                    fit: Fit::Stretch,
                    ..c.get_crop()
                }),
            })
        })
    }
}

/// Where a letterboxed image sits: an `EF_LETTERBOX_PLACEMENT_*` code.
///
/// @return 0, or `EINVAL` for an unknown code or `NULL`.
///
/// # Safety
/// `o` must be `NULL` or a live handle.
#[no_mangle]
pub unsafe extern "C" fn ef_convert_options_set_letterbox_placement(
    o: *mut EfConvertOptions,
    placement: u32,
) -> c_int {
    unsafe {
        update(o, |c| {
            Some(c.placement(LetterboxPlacement::from_code(placement)?))
        })
    }
}

/// Pad a letterbox with these four values as written to the destination
/// (after normalization; quantized for integer destinations) instead of
/// mapping the pixel pad.
///
/// @return 0, or `EINVAL` for `NULL`.
///
/// # Safety
/// `o` must be `NULL` or a live handle; `pad` must be four floats.
#[no_mangle]
pub unsafe extern "C" fn ef_convert_options_set_pad_output(
    o: *mut EfConvertOptions,
    pad: *const f32,
) -> c_int {
    // SAFETY: the caller's contract.
    let Some(pad) = (unsafe { pad.cast::<[f32; 4]>().as_ref() }).copied() else {
        return libc::EINVAL;
    };
    unsafe { update(o, |c| Some(c.pad_output(pad))) }
}

/// Normalize with a preset: an `EF_NORMALIZATION_*` code.
///
/// @return 0, or `EINVAL` for an unknown code or `NULL`.
///
/// # Safety
/// `o` must be `NULL` or a live handle.
#[no_mangle]
pub unsafe extern "C" fn ef_convert_options_set_normalization_preset(
    o: *mut EfConvertOptions,
    preset: u32,
) -> c_int {
    unsafe {
        update(o, |c| {
            Some(c.normalization(NormalizationPreset::from_code(preset)?))
        })
    }
}

/// Normalize as `(x - mean) / std` on 0-255 values, `n` values each: 1 for
/// every channel or 3 for RGB.
///
/// @return 0, or `EINVAL` for `NULL`, another `n`, a non-finite mean or a
/// zero or non-finite std.
///
/// # Safety
/// `o` must be `NULL` or a live handle; `mean` and `std` must hold `n` floats.
#[no_mangle]
pub unsafe extern "C" fn ef_convert_options_set_normalization(
    o: *mut EfConvertOptions,
    mean: *const f32,
    std: *const f32,
    n: usize,
) -> c_int {
    if mean.is_null() || std.is_null() {
        return libc::EINVAL;
    }
    // SAFETY: the caller's contract; both checked non-null.
    let (mean, std) = unsafe {
        (
            std::slice::from_raw_parts(mean, n),
            std::slice::from_raw_parts(std, n),
        )
    };
    let Ok(norm) = Normalization::mean_std(mean, std) else {
        return libc::EINVAL;
    };
    unsafe { update(o, |c| Some(c.normalization(norm))) }
}

/// Resample with this interpolation when a convert with `o` resizes, instead
/// of the processor's (see [`ef_image_processor_set_interpolation`]): an
/// `EF_INTERPOLATION_*` code. `EF_INTERPOLATION_BICUBIC` and
/// `EF_INTERPOLATION_AREA` run as bilinear, logged once per mode. The
/// `EDGEFIRST_INTERPOLATION` environment variable, when set, overrides it.
///
/// @return 0, or `EINVAL` for an unknown code or `NULL`.
///
/// # Safety
/// `o` must be `NULL` or a live handle.
#[no_mangle]
pub unsafe extern "C" fn ef_convert_options_set_interpolation(
    o: *mut EfConvertOptions,
    mode: u32,
) -> c_int {
    unsafe {
        update(o, |c| {
            Some(c.interpolation(Interpolation::from_code(mode)?))
        })
    }
}

/// Read a caller's `ef_preprocess_spec`.
///
/// # Safety
/// `spec` must be `NULL` or valid for reads.
pub(crate) unsafe fn spec_from_c(spec: *const EfPreprocessSpec) -> Result<PreprocessSpec, c_int> {
    // SAFETY: the caller's contract.
    let c = unsafe { spec.as_ref() }.ok_or(libc::EINVAL)?;
    let mut out = PreprocessSpec::new();
    out.fit = FitMode::from_code(c.fit).ok_or(libc::EINVAL)?;
    out.placement = LetterboxPlacement::from_code(c.placement).ok_or(libc::EINVAL)?;
    out.pad_space = PadSpace::from_code(c.pad_space).ok_or(libc::EINVAL)?;
    out.interpolation = Interpolation::from_code(c.interpolation).ok_or(libc::EINVAL)?;
    out.normalization = Normalization::mean_std(&c.mean, &c.std).map_err(|_| libc::EINVAL)?;
    out.pad = c.pad;
    out.size = (c.width != 0 && c.height != 0).then_some((c.width as usize, c.height as usize));
    Ok(out)
}

/// Options that apply a model's preprocessing: its normalization, fit,
/// letterbox placement, pad and interpolation (bicubic and area run as
/// bilinear). `spec` usually comes from the decoder library's
/// `ef_model_config_preprocessing`.
///
/// @return options to free with [`ef_convert_options_free`], or `NULL` with
/// `errno` `EINVAL` for a `NULL` spec, an unknown code, a zero or non-finite
/// `std`, or a pad outside 0-255 in pixel space.
///
/// # Safety
/// `spec` must be `NULL` or valid for reads.
#[no_mangle]
pub unsafe extern "C" fn ef_convert_options_from_spec(
    spec: *const EfPreprocessSpec,
) -> *mut EfConvertOptions {
    let fail = |code: c_int| {
        errno::set_errno(errno::Errno(code));
        std::ptr::null_mut()
    };
    catch_unwind(AssertUnwindSafe(|| {
        // SAFETY: the caller's contract.
        let spec = match unsafe { spec_from_c(spec) } {
            Ok(s) => s,
            Err(code) => return fail(code),
        };
        match ConvertOptions::from_spec(&spec) {
            Ok(inner) => Box::into_raw(Box::new(EfConvertOptions { inner })),
            Err(e) => fail(image_errno(&e)),
        }
    }))
    .unwrap_or_else(|_| fail(libc::EINVAL))
}

/// Where a convert with `o` puts the image content in a `dst_w`x`dst_h`
/// destination for a `src_w`x`src_h` source: the whole destination for a
/// stretch, the letterbox rectangle otherwise, the same pixels the convert
/// paints. Pass `normalized_out` to `ef_semantic_options_set_content_region`
/// or the draw functions' `content_region`.
///
/// `pixels_out` (`NULL` to skip) receives `[x, y, width, height]` in
/// destination pixels; `normalized_out` (`NULL` to skip) receives
/// `[xmin, ymin, xmax, ymax]` normalized by the destination size.
///
/// @return 0, or `EINVAL` for a `NULL` handle, a zero size, or a source
/// crop outside the source.
///
/// # Safety
/// `o` must be `NULL` or live; `pixels_out` and `normalized_out` `NULL` or
/// four writable elements.
#[no_mangle]
pub unsafe extern "C" fn ef_convert_options_content_region(
    o: *const EfConvertOptions,
    src_w: usize,
    src_h: usize,
    dst_w: usize,
    dst_h: usize,
    pixels_out: *mut u32,
    normalized_out: *mut f32,
) -> c_int {
    catch_unwind(AssertUnwindSafe(|| {
        // SAFETY: the caller's contract.
        let Some(o) = (unsafe { o.as_ref() }) else {
            return libc::EINVAL;
        };
        if src_w == 0 || src_h == 0 || dst_w == 0 || dst_h == 0 {
            return libc::EINVAL;
        }
        let region = match o.inner.content_region(src_w, src_h, dst_w, dst_h) {
            Ok(r) => r,
            Err(e) => return image_errno(&e),
        };
        let r = region.pixels;
        let pixels = [r.x, r.y, r.width, r.height].map(|v| u32::try_from(v).unwrap_or(u32::MAX));
        // SAFETY: the caller's contract: four writable elements when non-null.
        unsafe {
            if let Some(px) = pixels_out.cast::<[u32; 4]>().as_mut() {
                *px = pixels;
            }
            if let Some(n) = normalized_out.cast::<[f32; 4]>().as_mut() {
                *n = region.normalized;
            }
        }
        0
    }))
    .unwrap_or(libc::EINVAL)
}

/// The errno for a failed image operation: `EACCES` for a tensor whose
/// declared CPU access does not cover the work, `EINVAL` for arguments that
/// cannot apply, `ENOTSUP` for work no backend supports, `EIO` otherwise.
pub(crate) fn image_errno(e: &edgefirst_image::Error) -> c_int {
    use edgefirst_image::Error as E;
    if e.cpu_access_denied().is_some() {
        return libc::EACCES;
    }
    match e {
        E::InvalidOptions(_)
        | E::CropInvalid(_)
        | E::InvalidShape(_)
        | E::NotAnImage
        | E::AliasedBuffers(_) => libc::EINVAL,
        E::UnsupportedFormat(_) | E::NotSupported(_) | E::NotImplemented(_) | E::NoConverter => {
            libc::ENOTSUP
        }
        E::Io(io) => io.raw_os_error().unwrap_or(libc::EIO),
        _ => libc::EIO,
    }
}

/// Convert `src` into `dst`, scaling, converting colour and rotating as needed.
///
/// `src`/`dst` may have been minted by any EdgeFirst library -- every library
/// links the same shared tensor implementation, so both are read the same
/// way regardless of which one minted them. `options` may be `NULL` for the
/// defaults.
///
/// @return 0 on success; `EINVAL` for options that cannot apply to `dst`
/// (such as a normalization on an integer destination without quantization),
/// an invalid crop or shape, or a tensor with no pixel format; `ENOTSUP`
/// when no backend supports the conversion; `EACCES` when a tensor's
/// declared `ef_cpu_access` does not cover the CPU pass, including when
/// GL/G2D declined the convert and the CPU fallback was refused; otherwise
/// `EIO` or the backend's errno.
///
/// # Safety
/// `p`, `src` and `dst` must be live handles.
#[no_mangle]
pub unsafe extern "C" fn ef_image_processor_convert(
    p: *mut EfImageProcessor,
    src: *const EfTensor,
    dst: *mut EfTensor,
    options: *const EfConvertOptions,
) -> c_int {
    unsafe {
        catch_unwind(AssertUnwindSafe(|| {
            if p.is_null() {
                return libc::EINVAL;
            }
            use edgefirst_image::ImageProcessorTrait as _;
            let result = with_tensor(src, |s| {
                with_tensor_mut(dst, |d| (*p).inner.convert(s, d, options_from(options)))
            });
            match result {
                Ok(Ok(Ok(()))) => 0,
                Ok(Ok(Err(e))) => image_errno(&e),
                Ok(Err(e)) | Err(e) => e,
            }
        }))
        .unwrap_or(libc::EINVAL)
    }
}

/// Like [`ef_image_processor_convert`], but does not wait for the GPU.
///
/// @return the codes [`ef_image_processor_convert`] returns.
///
/// # Safety
/// `p`, `src` and `dst` must be live handles.
#[no_mangle]
pub unsafe extern "C" fn ef_image_processor_convert_deferred(
    p: *mut EfImageProcessor,
    src: *const EfTensor,
    dst: *mut EfTensor,
    options: *const EfConvertOptions,
) -> c_int {
    unsafe {
        catch_unwind(AssertUnwindSafe(|| {
            if p.is_null() {
                return libc::EINVAL;
            }
            use edgefirst_image::ImageProcessorTrait as _;
            let result = with_tensor(src, |s| {
                with_tensor_mut(dst, |d| {
                    (*p).inner.convert_deferred(s, d, options_from(options))
                })
            });
            match result {
                Ok(Ok(Ok(()))) => 0,
                Ok(Ok(Err(e))) => image_errno(&e),
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
/// `backend` is an `EF_COMPUTE_BACKEND_*` code. A forced backend
/// disables the fallback chain entirely — if it is unavailable the call fails
/// rather than quietly using another, which is the point of forcing one.
///
/// @return `NULL` when the backend is unknown or unavailable here.
#[no_mangle]
pub extern "C" fn ef_image_processor_new_with_backend(backend: u32) -> *mut EfImageProcessor {
    catch_unwind(|| {
        let Some(b) = edgefirst_image::ComputeBackend::from_code(backend) else {
            return std::ptr::null_mut();
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

/// Set how `ef_image_processor_convert` resamples when it resizes, for
/// converts whose options do not set their own
/// ([`ef_convert_options_set_interpolation`]).
///
/// `mode` is an `EF_INTERPOLATION_*` code: `EF_INTERPOLATION_BILINEAR` (the
/// default; half-pixel centres, as OpenCV `INTER_LINEAR`) or
/// `EF_INTERPOLATION_NEAREST`. `EF_INTERPOLATION_BICUBIC` and
/// `EF_INTERPOLATION_AREA` are accepted and run as bilinear, logged once per
/// mode. Applies to every backend the processor holds; the
/// `EDGEFIRST_INTERPOLATION` environment variable (`nearest` | `bilinear`),
/// read when the processor is created, overrides it.
///
/// @return 0 on success, `EINVAL` for a null processor or unknown mode,
/// `EIO` when a backend cannot apply the mode.
///
/// # Safety
/// `p` must be a live processor.
#[no_mangle]
pub unsafe extern "C" fn ef_image_processor_set_interpolation(
    p: *mut EfImageProcessor,
    mode: u32,
) -> c_int {
    unsafe {
        catch_unwind(AssertUnwindSafe(|| {
            if p.is_null() {
                return libc::EINVAL;
            }
            let Some(mode) = Interpolation::from_code(mode) else {
                return libc::EINVAL;
            };
            match (*p).inner.set_interpolation(mode) {
                Ok(()) => 0,
                Err(_) => libc::EIO,
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
/// `options` as for [`ef_image_processor_convert`].
///
/// @return the codes [`ef_image_processor_convert`] returns, and `ENOTSUP`
/// off Unix.
///
/// # Safety
/// `p`, `src`, `dst` must be live handles; `fence_fd` must be writable.
#[no_mangle]
pub unsafe extern "C" fn ef_image_processor_convert_fence(
    p: *mut EfImageProcessor,
    src: *const EfTensor,
    dst: *mut EfTensor,
    options: *const EfConvertOptions,
    fence_fd: *mut c_int,
) -> c_int {
    unsafe {
        catch_unwind(AssertUnwindSafe(|| {
            if p.is_null() || fence_fd.is_null() {
                return libc::EINVAL;
            }
            // Written before any early return, so a caller never reads an
            // uninitialised descriptor after a failure.
            *fence_fd = -1;
            #[cfg(not(unix))]
            {
                let _ = (p, src, dst, options);
                libc::ENOTSUP
            }
            #[cfg(unix)]
            {
                let result = with_tensor(src, |s| {
                    with_tensor_mut(dst, |d| {
                        (*p).inner.convert_with_fence(s, d, options_from(options))
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
                    Ok(Ok(Err(e))) => image_errno(&e),
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
/// `options` as for [`ef_image_processor_convert`].
///
/// @return the codes [`ef_image_processor_convert`] returns, and `ENOTSUP`
/// off Windows.
///
/// # Safety
/// `p`, `src`, `dst` must be live handles; `fence` must be writable.
#[no_mangle]
pub unsafe extern "C" fn ef_image_processor_convert_fence_handle(
    p: *mut EfImageProcessor,
    src: *const EfTensor,
    dst: *mut EfTensor,
    options: *const EfConvertOptions,
    fence: *mut *mut c_void,
) -> c_int {
    unsafe {
        catch_unwind(AssertUnwindSafe(|| {
            if p.is_null() || fence.is_null() {
                return libc::EINVAL;
            }
            // Written before any early return, as the fd sibling above does,
            // so a caller reading `*fence` after a failure never sees an
            // uninitialised pointer.
            *fence = std::ptr::null_mut();
            #[cfg(not(target_os = "windows"))]
            {
                let _ = (src, dst, options);
                libc::ENOTSUP
            }
            #[cfg(target_os = "windows")]
            {
                use std::os::windows::io::IntoRawHandle;
                let result = with_tensor(src, |s| {
                    with_tensor_mut(dst, |d| {
                        (*p).inner.convert_with_fence(s, d, options_from(options))
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
                    Ok(Ok(Err(e))) => image_errno(&e),
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
            let rc = ef_image_processor_convert(p, src, dst, std::ptr::null());
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
    fn an_unknown_option_code_is_refused() {
        // Treating an unknown code as "none" would rotate nothing and read as
        // a backend failure rather than a caller mistake.
        unsafe {
            let o = ef_convert_options_new();
            assert_eq!(ef_convert_options_set_rotation(o, 99), libc::EINVAL);
            assert_eq!(ef_convert_options_set_flip(o, 99), libc::EINVAL);
            assert_eq!(
                ef_convert_options_set_letterbox_placement(o, 2),
                libc::EINVAL
            );
            assert_eq!(
                ef_convert_options_set_normalization_preset(o, 4),
                libc::EINVAL
            );
            assert_eq!(
                ef_convert_options_set_rotation(std::ptr::null_mut(), 0),
                libc::EINVAL
            );
            assert_eq!(
                (*o).inner,
                ConvertOptions::new(),
                "a refused code changes nothing"
            );
            assert_eq!(ef_convert_options_set_rotation(o, 1), 0);
            assert_eq!((*o).inner.get_rotation(), Rotation::Clockwise90);
            ef_convert_options_free(o);
            ef_convert_options_free(std::ptr::null_mut());
        }
    }

    #[test]
    fn options_setters_build_the_rust_options() {
        unsafe {
            let o = ef_convert_options_new();
            assert_eq!(ef_convert_options_set_source(o, 1, 2, 0, 9), 0);
            assert_eq!(
                (*o).inner.get_crop().source,
                None,
                "a zero width is the whole source"
            );
            assert_eq!(ef_convert_options_set_source(o, 1, 2, 3, 4), 0);
            assert_eq!(
                ef_convert_options_set_letterbox(o, [0u8, 0, 0, 255].as_ptr()),
                0
            );
            assert_eq!(ef_convert_options_set_letterbox_placement(o, 1), 0);
            assert_eq!(
                ef_convert_options_set_pad_output(o, [0.0f32; 4].as_ptr()),
                0
            );
            assert_eq!(ef_convert_options_set_normalization_preset(o, 3), 0);
            let want = ConvertOptions::new()
                .source(Some(edgefirst_tensor::Region::new(1, 2, 3, 4)))
                .letterbox([0, 0, 0, 255])
                .placement(LetterboxPlacement::TopLeft)
                .pad_output([0.0; 4])
                .normalization(Normalization::imagenet());
            assert_eq!((*o).inner, want);
            let (mean, std) = ([0.0f32, 1.0], [1.0f32, 1.0]);
            assert_eq!(
                ef_convert_options_set_normalization(o, mean.as_ptr(), std.as_ptr(), 2),
                libc::EINVAL
            );
            assert_eq!(
                ef_convert_options_set_normalization(o, mean.as_ptr(), [0.0f32].as_ptr(), 1),
                libc::EINVAL
            );
            assert_eq!(ef_convert_options_set_letterbox(o, std::ptr::null()), 0);
            assert_eq!((*o).inner.get_crop().fit, Fit::Stretch);
            ef_convert_options_free(o);
        }
    }

    /// The spec a C caller fills for a centred letterbox with ImageNet.
    fn c_spec() -> EfPreprocessSpec {
        EfPreprocessSpec {
            width: 640,
            height: 640,
            fit: FitMode::Letterbox.code(),
            placement: LetterboxPlacement::Center.code(),
            pad_space: PadSpace::Pixel.code(),
            pad: [114.0; 3],
            mean: Normalization::imagenet().mean(),
            std: Normalization::imagenet().std(),
            interpolation: Interpolation::Bilinear.code(),
        }
    }

    #[test]
    fn a_spec_builds_the_options_the_rust_api_does() {
        let c = c_spec();
        let mut want = PreprocessSpec::new();
        want.normalization = Normalization::imagenet();
        want.size = Some((640, 640));
        assert_eq!(unsafe { spec_from_c(&c) }, Ok(want));
        unsafe {
            let o = ef_convert_options_from_spec(&c);
            assert!(!o.is_null());
            assert_eq!((*o).inner, ConvertOptions::from_spec(&want).unwrap());
            ef_convert_options_free(o);
        }
    }

    #[test]
    fn an_invalid_spec_is_einval() {
        let bad = |edit: &dyn Fn(&mut EfPreprocessSpec)| {
            let mut c = c_spec();
            edit(&mut c);
            errno::set_errno(errno::Errno(0));
            let o = unsafe { ef_convert_options_from_spec(&c) };
            assert!(o.is_null());
            assert_eq!(errno::errno().0, libc::EINVAL);
        };
        bad(&|c| c.fit = 2);
        bad(&|c| c.placement = 2);
        bad(&|c| c.pad_space = 2);
        bad(&|c| c.interpolation = 4);
        bad(&|c| c.std = [0.0; 3]);
        bad(&|c| c.pad = [300.0; 3]);
        assert!(unsafe { ef_convert_options_from_spec(std::ptr::null()) }.is_null());
    }

    #[test]
    fn the_spec_interpolation_is_a_convert_option() {
        let mut c = c_spec();
        c.interpolation = Interpolation::Nearest.code();
        unsafe {
            let o = ef_convert_options_from_spec(&c);
            assert!(!o.is_null());
            assert_eq!((*o).inner.get_interpolation(), Some(Interpolation::Nearest));
            ef_convert_options_free(o);
        }
    }

    #[test]
    fn the_interpolation_setter_takes_every_code() {
        unsafe {
            let o = ef_convert_options_new();
            assert_eq!((*o).inner.get_interpolation(), None);
            for &mode in Interpolation::all() {
                assert_eq!(ef_convert_options_set_interpolation(o, mode.code()), 0);
                assert_eq!((*o).inner.get_interpolation(), Some(mode));
            }
            assert_eq!(ef_convert_options_set_interpolation(o, 4), libc::EINVAL);
            assert_eq!(
                (*o).inner.get_interpolation(),
                Some(Interpolation::Area),
                "a refused code leaves the options unchanged"
            );
            assert_eq!(
                ef_convert_options_set_interpolation(std::ptr::null_mut(), 0),
                libc::EINVAL
            );
            ef_convert_options_free(o);
        }
    }

    #[test]
    fn every_interpolation_code_is_accepted() {
        let p = ef_image_processor_new_with_backend(1);
        assert!(!p.is_null());
        unsafe {
            for &mode in Interpolation::all() {
                assert_eq!(ef_image_processor_set_interpolation(p, mode.code()), 0);
            }
            assert_eq!(ef_image_processor_set_interpolation(p, 4), libc::EINVAL);
            assert_eq!(
                ef_image_processor_set_interpolation(std::ptr::null_mut(), 0),
                libc::EINVAL
            );
            ef_image_processor_free(p);
        }
    }

    #[test]
    fn content_region_matches_the_rust_options() {
        let c = c_spec();
        unsafe {
            let o = ef_convert_options_from_spec(&c);
            let mut px = [0u32; 4];
            let mut norm = [0f32; 4];
            assert_eq!(
                ef_convert_options_content_region(
                    o,
                    1920,
                    1080,
                    640,
                    640,
                    px.as_mut_ptr(),
                    norm.as_mut_ptr()
                ),
                0
            );
            let want = (*o).inner.content_region(1920, 1080, 640, 640).unwrap();
            assert_eq!(px, [0, 140, 640, 360]);
            assert_eq!(norm, want.normalized);
            assert_eq!(
                ef_convert_options_content_region(
                    o,
                    1920,
                    1080,
                    640,
                    640,
                    std::ptr::null_mut(),
                    std::ptr::null_mut()
                ),
                0
            );
            assert_eq!(
                ef_convert_options_content_region(
                    o,
                    0,
                    1080,
                    640,
                    640,
                    px.as_mut_ptr(),
                    std::ptr::null_mut()
                ),
                libc::EINVAL
            );
            assert_eq!(ef_convert_options_set_source(o, 1900, 0, 100, 100), 0);
            assert_eq!(
                ef_convert_options_content_region(
                    o,
                    1920,
                    1080,
                    640,
                    640,
                    px.as_mut_ptr(),
                    std::ptr::null_mut()
                ),
                libc::EINVAL,
                "a crop outside the source"
            );
            ef_convert_options_free(o);
            let stretch = ef_convert_options_new();
            assert_eq!(
                ef_convert_options_content_region(
                    stretch,
                    1920,
                    1080,
                    640,
                    640,
                    px.as_mut_ptr(),
                    norm.as_mut_ptr()
                ),
                0
            );
            assert_eq!((px, norm), ([0, 0, 640, 640], [0.0, 0.0, 1.0, 1.0]));
            ef_convert_options_free(stretch);
            assert_eq!(
                ef_convert_options_content_region(
                    std::ptr::null(),
                    1,
                    1,
                    1,
                    1,
                    px.as_mut_ptr(),
                    norm.as_mut_ptr()
                ),
                libc::EINVAL
            );
        }
    }

    #[test]
    fn integer_normalization_without_quantization_is_einval() {
        let p = processor();
        let rgba = std::ffi::CString::new("rgba8").unwrap();
        let rgb = std::ffi::CString::new("rgb8").unwrap();
        unsafe {
            let src = ef_image_processor_create_image(p, 16, 16, rgba.as_ptr(), 0, 0, 3);
            // dtype 1 is EF_DTYPE_I8.
            let dst = ef_image_processor_create_image(p, 16, 16, rgb.as_ptr(), 1, 0, 3);
            assert!(!src.is_null() && !dst.is_null());
            let o = ef_convert_options_new();
            assert_eq!(ef_convert_options_set_normalization_preset(o, 3), 0);
            assert_eq!(ef_image_processor_convert(p, src, dst, o), libc::EINVAL);
            ef_convert_options_free(o);
            edgefirst_tensor_ffi::ef_tensor_free(src);
            edgefirst_tensor_ffi::ef_tensor_free(dst);
            ef_image_processor_free(p);
        }
    }

    /// A CPU convert into a destination declared read-only, and a CPU draw
    /// into one declared write-only, are `EACCES`, as a tensor map is.
    #[test]
    fn a_cpu_access_refusal_is_eacces() {
        let p = ef_image_processor_new_with_backend(1);
        assert!(!p.is_null(), "the CPU backend exists everywhere");
        let rgba = std::ffi::CString::new("rgba8").unwrap();
        let image = |fmt, access| {
            TensorDyn::image(16, 16, fmt, DType::U8, Some(TensorMemory::Mem), access)
                .unwrap()
                .into_raw()
        };
        unsafe {
            let src = ef_image_processor_create_image(p, 16, 16, rgba.as_ptr(), 0, 0, 3);
            let read_only = image(PixelFormat::Rgb, CpuAccess::Read);
            let write_only = image(PixelFormat::Rgba, CpuAccess::Write);
            assert_eq!(
                ef_image_processor_convert(p, src, read_only, std::ptr::null()),
                libc::EACCES
            );
            assert_eq!(
                crate::draw::ef_image_processor_draw_decoded_masks(
                    p,
                    write_only,
                    std::ptr::null(),
                    0,
                    std::ptr::null(),
                    0,
                    std::ptr::null(),
                    1.0,
                    std::ptr::null(),
                    0,
                ),
                libc::EACCES
            );
            edgefirst_tensor_ffi::ef_tensor_free(src);
            edgefirst_tensor_ffi::ef_tensor_free(read_only);
            edgefirst_tensor_ffi::ef_tensor_free(write_only);
            ef_image_processor_free(p);
        }
        let fallback = edgefirst_image::Error::CpuFallbackDenied {
            declined: "opengl: unsupported source".into(),
            denied: edgefirst_tensor::Error::CpuAccessDenied {
                declared: CpuAccess::None,
                requested: CpuAccess::Write,
            },
        };
        assert_eq!(image_errno(&fallback), libc::EACCES);
        assert_eq!(
            image_errno(&edgefirst_image::Error::NotSupported("x".into())),
            libc::ENOTSUP
        );
    }

    /// A `create_image` destination requested with no CPU access still takes
    /// the CPU backend's convert: the processor adds the write it needs.
    #[test]
    fn create_image_none_then_cpu_convert_succeeds() {
        let p = ef_image_processor_new_with_backend(1);
        assert!(!p.is_null(), "the CPU backend exists everywhere");
        let rgba = std::ffi::CString::new("rgba8").unwrap();
        let rgb = std::ffi::CString::new("rgb8").unwrap();
        unsafe {
            let src = ef_image_processor_create_image(p, 16, 16, rgba.as_ptr(), 0, 0, 3);
            let dst = ef_image_processor_create_image(p, 16, 16, rgb.as_ptr(), 0, 0, 0);
            assert!(!src.is_null() && !dst.is_null());
            assert_eq!(
                edgefirst_tensor_ffi::ef_tensor_cpu_access(dst),
                CpuAccess::Write.code()
            );
            assert_eq!(ef_image_processor_convert(p, src, dst, std::ptr::null()), 0);
            edgefirst_tensor_ffi::ef_tensor_free(src);
            edgefirst_tensor_ffi::ef_tensor_free(dst);
            ef_image_processor_free(p);
        }
    }

    /// `tile_into` reports its failure through `errno`, a refusal as `EACCES`.
    #[test]
    fn tile_into_refusal_sets_errno() {
        use crate::tiling::{ef_image_processor_tile_into, ef_tiling_config_default};
        let p = ef_image_processor_new_with_backend(1);
        assert!(!p.is_null(), "the CPU backend exists everywhere");
        let cfg = ef_tiling_config_default(16, 16);
        let image = |access| {
            TensorDyn::image(
                16,
                16,
                PixelFormat::Rgb,
                DType::U8,
                Some(TensorMemory::Mem),
                access,
            )
            .unwrap()
            .into_raw()
        };
        unsafe {
            let src = image(CpuAccess::ReadWrite);
            let read_only = image(CpuAccess::Read);
            errno::set_errno(errno::Errno(0));
            assert!(
                ef_image_processor_tile_into(p, src, read_only, &cfg, std::ptr::null()).is_null()
            );
            assert_eq!(errno::errno().0, libc::EACCES);
            errno::set_errno(errno::Errno(0));
            assert!(ef_image_processor_tile_into(
                std::ptr::null_mut(),
                src,
                read_only,
                &cfg,
                std::ptr::null()
            )
            .is_null());
            assert_eq!(errno::errno().0, libc::EINVAL);
            edgefirst_tensor_ffi::ef_tensor_free(src);
            edgefirst_tensor_ffi::ef_tensor_free(read_only);
            ef_image_processor_free(p);
        }
    }

    #[test]
    fn every_rotation_and_flip_code_maps_and_the_rest_do_not() {
        for (c, r) in [
            (0, edgefirst_image::Rotation::None),
            (1, edgefirst_image::Rotation::Clockwise90),
            (2, edgefirst_image::Rotation::Rotate180),
            (3, edgefirst_image::Rotation::CounterClockwise90),
        ] {
            assert_eq!(
                edgefirst_image::Rotation::from_code(c),
                Some(r),
                "rotation code {c}"
            );
        }
        assert_eq!(edgefirst_image::Rotation::from_code(4), None);
        for (c, f) in [
            (0, edgefirst_image::Flip::None),
            (1, edgefirst_image::Flip::Vertical),
            (2, edgefirst_image::Flip::Horizontal),
        ] {
            assert_eq!(
                edgefirst_image::Flip::from_code(c),
                Some(f),
                "flip code {c}"
            );
        }
        assert_eq!(edgefirst_image::Flip::from_code(3), None);
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
        for &a in CpuAccess::all() {
            assert_eq!(cpu_access_from_code(a.code()), Some(a));
        }
        assert_eq!(cpu_access_from_code(CpuAccess::all().len() as u32), None);
    }

    /// A view with every field away from its default -- 64x48 NV12 U8,
    /// `Shm`, `ReadWrite`, `Any` compression.
    fn full_view() -> EfImageDescView {
        EfImageDescView {
            width: 64,
            height: 48,
            format: PixelFormat::Nv12.code(),
            dtype: DType::U8.code(),
            access: CpuAccess::ReadWrite.code(),
            memory: TensorMemory::Shm.code(),
            has_memory: 1,
            compression: EfCompressionRequest::Any as u32,
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
        // `Scheme` ("a specific vendor scheme") is a real state
        // `ef_tensor_image_desc_get` can report, but no C setter can ever
        // request one, so there is nothing to reconstruct -- refused, same
        // as an unrecognized code, not silently treated as `Any`.
        let mut v = full_view();
        v.compression = EfCompressionRequest::Scheme as u32;
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
    /// `tests/c/test_cross_library.c` drives
    /// `ef_image_processor_convert_fence_handle` from C on Windows as well
    /// (the MSVC harness in `lib.rs`); this is the Rust-side check of the
    /// same event contract, with no compiler between the test and the
    /// export, the role `a_texture_tensor_round_trips_through_the_exports`
    /// plays for the tensor leaf's D3D11 family (`tensor-capi/src/d3d11.rs`).
    #[cfg(target_os = "windows")]
    #[test]
    fn convert_fence_handle_returns_an_owned_event_set_on_completion() {
        if !edgefirst_tensor::is_gpu_buffer_available() {
            crate::artifacts::skip("no D3D11 device on this host");
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
            let rc =
                ef_image_processor_convert_fence_handle(p, src, dst, std::ptr::null(), &mut fence);
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
