// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! Turning model output tensors into detections and masks.

use std::ffi::{c_char, c_int, CStr, CString};
use std::panic::{catch_unwind, AssertUnwindSafe};

use edgefirst_decoder::{
    configs, dequantize_cpu_chunked, ConfigOutput, ConfigOutputs, DecoderBuilder, ProtoLayout,
    Quantization, Segmentation,
};
pub use edgefirst_decoder_abi::{EfDetectBox, EfSegmentation};
use edgefirst_tensor::{DetectBox, TensorDyn};
use edgefirst_tensor_ffi::EfTensor;

/// An owned list of detections.
///
/// Opaque and heap-owned rather than a `(ptr, len)` pair, so a consumer cannot
/// outlive the producer's allocation by holding the pointer. This is the one
/// implementation of `ef_detect_box_list`: the tracker and any other consumer
/// read a list via [`ef_detect_box_list_data`]/[`ef_detect_box_list_len`]
/// without linking this library.
pub struct EfDetectBoxList {
    boxes: Vec<EfDetectBox>,
}

/// Create an empty detection list. `NULL` on allocation failure.
#[no_mangle]
pub extern "C" fn ef_detect_box_list_new() -> *mut EfDetectBoxList {
    catch_unwind(|| Box::into_raw(Box::new(EfDetectBoxList { boxes: Vec::new() })))
        .unwrap_or(std::ptr::null_mut())
}

/// Append a detection.
///
/// @return 0 on success, `EINVAL` for a null list or box.
///
/// # Safety
/// `l` and `b` must be `NULL` or valid.
#[no_mangle]
pub unsafe extern "C" fn ef_detect_box_list_push(
    l: *mut EfDetectBoxList,
    b: *const EfDetectBox,
) -> c_int {
    unsafe {
        catch_unwind(AssertUnwindSafe(|| {
            if l.is_null() || b.is_null() {
                return libc::EINVAL;
            }
            (*l).boxes.push(*b);
            0
        }))
        .unwrap_or(libc::EINVAL)
    }
}

/// Decoder configuration, accumulated field by field.
///
/// A builder rather than a struct for the same reason the tensor builder is:
/// per-field errors, and room to grow without breaking a caller's layout.
pub struct EfDecoderParams {
    config_json: Option<String>,
    config_yaml: Option<String>,
    config_file: Option<String>,
    outputs: Vec<ConfigOutput>,
    decoder_version: Option<configs::DecoderVersion>,
    score_threshold: f32,
    iou_threshold: f32,
    nms: Option<configs::Nms>,
    pre_nms_top_k: usize,
    max_det: usize,
    input_dims: Option<(usize, usize)>,
}

/// An opaque decoder.
pub struct EfDecoder {
    inner: edgefirst_decoder::Decoder,
}

/// An owned list of segmentation masks.
pub struct EfSegmentationList {
    masks: Vec<Segmentation>,
    /// Borrowed views handed to C, materialised on first request.
    views: Option<Vec<EfSegmentation>>,
    /// Host pins keeping the mask buffers mapped for as long as C holds a
    /// pointer into them. Dropping these with a view still outstanding is what
    /// would turn a valid pointer into a dangling one.
    pins: Vec<edgefirst_tensor::HostPin<'static>>,
}

/// Create decoder parameters with the library's defaults.
#[no_mangle]
pub extern "C" fn ef_decoder_params_new() -> *mut EfDecoderParams {
    catch_unwind(|| {
        Box::into_raw(Box::new(EfDecoderParams {
            config_json: None,
            config_yaml: None,
            config_file: None,
            outputs: Vec::new(),
            decoder_version: None,
            score_threshold: 0.5,
            iou_threshold: 0.5,
            nms: Some(configs::Nms::Auto),
            pre_nms_top_k: 300,
            max_det: 100,
            input_dims: None,
        }))
    })
    .unwrap_or(std::ptr::null_mut())
}

/// Free decoder parameters. Freeing `NULL` is a no-op.
///
/// # Safety
/// `p` must be `NULL` or have come from this library.
#[no_mangle]
pub unsafe extern "C" fn ef_decoder_params_free(p: *mut EfDecoderParams) {
    unsafe {
        if p.is_null() {
            return;
        }
        let _ = catch_unwind(AssertUnwindSafe(|| drop(Box::from_raw(p))));
    }
}

/// Run a setter, rejecting a null handle.
unsafe fn with_params<F>(p: *mut EfDecoderParams, body: F) -> c_int
where
    F: FnOnce(&mut EfDecoderParams) -> c_int,
{
    unsafe {
        catch_unwind(AssertUnwindSafe(|| {
            if p.is_null() {
                return libc::EINVAL;
            }
            body(&mut *p)
        }))
        .unwrap_or(libc::EINVAL)
    }
}

/// Minimum confidence for a detection to be kept.
///
/// Written out rather than macro-generated: **cbindgen does not expand macros**,
/// so a macro-defined `extern "C"` fn is exported by the library and absent
/// from the header — present in `nm`, undeclarable by any C caller. That is a
/// silent break, caught by the leaf's header-parity tests.
///
/// # Safety
/// `p` must be `NULL` or a live parameter set.
#[no_mangle]
pub unsafe extern "C" fn ef_decoder_params_set_score_threshold(
    p: *mut EfDecoderParams,
    v: f32,
) -> c_int {
    unsafe {
        with_params(p, |p| {
            p.score_threshold = v;
            0
        })
    }
}

/// IoU above which NMS suppresses the lower-scoring box.
///
/// # Safety
/// `p` must be `NULL` or a live parameter set.
#[no_mangle]
pub unsafe extern "C" fn ef_decoder_params_set_iou_threshold(
    p: *mut EfDecoderParams,
    v: f32,
) -> c_int {
    unsafe {
        with_params(p, |p| {
            p.iou_threshold = v;
            0
        })
    }
}

/// How many candidates survive into NMS. Bounds the worst case.
///
/// # Safety
/// `p` must be `NULL` or a live parameter set.
#[no_mangle]
pub unsafe extern "C" fn ef_decoder_params_set_pre_nms_top_k(
    p: *mut EfDecoderParams,
    v: usize,
) -> c_int {
    unsafe {
        with_params(p, |p| {
            p.pre_nms_top_k = v;
            0
        })
    }
}

/// Maximum detections returned per frame.
///
/// # Safety
/// `p` must be `NULL` or a live parameter set.
#[no_mangle]
pub unsafe extern "C" fn ef_decoder_params_set_max_det(p: *mut EfDecoderParams, v: usize) -> c_int {
    unsafe {
        with_params(p, |p| {
            p.max_det = v;
            0
        })
    }
}

/// Set the model's input dimensions, when the config does not carry them.
///
/// # Safety
/// `p` must be `NULL` or a live parameter set.
#[no_mangle]
pub unsafe extern "C" fn ef_decoder_params_set_input_dims(
    p: *mut EfDecoderParams,
    width: usize,
    height: usize,
) -> c_int {
    unsafe {
        with_params(p, |p| {
            if width == 0 || height == 0 {
                return libc::EINVAL;
            }
            p.input_dims = Some((width, height));
            0
        })
    }
}

/// NMS mode: 0 = off, 1 = automatic, 2 = class-aware, 3 = class-agnostic.
///
/// # Safety
/// `p` must be `NULL` or a live parameter set.
#[no_mangle]
pub unsafe extern "C" fn ef_decoder_params_set_nms(p: *mut EfDecoderParams, nms: u32) -> c_int {
    unsafe {
        with_params(p, |p| {
            p.nms = match nms {
                0 => None,
                1 => Some(configs::Nms::Auto),
                2 => Some(configs::Nms::ClassAware),
                3 => Some(configs::Nms::ClassAgnostic),
                _ => return libc::EINVAL,
            };
            0
        })
    }
}

/// Read a C string of `len` bytes, or NUL-terminated when `len` is 0.
unsafe fn read_str(p: *const c_char, len: usize) -> Option<String> {
    unsafe {
        if p.is_null() {
            return None;
        }
        if len == 0 {
            CStr::from_ptr(p).to_str().ok().map(str::to_string)
        } else {
            // c_char is i8 on macOS and u8 on aarch64-linux, so this cast is
            // load-bearing on one target and a no-op on the other. Neither can be
            // written without the other's lint firing.
            #[allow(clippy::unnecessary_cast)]
            let bytes = std::slice::from_raw_parts(p as *const u8, len);
            std::str::from_utf8(bytes).ok().map(str::to_string)
        }
    }
}

/// Configure from a JSON string. `len` may be 0 for NUL-terminated.
///
/// # Safety
/// `json` must be readable for `len` bytes, or NUL-terminated.
#[no_mangle]
pub unsafe extern "C" fn ef_decoder_params_set_config_json(
    p: *mut EfDecoderParams,
    json: *const c_char,
    len: usize,
) -> c_int {
    unsafe {
        with_params(p, |p| match read_str(json, len) {
            Some(s) => {
                p.config_json = Some(s);
                0
            }
            None => libc::EINVAL,
        })
    }
}

/// Configure from a YAML string. `len` may be 0 for NUL-terminated.
///
/// # Safety
/// `yaml` must be readable for `len` bytes, or NUL-terminated.
#[no_mangle]
pub unsafe extern "C" fn ef_decoder_params_set_config_yaml(
    p: *mut EfDecoderParams,
    yaml: *const c_char,
    len: usize,
) -> c_int {
    unsafe {
        with_params(p, |p| match read_str(yaml, len) {
            Some(s) => {
                p.config_yaml = Some(s);
                0
            }
            None => libc::EINVAL,
        })
    }
}

/// Configure from a file, JSON or YAML detected by extension and content.
///
/// # Safety
/// `path` must be a NUL-terminated string.
#[no_mangle]
pub unsafe extern "C" fn ef_decoder_params_set_config_file(
    p: *mut EfDecoderParams,
    path: *const c_char,
) -> c_int {
    unsafe {
        with_params(p, |p| match read_str(path, 0) {
            Some(s) => {
                p.config_file = Some(s);
                0
            }
            None => libc::EINVAL,
        })
    }
}

/// Build a decoder. `NULL` on failure.
///
/// **Exactly one** configuration source must be set — JSON, YAML, or a file.
/// Two sources disagreeing has no defined resolution, so supplying none or more
/// than one is an error rather than a precedence rule nobody remembers.
///
/// The parameters are not consumed and may be reused.
///
/// # Safety
/// `p` must be a live parameter set.
#[no_mangle]
pub unsafe extern "C" fn ef_decoder_new(p: *const EfDecoderParams) -> *mut EfDecoder {
    unsafe {
        catch_unwind(AssertUnwindSafe(|| {
            if p.is_null() {
                return std::ptr::null_mut();
            }
            let p = &*p;
            let sources = p.config_json.is_some() as u8
                + p.config_yaml.is_some() as u8
                + p.config_file.is_some() as u8
                + (!p.outputs.is_empty()) as u8;
            if sources != 1 {
                return std::ptr::null_mut();
            }
            let mut b = DecoderBuilder::new()
                .with_score_threshold(p.score_threshold)
                .with_iou_threshold(p.iou_threshold)
                .with_pre_nms_top_k(p.pre_nms_top_k)
                .with_max_det(p.max_det)
                .with_nms(p.nms);
            if let Some((w, h)) = p.input_dims {
                b = b.with_input_dims(w, h);
            }
            if let Some(v) = p.decoder_version {
                b = b.with_decoder_version(v);
            }
            if let Some(j) = &p.config_json {
                b = b.with_config_json_str(j.clone());
            } else if let Some(y) = &p.config_yaml {
                b = b.with_config_yaml_str(y.clone());
            } else if !p.outputs.is_empty() {
                b = b.with_config(ConfigOutputs {
                    outputs: p.outputs.clone(),
                    nms: p.nms,
                    decoder_version: p.decoder_version,
                });
            } else if let Some(f) = &p.config_file {
                let Ok(content) = std::fs::read_to_string(f) else {
                    return std::ptr::null_mut();
                };
                // Extension first, then a content sniff: a `.yaml` holding JSON is
                // rarer than a config file with no extension at all.
                if f.ends_with(".json") || content.trim_start().starts_with('{') {
                    b = b.with_config_json_str(content);
                } else {
                    b = b.with_config_yaml_str(content);
                }
            }
            match b.build() {
                Ok(inner) => Box::into_raw(Box::new(EfDecoder { inner })),
                Err(_) => std::ptr::null_mut(),
            }
        }))
        .unwrap_or(std::ptr::null_mut())
    }
}

/// Free a decoder. Freeing `NULL` is a no-op.
///
/// # Safety
/// `d` must be `NULL` or have come from this library.
#[no_mangle]
pub unsafe extern "C" fn ef_decoder_free(d: *mut EfDecoder) {
    unsafe {
        if d.is_null() {
            return;
        }
        let _ = catch_unwind(AssertUnwindSafe(|| drop(Box::from_raw(d))));
    }
}

/// The model's input dimensions, when known.
///
/// @return 0 on success, `ENODATA` when the configuration did not declare them.
///
/// # Safety
/// `width` and `height` must be writable.
#[no_mangle]
pub unsafe extern "C" fn ef_decoder_input_dims(
    d: *const EfDecoder,
    width: *mut usize,
    height: *mut usize,
) -> c_int {
    unsafe {
        catch_unwind(AssertUnwindSafe(|| {
            if d.is_null() || width.is_null() || height.is_null() {
                return libc::EINVAL;
            }
            match (*d).inner.input_dims() {
                Some((w, h)) => {
                    *width = w;
                    *height = h;
                    0
                }
                None => libc::ENODATA,
            }
        }))
        .unwrap_or(libc::EINVAL)
    }
}

/// Whether the model emits normalized box coordinates.
///
/// @return 1 yes, 0 no, -1 when the configuration does not say.
///
/// # Safety
/// `d` must be `NULL` or live.
#[no_mangle]
pub unsafe extern "C" fn ef_decoder_normalized_boxes(d: *const EfDecoder) -> c_int {
    unsafe {
        if d.is_null() {
            return -1;
        }
        catch_unwind(AssertUnwindSafe(|| match (*d).inner.normalized_boxes() {
            Some(true) => 1,
            Some(false) => 0,
            None => -1,
        }))
        .unwrap_or(-1)
    }
}

/// The model type as a NUL-terminated string the caller must free with
/// [`ef_decoder_string_free`].
///
/// # Safety
/// `d` must be `NULL` or live.
#[no_mangle]
pub unsafe extern "C" fn ef_decoder_model_type(d: *const EfDecoder) -> *mut c_char {
    unsafe {
        catch_unwind(AssertUnwindSafe(|| {
            if d.is_null() {
                return std::ptr::null_mut();
            }
            let s = format!("{:?}", (*d).inner.model_type());
            match CString::new(s) {
                Ok(c) => c.into_raw(),
                Err(_) => std::ptr::null_mut(),
            }
        }))
        .unwrap_or(std::ptr::null_mut())
    }
}

/// Free a string this library returned. Freeing `NULL` is a no-op.
///
/// Its own entry point rather than `free(3)`: the allocation came from Rust,
/// and on some platforms the two allocators are not the same.
///
/// # Safety
/// `s` must be `NULL` or have come from this library.
#[no_mangle]
pub unsafe extern "C" fn ef_decoder_string_free(s: *mut c_char) {
    unsafe {
        if s.is_null() {
            return;
        }
        let _ = catch_unwind(AssertUnwindSafe(|| drop(CString::from_raw(s))));
    }
}

/// Decode model outputs into detections and, for segmentation models, masks.
///
/// `out_masks` may be `NULL` when masks are not wanted.
///
/// @return 0 on success. Both out-parameters are written only on success, so a
///         caller never has to free a partially-populated result.
///
/// # Safety
/// `outputs` must point to `count` live tensor handles.
#[no_mangle]
pub unsafe extern "C" fn ef_decoder_decode(
    d: *const EfDecoder,
    outputs: *const *const EfTensor,
    count: usize,
    out_boxes: *mut *mut EfDetectBoxList,
    out_masks: *mut *mut EfSegmentationList,
) -> c_int {
    unsafe {
        catch_unwind(AssertUnwindSafe(|| {
            if d.is_null() || outputs.is_null() || out_boxes.is_null() || count == 0 {
                return libc::EINVAL;
            }
            let handles = std::slice::from_raw_parts(outputs, count);
            // Every handle is a real `ef_tensor` from `libedgefirst_tensor.so`
            // (there is only one implementation), so each is borrowed
            // straight from its raw pointer -- no per-library dispatch, no
            // re-import. `ManuallyDrop` suppresses `TensorDyn`'s own `Drop`
            // (which would otherwise call `ef_tensor_free` on a handle the
            // caller still owns), and the `Vec` holds every borrow alive for
            // the duration of the decode call below.
            let mut borrowed: Vec<std::mem::ManuallyDrop<TensorDyn>> = Vec::with_capacity(count);
            for h in handles {
                if h.is_null() {
                    return libc::EINVAL;
                }
                borrowed.push(std::mem::ManuallyDrop::new(TensorDyn::from_raw(
                    *h as *mut EfTensor,
                )));
            }
            let refs: Vec<&TensorDyn> = borrowed.iter().map(|t| &**t).collect();

            let mut boxes: Vec<DetectBox> = Vec::new();
            let mut masks: Vec<Segmentation> = Vec::new();
            if (*d).inner.decode(&refs, &mut boxes, &mut masks).is_err() {
                return libc::EBADMSG;
            }

            // Written last, together: an early return after writing `out_boxes`
            // would leak the list the caller never learned about.
            let list = Box::into_raw(Box::new(EfDetectBoxList {
                boxes: boxes.iter().map(to_c_box).collect(),
            }));
            if !out_masks.is_null() {
                *out_masks = Box::into_raw(Box::new(EfSegmentationList {
                    masks,
                    views: None,
                    pins: Vec::new(),
                }));
            }
            *out_boxes = list;
            0
        }))
        .unwrap_or(libc::EINVAL)
    }
}

fn to_c_box(d: &DetectBox) -> EfDetectBox {
    EfDetectBox {
        xmin: d.bbox.xmin,
        ymin: d.bbox.ymin,
        xmax: d.bbox.xmax,
        ymax: d.bbox.ymax,
        score: d.score,
        label: d.label as u32,
    }
}

/// Wrap already-converted detections in an owned list.
pub(crate) fn box_list_from(boxes: Vec<EfDetectBox>) -> *mut EfDetectBoxList {
    Box::into_raw(Box::new(EfDetectBoxList { boxes }))
}

/// Number of detections. Zero for a `NULL` list.
///
/// # Safety
/// `l` must be `NULL` or valid.
#[no_mangle]
pub unsafe extern "C" fn ef_detect_box_list_len(l: *const EfDetectBoxList) -> usize {
    unsafe {
        if l.is_null() {
            return 0;
        }
        catch_unwind(AssertUnwindSafe(|| {
            let boxes = &(*l).boxes;
            boxes.len()
        }))
        .unwrap_or(0)
    }
}

/// Copy detection `index` into `out`.
///
/// Copies rather than lending a pointer: a borrowed element would dangle the
/// moment the list is freed or grown, and C gives the caller no way to notice.
///
/// @return 0 on success, `EINVAL` for a null argument or out-of-range index.
///
/// # Safety
/// `out` must be writable.
#[no_mangle]
pub unsafe extern "C" fn ef_detect_box_list_get(
    l: *const EfDetectBoxList,
    index: usize,
    out: *mut EfDetectBox,
) -> c_int {
    unsafe {
        catch_unwind(AssertUnwindSafe(|| {
            if l.is_null() || out.is_null() {
                return libc::EINVAL;
            }
            // Bind through an explicit reference: `(*l).boxes.get(..)` autorefs the
            // dereference of a raw pointer, which clippy rejects.
            let boxes = &(*l).boxes;
            match boxes.get(index) {
                Some(b) => {
                    *out = *b;
                    0
                }
                None => libc::EINVAL,
            }
        }))
        .unwrap_or(libc::EINVAL)
    }
}

/// Borrow the detections as a C array, for `ef_bytetrack_update`.
///
/// A plain array rather than an opaque handle, so the tracker reads it without
/// linking this library.
///
/// # Safety
/// `l` must be `NULL` or valid.
#[no_mangle]
pub unsafe extern "C" fn ef_detect_box_list_data(l: *const EfDetectBoxList) -> *const EfDetectBox {
    unsafe {
        if l.is_null() {
            return std::ptr::null();
        }
        catch_unwind(AssertUnwindSafe(|| {
            let boxes = &(*l).boxes;
            if boxes.is_empty() {
                std::ptr::null()
            } else {
                boxes.as_ptr()
            }
        }))
        .unwrap_or(std::ptr::null())
    }
}

/// Free a detection list. Freeing `NULL` is a no-op.
///
/// There is exactly one implementation of this type, in
/// `libedgefirst_decoder`, so any `ef_detect_box_list *` from any EdgeFirst
/// entry point is freed with this function.
///
/// # Safety
/// `l` must be `NULL` or have come from this library.
#[no_mangle]
pub unsafe extern "C" fn ef_detect_box_list_free(l: *mut EfDetectBoxList) {
    unsafe {
        if l.is_null() {
            return;
        }
        let _ = catch_unwind(AssertUnwindSafe(|| drop(Box::from_raw(l))));
    }
}

/// Number of masks. Zero for a `NULL` list.
///
/// # Safety
/// `l` must be `NULL` or valid.
#[no_mangle]
pub unsafe extern "C" fn ef_segmentation_list_len(l: *const EfSegmentationList) -> usize {
    unsafe {
        if l.is_null() {
            return 0;
        }
        catch_unwind(AssertUnwindSafe(|| {
            let masks = &(*l).masks;
            masks.len()
        }))
        .unwrap_or(0)
    }
}

/// The mask region for entry `index`, in normalized coordinates.
///
/// These bound the **mask region**, which is snapped to the proto grid and so
/// encloses — rather than equals — the companion detection's box.
///
/// @return 0 on success, `EINVAL` for a null argument or bad index.
///
/// # Safety
/// All out-parameters must be writable.
#[no_mangle]
pub unsafe extern "C" fn ef_segmentation_list_get_bbox(
    l: *const EfSegmentationList,
    index: usize,
    xmin: *mut f32,
    ymin: *mut f32,
    xmax: *mut f32,
    ymax: *mut f32,
) -> c_int {
    unsafe {
        catch_unwind(AssertUnwindSafe(|| {
            if l.is_null() || xmin.is_null() || ymin.is_null() || xmax.is_null() || ymax.is_null() {
                return libc::EINVAL;
            }
            let masks = &(*l).masks;
            match masks.get(index) {
                Some(s) => {
                    *xmin = s.xmin;
                    *ymin = s.ymin;
                    *xmax = s.xmax;
                    *ymax = s.ymax;
                    0
                }
                None => libc::EINVAL,
            }
        }))
        .unwrap_or(libc::EINVAL)
    }
}

/// Borrow the masks as a C array.
///
/// Materialises the borrowed views on first call and caches them, so repeated
/// calls are free and every returned pointer stays valid for the list's life.
///
/// This is what lets `libedgefirst-image` draw decoder output without linking
/// this library: it reads plain values and a borrowed byte pointer, needing no
/// shared allocator and no symbol to resolve.
///
/// @return the first element, or `NULL` for a null or empty list. Pair with
///         [`ef_segmentation_list_len`].
///
/// # Safety
/// `l` must be `NULL` or valid, and must outlive any use of the result.
#[no_mangle]
pub unsafe extern "C" fn ef_segmentation_list_data(
    l: *mut EfSegmentationList,
) -> *const EfSegmentation {
    unsafe {
        if l.is_null() {
            return std::ptr::null();
        }
        catch_unwind(AssertUnwindSafe(|| {
            let list = &mut *l;
            if list.views.is_none() {
                let mut views = Vec::with_capacity(list.masks.len());
                for m in &list.masks {
                    // Pin the mask so the pointer stays valid for the list's life;
                    // an unpinned map would unmap at end of scope and hand C a
                    // dangling pointer that reads fine until it does not.
                    let Ok(pin) = m.segmentation.pin_host(edgefirst_tensor::CpuAccess::Read) else {
                        return std::ptr::null();
                    };
                    let shape = m.segmentation.shape();
                    let (h, w) = if shape.len() >= 2 {
                        (shape[0], shape[1])
                    } else {
                        (0, 0)
                    };
                    views.push(EfSegmentation {
                        xmin: m.xmin,
                        ymin: m.ymin,
                        xmax: m.xmax,
                        ymax: m.ymax,
                        mask: pin.as_mut_ptr(),
                        width: w as u32,
                        height: h as u32,
                    });
                    list.pins.push(pin);
                }
                list.views = Some(views);
            }
            match &list.views {
                Some(v) if !v.is_empty() => v.as_ptr(),
                _ => std::ptr::null(),
            }
        }))
        .unwrap_or(std::ptr::null())
    }
}

fn dim_name_from(code: u32) -> configs::DimName {
    match code {
        0 => configs::DimName::Batch,
        1 => configs::DimName::Height,
        2 => configs::DimName::Width,
        3 => configs::DimName::NumClasses,
        4 => configs::DimName::NumFeatures,
        5 => configs::DimName::NumBoxes,
        6 => configs::DimName::NumProtos,
        7 => configs::DimName::NumAnchorsXFeatures,
        8 => configs::DimName::Padding,
        9 => configs::DimName::BoxCoords,
        _ => configs::DimName::Unknown,
    }
}

fn decoder_type_from(code: u32) -> Option<configs::DecoderType> {
    match code {
        0 => Some(configs::DecoderType::Ultralytics),
        1 => Some(configs::DecoderType::ModelPack),
        _ => None,
    }
}

/// Append a programmatic output spec. Returns the new index, or `-1`.
///
/// `type_`: 0 detection, 1 boxes, 2 scores, 3 protos, 4 segmentation,
/// 5 mask coefficients, 6 mask, 7 classes.
/// `decoder`: 0 ultralytics, 1 modelpack.
///
/// # Safety
/// `shape` must point to `ndim` sizes; `dims` may be NULL.
#[no_mangle]
pub unsafe extern "C" fn ef_decoder_params_add_output(
    p: *mut EfDecoderParams,
    type_: u32,
    decoder: u32,
    shape: *const usize,
    dims: *const u32,
    ndim: usize,
) -> c_int {
    unsafe {
        catch_unwind(AssertUnwindSafe(|| {
            if p.is_null() || shape.is_null() || ndim == 0 {
                return -1;
            }
            let Some(decoder_type) = decoder_type_from(decoder) else {
                return -1;
            };
            let shape_slice = std::slice::from_raw_parts(shape, ndim);
            let shape_vec = shape_slice.to_vec();
            let dshape: Vec<(configs::DimName, usize)> = if dims.is_null() {
                Vec::new()
            } else {
                std::slice::from_raw_parts(dims, ndim)
                    .iter()
                    .zip(shape_slice.iter())
                    .map(|(d, s)| (dim_name_from(*d), *s))
                    .collect()
            };
            let output = match type_ {
                0 => ConfigOutput::Detection(configs::Detection {
                    decoder: decoder_type,
                    shape: shape_vec,
                    dshape,
                    ..Default::default()
                }),
                1 => ConfigOutput::Boxes(configs::Boxes {
                    decoder: decoder_type,
                    shape: shape_vec,
                    dshape,
                    ..Default::default()
                }),
                2 => ConfigOutput::Scores(configs::Scores {
                    decoder: decoder_type,
                    shape: shape_vec,
                    dshape,
                    ..Default::default()
                }),
                3 => ConfigOutput::Protos(configs::Protos {
                    decoder: decoder_type,
                    shape: shape_vec,
                    dshape,
                    ..Default::default()
                }),
                4 => ConfigOutput::Segmentation(configs::Segmentation {
                    decoder: decoder_type,
                    shape: shape_vec,
                    dshape,
                    ..Default::default()
                }),
                5 => ConfigOutput::MaskCoefficients(configs::MaskCoefficients {
                    decoder: decoder_type,
                    shape: shape_vec,
                    dshape,
                    ..Default::default()
                }),
                6 => ConfigOutput::Mask(configs::Mask {
                    decoder: decoder_type,
                    shape: shape_vec,
                    dshape,
                    ..Default::default()
                }),
                7 => ConfigOutput::Classes(configs::Classes {
                    decoder: decoder_type,
                    shape: shape_vec,
                    dshape,
                    ..Default::default()
                }),
                _ => return -1,
            };
            (*p).outputs.push(output);
            ((*p).outputs.len() - 1) as c_int
        }))
        .unwrap_or(-1)
    }
}

/// Set quantization on output `index`.
///
/// # Safety
/// `p` must be `NULL` or a live handle from this library.
#[no_mangle]
pub unsafe extern "C" fn ef_decoder_params_output_set_quantization(
    p: *mut EfDecoderParams,
    index: c_int,
    scale: f32,
    zero_point: c_int,
) -> c_int {
    unsafe {
        with_params(p, |params| {
            let idx = index as usize;
            if idx >= params.outputs.len() {
                return libc::EINVAL;
            }
            let quant = Some(configs::QuantTuple(scale, zero_point));
            match &mut params.outputs[idx] {
                ConfigOutput::Detection(c) => c.quantization = quant,
                ConfigOutput::Boxes(c) => c.quantization = quant,
                ConfigOutput::Scores(c) => c.quantization = quant,
                ConfigOutput::Protos(c) => c.quantization = quant,
                ConfigOutput::Segmentation(c) => c.quantization = quant,
                ConfigOutput::MaskCoefficients(c) => c.quantization = quant,
                ConfigOutput::Mask(c) => c.quantization = quant,
                ConfigOutput::Classes(c) => c.quantization = quant,
            }
            0
        })
    }
}

/// Set anchors on a detection output. `anchors` is `num_anchors` pairs.
///
/// # Safety
/// `p` must be live; `anchors` must point to `num_anchors` pairs.
#[no_mangle]
pub unsafe extern "C" fn ef_decoder_params_output_set_anchors(
    p: *mut EfDecoderParams,
    index: c_int,
    anchors: *const [f32; 2],
    num_anchors: usize,
) -> c_int {
    unsafe {
        catch_unwind(AssertUnwindSafe(|| {
            if p.is_null() || anchors.is_null() {
                return libc::EINVAL;
            }
            let params = &mut *p;
            let idx = index as usize;
            if idx >= params.outputs.len() {
                return libc::EINVAL;
            }
            let vec = std::slice::from_raw_parts(anchors, num_anchors).to_vec();
            match &mut params.outputs[idx] {
                ConfigOutput::Detection(c) => c.anchors = Some(vec),
                _ => return libc::EINVAL,
            }
            0
        }))
        .unwrap_or(libc::EINVAL)
    }
}

/// Mark a detection/boxes output as normalized (`1`) or pixel (`0`).
///
/// # Safety
/// `p` must be `NULL` or a live handle from this library.
#[no_mangle]
pub unsafe extern "C" fn ef_decoder_params_output_set_normalized(
    p: *mut EfDecoderParams,
    index: c_int,
    normalized: c_int,
) -> c_int {
    unsafe {
        with_params(p, |params| {
            let idx = index as usize;
            if idx >= params.outputs.len() {
                return libc::EINVAL;
            }
            let norm = Some(normalized != 0);
            match &mut params.outputs[idx] {
                ConfigOutput::Detection(c) => c.normalized = norm,
                ConfigOutput::Boxes(c) => c.normalized = norm,
                _ => return libc::EINVAL,
            }
            0
        })
    }
}

/// Decoder version: 0 Yolov5, 1 Yolov8, 2 Yolo11, 3 Yolo26.
///
/// # Safety
/// `p` must be `NULL` or a live handle from this library.
#[no_mangle]
pub unsafe extern "C" fn ef_decoder_params_set_decoder_version(
    p: *mut EfDecoderParams,
    version: u32,
) -> c_int {
    unsafe {
        with_params(p, |params| {
            params.decoder_version = Some(match version {
                0 => configs::DecoderVersion::Yolov5,
                1 => configs::DecoderVersion::Yolov8,
                2 => configs::DecoderVersion::Yolo11,
                3 => configs::DecoderVersion::Yolo26,
                _ => return libc::EINVAL,
            });
            0
        })
    }
}

/// Prototype tensors from [`ef_decoder_decode_proto`].
pub struct EfProtoData {
    protos: Option<edgefirst_tensor::TensorDyn>,
    mask_coefficients: Option<edgefirst_tensor::TensorDyn>,
    layout: ProtoLayout,
}

/// Decode detections and, for segmentation models, proto tensors.
///
/// # Safety
/// `outputs` must point to `count` live handles.
#[no_mangle]
pub unsafe extern "C" fn ef_decoder_decode_proto(
    d: *const EfDecoder,
    outputs: *const *const EfTensor,
    count: usize,
    out_boxes: *mut *mut EfDetectBoxList,
) -> *mut EfProtoData {
    unsafe {
        catch_unwind(AssertUnwindSafe(|| {
            if d.is_null() || outputs.is_null() || out_boxes.is_null() || count == 0 {
                return std::ptr::null_mut();
            }
            let Ok((refs, _keep)) = borrow_outputs(outputs, count) else {
                return std::ptr::null_mut();
            };
            let mut boxes: Vec<DetectBox> = Vec::new();
            let proto = match (*d).inner.decode_proto(&refs, &mut boxes) {
                Ok(p) => p,
                Err(_) => return std::ptr::null_mut(),
            };
            *out_boxes = Box::into_raw(Box::new(EfDetectBoxList {
                boxes: boxes.iter().map(to_c_box).collect(),
            }));
            match proto {
                Some(p) => Box::into_raw(Box::new(EfProtoData {
                    protos: Some(p.protos),
                    mask_coefficients: Some(p.mask_coefficients),
                    layout: p.layout,
                })),
                None => std::ptr::null_mut(),
            }
        }))
        .unwrap_or(std::ptr::null_mut())
    }
}

/// Free proto data. NULL is a no-op.
///
/// # Safety
/// `proto` must be `NULL` or have come from this library.
#[no_mangle]
pub unsafe extern "C" fn ef_proto_data_free(proto: *mut EfProtoData) {
    unsafe {
        if proto.is_null() {
            return;
        }
        let _ = catch_unwind(AssertUnwindSafe(|| drop(Box::from_raw(proto))));
    }
}

/// Proto layout: 0 NHWC, 1 NCHW. `-1` for NULL.
///
/// # Safety
/// `proto` must be `NULL` or a live handle from this library.
#[no_mangle]
pub unsafe extern "C" fn ef_proto_data_layout(proto: *const EfProtoData) -> i32 {
    unsafe {
        if proto.is_null() {
            return -1;
        }
        match (*proto).layout {
            ProtoLayout::Nhwc => 0,
            ProtoLayout::Nchw => 1,
        }
    }
}

/// Take ownership of the proto tensor. NULL if already taken.
///
/// # Safety
/// `proto` must be `NULL` or a live handle from this library.
#[no_mangle]
pub unsafe extern "C" fn ef_proto_data_take_protos(proto: *mut EfProtoData) -> *mut EfTensor {
    unsafe {
        if proto.is_null() {
            return std::ptr::null_mut();
        }
        match (*proto).protos.take() {
            Some(t) => t.into_raw(),
            None => std::ptr::null_mut(),
        }
    }
}

/// Take ownership of the mask-coefficient tensor. NULL if already taken.
///
/// # Safety
/// `proto` must be `NULL` or a live handle from this library.
#[no_mangle]
pub unsafe extern "C" fn ef_proto_data_take_mask_coefficients(
    proto: *mut EfProtoData,
) -> *mut EfTensor {
    unsafe {
        if proto.is_null() {
            return std::ptr::null_mut();
        }
        match (*proto).mask_coefficients.take() {
            Some(t) => t.into_raw(),
            None => std::ptr::null_mut(),
        }
    }
}

fn borrow_outputs(
    outputs: *const *const EfTensor,
    count: usize,
) -> Result<
    (
        Vec<&'static TensorDyn>,
        Vec<std::mem::ManuallyDrop<TensorDyn>>,
    ),
    c_int,
> {
    let handles = unsafe { std::slice::from_raw_parts(outputs, count) };
    let mut borrowed = Vec::with_capacity(count);
    for h in handles {
        if h.is_null() {
            return Err(libc::EINVAL);
        }
        borrowed.push(std::mem::ManuallyDrop::new(unsafe {
            TensorDyn::from_raw(*h as *mut EfTensor)
        }));
    }
    let refs: Vec<&TensorDyn> = borrowed.iter().map(|t| &**t).collect();
    // SAFETY: the Vec is returned alongside the refs and outlives them.
    let refs: Vec<&'static TensorDyn> = unsafe { std::mem::transmute(refs) };
    Ok((refs, borrowed))
}

/// Dequantize an integer tensor into a pre-allocated f32 tensor.
///
/// # Safety
/// `input` and `output` must be live handles.
#[no_mangle]
pub unsafe extern "C" fn ef_dequantize(
    input: *const EfTensor,
    scale: f32,
    zero_point: c_int,
    output: *mut EfTensor,
) -> c_int {
    unsafe {
        catch_unwind(AssertUnwindSafe(|| {
            if input.is_null() || output.is_null() {
                return libc::EINVAL;
            }
            let quant = Quantization::new(scale, zero_point);
            TensorDyn::with_raw(input as *mut EfTensor, |inp| {
                TensorDyn::with_raw(output, |out| dequantize_dyn(inp, quant, out))
            })
        }))
        .unwrap_or(libc::EINVAL)
    }
}

fn dequantize_dyn(input: &TensorDyn, quant: Quantization, output: &mut TensorDyn) -> c_int {
    use edgefirst_tensor::{CpuAccess, DType};
    if output.dtype() != DType::F32 {
        return libc::EINVAL;
    }
    let in_pin = match input.pin_host(CpuAccess::Read) {
        Ok(p) => p,
        Err(_) => return libc::EIO,
    };
    let out_pin = match output.pin_host(CpuAccess::Write) {
        Ok(p) => p,
        Err(_) => return libc::EIO,
    };
    let in_bytes = unsafe { in_pin.as_slice() };
    let n = output.shape().iter().product::<usize>();
    let out = unsafe { std::slice::from_raw_parts_mut(out_pin.as_mut_ptr() as *mut f32, n) };
    let in_n = match input.dtype() {
        DType::U8 => in_bytes.len(),
        DType::I8 => in_bytes.len(),
        DType::U16 | DType::I16 => in_bytes.len() / 2,
        DType::U32 | DType::I32 => in_bytes.len() / 4,
        _ => return libc::EINVAL,
    };
    if in_n != n {
        return libc::EINVAL;
    }
    match input.dtype() {
        DType::U8 => dequantize_cpu_chunked(in_bytes, quant, out),
        DType::I8 => {
            let s = unsafe { std::slice::from_raw_parts(in_bytes.as_ptr() as *const i8, n) };
            dequantize_cpu_chunked(s, quant, out);
        }
        DType::U16 => {
            let s = unsafe { std::slice::from_raw_parts(in_bytes.as_ptr() as *const u16, n) };
            dequantize_cpu_chunked(s, quant, out);
        }
        DType::I16 => {
            let s = unsafe { std::slice::from_raw_parts(in_bytes.as_ptr() as *const i16, n) };
            dequantize_cpu_chunked(s, quant, out);
        }
        DType::U32 => {
            let s = unsafe { std::slice::from_raw_parts(in_bytes.as_ptr() as *const u32, n) };
            dequantize_cpu_chunked(s, quant, out);
        }
        DType::I32 => {
            let s = unsafe { std::slice::from_raw_parts(in_bytes.as_ptr() as *const i32, n) };
            dequantize_cpu_chunked(s, quant, out);
        }
        _ => return libc::EINVAL,
    }
    0
}

/// Convert segmentation `index` to a new `[H, W]` u8 tensor.
///
/// # Safety
/// `list` must be a live handle from this library.
#[no_mangle]
pub unsafe extern "C" fn ef_segmentation_to_mask(
    list: *const EfSegmentationList,
    index: usize,
) -> *mut EfTensor {
    unsafe {
        catch_unwind(AssertUnwindSafe(|| {
            let masks = &(*list).masks;
            if index >= masks.len() {
                return std::ptr::null_mut();
            }
            let seg = &masks[index];
            let pin = match seg.segmentation.pin_host(edgefirst_tensor::CpuAccess::Read) {
                Ok(p) => p,
                Err(_) => return std::ptr::null_mut(),
            };
            let shape = seg.segmentation.shape();
            if shape.len() < 2 {
                return std::ptr::null_mut();
            }
            let h = shape[0];
            let w = shape[1];
            let c = if shape.len() >= 3 { shape[2] } else { 1 };
            if c == 0 {
                return std::ptr::null_mut();
            }
            let bytes = pin.as_slice();
            let mask = match mask_from_bytes(bytes, h, w, c) {
                Some(m) => m,
                None => return std::ptr::null_mut(),
            };
            let out = match TensorDyn::new(
                &[h, w],
                edgefirst_tensor::DType::U8,
                Some(edgefirst_tensor::TensorMemory::Mem),
                None,
            ) {
                Ok(t) => t,
                Err(_) => return std::ptr::null_mut(),
            };
            if let Ok(out_pin) = out.pin_host(edgefirst_tensor::CpuAccess::Write) {
                let dst = std::slice::from_raw_parts_mut(out_pin.as_mut_ptr(), mask.len());
                dst.copy_from_slice(&mask);
            }
            out.into_raw()
        }))
        .unwrap_or(std::ptr::null_mut())
    }
}

fn mask_from_bytes(bytes: &[u8], h: usize, w: usize, c: usize) -> Option<Vec<u8>> {
    let need = h.checked_mul(w)?.checked_mul(c)?;
    if bytes.len() < need {
        return None;
    }
    let mut out = vec![0u8; h * w];
    if c == 1 {
        for (slot, &b) in out.iter_mut().zip(bytes.iter()) {
            *slot = u8::from(b >= 128);
        }
    } else {
        for (i, slot) in out.iter_mut().enumerate() {
            let base = i * c;
            let mut best = 0usize;
            let mut best_v = bytes[base];
            for k in 1..c {
                let v = bytes[base + k];
                if v >= best_v {
                    best_v = v;
                    best = k;
                }
            }
            *slot = best as u8;
        }
    }
    Some(out)
}

/// Decoder-local ByteTrack over `DetectBox`.
pub struct EfDecoderTracker {
    inner: edgefirst_tracker::ByteTrack<DetectBox>,
}

/// One track written by [`ef_decoder_decode_tracked`].
#[repr(C)]
#[derive(Clone, Copy)]
pub struct EfDecoderTrack {
    pub uuid: [u8; 16],
    pub xmin: f32,
    pub ymin: f32,
    pub xmax: f32,
    pub ymax: f32,
    pub count: i32,
    pub created: u64,
    pub last_updated: u64,
}

pub struct EfDecoderTrackList {
    tracks: Vec<EfDecoderTrack>,
}

#[no_mangle]
pub extern "C" fn ef_decoder_tracker_new() -> *mut EfDecoderTracker {
    catch_unwind(|| {
        Box::into_raw(Box::new(EfDecoderTracker {
            inner: edgefirst_tracker::ByteTrack::default(),
        }))
    })
    .unwrap_or(std::ptr::null_mut())
}

/// Free a decoder-local tracker. NULL is a no-op.
///
/// # Safety
/// `t` must be `NULL` or have come from [`ef_decoder_tracker_new`].
#[no_mangle]
pub unsafe extern "C" fn ef_decoder_tracker_free(t: *mut EfDecoderTracker) {
    unsafe {
        if t.is_null() {
            return;
        }
        let _ = catch_unwind(AssertUnwindSafe(|| drop(Box::from_raw(t))));
    }
}

/// Number of tracks. Zero for NULL.
///
/// # Safety
/// `l` must be `NULL` or a live handle from this library.
#[no_mangle]
pub unsafe extern "C" fn ef_decoder_track_list_len(l: *const EfDecoderTrackList) -> usize {
    unsafe {
        if l.is_null() {
            return 0;
        }
        catch_unwind(AssertUnwindSafe(|| (*l).tracks.len())).unwrap_or(0)
    }
}

/// Copy track `index` into `out`. Returns 0 on success.
///
/// # Safety
/// `l` and `out` must be live or NULL as documented.
#[no_mangle]
pub unsafe extern "C" fn ef_decoder_track_list_get(
    l: *const EfDecoderTrackList,
    index: usize,
    out: *mut EfDecoderTrack,
) -> c_int {
    unsafe {
        catch_unwind(AssertUnwindSafe(|| {
            if l.is_null() || out.is_null() {
                return libc::EINVAL;
            }
            let tracks = &(*l).tracks;
            match tracks.get(index) {
                Some(t) => {
                    *out = *t;
                    0
                }
                None => libc::EINVAL,
            }
        }))
        .unwrap_or(libc::EINVAL)
    }
}

/// Free a decoder track list. NULL is a no-op.
///
/// # Safety
/// `l` must be `NULL` or have come from this library.
#[no_mangle]
pub unsafe extern "C" fn ef_decoder_track_list_free(l: *mut EfDecoderTrackList) {
    unsafe {
        if l.is_null() {
            return;
        }
        let _ = catch_unwind(AssertUnwindSafe(|| drop(Box::from_raw(l))));
    }
}

/// Decode and update a decoder-owned tracker.
///
/// # Safety
/// `outputs` must point to `count` live handles.
#[no_mangle]
pub unsafe extern "C" fn ef_decoder_decode_tracked(
    d: *const EfDecoder,
    tracker: *mut EfDecoderTracker,
    timestamp: u64,
    outputs: *const *const EfTensor,
    count: usize,
    out_boxes: *mut *mut EfDetectBoxList,
    out_masks: *mut *mut EfSegmentationList,
    out_tracks: *mut *mut EfDecoderTrackList,
) -> c_int {
    unsafe {
        catch_unwind(AssertUnwindSafe(|| {
            if d.is_null()
                || tracker.is_null()
                || outputs.is_null()
                || out_boxes.is_null()
                || count == 0
            {
                return libc::EINVAL;
            }
            let Ok((refs, _keep)) = borrow_outputs(outputs, count) else {
                return libc::EINVAL;
            };
            let mut boxes: Vec<DetectBox> = Vec::new();
            let mut masks: Vec<Segmentation> = Vec::new();
            let mut tracks: Vec<edgefirst_tracker::TrackInfo> = Vec::new();
            if (*d)
                .inner
                .decode_tracked(
                    &mut (*tracker).inner,
                    timestamp,
                    &refs,
                    &mut boxes,
                    &mut masks,
                    &mut tracks,
                )
                .is_err()
            {
                return libc::EBADMSG;
            }
            if !out_masks.is_null() {
                *out_masks = Box::into_raw(Box::new(EfSegmentationList {
                    masks,
                    views: None,
                    pins: Vec::new(),
                }));
            }
            if !out_tracks.is_null() {
                *out_tracks = Box::into_raw(Box::new(EfDecoderTrackList {
                    tracks: tracks.iter().map(to_c_track).collect(),
                }));
            }
            *out_boxes = Box::into_raw(Box::new(EfDetectBoxList {
                boxes: boxes.iter().map(to_c_box).collect(),
            }));
            0
        }))
        .unwrap_or(libc::EINVAL)
    }
}

fn to_c_track(t: &edgefirst_tracker::TrackInfo) -> EfDecoderTrack {
    let [xmin, ymin, xmax, ymax] = t.tracked_location;
    EfDecoderTrack {
        uuid: *t.uuid.as_bytes(),
        xmin,
        ymin,
        xmax,
        ymax,
        count: t.count,
        created: t.created,
        last_updated: t.last_updated,
    }
}

/// Free a mask list. Freeing `NULL` is a no-op.
///
/// # Safety
/// `l` must be `NULL` or have come from this library.
#[no_mangle]
pub unsafe extern "C" fn ef_segmentation_list_free(l: *mut EfSegmentationList) {
    unsafe {
        if l.is_null() {
            return;
        }
        let _ = catch_unwind(AssertUnwindSafe(|| drop(Box::from_raw(l))));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn params_can_be_created_and_freed() {
        unsafe {
            let p = ef_decoder_params_new();
            assert!(!p.is_null());
            ef_decoder_params_free(p);
            ef_decoder_params_free(std::ptr::null_mut());
        }
    }

    #[test]
    fn every_scalar_setter_rejects_a_null_handle() {
        unsafe {
            let n = std::ptr::null_mut();
            assert_eq!(ef_decoder_params_set_score_threshold(n, 0.5), libc::EINVAL);
            assert_eq!(ef_decoder_params_set_iou_threshold(n, 0.5), libc::EINVAL);
            assert_eq!(ef_decoder_params_set_pre_nms_top_k(n, 10), libc::EINVAL);
            assert_eq!(ef_decoder_params_set_max_det(n, 10), libc::EINVAL);
            assert_eq!(ef_decoder_params_set_input_dims(n, 640, 640), libc::EINVAL);
            assert_eq!(ef_decoder_params_set_nms(n, 1), libc::EINVAL);
        }
    }

    #[test]
    fn an_unknown_nms_mode_is_refused() {
        unsafe {
            let p = ef_decoder_params_new();
            assert_eq!(ef_decoder_params_set_nms(p, 0), 0);
            assert_eq!(ef_decoder_params_set_nms(p, 3), 0);
            assert_eq!(ef_decoder_params_set_nms(p, 99), libc::EINVAL);
            ef_decoder_params_free(p);
        }
    }

    #[test]
    fn zero_input_dims_are_refused() {
        unsafe {
            let p = ef_decoder_params_new();
            assert_eq!(ef_decoder_params_set_input_dims(p, 0, 640), libc::EINVAL);
            assert_eq!(ef_decoder_params_set_input_dims(p, 640, 0), libc::EINVAL);
            assert_eq!(ef_decoder_params_set_input_dims(p, 640, 640), 0);
            ef_decoder_params_free(p);
        }
    }

    #[test]
    fn exactly_one_configuration_source_is_required() {
        unsafe {
            // None.
            let p = ef_decoder_params_new();
            assert!(ef_decoder_new(p).is_null(), "no config must not build");

            // Two. Rejected rather than resolved by a precedence rule, since
            // two sources disagreeing has no defined answer.
            let j = std::ffi::CString::new("{}").unwrap();
            let y = std::ffi::CString::new("a: 1").unwrap();
            ef_decoder_params_set_config_json(p, j.as_ptr(), 0);
            ef_decoder_params_set_config_yaml(p, y.as_ptr(), 0);
            assert!(ef_decoder_new(p).is_null(), "two configs must not build");
            ef_decoder_params_free(p);
        }
    }

    #[test]
    fn a_config_string_can_be_length_delimited_or_nul_terminated() {
        unsafe {
            let p = ef_decoder_params_new();
            let json = b"{\"x\":1}extra";
            // len given: the trailing bytes must be ignored.
            assert_eq!(
                ef_decoder_params_set_config_json(p, json.as_ptr() as *const c_char, 7),
                0
            );
            assert_eq!((*p).config_json.as_deref(), Some("{\"x\":1}"));
            ef_decoder_params_free(p);
        }
    }

    #[test]
    fn an_empty_mask_list_borrows_null_rather_than_a_dangling_pointer() {
        unsafe {
            let mut l = EfSegmentationList {
                masks: Vec::new(),
                views: None,
                pins: Vec::new(),
            };
            assert!(ef_segmentation_list_data(&mut l).is_null());
            assert!(ef_segmentation_list_data(std::ptr::null_mut()).is_null());
        }
    }

    #[test]
    fn empty_result_lists_behave() {
        unsafe {
            assert_eq!(ef_detect_box_list_len(std::ptr::null()), 0);
            assert!(ef_detect_box_list_data(std::ptr::null()).is_null());
            assert_eq!(ef_segmentation_list_len(std::ptr::null()), 0);
            let mut f = 0f32;
            assert_eq!(
                ef_segmentation_list_get_bbox(std::ptr::null(), 0, &mut f, &mut f, &mut f, &mut f),
                libc::EINVAL
            );
            ef_detect_box_list_free(std::ptr::null_mut());
            ef_segmentation_list_free(std::ptr::null_mut());
        }
    }

    #[test]
    fn a_list_round_trips_a_detection() {
        unsafe {
            let l = ef_detect_box_list_new();
            assert!(!l.is_null());
            assert_eq!(ef_detect_box_list_len(l), 0);
            let b = EfDetectBox {
                xmin: 0.1,
                ymin: 0.2,
                xmax: 0.3,
                ymax: 0.4,
                score: 0.9,
                label: 7,
            };
            assert_eq!(ef_detect_box_list_push(l, &b), 0);
            assert_eq!(ef_detect_box_list_len(l), 1);
            let mut got = EfDetectBox::default();
            assert_eq!(ef_detect_box_list_get(l, 0, &mut got), 0);
            assert_eq!(got, b, "every field must survive, not just the box");
            ef_detect_box_list_free(l);
        }
    }

    #[test]
    fn an_out_of_range_index_is_an_error_not_a_read() {
        unsafe {
            let l = ef_detect_box_list_new();
            let mut got = EfDetectBox::default();
            assert_eq!(ef_detect_box_list_get(l, 0, &mut got), libc::EINVAL);
            assert_eq!(
                ef_detect_box_list_get(l, usize::MAX, &mut got),
                libc::EINVAL
            );
            ef_detect_box_list_free(l);
        }
    }

    #[test]
    fn null_arguments_to_the_list_family_are_errors_not_crashes() {
        unsafe {
            let b = EfDetectBox::default();
            let mut out = EfDetectBox::default();
            assert_eq!(
                ef_detect_box_list_push(std::ptr::null_mut(), &b),
                libc::EINVAL
            );
            assert_eq!(ef_detect_box_list_len(std::ptr::null()), 0);
            assert_eq!(
                ef_detect_box_list_get(std::ptr::null(), 0, &mut out),
                libc::EINVAL
            );
            let l = ef_detect_box_list_new();
            assert_eq!(ef_detect_box_list_push(l, std::ptr::null()), libc::EINVAL);
            assert_eq!(
                ef_detect_box_list_get(l, 0, std::ptr::null_mut()),
                libc::EINVAL
            );
            ef_detect_box_list_free(l);
            ef_detect_box_list_free(std::ptr::null_mut());
        }
    }

    #[test]
    fn data_and_len_describe_the_same_elements() {
        // The borrowed-array route another library takes must agree with the
        // copying accessor, or the two views of one list disagree.
        unsafe {
            let l = ef_detect_box_list_new();
            for i in 0..3u32 {
                let b = EfDetectBox {
                    xmin: i as f32,
                    label: i,
                    ..Default::default()
                };
                ef_detect_box_list_push(l, &b);
            }
            let n = ef_detect_box_list_len(l);
            let data = ef_detect_box_list_data(l);
            assert_eq!(n, 3);
            assert!(!data.is_null());
            for i in 0..n {
                let mut copied = EfDetectBox::default();
                ef_detect_box_list_get(l, i, &mut copied);
                assert_eq!(*data.add(i), copied, "element {i} disagrees");
            }
            ef_detect_box_list_free(l);
        }
    }

    #[test]
    fn an_empty_list_borrows_null_rather_than_a_dangling_pointer() {
        unsafe {
            let l = ef_detect_box_list_new();
            assert!(ef_detect_box_list_data(l).is_null());
            assert!(ef_detect_box_list_data(std::ptr::null()).is_null());
            ef_detect_box_list_free(l);
        }
    }

    #[test]
    fn decode_rejects_bad_arguments_rather_than_reading() {
        unsafe {
            let mut boxes: *mut EfDetectBoxList = std::ptr::null_mut();
            assert_eq!(
                ef_decoder_decode(
                    std::ptr::null(),
                    std::ptr::null(),
                    0,
                    &mut boxes,
                    std::ptr::null_mut()
                ),
                libc::EINVAL
            );
        }
    }

    #[test]
    fn a_freed_string_entry_point_exists_for_rust_allocations() {
        unsafe {
            ef_decoder_string_free(std::ptr::null_mut());
            let s = std::ffi::CString::new("model").unwrap().into_raw();
            ef_decoder_string_free(s);
        }
    }
}
