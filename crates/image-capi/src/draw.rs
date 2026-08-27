// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! Draw and materialize masks from boxes + mask bytes or proto tensors.

use std::ffi::c_int;
use std::mem::ManuallyDrop;
use std::panic::{catch_unwind, AssertUnwindSafe};

use edgefirst_decoder_abi::{EfDetectBox, EfSegmentation};
use edgefirst_image::{ColorMode, ImageProcessorTrait, MaskOverlay, MaskResolution};
use edgefirst_tensor::{
    BoundingBox, CpuAccess, DType, DetectBox, ProtoData, ProtoLayout, Segmentation, TensorDyn,
};
use edgefirst_tensor_ffi::EfTensor;

use crate::processor::{with_tensor, with_tensor_mut, EfImageProcessor};

fn color_mode_from(code: u32) -> ColorMode {
    match code {
        1 => ColorMode::Instance,
        2 => ColorMode::Track,
        _ => ColorMode::Class,
    }
}

unsafe fn letterbox_from(ptr: *const f32) -> Option<[f32; 4]> {
    if ptr.is_null() {
        None
    } else {
        let s = unsafe { std::slice::from_raw_parts(ptr, 4) };
        Some([s[0], s[1], s[2], s[3]])
    }
}

fn boxes_from(ptr: *const EfDetectBox, n: usize) -> Vec<DetectBox> {
    if ptr.is_null() || n == 0 {
        return Vec::new();
    }
    unsafe { std::slice::from_raw_parts(ptr, n) }
        .iter()
        .map(|b| DetectBox {
            bbox: BoundingBox {
                xmin: b.xmin,
                ymin: b.ymin,
                xmax: b.xmax,
                ymax: b.ymax,
            },
            score: b.score,
            label: b.label as usize,
        })
        .collect()
}

unsafe fn segs_from(ptr: *const EfSegmentation, n: usize) -> Result<Vec<Segmentation>, c_int> {
    if ptr.is_null() || n == 0 {
        return Ok(Vec::new());
    }
    let raw = unsafe { std::slice::from_raw_parts(ptr, n) };
    let mut out = Vec::with_capacity(n);
    for s in raw {
        if s.mask.is_null() || s.width == 0 || s.height == 0 {
            return Err(libc::EINVAL);
        }
        let h = s.height as usize;
        let w = s.width as usize;
        let cap = h.saturating_mul(w);
        let td = unsafe {
            TensorDyn::from_raw_host_with_capacity(
                s.mask as *mut u8,
                &[h, w, 1],
                cap,
                DType::U8,
                None,
            )
        }
        .map_err(|_| libc::EINVAL)?;
        out.push(Segmentation {
            xmin: s.xmin,
            ymin: s.ymin,
            xmax: s.xmax,
            ymax: s.ymax,
            segmentation: td,
        });
    }
    Ok(out)
}

unsafe fn proto_from(
    protos: *mut EfTensor,
    coeffs: *mut EfTensor,
    layout: u32,
) -> Result<ManuallyDrop<ProtoData>, c_int> {
    if protos.is_null() || coeffs.is_null() {
        return Err(libc::EINVAL);
    }
    let layout = match layout {
        0 => ProtoLayout::Nhwc,
        1 => ProtoLayout::Nchw,
        _ => return Err(libc::EINVAL),
    };
    Ok(ManuallyDrop::new(ProtoData {
        protos: unsafe { TensorDyn::from_raw(protos) },
        mask_coefficients: unsafe { TensorDyn::from_raw(coeffs) },
        layout,
    }))
}

fn draw_with_overlay<F>(
    p: *mut EfImageProcessor,
    dst: *mut EfTensor,
    background: *const EfTensor,
    opacity: f32,
    letterbox: Option<[f32; 4]>,
    color_mode: ColorMode,
    body: F,
) -> c_int
where
    F: Fn(
        &mut edgefirst_image::ImageProcessor,
        &mut TensorDyn,
        MaskOverlay<'_>,
    ) -> Result<(), edgefirst_image::Error>,
{
    let run =
        |proc: &mut edgefirst_image::ImageProcessor, d: &mut TensorDyn, bg: Option<&TensorDyn>| {
            let overlay = MaskOverlay {
                background: bg,
                opacity: opacity.clamp(0.0, 1.0),
                letterbox,
                color_mode,
            };
            body(proc, d, overlay).map_err(|e| draw_err(&e))
        };
    unsafe {
        if background.is_null() {
            match with_tensor_mut(dst, |d| run(&mut (*p).inner, d, None)) {
                Ok(Ok(())) => 0,
                Ok(Err(e)) | Err(e) => e,
            }
        } else {
            match with_tensor(background, |bg| {
                with_tensor_mut(dst, |d| run(&mut (*p).inner, d, Some(bg)))
            }) {
                Ok(Ok(Ok(()))) => 0,
                Ok(Ok(Err(e))) => e,
                Ok(Err(e)) | Err(e) => e,
            }
        }
    }
}

fn draw_err(e: &edgefirst_image::Error) -> c_int {
    match e {
        edgefirst_image::Error::AliasedBuffers(_) => libc::EINVAL,
        _ => libc::EIO,
    }
}

/// Draw boxes and decoded masks onto `dst`.
///
/// # Safety
/// Pointers must be live or NULL as documented.
#[no_mangle]
pub unsafe extern "C" fn ef_image_processor_draw_decoded_masks(
    p: *mut EfImageProcessor,
    dst: *mut EfTensor,
    boxes: *const EfDetectBox,
    n_boxes: usize,
    masks: *const EfSegmentation,
    n_masks: usize,
    background: *const EfTensor,
    opacity: f32,
    letterbox: *const f32,
    color_mode: u32,
) -> c_int {
    unsafe {
        catch_unwind(AssertUnwindSafe(|| {
            if p.is_null() || dst.is_null() {
                return libc::EINVAL;
            }
            if !background.is_null() && std::ptr::eq(background, dst as *const _) {
                return libc::EINVAL;
            }
            let detect = boxes_from(boxes, n_boxes);
            let segs = match segs_from(masks, n_masks) {
                Ok(s) => s,
                Err(e) => return e,
            };
            let lb = letterbox_from(letterbox);
            let mode = color_mode_from(color_mode);
            draw_with_overlay(p, dst, background, opacity, lb, mode, |proc, d, overlay| {
                proc.draw_decoded_masks(d, &detect, &segs, overlay)
            })
        }))
        .unwrap_or(libc::EINVAL)
    }
}

/// Draw proto masks onto `dst`. `protos` and `coeffs` are borrowed, not taken.
///
/// # Safety
/// Tensor handles must stay live for the call.
#[no_mangle]
pub unsafe extern "C" fn ef_image_processor_draw_proto_masks(
    p: *mut EfImageProcessor,
    dst: *mut EfTensor,
    boxes: *const EfDetectBox,
    n_boxes: usize,
    protos: *mut EfTensor,
    coeffs: *mut EfTensor,
    layout: u32,
    background: *const EfTensor,
    opacity: f32,
    letterbox: *const f32,
    color_mode: u32,
) -> c_int {
    unsafe {
        catch_unwind(AssertUnwindSafe(|| {
            if p.is_null() || dst.is_null() {
                return libc::EINVAL;
            }
            if !background.is_null() && std::ptr::eq(background, dst as *const _) {
                return libc::EINVAL;
            }
            let detect = boxes_from(boxes, n_boxes);
            let proto = match proto_from(protos, coeffs, layout) {
                Ok(p) => p,
                Err(e) => return e,
            };
            let lb = letterbox_from(letterbox);
            let mode = color_mode_from(color_mode);
            draw_with_overlay(p, dst, background, opacity, lb, mode, |proc, d, overlay| {
                proc.draw_proto_masks(d, &detect, &proto, overlay)
            })
        }))
        .unwrap_or(libc::EINVAL)
    }
}

/// Owned list of materialized masks.
pub struct EfMaskList {
    masks: Vec<Segmentation>,
    views: Option<Vec<EfSegmentation>>,
    pins: Vec<edgefirst_tensor::HostPin<'static>>,
}

/// Materialize per-instance masks from proto tensors. Caller frees the list.
///
/// # Safety
/// Handles must be live.
#[no_mangle]
pub unsafe extern "C" fn ef_image_processor_materialize_masks(
    p: *mut EfImageProcessor,
    boxes: *const EfDetectBox,
    n_boxes: usize,
    protos: *mut EfTensor,
    coeffs: *mut EfTensor,
    layout: u32,
    letterbox: *const f32,
) -> *mut EfMaskList {
    unsafe {
        catch_unwind(AssertUnwindSafe(|| {
            if p.is_null() || boxes.is_null() {
                return std::ptr::null_mut();
            }
            let detect = boxes_from(boxes, n_boxes);
            let proto = match proto_from(protos, coeffs, layout) {
                Ok(p) => p,
                Err(_) => return std::ptr::null_mut(),
            };
            let lb = letterbox_from(letterbox);
            let masks = (*p)
                .inner
                .materialize_masks(&detect, &proto, lb, MaskResolution::Proto);
            match masks {
                Ok(masks) => Box::into_raw(Box::new(EfMaskList {
                    masks,
                    views: None,
                    pins: Vec::new(),
                })),
                Err(_) => std::ptr::null_mut(),
            }
        }))
        .unwrap_or(std::ptr::null_mut())
    }
}

/// Number of masks. Zero for NULL.
///
/// # Safety
/// `l` must be `NULL` or a live handle from this library.
#[no_mangle]
pub unsafe extern "C" fn ef_mask_list_len(l: *const EfMaskList) -> usize {
    unsafe {
        if l.is_null() {
            return 0;
        }
        catch_unwind(AssertUnwindSafe(|| (*l).masks.len())).unwrap_or(0)
    }
}

/// Borrow masks as `ef_segmentation` values. Valid until the list is freed.
///
/// # Safety
/// `l` must be `NULL` or a live handle from this library.
#[no_mangle]
pub unsafe extern "C" fn ef_mask_list_data(l: *mut EfMaskList) -> *const EfSegmentation {
    unsafe {
        if l.is_null() {
            return std::ptr::null();
        }
        catch_unwind(AssertUnwindSafe(|| {
            let list = &mut *l;
            if list.views.is_none() {
                let mut views = Vec::with_capacity(list.masks.len());
                for m in &list.masks {
                    let Ok(pin) = m.segmentation.pin_host(CpuAccess::Read) else {
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

/// Free a mask list. NULL is a no-op.
///
/// # Safety
/// `l` must be `NULL` or have come from this library.
#[no_mangle]
pub unsafe extern "C" fn ef_mask_list_free(l: *mut EfMaskList) {
    unsafe {
        if l.is_null() {
            return;
        }
        let _ = catch_unwind(AssertUnwindSafe(|| drop(Box::from_raw(l))));
    }
}
