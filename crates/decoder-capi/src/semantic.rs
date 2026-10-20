// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! Semantic segmentation decode: a class map written into a caller tensor.

use std::ffi::c_int;
use std::mem::ManuallyDrop;
use std::panic::{catch_unwind, AssertUnwindSafe};

use edgefirst_decoder::schema::DecodeMode;
use edgefirst_decoder::semantic::SemanticOptions;
use edgefirst_tensor::TensorDyn;
use edgefirst_tensor_ffi::EfTensor;

use crate::decode::EfDecoder;

/// Runtime options of a semantic decode. Build with
/// [`ef_semantic_options_new`], free with [`ef_semantic_options_free`].
pub struct EfSemanticOptions {
    inner: SemanticOptions,
}

/// New default options: the model's decode mode, no letterbox. `NULL` on
/// allocation failure.
#[no_mangle]
pub extern "C" fn ef_semantic_options_new() -> *mut EfSemanticOptions {
    catch_unwind(|| {
        Box::into_raw(Box::new(EfSemanticOptions {
            inner: SemanticOptions::new(),
        }))
    })
    .unwrap_or(std::ptr::null_mut())
}

/// Free options from [`ef_semantic_options_new`]. `NULL` is a no-op.
///
/// # Safety
/// `o` must be `NULL` or a live handle, not used afterwards.
#[no_mangle]
pub unsafe extern "C" fn ef_semantic_options_free(o: *mut EfSemanticOptions) {
    if !o.is_null() {
        // SAFETY: the caller's contract; the handle came from `Box::into_raw`.
        let _ = catch_unwind(AssertUnwindSafe(|| drop(unsafe { Box::from_raw(o) })));
    }
}

/// Override the model's decode mode: an `EF_DECODE_MODE_*` code.
///
/// @return 0, or `EINVAL` for an unknown code or `NULL`.
///
/// # Safety
/// `o` must be `NULL` or a live handle.
#[no_mangle]
pub unsafe extern "C" fn ef_semantic_options_set_mode(
    o: *mut EfSemanticOptions,
    mode: u32,
) -> c_int {
    // SAFETY: the caller's contract.
    let (Some(o), Some(mode)) = (unsafe { o.as_mut() }, DecodeMode::from_code(mode)) else {
        return libc::EINVAL;
    };
    o.inner = o.inner.with_mode(mode);
    0
}

/// Image content within the model input as normalized
/// `[xmin, ymin, xmax, ymax]`, as produced by a letterbox resize. `NULL`
/// clears it: the image fills the input.
///
/// @return 0, or `EINVAL` for a `NULL` handle.
///
/// # Safety
/// `o` must be `NULL` or a live handle; `letterbox` `NULL` or four floats.
#[no_mangle]
pub unsafe extern "C" fn ef_semantic_options_set_letterbox(
    o: *mut EfSemanticOptions,
    letterbox: *const f32,
) -> c_int {
    // SAFETY: the caller's contract.
    let Some(o) = (unsafe { o.as_mut() }) else {
        return libc::EINVAL;
    };
    // SAFETY: the caller's contract.
    o.inner.letterbox = unsafe { letterbox.cast::<[f32; 4]>().as_ref() }.copied();
    0
}

/// Number of classes of the model's semantic output: the channel count, 2
/// for a single thresholded channel, 0 for a class map.
///
/// @return 0, `ENOENT` when the model has no semantic output, or `EINVAL`
///         for a `NULL` argument.
///
/// # Safety
/// `d` must be `NULL` or live; `out` `NULL` or writable.
#[no_mangle]
pub unsafe extern "C" fn ef_decoder_semantic_classes(
    d: *const EfDecoder,
    out: *mut usize,
) -> c_int {
    // SAFETY: the caller's contract.
    let (Some(d), false) = (unsafe { d.as_ref() }, out.is_null()) else {
        return libc::EINVAL;
    };
    match d.inner.semantic() {
        Some(s) => {
            // SAFETY: checked non-null; the caller's contract.
            unsafe { *out = s.classes() };
            0
        }
        None => libc::ENOENT,
    }
}

/// Decode the model's semantic segmentation output into `dst`, an `[H, W]`
/// class map of `EF_DTYPE_U8` (up to 256 classes) or `EF_DTYPE_U16`, at any
/// resolution. Each output pixel is mapped through the letterbox region, so
/// the padding is cropped and `dst` covers the original image.
///
/// `confidence`, when not `NULL`, receives the winning class's probability
/// as an `[H, W]` map of `EF_DTYPE_F32` or `EF_DTYPE_U8` (scaled to 255),
/// computed at the model's resolution and upsampled: approximate, for
/// overlays. `options` may be `NULL` for the defaults.
///
/// @return 0, `ENOENT` when the model has no semantic output, `EBADMSG` when
///         the tensors do not match it or cannot hold the result, or
///         `EINVAL` for a `NULL` argument.
///
/// # Safety
/// `outputs` must point to `count` live tensor handles; `dst` and
/// `confidence` (when not `NULL`) must be live handles distinct from them
/// and from each other; `options` `NULL` or live.
#[no_mangle]
pub unsafe extern "C" fn ef_decoder_decode_semantic(
    d: *const EfDecoder,
    outputs: *const *const EfTensor,
    count: usize,
    dst: *mut EfTensor,
    confidence: *mut EfTensor,
    options: *const EfSemanticOptions,
) -> c_int {
    catch_unwind(AssertUnwindSafe(|| {
        // SAFETY: the caller's contract.
        let Some(d) = (unsafe { d.as_ref() }) else {
            return libc::EINVAL;
        };
        if outputs.is_null() || count == 0 || dst.is_null() {
            return libc::EINVAL;
        }
        if d.inner.semantic().is_none() {
            return libc::ENOENT;
        }
        // SAFETY: the caller's contract.
        let handles = unsafe { std::slice::from_raw_parts(outputs, count) };
        if handles.iter().any(|h| h.is_null()) {
            return libc::EINVAL;
        }
        // Borrowed handles: `ManuallyDrop` keeps `TensorDyn`'s `Drop` from
        // freeing tensors the caller still owns.
        let borrowed: Vec<ManuallyDrop<TensorDyn>> = handles
            .iter()
            // SAFETY: each handle is a live `ef_tensor`; the caller's contract.
            .map(|&h| ManuallyDrop::new(unsafe { TensorDyn::from_raw(h as *mut EfTensor) }))
            .collect();
        let refs: Vec<&TensorDyn> = borrowed.iter().map(|t| &**t).collect();
        // SAFETY: live, distinct handles; the caller's contract.
        let mut dst = ManuallyDrop::new(unsafe { TensorDyn::from_raw(dst) });
        let mut confidence = (!confidence.is_null())
            // SAFETY: as for `dst`.
            .then(|| ManuallyDrop::new(unsafe { TensorDyn::from_raw(confidence) }));
        // SAFETY: the caller's contract.
        let options = unsafe { options.as_ref() }
            .map(|o| o.inner)
            .unwrap_or_default();
        match d
            .inner
            .decode_semantic(&refs, &mut dst, confidence.as_deref_mut(), &options)
        {
            Ok(()) => 0,
            Err(_) => libc::EBADMSG,
        }
    }))
    .unwrap_or(libc::EINVAL)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::decode::{
        ef_decoder_free, ef_decoder_new, ef_decoder_params_free, ef_decoder_params_new,
        ef_decoder_params_set_config_json,
    };
    use edgefirst_tensor::{CpuAccess, DType, TensorMemory};

    const SCHEMA: &str = r#"{
      "schema_version": 2,
      "outputs": [{
        "name": "logits", "type": "segmentation", "decoder": "semantic",
        "shape": [1, 3, 2, 3],
        "dshape": [{"batch": 1}, {"num_classes": 3}, {"height": 2}, {"width": 3}],
        "dtype": "float32", "decode_mode": "fast"
      }]
    }"#;

    fn decoder(json: &str) -> *mut EfDecoder {
        let j = std::ffi::CString::new(json).unwrap();
        unsafe {
            let p = ef_decoder_params_new();
            assert_eq!(ef_decoder_params_set_config_json(p, j.as_ptr(), 0), 0);
            let d = ef_decoder_new(p);
            ef_decoder_params_free(p);
            assert!(!d.is_null());
            d
        }
    }

    fn tensor(shape: &[usize], dtype: DType) -> TensorDyn {
        TensorDyn::new(shape, dtype, Some(TensorMemory::Mem), None).unwrap()
    }

    /// Class `x` wins column `x` of the 2x3 score map.
    fn diagonal_scores() -> TensorDyn {
        let t = tensor(&[1, 3, 2, 3], DType::F32);
        let pin = t.pin_host(CpuAccess::Write).unwrap();
        // SAFETY: the pin covers the tensor's 18 f32 elements.
        let v = unsafe { std::slice::from_raw_parts_mut(pin.as_mut_ptr().cast::<f32>(), 18) };
        v.fill(0.0);
        for k in 0..3 {
            for y in 0..2 {
                v[k * 6 + y * 3 + k] = 1.0;
            }
        }
        drop(pin);
        t
    }

    fn bytes(t: &TensorDyn, n: usize) -> Vec<u8> {
        let pin = t.pin_host(CpuAccess::Read).unwrap();
        // SAFETY: the pin covers at least `n` bytes.
        unsafe { std::slice::from_raw_parts(pin.as_mut_ptr(), n) }.to_vec()
    }

    #[test]
    fn options_reject_null_and_unknown_codes() {
        unsafe {
            let o = ef_semantic_options_new();
            assert!(!o.is_null());
            assert_eq!(ef_semantic_options_set_mode(o, 1), 0);
            assert_eq!(ef_semantic_options_set_mode(o, 2), libc::EINVAL);
            assert_eq!(
                ef_semantic_options_set_mode(std::ptr::null_mut(), 0),
                libc::EINVAL
            );
            let lb = [0.0f32, 0.0, 1.0, 0.5];
            assert_eq!(ef_semantic_options_set_letterbox(o, lb.as_ptr()), 0);
            assert_eq!(ef_semantic_options_set_letterbox(o, std::ptr::null()), 0);
            assert_eq!(
                ef_semantic_options_set_letterbox(std::ptr::null_mut(), lb.as_ptr()),
                libc::EINVAL
            );
            ef_semantic_options_free(o);
            ef_semantic_options_free(std::ptr::null_mut());
        }
    }

    #[test]
    fn decodes_into_caller_tensors() {
        unsafe {
            let d = decoder(SCHEMA);
            let mut classes = 0usize;
            assert_eq!(ef_decoder_semantic_classes(d, &mut classes), 0);
            assert_eq!(classes, 3);

            let scores = diagonal_scores().into_raw();
            let outputs = [scores as *const EfTensor];
            let dst = tensor(&[2, 6], DType::U8).into_raw();
            let conf = tensor(&[2, 6], DType::U8).into_raw();
            let o = ef_semantic_options_new();
            assert_eq!(
                ef_decoder_decode_semantic(d, outputs.as_ptr(), 1, dst, conf, o),
                0
            );
            let dst = TensorDyn::from_raw(dst);
            assert_eq!(bytes(&dst, 12), [0, 0, 1, 1, 2, 2, 0, 0, 1, 1, 2, 2]);
            let conf = TensorDyn::from_raw(conf);
            // softmax max of logits (1, 0, 0): e / (e + 2) = 0.576 -> 147.
            assert!(bytes(&conf, 12).iter().all(|&p| p == 147));

            // Keep the left third of the model input.
            let lb = [0.0f32, 0.0, 1.0 / 3.0, 1.0];
            assert_eq!(ef_semantic_options_set_letterbox(o, lb.as_ptr()), 0);
            let narrow = tensor(&[2, 2], DType::U8).into_raw();
            assert_eq!(
                ef_decoder_decode_semantic(d, outputs.as_ptr(), 1, narrow, std::ptr::null_mut(), o),
                0
            );
            assert_eq!(bytes(&TensorDyn::from_raw(narrow), 4), [0, 0, 0, 0]);

            let wrong = tensor(&[2, 6], DType::F32).into_raw();
            assert_eq!(
                ef_decoder_decode_semantic(d, outputs.as_ptr(), 1, wrong, std::ptr::null_mut(), o),
                libc::EBADMSG
            );
            drop(TensorDyn::from_raw(wrong));
            let spare = tensor(&[2, 6], DType::U8).into_raw();
            assert_eq!(
                ef_decoder_decode_semantic(d, std::ptr::null(), 1, spare, std::ptr::null_mut(), o),
                libc::EINVAL
            );
            drop(TensorDyn::from_raw(spare));
            ef_semantic_options_free(o);
            drop(TensorDyn::from_raw(scores));
            ef_decoder_free(d);
        }
    }

    #[test]
    fn detection_models_have_no_semantic_output() {
        let d = decoder(
            r#"{"outputs": [{"decoder": "ultralytics", "type": "detection",
                "shape": [1, 84, 8400],
                "dshape": [["batch", 1], ["num_features", 84], ["num_boxes", 8400]]}]}"#,
        );
        unsafe {
            let mut classes = 0usize;
            assert_eq!(ef_decoder_semantic_classes(d, &mut classes), libc::ENOENT);
            let t = tensor(&[1, 84, 8400], DType::F32).into_raw();
            let dst = tensor(&[4, 4], DType::U8).into_raw();
            let outputs = [t as *const EfTensor];
            assert_eq!(
                ef_decoder_decode_semantic(
                    d,
                    outputs.as_ptr(),
                    1,
                    dst,
                    std::ptr::null_mut(),
                    std::ptr::null()
                ),
                libc::ENOENT
            );
            drop(TensorDyn::from_raw(t));
            drop(TensorDyn::from_raw(dst));
            ef_decoder_free(d);
        }
    }
}
