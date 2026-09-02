// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! C API for Ultralytics schema inference from raw model I/O signals.

use std::collections::BTreeMap;
use std::ffi::{c_char, c_int, CString};
use std::panic::{catch_unwind, AssertUnwindSafe};

use edgefirst_decoder::schema::{DType, Quantization};
use edgefirst_decoder::{
    infer_ultralytics_schema, InferredSchema, ModelSignals, ModelSource, TensorInfo,
};

use crate::decode::read_str;

/// Raw model I/O signals, accumulated field by field.
pub struct EfInferSignals {
    source: ModelSource,
    inputs: Vec<TensorInfo>,
    outputs: Vec<TensorInfo>,
    metadata: BTreeMap<String, String>,
}

/// An inferred Ultralytics schema. Its JSON views are rendered on demand by
/// [`ef_inferred_schema_json`]/[`ef_inferred_schema_labels_json`].
pub struct EfInferredSchema {
    inner: InferredSchema,
}

/// Maps the `source` code documented on [`ef_infer_signals_new`].
fn source_from(code: u32) -> Option<ModelSource> {
    match code {
        0 => Some(ModelSource::Onnx),
        1 => Some(ModelSource::TfLite),
        2 => Some(ModelSource::Other),
        _ => None,
    }
}

/// Maps an `EF_INFER_DTYPE_*` code to `schema::DType`. This is a distinct
/// vocabulary from `tensor.h`'s `EF_DTYPE_*` (which numbers
/// `edgefirst_tensor::DType`, a different, wider enum used for physical
/// tensor storage) -- `schema::DType` is the narrower quantized/float set a
/// model's logical I/O carries, so it gets its own codes rather than
/// reusing `EF_DTYPE_*` positions that would silently misalign.
///
/// The codes start at `EF_INFER_DTYPE_BASE` rather than `0` so the two
/// vocabularies occupy disjoint ranges: both cross the boundary as bare
/// `u32`, so overlapping ranges would have made every `EF_DTYPE_*` value a
/// valid -- and differently-meaning -- code here, silently misreading a
/// caller that passed the tensor dtype it already had. See the parity test
/// `the_header_infer_dtype_codes_match_dtype_from`.
pub(crate) const EF_INFER_DTYPE_BASE: u32 = 0x100;

pub(crate) fn dtype_from(code: u32) -> Option<DType> {
    match code.checked_sub(EF_INFER_DTYPE_BASE)? {
        0 => Some(DType::Int8),
        1 => Some(DType::Uint8),
        2 => Some(DType::Int16),
        3 => Some(DType::Uint16),
        4 => Some(DType::Int32),
        5 => Some(DType::Uint32),
        6 => Some(DType::Float16),
        7 => Some(DType::Float32),
        _ => None,
    }
}

/// Create empty signals for a model read from `source` (`0` onnx, `1`
/// tflite, `2` other). `NULL` for an unrecognized source or allocation
/// failure.
#[no_mangle]
pub extern "C" fn ef_infer_signals_new(source: u32) -> *mut EfInferSignals {
    let Some(source) = source_from(source) else {
        return std::ptr::null_mut();
    };
    catch_unwind(|| {
        Box::into_raw(Box::new(EfInferSignals {
            source,
            inputs: Vec::new(),
            outputs: Vec::new(),
            metadata: BTreeMap::new(),
        }))
    })
    .unwrap_or(std::ptr::null_mut())
}

/// Free signals. Freeing `NULL` is a no-op.
///
/// # Safety
/// `s` must be `NULL` or have come from this library.
#[no_mangle]
pub unsafe extern "C" fn ef_infer_signals_free(s: *mut EfInferSignals) {
    unsafe {
        if s.is_null() {
            return;
        }
        let _ = catch_unwind(AssertUnwindSafe(|| drop(Box::from_raw(s))));
    }
}

/// Run a setter, rejecting a null handle.
unsafe fn with_signals<F>(s: *mut EfInferSignals, body: F) -> c_int
where
    F: FnOnce(&mut EfInferSignals) -> c_int,
{
    unsafe {
        catch_unwind(AssertUnwindSafe(|| {
            if s.is_null() {
                return libc::EINVAL;
            }
            body(&mut *s)
        }))
        .unwrap_or(libc::EINVAL)
    }
}

/// Append an input tensor. `dtype` is an `EF_INFER_DTYPE_*` code.
///
/// @return 0 on success, `EINVAL` for a null/invalid argument.
///
/// # Safety
/// `name` must be NUL-terminated; `shape` must point to `rank` sizes.
#[no_mangle]
pub unsafe extern "C" fn ef_infer_signals_add_input(
    s: *mut EfInferSignals,
    name: *const c_char,
    shape: *const usize,
    rank: usize,
    dtype: u32,
) -> c_int {
    unsafe {
        with_signals(s, |s| {
            if shape.is_null() || rank == 0 {
                return libc::EINVAL;
            }
            let Some(name) = read_str(name, 0) else {
                return libc::EINVAL;
            };
            let Some(dtype) = dtype_from(dtype) else {
                return libc::EINVAL;
            };
            let shape = std::slice::from_raw_parts(shape, rank).to_vec();
            s.inputs.push(TensorInfo {
                name,
                shape,
                dtype,
                quantization: None,
            });
            0
        })
    }
}

/// Append an output tensor, with optional quantization.
///
/// `quant_len` is `0` for an unquantized tensor or `1` for per-tensor
/// quantization; `scale` and `zero_point` (when non-NULL) each carry
/// `quant_len` entries. `zero_point` may be `NULL` for symmetric
/// quantization.
///
/// A `quant_len` above `1` describes per-channel quantization, which this
/// setter accepts but [`ef_infer_ultralytics_schema`] then refuses: the
/// decoder consumes per-tensor quantization only, so such a schema would
/// build a decoder that fails. The refusal is deferred to inference so the
/// error arrives on the call that reports errors, with a message naming
/// the offending tensor.
///
/// @return 0 on success, `EINVAL` for a null/invalid argument (including a
///         nonzero `quant_len` with a `NULL` `scale`).
///
/// # Safety
/// `name` must be NUL-terminated; `shape` must point to `rank` sizes;
/// `scale`/`zero_point` must point to `quant_len` elements when non-NULL.
#[no_mangle]
pub unsafe extern "C" fn ef_infer_signals_add_output(
    s: *mut EfInferSignals,
    name: *const c_char,
    shape: *const usize,
    rank: usize,
    dtype: u32,
    scale: *const f32,
    zero_point: *const i32,
    quant_len: usize,
) -> c_int {
    unsafe {
        if s.is_null() {
            return libc::EINVAL;
        }
        if quant_len > 0 && scale.is_null() {
            return libc::EINVAL;
        }
        with_signals(s, |s| {
            if shape.is_null() || rank == 0 {
                return libc::EINVAL;
            }
            let Some(name) = read_str(name, 0) else {
                return libc::EINVAL;
            };
            let Some(dtype) = dtype_from(dtype) else {
                return libc::EINVAL;
            };
            let quantization = if quant_len == 0 {
                None
            } else {
                let scale = std::slice::from_raw_parts(scale, quant_len).to_vec();
                let zero_point = if zero_point.is_null() {
                    None
                } else {
                    Some(std::slice::from_raw_parts(zero_point, quant_len).to_vec())
                };
                Some(Quantization {
                    scale,
                    zero_point,
                    axis: None,
                    dtype: Some(dtype),
                })
            };
            let shape = std::slice::from_raw_parts(shape, rank).to_vec();
            s.outputs.push(TensorInfo {
                name,
                shape,
                dtype,
                quantization,
            });
            0
        })
    }
}

/// Insert a metadata key/value pair, as captured verbatim from the model's
/// container format (ONNX `metadata_props`, TFLite `metadata.json`).
///
/// @return 0 on success, `EINVAL` for a null handle or unreadable string.
///
/// # Safety
/// `key` and `value` must be NUL-terminated.
#[no_mangle]
pub unsafe extern "C" fn ef_infer_signals_add_metadata(
    s: *mut EfInferSignals,
    key: *const c_char,
    value: *const c_char,
) -> c_int {
    unsafe {
        with_signals(s, |s| {
            let Some(key) = read_str(key, 0) else {
                return libc::EINVAL;
            };
            let Some(value) = read_str(value, 0) else {
                return libc::EINVAL;
            };
            s.metadata.insert(key, value);
            0
        })
    }
}

/// Writes `msg` into `*err_out` when `err_out` is non-NULL, freed by the
/// caller with [`crate::decode::ef_decoder_string_free`]. Best-effort: an
/// embedded NUL (never produced by [`edgefirst_decoder::InferError`]'s
/// `Display`) silently leaves `*err_out` untouched rather than panicking.
///
/// # Safety
/// `err_out` must be `NULL` or writable.
unsafe fn set_err(err_out: *mut *mut c_char, msg: &str) {
    unsafe {
        if err_out.is_null() {
            return;
        }
        if let Ok(c) = CString::new(msg) {
            *err_out = c.into_raw();
        }
    }
}

/// Infer an Ultralytics schema from accumulated signals.
///
/// `NULL` on failure. When `err_out` is non-NULL, `*err_out` is set to a
/// message the caller frees with `ef_decoder_string_free`.
///
/// **Initialize your `char *` to `NULL` before calling.** `*err_out` is
/// left untouched on success, and while every failure path writes a
/// message, the write itself can still fail (a message carrying an
/// embedded NUL, or the allocation behind it). Detect failure from the
/// returned handle, and test `*err_out` separately before reading it:
///
/// ```c
/// char *err = NULL;
/// ef_inferred_schema *r = ef_infer_ultralytics_schema(s, &err);
/// if (!r) {
///     fprintf(stderr, "%s\n", err ? err : "(no message)");
///     ef_decoder_string_free(err); // freeing NULL is a no-op
/// }
/// ```
///
/// # Safety
/// `s` must be `NULL` or a live handle from this library; `err_out` must be
/// `NULL` or writable.
#[no_mangle]
pub unsafe extern "C" fn ef_infer_ultralytics_schema(
    s: *const EfInferSignals,
    err_out: *mut *mut c_char,
) -> *mut EfInferredSchema {
    unsafe {
        catch_unwind(AssertUnwindSafe(|| {
            if s.is_null() {
                set_err(err_out, "null signals handle");
                return std::ptr::null_mut();
            }
            let s = &*s;
            let signals = ModelSignals {
                source: s.source,
                inputs: s.inputs.clone(),
                outputs: s.outputs.clone(),
                metadata: s.metadata.clone(),
            };
            match infer_ultralytics_schema(&signals) {
                Ok(inner) => Box::into_raw(Box::new(EfInferredSchema { inner })),
                Err(e) => {
                    set_err(err_out, &e.to_string());
                    std::ptr::null_mut()
                }
            }
        }))
        .unwrap_or_else(|_| {
            // The one entry point where catch_unwind is load-bearing: an
            // unwind escaping `extern "C"` aborts the process. Write a
            // message here too, or a caught panic would be the single
            // failure that returns NULL with `*err_out` untouched --
            // exactly the case a caller trusting the contract would print.
            set_err(err_out, "internal error: panic during schema inference");
            std::ptr::null_mut()
        })
    }
}

/// Serializes `v` to a NUL-terminated JSON string, or `NULL` on failure.
fn to_json_c_string<T: serde::Serialize>(v: &T) -> *mut c_char {
    let Ok(json) = serde_json::to_string(v) else {
        return std::ptr::null_mut();
    };
    match CString::new(json) {
        Ok(c) => c.into_raw(),
        Err(_) => std::ptr::null_mut(),
    }
}

/// The inferred schema as `edgefirst.json` schema v2 JSON. The caller frees
/// the result with `ef_decoder_string_free`. `NULL` for a `NULL` handle or
/// on serialization failure.
///
/// # Safety
/// `r` must be `NULL` or a live handle from this library.
#[no_mangle]
pub unsafe extern "C" fn ef_inferred_schema_json(r: *const EfInferredSchema) -> *mut c_char {
    unsafe {
        if r.is_null() {
            return std::ptr::null_mut();
        }
        catch_unwind(AssertUnwindSafe(|| to_json_c_string(&(*r).inner.schema)))
            .unwrap_or(std::ptr::null_mut())
    }
}

/// The inferred class labels as a JSON array of strings, in index order.
/// The caller frees the result with `ef_decoder_string_free`. `NULL` for a
/// `NULL` handle or on serialization failure.
///
/// # Safety
/// `r` must be `NULL` or a live handle from this library.
#[no_mangle]
pub unsafe extern "C" fn ef_inferred_schema_labels_json(r: *const EfInferredSchema) -> *mut c_char {
    unsafe {
        if r.is_null() {
            return std::ptr::null_mut();
        }
        catch_unwind(AssertUnwindSafe(|| to_json_c_string(&(*r).inner.labels)))
            .unwrap_or(std::ptr::null_mut())
    }
}

/// A human-readable summary, e.g. "Ultralytics YOLO26 segment, 80 classes".
/// The caller frees the result with `ef_decoder_string_free`. `NULL` for a
/// `NULL` handle.
///
/// # Safety
/// `r` must be `NULL` or a live handle from this library.
#[no_mangle]
pub unsafe extern "C" fn ef_inferred_schema_description(r: *const EfInferredSchema) -> *mut c_char {
    unsafe {
        if r.is_null() {
            return std::ptr::null_mut();
        }
        catch_unwind(AssertUnwindSafe(|| {
            match CString::new((*r).inner.description.clone()) {
                Ok(c) => c.into_raw(),
                Err(_) => std::ptr::null_mut(),
            }
        }))
        .unwrap_or(std::ptr::null_mut())
    }
}

/// Free an inferred schema. Freeing `NULL` is a no-op.
///
/// # Safety
/// `r` must be `NULL` or have come from this library.
#[no_mangle]
pub unsafe extern "C" fn ef_inferred_schema_free(r: *mut EfInferredSchema) {
    unsafe {
        if r.is_null() {
            return;
        }
        let _ = catch_unwind(AssertUnwindSafe(|| drop(Box::from_raw(r))));
    }
}

#[cfg(test)]
mod tests {
    use std::ffi::CStr;

    use super::*;

    /// Builds a synthetic 80-class Ultralytics `names` dict-repr string,
    /// matching the format real ONNX/TFLite exports carry (see
    /// `edgefirst-decoder`'s own `synthetic_metadata` test helper).
    fn names_dict(nc: usize) -> CString {
        let body = (0..nc)
            .map(|i| format!("{i}: 'c{i}'"))
            .collect::<Vec<_>>()
            .join(", ");
        CString::new(format!("{{{body}}}")).unwrap()
    }

    /// Builds signals for the `yolov8n` fixture's shapes (see
    /// `crates/decoder/testdata/infer/yolov8n.signals.json`) through the
    /// extern "C" entry points, with a minimal (not the full captured)
    /// metadata set: `names`, `task`, `end2end`.
    unsafe fn yolov8n_signals() -> *mut EfInferSignals {
        unsafe {
            let s = ef_infer_signals_new(0); // onnx
            assert!(!s.is_null());

            let images = CString::new("images").unwrap();
            let input_shape: [usize; 4] = [1, 3, 640, 640];
            assert_eq!(
                ef_infer_signals_add_input(
                    s,
                    images.as_ptr(),
                    input_shape.as_ptr(),
                    input_shape.len(),
                    EF_INFER_DTYPE_BASE + 7, // float32
                ),
                0
            );

            let output0 = CString::new("output0").unwrap();
            let output_shape: [usize; 3] = [1, 84, 8400];
            assert_eq!(
                ef_infer_signals_add_output(
                    s,
                    output0.as_ptr(),
                    output_shape.as_ptr(),
                    output_shape.len(),
                    EF_INFER_DTYPE_BASE + 7, // float32
                    std::ptr::null(),
                    std::ptr::null(),
                    0, // unquantized
                ),
                0
            );

            let names_key = CString::new("names").unwrap();
            let names_val = names_dict(80);
            assert_eq!(
                ef_infer_signals_add_metadata(s, names_key.as_ptr(), names_val.as_ptr()),
                0
            );
            let task_key = CString::new("task").unwrap();
            let task_val = CString::new("detect").unwrap();
            assert_eq!(
                ef_infer_signals_add_metadata(s, task_key.as_ptr(), task_val.as_ptr()),
                0
            );
            let e2e_key = CString::new("end2end").unwrap();
            let e2e_val = CString::new("False").unwrap();
            assert_eq!(
                ef_infer_signals_add_metadata(s, e2e_key.as_ptr(), e2e_val.as_ptr()),
                0
            );

            s
        }
    }

    #[test]
    fn infers_a_schema_for_yolov8n_shapes() {
        unsafe {
            let s = yolov8n_signals();
            let mut err: *mut c_char = std::ptr::null_mut();
            let r = ef_infer_ultralytics_schema(s, &mut err);
            assert!(!r.is_null(), "inference must succeed on valid signals");
            assert!(err.is_null(), "err_out must stay untouched on success");

            let schema_json = ef_inferred_schema_json(r);
            assert!(!schema_json.is_null());
            let json_str = CStr::from_ptr(schema_json).to_str().unwrap();
            let parsed = edgefirst_decoder::schema::SchemaV2::parse_json(json_str)
                .expect("ef_inferred_schema_json must produce parseable SchemaV2 JSON");
            assert_eq!(parsed.schema_version, 2);

            let labels_json = ef_inferred_schema_labels_json(r);
            assert!(!labels_json.is_null());
            let labels_str = CStr::from_ptr(labels_json).to_str().unwrap();
            let labels: Vec<String> = serde_json::from_str(labels_str).unwrap();
            assert_eq!(labels.len(), 80);

            let description = ef_inferred_schema_description(r);
            assert!(!description.is_null());
            let description_str = CStr::from_ptr(description).to_str().unwrap();
            assert!(!description_str.is_empty());

            crate::decode::ef_decoder_string_free(schema_json);
            crate::decode::ef_decoder_string_free(labels_json);
            crate::decode::ef_decoder_string_free(description);
            ef_inferred_schema_free(r);
            ef_infer_signals_free(s);
        }
    }

    #[test]
    fn empty_metadata_is_rejected_with_an_error_message() {
        unsafe {
            let s = ef_infer_signals_new(0);
            assert!(!s.is_null());
            let mut err: *mut c_char = std::ptr::null_mut();
            let r = ef_infer_ultralytics_schema(s, &mut err);
            assert!(
                r.is_null(),
                "empty metadata carries no Ultralytics signature"
            );
            assert!(!err.is_null(), "a failure must set err_out");
            let msg = CStr::from_ptr(err).to_str().unwrap();
            assert!(!msg.is_empty());
            crate::decode::ef_decoder_string_free(err);
            ef_infer_signals_free(s);
        }
    }

    #[test]
    fn a_null_signals_handle_is_rejected_not_read() {
        unsafe {
            let mut err: *mut c_char = std::ptr::null_mut();
            let r = ef_infer_ultralytics_schema(std::ptr::null(), &mut err);
            assert!(r.is_null());
            assert!(!err.is_null());
            crate::decode::ef_decoder_string_free(err);
        }
    }

    #[test]
    fn an_unrecognized_source_yields_null() {
        assert!(ef_infer_signals_new(99).is_null());
    }

    #[test]
    fn every_setter_rejects_a_null_handle() {
        unsafe {
            let n = std::ptr::null_mut();
            let name = CString::new("x").unwrap();
            let shape: [usize; 1] = [1];
            assert_eq!(
                ef_infer_signals_add_input(n, name.as_ptr(), shape.as_ptr(), 1, 0),
                libc::EINVAL
            );
            assert_eq!(
                ef_infer_signals_add_output(
                    n,
                    name.as_ptr(),
                    shape.as_ptr(),
                    1,
                    0,
                    std::ptr::null(),
                    std::ptr::null(),
                    0
                ),
                libc::EINVAL
            );
            assert_eq!(
                ef_infer_signals_add_metadata(n, name.as_ptr(), name.as_ptr()),
                libc::EINVAL
            );
        }
    }

    #[test]
    fn an_unknown_dtype_code_is_refused() {
        unsafe {
            let s = ef_infer_signals_new(0);
            let name = CString::new("x").unwrap();
            let shape: [usize; 1] = [1];
            assert_eq!(
                ef_infer_signals_add_input(s, name.as_ptr(), shape.as_ptr(), 1, 99),
                libc::EINVAL
            );
            ef_infer_signals_free(s);
        }
    }

    #[test]
    fn a_tensor_dtype_code_is_refused_not_silently_misread() {
        // `EF_DTYPE_*` (tensor.h) numbers 0..=10 over a different, wider
        // vocabulary. Had `EF_INFER_DTYPE_*` also started at 0, a caller
        // passing the tensor dtype it already held would have been accepted
        // as a *different* dtype -- `EF_DTYPE_I64` (7) read as FLOAT32.
        // Disjoint ranges turn that mistake into an error.
        unsafe {
            let s = ef_infer_signals_new(0);
            let name = CString::new("x").unwrap();
            let shape: [usize; 1] = [1];
            for tensor_code in 0..=10u32 {
                assert_eq!(
                    ef_infer_signals_add_input(s, name.as_ptr(), shape.as_ptr(), 1, tensor_code),
                    libc::EINVAL,
                    "EF_DTYPE_* code {tensor_code} must not be a valid EF_INFER_DTYPE_* code"
                );
            }
            ef_infer_signals_free(s);
        }
    }

    #[test]
    fn the_header_infer_dtype_codes_match_dtype_from() {
        // Name-level checks prove a spelling exists; they say nothing about
        // which integer sits next to it. These macros are hand-written into
        // cbindgen.toml's header block and `dtype_from` is a separately
        // hand-written match, so nothing but this test links the two. Same
        // approach as tensor-capi's
        // `the_header_enumerator_values_match_the_rust_vocabulary`.
        let header = std::fs::read_to_string(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/include/edgefirst/decoder.h"
        ))
        .expect("cbindgen wrote include/edgefirst/decoder.h");

        let expected = [
            ("EF_INFER_DTYPE_INT8", DType::Int8),
            ("EF_INFER_DTYPE_UINT8", DType::Uint8),
            ("EF_INFER_DTYPE_INT16", DType::Int16),
            ("EF_INFER_DTYPE_UINT16", DType::Uint16),
            ("EF_INFER_DTYPE_INT32", DType::Int32),
            ("EF_INFER_DTYPE_UINT32", DType::Uint32),
            ("EF_INFER_DTYPE_FLOAT16", DType::Float16),
            ("EF_INFER_DTYPE_FLOAT32", DType::Float32),
        ];

        let mut seen = 0;
        for (name, want) in expected {
            let line = header
                .lines()
                .find(|l| l.starts_with(&format!("#define {name} ")))
                .unwrap_or_else(|| panic!("decoder.h does not define {name}"));
            let literal = line.rsplit(' ').next().expect("a value after the name");
            let code = u32::from_str_radix(literal.trim_start_matches("0x"), 16)
                .unwrap_or_else(|e| panic!("{name} value `{literal}` is not hex: {e}"));
            assert_eq!(
                dtype_from(code),
                Some(want),
                "{name} is {literal} in decoder.h, which dtype_from maps elsewhere"
            );
            seen += 1;
        }

        // Exhaustiveness: a new `schema::DType` variant must not silently
        // go unnamed by the header. This count is the only thing that
        // fails the build when one is added.
        assert_eq!(seen, 8, "decoder.h must name every schema::DType variant");
        assert_eq!(
            dtype_from(EF_INFER_DTYPE_BASE + 8),
            None,
            "an unnamed code past the last variant must not map"
        );
    }

    #[test]
    fn quantized_output_needs_a_scale_pointer() {
        unsafe {
            let s = ef_infer_signals_new(0);
            let name = CString::new("x").unwrap();
            let shape: [usize; 1] = [1];
            assert_eq!(
                ef_infer_signals_add_output(
                    s,
                    name.as_ptr(),
                    shape.as_ptr(),
                    1,
                    0,
                    std::ptr::null(), // scale is NULL
                    std::ptr::null(),
                    1, // but quant_len says per-tensor
                ),
                libc::EINVAL
            );
            ef_infer_signals_free(s);
        }
    }

    #[test]
    fn freeing_null_handles_is_a_no_op() {
        unsafe {
            ef_infer_signals_free(std::ptr::null_mut());
            ef_inferred_schema_free(std::ptr::null_mut());
        }
    }
}
