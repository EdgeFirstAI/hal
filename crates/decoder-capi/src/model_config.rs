// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! Model configuration discovery: a model's schema, labels and input
//! preprocessing from what it carries, with the source of each value.
//!
//! The preprocessing crosses to libedgefirst_image as an `ef_preprocess_spec`
//! (declared in `edgefirst/detect.h`), so neither library links the other.

use std::ffi::{c_char, c_int, CStr, CString};
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::path::PathBuf;

use edgefirst_decoder::model_config::{
    resolve_model_config, ConfigField, ModelConfigError, ModelOverrides, ResolvedModelConfig,
};
use edgefirst_decoder_abi::EfPreprocessSpec;
use edgefirst_tensor::PreprocessSpec;

use crate::infer::{set_err, to_json_c_string, EfInferSignals};

/// Overrides applied after every discovered source. Build with
/// [`ef_model_overrides_new`], free with [`ef_model_overrides_free`].
pub struct EfModelOverrides {
    inner: ModelOverrides,
}

/// A resolved model configuration. Free with [`ef_model_config_free`].
pub struct EfModelConfig {
    inner: ResolvedModelConfig,
}

/// The errno for a failed resolution.
fn config_errno(e: &ModelConfigError) -> c_int {
    use ModelConfigError as E;
    match e {
        E::Io { source, .. } => source.raw_os_error().unwrap_or(match source.kind() {
            std::io::ErrorKind::NotFound => libc::ENOENT,
            std::io::ErrorKind::IsADirectory => libc::EISDIR,
            std::io::ErrorKind::FileTooLarge => libc::EFBIG,
            std::io::ErrorKind::InvalidData => libc::EILSEQ,
            _ => libc::EIO,
        }),
        E::Container(_) | E::InvalidField { .. } | E::Infer(_) | E::Schema(_) => libc::EBADMSG,
        E::NotSupported { .. } | E::UnsupportedArchitecture(_) => libc::ENOTSUP,
        E::Unresolved { .. } => libc::ENODATA,
        _ => libc::EINVAL,
    }
}

/// `s` as a string the caller frees with `ef_decoder_string_free`; a NUL in
/// it (from a model's own metadata) becomes U+FFFD.
fn c_string(s: &str) -> *mut c_char {
    let s = s.replace('\0', "\u{FFFD}");
    CString::new(s).map_or(std::ptr::null_mut(), CString::into_raw)
}

/// Read a NUL-terminated path; bytes as-is on Unix, UTF-8 elsewhere.
///
/// # Safety
/// `p` must be `NULL` or NUL-terminated.
unsafe fn read_path(p: *const c_char) -> Option<PathBuf> {
    if p.is_null() {
        return None;
    }
    // SAFETY: the caller's contract.
    let c = unsafe { CStr::from_ptr(p) };
    #[cfg(unix)]
    {
        use std::os::unix::ffi::OsStrExt;
        Some(PathBuf::from(std::ffi::OsStr::from_bytes(c.to_bytes())))
    }
    #[cfg(not(unix))]
    {
        c.to_str().ok().map(PathBuf::from)
    }
}

/// New empty overrides.
#[no_mangle]
pub extern "C" fn ef_model_overrides_new() -> *mut EfModelOverrides {
    catch_unwind(|| {
        Box::into_raw(Box::new(EfModelOverrides {
            inner: ModelOverrides::new(),
        }))
    })
    .unwrap_or(std::ptr::null_mut())
}

/// Free overrides. `NULL` is a no-op.
///
/// # Safety
/// `o` must be `NULL` or have come from [`ef_model_overrides_new`].
#[no_mangle]
pub unsafe extern "C" fn ef_model_overrides_free(o: *mut EfModelOverrides) {
    if !o.is_null() {
        // SAFETY: the caller's contract; the handle came from `Box::into_raw`.
        let _ = catch_unwind(AssertUnwindSafe(|| drop(unsafe { Box::from_raw(o) })));
    }
}

/// Set the override named `key` from its string form:
///
/// | key | values |
/// |---|---|
/// | `task` | `detect`, `segment`, `semantic` |
/// | `normalize` | `unsigned`, `signed`, `raw`, `imagenet`, or `R,G,B/R,G,B` (mean/std on 0-1 values) |
/// | `resize` | `stretch`, `letterbox`, `letterbox-center`, `letterbox-top_left` |
/// | `decode` | `accurate`, `fast` |
/// | `labels` | comma-separated non-empty class names, e.g. `background,person` |
///
/// A bare `letterbox` follows the model family's convention.
/// `ef_model_config_describe` prints these keys in the same grammar.
///
/// @return 0, or `EINVAL` for a `NULL` argument, an unknown key or a value
/// outside the key's grammar; `*err_out` (when `err_out` is not `NULL`)
/// then receives a message listing the accepted forms, freed with
/// `ef_decoder_string_free`. The overrides are unchanged on failure.
///
/// # Safety
/// `o` must be `NULL` or live; `key` and `value` `NULL` or NUL-terminated;
/// `err_out` `NULL` or writable.
#[no_mangle]
pub unsafe extern "C" fn ef_model_overrides_set(
    o: *mut EfModelOverrides,
    key: *const c_char,
    value: *const c_char,
    err_out: *mut *mut c_char,
) -> c_int {
    catch_unwind(AssertUnwindSafe(|| {
        // SAFETY: the caller's contract.
        let Some(o) = (unsafe { o.as_mut() }) else {
            unsafe { set_err(err_out, "null overrides handle") };
            return libc::EINVAL;
        };
        // SAFETY: the caller's contract.
        let (Some(key), Some(value)) = (unsafe { crate::decode::read_str(key, 0) }, unsafe {
            crate::decode::read_str(value, 0)
        }) else {
            unsafe { set_err(err_out, "override key and value must be UTF-8 strings") };
            return libc::EINVAL;
        };
        let mut next = o.inner.clone();
        match next.set(&key, &value) {
            Ok(()) => {
                o.inner = next;
                0
            }
            Err(e) => {
                // SAFETY: the caller's contract.
                unsafe { set_err(err_out, &e.to_string()) };
                libc::EINVAL
            }
        }
    }))
    .unwrap_or_else(|_| {
        // SAFETY: the caller's contract.
        unsafe { set_err(err_out, "internal error: panic while setting an override") };
        libc::EINVAL
    })
}

/// Resolve the configuration of the model at `path` from, in order: an
/// `edgefirst.json` embedded in it, `edgefirst.json`/`labels.txt` beside it,
/// Hugging Face configs embedded in it, `config.json` and
/// `preprocessor_config.json` beside it, Ultralytics metadata in `signals`,
/// Hugging Face input/output names in `signals`, and the defaults; then
/// `overrides`. Each source fills only what the ones before it left unset.
///
/// `signals` (`NULL` for none) are the model's tensors as the runtime
/// reports them, built as for `ef_infer_ultralytics_schema`; a Hugging Face
/// config needs them to build a schema. `overrides` may be `NULL`.
///
/// @return the configuration, or `NULL` with `errno` set and, when `err_out`
/// is not `NULL`, a message in `*err_out` freed with `ef_decoder_string_free`
/// (initialise it to `NULL`; it is untouched on success): `ENOENT` when the
/// file does not exist, `ENODATA` when nothing configures a required piece
/// (the message names it and the override that supplies it), `EBADMSG` for a malformed
/// container or configuration file, `ENOTSUP` for a configuration the
/// library cannot run, `EINVAL` for a `NULL` path or an override that does
/// not apply, otherwise the errno of the failed read.
///
/// # Safety
/// `path` must be `NULL` or NUL-terminated; `signals` and `overrides` `NULL`
/// or live; `err_out` `NULL` or writable.
#[no_mangle]
pub unsafe extern "C" fn ef_model_config_resolve(
    path: *const c_char,
    signals: *const EfInferSignals,
    overrides: *const EfModelOverrides,
    err_out: *mut *mut c_char,
) -> *mut EfModelConfig {
    let fail = |code: c_int, msg: &str| {
        // SAFETY: the caller's contract.
        unsafe { set_err(err_out, msg) };
        errno::set_errno(errno::Errno(code));
        std::ptr::null_mut()
    };
    catch_unwind(AssertUnwindSafe(|| {
        // SAFETY: the caller's contract.
        let Some(path) = (unsafe { read_path(path) }) else {
            return fail(libc::EINVAL, "null or non-UTF-8 model path");
        };
        // SAFETY: the caller's contract.
        let signals = unsafe { signals.as_ref() }.map(EfInferSignals::to_signals);
        let empty = ModelOverrides::new();
        // SAFETY: the caller's contract.
        let overrides = unsafe { overrides.as_ref() }.map_or(&empty, |o| &o.inner);
        match resolve_model_config(&path, signals, overrides) {
            Ok(inner) => Box::into_raw(Box::new(EfModelConfig { inner })),
            Err(e) => fail(config_errno(&e), &e.to_string()),
        }
    }))
    .unwrap_or_else(|_| {
        fail(
            libc::EINVAL,
            "internal error: panic during model configuration",
        )
    })
}

/// Free a configuration. `NULL` is a no-op.
///
/// # Safety
/// `c` must be `NULL` or have come from [`ef_model_config_resolve`].
#[no_mangle]
pub unsafe extern "C" fn ef_model_config_free(c: *mut EfModelConfig) {
    if !c.is_null() {
        // SAFETY: the caller's contract; the handle came from `Box::into_raw`.
        let _ = catch_unwind(AssertUnwindSafe(|| drop(unsafe { Box::from_raw(c) })));
    }
}

/// Render a string view of a live configuration, `NULL` for a `NULL` handle.
///
/// # Safety
/// `c` must be `NULL` or live.
unsafe fn string_view(
    c: *const EfModelConfig,
    f: impl FnOnce(&ResolvedModelConfig) -> *mut c_char,
) -> *mut c_char {
    // SAFETY: the caller's contract.
    let Some(c) = (unsafe { c.as_ref() }) else {
        return std::ptr::null_mut();
    };
    catch_unwind(AssertUnwindSafe(|| f(&c.inner))).unwrap_or(std::ptr::null_mut())
}

/// The output schema as `edgefirst.json` schema v2 JSON, ready for
/// `ef_decoder_params_set_config_json`. Its `input.preprocessing` carries
/// the resolved values when the schema has an input. Freed with
/// `ef_decoder_string_free`; `NULL` for a `NULL` handle.
///
/// # Safety
/// `c` must be `NULL` or live.
#[no_mangle]
pub unsafe extern "C" fn ef_model_config_schema_json(c: *const EfModelConfig) -> *mut c_char {
    unsafe { string_view(c, |c| to_json_c_string(&c.schema)) }
}

/// The class labels as a JSON array of strings in channel order, `[]` when
/// no source names them. Freed with `ef_decoder_string_free`; `NULL` for a
/// `NULL` handle.
///
/// # Safety
/// `c` must be `NULL` or live.
#[no_mangle]
pub unsafe extern "C" fn ef_model_config_labels_json(c: *const EfModelConfig) -> *mut c_char {
    unsafe { string_view(c, |c| to_json_c_string(&c.labels)) }
}

/// A one-line summary of the model. Freed with `ef_decoder_string_free`;
/// `NULL` for a `NULL` handle.
///
/// # Safety
/// `c` must be `NULL` or live.
#[no_mangle]
pub unsafe extern "C" fn ef_model_config_description(c: *const EfModelConfig) -> *mut c_char {
    unsafe { string_view(c, |c| c_string(&c.description)) }
}

/// One line per resolved value with its source, newline-separated, for
/// logging, e.g. `resize=stretch [preprocessor_config.json]`. Freed
/// with `ef_decoder_string_free`; `NULL` for a `NULL` handle.
///
/// # Safety
/// `c` must be `NULL` or live.
#[no_mangle]
pub unsafe extern "C" fn ef_model_config_describe(c: *const EfModelConfig) -> *mut c_char {
    unsafe { string_view(c, |c| c_string(&c.describe().join("\n"))) }
}

/// Deviations and skipped files worth logging, newline-separated; empty when
/// there are none. Freed with `ef_decoder_string_free`; `NULL` for a `NULL`
/// handle.
///
/// # Safety
/// `c` must be `NULL` or live.
#[no_mangle]
pub unsafe extern "C" fn ef_model_config_notes(c: *const EfModelConfig) -> *mut c_char {
    unsafe { string_view(c, |c| c_string(&c.notes.join("\n"))) }
}

/// Write a value read from a live configuration to `out`.
///
/// # Safety
/// `c` must be `NULL` or live; `out` `NULL` or writable.
unsafe fn code_view<T>(
    c: *const EfModelConfig,
    out: *mut T,
    f: impl FnOnce(&ResolvedModelConfig) -> Option<T>,
) -> c_int {
    // SAFETY: the caller's contract.
    let (Some(c), false) = (unsafe { c.as_ref() }, out.is_null()) else {
        return libc::EINVAL;
    };
    match f(&c.inner) {
        Some(code) => {
            // SAFETY: checked non-null; the caller's contract.
            unsafe { *out = code };
            0
        }
        None => libc::ENOENT,
    }
}

/// The model family: an `EF_MODEL_FAMILY_*` code.
///
/// @return 0, or `EINVAL` for a `NULL` argument.
///
/// # Safety
/// `c` must be `NULL` or live; `out` `NULL` or writable.
#[no_mangle]
pub unsafe extern "C" fn ef_model_config_family(c: *const EfModelConfig, out: *mut u32) -> c_int {
    unsafe { code_view(c, out, |c| Some(c.family.code())) }
}

/// What the outputs decode to: an `EF_MODEL_TASK_*` code.
///
/// @return 0, or `EINVAL` for a `NULL` argument.
///
/// # Safety
/// `c` must be `NULL` or live; `out` `NULL` or writable.
#[no_mangle]
pub unsafe extern "C" fn ef_model_config_task(c: *const EfModelConfig, out: *mut u32) -> c_int {
    unsafe { code_view(c, out, |c| Some(c.task.code())) }
}

/// The semantic decode mode: an `EF_DECODE_MODE_*` code.
///
/// @return 0, `ENOENT` for a model with no semantic output, or `EINVAL` for
/// a `NULL` argument.
///
/// # Safety
/// `c` must be `NULL` or live; `out` `NULL` or writable.
#[no_mangle]
pub unsafe extern "C" fn ef_model_config_decode_mode(
    c: *const EfModelConfig,
    out: *mut u32,
) -> c_int {
    unsafe { code_view(c, out, |c| c.decode_mode.map(|m| m.code())) }
}

/// Where the value of `field` (an `EF_CONFIG_FIELD_*` code) came from:
/// `*source_out` receives an `EF_CONFIG_SOURCE_*` code and, when
/// `detail_out` is not `NULL`, `*detail_out` the file or field it was read
/// from or the reason for a guess (freed with `ef_decoder_string_free`), or
/// `NULL` when there is none. Neither is written on failure.
///
/// @return 0, `ENOENT` for a field with no value (such as the decode mode of
/// a detection model), or `EINVAL` for a `NULL` argument or unknown field.
///
/// # Safety
/// `c` must be `NULL` or live; `source_out` `NULL` or writable; `detail_out`
/// `NULL` or writable.
#[no_mangle]
pub unsafe extern "C" fn ef_model_config_source(
    c: *const EfModelConfig,
    field: u32,
    source_out: *mut u32,
    detail_out: *mut *mut c_char,
) -> c_int {
    catch_unwind(AssertUnwindSafe(|| {
        // SAFETY: the caller's contract.
        let (Some(c), false) = (unsafe { c.as_ref() }, source_out.is_null()) else {
            return libc::EINVAL;
        };
        let Some(field) = ConfigField::from_code(field) else {
            return libc::EINVAL;
        };
        let Some(p) = c.inner.source(field) else {
            return libc::ENOENT;
        };
        // SAFETY: checked non-null; the caller's contract.
        unsafe { *source_out = p.source.code() };
        if !detail_out.is_null() {
            let detail = p.detail.as_deref().map_or(std::ptr::null_mut(), c_string);
            // SAFETY: checked non-null; the caller's contract.
            unsafe { *detail_out = detail };
        }
        0
    }))
    .unwrap_or(libc::EINVAL)
}

/// Hugging Face `do_reduce_labels`: `*out` is 1 when the reference dataset
/// drops its background class and shifts the rest down, else 0. Reported,
/// not applied.
///
/// @return 0, `ENOENT` when no source says, or `EINVAL` for a `NULL`
/// argument.
///
/// # Safety
/// `c` must be `NULL` or live; `out` `NULL` or writable.
#[no_mangle]
pub unsafe extern "C" fn ef_model_config_reduce_labels(
    c: *const EfModelConfig,
    out: *mut c_int,
) -> c_int {
    unsafe { code_view(c, out, |c| c.reduce_labels.map(c_int::from)) }
}

/// The C layout of `spec`.
fn spec_to_c(spec: &PreprocessSpec) -> EfPreprocessSpec {
    let (width, height) = spec
        .size
        .and_then(|(w, h)| Some((u32::try_from(w).ok()?, u32::try_from(h).ok()?)))
        .unwrap_or((0, 0));
    EfPreprocessSpec {
        width,
        height,
        fit: spec.fit.code(),
        placement: spec.placement.code(),
        pad_space: spec.pad_space.code(),
        pad: spec.pad,
        mean: spec.normalization.mean(),
        std: spec.normalization.std(),
        interpolation: spec.interpolation.code(),
    }
}

/// The model's input preprocessing, for `ef_convert_options_from_spec` in
/// libedgefirst_image. Its interpolation is the reference pipeline's; the
/// image library runs bicubic and area as bilinear, and
/// `ef_model_config_notes` says so.
///
/// @return 0, or `EINVAL` for a `NULL` argument.
///
/// # Safety
/// `c` must be `NULL` or live; `out` `NULL` or valid for writes.
#[no_mangle]
pub unsafe extern "C" fn ef_model_config_preprocessing(
    c: *const EfModelConfig,
    out: *mut EfPreprocessSpec,
) -> c_int {
    unsafe { code_view(c, out, |c| Some(spec_to_c(&c.preprocessing))) }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::codes::{EfConfigField, EfConfigSource, EfModelFamily, EfModelSource, EfModelTask};
    use crate::decode::{
        ef_decoder_free, ef_decoder_new, ef_decoder_params_free, ef_decoder_params_new,
        ef_decoder_params_set_config_json, ef_decoder_string_free,
    };
    use crate::infer::{
        ef_infer_signals_add_input, ef_infer_signals_add_output, ef_infer_signals_free,
        ef_infer_signals_new,
    };
    use crate::semantic::ef_decoder_semantic_classes;
    use edgefirst_decoder_abi::{EfFit, EfInterpolation, EfLetterboxPlacement, EfPadSpace};

    /// A model file with SegFormer-B0 ADE20k Hugging Face configs beside it,
    /// in a directory of its own.
    struct Sidecar {
        dir: PathBuf,
        model: CString,
    }

    impl Sidecar {
        fn new(tag: &str) -> Self {
            let hub =
                PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../testdata/hf/segformer-512");
            let dir =
                std::env::temp_dir().join(format!("ef_model_config-{tag}-{}", std::process::id()));
            std::fs::create_dir_all(&dir).unwrap();
            for name in ["config.json", "preprocessor_config.json"] {
                std::fs::copy(hub.join(name), dir.join(name)).unwrap();
            }
            let model = dir.join("model.bin");
            std::fs::write(&model, b"not a container").unwrap();
            let model = CString::new(model.to_str().unwrap()).unwrap();
            Self { dir, model }
        }
    }

    impl Drop for Sidecar {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.dir);
        }
    }

    /// The tensors a runtime reports for SegFormer-B0 at 512x512.
    fn segformer_signals() -> *mut EfInferSignals {
        let f32 = edgefirst_tensor::DType::F32.code();
        let input = CString::new("pixel_values").unwrap();
        let output = CString::new("logits").unwrap();
        unsafe {
            let s = ef_infer_signals_new(EfModelSource::Onnx as u32);
            let ishape = [1usize, 3, 512, 512];
            let oshape = [1usize, 150, 128, 128];
            assert_eq!(
                ef_infer_signals_add_input(s, input.as_ptr(), ishape.as_ptr(), 4, f32),
                0
            );
            assert_eq!(
                ef_infer_signals_add_output(
                    s,
                    output.as_ptr(),
                    oshape.as_ptr(),
                    4,
                    f32,
                    std::ptr::null(),
                    std::ptr::null(),
                    0
                ),
                0
            );
            s
        }
    }

    fn take(s: *mut c_char) -> String {
        assert!(!s.is_null());
        let out = unsafe { CStr::from_ptr(s) }.to_str().unwrap().to_owned();
        unsafe { ef_decoder_string_free(s) };
        out
    }

    fn resolve(
        model: &CStr,
        signals: *const EfInferSignals,
        overrides: *const EfModelOverrides,
    ) -> Result<*mut EfModelConfig, (c_int, String)> {
        let mut err: *mut c_char = std::ptr::null_mut();
        errno::set_errno(errno::Errno(0));
        let c = unsafe { ef_model_config_resolve(model.as_ptr(), signals, overrides, &mut err) };
        if c.is_null() {
            let code = errno::errno().0;
            Err((code, take(err)))
        } else {
            assert!(err.is_null(), "err_out is untouched on success");
            Ok(c)
        }
    }

    fn spec(c: *const EfModelConfig) -> EfPreprocessSpec {
        let mut out = EfPreprocessSpec::default();
        assert_eq!(unsafe { ef_model_config_preprocessing(c, &mut out) }, 0);
        out
    }

    #[test]
    fn a_sidecar_segformer_resolves_to_a_semantic_model() {
        let side = Sidecar::new("segformer");
        let signals = segformer_signals();
        let c = resolve(&side.model, signals, std::ptr::null()).unwrap();
        unsafe {
            ef_infer_signals_free(signals);
            let mut code = u32::MAX;
            assert_eq!(ef_model_config_family(c, &mut code), 0);
            assert_eq!(code, EfModelFamily::HuggingFace as u32);
            assert_eq!(ef_model_config_task(c, &mut code), 0);
            assert_eq!(code, EfModelTask::Semantic as u32);
            assert_eq!(ef_model_config_decode_mode(c, &mut code), 0);
            assert_eq!(code, crate::codes::EfDecodeMode::Accurate as u32);
            let mut reduce = -1;
            assert_eq!(ef_model_config_reduce_labels(c, &mut reduce), 0);
            assert_eq!(reduce, 1);

            let mut detail: *mut c_char = std::ptr::null_mut();
            assert_eq!(
                ef_model_config_source(
                    c,
                    EfConfigField::Normalization as u32,
                    &mut code,
                    &mut detail
                ),
                0
            );
            assert_eq!(code, EfConfigSource::SidecarHf as u32);
            assert_eq!(take(detail), "preprocessor_config.json");
            assert_eq!(
                ef_model_config_source(c, 99, &mut code, std::ptr::null_mut()),
                libc::EINVAL
            );

            let labels = take(ef_model_config_labels_json(c));
            let labels: Vec<String> = serde_json::from_str(&labels).unwrap();
            assert_eq!(labels.len(), 150);
            assert!(!take(ef_model_config_description(c)).is_empty());
            let describe = take(ef_model_config_describe(c));
            assert!(
                describe.lines().any(|l| l.starts_with("normalize=")),
                "{describe}"
            );
            let _ = take(ef_model_config_notes(c));

            let s = spec(c);
            assert_eq!((s.width, s.height), (512, 512));
            assert_eq!(s.fit, EfFit::Stretch as u32);
            assert_eq!(s.pad_space, EfPadSpace::Pixel as u32);
            assert_eq!(s.interpolation, EfInterpolation::Bilinear as u32);
            let imagenet = edgefirst_tensor::Normalization::imagenet();
            assert_eq!((s.mean, s.std), (imagenet.mean(), imagenet.std()));

            // The schema builds a decoder with the model's semantic output.
            let schema = ef_model_config_schema_json(c);
            let p = ef_decoder_params_new();
            assert_eq!(ef_decoder_params_set_config_json(p, schema, 0), 0);
            ef_decoder_string_free(schema);
            let d = ef_decoder_new(p);
            ef_decoder_params_free(p);
            assert!(!d.is_null());
            let mut classes = 0usize;
            assert_eq!(ef_decoder_semantic_classes(d, &mut classes), 0);
            assert_eq!(classes, 150);
            ef_decoder_free(d);
            ef_model_config_free(c);
        }
    }

    #[test]
    fn overrides_apply_last_and_are_recorded() {
        let side = Sidecar::new("overrides");
        let signals = segformer_signals();
        let o = ef_model_overrides_new();
        let set = |k: &str, v: &str| {
            let (k, v) = (CString::new(k).unwrap(), CString::new(v).unwrap());
            let mut err: *mut c_char = std::ptr::null_mut();
            let rc = unsafe { ef_model_overrides_set(o, k.as_ptr(), v.as_ptr(), &mut err) };
            (rc, (!err.is_null()).then(|| take(err)))
        };
        assert_eq!(set("resize", "letterbox-top_left"), (0, None));
        assert_eq!(set("decode", "fast"), (0, None));
        let (rc, msg) = set("resize", "fill");
        assert_eq!(rc, libc::EINVAL);
        assert!(
            msg.unwrap().contains("letterbox-top_left"),
            "lists the accepted forms"
        );
        let (rc, msg) = set("colour", "red");
        assert_eq!(rc, libc::EINVAL);
        assert!(msg.unwrap().contains("unknown override"));
        let c = resolve(&side.model, signals, o).unwrap();
        unsafe {
            let s = spec(c);
            assert_eq!(s.fit, EfFit::Letterbox as u32);
            assert_eq!(s.placement, EfLetterboxPlacement::TopLeft as u32);
            let mut code = u32::MAX;
            assert_eq!(
                ef_model_config_source(
                    c,
                    EfConfigField::Resize as u32,
                    &mut code,
                    std::ptr::null_mut()
                ),
                0
            );
            assert_eq!(code, EfConfigSource::Override as u32);
            assert_eq!(ef_model_config_decode_mode(c, &mut code), 0);
            assert_eq!(code, crate::codes::EfDecodeMode::Fast as u32);
            ef_model_config_free(c);
            ef_model_overrides_free(o);
            ef_model_overrides_free(std::ptr::null_mut());
            ef_infer_signals_free(signals);
        }
    }

    #[test]
    fn failures_set_errno_and_a_message() {
        let side = Sidecar::new("failures");
        let (code, msg) = resolve(&side.model, std::ptr::null(), std::ptr::null()).unwrap_err();
        assert_eq!(
            code,
            libc::ENODATA,
            "a Hugging Face config needs signals: {msg}"
        );
        assert!(!msg.is_empty());
        let missing = CString::new(side.dir.join("absent.onnx").to_str().unwrap()).unwrap();
        let (code, _) = resolve(&missing, std::ptr::null(), std::ptr::null()).unwrap_err();
        assert_eq!(code, libc::ENOENT);
        unsafe {
            let mut err: *mut c_char = std::ptr::null_mut();
            let c = ef_model_config_resolve(
                std::ptr::null(),
                std::ptr::null(),
                std::ptr::null(),
                &mut err,
            );
            assert!(c.is_null());
            assert_eq!(errno::errno().0, libc::EINVAL);
            ef_decoder_string_free(err);
            // A NULL err_out is allowed.
            assert!(ef_model_config_resolve(
                missing.as_ptr(),
                std::ptr::null(),
                std::ptr::null(),
                std::ptr::null_mut()
            )
            .is_null());
            ef_model_config_free(std::ptr::null_mut());
            assert!(ef_model_config_schema_json(std::ptr::null()).is_null());
            let mut code = 0;
            assert_eq!(
                ef_model_config_task(std::ptr::null(), &mut code),
                libc::EINVAL
            );
        }
    }

    #[test]
    fn preprocessing_refuses_null() {
        let side = Sidecar::new("preprocess");
        let signals = segformer_signals();
        let c = resolve(&side.model, signals, std::ptr::null()).unwrap();
        unsafe {
            ef_infer_signals_free(signals);
            assert_eq!(
                ef_model_config_preprocessing(c, std::ptr::null_mut()),
                libc::EINVAL
            );
            assert_eq!(
                ef_model_config_preprocessing(std::ptr::null(), &mut EfPreprocessSpec::default()),
                libc::EINVAL
            );
            ef_model_config_free(c);
        }
    }

    #[test]
    fn a_nul_in_a_string_is_replaced() {
        let s = c_string("a\0b");
        assert_eq!(take(s), "a\u{FFFD}b");
    }

    #[test]
    fn the_error_codes_follow_the_error_kind() {
        let e = ModelConfigError::Override("x".into());
        assert_eq!(config_errno(&e), libc::EINVAL);
        let e = ModelConfigError::UnsupportedArchitecture("detr".into());
        assert_eq!(config_errno(&e), libc::ENOTSUP);
        let e = ModelConfigError::Container("x".into());
        assert_eq!(config_errno(&e), libc::EBADMSG);
        let e = ModelConfigError::Unresolved {
            missing: "x".into(),
            overrides: "y".into(),
        };
        assert_eq!(config_errno(&e), libc::ENODATA);
        let e = ModelConfigError::Io {
            path: "x".into(),
            source: std::io::ErrorKind::IsADirectory.into(),
        };
        assert_eq!(config_errno(&e), libc::EISDIR);
    }
}
