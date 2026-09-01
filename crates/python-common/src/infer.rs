// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! Python binding for Ultralytics schema inference from raw model I/O
//! signals. Registered on `edgefirst.decoder`.

use std::collections::BTreeMap;
use std::fmt;

use edgefirst_decoder::schema::{DType, Quantization};
use edgefirst_decoder::{infer_ultralytics_schema, ModelSignals, ModelSource, TensorInfo};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;

/// Maps a Python dtype string to `schema::DType`. A distinct, narrower
/// vocabulary from `edgefirst_tensor::DType` -- see `decoder-capi`'s
/// `dtype_from` for the same split at the C boundary.
fn dtype_from_str(s: &str) -> PyResult<DType> {
    match s {
        "int8" => Ok(DType::Int8),
        "uint8" => Ok(DType::Uint8),
        "int16" => Ok(DType::Int16),
        "uint16" => Ok(DType::Uint16),
        "int32" => Ok(DType::Int32),
        "uint32" => Ok(DType::Uint32),
        "float16" => Ok(DType::Float16),
        "float32" => Ok(DType::Float32),
        other => Err(PyValueError::new_err(format!(
            "unknown dtype `{other}` (expected one of: int8, uint8, int16, \
             uint16, int32, uint32, float16, float32)"
        ))),
    }
}

/// Maps a Python source string to `ModelSource`.
fn source_from_str(s: &str) -> PyResult<ModelSource> {
    match s {
        "onnx" => Ok(ModelSource::Onnx),
        "tflite" => Ok(ModelSource::TfLite),
        "other" => Ok(ModelSource::Other),
        other => Err(PyValueError::new_err(format!(
            "unknown source `{other}` (expected one of: onnx, tflite, other)"
        ))),
    }
}

/// An output tensor's optional per-tensor quantization, `(scales,
/// zero_points)`.
type QuantArg = (Vec<f32>, Vec<i32>);

/// One reported shape dimension.
///
/// A dynamic export does not report an integer for every axis: ONNX
/// exported with `dynamic=True` reports the *symbolic name* (the string
/// `"batch"`) and TFLite reports `-1`. Both are accepted into this enum
/// so the refusal is a `ValueError` naming the tensor and the axis,
/// rather than the `TypeError`/`OverflowError` a bare `Vec<usize>`
/// parameter produced from inside PyO3's extraction machinery.
#[derive(FromPyObject)]
pub enum Dim {
    Concrete(i64),
    Symbolic(String),
}

impl fmt::Display for Dim {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Dim::Concrete(v) => write!(f, "{v}"),
            Dim::Symbolic(s) => write!(f, "`{s}`"),
        }
    }
}

/// Converts a tensor's reported shape to concrete dimensions, refusing a
/// dynamic axis by name.
fn concrete_shape(name: &str, shape: Vec<Dim>) -> PyResult<Vec<usize>> {
    shape
        .into_iter()
        .enumerate()
        .map(|(axis, dim)| {
            let concrete = match dim {
                Dim::Concrete(v) => usize::try_from(v).ok(),
                Dim::Symbolic(_) => None,
            };
            concrete.ok_or_else(|| {
                PyValueError::new_err(format!(
                    "tensor `{name}` has a dynamic dimension ({dim}) on \
                     axis {axis}; schema inference needs concrete shapes, \
                     so re-export the model with a fixed input size"
                ))
            })
        })
        .collect()
}

/// A tensor spec, accepted as either a 3-tuple `(name, shape, dtype)` or a
/// 4-tuple with an explicit quantization slot. Both `inputs` and `outputs`
/// take this shape, so a caller can build them from the same runtime
/// description without remembering which argument allows which arity. `#[pyo3
/// (transparent)]` extracts each variant's tuple field directly from the
/// Python object rather than treating the variant itself as a 1-tuple; Rust's
/// built-in tuple `FromPyObject` impl then requires an exact arity match, so
/// a 3-element Python tuple fails `WithQuant` (needs 4) and falls through to
/// `Bare` -- a plain 3-tuple and an explicit `None` 4th element both mean
/// "unquantized".
#[derive(FromPyObject)]
pub enum TensorArg {
    #[pyo3(transparent)]
    WithQuant((String, Vec<Dim>, String, Option<QuantArg>)),
    #[pyo3(transparent)]
    Bare((String, Vec<Dim>, String)),
}

impl TensorArg {
    fn into_parts(self) -> (String, Vec<Dim>, String, Option<QuantArg>) {
        match self {
            TensorArg::WithQuant(t) => t,
            TensorArg::Bare((name, shape, dtype)) => (name, shape, dtype, None),
        }
    }
}

/// Builds a `TensorInfo` from a tensor spec's parts, mapping the dtype
/// string and (when present) folding `quant` into a `Quantization`.
fn tensor_info(
    name: String,
    shape: Vec<Dim>,
    dtype: &str,
    quant: Option<QuantArg>,
) -> PyResult<TensorInfo> {
    let shape = concrete_shape(&name, shape)?;
    let dtype = dtype_from_str(dtype)?;
    let quantization = quant.map(|(scale, zero_point)| Quantization {
        scale,
        zero_point: Some(zero_point),
        axis: None,
        dtype: Some(dtype),
    });
    Ok(TensorInfo {
        name,
        shape,
        dtype,
        quantization,
    })
}

/// Infers an Ultralytics YOLO schema from raw model I/O signals and
/// metadata. Supports detection and segmentation, both Ultralytics v8/v11
/// pre-NMS heads and YOLO26 end-to-end heads.
///
/// :param source: Container format the signals were read from: ``"onnx"``,
///     ``"tflite"``, or ``"other"``.
/// :param inputs: Input tensors as ``(name, shape, dtype)``, or
///     ``(name, shape, dtype, quantization)`` -- an input's quantization is
///     accepted for symmetry with ``outputs`` and ignored.
/// :param outputs: Output tensors as ``(name, shape, dtype)`` for an
///     unquantized tensor, or ``(name, shape, dtype, quantization)`` where
///     ``quantization`` is ``(scales, zero_points)`` or ``None``. Only
///     per-tensor quantization (one scale, one zero point) is usable: the
///     decoder consumes per-tensor only, so more than one scale is
///     rejected rather than turned into a schema that cannot build.
/// :param metadata: Raw model metadata key/values, passed through verbatim
///     (ONNX ``metadata_props``, or the TFLite ``metadata.json`` envelope
///     under whichever key it was captured).
/// :returns: ``(schema, labels, description)``: the inferred schema as an
///     ``edgefirst.json`` schema v2 dict ready for ``Decoder(schema)``,
///     class names in index order, and a human-readable summary (e.g.
///     ``"Ultralytics YOLOv8/11 detect, 80 classes"``). The result is a
///     named tuple (``InferredSchema``), so the fields can be read by name
///     or unpacked positionally.
/// :raises ValueError: the signals carry no recognizable Ultralytics schema
///     (bad/missing metadata, an unsupported task, a class-count mismatch,
///     an unrecognized output layout, or per-channel quantization), or a
///     dtype/source string isn't one of the values listed above.
#[pyfunction]
#[pyo3(name = "infer_ultralytics_schema")]
pub fn py_infer_ultralytics_schema(
    source: &str,
    inputs: Vec<TensorArg>,
    outputs: Vec<TensorArg>,
    metadata: BTreeMap<String, String>,
) -> PyResult<Py<PyAny>> {
    let source = source_from_str(source)?;
    let inputs = inputs
        .into_iter()
        .map(|t| {
            // A model input's own quantization plays no part in
            // output-layout inference, so the slot is accepted and ignored
            // rather than rejected -- callers build input and output specs
            // from the same runtime dict, and an arity difference between
            // the two arguments is a trap, not a safeguard.
            let (name, shape, dtype, _) = t.into_parts();
            tensor_info(name, shape, &dtype, None)
        })
        .collect::<PyResult<Vec<_>>>()?;
    let outputs = outputs
        .into_iter()
        .map(|o| {
            let (name, shape, dtype, quant) = o.into_parts();
            tensor_info(name, shape, &dtype, quant)
        })
        .collect::<PyResult<Vec<_>>>()?;

    let signals = ModelSignals {
        source,
        inputs,
        outputs,
        metadata,
    };

    let result =
        infer_ultralytics_schema(&signals).map_err(|e| PyValueError::new_err(e.to_string()))?;

    // The schema crosses as a dict, not a JSON string: `Decoder(config)`
    // takes a dict directly, so a string would only be re-parsed by every
    // caller. The C binding still renders JSON, because C has no dict.
    Python::attach(|py| {
        let schema = pythonize::pythonize(py, &result.schema).map_err(|e| {
            PyRuntimeError::new_err(format!("failed to convert inferred schema: {e}"))
        })?;
        let named = py
            .import("edgefirst.decoder")?
            .getattr("InferredSchema")?
            .call1((schema, result.labels, result.description))?;
        Ok(named.unbind())
    })
}
