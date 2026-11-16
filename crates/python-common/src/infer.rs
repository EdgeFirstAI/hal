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

/// Maps a Python dtype string to `schema::DType`: the tensor dtype names
/// (`"u8"`, `"f32"`, ...) or the `"uint8"`, `"float32"`, ... spellings.
fn dtype_from_str(s: &str) -> PyResult<DType> {
    DType::from_str_code(s).ok_or_else(|| {
        let expected: Vec<&str> = DType::accepted_names().collect();
        PyValueError::new_err(format!(
            "unknown dtype `{s}` (expected one of: {})",
            expected.join(", ")
        ))
    })
}

/// Maps a Python source string to `ModelSource`.
fn source_from_str(s: &str) -> PyResult<ModelSource> {
    ModelSource::from_str_code(s).ok_or_else(|| {
        let expected: Vec<&str> = ModelSource::all().iter().map(|m| m.as_str()).collect();
        PyValueError::new_err(format!(
            "unknown source `{s}` (expected one of: {})",
            expected.join(", ")
        ))
    })
}

/// An output tensor's optional per-tensor quantization, `(scales,
/// zero_points)`.
type QuantArg = (Vec<f32>, Vec<i32>);

/// One reported shape dimension: an integer, or the symbolic name a dynamic
/// export reports. A symbolic or negative dimension is refused with a
/// `ValueError` naming the tensor and the axis.
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

/// A tensor spec: a 3-tuple `(name, shape, dtype)` or a 4-tuple with a
/// quantization slot. A 3-tuple and a 4-tuple whose last element is `None`
/// both mean "unquantized".
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

/// Builds `ModelSignals` from the Python argument forms of
/// :func:`infer_ultralytics_schema`.
pub(crate) fn model_signals(
    source: &str,
    inputs: Vec<TensorArg>,
    outputs: Vec<TensorArg>,
    metadata: BTreeMap<String, String>,
) -> PyResult<ModelSignals> {
    let source = source_from_str(source)?;
    let inputs = inputs
        .into_iter()
        .map(|t| {
            // An input's quantization slot is accepted and ignored.
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
    Ok(ModelSignals {
        source,
        inputs,
        outputs,
        metadata,
    })
}

/// Builds `ModelSignals` from an `edgefirst.decoder.ModelSignals` or any
/// object with `source`, `inputs`, `outputs` and (optionally) `metadata`.
pub(crate) fn model_signals_from_object(obj: &Bound<'_, PyAny>) -> PyResult<ModelSignals> {
    let source: String = obj.getattr("source")?.extract()?;
    let inputs: Vec<TensorArg> = obj.getattr("inputs")?.extract()?;
    let outputs: Vec<TensorArg> = obj.getattr("outputs")?.extract()?;
    let metadata: Option<BTreeMap<String, String>> = match obj.getattr_opt("metadata")? {
        Some(m) => m.extract()?,
        None => None,
    };
    model_signals(&source, inputs, outputs, metadata.unwrap_or_default())
}

/// Infers an Ultralytics YOLO schema from raw model I/O signals and
/// metadata. Supports detection and segmentation, both Ultralytics v8/v11
/// pre-NMS heads and YOLO26 end-to-end heads.
///
/// :param source: Container format the signals were read from: ``"onnx"``,
///     ``"tflite"``, ``"coreml"``, or ``"other"``. CoreML resolves
///     pixel-space box coordinates as ONNX does, for the no-NMS anchor-grid
///     export only; a ``nms=True`` CoreML export is refused.
/// :param inputs: Input tensors as ``(name, shape, dtype)``, or
///     ``(name, shape, dtype, quantization)``; an input's quantization is
///     accepted and ignored.
/// :param outputs: Output tensors as ``(name, shape, dtype)`` for an
///     unquantized tensor, or ``(name, shape, dtype, quantization)`` where
///     ``quantization`` is ``(scales, zero_points)`` or ``None``. Only
///     per-tensor quantization (one scale, one zero point) is accepted; more
///     than one scale is rejected.
/// :param metadata: Raw model metadata key/values, passed through verbatim
///     (ONNX ``metadata_props``, the TFLite ``metadata.json`` envelope
///     under whichever key it was captured, or CoreML's own metadata
///     alongside Ultralytics' props).
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
    let signals = model_signals(source, inputs, outputs, metadata)?;

    let result =
        infer_ultralytics_schema(&signals).map_err(|e| PyValueError::new_err(e.to_string()))?;

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
