// SPDX-FileCopyrightText: Copyright 2025 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

use edgefirst_decoder::{
    configs, configs::Nms, schema::SchemaV2, ConfigOutput, ConfigOutputs, Decoder, DecoderBuilder,
};
use edgefirst_tensor::{DetectBox, ProtoData, ProtoLayout, Segmentation};

/// NMS (Non-Maximum Suppression) mode for filtering overlapping detections.
///
/// - `ClassAgnostic` — suppress overlapping boxes regardless of class label
///   (default)
/// - `ClassAware` — only suppress boxes that share the same class label AND
///   overlap
///
/// Pass `None` to bypass NMS entirely (for end-to-end models with embedded
/// NMS).
#[pyo3::pyclass(name = "Nms", eq, eq_int, from_py_object)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PyNms {
    /// Suppress overlapping boxes regardless of class label (default)
    ClassAgnostic = 0,
    /// Only suppress boxes with the same class label that overlap
    ClassAware = 1,
    /// Use the model config (e.g. ``edgefirst.json``) to decide NMS mode,
    /// falling back to :attr:`ClassAgnostic` when no config specifies one.
    Auto = 2,
}

// Single-package (only `edgefirst.decoder` registers this type -- see its
// `lib.rs`), so `eq_int`'s native-or-bare-int richcmp above has no
// cross-package identity problem to fix. It was still unhashable (`eq`
// without `hash` -- Python's own rule for a class defining `__eq__`), which
// is independently worth fixing: hash the discriminant, matching both this
// enum's own equality and `int.__hash__`'s identity behaviour for small
// values, so `{Nms.ClassAgnostic: ...}` and `{Nms.ClassAgnostic, 0}` behave.
#[pyo3::pymethods]
impl PyNms {
    fn __hash__(&self) -> isize {
        *self as isize
    }
}

impl From<PyNms> for Nms {
    fn from(py: PyNms) -> Self {
        match py {
            PyNms::ClassAgnostic => Nms::ClassAgnostic,
            PyNms::ClassAware => Nms::ClassAware,
            PyNms::Auto => Nms::Auto,
        }
    }
}

impl From<Nms> for PyNms {
    fn from(nms: Nms) -> Self {
        match nms {
            Nms::ClassAgnostic => PyNms::ClassAgnostic,
            Nms::ClassAware => PyNms::ClassAware,
            Nms::Auto => PyNms::Auto,
        }
    }
}

/// Decoder type — selects the post-processing algorithm family.
#[pyo3::pyclass(name = "DecoderType", eq, eq_int, from_py_object)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PyDecoderType {
    /// Ultralytics YOLO models (YOLOv5, YOLOv8, YOLO11, YOLO26)
    Ultralytics = 0,
    /// ModelPack models
    ModelPack = 1,
}

/// Single-package; see `PyNms`'s `__hash__` comment for why this is still
/// worth fixing despite no cross-package equality problem.
#[pyo3::pymethods]
impl PyDecoderType {
    fn __hash__(&self) -> isize {
        *self as isize
    }
}

impl From<PyDecoderType> for configs::DecoderType {
    fn from(py: PyDecoderType) -> Self {
        match py {
            PyDecoderType::Ultralytics => configs::DecoderType::Ultralytics,
            PyDecoderType::ModelPack => configs::DecoderType::ModelPack,
        }
    }
}

impl From<configs::DecoderType> for PyDecoderType {
    fn from(dt: configs::DecoderType) -> Self {
        match dt {
            configs::DecoderType::Ultralytics => PyDecoderType::Ultralytics,
            configs::DecoderType::ModelPack => PyDecoderType::ModelPack,
        }
    }
}

/// Decoder version for Ultralytics models.
///
/// Specifies the YOLO architecture version, which determines the decoding
/// strategy:
/// - `Yolov5`, `Yolov8`, `Yolo11`: Traditional models requiring external NMS
/// - `Yolo26`: End-to-end models with NMS embedded in the model architecture
#[pyo3::pyclass(name = "DecoderVersion", eq, eq_int, from_py_object)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PyDecoderVersion {
    /// YOLOv5 - anchor-based decoder, requires external NMS
    Yolov5 = 0,
    /// YOLOv8 - anchor-free DFL decoder, requires external NMS
    Yolov8 = 1,
    /// YOLO11 - anchor-free DFL decoder, requires external NMS
    Yolo11 = 2,
    /// YOLO26 - end-to-end model with embedded NMS
    Yolo26 = 3,
}

/// Single-package; see `PyNms`'s `__hash__` comment.
#[pyo3::pymethods]
impl PyDecoderVersion {
    fn __hash__(&self) -> isize {
        *self as isize
    }
}

impl From<PyDecoderVersion> for configs::DecoderVersion {
    fn from(py: PyDecoderVersion) -> Self {
        match py {
            PyDecoderVersion::Yolov5 => configs::DecoderVersion::Yolov5,
            PyDecoderVersion::Yolov8 => configs::DecoderVersion::Yolov8,
            PyDecoderVersion::Yolo11 => configs::DecoderVersion::Yolo11,
            PyDecoderVersion::Yolo26 => configs::DecoderVersion::Yolo26,
        }
    }
}

impl From<configs::DecoderVersion> for PyDecoderVersion {
    fn from(dv: configs::DecoderVersion) -> Self {
        match dv {
            configs::DecoderVersion::Yolov5 => PyDecoderVersion::Yolov5,
            configs::DecoderVersion::Yolov8 => PyDecoderVersion::Yolov8,
            configs::DecoderVersion::Yolo11 => PyDecoderVersion::Yolo11,
            configs::DecoderVersion::Yolo26 => PyDecoderVersion::Yolo26,
        }
    }
}

/// Named dimension for model output tensors.
///
/// Used with `dshape` to give semantic meaning to each dimension,
/// enabling the decoder to validate and interpret the tensor layout.
#[pyo3::pyclass(name = "DimName", eq, eq_int, from_py_object)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PyDimName {
    /// Batch dimension (typically 1)
    Batch = 0,
    /// Spatial height
    Height = 1,
    /// Spatial width
    Width = 2,
    /// Number of object classes
    NumClasses = 3,
    /// Number of features per box (e.g. 4 coords + N classes)
    NumFeatures = 4,
    /// Number of candidate boxes / anchors
    NumBoxes = 5,
    /// Number of segmentation prototype channels
    NumProtos = 6,
    /// Product of anchors and features (ModelPack split format)
    NumAnchorsXFeatures = 7,
    /// Padding dimension
    Padding = 8,
    /// Box coordinate dimension (typically 4)
    BoxCoords = 9,
}

/// Single-package; see `PyNms`'s `__hash__` comment.
#[pyo3::pymethods]
impl PyDimName {
    fn __hash__(&self) -> isize {
        *self as isize
    }
}

impl From<PyDimName> for configs::DimName {
    fn from(py: PyDimName) -> Self {
        match py {
            PyDimName::Batch => configs::DimName::Batch,
            PyDimName::Height => configs::DimName::Height,
            PyDimName::Width => configs::DimName::Width,
            PyDimName::NumClasses => configs::DimName::NumClasses,
            PyDimName::NumFeatures => configs::DimName::NumFeatures,
            PyDimName::NumBoxes => configs::DimName::NumBoxes,
            PyDimName::NumProtos => configs::DimName::NumProtos,
            PyDimName::NumAnchorsXFeatures => configs::DimName::NumAnchorsXFeatures,
            PyDimName::Padding => configs::DimName::Padding,
            PyDimName::BoxCoords => configs::DimName::BoxCoords,
        }
    }
}

/// A model output configuration for programmatic decoder setup.
///
/// Use the static factory methods (`detection`, `boxes`, `scores`, etc.) to
/// create outputs, then pass them to `Decoder.new_from_outputs()`.
///
/// Shape can be specified as either:
/// - `shape`: anonymous integer dimensions (e.g. `[1, 25200, 85]`)
/// - `dshape`: named dimensions (e.g. `[(DimName.Batch, 1), ...]`)
///
/// Provide one or the other, not both. If `dshape` is provided, `shape` is
/// derived automatically.
#[pyclass(name = "Output", from_py_object, module = "edgefirst.decoder")]
#[derive(Debug, Clone)]
pub struct PyOutput {
    inner: ConfigOutput,
}

type ShapeDshape = (Vec<usize>, Vec<(configs::DimName, usize)>);

/// Helper: parse shape/dshape parameters and return (shape, dshape).
fn parse_shape_dshape(
    shape: Option<Vec<usize>>,
    dshape: Option<Vec<(PyDimName, usize)>>,
) -> PyResult<ShapeDshape> {
    match (shape, dshape) {
        (Some(_), Some(_)) => Err(pyo3::exceptions::PyValueError::new_err(
            "Provide either 'shape' or 'dshape', not both",
        )),
        (None, None) => Err(pyo3::exceptions::PyValueError::new_err(
            "Either 'shape' or 'dshape' must be provided",
        )),
        (Some(s), None) => Ok((s, Vec::new())),
        (None, Some(ds)) => {
            let dshape = ds
                .iter()
                .map(|(name, size)| ((*name).into(), *size))
                .collect();
            // shape left empty; DecoderBuilder::add_output() -> normalize_output()
            // will derive it from dshape.
            Ok((Vec::new(), dshape))
        }
    }
}

#[pymethods]
impl PyOutput {
    /// Create a detection output (combined boxes + scores in one tensor).
    #[staticmethod]
    #[pyo3(signature = (shape=None, dshape=None, decoder=PyDecoderType::Ultralytics))]
    fn detection(
        shape: Option<Vec<usize>>,
        dshape: Option<Vec<(PyDimName, usize)>>,
        decoder: PyDecoderType,
    ) -> PyResult<Self> {
        let (shape, dshape) = parse_shape_dshape(shape, dshape)?;
        Ok(Self {
            inner: ConfigOutput::Detection(configs::Detection {
                decoder: decoder.into(),
                shape,
                dshape,
                ..Default::default()
            }),
        })
    }

    /// Create a boxes-only output (split detection format).
    #[staticmethod]
    #[pyo3(signature = (shape=None, dshape=None, decoder=PyDecoderType::Ultralytics))]
    fn boxes(
        shape: Option<Vec<usize>>,
        dshape: Option<Vec<(PyDimName, usize)>>,
        decoder: PyDecoderType,
    ) -> PyResult<Self> {
        let (shape, dshape) = parse_shape_dshape(shape, dshape)?;
        Ok(Self {
            inner: ConfigOutput::Boxes(configs::Boxes {
                decoder: decoder.into(),
                shape,
                dshape,
                ..Default::default()
            }),
        })
    }

    /// Create a scores-only output (split detection format).
    #[staticmethod]
    #[pyo3(signature = (shape=None, dshape=None, decoder=PyDecoderType::Ultralytics))]
    fn scores(
        shape: Option<Vec<usize>>,
        dshape: Option<Vec<(PyDimName, usize)>>,
        decoder: PyDecoderType,
    ) -> PyResult<Self> {
        let (shape, dshape) = parse_shape_dshape(shape, dshape)?;
        Ok(Self {
            inner: ConfigOutput::Scores(configs::Scores {
                decoder: decoder.into(),
                shape,
                dshape,
                ..Default::default()
            }),
        })
    }

    /// Create a protos output (segmentation prototype tensor).
    #[staticmethod]
    #[pyo3(signature = (shape=None, dshape=None, decoder=PyDecoderType::Ultralytics))]
    fn protos(
        shape: Option<Vec<usize>>,
        dshape: Option<Vec<(PyDimName, usize)>>,
        decoder: PyDecoderType,
    ) -> PyResult<Self> {
        let (shape, dshape) = parse_shape_dshape(shape, dshape)?;
        Ok(Self {
            inner: ConfigOutput::Protos(configs::Protos {
                decoder: decoder.into(),
                shape,
                dshape,
                ..Default::default()
            }),
        })
    }

    /// Create a segmentation output.
    #[staticmethod]
    #[pyo3(signature = (shape=None, dshape=None, decoder=PyDecoderType::Ultralytics))]
    fn segmentation(
        shape: Option<Vec<usize>>,
        dshape: Option<Vec<(PyDimName, usize)>>,
        decoder: PyDecoderType,
    ) -> PyResult<Self> {
        let (shape, dshape) = parse_shape_dshape(shape, dshape)?;
        Ok(Self {
            inner: ConfigOutput::Segmentation(configs::Segmentation {
                decoder: decoder.into(),
                shape,
                dshape,
                ..Default::default()
            }),
        })
    }

    /// Create a mask coefficients output.
    #[staticmethod]
    #[pyo3(signature = (shape=None, dshape=None, decoder=PyDecoderType::Ultralytics))]
    fn mask_coefficients(
        shape: Option<Vec<usize>>,
        dshape: Option<Vec<(PyDimName, usize)>>,
        decoder: PyDecoderType,
    ) -> PyResult<Self> {
        let (shape, dshape) = parse_shape_dshape(shape, dshape)?;
        Ok(Self {
            inner: ConfigOutput::MaskCoefficients(configs::MaskCoefficients {
                decoder: decoder.into(),
                shape,
                dshape,
                ..Default::default()
            }),
        })
    }

    /// Create a classes output (class label indices for end-to-end split models).
    #[staticmethod]
    #[pyo3(signature = (shape=None, dshape=None, decoder=PyDecoderType::Ultralytics))]
    fn classes(
        shape: Option<Vec<usize>>,
        dshape: Option<Vec<(PyDimName, usize)>>,
        decoder: PyDecoderType,
    ) -> PyResult<Self> {
        let (shape, dshape) = parse_shape_dshape(shape, dshape)?;
        Ok(Self {
            inner: ConfigOutput::Classes(configs::Classes {
                decoder: decoder.into(),
                shape,
                dshape,
                ..Default::default()
            }),
        })
    }

    /// Create a mask output.
    #[staticmethod]
    #[pyo3(signature = (shape=None, dshape=None, decoder=PyDecoderType::Ultralytics))]
    fn mask(
        shape: Option<Vec<usize>>,
        dshape: Option<Vec<(PyDimName, usize)>>,
        decoder: PyDecoderType,
    ) -> PyResult<Self> {
        let (shape, dshape) = parse_shape_dshape(shape, dshape)?;
        Ok(Self {
            inner: ConfigOutput::Mask(configs::Mask {
                decoder: decoder.into(),
                shape,
                dshape,
                ..Default::default()
            }),
        })
    }

    /// Set quantization parameters for this output. Returns self for chaining.
    #[pyo3(signature = (scale, zero_point))]
    fn with_quantization(self_: Bound<'_, Self>, scale: f32, zero_point: i32) -> Bound<'_, Self> {
        let quant = Some(configs::QuantTuple(scale, zero_point));
        match &mut self_.borrow_mut().inner {
            ConfigOutput::Detection(c) => c.quantization = quant,
            ConfigOutput::Boxes(c) => c.quantization = quant,
            ConfigOutput::Scores(c) => c.quantization = quant,
            ConfigOutput::Protos(c) => c.quantization = quant,
            ConfigOutput::Segmentation(c) => c.quantization = quant,
            ConfigOutput::MaskCoefficients(c) => c.quantization = quant,
            ConfigOutput::Mask(c) => c.quantization = quant,
            ConfigOutput::Classes(c) => c.quantization = quant,
        }
        self_
    }

    /// Set anchors for this output (detection outputs only). Returns self for chaining.
    #[pyo3(signature = (anchors))]
    fn with_anchors(self_: Bound<'_, Self>, anchors: Vec<[f32; 2]>) -> PyResult<Bound<'_, Self>> {
        match &mut self_.borrow_mut().inner {
            ConfigOutput::Detection(c) => {
                c.anchors = Some(anchors);
                Ok(self_)
            }
            _ => Err(pyo3::exceptions::PyValueError::new_err(
                "with_anchors() is only valid for detection outputs",
            )),
        }
    }

    /// Set the normalized flag for this output (detection/boxes outputs only).
    /// Returns self for chaining.
    #[pyo3(signature = (normalized))]
    fn with_normalized(self_: Bound<'_, Self>, normalized: bool) -> PyResult<Bound<'_, Self>> {
        match &mut self_.borrow_mut().inner {
            ConfigOutput::Detection(c) => {
                c.normalized = Some(normalized);
                Ok(self_)
            }
            ConfigOutput::Boxes(c) => {
                c.normalized = Some(normalized);
                Ok(self_)
            }
            _ => Err(pyo3::exceptions::PyValueError::new_err(
                "with_normalized() is only valid for detection or boxes outputs",
            )),
        }
    }
}

use crate::detect_boxes::{convert_detect_box, convert_seg_mask, PyDetOutput};
use ndarray::{Array1, Array2};
use numpy::{IntoPyArray, PyArray1, PyArray2, PyArray3, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::types::{PyAnyMethods, PyDict, PyList};
use pyo3::{pyclass, pymethods, Bound, PyAny, PyErr, PyRef, PyResult, Python};

pub type PySegDetOutput<'py> = (
    Bound<'py, PyArray2<f32>>,
    Bound<'py, PyArray1<f32>>,
    Bound<'py, PyArray1<usize>>,
    Vec<Bound<'py, PyArray3<u8>>>,
);

pub type PySegDetTrackedOutput<'py> = (
    Bound<'py, PyArray2<f32>>,
    Bound<'py, PyArray1<f32>>,
    Bound<'py, PyArray1<usize>>,
    Vec<Bound<'py, PyArray3<u8>>>,
    Bound<'py, PyAny>,
);

type PyAssociatedDetections<'py> = (
    Vec<[f32; 4]>,
    Vec<f32>,
    Vec<usize>,
    Vec<Bound<'py, PyAny>>,
    Vec<usize>,
);

/// Decode to Rust boxes/masks. `self_` stays on the caller's stack so the
/// PyO3 borrow flag remains set across `py.detach` (see `decode`).
fn decode_native(
    decoder: &Decoder,
    py: Python<'_>,
    model_output: Vec<Bound<'_, PyAny>>,
    max_boxes: usize,
) -> PyResult<(Vec<DetectBox>, Vec<Segmentation>)> {
    let model_output: Vec<crate::interop::TensorArg> = model_output
        .iter()
        .map(|o| crate::interop::TensorArg::extract(o, None))
        .collect::<PyResult<_>>()?;
    let (mut output_boxes, mut output_masks) = if model_output
        .iter()
        .all(crate::interop::TensorArg::can_detach)
    {
        let raw_inputs: Vec<crate::interop::RawTensorAccess> = model_output
            .into_iter()
            .map(|t| t.into_raw_access())
            .collect::<PyResult<_>>()?;
        py.detach(move || {
            let tensor_refs: Vec<&edgefirst_tensor::TensorDyn> =
                raw_inputs.iter().map(|t| t.as_ref()).collect();
            let mut output_boxes = Vec::with_capacity(max_boxes);
            let mut output_masks = Vec::with_capacity(max_boxes);
            decoder
                .decode(&tensor_refs, &mut output_boxes, &mut output_masks)
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("{e:#?}")))?;
            Ok::<_, PyErr>((output_boxes, output_masks))
        })?
    } else {
        let tensor_refs: Vec<&edgefirst_tensor::TensorDyn> =
            model_output.iter().map(|t| t.as_ref()).collect();
        let mut output_boxes = Vec::with_capacity(max_boxes);
        let mut output_masks = Vec::with_capacity(max_boxes);
        decoder
            .decode(&tensor_refs, &mut output_boxes, &mut output_masks)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("{e:#?}")))?;
        (output_boxes, output_masks)
    };
    output_boxes.truncate(max_boxes);
    output_masks.truncate(max_boxes);
    Ok((output_boxes, output_masks))
}

/// Opaque prototype data from a segmentation model's decode step.
///
/// Holds raw mask coefficients and prototype tensors produced by
/// :meth:`Decoder.decode_proto`. Pass to
/// :meth:`ImageProcessor.materialize_masks` to compute per-instance masks
/// for analytics or export, or use :meth:`ImageProcessor.draw_masks` for
/// fused GPU rendering instead.
///
/// For detection-only models, :meth:`Decoder.decode_proto` returns ``None``
/// instead of a ``ProtoData`` instance.
#[pyclass(name = "ProtoData", module = "edgefirst.decoder")]
pub struct PyProtoData(pub(crate) ProtoData);

#[pymethods]
impl PyProtoData {
    /// Take ownership of the prototype masks tensor.
    ///
    /// Returns a Tensor whose shape depends on :attr:`layout`:
    ///
    /// - ``"nhwc"``: shape is ``(H, W, num_protos)``
    /// - ``"nchw"``: shape is ``(num_protos, H, W)``
    ///
    /// For quantized models, the returned tensor carries quantization metadata
    /// accessible via the ``quantization`` property.
    ///
    /// Consumes the proto data's ``protos`` field — subsequent calls
    /// return ``None``.
    fn take_protos(&mut self) -> Option<crate::tensor::PyTensor> {
        let taken = std::mem::replace(&mut self.0.protos, empty_sentinel_tensor_dyn());
        if is_empty_sentinel(&taken) {
            self.0.protos = taken;
            return None;
        }
        Some(crate::tensor::PyTensor(taken))
    }

    /// Take ownership of the per-detection mask coefficients tensor.
    ///
    /// Returns a Tensor with shape ``(num_detections, num_protos)``.
    ///
    /// Consumes the proto data's ``mask_coefficients`` field — subsequent
    /// calls return ``None``.
    fn take_mask_coefficients(&mut self) -> Option<crate::tensor::PyTensor> {
        let taken = std::mem::replace(&mut self.0.mask_coefficients, empty_sentinel_tensor_dyn());
        if is_empty_sentinel(&taken) {
            self.0.mask_coefficients = taken;
            return None;
        }
        Some(crate::tensor::PyTensor(taken))
    }

    /// Physical memory layout of the prototype tensor.
    ///
    /// Returns ``"nhwc"`` when protos shape is ``(H, W, K)`` or ``"nchw"``
    /// when shape is ``(K, H, W)``. Use this to interpret the tensor returned
    /// by :meth:`take_protos`.
    #[getter]
    fn layout(&self) -> &'static str {
        match self.0.layout {
            ProtoLayout::Nhwc => "nhwc",
            ProtoLayout::Nchw => "nchw",
        }
    }

    /// Producer half of the cross-package ``ProtoData`` protocol.
    ///
    /// Composes the existing ``__edgefirst_tensor__`` capsule protocol
    /// rather than describing its own layout: a ``ProtoData`` is just two
    /// tensors and an enum, and both tensors already cross packages safely
    /// through that protocol (see ``interop::TensorArg``). Returns the
    /// ``mask_coefficients`` and ``protos`` tensors as ``edgefirst_tensor_v1``
    /// capsules -- exactly what ``Tensor.__edgefirst_tensor__()`` would
    /// return for each -- plus the prototype layout as a string. No raw
    /// pointer to ``ProtoData`` itself is ever exchanged, so unlike
    /// ``Decoder.__edgefirst_decoder__`` this protocol carries no version
    /// coupling between packages: it is sound by construction.
    ///
    /// Both tensors are pinned for host reads: ``materialize_masks`` always
    /// computes ``mask_coeff @ protos`` on the CPU, so a consumer needs an
    /// addressable pointer, not just a GPU/DMA handle.
    fn __edgefirst_protodata__<'py>(
        &self,
        py: Python<'py>,
    ) -> PyResult<(
        Bound<'py, pyo3::types::PyCapsule>,
        Bound<'py, pyo3::types::PyCapsule>,
        &'static str,
    )> {
        let mask_coefficients = tensor_capsule(py, &self.0.mask_coefficients)?;
        let protos = tensor_capsule(py, &self.0.protos)?;
        let layout = match self.0.layout {
            ProtoLayout::Nhwc => "nhwc",
            ProtoLayout::Nchw => "nchw",
        };
        Ok((mask_coefficients, protos, layout))
    }
}

/// Build an ``edgefirst_tensor_v1`` capsule for `tensor`, pinned for host
/// reads. Shared by [`PyProtoData::__edgefirst_protodata__`]; the payload
/// shape (`crate::interop::TensorCapsulePayload`) is exactly what
/// `PyTensor::__edgefirst_tensor__` produces, so any
/// ``__edgefirst_tensor__`` consumer (i.e. `interop::TensorArg`) imports it
/// unmodified.
fn tensor_capsule<'py>(
    py: Python<'py>,
    tensor: &edgefirst_tensor::TensorDyn,
) -> PyResult<Bound<'py, pyo3::types::PyCapsule>> {
    use edgefirst_tensor::CpuAccess;
    let pin = tensor
        .pin_host(CpuAccess::Read)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("{e:#?}")))?;
    let desc = tensor.descriptor_pinned(Some(&pin));
    let payload = crate::interop::TensorCapsulePayload {
        desc,
        pin: Some(pin),
        // `pin_host` above already succeeded, and PBO refuses `pin_host`
        // outright -- so `tensor` is never PBO-backed here in practice, but
        // asking is free and keeps this in sync with `PyTensor::
        // __edgefirst_tensor__`'s shape rather than hand-assuming `None`.
        pbo_keepalive: tensor.pbo_keepalive(),
    };
    pyo3::types::PyCapsule::new_with_value(py, payload, c"edgefirst_tensor_v1").map_err(|e| {
        pyo3::exceptions::PyRuntimeError::new_err(format!("failed to build tensor capsule: {e}"))
    })
}

fn empty_sentinel_tensor_dyn() -> edgefirst_tensor::TensorDyn {
    use edgefirst_tensor::{Tensor, TensorMemory};
    let t = Tensor::<u8>::new(&[0], Some(TensorMemory::Mem), Some("__taken__"))
        .expect("sentinel allocation never fails");
    t.into()
}

fn is_empty_sentinel(t: &edgefirst_tensor::TensorDyn) -> bool {
    t.dtype() == edgefirst_tensor::DType::U8 && t.shape() == [0] && t.name() == "__taken__"
}

/// ``(boxes, scores, classes, proto_data)`` where ``proto_data`` is ``None``
/// for detection-only models.
pub type PyProtoDetOutput<'py> = (
    Bound<'py, PyArray2<f32>>,
    Bound<'py, PyArray1<f32>>,
    Bound<'py, PyArray1<usize>>,
    Option<PyProtoData>,
);

#[pyclass(name = "Decoder", module = "edgefirst.decoder")]
pub struct PyDecoder {
    pub(crate) decoder: Decoder,
}

unsafe impl Send for PyDecoder {}
unsafe impl Sync for PyDecoder {}

// `Decoder` needs no `RawDecoderAccess`-style resolved-argument type the way
// `TensorArg`/`RawTensorAccess` do: `decode`/`decode_proto` below borrow
// `&Decoder` straight from `self_: PyRef<'py, Self>`, which they keep alive
// on their own stack frame for the whole `py.detach` region rather than
// converting into anything `'static`-ish. See `decode`'s comment at the
// borrow site for why that -- not dropping the runtime borrow early, the
// way an earlier revision of this function did -- is what soundness here
// actually requires.

#[pymethods]
impl PyDecoder {
    /// Create a new Decoder from a configuration dictionary.
    ///
    /// Args:
    ///     config: Model output configuration dictionary.
    ///     score_threshold: Score threshold for filtering detections (default
    ///         ``0.1``).
    ///     iou_threshold: IoU threshold for NMS (default ``0.7``).
    ///     nms: NMS mode - ``Nms.Auto`` (default, uses config or
    ///         ``ClassAgnostic``), ``Nms.ClassAgnostic``, ``Nms.ClassAware``,
    ///         or ``None`` to bypass NMS.
    ///     input_dims: Optional ``(width, height)`` model input override
    ///         consumed by the EDGEAI-1303 normalization path. When set,
    ///         takes precedence over schema-derived dims; pass when building
    ///         from a config that does not declare an input shape but the
    ///         model emits pixel-space boxes (``Detection.normalized = False``).
    #[new]
    #[pyo3(signature = (config, score_threshold=0.1, iou_threshold=0.7, nms=PyNms::Auto, input_dims=None))]
    pub fn new(
        config: Bound<PyAny>,
        score_threshold: f32,
        iou_threshold: f32,
        nms: Option<PyNms>,
        input_dims: Option<(usize, usize)>,
    ) -> PyResult<Self> {
        let nms: Option<Nms> = nms.map(Into::into);
        let mut builder = DecoderBuilder::default()
            .with_score_threshold(score_threshold)
            .with_iou_threshold(iou_threshold)
            .with_nms(nms);
        if let Some((w, h)) = input_dims {
            builder = builder.with_input_dims(w, h);
        }

        // EDGEAI-1081: discriminate v2 vs legacy on the authoritative
        // `schema_version` field. v2 dicts carry object-form quantization
        // and spec-vocabulary type tags that the legacy `ConfigOutputs`
        // deserialiser rejects; v1 (or version-less) dicts continue
        // through the legacy path unchanged.
        let value: serde_json::Value = pythonize::depythonize(&config)?;
        let schema_version: Option<u32> = value
            .get("schema_version")
            .cloned()
            .map(serde_json::from_value::<u32>)
            .transpose()
            .map_err(|e| {
                pyo3::exceptions::PyRuntimeError::new_err(format!("invalid schema_version: {e}"))
            })?;

        let decoder = if schema_version.is_some_and(|v| v >= 2) {
            let schema = SchemaV2::from_json_value(value)
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("{e:#?}")))?;
            builder.with_schema(schema).build()
        } else {
            let legacy: ConfigOutputs = serde_json::from_value(value)
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("{e}")))?;
            builder.with_config(legacy).build()
        };

        match decoder {
            Ok(decoder) => Ok(Self { decoder }),
            Err(e) => Err(pyo3::exceptions::PyRuntimeError::new_err(format!("{e:#?}"))),
        }
    }

    /// Create a new Decoder from a list of Output objects.
    ///
    /// The default thresholds (0.25 / 0.45) are tuned for typical YOLO models.
    /// The dict/JSON/YAML constructors use lower defaults (0.1 / 0.7) for
    /// backward compatibility.
    ///
    /// Args:
    ///     outputs: List of Output objects describing the model outputs.
    ///     score_threshold: Score threshold for filtering detections (default
    ///         ``0.25``).
    ///     iou_threshold: IoU threshold for NMS (default ``0.45``).
    ///     nms: NMS mode - ``Nms.ClassAgnostic`` (default), ``Nms.ClassAware``,
    ///         or ``None`` to bypass NMS.
    ///     decoder_version: Optional decoder version for Ultralytics models.
    ///     input_dims: Optional ``(width, height)`` model input override.
    ///         See :meth:`__init__` for semantics.
    #[staticmethod]
    #[pyo3(signature = (outputs, score_threshold=0.25, iou_threshold=0.45, nms=PyNms::Auto, decoder_version=None, input_dims=None))]
    pub fn new_from_outputs(
        outputs: Vec<PyRef<PyOutput>>,
        score_threshold: f32,
        iou_threshold: f32,
        nms: Option<PyNms>,
        decoder_version: Option<PyDecoderVersion>,
        input_dims: Option<(usize, usize)>,
    ) -> PyResult<Self> {
        let nms: Option<Nms> = nms.map(Into::into);
        let mut builder = DecoderBuilder::default()
            .with_score_threshold(score_threshold)
            .with_iou_threshold(iou_threshold)
            .with_nms(nms);
        for output in outputs {
            builder = builder.add_output(output.inner.clone());
        }
        if let Some(version) = decoder_version {
            builder = builder.with_decoder_version(version.into());
        }
        if let Some((w, h)) = input_dims {
            builder = builder.with_input_dims(w, h);
        }
        match builder.build() {
            Ok(decoder) => Ok(Self { decoder }),
            Err(e) => Err(pyo3::exceptions::PyRuntimeError::new_err(format!("{e:#?}"))),
        }
    }

    /// Create a new Decoder from a JSON configuration string.
    ///
    /// Args:
    ///     json_str: JSON-encoded model configuration (v1 or v2 schema).
    ///     score_threshold: Score threshold for filtering detections (default
    ///         ``0.1``).
    ///     iou_threshold: IoU threshold for NMS (default ``0.7``).
    ///     nms: NMS mode - ``Nms.Auto`` (default, uses config or
    ///         ``ClassAgnostic``), ``Nms.ClassAgnostic``, ``Nms.ClassAware``,
    ///         or ``None`` to bypass NMS.
    ///     input_dims: Optional ``(width, height)`` model input override.
    ///         See :meth:`__init__` for semantics.
    #[staticmethod]
    #[pyo3(signature = (json_str, score_threshold=0.1, iou_threshold=0.7, nms=PyNms::Auto, input_dims=None))]
    pub fn new_from_json_str(
        json_str: &str,
        score_threshold: f32,
        iou_threshold: f32,
        nms: Option<PyNms>,
        input_dims: Option<(usize, usize)>,
    ) -> PyResult<Self> {
        let nms: Option<Nms> = nms.map(Into::into);
        let mut builder = DecoderBuilder::default()
            .with_score_threshold(score_threshold)
            .with_iou_threshold(iou_threshold)
            .with_nms(nms)
            .with_config_json_str(json_str.to_string());
        if let Some((w, h)) = input_dims {
            builder = builder.with_input_dims(w, h);
        }
        match builder.build() {
            Ok(decoder) => Ok(Self { decoder }),
            Err(e) => Err(pyo3::exceptions::PyRuntimeError::new_err(format!("{e:#?}"))),
        }
    }

    /// Create a new Decoder from a YAML configuration string.
    ///
    /// Args:
    ///     yaml_str: YAML-encoded model configuration (v1 or v2 schema).
    ///     score_threshold: Score threshold for filtering detections (default
    ///         ``0.1``).
    ///     iou_threshold: IoU threshold for NMS (default ``0.7``).
    ///     nms: NMS mode - ``Nms.Auto`` (default, uses config or
    ///         ``ClassAgnostic``), ``Nms.ClassAgnostic``, ``Nms.ClassAware``,
    ///         or ``None`` to bypass NMS.
    ///     input_dims: Optional ``(width, height)`` model input override.
    ///         See :meth:`__init__` for semantics.
    #[staticmethod]
    #[pyo3(signature = (yaml_str, score_threshold=0.1, iou_threshold=0.7, nms=PyNms::Auto, input_dims=None))]
    pub fn new_from_yaml_str(
        yaml_str: &str,
        score_threshold: f32,
        iou_threshold: f32,
        nms: Option<PyNms>,
        input_dims: Option<(usize, usize)>,
    ) -> PyResult<Self> {
        let nms: Option<Nms> = nms.map(Into::into);
        let mut builder = DecoderBuilder::default()
            .with_score_threshold(score_threshold)
            .with_iou_threshold(iou_threshold)
            .with_nms(nms)
            .with_config_yaml_str(yaml_str.to_string());
        if let Some((w, h)) = input_dims {
            builder = builder.with_input_dims(w, h);
        }
        match builder.build() {
            Ok(decoder) => Ok(Self { decoder }),
            Err(e) => Err(pyo3::exceptions::PyRuntimeError::new_err(format!("{e:#?}"))),
        }
    }

    /// Decode model outputs into ``(boxes, scores, classes, masks)`` tuples.
    ///
    /// Args:
    ///     model_output: List of output :class:`Tensor` from model inference.
    ///     max_boxes: Per-call **post-truncate cap** layered on top of the
    ///         decoder's own :attr:`max_det` (default 300; set by assigning
    ///         to :attr:`max_det` on this instance). The smaller of the two
    ///         wins. ``max_boxes`` also pre-allocates the underlying box /
    ///         mask buffers; the Rust decoder does not use buffer capacity
    ///         as a semantic cap (EDGEAI-1302). Default ``100``. Also see
    ///         :attr:`max_det` for the decoder-side cap and
    ///         :attr:`pre_nms_top_k` for the pre-NMS candidate cap.
    ///
    /// Returns:
    ///     Tuple ``(boxes, scores, classes, masks)`` where ``boxes`` is a
    ///     ``(N, 4)`` ``float32`` array of ``[xmin, ymin, xmax, ymax]``
    ///     coords, ``scores`` is ``(N,)`` ``float32``, and ``classes`` is
    ///     ``(N,)`` ``uintp``. Consult :attr:`normalized_boxes` for the
    ///     coordinate space of ``boxes``.
    ///
    ///     ``masks`` is a list of N ``(H, W, C)`` ``uint8`` arrays at
    ///     prototype resolution, empty for detection-only models. Instance
    ///     segmentation gives ``C == 1`` (binary, threshold at 128);
    ///     semantic segmentation gives ``C == num_classes`` (per-pixel
    ///     scores, take ``argmax`` over the last axis).
    #[pyo3(signature = (model_output, max_boxes=100))]
    pub fn decode<'py>(
        self_: PyRef<'py, Self>,
        model_output: Vec<Bound<'py, PyAny>>,
        max_boxes: usize,
    ) -> PyResult<PySegDetOutput<'py>> {
        let _span =
            tracing::trace_span!("python.decode", n_tensors = model_output.len(), max_boxes)
                .entered();
        let py = self_.py();
        // Model outputs are read here, never written. `self_` stays on this
        // stack so the PyO3 borrow flag remains set across `py.detach`
        // inside `decode_native` — a concurrent setter would otherwise
        // alias `&mut Decoder` (see `TensorArg::into_raw_access`).
        let (output_boxes, output_masks) =
            decode_native(&self_.decoder, py, model_output, max_boxes)?;
        let (boxes, scores, classes) = convert_detect_box(py, &output_boxes);
        let masks = convert_seg_mask(py, &output_masks);
        Ok((boxes, scores, classes, masks))
    }

    #[pyo3(signature = (tracker, timestamp, model_output, max_boxes=100))]
    pub fn decode_tracked<'py>(
        self_: PyRef<'py, Self>,
        tracker: &Bound<'py, PyAny>,
        timestamp: u64,
        model_output: Vec<Bound<'py, PyAny>>,
        max_boxes: usize,
    ) -> PyResult<PySegDetTrackedOutput<'py>> {
        let py = self_.py();
        let (output_boxes, output_masks) =
            decode_native(&self_.decoder, py, model_output, max_boxes)?;

        // ByteTrack._associate_detections runs Kalman + the live/coasting
        // rewrite off the GIL. Duck-typed trackers without that method keep
        // the Python `update` / getattr rewrite path.
        let boxes: Vec<[f32; 4]> = output_boxes
            .iter()
            .map(|b| <[f32; 4]>::from(b.bbox))
            .collect();
        let scores: Vec<f32> = output_boxes.iter().map(|b| b.score).collect();
        let classes: Vec<usize> = output_boxes.iter().map(|b| b.label).collect();
        if let Ok(assoc) = tracker.getattr("_associate_detections") {
            let result = assoc.call1((boxes, scores, classes, timestamp))?;
            let (out_boxes, out_scores, out_classes, track_objs, mask_idx): PyAssociatedDetections<
                'py,
            > = result.extract()?;
            let mut out_masks = Vec::new();
            for i in mask_idx {
                if i < output_masks.len() {
                    out_masks.extend(convert_seg_mask(py, std::slice::from_ref(&output_masks[i])));
                }
            }
            let num = out_scores.len();
            let boxes_flat: Vec<f32> = out_boxes.into_iter().flatten().collect();
            let boxes_arr = Array2::from_shape_vec((num, 4), boxes_flat).map_err(|e| {
                pyo3::exceptions::PyRuntimeError::new_err(format!("tracked boxes reshape: {e}"))
            })?;
            let tracks_out = PyList::new(py, track_objs)?;
            return Ok((
                boxes_arr.into_pyarray(py),
                Array1::from_vec(out_scores).into_pyarray(py),
                Array1::from_vec(out_classes).into_pyarray(py),
                out_masks,
                tracks_out.into_any(),
            ));
        }

        let (boxes, scores, classes) = convert_detect_box(py, &output_boxes);
        let masks = convert_seg_mask(py, &output_masks);
        let update_tracks =
            tracker.call_method1("update", (&boxes, &scores, &classes, timestamp))?;

        let Ok(active) = tracker.call_method0("get_active_tracks") else {
            return Ok((boxes, scores, classes, masks, update_tracks));
        };

        let b: PyReadonlyArray2<f32> = boxes.extract()?;
        let s: PyReadonlyArray1<f32> = scores.extract()?;
        let c: PyReadonlyArray1<usize> = classes.extract()?;
        let b = b.as_array();
        let s = s.as_array();
        let c = c.as_array();
        let n = b.shape()[0];

        let mut out_boxes: Vec<f32> = Vec::new();
        let mut out_scores: Vec<f32> = Vec::new();
        let mut out_classes: Vec<usize> = Vec::new();
        let mut out_masks: Vec<Bound<'py, PyArray3<u8>>> = Vec::new();
        let mut out_track_objs: Vec<Bound<'py, PyAny>> = Vec::new();

        for (i, item) in update_tracks.try_iter()?.enumerate() {
            let item = item?;
            if item.is_none() || i >= n {
                continue;
            }
            let loc: [f32; 4] = item.getattr("tracked_location")?.extract()?;
            out_boxes.extend_from_slice(&loc);
            out_scores.push(s[i]);
            out_classes.push(c[i]);
            if i < masks.len() {
                out_masks.push(masks[i].clone());
            }
            out_track_objs.push(item);
        }

        for item in active.try_iter()? {
            let item = item?;
            let info = item.getattr("info")?;
            let last_updated: u64 = info.getattr("last_updated")?.extract()?;
            if last_updated == timestamp {
                continue;
            }
            let loc: [f32; 4] = info.getattr("tracked_location")?.extract()?;
            let (_bbox, score, label): ([f32; 4], f32, usize) =
                item.getattr("last_box")?.extract()?;
            out_boxes.extend_from_slice(&loc);
            out_scores.push(score);
            out_classes.push(label);
            out_track_objs.push(info);
        }

        let num = out_scores.len();
        let boxes_arr = Array2::from_shape_vec((num, 4), out_boxes).map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(format!("tracked boxes reshape: {e}"))
        })?;
        let scores_arr = Array1::from_vec(out_scores);
        let classes_arr = Array1::from_vec(out_classes);
        let tracks_out = PyList::new(py, out_track_objs)?;
        Ok((
            boxes_arr.into_pyarray(py),
            scores_arr.into_pyarray(py),
            classes_arr.into_pyarray(py),
            out_masks,
            tracks_out.into_any(),
        ))
    }

    /// Decode model outputs and draw onto ``processor`` (an
    /// ``edgefirst.image.ImageProcessor``) without this extension linking
    /// image. Prefers ``draw_proto_masks`` when the model yields prototype
    /// data; otherwise ``draw_decoded_masks``.
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (processor, model_output, dst, background=None, opacity=1.0, letterbox=None, color_mode=None))]
    pub fn draw_onto<'py>(
        self_: PyRef<'py, Self>,
        processor: &Bound<'py, PyAny>,
        model_output: Vec<Bound<'py, PyAny>>,
        dst: &Bound<'py, PyAny>,
        background: Option<&Bound<'py, PyAny>>,
        opacity: f32,
        letterbox: Option<[f32; 4]>,
        color_mode: Option<&Bound<'py, PyAny>>,
        py: Python<'py>,
    ) -> PyResult<PyDetOutput<'py>> {
        let (boxes, scores, classes, proto) = Self::decode_proto(self_, model_output, 100)?;
        let kwargs = PyDict::new(py);
        kwargs.set_item("opacity", opacity)?;
        if let Some(bg) = background {
            kwargs.set_item("background", bg)?;
        }
        if let Some(lb) = letterbox {
            kwargs.set_item("letterbox", lb)?;
        }
        if let Some(cm) = color_mode {
            kwargs.set_item("color_mode", cm)?;
        }
        if let Some(proto) = proto {
            processor.call_method(
                "draw_proto_masks",
                (dst, &boxes, &scores, &classes, proto),
                Some(&kwargs),
            )?;
        } else {
            let empty_masks: Vec<Bound<'py, PyAny>> = Vec::new();
            processor.call_method(
                "draw_decoded_masks",
                (dst, &boxes, &scores, &classes, empty_masks),
                Some(&kwargs),
            )?;
        }
        Ok((boxes, scores, classes))
    }

    /// Decode model outputs into detection boxes and optional prototype data.
    ///
    /// For segmentation models, returns a :class:`ProtoData` instance that can
    /// be passed to :meth:`ImageProcessor.materialize_masks` to compute
    /// per-instance masks for analytics, export, or IoU computation. For
    /// detection-only models, returns ``None`` for ``proto_data`` but still
    /// populates detection boxes.
    ///
    /// Note:
    ///     Calling ``decode_proto`` + ``materialize_masks`` +
    ///     ``draw_decoded_masks`` separately prevents the HAL from using its
    ///     internal fused optimization. For render-only use cases, prefer
    ///     :meth:`Decoder.draw_onto` which is 1.6–27× faster on tested
    ///     platforms.
    ///
    /// Args:
    ///     model_output: List of output :class:`Tensor` from model inference.
    ///     max_boxes: Pre-allocates ``output_boxes`` capacity. The actual
    ///         detection-count cap is the decoder's :attr:`max_det` (default
    ///         300); the Rust decoder does not use buffer capacity as a
    ///         semantic cap (EDGEAI-1302). Default ``100``.
    ///
    /// Returns:
    ///     Tuple ``(boxes, scores, classes, proto_data)`` where ``proto_data``
    ///     is ``None`` for detection-only models. ``boxes`` is ``(N, 4)``
    ///     ``numpy.ndarray``, ``scores`` is ``(N,)`` ``float32``, ``classes``
    ///     is ``(N,)`` ``int64``.
    #[pyo3(signature = (model_output, max_boxes=100))]
    pub fn decode_proto<'py>(
        self_: PyRef<'py, Self>,
        model_output: Vec<Bound<'py, PyAny>>,
        max_boxes: usize,
    ) -> PyResult<PyProtoDetOutput<'py>> {
        let _span = tracing::trace_span!(
            "python.decode_proto",
            n_tensors = model_output.len(),
            max_boxes,
        )
        .entered();
        let py = self_.py();
        // Model outputs are read here, never written.
        let model_output: Vec<crate::interop::TensorArg> = model_output
            .iter()
            .map(|o| crate::interop::TensorArg::extract(o, None))
            .collect::<PyResult<_>>()?;
        let (output_boxes, proto_data) = if model_output
            .iter()
            .all(crate::interop::TensorArg::can_detach)
        {
            // See `decode`: every input tensor's Python guard is gone by
            // this point, and `self_`'s own guard is kept alive right here
            // -- not moved into the closure -- for the whole detached
            // region, so PyO3's borrow flag stays set and a concurrent
            // `&mut self` setter errors instead of aliasing.
            let raw_inputs: Vec<crate::interop::RawTensorAccess> = model_output
                .into_iter()
                .map(|t| t.into_raw_access())
                .collect::<PyResult<_>>()?;
            let decoder: &Decoder = &self_.decoder;
            py.detach(move || {
                let tensor_refs: Vec<&edgefirst_tensor::TensorDyn> =
                    raw_inputs.iter().map(|t| t.as_ref()).collect();
                let mut output_boxes = Vec::with_capacity(max_boxes);
                let proto_data = decoder
                    .decode_proto(&tensor_refs, &mut output_boxes)
                    .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("{e:#?}")))?;
                Ok::<_, PyErr>((output_boxes, proto_data))
            })?
        } else {
            // See `decode`: a GL-PBO-backed model output tensor keeps
            // the GIL held.
            let tensor_refs: Vec<&edgefirst_tensor::TensorDyn> =
                model_output.iter().map(|t| t.as_ref()).collect();
            let mut output_boxes = Vec::with_capacity(max_boxes);
            let proto_data = self_
                .decoder
                .decode_proto(&tensor_refs, &mut output_boxes)
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("{e:#?}")))?;
            (output_boxes, proto_data)
        };
        // Note: output_boxes and proto_data.mask_coefficients must stay in sync
        // (same row count). Truncation here would break materialize_masks.
        // The decoder's max_det (default 300) already caps output count.
        let (boxes, scores, classes) = convert_detect_box(py, &output_boxes);
        Ok((boxes, scores, classes, proto_data.map(PyProtoData)))
    }

    #[getter(score_threshold)]
    fn get_score_threshold(&self) -> PyResult<f32> {
        Ok(self.decoder.score_threshold)
    }

    #[setter(score_threshold)]
    fn set_score_threshold(&mut self, value: f32) -> PyResult<()> {
        self.decoder.score_threshold = value;
        Ok(())
    }

    #[getter(iou_threshold)]
    fn get_iou_threshold(&self) -> PyResult<f32> {
        Ok(self.decoder.iou_threshold)
    }

    #[setter(iou_threshold)]
    fn set_iou_threshold(&mut self, value: f32) -> PyResult<()> {
        self.decoder.iou_threshold = value;
        Ok(())
    }

    /// Get the NMS mode.
    /// Returns Nms.ClassAgnostic, Nms.ClassAware, or None if NMS is bypassed.
    #[getter(nms)]
    fn get_nms(&self) -> Option<PyNms> {
        self.decoder.nms.map(|nms| nms.into())
    }

    /// Maximum number of candidates fed into NMS after score filtering.
    /// Uses O(N) partial sort to reduce O(N²) NMS cost. Default: 300.
    ///
    /// .. warning::
    ///
    ///    The default of 300 is tuned for **deployment** (``score_threshold >= 0.25``)
    ///    where few anchors pass the score filter. For **COCO mAP evaluation**
    ///    (``score_threshold = 0.001``), set this to the total anchor count
    ///    (8400 for 640×640 YOLO models) or to ``0`` (no limit) to avoid
    ///    discarding ~74% of valid candidates before NMS, which causes
    ///    **~9 pp box mAP loss**.
    ///
    ///    Deployment::
    ///
    ///        decoder.score_threshold = 0.25
    ///        # decoder.pre_nms_top_k = 300  (default, appropriate)
    ///
    ///    COCO mAP evaluation::
    ///
    ///        decoder.score_threshold = 0.001
    ///        decoder.pre_nms_top_k = 8400   # all anchors
    ///        decoder.max_det = 300
    ///
    ///    Post-processing latency scales with candidate count. At deployment
    ///    thresholds the cost difference is negligible; at validation thresholds
    ///    it is measurable but necessary for correct recall.
    #[getter(pre_nms_top_k)]
    fn get_pre_nms_top_k(&self) -> usize {
        self.decoder.pre_nms_top_k
    }

    #[setter(pre_nms_top_k)]
    fn set_pre_nms_top_k(&mut self, value: usize) -> PyResult<()> {
        self.decoder.pre_nms_top_k = value;
        Ok(())
    }

    /// Maximum number of detections returned after NMS.
    /// Matches the Ultralytics max_det parameter. Default: 300.
    #[getter(max_det)]
    fn get_max_det(&self) -> usize {
        self.decoder.max_det
    }

    #[setter(max_det)]
    fn set_max_det(&mut self, value: usize) -> PyResult<()> {
        self.decoder.max_det = value;
        Ok(())
    }

    /// Returns the coordinate format of the boxes the decoder emits to
    /// the caller.
    ///
    /// - ``True``: Boxes are in normalized ``[0, 1]`` coordinates
    /// - ``False``: Boxes are in pixel coordinates relative to model input
    /// - ``None``: Unknown, caller must infer (e.g., check if any
    ///   coordinate > 1.0)
    ///
    /// Four decode paths invoke the normalization helper uniformly across
    /// all entry points (``decode``, ``decode_proto``, ``decode_tracked``,
    /// ``decode_tracked_proto``, both quantized and float). For these,
    /// this getter reports the post-decode coordinate space rather than
    /// the raw schema annotation:
    ///
    /// - **Per-scale decoders**: the bridge always divides by ``(W, H)``
    ///   before returning.
    /// - **Combined-output segmentation** models: the helper fires across
    ///   all entry points and element-type variants.
    /// - **Split-output segmentation** models: aligned across all four
    ///   entry points for both quantized and float variants.
    /// - **Two-way segmentation** models: same four entry points and both
    ///   element-type variants.
    ///
    /// For all four paths, when the schema declares ``normalized: false``
    /// and :attr:`input_dims` is a valid ``(W, H)`` tuple, the decoder
    /// has already divided and returns ``True``. When :attr:`input_dims`
    /// is ``None`` or zero, pixel-space leaks out and returns ``False``.
    ///
    /// **All other decoders** — detection-only, end-to-end YOLO, and
    /// ModelPack — return the raw schema annotation. Callers that receive
    /// ``False`` from these model types must consult :attr:`input_dims`
    /// and divide themselves if ``[0, 1]`` output is required.
    ///
    /// Callers must not re-normalize when this returns ``True``; dividing
    /// already-normalized coordinates by ``(W, H)`` collapses detections
    /// to ~0.
    #[getter(normalized_boxes)]
    fn get_normalized_boxes(&self) -> Option<bool> {
        self.decoder.normalized_boxes()
    }

    /// Model input dimensions ``(width, height)`` consumed by the
    /// EDGEAI-1303 normalization path.
    ///
    /// Set to a non-``None`` value via the ``input_dims`` constructor
    /// kwarg, or sourced from the schema's ``input.shape`` /
    /// ``input.dshape`` when building from a v2 schema. On the per-scale
    /// path and on the combined, split, and two-way segmentation paths,
    /// when the schema declares pixel-space outputs and ``input_dims`` is
    /// a valid tuple, the decoder divides post-NMS box coordinates by
    /// ``(W, H)`` so they enter the canonical ``[0, 1]`` range before
    /// mask cropping;
    /// :attr:`normalized_boxes` then reports ``True`` to match. All other
    /// decode paths (detection-only, end-to-end YOLO, ModelPack) do not
    /// apply this division — see :attr:`normalized_boxes` for the
    /// per-path contract. When ``None``, the four uniform-normalization
    /// paths skip division and pixel-space boxes will trip the
    /// ``protobox`` safety guard.
    #[getter(input_dims)]
    fn get_input_dims(&self) -> Option<(usize, usize)> {
        self.decoder.input_dims()
    }

    /// Producer half of the cross-package decoder protocol.
    ///
    /// The capsule borrows this decoder: it is valid only for the duration
    /// of the call it is passed into, and must not be stored.
    ///
    /// # Safety hazard
    ///
    /// Unlike every other ``__edgefirst_*__`` producer in this crate (see
    /// [`PyProtoData::__edgefirst_protodata__`], `Tensor.__edgefirst_tensor__`),
    /// this one has no self-describing fallback: a `Decoder` is a live Rust
    /// object carrying internal post-processing state, not a value that can
    /// be decomposed into tensors and enums. The capsule therefore carries a
    /// raw pointer, and the consumer's `unsafe { &*ptr }` is sound only if
    /// the producer's and consumer's copies of `edgefirst-decoder` agree
    /// bit-for-bit on `Decoder`'s memory layout.
    ///
    /// The guard (`interop::DecoderArg::extract`) checks
    /// `size_of::<Decoder>()`/`align_of::<Decoder>()` equality, not version
    /// string equality. Two reasons:
    ///
    /// - A version string says nothing about Cargo *features*. `Decoder`'s
    ///   fields can be cfg-gated by features `edgefirst-decoder` exposes
    ///   (e.g. `tracker`), and the two wheels can genuinely compute that
    ///   feature set through different dependency graphs
    ///   (`crates/image/Cargo.toml`'s `decode`/`tracker` features flow into
    ///   `edgefirst-decoder` differently than `edgefirst-python-decoder`'s
    ///   do). Comparing `size_of`/`align_of` catches that regardless of
    ///   which features produced the mismatch; enumerating features by name
    ///   is a list someone would forget to update.
    /// - `crates/python-image/pyproject.toml` pins `edgefirst-decoder` with a
    ///   `~=` compatible-release specifier, which admits patch releases against
    ///   the version it was built for. Exact version-string equality would turn
    ///   a pip-legal install into a hard `RuntimeError`; gating on layout
    ///   instead lets a layout-compatible patch release work while still
    ///   catching any layout change regardless of version.
    ///
    /// No version string rides along in the payload: an earlier revision
    /// carried one as diagnostic-only text, but it was this crate's
    /// (`edgefirst-python-common`'s) own `CARGO_PKG_VERSION`, not
    /// `edgefirst-decoder`'s, and the error message named the wrong crate --
    /// dropped along with the `&'static str` fat-pointer-in-`#[repr(C)]`
    /// hazard it carried. The mismatch error below reports `size`/`align`
    /// only, which is what the guard actually evaluated.
    ///
    /// This is accepted risk, not a solved problem: two `Decoder` layouts
    /// of equal size and alignment but permuted field order would still
    /// pass the guard undetected. `size_of`/`align_of` narrows the failure
    /// mode from "any layout drift" to "a layout drift that also changes
    /// size or alignment"; it does not eliminate it.
    fn __edgefirst_decoder__<'py>(
        &self,
        py: Python<'py>,
    ) -> PyResult<Bound<'py, pyo3::types::PyCapsule>> {
        let ptr: *const edgefirst_decoder::Decoder = &self.decoder;
        let payload = crate::interop::DecoderCapsulePayload {
            ptr: ptr as usize,
            decoder_size: std::mem::size_of::<edgefirst_decoder::Decoder>(),
            decoder_align: std::mem::align_of::<edgefirst_decoder::Decoder>(),
        };
        pyo3::types::PyCapsule::new_with_value(py, payload, c"edgefirst_decoder_v1").map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(format!(
                "failed to build decoder capsule: {e}"
            ))
        })
    }
}
