// SPDX-FileCopyrightText: Copyright 2025 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

use crate::tracker::{PyByteTrack, PyTrackInfo};
use edgefirst_hal::decoder::{
    configs, configs::Nms, schema::SchemaV2, ConfigOutput, ConfigOutputs, Decoder, DecoderBuilder,
    DetectBox, ProtoData, Segmentation,
};

/// NMS (Non-Maximum Suppression) mode for filtering overlapping detections.
///
/// - `ClassAgnostic` — suppress overlapping boxes regardless of class label
///   (default)
/// - `ClassAware` — only suppress boxes that share the same class label AND
///   overlap
///
/// Pass `None` to bypass NMS entirely (for end-to-end models with embedded
/// NMS).
#[pyo3::pyclass(name = "Nms", eq, eq_int)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PyNms {
    /// Suppress overlapping boxes regardless of class label (default)
    ClassAgnostic = 0,
    /// Only suppress boxes with the same class label that overlap
    ClassAware = 1,
}

impl From<PyNms> for Nms {
    fn from(py: PyNms) -> Self {
        match py {
            PyNms::ClassAgnostic => Nms::ClassAgnostic,
            PyNms::ClassAware => Nms::ClassAware,
        }
    }
}

impl From<Nms> for PyNms {
    fn from(nms: Nms) -> Self {
        match nms {
            Nms::ClassAgnostic => PyNms::ClassAgnostic,
            Nms::ClassAware => PyNms::ClassAware,
        }
    }
}

/// Decoder type — selects the post-processing algorithm family.
#[pyo3::pyclass(name = "DecoderType", eq, eq_int)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PyDecoderType {
    /// Ultralytics YOLO models (YOLOv5, YOLOv8, YOLO11, YOLO26)
    Ultralytics = 0,
    /// ModelPack models
    ModelPack = 1,
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
#[pyo3::pyclass(name = "DecoderVersion", eq, eq_int)]
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
#[pyo3::pyclass(name = "DimName", eq, eq_int)]
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

impl From<configs::DimName> for PyDimName {
    fn from(dn: configs::DimName) -> Self {
        match dn {
            configs::DimName::Batch => PyDimName::Batch,
            configs::DimName::Height => PyDimName::Height,
            configs::DimName::Width => PyDimName::Width,
            configs::DimName::NumClasses => PyDimName::NumClasses,
            configs::DimName::NumFeatures => PyDimName::NumFeatures,
            configs::DimName::NumBoxes => PyDimName::NumBoxes,
            configs::DimName::NumProtos => PyDimName::NumProtos,
            configs::DimName::NumAnchorsXFeatures => PyDimName::NumAnchorsXFeatures,
            configs::DimName::Padding => PyDimName::Padding,
            configs::DimName::BoxCoords => PyDimName::BoxCoords,
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
#[pyclass(name = "Output")]
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

use ndarray::{Array1, Array2};
use numpy::{IntoPyArray, PyArray1, PyArray2, PyArray3, ToPyArray};
use pyo3::{pyclass, pymethods, Bound, PyAny, PyRef, PyResult, Python};

pub type PyDetOutput<'py> = (
    Bound<'py, PyArray2<f32>>,
    Bound<'py, PyArray1<f32>>,
    Bound<'py, PyArray1<usize>>,
);

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
    Vec<PyTrackInfo>,
);

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
#[pyclass(name = "ProtoData")]
pub struct PyProtoData(pub(crate) ProtoData);

#[pymethods]
impl PyProtoData {
    /// Take ownership of the prototype masks tensor.
    ///
    /// Returns a Tensor with shape ``(H, W, num_protos)``. For quantized
    /// models, the returned tensor carries quantization metadata
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
}

fn empty_sentinel_tensor_dyn() -> edgefirst_hal::tensor::TensorDyn {
    use edgefirst_hal::tensor::{Tensor, TensorDyn, TensorMemory};
    let t = Tensor::<u8>::new(&[0], Some(TensorMemory::Mem), Some("__taken__"))
        .expect("sentinel allocation never fails");
    TensorDyn::U8(t)
}

fn is_empty_sentinel(t: &edgefirst_hal::tensor::TensorDyn) -> bool {
    use edgefirst_hal::tensor::TensorDyn;
    matches!(t, TensorDyn::U8(_)) && t.shape() == [0] && t.name() == "__taken__"
}

/// ``(boxes, scores, classes, proto_data)`` where ``proto_data`` is ``None``
/// for detection-only models.
pub type PyProtoDetOutput<'py> = (
    Bound<'py, PyArray2<f32>>,
    Bound<'py, PyArray1<f32>>,
    Bound<'py, PyArray1<usize>>,
    Option<PyProtoData>,
);

#[pyclass(name = "Decoder")]
pub struct PyDecoder {
    pub(crate) decoder: Decoder,
}

unsafe impl Send for PyDecoder {}
unsafe impl Sync for PyDecoder {}

#[pymethods]
impl PyDecoder {
    /// Create a new Decoder from a configuration dictionary.
    ///
    /// Args:
    ///     config: Model output configuration dictionary
    ///     score_threshold: Score threshold for filtering detections (default:
    /// 0.1)     iou_threshold: IoU threshold for NMS (default: 0.7)
    ///     nms: NMS mode - Nms.ClassAgnostic (default), Nms.ClassAware, or None
    /// to bypass NMS
    #[new]
    #[pyo3(signature = (config, score_threshold=0.1, iou_threshold=0.7, nms=PyNms::ClassAgnostic,))]
    pub fn new(
        config: Bound<PyAny>,
        score_threshold: f32,
        iou_threshold: f32,
        nms: Option<PyNms>,
    ) -> PyResult<Self> {
        let nms: Option<Nms> = nms.map(|py_nms| py_nms.into());
        let builder = DecoderBuilder::default()
            .with_score_threshold(score_threshold)
            .with_iou_threshold(iou_threshold)
            .with_nms(nms);

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
    ///     score_threshold: Score threshold for filtering detections (default:
    /// 0.25)     iou_threshold: IoU threshold for NMS (default: 0.45)
    ///     nms: NMS mode - Nms.ClassAgnostic (default), Nms.ClassAware, or None
    /// to bypass NMS     decoder_version: Optional decoder version for
    /// Ultralytics models
    #[staticmethod]
    #[pyo3(signature = (outputs, score_threshold=0.25, iou_threshold=0.45, nms=PyNms::ClassAgnostic, decoder_version=None))]
    pub fn new_from_outputs(
        outputs: Vec<PyRef<PyOutput>>,
        score_threshold: f32,
        iou_threshold: f32,
        nms: Option<PyNms>,
        decoder_version: Option<PyDecoderVersion>,
    ) -> PyResult<Self> {
        let nms: Option<Nms> = nms.map(|py_nms| py_nms.into());
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
        match builder.build() {
            Ok(decoder) => Ok(Self { decoder }),
            Err(e) => Err(pyo3::exceptions::PyRuntimeError::new_err(format!("{e:#?}"))),
        }
    }

    #[staticmethod]
    #[pyo3(signature = (json_str, score_threshold=0.1, iou_threshold=0.7, nms=PyNms::ClassAgnostic))]
    pub fn new_from_json_str(
        json_str: &str,
        score_threshold: f32,
        iou_threshold: f32,
        nms: Option<PyNms>,
    ) -> PyResult<Self> {
        let nms: Option<Nms> = nms.map(|py_nms| py_nms.into());
        let decoder = DecoderBuilder::default()
            .with_score_threshold(score_threshold)
            .with_iou_threshold(iou_threshold)
            .with_nms(nms)
            .with_config_json_str(json_str.to_string())
            .build();
        match decoder {
            Ok(decoder) => Ok(Self { decoder }),
            Err(e) => Err(pyo3::exceptions::PyRuntimeError::new_err(format!("{e:#?}"))),
        }
    }

    #[staticmethod]
    #[pyo3(signature = (yaml_str, score_threshold=0.1, iou_threshold=0.7, nms=PyNms::ClassAgnostic))]
    pub fn new_from_yaml_str(
        yaml_str: &str,
        score_threshold: f32,
        iou_threshold: f32,
        nms: Option<PyNms>,
    ) -> PyResult<Self> {
        let nms: Option<Nms> = nms.map(|py_nms| py_nms.into());
        let decoder = DecoderBuilder::default()
            .with_score_threshold(score_threshold)
            .with_iou_threshold(iou_threshold)
            .with_nms(nms)
            .with_config_yaml_str(yaml_str.to_string())
            .build();
        match decoder {
            Ok(decoder) => Ok(Self { decoder }),
            Err(e) => Err(pyo3::exceptions::PyRuntimeError::new_err(format!("{e:#?}"))),
        }
    }

    #[pyo3(signature = (model_output, max_boxes=100))]
    pub fn decode<'py>(
        self_: PyRef<'py, Self>,
        model_output: Vec<PyRef<'py, crate::tensor::PyTensor>>,
        max_boxes: usize,
    ) -> PyResult<PySegDetOutput<'py>> {
        let tensor_refs: Vec<&edgefirst_hal::tensor::TensorDyn> =
            model_output.iter().map(|t| &t.0).collect();
        let mut output_boxes = Vec::with_capacity(max_boxes);
        let mut output_masks = Vec::with_capacity(max_boxes);
        self_
            .decoder
            .decode(&tensor_refs, &mut output_boxes, &mut output_masks)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("{e:#?}")))?;
        let py = self_.py();
        let (boxes, scores, classes) = convert_detect_box(py, &output_boxes);
        let masks = convert_seg_mask(py, &output_masks);
        Ok((boxes, scores, classes, masks))
    }

    #[pyo3(signature = (tracker, timestamp, model_output, max_boxes=100))]
    pub fn decode_tracked<'py>(
        self_: PyRef<'py, Self>,
        tracker: &mut PyByteTrack,
        timestamp: u64,
        model_output: Vec<PyRef<'py, crate::tensor::PyTensor>>,
        max_boxes: usize,
    ) -> PyResult<PySegDetTrackedOutput<'py>> {
        let tensor_refs: Vec<&edgefirst_hal::tensor::TensorDyn> =
            model_output.iter().map(|t| &t.0).collect();
        let mut output_boxes = Vec::with_capacity(max_boxes);
        let mut output_masks = Vec::with_capacity(max_boxes);
        let mut output_tracks = Vec::with_capacity(max_boxes);
        self_
            .decoder
            .decode_tracked(
                tracker,
                timestamp,
                &tensor_refs,
                &mut output_boxes,
                &mut output_masks,
                &mut output_tracks,
            )
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("{e:#?}")))?;
        let py = self_.py();
        let (boxes, scores, classes) = convert_detect_box(py, &output_boxes);
        let masks = convert_seg_mask(py, &output_masks);
        let tracks = output_tracks.into_iter().map(|t| t.into()).collect();
        Ok((boxes, scores, classes, masks, tracks))
    }

    /// Decode model outputs into detection boxes and optional prototype data.
    ///
    /// For segmentation models, returns a :class:`ProtoData` instance that can
    /// be passed to :meth:`ImageProcessor.materialize_masks` to compute
    /// per-instance masks for analytics, export, or IoU computation.
    ///
    /// For detection-only models, returns ``None`` for proto_data but still
    /// populates detection boxes.
    ///
    /// .. note::
    ///
    ///     Calling ``decode_proto`` + ``materialize_masks`` +
    ///     ``draw_decoded_masks`` separately prevents the HAL from using its
    ///     internal fused optimization. For render-only use cases, prefer
    ///     :meth:`ImageProcessor.draw_masks` which is 1.6–27× faster on tested
    ///     platforms.
    ///
    /// :param model_output: list of output :class:`Tensor` from model inference
    /// :param max_boxes: maximum number of detections to return
    /// :returns: ``(boxes, scores, classes, proto_data)`` where ``proto_data``
    ///     is ``None`` for detection-only models
    /// :rtype: tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray, ProtoData | None]
    #[pyo3(signature = (model_output, max_boxes=100))]
    pub fn decode_proto<'py>(
        self_: PyRef<'py, Self>,
        model_output: Vec<PyRef<'py, crate::tensor::PyTensor>>,
        max_boxes: usize,
    ) -> PyResult<PyProtoDetOutput<'py>> {
        let tensor_refs: Vec<&edgefirst_hal::tensor::TensorDyn> =
            model_output.iter().map(|t| &t.0).collect();
        let mut output_boxes = Vec::with_capacity(max_boxes);
        let proto_data = self_
            .decoder
            .decode_proto(&tensor_refs, &mut output_boxes)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("{e:#?}")))?;
        let py = self_.py();
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
    /// Uses O(N) partial sort to reduce O(N²) NMS cost. Default: 3000.
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

    /// Returns the box coordinate format if known from the model config.
    ///
    /// - `True`: Boxes are in normalized [0,1] coordinates
    /// - `False`: Boxes are in pixel coordinates relative to model input
    /// - `None`: Unknown, caller must infer (e.g., check if any coordinate >
    ///   1.0)
    ///
    /// This is determined by the model config's `normalized` field, not the NMS
    /// mode.
    #[getter(normalized_boxes)]
    fn get_normalized_boxes(&self) -> Option<bool> {
        self.decoder.normalized_boxes()
    }
}

/// Helpers for converting decoder output to Python arrays (not exposed to Python).
pub(crate) fn convert_detect_box<'py>(
    py: Python<'py>,
    output_boxes: &[DetectBox],
) -> PyDetOutput<'py> {
    let boxes = output_boxes
        .iter()
        .flat_map(|b| <[f32; 4]>::from(b.bbox))
        .collect::<Vec<_>>();
    let scores = output_boxes.iter().map(|b| b.score).collect::<Vec<_>>();
    let classes = output_boxes.iter().map(|b| b.label).collect::<Vec<_>>();
    let num_boxes = output_boxes.len();
    let boxes = Array2::from_shape_vec((num_boxes, 4), boxes).unwrap();
    let scores = Array1::from_vec(scores);
    let classes = Array1::from_vec(classes);
    (
        boxes.into_pyarray(py),
        scores.into_pyarray(py),
        classes.into_pyarray(py),
    )
}

pub(crate) fn convert_seg_mask<'py>(
    py: Python<'py>,
    output_masks: &[Segmentation],
) -> Vec<Bound<'py, PyArray3<u8>>> {
    output_masks
        .iter()
        .map(|x| x.segmentation.to_pyarray(py))
        .collect()
}
