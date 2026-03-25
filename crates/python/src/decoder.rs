// SPDX-FileCopyrightText: Copyright 2025 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

use edgefirst_hal::decoder::{
    configs, configs::Nms, dequantize_cpu, modelpack::ModelPackDetectionConfig,
    segmentation_to_mask, ConfigOutput, Decoder, DecoderBuilder, DetectBox, Quantization,
    Segmentation,
};
use edgefirst_hal::image::ImageProcessorTrait;

use crate::image::PyImageProcessor;
use crate::tensor::PyTensor;

use crate::tracker::{PyByteTrack, PyTrackInfo};

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
use numpy::{
    IntoPyArray, PyArray1, PyArray2, PyArray3, PyArrayDyn, PyArrayLikeDyn, PyArrayMethods,
    PyReadonlyArray2, PyReadonlyArray3, PyReadonlyArrayDyn, PyReadwriteArrayDyn, PyUntypedArray,
    ToPyArray,
};
use pyo3::{
    pyclass, pymethods, types::PyAnyMethods, Bound, FromPyObject, PyAny, PyRef, PyResult, Python,
};

use crate::FunctionTimer;
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

#[derive(FromPyObject)]
pub enum ArrayQuantized<'a> {
    UInt8(PyReadonlyArrayDyn<'a, u8>),
    Int8(PyReadonlyArrayDyn<'a, i8>),
    UInt16(PyReadonlyArrayDyn<'a, u16>),
    Int16(PyReadonlyArrayDyn<'a, i16>),
    UInt32(PyReadonlyArrayDyn<'a, u32>),
    Int32(PyReadonlyArrayDyn<'a, i32>),
}

#[derive(FromPyObject)]
pub enum ListOfReadOnlyArrayGenericDyn<'py> {
    Quantized(Vec<ArrayQuantized<'py>>),
    Float16(Vec<WithInt32Array<'py, PyArrayF16_<'py>>>),
    Float32(Vec<WithInt32Array<'py, PyArrayLikeDyn<'py, f32>>>),
    Float64(Vec<WithInt32Array<'py, PyArrayLikeDyn<'py, f64>>>),
}

pub struct PyArrayF16_<'py> {
    pub arr: PyReadonlyArrayDyn<'py, half::f16>,
}

impl<'py> FromPyObject<'py> for PyArrayF16_<'py> {
    fn extract_bound(ob: &Bound<'py, PyAny>) -> PyResult<Self> {
        if let Ok(array) = ob.downcast::<PyArrayDyn<half::f16>>() {
            return Ok(Self {
                arr: array.readonly(),
            });
        }
        Err(pyo3::exceptions::PyRuntimeError::new_err(
            "Could not parse array as f16 numpy array".to_string(),
        ))
    }
}

#[derive(FromPyObject)]
pub enum WithInt32Array<'py, T>
where
    T: FromPyObject<'py>,
{
    Val(T),
    Int32(PyArrayLikeDyn<'py, i32>),
}

#[derive(FromPyObject)]
pub enum ListOfReadOnlyArrayGeneric3<'py> {
    UInt8(Vec<PyReadonlyArray3<'py, u8>>),
    Int8(Vec<PyReadonlyArray3<'py, i8>>),
    Float32(Vec<PyReadonlyArray3<'py, f32>>),
    Float64(Vec<PyReadonlyArray3<'py, f64>>),
}

#[derive(FromPyObject)]
pub enum ReadOnlyArrayGeneric2<'py> {
    UInt8(PyReadonlyArray2<'py, u8>),
    Int8(PyReadonlyArray2<'py, i8>),
    Float32(PyReadonlyArray2<'py, f32>),
    Float64(PyReadonlyArray2<'py, f64>),
}

#[derive(FromPyObject)]
pub enum ReadOnlyArrayGeneric3<'py> {
    UInt8(PyReadonlyArray3<'py, u8>),
    Int8(PyReadonlyArray3<'py, i8>),
    Float32(PyReadonlyArray3<'py, f32>),
    Float64(PyReadonlyArray3<'py, f64>),
}

// #[derive(FromPy<PyAny>)]
pub enum ReadOnlyArrayGenericQuantized<'a> {
    UInt8(PyReadonlyArrayDyn<'a, u8>),
    Int8(PyReadonlyArrayDyn<'a, i8>),
    UInt16(PyReadonlyArrayDyn<'a, u16>),
    Int16(PyReadonlyArrayDyn<'a, i16>),
    UInt32(PyReadonlyArrayDyn<'a, u32>),
    Int32(PyReadonlyArrayDyn<'a, i32>),
}

impl<'py> FromPyObject<'py> for ReadOnlyArrayGenericQuantized<'py> {
    fn extract_bound(ob: &Bound<'py, PyAny>) -> PyResult<Self> {
        let _time = FunctionTimer::new("ReadOnlyArrayGenericQuantized FromPyObject".to_string());
        let untyped: &Bound<'_, PyUntypedArray>;
        if let Ok(array) = ob.downcast::<PyUntypedArray>() {
            untyped = array;
        } else {
            return Err(pyo3::exceptions::PyRuntimeError::new_err(
                "Could not parse array as u8, i8, u16, i16, u32, or i32 numpy array".to_string(),
            ));
        }

        if let Ok(array) = untyped.downcast::<PyArrayDyn<u8>>() {
            return Ok(Self::UInt8(array.readonly()));
        }
        if let Ok(array) = untyped.downcast::<PyArrayDyn<i8>>() {
            return Ok(Self::Int8(array.readonly()));
        }
        if let Ok(array) = untyped.downcast::<PyArrayDyn<u16>>() {
            return Ok(Self::UInt16(array.readonly()));
        }
        if let Ok(array) = untyped.downcast::<PyArrayDyn<i16>>() {
            return Ok(Self::Int16(array.readonly()));
        }
        if let Ok(array) = untyped.downcast::<PyArrayDyn<u32>>() {
            return Ok(Self::UInt32(array.readonly()));
        }
        if let Ok(array) = untyped.downcast::<PyArrayDyn<i32>>() {
            return Ok(Self::Int32(array.readonly()));
        }
        Err(pyo3::exceptions::PyRuntimeError::new_err(
            "Could not parse array as u8, i8, u16, i16, u32, or i32 numpy array".to_string(),
        ))
    }
}

#[derive(FromPyObject)]
pub enum ArrayGenericFloat<'a> {
    Float32(PyReadwriteArrayDyn<'a, f32>),
    Float64(PyReadwriteArrayDyn<'a, f64>),
}
#[pyclass(name = "Decoder")]
pub struct PyDecoder {
    decoder: Decoder,
}

unsafe impl Send for PyDecoder {}
unsafe impl Sync for PyDecoder {}

macro_rules! dequantize {
    ($inp:expr, $outp:expr, $quant_boxes:expr) => {{
        let input = $inp.as_slice()?;
        let output = $outp.as_slice_mut()?;
        if output.len() < input.len() {
            return Err(pyo3::exceptions::PyRuntimeError::new_err(
                "Output tensor length is too short".to_string(),
            ));
        }
        dequantize_cpu(input, Quantization::from($quant_boxes), output);
    }};
}

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
        let config = pythonize::depythonize(&config)?;
        let nms: Option<Nms> = nms.map(|py_nms| py_nms.into());
        let decoder = DecoderBuilder::default()
            .with_score_threshold(score_threshold)
            .with_iou_threshold(iou_threshold)
            .with_nms(nms)
            .with_config(config)
            .build();
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
        model_output: ListOfReadOnlyArrayGenericDyn,
        max_boxes: usize,
    ) -> PyResult<PySegDetOutput<'py>> {
        let mut output_boxes = Vec::with_capacity(max_boxes);
        let mut output_masks = Vec::with_capacity(max_boxes);
        let result = match model_output {
            ListOfReadOnlyArrayGenericDyn::Quantized(items) => {
                let outputs = items
                    .iter()
                    .map(|x| match x {
                        ArrayQuantized::UInt8(arr) => arr.as_array().into(),
                        ArrayQuantized::Int8(arr) => arr.as_array().into(),
                        ArrayQuantized::UInt16(arr) => arr.as_array().into(),
                        ArrayQuantized::Int16(arr) => arr.as_array().into(),
                        ArrayQuantized::UInt32(arr) => arr.as_array().into(),
                        ArrayQuantized::Int32(arr) => arr.as_array().into(),
                    })
                    .collect::<Vec<_>>();
                self_
                    .decoder
                    .decode_quantized(&outputs, &mut output_boxes, &mut output_masks)
            }
            ListOfReadOnlyArrayGenericDyn::Float16(items) => {
                let outputs = items
                    .iter()
                    .filter_map(|x| match x {
                        WithInt32Array::Val(x) => Some(x.arr.as_array()),
                        WithInt32Array::Int32(_) => None,
                    })
                    .collect::<Vec<_>>();

                let outputs = outputs
                    .iter()
                    .map(|arr| arr.map(|x| x.to_f32()))
                    .collect::<Vec<_>>();
                let output_views = outputs
                    .iter()
                    .map(|output| output.view())
                    .collect::<Vec<_>>();
                self_
                    .decoder
                    .decode_float(&output_views, &mut output_boxes, &mut output_masks)
            }
            ListOfReadOnlyArrayGenericDyn::Float32(items) => {
                let outputs = items
                    .iter()
                    .filter_map(|x| match x {
                        WithInt32Array::Val(x) => Some(x.as_array()),
                        WithInt32Array::Int32(_) => None,
                    })
                    .collect::<Vec<_>>();
                self_
                    .decoder
                    .decode_float(&outputs, &mut output_boxes, &mut output_masks)
            }
            ListOfReadOnlyArrayGenericDyn::Float64(items) => {
                let outputs = items
                    .iter()
                    .filter_map(|x| match x {
                        WithInt32Array::Val(x) => Some(x.as_array()),
                        WithInt32Array::Int32(_) => None,
                    })
                    .collect::<Vec<_>>();
                self_
                    .decoder
                    .decode_float(&outputs, &mut output_boxes, &mut output_masks)
            }
        };
        if let Err(e) = result {
            return Err(pyo3::exceptions::PyRuntimeError::new_err(format!("{e:#?}")));
        }
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
        model_output: ListOfReadOnlyArrayGenericDyn,
        max_boxes: usize,
    ) -> PyResult<PySegDetTrackedOutput<'py>> {
        let mut output_boxes = Vec::with_capacity(max_boxes);
        let mut output_masks = Vec::with_capacity(max_boxes);
        let mut output_tracks = Vec::with_capacity(max_boxes);
        let result = match model_output {
            ListOfReadOnlyArrayGenericDyn::Quantized(items) => {
                let outputs = items
                    .iter()
                    .map(|x| match x {
                        ArrayQuantized::UInt8(arr) => arr.as_array().into(),
                        ArrayQuantized::Int8(arr) => arr.as_array().into(),
                        ArrayQuantized::UInt16(arr) => arr.as_array().into(),
                        ArrayQuantized::Int16(arr) => arr.as_array().into(),
                        ArrayQuantized::UInt32(arr) => arr.as_array().into(),
                        ArrayQuantized::Int32(arr) => arr.as_array().into(),
                    })
                    .collect::<Vec<_>>();
                self_.decoder.decode_tracked_quantized(
                    tracker,
                    timestamp,
                    &outputs,
                    &mut output_boxes,
                    &mut output_masks,
                    &mut output_tracks,
                )
            }
            ListOfReadOnlyArrayGenericDyn::Float16(items) => {
                let outputs = items
                    .iter()
                    .filter_map(|x| match x {
                        WithInt32Array::Val(x) => Some(x.arr.as_array()),
                        WithInt32Array::Int32(_) => None,
                    })
                    .collect::<Vec<_>>();

                let outputs = outputs
                    .iter()
                    .map(|arr| arr.map(|x| x.to_f32()))
                    .collect::<Vec<_>>();
                let output_views = outputs
                    .iter()
                    .map(|output| output.view())
                    .collect::<Vec<_>>();
                self_.decoder.decode_tracked_float(
                    tracker,
                    timestamp,
                    &output_views,
                    &mut output_boxes,
                    &mut output_masks,
                    &mut output_tracks,
                )
            }
            ListOfReadOnlyArrayGenericDyn::Float32(items) => {
                let outputs = items
                    .iter()
                    .filter_map(|x| match x {
                        WithInt32Array::Val(x) => Some(x.as_array()),
                        WithInt32Array::Int32(_) => None,
                    })
                    .collect::<Vec<_>>();
                self_.decoder.decode_tracked_float(
                    tracker,
                    timestamp,
                    &outputs,
                    &mut output_boxes,
                    &mut output_masks,
                    &mut output_tracks,
                )
            }
            ListOfReadOnlyArrayGenericDyn::Float64(items) => {
                let outputs = items
                    .iter()
                    .filter_map(|x| match x {
                        WithInt32Array::Val(x) => Some(x.as_array()),
                        WithInt32Array::Int32(_) => None,
                    })
                    .collect::<Vec<_>>();
                self_.decoder.decode_tracked_float(
                    tracker,
                    timestamp,
                    &outputs,
                    &mut output_boxes,
                    &mut output_masks,
                    &mut output_tracks,
                )
            }
        };
        if let Err(e) = result {
            return Err(pyo3::exceptions::PyRuntimeError::new_err(format!("{e:#?}")));
        }
        let py = self_.py();
        let (boxes, scores, classes) = convert_detect_box(py, &output_boxes);
        let masks = convert_seg_mask(py, &output_masks);
        let tracks = output_tracks.into_iter().map(|t| t.into()).collect();
        Ok((boxes, scores, classes, masks, tracks))
    }

    /// Decode model outputs and draw masks directly onto the destination
    /// image in a single call. Masks never leave Rust, eliminating the
    /// Python round-trip overhead of `decode()` + `draw_masks()`.
    ///
    /// For segmentation models, prototype data is passed directly to the
    /// renderer without materializing intermediate mask arrays in Python.
    /// For detection-only models, this falls back to the standard rendering
    /// path.
    ///
    /// Returns `(boxes, scores, classes)` — no mask arrays are returned.
    #[pyo3(signature = (model_output, processor, dst, max_boxes=100, background=None, opacity=1.0))]
    pub fn draw_masks<'py>(
        self_: PyRef<'py, Self>,
        model_output: ListOfReadOnlyArrayGenericDyn,
        processor: &mut PyImageProcessor,
        dst: &mut PyTensor,
        max_boxes: usize,
        background: Option<&PyTensor>,
        opacity: f32,
    ) -> PyResult<PyDetOutput<'py>> {
        let mut output_boxes = Vec::with_capacity(max_boxes);
        let mut output_masks = Vec::with_capacity(max_boxes);

        // Try the proto path first (returns ProtoData for seg models, None for det-only)
        let proto_result = match &model_output {
            ListOfReadOnlyArrayGenericDyn::Quantized(items) => {
                let outputs = items
                    .iter()
                    .map(|x| match x {
                        ArrayQuantized::UInt8(arr) => arr.as_array().into(),
                        ArrayQuantized::Int8(arr) => arr.as_array().into(),
                        ArrayQuantized::UInt16(arr) => arr.as_array().into(),
                        ArrayQuantized::Int16(arr) => arr.as_array().into(),
                        ArrayQuantized::UInt32(arr) => arr.as_array().into(),
                        ArrayQuantized::Int32(arr) => arr.as_array().into(),
                    })
                    .collect::<Vec<_>>();
                self_
                    .decoder
                    .decode_quantized_proto(&outputs, &mut output_boxes)
                    .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("{e:#?}")))?
            }
            ListOfReadOnlyArrayGenericDyn::Float16(items) => {
                let outputs = items
                    .iter()
                    .filter_map(|x| match x {
                        WithInt32Array::Val(x) => Some(x.arr.as_array()),
                        WithInt32Array::Int32(_) => None,
                    })
                    .collect::<Vec<_>>();
                let outputs = outputs
                    .iter()
                    .map(|arr| arr.map(|x| x.to_f32()))
                    .collect::<Vec<_>>();
                let output_views = outputs
                    .iter()
                    .map(|output| output.view())
                    .collect::<Vec<_>>();
                self_
                    .decoder
                    .decode_float_proto(&output_views, &mut output_boxes)
                    .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("{e:#?}")))?
            }
            ListOfReadOnlyArrayGenericDyn::Float32(items) => {
                let outputs = items
                    .iter()
                    .filter_map(|x| match x {
                        WithInt32Array::Val(x) => Some(x.as_array()),
                        WithInt32Array::Int32(_) => None,
                    })
                    .collect::<Vec<_>>();
                self_
                    .decoder
                    .decode_float_proto(&outputs, &mut output_boxes)
                    .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("{e:#?}")))?
            }
            ListOfReadOnlyArrayGenericDyn::Float64(items) => {
                let outputs = items
                    .iter()
                    .filter_map(|x| match x {
                        WithInt32Array::Val(x) => Some(x.as_array()),
                        WithInt32Array::Int32(_) => None,
                    })
                    .collect::<Vec<_>>();
                self_
                    .decoder
                    .decode_float_proto(&outputs, &mut output_boxes)
                    .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("{e:#?}")))?
            }
        };

        // Render based on whether we got proto data or not
        if let Some(bg) = &background {
            if std::ptr::eq(&bg.0 as *const _, &dst.0 as *const _) {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "background must not be the same tensor as dst",
                ));
            }
        }
        let overlay = edgefirst_hal::image::MaskOverlay {
            background: background.map(|b| &b.0),
            opacity: opacity.clamp(0.0, 1.0),
        };
        if let Ok(mut l) = processor.0.lock() {
            if let Some(proto_data) = proto_result {
                // Fused path: render directly from proto data (masks stay in Rust)
                l.draw_masks_proto(&mut dst.0, &output_boxes, &proto_data, overlay)
                    .map_err(|e| {
                        pyo3::exceptions::PyRuntimeError::new_err(format!(
                            "draw_masks_proto: {e:#?}"
                        ))
                    })?;
            } else {
                // Detection-only or unsupported model: fall back to standard decode + render
                let result = match model_output {
                    ListOfReadOnlyArrayGenericDyn::Quantized(items) => {
                        let outputs = items
                            .iter()
                            .map(|x| match x {
                                ArrayQuantized::UInt8(arr) => arr.as_array().into(),
                                ArrayQuantized::Int8(arr) => arr.as_array().into(),
                                ArrayQuantized::UInt16(arr) => arr.as_array().into(),
                                ArrayQuantized::Int16(arr) => arr.as_array().into(),
                                ArrayQuantized::UInt32(arr) => arr.as_array().into(),
                                ArrayQuantized::Int32(arr) => arr.as_array().into(),
                            })
                            .collect::<Vec<_>>();
                        self_.decoder.decode_quantized(
                            &outputs,
                            &mut output_boxes,
                            &mut output_masks,
                        )
                    }
                    ListOfReadOnlyArrayGenericDyn::Float16(items) => {
                        let outputs = items
                            .iter()
                            .filter_map(|x| match x {
                                WithInt32Array::Val(x) => Some(x.arr.as_array()),
                                WithInt32Array::Int32(_) => None,
                            })
                            .collect::<Vec<_>>();
                        let outputs = outputs
                            .iter()
                            .map(|arr| arr.map(|x| x.to_f32()))
                            .collect::<Vec<_>>();
                        let output_views = outputs
                            .iter()
                            .map(|output| output.view())
                            .collect::<Vec<_>>();
                        self_.decoder.decode_float(
                            &output_views,
                            &mut output_boxes,
                            &mut output_masks,
                        )
                    }
                    ListOfReadOnlyArrayGenericDyn::Float32(items) => {
                        let outputs = items
                            .iter()
                            .filter_map(|x| match x {
                                WithInt32Array::Val(x) => Some(x.as_array()),
                                WithInt32Array::Int32(_) => None,
                            })
                            .collect::<Vec<_>>();
                        self_
                            .decoder
                            .decode_float(&outputs, &mut output_boxes, &mut output_masks)
                    }
                    ListOfReadOnlyArrayGenericDyn::Float64(items) => {
                        let outputs = items
                            .iter()
                            .filter_map(|x| match x {
                                WithInt32Array::Val(x) => Some(x.as_array()),
                                WithInt32Array::Int32(_) => None,
                            })
                            .collect::<Vec<_>>();
                        self_
                            .decoder
                            .decode_float(&outputs, &mut output_boxes, &mut output_masks)
                    }
                };
                if let Err(e) = result {
                    return Err(pyo3::exceptions::PyRuntimeError::new_err(format!("{e:#?}")));
                }
                l.draw_masks(&mut dst.0, &output_boxes, &output_masks, overlay)
                    .map_err(|e| {
                        pyo3::exceptions::PyRuntimeError::new_err(format!("draw_masks: {e:#?}"))
                    })?;
            }
        }

        let py = self_.py();
        let (boxes, scores, classes) = convert_detect_box(py, &output_boxes);
        Ok((boxes, scores, classes))
    }

    #[staticmethod]
    #[pyo3(signature = (boxes, quant_boxes=(1.0, 0), score_threshold=0.1, iou_threshold=0.7, nms=PyNms::ClassAgnostic, max_boxes=100))]
    pub fn decode_yolo_det<'py>(
        py: Python<'py>,
        boxes: ReadOnlyArrayGeneric2,
        quant_boxes: (f64, i64),
        score_threshold: f64,
        iou_threshold: f64,
        nms: Option<PyNms>,
        max_boxes: usize,
    ) -> PyResult<PyDetOutput<'py>> {
        let mut output_boxes = Vec::with_capacity(max_boxes);
        let nms: Option<Nms> = nms.map(|py_nms| py_nms.into());
        match boxes {
            ReadOnlyArrayGeneric2::UInt8(output) => edgefirst_hal::decoder::yolo::decode_yolo_det(
                (output.as_array(), Quantization::from(quant_boxes)),
                score_threshold as f32,
                iou_threshold as f32,
                nms,
                &mut output_boxes,
            ),
            ReadOnlyArrayGeneric2::Int8(output) => edgefirst_hal::decoder::yolo::decode_yolo_det(
                (output.as_array(), Quantization::from(quant_boxes)),
                score_threshold as f32,
                iou_threshold as f32,
                nms,
                &mut output_boxes,
            ),
            ReadOnlyArrayGeneric2::Float32(output) => {
                edgefirst_hal::decoder::yolo::decode_yolo_det_float(
                    output.as_array(),
                    score_threshold as f32,
                    iou_threshold as f32,
                    nms,
                    &mut output_boxes,
                )
            }
            ReadOnlyArrayGeneric2::Float64(output) => {
                edgefirst_hal::decoder::yolo::decode_yolo_det_float(
                    output.as_array(),
                    score_threshold as f32,
                    iou_threshold as f32,
                    nms,
                    &mut output_boxes,
                );
            }
        };
        Ok(convert_detect_box(py, &output_boxes))
    }

    #[allow(clippy::too_many_arguments)]
    #[staticmethod]
    #[pyo3(signature = (boxes, protos, quant_boxes=(1.0, 0), quant_protos=(1.0, 0), score_threshold=0.1, iou_threshold=0.7, nms=PyNms::ClassAgnostic, max_boxes=100))]
    pub fn decode_yolo_segdet<'py>(
        py: Python<'py>,
        boxes: ReadOnlyArrayGeneric2,
        protos: ReadOnlyArrayGeneric3,
        quant_boxes: (f64, i64),
        quant_protos: (f64, i64),
        score_threshold: f64,
        iou_threshold: f64,
        nms: Option<PyNms>,
        max_boxes: usize,
    ) -> PyResult<PySegDetOutput<'py>> {
        let mut output_boxes = Vec::with_capacity(max_boxes);
        let mut output_masks = Vec::with_capacity(max_boxes);
        let nms: Option<Nms> = nms.map(|py_nms| py_nms.into());

        match (boxes, protos) {
            (ReadOnlyArrayGeneric2::UInt8(boxes), ReadOnlyArrayGeneric3::UInt8(protos)) => {
                edgefirst_hal::decoder::yolo::decode_yolo_segdet_quant(
                    (boxes.as_array(), Quantization::from(quant_boxes)),
                    (protos.as_array(), Quantization::from(quant_protos)),
                    score_threshold as f32,
                    iou_threshold as f32,
                    nms,
                    &mut output_boxes,
                    &mut output_masks,
                )
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("{e:#?}")))?;
            }
            (ReadOnlyArrayGeneric2::Int8(boxes), ReadOnlyArrayGeneric3::Int8(protos)) => {
                edgefirst_hal::decoder::yolo::decode_yolo_segdet_quant(
                    (boxes.as_array(), Quantization::from(quant_boxes)),
                    (protos.as_array(), Quantization::from(quant_protos)),
                    score_threshold as f32,
                    iou_threshold as f32,
                    nms,
                    &mut output_boxes,
                    &mut output_masks,
                )
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("{e:#?}")))?;
            }
            (ReadOnlyArrayGeneric2::Float32(boxes), ReadOnlyArrayGeneric3::Float32(protos)) => {
                edgefirst_hal::decoder::yolo::decode_yolo_segdet_float(
                    boxes.as_array(),
                    protos.as_array(),
                    score_threshold as f32,
                    iou_threshold as f32,
                    nms,
                    &mut output_boxes,
                    &mut output_masks,
                )
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("{e:#?}")))?;
            }
            (ReadOnlyArrayGeneric2::Float64(boxes), ReadOnlyArrayGeneric3::Float64(protos)) => {
                edgefirst_hal::decoder::yolo::decode_yolo_segdet_float(
                    boxes.as_array(),
                    protos.as_array(),
                    score_threshold as f32,
                    iou_threshold as f32,
                    nms,
                    &mut output_boxes,
                    &mut output_masks,
                )
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("{e:#?}")))?;
            }
            _ => {
                return Err(pyo3::exceptions::PyRuntimeError::new_err(
                    "Boxes and Protos are not the same type".to_string(),
                ));
            }
        };

        let (boxes, scores, classes) = convert_detect_box(py, &output_boxes);
        let masks = convert_seg_mask(py, &output_masks);
        Ok((boxes, scores, classes, masks))
    }

    #[allow(clippy::too_many_arguments)]
    #[staticmethod]
    #[pyo3(signature = (boxes, scores, quant_boxes=(1.0, 0), quant_scores=(1.0, 0), score_threshold=0.1, iou_threshold=0.7, max_boxes=100))]
    pub fn decode_modelpack_det<'py>(
        py: Python<'py>,
        boxes: ReadOnlyArrayGeneric2,
        scores: ReadOnlyArrayGeneric2,
        quant_boxes: (f64, i64),
        quant_scores: (f64, i64),
        score_threshold: f32,
        iou_threshold: f32,
        max_boxes: usize,
    ) -> PyResult<PyDetOutput<'py>> {
        let mut output_boxes = Vec::with_capacity(max_boxes);

        match (boxes, scores) {
            (ReadOnlyArrayGeneric2::UInt8(boxes), ReadOnlyArrayGeneric2::UInt8(scores)) => {
                let (boxes, scores) = (boxes.as_array(), scores.as_array());
                edgefirst_hal::decoder::modelpack::decode_modelpack_det(
                    (boxes.view(), Quantization::from(quant_boxes)),
                    (scores.view(), Quantization::from(quant_scores)),
                    score_threshold,
                    iou_threshold,
                    &mut output_boxes,
                );
            }
            (ReadOnlyArrayGeneric2::Int8(boxes), ReadOnlyArrayGeneric2::Int8(scores)) => {
                let (boxes, scores) = (boxes.as_array(), scores.as_array());
                edgefirst_hal::decoder::modelpack::decode_modelpack_det(
                    (boxes.view(), Quantization::from(quant_boxes)),
                    (scores.view(), Quantization::from(quant_scores)),
                    score_threshold,
                    iou_threshold,
                    &mut output_boxes,
                );
            }
            (ReadOnlyArrayGeneric2::Float32(boxes), ReadOnlyArrayGeneric2::Float32(scores)) => {
                let (boxes, scores) = (boxes.as_array(), scores.as_array());
                edgefirst_hal::decoder::modelpack::decode_modelpack_float(
                    boxes.view(),
                    scores.view(),
                    score_threshold,
                    iou_threshold,
                    &mut output_boxes,
                );
            }
            (ReadOnlyArrayGeneric2::Float64(boxes), ReadOnlyArrayGeneric2::Float64(scores)) => {
                let (boxes, scores) = (boxes.as_array(), scores.as_array());
                edgefirst_hal::decoder::modelpack::decode_modelpack_float(
                    boxes.view(),
                    scores.view(),
                    score_threshold,
                    iou_threshold,
                    &mut output_boxes,
                );
            }
            _ => {
                return Err(pyo3::exceptions::PyRuntimeError::new_err(
                    "Box and Scores are not the same type".to_string(),
                ));
            }
        };

        Ok(convert_detect_box(py, &output_boxes))
    }

    #[staticmethod]
    #[pyo3(signature = (boxes, anchors, quant=Vec::new(), score_threshold=0.1, iou_threshold=0.7, max_boxes=100))]
    pub fn decode_modelpack_det_split<'py>(
        py: Python<'py>,
        boxes: ListOfReadOnlyArrayGeneric3,
        anchors: Vec<Vec<[f32; 2]>>,
        quant: Vec<(f64, i64)>,
        score_threshold: f64,
        iou_threshold: f64,
        max_boxes: usize,
    ) -> PyResult<PyDetOutput<'py>> {
        let mut output_boxes = Vec::with_capacity(max_boxes);

        match boxes {
            ListOfReadOnlyArrayGeneric3::UInt8(items) => {
                let outputs = items.iter().map(|x| x.as_array()).collect::<Vec<_>>();

                let mut quant = quant
                    .into_iter()
                    .map(|x| Some(Quantization::from(x)))
                    .collect::<Vec<_>>();
                if quant.len() < outputs.len() {
                    quant.extend(std::iter::repeat_n(None, outputs.len() - quant.len()));
                }
                let configs = anchors
                    .into_iter()
                    .zip(quant)
                    .map(|(a, q)| ModelPackDetectionConfig {
                        anchors: a,
                        quantization: q,
                    })
                    .collect::<Vec<_>>();

                edgefirst_hal::decoder::modelpack::decode_modelpack_split_quant(
                    &outputs,
                    &configs,
                    score_threshold as f32,
                    iou_threshold as f32,
                    &mut output_boxes,
                )
            }
            ListOfReadOnlyArrayGeneric3::Int8(items) => {
                let outputs = items.iter().map(|x| x.as_array()).collect::<Vec<_>>();
                let mut quant = quant
                    .into_iter()
                    .map(|x| Some(Quantization::from(x)))
                    .collect::<Vec<_>>();
                if quant.len() < outputs.len() {
                    quant.extend(std::iter::repeat_n(None, outputs.len() - quant.len()));
                }
                let configs = anchors
                    .into_iter()
                    .zip(quant)
                    .map(|(a, q)| ModelPackDetectionConfig {
                        anchors: a,
                        quantization: q,
                    })
                    .collect::<Vec<_>>();

                edgefirst_hal::decoder::modelpack::decode_modelpack_split_quant(
                    &outputs,
                    &configs,
                    score_threshold as f32,
                    iou_threshold as f32,
                    &mut output_boxes,
                )
            }
            ListOfReadOnlyArrayGeneric3::Float32(items) => {
                let outputs = items.iter().map(|x| x.as_array()).collect::<Vec<_>>();

                let configs = anchors
                    .into_iter()
                    .map(|a| ModelPackDetectionConfig {
                        anchors: a,
                        quantization: None,
                    })
                    .collect::<Vec<_>>();

                edgefirst_hal::decoder::modelpack::decode_modelpack_split_float(
                    &outputs,
                    &configs,
                    score_threshold as f32,
                    iou_threshold as f32,
                    &mut output_boxes,
                )
            }
            ListOfReadOnlyArrayGeneric3::Float64(items) => {
                let outputs = items.iter().map(|x| x.as_array()).collect::<Vec<_>>();
                let configs = anchors
                    .into_iter()
                    .map(|a| ModelPackDetectionConfig {
                        anchors: a,
                        quantization: None,
                    })
                    .collect::<Vec<_>>();

                edgefirst_hal::decoder::modelpack::decode_modelpack_split_float(
                    &outputs,
                    &configs,
                    score_threshold as f32,
                    iou_threshold as f32,
                    &mut output_boxes,
                )
            }
        };

        Ok(convert_detect_box(py, &output_boxes))
    }

    #[staticmethod]
    #[pyo3(signature = (quantized, quant_boxes, dequant_into))]
    pub fn dequantize<'py>(
        quantized: ReadOnlyArrayGenericQuantized<'py>,
        quant_boxes: (f64, i64),
        dequant_into: ArrayGenericFloat<'py>,
    ) -> PyResult<()> {
        log::trace!("enter {:?}", std::time::Instant::now());
        let _timer = FunctionTimer::new("dequant".to_string());
        match (quantized, dequant_into) {
            (
                ReadOnlyArrayGenericQuantized::Int8(input),
                ArrayGenericFloat::Float32(mut output),
            ) => dequantize!(input, output, quant_boxes),
            (
                ReadOnlyArrayGenericQuantized::UInt8(input),
                ArrayGenericFloat::Float32(mut output),
            ) => dequantize!(input, output, quant_boxes),
            (
                ReadOnlyArrayGenericQuantized::Int8(input),
                ArrayGenericFloat::Float64(mut output),
            ) => dequantize!(input, output, quant_boxes),
            (
                ReadOnlyArrayGenericQuantized::UInt8(input),
                ArrayGenericFloat::Float64(mut output),
            ) => dequantize!(input, output, quant_boxes),
            (
                ReadOnlyArrayGenericQuantized::UInt16(input),
                ArrayGenericFloat::Float32(mut output),
            ) => dequantize!(input, output, quant_boxes),
            (
                ReadOnlyArrayGenericQuantized::UInt16(input),
                ArrayGenericFloat::Float64(mut output),
            ) => dequantize!(input, output, quant_boxes),
            (
                ReadOnlyArrayGenericQuantized::Int16(input),
                ArrayGenericFloat::Float32(mut output),
            ) => dequantize!(input, output, quant_boxes),
            (
                ReadOnlyArrayGenericQuantized::Int16(input),
                ArrayGenericFloat::Float64(mut output),
            ) => dequantize!(input, output, quant_boxes),
            (
                ReadOnlyArrayGenericQuantized::UInt32(input),
                ArrayGenericFloat::Float32(mut output),
            ) => dequantize!(input, output, quant_boxes),
            (
                ReadOnlyArrayGenericQuantized::UInt32(input),
                ArrayGenericFloat::Float64(mut output),
            ) => dequantize!(input, output, quant_boxes),
            (
                ReadOnlyArrayGenericQuantized::Int32(input),
                ArrayGenericFloat::Float32(mut output),
            ) => dequantize!(input, output, quant_boxes),
            (
                ReadOnlyArrayGenericQuantized::Int32(input),
                ArrayGenericFloat::Float64(mut output),
            ) => dequantize!(input, output, quant_boxes),
        }
        log::trace!("exit {:?}", std::time::Instant::now());
        Ok(())
    }

    #[staticmethod]
    #[pyo3(signature = (segmentation))]
    pub fn segmentation_to_mask<'py>(
        segmentation: PyReadonlyArray3<'py, u8>,
    ) -> PyResult<Bound<'py, PyArray2<u8>>> {
        let result = segmentation_to_mask(segmentation.as_array())
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        Ok(result.to_pyarray(segmentation.py()))
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

/// Private helpers for PyDecoder (not exposed to Python).
fn convert_detect_box<'py>(py: Python<'py>, output_boxes: &[DetectBox]) -> PyDetOutput<'py> {
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

fn convert_seg_mask<'py>(
    py: Python<'py>,
    output_masks: &[Segmentation],
) -> Vec<Bound<'py, PyArray3<u8>>> {
    output_masks
        .iter()
        .map(|x| x.segmentation.to_pyarray(py))
        .collect()
}
