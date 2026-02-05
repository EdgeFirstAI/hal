// SPDX-FileCopyrightText: Copyright 2025 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

#![allow(clippy::too_many_arguments, clippy::type_complexity)]
use edgefirst_hal::decoder::{
    configs::Nms, dequantize_cpu, modelpack::ModelPackDetectionConfig, segmentation_to_mask,
    Decoder, DecoderBuilder, DetectBox, Quantization, Segmentation,
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
    #[pyo3(signature = (config, score_threshold=0.1, iou_threshold=0.7, nms=PyNms::ClassAgnostic))]
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
                );
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
                );
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
                );
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
                );
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
