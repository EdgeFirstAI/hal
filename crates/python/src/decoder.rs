#![allow(clippy::too_many_arguments, clippy::type_complexity)]
use edgefirst::decoder::{
    BBoxTypeTrait, DetectBox, DetectBoxF64, Quantization, QuantizationF64, XYWH, XYXY,
    dequantize_cpu_chunked, dequantize_cpu_chunked_f64,
};
use ndarray::{Array1, Array2};
use numpy::{
    IntoPyArray, PyArray1, PyArray2, PyReadonlyArray2, PyReadonlyArrayDyn, PyReadwriteArrayDyn,
};
use pyo3::{Bound, FromPyObject, PyResult, Python, pyclass, pymethods};

pub type PyBoxesOutput<'py> = (
    Bound<'py, PyArray2<f32>>,
    Bound<'py, PyArray1<f32>>,
    Bound<'py, PyArray1<usize>>,
);

#[derive(FromPyObject)]
pub enum ReadOnlyArrayGeneric2<'a> {
    UInt8(PyReadonlyArray2<'a, u8>),
    Int8(PyReadonlyArray2<'a, i8>),
    Float32(PyReadonlyArray2<'a, f32>),
    Float64(PyReadonlyArray2<'a, f64>),
}

#[derive(FromPyObject)]
pub enum ReadOnlyArrayGenericQuantized<'a> {
    UInt8(PyReadonlyArrayDyn<'a, u8>),
    Int8(PyReadonlyArrayDyn<'a, i8>),
}

#[derive(FromPyObject)]
pub enum ArrayGenericFloat<'a> {
    Float32(PyReadwriteArrayDyn<'a, f32>),
    Float64(PyReadwriteArrayDyn<'a, f64>),
}

#[pyclass(name = "BBoxType", eq)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PyBBoxType {
    #[pyo3(name = "XYXY")]
    Xyxy = 0,
    #[pyo3(name = "XYWH")]
    Xywh = 1,
}

#[pymethods]
impl PyBBoxType {
    #[getter]
    fn value(&self) -> isize {
        self.__pyo3__int__()
    }
}

#[derive(Default)]
#[pyclass(name = "Decoder")]
pub struct PyDecoder();

unsafe impl Send for PyDecoder {}
unsafe impl Sync for PyDecoder {}

#[pymethods]
impl PyDecoder {
    // #[new]
    // pub fn new() -> Self {
    //     Self::default()
    // }

    #[staticmethod]
    #[pyo3(signature = (model_output, num_classes, scale=1.0, zero_point=0, score_threshold=0.1, iou_threshold=0.7, max_boxes=100, bbox_type=PyBBoxType::Xyxy))]
    pub fn decode<'py>(
        py: Python<'py>,
        model_output: ReadOnlyArrayGeneric2,
        num_classes: usize,
        scale: f64,
        zero_point: i64,
        score_threshold: f64,
        iou_threshold: f64,
        max_boxes: usize,
        bbox_type: PyBBoxType,
    ) -> PyResult<PyBoxesOutput<'py>> {
        let out = match model_output {
            ReadOnlyArrayGeneric2::UInt8(output) => Self::decode_u8(
                py,
                output,
                num_classes,
                scale as f32,
                u8::try_from(zero_point)?,
                score_threshold as f32,
                iou_threshold as f32,
                max_boxes,
                bbox_type,
            ),
            ReadOnlyArrayGeneric2::Int8(output) => Self::decode_i8(
                py,
                output,
                num_classes,
                scale as f32,
                i8::try_from(zero_point)?,
                score_threshold as f32,
                iou_threshold as f32,
                max_boxes,
                bbox_type,
            ),
            ReadOnlyArrayGeneric2::Float32(output) => Self::decode_f32(
                py,
                output,
                num_classes,
                score_threshold as f32,
                iou_threshold as f32,
                max_boxes,
                bbox_type,
            ),
            ReadOnlyArrayGeneric2::Float64(output) => Self::decode_f64(
                py,
                output,
                num_classes,
                score_threshold,
                iou_threshold,
                max_boxes,
                bbox_type,
            ),
        };
        Ok(out)
    }

    #[staticmethod]
    #[pyo3(signature = (model_output, num_classes, scale, zero_point, score_threshold=0.1, iou_threshold=0.7, max_boxes=100, bbox_type=PyBBoxType::Xyxy))]
    pub fn decode_u8<'py>(
        py: Python<'py>,
        model_output: PyReadonlyArray2<u8>,
        num_classes: usize,
        scale: f32,
        zero_point: u8,
        score_threshold: f32,
        iou_threshold: f32,
        max_boxes: usize,
        bbox_type: PyBBoxType,
    ) -> PyBoxesOutput<'py> {
        let mut output_boxes = Vec::with_capacity(max_boxes);
        let func = match bbox_type {
            PyBBoxType::Xyxy => edgefirst::decoder::bits8::decode_u8::<XYXY>,
            PyBBoxType::Xywh => edgefirst::decoder::bits8::decode_u8::<XYWH>,
        };
        func(
            model_output.as_array(),
            num_classes,
            &Quantization { zero_point, scale },
            score_threshold,
            iou_threshold,
            &mut output_boxes,
        );
        convert_detect_box(py, &output_boxes)
    }

    #[staticmethod]
    #[pyo3(signature = (model_output, num_classes, scale, zero_point, score_threshold=0.1, iou_threshold=0.7, max_boxes=100, bbox_type=PyBBoxType::Xyxy))]
    pub fn decode_i8<'py>(
        py: Python<'py>,
        model_output: PyReadonlyArray2<i8>,
        num_classes: usize,
        scale: f32,
        zero_point: i8,
        score_threshold: f32,
        iou_threshold: f32,
        max_boxes: usize,
        bbox_type: PyBBoxType,
    ) -> PyBoxesOutput<'py> {
        let mut output_boxes = Vec::with_capacity(max_boxes);
        let func = match bbox_type {
            PyBBoxType::Xyxy => edgefirst::decoder::bits8::decode_i8::<XYXY>,
            PyBBoxType::Xywh => edgefirst::decoder::bits8::decode_i8::<XYWH>,
        };
        func(
            model_output.as_array(),
            num_classes,
            &Quantization { zero_point, scale },
            score_threshold,
            iou_threshold,
            &mut output_boxes,
        );
        convert_detect_box(py, &output_boxes)
    }

    #[staticmethod]
    #[pyo3(signature = (model_output, num_classes, score_threshold=0.1, iou_threshold=0.7, max_boxes=100, bbox_type=PyBBoxType::Xyxy))]
    pub fn decode_f32<'py>(
        py: Python<'py>,
        model_output: PyReadonlyArray2<f32>,
        num_classes: usize,
        score_threshold: f32,
        iou_threshold: f32,
        max_boxes: usize,
        bbox_type: PyBBoxType,
    ) -> PyBoxesOutput<'py> {
        let mut output_boxes = Vec::with_capacity(max_boxes);
        let func = match bbox_type {
            PyBBoxType::Xyxy => edgefirst::decoder::float::decode_f32::<XYXY>,
            PyBBoxType::Xywh => edgefirst::decoder::float::decode_f32::<XYWH>,
        };
        func(
            model_output.as_array(),
            num_classes,
            score_threshold,
            iou_threshold,
            &mut output_boxes,
        );
        convert_detect_box(py, &output_boxes)
    }

    #[staticmethod]
    #[pyo3(signature = (model_output, num_classes, score_threshold=0.1, iou_threshold=0.7, max_boxes=100, bbox_type=PyBBoxType::Xyxy))]
    pub fn decode_f64<'py>(
        py: Python<'py>,
        model_output: PyReadonlyArray2<f64>,
        num_classes: usize,
        score_threshold: f64,
        iou_threshold: f64,
        max_boxes: usize,
        bbox_type: PyBBoxType,
    ) -> PyBoxesOutput<'py> {
        let mut output_boxes = Vec::with_capacity(max_boxes);
        let func = match bbox_type {
            PyBBoxType::Xyxy => edgefirst::decoder::float::decode_f64::<XYXY>,
            PyBBoxType::Xywh => edgefirst::decoder::float::decode_f64::<XYWH>,
        };
        func(
            model_output.as_array(),
            num_classes,
            score_threshold,
            iou_threshold,
            &mut output_boxes,
        );
        convert_detect_box_f64(py, &output_boxes)
    }

    #[staticmethod]
    pub fn dequantize<'py>(
        quantized: ReadOnlyArrayGenericQuantized<'py>,
        scale: f64,
        zero_point: i128,
        dequant_into: ArrayGenericFloat<'py>,
    ) -> PyResult<()> {
        match (quantized, dequant_into) {
            (
                ReadOnlyArrayGenericQuantized::Int8(input),
                ArrayGenericFloat::Float32(mut output),
            ) => {
                let input = input.as_slice()?;
                let output = output.as_slice_mut()?;
                if output.len() < input.len() {
                    return Err(pyo3::exceptions::PyRuntimeError::new_err(
                        "Output tensor length is too short".to_string(),
                    ));
                }
                dequantize_cpu_chunked(
                    &Quantization {
                        scale: scale as f32,
                        zero_point: i8::try_from(zero_point)?,
                    },
                    input,
                    output,
                );
            }
            (
                ReadOnlyArrayGenericQuantized::UInt8(input),
                ArrayGenericFloat::Float32(mut output),
            ) => {
                let input = input.as_slice()?;
                let output = output.as_slice_mut()?;
                if output.len() < input.len() {
                    return Err(pyo3::exceptions::PyRuntimeError::new_err(
                        "Output tensor length is too short".to_string(),
                    ));
                }
                dequantize_cpu_chunked(
                    &Quantization {
                        scale: scale as f32,
                        zero_point: u8::try_from(zero_point)?,
                    },
                    input,
                    output,
                );
            }
            (
                ReadOnlyArrayGenericQuantized::Int8(input),
                ArrayGenericFloat::Float64(mut output),
            ) => {
                let input = input.as_slice()?;
                let output = output.as_slice_mut()?;
                if output.len() < input.len() {
                    return Err(pyo3::exceptions::PyRuntimeError::new_err(
                        "Output tensor length is too short".to_string(),
                    ));
                }
                dequantize_cpu_chunked_f64(
                    &QuantizationF64 {
                        scale,
                        zero_point: i8::try_from(zero_point)?,
                    },
                    input,
                    output,
                );
            }
            (
                ReadOnlyArrayGenericQuantized::UInt8(input),
                ArrayGenericFloat::Float64(mut output),
            ) => {
                let input = input.as_slice()?;
                let output = output.as_slice_mut()?;
                if output.len() < input.len() {
                    return Err(pyo3::exceptions::PyRuntimeError::new_err(
                        "Output tensor length is too short".to_string(),
                    ));
                }
                dequantize_cpu_chunked_f64(
                    &QuantizationF64 {
                        scale,
                        zero_point: u8::try_from(zero_point)?,
                    },
                    input,
                    output,
                );
            }
        }
        Ok(())
    }
}

fn convert_detect_box<'py>(py: Python<'py>, output_boxes: &[DetectBox]) -> PyBoxesOutput<'py> {
    let boxes = output_boxes
        .iter()
        .flat_map(|b| [b.xmin, b.ymin, b.xmax, b.ymax])
        .collect::<Vec<_>>();
    let scores = output_boxes.iter().map(|b| b.score).collect::<Vec<_>>();
    let classes = output_boxes.iter().map(|b| b.label).collect::<Vec<_>>();
    let num_boxes = output_boxes.len();
    let boxes = Array2::from_shape_vec((num_boxes, 4), boxes).unwrap();
    let scores = Array1::from_vec(scores);
    let classes = Array1::from_vec(classes);
    // let py = self_.py();
    (
        boxes.into_pyarray(py),
        scores.into_pyarray(py),
        classes.into_pyarray(py),
    )
}

fn convert_detect_box_f64<'py>(
    py: Python<'py>,
    output_boxes: &[DetectBoxF64],
) -> PyBoxesOutput<'py> {
    let boxes = output_boxes
        .iter()
        .flat_map(|b| [b.xmin as f32, b.ymin as f32, b.xmax as f32, b.ymax as f32])
        .collect::<Vec<_>>();
    let scores = output_boxes
        .iter()
        .map(|b| b.score as f32)
        .collect::<Vec<_>>();
    let classes = output_boxes.iter().map(|b| b.label).collect::<Vec<_>>();
    let num_boxes = output_boxes.len();
    let boxes = Array2::from_shape_vec((num_boxes, 4), boxes).unwrap();
    let scores = Array1::from_vec(scores);
    let classes = Array1::from_vec(classes);
    // let py = self_.py();
    (
        boxes.into_pyarray(py),
        scores.into_pyarray(py),
        classes.into_pyarray(py),
    )
}
