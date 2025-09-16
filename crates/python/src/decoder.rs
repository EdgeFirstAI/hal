#![allow(clippy::too_many_arguments, clippy::type_complexity)]
use edgefirst::decoder::{
    Decoder, DecoderBuilder, DetectBox, Quantization, QuantizationF64, SegmentationMask,
    dequantize_cpu_chunked, dequantize_cpu_chunked_f64,
};
use ndarray::{Array1, Array2};
use numpy::{
    IntoPyArray, PyArray1, PyArray2, PyArray3, PyReadonlyArray2, PyReadonlyArray3,
    PyReadonlyArrayDyn, PyReadwriteArrayDyn, ToPyArray,
};
use pyo3::{Bound, FromPyObject, PyRef, PyResult, Python, pyclass, pymethods};

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
pub enum ListOfReadOnlyArrayGenericDyn<'a> {
    UInt8(Vec<PyReadonlyArrayDyn<'a, u8>>),
    Int8(Vec<PyReadonlyArrayDyn<'a, i8>>),
    Float32(Vec<PyReadonlyArrayDyn<'a, f32>>),
    Float64(Vec<PyReadonlyArrayDyn<'a, f64>>),
}

#[derive(FromPyObject)]
pub enum ReadOnlyArrayGeneric2<'a> {
    UInt8(PyReadonlyArray2<'a, u8>),
    Int8(PyReadonlyArray2<'a, i8>),
    Float32(PyReadonlyArray2<'a, f32>),
    Float64(PyReadonlyArray2<'a, f64>),
}

#[derive(FromPyObject)]
pub enum ReadOnlyArrayGeneric3<'a> {
    UInt8(PyReadonlyArray3<'a, u8>),
    Int8(PyReadonlyArray3<'a, i8>),
    Float32(PyReadonlyArray3<'a, f32>),
    Float64(PyReadonlyArray3<'a, f64>),
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
#[pyclass(name = "Decoder")]
pub struct PyDecoder {
    decoder: Decoder,
}

unsafe impl Send for PyDecoder {}
unsafe impl Sync for PyDecoder {}

#[pymethods]
impl PyDecoder {
    #[staticmethod]
    #[pyo3(signature = (json_str, score_threshold=0.1, iou_threshold=0.7))]
    pub fn new_from_json_str(
        json_str: &str,
        score_threshold: f32,
        iou_threshold: f32,
    ) -> PyResult<Self> {
        let decoder = DecoderBuilder::default()
            .with_score_threshold(score_threshold)
            .with_iou_threshold(iou_threshold)
            .with_config_json_str(json_str.to_string())
            .build();
        match decoder {
            Ok(decoder) => Ok(Self { decoder }),
            Err(e) => Err(pyo3::exceptions::PyRuntimeError::new_err(format!("{e:#?}"))),
        }
    }

    #[staticmethod]
    #[pyo3(signature = (yaml_str, score_threshold=0.1, iou_threshold=0.7))]
    pub fn new_from_yaml_str(
        yaml_str: &str,
        score_threshold: f32,
        iou_threshold: f32,
    ) -> PyResult<Self> {
        let decoder = DecoderBuilder::default()
            .with_score_threshold(score_threshold)
            .with_iou_threshold(iou_threshold)
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
            ListOfReadOnlyArrayGenericDyn::UInt8(items) => {
                let outputs = items.iter().map(|x| x.as_array()).collect::<Vec<_>>();
                self_
                    .decoder
                    .decode_u8(&outputs, &mut output_boxes, &mut output_masks)
            }
            ListOfReadOnlyArrayGenericDyn::Int8(items) => {
                let outputs = items.iter().map(|x| x.as_array()).collect::<Vec<_>>();
                self_
                    .decoder
                    .decode_i8(&outputs, &mut output_boxes, &mut output_masks)
            }
            ListOfReadOnlyArrayGenericDyn::Float32(_) => todo!(),
            ListOfReadOnlyArrayGenericDyn::Float64(_) => todo!(),
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
    #[pyo3(signature = (model_output, scale=1.0, zero_point=0, score_threshold=0.1, iou_threshold=0.7, max_boxes=100))]
    pub fn decode_yolo<'py>(
        py: Python<'py>,
        model_output: ReadOnlyArrayGeneric2,
        scale: f64,
        zero_point: i64,
        score_threshold: f64,
        iou_threshold: f64,
        max_boxes: usize,
    ) -> PyResult<PyDetOutput<'py>> {
        let out = match model_output {
            ReadOnlyArrayGeneric2::UInt8(output) => Self::decode_yolo_u8(
                py,
                output,
                scale as f32,
                u8::try_from(zero_point)?,
                score_threshold as f32,
                iou_threshold as f32,
                max_boxes,
            ),
            ReadOnlyArrayGeneric2::Int8(output) => Self::decode_yolo_i8(
                py,
                output,
                scale as f32,
                i8::try_from(zero_point)?,
                score_threshold as f32,
                iou_threshold as f32,
                max_boxes,
            ),
            ReadOnlyArrayGeneric2::Float32(output) => Self::decode_yolo_f32(
                py,
                output,
                score_threshold as f32,
                iou_threshold as f32,
                max_boxes,
            ),
            ReadOnlyArrayGeneric2::Float64(output) => {
                Self::decode_yolo_f64(py, output, score_threshold, iou_threshold, max_boxes)
            }
        };
        Ok(out)
    }

    #[staticmethod]
    #[pyo3(signature = (model_output, scale, zero_point, score_threshold=0.1, iou_threshold=0.7, max_boxes=100))]
    pub fn decode_yolo_u8<'py>(
        py: Python<'py>,
        model_output: PyReadonlyArray2<u8>,
        scale: f32,
        zero_point: u8,
        score_threshold: f32,
        iou_threshold: f32,
        max_boxes: usize,
    ) -> PyDetOutput<'py> {
        let mut output_boxes = Vec::with_capacity(max_boxes);
        edgefirst::decoder::yolo::decode_yolo_u8(
            model_output.as_array(),
            &Quantization { zero_point, scale },
            score_threshold,
            iou_threshold,
            &mut output_boxes,
        );
        convert_detect_box(py, &output_boxes)
    }

    #[staticmethod]
    #[pyo3(signature = (model_output, scale, zero_point, score_threshold=0.1, iou_threshold=0.7, max_boxes=100))]
    pub fn decode_yolo_i8<'py>(
        py: Python<'py>,
        model_output: PyReadonlyArray2<i8>,
        scale: f32,
        zero_point: i8,
        score_threshold: f32,
        iou_threshold: f32,
        max_boxes: usize,
    ) -> PyDetOutput<'py> {
        let mut output_boxes = Vec::with_capacity(max_boxes);
        edgefirst::decoder::yolo::decode_yolo_i8(
            model_output.as_array(),
            &Quantization { zero_point, scale },
            score_threshold,
            iou_threshold,
            &mut output_boxes,
        );
        convert_detect_box(py, &output_boxes)
    }

    #[staticmethod]
    #[pyo3(signature = (model_output, score_threshold=0.1, iou_threshold=0.7, max_boxes=100))]
    pub fn decode_yolo_f32<'py>(
        py: Python<'py>,
        model_output: PyReadonlyArray2<f32>,
        score_threshold: f32,
        iou_threshold: f32,
        max_boxes: usize,
    ) -> PyDetOutput<'py> {
        let mut output_boxes = Vec::with_capacity(max_boxes);
        edgefirst::decoder::yolo::decode_yolo_f32(
            model_output.as_array(),
            score_threshold,
            iou_threshold,
            &mut output_boxes,
        );
        convert_detect_box(py, &output_boxes)
    }

    #[staticmethod]
    #[pyo3(signature = (model_output, score_threshold=0.1, iou_threshold=0.7, max_boxes=100,))]
    pub fn decode_yolo_f64<'py>(
        py: Python<'py>,
        model_output: PyReadonlyArray2<f64>,
        score_threshold: f64,
        iou_threshold: f64,
        max_boxes: usize,
    ) -> PyDetOutput<'py> {
        let mut output_boxes = Vec::with_capacity(max_boxes);
        edgefirst::decoder::yolo::decode_yolo_f64(
            model_output.as_array(),
            score_threshold,
            iou_threshold,
            &mut output_boxes,
        );
        convert_detect_box(py, &output_boxes)
    }

    #[staticmethod]
    #[pyo3(signature = (boxes, protos, quant_boxes=(1.0, 0), quant_protos=(1.0, 0), score_threshold=0.1, iou_threshold=0.7, max_boxes=100))]
    pub fn decode_yolo_segdet<'py>(
        py: Python<'py>,
        boxes: ReadOnlyArrayGeneric2,
        protos: ReadOnlyArrayGeneric3,
        quant_boxes: (f64, i64),
        quant_protos: (f64, i64),
        score_threshold: f64,
        iou_threshold: f64,
        max_boxes: usize,
    ) -> PyResult<PySegDetOutput<'py>> {
        let mut output_boxes = Vec::with_capacity(max_boxes);
        let mut output_masks = Vec::with_capacity(max_boxes);

        match (boxes, protos) {
            (ReadOnlyArrayGeneric2::UInt8(boxes), ReadOnlyArrayGeneric3::UInt8(protos)) => {
                let (boxes, protos) = (boxes.as_array(), protos.as_array());
                edgefirst::decoder::yolo::decode_yolo_masks_u8(
                    boxes.view(),
                    protos.view(),
                    &Quantization {
                        scale: quant_boxes.0 as f32,
                        zero_point: quant_boxes.1 as u8,
                    },
                    &Quantization {
                        scale: quant_protos.0 as f32,
                        zero_point: quant_protos.1 as u8,
                    },
                    score_threshold as f32,
                    iou_threshold as f32,
                    &mut output_boxes,
                    &mut output_masks,
                );
            }
            (ReadOnlyArrayGeneric2::Int8(boxes), ReadOnlyArrayGeneric3::Int8(protos)) => {
                let (boxes, protos) = (boxes.as_array(), protos.as_array());
                edgefirst::decoder::yolo::decode_yolo_masks_i8(
                    boxes.view(),
                    protos.view(),
                    &Quantization {
                        scale: quant_boxes.0 as f32,
                        zero_point: quant_boxes.1 as i8,
                    },
                    &Quantization {
                        scale: quant_protos.0 as f32,
                        zero_point: quant_protos.1 as i8,
                    },
                    score_threshold as f32,
                    iou_threshold as f32,
                    &mut output_boxes,
                    &mut output_masks,
                );
            }
            (ReadOnlyArrayGeneric2::Float32(boxes), ReadOnlyArrayGeneric3::Float32(protos)) => {
                let (boxes, protos) = (boxes.as_array(), protos.as_array());
                edgefirst::decoder::yolo::decode_yolo_masks_f32(
                    boxes.view(),
                    protos.view(),
                    score_threshold as f32,
                    iou_threshold as f32,
                    &mut output_boxes,
                    &mut output_masks,
                );
            }
            (ReadOnlyArrayGeneric2::Float64(boxes), ReadOnlyArrayGeneric3::Float64(protos)) => {
                let (boxes, protos) = (boxes.as_array(), protos.as_array());
                edgefirst::decoder::yolo::decode_yolo_masks_f64(
                    boxes.view(),
                    protos.view(),
                    score_threshold,
                    iou_threshold,
                    &mut output_boxes,
                    &mut output_masks,
                );
            }
            _ => {
                return Err(pyo3::exceptions::PyRuntimeError::new_err(
                    "Box and Protos are not the same type".to_string(),
                ));
            }
        };

        let (boxes, scores, classes) = convert_detect_box(py, &output_boxes);
        let masks = convert_seg_mask(py, &output_masks);
        Ok((boxes, scores, classes, masks))
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

fn convert_detect_box<'py>(py: Python<'py>, output_boxes: &[DetectBox]) -> PyDetOutput<'py> {
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
    (
        boxes.into_pyarray(py),
        scores.into_pyarray(py),
        classes.into_pyarray(py),
    )
}

fn convert_seg_mask<'py>(
    py: Python<'py>,
    output_masks: &[SegmentationMask],
) -> Vec<Bound<'py, PyArray3<u8>>> {
    output_masks.iter().map(|x| x.mask.to_pyarray(py)).collect()
}
