#![allow(clippy::too_many_arguments, clippy::type_complexity)]
use edgefirst::decoder::{
    Decoder, DecoderBuilder, DetectBox, Quantization, QuantizationF64, Segmentation,
    dequantize_cpu_chunked, dequantize_cpu_chunked_f64, modelpack::ModelPackDetectionConfig,
    segmentation_to_mask,
};
use ndarray::{Array1, Array2};
use numpy::{
    IntoPyArray, PyArray1, PyArray2, PyArray3, PyArrayLike2, PyArrayLike3, PyArrayLikeDyn,
    PyReadonlyArray3, PyReadwriteArrayDyn, ToPyArray,
};
use pyo3::{Bound, FromPyObject, PyAny, PyRef, PyResult, Python, pyclass, pymethods};

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
pub enum ListOfReadOnlyArrayGenericDyn<'py> {
    UInt8(Vec<WithInt32Array<'py, PyArrayLikeDyn<'py, u8>>>),
    Int8(Vec<WithInt32Array<'py, PyArrayLikeDyn<'py, i8>>>),
    Float32(Vec<WithInt32Array<'py, PyArrayLikeDyn<'py, f32>>>),
    Float64(Vec<WithInt32Array<'py, PyArrayLikeDyn<'py, f64>>>),
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
    UInt8(Vec<PyArrayLike3<'py, u8>>),
    Int8(Vec<PyArrayLike3<'py, i8>>),
    Float32(Vec<PyArrayLike3<'py, f32>>),
    Float64(Vec<PyArrayLike3<'py, f64>>),
}

#[derive(FromPyObject)]
pub enum ReadOnlyArrayGeneric2<'py> {
    UInt8(PyArrayLike2<'py, u8>),
    Int8(PyArrayLike2<'py, i8>),
    Float32(PyArrayLike2<'py, f32>),
    Float64(PyArrayLike2<'py, f64>),
}

#[derive(FromPyObject)]
pub enum ReadOnlyArrayGeneric3<'py> {
    UInt8(PyArrayLike3<'py, u8>),
    Int8(PyArrayLike3<'py, i8>),
    Float32(PyArrayLike3<'py, f32>),
    Float64(PyArrayLike3<'py, f64>),
}

#[derive(FromPyObject)]
pub enum ReadOnlyArrayGenericQuantized<'a> {
    UInt8(PyArrayLikeDyn<'a, u8>),
    Int8(PyArrayLikeDyn<'a, i8>),
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
    #[new]
    #[pyo3(signature = (config, score_threshold=0.1, iou_threshold=0.7))]
    pub fn new(config: Bound<PyAny>, score_threshold: f32, iou_threshold: f32) -> PyResult<Self> {
        let config = pythonize::depythonize(&config)?;
        let decoder = DecoderBuilder::default()
            .with_score_threshold(score_threshold)
            .with_iou_threshold(iou_threshold)
            .with_config(config)
            .build();
        match decoder {
            Ok(decoder) => Ok(Self { decoder }),
            Err(e) => Err(pyo3::exceptions::PyRuntimeError::new_err(format!("{e:#?}"))),
        }
    }

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
                let outputs = items
                    .iter()
                    .filter_map(|x| match x {
                        WithInt32Array::Val(x) => Some(x.as_array()),
                        WithInt32Array::Int32(_) => None,
                    })
                    .collect::<Vec<_>>();
                self_
                    .decoder
                    .decode_u8(&outputs, &mut output_boxes, &mut output_masks)
            }
            ListOfReadOnlyArrayGenericDyn::Int8(items) => {
                let outputs = items
                    .iter()
                    .filter_map(|x| match x {
                        WithInt32Array::Val(x) => Some(x.as_array()),
                        WithInt32Array::Int32(_) => None,
                    })
                    .collect::<Vec<_>>();
                self_
                    .decoder
                    .decode_i8(&outputs, &mut output_boxes, &mut output_masks)
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
                    .decode_f32(&outputs, &mut output_boxes, &mut output_masks)
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
                    .decode_f64(&outputs, &mut output_boxes, &mut output_masks)
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
    #[pyo3(signature = (boxes, quant_boxes=(1.0, 0), score_threshold=0.1, iou_threshold=0.7, max_boxes=100))]
    pub fn decode_yolo<'py>(
        py: Python<'py>,
        boxes: ReadOnlyArrayGeneric2,
        quant_boxes: (f64, i64),
        score_threshold: f64,
        iou_threshold: f64,
        max_boxes: usize,
    ) -> PyResult<PyDetOutput<'py>> {
        let mut output_boxes = Vec::with_capacity(max_boxes);
        match boxes {
            ReadOnlyArrayGeneric2::UInt8(output) => edgefirst::decoder::yolo::decode_yolo_u8(
                (
                    output.as_array(),
                    Quantization::try_from((quant_boxes.0, quant_boxes.1))?,
                ),
                score_threshold as f32,
                iou_threshold as f32,
                &mut output_boxes,
            ),
            ReadOnlyArrayGeneric2::Int8(output) => edgefirst::decoder::yolo::decode_yolo_i8(
                (
                    output.as_array(),
                    Quantization::try_from((quant_boxes.0, quant_boxes.1))?,
                ),
                score_threshold as f32,
                iou_threshold as f32,
                &mut output_boxes,
            ),
            ReadOnlyArrayGeneric2::Float32(output) => edgefirst::decoder::yolo::decode_yolo_f32(
                output.as_array(),
                score_threshold as f32,
                iou_threshold as f32,
                &mut output_boxes,
            ),
            ReadOnlyArrayGeneric2::Float64(output) => {
                edgefirst::decoder::yolo::decode_yolo_f64(
                    output.as_array(),
                    score_threshold,
                    iou_threshold,
                    &mut output_boxes,
                );
            }
        };
        Ok(convert_detect_box(py, &output_boxes))
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
                edgefirst::decoder::yolo::decode_yolo_segdet_u8(
                    (boxes.as_array(), Quantization::try_from(quant_boxes)?),
                    (protos.as_array(), Quantization::try_from(quant_protos)?),
                    score_threshold as f32,
                    iou_threshold as f32,
                    &mut output_boxes,
                    &mut output_masks,
                );
            }
            (ReadOnlyArrayGeneric2::Int8(boxes), ReadOnlyArrayGeneric3::Int8(protos)) => {
                edgefirst::decoder::yolo::decode_yolo_segdet_i8(
                    (boxes.as_array(), Quantization::try_from(quant_boxes)?),
                    (protos.as_array(), Quantization::try_from(quant_protos)?),
                    score_threshold as f32,
                    iou_threshold as f32,
                    &mut output_boxes,
                    &mut output_masks,
                );
            }
            (ReadOnlyArrayGeneric2::Float32(boxes), ReadOnlyArrayGeneric3::Float32(protos)) => {
                edgefirst::decoder::yolo::decode_yolo_segdet_f32(
                    boxes.as_array(),
                    protos.as_array(),
                    score_threshold as f32,
                    iou_threshold as f32,
                    &mut output_boxes,
                    &mut output_masks,
                );
            }
            (ReadOnlyArrayGeneric2::Float64(boxes), ReadOnlyArrayGeneric3::Float64(protos)) => {
                edgefirst::decoder::yolo::decode_yolo_segdet_f64(
                    boxes.as_array(),
                    protos.as_array(),
                    score_threshold,
                    iou_threshold,
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
        score_threshold: f64,
        iou_threshold: f64,
        max_boxes: usize,
    ) -> PyResult<PyDetOutput<'py>> {
        let mut output_boxes = Vec::with_capacity(max_boxes);

        match (boxes, scores) {
            (ReadOnlyArrayGeneric2::UInt8(boxes), ReadOnlyArrayGeneric2::UInt8(scores)) => {
                let (boxes, scores) = (boxes.as_array(), scores.as_array());
                edgefirst::decoder::modelpack::decode_modelpack_u8(
                    (boxes.view(), Quantization::try_from(quant_boxes)?),
                    (scores.view(), Quantization::try_from(quant_scores)?),
                    score_threshold as f32,
                    iou_threshold as f32,
                    &mut output_boxes,
                );
            }
            (ReadOnlyArrayGeneric2::Int8(boxes), ReadOnlyArrayGeneric2::Int8(scores)) => {
                let (boxes, scores) = (boxes.as_array(), scores.as_array());
                edgefirst::decoder::modelpack::decode_modelpack_i8(
                    (boxes.view(), Quantization::try_from(quant_boxes)?),
                    (scores.view(), Quantization::try_from(quant_scores)?),
                    score_threshold as f32,
                    iou_threshold as f32,
                    &mut output_boxes,
                );
            }
            (ReadOnlyArrayGeneric2::Float32(boxes), ReadOnlyArrayGeneric2::Float32(scores)) => {
                let (boxes, scores) = (boxes.as_array(), scores.as_array());
                edgefirst::decoder::modelpack::decode_modelpack_f32(
                    boxes.view(),
                    scores.view(),
                    score_threshold as f32,
                    iou_threshold as f32,
                    &mut output_boxes,
                );
            }
            (ReadOnlyArrayGeneric2::Float64(boxes), ReadOnlyArrayGeneric2::Float64(scores)) => {
                let (boxes, scores) = (boxes.as_array(), scores.as_array());
                edgefirst::decoder::modelpack::decode_modelpack_f64(
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
                    .map(|x| Some(Quantization::<u8>::from_tuple_truncate(x)))
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

                edgefirst::decoder::modelpack::decode_modelpack_split(
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
                    .map(|x| Some(Quantization::<i8>::from_tuple_truncate(x)))
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

                edgefirst::decoder::modelpack::decode_modelpack_split(
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

                edgefirst::decoder::modelpack::decode_modelpack_split(
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

                edgefirst::decoder::modelpack::decode_modelpack_split(
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
                dequantize_cpu_chunked(input, Quantization::try_from(quant_boxes)?, output);
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
                dequantize_cpu_chunked(input, Quantization::try_from(quant_boxes)?, output);
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
                    input,
                    QuantizationF64 {
                        scale: quant_boxes.0,
                        zero_point: i8::try_from(quant_boxes.1)?,
                    },
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
                    input,
                    QuantizationF64 {
                        scale: quant_boxes.0,
                        zero_point: u8::try_from(quant_boxes.1)?,
                    },
                    output,
                );
            }
        }
        Ok(())
    }

    #[staticmethod]
    #[pyo3(signature = (segmentation))]
    pub fn segmentation_to_mask<'py>(
        segmentation: PyReadonlyArray3<'py, u8>,
    ) -> Bound<'py, PyArray2<u8>> {
        segmentation_to_mask(segmentation.as_array()).to_pyarray(segmentation.py())
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
