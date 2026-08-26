// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! Shared DetectBox ↔ numpy conversions used by image drawing and decoder
//! tiling merge. Lives here so neither module depends on the other.

use edgefirst_tensor::{BoundingBox, DetectBox, Segmentation};
use ndarray::parallel::prelude::{IntoParallelIterator, ParallelIterator};
#[cfg(feature = "decoder")]
use ndarray::{Array1, Array2};
use ndarray::{ArrayView1, ArrayView2, Zip};
#[cfg(feature = "decoder")]
use numpy::{IntoPyArray, PyArray1, PyArray2};
use numpy::{PyArray3, PyReadonlyArray1, PyReadonlyArray2, PyUntypedArrayMethods, ToPyArray};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

#[cfg(feature = "decoder")]
pub(crate) type PyDetOutput<'py> = (
    Bound<'py, PyArray2<f32>>,
    Bound<'py, PyArray1<f32>>,
    Bound<'py, PyArray1<usize>>,
);

pub(crate) fn numpy_to_detect_boxes(
    bbox: &PyReadonlyArray2<f32>,
    scores: &PyReadonlyArray1<f32>,
    classes: &PyReadonlyArray1<usize>,
) -> PyResult<Vec<DetectBox>> {
    if bbox.shape()[1] != 4 {
        return Err(PyValueError::new_err("bbox shape must be (N, 4)"));
    }
    if bbox.shape()[0] != scores.shape()[0] || bbox.shape()[0] != classes.shape()[0] {
        return Err(PyValueError::new_err(
            "bbox, scores, classes must have the same length",
        ));
    }
    let bbox: ArrayView2<f32> = bbox.as_array();
    let scores: ArrayView1<f32> = scores.as_array();
    let classes: ArrayView1<usize> = classes.as_array();
    Ok(Zip::from(bbox.rows())
        .and(scores)
        .and(classes)
        .into_par_iter()
        .map(|(b, s, c)| DetectBox {
            bbox: BoundingBox::new(b[0], b[1], b[2], b[3]),
            score: *s,
            label: *c,
        })
        .collect())
}

#[cfg(feature = "decoder")]
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
        .map(|x| {
            use edgefirst_tensor::{TensorMapTrait as _, TensorTrait as _};
            let sh = x.segmentation.shape();
            let t = x.segmentation.as_u8().expect("mask must be U8");
            let m = t.map_read().expect("map mask");
            ndarray::Array3::from_shape_vec((sh[0], sh[1], sh[2]), m.as_slice().to_vec())
                .expect("mask shape")
                .to_pyarray(py)
        })
        .collect()
}
