use edgefirst::decoder::{Quantization, dequantize_cpu};
use edgefirst_python::edgefirst_python;

use numpy::{PyArray1, PyArray2, PyArrayMethods};
use pyo3::{
    PyResult, Python,
    ffi::c_str,
    types::{IntoPyDict, PyAnyMethods, PyDict, PyList},
};
#[test]
fn test_decoder_i8() -> PyResult<()> {
    pyo3::append_to_inittab!(edgefirst_python);
    pyo3::prepare_freethreaded_python();
    Python::with_gil(|py| {
        let out: pyo3::Bound<'_, PyDict> = PyDict::new(py);
        py.run(c_str!(include_str!("decoder_i8.py")), None, Some(&out))?;

        let py_boxes = out
            .get_item("boxes")
            .unwrap()
            .downcast_into::<PyArray2<f32>>()?;

        let py_scores = out
            .get_item("scores")
            .unwrap()
            .downcast_into::<PyArray1<f32>>()?;

        let py_classes = out
            .get_item("classes")
            .unwrap()
            .downcast_into::<PyArray1<usize>>()?;

        let py_boxes = py_boxes.readonly();

        let boxes = [
            0.5285137, 0.05305544, 0.87541467, 0.999890, 0.130598, 0.43260583, 0.35098213,
            0.9958097,
        ];

        assert!(approx::abs_diff_eq!(
            py_boxes.as_slice()?,
            boxes.as_slice(),
            epsilon = 1e-6
        ));

        let py_scores = py_scores.readonly();

        let scores = [0.5591227, 0.33057618];

        assert!(approx::abs_diff_eq!(
            py_scores.as_slice()?,
            scores.as_slice(),
            epsilon = 1e-6
        ));

        let py_classes = py_classes.readonly();

        let classes = [0, 75];

        assert_eq!(py_classes.as_slice()?, classes.as_slice());
        Ok(())
    })
}

#[test]
fn test_decoder_generic_i8() -> PyResult<()> {
    pyo3::append_to_inittab!(edgefirst_python);
    pyo3::prepare_freethreaded_python();
    Python::with_gil(|py| {
        let out: pyo3::Bound<'_, PyDict> = PyDict::new(py);
        py.run(
            c_str!(include_str!("decoder_generic_i8.py")),
            None,
            Some(&out),
        )?;

        let py_boxes = out
            .get_item("boxes")
            .unwrap()
            .downcast_into::<PyArray2<f32>>()?;

        let py_scores = out
            .get_item("scores")
            .unwrap()
            .downcast_into::<PyArray1<f32>>()?;

        let py_classes = out
            .get_item("classes")
            .unwrap()
            .downcast_into::<PyArray1<usize>>()?;

        let py_boxes = py_boxes.readonly();

        let boxes = [
            0.5285137, 0.05305544, 0.87541467, 0.999890, 0.130598, 0.43260583, 0.35098213,
            0.9958097,
        ];

        assert!(approx::abs_diff_eq!(
            py_boxes.as_slice()?,
            boxes.as_slice(),
            epsilon = 1e-6
        ));

        let py_scores = py_scores.readonly();

        let scores = [0.5591227, 0.33057618];

        assert!(approx::abs_diff_eq!(
            py_scores.as_slice()?,
            scores.as_slice(),
            epsilon = 1e-6
        ));

        let py_classes = py_classes.readonly();

        let classes = [0, 75];

        assert_eq!(py_classes.as_slice()?, classes.as_slice());
        Ok(())
    })
}

#[test]
fn test_decoder_parse_config_modelpack_split_u8() -> PyResult<()> {
    pyo3::append_to_inittab!(edgefirst_python);
    pyo3::prepare_freethreaded_python();
    Python::with_gil(|py| {
        let out: pyo3::Bound<'_, PyDict> = PyDict::new(py);
        py.run(
            c_str!(include_str!("decoder_modelpack_config.py")),
            None,
            Some(&out),
        )?;

        let py_boxes = out
            .get_item("boxes")
            .unwrap()
            .downcast_into::<PyArray2<f32>>()?;

        let py_scores = out
            .get_item("scores")
            .unwrap()
            .downcast_into::<PyArray1<f32>>()?;

        let py_classes = out
            .get_item("classes")
            .unwrap()
            .downcast_into::<PyArray1<usize>>()?;

        let py_masks = out.get_item("masks").unwrap().downcast_into::<PyList>()?;

        let py_boxes = py_boxes.readonly();

        let boxes = [0.43171933, 0.68243736, 0.5626645, 0.808863];

        assert!(approx::abs_diff_eq!(
            py_boxes.as_slice()?,
            boxes.as_slice(),
            epsilon = 1e-6
        ));

        let py_scores = py_scores.readonly();

        let scores = [0.49577647];

        assert!(approx::abs_diff_eq!(
            py_scores.as_slice()?,
            scores.as_slice(),
            epsilon = 1e-6
        ));

        let py_classes = py_classes.readonly();

        let classes = [0];

        assert_eq!(py_classes.as_slice()?, classes.as_slice());

        assert_eq!(py_masks.len()?, 0);

        Ok(())
    })
}

#[test]
fn test_dequantize() -> PyResult<()> {
    pyo3::append_to_inittab!(edgefirst_python);
    pyo3::prepare_freethreaded_python();
    Python::with_gil(|py| {
        let out: pyo3::Bound<'_, PyDict> = PyDict::new(py);
        py.run(
            c_str!(include_str!("decoder_dequantize.py")),
            None,
            Some(&out),
        )?;

        let output = out
            .get_item("output")
            .unwrap()
            .downcast_into::<PyArray2<f32>>()?;
        let output = output.readonly();

        let input = include_bytes!("../../../testdata/yolov8s_80_classes.bin");
        let input = unsafe { std::slice::from_raw_parts(input.as_ptr() as *const i8, input.len()) };
        let mut rust_dequantize = vec![0.0; 84 * 8400];
        dequantize_cpu(
            &Quantization {
                scale: 0.0040811873,
                zero_point: -123,
            },
            input,
            &mut rust_dequantize,
        );

        assert!(approx::abs_diff_eq!(
            output.as_slice()?,
            rust_dequantize.as_slice(),
            epsilon = 1e-6
        ));

        Ok(())
    })
}

#[test]
fn test_import_numpy() -> PyResult<()> {
    pyo3::prepare_freethreaded_python();
    Python::with_gil(|py| {
        let np = py.import("numpy")?;
        let locals = [("np", np)].into_py_dict(py)?;

        let pyarray = py
            .eval(
                c_str!("np.absolute(np.array([-1, -2, -3], dtype='int32'))"),
                Some(&locals),
                None,
            )?
            .downcast_into::<PyArray1<i32>>()?;

        let readonly = pyarray.readonly();
        let slice = readonly.as_slice()?;
        assert_eq!(slice, &[1, 2, 3]);

        Ok(())
    })
}
