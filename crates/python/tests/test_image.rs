use edgefirst_python::edgefirst_python;

use numpy::{PyArray1, PyArray2, PyArrayMethods};
use pyo3::{
    PyResult, Python,
    ffi::c_str,
    types::{IntoPyDict, PyAnyMethods, PyDict, PyList},
};

#[test]
fn test_rgba_to_rgb() -> PyResult<()> {
    pyo3::append_to_inittab!(edgefirst_python);
    pyo3::prepare_freethreaded_python();
    Python::with_gil(|py| {
        let out: pyo3::Bound<'_, PyDict> = PyDict::new(py);
        py.run(
            c_str!(include_str!("image/rgba_to_rgb.py")),
            None,
            Some(&out),
        )?;

        Ok(())
    })
}
