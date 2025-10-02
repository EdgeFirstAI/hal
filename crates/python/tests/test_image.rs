use edgefirst_python::edgefirst_python as edgefirst_python_module;
use numpy::{PyArray3, PyUntypedArrayMethods};
use pyo3::{
    PyResult, Python,
    ffi::c_str,
    types::{PyAnyMethods, PyDict},
};

#[test]
fn test_rgba_to_rgb() -> PyResult<()> {
    pyo3::append_to_inittab!(edgefirst_python_module);
    pyo3::prepare_freethreaded_python();
    Python::with_gil(|py| {
        let out: pyo3::Bound<'_, PyDict> = PyDict::new(py);
        py.run(
            c_str!(include_str!("image/rgba_to_rgb.py")),
            None,
            Some(&out),
        )?;
        let src = out
            .get_item("src")
            .unwrap()
            .downcast_into::<edgefirst_python::image::PyTensorImage>()?;
        assert_eq!(
            src.borrow().format().unwrap(),
            edgefirst_python::image::FourCC::RGBA
        );
        let n = out.get_item("n").unwrap().downcast_into::<PyArray3<u8>>()?;
        assert_eq!(n.shape(), [720, 1280, 3]);

        Ok(())
    })
}

#[test]
fn test_rgb_resize() -> PyResult<()> {
    pyo3::append_to_inittab!(edgefirst_python_module);
    pyo3::prepare_freethreaded_python();
    Python::with_gil(|py| {
        let out: pyo3::Bound<'_, PyDict> = PyDict::new(py);
        py.run(
            c_str!(include_str!("image/rgb_resize.py")),
            None,
            Some(&out),
        )?;
        let src = out
            .get_item("src")
            .unwrap()
            .downcast_into::<edgefirst_python::image::PyTensorImage>()?;
        assert_eq!(
            src.borrow().format().unwrap(),
            edgefirst_python::image::FourCC::RGB
        );
        assert_eq!(src.borrow().width(), 1280);
        assert_eq!(src.borrow().height(), 720);
        let n = out.get_item("n").unwrap().downcast_into::<PyArray3<u8>>()?;
        assert_eq!(n.shape(), [640, 640, 3]);

        Ok(())
    })
}

#[test]
fn test_flip() -> PyResult<()> {
    pyo3::append_to_inittab!(edgefirst_python_module);
    pyo3::prepare_freethreaded_python();
    Python::with_gil(|py| {
        let out: pyo3::Bound<'_, PyDict> = PyDict::new(py);
        py.run(c_str!(include_str!("image/flip.py")), None, Some(&out))?;
        let src = out
            .get_item("src")
            .unwrap()
            .downcast_into::<edgefirst_python::image::PyTensorImage>()?;
        assert_eq!(
            src.borrow().format().unwrap(),
            edgefirst_python::image::FourCC::RGBA
        );
        let _ = out.get_item("n").unwrap().downcast_into::<PyArray3<u8>>()?;
        Ok(())
    })
}

#[test]
fn test_grey_load() -> PyResult<()> {
    pyo3::append_to_inittab!(edgefirst_python_module);
    pyo3::prepare_freethreaded_python();
    Python::with_gil(|py| {
        let out: pyo3::Bound<'_, PyDict> = PyDict::new(py);
        py.run(c_str!(include_str!("image/grey_load.py")), None, Some(&out))?;
        let rgba = out
            .get_item("rgba")
            .unwrap()
            .downcast_into::<PyArray3<u8>>()?;

        assert_eq!(rgba.shape(), [681, 1024, 4]);

        let grey = out
            .get_item("grey")
            .unwrap()
            .downcast_into::<PyArray3<u8>>()?;
        assert_eq!(grey.shape(), [681, 1024, 1]);

        let default = out
            .get_item("default")
            .unwrap()
            .downcast_into::<PyArray3<u8>>()?;
        assert_eq!(default.shape(), [681, 1024, 1]);
        Ok(())
    })
}
