#![cfg(not(feature = "extension-module"))]

use edgefirst_hal::edgefirst_hal as edgefirst_hal_module;
use numpy::{PyArray3, PyUntypedArrayMethods};
use pyo3::{
    PyResult, Python,
    ffi::c_str,
    types::{PyAnyMethods, PyDict},
};

fn change_dir() -> Result<(), std::io::Error> {
    let manifest_dir = concat!(env!("CARGO_MANIFEST_DIR"), "/../../");
    std::env::set_current_dir(manifest_dir)
}

#[test]
fn test_rgba_to_rgb() -> PyResult<()> {
    pyo3::append_to_inittab!(edgefirst_hal_module);
    pyo3::prepare_freethreaded_python();
    change_dir()?;
    Python::with_gil(|py| {
        let out: pyo3::Bound<'_, PyDict> = PyDict::new(py);
        py.run(
            c_str!(include_str!("../../../tests/image/test_rgba_to_rgb.py")),
            None,
            Some(&out),
        )?;
        let src = out
            .get_item("src")
            .unwrap()
            .downcast_into::<edgefirst_hal::image::PyTensorImage>()?;
        assert_eq!(
            src.borrow().format().unwrap(),
            edgefirst_hal::image::FourCC::RGBA
        );
        let n = out.get_item("n").unwrap().downcast_into::<PyArray3<u8>>()?;
        assert_eq!(n.shape(), [720, 1280, 3]);

        Ok(())
    })
}

#[test]
fn test_rgb_resize() -> PyResult<()> {
    pyo3::append_to_inittab!(edgefirst_hal_module);
    pyo3::prepare_freethreaded_python();
    change_dir()?;
    Python::with_gil(|py| {
        let out: pyo3::Bound<'_, PyDict> = PyDict::new(py);
        py.run(
            c_str!(include_str!("../../../tests/image/test_rgb_resize.py")),
            None,
            Some(&out),
        )?;
        let src = out
            .get_item("src")
            .unwrap()
            .downcast_into::<edgefirst_hal::image::PyTensorImage>()?;
        assert_eq!(
            src.borrow().format().unwrap(),
            edgefirst_hal::image::FourCC::RGB
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
    pyo3::append_to_inittab!(edgefirst_hal_module);
    pyo3::prepare_freethreaded_python();
    change_dir()?;
    Python::with_gil(|py| {
        let out: pyo3::Bound<'_, PyDict> = PyDict::new(py);
        py.run(
            c_str!(include_str!("../../../tests/image/test_flip.py")),
            None,
            Some(&out),
        )?;
        let src = out
            .get_item("src")
            .unwrap()
            .downcast_into::<edgefirst_hal::image::PyTensorImage>()?;
        assert_eq!(
            src.borrow().format().unwrap(),
            edgefirst_hal::image::FourCC::RGBA
        );
        let _ = out.get_item("n").unwrap().downcast_into::<PyArray3<u8>>()?;
        Ok(())
    })
}

#[test]
fn test_grey_load() -> PyResult<()> {
    pyo3::append_to_inittab!(edgefirst_hal_module);
    pyo3::prepare_freethreaded_python();
    change_dir()?;
    Python::with_gil(|py| {
        let out: pyo3::Bound<'_, PyDict> = PyDict::new(py);
        py.run(
            c_str!(include_str!("../../../tests/image/test_grey_load.py")),
            None,
            Some(&out),
        )?;
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
        assert_eq!(default.shape(), [681, 1024, 3]);
        Ok(())
    })
}

#[test]
fn test_normalize() -> PyResult<()> {
    pyo3::append_to_inittab!(edgefirst_hal_module);
    pyo3::prepare_freethreaded_python();
    change_dir()?;
    Python::with_gil(|py| {
        let out: pyo3::Bound<'_, PyDict> = PyDict::new(py);
        py.run(
            c_str!(include_str!("../../../tests/image/test_normalize.py")),
            None,
            Some(&out),
        )?;
        let unique_vals: Vec<i8> = out.get_item("unique_vals").unwrap().extract()?;
        assert_eq!(unique_vals, (-128..=127).collect::<Vec<i8>>());

        let unique_vals0: Vec<i8> = out.get_item("unique_vals0").unwrap().extract()?;
        assert_eq!(unique_vals0, (-127..=127).collect::<Vec<i8>>());
        Ok(())
    })
}
