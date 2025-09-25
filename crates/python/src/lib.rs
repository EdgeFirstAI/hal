use pyo3::prelude::*;

pub mod decoder;
pub mod image;
pub mod tensor;
#[pymodule]
pub mod edgefirst_python {
    pub use super::*;

    #[pymodule_init]
    fn init(m: &Bound<'_, PyModule>) -> PyResult<()> {
        env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();

        m.add_function(wrap_pyfunction!(version, m)?)?;

        m.add_class::<tensor::PyTensor>()?;
        m.add_class::<image::FourCC>()?;
        m.add_class::<image::PyRect>()?;
        m.add_class::<image::PyRotation>()?;
        m.add_class::<image::PyImageConverter>()?;
        m.add_class::<image::PyTensorImage>()?;
        m.add_class::<decoder::PyDecoder>()?;

        Ok(())
    }

    #[pyfunction]
    fn version() -> String {
        env!("CARGO_PKG_VERSION").to_string()
    }
}
