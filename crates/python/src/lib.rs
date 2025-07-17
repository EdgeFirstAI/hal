use pyo3::prelude::*;

mod image;
mod tensor;

#[pymodule]
mod edgefirst {
    use super::*;

    #[pymodule_init]
    fn init(m: &Bound<'_, PyModule>) -> PyResult<()> {
        env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();

        m.add_function(wrap_pyfunction!(version, m)?)?;

        m.add_class::<tensor::PyTensor>()?;
        m.add_class::<image::FourCC>()?;

        Ok(())
    }

    #[pyfunction]
    fn version() -> String {
        env!("CARGO_PKG_VERSION").to_string()
    }
}
