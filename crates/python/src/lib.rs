// SPDX-FileCopyrightText: Copyright 2025 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

#![cfg_attr(nightly, feature(f16))]

use pyo3::prelude::*;

pub mod decoder;
pub mod image;
pub mod tensor;
pub mod tracker;

pub struct FunctionTimer {
    name: String,
    start: std::time::Instant,
}

impl FunctionTimer {
    pub fn new(name: String) -> Self {
        Self {
            name,
            start: std::time::Instant::now(),
        }
    }
}

impl Drop for FunctionTimer {
    fn drop(&mut self) {
        log::trace!("{} elapsed: {:?}", self.name, self.start.elapsed())
    }
}

#[pymodule]
pub mod edgefirst_hal {
    pub use super::*;

    #[pymodule_init]
    fn init(m: &Bound<'_, PyModule>) -> PyResult<()> {
        env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();

        m.add_function(wrap_pyfunction!(version, m)?)?;
        m.add_function(wrap_pyfunction!(build_info, m)?)?;

        m.add_class::<tensor::PyTensor>()?;
        m.add_class::<tensor::PyTensorMemory>()?;
        m.add_class::<tensor::PyQuantization>()?;
        m.add_class::<image::PyPixelFormat>()?;
        m.add_class::<image::Normalization>()?;
        m.add_class::<image::PyRect>()?;
        m.add_class::<image::PyRotation>()?;
        m.add_class::<image::PyFlip>()?;
        m.add_class::<image::PyColorMode>()?;
        m.add_class::<image::PyMaskResolution>()?;
        m.add_class::<image::PyImageProcessor>()?;
        m.add_class::<image::PyEglDisplayKind>()?;
        m.add_function(wrap_pyfunction!(image::align_width_for_gpu_pitch, m)?)?;
        m.add_function(wrap_pyfunction!(image::align_width_for_pixel_format, m)?)?;
        m.add_function(wrap_pyfunction!(
            image::gpu_dma_buf_pitch_alignment_bytes,
            m
        )?)?;
        #[cfg(target_os = "linux")]
        {
            m.add_class::<image::PyEglDisplayInfo>()?;
            m.add_function(wrap_pyfunction!(image::probe_egl_displays, m)?)?;
        }
        m.add_class::<decoder::PyDecoder>()?;
        m.add_class::<decoder::PyProtoData>()?;
        m.add_class::<decoder::PyNms>()?;
        m.add_class::<decoder::PyDecoderType>()?;
        m.add_class::<decoder::PyDecoderVersion>()?;
        m.add_class::<decoder::PyDimName>()?;
        m.add_class::<decoder::PyOutput>()?;

        m.add_class::<tracker::PyTrackInfo>()?;
        m.add_class::<tracker::PyByteTrack>()?;
        m.add_class::<tracker::PyActiveTrackInfo>()?;

        m.add_class::<PyTracing>()?;

        Ok(())
    }

    #[pyfunction]
    fn version() -> String {
        env!("CARGO_PKG_VERSION").to_string()
    }

    /// Returns build configuration information including f16 implementation
    #[pyfunction]
    fn build_info() -> String {
        #[cfg(nightly)]
        let f16_impl = "native f16 (nightly, optimized)";
        #[cfg(not(nightly))]
        let f16_impl = "half::f16 (stable, compatible)";

        format!(
            "edgefirst-hal v{}\nf16 implementation: {}",
            env!("CARGO_PKG_VERSION"),
            f16_impl
        )
    }

    /// Trace capture context manager for Perfetto/Chrome JSON output.
    ///
    /// Usage:
    /// ```python
    /// import edgefirst_hal as hal
    ///
    /// with hal.Tracing("/tmp/trace.json"):
    ///     # ... inference pipeline runs here ...
    ///     pass
    /// # trace file flushed and closed
    /// ```
    ///
    /// Or without context manager:
    /// ```python
    /// guard = hal.Tracing("/tmp/trace.json")
    /// guard.start()
    /// # ... work ...
    /// guard.stop()
    /// ```
    ///
    /// Note: Raises RuntimeError if tracing support was not compiled in
    /// (built without the `tracing` feature).
    #[pyclass(name = "Tracing")]
    #[allow(dead_code)]
    struct PyTracing {
        path: String,
        active: bool,
    }

    #[pymethods]
    impl PyTracing {
        #[new]
        fn new(path: String) -> Self {
            Self {
                path,
                active: false,
            }
        }

        /// Start trace capture.
        fn start(&mut self) -> PyResult<()> {
            #[cfg(not(feature = "tracing"))]
            {
                Err(pyo3::exceptions::PyRuntimeError::new_err(
                    "tracing support not compiled in (built without 'tracing' feature)",
                ))
            }
            #[cfg(feature = "tracing")]
            {
                ::edgefirst_hal::trace::start_tracing(&self.path)
                    .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
                self.active = true;
                Ok(())
            }
        }

        /// Stop trace capture and flush the trace file.
        fn stop(&mut self) {
            #[cfg(feature = "tracing")]
            if self.active {
                ::edgefirst_hal::trace::stop_tracing();
                self.active = false;
            }
            #[cfg(not(feature = "tracing"))]
            {
                self.active = false;
            }
        }

        fn __enter__(mut slf: PyRefMut<'_, Self>) -> PyResult<PyRefMut<'_, Self>> {
            slf.start()?;
            Ok(slf)
        }

        #[pyo3(signature = (_exc_type=None, _exc_val=None, _exc_tb=None))]
        fn __exit__(
            &mut self,
            _exc_type: Option<&Bound<'_, pyo3::types::PyAny>>,
            _exc_val: Option<&Bound<'_, pyo3::types::PyAny>>,
            _exc_tb: Option<&Bound<'_, pyo3::types::PyAny>>,
        ) -> bool {
            self.stop();
            false // don't suppress exceptions
        }
    }
}
