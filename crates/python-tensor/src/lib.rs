// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! `edgefirst.tensor` — buffers, memory backends and colorimetry.

use edgefirst_python_common::{colorimetry, tensor};
use pyo3::prelude::*;

#[pymodule]
fn _tensor(m: &Bound<'_, PyModule>) -> PyResult<()> {
    edgefirst_python_common::init_module();

    m.add_class::<tensor::PyTensor>()?;
    m.add_class::<tensor::PyTensorMemory>()?;
    m.add_class::<tensor::PyQuantization>()?;
    m.add_class::<tensor::PyCudaMap>()?;
    m.add_class::<tensor::PyTensorMap>()?;
    m.add_class::<tensor::PyHostPin>()?;
    m.add_class::<tensor::PyCpuAccessGuard>()?;
    // Both wrap edgefirst_tensor types and now live in tensor.rs; edgefirst.image
    // re-exports them so existing image-side usage is unchanged.
    m.add_class::<tensor::PyPixelFormat>()?;
    m.add_class::<tensor::PyRegion>()?;

    m.add_class::<colorimetry::PyColorSpace>()?;
    m.add_class::<colorimetry::PyColorTransfer>()?;
    m.add_class::<colorimetry::PyColorEncoding>()?;
    m.add_class::<colorimetry::PyColorRange>()?;
    m.add_class::<colorimetry::PyColorimetry>()?;

    m.add_function(wrap_pyfunction!(is_dma_available, m)?)?;
    m.add_function(wrap_pyfunction!(is_iosurface_available, m)?)?;
    m.add_function(wrap_pyfunction!(is_gpu_buffer_available, m)?)?;
    m.add_function(wrap_pyfunction!(is_shm_available, m)?)?;
    m.add_function(wrap_pyfunction!(is_cuda_available, m)?)?;
    m.add_function(wrap_pyfunction!(version, m)?)?;
    m.add_function(wrap_pyfunction!(build_info, m)?)?;
    m.add_class::<PyTracing>()?;
    Ok(())
}

#[pyfunction]
fn version() -> String {
    env!("CARGO_PKG_VERSION").to_string()
}

/// True when Linux DMA-BUF heap allocation is available.
#[pyfunction]
fn is_dma_available() -> bool {
    edgefirst_tensor::is_dma_available()
}

/// True when macOS IOSurface allocation is available.
#[pyfunction]
fn is_iosurface_available() -> bool {
    edgefirst_tensor::is_iosurface_available()
}

/// True when a platform-native GPU-coherent buffer kind is available.
#[pyfunction]
fn is_gpu_buffer_available() -> bool {
    edgefirst_tensor::is_gpu_buffer_available()
}

/// True when POSIX shared memory allocation is available.
#[pyfunction]
fn is_shm_available() -> bool {
    edgefirst_tensor::is_shm_available()
}

/// True when libcudart is loaded and all CUDA interop symbols resolved.
#[pyfunction]
fn is_cuda_available() -> bool {
    edgefirst_tensor::is_cuda_available()
}

/// Build configuration, including which f16 implementation is compiled in.
#[pyfunction]
fn build_info() -> String {
    #[cfg(nightly)]
    let f16_impl = "native f16 (nightly, optimized)";
    #[cfg(not(nightly))]
    let f16_impl = "half::f16 (stable, compatible)";
    format!(
        "edgefirst-tensor v{}\nf16 implementation: {}",
        env!("CARGO_PKG_VERSION"),
        f16_impl
    )
}

/// Trace capture context manager for Perfetto/Chrome JSON output.
///
/// Lives in `edgefirst.tensor` because every other package depends on it, so
/// this is the one import a tracing user is guaranteed to already have.
///
/// ```python
/// from edgefirst.tensor import Tracing
/// with Tracing("/tmp/trace.json"):
///     ...
/// ```
#[pyclass(name = "Tracing", module = "edgefirst.tensor")]
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
            edgefirst_tensor::trace::start_tracing(&self.path)
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
            self.active = true;
            Ok(())
        }
    }

    /// Stop trace capture and flush the trace file.
    fn stop(&mut self) {
        #[cfg(feature = "tracing")]
        if self.active {
            edgefirst_tensor::trace::stop_tracing();
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
        false
    }
}
