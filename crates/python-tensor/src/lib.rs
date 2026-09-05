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
    // Windows-only, like the `Tensor` methods that return and consume it:
    // off Windows the name is absent rather than raising, so `hasattr` is
    // the portable capability check.
    #[cfg(target_os = "windows")]
    m.add_class::<tensor::PyD3d11Layout>()?;

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
    #[cfg(target_os = "windows")]
    {
        m.add_function(wrap_pyfunction!(d3d11_device, m)?)?;
        m.add_function(wrap_pyfunction!(d3d11_use_external_device, m)?)?;
    }
    m.add_class::<PyTracing>()?;
    Ok(())
}

#[pyfunction]
fn version() -> String {
    env!("CARGO_PKG_VERSION").to_string()
}

/// True when Linux DMA-BUF heap allocation is available.
///
/// Declared and answered on every platform; ``True`` only on Linux. Use
/// :func:`is_gpu_buffer_available` for the portable question.
#[pyfunction]
fn is_dma_available() -> bool {
    edgefirst_tensor::is_dma_available()
}

/// True when macOS IOSurface allocation is available.
///
/// Declared and answered on every platform; ``True`` only on macOS and iOS.
/// Use :func:`is_gpu_buffer_available` for the portable question.
#[pyfunction]
fn is_iosurface_available() -> bool {
    edgefirst_tensor::is_iosurface_available()
}

/// True when a platform-native GPU-coherent buffer kind is available.
///
/// A DMA-BUF on Linux, an IOSurface on macOS and iOS, an AHardwareBuffer on
/// Android, a D3D11 texture on Windows. Declared and answered on every
/// platform. Use this when you only care whether ``TensorMemory.DMABUF``
/// will succeed without caring which primitive backs it.
#[pyfunction]
fn is_gpu_buffer_available() -> bool {
    edgefirst_tensor::is_gpu_buffer_available()
}

/// True when POSIX shared memory allocation is available.
///
/// Declared and answered on every platform; ``True`` only where ``/dev/shm``
/// is writable, so not on Windows.
#[pyfunction]
fn is_shm_available() -> bool {
    edgefirst_tensor::is_shm_available()
}

/// True when libcudart is loaded and all CUDA interop symbols resolved.
///
/// Declared and answered on every platform; ``True`` only where a CUDA
/// runtime loaded.
#[pyfunction]
fn is_cuda_available() -> bool {
    edgefirst_tensor::is_cuda_available()
}

/// Borrowed ``ID3D11Device*`` of the process device, as an integer.
///
/// Platforms:
///     Windows.
///
/// The device is created on the first call. Wrap with ``ctypes.c_void_p``
/// for native callers; no reference is transferred, so never ``Release``
/// this pointer.
///
/// This is the device the tensors this wheel allocates live on, which is the
/// one inside ``libedgefirst_tensor`` -- not a second one in the wheel's own
/// linked copy.
#[cfg(target_os = "windows")]
#[pyfunction]
fn d3d11_device() -> PyResult<usize> {
    // Backend-routed: on the dynamic backend the allocations happen inside
    // the library, so its device is the answer, not this copy's.
    let device = edgefirst_tensor::d3d11::backend_device()
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("d3d11_device: {e}")))?;
    Ok(device as usize)
}

/// Install a host-owned D3D11 device before the HAL's first use.
///
/// Platforms:
///     Windows.
///
/// Tensors the HAL allocates then live on the caller's device instead of a
/// second one. The reference stays the caller's. Must run before anything
/// that creates the device as a side effect: :func:`d3d11_device`,
/// :func:`is_gpu_buffer_available`, or any texture allocation.
///
/// The pointer is installed into ``libedgefirst_tensor``, which is where the
/// allocations happen, so it takes effect with no intervening call.
///
/// Raises:
///     RuntimeError: If the device is already initialized, or ``ptr`` is
///         not a live ``ID3D11Device``.
#[cfg(target_os = "windows")]
#[pyfunction]
fn d3d11_use_external_device(ptr: usize) -> PyResult<()> {
    // SAFETY: the caller is responsible for `ptr` naming a live
    // `ID3D11Device` that stays live until the HAL takes its reference; that
    // is what the docstring above asks of them, and a Python caller cannot
    // hand over anything the runtime could check further.
    //
    // Backend-routed for the same reason `d3d11_device` is: installing into
    // this copy's own slot would leave the library free to create a device
    // of its own, and nothing would report it.
    unsafe { edgefirst_tensor::d3d11::backend_use_external_device(ptr as *mut std::ffi::c_void) }
        .map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(format!("d3d11_use_external_device: {e}"))
        })
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
