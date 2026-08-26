// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! Process-wide capability probes and telemetry counters.

use std::ffi::{c_char, c_int, CStr};
use std::panic::{catch_unwind, AssertUnwindSafe};

use edgefirst_tensor::{DType, PixelFormat};

/// Whether CUDA interop symbols resolved.
#[no_mangle]
pub extern "C" fn ef_is_cuda_available() -> c_int {
    catch_unwind(|| i32::from(edgefirst_tensor::is_cuda_available())).unwrap_or(0)
}

/// Whether Linux DMA-BUF allocation is available.
#[no_mangle]
pub extern "C" fn ef_is_dma_available() -> c_int {
    catch_unwind(|| i32::from(edgefirst_tensor::is_dma_available())).unwrap_or(0)
}

/// Whether a platform GPU-coherent buffer kind can be allocated.
#[no_mangle]
pub extern "C" fn ef_is_gpu_buffer_available() -> c_int {
    catch_unwind(|| i32::from(edgefirst_tensor::is_gpu_buffer_available())).unwrap_or(0)
}

/// Whether IOSurface allocation is available.
#[no_mangle]
pub extern "C" fn ef_is_iosurface_available() -> c_int {
    catch_unwind(|| i32::from(edgefirst_tensor::is_iosurface_available())).unwrap_or(0)
}

/// Whether POSIX shared memory allocation is available.
#[no_mangle]
pub extern "C" fn ef_is_shm_available() -> c_int {
    catch_unwind(|| i32::from(edgefirst_tensor::is_shm_available())).unwrap_or(0)
}

/// Whether this platform can honour a tile-compression request for `format`/`dtype`.
///
/// `format` is a wire code (`"NV12"`, `"rgba8"`). Returns 1 when a request can
/// be honoured, 0 otherwise (including unknown format/dtype).
///
/// # Safety
/// `format` must be `NULL` or a NUL-terminated string.
#[no_mangle]
pub unsafe extern "C" fn ef_platform_compression_support(
    format: *const c_char,
    dtype: u32,
) -> c_int {
    unsafe {
        catch_unwind(AssertUnwindSafe(|| {
            if format.is_null() {
                return 0;
            }
            let Ok(s) = CStr::from_ptr(format).to_str() else {
                return 0;
            };
            let Some(fmt) = PixelFormat::from_str_code(s) else {
                return 0;
            };
            let Some(dt) = DType::from_code(dtype) else {
                return 0;
            };
            i32::from(edgefirst_tensor::compression_support(fmt, dt))
        }))
        .unwrap_or(0)
    }
}

/// `HAL_COMPRESSION_ANY` requests that fell back to a linear layout.
#[no_mangle]
pub extern "C" fn ef_compression_fallback_count() -> u64 {
    catch_unwind(edgefirst_tensor::compression_fallback_count).unwrap_or(0)
}

/// Maps that exceeded a buffer's declared CPU access.
#[no_mangle]
pub extern "C" fn ef_unplanned_cpu_access_count() -> u64 {
    catch_unwind(edgefirst_tensor::unplanned_cpu_access_count).unwrap_or(0)
}
