// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! Hand-rolled FFI for the CUDA nvJPEG decode API.
//!
//! These declarations mirror the on-target `nvjpeg.h` (CUDA 12.6 / nvJPEG
//! 12.3.3, verified on the Orin Nano). The library is loaded via `dlopen` (see
//! [`super::loader`]) — there is no link-time dependency, so only the subset of
//! types and entry points the backend uses is declared here. The `CudaStream`
//! is the `cudaStream_t` from `edgefirst_tensor::CudaStream`.

#![allow(non_camel_case_types)]

use std::os::raw::{c_int, c_uchar, c_void};

/// `NVJPEG_MAX_COMPONENT` — the channel/pitch array length in `nvjpegImage_t`.
pub const NVJPEG_MAX_COMPONENT: usize = 4;

/// `nvjpegStatus_t` — `0` is `NVJPEG_STATUS_SUCCESS`.
pub type NvjpegStatus = c_int;
pub const NVJPEG_STATUS_SUCCESS: NvjpegStatus = 0;

/// `nvjpegOutputFormat_t` — only the interleaved-RGB output is used. nvJPEG
/// writes a single packed RGB plane into `channel[0]` at `pitch[0]`, doing the
/// YCbCr→RGB conversion on the GPU.
pub type NvjpegOutputFormat = c_int;
pub const NVJPEG_OUTPUT_RGBI: NvjpegOutputFormat = 5;

/// `nvjpegBackend_t` — `DEFAULT` resolves to GPU-hybrid on this build. The
/// dedicated-hardware backend (`NVJPEG_BACKEND_HARDWARE = 3`) is unsupported on
/// Orin (`nvjpegCreateEx` returns status 7), so it is never requested.
pub type NvjpegBackend = c_int;
pub const NVJPEG_BACKEND_DEFAULT: NvjpegBackend = 0;

/// Opaque library handle (`nvjpegHandle_t`).
pub type NvjpegHandle = *mut c_void;
/// Opaque per-decode state (`nvjpegJpegState_t`).
pub type NvjpegJpegState = *mut c_void;
/// `cudaStream_t`, shared with the tensor crate's CUDA interop.
pub type CudaStream = edgefirst_tensor::CudaStream;

/// `nvjpegImage_t { unsigned char* channel[4]; size_t pitch[4]; }`.
///
/// The `channel` pointers are caller-provided **device** pointers and `pitch`
/// the per-plane row stride in bytes. For `NVJPEG_OUTPUT_RGBI` only
/// `channel[0]`/`pitch[0]` are consulted.
#[repr(C)]
pub struct NvjpegImage {
    pub channel: [*mut c_uchar; NVJPEG_MAX_COMPONENT],
    pub pitch: [usize; NVJPEG_MAX_COMPONENT],
}

impl Default for NvjpegImage {
    fn default() -> Self {
        Self {
            channel: [std::ptr::null_mut(); NVJPEG_MAX_COMPONENT],
            pitch: [0; NVJPEG_MAX_COMPONENT],
        }
    }
}

/// `nvjpegCreateEx(backend, dev_allocator, pinned_allocator, flags, &handle)`.
/// The allocators are passed as null (library-managed) and `flags` as 0.
pub type FnCreateEx = unsafe extern "C" fn(
    NvjpegBackend,
    *const c_void,
    *const c_void,
    c_int,
    *mut NvjpegHandle,
) -> NvjpegStatus;

/// `nvjpegJpegStateCreate(handle, &state)`.
pub type FnJpegStateCreate =
    unsafe extern "C" fn(NvjpegHandle, *mut NvjpegJpegState) -> NvjpegStatus;

/// `nvjpegGetImageInfo(handle, data, len, &nComponents, &subsampling, widths[4], heights[4])`.
pub type FnGetImageInfo = unsafe extern "C" fn(
    NvjpegHandle,
    *const c_uchar,
    usize,
    *mut c_int,
    *mut c_int,
    *mut c_int,
    *mut c_int,
) -> NvjpegStatus;

/// `nvjpegDecode(handle, state, data, len, output_format, &image, stream)`.
/// Asynchronous: GPU work is submitted to `stream`.
pub type FnDecode = unsafe extern "C" fn(
    NvjpegHandle,
    NvjpegJpegState,
    *const c_uchar,
    usize,
    NvjpegOutputFormat,
    *mut NvjpegImage,
    CudaStream,
) -> NvjpegStatus;

/// `nvjpegDestroy(handle)`.
pub type FnDestroy = unsafe extern "C" fn(NvjpegHandle) -> NvjpegStatus;

/// `nvjpegJpegStateDestroy(state)`.
pub type FnJpegStateDestroy = unsafe extern "C" fn(NvjpegJpegState) -> NvjpegStatus;
