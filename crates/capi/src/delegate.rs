// SPDX-FileCopyrightText: Copyright 2025 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! Delegate DMA-BUF API — shared type definitions.
//!
//! This module defines the ABI contract for querying DMA-BUF tensor
//! information from external TFLite delegates (e.g., NXP Neutron NPU).
//! The HAL owns the type definitions; function implementations live in
//! delegate integrators such as `edgefirst-tflite`.
//!
//! See ARCHITECTURE.md "Delegate DMA-BUF Framework" for the full
//! specification including expected function signatures.

use crate::tensor::HalDtype;
use libc::{c_int, size_t};

/// Maximum number of dimensions in a delegate tensor shape.
pub const HAL_DMABUF_MAX_NDIM: usize = 8;

/// DMA-BUF tensor information returned by a delegate.
///
/// Describes a single tensor's DMA-BUF allocation, including the file
/// descriptor, buffer geometry, and element type. The fd is borrowed
/// from the delegate and must NOT be closed by the caller.
///
/// Fields are ordered to eliminate padding on LP64: all `size_t` fields
/// first (8-byte aligned), then smaller `int` and enum fields (4 bytes
/// each) at the end. Total size: 96 bytes on LP64.
///
/// @par Versioning
/// The companion hal_dmabuf_get_tensor_info() function accepts an
/// info_size parameter so that the struct can grow in future versions
/// without breaking ABI. Implementations must zero-initialize the
/// struct with memset(info, 0, info_size) before populating it, and
/// only write fields whose offset + size fits within info_size.
#[repr(C)]
#[derive(Debug, Clone)]
pub struct HalDmabufTensorInfo {
    /// Buffer size in bytes.
    pub size: size_t,
    /// Byte offset within the DMA-BUF.
    pub offset: size_t,
    /// Tensor dimensions (up to HAL_DMABUF_MAX_NDIM).
    ///
    /// Uses a literal length so cbindgen can emit the array in the C header.
    pub shape: [size_t; 8],
    /// Number of valid entries in `shape`.
    pub ndim: size_t,
    /// DMA-BUF file descriptor (borrowed — do not close).
    pub fd: c_int,
    /// Element data type.
    pub dtype: HalDtype,
}

impl Default for HalDmabufTensorInfo {
    fn default() -> Self {
        Self {
            size: 0,
            offset: 0,
            shape: [0; HAL_DMABUF_MAX_NDIM],
            ndim: 0,
            fd: -1,
            dtype: HalDtype::U8,
        }
    }
}

// Compile-time layout assertion: 11 × size_t + 1 × c_int + 1 × HalDtype
// = 11×8 + 4 + 4 = 96 bytes on LP64 with no internal padding.
#[cfg(target_pointer_width = "64")]
const _: () = assert!(std::mem::size_of::<HalDmabufTensorInfo>() == 96);
