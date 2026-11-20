// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! Per-plane geometry of a planar IOSurface.

#![cfg(any(target_os = "macos", target_os = "ios"))]

use std::ffi::c_void;

#[link(name = "IOSurface", kind = "framework")]
extern "C" {
    fn IOSurfaceGetPlaneCount(surface: *mut c_void) -> usize;
    fn IOSurfaceGetPixelFormat(surface: *mut c_void) -> u32;
    fn IOSurfaceGetAllocSize(surface: *mut c_void) -> usize;
    fn IOSurfaceGetWidth(surface: *mut c_void) -> usize;
    fn IOSurfaceGetHeight(surface: *mut c_void) -> usize;
    fn IOSurfaceGetBytesPerRow(surface: *mut c_void) -> usize;
    fn IOSurfaceGetBaseAddress(surface: *mut c_void) -> *mut c_void;
    fn IOSurfaceGetBaseAddressOfPlane(surface: *mut c_void, plane: usize) -> *mut c_void;
    fn IOSurfaceGetBytesPerRowOfPlane(surface: *mut c_void, plane: usize) -> usize;
    fn IOSurfaceGetWidthOfPlane(surface: *mut c_void, plane: usize) -> usize;
    fn IOSurfaceGetHeightOfPlane(surface: *mut c_void, plane: usize) -> usize;
    fn IOSurfaceGetBytesPerElementOfPlane(surface: *mut c_void, plane: usize) -> usize;
}

/// Location and shape of one plane inside an IOSurface.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct PlaneGeometry {
    /// Byte offset of the plane from the surface base address.
    pub offset: usize,
    /// Row pitch of the plane in bytes.
    pub bytes_per_row: usize,
    /// Plane width in elements.
    pub width: usize,
    /// Plane height in rows.
    pub height: usize,
    /// Bytes per element of the plane.
    pub bytes_per_element: usize,
}

/// Geometry of each plane of `surface`; empty when the surface is not planar.
///
/// # Safety
///
/// `surface` must be a valid live `IOSurfaceRef`.
pub(crate) unsafe fn plane_geometry(surface: *mut c_void) -> Vec<PlaneGeometry> {
    unsafe {
        let count = IOSurfaceGetPlaneCount(surface);
        let base = IOSurfaceGetBaseAddress(surface) as usize;
        (0..count)
            .map(|p| PlaneGeometry {
                offset: (IOSurfaceGetBaseAddressOfPlane(surface, p) as usize).wrapping_sub(base),
                bytes_per_row: IOSurfaceGetBytesPerRowOfPlane(surface, p),
                width: IOSurfaceGetWidthOfPlane(surface, p),
                height: IOSurfaceGetHeightOfPlane(surface, p),
                bytes_per_element: IOSurfaceGetBytesPerElementOfPlane(surface, p),
            })
            .collect()
    }
}

/// The surface's four-character pixel format code, or 0 when it has none.
///
/// # Safety
///
/// `surface` must be a valid live `IOSurfaceRef`.
pub(crate) unsafe fn pixel_format(surface: *mut c_void) -> u32 {
    unsafe { IOSurfaceGetPixelFormat(surface) }
}

/// Overall size of an IOSurface.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct SurfaceExtent {
    /// Surface width in pixels.
    pub width: usize,
    /// Surface height in rows.
    pub height: usize,
    /// Row pitch in bytes; meaningful only for a non-planar surface.
    pub bytes_per_row: usize,
    /// Allocated size in bytes.
    pub alloc_size: usize,
}

/// Width, height, pitch and allocated size of `surface`.
///
/// # Safety
///
/// `surface` must be a valid live `IOSurfaceRef`.
pub(crate) unsafe fn surface_extent(surface: *mut c_void) -> SurfaceExtent {
    unsafe {
        SurfaceExtent {
            width: IOSurfaceGetWidth(surface),
            height: IOSurfaceGetHeight(surface),
            bytes_per_row: IOSurfaceGetBytesPerRow(surface),
            alloc_size: IOSurfaceGetAllocSize(surface),
        }
    }
}
