// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! Wrapping CoreVideo IOSurfaces as image tensors.

#![cfg(target_os = "macos")]

use edgefirst_tensor::{Error, PixelFormat, Tensor, TensorDyn, TensorMapTrait, TensorTrait};
use std::ffi::c_void;

type CFTypeRef = *const c_void;

const FMT_420V: u32 = u32::from_be_bytes(*b"420v");
const FMT_420F: u32 = u32::from_be_bytes(*b"420f");
const FMT_YUVS: u32 = u32::from_be_bytes(*b"yuvs");
const FMT_YUVF: u32 = u32::from_be_bytes(*b"yuvf");
const FMT_2VUY: u32 = u32::from_be_bytes(*b"2vuy");

#[repr(C)]
struct CFDictionaryKeyCallBacks {
    _opaque: [usize; 6],
}

#[repr(C)]
struct CFDictionaryValueCallBacks {
    _opaque: [usize; 5],
}

#[allow(clippy::duplicated_attributes)]
#[link(name = "CoreVideo", kind = "framework")]
#[link(name = "CoreFoundation", kind = "framework")]
extern "C" {
    static kCVPixelBufferIOSurfacePropertiesKey: CFTypeRef;
    static kCFTypeDictionaryKeyCallBacks: CFDictionaryKeyCallBacks;
    static kCFTypeDictionaryValueCallBacks: CFDictionaryValueCallBacks;

    fn CFDictionaryCreate(
        allocator: CFTypeRef,
        keys: *const CFTypeRef,
        values: *const CFTypeRef,
        count: isize,
        key_callbacks: *const CFDictionaryKeyCallBacks,
        value_callbacks: *const CFDictionaryValueCallBacks,
    ) -> CFTypeRef;
    fn CFRelease(object: CFTypeRef);

    fn CVPixelBufferCreate(
        allocator: CFTypeRef,
        width: usize,
        height: usize,
        pixel_format: u32,
        attributes: CFTypeRef,
        out: *mut *mut c_void,
    ) -> i32;
    fn IOSurfaceGetID(surface: *mut c_void) -> u32;
    fn CVPixelBufferGetIOSurface(buffer: *mut c_void) -> *mut c_void;
    fn CVPixelBufferLockBaseAddress(buffer: *mut c_void, flags: u64) -> i32;
    fn CVPixelBufferUnlockBaseAddress(buffer: *mut c_void, flags: u64) -> i32;
    fn CVPixelBufferGetPlaneCount(buffer: *mut c_void) -> usize;
    fn CVPixelBufferGetBaseAddress(buffer: *mut c_void) -> *mut c_void;
    fn CVPixelBufferGetBytesPerRow(buffer: *mut c_void) -> usize;
    fn CVPixelBufferGetHeight(buffer: *mut c_void) -> usize;
    fn CVPixelBufferGetBaseAddressOfPlane(buffer: *mut c_void, plane: usize) -> *mut c_void;
    fn CVPixelBufferGetBytesPerRowOfPlane(buffer: *mut c_void, plane: usize) -> usize;
    fn CVPixelBufferGetHeightOfPlane(buffer: *mut c_void, plane: usize) -> usize;
    fn CVPixelBufferRelease(buffer: *mut c_void);
}

/// An IOSurface-backed CoreVideo buffer, released on drop.
struct CoreVideoBuffer {
    buffer: *mut c_void,
}

impl CoreVideoBuffer {
    fn new(width: usize, height: usize, format: u32) -> Self {
        let mut buffer: *mut c_void = std::ptr::null_mut();
        let rc = unsafe {
            let empty = CFDictionaryCreate(
                std::ptr::null(),
                std::ptr::null(),
                std::ptr::null(),
                0,
                &kCFTypeDictionaryKeyCallBacks,
                &kCFTypeDictionaryValueCallBacks,
            );
            let key = kCVPixelBufferIOSurfacePropertiesKey;
            let attributes = CFDictionaryCreate(
                std::ptr::null(),
                &key,
                &empty,
                1,
                &kCFTypeDictionaryKeyCallBacks,
                &kCFTypeDictionaryValueCallBacks,
            );
            let rc = CVPixelBufferCreate(
                std::ptr::null(),
                width,
                height,
                format,
                attributes,
                &mut buffer,
            );
            CFRelease(attributes);
            CFRelease(empty);
            rc
        };
        assert_eq!(rc, 0, "CVPixelBufferCreate failed (rc={rc})");
        assert!(!buffer.is_null());
        Self { buffer }
    }

    fn nv12(width: usize, height: usize, full_range: bool) -> Self {
        Self::new(width, height, if full_range { FMT_420F } else { FMT_420V })
    }

    fn surface(&self) -> *mut c_void {
        let surface = unsafe { CVPixelBufferGetIOSurface(self.buffer) };
        assert!(!surface.is_null(), "pixel buffer has no IOSurface");
        surface
    }

    /// Fills plane 0 with `y` and plane 1 with `uv`; a non-planar buffer is
    /// filled with `y`.
    fn fill(&self, y: u8, uv: u8) {
        unsafe {
            assert_eq!(CVPixelBufferLockBaseAddress(self.buffer, 0), 0);
            let planes = CVPixelBufferGetPlaneCount(self.buffer);
            if planes == 0 {
                let base = CVPixelBufferGetBaseAddress(self.buffer) as *mut u8;
                let pitch = CVPixelBufferGetBytesPerRow(self.buffer);
                std::ptr::write_bytes(base, y, pitch * CVPixelBufferGetHeight(self.buffer));
            }
            for plane in 0..planes {
                let base = CVPixelBufferGetBaseAddressOfPlane(self.buffer, plane) as *mut u8;
                let pitch = CVPixelBufferGetBytesPerRowOfPlane(self.buffer, plane);
                let rows = CVPixelBufferGetHeightOfPlane(self.buffer, plane);
                let value = if plane == 0 { y } else { uv };
                std::ptr::write_bytes(base, value, pitch * rows);
            }
            assert_eq!(CVPixelBufferUnlockBaseAddress(self.buffer, 0), 0);
        }
    }
}

impl Drop for CoreVideoBuffer {
    fn drop(&mut self) {
        unsafe { CVPixelBufferRelease(self.buffer) };
    }
}

fn wrap(surface: *mut c_void) -> Result<TensorDyn, Error> {
    unsafe { TensorDyn::from_planar_iosurface(surface, None) }
}

fn plane(t: &TensorDyn) -> &Tensor<u8> {
    t.as_u8().expect("u8 tensor")
}

/// Asserts that both planes of `luma` wrap `surface`.
fn assert_shared_surface(luma: &Tensor<u8>, surface: *mut c_void) {
    let chroma = luma.chroma().expect("chroma plane");
    let id = |t: &Tensor<u8>| unsafe { IOSurfaceGetID(t.iosurface_ref().expect("surface")) };
    let expected = unsafe { IOSurfaceGetID(surface) };
    assert_ne!(expected, 0);
    assert_eq!(id(luma), expected);
    assert_eq!(id(chroma), expected);
}

/// Asserts that row `row` of `plane`, `row_bytes` wide, reads `value`.
fn assert_row(plane: &Tensor<u8>, row: usize, row_bytes: usize, value: u8) {
    let stride = plane.effective_row_stride().expect("row stride");
    let map = plane.map().expect("map");
    let bytes = map.as_slice();
    let row_data = &bytes[row * stride..row * stride + row_bytes];
    assert!(
        row_data.iter().all(|&b| b == value),
        "row {row} is not all {value:#04x}"
    );
}

#[test]
fn a_corevideo_nv12_buffer_wraps_as_multiplane() {
    let buf = CoreVideoBuffer::nv12(1920, 1080, false);
    buf.fill(0x10, 0x80);
    let t = wrap(buf.surface()).expect("wrap");

    assert_eq!(t.format(), Some(PixelFormat::Nv12));
    assert!(t.is_multiplane());
    assert_eq!(t.width(), Some(1920));
    assert_eq!(t.height(), Some(1080));

    let luma = plane(&t);
    assert_eq!(luma.effective_row_stride(), Some(1920));
    assert_eq!(luma.plane_offset(), Some(512));
    let chroma = luma.chroma().expect("chroma plane");
    assert_eq!(chroma.plane_offset(), Some(2_074_112));
    assert_shared_surface(luma, buf.surface());

    assert_row(luma, 0, 1920, 0x10);
    assert_row(luma, 1079, 1920, 0x10);
    assert_row(chroma, 0, 1920, 0x80);
    assert_row(chroma, 539, 1920, 0x80);
}

#[test]
fn a_padded_corevideo_nv12_buffer_wraps_at_its_planes() {
    let buf = CoreVideoBuffer::nv12(1000, 562, false);
    buf.fill(0x10, 0x80);
    let t = wrap(buf.surface()).expect("wrap");

    assert_eq!(t.width(), Some(1000));
    assert_eq!(t.height(), Some(562));
    let luma = plane(&t);
    assert_eq!(luma.effective_row_stride(), Some(1024));
    let chroma = luma.chroma().expect("chroma plane");
    assert_eq!(chroma.plane_offset(), Some(582_144));
    assert_shared_surface(luma, buf.surface());

    assert_row(luma, 561, 1000, 0x10);
    assert_row(chroma, 280, 1000, 0x80);
}

#[test]
fn a_full_range_buffer_also_wraps() {
    let buf = CoreVideoBuffer::nv12(640, 480, true);
    let t = wrap(buf.surface()).expect("wrap");
    assert_eq!(t.format(), Some(PixelFormat::Nv12));
    assert!(t.is_multiplane());
}

#[test]
fn an_unknown_pixel_format_is_refused() {
    let buf = CoreVideoBuffer::new(640, 480, FMT_2VUY);
    match wrap(buf.surface()) {
        Err(Error::InvalidArgument(msg)) => {
            assert!(msg.contains("2vuy"), "message lacks the format: {msg}")
        }
        other => panic!("expected InvalidArgument, got {other:?}"),
    }
}

#[test]
fn a_null_surface_is_refused() {
    assert!(matches!(
        wrap(std::ptr::null_mut()),
        Err(Error::InvalidArgument(_))
    ));
}

#[test]
fn a_yuvs_buffer_wraps_as_one_yuyv_tensor() {
    let buf = CoreVideoBuffer::new(640, 480, FMT_YUVS);
    let t = wrap(buf.surface()).expect("wrap");
    assert_eq!(t.format(), Some(PixelFormat::Yuyv));
    assert!(!t.is_multiplane());
    assert_eq!(t.width(), Some(640));
    assert_eq!(t.height(), Some(480));
    assert_eq!(t.effective_row_stride(), Some(1280));
}

#[test]
fn a_yuvf_buffer_wraps_as_one_yuyv_tensor() {
    let buf = CoreVideoBuffer::new(640, 480, FMT_YUVF);
    let t = wrap(buf.surface()).expect("wrap");
    assert_eq!(t.format(), Some(PixelFormat::Yuyv));
    assert!(!t.is_multiplane());
    assert_eq!(t.width(), Some(640));
    assert_eq!(t.height(), Some(480));
    assert_eq!(t.effective_row_stride(), Some(1280));
    assert_eq!(t.colorimetry(), None);
}

#[test]
fn the_wrap_retains_its_surface() {
    let buf = CoreVideoBuffer::nv12(640, 480, false);
    buf.fill(0x10, 0x80);
    let t = wrap(buf.surface()).expect("wrap");
    drop(buf);

    let luma = plane(&t);
    assert_row(luma, 479, 640, 0x10);
    assert_row(luma.chroma().expect("chroma plane"), 239, 640, 0x80);
}
