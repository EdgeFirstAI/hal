// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! Shared test-only helpers for reporting a skipped test. The report is written
//! directly to `std::io::stderr()`, which libtest's output capture does not
//! hide.

/// Reports a test skip with `reason`, writing `SKIPPED: {reason}` directly to
/// stderr (not via `eprintln!`).
#[cfg(test)]
pub(crate) fn report_skip(reason: &str) {
    use std::io::Write;
    let _ = writeln!(&mut std::io::stderr(), "WARNING: SKIPPED: {reason}");
}

#[cfg(test)]
#[path = "../tests/support/dma_require.rs"]
mod dma_require;

/// `true` when a DMA-heap test should run; `false` after reporting a skip.
///
/// # Panics
///
/// When `HAL_TEST_REQUIRE_DMA=1` and no DMA heap is usable.
#[cfg(test)]
pub(crate) fn dma_or_skip(what: &str) -> bool {
    dma_require::available_or_skip(crate::is_dma_available(), what)
}

#[cfg(test)]
#[path = "../tests/support/outcome.rs"]
#[allow(dead_code)]
mod outcome;
#[cfg(test)]
#[allow(unused_imports)]
pub(crate) use outcome::{allocated, Expect};

/// Unwraps a DMA-BUF allocation: it must succeed when the heap is usable (under
/// `HAL_TEST_REQUIRE_DMA`), and may fail with a warning otherwise.
#[cfg(test)]
#[allow(dead_code)]
pub(crate) fn dma_allocated<T, E: std::fmt::Debug>(result: Result<T, E>, what: &str) -> Option<T> {
    match result {
        Ok(value) => Some(value),
        Err(e) => {
            let usable = dma_require::decide(
                crate::is_dma_available(),
                dma_require::required(),
                what,
                "no usable DMA heap",
            );
            allocated(Err(e), what, outcome::expect_dma(usable))
        }
    }
}

#[cfg(target_os = "macos")]
pub(crate) use corevideo::corevideo_nv12;

#[cfg(target_os = "macos")]
mod corevideo {
    use std::ffi::c_void;

    type CFTypeRef = *const c_void;

    const K_CF_STRING_ENCODING_UTF8: u32 = 0x0800_0100;
    const K_CV_PIXEL_FORMAT_420_VIDEO_RANGE: u32 = 0x3432_3076;
    const K_CV_PIXEL_FORMAT_420_FULL_RANGE: u32 = 0x3432_3066;

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
        fn CFStringCreateWithCString(
            allocator: *const c_void,
            cstr: *const i8,
            encoding: u32,
        ) -> *mut c_void;
        fn CFRelease(object: CFTypeRef);

        fn CVPixelBufferCreate(
            allocator: CFTypeRef,
            width: usize,
            height: usize,
            pixel_format: u32,
            attributes: CFTypeRef,
            out: *mut *mut c_void,
        ) -> i32;
        fn CVPixelBufferGetIOSurface(buffer: *mut c_void) -> *mut c_void;
        fn CVPixelBufferLockBaseAddress(buffer: *mut c_void, flags: u64) -> i32;
        fn CVPixelBufferUnlockBaseAddress(buffer: *mut c_void, flags: u64) -> i32;
        fn CVPixelBufferGetPlaneCount(buffer: *mut c_void) -> usize;
        fn CVPixelBufferGetBaseAddressOfPlane(buffer: *mut c_void, plane: usize) -> *mut c_void;
        fn CVPixelBufferGetBytesPerRowOfPlane(buffer: *mut c_void, plane: usize) -> usize;
        fn CVPixelBufferGetHeightOfPlane(buffer: *mut c_void, plane: usize) -> usize;
        fn CVPixelBufferRelease(buffer: *mut c_void);
    }

    /// An IOSurface-backed CoreVideo NV12 buffer, released on drop.
    pub(crate) struct CoreVideoBuffer {
        buffer: *mut c_void,
    }

    /// Creates an IOSurface-backed `420v` (or `420f` when `full_range`) buffer.
    pub(crate) fn corevideo_nv12(width: usize, height: usize, full_range: bool) -> CoreVideoBuffer {
        let format = if full_range {
            K_CV_PIXEL_FORMAT_420_FULL_RANGE
        } else {
            K_CV_PIXEL_FORMAT_420_VIDEO_RANGE
        };
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
            let key = CFStringCreateWithCString(
                std::ptr::null(),
                c"IOSurfaceProperties".as_ptr(),
                K_CF_STRING_ENCODING_UTF8,
            );
            assert!(!key.is_null());
            let attributes = CFDictionaryCreate(
                std::ptr::null(),
                &(key as CFTypeRef),
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
            CFRelease(key);
            CFRelease(empty);
            rc
        };
        assert_eq!(rc, 0, "CVPixelBufferCreate failed (rc={rc})");
        assert!(!buffer.is_null());
        CoreVideoBuffer { buffer }
    }

    impl CoreVideoBuffer {
        /// The backing `IOSurfaceRef`, valid while this buffer lives.
        pub(crate) fn surface(&self) -> *mut c_void {
            let surface = unsafe { CVPixelBufferGetIOSurface(self.buffer) };
            assert!(!surface.is_null(), "pixel buffer has no IOSurface");
            surface
        }

        /// Fills plane 0 with `y` and plane 1 with `uv`, row by row at each
        /// plane's own pitch.
        pub(crate) fn fill(&self, y: u8, uv: u8) {
            unsafe {
                assert_eq!(CVPixelBufferLockBaseAddress(self.buffer, 0), 0);
                for plane in 0..CVPixelBufferGetPlaneCount(self.buffer) {
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
}
