// SPDX-FileCopyrightText: Copyright 2025 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! Shared utilities for image processing benchmarks.
#![allow(dead_code, unused_imports)]

use edgefirst_image::{
    BGRA, GREY, NV12, NV16, PLANAR_RGB, PLANAR_RGB_INT8, RGB, RGBA, RGB_INT8, VYUY, YUYV,
};
use four_char_code::FourCharCode;
use std::path::Path;
use std::sync::{LazyLock, OnceLock};

#[cfg(target_os = "linux")]
use edgefirst_image::{G2DProcessor, TensorImage};
#[cfg(target_os = "linux")]
use edgefirst_tensor::TensorMemory;

// Re-export the benchmark harness from the shared crate.
pub use edgefirst_bench::{run_bench, BenchResult, BenchSuite};

// =============================================================================
// Hardware Availability Cache
// =============================================================================

static G2D_AVAILABLE: OnceLock<bool> = OnceLock::new();
#[cfg(all(target_os = "linux", feature = "opengl"))]
static OPENGL_AVAILABLE: OnceLock<bool> = OnceLock::new();

// =============================================================================
// Hardware Detection
// =============================================================================

/// Check if DMA memory allocation is available on this system.
#[cfg(target_os = "linux")]
pub fn dma_available() -> bool {
    TensorImage::new(64, 64, RGBA, Some(TensorMemory::Dma)).is_ok()
}

#[cfg(not(target_os = "linux"))]
pub fn dma_available() -> bool {
    false
}

/// Check if G2D hardware acceleration is available.
#[cfg(target_os = "linux")]
pub fn g2d_available() -> bool {
    *G2D_AVAILABLE.get_or_init(|| dma_available() && G2DProcessor::new().is_ok())
}

#[cfg(not(target_os = "linux"))]
pub fn g2d_available() -> bool {
    false
}

/// Check if OpenGL hardware acceleration is available.
#[cfg(all(target_os = "linux", feature = "opengl"))]
pub fn opengl_available() -> bool {
    use edgefirst_image::GLProcessorThreaded;
    *OPENGL_AVAILABLE.get_or_init(|| dma_available() && GLProcessorThreaded::new(None).is_ok())
}

#[cfg(not(all(target_os = "linux", feature = "opengl")))]
pub fn opengl_available() -> bool {
    false
}

// =============================================================================
// Test Data
// =============================================================================

/// Locate a test file in the testdata directory, handling different working
/// directories.
pub fn find_testdata_path(filename: &str) -> std::path::PathBuf {
    let candidates = [
        Path::new("testdata").join(filename),
        Path::new("../testdata").join(filename),
        Path::new("../../testdata").join(filename),
        Path::new("../../../testdata").join(filename),
    ];

    for path in &candidates {
        if path.exists() {
            return path.clone();
        }
    }

    panic!("Unable to locate test file: {filename}");
}

/// Get the human-readable name for a FourCC format.
pub fn format_name(f: FourCharCode) -> &'static str {
    if f == YUYV {
        "YUYV"
    } else if f == VYUY {
        "VYUY"
    } else if f == NV12 {
        "NV12"
    } else if f == RGB {
        "RGB"
    } else if f == RGB_INT8 {
        "RGBi"
    } else if f == RGBA {
        "RGBA"
    } else if f == PLANAR_RGB {
        "8BPS"
    } else if f == PLANAR_RGB_INT8 {
        "8BPi"
    } else if f == BGRA {
        "BGRA"
    } else if f == NV16 {
        "NV16"
    } else if f == GREY {
        "GREY"
    } else {
        "???"
    }
}

// =============================================================================
// Letterbox Calculation
// =============================================================================

/// Calculate letterbox dimensions to fit source into destination while
/// preserving aspect ratio.
///
/// Returns (left_offset, top_offset, scaled_width, scaled_height).
pub fn calculate_letterbox(
    src_w: usize,
    src_h: usize,
    dst_w: usize,
    dst_h: usize,
) -> (usize, usize, usize, usize) {
    let src_aspect = src_w as f64 / src_h as f64;
    let dst_aspect = dst_w as f64 / dst_h as f64;

    let (new_w, new_h) = if src_aspect > dst_aspect {
        let new_h = (dst_w as f64 / src_aspect).round() as usize;
        (dst_w, new_h)
    } else {
        let new_w = (dst_h as f64 * src_aspect).round() as usize;
        (new_w, dst_h)
    };

    let left = (dst_w - new_w) / 2;
    let top = (dst_h - new_h) / 2;

    (left, top, new_w, new_h)
}

// =============================================================================
// Benchmark Configuration
// =============================================================================

/// Configuration for a single benchmark case.
#[derive(Clone)]
pub struct BenchConfig {
    pub in_w: usize,
    pub in_h: usize,
    pub out_w: usize,
    pub out_h: usize,
    pub in_fmt: FourCharCode,
    pub out_fmt: FourCharCode,
}

impl BenchConfig {
    /// Create a new benchmark configuration.
    pub fn new(
        in_w: usize,
        in_h: usize,
        out_w: usize,
        out_h: usize,
        in_fmt: FourCharCode,
        out_fmt: FourCharCode,
    ) -> Self {
        Self {
            in_w,
            in_h,
            out_w,
            out_h,
            in_fmt,
            out_fmt,
        }
    }

    /// Generate a human-readable benchmark ID.
    pub fn id(&self) -> String {
        if self.in_w == self.out_w && self.in_h == self.out_h {
            format!(
                "{}x{}/{}->{}",
                self.in_w,
                self.in_h,
                format_name(self.in_fmt),
                format_name(self.out_fmt)
            )
        } else {
            format!(
                "{}x{}/{}->{}x{}/{}",
                self.in_w,
                self.in_h,
                format_name(self.in_fmt),
                self.out_w,
                self.out_h,
                format_name(self.out_fmt)
            )
        }
    }

    /// Calculate throughput in bytes based on input size.
    pub fn throughput(&self) -> u64 {
        let bytes = match self.in_fmt {
            f if f == YUYV || f == VYUY => self.in_w * self.in_h * 2,
            f if f == NV12 => self.in_w * self.in_h * 3 / 2,
            f if f == NV16 => self.in_w * self.in_h * 2, // 4:2:2 like YUYV
            f if f == RGB || f == RGB_INT8 || f == PLANAR_RGB || f == PLANAR_RGB_INT8 => {
                self.in_w * self.in_h * 3
            }
            f if f == RGBA => self.in_w * self.in_h * 4,
            f if f == BGRA => self.in_w * self.in_h * 4,
            f if f == GREY => self.in_w * self.in_h,
            _ => self.in_w * self.in_h,
        };
        bytes as u64
    }
}

// =============================================================================
// Embedded Test Data
// =============================================================================

pub const CAMERA_720P_YUYV: &[u8] = include_bytes!("../../../testdata/camera720p.yuyv");
pub const CAMERA_720P_VYUY: &[u8] = include_bytes!("../../../testdata/camera720p.vyuy");
pub const CAMERA_720P_NV12: &[u8] = include_bytes!("../../../testdata/camera720p.nv12");
pub const CAMERA_720P_RGB: &[u8] = include_bytes!("../../../testdata/camera720p.rgb");
pub const CAMERA_1080P_YUYV: &[u8] = include_bytes!("../../../testdata/camera1080p.yuyv");
pub const CAMERA_1080P_NV12: &[u8] = include_bytes!("../../../testdata/camera1080p.nv12");
pub const CAMERA_1080P_RGB: &[u8] = include_bytes!("../../../testdata/camera1080p.rgb");
pub const CAMERA_1080P_VYUY: &[u8] = include_bytes!("../../../testdata/camera1080p.vyuy");
pub const CAMERA_4K_YUYV: &[u8] = include_bytes!("../../../testdata/camera4k.yuyv");
pub const CAMERA_4K_VYUY: &[u8] = include_bytes!("../../../testdata/camera4k.vyuy");
pub const CAMERA_4K_NV12: &[u8] = include_bytes!("../../../testdata/camera4k.nv12");
pub const CAMERA_4K_RGB: &[u8] = include_bytes!("../../../testdata/camera4k.rgb");

// NV16 is 4:2:2 semi-planar: Y plane (W*H) + interleaved UV plane (W*H).
// No real capture file available; synthetic mid-gray data is sufficient for
// throughput benchmarks.
static CAMERA_1080P_NV16: LazyLock<Vec<u8>> = LazyLock::new(|| vec![128u8; 1920 * 1080 * 2]);

// RGBA synthetic data for format conversion benchmarks (e.g. RGBA -> BGRA,
// RGBA -> GREY). No real capture file needed; mid-gray is sufficient for
// throughput measurement.
static CAMERA_1080P_RGBA: LazyLock<Vec<u8>> = LazyLock::new(|| vec![128u8; 1920 * 1080 * 4]);

/// Get embedded test data for a given resolution and format.
pub fn get_test_data(width: usize, height: usize, format: FourCharCode) -> &'static [u8] {
    match (width, height, format) {
        (1280, 720, f) if f == YUYV => CAMERA_720P_YUYV,
        (1280, 720, f) if f == VYUY => CAMERA_720P_VYUY,
        (1280, 720, f) if f == NV12 => CAMERA_720P_NV12,
        (1280, 720, f) if f == RGB => CAMERA_720P_RGB,
        (1920, 1080, f) if f == YUYV => CAMERA_1080P_YUYV,
        (1920, 1080, f) if f == VYUY => CAMERA_1080P_VYUY,
        (1920, 1080, f) if f == NV12 => CAMERA_1080P_NV12,
        (1920, 1080, f) if f == NV16 => &CAMERA_1080P_NV16,
        (1920, 1080, f) if f == RGB => CAMERA_1080P_RGB,
        (1920, 1080, f) if f == RGBA => &CAMERA_1080P_RGBA,
        (3840, 2160, f) if f == YUYV => CAMERA_4K_YUYV,
        (3840, 2160, f) if f == VYUY => CAMERA_4K_VYUY,
        (3840, 2160, f) if f == NV12 => CAMERA_4K_NV12,
        (3840, 2160, f) if f == RGB => CAMERA_4K_RGB,
        _ => panic!("No test data for {}x{} {:?}", width, height, format),
    }
}
