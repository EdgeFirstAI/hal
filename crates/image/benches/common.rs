// SPDX-FileCopyrightText: Copyright 2025 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! Shared utilities for image processing benchmarks.
#![allow(dead_code)]

use edgefirst_image::{NV12, RGB, RGBA, TensorImage, YUYV};
use edgefirst_tensor::TensorMemory;
use four_char_code::FourCharCode;
use std::{path::Path, sync::OnceLock};

#[cfg(target_os = "linux")]
use edgefirst_image::G2DProcessor;

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
    *OPENGL_AVAILABLE.get_or_init(|| dma_available() && GLProcessorThreaded::new().is_ok())
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
    } else if f == NV12 {
        "NV12"
    } else if f == RGB {
        "RGB"
    } else if f == RGBA {
        "RGBA"
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
// Benchmark Configuration (for Criterion benchmarks)
// =============================================================================

use criterion::Throughput;

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

    /// Calculate throughput based on input size.
    pub fn throughput(&self) -> Throughput {
        let bytes = match self.in_fmt {
            f if f == YUYV => self.in_w * self.in_h * 2,
            f if f == NV12 => self.in_w * self.in_h * 3 / 2,
            f if f == RGB => self.in_w * self.in_h * 3,
            f if f == RGBA => self.in_w * self.in_h * 4,
            _ => self.in_w * self.in_h,
        };
        Throughput::Bytes(bytes as u64)
    }
}

// =============================================================================
// Embedded Test Data (for Criterion benchmarks)
// =============================================================================

pub const CAMERA_1080P_YUYV: &[u8] = include_bytes!("../../../testdata/camera1080p.yuyv");
pub const CAMERA_1080P_NV12: &[u8] = include_bytes!("../../../testdata/camera1080p.nv12");
pub const CAMERA_1080P_RGB: &[u8] = include_bytes!("../../../testdata/camera1080p.rgb");
pub const CAMERA_4K_YUYV: &[u8] = include_bytes!("../../../testdata/camera4k.yuyv");
pub const CAMERA_4K_NV12: &[u8] = include_bytes!("../../../testdata/camera4k.nv12");
pub const CAMERA_4K_RGB: &[u8] = include_bytes!("../../../testdata/camera4k.rgb");

/// Get embedded test data for a given resolution and format.
pub fn get_test_data(width: usize, height: usize, format: FourCharCode) -> &'static [u8] {
    match (width, height, format) {
        (1920, 1080, f) if f == YUYV => CAMERA_1080P_YUYV,
        (1920, 1080, f) if f == NV12 => CAMERA_1080P_NV12,
        (1920, 1080, f) if f == RGB => CAMERA_1080P_RGB,
        (3840, 2160, f) if f == YUYV => CAMERA_4K_YUYV,
        (3840, 2160, f) if f == NV12 => CAMERA_4K_NV12,
        (3840, 2160, f) if f == RGB => CAMERA_4K_RGB,
        _ => panic!("No test data for {}x{} {:?}", width, height, format),
    }
}
