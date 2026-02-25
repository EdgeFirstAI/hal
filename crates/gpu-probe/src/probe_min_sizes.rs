// SPDX-FileCopyrightText: Copyright 2025 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! Minimum EGLImage size probe — determines the smallest texture dimensions
//! each DRM fourcc format supports via EGLImage import. Some GPU drivers
//! (notably ARM Mali) reject small R8 and RG88 EGLImages, so knowing the
//! minimum accepted size is essential for choosing the right format at the
//! right resolution.

use crate::egl_context::GpuContext;
use edgefirst_tensor::{Tensor, TensorMemory, TensorTrait};
use std::os::unix::io::AsRawFd;

/// Candidate square sizes to probe, ordered smallest to largest.
/// Includes powers of two plus common ML inference dimensions.
const SIZES: &[u32] = &[1, 2, 4, 8, 16, 32, 48, 64, 128, 256, 320, 512, 640, 1024];

struct FormatSpec {
    name: &'static str,
    fourcc: u32,
    bpp: u32, // bytes per pixel
}

const FORMATS: &[FormatSpec] = &[
    FormatSpec {
        name: "R8",
        fourcc: gbm::drm::buffer::DrmFourcc::R8 as u32,
        bpp: 1,
    },
    FormatSpec {
        name: "GR88 (RG)",
        fourcc: gbm::drm::buffer::DrmFourcc::Gr88 as u32,
        bpp: 2,
    },
    FormatSpec {
        name: "ABGR8888 (RGBA)",
        fourcc: gbm::drm::buffer::DrmFourcc::Abgr8888 as u32,
        bpp: 4,
    },
];

/// Try to create an EGLImage at the given size and format.
/// Returns true if creation succeeds.
fn try_egl_image(ctx: &GpuContext, fourcc: u32, bpp: u32, w: u32, h: u32) -> bool {
    let pitch = w * bpp;
    let byte_count = (pitch * h) as usize;

    let tensor = match Tensor::<u8>::new(&[byte_count], Some(TensorMemory::Dma), None) {
        Ok(t) => t,
        Err(_) => return false,
    };

    let fd_owned = match tensor.clone_fd() {
        Ok(fd) => fd,
        Err(_) => return false,
    };

    match ctx.create_egl_image_dma(
        fd_owned.as_raw_fd(),
        w as i32,
        h as i32,
        fourcc,
        pitch as i32,
    ) {
        Ok(img) => {
            let _ = ctx.destroy_egl_image(img);
            true
        }
        Err(_) => false,
    }
}

/// Run the minimum EGLImage size probe for each format.
pub fn run(ctx: &GpuContext) {
    println!("=== Minimum EGLImage Size by Format ===");

    for fmt in FORMATS {
        print!("  {:<20}", fmt.name);

        let mut min_size: Option<u32> = None;
        let mut results = Vec::new();

        for &size in SIZES {
            let ok = try_egl_image(ctx, fmt.fourcc, fmt.bpp, size, size);
            results.push((size, ok));
            if ok && min_size.is_none() {
                min_size = Some(size);
            }
        }

        match min_size {
            Some(min) => print!("min={min}x{min}"),
            None => print!("NOT SUPPORTED"),
        }

        // Print compact pass/fail for each size
        print!("  [");
        for (i, &(size, ok)) in results.iter().enumerate() {
            if i > 0 {
                print!(" ");
            }
            print!("{size}:{}", if ok { "ok" } else { "FAIL" });
        }
        println!("]");
    }

    // Test the actual R8 output dimensions used by RGB packing strategies
    // at standard pipeline resolutions.
    println!();
    println!("  R8 EGLImage at pipeline output dimensions:");
    let r8_fourcc = gbm::drm::buffer::DrmFourcc::R8 as u32;
    let r8_configs: &[(u32, u32, &str)] = &[
        // packed_r8: dst_w*3 x dst_h
        (1920, 640, "1920x640  (packed_r8 for 640x640)"),
        (960, 320, "960x320   (packed_r8 for 320x320)"),
        // planar_r8: dst_w x dst_h*3
        (640, 1920, "640x1920  (planar_r8 for 640x640)"),
        (320, 960, "320x960   (planar_r8 for 320x320)"),
    ];
    for &(w, h, label) in r8_configs {
        let ok = try_egl_image(ctx, r8_fourcc, 1, w, h);
        println!(
            "    {:<42} {}",
            label,
            if ok { "ok" } else { "FAIL" }
        );
    }

    // Test RGBA at the non-square output dimensions used by packing strategies
    println!();
    println!("  RGBA EGLImage at pipeline output dimensions:");
    let rgba_fourcc = gbm::drm::buffer::DrmFourcc::Abgr8888 as u32;
    let rgba_configs: &[(u32, u32, &str)] = &[
        // packed_rgba8: dst_w*3/4 x dst_h
        (480, 640, "480x640   (packed_rgba8 for 640x640)"),
        (240, 320, "240x320   (packed_rgba8 for 320x320)"),
        // planar_rgba8: dst_w/4 x dst_h*3
        (160, 1920, "160x1920  (planar_rgba8 for 640x640)"),
        (80, 960, "80x960    (planar_rgba8 for 320x320)"),
        // Standard output sizes
        (640, 640, "640x640   (standard output)"),
        (320, 320, "320x320   (standard output)"),
        (1920, 1080, "1920x1080 (standard input)"),
    ];
    for &(w, h, label) in rgba_configs {
        let ok = try_egl_image(ctx, rgba_fourcc, 4, w, h);
        println!(
            "    {:<42} {}",
            label,
            if ok { "ok" } else { "FAIL" }
        );
    }

    println!();
}
