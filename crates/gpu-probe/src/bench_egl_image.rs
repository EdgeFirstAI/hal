// SPDX-FileCopyrightText: Copyright 2025 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! EGLImage lifecycle benchmarks — measures create/destroy and
//! create/bind/destroy latency for DMA-backed EGLImages at various
//! resolutions.

use crate::bench::{run_bench, BenchResult};
use crate::egl_context::GpuContext;
use edgefirst_tensor::{Tensor, TensorMemory, TensorTrait};
use std::os::unix::io::AsRawFd;

/// Run EGLImage lifecycle benchmarks and return collected results.
pub fn run(ctx: &GpuContext) -> Vec<BenchResult> {
    println!("== Benchmark: EGLImage Lifecycle ==");

    let fourcc: u32 = gbm::drm::buffer::DrmFourcc::Abgr8888 as u32;

    let configs: &[(u32, u32, &str)] = &[
        (640, 640, "640x640"),
        (1920, 1080, "1080p"),
        (3840, 2160, "4K"),
    ];

    let mut results = Vec::new();

    for &(w, h, label) in configs {
        let pitch = w * 4;
        let byte_count = (pitch * h) as usize;

        // Allocate a DMA tensor for this resolution.
        let tensor = match Tensor::<u8>::new(&[byte_count], Some(TensorMemory::Dma), None) {
            Ok(t) => t,
            Err(e) => {
                println!("  SKIP {label}: DMA allocation failed: {e}");
                continue;
            }
        };

        let owned_fd = tensor.clone_fd().unwrap();
        let raw_fd = owned_fd.as_raw_fd();

        // Benchmark: create + destroy
        {
            let name = format!("egl_image_create_destroy/{label}");
            let r = run_bench(&name, 10, 500, || {
                let img = ctx
                    .create_egl_image_dma(raw_fd, w as i32, h as i32, fourcc, pitch as i32)
                    .expect("create_egl_image_dma failed");
                ctx.destroy_egl_image(img)
                    .expect("destroy_egl_image failed");
            });
            r.print_summary();
            results.push(r);
        }

        // Benchmark: create + bind texture + destroy
        {
            let name = format!("egl_image_create_bind_destroy/{label}");
            let r = run_bench(&name, 10, 500, || {
                let img = ctx
                    .create_egl_image_dma(raw_fd, w as i32, h as i32, fourcc, pitch as i32)
                    .expect("create_egl_image_dma failed");

                unsafe {
                    let mut tex: u32 = 0;
                    gls::gl::GenTextures(1, &mut tex);
                    gls::gl::BindTexture(gls::gl::TEXTURE_2D, tex);
                    gls::gl::EGLImageTargetTexture2DOES(gls::gl::TEXTURE_2D, img.as_ptr());
                    gls::gl::DeleteTextures(1, &tex);
                }

                ctx.destroy_egl_image(img)
                    .expect("destroy_egl_image failed");
            });
            r.print_summary();
            results.push(r);
        }
    }

    println!();
    results
}
