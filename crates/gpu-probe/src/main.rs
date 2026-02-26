// SPDX-FileCopyrightText: Copyright 2025 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

use edgefirst_bench as bench;
mod bench_dma;
mod bench_egl_image;
mod bench_fbo;
mod bench_pipeline;
mod bench_render;
mod bench_rgb_packing;
mod bench_shader;
mod bench_texture;
mod egl_context;
mod probe;
mod probe_int_textures;
mod probe_min_sizes;

use egl_context::GpuContext;

fn main() {
    env_logger::init();

    let probe_only = std::env::args().any(|a| a == "--probe-only");
    let bench_only = std::env::args().any(|a| a == "--bench-only");
    let skip_pipeline = std::env::args().any(|a| a == "--skip-pipeline");

    println!("gpu-probe: Low-level GPU platform exploration tool");
    println!("  arch: {}", std::env::consts::ARCH);
    println!();

    let ctx = match GpuContext::new() {
        Ok(ctx) => ctx,
        Err(e) => {
            eprintln!("ERROR: GPU context creation failed: {e}");
            if !bench_only {
                // Still report DMA availability even without GPU
                println!("== DMA-buf Availability (no GPU) ==");
                bench_dma::run();
            }
            std::process::exit(1);
        }
    };

    // Capability probe
    if !bench_only {
        probe::run_probes(&ctx);
        probe_int_textures::run(&ctx);
        probe_min_sizes::run(&ctx);
    }

    // Benchmarks
    if !probe_only {
        bench_dma::run();

        let has_egl_image = ctx.has_egl_create_image_khr();

        if has_egl_image {
            bench_egl_image::run(&ctx);
        } else {
            println!(
                "== Benchmark: EGL Image == SKIP (EGL_EXT_image_dma_buf_import not available)"
            );
            println!();
        }

        bench_fbo::run(&ctx);
        bench_texture::run(&ctx);
        bench_shader::run(&ctx);

        if has_egl_image {
            bench_render::run(&ctx);
        } else {
            println!("== Benchmark: Full Render Pipeline == SKIP (EGL_EXT_image_dma_buf_import not available)");
            println!();
        }

        if has_egl_image && !skip_pipeline {
            bench_pipeline::run_verify(&ctx);
            bench_pipeline::run(&ctx);
            bench_rgb_packing::run_verify(&ctx);
            bench_rgb_packing::run(&ctx);
        } else if !has_egl_image {
            println!("== Verification: DMA-buf Pipeline == SKIP (EGL_EXT_image_dma_buf_import not available)");
            println!();
            println!("== Benchmark: DMA-buf Pipeline == SKIP (EGL_EXT_image_dma_buf_import not available)");
            println!();
        } else {
            println!("== Verification: DMA-buf Pipeline == SKIP (--skip-pipeline)");
            println!();
            println!("== Benchmark: DMA-buf Pipeline == SKIP (--skip-pipeline)");
            println!();
        }
    }
}
