// SPDX-FileCopyrightText: Copyright 2025 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

mod bench;
mod bench_dma;
mod bench_egl_image;
mod bench_fbo;
mod bench_pipeline;
mod bench_render;
mod bench_shader;
mod bench_texture;
mod egl_context;
mod probe;

use egl_context::GpuContext;

fn main() {
    env_logger::init();

    let probe_only = std::env::args().any(|a| a == "--probe-only");
    let bench_only = std::env::args().any(|a| a == "--bench-only");

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
    }

    // Benchmarks
    if !probe_only {
        bench_dma::run();
        bench_egl_image::run(&ctx);
        bench_fbo::run(&ctx);
        bench_texture::run(&ctx);
        bench_shader::run(&ctx);
        bench_render::run(&ctx);
        bench_pipeline::run_verify(&ctx);
        bench_pipeline::run(&ctx);
    }
}
