// SPDX-FileCopyrightText: Copyright 2025 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! DMA-buf and fstat benchmarks — measures DMA allocation latency at various
//! buffer sizes and compares fstat syscall cost on DMA-buf fds versus
//! /dev/null.

use crate::bench::{run_bench, BenchResult};
use edgefirst_tensor::{Tensor, TensorMemory, TensorTrait};
use std::hint::black_box;

/// Run DMA-buf and fstat benchmarks and return collected results.
pub fn run() -> Vec<BenchResult> {
    println!("== Benchmark: DMA-buf Operations ==");

    let mut results = Vec::new();

    // 1. DMA allocation at various sizes
    let alloc_configs: &[(usize, &str)] = &[
        (640 * 640 * 4, "640x640_RGBA"),
        (1920 * 1080 * 4, "1080p_RGBA"),
        (1920 * 1080 * 3 / 2, "1080p_NV12"),
        (3840 * 2160 * 4, "4K_RGBA"),
    ];

    for &(byte_count, label) in alloc_configs {
        let name = format!("dma_alloc/{label}");

        // Verify DMA is available before benchmarking.
        match Tensor::<u8>::new(&[byte_count], Some(TensorMemory::Dma), None) {
            Ok(_) => {}
            Err(e) => {
                println!("  SKIP {label}: DMA allocation failed: {e}");
                continue;
            }
        }

        let r = run_bench(&name, 5, 100, || {
            let t = Tensor::<u8>::new(&[byte_count], Some(TensorMemory::Dma), None);
            black_box(&t);
        });
        r.print_summary();
        results.push(r);
    }

    // 2. fstat on DMA-buf fd
    {
        let tensor_result = Tensor::<u8>::new(&[1920 * 1080 * 4], Some(TensorMemory::Dma), None);
        match tensor_result {
            Ok(tensor) => {
                let owned_fd = tensor.clone_fd().unwrap();

                let r = run_bench("fstat_dma_fd", 100, 10000, || {
                    let stat = nix::sys::stat::fstat(&owned_fd).unwrap();
                    black_box(stat.st_ino);
                });
                r.print_summary();
                results.push(r);
            }
            Err(e) => {
                println!("  SKIP fstat_dma_fd: DMA allocation failed: {e}");
            }
        }
    }

    // 3. fstat on /dev/null (comparison baseline)
    {
        let dev_null = std::fs::File::open("/dev/null").expect("failed to open /dev/null");

        let r = run_bench("fstat_dev_null", 100, 10000, || {
            let stat = nix::sys::stat::fstat(&dev_null).unwrap();
            black_box(stat.st_ino);
        });
        r.print_summary();
        results.push(r);
    }

    println!();
    results
}
