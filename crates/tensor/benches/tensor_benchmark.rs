// SPDX-FileCopyrightText: Copyright 2025 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! Tensor memory allocation and mapping benchmarks.
//!
//! Benchmarks alloc and map for Mem, Shm (Linux), and Dma (Linux + DMA)
//! backends across four resolutions using the fork-free `edgefirst-bench`
//! harness. Tests both u8 (unsigned) and i8 (signed, used for int8 models).
//!
//! ## Run benchmarks (host)
//! ```bash
//! cargo bench -p edgefirst-tensor --bench tensor_benchmark -- --bench
//! ```
//!
//! ## Run benchmarks (on-target, cross-compiled)
//! ```bash
//! ./tensor_benchmark --bench
//! ```

#[cfg(target_os = "linux")]
use edgefirst_tensor::is_dma_available;
use edgefirst_tensor::{Tensor, TensorMapTrait as _, TensorMemory, TensorTrait as _};
use num_traits::Num;

use edgefirst_bench::{run_bench, BenchSuite};

const WARMUP: usize = 10;
const ITERATIONS: usize = 200;

/// Size configs: [batch, height, width, channels]
const SIZES: &[([usize; 4], &str)] = &[
    ([1, 360, 640, 4], "360p"),
    ([1, 720, 1280, 4], "720p"),
    ([1, 1080, 1920, 4], "1080p"),
    ([1, 2160, 3840, 4], "4K"),
];

fn bench_backend<T: Num + Clone + Send + Sync + std::fmt::Debug>(
    suite: &mut BenchSuite,
    mem: TensorMemory,
    backend: &str,
    type_name: &str,
) {
    println!("\n== alloc/{backend}/{type_name} ==\n");
    for (size, res) in SIZES {
        let name = format!("alloc/{backend}/{type_name}/{res}");
        let result = run_bench(&name, WARMUP, ITERATIONS, || {
            let _t = Tensor::<T>::new(size.as_slice(), Some(mem), None)
                .expect("Failed to allocate tensor");
        });
        result.print_summary();
        suite.record(&result);
    }

    println!("\n== map/{backend}/{type_name} ==\n");
    for (size, res) in SIZES {
        let name = format!("map/{backend}/{type_name}/{res}");
        let tensor =
            Tensor::<T>::new(size.as_slice(), Some(mem), None).expect("Failed to allocate tensor");
        let result = run_bench(&name, WARMUP, ITERATIONS, || {
            let _m = tensor.map().expect("Failed to map tensor");
        });
        result.print_summary();
        suite.record(&result);
    }
}

fn bench_memcpy(suite: &mut BenchSuite, mem: TensorMemory, backend: &str) {
    println!("\n== memcpy/{backend} ==\n");
    for (size, res) in SIZES {
        let name = format!("memcpy/{backend}/{res}");
        let tensor =
            Tensor::<u8>::new(size.as_slice(), Some(mem), None).expect("Failed to allocate tensor");
        let nbytes = size.iter().product::<usize>();
        let src_data = vec![0xAAu8; nbytes];

        let result = run_bench(&name, WARMUP, ITERATIONS, || {
            let mut map = tensor.map().expect("Failed to map tensor");
            map.as_mut_slice().copy_from_slice(&src_data);
        });
        result.print_summary_with_throughput(nbytes as u64);
        suite.record(&result);
    }
}

fn main() {
    env_logger::init();

    println!("Tensor Benchmark — edgefirst-bench in-process harness (no fork)");
    println!("  warmup={WARMUP}  iterations={ITERATIONS}");

    let mut suite = BenchSuite::from_args();

    // Mem backend — always available
    bench_backend::<u8>(&mut suite, TensorMemory::Mem, "mem", "u8");
    bench_backend::<i8>(&mut suite, TensorMemory::Mem, "mem", "i8");

    // Shm and Dma backends — Linux only
    #[cfg(target_os = "linux")]
    {
        bench_backend::<u8>(&mut suite, TensorMemory::Shm, "shm", "u8");
        bench_backend::<i8>(&mut suite, TensorMemory::Shm, "shm", "i8");

        if is_dma_available() {
            bench_backend::<u8>(&mut suite, TensorMemory::Dma, "dma", "u8");
            bench_backend::<i8>(&mut suite, TensorMemory::Dma, "dma", "i8");
        } else {
            println!("\n  {:50} [skipped: DMA unavailable]", "alloc+map/dma/*");
        }
    }

    bench_memcpy(&mut suite, TensorMemory::Mem, "mem");
    #[cfg(target_os = "linux")]
    {
        bench_memcpy(&mut suite, TensorMemory::Shm, "shm");
        if is_dma_available() {
            bench_memcpy(&mut suite, TensorMemory::Dma, "dma");
        } else {
            println!("\n  {:50} [skipped: DMA unavailable]", "memcpy/dma/*");
        }
    }

    suite.finish();
    println!("\nDone.");
}
