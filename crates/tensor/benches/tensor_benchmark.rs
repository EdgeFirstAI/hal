// SPDX-FileCopyrightText: Copyright 2025 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

use edgefirst_tensor::{Tensor, TensorMemory, TensorTrait as _};
use num_traits::Num;

#[divan::bench(types = [u8, f64], args = [[1, 360, 640, 4], [1, 720, 1280, 4], [1, 1080, 1920, 4], [1, 2160, 3840, 4]])]
fn tensor_alloc_mem<T>(size: [usize; 4])
where
    T: Num + Clone + Send + Sync + std::fmt::Debug,
{
    let _tensor =
        Tensor::<T>::new(&size, Some(TensorMemory::Mem), None).expect("Failed to allocate tensor");
}

#[divan::bench(types = [u8, f64], args = [[1, 360, 640, 4], [1, 720, 1280, 4], [1, 1080, 1920, 4], [1, 2160, 3840, 4]])]
fn tensor_map_mem<T>(bench: divan::Bencher, size: [usize; 4])
where
    T: Num + Clone + Send + Sync + std::fmt::Debug,
{
    let tensor =
        Tensor::<T>::new(&size, Some(TensorMemory::Mem), None).expect("Failed to allocate tensor");
    bench.bench_local(|| tensor.map().expect("Failed to map tensor"));
}

#[cfg(target_os = "linux")]
#[divan::bench(types = [u8, f64], args = [[1, 360, 640, 4], [1, 720, 1280, 4], [1, 1080, 1920, 4], [1, 2160, 3840, 4]])]
fn tensor_alloc_shm<T>(size: [usize; 4])
where
    T: Num + Clone + Send + Sync + std::fmt::Debug,
{
    let _tensor =
        Tensor::<T>::new(&size, Some(TensorMemory::Shm), None).expect("Failed to allocate tensor");
}

#[cfg(target_os = "linux")]
#[divan::bench(types = [u8, f64], args = [[1, 360, 640, 4], [1, 720, 1280, 4], [1, 1080, 1920, 4], [1, 2160, 3840, 4]])]
fn tensor_map_shm<T>(bench: divan::Bencher, size: [usize; 4])
where
    T: Num + Clone + Send + Sync + std::fmt::Debug,
{
    let tensor =
        Tensor::<T>::new(&size, Some(TensorMemory::Shm), None).expect("Failed to allocate tensor");
    bench.bench_local(|| tensor.map().expect("Failed to map tensor"));
}

#[cfg(target_os = "linux")]
#[divan::bench(types = [u8, f64], args = [[1, 360, 640, 4], [1, 720, 1280, 4], [1, 1080, 1920, 4], [1, 2160, 3840, 4]], ignore = !dma_available())]
fn tensor_alloc_dma<T>(size: [usize; 4])
where
    T: Num + Clone + Send + Sync + std::fmt::Debug,
{
    let _tensor =
        Tensor::<T>::new(&size, Some(TensorMemory::Dma), None).expect("Failed to allocate tensor");
}

#[cfg(target_os = "linux")]
#[divan::bench(types = [u8, f64], args = [[1, 360, 640, 4], [1, 720, 1280, 4], [1, 1080, 1920, 4], [1, 2160, 3840, 4]], ignore = !dma_available())]
fn tensor_map_dma<T>(bench: divan::Bencher, size: [usize; 4])
where
    T: Num + Clone + Send + Sync + std::fmt::Debug,
{
    let tensor =
        Tensor::<T>::new(&size, Some(TensorMemory::Dma), None).expect("Failed to allocate tensor");
    bench.bench_local(|| tensor.map().expect("Failed to map tensor"));
}

fn dma_available() -> bool {
    #[cfg(target_os = "linux")]
    {
        dma_heap::Heap::new(dma_heap::HeapKind::Cma)
            .or_else(|_| dma_heap::Heap::new(dma_heap::HeapKind::System))
            .is_ok()
    }
    #[cfg(not(target_os = "linux"))]
    false
}

fn main() {
    env_logger::init();
    divan::main();
}
