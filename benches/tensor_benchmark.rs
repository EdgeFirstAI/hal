use criterion::{Criterion, criterion_group, criterion_main};
use edgefirst_hal::{Tensor, TensorMemory, TensorTrait};

const SHAPES: [(usize, usize, usize, usize); 4] = [
    (1, 360, 640, 4),
    (1, 720, 1280, 4),
    (1, 1080, 1920, 4),
    (1, 2160, 3840, 4),
];

fn tensor_mem_alloc(c: &mut Criterion) {
    let mut group = c.benchmark_group("tensor_mem_alloc");
    for (batch, height, width, channels) in SHAPES {
        group.bench_function(format!("{}x{}", width, height), |b| {
            b.iter(|| {
                let _tensor = vec![0u8; batch * height * width * channels];
            });
        });
    }
    group.finish();

    let mut group = c.benchmark_group("tensor_mem_fill");
    for (batch, height, width, channels) in SHAPES {
        group.bench_function(format!("{}x{}", width, height), |b| {
            b.iter(|| {
                let mut tensor = vec![0u8; batch * height * width * channels];
                tensor.fill(1);
            });
        });
    }
    group.finish();
}

fn tensor_dma_alloc(c: &mut Criterion) {
    if Tensor::<u8>::new(&[1], Some(TensorMemory::Dma), None).is_err() {
        eprintln!("DMA memory allocation is not supported on this platform.");
        return;
    }

    let mut group = c.benchmark_group("tensor_dma_alloc");
    for (batch, height, width, channels) in SHAPES {
        group.bench_function(format!("{}x{}", width, height), |b| {
            b.iter(|| {
                let _tensor = Tensor::<u8>::new(
                    &[batch, height, width, channels],
                    Some(TensorMemory::Dma),
                    None,
                )
                .expect("Failed to allocate DMA tensor");
            });
        });
    }
    group.finish();

    let mut group = c.benchmark_group("tensor_dma_fill");
    for (batch, height, width, channels) in SHAPES {
        group.bench_function(format!("{}x{}", width, height), |b| {
            b.iter(|| {
                let tensor = Tensor::<u8>::new(
                    &[batch, height, width, channels],
                    Some(TensorMemory::Dma),
                    None,
                )
                .expect("Failed to allocate DMA tensor");

                let mut tensor_map = tensor.map().expect("Failed to map DMA tensor");
                tensor_map.fill(1);
            });
        });
    }
    group.finish();
}

fn tensor_shm_alloc(c: &mut Criterion) {
    let mut group = c.benchmark_group("tensor_shm_alloc");
    for (batch, height, width, channels) in SHAPES {
        group.bench_function(format!("{}x{}", width, height), |b| {
            b.iter(|| {
                let _tensor = Tensor::<u8>::new(
                    &[batch, height, width, channels],
                    Some(TensorMemory::Shm),
                    None,
                )
                .expect("Failed to allocate SHM tensor");
            });
        });
    }
    group.finish();

    let mut group = c.benchmark_group("tensor_shm_fill");
    for (batch, height, width, channels) in SHAPES {
        group.bench_function(format!("{}x{}", width, height), |b| {
            b.iter(|| {
                let tensor = Tensor::<u8>::new(
                    &[batch, height, width, channels],
                    Some(TensorMemory::Shm),
                    None,
                )
                .expect("Failed to allocate SHM tensor");

                let mut tensor_map = tensor.map().expect("Failed to map SHM tensor");
                tensor_map.fill(1);
            });
        });
    }
    group.finish();
}

criterion_group!(
    benches,
    tensor_mem_alloc,
    tensor_dma_alloc,
    tensor_shm_alloc
);
criterion_main!(benches);
