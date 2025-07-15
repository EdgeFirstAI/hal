use num_traits::Num;
use tensor::{Tensor, TensorMemory, TensorTrait as _};

const MEMORY: &[char] = &[
    #[cfg(target_os = "linux")]
    'D',
    'S',
    'M',
    'N',
];

#[divan::bench(types = [u8, f64], consts = MEMORY, args = [[1, 360, 640, 4], [1, 720, 1280, 4], [1, 1080, 1920, 4], [1, 2160, 3840, 4]])]
fn tensor_alloc<T, const M: char>(size: [usize; 4])
where
    T: Num + Clone + Send + Sync + std::fmt::Debug,
{
    let _tensor = Tensor::<T>::new(&size, memory::<M>(), None).expect("Failed to allocate tensor");
}

#[divan::bench(types = [u8, f64], consts = MEMORY, args = [[1, 360, 640, 4], [1, 720, 1280, 4], [1, 1080, 1920, 4], [1, 2160, 3840, 4]])]
fn tensor_map<T, const M: char>(size: [usize; 4])
where
    T: Num + Clone + Send + Sync + std::fmt::Debug,
{
    let tensor = Tensor::<T>::new(&size, memory::<M>(), None).expect("Failed to allocate tensor");
    let _map = tensor.map().expect("Failed to map tensor");
}

fn memory<const M: char>() -> Option<TensorMemory> {
    match M {
        'D' => Some(TensorMemory::Dma),
        'S' => Some(TensorMemory::Shm),
        'M' => Some(TensorMemory::Mem),
        'N' => None,
        _ => None,
    }
}

fn main() {
    divan::main();
}
