use edgefirst_tensor::{dequantize_cpu, dequantize_simd};

#[divan::bench(consts = [1000,10000,100000,1000000])]
fn dequantize_cpu_bench<const N: usize>(bencher: divan::Bencher) {
    let input = rand::random_iter().take(N).collect::<Vec<_>>();
    let mut output = vec![0.0; N];
    let zero_point: u8 = 30;
    let scale: f32 = 0.03;
    bencher.bench_local(|| {
        dequantize_cpu(zero_point as i32, scale, &input, &mut output);
    });
}

#[divan::bench(consts = [1000,10000,100000,1000000])]
fn dequantize_simd_bench<const N: usize>(bencher: divan::Bencher) {
    let input = rand::random_iter().take(N).collect::<Vec<_>>();
    let mut output = vec![0.0; N];
    let zero_point: u8 = 30;
    let scale: f32 = 0.03;
    bencher.bench_local(|| {
        dequantize_simd(zero_point as i32, scale, &input, &mut output);
    });
}

fn main() {
    divan::main();
}
