use criterion::{Criterion, criterion_group, criterion_main};
use edgefirst_hal::{Tensor, TensorMemory, TensorTrait};
use fast_image_resize::{FilterType, PixelType, ResizeAlg, ResizeOptions, Resizer, images::Image};
use glob::glob;
use std::{fs::read, vec};
use zune_jpeg::JpegDecoder;

const TARGET_SIZE: [(usize, usize); 4] = [(640, 360), (1280, 720), (1920, 1080), (3840, 2160)];

fn image_header(c: &mut Criterion) {
    let mut group = c.benchmark_group("image_header");
    for img in glob("testdata/*.jpg").expect("Failed to read glob pattern") {
        match img {
            Ok(path) => {
                group.bench_function(path.file_name().unwrap().to_str().unwrap(), |b| {
                    b.iter(|| {
                        let file = read(&path).unwrap();
                        let mut decoder = JpegDecoder::new(&file);
                        decoder.decode_headers().unwrap();
                        let _image_info = decoder.info().unwrap();
                    });
                });
            }
            Err(e) => eprintln!("Error reading file: {}", e),
        }
    }
    group.finish();
}

fn image_mem(c: &mut Criterion) {
    let mut group = c.benchmark_group("image_mem_decode");
    for img in glob("testdata/*.jpg").expect("Failed to read glob pattern") {
        match img {
            Ok(path) => {
                group.bench_function(path.file_name().unwrap().to_str().unwrap(), |b| {
                    b.iter(|| {
                        let file = read(&path).unwrap();
                        let mut decoder = JpegDecoder::new(&file);
                        decoder.decode_headers().unwrap();
                        let image_info = decoder.info().unwrap();
                        let mut img =
                            vec![0u8; image_info.width as usize * image_info.height as usize * 3];
                        decoder.decode_into(&mut img).unwrap();
                    });
                });
            }
            Err(e) => eprintln!("Error reading file: {}", e),
        }
    }
    group.finish();

    let mut group = c.benchmark_group("image_mem_resize");
    for img in glob("testdata/*.jpg").expect("Failed to read glob pattern") {
        match img {
            Ok(path) => {
                let file = read(&path).unwrap();
                let mut decoder = JpegDecoder::new(&file);
                decoder.decode_headers().unwrap();
                let image_info = decoder.info().unwrap();
                let mut img = vec![0u8; image_info.width as usize * image_info.height as usize * 3];
                decoder.decode_into(&mut img).unwrap();

                let mut resizer = Resizer::new();
                let options = ResizeOptions::new()
                    .resize_alg(ResizeAlg::Convolution(FilterType::Hamming))
                    .use_alpha(false);

                for (width, height) in TARGET_SIZE {
                    let mut tgt = vec![0u8; width * height * 3];
                    let filename = path.file_name().unwrap().to_str().unwrap();

                    group.bench_function(format!("{}_{}x{}", filename, width, height), |b| {
                        b.iter(|| {
                            let src_view = Image::from_slice_u8(
                                image_info.width as u32,
                                image_info.height as u32,
                                &mut img,
                                PixelType::U8x3,
                            )
                            .unwrap();
                            let mut tgt_view = Image::from_slice_u8(
                                width as u32,
                                height as u32,
                                &mut tgt,
                                PixelType::U8x3,
                            )
                            .unwrap();
                            resizer.resize(&src_view, &mut tgt_view, &options).unwrap();
                        });
                    });
                }
            }
            Err(e) => eprintln!("Error reading file: {}", e),
        }
    }
    group.finish();
}

fn image_dma(c: &mut Criterion) {
    if Tensor::<u8>::new(&[1], Some(TensorMemory::Dma), None).is_err() {
        eprintln!("DMA memory allocation is not supported on this platform.");
        return;
    }

    let mut group = c.benchmark_group("image_dma_decode");
    for img in glob("testdata/*.jpg").expect("Failed to read glob pattern") {
        match img {
            Ok(path) => {
                group.bench_function(path.file_name().unwrap().to_str().unwrap(), |b| {
                    b.iter(|| {
                        let file = read(&path).unwrap();
                        let mut decoder = JpegDecoder::new(&file);
                        decoder.decode_headers().unwrap();
                        let image_info = decoder.info().unwrap();

                        let tensor = Tensor::<u8>::new(
                            &[image_info.height as usize, image_info.width as usize, 3],
                            Some(TensorMemory::Dma),
                            None,
                        )
                        .expect("Failed to allocate DmaTensor");

                        let mut tensor_map = tensor.map().expect("Failed to map DmaTensor");
                        decoder.decode_into(&mut tensor_map).unwrap();
                    });
                });
            }
            Err(e) => eprintln!("Error reading file: {}", e),
        }
    }
    group.finish();

    let mut group = c.benchmark_group("image_dma_resize");
    for img in glob("testdata/*.jpg").expect("Failed to read glob pattern") {
        match img {
            Ok(path) => {
                let file = read(&path).unwrap();
                let mut decoder = JpegDecoder::new(&file);
                decoder.decode_headers().unwrap();
                let image_info = decoder.info().unwrap();

                let src = Tensor::<u8>::new(
                    &[image_info.height as usize, image_info.width as usize, 3],
                    Some(TensorMemory::Dma),
                    None,
                )
                .expect("Failed to allocate DmaTensor");

                {
                    let mut src_map = src.map().expect("Failed to map DmaTensor");
                    decoder.decode_into(&mut src_map).unwrap();
                }

                let mut resizer = Resizer::new();
                let options = ResizeOptions::new()
                    .resize_alg(ResizeAlg::Convolution(FilterType::Hamming))
                    .use_alpha(false);

                for (width, height) in TARGET_SIZE {
                    let tgt = Tensor::<u8>::new(
                        &[height as usize, width as usize, 3],
                        Some(TensorMemory::Dma),
                        None,
                    )
                    .expect("Failed to allocate DmaTensor");
                    let filename = path.file_name().unwrap().to_str().unwrap();

                    group.bench_function(format!("{}_{}x{}", filename, width, height), |b| {
                        b.iter(|| {
                            let mut src_map = src.map().expect("Failed to map DmaTensor");
                            let mut tgt_map = tgt.map().expect("Failed to map DmaTensor");

                            let src_view = Image::from_slice_u8(
                                image_info.width as u32,
                                image_info.height as u32,
                                &mut src_map,
                                PixelType::U8x3,
                            )
                            .unwrap();
                            let mut tgt_view = Image::from_slice_u8(
                                width as u32,
                                height as u32,
                                &mut tgt_map,
                                PixelType::U8x3,
                            )
                            .unwrap();

                            resizer.resize(&src_view, &mut tgt_view, &options).unwrap();
                        });
                    });
                }
            }
            Err(e) => eprintln!("Error reading file: {}", e),
        }
    }
    group.finish();
}

fn image_shm(c: &mut Criterion) {
    let mut group = c.benchmark_group("image_shm_decode");
    for img in glob("testdata/*.jpg").expect("Failed to read glob pattern") {
        match img {
            Ok(path) => {
                group.bench_function(path.file_name().unwrap().to_str().unwrap(), |b| {
                    b.iter(|| {
                        let file = read(&path).unwrap();
                        let mut decoder = JpegDecoder::new(&file);
                        decoder.decode_headers().unwrap();
                        let image_info = decoder.info().unwrap();

                        let tensor = Tensor::<u8>::new(
                            &[image_info.height as usize, image_info.width as usize, 3],
                            Some(TensorMemory::Shm),
                            None,
                        )
                        .expect("Failed to allocate ShmTensor");

                        let mut tensor_map = tensor.map().expect("Failed to map ShmTensor");
                        decoder.decode_into(&mut tensor_map).unwrap();
                    });
                });
            }
            Err(e) => eprintln!("Error reading file: {}", e),
        }
    }
    group.finish();

    let mut group = c.benchmark_group("image_shm_resize");
    for img in glob("testdata/*.jpg").expect("Failed to read glob pattern") {
        match img {
            Ok(path) => {
                let file = read(&path).unwrap();
                let mut decoder = JpegDecoder::new(&file);
                decoder.decode_headers().unwrap();
                let image_info = decoder.info().unwrap();

                let src = Tensor::<u8>::new(
                    &[image_info.height as usize, image_info.width as usize, 3],
                    Some(TensorMemory::Shm),
                    None,
                )
                .expect("Failed to allocate ShmTensor");

                {
                    let mut src_map = src.map().expect("Failed to map ShmTensor");
                    decoder.decode_into(&mut src_map).unwrap();
                }

                let mut resizer = Resizer::new();
                let options = ResizeOptions::new()
                    .resize_alg(ResizeAlg::Convolution(FilterType::Hamming))
                    .use_alpha(false);

                for (width, height) in TARGET_SIZE {
                    let tgt = Tensor::<u8>::new(
                        &[height as usize, width as usize, 3],
                        Some(TensorMemory::Shm),
                        None,
                    )
                    .expect("Failed to allocate ShmTensor");
                    let filename = path.file_name().unwrap().to_str().unwrap();

                    group.bench_function(format!("{}_{}x{}", filename, width, height), |b| {
                        b.iter(|| {
                            let mut src_map = src.map().expect("Failed to map ShmTensor");
                            let mut tgt_map = tgt.map().expect("Failed to map ShmTensor");

                            let src_view = Image::from_slice_u8(
                                image_info.width as u32,
                                image_info.height as u32,
                                &mut src_map,
                                PixelType::U8x3,
                            )
                            .unwrap();
                            let mut tgt_view = Image::from_slice_u8(
                                width as u32,
                                height as u32,
                                &mut tgt_map,
                                PixelType::U8x3,
                            )
                            .unwrap();

                            resizer.resize(&src_view, &mut tgt_view, &options).unwrap();
                        });
                    });
                }
            }
            Err(e) => eprintln!("Error reading file: {}", e),
        }
    }
    group.finish();
}

criterion_group!(benches, image_header, image_mem, image_dma, image_shm);
criterion_main!(benches);
