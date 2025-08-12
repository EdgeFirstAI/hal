use edgefirst_image::{
    CPUConverter, G2DConverter, GLConverter, ImageConverterTrait as _, RGBA, Rotation, TensorImage,
    YUYV,
};
use edgefirst_tensor::{TensorMapTrait, TensorMemory, TensorTrait};
use std::path::Path;

trait TestImage {
    fn filename() -> &'static str;
}

macro_rules! test_images {
    ($($name:ident),* $(,)?) => {
        $(
            struct $name;

            impl TestImage for $name {
                fn filename() -> &'static str {
                    stringify!($name).to_lowercase().leak()
                }
            }
        )*
    };
}

test_images!(Person, Jaguar, Zidane);

trait TestRotation {
    fn value() -> Rotation;
}

macro_rules! test_rotations{
    ($($name:ident),* $(,)?) => {
        $(
            struct $name;

            impl TestRotation for $name {
                fn value() -> Rotation {
                    let angle = stringify!($name).to_lowercase().strip_prefix("rotate").unwrap().parse::<usize>().unwrap();
                    Rotation::from_degrees_clockwise(angle)
                }
            }
        )*
    };
}

test_rotations!(Rotate0, Rotate90, Rotate180, Rotate270);

#[divan::bench(types = [Jaguar, Person, Zidane])]
fn load_image_mem<IMAGE>(bencher: divan::Bencher)
where
    IMAGE: TestImage,
{
    let name = format!("{}.jpg", IMAGE::filename());
    let path = Path::new("testdata").join(&name);
    let path = match path.exists() {
        true => path,
        false => {
            let path = Path::new("../testdata").join(&name);
            if path.exists() {
                path
            } else {
                Path::new("../../testdata").join(&name)
            }
        }
    };

    assert!(path.exists(), "unable to locate test image at {path:?}");

    bencher.bench_local(|| {
        let file = std::fs::read(&path).unwrap();
        TensorImage::load(&file, Some(RGBA), Some(TensorMemory::Mem)).expect("Failed to load image")
    });
}

#[divan::bench(types = [Jaguar, Person, Zidane])]
fn load_image_shm<IMAGE>(bencher: divan::Bencher)
where
    IMAGE: TestImage,
{
    let name = format!("{}.jpg", IMAGE::filename());
    let path = Path::new("testdata").join(&name);
    let path = match path.exists() {
        true => path,
        false => {
            let path = Path::new("../testdata").join(&name);
            if path.exists() {
                path
            } else {
                Path::new("../../testdata").join(&name)
            }
        }
    };

    assert!(path.exists(), "unable to locate test image at {path:?}");

    bencher.bench_local(|| {
        let file = std::fs::read(&path).unwrap();
        TensorImage::load(&file, Some(RGBA), Some(TensorMemory::Shm)).expect("Failed to load image")
    });
}

#[cfg(target_os = "linux")]
#[divan::bench(types = [Jaguar, Person, Zidane], ignore = !dma_available())]
fn load_image_dma<IMAGE>(bencher: divan::Bencher)
where
    IMAGE: TestImage,
{
    let name = format!("{}.jpg", IMAGE::filename());
    let path = Path::new("testdata").join(&name);
    let path = match path.exists() {
        true => path,
        false => {
            let path = Path::new("../testdata").join(&name);
            if path.exists() {
                path
            } else {
                Path::new("../../testdata").join(&name)
            }
        }
    };

    assert!(path.exists(), "unable to locate test image at {path:?}");

    bencher.bench_local(|| {
        let file = std::fs::read(&path).unwrap();
        TensorImage::load(&file, Some(RGBA), Some(TensorMemory::Dma)).expect("Failed to load image")
    });
}

#[divan::bench(types = [Jaguar, Person, Zidane], args = [(640, 360), (960, 540), (1280, 720), (1920, 1080)])]
fn resize_cpu<IMAGE>(bencher: divan::Bencher, size: (usize, usize))
where
    IMAGE: TestImage,
{
    use edgefirst_image::CPUConverter;

    let (width, height) = size;
    let name = format!("{}.jpg", IMAGE::filename());
    let path = Path::new("testdata").join(&name);
    let path = match path.exists() {
        true => path,
        false => {
            let path = Path::new("../testdata").join(&name);
            if path.exists() {
                path
            } else {
                Path::new("../../testdata").join(&name)
            }
        }
    };

    assert!(path.exists(), "unable to locate test image at {path:?}");

    let file = std::fs::read(path).unwrap();
    let src = TensorImage::load(&file, Some(RGBA), None).unwrap();
    let mut dst = TensorImage::new(width, height, RGBA, None).unwrap();

    let mut converter = CPUConverter::new().unwrap();

    bencher.bench_local(|| {
        converter
            .convert(&mut dst, &src, Rotation::None, None)
            .unwrap()
    });
}

#[cfg(target_os = "linux")]
#[divan::bench(types = [Jaguar, Person, Zidane], args = [(640, 360), (960, 540), (1280, 720), (1920, 1080)], ignore = !dma_available())]
fn resize_g2d<IMAGE>(bencher: divan::Bencher, size: (usize, usize))
where
    IMAGE: TestImage,
{
    use edgefirst_image::G2DConverter;

    let (width, height) = size;
    let name = format!("{}.jpg", IMAGE::filename());
    let path = Path::new("testdata").join(&name);
    let path = match path.exists() {
        true => path,
        false => {
            let path = Path::new("../testdata").join(&name);
            if path.exists() {
                path
            } else {
                Path::new("../../testdata").join(&name)
            }
        }
    };

    assert!(path.exists(), "unable to locate test image at {path:?}");

    let file = std::fs::read(path).unwrap();
    let src = TensorImage::load(&file, Some(RGBA), None).unwrap();
    let mut dst = TensorImage::new(width, height, RGBA, None).unwrap();

    let mut converter = G2DConverter::new().unwrap();

    bencher.bench_local(|| {
        converter
            .convert(&mut dst, &src, Rotation::None, None)
            .unwrap()
    });
}

#[cfg(target_os = "linux")]
#[divan::bench(types = [Person, Zidane], args = [(640, 360), (960, 540), (1280, 720), (1920, 1080)], ignore = !dma_available())]
fn resize_opengl<IMAGE>(bencher: divan::Bencher, size: (usize, usize))
where
    IMAGE: TestImage,
{
    let (width, height) = size;
    let name = format!("{}.jpg", IMAGE::filename());
    let path = Path::new("testdata").join(&name);
    let path = match path.exists() {
        true => path,
        false => {
            let path = Path::new("../testdata").join(&name);
            if path.exists() {
                path
            } else {
                Path::new("../../testdata").join(&name)
            }
        }
    };

    assert!(path.exists(), "unable to locate test image at {path:?}");

    let file = std::fs::read(path).unwrap();

    let src = TensorImage::load(&file, Some(RGBA), Some(TensorMemory::Dma)).unwrap();
    let mut gl_dst = TensorImage::new(width, height, RGBA, Some(TensorMemory::Dma)).unwrap();
    let mut gl_converter =
        edgefirst_image::GLConverter::new_with_size(width, height, false).unwrap();

    bencher.bench_local(|| {
        gl_converter
            .convert(&mut gl_dst, &src, Rotation::None, None)
            .unwrap()
    });
    drop(gl_dst);
}

#[divan::bench(types = [Rotate0, Rotate90, Rotate180, Rotate270], args = [(640, 360), (960, 540), (1280, 720), (1920, 1080)])]
fn rotate_cpu<R: TestRotation>(bencher: divan::Bencher, params: (usize, usize)) {
    let (mut width, mut height) = params;
    let rot = R::value();

    match rot {
        Rotation::Rotate90Clockwise | Rotation::Rotate90CounterClockwise => {
            (width, height) = (height, width)
        }
        _ => {}
    }
    let file = include_bytes!("../../../testdata/zidane.jpg").to_vec();
    let src = TensorImage::load(&file, Some(RGBA), None).unwrap();
    let mut dst = TensorImage::new(width, height, RGBA, None).unwrap();

    let mut converter = CPUConverter::new().unwrap();

    bencher.bench_local(|| converter.convert(&mut dst, &src, rot, None).unwrap());
}

#[divan::bench(types = [Rotate90, Rotate180, Rotate270], args = [(640, 360), (960, 540), (1280, 720), (1920, 1080)], ignore = !dma_available())]
fn rotate_opengl<R: TestRotation>(bencher: divan::Bencher, params: (usize, usize)) {
    let (mut width, mut height) = params;
    let rot = R::value();

    match rot {
        Rotation::Rotate90Clockwise | Rotation::Rotate90CounterClockwise => {
            (width, height) = (height, width)
        }
        _ => {}
    }
    let file = include_bytes!("../../../testdata/zidane.jpg").to_vec();

    let src = TensorImage::load(&file, Some(RGBA), Some(TensorMemory::Dma)).unwrap();
    let mut dst = TensorImage::new(width, height, RGBA, Some(TensorMemory::Dma)).unwrap();

    let mut converter = GLConverter::new_with_size(width, height, false).unwrap();

    bencher.bench_local(|| converter.convert(&mut dst, &src, rot, None).unwrap());
    drop(dst);
}

#[divan::bench(types = [Rotate90, Rotate180, Rotate270], args = [(640, 360), (960, 540), (1280, 720), (1920, 1080)], ignore = !dma_available())]
fn rotate_g2d<R: TestRotation>(bencher: divan::Bencher, params: (usize, usize)) {
    let (mut width, mut height) = params;
    let rot = R::value();

    match rot {
        Rotation::Rotate90Clockwise | Rotation::Rotate90CounterClockwise => {
            (width, height) = (height, width)
        }
        _ => {}
    }
    let file = include_bytes!("../../../testdata/zidane.jpg").to_vec();

    let src = TensorImage::load(&file, Some(RGBA), Some(TensorMemory::Dma)).unwrap();
    let mut dst = TensorImage::new(width, height, RGBA, Some(TensorMemory::Dma)).unwrap();

    let mut converter = G2DConverter::new().unwrap();

    bencher.bench_local(|| converter.convert(&mut dst, &src, rot, None).unwrap());
}

#[divan::bench(args = [(640, 360), (960, 540), (1280, 720), (1920, 1080)])]
fn convert_cpu(bencher: divan::Bencher, params: (usize, usize)) {
    let file = include_bytes!("../../../testdata/camera720p.yuyv").to_vec();
    let src = TensorImage::new(1280, 720, YUYV, None).unwrap();
    src.tensor()
        .map()
        .unwrap()
        .as_mut_slice()
        .copy_from_slice(&file);

    let (width, height) = params;
    let mut dst = TensorImage::new(width, height, RGBA, None).unwrap();

    let mut converter = CPUConverter::new().unwrap();

    bencher.bench_local(|| {
        converter
            .convert(&mut dst, &src, Rotation::None, None)
            .unwrap()
    });
}

#[divan::bench(args = [(640, 360), (960, 540), (1280, 720), (1920, 1080)], ignore = !dma_available())]
fn convert_g2d(bencher: divan::Bencher, params: (usize, usize)) {
    let file = include_bytes!("../../../testdata/camera720p.yuyv").to_vec();
    let src = TensorImage::new(1280, 720, YUYV, None).unwrap();
    src.tensor()
        .map()
        .unwrap()
        .as_mut_slice()
        .copy_from_slice(&file);

    let (width, height) = params;
    let mut dst = TensorImage::new(width, height, RGBA, Some(TensorMemory::Dma)).unwrap();

    let mut converter = G2DConverter::new().unwrap();

    bencher.bench_local(|| {
        converter
            .convert(&mut dst, &src, Rotation::None, None)
            .unwrap()
    });
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
