use edgefirst_image::{CPUConverter, ImageConverterTrait as _, RGBA, Rotation, TensorImage};
use edgefirst_tensor::{TensorMemory, TensorTrait};
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
fn converter_cpu<IMAGE>(bencher: divan::Bencher, size: (usize, usize))
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
fn converter_g2d<IMAGE>(bencher: divan::Bencher, size: (usize, usize))
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
fn converter_opengl<IMAGE>(bencher: divan::Bencher, size: (usize, usize))
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
