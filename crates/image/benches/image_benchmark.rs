#[cfg(target_os = "linux")]
use edgefirst_image::G2DConverter;
#[cfg(feature = "opengl")]
#[cfg(target_os = "linux")]
use edgefirst_image::GLConverter;
use edgefirst_image::{
    CPUConverter, Flip, GREY, ImageConverterTrait as _, RGB, RGBA, Rotation, TensorImage, YUYV,
};
use edgefirst_tensor::{TensorMapTrait, TensorMemory, TensorTrait};
use four_char_code::FourCharCode;
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

trait FourCC {
    fn val() -> FourCharCode;
}

struct RgbaType;
impl FourCC for RgbaType {
    fn val() -> FourCharCode {
        RGBA
    }
}

struct RgbType;
impl FourCC for RgbType {
    fn val() -> FourCharCode {
        RGB
    }
}

struct GreyType;
impl FourCC for GreyType {
    fn val() -> FourCharCode {
        GREY
    }
}

// struct YuyvType;
// impl FourCC for YuyvType {
//     fn val() -> FourCharCode {
//         YUYV
//     }
// }

// struct Nv12Type;
// impl FourCC for Nv12Type {
//     fn val() -> FourCharCode {
//         NV12
//     }
// }
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
        TensorImage::load_jpeg(&file, Some(RGBA), Some(TensorMemory::Mem))
            .expect("Failed to load image")
    });
}

#[cfg(target_os = "linux")]
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
        TensorImage::load_jpeg(&file, Some(RGBA), Some(TensorMemory::Shm))
            .expect("Failed to load image")
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
        TensorImage::load_jpeg(&file, Some(RGBA), Some(TensorMemory::Dma))
            .expect("Failed to load image")
    });
}

#[divan::bench(types = [RgbaType, RgbType, GreyType])]
fn load_image_types<T>(bencher: divan::Bencher)
where
    T: FourCC,
{
    let file = include_bytes!("../../../testdata/person.jpg");
    bencher.bench_local(|| {
        let im = TensorImage::load_jpeg(file, Some(T::val()), Some(TensorMemory::Mem))
            .expect("Failed to load image");

        std::hint::black_box(im);
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
    let src = TensorImage::load_jpeg(&file, Some(RGBA), None).unwrap();
    let mut dst = TensorImage::new(width, height, RGBA, None).unwrap();

    let mut converter = CPUConverter::new().unwrap();

    bencher.bench_local(|| {
        converter
            .convert(&src, &mut dst, Rotation::None, Flip::None, None)
            .unwrap()
    });
}

#[divan::bench(types = [Jaguar, Person, Zidane], args = [(640, 360), (960, 540), (1280, 720), (1920, 1080)])]
fn resize_cpu_grayscale_upscale<IMAGE>(bencher: divan::Bencher, size: (usize, usize))
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
    let src = TensorImage::load_jpeg(&file, Some(RGBA), None).unwrap();
    let mut grey = TensorImage::new(160, 160, GREY, None).unwrap();
    let mut dst = TensorImage::new(width, height, GREY, None).unwrap();

    let mut converter = CPUConverter::new().unwrap();

    converter
        .convert(&src, &mut grey, Rotation::None, Flip::None, None)
        .unwrap();

    bencher.bench_local(|| {
        converter
            .convert(&grey, &mut dst, Rotation::None, Flip::None, None)
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
    let src = TensorImage::load_jpeg(&file, Some(RGBA), None).unwrap();
    let mut dst = TensorImage::new(width, height, RGBA, None).unwrap();

    let mut converter = G2DConverter::new().unwrap();

    bencher.bench_local(|| {
        converter
            .convert(&src, &mut dst, Rotation::None, Flip::None, None)
            .unwrap()
    });
}

#[cfg(target_os = "linux")]
#[cfg(feature = "opengl")]
#[divan::bench(types = [Person, Zidane], args = [(640, 360), (960, 540), (1280, 720), (1920, 1080)])]
fn resize_opengl_mem<IMAGE>(bencher: divan::Bencher, size: (usize, usize))
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

    let src = TensorImage::load_jpeg(&file, Some(RGBA), Some(TensorMemory::Mem)).unwrap();
    let mut gl_dst = TensorImage::new(width, height, RGBA, Some(TensorMemory::Dma)).unwrap();
    let mut gl_converter = edgefirst_image::GLConverter::new().unwrap();

    bencher.bench_local(|| {
        gl_converter
            .convert(&src, &mut gl_dst, Rotation::None, Flip::None, None)
            .unwrap()
    });
    drop(gl_dst);
}

#[cfg(target_os = "linux")]
#[cfg(feature = "opengl")]
#[divan::bench(types = [Person, Zidane], args = [(640, 360), (960, 540), (1280, 720), (1920, 1080)], ignore = !dma_available())]
fn resize_opengl_dma<IMAGE>(bencher: divan::Bencher, size: (usize, usize))
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

    let src = TensorImage::load_jpeg(&file, Some(RGBA), Some(TensorMemory::Dma)).unwrap();
    let mut gl_dst = TensorImage::new(width, height, RGBA, Some(TensorMemory::Dma)).unwrap();
    let mut gl_converter = edgefirst_image::GLConverter::new().unwrap();

    bencher.bench_local(|| {
        gl_converter
            .convert(&src, &mut gl_dst, Rotation::None, Flip::None, None)
            .unwrap()
    });
    drop(gl_dst);
}

#[divan::bench(types = [Rotate0, Rotate90, Rotate180, Rotate270], args = [(640, 360), (960, 540), (1280, 720), (1920, 1080)])]
fn rotate_cpu<R: TestRotation>(bencher: divan::Bencher, params: (usize, usize)) {
    let (mut width, mut height) = params;
    let rot = R::value();

    match rot {
        Rotation::Clockwise90 | Rotation::CounterClockwise90 => (width, height) = (height, width),
        _ => {}
    }
    let file = include_bytes!("../../../testdata/zidane.jpg").to_vec();
    let src = TensorImage::load_jpeg(&file, Some(RGBA), None).unwrap();
    let mut dst = TensorImage::new(width, height, RGBA, None).unwrap();

    let mut converter = CPUConverter::new().unwrap();

    bencher.bench_local(|| {
        converter
            .convert(&src, &mut dst, rot, Flip::None, None)
            .unwrap()
    });
}

#[cfg(target_os = "linux")]
#[cfg(feature = "opengl")]
#[divan::bench(types = [Rotate90, Rotate180, Rotate270], args = [(640, 360), (960, 540), (1280, 720), (1920, 1080)], ignore = !dma_available())]
fn rotate_opengl<R: TestRotation>(bencher: divan::Bencher, params: (usize, usize)) {
    let (mut width, mut height) = params;
    let rot = R::value();

    match rot {
        Rotation::Clockwise90 | Rotation::CounterClockwise90 => (width, height) = (height, width),
        _ => {}
    }
    let file = include_bytes!("../../../testdata/zidane.jpg").to_vec();

    let src = TensorImage::load_jpeg(&file, Some(RGBA), Some(TensorMemory::Dma)).unwrap();
    let mut dst = TensorImage::new(width, height, RGBA, Some(TensorMemory::Dma)).unwrap();

    let mut converter = GLConverter::new().unwrap();

    bencher.bench_local(|| {
        converter
            .convert(&src, &mut dst, rot, Flip::None, None)
            .unwrap()
    });
    drop(dst);
}

#[cfg(target_os = "linux")]
#[divan::bench(types = [Rotate90, Rotate180, Rotate270], args = [(640, 360), (960, 540), (1280, 720), (1920, 1080)], ignore = !dma_available())]
fn rotate_g2d<R: TestRotation>(bencher: divan::Bencher, params: (usize, usize)) {
    let (mut width, mut height) = params;
    let rot = R::value();

    match rot {
        Rotation::Clockwise90 | Rotation::CounterClockwise90 => (width, height) = (height, width),
        _ => {}
    }
    let file = include_bytes!("../../../testdata/zidane.jpg").to_vec();

    let src = TensorImage::load_jpeg(&file, Some(RGBA), Some(TensorMemory::Dma)).unwrap();
    let mut dst = TensorImage::new(width, height, RGBA, Some(TensorMemory::Dma)).unwrap();

    let mut converter = G2DConverter::new().unwrap();

    bencher.bench_local(|| {
        converter
            .convert(&src, &mut dst, rot, Flip::None, None)
            .unwrap()
    });
}

#[divan::bench(args = [(640, 360), (960, 540), (1280, 720), (1920, 1080)])]
fn convert_cpu_yuyv_to_rgba(bencher: divan::Bencher, params: (usize, usize)) {
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
            .convert(&src, &mut dst, Rotation::None, Flip::None, None)
            .unwrap()
    });
}

#[divan::bench(args = [(640, 360), (960, 540), (1280, 720), (1920, 1080)])]
fn convert_cpu_yuyv_to_rgb(bencher: divan::Bencher, params: (usize, usize)) {
    let file = include_bytes!("../../../testdata/camera720p.yuyv").to_vec();
    let src = TensorImage::new(1280, 720, YUYV, None).unwrap();
    src.tensor()
        .map()
        .unwrap()
        .as_mut_slice()
        .copy_from_slice(&file);

    let (width, height) = params;
    let mut dst = TensorImage::new(width, height, RGB, None).unwrap();

    let mut converter = CPUConverter::new().unwrap();

    bencher.bench_local(|| {
        converter
            .convert(&src, &mut dst, Rotation::None, Flip::None, None)
            .unwrap()
    });
}

#[divan::bench(args = [(640, 360), (960, 540), (1280, 720), (1920, 1080)])]
fn convert_cpu_yuyv_to_yuyv(bencher: divan::Bencher, params: (usize, usize)) {
    let file = include_bytes!("../../../testdata/camera720p.yuyv").to_vec();
    let src = TensorImage::new(1280, 720, YUYV, None).unwrap();
    src.tensor()
        .map()
        .unwrap()
        .as_mut_slice()
        .copy_from_slice(&file);

    let (width, height) = params;
    let mut dst = TensorImage::new(width, height, YUYV, None).unwrap();

    let mut converter = CPUConverter::new().unwrap();

    bencher.bench_local(|| {
        converter
            .convert(&src, &mut dst, Rotation::None, Flip::None, None)
            .unwrap()
    });
}

#[divan::bench(args = [(640, 360), (960, 540), (1280, 720), (1920, 1080)])]
fn convert_cpu_rgba_to_yuyv(bencher: divan::Bencher, params: (usize, usize)) {
    let file = include_bytes!("../../../testdata/camera720p.rgba").to_vec();
    let src = TensorImage::new(1280, 720, RGBA, None).unwrap();
    src.tensor()
        .map()
        .unwrap()
        .as_mut_slice()
        .copy_from_slice(&file);

    let (width, height) = params;
    let mut dst = TensorImage::new(width, height, YUYV, None).unwrap();

    let mut converter = CPUConverter::new().unwrap();

    bencher.bench_local(|| {
        converter
            .convert(&src, &mut dst, Rotation::None, Flip::None, None)
            .unwrap()
    });
}

#[divan::bench(types = [Jaguar, Person, Zidane])]
fn convert_cpu_rgba_to_rgb<IMAGE>(bencher: divan::Bencher)
where
    IMAGE: TestImage,
{
    let name = format!("{}.jpg", IMAGE::filename());
    let path = if Path::new("./testdata").join(&name).exists() {
        Path::new("./testdata").join(&name)
    } else if Path::new("../testdata").join(&name).exists() {
        Path::new("../testdata").join(&name)
    } else if Path::new("../../testdata").join(&name).exists() {
        Path::new("../../testdata").join(&name)
    } else {
        Path::new("../../../testdata").join(&name)
    };
    assert!(path.exists(), "unable to locate test image at {path:?}");

    let file = std::fs::read(path).unwrap();

    let src = TensorImage::load_jpeg(&file, Some(RGBA), None).unwrap();
    let mut dst = TensorImage::new(src.width(), src.height(), RGB, None).unwrap();

    let mut converter = CPUConverter::new().unwrap();

    bencher.bench_local(|| {
        converter
            .convert(&src, &mut dst, Rotation::None, Flip::None, None)
            .unwrap()
    });
}

#[cfg(target_os = "linux")]
#[divan::bench(args = [(640, 360), (960, 540), (1280, 720), (1920, 1080)], ignore = !dma_available())]
fn convert_g2d_yuyv_to_rgba(bencher: divan::Bencher, params: (usize, usize)) {
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
            .convert(&src, &mut dst, Rotation::None, Flip::None, None)
            .unwrap()
    });
}

#[cfg(target_os = "linux")]
#[divan::bench(args = [(640, 360), (960, 540), (1280, 720), (1920, 1080)], ignore = !dma_available())]
fn convert_g2d_yuyv_to_rgb(bencher: divan::Bencher, params: (usize, usize)) {
    let file = include_bytes!("../../../testdata/camera720p.yuyv").to_vec();
    let src = TensorImage::new(1280, 720, YUYV, None).unwrap();
    src.tensor()
        .map()
        .unwrap()
        .as_mut_slice()
        .copy_from_slice(&file);

    let (width, height) = params;
    let mut dst = TensorImage::new(width, height, RGB, Some(TensorMemory::Dma)).unwrap();

    let mut converter = G2DConverter::new().unwrap();

    bencher.bench_local(|| {
        converter
            .convert(&src, &mut dst, Rotation::None, Flip::None, None)
            .unwrap()
    });
}

#[cfg(target_os = "linux")]
#[divan::bench(args = [(640, 360), (960, 540), (1280, 720), (1920, 1080)], ignore = !dma_available())]
fn convert_g2d_yuyv_to_yuyv(bencher: divan::Bencher, params: (usize, usize)) {
    let file = include_bytes!("../../../testdata/camera720p.yuyv").to_vec();
    let src = TensorImage::new(1280, 720, YUYV, None).unwrap();
    src.tensor()
        .map()
        .unwrap()
        .as_mut_slice()
        .copy_from_slice(&file);

    let (width, height) = params;
    let mut dst = TensorImage::new(width, height, YUYV, Some(TensorMemory::Dma)).unwrap();

    let mut converter = G2DConverter::new().unwrap();

    bencher.bench_local(|| {
        converter
            .convert(&src, &mut dst, Rotation::None, Flip::None, None)
            .unwrap()
    });
}

#[cfg(target_os = "linux")]
#[cfg(feature = "opengl")]
#[divan::bench(args = [(640, 360), (960, 540), (1280, 720), (1920, 1080)], ignore = !dma_available())]
fn convert_opengl_yuyv_to_rgba(bencher: divan::Bencher, params: (usize, usize)) {
    let file = include_bytes!("../../../testdata/camera720p.yuyv").to_vec();
    let src = TensorImage::new(1280, 720, YUYV, Some(TensorMemory::Dma)).unwrap();
    src.tensor()
        .map()
        .unwrap()
        .as_mut_slice()
        .copy_from_slice(&file);

    let (width, height) = params;
    let mut dst = TensorImage::new(width, height, RGBA, None).unwrap();

    let mut converter = GLConverter::new().unwrap();

    bencher.bench_local(|| {
        converter
            .convert(&src, &mut dst, Rotation::None, Flip::None, None)
            .unwrap()
    });
    drop(dst);
}

#[cfg(target_os = "linux")]
#[cfg(feature = "opengl")]
#[divan::bench(args = [(640, 360), (960, 540), (1280, 720), (1920, 1080)], ignore = !dma_available())]
fn convert_opengl_yuyv_to_yuyv(bencher: divan::Bencher, params: (usize, usize)) {
    let file = include_bytes!("../../../testdata/camera720p.yuyv").to_vec();
    let src = TensorImage::new(1280, 720, YUYV, Some(TensorMemory::Dma)).unwrap();
    src.tensor()
        .map()
        .unwrap()
        .as_mut_slice()
        .copy_from_slice(&file);

    let (width, height) = params;
    let mut dst = TensorImage::new(width, height, YUYV, None).unwrap();

    let mut converter = GLConverter::new().unwrap();

    bencher.bench_local(|| {
        converter
            .convert(&src, &mut dst, Rotation::None, Flip::None, None)
            .unwrap()
    });
    drop(dst);
}

#[cfg(target_os = "linux")]
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
