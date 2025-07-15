use image_converter::{ImageConverter, RGBA, TensorImage};
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

#[divan::bench(types = [Jaguar, Person, Zidane], args = [(640, 360), (960, 540), (1280, 720), (1920, 1080)])]
fn image_benchmark<IMAGE>(bencher: divan::Bencher, size: (usize, usize))
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
    let src = TensorImage::load(&file, Some(RGBA), None).unwrap();
    let mut dst = TensorImage::new(width, height, RGBA, None).unwrap();
    let mut converter = ImageConverter::new().unwrap();

    bencher.bench_local(|| converter.convert(&mut dst, &src).unwrap());
}

fn main() {
    divan::main();
}
