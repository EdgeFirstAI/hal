use criterion::{Criterion, criterion_group, criterion_main};
use glob::glob;

fn image_decode(c: &mut Criterion) {
    let mut group = c.benchmark_group("image_decode");
    for img in glob("testdata/*.jpg").expect("Failed to read glob pattern") {
        match img {
            Ok(path) => {
                group.bench_function(path.file_name().unwrap().to_str().unwrap(), |b| {
                    b.iter(|| {
                        let img = image::open(&path).expect("Failed to open image");
                        let _ = img.to_rgb8(); // Decode to RGB
                    });
                });
            }
            Err(e) => eprintln!("Error reading file: {}", e),
        }
    }
    group.finish();
}

criterion_group!(benches, image_decode);
criterion_main!(benches);
