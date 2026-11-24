#![cfg(target_os = "linux")]

use edgefirst_image::{G2DProcessor, GLProcessorThreaded, ImageProcessorTrait};
use edgefirst_tensor::{
    CpuAccess, DType, PixelFormat, TensorDyn, TensorMapTrait, TensorMemory, TensorTrait,
};

fn iters() -> usize {
    std::env::var("REPRO_ITERS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(300)
}

fn img(w: usize, h: usize, f: PixelFormat) -> TensorDyn {
    TensorDyn::image(
        w,
        h,
        f,
        DType::U8,
        Some(TensorMemory::DmaBuf),
        CpuAccess::ReadWrite,
    )
    .unwrap()
}

fn nv12(w: usize, h: usize, seed: u8) -> TensorDyn {
    let t = img(w, h, PixelFormat::Nv12);
    {
        let mut m = t.as_u8().unwrap().map().unwrap();
        for (i, b) in m.as_mut_slice().iter_mut().enumerate() {
            *b = (i as u8).wrapping_add(seed);
        }
    }
    t
}

fn rgba(w: usize, h: usize) -> TensorDyn {
    let t = img(w, h, PixelFormat::Rgba);
    {
        let mut m = t.as_u8().unwrap().map().unwrap();
        for (i, b) in m.as_mut_slice().iter_mut().enumerate() {
            *b = (i * 7) as u8;
        }
    }
    t
}

fn nv12_loop(n: usize, tag: &str) {
    let mut gl = GLProcessorThreaded::new(None).unwrap();
    let (w, h) = (128, 96);
    let pool: Vec<_> = (0..3).map(|i| nv12(w, h, i as u8)).collect();
    let mut d1 = img(w, h, PixelFormat::Rgba);
    let mut d2 = img(w * 3 / 2, h * 3 / 2, PixelFormat::Rgba);
    for (i, src) in pool.iter().cycle().take(n).enumerate() {
        gl.convert(src, &mut d1, None).unwrap();
        gl.convert(src, &mut d2, None).unwrap();
        if i % 100 == 0 {
            eprintln!("{tag} {i}");
        }
    }
}

fn rgba_loop(n: usize, tag: &str) {
    let mut gl = GLProcessorThreaded::new(None).unwrap();
    let src = rgba(160, 120);
    let mut d1 = img(224, 224, PixelFormat::Rgb);
    let mut d2 = img(224, 224, PixelFormat::PlanarRgb);
    let mut d3 = img(320, 240, PixelFormat::Rgba);
    for i in 0..n {
        gl.convert(&src, &mut d1, None).unwrap();
        gl.convert(&src, &mut d2, None).unwrap();
        gl.convert(&src, &mut d3, None).unwrap();
        if i % 100 == 0 {
            eprintln!("{tag} {i}");
        }
    }
}

fn churn_loop(n: usize, tag: &str) {
    for i in 0..n / 10 {
        let mut gl = GLProcessorThreaded::new(None).unwrap();
        let src = nv12(64, 48, i as u8);
        let mut d = img(64, 48, PixelFormat::Rgba);
        gl.convert(&src, &mut d, None).unwrap();
        drop(gl);
        if i % 10 == 0 {
            eprintln!("{tag} {i}");
        }
    }
}

fn g2d_loop(n: usize, tag: &str) {
    let mut g = G2DProcessor::new().unwrap();
    let src = nv12(640, 480, 3);
    let mut d = img(320, 240, PixelFormat::Rgba);
    for i in 0..n {
        let _ = g.convert(&src, &mut d, None);
        if i % 100 == 0 {
            eprintln!("{tag} {i}");
        }
    }
}

fn nv12dst_loop(n: usize, tag: &str) {
    let mut gl = GLProcessorThreaded::new(None).unwrap();
    let cases = [
        ((64usize, 32usize), (32usize, 16usize)),
        ((1920, 1080), (960, 540)),
        ((640, 480), (320, 240)),
    ];
    let srcs: Vec<_> = cases.iter().map(|((w, h), _)| rgba(*w, *h)).collect();
    let mut dsts: Vec<_> = cases
        .iter()
        .map(|(_, (w, h))| img(*w, *h, PixelFormat::Nv12))
        .collect();
    for i in 0..n / 10 {
        for (s, d) in srcs.iter().zip(dsts.iter_mut()) {
            gl.convert(s, d, None).unwrap();
        }
        if i % 10 == 0 {
            eprintln!("{tag} {i}");
        }
    }
}

fn grey_loop(n: usize, tag: &str) {
    let mut gl = GLProcessorThreaded::new(None).unwrap();
    let src = rgba(322, 242);
    let mut d1 = img(322, 242, PixelFormat::Grey);
    let mut d2 = img(160, 120, PixelFormat::Grey);
    for i in 0..n / 2 {
        gl.convert(&src, &mut d1, None).unwrap();
        gl.convert(&src, &mut d2, None).unwrap();
        if i % 50 == 0 {
            eprintln!("{tag} {i}");
        }
    }
}

/// NV12 and Grey destination renders on `gl`. Returns the number of
/// conversions GL rendered.
fn r8_targets(gl: &mut GLProcessorThreaded) -> usize {
    use edgefirst_image::ConvertOptions;
    let mut rendered = 0;
    let bars = ConvertOptions::new().letterbox([200, 30, 90, 255]);
    for (w, h) in [(64usize, 32usize), (1920, 1080)] {
        let cases = [
            (rgba(w, h), (w, h), ConvertOptions::new()),
            (rgba(w, h), (w / 2, h / 2), ConvertOptions::new()),
            (rgba(h, h), (w, h), bars),
            (rgba(h - 2, h), (w, h), bars),
        ];
        for (src, (dw, dh), opts) in cases {
            let mut dst = img(dw, dh, PixelFormat::Nv12);
            match gl.convert(&src, &mut dst, Some(opts)) {
                Ok(()) => rendered += 1,
                Err(e) => eprintln!("nv12 {w}x{h} -> {dw}x{dh}: {e}"),
            }
        }
    }
    let src = rgba(322, 242);
    for (dw, dh) in [(322usize, 242usize), (160, 120)] {
        let mut dst = img(dw, dh, PixelFormat::Grey);
        match gl.convert(&src, &mut dst, None) {
            Ok(()) => rendered += 1,
            Err(e) => eprintln!("grey -> {dw}x{dh}: {e}"),
        }
    }
    let nv = nv12(322, 242, 9);
    let mut dst = img(322, 242, PixelFormat::Grey);
    match gl.convert(&nv, &mut dst, None) {
        Ok(()) => rendered += 1,
        Err(e) => eprintln!("nv12 -> grey: {e}"),
    }
    eprintln!("r8 targets rendered: {rendered}");
    rendered
}

/// The dma_pool steady-state loop: three NV12 DMA-BUF sources converted
/// into two RGBA DMA-BUF destinations, `frames` times after a warmup.
fn dma_pool(gl: &mut GLProcessorThreaded, frames: usize) {
    let (w, h) = (128, 96);
    let pool: Vec<_> = (0..3).map(|i| nv12(w, h, i as u8)).collect();
    let mut d1 = img(w, h, PixelFormat::Rgba);
    let mut d2 = img(w * 3 / 2, h * 3 / 2, PixelFormat::Rgba);
    for (i, src) in pool.iter().cycle().take(6 + frames).enumerate() {
        gl.convert(src, &mut d1, None).unwrap();
        gl.convert(src, &mut d2, None).unwrap();
        if i % 100 == 0 {
            eprintln!("dma_pool {i}");
        }
    }
}

#[test]
#[ignore]
fn e1_one_processor() {
    let mut gl = GLProcessorThreaded::new(None).unwrap();
    assert!(r8_targets(&mut gl) > 0);
    dma_pool(&mut gl, 300);
}

#[test]
#[ignore]
fn e2_r8_processor_dropped_first() {
    {
        let mut a = GLProcessorThreaded::new(None).unwrap();
        assert!(r8_targets(&mut a) > 0);
    }
    let mut b = GLProcessorThreaded::new(None).unwrap();
    dma_pool(&mut b, 300);
}

#[test]
#[ignore]
fn e3_r8_processor_alive_idle() {
    let mut a = GLProcessorThreaded::new(None).unwrap();
    assert!(r8_targets(&mut a) > 0);
    let mut b = GLProcessorThreaded::new(None).unwrap();
    dma_pool(&mut b, 300);
    drop(a);
}

#[test]
#[ignore]
fn e3b_dma_pool_resumed_after_r8() {
    let mut b = GLProcessorThreaded::new(None).unwrap();
    dma_pool(&mut b, 30);
    let mut a = GLProcessorThreaded::new(None).unwrap();
    assert!(r8_targets(&mut a) > 0);
    dma_pool(&mut b, 300);
    drop(a);
}

fn yuyv(w: usize, h: usize, seed: u8) -> TensorDyn {
    let t = img(w, h, PixelFormat::Yuyv);
    {
        let mut m = t.as_u8().unwrap().map().unwrap();
        for (i, b) in m.as_mut_slice().iter_mut().enumerate() {
            *b = (i as u8).wrapping_mul(3).wrapping_add(seed);
        }
    }
    t
}

fn mp_nv12(w: usize, h: usize, seed: u8) -> TensorDyn {
    use edgefirst_tensor::Tensor;
    let luma = Tensor::<u8>::new(&[h, w], Some(TensorMemory::DmaBuf), Some("mp_luma")).unwrap();
    let chroma =
        Tensor::<u8>::new(&[h / 2, w], Some(TensorMemory::DmaBuf), Some("mp_chroma")).unwrap();
    for (k, t) in [&luma, &chroma].into_iter().enumerate() {
        let mut m = t.map().unwrap();
        for (i, b) in m.as_mut_slice().iter_mut().enumerate() {
            *b = (i as u8).wrapping_add(seed).wrapping_add(k as u8 * 64);
        }
    }
    let t = Tensor::<u8>::from_planes(luma, chroma, PixelFormat::Nv12).unwrap();
    assert!(t.is_multiplane());
    TensorDyn::from(t)
}

/// A steady-state loop over three sources from `make` into two RGBA
/// DMA-BUF destinations.
fn src_pool(gl: &mut GLProcessorThreaded, make: fn(usize, usize, u8) -> TensorDyn, frames: usize) {
    let (w, h) = (128, 96);
    let pool: Vec<_> = (0..3).map(|i| make(w, h, i as u8)).collect();
    let mut d1 = img(w, h, PixelFormat::Rgba);
    let mut d2 = img(w * 3 / 2, h * 3 / 2, PixelFormat::Rgba);
    for (i, src) in pool.iter().cycle().take(6 + frames).enumerate() {
        gl.convert(src, &mut d1, None).unwrap();
        gl.convert(src, &mut d2, None).unwrap();
        if i % 100 == 0 {
            eprintln!("src_pool {i}");
        }
    }
}

#[test]
#[ignore]
fn r8_then_yuyv_pool() {
    let mut gl = GLProcessorThreaded::new(None).unwrap();
    src_pool(&mut gl, yuyv, 30);
    assert!(r8_targets(&mut gl) > 0);
    src_pool(&mut gl, yuyv, 300);
}

#[test]
#[ignore]
fn r8_then_multiplane_nv12_pool() {
    let mut gl = GLProcessorThreaded::new(None).unwrap();
    src_pool(&mut gl, mp_nv12, 30);
    assert!(r8_targets(&mut gl) > 0);
    src_pool(&mut gl, mp_nv12, 300);
}

fn median_ms(mut v: Vec<f64>) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[v.len() / 2]
}

/// Median convert time of `make` sources into each destination format at
/// each size: 10 warm-up frames, then 50 timed frames.
fn bench(make: fn(usize, usize, u8) -> TensorDyn) {
    let mut gl = GLProcessorThreaded::new(None).unwrap();
    let path = std::env::var("EDGEFIRST_NV_CONVERT_PATH").unwrap_or_else(|_| "auto".into());
    for (w, h) in [(640usize, 640usize), (1920, 1080)] {
        let src = make(w, h, 5);
        for fmt in [PixelFormat::Rgba, PixelFormat::Rgb, PixelFormat::PlanarRgb] {
            let (dw, dh) = (640, 640);
            let mut dst = img(dw, dh, fmt);
            for _ in 0..10 {
                gl.convert(&src, &mut dst, None).unwrap();
            }
            let mut t = Vec::new();
            for _ in 0..50 {
                let s = std::time::Instant::now();
                gl.convert(&src, &mut dst, None).unwrap();
                t.push(s.elapsed().as_secs_f64() * 1e3);
            }
            let stats = gl.convert_stats().unwrap();
            eprintln!(
                "BENCH path={path} src={w}x{h} dst={fmt:?} {dw}x{dh} median_ms={:.3} min_ms={:.3} stats={stats:?}",
                median_ms(t.clone()),
                t.iter().cloned().fold(f64::MAX, f64::min)
            );
        }
    }
}

#[test]
#[ignore]
fn bench_nv12() {
    bench(nv12);
}

#[test]
#[ignore]
fn bench_mp_nv12() {
    bench(mp_nv12);
}

fn run(jobs: Vec<(&'static str, fn(usize, &str))>) {
    let n = iters();
    let hs: Vec<_> = jobs
        .into_iter()
        .enumerate()
        .map(|(k, (name, f))| std::thread::spawn(move || f(n, &format!("{name}#{k}"))))
        .collect();
    for h in hs {
        h.join().unwrap();
    }
}

#[test]
#[ignore]
fn repro_nv12_x4() {
    run(vec![
        ("nv12", nv12_loop),
        ("nv12", nv12_loop),
        ("nv12", nv12_loop),
        ("nv12", nv12_loop),
    ]);
}

#[test]
#[ignore]
fn repro_nv12_rgba() {
    run(vec![
        ("nv12", nv12_loop),
        ("nv12", nv12_loop),
        ("rgba", rgba_loop),
        ("rgba", rgba_loop),
    ]);
}

#[test]
#[ignore]
fn repro_nv12_churn() {
    run(vec![
        ("nv12", nv12_loop),
        ("churn", churn_loop),
        ("churn", churn_loop),
        ("churn", churn_loop),
    ]);
}

#[test]
#[ignore]
fn repro_nv12_g2d() {
    run(vec![
        ("nv12", nv12_loop),
        ("nv12", nv12_loop),
        ("g2d", g2d_loop),
        ("g2d", g2d_loop),
    ]);
}

#[test]
#[ignore]
fn repro_nv12_single() {
    run(vec![("nv12", nv12_loop)]);
}

#[test]
#[ignore]
fn repro_nv12_nv12dst() {
    run(vec![("nv12", nv12_loop), ("nv12dst", nv12dst_loop)]);
}

#[test]
#[ignore]
fn repro_nv12dst_single() {
    run(vec![("nv12dst", nv12dst_loop)]);
}

#[test]
#[ignore]
fn repro_nv12_grey() {
    run(vec![("nv12", nv12_loop), ("grey", grey_loop)]);
}
