// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

use crate::{
    CPUProcessor, ConvertOptions, Flip, ImageProcessor, ImageProcessorTrait, Region, Rotation,
};
use edgefirst_tensor::TensorMapTrait as _;
use edgefirst_tensor::TensorTrait as _;
use edgefirst_tensor::{
    ColorEncoding, ColorRange, Colorimetry, CpuAccess, DType, PixelFormat, TensorDyn, TensorMemory,
};

type Res = crate::Result<()>;

fn img(w: usize, h: usize, fmt: PixelFormat, mem: TensorMemory) -> Option<TensorDyn> {
    TensorDyn::image(w, h, fmt, DType::U8, Some(mem), CpuAccess::ReadWrite).ok()
}

fn gradient_rgba(w: usize, h: usize) -> TensorDyn {
    let t = img(w, h, PixelFormat::Rgba, TensorMemory::Mem).unwrap();
    {
        let u = t.as_u8().unwrap();
        let stride = u.effective_row_stride().unwrap();
        let mut m = u.map().unwrap();
        for y in 0..h {
            for x in 0..w {
                let o = y * stride + x * 4;
                let s = m.as_mut_slice();
                s[o] = (40 + x * 170 / w) as u8;
                s[o + 1] = (40 + y * 170 / h) as u8;
                s[o + 2] = (200 - (x + y) * 150 / (w + h)) as u8;
                s[o + 3] = 255;
            }
        }
    }
    t
}

fn tight(t: &TensorDyn, bpp: usize) -> Vec<u8> {
    let (w, h) = (t.width().unwrap(), t.height().unwrap());
    let u = t.as_u8().unwrap();
    let stride = u.effective_row_stride().unwrap();
    let m = u.map().unwrap();
    (0..h)
        .flat_map(|r| m.as_slice()[r * stride..][..w * bpp].to_vec())
        .collect()
}

fn make_src(rgba: &TensorDyn, fmt: PixelFormat, tag: Option<Colorimetry>) -> TensorDyn {
    let (w, h) = (rgba.width().unwrap(), rgba.height().unwrap());
    let mut cpu = CPUProcessor::new();
    match fmt {
        PixelFormat::Rgba => {
            let mut t = img(w, h, fmt, TensorMemory::Mem).unwrap();
            cpu.convert(rgba, &mut t, None).unwrap();
            t
        }
        PixelFormat::Vyuy => {
            let mut y = img(w, h, PixelFormat::Yuyv, TensorMemory::Mem).unwrap();
            y.set_colorimetry(tag);
            cpu.convert(rgba, &mut y, None).unwrap();
            let b = tight(&y, 2);
            let v = img(w, h, PixelFormat::Vyuy, TensorMemory::Mem).unwrap();
            {
                let u = v.as_u8().unwrap();
                let stride = u.effective_row_stride().unwrap();
                let mut m = u.map().unwrap();
                for r in 0..h {
                    for i in 0..w / 2 {
                        let s = &b[r * w * 2 + i * 4..][..4];
                        let o = r * stride + i * 4;
                        m.as_mut_slice()[o] = s[3];
                        m.as_mut_slice()[o + 1] = s[0];
                        m.as_mut_slice()[o + 2] = s[1];
                        m.as_mut_slice()[o + 3] = s[2];
                    }
                }
            }
            let mut v = v;
            v.set_colorimetry(tag);
            v
        }
        _ => {
            let mut t = img(w, h, fmt, TensorMemory::Mem).unwrap();
            t.set_colorimetry(tag);
            cpu.convert(rgba, &mut t, None).unwrap();
            t.set_colorimetry(tag);
            t
        }
    }
}

struct Stats {
    mean: f64,
    max: u8,
}

fn stats(a: &[u8], b: &[u8], skip_alpha: bool) -> Stats {
    let mut d: Vec<u8> = a
        .iter()
        .zip(b)
        .enumerate()
        .filter(|(i, _)| !(skip_alpha && i % 4 == 3))
        .map(|(_, (x, y))| x.abs_diff(*y))
        .collect();
    let mean = d.iter().map(|v| *v as f64).sum::<f64>() / d.len() as f64;
    d.sort_unstable();
    Stats {
        mean,
        max: *d.last().unwrap(),
    }
}

fn cm(enc: ColorEncoding, range: ColorRange) -> Option<Colorimetry> {
    Some(Colorimetry::default().with_encoding(enc).with_range(range))
}

fn decode(nv12: &TensorDyn) -> Vec<u8> {
    let (w, h) = (nv12.width().unwrap(), nv12.height().unwrap());
    let mut out = img(w, h, PixelFormat::Rgba, TensorMemory::Mem).unwrap();
    CPUProcessor::new().convert(nv12, &mut out, None).unwrap();
    tight(&out, 4)
}

fn nv12_planes(t: &TensorDyn) -> (Vec<u8>, Vec<u8>) {
    let (w, h) = (t.width().unwrap(), t.height().unwrap());
    let u = t.as_u8().unwrap();
    let ystride = u.effective_row_stride().unwrap();
    let rows = |map: &[u8], first: usize, n: usize, stride: usize| -> Vec<u8> {
        (first..first + n)
            .flat_map(|r| map[r * stride..][..w].to_vec())
            .collect()
    };
    if let Some(c) = u.chroma() {
        let cstride = c.effective_row_stride().unwrap_or(w);
        let ym = u.map().unwrap();
        let cm = c.map().unwrap();
        (
            rows(ym.as_slice(), 0, h, ystride),
            rows(cm.as_slice(), 0, h / 2, cstride),
        )
    } else {
        let m = u.map().unwrap();
        (
            rows(m.as_slice(), 0, h, ystride),
            rows(m.as_slice(), h, h / 2, ystride),
        )
    }
}

fn plane_diff(a: &[u8], b: &[u8]) -> (u8, f64) {
    let max = a.iter().zip(b).map(|(x, y)| x.abs_diff(*y)).max().unwrap();
    let mean = a
        .iter()
        .zip(b)
        .map(|(x, y)| x.abs_diff(*y) as f64)
        .sum::<f64>()
        / a.len() as f64;
    (max, mean)
}

fn solid(w: usize, h: usize, rgb: [u8; 3]) -> TensorDyn {
    let t = img(w, h, PixelFormat::Rgba, TensorMemory::Mem).unwrap();
    {
        let u = t.as_u8().unwrap();
        let stride = u.effective_row_stride().unwrap();
        let mut m = u.map().unwrap();
        for y in 0..h {
            for x in 0..w {
                let o = y * stride + x * 4;
                m.as_mut_slice()[o..o + 4].copy_from_slice(&[rgb[0], rgb[1], rgb[2], 255]);
            }
        }
    }
    t
}

fn expected_yuv(rgb: [u8; 3], enc: ColorEncoding, range: ColorRange) -> [i32; 3] {
    let (kr, kb) = match enc {
        ColorEncoding::Bt601 => (0.299, 0.114),
        _ => (0.2126, 0.0722),
    };
    let kg = 1.0 - kr - kb;
    let [r, g, b] = rgb.map(|v| v as f64 / 255.0);
    let y = kr * r + kg * g + kb * b;
    let u = (b - y) / (2.0 * (1.0 - kb));
    let v = (r - y) / (2.0 * (1.0 - kr));
    let (ys, yo, cs) = match range {
        ColorRange::Full => (255.0, 0.0, 255.0),
        _ => (219.0, 16.0, 224.0),
    };
    [
        (y * ys + yo).round().clamp(0.0, 255.0) as i32,
        (u * cs + 128.0).round().clamp(0.0, 255.0) as i32,
        (v * cs + 128.0).round().clamp(0.0, 255.0) as i32,
    ]
}

fn all_geometries(pad: [u8; 4]) -> Vec<(&'static str, (usize, usize), ConvertOptions)> {
    vec![
        ("same-size", (96, 64), ConvertOptions::new()),
        ("resize", (64, 48), ConvertOptions::new()),
        (
            "crop-odd-offset",
            (64, 32),
            ConvertOptions::new().source(Some(Region {
                x: 3,
                y: 5,
                width: 50,
                height: 30,
            })),
        ),
        (
            "flip-h",
            (64, 48),
            ConvertOptions::new().flip(Flip::Horizontal),
        ),
        (
            "flip-v",
            (64, 48),
            ConvertOptions::new().flip(Flip::Vertical),
        ),
        (
            "rot90",
            (48, 64),
            ConvertOptions::new().rotation(Rotation::Clockwise90),
        ),
        (
            "rot180",
            (64, 48),
            ConvertOptions::new().rotation(Rotation::Rotate180),
        ),
        (
            "rot270",
            (48, 64),
            ConvertOptions::new().rotation(Rotation::CounterClockwise90),
        ),
        (
            "letterbox-even",
            (96, 96),
            ConvertOptions::new().letterbox(pad),
        ),
        (
            "letterbox-odd-offset",
            (100, 60),
            ConvertOptions::new().letterbox(pad),
        ),
        (
            "letterbox-odd-both",
            (66, 96),
            ConvertOptions::new().letterbox(pad),
        ),
    ]
}

/// Every source format into NV12 in heap and DMA memory, through the CPU and through the
/// processor chain, compared with the same geometry rendered to RGBA. A YUV source and its
/// destination carry the same colorimetry.
#[test]
fn nv12_destination_matches_the_rgba_rendering() -> Res {
    let rgba = gradient_rgba(96, 64);
    let pad = [200u8, 30, 90, 255];
    let tags: [(&str, Option<Colorimetry>); 4] = [
        ("untagged", None),
        (
            "bt601-limited",
            cm(ColorEncoding::Bt601, ColorRange::Limited),
        ),
        (
            "bt709-limited",
            cm(ColorEncoding::Bt709, ColorRange::Limited),
        ),
        ("bt709-full", cm(ColorEncoding::Bt709, ColorRange::Full)),
    ];
    let mut chain = ImageProcessor::new()?;
    let mut cpu = CPUProcessor::new();
    let mut fails = Vec::new();
    for fmt in [
        PixelFormat::Yuyv,
        PixelFormat::Vyuy,
        PixelFormat::Rgb,
        PixelFormat::Rgba,
        PixelFormat::Grey,
        PixelFormat::Nv12,
    ] {
        let yuv = matches!(
            fmt,
            PixelFormat::Yuyv | PixelFormat::Vyuy | PixelFormat::Nv12
        );
        for (tname, tag) in tags {
            let src = make_src(&rgba, fmt, if yuv { tag } else { None });
            for mem in [TensorMemory::Mem, TensorMemory::DmaBuf] {
                for (gname, (dw, dh), opts) in all_geometries(pad) {
                    let label = format!("{fmt:?} {mem:?} dst={tname} {gname}");
                    let Some(mut cpu_dst) = img(dw, dh, PixelFormat::Nv12, mem) else {
                        continue;
                    };
                    cpu_dst.set_colorimetry(tag);
                    let mut chain_dst = img(dw, dh, PixelFormat::Nv12, mem).unwrap();
                    chain_dst.set_colorimetry(tag);
                    let mut reference = img(dw, dh, PixelFormat::Rgba, TensorMemory::Mem).unwrap();
                    cpu.convert(&src, &mut reference, Some(opts)).unwrap();
                    if let Err(e) = cpu.convert(&src, &mut cpu_dst, Some(opts)) {
                        fails.push(format!("{label}: cpu {e:?}"));
                        continue;
                    }
                    if let Err(e) = chain.convert(&src, &mut chain_dst, Some(opts)) {
                        fails.push(format!("{label}: chain {e:?}"));
                        continue;
                    }
                    let want = tight(&reference, 4);
                    let from_cpu = stats(&decode(&cpu_dst), &want, true);
                    let from_chain = stats(&decode(&chain_dst), &want, true);
                    let between = plane_diff(&nv12_planes(&cpu_dst).1, &nv12_planes(&chain_dst).1);
                    let odd_edge = gname == "letterbox-odd-offset";
                    if from_cpu.mean > 3.0 || from_chain.mean > 3.0 {
                        fails.push(format!(
                            "{label}: decoded mean diff cpu {:.2} chain {:.2}",
                            from_cpu.mean, from_chain.mean
                        ));
                    }
                    if !odd_edge && (from_cpu.max > 40 || from_chain.max > 40) {
                        fails.push(format!(
                            "{label}: decoded max diff cpu {} chain {}",
                            from_cpu.max, from_chain.max
                        ));
                    }
                    if between.1 > 2.0 {
                        fails.push(format!(
                            "{label}: cpu vs chain chroma mean {:.2}",
                            between.1
                        ));
                    }
                }
            }
        }
    }
    assert!(
        fails.is_empty(),
        "{} failures:\n{}",
        fails.len(),
        fails.join("\n")
    );
    Ok(())
}

/// Saturated and range-edge colours encode to the analytic Y, U and V values.
#[test]
fn nv12_destination_encodes_extreme_colours() -> Res {
    let colours: [[u8; 3]; 10] = [
        [0, 0, 0],
        [255, 255, 255],
        [255, 0, 0],
        [0, 255, 0],
        [0, 0, 255],
        [255, 255, 0],
        [0, 255, 255],
        [255, 0, 255],
        [16, 16, 16],
        [235, 235, 235],
    ];
    let mut chain = ImageProcessor::new()?;
    let mut cpu = CPUProcessor::new();
    let mut fails = Vec::new();
    for fmt in [PixelFormat::Rgba, PixelFormat::Rgb, PixelFormat::Yuyv] {
        for (enc, range) in [
            (ColorEncoding::Bt601, ColorRange::Limited),
            (ColorEncoding::Bt709, ColorRange::Limited),
            (ColorEncoding::Bt601, ColorRange::Full),
            (ColorEncoding::Bt709, ColorRange::Full),
        ] {
            for rgb in colours {
                let tag = cm(enc, range);
                let src = make_src(&solid(64, 64, rgb), fmt, tag);
                for (gname, dims) in [("same-size", 64usize), ("resize", 32)] {
                    for mem in [TensorMemory::Mem, TensorMemory::DmaBuf] {
                        for engine in ["cpu", "chain"] {
                            let Some(mut dst) = img(dims, dims, PixelFormat::Nv12, mem) else {
                                continue;
                            };
                            dst.set_colorimetry(tag);
                            let r = if engine == "cpu" {
                                cpu.convert(&src, &mut dst, None)
                            } else {
                                chain.convert(&src, &mut dst, None)
                            };
                            let label = format!(
                                "{fmt:?} {enc:?}/{range:?} rgb={rgb:?} {gname} {engine} {mem:?}"
                            );
                            if let Err(e) = r {
                                fails.push(format!("{label}: {e:?}"));
                                continue;
                            }
                            let (y, uv) = nv12_planes(&dst);
                            let want = expected_yuv(rgb, enc, range);
                            let at = dims / 2;
                            let got = [
                                y[at * dims + at] as i32,
                                uv[(at / 2) * dims + (at & !1)] as i32,
                                uv[(at / 2) * dims + (at & !1) + 1] as i32,
                            ];
                            let tol = if fmt == PixelFormat::Yuyv { 3 } else { 2 };
                            if got.iter().zip(want).any(|(g, w)| (g - w).abs() > tol) {
                                fails.push(format!("{label}: got {got:?} want {want:?}"));
                            }
                        }
                    }
                }
            }
        }
    }
    assert!(
        fails.is_empty(),
        "{} failures:\n{}",
        fails.len(),
        fails.join("\n")
    );
    Ok(())
}

fn multiplane(
    w: usize,
    h: usize,
    luma_pitch: usize,
    chroma_pitch: usize,
    mem: TensorMemory,
) -> Option<TensorDyn> {
    use edgefirst_tensor::Tensor;
    let mut luma = Tensor::<u8>::new(&[h, luma_pitch], Some(mem), None).ok()?;
    let mut chroma = Tensor::<u8>::new(&[h / 2, chroma_pitch], Some(mem), None).ok()?;
    luma.map_mut().ok()?.as_mut_slice().fill(0xA5);
    chroma.map_mut().ok()?.as_mut_slice().fill(0xA5);
    luma.set_logical_shape(&[h, w]).ok()?;
    chroma.set_logical_shape(&[h / 2, w]).ok()?;
    chroma.set_row_stride_unchecked(chroma_pitch);
    let mut dst = TensorDyn::from(Tensor::<u8>::from_planes(luma, chroma, PixelFormat::Nv12).ok()?);
    dst.set_row_stride(luma_pitch).ok()?;
    Some(dst)
}

/// A two-plane destination with padded luma and chroma pitches matches the single-plane
/// rendering and leaves the padding bytes alone.
#[test]
fn two_plane_nv12_destination_with_padded_pitches() -> Res {
    let rgba = gradient_rgba(96, 64);
    let geoms = [
        ("same-size", (96usize, 64usize), ConvertOptions::new()),
        ("resize", (64, 48), ConvertOptions::new()),
        (
            "flip-h",
            (64, 48),
            ConvertOptions::new().flip(Flip::Horizontal),
        ),
        (
            "rot90",
            (48, 64),
            ConvertOptions::new().rotation(Rotation::Clockwise90),
        ),
        (
            "letterbox-odd-offset",
            (100, 60),
            ConvertOptions::new().letterbox([200, 30, 90, 255]),
        ),
    ];
    let mut chain = ImageProcessor::new()?;
    let mut cpu = CPUProcessor::new();
    let mut fails = Vec::new();
    for fmt in [PixelFormat::Yuyv, PixelFormat::Rgba, PixelFormat::Nv12] {
        let src = make_src(&rgba, fmt, None);
        for mem in [TensorMemory::Mem, TensorMemory::DmaBuf] {
            for (lp_pad, cp_pad) in [(0usize, 0usize), (64, 64), (0, 192), (128, 32)] {
                for (gname, (dw, dh), opts) in &geoms {
                    let label = format!("{fmt:?} {mem:?} pad=({lp_pad},{cp_pad}) {gname}");
                    let Some(mut got) = multiplane(*dw, *dh, dw + lp_pad, dw + cp_pad, mem) else {
                        fails.push(format!("{label}: two-plane destination not built"));
                        continue;
                    };
                    let mut want = img(*dw, *dh, PixelFormat::Nv12, TensorMemory::Mem).unwrap();
                    cpu.convert(&src, &mut want, Some(*opts)).unwrap();
                    if let Err(e) = chain.convert(&src, &mut got, Some(*opts)) {
                        fails.push(format!("{label}: {e:?}"));
                        continue;
                    }
                    let (gy, guv) = nv12_planes(&got);
                    let (wy, wuv) = nv12_planes(&want);
                    let (ym, ys) = plane_diff(&gy, &wy);
                    let (um, us) = plane_diff(&guv, &wuv);
                    if ys > 2.0 || us > 2.0 {
                        fails.push(format!(
                            "{label}: Y max {ym} mean {ys:.2}, UV max {um} mean {us:.2}"
                        ));
                    }
                    let u = got.as_u8().unwrap();
                    let luma = u.map().unwrap();
                    let lstride = u.effective_row_stride().unwrap();
                    let chroma = u.chroma().unwrap();
                    let cstride = chroma.effective_row_stride().unwrap_or(*dw);
                    let cmap = chroma.map().unwrap();
                    let luma_dirty = (0..*dh).any(|r| {
                        luma.as_slice()[r * lstride + dw..][..lstride - dw]
                            .iter()
                            .any(|b| *b != 0xA5)
                    });
                    let chroma_dirty = (0..*dh / 2).any(|r| {
                        cmap.as_slice()[r * cstride + dw..][..cstride - dw]
                            .iter()
                            .any(|b| *b != 0xA5)
                    });
                    if luma_dirty || chroma_dirty {
                        fails.push(format!(
                            "{label}: padding written (luma {luma_dirty}, chroma {chroma_dirty})"
                        ));
                    }
                }
            }
        }
    }
    assert!(
        fails.is_empty(),
        "{} failures:\n{}",
        fails.len(),
        fails.join("\n")
    );
    Ok(())
}

/// An odd width or height is refused with `NotSupported` by the CPU and by the processor chain,
/// and the smallest even size converts.
#[test]
fn odd_nv12_destination_sizes_are_refused() -> Res {
    let mut chain = ImageProcessor::new()?;
    let mut cpu = CPUProcessor::new();
    for fmt in [
        PixelFormat::Rgba,
        PixelFormat::Yuyv,
        PixelFormat::Nv12,
        PixelFormat::Grey,
    ] {
        let src = make_src(&gradient_rgba(96, 64), fmt, None);
        for (dw, dh) in [(63usize, 64usize), (64, 63), (63, 63), (1, 1)] {
            for mem in [TensorMemory::Mem, TensorMemory::DmaBuf] {
                let mut dst = img(dw, dh, PixelFormat::Nv12, mem).unwrap();
                for (engine, r) in [
                    ("cpu", cpu.convert(&src, &mut dst, None)),
                    ("chain", chain.convert(&src, &mut dst, None)),
                ] {
                    assert!(
                        matches!(r, Err(crate::Error::NotSupported(_))),
                        "{fmt:?}->{dw}x{dh} {mem:?} {engine}: {r:?}"
                    );
                }
            }
        }
        let mut dst = img(2, 2, PixelFormat::Nv12, TensorMemory::Mem).unwrap();
        cpu.convert(&src, &mut dst, None)?;
        chain.convert(&src, &mut dst, None)?;
    }
    Ok(())
}

/// With an odd letterbox offset, CPU and GL agree on the chroma pair that straddles the edge for
/// every source format.
#[test]
fn odd_letterbox_edge_chroma_agrees_between_engines() -> Res {
    let rgba = gradient_rgba(96, 64);
    let mut chain = ImageProcessor::new()?;
    let mut cpu = CPUProcessor::new();
    let opts = ConvertOptions::new().letterbox([200, 30, 90, 255]);
    let mut fails = Vec::new();
    for fmt in [
        PixelFormat::Rgba,
        PixelFormat::Rgb,
        PixelFormat::Grey,
        PixelFormat::Yuyv,
        PixelFormat::Nv12,
    ] {
        let src = make_src(&rgba, fmt, None);
        let mut want = img(100, 60, PixelFormat::Nv12, TensorMemory::Mem).unwrap();
        cpu.convert(&src, &mut want, Some(opts))?;
        let mut got = img(100, 60, PixelFormat::Nv12, TensorMemory::DmaBuf).unwrap();
        chain.convert(&src, &mut got, Some(opts))?;
        let (um, _) = plane_diff(&nv12_planes(&got).1, &nv12_planes(&want).1);
        if um > 4 {
            fails.push(format!("{fmt:?}: chroma max diff {um}"));
        }
    }
    assert!(fails.is_empty(), "{}", fails.join("\n"));
    Ok(())
}

/// NV16 and NV24 sources convert to NV12 on the CPU and through the processor chain, whatever the
/// destination memory, and decode to the same picture as the CPU's RGBA conversion of the source.
#[test]
fn nv12_destination_support_does_not_depend_on_the_destination_memory() -> Res {
    let mut chain = ImageProcessor::new()?;
    let mut cpu = CPUProcessor::new();
    let (w, h) = (96usize, 64usize);
    let yuyv = make_src(&gradient_rgba(w, h), PixelFormat::Yuyv, None);
    let mut nv16 = img(w, h, PixelFormat::Nv16, TensorMemory::Mem).unwrap();
    cpu.convert(&yuyv, &mut nv16, None)?;
    let nv24 = img(w, h, PixelFormat::Nv24, TensorMemory::Mem).unwrap();
    {
        let s = nv16.as_u8().unwrap().map().unwrap();
        let mut d = nv24.as_u8().unwrap().map().unwrap();
        let (src, dst) = (s.as_slice(), d.as_mut_slice());
        dst[..w * h].copy_from_slice(&src[..w * h]);
        for (i, pair) in src[w * h..w * h * 2].chunks_exact(2).enumerate() {
            let o = w * h + i * 4;
            dst[o..o + 2].copy_from_slice(pair);
            dst[o + 2..o + 4].copy_from_slice(pair);
        }
    }
    let mut fails = Vec::new();
    for (fmt, src) in [(PixelFormat::Nv16, &nv16), (PixelFormat::Nv24, &nv24)] {
        for (gname, (dw, dh), opts) in all_geometries([0, 0, 0, 255])
            .into_iter()
            .filter(|(n, _, _)| ["same-size", "resize", "crop-odd-offset"].contains(n))
        {
            let mut reference = img(dw, dh, PixelFormat::Rgba, TensorMemory::Mem).unwrap();
            cpu.convert(src, &mut reference, Some(opts))?;
            let want = tight(&reference, 4);
            let mut cpu_dst = img(dw, dh, PixelFormat::Nv12, TensorMemory::Mem).unwrap();
            if let Err(e) = cpu.convert(src, &mut cpu_dst, Some(opts)) {
                fails.push(format!("{fmt:?} {gname}: cpu {e:?}"));
            } else {
                let st = stats(&decode(&cpu_dst), &want, true);
                if st.mean > 3.0 {
                    fails.push(format!("{fmt:?} {gname}: cpu decoded mean {:.2}", st.mean));
                }
            }
            let mut oks = Vec::new();
            for mem in [TensorMemory::Mem, TensorMemory::DmaBuf] {
                let Some(mut dst) = img(dw, dh, PixelFormat::Nv12, mem) else {
                    continue;
                };
                let ok = chain.convert(src, &mut dst, Some(opts));
                if let Err(e) = &ok {
                    fails.push(format!("{fmt:?} {gname} {mem:?}: chain {e:?}"));
                } else {
                    let st = stats(&decode(&dst), &want, true);
                    if st.mean > 3.0 {
                        fails.push(format!(
                            "{fmt:?} {gname} {mem:?}: chain decoded mean {:.2}",
                            st.mean
                        ));
                    }
                }
                oks.push(ok.is_ok());
            }
            if oks.windows(2).any(|p| p[0] != p[1]) {
                fails.push(format!(
                    "{fmt:?} {gname}: heap and DMA destinations disagree"
                ));
            }
        }
    }
    assert!(fails.is_empty(), "{}", fails.join("\n"));
    Ok(())
}

/// A letterbox fill over a partially written NV12 destination leaves the interior bytes alone.
#[test]
fn nv12_letterbox_fill_preserves_the_interior_of_a_partial_write() -> Res {
    use crate::Rect;
    let (w, h) = (64usize, 48usize);
    let interior = Rect::new(16, 12, 32, 24);
    for mem in [TensorMemory::Mem, TensorMemory::DmaBuf] {
        let Some(mut dst) = img(w, h, PixelFormat::Nv12, mem) else {
            continue;
        };
        {
            let u = dst.as_u8().unwrap();
            u.map().unwrap().as_mut_slice().fill(0x5A);
            if let Some(c) = u.chroma() {
                c.map().unwrap().as_mut_slice().fill(0x5A);
            }
        }
        CPUProcessor::fill_image_outside_crop_u8(
            dst.as_u8_mut().unwrap(),
            [200, 30, 90, 255],
            interior,
            true,
        )?;
        let (y, uv) = nv12_planes(&dst);
        for row in 0..h {
            for col in 0..w {
                let inside = (interior.left..interior.left + interior.width).contains(&col)
                    && (interior.top..interior.top + interior.height).contains(&row);
                let v = y[row * w + col];
                assert_eq!(v == 0x5A, inside, "{mem:?} luma at ({col},{row}) = {v:#x}");
            }
        }
        for row in 0..h / 2 {
            for col in 0..w {
                let inside = (interior.left..interior.left + interior.width).contains(&col)
                    && (interior.top / 2..(interior.top + interior.height) / 2).contains(&row);
                let v = uv[row * w + col];
                assert_eq!(
                    v == 0x5A,
                    inside,
                    "{mem:?} chroma at ({col},{row}) = {v:#x}"
                );
            }
        }
    }
    Ok(())
}

/// A YUV source copied into an NV12 destination with a different colorimetry tag is
/// re-encoded for the destination.
#[test]
fn same_size_yuv_copy_honours_the_destination_tag() -> Res {
    let rgba = gradient_rgba(96, 64);
    let mut cpu = CPUProcessor::new();
    let mut fails = Vec::new();
    for fmt in [PixelFormat::Yuyv, PixelFormat::Vyuy, PixelFormat::Nv12] {
        let src = make_src(&rgba, fmt, None);
        for tag in [
            cm(ColorEncoding::Bt709, ColorRange::Limited),
            cm(ColorEncoding::Bt709, ColorRange::Full),
            cm(ColorEncoding::Bt601, ColorRange::Full),
        ] {
            let mut dst = img(96, 64, PixelFormat::Nv12, TensorMemory::Mem).unwrap();
            dst.set_colorimetry(tag);
            cpu.convert(&src, &mut dst, None)?;
            let mut want = img(96, 64, PixelFormat::Rgba, TensorMemory::Mem).unwrap();
            cpu.convert(&src, &mut want, None)?;
            let s = stats(&decode(&dst), &tight(&want, 4), true);
            if s.mean > 2.0 {
                fails.push(format!(
                    "{fmt:?} -> {tag:?}: decoded mean diff {:.2} max {}",
                    s.mean, s.max
                ));
            }
        }
    }
    assert!(fails.is_empty(), "{}", fails.join("\n"));
    Ok(())
}

/// An untagged NV12 destination fed by an untagged YUV source decodes, as the destination's own
/// size resolves its colorimetry, to the colours the source decodes to.
#[test]
fn untagged_resize_across_the_hd_threshold_round_trips() -> Res {
    let src = make_src(&gradient_rgba(1280, 720), PixelFormat::Yuyv, None);
    let mut cpu = CPUProcessor::new();
    let mut dst = img(640, 360, PixelFormat::Nv12, TensorMemory::Mem).unwrap();
    cpu.convert(&src, &mut dst, None)?;
    let mut want = img(640, 360, PixelFormat::Rgba, TensorMemory::Mem).unwrap();
    cpu.convert(&src, &mut want, None)?;
    let s = stats(&decode(&dst), &tight(&want, 4), true);
    assert!(
        s.mean <= 2.0,
        "decoded mean diff {:.2} max {}",
        s.mean,
        s.max
    );
    Ok(())
}

/// An untagged NV12 destination is tagged with the colorimetry its own height implies, and keeps
/// that tag when a later source carries another.
#[test]
fn pooled_yuv_destination_keeps_its_own_colorimetry() -> Res {
    let mut cpu = CPUProcessor::new();
    let mut dst = img(640, 360, PixelFormat::Nv12, TensorMemory::Mem).unwrap();
    let hd = make_src(
        &gradient_rgba(1280, 720),
        PixelFormat::Yuyv,
        cm(ColorEncoding::Bt709, ColorRange::Limited),
    );
    cpu.convert(&hd, &mut dst, None)?;
    let tag = dst.colorimetry().expect("the destination is tagged");
    assert_eq!(tag.encoding, Some(ColorEncoding::Bt601));
    assert_eq!(tag.range, Some(ColorRange::Limited));
    let full_hd = make_src(
        &gradient_rgba(1920, 1080),
        PixelFormat::Yuyv,
        cm(ColorEncoding::Bt709, ColorRange::Full),
    );
    cpu.convert(&full_hd, &mut dst, None)?;
    assert_eq!(dst.colorimetry(), Some(tag));
    Ok(())
}

/// Every conversion into an untagged YUV destination tags it with the colorimetry its height
/// implies, on the CPU and through the processor chain's blocking and fenced converts.
#[test]
fn convert_tags_every_untagged_yuv_destination() -> Res {
    let src = gradient_rgba(320, 240);
    let mut cpu = CPUProcessor::new();
    let mut chain = ImageProcessor::new()?;
    let mut fails = Vec::new();
    for fmt in [PixelFormat::Nv12, PixelFormat::Nv16, PixelFormat::Yuyv] {
        for (w, h) in [(320usize, 240usize), (640, 360), (1280, 720)] {
            let want = Some(crate::colorimetry::resolve_colorimetry(None, Some(h)));
            let mut dst = img(w, h, fmt, TensorMemory::Mem).unwrap();
            cpu.convert(&src, &mut dst, None)?;
            if dst.colorimetry() != want {
                fails.push(format!("cpu {fmt:?} {w}x{h}: {:?}", dst.colorimetry()));
            }
            for mem in [TensorMemory::Mem, TensorMemory::DmaBuf] {
                let Some(mut dst) = img(w, h, fmt, mem) else {
                    continue;
                };
                chain.convert(&src, &mut dst, None)?;
                if dst.colorimetry() != want {
                    fails.push(format!(
                        "chain {fmt:?} {w}x{h} {mem:?}: {:?}",
                        dst.colorimetry()
                    ));
                }
                let mut fenced = img(w, h, fmt, mem).unwrap();
                let fence = chain.convert_with_fence(&src, &mut fenced, None)?;
                if fenced.colorimetry() != want {
                    fails.push(format!(
                        "chain fenced {fmt:?} {w}x{h} {mem:?}: {:?}",
                        fenced.colorimetry()
                    ));
                }
                chain.convert(&src, &mut fenced, None)?;
                drop(fence);
            }
        }
    }
    assert!(fails.is_empty(), "{}", fails.join("\n"));
    Ok(())
}

/// A conversion that fails leaves an untagged destination untagged.
#[test]
fn failed_convert_leaves_destination_untagged() -> Res {
    let rgba = gradient_rgba(96, 64);
    let mut cpu = CPUProcessor::new();
    let mut chain = ImageProcessor::new()?;
    let mut odd = img(63, 64, PixelFormat::Nv12, TensorMemory::Mem).unwrap();
    assert!(cpu.convert(&rgba, &mut odd, None).is_err());
    assert!(chain.convert(&rgba, &mut odd, None).is_err());
    assert_eq!(odd.colorimetry(), None);
    let vyuy = make_src(
        &rgba,
        PixelFormat::Vyuy,
        cm(ColorEncoding::Bt709, ColorRange::Full),
    );
    let mut dst = img(96, 64, PixelFormat::Vyuy, TensorMemory::Mem).unwrap();
    let r = chain.convert(&vyuy, &mut dst, None);
    assert!(matches!(r, Err(crate::Error::NotSupported(_))), "{r:?}");
    assert_eq!(dst.colorimetry(), None);
    Ok(())
}

/// The luma samples of a YUYV, VYUY, NV12 or NV16 image, row by row.
fn luma(t: &TensorDyn) -> Vec<u8> {
    match t.format().unwrap() {
        PixelFormat::Yuyv => tight(t, 2).into_iter().step_by(2).collect(),
        PixelFormat::Vyuy => tight(t, 2).into_iter().skip(1).step_by(2).collect(),
        _ => tight(t, 1),
    }
}

/// A same-size YUV to YUV conversion copies the samples when both sides resolve to the same
/// coding, re-encodes them when the codings differ, and refuses a VYUY destination it would
/// have to re-encode.
#[test]
fn same_size_yuv_conversions_follow_the_destination_coding() -> Res {
    let rgba = gradient_rgba(96, 64);
    let mut cpu = CPUProcessor::new();
    let mut fails = Vec::new();
    let pairs = [
        (PixelFormat::Yuyv, PixelFormat::Yuyv),
        (PixelFormat::Yuyv, PixelFormat::Nv16),
        (PixelFormat::Yuyv, PixelFormat::Nv12),
        (PixelFormat::Vyuy, PixelFormat::Vyuy),
        (PixelFormat::Vyuy, PixelFormat::Nv16),
        (PixelFormat::Vyuy, PixelFormat::Nv12),
        (PixelFormat::Nv12, PixelFormat::Nv12),
    ];
    for (src_fmt, dst_fmt) in pairs {
        let src = make_src(&rgba, src_fmt, None);
        let mut want = img(96, 64, PixelFormat::Rgba, TensorMemory::Mem).unwrap();
        cpu.convert(&src, &mut want, None)?;
        let want = tight(&want, 4);
        for tag in [None, cm(ColorEncoding::Bt601, ColorRange::Limited)] {
            let mut dst = img(96, 64, dst_fmt, TensorMemory::Mem).unwrap();
            dst.set_colorimetry(tag);
            cpu.convert(&src, &mut dst, None)?;
            if luma(&dst) != luma(&src) {
                fails.push(format!(
                    "{src_fmt:?} -> {dst_fmt:?} {tag:?}: luma is not a copy"
                ));
            }
        }
        for tag in [
            cm(ColorEncoding::Bt709, ColorRange::Limited),
            cm(ColorEncoding::Bt709, ColorRange::Full),
            cm(ColorEncoding::Bt601, ColorRange::Full),
        ] {
            let mut dst = img(96, 64, dst_fmt, TensorMemory::Mem).unwrap();
            dst.set_colorimetry(tag);
            let r = cpu.convert(&src, &mut dst, None);
            if dst_fmt == PixelFormat::Vyuy {
                if !matches!(r, Err(crate::Error::NotSupported(_))) {
                    fails.push(format!("{src_fmt:?} -> {dst_fmt:?} {tag:?}: {r:?}"));
                }
                continue;
            }
            r?;
            let s = stats(&decode(&dst), &want, true);
            if s.mean > 2.0 {
                fails.push(format!(
                    "{src_fmt:?} -> {dst_fmt:?} {tag:?}: decoded mean diff {:.2} max {}",
                    s.mean, s.max
                ));
            }
        }
    }
    assert!(fails.is_empty(), "{}", fails.join("\n"));
    Ok(())
}
