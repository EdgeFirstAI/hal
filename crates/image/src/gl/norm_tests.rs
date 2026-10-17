// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! GL output mapping (normalization, quantization, letterbox placement and
//! output-domain pad) against the CPU converter on the same source.

#[cfg(test)]
#[cfg(feature = "opengl")]
mod gl_norm_tests {
    use crate::{
        CPUProcessor, ComputeBackend, ConvertOptions, ImageProcessor, ImageProcessorConfig,
        ImageProcessorTrait, LetterboxPlacement, Normalization,
    };
    use edgefirst_tensor::{
        CpuAccess, DType, PixelFormat, Quantization, TensorDyn, TensorMapTrait, TensorMemory,
        TensorTrait,
    };

    fn gl_processor() -> Option<ImageProcessor> {
        match ImageProcessor::with_config(ImageProcessorConfig {
            backend: ComputeBackend::OpenGl,
            ..Default::default()
        }) {
            Ok(p) => Some(p),
            Err(e) => {
                crate::test_support::report_skip(&format!("OpenGL backend unavailable: {e}"));
                None
            }
        }
    }

    /// A gradient RGBA source with distinct channels.
    fn fill_source(t: &TensorDyn) {
        let w = t.width().unwrap();
        let mut m = t.as_u8().unwrap().map().unwrap();
        for (i, px) in m.as_chunks_mut::<4>().0.iter_mut().enumerate() {
            let (x, y) = (i % w, i / w);
            *px = [
                (x * 13 + y * 7) as u8,
                (255 - x * 11) as u8,
                (y * 29 + 3) as u8,
                255,
            ];
        }
    }

    /// The logical elements of an image tensor, row by row at its stride.
    fn values(t: &TensorDyn) -> Vec<f64> {
        let fmt = t.format().unwrap();
        let (w, h) = (t.width().unwrap(), t.height().unwrap());
        let (rows, row_len) = match fmt.layout() {
            edgefirst_tensor::PixelLayout::Planar => (fmt.channels() * h, w),
            _ => (h, w * fmt.channels()),
        };
        let stride = t.effective_row_stride().unwrap() / t.dtype().size();
        let flat: Vec<f64> = match t.dtype() {
            DType::F32 => t
                .as_f32()
                .unwrap()
                .map()
                .unwrap()
                .as_slice()
                .iter()
                .map(|&v| v as f64)
                .collect(),
            DType::F16 => t
                .as_f16()
                .unwrap()
                .map()
                .unwrap()
                .as_slice()
                .iter()
                .map(|v| v.to_f64())
                .collect(),
            DType::U8 => t
                .as_u8()
                .unwrap()
                .map()
                .unwrap()
                .as_slice()
                .iter()
                .map(|&v| v as f64)
                .collect(),
            DType::I8 => t
                .as_i8()
                .unwrap()
                .map()
                .unwrap()
                .as_slice()
                .iter()
                .map(|&v| v as f64)
                .collect(),
            _ => unreachable!(),
        };
        flat.chunks(stride)
            .take(rows)
            .flat_map(|r| r[..row_len].to_vec())
            .collect()
    }

    /// Run `opts` through forced GL and through the CPU converter and compare.
    /// The content is placed 1:1 (16x8 source, 16x16 destination), so both
    /// sample texel centres and differ only by arithmetic precision.
    fn compare(fmt: PixelFormat, dtype: DType, opts: ConvertOptions, quant: Option<Quantization>) {
        let Some(mut proc) = gl_processor() else {
            return;
        };
        let src = proc
            .create_image(
                16,
                8,
                PixelFormat::Rgba,
                DType::U8,
                None,
                CpuAccess::ReadWrite,
            )
            .unwrap();
        fill_source(&src);
        let float = matches!(dtype, DType::F32 | DType::F16);
        let support = proc.supported_render_dtypes();
        if float
            && !(if dtype == DType::F32 {
                support.f32
            } else {
                support.f16
            })
        {
            crate::test_support::report_skip(&format!("{fmt:?} {dtype:?}: no GL float render"));
            return;
        }
        let mut gl_dst = match proc.create_image(16, 16, fmt, dtype, None, CpuAccess::ReadWrite) {
            Ok(t) => t,
            Err(e) => {
                crate::test_support::report_skip(&format!("{fmt:?} {dtype:?}: {e}"));
                return;
            }
        };
        // Float renders reach a PBO, or a DMA-BUF only as F16 PlanarRgb
        // (Linux has no other float DMA-BUF format).
        let mem = gl_dst.memory();
        let expect_gl = !float
            || mem == TensorMemory::Pbo
            || (mem == TensorMemory::DmaBuf
                && fmt == PixelFormat::PlanarRgb
                && dtype == DType::F16);
        let mut cpu_src = TensorDyn::image(
            16,
            8,
            PixelFormat::Rgba,
            DType::U8,
            Some(TensorMemory::Mem),
            CpuAccess::ReadWrite,
        )
        .unwrap();
        cpu_src
            .as_u8_mut()
            .unwrap()
            .map()
            .unwrap()
            .as_mut_slice()
            .copy_from_slice(src.as_u8().unwrap().map().unwrap().as_slice());
        let mut cpu_dst = TensorDyn::image(
            16,
            16,
            fmt,
            dtype,
            Some(TensorMemory::Mem),
            CpuAccess::ReadWrite,
        )
        .unwrap();
        if let Some(q) = quant {
            gl_dst.set_quantization(q.clone()).unwrap();
            cpu_dst.set_quantization(q).unwrap();
        }
        let before = proc.convert_fallback_count();
        proc.convert(&src, &mut gl_dst, Some(opts)).unwrap();
        let fell_back = proc.convert_fallback_count() != before;
        if fell_back && !expect_gl {
            crate::test_support::report_skip(&format!(
                "{fmt:?} {dtype:?} in {mem:?}: no GL float path for this memory"
            ));
            return;
        }
        assert!(
            !fell_back,
            "{fmt:?} {dtype:?}: GL declined and the CPU converted"
        );
        CPUProcessor::default()
            .convert(&cpu_src, &mut cpu_dst, Some(opts))
            .unwrap();
        let (g, c) = (values(&gl_dst), values(&cpu_dst));
        assert_eq!(g.len(), c.len());
        // GPU filtering resolves sub-texel weights to about 1/256, which on a
        // steep gradient is a few hundredths of a pixel value: compare to
        // within one ImageNet pixel step (1/57.12), plus f16 rounding.
        let (abs, rel) = match dtype {
            DType::F32 => (0.02, 0.0),
            DType::F16 => (0.02, 2e-3),
            _ => (1.0, 0.0),
        };
        for (i, (a, b)) in g.iter().zip(&c).enumerate() {
            assert!(
                (a - b).abs() <= abs + rel * b.abs(),
                "{fmt:?} {dtype:?} element {i}: gl {a} cpu {b}"
            );
        }
    }

    fn segformer() -> ConvertOptions {
        ConvertOptions::new()
            .letterbox([0, 0, 0, 255])
            .placement(LetterboxPlacement::TopLeft)
            .pad_output([0.0; 4])
            .normalization(Normalization::imagenet())
    }

    #[test]
    fn gl_float_normalization_matches_cpu() {
        for (fmt, dtype) in [
            (PixelFormat::Rgb, DType::F32),
            (PixelFormat::PlanarRgb, DType::F16),
            (PixelFormat::PlanarRgb, DType::F32),
            (PixelFormat::Rgb, DType::F16),
        ] {
            compare(fmt, dtype, segformer(), None);
            let pixel_pad = ConvertOptions::new()
                .letterbox([114, 114, 114, 255])
                .normalization(Normalization::signed());
            compare(fmt, dtype, pixel_pad, None);
        }
    }

    #[test]
    fn gl_int_quantized_normalization_matches_cpu() {
        let q = Quantization::per_tensor(0.0187, -14);
        let pixel_pad = ConvertOptions::new()
            .letterbox([114, 114, 114, 255])
            .normalization(Normalization::imagenet());
        for dtype in [DType::I8, DType::U8] {
            for fmt in [PixelFormat::Rgb, PixelFormat::Rgba, PixelFormat::PlanarRgb] {
                compare(fmt, dtype, pixel_pad, Some(q.clone()));
            }
            // An output-domain pad needs the single-pass route (pass 1 of the
            // planar two-pass route can only pad with a pixel colour).
            for fmt in [PixelFormat::Rgb, PixelFormat::Rgba] {
                compare(fmt, dtype, segformer(), Some(q.clone()));
            }
        }
        // A pad whose channels differ fills each plane with its own value.
        let colour_pad = ConvertOptions::new().letterbox([0, 64, 255, 255]);
        compare(PixelFormat::PlanarRgb, DType::U8, colour_pad, None);
        compare(PixelFormat::PlanarRgb, DType::I8, colour_pad, None);
        // Default normalization with the implicit quantization stays the
        // plain i8 bias on both backends.
        compare(
            PixelFormat::Rgb,
            DType::I8,
            ConvertOptions::new().letterbox([114; 4]),
            None,
        );
    }

    #[test]
    fn gl_nv12_float_output_pad_matches_cpu() {
        let Some(mut proc) = gl_processor() else {
            return;
        };
        if !proc.supported_render_dtypes().f16 {
            crate::test_support::report_skip("no GL F16 render");
            return;
        }
        // Flat chroma: GL and CPU interpolate NV chroma differently at edges.
        let (w, h) = (32, 16);
        let src = proc
            .create_image(
                w,
                h,
                PixelFormat::Nv12,
                DType::U8,
                None,
                CpuAccess::ReadWrite,
            )
            .unwrap();
        {
            let mut m = src.as_u8().unwrap().map().unwrap();
            let stride = src.effective_row_stride().unwrap_or(w);
            let bytes = m.as_mut_slice();
            for (i, b) in bytes.iter_mut().enumerate() {
                let (row, col) = (i / stride, i % stride);
                *b = if row < h {
                    (16 + col * 6 + row * 3) as u8
                } else {
                    128
                };
            }
        }
        let mut gl_dst = proc
            .create_image(
                32,
                32,
                PixelFormat::PlanarRgb,
                DType::F16,
                None,
                CpuAccess::ReadWrite,
            )
            .unwrap();
        let opts = segformer();
        let before = proc.convert_fallback_count();
        proc.convert(&src, &mut gl_dst, Some(opts)).unwrap();
        if proc.convert_fallback_count() != before {
            crate::test_support::report_skip(&format!(
                "NV12 -> PlanarRgb F16 in {:?}: no GL path",
                gl_dst.memory()
            ));
            return;
        }
        let g = values(&gl_dst);
        // Top-left placement: rows 16.. of each plane are pad, written as 0.0.
        for c in 0..3 {
            let plane = &g[c * 32 * 32..(c + 1) * 32 * 32];
            assert!(plane[16 * 32..].iter().all(|&v| v == 0.0), "plane {c} pad");
            assert!(
                plane[..16 * 32].iter().any(|&v| v != 0.0),
                "plane {c} content"
            );
        }
    }
}
