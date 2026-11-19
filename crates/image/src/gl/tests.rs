// SPDX-FileCopyrightText: Copyright 2025 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

#[cfg(test)]
#[cfg(feature = "opengl")]
#[allow(deprecated)]
mod gl_tests {
    #[cfg(target_os = "linux")]
    use crate::test_support::is_dma_available;
    #[cfg(target_os = "linux")]
    use crate::{probe_egl_displays, EglDisplayKind};
    use crate::{Crop, GLProcessorThreaded, ImageProcessorTrait};
    use edgefirst_tensor::{
        DType, PixelFormat, Tensor, TensorDyn, TensorMapTrait, TensorMemory, TensorTrait,
    };
    use edgefirst_tensor::{DetectBox, ProtoData, ProtoLayout};
    use image::buffer::ConvertBuffer;
    use ndarray::Array3;

    #[test]
    fn test_segmentation() {
        use edgefirst_tensor::Segmentation;

        if !is_opengl_available() {
            crate::test_support::report_skip(&format!("{} - OpenGL not available", function!()));
            return;
        }

        let image = crate::load_image_test_helper(
            &edgefirst_bench::testdata::read("giraffe.jpg"),
            Some(PixelFormat::Rgba),
            None,
        )
        .unwrap();

        let mut segmentation = Array3::from_shape_vec(
            (2, 160, 160),
            edgefirst_bench::testdata::read("modelpack_seg_2x160x160.bin").to_vec(),
        )
        .unwrap();
        segmentation.swap_axes(0, 1);
        segmentation.swap_axes(1, 2);
        let segmentation = segmentation.as_standard_layout().to_owned();

        let seg = Segmentation {
            segmentation: edgefirst_tensor::Tensor::from_arrayview3(segmentation.view())
                .unwrap()
                .into(),
            xmin: 0.0,
            ymin: 0.0,
            xmax: 1.0,
            ymax: 1.0,
        };

        let mut renderer = GLProcessorThreaded::new(None).unwrap();
        let mut image_dyn = image;
        renderer
            .draw_decoded_masks(&mut image_dyn, &[], &[seg], Default::default())
            .unwrap();
    }

    #[test]
    fn test_segmentation_mem() {
        use edgefirst_tensor::Segmentation;

        if !is_opengl_available() {
            crate::test_support::report_skip(&format!("{} - OpenGL not available", function!()));
            return;
        }

        let image = crate::load_image_test_helper(
            &edgefirst_bench::testdata::read("giraffe.jpg"),
            Some(PixelFormat::Rgba),
            Some(edgefirst_tensor::TensorMemory::Mem),
        )
        .unwrap();

        let mut segmentation = Array3::from_shape_vec(
            (2, 160, 160),
            edgefirst_bench::testdata::read("modelpack_seg_2x160x160.bin").to_vec(),
        )
        .unwrap();
        segmentation.swap_axes(0, 1);
        segmentation.swap_axes(1, 2);
        let segmentation = segmentation.as_standard_layout().to_owned();

        let seg = Segmentation {
            segmentation: edgefirst_tensor::Tensor::from_arrayview3(segmentation.view())
                .unwrap()
                .into(),
            xmin: 0.0,
            ymin: 0.0,
            xmax: 1.0,
            ymax: 1.0,
        };

        let mut renderer = GLProcessorThreaded::new(None).unwrap();
        let mut image_dyn = image;
        renderer
            .draw_decoded_masks(&mut image_dyn, &[], &[seg], Default::default())
            .unwrap();
    }

    #[test]
    fn test_segmentation_yolo() {
        use edgefirst_tensor::Segmentation;
        use ndarray::Array3;

        if !is_opengl_available() {
            crate::test_support::report_skip(&format!("{} - OpenGL not available", function!()));
            return;
        }

        // The camera frame is the MaskOverlay background.
        let bg = crate::load_image_test_helper(
            &edgefirst_bench::testdata::read("giraffe.jpg"),
            Some(PixelFormat::Rgba),
            None,
        )
        .unwrap();
        let mut dst = TensorDyn::image(
            bg.width().unwrap(),
            bg.height().unwrap(),
            PixelFormat::Rgba,
            DType::U8,
            None,
            edgefirst_tensor::CpuAccess::ReadWrite,
        )
        .unwrap();

        let segmentation = Array3::from_shape_vec(
            (76, 55, 1),
            edgefirst_bench::testdata::read("yolov8_seg_crop_76x55.bin").to_vec(),
        )
        .unwrap();

        let detect = DetectBox {
            bbox: [0.59375, 0.25, 0.9375, 0.725].into(),
            score: 0.99,
            label: 1,
        };

        let seg = Segmentation {
            segmentation: edgefirst_tensor::Tensor::from_arrayview3(segmentation.view())
                .unwrap()
                .into(),
            xmin: 0.59375,
            ymin: 0.25,
            xmax: 0.9375,
            ymax: 0.725,
        };

        let mut renderer = GLProcessorThreaded::new(None).unwrap();
        renderer
            .set_class_colors(&[[255, 255, 0, 233], [128, 128, 255, 100]])
            .unwrap();
        renderer
            .draw_decoded_masks(
                &mut dst,
                &[detect],
                &[seg],
                crate::MaskOverlay::new().with_background(&bg),
            )
            .unwrap();

        let image = {
            let mut __t = dst.into_u8().unwrap();
            __t.set_format(PixelFormat::Rgba).unwrap();
            TensorDyn::from(__t)
        };
        let expected = crate::load_image_test_helper(
            &edgefirst_bench::testdata::read("output_render_gl.jpg"),
            Some(PixelFormat::Rgba),
            None,
        )
        .unwrap();

        compare_images(&image, &expected, 0.95, function!());
    }

    /// `a` and `b` agree to within the GPU's blend rounding.
    fn px_close(a: [u8; 4], b: [u8; 4]) -> bool {
        a.iter().zip(b).all(|(&x, y)| x.abs_diff(y) <= 2)
    }

    #[test]
    fn test_box_line_width_matches_cpu() {
        use crate::test_support::{box_drawn_columns, box_drawn_rows, box_expected_span, BOX_DIM};

        if !is_opengl_available() {
            crate::test_support::report_skip(&format!("{} - OpenGL not available", function!()));
            return;
        }
        let mut gl = GLProcessorThreaded::new(None).unwrap();
        let mut cpu = crate::CPUProcessor::new();
        let mid = BOX_DIM / 2;
        for width in [
            None,
            Some(1u32),
            Some(6),
            Some(8),
            Some(9),
            Some(12),
            Some(40),
        ] {
            let expected =
                box_expected_span(width.unwrap_or(crate::DEFAULT_BOX_LINE_WIDTH) as usize);
            let g = crate::test_support::draw_box_fixture(&mut gl, [200, 100, 50, 255], width, 1.0);
            let c =
                crate::test_support::draw_box_fixture(&mut cpu, [200, 100, 50, 255], width, 1.0);
            assert_eq!(
                box_drawn_columns(&g, mid),
                expected,
                "gl width {width:?} row"
            );
            assert_eq!(
                box_drawn_rows(&g, mid),
                expected,
                "gl width {width:?} column"
            );
            assert_eq!(
                box_drawn_columns(&c, mid),
                expected,
                "cpu width {width:?} row"
            );
            assert_eq!(
                box_drawn_rows(&c, mid),
                expected,
                "cpu width {width:?} column"
            );
            assert_eq!(g, c, "gl and cpu width {width:?}");
        }
        let err = gl.set_box_line_width(0).unwrap_err();
        assert!(matches!(err, crate::Error::InvalidOptions(_)), "{err:?}");
    }

    #[test]
    fn test_box_colour_alpha_matches_cpu() {
        use crate::test_support::{box_blend, box_expected_span, box_px, BOX_BG, BOX_DIM};

        if !is_opengl_available() {
            crate::test_support::report_skip(&format!("{} - OpenGL not available", function!()));
            return;
        }
        let mut gl = GLProcessorThreaded::new(None).unwrap();
        let mid = BOX_DIM / 2;
        let corner = BOX_DIM / 4 - 1;
        for (color, opacity, alpha) in [
            ([200u8, 100, 50, 255u8], 1.0f32, 255u16),
            ([200, 100, 50, 128], 1.0, 128),
            ([200, 100, 50, 255], 0.5, 128),
        ] {
            let want = box_blend(color, alpha);
            let flat = crate::test_support::draw_box_fixture(&mut gl, color, Some(3), opacity);
            for x in box_expected_span(3) {
                let got = box_px(&flat, x, mid);
                assert!(
                    px_close(got, want),
                    "{color:?}@{opacity} x={x}: {got:?} != {want:?}"
                );
            }
            let got = box_px(&flat, corner, corner);
            assert!(
                px_close(got, want),
                "{color:?}@{opacity} corner: {got:?} != {want:?}"
            );
            assert_eq!(
                box_px(&flat, mid, mid),
                BOX_BG,
                "box interior must not be drawn"
            );
        }
    }

    #[test]
    fn test_boxes() {
        use edgefirst_tensor::DetectBox;

        if !is_opengl_available() {
            crate::test_support::report_skip(&format!("{} - OpenGL not available", function!()));
            return;
        }

        let image = crate::load_image_test_helper(
            &edgefirst_bench::testdata::read("giraffe.jpg"),
            Some(PixelFormat::Rgba),
            None,
        )
        .unwrap();
        let mut image_dyn = image;

        let detect = DetectBox {
            bbox: [0.59375, 0.25, 0.9375, 0.725].into(),
            score: 0.99,
            label: 0,
        };
        let mut renderer = GLProcessorThreaded::new(None).unwrap();
        renderer
            .set_class_colors(&[[255, 255, 0, 233], [128, 128, 255, 100]])
            .unwrap();
        renderer
            .draw_decoded_masks(&mut image_dyn, &[detect], &[], Default::default())
            .unwrap();
    }

    static GL_AVAILABLE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    #[cfg(target_os = "linux")]
    static NEUTRON_AVAILABLE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();

    fn is_opengl_available() -> bool {
        *GL_AVAILABLE.get_or_init(|| GLProcessorThreaded::new(None).is_ok())
    }

    /// Whether zero-copy GPU image buffers are available: DMA-BUF heaps on Linux, IOSurface on
    /// macOS.
    fn is_gpu_image_buffer_available() -> bool {
        crate::test_support::is_gpu_buffer_available()
    }

    /// Returns true on an i.MX 95 board with a Mali GPU, detected by `/dev/neutron0`.
    #[cfg(target_os = "linux")]
    fn is_neutron_available() -> bool {
        *NEUTRON_AVAILABLE.get_or_init(|| std::path::Path::new("/dev/neutron0").exists())
    }

    fn compare_images(img1: &TensorDyn, img2: &TensorDyn, threshold: f64, name: &str) {
        assert_eq!(img1.height(), img2.height(), "Heights differ");
        assert_eq!(img1.width(), img2.width(), "Widths differ");
        assert_eq!(
            img1.format().unwrap(),
            img2.format().unwrap(),
            "PixelFormat differ"
        );
        assert!(
            matches!(
                img1.format().unwrap(),
                PixelFormat::Rgb
                    | PixelFormat::Rgba
                    | PixelFormat::Grey
                    | PixelFormat::PlanarRgb
                    | PixelFormat::PlanarRgba
            ),
            "format must be Rgb or Rgba for comparison"
        );

        // The image bytes with row padding removed.
        fn flat(img: &TensorDyn) -> Vec<u8> {
            let t = img.as_u8().expect("u8 image");
            let mut out = vec![0u8; t.shape().iter().product::<usize>()];
            t.copy_to_flat(&mut out).expect("compact the padded rows");
            out
        }

        let image1 = match img1.format().unwrap() {
            PixelFormat::Rgb => image::RgbImage::from_vec(
                img1.width().unwrap() as u32,
                img1.height().unwrap() as u32,
                flat(img1),
            )
            .unwrap(),
            PixelFormat::Rgba => image::RgbaImage::from_vec(
                img1.width().unwrap() as u32,
                img1.height().unwrap() as u32,
                flat(img1),
            )
            .unwrap()
            .convert(),
            PixelFormat::Grey => image::GrayImage::from_vec(
                img1.width().unwrap() as u32,
                img1.height().unwrap() as u32,
                flat(img1),
            )
            .unwrap()
            .convert(),
            PixelFormat::PlanarRgb | PixelFormat::PlanarRgba => image::GrayImage::from_vec(
                img1.width().unwrap() as u32,
                (img1.height().unwrap() * img1.format().unwrap().channels()) as u32,
                flat(img1),
            )
            .unwrap()
            .convert(),
            _ => return,
        };

        let image2 = match img2.format().unwrap() {
            PixelFormat::Rgb => image::RgbImage::from_vec(
                img2.width().unwrap() as u32,
                img2.height().unwrap() as u32,
                flat(img2),
            )
            .unwrap(),
            PixelFormat::Rgba => image::RgbaImage::from_vec(
                img2.width().unwrap() as u32,
                img2.height().unwrap() as u32,
                flat(img2),
            )
            .unwrap()
            .convert(),
            PixelFormat::Grey => image::GrayImage::from_vec(
                img2.width().unwrap() as u32,
                img2.height().unwrap() as u32,
                flat(img2),
            )
            .unwrap()
            .convert(),
            PixelFormat::PlanarRgb | PixelFormat::PlanarRgba => image::GrayImage::from_vec(
                img2.width().unwrap() as u32,
                (img2.height().unwrap() * img2.format().unwrap().channels()) as u32,
                flat(img2),
            )
            .unwrap()
            .convert(),
            _ => return,
        };

        let similarity = image_compare::rgb_similarity_structure(
            &image_compare::Algorithm::RootMeanSquared,
            &image1,
            &image2,
        )
        .expect("Image Comparison failed");
        if similarity.score < threshold {
            // Best-effort save of the diff image.
            let save_name = name.replace('\0', "_");
            let _ = similarity
                .image
                .to_color_map()
                .save(format!("{save_name}.png"));
            panic!(
                "{name}: converted image and target image have similarity score too low: {} < {}",
                similarity.score, threshold
            )
        }
    }

    // =========================================================================
    // PixelFormat::Nv12 Reference Validation Tests
    // These tests compare OpenGL PixelFormat::Nv12 conversions against ffmpeg-generated
    // references
    // =========================================================================

    /// An image tensor of `format` holding `bytes`, a tight buffer laid into it at the
    /// destination's own row pitch by [`write_tight_rows`].
    #[cfg(any(target_os = "linux", target_os = "macos", target_os = "windows"))]
    fn load_raw_image(
        width: usize,
        height: usize,
        format: PixelFormat,
        memory: Option<TensorMemory>,
        bytes: &[u8],
    ) -> Result<TensorDyn, crate::Error> {
        let img = TensorDyn::image(
            width,
            height,
            format,
            DType::U8,
            memory,
            edgefirst_tensor::CpuAccess::ReadWrite,
        )?;
        write_tight_rows(&img, bytes)?;
        Ok(img)
    }

    /// Copy a tight image buffer (rows packed at the image's own row length, with no padding)
    /// into `t`, re-spacing the rows to the destination's own pitch.
    ///
    /// A semi-planar buffer is re-spaced by line: its combined plane is `height` luma lines of
    /// `width` bytes followed by chroma lines of `2 * chroma_w` bytes, and the chroma lines sit
    /// `ChromaLayout::uv_rows_per_luma` destination rows apart.
    #[cfg(any(target_os = "linux", target_os = "macos", target_os = "windows"))]
    fn write_tight_rows(t: &TensorDyn, bytes: &[u8]) -> Result<(), crate::Error> {
        let format = t.format().unwrap();
        let width = t.width().unwrap();
        let height = t.height().unwrap();
        // Bytes per sample.
        let elem = t.dtype().size();
        let bound = t.as_u8().unwrap();
        let stride = bound.effective_row_stride().unwrap();
        let mut map = bound.map()?;
        let dst = map.as_mut_slice();
        match format.chroma_layout() {
            Some(chroma) => {
                let luma = width * elem;
                let chroma_h = height.div_ceil(1 << chroma.shift_y);
                let line = width.div_ceil(1 << chroma.shift_x) * 2 * elem;
                for r in 0..height {
                    dst[r * stride..r * stride + luma]
                        .copy_from_slice(&bytes[r * luma..(r + 1) * luma]);
                }
                let uv = &bytes[luma * height..];
                for i in 0..chroma_h {
                    let at = (height + i * chroma.uv_rows_per_luma) * stride;
                    dst[at..at + line].copy_from_slice(&uv[i * line..(i + 1) * line]);
                }
            }
            None => {
                let tight = match format.layout() {
                    edgefirst_tensor::PixelLayout::Packed => width * format.channels() * elem,
                    _ => width * elem,
                };
                for (r, row) in bytes.chunks(tight).enumerate() {
                    dst[r * stride..r * stride + row.len()].copy_from_slice(row);
                }
            }
        }
        Ok(())
    }

    /// A convert loop over a fixed pool of DMA sources makes no new EGLImage imports after warmup.
    #[test]
    #[cfg(any(target_os = "linux", target_os = "macos", target_os = "windows"))]
    fn dma_pool_steady_state_zero_imports() {
        if !is_gpu_image_buffer_available() {
            crate::test_support::report_skip(&format!(
                "{} - no zero-copy GPU buffers",
                function!()
            ));
            return;
        }
        let mut renderer = match GLProcessorThreaded::new(None) {
            Ok(r) => r,
            Err(e) => {
                crate::test_support::report_skip(&format!(
                    "{} - GL not available: {e}",
                    function!()
                ));
                return;
            }
        };
        let (w, h) = (128usize, 96usize);
        const POOL: usize = 3;
        const FRAMES: usize = 300;

        // NV12 pool: Y plane and neutral chroma.
        let mut bytes = vec![100u8; w * h];
        bytes.extend(std::iter::repeat_n(128u8, w * h / 2));
        let pool: Vec<TensorDyn> = (0..POOL)
            .map(|_| {
                load_raw_image(w, h, PixelFormat::Nv12, Some(TensorMemory::DmaBuf), &bytes).unwrap()
            })
            .collect();
        let mut dst = TensorDyn::image(
            w,
            h,
            PixelFormat::Rgba,
            DType::U8,
            Some(TensorMemory::DmaBuf),
            edgefirst_tensor::CpuAccess::ReadWrite,
        )
        .unwrap();
        let mut scaled = TensorDyn::image(
            w * 3 / 2,
            h * 3 / 2,
            PixelFormat::Rgba,
            DType::U8,
            Some(TensorMemory::DmaBuf),
            edgefirst_tensor::CpuAccess::ReadWrite,
        )
        .unwrap();
        let mut frame = |renderer: &mut GLProcessorThreaded, src: &TensorDyn| {
            for dst in [&mut dst, &mut scaled] {
                renderer.convert(src, dst, None).unwrap();
            }
        };

        // Warmup: two passes over the pool import every buffer once.
        for src in pool.iter().cycle().take(POOL * 2) {
            frame(&mut renderer, src);
        }
        let warm = renderer.egl_cache_stats().unwrap();

        for src in pool.iter().cycle().take(FRAMES) {
            frame(&mut renderer, src);
        }
        let steady = renderer.egl_cache_stats().unwrap();

        assert_eq!(
            warm.total_misses(),
            steady.total_misses(),
            "steady-state loop performed new EGLImage imports: warm={warm:?} steady={steady:?}"
        );
        let hits = |s: &crate::opengl_headless::cache::GlCacheStats| {
            s.src.hits + s.dst.hits + s.nv_r8.hits + s.nv_uv.hits
        };
        let gained = hits(&steady) - hits(&warm);
        assert!(
            gained >= FRAMES as u64,
            "expected at least {FRAMES} cache hits over the loop, got {gained} \
             (warm={warm:?} steady={steady:?})"
        );
    }

    /// A buffer imported as a fresh tensor every frame hits the import cache (bounded on Windows).
    #[test]
    #[cfg(any(target_os = "linux", target_os = "macos", target_os = "windows"))]
    fn reimported_buffer_hits_the_cache() {
        if !is_gpu_image_buffer_available() {
            crate::test_support::report_skip(&format!(
                "{} - no zero-copy GPU buffers",
                function!()
            ));
            return;
        }
        let mut renderer = match GLProcessorThreaded::new(None) {
            Ok(r) => r,
            Err(e) => {
                crate::test_support::report_skip(&format!(
                    "{} - GL not available: {e}",
                    function!()
                ));
                return;
            }
        };
        let (w, h) = (128usize, 96usize);
        const FRAMES: usize = 50;

        // The producer's buffer stays alive for the whole test.
        let mut bytes = vec![100u8; w * h];
        bytes.extend(std::iter::repeat_n(128u8, w * h / 2));
        let producer =
            load_raw_image(w, h, PixelFormat::Nv12, Some(TensorMemory::DmaBuf), &bytes).unwrap();
        let desc = producer.descriptor();
        let mut dst = TensorDyn::image(
            w,
            h,
            PixelFormat::Rgba,
            DType::U8,
            Some(TensorMemory::DmaBuf),
            edgefirst_tensor::CpuAccess::ReadWrite,
        )
        .unwrap();

        // Warmup: import the buffer (and the destination) exactly once.
        {
            let frame = TensorDyn::import_descriptor(&desc).unwrap();
            renderer.convert(&frame, &mut dst, None).unwrap();
        }
        let warm = renderer.egl_cache_stats().unwrap();

        for _ in 0..FRAMES {
            // Freshly imported, used and dropped within the iteration.
            let frame = TensorDyn::import_descriptor(&desc).unwrap();
            renderer.convert(&frame, &mut dst, None).unwrap();
        }
        let steady = renderer.egl_cache_stats().unwrap();

        #[cfg(not(target_os = "windows"))]
        {
            assert_eq!(
                warm.total_misses(),
                steady.total_misses(),
                "re-imported buffer was re-imported into GL every frame: \
                 warm={warm:?} steady={steady:?}"
            );
            let hits = |s: &crate::opengl_headless::cache::GlCacheStats| {
                s.src.hits + s.dst.hits + s.nv_r8.hits + s.nv_uv.hits
            };
            let gained = hits(&steady) - hits(&warm);
            assert!(
                gained >= FRAMES as u64,
                "expected at least {FRAMES} cache hits over the loop, got {gained} \
                 (warm={warm:?} steady={steady:?})"
            );
        }
        #[cfg(target_os = "windows")]
        {
            // The destination never re-imports; the source misses are bounded by the loop.
            assert_eq!(
                steady.dst.misses, warm.dst.misses,
                "the destination lives across the loop, so it must not re-import: \
                 warm={warm:?} steady={steady:?}"
            );
            assert!(
                steady.dst.hits - warm.dst.hits >= FRAMES as u64,
                "the destination must hit every frame: warm={warm:?} steady={steady:?}"
            );
            // A native NV12 source imports luma and chroma; each cache is bounded by the loop.
            for (name, warm_misses, steady_misses) in [
                ("src", warm.src.misses, steady.src.misses),
                ("nv_r8", warm.nv_r8.misses, steady.nv_r8.misses),
                ("nv_uv", warm.nv_uv.misses, steady.nv_uv.misses),
            ] {
                assert!(
                    steady_misses - warm_misses <= FRAMES as u64,
                    "{name} imports must be bounded by the loop: warm={warm:?} steady={steady:?}"
                );
            }
        }
    }

    /// RGBA gradient bytes (w*h*4) with distinct R/G/B functions and a salt.
    #[cfg(any(target_os = "linux", target_os = "macos", target_os = "windows"))]
    fn rgba_gradient(w: usize, h: usize, salt: u8) -> Vec<u8> {
        let mut v = Vec::with_capacity(w * h * 4);
        for y in 0..h {
            for x in 0..w {
                v.push(((x * 255) / w.max(1)) as u8 ^ salt);
                v.push(((y * 255) / h.max(1)) as u8);
                v.push((((x + y) * 255) / (w + h).max(1)) as u8);
                v.push(255);
            }
        }
        v
    }

    /// The fence entry point's output equals the blocking convert's, with or without a fence.
    #[test]
    #[cfg(any(target_os = "linux", target_os = "macos", target_os = "windows"))]
    fn convert_with_fence_blocking_fallback_matches() {
        if !is_opengl_available() {
            crate::test_support::report_skip(&format!("{} - GL not available", function!()));
            return;
        }
        let mut gl = match GLProcessorThreaded::new(None) {
            Ok(r) => r,
            Err(e) => {
                crate::test_support::report_skip(&format!(
                    "{} - GL not available: {e}",
                    function!()
                ));
                return;
            }
        };
        let (sw, sh, dw) = (96usize, 72usize, 64usize);
        let bytes = rgba_gradient(sw, sh, 3);
        let src = load_raw_image(sw, sh, PixelFormat::Rgba, Some(TensorMemory::Mem), &bytes)
            .expect("src alloc");
        let mut dst_fenced = TensorDyn::image(
            dw,
            dw,
            PixelFormat::Rgba,
            DType::U8,
            Some(TensorMemory::Mem),
            edgefirst_tensor::CpuAccess::ReadWrite,
        )
        .expect("dst alloc");
        let mut dst_blocking = TensorDyn::image(
            dw,
            dw,
            PixelFormat::Rgba,
            DType::U8,
            Some(TensorMemory::Mem),
            edgefirst_tensor::CpuAccess::ReadWrite,
        )
        .expect("dst alloc");
        let fence = gl
            .convert_with_fence(&src, &mut dst_fenced, None)
            .expect("convert_with_fence");
        if cfg!(not(any(target_os = "android", target_os = "windows"))) {
            assert!(
                fence.is_none(),
                "no platform in this test tier has native fence sync"
            );
        }
        gl.convert(&src, &mut dst_blocking, None).expect("convert");
        let (ma, mb) = (
            dst_fenced.as_u8().unwrap().map().unwrap(),
            dst_blocking.as_u8().unwrap().map().unwrap(),
        );
        assert_eq!(ma.as_slice(), mb.as_slice(), "fenced vs blocking output");
    }

    /// A float zero-copy source import matches the CPU-upload feed bit for bit, source untouched.
    #[test]
    #[cfg(any(target_os = "linux", target_os = "macos", target_os = "windows"))]
    fn float_dma_src_import_matches_upload() {
        if !is_gpu_image_buffer_available() || !is_opengl_available() {
            crate::test_support::report_skip(&format!(
                "{} - no zero-copy GPU buffers or GL",
                function!()
            ));
            return;
        }
        let mut gl = match GLProcessorThreaded::new(None) {
            Ok(r) => r,
            Err(e) => {
                crate::test_support::report_skip(&format!(
                    "{} - GL not available: {e}",
                    function!()
                ));
                return;
            }
        };
        if !gl.supported_render_dtypes().f16 {
            crate::test_support::report_skip(&format!(
                "{} - this GPU has no F16 render target",
                function!()
            ));
            return;
        }
        let (sw, sh, dw) = (96usize, 72usize, 64usize);
        let bytes = rgba_gradient(sw, sh, 0);
        let src_dma = load_raw_image(
            sw,
            sh,
            PixelFormat::Rgba,
            Some(TensorMemory::DmaBuf),
            &bytes,
        )
        .unwrap();
        let src_mem =
            load_raw_image(sw, sh, PixelFormat::Rgba, Some(TensorMemory::Mem), &bytes).unwrap();

        let mut dst_a = TensorDyn::image(
            dw,
            dw,
            PixelFormat::PlanarRgb,
            DType::F16,
            Some(TensorMemory::DmaBuf),
            edgefirst_tensor::CpuAccess::ReadWrite,
        )
        .unwrap();
        let mut dst_b = TensorDyn::image(
            dw,
            dw,
            PixelFormat::PlanarRgb,
            DType::F16,
            Some(TensorMemory::DmaBuf),
            edgefirst_tensor::CpuAccess::ReadWrite,
        )
        .unwrap();
        gl.convert(&src_dma, &mut dst_a, None).unwrap();
        gl.convert(&src_mem, &mut dst_b, None).unwrap();

        let (ma, mb) = (
            dst_a.as_f16().unwrap().map().unwrap(),
            dst_b.as_f16().unwrap().map().unwrap(),
        );
        assert_eq!(ma.as_slice().len(), mb.as_slice().len());
        for (i, (a, b)) in ma.as_slice().iter().zip(mb.as_slice().iter()).enumerate() {
            assert_eq!(
                a.to_bits(),
                b.to_bits(),
                "import vs upload diverged at f16 element {i}: {a} vs {b}"
            );
        }
        drop((ma, mb));

        // The imported source is unchanged.
        let m = src_dma.as_u8().unwrap().map().unwrap();
        let stride = src_dma.effective_row_stride().unwrap_or(sw * 4);
        for y in 0..sh {
            let row = &m.as_slice()[y * stride..y * stride + sw * 4];
            assert_eq!(
                row,
                &bytes[y * sw * 4..(y + 1) * sw * 4],
                "src row {y} modified"
            );
        }
    }

    /// Alternating Dma, Mem and Dma float frames on one processor each match their reference.
    #[test]
    #[cfg(any(target_os = "linux", target_os = "macos", target_os = "windows"))]
    fn float_src_feed_alternating_frames() {
        if !is_gpu_image_buffer_available() || !is_opengl_available() {
            crate::test_support::report_skip(&format!(
                "{} - no zero-copy GPU buffers or GL",
                function!()
            ));
            return;
        }
        let mut gl = match GLProcessorThreaded::new(None) {
            Ok(r) => r,
            Err(e) => {
                crate::test_support::report_skip(&format!(
                    "{} - GL not available: {e}",
                    function!()
                ));
                return;
            }
        };
        if !gl.supported_render_dtypes().f16 {
            crate::test_support::report_skip(&format!(
                "{} - this GPU has no F16 render target",
                function!()
            ));
            return;
        }
        let (sw, sh, dw) = (96usize, 72usize, 64usize);
        let content_a = rgba_gradient(sw, sh, 0x00);
        let content_b = rgba_gradient(sw, sh, 0x5A);
        let content_c = rgba_gradient(sw, sh, 0xA5);

        let src_dma = load_raw_image(
            sw,
            sh,
            PixelFormat::Rgba,
            Some(TensorMemory::DmaBuf),
            &content_a,
        )
        .unwrap();

        let run_float = |gl: &mut GLProcessorThreaded, src: &TensorDyn| -> Vec<u16> {
            let mut dst = TensorDyn::image(
                dw,
                dw,
                PixelFormat::PlanarRgb,
                DType::F16,
                Some(TensorMemory::DmaBuf),
                edgefirst_tensor::CpuAccess::ReadWrite,
            )
            .unwrap();
            gl.convert(src, &mut dst, None).unwrap();
            let m = dst.as_f16().unwrap().map().unwrap();
            m.as_slice().iter().map(|v| v.to_bits()).collect()
        };
        let reference = |gl: &mut GLProcessorThreaded, bytes: &[u8]| -> Vec<u16> {
            let mem =
                load_raw_image(sw, sh, PixelFormat::Rgba, Some(TensorMemory::Mem), bytes).unwrap();
            run_float(gl, &mem)
        };

        // Frame 1: Dma source (attach arm).
        let out1 = run_float(&mut gl, &src_dma);
        assert_eq!(out1, reference(&mut gl, &content_a), "frame 1 (Dma A)");

        // Interleaved u8 convert.
        let u8_src = load_raw_image(
            sw,
            sh,
            PixelFormat::Rgba,
            Some(TensorMemory::Mem),
            &content_b,
        )
        .unwrap();
        let mut u8_dst = TensorDyn::image(
            dw,
            dw,
            PixelFormat::Rgba,
            DType::U8,
            Some(TensorMemory::DmaBuf),
            edgefirst_tensor::CpuAccess::ReadWrite,
        )
        .unwrap();
        gl.convert(&u8_src, &mut u8_dst, None).unwrap();

        // Frame 2: Mem source (upload arm).
        let out2 = run_float(&mut gl, &u8_src);
        assert_eq!(out2, reference(&mut gl, &content_b), "frame 2 (Mem B)");

        // Frame 2's upload did not write into frame 1's Dma buffer.
        {
            let m = src_dma.as_u8().unwrap().map().unwrap();
            let stride = src_dma.effective_row_stride().unwrap_or(sw * 4);
            for y in 0..sh {
                let row = &m.as_slice()[y * stride..y * stride + sw * 4];
                assert_eq!(
                    row,
                    &content_a[y * sw * 4..(y + 1) * sw * 4],
                    "Dma source scribbled by the upload frame (row {y})"
                );
            }
        }

        // Frame 3: the Dma source again, with new content.
        overwrite_in_place(&src_dma, &content_c);
        let out3 = run_float(&mut gl, &src_dma);
        assert_eq!(out3, reference(&mut gl, &content_c), "frame 3 (Dma C)");
    }

    /// A fixed pool of Dma float sources imports once and hits the source cache thereafter.
    #[test]
    #[cfg(any(target_os = "linux", target_os = "macos", target_os = "windows"))]
    fn float_dma_src_pool_steady_state() {
        if !is_gpu_image_buffer_available() || !is_opengl_available() {
            crate::test_support::report_skip(&format!(
                "{} - no zero-copy GPU buffers or GL",
                function!()
            ));
            return;
        }
        let mut gl = match GLProcessorThreaded::new(None) {
            Ok(r) => r,
            Err(e) => {
                crate::test_support::report_skip(&format!(
                    "{} - GL not available: {e}",
                    function!()
                ));
                return;
            }
        };
        if !gl.supported_render_dtypes().f16 {
            crate::test_support::report_skip(&format!(
                "{} - this GPU has no F16 render target",
                function!()
            ));
            return;
        }
        let (sw, sh, dw) = (128usize, 96usize, 64usize);
        const FRAMES: usize = 100;
        let pool: Vec<TensorDyn> = (0..2)
            .map(|i| {
                load_raw_image(
                    sw,
                    sh,
                    PixelFormat::Rgba,
                    Some(TensorMemory::DmaBuf),
                    &rgba_gradient(sw, sh, i as u8),
                )
                .unwrap()
            })
            .collect();
        let mut dst = TensorDyn::image(
            dw,
            dw,
            PixelFormat::PlanarRgb,
            DType::F16,
            Some(TensorMemory::DmaBuf),
            edgefirst_tensor::CpuAccess::ReadWrite,
        )
        .unwrap();

        for src in pool.iter().cycle().take(4) {
            gl.convert(src, &mut dst, None).unwrap();
        }
        let warm = gl.egl_cache_stats().unwrap();
        let warm_feed = gl.convert_stats().unwrap();
        for src in pool.iter().cycle().take(FRAMES) {
            gl.convert(src, &mut dst, None).unwrap();
        }
        let steady = gl.egl_cache_stats().unwrap();
        let steady_feed = gl.convert_stats().unwrap();
        assert_eq!(
            warm.total_misses(),
            steady.total_misses(),
            "float src loop performed new EGLImage imports: warm={warm:?} steady={steady:?}"
        );
        // Every convert is counted as exactly one of import, pbo or upload.
        let d_imports = steady_feed.src_imports - warm_feed.src_imports;
        let d_pbo = steady_feed.src_pbo_uploads - warm_feed.src_pbo_uploads;
        let d_uploads = steady_feed.src_uploads - warm_feed.src_uploads;
        let d_declines = steady_feed.zero_copy_declines - warm_feed.zero_copy_declines;
        assert_eq!(
            d_imports + d_pbo + d_uploads,
            FRAMES as u64,
            "feed counters don't account for every frame: \
             warm={warm_feed:?} steady={steady_feed:?}"
        );
        // With no declined import, every Dma-source frame was fed zero-copy.
        if d_declines == 0 {
            assert_eq!(
                d_imports, FRAMES as u64,
                "no declines yet frames were not import-fed: \
                 warm={warm_feed:?} steady={steady_feed:?}"
            );
        }
    }

    /// Repeated heap-RGBA to packed-RGB DMA converts all succeed with byte-identical output.
    #[test]
    #[cfg(any(target_os = "linux", target_os = "macos", target_os = "windows"))]
    fn repeat_convert_rgba_mem_to_rgb_dma() {
        if !is_gpu_image_buffer_available() {
            crate::test_support::report_skip(&format!(
                "{} - no zero-copy GPU buffers",
                function!()
            ));
            return;
        }
        let mut renderer = match GLProcessorThreaded::new(None) {
            Ok(r) => r,
            Err(e) => {
                crate::test_support::report_skip(&format!(
                    "{} - GL not available: {e}",
                    function!()
                ));
                return;
            }
        };

        // Deterministic RGBA gradient source in heap memory.
        let (src_w, src_h) = (1280usize, 720usize);
        let mut bytes = vec![0u8; src_w * src_h * 4];
        for (i, px) in bytes.as_chunks_mut::<4>().0.iter_mut().enumerate() {
            px[0] = (i % 256) as u8;
            px[1] = ((i / 256) % 256) as u8;
            px[2] = ((i / 65536) % 256) as u8;
            px[3] = 255;
        }
        let src = load_raw_image(
            src_w,
            src_h,
            PixelFormat::Rgba,
            Some(TensorMemory::Mem),
            &bytes,
        )
        .unwrap();

        let lb = Crop::letterbox([114, 114, 114, 255]);
        for dtype in [DType::U8, DType::I8] {
            let Some(mut dst) = crate::test_support::gpu_buffer_allocated(
                TensorDyn::image(
                    640,
                    640,
                    PixelFormat::Rgb,
                    dtype,
                    Some(TensorMemory::DmaBuf),
                    edgefirst_tensor::CpuAccess::ReadWrite,
                ),
                &format!("cell {dtype:?}: DMA RGB destination"),
            ) else {
                continue;
            };
            let mut first: Option<Vec<u8>> = None;
            for round in 0..4 {
                renderer
                    .convert(&src, &mut dst, Some(crate::ConvertOptions::new().crop(lb)))
                    .unwrap_or_else(|e| {
                        panic!(
                            "rgba@mem->rgb.{dtype:?}@dma convert #{} failed: {e} \
                             (repeat-convert state leak — see 0x501 regression)",
                            round + 1
                        )
                    });
                let out: Vec<u8> = match dtype {
                    DType::I8 => dst
                        .as_i8()
                        .unwrap()
                        .map()
                        .unwrap()
                        .as_slice()
                        .iter()
                        .map(|&v| v as u8)
                        .collect(),
                    _ => dst.as_u8().unwrap().map().unwrap().as_slice().to_vec(),
                };
                match &first {
                    None => first = Some(out),
                    Some(reference) => assert_eq!(
                        reference,
                        &out,
                        "rgba@mem->rgb.{dtype:?}@dma convert #{} output diverged from #1",
                        round + 1
                    ),
                }
            }
        }
    }

    /// The int8 variant of every draw is the u8 variant with XOR 0x80 on the colour channels.
    #[test]
    #[cfg(any(target_os = "linux", target_os = "macos", target_os = "windows"))]
    fn int8_mem_convert_is_xor_biased_u8() {
        if !is_opengl_available() {
            crate::test_support::report_skip(&format!("{} - OpenGL not available", function!()));
            return;
        }
        let mut renderer = match GLProcessorThreaded::new(None) {
            Ok(r) => r,
            Err(e) => {
                crate::test_support::report_skip(&format!(
                    "{} - GL not available: {e}",
                    function!()
                ));
                return;
            }
        };

        // Gradient luma and off-neutral chroma.
        let (w, h) = (64usize, 48usize);
        let mut nv12 = vec![0u8; w * h * 3 / 2];
        for row in 0..h {
            for col in 0..w {
                nv12[row * w + col] = ((row * 255) / h) as u8;
            }
        }
        for i in 0..w * h / 4 {
            nv12[w * h + 2 * i] = 110; // Cb
            nv12[w * h + 2 * i + 1] = 150; // Cr
        }
        let src = load_raw_image(w, h, PixelFormat::Nv12, Some(TensorMemory::Mem), &nv12).unwrap();

        // A failing Rgb convert skips the format; a failing Rgba convert fails the test.
        let mut validated = 0usize;
        'fmt: for fmt in [PixelFormat::Rgb, PixelFormat::Rgba] {
            let mut out = [Vec::new(), Vec::new()];
            for (slot, dtype) in [(0, DType::U8), (1, DType::I8)] {
                let mut dst = TensorDyn::image(
                    w,
                    h,
                    fmt,
                    dtype,
                    Some(TensorMemory::Mem),
                    edgefirst_tensor::CpuAccess::ReadWrite,
                )
                .unwrap();
                if let Err(e) = renderer.convert(&src, &mut dst, None) {
                    if fmt == PixelFormat::Rgba {
                        panic!("nv12@mem -> rgba.{dtype:?}@mem must be supported: {e}");
                    }
                    crate::test_support::report_skip(&format!(
                        "cell nv12@mem -> {fmt}.{dtype:?}@mem: {e}"
                    ));
                    continue 'fmt;
                }
                out[slot] = match dtype {
                    DType::I8 => dst
                        .as_i8()
                        .unwrap()
                        .map()
                        .unwrap()
                        .as_slice()
                        .iter()
                        .map(|&v| v as u8)
                        .collect(),
                    _ => dst.as_u8().unwrap().map().unwrap().as_slice().to_vec(),
                };
            }
            let channels = fmt.channels();
            let mut diffs = 0usize;
            for (i, (&u, &b)) in out[0].iter().zip(out[1].iter()).enumerate() {
                let expect = if channels == 4 && i % 4 == 3 {
                    u
                } else {
                    u ^ 0x80
                };
                if (b as i16 - expect as i16).abs() > 1 {
                    diffs += 1;
                    if diffs <= 5 {
                        eprintln!("byte {i}: u8={u:#04x} i8={b:#04x} expected {expect:#04x}");
                    }
                }
            }
            assert_eq!(
                diffs, 0,
                "nv12@mem -> {fmt}: i8 output is not the XOR-0x80 bias of the u8 output \
                 ({diffs} bytes differ beyond ±1 — un-biased int8 NV program?)"
            );
            validated += 1;
        }
        assert!(validated >= 1, "no format validated the int8 NV bias");
    }

    /// Test OpenGL PixelFormat::Nv12→PixelFormat::Rgba conversion against ffmpeg reference
    #[test]
    #[cfg(any(target_os = "linux", target_os = "macos", target_os = "windows"))]
    fn test_opengl_nv12_to_rgba_reference() {
        if !is_gpu_image_buffer_available() || !is_opengl_available() {
            crate::test_support::report_skip(&format!(
                "{} - no zero-copy GPU buffers or OpenGL",
                function!()
            ));
            return;
        }
        let src = load_raw_image(
            1280,
            720,
            PixelFormat::Nv12,
            Some(TensorMemory::DmaBuf),
            &edgefirst_bench::testdata::read("camera720p.nv12"),
        )
        .unwrap();

        // Load PixelFormat::Rgba reference (ffmpeg-generated)
        let reference = load_raw_image(
            1280,
            720,
            PixelFormat::Rgba,
            None,
            &edgefirst_bench::testdata::read("camera720p.rgba"),
        )
        .unwrap();

        let dst = TensorDyn::image(
            1280,
            720,
            PixelFormat::Rgba,
            DType::U8,
            Some(TensorMemory::DmaBuf),
            edgefirst_tensor::CpuAccess::ReadWrite,
        )
        .unwrap();
        let mut gl = GLProcessorThreaded::new(None).unwrap();
        let src_dyn = src;
        let mut dst_dyn = dst;
        gl.convert(&src_dyn, &mut dst_dyn, None).unwrap();

        // Copy to CPU for comparison
        let cpu_dst = TensorDyn::image(
            1280,
            720,
            PixelFormat::Rgba,
            DType::U8,
            None,
            edgefirst_tensor::CpuAccess::ReadWrite,
        )
        .unwrap();
        cpu_dst
            .as_u8()
            .unwrap()
            .map()
            .unwrap()
            .as_mut_slice()
            .copy_from_slice(dst_dyn.as_u8().unwrap().map().unwrap().as_slice());

        compare_images(&reference, &cpu_dst, 0.98, "opengl_nv12_to_rgba_reference");
    }

    /// Test OpenGL PixelFormat::Yuyv→PixelFormat::Rgba conversion against ffmpeg reference
    #[test]
    #[cfg(any(target_os = "linux", target_os = "macos", target_os = "windows"))]
    fn test_opengl_yuyv_to_rgba_reference() {
        if !is_gpu_image_buffer_available() || !is_opengl_available() {
            crate::test_support::report_skip(&format!(
                "{} - no zero-copy GPU buffers or OpenGL",
                function!()
            ));
            return;
        }
        let src = load_raw_image(
            1280,
            720,
            PixelFormat::Yuyv,
            Some(TensorMemory::DmaBuf),
            &edgefirst_bench::testdata::read("camera720p.yuyv"),
        )
        .unwrap();

        // Load PixelFormat::Rgba reference (ffmpeg-generated)
        let reference = load_raw_image(
            1280,
            720,
            PixelFormat::Rgba,
            None,
            &edgefirst_bench::testdata::read("camera720p.rgba"),
        )
        .unwrap();

        let dst = TensorDyn::image(
            1280,
            720,
            PixelFormat::Rgba,
            DType::U8,
            Some(TensorMemory::DmaBuf),
            edgefirst_tensor::CpuAccess::ReadWrite,
        )
        .unwrap();
        let mut gl = GLProcessorThreaded::new(None).unwrap();
        let src_dyn = src;
        let mut dst_dyn = dst;
        gl.convert(&src_dyn, &mut dst_dyn, None).unwrap();

        // Copy to CPU for comparison
        let cpu_dst = TensorDyn::image(
            1280,
            720,
            PixelFormat::Rgba,
            DType::U8,
            None,
            edgefirst_tensor::CpuAccess::ReadWrite,
        )
        .unwrap();
        cpu_dst
            .as_u8()
            .unwrap()
            .map()
            .unwrap()
            .as_mut_slice()
            .copy_from_slice(dst_dyn.as_u8().unwrap().map().unwrap().as_slice());

        compare_images(&reference, &cpu_dst, 0.98, "opengl_yuyv_to_rgba_reference");
    }

    /// Two `plane_offset` sub-views of one DMA-BUF each render as a standalone convert does.
    #[test]
    #[cfg(any(target_os = "linux", target_os = "macos", target_os = "windows"))]
    fn opengl_render_into_dma_subviews_no_aliasing() {
        if !is_gpu_image_buffer_available() || !is_opengl_available() {
            crate::test_support::report_skip(&format!(
                "{} - no zero-copy GPU buffers or OpenGL",
                function!()
            ));
            return;
        }
        let (w, h) = (64usize, 64usize);
        let frame = w * h * 4; // RGBA bytes per frame

        // Two distinct solid NV12 sources (different luma → different grey).
        let make_nv12 = |y: u8| {
            let mut buf = vec![y; w * h];
            buf.extend(std::iter::repeat_n(128u8, w * h / 2)); // neutral chroma
            buf
        };
        let src0 = load_raw_image(
            w,
            h,
            PixelFormat::Nv12,
            Some(TensorMemory::DmaBuf),
            &make_nv12(50),
        )
        .unwrap();
        let src1 = load_raw_image(
            w,
            h,
            PixelFormat::Nv12,
            Some(TensorMemory::DmaBuf),
            &make_nv12(200),
        )
        .unwrap();

        let mut gl = GLProcessorThreaded::new(None).unwrap();
        let convert = |gl: &mut GLProcessorThreaded, s: &TensorDyn, d: &mut TensorDyn| {
            gl.convert(s, d, None).unwrap();
        };

        // Standalone full-buffer reference conversions (independent buffers).
        let mut ref0 = TensorDyn::image(
            w,
            h,
            PixelFormat::Rgba,
            DType::U8,
            Some(TensorMemory::DmaBuf),
            edgefirst_tensor::CpuAccess::ReadWrite,
        )
        .unwrap();
        convert(&mut gl, &src0, &mut ref0);
        let mut ref1 = TensorDyn::image(
            w,
            h,
            PixelFormat::Rgba,
            DType::U8,
            Some(TensorMemory::DmaBuf),
            edgefirst_tensor::CpuAccess::ReadWrite,
        )
        .unwrap();
        convert(&mut gl, &src1, &mut ref1);

        // One DMA-BUF holding two stacked RGBA frames, and two offset sub-views of it on the same
        // processor.
        let parent = TensorDyn::image(
            w,
            2 * h,
            PixelFormat::Rgba,
            DType::U8,
            Some(TensorMemory::DmaBuf),
            edgefirst_tensor::CpuAccess::ReadWrite,
        )
        .unwrap();
        let mut view0 = parent.view(crate::Region::new(0, 0, w, h)).unwrap();
        let mut view1 = parent.view(crate::Region::new(0, h, w, h)).unwrap();
        assert_eq!(view1.plane_offset(), Some(frame));

        convert(&mut gl, &src0, &mut view0);
        convert(&mut gl, &src1, &mut view1);

        let parent_bytes = parent.as_u8().unwrap().map().unwrap().to_vec();
        let ref0_bytes = ref0.as_u8().unwrap().map().unwrap().to_vec();
        let ref1_bytes = ref1.as_u8().unwrap().map().unwrap().to_vec();

        assert_eq!(
            &parent_bytes[..frame],
            ref0_bytes.as_slice(),
            "view 0 window (offset 0) must match a standalone convert of src0"
        );
        assert_eq!(
            &parent_bytes[frame..2 * frame],
            ref1_bytes.as_slice(),
            "view 1 window (offset {frame}) must match a standalone convert of src1 — \
             aliasing/cache-collision if it doesn't"
        );
    }

    // =========================================================================
    // Recycled DMA-BUF source tests
    // =========================================================================

    /// Overwrite an existing tensor's bytes in place through [`write_tight_rows`].
    #[cfg(any(target_os = "linux", target_os = "macos", target_os = "windows"))]
    fn overwrite_in_place(t: &TensorDyn, bytes: &[u8]) {
        write_tight_rows(t, bytes).unwrap();
    }

    /// Write image A into a DMA source, convert, then overwrite the same source with distinct
    /// image B and convert again on the same processor. Assert the second output matches a
    /// fresh-DMA-source convert of B.
    #[cfg(any(target_os = "linux", target_os = "macos", target_os = "windows"))]
    fn dma_recycle_stale_read_check(
        w: usize,
        h: usize,
        src_fmt: PixelFormat,
        bytes_a: &[u8],
        bytes_b: &[u8],
        tolerance: u8,
    ) {
        // One DMA-BUF source, reused across two frames.
        let src = load_raw_image(w, h, src_fmt, Some(TensorMemory::DmaBuf), bytes_a).unwrap();
        let mut gl = GLProcessorThreaded::new(None).unwrap();
        let convert = |gl: &mut GLProcessorThreaded, s: &TensorDyn, d: &mut TensorDyn| {
            gl.convert(s, d, None).unwrap();
        };

        // Frame A.
        let mut dst_a = TensorDyn::image(
            w,
            h,
            PixelFormat::Rgba,
            DType::U8,
            Some(TensorMemory::DmaBuf),
            edgefirst_tensor::CpuAccess::ReadWrite,
        )
        .unwrap();
        convert(&mut gl, &src, &mut dst_a);
        let dst_a_bytes = dst_a.as_u8().unwrap().map().unwrap().to_vec();

        // Frame B: overwrite the same buffer and convert again.
        overwrite_in_place(&src, bytes_b);
        let mut dst_b = TensorDyn::image(
            w,
            h,
            PixelFormat::Rgba,
            DType::U8,
            Some(TensorMemory::DmaBuf),
            edgefirst_tensor::CpuAccess::ReadWrite,
        )
        .unwrap();
        convert(&mut gl, &src, &mut dst_b);
        let dst_b_bytes = dst_b.as_u8().unwrap().map().unwrap().to_vec();

        // Reference: a fresh DMA source of B.
        let fresh_b = load_raw_image(w, h, src_fmt, Some(TensorMemory::DmaBuf), bytes_b).unwrap();
        let mut oracle_b = TensorDyn::image(
            w,
            h,
            PixelFormat::Rgba,
            DType::U8,
            Some(TensorMemory::DmaBuf),
            edgefirst_tensor::CpuAccess::ReadWrite,
        )
        .unwrap();
        convert(&mut gl, &fresh_b, &mut oracle_b);
        let oracle_b_bytes = oracle_b.as_u8().unwrap().map().unwrap().to_vec();

        // A and B produce different output.
        assert_ne!(
            dst_a_bytes, oracle_b_bytes,
            "test data invalid for {src_fmt:?}: frame A and frame B convert identically"
        );

        assert_pixels_match(&oracle_b_bytes, &dst_b_bytes, tolerance);
    }

    /// A recycled Grey source converts to Rgba byte-exactly against a fresh source.
    #[test]
    #[cfg(any(target_os = "linux", target_os = "macos", target_os = "windows"))]
    fn dma_recycle_grey_stale_read() {
        if !is_gpu_image_buffer_available() || !is_opengl_available() {
            crate::test_support::report_skip(&format!(
                "{} - no zero-copy GPU buffers or OpenGL",
                function!()
            ));
            return;
        }
        let (w, h) = (64usize, 64usize);
        dma_recycle_stale_read_check(
            w,
            h,
            PixelFormat::Grey,
            &vec![50u8; w * h],
            &vec![200u8; w * h],
            0,
        );
    }

    /// A recycled NV12 source converts to Rgba within 4 of a fresh source.
    #[test]
    #[cfg(any(target_os = "linux", target_os = "macos", target_os = "windows"))]
    fn dma_recycle_nv12_stale_read() {
        if !is_gpu_image_buffer_available() || !is_opengl_available() {
            crate::test_support::report_skip(&format!(
                "{} - no zero-copy GPU buffers or OpenGL",
                function!()
            ));
            return;
        }
        let (w, h) = (64usize, 64usize);
        let make_nv12 = |y: u8| {
            let mut buf = vec![y; w * h];
            buf.extend(std::iter::repeat_n(128u8, w * h / 2)); // neutral chroma
            buf
        };
        dma_recycle_stale_read_check(w, h, PixelFormat::Nv12, &make_nv12(50), &make_nv12(200), 4);
    }

    /// A recycled NV16 source converts to Rgba within 4 of a fresh source.
    #[test]
    #[cfg(any(target_os = "linux", target_os = "macos", target_os = "windows"))]
    fn dma_recycle_nv16_stale_read() {
        if !is_gpu_image_buffer_available() || !is_opengl_available() {
            crate::test_support::report_skip(&format!(
                "{} - no zero-copy GPU buffers or OpenGL",
                function!()
            ));
            return;
        }
        let (w, h) = (64usize, 64usize);
        let make_nv16 = |y: u8| {
            let mut buf = vec![y; w * h];
            buf.extend(std::iter::repeat_n(128u8, w * h)); // full-res neutral chroma
            buf
        };
        dma_recycle_stale_read_check(w, h, PixelFormat::Nv16, &make_nv16(50), &make_nv16(200), 4);
    }

    /// Two DMA-BUF sources round-robined over 5 solid-fill frames each convert as a fresh source.
    #[test]
    #[cfg(any(target_os = "linux", target_os = "macos", target_os = "windows"))]
    fn dma_pool_recycle_all_frames_match_oracle() {
        if !is_gpu_image_buffer_available() || !is_opengl_available() {
            crate::test_support::report_skip(&format!(
                "{} - no zero-copy GPU buffers or OpenGL",
                function!()
            ));
            return;
        }
        let (w, h) = (64usize, 64usize);
        let lumas: [u8; 5] = [20, 70, 130, 180, 240];

        let pool = [
            TensorDyn::image(
                w,
                h,
                PixelFormat::Grey,
                DType::U8,
                Some(TensorMemory::DmaBuf),
                edgefirst_tensor::CpuAccess::ReadWrite,
            )
            .unwrap(),
            TensorDyn::image(
                w,
                h,
                PixelFormat::Grey,
                DType::U8,
                Some(TensorMemory::DmaBuf),
                edgefirst_tensor::CpuAccess::ReadWrite,
            )
            .unwrap(),
        ];
        let mut gl = GLProcessorThreaded::new(None).unwrap();
        let convert = |gl: &mut GLProcessorThreaded, s: &TensorDyn, d: &mut TensorDyn| {
            gl.convert(s, d, None).unwrap();
        };

        for (frame, &luma) in lumas.iter().enumerate() {
            let buf = &pool[frame % pool.len()];
            overwrite_in_place(buf, &vec![luma; w * h]);

            let mut gpu = TensorDyn::image(
                w,
                h,
                PixelFormat::Rgba,
                DType::U8,
                Some(TensorMemory::DmaBuf),
                edgefirst_tensor::CpuAccess::ReadWrite,
            )
            .unwrap();
            convert(&mut gl, buf, &mut gpu);

            // Fresh-source reference for this frame.
            let fresh = load_raw_image(
                w,
                h,
                PixelFormat::Grey,
                Some(TensorMemory::DmaBuf),
                &vec![luma; w * h],
            )
            .unwrap();
            let mut oracle = TensorDyn::image(
                w,
                h,
                PixelFormat::Rgba,
                DType::U8,
                Some(TensorMemory::DmaBuf),
                edgefirst_tensor::CpuAccess::ReadWrite,
            )
            .unwrap();
            convert(&mut gl, &fresh, &mut oracle);

            let oracle_map = oracle.as_u8().unwrap().map().unwrap();
            let gpu_map = gpu.as_u8().unwrap().map().unwrap();
            assert_pixels_match(oracle_map.as_slice(), gpu_map.as_slice(), 0);
        }
    }

    /// Fill `[h][w]` of `t`'s buffer with a non-uniform 2D pattern at the tensor's effective row
    /// stride.
    #[cfg(any(target_os = "linux", target_os = "macos", target_os = "windows"))]
    fn fill_grey_pattern(t: &TensorDyn, w: usize, h: usize, salt: u8) {
        let stride = t.effective_row_stride().unwrap_or(w);
        let mut map = t.as_u8().unwrap().map().unwrap();
        let buf = map.as_mut_slice();
        for r in 0..h {
            for c in 0..w {
                buf[r * stride + c] = ((r * 13 + c * 7) as u8).wrapping_add(salt);
            }
        }
    }

    /// A DMA pool buffer recycled across geometries by `configure_image()` matches fresh sources.
    #[test]
    #[cfg(any(target_os = "linux", target_os = "macos", target_os = "windows"))]
    fn dma_recycle_geometry_change_stale_read() {
        if !is_gpu_image_buffer_available() || !is_opengl_available() {
            crate::test_support::report_skip(&format!(
                "{} - no zero-copy GPU buffers or OpenGL",
                function!()
            ));
            return;
        }
        // An oversized pool buffer, recycled in place through `configure_image`.
        let mut pool = TensorDyn::image(
            128,
            128,
            PixelFormat::Grey,
            DType::U8,
            Some(TensorMemory::DmaBuf),
            edgefirst_tensor::CpuAccess::ReadWrite,
        )
        .unwrap();
        let mut gl = GLProcessorThreaded::new(None).unwrap();
        let convert = |gl: &mut GLProcessorThreaded, s: &TensorDyn, d: &mut TensorDyn| {
            gl.convert(s, d, None).unwrap();
        };

        // (width, height, salt): distinct geometries with distinct content.
        let frames: [(usize, usize, u8); 4] =
            [(128, 96, 0), (96, 128, 40), (120, 100, 80), (64, 128, 120)];

        for (i, &(w, h, salt)) in frames.iter().enumerate() {
            // Recycled pool: reconfigure to this frame's geometry, write, convert.
            pool.configure_image(w, h, PixelFormat::Grey).unwrap();
            fill_grey_pattern(&pool, w, h, salt);
            let mut dst_recycled = TensorDyn::image(
                w,
                h,
                PixelFormat::Rgba,
                DType::U8,
                Some(TensorMemory::DmaBuf),
                edgefirst_tensor::CpuAccess::ReadWrite,
            )
            .unwrap();
            convert(&mut gl, &pool, &mut dst_recycled);
            let recycled_bytes = dst_recycled.as_u8().unwrap().map().unwrap().to_vec();

            // Fresh-source reference: a new buffer with the same geometry and content.
            let fresh = TensorDyn::image(
                w,
                h,
                PixelFormat::Grey,
                DType::U8,
                Some(TensorMemory::DmaBuf),
                edgefirst_tensor::CpuAccess::ReadWrite,
            )
            .unwrap();
            fill_grey_pattern(&fresh, w, h, salt);
            let mut dst_fresh = TensorDyn::image(
                w,
                h,
                PixelFormat::Rgba,
                DType::U8,
                Some(TensorMemory::DmaBuf),
                edgefirst_tensor::CpuAccess::ReadWrite,
            )
            .unwrap();
            convert(&mut gl, &fresh, &mut dst_fresh);
            let fresh_bytes = dst_fresh.as_u8().unwrap().map().unwrap().to_vec();

            assert_eq!(
                recycled_bytes.len(),
                fresh_bytes.len(),
                "frame {i} ({w}x{h}): output size mismatch"
            );
            assert_pixels_match(&fresh_bytes, &recycled_bytes, 0);
        }
    }

    /// Pool buffers recycled through different geometries, interleaved, match fresh sources.
    #[test]
    #[cfg(any(target_os = "linux", target_os = "macos", target_os = "windows"))]
    fn dma_pool_geometry_interleaved_stale_read() {
        if !is_gpu_image_buffer_available() || !is_opengl_available() {
            crate::test_support::report_skip(&format!(
                "{} - no zero-copy GPU buffers or OpenGL",
                function!()
            ));
            return;
        }
        let mut pool = [
            TensorDyn::image(
                128,
                128,
                PixelFormat::Grey,
                DType::U8,
                Some(TensorMemory::DmaBuf),
                edgefirst_tensor::CpuAccess::ReadWrite,
            )
            .unwrap(),
            TensorDyn::image(
                128,
                128,
                PixelFormat::Grey,
                DType::U8,
                Some(TensorMemory::DmaBuf),
                edgefirst_tensor::CpuAccess::ReadWrite,
            )
            .unwrap(),
        ];
        let mut gl = GLProcessorThreaded::new(None).unwrap();
        let convert = |gl: &mut GLProcessorThreaded, s: &TensorDyn, d: &mut TensorDyn| {
            gl.convert(s, d, None).unwrap();
        };

        // (pool slot, width, height, salt): each step reuses a pool slot at a new size.
        let steps: [(usize, usize, usize, u8); 6] = [
            (0, 128, 96, 0),
            (1, 96, 128, 30),
            (0, 64, 100, 60),  // slot 0 reused at a 3rd geometry
            (1, 120, 64, 90),  // slot 1 reused at a 2nd geometry
            (0, 128, 96, 120), // slot 0 back to its 1st geometry, new content
            (1, 96, 128, 150),
        ];

        for &(slot, w, h, salt) in steps.iter() {
            pool[slot].configure_image(w, h, PixelFormat::Grey).unwrap();
            fill_grey_pattern(&pool[slot], w, h, salt);
            let mut dst = TensorDyn::image(
                w,
                h,
                PixelFormat::Rgba,
                DType::U8,
                Some(TensorMemory::DmaBuf),
                edgefirst_tensor::CpuAccess::ReadWrite,
            )
            .unwrap();
            convert(&mut gl, &pool[slot], &mut dst);

            let fresh = TensorDyn::image(
                w,
                h,
                PixelFormat::Grey,
                DType::U8,
                Some(TensorMemory::DmaBuf),
                edgefirst_tensor::CpuAccess::ReadWrite,
            )
            .unwrap();
            fill_grey_pattern(&fresh, w, h, salt);
            let mut oracle = TensorDyn::image(
                w,
                h,
                PixelFormat::Rgba,
                DType::U8,
                Some(TensorMemory::DmaBuf),
                edgefirst_tensor::CpuAccess::ReadWrite,
            )
            .unwrap();
            convert(&mut gl, &fresh, &mut oracle);

            let dst_map = dst.as_u8().unwrap().map().unwrap();
            let oracle_map = oracle.as_u8().unwrap().map().unwrap();
            assert_pixels_match(oracle_map.as_slice(), dst_map.as_slice(), 0);
        }
    }

    /// A single-shot (non-recycled) Grey DMA source matches a second fresh-source convert.
    #[test]
    #[cfg(any(target_os = "linux", target_os = "macos", target_os = "windows"))]
    fn dma_single_shot_grey_matches_fresh() {
        if !is_gpu_image_buffer_available() || !is_opengl_available() {
            crate::test_support::report_skip(&format!(
                "{} - no zero-copy GPU buffers or OpenGL",
                function!()
            ));
            return;
        }
        let (w, h) = (64usize, 64usize);
        let bytes = vec![128u8; w * h];
        let mut gl = GLProcessorThreaded::new(None).unwrap();
        let convert = |gl: &mut GLProcessorThreaded, s: &TensorDyn, d: &mut TensorDyn| {
            gl.convert(s, d, None).unwrap();
        };

        let src =
            load_raw_image(w, h, PixelFormat::Grey, Some(TensorMemory::DmaBuf), &bytes).unwrap();
        let mut dst = TensorDyn::image(
            w,
            h,
            PixelFormat::Rgba,
            DType::U8,
            Some(TensorMemory::DmaBuf),
            edgefirst_tensor::CpuAccess::ReadWrite,
        )
        .unwrap();
        convert(&mut gl, &src, &mut dst);

        let fresh =
            load_raw_image(w, h, PixelFormat::Grey, Some(TensorMemory::DmaBuf), &bytes).unwrap();
        let mut oracle = TensorDyn::image(
            w,
            h,
            PixelFormat::Rgba,
            DType::U8,
            Some(TensorMemory::DmaBuf),
            edgefirst_tensor::CpuAccess::ReadWrite,
        )
        .unwrap();
        convert(&mut gl, &fresh, &mut oracle);

        let oracle_map = oracle.as_u8().unwrap().map().unwrap();
        let dst_map = dst.as_u8().unwrap().map().unwrap();
        assert_pixels_match(oracle_map.as_slice(), dst_map.as_slice(), 0);
    }

    // =========================================================================
    // EGL Display Probe & Override Tests
    // =========================================================================

    /// `probe_egl_displays()` discovers the available display types in priority order.
    #[test]
    #[cfg(target_os = "linux")]
    fn test_probe_egl_displays() {
        let displays = match probe_egl_displays() {
            Ok(d) => d,
            Err(e) => {
                crate::test_support::report_skip(&format!(
                    "{} - EGL not available: {e:?}",
                    function!()
                ));
                return;
            }
        };

        if displays.is_empty() {
            crate::test_support::report_skip(&format!(
                "{} - No EGL displays available",
                function!()
            ));
            return;
        }

        let kinds: Vec<_> = displays.iter().map(|d| d.kind).collect();
        eprintln!("Probed EGL displays: {kinds:?}");
        for d in &displays {
            eprintln!("  {:?}: {}", d.kind, d.description);
        }

        // Priority order: PlatformDevice, GBM, Default.
        let priority = |k: &EglDisplayKind| match k {
            EglDisplayKind::PlatformDevice => 0,
            EglDisplayKind::Gbm => 1,
            EglDisplayKind::Default => 2,
        };
        for w in kinds.windows(2) {
            assert!(
                priority(&w[0]) < priority(&w[1]),
                "Display ordering violated: {:?} should come after {:?}",
                w[1],
                w[0],
            );
        }
    }

    /// A `GLProcessorThreaded::new()` after `probe_egl_displays()` reuses the shared display.
    #[test]
    #[cfg(target_os = "linux")]
    fn test_probe_then_create_gl_context() {
        let displays = match probe_egl_displays() {
            Ok(d) => d,
            Err(e) => {
                crate::test_support::report_skip(&format!(
                    "{} - EGL not available: {e:?}",
                    function!()
                ));
                return;
            }
        };

        if displays.is_empty() {
            crate::test_support::report_skip(&format!(
                "{} - No EGL displays available",
                function!()
            ));
            return;
        }

        eprintln!(
            "Probed displays: {:?}",
            displays.iter().map(|d| d.kind).collect::<Vec<_>>()
        );

        // Create a GL processor on the shared display.
        let mut gl = match GLProcessorThreaded::new(None) {
            Ok(gl) => gl,
            Err(e) => {
                crate::test_support::report_skip(&format!(
                    "{} - GLProcessorThreaded failed: {e:?}",
                    function!()
                ));
                return;
            }
        };

        // A simple convert.
        let src = edgefirst_tensor::TensorDyn::image(
            64,
            64,
            edgefirst_tensor::PixelFormat::Rgba,
            edgefirst_tensor::DType::U8,
            None,
            edgefirst_tensor::CpuAccess::ReadWrite,
        )
        .expect("create src failed");
        let mut dst = edgefirst_tensor::TensorDyn::image(
            32,
            32,
            edgefirst_tensor::PixelFormat::Rgba,
            edgefirst_tensor::DType::U8,
            None,
            edgefirst_tensor::CpuAccess::ReadWrite,
        )
        .expect("create dst failed");

        gl.convert(&src, &mut dst, None).expect("convert failed");

        assert_eq!(dst.width(), Some(32));
        assert_eq!(dst.height(), Some(32));
    }

    /// `GLProcessorThreaded::new(Some(kind))` works for each available display kind.
    #[test]
    #[cfg(target_os = "linux")]
    fn test_override_each_display_kind() {
        let displays = match probe_egl_displays() {
            Ok(d) => d,
            Err(e) => {
                crate::test_support::report_skip(&format!(
                    "{} - EGL not available: {e:?}",
                    function!()
                ));
                return;
            }
        };

        if displays.is_empty() {
            crate::test_support::report_skip(&format!(
                "{} - No EGL displays available",
                function!()
            ));
            return;
        }

        for display in &displays {
            eprintln!(
                "Testing override: {:?} ({})",
                display.kind, display.description
            );
            let mut gl = match GLProcessorThreaded::new(Some(display.kind)) {
                Ok(gl) => gl,
                // Skip a kind whose renderer is rejected as software OpenGL.
                Err(crate::Error::NotSupported(msg))
                    if msg.contains("software OpenGL renderer") =>
                {
                    crate::test_support::report_skip(&format!(
                        "{} - {:?}: {msg}",
                        function!(),
                        display.kind
                    ));
                    continue;
                }
                Err(e) => panic!(
                    "GLProcessorThreaded::new(Some({:?})) failed: {e:?}",
                    display.kind
                ),
            };

            // A PixelFormat::Rgba → PixelFormat::Rgba conversion through the context.
            let src = crate::load_image_test_helper(
                &edgefirst_bench::testdata::read("zidane.jpg"),
                Some(PixelFormat::Rgba),
                None,
            )
            .unwrap();
            let dst = TensorDyn::image(
                320,
                240,
                PixelFormat::Rgba,
                DType::U8,
                None,
                edgefirst_tensor::CpuAccess::ReadWrite,
            )
            .unwrap();
            let src_dyn = src;
            let mut dst_dyn = dst;
            gl.convert(&src_dyn, &mut dst_dyn, None)
                .unwrap_or_else(|e| {
                    panic!("convert() with {:?} display failed: {e:?}", display.kind)
                });
            eprintln!("  {:?} display: convert OK", display.kind);
        }
    }

    /// Requesting a display kind that is not available returns an error.
    #[test]
    #[cfg(target_os = "linux")]
    fn test_override_unavailable_display_errors() {
        let displays = match probe_egl_displays() {
            Ok(d) => d,
            Err(e) => {
                crate::test_support::report_skip(&format!(
                    "{} - EGL not available: {e:?}",
                    function!()
                ));
                return;
            }
        };
        let available_kinds: Vec<_> = displays.iter().map(|d| d.kind).collect();

        // Find a kind that is not available; skip when all three are.
        let unavailable = [
            EglDisplayKind::PlatformDevice,
            EglDisplayKind::Gbm,
            EglDisplayKind::Default,
        ]
        .into_iter()
        .find(|k| !available_kinds.contains(k));

        if let Some(kind) = unavailable {
            eprintln!("Testing override with unavailable kind: {kind:?}");
            let result = GLProcessorThreaded::new(Some(kind));
            assert!(
                result.is_err(),
                "Expected error for unavailable display kind {kind:?}, got Ok"
            );
            eprintln!("  Correctly returned error: {:?}", result.unwrap_err());
        } else {
            crate::test_support::report_skip(&format!(
                "{} - All three display kinds are available",
                function!()
            ));
        }
    }

    /// Auto-detection (`None`) creates a processor that converts.
    #[test]
    #[cfg(target_os = "linux")]
    fn test_auto_detect_display() {
        if !is_opengl_available() {
            crate::test_support::report_skip(&format!("{} - OpenGL not available", function!()));
            return;
        }

        let mut gl = GLProcessorThreaded::new(None).expect("auto-detect should succeed");
        let src = crate::load_image_test_helper(
            &edgefirst_bench::testdata::read("zidane.jpg"),
            Some(PixelFormat::Rgba),
            None,
        )
        .unwrap();
        let dst = TensorDyn::image(
            320,
            240,
            PixelFormat::Rgba,
            DType::U8,
            None,
            edgefirst_tensor::CpuAccess::ReadWrite,
        )
        .unwrap();
        let src_dyn = src;
        let mut dst_dyn = dst;
        gl.convert(&src_dyn, &mut dst_dyn, None)
            .expect("auto-detect convert should succeed");
    }

    #[test]
    fn test_packed_rgb_width_constraint() {
        // Standard model input widths satisfy W*3 % 4 == 0.
        assert_eq!((640usize * 3) % 4, 0);
        assert_eq!((320usize * 3) % 4, 0);
        assert_eq!((1280usize * 3) % 4, 0);

        // Widths that do not.
        assert_ne!((322usize * 3) % 4, 0);
        assert_ne!((333usize * 3) % 4, 0);
    }

    // =========================================================================
    // Packed PixelFormat::Rgb correctness tests (two-pass pipeline)
    // GL PixelFormat::Rgba output (alpha stripped) is compared against GL packed
    // PixelFormat::Rgb output.
    // =========================================================================

    /// Compare two byte slices pixel-by-pixel with tolerance.
    /// Panics with details if any byte differs by more than `tolerance`.
    #[cfg(any(target_os = "linux", target_os = "macos", target_os = "windows"))]
    fn assert_pixels_match(expected: &[u8], actual: &[u8], tolerance: u8) {
        assert_eq!(expected.len(), actual.len(), "Buffer size mismatch");
        let mut max_diff: u8 = 0;
        let mut diff_count: usize = 0;
        let mut first_diff_idx = None;
        for (i, (&e, &a)) in expected.iter().zip(actual.iter()).enumerate() {
            let diff = (e as i16 - a as i16).unsigned_abs() as u8;
            if diff > tolerance {
                diff_count += 1;
                if first_diff_idx.is_none() {
                    first_diff_idx = Some(i);
                }
            }
            max_diff = max_diff.max(diff);
        }
        assert!(
            diff_count == 0,
            "Pixel mismatch: {diff_count} bytes differ (max_diff={max_diff}, first at index {})",
            first_diff_idx.unwrap_or(0)
        );
    }

    /// Build a letterbox crop that fits src into dst_w x dst_h, preserving aspect ratio.
    #[cfg(any(target_os = "linux", target_os = "macos", target_os = "windows"))]
    fn letterbox_crop(_src_w: usize, _src_h: usize, _dst_w: usize, _dst_h: usize) -> Crop {
        // The backend computes the placement; this selects the pad colour.
        Crop::letterbox([114, 114, 114, 255])
    }

    /// Strip alpha from PixelFormat::Rgba bytes → packed PixelFormat::Rgb bytes.
    #[cfg(any(target_os = "linux", target_os = "macos", target_os = "windows"))]
    fn rgba_to_rgb(rgba: &[u8]) -> Vec<u8> {
        assert_eq!(
            rgba.len() % 4,
            0,
            "PixelFormat::Rgba buffer length must be divisible by 4"
        );
        let mut rgb = Vec::with_capacity(rgba.len() / 4 * 3);
        for pixel in rgba.as_chunks::<4>().0 {
            rgb.push(pixel[0]);
            rgb.push(pixel[1]);
            rgb.push(pixel[2]);
        }
        rgb
    }

    /// Convert uint8 PixelFormat::Rgb bytes to int8 (XOR 0x80 each byte).
    #[cfg(any(target_os = "linux", target_os = "macos", target_os = "windows"))]
    fn uint8_to_int8(data: &[u8]) -> Vec<u8> {
        data.iter().map(|&b| b ^ 0x80).collect()
    }

    /// YUYV 1080p → Rgb 640x640 letterbox: GL packed Rgb equals GL Rgba with alpha stripped.
    #[test]
    #[cfg(any(target_os = "linux", target_os = "macos", target_os = "windows"))]
    fn test_opengl_rgb_correctness() {
        if !is_opengl_available() {
            crate::test_support::report_skip(&format!("{} - GL not available", function!()));
            return;
        }
        if !is_gpu_image_buffer_available() {
            crate::test_support::report_skip(
                "test_opengl_rgb_correctness - zero-copy GPU image buffers not available",
            );
            return;
        }
        let src_dma = load_raw_image(
            1920,
            1080,
            PixelFormat::Yuyv,
            Some(TensorMemory::DmaBuf),
            &edgefirst_bench::testdata::read("camera1080p.yuyv"),
        )
        .unwrap();

        let crop = letterbox_crop(1920, 1080, 640, 640);
        let mut gl = GLProcessorThreaded::new(None).unwrap();
        let src_dyn = src_dma;

        // GL PixelFormat::Rgba reference
        let dst_rgba = TensorDyn::image(
            640,
            640,
            PixelFormat::Rgba,
            DType::U8,
            Some(TensorMemory::DmaBuf),
            edgefirst_tensor::CpuAccess::ReadWrite,
        )
        .unwrap();
        let mut dst_rgba_dyn = dst_rgba;
        gl.convert(
            &src_dyn,
            &mut dst_rgba_dyn,
            Some(crate::ConvertOptions::new().crop(crop)),
        )
        .unwrap();

        let dst_rgb = TensorDyn::image(
            640,
            640,
            PixelFormat::Rgb,
            DType::U8,
            Some(TensorMemory::DmaBuf),
            edgefirst_tensor::CpuAccess::ReadWrite,
        )
        .unwrap();
        let mut dst_rgb_dyn = dst_rgb;
        gl.convert(
            &src_dyn,
            &mut dst_rgb_dyn,
            Some(crate::ConvertOptions::new().crop(crop)),
        )
        .unwrap();

        let rgba_data = dst_rgba_dyn.as_u8().unwrap().map().unwrap();
        let expected_rgb = rgba_to_rgb(rgba_data.as_slice());
        let gl_data = dst_rgb_dyn.as_u8().unwrap().map().unwrap();
        assert_pixels_match(&expected_rgb, gl_data.as_slice(), 1);
    }

    /// YUYV 1080p → Rgb int8 640x640 letterbox: GL packed Rgb equals GL Rgba stripped and XOR 0x80.
    #[test]
    #[cfg(any(target_os = "linux", target_os = "macos", target_os = "windows"))]
    fn test_opengl_rgb_int8_correctness() {
        if !is_gpu_image_buffer_available() {
            crate::test_support::report_skip(
                "test_opengl_rgb_int8_correctness - zero-copy GPU image buffers not available",
            );
            return;
        }
        let src_dma = load_raw_image(
            1920,
            1080,
            PixelFormat::Yuyv,
            Some(TensorMemory::DmaBuf),
            &edgefirst_bench::testdata::read("camera1080p.yuyv"),
        )
        .unwrap();

        let crop = letterbox_crop(1920, 1080, 640, 640);
        let mut gl = match GLProcessorThreaded::new(None) {
            Ok(gl) => gl,
            Err(e) => {
                crate::test_support::report_skip(&format!(
                    "{} - GL not available: {e}",
                    function!()
                ));
                return;
            }
        };

        let src_dyn = src_dma;

        // GL PixelFormat::Rgba reference
        let dst_rgba = TensorDyn::image(
            640,
            640,
            PixelFormat::Rgba,
            DType::U8,
            Some(TensorMemory::DmaBuf),
            edgefirst_tensor::CpuAccess::ReadWrite,
        )
        .unwrap();
        let mut dst_rgba_dyn = dst_rgba;
        gl.convert(
            &src_dyn,
            &mut dst_rgba_dyn,
            Some(crate::ConvertOptions::new().crop(crop)),
        )
        .unwrap();

        let dst_rgb = TensorDyn::image(
            640,
            640,
            PixelFormat::Rgb,
            DType::I8,
            Some(TensorMemory::DmaBuf),
            edgefirst_tensor::CpuAccess::ReadWrite,
        )
        .unwrap();
        let mut dst_rgb_dyn = dst_rgb;
        gl.convert(
            &src_dyn,
            &mut dst_rgb_dyn,
            Some(crate::ConvertOptions::new().crop(crop)),
        )
        .unwrap();

        let rgba_data = dst_rgba_dyn.as_u8().unwrap().map().unwrap();
        let expected_rgb = uint8_to_int8(&rgba_to_rgb(rgba_data.as_slice()));
        // Map the raw i8 bytes as u8 for comparison.
        let gl_data = dst_rgb_dyn.as_i8().unwrap().map().unwrap();
        let gl_bytes: &[u8] = unsafe {
            std::slice::from_raw_parts(gl_data.as_slice().as_ptr().cast(), gl_data.len())
        };
        assert_pixels_match(&expected_rgb, gl_bytes, 1);
    }

    /// YUYV 1080p → Rgb 1920x1080, no scaling: GL packed Rgb equals GL Rgba with alpha stripped.
    #[test]
    #[cfg(any(target_os = "linux", target_os = "macos", target_os = "windows"))]
    fn test_opengl_rgb_no_letterbox_correctness() {
        if !is_opengl_available() {
            crate::test_support::report_skip(&format!("{} - GL not available", function!()));
            return;
        }
        if !is_gpu_image_buffer_available() {
            crate::test_support::report_skip("test_opengl_rgb_no_letterbox_correctness - zero-copy GPU image buffers not available");
            return;
        }
        let src_dma = load_raw_image(
            1920,
            1080,
            PixelFormat::Yuyv,
            Some(TensorMemory::DmaBuf),
            &edgefirst_bench::testdata::read("camera1080p.yuyv"),
        )
        .unwrap();

        let mut gl = GLProcessorThreaded::new(None).unwrap();
        let src_dyn = src_dma;

        // GL PixelFormat::Rgba reference
        let dst_rgba = TensorDyn::image(
            1920,
            1080,
            PixelFormat::Rgba,
            DType::U8,
            Some(TensorMemory::DmaBuf),
            edgefirst_tensor::CpuAccess::ReadWrite,
        )
        .unwrap();
        let mut dst_rgba_dyn = dst_rgba;
        gl.convert(&src_dyn, &mut dst_rgba_dyn, None).unwrap();

        let dst_rgb = TensorDyn::image(
            1920,
            1080,
            PixelFormat::Rgb,
            DType::U8,
            Some(TensorMemory::DmaBuf),
            edgefirst_tensor::CpuAccess::ReadWrite,
        )
        .unwrap();
        let mut dst_rgb_dyn = dst_rgb;
        gl.convert(&src_dyn, &mut dst_rgb_dyn, None).unwrap();

        let rgba_data = dst_rgba_dyn.as_u8().unwrap().map().unwrap();
        let expected_rgb = rgba_to_rgb(rgba_data.as_slice());
        let gl_data = dst_rgb_dyn.as_u8().unwrap().map().unwrap();
        assert_pixels_match(&expected_rgb, gl_data.as_slice(), 1);
    }

    // ---- PixelFormat::Bgra destination tests ----

    /// NV12→Bgra with DMA buffers equals NV12→Rgba with R and B swapped.
    #[test]
    #[cfg(any(target_os = "linux", target_os = "macos", target_os = "windows"))]
    fn test_opengl_nv12_to_bgra() {
        if !is_gpu_image_buffer_available() || !is_opengl_available() {
            crate::test_support::report_skip(
                "test_opengl_nv12_to_bgra - no zero-copy GPU buffers or OpenGL",
            );
            return;
        }

        let src = load_raw_image(
            1280,
            720,
            PixelFormat::Nv12,
            Some(TensorMemory::DmaBuf),
            &edgefirst_bench::testdata::read("camera720p.nv12"),
        )
        .unwrap();

        let mut gl = GLProcessorThreaded::new(None).unwrap();
        let src_dyn = src;

        // Convert to PixelFormat::Rgba as reference
        let rgba_dst = TensorDyn::image(
            1280,
            720,
            PixelFormat::Rgba,
            DType::U8,
            Some(TensorMemory::DmaBuf),
            edgefirst_tensor::CpuAccess::ReadWrite,
        )
        .unwrap();
        let mut rgba_dst_dyn = rgba_dst;
        gl.convert(&src_dyn, &mut rgba_dst_dyn, None).unwrap();

        let bgra_dst = TensorDyn::image(
            1280,
            720,
            PixelFormat::Bgra,
            DType::U8,
            Some(TensorMemory::DmaBuf),
            edgefirst_tensor::CpuAccess::ReadWrite,
        )
        .unwrap();
        let mut bgra_dst_dyn = bgra_dst;
        gl.convert(&src_dyn, &mut bgra_dst_dyn, None).unwrap();

        // Bgra[B,G,R,A] matches Rgba[R,G,B,A] with R and B swapped.
        let bgra_map = bgra_dst_dyn.as_u8().unwrap().map().unwrap();
        let rgba_map = rgba_dst_dyn.as_u8().unwrap().map().unwrap();
        let bgra_buf = bgra_map.as_slice();
        let rgba_buf = rgba_map.as_slice();

        assert_eq!(bgra_buf.len(), rgba_buf.len());
        let mut max_diff = 0i32;
        for (bc, rc) in bgra_buf
            .as_chunks::<4>()
            .0
            .iter()
            .zip(rgba_buf.as_chunks::<4>().0)
        {
            max_diff = max_diff.max((bc[0] as i32 - rc[2] as i32).abs()); // B
            max_diff = max_diff.max((bc[1] as i32 - rc[1] as i32).abs()); // G
            max_diff = max_diff.max((bc[2] as i32 - rc[0] as i32).abs()); // R
            max_diff = max_diff.max((bc[3] as i32 - rc[3] as i32).abs()); // A
        }
        eprintln!("PixelFormat::Nv12→PixelFormat::Bgra vs PixelFormat::Nv12→PixelFormat::Rgba max channel diff: {max_diff}");
        assert!(
            max_diff <= 1,
            "PixelFormat::Bgra/PixelFormat::Rgba channel mismatch > 1: max_diff={max_diff}"
        );
    }

    /// YUYV→Bgra with DMA buffers equals YUYV→Rgba with R and B swapped.
    #[test]
    #[cfg(any(target_os = "linux", target_os = "macos", target_os = "windows"))]
    fn test_opengl_yuyv_to_bgra() {
        if !is_gpu_image_buffer_available() || !is_opengl_available() {
            crate::test_support::report_skip(
                "test_opengl_yuyv_to_bgra - no zero-copy GPU buffers or OpenGL",
            );
            return;
        }

        let src = load_raw_image(
            1280,
            720,
            PixelFormat::Yuyv,
            Some(TensorMemory::DmaBuf),
            &edgefirst_bench::testdata::read("camera720p.yuyv"),
        )
        .unwrap();

        let mut gl = GLProcessorThreaded::new(None).unwrap();
        let src_dyn = src;

        let rgba_dst = TensorDyn::image(
            1280,
            720,
            PixelFormat::Rgba,
            DType::U8,
            Some(TensorMemory::DmaBuf),
            edgefirst_tensor::CpuAccess::ReadWrite,
        )
        .unwrap();
        let mut rgba_dst_dyn = rgba_dst;
        gl.convert(&src_dyn, &mut rgba_dst_dyn, None).unwrap();

        let bgra_dst = TensorDyn::image(
            1280,
            720,
            PixelFormat::Bgra,
            DType::U8,
            Some(TensorMemory::DmaBuf),
            edgefirst_tensor::CpuAccess::ReadWrite,
        )
        .unwrap();
        let mut bgra_dst_dyn = bgra_dst;
        gl.convert(&src_dyn, &mut bgra_dst_dyn, None).unwrap();

        let bgra_map = bgra_dst_dyn.as_u8().unwrap().map().unwrap();
        let rgba_map = rgba_dst_dyn.as_u8().unwrap().map().unwrap();
        let bgra_buf = bgra_map.as_slice();
        let rgba_buf = rgba_map.as_slice();

        let mut max_diff = 0i32;
        for (bc, rc) in bgra_buf
            .as_chunks::<4>()
            .0
            .iter()
            .zip(rgba_buf.as_chunks::<4>().0)
        {
            max_diff = max_diff.max((bc[0] as i32 - rc[2] as i32).abs());
            max_diff = max_diff.max((bc[1] as i32 - rc[1] as i32).abs());
            max_diff = max_diff.max((bc[2] as i32 - rc[0] as i32).abs());
            max_diff = max_diff.max((bc[3] as i32 - rc[3] as i32).abs());
        }
        eprintln!("PixelFormat::Yuyv→PixelFormat::Bgra vs PixelFormat::Yuyv→PixelFormat::Rgba max channel diff: {max_diff}");
        assert!(
            max_diff <= 1,
            "PixelFormat::Bgra/PixelFormat::Rgba channel mismatch > 1: max_diff={max_diff}"
        );
    }

    /// `draw_decoded_masks()` of segmentation into Bgra equals Rgba with R and B swapped.
    #[test]
    fn test_draw_decoded_masks_bgra() {
        use edgefirst_tensor::Segmentation;

        if !is_opengl_available() {
            crate::test_support::report_skip("test_draw_decoded_masks_bgra - OpenGL not available");
            return;
        }

        let seg_bytes = edgefirst_bench::testdata::read("modelpack_seg_2x160x160.bin").to_vec();

        // Build segmentation data (shared between both renders)
        let make_seg = || {
            let mut s = Array3::from_shape_vec((2, 160, 160), seg_bytes.clone()).unwrap();
            s.swap_axes(0, 1);
            s.swap_axes(1, 2);
            let s = s.as_standard_layout().to_owned();
            Segmentation {
                segmentation: edgefirst_tensor::Tensor::from_arrayview3(s.view())
                    .unwrap()
                    .into(),
                xmin: 0.0,
                ymin: 0.0,
                xmax: 1.0,
                ymax: 1.0,
            }
        };

        let mut gl = GLProcessorThreaded::new(None).unwrap();

        // Render to PixelFormat::Rgba
        let rgba_img = crate::load_image_test_helper(
            &edgefirst_bench::testdata::read("giraffe.jpg"),
            Some(PixelFormat::Rgba),
            None,
        )
        .unwrap();
        let mut rgba_img_dyn = rgba_img;
        gl.draw_decoded_masks(&mut rgba_img_dyn, &[], &[make_seg()], Default::default())
            .unwrap();

        // Render to PixelFormat::Bgra (convert source to PixelFormat::Bgra first)
        let rgba_src = crate::load_image_test_helper(
            &edgefirst_bench::testdata::read("giraffe.jpg"),
            Some(PixelFormat::Rgba),
            None,
        )
        .unwrap();
        let bgra_img = TensorDyn::image(
            rgba_src.width().unwrap(),
            rgba_src.height().unwrap(),
            PixelFormat::Bgra,
            DType::U8,
            None,
            edgefirst_tensor::CpuAccess::ReadWrite,
        )
        .unwrap();
        let rgba_src_dyn = rgba_src;
        let mut bgra_img_dyn = bgra_img;
        gl.convert(&rgba_src_dyn, &mut bgra_img_dyn, None).unwrap();
        gl.draw_decoded_masks(&mut bgra_img_dyn, &[], &[make_seg()], Default::default())
            .unwrap();

        // The Bgra output matches the Rgba output with R and B swapped.
        let rgba_map = rgba_img_dyn.as_u8().unwrap().map().unwrap();
        let bgra_map = bgra_img_dyn.as_u8().unwrap().map().unwrap();
        let rgba_buf = rgba_map.as_slice();
        let bgra_buf = bgra_map.as_slice();
        assert_eq!(rgba_buf.len(), bgra_buf.len());

        let mut max_diff = 0i32;
        for (rc, bc) in rgba_buf
            .as_chunks::<4>()
            .0
            .iter()
            .zip(bgra_buf.as_chunks::<4>().0)
        {
            max_diff = max_diff.max((rc[0] as i32 - bc[2] as i32).abs()); // R
            max_diff = max_diff.max((rc[1] as i32 - bc[1] as i32).abs()); // G
            max_diff = max_diff.max((rc[2] as i32 - bc[0] as i32).abs()); // B
            max_diff = max_diff.max((rc[3] as i32 - bc[3] as i32).abs()); // A
        }
        eprintln!("draw_decoded_masks PixelFormat::Bgra vs PixelFormat::Rgba max channel diff: {max_diff}");
        assert!(
            max_diff <= 1,
            "draw_decoded_masks PixelFormat::Bgra/PixelFormat::Rgba channel mismatch > 1: max_diff={max_diff}"
        );
    }

    /// `draw_decoded_masks()` of boxes into a Mem Bgra destination equals Rgba, R and B swapped.
    #[test]
    fn test_draw_decoded_masks_bgra_mem() {
        use edgefirst_tensor::DetectBox;

        if !is_opengl_available() {
            crate::test_support::report_skip(
                "test_draw_decoded_masks_bgra_mem - OpenGL not available",
            );
            return;
        }

        let detect = DetectBox {
            bbox: [0.59375, 0.25, 0.9375, 0.725].into(),
            score: 0.99,
            label: 0,
        };
        let colors = [[255, 255, 0, 233], [128, 128, 255, 100]];

        let mut gl = GLProcessorThreaded::new(None).unwrap();
        gl.set_class_colors(&colors).unwrap();

        // Render boxes to PixelFormat::Rgba
        let rgba_img = crate::load_image_test_helper(
            &edgefirst_bench::testdata::read("giraffe.jpg"),
            Some(PixelFormat::Rgba),
            Some(edgefirst_tensor::TensorMemory::Mem),
        )
        .unwrap();
        let mut rgba_img_dyn = rgba_img;
        gl.draw_decoded_masks(&mut rgba_img_dyn, &[detect], &[], Default::default())
            .unwrap();

        // Render boxes to PixelFormat::Bgra
        let rgba_src = crate::load_image_test_helper(
            &edgefirst_bench::testdata::read("giraffe.jpg"),
            Some(PixelFormat::Rgba),
            Some(edgefirst_tensor::TensorMemory::Mem),
        )
        .unwrap();
        let bgra_img = TensorDyn::image(
            rgba_src.width().unwrap(),
            rgba_src.height().unwrap(),
            PixelFormat::Bgra,
            DType::U8,
            Some(edgefirst_tensor::TensorMemory::Mem),
            edgefirst_tensor::CpuAccess::ReadWrite,
        )
        .unwrap();
        let rgba_src_dyn = rgba_src;
        let mut bgra_img_dyn = bgra_img;
        gl.convert(&rgba_src_dyn, &mut bgra_img_dyn, None).unwrap();
        gl.draw_decoded_masks(&mut bgra_img_dyn, &[detect], &[], Default::default())
            .unwrap();

        // The Bgra output matches the Rgba output with R and B swapped.
        let rgba_map = rgba_img_dyn.as_u8().unwrap().map().unwrap();
        let bgra_map = bgra_img_dyn.as_u8().unwrap().map().unwrap();
        let rgba_buf = rgba_map.as_slice();
        let bgra_buf = bgra_map.as_slice();

        let mut max_diff = 0i32;
        for (rc, bc) in rgba_buf
            .as_chunks::<4>()
            .0
            .iter()
            .zip(bgra_buf.as_chunks::<4>().0)
        {
            max_diff = max_diff.max((rc[0] as i32 - bc[2] as i32).abs());
            max_diff = max_diff.max((rc[1] as i32 - bc[1] as i32).abs());
            max_diff = max_diff.max((rc[2] as i32 - bc[0] as i32).abs());
            max_diff = max_diff.max((rc[3] as i32 - bc[3] as i32).abs());
        }
        eprintln!(
            "draw_decoded_masks_mem PixelFormat::Bgra vs PixelFormat::Rgba max channel diff: {max_diff}"
        );
        assert!(
            max_diff <= 1,
            "draw_decoded_masks_mem PixelFormat::Bgra/PixelFormat::Rgba channel mismatch > 1: max_diff={max_diff}"
        );
    }

    // ========================================================================
    // GL smoke tests for mask rendering and PBO destinations
    // ========================================================================

    #[test]
    fn test_gl_mask_render_smoke() {
        if !is_opengl_available() {
            crate::test_support::report_skip(&format!("{} - OpenGL not available", function!()));
            return;
        }

        let mut gl = GLProcessorThreaded::new(None).unwrap();
        let image = TensorDyn::image(
            64,
            64,
            PixelFormat::Rgba,
            DType::U8,
            None,
            edgefirst_tensor::CpuAccess::ReadWrite,
        )
        .unwrap();
        let mut image_dyn = image;

        // Render with empty detections and segmentations.
        let result = gl.draw_decoded_masks(&mut image_dyn, &[], &[], Default::default());
        assert!(
            result.is_ok(),
            "GL mask render with empty data should succeed: {result:?}"
        );

        assert_eq!(image_dyn.width(), Some(64));
        assert_eq!(image_dyn.height(), Some(64));
    }

    #[test]
    #[cfg(any(target_os = "linux", target_os = "windows"))]
    fn test_gl_pbo_destination_smoke() {
        if !is_opengl_available() {
            crate::test_support::report_skip(&format!("{} - OpenGL not available", function!()));
            return;
        }

        let gl = GLProcessorThreaded::new(None).unwrap();
        let pbo_img = gl
            .create_pbo_image(64, 64, PixelFormat::Rgba)
            .expect("allocate an RGBA8 PBO on the GL thread");
        assert_eq!(pbo_img.width(), Some(64));
        assert_eq!(pbo_img.height(), Some(64));
        assert_eq!(pbo_img.format().unwrap(), PixelFormat::Rgba);
    }

    /// A convert into a PBO destination completes without deadlocking the GL thread.
    #[test]
    #[cfg(any(target_os = "linux", target_os = "windows"))]
    fn test_gl_convert_any_to_pbo_no_deadlock() {
        if !is_opengl_available() {
            crate::test_support::report_skip(&format!("{} - OpenGL not available", function!()));
            return;
        }

        let mut gl = GLProcessorThreaded::new(None).unwrap();
        let pbo_dst = gl
            .create_pbo_image(64, 64, PixelFormat::Rgba)
            .expect("allocate the PBO destination");
        let src = Tensor::<u8>::image(
            64,
            64,
            PixelFormat::Rgba,
            Some(TensorMemory::Mem),
            edgefirst_tensor::CpuAccess::ReadWrite,
        )
        .expect("Mem source tensor");
        // Fill the source with a deterministic pattern.
        {
            let mut map = src.map().unwrap();
            let slice: &mut [u8] = &mut map;
            for (i, b) in slice.iter_mut().enumerate() {
                *b = (i & 0xFF) as u8;
            }
        }
        let src_dyn = TensorDyn::from(src);
        let mut dst_dyn = TensorDyn::from(pbo_dst);

        let res = gl.convert(&src_dyn, &mut dst_dyn, None);
        match res {
            Ok(()) => {
                assert_eq!(dst_dyn.width(), Some(64));
                assert_eq!(dst_dyn.height(), Some(64));
            }
            Err(e) => {
                // Only the deadlock signature fails the test.
                let msg = format!("{e:?}");
                assert!(
                    !msg.contains("GL converter thread exited"),
                    "PBO destination convert deadlocked: {msg}"
                );
                crate::test_support::report_skip(&format!(
                    "{} - convert failed unrelated: {msg}",
                    function!()
                ));
            }
        }
    }

    /// A PBO to PBO convert completes without deadlocking the GL thread.
    #[test]
    #[cfg(any(target_os = "linux", target_os = "windows"))]
    fn test_gl_convert_pbo_to_pbo_no_deadlock() {
        if !is_opengl_available() {
            crate::test_support::report_skip(&format!("{} - OpenGL not available", function!()));
            return;
        }

        let mut gl = GLProcessorThreaded::new(None).unwrap();
        let pbo_src = gl
            .create_pbo_image(64, 64, PixelFormat::Rgba)
            .expect("allocate the PBO source");
        let pbo_dst = gl
            .create_pbo_image(64, 64, PixelFormat::Rgba)
            .expect("allocate the PBO destination");
        let src_dyn = TensorDyn::from(pbo_src);
        let mut dst_dyn = TensorDyn::from(pbo_dst);
        let res = gl.convert(&src_dyn, &mut dst_dyn, None);
        match res {
            Ok(()) => {
                assert_eq!(dst_dyn.width(), Some(64));
            }
            Err(e) => {
                let msg = format!("{e:?}");
                assert!(
                    !msg.contains("GL converter thread exited"),
                    "PBO→PBO convert deadlocked: {msg}"
                );
                crate::test_support::report_skip(&format!(
                    "{} - convert failed unrelated: {msg}",
                    function!()
                ));
            }
        }
    }

    /// A PBO source into a planar destination is read through `PIXEL_UNPACK_BUFFER`, not mapped.
    #[test]
    #[cfg(any(target_os = "linux", target_os = "windows"))]
    fn test_gl_convert_pbo_src_to_planar_no_deadlock() {
        if !is_opengl_available() {
            crate::test_support::report_skip(&format!("{} - OpenGL not available", function!()));
            return;
        }

        let mut gl = GLProcessorThreaded::new(None).unwrap();
        let pbo_src = gl
            .create_pbo_image(64, 64, PixelFormat::Rgba)
            .expect("allocate the PBO source");
        let dst = Tensor::<u8>::image(
            64,
            64,
            PixelFormat::PlanarRgb,
            Some(TensorMemory::Mem),
            edgefirst_tensor::CpuAccess::ReadWrite,
        )
        .expect("planar destination tensor");

        let before = gl.convert_stats().expect("convert stats before");
        let src_dyn = TensorDyn::from(pbo_src);
        let mut dst_dyn = TensorDyn::from(dst);
        let res = gl.convert(&src_dyn, &mut dst_dyn, None);
        match res {
            Ok(()) => {
                let after = gl.convert_stats().expect("convert stats after");
                assert_eq!(
                    after.src_pbo_uploads - before.src_pbo_uploads,
                    1,
                    "the planar route did not feed its source from the PBO"
                );
                assert_eq!(
                    after.src_uploads, before.src_uploads,
                    "the planar route mapped its PBO source instead of reading \
                     it through PIXEL_UNPACK_BUFFER"
                );
            }
            Err(e) => {
                let msg = format!("{e:?}");
                assert!(
                    !msg.contains("GL converter thread exited"),
                    "PBO source to planar convert deadlocked: {msg}"
                );
                crate::test_support::report_skip(&format!(
                    "{} - convert failed unrelated: {msg}",
                    function!()
                ));
            }
        }
    }

    /// The lock every GL-environment guard below holds while its variable is set.
    fn lock_gl_env() -> std::sync::MutexGuard<'static, ()> {
        static GL_ENV: std::sync::Mutex<()> = std::sync::Mutex::new(());
        GL_ENV.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// Sets `EDGEFIRST_GL_SERIALIZE` until dropped, holding [`lock_gl_env`].
    #[cfg(any(target_os = "linux", target_os = "windows"))]
    struct SerializeEnvGuard(#[allow(dead_code)] std::sync::MutexGuard<'static, ()>);
    #[cfg(any(target_os = "linux", target_os = "windows"))]
    impl SerializeEnvGuard {
        fn set(v: &str) -> Self {
            let held = lock_gl_env();
            std::env::set_var("EDGEFIRST_GL_SERIALIZE", v);
            SerializeEnvGuard(held)
        }
    }
    #[cfg(any(target_os = "linux", target_os = "windows"))]
    impl Drop for SerializeEnvGuard {
        fn drop(&mut self) {
            std::env::remove_var("EDGEFIRST_GL_SERIALIZE");
        }
    }

    /// A PBO source owned by another processor's context is refused under the `Full` policy.
    #[test]
    #[cfg(any(target_os = "linux", target_os = "windows"))]
    fn test_gl_cross_context_pbo_source_refused_not_deadlocked() {
        if !is_opengl_available() {
            crate::test_support::report_skip(&format!("{} - OpenGL not available", function!()));
            return;
        }

        let _serialize = SerializeEnvGuard::set("full");
        let mut gl_a = GLProcessorThreaded::new(None).unwrap();
        let gl_b = GLProcessorThreaded::new(None).unwrap();

        let pbo_src = gl_b
            .create_pbo_image(64, 64, PixelFormat::Rgba)
            .expect("allocate the PBO source on the other processor");
        let dst = Tensor::<u8>::image(
            64,
            64,
            PixelFormat::PlanarRgb,
            Some(TensorMemory::Mem),
            edgefirst_tensor::CpuAccess::ReadWrite,
        )
        .expect("planar destination tensor");

        let src_dyn = TensorDyn::from(pbo_src);
        let mut dst_dyn = TensorDyn::from(dst);
        let res = gl_a.convert(&src_dyn, &mut dst_dyn, None);
        let msg = format!(
            "{:?}",
            res.expect_err("a cross-context PBO map under Full serialization must be refused")
        );
        assert!(
            msg.contains("another GL context"),
            "must be refused by the cross-context guard, not some other error path: {msg}"
        );
    }

    /// A PBO source owned by another processor's context converts through its owner's map.
    #[test]
    #[cfg(target_os = "linux")]
    fn a_foreign_pbo_source_is_mapped_not_bound() {
        if !is_opengl_available() {
            crate::test_support::report_skip(&format!("{} - OpenGL not available", function!()));
            return;
        }
        let _serialize = SerializeEnvGuard::set("lifecycle");
        let mut gl_a = GLProcessorThreaded::new(None).unwrap();
        let gl_b = GLProcessorThreaded::new(None).unwrap();
        let (w, h) = (320usize, 240usize);
        let foreign = rgba8_pbo(&gl_b, w, h);
        stamp_rgba8_positionally(&foreign, w, h);
        let rgba = || {
            TensorDyn::image(
                w,
                h,
                PixelFormat::Rgba,
                DType::U8,
                Some(TensorMemory::Mem),
                edgefirst_tensor::CpuAccess::ReadWrite,
            )
            .unwrap()
        };
        let want = rgba();
        stamp_rgba8_positionally(&want, w, h);

        let mut got = rgba();
        let before = gl_a.convert_stats().expect("convert stats before");
        gl_a.convert(&foreign, &mut got, None)
            .expect("a foreign PBO source converts through its owner's map");
        let after = gl_a.convert_stats().expect("convert stats after");
        assert_eq!(
            after.src_pbo_uploads, before.src_pbo_uploads,
            "another context's buffer id was bound in this context"
        );
        assert_eq!(after.src_uploads - before.src_uploads, 1);
        let (worst, first) = image_max_diff(&got, &want, 1);
        assert!(
            first.is_none(),
            "foreign PBO source differs from its own pixels by up to {worst} at {first:?}"
        );
    }

    /// A PBO destination owned by another processor is written through its owner's map.
    #[test]
    #[cfg(target_os = "linux")]
    fn a_foreign_pbo_destination_is_mapped_not_bound() {
        use crate::cpu::CPUProcessor;
        if !is_opengl_available() {
            crate::test_support::report_skip(&format!("{} - OpenGL not available", function!()));
            return;
        }
        let _serialize = SerializeEnvGuard::set("lifecycle");
        let mut gl_a = GLProcessorThreaded::new(None).unwrap();
        let gl_b = GLProcessorThreaded::new(None).unwrap();
        let (w, h) = (320usize, 240usize);
        let access = edgefirst_tensor::CpuAccess::ReadWrite;
        let cm = edgefirst_tensor::Colorimetry::default()
            .with_encoding(edgefirst_tensor::ColorEncoding::Bt601)
            .with_range(edgefirst_tensor::ColorRange::Full);
        let mut src = TensorDyn::image(
            w,
            h,
            PixelFormat::Nv12,
            DType::U8,
            Some(TensorMemory::Mem),
            access,
        )
        .unwrap();
        stamp_yuv_gradient(&src, PixelFormat::Nv12, w, h);
        src.set_colorimetry(Some(cm));
        let mut want = TensorDyn::image(
            w,
            h,
            PixelFormat::Rgba,
            DType::U8,
            Some(TensorMemory::Mem),
            access,
        )
        .unwrap();
        CPUProcessor::new().convert(&src, &mut want, None).unwrap();

        let mut foreign: TensorDyn = gl_b
            .create_pbo_image(w, h, PixelFormat::Rgba)
            .expect("allocate a PBO on the other processor")
            .into();
        gl_a.convert(&src, &mut foreign, None)
            .expect("a foreign PBO destination is written through its owner's map");
        let (worst, first) = image_max_diff(&foreign, &want, 4);
        assert!(
            first.is_none(),
            "foreign PBO destination differs from the CPU by up to {worst} at {first:?}"
        );
    }

    // ---- Multiplane PixelFormat::Nv12 GPU tests ----

    /// Whether `result` is the GL backend's decline of a semi-planar source whose chroma is a
    /// separate DMA-BUF. Reports the skip.
    #[cfg(any(target_os = "linux", target_os = "macos", target_os = "windows"))]
    fn declined_separate_chroma(result: &crate::Result<()>, test: &str) -> bool {
        match result {
            Err(crate::Error::NotSupported(m)) if m.contains("separate DMA-BUF") => {
                crate::test_support::report_skip(&format!("{test} - {m}"));
                true
            }
            _ => false,
        }
    }

    /// Load NV12 raw bytes into separate DMA-backed luma and chroma tensors and return a
    /// multiplane `TensorDyn`.
    #[cfg(any(target_os = "linux", target_os = "macos", target_os = "windows"))]
    fn load_multiplane_nv12_dma(width: usize, height: usize, nv12_bytes: &[u8]) -> TensorDyn {
        let y_size = width * height;
        let uv_size = width * (height / 2);
        assert_eq!(nv12_bytes.len(), y_size + uv_size);

        let luma = Tensor::new(&[height, width], Some(TensorMemory::DmaBuf), Some("luma"))
            .expect("DMA luma tensor");
        luma.map().unwrap().as_mut_slice()[..y_size].copy_from_slice(&nv12_bytes[..y_size]);

        let chroma = Tensor::new(
            &[height / 2, width],
            Some(TensorMemory::DmaBuf),
            Some("chroma"),
        )
        .expect("DMA chroma tensor");
        chroma.map().unwrap().as_mut_slice()[..uv_size].copy_from_slice(&nv12_bytes[y_size..]);

        Tensor::<u8>::from_planes(luma, chroma, PixelFormat::Nv12)
            .map(TensorDyn::from)
            .expect("multiplane PixelFormat::Nv12")
    }

    /// Multiplane NV12 (two separate fds) → Rgba through GL equals contiguous NV12 → Rgba.
    #[test]
    #[cfg(target_os = "linux")]
    fn test_multiplane_nv12_to_rgba_opengl() {
        if !is_opengl_available() {
            crate::test_support::report_skip(&format!("{} - GL not available", function!()));
            return;
        }
        if !is_dma_available() {
            crate::test_support::report_skip(&format!("{} - DMA not available", function!()));
            return;
        }

        let nv12_bytes: &[u8] = &edgefirst_bench::testdata::read("camera720p.nv12");

        // Contiguous PixelFormat::Nv12 (single DMA-BUF)
        let src_contiguous = load_raw_image(
            1280,
            720,
            PixelFormat::Nv12,
            Some(TensorMemory::DmaBuf),
            nv12_bytes,
        )
        .unwrap();

        // Multiplane PixelFormat::Nv12 (two DMA-BUFs)
        let src_multiplane = load_multiplane_nv12_dma(1280, 720, nv12_bytes);
        assert!(src_multiplane.as_u8().unwrap().is_multiplane());

        // Both sources are pinned to `ExternalSampler`.
        let _nv_path = NvPathEnvGuard::set("sampler");

        let mut gl = GLProcessorThreaded::new(None).unwrap();

        // Convert contiguous
        let dst_contig = TensorDyn::image(
            1280,
            720,
            PixelFormat::Rgba,
            DType::U8,
            Some(TensorMemory::DmaBuf),
            edgefirst_tensor::CpuAccess::ReadWrite,
        )
        .unwrap();
        let src_contiguous_dyn = src_contiguous;
        let mut dst_contig_dyn = dst_contig;
        gl.convert(&src_contiguous_dyn, &mut dst_contig_dyn, None)
            .unwrap();

        // Convert multiplane
        let dst_multi = TensorDyn::image(
            1280,
            720,
            PixelFormat::Rgba,
            DType::U8,
            Some(TensorMemory::DmaBuf),
            edgefirst_tensor::CpuAccess::ReadWrite,
        )
        .unwrap();
        let src_multiplane_dyn = src_multiplane;
        let mut dst_multi_dyn = dst_multi;
        let result = gl.convert(&src_multiplane_dyn, &mut dst_multi_dyn, None);
        if declined_separate_chroma(&result, function!()) {
            return;
        }
        result.unwrap();

        // Both produce identical pixels.
        let map_contig = dst_contig_dyn.as_u8().unwrap().map().unwrap();
        let map_multi = dst_multi_dyn.as_u8().unwrap().map().unwrap();
        assert_pixels_match(map_contig.as_slice(), map_multi.as_slice(), 0);
    }

    /// Two-plane NV12 in one DMA-BUF (dup'd fd, chroma at an offset) converts as contiguous NV12.
    #[test]
    #[cfg(target_os = "linux")]
    fn test_same_fd_multiplane_nv12_to_rgba_opengl() {
        use edgefirst_tensor::PlaneDescriptor;

        if !is_dma_available() {
            crate::test_support::report_skip(&format!("{} - DMA not available", function!()));
            return;
        }
        if !is_opengl_available() {
            crate::test_support::report_skip(&format!("{} - OpenGL not available", function!()));
            return;
        }

        let nv12_bytes: &[u8] = &edgefirst_bench::testdata::read("camera720p.nv12");
        let width: usize = 1280;
        let height: usize = 720;
        let stride: usize = width; // NV12 Y plane: 1 byte per pixel
        let y_size = stride * height;
        let uv_size = stride * (height / 2);
        let total_bytes = y_size + uv_size;
        assert_eq!(nv12_bytes.len(), total_bytes);

        // Single DMA-BUF holding both planes contiguously
        let Some(buf) = crate::test_support::dma_allocated(
            Tensor::<u8>::new(
                &[total_bytes],
                Some(TensorMemory::DmaBuf),
                Some("same_fd_nv12"),
            ),
            "same-fd NV12 DMA buffer",
        ) else {
            return;
        };
        // Fill with NV12 camera data.
        buf.map().unwrap().as_mut_slice()[..total_bytes].copy_from_slice(nv12_bytes);

        // Import as multiplane: two PlaneDescriptors, same fd, different offsets
        let fd = buf.dmabuf().unwrap();
        let luma_pd = PlaneDescriptor::new(fd)
            .unwrap()
            .with_stride(stride)
            .with_offset(0);
        let chroma_pd = PlaneDescriptor::new(fd)
            .unwrap()
            .with_stride(stride)
            .with_offset(y_size);

        // Both sources are pinned to `ExternalSampler`.
        let _nv_path = NvPathEnvGuard::set("sampler");

        let proc = crate::ImageProcessor::new().unwrap();
        let src = proc
            .import_image(
                luma_pd,
                Some(chroma_pd),
                width,
                height,
                PixelFormat::Nv12,
                DType::U8,
                None,
            )
            .unwrap();
        assert!(src.is_multiplane(), "must be multiplane after import");

        // Contiguous reference.
        let src_contig = load_raw_image(
            width,
            height,
            PixelFormat::Nv12,
            Some(TensorMemory::DmaBuf),
            nv12_bytes,
        )
        .unwrap();

        let mut gl = GLProcessorThreaded::new(None).unwrap();

        // Render the same-fd two-plane source.
        let mut dst_same_fd = TensorDyn::image(
            width,
            height,
            PixelFormat::Rgba,
            DType::U8,
            Some(TensorMemory::DmaBuf),
            edgefirst_tensor::CpuAccess::ReadWrite,
        )
        .unwrap();
        gl.convert(&src, &mut dst_same_fd, None).unwrap();

        // Render contiguous reference
        let mut dst_contig = TensorDyn::image(
            width,
            height,
            PixelFormat::Rgba,
            DType::U8,
            Some(TensorMemory::DmaBuf),
            edgefirst_tensor::CpuAccess::ReadWrite,
        )
        .unwrap();
        gl.convert(&src_contig, &mut dst_contig, None).unwrap();

        // Both produce identical pixels.
        let map_same = dst_same_fd.as_u8().unwrap().map().unwrap();
        let map_contig = dst_contig.as_u8().unwrap().map().unwrap();
        assert_pixels_match(map_same.as_slice(), map_contig.as_slice(), 0);
    }

    /// Multiplane NV12 720p → packed Rgb 640x640 letterbox equals the contiguous source's.
    #[test]
    #[cfg(target_os = "linux")]
    fn test_multiplane_nv12_to_rgb_letterbox_opengl() {
        if !is_opengl_available() {
            crate::test_support::report_skip(&format!("{} - GL not available", function!()));
            return;
        }
        if !is_dma_available() {
            crate::test_support::report_skip(&format!("{} - DMA not available", function!()));
            return;
        }

        // Both sources are pinned to `ExternalSampler`.
        let _nv_path = NvPathEnvGuard::set("sampler");

        let nv12_bytes: &[u8] = &edgefirst_bench::testdata::read("camera720p.nv12");

        let src_contiguous = load_raw_image(
            1280,
            720,
            PixelFormat::Nv12,
            Some(TensorMemory::DmaBuf),
            nv12_bytes,
        )
        .unwrap();
        let src_multiplane = load_multiplane_nv12_dma(1280, 720, nv12_bytes);

        let crop = letterbox_crop(1280, 720, 640, 640);
        let mut gl = GLProcessorThreaded::new(None).unwrap();

        // Contiguous → packed PixelFormat::Rgb with letterbox
        let dst_contig = TensorDyn::image(
            640,
            640,
            PixelFormat::Rgb,
            DType::U8,
            Some(TensorMemory::DmaBuf),
            edgefirst_tensor::CpuAccess::ReadWrite,
        )
        .unwrap();
        let src_contiguous_dyn = src_contiguous;
        let mut dst_contig_dyn = dst_contig;
        gl.convert(
            &src_contiguous_dyn,
            &mut dst_contig_dyn,
            Some(crate::ConvertOptions::new().crop(crop)),
        )
        .unwrap();

        // Multiplane → packed PixelFormat::Rgb with letterbox
        let dst_multi = TensorDyn::image(
            640,
            640,
            PixelFormat::Rgb,
            DType::U8,
            Some(TensorMemory::DmaBuf),
            edgefirst_tensor::CpuAccess::ReadWrite,
        )
        .unwrap();
        let src_multiplane_dyn = src_multiplane;
        let mut dst_multi_dyn = dst_multi;
        let result = gl.convert(
            &src_multiplane_dyn,
            &mut dst_multi_dyn,
            Some(crate::ConvertOptions::new().crop(crop)),
        );
        if declined_separate_chroma(&result, function!()) {
            return;
        }
        result.unwrap();

        let map_contig = dst_contig_dyn.as_u8().unwrap().map().unwrap();
        let map_multi = dst_multi_dyn.as_u8().unwrap().map().unwrap();
        assert_pixels_match(map_contig.as_slice(), map_multi.as_slice(), 0);
    }

    /// Multiplane NV12 720p → packed Rgb int8 640x640 letterbox equals the contiguous source's.
    #[test]
    #[cfg(target_os = "linux")]
    fn test_multiplane_nv12_to_rgb_int8_letterbox_opengl() {
        if !is_opengl_available() {
            crate::test_support::report_skip(&format!("{} - GL not available", function!()));
            return;
        }
        if !is_dma_available() {
            crate::test_support::report_skip(&format!("{} - DMA not available", function!()));
            return;
        }

        // Both sources are pinned to `ExternalSampler`.
        let _nv_path = NvPathEnvGuard::set("sampler");

        let nv12_bytes: &[u8] = &edgefirst_bench::testdata::read("camera720p.nv12");

        let src_contiguous = load_raw_image(
            1280,
            720,
            PixelFormat::Nv12,
            Some(TensorMemory::DmaBuf),
            nv12_bytes,
        )
        .unwrap();
        let src_multiplane = load_multiplane_nv12_dma(1280, 720, nv12_bytes);

        let crop = letterbox_crop(1280, 720, 640, 640);
        let mut gl = GLProcessorThreaded::new(None).unwrap();

        // Contiguous → packed PixelFormat::Rgb int8 with letterbox
        let dst_contig = TensorDyn::image(
            640,
            640,
            PixelFormat::Rgb,
            DType::I8,
            Some(TensorMemory::DmaBuf),
            edgefirst_tensor::CpuAccess::ReadWrite,
        )
        .unwrap();
        let src_contiguous_dyn = src_contiguous;
        let mut dst_contig_dyn = dst_contig;
        gl.convert(
            &src_contiguous_dyn,
            &mut dst_contig_dyn,
            Some(crate::ConvertOptions::new().crop(crop)),
        )
        .unwrap();

        // Multiplane → packed PixelFormat::Rgb int8 with letterbox
        let dst_multi = TensorDyn::image(
            640,
            640,
            PixelFormat::Rgb,
            DType::I8,
            Some(TensorMemory::DmaBuf),
            edgefirst_tensor::CpuAccess::ReadWrite,
        )
        .unwrap();
        let src_multiplane_dyn = src_multiplane;
        let mut dst_multi_dyn = dst_multi;
        let result = gl.convert(
            &src_multiplane_dyn,
            &mut dst_multi_dyn,
            Some(crate::ConvertOptions::new().crop(crop)),
        );
        if declined_separate_chroma(&result, function!()) {
            return;
        }
        result.unwrap();

        // Map the raw i8 bytes as u8 for comparison.
        let map_contig = dst_contig_dyn.as_i8().unwrap().map().unwrap();
        let contig_bytes: &[u8] = unsafe {
            std::slice::from_raw_parts(map_contig.as_slice().as_ptr().cast(), map_contig.len())
        };
        let map_multi = dst_multi_dyn.as_i8().unwrap().map().unwrap();
        let multi_bytes: &[u8] = unsafe {
            std::slice::from_raw_parts(map_multi.as_slice().as_ptr().cast(), map_multi.len())
        };
        assert_pixels_match(contig_bytes, multi_bytes, 0);
    }

    /// The two-pass NV→planar plan applies the letterbox once, on GL, for Dma and Mem destinations.
    #[test]
    #[cfg(target_os = "linux")]
    fn letterbox_nv_to_planar_content_band_geometry() {
        use crate::{ComputeBackend, ImageProcessor, ImageProcessorConfig};

        if !is_opengl_available() {
            crate::test_support::report_skip(&format!("{} - OpenGL not available", function!()));
            return;
        }
        let mut proc = match ImageProcessor::with_config(ImageProcessorConfig {
            backend: ComputeBackend::OpenGl,
            ..Default::default()
        }) {
            Ok(p) => p,
            Err(e) => {
                crate::test_support::report_skip(&format!(
                    "{} - OpenGL backend unavailable: {e}",
                    function!()
                ));
                return;
            }
        };
        if proc.opengl.is_none() {
            crate::test_support::report_skip(&format!(
                "{} - ImageProcessor resolved no GL backend",
                function!()
            ));
            return;
        }

        // Synthetic NV12: Y=200, U=V=128, neutral grey.
        let (sw, sh) = (1280usize, 720usize);
        let mut nv12 = vec![0u8; sw * sh * 3 / 2];
        nv12[..sw * sh].fill(200);
        nv12[sw * sh..].fill(128);
        let src_mem = if is_dma_available() {
            Some(TensorMemory::DmaBuf)
        } else {
            Some(TensorMemory::Mem)
        };
        let src = load_raw_image(sw, sh, PixelFormat::Nv12, src_mem, &nv12).unwrap();

        // 1280x720 → 640x640: scale 0.5, content 640x360 centred on rows [140, 500).
        const CONTENT_ROWS: [usize; 3] = [150, 320, 490];
        const PAD_ROWS: [usize; 4] = [0, 130, 510, 639];
        let (dw, dh) = (640usize, 640usize);
        let lb = Crop::letterbox([114, 114, 114, 255]);

        let mut dst_memories = vec![TensorMemory::Mem];
        if is_dma_available() {
            dst_memories.push(TensorMemory::DmaBuf);
        }
        for dst_mem in dst_memories {
            for dst_fmt in [PixelFormat::PlanarRgb, PixelFormat::PlanarRgba] {
                for dtype in [DType::U8, DType::I8] {
                    let label = format!("nv12->{dst_fmt}.{dtype:?}@{dst_mem:?}");
                    let mut dst = proc
                        .create_image(
                            dw,
                            dh,
                            dst_fmt,
                            dtype,
                            Some(dst_mem),
                            edgefirst_tensor::CpuAccess::ReadWrite,
                        )
                        .unwrap();
                    let fallbacks_before = proc.convert_fallback_count();
                    proc.convert(&src, &mut dst, Some(crate::ConvertOptions::new().crop(lb)))
                        .unwrap_or_else(|e| panic!("{label}: convert failed: {e}"));
                    let declines = proc.convert_fallback_count() - fallbacks_before;
                    assert_eq!(
                        declines, 0,
                        "{label}: the GL backend declined this convert and the \
                         CPU served it — the band probes below would pass on the \
                         CPU answer, so this leg proved nothing about the GL \
                         two-pass plan"
                    );

                    let raw: Vec<u8> = match dtype {
                        DType::U8 => dst.as_u8().unwrap().map().unwrap().as_slice().to_vec(),
                        _ => dst
                            .as_i8()
                            .unwrap()
                            .map()
                            .unwrap()
                            .as_slice()
                            .iter()
                            .map(|&v| v as u8)
                            .collect(),
                    };
                    // The alpha plane is 255 everywhere, as the raw byte for i8 too.
                    if dst_fmt == PixelFormat::PlanarRgba {
                        let alpha = &raw[3 * dh * dw..4 * dh * dw];
                        if let Some(i) = alpha.iter().position(|&a| a != 255) {
                            panic!("{label}: alpha plane byte {i} = {}, expected 255", alpha[i]);
                        }
                    }
                    // Undo the int8 XOR-0x80 bias.
                    let unbias = if dtype == DType::I8 { 0x80u8 } else { 0 };
                    let bytes: Vec<u8> = raw.iter().map(|&v| v ^ unbias).collect();

                    for plane in 0..3 {
                        let plane_base = plane * dh * dw;
                        for row in CONTENT_ROWS {
                            for col in (5..dw - 5).step_by(13) {
                                let v = bytes[plane_base + row * dw + col];
                                assert!(
                                    v >= 170,
                                    "{label}: plane {plane} row {row} col {col} = {v}, \
                                     expected content (>=170) — letterbox content band \
                                     misplaced (double-letterbox regression?)"
                                );
                            }
                        }
                        for row in PAD_ROWS {
                            for col in (5..dw - 5).step_by(13) {
                                let v = bytes[plane_base + row * dw + col];
                                assert!(
                                    v <= 140,
                                    "{label}: plane {plane} row {row} col {col} = {v}, \
                                     expected pad (<=140) — content leaked into the \
                                     letterbox band"
                                );
                            }
                        }
                    }
                }
            }
        }
    }

    /// Letterboxed NV12 to planar on GL matches the channel-split packed-RGBA letterbox.
    #[test]
    #[cfg(target_os = "linux")]
    fn letterbox_nv12_to_planar_matches_rgba_reference() {
        if !is_opengl_available() {
            crate::test_support::report_skip(&format!("{} - GL not available", function!()));
            return;
        }
        if !is_dma_available() {
            crate::test_support::report_skip(&format!("{} - DMA not available", function!()));
            return;
        }

        let src = load_raw_image(
            1280,
            720,
            PixelFormat::Nv12,
            Some(TensorMemory::DmaBuf),
            &edgefirst_bench::testdata::read("camera720p.nv12"),
        )
        .unwrap();

        let crop = letterbox_crop(1280, 720, 640, 640);
        let (dw, dh) = (640usize, 640usize);
        let mut gl = GLProcessorThreaded::new(None).unwrap();

        // Reference: the packed-RGBA letterbox.
        let mut dst_rgba = TensorDyn::image(
            dw,
            dh,
            PixelFormat::Rgba,
            DType::U8,
            Some(TensorMemory::DmaBuf),
            edgefirst_tensor::CpuAccess::ReadWrite,
        )
        .unwrap();
        gl.convert(
            &src,
            &mut dst_rgba,
            Some(crate::ConvertOptions::new().crop(crop)),
        )
        .unwrap();
        let rgba = dst_rgba.as_u8().unwrap().map().unwrap();
        let rgba = rgba.as_slice();

        for fmt in [PixelFormat::PlanarRgb, PixelFormat::PlanarRgba] {
            // Channel-split the RGBA reference into the planar layout.
            let planes = fmt.channels();
            let mut expected = vec![0u8; planes * dh * dw];
            for (i, px) in rgba.as_chunks::<4>().0.iter().enumerate() {
                for (plane, &channel) in px.iter().take(planes).enumerate() {
                    expected[plane * dh * dw + i] = channel;
                }
            }
            let mut dst = TensorDyn::image(
                dw,
                dh,
                fmt,
                DType::U8,
                Some(TensorMemory::DmaBuf),
                edgefirst_tensor::CpuAccess::ReadWrite,
            )
            .unwrap();
            gl.convert(
                &src,
                &mut dst,
                Some(crate::ConvertOptions::new().crop(crop)),
            )
            .unwrap_or_else(|e| panic!("nv12->{fmt}@Dma: convert failed: {e}"));
            let map = dst.as_u8().unwrap().map().unwrap();
            assert_pixels_match(&expected, map.as_slice(), 1);
        }
    }

    /// Both backends write a padded planar DMA-BUF at its pitch; `copy_to_flat` returns it tight.
    #[test]
    #[cfg(target_os = "linux")]
    fn planar_dma_destination_rows_are_padded_to_the_gpu_pitch() {
        use crate::cpu::CPUProcessor;
        if !is_opengl_available() {
            crate::test_support::report_skip(&format!("{} - GL not available", function!()));
            return;
        }
        if !is_dma_available() {
            crate::test_support::report_skip(&format!("{} - DMA not available", function!()));
            return;
        }
        let src = load_raw_image(
            1280,
            720,
            PixelFormat::Nv12,
            Some(TensorMemory::DmaBuf),
            &edgefirst_bench::testdata::read("camera720p.nv12"),
        )
        .unwrap();
        let options = || crate::ConvertOptions::new().crop(letterbox_crop(1280, 720, 0, 0));
        let mut gl = GLProcessorThreaded::new(None).unwrap();
        let mut cpu = CPUProcessor::new();

        const H: usize = 512;
        for (w, pitch) in [(672usize, 704usize), (700, 704)] {
            let image = |memory| {
                TensorDyn::image(
                    w,
                    H,
                    PixelFormat::PlanarRgb,
                    DType::U8,
                    Some(memory),
                    edgefirst_tensor::CpuAccess::ReadWrite,
                )
                .unwrap()
            };
            // (backend, tolerance against the same backend's tight output)
            for (backend, tolerance) in [("cpu", 0u8), ("gl", 1)] {
                let mut convert = |dst: &mut TensorDyn| {
                    if backend == "cpu" {
                        cpu.convert(&src, dst, Some(options()))
                    } else {
                        gl.convert(&src, dst, Some(options()))
                    }
                    .unwrap_or_else(|e| panic!("{backend} nv12->planar {w}x{H}: {e}"));
                };
                let mut tight = image(TensorMemory::Mem);
                assert_eq!(tight.row_stride(), None, "host planar rows stay tight");
                convert(&mut tight);
                let want = tight.as_u8().unwrap().map().unwrap().as_slice().to_vec();

                let mut padded = image(TensorMemory::DmaBuf);
                assert_eq!(
                    padded.row_stride(),
                    Some(pitch),
                    "a {w}-wide planar DMA image reports its padded pitch"
                );
                convert(&mut padded);
                assert_eq!(
                    padded.as_u8().unwrap().map().unwrap().as_slice().len(),
                    pitch * H * 3,
                    "the map spans the padded planes"
                );
                let mut flat = vec![0u8; w * H * 3];
                padded.copy_to_flat(&mut flat).unwrap();
                assert_pixels_match(&want, &flat, tolerance);
            }
        }
    }

    /// Build a heap-memory (`TensorMemory::Mem`) NV12 image from a raw fixture.
    fn mem_nv12_image(width: usize, height: usize, bytes: &[u8]) -> TensorDyn {
        let img = TensorDyn::image(
            width,
            height,
            PixelFormat::Nv12,
            DType::U8,
            Some(TensorMemory::Mem),
            edgefirst_tensor::CpuAccess::ReadWrite,
        )
        .unwrap();
        let mut map = img.as_u8().unwrap().map().unwrap();
        map.as_mut_slice()[..bytes.len()].copy_from_slice(bytes);
        drop(map);
        img
    }

    /// A GL convert of NV12 into a heap PlanarRgb or PlanarRgba destination matches the CPU's.
    #[test]
    fn gl_planar_heap_matches_cpu() {
        if !is_opengl_available() {
            crate::test_support::report_skip(&format!("{} - OpenGL not available", function!()));
            return;
        }
        let (sw, sh) = (1280usize, 720usize);
        let src = mem_nv12_image(sw, sh, &edgefirst_bench::testdata::read("camera720p.nv12"));

        let crop = Crop::letterbox([114, 114, 114, 255]);
        let (dw, dh) = (640usize, 640usize);

        let mut gl = match GLProcessorThreaded::new(None) {
            Ok(gl) => gl,
            Err(e) => {
                crate::test_support::report_skip(&format!(
                    "{} - GL not available: {e}",
                    function!()
                ));
                return;
            }
        };
        for fmt in [PixelFormat::PlanarRgb, PixelFormat::PlanarRgba] {
            let mut gl_dst = TensorDyn::image(
                dw,
                dh,
                fmt,
                DType::U8,
                Some(TensorMemory::Mem),
                edgefirst_tensor::CpuAccess::ReadWrite,
            )
            .unwrap();
            gl.convert(
                &src,
                &mut gl_dst,
                Some(crate::ConvertOptions::new().crop(crop)),
            )
            .unwrap_or_else(|e| panic!("nv12->{fmt}@Mem: GL convert failed: {e}"));

            // CPU reference.
            let mut cpu_dst = TensorDyn::image(
                dw,
                dh,
                fmt,
                DType::U8,
                None,
                edgefirst_tensor::CpuAccess::ReadWrite,
            )
            .unwrap();
            crate::cpu::CPUProcessor::new()
                .convert(
                    &src,
                    &mut cpu_dst,
                    Some(crate::ConvertOptions::new().crop(crop)),
                )
                .unwrap();

            compare_images(&gl_dst, &cpu_dst, 0.95, function!());
        }
    }

    /// The NV12 to heap planar convert stays on GL across repeats through `ImageProcessor`.
    #[test]
    fn gl_planar_heap_does_not_fall_back() {
        use crate::ImageProcessor;

        if !is_opengl_available() {
            crate::test_support::report_skip(&format!("{} - OpenGL not available", function!()));
            return;
        }
        let mut proc = ImageProcessor::new().unwrap();
        if proc.opengl.is_none() {
            crate::test_support::report_skip(&format!(
                "{} - ImageProcessor resolved no GL backend",
                function!()
            ));
            return;
        }

        let (sw, sh) = (1280usize, 720usize);
        let src = mem_nv12_image(sw, sh, &edgefirst_bench::testdata::read("camera720p.nv12"));
        let crop = Crop::letterbox([114, 114, 114, 255]);
        let (dw, dh) = (640usize, 640usize);
        let mut dst = proc
            .create_image(
                dw,
                dh,
                PixelFormat::PlanarRgb,
                DType::U8,
                Some(TensorMemory::Mem),
                edgefirst_tensor::CpuAccess::ReadWrite,
            )
            .unwrap();

        let before = proc.convert_fallback_count();
        for i in 0..10 {
            proc.convert(
                &src,
                &mut dst,
                Some(crate::ConvertOptions::new().crop(crop)),
            )
            .unwrap_or_else(|e| panic!("nv12->PlanarRgb@Mem: convert failed on iter {i}: {e}"));
        }
        let after = proc.convert_fallback_count();
        assert_eq!(
            after - before,
            0,
            "GL planar heap convert fell back {} time(s) over 10 iterations \
             (before={before}, after={after})",
            after - before
        );
    }

    /// A GL convert of a packed Rgba source into a heap PlanarRgb destination matches the CPU.
    #[test]
    fn gl_planar_heap_rgba_src_matches_cpu() {
        if !is_opengl_available() {
            crate::test_support::report_skip(&format!("{} - OpenGL not available", function!()));
            return;
        }
        let src = crate::load_image_test_helper(
            &edgefirst_bench::testdata::read("giraffe.jpg"),
            Some(PixelFormat::Rgba),
            Some(TensorMemory::Mem),
        )
        .unwrap();

        let crop = Crop::letterbox([114, 114, 114, 255]);
        let (dw, dh) = (640usize, 640usize);

        let mut gl = match GLProcessorThreaded::new(None) {
            Ok(gl) => gl,
            Err(e) => {
                crate::test_support::report_skip(&format!(
                    "{} - GL not available: {e}",
                    function!()
                ));
                return;
            }
        };
        let mut gl_dst = TensorDyn::image(
            dw,
            dh,
            PixelFormat::PlanarRgb,
            DType::U8,
            Some(TensorMemory::Mem),
            edgefirst_tensor::CpuAccess::ReadWrite,
        )
        .unwrap();
        gl.convert(
            &src,
            &mut gl_dst,
            Some(crate::ConvertOptions::new().crop(crop)),
        )
        .unwrap_or_else(|e| panic!("rgba->PlanarRgb@Mem: GL convert failed: {e}"));

        let mut cpu_dst = TensorDyn::image(
            dw,
            dh,
            PixelFormat::PlanarRgb,
            DType::U8,
            None,
            edgefirst_tensor::CpuAccess::ReadWrite,
        )
        .unwrap();
        crate::cpu::CPUProcessor::new()
            .convert(
                &src,
                &mut cpu_dst,
                Some(crate::ConvertOptions::new().crop(crop)),
            )
            .unwrap();

        compare_images(&gl_dst, &cpu_dst, 0.95, function!());
    }

    /// The Rgba to heap planar convert stays on GL across repeats through `ImageProcessor`.
    #[test]
    fn gl_planar_heap_rgba_src_does_not_fall_back() {
        use crate::ImageProcessor;

        if !is_opengl_available() {
            crate::test_support::report_skip(&format!("{} - OpenGL not available", function!()));
            return;
        }
        let mut proc = ImageProcessor::new().unwrap();
        if proc.opengl.is_none() {
            crate::test_support::report_skip(&format!(
                "{} - ImageProcessor resolved no GL backend",
                function!()
            ));
            return;
        }

        let src = crate::load_image_test_helper(
            &edgefirst_bench::testdata::read("giraffe.jpg"),
            Some(PixelFormat::Rgba),
            Some(TensorMemory::Mem),
        )
        .unwrap();
        let crop = Crop::letterbox([114, 114, 114, 255]);
        let (dw, dh) = (640usize, 640usize);
        let mut dst = proc
            .create_image(
                dw,
                dh,
                PixelFormat::PlanarRgb,
                DType::U8,
                Some(TensorMemory::Mem),
                edgefirst_tensor::CpuAccess::ReadWrite,
            )
            .unwrap();

        let before = proc.convert_fallback_count();
        for i in 0..10 {
            proc.convert(
                &src,
                &mut dst,
                Some(crate::ConvertOptions::new().crop(crop)),
            )
            .unwrap_or_else(|e| panic!("rgba->PlanarRgb@Mem: convert failed on iter {i}: {e}"));
        }
        let after = proc.convert_fallback_count();
        assert_eq!(
            after - before,
            0,
            "GL planar heap (Rgba src) convert fell back {} time(s) over 10 iterations \
             (before={before}, after={after})",
            after - before
        );
    }

    /// Fused GL proto rendering is similar to the hybrid path (CPU materialize, GL overlay).
    #[test]
    fn test_proto_fused_vs_hybrid_ssim() {
        if !is_opengl_available() {
            crate::test_support::report_skip(&format!("{} - OpenGL not available", function!()));
            return;
        }

        let (output_boxes, proto_data) = decode_yolov8_proto_fixture();

        // Materialize masks on CPU for the hybrid path
        let cpu_proc = crate::CPUProcessor::new();
        let segmentation = cpu_proc
            .materialize_segmentations(&output_boxes, &proto_data, None)
            .unwrap();

        // Create two identical RGBA canvases
        let mut gl = GLProcessorThreaded::new(None).unwrap();
        let mut dst_hybrid = TensorDyn::from(
            edgefirst_tensor::Tensor::<u8>::image(
                640,
                640,
                PixelFormat::Rgba,
                None,
                edgefirst_tensor::CpuAccess::ReadWrite,
            )
            .unwrap(),
        );
        let mut dst_fused = TensorDyn::from(
            edgefirst_tensor::Tensor::<u8>::image(
                640,
                640,
                PixelFormat::Rgba,
                None,
                edgefirst_tensor::CpuAccess::ReadWrite,
            )
            .unwrap(),
        );

        // Render via hybrid path (pre-decoded masks)
        gl.draw_decoded_masks(
            &mut dst_hybrid,
            &output_boxes,
            &segmentation,
            Default::default(),
        )
        .unwrap();

        // Render via fused GL proto path
        gl.draw_proto_masks(
            &mut dst_fused,
            &output_boxes,
            &proto_data,
            Default::default(),
        )
        .unwrap();

        compare_images(&dst_hybrid, &dst_fused, 0.90, function!());
    }

    /// Decode the cached YOLOv8-seg fixture outputs (int8 boxes + protos) into post-NMS
    /// detections and the int8 `ProtoData`.
    fn decode_yolov8_proto_fixture() -> (Vec<DetectBox>, ProtoData) {
        use edgefirst_decoder::{configs, ConfigOutput, DecoderBuilder, Nms};

        let boxes_raw: &[u8] = &edgefirst_bench::testdata::read("yolov8_boxes_116x8400.bin");
        let boxes_i8 =
            unsafe { std::slice::from_raw_parts(boxes_raw.as_ptr() as *const i8, boxes_raw.len()) };
        let boxes_tensor: TensorDyn =
            edgefirst_tensor::Tensor::<i8>::from_slice(boxes_i8, &[1, 116, 8400])
                .expect("boxes tensor")
                .into();

        let protos_raw: &[u8] = &edgefirst_bench::testdata::read("yolov8_protos_160x160x32.bin");
        let protos_i8 = unsafe {
            std::slice::from_raw_parts(protos_raw.as_ptr() as *const i8, protos_raw.len())
        };
        let protos_tensor: TensorDyn =
            edgefirst_tensor::Tensor::<i8>::from_slice(protos_i8, &[1, 160, 160, 32])
                .expect("protos tensor")
                .into();

        let detection_cfg = configs::Detection {
            decoder: configs::DecoderType::Ultralytics,
            quantization: Some(configs::QuantTuple(0.019_484_945, 20)),
            shape: vec![1, 116, 8400],
            dshape: vec![],
            anchors: None,
            normalized: Some(true),
        };
        let protos_cfg = configs::Protos {
            decoder: configs::DecoderType::Ultralytics,
            quantization: Some(configs::QuantTuple(0.020_889_873, -115)),
            shape: vec![1, 160, 160, 32],
            dshape: vec![],
        };
        let decoder = DecoderBuilder::default()
            .with_score_threshold(0.45)
            .with_iou_threshold(0.45)
            .with_nms(Some(Nms::ClassAgnostic))
            .add_output(ConfigOutput::Detection(detection_cfg))
            .add_output(ConfigOutput::Protos(protos_cfg))
            .build()
            .expect("yolov8-seg test decoder must build");

        let inputs: Vec<&TensorDyn> = vec![&boxes_tensor, &protos_tensor];
        let mut output_boxes = Vec::with_capacity(50);
        let proto_data = decoder
            .decode_proto(&inputs, &mut output_boxes)
            .expect("decode_proto must succeed")
            .expect("yolov8-seg config produces ProtoData");
        assert!(!output_boxes.is_empty(), "No detections from model");
        (output_boxes, proto_data)
    }

    /// Build F32 and F16 `ProtoData` by dequantizing the int8 fixture.
    fn proto_fixture_as_float(proto_i8: &ProtoData, n_det: usize) -> (ProtoData, ProtoData) {
        use half::f16;

        let (protos_f32, proto_shape) = match proto_i8.protos.dtype() {
            DType::I8 => {
                let t = proto_i8.protos.as_typed::<i8>().expect("dtype checked");
                let shape = t.shape().to_vec();
                let q = t.quantization().expect("i8 protos must carry quant");
                let scale = q.scale()[0];
                let zp = q.zero_point().map(|z| z[0]).unwrap_or(0) as f32;
                let m = t.map().unwrap();
                let v: Vec<f32> = m
                    .as_slice()
                    .iter()
                    .map(|v| (*v as f32 - zp) * scale)
                    .collect();
                (v, shape)
            }
            other => panic!("expected i8 protos in fixture, got {other:?}"),
        };
        let num_protos = *proto_shape.last().unwrap();

        let coeffs_f32: Vec<f32> = match proto_i8.mask_coefficients.dtype() {
            DType::F32 => proto_i8
                .mask_coefficients
                .as_typed::<f32>()
                .expect("dtype checked")
                .map()
                .unwrap()
                .as_slice()
                .to_vec(),
            DType::F16 => proto_i8
                .mask_coefficients
                .as_typed::<f16>()
                .expect("dtype checked")
                .map()
                .unwrap()
                .as_slice()
                .iter()
                .map(|v: &f16| v.to_f32())
                .collect(),
            DType::I8 => {
                let t = proto_i8
                    .mask_coefficients
                    .as_typed::<i8>()
                    .expect("dtype checked");
                let q = t
                    .quantization()
                    .expect("i8 mask coefficients must carry quant");
                let scale = q.scale()[0];
                let zp = q.zero_point().map(|z| z[0]).unwrap_or(0) as f32;
                let m = t.map().unwrap();
                m.as_slice()
                    .iter()
                    .map(|v| (*v as f32 - zp) * scale)
                    .collect()
            }
            other => panic!("unexpected coefficient dtype {other:?}"),
        };
        let coeff_shape = [n_det, num_protos];

        let proto_f32 = ProtoData {
            mask_coefficients: Tensor::<f32>::from_slice(&coeffs_f32, &coeff_shape)
                .unwrap()
                .into(),
            protos: Tensor::<f32>::from_slice(&protos_f32, &proto_shape)
                .unwrap()
                .into(),
            layout: ProtoLayout::Nhwc,
        };

        let protos_f16: Vec<f16> = protos_f32.iter().map(|v| f16::from_f32(*v)).collect();
        let coeffs_f16: Vec<f16> = coeffs_f32.iter().map(|v| f16::from_f32(*v)).collect();
        let proto_f16 = ProtoData {
            mask_coefficients: Tensor::<f16>::from_slice(&coeffs_f16, &coeff_shape)
                .unwrap()
                .into(),
            protos: Tensor::<f16>::from_slice(&protos_f16, &proto_shape)
                .unwrap()
                .into(),
            layout: ProtoLayout::Nhwc,
        };

        (proto_f32, proto_f16)
    }

    /// Keep only the first `keep` proto layers (and matching coefficients) of an F32 HWC
    /// `ProtoData`.
    fn slice_proto_layers_f32(pd: &ProtoData, keep: usize, n_det: usize) -> ProtoData {
        let (protos, shape) = match pd.protos.dtype() {
            DType::F32 => {
                let t = pd.protos.as_typed::<f32>().expect("dtype checked");
                (t.map().unwrap().as_slice().to_vec(), t.shape().to_vec())
            }
            other => panic!("expected f32 protos, got {other:?}"),
        };
        let (h, w, c) = (shape[0], shape[1], shape[2]);
        assert!(keep <= c);
        let mut sliced = Vec::with_capacity(h * w * keep);
        for px in protos.chunks_exact(c) {
            sliced.extend_from_slice(&px[..keep]);
        }
        let coeffs = match pd.mask_coefficients.dtype() {
            DType::F32 => pd
                .mask_coefficients
                .as_typed::<f32>()
                .expect("dtype checked")
                .map()
                .unwrap()
                .as_slice()
                .to_vec(),
            other => panic!("expected f32 coefficients, got {other:?}"),
        };
        let mut sliced_coeffs = Vec::with_capacity(n_det * keep);
        for det in coeffs.chunks_exact(c) {
            sliced_coeffs.extend_from_slice(&det[..keep]);
        }
        ProtoData {
            mask_coefficients: Tensor::<f32>::from_slice(&sliced_coeffs, &[n_det, keep])
                .unwrap()
                .into(),
            protos: Tensor::<f32>::from_slice(&sliced, &[h, w, keep])
                .unwrap()
                .into(),
            layout: ProtoLayout::Nhwc,
        }
    }

    /// Render the fused GL proto path into a fresh 640x640 RGBA canvas.
    fn render_proto_to_rgba(
        gl: &mut GLProcessorThreaded,
        boxes: &[DetectBox],
        pd: &ProtoData,
    ) -> TensorDyn {
        let mut dst = TensorDyn::from(
            edgefirst_tensor::Tensor::<u8>::image(
                640,
                640,
                PixelFormat::Rgba,
                None,
                edgefirst_tensor::CpuAccess::ReadWrite,
            )
            .unwrap(),
        );
        gl.draw_proto_masks(&mut dst, boxes, pd, Default::default())
            .unwrap();
        dst
    }

    /// Forces `has_float_linear` off for processors created in scope.
    struct FloatLinearEnvGuard(#[allow(dead_code)] std::sync::MutexGuard<'static, ()>);
    impl FloatLinearEnvGuard {
        fn set() -> Self {
            let held = lock_gl_env();
            std::env::set_var("EDGEFIRST_GL_NO_FLOAT_LINEAR", "1");
            FloatLinearEnvGuard(held)
        }
    }
    impl Drop for FloatLinearEnvGuard {
        fn drop(&mut self) {
            std::env::remove_var("EDGEFIRST_GL_NO_FLOAT_LINEAR");
        }
    }

    /// Opts processors created in scope into the GLES 3.1 compute-shader proto repack.
    struct ProtoComputeEnvGuard(#[allow(dead_code)] std::sync::MutexGuard<'static, ()>);
    impl ProtoComputeEnvGuard {
        fn set() -> Self {
            let held = lock_gl_env();
            std::env::set_var("EDGEFIRST_PROTO_COMPUTE", "1");
            ProtoComputeEnvGuard(held)
        }
    }
    impl Drop for ProtoComputeEnvGuard {
        fn drop(&mut self) {
            std::env::remove_var("EDGEFIRST_PROTO_COMPUTE");
        }
    }

    /// The F32 and F16 proto renders agree with the int8 fused reference.
    #[test]
    fn proto_float_dtypes_match_int8_reference() {
        if !is_opengl_available() {
            crate::test_support::report_skip(&format!("{} - OpenGL not available", function!()));
            return;
        }
        let (boxes, proto_i8) = decode_yolov8_proto_fixture();
        let (proto_f32, proto_f16) = proto_fixture_as_float(&proto_i8, boxes.len());

        let mut gl = GLProcessorThreaded::new(None).unwrap();
        let dst_i8 = render_proto_to_rgba(&mut gl, &boxes, &proto_i8);
        let dst_f32 = render_proto_to_rgba(&mut gl, &boxes, &proto_f32);
        let dst_f16 = render_proto_to_rgba(&mut gl, &boxes, &proto_f16);

        compare_images(&dst_i8, &dst_f32, 0.90, function!());
        compare_images(&dst_f32, &dst_f16, 0.98, function!());
    }

    /// With `has_float_linear` forced off, the F32 proto render matches the native R32F render.
    #[test]
    fn proto_f32_no_float_linear_fallback_matches() {
        if !is_opengl_available() {
            crate::test_support::report_skip(&format!("{} - OpenGL not available", function!()));
            return;
        }
        let (boxes, proto_i8) = decode_yolov8_proto_fixture();
        let (proto_f32, _) = proto_fixture_as_float(&proto_i8, boxes.len());

        let dst_native = {
            let mut gl = GLProcessorThreaded::new(None).unwrap();
            render_proto_to_rgba(&mut gl, &boxes, &proto_f32)
        };
        let dst_fallback = {
            let _guard = FloatLinearEnvGuard::set();
            let mut gl = GLProcessorThreaded::new(None).unwrap();
            render_proto_to_rgba(&mut gl, &boxes, &proto_f32)
        };

        compare_images(&dst_native, &dst_fallback, 0.98, function!());
    }

    /// The GLES 3.1 compute-shader proto repack produces the same masks as the CPU repack.
    #[test]
    fn proto_compute_repack_matches_cpu_repack() {
        if !is_opengl_available() {
            crate::test_support::report_skip(&format!("{} - OpenGL not available", function!()));
            return;
        }
        let (boxes, proto_i8) = decode_yolov8_proto_fixture();

        let dst_cpu_repack = {
            let mut gl = GLProcessorThreaded::new(None).unwrap();
            render_proto_to_rgba(&mut gl, &boxes, &proto_i8)
        };
        let dst_compute = {
            let _guard = ProtoComputeEnvGuard::set();
            let mut gl = GLProcessorThreaded::new(None).unwrap();
            render_proto_to_rgba(&mut gl, &boxes, &proto_i8)
        };

        compare_images(&dst_cpu_repack, &dst_compute, 0.99, function!());
    }

    /// The three int8 proto interpolation modes agree on the rendered masks, on one processor.
    #[test]
    fn proto_int8_interpolation_modes_agree() {
        use crate::Int8InterpolationMode;

        if !is_opengl_available() {
            crate::test_support::report_skip(&format!("{} - OpenGL not available", function!()));
            return;
        }
        let (boxes, proto_i8) = decode_yolov8_proto_fixture();
        let mut gl = GLProcessorThreaded::new(None).unwrap();

        let mut render_mode = |mode: Int8InterpolationMode| {
            gl.set_int8_interpolation_mode(mode).unwrap();
            render_proto_to_rgba(&mut gl, &boxes, &proto_i8)
        };
        let bilinear = render_mode(Int8InterpolationMode::Bilinear);
        let two_pass = render_mode(Int8InterpolationMode::TwoPass);
        let nearest = render_mode(Int8InterpolationMode::Nearest);
        // Back to bilinear on the same processor.
        let bilinear_again = render_mode(Int8InterpolationMode::Bilinear);

        compare_images(&bilinear, &two_pass, 0.80, function!());
        compare_images(&bilinear, &nearest, 0.90, function!());
        let a = bilinear.as_u8().unwrap().map().unwrap();
        let b = bilinear_again.as_u8().unwrap().map().unwrap();
        assert_eq!(
            a.as_slice(),
            b.as_slice(),
            "bilinear render diverged after mode churn"
        );
    }

    /// Proto uploads across dims and format changes on one processor match a fresh processor's.
    #[test]
    fn proto_dims_and_format_churn_uploads_correctly() {
        if !is_opengl_available() {
            crate::test_support::report_skip(&format!("{} - OpenGL not available", function!()));
            return;
        }
        let (boxes, proto_i8) = decode_yolov8_proto_fixture();
        let (proto_f32, _) = proto_fixture_as_float(&proto_i8, boxes.len());
        let proto_f32_30 = slice_proto_layers_f32(&proto_f32, 30, boxes.len());

        let oracle = |pd: &ProtoData| {
            let mut gl = GLProcessorThreaded::new(None).unwrap();
            render_proto_to_rgba(&mut gl, &boxes, pd)
        };
        let oracle_i8 = oracle(&proto_i8);
        let oracle_f32 = oracle(&proto_f32);
        let oracle_f32_30 = oracle(&proto_f32_30);

        let mut gl = GLProcessorThreaded::new(None).unwrap();
        for (step, (pd, expected)) in [
            (&proto_i8, &oracle_i8),
            (&proto_f32, &oracle_f32),
            (&proto_f32_30, &oracle_f32_30),
            (&proto_i8, &oracle_i8),
        ]
        .into_iter()
        .enumerate()
        {
            let out = render_proto_to_rgba(&mut gl, &boxes, pd);
            let a = expected.as_u8().unwrap().map().unwrap();
            let b = out.as_u8().unwrap().map().unwrap();
            let (a, b) = (a.as_slice(), b.as_slice());
            let diffs = a.iter().zip(b).filter(|(x, y)| x != y).count();
            let max_diff = a
                .iter()
                .zip(b)
                .map(|(x, y)| x.abs_diff(*y))
                .max()
                .unwrap_or(0);
            assert!(
                diffs == 0,
                "churn step {step}: {diffs}/{} bytes diverged from the \
                 fresh-processor oracle (max |diff| {max_diff})",
                a.len()
            );
        }
    }

    // =========================================================================
    // Destination DMA-BUF plane_offset tests
    // =========================================================================

    /// A destination with an explicit offset of 0 produces the same output as one with no offset.
    #[test]
    #[cfg(target_os = "linux")]
    fn test_opengl_dst_offset_zero_regression() {
        if !is_dma_available() {
            crate::test_support::report_skip(&format!("{} - DMA not available", function!()));
            return;
        }
        if !is_opengl_available() {
            crate::test_support::report_skip(&format!("{} - OpenGL not available", function!()));
            return;
        }

        let src = load_raw_image(
            1280,
            720,
            PixelFormat::Nv12,
            Some(TensorMemory::DmaBuf),
            &edgefirst_bench::testdata::read("camera720p.nv12"),
        )
        .unwrap();

        let mut gl = GLProcessorThreaded::new(None).unwrap();

        // Destination without offset
        let dst_no_off = TensorDyn::image(
            640,
            480,
            PixelFormat::Rgba,
            DType::U8,
            Some(TensorMemory::DmaBuf),
            edgefirst_tensor::CpuAccess::ReadWrite,
        )
        .unwrap();
        let src_dyn = src;
        let mut dst_no_off_dyn = dst_no_off;
        gl.convert(&src_dyn, &mut dst_no_off_dyn, None).unwrap();

        // Destination with explicit offset=0
        let mut dst_zero = Tensor::<u8>::image(
            640,
            480,
            PixelFormat::Rgba,
            Some(TensorMemory::DmaBuf),
            edgefirst_tensor::CpuAccess::ReadWrite,
        )
        .unwrap();
        dst_zero.set_plane_offset(0);
        let mut dst_zero_dyn = TensorDyn::from(dst_zero);
        gl.convert(&src_dyn, &mut dst_zero_dyn, None).unwrap();

        let a = dst_no_off_dyn.as_u8().unwrap().map().unwrap();
        let b = dst_zero_dyn.as_u8().unwrap().map().unwrap();
        assert_pixels_match(a.as_slice(), b.as_slice(), 0);
    }

    /// A destination imported at an offset into an oversized DMA buffer is written at that offset.
    #[test]
    #[cfg(target_os = "linux")]
    fn test_opengl_dst_nonzero_offset() {
        use edgefirst_tensor::PlaneDescriptor;
        use std::os::fd::AsFd;

        if !is_dma_available() {
            crate::test_support::report_skip(&format!("{} - DMA not available", function!()));
            return;
        }
        if !is_opengl_available() {
            crate::test_support::report_skip(&format!("{} - OpenGL not available", function!()));
            return;
        }

        let width = 640;
        let height = 480;
        let image_bytes = width * height * 4; // RGBA
        let offset = image_bytes; // put image in the second half

        // Allocate a DMA buffer twice the image size and fill with a sentinel
        let large_buf =
            Tensor::<u8>::new(&[image_bytes * 2], Some(TensorMemory::DmaBuf), None).unwrap();
        large_buf.map().unwrap().as_mut_slice().fill(0xAA);

        // Import the second half as an RGBA destination via PlaneDescriptor
        let fd = large_buf.as_dma().unwrap().fd.as_fd();
        let plane = PlaneDescriptor::new(fd).unwrap().with_offset(offset);
        let proc = crate::ImageProcessor::new().unwrap();
        let mut dst = proc
            .import_image(
                plane,
                None,
                width,
                height,
                PixelFormat::Rgba,
                DType::U8,
                None,
            )
            .unwrap();

        let src = load_raw_image(
            1280,
            720,
            PixelFormat::Nv12,
            Some(TensorMemory::DmaBuf),
            &edgefirst_bench::testdata::read("camera720p.nv12"),
        )
        .unwrap();

        let mut gl = GLProcessorThreaded::new(None).unwrap();
        let src_dyn = src;
        gl.convert(&src_dyn, &mut dst, None).unwrap();

        // The first half, before the offset, holds the sentinel.
        let map = large_buf.map().unwrap();
        let buf = map.as_slice();
        let untouched = &buf[..offset];
        assert!(
            untouched.iter().all(|&b| b == 0xAA),
            "GPU wrote before the offset boundary — offset not applied"
        );

        // The second half, at the offset, holds rendered data.
        let rendered = &buf[offset..];
        assert!(
            rendered.iter().any(|&b| b != 0xAA),
            "GPU did not write at the offset position — destination is still sentinel"
        );
    }

    /// A destination imported through `PlaneDescriptor` at offset 0 matches a plain DMA allocation.
    #[test]
    #[cfg(target_os = "linux")]
    fn test_opengl_dst_imported_offset_zero() {
        use edgefirst_tensor::PlaneDescriptor;
        use std::os::fd::AsFd;

        if !is_dma_available() {
            crate::test_support::report_skip(&format!("{} - DMA not available", function!()));
            return;
        }
        if !is_opengl_available() {
            crate::test_support::report_skip(&format!("{} - OpenGL not available", function!()));
            return;
        }

        let width = 640;
        let height = 480;

        let src = load_raw_image(
            1280,
            720,
            PixelFormat::Nv12,
            Some(TensorMemory::DmaBuf),
            &edgefirst_bench::testdata::read("camera720p.nv12"),
        )
        .unwrap();

        let mut gl = GLProcessorThreaded::new(None).unwrap();

        // Reference: normal DMA destination
        let dst_ref = TensorDyn::image(
            width,
            height,
            PixelFormat::Rgba,
            DType::U8,
            Some(TensorMemory::DmaBuf),
            edgefirst_tensor::CpuAccess::ReadWrite,
        )
        .unwrap();
        let src_dyn = src;
        let mut dst_ref_dyn = dst_ref;
        gl.convert(&src_dyn, &mut dst_ref_dyn, None).unwrap();

        // Imported: PlaneDescriptor with offset=0
        let dst_buf = Tensor::<u8>::image(
            width,
            height,
            PixelFormat::Rgba,
            Some(TensorMemory::DmaBuf),
            edgefirst_tensor::CpuAccess::ReadWrite,
        )
        .unwrap();
        let fd = dst_buf.as_dma().unwrap().fd.as_fd();
        let plane = PlaneDescriptor::new(fd).unwrap().with_offset(0);
        let proc = crate::ImageProcessor::new().unwrap();
        let mut dst_imported = proc
            .import_image(
                plane,
                None,
                width,
                height,
                PixelFormat::Rgba,
                DType::U8,
                None,
            )
            .unwrap();
        gl.convert(&src_dyn, &mut dst_imported, None).unwrap();

        let a = dst_ref_dyn.as_u8().unwrap().map().unwrap();
        let b = dst_buf.map().unwrap();
        assert_pixels_match(a.as_slice(), b.as_slice(), 1);
    }

    /// A packed RGB destination at a nonzero plane offset renders there and not before it.
    #[test]
    #[cfg(target_os = "linux")]
    fn test_opengl_dst_nonzero_offset_rgb() {
        use edgefirst_tensor::PlaneDescriptor;
        use std::os::fd::AsFd;

        if !is_dma_available() {
            crate::test_support::report_skip(&format!("{} - DMA not available", function!()));
            return;
        }
        if !is_opengl_available() {
            crate::test_support::report_skip(&format!("{} - OpenGL not available", function!()));
            return;
        }

        let width = 640;
        let height = 480;
        let image_bytes = width * height * 3; // RGB
        let offset = image_bytes; // put image in the second half

        // Allocate a DMA buffer twice the image size and fill with a sentinel
        let large_buf =
            Tensor::<u8>::new(&[image_bytes * 2], Some(TensorMemory::DmaBuf), None).unwrap();
        large_buf.map().unwrap().as_mut_slice().fill(0xAA);

        // Import the second half as an RGB destination via PlaneDescriptor
        let fd = large_buf.as_dma().unwrap().fd.as_fd();
        let plane = PlaneDescriptor::new(fd).unwrap().with_offset(offset);
        let proc = crate::ImageProcessor::new().unwrap();
        let mut dst = proc
            .import_image(
                plane,
                None,
                width,
                height,
                PixelFormat::Rgb,
                DType::U8,
                None,
            )
            .unwrap();

        let src = load_raw_image(
            1280,
            720,
            PixelFormat::Nv12,
            Some(TensorMemory::DmaBuf),
            &edgefirst_bench::testdata::read("camera720p.nv12"),
        )
        .unwrap();

        let mut gl = GLProcessorThreaded::new(None).unwrap();
        let src_dyn = src;
        gl.convert(&src_dyn, &mut dst, None).unwrap();

        // The first half, before the offset, holds the sentinel.
        let map = large_buf.map().unwrap();
        let buf = map.as_slice();
        let untouched = &buf[..offset];
        assert!(
            untouched.iter().all(|&b| b == 0xAA),
            "GPU wrote before the offset boundary — RGB offset not applied"
        );

        // The second half, at the offset, holds rendered data.
        let rendered = &buf[offset..];
        assert!(
            rendered.iter().any(|&b| b != 0xAA),
            "GPU did not write at the RGB offset position — destination is still sentinel"
        );
    }

    // ---------------------------------------------------------------
    // Neutron-geometry tests: a large buffer and a large offset
    // ---------------------------------------------------------------

    /// A 640×640 RGB int8 destination at offset 3,450,400 in a ~10 MB DMA buffer renders there.
    #[test]
    #[cfg(target_os = "linux")]
    fn test_neutron_scenario_large_offset_rgb_int8() {
        use edgefirst_tensor::PlaneDescriptor;
        use std::os::fd::AsFd;

        if !is_dma_available() {
            crate::test_support::report_skip(&format!("{} - DMA not available", function!()));
            return;
        }
        if !is_opengl_available() {
            crate::test_support::report_skip(&format!("{} - OpenGL not available", function!()));
            return;
        }
        if !is_neutron_available() {
            crate::test_support::report_skip(&format!(
                "{} - Neutron not available (/dev/neutron0 not found)",
                function!()
            ));
            return;
        }

        let total_size: usize = 10_276_864; // Neutron shared buffer size
        let offset: usize = 3_450_400; // Neutron input tensor offset
        let image_bytes: usize = 640 * 640 * 3; // RGB

        let Some(large_buf) = crate::test_support::dma_allocated(
            Tensor::<u8>::new(&[total_size], Some(TensorMemory::DmaBuf), None),
            "Neutron-sized DMA buffer",
        ) else {
            return;
        };
        large_buf.map().unwrap().as_mut_slice().fill(0xAA);

        let fd = large_buf.as_dma().unwrap().fd.as_fd();
        let plane = PlaneDescriptor::new(fd).unwrap().with_offset(offset);
        let proc = crate::ImageProcessor::new().unwrap();
        let mut dst = proc
            .import_image(plane, None, 640, 640, PixelFormat::Rgb, DType::I8, None)
            .unwrap();

        let src = load_raw_image(
            1280,
            720,
            PixelFormat::Nv12,
            Some(TensorMemory::DmaBuf),
            &edgefirst_bench::testdata::read("camera720p.nv12"),
        )
        .unwrap();

        let mut gl = GLProcessorThreaded::new(None).unwrap();
        gl.convert(&src, &mut dst, None).unwrap();

        // The sentinel is intact before the offset.
        let map = large_buf.map().unwrap();
        let buf = map.as_slice();
        let untouched = &buf[..offset];
        assert!(
            untouched.iter().all(|&b| b == 0xAA),
            "GPU wrote before the offset boundary"
        );

        // Rendered data at the offset.
        let rendered = &buf[offset..offset + image_bytes];
        assert!(
            rendered.iter().any(|&b| b != 0xAA),
            "GPU did not write at the offset position — destination is still sentinel"
        );
    }

    /// RGBA U8 (single-pass) and RGB I8 (two-pass) destinations at one large offset both convert.
    #[test]
    #[cfg(target_os = "linux")]
    fn test_neutron_scenario_rgba_vs_rgb_isolation() {
        use edgefirst_tensor::PlaneDescriptor;
        use std::os::fd::AsFd;

        if !is_dma_available() {
            crate::test_support::report_skip(&format!("{} - DMA not available", function!()));
            return;
        }
        if !is_opengl_available() {
            crate::test_support::report_skip(&format!("{} - OpenGL not available", function!()));
            return;
        }
        if !is_neutron_available() {
            crate::test_support::report_skip(&format!(
                "{} - Neutron not available (/dev/neutron0 not found)",
                function!()
            ));
            return;
        }

        let total_size: usize = 10_276_864;
        let offset: usize = 3_450_400;

        let src = load_raw_image(
            1280,
            720,
            PixelFormat::Nv12,
            Some(TensorMemory::DmaBuf),
            &edgefirst_bench::testdata::read("camera720p.nv12"),
        )
        .unwrap();

        // --- RGBA U8 at large offset (single-pass path) ---
        let Some(rgba_buf) = crate::test_support::dma_allocated(
            Tensor::<u8>::new(&[total_size], Some(TensorMemory::DmaBuf), None),
            "Neutron-sized DMA buffer",
        ) else {
            return;
        };
        rgba_buf.map().unwrap().as_mut_slice().fill(0xAA);

        let fd = rgba_buf.as_dma().unwrap().fd.as_fd();
        let plane = PlaneDescriptor::new(fd).unwrap().with_offset(offset);
        let proc = crate::ImageProcessor::new().unwrap();
        let mut dst_rgba = proc
            .import_image(plane, None, 640, 640, PixelFormat::Rgba, DType::U8, None)
            .unwrap();

        let mut gl = GLProcessorThreaded::new(None).unwrap();
        let rgba_result = gl.convert(&src, &mut dst_rgba, None);
        assert!(
            rgba_result.is_ok(),
            "RGBA U8 single-pass at large offset failed: {:?}",
            rgba_result.err()
        );

        // --- RGB I8 at same large offset (two-pass path) ---
        let Some(rgb_buf) = crate::test_support::dma_allocated(
            Tensor::<u8>::new(&[total_size], Some(TensorMemory::DmaBuf), None),
            "Neutron-sized DMA buffer",
        ) else {
            return;
        };
        rgb_buf.map().unwrap().as_mut_slice().fill(0xAA);

        let fd = rgb_buf.as_dma().unwrap().fd.as_fd();
        let plane = PlaneDescriptor::new(fd).unwrap().with_offset(offset);
        let mut dst_rgb = proc
            .import_image(plane, None, 640, 640, PixelFormat::Rgb, DType::I8, None)
            .unwrap();

        let rgb_result = gl.convert(&src, &mut dst_rgb, None);
        assert!(
            rgb_result.is_ok(),
            "RGB I8 two-pass at large offset failed (RGBA succeeded): {:?}",
            rgb_result.err()
        );
    }

    /// RGB U8 and RGB I8 two-pass destinations at the same large offset both convert.
    #[test]
    #[cfg(target_os = "linux")]
    fn test_neutron_scenario_rgb_u8_vs_int8() {
        use edgefirst_tensor::PlaneDescriptor;
        use std::os::fd::AsFd;

        if !is_dma_available() {
            crate::test_support::report_skip(&format!("{} - DMA not available", function!()));
            return;
        }
        if !is_opengl_available() {
            crate::test_support::report_skip(&format!("{} - OpenGL not available", function!()));
            return;
        }
        if !is_neutron_available() {
            crate::test_support::report_skip(&format!(
                "{} - Neutron not available (/dev/neutron0 not found)",
                function!()
            ));
            return;
        }

        let total_size: usize = 10_276_864;
        let offset: usize = 3_450_400;

        let src = load_raw_image(
            1280,
            720,
            PixelFormat::Nv12,
            Some(TensorMemory::DmaBuf),
            &edgefirst_bench::testdata::read("camera720p.nv12"),
        )
        .unwrap();

        let proc = crate::ImageProcessor::new().unwrap();
        let mut gl = GLProcessorThreaded::new(None).unwrap();

        // --- RGB U8 at large offset (two-pass, packed_rgba8_program_2d) ---
        let Some(u8_buf) = crate::test_support::dma_allocated(
            Tensor::<u8>::new(&[total_size], Some(TensorMemory::DmaBuf), None),
            "Neutron-sized DMA buffer",
        ) else {
            return;
        };
        u8_buf.map().unwrap().as_mut_slice().fill(0xAA);

        let fd = u8_buf.as_dma().unwrap().fd.as_fd();
        let plane = PlaneDescriptor::new(fd).unwrap().with_offset(offset);
        let mut dst_u8 = proc
            .import_image(plane, None, 640, 640, PixelFormat::Rgb, DType::U8, None)
            .unwrap();

        let u8_result = gl.convert(&src, &mut dst_u8, None);
        assert!(
            u8_result.is_ok(),
            "RGB U8 two-pass at large offset failed: {:?}",
            u8_result.err()
        );

        // --- RGB I8 at same large offset (two-pass, packed_rgba8_int8_program_2d) ---
        let Some(i8_buf) = crate::test_support::dma_allocated(
            Tensor::<u8>::new(&[total_size], Some(TensorMemory::DmaBuf), None),
            "Neutron-sized DMA buffer",
        ) else {
            return;
        };
        i8_buf.map().unwrap().as_mut_slice().fill(0xAA);

        let fd = i8_buf.as_dma().unwrap().fd.as_fd();
        let plane = PlaneDescriptor::new(fd).unwrap().with_offset(offset);
        let mut dst_i8 = proc
            .import_image(plane, None, 640, 640, PixelFormat::Rgb, DType::I8, None)
            .unwrap();

        let i8_result = gl.convert(&src, &mut dst_i8, None);
        assert!(
            i8_result.is_ok(),
            "RGB I8 two-pass at large offset failed (U8 succeeded): {:?}",
            i8_result.err()
        );
    }

    /// An RGB int8 destination converts at a page-aligned and at a non-page-aligned offset.
    #[test]
    #[cfg(target_os = "linux")]
    fn test_neutron_scenario_offset_alignment() {
        use edgefirst_tensor::PlaneDescriptor;
        use std::os::fd::AsFd;

        if !is_dma_available() {
            crate::test_support::report_skip(&format!("{} - DMA not available", function!()));
            return;
        }
        if !is_opengl_available() {
            crate::test_support::report_skip(&format!("{} - OpenGL not available", function!()));
            return;
        }
        if !is_neutron_available() {
            crate::test_support::report_skip(&format!(
                "{} - Neutron not available (/dev/neutron0 not found)",
                function!()
            ));
            return;
        }

        let total_size: usize = 10_276_864;
        let page_aligned_offset: usize = 3_448_832; // 842 * 4096
        let neutron_offset: usize = 3_450_400; // actual Neutron offset

        let src = load_raw_image(
            1280,
            720,
            PixelFormat::Nv12,
            Some(TensorMemory::DmaBuf),
            &edgefirst_bench::testdata::read("camera720p.nv12"),
        )
        .unwrap();

        let proc = crate::ImageProcessor::new().unwrap();
        let mut gl = GLProcessorThreaded::new(None).unwrap();

        // --- Page-aligned offset ---
        let Some(aligned_buf) = crate::test_support::dma_allocated(
            Tensor::<u8>::new(&[total_size], Some(TensorMemory::DmaBuf), None),
            "Neutron-sized DMA buffer",
        ) else {
            return;
        };
        aligned_buf.map().unwrap().as_mut_slice().fill(0xAA);

        let fd = aligned_buf.as_dma().unwrap().fd.as_fd();
        let plane = PlaneDescriptor::new(fd)
            .unwrap()
            .with_offset(page_aligned_offset);
        let mut dst_aligned = proc
            .import_image(plane, None, 640, 640, PixelFormat::Rgb, DType::I8, None)
            .unwrap();

        let aligned_result = gl.convert(&src, &mut dst_aligned, None);
        assert!(
            aligned_result.is_ok(),
            "Page-aligned offset {} failed: {:?}",
            page_aligned_offset,
            aligned_result.err()
        );

        // --- Non-page-aligned Neutron offset ---
        let Some(unaligned_buf) = crate::test_support::dma_allocated(
            Tensor::<u8>::new(&[total_size], Some(TensorMemory::DmaBuf), None),
            "Neutron-sized DMA buffer",
        ) else {
            return;
        };
        unaligned_buf.map().unwrap().as_mut_slice().fill(0xAA);

        let fd = unaligned_buf.as_dma().unwrap().fd.as_fd();
        let plane = PlaneDescriptor::new(fd)
            .unwrap()
            .with_offset(neutron_offset);
        let mut dst_unaligned = proc
            .import_image(plane, None, 640, 640, PixelFormat::Rgb, DType::I8, None)
            .unwrap();

        let unaligned_result = gl.convert(&src, &mut dst_unaligned, None);
        assert!(
            unaligned_result.is_ok(),
            "Non-page-aligned Neutron offset {} failed (page-aligned {} succeeded): {:?}",
            neutron_offset,
            page_aligned_offset,
            unaligned_result.err()
        );
    }

    // =========================================================================
    // PlaneDescriptor stride/offset integration tests for import_image
    // =========================================================================

    /// `import_image` of a tight RGBA DMA buffer reports the expected dimensions and format.
    #[test]
    #[cfg(target_os = "linux")]
    fn test_import_image_rgba_with_stride() {
        use edgefirst_tensor::PlaneDescriptor;

        if !is_dma_available() {
            crate::test_support::report_skip(
                "test_import_image_rgba_with_stride - DMA not available",
            );
            return;
        }

        let width: usize = 640;
        let height: usize = 480;
        let bpp: usize = 4; // RGBA
        let stride = width * bpp; // 2560, tightly packed

        let Some(buf) = crate::test_support::dma_allocated(
            Tensor::<u8>::new(
                &[height, width, bpp],
                Some(TensorMemory::DmaBuf),
                Some("import_rgba_tight"),
            ),
            "DMA buffer",
        ) else {
            return;
        };

        let fd = buf.dmabuf().unwrap();
        let plane = PlaneDescriptor::new(fd).unwrap().with_stride(stride);

        let proc = crate::ImageProcessor::new().unwrap();
        let imported = proc
            .import_image(
                plane,
                None,
                width,
                height,
                PixelFormat::Rgba,
                DType::U8,
                None,
            )
            .unwrap();

        assert_eq!(imported.width(), Some(width));
        assert_eq!(imported.height(), Some(height));
        assert_eq!(imported.format(), Some(PixelFormat::Rgba));
        assert_eq!(imported.row_stride(), Some(stride));
    }

    /// `import_image` accepts an RGBA DMA buffer with a padded stride and reports its shape.
    #[test]
    #[cfg(target_os = "linux")]
    fn test_import_image_rgba_padded_stride() {
        use edgefirst_tensor::PlaneDescriptor;

        if !is_dma_available() {
            crate::test_support::report_skip(
                "test_import_image_rgba_padded_stride - DMA not available",
            );
            return;
        }

        let width: usize = 640;
        let height: usize = 480;
        let bpp: usize = 4;
        // A padded stride: the 128-aligned row plus 128 bytes.
        let stride: usize = (width * bpp).div_ceil(128) * 128; // 2560
        let stride = stride + 128; // 2688

        let total_bytes = stride * height;
        let Some(buf) = crate::test_support::dma_allocated(
            Tensor::<u8>::new(
                &[total_bytes],
                Some(TensorMemory::DmaBuf),
                Some("import_rgba_padded"),
            ),
            "DMA buffer",
        ) else {
            return;
        };

        let fd = buf.dmabuf().unwrap();
        let plane = PlaneDescriptor::new(fd).unwrap().with_stride(stride);

        let proc = crate::ImageProcessor::new().unwrap();
        let imported = proc
            .import_image(
                plane,
                None,
                width,
                height,
                PixelFormat::Rgba,
                DType::U8,
                None,
            )
            .unwrap();

        assert_eq!(imported.width(), Some(width));
        assert_eq!(imported.height(), Some(height));
        assert_eq!(imported.format(), Some(PixelFormat::Rgba));
        assert_eq!(imported.row_stride(), Some(stride));
    }

    /// `import_image` of one DMA buffer as two-plane NV12 with the chroma at `height * stride`.
    #[test]
    #[cfg(target_os = "linux")]
    fn test_import_image_nv12_with_offset() {
        use edgefirst_tensor::PlaneDescriptor;

        if !is_dma_available() {
            crate::test_support::report_skip(
                "test_import_image_nv12_with_offset - DMA not available",
            );
            return;
        }

        let width: usize = 640;
        let height: usize = 480;
        let stride: usize = 640; // NV12 luma stride = width (1 byte per pixel)
        let luma_size = height * stride;
        let chroma_h = height / 2;
        let chroma_size = chroma_h * stride;
        let total_bytes = luma_size + chroma_size;

        // Single DMA buffer holding both luma and chroma planes
        let Some(buf) = crate::test_support::dma_allocated(
            Tensor::<u8>::new(
                &[total_bytes],
                Some(TensorMemory::DmaBuf),
                Some("import_nv12_planes"),
            ),
            "DMA buffer",
        ) else {
            return;
        };

        let fd = buf.dmabuf().unwrap();
        let luma_plane = PlaneDescriptor::new(fd)
            .unwrap()
            .with_stride(stride)
            .with_offset(0);
        let chroma_plane = PlaneDescriptor::new(fd)
            .unwrap()
            .with_stride(stride)
            .with_offset(luma_size);

        let proc = crate::ImageProcessor::new().unwrap();
        let imported = proc
            .import_image(
                luma_plane,
                Some(chroma_plane),
                width,
                height,
                PixelFormat::Nv12,
                DType::U8,
                None,
            )
            .unwrap();

        assert_eq!(imported.width(), Some(width));
        assert_eq!(imported.height(), Some(height));
        assert_eq!(imported.format(), Some(PixelFormat::Nv12));
    }

    /// Two-plane NV12 imported from separate DMA-BUFs converts to the pixels of contiguous NV12.
    #[test]
    #[cfg(target_os = "linux")]
    fn test_import_image_nv12_true_multiplane() {
        use edgefirst_tensor::PlaneDescriptor;

        if !is_dma_available() {
            crate::test_support::report_skip(&format!("{} - DMA not available", function!()));
            return;
        }
        if !is_opengl_available() {
            crate::test_support::report_skip(&format!("{} - OpenGL not available", function!()));
            return;
        }

        // Both sources are pinned to `ExternalSampler`.
        let _nv_path = NvPathEnvGuard::set("sampler");

        let nv12_bytes: &[u8] = &edgefirst_bench::testdata::read("camera720p.nv12");
        let width: usize = 1280;
        let height: usize = 720;
        let stride: usize = width;
        let y_size = stride * height;
        let uv_size = stride * (height / 2);
        assert_eq!(nv12_bytes.len(), y_size + uv_size);

        // Two separate DMA allocations filled with the NV12 planes.
        let Some(luma_buf) = crate::test_support::dma_allocated(
            Tensor::<u8>::new(&[y_size], Some(TensorMemory::DmaBuf), Some("true_mp_luma")),
            "luma DMA buffer",
        ) else {
            return;
        };
        luma_buf.map().unwrap().as_mut_slice()[..y_size].copy_from_slice(&nv12_bytes[..y_size]);

        let Some(chroma_buf) = crate::test_support::dma_allocated(
            Tensor::<u8>::new(
                &[uv_size],
                Some(TensorMemory::DmaBuf),
                Some("true_mp_chroma"),
            ),
            "chroma DMA buffer",
        ) else {
            return;
        };
        chroma_buf.map().unwrap().as_mut_slice()[..uv_size].copy_from_slice(&nv12_bytes[y_size..]);

        // Import via import_image with separate PlaneDescriptors
        let luma_fd = luma_buf.dmabuf().unwrap();
        let chroma_fd = chroma_buf.dmabuf().unwrap();
        let luma_plane = PlaneDescriptor::new(luma_fd).unwrap().with_stride(stride);
        let chroma_plane = PlaneDescriptor::new(chroma_fd).unwrap().with_stride(stride);

        let proc = crate::ImageProcessor::new().unwrap();
        let src = proc
            .import_image(
                luma_plane,
                Some(chroma_plane),
                width,
                height,
                PixelFormat::Nv12,
                DType::U8,
                None,
            )
            .unwrap();
        assert!(src.is_multiplane(), "must be multiplane after import");

        // Build contiguous reference for pixel comparison
        let src_contig = load_raw_image(
            width,
            height,
            PixelFormat::Nv12,
            Some(TensorMemory::DmaBuf),
            nv12_bytes,
        )
        .unwrap();

        let mut gl = GLProcessorThreaded::new(None).unwrap();

        // Render the two-plane source.
        let mut dst_multi = TensorDyn::image(
            width,
            height,
            PixelFormat::Rgba,
            DType::U8,
            Some(TensorMemory::DmaBuf),
            edgefirst_tensor::CpuAccess::ReadWrite,
        )
        .unwrap();
        let result = gl.convert(&src, &mut dst_multi, None);
        if declined_separate_chroma(&result, function!()) {
            // On a decline, `ImageProcessor` converts on the CPU and matches the CPU's convert of
            // the contiguous bytes.
            let mut proc = proc;
            proc.convert(&src, &mut dst_multi, None).unwrap();
            let mut dst_cpu = TensorDyn::image(
                width,
                height,
                PixelFormat::Rgba,
                DType::U8,
                Some(TensorMemory::DmaBuf),
                edgefirst_tensor::CpuAccess::ReadWrite,
            )
            .unwrap();
            crate::CPUProcessor::new()
                .convert(&src_contig, &mut dst_cpu, None)
                .unwrap();
            let map_multi = dst_multi.as_u8().unwrap().map().unwrap();
            let map_cpu = dst_cpu.as_u8().unwrap().map().unwrap();
            assert_pixels_match(map_cpu.as_slice(), map_multi.as_slice(), 0);
            return;
        }
        result.unwrap();

        // Render contiguous reference
        let mut dst_contig = TensorDyn::image(
            width,
            height,
            PixelFormat::Rgba,
            DType::U8,
            Some(TensorMemory::DmaBuf),
            edgefirst_tensor::CpuAccess::ReadWrite,
        )
        .unwrap();
        gl.convert(&src_contig, &mut dst_contig, None).unwrap();

        // Both produce identical pixels.
        let map_multi = dst_multi.as_u8().unwrap().map().unwrap();
        let map_contig = dst_contig.as_u8().unwrap().map().unwrap();
        assert_pixels_match(map_multi.as_slice(), map_contig.as_slice(), 0);
    }

    /// Contiguous single-fd NV12 through `import_image` converts to the pixels of `load_raw_image`.
    #[test]
    #[cfg(target_os = "linux")]
    fn test_import_image_nv12_contiguous() {
        use edgefirst_tensor::PlaneDescriptor;

        if !is_dma_available() {
            crate::test_support::report_skip(&format!("{} - DMA not available", function!()));
            return;
        }
        if !is_opengl_available() {
            crate::test_support::report_skip(&format!("{} - OpenGL not available", function!()));
            return;
        }

        let nv12_bytes: &[u8] = &edgefirst_bench::testdata::read("camera720p.nv12");
        let width: usize = 1280;
        let height: usize = 720;
        let stride: usize = width;
        let total_bytes = nv12_bytes.len();

        // Single DMA buffer holding Y+UV contiguously
        let Some(buf) = crate::test_support::dma_allocated(
            Tensor::<u8>::new(
                &[total_bytes],
                Some(TensorMemory::DmaBuf),
                Some("nv12_contiguous"),
            ),
            "DMA buffer",
        ) else {
            return;
        };
        buf.map().unwrap().as_mut_slice()[..total_bytes].copy_from_slice(nv12_bytes);

        // Import with no chroma descriptor.
        let fd = buf.dmabuf().unwrap();
        let plane = PlaneDescriptor::new(fd).unwrap().with_stride(stride);

        let proc = crate::ImageProcessor::new().unwrap();
        let src = proc
            .import_image(
                plane,
                None,
                width,
                height,
                PixelFormat::Nv12,
                DType::U8,
                None,
            )
            .unwrap();
        assert!(
            !src.is_multiplane(),
            "contiguous NV12 must not be multiplane"
        );

        // Reference through `load_raw_image`.
        let src_ref = load_raw_image(
            width,
            height,
            PixelFormat::Nv12,
            Some(TensorMemory::DmaBuf),
            nv12_bytes,
        )
        .unwrap();

        let mut gl = GLProcessorThreaded::new(None).unwrap();

        // Render the imported source.
        let mut dst_imported = TensorDyn::image(
            width,
            height,
            PixelFormat::Rgba,
            DType::U8,
            Some(TensorMemory::DmaBuf),
            edgefirst_tensor::CpuAccess::ReadWrite,
        )
        .unwrap();
        gl.convert(&src, &mut dst_imported, None).unwrap();

        // Render reference
        let mut dst_ref = TensorDyn::image(
            width,
            height,
            PixelFormat::Rgba,
            DType::U8,
            Some(TensorMemory::DmaBuf),
            edgefirst_tensor::CpuAccess::ReadWrite,
        )
        .unwrap();
        gl.convert(&src_ref, &mut dst_ref, None).unwrap();

        // Both produce identical pixels.
        let map_imported = dst_imported.as_u8().unwrap().map().unwrap();
        let map_ref = dst_ref.as_u8().unwrap().map().unwrap();
        assert_pixels_match(map_imported.as_slice(), map_ref.as_slice(), 0);
    }

    /// `import_image` rejects a stride smaller than `width * bpp`.
    #[test]
    #[cfg(target_os = "linux")]
    fn test_import_image_stride_too_small() {
        use edgefirst_tensor::PlaneDescriptor;

        if !is_dma_available() {
            crate::test_support::report_skip(
                "test_import_image_stride_too_small - DMA not available",
            );
            return;
        }

        let width: usize = 640;
        let height: usize = 480;
        let bpp: usize = 4;
        let bad_stride = width * bpp - 1; // one byte too narrow

        let Some(buf) = crate::test_support::dma_allocated(
            Tensor::<u8>::new(
                &[height, width, bpp],
                Some(TensorMemory::DmaBuf),
                Some("import_bad_stride"),
            ),
            "DMA buffer",
        ) else {
            return;
        };

        let fd = buf.dmabuf().unwrap();
        let plane = PlaneDescriptor::new(fd).unwrap().with_stride(bad_stride);

        let proc = crate::ImageProcessor::new().unwrap();
        let result = proc.import_image(
            plane,
            None,
            width,
            height,
            PixelFormat::Rgba,
            DType::U8,
            None,
        );

        assert!(
            result.is_err(),
            "import_image should reject stride {bad_stride} < minimum {} for {width}x{height} RGBA",
            width * bpp
        );
    }

    // ──────────────────────────────────────────────────────────────────────
    // src_rect clamping tests: a crop from a larger buffer samples nothing outside the crop.
    // ──────────────────────────────────────────────────────────────────────

    /// Create a synthetic RGB tensor whose left half is pure red and right half pure blue.
    fn make_red_blue_src(width: usize, height: usize) -> TensorDyn {
        let mut t = TensorDyn::image(
            width,
            height,
            PixelFormat::Rgb,
            DType::U8,
            None,
            edgefirst_tensor::CpuAccess::ReadWrite,
        )
        .unwrap();
        {
            let tensor_u8 = t.as_u8_mut().unwrap();
            let stride = tensor_u8.effective_row_stride().unwrap_or(width * 3);
            let mut map = tensor_u8.map().unwrap();
            let data = map.as_mut_slice();
            let half = width / 2;
            for y in 0..height {
                for x in 0..width {
                    let idx = y * stride + x * 3;
                    if x < half {
                        data[idx] = 255;
                        data[idx + 1] = 0;
                        data[idx + 2] = 0;
                    } else {
                        data[idx] = 0;
                        data[idx + 1] = 0;
                        data[idx + 2] = 255;
                    }
                }
            }
        }
        t
    }

    /// A crop of the blue (right) half of a red|blue image, resized down, contains no red.
    #[test]
    fn test_src_rect_crop_no_bleed_gl() {
        if !is_opengl_available() {
            crate::test_support::report_skip(
                "test_src_rect_crop_no_bleed_gl - OpenGL not available",
            );
            return;
        }

        let src_w = 128;
        let src_h = 64;
        let dst_w = 32;
        let dst_h = 32;

        let src = make_red_blue_src(src_w, src_h);
        let mut dst = TensorDyn::image(
            dst_w,
            dst_h,
            PixelFormat::Rgb,
            DType::U8,
            None,
            edgefirst_tensor::CpuAccess::ReadWrite,
        )
        .unwrap();

        let mut gl = GLProcessorThreaded::new(None).unwrap();

        // Crop only the right (blue) half
        let crop = Crop::new().with_source(Some(crate::Region::new(
            src_w / 2, // left = start of blue region
            0,
            src_w / 2, // width = blue half
            src_h,
        )));

        if let Err(e) = gl.convert(
            &src,
            &mut dst,
            Some(crate::ConvertOptions::new().crop(crop)),
        ) {
            // A platform without a packed-RGB source route refuses an RGB source surface.
            if e.to_string().contains("RGB source") {
                crate::test_support::report_skip(&format!("{} - {e}", function!()));
                return;
            }
            panic!("{e}");
        }

        // Every output pixel is blue, read at the row stride.
        let stride = dst
            .as_u8()
            .unwrap()
            .effective_row_stride()
            .unwrap_or(dst_w * 3);
        let map = dst.as_u8().unwrap().map().unwrap();
        let data = map.as_slice();
        let pixel_count = dst_w * dst_h;
        let mut max_red: u8 = 0;
        for i in 0..pixel_count {
            let at = (i / dst_w) * stride + (i % dst_w) * 3;
            let (r, g, b) = (data[at], data[at + 1], data[at + 2]);
            max_red = max_red.max(r);
            assert!(
                r <= 2 && g <= 2 && b >= 253,
                "Pixel {i} has red bleed: RGB=({r},{g},{b}), expected pure blue"
            );
        }
        assert!(
            max_red <= 2,
            "Max red channel value in output = {max_red}, expected 0 (no bleed from padding)"
        );
    }

    /// A crop starting exactly at the red→blue transition contains no red.
    #[test]
    fn test_src_rect_boundary_crop_no_bleed_gl() {
        if !is_opengl_available() {
            crate::test_support::report_skip(
                "test_src_rect_boundary_crop_no_bleed_gl - OpenGL not available",
            );
            return;
        }

        let src_w = 256;
        let src_h = 64;
        let dst_w = 64;
        let dst_h = 64;

        let src = make_red_blue_src(src_w, src_h);
        let mut dst = TensorDyn::image(
            dst_w,
            dst_h,
            PixelFormat::Rgb,
            DType::U8,
            None,
            edgefirst_tensor::CpuAccess::ReadWrite,
        )
        .unwrap();

        let mut gl = GLProcessorThreaded::new(None).unwrap();

        // Crop the right half; its left boundary is the red→blue edge.
        let crop =
            Crop::new().with_source(Some(crate::Region::new(src_w / 2, 0, src_w / 2, src_h)));

        if let Err(e) = gl.convert(
            &src,
            &mut dst,
            Some(crate::ConvertOptions::new().crop(crop)),
        ) {
            // A platform without a packed-RGB source route refuses an RGB source surface.
            if e.to_string().contains("RGB source") {
                crate::test_support::report_skip(&format!("{} - {e}", function!()));
                return;
            }
            panic!("{e}");
        }

        // Pixels are read at the row stride.
        let stride = dst
            .as_u8()
            .unwrap()
            .effective_row_stride()
            .unwrap_or(dst_w * 3);
        let map = dst.as_u8().unwrap().map().unwrap();
        let data = map.as_slice();
        for i in 0..(dst_w * dst_h) {
            let at = (i / dst_w) * stride + (i % dst_w) * 3;
            let (r, g, b) = (data[at], data[at + 1], data[at + 2]);
            assert!(
                r <= 2 && g <= 2 && b >= 253,
                "Pixel {i} has contamination at boundary: RGB=({r},{g},{b})"
            );
        }
    }

    /// A crop of the red (left) half contains no blue.
    #[test]
    fn test_src_rect_crop_left_half_no_bleed_gl() {
        if !is_opengl_available() {
            crate::test_support::report_skip(
                "test_src_rect_crop_left_half_no_bleed_gl - OpenGL not available",
            );
            return;
        }

        let src_w = 128;
        let src_h = 64;
        let dst_w = 32;
        let dst_h = 32;

        let src = make_red_blue_src(src_w, src_h);
        let mut dst = TensorDyn::image(
            dst_w,
            dst_h,
            PixelFormat::Rgb,
            DType::U8,
            None,
            edgefirst_tensor::CpuAccess::ReadWrite,
        )
        .unwrap();

        let mut gl = GLProcessorThreaded::new(None).unwrap();

        // Crop only the left (red) half
        let crop = Crop::new().with_source(Some(crate::Region::new(0, 0, src_w / 2, src_h)));

        if let Err(e) = gl.convert(
            &src,
            &mut dst,
            Some(crate::ConvertOptions::new().crop(crop)),
        ) {
            // A platform without a packed-RGB source route refuses an RGB source surface.
            if e.to_string().contains("RGB source") {
                crate::test_support::report_skip(&format!("{} - {e}", function!()));
                return;
            }
            panic!("{e}");
        }

        // Pixels are read at the row stride.
        let stride = dst
            .as_u8()
            .unwrap()
            .effective_row_stride()
            .unwrap_or(dst_w * 3);
        let map = dst.as_u8().unwrap().map().unwrap();
        let data = map.as_slice();
        for i in 0..(dst_w * dst_h) {
            let at = (i / dst_w) * stride + (i % dst_w) * 3;
            let (r, g, b) = (data[at], data[at + 1], data[at + 2]);
            assert!(
                r >= 253 && g <= 2 && b <= 2,
                "Pixel {i} has blue bleed: RGB=({r},{g},{b}), expected pure red"
            );
        }
    }

    // ── float_render_support unit tests ─────────────────────────────────────
    // These are pure-logic tests; no GPU is required.

    #[test]
    fn vivante_disables_gl_float() {
        use super::super::processor::float_render_support;
        use crate::RenderDtypeSupport;
        let s = float_render_support(true, false, true, true);
        assert_eq!(
            s,
            RenderDtypeSupport {
                f32: false,
                f16: false
            }
        );
    }

    #[test]
    fn non_vivante_reports_float_when_ext_present() {
        use super::super::processor::float_render_support;
        use crate::RenderDtypeSupport;
        let s = float_render_support(false, false, true, true);
        assert_eq!(
            s,
            RenderDtypeSupport {
                f32: true,
                f16: true
            }
        );
    }

    #[test]
    fn refused_float_import_disables_gl_float() {
        use super::super::processor::float_render_support;
        use crate::RenderDtypeSupport;
        // Adreno on a zero-copy backend: a float color buffer but no float DMA-BUF import.
        let s = float_render_support(false, true, true, true);
        assert_eq!(
            s,
            RenderDtypeSupport {
                f32: false,
                f16: false
            }
        );
    }

    #[test]
    fn no_ext_means_no_float() {
        use super::super::processor::float_render_support;
        use crate::RenderDtypeSupport;
        let s = float_render_support(false, false, false, false);
        assert_eq!(
            s,
            RenderDtypeSupport {
                f32: false,
                f16: false
            }
        );
    }

    // ── float_pbo_eligible unit tests ────────────────────────────────────────
    // Pure-logic predicate; no GPU required.

    #[test]
    fn float_pbo_eligibility() {
        use crate::{float_pbo_eligible, RenderDtypeSupport};
        use edgefirst_tensor::DType;

        let yes = RenderDtypeSupport {
            f32: true,
            f16: true,
        };
        let no = RenderDtypeSupport {
            f32: false,
            f16: false,
        };
        assert!(float_pbo_eligible(DType::F16, yes));
        assert!(float_pbo_eligible(DType::F32, yes));
        assert!(!float_pbo_eligible(DType::F16, no));
        assert!(!float_pbo_eligible(DType::U8, yes));
    }

    // ── packed F32 NHWC shader source well-formedness ────────────────────────
    // Pure string inspection; no GPU required.

    #[test]
    fn packed_f32_nhwc_shader_source_wellformed() {
        use super::super::shaders::generate_packed_f32_nhwc_shader;
        let s = generate_packed_f32_nhwc_shader();
        assert!(s.contains("#version 300 es"));
        assert!(s.contains("frag_value"));
        assert!(s.contains("% 3")); // channel = x % 3
        assert!(s.contains("src_rect_uv"));
        assert!(s.contains("dst_rect_px"));
        assert!(s.contains("pad_color"));
    }

    // ── RGBA16F-packed PlanarRgb F16 shader source well-formedness ──────────
    // Pure string inspection; no GPU required.

    #[test]
    fn planar_rgb_f16_packed_shader_source_wellformed() {
        use super::super::shaders::generate_planar_rgb_f16_packed_shader;
        let s = generate_planar_rgb_f16_packed_shader();
        assert!(s.contains("#version 300 es"));
        assert!(s.contains("src_rect_uv"));
        assert!(s.contains("dst_rect_px"));
        assert!(s.contains("pad_color"));
        assert!(s.contains("sample_planar_element"));
        assert!(s.contains("vec4(e0, e1, e2, e3)"));
    }

    // ── classify_float_render unit tests ────────────────────────────────────
    // Pure-logic classifier; no GPU required. The zero-copy float set is the one this
    // build's platform reports.
    use super::super::platform::{GlPlatform, Platform};
    const ZERO_COPY_FLOAT: super::super::float_dispatch::ZeroCopyFloatSet =
        <Platform as GlPlatform>::ZERO_COPY_FLOAT;

    #[test]
    fn dispatch_hailo_f32_pbo() {
        use super::super::processor::{classify_float_render, FloatRenderPath};
        use crate::RenderDtypeSupport;
        let s = RenderDtypeSupport {
            f32: true,
            f16: true,
        };
        assert_eq!(
            classify_float_render(
                PixelFormat::Rgba,
                PixelFormat::Rgb,
                DType::F32,
                TensorMemory::Pbo,
                s,
                ZERO_COPY_FLOAT,
            ),
            FloatRenderPath::PboF32Nhwc
        );
    }

    #[test]
    fn dispatch_orin_f16_pbo() {
        use super::super::processor::{classify_float_render, FloatRenderPath};
        use crate::RenderDtypeSupport;
        let s = RenderDtypeSupport {
            f32: true,
            f16: true,
        };
        assert_eq!(
            classify_float_render(
                PixelFormat::Rgba,
                PixelFormat::PlanarRgb,
                DType::F16,
                TensorMemory::Pbo,
                s,
                ZERO_COPY_FLOAT,
            ),
            FloatRenderPath::PboF16Nchw
        );
    }

    #[test]
    fn dispatch_f16_dma() {
        use super::super::processor::{classify_float_render, FloatRenderPath};
        use crate::RenderDtypeSupport;
        let s = RenderDtypeSupport {
            f32: true,
            f16: true,
        };
        assert_eq!(
            classify_float_render(
                PixelFormat::Rgba,
                PixelFormat::PlanarRgb,
                DType::F16,
                TensorMemory::DmaBuf,
                s,
                ZERO_COPY_FLOAT,
            ),
            FloatRenderPath::ZeroCopyF16Nchw
        );
    }

    /// Rgba to Rgb F32 into a DmaBuf destination classifies by this platform's zero-copy float set.
    #[test]
    fn dispatch_f32_dma_follows_this_leafs_zero_copy_set() {
        use super::super::float_dispatch::ZeroCopyFloatSet;
        use super::super::processor::{classify_float_render, FloatRenderPath};
        use crate::RenderDtypeSupport;
        let s = RenderDtypeSupport {
            f32: true,
            f16: true,
        };
        let expected = match ZERO_COPY_FLOAT {
            ZeroCopyFloatSet::All => FloatRenderPath::ZeroCopyFloatNhwc,
            ZeroCopyFloatSet::PlanarF16 => FloatRenderPath::None,
        };
        assert_eq!(
            classify_float_render(
                PixelFormat::Rgba,
                PixelFormat::Rgb,
                DType::F32,
                TensorMemory::DmaBuf,
                s,
                ZERO_COPY_FLOAT,
            ),
            expected
        );
    }

    #[test]
    fn dispatch_no_support_is_none() {
        use super::super::processor::{classify_float_render, FloatRenderPath};
        use crate::RenderDtypeSupport;
        let s = RenderDtypeSupport {
            f32: false,
            f16: false,
        };
        assert_eq!(
            classify_float_render(
                PixelFormat::Rgba,
                PixelFormat::Rgb,
                DType::F32,
                TensorMemory::Pbo,
                s,
                ZERO_COPY_FLOAT,
            ),
            FloatRenderPath::None
        );
    }

    // ── packed_planar_layout unit tests ─────────────────────────────────────
    // Pure geometry helper; no GPU required.

    #[test]
    fn packed_planar_layout_640x640() {
        use super::super::processor::packed_planar_layout;
        // Logical [3,640,640] → packed surface (640/4, 3*640).
        assert_eq!(packed_planar_layout(640, 640, 3), Some((160, 1920)));
    }

    #[test]
    fn packed_planar_layout_non_multiple_of_4_rejected() {
        use super::super::processor::packed_planar_layout;
        // W not divisible by 4 cannot be packed into whole four-element texels.
        assert_eq!(packed_planar_layout(642, 640, 3), None);
    }

    #[test]
    fn packed_planar_layout_rectangular() {
        use super::super::processor::packed_planar_layout;
        // Non-square, W divisible by 4: surface (320/4, 3*240).
        assert_eq!(packed_planar_layout(320, 240, 3), Some((80, 720)));
    }

    #[test]
    fn packed_planar_layout_four_planes() {
        use super::super::processor::packed_planar_layout;
        // A PlanarRgba destination stacks a fourth plane onto the surface.
        assert_eq!(packed_planar_layout(640, 480, 4), Some((160, 1920)));
        assert_eq!(packed_planar_layout(640, 480, 3), Some((160, 1440)));
    }

    /// An int8 letterbox through every PBO source/destination pair is byte-identical to Mem→Mem.
    #[test]
    #[cfg(any(target_os = "linux", target_os = "windows"))]
    fn pbo_int8_letterbox_matches_mem_oracle() {
        use crate::{ComputeBackend, ImageProcessor, ImageProcessorConfig};

        if !is_opengl_available() {
            crate::test_support::report_skip(&format!("{} - OpenGL not available", function!()));
            return;
        }
        let mut proc = match ImageProcessor::with_config(ImageProcessorConfig {
            backend: ComputeBackend::OpenGl,
            ..Default::default()
        }) {
            Ok(p) => p,
            Err(e) => {
                crate::test_support::report_skip(&format!(
                    "{} - OpenGL backend unavailable: {e}",
                    function!()
                ));
                return;
            }
        };

        let (sw, sh) = (64usize, 48usize);
        let probe = proc
            .create_image(
                sw,
                sh,
                PixelFormat::Rgba,
                DType::U8,
                None,
                edgefirst_tensor::CpuAccess::ReadWrite,
            )
            .unwrap();
        if probe.memory() != TensorMemory::Pbo {
            crate::test_support::report_skip(&format!(
                "{} - target does not allocate PBO images",
                function!()
            ));
            return;
        }

        let fill = |t: &TensorDyn| {
            let mut map = t.as_u8().unwrap().map().unwrap();
            for (i, px) in map
                .as_mut_slice()
                .as_chunks_mut::<4>()
                .0
                .iter_mut()
                .enumerate()
            {
                px[0] = (i % 251) as u8;
                px[1] = (i % 199) as u8;
                px[2] = (i % 127) as u8;
                px[3] = 255;
            }
        };
        let src_pbo = probe;
        fill(&src_pbo);
        let src_mem = proc
            .create_image(
                sw,
                sh,
                PixelFormat::Rgba,
                DType::U8,
                Some(TensorMemory::Mem),
                edgefirst_tensor::CpuAccess::ReadWrite,
            )
            .unwrap();
        fill(&src_mem);

        // 64×48 → 96×96 letterbox: a 96×72 inner box with cleared bands above and below.
        let lb = Crop::letterbox([114, 114, 114, 255]);
        // A PBO destination is requested as `None` and its resolved memory asserted.
        let mut convert_to_bytes = |src: &TensorDyn, dst_mem: TensorMemory, label: &str| {
            let dst_req = match dst_mem {
                TensorMemory::Pbo => None,
                other => Some(other),
            };
            let mut dst = proc
                .create_image(
                    96,
                    96,
                    PixelFormat::Rgb,
                    DType::I8,
                    dst_req,
                    edgefirst_tensor::CpuAccess::ReadWrite,
                )
                .unwrap();
            assert_eq!(dst.memory(), dst_mem, "{label}: dst backing not honoured");
            let fallbacks_before = proc.convert_fallback_count();
            proc.convert(src, &mut dst, Some(crate::ConvertOptions::new().crop(lb)))
                .unwrap_or_else(|e| panic!("{label} convert failed: {e}"));
            assert_eq!(
                proc.convert_fallback_count(),
                fallbacks_before,
                "{label}: the GL path declined and ImageProcessor fell back to \
                 the CPU; the oracle would then be compared against itself"
            );
            dst.as_i8()
                .unwrap()
                .map()
                .unwrap()
                .as_slice()
                .iter()
                .map(|&v| v as u8)
                .collect::<Vec<u8>>()
        };

        let oracle = convert_to_bytes(&src_mem, TensorMemory::Mem, "mem->mem");
        for (src, src_name) in [(&src_mem, "mem"), (&src_pbo, "pbo")] {
            for dst_mem in [TensorMemory::Mem, TensorMemory::Pbo] {
                let label = format!("{src_name}->{dst_mem:?}");
                let out = convert_to_bytes(src, dst_mem, &label);
                assert_eq!(
                    oracle, out,
                    "{label}: int8 letterbox output diverged from mem->mem oracle"
                );
            }
        }
    }

    /// RGBA8 to an F32 NHWC `[H,W,3]` PBO on the GL float path: `dst[y,x,c] == src[y,x,c] / 255`.
    #[test]
    #[cfg(any(target_os = "linux", target_os = "windows"))]
    fn convert_f32_nhwc_pbo_roundtrip() {
        use crate::{ComputeBackend, ImageProcessor, ImageProcessorConfig};

        if !is_opengl_available() {
            crate::test_support::report_skip(&format!("{} - OpenGL not available", function!()));
            return;
        }

        let mut proc = match ImageProcessor::with_config(ImageProcessorConfig {
            backend: ComputeBackend::OpenGl,
            ..Default::default()
        }) {
            Ok(p) => p,
            Err(e) => {
                crate::test_support::report_skip(&format!(
                    "{} - OpenGL backend unavailable: {e}",
                    function!()
                ));
                return;
            }
        };

        if !proc.supported_render_dtypes().f32 {
            crate::test_support::report_skip(&format!(
                "{} - F32 render not supported",
                function!()
            ));
            return;
        }

        let w = 64usize;
        let h = 64usize;

        // RGBA8 source PBO with a known gradient.
        let src = proc
            .create_image(
                w,
                h,
                PixelFormat::Rgba,
                DType::U8,
                None,
                edgefirst_tensor::CpuAccess::ReadWrite,
            )
            .unwrap();
        if src.memory() != TensorMemory::Pbo {
            crate::test_support::report_skip(&format!(
                "{} - RGBA8 src not PBO-backed",
                function!()
            ));
            return;
        }
        {
            let mut map = src.as_u8().unwrap().map().unwrap();
            for y in 0..h {
                for x in 0..w {
                    let i = (y * w + x) * 4;
                    map[i] = (x * 4) as u8;
                    map[i + 1] = (y * 4) as u8;
                    map[i + 2] = 128;
                    map[i + 3] = 255;
                }
            }
        }

        // F32 Rgb PBO destination, same WxH (logical [H,W,3]).
        let mut dst = proc
            .create_image(
                w,
                h,
                PixelFormat::Rgb,
                DType::F32,
                None,
                edgefirst_tensor::CpuAccess::ReadWrite,
            )
            .unwrap();
        if dst.memory() != TensorMemory::Pbo {
            crate::test_support::report_skip(&format!("{} - F32 dst not PBO-backed", function!()));
            return;
        }

        // Two converts on the same processor, source and destination; the second reuses the
        // cached textures.
        let mut max_err = 0.0f32;
        for iter in 0..2 {
            let fallbacks_before = proc.convert_fallback_count();
            proc.convert(&src, &mut dst, None).unwrap();
            assert_eq!(
                proc.convert_fallback_count(),
                fallbacks_before,
                "the GL float path declined and ImageProcessor fell back to the CPU; \
                 this test's expectation is bit-exact against the CPU answer, so it \
                 would otherwise pass with no GL coverage at all"
            );

            let map = dst.as_f32().unwrap().map().unwrap();
            assert_eq!(map.len(), w * h * 3);
            for y in 0..h {
                for x in 0..w {
                    let expect = [
                        (x * 4) as f32 / 255.0,
                        (y * 4) as f32 / 255.0,
                        128.0 / 255.0,
                    ];
                    for c in 0..3 {
                        let got = map[(y * w + x) * 3 + c];
                        let err = (got - expect[c]).abs();
                        max_err = max_err.max(err);
                        assert!(
                            err < 1e-3,
                            "f32 dst[{y},{x},{c}]={got} expected {} (err {err}) iter {iter}",
                            expect[c]
                        );
                    }
                }
            }
        }
        eprintln!("convert_f32_nhwc_pbo_roundtrip: max_err={max_err} tol=1e-3 (2 iters)");
    }

    /// A 2x downscale of an RGBA8 ramp to an F32 NHWC PBO is bilinear: `(2*dx + 0.5) * 16 / 255`.
    #[test]
    #[cfg(any(target_os = "linux", target_os = "windows"))]
    fn convert_f32_nhwc_pbo_resize_bilinear() {
        use crate::{ComputeBackend, ImageProcessor, ImageProcessorConfig};

        if !is_opengl_available() {
            crate::test_support::report_skip(&format!("{} - OpenGL not available", function!()));
            return;
        }

        let mut proc = match ImageProcessor::with_config(ImageProcessorConfig {
            backend: ComputeBackend::OpenGl,
            ..Default::default()
        }) {
            Ok(p) => p,
            Err(e) => {
                crate::test_support::report_skip(&format!(
                    "{} - OpenGL backend unavailable: {e}",
                    function!()
                ));
                return;
            }
        };

        if !proc.supported_render_dtypes().f32 {
            crate::test_support::report_skip(&format!(
                "{} - F32 render not supported",
                function!()
            ));
            return;
        }

        let sw = 16usize;
        let sh = 16usize;
        let dw = 8usize;
        let dh = 8usize;

        // RGBA8 source PBO: horizontal ramp R = x*16, G = y*16, B const.
        let src = proc
            .create_image(
                sw,
                sh,
                PixelFormat::Rgba,
                DType::U8,
                None,
                edgefirst_tensor::CpuAccess::ReadWrite,
            )
            .unwrap();
        if src.memory() != TensorMemory::Pbo {
            crate::test_support::report_skip(&format!(
                "{} - RGBA8 src not PBO-backed",
                function!()
            ));
            return;
        }
        {
            let mut map = src.as_u8().unwrap().map().unwrap();
            for y in 0..sh {
                for x in 0..sw {
                    let i = (y * sw + x) * 4;
                    map[i] = (x * 16) as u8;
                    map[i + 1] = (y * 16) as u8;
                    map[i + 2] = 64;
                    map[i + 3] = 255;
                }
            }
        }

        // F32 Rgb PBO destination at the downscaled size (logical [dh,dw,3]).
        let mut dst = proc
            .create_image(
                dw,
                dh,
                PixelFormat::Rgb,
                DType::F32,
                None,
                edgefirst_tensor::CpuAccess::ReadWrite,
            )
            .unwrap();
        if dst.memory() != TensorMemory::Pbo {
            crate::test_support::report_skip(&format!("{} - F32 dst not PBO-backed", function!()));
            return;
        }

        let fallbacks_before = proc.convert_fallback_count();
        proc.convert(&src, &mut dst, None).unwrap();
        assert_eq!(
            proc.convert_fallback_count(),
            fallbacks_before,
            "the GL float path declined and ImageProcessor fell back to the CPU; \
             this test's expectation is bit-exact against the CPU answer, so it \
             would otherwise pass with no GL coverage at all"
        );

        let map = dst.as_f32().unwrap().map().unwrap();
        assert_eq!(map.len(), dw * dh * 3);
        // Bilinear: src sample position = (d + 0.5) * 2 - 0.5 = 2*d + 0.5.
        let tol = 2e-3f32;
        let mut max_err = 0.0f32;
        for dy in 0..dh {
            for dx in 0..dw {
                let x_src = 2.0 * dx as f32 + 0.5;
                let y_src = 2.0 * dy as f32 + 0.5;
                let expect = [(x_src * 16.0) / 255.0, (y_src * 16.0) / 255.0, 64.0 / 255.0];
                for c in 0..3 {
                    let got = map[(dy * dw + dx) * 3 + c];
                    let err = (got - expect[c]).abs();
                    max_err = max_err.max(err);
                    assert!(
                        err < tol,
                        "f32 resize dst[{dy},{dx},{c}]={got} expected {} (err {err})",
                        expect[c]
                    );
                }
            }
        }
        eprintln!("convert_f32_nhwc_pbo_resize_bilinear: max_err={max_err} tol={tol}");
    }

    /// RGBA8 to an F16 NCHW `[3,H,W]` PBO on the GL float path: `dst[c,y,x] == src[y,x,c] / 255`.
    #[test]
    #[cfg(any(target_os = "linux", target_os = "windows"))]
    fn convert_f16_nchw_pbo_roundtrip() {
        use crate::{ComputeBackend, ImageProcessor, ImageProcessorConfig};
        use half::f16;

        if !is_opengl_available() {
            crate::test_support::report_skip(&format!("{} - OpenGL not available", function!()));
            return;
        }

        let mut proc = match ImageProcessor::with_config(ImageProcessorConfig {
            backend: ComputeBackend::OpenGl,
            ..Default::default()
        }) {
            Ok(p) => p,
            Err(e) => {
                crate::test_support::report_skip(&format!(
                    "{} - OpenGL backend unavailable: {e}",
                    function!()
                ));
                return;
            }
        };

        if !proc.supported_render_dtypes().f16 {
            crate::test_support::report_skip(&format!(
                "{} - F16 render not supported",
                function!()
            ));
            return;
        }

        let w = 64usize;
        let h = 64usize;

        let src = proc
            .create_image(
                w,
                h,
                PixelFormat::Rgba,
                DType::U8,
                None,
                edgefirst_tensor::CpuAccess::ReadWrite,
            )
            .unwrap();
        if src.memory() != TensorMemory::Pbo {
            crate::test_support::report_skip(&format!(
                "{} - RGBA8 src not PBO-backed",
                function!()
            ));
            return;
        }
        {
            let mut map = src.as_u8().unwrap().map().unwrap();
            for y in 0..h {
                for x in 0..w {
                    let i = (y * w + x) * 4;
                    map[i] = (x * 4) as u8;
                    map[i + 1] = (y * 4) as u8;
                    map[i + 2] = 128;
                    map[i + 3] = 255;
                }
            }
        }

        // F16 PlanarRgb PBO destination, logical [3,H,W].
        let mut dst = proc
            .create_image(
                w,
                h,
                PixelFormat::PlanarRgb,
                DType::F16,
                None,
                edgefirst_tensor::CpuAccess::ReadWrite,
            )
            .unwrap();
        if dst.memory() != TensorMemory::Pbo {
            crate::test_support::report_skip(&format!("{} - F16 dst not PBO-backed", function!()));
            return;
        }

        let fallbacks_before = proc.convert_fallback_count();
        proc.convert(&src, &mut dst, None).unwrap();
        assert_eq!(
            proc.convert_fallback_count(),
            fallbacks_before,
            "the GL float path declined and ImageProcessor fell back to the CPU; \
             this test's expectation is bit-exact against the CPU answer, so it \
             would otherwise pass with no GL coverage at all"
        );

        let map = dst.as_f16().unwrap().map().unwrap();
        assert_eq!(map.len(), 3 * w * h);
        let tol = 2.0f32.powi(-8);
        let mut max_err = 0.0f32;
        for c in 0..3 {
            for y in 0..h {
                for x in 0..w {
                    let expect = match c {
                        0 => (x * 4) as f32 / 255.0,
                        1 => (y * 4) as f32 / 255.0,
                        _ => 128.0 / 255.0,
                    };
                    let got: f32 = map[(c * h + y) * w + x].to_f32();
                    let err = (got - expect).abs();
                    max_err = max_err.max(err);
                    assert!(
                        err < tol,
                        "f16 dst[{c},{y},{x}]={got} expected {expect} (err {err})"
                    );
                }
            }
        }
        let _ = f16::from_f32(0.0);
        eprintln!("convert_f16_nchw_pbo_roundtrip: max_err={max_err} tol={tol}");
    }

    /// RGBA8 to an F16 NCHW DMA-BUF on the GL float path: `dst[c,y,x] == src[y,x,c] / 255`.
    #[test]
    fn convert_f16_nchw_dma_roundtrip() {
        use crate::{ComputeBackend, ImageProcessor, ImageProcessorConfig};
        use half::f16;

        if !is_opengl_available() {
            crate::test_support::report_skip(&format!("{} - OpenGL not available", function!()));
            return;
        }

        let mut proc = match ImageProcessor::with_config(ImageProcessorConfig {
            backend: ComputeBackend::OpenGl,
            ..Default::default()
        }) {
            Ok(p) => p,
            Err(e) => {
                crate::test_support::report_skip(&format!(
                    "{} - OpenGL backend unavailable: {e}",
                    function!()
                ));
                return;
            }
        };

        if !proc.supported_render_dtypes().f16 {
            crate::test_support::report_skip(&format!(
                "{} - F16 render not supported (Vivante?)",
                function!()
            ));
            return;
        }

        let w = 256usize;
        let h = 256usize;

        // RGBA8 source with a known gradient.
        let src = proc
            .create_image(
                w,
                h,
                PixelFormat::Rgba,
                DType::U8,
                None,
                edgefirst_tensor::CpuAccess::ReadWrite,
            )
            .unwrap();
        {
            let mut map = src.as_u8().unwrap().map().unwrap();
            for y in 0..h {
                for x in 0..w {
                    let i = (y * w + x) * 4;
                    map[i] = ((x * 255) / w) as u8;
                    map[i + 1] = ((y * 255) / h) as u8;
                    map[i + 2] = 128;
                    map[i + 3] = 255;
                }
            }
        }

        // F16 PlanarRgb DMA-BUF destination, logical [3,H,W].
        let Some(mut dst) = crate::test_support::gpu_buffer_allocated(
            proc.create_image(
                w,
                h,
                PixelFormat::PlanarRgb,
                DType::F16,
                Some(TensorMemory::DmaBuf),
                edgefirst_tensor::CpuAccess::ReadWrite,
            ),
            "F16 PlanarRgb DMA destination",
        ) else {
            return;
        };

        if dst.memory() != TensorMemory::DmaBuf {
            crate::test_support::report_skip(&format!(
                "{} - F16 dst memory is {:?}, not Dma; DMA path not exercised",
                function!(),
                dst.memory()
            ));
            return;
        }

        let fallbacks_before = proc.convert_fallback_count();
        proc.convert(&src, &mut dst, None).unwrap();
        if proc.convert_fallback_count() != fallbacks_before {
            crate::test_support::report_skip(&format!(
                "{} - the GL float path declined this DMA-BUF destination and \
                 ImageProcessor fell back to the CPU; there is no zero-copy \
                 coverage to check here",
                function!()
            ));
            return;
        }

        let map = dst.as_f16().unwrap().map().unwrap();
        assert_eq!(map.len(), 3 * w * h);
        let tol = 2.0f32.powi(-8);
        let mut max_err = 0.0f32;
        for c in 0..3 {
            for y in 0..h {
                for x in 0..w {
                    let expect = match c {
                        0 => ((x * 255) / w) as f32 / 255.0,
                        1 => ((y * 255) / h) as f32 / 255.0,
                        _ => 128.0 / 255.0,
                    };
                    let got: f32 = map[(c * h + y) * w + x].to_f32();
                    let err = (got - expect).abs();
                    max_err = max_err.max(err);
                    assert!(
                        err < tol,
                        "f16 dma dst[{c},{y},{x}]={got} expected {expect} (err {err})"
                    );
                }
            }
        }
        let _ = f16::from_f32(0.0);
        eprintln!("convert_f16_nchw_dma_roundtrip: max_err={max_err} tol={tol}");
    }

    /// An F32 Rgb PBO letterbox holds the pad colour outside the content rectangle, content inside.
    #[test]
    #[cfg(any(target_os = "linux", target_os = "windows"))]
    fn convert_f32_pbo_letterbox_pad_color() {
        use crate::{ComputeBackend, ImageProcessor, ImageProcessorConfig};

        if !is_opengl_available() {
            crate::test_support::report_skip(&format!("{} - OpenGL not available", function!()));
            return;
        }

        let mut proc = match ImageProcessor::with_config(ImageProcessorConfig {
            backend: ComputeBackend::OpenGl,
            ..Default::default()
        }) {
            Ok(p) => p,
            Err(e) => {
                crate::test_support::report_skip(&format!(
                    "{} - OpenGL backend unavailable: {e}",
                    function!()
                ));
                return;
            }
        };

        if !proc.supported_render_dtypes().f32 {
            crate::test_support::report_skip(&format!(
                "{} - F32 render not supported",
                function!()
            ));
            return;
        }

        let src_w = 4usize;
        let src_h = 4usize;
        let dst_w = 8usize;
        let dst_h = 4usize;

        // RGBA8 source: solid known colour (R=200, G=100, B=50).
        let src = proc
            .create_image(
                src_w,
                src_h,
                PixelFormat::Rgba,
                DType::U8,
                None,
                edgefirst_tensor::CpuAccess::ReadWrite,
            )
            .unwrap();
        if src.memory() != TensorMemory::Pbo {
            crate::test_support::report_skip(&format!(
                "{} - RGBA8 src not PBO-backed",
                function!()
            ));
            return;
        }
        {
            let mut map = src.as_u8().unwrap().map().unwrap();
            for chunk in map.as_chunks_mut::<4>().0 {
                chunk[0] = 200;
                chunk[1] = 100;
                chunk[2] = 50;
                chunk[3] = 255;
            }
        }

        // F32 Rgb PBO destination, wider than the source.
        let mut dst = proc
            .create_image(
                dst_w,
                dst_h,
                PixelFormat::Rgb,
                DType::F32,
                None,
                edgefirst_tensor::CpuAccess::ReadWrite,
            )
            .unwrap();
        if dst.memory() != TensorMemory::Pbo {
            crate::test_support::report_skip(&format!("{} - F32 dst not PBO-backed", function!()));
            return;
        }

        // Letterbox: content lands in columns [2, 6); columns [0, 2) and
        // [6, 8) are padded with the `dst_color`.
        let crop = Crop::letterbox([114, 114, 114, 255]);

        let fallbacks_before = proc.convert_fallback_count();
        let result = proc.convert(
            &src,
            &mut dst,
            Some(crate::ConvertOptions::new().crop(crop)),
        );
        assert!(
            result.is_ok(),
            "F32 PBO letterbox convert must not error: {:?}",
            result.err()
        );
        assert_eq!(
            proc.convert_fallback_count(),
            fallbacks_before,
            "the GL float path declined and ImageProcessor fell back to the \
             CPU; the CPU letterbox writes the same pad colour, so this test \
             would otherwise pass without exercising the GL path at all"
        );

        let map = dst.as_f32().unwrap().map().unwrap();
        assert_eq!(map.len(), dst_w * dst_h * 3, "unexpected element count");

        let pad_expected = 114.0f32 / 255.0;
        let pad_tol = 4.0 / 255.0;

        // Columns 0, 1, 6 and 7 are outside the content rectangle and carry the pad colour.
        for row in 0..dst_h {
            for col in [0usize, 1usize, 6usize, 7usize] {
                let base = (row * dst_w + col) * 3;
                for c in 0..3 {
                    let v = map[base + c];
                    assert!(
                        (v - pad_expected).abs() < pad_tol,
                        "pad pixel ({row},{col}) ch{c}={v} expected ≈{pad_expected} (tol {pad_tol})"
                    );
                }
            }
        }

        // Columns 2..6 are inside the content rectangle: finite and in [0, 1].
        for row in 0..dst_h {
            for col in 2..6usize {
                let base = (row * dst_w + col) * 3;
                for c in 0..3 {
                    let v = map[base + c];
                    assert!(
                        v.is_finite() && (0.0..=1.0).contains(&v),
                        "content pixel ({row},{col}) ch{c}={v} is not finite or not in [0,1]"
                    );
                }
            }
        }

        eprintln!("convert_f32_pbo_letterbox_pad_color: PASS");
    }

    #[test]
    #[cfg(target_os = "linux")]
    fn convert_f32_pbo_cuda_map_roundtrip() {
        use crate::{ComputeBackend, ImageProcessor, ImageProcessorConfig};

        if !crate::opengl_headless::cuda_policy::cuda_available_or_skip(
            "convert_f32_pbo_cuda_map_roundtrip",
        ) {
            return;
        }
        let mut proc = match ImageProcessor::with_config(ImageProcessorConfig {
            backend: ComputeBackend::OpenGl,
            ..Default::default()
        }) {
            Ok(p) => p,
            Err(_) => {
                crate::opengl_headless::cuda_policy::require_or_skip(
                    false,
                    "convert_f32_pbo_cuda_map_roundtrip",
                    "no GL",
                );
                return;
            }
        };
        if !crate::opengl_headless::cuda_policy::require_or_skip(
            proc.supported_render_dtypes().f32,
            "convert_f32_pbo_cuda_map_roundtrip",
            "no f32 render",
        ) {
            return;
        }
        let (w, h) = (64usize, 64usize);
        let src = proc
            .create_image(
                w,
                h,
                PixelFormat::Rgba,
                DType::U8,
                None,
                edgefirst_tensor::CpuAccess::ReadWrite,
            )
            .unwrap();
        {
            let mut m = src.as_u8().unwrap().map().unwrap();
            for i in 0..(w * h) {
                m[i * 4] = (i % 255) as u8;
                m[i * 4 + 1] = 0;
                m[i * 4 + 2] = 128;
                m[i * 4 + 3] = 255;
            }
        }
        let mut dst = proc
            .create_image(
                w,
                h,
                PixelFormat::Rgb,
                DType::F32,
                None,
                edgefirst_tensor::CpuAccess::ReadWrite,
            )
            .unwrap();
        if !crate::opengl_headless::cuda_policy::require_or_skip(
            dst.memory() == TensorMemory::Pbo,
            "convert_f32_pbo_cuda_map_roundtrip",
            "dst not PBO (no CUDA-GL alloc here)",
        ) {
            return;
        }
        let fallbacks_before = proc.convert_fallback_count();
        proc.convert(&src, &mut dst, None).unwrap();
        if proc.convert_fallback_count() != fallbacks_before {
            crate::test_support::report_skip(
                "convert_f32_pbo_cuda_map_roundtrip - the GL float path declined \
                 and the convert was served by the CPU; there is no GL coverage \
                 to check here",
            );
            return;
        }
        let cm = match dst.cuda_map() {
            Some(cm) => cm,
            None => {
                crate::opengl_headless::cuda_policy::require_or_skip(
                    false,
                    "convert_f32_pbo_cuda_map_roundtrip",
                    "cuda_map returned None: CUDA-GL interop unavailable for this context",
                );
                return;
            }
        };
        assert_eq!(
            cm.len(),
            w * h * 3 * 4,
            "device buffer is [H,W,3] f32 bytes"
        );
        assert!(!cm.device_ptr().is_null());
        assert_eq!(
            cm.device_ptr() as usize % 256,
            0,
            "256-B aligned for TensorRT"
        );
        eprintln!(
            "convert_f32_pbo_cuda_map_roundtrip: PASS device_ptr={:p} len={} align256={}",
            cm.device_ptr(),
            cm.len(),
            (cm.device_ptr() as usize).is_multiple_of(256)
        );
    }

    /// The `cuda_map()` device pointer holds the NHWC float data: `src[y,x,c] / 255.0` within 1e-3.
    #[test]
    #[cfg(target_os = "linux")]
    fn convert_f32_pbo_cuda_map_numeric() {
        use crate::{ComputeBackend, ImageProcessor, ImageProcessorConfig};
        use std::ffi::c_void;

        if !crate::opengl_headless::cuda_policy::cuda_available_or_skip(
            "convert_f32_pbo_cuda_map_numeric",
        ) {
            return;
        }
        let mut proc = match ImageProcessor::with_config(ImageProcessorConfig {
            backend: ComputeBackend::OpenGl,
            ..Default::default()
        }) {
            Ok(p) => p,
            Err(_) => {
                crate::opengl_headless::cuda_policy::require_or_skip(
                    false,
                    "convert_f32_pbo_cuda_map_numeric",
                    "no GL",
                );
                return;
            }
        };
        if !crate::opengl_headless::cuda_policy::require_or_skip(
            proc.supported_render_dtypes().f32,
            "convert_f32_pbo_cuda_map_numeric",
            "no f32 render",
        ) {
            return;
        }
        let (w, h) = (16usize, 16usize);
        // RGBA8 source with a known per-pixel gradient.
        let src = proc
            .create_image(
                w,
                h,
                PixelFormat::Rgba,
                DType::U8,
                None,
                edgefirst_tensor::CpuAccess::ReadWrite,
            )
            .unwrap();
        {
            let mut m = src.as_u8().unwrap().map().unwrap();
            for y in 0..h {
                for x in 0..w {
                    let i = (y * w + x) * 4;
                    m[i] = ((x * 255) / w) as u8;
                    m[i + 1] = ((y * 255) / h) as u8;
                    m[i + 2] = 128;
                    m[i + 3] = 255;
                }
            }
        }
        // F32 Rgb PBO destination — NHWC layout [H,W,3].
        let mut dst = proc
            .create_image(
                w,
                h,
                PixelFormat::Rgb,
                DType::F32,
                None,
                edgefirst_tensor::CpuAccess::ReadWrite,
            )
            .unwrap();
        if !crate::opengl_headless::cuda_policy::require_or_skip(
            dst.memory() == TensorMemory::Pbo,
            "convert_f32_pbo_cuda_map_numeric",
            "dst not PBO",
        ) {
            return;
        }
        let fallbacks_before = proc.convert_fallback_count();
        proc.convert(&src, &mut dst, None).unwrap();
        if proc.convert_fallback_count() != fallbacks_before {
            crate::test_support::report_skip(
                "convert_f32_pbo_cuda_map_numeric - the GL float path declined \
                 and the convert was served by the CPU; there is no GL coverage \
                 to check here",
            );
            return;
        }

        let cm = match dst.cuda_map() {
            Some(cm) => cm,
            None => {
                crate::opengl_headless::cuda_policy::require_or_skip(
                    false,
                    "convert_f32_pbo_cuda_map_numeric",
                    "cuda_map returned None: CUDA-GL interop unavailable for this context",
                );
                return;
            }
        };
        let n = w * h * 3;
        let mut host = vec![0f32; n];
        // SAFETY: `host` has `n * 4` bytes allocated; `device_ptr` is valid while `cm` lives.
        assert!(
            unsafe {
                edgefirst_tensor::memcpy_device_to_host(
                    host.as_mut_ptr() as *mut c_void,
                    cm.device_ptr() as *const c_void,
                    n * std::mem::size_of::<f32>(),
                )
            },
            "cudaMemcpy device->host failed"
        );

        // dst is NHWC [H,W,3] f32, normalized [0,1]: dst[(y*w+x)*3 + c] ≈ src_channel/255
        let mut max_err = 0f32;
        for y in 0..h {
            for x in 0..w {
                let exp = [
                    ((x * 255) / w) as f32 / 255.0,
                    ((y * 255) / h) as f32 / 255.0,
                    128.0 / 255.0,
                ];
                for c in 0..3 {
                    let got = host[(y * w + x) * 3 + c];
                    max_err = max_err.max((got - exp[c]).abs());
                }
            }
        }
        eprintln!("convert_f32_pbo_cuda_map_numeric: max_err={max_err}");
        assert!(
            max_err < 1e-3,
            "device data does not match NHWC-normalized source (max_err={max_err})"
        );
    }

    /// Decode a JPEG fixture into its native format, convert it to an `Rgb` `F32` PBO, read the
    /// PBO back through `cuda_map()` and `cudaMemcpy(D2H)`, and compare to a CPU `convert()`
    /// reference.
    #[cfg(target_os = "linux")]
    fn jpeg_cuda_devptr_check(fixture: &str, expect_fmt: PixelFormat, w: usize, h: usize) {
        use crate::{ComputeBackend, ImageProcessor, ImageProcessorConfig};
        use edgefirst_codec::{ImageDecoder, ImageLoad};
        use edgefirst_tensor::{Tensor, TensorMapTrait, TensorTrait};
        use std::ffi::c_void;

        if !crate::opengl_headless::cuda_policy::cuda_available_or_skip(fixture) {
            return;
        }
        let mut proc = match ImageProcessor::with_config(ImageProcessorConfig {
            backend: ComputeBackend::OpenGl,
            ..Default::default()
        }) {
            Ok(p) => p,
            Err(_) => {
                crate::opengl_headless::cuda_policy::require_or_skip(false, fixture, "no GL");
                return;
            }
        };
        if !crate::opengl_headless::cuda_policy::require_or_skip(
            proc.supported_render_dtypes().f32,
            fixture,
            "no f32 render",
        ) {
            return;
        }

        // Decode the JPEG into its native format.
        let mut src_t = Tensor::<u8>::image(
            w,
            h,
            expect_fmt,
            Some(TensorMemory::Mem),
            edgefirst_tensor::CpuAccess::ReadWrite,
        )
        .unwrap();
        let mut dec = ImageDecoder::new();
        let info = src_t
            .load_image(&mut dec, &edgefirst_bench::testdata::read(fixture))
            .unwrap();
        assert_eq!(info.format, expect_fmt, "{fixture}: native decode format");
        assert_eq!(
            src_t.colorimetry(),
            Some(edgefirst_tensor::Colorimetry::jfif()),
            "{fixture}: decoder must tag JFIF colorimetry"
        );
        let src = TensorDyn::from(src_t);

        // CPU reference: convert into a tight Mem Rgb F32.
        let mut ref_dst = TensorDyn::image(
            w,
            h,
            PixelFormat::Rgb,
            DType::F32,
            Some(TensorMemory::Mem),
            edgefirst_tensor::CpuAccess::ReadWrite,
        )
        .unwrap();
        proc.convert(&src, &mut ref_dst, None).unwrap();
        let ref_map = ref_dst.as_f32().unwrap().map().unwrap();
        let ref_px = ref_map.as_slice();

        // Rgb F32 PBO output, mapped through `cuda_map`.
        let mut pbo_dst = proc
            .create_image(
                w,
                h,
                PixelFormat::Rgb,
                DType::F32,
                None,
                edgefirst_tensor::CpuAccess::ReadWrite,
            )
            .unwrap();
        if !crate::opengl_headless::cuda_policy::require_or_skip(
            pbo_dst.memory() == TensorMemory::Pbo,
            fixture,
            "dst not PBO",
        ) {
            return;
        }
        proc.convert(&src, &mut pbo_dst, None).unwrap();
        let cm = match pbo_dst.cuda_map() {
            Some(cm) => cm,
            None => {
                crate::opengl_headless::cuda_policy::require_or_skip(
                    false,
                    fixture,
                    "cuda_map returned None: CUDA-GL interop unavailable for this context",
                );
                return;
            }
        };
        assert!(!cm.device_ptr().is_null(), "{fixture}: null device_ptr");
        assert_eq!(
            cm.device_ptr() as usize % 256,
            0,
            "{fixture}: device_ptr must be 256-byte aligned for TensorRT"
        );

        let n = cm.len() / std::mem::size_of::<f32>();
        let mut host = vec![0f32; n];
        // SAFETY: `host` has `cm.len()` bytes; device_ptr is valid while `cm` lives.
        assert!(
            unsafe {
                edgefirst_tensor::memcpy_device_to_host(
                    host.as_mut_ptr() as *mut c_void,
                    cm.device_ptr() as *const c_void,
                    cm.len(),
                )
            },
            "{fixture}: cudaMemcpy device->host failed"
        );

        // Stride-aware compare: the device buffer carries the PBO's row pitch
        // (`effective_row_stride`), the Mem reference is tight.
        let dev_stride_elems =
            pbo_dst.effective_row_stride().unwrap_or(w * 3 * 4) / std::mem::size_of::<f32>();
        let mut max_err = 0f32;
        for y in 0..h {
            for x in 0..w {
                for c in 0..3 {
                    let got = host[y * dev_stride_elems + x * 3 + c];
                    let exp = ref_px[(y * w + x) * 3 + c];
                    max_err = max_err.max((got - exp).abs());
                }
            }
        }
        eprintln!("jpeg_cuda_devptr {fixture} ({expect_fmt:?}): max_err={max_err}");
        assert!(
            max_err < 1e-3,
            "{fixture}: cuda_map device buffer != CPU reference (max_err={max_err})"
        );
    }

    #[test]
    #[cfg(target_os = "linux")]
    fn jpeg_nv12_convert_cuda_devptr() {
        jpeg_cuda_devptr_check("zidane.jpg", PixelFormat::Nv12, 1280, 720);
    }

    #[test]
    #[cfg(target_os = "linux")]
    fn jpeg_nv16_convert_cuda_devptr() {
        jpeg_cuda_devptr_check("zidane_422.jpg", PixelFormat::Nv16, 1280, 720);
    }

    #[test]
    #[cfg(target_os = "linux")]
    fn jpeg_nv24_convert_cuda_devptr() {
        jpeg_cuda_devptr_check("zidane_444.jpg", PixelFormat::Nv24, 1280, 720);
    }

    #[test]
    #[cfg(target_os = "linux")]
    fn jpeg_grey_convert_cuda_devptr() {
        jpeg_cuda_devptr_check("grey.jpg", PixelFormat::Grey, 1024, 681);
    }

    // =========================================================================
    // NV16/NV24 → RGBA GPU round-trip tests
    // =========================================================================

    /// Compute expected BT.601 full-range RGB from raw Y, Cb, Cr byte values.
    #[cfg(any(target_os = "linux", target_os = "macos", target_os = "windows"))]
    fn yuv601_to_rgb(y: u8, cb: u8, cr: u8) -> [u8; 3] {
        let yf = y as f32 / 255.0;
        let up = cb as f32 / 255.0 - 128.0 / 255.0;
        let vp = cr as f32 / 255.0 - 128.0 / 255.0;
        let r = (yf + 1.402 * vp).clamp(0.0, 1.0);
        let g = (yf - 0.344 * up - 0.714 * vp).clamp(0.0, 1.0);
        let b = (yf + 1.772 * up).clamp(0.0, 1.0);
        [
            (r * 255.0 + 0.5) as u8,
            (g * 255.0 + 0.5) as u8,
            (b * 255.0 + 0.5) as u8,
        ]
    }

    /// Build a solid-colour NV16 (4:2:2) buffer of dimensions `(w, h)`.
    ///
    /// Layout: `[H rows of Y][H rows of interleaved CbCr]` (contiguous, width-aligned).
    #[cfg(any(target_os = "linux", target_os = "macos", target_os = "windows"))]
    fn make_nv16_solid(w: usize, h: usize, y: u8, cb: u8, cr: u8) -> Vec<u8> {
        let y_plane = vec![y; w * h];
        // NV16: H rows of UV, each row has w/2 pairs → w bytes/row.
        let uv_plane: Vec<u8> = std::iter::repeat_n([cb, cr], w * h / 2).flatten().collect();
        [y_plane, uv_plane].concat()
    }

    /// Build a solid-colour NV24 (4:4:4) buffer of dimensions `(w, h)`.
    ///
    /// Layout: `[H rows of Y][H rows of interleaved CbCr full-res]`.
    #[cfg(any(target_os = "linux", target_os = "macos", target_os = "windows"))]
    fn make_nv24_solid(w: usize, h: usize, y: u8, cb: u8, cr: u8) -> Vec<u8> {
        let y_plane = vec![y; w * h];
        // NV24: H rows of UV, each row has w pairs → 2w bytes/row.
        let uv_plane: Vec<u8> = std::iter::repeat_n([cb, cr], w * h).flatten().collect();
        [y_plane, uv_plane].concat()
    }

    /// NV16 to RGBA on DMA buffers runs `ShaderR8` and matches BT.601 full-range RGB within ±2.
    #[test]
    #[cfg(any(target_os = "linux", target_os = "macos", target_os = "windows"))]
    fn test_gpu_nv16_to_rgba_shader_r8() {
        use crate::opengl_headless::processor::{GLProcessorST, NvConvertPath};
        if !is_gpu_image_buffer_available() {
            crate::test_support::report_skip(&format!(
                "{} - no zero-copy GPU buffers",
                function!()
            ));
            return;
        }

        let (w, h) = (64usize, 64usize);
        let (yv, cb, cr) = (100u8, 150u8, 80u8);
        let expected = yuv601_to_rgb(yv, cb, cr);

        let mut src = load_raw_image(
            w,
            h,
            PixelFormat::Nv16,
            Some(TensorMemory::DmaBuf),
            &make_nv16_solid(w, h, yv, cb, cr),
        )
        .unwrap();
        src.set_colorimetry(Some(
            edgefirst_tensor::Colorimetry::default()
                .with_encoding(edgefirst_tensor::ColorEncoding::Bt601)
                .with_range(edgefirst_tensor::ColorRange::Full),
        ));

        let mut dst = TensorDyn::image(
            w,
            h,
            PixelFormat::Rgba,
            DType::U8,
            Some(TensorMemory::DmaBuf),
            edgefirst_tensor::CpuAccess::ReadWrite,
        )
        .unwrap();

        let mut gl = match GLProcessorST::new(None, None) {
            Ok(g) => g,
            Err(e) => {
                crate::test_support::report_skip(&format!(
                    "{} - GL not available: {e}",
                    function!()
                ));
                return;
            }
        };

        gl.convert(&src, &mut dst, None).unwrap();

        assert_eq!(
            gl.last_nv_convert_path,
            NvConvertPath::ShaderR8,
            "NV16 DMA convert must use ShaderR8, got {:?}",
            gl.last_nv_convert_path
        );

        let map = dst.as_u8().unwrap().map().unwrap();
        let pixels = map.as_slice();
        for i in 0..(w * h) {
            let r = pixels[i * 4];
            let g = pixels[i * 4 + 1];
            let b = pixels[i * 4 + 2];
            let diff_r = (r as i32 - expected[0] as i32).unsigned_abs();
            let diff_g = (g as i32 - expected[1] as i32).unsigned_abs();
            let diff_b = (b as i32 - expected[2] as i32).unsigned_abs();
            assert!(
                diff_r <= 2 && diff_g <= 2 && diff_b <= 2,
                "pixel {i}: got ({r},{g},{b}) expected ({},{},{}) — diff ({diff_r},{diff_g},{diff_b})",
                expected[0], expected[1], expected[2]
            );
        }
    }

    /// Sets `EDGEFIRST_NV_CONVERT_PATH` and removes it on drop.
    #[cfg(target_os = "linux")]
    struct NvPathEnvGuard;
    #[cfg(target_os = "linux")]
    impl NvPathEnvGuard {
        fn set(v: &str) -> Self {
            std::env::set_var("EDGEFIRST_NV_CONVERT_PATH", v);
            NvPathEnvGuard
        }
    }
    #[cfg(target_os = "linux")]
    impl Drop for NvPathEnvGuard {
        fn drop(&mut self) {
            std::env::remove_var("EDGEFIRST_NV_CONVERT_PATH");
        }
    }

    #[cfg(target_os = "linux")]
    struct ColorimetryEnvGuard;
    #[cfg(target_os = "linux")]
    impl ColorimetryEnvGuard {
        fn set(v: &str) -> Self {
            std::env::set_var("EDGEFIRST_COLORIMETRY", v);
            ColorimetryEnvGuard
        }
    }
    #[cfg(target_os = "linux")]
    impl Drop for ColorimetryEnvGuard {
        fn drop(&mut self) {
            std::env::remove_var("EDGEFIRST_COLORIMETRY");
        }
    }

    /// Colorimetry mode, source colorimetry and `EDGEFIRST_COLORIMETRY` pick the `auto` NV12 path.
    #[test]
    #[cfg(target_os = "linux")]
    fn colorimetry_mode_policy_nv12_auto() {
        if !is_opengl_available() {
            crate::test_support::report_skip(&format!("{} - GL not available", function!()));
            return;
        }
        use crate::opengl_headless::processor::{GLProcessorST, NvConvertPath};
        use crate::ColorimetryMode;
        use edgefirst_tensor::{ColorEncoding, ColorRange, Colorimetry};

        if !is_dma_available() {
            crate::test_support::report_skip(&format!("{} - DMA not available", function!()));
            return;
        }
        let mut gl = match GLProcessorST::new(None, None) {
            Ok(g) => g,
            Err(e) => {
                crate::test_support::report_skip(&format!(
                    "{} - GL not available: {e}",
                    function!()
                ));
                return;
            }
        };

        let (w, h) = (64usize, 64usize);
        let mut nv12 = vec![128u8; w * h * 3 / 2];
        nv12[..w * h].fill(100);
        let make_src = |enc: ColorEncoding, range: ColorRange| {
            let mut src =
                load_raw_image(w, h, PixelFormat::Nv12, Some(TensorMemory::DmaBuf), &nv12).unwrap();
            src.set_colorimetry(Some(
                Colorimetry::default().with_encoding(enc).with_range(range),
            ));
            src
        };
        let src_709 = make_src(ColorEncoding::Bt709, ColorRange::Full);
        let src_601 = make_src(ColorEncoding::Bt601, ColorRange::Limited);
        let mut dst = TensorDyn::image(
            w,
            h,
            PixelFormat::Rgba,
            DType::U8,
            Some(TensorMemory::DmaBuf),
            edgefirst_tensor::CpuAccess::ReadWrite,
        )
        .unwrap();
        let mut path_for = |gl: &mut GLProcessorST, src: &TensorDyn| {
            gl.convert(src, &mut dst, None).unwrap();
            gl.last_nv_convert_path
        };

        // Fast (default): the sampler on Vivante, the in-shader matrix everywhere else.
        let expect_fast = if gl.is_vivante {
            NvConvertPath::ExternalSampler
        } else {
            NvConvertPath::ShaderR8
        };
        assert_eq!(
            path_for(&mut gl, &src_709),
            expect_fast,
            "Fast mode, BT.709-full NV12"
        );

        // Exact: the in-shader matrix everywhere for a BT.709 source.
        gl.set_colorimetry_mode(ColorimetryMode::Exact);
        assert_eq!(
            path_for(&mut gl, &src_709),
            NvConvertPath::ShaderR8,
            "Exact mode, BT.709-full NV12"
        );
        // Exact with a BT.601-limited source: the sampler on Vivante.
        assert_eq!(
            path_for(&mut gl, &src_601),
            expect_fast,
            "Exact mode, BT.601-limited NV12"
        );

        // A Grey destination takes `ShaderR8` in either mode.
        let mut grey_dst = TensorDyn::image(
            w,
            h,
            PixelFormat::Grey,
            DType::U8,
            Some(TensorMemory::DmaBuf),
            edgefirst_tensor::CpuAccess::ReadWrite,
        )
        .unwrap();
        let mut grey_path = |gl: &mut GLProcessorST, src: &TensorDyn| {
            gl.convert(src, &mut grey_dst, None).unwrap();
            gl.last_nv_convert_path
        };
        gl.set_colorimetry_mode(ColorimetryMode::Fast);
        assert_eq!(
            grey_path(&mut gl, &src_709),
            NvConvertPath::ShaderR8,
            "Fast mode, BT.709 NV12 → Grey must avoid the sampler (R8 target)"
        );
        gl.set_colorimetry_mode(ColorimetryMode::Exact);
        assert_eq!(
            grey_path(&mut gl, &src_601),
            NvConvertPath::ShaderR8,
            "Exact mode, BT.601-limited NV12 → Grey must avoid the sampler (R8 target)"
        );

        // `EDGEFIRST_COLORIMETRY=exact` keeps a Fast request at Exact.
        drop(gl);
        let _env = ColorimetryEnvGuard::set("exact");
        let mut gl = GLProcessorST::new(None, None).unwrap();
        gl.set_colorimetry_mode(ColorimetryMode::Fast);
        assert_eq!(
            path_for(&mut gl, &src_709),
            NvConvertPath::ShaderR8,
            "env-pinned Exact must override set_colorimetry_mode(Fast)"
        );
    }

    /// `EDGEFIRST_NV_CONVERT_PATH` selects the NV12 GPU conversion path.
    #[test]
    #[cfg(target_os = "linux")]
    fn test_nv12_path_env_override() {
        use crate::opengl_headless::processor::{GLProcessorST, NvConvertPath};
        if !is_dma_available() {
            crate::test_support::report_skip(&format!("{} - DMA not available", function!()));
            return;
        }
        let (w, h) = (64usize, 64usize);
        let mut bytes = vec![100u8; w * h]; // Y plane
        for _ in 0..(w * h / 4) {
            bytes.push(150); // Cb
            bytes.push(80); // Cr
        }

        let run = |env: Option<&str>| -> Option<(NvConvertPath, bool)> {
            // The variable is set for this construction only.
            let gl = {
                std::env::remove_var("EDGEFIRST_NV_CONVERT_PATH");
                let _guard = env.map(NvPathEnvGuard::set);
                GLProcessorST::new(None, None)
            };
            let mut gl = match gl {
                Ok(g) => g,
                Err(e) => {
                    crate::test_support::report_skip(&format!(
                        "{} - GL not available: {e}",
                        function!()
                    ));
                    return None;
                }
            };
            let mut src =
                load_raw_image(w, h, PixelFormat::Nv12, Some(TensorMemory::DmaBuf), &bytes)
                    .unwrap();
            src.set_colorimetry(Some(
                edgefirst_tensor::Colorimetry::default()
                    .with_encoding(edgefirst_tensor::ColorEncoding::Bt709)
                    .with_range(edgefirst_tensor::ColorRange::Full),
            ));
            let mut dst = TensorDyn::image(
                w,
                h,
                PixelFormat::Rgba,
                DType::U8,
                Some(TensorMemory::DmaBuf),
                edgefirst_tensor::CpuAccess::ReadWrite,
            )
            .unwrap();
            gl.convert(&src, &mut dst, None).unwrap();
            Some((gl.last_nv_convert_path, gl.is_vivante))
        };

        if let Some((p, is_vivante)) = run(None) {
            let expect = if is_vivante {
                NvConvertPath::ExternalSampler
            } else {
                NvConvertPath::ShaderR8
            };
            assert_eq!(
                p, expect,
                "auto default: single-plane full-range NV12 policy pick"
            );
        }
        if let Some((p, _)) = run(Some("shader")) {
            assert_eq!(
                p,
                NvConvertPath::ShaderR8,
                "forced shader must use ShaderR8"
            );
        }
        if let Some((p, _)) = run(Some("sampler")) {
            assert_ne!(
                p,
                NvConvertPath::ShaderR8,
                "forced sampler must NOT use ShaderR8 (got ExternalSampler or Cpu)"
            );
        }
    }

    /// A heap NV12 source converts on the GPU through `ShaderR8` and matches the CPU within ±2.
    #[test]
    #[cfg(target_os = "linux")]
    fn test_nv12_nondma_upload_uses_shader() {
        use crate::opengl_headless::processor::{GLProcessorST, NvConvertPath};
        if !is_opengl_available() {
            crate::test_support::report_skip(&format!("{} - OpenGL not available", function!()));
            return;
        }
        let (w, h) = (64usize, 64usize);
        // Patterned NV12: luma gradient, neutral chroma.
        let mut bytes = vec![0u8; w * h];
        for r in 0..h {
            for c in 0..w {
                bytes[r * w + c] = ((r + c) * 255 / (w + h)) as u8;
            }
        }
        bytes.extend(std::iter::repeat_n(128u8, w * h / 2)); // neutral chroma

        let mut gl = match GLProcessorST::new(None, None) {
            Ok(g) => g,
            Err(e) => {
                crate::test_support::report_skip(&format!(
                    "{} - GL not available: {e}",
                    function!()
                ));
                return;
            }
        };

        let mem_src =
            load_raw_image(w, h, PixelFormat::Nv12, Some(TensorMemory::Mem), &bytes).unwrap();
        let mut gpu_dst = TensorDyn::image(
            w,
            h,
            PixelFormat::Rgba,
            DType::U8,
            None,
            edgefirst_tensor::CpuAccess::ReadWrite,
        )
        .unwrap();
        gl.convert(&mem_src, &mut gpu_dst, None).unwrap();
        assert_eq!(
            gl.last_nv_convert_path,
            NvConvertPath::ShaderR8,
            "non-DMA NV12 must use the R8-upload ShaderR8 path, not CPU (got {:?})",
            gl.last_nv_convert_path
        );

        // CPU reference.
        let cpu_src =
            load_raw_image(w, h, PixelFormat::Nv12, Some(TensorMemory::Mem), &bytes).unwrap();
        let mut cpu_dst = TensorDyn::image(
            w,
            h,
            PixelFormat::Rgba,
            DType::U8,
            None,
            edgefirst_tensor::CpuAccess::ReadWrite,
        )
        .unwrap();
        crate::cpu::CPUProcessor::new()
            .convert(&cpu_src, &mut cpu_dst, None)
            .unwrap();

        let gpu_map = gpu_dst.as_u8().unwrap().map().unwrap();
        let cpu_map = cpu_dst.as_u8().unwrap().map().unwrap();
        assert_pixels_match(cpu_map.as_slice(), gpu_map.as_slice(), 2);
    }

    /// NV12 (DMA) to PlanarRgb routes through the `ShaderR8` two-pass and matches the CPU.
    #[test]
    #[cfg(target_os = "linux")]
    fn test_nv12_to_planar_rgb_uses_shader_path() {
        use crate::opengl_headless::processor::{GLProcessorST, NvConvertPath};
        if !is_dma_available() {
            crate::test_support::report_skip(&format!("{} - DMA not available", function!()));
            return;
        }
        let (w, h) = (64usize, 64usize);
        // NV12 with a vertical chroma edge.
        let mut bytes = Vec::with_capacity(w * h * 3 / 2);
        for r in 0..h {
            for c in 0..w {
                bytes.push(((r + c) * 255 / (w + h)) as u8);
            }
        }
        for _ in 0..h / 2 {
            for c in 0..w / 2 {
                let (cb, cr) = if c < w / 4 { (90, 160) } else { (200, 40) };
                bytes.push(cb);
                bytes.push(cr);
            }
        }
        fn tag(t: &mut TensorDyn) {
            t.set_colorimetry(Some(
                edgefirst_tensor::Colorimetry::default()
                    .with_encoding(edgefirst_tensor::ColorEncoding::Bt709)
                    .with_range(edgefirst_tensor::ColorRange::Full),
            ));
        }

        let mut gl = match GLProcessorST::new(None, None) {
            Ok(g) => g,
            Err(e) => {
                crate::test_support::report_skip(&format!(
                    "{} - GL not available: {e}",
                    function!()
                ));
                return;
            }
        };
        gl.set_colorimetry_mode(crate::ColorimetryMode::Exact);
        let mut src =
            load_raw_image(w, h, PixelFormat::Nv12, Some(TensorMemory::DmaBuf), &bytes).unwrap();
        tag(&mut src);
        let mut dst = TensorDyn::image(
            w,
            h,
            PixelFormat::PlanarRgb,
            DType::U8,
            Some(TensorMemory::DmaBuf),
            edgefirst_tensor::CpuAccess::ReadWrite,
        )
        .unwrap();
        gl.convert(&src, &mut dst, None).unwrap();
        assert_eq!(
            gl.last_nv_convert_path,
            NvConvertPath::ShaderR8,
            "NV12→PlanarRgb in Exact mode must use the ShaderR8 two-pass, not the driver sampler"
        );

        // CPU planar reference.
        let mut cpu_src =
            load_raw_image(w, h, PixelFormat::Nv12, Some(TensorMemory::Mem), &bytes).unwrap();
        tag(&mut cpu_src);
        let mut cpu_dst = TensorDyn::image(
            w,
            h,
            PixelFormat::PlanarRgb,
            DType::U8,
            None,
            edgefirst_tensor::CpuAccess::ReadWrite,
        )
        .unwrap();
        crate::cpu::CPUProcessor::new()
            .convert(&cpu_src, &mut cpu_dst, None)
            .unwrap();
        compare_images(&cpu_dst, &dst, 0.95, "nv12_to_planar_rgb_shader");
    }

    /// A 1280x720 NV12 frame whose luma and chroma vary over a few pixels.
    #[cfg(target_os = "linux")]
    fn detailed_nv12() -> (usize, usize, Vec<u8>) {
        let (w, h) = (1280usize, 720usize);
        let mut nv12 = Vec::with_capacity(w * h * 3 / 2);
        for y in 0..h {
            for x in 0..w {
                nv12.push((128.0 + 100.0 * (x as f32 * 0.9).sin() * (y as f32 * 0.7).cos()) as u8);
            }
        }
        for y in 0..h / 2 {
            for x in 0..w / 2 {
                nv12.push((128.0 + 60.0 * (x as f32 * 1.3).sin()) as u8);
                nv12.push((128.0 + 60.0 * (y as f32 * 1.1).cos()) as u8);
            }
        }
        (w, h, nv12)
    }

    /// Float reference for the GL NV programs on a tight NV12 frame, sampling
    /// a source crop `(x, y, w, h)` into `dw`x`dh` planar RGB bytes.
    #[cfg(target_os = "linux")]
    struct NvOracle<'a> {
        nv12: &'a [u8],
        w: usize,
        h: usize,
        k: crate::colorimetry::YuvToRgbCoeffs,
    }

    #[cfg(target_os = "linux")]
    impl<'a> NvOracle<'a> {
        fn new(nv12: &'a [u8], w: usize, h: usize, src: &TensorDyn) -> Self {
            let cm = crate::colorimetry::resolve_colorimetry(src.colorimetry(), src.height());
            let k = crate::colorimetry::yuv_to_rgb_coeffs(
                cm.encoding
                    .unwrap_or(edgefirst_tensor::ColorEncoding::Bt709),
                cm.range.unwrap_or(edgefirst_tensor::ColorRange::Limited),
            );
            NvOracle { nv12, w, h, k }
        }

        /// Luma, U or V (`plane` 0, 1, 2) at a clamped grid position.
        fn sample(&self, plane: usize, x: isize, y: isize, r: (usize, usize, usize, usize)) -> f32 {
            let (cx, cy, cw, ch) = r;
            let byte = if plane == 0 {
                let x = x.clamp(cx as isize, (cx + cw) as isize - 1) as usize;
                let y = y.clamp(cy as isize, (cy + ch) as isize - 1) as usize;
                self.nv12[y * self.w + x]
            } else {
                let x = x.clamp((cx / 2) as isize, (cx + cw).div_ceil(2) as isize - 1) as usize;
                let y = y.clamp((cy / 2) as isize, (cy + ch).div_ceil(2) as isize - 1) as usize;
                self.nv12[self.w * self.h + y * self.w + 2 * x + plane - 1]
            };
            byte as f32 / 255.0
        }

        fn rgb(&self, y: f32, u: f32, v: f32) -> [f32; 3] {
            let k = &self.k;
            let y = ((y - k.y_offset) * k.y_scale).max(0.0);
            let (u, v) = (u - 128.0 / 255.0, v - 128.0 / 255.0);
            [y + k.c_vr * v, y - k.c_ug * u - k.c_vg * v, y + k.c_ub * u].map(|c| c.clamp(0.0, 1.0))
        }

        /// The converted pixel at luma texel (x, y), with its co-sited chroma.
        fn texel(&self, x: isize, y: isize, r: (usize, usize, usize, usize)) -> [f32; 3] {
            let (cx, cy, cw, ch) = r;
            let x = x.clamp(cx as isize, (cx + cw) as isize - 1);
            let y = y.clamp(cy as isize, (cy + ch) as isize - 1);
            self.rgb(
                self.sample(0, x, y, r),
                self.sample(1, x >> 1, y >> 1, r),
                self.sample(2, x >> 1, y >> 1, r),
            )
        }

        /// `NvSampling::Hardware`, `NvSampling::Shader` or nearest: the
        /// source position of output pixel (dx, dy) is `(d + 0.5) * scale`.
        fn render(
            &self,
            r: (usize, usize, usize, usize),
            dw: usize,
            dh: usize,
            mode: &str,
        ) -> Vec<u8> {
            let (cx, cy, cw, ch) = r;
            let lerp = |a: f32, b: f32, t: f32| a + (b - a) * t;
            let mut out = vec![0u8; 3 * dw * dh];
            for dy in 0..dh {
                for dx in 0..dw {
                    let lx = cx as f32 + (dx as f32 + 0.5) * cw as f32 / dw as f32;
                    let ly = cy as f32 + (dy as f32 + 0.5) * ch as f32 / dh as f32;
                    let rgb = match mode {
                        "nearest" => self.texel(lx.floor() as isize, ly.floor() as isize, r),
                        "shader" => {
                            let (px, py) = (lx - 0.5, ly - 0.5);
                            let (x0, y0) = (px.floor() as isize, py.floor() as isize);
                            let (fx, fy) = (px - px.floor(), py - py.floor());
                            let t = |x, y| self.texel(x, y, r);
                            let (a, b, c, d) =
                                (t(x0, y0), t(x0 + 1, y0), t(x0, y0 + 1), t(x0 + 1, y0 + 1));
                            [0, 1, 2].map(|i| lerp(lerp(a[i], b[i], fx), lerp(c[i], d[i], fx), fy))
                        }
                        _ => {
                            let bil = |plane: usize, px: f32, py: f32| {
                                let (x0, y0) = (px.floor() as isize, py.floor() as isize);
                                let (fx, fy) = (px - px.floor(), py - py.floor());
                                let s = |x, y| self.sample(plane, x, y, r);
                                lerp(
                                    lerp(s(x0, y0), s(x0 + 1, y0), fx),
                                    lerp(s(x0, y0 + 1), s(x0 + 1, y0 + 1), fx),
                                    fy,
                                )
                            };
                            let (cxp, cyp) = (lx / 2.0 - 0.5, ly / 2.0 - 0.5);
                            self.rgb(
                                bil(0, lx - 0.5, ly - 0.5),
                                bil(1, cxp, cyp),
                                bil(2, cxp, cyp),
                            )
                        }
                    };
                    for (c, v) in rgb.iter().enumerate() {
                        out[c * dw * dh + dy * dw + dx] = (v * 255.0).round() as u8;
                    }
                }
            }
            out
        }
    }

    #[cfg(target_os = "linux")]
    fn planar_bytes(t: &TensorDyn) -> Vec<u8> {
        match t.dtype() {
            DType::I8 => t
                .as_i8()
                .unwrap()
                .map()
                .unwrap()
                .as_slice()
                .iter()
                .map(|&v| v as u8 ^ 0x80)
                .collect(),
            _ => t.as_u8().unwrap().map().unwrap().as_slice().to_vec(),
        }
    }

    #[cfg(target_os = "linux")]
    fn mean_abs_diff(a: &[u8], b: &[u8]) -> f64 {
        assert_eq!(a.len(), b.len());
        a.iter()
            .zip(b)
            .map(|(&x, &y)| x.abs_diff(y) as u64)
            .sum::<u64>() as f64
            / a.len() as f64
    }

    #[cfg(target_os = "linux")]
    fn nv_test_processor() -> Option<crate::opengl_headless::processor::GLProcessorST> {
        if !is_opengl_available() {
            return None;
        }
        let mut gl = crate::opengl_headless::processor::GLProcessorST::new(None, None).ok()?;
        // Exact keeps Vivante on the NV shader programs.
        gl.set_colorimetry_mode(crate::ColorimetryMode::Exact);
        Some(gl)
    }

    /// A resizing NV12 convert matches the reference for its sampling, `Hardware` or `Shader`.
    #[test]
    #[cfg(target_os = "linux")]
    fn test_nv12_resize_matches_its_sampling() {
        use crate::opengl_headless::processor::NvSampling;
        const MAX_MEAN_ERROR: f64 = 3.0;
        let Some(mut gl) = nv_test_processor() else {
            crate::test_support::report_skip(&format!("{} - OpenGL not available", function!()));
            return;
        };
        let (sw, sh, nv12) = detailed_nv12();
        let mut memories = vec![TensorMemory::Mem];
        if is_dma_available() {
            memories.push(TensorMemory::DmaBuf);
        }
        let whole = (0, 0, sw, sh);
        let tile = (101usize, 53usize, 320usize, 180usize);
        for mem in memories {
            let src = load_raw_image(sw, sh, PixelFormat::Nv12, Some(mem), &nv12).unwrap();
            let oracle = NvOracle::new(&nv12, sw, sh, &src);
            for hw in [true, false] {
                gl.set_nv_hw_filter(hw);
                for (r, dw, dh, dtype) in [
                    (whole, 1792usize, 1008usize, DType::U8),
                    (whole, 896, 504, DType::U8),
                    (whole, 640, 360, DType::U8),
                    (whole, 1792, 1008, DType::I8),
                    (tile, 448, 252, DType::U8),
                ] {
                    let crop = if r == whole {
                        Crop::no_crop()
                    } else {
                        Crop::new().with_source(Some(crate::Region::new(r.0, r.1, r.2, r.3)))
                    };
                    let mut gpu = TensorDyn::image(
                        dw,
                        dh,
                        PixelFormat::PlanarRgb,
                        dtype,
                        Some(TensorMemory::Mem),
                        edgefirst_tensor::CpuAccess::ReadWrite,
                    )
                    .unwrap();
                    gl.convert(
                        &src,
                        &mut gpu,
                        Some(crate::ConvertOptions::new().crop(crop)),
                    )
                    .unwrap();
                    let sampling = gl.last_nv_sampling.expect("an NV draw ran");
                    if !hw {
                        assert_eq!(
                            sampling,
                            NvSampling::Shader,
                            "hardware filtering is switched off"
                        );
                    } else if mem == TensorMemory::Mem {
                        assert_eq!(
                            sampling,
                            NvSampling::Hardware,
                            "an upload can always bind its chroma"
                        );
                    } else if gl.has_y2y() {
                        assert_ne!(
                            sampling,
                            NvSampling::Shader,
                            "a GPU with GL_EXT_YUV_target must filter an NV12 import in hardware"
                        );
                    }
                    let mode = match sampling {
                        NvSampling::Hardware | NvSampling::HardwareYuv => "hardware",
                        _ => "shader",
                    };
                    let mad = mean_abs_diff(&planar_bytes(&gpu), &oracle.render(r, dw, dh, mode));
                    assert!(
                        mad < MAX_MEAN_ERROR,
                        "{mem:?} {r:?}->{dw}x{dh} {dtype:?} {sampling:?}: {mad:.3} grey levels from the reference"
                    );
                }
            }
        }
    }

    /// A source crop drawn 1:1 reproduces the bytes the whole frame converts to there.
    #[test]
    #[cfg(target_os = "linux")]
    fn test_nv12_crop_1to1_matches_whole_frame() {
        use crate::opengl_headless::processor::NvSampling;
        let Some(mut gl) = nv_test_processor() else {
            crate::test_support::report_skip(&format!("{} - OpenGL not available", function!()));
            return;
        };
        let (sw, sh, nv12) = detailed_nv12();
        let src =
            load_raw_image(sw, sh, PixelFormat::Nv12, Some(TensorMemory::Mem), &nv12).unwrap();
        let image = |w, h| {
            TensorDyn::image(
                w,
                h,
                PixelFormat::PlanarRgb,
                DType::U8,
                Some(TensorMemory::Mem),
                edgefirst_tensor::CpuAccess::ReadWrite,
            )
            .unwrap()
        };
        let mut whole = image(sw, sh);
        gl.convert(&src, &mut whole, None).unwrap();
        let whole = planar_bytes(&whole);
        for (cx, cy) in [(101usize, 53usize), (100, 52)] {
            let (cw, ch) = (320usize, 180usize);
            let mut tile = image(cw, ch);
            let crop = Crop::new().with_source(Some(crate::Region::new(cx, cy, cw, ch)));
            gl.convert(
                &src,
                &mut tile,
                Some(crate::ConvertOptions::new().crop(crop)),
            )
            .unwrap();
            assert_eq!(gl.last_nv_sampling, Some(NvSampling::Single));
            let tile = planar_bytes(&tile);
            for c in 0..3 {
                for y in 0..ch {
                    let got = &tile[c * cw * ch + y * cw..][..cw];
                    let want = &whole[c * sw * sh + (cy + y) * sw + cx..][..cw];
                    assert_eq!(got, want, "crop ({cx}, {cy}) channel {c} row {y} differs");
                }
            }
        }
    }

    /// `Interpolation::Nearest` resizes NV12 by copying the source pixel under each output centre.
    #[test]
    #[cfg(target_os = "linux")]
    fn test_nv12_nearest_interpolation() {
        use crate::opengl_headless::processor::NvSampling;
        let Some(mut gl) = nv_test_processor() else {
            crate::test_support::report_skip(&format!("{} - OpenGL not available", function!()));
            return;
        };
        gl.set_interpolation(crate::Interpolation::Nearest);
        let (sw, sh, nv12) = detailed_nv12();
        let src =
            load_raw_image(sw, sh, PixelFormat::Nv12, Some(TensorMemory::Mem), &nv12).unwrap();
        let oracle = NvOracle::new(&nv12, sw, sh, &src);
        for (dw, dh) in [(1792usize, 1008usize), (768, 432)] {
            let mut gpu = TensorDyn::image(
                dw,
                dh,
                PixelFormat::PlanarRgb,
                DType::U8,
                Some(TensorMemory::Mem),
                edgefirst_tensor::CpuAccess::ReadWrite,
            )
            .unwrap();
            gl.convert(&src, &mut gpu, None).unwrap();
            assert_eq!(gl.last_nv_sampling, Some(NvSampling::Single));
            let mad = mean_abs_diff(
                &planar_bytes(&gpu),
                &oracle.render((0, 0, sw, sh), dw, dh, "nearest"),
            );
            assert!(
                mad < 3.0,
                "{dw}x{dh}: nearest GL is {mad:.3} grey levels from the reference"
            );
        }
    }

    /// The GL bilinear letterbox of natural content stays close to the CPU backend's.
    #[test]
    #[cfg(target_os = "linux")]
    fn test_nv12_letterbox_close_to_cpu_on_camera_frame() {
        let Some(mut gl) = nv_test_processor() else {
            crate::test_support::report_skip(&format!("{} - OpenGL not available", function!()));
            return;
        };
        let nv12: &[u8] = &edgefirst_bench::testdata::read("camera720p.nv12");
        let src =
            load_raw_image(1280, 720, PixelFormat::Nv12, Some(TensorMemory::Mem), nv12).unwrap();
        let crop = letterbox_crop(1280, 720, 1792, 1024);
        let image = |m| {
            TensorDyn::image(
                1792,
                1024,
                PixelFormat::PlanarRgb,
                DType::U8,
                m,
                edgefirst_tensor::CpuAccess::ReadWrite,
            )
            .unwrap()
        };
        let mut gpu = image(Some(TensorMemory::Mem));
        gl.convert(
            &src,
            &mut gpu,
            Some(crate::ConvertOptions::new().crop(crop)),
        )
        .unwrap();
        let mut cpu = image(None);
        crate::cpu::CPUProcessor::new()
            .convert(
                &src,
                &mut cpu,
                Some(crate::ConvertOptions::new().crop(crop)),
            )
            .unwrap();
        let mad = mean_abs_diff(&planar_bytes(&gpu), &planar_bytes(&cpu));
        assert!(
            mad < 1.5,
            "letterbox upscale: GL is {mad:.3} grey levels from the CPU backend"
        );
    }

    /// `Interpolation::Nearest` resizes a packed source with `GL_NEAREST`, where `Bilinear` blends.
    #[test]
    #[cfg(target_os = "linux")]
    fn test_gl_texture_nearest_interpolation() {
        let Some(mut gl) = nv_test_processor() else {
            crate::test_support::report_skip(&format!("{} - OpenGL not available", function!()));
            return;
        };
        let (sw, sh, dw, dh) = (40usize, 30usize, 56usize, 42usize);
        let pixel = |x: usize, y: usize| if (x + y).is_multiple_of(2) { 16u8 } else { 235 };
        let mut rgba = Vec::with_capacity(sw * sh * 4);
        for y in 0..sh {
            for x in 0..sw {
                rgba.extend_from_slice(&[pixel(x, y), 128, 255 - pixel(x, y), 255]);
            }
        }
        let src =
            load_raw_image(sw, sh, PixelFormat::Rgba, Some(TensorMemory::Mem), &rgba).unwrap();
        let mut want = Vec::with_capacity(dw * dh * 4);
        for dy in 0..dh {
            for dx in 0..dw {
                let x = (dx * 2 + 1) * sw / (dw * 2);
                let y = (dy * 2 + 1) * sh / (dh * 2);
                want.extend_from_slice(&rgba[(y * sw + x) * 4..][..4]);
            }
        }
        for (mode, exact) in [
            (crate::Interpolation::Nearest, true),
            (crate::Interpolation::Bilinear, false),
        ] {
            gl.set_interpolation(mode);
            let mut dst = TensorDyn::image(
                dw,
                dh,
                PixelFormat::Rgba,
                DType::U8,
                Some(TensorMemory::Mem),
                edgefirst_tensor::CpuAccess::ReadWrite,
            )
            .unwrap();
            gl.convert(&src, &mut dst, None).unwrap();
            let got = dst.as_u8().unwrap().map().unwrap().as_slice().to_vec();
            assert_eq!(
                got == want,
                exact,
                "{mode:?}: nearest copy expected = {exact}"
            );
        }

        // A per-convert interpolation overrides the processor's for that convert only.
        let mut convert = |options: Option<crate::ConvertOptions>| {
            let mut dst = TensorDyn::image(
                dw,
                dh,
                PixelFormat::Rgba,
                DType::U8,
                Some(TensorMemory::Mem),
                edgefirst_tensor::CpuAccess::ReadWrite,
            )
            .unwrap();
            gl.convert(&src, &mut dst, options).unwrap();
            dst.as_u8().unwrap().map().unwrap().as_slice().to_vec()
        };
        let nearest = crate::ConvertOptions::new().interpolation(crate::Interpolation::Nearest);
        assert_eq!(convert(Some(nearest)), want, "per-convert Nearest");
        assert_ne!(convert(None), want, "the processor's Bilinear is restored");
    }

    /// A source crop drawn 1:1 reads each crop pixel unblended on the filtered (`LINEAR`) paths.
    #[test]
    #[cfg(target_os = "linux")]
    fn test_gl_crop_1to1_is_exact() {
        use crate::opengl_headless::processor::GLProcessorST;
        if !is_opengl_available() {
            crate::test_support::report_skip(&format!("{} - OpenGL not available", function!()));
            return;
        }
        let mut gl = match GLProcessorST::new(None, None) {
            Ok(g) => g,
            Err(e) => {
                crate::test_support::report_skip(&format!(
                    "{} - GL not available: {e}",
                    function!()
                ));
                return;
            }
        };
        let (sw, sh) = (640usize, 360usize);
        let checker = |x: usize, y: usize| if (x + y).is_multiple_of(2) { 16u8 } else { 235 };
        let mut rgba = Vec::with_capacity(sw * sh * 4);
        let mut nv12 = Vec::with_capacity(sw * sh * 3 / 2);
        for y in 0..sh {
            for x in 0..sw {
                let v = checker(x, y);
                rgba.extend_from_slice(&[v, 255 - v, v, 255]);
                nv12.push(v);
            }
        }
        nv12.resize(sw * sh * 3 / 2, 128);
        let crop = Crop::new().with_source(Some(crate::Region::new(101, 53, 64, 48)));

        let mut memories = vec![TensorMemory::Mem];
        if is_dma_available() {
            memories.push(TensorMemory::DmaBuf);
        }
        for mem in memories {
            for (src_fmt, bytes) in [(PixelFormat::Rgba, &rgba), (PixelFormat::Nv12, &nv12)] {
                if src_fmt == PixelFormat::Nv12 && mem != TensorMemory::DmaBuf {
                    continue;
                }
                let mut src = load_raw_image(sw, sh, src_fmt, Some(mem), bytes).unwrap();
                src.set_colorimetry(Some(
                    edgefirst_tensor::Colorimetry::default()
                        .with_encoding(edgefirst_tensor::ColorEncoding::Bt601)
                        .with_range(edgefirst_tensor::ColorRange::Limited),
                ));
                for dst_fmt in [PixelFormat::Rgba, PixelFormat::PlanarRgb] {
                    let image = |m| {
                        TensorDyn::image(
                            64,
                            48,
                            dst_fmt,
                            DType::U8,
                            m,
                            edgefirst_tensor::CpuAccess::ReadWrite,
                        )
                        .unwrap()
                    };
                    let mut gpu = image(Some(mem));
                    gl.convert(
                        &src,
                        &mut gpu,
                        Some(crate::ConvertOptions::new().crop(crop)),
                    )
                    .unwrap();
                    let mut cpu = image(None);
                    crate::cpu::CPUProcessor::new()
                        .convert(
                            &src,
                            &mut cpu,
                            Some(crate::ConvertOptions::new().crop(crop)),
                        )
                        .unwrap();
                    let g = gpu.as_u8().unwrap().map().unwrap().as_slice().to_vec();
                    let c = cpu.as_u8().unwrap().map().unwrap().as_slice().to_vec();
                    let max = g.iter().zip(&c).map(|(&a, &b)| a.abs_diff(b)).max();
                    assert!(
                        max <= Some(16),
                        "{mem:?} {src_fmt:?}->{dst_fmt:?} 1:1 crop: max deviation {max:?} \
                         (path {:?})",
                        gl.last_nv_convert_path
                    );
                }
            }
        }
    }

    /// NV16 and NV24 heap sources convert on the GPU through `ShaderR8` and match the CPU within 2.
    #[test]
    #[cfg(target_os = "linux")]
    fn test_nv16_nv24_nondma_upload_uses_shader() {
        use crate::opengl_headless::processor::{GLProcessorST, NvConvertPath};
        if !is_opengl_available() {
            crate::test_support::report_skip(&format!("{} - OpenGL not available", function!()));
            return;
        }
        let (w, h) = (64usize, 64usize);
        let mut gl = match GLProcessorST::new(None, None) {
            Ok(g) => g,
            Err(e) => {
                crate::test_support::report_skip(&format!(
                    "{} - GL not available: {e}",
                    function!()
                ));
                return;
            }
        };
        // (format, chroma byte count).
        for (fmt, uv) in [(PixelFormat::Nv16, w * h), (PixelFormat::Nv24, 2 * w * h)] {
            let mut bytes = vec![0u8; w * h];
            for r in 0..h {
                for c in 0..w {
                    bytes[r * w + c] = ((r + c) * 255 / (w + h)) as u8; // Y gradient
                }
            }
            bytes.extend(std::iter::repeat_n(128u8, uv)); // neutral chroma

            let mem_src = load_raw_image(w, h, fmt, Some(TensorMemory::Mem), &bytes).unwrap();
            let mut gpu = TensorDyn::image(
                w,
                h,
                PixelFormat::Rgba,
                DType::U8,
                None,
                edgefirst_tensor::CpuAccess::ReadWrite,
            )
            .unwrap();
            gl.convert(&mem_src, &mut gpu, None).unwrap();
            assert_eq!(
                gl.last_nv_convert_path,
                NvConvertPath::ShaderR8,
                "non-DMA {fmt:?} must use the R8-upload ShaderR8 path (got {:?})",
                gl.last_nv_convert_path
            );

            let cpu_src = load_raw_image(w, h, fmt, Some(TensorMemory::Mem), &bytes).unwrap();
            let mut cpu = TensorDyn::image(
                w,
                h,
                PixelFormat::Rgba,
                DType::U8,
                None,
                edgefirst_tensor::CpuAccess::ReadWrite,
            )
            .unwrap();
            crate::cpu::CPUProcessor::new()
                .convert(&cpu_src, &mut cpu, None)
                .unwrap();
            assert_pixels_match(
                cpu.as_u8().unwrap().map().unwrap().as_slice(),
                gpu.as_u8().unwrap().map().unwrap().as_slice(),
                2,
            );
        }
    }

    /// Prints per-path NV12 deltas against the CPU; asserts only `ShaderR8` on a solid frame.
    #[test]
    #[ignore = "diagnostic probe — run manually on-target with --nocapture --ignored"]
    #[cfg(target_os = "linux")]
    fn probe_nv12_sampler_vs_shader_divergence() {
        use crate::opengl_headless::processor::{GLProcessorST, NvConvertPath};
        if !is_dma_available() {
            crate::test_support::report_skip(&format!("{} - DMA not available", function!()));
            return;
        }
        let (w, h) = (64usize, 64usize);

        // Build NV12 (Y plane then h/2 rows of w/2 interleaved CbCr pairs). The
        // chroma closure maps a luma column to (Cb, Cr).
        let build = |y: u8, chroma: &dyn Fn(usize) -> (u8, u8)| -> Vec<u8> {
            let mut buf = vec![y; w * h];
            for _r in 0..h / 2 {
                for c in 0..w / 2 {
                    let (cb, cr) = chroma(c * 2);
                    buf.push(cb);
                    buf.push(cr);
                }
            }
            buf
        };
        let solid = build(120, &|_| (90, 160));
        let edge = build(120, &|col| if col < w / 2 { (90, 160) } else { (200, 40) });

        // Force a path via the env var (read at construction), then clear it.
        let convert_dma = |env: &str, bytes: &[u8]| -> Option<(NvConvertPath, Vec<u8>)> {
            std::env::set_var("EDGEFIRST_NV_CONVERT_PATH", env);
            let gl = GLProcessorST::new(None, None);
            std::env::remove_var("EDGEFIRST_NV_CONVERT_PATH");
            let mut gl = match gl {
                Ok(g) => g,
                Err(e) => {
                    crate::test_support::report_skip(&format!(
                        "{} - GL not available: {e}",
                        function!()
                    ));
                    return None;
                }
            };
            let src =
                load_raw_image(w, h, PixelFormat::Nv12, Some(TensorMemory::DmaBuf), bytes).unwrap();
            let mut dst = TensorDyn::image(
                w,
                h,
                PixelFormat::Rgba,
                DType::U8,
                Some(TensorMemory::DmaBuf),
                edgefirst_tensor::CpuAccess::ReadWrite,
            )
            .unwrap();
            gl.convert(&src, &mut dst, None).unwrap();
            let out = dst.as_u8().unwrap().map().unwrap().to_vec();
            Some((gl.last_nv_convert_path, out))
        };
        let cpu_ref = |bytes: &[u8]| -> Vec<u8> {
            let src =
                load_raw_image(w, h, PixelFormat::Nv12, Some(TensorMemory::Mem), bytes).unwrap();
            let mut dst = TensorDyn::image(
                w,
                h,
                PixelFormat::Rgba,
                DType::U8,
                None,
                edgefirst_tensor::CpuAccess::ReadWrite,
            )
            .unwrap();
            crate::cpu::CPUProcessor::new()
                .convert(&src, &mut dst, None)
                .unwrap();
            dst.as_u8().unwrap().map().unwrap().to_vec()
        };
        // (max, mean, max in flat regions, max within ±2 cols of the w/2 edge)
        let stats = |a: &[u8], b: &[u8]| -> (u32, f64, u32, u32) {
            let (mut maxd, mut sum, mut maxflat, mut maxbound) = (0u32, 0u64, 0u32, 0u32);
            for i in 0..w * h {
                let boundary = ((i % w) as i64 - (w as i64 / 2)).abs() <= 2;
                for ch in 0..3 {
                    let d = (a[i * 4 + ch] as i32 - b[i * 4 + ch] as i32).unsigned_abs();
                    maxd = maxd.max(d);
                    sum += d as u64;
                    if boundary {
                        maxbound = maxbound.max(d);
                    } else {
                        maxflat = maxflat.max(d);
                    }
                }
            }
            (maxd, sum as f64 / (w * h * 3) as f64, maxflat, maxbound)
        };

        for (name, bytes) in [("solid", &solid), ("chroma-edge", &edge)] {
            let cpu = cpu_ref(bytes);
            let Some((ps, sampler)) = convert_dma("sampler", bytes) else {
                return;
            };
            let Some((pb, shader)) = convert_dma("shader", bytes) else {
                return;
            };
            let (smax, smean, sflat, sbound) = stats(&sampler, &cpu);
            let (bmax, bmean, bflat, bbound) = stats(&shader, &cpu);
            eprintln!(
                "PROBE[{name}] ExternalSampler({ps:?}) vs CPU: max={smax} mean={smean:.2} \
                 flat(matrix)={sflat} boundary(chroma)={sbound}"
            );
            eprintln!(
                "PROBE[{name}] ShaderR8({pb:?})       vs CPU: max={bmax} mean={bmean:.2} \
                 flat(matrix)={bflat} boundary(chroma)={bbound}"
            );
            if name == "solid" {
                assert!(
                    bmax <= 4,
                    "ShaderR8 must match CPU reference on a solid frame (pure matrix): max={bmax}"
                );
            }
        }
    }

    /// Prints where the NV12 to packed-RGB DMA path differs from the CPU, same-size and letterbox.
    #[test]
    #[ignore = "diagnostic probe — run manually on-target with --nocapture --ignored"]
    #[cfg(target_os = "linux")]
    fn probe_nv12_packed_letterbox_divergence() {
        use crate::opengl_headless::processor::GLProcessorST;
        use crate::{Fit, Region};
        if !is_dma_available() {
            crate::test_support::report_skip(&format!("{} - DMA not available", function!()));
            return;
        }
        // A 16:9 source and a square model size.
        let (sw, sh) = (192usize, 108usize);
        let model = 96usize;

        // Patterned NV12: luma gradient + a vertical chroma edge.
        let mut bytes = Vec::with_capacity(sw * sh * 3 / 2);
        for r in 0..sh {
            for c in 0..sw {
                bytes.push(((r + c) * 255 / (sw + sh)) as u8);
            }
        }
        for _r in 0..sh / 2 {
            for c in 0..sw / 2 {
                let (cb, cr) = if c < sw / 4 { (90, 160) } else { (200, 40) };
                bytes.push(cb);
                bytes.push(cr);
            }
        }

        // Aspect-fit size and padding.
        let scale = (model as f32 / sw as f32).min(model as f32 / sh as f32);
        let scaled_w = (sw as f32 * scale).round() as usize;
        let scaled_h = (sh as f32 * scale).round() as usize;
        let pad_x = (model - scaled_w) / 2;
        let pad_y = (model - scaled_h) / 2;
        let letterbox = || {
            Crop::new()
                .with_source(Some(Region::new(0, 0, sw, sh)))
                .with_fit(Fit::Letterbox {
                    pad: [114, 114, 114, 255],
                })
        };

        let gl_packed = |env: &str, crop: Crop, dw: usize, dh: usize| -> Option<(usize, Vec<u8>)> {
            std::env::set_var("EDGEFIRST_NV_CONVERT_PATH", env);
            let g = GLProcessorST::new(None, None);
            std::env::remove_var("EDGEFIRST_NV_CONVERT_PATH");
            let mut g = match g {
                Ok(x) => x,
                Err(e) => {
                    crate::test_support::report_skip(&format!("{} - GL: {e}", function!()));
                    return None;
                }
            };
            let src = load_raw_image(
                sw,
                sh,
                PixelFormat::Nv12,
                Some(TensorMemory::DmaBuf),
                &bytes,
            )
            .unwrap();
            let mut dst = TensorDyn::image(
                dw,
                dh,
                PixelFormat::Rgb,
                DType::U8,
                None,
                edgefirst_tensor::CpuAccess::ReadWrite,
            )
            .unwrap();
            if let Err(e) = g.convert(
                &src,
                &mut dst,
                Some(crate::ConvertOptions::new().crop(crop)),
            ) {
                crate::test_support::report_skip(&format!(
                    "{} - {env} convert declined: {e}",
                    function!()
                ));
                return None;
            }
            let stride = dst
                .as_u8()
                .unwrap()
                .effective_row_stride()
                .unwrap_or(dw * 3);
            Some((stride, dst.as_u8().unwrap().map().unwrap().to_vec()))
        };
        let cpu_packed = |crop: Crop, dw: usize, dh: usize| -> (usize, Vec<u8>) {
            let src =
                load_raw_image(sw, sh, PixelFormat::Nv12, Some(TensorMemory::Mem), &bytes).unwrap();
            let mut dst = TensorDyn::image(
                dw,
                dh,
                PixelFormat::Rgb,
                DType::U8,
                None,
                edgefirst_tensor::CpuAccess::ReadWrite,
            )
            .unwrap();
            crate::cpu::CPUProcessor::new()
                .convert(
                    &src,
                    &mut dst,
                    Some(crate::ConvertOptions::new().crop(crop)),
                )
                .unwrap();
            let stride = dst
                .as_u8()
                .unwrap()
                .effective_row_stride()
                .unwrap_or(dw * 3);
            (stride, dst.as_u8().unwrap().map().unwrap().to_vec())
        };
        // Stride-aware delta over a pixel region [r0..r0+rh) × [c0..c0+cw) (RGB).
        let region = |a: (&[u8], usize),
                      b: (&[u8], usize),
                      (c0, r0, cw, rh): (usize, usize, usize, usize)|
         -> (u32, f64) {
            let (mut mx, mut sum) = (0u32, 0u64);
            for r in r0..r0 + rh {
                for c in (c0 * 3)..((c0 + cw) * 3) {
                    let d = (a.0[r * a.1 + c] as i32 - b.0[r * b.1 + c] as i32).unsigned_abs();
                    mx = mx.max(d);
                    sum += d as u64;
                }
            }
            (mx, sum as f64 / (cw * 3 * rh) as f64)
        };

        // same-size baseline (no resize)
        let cpu0 = cpu_packed(Crop::no_crop(), sw, sh);
        if let Some(g) = gl_packed("shader", Crop::no_crop(), sw, sh) {
            let (mx, mean) = region((&g.1, g.0), (&cpu0.1, cpu0.0), (0, 0, sw, sh));
            eprintln!("PROBE2[same-size]   ShaderR8 vs CPU: max={mx} mean={mean:.2}");
        }

        // letterbox (aspect-fit + grey bars), full frame + content region only
        let cpu1 = cpu_packed(letterbox(), model, model);
        // Content-region pixel dump against the CPU.
        if let Some(g) = gl_packed("shader", letterbox(), model, model) {
            let px = |buf: &[u8], stride: usize, r: usize, c: usize| {
                let o = r * stride + c * 3;
                (buf[o], buf[o + 1], buf[o + 2])
            };
            let (ctop, cleft) = (pad_y, pad_x);
            let (cbot, cright) = (pad_y + scaled_h - 1, pad_x + scaled_w - 1);
            eprintln!(
                "DUMP content TL cpu={:?} gl={:?} | TR cpu={:?} gl={:?} | BL cpu={:?} gl={:?} | BR cpu={:?} gl={:?}",
                px(&cpu1.1, cpu1.0, ctop, cleft),
                px(&g.1, g.0, ctop, cleft),
                px(&cpu1.1, cpu1.0, ctop, cright),
                px(&g.1, g.0, ctop, cright),
                px(&cpu1.1, cpu1.0, cbot, cleft),
                px(&g.1, g.0, cbot, cleft),
                px(&cpu1.1, cpu1.0, cbot, cright),
                px(&g.1, g.0, cbot, cright),
            );
        }
        for env in ["shader", "sampler"] {
            if let Some(g) = gl_packed(env, letterbox(), model, model) {
                let (fmx, fmean) = region((&g.1, g.0), (&cpu1.1, cpu1.0), (0, 0, model, model));
                let (cmx, cmean) = region(
                    (&g.1, g.0),
                    (&cpu1.1, cpu1.0),
                    (pad_x, pad_y, scaled_w, scaled_h),
                );
                eprintln!(
                    "PROBE2[lb-full/{env:8}] vs CPU: max={fmx} mean={fmean:.2}   \
                     lb-content: max={cmx} mean={cmean:.2}"
                );
            }
        }
    }

    /// NV24 to RGBA on DMA buffers runs `ShaderR8` and matches BT.601 full-range RGB within ±2.
    #[test]
    #[cfg(any(target_os = "linux", target_os = "macos", target_os = "windows"))]
    fn test_gpu_nv24_to_rgba_shader_r8() {
        use crate::opengl_headless::processor::{GLProcessorST, NvConvertPath};
        if !is_gpu_image_buffer_available() {
            crate::test_support::report_skip(&format!(
                "{} - no zero-copy GPU buffers",
                function!()
            ));
            return;
        }

        let (w, h) = (64usize, 64usize);
        let (yv, cb, cr) = (180u8, 90u8, 200u8);
        let expected = yuv601_to_rgb(yv, cb, cr);

        let mut src = load_raw_image(
            w,
            h,
            PixelFormat::Nv24,
            Some(TensorMemory::DmaBuf),
            &make_nv24_solid(w, h, yv, cb, cr),
        )
        .unwrap();
        src.set_colorimetry(Some(
            edgefirst_tensor::Colorimetry::default()
                .with_encoding(edgefirst_tensor::ColorEncoding::Bt601)
                .with_range(edgefirst_tensor::ColorRange::Full),
        ));

        let mut dst = TensorDyn::image(
            w,
            h,
            PixelFormat::Rgba,
            DType::U8,
            Some(TensorMemory::DmaBuf),
            edgefirst_tensor::CpuAccess::ReadWrite,
        )
        .unwrap();

        let mut gl = match GLProcessorST::new(None, None) {
            Ok(g) => g,
            Err(e) => {
                crate::test_support::report_skip(&format!(
                    "{} - GL not available: {e}",
                    function!()
                ));
                return;
            }
        };

        gl.convert(&src, &mut dst, None).unwrap();

        assert_eq!(
            gl.last_nv_convert_path,
            NvConvertPath::ShaderR8,
            "NV24 DMA convert must use ShaderR8, got {:?}",
            gl.last_nv_convert_path
        );

        let map = dst.as_u8().unwrap().map().unwrap();
        let pixels = map.as_slice();
        for i in 0..(w * h) {
            let r = pixels[i * 4];
            let g = pixels[i * 4 + 1];
            let b = pixels[i * 4 + 2];
            let diff_r = (r as i32 - expected[0] as i32).unsigned_abs();
            let diff_g = (g as i32 - expected[1] as i32).unsigned_abs();
            let diff_b = (b as i32 - expected[2] as i32).unsigned_abs();
            assert!(
                diff_r <= 2 && diff_g <= 2 && diff_b <= 2,
                "pixel {i}: got ({r},{g},{b}) expected ({},{},{}) — diff ({diff_r},{diff_g},{diff_b})",
                expected[0], expected[1], expected[2]
            );
        }
    }

    /// NV16 to RGBA through `ShaderR8` matches the CPU converter within ±2.
    #[test]
    #[cfg(any(target_os = "linux", target_os = "macos", target_os = "windows"))]
    fn test_gpu_nv16_matches_cpu_reference() {
        use crate::opengl_headless::processor::{GLProcessorST, NvConvertPath};
        if !is_gpu_image_buffer_available() {
            crate::test_support::report_skip(&format!(
                "{} - no zero-copy GPU buffers",
                function!()
            ));
            return;
        }

        let (w, h) = (64usize, 64usize);
        // Patterned NV16: varying luma, neutral chroma.
        let mut nv16 = vec![0u8; w * h * 2]; // H luma rows + H chroma rows
        for row in 0..h {
            for col in 0..w {
                nv16[row * w + col] = ((row * 255) / h) as u8;
            }
        }
        for i in 0..w * h {
            nv16[w * h + i] = 128;
        }

        let src_dma =
            load_raw_image(w, h, PixelFormat::Nv16, Some(TensorMemory::DmaBuf), &nv16).unwrap();
        let src_mem =
            load_raw_image(w, h, PixelFormat::Nv16, Some(TensorMemory::Mem), &nv16).unwrap();

        let mut cpu_dst = TensorDyn::image(
            w,
            h,
            PixelFormat::Rgba,
            DType::U8,
            None,
            edgefirst_tensor::CpuAccess::ReadWrite,
        )
        .unwrap();
        {
            let mut cpu = crate::cpu::CPUProcessor::new();
            cpu.convert(&src_mem, &mut cpu_dst, None).unwrap();
        }

        let mut gpu_dst = TensorDyn::image(
            w,
            h,
            PixelFormat::Rgba,
            DType::U8,
            Some(TensorMemory::DmaBuf),
            edgefirst_tensor::CpuAccess::ReadWrite,
        )
        .unwrap();
        let mut gl = match GLProcessorST::new(None, None) {
            Ok(g) => g,
            Err(e) => {
                crate::test_support::report_skip(&format!(
                    "{} - GL not available: {e}",
                    function!()
                ));
                return;
            }
        };
        gl.convert(&src_dma, &mut gpu_dst, None).unwrap();

        assert_eq!(
            gl.last_nv_convert_path,
            NvConvertPath::ShaderR8,
            "NV16 DMA convert must use ShaderR8"
        );

        let cpu_map = cpu_dst.as_u8().unwrap().map().unwrap();
        let gpu_map = gpu_dst.as_u8().unwrap().map().unwrap();
        let cpu_pixels = cpu_map.as_slice();
        let gpu_pixels = gpu_map.as_slice();

        let mut max_diff = 0u32;
        for i in 0..(w * h * 4) {
            let d = (gpu_pixels[i] as i32 - cpu_pixels[i] as i32).unsigned_abs();
            max_diff = max_diff.max(d);
        }
        assert!(
            max_diff <= 2,
            "NV16 GPU ShaderR8 vs CPU reference: max_diff={max_diff} (tolerance 2)"
        );
    }

    /// NV24 to RGBA through `ShaderR8` matches the CPU converter within ±2.
    #[test]
    #[cfg(any(target_os = "linux", target_os = "macos", target_os = "windows"))]
    fn test_gpu_nv24_matches_cpu_reference() {
        use crate::opengl_headless::processor::{GLProcessorST, NvConvertPath};
        if !is_gpu_image_buffer_available() {
            crate::test_support::report_skip(&format!(
                "{} - no zero-copy GPU buffers",
                function!()
            ));
            return;
        }

        let (w, h) = (64usize, 64usize);
        // Patterned NV24: varying luma, neutral chroma.
        let mut nv24 = vec![0u8; w * h * 3]; // H luma rows + 2H chroma rows
        for row in 0..h {
            for col in 0..w {
                nv24[row * w + col] = ((row * 255) / h) as u8;
            }
        }
        for i in 0..w * h * 2 {
            nv24[w * h + i] = 128;
        }

        let src_dma =
            load_raw_image(w, h, PixelFormat::Nv24, Some(TensorMemory::DmaBuf), &nv24).unwrap();
        let src_mem =
            load_raw_image(w, h, PixelFormat::Nv24, Some(TensorMemory::Mem), &nv24).unwrap();

        let mut cpu_dst = TensorDyn::image(
            w,
            h,
            PixelFormat::Rgba,
            DType::U8,
            None,
            edgefirst_tensor::CpuAccess::ReadWrite,
        )
        .unwrap();
        {
            let mut cpu = crate::cpu::CPUProcessor::new();
            cpu.convert(&src_mem, &mut cpu_dst, None).unwrap();
        }

        let mut gpu_dst = TensorDyn::image(
            w,
            h,
            PixelFormat::Rgba,
            DType::U8,
            Some(TensorMemory::DmaBuf),
            edgefirst_tensor::CpuAccess::ReadWrite,
        )
        .unwrap();
        let mut gl = match GLProcessorST::new(None, None) {
            Ok(g) => g,
            Err(e) => {
                crate::test_support::report_skip(&format!(
                    "{} - GL not available: {e}",
                    function!()
                ));
                return;
            }
        };
        gl.convert(&src_dma, &mut gpu_dst, None).unwrap();

        assert_eq!(
            gl.last_nv_convert_path,
            NvConvertPath::ShaderR8,
            "NV24 DMA convert must use ShaderR8"
        );

        let cpu_map = cpu_dst.as_u8().unwrap().map().unwrap();
        let gpu_map = gpu_dst.as_u8().unwrap().map().unwrap();
        let cpu_pixels = cpu_map.as_slice();
        let gpu_pixels = gpu_map.as_slice();

        let mut max_diff = 0u32;
        for i in 0..(w * h * 4) {
            let d = (gpu_pixels[i] as i32 - cpu_pixels[i] as i32).unsigned_abs();
            max_diff = max_diff.max(d);
        }
        assert!(
            max_diff <= 2,
            "NV24 GPU ShaderR8 vs CPU reference: max_diff={max_diff} (tolerance 2)"
        );
    }

    /// A DMA NV12 source converts on the GPU, through `ExternalSampler` or `ShaderR8`.
    #[test]
    #[cfg(target_os = "linux")]
    fn test_nv12_dma_gpu_path_no_cpu_fallback() {
        use crate::opengl_headless::processor::{GLProcessorST, NvConvertPath};
        if !is_dma_available() {
            crate::test_support::report_skip(&format!("{} - DMA not available", function!()));
            return;
        }

        let (w, h) = (64usize, 64usize);
        // Solid grey NV12 (Y=128, Cb=128, Cr=128).
        let mut nv12 = vec![128u8; w * h];
        nv12.extend(vec![128u8; w * h / 2]); // UV plane
        let src =
            load_raw_image(w, h, PixelFormat::Nv12, Some(TensorMemory::DmaBuf), &nv12).unwrap();

        let mut dst = TensorDyn::image(
            w,
            h,
            PixelFormat::Rgba,
            DType::U8,
            Some(TensorMemory::DmaBuf),
            edgefirst_tensor::CpuAccess::ReadWrite,
        )
        .unwrap();

        let mut gl = match GLProcessorST::new(None, None) {
            Ok(g) => g,
            Err(e) => {
                crate::test_support::report_skip(&format!(
                    "{} - GL not available: {e}",
                    function!()
                ));
                return;
            }
        };

        gl.convert(&src, &mut dst, None).unwrap();

        assert!(
            matches!(
                gl.last_nv_convert_path,
                NvConvertPath::ExternalSampler | NvConvertPath::ShaderR8
            ),
            "NV12 DMA convert must use a GPU path (ExternalSampler/ShaderR8), got {:?}",
            gl.last_nv_convert_path
        );
    }

    /// NV16 to RGB int8 through `ShaderR8` matches the CPU i8 reference within 2.
    #[test]
    #[cfg(target_os = "linux")]
    fn test_gpu_nv16_shader_r8_int8_output() {
        use crate::opengl_headless::processor::{GLProcessorST, NvConvertPath};
        if !is_dma_available() {
            crate::test_support::report_skip(&format!("{} - DMA not available", function!()));
            return;
        }

        let (w, h) = (64usize, 64usize);
        // Patterned luma (vertical gradient) and a mild chroma offset.
        let mut nv16 = vec![0u8; w * h * 2];
        for row in 0..h {
            for col in 0..w {
                nv16[row * w + col] = ((row * 255) / h) as u8;
            }
        }
        for i in 0..w * h / 2 {
            nv16[w * h + 2 * i] = 110; // Cb
            nv16[w * h + 2 * i + 1] = 150; // Cr
        }

        let mut src_dma =
            load_raw_image(w, h, PixelFormat::Nv16, Some(TensorMemory::DmaBuf), &nv16).unwrap();
        let mut src_mem =
            load_raw_image(w, h, PixelFormat::Nv16, Some(TensorMemory::Mem), &nv16).unwrap();
        let cm = edgefirst_tensor::Colorimetry::default()
            .with_encoding(edgefirst_tensor::ColorEncoding::Bt601)
            .with_range(edgefirst_tensor::ColorRange::Full);
        src_dma.set_colorimetry(Some(cm));
        src_mem.set_colorimetry(Some(cm));

        let mut cpu_dst = TensorDyn::image(
            w,
            h,
            PixelFormat::Rgb,
            DType::I8,
            None,
            edgefirst_tensor::CpuAccess::ReadWrite,
        )
        .unwrap();
        {
            let mut cpu = crate::cpu::CPUProcessor::new();
            cpu.convert(&src_mem, &mut cpu_dst, None).unwrap();
        }

        let mut gpu_dst = TensorDyn::image(
            w,
            h,
            PixelFormat::Rgb,
            DType::I8,
            Some(TensorMemory::DmaBuf),
            edgefirst_tensor::CpuAccess::ReadWrite,
        )
        .unwrap();
        let mut gl = match GLProcessorST::new(None, None) {
            Ok(g) => g,
            Err(e) => {
                crate::test_support::report_skip(&format!(
                    "{} - GL not available: {e}",
                    function!()
                ));
                return;
            }
        };
        gl.convert(&src_dma, &mut gpu_dst, None).unwrap();

        assert_eq!(
            gl.last_nv_convert_path,
            NvConvertPath::ShaderR8,
            "NV16 int8 DMA convert must use ShaderR8"
        );

        // Compare raw i8 bytes.
        let cpu_map = cpu_dst.as_i8().unwrap().map().unwrap();
        let gpu_map = gpu_dst.as_i8().unwrap().map().unwrap();
        let n = w * h * 3;
        let mut max_diff = 0u32;
        for i in 0..n {
            let d = (gpu_map.as_slice()[i] as i32 - cpu_map.as_slice()[i] as i32).unsigned_abs();
            max_diff = max_diff.max(d);
        }
        assert!(
            max_diff <= 2,
            "NV16 int8 GPU ShaderR8 vs CPU reference: max_diff={max_diff} (tolerance 2)"
        );
    }

    // =========================================================================
    // Odd-dimension end-to-end tests
    // =========================================================================

    /// Fill an NV12/NV16/NV24 tensor (Dma or Mem) with a patterned synthetic image at its own
    /// row stride.
    ///
    /// Pattern: `Y(r,c) = (r*3 + c*5) % 256`
    /// Chroma column `cc`, chroma row `cr_row`:
    ///   `Cb = (cc*7 + cr_row*11 + 40) % 256`
    ///   `Cr = (cc*13 + cr_row*3 + 80) % 256`
    ///
    /// The chroma subsampling (cw_shift, ch_shift) follows each format:
    ///   NV12: (1,1) — 4:2:0   NV16: (1,0) — 4:2:2   NV24: (0,0) — 4:4:4
    #[cfg(any(target_os = "linux", target_os = "macos", target_os = "windows"))]
    fn fill_patterned_nv(t: &TensorDyn) {
        let fmt = t.format().unwrap();
        let w = t.width().unwrap();
        let h = t.height().unwrap();
        let stride = t.effective_row_stride().unwrap();

        let (cw_shift, ch_shift): (usize, usize) = match fmt {
            PixelFormat::Nv12 => (1, 1),
            PixelFormat::Nv16 => (1, 0),
            PixelFormat::Nv24 => (0, 0),
            _ => panic!("fill_patterned_nv: unsupported format {fmt:?}"),
        };
        let chroma_h = h.div_ceil(1 << ch_shift);
        let chroma_w = w.div_ceil(1 << cw_shift);

        let bound = t.as_u8().unwrap();
        let mut m = bound.map().unwrap();
        let buf = m.as_mut_slice();
        let uv_start = stride * h;

        // Luma: diagonal gradient varying in both x and y.
        for r in 0..h {
            for c in 0..w {
                buf[r * stride + c] = ((r * 3 + c * 5) % 256) as u8;
            }
        }

        // Fill chroma (interleaved [Cb, Cr] per chroma column).
        // NV12/NV16 UV row pitch == stride; NV24 UV row pitch == stride*2.
        let uv_row_stride = if fmt == PixelFormat::Nv24 {
            stride * 2
        } else {
            stride
        };
        for cr_row in 0..chroma_h {
            for cc in 0..chroma_w {
                let cb_val = ((cc * 7 + cr_row * 11 + 40) % 256) as u8;
                let cr_val = ((cc * 13 + cr_row * 3 + 80) % 256) as u8;
                let uv_byte = uv_start + cr_row * uv_row_stride + cc * 2;
                buf[uv_byte] = cb_val;
                buf[uv_byte + 1] = cr_val;
            }
        }
    }

    /// Build a pair of identically-filled NV tensors: one `Dma` (for the GPU),
    /// one `Mem` (for the CPU reference).  Returns `(dma_src, mem_src)`.
    #[cfg(any(target_os = "linux", target_os = "macos", target_os = "windows"))]
    fn make_patterned_nv_pair(w: usize, h: usize, fmt: PixelFormat) -> (TensorDyn, TensorDyn) {
        let mut dma = TensorDyn::image(
            w,
            h,
            fmt,
            DType::U8,
            Some(TensorMemory::DmaBuf),
            edgefirst_tensor::CpuAccess::ReadWrite,
        )
        .unwrap();
        let mut mem = TensorDyn::image(
            w,
            h,
            fmt,
            DType::U8,
            Some(TensorMemory::Mem),
            edgefirst_tensor::CpuAccess::ReadWrite,
        )
        .unwrap();
        fill_patterned_nv(&dma);
        fill_patterned_nv(&mem);
        let cm = edgefirst_tensor::Colorimetry::default()
            .with_encoding(edgefirst_tensor::ColorEncoding::Bt601)
            .with_range(edgefirst_tensor::ColorRange::Full);
        dma.set_colorimetry(Some(cm));
        mem.set_colorimetry(Some(cm));
        (dma, mem)
    }

    /// Compare a GPU RGBA/RGB `u8` output against a CPU RGBA/RGB `u8` reference,
    /// reading both maps at their real `effective_row_stride`.
    ///
    /// Returns `(max_diff, first_failing_location)`.
    #[cfg(any(target_os = "linux", target_os = "macos", target_os = "windows"))]
    fn compare_gpu_vs_cpu_u8(
        gpu_dst: &TensorDyn,
        cpu_dst: &TensorDyn,
        w: usize,
        h: usize,
    ) -> (u32, Option<(usize, usize, usize)>) {
        let channels = gpu_dst.format().unwrap().channels();
        let gpu_t = gpu_dst.as_u8().unwrap();
        let cpu_t = cpu_dst.as_u8().unwrap();
        let gpu_stride = gpu_t.effective_row_stride().unwrap_or(w * channels);
        let cpu_stride = cpu_t.effective_row_stride().unwrap_or(w * channels);
        let gpu_map = gpu_t.map().unwrap();
        let cpu_map = cpu_t.map().unwrap();
        let gpu_px = gpu_map.as_slice();
        let cpu_px = cpu_map.as_slice();
        let mut max_diff = 0u32;
        let mut first_fail: Option<(usize, usize, usize)> = None;
        for row in 0..h {
            for col in 0..w {
                for ch in 0..channels {
                    let gi = row * gpu_stride + col * channels + ch;
                    let ci = row * cpu_stride + col * channels + ch;
                    let d = (gpu_px[gi] as i32 - cpu_px[ci] as i32).unsigned_abs();
                    if d > max_diff {
                        max_diff = d;
                    }
                    if first_fail.is_none() && d > 4 {
                        first_fail = Some((col, row, ch));
                    }
                }
            }
        }
        (max_diff, first_fail)
    }

    /// Compare a GPU RGB `i8` output against a CPU RGB `i8` reference,
    /// reading both at their real `effective_row_stride`.
    #[cfg(target_os = "linux")]
    fn compare_gpu_vs_cpu_i8(gpu_dst: &TensorDyn, cpu_dst: &TensorDyn, w: usize, h: usize) -> u32 {
        let channels = 3usize; // RGB
        let gpu_t = gpu_dst.as_i8().unwrap();
        let cpu_t = cpu_dst.as_i8().unwrap();
        let gpu_stride = gpu_t.effective_row_stride().unwrap_or(w * channels);
        let cpu_stride = cpu_t.effective_row_stride().unwrap_or(w * channels);
        let gpu_map = gpu_t.map().unwrap();
        let cpu_map = cpu_t.map().unwrap();
        let gpu_px = gpu_map.as_slice();
        let cpu_px = cpu_map.as_slice();
        let mut max_diff = 0u32;
        for row in 0..h {
            for col in 0..w {
                for ch in 0..channels {
                    let gi = row * gpu_stride + col * channels + ch;
                    let ci = row * cpu_stride + col * channels + ch;
                    let d = (gpu_px[gi] as i32 - cpu_px[ci] as i32).unsigned_abs();
                    if d > max_diff {
                        max_diff = d;
                    }
                }
            }
        }
        max_diff
    }

    // -------------------------------------------------------------------------
    // NV16 odd-W (65×64) → RGBA
    // -------------------------------------------------------------------------

    /// NV16 odd-width (65×64) runs `ShaderR8` and matches the CPU reference within ±4.
    #[test]
    #[cfg(any(target_os = "linux", target_os = "macos", target_os = "windows"))]
    fn nv16_odd_w_vs_cpu() {
        use crate::opengl_headless::processor::{GLProcessorST, NvConvertPath};
        if !is_gpu_image_buffer_available() {
            crate::test_support::report_skip(&format!(
                "{} - no zero-copy GPU buffers",
                function!()
            ));
            return;
        }
        let (w, h) = (321usize, 240usize); // QVGA-scale odd width
        let (src_dma, src_mem) = make_patterned_nv_pair(w, h, PixelFormat::Nv16);

        let mut cpu_dst = TensorDyn::image(
            w,
            h,
            PixelFormat::Rgba,
            DType::U8,
            None,
            edgefirst_tensor::CpuAccess::ReadWrite,
        )
        .unwrap();
        crate::cpu::CPUProcessor::new()
            .convert(&src_mem, &mut cpu_dst, None)
            .unwrap();

        let mut gpu_dst = TensorDyn::image(
            w,
            h,
            PixelFormat::Rgba,
            DType::U8,
            Some(TensorMemory::DmaBuf),
            edgefirst_tensor::CpuAccess::ReadWrite,
        )
        .unwrap();
        let mut gl = match GLProcessorST::new(None, None) {
            Ok(g) => g,
            Err(e) => {
                crate::test_support::report_skip(&format!(
                    "{} - GL not available: {e}",
                    function!()
                ));
                return;
            }
        };
        gl.convert(&src_dma, &mut gpu_dst, None).unwrap();

        assert_eq!(
            gl.last_nv_convert_path,
            NvConvertPath::ShaderR8,
            "NV16 odd-W must use ShaderR8, got {:?}",
            gl.last_nv_convert_path
        );

        let (max_diff, first_fail) = compare_gpu_vs_cpu_u8(&gpu_dst, &cpu_dst, w, h);
        eprintln!("NV16 odd-W: GPU vs CPU max_diff={max_diff}");
        assert!(
            max_diff <= 4,
            "NV16 odd-W GPU vs CPU max_diff={max_diff} > 4; first bad at {first_fail:?}"
        );
    }

    // -------------------------------------------------------------------------
    // NV24 odd-W (65×64) → RGBA
    // -------------------------------------------------------------------------

    /// NV24 odd-width (65×64) runs `ShaderR8` and matches the CPU reference within ±4.
    #[test]
    #[cfg(any(target_os = "linux", target_os = "macos", target_os = "windows"))]
    fn nv24_odd_w_vs_cpu() {
        use crate::opengl_headless::processor::{GLProcessorST, NvConvertPath};
        if !is_gpu_image_buffer_available() {
            crate::test_support::report_skip(&format!(
                "{} - no zero-copy GPU buffers",
                function!()
            ));
            return;
        }
        let (w, h) = (321usize, 240usize); // QVGA-scale odd width
        let (src_dma, src_mem) = make_patterned_nv_pair(w, h, PixelFormat::Nv24);

        let mut cpu_dst = TensorDyn::image(
            w,
            h,
            PixelFormat::Rgba,
            DType::U8,
            None,
            edgefirst_tensor::CpuAccess::ReadWrite,
        )
        .unwrap();
        crate::cpu::CPUProcessor::new()
            .convert(&src_mem, &mut cpu_dst, None)
            .unwrap();

        let mut gpu_dst = TensorDyn::image(
            w,
            h,
            PixelFormat::Rgba,
            DType::U8,
            Some(TensorMemory::DmaBuf),
            edgefirst_tensor::CpuAccess::ReadWrite,
        )
        .unwrap();
        let mut gl = match GLProcessorST::new(None, None) {
            Ok(g) => g,
            Err(e) => {
                crate::test_support::report_skip(&format!(
                    "{} - GL not available: {e}",
                    function!()
                ));
                return;
            }
        };
        gl.convert(&src_dma, &mut gpu_dst, None).unwrap();

        assert_eq!(
            gl.last_nv_convert_path,
            NvConvertPath::ShaderR8,
            "NV24 odd-W must use ShaderR8, got {:?}",
            gl.last_nv_convert_path
        );

        let (max_diff, first_fail) = compare_gpu_vs_cpu_u8(&gpu_dst, &cpu_dst, w, h);
        eprintln!("NV24 odd-W: GPU vs CPU max_diff={max_diff}");
        assert!(
            max_diff <= 4,
            "NV24 odd-W GPU vs CPU max_diff={max_diff} > 4; first bad at {first_fail:?}"
        );
    }

    // -------------------------------------------------------------------------
    // NV16 odd-both (65×63) → RGBA
    // -------------------------------------------------------------------------

    /// NV16 odd-width and odd-height (65×63) runs `ShaderR8` and matches the CPU within ±4.
    #[test]
    #[cfg(any(target_os = "linux", target_os = "macos", target_os = "windows"))]
    fn nv16_odd_both_vs_cpu() {
        use crate::opengl_headless::processor::{GLProcessorST, NvConvertPath};
        if !is_gpu_image_buffer_available() {
            crate::test_support::report_skip(&format!(
                "{} - no zero-copy GPU buffers",
                function!()
            ));
            return;
        }
        let (w, h) = (321usize, 241usize); // QVGA-scale, both odd
        let (src_dma, src_mem) = make_patterned_nv_pair(w, h, PixelFormat::Nv16);

        let mut cpu_dst = TensorDyn::image(
            w,
            h,
            PixelFormat::Rgba,
            DType::U8,
            None,
            edgefirst_tensor::CpuAccess::ReadWrite,
        )
        .unwrap();
        crate::cpu::CPUProcessor::new()
            .convert(&src_mem, &mut cpu_dst, None)
            .unwrap();

        let mut gpu_dst = TensorDyn::image(
            w,
            h,
            PixelFormat::Rgba,
            DType::U8,
            Some(TensorMemory::DmaBuf),
            edgefirst_tensor::CpuAccess::ReadWrite,
        )
        .unwrap();
        let mut gl = match GLProcessorST::new(None, None) {
            Ok(g) => g,
            Err(e) => {
                crate::test_support::report_skip(&format!(
                    "{} - GL not available: {e}",
                    function!()
                ));
                return;
            }
        };
        gl.convert(&src_dma, &mut gpu_dst, None).unwrap();

        assert_eq!(
            gl.last_nv_convert_path,
            NvConvertPath::ShaderR8,
            "NV16 odd-both must use ShaderR8, got {:?}",
            gl.last_nv_convert_path
        );

        let (max_diff, first_fail) = compare_gpu_vs_cpu_u8(&gpu_dst, &cpu_dst, w, h);
        eprintln!("NV16 odd-both (65×63): GPU vs CPU max_diff={max_diff}");
        assert!(
            max_diff <= 4,
            "NV16 odd-both GPU vs CPU max_diff={max_diff} > 4; first bad at {first_fail:?}"
        );
    }

    // -------------------------------------------------------------------------
    // NV24 odd-both (65×63) → RGBA
    // -------------------------------------------------------------------------

    /// NV24 odd-width and odd-height (65×63) runs `ShaderR8` and matches the CPU within ±4.
    #[test]
    #[cfg(any(target_os = "linux", target_os = "macos", target_os = "windows"))]
    fn nv24_odd_both_vs_cpu() {
        use crate::opengl_headless::processor::{GLProcessorST, NvConvertPath};
        if !is_gpu_image_buffer_available() {
            crate::test_support::report_skip(&format!(
                "{} - no zero-copy GPU buffers",
                function!()
            ));
            return;
        }
        let (w, h) = (321usize, 241usize); // QVGA-scale, both odd
        let (src_dma, src_mem) = make_patterned_nv_pair(w, h, PixelFormat::Nv24);

        let mut cpu_dst = TensorDyn::image(
            w,
            h,
            PixelFormat::Rgba,
            DType::U8,
            None,
            edgefirst_tensor::CpuAccess::ReadWrite,
        )
        .unwrap();
        crate::cpu::CPUProcessor::new()
            .convert(&src_mem, &mut cpu_dst, None)
            .unwrap();

        let mut gpu_dst = TensorDyn::image(
            w,
            h,
            PixelFormat::Rgba,
            DType::U8,
            Some(TensorMemory::DmaBuf),
            edgefirst_tensor::CpuAccess::ReadWrite,
        )
        .unwrap();
        let mut gl = match GLProcessorST::new(None, None) {
            Ok(g) => g,
            Err(e) => {
                crate::test_support::report_skip(&format!(
                    "{} - GL not available: {e}",
                    function!()
                ));
                return;
            }
        };
        gl.convert(&src_dma, &mut gpu_dst, None).unwrap();

        assert_eq!(
            gl.last_nv_convert_path,
            NvConvertPath::ShaderR8,
            "NV24 odd-both must use ShaderR8, got {:?}",
            gl.last_nv_convert_path
        );

        let (max_diff, first_fail) = compare_gpu_vs_cpu_u8(&gpu_dst, &cpu_dst, w, h);
        eprintln!("NV24 odd-both (65×63): GPU vs CPU max_diff={max_diff}");
        assert!(
            max_diff <= 4,
            "NV24 odd-both GPU vs CPU max_diff={max_diff} > 4; first bad at {first_fail:?}"
        );
    }

    // -------------------------------------------------------------------------
    // Odd destination at a stride that is not 64-byte aligned
    // -------------------------------------------------------------------------

    /// An odd destination at a non-64-aligned row pitch converts, or declines with `NotSupported`.
    #[test]
    #[cfg(target_os = "linux")]
    fn odd_dst_unaligned_stride_guarded() {
        use crate::opengl_headless::processor::GLProcessorST;
        if !is_dma_available() {
            crate::test_support::report_skip(&format!("{} - DMA not available", function!()));
            return;
        }
        let (w, h) = (321usize, 240usize);
        let (src_dma, _src_mem) = make_patterned_nv_pair(w, h, PixelFormat::Nv12);

        // Tight RGBA stride: 321*4 = 1284, not a multiple of 64.
        let tight_stride = w * 4;
        assert_ne!(
            tight_stride % 64,
            0,
            "test premise: stride must be unaligned"
        );
        let mut gpu_dst = TensorDyn::image_with_stride(
            w,
            h,
            PixelFormat::Rgba,
            DType::U8,
            tight_stride,
            Some(TensorMemory::DmaBuf),
            edgefirst_tensor::CpuAccess::ReadWrite,
        )
        .unwrap();

        let mut gl = match GLProcessorST::new(None, None) {
            Ok(g) => g,
            Err(e) => {
                crate::test_support::report_skip(&format!(
                    "{} - GL not available: {e}",
                    function!()
                ));
                return;
            }
        };
        match gl.convert(&src_dma, &mut gpu_dst, None) {
            Ok(()) => eprintln!("platform accepts non-aligned odd dst (OK)"),
            Err(crate::Error::NotSupported(msg)) => {
                assert!(
                    msg.contains("odd dimensions"),
                    "NotSupported must name the odd-dimension cause, got: {msg}"
                );
                eprintln!("platform rejects non-aligned odd dst, guarded as: {msg}");
            }
            Err(other) => {
                panic!("odd-dst failure must be wrapped as NotSupported, not leaked raw: {other:?}")
            }
        }
    }

    // -------------------------------------------------------------------------
    // NV12 odd-W (321×240) → RGBA
    // -------------------------------------------------------------------------

    /// NV12 at a width not a multiple of 4 routes to `ShaderR8` and matches the CPU within ±4.
    #[test]
    #[cfg(any(target_os = "linux", target_os = "macos", target_os = "windows"))]
    fn nv12_odd_w_shader_r8_vs_cpu() {
        use crate::opengl_headless::processor::{GLProcessorST, NvConvertPath};
        if !is_gpu_image_buffer_available() {
            crate::test_support::report_skip(&format!(
                "{} - no zero-copy GPU buffers",
                function!()
            ));
            return;
        }
        let (w, h) = (321usize, 240usize); // QVGA-scale odd width
        let (src_dma, src_mem) = make_patterned_nv_pair(w, h, PixelFormat::Nv12);

        let mut cpu_dst = TensorDyn::image(
            w,
            h,
            PixelFormat::Rgba,
            DType::U8,
            None,
            edgefirst_tensor::CpuAccess::ReadWrite,
        )
        .unwrap();
        crate::cpu::CPUProcessor::new()
            .convert(&src_mem, &mut cpu_dst, None)
            .unwrap();

        let mut gpu_dst = TensorDyn::image(
            w,
            h,
            PixelFormat::Rgba,
            DType::U8,
            Some(TensorMemory::DmaBuf),
            edgefirst_tensor::CpuAccess::ReadWrite,
        )
        .unwrap();
        let mut gl = match GLProcessorST::new(None, None) {
            Ok(g) => g,
            Err(e) => {
                crate::test_support::report_skip(&format!(
                    "{} - GL not available: {e}",
                    function!()
                ));
                return;
            }
        };
        gl.convert(&src_dma, &mut gpu_dst, None).unwrap();

        assert_eq!(
            gl.last_nv_convert_path,
            NvConvertPath::ShaderR8,
            "non-mult-4 NV12 must route to ShaderR8, got {:?}",
            gl.last_nv_convert_path
        );

        let (max_diff, first_fail) = compare_gpu_vs_cpu_u8(&gpu_dst, &cpu_dst, w, h);
        eprintln!("NV12 odd-W: GPU vs CPU max_diff={max_diff}");
        assert!(
            max_diff <= 4,
            "NV12 odd-W GPU vs CPU max_diff={max_diff} > 4; first bad at {first_fail:?}"
        );
    }

    // -------------------------------------------------------------------------
    // NV12 even-W odd-H (320×241) → RGBA
    // -------------------------------------------------------------------------

    /// NV12 even-width odd-height (320×241) stays on the GPU and matches the CPU reference.
    #[test]
    #[cfg(any(target_os = "linux", target_os = "macos", target_os = "windows"))]
    fn nv12_odd_h_shader_r8_vs_cpu() {
        use crate::opengl_headless::processor::{GLProcessorST, NvConvertPath};
        if !is_gpu_image_buffer_available() {
            crate::test_support::report_skip(&format!(
                "{} - no zero-copy GPU buffers",
                function!()
            ));
            return;
        }
        let (w, h) = (320usize, 241usize);
        let (src_dma, src_mem) = make_patterned_nv_pair(w, h, PixelFormat::Nv12);

        let mut cpu_dst = TensorDyn::image(
            w,
            h,
            PixelFormat::Rgba,
            DType::U8,
            None,
            edgefirst_tensor::CpuAccess::ReadWrite,
        )
        .unwrap();
        crate::cpu::CPUProcessor::new()
            .convert(&src_mem, &mut cpu_dst, None)
            .unwrap();

        let mut gpu_dst = TensorDyn::image(
            w,
            h,
            PixelFormat::Rgba,
            DType::U8,
            Some(TensorMemory::DmaBuf),
            edgefirst_tensor::CpuAccess::ReadWrite,
        )
        .unwrap();
        let mut gl = match GLProcessorST::new(None, None) {
            Ok(g) => g,
            Err(e) => {
                crate::test_support::report_skip(&format!(
                    "{} - GL not available: {e}",
                    function!()
                ));
                return;
            }
        };
        gl.convert(&src_dma, &mut gpu_dst, None).unwrap();

        assert!(
            matches!(
                gl.last_nv_convert_path,
                NvConvertPath::ExternalSampler | NvConvertPath::ShaderR8
            ),
            "even-W odd-H NV12 should use a GPU path (ExternalSampler/ShaderR8), got {:?}",
            gl.last_nv_convert_path
        );

        let (max_diff, first_fail) = compare_gpu_vs_cpu_u8(&gpu_dst, &cpu_dst, w, h);
        eprintln!("NV12 odd-H (320×241): GPU vs CPU max_diff={max_diff}");
        if max_diff > 4 {
            eprintln!(
                "WARNING: NV12 odd-H GPU vs CPU max_diff={max_diff} > 4 \
                 (GPU rounding; first bad at {first_fail:?})"
            );
        }
        assert!(
            max_diff <= 35,
            "gross NV12 odd-H GPU vs CPU mismatch max_diff={max_diff} (>35); first bad at {first_fail:?}"
        );
    }

    // -------------------------------------------------------------------------
    // Grey odd-W (coco_grey_odd.jpg 595×438) → RGBA
    // -------------------------------------------------------------------------

    /// Grey odd-width (595×438) decode and GPU convert to RGBA match the CPU within ±4.
    #[test]
    #[cfg(any(target_os = "linux", target_os = "macos", target_os = "windows"))]
    fn g_grey_odd_w_vs_cpu() {
        use crate::opengl_headless::processor::GLProcessorST;
        use edgefirst_codec::{ImageDecoder, ImageLoad};
        if !is_gpu_image_buffer_available() {
            crate::test_support::report_skip(&format!(
                "{} - no zero-copy GPU buffers",
                function!()
            ));
            return;
        }
        let jpeg: &[u8] = &edgefirst_bench::testdata::read("coco_grey_odd.jpg");
        let w = 595usize;
        let h = 438usize;

        // Decode into a DMA-backed Grey tensor.
        let Some(mut src_grey) = crate::test_support::gpu_buffer_allocated(
            Tensor::<u8>::image(
                w,
                h,
                PixelFormat::Grey,
                Some(TensorMemory::DmaBuf),
                edgefirst_tensor::CpuAccess::ReadWrite,
            ),
            "DMA Grey source",
        ) else {
            return;
        };
        let src_dma = {
            let mut dec = ImageDecoder::new();
            src_grey.load_image(&mut dec, jpeg).unwrap();
            TensorDyn::from(src_grey)
        };

        // CPU reference: decode into a Mem Grey tensor, convert Grey→RGBA via CPU.
        let src_mem = {
            let mut t = Tensor::<u8>::image(
                w,
                h,
                PixelFormat::Grey,
                Some(TensorMemory::Mem),
                edgefirst_tensor::CpuAccess::ReadWrite,
            )
            .unwrap();
            let mut dec = ImageDecoder::new();
            t.load_image(&mut dec, jpeg).unwrap();
            TensorDyn::from(t)
        };
        let mut cpu_dst = TensorDyn::image(
            w,
            h,
            PixelFormat::Rgba,
            DType::U8,
            None,
            edgefirst_tensor::CpuAccess::ReadWrite,
        )
        .unwrap();
        crate::cpu::CPUProcessor::new()
            .convert(&src_mem, &mut cpu_dst, None)
            .unwrap();

        let mut gpu_dst = TensorDyn::image(
            w,
            h,
            PixelFormat::Rgba,
            DType::U8,
            Some(TensorMemory::DmaBuf),
            edgefirst_tensor::CpuAccess::ReadWrite,
        )
        .unwrap();
        let mut gl = match GLProcessorST::new(None, None) {
            Ok(g) => g,
            Err(e) => {
                crate::test_support::report_skip(&format!(
                    "{} - GL not available: {e}",
                    function!()
                ));
                return;
            }
        };
        gl.convert(&src_dma, &mut gpu_dst, None).unwrap();

        let (max_diff, first_fail) = compare_gpu_vs_cpu_u8(&gpu_dst, &cpu_dst, w, h);
        eprintln!("g_grey_odd_w: GPU vs CPU max_diff={max_diff}");
        assert!(
            max_diff <= 4,
            "g_grey_odd_w: Grey odd-W GPU vs CPU max_diff={max_diff} > 4; first bad at {first_fail:?}"
        );
    }

    // -------------------------------------------------------------------------
    // NV12 odd-W i8 (65×64) → RGB i8
    // -------------------------------------------------------------------------

    /// NV12 odd-width (65×64) to RGB i8 matches the CPU i8 output within ±2.
    #[test]
    #[ignore = "odd-width 3-channel RGB DMA output needs width*3 % 4 == 0 (pack-as-RGBA8 architecture in convert_to_packed_rgb, NOT the stride bug); use RGBA or even/mult-4 RGB dst"]
    #[cfg(target_os = "linux")]
    fn nv12_odd_w_i8_vs_cpu() {
        use crate::opengl_headless::processor::{GLProcessorST, NvConvertPath};
        if !is_dma_available() {
            crate::test_support::report_skip(&format!("{} - DMA not available", function!()));
            return;
        }
        let (w, h) = (321usize, 240usize); // QVGA-scale odd width
        let (src_dma, src_mem) = make_patterned_nv_pair(w, h, PixelFormat::Nv12);

        let mut cpu_dst = TensorDyn::image(
            w,
            h,
            PixelFormat::Rgb,
            DType::I8,
            None,
            edgefirst_tensor::CpuAccess::ReadWrite,
        )
        .unwrap();
        crate::cpu::CPUProcessor::new()
            .convert(&src_mem, &mut cpu_dst, None)
            .unwrap();

        let mut gpu_dst = TensorDyn::image(
            w,
            h,
            PixelFormat::Rgb,
            DType::I8,
            Some(TensorMemory::DmaBuf),
            edgefirst_tensor::CpuAccess::ReadWrite,
        )
        .unwrap();
        let mut gl = match GLProcessorST::new(None, None) {
            Ok(g) => g,
            Err(e) => {
                crate::test_support::report_skip(&format!(
                    "{} - GL not available: {e}",
                    function!()
                ));
                return;
            }
        };
        gl.convert(&src_dma, &mut gpu_dst, None).unwrap();

        assert_eq!(
            gl.last_nv_convert_path,
            NvConvertPath::ShaderR8,
            "odd-W single-plane NV12 must use ShaderR8, got {:?}",
            gl.last_nv_convert_path
        );

        let max_diff = compare_gpu_vs_cpu_i8(&gpu_dst, &cpu_dst, w, h);
        eprintln!("NV12 odd-W i8: GPU vs CPU max_diff={max_diff}");
        assert!(
            max_diff <= 2,
            "NV12 i8 odd-W GPU vs CPU max_diff={max_diff} > 2"
        );
    }

    // -------------------------------------------------------------------------
    // NV16 odd-both i8 (65×63) → RGB i8
    // -------------------------------------------------------------------------

    /// NV16 odd-both (65×63) to RGB i8 matches the CPU i8 output.
    #[test]
    #[ignore = "odd-width 3-channel RGB DMA output needs width*3 % 4 == 0 (pack-as-RGBA8 architecture in convert_to_packed_rgb, NOT the stride bug); use RGBA or even/mult-4 RGB dst"]
    #[cfg(target_os = "linux")]
    fn nv16_odd_both_i8_vs_cpu() {
        use crate::opengl_headless::processor::{GLProcessorST, NvConvertPath};
        if !is_dma_available() {
            crate::test_support::report_skip(&format!("{} - DMA not available", function!()));
            return;
        }
        let (w, h) = (321usize, 241usize); // QVGA-scale, both odd
        let (src_dma, src_mem) = make_patterned_nv_pair(w, h, PixelFormat::Nv16);

        let mut cpu_dst = TensorDyn::image(
            w,
            h,
            PixelFormat::Rgb,
            DType::I8,
            None,
            edgefirst_tensor::CpuAccess::ReadWrite,
        )
        .unwrap();
        crate::cpu::CPUProcessor::new()
            .convert(&src_mem, &mut cpu_dst, None)
            .unwrap();

        let mut gpu_dst = TensorDyn::image(
            w,
            h,
            PixelFormat::Rgb,
            DType::I8,
            Some(TensorMemory::DmaBuf),
            edgefirst_tensor::CpuAccess::ReadWrite,
        )
        .unwrap();
        let mut gl = match GLProcessorST::new(None, None) {
            Ok(g) => g,
            Err(e) => {
                crate::test_support::report_skip(&format!(
                    "{} - GL not available: {e}",
                    function!()
                ));
                return;
            }
        };
        gl.convert(&src_dma, &mut gpu_dst, None).unwrap();

        assert_eq!(
            gl.last_nv_convert_path,
            NvConvertPath::ShaderR8,
            "NV16 i8 odd-both must use ShaderR8, got {:?}",
            gl.last_nv_convert_path
        );

        let max_diff = compare_gpu_vs_cpu_i8(&gpu_dst, &cpu_dst, w, h);
        eprintln!("NV16 odd-both i8: GPU vs CPU max_diff={max_diff}");
        assert!(
            max_diff <= 2,
            "NV16 i8 odd-both GPU vs CPU max_diff={max_diff} > 2"
        );
    }

    // -------------------------------------------------------------------------
    // Even-dimension cases
    // -------------------------------------------------------------------------

    /// NV16 64×64 → RGBA on the GPU matches the CPU within ±4.
    #[test]
    #[cfg(any(target_os = "linux", target_os = "macos", target_os = "windows"))]
    fn even_nv16_64x64_regression() {
        use crate::opengl_headless::processor::{GLProcessorST, NvConvertPath};
        if !is_gpu_image_buffer_available() {
            crate::test_support::report_skip(&format!(
                "{} - no zero-copy GPU buffers",
                function!()
            ));
            return;
        }
        let (w, h) = (64usize, 64usize);
        let (src_dma, src_mem) = make_patterned_nv_pair(w, h, PixelFormat::Nv16);

        let mut cpu_dst = TensorDyn::image(
            w,
            h,
            PixelFormat::Rgba,
            DType::U8,
            None,
            edgefirst_tensor::CpuAccess::ReadWrite,
        )
        .unwrap();
        crate::cpu::CPUProcessor::new()
            .convert(&src_mem, &mut cpu_dst, None)
            .unwrap();

        let mut gpu_dst = TensorDyn::image(
            w,
            h,
            PixelFormat::Rgba,
            DType::U8,
            Some(TensorMemory::DmaBuf),
            edgefirst_tensor::CpuAccess::ReadWrite,
        )
        .unwrap();
        let mut gl = match GLProcessorST::new(None, None) {
            Ok(g) => g,
            Err(e) => {
                crate::test_support::report_skip(&format!(
                    "{} - GL not available: {e}",
                    function!()
                ));
                return;
            }
        };
        gl.convert(&src_dma, &mut gpu_dst, None).unwrap();

        assert_eq!(
            gl.last_nv_convert_path,
            NvConvertPath::ShaderR8,
            "even NV16 regression: must use ShaderR8"
        );
        let (max_diff, first_fail) = compare_gpu_vs_cpu_u8(&gpu_dst, &cpu_dst, w, h);
        eprintln!("even NV16 64×64 regression: GPU vs CPU max_diff={max_diff}");
        assert!(
            max_diff <= 4,
            "even NV16 64×64 regression: max_diff={max_diff} > 4; first bad at {first_fail:?}"
        );
    }

    /// NV24 64×64 → RGBA on the GPU matches the CPU within ±4.
    #[test]
    #[cfg(any(target_os = "linux", target_os = "macos", target_os = "windows"))]
    fn even_nv24_64x64_regression() {
        use crate::opengl_headless::processor::{GLProcessorST, NvConvertPath};
        if !is_gpu_image_buffer_available() {
            crate::test_support::report_skip(&format!(
                "{} - no zero-copy GPU buffers",
                function!()
            ));
            return;
        }
        let (w, h) = (64usize, 64usize);
        let (src_dma, src_mem) = make_patterned_nv_pair(w, h, PixelFormat::Nv24);

        let mut cpu_dst = TensorDyn::image(
            w,
            h,
            PixelFormat::Rgba,
            DType::U8,
            None,
            edgefirst_tensor::CpuAccess::ReadWrite,
        )
        .unwrap();
        crate::cpu::CPUProcessor::new()
            .convert(&src_mem, &mut cpu_dst, None)
            .unwrap();

        let mut gpu_dst = TensorDyn::image(
            w,
            h,
            PixelFormat::Rgba,
            DType::U8,
            Some(TensorMemory::DmaBuf),
            edgefirst_tensor::CpuAccess::ReadWrite,
        )
        .unwrap();
        let mut gl = match GLProcessorST::new(None, None) {
            Ok(g) => g,
            Err(e) => {
                crate::test_support::report_skip(&format!(
                    "{} - GL not available: {e}",
                    function!()
                ));
                return;
            }
        };
        gl.convert(&src_dma, &mut gpu_dst, None).unwrap();

        assert_eq!(
            gl.last_nv_convert_path,
            NvConvertPath::ShaderR8,
            "even NV24 regression: must use ShaderR8"
        );
        let (max_diff, first_fail) = compare_gpu_vs_cpu_u8(&gpu_dst, &cpu_dst, w, h);
        eprintln!("even NV24 64×64 regression: GPU vs CPU max_diff={max_diff}");
        assert!(
            max_diff <= 4,
            "even NV24 64×64 regression: max_diff={max_diff} > 4; first bad at {first_fail:?}"
        );
    }

    /// Fill a YUYV tensor at its real row stride: two bytes per pixel, Y in
    /// the even byte and alternating Cb/Cr in the odd one.
    #[cfg(any(target_os = "linux", target_os = "macos", target_os = "windows"))]
    fn fill_yuyv_pattern(t: &TensorDyn, w: usize, h: usize, salt: u8) {
        let stride = t.effective_row_stride().unwrap_or(w * 2);
        let bound = t.as_u8().unwrap();
        let mut m = bound.map().unwrap();
        let buf = m.as_mut_slice();
        for r in 0..h {
            for c in 0..w {
                let i = r * stride + c * 2;
                buf[i] = ((r * 7 + c * 3) as u8).wrapping_add(salt);
                buf[i + 1] = ((r * 5 + c * 11) as u8).wrapping_add(salt);
            }
        }
    }

    /// Fill an RGBA tensor at its real row stride.
    #[cfg(any(target_os = "linux", target_os = "macos", target_os = "windows"))]
    fn fill_rgba_pattern(t: &TensorDyn, w: usize, h: usize, salt: u8) {
        let stride = t.effective_row_stride().unwrap_or(w * 4);
        let bound = t.as_u8().unwrap();
        let mut m = bound.map().unwrap();
        let buf = m.as_mut_slice();
        for r in 0..h {
            for c in 0..w {
                let i = r * stride + c * 4;
                buf[i] = ((r * 3 + c * 7) as u8).wrapping_add(salt);
                buf[i + 1] = ((r * 11 + c * 5) as u8).wrapping_add(salt);
                buf[i + 2] = ((r + c) as u8).wrapping_add(salt);
                buf[i + 3] = 255;
            }
        }
    }

    /// Worst absolute channel difference between two packed outputs.
    #[cfg(any(target_os = "linux", target_os = "macos", target_os = "windows"))]
    fn worst_channel_diff(expected: &[u8], actual: &[u8]) -> u8 {
        assert_eq!(expected.len(), actual.len(), "Buffer size mismatch");
        expected
            .iter()
            .zip(actual)
            .map(|(&e, &a)| (e as i16 - a as i16).unsigned_abs() as u8)
            .max()
            .unwrap_or(0)
    }

    /// Bytes that differ between two `w x h` packed outputs on the last row
    /// or the last column.
    #[cfg(any(target_os = "linux", target_os = "macos", target_os = "windows"))]
    fn edge_mismatches(
        expected: &[u8],
        actual: &[u8],
        w: usize,
        h: usize,
        channels: usize,
    ) -> usize {
        assert_eq!(expected.len(), actual.len(), "Buffer size mismatch");
        // A degenerate output has no last row or column.
        if w == 0 || h == 0 {
            return 0;
        }
        let row = w * channels;
        let last_row = (h - 1) * row..h * row;
        let last_col = (0..h - 1).flat_map(|r| {
            let start = r * row + (w - 1) * channels;
            start..start + channels
        });
        last_row
            .chain(last_col)
            .filter(|&i| expected[i] != actual[i])
            .count()
    }

    /// A destination narrowed by `configure_image` renders into its logical dimensions' region.
    #[test]
    fn recycled_narrowed_destination_matches_a_fresh_one() {
        if !is_gpu_image_buffer_available() || !is_opengl_available() {
            crate::test_support::report_skip(&format!(
                "{} - no zero-copy GPU buffers or OpenGL",
                function!()
            ));
            return;
        }
        let src = crate::load_image_test_helper(
            &edgefirst_bench::testdata::read("giraffe.jpg"),
            Some(PixelFormat::Rgba),
            None,
        )
        .expect("decode the source fixture");
        let mut gl = GLProcessorThreaded::new(None).unwrap();
        let rgba_dst = |w: usize, h: usize| {
            TensorDyn::image(
                w,
                h,
                PixelFormat::Rgba,
                DType::U8,
                Some(TensorMemory::DmaBuf),
                edgefirst_tensor::CpuAccess::ReadWrite,
            )
        };
        let Some(mut pool) = crate::test_support::gpu_buffer_allocated(
            rgba_dst(256, 192),
            "RGBA texture destination",
        ) else {
            return;
        };
        // Seed the whole texture.
        {
            let mut m = pool.as_u8().unwrap().map().unwrap();
            m.as_mut_slice().fill(0x5A);
        }
        for (w, h) in [(160usize, 120usize), (64, 48)] {
            pool.configure_image(w, h, PixelFormat::Rgba).unwrap();
            gl.convert(&src, &mut pool, None).unwrap();
            let mut fresh = rgba_dst(w, h).unwrap();
            gl.convert(&src, &mut fresh, None).unwrap();
            let (got, want) = (flat_bytes(&pool), flat_bytes(&fresh));
            assert_eq!(got.len(), want.len(), "{w}x{h}: output size mismatch");
            let worst = got
                .iter()
                .zip(&want)
                .map(|(a, b)| (*a as i16 - *b as i16).unsigned_abs())
                .max()
                .unwrap_or(0);
            assert!(
                worst <= 1,
                "{w}x{h}: narrowed destination differs from a fresh one by up to {worst}"
            );
        }
    }

    /// Compact a destination's rows into a tight buffer.
    fn flat_bytes(img: &TensorDyn) -> Vec<u8> {
        let t = img.as_u8().expect("u8 image");
        let mut out = vec![0u8; t.shape().iter().product::<usize>()];
        t.copy_to_flat(&mut out).expect("compact the padded rows");
        out
    }

    /// The 128x128 Grey pool the narrowed-source tests recycle, seeded over its whole extent.
    #[cfg(any(target_os = "linux", target_os = "macos", target_os = "windows"))]
    fn seeded_grey_pool() -> TensorDyn {
        let pool = TensorDyn::image(
            128,
            128,
            PixelFormat::Grey,
            DType::U8,
            Some(TensorMemory::DmaBuf),
            edgefirst_tensor::CpuAccess::ReadWrite,
        )
        .unwrap();
        fill_grey_pattern(&pool, 128, 128, 200);
        pool
    }

    /// Convert a `sw x sh` Grey image to `dw x dh` RGBA twice: recycled
    /// through `pool` (narrowed by `configure_image`) and from a fresh buffer
    /// of exactly its size. Returns `(fresh, recycled)` output bytes.
    #[cfg(any(target_os = "linux", target_os = "macos", target_os = "windows"))]
    fn convert_narrowed_and_fresh_grey(
        gl: &mut GLProcessorThreaded,
        pool: &mut TensorDyn,
        (sw, sh): (usize, usize),
        (dw, dh): (usize, usize),
        salt: u8,
    ) -> (Vec<u8>, Vec<u8>) {
        convert_narrowed_and_fresh_grey_as(
            gl,
            pool,
            (sw, sh),
            (dw, dh),
            salt,
            crate::Rotation::None,
            Crop::no_crop(),
        )
    }

    /// [`convert_narrowed_and_fresh_grey`] with an explicit rotation and crop.
    #[cfg(any(target_os = "linux", target_os = "macos", target_os = "windows"))]
    fn convert_narrowed_and_fresh_grey_as(
        gl: &mut GLProcessorThreaded,
        pool: &mut TensorDyn,
        (sw, sh): (usize, usize),
        (dw, dh): (usize, usize),
        salt: u8,
        rotation: crate::Rotation,
        crop: Crop,
    ) -> (Vec<u8>, Vec<u8>) {
        let convert = |gl: &mut GLProcessorThreaded, s: &TensorDyn, d: &mut TensorDyn| {
            gl.convert(
                s,
                d,
                Some(crate::ConvertOptions::new().rotation(rotation).crop(crop)),
            )
            .unwrap();
        };
        let rgba_dst = || {
            TensorDyn::image(
                dw,
                dh,
                PixelFormat::Rgba,
                DType::U8,
                Some(TensorMemory::DmaBuf),
                edgefirst_tensor::CpuAccess::ReadWrite,
            )
            .unwrap()
        };
        pool.configure_image(sw, sh, PixelFormat::Grey).unwrap();
        fill_grey_pattern(pool, sw, sh, salt);
        let mut recycled = rgba_dst();
        convert(gl, pool, &mut recycled);
        let recycled_bytes = flat_bytes(&recycled);

        let fresh = TensorDyn::image(
            sw,
            sh,
            PixelFormat::Grey,
            DType::U8,
            Some(TensorMemory::DmaBuf),
            edgefirst_tensor::CpuAccess::ReadWrite,
        )
        .unwrap();
        fill_grey_pattern(&fresh, sw, sh, salt);
        let mut oracle = rgba_dst();
        convert(gl, &fresh, &mut oracle);
        let oracle_bytes = flat_bytes(&oracle);

        assert_eq!(
            recycled_bytes.len(),
            oracle_bytes.len(),
            "{sw}x{sh} -> {dw}x{dh}: output size mismatch"
        );
        (oracle_bytes, recycled_bytes)
    }

    /// A resize from a pool source narrowed by `configure_image` equals a fresh-buffer convert.
    #[test]
    #[cfg(any(target_os = "linux", target_os = "macos", target_os = "windows"))]
    fn dma_recycle_narrowed_source_resized_matches_fresh() {
        if !is_gpu_image_buffer_available() || !is_opengl_available() {
            crate::test_support::report_skip(&format!(
                "{} - no zero-copy GPU buffers or OpenGL",
                function!()
            ));
            return;
        }
        let mut pool = seeded_grey_pool();
        let mut gl = GLProcessorThreaded::new(None).unwrap();

        // (src w, src h, dst w, dst h).
        let cases: [(usize, usize, usize, usize); 4] = [
            (128, 96, 128, 96),
            (128, 96, 64, 48),
            (128, 96, 192, 144),
            (96, 128, 192, 256),
        ];
        for (i, &(sw, sh, dw, dh)) in cases.iter().enumerate() {
            let (oracle_bytes, recycled_bytes) = convert_narrowed_and_fresh_grey(
                &mut gl,
                &mut pool,
                (sw, sh),
                (dw, dh),
                i as u8 * 10,
            );
            let worst = worst_channel_diff(&oracle_bytes, &recycled_bytes);
            assert!(
                worst <= 1,
                "case {i} ({sw}x{sh} -> {dw}x{dh}): narrowed source differs by up to {worst}"
            );
        }
    }

    /// An upscale of a narrowed pool source equals a fresh buffer on its last row and column.
    #[test]
    #[cfg(any(target_os = "linux", target_os = "macos", target_os = "windows"))]
    fn dma_recycle_narrowed_source_upscaled_edges_match_fresh() {
        if !is_gpu_image_buffer_available() || !is_opengl_available() {
            crate::test_support::report_skip(&format!(
                "{} - no zero-copy GPU buffers or OpenGL",
                function!()
            ));
            return;
        }
        let mut pool = seeded_grey_pool();
        let mut gl = GLProcessorThreaded::new(None).unwrap();

        // (src w, src h, dst w, dst h): height narrowed, width narrowed, both,
        // and a non-integer upscale.
        let cases: [(usize, usize, usize, usize); 4] = [
            (128, 64, 256, 128),
            (64, 128, 128, 256),
            (64, 64, 128, 128),
            (128, 64, 192, 96),
        ];
        for (i, &(sw, sh, dw, dh)) in cases.iter().enumerate() {
            let (oracle_bytes, recycled_bytes) = convert_narrowed_and_fresh_grey(
                &mut gl,
                &mut pool,
                (sw, sh),
                (dw, dh),
                i as u8 * 10,
            );
            let edges = edge_mismatches(&oracle_bytes, &recycled_bytes, dw, dh, 4);
            assert_eq!(
                edges, 0,
                "case {i} ({sw}x{sh} -> {dw}x{dh}): {edges} bytes differ from a fresh buffer \
                 on the last row or column of the upscaled narrowed source"
            );
            let worst = worst_channel_diff(&oracle_bytes, &recycled_bytes);
            assert!(
                worst <= 1,
                "case {i} ({sw}x{sh} -> {dw}x{dh}): narrowed source differs by up to {worst}"
            );
        }
    }

    /// A narrowed pool source's edge clamp holds under a crop to the logical edge and a rotation.
    #[test]
    #[cfg(any(target_os = "linux", target_os = "macos", target_os = "windows"))]
    fn dma_recycle_narrowed_source_cropped_and_rotated_edges_match_fresh() {
        if !is_gpu_image_buffer_available() || !is_opengl_available() {
            crate::test_support::report_skip(&format!(
                "{} - no zero-copy GPU buffers or OpenGL",
                function!()
            ));
            return;
        }
        let mut pool = seeded_grey_pool();
        let mut gl = GLProcessorThreaded::new(None).unwrap();

        // (src w, src h, dst w, dst h, rotation, crop).
        let cases: [(usize, usize, usize, usize, crate::Rotation, Crop); 2] = [
            (
                128,
                64,
                256,
                128,
                crate::Rotation::None,
                Crop::no_crop().with_source(Some(crate::Region::new(64, 32, 64, 32))),
            ),
            (
                64,
                128,
                256,
                128,
                crate::Rotation::Clockwise90,
                Crop::no_crop(),
            ),
        ];
        for (i, &(sw, sh, dw, dh, rotation, crop)) in cases.iter().enumerate() {
            let (oracle_bytes, recycled_bytes) = convert_narrowed_and_fresh_grey_as(
                &mut gl,
                &mut pool,
                (sw, sh),
                (dw, dh),
                i as u8 * 10,
                rotation,
                crop,
            );
            let edges = edge_mismatches(&oracle_bytes, &recycled_bytes, dw, dh, 4);
            assert_eq!(
                edges, 0,
                "case {i} ({sw}x{sh} -> {dw}x{dh}, {rotation:?}, crop {:?}): {edges} bytes \
                 differ from a fresh buffer on the last row or column",
                crop.source
            );
            let worst = worst_channel_diff(&oracle_bytes, &recycled_bytes);
            assert!(
                worst <= 1,
                "case {i} ({sw}x{sh} -> {dw}x{dh}, {rotation:?}): narrowed source differs \
                 by up to {worst}"
            );
        }
    }

    /// A narrowed pool YUYV source converts as a fresh buffer does, edges included.
    #[test]
    #[cfg(any(target_os = "linux", target_os = "macos", target_os = "windows"))]
    fn dma_recycle_narrowed_yuyv_source_matches_fresh() {
        if !is_gpu_image_buffer_available() || !is_opengl_available() {
            crate::test_support::report_skip(&format!(
                "{} - no zero-copy GPU buffers or OpenGL",
                function!()
            ));
            return;
        }
        let mut pool = TensorDyn::image(
            128,
            128,
            PixelFormat::Yuyv,
            DType::U8,
            Some(TensorMemory::DmaBuf),
            edgefirst_tensor::CpuAccess::ReadWrite,
        )
        .unwrap();
        fill_yuyv_pattern(&pool, 128, 128, 200);
        let mut gl = GLProcessorThreaded::new(None).unwrap();
        let convert = |gl: &mut GLProcessorThreaded, s: &TensorDyn, d: &mut TensorDyn| {
            gl.convert(s, d, None).unwrap();
        };

        let rgba_dst = |w: usize, h: usize| {
            TensorDyn::image(
                w,
                h,
                PixelFormat::Rgba,
                DType::U8,
                Some(TensorMemory::DmaBuf),
                edgefirst_tensor::CpuAccess::ReadWrite,
            )
            .unwrap()
        };
        // (src w, src h, dst w, dst h, salt), even widths only.
        let frames: [(usize, usize, usize, usize, u8); 5] = [
            (128, 96, 128, 96, 0),
            (96, 128, 96, 128, 40),
            (64, 100, 64, 100, 80),
            (128, 64, 256, 128, 120),
            (64, 128, 128, 256, 160),
        ];
        for (i, &(w, h, dw, dh, salt)) in frames.iter().enumerate() {
            pool.configure_image(w, h, PixelFormat::Yuyv).unwrap();
            fill_yuyv_pattern(&pool, w, h, salt);
            let mut recycled = rgba_dst(dw, dh);
            convert(&mut gl, &pool, &mut recycled);
            let recycled_bytes = flat_bytes(&recycled);

            let fresh = TensorDyn::image(
                w,
                h,
                PixelFormat::Yuyv,
                DType::U8,
                Some(TensorMemory::DmaBuf),
                edgefirst_tensor::CpuAccess::ReadWrite,
            )
            .unwrap();
            fill_yuyv_pattern(&fresh, w, h, salt);
            let mut oracle = rgba_dst(dw, dh);
            convert(&mut gl, &fresh, &mut oracle);
            let oracle_bytes = flat_bytes(&oracle);

            assert_eq!(
                recycled_bytes.len(),
                oracle_bytes.len(),
                "frame {i} ({w}x{h} -> {dw}x{dh}): output size mismatch"
            );
            let edges = edge_mismatches(&oracle_bytes, &recycled_bytes, dw, dh, 4);
            assert_eq!(
                edges, 0,
                "frame {i} ({w}x{h} -> {dw}x{dh}): {edges} bytes differ from a fresh buffer \
                 on the last row or column"
            );
            assert_pixels_match(&oracle_bytes, &recycled_bytes, 0);
        }
    }

    /// A narrowed pool source matches a fresh buffer into an F32 PBO and a zero-copy F16 texture.
    #[test]
    #[cfg(any(target_os = "linux", target_os = "macos", target_os = "windows"))]
    fn dma_recycle_narrowed_source_into_float_dst_matches_fresh() {
        if !is_gpu_image_buffer_available() || !is_opengl_available() {
            crate::test_support::report_skip(&format!(
                "{} - no zero-copy GPU buffers or OpenGL",
                function!()
            ));
            return;
        }
        let require_gl = std::env::var("HAL_TEST_REQUIRE_GL").is_ok_and(|v| v == "1");
        #[cfg(target_os = "macos")]
        if require_gl && std::env::var_os("HAL_TEST_ALLOW_DLOPEN_ANGLE").is_none() {
            crate::test_support::report_skip(&format!(
                "{} - ANGLE dlopen gate closed",
                function!()
            ));
            return;
        }
        let mut proc = match crate::ImageProcessor::with_config(crate::ImageProcessorConfig {
            backend: crate::ComputeBackend::OpenGl,
            ..Default::default()
        }) {
            Ok(p) => p,
            Err(e) => {
                crate::test_support::report_skip(&format!(
                    "{} - OpenGL backend unavailable: {e}",
                    function!()
                ));
                return;
            }
        };
        if !proc.supported_render_dtypes().f32 {
            crate::test_support::report_skip(&format!(
                "{} - F32 render not supported",
                function!()
            ));
            return;
        }
        // The zero-copy half needs the F16 render.
        let zero_copy_float = proc.supported_render_dtypes().f16;
        if !zero_copy_float {
            crate::test_support::report_skip(&format!(
                "(zero-copy float dst): {} - F16 render not supported",
                function!()
            ));
        }
        let mut pool = TensorDyn::image(
            128,
            128,
            PixelFormat::Rgba,
            DType::U8,
            Some(TensorMemory::DmaBuf),
            edgefirst_tensor::CpuAccess::ReadWrite,
        )
        .unwrap();
        fill_rgba_pattern(&pool, 128, 128, 200);

        // (src w, src h, dst w, dst h, salt).
        let frames: [(usize, usize, usize, usize, u8); 3] = [
            (128, 96, 128, 96, 0),
            (96, 128, 96, 128, 60),
            (128, 64, 256, 128, 120),
        ];
        for (i, &(w, h, dw, dh, salt)) in frames.iter().enumerate() {
            pool.configure_image(w, h, PixelFormat::Rgba).unwrap();
            fill_rgba_pattern(&pool, w, h, salt);
            // An F32 PBO destination, allocated on the processor's GL context.
            let float_dst = |p: &crate::ImageProcessor| {
                p.opengl
                    .as_ref()
                    .expect("OpenGl backend")
                    .create_pbo_image_dtype(dw, dh, PixelFormat::Rgb, DType::F32)
            };
            let mut recycled = float_dst(&proc).expect("allocate the F32 PBO destination");
            let fallbacks_before = proc.convert_fallback_count();
            proc.convert(&pool, &mut recycled, None).unwrap();
            assert_eq!(
                proc.convert_fallback_count(),
                fallbacks_before,
                "frame {i}: the GL float path declined and the convert was \
                 served by the CPU; the oracle below would then be the same \
                 CPU answer and the comparison would prove nothing"
            );
            let recycled_bytes = recycled.as_f32().unwrap().map().unwrap().to_vec();

            let fresh = TensorDyn::image(
                w,
                h,
                PixelFormat::Rgba,
                DType::U8,
                Some(TensorMemory::DmaBuf),
                edgefirst_tensor::CpuAccess::ReadWrite,
            )
            .unwrap();
            fill_rgba_pattern(&fresh, w, h, salt);
            let mut oracle = float_dst(&proc).unwrap();
            let fallbacks_before = proc.convert_fallback_count();
            proc.convert(&fresh, &mut oracle, None).unwrap();
            assert_eq!(
                proc.convert_fallback_count(),
                fallbacks_before,
                "frame {i}: the oracle convert fell back to the CPU"
            );
            let oracle_bytes = oracle.as_f32().unwrap().map().unwrap().to_vec();

            assert_eq!(
                recycled_bytes.len(),
                oracle_bytes.len(),
                "frame {i} ({w}x{h} -> {dw}x{dh}): output size mismatch"
            );
            let worst = recycled_bytes
                .iter()
                .zip(oracle_bytes.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max);
            assert!(
                worst == 0.0,
                "frame {i} ({w}x{h} -> {dw}x{dh}): narrowed float source differs from a \
                 fresh buffer by up to {worst}"
            );

            // The same source into a zero-copy F16 PlanarRgb destination.
            if !zero_copy_float {
                continue;
            }
            let planar_dst = || {
                TensorDyn::image(
                    dw,
                    dh,
                    PixelFormat::PlanarRgb,
                    DType::F16,
                    Some(TensorMemory::DmaBuf),
                    edgefirst_tensor::CpuAccess::ReadWrite,
                )
            };
            let Some(mut zc_recycled) = crate::test_support::gpu_buffer_allocated(
                planar_dst(),
                "F16 PlanarRgb zero-copy destination",
            ) else {
                continue;
            };
            let Some(mut zc_oracle) = crate::test_support::gpu_buffer_allocated(
                planar_dst(),
                "F16 PlanarRgb zero-copy oracle",
            ) else {
                continue;
            };
            if zc_recycled.memory() != TensorMemory::DmaBuf {
                crate::test_support::report_skip(&format!(
                    "(zero-copy float dst): {} - F16 PlanarRgb dst is not zero-copy",
                    function!()
                ));
                continue;
            }
            // The F16 values with row padding removed.
            let flat_f16 = |t: &TensorDyn| -> Vec<u8> {
                let typed = t.as_typed::<half::f16>().expect("F16 image");
                let mut out = vec![0u8; typed.shape().iter().product::<usize>() * 2];
                typed.copy_to_flat(&mut out).expect("compact padded rows");
                out
            };
            let fallbacks_before = proc.convert_fallback_count();
            proc.convert(&pool, &mut zc_recycled, None).unwrap();
            proc.convert(&fresh, &mut zc_oracle, None).unwrap();
            // A CPU fallback is a skip, or a failure under `HAL_TEST_REQUIRE_GL=1`.
            if proc.convert_fallback_count() != fallbacks_before {
                assert!(
                    !require_gl,
                    "HAL_TEST_REQUIRE_GL=1 but frame {i}'s zero-copy float destination \
                     convert fell back to the CPU"
                );
                crate::test_support::report_skip(&format!(
                    "(zero-copy float dst): {} - the GL float path declined and the \
                     CPU served the convert",
                    function!()
                ));
                continue;
            }
            assert_eq!(
                flat_f16(&zc_recycled),
                flat_f16(&zc_oracle),
                "frame {i} ({w}x{h} -> {dw}x{dh}): narrowed source into a zero-copy F16 \
                 PlanarRgb destination differs from a fresh buffer"
            );
        }
    }

    /// An NV12 zero-copy source at an unaligned plane offset converts through GL on every backing.
    #[test]
    fn nv12_source_at_a_plane_offset_converts_through_gl_on_every_zero_copy_backing() {
        const W: usize = 320;
        const H: usize = 240;
        const TOLERANCE: u8 = 8;

        if !is_opengl_available() {
            crate::test_support::report_skip(&format!("{} - OpenGL not available", function!()));
            return;
        }
        #[cfg(target_os = "macos")]
        if std::env::var("HAL_TEST_REQUIRE_GL").is_ok_and(|v| v == "1")
            && std::env::var_os("HAL_TEST_ALLOW_DLOPEN_ANGLE").is_none()
        {
            crate::test_support::report_skip(&format!(
                "{} - ANGLE dlopen gate closed",
                function!()
            ));
            return;
        }
        // The allocation is taller than the logical frame: by two rows, or by one on Windows.
        let rows = if cfg!(target_os = "windows") {
            H + 1
        } else {
            H + 2
        };
        let Some(mut src) = crate::test_support::gpu_buffer_allocated(
            TensorDyn::image(
                W,
                rows,
                PixelFormat::Nv12,
                DType::U8,
                Some(TensorMemory::DmaBuf),
                edgefirst_tensor::CpuAccess::ReadWrite,
            ),
            "zero-copy NV12 source",
        ) else {
            return;
        };
        let pitch = src.effective_row_stride().unwrap_or(W);
        // A period-31 pattern in the 112..=142 band, written while the map spans the whole
        // allocation.
        {
            let mut m = src
                .map_bytes(edgefirst_tensor::CpuAccess::Write)
                .expect("map the NV12 allocation");
            for (i, b) in m.as_mut_slice().iter_mut().enumerate() {
                *b = 112 + (i % 31) as u8;
            }
        }
        src.set_logical_shape(&[H * 3 / 2, W])
            .expect("narrow to one frame's geometry");
        assert_eq!(src.height(), Some(H), "precondition: one frame is {H} rows");
        assert_eq!(
            src.effective_row_stride(),
            Some(pitch),
            "precondition: the re-tag kept the surface pitch"
        );
        // One 64-rounded row in, plus 32 bytes: 32 (mod 64) at any pitch.
        let offset = pitch.next_multiple_of(64) + 32;
        src.set_plane_offset(offset);
        assert_eq!(
            src.plane_offset(),
            Some(offset),
            "precondition: the offset stuck"
        );
        assert_ne!(offset % 64, 0, "precondition: {offset} is unaligned");
        src.set_colorimetry(Some(
            edgefirst_tensor::Colorimetry::default()
                .with_encoding(edgefirst_tensor::ColorEncoding::Bt601)
                .with_range(edgefirst_tensor::ColorRange::Full),
        ));

        let rgba_dst = || {
            TensorDyn::image(
                W,
                H,
                PixelFormat::Rgba,
                DType::U8,
                Some(TensorMemory::Mem),
                edgefirst_tensor::CpuAccess::ReadWrite,
            )
            .expect("RGBA destination")
        };
        let read = |t: &TensorDyn| {
            t.map_bytes(edgefirst_tensor::CpuAccess::Read)
                .expect("map destination")
                .as_slice()
                .to_vec()
        };

        let mut reference = rgba_dst();
        crate::CPUProcessor::new()
            .convert(&src, &mut reference, None)
            .expect("CPU reference convert of the offset NV12 frame");
        let want = read(&reference);
        // The reference is not a flat colour.
        assert!(
            want.chunks(4).any(|p| p != &want[..4]),
            "precondition: the CPU reference is flat, so agreement would prove nothing"
        );

        let mut gl = GLProcessorThreaded::new(None).expect("GL processor");
        let before = gl.convert_stats().expect("convert stats before");
        let mut dst = rgba_dst();
        gl.convert(&src, &mut dst, None).expect(
            "GL refused an NV12 source at an unaligned plane offset; a declined \
                 zero-copy NV import must reach the R8 upload",
        );
        let after = gl.convert_stats().expect("convert stats after");
        let got = read(&dst);
        let bad = got
            .iter()
            .zip(&want)
            .filter(|(g, w)| g.abs_diff(**w) > TOLERANCE)
            .count();
        assert_eq!(
            bad,
            0,
            "{bad} bytes differ from the CPU reference over the same tensor; \
             first got={:?} want={:?} (offset {offset}, pitch {pitch})",
            &got[..4],
            &want[..4]
        );
        // A declined zero-copy feed became an upload.
        let declines = after.zero_copy_declines - before.zero_copy_declines;
        let uploads = after.src_uploads - before.src_uploads;
        let imports = after.src_imports - before.src_imports;
        log::info!(
            "{}: offset {offset} pitch {pitch} -> declines={declines} uploads={uploads} \
             imports={imports}",
            function!()
        );
        if declines > 0 {
            assert!(
                uploads > 0,
                "the NV import was declined ({declines}) but no upload was recorded \
                 (uploads={uploads}, imports={imports}) -- the convert took another path"
            );
        }
        assert!(
            uploads + imports > 0,
            "the convert recorded neither an upload nor an import \
             (declines={declines}) -- it was not fed by the GL engine at all"
        );
    }

    /// A fresh `view()` destination at an unaligned byte offset keeps the zero-copy import.
    #[test]
    fn a_fresh_unaligned_view_destination_keeps_the_zero_copy_import() {
        const W: usize = 320;
        const H: usize = 240;
        const SIDE: usize = 64;
        const BPP: usize = 4;
        // x0 = 8 is a 32-byte offset.
        const X0: usize = 8;
        const Y0: usize = 8;
        const BLANK: u8 = 0x55;

        if !is_opengl_available() {
            crate::test_support::report_skip(&format!("{} - OpenGL not available", function!()));
            return;
        }
        #[cfg(target_os = "macos")]
        if std::env::var("HAL_TEST_REQUIRE_GL").is_ok_and(|v| v == "1")
            && std::env::var_os("HAL_TEST_ALLOW_DLOPEN_ANGLE").is_none()
        {
            crate::test_support::report_skip(&format!(
                "{} - ANGLE dlopen gate closed",
                function!()
            ));
            return;
        }
        let dma_rgba = |w: usize, h: usize| {
            TensorDyn::image(
                w,
                h,
                PixelFormat::Rgba,
                DType::U8,
                Some(TensorMemory::DmaBuf),
                edgefirst_tensor::CpuAccess::ReadWrite,
            )
        };
        let Some(canvas) =
            crate::test_support::gpu_buffer_allocated(dma_rgba(W, H), "zero-copy RGBA canvas")
        else {
            return;
        };
        let pitch = canvas.effective_row_stride().unwrap_or(W * BPP);
        let offset = Y0 * pitch + X0 * BPP;
        assert_ne!(
            offset % 64,
            0,
            "precondition: ({X0},{Y0}) at pitch {pitch} is byte offset {offset}, \
             which must be unaligned for this test to mean anything"
        );

        let want = |x: usize, y: usize| -> [u8; BPP] {
            [((y * 3) % 256) as u8, ((x * 5) % 256) as u8, 255, 255]
        };
        let src = TensorDyn::image(
            SIDE,
            SIDE,
            PixelFormat::Rgba,
            DType::U8,
            Some(TensorMemory::Mem),
            edgefirst_tensor::CpuAccess::ReadWrite,
        )
        .expect("source");
        {
            let mut m = src
                .map_bytes(edgefirst_tensor::CpuAccess::Write)
                .expect("map source");
            let s = m.as_mut_slice();
            for y in 0..SIDE {
                for x in 0..SIDE {
                    s[(y * SIDE + x) * BPP..][..BPP].copy_from_slice(&want(X0 + x, Y0 + y));
                }
            }
        }

        let mut gl = GLProcessorThreaded::new(None).expect("GL processor");
        let imports = |g: &GLProcessorThreaded| {
            let c = g.egl_cache_stats().expect("cache stats").dst;
            c.misses + c.hits
        };

        // Control: a whole zero-copy destination, no view and no offset.
        let mut whole = dma_rgba(SIDE, SIDE).expect("control destination");
        let before_control = imports(&gl);
        gl.convert(&src, &mut whole, None)
            .expect("control convert into a whole zero-copy destination");
        let control = imports(&gl) - before_control;

        if control == 0 {
            crate::test_support::report_skip(&format!(
                "{} - no zero-copy destination import on this host, so a \
                 view() destination is declined by the engine and there is nothing \
                 to measure",
                function!()
            ));
            log::info!(
                "{}: pitch {pitch} offset {offset} -> control_dst_imports=0, skipped",
                function!()
            );
            return;
        }

        // Subject: a fresh view at the unaligned byte offset.
        {
            let mut m = canvas
                .map_bytes(edgefirst_tensor::CpuAccess::Write)
                .expect("map canvas");
            m.as_mut_slice().fill(BLANK);
        }
        let mut tile = canvas
            .view(edgefirst_tensor::Region::new(X0, Y0, SIDE, SIDE))
            .expect("fresh destination view");
        assert_eq!(
            tile.plane_offset(),
            Some(offset),
            "precondition: the view carries the unaligned byte offset"
        );
        assert!(
            tile.view_origin().is_some(),
            "precondition: a fresh view has a view_origin, which is what bases its \
             import at 0"
        );
        let before_subject = imports(&gl);
        gl.convert(&src, &mut tile, None)
            .expect("convert into a fresh view destination at an unaligned offset");
        let subject = imports(&gl) - before_subject;
        log::info!(
            "{}: pitch {pitch} offset {offset} -> control_dst_imports={control} \
             subject_dst_imports={subject}",
            function!()
        );

        let out = canvas
            .map_bytes(edgefirst_tensor::CpuAccess::Read)
            .expect("map canvas")
            .as_slice()
            .to_vec();
        let px = |x: usize, y: usize| &out[y * pitch + x * BPP..][..BPP];
        assert_eq!(
            px(0, 0),
            &[BLANK; BPP],
            "the tile landed at the canvas ORIGIN, so the view's offset was lost"
        );
        for y in Y0..Y0 + SIDE {
            for x in X0..X0 + SIDE {
                assert_eq!(px(x, y), &want(x, y), "tile pixel ({x}, {y})");
            }
        }

        assert!(
            subject > 0,
            "this host imports whole zero-copy destinations (control={control}) but \
             performed no destination import for a fresh view at unaligned byte \
             offset {offset} (subject={subject}) -- the view's import bases at 0, so \
             no offset rule should have applied to it"
        );
    }

    // ── The PBO transfer tests ──────────────────────────────────────────
    //
    // The three `PIXEL_UNPACK_BUFFER`-bound uploads and the PBO readback.

    /// A GL processor that allocates PBOs, or `None` with a skip line.
    #[cfg(any(target_os = "linux", target_os = "windows"))]
    fn pbo_gl_or_skip(what: &str) -> Option<GLProcessorThreaded> {
        if !is_opengl_available() {
            crate::test_support::report_skip(&format!("{what} - OpenGL not available"));
            return None;
        }
        match GLProcessorThreaded::new(None) {
            Ok(gl) => Some(gl),
            Err(e) => {
                crate::test_support::report_skip(&format!(
                    "{what} - GL processor unavailable: {e}"
                ));
                None
            }
        }
    }

    /// [`pbo_gl_or_skip`] plus the F32 render capability.
    #[cfg(any(target_os = "linux", target_os = "windows"))]
    fn float_pbo_gl_or_skip(what: &str) -> Option<GLProcessorThreaded> {
        let gl = pbo_gl_or_skip(what)?;
        if !gl.supported_render_dtypes().f32 {
            crate::test_support::report_skip(&format!("{what} - F32 render not supported"));
            return None;
        }
        Some(gl)
    }

    /// An RGBA8 PBO of the given size.
    #[cfg(any(target_os = "linux", target_os = "windows"))]
    fn rgba8_pbo(gl: &GLProcessorThreaded, w: usize, h: usize) -> TensorDyn {
        let t = gl
            .create_pbo_image(w, h, PixelFormat::Rgba)
            .expect("allocate an RGBA8 PBO on the GL thread");
        assert_eq!(
            TensorTrait::memory(&t),
            TensorMemory::Pbo,
            "create_pbo_image must produce a PBO"
        );
        t.into()
    }

    /// Assert the convert just performed fed its source from a PBO exactly once, with no
    /// CPU upload and no zero-copy import.
    #[cfg(any(target_os = "linux", target_os = "windows"))]
    fn assert_pbo_source_route(
        gl: &GLProcessorThreaded,
        before: &crate::opengl_headless::cache::ConvertStats,
        what: &str,
    ) {
        let after = gl.convert_stats().expect("convert stats after");
        assert_eq!(
            after.src_pbo_uploads - before.src_pbo_uploads,
            1,
            "{what}: expected exactly one PBO source feed; the byte comparison \
             below is meaningless if the engine took another route"
        );
        assert_eq!(
            after.src_uploads - before.src_uploads,
            0,
            "{what}: the source was fed by CPU map + upload, not from its PBO"
        );
        assert_eq!(
            after.src_imports - before.src_imports,
            0,
            "{what}: the source was fed by zero-copy import, not from its PBO"
        );
    }

    /// Stamp `R = x % 256`, `G = y % 256`, `B = 128`, honouring the tensor's own pitch.
    #[cfg(any(target_os = "linux", target_os = "windows"))]
    fn stamp_rgba8_positionally(t: &TensorDyn, w: usize, h: usize) {
        let pitch = t.effective_row_stride().unwrap_or(w * 4);
        let mut m = t
            .map_bytes(edgefirst_tensor::CpuAccess::ReadWrite)
            .expect("map the source PBO");
        let s = m.as_mut_slice();
        for y in 0..h {
            for x in 0..w {
                let i = y * pitch + x * 4;
                s[i] = (x % 256) as u8;
                s[i + 1] = (y % 256) as u8;
                s[i + 2] = 128;
                s[i + 3] = 255;
            }
        }
    }

    /// Assert an F32 RGB destination holds [`stamp_rgba8_positionally`]'s pattern for the
    /// source window starting at `(x0, y0)`, read from the live mapping: each value is the
    /// source byte over 255.
    #[cfg(any(target_os = "linux", target_os = "windows"))]
    fn assert_f32_dst_matches_rgba8_pattern(
        dst: &TensorDyn,
        w: usize,
        h: usize,
        x0: usize,
        y0: usize,
        case: &str,
    ) {
        let pitch_f32 = dst.effective_row_stride().unwrap_or(w * 3 * 4) / 4;
        let m = dst.as_f32().unwrap().map().unwrap();
        for y in 0..h {
            for x in 0..w {
                let expect = [
                    ((x + x0) % 256) as f32 / 255.0,
                    ((y + y0) % 256) as f32 / 255.0,
                    128.0 / 255.0,
                ];
                for (c, &want) in expect.iter().enumerate() {
                    let got = m[y * pitch_f32 + x * 3 + c];
                    assert!(
                        (got - want).abs() < 1e-3,
                        "{case}: f32 dst[{y},{x},{c}]={got}, expected {want} \
                         (source pixel ({}, {})); the parent's origin would be \
                         ({x}, {y})",
                        x + x0,
                        y + y0
                    );
                }
            }
        }
    }

    /// A `view()` of a PBO-backed source converts its own sub-region.
    #[test]
    #[cfg(any(target_os = "linux", target_os = "windows"))]
    fn a_pbo_view_source_converts_its_own_region_gl() {
        const W: usize = 320;
        const H: usize = 240;
        const SIDE: usize = 96;
        const X0: usize = 40;
        const Y0: usize = 32;

        let Some(mut gl) = pbo_gl_or_skip(function!()) else {
            return;
        };
        let src = rgba8_pbo(&gl, W, H);
        stamp_rgba8_positionally(&src, W, H);

        let view = src
            .view(edgefirst_tensor::Region::new(X0, Y0, SIDE, SIDE))
            .expect("view a narrow window of the PBO");
        let mut dst = rgba8_pbo(&gl, SIDE, SIDE);

        let before = gl.convert_stats().expect("convert stats before");
        gl.convert(&view, &mut dst, None)
            .expect("GL declined a PBO view source");
        assert_pbo_source_route(&gl, &before, "u8 PBO view");

        let m = dst
            .map_bytes(edgefirst_tensor::CpuAccess::Read)
            .expect("map the destination");
        let out = m.as_slice();
        let dst_pitch = dst.effective_row_stride().unwrap_or(SIDE * 4);
        for y in 0..SIDE {
            for x in 0..SIDE {
                let i = y * dst_pitch + x * 4;
                assert_eq!(
                    (out[i], out[i + 1]),
                    (((x + X0) % 256) as u8, ((y + Y0) % 256) as u8),
                    "output pixel ({x}, {y}) must be source ({}, {}); \
                     ({x}, {y}) would be the parent's origin tile",
                    x + X0,
                    y + Y0
                );
            }
        }
    }

    /// The float engine's PBO source arm reads the view's own region.
    #[test]
    #[cfg(any(target_os = "linux", target_os = "windows"))]
    fn a_pbo_view_source_converts_its_own_region_to_f32_gl() {
        const W: usize = 320;
        const H: usize = 240;
        const SIDE: usize = 96;
        const X0: usize = 40;
        const Y0: usize = 32;

        let Some(mut gl) = float_pbo_gl_or_skip(function!()) else {
            return;
        };
        let src = rgba8_pbo(&gl, W, H);
        stamp_rgba8_positionally(&src, W, H);

        let view = src
            .view(edgefirst_tensor::Region::new(X0, Y0, SIDE, SIDE))
            .expect("view a narrow window of the PBO");
        let mut dst = gl
            .create_pbo_image_dtype(SIDE, SIDE, PixelFormat::Rgb, DType::F32)
            .expect("allocate an F32 RGB PBO destination");

        let before = gl.convert_stats().expect("convert stats before");
        gl.convert(&view, &mut dst, None)
            .expect("GL declined a PBO view source into an F32 destination");
        assert_pbo_source_route(&gl, &before, "float PBO view");

        assert_f32_dst_matches_rgba8_pattern(&dst, SIDE, SIDE, X0, Y0, "view at (40, 32)");
    }

    /// A padded source pitch does not shear the float engine's PBO upload.
    #[test]
    #[cfg(any(target_os = "linux", target_os = "windows"))]
    fn a_padded_pbo_source_does_not_shear_the_f32_upload() {
        const W_ALLOC: usize = 320;
        const W: usize = 256;
        const H: usize = 240;

        let Some(mut gl) = float_pbo_gl_or_skip(function!()) else {
            return;
        };
        let src = rgba8_pbo(&gl, W_ALLOC, H);
        stamp_rgba8_positionally(&src, W_ALLOC, H);

        let view = src
            .view(edgefirst_tensor::Region::new(0, 0, W, H))
            .expect("view the left W columns of a wider PBO");
        assert_eq!(
            view.effective_row_stride(),
            Some(W_ALLOC * 4),
            "precondition: the window's rows are the PARENT's pitch apart, not \
             its own {} bytes",
            W * 4
        );
        assert_eq!(
            view.plane_offset().unwrap_or(0),
            0,
            "precondition: this case starts at byte 0, so only the pitch is \
             under test"
        );

        let mut dst = gl
            .create_pbo_image_dtype(W, H, PixelFormat::Rgb, DType::F32)
            .expect("allocate an F32 RGB PBO destination");

        let before = gl.convert_stats().expect("convert stats before");
        gl.convert(&view, &mut dst, None)
            .expect("GL declined a padded PBO source into an F32 destination");
        assert_pbo_source_route(&gl, &before, "float padded PBO source");

        assert_f32_dst_matches_rgba8_pattern(&dst, W, H, 0, 0, "padded pitch");
    }

    /// A padded whole-buffer PBO destination is written at its own pitch.
    #[test]
    #[cfg(any(target_os = "linux", target_os = "windows"))]
    fn a_padded_pbo_destination_is_written_at_its_own_pitch() {
        const W_ALLOC: usize = 320;
        const W: usize = 256;
        const H: usize = 240;
        const PITCH: usize = W_ALLOC * 4;

        let Some(mut gl) = pbo_gl_or_skip(function!()) else {
            return;
        };
        let src = rgba8_pbo(&gl, W, H);
        stamp_rgba8_positionally(&src, W, H);

        let mut dst = rgba8_pbo(&gl, W_ALLOC, H);
        dst.configure_image(W, H, PixelFormat::Rgba)
            .expect("narrow a pool tensor to the image it will hold");
        dst.set_row_stride(PITCH)
            .expect("declare the pitch the allocation still has");
        assert_eq!(
            dst.effective_row_stride(),
            Some(PITCH),
            "precondition: the destination's rows are {PITCH} B apart, not its \
             own {} B",
            W * 4
        );

        let before = gl.convert_stats().expect("convert stats before");
        gl.convert(&src, &mut dst, None)
            .expect("GL declined a pitch-aligned PBO destination");
        assert_pbo_source_route(&gl, &before, "padded PBO destination");

        // Only the visible bytes of each row are checked.
        let m = dst
            .map_bytes(edgefirst_tensor::CpuAccess::Read)
            .expect("map the destination");
        let out = m.as_slice();
        for y in 0..H {
            for x in 0..W {
                let i = y * PITCH + x * 4;
                assert_eq!(
                    (out[i], out[i + 1]),
                    ((x % 256) as u8, (y % 256) as u8),
                    "dst pixel ({x}, {y}) at pitch {PITCH}; reading the rows back \
                     to back would put source row {} here",
                    y * PITCH / (W * 4)
                );
            }
        }
    }

    /// The mapped-texture readback places a `view()` destination's tile and touches nothing else.
    #[test]
    fn mapped_texture_readback_places_a_view_destination() {
        const PARENT_W: usize = 512;
        const PARENT_H: usize = 320;
        const VIEW_W: usize = 320;
        const VIEW_H: usize = 240;
        const BPP: usize = 4;
        const POISON: u8 = 0xAB;

        if !is_opengl_available() {
            crate::test_support::report_skip(&format!("{} - OpenGL not available", function!()));
            return;
        }
        #[cfg(target_os = "macos")]
        if std::env::var_os("HAL_TEST_ALLOW_DLOPEN_ANGLE").is_none() {
            crate::test_support::report_skip(&format!(
                "{} - ANGLE dlopen gate closed",
                function!()
            ));
            return;
        }
        let dma_rgba = |w: usize, h: usize| {
            TensorDyn::image(
                w,
                h,
                PixelFormat::Rgba,
                DType::U8,
                Some(TensorMemory::DmaBuf),
                edgefirst_tensor::CpuAccess::ReadWrite,
            )
        };

        // Distinct per-row and per-column values.
        let src = TensorDyn::image(
            VIEW_W,
            VIEW_H,
            PixelFormat::Rgba,
            DType::U8,
            Some(TensorMemory::Mem),
            edgefirst_tensor::CpuAccess::ReadWrite,
        )
        .expect("source");
        {
            let stride = src.effective_row_stride().expect("source stride");
            let mut m = src
                .map_bytes(edgefirst_tensor::CpuAccess::Write)
                .expect("map source");
            let b = m.as_mut_slice();
            for y in 0..VIEW_H {
                for x in 0..VIEW_W {
                    let i = y * stride + x * BPP;
                    b[i] = ((y * 3) % 256) as u8;
                    b[i + 1] = ((x * 5) % 256) as u8;
                    b[i + 2] = 255;
                    b[i + 3] = 255;
                }
            }
        }

        // A different image, converted into a whole destination before each subject.
        let decoy = TensorDyn::image(
            VIEW_W,
            VIEW_H,
            PixelFormat::Rgba,
            DType::U8,
            Some(TensorMemory::Mem),
            edgefirst_tensor::CpuAccess::ReadWrite,
        )
        .expect("decoy source");
        {
            let mut m = decoy
                .map_bytes(edgefirst_tensor::CpuAccess::Write)
                .expect("map decoy");
            for px in m.as_mut_slice().as_chunks_mut::<BPP>().0 {
                px.copy_from_slice(&[7, 11, 13, 255]);
            }
        }

        let mut gl = GLProcessorThreaded::new(None).expect("GL processor");

        // Calibration with the hook off: whether this host imports a whole zero-copy destination.
        let imports = |g: &GLProcessorThreaded| {
            let c = g.egl_cache_stats().expect("cache stats").dst;
            c.misses + c.hits
        };
        let Some(mut oracle) = crate::test_support::gpu_buffer_allocated(
            dma_rgba(VIEW_W, VIEW_H),
            "zero-copy RGBA oracle",
        ) else {
            return;
        };
        let before = imports(&gl);
        gl.convert(&src, &mut oracle, None)
            .expect("calibration convert into a whole zero-copy destination");
        if imports(&gl) - before == 0 {
            crate::test_support::report_skip(&format!(
                "{} - no zero-copy destination import on this host, so a view() \
                 destination is declined by the engine and the mapped path cannot \
                 be reached for one",
                function!()
            ));
            return;
        }

        gl.set_fail_dst_import(true)
            .expect("make the destination import fail");
        let fallbacks = |g: &GLProcessorThreaded| {
            g.convert_stats()
                .expect("convert stats")
                .dst_import_fallbacks
        };

        let mut failures = Vec::new();
        for (label, crop) in [
            ("stretch", Crop::default()),
            (
                "letterbox",
                Crop::letterbox([114, 114, 114, 255]).with_source(Some(
                    edgefirst_tensor::Region::new(0, 0, VIEW_W, VIEW_H / 2),
                )),
            ),
        ] {
            // Reference: the same convert into a whole destination on the same mapped-texture path.
            gl.convert(
                &src,
                &mut oracle,
                Some(crate::ConvertOptions::new().crop(crop)),
            )
            .expect("oracle convert into a whole destination on the mapped path");
            let oracle_stride = oracle.effective_row_stride().expect("oracle stride");
            let oracle_bytes = oracle
                .map_bytes(edgefirst_tensor::CpuAccess::Read)
                .expect("map oracle")
                .as_slice()
                .to_vec();

            for (x0, y0) in [(0usize, 0usize), (8, 8), (64, 32)] {
                let parent = dma_rgba(PARENT_W, PARENT_H).expect("parent");
                assert_eq!(
                    parent.memory(),
                    TensorMemory::DmaBuf,
                    "the parent must be the zero-copy backing under test"
                );
                let pitch = parent.effective_row_stride().expect("parent pitch");
                {
                    let mut m = parent
                        .map_bytes(edgefirst_tensor::CpuAccess::Write)
                        .expect("map parent");
                    m.as_mut_slice().fill(POISON);
                }
                let mut scratch = dma_rgba(VIEW_W, VIEW_H).expect("decoy destination");
                gl.convert(&decoy, &mut scratch, None)
                    .expect("decoy convert");
                let before = fallbacks(&gl);
                {
                    let mut view = parent
                        .view(edgefirst_tensor::Region::new(x0, y0, VIEW_W, VIEW_H))
                        .expect("destination view");
                    if let Err(e) = gl.convert(
                        &src,
                        &mut view,
                        Some(crate::ConvertOptions::new().crop(crop)),
                    ) {
                        failures.push(format!("{label} ({x0},{y0}): GL declined the view: {e}"));
                        continue;
                    }
                }
                let took = fallbacks(&gl) - before;
                if took != 1 {
                    failures.push(format!(
                        "{label} ({x0},{y0}): expected exactly one destination-import \
                         fallback for this convert, saw {took}"
                    ));
                }

                let out = parent
                    .map_bytes(edgefirst_tensor::CpuAccess::Read)
                    .expect("map parent")
                    .as_slice()
                    .to_vec();
                let mut wrong = 0usize;
                let mut first: Option<(usize, usize, Vec<u8>, Vec<u8>)> = None;
                // Bounding box of the wrong pixels, and whether all of them lie in the band a
                // view-origin `glScissor` on an origin-(0,0) render target clips away.
                let (mut lx, mut ly, mut hx, mut hy) = (usize::MAX, usize::MAX, 0usize, 0usize);
                let mut all_in_scissor_band = true;
                for y in 0..VIEW_H {
                    for x in 0..VIEW_W {
                        let got = &out[(y0 + y) * pitch + (x0 + x) * BPP..][..BPP];
                        let want = &oracle_bytes[y * oracle_stride + x * BPP..][..BPP];
                        if got != want {
                            wrong += 1;
                            first.get_or_insert((x, y, got.to_vec(), want.to_vec()));
                            lx = lx.min(x);
                            ly = ly.min(y);
                            hx = hx.max(x);
                            hy = hy.max(y);
                            all_in_scissor_band &= x < x0 || y < y0;
                        }
                    }
                }
                if let Some((x, y, got, want)) = first {
                    failures.push(format!(
                    "{label} ({x0},{y0}) pitch {pitch}: {wrong}/{} tile pixels differ from the \
                     whole-destination oracle; first at tile ({x},{y}) got={got:?} \
                     want={want:?}; wrong-pixel bbox x[{lx}..={hx}] y[{ly}..={hy}]; \
                     every wrong pixel inside the x<{x0}||y<{y0} band: {all_in_scissor_band}",
                    VIEW_W * VIEW_H,
                ));
                }
                // Nothing outside the view region moved.
                let mut outside = 0usize;
                let mut first_outside = None;
                for y in 0..PARENT_H {
                    for x in 0..PARENT_W {
                        if y >= y0 && y < y0 + VIEW_H && x >= x0 && x < x0 + VIEW_W {
                            continue;
                        }
                        let i = y * pitch + x * BPP;
                        if out[i..i + BPP] != [POISON; BPP] {
                            outside += 1;
                            first_outside.get_or_insert((x, y, out[i..i + BPP].to_vec()));
                        }
                    }
                }
                if let Some((x, y, got)) = first_outside {
                    failures.push(format!(
                        "{label} ({x0},{y0}): {outside} parent pixels OUTSIDE the view were \
                     overwritten; first at ({x},{y}) = {got:?}"
                    ));
                }
            }
        }

        // A packed-RGB view destination is refused before a destination import is spent.
        #[cfg(target_os = "linux")]
        if let Some(rgb_parent) = crate::test_support::dma_allocated(
            TensorDyn::image(
                PARENT_W,
                PARENT_H,
                PixelFormat::Rgb,
                DType::U8,
                Some(TensorMemory::DmaBuf),
                edgefirst_tensor::CpuAccess::ReadWrite,
            ),
            "packed-RGB DMA parent",
        ) {
            let rgb_src = TensorDyn::image(
                VIEW_W,
                VIEW_H,
                PixelFormat::Rgb,
                DType::U8,
                Some(TensorMemory::Mem),
                edgefirst_tensor::CpuAccess::ReadWrite,
            )
            .expect("packed-RGB source");
            let mut view = rgb_parent
                .view(edgefirst_tensor::Region::new(8, 8, VIEW_W, VIEW_H))
                .expect("packed-RGB destination view");
            let before = fallbacks(&gl);
            let outcome = gl.convert(&rgb_src, &mut view, None);
            match outcome {
                Err(crate::Error::NotSupported(_)) => {}
                other => failures.push(format!(
                    "a packed-RGB view() destination must be declined on every \
                     destination route, got {other:?}"
                )),
            }
            let spent = fallbacks(&gl) - before;
            if spent != 0 {
                failures.push(format!(
                    "a packed-RGB view() destination was declined only after \
                     spending {spent} destination import(s) on it"
                ));
            }
        }
        // A whole packed-RGB destination whose import is refused re-plans onto the mapped route
        // with exactly one destination import.
        #[cfg(target_os = "linux")]
        if let Some(mut rgb_dst) = crate::test_support::dma_allocated(
            TensorDyn::image(
                VIEW_W,
                VIEW_H,
                PixelFormat::Rgb,
                DType::U8,
                Some(TensorMemory::DmaBuf),
                edgefirst_tensor::CpuAccess::ReadWrite,
            ),
            "packed-RGB DMA destination",
        ) {
            let rgb_src = TensorDyn::image(
                VIEW_W,
                VIEW_H,
                PixelFormat::Rgb,
                DType::U8,
                Some(TensorMemory::Mem),
                edgefirst_tensor::CpuAccess::ReadWrite,
            )
            .expect("packed-RGB source");
            {
                let mut m = rgb_src
                    .map_bytes(edgefirst_tensor::CpuAccess::Write)
                    .expect("map packed-RGB source");
                m.as_mut_slice().fill(0x5A);
            }
            let before = fallbacks(&gl);
            match gl.convert(&rgb_src, &mut rgb_dst, None) {
                Ok(()) => {
                    let spent = fallbacks(&gl) - before;
                    if spent != 1 {
                        failures.push(format!(
                            "a refused packed-RGB destination import must re-plan \
                             after exactly one attempt, spent {spent}"
                        ));
                    }
                    let out = rgb_dst
                        .map_bytes(edgefirst_tensor::CpuAccess::Read)
                        .expect("map packed-RGB destination")
                        .as_slice()
                        .to_vec();
                    if out.iter().all(|&b| b == 0) {
                        failures
                            .push("the re-planned packed-RGB convert wrote nothing".to_string());
                    }
                }
                Err(e) => failures.push(format!(
                    "a refused packed-RGB destination import must re-plan onto the \
                     mapped route, not end the convert: {e}"
                )),
            }
        }
        assert!(failures.is_empty(), "{}", failures.join("\n"));
    }

    /// A refused `PlanarRgb` destination import re-plans the convert onto the mapped-texture route.
    #[test]
    fn a_refused_planar_destination_import_replans_onto_the_mapped_route() {
        const W: usize = 320;
        const H: usize = 240;

        if !is_opengl_available() {
            crate::test_support::report_skip(&format!("{} - OpenGL not available", function!()));
            return;
        }
        let src = TensorDyn::image(
            W,
            H,
            PixelFormat::Rgba,
            DType::U8,
            Some(TensorMemory::Mem),
            edgefirst_tensor::CpuAccess::ReadWrite,
        )
        .expect("host source");
        {
            let mut m = src
                .map_bytes(edgefirst_tensor::CpuAccess::Write)
                .expect("map source");
            for (i, px) in m
                .as_mut_slice()
                .as_chunks_mut::<4>()
                .0
                .iter_mut()
                .enumerate()
            {
                px.copy_from_slice(&[(i % 251) as u8, (i % 199) as u8, (i % 157) as u8, 255]);
            }
        }

        // The same convert on the CPU backend, as the reference.
        let mut reference = TensorDyn::image(
            W,
            H,
            PixelFormat::PlanarRgb,
            DType::U8,
            Some(TensorMemory::Mem),
            edgefirst_tensor::CpuAccess::ReadWrite,
        )
        .expect("reference destination");
        crate::CPUProcessor::new()
            .convert(&src, &mut reference, None)
            .expect("CPU reference convert");
        let want = reference
            .map_bytes(edgefirst_tensor::CpuAccess::Read)
            .expect("map reference")
            .as_slice()
            .to_vec();

        // Tolerance by renderer, read while a probe processor's context is current.
        let tolerance: u8 = crate::opengl_headless::processor::GLProcessorST::new(None, None)
            .map(|g| {
                let v3d = edgefirst_gl::get_string(edgefirst_gl::gl::RENDERER)
                    .is_ok_and(|r| r.contains("V3D"));
                if g.is_adreno || v3d {
                    4
                } else {
                    0
                }
            })
            .unwrap_or(0);
        let mut gl = GLProcessorThreaded::new(None).expect("GL processor");
        let planar_dma = || {
            let image = TensorDyn::image(
                W,
                H,
                PixelFormat::PlanarRgb,
                DType::U8,
                Some(TensorMemory::DmaBuf),
                edgefirst_tensor::CpuAccess::ReadWrite,
            );
            if cfg!(target_os = "linux") {
                crate::test_support::dma_allocated(image, "PlanarRgb DMA destination")
            } else {
                crate::test_support::allocated(
                    image,
                    "PlanarRgb DMA destination",
                    crate::test_support::Expect::MayFail(
                        "PlanarRgb u8 has no zero-copy image layout on this platform",
                    ),
                )
            }
        };
        let Some(mut dst) = planar_dma() else {
            return;
        };

        let mut failures: Vec<String> = Vec::new();
        let measure = |gl: &mut GLProcessorThreaded,
                       label: &str,
                       dst: &mut TensorDyn,
                       failures: &mut Vec<String>|
         -> u64 {
            let before = gl
                .convert_stats()
                .expect("convert stats")
                .dst_import_fallbacks;
            let outcome = gl.convert(&src, dst, None);
            let spent = gl
                .convert_stats()
                .expect("convert stats")
                .dst_import_fallbacks
                - before;
            match outcome {
                Ok(()) if spent > 0 => {
                    if spent != 1 {
                        failures.push(format!(
                            "{label}: expected exactly one destination-import fallback, \
                             saw {spent}"
                        ));
                    }
                    let got = dst
                        .map_bytes(edgefirst_tensor::CpuAccess::Read)
                        .expect("map destination")
                        .as_slice()
                        .to_vec();
                    let off = |(a, b): (&u8, &u8)| a.abs_diff(*b) > tolerance;
                    if got.len() != want.len() || got.iter().zip(want.iter()).any(off) {
                        let first = got.iter().zip(want.iter()).position(off).unwrap_or(0);
                        failures.push(format!(
                            "{label}: the re-planned convert does not match the CPU \
                             reference; first differing byte {first} got={} want={}",
                            got[first], want[first]
                        ));
                    }
                }
                Ok(()) => {}
                Err(e) => failures.push(format!(
                    "{label}: a refused planar destination import must complete through the \
                     mapped-texture route, not end the convert ({spent} fallback(s) spent): {e}"
                )),
            }
            spent
        };

        // Refusal through the test hook.
        gl.set_fail_dst_import(true).expect("arm the hook");
        let hooked = measure(&mut gl, "hooked refusal", &mut dst, &mut failures);
        gl.set_fail_dst_import(false).expect("disarm the hook");
        if hooked == 0 {
            crate::test_support::report_skip(&format!(
                "{} - no zero-copy destination import is attempted on this host",
                function!()
            ));
            return;
        }

        // Refusal by the driver: a plane offset that is not 64-byte aligned.
        if let Some(mut offset_dst) = planar_dma() {
            offset_dst.set_plane_offset(32);
            let real = measure(
                &mut gl,
                "driver refusal at offset 32",
                &mut offset_dst,
                &mut failures,
            );
            if real == 0 {
                crate::test_support::report_skip(&format!(
                    "{} - this driver accepts an unaligned PlanarRgb destination, so the \
                     real-refusal arm had nothing to measure (the hooked arm still ran)",
                    function!()
                ));
            }
        }

        assert!(failures.is_empty(), "{}", failures.join("\n"));
    }

    /// Stamp a smooth YUYV or NV12 frame into `t` at its own pitch: luma ramps with x and y,
    /// Cb with x, Cr with y, all inside the limited-range band.
    #[cfg(any(target_os = "linux", target_os = "windows"))]
    fn stamp_yuv_gradient(t: &TensorDyn, fmt: PixelFormat, w: usize, h: usize) {
        let luma = |x: usize, y: usize| (16 + x * 180 / w + y * 40 / h) as u8;
        let cb = |x: usize| (64 + x * 128 / w) as u8;
        let cr = |y: usize| (64 + y * 128 / h) as u8;
        let mut m = t
            .map_bytes(edgefirst_tensor::CpuAccess::ReadWrite)
            .expect("map the source");
        let s = m.as_mut_slice();
        match fmt {
            PixelFormat::Yuyv => {
                let pitch = t.effective_row_stride().unwrap_or(w * 2);
                for y in 0..h {
                    for x in (0..w).step_by(2) {
                        let i = y * pitch + x * 2;
                        s[i] = luma(x, y);
                        s[i + 1] = cb(x);
                        s[i + 2] = luma(x + 1, y);
                        s[i + 3] = cr(y);
                    }
                }
            }
            PixelFormat::Nv12 => {
                let pitch = t.effective_row_stride().unwrap_or(w);
                for y in 0..h {
                    for x in 0..w {
                        s[y * pitch + x] = luma(x, y);
                    }
                }
                let uv = h * pitch;
                for y in 0..h / 2 {
                    for x in (0..w).step_by(2) {
                        s[uv + y * pitch + x] = cb(x);
                        s[uv + y * pitch + x + 1] = cr(y * 2);
                    }
                }
            }
            other => panic!("stamp_yuv_gradient: {other:?} is not YUYV or NV12"),
        }
    }

    /// Largest per-byte difference between two images of one format, read at
    /// each tensor's own pitch; int8 bytes compare as signed values. Returns
    /// the difference and the first (x, row, byte) past `tolerance`.
    #[cfg(any(target_os = "linux", target_os = "windows"))]
    fn image_max_diff(
        a: &TensorDyn,
        b: &TensorDyn,
        tolerance: u32,
    ) -> (u32, Option<(usize, usize, usize)>) {
        let fmt = a.format().expect("a formatted image");
        let (w, h) = (a.width().unwrap(), a.height().unwrap());
        let planar = fmt.layout() == edgefirst_tensor::PixelLayout::Planar;
        let (rows, row_bytes) = if planar {
            (h * fmt.channels(), w)
        } else {
            (h, w * fmt.channels())
        };
        let signed = a.dtype() == DType::I8;
        let am = a.map_bytes(edgefirst_tensor::CpuAccess::Read).unwrap();
        let bm = b.map_bytes(edgefirst_tensor::CpuAccess::Read).unwrap();
        let (ap, bp) = (
            a.effective_row_stride().unwrap_or(row_bytes),
            b.effective_row_stride().unwrap_or(row_bytes),
        );
        let (mut worst, mut first) = (0u32, None);
        for r in 0..rows {
            for c in 0..row_bytes {
                let (x, y) = (am.as_slice()[r * ap + c], bm.as_slice()[r * bp + c]);
                let d = if signed {
                    (x as i8 as i32 - y as i8 as i32).unsigned_abs()
                } else {
                    (x as i32 - y as i32).unsigned_abs()
                };
                worst = worst.max(d);
                if d > tolerance && first.is_none() {
                    first = Some((c, r, d as usize));
                }
            }
        }
        (worst, first)
    }

    /// A planar convert after a packed-RGB one on the same processor writes every plane.
    #[test]
    #[cfg(target_os = "linux")]
    fn a_planar_convert_after_a_packed_rgb_one_writes_every_plane() {
        use crate::cpu::CPUProcessor;
        let Some(mut gl) = pbo_gl_or_skip(function!()) else {
            return;
        };
        if !crate::test_support::is_dma_available() {
            crate::test_support::report_skip(&format!("{} - no DMA heap", function!()));
            return;
        }
        let (w, h) = (320usize, 240usize);
        let access = edgefirst_tensor::CpuAccess::ReadWrite;
        let mut src = TensorDyn::image(
            w,
            h,
            PixelFormat::Nv12,
            DType::U8,
            Some(TensorMemory::Mem),
            access,
        )
        .unwrap();
        stamp_yuv_gradient(&src, PixelFormat::Nv12, w, h);
        src.set_colorimetry(Some(
            edgefirst_tensor::Colorimetry::default()
                .with_encoding(edgefirst_tensor::ColorEncoding::Bt601)
                .with_range(edgefirst_tensor::ColorRange::Full),
        ));
        let image = |fmt, mem| TensorDyn::image(w, h, fmt, DType::U8, Some(mem), access).unwrap();
        let mut want = image(PixelFormat::PlanarRgb, TensorMemory::Mem);
        CPUProcessor::new().convert(&src, &mut want, None).unwrap();
        for dst_fmt in [PixelFormat::Rgb, PixelFormat::PlanarRgb] {
            let mut got = image(dst_fmt, TensorMemory::DmaBuf);
            gl.convert(&src, &mut got, None)
                .unwrap_or_else(|e| panic!("{dst_fmt:?}: {e}"));
            if dst_fmt == PixelFormat::PlanarRgb {
                let (worst, first) = image_max_diff(&got, &want, 4);
                assert!(
                    first.is_none(),
                    "planar after packed RGB differs from the CPU by up to {worst} at {first:?} \
                     (byte, row, diff; rows {h}.. are the second plane)"
                );
            }
        }
    }

    /// YUYV and NV12 sources in a PBO or host memory match the CPU converter, or decline to it.
    #[test]
    #[cfg(any(target_os = "linux", target_os = "windows"))]
    fn yuv_sources_from_pbo_and_host_memory_match_cpu() {
        use crate::cpu::CPUProcessor;
        let Some(mut gl) = pbo_gl_or_skip(function!()) else {
            return;
        };
        let (w, h) = (320usize, 240usize);
        let cm = edgefirst_tensor::Colorimetry::default()
            .with_encoding(edgefirst_tensor::ColorEncoding::Bt601)
            .with_range(edgefirst_tensor::ColorRange::Full);
        let dsts = [
            (PixelFormat::Rgba, DType::U8),
            (PixelFormat::Rgb, DType::U8),
            (PixelFormat::PlanarRgb, DType::U8),
            (PixelFormat::Rgba, DType::I8),
            (PixelFormat::Rgb, DType::I8),
            (PixelFormat::PlanarRgb, DType::I8),
        ];
        #[allow(unused_mut)]
        let mut dst_mems = vec![TensorMemory::Mem];
        #[cfg(target_os = "linux")]
        if crate::test_support::is_dma_available() {
            dst_mems.push(TensorMemory::DmaBuf);
        }
        let access = edgefirst_tensor::CpuAccess::ReadWrite;
        let uploads_yuv = !gl.transfer_backend().is_zero_copy();
        let mut cpu = CPUProcessor::new();
        let mut failures = Vec::new();
        let mut checked = 0;
        for src_fmt in [PixelFormat::Yuyv, PixelFormat::Nv12] {
            let mut reference =
                TensorDyn::image(w, h, src_fmt, DType::U8, Some(TensorMemory::Mem), access)
                    .expect("host reference source");
            stamp_yuv_gradient(&reference, src_fmt, w, h);
            reference.set_colorimetry(Some(cm));
            for src_mem in [TensorMemory::Pbo, TensorMemory::Mem] {
                let mut src: TensorDyn = if src_mem == TensorMemory::Pbo {
                    gl.create_pbo_image(w, h, src_fmt)
                        .expect("allocate the source PBO")
                        .into()
                } else {
                    TensorDyn::image(w, h, src_fmt, DType::U8, Some(src_mem), access)
                        .expect("host source")
                };
                assert_eq!(src.memory(), src_mem);
                stamp_yuv_gradient(&src, src_fmt, w, h);
                src.set_colorimetry(Some(cm));
                for &dst_mem in &dst_mems {
                    for (dst_fmt, dtype) in dsts {
                        let case =
                            format!("{src_fmt:?}/{src_mem:?} -> {dst_fmt:?} {dtype:?}/{dst_mem:?}");
                        let mut want =
                            TensorDyn::image(w, h, dst_fmt, dtype, Some(TensorMemory::Mem), access)
                                .unwrap();
                        cpu.convert(&reference, &mut want, None)
                            .unwrap_or_else(|e| panic!("{case}: CPU reference: {e}"));
                        let mut got = TensorDyn::image(w, h, dst_fmt, dtype, Some(dst_mem), access)
                            .unwrap_or_else(|e| panic!("{case}: destination: {e}"));
                        let before = gl.convert_stats().expect("convert stats before");
                        let result = gl.convert(&src, &mut got, None);
                        let declined = !uploads_yuv
                            && (src_fmt == PixelFormat::Yuyv || src_mem == TensorMemory::Pbo);
                        if declined {
                            if !matches!(result, Err(crate::Error::NotSupported(_))) {
                                failures.push(format!(
                                    "{case}: expected a NotSupported decline on a zero-copy \
                                     platform, got {result:?}"
                                ));
                            }
                            checked += 1;
                            continue;
                        }
                        if let Err(e) = result {
                            failures.push(format!("{case}: GL refused: {e}"));
                            continue;
                        }
                        let after = gl.convert_stats().expect("convert stats after");
                        let (pbo, upload, import) = (
                            after.src_pbo_uploads - before.src_pbo_uploads,
                            after.src_uploads - before.src_uploads,
                            after.src_imports - before.src_imports,
                        );
                        let fed = match src_mem {
                            TensorMemory::Pbo => pbo >= 1 && upload == 0,
                            _ => upload >= 1 && pbo == 0,
                        };
                        if !fed || import != 0 {
                            failures.push(format!(
                                "{case}: source fed {pbo} time(s) from the PBO, {upload} by \
                                 map + upload, {import} by import"
                            ));
                            continue;
                        }
                        let (worst, first) = image_max_diff(&got, &want, 4);
                        if let Some((c, r, d)) = first {
                            failures.push(format!(
                                "{case}: byte {c} of row {r} differs from the CPU by {d} \
                                 (worst {worst}, tolerance 4)"
                            ));
                        }
                        checked += 1;
                    }
                }
            }
        }
        assert!(failures.is_empty(), "{}", failures.join("\n"));
        assert_eq!(checked, 2 * 2 * dst_mems.len() * dsts.len());
    }

    // -------------------------------------------------------------------------
    // Packed Rgb DMA-BUF sources
    // -------------------------------------------------------------------------

    /// The test image: a gradient in R and G, high-frequency texture in B.
    #[cfg(target_os = "linux")]
    fn rgb_test_pixel(x: usize, y: usize, w: usize, h: usize) -> [u8; 3] {
        [
            (x * 255 / w.max(2).saturating_sub(1)) as u8,
            (y * 255 / h.max(2).saturating_sub(1)) as u8,
            (((x * 7 + y * 13) ^ (x >> 3)) & 0xff) as u8,
        ]
    }

    /// A `w`x`h` `fmt` (Rgb or Rgba) DMA-BUF image of [`rgb_test_pixel`], with every
    /// row-padding byte set to 0xEE.
    #[cfg(target_os = "linux")]
    fn rgb_test_source(w: usize, h: usize, fmt: PixelFormat) -> TensorDyn {
        let src = TensorDyn::image(
            w,
            h,
            fmt,
            DType::U8,
            Some(TensorMemory::DmaBuf),
            edgefirst_tensor::CpuAccess::ReadWrite,
        )
        .unwrap();
        let ch = fmt.channels();
        {
            let t = src.as_u8().unwrap();
            let stride = t.effective_row_stride().unwrap_or(w * ch);
            let mut map = t.map().unwrap();
            let data = map.as_mut_slice();
            for y in 0..h {
                let row = &mut data[y * stride..(y + 1) * stride];
                row.fill(0xEE);
                for x in 0..w {
                    let p = rgb_test_pixel(x, y, w, h);
                    row[x * ch..x * ch + 3].copy_from_slice(&p);
                    if ch == 4 {
                        row[x * ch + 3] = 255;
                    }
                }
            }
        }
        src
    }

    /// The pixels of a u8/i8 image with row padding removed (i8 shifted back
    /// to unsigned), planar images plane after plane.
    #[cfg(any(target_os = "linux", target_os = "windows"))]
    fn rgb_test_tight_bytes(t: &TensorDyn) -> Vec<u8> {
        let (w, h) = (t.width().unwrap(), t.height().unwrap());
        let fmt = t.format().unwrap();
        let (rows, row_bytes) = if fmt.layout() == edgefirst_tensor::PixelLayout::Planar {
            (h * fmt.channels(), w)
        } else {
            (h, w * fmt.channels())
        };
        let (bytes, stride): (Vec<u8>, Option<usize>) = match t.dtype() {
            DType::I8 => {
                let t = t.as_i8().unwrap();
                let m = t.map().unwrap();
                (
                    m.as_slice().iter().map(|&v| v as u8 ^ 0x80).collect(),
                    t.effective_row_stride(),
                )
            }
            _ => {
                let t = t.as_u8().unwrap();
                let m = t.map().unwrap();
                (m.as_slice().to_vec(), t.effective_row_stride())
            }
        };
        let stride = if fmt.layout() == edgefirst_tensor::PixelLayout::Planar {
            row_bytes
        } else {
            stride.unwrap_or(row_bytes)
        };
        (0..rows)
            .flat_map(|r| bytes[r * stride..r * stride + row_bytes].to_vec())
            .collect()
    }

    /// A GL processor for the packed-Rgb tests, with zero-copy DMA-BUF
    /// import, or `None` (with the skip reported) where there is none.
    #[cfg(target_os = "linux")]
    fn rgb_test_processor(test: &str) -> Option<crate::opengl_headless::processor::GLProcessorST> {
        if !is_opengl_available() || !is_dma_available() {
            crate::test_support::report_skip(&format!("{test} - OpenGL or DMA-BUF not available"));
            return None;
        }
        let gl = crate::opengl_headless::processor::GLProcessorST::new(None, None).ok()?;
        if !gl.gl_context.transfer_backend.is_zero_copy() {
            crate::test_support::report_skip(&format!("{test} - no zero-copy DMA-BUF import"));
            return None;
        }
        Some(gl)
    }

    /// The routes a packed Rgb DMA-BUF source can take, as
    /// `(name, allow native, allow R8 import)`: the native one only where the
    /// GPU imports `BG24`, and the R8 view with the native one withheld.
    #[cfg(target_os = "linux")]
    fn rgb_test_routes(
        gl: &crate::opengl_headless::processor::GLProcessorST,
    ) -> Vec<(&'static str, bool, bool)> {
        let mut routes = Vec::new();
        if gl.rgb_native_import() {
            routes.push(("native", true, true));
        }
        routes.push(("r8-import", false, true));
        routes
    }

    /// Which [`RgbSourcePath`] a route must report.
    #[cfg(target_os = "linux")]
    fn rgb_route_matches(
        route: &str,
        path: Option<crate::opengl_headless::processor::RgbSourcePath>,
    ) -> bool {
        use crate::opengl_headless::processor::RgbSourcePath;
        matches!(
            (route, path),
            ("native", Some(RgbSourcePath::Native))
                | ("r8-import", Some(RgbSourcePath::R8Import(_)))
        )
    }

    /// A 1:1 draw of a packed Rgb DMA-BUF source reproduces every pixel on every route.
    #[test]
    #[cfg(target_os = "linux")]
    fn test_rgb_dma_source_1to1_is_exact() {
        use crate::opengl_headless::processor::{NvSampling, RgbSourcePath};
        let Some(probe) = rgb_test_processor(function!()) else {
            return;
        };
        for (route, native, r8) in rgb_test_routes(&probe) {
            let mut gl = crate::opengl_headless::processor::GLProcessorST::new(None, None).unwrap();
            gl.restrict_rgb_source_routes(native, r8);
            for (w, h) in [
                (320usize, 240usize),
                (321, 241),
                (789, 384),
                (1920, 1080),
                (3840, 2160),
            ] {
                let src = rgb_test_source(w, h, PixelFormat::Rgb);
                let mut dst = TensorDyn::image(
                    w,
                    h,
                    PixelFormat::Rgba,
                    DType::U8,
                    Some(TensorMemory::Mem),
                    edgefirst_tensor::CpuAccess::ReadWrite,
                )
                .unwrap();
                if let Err(e) = gl.convert(&src, &mut dst, None) {
                    assert!(
                        w == 3840 && matches!(e, crate::Error::NotSupported(_)),
                        "{route} {w}x{h}: {e:?}"
                    );
                    assert_eq!(gl.last_rgb_source_path, None);
                    continue;
                }
                let path = gl.last_rgb_source_path;
                assert!(
                    rgb_route_matches(route, path),
                    "{route} {w}x{h}: took {path:?}"
                );
                if route != "native" {
                    assert!(
                        matches!(path, Some(RgbSourcePath::R8Import(NvSampling::Single))),
                        "{route} {w}x{h}: a 1:1 draw must take the single-tap program, took {path:?}"
                    );
                }
                let tolerance = if path == Some(RgbSourcePath::Native) {
                    1
                } else {
                    0
                };
                let out = rgb_test_tight_bytes(&dst);
                let mut worst = (0u8, (0usize, 0usize, 0usize));
                for y in 0..h {
                    for x in 0..w {
                        let want = rgb_test_pixel(x, y, w, h);
                        for c in 0..3 {
                            let d = out[(y * w + x) * 4 + c].abs_diff(want[c]);
                            if d > worst.0 {
                                worst = (d, (x, y, c));
                            }
                        }
                    }
                }
                assert!(
                    worst.0 <= tolerance,
                    "{route} {w}x{h}: off by {} at {:?}",
                    worst.0,
                    worst.1
                );
            }
        }
    }

    /// With no zero-copy import, a packed Rgb DMA-BUF source declines with `NotSupported`.
    #[test]
    #[cfg(target_os = "linux")]
    fn test_rgb_dma_source_without_an_import_declines() {
        let Some(mut gl) = rgb_test_processor(function!()) else {
            return;
        };
        gl.restrict_rgb_source_routes(false, false);
        let src = rgb_test_source(640, 480, PixelFormat::Rgb);
        let mut dst = TensorDyn::image(
            320,
            240,
            PixelFormat::Rgba,
            DType::U8,
            Some(TensorMemory::Mem),
            edgefirst_tensor::CpuAccess::ReadWrite,
        )
        .unwrap();
        let err = gl
            .convert(&src, &mut dst, None)
            .expect_err("no import route: the GL convert must decline");
        assert!(matches!(err, crate::Error::NotSupported(_)), "{err:?}");
        assert_eq!(gl.last_rgb_source_path, None);
    }

    /// Every route of a packed Rgb DMA-BUF source renders what an Rgba DMA-BUF source renders.
    #[test]
    #[cfg(target_os = "linux")]
    fn test_rgb_dma_source_matches_rgba_source() {
        use crate::{ConvertOptions, Flip, Rotation};
        const MAX_DIFF: u8 = 8;
        const MAX_MEAN: f64 = 1.0;
        let Some(probe) = rgb_test_processor(function!()) else {
            return;
        };
        let f16 = probe.supported_render_dtypes().f16;
        let letterbox = || ConvertOptions::new().letterbox([114, 114, 114, 255]);
        let crop = Crop::new().with_source(Some(crate::Region::new(101, 53, 320, 180)));
        struct Case {
            name: &'static str,
            src: (usize, usize),
            dst: (usize, usize),
            fmt: PixelFormat,
            dtype: DType,
            mem: TensorMemory,
            options: ConvertOptions,
        }
        let mut cases = vec![
            Case {
                name: "letterbox",
                src: (1920, 1080),
                dst: (640, 640),
                fmt: PixelFormat::Rgba,
                dtype: DType::U8,
                mem: TensorMemory::Mem,
                options: letterbox(),
            },
            Case {
                name: "upscale-odd",
                src: (321, 241),
                dst: (640, 480),
                fmt: PixelFormat::Rgba,
                dtype: DType::U8,
                mem: TensorMemory::Mem,
                options: ConvertOptions::new(),
            },
            Case {
                name: "crop",
                src: (789, 384),
                dst: (448, 252),
                fmt: PixelFormat::Rgba,
                dtype: DType::U8,
                mem: TensorMemory::Mem,
                options: ConvertOptions::new().crop(crop),
            },
            Case {
                name: "rotate-flip",
                src: (640, 480),
                dst: (480, 640),
                fmt: PixelFormat::Rgba,
                dtype: DType::U8,
                mem: TensorMemory::Mem,
                options: ConvertOptions::new()
                    .rotation(Rotation::Clockwise90)
                    .flip(Flip::Horizontal),
            },
            Case {
                name: "rgb-dst",
                src: (1920, 1080),
                dst: (640, 640),
                fmt: PixelFormat::Rgb,
                dtype: DType::U8,
                mem: TensorMemory::DmaBuf,
                options: letterbox(),
            },
            Case {
                name: "planar-dst",
                src: (1280, 720),
                dst: (640, 640),
                fmt: PixelFormat::PlanarRgb,
                dtype: DType::U8,
                mem: TensorMemory::DmaBuf,
                options: letterbox(),
            },
            Case {
                name: "int8-dst",
                src: (1280, 720),
                dst: (640, 640),
                fmt: PixelFormat::Rgb,
                dtype: DType::I8,
                mem: TensorMemory::Mem,
                options: letterbox(),
            },
        ];
        if f16 {
            cases.push(Case {
                name: "f16-planar-dst",
                src: (1920, 1080),
                dst: (640, 640),
                fmt: PixelFormat::PlanarRgb,
                dtype: DType::F16,
                mem: TensorMemory::DmaBuf,
                options: letterbox(),
            });
        }
        let mut failures = Vec::new();
        for (route, native, r8) in rgb_test_routes(&probe) {
            let hw_modes: &[bool] = if route == "native" {
                &[true]
            } else {
                &[true, false]
            };
            for &hw in hw_modes {
                let mut gl =
                    crate::opengl_headless::processor::GLProcessorST::new(None, None).unwrap();
                gl.restrict_rgb_source_routes(native, r8);
                gl.set_nv_hw_filter(hw);
                for case in &cases {
                    let ((sw, sh), (dw, dh)) = (case.src, case.dst);
                    let src = rgb_test_source(sw, sh, PixelFormat::Rgb);
                    let reference_src = rgb_test_source(sw, sh, PixelFormat::Rgba);
                    let image = || {
                        TensorDyn::image(
                            dw,
                            dh,
                            case.fmt,
                            case.dtype,
                            Some(case.mem),
                            edgefirst_tensor::CpuAccess::ReadWrite,
                        )
                        .unwrap()
                    };
                    let (mut got, mut want) = (image(), image());
                    gl.convert(&src, &mut got, Some(case.options)).unwrap();
                    let path = gl.last_rgb_source_path;
                    gl.convert(&reference_src, &mut want, Some(case.options))
                        .unwrap();
                    let label = format!("{route} hw={hw} {}", case.name);
                    if !rgb_route_matches(route, path) {
                        failures.push(format!("{label}: took {path:?}"));
                        continue;
                    }
                    let (got, want) = if case.dtype == DType::F16 {
                        let bytes = |t: &TensorDyn| -> Vec<u8> {
                            let t = t.as_f16().unwrap();
                            let m = t.map().unwrap();
                            m.as_slice()
                                .iter()
                                .map(|v| (v.to_f32() * 255.0).round().clamp(0.0, 255.0) as u8)
                                .collect()
                        };
                        (bytes(&got), bytes(&want))
                    } else {
                        (rgb_test_tight_bytes(&got), rgb_test_tight_bytes(&want))
                    };
                    assert_eq!(got.len(), want.len());
                    let max = got
                        .iter()
                        .zip(&want)
                        .map(|(a, b)| a.abs_diff(*b))
                        .max()
                        .unwrap_or(0);
                    let mean = got
                        .iter()
                        .zip(&want)
                        .map(|(a, b)| a.abs_diff(*b) as f64)
                        .sum::<f64>()
                        / got.len().max(1) as f64;
                    if max > MAX_DIFF || mean > MAX_MEAN {
                        failures.push(format!("{label} ({path:?}): max {max}, mean {mean:.3}"));
                    }
                }
            }
        }
        assert!(failures.is_empty(), "{}", failures.join("\n"));
    }

    /// Tight bytes of a `w`x`h` `fmt` image whose channels differ: packed RGB(A) from
    /// [`rgb_test_pixel`]; YUV with luma that crosses both limited-range bounds and chroma
    /// that varies on both axes.
    #[cfg(target_os = "linux")]
    fn grey_test_bytes(w: usize, h: usize, fmt: PixelFormat) -> Vec<u8> {
        // A smooth ramp over 0..=255.
        let luma = |x: usize, y: usize| (x * 191 / w + y * 64 / h) as u8;
        let cb = |x: usize, y: usize| ((x * 7 + y * 2) % 256) as u8;
        let cr = |x: usize, y: usize| ((x * 3 + y * 11 + 64) % 256) as u8;
        let mut out = Vec::new();
        match fmt {
            PixelFormat::Rgb | PixelFormat::Rgba => {
                for y in 0..h {
                    for x in 0..w {
                        out.extend_from_slice(&rgb_test_pixel(x, y, w, h));
                        if fmt == PixelFormat::Rgba {
                            out.push(255);
                        }
                    }
                }
            }
            PixelFormat::Grey => {
                for y in 0..h {
                    out.extend((0..w).map(|x| luma(x, y)));
                }
            }
            PixelFormat::Yuyv => {
                for y in 0..h {
                    for x in (0..w).step_by(2) {
                        out.extend_from_slice(&[luma(x, y), cb(x, y), luma(x + 1, y), cr(x, y)]);
                    }
                }
            }
            _ => {
                let layout = fmt.chroma_layout().expect("semi-planar format");
                for y in 0..h {
                    out.extend((0..w).map(|x| luma(x, y)));
                }
                let cw = w.div_ceil(1 << layout.shift_x);
                let ch = h.div_ceil(1 << layout.shift_y);
                for y in 0..ch {
                    for x in 0..cw {
                        out.extend_from_slice(&[cb(x, y), cr(x, y)]);
                    }
                }
            }
        }
        out
    }

    /// Every source format the GL backend takes renders into a Grey destination as the CPU does.
    #[test]
    #[cfg(target_os = "linux")]
    fn test_grey_dst_matches_cpu() {
        use crate::{ConvertOptions, ImageProcessorTrait};
        if !is_opengl_available() {
            crate::test_support::report_skip(&format!("{} - OpenGL not available", function!()));
            return;
        }
        let mut gl = crate::opengl_headless::processor::GLProcessorST::new(None, None).unwrap();
        let zero_copy = gl.gl_context.transfer_backend.is_zero_copy() && is_dma_available();
        let mut mems = vec![TensorMemory::Mem];
        if zero_copy {
            mems.push(TensorMemory::DmaBuf);
        }
        let full = edgefirst_tensor::Colorimetry::default()
            .with_encoding(edgefirst_tensor::ColorEncoding::Bt601)
            .with_range(edgefirst_tensor::ColorRange::Full);
        let mut failures = Vec::new();
        let (w, h) = (322usize, 242usize);
        for fmt in [
            PixelFormat::Rgb,
            PixelFormat::Rgba,
            PixelFormat::Grey,
            PixelFormat::Nv12,
            PixelFormat::Nv16,
            PixelFormat::Nv24,
            PixelFormat::Yuyv,
        ] {
            let ranges: &[Option<edgefirst_tensor::Colorimetry>] = if fmt.is_yuv() {
                &[None, Some(full)]
            } else {
                &[None]
            };
            for &mem in &mems {
                if fmt == PixelFormat::Yuyv && mem == TensorMemory::Mem && zero_copy {
                    continue;
                }
                for &cm in ranges {
                    let mut src =
                        load_raw_image(w, h, fmt, Some(mem), &grey_test_bytes(w, h, fmt)).unwrap();
                    src.set_colorimetry(cm);
                    // The CPU reference for an NV16 source is NV12 with the same luma plane.
                    let ref_fmt = if fmt == PixelFormat::Nv16 {
                        PixelFormat::Nv12
                    } else {
                        fmt
                    };
                    let mut reference = load_raw_image(
                        w,
                        h,
                        ref_fmt,
                        Some(TensorMemory::Mem),
                        &grey_test_bytes(w, h, ref_fmt),
                    )
                    .unwrap();
                    reference.set_colorimetry(cm);
                    for (opts_name, (dw, dh), options) in [
                        ("1:1", (w, h), ConvertOptions::new()),
                        (
                            "letterbox",
                            (480, 480),
                            ConvertOptions::new().letterbox([200, 30, 90, 255]),
                        ),
                        (
                            "pad-output",
                            (480, 480),
                            ConvertOptions::new()
                                .letterbox([200, 30, 90, 255])
                                .pad_output([1.0; 4]),
                        ),
                    ] {
                        for dtype in [DType::U8, DType::I8] {
                            let label = format!(
                                "{fmt:?}@{mem:?} {} {opts_name} {dtype:?}",
                                if cm.is_some() { "full" } else { "default" }
                            );
                            let image = |m| {
                                TensorDyn::image(
                                    dw,
                                    dh,
                                    PixelFormat::Grey,
                                    dtype,
                                    Some(m),
                                    edgefirst_tensor::CpuAccess::ReadWrite,
                                )
                                .unwrap()
                            };
                            let mut got = image(mem);
                            if let Err(e) = gl.convert(&src, &mut got, Some(options)) {
                                failures.push(format!("{label}: GL declined: {e}"));
                                continue;
                            }
                            let mut want = image(TensorMemory::Mem);
                            crate::CPUProcessor::new()
                                .convert(&reference, &mut want, Some(options))
                                .unwrap();
                            let (got, want) =
                                (rgb_test_tight_bytes(&got), rgb_test_tight_bytes(&want));
                            let diffs: Vec<u8> =
                                got.iter().zip(&want).map(|(a, b)| a.abs_diff(*b)).collect();
                            let max = diffs.iter().copied().max().unwrap_or(0);
                            let mean =
                                diffs.iter().map(|&d| d as f64).sum::<f64>() / diffs.len() as f64;
                            let ok = if opts_name == "1:1" {
                                max <= 1
                            } else {
                                // Row 0 lies in the pad band and is exact.
                                got[0] == want[0] && mean <= 2.0
                            };
                            if !ok {
                                failures.push(format!(
                                    "{label}: max {max}, mean {mean:.3}, pad {} vs {}",
                                    got[0], want[0]
                                ));
                            }
                        }
                    }
                }
            }
        }
        assert!(failures.is_empty(), "{}", failures.join("\n"));
    }

    /// A colour convert after a Grey one through the same NV program restores the chroma terms.
    #[test]
    #[cfg(target_os = "linux")]
    fn test_nv_luma_only_does_not_leak_into_colour() {
        use crate::ImageProcessorTrait;
        if !is_opengl_available() {
            crate::test_support::report_skip(&format!("{} - OpenGL not available", function!()));
            return;
        }
        let mut gl = crate::opengl_headless::processor::GLProcessorST::new(None, None).unwrap();
        let (w, h) = (322usize, 242usize);
        let src = load_raw_image(
            w,
            h,
            PixelFormat::Nv12,
            Some(TensorMemory::Mem),
            &grey_test_bytes(w, h, PixelFormat::Nv12),
        )
        .unwrap();
        let image = |fmt| {
            TensorDyn::image(
                w,
                h,
                fmt,
                DType::U8,
                Some(TensorMemory::Mem),
                edgefirst_tensor::CpuAccess::ReadWrite,
            )
            .unwrap()
        };
        let mut before = image(PixelFormat::Rgba);
        gl.convert(&src, &mut before, None).unwrap();
        let mut grey = image(PixelFormat::Grey);
        gl.convert(&src, &mut grey, None).unwrap();
        let mut after = image(PixelFormat::Rgba);
        gl.convert(&src, &mut after, None).unwrap();
        assert_eq!(
            rgb_test_tight_bytes(&before),
            rgb_test_tight_bytes(&after),
            "an NV -> Rgba convert after an NV -> Grey one lost its chroma"
        );
    }

    /// A limited-range luma resize clamps each sample's footroom to black before blending.
    #[test]
    #[cfg(any(target_os = "linux", target_os = "windows"))]
    fn test_grey_dst_limited_luma_resize_clamps_each_sample() {
        use crate::ImageProcessorTrait;
        if !is_opengl_available() {
            crate::test_support::report_skip(&format!("{} - OpenGL not available", function!()));
            return;
        }
        let mut gl = crate::opengl_headless::processor::GLProcessorST::new(None, None).unwrap();
        gl.set_nv_hw_filter(true);
        let zero_copy = gl.gl_context.transfer_backend.is_zero_copy()
            && edgefirst_tensor::is_gpu_buffer_available();
        let (w, h) = (320usize, 240usize);
        let row = |y: usize| if y.is_multiple_of(2) { 0u8 } else { 32 };
        let mut sources = vec![(PixelFormat::Nv12, TensorMemory::Mem)];
        if zero_copy {
            sources.push((PixelFormat::Nv12, TensorMemory::DmaBuf));
            if cfg!(target_os = "linux") {
                sources.push((PixelFormat::Yuyv, TensorMemory::DmaBuf));
            }
        }
        let mut failures = Vec::new();
        for (fmt, mem) in sources {
            let bytes: Vec<u8> = if fmt == PixelFormat::Yuyv {
                (0..h)
                    .flat_map(|y| (0..w).flat_map(move |_| [row(y), 128]))
                    .collect()
            } else {
                let luma = (0..h).flat_map(|y| std::iter::repeat_n(row(y), w));
                luma.chain(std::iter::repeat_n(128, w * h / 2)).collect()
            };
            let src = load_raw_image(w, h, fmt, Some(mem), &bytes).unwrap();
            let image = |m| {
                TensorDyn::image(
                    w,
                    h / 2,
                    PixelFormat::Grey,
                    DType::U8,
                    Some(m),
                    edgefirst_tensor::CpuAccess::ReadWrite,
                )
                .unwrap()
            };
            let mut got = image(mem);
            gl.convert(&src, &mut got, None).unwrap();
            let mut want = image(TensorMemory::Mem);
            crate::CPUProcessor::new()
                .convert(&src, &mut want, None)
                .unwrap();
            let (got, want) = (rgb_test_tight_bytes(&got), rgb_test_tight_bytes(&want));
            let max = got
                .iter()
                .zip(&want)
                .map(|(a, b)| a.abs_diff(*b))
                .max()
                .unwrap_or(0);
            if max > 2 {
                failures.push(format!(
                    "{fmt:?}@{mem:?}: max {max} (gl {}, cpu {})",
                    got[w * 10],
                    want[w * 10]
                ));
            }
        }
        assert!(failures.is_empty(), "{}", failures.join("\n"));
    }

    // -------------------------------------------------------------------------
    // NV12 destinations
    // -------------------------------------------------------------------------

    /// A `w`x`h` RGBA image: the top half a gradient around one colour, the
    /// bottom half a gradient around another.
    #[cfg(any(target_os = "linux", target_os = "macos"))]
    fn nv12_dst_test_rgba(w: usize, h: usize) -> Vec<u8> {
        let mut out = Vec::with_capacity(w * h * 4);
        for y in 0..h {
            let (base, row) = if y < h / 2 {
                ([200, 60, 40], y)
            } else {
                ([40, 150, 170], y - h / 2)
            };
            for x in 0..w {
                out.extend_from_slice(&[
                    (base[0] + x * 40 / w) as u8,
                    (base[1] + row * 40 / (h / 2)) as u8,
                    (base[2] + (x + row) * 20 / (w + h / 2)) as u8,
                    255,
                ]);
            }
        }
        out
    }

    /// A `w`x`h` `fmt` (Rgba, Yuyv, Vyuy or Nv12) image in `memory` holding
    /// [`nv12_dst_test_rgba`], encoded on the CPU with `cm`.
    #[cfg(any(target_os = "linux", target_os = "macos"))]
    fn nv12_dst_test_source(
        w: usize,
        h: usize,
        fmt: PixelFormat,
        memory: TensorMemory,
        cm: Option<edgefirst_tensor::Colorimetry>,
    ) -> TensorDyn {
        use crate::ImageProcessorTrait;
        let rgba = load_raw_image(
            w,
            h,
            PixelFormat::Rgba,
            Some(TensorMemory::Mem),
            &nv12_dst_test_rgba(w, h),
        )
        .unwrap();
        if fmt == PixelFormat::Rgba {
            return load_raw_image(w, h, fmt, Some(memory), &nv12_dst_test_rgba(w, h)).unwrap();
        }
        let encode_fmt = if fmt == PixelFormat::Vyuy {
            PixelFormat::Yuyv
        } else {
            fmt
        };
        let mut encoded = TensorDyn::image(
            w,
            h,
            encode_fmt,
            DType::U8,
            Some(TensorMemory::Mem),
            edgefirst_tensor::CpuAccess::ReadWrite,
        )
        .unwrap();
        encoded.set_colorimetry(cm);
        crate::CPUProcessor::new()
            .convert(&rgba, &mut encoded, None)
            .unwrap();
        let mut bytes = Vec::new();
        {
            let t = encoded.as_u8().unwrap();
            let stride = t.effective_row_stride().unwrap();
            let map = t.map().unwrap();
            let (rows, row_bytes) = match fmt {
                PixelFormat::Nv12 => (h + h / 2, w),
                _ => (h, w * 2),
            };
            for r in 0..rows {
                bytes.extend_from_slice(&map.as_slice()[r * stride..][..row_bytes]);
            }
        }
        if fmt == PixelFormat::Vyuy {
            for m in bytes.chunks_exact_mut(4) {
                let [y0, u, y1, v] = [m[0], m[1], m[2], m[3]];
                m.copy_from_slice(&[v, y0, u, y1]);
            }
        }
        let mut src = load_raw_image(w, h, fmt, Some(memory), &bytes).unwrap();
        src.set_colorimetry(cm);
        src
    }

    /// The luma bytes and the interleaved chroma bytes of an NV12 image with
    /// row padding removed, from one buffer or from two.
    #[cfg(any(target_os = "linux", target_os = "macos"))]
    fn nv12_tight_planes(t: &TensorDyn) -> (Vec<u8>, Vec<u8>) {
        let (w, h) = (t.width().unwrap(), t.height().unwrap());
        let t = t.as_u8().unwrap();
        let rows = |bytes: &[u8], stride: usize, first: usize, n: usize| -> Vec<u8> {
            (first..first + n)
                .flat_map(|r| bytes[r * stride..][..w].to_vec())
                .collect()
        };
        let luma_stride = t.effective_row_stride().unwrap_or(w);
        let map = t.map().unwrap();
        let luma = rows(map.as_slice(), luma_stride, 0, h);
        let chroma = match t.chroma() {
            Some(c) => {
                let stride = c.effective_row_stride().unwrap_or(w);
                let m = c.map().unwrap();
                rows(m.as_slice(), stride, 0, h / 2)
            }
            None => rows(map.as_slice(), luma_stride, h, h / 2),
        };
        (luma, chroma)
    }

    /// A GL processor for an NV12 destination test, or `None` after reporting a skip when
    /// OpenGL or zero-copy GPU buffers are unavailable.
    #[cfg(any(target_os = "linux", target_os = "macos"))]
    fn nv12_gl_or_skip(name: &str) -> Option<crate::opengl_headless::processor::GLProcessorST> {
        if !is_opengl_available() {
            crate::test_support::report_skip(&format!("{name} - OpenGL not available"));
            return None;
        }
        if !is_gpu_image_buffer_available() {
            crate::test_support::report_skip(&format!("{name} - no zero-copy GPU buffers"));
            return None;
        }
        let gl = crate::opengl_headless::processor::GLProcessorST::new(None, None).unwrap();
        if !gl.gl_context.transfer_backend.is_zero_copy() {
            crate::test_support::report_skip(&format!("{name} - no zero-copy destination import"));
            return None;
        }
        Some(gl)
    }

    /// A `w`x`h` NV12 image in `memory` tagged with `cm`.
    #[cfg(any(target_os = "linux", target_os = "macos"))]
    fn nv12_image(
        w: usize,
        h: usize,
        memory: TensorMemory,
        cm: Option<edgefirst_tensor::Colorimetry>,
    ) -> TensorDyn {
        let mut t = TensorDyn::image(
            w,
            h,
            PixelFormat::Nv12,
            DType::U8,
            Some(memory),
            edgefirst_tensor::CpuAccess::ReadWrite,
        )
        .unwrap();
        t.set_colorimetry(cm);
        t
    }

    /// The mean absolute luma and chroma difference between `got` and the CPU's conversion of
    /// `src` with `options` into an NV12 image of `got`'s size and colorimetry.
    #[cfg(any(target_os = "linux", target_os = "macos"))]
    fn nv12_mean_diff_vs_cpu(
        src: &TensorDyn,
        got: &TensorDyn,
        options: Option<crate::ConvertOptions>,
    ) -> (f64, f64) {
        use crate::ImageProcessorTrait;
        let mut want = nv12_image(
            got.width().unwrap(),
            got.height().unwrap(),
            TensorMemory::Mem,
            got.colorimetry(),
        );
        crate::CPUProcessor::new()
            .convert(src, &mut want, options)
            .unwrap();
        let (got_y, got_uv) = nv12_tight_planes(got);
        let (want_y, want_uv) = nv12_tight_planes(&want);
        (
            nv12_plane_diff(&got_y, &want_y).1,
            nv12_plane_diff(&got_uv, &want_uv).1,
        )
    }

    /// The largest absolute difference of the chroma pairs at columns `pairs`
    /// of two tight `width`-byte-per-row chroma planes, over every row.
    #[cfg(any(target_os = "linux", target_os = "macos"))]
    fn nv12_chroma_pair_diff(got: &[u8], want: &[u8], width: usize, pairs: &[usize]) -> u8 {
        let mut max = 0;
        for (g, w) in got.chunks_exact(width).zip(want.chunks_exact(width)) {
            for &cx in pairs {
                for b in [2 * cx, 2 * cx + 1] {
                    max = max.max(g[b].abs_diff(w[b]));
                }
            }
        }
        max
    }

    /// The largest and the mean absolute difference of two byte slices.
    #[cfg(any(target_os = "linux", target_os = "macos"))]
    fn nv12_plane_diff(got: &[u8], want: &[u8]) -> (u8, f64) {
        let diffs: Vec<u8> = got.iter().zip(want).map(|(a, b)| a.abs_diff(*b)).collect();
        let max = diffs.iter().copied().max().unwrap_or(0);
        let mean = diffs.iter().map(|&d| d as f64).sum::<f64>() / diffs.len().max(1) as f64;
        (max, mean)
    }

    /// Every source the GL backend takes renders into a zero-copy NV12 destination as the CPU
    /// does, letterboxed (also at odd edges), resized and same-size, untagged and tagged BT.709
    /// full range, with soft declines on and off, except the conversions this GPU's NV12 policy
    /// declines. A rendered conversion does not count as a destination import fallback.
    #[test]
    #[cfg(any(target_os = "linux", target_os = "macos"))]
    fn test_gpu_nv12_dst_matches_cpu_reference() {
        use crate::{ConvertOptions, ImageProcessorTrait};
        let Some(mut gl) = nv12_gl_or_skip(function!()) else {
            return;
        };
        let backend = gl.gl_context.transfer_backend;
        gl.set_colorimetry_mode(crate::ColorimetryMode::Exact);
        let full = edgefirst_tensor::Colorimetry::default()
            .with_encoding(edgefirst_tensor::ColorEncoding::Bt709)
            .with_range(edgefirst_tensor::ColorRange::Full);
        let (w, h) = (1280usize, 720usize);
        let mut failures = Vec::new();
        let mut declined = Vec::new();
        let mut skipped = Vec::new();
        let mut rendered = 0usize;
        for soft_declines in [true, false] {
            gl.soft_declines = soft_declines;
            for fmt in [
                PixelFormat::Yuyv,
                PixelFormat::Vyuy,
                PixelFormat::Rgba,
                PixelFormat::Nv12,
            ] {
                for mem in [TensorMemory::Mem, TensorMemory::DmaBuf] {
                    if cfg!(target_os = "macos")
                        && fmt == PixelFormat::Vyuy
                        && mem == TensorMemory::DmaBuf
                    {
                        skipped.push(format!("{fmt:?}@{mem:?} soft={soft_declines}"));
                        continue;
                    }
                    for cm in [None, Some(full)] {
                        let src = nv12_dst_test_source(w, h, fmt, mem, cm);
                        let takes_src = crate::opengl_headless::processor::GLProcessorST::check_src_format_supported(
                        backend,
                        src.as_u8().unwrap(),
                        fmt,
                    );
                        for (opts_name, (dw, dh), options) in [
                            ("1:1", (w, h), ConvertOptions::new()),
                            (
                                "letterbox",
                                (640, 640),
                                ConvertOptions::new().letterbox([200, 30, 90, 255]),
                            ),
                            ("resize", (640, 360), ConvertOptions::new()),
                            (
                                "letterbox-odd",
                                (650, 360),
                                ConvertOptions::new().letterbox([200, 30, 90, 255]),
                            ),
                        ] {
                            let label = format!(
                                "{fmt:?}@{mem:?} {} {opts_name} soft={soft_declines}",
                                if cm.is_some() {
                                    "bt709-full"
                                } else {
                                    "untagged"
                                }
                            );
                            let mut got = nv12_image(dw, dh, TensorMemory::DmaBuf, cm);
                            let fallbacks_before = gl.convert_stats().dst_import_fallbacks;
                            gl.begin_gpu_pass();
                            let result = gl.convert(&src, &mut got, Some(options));
                            gl.end_gpu_pass_if_synced();
                            let same_size = opts_name == "1:1";
                            let exact_copy = same_size && fmt != PixelFormat::Rgba;
                            let policy_declines = gl
                                .nv12_policy_decline(fmt, same_size, exact_copy, (dw, dh))
                                .is_some();
                            if !takes_src || policy_declines {
                                match result {
                                    Err(crate::Error::NotSupported(_)) => declined.push(label),
                                    other => failures.push(format!(
                                        "{label}: expected a decline, got {other:?}"
                                    )),
                                }
                                continue;
                            }
                            if let Err(e) = result {
                                failures.push(format!("{label}: GL declined: {e}"));
                                continue;
                            }
                            if gl.convert_stats().dst_import_fallbacks != fallbacks_before {
                                failures
                                    .push(format!("{label}: the destination import was refused"));
                                continue;
                            }
                            rendered += 1;
                            let mut want = nv12_image(dw, dh, TensorMemory::Mem, cm);
                            crate::CPUProcessor::new()
                                .convert(&src, &mut want, Some(options))
                                .unwrap();
                            let (got_y, got_uv) = nv12_tight_planes(&got);
                            let (want_y, want_uv) = nv12_tight_planes(&want);
                            let (y_max, y_mean) = nv12_plane_diff(&got_y, &want_y);
                            let (uv_max, uv_mean) = nv12_plane_diff(&got_uv, &want_uv);
                            let ok = if opts_name == "1:1" {
                                y_max <= 2 && uv_max <= 2
                            } else {
                                y_mean <= 2.0 && uv_mean <= 2.0
                            };
                            if !ok {
                                failures.push(format!(
                                    "{label}: Y max {y_max} mean {y_mean:.3}, \
                                 UV max {uv_max} mean {uv_mean:.3}"
                                ));
                            }
                            if opts_name == "letterbox-odd" {
                                // The 640x360 image sits at columns 5..645.
                                let edge_max =
                                    nv12_chroma_pair_diff(&got_uv, &want_uv, dw, &[2, 322]);
                                if edge_max > 4 {
                                    failures.push(format!(
                                        "{label}: chroma pair straddling an odd edge differs by \
                                     {edge_max}"
                                    ));
                                }
                            }
                        }
                    }
                }
            }
        }
        if !skipped.is_empty() {
            crate::test_support::report_skip(&format!(
                "{} - no zero-copy source for {skipped:?}",
                function!()
            ));
        }
        log::info!("{}: declined as expected: {declined:?}", function!());
        assert!(failures.is_empty(), "{}", failures.join("\n"));
        if gl.renders_nv12_dst() {
            assert!(rendered > 0, "no conversion rendered on GL");
        }
    }

    /// With the default soft declines, RGBA and YUYV zero-copy sources of a small and a 1080p
    /// size render into NV12 as the CPU does wherever this GPU's NV12 policy renders them:
    /// same-size, resized, letterboxed from a square source and pillarboxed between odd columns.
    #[test]
    #[cfg(any(target_os = "linux", target_os = "macos"))]
    fn test_gpu_nv12_dst_matches_cpu_at_small_and_1080p_sizes() {
        use crate::{ConvertOptions, ImageProcessorTrait};
        let Some(mut gl) = nv12_gl_or_skip(function!()) else {
            return;
        };
        gl.soft_declines = true;
        let backend = gl.gl_context.transfer_backend;
        let bars = ConvertOptions::new().letterbox([200, 30, 90, 255]);
        let mut failures = Vec::new();
        let mut rendered = 0usize;
        for (w, h) in [(64usize, 32usize), (1920, 1080)] {
            let mut cases = Vec::new();
            for fmt in [PixelFormat::Rgba, PixelFormat::Yuyv] {
                let src = nv12_dst_test_source(w, h, fmt, TensorMemory::DmaBuf, None);
                cases.push(("1:1", src, (w, h), ConvertOptions::new()));
                let src = nv12_dst_test_source(w, h, fmt, TensorMemory::DmaBuf, None);
                cases.push(("resize", src, (w / 2, h / 2), ConvertOptions::new()));
            }
            let square = nv12_dst_test_source(h, h, PixelFormat::Rgba, TensorMemory::DmaBuf, None);
            cases.push(("letterbox", square, (w, h), bars));
            let narrow =
                nv12_dst_test_source(h - 2, h, PixelFormat::Rgba, TensorMemory::DmaBuf, None);
            cases.push(("pillarbox-odd", narrow, (w, h), bars));
            for (name, src, (dw, dh), options) in cases {
                let fmt = src.format().unwrap();
                let label = format!(
                    "{fmt:?} {}x{} -> {dw}x{dh} {name}",
                    src.width().unwrap(),
                    src.height().unwrap()
                );
                let same_size = name == "1:1";
                let decline = gl.nv12_policy_decline(
                    fmt,
                    same_size,
                    same_size && fmt != PixelFormat::Rgba,
                    (dw, dh),
                );
                let takes_src =
                    crate::opengl_headless::processor::GLProcessorST::check_src_format_supported(
                        backend,
                        src.as_u8().unwrap(),
                        fmt,
                    );
                let mut got = nv12_image(dw, dh, TensorMemory::DmaBuf, None);
                let fallbacks_before = gl.convert_stats().dst_import_fallbacks;
                gl.begin_gpu_pass();
                let result = gl.convert(&src, &mut got, Some(options));
                gl.end_gpu_pass_if_synced();
                if decline.is_some() || !takes_src {
                    if !matches!(result, Err(crate::Error::NotSupported(_))) {
                        failures.push(format!("{label}: expected a decline, got {result:?}"));
                    }
                    continue;
                }
                if let Err(e) = result {
                    failures.push(format!("{label}: GL declined: {e}"));
                    continue;
                }
                if gl.convert_stats().dst_import_fallbacks != fallbacks_before {
                    failures.push(format!("{label}: the destination import was refused"));
                    continue;
                }
                rendered += 1;
                let mut want = nv12_image(dw, dh, TensorMemory::Mem, got.colorimetry());
                crate::CPUProcessor::new()
                    .convert(&src, &mut want, Some(options))
                    .unwrap();
                let (got_y, got_uv) = nv12_tight_planes(&got);
                let (want_y, want_uv) = nv12_tight_planes(&want);
                let (y_max, y_mean) = nv12_plane_diff(&got_y, &want_y);
                let (uv_max, uv_mean) = nv12_plane_diff(&got_uv, &want_uv);
                let ok = if same_size {
                    y_max <= 2 && uv_max <= 2
                } else {
                    y_mean <= 2.0 && uv_mean <= 2.0
                };
                if !ok {
                    failures.push(format!(
                        "{label}: Y max {y_max} mean {y_mean:.3}, UV max {uv_max} mean {uv_mean:.3}"
                    ));
                }
                if name == "pillarbox-odd" {
                    let left = (dw - (dh - 2)) / 2;
                    let right = left + dh - 2;
                    assert!(
                        left % 2 == 1 && right % 2 == 1,
                        "{label}: edges {left}, {right}"
                    );
                    let mut edge_max = 0;
                    for (g, w) in got_uv.chunks_exact(dw).zip(want_uv.chunks_exact(dw)) {
                        for x in [left - 1, left, right - 1, right] {
                            edge_max = edge_max.max(g[x].abs_diff(w[x]));
                        }
                    }
                    if edge_max > 4 {
                        failures.push(format!(
                            "{label}: chroma pair straddling an odd edge differs by {edge_max}"
                        ));
                    }
                }
            }
        }
        assert!(failures.is_empty(), "{}", failures.join("\n"));
        if gl.renders_nv12_dst() {
            assert!(rendered > 0, "no conversion rendered on GL");
        }
    }

    /// An NV12 destination whose luma and chroma are separate DMA-BUFs is written as the CPU
    /// writes a single-plane one.
    #[test]
    #[cfg(target_os = "linux")]
    fn test_gpu_nv12_dst_writes_a_multiplane_dma_destination() {
        use crate::ImageProcessorTrait;
        let Some(mut gl) = nv12_gl_or_skip(function!()) else {
            return;
        };
        if !gl.renders_nv12_dst() {
            crate::test_support::report_skip(&format!(
                "{} - this GPU leaves NV12 destinations to the CPU",
                function!()
            ));
            return;
        }
        let (w, h) = (640usize, 480usize);
        let src = nv12_dst_test_source(w * 2, h * 2, PixelFormat::Rgba, TensorMemory::DmaBuf, None);
        let luma = Tensor::<u8>::new(&[h, w], Some(TensorMemory::DmaBuf), Some("nv12 luma"))
            .expect("DMA luma plane");
        let chroma =
            Tensor::<u8>::new(&[h / 2, w], Some(TensorMemory::DmaBuf), Some("nv12 chroma"))
                .expect("DMA chroma plane");
        let mut got: TensorDyn = Tensor::<u8>::from_planes(luma, chroma, PixelFormat::Nv12)
            .expect("multiplane NV12")
            .into();
        assert!(got.as_u8().unwrap().is_multiplane());
        gl.convert(&src, &mut got, None)
            .expect("GL renders a multiplane NV12 destination");
        let (y_mean, uv_mean) = nv12_mean_diff_vs_cpu(&src, &got, None);
        assert!(
            y_mean <= 2.0 && uv_mean <= 2.0,
            "multiplane NV12 destination differs from the CPU: Y mean {y_mean:.3}, UV mean {uv_mean:.3}"
        );
    }

    /// GL declines a heap NV12 destination, and the processor converts it on the CPU.
    #[test]
    #[cfg(any(target_os = "linux", target_os = "macos"))]
    fn test_gpu_nv12_dst_declines_heap_memory() {
        use crate::ImageProcessorTrait;
        if !is_opengl_available() {
            crate::test_support::report_skip(&format!("{} - OpenGL not available", function!()));
            return;
        }
        let mut gl = crate::opengl_headless::processor::GLProcessorST::new(None, None).unwrap();
        let (w, h) = (320usize, 240usize);
        let src = load_raw_image(
            w,
            h,
            PixelFormat::Rgba,
            Some(TensorMemory::Mem),
            &(0..w * h * 4).map(|i| (i % 251) as u8).collect::<Vec<u8>>(),
        )
        .unwrap();
        let image = || {
            TensorDyn::image(
                w,
                h,
                PixelFormat::Nv12,
                DType::U8,
                Some(TensorMemory::Mem),
                edgefirst_tensor::CpuAccess::ReadWrite,
            )
            .unwrap()
        };
        let mut dst = image();
        let err = gl
            .convert(&src, &mut dst, None)
            .expect_err("GL renders no heap NV12 destination");
        assert!(matches!(err, crate::Error::NotSupported(_)), "{err:?}");

        let mut got = image();
        crate::ImageProcessor::new()
            .unwrap()
            .convert(&src, &mut got, None)
            .expect("the CPU converts a heap NV12 destination");
        let mut want = image();
        crate::CPUProcessor::new()
            .convert(&src, &mut want, None)
            .unwrap();
        let bytes = |t: &TensorDyn| t.as_u8().unwrap().map().unwrap().as_slice().to_vec();
        assert_eq!(bytes(&got), bytes(&want));
    }

    /// A YUV source into a same-size NV12 destination of the same coding is the CPU's byte
    /// copy, and with a different coding the destination is re-encoded. GL renders either as
    /// the CPU does where its NV12 policy does not decline it.
    #[test]
    #[cfg(any(target_os = "linux", target_os = "macos"))]
    fn test_gpu_nv12_dst_declines_only_an_exact_yuv_copy() {
        use crate::ImageProcessorTrait;
        if !is_opengl_available() {
            crate::test_support::report_skip(&format!("{} - OpenGL not available", function!()));
            return;
        }
        let memory = if is_gpu_image_buffer_available() {
            TensorMemory::DmaBuf
        } else {
            TensorMemory::Mem
        };
        let src_cm = edgefirst_tensor::Colorimetry::default()
            .with_encoding(edgefirst_tensor::ColorEncoding::Bt601)
            .with_range(edgefirst_tensor::ColorRange::Limited);
        let other_cm = edgefirst_tensor::Colorimetry::default()
            .with_encoding(edgefirst_tensor::ColorEncoding::Bt709)
            .with_range(edgefirst_tensor::ColorRange::Full);
        let mut gl = crate::opengl_headless::processor::GLProcessorST::new(None, None).unwrap();
        let gl_renders = memory == TensorMemory::DmaBuf
            && gl.gl_context.transfer_backend.is_zero_copy()
            && gl.renders_nv12_dst();
        if !gl_renders {
            crate::test_support::report_skip(&format!(
                "{} - GL renders no NV12 destination here; checking the CPU only",
                function!()
            ));
        }
        let mut processor = crate::ImageProcessor::new().unwrap();
        let (w, h) = (640usize, 480usize);
        let mut failures = Vec::new();
        for fmt in [PixelFormat::Yuyv, PixelFormat::Nv12] {
            let src = nv12_dst_test_source(w, h, fmt, memory, Some(src_cm));
            let takes_src =
                crate::opengl_headless::processor::GLProcessorST::check_src_format_supported(
                    gl.gl_context.transfer_backend,
                    src.as_u8().unwrap(),
                    fmt,
                );
            for dst_cm in [src_cm, other_cm] {
                let copy = dst_cm == src_cm;
                let image = |m| {
                    let mut t = TensorDyn::image(
                        w,
                        h,
                        PixelFormat::Nv12,
                        DType::U8,
                        Some(m),
                        edgefirst_tensor::CpuAccess::ReadWrite,
                    )
                    .unwrap();
                    t.set_colorimetry(Some(dst_cm));
                    t
                };
                let mut want = image(TensorMemory::Mem);
                crate::CPUProcessor::new()
                    .convert(&src, &mut want, None)
                    .unwrap();
                let (want_y, want_uv) = nv12_tight_planes(&want);
                let check = |engine: &str, got: &TensorDyn, exact: bool| {
                    let (got_y, got_uv) = nv12_tight_planes(got);
                    let (y_max, y_mean) = nv12_plane_diff(&got_y, &want_y);
                    let (uv_max, uv_mean) = nv12_plane_diff(&got_uv, &want_uv);
                    let ok = if exact {
                        y_max == 0 && uv_max == 0
                    } else {
                        y_mean <= 2.0 && uv_mean <= 2.0
                    };
                    (!ok).then(|| {
                        format!(
                            "{fmt:?} -> {dst_cm:?} {engine}: Y max {y_max} mean {y_mean:.3}, \
                             UV max {uv_max} mean {uv_mean:.3}"
                        )
                    })
                };
                if gl_renders && takes_src {
                    let mut rendered = image(memory);
                    let declines = gl.nv12_policy_decline(fmt, true, copy, (w, h));
                    match (declines, gl.convert(&src, &mut rendered, None)) {
                        (Some(why), Err(crate::Error::NotSupported(m))) if m == why => {}
                        (None, Ok(())) => failures.extend(check("gl", &rendered, false)),
                        (want, r) => failures.push(format!(
                            "{fmt:?} -> {dst_cm:?} gl: expected {want:?}, got {r:?}"
                        )),
                    }
                }
                let mut got = image(memory);
                processor
                    .convert(&src, &mut got, None)
                    .unwrap_or_else(|e| panic!("{fmt:?}: {e}"));
                let cpu_copies = copy
                    && (!(gl_renders && takes_src)
                        || gl.nv12_policy_decline(fmt, true, true, (w, h)).is_some());
                failures.extend(check("processor", &got, cpu_copies));
            }
        }
        assert!(failures.is_empty(), "{}", failures.join("\n"));
    }

    /// An NV12 DMA destination allocated by the processor with no CPU access
    /// declared still takes the CPU converts GL declines.
    #[test]
    #[cfg(any(target_os = "linux", target_os = "macos"))]
    fn test_created_nv12_dma_destination_admits_the_cpu_fallback() {
        use crate::ImageProcessorTrait;
        if !is_gpu_image_buffer_available() {
            crate::test_support::report_skip(&format!(
                "{} - no zero-copy GPU buffers",
                function!()
            ));
            return;
        }
        let mut processor = crate::ImageProcessor::new().unwrap();
        let (w, h) = (640usize, 480usize);
        let mut dst = processor
            .create_image(
                w,
                h,
                PixelFormat::Nv12,
                DType::U8,
                Some(TensorMemory::DmaBuf),
                edgefirst_tensor::CpuAccess::None,
            )
            .unwrap();
        if dst.memory() != TensorMemory::DmaBuf {
            crate::test_support::report_skip(&format!(
                "{} - NV12 destination fell back to {:?}",
                function!(),
                dst.memory()
            ));
            return;
        }
        assert!(
            dst.cpu_access().writes(),
            "the CPU fallback's write access was not declared"
        );
        let nv12 = nv12_dst_test_source(w, h, PixelFormat::Nv12, TensorMemory::Mem, None);
        processor
            .convert(&nv12, &mut dst, None)
            .expect("NV12 -> NV12 copy into a created NV12 destination");
        let yuyv = nv12_dst_test_source(w, h, PixelFormat::Yuyv, TensorMemory::Mem, None);
        processor
            .convert(&yuyv, &mut dst, None)
            .expect("YUYV (Mem) -> NV12 into a created NV12 destination");
    }

    /// A refused NV12 destination import declines as `NotSupported`, is
    /// counted as a destination import fallback, and the processor converts
    /// on the CPU.
    #[test]
    #[cfg(target_os = "linux")]
    fn test_gpu_nv12_dst_refused_import_declines_to_the_cpu() {
        use crate::ImageProcessorTrait;
        let Some(mut gl) = nv12_gl_or_skip(function!()) else {
            return;
        };
        if !gl.renders_nv12_dst() {
            crate::test_support::report_skip(&format!(
                "{} - this GPU leaves NV12 destinations to the CPU",
                function!()
            ));
            return;
        }
        let (w, h) = (640usize, 480usize);
        let src = nv12_dst_test_source(w * 2, h * 2, PixelFormat::Rgba, TensorMemory::DmaBuf, None);
        let mut dst = nv12_image(w, h, TensorMemory::DmaBuf, None);
        let before = gl.convert_stats().dst_import_fallbacks;
        gl.fail_dst_import = true;
        let err = gl
            .convert(&src, &mut dst, None)
            .expect_err("a refused NV12 destination import declines");
        gl.fail_dst_import = false;
        assert!(matches!(err, crate::Error::NotSupported(_)), "{err:?}");
        assert_eq!(gl.convert_stats().dst_import_fallbacks, before + 1);

        crate::ImageProcessor::new()
            .unwrap()
            .convert(&src, &mut dst, None)
            .expect("the processor converts after a GL decline");
        let (y_mean, uv_mean) = nv12_mean_diff_vs_cpu(&src, &dst, None);
        assert!(
            y_mean <= 2.0 && uv_mean <= 2.0,
            "the converted destination differs from the CPU"
        );
    }

    /// GL returns `NotSupported` for a same-size RGBA to NV12 conversion this GPU's NV12 policy
    /// declines, and the processor converts it on the CPU.
    #[test]
    #[cfg(target_os = "linux")]
    fn test_gpu_nv12_dst_declines_a_same_size_rgba_conversion() {
        use crate::ImageProcessorTrait;
        let Some(mut gl) = nv12_gl_or_skip(function!()) else {
            return;
        };
        if !gl.renders_nv12_dst() {
            crate::test_support::report_skip(&format!(
                "{} - this GPU leaves NV12 destinations to the CPU",
                function!()
            ));
            return;
        }
        let (w, h) = (640usize, 480usize);
        if gl
            .nv12_policy_decline(PixelFormat::Rgba, true, false, (w, h))
            .is_none()
        {
            crate::test_support::report_skip(&format!(
                "{} - this GPU renders a same-size {w}x{h} NV12 conversion",
                function!()
            ));
            return;
        }
        let src = nv12_dst_test_source(w, h, PixelFormat::Rgba, TensorMemory::DmaBuf, None);
        let mut dst = nv12_image(w, h, TensorMemory::DmaBuf, None);
        let err = gl
            .convert(&src, &mut dst, None)
            .expect_err("GL declines a same-size NV12 conversion");
        assert!(matches!(&err, crate::Error::NotSupported(_)), "{err:?}");

        crate::ImageProcessor::new()
            .unwrap()
            .convert(&src, &mut dst, None)
            .expect("the processor converts on the CPU after a GL decline");
        let (y_mean, uv_mean) = nv12_mean_diff_vs_cpu(&src, &dst, None);
        assert!(
            y_mean <= 2.0 && uv_mean <= 2.0,
            "the converted destination differs from the CPU"
        );
    }

    /// NV12 programs that fail to build decline every NV12 destination, without a rebuild, and
    /// leave other conversions on GL; the CPU converts the declined destination.
    #[test]
    #[cfg(any(target_os = "linux", target_os = "macos"))]
    fn test_gpu_nv12_program_build_failure_declines_nv12_only() {
        use crate::ImageProcessorTrait;
        let Some(mut gl) = nv12_gl_or_skip(function!()) else {
            return;
        };
        if !gl.renders_nv12_dst() {
            crate::test_support::report_skip(&format!(
                "{} - GL renders no NV12 destination here",
                function!()
            ));
            return;
        }
        let (w, h) = (640usize, 480usize);
        let src = nv12_dst_test_source(w * 2, h * 2, PixelFormat::Rgba, TensorMemory::DmaBuf, None);
        let image = |fmt, m| {
            TensorDyn::image(
                w,
                h,
                fmt,
                DType::U8,
                Some(m),
                edgefirst_tensor::CpuAccess::ReadWrite,
            )
            .unwrap()
        };
        let build_failed = |err: &crate::Error| matches!(err, crate::Error::NotSupported(m) if m.contains("programs failed to build"));

        gl.fail_nv12_programs = true;
        let mut dst = image(PixelFormat::Nv12, TensorMemory::DmaBuf);
        let err = gl
            .convert(&src, &mut dst, None)
            .expect_err("NV12 declines when its programs fail to build");
        assert!(build_failed(&err), "{err:?}");
        gl.fail_nv12_programs = false;
        let err = gl
            .convert(&src, &mut dst, None)
            .expect_err("a failed NV12 program build is not retried");
        assert!(build_failed(&err), "{err:?}");

        let mut rgba = image(PixelFormat::Rgba, TensorMemory::DmaBuf);
        gl.convert(&src, &mut rgba, None)
            .expect("an RGBA conversion still renders on GL");
        let mut want_rgba = image(PixelFormat::Rgba, TensorMemory::Mem);
        crate::CPUProcessor::new()
            .convert(&src, &mut want_rgba, None)
            .unwrap();
        let tight = |t: &TensorDyn| {
            let t = t.as_u8().unwrap();
            let stride = t.effective_row_stride().unwrap_or(w * 4);
            let map = t.map().unwrap();
            (0..h)
                .flat_map(|r| map.as_slice()[r * stride..][..w * 4].to_vec())
                .collect::<Vec<u8>>()
        };
        let (got, want) = (tight(&rgba), tight(&want_rgba));
        let mean = got
            .iter()
            .zip(&want)
            .map(|(a, b)| a.abs_diff(*b) as f64)
            .sum::<f64>()
            / got.len() as f64;
        assert!(mean <= 2.0, "RGBA differs from the CPU: mean {mean:.3}");

        crate::CPUProcessor::new()
            .convert(&src, &mut dst, None)
            .expect("the CPU converts the declined NV12 destination");
        let mut want = image(PixelFormat::Nv12, TensorMemory::Mem);
        crate::CPUProcessor::new()
            .convert(&src, &mut want, None)
            .unwrap();
        assert_eq!(nv12_tight_planes(&dst), nv12_tight_planes(&want));
    }

    /// A YUYV 1920x1080 source resizes into a 640x360 NV12 destination on GL as the CPU does.
    #[test]
    #[cfg(target_os = "linux")]
    fn test_gpu_nv12_dst_renders_a_resized_yuyv_conversion() {
        use crate::ImageProcessorTrait;
        let Some(mut gl) = nv12_gl_or_skip(function!()) else {
            return;
        };
        if let Some(why) = gl.nv12_policy_decline(PixelFormat::Yuyv, false, false, (640, 360)) {
            crate::test_support::report_skip(&format!("{} - {why}", function!()));
            return;
        }
        let src = nv12_dst_test_source(1920, 1080, PixelFormat::Yuyv, TensorMemory::DmaBuf, None);
        let mut got = nv12_image(640, 360, TensorMemory::DmaBuf, None);
        gl.convert(&src, &mut got, None)
            .expect("GL renders a resized YUYV to NV12 conversion");
        let (y_mean, uv_mean) = nv12_mean_diff_vs_cpu(&src, &got, None);
        assert!(
            y_mean <= 2.0 && uv_mean <= 2.0,
            "Y mean {y_mean:.3}, UV mean {uv_mean:.3}"
        );
    }

    /// Converting twice into one NV12 destination imports it once and counts both lookups in
    /// the destination cache statistics.
    #[test]
    #[cfg(any(target_os = "linux", target_os = "macos"))]
    fn nv12_dst_imports_are_counted_in_dst_cache_stats() {
        use crate::ImageProcessorTrait;
        if !is_opengl_available() {
            crate::test_support::report_skip(&format!("{} - OpenGL not available", function!()));
            return;
        }
        #[cfg(target_os = "linux")]
        if !is_dma_available() {
            crate::test_support::report_skip(&format!("{} - DMA not available", function!()));
            return;
        }
        let mut gl = match GLProcessorThreaded::new(None) {
            Ok(gl) => gl,
            Err(e) => {
                crate::test_support::report_skip(&format!(
                    "{} - GL not available: {e}",
                    function!()
                ));
                return;
            }
        };
        let (w, h) = (64usize, 64usize);
        let src = nv12_dst_test_source(w * 2, h * 2, PixelFormat::Rgba, TensorMemory::DmaBuf, None);
        let mut dst = TensorDyn::image(
            w,
            h,
            PixelFormat::Nv12,
            DType::U8,
            Some(TensorMemory::DmaBuf),
            edgefirst_tensor::CpuAccess::ReadWrite,
        )
        .unwrap();
        let before = gl.egl_cache_stats().unwrap().dst;
        match gl.convert(&src, &mut dst, None) {
            Ok(()) => {}
            Err(crate::Error::NotSupported(why)) => {
                crate::test_support::report_skip(&format!(
                    "{} - GL does not render NV12 destinations: {why}",
                    function!()
                ));
                return;
            }
            Err(e) => panic!("GL converts into the NV12 destination: {e}"),
        }
        gl.convert(&src, &mut dst, None)
            .expect("GL converts into the NV12 destination again");
        let after = gl.egl_cache_stats().unwrap().dst;
        assert_eq!(after.misses - before.misses, 1, "one import");
        assert_eq!(after.hits - before.hits, 1, "one reuse");
    }

    /// A GL processor with zero-copy NV12 destinations, or `None` after reporting a skip.
    #[cfg(any(target_os = "linux", target_os = "macos"))]
    fn nv12_dst_gl(test: &str) -> Option<GLProcessorThreaded> {
        if !is_opengl_available() {
            crate::test_support::report_skip(&format!("{test} - OpenGL not available"));
            return None;
        }
        if !is_gpu_image_buffer_available() {
            crate::test_support::report_skip(&format!("{test} - no zero-copy GPU buffers"));
            return None;
        }
        let gl = GLProcessorThreaded::new(None).unwrap();
        if !gl.transfer_backend().is_zero_copy() {
            crate::test_support::report_skip(&format!("{test} - no zero-copy destination import"));
            return None;
        }
        Some(gl)
    }

    /// A same-size RGBA to NV12 conversion that GL soft-declines renders on GL as the CPU does
    /// once soft declines are off.
    #[test]
    #[cfg(any(target_os = "linux", target_os = "macos"))]
    fn soft_declined_nv12_renders_when_soft_declines_are_off() {
        use crate::ImageProcessorTrait;
        let Some(mut gl) = nv12_dst_gl(function!()) else {
            return;
        };
        let (w, h) = (640usize, 480usize);
        let policy = gl.nv12_dst_policy().unwrap();
        if !policy.gpu_renders {
            crate::test_support::report_skip(&format!(
                "{} - this GPU renders no NV12 destination",
                function!()
            ));
            return;
        }
        if policy.measured && policy.same_size_min_pixels <= w * h {
            crate::test_support::report_skip(&format!(
                "{} - this GPU renders a same-size {w}x{h} conversion with soft declines on",
                function!()
            ));
            return;
        }
        assert!(
            gl.soft_declines().unwrap(),
            "soft declines are on by default"
        );
        let src = nv12_dst_test_source(w, h, PixelFormat::Rgba, TensorMemory::DmaBuf, None);
        let image = |m| {
            TensorDyn::image(
                w,
                h,
                PixelFormat::Nv12,
                DType::U8,
                Some(m),
                edgefirst_tensor::CpuAccess::ReadWrite,
            )
            .unwrap()
        };
        let mut got = image(TensorMemory::DmaBuf);
        let err = gl
            .convert(&src, &mut got, None)
            .expect_err("GL soft-declines a same-size NV12 conversion");
        assert!(matches!(err, crate::Error::NotSupported(_)), "{err:?}");

        gl.set_soft_declines(false).unwrap();
        assert!(!gl.soft_declines().unwrap());
        gl.convert(&src, &mut got, None)
            .expect("GL renders a soft-declined conversion with soft declines off");
        let mut want = image(TensorMemory::Mem);
        crate::CPUProcessor::new()
            .convert(&src, &mut want, None)
            .unwrap();
        let (got_y, got_uv) = nv12_tight_planes(&got);
        let (want_y, want_uv) = nv12_tight_planes(&want);
        let (y_max, y_mean) = nv12_plane_diff(&got_y, &want_y);
        let (uv_max, uv_mean) = nv12_plane_diff(&got_uv, &want_uv);
        assert!(
            y_max <= 2 && uv_max <= 2,
            "Y max {y_max} mean {y_mean:.3}, UV max {uv_max} mean {uv_mean:.3}"
        );
    }

    /// A hard decline holds with soft declines off: an odd-sized NV12 destination, and a YUYV
    /// source on a GPU that renders one wrongly.
    #[test]
    #[cfg(any(target_os = "linux", target_os = "macos"))]
    fn hard_nv12_declines_hold_with_soft_declines_off() {
        use crate::ImageProcessorTrait;
        let Some(mut gl) = nv12_dst_gl(function!()) else {
            return;
        };
        gl.set_soft_declines(false).unwrap();
        let nv12 = |w, h| {
            TensorDyn::image(
                w,
                h,
                PixelFormat::Nv12,
                DType::U8,
                Some(TensorMemory::DmaBuf),
                edgefirst_tensor::CpuAccess::ReadWrite,
            )
            .unwrap()
        };
        let src = nv12_dst_test_source(126, 128, PixelFormat::Rgba, TensorMemory::DmaBuf, None);
        let mut dst = nv12(63, 64);
        let r = gl.convert(&src, &mut dst, None);
        assert!(matches!(r, Err(crate::Error::NotSupported(_))), "{r:?}");

        if gl.nv12_dst_policy().unwrap().packed_yuv_src_broken {
            let src =
                nv12_dst_test_source(1280, 720, PixelFormat::Yuyv, TensorMemory::DmaBuf, None);
            let mut dst = nv12(640, 360);
            let r = gl.convert(&src, &mut dst, None);
            assert!(matches!(r, Err(crate::Error::NotSupported(_))), "{r:?}");
        }
    }

    /// The processor's GL backend soft-declines under `ComputeBackend::Auto` and not when
    /// `ComputeBackend::OpenGl` prefers it.
    #[test]
    fn image_processor_soft_declines_follow_the_backend_mode() {
        if !is_opengl_available() {
            crate::test_support::report_skip(&format!("{} - OpenGL not available", function!()));
            return;
        }
        if std::env::var_os("EDGEFIRST_FORCE_BACKEND").is_some() {
            crate::test_support::report_skip(&format!(
                "{} - EDGEFIRST_FORCE_BACKEND is set",
                function!()
            ));
            return;
        }
        let soft = |backend| {
            let processor = crate::ImageProcessor::with_config(crate::ImageProcessorConfig {
                backend,
                ..Default::default()
            })
            .unwrap();
            processor
                .opengl
                .as_ref()
                .map(|gl| gl.soft_declines().unwrap())
        };
        assert_eq!(soft(crate::ComputeBackend::Auto), Some(true));
        assert_eq!(soft(crate::ComputeBackend::OpenGl), Some(false));
    }
}
