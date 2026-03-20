// SPDX-FileCopyrightText: Copyright 2025 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

#[cfg(test)]
#[cfg(feature = "opengl")]
mod gl_tests {
    #[cfg(feature = "dma_test_formats")]
    use crate::opengl_headless::processor::GLProcessorST;
    use crate::{
        probe_egl_displays, Crop, EglDisplayKind, Flip, GLProcessorThreaded, ImageProcessorTrait,
        Rotation,
    };
    use edgefirst_decoder::DetectBox;
    #[cfg(feature = "dma_test_formats")]
    use edgefirst_tensor::{is_dma_available, Tensor, TensorMemory};
    use edgefirst_tensor::{DType, PixelFormat, TensorDyn, TensorMapTrait, TensorTrait};
    use image::buffer::ConvertBuffer;
    use ndarray::Array3;

    #[test]
    fn test_segmentation() {
        use edgefirst_decoder::Segmentation;

        if !is_opengl_available() {
            eprintln!("SKIPPED: {} - OpenGL not available", function!());
            return;
        }

        let image = crate::load_image(
            include_bytes!(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/../../testdata/giraffe.jpg"
            )),
            Some(PixelFormat::Rgba),
            None,
        )
        .unwrap();

        let mut segmentation = Array3::from_shape_vec(
            (2, 160, 160),
            include_bytes!(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/../../testdata/modelpack_seg_2x160x160.bin"
            ))
            .to_vec(),
        )
        .unwrap();
        segmentation.swap_axes(0, 1);
        segmentation.swap_axes(1, 2);
        let segmentation = segmentation.as_standard_layout().to_owned();

        let seg = Segmentation {
            segmentation,
            xmin: 0.0,
            ymin: 0.0,
            xmax: 1.0,
            ymax: 1.0,
        };

        let mut renderer = GLProcessorThreaded::new(None).unwrap();
        let mut image_dyn = image;
        renderer.draw_masks(&mut image_dyn, &[], &[seg]).unwrap();
    }

    #[test]
    fn test_segmentation_mem() {
        use edgefirst_decoder::Segmentation;

        if !is_opengl_available() {
            eprintln!("SKIPPED: {} - OpenGL not available", function!());
            return;
        }

        let image = crate::load_image(
            include_bytes!(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/../../testdata/giraffe.jpg"
            )),
            Some(PixelFormat::Rgba),
            Some(edgefirst_tensor::TensorMemory::Mem),
        )
        .unwrap();

        let mut segmentation = Array3::from_shape_vec(
            (2, 160, 160),
            include_bytes!(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/../../testdata/modelpack_seg_2x160x160.bin"
            ))
            .to_vec(),
        )
        .unwrap();
        segmentation.swap_axes(0, 1);
        segmentation.swap_axes(1, 2);
        let segmentation = segmentation.as_standard_layout().to_owned();

        let seg = Segmentation {
            segmentation,
            xmin: 0.0,
            ymin: 0.0,
            xmax: 1.0,
            ymax: 1.0,
        };

        let mut renderer = GLProcessorThreaded::new(None).unwrap();
        let mut image_dyn = image;
        renderer.draw_masks(&mut image_dyn, &[], &[seg]).unwrap();
    }

    #[test]
    fn test_segmentation_yolo() {
        use edgefirst_decoder::Segmentation;
        use ndarray::Array3;

        if !is_opengl_available() {
            eprintln!("SKIPPED: {} - OpenGL not available", function!());
            return;
        }

        let image = crate::load_image(
            include_bytes!(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/../../testdata/giraffe.jpg"
            )),
            Some(PixelFormat::Rgba),
            None,
        )
        .unwrap();
        let mut image_dyn = image;

        let segmentation = Array3::from_shape_vec(
            (76, 55, 1),
            include_bytes!(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/../../testdata/yolov8_seg_crop_76x55.bin"
            ))
            .to_vec(),
        )
        .unwrap();

        let detect = DetectBox {
            bbox: [0.59375, 0.25, 0.9375, 0.725].into(),
            score: 0.99,
            label: 1,
        };

        let seg = Segmentation {
            segmentation,
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
            .draw_masks(&mut image_dyn, &[detect], &[seg])
            .unwrap();

        let image = {
            let mut __t = image_dyn.into_u8().unwrap();
            __t.set_format(PixelFormat::Rgba).unwrap();
            TensorDyn::from(__t)
        };
        let expected = crate::load_image(
            include_bytes!(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/../../testdata/output_render_gl.jpg"
            )),
            Some(PixelFormat::Rgba),
            None,
        )
        .unwrap();

        compare_images(&image, &expected, 0.99, function!());
    }

    #[test]
    fn test_boxes() {
        use edgefirst_decoder::DetectBox;

        if !is_opengl_available() {
            eprintln!("SKIPPED: {} - OpenGL not available", function!());
            return;
        }

        let image = crate::load_image(
            include_bytes!(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/../../testdata/giraffe.jpg"
            )),
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
        renderer.draw_masks(&mut image_dyn, &[detect], &[]).unwrap();
    }

    static GL_AVAILABLE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    // Helper function to check if OpenGL is available
    fn is_opengl_available() -> bool {
        #[cfg(all(target_os = "linux", feature = "opengl"))]
        {
            *GL_AVAILABLE.get_or_init(|| GLProcessorThreaded::new(None).is_ok())
        }

        #[cfg(not(all(target_os = "linux", feature = "opengl")))]
        {
            false
        }
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
                PixelFormat::Rgb | PixelFormat::Rgba | PixelFormat::Grey | PixelFormat::PlanarRgb
            ),
            "format must be Rgb or Rgba for comparison"
        );

        let image1 = match img1.format().unwrap() {
            PixelFormat::Rgb => image::RgbImage::from_vec(
                img1.width().unwrap() as u32,
                img1.height().unwrap() as u32,
                img1.as_u8().unwrap().map().unwrap().to_vec(),
            )
            .unwrap(),
            PixelFormat::Rgba => image::RgbaImage::from_vec(
                img1.width().unwrap() as u32,
                img1.height().unwrap() as u32,
                img1.as_u8().unwrap().map().unwrap().to_vec(),
            )
            .unwrap()
            .convert(),
            PixelFormat::Grey => image::GrayImage::from_vec(
                img1.width().unwrap() as u32,
                img1.height().unwrap() as u32,
                img1.as_u8().unwrap().map().unwrap().to_vec(),
            )
            .unwrap()
            .convert(),
            PixelFormat::PlanarRgb => image::GrayImage::from_vec(
                img1.width().unwrap() as u32,
                (img1.height().unwrap() * 3) as u32,
                img1.as_u8().unwrap().map().unwrap().to_vec(),
            )
            .unwrap()
            .convert(),
            _ => return,
        };

        let image2 = match img2.format().unwrap() {
            PixelFormat::Rgb => image::RgbImage::from_vec(
                img2.width().unwrap() as u32,
                img2.height().unwrap() as u32,
                img2.as_u8().unwrap().map().unwrap().to_vec(),
            )
            .unwrap(),
            PixelFormat::Rgba => image::RgbaImage::from_vec(
                img2.width().unwrap() as u32,
                img2.height().unwrap() as u32,
                img2.as_u8().unwrap().map().unwrap().to_vec(),
            )
            .unwrap()
            .convert(),
            PixelFormat::Grey => image::GrayImage::from_vec(
                img2.width().unwrap() as u32,
                img2.height().unwrap() as u32,
                img2.as_u8().unwrap().map().unwrap().to_vec(),
            )
            .unwrap()
            .convert(),
            PixelFormat::PlanarRgb => image::GrayImage::from_vec(
                img2.width().unwrap() as u32,
                (img2.height().unwrap() * 3) as u32,
                img2.as_u8().unwrap().map().unwrap().to_vec(),
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
            // image1.save(format!("{name}_1.png"));
            // image2.save(format!("{name}_2.png"));
            similarity
                .image
                .to_color_map()
                .save(format!("{name}.png"))
                .unwrap();
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

    #[cfg(feature = "dma_test_formats")]
    fn load_raw_image(
        width: usize,
        height: usize,
        format: PixelFormat,
        memory: Option<TensorMemory>,
        bytes: &[u8],
    ) -> Result<TensorDyn, crate::Error> {
        let img = TensorDyn::image(width, height, format, DType::U8, memory)?;
        let mut map = img.as_u8().unwrap().map()?;
        map.as_mut_slice()[..bytes.len()].copy_from_slice(bytes);
        Ok(img)
    }

    /// Test OpenGL PixelFormat::Nv12→PixelFormat::Rgba conversion against ffmpeg reference
    #[test]
    #[cfg(all(target_os = "linux", feature = "dma_test_formats"))]
    fn test_opengl_nv12_to_rgba_reference() {
        if !is_dma_available() {
            return;
        }
        // Load PixelFormat::Nv12 source with DMA
        let src = load_raw_image(
            1280,
            720,
            PixelFormat::Nv12,
            Some(TensorMemory::Dma),
            include_bytes!(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/../../testdata/camera720p.nv12"
            )),
        )
        .unwrap();

        // Load PixelFormat::Rgba reference (ffmpeg-generated)
        let reference = load_raw_image(
            1280,
            720,
            PixelFormat::Rgba,
            None,
            include_bytes!(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/../../testdata/camera720p.rgba"
            )),
        )
        .unwrap();

        // Convert using OpenGL
        let dst = TensorDyn::image(
            1280,
            720,
            PixelFormat::Rgba,
            DType::U8,
            Some(TensorMemory::Dma),
        )
        .unwrap();
        let mut gl = GLProcessorThreaded::new(None).unwrap();
        let src_dyn = src;
        let mut dst_dyn = dst;
        gl.convert(
            &src_dyn,
            &mut dst_dyn,
            Rotation::None,
            Flip::None,
            Crop::no_crop(),
        )
        .unwrap();

        // Copy to CPU for comparison
        let cpu_dst = TensorDyn::image(1280, 720, PixelFormat::Rgba, DType::U8, None).unwrap();
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
    #[cfg(all(target_os = "linux", feature = "dma_test_formats"))]
    fn test_opengl_yuyv_to_rgba_reference() {
        if !is_dma_available() {
            return;
        }
        // Load PixelFormat::Yuyv source with DMA
        let src = load_raw_image(
            1280,
            720,
            PixelFormat::Yuyv,
            Some(TensorMemory::Dma),
            include_bytes!(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/../../testdata/camera720p.yuyv"
            )),
        )
        .unwrap();

        // Load PixelFormat::Rgba reference (ffmpeg-generated)
        let reference = load_raw_image(
            1280,
            720,
            PixelFormat::Rgba,
            None,
            include_bytes!(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/../../testdata/camera720p.rgba"
            )),
        )
        .unwrap();

        // Convert using OpenGL
        let dst = TensorDyn::image(
            1280,
            720,
            PixelFormat::Rgba,
            DType::U8,
            Some(TensorMemory::Dma),
        )
        .unwrap();
        let mut gl = GLProcessorThreaded::new(None).unwrap();
        let src_dyn = src;
        let mut dst_dyn = dst;
        gl.convert(
            &src_dyn,
            &mut dst_dyn,
            Rotation::None,
            Flip::None,
            Crop::no_crop(),
        )
        .unwrap();

        // Copy to CPU for comparison
        let cpu_dst = TensorDyn::image(1280, 720, PixelFormat::Rgba, DType::U8, None).unwrap();
        cpu_dst
            .as_u8()
            .unwrap()
            .map()
            .unwrap()
            .as_mut_slice()
            .copy_from_slice(dst_dyn.as_u8().unwrap().map().unwrap().as_slice());

        compare_images(&reference, &cpu_dst, 0.98, "opengl_yuyv_to_rgba_reference");
    }

    // =========================================================================
    // EGL Display Probe & Override Tests
    // =========================================================================

    /// Validate that probe_egl_displays() discovers available display types
    /// and returns them in priority order (GBM first).
    ///
    /// On headless i.MX hardware, GBM and PlatformDevice are typically
    /// available. Default requires a running compositor (Wayland/X11) and
    /// may not be present on headless targets.
    #[test]
    fn test_probe_egl_displays() {
        let displays = match probe_egl_displays() {
            Ok(d) => d,
            Err(e) => {
                eprintln!("SKIPPED: {} - EGL not available: {e:?}", function!());
                return;
            }
        };

        if displays.is_empty() {
            eprintln!("SKIPPED: {} - No EGL displays available", function!());
            return;
        }

        let kinds: Vec<_> = displays.iter().map(|d| d.kind).collect();
        eprintln!("Probed EGL displays: {kinds:?}");
        for d in &displays {
            eprintln!("  {:?}: {}", d.kind, d.description);
        }

        // Verify priority ordering: PlatformDevice > GBM > Default.
        // Not all display types are available on every system, but the
        // ones that are present must appear in this order.
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

    /// Validate that explicitly selecting each available display kind via
    /// GLProcessorThreaded::new(Some(kind)) succeeds and produces a working
    /// converter.
    #[test]
    fn test_override_each_display_kind() {
        let displays = match probe_egl_displays() {
            Ok(d) => d,
            Err(e) => {
                eprintln!("SKIPPED: {} - EGL not available: {e:?}", function!());
                return;
            }
        };

        if displays.is_empty() {
            eprintln!("SKIPPED: {} - No EGL displays available", function!());
            return;
        }

        for display in &displays {
            eprintln!(
                "Testing override: {:?} ({})",
                display.kind, display.description
            );
            let mut gl = GLProcessorThreaded::new(Some(display.kind)).unwrap_or_else(|e| {
                panic!(
                    "GLProcessorThreaded::new(Some({:?})) failed: {e:?}",
                    display.kind
                )
            });

            // Smoke test: do a simple PixelFormat::Rgba → PixelFormat::Rgba conversion to verify the
            // GL context is fully functional.
            let src = crate::load_image(
                include_bytes!(concat!(
                    env!("CARGO_MANIFEST_DIR"),
                    "/../../testdata/zidane.jpg"
                )),
                Some(PixelFormat::Rgba),
                None,
            )
            .unwrap();
            let dst = TensorDyn::image(320, 240, PixelFormat::Rgba, DType::U8, None).unwrap();
            let src_dyn = src;
            let mut dst_dyn = dst;
            gl.convert(
                &src_dyn,
                &mut dst_dyn,
                Rotation::None,
                Flip::None,
                Crop::no_crop(),
            )
            .unwrap_or_else(|e| panic!("convert() with {:?} display failed: {e:?}", display.kind));
            eprintln!("  {:?} display: convert OK", display.kind);
        }
    }

    /// Validate that requesting a display kind that doesn't exist on the
    /// system returns an error rather than falling back silently.
    #[test]
    fn test_override_unavailable_display_errors() {
        let displays = match probe_egl_displays() {
            Ok(d) => d,
            Err(e) => {
                eprintln!("SKIPPED: {} - EGL not available: {e:?}", function!());
                return;
            }
        };
        let available_kinds: Vec<_> = displays.iter().map(|d| d.kind).collect();

        // Find a kind that is NOT available; if all three are available,
        // this test has nothing to verify — skip it.
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
            eprintln!(
                "SKIPPED: {} - All three display kinds are available",
                function!()
            );
        }
    }

    /// Validate that auto-detection (None) still works — this is the existing
    /// default behaviour and must not regress.
    #[test]
    fn test_auto_detect_display() {
        if !is_opengl_available() {
            eprintln!("SKIPPED: {} - OpenGL not available", function!());
            return;
        }

        let mut gl = GLProcessorThreaded::new(None).expect("auto-detect should succeed");
        let src = crate::load_image(
            include_bytes!(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/../../testdata/zidane.jpg"
            )),
            Some(PixelFormat::Rgba),
            None,
        )
        .unwrap();
        let dst = TensorDyn::image(320, 240, PixelFormat::Rgba, DType::U8, None).unwrap();
        let src_dyn = src;
        let mut dst_dyn = dst;
        gl.convert(
            &src_dyn,
            &mut dst_dyn,
            Rotation::None,
            Flip::None,
            Crop::no_crop(),
        )
        .expect("auto-detect convert should succeed");
    }

    #[test]
    fn test_packed_rgb_width_constraint() {
        // Standard ML model input widths — all satisfy W*3 % 4 == 0
        assert_eq!((640usize * 3) % 4, 0);
        assert_eq!((320usize * 3) % 4, 0);
        assert_eq!((1280usize * 3) % 4, 0);

        // Non-divisible widths should be rejected
        assert_ne!((322usize * 3) % 4, 0);
        assert_ne!((333usize * 3) % 4, 0);
    }

    // =========================================================================
    // Packed PixelFormat::Rgb Correctness Tests (two-pass pipeline)
    // These tests compare GL PixelFormat::Rgba output (alpha stripped) against GL packed
    // PixelFormat::Rgb output. Both use the same GPU color conversion, so differences
    // isolate packing shader bugs rather than CPU-vs-GPU YUV conversion.
    // They require DMA + OpenGL hardware (on-target only).
    // =========================================================================

    /// Compare two byte slices pixel-by-pixel with tolerance.
    /// Panics with details if any byte differs by more than `tolerance`.
    #[cfg(feature = "dma_test_formats")]
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
    #[cfg(feature = "dma_test_formats")]
    fn letterbox_crop(src_w: usize, src_h: usize, dst_w: usize, dst_h: usize) -> Crop {
        let src_aspect = src_w as f64 / src_h as f64;
        let dst_aspect = dst_w as f64 / dst_h as f64;
        let (new_w, new_h) = if src_aspect > dst_aspect {
            let new_h = (dst_w as f64 / src_aspect).round() as usize;
            (dst_w, new_h)
        } else {
            let new_w = (dst_h as f64 * src_aspect).round() as usize;
            (new_w, dst_h)
        };
        let left = (dst_w - new_w) / 2;
        let top = (dst_h - new_h) / 2;
        Crop::new()
            .with_dst_rect(Some(crate::Rect::new(left, top, new_w, new_h)))
            .with_dst_color(Some([114, 114, 114, 255]))
    }

    /// Strip alpha from PixelFormat::Rgba bytes → packed PixelFormat::Rgb bytes.
    #[cfg(feature = "dma_test_formats")]
    fn rgba_to_rgb(rgba: &[u8]) -> Vec<u8> {
        assert_eq!(
            rgba.len() % 4,
            0,
            "PixelFormat::Rgba buffer length must be divisible by 4"
        );
        let mut rgb = Vec::with_capacity(rgba.len() / 4 * 3);
        for pixel in rgba.chunks_exact(4) {
            rgb.push(pixel[0]);
            rgb.push(pixel[1]);
            rgb.push(pixel[2]);
        }
        rgb
    }

    /// Convert uint8 PixelFormat::Rgb bytes to int8 (XOR 0x80 each byte).
    #[cfg(feature = "dma_test_formats")]
    fn uint8_to_int8(data: &[u8]) -> Vec<u8> {
        data.iter().map(|&b| b ^ 0x80).collect()
    }

    /// PixelFormat::Yuyv 1080p → PixelFormat::Rgb 640x640 with letterbox (two-pass packed PixelFormat::Rgb pipeline).
    /// Compares GL PixelFormat::Rgba (alpha-stripped) against GL packed PixelFormat::Rgb to validate packing.
    #[test]
    #[cfg(all(target_os = "linux", feature = "dma_test_formats"))]
    fn test_opengl_rgb_correctness() {
        if !is_dma_available() {
            return;
        }
        let src_dma = load_raw_image(
            1920,
            1080,
            PixelFormat::Yuyv,
            Some(TensorMemory::Dma),
            include_bytes!(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/../../testdata/camera1080p.yuyv"
            )),
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
            Some(TensorMemory::Dma),
        )
        .unwrap();
        let mut dst_rgba_dyn = dst_rgba;
        gl.convert(
            &src_dyn,
            &mut dst_rgba_dyn,
            Rotation::None,
            Flip::None,
            crop,
        )
        .unwrap();

        // GL packed PixelFormat::Rgb output
        let dst_rgb = TensorDyn::image(
            640,
            640,
            PixelFormat::Rgb,
            DType::U8,
            Some(TensorMemory::Dma),
        )
        .unwrap();
        let mut dst_rgb_dyn = dst_rgb;
        gl.convert(&src_dyn, &mut dst_rgb_dyn, Rotation::None, Flip::None, crop)
            .unwrap();

        let rgba_data = dst_rgba_dyn.as_u8().unwrap().map().unwrap();
        let expected_rgb = rgba_to_rgb(rgba_data.as_slice());
        let gl_data = dst_rgb_dyn.as_u8().unwrap().map().unwrap();
        assert_pixels_match(&expected_rgb, gl_data.as_slice(), 1);
    }

    /// PixelFormat::Yuyv 1080p → PixelFormat::Rgb 640x640 with letterbox.
    /// Compares GL PixelFormat::Rgba (alpha-stripped, XOR 0x80) against GL packed PixelFormat::Rgb.
    #[test]
    #[cfg(all(target_os = "linux", feature = "dma_test_formats"))]
    fn test_opengl_rgb_int8_correctness() {
        if !is_dma_available() {
            return;
        }
        let src_dma = load_raw_image(
            1920,
            1080,
            PixelFormat::Yuyv,
            Some(TensorMemory::Dma),
            include_bytes!(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/../../testdata/camera1080p.yuyv"
            )),
        )
        .unwrap();

        let crop = letterbox_crop(1920, 1080, 640, 640);
        let mut gl = match GLProcessorST::new(None) {
            Ok(gl) => gl,
            Err(e) => {
                eprintln!("SKIPPED: {} - GL not available: {e}", function!());
                return;
            }
        };

        // Two-pass packed PixelFormat::Rgb is intentionally rejected on the DMA backend.
        // This test validates the two-pass int8 pipeline which requires non-DMA.
        if gl.gl_context.transfer_backend.is_dma() {
            eprintln!(
                "SKIPPED: {} - DMA backend does not support two-pass packed PixelFormat::Rgb",
                function!()
            );
            return;
        }
        let src_dyn = src_dma;

        // GL PixelFormat::Rgba reference
        let dst_rgba = TensorDyn::image(
            640,
            640,
            PixelFormat::Rgba,
            DType::U8,
            Some(TensorMemory::Dma),
        )
        .unwrap();
        let mut dst_rgba_dyn = dst_rgba;
        gl.convert(
            &src_dyn,
            &mut dst_rgba_dyn,
            Rotation::None,
            Flip::None,
            crop,
        )
        .unwrap();

        // GL packed PixelFormat::Rgb output (two-pass path)
        let dst_rgb = TensorDyn::image(
            640,
            640,
            PixelFormat::Rgb,
            DType::U8,
            Some(TensorMemory::Dma),
        )
        .unwrap();
        let mut dst_rgb_dyn = dst_rgb;
        gl.convert(&src_dyn, &mut dst_rgb_dyn, Rotation::None, Flip::None, crop)
            .unwrap();

        let rgba_data = dst_rgba_dyn.as_u8().unwrap().map().unwrap();
        let expected_rgb = uint8_to_int8(&rgba_to_rgb(rgba_data.as_slice()));
        let gl_data = dst_rgb_dyn.as_u8().unwrap().map().unwrap();
        assert_pixels_match(&expected_rgb, gl_data.as_slice(), 1);
    }

    /// PixelFormat::Yuyv 1080p → PixelFormat::Rgb 1920x1080 (no letterbox, same size).
    /// Compares GL PixelFormat::Rgba (alpha-stripped) against GL packed PixelFormat::Rgb without scaling.
    #[test]
    #[cfg(all(target_os = "linux", feature = "dma_test_formats"))]
    fn test_opengl_rgb_no_letterbox_correctness() {
        if !is_dma_available() {
            return;
        }
        let src_dma = load_raw_image(
            1920,
            1080,
            PixelFormat::Yuyv,
            Some(TensorMemory::Dma),
            include_bytes!(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/../../testdata/camera1080p.yuyv"
            )),
        )
        .unwrap();

        let mut gl = GLProcessorThreaded::new(None).unwrap();
        let src_dyn = src_dma;

        // GL PixelFormat::Rgba reference (no letterbox — 1920 satisfies W*3 % 4 == 0)
        let dst_rgba = TensorDyn::image(
            1920,
            1080,
            PixelFormat::Rgba,
            DType::U8,
            Some(TensorMemory::Dma),
        )
        .unwrap();
        let mut dst_rgba_dyn = dst_rgba;
        gl.convert(
            &src_dyn,
            &mut dst_rgba_dyn,
            Rotation::None,
            Flip::None,
            Crop::no_crop(),
        )
        .unwrap();

        // GL packed PixelFormat::Rgb output
        let dst_rgb = TensorDyn::image(
            1920,
            1080,
            PixelFormat::Rgb,
            DType::U8,
            Some(TensorMemory::Dma),
        )
        .unwrap();
        let mut dst_rgb_dyn = dst_rgb;
        gl.convert(
            &src_dyn,
            &mut dst_rgb_dyn,
            Rotation::None,
            Flip::None,
            Crop::no_crop(),
        )
        .unwrap();

        let rgba_data = dst_rgba_dyn.as_u8().unwrap().map().unwrap();
        let expected_rgb = rgba_to_rgb(rgba_data.as_slice());
        let gl_data = dst_rgb_dyn.as_u8().unwrap().map().unwrap();
        assert_pixels_match(&expected_rgb, gl_data.as_slice(), 1);
    }

    // ---- PixelFormat::Bgra destination tests ----

    /// Test OpenGL PixelFormat::Nv12→PixelFormat::Bgra conversion with DMA buffers.
    /// Compares against PixelFormat::Nv12→PixelFormat::Rgba by verifying R↔B swap.
    #[test]
    #[cfg(all(target_os = "linux", feature = "dma_test_formats"))]
    fn test_opengl_nv12_to_bgra() {
        if !is_dma_available() {
            eprintln!("SKIPPED: test_opengl_nv12_to_bgra - DMA not available");
            return;
        }

        let src = load_raw_image(
            1280,
            720,
            PixelFormat::Nv12,
            Some(TensorMemory::Dma),
            include_bytes!(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/../../testdata/camera720p.nv12"
            )),
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
            Some(TensorMemory::Dma),
        )
        .unwrap();
        let mut rgba_dst_dyn = rgba_dst;
        gl.convert(
            &src_dyn,
            &mut rgba_dst_dyn,
            Rotation::None,
            Flip::None,
            Crop::no_crop(),
        )
        .unwrap();

        // Convert to PixelFormat::Bgra
        let bgra_dst = TensorDyn::image(
            1280,
            720,
            PixelFormat::Bgra,
            DType::U8,
            Some(TensorMemory::Dma),
        )
        .unwrap();
        let mut bgra_dst_dyn = bgra_dst;
        gl.convert(
            &src_dyn,
            &mut bgra_dst_dyn,
            Rotation::None,
            Flip::None,
            Crop::no_crop(),
        )
        .unwrap();

        // Compare: PixelFormat::Bgra[B,G,R,A] should match PixelFormat::Rgba[R,G,B,A] with R↔B swapped
        let bgra_map = bgra_dst_dyn.as_u8().unwrap().map().unwrap();
        let rgba_map = rgba_dst_dyn.as_u8().unwrap().map().unwrap();
        let bgra_buf = bgra_map.as_slice();
        let rgba_buf = rgba_map.as_slice();

        assert_eq!(bgra_buf.len(), rgba_buf.len());
        let mut max_diff = 0i32;
        for (bc, rc) in bgra_buf.chunks_exact(4).zip(rgba_buf.chunks_exact(4)) {
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

    /// Test OpenGL PixelFormat::Yuyv→PixelFormat::Bgra conversion with DMA buffers.
    #[test]
    #[cfg(all(target_os = "linux", feature = "dma_test_formats"))]
    fn test_opengl_yuyv_to_bgra() {
        if !is_dma_available() {
            eprintln!("SKIPPED: test_opengl_yuyv_to_bgra - DMA not available");
            return;
        }

        let src = load_raw_image(
            1280,
            720,
            PixelFormat::Yuyv,
            Some(TensorMemory::Dma),
            include_bytes!(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/../../testdata/camera720p.yuyv"
            )),
        )
        .unwrap();

        let mut gl = GLProcessorThreaded::new(None).unwrap();
        let src_dyn = src;

        let rgba_dst = TensorDyn::image(
            1280,
            720,
            PixelFormat::Rgba,
            DType::U8,
            Some(TensorMemory::Dma),
        )
        .unwrap();
        let mut rgba_dst_dyn = rgba_dst;
        gl.convert(
            &src_dyn,
            &mut rgba_dst_dyn,
            Rotation::None,
            Flip::None,
            Crop::no_crop(),
        )
        .unwrap();

        let bgra_dst = TensorDyn::image(
            1280,
            720,
            PixelFormat::Bgra,
            DType::U8,
            Some(TensorMemory::Dma),
        )
        .unwrap();
        let mut bgra_dst_dyn = bgra_dst;
        gl.convert(
            &src_dyn,
            &mut bgra_dst_dyn,
            Rotation::None,
            Flip::None,
            Crop::no_crop(),
        )
        .unwrap();

        let bgra_map = bgra_dst_dyn.as_u8().unwrap().map().unwrap();
        let rgba_map = rgba_dst_dyn.as_u8().unwrap().map().unwrap();
        let bgra_buf = bgra_map.as_slice();
        let rgba_buf = rgba_map.as_slice();

        let mut max_diff = 0i32;
        for (bc, rc) in bgra_buf.chunks_exact(4).zip(rgba_buf.chunks_exact(4)) {
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

    /// Test draw_masks() with PixelFormat::Bgra destination (segmentation).
    /// Draws the same masks to both PixelFormat::Rgba and PixelFormat::Bgra, then verifies R↔B swap.
    #[test]
    fn test_draw_masks_bgra() {
        use edgefirst_decoder::Segmentation;

        if !is_opengl_available() {
            eprintln!("SKIPPED: test_draw_masks_bgra - OpenGL not available");
            return;
        }

        let seg_bytes = include_bytes!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../testdata/modelpack_seg_2x160x160.bin"
        ))
        .to_vec();

        // Build segmentation data (shared between both renders)
        let make_seg = || {
            let mut s = Array3::from_shape_vec((2, 160, 160), seg_bytes.clone()).unwrap();
            s.swap_axes(0, 1);
            s.swap_axes(1, 2);
            let s = s.as_standard_layout().to_owned();
            Segmentation {
                segmentation: s,
                xmin: 0.0,
                ymin: 0.0,
                xmax: 1.0,
                ymax: 1.0,
            }
        };

        let mut gl = GLProcessorThreaded::new(None).unwrap();

        // Render to PixelFormat::Rgba
        let rgba_img = crate::load_image(
            include_bytes!(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/../../testdata/giraffe.jpg"
            )),
            Some(PixelFormat::Rgba),
            None,
        )
        .unwrap();
        let mut rgba_img_dyn = rgba_img;
        gl.draw_masks(&mut rgba_img_dyn, &[], &[make_seg()])
            .unwrap();

        // Render to PixelFormat::Bgra (convert source to PixelFormat::Bgra first)
        let rgba_src = crate::load_image(
            include_bytes!(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/../../testdata/giraffe.jpg"
            )),
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
        )
        .unwrap();
        let rgba_src_dyn = rgba_src;
        let mut bgra_img_dyn = bgra_img;
        gl.convert(
            &rgba_src_dyn,
            &mut bgra_img_dyn,
            Rotation::None,
            Flip::None,
            Crop::no_crop(),
        )
        .unwrap();
        gl.draw_masks(&mut bgra_img_dyn, &[], &[make_seg()])
            .unwrap();

        // Verify PixelFormat::Bgra output matches PixelFormat::Rgba output with R↔B swapped
        let rgba_map = rgba_img_dyn.as_u8().unwrap().map().unwrap();
        let bgra_map = bgra_img_dyn.as_u8().unwrap().map().unwrap();
        let rgba_buf = rgba_map.as_slice();
        let bgra_buf = bgra_map.as_slice();
        assert_eq!(rgba_buf.len(), bgra_buf.len());

        let mut max_diff = 0i32;
        for (rc, bc) in rgba_buf.chunks_exact(4).zip(bgra_buf.chunks_exact(4)) {
            max_diff = max_diff.max((rc[0] as i32 - bc[2] as i32).abs()); // R
            max_diff = max_diff.max((rc[1] as i32 - bc[1] as i32).abs()); // G
            max_diff = max_diff.max((rc[2] as i32 - bc[0] as i32).abs()); // B
            max_diff = max_diff.max((rc[3] as i32 - bc[3] as i32).abs()); // A
        }
        eprintln!("draw_masks PixelFormat::Bgra vs PixelFormat::Rgba max channel diff: {max_diff}");
        assert!(
            max_diff <= 1,
            "draw_masks PixelFormat::Bgra/PixelFormat::Rgba channel mismatch > 1: max_diff={max_diff}"
        );
    }

    /// Test draw_masks() with PixelFormat::Bgra destination using Mem memory (boxes).
    /// Draws same boxes to PixelFormat::Rgba and PixelFormat::Bgra, then verifies R↔B swap.
    #[test]
    fn test_draw_masks_bgra_mem() {
        use edgefirst_decoder::DetectBox;

        if !is_opengl_available() {
            eprintln!("SKIPPED: test_draw_masks_bgra_mem - OpenGL not available");
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
        let rgba_img = crate::load_image(
            include_bytes!(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/../../testdata/giraffe.jpg"
            )),
            Some(PixelFormat::Rgba),
            Some(edgefirst_tensor::TensorMemory::Mem),
        )
        .unwrap();
        let mut rgba_img_dyn = rgba_img;
        gl.draw_masks(&mut rgba_img_dyn, &[detect], &[]).unwrap();

        // Render boxes to PixelFormat::Bgra
        let rgba_src = crate::load_image(
            include_bytes!(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/../../testdata/giraffe.jpg"
            )),
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
        )
        .unwrap();
        let rgba_src_dyn = rgba_src;
        let mut bgra_img_dyn = bgra_img;
        gl.convert(
            &rgba_src_dyn,
            &mut bgra_img_dyn,
            Rotation::None,
            Flip::None,
            Crop::no_crop(),
        )
        .unwrap();
        gl.draw_masks(&mut bgra_img_dyn, &[detect], &[]).unwrap();

        // Verify PixelFormat::Bgra output matches PixelFormat::Rgba output with R↔B swapped
        let rgba_map = rgba_img_dyn.as_u8().unwrap().map().unwrap();
        let bgra_map = bgra_img_dyn.as_u8().unwrap().map().unwrap();
        let rgba_buf = rgba_map.as_slice();
        let bgra_buf = bgra_map.as_slice();

        let mut max_diff = 0i32;
        for (rc, bc) in rgba_buf.chunks_exact(4).zip(bgra_buf.chunks_exact(4)) {
            max_diff = max_diff.max((rc[0] as i32 - bc[2] as i32).abs());
            max_diff = max_diff.max((rc[1] as i32 - bc[1] as i32).abs());
            max_diff = max_diff.max((rc[2] as i32 - bc[0] as i32).abs());
            max_diff = max_diff.max((rc[3] as i32 - bc[3] as i32).abs());
        }
        eprintln!(
            "draw_masks_mem PixelFormat::Bgra vs PixelFormat::Rgba max channel diff: {max_diff}"
        );
        assert!(
            max_diff <= 1,
            "draw_masks_mem PixelFormat::Bgra/PixelFormat::Rgba channel mismatch > 1: max_diff={max_diff}"
        );
    }

    // ========================================================================
    // GL smoke tests for mask rendering and PBO destinations
    // ========================================================================

    #[test]
    fn test_gl_mask_render_smoke() {
        if !is_opengl_available() {
            eprintln!("SKIPPED: {} - OpenGL not available", function!());
            return;
        }

        let mut gl = GLProcessorThreaded::new(None).unwrap();
        let image = TensorDyn::image(64, 64, PixelFormat::Rgba, DType::U8, None).unwrap();
        let mut image_dyn = image;

        // Render with empty detections and segmentations — should succeed trivially
        let result = gl.draw_masks(&mut image_dyn, &[], &[]);
        assert!(
            result.is_ok(),
            "GL mask render with empty data should succeed: {result:?}"
        );

        // Verify output dimensions are unchanged
        assert_eq!(image_dyn.width(), Some(64));
        assert_eq!(image_dyn.height(), Some(64));
    }

    #[test]
    fn test_gl_pbo_destination_smoke() {
        if !is_opengl_available() {
            eprintln!("SKIPPED: {} - OpenGL not available", function!());
            return;
        }

        let gl = GLProcessorThreaded::new(None).unwrap();
        let result = gl.create_pbo_image(64, 64, PixelFormat::Rgba);
        match result {
            Ok(pbo_img) => {
                assert_eq!(pbo_img.width(), Some(64));
                assert_eq!(pbo_img.height(), Some(64));
                assert_eq!(pbo_img.format().unwrap(), PixelFormat::Rgba);
            }
            Err(e) => {
                // PBO may not be supported on all GL implementations
                eprintln!("SKIPPED: {} - PBO not supported: {e:?}", function!());
            }
        }
    }

    // ---- Multiplane PixelFormat::Nv12 GPU tests ----

    /// Helper: load PixelFormat::Nv12 raw bytes into separate DMA-backed luma and chroma tensors,
    /// returning a multiplane TensorDyn suitable for GPU EGLImage import.
    #[cfg(feature = "dma_test_formats")]
    fn load_multiplane_nv12_dma(width: usize, height: usize, nv12_bytes: &[u8]) -> TensorDyn {
        let y_size = width * height;
        let uv_size = width * (height / 2);
        assert_eq!(nv12_bytes.len(), y_size + uv_size);

        let luma = Tensor::new(&[height, width], Some(TensorMemory::Dma), Some("luma"))
            .expect("DMA luma tensor");
        luma.map().unwrap().as_mut_slice()[..y_size].copy_from_slice(&nv12_bytes[..y_size]);

        let chroma = Tensor::new(
            &[height / 2, width],
            Some(TensorMemory::Dma),
            Some("chroma"),
        )
        .expect("DMA chroma tensor");
        chroma.map().unwrap().as_mut_slice()[..uv_size].copy_from_slice(&nv12_bytes[y_size..]);

        Tensor::<u8>::from_planes(luma, chroma, PixelFormat::Nv12)
            .map(TensorDyn::from)
            .expect("multiplane PixelFormat::Nv12")
    }

    /// Multiplane PixelFormat::Nv12 → PixelFormat::Rgba via OpenGL DMA-BUF EGLImage (two separate FDs).
    /// Compares against contiguous PixelFormat::Nv12 → PixelFormat::Rgba to prove EGL multi-plane import works.
    #[test]
    #[cfg(all(target_os = "linux", feature = "dma_test_formats"))]
    fn test_multiplane_nv12_to_rgba_opengl() {
        if !is_dma_available() {
            eprintln!("SKIPPED: {} - DMA not available", function!());
            return;
        }

        let nv12_bytes: &[u8] = include_bytes!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../testdata/camera720p.nv12"
        ));

        // Contiguous PixelFormat::Nv12 (single DMA-BUF)
        let src_contiguous = load_raw_image(
            1280,
            720,
            PixelFormat::Nv12,
            Some(TensorMemory::Dma),
            nv12_bytes,
        )
        .unwrap();

        // Multiplane PixelFormat::Nv12 (two DMA-BUFs)
        let src_multiplane = load_multiplane_nv12_dma(1280, 720, nv12_bytes);
        assert!(src_multiplane.as_u8().unwrap().is_multiplane());

        let mut gl = GLProcessorThreaded::new(None).unwrap();

        // Convert contiguous
        let dst_contig = TensorDyn::image(
            1280,
            720,
            PixelFormat::Rgba,
            DType::U8,
            Some(TensorMemory::Dma),
        )
        .unwrap();
        let src_contiguous_dyn = src_contiguous;
        let mut dst_contig_dyn = dst_contig;
        gl.convert(
            &src_contiguous_dyn,
            &mut dst_contig_dyn,
            Rotation::None,
            Flip::None,
            Crop::no_crop(),
        )
        .unwrap();

        // Convert multiplane
        let dst_multi = TensorDyn::image(
            1280,
            720,
            PixelFormat::Rgba,
            DType::U8,
            Some(TensorMemory::Dma),
        )
        .unwrap();
        let src_multiplane_dyn = src_multiplane;
        let mut dst_multi_dyn = dst_multi;
        gl.convert(
            &src_multiplane_dyn,
            &mut dst_multi_dyn,
            Rotation::None,
            Flip::None,
            Crop::no_crop(),
        )
        .unwrap();

        // Compare pixel-for-pixel (should be identical — same data, different import path)
        let map_contig = dst_contig_dyn.as_u8().unwrap().map().unwrap();
        let map_multi = dst_multi_dyn.as_u8().unwrap().map().unwrap();
        assert_pixels_match(map_contig.as_slice(), map_multi.as_slice(), 0);
    }

    /// Multiplane PixelFormat::Nv12 720p → packed PixelFormat::Rgb 640x640 with letterbox resize via GL.
    /// Validates the packing shader works with multiplane EGLImage source.
    #[test]
    #[cfg(all(target_os = "linux", feature = "dma_test_formats"))]
    fn test_multiplane_nv12_to_rgb_letterbox_opengl() {
        if !is_dma_available() {
            eprintln!("SKIPPED: {} - DMA not available", function!());
            return;
        }

        let nv12_bytes: &[u8] = include_bytes!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../testdata/camera720p.nv12"
        ));

        let src_contiguous = load_raw_image(
            1280,
            720,
            PixelFormat::Nv12,
            Some(TensorMemory::Dma),
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
            Some(TensorMemory::Dma),
        )
        .unwrap();
        let src_contiguous_dyn = src_contiguous;
        let mut dst_contig_dyn = dst_contig;
        gl.convert(
            &src_contiguous_dyn,
            &mut dst_contig_dyn,
            Rotation::None,
            Flip::None,
            crop,
        )
        .unwrap();

        // Multiplane → packed PixelFormat::Rgb with letterbox
        let dst_multi = TensorDyn::image(
            640,
            640,
            PixelFormat::Rgb,
            DType::U8,
            Some(TensorMemory::Dma),
        )
        .unwrap();
        let src_multiplane_dyn = src_multiplane;
        let mut dst_multi_dyn = dst_multi;
        gl.convert(
            &src_multiplane_dyn,
            &mut dst_multi_dyn,
            Rotation::None,
            Flip::None,
            crop,
        )
        .unwrap();

        let map_contig = dst_contig_dyn.as_u8().unwrap().map().unwrap();
        let map_multi = dst_multi_dyn.as_u8().unwrap().map().unwrap();
        assert_pixels_match(map_contig.as_slice(), map_multi.as_slice(), 0);
    }

    /// Multiplane PixelFormat::Nv12 720p → packed PixelFormat::Rgb 640x640 with letterbox resize via GL.
    /// Validates the int8 packing shader (XOR 0x80) works with multiplane source.
    #[test]
    #[cfg(all(target_os = "linux", feature = "dma_test_formats"))]
    fn test_multiplane_nv12_to_rgb_int8_letterbox_opengl() {
        if !is_dma_available() {
            eprintln!("SKIPPED: {} - DMA not available", function!());
            return;
        }

        let nv12_bytes: &[u8] = include_bytes!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../testdata/camera720p.nv12"
        ));

        let src_contiguous = load_raw_image(
            1280,
            720,
            PixelFormat::Nv12,
            Some(TensorMemory::Dma),
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
            Some(TensorMemory::Dma),
        )
        .unwrap();
        let src_contiguous_dyn = src_contiguous;
        let mut dst_contig_dyn = dst_contig;
        gl.convert(
            &src_contiguous_dyn,
            &mut dst_contig_dyn,
            Rotation::None,
            Flip::None,
            crop,
        )
        .unwrap();

        // Multiplane → packed PixelFormat::Rgb with letterbox
        let dst_multi = TensorDyn::image(
            640,
            640,
            PixelFormat::Rgb,
            DType::U8,
            Some(TensorMemory::Dma),
        )
        .unwrap();
        let src_multiplane_dyn = src_multiplane;
        let mut dst_multi_dyn = dst_multi;
        gl.convert(
            &src_multiplane_dyn,
            &mut dst_multi_dyn,
            Rotation::None,
            Flip::None,
            crop,
        )
        .unwrap();

        let map_contig = dst_contig_dyn.as_u8().unwrap().map().unwrap();
        let map_multi = dst_multi_dyn.as_u8().unwrap().map().unwrap();
        assert_pixels_match(map_contig.as_slice(), map_multi.as_slice(), 0);
    }
}
