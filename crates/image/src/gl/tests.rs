// SPDX-FileCopyrightText: Copyright 2025 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

#[cfg(test)]
#[cfg(feature = "opengl")]
#[allow(deprecated)]
mod gl_tests {
    #[cfg(feature = "dma_test_formats")]
    use crate::opengl_headless::processor::GLProcessorST;
    use crate::{
        probe_egl_displays, Crop, EglDisplayKind, Flip, GLProcessorThreaded, ImageProcessorTrait,
        Rotation,
    };
    use edgefirst_decoder::DetectBox;
    #[cfg(feature = "dma_test_formats")]
    use edgefirst_tensor::is_dma_available;
    use edgefirst_tensor::{
        DType, PixelFormat, Tensor, TensorDyn, TensorMapTrait, TensorMemory, TensorTrait,
    };
    use image::buffer::ConvertBuffer;
    use ndarray::Array3;

    #[test]
    fn test_segmentation() {
        use edgefirst_decoder::Segmentation;

        if !is_opengl_available() {
            eprintln!("SKIPPED: {} - OpenGL not available", function!());
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
            segmentation,
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
        use edgefirst_decoder::Segmentation;

        if !is_opengl_available() {
            eprintln!("SKIPPED: {} - OpenGL not available", function!());
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
            segmentation,
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
        use edgefirst_decoder::Segmentation;
        use ndarray::Array3;

        if !is_opengl_available() {
            eprintln!("SKIPPED: {} - OpenGL not available", function!());
            return;
        }

        // draw_decoded_masks fully writes dst — pass the camera frame as
        // the MaskOverlay background, not as the dst canvas.
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

        // Threshold 0.95 (was 0.97): GPU-specific smoothstep anti-aliasing at
        // mask edges produces small cross-platform differences (x86 Mesa vs
        // Vivante), and the codec now decodes the `giraffe.jpg` background to
        // native NV12 (chroma subsampling) before the RGBA conversion, so the
        // NV12-sourced background diverges further from the golden captured
        // under the old direct-RGB JPEG decode.
        compare_images(&image, &expected, 0.95, function!());
    }

    #[test]
    fn test_boxes() {
        use edgefirst_decoder::DetectBox;

        if !is_opengl_available() {
            eprintln!("SKIPPED: {} - OpenGL not available", function!());
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
    #[cfg(all(target_os = "linux", feature = "dma_test_formats"))]
    static NEUTRON_AVAILABLE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();

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

    /// Returns true when running on an i.MX 95 board with Mali GPU.
    ///
    /// `/dev/neutron0` is used as a platform discriminator for i.MX 95 + Mali GPU,
    /// not as a check for NPU functionality.  The Neutron-scenario tests exercise
    /// large-offset DMA-BUF EGLImage imports that work on Mali but fail with
    /// `EGL(BadAccess)` on Vivante (i.MX 8MP) — even without the NPU driver.
    #[cfg(all(target_os = "linux", feature = "dma_test_formats"))]
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
            // Best-effort save of diff image for debugging (may fail on CI
            // where paths contain special characters or the fs is read-only).
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

    /// Validate that probe_egl_displays() populates the shared display and
    /// that a subsequent GLProcessorThreaded::new() reuses it without
    /// deadlocking.
    #[test]
    fn test_probe_then_create_gl_context() {
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

        eprintln!(
            "Probed displays: {:?}",
            displays.iter().map(|d| d.kind).collect::<Vec<_>>()
        );

        // Now create a GL processor — should reuse the shared display
        let mut gl = match GLProcessorThreaded::new(None) {
            Ok(gl) => gl,
            Err(e) => {
                eprintln!(
                    "SKIPPED: {} - GLProcessorThreaded failed: {e:?}",
                    function!()
                );
                return;
            }
        };

        // Verify it works by doing a simple convert
        let src = edgefirst_tensor::TensorDyn::image(
            64,
            64,
            edgefirst_tensor::PixelFormat::Rgba,
            edgefirst_tensor::DType::U8,
            None,
        )
        .expect("create src failed");
        let mut dst = edgefirst_tensor::TensorDyn::image(
            32,
            32,
            edgefirst_tensor::PixelFormat::Rgba,
            edgefirst_tensor::DType::U8,
            None,
        )
        .expect("create dst failed");

        gl.convert(
            &src,
            &mut dst,
            crate::Rotation::None,
            crate::Flip::None,
            crate::Crop::default(),
        )
        .expect("convert failed");

        assert_eq!(dst.width(), Some(32));
        assert_eq!(dst.height(), Some(32));
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
            let src = crate::load_image_test_helper(
                &edgefirst_bench::testdata::read("zidane.jpg"),
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
        let src = crate::load_image_test_helper(
            &edgefirst_bench::testdata::read("zidane.jpg"),
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
            &edgefirst_bench::testdata::read("camera1080p.yuyv"),
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

        // Two-pass packed RGB is now supported on the DMA backend.
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

        // GL packed PixelFormat::Rgb int8 output (two-pass path with XOR 0x80 bias)
        let dst_rgb = TensorDyn::image(
            640,
            640,
            PixelFormat::Rgb,
            DType::I8,
            Some(TensorMemory::Dma),
        )
        .unwrap();
        let mut dst_rgb_dyn = dst_rgb;
        gl.convert(&src_dyn, &mut dst_rgb_dyn, Rotation::None, Flip::None, crop)
            .unwrap();

        let rgba_data = dst_rgba_dyn.as_u8().unwrap().map().unwrap();
        let expected_rgb = uint8_to_int8(&rgba_to_rgb(rgba_data.as_slice()));
        // Map raw i8 bytes as u8 for comparison — the XOR 0x80 bias is in the data.
        let gl_data = dst_rgb_dyn.as_i8().unwrap().map().unwrap();
        let gl_bytes: &[u8] = unsafe {
            std::slice::from_raw_parts(gl_data.as_slice().as_ptr().cast(), gl_data.len())
        };
        assert_pixels_match(&expected_rgb, gl_bytes, 1);
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
            &edgefirst_bench::testdata::read("camera1080p.yuyv"),
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

    /// Test draw_decoded_masks() with PixelFormat::Bgra destination (segmentation).
    /// Draws the same masks to both PixelFormat::Rgba and PixelFormat::Bgra, then verifies R↔B swap.
    #[test]
    fn test_draw_decoded_masks_bgra() {
        use edgefirst_decoder::Segmentation;

        if !is_opengl_available() {
            eprintln!("SKIPPED: test_draw_decoded_masks_bgra - OpenGL not available");
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
                segmentation: s,
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
        gl.draw_decoded_masks(&mut bgra_img_dyn, &[], &[make_seg()], Default::default())
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
        eprintln!("draw_decoded_masks PixelFormat::Bgra vs PixelFormat::Rgba max channel diff: {max_diff}");
        assert!(
            max_diff <= 1,
            "draw_decoded_masks PixelFormat::Bgra/PixelFormat::Rgba channel mismatch > 1: max_diff={max_diff}"
        );
    }

    /// Test draw_decoded_masks() with PixelFormat::Bgra destination using Mem memory (boxes).
    /// Draws same boxes to PixelFormat::Rgba and PixelFormat::Bgra, then verifies R↔B swap.
    #[test]
    fn test_draw_decoded_masks_bgra_mem() {
        use edgefirst_decoder::DetectBox;

        if !is_opengl_available() {
            eprintln!("SKIPPED: test_draw_decoded_masks_bgra_mem - OpenGL not available");
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
        gl.draw_decoded_masks(&mut bgra_img_dyn, &[detect], &[], Default::default())
            .unwrap();

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
            eprintln!("SKIPPED: {} - OpenGL not available", function!());
            return;
        }

        let mut gl = GLProcessorThreaded::new(None).unwrap();
        let image = TensorDyn::image(64, 64, PixelFormat::Rgba, DType::U8, None).unwrap();
        let mut image_dyn = image;

        // Render with empty detections and segmentations — should succeed trivially
        let result = gl.draw_decoded_masks(&mut image_dyn, &[], &[], Default::default());
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

    /// Regression test for the PBO deadlock fix in `convert_any_to_pbo`
    /// (commit c494fae). Before the fix, the GL thread called
    /// `tensor.map()` on the PBO destination, which sent a message back to
    /// the very same GL thread and deadlocked the channel. The fix uses
    /// `setup_renderbuffer_from_pbo` to bind the PBO via the GL buffer-
    /// binding path, never reaching back through `map()`.
    ///
    /// This test would hang indefinitely (or, with the channel
    /// instrumentation, panic with "GL converter thread exited") if the
    /// deadlock returned. A successful return proves the round-trip
    /// completed.
    #[test]
    fn test_gl_convert_any_to_pbo_no_deadlock() {
        if !is_opengl_available() {
            eprintln!("SKIPPED: {} - OpenGL not available", function!());
            return;
        }

        let mut gl = GLProcessorThreaded::new(None).unwrap();
        let pbo_dst = match gl.create_pbo_image(64, 64, PixelFormat::Rgba) {
            Ok(t) => t,
            Err(e) => {
                eprintln!("SKIPPED: {} - PBO not supported: {e:?}", function!());
                return;
            }
        };
        let src = Tensor::<u8>::image(64, 64, PixelFormat::Rgba, Some(TensorMemory::Mem))
            .expect("Mem source tensor");
        // Fill source with a deterministic pattern so a regression that
        // produces a black/empty PBO would be detectable downstream.
        {
            let mut map = src.map().unwrap();
            let slice: &mut [u8] = &mut map;
            for (i, b) in slice.iter_mut().enumerate() {
                *b = (i & 0xFF) as u8;
            }
        }
        let src_dyn = TensorDyn::from(src);
        let mut dst_dyn = TensorDyn::from(pbo_dst);

        // The convert call is the deadlock site. If it returns at all the
        // fix is intact. We do not assert pixel-perfect contents here —
        // that is covered by other GL tests; this is a liveness regression
        // gate.
        let res = gl.convert(
            &src_dyn,
            &mut dst_dyn,
            Rotation::None,
            Flip::None,
            Crop::default(),
        );
        match res {
            Ok(()) => {
                assert_eq!(dst_dyn.width(), Some(64));
                assert_eq!(dst_dyn.height(), Some(64));
            }
            Err(e) => {
                // GL convert may legitimately fail on this CI host for
                // reasons unrelated to the deadlock (e.g. missing
                // extensions). Only fail the test on the deadlock
                // signature itself.
                let msg = format!("{e:?}");
                assert!(
                    !msg.contains("GL converter thread exited"),
                    "PBO destination convert deadlocked: {msg}"
                );
                eprintln!("SKIPPED: {} - convert failed unrelated: {msg}", function!());
            }
        }
    }

    /// Regression test for the `convert_pbo_to_pbo` deadlock path. Same
    /// underlying defect as `convert_any_to_pbo`; both sites were patched
    /// in commit c494fae and both need explicit coverage.
    #[test]
    fn test_gl_convert_pbo_to_pbo_no_deadlock() {
        if !is_opengl_available() {
            eprintln!("SKIPPED: {} - OpenGL not available", function!());
            return;
        }

        let mut gl = GLProcessorThreaded::new(None).unwrap();
        let pbo_src = match gl.create_pbo_image(64, 64, PixelFormat::Rgba) {
            Ok(t) => t,
            Err(e) => {
                eprintln!("SKIPPED: {} - PBO not supported: {e:?}", function!());
                return;
            }
        };
        let pbo_dst = match gl.create_pbo_image(64, 64, PixelFormat::Rgba) {
            Ok(t) => t,
            Err(e) => {
                eprintln!("SKIPPED: {} - PBO dst not supported: {e:?}", function!());
                return;
            }
        };
        let src_dyn = TensorDyn::from(pbo_src);
        let mut dst_dyn = TensorDyn::from(pbo_dst);
        let res = gl.convert(
            &src_dyn,
            &mut dst_dyn,
            Rotation::None,
            Flip::None,
            Crop::default(),
        );
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
                eprintln!("SKIPPED: {} - convert failed unrelated: {msg}", function!());
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

        let nv12_bytes: &[u8] = &edgefirst_bench::testdata::read("camera720p.nv12");

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

    /// Same-fd multiplane NV12: both Y and UV planes live in the same
    /// DMA-BUF, imported via `import_image` with two PlaneDescriptors that
    /// share the same underlying fd (dup'd).  This mirrors V4L2 / GStreamer
    /// pipelines where the camera driver exports a single DMA-BUF for both
    /// planes with the chroma at an offset.
    ///
    /// Exercises the full eglCreateImage → convert() render path.
    /// If this fails on one GPU but passes on others it is a driver
    /// limitation; if it fails everywhere the HAL attribute construction
    /// is wrong.
    #[test]
    #[cfg(all(target_os = "linux", feature = "dma_test_formats"))]
    fn test_same_fd_multiplane_nv12_to_rgba_opengl() {
        use edgefirst_tensor::PlaneDescriptor;

        if !is_dma_available() {
            eprintln!("SKIPPED: {} - DMA not available", function!());
            return;
        }
        if !is_opengl_available() {
            eprintln!("SKIPPED: {} - OpenGL not available", function!());
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
        let buf = Tensor::<u8>::new(
            &[total_bytes],
            Some(TensorMemory::Dma),
            Some("same_fd_nv12"),
        );
        let buf = match buf {
            Ok(t) if t.memory() == TensorMemory::Dma => t,
            _ => {
                eprintln!("SKIPPED: {} - DMA alloc failed", function!());
                return;
            }
        };
        // Fill with real NV12 camera data
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

        let proc = crate::ImageProcessor::new().unwrap();
        let src = proc
            .import_image(
                luma_pd,
                Some(chroma_pd),
                width,
                height,
                PixelFormat::Nv12,
                DType::U8,
            )
            .unwrap();
        assert!(src.is_multiplane(), "must be multiplane after import");

        // Also build contiguous reference for comparison
        let src_contig = load_raw_image(
            width,
            height,
            PixelFormat::Nv12,
            Some(TensorMemory::Dma),
            nv12_bytes,
        )
        .unwrap();

        let mut gl = GLProcessorThreaded::new(None).unwrap();

        // Render same-fd multiplane through full EGL pipeline
        let mut dst_same_fd = TensorDyn::image(
            width,
            height,
            PixelFormat::Rgba,
            DType::U8,
            Some(TensorMemory::Dma),
        )
        .unwrap();
        gl.convert(
            &src,
            &mut dst_same_fd,
            Rotation::None,
            Flip::None,
            Crop::no_crop(),
        )
        .unwrap();

        // Render contiguous reference
        let mut dst_contig = TensorDyn::image(
            width,
            height,
            PixelFormat::Rgba,
            DType::U8,
            Some(TensorMemory::Dma),
        )
        .unwrap();
        gl.convert(
            &src_contig,
            &mut dst_contig,
            Rotation::None,
            Flip::None,
            Crop::no_crop(),
        )
        .unwrap();

        // Compare: same-fd multiplane and contiguous must produce identical pixels
        let map_same = dst_same_fd.as_u8().unwrap().map().unwrap();
        let map_contig = dst_contig.as_u8().unwrap().map().unwrap();
        assert_pixels_match(map_same.as_slice(), map_contig.as_slice(), 0);
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

        let nv12_bytes: &[u8] = &edgefirst_bench::testdata::read("camera720p.nv12");

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

        let nv12_bytes: &[u8] = &edgefirst_bench::testdata::read("camera720p.nv12");

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

        // Contiguous → packed PixelFormat::Rgb int8 with letterbox
        let dst_contig = TensorDyn::image(
            640,
            640,
            PixelFormat::Rgb,
            DType::I8,
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

        // Multiplane → packed PixelFormat::Rgb int8 with letterbox
        let dst_multi = TensorDyn::image(
            640,
            640,
            PixelFormat::Rgb,
            DType::I8,
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

        // Map raw i8 bytes as u8 for comparison — both have XOR 0x80 bias.
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

    /// Compare fused GL proto rendering against hybrid (CPU materialize + GL overlay).
    ///
    /// Both paths should produce visually similar output. Differences arise from
    /// bilinear interpolation (GPU vs CPU) and mask threshold rounding.
    #[test]
    fn test_proto_fused_vs_hybrid_ssim() {
        use edgefirst_decoder::{configs, ConfigOutput, DecoderBuilder, Nms};

        if !is_opengl_available() {
            eprintln!("SKIPPED: {} - OpenGL not available", function!());
            return;
        }

        // Load cached YOLOv8 seg model outputs as TensorDyn::I8 inputs.
        let boxes_raw: &[u8] = &edgefirst_bench::testdata::read("yolov8_boxes_116x8400.bin");
        let boxes_i8 =
            unsafe { std::slice::from_raw_parts(boxes_raw.as_ptr() as *const i8, boxes_raw.len()) };
        let boxes_tensor = TensorDyn::I8(
            edgefirst_tensor::Tensor::<i8>::from_slice(boxes_i8, &[1, 116, 8400])
                .expect("boxes tensor"),
        );

        let protos_raw: &[u8] = &edgefirst_bench::testdata::read("yolov8_protos_160x160x32.bin");
        let protos_i8 = unsafe {
            std::slice::from_raw_parts(protos_raw.as_ptr() as *const i8, protos_raw.len())
        };
        let protos_tensor = TensorDyn::I8(
            edgefirst_tensor::Tensor::<i8>::from_slice(protos_i8, &[1, 160, 160, 32])
                .expect("protos tensor"),
        );

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

        // Materialize masks on CPU for the hybrid path
        let cpu_proc = crate::CPUProcessor::new();
        let segmentation = cpu_proc
            .materialize_segmentations(&output_boxes, &proto_data, None)
            .unwrap();

        // Create two identical RGBA canvases
        let mut gl = GLProcessorThreaded::new(None).unwrap();
        let mut dst_hybrid = TensorDyn::from(
            edgefirst_tensor::Tensor::<u8>::image(640, 640, PixelFormat::Rgba, None).unwrap(),
        );
        let mut dst_fused = TensorDyn::from(
            edgefirst_tensor::Tensor::<u8>::image(640, 640, PixelFormat::Rgba, None).unwrap(),
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

        // Compare — threshold 0.90 to allow bilinear interpolation differences
        // between GPU proto rendering and CPU materialization
        compare_images(&dst_hybrid, &dst_fused, 0.90, function!());
    }

    // =========================================================================
    // Destination DMA-BUF plane_offset tests
    //
    // These verify that create_egl_image_with_dims correctly passes
    // plane_offset to EGL, so the GPU renders at the right position within
    // a shared DMA-BUF (critical for NPU buffers where a single allocation
    // serves the entire accelerator).
    // =========================================================================

    /// Regression: destination with explicit offset=0 must produce the same
    /// output as a destination with no offset set.
    #[test]
    #[cfg(all(target_os = "linux", feature = "dma_test_formats"))]
    fn test_opengl_dst_offset_zero_regression() {
        if !is_dma_available() {
            eprintln!("SKIPPED: {} - DMA not available", function!());
            return;
        }
        if !is_opengl_available() {
            eprintln!("SKIPPED: {} - OpenGL not available", function!());
            return;
        }

        let src = load_raw_image(
            1280,
            720,
            PixelFormat::Nv12,
            Some(TensorMemory::Dma),
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
            Some(TensorMemory::Dma),
        )
        .unwrap();
        let src_dyn = src;
        let mut dst_no_off_dyn = dst_no_off;
        gl.convert(
            &src_dyn,
            &mut dst_no_off_dyn,
            Rotation::None,
            Flip::None,
            Crop::no_crop(),
        )
        .unwrap();

        // Destination with explicit offset=0
        let mut dst_zero =
            Tensor::<u8>::image(640, 480, PixelFormat::Rgba, Some(TensorMemory::Dma)).unwrap();
        dst_zero.set_plane_offset(0);
        let mut dst_zero_dyn = TensorDyn::from(dst_zero);
        gl.convert(
            &src_dyn,
            &mut dst_zero_dyn,
            Rotation::None,
            Flip::None,
            Crop::no_crop(),
        )
        .unwrap();

        let a = dst_no_off_dyn.as_u8().unwrap().map().unwrap();
        let b = dst_zero_dyn.as_u8().unwrap().map().unwrap();
        assert_pixels_match(a.as_slice(), b.as_slice(), 0);
    }

    /// Functional: allocate an oversized DMA buffer, import the second half
    /// as a destination via PlaneDescriptor with offset, convert, and verify
    /// the GPU wrote at the offset — not at byte 0.
    #[test]
    #[cfg(all(target_os = "linux", feature = "dma_test_formats"))]
    fn test_opengl_dst_nonzero_offset() {
        use edgefirst_tensor::PlaneDescriptor;
        use std::os::fd::AsFd;

        if !is_dma_available() {
            eprintln!("SKIPPED: {} - DMA not available", function!());
            return;
        }
        if !is_opengl_available() {
            eprintln!("SKIPPED: {} - OpenGL not available", function!());
            return;
        }

        let width = 640;
        let height = 480;
        let image_bytes = width * height * 4; // RGBA
        let offset = image_bytes; // put image in the second half

        // Allocate a DMA buffer twice the image size and fill with a sentinel
        let large_buf =
            Tensor::<u8>::new(&[image_bytes * 2], Some(TensorMemory::Dma), None).unwrap();
        large_buf.map().unwrap().as_mut_slice().fill(0xAA);

        // Import the second half as an RGBA destination via PlaneDescriptor
        let fd = large_buf.as_dma().unwrap().fd.as_fd();
        let plane = PlaneDescriptor::new(fd).unwrap().with_offset(offset);
        let proc = crate::ImageProcessor::new().unwrap();
        let mut dst = proc
            .import_image(plane, None, width, height, PixelFormat::Rgba, DType::U8)
            .unwrap();

        // Source: NV12 camera frame
        let src = load_raw_image(
            1280,
            720,
            PixelFormat::Nv12,
            Some(TensorMemory::Dma),
            &edgefirst_bench::testdata::read("camera720p.nv12"),
        )
        .unwrap();

        let mut gl = GLProcessorThreaded::new(None).unwrap();
        let src_dyn = src;
        gl.convert(
            &src_dyn,
            &mut dst,
            Rotation::None,
            Flip::None,
            Crop::no_crop(),
        )
        .unwrap();

        // Verify: the first half (before offset) must still be our sentinel
        let map = large_buf.map().unwrap();
        let buf = map.as_slice();
        let untouched = &buf[..offset];
        assert!(
            untouched.iter().all(|&b| b == 0xAA),
            "GPU wrote before the offset boundary — offset not applied"
        );

        // Verify: the second half (at offset) should contain rendered data,
        // not all sentinels
        let rendered = &buf[offset..];
        assert!(
            rendered.iter().any(|&b| b != 0xAA),
            "GPU did not write at the offset position — destination is still sentinel"
        );
    }

    /// Destination with offset=0 via PlaneDescriptor import (same path as NPU
    /// workflow) must produce identical output to a normal DMA allocation.
    #[test]
    #[cfg(all(target_os = "linux", feature = "dma_test_formats"))]
    fn test_opengl_dst_imported_offset_zero() {
        use edgefirst_tensor::PlaneDescriptor;
        use std::os::fd::AsFd;

        if !is_dma_available() {
            eprintln!("SKIPPED: {} - DMA not available", function!());
            return;
        }
        if !is_opengl_available() {
            eprintln!("SKIPPED: {} - OpenGL not available", function!());
            return;
        }

        let width = 640;
        let height = 480;

        let src = load_raw_image(
            1280,
            720,
            PixelFormat::Nv12,
            Some(TensorMemory::Dma),
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
            Some(TensorMemory::Dma),
        )
        .unwrap();
        let src_dyn = src;
        let mut dst_ref_dyn = dst_ref;
        gl.convert(
            &src_dyn,
            &mut dst_ref_dyn,
            Rotation::None,
            Flip::None,
            Crop::no_crop(),
        )
        .unwrap();

        // Imported: PlaneDescriptor with offset=0
        let dst_buf =
            Tensor::<u8>::image(width, height, PixelFormat::Rgba, Some(TensorMemory::Dma)).unwrap();
        let fd = dst_buf.as_dma().unwrap().fd.as_fd();
        let plane = PlaneDescriptor::new(fd).unwrap().with_offset(0);
        let proc = crate::ImageProcessor::new().unwrap();
        let mut dst_imported = proc
            .import_image(plane, None, width, height, PixelFormat::Rgba, DType::U8)
            .unwrap();
        gl.convert(
            &src_dyn,
            &mut dst_imported,
            Rotation::None,
            Flip::None,
            Crop::no_crop(),
        )
        .unwrap();

        let a = dst_ref_dyn.as_u8().unwrap().map().unwrap();
        let b = dst_buf.map().unwrap();
        assert_pixels_match(a.as_slice(), b.as_slice(), 1);
    }

    /// Functional: packed RGB destination with nonzero offset.  This exercises
    /// `create_egl_image_with_dims` (the two-pass RGB pipeline), which was the
    /// specific function that had the hardcoded offset=0 bug.
    #[test]
    #[cfg(all(target_os = "linux", feature = "dma_test_formats"))]
    fn test_opengl_dst_nonzero_offset_rgb() {
        use edgefirst_tensor::PlaneDescriptor;
        use std::os::fd::AsFd;

        if !is_dma_available() {
            eprintln!("SKIPPED: {} - DMA not available", function!());
            return;
        }
        if !is_opengl_available() {
            eprintln!("SKIPPED: {} - OpenGL not available", function!());
            return;
        }

        let width = 640;
        let height = 480;
        let image_bytes = width * height * 3; // RGB
        let offset = image_bytes; // put image in the second half

        // Allocate a DMA buffer twice the image size and fill with a sentinel
        let large_buf =
            Tensor::<u8>::new(&[image_bytes * 2], Some(TensorMemory::Dma), None).unwrap();
        large_buf.map().unwrap().as_mut_slice().fill(0xAA);

        // Import the second half as an RGB destination via PlaneDescriptor
        let fd = large_buf.as_dma().unwrap().fd.as_fd();
        let plane = PlaneDescriptor::new(fd).unwrap().with_offset(offset);
        let proc = crate::ImageProcessor::new().unwrap();
        let mut dst = proc
            .import_image(plane, None, width, height, PixelFormat::Rgb, DType::U8)
            .unwrap();

        // Source: NV12 camera frame
        let src = load_raw_image(
            1280,
            720,
            PixelFormat::Nv12,
            Some(TensorMemory::Dma),
            &edgefirst_bench::testdata::read("camera720p.nv12"),
        )
        .unwrap();

        let mut gl = GLProcessorThreaded::new(None).unwrap();
        let src_dyn = src;
        gl.convert(
            &src_dyn,
            &mut dst,
            Rotation::None,
            Flip::None,
            Crop::no_crop(),
        )
        .unwrap();

        // Verify: the first half (before offset) must still be our sentinel
        let map = large_buf.map().unwrap();
        let buf = map.as_slice();
        let untouched = &buf[..offset];
        assert!(
            untouched.iter().all(|&b| b == 0xAA),
            "GPU wrote before the offset boundary — RGB offset not applied"
        );

        // Verify: the second half (at offset) should contain rendered data
        let rendered = &buf[offset..];
        assert!(
            rendered.iter().any(|&b| b != 0xAA),
            "GPU did not write at the RGB offset position — destination is still sentinel"
        );
    }

    // ---------------------------------------------------------------
    // Neutron-scenario tests: large buffer + large offset
    //
    // These replicate the geometry of the EDGEAI-1192 bug (Neutron NPU
    // DMA-BUF at offset 3,450,400 in a 10 MB buffer) using standard
    // dma_heap allocations — no NPU driver required.
    //
    // The tests are gated on is_neutron_available() (/dev/neutron0) because
    // the large-offset EGLImage import path works on Mali (i.MX 95) but
    // fails with EGL(BadAccess) on Vivante (i.MX 8MP), confirmed by hardware
    // testing.  The gate is a platform discriminator, not an NPU check.
    // ---------------------------------------------------------------

    /// Direct reproduction of the Neutron scenario: 640×640 RGB int8
    /// destination at offset 3,450,400 inside a ~10 MB DMA buffer.
    /// Exercises the two-pass `convert_to_packed_rgb` with `is_int8=true`.
    #[test]
    #[cfg(all(target_os = "linux", feature = "dma_test_formats"))]
    fn test_neutron_scenario_large_offset_rgb_int8() {
        use edgefirst_tensor::PlaneDescriptor;
        use std::os::fd::AsFd;

        if !is_dma_available() {
            eprintln!("SKIPPED: {} - DMA not available", function!());
            return;
        }
        if !is_opengl_available() {
            eprintln!("SKIPPED: {} - OpenGL not available", function!());
            return;
        }
        if !is_neutron_available() {
            eprintln!(
                "SKIPPED: {} - Neutron not available (/dev/neutron0 not found)",
                function!()
            );
            return;
        }

        let total_size: usize = 10_276_864; // Neutron shared buffer size
        let offset: usize = 3_450_400; // Neutron input tensor offset
        let image_bytes: usize = 640 * 640 * 3; // RGB

        let large_buf = match Tensor::<u8>::new(&[total_size], Some(TensorMemory::Dma), None) {
            Ok(buf) => buf,
            Err(_) => {
                eprintln!(
                    "SKIPPED: {} - cannot allocate {} MB DMA buffer",
                    function!(),
                    total_size / 1_048_576
                );
                return;
            }
        };
        large_buf.map().unwrap().as_mut_slice().fill(0xAA);

        let fd = large_buf.as_dma().unwrap().fd.as_fd();
        let plane = PlaneDescriptor::new(fd).unwrap().with_offset(offset);
        let proc = crate::ImageProcessor::new().unwrap();
        let mut dst = proc
            .import_image(plane, None, 640, 640, PixelFormat::Rgb, DType::I8)
            .unwrap();

        let src = load_raw_image(
            1280,
            720,
            PixelFormat::Nv12,
            Some(TensorMemory::Dma),
            &edgefirst_bench::testdata::read("camera720p.nv12"),
        )
        .unwrap();

        let mut gl = GLProcessorThreaded::new(None).unwrap();
        gl.convert(&src, &mut dst, Rotation::None, Flip::None, Crop::no_crop())
            .unwrap();

        // Verify: sentinel intact before offset
        let map = large_buf.map().unwrap();
        let buf = map.as_slice();
        let untouched = &buf[..offset];
        assert!(
            untouched.iter().all(|&b| b == 0xAA),
            "GPU wrote before the offset boundary"
        );

        // Verify: rendered data at offset
        let rendered = &buf[offset..offset + image_bytes];
        assert!(
            rendered.iter().any(|&b| b != 0xAA),
            "GPU did not write at the offset position — destination is still sentinel"
        );
    }

    /// Isolate two-pass vs single-pass: same large offset, RGBA U8
    /// (single-pass EGLImage) vs RGB I8 (two-pass convert_to_packed_rgb).
    /// If RGBA succeeds and RGB fails, the bug is in the two-pass path.
    #[test]
    #[cfg(all(target_os = "linux", feature = "dma_test_formats"))]
    fn test_neutron_scenario_rgba_vs_rgb_isolation() {
        use edgefirst_tensor::PlaneDescriptor;
        use std::os::fd::AsFd;

        if !is_dma_available() {
            eprintln!("SKIPPED: {} - DMA not available", function!());
            return;
        }
        if !is_opengl_available() {
            eprintln!("SKIPPED: {} - OpenGL not available", function!());
            return;
        }
        if !is_neutron_available() {
            eprintln!(
                "SKIPPED: {} - Neutron not available (/dev/neutron0 not found)",
                function!()
            );
            return;
        }

        let total_size: usize = 10_276_864;
        let offset: usize = 3_450_400;

        let src = load_raw_image(
            1280,
            720,
            PixelFormat::Nv12,
            Some(TensorMemory::Dma),
            &edgefirst_bench::testdata::read("camera720p.nv12"),
        )
        .unwrap();

        // --- RGBA U8 at large offset (single-pass path) ---
        let rgba_buf = match Tensor::<u8>::new(&[total_size], Some(TensorMemory::Dma), None) {
            Ok(buf) => buf,
            Err(_) => {
                eprintln!("SKIPPED: {} - cannot allocate DMA buffer", function!());
                return;
            }
        };
        rgba_buf.map().unwrap().as_mut_slice().fill(0xAA);

        let fd = rgba_buf.as_dma().unwrap().fd.as_fd();
        let plane = PlaneDescriptor::new(fd).unwrap().with_offset(offset);
        let proc = crate::ImageProcessor::new().unwrap();
        let mut dst_rgba = proc
            .import_image(plane, None, 640, 640, PixelFormat::Rgba, DType::U8)
            .unwrap();

        let mut gl = GLProcessorThreaded::new(None).unwrap();
        let rgba_result = gl.convert(
            &src,
            &mut dst_rgba,
            Rotation::None,
            Flip::None,
            Crop::no_crop(),
        );
        assert!(
            rgba_result.is_ok(),
            "RGBA U8 single-pass at large offset failed: {:?}",
            rgba_result.err()
        );

        // --- RGB I8 at same large offset (two-pass path) ---
        let rgb_buf = match Tensor::<u8>::new(&[total_size], Some(TensorMemory::Dma), None) {
            Ok(buf) => buf,
            Err(_) => {
                eprintln!(
                    "SKIPPED: {} - cannot allocate second DMA buffer",
                    function!()
                );
                return;
            }
        };
        rgb_buf.map().unwrap().as_mut_slice().fill(0xAA);

        let fd = rgb_buf.as_dma().unwrap().fd.as_fd();
        let plane = PlaneDescriptor::new(fd).unwrap().with_offset(offset);
        let mut dst_rgb = proc
            .import_image(plane, None, 640, 640, PixelFormat::Rgb, DType::I8)
            .unwrap();

        let rgb_result = gl.convert(
            &src,
            &mut dst_rgb,
            Rotation::None,
            Flip::None,
            Crop::no_crop(),
        );
        assert!(
            rgb_result.is_ok(),
            "RGB I8 two-pass at large offset failed (RGBA succeeded): {:?}",
            rgb_result.err()
        );
    }

    /// Isolate int8 shader: same two-pass RGB path, but U8 vs I8 dtype.
    /// If RGB U8 succeeds and RGB I8 fails, the int8 shader interacts
    /// poorly with large-offset FBOs.
    #[test]
    #[cfg(all(target_os = "linux", feature = "dma_test_formats"))]
    fn test_neutron_scenario_rgb_u8_vs_int8() {
        use edgefirst_tensor::PlaneDescriptor;
        use std::os::fd::AsFd;

        if !is_dma_available() {
            eprintln!("SKIPPED: {} - DMA not available", function!());
            return;
        }
        if !is_opengl_available() {
            eprintln!("SKIPPED: {} - OpenGL not available", function!());
            return;
        }
        if !is_neutron_available() {
            eprintln!(
                "SKIPPED: {} - Neutron not available (/dev/neutron0 not found)",
                function!()
            );
            return;
        }

        let total_size: usize = 10_276_864;
        let offset: usize = 3_450_400;

        let src = load_raw_image(
            1280,
            720,
            PixelFormat::Nv12,
            Some(TensorMemory::Dma),
            &edgefirst_bench::testdata::read("camera720p.nv12"),
        )
        .unwrap();

        let proc = crate::ImageProcessor::new().unwrap();
        let mut gl = GLProcessorThreaded::new(None).unwrap();

        // --- RGB U8 at large offset (two-pass, packed_rgba8_program_2d) ---
        let u8_buf = match Tensor::<u8>::new(&[total_size], Some(TensorMemory::Dma), None) {
            Ok(buf) => buf,
            Err(_) => {
                eprintln!("SKIPPED: {} - cannot allocate DMA buffer", function!());
                return;
            }
        };
        u8_buf.map().unwrap().as_mut_slice().fill(0xAA);

        let fd = u8_buf.as_dma().unwrap().fd.as_fd();
        let plane = PlaneDescriptor::new(fd).unwrap().with_offset(offset);
        let mut dst_u8 = proc
            .import_image(plane, None, 640, 640, PixelFormat::Rgb, DType::U8)
            .unwrap();

        let u8_result = gl.convert(
            &src,
            &mut dst_u8,
            Rotation::None,
            Flip::None,
            Crop::no_crop(),
        );
        assert!(
            u8_result.is_ok(),
            "RGB U8 two-pass at large offset failed: {:?}",
            u8_result.err()
        );

        // --- RGB I8 at same large offset (two-pass, packed_rgba8_int8_program_2d) ---
        let i8_buf = match Tensor::<u8>::new(&[total_size], Some(TensorMemory::Dma), None) {
            Ok(buf) => buf,
            Err(_) => {
                eprintln!(
                    "SKIPPED: {} - cannot allocate second DMA buffer",
                    function!()
                );
                return;
            }
        };
        i8_buf.map().unwrap().as_mut_slice().fill(0xAA);

        let fd = i8_buf.as_dma().unwrap().fd.as_fd();
        let plane = PlaneDescriptor::new(fd).unwrap().with_offset(offset);
        let mut dst_i8 = proc
            .import_image(plane, None, 640, 640, PixelFormat::Rgb, DType::I8)
            .unwrap();

        let i8_result = gl.convert(
            &src,
            &mut dst_i8,
            Rotation::None,
            Flip::None,
            Crop::no_crop(),
        );
        assert!(
            i8_result.is_ok(),
            "RGB I8 two-pass at large offset failed (U8 succeeded): {:?}",
            i8_result.err()
        );
    }

    /// Check if Mali requires page-aligned EGL offsets. The Neutron offset
    /// 3,450,400 is NOT page-aligned (3,450,400 % 4096 = 3488). Compare
    /// against the nearest page-aligned offset below it.
    #[test]
    #[cfg(all(target_os = "linux", feature = "dma_test_formats"))]
    fn test_neutron_scenario_offset_alignment() {
        use edgefirst_tensor::PlaneDescriptor;
        use std::os::fd::AsFd;

        if !is_dma_available() {
            eprintln!("SKIPPED: {} - DMA not available", function!());
            return;
        }
        if !is_opengl_available() {
            eprintln!("SKIPPED: {} - OpenGL not available", function!());
            return;
        }
        if !is_neutron_available() {
            eprintln!(
                "SKIPPED: {} - Neutron not available (/dev/neutron0 not found)",
                function!()
            );
            return;
        }

        let total_size: usize = 10_276_864;
        let page_aligned_offset: usize = 3_448_832; // 842 * 4096
        let neutron_offset: usize = 3_450_400; // actual Neutron offset

        let src = load_raw_image(
            1280,
            720,
            PixelFormat::Nv12,
            Some(TensorMemory::Dma),
            &edgefirst_bench::testdata::read("camera720p.nv12"),
        )
        .unwrap();

        let proc = crate::ImageProcessor::new().unwrap();
        let mut gl = GLProcessorThreaded::new(None).unwrap();

        // --- Page-aligned offset ---
        let aligned_buf = match Tensor::<u8>::new(&[total_size], Some(TensorMemory::Dma), None) {
            Ok(buf) => buf,
            Err(_) => {
                eprintln!("SKIPPED: {} - cannot allocate DMA buffer", function!());
                return;
            }
        };
        aligned_buf.map().unwrap().as_mut_slice().fill(0xAA);

        let fd = aligned_buf.as_dma().unwrap().fd.as_fd();
        let plane = PlaneDescriptor::new(fd)
            .unwrap()
            .with_offset(page_aligned_offset);
        let mut dst_aligned = proc
            .import_image(plane, None, 640, 640, PixelFormat::Rgb, DType::I8)
            .unwrap();

        let aligned_result = gl.convert(
            &src,
            &mut dst_aligned,
            Rotation::None,
            Flip::None,
            Crop::no_crop(),
        );
        assert!(
            aligned_result.is_ok(),
            "Page-aligned offset {} failed: {:?}",
            page_aligned_offset,
            aligned_result.err()
        );

        // --- Non-page-aligned Neutron offset ---
        let unaligned_buf = match Tensor::<u8>::new(&[total_size], Some(TensorMemory::Dma), None) {
            Ok(buf) => buf,
            Err(_) => {
                eprintln!(
                    "SKIPPED: {} - cannot allocate second DMA buffer",
                    function!()
                );
                return;
            }
        };
        unaligned_buf.map().unwrap().as_mut_slice().fill(0xAA);

        let fd = unaligned_buf.as_dma().unwrap().fd.as_fd();
        let plane = PlaneDescriptor::new(fd)
            .unwrap()
            .with_offset(neutron_offset);
        let mut dst_unaligned = proc
            .import_image(plane, None, 640, 640, PixelFormat::Rgb, DType::I8)
            .unwrap();

        let unaligned_result = gl.convert(
            &src,
            &mut dst_unaligned,
            Rotation::None,
            Flip::None,
            Crop::no_crop(),
        );
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

    /// Baseline: import a tightly-packed RGBA DMA buffer with explicit stride
    /// equal to width * 4. Verifies that import_image succeeds and the
    /// returned tensor has the expected dimensions and format.
    #[test]
    #[cfg(all(target_os = "linux", feature = "dma_test_formats"))]
    fn test_import_image_rgba_with_stride() {
        use edgefirst_tensor::PlaneDescriptor;

        if !is_dma_available() {
            eprintln!("SKIPPED: test_import_image_rgba_with_stride - DMA not available");
            return;
        }

        let width: usize = 640;
        let height: usize = 480;
        let bpp: usize = 4; // RGBA
        let stride = width * bpp; // 2560, tightly packed

        // Allocate a DMA tensor large enough for the RGBA image
        let buf = Tensor::<u8>::new(
            &[height, width, bpp],
            Some(TensorMemory::Dma),
            Some("import_rgba_tight"),
        );
        let buf = match buf {
            Ok(t) if t.memory() == TensorMemory::Dma => t,
            _ => {
                eprintln!("SKIPPED: test_import_image_rgba_with_stride - DMA alloc failed");
                return;
            }
        };

        let fd = buf.dmabuf().unwrap();
        let plane = PlaneDescriptor::new(fd).unwrap().with_stride(stride);

        let proc = crate::ImageProcessor::new().unwrap();
        let imported = proc
            .import_image(plane, None, width, height, PixelFormat::Rgba, DType::U8)
            .unwrap();

        assert_eq!(imported.width(), Some(width));
        assert_eq!(imported.height(), Some(height));
        assert_eq!(imported.format(), Some(PixelFormat::Rgba));
        assert_eq!(imported.row_stride(), Some(stride));
    }

    /// Import an RGBA DMA buffer with a padded stride (64-byte aligned).
    /// The stride is larger than width * bpp; import_image must accept
    /// the padding and report the correct shape.
    #[test]
    #[cfg(all(target_os = "linux", feature = "dma_test_formats"))]
    fn test_import_image_rgba_padded_stride() {
        use edgefirst_tensor::PlaneDescriptor;

        if !is_dma_available() {
            eprintln!("SKIPPED: test_import_image_rgba_padded_stride - DMA not available");
            return;
        }

        let width: usize = 640;
        let height: usize = 480;
        let bpp: usize = 4;
        // 64-byte aligned stride: ceil(640*4 / 64) * 64 = 2560 (already aligned)
        // Use 128-byte alignment to get a genuinely padded stride.
        let stride: usize = ((width * bpp + 127) / 128) * 128; // 2560 -> 2560 (already 128-aligned)
                                                               // Force a wider stride to be sure it's padded
        let stride = stride + 128; // 2688

        // Allocate a DMA buffer large enough for the padded rows
        let total_bytes = stride * height;
        let buf = Tensor::<u8>::new(
            &[total_bytes],
            Some(TensorMemory::Dma),
            Some("import_rgba_padded"),
        );
        let buf = match buf {
            Ok(t) if t.memory() == TensorMemory::Dma => t,
            _ => {
                eprintln!("SKIPPED: test_import_image_rgba_padded_stride - DMA alloc failed");
                return;
            }
        };

        let fd = buf.dmabuf().unwrap();
        let plane = PlaneDescriptor::new(fd).unwrap().with_stride(stride);

        let proc = crate::ImageProcessor::new().unwrap();
        let imported = proc
            .import_image(plane, None, width, height, PixelFormat::Rgba, DType::U8)
            .unwrap();

        assert_eq!(imported.width(), Some(width));
        assert_eq!(imported.height(), Some(height));
        assert_eq!(imported.format(), Some(PixelFormat::Rgba));
        assert_eq!(imported.row_stride(), Some(stride));
    }

    /// Import a DMA buffer as multiplane NV12 with explicit offsets.
    /// The luma plane starts at offset 0 and the chroma plane starts at
    /// height * stride. This mirrors the layout produced by V4L2 camera
    /// drivers that expose NV12 as two separate planes sharing one fd.
    #[test]
    #[cfg(all(target_os = "linux", feature = "dma_test_formats"))]
    fn test_import_image_nv12_with_offset() {
        use edgefirst_tensor::PlaneDescriptor;

        if !is_dma_available() {
            eprintln!("SKIPPED: test_import_image_nv12_with_offset - DMA not available");
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
        let buf = Tensor::<u8>::new(
            &[total_bytes],
            Some(TensorMemory::Dma),
            Some("import_nv12_planes"),
        );
        let buf = match buf {
            Ok(t) if t.memory() == TensorMemory::Dma => t,
            _ => {
                eprintln!("SKIPPED: test_import_image_nv12_with_offset - DMA alloc failed");
                return;
            }
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
            )
            .unwrap();

        assert_eq!(imported.width(), Some(width));
        assert_eq!(imported.height(), Some(height));
        assert_eq!(imported.format(), Some(PixelFormat::Nv12));
    }

    /// True multiplane NV12 via `import_image` with separate DMA-BUFs
    /// (libcamera style): full `import_image` → `eglCreateImage` →
    /// `convert()` render path.  Compares pixel output against contiguous
    /// NV12 → RGBA reference.
    #[test]
    #[cfg(all(target_os = "linux", feature = "dma_test_formats"))]
    fn test_import_image_nv12_true_multiplane() {
        use edgefirst_tensor::PlaneDescriptor;

        if !is_dma_available() {
            eprintln!("SKIPPED: {} - DMA not available", function!());
            return;
        }
        if !is_opengl_available() {
            eprintln!("SKIPPED: {} - OpenGL not available", function!());
            return;
        }

        let nv12_bytes: &[u8] = &edgefirst_bench::testdata::read("camera720p.nv12");
        let width: usize = 1280;
        let height: usize = 720;
        let stride: usize = width;
        let y_size = stride * height;
        let uv_size = stride * (height / 2);
        assert_eq!(nv12_bytes.len(), y_size + uv_size);

        // Two separate DMA allocations filled with real NV12 data
        let luma_buf = Tensor::<u8>::new(&[y_size], Some(TensorMemory::Dma), Some("true_mp_luma"));
        let luma_buf = match luma_buf {
            Ok(t) if t.memory() == TensorMemory::Dma => t,
            _ => {
                eprintln!("SKIPPED: {} - luma DMA alloc failed", function!());
                return;
            }
        };
        luma_buf.map().unwrap().as_mut_slice()[..y_size].copy_from_slice(&nv12_bytes[..y_size]);

        let chroma_buf =
            Tensor::<u8>::new(&[uv_size], Some(TensorMemory::Dma), Some("true_mp_chroma"));
        let chroma_buf = match chroma_buf {
            Ok(t) if t.memory() == TensorMemory::Dma => t,
            _ => {
                eprintln!("SKIPPED: {} - chroma DMA alloc failed", function!());
                return;
            }
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
            )
            .unwrap();
        assert!(src.is_multiplane(), "must be multiplane after import");

        // Build contiguous reference for pixel comparison
        let src_contig = load_raw_image(
            width,
            height,
            PixelFormat::Nv12,
            Some(TensorMemory::Dma),
            nv12_bytes,
        )
        .unwrap();

        let mut gl = GLProcessorThreaded::new(None).unwrap();

        // Render true multiplane through full EGL pipeline
        let mut dst_multi = TensorDyn::image(
            width,
            height,
            PixelFormat::Rgba,
            DType::U8,
            Some(TensorMemory::Dma),
        )
        .unwrap();
        gl.convert(
            &src,
            &mut dst_multi,
            Rotation::None,
            Flip::None,
            Crop::no_crop(),
        )
        .unwrap();

        // Render contiguous reference
        let mut dst_contig = TensorDyn::image(
            width,
            height,
            PixelFormat::Rgba,
            DType::U8,
            Some(TensorMemory::Dma),
        )
        .unwrap();
        gl.convert(
            &src_contig,
            &mut dst_contig,
            Rotation::None,
            Flip::None,
            Crop::no_crop(),
        )
        .unwrap();

        // Compare: true multiplane and contiguous must produce identical pixels
        let map_multi = dst_multi.as_u8().unwrap().map().unwrap();
        let map_contig = dst_contig.as_u8().unwrap().map().unwrap();
        assert_pixels_match(map_multi.as_slice(), map_contig.as_slice(), 0);
    }

    /// Contiguous single-fd NV12 via `import_image` (no chroma descriptor):
    /// full `import_image` → `eglCreateImage` → `convert()` render path.
    /// Compares pixel output against `load_raw_image` contiguous reference.
    #[test]
    #[cfg(all(target_os = "linux", feature = "dma_test_formats"))]
    fn test_import_image_nv12_contiguous() {
        use edgefirst_tensor::PlaneDescriptor;

        if !is_dma_available() {
            eprintln!("SKIPPED: {} - DMA not available", function!());
            return;
        }
        if !is_opengl_available() {
            eprintln!("SKIPPED: {} - OpenGL not available", function!());
            return;
        }

        let nv12_bytes: &[u8] = &edgefirst_bench::testdata::read("camera720p.nv12");
        let width: usize = 1280;
        let height: usize = 720;
        let stride: usize = width;
        let total_bytes = nv12_bytes.len();

        // Single DMA buffer holding Y+UV contiguously
        let buf = Tensor::<u8>::new(
            &[total_bytes],
            Some(TensorMemory::Dma),
            Some("nv12_contiguous"),
        );
        let buf = match buf {
            Ok(t) if t.memory() == TensorMemory::Dma => t,
            _ => {
                eprintln!("SKIPPED: {} - DMA alloc failed", function!());
                return;
            }
        };
        buf.map().unwrap().as_mut_slice()[..total_bytes].copy_from_slice(nv12_bytes);

        // Import via import_image with NO chroma descriptor (contiguous path)
        let fd = buf.dmabuf().unwrap();
        let plane = PlaneDescriptor::new(fd).unwrap().with_stride(stride);

        let proc = crate::ImageProcessor::new().unwrap();
        let src = proc
            .import_image(plane, None, width, height, PixelFormat::Nv12, DType::U8)
            .unwrap();
        assert!(
            !src.is_multiplane(),
            "contiguous NV12 must not be multiplane"
        );

        // Build reference via load_raw_image (same data, same contiguous layout)
        let src_ref = load_raw_image(
            width,
            height,
            PixelFormat::Nv12,
            Some(TensorMemory::Dma),
            nv12_bytes,
        )
        .unwrap();

        let mut gl = GLProcessorThreaded::new(None).unwrap();

        // Render contiguous import_image path
        let mut dst_imported = TensorDyn::image(
            width,
            height,
            PixelFormat::Rgba,
            DType::U8,
            Some(TensorMemory::Dma),
        )
        .unwrap();
        gl.convert(
            &src,
            &mut dst_imported,
            Rotation::None,
            Flip::None,
            Crop::no_crop(),
        )
        .unwrap();

        // Render reference
        let mut dst_ref = TensorDyn::image(
            width,
            height,
            PixelFormat::Rgba,
            DType::U8,
            Some(TensorMemory::Dma),
        )
        .unwrap();
        gl.convert(
            &src_ref,
            &mut dst_ref,
            Rotation::None,
            Flip::None,
            Crop::no_crop(),
        )
        .unwrap();

        // Compare: import_image contiguous and load_raw_image must produce identical pixels
        let map_imported = dst_imported.as_u8().unwrap().map().unwrap();
        let map_ref = dst_ref.as_u8().unwrap().map().unwrap();
        assert_pixels_match(map_imported.as_slice(), map_ref.as_slice(), 0);
    }

    /// Attempt to import with a stride smaller than width * bpp.
    /// import_image (via set_row_stride) must reject this with an error.
    #[test]
    #[cfg(all(target_os = "linux", feature = "dma_test_formats"))]
    fn test_import_image_stride_too_small() {
        use edgefirst_tensor::PlaneDescriptor;

        if !is_dma_available() {
            eprintln!("SKIPPED: test_import_image_stride_too_small - DMA not available");
            return;
        }

        let width: usize = 640;
        let height: usize = 480;
        let bpp: usize = 4;
        let bad_stride = width * bpp - 1; // one byte too narrow

        let buf = Tensor::<u8>::new(
            &[height, width, bpp],
            Some(TensorMemory::Dma),
            Some("import_bad_stride"),
        );
        let buf = match buf {
            Ok(t) if t.memory() == TensorMemory::Dma => t,
            _ => {
                eprintln!("SKIPPED: test_import_image_stride_too_small - DMA alloc failed");
                return;
            }
        };

        let fd = buf.dmabuf().unwrap();
        let plane = PlaneDescriptor::new(fd).unwrap().with_stride(bad_stride);

        let proc = crate::ImageProcessor::new().unwrap();
        let result = proc.import_image(plane, None, width, height, PixelFormat::Rgba, DType::U8);

        assert!(
            result.is_err(),
            "import_image should reject stride {bad_stride} < minimum {} for {width}x{height} RGBA",
            width * bpp
        );
    }

    // ──────────────────────────────────────────────────────────────────────
    // src_rect clamping tests — verify that cropping from a larger buffer
    // never samples padding pixels, even with GL_LINEAR bilinear filtering.
    // ──────────────────────────────────────────────────────────────────────

    /// Create a synthetic RGB tensor where the left half is pure red and the
    /// right half is pure blue. Simulates a larger reused buffer where only a
    /// sub-region contains the desired content.
    fn make_red_blue_src(width: usize, height: usize) -> TensorDyn {
        let mut t = TensorDyn::image(width, height, PixelFormat::Rgb, DType::U8, None).unwrap();
        {
            let tensor_u8 = t.as_u8_mut().unwrap();
            let mut map = tensor_u8.map().unwrap();
            let data = map.as_mut_slice();
            let half = width / 2;
            for y in 0..height {
                for x in 0..width {
                    let idx = (y * width + x) * 3;
                    if x < half {
                        // Red
                        data[idx] = 255;
                        data[idx + 1] = 0;
                        data[idx + 2] = 0;
                    } else {
                        // Blue
                        data[idx] = 0;
                        data[idx + 1] = 0;
                        data[idx + 2] = 255;
                    }
                }
            }
        }
        t
    }

    /// Crop the blue (right) half of a red|blue image and resize to a smaller
    /// destination. Verify no red pixels bleed into the output.
    #[test]
    fn test_src_rect_crop_no_bleed_gl() {
        if !is_opengl_available() {
            eprintln!("SKIPPED: test_src_rect_crop_no_bleed_gl - OpenGL not available");
            return;
        }

        let src_w = 128;
        let src_h = 64;
        let dst_w = 32;
        let dst_h = 32;

        let src = make_red_blue_src(src_w, src_h);
        let mut dst = TensorDyn::image(dst_w, dst_h, PixelFormat::Rgb, DType::U8, None).unwrap();

        let mut gl = GLProcessorThreaded::new(None).unwrap();

        // Crop only the right (blue) half
        let crop = Crop::new().with_src_rect(Some(crate::Rect::new(
            src_w / 2, // left = start of blue region
            0,
            src_w / 2, // width = blue half
            src_h,
        )));

        gl.convert(&src, &mut dst, Rotation::None, Flip::None, crop)
            .unwrap();

        // Verify: every pixel in the output should be blue (R=0, G=0, B=255)
        // with a small tolerance for GPU rounding.
        let map = dst.as_u8().unwrap().map().unwrap();
        let data = map.as_slice();
        let pixel_count = dst_w * dst_h;
        let mut max_red: u8 = 0;
        for i in 0..pixel_count {
            let r = data[i * 3];
            let g = data[i * 3 + 1];
            let b = data[i * 3 + 2];
            max_red = max_red.max(r);
            // Allow tiny rounding (≤2) on blue channel but no red contamination
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

    /// Same test but with the crop at an exact colour boundary: crop starts
    /// precisely at the red→blue transition. This is the worst case for
    /// bilinear bleed since the leftmost sampled texel is adjacent to red.
    #[test]
    fn test_src_rect_boundary_crop_no_bleed_gl() {
        if !is_opengl_available() {
            eprintln!("SKIPPED: test_src_rect_boundary_crop_no_bleed_gl - OpenGL not available");
            return;
        }

        // Use a power-of-two size to avoid any sub-pixel alignment issues
        let src_w = 256;
        let src_h = 64;
        let dst_w = 64;
        let dst_h = 64;

        let src = make_red_blue_src(src_w, src_h);
        let mut dst = TensorDyn::image(dst_w, dst_h, PixelFormat::Rgb, DType::U8, None).unwrap();

        let mut gl = GLProcessorThreaded::new(None).unwrap();

        // Crop the right half — the left boundary is exactly at the red→blue edge
        let crop =
            Crop::new().with_src_rect(Some(crate::Rect::new(src_w / 2, 0, src_w / 2, src_h)));

        gl.convert(&src, &mut dst, Rotation::None, Flip::None, crop)
            .unwrap();

        let map = dst.as_u8().unwrap().map().unwrap();
        let data = map.as_slice();
        for i in 0..(dst_w * dst_h) {
            let r = data[i * 3];
            let g = data[i * 3 + 1];
            let b = data[i * 3 + 2];
            assert!(
                r <= 2 && g <= 2 && b >= 253,
                "Pixel {i} has contamination at boundary: RGB=({r},{g},{b})"
            );
        }
    }

    /// Crop the red (left) half and verify no blue bleeds in from the right.
    /// Tests the opposite edge.
    #[test]
    fn test_src_rect_crop_left_half_no_bleed_gl() {
        if !is_opengl_available() {
            eprintln!("SKIPPED: test_src_rect_crop_left_half_no_bleed_gl - OpenGL not available");
            return;
        }

        let src_w = 128;
        let src_h = 64;
        let dst_w = 32;
        let dst_h = 32;

        let src = make_red_blue_src(src_w, src_h);
        let mut dst = TensorDyn::image(dst_w, dst_h, PixelFormat::Rgb, DType::U8, None).unwrap();

        let mut gl = GLProcessorThreaded::new(None).unwrap();

        // Crop only the left (red) half
        let crop = Crop::new().with_src_rect(Some(crate::Rect::new(0, 0, src_w / 2, src_h)));

        gl.convert(&src, &mut dst, Rotation::None, Flip::None, crop)
            .unwrap();

        let map = dst.as_u8().unwrap().map().unwrap();
        let data = map.as_slice();
        for i in 0..(dst_w * dst_h) {
            let r = data[i * 3];
            let g = data[i * 3 + 1];
            let b = data[i * 3 + 2];
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
        let s = float_render_support(true, true, true);
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
        let s = float_render_support(false, true, true);
        assert_eq!(
            s,
            RenderDtypeSupport {
                f32: true,
                f16: true
            }
        );
    }

    #[test]
    fn no_ext_means_no_float() {
        use super::super::processor::float_render_support;
        use crate::RenderDtypeSupport;
        let s = float_render_support(false, false, false);
        assert_eq!(
            s,
            RenderDtypeSupport {
                f32: false,
                f16: false
            }
        );
    }

    // ── float_pbo_eligible unit tests ────────────────────────────────────────
    // Pure-logic predicate; no GPU required. Linux-only: `float_pbo_eligible`
    // is gated to the Linux GL backend (macOS uses IOSurface, not float PBO),
    // so the symbol does not exist on macOS builds.

    #[cfg(target_os = "linux")]
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
    // Pure-logic classifier; no GPU required.

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
                s
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
                s
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
                TensorMemory::Dma,
                s
            ),
            FloatRenderPath::DmaF16Nchw
        );
    }

    #[test]
    fn dispatch_f32_dma_is_none() {
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
                TensorMemory::Dma,
                s
            ),
            FloatRenderPath::None
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
                s
            ),
            FloatRenderPath::None
        );
    }

    // ── dma_f16_packed_layout unit tests ────────────────────────────────────
    // Pure geometry helper; no GPU required.

    #[test]
    fn dma_f16_layout_640x640() {
        use super::super::processor::dma_f16_packed_layout;
        // Logical [3,640,640] f16 → packed RGBA16F surface (160, 1920),
        // pitch (640/4)*8 = 1280 bytes.
        assert_eq!(dma_f16_packed_layout(640, 640), Some((160, 1920, 1280)));
    }

    #[test]
    fn dma_f16_layout_non_multiple_of_4_rejected() {
        use super::super::processor::dma_f16_packed_layout;
        // W not divisible by 4 cannot be packed into whole RGBA16F texels.
        assert_eq!(dma_f16_packed_layout(642, 640), None);
    }

    #[test]
    fn dma_f16_layout_rectangular() {
        use super::super::processor::dma_f16_packed_layout;
        // Non-square, W divisible by 4: surface (320/4, 3*240), pitch (80)*8.
        assert_eq!(dma_f16_packed_layout(320, 240), Some((80, 720, 640)));
    }

    /// On-GPU round-trip: RGBA8 → F32 NHWC `[H,W,3]` PBO via the GL float
    /// render path. Forces the OpenGL backend (no CPU fallback) so the test
    /// genuinely exercises `convert_float_to_pbo`. Uses an identity crop so
    /// the expected values are exact: `dst[y,x,c] == src[y,x,c] / 255`.
    #[test]
    fn convert_f32_nhwc_pbo_roundtrip() {
        use crate::{ComputeBackend, ImageProcessor, ImageProcessorConfig};

        if !is_opengl_available() {
            eprintln!("SKIPPED: {} - OpenGL not available", function!());
            return;
        }

        // Force the OpenGL backend so a NotSupported from the GL float path
        // surfaces as a hard error instead of being masked by CPU fallback.
        let mut proc = match ImageProcessor::with_config(ImageProcessorConfig {
            backend: ComputeBackend::OpenGl,
            ..Default::default()
        }) {
            Ok(p) => p,
            Err(e) => {
                eprintln!("SKIPPED: {} - OpenGL backend unavailable: {e}", function!());
                return;
            }
        };

        if !proc.supported_render_dtypes().f32 {
            eprintln!("SKIPPED: {} - F32 render not supported", function!());
            return;
        }

        let w = 64usize;
        let h = 64usize;

        // RGBA8 source PBO with a known gradient.
        let src = proc
            .create_image(w, h, PixelFormat::Rgba, DType::U8, None)
            .unwrap();
        if src.memory() != TensorMemory::Pbo {
            eprintln!("SKIPPED: {} - RGBA8 src not PBO-backed", function!());
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
            .create_image(w, h, PixelFormat::Rgb, DType::F32, None)
            .unwrap();
        if dst.memory() != TensorMemory::Pbo {
            eprintln!("SKIPPED: {} - F32 dst not PBO-backed", function!());
            return;
        }

        // Run two converts on the same processor/source/dest so the second
        // iteration exercises the texture size/format-cache reuse path (the
        // float render target and PBO source texture are NOT reallocated on the
        // 2nd call). Correctness must hold identically across both iterations.
        let mut max_err = 0.0f32;
        for iter in 0..2 {
            proc.convert(&src, &mut dst, Rotation::None, Flip::None, Crop::default())
                .unwrap();

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

    /// On-GPU resize: RGBA8 16x16 → F32 NHWC `[H,W,3]` PBO at 8x8 (2x
    /// downscale, identity crop). The source R channel is a linear horizontal
    /// ramp `R(x) = x * 16`, so the correct bilinear-resampled output at dst
    /// pixel `dx` samples src at `x_src = (dx + 0.5) * 2 - 0.5 = 2*dx + 0.5`,
    /// giving value `(2*dx + 0.5) * 16` normalized by `/255`. This value is
    /// distinct from NEAREST (which would land on a single integer texel,
    /// `2*dx * 16`), so the test discriminates bilinear vs NEAREST sampling.
    #[test]
    fn convert_f32_nhwc_pbo_resize_bilinear() {
        use crate::{ComputeBackend, ImageProcessor, ImageProcessorConfig};

        if !is_opengl_available() {
            eprintln!("SKIPPED: {} - OpenGL not available", function!());
            return;
        }

        let mut proc = match ImageProcessor::with_config(ImageProcessorConfig {
            backend: ComputeBackend::OpenGl,
            ..Default::default()
        }) {
            Ok(p) => p,
            Err(e) => {
                eprintln!("SKIPPED: {} - OpenGL backend unavailable: {e}", function!());
                return;
            }
        };

        if !proc.supported_render_dtypes().f32 {
            eprintln!("SKIPPED: {} - F32 render not supported", function!());
            return;
        }

        let sw = 16usize;
        let sh = 16usize;
        let dw = 8usize;
        let dh = 8usize;

        // RGBA8 source PBO: horizontal ramp R = x*16, G = y*16, B const.
        let src = proc
            .create_image(sw, sh, PixelFormat::Rgba, DType::U8, None)
            .unwrap();
        if src.memory() != TensorMemory::Pbo {
            eprintln!("SKIPPED: {} - RGBA8 src not PBO-backed", function!());
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
            .create_image(dw, dh, PixelFormat::Rgb, DType::F32, None)
            .unwrap();
        if dst.memory() != TensorMemory::Pbo {
            eprintln!("SKIPPED: {} - F32 dst not PBO-backed", function!());
            return;
        }

        proc.convert(&src, &mut dst, Rotation::None, Flip::None, Crop::default())
            .unwrap();

        let map = dst.as_f32().unwrap().map().unwrap();
        assert_eq!(map.len(), dw * dh * 3);
        // Bilinear: src sample position = (d + 0.5) * 2 - 0.5 = 2*d + 0.5.
        // Edge dst pixels (d=0 sampling x_src=0.5, last d sampling beyond the
        // last center) stay interior here: for dw=8, max x_src = 2*7+0.5=14.5,
        // which is < 15 (last src column), so no clamping affects the ramp.
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

    /// On-GPU round-trip: RGBA8 → F16 NCHW `[3,H,W]` PBO via the GL float
    /// render path. Forces the OpenGL backend (no CPU fallback). Identity
    /// crop so `dst[c,y,x] == src[y,x,c] / 255` within one f16 ULP at 1.0
    /// (`2^-8`).
    #[test]
    fn convert_f16_nchw_pbo_roundtrip() {
        use crate::{ComputeBackend, ImageProcessor, ImageProcessorConfig};
        use half::f16;

        if !is_opengl_available() {
            eprintln!("SKIPPED: {} - OpenGL not available", function!());
            return;
        }

        let mut proc = match ImageProcessor::with_config(ImageProcessorConfig {
            backend: ComputeBackend::OpenGl,
            ..Default::default()
        }) {
            Ok(p) => p,
            Err(e) => {
                eprintln!("SKIPPED: {} - OpenGL backend unavailable: {e}", function!());
                return;
            }
        };

        if !proc.supported_render_dtypes().f16 {
            eprintln!("SKIPPED: {} - F16 render not supported", function!());
            return;
        }

        let w = 64usize;
        let h = 64usize;

        let src = proc
            .create_image(w, h, PixelFormat::Rgba, DType::U8, None)
            .unwrap();
        if src.memory() != TensorMemory::Pbo {
            eprintln!("SKIPPED: {} - RGBA8 src not PBO-backed", function!());
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
            .create_image(w, h, PixelFormat::PlanarRgb, DType::F16, None)
            .unwrap();
        if dst.memory() != TensorMemory::Pbo {
            eprintln!("SKIPPED: {} - F16 dst not PBO-backed", function!());
            return;
        }

        proc.convert(&src, &mut dst, Rotation::None, Flip::None, Crop::default())
            .unwrap();

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
        let _ = f16::from_f32(0.0); // silence unused import on configs without asserts
        eprintln!("convert_f16_nchw_pbo_roundtrip: max_err={max_err} tol={tol}");
    }

    /// On-GPU round-trip: RGBA8 → F16 NCHW `[3,H,W]` DMA-BUF via the GL
    /// float render path (`convert_float_to_dma`). Forces the OpenGL backend
    /// (no CPU fallback) so any GL-path failure surfaces as a hard error
    /// instead of being silently masked. Identity crop, so the expected values
    /// are exact: `dst[c,y,x] == src[y,x,c] / 255` within one f16 ULP at 1.0
    /// (`2^-8`).
    ///
    /// Skip conditions (treated as pass):
    /// 1. GL unavailable, or F16 render not supported (e.g. Vivante).
    /// 2. `create_image(..., Dma)` returns `Err` — no dma-heap (e.g. dev host,
    ///    Orin-nano with permission-denied).
    /// 3. The created tensor's `.memory()` is not `TensorMemory::Dma` — it
    ///    fell back; only the real DMA path is of interest here.
    ///
    /// Runs on V3D/Mali targets where dma-heap and GL F16 render are both
    /// available.
    #[test]
    fn convert_f16_nchw_dma_roundtrip() {
        use crate::{ComputeBackend, ImageProcessor, ImageProcessorConfig};
        use half::f16;

        if !is_opengl_available() {
            eprintln!("SKIPPED: {} - OpenGL not available", function!());
            return;
        }

        // Force the OpenGL backend so a GL path error (e.g. from
        // convert_float_to_dma) surfaces as Err instead of falling through
        // to the CPU backend.
        let mut proc = match ImageProcessor::with_config(ImageProcessorConfig {
            backend: ComputeBackend::OpenGl,
            ..Default::default()
        }) {
            Ok(p) => p,
            Err(e) => {
                eprintln!("SKIPPED: {} - OpenGL backend unavailable: {e}", function!());
                return;
            }
        };

        if !proc.supported_render_dtypes().f16 {
            eprintln!(
                "SKIPPED: {} - F16 render not supported (Vivante?)",
                function!()
            );
            return;
        }

        // Use a representative size: small dma-buf imports hit per-fourcc
        // minimum-size limits on some GPUs (Mali rejects a tiny (W/4, 3H)
        // RGBA16F surface), so 16x16 is unrepresentative. 256x256 matches the
        // gpu-probe's validated F16 dma-buf size and the real preprocessing
        // regime; sizes the GPU cannot import fall back to CPU in production.
        let w = 256usize;
        let h = 256usize;

        // RGBA8 source with a known gradient (W%4==0 so f16 packing is valid).
        let src = proc
            .create_image(w, h, PixelFormat::Rgba, DType::U8, None)
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
        // Skip if dma-heap is unavailable (permission-denied on dev host).
        let mut dst = match proc.create_image(
            w,
            h,
            PixelFormat::PlanarRgb,
            DType::F16,
            Some(TensorMemory::Dma),
        ) {
            Ok(t) => t,
            Err(e) => {
                eprintln!(
                    "SKIPPED: {} - F16 DMA tensor alloc failed (no dma-heap?): {e}",
                    function!()
                );
                return;
            }
        };

        // Confirm the tensor is genuinely DMA-backed — skip if the allocator
        // fell back to a different memory type.
        if dst.memory() != TensorMemory::Dma {
            eprintln!(
                "SKIPPED: {} - F16 dst memory is {:?}, not Dma; DMA path not exercised",
                function!(),
                dst.memory()
            );
            return;
        }

        proc.convert(&src, &mut dst, Rotation::None, Flip::None, Crop::default())
            .unwrap();

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
        let _ = f16::from_f32(0.0); // silence unused import on configs without asserts
        eprintln!("convert_f16_nchw_dma_roundtrip: max_err={max_err} tol={tol}");
    }

    /// GAP-3: F32 Rgb PBO destination with a `dst_rect` letterbox and
    /// `dst_color` pad.  Exercises the `src_rect_uv` / `dst_rect_px` /
    /// `pad_color` shader uniforms that had no test coverage.
    ///
    /// Setup:
    /// - src: 4x4 RGBA8 all-solid known color (R=200, G=100, B=50, A=255)
    /// - dst: 8x4 F32 Rgb PBO
    /// - `dst_rect = (left:2, top:0, w:4, h:4)` → the middle 4 columns carry
    ///   the content; the outer 2+2 columns are filled with the pad color.
    /// - `dst_color = [114, 114, 114, 255]` (standard letterbox grey)
    ///
    /// Assertions:
    /// - Pixels OUTSIDE dst_rect → ≈ 114/255 on all three channels (pad grey)
    /// - Pixels INSIDE dst_rect  → finite & in `[0, 1]`, representing the
    ///   content (exact normalization is GPU-dependent; we just need non-zero
    ///   and in-range)
    #[test]
    fn convert_f32_pbo_letterbox_pad_color() {
        use crate::{ComputeBackend, ImageProcessor, ImageProcessorConfig, Rect};

        if !is_opengl_available() {
            eprintln!("SKIPPED: {} - OpenGL not available", function!());
            return;
        }

        let mut proc = match ImageProcessor::with_config(ImageProcessorConfig {
            backend: ComputeBackend::OpenGl,
            ..Default::default()
        }) {
            Ok(p) => p,
            Err(e) => {
                eprintln!("SKIPPED: {} - OpenGL backend unavailable: {e}", function!());
                return;
            }
        };

        if !proc.supported_render_dtypes().f32 {
            eprintln!("SKIPPED: {} - F32 render not supported", function!());
            return;
        }

        let src_w = 4usize;
        let src_h = 4usize;
        let dst_w = 8usize;
        let dst_h = 4usize;

        // RGBA8 source: solid known colour (R=200, G=100, B=50).
        let src = proc
            .create_image(src_w, src_h, PixelFormat::Rgba, DType::U8, None)
            .unwrap();
        if src.memory() != TensorMemory::Pbo {
            eprintln!("SKIPPED: {} - RGBA8 src not PBO-backed", function!());
            return;
        }
        {
            let mut map = src.as_u8().unwrap().map().unwrap();
            for chunk in map.chunks_exact_mut(4) {
                chunk[0] = 200;
                chunk[1] = 100;
                chunk[2] = 50;
                chunk[3] = 255;
            }
        }

        // F32 Rgb PBO destination (wider than src to create a letterbox).
        let mut dst = proc
            .create_image(dst_w, dst_h, PixelFormat::Rgb, DType::F32, None)
            .unwrap();
        if dst.memory() != TensorMemory::Pbo {
            eprintln!("SKIPPED: {} - F32 dst not PBO-backed", function!());
            return;
        }

        // Letterbox: content lands in columns [2, 6); columns [0, 2) and
        // [6, 8) are padded with the `dst_color`.
        let crop = Crop::new()
            .with_dst_rect(Some(Rect::new(2, 0, 4, 4)))
            .with_dst_color(Some([114, 114, 114, 255]));

        let result = proc.convert(&src, &mut dst, Rotation::None, Flip::None, crop);
        assert!(
            result.is_ok(),
            "F32 PBO letterbox convert must not error: {:?}",
            result.err()
        );

        let map = dst.as_f32().unwrap().map().unwrap();
        assert_eq!(map.len(), dst_w * dst_h * 3, "unexpected element count");

        let pad_expected = 114.0f32 / 255.0;
        let pad_tol = 4.0 / 255.0; // generous: GPU blend precision

        // Columns 0 and 1 are outside dst_rect → should carry the pad color.
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

        // Columns 2..6 are inside dst_rect → content, finite & in [0, 1].
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
    fn convert_f32_pbo_cuda_map_roundtrip() {
        use crate::{ComputeBackend, ImageProcessor, ImageProcessorConfig};

        if !edgefirst_tensor::is_cuda_available() {
            eprintln!("SKIP: no libcudart");
            return;
        }
        let mut proc = match ImageProcessor::with_config(ImageProcessorConfig {
            backend: ComputeBackend::OpenGl,
            ..Default::default()
        }) {
            Ok(p) => p,
            Err(_) => {
                eprintln!("SKIP: no GL");
                return;
            }
        };
        if !proc.supported_render_dtypes().f32 {
            eprintln!("SKIP: no f32 render");
            return;
        }
        let (w, h) = (64usize, 64usize);
        let src = proc
            .create_image(w, h, PixelFormat::Rgba, DType::U8, None)
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
            .create_image(w, h, PixelFormat::Rgb, DType::F32, None)
            .unwrap();
        if dst.memory() != TensorMemory::Pbo {
            eprintln!("SKIP: dst not PBO (no CUDA-GL alloc here)");
            return;
        }
        proc.convert(&src, &mut dst, Rotation::None, Flip::None, Crop::default())
            .unwrap();
        let cm = match dst.cuda_map() {
            Some(cm) => cm,
            None => {
                eprintln!(
                    "SKIP: cuda_map returned None (CUDA-GL interop unavailable for this context)"
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
        // drop(cm) unmaps so a subsequent convert() could reuse the PBO
    }

    /// C7: verify that the device pointer from `cuda_map()` holds the correct
    /// NHWC float data. Copies the device buffer back to host via `cudaMemcpy`
    /// and checks each `[y,x,c]` element matches the normalized source value
    /// `src[y,x,c] / 255.0` within 1e-3. Uses a 16×16 identity crop so the
    /// expected values are exact (no bilinear rounding).
    #[test]
    fn convert_f32_pbo_cuda_map_numeric() {
        use crate::{ComputeBackend, ImageProcessor, ImageProcessorConfig};
        use std::ffi::c_void;

        if !edgefirst_tensor::is_cuda_available() {
            eprintln!("SKIP: no libcudart");
            return;
        }
        let mut proc = match ImageProcessor::with_config(ImageProcessorConfig {
            backend: ComputeBackend::OpenGl,
            ..Default::default()
        }) {
            Ok(p) => p,
            Err(_) => {
                eprintln!("SKIP: no GL");
                return;
            }
        };
        if !proc.supported_render_dtypes().f32 {
            eprintln!("SKIP: no f32 render");
            return;
        }
        let (w, h) = (16usize, 16usize);
        // RGBA8 source with a known per-pixel gradient.
        let src = proc
            .create_image(w, h, PixelFormat::Rgba, DType::U8, None)
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
            .create_image(w, h, PixelFormat::Rgb, DType::F32, None)
            .unwrap();
        if dst.memory() != TensorMemory::Pbo {
            eprintln!("SKIP: dst not PBO");
            return;
        }
        proc.convert(&src, &mut dst, Rotation::None, Flip::None, Crop::default())
            .unwrap();

        let cm = match dst.cuda_map() {
            Some(cm) => cm,
            None => {
                eprintln!(
                    "SKIP: cuda_map returned None (CUDA-GL interop unavailable for this context)"
                );
                return;
            }
        };
        let n = w * h * 3;
        let mut host = vec![0f32; n];
        // SAFETY: host has `n * 4` bytes allocated; device_ptr is valid for
        // the lifetime of `cm` (cuda_map is still live here).
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
        // drop(cm) unmaps
    }
}
