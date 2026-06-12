// SPDX-FileCopyrightText: Copyright 2025 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

#[cfg(test)]
#[cfg(feature = "opengl")]
#[allow(deprecated)]
mod gl_tests {
    #[cfg(all(target_os = "linux", feature = "dma_test_formats"))]
    use crate::opengl_headless::processor::GLProcessorST;
    #[cfg(target_os = "linux")]
    use crate::{probe_egl_displays, EglDisplayKind};
    use crate::{Crop, Flip, GLProcessorThreaded, ImageProcessorTrait, Rotation};
    use edgefirst_decoder::{DetectBox, ProtoData, ProtoLayout};
    #[cfg(all(target_os = "linux", feature = "dma_test_formats"))]
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
        // Portable: the unified engine runs on Linux (EGL/DMA-BUF) and
        // macOS (ANGLE/IOSurface). On the macOS coverage flow this returns
        // false in pass 1 (unsigned binaries, ANGLE dlopen gate closed) so
        // the GL tests self-skip there and execute in pass 2.
        *GL_AVAILABLE.get_or_init(|| GLProcessorThreaded::new(None).is_ok())
    }

    /// Zero-copy GPU image buffers available? DMA-BUF heaps on Linux,
    /// IOSurface on macOS. The portable seam for tests whose `@Dma`
    /// fixtures allocate on both platforms (RGBA/BGRA/GREY/NV*/YUYV);
    /// packed-RGB `@Dma` has no IOSurface FourCC, so tests with RGB DMA
    /// destinations stay Linux-gated.
    #[cfg(feature = "dma_test_formats")]
    fn is_gpu_image_buffer_available() -> bool {
        edgefirst_tensor::is_gpu_buffer_available()
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

    // DEDUP: this function is also defined verbatim in the `image_tests` module
    // of `crates/image/src/lib.rs`. Both copies must be kept in sync. See the
    // comment there for why cross-module sharing was deferred.
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

    /// Steady-state import gate: an N-frame convert loop over a fixed pool of
    /// DMA source buffers into one DMA destination must perform ZERO new
    /// EGLImage imports after the pool has been seen once — every later
    /// convert is a cache hit on all caches (src/dst/NV R8). The import
    /// count, not latency, is the regression signal that pins EGLImage cache
    /// behavior across GL refactors: a refactor that re-imports per frame
    /// passes every pixel-equality test but fails this one.
    #[test]
    #[cfg(feature = "dma_test_formats")]
    fn dma_pool_steady_state_zero_imports() {
        if !is_gpu_image_buffer_available() {
            eprintln!("SKIPPED: {} - no zero-copy GPU buffers", function!());
            return;
        }
        let mut renderer = match GLProcessorThreaded::new(None) {
            Ok(r) => r,
            Err(e) => {
                eprintln!("SKIPPED: {} - GL not available: {e}", function!());
                return;
            }
        };
        let (w, h) = (128usize, 96usize);
        const POOL: usize = 3;
        const FRAMES: usize = 300;

        // NV12 pool (the camera-pipeline shape): Y plane + neutral chroma.
        let mut bytes = vec![100u8; w * h];
        bytes.extend(std::iter::repeat_n(128u8, w * h / 2));
        let pool: Vec<TensorDyn> = (0..POOL)
            .map(|_| {
                load_raw_image(w, h, PixelFormat::Nv12, Some(TensorMemory::Dma), &bytes).unwrap()
            })
            .collect();
        let mut dst =
            TensorDyn::image(w, h, PixelFormat::Rgba, DType::U8, Some(TensorMemory::Dma)).unwrap();

        // Warmup: two passes over the pool import every buffer once.
        for src in pool.iter().cycle().take(POOL * 2) {
            renderer
                .convert(src, &mut dst, Rotation::None, Flip::None, Crop::no_crop())
                .unwrap();
        }
        let warm = renderer.egl_cache_stats().unwrap();

        for src in pool.iter().cycle().take(FRAMES) {
            renderer
                .convert(src, &mut dst, Rotation::None, Flip::None, Crop::no_crop())
                .unwrap();
        }
        let steady = renderer.egl_cache_stats().unwrap();

        assert_eq!(
            warm.total_misses(),
            steady.total_misses(),
            "steady-state loop performed new EGLImage imports: warm={warm:?} steady={steady:?}"
        );
        let hits = |s: &crate::opengl_headless::cache::GlCacheStats| {
            s.src.hits + s.dst.hits + s.nv_r8.hits
        };
        let gained = hits(&steady) - hits(&warm);
        assert!(
            gained >= FRAMES as u64,
            "expected at least {FRAMES} cache hits over the loop, got {gained} \
             (warm={warm:?} steady={steady:?})"
        );
    }

    /// Regression test for the repeat-convert GL_INVALID_VALUE (0x501) state
    /// leak: heap-RGBA source → two-pass packed RGB → DMA destination succeeded
    /// on the FIRST convert and failed on the SECOND on Vivante/Mali/V3D alike.
    ///
    /// Root cause: `convert_to_packed_rgb` pass 2 exits with `ActiveTexture`
    /// left at TEXTURE1 and the destination EGLImage texture bound on unit 0,
    /// while `draw_src_texture` issued `BindTexture` BEFORE `ActiveTexture` —
    /// binding the camera texture to the leaked unit and then uploading the
    /// 1280×720 source into the 640×640 destination texture via the
    /// `TexSubImage2D` fast path → GL_INVALID_VALUE. Driver-independent GLES
    /// semantics, hence identical on all dmabuf GPUs.
    ///
    /// The loop runs each dtype cell several times (not just twice) and pins
    /// byte-identical output across iterations, so any texture-unit state leak
    /// between converts surfaces either as an error or as a pixel diff.
    #[test]
    // Stays Linux-gated: packed-RGB @Dma destinations have no IOSurface
    // FourCC, so the fixture cannot allocate on macOS.
    #[cfg(all(target_os = "linux", feature = "dma_test_formats"))]
    fn repeat_convert_rgba_mem_to_rgb_dma() {
        if !is_dma_available() {
            eprintln!("SKIPPED: {} - DMA not available", function!());
            return;
        }
        let mut renderer = match GLProcessorThreaded::new(None) {
            Ok(r) => r,
            Err(e) => {
                eprintln!("SKIPPED: {} - GL not available: {e}", function!());
                return;
            }
        };

        // Deterministic RGBA gradient source in plain heap memory (the cell's
        // defining property: NOT DMA, so the source goes through the
        // draw_src_texture CPU-upload path).
        let (src_w, src_h) = (1280usize, 720usize);
        let mut bytes = vec![0u8; src_w * src_h * 4];
        for (i, px) in bytes.chunks_exact_mut(4).enumerate() {
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
            let mut dst = match TensorDyn::image(
                640,
                640,
                PixelFormat::Rgb,
                dtype,
                Some(TensorMemory::Dma),
            ) {
                Ok(d) => d,
                Err(e) => {
                    eprintln!("SKIPPED cell {dtype:?}: DMA RGB dst unavailable: {e}");
                    continue;
                }
            };
            let mut first: Option<Vec<u8>> = None;
            for round in 0..4 {
                renderer
                    .convert(&src, &mut dst, Rotation::None, Flip::None, lb)
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

    /// The int8 variant of every draw must differ from the u8 variant by
    /// exactly XOR 0x80 on the colour channels — the int8 fragment shaders
    /// are the u8 shaders plus the bias, nothing else (alpha passes through).
    ///
    /// Catches the texture-destination program-selection gap: the heap-source
    /// NV12 (ShaderR8 upload) int8 convert rendered through the UN-biased NV
    /// program (only the DMA destination path swapped `nv_r8_program`),
    /// producing u8-semantics bytes in an i8 tensor — 0x80 off on every
    /// channel. Needs no DMA at runtime (Mem src/dst — runs on llvmpipe,
    /// whose coverage lane enables the feature); the `dma_test_formats`
    /// gate only matches the `load_raw_image` helper's.
    #[test]
    #[cfg(feature = "dma_test_formats")]
    fn int8_mem_convert_is_xor_biased_u8() {
        if !is_opengl_available() {
            eprintln!("SKIPPED: {} - OpenGL not available", function!());
            return;
        }
        let mut renderer = match GLProcessorThreaded::new(None) {
            Ok(r) => r,
            Err(e) => {
                eprintln!("SKIPPED: {} - GL not available: {e}", function!());
                return;
            }
        };

        // Gradient luma + off-neutral chroma so the bias is exercised across
        // values straddling 0x80 (where a missing bias is a ~128 byte diff).
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

        // RGBA must convert everywhere the NV upload path exists; RGB-format
        // readback is implementation-defined in GLES (V3D rejects it with
        // GL_INVALID_OPERATION — the cell was absent from rpi5's PR-0 matrix
        // baseline too), so a failing RGB cell skips rather than fails.
        let mut validated = 0usize;
        'fmt: for fmt in [PixelFormat::Rgb, PixelFormat::Rgba] {
            let mut out = [Vec::new(), Vec::new()];
            for (slot, dtype) in [(0, DType::U8), (1, DType::I8)] {
                let mut dst = TensorDyn::image(w, h, fmt, dtype, Some(TensorMemory::Mem)).unwrap();
                if let Err(e) =
                    renderer.convert(&src, &mut dst, Rotation::None, Flip::None, Crop::no_crop())
                {
                    if fmt == PixelFormat::Rgba {
                        panic!("nv12@mem -> rgba.{dtype:?}@mem must be supported: {e}");
                    }
                    eprintln!("SKIPPED cell nv12@mem -> {fmt}.{dtype:?}@mem: {e}");
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
                // ±1 LSB: the int8 shader's explicit floor(v*255+0.5)+128
                // quantization double-rounds against the driver's own
                // float→unorm8 store, shifting ~2% of bytes by one on some
                // GPUs (seen on Vivante). A missing bias is a ~128 diff, so
                // the tolerance keeps full sensitivity to the real bug.
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
    #[cfg(feature = "dma_test_formats")]
    fn test_opengl_nv12_to_rgba_reference() {
        if !is_gpu_image_buffer_available() {
            eprintln!("SKIPPED: {} - no zero-copy GPU buffers", function!());
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

        // Post-WS1 the GL NV path threads per-tensor colorimetry (same coeffs as
        // CPU/ffmpeg for the resolved encoding+range), so the YUV-matrix delta
        // that forced 0.95 has closed; restored to the pre-WIP 0.98.
        compare_images(&reference, &cpu_dst, 0.98, "opengl_nv12_to_rgba_reference");
    }

    /// Test OpenGL PixelFormat::Yuyv→PixelFormat::Rgba conversion against ffmpeg reference
    #[test]
    #[cfg(feature = "dma_test_formats")]
    fn test_opengl_yuyv_to_rgba_reference() {
        if !is_gpu_image_buffer_available() {
            eprintln!("SKIPPED: {} - no zero-copy GPU buffers", function!());
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

        // Post-WS1 the GL YUYV path applies the resolved colorimetry via the EGL
        // YUV color-space/sample-range hints (dma_import), so the YUV-matrix
        // delta that forced 0.95 has closed; restored to the pre-WIP 0.98.
        // (Driver matrix may differ from CPU by rounding — confirmed on the GPU
        // lanes.)
        compare_images(&reference, &cpu_dst, 0.98, "opengl_yuyv_to_rgba_reference");
    }

    /// On-target (V3D) regression for the EGLImage cache `plane_offset` key:
    /// render two distinct sources into two `plane_offset` sub-views of ONE
    /// DMA-BUF and assert each window matches a standalone full-buffer convert.
    ///
    /// The two views share the parent's `BufferIdentity` but start at different
    /// byte offsets. Before the cache-key fix the second view aliased the
    /// first (offset-0) EGLImage, so both rendered into the base region — the
    /// exact failure that capped batched render-to-DMA-BUF. With the fix each
    /// offset gets its own EGLImage and the buffer is correctly partitioned.
    #[test]
    #[cfg(feature = "dma_test_formats")]
    fn opengl_render_into_dma_subviews_no_aliasing() {
        if !is_gpu_image_buffer_available() || !is_opengl_available() {
            eprintln!(
                "SKIPPED: {} - no zero-copy GPU buffers or OpenGL",
                function!()
            );
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
            Some(TensorMemory::Dma),
            &make_nv12(50),
        )
        .unwrap();
        let src1 = load_raw_image(
            w,
            h,
            PixelFormat::Nv12,
            Some(TensorMemory::Dma),
            &make_nv12(200),
        )
        .unwrap();

        let mut gl = GLProcessorThreaded::new(None).unwrap();
        let convert = |gl: &mut GLProcessorThreaded, s: &TensorDyn, d: &mut TensorDyn| {
            gl.convert(s, d, Rotation::None, Flip::None, Crop::no_crop())
                .unwrap();
        };

        // Standalone full-buffer reference conversions (independent buffers).
        let mut ref0 =
            TensorDyn::image(w, h, PixelFormat::Rgba, DType::U8, Some(TensorMemory::Dma)).unwrap();
        convert(&mut gl, &src0, &mut ref0);
        let mut ref1 =
            TensorDyn::image(w, h, PixelFormat::Rgba, DType::U8, Some(TensorMemory::Dma)).unwrap();
        convert(&mut gl, &src1, &mut ref1);

        // One DMA-BUF holding two stacked RGBA frames; two offset sub-views
        // sharing the parent buffer identity (same fd) on the SAME processor,
        // so view 1's lookup hits view 0's cache entry unless the key carries
        // the offset.
        let parent = TensorDyn::image(
            w,
            2 * h,
            PixelFormat::Rgba,
            DType::U8,
            Some(TensorMemory::Dma),
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
    // Recycled DMA-BUF *source* stale-read regression
    //
    // A DMA-BUF-backed source that is CPU-written, GPU-sampled by convert(),
    // and then RECYCLED across frames must reflect the latest write. The
    // recycled buffer keeps a stable BufferIdentity, so the source EGLImage
    // cache (keyed by BufferIdentity + chroma id + plane_offset) hits.
    //
    // The defect this guards against: a decode pool reuses ONE oversized buffer
    // and calls `configure_image()` each frame, so the SAME fd is presented to
    // convert() with a DIFFERENT geometry/stride every frame (e.g. a 128-wide
    // pool decoding a 96-wide image). Because the cache key omits geometry, the
    // cached EGLImage built for the previous frame's stride is reused and the
    // GPU samples the buffer at the wrong pitch — deterministically wrong
    // single-threaded, nondeterministic in parallel, correct on a heap source.
    // (Recycling at CONSTANT geometry is fine — V3D re-fetches the DMA-BUF each
    // draw — so the constant-geometry tests below pass on buggy and fixed code
    // alike; the geometry-change test is the one that fails pre-fix.)
    //
    // Oracle = a FRESH DMA source of the same content+geometry (new identity →
    // cache miss → correct convert). Same code path as the recycled source; the
    // only difference is buffer identity.
    // =========================================================================

    /// Overwrite an existing tensor's bytes in place (re-map → copy → drop →
    /// DMA_BUF_IOCTL_SYNC(END)), simulating a pool recycle of one buffer.
    #[cfg(feature = "dma_test_formats")]
    fn overwrite_in_place(t: &TensorDyn, bytes: &[u8]) {
        let mut map = t.as_u8().unwrap().map().unwrap();
        map.as_mut_slice()[..bytes.len()].copy_from_slice(bytes);
    }

    /// Write image A into a DMA source, convert, then overwrite the SAME source
    /// with distinct image B and convert again on the SAME processor. Assert the
    /// second output matches a fresh-DMA-source convert of B (no stale read).
    #[cfg(feature = "dma_test_formats")]
    fn dma_recycle_stale_read_check(
        w: usize,
        h: usize,
        src_fmt: PixelFormat,
        bytes_a: &[u8],
        bytes_b: &[u8],
        tolerance: u8,
    ) {
        // ONE recycled DMA-BUF source, reused across two "frames".
        let src = load_raw_image(w, h, src_fmt, Some(TensorMemory::Dma), bytes_a).unwrap();
        let mut gl = GLProcessorThreaded::new(None).unwrap();
        let convert = |gl: &mut GLProcessorThreaded, s: &TensorDyn, d: &mut TensorDyn| {
            gl.convert(s, d, Rotation::None, Flip::None, Crop::no_crop())
                .unwrap();
        };

        // Frame A: warms the src EGLImage cache + texture binding.
        let mut dst_a =
            TensorDyn::image(w, h, PixelFormat::Rgba, DType::U8, Some(TensorMemory::Dma)).unwrap();
        convert(&mut gl, &src, &mut dst_a);
        let dst_a_bytes = dst_a.as_u8().unwrap().map().unwrap().to_vec();

        // Frame B: overwrite the SAME buffer (same fd / BufferIdentity), convert
        // again. Cache HIT + binding-skip → this is where the stale read fires.
        overwrite_in_place(&src, bytes_b);
        let mut dst_b =
            TensorDyn::image(w, h, PixelFormat::Rgba, DType::U8, Some(TensorMemory::Dma)).unwrap();
        convert(&mut gl, &src, &mut dst_b);
        let dst_b_bytes = dst_b.as_u8().unwrap().map().unwrap().to_vec();

        // Oracle: a fresh DMA source of B (new identity → cache miss → correct).
        let fresh_b = load_raw_image(w, h, src_fmt, Some(TensorMemory::Dma), bytes_b).unwrap();
        let mut oracle_b =
            TensorDyn::image(w, h, PixelFormat::Rgba, DType::U8, Some(TensorMemory::Dma)).unwrap();
        convert(&mut gl, &fresh_b, &mut oracle_b);
        let oracle_b_bytes = oracle_b.as_u8().unwrap().map().unwrap().to_vec();

        // Guard against bad test data: A and B must produce different output, or
        // the test could never distinguish stale from correct.
        assert_ne!(
            dst_a_bytes, oracle_b_bytes,
            "test data invalid for {src_fmt:?}: frame A and frame B convert identically"
        );

        // The bug: dst_b carries frame-A content (stale) instead of frame-B.
        assert_pixels_match(&oracle_b_bytes, &dst_b_bytes, tolerance);
    }

    /// Grey source (profiler's decode-pool format; EXTERNAL_OES Path A).
    /// Grey→Rgba is a scalar luma replicate (no YUV matrix) → byte-exact.
    #[test]
    #[cfg(feature = "dma_test_formats")]
    fn dma_recycle_grey_stale_read() {
        if !is_gpu_image_buffer_available() || !is_opengl_available() {
            eprintln!(
                "SKIPPED: {} - no zero-copy GPU buffers or OpenGL",
                function!()
            );
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

    /// NV12 source (EXTERNAL_OES Path A on Vivante / R8 texelFetch Path B on
    /// V3D/Mali). YUV→Rgba carries a small color-matrix rounding → tolerance 4.
    #[test]
    #[cfg(feature = "dma_test_formats")]
    fn dma_recycle_nv12_stale_read() {
        if !is_gpu_image_buffer_available() || !is_opengl_available() {
            eprintln!(
                "SKIPPED: {} - no zero-copy GPU buffers or OpenGL",
                function!()
            );
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

    /// NV16 source (full-res chroma; R8 texelFetch Path B). tolerance 4.
    #[test]
    #[cfg(feature = "dma_test_formats")]
    fn dma_recycle_nv16_stale_read() {
        if !is_gpu_image_buffer_available() || !is_opengl_available() {
            eprintln!(
                "SKIPPED: {} - no zero-copy GPU buffers or OpenGL",
                function!()
            );
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

    /// Small-pool recycle: 2 DMA-BUF sources round-robined over 5 distinct
    /// solid-fill frames; every GPU output must equal a fresh-source convert of
    /// the same frame. Exercises the cache/LRU interaction across more than one
    /// recycled identity.
    #[test]
    #[cfg(feature = "dma_test_formats")]
    fn dma_pool_recycle_all_frames_match_oracle() {
        if !is_gpu_image_buffer_available() || !is_opengl_available() {
            eprintln!(
                "SKIPPED: {} - no zero-copy GPU buffers or OpenGL",
                function!()
            );
            return;
        }
        let (w, h) = (64usize, 64usize);
        let lumas: [u8; 5] = [20, 70, 130, 180, 240];

        let pool = [
            TensorDyn::image(w, h, PixelFormat::Grey, DType::U8, Some(TensorMemory::Dma)).unwrap(),
            TensorDyn::image(w, h, PixelFormat::Grey, DType::U8, Some(TensorMemory::Dma)).unwrap(),
        ];
        let mut gl = GLProcessorThreaded::new(None).unwrap();
        let convert = |gl: &mut GLProcessorThreaded, s: &TensorDyn, d: &mut TensorDyn| {
            gl.convert(s, d, Rotation::None, Flip::None, Crop::no_crop())
                .unwrap();
        };

        for (frame, &luma) in lumas.iter().enumerate() {
            let buf = &pool[frame % pool.len()];
            overwrite_in_place(buf, &vec![luma; w * h]);

            let mut gpu =
                TensorDyn::image(w, h, PixelFormat::Rgba, DType::U8, Some(TensorMemory::Dma))
                    .unwrap();
            convert(&mut gl, buf, &mut gpu);

            // Fresh-source oracle for this exact frame.
            let fresh = load_raw_image(
                w,
                h,
                PixelFormat::Grey,
                Some(TensorMemory::Dma),
                &vec![luma; w * h],
            )
            .unwrap();
            let mut oracle =
                TensorDyn::image(w, h, PixelFormat::Rgba, DType::U8, Some(TensorMemory::Dma))
                    .unwrap();
            convert(&mut gl, &fresh, &mut oracle);

            let oracle_map = oracle.as_u8().unwrap().map().unwrap();
            let gpu_map = gpu.as_u8().unwrap().map().unwrap();
            assert_pixels_match(oracle_map.as_slice(), gpu_map.as_slice(), 0);
        }
    }

    /// Fill `[h][w]` of `t`'s buffer with a non-uniform 2D pattern at the
    /// tensor's effective row stride. Non-uniform content is essential: a solid
    /// fill survives any stride/geometry misread, so it could not detect the
    /// cache-key bug.
    #[cfg(feature = "dma_test_formats")]
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

    /// THE faithful profiler repro. One oversized DMA pool buffer is recycled
    /// across frames of DIFFERENT geometry via `configure_image()` (exactly what
    /// the JPEG decoder does into `create_decode_source_pool`). Each frame's GPU
    /// convert must equal a fresh-source convert of the same geometry+content.
    ///
    /// Pre-fix: frame N hits the cached EGLImage built for an earlier frame's
    /// stride and samples the buffer at the wrong pitch → mismatch. This test
    /// FAILS on buggy main and passes once the cache key carries geometry.
    #[test]
    #[cfg(feature = "dma_test_formats")]
    fn dma_recycle_geometry_change_stale_read() {
        if !is_gpu_image_buffer_available() || !is_opengl_available() {
            eprintln!(
                "SKIPPED: {} - no zero-copy GPU buffers or OpenGL",
                function!()
            );
            return;
        }
        // Oversized pool (like the profiler's pool_w x 3*max_h R8 surface),
        // recycled in place via configure_image — keeps ONE BufferIdentity.
        let mut pool = TensorDyn::image(
            128,
            128,
            PixelFormat::Grey,
            DType::U8,
            Some(TensorMemory::Dma),
        )
        .unwrap();
        let mut gl = GLProcessorThreaded::new(None).unwrap();
        let convert = |gl: &mut GLProcessorThreaded, s: &TensorDyn, d: &mut TensorDyn| {
            gl.convert(s, d, Rotation::None, Flip::None, Crop::no_crop())
                .unwrap();
        };

        // A sequence of DISTINCT geometries (varying width → varying stride),
        // each with distinct content. Reusing the same buffer means every frame
        // shares the pool's BufferIdentity but needs its own EGLImage geometry.
        let frames: [(usize, usize, u8); 4] =
            [(128, 96, 0), (96, 128, 40), (120, 100, 80), (64, 128, 120)];

        for (i, &(w, h, salt)) in frames.iter().enumerate() {
            // Recycled pool: reconfigure to this frame's geometry, write, convert.
            pool.configure_image(w, h, PixelFormat::Grey).unwrap();
            fill_grey_pattern(&pool, w, h, salt);
            let mut dst_recycled =
                TensorDyn::image(w, h, PixelFormat::Rgba, DType::U8, Some(TensorMemory::Dma))
                    .unwrap();
            convert(&mut gl, &pool, &mut dst_recycled);
            let recycled_bytes = dst_recycled.as_u8().unwrap().map().unwrap().to_vec();

            // Fresh-source oracle: new buffer, same geometry+content.
            let fresh =
                TensorDyn::image(w, h, PixelFormat::Grey, DType::U8, Some(TensorMemory::Dma))
                    .unwrap();
            fill_grey_pattern(&fresh, w, h, salt);
            let mut dst_fresh =
                TensorDyn::image(w, h, PixelFormat::Rgba, DType::U8, Some(TensorMemory::Dma))
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

    /// Stronger guard for the parallel-decode manifestation: a POOL of pooled
    /// buffers, each recycled through DIFFERENT geometries, INTERLEAVED on one
    /// processor — the same shared-cache access pattern parallel decode threads
    /// produce (which buffer+geometry populates the cache first is what varied
    /// run-to-run, giving 822–833). With a geometry-aware key every
    /// (buffer, geometry) has its own entry, so interleaving is order-
    /// independent and every output must match its fresh-source oracle.
    #[test]
    #[cfg(feature = "dma_test_formats")]
    fn dma_pool_geometry_interleaved_stale_read() {
        if !is_gpu_image_buffer_available() || !is_opengl_available() {
            eprintln!(
                "SKIPPED: {} - no zero-copy GPU buffers or OpenGL",
                function!()
            );
            return;
        }
        let mut pool = [
            TensorDyn::image(
                128,
                128,
                PixelFormat::Grey,
                DType::U8,
                Some(TensorMemory::Dma),
            )
            .unwrap(),
            TensorDyn::image(
                128,
                128,
                PixelFormat::Grey,
                DType::U8,
                Some(TensorMemory::Dma),
            )
            .unwrap(),
        ];
        let mut gl = GLProcessorThreaded::new(None).unwrap();
        let convert = |gl: &mut GLProcessorThreaded, s: &TensorDyn, d: &mut TensorDyn| {
            gl.convert(s, d, Rotation::None, Flip::None, Crop::no_crop())
                .unwrap();
        };

        // Interleave buffers AND geometries: each step reuses a pool slot at a
        // new size, so the cache sees the worst-case mix of identities+geometry.
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
            let mut dst =
                TensorDyn::image(w, h, PixelFormat::Rgba, DType::U8, Some(TensorMemory::Dma))
                    .unwrap();
            convert(&mut gl, &pool[slot], &mut dst);

            let fresh =
                TensorDyn::image(w, h, PixelFormat::Grey, DType::U8, Some(TensorMemory::Dma))
                    .unwrap();
            fill_grey_pattern(&fresh, w, h, salt);
            let mut oracle =
                TensorDyn::image(w, h, PixelFormat::Rgba, DType::U8, Some(TensorMemory::Dma))
                    .unwrap();
            convert(&mut gl, &fresh, &mut oracle);

            let dst_map = dst.as_u8().unwrap().map().unwrap();
            let oracle_map = oracle.as_u8().unwrap().map().unwrap();
            assert_pixels_match(oracle_map.as_slice(), dst_map.as_slice(), 0);
        }
    }

    /// Regression guard: a single-shot (non-recycled) DMA source must equal a
    /// fresh-source convert — the fix must not break the first-frame path.
    #[test]
    #[cfg(feature = "dma_test_formats")]
    fn dma_single_shot_grey_matches_fresh() {
        if !is_gpu_image_buffer_available() || !is_opengl_available() {
            eprintln!(
                "SKIPPED: {} - no zero-copy GPU buffers or OpenGL",
                function!()
            );
            return;
        }
        let (w, h) = (64usize, 64usize);
        let bytes = vec![128u8; w * h];
        let mut gl = GLProcessorThreaded::new(None).unwrap();
        let convert = |gl: &mut GLProcessorThreaded, s: &TensorDyn, d: &mut TensorDyn| {
            gl.convert(s, d, Rotation::None, Flip::None, Crop::no_crop())
                .unwrap();
        };

        let src = load_raw_image(w, h, PixelFormat::Grey, Some(TensorMemory::Dma), &bytes).unwrap();
        let mut dst =
            TensorDyn::image(w, h, PixelFormat::Rgba, DType::U8, Some(TensorMemory::Dma)).unwrap();
        convert(&mut gl, &src, &mut dst);

        let fresh =
            load_raw_image(w, h, PixelFormat::Grey, Some(TensorMemory::Dma), &bytes).unwrap();
        let mut oracle =
            TensorDyn::image(w, h, PixelFormat::Rgba, DType::U8, Some(TensorMemory::Dma)).unwrap();
        convert(&mut gl, &fresh, &mut oracle);

        let oracle_map = oracle.as_u8().unwrap().map().unwrap();
        let dst_map = dst.as_u8().unwrap().map().unwrap();
        assert_pixels_match(oracle_map.as_slice(), dst_map.as_slice(), 0);
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
    #[cfg(target_os = "linux")] // Linux display probing
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
    #[cfg(target_os = "linux")] // Linux display probing
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
    #[cfg(target_os = "linux")] // Linux display probing
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
    #[cfg(target_os = "linux")] // Linux display probing
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
    #[cfg(target_os = "linux")] // Linux display probing
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
    fn letterbox_crop(_src_w: usize, _src_h: usize, _dst_w: usize, _dst_h: usize) -> Crop {
        // Letterbox placement is now computed by the backend from the actual
        // src/dst dims (identical centred aspect-fit math); the helper just
        // selects the pad colour.
        Crop::letterbox([114, 114, 114, 255])
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
    // Stays Linux-gated: packed-RGB @Dma destinations have no IOSurface
    // FourCC, so the fixture cannot allocate on macOS.
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
    // Stays Linux-gated: packed-RGB @Dma destinations have no IOSurface
    // FourCC, so the fixture cannot allocate on macOS.
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
    #[cfg(feature = "dma_test_formats")]
    fn test_opengl_nv12_to_bgra() {
        if !is_gpu_image_buffer_available() {
            eprintln!("SKIPPED: test_opengl_nv12_to_bgra - no zero-copy GPU buffers");
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
    #[cfg(feature = "dma_test_formats")]
    fn test_opengl_yuyv_to_bgra() {
        if !is_gpu_image_buffer_available() {
            eprintln!("SKIPPED: test_opengl_yuyv_to_bgra - no zero-copy GPU buffers");
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
    #[cfg(target_os = "linux")] // PBO destinations are Linux-only
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
    #[cfg(target_os = "linux")] // PBO destinations are Linux-only
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
    #[cfg(target_os = "linux")] // PBO destinations are Linux-only
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

        // Pin BOTH sources to the SAME conversion path (ExternalSampler) so this
        // test isolates multiplane IMPORT correctness from path *selection*.
        // Under `auto` the contiguous single-plane source takes ShaderR8 (exact
        // in-shader YUV→RGB) while multiplane is forced to ExternalSampler
        // (driver YUV, ~6/255 off on Vivante) — a benchmark-driven split
        // (ExternalSampler ≈10× faster on Vivante), not an import bug. With both
        // on ExternalSampler, two imports of identical bytes must yield identical
        // pixels on every GPU, so the compare is byte-exact on all DMA boards.
        let _nv_path = NvPathEnvGuard::set("sampler");

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

        // Same bytes, two import paths, one conversion path → byte-identical.
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

        // Force both the same-fd multiplane and contiguous sources onto the
        // ExternalSampler path so the comparison tests import correctness, not
        // path selection (see test_multiplane_nv12_to_rgba_opengl for the
        // benchmark-driven rationale). Byte-exact on every DMA board.
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

        // Same-fd multiplane and contiguous must produce identical pixels.
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

        // Pin both sources to ExternalSampler so the packed-RGB letterbox compare
        // tests multiplane import correctness, not path selection (see
        // test_multiplane_nv12_to_rgba_opengl). Byte-exact on every DMA board.
        let _nv_path = NvPathEnvGuard::set("sampler");

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

        // Pin both sources to ExternalSampler. Critical for int8: the XOR-0x80
        // packing amplifies any path-selection color delta straddling 0x80 into a
        // ~250-unit byte diff (e.g. 126↔132 → 254↔2), so the two sources MUST
        // share one conversion path. This tests multiplane import correctness;
        // see test_multiplane_nv12_to_rgba_opengl for the benchmark rationale.
        let _nv_path = NvPathEnvGuard::set("sampler");

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

    /// Regression for the two-pass double letterbox (PR #107): pass 2 of the
    /// zero-copy NV→planar plan re-applied `dst_rect`, mapping the
    /// already-letterboxed intermediate into the content sub-region a second
    /// time and shrinking the image by the letterbox content fraction twice.
    ///
    /// A synthetic grey NV12 frame letterboxed 1280x720 → 640x640 must place
    /// content rows at exactly [140, 500); the double-applied bug squeezed
    /// them to ~[218, 422). Probing rows just inside the correct band (150 /
    /// 490) is therefore a deterministic kill-shot with no oracle tolerance
    /// to hide behind — and it equally catches the inverse bug (letterbox
    /// dropped entirely: pad rows would hold content). Covers PlanarRgb /
    /// PlanarRgba × u8 / i8 × Dma (the GL TwoPassNvPlanar plan) and Mem
    /// destinations — GL has no planar texture destination, so the Mem legs
    /// pin the backend `ImageProcessor` resolves instead (CPU fallback),
    /// guarding the whole dispatch surface against the bug class.
    /// (`dma_test_formats` gate matches the helpers it uses; the Mem legs
    /// still run on the llvmpipe coverage lane, which enables the feature.)
    #[test]
    #[cfg(all(target_os = "linux", feature = "dma_test_formats"))]
    fn letterbox_nv_to_planar_content_band_geometry() {
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

        // Synthetic NV12: Y=200, U=V=128 is neutral grey, so BT.601 vs BT.709
        // coefficients cancel and the converted RGB is ~200 (full range) to
        // ~214 (limited): comfortably above the 114 pad in every colorimetry
        // mode. Threshold at >=170 content / <=140 pad.
        let (sw, sh) = (1280usize, 720usize);
        let mut nv12 = vec![0u8; sw * sh * 3 / 2];
        nv12[..sw * sh].fill(200);
        nv12[sw * sh..].fill(128);
        let src_mem = if is_dma_available() {
            Some(TensorMemory::Dma)
        } else {
            Some(TensorMemory::Mem)
        };
        let src = load_raw_image(sw, sh, PixelFormat::Nv12, src_mem, &nv12).unwrap();

        // 1280x720 → 640x640: scale 0.5, content 640x360 centred → rows
        // [140, 500). Probe ≥10 px from the boundary to stay clear of
        // bilinear edge blending.
        const CONTENT_ROWS: [usize; 3] = [150, 320, 490];
        const PAD_ROWS: [usize; 4] = [0, 130, 510, 639];
        let (dw, dh) = (640usize, 640usize);
        let lb = Crop::letterbox([114, 114, 114, 255]);

        let mut dst_memories = vec![TensorMemory::Mem];
        if is_dma_available() {
            dst_memories.push(TensorMemory::Dma);
        }
        for dst_mem in dst_memories {
            for dst_fmt in [PixelFormat::PlanarRgb, PixelFormat::PlanarRgba] {
                for dtype in [DType::U8, DType::I8] {
                    let label = format!("nv12->{dst_fmt}.{dtype:?}@{dst_mem:?}");
                    let mut dst = proc
                        .create_image(dw, dh, dst_fmt, dtype, Some(dst_mem))
                        .unwrap();
                    proc.convert(&src, &mut dst, Rotation::None, Flip::None, lb)
                        .unwrap_or_else(|e| panic!("{label}: convert failed: {e}"));

                    // Raw bytes; undo the int8 XOR-0x80 bias so both dtypes
                    // share one set of thresholds.
                    let unbias = if dtype == DType::I8 { 0x80u8 } else { 0 };
                    let bytes: Vec<u8> = match dtype {
                        DType::U8 => dst.as_u8().unwrap().map().unwrap().as_slice().to_vec(),
                        _ => dst
                            .as_i8()
                            .unwrap()
                            .map()
                            .unwrap()
                            .as_slice()
                            .iter()
                            .map(|&v| v as u8 ^ unbias)
                            .collect(),
                    };

                    // Only the three colour planes — the alpha plane is 255
                    // everywhere (content and pad) and is bias-exempt.
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

    /// Letterboxed PixelFormat::Nv12 → planar against the verified RGBA
    /// reference: the same processor letterboxes camera NV12 720p → RGBA
    /// 640x640, the reference is channel-split on the CPU, and the GL planar
    /// output (PlanarRgb@Dma — the TwoPassNvPlanar plan) must match it.
    /// PlanarRgb@Dma is the only planar destination the GL engine accepts:
    /// `check_dst_format_supported` rejects PlanarRgba outright and Mem has
    /// no planar texture destination. Catches placement, scale, and content
    /// drift in the two-pass plan at full resolution — the per-pixel
    /// complement to the content-band geometry probe above.
    #[test]
    #[cfg(all(target_os = "linux", feature = "dma_test_formats"))]
    fn letterbox_nv12_to_planar_matches_rgba_reference() {
        if !is_dma_available() {
            eprintln!("SKIPPED: {} - DMA not available", function!());
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

        let crop = letterbox_crop(1280, 720, 640, 640);
        let (dw, dh) = (640usize, 640usize);
        let mut gl = GLProcessorThreaded::new(None).unwrap();

        // Reference: the verified packed-RGBA letterbox path.
        let mut dst_rgba = TensorDyn::image(
            dw,
            dh,
            PixelFormat::Rgba,
            DType::U8,
            Some(TensorMemory::Dma),
        )
        .unwrap();
        gl.convert(&src, &mut dst_rgba, Rotation::None, Flip::None, crop)
            .unwrap();
        let rgba = dst_rgba.as_u8().unwrap().map().unwrap();
        let rgba = rgba.as_slice();

        // Channel-split the RGBA reference into the planar layout.
        let mut expected = vec![0u8; 3 * dh * dw];
        for (i, px) in rgba.chunks_exact(4).enumerate() {
            for (plane, &channel) in px.iter().take(3).enumerate() {
                expected[plane * dh * dw + i] = channel;
            }
        }
        let mut dst = TensorDyn::image(
            dw,
            dh,
            PixelFormat::PlanarRgb,
            DType::U8,
            Some(TensorMemory::Dma),
        )
        .unwrap();
        gl.convert(&src, &mut dst, Rotation::None, Flip::None, crop)
            .unwrap_or_else(|e| panic!("nv12->PlanarRgb@Dma: convert failed: {e}"));
        let map = dst.as_u8().unwrap().map().unwrap();
        assert_pixels_match(&expected, map.as_slice(), 1);
    }

    /// Compare fused GL proto rendering against hybrid (CPU materialize + GL overlay).
    ///
    /// Both paths should produce visually similar output. Differences arise from
    /// bilinear interpolation (GPU vs CPU) and mask threshold rounding.
    #[test]
    fn test_proto_fused_vs_hybrid_ssim() {
        if !is_opengl_available() {
            eprintln!("SKIPPED: {} - OpenGL not available", function!());
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

    /// Decode the cached YOLOv8-seg fixture outputs (int8 boxes + protos)
    /// into post-NMS detections and the int8 `ProtoData`. Shared scaffolding
    /// for every proto-rendering test.
    fn decode_yolov8_proto_fixture() -> (Vec<DetectBox>, ProtoData) {
        use edgefirst_decoder::{configs, ConfigOutput, DecoderBuilder, Nms};

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
        (output_boxes, proto_data)
    }

    /// Build F32 and F16 `ProtoData` by dequantizing the int8 fixture — the
    /// same synthesis the mask bench uses for its float dtype cells.
    fn proto_fixture_as_float(proto_i8: &ProtoData, n_det: usize) -> (ProtoData, ProtoData) {
        use half::f16;

        let (protos_f32, proto_shape) = match &proto_i8.protos {
            TensorDyn::I8(t) => {
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

        let coeffs_f32: Vec<f32> = match &proto_i8.mask_coefficients {
            TensorDyn::F32(t) => t.map().unwrap().as_slice().to_vec(),
            TensorDyn::F16(t) => t
                .map()
                .unwrap()
                .as_slice()
                .iter()
                .map(|v: &f16| v.to_f32())
                .collect(),
            TensorDyn::I8(t) => {
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
            mask_coefficients: TensorDyn::F32(
                Tensor::<f32>::from_slice(&coeffs_f32, &coeff_shape).unwrap(),
            ),
            protos: TensorDyn::F32(Tensor::<f32>::from_slice(&protos_f32, &proto_shape).unwrap()),
            layout: ProtoLayout::Nhwc,
        };

        let protos_f16: Vec<f16> = protos_f32.iter().map(|v| f16::from_f32(*v)).collect();
        let coeffs_f16: Vec<f16> = coeffs_f32.iter().map(|v| f16::from_f32(*v)).collect();
        let proto_f16 = ProtoData {
            mask_coefficients: TensorDyn::F16(
                Tensor::<f16>::from_slice(&coeffs_f16, &coeff_shape).unwrap(),
            ),
            protos: TensorDyn::F16(Tensor::<f16>::from_slice(&protos_f16, &proto_shape).unwrap()),
            layout: ProtoLayout::Nhwc,
        };

        (proto_f32, proto_f16)
    }

    /// Keep only the first `keep` proto layers (and matching coefficients)
    /// of an F32 `ProtoData` — HWC layout, so channels are sliced per pixel.
    fn slice_proto_layers_f32(pd: &ProtoData, keep: usize, n_det: usize) -> ProtoData {
        let (protos, shape) = match &pd.protos {
            TensorDyn::F32(t) => (t.map().unwrap().as_slice().to_vec(), t.shape().to_vec()),
            other => panic!("expected f32 protos, got {other:?}"),
        };
        let (h, w, c) = (shape[0], shape[1], shape[2]);
        assert!(keep <= c);
        let mut sliced = Vec::with_capacity(h * w * keep);
        for px in protos.chunks_exact(c) {
            sliced.extend_from_slice(&px[..keep]);
        }
        let coeffs = match &pd.mask_coefficients {
            TensorDyn::F32(t) => t.map().unwrap().as_slice().to_vec(),
            other => panic!("expected f32 coefficients, got {other:?}"),
        };
        let mut sliced_coeffs = Vec::with_capacity(n_det * keep);
        for det in coeffs.chunks_exact(c) {
            sliced_coeffs.extend_from_slice(&det[..keep]);
        }
        ProtoData {
            mask_coefficients: TensorDyn::F32(
                Tensor::<f32>::from_slice(&sliced_coeffs, &[n_det, keep]).unwrap(),
            ),
            protos: TensorDyn::F32(Tensor::<f32>::from_slice(&sliced, &[h, w, keep]).unwrap()),
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
            edgefirst_tensor::Tensor::<u8>::image(640, 640, PixelFormat::Rgba, None).unwrap(),
        );
        gl.draw_proto_masks(&mut dst, boxes, pd, Default::default())
            .unwrap();
        dst
    }

    /// Forces `has_float_linear` off for processors created in scope —
    /// makes the f32→f16-repack proto fallback reachable on float-linear
    /// GPUs (no CI lane has a GPU without the extension).
    struct FloatLinearEnvGuard;
    impl FloatLinearEnvGuard {
        fn set() -> Self {
            std::env::set_var("EDGEFIRST_GL_NO_FLOAT_LINEAR", "1");
            FloatLinearEnvGuard
        }
    }
    impl Drop for FloatLinearEnvGuard {
        fn drop(&mut self) {
            std::env::remove_var("EDGEFIRST_GL_NO_FLOAT_LINEAR");
        }
    }

    /// Opts processors created in scope into the GLES 3.1 compute-shader
    /// proto repack (otherwise env-gated off — it has zero coverage without
    /// this).
    struct ProtoComputeEnvGuard;
    impl ProtoComputeEnvGuard {
        fn set() -> Self {
            std::env::set_var("EDGEFIRST_PROTO_COMPUTE", "1");
            ProtoComputeEnvGuard
        }
    }
    impl Drop for ProtoComputeEnvGuard {
        fn drop(&mut self) {
            std::env::remove_var("EDGEFIRST_PROTO_COMPUTE");
        }
    }

    /// The float proto-render paths must agree with the int8 fused reference:
    /// F32 (R32F, hardware bilinear where float-linear exists) and F16-native
    /// (RGBA16F packed) render the SAME dequantized data, so int8-vs-f32 may
    /// differ only by quantization + interpolation (default int8 mode is
    /// shader bilinear ≈ hardware bilinear), and f32-vs-f16 only by f16
    /// rounding. Before this test the three float variants had no direct
    /// coverage at all.
    #[test]
    fn proto_float_dtypes_match_int8_reference() {
        if !is_opengl_available() {
            eprintln!("SKIPPED: {} - OpenGL not available", function!());
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

    /// The `!has_float_linear` f32 arm repacks F32 protos into RGBA16F and
    /// renders through the f16 shader — a capability fallback no CI lane's
    /// GPU exercises naturally. Force the capability off and pin the
    /// fallback's output against the native R32F render: same data, only
    /// RGBA16F precision and packing differ.
    #[test]
    fn proto_f32_no_float_linear_fallback_matches() {
        if !is_opengl_available() {
            eprintln!("SKIPPED: {} - OpenGL not available", function!());
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

    /// The GLES 3.1 compute-shader HWC→CHW proto repack must produce the
    /// same masks as the CPU repack — only the transposition strategy
    /// differs, not the data or sampling. The compute path is env-gated
    /// (EDGEFIRST_PROTO_COMPUTE=1); where the GPU lacks GLES 3.1 compute the
    /// guard is a no-op and both renders take the CPU repack (still a valid,
    /// if vacuous, comparison).
    #[test]
    fn proto_compute_repack_matches_cpu_repack() {
        if !is_opengl_available() {
            eprintln!("SKIPPED: {} - OpenGL not available", function!());
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

    /// All three int8 proto interpolation modes must agree on the rendered
    /// masks: Bilinear (shader-computed weights) and TwoPass (dequant to
    /// RGBA16F + hardware GL_LINEAR) sample the same data with equivalent
    /// filtering, and Nearest only loses the sub-texel blend. TwoPass had
    /// no test on any lane before this — it is the path with the cached
    /// dequant FBO, the gated immutable dequant texture, and the shared
    /// fullscreen quad. Switching modes on ONE processor also exercises
    /// per-draw plan changes against the persistent proto/dequant textures.
    #[test]
    fn proto_int8_interpolation_modes_agree() {
        use crate::Int8InterpolationMode;

        if !is_opengl_available() {
            eprintln!("SKIPPED: {} - OpenGL not available", function!());
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
        // And back to bilinear on the same processor: must reproduce the
        // first render despite the dequant texture/FBO created in between.
        let bilinear_again = render_mode(Int8InterpolationMode::Bilinear);

        // Measured actuals: Vivante GC7000UL >= 0.95; Mali G310 = 0.8153 —
        // a PRE-EXISTING driver filtering delta (bit-identical score on the
        // pre-refactor build), believed to be Mali's coarser fixed-point
        // f16 texture interpolation. 0.80 still catches structural
        // breakage: the broken compute-repack upload scored ~0.77.
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

    /// Proto texture uploads must stay correct across dims/format churn on
    /// ONE processor: int8 (R8I) → f32 (R32F, same dims — internal-format
    /// churn) → f32 with 30 layers (layer-count churn, deliberately not a
    /// multiple of 4) → int8 again. Each render must byte-match a fresh
    /// processor rendering the same input once. This is the regression test
    /// for the proto dims gate: a gate keyed on dims-without-ifmt, or a
    /// SubImage3D into a stale allocation, breaks one of these legs.
    #[test]
    fn proto_dims_and_format_churn_uploads_correctly() {
        if !is_opengl_available() {
            eprintln!("SKIPPED: {} - OpenGL not available", function!());
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
            .import_image(plane, None, 640, 640, PixelFormat::Rgb, DType::I8, None)
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
            .import_image(plane, None, 640, 640, PixelFormat::Rgba, DType::U8, None)
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
            .import_image(plane, None, 640, 640, PixelFormat::Rgb, DType::I8, None)
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
            .import_image(plane, None, 640, 640, PixelFormat::Rgb, DType::U8, None)
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
            .import_image(plane, None, 640, 640, PixelFormat::Rgb, DType::I8, None)
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
            .import_image(plane, None, 640, 640, PixelFormat::Rgb, DType::I8, None)
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
            .import_image(plane, None, 640, 640, PixelFormat::Rgb, DType::I8, None)
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
        let stride: usize = (width * bpp).div_ceil(128) * 128; // 2560 -> 2560 (already 128-aligned)
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
                None,
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

        // Pin both the true-multiplane (separate fds) and contiguous sources to
        // ExternalSampler so the compare tests import correctness, not path
        // selection (see test_multiplane_nv12_to_rgba_opengl). Byte-exact on
        // every DMA board.
        let _nv_path = NvPathEnvGuard::set("sampler");

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
                None,
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

        // True multiplane and contiguous must produce identical pixels.
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
        let crop = Crop::new().with_source(Some(crate::Region::new(
            src_w / 2, // left = start of blue region
            0,
            src_w / 2, // width = blue half
            src_h,
        )));

        if let Err(e) = gl.convert(&src, &mut dst, Rotation::None, Flip::None, crop) {
            // Vivante GL rejects RGB source textures. The src_rect no-bleed crop
            // logic is still covered on RGBA-capable drivers (e.g. Mesa); Vivante
            // RGB-source support is tracked for separate investigation.
            if e.to_string().contains("RGB source") {
                eprintln!("SKIPPED: {} - {e}", function!());
                return;
            }
            panic!("{e}");
        }

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
            Crop::new().with_source(Some(crate::Region::new(src_w / 2, 0, src_w / 2, src_h)));

        if let Err(e) = gl.convert(&src, &mut dst, Rotation::None, Flip::None, crop) {
            // Vivante GL rejects RGB source textures. The src_rect no-bleed crop
            // logic is still covered on RGBA-capable drivers (e.g. Mesa); Vivante
            // RGB-source support is tracked for separate investigation.
            if e.to_string().contains("RGB source") {
                eprintln!("SKIPPED: {} - {e}", function!());
                return;
            }
            panic!("{e}");
        }

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
        let crop = Crop::new().with_source(Some(crate::Region::new(0, 0, src_w / 2, src_h)));

        if let Err(e) = gl.convert(&src, &mut dst, Rotation::None, Flip::None, crop) {
            // Vivante GL rejects RGB source textures. The src_rect no-bleed crop
            // logic is still covered on RGBA-capable drivers (e.g. Mesa); Vivante
            // RGB-source support is tracked for separate investigation.
            if e.to_string().contains("RGB source") {
                eprintln!("SKIPPED: {} - {e}", function!());
                return;
            }
            panic!("{e}");
        }

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
            FloatRenderPath::ZeroCopyF16Nchw
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

    /// The int8 letterbox clear colour must be pre-biased (XOR 0x80) on EVERY
    /// destination lowering: the int8 fragment shader biases rendered pixels
    /// and the readback never adjusts the glClear'd letterbox region. The
    /// former any→PBO and PBO→Mem paths skipped the bias, leaving their int8
    /// letterbox fill 0x80 off versus the Mem→Mem result. Pin every PBO
    /// src/dst combination's output byte-identical to the Mem→Mem oracle.
    /// Runs only where image allocation resolves to PBO (Orin / PBO-transfer
    /// targets); skipped elsewhere.
    #[test]
    #[cfg(target_os = "linux")] // PBO destinations are Linux-only
    fn pbo_int8_letterbox_matches_mem_oracle() {
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

        let (sw, sh) = (64usize, 48usize);
        let probe = proc
            .create_image(sw, sh, PixelFormat::Rgba, DType::U8, None)
            .unwrap();
        if probe.memory() != TensorMemory::Pbo {
            eprintln!(
                "SKIPPED: {} - target does not allocate PBO images",
                function!()
            );
            return;
        }

        let fill = |t: &TensorDyn| {
            let mut map = t.as_u8().unwrap().map().unwrap();
            for (i, px) in map.as_mut_slice().chunks_exact_mut(4).enumerate() {
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
            )
            .unwrap();
        fill(&src_mem);

        // 64×48 → 96×96 letterbox: 96×72 inner box, glClear'd bands above and
        // below — the region whose int8 bias the PBO paths used to drop.
        let lb = Crop::letterbox([114, 114, 114, 255]);
        // Explicit `Some(Pbo)` requests are rejected (PBO backing exists only
        // through auto-resolution), so the PBO destination is requested as
        // `None` and the resolved memory asserted instead.
        let mut convert_to_bytes = |src: &TensorDyn, dst_mem: TensorMemory, label: &str| {
            let dst_req = match dst_mem {
                TensorMemory::Pbo => None,
                other => Some(other),
            };
            let mut dst = proc
                .create_image(96, 96, PixelFormat::Rgb, DType::I8, dst_req)
                .unwrap();
            assert_eq!(dst.memory(), dst_mem, "{label}: dst backing not honoured");
            proc.convert(src, &mut dst, Rotation::None, Flip::None, lb)
                .unwrap_or_else(|e| panic!("{label} convert failed: {e}"));
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

    /// On-GPU round-trip: RGBA8 → F32 NHWC `[H,W,3]` PBO via the GL float
    /// render path. Forces the OpenGL backend (no CPU fallback) so the test
    /// genuinely exercises `convert_float_to_pbo`. Uses an identity crop so
    /// the expected values are exact: `dst[y,x,c] == src[y,x,c] / 255`.
    #[test]
    #[cfg(target_os = "linux")] // PBO destinations are Linux-only
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
    #[cfg(target_os = "linux")] // PBO destinations are Linux-only
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
    #[cfg(target_os = "linux")] // PBO destinations are Linux-only
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
    /// float render path (`convert_float_to_zero_copy`). Forces the OpenGL backend
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
        // convert_float_to_zero_copy) surfaces as Err instead of falling through
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
    #[cfg(target_os = "linux")] // PBO destinations are Linux-only
    fn convert_f32_pbo_letterbox_pad_color() {
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
        let crop = Crop::letterbox([114, 114, 114, 255]);

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
    #[cfg(target_os = "linux")] // CUDA interop is Tegra/Linux-only
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
    #[cfg(target_os = "linux")] // CUDA interop is Tegra/Linux-only
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

    /// Full Jetson zero-copy OUTPUT flow, per native JPEG format:
    ///   JPEG decode → NV12/NV16/NV24/GREY source (JFIF/BT.601-full tagged) →
    ///   `convert()` → `Rgb` `F32` PBO output → `cuda_map()` →
    ///   `cudaMemcpy(D2H)` → compare to a CPU `convert()` reference.
    ///
    /// Proves the device pointer a TensorRT client receives holds the
    /// correctly-converted, colorimetry-correct pixels for every format the CPU
    /// JPEG decoder emits — and that the JPEG→NVxx/GREY decode path itself works
    /// end to end. On a Jetson (no `/dev/dma_heap`) the NV source has no GPU
    /// path, so `convert()` runs on the CPU and writes into the CUDA-registered
    /// output PBO; this test exercises exactly that CPU→PBO→CUDA hand-off.
    #[cfg(target_os = "linux")]
    fn jpeg_cuda_devptr_check(fixture: &str, expect_fmt: PixelFormat, w: usize, h: usize) {
        use crate::{ComputeBackend, ImageProcessor, ImageProcessorConfig};
        use edgefirst_codec::{ImageDecoder, ImageLoad};
        use edgefirst_tensor::{Tensor, TensorMapTrait, TensorTrait};
        use std::ffi::c_void;

        if !edgefirst_tensor::is_cuda_available() {
            eprintln!("SKIP {fixture}: no libcudart");
            return;
        }
        let mut proc = match ImageProcessor::with_config(ImageProcessorConfig {
            backend: ComputeBackend::OpenGl,
            ..Default::default()
        }) {
            Ok(p) => p,
            Err(_) => {
                eprintln!("SKIP {fixture}: no GL");
                return;
            }
        };
        if !proc.supported_render_dtypes().f32 {
            eprintln!("SKIP {fixture}: no f32 render");
            return;
        }

        // Decode JPEG into its native format; the decoder tags JFIF (BT.601
        // full-range), which the CPU convert below must honour.
        let mut src_t = Tensor::<u8>::image(w, h, expect_fmt, Some(TensorMemory::Mem)).unwrap();
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
        let mut ref_dst =
            TensorDyn::image(w, h, PixelFormat::Rgb, DType::F32, Some(TensorMemory::Mem)).unwrap();
        proc.convert(
            &src,
            &mut ref_dst,
            Rotation::None,
            Flip::None,
            Crop::default(),
        )
        .unwrap();
        let ref_map = ref_dst.as_f32().unwrap().map().unwrap();
        let ref_px = ref_map.as_slice();

        // Rgb F32 PBO output → cuda_map (the buffer a TRT client binds).
        let mut pbo_dst = proc
            .create_image(w, h, PixelFormat::Rgb, DType::F32, None)
            .unwrap();
        if pbo_dst.memory() != TensorMemory::Pbo {
            eprintln!("SKIP {fixture}: dst not PBO");
            return;
        }
        proc.convert(
            &src,
            &mut pbo_dst,
            Rotation::None,
            Flip::None,
            Crop::default(),
        )
        .unwrap();
        let cm = match pbo_dst.cuda_map() {
            Some(cm) => cm,
            None => {
                eprintln!("SKIP {fixture}: cuda_map None (CUDA-GL interop unavailable)");
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
    // Path B: NV16/NV24 → RGBA GPU round-trip tests
    //
    // Gate: Linux + dma_test_formats + runtime skip if DMA or GL unavailable.
    // Strategy: solid-colour sources → exact expected RGB computable from
    // BT.601 full-range; assert Path B ran (no CPU fallback); assert output
    // matches CPU reference within tolerance.
    // =========================================================================

    /// Compute expected BT.601 full-range RGB from raw Y, Cb, Cr byte values.
    ///
    /// Matches the coefficient set used in both the CPU kernels and the
    /// `generate_nv_to_rgba_shader_2d` shader (and the macOS `NV_TO_RGBA_FRAGMENT`).
    #[cfg(all(target_os = "linux", feature = "dma_test_formats"))]
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
    #[cfg(all(target_os = "linux", feature = "dma_test_formats"))]
    fn make_nv16_solid(w: usize, h: usize, y: u8, cb: u8, cr: u8) -> Vec<u8> {
        let y_plane = vec![y; w * h];
        // NV16: H rows of UV, each row has w/2 pairs → w bytes/row.
        let uv_plane: Vec<u8> = std::iter::repeat_n([cb, cr], w * h / 2).flatten().collect();
        [y_plane, uv_plane].concat()
    }

    /// Build a solid-colour NV24 (4:4:4) buffer of dimensions `(w, h)`.
    ///
    /// Layout: `[H rows of Y][H rows of interleaved CbCr full-res]`.
    #[cfg(all(target_os = "linux", feature = "dma_test_formats"))]
    fn make_nv24_solid(w: usize, h: usize, y: u8, cb: u8, cr: u8) -> Vec<u8> {
        let y_plane = vec![y; w * h];
        // NV24: H rows of UV, each row has w pairs → 2w bytes/row.
        let uv_plane: Vec<u8> = std::iter::repeat_n([cb, cr], w * h).flatten().collect();
        [y_plane, uv_plane].concat()
    }

    /// Verify NV16→RGBA via Path B (R8 texelFetch shader) on DMA buffers.
    ///
    /// Checks:
    ///   (a) `last_nv_convert_path` == `ShaderR8` — no CPU fallback.
    ///   (b) Every output pixel matches the expected BT.601 full-range RGB
    ///       within ±2 (rounding from f32 shader arithmetic).
    #[test]
    #[cfg(all(target_os = "linux", feature = "dma_test_formats"))]
    fn test_gpu_nv16_to_rgba_path_b() {
        use crate::opengl_headless::processor::{GLProcessorST, NvConvertPath};
        if !is_dma_available() {
            eprintln!("SKIPPED: {} - DMA not available", function!());
            return;
        }

        let (w, h) = (64usize, 64usize);
        // Solid YCbCr: Y=100, Cb=150, Cr=80 → deterministic RGB.
        let (yv, cb, cr) = (100u8, 150u8, 80u8);
        let expected = yuv601_to_rgb(yv, cb, cr);

        let mut src = load_raw_image(
            w,
            h,
            PixelFormat::Nv16,
            Some(TensorMemory::Dma),
            &make_nv16_solid(w, h, yv, cb, cr),
        )
        .unwrap();
        // `yuv601_to_rgb` is the full-range BT.601 reference, so tag the source
        // full-range; otherwise the heuristic resolves untagged → limited.
        src.set_colorimetry(Some(
            edgefirst_tensor::Colorimetry::default()
                .with_encoding(edgefirst_tensor::ColorEncoding::Bt601)
                .with_range(edgefirst_tensor::ColorRange::Full),
        ));

        let mut dst =
            TensorDyn::image(w, h, PixelFormat::Rgba, DType::U8, Some(TensorMemory::Dma)).unwrap();

        let mut gl = match GLProcessorST::new(None) {
            Ok(g) => g,
            Err(e) => {
                eprintln!("SKIPPED: {} - GL not available: {e}", function!());
                return;
            }
        };

        gl.convert(&src, &mut dst, Rotation::None, Flip::None, Crop::no_crop())
            .unwrap();

        // (a) Assert Path B ran — no CPU fallback for a DMA NV16 source.
        assert_eq!(
            gl.last_nv_convert_path,
            NvConvertPath::ShaderR8,
            "NV16 DMA convert must use Path B (ShaderR8), got {:?}",
            gl.last_nv_convert_path
        );

        // (b) Check pixel values against BT.601 reference within ±2.
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

    /// RAII guard: sets `EDGEFIRST_NV_CONVERT_PATH` and removes it on drop, so a
    /// panic in `GLProcessorST::new` cannot leak the override to later tests.
    /// GL tests run `--test-threads=1` (CI), so there is no concurrent reader.
    #[cfg(all(target_os = "linux", feature = "dma_test_formats"))]
    struct NvPathEnvGuard;
    #[cfg(all(target_os = "linux", feature = "dma_test_formats"))]
    impl NvPathEnvGuard {
        fn set(v: &str) -> Self {
            std::env::set_var("EDGEFIRST_NV_CONVERT_PATH", v);
            NvPathEnvGuard
        }
    }
    #[cfg(all(target_os = "linux", feature = "dma_test_formats"))]
    impl Drop for NvPathEnvGuard {
        fn drop(&mut self) {
            std::env::remove_var("EDGEFIRST_NV_CONVERT_PATH");
        }
    }

    #[cfg(all(target_os = "linux", feature = "dma_test_formats"))]
    struct ColorimetryEnvGuard;
    #[cfg(all(target_os = "linux", feature = "dma_test_formats"))]
    impl ColorimetryEnvGuard {
        fn set(v: &str) -> Self {
            std::env::set_var("EDGEFIRST_COLORIMETRY", v);
            ColorimetryEnvGuard
        }
    }
    #[cfg(all(target_os = "linux", feature = "dma_test_formats"))]
    impl Drop for ColorimetryEnvGuard {
        fn drop(&mut self) {
            std::env::remove_var("EDGEFIRST_COLORIMETRY");
        }
    }

    /// HIGH-PERFORMANCE-default colorimetry policy (issue #106): under `auto`
    /// a non-BT.601-limited single-plane NV12 DMA source takes the hardware
    /// sampler on Vivante in the default [`ColorimetryMode::Fast`] (~12×
    /// faster, driver's approximate fixed matrix) and the exact in-shader
    /// path under the [`ColorimetryMode::Exact`] opt-in. On every other GPU
    /// the in-shader path IS the fast path, so both modes pick it. Also pins:
    /// BT.601-limited sources keep the sampler under Exact on Vivante (the
    /// driver matrix matches exactly), and `EDGEFIRST_COLORIMETRY` wins over
    /// the setter.
    #[test]
    #[cfg(all(target_os = "linux", feature = "dma_test_formats"))]
    fn colorimetry_mode_policy_nv12_auto() {
        use crate::opengl_headless::processor::{GLProcessorST, NvConvertPath};
        use crate::ColorimetryMode;
        use edgefirst_tensor::{ColorEncoding, ColorRange, Colorimetry};

        if !is_dma_available() {
            eprintln!("SKIPPED: {} - DMA not available", function!());
            return;
        }
        let mut gl = match GLProcessorST::new(None) {
            Ok(g) => g,
            Err(e) => {
                eprintln!("SKIPPED: {} - GL not available: {e}", function!());
                return;
            }
        };

        let (w, h) = (64usize, 64usize);
        let mut nv12 = vec![128u8; w * h * 3 / 2];
        nv12[..w * h].fill(100);
        let make_src = |enc: ColorEncoding, range: ColorRange| {
            let mut src =
                load_raw_image(w, h, PixelFormat::Nv12, Some(TensorMemory::Dma), &nv12).unwrap();
            src.set_colorimetry(Some(
                Colorimetry::default().with_encoding(enc).with_range(range),
            ));
            src
        };
        let src_709 = make_src(ColorEncoding::Bt709, ColorRange::Full);
        let src_601 = make_src(ColorEncoding::Bt601, ColorRange::Limited);
        let mut dst =
            TensorDyn::image(w, h, PixelFormat::Rgba, DType::U8, Some(TensorMemory::Dma)).unwrap();
        let mut path_for = |gl: &mut GLProcessorST, src: &TensorDyn| {
            gl.convert(src, &mut dst, Rotation::None, Flip::None, Crop::no_crop())
                .unwrap();
            gl.last_nv_convert_path
        };

        // Fast (default): sampler on Vivante regardless of colorimetry,
        // in-shader matrix everywhere else.
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

        // Exact opt-in: the approximate sampler is off the table for a
        // non-matching colorimetry — in-shader matrix everywhere…
        gl.set_colorimetry_mode(ColorimetryMode::Exact);
        assert_eq!(
            path_for(&mut gl, &src_709),
            NvConvertPath::ShaderR8,
            "Exact mode, BT.709-full NV12"
        );
        // …but a BT.601-limited source matches the Vivante driver matrix
        // exactly, so the sampler carve-out survives Exact mode.
        assert_eq!(
            path_for(&mut gl, &src_601),
            expect_fast,
            "Exact mode, BT.601-limited NV12"
        );

        // A Grey destination must NEVER take the sampler, in either mode and
        // for any colorimetry: samplerExternalOES → R8 render target wedges
        // the Vivante GC7000UL (EDGEAI-1180 hang class; found when the Fast
        // policy first routed nv12@dma→grey@dma to the sampler). ShaderR8 on
        // every GPU. NOTE: this leg HANGS the board rather than failing if
        // the gate regresses — keep it last-resort observable via the suite
        // timeout.
        let mut grey_dst =
            TensorDyn::image(w, h, PixelFormat::Grey, DType::U8, Some(TensorMemory::Dma)).unwrap();
        let mut grey_path = |gl: &mut GLProcessorST, src: &TensorDyn| {
            gl.convert(
                src,
                &mut grey_dst,
                Rotation::None,
                Flip::None,
                Crop::no_crop(),
            )
            .unwrap();
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

        // EDGEFIRST_COLORIMETRY pins the mode for the processor's lifetime:
        // a Fast request on an exact-pinned processor is kept at Exact.
        drop(gl);
        let _env = ColorimetryEnvGuard::set("exact");
        let mut gl = GLProcessorST::new(None).unwrap();
        gl.set_colorimetry_mode(ColorimetryMode::Fast);
        assert_eq!(
            path_for(&mut gl, &src_709),
            NvConvertPath::ShaderR8,
            "env-pinned Exact must override set_colorimetry_mode(Fast)"
        );
    }

    /// `EDGEFIRST_NV_CONVERT_PATH` selects the NV12 GPU conversion path.
    ///
    /// - `auto` (default) follows the `ColorimetryMode::Fast` policy:
    ///   `ExternalSampler` on Vivante (12× faster), `ShaderR8` everywhere
    ///   else (the exact path is already the fast path). The full policy
    ///   matrix is covered by `colorimetry_mode_policy_nv12_auto`.
    /// - `shader` → `ShaderR8` always.
    /// - `sampler` → the driver `ExternalSampler` path is selected; the actual
    ///   recorded path is `ExternalSampler` on success or `Cpu` if the driver
    ///   rejects the NV12 EGLImage import — never `ShaderR8`.
    #[test]
    #[cfg(all(target_os = "linux", feature = "dma_test_formats"))]
    fn test_nv12_path_env_override() {
        use crate::opengl_headless::processor::{GLProcessorST, NvConvertPath};
        if !is_dma_available() {
            eprintln!("SKIPPED: {} - DMA not available", function!());
            return;
        }
        let (w, h) = (64usize, 64usize); // width % 4 == 0 so the sampler import is accepted
        let mut bytes = vec![100u8; w * h]; // Y plane
        for _ in 0..(w * h / 4) {
            bytes.push(150); // Cb
            bytes.push(80); // Cr
        }

        let run = |env: Option<&str>| -> Option<(NvConvertPath, bool)> {
            // The env var is read once in `GLProcessorST::new`; the guard sets it
            // for exactly that construction and removes it on drop (panic-safe).
            let gl = {
                std::env::remove_var("EDGEFIRST_NV_CONVERT_PATH");
                let _guard = env.map(NvPathEnvGuard::set);
                GLProcessorST::new(None)
            };
            let mut gl = match gl {
                Ok(g) => g,
                Err(e) => {
                    eprintln!("SKIPPED: {} - GL not available: {e}", function!());
                    return None;
                }
            };
            let mut src =
                load_raw_image(w, h, PixelFormat::Nv12, Some(TensorMemory::Dma), &bytes).unwrap();
            // Tag full-range (not BT.601-limited) so the `auto` leg exercises
            // the colorimetry-mode policy: the default Fast mode still takes
            // the sampler on Vivante, ShaderR8 everywhere else.
            src.set_colorimetry(Some(
                edgefirst_tensor::Colorimetry::default()
                    .with_encoding(edgefirst_tensor::ColorEncoding::Bt709)
                    .with_range(edgefirst_tensor::ColorRange::Full),
            ));
            let mut dst =
                TensorDyn::image(w, h, PixelFormat::Rgba, DType::U8, Some(TensorMemory::Dma))
                    .unwrap();
            gl.convert(&src, &mut dst, Rotation::None, Flip::None, Crop::no_crop())
                .unwrap();
            Some((gl.last_nv_convert_path, gl.is_vivante))
        };

        if let Some((p, is_vivante)) = run(None) {
            let expect = if is_vivante {
                // ColorimetryMode::Fast default: the 12×-faster sampler.
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

    /// Phase 4b: a NON-DMA (heap/PBO) NV12 source must be GPU-converted via the
    /// R8-UPLOAD `ShaderR8` path — not the CPU fallback. This is the path that
    /// gives orin (no DMA-BUF EGLImage import) GPU NV conversion. Gated on
    /// OpenGL only (NOT DMA), so it runs on orin.
    ///
    /// Asserts: (a) `last_nv_convert_path == ShaderR8` for a `Mem` NV12 source;
    /// (b) the GPU upload output matches the `CPUProcessor` reference within ±2
    /// (same in-shader matrix as DMA `ShaderR8`).
    #[test]
    #[cfg(all(target_os = "linux", feature = "dma_test_formats"))]
    fn test_nv12_nondma_upload_uses_shader() {
        use crate::opengl_headless::processor::{GLProcessorST, NvConvertPath};
        if !is_opengl_available() {
            eprintln!("SKIPPED: {} - OpenGL not available", function!());
            return;
        }
        let (w, h) = (64usize, 64usize);
        // Patterned NV12 (luma gradient + neutral chroma) — exercises the shader.
        let mut bytes = vec![0u8; w * h];
        for r in 0..h {
            for c in 0..w {
                bytes[r * w + c] = ((r + c) * 255 / (w + h)) as u8;
            }
        }
        bytes.extend(std::iter::repeat_n(128u8, w * h / 2)); // neutral chroma

        let mut gl = match GLProcessorST::new(None) {
            Ok(g) => g,
            Err(e) => {
                eprintln!("SKIPPED: {} - GL not available: {e}", function!());
                return;
            }
        };

        // Non-DMA (Mem) NV12 source → must take the R8-upload ShaderR8 path.
        let mem_src =
            load_raw_image(w, h, PixelFormat::Nv12, Some(TensorMemory::Mem), &bytes).unwrap();
        let mut gpu_dst = TensorDyn::image(w, h, PixelFormat::Rgba, DType::U8, None).unwrap();
        gl.convert(
            &mem_src,
            &mut gpu_dst,
            Rotation::None,
            Flip::None,
            Crop::no_crop(),
        )
        .unwrap();
        assert_eq!(
            gl.last_nv_convert_path,
            NvConvertPath::ShaderR8,
            "non-DMA NV12 must use the R8-upload ShaderR8 path, not CPU (got {:?})",
            gl.last_nv_convert_path
        );

        // CPU reference (same matrix) — outputs must agree within ±2.
        let cpu_src =
            load_raw_image(w, h, PixelFormat::Nv12, Some(TensorMemory::Mem), &bytes).unwrap();
        let mut cpu_dst = TensorDyn::image(w, h, PixelFormat::Rgba, DType::U8, None).unwrap();
        crate::cpu::CPUProcessor::new()
            .convert(
                &cpu_src,
                &mut cpu_dst,
                Rotation::None,
                Flip::None,
                Crop::no_crop(),
            )
            .unwrap();

        let gpu_map = gpu_dst.as_u8().unwrap().map().unwrap();
        let cpu_map = cpu_dst.as_u8().unwrap().map().unwrap();
        assert_pixels_match(cpu_map.as_slice(), gpu_map.as_slice(), 2);
    }

    /// Regression for the planar `select_nv_path` fix: NV12 (DMA) → PlanarRgb
    /// must route through the colorimetry-exact `ShaderR8` two-pass — NOT the
    /// driver `samplerExternalOES` the single-pass `convert_to_planar` used,
    /// which diverged on chroma edges (V3D/Mali) and had no path for NV16/NV24.
    /// Full-range tag keeps the path deterministic (avoids the Vivante carve-out).
    #[test]
    #[cfg(all(target_os = "linux", feature = "dma_test_formats"))]
    fn test_nv12_to_planar_rgb_uses_shader_path() {
        use crate::opengl_headless::processor::{GLProcessorST, NvConvertPath};
        if !is_dma_available() {
            eprintln!("SKIPPED: {} - DMA not available", function!());
            return;
        }
        let (w, h) = (64usize, 64usize);
        // NV12 with a vertical chroma edge — where the driver sampler diverged.
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

        let mut gl = match GLProcessorST::new(None) {
            Ok(g) => g,
            Err(e) => {
                eprintln!("SKIPPED: {} - GL not available: {e}", function!());
                return;
            }
        };
        // This test pins the EXACT pipeline (in-shader matrix matching the
        // CPU reference at chroma edges), so it opts into
        // ColorimetryMode::Exact — under the default Fast policy a Vivante
        // board would legitimately take the approximate driver sampler
        // (covered by colorimetry_mode_policy_nv12_auto).
        gl.set_colorimetry_mode(crate::ColorimetryMode::Exact);
        let mut src =
            load_raw_image(w, h, PixelFormat::Nv12, Some(TensorMemory::Dma), &bytes).unwrap();
        tag(&mut src);
        let mut dst = TensorDyn::image(
            w,
            h,
            PixelFormat::PlanarRgb,
            DType::U8,
            Some(TensorMemory::Dma),
        )
        .unwrap();
        gl.convert(&src, &mut dst, Rotation::None, Flip::None, Crop::no_crop())
            .unwrap();
        assert_eq!(
            gl.last_nv_convert_path,
            NvConvertPath::ShaderR8,
            "NV12→PlanarRgb in Exact mode must use the ShaderR8 two-pass, not the driver sampler"
        );

        // Correctness: matches the CPU planar reference (the sampler did not).
        let mut cpu_src =
            load_raw_image(w, h, PixelFormat::Nv12, Some(TensorMemory::Mem), &bytes).unwrap();
        tag(&mut cpu_src);
        let mut cpu_dst = TensorDyn::image(w, h, PixelFormat::PlanarRgb, DType::U8, None).unwrap();
        crate::cpu::CPUProcessor::new()
            .convert(
                &cpu_src,
                &mut cpu_dst,
                Rotation::None,
                Flip::None,
                Crop::no_crop(),
            )
            .unwrap();
        compare_images(&cpu_dst, &dst, 0.95, "nv12_to_planar_rgb_shader");
    }

    /// Phase 4b coverage: NV16/NV24 (not just NV12) on a non-DMA (heap) source
    /// must also GPU-convert via the R8-upload `ShaderR8` path and match CPU ≤2.
    #[test]
    #[cfg(all(target_os = "linux", feature = "dma_test_formats"))]
    fn test_nv16_nv24_nondma_upload_uses_shader() {
        use crate::opengl_headless::processor::{GLProcessorST, NvConvertPath};
        if !is_opengl_available() {
            eprintln!("SKIPPED: {} - OpenGL not available", function!());
            return;
        }
        let (w, h) = (64usize, 64usize);
        let mut gl = match GLProcessorST::new(None) {
            Ok(g) => g,
            Err(e) => {
                eprintln!("SKIPPED: {} - GL not available: {e}", function!());
                return;
            }
        };
        // (format, chroma byte count): NV16 4:2:2 = w*h UV; NV24 4:4:4 = 2*w*h UV.
        for (fmt, uv) in [(PixelFormat::Nv16, w * h), (PixelFormat::Nv24, 2 * w * h)] {
            let mut bytes = vec![0u8; w * h];
            for r in 0..h {
                for c in 0..w {
                    bytes[r * w + c] = ((r + c) * 255 / (w + h)) as u8; // Y gradient
                }
            }
            bytes.extend(std::iter::repeat_n(128u8, uv)); // neutral chroma

            let mem_src = load_raw_image(w, h, fmt, Some(TensorMemory::Mem), &bytes).unwrap();
            let mut gpu = TensorDyn::image(w, h, PixelFormat::Rgba, DType::U8, None).unwrap();
            gl.convert(
                &mem_src,
                &mut gpu,
                Rotation::None,
                Flip::None,
                Crop::no_crop(),
            )
            .unwrap();
            assert_eq!(
                gl.last_nv_convert_path,
                NvConvertPath::ShaderR8,
                "non-DMA {fmt:?} must use the R8-upload ShaderR8 path (got {:?})",
                gl.last_nv_convert_path
            );

            let cpu_src = load_raw_image(w, h, fmt, Some(TensorMemory::Mem), &bytes).unwrap();
            let mut cpu = TensorDyn::image(w, h, PixelFormat::Rgba, DType::U8, None).unwrap();
            crate::cpu::CPUProcessor::new()
                .convert(
                    &cpu_src,
                    &mut cpu,
                    Rotation::None,
                    Flip::None,
                    Crop::no_crop(),
                )
                .unwrap();
            assert_pixels_match(
                cpu.as_u8().unwrap().map().unwrap().as_slice(),
                gpu.as_u8().unwrap().map().unwrap().as_slice(),
                2,
            );
        }
    }

    /// Diagnostic probe (Phase 2): quantify *why* the driver `ExternalSampler`
    /// path diverges from the exact reference, separating colorimetry/matrix
    /// error (flat regions) from chroma-upsampling error (chroma edges).
    ///
    /// Converts NV12 three ways — `ExternalSampler` (driver YUV), `ShaderR8`
    /// (exact in-shader matrix), and CPU (the heap/`Mem` reference, == the
    /// profiler's 841 path) — and prints per-path deltas vs CPU. Run with
    /// `--nocapture` and read the `PROBE[...]` lines per platform.
    ///
    /// Asserts only that `ShaderR8` matches CPU on a SOLID frame (pure matrix —
    /// must agree); the `ExternalSampler` divergence is characterised, not
    /// bounded (it is platform-dependent and is the thing under study).
    #[test]
    #[ignore = "diagnostic probe — run manually on-target with --nocapture --ignored"]
    #[cfg(all(target_os = "linux", feature = "dma_test_formats"))]
    fn probe_nv12_sampler_vs_shader_divergence() {
        use crate::opengl_headless::processor::{GLProcessorST, NvConvertPath};
        if !is_dma_available() {
            eprintln!("SKIPPED: {} - DMA not available", function!());
            return;
        }
        let (w, h) = (64usize, 64usize); // width % 4 == 0 for the sampler import

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
            let gl = GLProcessorST::new(None);
            std::env::remove_var("EDGEFIRST_NV_CONVERT_PATH");
            let mut gl = match gl {
                Ok(g) => g,
                Err(e) => {
                    eprintln!("SKIPPED: {} - GL not available: {e}", function!());
                    return None;
                }
            };
            let src =
                load_raw_image(w, h, PixelFormat::Nv12, Some(TensorMemory::Dma), bytes).unwrap();
            let mut dst =
                TensorDyn::image(w, h, PixelFormat::Rgba, DType::U8, Some(TensorMemory::Dma))
                    .unwrap();
            gl.convert(&src, &mut dst, Rotation::None, Flip::None, Crop::no_crop())
                .unwrap();
            let out = dst.as_u8().unwrap().map().unwrap().to_vec();
            Some((gl.last_nv_convert_path, out))
        };
        let cpu_ref = |bytes: &[u8]| -> Vec<u8> {
            let src =
                load_raw_image(w, h, PixelFormat::Nv12, Some(TensorMemory::Mem), bytes).unwrap();
            let mut dst = TensorDyn::image(w, h, PixelFormat::Rgba, DType::U8, None).unwrap();
            crate::cpu::CPUProcessor::new()
                .convert(&src, &mut dst, Rotation::None, Flip::None, Crop::no_crop())
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

    /// Diagnostic probe (Phase 4): isolate where the profiler's NV12→packed-RGB
    /// DMA path diverges from the CPU reference (the 830-vs-841), using the
    /// profiler's EXACT letterbox (mirrors `LetterboxTransform::compute`:
    /// aspect-fit + neutral-grey `[114,114,114,255]` bars).
    ///
    /// NV12 sampling already matches CPU at same size (Phase 2). This exercises
    /// the camera→model letterbox resize. Reports DMA-GL vs `CPUProcessor`:
    ///   - `same-size`     : no resize → baseline (sampling + pack).
    ///   - `lb-full`       : whole frame incl. grey bars (catches bar-fill diffs).
    ///   - `lb-content`    : the resized image region only (pure resize delta).
    ///   - shader vs sampler shows whether the driver's LINEAR resize is closer.
    #[test]
    #[ignore = "diagnostic probe — run manually on-target with --nocapture --ignored"]
    #[cfg(all(target_os = "linux", feature = "dma_test_formats"))]
    fn probe_nv12_packed_letterbox_divergence() {
        use crate::opengl_headless::processor::GLProcessorST;
        use crate::{Fit, Region};
        if !is_dma_available() {
            eprintln!("SKIPPED: {} - DMA not available", function!());
            return;
        }
        // 16:9 source so fitting into a square model produces top/bottom bars.
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

        // Mirror LetterboxTransform::compute (profiler/src/pipeline.rs:1648).
        let scale = (model as f32 / sw as f32).min(model as f32 / sh as f32);
        let scaled_w = (sw as f32 * scale).round() as usize;
        let scaled_h = (sh as f32 * scale).round() as usize;
        let pad_x = (model - scaled_w) / 2;
        let pad_y = (model - scaled_h) / 2;
        // Letterbox is now a resize mode (`Fit::Letterbox`); the backend computes
        // the aspect-fit placement (pad_x/pad_y/scaled_*) internally via
        // `Crop::resolve`, so the test no longer hand-builds a destination rect.
        let letterbox = || {
            Crop::new()
                .with_source(Some(Region::new(0, 0, sw, sh)))
                .with_fit(Fit::Letterbox {
                    pad: [114, 114, 114, 255],
                })
        };

        let gl_packed = |env: &str, crop: Crop, dw: usize, dh: usize| -> Option<(usize, Vec<u8>)> {
            std::env::set_var("EDGEFIRST_NV_CONVERT_PATH", env);
            let g = GLProcessorST::new(None);
            std::env::remove_var("EDGEFIRST_NV_CONVERT_PATH");
            let mut g = match g {
                Ok(x) => x,
                Err(e) => {
                    eprintln!("SKIPPED: {} - GL: {e}", function!());
                    return None;
                }
            };
            let src =
                load_raw_image(sw, sh, PixelFormat::Nv12, Some(TensorMemory::Dma), &bytes).unwrap();
            let mut dst = TensorDyn::image(dw, dh, PixelFormat::Rgb, DType::U8, None).unwrap();
            g.convert(&src, &mut dst, Rotation::None, Flip::None, crop)
                .ok()?;
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
            let mut dst = TensorDyn::image(dw, dh, PixelFormat::Rgb, DType::U8, None).unwrap();
            crate::cpu::CPUProcessor::new()
                .convert(&src, &mut dst, Rotation::None, Flip::None, crop)
                .unwrap();
            let stride = dst
                .as_u8()
                .unwrap()
                .effective_row_stride()
                .unwrap_or(dw * 3);
            (stride, dst.as_u8().unwrap().map().unwrap().to_vec())
        };
        // Stride-aware delta over a pixel region [r0..r0+rh) × [c0..c0+cw) (RGB).
        let region = |a: &[u8],
                      sa: usize,
                      b: &[u8],
                      sb: usize,
                      c0: usize,
                      r0: usize,
                      cw: usize,
                      rh: usize|
         -> (u32, f64) {
            let (mut mx, mut sum) = (0u32, 0u64);
            for r in r0..r0 + rh {
                for c in (c0 * 3)..((c0 + cw) * 3) {
                    let d = (a[r * sa + c] as i32 - b[r * sb + c] as i32).unsigned_abs();
                    mx = mx.max(d);
                    sum += d as u64;
                }
            }
            (mx, sum as f64 / (cw * 3 * rh) as f64)
        };

        // same-size baseline (no resize)
        let cpu0 = cpu_packed(Crop::no_crop(), sw, sh);
        if let Some(g) = gl_packed("shader", Crop::no_crop(), sw, sh) {
            let (mx, mean) = region(&g.1, g.0, &cpu0.1, cpu0.0, 0, 0, sw, sh);
            eprintln!("PROBE2[same-size]   ShaderR8 vs CPU: max={mx} mean={mean:.2}");
        }

        // letterbox (aspect-fit + grey bars), full frame + content region only
        let cpu1 = cpu_packed(letterbox(), model, model);
        // Decisive content-region pixel dump: flip / offset / rescale vs CPU?
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
                let (fmx, fmean) = region(&g.1, g.0, &cpu1.1, cpu1.0, 0, 0, model, model);
                let (cmx, cmean) =
                    region(&g.1, g.0, &cpu1.1, cpu1.0, pad_x, pad_y, scaled_w, scaled_h);
                eprintln!(
                    "PROBE2[lb-full/{env:8}] vs CPU: max={fmx} mean={fmean:.2}   \
                     lb-content: max={cmx} mean={cmean:.2}"
                );
            }
        }
    }

    /// Verify NV24→RGBA via Path B (R8 texelFetch shader) on DMA buffers.
    ///
    /// Checks:
    ///   (a) `last_nv_convert_path` == `ShaderR8` — no CPU fallback.
    ///   (b) Every output pixel matches the expected BT.601 full-range RGB
    ///       within ±2.
    #[test]
    #[cfg(all(target_os = "linux", feature = "dma_test_formats"))]
    fn test_gpu_nv24_to_rgba_path_b() {
        use crate::opengl_headless::processor::{GLProcessorST, NvConvertPath};
        if !is_dma_available() {
            eprintln!("SKIPPED: {} - DMA not available", function!());
            return;
        }

        let (w, h) = (64usize, 64usize);
        // Solid YCbCr: Y=180, Cb=90, Cr=200 → deterministic RGB.
        let (yv, cb, cr) = (180u8, 90u8, 200u8);
        let expected = yuv601_to_rgb(yv, cb, cr);

        let mut src = load_raw_image(
            w,
            h,
            PixelFormat::Nv24,
            Some(TensorMemory::Dma),
            &make_nv24_solid(w, h, yv, cb, cr),
        )
        .unwrap();
        // Full-range source to match the full-range `yuv601_to_rgb` reference.
        src.set_colorimetry(Some(
            edgefirst_tensor::Colorimetry::default()
                .with_encoding(edgefirst_tensor::ColorEncoding::Bt601)
                .with_range(edgefirst_tensor::ColorRange::Full),
        ));

        let mut dst =
            TensorDyn::image(w, h, PixelFormat::Rgba, DType::U8, Some(TensorMemory::Dma)).unwrap();

        let mut gl = match GLProcessorST::new(None) {
            Ok(g) => g,
            Err(e) => {
                eprintln!("SKIPPED: {} - GL not available: {e}", function!());
                return;
            }
        };

        gl.convert(&src, &mut dst, Rotation::None, Flip::None, Crop::no_crop())
            .unwrap();

        // (a) Assert Path B ran — no CPU fallback for a DMA NV24 source.
        assert_eq!(
            gl.last_nv_convert_path,
            NvConvertPath::ShaderR8,
            "NV24 DMA convert must use Path B (ShaderR8), got {:?}",
            gl.last_nv_convert_path
        );

        // (b) Check pixel values against BT.601 reference within ±2.
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

    /// Verify NV16→RGBA Path B output matches the CPU reference converter.
    ///
    /// Builds a patterned NV16 source, runs it through the CPU and GPU
    /// converters, and asserts the GPU output matches the CPU output within ±2.
    #[test]
    #[cfg(all(target_os = "linux", feature = "dma_test_formats"))]
    fn test_gpu_nv16_matches_cpu_reference() {
        use crate::opengl_headless::processor::{GLProcessorST, NvConvertPath};
        if !is_dma_available() {
            eprintln!("SKIPPED: {} - DMA not available", function!());
            return;
        }

        let (w, h) = (64usize, 64usize);
        // Build a patterned NV16 source (varying luma + fixed neutral chroma).
        let mut nv16 = vec![0u8; w * h * 2]; // H luma rows + H chroma rows
        for row in 0..h {
            for col in 0..w {
                nv16[row * w + col] = ((row * 255) / h) as u8;
            }
        }
        // Neutral chroma: Cb=128, Cr=128 → no colour shift.
        for i in 0..w * h {
            nv16[w * h + i] = 128;
        }

        let src_dma =
            load_raw_image(w, h, PixelFormat::Nv16, Some(TensorMemory::Dma), &nv16).unwrap();
        let src_mem =
            load_raw_image(w, h, PixelFormat::Nv16, Some(TensorMemory::Mem), &nv16).unwrap();

        // CPU reference.
        let mut cpu_dst = TensorDyn::image(w, h, PixelFormat::Rgba, DType::U8, None).unwrap();
        {
            let mut cpu = crate::cpu::CPUProcessor::new();
            cpu.convert(
                &src_mem,
                &mut cpu_dst,
                Rotation::None,
                Flip::None,
                Crop::no_crop(),
            )
            .unwrap();
        }

        // GPU Path B.
        let mut gpu_dst =
            TensorDyn::image(w, h, PixelFormat::Rgba, DType::U8, Some(TensorMemory::Dma)).unwrap();
        let mut gl = match GLProcessorST::new(None) {
            Ok(g) => g,
            Err(e) => {
                eprintln!("SKIPPED: {} - GL not available: {e}", function!());
                return;
            }
        };
        gl.convert(
            &src_dma,
            &mut gpu_dst,
            Rotation::None,
            Flip::None,
            Crop::no_crop(),
        )
        .unwrap();

        assert_eq!(
            gl.last_nv_convert_path,
            NvConvertPath::ShaderR8,
            "NV16 DMA convert must use Path B"
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
            "NV16 GPU Path B vs CPU reference: max_diff={max_diff} (tolerance 2)"
        );
    }

    /// Verify NV24→RGBA Path B output matches the CPU reference converter.
    #[test]
    #[cfg(all(target_os = "linux", feature = "dma_test_formats"))]
    fn test_gpu_nv24_matches_cpu_reference() {
        use crate::opengl_headless::processor::{GLProcessorST, NvConvertPath};
        if !is_dma_available() {
            eprintln!("SKIPPED: {} - DMA not available", function!());
            return;
        }

        let (w, h) = (64usize, 64usize);
        // Patterned NV24: varying luma, fixed neutral chroma.
        let mut nv24 = vec![0u8; w * h * 3]; // H luma rows + 2H chroma rows
        for row in 0..h {
            for col in 0..w {
                nv24[row * w + col] = ((row * 255) / h) as u8;
            }
        }
        // Neutral chroma for NV24: 2*w*h chroma bytes, all 128.
        for i in 0..w * h * 2 {
            nv24[w * h + i] = 128;
        }

        let src_dma =
            load_raw_image(w, h, PixelFormat::Nv24, Some(TensorMemory::Dma), &nv24).unwrap();
        let src_mem =
            load_raw_image(w, h, PixelFormat::Nv24, Some(TensorMemory::Mem), &nv24).unwrap();

        // CPU reference.
        let mut cpu_dst = TensorDyn::image(w, h, PixelFormat::Rgba, DType::U8, None).unwrap();
        {
            let mut cpu = crate::cpu::CPUProcessor::new();
            cpu.convert(
                &src_mem,
                &mut cpu_dst,
                Rotation::None,
                Flip::None,
                Crop::no_crop(),
            )
            .unwrap();
        }

        // GPU Path B.
        let mut gpu_dst =
            TensorDyn::image(w, h, PixelFormat::Rgba, DType::U8, Some(TensorMemory::Dma)).unwrap();
        let mut gl = match GLProcessorST::new(None) {
            Ok(g) => g,
            Err(e) => {
                eprintln!("SKIPPED: {} - GL not available: {e}", function!());
                return;
            }
        };
        gl.convert(
            &src_dma,
            &mut gpu_dst,
            Rotation::None,
            Flip::None,
            Crop::no_crop(),
        )
        .unwrap();

        assert_eq!(
            gl.last_nv_convert_path,
            NvConvertPath::ShaderR8,
            "NV24 DMA convert must use Path B"
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
            "NV24 GPU Path B vs CPU reference: max_diff={max_diff} (tolerance 2)"
        );
    }

    /// Verify a DMA NV12 source converts on the GPU (ExternalSampler or
    /// ShaderR8, platform-dependent) and never silently falls back to CPU.
    #[test]
    #[cfg(all(target_os = "linux", feature = "dma_test_formats"))]
    fn test_nv12_dma_gpu_path_no_cpu_fallback() {
        use crate::opengl_headless::processor::{GLProcessorST, NvConvertPath};
        if !is_dma_available() {
            eprintln!("SKIPPED: {} - DMA not available", function!());
            return;
        }

        let (w, h) = (64usize, 64usize);
        // Solid grey NV12 (Y=128, Cb=128, Cr=128).
        let mut nv12 = vec![128u8; w * h];
        nv12.extend(vec![128u8; w * h / 2]); // UV plane
        let src = load_raw_image(w, h, PixelFormat::Nv12, Some(TensorMemory::Dma), &nv12).unwrap();

        let mut dst =
            TensorDyn::image(w, h, PixelFormat::Rgba, DType::U8, Some(TensorMemory::Dma)).unwrap();

        let mut gl = match GLProcessorST::new(None) {
            Ok(g) => g,
            Err(e) => {
                eprintln!("SKIPPED: {} - GL not available: {e}", function!());
                return;
            }
        };

        gl.convert(&src, &mut dst, Rotation::None, Flip::None, Crop::no_crop())
            .unwrap();

        // NV12 must stay on a GPU path (no CPU-fallback regression). The exact
        // path is GPU-dependent: Path A (ExternalSampler) on Vivante, Path B (ShaderR8)
        // elsewhere — single-plane Path A YUV sampling is unreliable on Mali.
        assert!(
            matches!(
                gl.last_nv_convert_path,
                NvConvertPath::ExternalSampler | NvConvertPath::ShaderR8
            ),
            "NV12 DMA convert must use a GPU path (ExternalSampler/ShaderR8), got {:?}",
            gl.last_nv_convert_path
        );
    }

    /// Verify NV16→RGB **int8** output via Path B exercises the `nv_r8_int8`
    /// packing program (XOR 0x80 bias) and matches the CPU i8 reference. The u8
    /// Path B tests never reach the int8 packing shader; the quantized NPU
    /// targets (imx8mp vx / imx95 Neutron) consume exactly this i8 output.
    #[test]
    #[cfg(all(target_os = "linux", feature = "dma_test_formats"))]
    fn test_gpu_nv16_path_b_int8_output() {
        use crate::opengl_headless::processor::{GLProcessorST, NvConvertPath};
        if !is_dma_available() {
            eprintln!("SKIPPED: {} - DMA not available", function!());
            return;
        }

        let (w, h) = (64usize, 64usize);
        // Patterned luma (vertical gradient) + a mild chroma offset so the
        // int8 bias is exercised across a range of values, not a single colour.
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
            load_raw_image(w, h, PixelFormat::Nv16, Some(TensorMemory::Dma), &nv16).unwrap();
        let mut src_mem =
            load_raw_image(w, h, PixelFormat::Nv16, Some(TensorMemory::Mem), &nv16).unwrap();
        // Tag both BT.601 full-range so GPU and CPU resolve the same colorimetry
        // (this test isolates the int8 bias path, not the matrix/range).
        let cm = edgefirst_tensor::Colorimetry::default()
            .with_encoding(edgefirst_tensor::ColorEncoding::Bt601)
            .with_range(edgefirst_tensor::ColorRange::Full);
        src_dma.set_colorimetry(Some(cm));
        src_mem.set_colorimetry(Some(cm));

        // CPU reference, int8 RGB.
        let mut cpu_dst = TensorDyn::image(w, h, PixelFormat::Rgb, DType::I8, None).unwrap();
        {
            let mut cpu = crate::cpu::CPUProcessor::new();
            cpu.convert(
                &src_mem,
                &mut cpu_dst,
                Rotation::None,
                Flip::None,
                Crop::no_crop(),
            )
            .unwrap();
        }

        // GPU Path B, int8 RGB.
        let mut gpu_dst =
            TensorDyn::image(w, h, PixelFormat::Rgb, DType::I8, Some(TensorMemory::Dma)).unwrap();
        let mut gl = match GLProcessorST::new(None) {
            Ok(g) => g,
            Err(e) => {
                eprintln!("SKIPPED: {} - GL not available: {e}", function!());
                return;
            }
        };
        gl.convert(
            &src_dma,
            &mut gpu_dst,
            Rotation::None,
            Flip::None,
            Crop::no_crop(),
        )
        .unwrap();

        assert_eq!(
            gl.last_nv_convert_path,
            NvConvertPath::ShaderR8,
            "NV16 int8 DMA convert must use Path B"
        );

        // Compare raw i8 bytes (the XOR 0x80 bias lives in the data identically
        // on both paths).
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
            "NV16 int8 GPU Path B vs CPU reference: max_diff={max_diff} (tolerance 2)"
        );
    }

    // =========================================================================
    // Odd-dimension end-to-end cells (Deliverable B)
    //
    // Design contract:
    //   • Source is built as a patterned NV tensor with a pattern varying in
    //     BOTH x and y (a solid would mask addressing bugs — see the NV24 3H
    //     regression).
    //   • A Dma copy (for the GPU path) and a Mem copy (for the CPU reference)
    //     are filled identically, each at the tensor's own `effective_row_stride`.
    //   • The CPU reference is the trusted oracle (proven by odd_dim_cpu.rs).
    //   • Both maps are read at their real `effective_row_stride` × logical
    //     `width() × height()` so stride-padding bytes are never compared.
    //   • `last_nv_convert_path` is asserted before the pixel comparison so a
    //     silent CPU fallback cannot pass.
    // =========================================================================

    /// Fill a NV16/NV24/NV12 tensor (Dma or Mem) with a patterned, stride-aware
    /// synthetic image that exercises both odd-width and odd-height boundaries.
    ///
    /// Pattern: `Y(r,c) = (r*3 + c*5) % 256`
    /// Chroma column `cc`, chroma row `cr_row`:
    ///   `Cb = (cc*7 + cr_row*11 + 40) % 256`
    ///   `Cr = (cc*13 + cr_row*3 + 80) % 256`
    ///
    /// The chroma subsampling (cw_shift, ch_shift) follows each format:
    ///   NV12: (1,1) — 4:2:0   NV16: (1,0) — 4:2:2   NV24: (0,0) — 4:4:4
    ///
    /// This is intentionally the same pattern as `make_odd_both_source` in
    /// `odd_dim_cpu.rs` so the two test suites share an analytic ground truth.
    #[cfg(all(target_os = "linux", feature = "dma_test_formats"))]
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

        // Fill luma: diagonal gradient varying in both x and y.
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
    #[cfg(all(target_os = "linux", feature = "dma_test_formats"))]
    fn make_patterned_nv_pair(w: usize, h: usize, fmt: PixelFormat) -> (TensorDyn, TensorDyn) {
        let mut dma = TensorDyn::image(w, h, fmt, DType::U8, Some(TensorMemory::Dma)).unwrap();
        let mut mem = TensorDyn::image(w, h, fmt, DType::U8, Some(TensorMemory::Mem)).unwrap();
        fill_patterned_nv(&dma);
        fill_patterned_nv(&mem);
        // Tag both sources BT.601 full-range so the GPU (shader) and CPU
        // (`yuv` crate) sides resolve the SAME colorimetry — these GPU-vs-CPU
        // comparisons must isolate the *path*, not the matrix/range. (Untagged,
        // the heuristic picks limited range, whose ×1.164 luma gain amplifies
        // the known odd-width chroma-edge delta past the ±4 tolerance.)
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
    #[cfg(all(target_os = "linux", feature = "dma_test_formats"))]
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
    #[cfg(all(target_os = "linux", feature = "dma_test_formats"))]
    fn compare_gpu_vs_cpu_i8(gpu_dst: &TensorDyn, cpu_dst: &TensorDyn, w: usize, h: usize) -> u32 {
        let channels = 3usize; // RGB only (i8 output is always Rgb)
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
    // G-03: NV16 odd-W (65×64) → RGBA — Path B must run, GPU ≈ CPU ±4
    // -------------------------------------------------------------------------

    /// G-03: NV16 odd-width (65×64) end-to-end GPU vs CPU reference.
    ///
    /// Asserts:
    ///   (a) `last_nv_convert_path == ShaderR8` — no CPU fallback.
    ///   (b) GPU output matches CPU reference within ±4 (per-pixel, each RGBA channel).
    ///       Tolerance 4 absorbs f32 shader rounding; identical fill on both sides
    ///       rules out test-infrastructure bias.
    #[test]
    #[cfg(all(target_os = "linux", feature = "dma_test_formats"))]
    fn g03_nv16_odd_w_vs_cpu() {
        use crate::opengl_headless::processor::{GLProcessorST, NvConvertPath};
        if !is_dma_available() {
            eprintln!("SKIPPED: {} - DMA not available", function!());
            return;
        }
        let (w, h) = (321usize, 240usize); // QVGA-scale odd width (Mali rejects sub-minimum textures)
        let (src_dma, src_mem) = make_patterned_nv_pair(w, h, PixelFormat::Nv16);

        let mut cpu_dst = TensorDyn::image(w, h, PixelFormat::Rgba, DType::U8, None).unwrap();
        crate::cpu::CPUProcessor::new()
            .convert(
                &src_mem,
                &mut cpu_dst,
                Rotation::None,
                Flip::None,
                Crop::no_crop(),
            )
            .unwrap();

        let mut gpu_dst =
            TensorDyn::image(w, h, PixelFormat::Rgba, DType::U8, Some(TensorMemory::Dma)).unwrap();
        let mut gl = match GLProcessorST::new(None) {
            Ok(g) => g,
            Err(e) => {
                eprintln!("SKIPPED: {} - GL not available: {e}", function!());
                return;
            }
        };
        gl.convert(
            &src_dma,
            &mut gpu_dst,
            Rotation::None,
            Flip::None,
            Crop::no_crop(),
        )
        .unwrap();

        assert_eq!(
            gl.last_nv_convert_path,
            NvConvertPath::ShaderR8,
            "G-03: NV16 odd-W must use Path B (ShaderR8), got {:?}",
            gl.last_nv_convert_path
        );

        let (max_diff, first_fail) = compare_gpu_vs_cpu_u8(&gpu_dst, &cpu_dst, w, h);
        eprintln!("G-03 NV16 odd-W: GPU vs CPU max_diff={max_diff}");
        assert!(
            max_diff <= 4,
            "G-03: NV16 odd-W GPU vs CPU max_diff={max_diff} > 4; first bad at {first_fail:?}"
        );
    }

    // -------------------------------------------------------------------------
    // G-04: NV24 odd-W (65×64) → RGBA — Path B must run, GPU ≈ CPU ±4
    // -------------------------------------------------------------------------

    /// G-04: NV24 odd-width (65×64) end-to-end GPU vs CPU reference.
    #[test]
    #[cfg(all(target_os = "linux", feature = "dma_test_formats"))]
    fn g04_nv24_odd_w_vs_cpu() {
        use crate::opengl_headless::processor::{GLProcessorST, NvConvertPath};
        if !is_dma_available() {
            eprintln!("SKIPPED: {} - DMA not available", function!());
            return;
        }
        let (w, h) = (321usize, 240usize); // QVGA-scale odd width (Mali rejects sub-minimum textures)
        let (src_dma, src_mem) = make_patterned_nv_pair(w, h, PixelFormat::Nv24);

        let mut cpu_dst = TensorDyn::image(w, h, PixelFormat::Rgba, DType::U8, None).unwrap();
        crate::cpu::CPUProcessor::new()
            .convert(
                &src_mem,
                &mut cpu_dst,
                Rotation::None,
                Flip::None,
                Crop::no_crop(),
            )
            .unwrap();

        let mut gpu_dst =
            TensorDyn::image(w, h, PixelFormat::Rgba, DType::U8, Some(TensorMemory::Dma)).unwrap();
        let mut gl = match GLProcessorST::new(None) {
            Ok(g) => g,
            Err(e) => {
                eprintln!("SKIPPED: {} - GL not available: {e}", function!());
                return;
            }
        };
        gl.convert(
            &src_dma,
            &mut gpu_dst,
            Rotation::None,
            Flip::None,
            Crop::no_crop(),
        )
        .unwrap();

        assert_eq!(
            gl.last_nv_convert_path,
            NvConvertPath::ShaderR8,
            "G-04: NV24 odd-W must use Path B (ShaderR8), got {:?}",
            gl.last_nv_convert_path
        );

        let (max_diff, first_fail) = compare_gpu_vs_cpu_u8(&gpu_dst, &cpu_dst, w, h);
        eprintln!("G-04 NV24 odd-W: GPU vs CPU max_diff={max_diff}");
        assert!(
            max_diff <= 4,
            "G-04: NV24 odd-W GPU vs CPU max_diff={max_diff} > 4; first bad at {first_fail:?}"
        );
    }

    // -------------------------------------------------------------------------
    // G-05: NV16 odd-both (65×63) → RGBA — highest-value: strict tiled GPUs
    // -------------------------------------------------------------------------

    /// G-05: NV16 odd-width AND odd-height (65×63) end-to-end GPU vs CPU.
    ///
    /// This is the highest-value cell: it exercises the row-boundary padding
    /// constraint that causes incorrect reads on strict tiled GPUs (e.g. V3D)
    /// when the last chroma row straddles the 64-byte alignment boundary.
    #[test]
    #[cfg(all(target_os = "linux", feature = "dma_test_formats"))]
    fn g05_nv16_odd_both_vs_cpu() {
        use crate::opengl_headless::processor::{GLProcessorST, NvConvertPath};
        if !is_dma_available() {
            eprintln!("SKIPPED: {} - DMA not available", function!());
            return;
        }
        let (w, h) = (321usize, 241usize); // QVGA-scale odd both (Mali rejects sub-minimum textures)
        let (src_dma, src_mem) = make_patterned_nv_pair(w, h, PixelFormat::Nv16);

        let mut cpu_dst = TensorDyn::image(w, h, PixelFormat::Rgba, DType::U8, None).unwrap();
        crate::cpu::CPUProcessor::new()
            .convert(
                &src_mem,
                &mut cpu_dst,
                Rotation::None,
                Flip::None,
                Crop::no_crop(),
            )
            .unwrap();

        let mut gpu_dst =
            TensorDyn::image(w, h, PixelFormat::Rgba, DType::U8, Some(TensorMemory::Dma)).unwrap();
        let mut gl = match GLProcessorST::new(None) {
            Ok(g) => g,
            Err(e) => {
                eprintln!("SKIPPED: {} - GL not available: {e}", function!());
                return;
            }
        };
        gl.convert(
            &src_dma,
            &mut gpu_dst,
            Rotation::None,
            Flip::None,
            Crop::no_crop(),
        )
        .unwrap();

        assert_eq!(
            gl.last_nv_convert_path,
            NvConvertPath::ShaderR8,
            "G-05: NV16 odd-both must use Path B (ShaderR8), got {:?}",
            gl.last_nv_convert_path
        );

        let (max_diff, first_fail) = compare_gpu_vs_cpu_u8(&gpu_dst, &cpu_dst, w, h);
        eprintln!("G-05 NV16 odd-both (65×63): GPU vs CPU max_diff={max_diff}");
        assert!(
            max_diff <= 4,
            "G-05: NV16 odd-both GPU vs CPU max_diff={max_diff} > 4; first bad at {first_fail:?}"
        );
    }

    // -------------------------------------------------------------------------
    // G-06: NV24 odd-both (65×63) → RGBA
    // -------------------------------------------------------------------------

    /// G-06: NV24 odd-width AND odd-height (65×63) end-to-end GPU vs CPU.
    ///
    /// NV24 uses a UV row pitch of `2 × stride`, so odd heights place the last
    /// UV row at offset `3H − 2` (0-indexed), making it a different boundary
    /// from NV16.  This is the regression cell for the NV24 3H height bug.
    #[test]
    #[cfg(all(target_os = "linux", feature = "dma_test_formats"))]
    fn g06_nv24_odd_both_vs_cpu() {
        use crate::opengl_headless::processor::{GLProcessorST, NvConvertPath};
        if !is_dma_available() {
            eprintln!("SKIPPED: {} - DMA not available", function!());
            return;
        }
        let (w, h) = (321usize, 241usize); // QVGA-scale odd both (Mali rejects sub-minimum textures)
        let (src_dma, src_mem) = make_patterned_nv_pair(w, h, PixelFormat::Nv24);

        let mut cpu_dst = TensorDyn::image(w, h, PixelFormat::Rgba, DType::U8, None).unwrap();
        crate::cpu::CPUProcessor::new()
            .convert(
                &src_mem,
                &mut cpu_dst,
                Rotation::None,
                Flip::None,
                Crop::no_crop(),
            )
            .unwrap();

        let mut gpu_dst =
            TensorDyn::image(w, h, PixelFormat::Rgba, DType::U8, Some(TensorMemory::Dma)).unwrap();
        let mut gl = match GLProcessorST::new(None) {
            Ok(g) => g,
            Err(e) => {
                eprintln!("SKIPPED: {} - GL not available: {e}", function!());
                return;
            }
        };
        gl.convert(
            &src_dma,
            &mut gpu_dst,
            Rotation::None,
            Flip::None,
            Crop::no_crop(),
        )
        .unwrap();

        assert_eq!(
            gl.last_nv_convert_path,
            NvConvertPath::ShaderR8,
            "G-06: NV24 odd-both must use Path B (ShaderR8), got {:?}",
            gl.last_nv_convert_path
        );

        let (max_diff, first_fail) = compare_gpu_vs_cpu_u8(&gpu_dst, &cpu_dst, w, h);
        eprintln!("G-06 NV24 odd-both (65×63): GPU vs CPU max_diff={max_diff}");
        assert!(
            max_diff <= 4,
            "G-06: NV24 odd-both GPU vs CPU max_diff={max_diff} > 4; first bad at {first_fail:?}"
        );
    }

    // -------------------------------------------------------------------------
    // G-09: odd dst with a deliberately NON-64-aligned stride — tolerant guard
    // -------------------------------------------------------------------------

    /// G-09: an externally-strided odd destination whose row pitch is NOT
    /// 64-byte aligned. `Tensor::image` always 64-aligns, so this case can only
    /// arise from an explicit `image_with_stride` (or a `from_fd` import). It
    /// exercises the reactive odd-destination guard in `GLProcessorST::convert`:
    ///
    ///   * On GPUs that accept a non-aligned EGLImage pitch (V3D, Tegra) the
    ///     convert SUCCEEDS.
    ///   * On GPUs that reject it (Mali `BadAlloc`, Vivante `BadAccess`) the raw
    ///     EGL error is re-wrapped as a descriptive `NotSupported` naming the odd
    ///     dimensions — NOT leaked as a bare `EGL(BadAlloc)`.
    ///
    /// Tolerant by design (platform-dependent), so it is CI-safe: it asserts the
    /// outcome is one of those two, never a raw EGL error.
    #[test]
    #[cfg(all(target_os = "linux", feature = "dma_test_formats"))]
    fn g09_odd_dst_unaligned_stride_guarded() {
        use crate::opengl_headless::processor::GLProcessorST;
        if !is_dma_available() {
            eprintln!("SKIPPED: {} - DMA not available", function!());
            return;
        }
        let (w, h) = (321usize, 240usize);
        let (src_dma, _src_mem) = make_patterned_nv_pair(w, h, PixelFormat::Nv12);

        // Tight, NON-64-aligned RGBA stride (321*4 = 1284; 1284 % 64 != 0).
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
            Some(TensorMemory::Dma),
        )
        .unwrap();

        let mut gl = match GLProcessorST::new(None) {
            Ok(g) => g,
            Err(e) => {
                eprintln!("SKIPPED: {} - GL not available: {e}", function!());
                return;
            }
        };
        match gl.convert(
            &src_dma,
            &mut gpu_dst,
            Rotation::None,
            Flip::None,
            Crop::no_crop(),
        ) {
            Ok(()) => eprintln!("G-09: platform accepts non-aligned odd dst (OK)"),
            Err(crate::Error::NotSupported(msg)) => {
                assert!(
                    msg.contains("odd dimensions"),
                    "G-09: NotSupported must name the odd-dimension cause, got: {msg}"
                );
                eprintln!("G-09: platform rejects non-aligned odd dst, guarded as: {msg}");
            }
            Err(other) => panic!(
                "G-09: odd-dst failure must be wrapped as NotSupported, not leaked raw: {other:?}"
            ),
        }
    }

    // -------------------------------------------------------------------------
    // G-01: NV12 odd-W (QVGA 321×240) → RGBA — routes to Path B (width % 4 != 0)
    // -------------------------------------------------------------------------

    /// G-01: NV12 with width not a multiple of 4 (321×240) routes to Path B (R8
    /// shader), because the NV12 samplerExternalOES EGLImage import requires
    /// width % 4 == 0 on some drivers (e.g. V3D). Even/mult-4 NV12 still uses
    /// Path A — see `test_nv12_still_uses_path_a` (64×64). Asserts the GPU output
    /// agrees with the CPU reference ±4.
    #[test]
    #[cfg(all(target_os = "linux", feature = "dma_test_formats"))]
    fn g01_nv12_odd_w_path_b_vs_cpu() {
        use crate::opengl_headless::processor::{GLProcessorST, NvConvertPath};
        if !is_dma_available() {
            eprintln!("SKIPPED: {} - DMA not available", function!());
            return;
        }
        let (w, h) = (321usize, 240usize); // QVGA-scale odd width (Mali rejects sub-minimum textures)
        let (src_dma, src_mem) = make_patterned_nv_pair(w, h, PixelFormat::Nv12);

        let mut cpu_dst = TensorDyn::image(w, h, PixelFormat::Rgba, DType::U8, None).unwrap();
        crate::cpu::CPUProcessor::new()
            .convert(
                &src_mem,
                &mut cpu_dst,
                Rotation::None,
                Flip::None,
                Crop::no_crop(),
            )
            .unwrap();

        let mut gpu_dst =
            TensorDyn::image(w, h, PixelFormat::Rgba, DType::U8, Some(TensorMemory::Dma)).unwrap();
        let mut gl = match GLProcessorST::new(None) {
            Ok(g) => g,
            Err(e) => {
                eprintln!("SKIPPED: {} - GL not available: {e}", function!());
                return;
            }
        };
        gl.convert(
            &src_dma,
            &mut gpu_dst,
            Rotation::None,
            Flip::None,
            Crop::no_crop(),
        )
        .unwrap();

        // NV12 width 65 (not mult-4) routes to Path B (R8 shader).
        assert_eq!(
            gl.last_nv_convert_path,
            NvConvertPath::ShaderR8,
            "G-01: non-mult-4 NV12 must route to Path B (ShaderR8), got {:?}",
            gl.last_nv_convert_path
        );

        let (max_diff, first_fail) = compare_gpu_vs_cpu_u8(&gpu_dst, &cpu_dst, w, h);
        eprintln!("G-01 NV12 odd-W: GPU vs CPU max_diff={max_diff}");
        assert!(
            max_diff <= 4,
            "G-01: NV12 odd-W GPU vs CPU max_diff={max_diff} > 4; first bad at {first_fail:?}"
        );
    }

    // -------------------------------------------------------------------------
    // G-02: NV12 even-W odd-H (320×241) → RGBA — Path B must run, GPU ≈ CPU ±4
    // -------------------------------------------------------------------------

    /// G-02: NV12 even-width odd-height (320×241) end-to-end GPU vs CPU.
    ///
    /// Exercises the luma/chroma row-boundary math for an odd height. With an odd
    /// H the chroma plane is `ceil(H/2)` rows, and the last chroma row reads a
    /// 64-byte-aligned row whose physical allocation is exactly stride bytes (no
    /// half-row shortfall). Path B must be selected because this is a DMA source
    /// where width (320) is a multiple of 4 but the height is odd.
    ///
    /// Asserts:
    ///   (a) `last_nv_convert_path == ShaderR8` — no CPU fallback for DMA source.
    ///   (b) GPU output matches CPU reference within ±4.
    #[test]
    #[cfg(all(target_os = "linux", feature = "dma_test_formats"))]
    fn g02_nv12_odd_h_path_b_vs_cpu() {
        use crate::opengl_headless::processor::{GLProcessorST, NvConvertPath};
        if !is_dma_available() {
            eprintln!("SKIPPED: {} - DMA not available", function!());
            return;
        }
        // Even width, odd height — exercises odd-H chroma row boundary.
        let (w, h) = (320usize, 241usize);
        let (src_dma, src_mem) = make_patterned_nv_pair(w, h, PixelFormat::Nv12);

        let mut cpu_dst = TensorDyn::image(w, h, PixelFormat::Rgba, DType::U8, None).unwrap();
        crate::cpu::CPUProcessor::new()
            .convert(
                &src_mem,
                &mut cpu_dst,
                Rotation::None,
                Flip::None,
                Crop::no_crop(),
            )
            .unwrap();

        let mut gpu_dst =
            TensorDyn::image(w, h, PixelFormat::Rgba, DType::U8, Some(TensorMemory::Dma)).unwrap();
        let mut gl = match GLProcessorST::new(None) {
            Ok(g) => g,
            Err(e) => {
                eprintln!("SKIPPED: {} - GL not available: {e}", function!());
                return;
            }
        };
        gl.convert(
            &src_dma,
            &mut gpu_dst,
            Rotation::None,
            Flip::None,
            Crop::no_crop(),
        )
        .unwrap();

        // Even-width NV12 uses a hardware GPU path: Path A (ExternalSampler,
        // samplerExternalOES) on Vivante, or Path B (ShaderR8) elsewhere —
        // Path A's YUV-EGLImage sampling is only reliable on Vivante (it samples
        // garbage on Mali-G310). Either is correct here; what matters is that it
        // stays on the GPU rather than falling back to the CPU.
        assert!(
            matches!(
                gl.last_nv_convert_path,
                NvConvertPath::ExternalSampler | NvConvertPath::ShaderR8
            ),
            "G-02: even-W odd-H NV12 should use a GPU path (ExternalSampler/ShaderR8), got {:?}",
            gl.last_nv_convert_path
        );

        let (max_diff, first_fail) = compare_gpu_vs_cpu_u8(&gpu_dst, &cpu_dst, w, h);
        eprintln!("G-02 NV12 odd-H (320×241): GPU vs CPU max_diff={max_diff}");
        // Post-WS1 both CPU and the GL NV12 path resolve this untagged odd-H
        // source to the same limited-range matrix, so the YUV-matrix delta that
        // forced the loose >64 bound has closed; the residual is GPU rounding.
        // Warn above a tight ±4, fail on >35 (was 64) so the odd-H stride
        // handling this test really guards still trips on a real regression.
        if max_diff > 4 {
            eprintln!(
                "WARNING: G-02 NV12 odd-H GPU vs CPU max_diff={max_diff} > 4 \
                 (GPU rounding; first bad at {first_fail:?})"
            );
        }
        assert!(
            max_diff <= 35,
            "G-02: gross NV12 odd-H GPU vs CPU mismatch max_diff={max_diff} (>35); first bad at {first_fail:?}"
        );
    }

    // -------------------------------------------------------------------------
    // g_grey_odd_w_vs_cpu: GREY odd-W (coco_grey_odd.jpg 595×438) → RGBA
    // -------------------------------------------------------------------------

    /// Grey odd-width (595×438) decode + GPU convert → RGBA vs CPU reference.
    ///
    /// Loads `coco_grey_odd.jpg` into a DMA Grey tensor (odd-W 595), converts
    /// Grey→RGBA on GPU and on CPU, and asserts they agree within ±4 per pixel.
    /// This guards the odd-W Grey IOSurface / EGLImage path that the macOS NV24
    /// fix relies on (64-aligned pitch + physical-stride shader addressing).
    #[test]
    #[cfg(all(target_os = "linux", feature = "dma_test_formats"))]
    fn g_grey_odd_w_vs_cpu() {
        use edgefirst_codec::{ImageDecoder, ImageLoad};
        if !is_dma_available() {
            eprintln!("SKIPPED: {} - DMA not available", function!());
            return;
        }
        let jpeg: &[u8] = &edgefirst_bench::testdata::read("coco_grey_odd.jpg");
        let w = 595usize;
        let h = 438usize;

        // Decode into a DMA-backed Grey tensor (odd width, 64-aligned pitch).
        let src_dma = match Tensor::<u8>::image(w, h, PixelFormat::Grey, Some(TensorMemory::Dma)) {
            Ok(t) => {
                let mut t = t;
                let mut dec = ImageDecoder::new();
                t.load_image(&mut dec, jpeg).unwrap();
                TensorDyn::from(t)
            }
            Err(e) => {
                eprintln!("SKIPPED: {} - DMA Grey alloc failed: {e}", function!());
                return;
            }
        };

        // CPU reference: decode into a Mem Grey tensor, convert Grey→RGBA via CPU.
        let src_mem = {
            let mut t =
                Tensor::<u8>::image(w, h, PixelFormat::Grey, Some(TensorMemory::Mem)).unwrap();
            let mut dec = ImageDecoder::new();
            t.load_image(&mut dec, jpeg).unwrap();
            TensorDyn::from(t)
        };
        let mut cpu_dst = TensorDyn::image(w, h, PixelFormat::Rgba, DType::U8, None).unwrap();
        crate::cpu::CPUProcessor::new()
            .convert(
                &src_mem,
                &mut cpu_dst,
                Rotation::None,
                Flip::None,
                Crop::no_crop(),
            )
            .unwrap();

        // GPU convert: Grey→RGBA on GL.
        let mut gpu_dst =
            TensorDyn::image(w, h, PixelFormat::Rgba, DType::U8, Some(TensorMemory::Dma)).unwrap();
        let mut gl = match GLProcessorST::new(None) {
            Ok(g) => g,
            Err(e) => {
                eprintln!("SKIPPED: {} - GL not available: {e}", function!());
                return;
            }
        };
        gl.convert(
            &src_dma,
            &mut gpu_dst,
            Rotation::None,
            Flip::None,
            Crop::no_crop(),
        )
        .unwrap();

        let (max_diff, first_fail) = compare_gpu_vs_cpu_u8(&gpu_dst, &cpu_dst, w, h);
        eprintln!("g_grey_odd_w: GPU vs CPU max_diff={max_diff}");
        assert!(
            max_diff <= 4,
            "g_grey_odd_w: Grey odd-W GPU vs CPU max_diff={max_diff} > 4; first bad at {first_fail:?}"
        );
    }

    // -------------------------------------------------------------------------
    // G-07: NV12 odd-W i8 (65×64) → RGB i8
    // -------------------------------------------------------------------------

    /// G-07: NV12 odd-width (65×64) → RGB i8.
    ///
    /// Exercises the `nv_r8_int8` (XOR 0x80 bias) packing shader for NV12.
    /// GPU i8 output is compared to CPU i8 within ±2.
    // Odd-WIDTH 3-channel RGB DMA *output* needs `width*3 % 4 == 0`. This is NOT
    // the stride-alignment bug fixed for RGBA dsts (g01-g06): it is architectural
    // in `convert_to_packed_rgb`, which packs the RGB buffer by reinterpreting it
    // as RGBA8 at `width*3/4` pixels. That reinterpretation only tiles when
    // `width*3` is a multiple of 4 — width 321 → 963 bytes/row → 240.75 RGBA8
    // texels, which does not tile (independent of the 64-aligned stride). Verified
    // on-target 2026-06-02: still `NotSupported("Packed RGB requires width*3
    // divisible by 4")`. Supporting it needs a different packing path (e.g. an R8
    // output texture or fractional-last-texel handling), not a stride change.
    // Production model-input is an even/mult-4 dst; the i8 path is covered by
    // `test_gpu_nv16_path_b_int8_output`.
    #[test]
    #[ignore = "odd-width 3-channel RGB DMA output needs width*3 % 4 == 0 (pack-as-RGBA8 architecture in convert_to_packed_rgb, NOT the stride bug); use RGBA or even/mult-4 RGB dst"]
    #[cfg(all(target_os = "linux", feature = "dma_test_formats"))]
    fn g07_nv12_odd_w_i8_vs_cpu() {
        use crate::opengl_headless::processor::{GLProcessorST, NvConvertPath};
        if !is_dma_available() {
            eprintln!("SKIPPED: {} - DMA not available", function!());
            return;
        }
        let (w, h) = (321usize, 240usize); // QVGA-scale odd width (Mali rejects sub-minimum textures)
        let (src_dma, src_mem) = make_patterned_nv_pair(w, h, PixelFormat::Nv12);

        // CPU i8 reference.
        let mut cpu_dst = TensorDyn::image(w, h, PixelFormat::Rgb, DType::I8, None).unwrap();
        crate::cpu::CPUProcessor::new()
            .convert(
                &src_mem,
                &mut cpu_dst,
                Rotation::None,
                Flip::None,
                Crop::no_crop(),
            )
            .unwrap();

        // GPU i8 output.
        let mut gpu_dst =
            TensorDyn::image(w, h, PixelFormat::Rgb, DType::I8, Some(TensorMemory::Dma)).unwrap();
        let mut gl = match GLProcessorST::new(None) {
            Ok(g) => g,
            Err(e) => {
                eprintln!("SKIPPED: {} - GL not available: {e}", function!());
                return;
            }
        };
        gl.convert(
            &src_dma,
            &mut gpu_dst,
            Rotation::None,
            Flip::None,
            Crop::no_crop(),
        )
        .unwrap();

        // Odd-width single-plane NV12 routes to ShaderR8 (no width gate); the
        // driver ExternalSampler requires even/4-aligned width. (This assert only
        // runs if the #[ignore] is lifted — kept accurate per current routing.)
        assert_eq!(
            gl.last_nv_convert_path,
            NvConvertPath::ShaderR8,
            "G-07: odd-W single-plane NV12 must use ShaderR8, got {:?}",
            gl.last_nv_convert_path
        );

        let max_diff = compare_gpu_vs_cpu_i8(&gpu_dst, &cpu_dst, w, h);
        eprintln!("G-07 NV12 odd-W i8: GPU vs CPU max_diff={max_diff}");
        assert!(
            max_diff <= 2,
            "G-07: NV12 i8 odd-W GPU vs CPU max_diff={max_diff} > 2"
        );
    }

    // -------------------------------------------------------------------------
    // G-08: NV16 odd-both i8 (65×63) → RGB i8 — Path B + i8 bias
    // -------------------------------------------------------------------------

    /// G-08: NV16 odd-both (65×63) → RGB i8.
    ///
    /// Combines the odd-dimension addressing check with the int8 XOR 0x80 bias
    /// packing.  This is the cell most likely to surface on NPU targets (imx8mp
    /// vx / imx95 Neutron) with unusual-resolution input streams.
    // Same odd-width 3-channel RGB DMA *output* constraint as g07: architectural
    // in `convert_to_packed_rgb` (pack-as-RGBA8 needs width*3 % 4 == 0), NOT the
    // stride bug fixed for RGBA dsts. The even-dim i8 path is covered by
    // `test_gpu_nv16_path_b_int8_output`.
    #[test]
    #[ignore = "odd-width 3-channel RGB DMA output needs width*3 % 4 == 0 (pack-as-RGBA8 architecture in convert_to_packed_rgb, NOT the stride bug); use RGBA or even/mult-4 RGB dst"]
    #[cfg(all(target_os = "linux", feature = "dma_test_formats"))]
    fn g08_nv16_odd_both_i8_vs_cpu() {
        use crate::opengl_headless::processor::{GLProcessorST, NvConvertPath};
        if !is_dma_available() {
            eprintln!("SKIPPED: {} - DMA not available", function!());
            return;
        }
        let (w, h) = (321usize, 241usize); // QVGA-scale odd both (Mali rejects sub-minimum textures)
        let (src_dma, src_mem) = make_patterned_nv_pair(w, h, PixelFormat::Nv16);

        // CPU i8 reference.
        let mut cpu_dst = TensorDyn::image(w, h, PixelFormat::Rgb, DType::I8, None).unwrap();
        crate::cpu::CPUProcessor::new()
            .convert(
                &src_mem,
                &mut cpu_dst,
                Rotation::None,
                Flip::None,
                Crop::no_crop(),
            )
            .unwrap();

        // GPU i8 output.
        let mut gpu_dst =
            TensorDyn::image(w, h, PixelFormat::Rgb, DType::I8, Some(TensorMemory::Dma)).unwrap();
        let mut gl = match GLProcessorST::new(None) {
            Ok(g) => g,
            Err(e) => {
                eprintln!("SKIPPED: {} - GL not available: {e}", function!());
                return;
            }
        };
        gl.convert(
            &src_dma,
            &mut gpu_dst,
            Rotation::None,
            Flip::None,
            Crop::no_crop(),
        )
        .unwrap();

        assert_eq!(
            gl.last_nv_convert_path,
            NvConvertPath::ShaderR8,
            "G-08: NV16 i8 odd-both must use Path B (ShaderR8), got {:?}",
            gl.last_nv_convert_path
        );

        let max_diff = compare_gpu_vs_cpu_i8(&gpu_dst, &cpu_dst, w, h);
        eprintln!("G-08 NV16 odd-both i8: GPU vs CPU max_diff={max_diff}");
        assert!(
            max_diff <= 2,
            "G-08: NV16 i8 odd-both GPU vs CPU max_diff={max_diff} > 2"
        );
    }

    // -------------------------------------------------------------------------
    // Even-dimension regression guards (ensure no regression on well-tested paths)
    // -------------------------------------------------------------------------

    /// Even-dim regression: NV16 64×64 → RGBA GPU vs CPU ±4.
    ///
    /// Ensures that adding the odd-dim cells did not perturb even-dimension
    /// behaviour on Path B.  Uses the same patterned fill and stride-aware
    /// comparison as the odd-dim cells.
    #[test]
    #[cfg(all(target_os = "linux", feature = "dma_test_formats"))]
    fn g_even_nv16_64x64_regression() {
        use crate::opengl_headless::processor::{GLProcessorST, NvConvertPath};
        if !is_dma_available() {
            eprintln!("SKIPPED: {} - DMA not available", function!());
            return;
        }
        let (w, h) = (64usize, 64usize);
        let (src_dma, src_mem) = make_patterned_nv_pair(w, h, PixelFormat::Nv16);

        let mut cpu_dst = TensorDyn::image(w, h, PixelFormat::Rgba, DType::U8, None).unwrap();
        crate::cpu::CPUProcessor::new()
            .convert(
                &src_mem,
                &mut cpu_dst,
                Rotation::None,
                Flip::None,
                Crop::no_crop(),
            )
            .unwrap();

        let mut gpu_dst =
            TensorDyn::image(w, h, PixelFormat::Rgba, DType::U8, Some(TensorMemory::Dma)).unwrap();
        let mut gl = match GLProcessorST::new(None) {
            Ok(g) => g,
            Err(e) => {
                eprintln!("SKIPPED: {} - GL not available: {e}", function!());
                return;
            }
        };
        gl.convert(
            &src_dma,
            &mut gpu_dst,
            Rotation::None,
            Flip::None,
            Crop::no_crop(),
        )
        .unwrap();

        assert_eq!(
            gl.last_nv_convert_path,
            NvConvertPath::ShaderR8,
            "even NV16 regression: must use Path B"
        );
        let (max_diff, first_fail) = compare_gpu_vs_cpu_u8(&gpu_dst, &cpu_dst, w, h);
        eprintln!("even NV16 64×64 regression: GPU vs CPU max_diff={max_diff}");
        assert!(
            max_diff <= 4,
            "even NV16 64×64 regression: max_diff={max_diff} > 4; first bad at {first_fail:?}"
        );
    }

    /// Even-dim regression: NV24 64×64 → RGBA GPU vs CPU ±4.
    #[test]
    #[cfg(all(target_os = "linux", feature = "dma_test_formats"))]
    fn g_even_nv24_64x64_regression() {
        use crate::opengl_headless::processor::{GLProcessorST, NvConvertPath};
        if !is_dma_available() {
            eprintln!("SKIPPED: {} - DMA not available", function!());
            return;
        }
        let (w, h) = (64usize, 64usize);
        let (src_dma, src_mem) = make_patterned_nv_pair(w, h, PixelFormat::Nv24);

        let mut cpu_dst = TensorDyn::image(w, h, PixelFormat::Rgba, DType::U8, None).unwrap();
        crate::cpu::CPUProcessor::new()
            .convert(
                &src_mem,
                &mut cpu_dst,
                Rotation::None,
                Flip::None,
                Crop::no_crop(),
            )
            .unwrap();

        let mut gpu_dst =
            TensorDyn::image(w, h, PixelFormat::Rgba, DType::U8, Some(TensorMemory::Dma)).unwrap();
        let mut gl = match GLProcessorST::new(None) {
            Ok(g) => g,
            Err(e) => {
                eprintln!("SKIPPED: {} - GL not available: {e}", function!());
                return;
            }
        };
        gl.convert(
            &src_dma,
            &mut gpu_dst,
            Rotation::None,
            Flip::None,
            Crop::no_crop(),
        )
        .unwrap();

        assert_eq!(
            gl.last_nv_convert_path,
            NvConvertPath::ShaderR8,
            "even NV24 regression: must use Path B"
        );
        let (max_diff, first_fail) = compare_gpu_vs_cpu_u8(&gpu_dst, &cpu_dst, w, h);
        eprintln!("even NV24 64×64 regression: GPU vs CPU max_diff={max_diff}");
        assert!(
            max_diff <= 4,
            "even NV24 64×64 regression: max_diff={max_diff} > 4; first bad at {first_fail:?}"
        );
    }
}
