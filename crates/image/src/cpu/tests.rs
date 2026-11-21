// SPDX-FileCopyrightText: Copyright 2025 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
#[allow(deprecated)]
mod cpu_tests {

    use crate::{
        CPUProcessor, Crop, Error, Flip, ImageProcessorTrait, Rect, Region, Result, Rotation,
    };
    use edgefirst_tensor::DetectBox;

    /// The instance renderer rejects a `C > 1` mask in a list whose first mask is `C == 1`.
    #[test]
    fn instance_renderer_reads_channel_zero_of_a_multi_channel_mask() {
        use edgefirst_tensor::Segmentation;

        const W: usize = 8;
        const H: usize = 8;

        let chan0: Vec<u8> = (0..H * W)
            .map(|i| if i % 3 == 0 { 255 } else { 0 })
            .collect();
        // C == 3 with channel 0 == chan0 and the others inverted.
        let mut wide = vec![0u8; H * W * 3];
        for (px, &v) in chan0.iter().enumerate() {
            wide[px * 3] = v;
            wide[px * 3 + 1] = 255 - v;
            wide[px * 3 + 2] = 255 - v;
        }
        // Slot 0 is C == 1.
        let first: Vec<u8> = (0..H * W)
            .map(|i| if i % 2 == 0 { 255 } else { 0 })
            .collect();

        let build = |bytes: &[u8], shape: &[usize]| {
            let t = Tensor::<u8>::new(shape, Some(TensorMemory::Mem), None).expect("alloc mask");
            t.map_mut().unwrap().as_mut_slice().copy_from_slice(bytes);
            TensorDyn::from(t)
        };
        let seg = |m: TensorDyn| Segmentation {
            segmentation: m,
            xmin: 0.0,
            ymin: 0.0,
            xmax: 1.0,
            ymax: 1.0,
        };

        let render = |second: TensorDyn| -> crate::Result<Vec<u8>> {
            let mut dst = TensorDyn::image(
                16,
                16,
                PixelFormat::Rgba,
                DType::U8,
                None,
                edgefirst_tensor::CpuAccess::ReadWrite,
            )
            .expect("dst");
            {
                let t = dst.as_u8_mut().unwrap();
                t.map_mut().unwrap().as_mut_slice().fill(0);
            }
            let det = [
                DetectBox {
                    bbox: [0.0, 0.0, 0.5, 1.0].into(),
                    score: 0.9,
                    label: 0,
                },
                DetectBox {
                    bbox: [0.5, 0.0, 1.0, 1.0].into(),
                    score: 0.9,
                    label: 1,
                },
            ];
            let segs = [seg(build(&first, &[H, W, 1])), seg(second)];
            let mut cpu = CPUProcessor::new();
            cpu.draw_decoded_masks(&mut dst, &det, &segs, Default::default())?;
            let t = dst.as_u8().unwrap();
            let m = t.map_read().unwrap();
            Ok(m.as_slice().to_vec())
        };

        let want = render(build(&chan0, &[H, W, 1])).expect("C=1 list must render");
        assert!(want.iter().any(|&v| v != 0), "oracle drew nothing");

        let err = render(build(&wide, &[H, W, 3])).expect_err("C=3 must be rejected");
        let msg = format!("{err:?}");
        assert!(
            msg.contains("instance mask must be"),
            "expected a shape error naming the constraint, got {msg}"
        );
    }

    /// The mask bytes of a `Segmentation`, as a flat `Vec<u8>`.
    fn seg_bytes(s: &edgefirst_tensor::Segmentation) -> Vec<u8> {
        let t = s.segmentation.as_u8().expect("segmentation must be U8");
        let m = t.map_read().expect("map segmentation");
        m.as_slice().to_vec()
    }

    use edgefirst_tensor::{
        DType, PixelFormat, Tensor, TensorDyn, TensorMapTrait, TensorMemory, TensorTrait,
    };

    /// A convert into a view needs a `ReadWrite` destination; a letterbox into a whole destination takes `Write`.
    #[test]
    fn partial_destination_write_needs_readwrite() {
        use edgefirst_tensor::{CpuAccess, Error as TensorError};
        let image = |w, h, access| {
            TensorDyn::image(
                w,
                h,
                PixelFormat::Rgb,
                DType::U8,
                Some(TensorMemory::Mem),
                access,
            )
            .unwrap()
        };
        let src = image(32, 16, CpuAccess::ReadWrite);
        let mut cpu = CPUProcessor::new();
        let parent = image(64, 64, CpuAccess::Write);
        let mut view = parent.view(Region::new(8, 8, 32, 32)).unwrap();
        match cpu.convert(&src, &mut view, None) {
            Err(Error::Tensor(TensorError::CpuAccessDenied { requested, .. })) => {
                assert_eq!(requested, CpuAccess::ReadWrite)
            }
            other => panic!("expected CpuAccessDenied, got {other:?}"),
        }
        // The view rebuilt from its descriptor is refused alike.
        let pin = view.pin_host(CpuAccess::Write).unwrap();
        let mut rebuilt =
            TensorDyn::import_descriptor(&view.descriptor_pinned(Some(&pin))).unwrap();
        assert!(rebuilt.view_origin().is_none() && rebuilt.is_view());
        match cpu.convert(&src, &mut rebuilt, None) {
            Err(Error::Tensor(TensorError::CpuAccessDenied { requested, .. })) => {
                assert_eq!(requested, CpuAccess::ReadWrite)
            }
            other => panic!("expected CpuAccessDenied, got {other:?}"),
        }
        let rw_parent = image(64, 64, CpuAccess::ReadWrite);
        let mut rw_view = rw_parent.view(Region::new(8, 8, 32, 32)).unwrap();
        cpu.convert(&src, &mut rw_view, None).unwrap();
        let mut whole = image(64, 64, CpuAccess::Write);
        let letterbox = crate::ConvertOptions::default().crop(Crop::letterbox([0, 0, 0, 255]));
        cpu.convert(&src, &mut whole, Some(letterbox)).unwrap();
    }

    /// A letterbox into a padded write-only destination succeeds.
    #[test]
    fn padded_write_only_destination_letterboxes() {
        use edgefirst_tensor::CpuAccess;
        let src = TensorDyn::image(
            32,
            16,
            PixelFormat::Rgb,
            DType::U8,
            Some(TensorMemory::Mem),
            CpuAccess::ReadWrite,
        )
        .unwrap();
        src.as_u8().unwrap().map().unwrap().as_mut_slice().fill(200);
        let letterbox = crate::ConvertOptions::default().crop(Crop::letterbox([10, 20, 30, 255]));
        let mut cpu = CPUProcessor::new();
        for access in [CpuAccess::Write, CpuAccess::ReadWrite] {
            let mut dst = TensorDyn::image(
                64,
                40,
                PixelFormat::Rgb,
                DType::U8,
                Some(TensorMemory::Mem),
                access,
            )
            .unwrap();
            dst.configure_image(40, 40, PixelFormat::Rgb).unwrap();
            dst.set_row_stride(192).unwrap();
            cpu.convert(&src, &mut dst, Some(letterbox)).unwrap();
            if access.reads() {
                let map = dst.as_u8().unwrap().map_read().unwrap();
                let bytes = map.as_slice();
                assert_eq!(&bytes[..3], &[10, 20, 30], "letterbox band");
                let centre = 20 * 192 + 20 * 3;
                assert_eq!(&bytes[centre..centre + 3], &[200, 200, 200], "interior");
            }
        }
    }

    /// The buffer-geometry guards reject undersized and overflowing inputs with `InvalidShape`.
    #[test]
    fn buffer_geometry_guards_reject_bad_input() {
        use super::super::{guard_plane, split_semi_planar_mut};

        // NV16 combined-plane height = 2*h ⇒ stride 4 × h 2 needs 16 bytes.
        let mut ok = [0u8; 16];
        let (y, uv) = split_semi_planar_mut(&mut ok, 4, 2, PixelFormat::Nv16).unwrap();
        assert_eq!((y.len(), uv.len()), (8, 8)); // luma split at stride*h = 8

        let mut small = [0u8; 12];
        assert!(matches!(
            split_semi_planar_mut(&mut small, 4, 2, PixelFormat::Nv16),
            Err(Error::InvalidShape(_))
        ));
        let mut tiny = [0u8; 1];
        assert!(matches!(
            split_semi_planar_mut(&mut tiny, usize::MAX, 4, PixelFormat::Nv16),
            Err(Error::InvalidShape(_))
        ));

        // guard_plane: ok, row_bytes>stride, buffer too small, overflow.
        assert!(guard_plane(16, 4, 2, 4, "t").is_ok());
        assert!(matches!(
            guard_plane(16, 4, 2, 5, "t"),
            Err(Error::InvalidShape(_))
        ));
        assert!(matches!(
            guard_plane(7, 4, 2, 4, "t"),
            Err(Error::InvalidShape(_))
        ));
        assert!(matches!(
            guard_plane(1, usize::MAX, 4, 1, "t"),
            Err(Error::InvalidShape(_))
        ));
    }

    macro_rules! function {
        () => {{
            fn f() {}
            fn type_name_of<T>(_: T) -> &'static str {
                std::any::type_name::<T>()
            }
            let name = type_name_of(f);

            // Find and cut the rest of the path
            match &name[..name.len() - 3].rfind(':') {
                Some(pos) => &name[pos + 1..name.len() - 3],
                None => &name[..name.len() - 3],
            }
        }};
    }

    fn compare_images_convert_to_grey(
        img1: &TensorDyn,
        img2: &TensorDyn,
        threshold: f64,
        name: &str,
    ) {
        assert_eq!(img1.height(), img2.height(), "Heights differ");
        assert_eq!(img1.width(), img2.width(), "Widths differ");

        // Compare raw bytes as greyscale strip
        let w = img1.width().unwrap() as u32;
        let data1 = img1.as_u8().unwrap().map().unwrap().to_vec();
        let data2 = img2.as_u8().unwrap().map().unwrap().to_vec();
        let h1 = (data1.len() as u32) / w;
        let h2 = (data2.len() as u32) / w;

        let image1 = image::GrayImage::from_vec(w, h1, data1).unwrap();
        let image2 = image::GrayImage::from_vec(w, h2, data2).unwrap();

        let similarity = image_compare::gray_similarity_structure(
            &image_compare::Algorithm::RootMeanSquared,
            &image1,
            &image2,
        )
        .expect("Image Comparison failed");
        if similarity.score < threshold {
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

    fn compare_images_convert_to_rgb(
        img1: &TensorDyn,
        img2: &TensorDyn,
        threshold: f64,
        name: &str,
    ) {
        assert_eq!(img1.height(), img2.height(), "Heights differ");
        assert_eq!(img1.width(), img2.width(), "Widths differ");

        // Converts both to RGB for comparison, or compares raw bytes when the
        // conversion is not supported.
        let mut converter = CPUProcessor::default();
        let w = img1.width().unwrap();
        let h = img1.height().unwrap();

        let (img_rgb1, img_rgb2) = {
            let mut rgb1 = TensorDyn::image(
                w,
                h,
                PixelFormat::Rgb,
                DType::U8,
                mem(),
                edgefirst_tensor::CpuAccess::ReadWrite,
            )
            .unwrap();
            let mut rgb2 = TensorDyn::image(
                w,
                h,
                PixelFormat::Rgb,
                DType::U8,
                mem(),
                edgefirst_tensor::CpuAccess::ReadWrite,
            )
            .unwrap();
            let r1 = converter.convert(img1, &mut rgb1, None);
            let r2 = converter.convert(img2, &mut rgb2, None);
            if r1.is_err() || r2.is_err() {
                // Compares raw bytes as a greyscale strip.
                compare_images_convert_to_grey(img1, img2, threshold, name);
                return;
            }
            (rgb1, rgb2)
        };

        let image1 = image::RgbImage::from_vec(
            img_rgb1.width().unwrap() as u32,
            img_rgb1.height().unwrap() as u32,
            img_rgb1.as_u8().unwrap().map().unwrap().to_vec(),
        )
        .unwrap();

        let image2 = image::RgbImage::from_vec(
            img_rgb2.width().unwrap() as u32,
            img_rgb2.height().unwrap() as u32,
            img_rgb2.as_u8().unwrap().map().unwrap().to_vec(),
        )
        .unwrap();

        let similarity = image_compare::rgb_similarity_structure(
            &image_compare::Algorithm::RootMeanSquared,
            &image1,
            &image2,
        )
        .expect("Image Comparison failed");
        if similarity.score < threshold {
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

    /// Host memory: a tight `Mem` tensor whose row stride equals `width * bpp`.
    fn mem() -> Option<TensorMemory> {
        Some(TensorMemory::Mem)
    }

    fn load_bytes_to_tensor(
        width: usize,
        height: usize,
        format: PixelFormat,
        memory: Option<TensorMemory>,
        bytes: &[u8],
    ) -> Result<TensorDyn, Error> {
        log::debug!("Current function is {}", function!());
        // Defaults to host memory.
        let memory = memory.or_else(mem);
        let src = TensorDyn::image(
            width,
            height,
            format,
            DType::U8,
            memory,
            edgefirst_tensor::CpuAccess::ReadWrite,
        )?;
        src.as_u8().unwrap().map()?.as_mut_slice()[0..bytes.len()].copy_from_slice(bytes);
        Ok(src)
    }

    macro_rules! generate_conversion_tests {
        (
        $src_fmt:expr,  $src_file:expr, $dst_fmt:expr, $dst_file:expr
    ) => {{
            let src = load_bytes_to_tensor(
                1280,
                720,
                $src_fmt,
                None,
                &edgefirst_bench::testdata::read($src_file),
            )?;

            let dst = load_bytes_to_tensor(
                1280,
                720,
                $dst_fmt,
                None,
                &edgefirst_bench::testdata::read($dst_file),
            )?;

            let mut converter = CPUProcessor::default();

            let mut converted = TensorDyn::image(
                src.width().unwrap(),
                src.height().unwrap(),
                dst.format().unwrap(),
                DType::U8,
                None,
                edgefirst_tensor::CpuAccess::ReadWrite,
            )?;

            converter.convert(&src, &mut converted, None)?;

            compare_images_convert_to_rgb(&dst, &converted, 0.99, function!());

            Ok(())
        }};
    }

    macro_rules! generate_conversion_tests_greyscale {
        (
        $src_fmt:expr,  $src_file:expr, $dst_fmt:expr, $dst_file:expr
    ) => {{
            let src = load_bytes_to_tensor(
                1280,
                720,
                $src_fmt,
                None,
                &edgefirst_bench::testdata::read($src_file),
            )?;

            let dst = load_bytes_to_tensor(
                1280,
                720,
                $dst_fmt,
                None,
                &edgefirst_bench::testdata::read($dst_file),
            )?;

            let mut converter = CPUProcessor::default();

            let mut converted = TensorDyn::image(
                src.width().unwrap(),
                src.height().unwrap(),
                dst.format().unwrap(),
                DType::U8,
                None,
                edgefirst_tensor::CpuAccess::ReadWrite,
            )?;

            converter.convert(&src, &mut converted, None)?;

            compare_images_convert_to_grey(&dst, &converted, 0.97, function!());

            Ok(())
        }};
    }

    #[test]
    fn test_cpu_yuyv_to_yuyv() -> Result<()> {
        generate_conversion_tests!(
            PixelFormat::Yuyv,
            "camera720p.yuyv",
            PixelFormat::Yuyv,
            "camera720p.yuyv"
        )
    }

    #[test]
    fn test_cpu_yuyv_to_rgb() -> Result<()> {
        generate_conversion_tests!(
            PixelFormat::Yuyv,
            "camera720p.yuyv",
            PixelFormat::Rgb,
            "camera720p.rgb"
        )
    }

    #[test]
    fn test_cpu_yuyv_to_rgba() -> Result<()> {
        generate_conversion_tests!(
            PixelFormat::Yuyv,
            "camera720p.yuyv",
            PixelFormat::Rgba,
            "camera720p.rgba"
        )
    }

    #[test]
    fn test_cpu_yuyv_to_grey() -> Result<()> {
        generate_conversion_tests!(
            PixelFormat::Yuyv,
            "camera720p.yuyv",
            PixelFormat::Grey,
            "camera720p.y800"
        )
    }

    #[test]
    fn test_cpu_yuyv_to_nv16() -> Result<()> {
        generate_conversion_tests!(
            PixelFormat::Yuyv,
            "camera720p.yuyv",
            PixelFormat::Nv16,
            "camera720p.nv16"
        )
    }

    #[test]
    fn test_cpu_yuyv_to_planar_rgb() -> Result<()> {
        generate_conversion_tests!(
            PixelFormat::Yuyv,
            "camera720p.yuyv",
            PixelFormat::PlanarRgb,
            "camera720p.8bps"
        )
    }

    #[test]
    fn test_cpu_yuyv_to_planar_rgba() -> Result<()> {
        generate_conversion_tests!(
            PixelFormat::Yuyv,
            "camera720p.yuyv",
            PixelFormat::PlanarRgba,
            "camera720p.8bpa"
        )
    }

    #[test]
    fn test_cpu_rgb_to_yuyv() -> Result<()> {
        generate_conversion_tests!(
            PixelFormat::Rgb,
            "camera720p.rgb",
            PixelFormat::Yuyv,
            "camera720p.yuyv"
        )
    }

    #[test]
    fn test_cpu_rgb_to_rgb() -> Result<()> {
        generate_conversion_tests!(
            PixelFormat::Rgb,
            "camera720p.rgb",
            PixelFormat::Rgb,
            "camera720p.rgb"
        )
    }

    #[test]
    fn test_cpu_rgb_to_rgba() -> Result<()> {
        generate_conversion_tests!(
            PixelFormat::Rgb,
            "camera720p.rgb",
            PixelFormat::Rgba,
            "camera720p.rgba"
        )
    }

    #[test]
    fn test_cpu_rgb_to_grey() -> Result<()> {
        generate_conversion_tests!(
            PixelFormat::Rgb,
            "camera720p.rgb",
            PixelFormat::Grey,
            "camera720p.y800"
        )
    }

    #[test]
    fn test_cpu_rgb_to_nv16() -> Result<()> {
        generate_conversion_tests!(
            PixelFormat::Rgb,
            "camera720p.rgb",
            PixelFormat::Nv16,
            "camera720p.nv16"
        )
    }

    #[test]
    fn test_cpu_rgb_to_planar_rgb() -> Result<()> {
        generate_conversion_tests!(
            PixelFormat::Rgb,
            "camera720p.rgb",
            PixelFormat::PlanarRgb,
            "camera720p.8bps"
        )
    }

    #[test]
    fn test_cpu_rgb_to_planar_rgba() -> Result<()> {
        generate_conversion_tests!(
            PixelFormat::Rgb,
            "camera720p.rgb",
            PixelFormat::PlanarRgba,
            "camera720p.8bpa"
        )
    }

    #[test]
    fn test_cpu_rgba_to_yuyv() -> Result<()> {
        generate_conversion_tests!(
            PixelFormat::Rgba,
            "camera720p.rgba",
            PixelFormat::Yuyv,
            "camera720p.yuyv"
        )
    }

    #[test]
    fn test_cpu_rgba_to_rgb() -> Result<()> {
        generate_conversion_tests!(
            PixelFormat::Rgba,
            "camera720p.rgba",
            PixelFormat::Rgb,
            "camera720p.rgb"
        )
    }

    #[test]
    fn test_cpu_rgba_to_rgba() -> Result<()> {
        generate_conversion_tests!(
            PixelFormat::Rgba,
            "camera720p.rgba",
            PixelFormat::Rgba,
            "camera720p.rgba"
        )
    }

    #[test]
    fn test_cpu_rgba_to_grey() -> Result<()> {
        generate_conversion_tests!(
            PixelFormat::Rgba,
            "camera720p.rgba",
            PixelFormat::Grey,
            "camera720p.y800"
        )
    }

    #[test]
    fn test_cpu_rgba_to_nv16() -> Result<()> {
        generate_conversion_tests!(
            PixelFormat::Rgba,
            "camera720p.rgba",
            PixelFormat::Nv16,
            "camera720p.nv16"
        )
    }

    #[test]
    fn test_cpu_rgba_to_planar_rgb() -> Result<()> {
        generate_conversion_tests!(
            PixelFormat::Rgba,
            "camera720p.rgba",
            PixelFormat::PlanarRgb,
            "camera720p.8bps"
        )
    }

    #[test]
    fn test_cpu_rgba_to_planar_rgba() -> Result<()> {
        generate_conversion_tests!(
            PixelFormat::Rgba,
            "camera720p.rgba",
            PixelFormat::PlanarRgba,
            "camera720p.8bpa"
        )
    }

    #[test]
    fn test_cpu_nv12_to_rgb() -> Result<()> {
        generate_conversion_tests!(
            PixelFormat::Nv12,
            "camera720p.nv12",
            PixelFormat::Rgb,
            "camera720p.rgb"
        )
    }

    #[test]
    fn test_cpu_nv12_to_yuyv() -> Result<()> {
        generate_conversion_tests!(
            PixelFormat::Nv12,
            "camera720p.nv12",
            PixelFormat::Yuyv,
            "camera720p.yuyv"
        )
    }

    #[test]
    fn test_cpu_nv12_to_rgba() -> Result<()> {
        generate_conversion_tests!(
            PixelFormat::Nv12,
            "camera720p.nv12",
            PixelFormat::Rgba,
            "camera720p.rgba"
        )
    }

    #[test]
    fn test_cpu_nv12_to_grey() -> Result<()> {
        generate_conversion_tests!(
            PixelFormat::Nv12,
            "camera720p.nv12",
            PixelFormat::Grey,
            "camera720p.y800"
        )
    }

    #[test]
    fn test_cpu_nv12_to_nv16() -> Result<()> {
        generate_conversion_tests!(
            PixelFormat::Nv12,
            "camera720p.nv12",
            PixelFormat::Nv16,
            "camera720p.nv16"
        )
    }

    #[test]
    fn test_cpu_nv12_to_planar_rgb() -> Result<()> {
        generate_conversion_tests!(
            PixelFormat::Nv12,
            "camera720p.nv12",
            PixelFormat::PlanarRgb,
            "camera720p.8bps"
        )
    }

    #[test]
    fn test_cpu_nv12_to_planar_rgba() -> Result<()> {
        generate_conversion_tests!(
            PixelFormat::Nv12,
            "camera720p.nv12",
            PixelFormat::PlanarRgba,
            "camera720p.8bpa"
        )
    }

    #[test]
    fn test_cpu_grey_to_yuyv() -> Result<()> {
        generate_conversion_tests_greyscale!(
            PixelFormat::Grey,
            "camera720p.y800",
            PixelFormat::Yuyv,
            "camera720p.yuyv"
        )
    }

    #[test]
    fn test_cpu_grey_to_rgb() -> Result<()> {
        generate_conversion_tests_greyscale!(
            PixelFormat::Grey,
            "camera720p.y800",
            PixelFormat::Rgb,
            "camera720p.rgb"
        )
    }

    #[test]
    fn test_cpu_grey_to_rgba() -> Result<()> {
        generate_conversion_tests_greyscale!(
            PixelFormat::Grey,
            "camera720p.y800",
            PixelFormat::Rgba,
            "camera720p.rgba"
        )
    }

    #[test]
    fn test_cpu_grey_to_grey() -> Result<()> {
        generate_conversion_tests_greyscale!(
            PixelFormat::Grey,
            "camera720p.y800",
            PixelFormat::Grey,
            "camera720p.y800"
        )
    }

    #[test]
    fn test_cpu_grey_to_nv16() -> Result<()> {
        generate_conversion_tests_greyscale!(
            PixelFormat::Grey,
            "camera720p.y800",
            PixelFormat::Nv16,
            "camera720p.nv16"
        )
    }

    #[test]
    fn test_cpu_grey_to_planar_rgb() -> Result<()> {
        generate_conversion_tests_greyscale!(
            PixelFormat::Grey,
            "camera720p.y800",
            PixelFormat::PlanarRgb,
            "camera720p.8bps"
        )
    }

    #[test]
    fn test_cpu_grey_to_planar_rgba() -> Result<()> {
        generate_conversion_tests_greyscale!(
            PixelFormat::Grey,
            "camera720p.y800",
            PixelFormat::PlanarRgba,
            "camera720p.8bpa"
        )
    }

    // ========================================================================
    // VYUY conversion tests
    // ========================================================================

    #[test]
    fn test_cpu_vyuy_to_grey() -> Result<()> {
        generate_conversion_tests!(
            PixelFormat::Vyuy,
            "camera720p.vyuy",
            PixelFormat::Grey,
            "camera720p.y800"
        )
    }

    #[test]
    fn test_cpu_vyuy_to_planar_rgb() -> Result<()> {
        generate_conversion_tests!(
            PixelFormat::Vyuy,
            "camera720p.vyuy",
            PixelFormat::PlanarRgb,
            "camera720p.8bps"
        )
    }

    #[test]
    fn test_cpu_vyuy_to_nv16() -> Result<()> {
        generate_conversion_tests!(
            PixelFormat::Vyuy,
            "camera720p.vyuy",
            PixelFormat::Nv16,
            "camera720p.nv16"
        )
    }

    #[test]
    fn test_cpu_yuyv_to_nv12() -> Result<()> {
        generate_conversion_tests!(
            PixelFormat::Yuyv,
            "camera720p.yuyv",
            PixelFormat::Nv12,
            "camera720p.nv12"
        )
    }

    #[test]
    fn test_cpu_vyuy_to_nv12() -> Result<()> {
        generate_conversion_tests!(
            PixelFormat::Vyuy,
            "camera720p.vyuy",
            PixelFormat::Nv12,
            "camera720p.nv12"
        )
    }

    #[test]
    fn test_cpu_rgba_to_nv12() -> Result<()> {
        generate_conversion_tests!(
            PixelFormat::Rgba,
            "camera720p.rgba",
            PixelFormat::Nv12,
            "camera720p.nv12"
        )
    }

    #[test]
    fn test_cpu_rgb_to_nv12() -> Result<()> {
        generate_conversion_tests!(
            PixelFormat::Rgb,
            "camera720p.rgb",
            PixelFormat::Nv12,
            "camera720p.nv12"
        )
    }

    #[test]
    fn grey_to_nv12_is_neutral_chroma() -> Result<()> {
        let (w, h) = (64, 16);
        let src = Tensor::<u8>::image(
            w,
            h,
            PixelFormat::Grey,
            Some(TensorMemory::Mem),
            edgefirst_tensor::CpuAccess::ReadWrite,
        )?;
        {
            let mut map = src.map_mut()?;
            for (i, b) in map.as_mut_slice().iter_mut().enumerate() {
                *b = if i % 2 == 0 { 0 } else { 255 };
            }
        }
        let src = TensorDyn::from(src);
        let mut dst = TensorDyn::image(
            w,
            h,
            PixelFormat::Nv12,
            DType::U8,
            Some(TensorMemory::Mem),
            edgefirst_tensor::CpuAccess::ReadWrite,
        )?;
        CPUProcessor::new().convert(&src, &mut dst, None)?;
        let map = dst.as_u8().unwrap().map_read()?;
        let (luma, chroma) = map.as_slice().split_at(w * h);
        assert!(chroma[..w * h / 2].iter().all(|&c| c == 128));
        for (i, &y) in luma.iter().enumerate() {
            assert_eq!(y, if i % 2 == 0 { 16 } else { 235 }, "luma {i}");
        }
        Ok(())
    }

    /// A multiplane NV12 destination whose luma pitch is `luma_pitch` and chroma pitch is
    /// `chroma_pitch`, with every byte set to `fill`.
    fn padded_multiplane_nv12(
        w: usize,
        h: usize,
        luma_pitch: usize,
        chroma_pitch: usize,
        fill: u8,
    ) -> Result<TensorDyn> {
        use edgefirst_tensor::TensorTrait;
        let mut luma = Tensor::<u8>::new(&[h, luma_pitch], Some(TensorMemory::Mem), None)?;
        let mut chroma = Tensor::<u8>::new(&[h / 2, chroma_pitch], Some(TensorMemory::Mem), None)?;
        luma.map_mut()?.as_mut_slice().fill(fill);
        chroma.map_mut()?.as_mut_slice().fill(fill);
        luma.set_logical_shape(&[h, w])?;
        chroma.set_logical_shape(&[h / 2, w])?;
        chroma.set_row_stride_unchecked(chroma_pitch);
        let mut dst = TensorDyn::from(Tensor::<u8>::from_planes(luma, chroma, PixelFormat::Nv12)?);
        dst.set_row_stride(luma_pitch)?;
        Ok(dst)
    }

    #[test]
    fn nv12_to_nv12_copies_between_strides() -> Result<()> {
        let (w, h, pitch) = (64, 16, 128);
        let src = Tensor::<u8>::image(
            w,
            h,
            PixelFormat::Nv12,
            Some(TensorMemory::Mem),
            edgefirst_tensor::CpuAccess::ReadWrite,
        )?;
        let bytes: Vec<u8> = (0..w * h * 3 / 2).map(|i| (i * 7 % 251) as u8).collect();
        src.map_mut()?.as_mut_slice().copy_from_slice(&bytes);
        let src = TensorDyn::from(src);
        let mut dst = padded_multiplane_nv12(w, h, pitch, pitch, 0xA5)?;
        CPUProcessor::new().convert(&src, &mut dst, None)?;
        let dst = dst.as_u8().unwrap();
        let luma_map = dst.map_read()?;
        let chroma_map = dst.chroma().unwrap().map_read()?;
        let (luma, chroma) = bytes.split_at(w * h);
        for row in 0..h {
            assert_eq!(
                &luma_map.as_slice()[row * pitch..][..w],
                &luma[row * w..][..w]
            );
            assert!(luma_map.as_slice()[row * pitch + w..][..pitch - w]
                .iter()
                .all(|&b| b == 0xA5));
        }
        for row in 0..h / 2 {
            assert_eq!(
                &chroma_map.as_slice()[row * pitch..][..w],
                &chroma[row * w..][..w]
            );
            assert!(chroma_map.as_slice()[row * pitch + w..][..pitch - w]
                .iter()
                .all(|&b| b == 0xA5));
        }
        Ok(())
    }

    #[test]
    fn rgba_to_nv12_writes_a_multiplane_destination() -> Result<()> {
        let (w, h) = (64, 16);
        let src = Tensor::<u8>::image(
            w,
            h,
            PixelFormat::Rgba,
            Some(TensorMemory::Mem),
            edgefirst_tensor::CpuAccess::ReadWrite,
        )?;
        for (i, b) in src.map_mut()?.as_mut_slice().iter_mut().enumerate() {
            *b = (i * 13 % 253) as u8;
        }
        let src = TensorDyn::from(src);
        let mut want = TensorDyn::image(
            w,
            h,
            PixelFormat::Nv12,
            DType::U8,
            Some(TensorMemory::Mem),
            edgefirst_tensor::CpuAccess::ReadWrite,
        )?;
        CPUProcessor::new().convert(&src, &mut want, None)?;
        let want = want.as_u8().unwrap().map_read()?;
        let (want_luma, want_chroma) = want.as_slice().split_at(w * h);

        for (luma_pitch, chroma_pitch) in [(w, w), (128, 128)] {
            let mut dst = padded_multiplane_nv12(w, h, luma_pitch, chroma_pitch, 0xA5)?;
            CPUProcessor::new().convert(&src, &mut dst, None)?;
            let dst = dst.as_u8().unwrap();
            let luma_map = dst.map_read()?;
            let chroma_map = dst.chroma().unwrap().map_read()?;
            for row in 0..h {
                let got = &luma_map.as_slice()[row * luma_pitch..][..luma_pitch];
                assert_eq!(&got[..w], &want_luma[row * w..][..w], "luma row {row}");
                assert!(got[w..].iter().all(|&b| b == 0xA5), "luma pad row {row}");
            }
            for row in 0..h / 2 {
                let got = &chroma_map.as_slice()[row * chroma_pitch..][..chroma_pitch];
                assert_eq!(&got[..w], &want_chroma[row * w..][..w], "chroma row {row}");
                assert!(got[w..].iter().all(|&b| b == 0xA5), "chroma pad row {row}");
            }
        }
        Ok(())
    }

    /// Packed 4:2:2 source with `[Y0, U, Y1, V]` (or the VYUY rotation) at a padded stride.
    fn packed422_source(
        w: usize,
        h: usize,
        stride: usize,
        fmt: PixelFormat,
    ) -> Result<(TensorDyn, Vec<u8>)> {
        let mut t = Tensor::<u8>::image(
            stride / 2,
            h,
            fmt,
            Some(TensorMemory::Mem),
            edgefirst_tensor::CpuAccess::ReadWrite,
        )?;
        t.configure_image(w, h, fmt)?;
        t.set_row_stride(stride)?;
        let mut bytes = vec![0xEEu8; stride * h];
        for y in 0..h {
            for i in 0..w * 2 {
                bytes[y * stride + i] = ((y * 37 + i * 11) % 251) as u8;
            }
        }
        t.map_mut()?.as_mut_slice().copy_from_slice(&bytes);
        Ok((TensorDyn::from(t), bytes))
    }

    /// Reference NV12 planes `(luma w*h, chroma w*h/2)` of a packed 4:2:2 buffer.
    fn nv12_reference(
        bytes: &[u8],
        stride: usize,
        w: usize,
        h: usize,
        fmt: PixelFormat,
    ) -> (Vec<u8>, Vec<u8>) {
        let (y0, u, y1, v) = if fmt == PixelFormat::Yuyv {
            (0, 1, 2, 3)
        } else {
            (1, 2, 3, 0)
        };
        let mut luma = vec![0u8; w * h];
        for y in 0..h {
            for m in 0..w / 2 {
                luma[y * w + m * 2] = bytes[y * stride + m * 4 + y0];
                luma[y * w + m * 2 + 1] = bytes[y * stride + m * 4 + y1];
            }
        }
        let mut chroma = vec![0u8; w * h / 2];
        let mean = |a: u8, b: u8| ((a as u16 + b as u16 + 1) >> 1) as u8;
        for r in 0..h / 2 {
            for m in 0..w / 2 {
                let (t, b) = (2 * r * stride + m * 4, (2 * r + 1) * stride + m * 4);
                chroma[r * w + m * 2] = mean(bytes[t + u], bytes[b + u]);
                chroma[r * w + m * 2 + 1] = mean(bytes[t + v], bytes[b + v]);
            }
        }
        (luma, chroma)
    }

    fn packed422_to_nv12_matches_a_reference(fmt: PixelFormat) -> Result<()> {
        let (w, h, stride) = (64, 16, 160);
        let (src, bytes) = packed422_source(w, h, stride, fmt)?;
        let mut dst = TensorDyn::image(
            w,
            h,
            PixelFormat::Nv12,
            DType::U8,
            Some(TensorMemory::Mem),
            edgefirst_tensor::CpuAccess::ReadWrite,
        )?;
        CPUProcessor::new().convert(&src, &mut dst, None)?;
        let (luma, chroma) = nv12_reference(&bytes, stride, w, h, fmt);
        let map = dst.as_u8().unwrap().map_read()?;
        assert_eq!(&map.as_slice()[..w * h], &luma[..]);
        assert_eq!(&map.as_slice()[w * h..w * h + w * h / 2], &chroma[..]);
        Ok(())
    }

    #[test]
    fn yuyv_to_nv12_matches_a_reference_on_a_padded_source() -> Result<()> {
        packed422_to_nv12_matches_a_reference(PixelFormat::Yuyv)
    }

    #[test]
    fn vyuy_to_nv12_matches_a_reference() -> Result<()> {
        packed422_to_nv12_matches_a_reference(PixelFormat::Vyuy)
    }

    #[test]
    fn yuyv_to_nv12_writes_a_multiplane_destination() -> Result<()> {
        let (w, h, stride) = (64, 16, 160);
        let (src, bytes) = packed422_source(w, h, stride, PixelFormat::Yuyv)?;
        let luma = Tensor::<u8>::new(&[h, w], Some(TensorMemory::Mem), None)?;
        let chroma = Tensor::<u8>::new(&[h / 2, w], Some(TensorMemory::Mem), None)?;
        luma.map_mut()?.as_mut_slice().fill(0);
        chroma.map_mut()?.as_mut_slice().fill(0);
        let mut dst = TensorDyn::from(Tensor::<u8>::from_planes(luma, chroma, PixelFormat::Nv12)?);
        CPUProcessor::new().convert(&src, &mut dst, None)?;
        let (want_luma, want_chroma) = nv12_reference(&bytes, stride, w, h, PixelFormat::Yuyv);
        let dst = dst.as_u8().unwrap();
        let luma_map = dst.map_read()?;
        let chroma_map = dst.chroma().unwrap().map_read()?;
        assert_eq!(luma_map.as_slice(), &want_luma[..]);
        assert_eq!(chroma_map.as_slice(), &want_chroma[..]);
        Ok(())
    }

    #[test]
    fn yuyv_to_nv12_leaves_destination_padding_alone() -> Result<()> {
        let (w, h, pitch) = (96, 16, 128);
        let (src, _) = packed422_source(w, h, w * 2, PixelFormat::Yuyv)?;
        let mut dst = TensorDyn::image(
            pitch,
            h,
            PixelFormat::Nv12,
            DType::U8,
            Some(TensorMemory::Mem),
            edgefirst_tensor::CpuAccess::ReadWrite,
        )?;
        dst.configure_image(w, h, PixelFormat::Nv12)?;
        dst.set_row_stride(pitch)?;
        dst.as_u8().unwrap().map_mut()?.as_mut_slice().fill(0xA5);
        CPUProcessor::new().convert(&src, &mut dst, None)?;
        let map = dst.as_u8().unwrap().map_read()?;
        let bytes = map.as_slice();
        for row in 0..h + h / 2 {
            assert!(
                bytes[row * pitch + w..(row + 1) * pitch]
                    .iter()
                    .all(|&b| b == 0xA5),
                "row {row} padding was written"
            );
        }
        Ok(())
    }

    #[test]
    fn an_odd_nv12_destination_is_refused() -> Result<()> {
        let (src, _) = packed422_source(64, 16, 128, PixelFormat::Yuyv)?;
        for (w, h) in [(63usize, 16usize), (64, 15)] {
            let mut dst = TensorDyn::image(
                w,
                h,
                PixelFormat::Nv12,
                DType::U8,
                Some(TensorMemory::Mem),
                edgefirst_tensor::CpuAccess::ReadWrite,
            )?;
            let err = CPUProcessor::new()
                .convert(&src, &mut dst, None)
                .unwrap_err();
            assert!(
                matches!(&err, Error::NotSupported(m) if m.contains(&format!("{w}x{h}"))),
                "{w}x{h}: {err:?}"
            );
        }
        Ok(())
    }

    fn nv12_dst(w: usize, h: usize) -> TensorDyn {
        TensorDyn::image(
            w,
            h,
            PixelFormat::Nv12,
            DType::U8,
            Some(TensorMemory::Mem),
            edgefirst_tensor::CpuAccess::ReadWrite,
        )
        .unwrap()
    }

    fn rgb_dst(w: usize, h: usize) -> TensorDyn {
        TensorDyn::image(
            w,
            h,
            PixelFormat::Rgb,
            DType::U8,
            Some(TensorMemory::Mem),
            edgefirst_tensor::CpuAccess::ReadWrite,
        )
        .unwrap()
    }

    fn to_rgb(src: &TensorDyn) -> TensorDyn {
        let mut out = rgb_dst(src.width().unwrap(), src.height().unwrap());
        CPUProcessor::new().convert(src, &mut out, None).unwrap();
        out
    }

    fn rgb_bytes(t: &TensorDyn) -> Vec<u8> {
        t.as_u8().unwrap().map_read().unwrap().as_slice().to_vec()
    }

    fn mean_abs_diff(a: &TensorDyn, b: &TensorDyn) -> f64 {
        let (a, b) = (rgb_bytes(a), rgb_bytes(b));
        assert_eq!(a.len(), b.len());
        a.iter()
            .zip(&b)
            .map(|(&x, &y)| (x as f64 - y as f64).abs())
            .sum::<f64>()
            / a.len() as f64
    }

    fn rgb_ssim(a: &TensorDyn, b: &TensorDyn) -> f64 {
        let to_img = |t: &TensorDyn| {
            image::RgbImage::from_vec(
                t.width().unwrap() as u32,
                t.height().unwrap() as u32,
                rgb_bytes(t),
            )
            .unwrap()
        };
        image_compare::rgb_similarity_structure(
            &image_compare::Algorithm::RootMeanSquared,
            &to_img(a),
            &to_img(b),
        )
        .expect("Image Comparison failed")
        .score
    }

    fn tagged(
        mut t: TensorDyn,
        encoding: edgefirst_tensor::ColorEncoding,
        range: edgefirst_tensor::ColorRange,
    ) -> TensorDyn {
        t.set_colorimetry(Some(
            edgefirst_tensor::Colorimetry::default()
                .with_encoding(encoding)
                .with_range(range),
        ));
        t
    }

    fn camera720p_yuyv() -> TensorDyn {
        load_bytes_to_tensor(
            1280,
            720,
            PixelFormat::Yuyv,
            None,
            &edgefirst_bench::testdata::read("camera720p.yuyv"),
        )
        .unwrap()
    }

    #[test]
    fn a_resized_yuyv_decodes_back_to_its_rgb() -> Result<()> {
        let src = camera720p_yuyv();
        let mut cpu = CPUProcessor::new();
        let mut nv12 = nv12_dst(640, 360);
        cpu.convert(&src, &mut nv12, None)?;
        let mut want = rgb_dst(640, 360);
        cpu.convert(&to_rgb(&src), &mut want, None)?;
        let ssim = rgb_ssim(&to_rgb(&nv12), &want);
        assert!(ssim >= 0.98, "ssim {ssim}");
        Ok(())
    }

    #[test]
    fn an_odd_height_source_resizes_into_nv12() -> Result<()> {
        let bytes = edgefirst_bench::testdata::read("camera720p.yuyv");
        let mut rows = bytes.to_vec();
        rows.extend_from_slice(&bytes[..1280 * 2]);
        assert_eq!(rows.len(), 1280 * 721 * 2);
        let src = load_bytes_to_tensor(1280, 721, PixelFormat::Yuyv, None, &rows)?;
        let mut nv12 = nv12_dst(640, 360);
        CPUProcessor::new().convert(&src, &mut nv12, None)?;
        Ok(())
    }

    #[test]
    fn a_letterboxed_nv12_fills_its_chroma_band() -> Result<()> {
        let (w, h) = (64, 64);
        let red = [255, 0, 0, 255];
        let src = load_bytes_to_tensor(
            w,
            h / 2,
            PixelFormat::Rgba,
            None,
            &vec![0x80; w * h / 2 * 4],
        )?;
        let mut dst = nv12_dst(w, h);
        CPUProcessor::new().convert(
            &src,
            &mut dst,
            Some(crate::ConvertOptions::new().crop(Crop::letterbox(red))),
        )?;

        let one = load_bytes_to_tensor(
            64,
            16,
            PixelFormat::Rgba,
            None,
            &[255, 0, 0, 255].repeat(64 * 16),
        )?;
        let mut one_nv12 = nv12_dst(64, 16);
        CPUProcessor::new().convert(&one, &mut one_nv12, None)?;
        let one_map = one_nv12.as_u8().unwrap().map_read()?;
        let (want_y, want_u, want_v) = (
            one_map.as_slice()[0],
            one_map.as_slice()[64 * 16],
            one_map.as_slice()[64 * 16 + 1],
        );

        let map = dst.as_u8().unwrap().map_read()?;
        let (luma, chroma) = map.as_slice().split_at(w * h);
        for row in (0..16).chain(48..64) {
            assert!(
                luma[row * w..(row + 1) * w].iter().all(|&y| y == want_y),
                "luma row {row}"
            );
        }
        for row in (0..8).chain(24..32) {
            for pair in chroma[row * w..(row + 1) * w].chunks(2) {
                assert_eq!(pair, [want_u, want_v], "chroma row {row}");
            }
        }
        Ok(())
    }

    /// A letterbox at odd edges, through a processor whose intermediate holds another colour,
    /// encodes each chroma pair straddling an edge from the fill and image colours, with the
    /// destination's colorimetry, and keeps every image pixel next to the edge.
    #[test]
    fn a_letterbox_at_odd_edges_encodes_straddling_chroma_from_the_fill() -> Result<()> {
        use edgefirst_tensor::{ColorEncoding, ColorRange};
        let (w, h) = (64usize, 32usize);
        let (green, blue, red) = ([20u8, 200, 40, 255], [0u8, 0, 255, 255], [255u8, 0, 0, 255]);
        let solid = |sw: usize, sh: usize, c: [u8; 4]| {
            load_bytes_to_tensor(sw, sh, PixelFormat::Rgba, None, &c.repeat(sw * sh))
        };
        // Source size, and the destination columns and rows the image covers.
        let shapes = [((50usize, 32usize), 7..57, 0..32), ((64, 25), 0..64, 3..28)];
        for fmt in [PixelFormat::Nv12, PixelFormat::Nv16, PixelFormat::Yuyv] {
            let chroma_rows = if fmt == PixelFormat::Nv12 { h / 2 } else { h };
            let dst = || {
                let t = TensorDyn::image(
                    w,
                    h,
                    fmt,
                    DType::U8,
                    mem(),
                    edgefirst_tensor::CpuAccess::ReadWrite,
                )
                .unwrap();
                tagged(t, ColorEncoding::Bt709, ColorRange::Full)
            };
            for ((sw, sh), cols, rows) in shapes.clone() {
                let label = format!("{fmt:?} {sw}x{sh}");
                let mut cpu = CPUProcessor::new();
                cpu.convert(&solid(w / 2, h, blue)?, &mut dst(), None)?;
                let mut got = dst();
                cpu.convert(
                    &solid(sw, sh, green)?,
                    &mut got,
                    Some(crate::ConvertOptions::new().crop(Crop::letterbox(red))),
                )?;

                let mut picture = Vec::with_capacity(w * h * 4);
                for y in 0..h {
                    for x in 0..w {
                        let inside = cols.contains(&x) && rows.contains(&y);
                        picture.extend_from_slice(if inside { &green } else { &red });
                    }
                }
                let mut want = dst();
                CPUProcessor::new().convert(
                    &load_bytes_to_tensor(w, h, PixelFormat::Rgba, None, &picture)?,
                    &mut want,
                    None,
                )?;

                let got_map = got.as_u8().unwrap().map_read()?;
                let want_map = want.as_u8().unwrap().map_read()?;
                if fmt == PixelFormat::Yuyv {
                    let (got_b, want_b) = (got_map.as_slice(), want_map.as_slice());
                    for (i, (g, e)) in got_b[..w * h * 2]
                        .iter()
                        .zip(&want_b[..w * h * 2])
                        .enumerate()
                    {
                        assert!(
                            g.abs_diff(*e) <= 1,
                            "{label}: row {} column {} byte {}: {g} vs {e}",
                            i / (w * 2),
                            (i % (w * 2)) / 2,
                            i % 4
                        );
                    }
                    let chroma = |r: usize, m: usize| {
                        let o = r * w * 2 + m * 4;
                        [want_b[o + 1], want_b[o + 3]]
                    };
                    if sw < w {
                        for m in [3, 28] {
                            let mixed = chroma(0, m);
                            assert!(
                                mixed != chroma(0, 0) && mixed != chroma(10, 10),
                                "{label}: straddling macropixel {m} chroma {mixed:?}"
                            );
                        }
                    }
                    continue;
                }
                let (got_y, got_uv) = got_map.as_slice().split_at(w * h);
                let (want_y, want_uv) = want_map.as_slice().split_at(w * h);
                for (i, (g, e)) in got_y.iter().zip(want_y).enumerate() {
                    assert!(g.abs_diff(*e) <= 1, "{label}: luma {i}: {g} vs {e}");
                }
                for (i, (g, e)) in got_uv[..w * chroma_rows]
                    .iter()
                    .zip(&want_uv[..w * chroma_rows])
                    .enumerate()
                {
                    assert!(
                        g.abs_diff(*e) <= 1,
                        "{label}: chroma row {} byte {}: {g} vs {e}",
                        i / w,
                        i % w
                    );
                }

                // A straddling pair differs from the fill's and the image's.
                let pair = |r: usize, cx: usize| &want_uv[r * w + 2 * cx..][..2];
                let straddle = match (sw < w, fmt) {
                    (true, _) => Some((0, 3)),
                    (false, PixelFormat::Nv12) => Some((1, 0)),
                    _ => None,
                };
                if let Some((r, cx)) = straddle {
                    let mixed = pair(r, cx);
                    assert!(
                        mixed != pair(0, 0) && mixed != pair(10, 10),
                        "{label}: straddling pair {mixed:?}"
                    );
                }
            }
        }
        Ok(())
    }

    #[test]
    fn an_untagged_nv12_destination_takes_the_colorimetry_of_its_height() -> Result<()> {
        use edgefirst_tensor::{ColorEncoding, ColorRange};
        let mut bytes = Vec::with_capacity(1280 * 720 * 2);
        for y in 0..720usize {
            for m in 0..640usize {
                let luma = 40 + (m * 170 / 640) as u8;
                let (u, v) = (
                    if y % 128 < 64 { 200 } else { 60 },
                    70 + (y * 150 / 720) as u8,
                );
                bytes.extend_from_slice(&[luma, u, luma, v]);
            }
        }
        let src = load_bytes_to_tensor(1280, 720, PixelFormat::Yuyv, None, &bytes)?;
        let src = tagged(src, ColorEncoding::Bt709, ColorRange::Limited);
        let mut cpu = CPUProcessor::new();
        let mut nv12 = nv12_dst(640, 360);
        cpu.convert(&src, &mut nv12, None)?;
        let tag = nv12.colorimetry().expect("the destination is tagged");
        assert_eq!(tag.encoding, Some(ColorEncoding::Bt601));
        assert_eq!(tag.range, Some(ColorRange::Limited));
        let mut want = rgb_dst(640, 360);
        cpu.convert(&to_rgb(&src), &mut want, None)?;
        let diff = mean_abs_diff(&to_rgb(&nv12), &want);
        assert!(diff <= 3.0, "mean diff {diff}");
        Ok(())
    }

    #[test]
    fn a_tagged_destination_encodes_with_its_own_colorimetry() -> Result<()> {
        use edgefirst_tensor::{ColorEncoding, ColorRange};
        let src = tagged(camera720p_yuyv(), ColorEncoding::Bt601, ColorRange::Limited);
        let mut cpu = CPUProcessor::new();
        let mut nv12 = tagged(nv12_dst(640, 360), ColorEncoding::Bt709, ColorRange::Full);
        cpu.convert(&src, &mut nv12, None)?;
        let mut want = rgb_dst(640, 360);
        cpu.convert(&to_rgb(&src), &mut want, None)?;
        let diff = mean_abs_diff(&to_rgb(&nv12), &want);
        assert!(diff <= 3.0, "mean diff {diff}");
        Ok(())
    }

    #[test]
    fn flips_and_rotations_into_nv12_match_rgba() -> Result<()> {
        let src = load_bytes_to_tensor(
            1280,
            720,
            PixelFormat::Rgba,
            None,
            &edgefirst_bench::testdata::read("camera720p.rgba"),
        )?;
        let rgba_dst = |w, h| {
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
        let cases = [
            (Rotation::None, Flip::Horizontal, (640, 360)),
            (Rotation::None, Flip::Vertical, (640, 360)),
            (Rotation::Clockwise90, Flip::None, (360, 640)),
        ];
        for (rotation, flip, (w, h)) in cases {
            let opts = crate::ConvertOptions::new().rotation(rotation).flip(flip);
            let mut cpu = CPUProcessor::new();
            let mut nv12 = nv12_dst(w, h);
            cpu.convert(&src, &mut nv12, Some(opts))?;
            let mut want = rgba_dst(w, h);
            cpu.convert(&src, &mut want, Some(opts))?;
            let mut back = rgb_dst(w, h);
            cpu.convert(&nv12, &mut back, None)?;
            let mut want_rgb = rgb_dst(w, h);
            cpu.convert(&want, &mut want_rgb, None)?;
            let ssim = rgb_ssim(&back, &want_rgb);
            assert!(ssim >= 0.98, "{rotation:?} {flip:?}: ssim {ssim}");
        }
        Ok(())
    }

    // ========================================================================
    // NV16 conversion tests
    // ========================================================================

    #[test]
    fn test_cpu_nv16_to_rgb() -> Result<()> {
        generate_conversion_tests!(
            PixelFormat::Nv16,
            "camera720p.nv16",
            PixelFormat::Rgb,
            "camera720p.rgb"
        )
    }

    #[test]
    fn test_cpu_nv16_to_rgba() -> Result<()> {
        generate_conversion_tests!(
            PixelFormat::Nv16,
            "camera720p.nv16",
            PixelFormat::Rgba,
            "camera720p.rgba"
        )
    }

    #[test]
    fn test_cpu_nearest() -> Result<()> {
        let src = load_bytes_to_tensor(2, 1, PixelFormat::Rgb, None, &[0, 0, 0, 255, 255, 255])?;

        let mut converter = CPUProcessor::new_nearest();

        let converted = TensorDyn::image(
            4,
            1,
            PixelFormat::Rgb,
            DType::U8,
            mem(),
            edgefirst_tensor::CpuAccess::ReadWrite,
        )?;
        let src_dyn = src;
        let mut converted_dyn = converted;

        converter.convert(&src_dyn, &mut converted_dyn, None)?;

        assert_eq!(
            &converted_dyn.as_u8().unwrap().map()?.as_slice(),
            &[0, 0, 0, 0, 0, 0, 255, 255, 255, 255, 255, 255]
        );

        Ok(())
    }

    #[test]
    fn test_cpu_rotate_cw() -> Result<()> {
        let src = load_bytes_to_tensor(
            2,
            2,
            PixelFormat::Rgba,
            None,
            &[0, 0, 0, 255, 1, 1, 1, 255, 2, 2, 2, 255, 3, 3, 3, 255],
        )?;

        let mut converter = CPUProcessor::default();

        let converted = TensorDyn::image(
            4,
            4,
            PixelFormat::Rgba,
            DType::U8,
            mem(),
            edgefirst_tensor::CpuAccess::ReadWrite,
        )?;
        let src_dyn = src;
        let mut converted_dyn = converted;

        converter.convert(
            &src_dyn,
            &mut converted_dyn,
            Some(crate::ConvertOptions::new().rotation(Rotation::Clockwise90)),
        )?;

        let map = converted_dyn.as_u8().unwrap().map()?;
        assert_eq!(&map.as_slice()[0..4], &[2, 2, 2, 255]);
        assert_eq!(&map.as_slice()[12..16], &[0, 0, 0, 255]);
        assert_eq!(&map.as_slice()[48..52], &[3, 3, 3, 255]);
        assert_eq!(&map.as_slice()[60..64], &[1, 1, 1, 255]);

        Ok(())
    }

    #[test]
    fn test_cpu_rotate_ccw() -> Result<()> {
        let src = load_bytes_to_tensor(
            2,
            2,
            PixelFormat::Rgba,
            None,
            &[0, 0, 0, 255, 1, 1, 1, 255, 2, 2, 2, 255, 3, 3, 3, 255],
        )?;

        let mut converter = CPUProcessor::default();

        let converted = TensorDyn::image(
            4,
            4,
            PixelFormat::Rgba,
            DType::U8,
            mem(),
            edgefirst_tensor::CpuAccess::ReadWrite,
        )?;
        let src_dyn = src;
        let mut converted_dyn = converted;

        converter.convert(
            &src_dyn,
            &mut converted_dyn,
            Some(crate::ConvertOptions::new().rotation(Rotation::CounterClockwise90)),
        )?;

        let map = converted_dyn.as_u8().unwrap().map()?;
        assert_eq!(&map.as_slice()[0..4], &[1, 1, 1, 255]);
        assert_eq!(&map.as_slice()[12..16], &[3, 3, 3, 255]);
        assert_eq!(&map.as_slice()[48..52], &[0, 0, 0, 255]);
        assert_eq!(&map.as_slice()[60..64], &[2, 2, 2, 255]);

        Ok(())
    }

    #[test]
    fn test_cpu_rotate_180() -> Result<()> {
        let src = load_bytes_to_tensor(
            2,
            2,
            PixelFormat::Rgba,
            None,
            &[0, 0, 0, 255, 1, 1, 1, 255, 2, 2, 2, 255, 3, 3, 3, 255],
        )?;

        let mut converter = CPUProcessor::default();

        let converted = TensorDyn::image(
            4,
            4,
            PixelFormat::Rgba,
            DType::U8,
            mem(),
            edgefirst_tensor::CpuAccess::ReadWrite,
        )?;
        let src_dyn = src;
        let mut converted_dyn = converted;

        converter.convert(
            &src_dyn,
            &mut converted_dyn,
            Some(crate::ConvertOptions::new().rotation(Rotation::Rotate180)),
        )?;

        let map = converted_dyn.as_u8().unwrap().map()?;
        assert_eq!(&map.as_slice()[0..4], &[3, 3, 3, 255]);
        assert_eq!(&map.as_slice()[12..16], &[2, 2, 2, 255]);
        assert_eq!(&map.as_slice()[48..52], &[1, 1, 1, 255]);
        assert_eq!(&map.as_slice()[60..64], &[0, 0, 0, 255]);

        Ok(())
    }

    #[test]
    fn test_cpu_flip_v() -> Result<()> {
        let src = load_bytes_to_tensor(
            2,
            2,
            PixelFormat::Rgba,
            None,
            &[0, 0, 0, 255, 1, 1, 1, 255, 2, 2, 2, 255, 3, 3, 3, 255],
        )?;

        let mut converter = CPUProcessor::default();

        let converted = TensorDyn::image(
            4,
            4,
            PixelFormat::Rgba,
            DType::U8,
            mem(),
            edgefirst_tensor::CpuAccess::ReadWrite,
        )?;
        let src_dyn = src;
        let mut converted_dyn = converted;

        converter.convert(
            &src_dyn,
            &mut converted_dyn,
            Some(crate::ConvertOptions::new().flip(Flip::Vertical)),
        )?;

        let map = converted_dyn.as_u8().unwrap().map()?;
        assert_eq!(&map.as_slice()[0..4], &[2, 2, 2, 255]);
        assert_eq!(&map.as_slice()[12..16], &[3, 3, 3, 255]);
        assert_eq!(&map.as_slice()[48..52], &[0, 0, 0, 255]);
        assert_eq!(&map.as_slice()[60..64], &[1, 1, 1, 255]);

        Ok(())
    }

    #[test]
    fn test_cpu_flip_h() -> Result<()> {
        let src = load_bytes_to_tensor(
            2,
            2,
            PixelFormat::Rgba,
            None,
            &[0, 0, 0, 255, 1, 1, 1, 255, 2, 2, 2, 255, 3, 3, 3, 255],
        )?;

        let mut converter = CPUProcessor::default();

        let converted = TensorDyn::image(
            4,
            4,
            PixelFormat::Rgba,
            DType::U8,
            mem(),
            edgefirst_tensor::CpuAccess::ReadWrite,
        )?;
        let src_dyn = src;
        let mut converted_dyn = converted;

        converter.convert(
            &src_dyn,
            &mut converted_dyn,
            Some(crate::ConvertOptions::new().flip(Flip::Horizontal)),
        )?;

        let map = converted_dyn.as_u8().unwrap().map()?;
        assert_eq!(&map.as_slice()[0..4], &[1, 1, 1, 255]);
        assert_eq!(&map.as_slice()[12..16], &[0, 0, 0, 255]);
        assert_eq!(&map.as_slice()[48..52], &[3, 3, 3, 255]);
        assert_eq!(&map.as_slice()[60..64], &[2, 2, 2, 255]);

        Ok(())
    }

    #[test]
    fn test_cpu_src_crop() -> Result<()> {
        let src = load_bytes_to_tensor(2, 2, PixelFormat::Grey, None, &[10, 20, 30, 40])?;

        let mut converter = CPUProcessor::default();

        let converted = TensorDyn::image(
            2,
            2,
            PixelFormat::Rgba,
            DType::U8,
            mem(),
            edgefirst_tensor::CpuAccess::ReadWrite,
        )?;
        let src_dyn = src;
        let mut converted_dyn = converted;

        converter.convert(
            &src_dyn,
            &mut converted_dyn,
            Some(
                crate::ConvertOptions::new()
                    .crop(Crop::new().with_source(Some(Region::new(0, 0, 1, 2)))),
            ),
        )?;

        assert_eq!(
            converted_dyn.as_u8().unwrap().map()?.as_slice(),
            &[10, 10, 10, 255, 13, 13, 13, 255, 30, 30, 30, 255, 33, 33, 33, 255]
        );
        Ok(())
    }

    #[test]
    fn test_cpu_dst_crop() -> Result<()> {
        let src = load_bytes_to_tensor(2, 2, PixelFormat::Grey, None, &[2, 4, 6, 8])?;

        let mut converter = CPUProcessor::default();

        let converted = load_bytes_to_tensor(
            2,
            2,
            PixelFormat::Yuyv,
            None,
            &[200, 128, 200, 128, 200, 128, 200, 128],
        )?;
        let src_dyn = src;
        let converted_dyn = converted;

        // Renders into the top-row sub-region view; the bottom row keeps its pre-fill.
        let mut dst_view = converted_dyn.view(Region::new(0, 0, 2, 1))?;
        converter.convert(&src_dyn, &mut dst_view, None)?;

        // The untagged Grey source resolves to BT.601 limited range: the
        // row-averaged grey [4, 6] yields Y = 19, 21.
        assert_eq!(
            converted_dyn.as_u8().unwrap().map()?.as_slice(),
            &[19, 128, 21, 128, 200, 128, 200, 128]
        );
        Ok(())
    }

    #[test]
    fn test_cpu_fill_rgba() -> Result<()> {
        let src = load_bytes_to_tensor(1, 1, PixelFormat::Rgba, None, &[3, 3, 3, 255])?;

        let mut converter = CPUProcessor::default();

        let converted = TensorDyn::image(
            2,
            2,
            PixelFormat::Rgba,
            DType::U8,
            mem(),
            edgefirst_tensor::CpuAccess::ReadWrite,
        )?;
        let src_dyn = src;
        let mut converted_dyn = converted;

        // Fills the whole dst with the pad colour, then renders the source into
        // the sub-region view.
        let pad = load_bytes_to_tensor(1, 1, PixelFormat::Rgba, None, &[255, 0, 0, 255])?;
        converter.convert(&pad, &mut converted_dyn, None)?;
        let mut dst_view = converted_dyn.view(Region::new(1, 1, 1, 1))?;
        converter.convert(&src_dyn, &mut dst_view, None)?;

        assert_eq!(
            converted_dyn.as_u8().unwrap().map()?.as_slice(),
            &[255, 0, 0, 255, 255, 0, 0, 255, 255, 0, 0, 255, 3, 3, 3, 255]
        );
        Ok(())
    }

    #[test]
    fn test_cpu_fill_yuyv() -> Result<()> {
        let src =
            load_bytes_to_tensor(2, 1, PixelFormat::Rgba, None, &[3, 3, 3, 255, 3, 3, 3, 255])?;

        let mut converter = CPUProcessor::default();

        let converted = TensorDyn::image(
            2,
            3,
            PixelFormat::Yuyv,
            DType::U8,
            mem(),
            edgefirst_tensor::CpuAccess::ReadWrite,
        )?;
        let src_dyn = src;
        let mut converted_dyn = converted;

        let pad = load_bytes_to_tensor(1, 1, PixelFormat::Rgba, None, &[255, 0, 0, 255])?;
        converter.convert(&pad, &mut converted_dyn, None)?;
        let mut dst_view = converted_dyn.view(Region::new(0, 1, 2, 1))?;
        converter.convert(&src_dyn, &mut dst_view, None)?;

        // Red (255,0,0) is Y=81, U=90, V=240 under BT.601 limited.
        assert_eq!(
            converted_dyn.as_u8().unwrap().map()?.as_slice(),
            &[81, 90, 81, 240, 19, 128, 19, 128, 81, 90, 81, 240]
        );
        Ok(())
    }

    #[test]
    fn test_cpu_fill_grey() -> Result<()> {
        let src =
            load_bytes_to_tensor(2, 1, PixelFormat::Rgba, None, &[3, 3, 3, 255, 3, 3, 3, 255])?;

        let mut converter = CPUProcessor::default();

        let converted = TensorDyn::image(
            2,
            3,
            PixelFormat::Grey,
            DType::U8,
            mem(),
            edgefirst_tensor::CpuAccess::ReadWrite,
        )?;
        let src_dyn = src;
        let mut converted_dyn = converted;

        let pad = load_bytes_to_tensor(1, 1, PixelFormat::Rgba, None, &[200, 200, 200, 255])?;
        converter.convert(&pad, &mut converted_dyn, None)?;
        let mut dst_view = converted_dyn.view(Region::new(0, 1, 2, 1))?;
        converter.convert(&src_dyn, &mut dst_view, None)?;

        assert_eq!(
            converted_dyn.as_u8().unwrap().map()?.as_slice(),
            &[200, 200, 3, 3, 200, 200]
        );
        Ok(())
    }

    #[test]
    fn test_segmentation() {
        use edgefirst_tensor::Segmentation;
        use ndarray::Array3;

        let image = crate::load_image_test_helper(
            &edgefirst_bench::testdata::read("giraffe.jpg"),
            Some(PixelFormat::Rgba),
            None,
        )
        .unwrap();
        let mut image_dyn = image;

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

        let mut renderer = CPUProcessor::new();
        renderer
            .draw_decoded_masks(&mut image_dyn, &[], &[seg], Default::default())
            .unwrap();

        let image = {
            let mut __t = image_dyn.into_u8().unwrap();
            __t.set_format(PixelFormat::Rgba).unwrap();
            TensorDyn::from(__t)
        };
        crate::save_jpeg(&image, "test_segmentation.jpg", 80).unwrap();
    }

    #[test]
    fn test_segmentation_yolo() {
        use edgefirst_tensor::Segmentation;
        use ndarray::Array3;

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

        let mut renderer = CPUProcessor::new();
        renderer
            .set_class_colors(&[[255, 255, 0, 233], [128, 128, 255, 100]])
            .unwrap();
        assert_eq!(renderer.colors[1], [128, 128, 255, 100]);
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
            &edgefirst_bench::testdata::read("output_render_cpu.jpg"),
            Some(PixelFormat::Rgba),
            None,
        )
        .unwrap();
        compare_images_convert_to_rgb(&image, &expected, 0.97, function!());
    }

    // =========================================================================
    // Generic Conversion Tests
    // (These tests use TensorDyn for all image representations)
    // =========================================================================

    #[test]
    fn test_convert_rgb_to_planar_rgb_generic() {
        let src = TensorDyn::image(
            4,
            4,
            PixelFormat::Rgb,
            DType::U8,
            mem(),
            edgefirst_tensor::CpuAccess::ReadWrite,
        )
        .unwrap();
        {
            let mut map = src.as_u8().unwrap().map().unwrap();
            let data = map.as_mut_slice();
            for i in 0..16 {
                data[i * 3] = (i * 10) as u8;
                data[i * 3 + 1] = (i * 10 + 1) as u8;
                data[i * 3 + 2] = (i * 10 + 2) as u8;
            }
        }

        let mut dst = TensorDyn::image(
            4,
            4,
            PixelFormat::PlanarRgb,
            DType::U8,
            mem(),
            edgefirst_tensor::CpuAccess::ReadWrite,
        )
        .unwrap();

        {
            let mut __cv = CPUProcessor::default();
            __cv.convert(&src, &mut dst, None).unwrap();
        }

        let map = dst.as_u8().unwrap().map().unwrap();
        let data = map.as_slice();

        // R plane starts at 0, G at 16, B at 32
        assert_eq!(data[0], 0); // R of pixel 0
        assert_eq!(data[16], 1); // G of pixel 0
        assert_eq!(data[32], 2); // B of pixel 0

        assert_eq!(data[1], 10); // R of pixel 1
        assert_eq!(data[17], 11); // G of pixel 1
        assert_eq!(data[33], 12); // B of pixel 1
    }

    #[test]
    fn test_convert_rgba_to_planar_rgb_generic() {
        let src = TensorDyn::image(
            4,
            4,
            PixelFormat::Rgba,
            DType::U8,
            mem(),
            edgefirst_tensor::CpuAccess::ReadWrite,
        )
        .unwrap();
        {
            let mut map = src.as_u8().unwrap().map().unwrap();
            let data = map.as_mut_slice();
            for i in 0..16 {
                data[i * 4] = (i * 10) as u8; // R
                data[i * 4 + 1] = (i * 10 + 1) as u8; // G
                data[i * 4 + 2] = (i * 10 + 2) as u8; // B
                data[i * 4 + 3] = 255; // A (ignored)
            }
        }

        let mut dst = TensorDyn::image(
            4,
            4,
            PixelFormat::PlanarRgb,
            DType::U8,
            mem(),
            edgefirst_tensor::CpuAccess::ReadWrite,
        )
        .unwrap();

        {
            let mut __cv = CPUProcessor::default();
            __cv.convert(&src, &mut dst, None).unwrap();
        }

        let map = dst.as_u8().unwrap().map().unwrap();
        let data = map.as_slice();

        assert_eq!(data[0], 0); // R of pixel 0
        assert_eq!(data[16], 1); // G of pixel 0
        assert_eq!(data[32], 2); // B of pixel 0
    }

    #[test]
    fn test_copy_image_generic_same_format() {
        let src = TensorDyn::image(
            4,
            4,
            PixelFormat::Rgb,
            DType::U8,
            mem(),
            edgefirst_tensor::CpuAccess::ReadWrite,
        )
        .unwrap();
        {
            let mut map = src.as_u8().unwrap().map().unwrap();
            let data = map.as_mut_slice();
            for (i, byte) in data.iter_mut().enumerate() {
                *byte = (i % 256) as u8;
            }
        }

        let mut dst = TensorDyn::image(
            4,
            4,
            PixelFormat::Rgb,
            DType::U8,
            mem(),
            edgefirst_tensor::CpuAccess::ReadWrite,
        )
        .unwrap();

        {
            let mut __cv = CPUProcessor::default();
            __cv.convert(&src, &mut dst, None).unwrap();
        }

        let src_map = src.as_u8().unwrap().map().unwrap();
        let dst_map = dst.as_u8().unwrap().map().unwrap();
        assert_eq!(src_map.as_slice(), dst_map.as_slice());
    }

    #[test]
    fn test_convert_unsupported_format_pair() {
        // NV16 → VYUY is not supported by the CPU converter.
        let src = TensorDyn::image(
            8,
            8,
            PixelFormat::Nv16,
            DType::U8,
            mem(),
            edgefirst_tensor::CpuAccess::ReadWrite,
        )
        .unwrap();
        let mut dst = TensorDyn::image(
            8,
            8,
            PixelFormat::Vyuy,
            DType::U8,
            mem(),
            edgefirst_tensor::CpuAccess::ReadWrite,
        )
        .unwrap();

        let result = {
            let mut __cv = CPUProcessor::default();
            __cv.convert(&src, &mut dst, None)
        };
        assert!(result.is_err());
        assert!(matches!(result, Err(Error::NotSupported(_))));
    }

    #[test]
    fn test_fill_image_outside_crop_generic_rgba() {
        let mut dst = TensorDyn::image(
            4,
            4,
            PixelFormat::Rgba,
            DType::U8,
            mem(),
            edgefirst_tensor::CpuAccess::ReadWrite,
        )
        .unwrap();
        dst.as_u8().unwrap().map().unwrap().as_mut_slice().fill(0);

        // Fill outside a 2x2 crop in the center with red
        let crop = Rect::new(1, 1, 2, 2);
        CPUProcessor::fill_image_outside_crop_u8(
            dst.as_u8_mut().unwrap(),
            [255, 0, 0, 255],
            crop,
            false,
        )
        .unwrap();

        let map = dst.as_u8().unwrap().map().unwrap();
        let data = map.as_slice();

        assert_eq!(&data[0..4], &[255, 0, 0, 255]);

        // Pixel (1,1): offset = (1*4 + 1) * 4 = 20.
        let center_offset = 20;
        assert_eq!(&data[center_offset..center_offset + 4], &[0, 0, 0, 0]);
    }

    #[test]
    fn test_fill_image_outside_crop_generic_rgb() {
        let mut dst = TensorDyn::image(
            4,
            4,
            PixelFormat::Rgb,
            DType::U8,
            mem(),
            edgefirst_tensor::CpuAccess::ReadWrite,
        )
        .unwrap();
        dst.as_u8().unwrap().map().unwrap().as_mut_slice().fill(0);

        let crop = Rect::new(1, 1, 2, 2);
        CPUProcessor::fill_image_outside_crop_u8(
            dst.as_u8_mut().unwrap(),
            [0, 255, 0, 255],
            crop,
            false,
        )
        .unwrap();

        let map = dst.as_u8().unwrap().map().unwrap();
        let data = map.as_slice();

        assert_eq!(&data[0..3], &[0, 255, 0]);

        // Pixel (1,1): offset = (1*4 + 1) * 3 = 15.
        let center_offset = 15;
        assert_eq!(&data[center_offset..center_offset + 3], &[0, 0, 0]);
    }

    #[test]
    fn test_fill_image_outside_crop_generic_planar_rgb() {
        let mut dst = TensorDyn::image(
            4,
            4,
            PixelFormat::PlanarRgb,
            DType::U8,
            mem(),
            edgefirst_tensor::CpuAccess::ReadWrite,
        )
        .unwrap();
        dst.as_u8().unwrap().map().unwrap().as_mut_slice().fill(0);

        let crop = Rect::new(1, 1, 2, 2);
        CPUProcessor::fill_image_outside_crop_u8(
            dst.as_u8_mut().unwrap(),
            [128, 64, 32, 255],
            crop,
            false,
        )
        .unwrap();

        let map = dst.as_u8().unwrap().map().unwrap();
        let data = map.as_slice();

        // Planar: R plane is [0..16], G plane is [16..32], B plane is [32..48].
        assert_eq!(data[0], 128); // R plane, pixel 0
        assert_eq!(data[16], 64); // G plane, pixel 0
        assert_eq!(data[32], 32); // B plane, pixel 0

        // Pixel (1,1): index = 1*4 + 1 = 5.
        let center_idx = 5;
        assert_eq!(data[center_idx], 0); // R
        assert_eq!(data[16 + center_idx], 0); // G
        assert_eq!(data[32 + center_idx], 0); // B
    }

    /// The NV16 letterbox fill splits luma from chroma at `row_stride * height` and advances both planes by the stride.
    #[test]
    fn test_fill_image_outside_crop_nv16_strided() {
        const SENTINEL: u8 = 0x11;
        let (w, h) = (96usize, 8usize);

        let mut dst = TensorDyn::image(
            w,
            h,
            PixelFormat::Nv16,
            DType::U8,
            mem(),
            edgefirst_tensor::CpuAccess::ReadWrite,
        )
        .unwrap();

        let stride = dst.effective_row_stride().unwrap();
        assert!(
            stride > w,
            "test needs a padded Nv16 destination to be meaningful (stride={stride}, w={w})"
        );
        {
            let t = dst.as_u8_mut().unwrap();
            let mut map = t.map_mut().unwrap();
            assert_eq!(
                map.as_slice().len(),
                stride * h * 2,
                "Nv16 maps luma + full-height chroma at the padded pitch"
            );
            map.as_mut_slice().fill(SENTINEL);
        }

        // The fill colour as `fill_image_outside_crop_u8` resolves it.
        let rgba = [255u8, 0, 0, 255];
        let (y_fill, u_fill, v_fill) = {
            let t = dst.as_u8().unwrap();
            let cm = crate::colorimetry::resolve_colorimetry(t.colorimetry(), t.height());
            let cp = crate::cpu::ColorParams::uniform(cm.encoding.unwrap(), cm.range.unwrap());
            let yuyv = CPUProcessor::rgba_to_yuyv(rgba, cp);
            (yuyv[0], yuyv[1], yuyv[3])
        };
        assert_ne!(y_fill, SENTINEL, "fill luma must differ from the sentinel");

        // Content occupies columns 32..64 of rows 2..6; everything else is border.
        let crop = Rect::new(32, 2, 32, 4);
        CPUProcessor::fill_image_outside_crop_u8(dst.as_u8_mut().unwrap(), rgba, crop, false)
            .unwrap();

        let t = dst.as_u8().unwrap();
        let map = t.map().unwrap();
        let data = map.as_slice();
        let (luma, chroma) = data.split_at(stride * h);

        for y in 0..h {
            let border_row = y < crop.top || y >= crop.top + crop.height;
            for x in 0..w {
                let inside = !border_row && x >= crop.left && x < crop.left + crop.width;
                let want = if inside { SENTINEL } else { y_fill };
                assert_eq!(
                    luma[y * stride + x],
                    want,
                    "luma ({x},{y}): expected {want:#04x} (inside_crop={inside})"
                );
            }
            // Row padding past the logical width is not filled.
            for x in w..stride {
                assert_eq!(
                    luma[y * stride + x],
                    SENTINEL,
                    "luma row {y} padding byte {x} was filled"
                );
            }
        }

        // Nv16 chroma is full height with one (Cb,Cr) pair per two luma columns,
        // at the same pitch.
        for y in 0..h {
            let border_row = y < crop.top || y >= crop.top + crop.height;
            for cx in 0..w / 2 {
                let inside =
                    !border_row && cx >= crop.left / 2 && cx < (crop.left + crop.width) / 2;
                let (want_u, want_v) = if inside {
                    (SENTINEL, SENTINEL)
                } else {
                    (u_fill, v_fill)
                };
                let off = y * stride + cx * 2;
                assert_eq!(
                    (chroma[off], chroma[off + 1]),
                    (want_u, want_v),
                    "chroma pair {cx} of row {y} (inside_crop={inside})"
                );
            }
            for x in w..stride {
                assert_eq!(
                    chroma[y * stride + x],
                    SENTINEL,
                    "chroma row {y} padding byte {x} was filled"
                );
            }
        }
    }

    /// A chroma pair straddling a crop edge keeps its in-crop value.
    #[test]
    fn test_fill_image_outside_crop_nv16_odd_crop_edges() {
        const SENTINEL: u8 = 0x11;
        let (w, h) = (16usize, 4usize);
        for crop in [Rect::new(3, 1, 5, 2), Rect::new(3, 1, 6, 2)] {
            let mut dst = TensorDyn::image(
                w,
                h,
                PixelFormat::Nv16,
                DType::U8,
                mem(),
                edgefirst_tensor::CpuAccess::ReadWrite,
            )
            .unwrap();
            let stride = dst.effective_row_stride().unwrap();
            dst.as_u8_mut()
                .unwrap()
                .map_mut()
                .unwrap()
                .as_mut_slice()
                .fill(SENTINEL);

            CPUProcessor::fill_image_outside_crop_u8(
                dst.as_u8_mut().unwrap(),
                [255, 0, 0, 255],
                crop,
                false,
            )
            .unwrap();

            let map = dst.as_u8().unwrap().map().unwrap();
            let (_, chroma) = map.as_slice().split_at(stride * h);
            let first = crop.left / 2;
            let end = (crop.left + crop.width).div_ceil(2);
            for y in 0..h {
                let in_rows = y >= crop.top && y < crop.top + crop.height;
                for cx in 0..w / 2 {
                    let kept = in_rows && cx >= first && cx < end;
                    let off = y * stride + cx * 2;
                    let pair = (chroma[off], chroma[off + 1]);
                    if kept {
                        assert_eq!(
                            pair,
                            (SENTINEL, SENTINEL),
                            "pair {cx} of row {y} for {crop:?}"
                        );
                    } else {
                        assert_ne!(
                            pair,
                            (SENTINEL, SENTINEL),
                            "pair {cx} of row {y} for {crop:?}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn test_fill_image_outside_crop_yuyv_odd_width() {
        // An odd-width Yuyv fill covers the trailing unpaired pixel's `[Y, U]` bytes.
        const SENTINEL: u8 = 0xAB;
        let (w, h) = (5usize, 4usize);

        let mut dst = TensorDyn::image(
            w,
            h,
            PixelFormat::Yuyv,
            DType::U8,
            mem(),
            edgefirst_tensor::CpuAccess::ReadWrite,
        )
        .unwrap();
        let stride = dst.effective_row_stride().unwrap();
        dst.as_u8()
            .unwrap()
            .map()
            .unwrap()
            .as_mut_slice()
            .fill(SENTINEL);

        // The fill colour as `fill_image_outside_crop_u8` resolves it.
        let rgba = [255u8, 0, 0, 255];
        let yuyv = {
            let t = dst.as_u8().unwrap();
            let cm = crate::colorimetry::resolve_colorimetry(t.colorimetry(), t.height());
            let cp = crate::cpu::ColorParams::uniform(cm.encoding.unwrap(), cm.range.unwrap());
            CPUProcessor::rgba_to_yuyv(rgba, cp)
        };

        // Crop covers rows 1..3, columns 0..4 — the trailing pixel (x = 4)
        // is outside the crop on every row.
        let crop = Rect::new(0, 1, 4, 2);
        CPUProcessor::fill_image_outside_crop_u8(dst.as_u8_mut().unwrap(), rgba, crop, false)
            .unwrap();

        let t = dst.as_u8().unwrap();
        let map = t.map().unwrap();
        let data = map.as_slice();
        for y in 0..h {
            // The trailing unpaired pixel's [Y, U].
            let tail = y * stride + (w - 1) * 2;
            assert_eq!(
                (data[tail], data[tail + 1]),
                (yuyv[0], yuyv[1]),
                "row {y}: trailing odd-width pixel's [Y, U] must be filled"
            );
            // The crop interior is untouched.
            let border_row = y < crop.top || y >= crop.top + crop.height;
            if !border_row {
                let inside = y * stride;
                assert_eq!(
                    (data[inside], data[inside + 1]),
                    (SENTINEL, SENTINEL),
                    "row {y}: crop-interior bytes were overwritten"
                );
            }
        }
    }

    #[test]
    fn test_convert_rgba_to_bgra() {
        use edgefirst_tensor::TensorMemory;
        // 2x1 image: pixel0 = [R=10, G=20, B=30, A=255], pixel1 = [R=40, G=50, B=60, A=128]
        let src = TensorDyn::image(
            2,
            1,
            PixelFormat::Rgba,
            DType::U8,
            Some(TensorMemory::Mem),
            edgefirst_tensor::CpuAccess::ReadWrite,
        )
        .unwrap();
        {
            let mut map = src.as_u8().unwrap().map().unwrap();
            let buf = map.as_mut_slice();
            buf[0..4].copy_from_slice(&[10, 20, 30, 255]);
            buf[4..8].copy_from_slice(&[40, 50, 60, 128]);
        }
        let mut dst = TensorDyn::image(
            2,
            1,
            PixelFormat::Bgra,
            DType::U8,
            Some(TensorMemory::Mem),
            edgefirst_tensor::CpuAccess::ReadWrite,
        )
        .unwrap();
        {
            let mut __cv = CPUProcessor::default();
            __cv.convert(&src, &mut dst, None).unwrap();
        }
        let map = dst.as_u8().unwrap().map().unwrap();
        let buf = map.as_slice();
        // PixelFormat::Bgra byte order: [B, G, R, A]
        assert_eq!(&buf[0..4], &[30, 20, 10, 255]);
        assert_eq!(&buf[4..8], &[60, 50, 40, 128]);
    }

    #[test]
    fn test_convert_rgb_to_bgra() {
        // Converts Rgb to Rgba and to Bgra and compares them with R and B swapped.
        let src = TensorDyn::image(
            2,
            1,
            PixelFormat::Rgb,
            DType::U8,
            Some(TensorMemory::Mem),
            edgefirst_tensor::CpuAccess::ReadWrite,
        )
        .unwrap();
        {
            let mut map = src.as_u8().unwrap().map().unwrap();
            let buf = map.as_mut_slice();
            buf[0..3].copy_from_slice(&[100, 150, 200]);
            buf[3..6].copy_from_slice(&[50, 75, 25]);
        }
        let mut rgba_dst = TensorDyn::image(
            2,
            1,
            PixelFormat::Rgba,
            DType::U8,
            Some(TensorMemory::Mem),
            edgefirst_tensor::CpuAccess::ReadWrite,
        )
        .unwrap();
        {
            let mut __cv = CPUProcessor::default();
            __cv.convert(&src, &mut rgba_dst, None).unwrap();
        }

        let mut bgra_dst = TensorDyn::image(
            2,
            1,
            PixelFormat::Bgra,
            DType::U8,
            Some(TensorMemory::Mem),
            edgefirst_tensor::CpuAccess::ReadWrite,
        )
        .unwrap();
        {
            let mut __cv = CPUProcessor::default();
            __cv.convert(&src, &mut bgra_dst, None).unwrap();
        }

        assert_bgra_matches_rgba(&bgra_dst, &rgba_dst);

        let map = bgra_dst.as_u8().unwrap().map().unwrap();
        let buf = map.as_slice();
        assert_eq!(buf[0], 200, "pixel 0 B");
        assert_eq!(buf[1], 150, "pixel 0 G");
        assert_eq!(buf[2], 100, "pixel 0 R");
        assert_eq!(buf[4], 25, "pixel 1 B");
        assert_eq!(buf[5], 75, "pixel 1 G");
        assert_eq!(buf[6], 50, "pixel 1 R");
    }

    #[test]
    fn test_convert_grey_to_bgra() {
        let src = TensorDyn::image(
            2,
            1,
            PixelFormat::Grey,
            DType::U8,
            Some(TensorMemory::Mem),
            edgefirst_tensor::CpuAccess::ReadWrite,
        )
        .unwrap();
        {
            let mut map = src.as_u8().unwrap().map().unwrap();
            let buf = map.as_mut_slice();
            buf[0] = 128;
            buf[1] = 64;
        }
        let mut dst = TensorDyn::image(
            2,
            1,
            PixelFormat::Bgra,
            DType::U8,
            Some(TensorMemory::Mem),
            edgefirst_tensor::CpuAccess::ReadWrite,
        )
        .unwrap();
        {
            let mut __cv = CPUProcessor::default();
            __cv.convert(&src, &mut dst, None).unwrap();
        }
        let map = dst.as_u8().unwrap().map().unwrap();
        let buf = map.as_slice();
        // Grey → Bgra: all channels the same value, A=255.
        assert_eq!(&buf[0..4], &[128, 128, 128, 255]);
        assert_eq!(&buf[4..8], &[64, 64, 64, 255]);
    }

    #[test]
    fn test_convert_bgra_to_bgra_copy() {
        // Bgra → Bgra is a straight copy.
        let src = TensorDyn::image(
            2,
            1,
            PixelFormat::Bgra,
            DType::U8,
            Some(TensorMemory::Mem),
            edgefirst_tensor::CpuAccess::ReadWrite,
        )
        .unwrap();
        {
            let mut map = src.as_u8().unwrap().map().unwrap();
            let buf = map.as_mut_slice();
            buf[0..4].copy_from_slice(&[10, 20, 30, 255]);
            buf[4..8].copy_from_slice(&[40, 50, 60, 128]);
        }
        let mut dst = TensorDyn::image(
            2,
            1,
            PixelFormat::Bgra,
            DType::U8,
            Some(TensorMemory::Mem),
            edgefirst_tensor::CpuAccess::ReadWrite,
        )
        .unwrap();
        {
            let mut __cv = CPUProcessor::default();
            __cv.convert(&src, &mut dst, None).unwrap();
        }
        let map = dst.as_u8().unwrap().map().unwrap();
        let buf = map.as_slice();
        assert_eq!(&buf[0..4], &[10, 20, 30, 255]);
        assert_eq!(&buf[4..8], &[40, 50, 60, 128]);
    }

    /// Asserts `bgra` equals `rgba` with R and B swapped.
    fn assert_bgra_matches_rgba(bgra: &TensorDyn, rgba: &TensorDyn) {
        assert_eq!(bgra.format().unwrap(), PixelFormat::Bgra);
        assert_eq!(rgba.format().unwrap(), PixelFormat::Rgba);
        assert_eq!(bgra.width(), rgba.width());
        assert_eq!(bgra.height(), rgba.height());

        let bgra_map = bgra.as_u8().unwrap().map().unwrap();
        let rgba_map = rgba.as_u8().unwrap().map().unwrap();
        let bgra_buf = bgra_map.as_slice();
        let rgba_buf = rgba_map.as_slice();

        assert_eq!(bgra_buf.len(), rgba_buf.len());
        for (i, (bc, rc)) in bgra_buf
            .as_chunks::<4>()
            .0
            .iter()
            .zip(rgba_buf.as_chunks::<4>().0)
            .enumerate()
        {
            assert_eq!(bc[0], rc[2], "pixel {i}: B(bgra) != B(rgba)");
            assert_eq!(bc[1], rc[1], "pixel {i}: G mismatch");
            assert_eq!(bc[2], rc[0], "pixel {i}: R(bgra) != R(rgba)");
            assert_eq!(bc[3], rc[3], "pixel {i}: A mismatch");
        }
    }

    #[test]
    fn test_convert_nv12_to_bgra() {
        let src = load_bytes_to_tensor(
            1280,
            720,
            PixelFormat::Nv12,
            None,
            &edgefirst_bench::testdata::read("camera720p.nv12"),
        )
        .unwrap();

        let mut rgba_dst = TensorDyn::image(
            1280,
            720,
            PixelFormat::Rgba,
            DType::U8,
            mem(),
            edgefirst_tensor::CpuAccess::ReadWrite,
        )
        .unwrap();
        {
            let mut __cv = CPUProcessor::default();
            __cv.convert(&src, &mut rgba_dst, None).unwrap();
        }

        let mut bgra_dst = TensorDyn::image(
            1280,
            720,
            PixelFormat::Bgra,
            DType::U8,
            mem(),
            edgefirst_tensor::CpuAccess::ReadWrite,
        )
        .unwrap();
        {
            let mut __cv = CPUProcessor::default();
            __cv.convert(&src, &mut bgra_dst, None).unwrap();
        }

        assert_bgra_matches_rgba(&bgra_dst, &rgba_dst);
    }

    #[test]
    fn test_convert_yuyv_to_bgra() {
        let src = load_bytes_to_tensor(
            1280,
            720,
            PixelFormat::Yuyv,
            None,
            &edgefirst_bench::testdata::read("camera720p.yuyv"),
        )
        .unwrap();

        let mut rgba_dst = TensorDyn::image(
            1280,
            720,
            PixelFormat::Rgba,
            DType::U8,
            mem(),
            edgefirst_tensor::CpuAccess::ReadWrite,
        )
        .unwrap();
        {
            let mut __cv = CPUProcessor::default();
            __cv.convert(&src, &mut rgba_dst, None).unwrap();
        }

        let mut bgra_dst = TensorDyn::image(
            1280,
            720,
            PixelFormat::Bgra,
            DType::U8,
            mem(),
            edgefirst_tensor::CpuAccess::ReadWrite,
        )
        .unwrap();
        {
            let mut __cv = CPUProcessor::default();
            __cv.convert(&src, &mut bgra_dst, None).unwrap();
        }

        assert_bgra_matches_rgba(&bgra_dst, &rgba_dst);
    }

    #[test]
    fn test_convert_vyuy_to_bgra() {
        let src = load_bytes_to_tensor(
            1280,
            720,
            PixelFormat::Vyuy,
            None,
            &edgefirst_bench::testdata::read("camera720p.vyuy"),
        )
        .unwrap();

        let mut rgba_dst = TensorDyn::image(
            1280,
            720,
            PixelFormat::Rgba,
            DType::U8,
            mem(),
            edgefirst_tensor::CpuAccess::ReadWrite,
        )
        .unwrap();
        {
            let mut __cv = CPUProcessor::default();
            __cv.convert(&src, &mut rgba_dst, None).unwrap();
        }

        let mut bgra_dst = TensorDyn::image(
            1280,
            720,
            PixelFormat::Bgra,
            DType::U8,
            mem(),
            edgefirst_tensor::CpuAccess::ReadWrite,
        )
        .unwrap();
        {
            let mut __cv = CPUProcessor::default();
            __cv.convert(&src, &mut bgra_dst, None).unwrap();
        }

        assert_bgra_matches_rgba(&bgra_dst, &rgba_dst);
    }

    #[test]
    fn test_convert_nv16_to_bgra() {
        let src = load_bytes_to_tensor(
            1280,
            720,
            PixelFormat::Nv16,
            None,
            &edgefirst_bench::testdata::read("camera720p.nv16"),
        )
        .unwrap();

        let mut rgba_dst = TensorDyn::image(
            1280,
            720,
            PixelFormat::Rgba,
            DType::U8,
            mem(),
            edgefirst_tensor::CpuAccess::ReadWrite,
        )
        .unwrap();
        {
            let mut __cv = CPUProcessor::default();
            __cv.convert(&src, &mut rgba_dst, None).unwrap();
        }

        let mut bgra_dst = TensorDyn::image(
            1280,
            720,
            PixelFormat::Bgra,
            DType::U8,
            mem(),
            edgefirst_tensor::CpuAccess::ReadWrite,
        )
        .unwrap();
        {
            let mut __cv = CPUProcessor::default();
            __cv.convert(&src, &mut bgra_dst, None).unwrap();
        }

        assert_bgra_matches_rgba(&bgra_dst, &rgba_dst);
    }

    // ========================================================================
    // Tests for materialize_segmentations
    // ========================================================================

    fn make_proto_data(
        proto_h: usize,
        proto_w: usize,
        num_protos: usize,
        coefficients: Vec<Vec<f32>>,
    ) -> crate::ProtoData {
        make_proto_data_with_values(
            proto_h,
            proto_w,
            num_protos,
            vec![0.0_f32; proto_h * proto_w * num_protos],
            coefficients,
        )
    }

    fn make_proto_data_with_values(
        proto_h: usize,
        proto_w: usize,
        num_protos: usize,
        proto_values: Vec<f32>,
        coefficients: Vec<Vec<f32>>,
    ) -> crate::ProtoData {
        use edgefirst_tensor::Tensor;
        assert_eq!(proto_values.len(), proto_h * proto_w * num_protos);
        let n = coefficients.len();
        let row_len = coefficients.first().map(|r| r.len()).unwrap_or(num_protos);
        let mut flat = Vec::with_capacity(n * row_len);
        for row in &coefficients {
            flat.extend_from_slice(row);
        }
        let coeff_t = Tensor::<f32>::from_slice(&flat, &[n, row_len]).unwrap();
        let protos_t =
            Tensor::<f32>::from_slice(&proto_values, &[proto_h, proto_w, num_protos]).unwrap();
        crate::ProtoData {
            mask_coefficients: coeff_t.into(),
            protos: protos_t.into(),
            layout: edgefirst_tensor::ProtoLayout::Nhwc,
        }
    }

    fn make_detect_box(xmin: f32, ymin: f32, xmax: f32, ymax: f32) -> crate::DetectBox {
        crate::DetectBox {
            bbox: edgefirst_tensor::BoundingBox {
                xmin,
                ymin,
                xmax,
                ymax,
            },
            score: 0.9,
            label: 0,
        }
    }

    #[test]
    fn test_materialize_empty_detections() {
        let cpu = CPUProcessor::new();
        let proto_data = make_proto_data(8, 8, 4, vec![vec![1.0; 4]]);
        let result = cpu.materialize_segmentations(&[], &proto_data, None);
        assert!(result.is_ok());
        assert!(result.unwrap().is_empty());
    }

    #[test]
    fn test_materialize_empty_proto_data() {
        // A `ProtoData` with an empty coefficient tensor and no detections
        // returns an empty segmentation list.
        let cpu = CPUProcessor::new();
        let proto_data = make_proto_data(8, 8, 4, vec![]);
        let result = cpu.materialize_segmentations(&[], &proto_data, None);
        assert!(result.is_ok());
        assert!(result.unwrap().is_empty());
    }

    #[test]
    fn test_materialize_single_detection() {
        let cpu = CPUProcessor::new();
        let proto_data = make_proto_data(8, 8, 4, vec![vec![0.5; 4]]);
        let det = [make_detect_box(0.1, 0.1, 0.5, 0.5)];
        let result = cpu.materialize_segmentations(&det, &proto_data, None);
        assert!(result.is_ok());
        let segs = result.unwrap();
        assert_eq!(segs.len(), 1);
        assert!(segs[0].segmentation.shape()[0] > 0);
        assert!(segs[0].segmentation.shape()[1] > 0);
        assert_eq!(segs[0].segmentation.shape()[2], 1);
    }

    #[test]
    fn test_materialize_bbox_edge_one() {
        let cpu = CPUProcessor::new();
        let proto_data = make_proto_data(8, 8, 4, vec![vec![0.5; 4]]);
        let det = [make_detect_box(0.5, 0.5, 1.0, 1.0)];
        let result = cpu.materialize_segmentations(&det, &proto_data, None);
        assert!(
            result.is_ok(),
            "bbox at exact boundary (1.0) should not panic"
        );
        let segs = result.unwrap();
        assert_eq!(segs.len(), 1);
    }

    #[test]
    fn test_materialize_bbox_negative_clamp() {
        let cpu = CPUProcessor::new();
        let proto_data = make_proto_data(8, 8, 4, vec![vec![0.5; 4]]);
        let det = [make_detect_box(-0.5, -0.5, 0.5, 0.5)];
        let result = cpu.materialize_segmentations(&det, &proto_data, None);
        assert!(
            result.is_ok(),
            "negative coordinates should be clamped to 0"
        );
        let segs = result.unwrap();
        assert_eq!(segs.len(), 1);
        assert!((segs[0].xmin - 0.0).abs() < 0.01);
        assert!((segs[0].ymin - 0.0).abs() < 0.01);
    }

    #[test]
    fn test_materialize_invalid_coeff_shape() {
        let cpu = CPUProcessor::new();
        // 4 proto channels against 6 coefficients.
        let proto_data = make_proto_data(8, 8, 4, vec![vec![0.5; 6]]);
        let det = [make_detect_box(0.1, 0.1, 0.5, 0.5)];
        let result = cpu.materialize_segmentations(&det, &proto_data, None);
        assert!(
            result.is_err(),
            "mismatched coeff count vs proto channels should error"
        );
        let err = result.unwrap_err();
        assert!(
            matches!(&err, crate::Error::InvalidShape(s) if s.contains("mask_coefficients")),
            "error should mention coefficient shape: {err:?}"
        );
    }

    #[test]
    fn test_materialize_multiple_detections() {
        let cpu = CPUProcessor::new();
        let proto_data = make_proto_data(8, 8, 4, vec![vec![0.5; 4], vec![0.3; 4], vec![0.1; 4]]);
        let det = [
            make_detect_box(0.0, 0.0, 0.5, 0.5),
            make_detect_box(0.5, 0.0, 1.0, 0.5),
            make_detect_box(0.0, 0.5, 0.5, 1.0),
        ];
        let result = cpu.materialize_segmentations(&det, &proto_data, None);
        assert!(result.is_ok());
        assert_eq!(result.unwrap().len(), 3);
    }

    #[test]
    fn test_materialize_zero_area_bbox() {
        let cpu = CPUProcessor::new();
        let proto_data = make_proto_data(8, 8, 4, vec![vec![0.5; 4]]);
        // xmin == xmax → zero-width bbox
        let det = [make_detect_box(0.5, 0.1, 0.5, 0.5)];
        let result = cpu.materialize_segmentations(&det, &proto_data, None);
        assert!(
            result.is_ok(),
            "zero-area bbox should return Ok with degenerate segmentation"
        );
        let segs = result.unwrap();
        assert_eq!(segs.len(), 1);
    }

    #[test]
    fn test_materialize_scaled_invalid_coeff_rows() {
        let cpu = CPUProcessor::new();
        let proto_data = make_proto_data(8, 8, 4, vec![vec![0.5; 4]]);
        let det = [
            make_detect_box(0.1, 0.1, 0.5, 0.5),
            make_detect_box(0.5, 0.5, 0.9, 0.9),
        ];
        let result = cpu.materialize_scaled_segmentations(&det, &proto_data, None, 64, 64);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(
            matches!(&err, crate::Error::Internal(s) if s.contains("mask_coefficients rows")),
            "error should report coefficient rows vs detection count mismatch: {err:?}"
        );
    }

    #[test]
    fn test_materialize_scaled_bbox_edge_one() {
        let cpu = CPUProcessor::new();
        let proto_data = make_proto_data_with_values(8, 8, 1, vec![1.0_f32; 64], vec![vec![1.0]]);
        let det = [make_detect_box(0.5, 0.5, 1.0, 1.0)];
        let segs = cpu
            .materialize_scaled_segmentations(&det, &proto_data, None, 64, 64)
            .expect("bbox at exact boundary (1.0) should be handled on scaled path");
        assert_eq!(segs.len(), 1);
        let shape = segs[0].segmentation.shape();
        assert!(shape[0] > 0 && shape[1] > 0);
        assert_eq!(shape[2], 1);
        {
            use edgefirst_tensor::{TensorMapTrait as _, TensorTrait as _};
            let m = segs[0].segmentation.as_u8().unwrap().map_read().unwrap();
            assert!(
                m.as_slice().iter().all(|&v| matches!(v, 0_u8 | 255_u8)),
                "scaled path output must remain binarized u8"
            );
        }
    }

    #[test]
    fn test_materialize_scaled_letterbox_mapping() {
        let cpu = CPUProcessor::new();
        let proto_data = make_proto_data_with_values(8, 8, 1, vec![1.0_f32; 64], vec![vec![1.0]]);
        let det = [make_detect_box(0.25, 0.1, 0.75, 0.9)];
        let segs = cpu
            .materialize_scaled_segmentations(
                &det,
                &proto_data,
                Some([0.25, 0.1, 0.75, 0.9]),
                16,
                10,
            )
            .expect("scaled path with letterbox should succeed");
        assert_eq!(segs.len(), 1);
        let seg = &segs[0];
        assert!((seg.xmin - 0.0).abs() < 1e-6);
        assert!((seg.ymin - 0.0).abs() < 1e-6);
        assert!((seg.xmax - 1.0).abs() < 1e-6);
        assert!((seg.ymax - 1.0).abs() < 1e-6);
        let shape = seg.segmentation.shape();
        assert_eq!(shape[0], 10);
        assert_eq!(shape[1], 16);
        assert_eq!(shape[2], 1);
        assert!(
            seg_bytes(seg).iter().all(|&v| v == 255),
            "positive logits should remain foreground after scaled letterbox mapping"
        );
    }

    #[test]
    fn test_materialize_scaled_out_of_range_letterbox_no_panic() {
        let cpu = CPUProcessor::new();
        let proto_data = make_proto_data_with_values(8, 8, 1, vec![1.0_f32; 64], vec![vec![1.0]]);
        let det = [make_detect_box(0.95, 0.95, 1.0, 1.0)];
        // Out-of-range letterbox values put proto_x0/proto_y0 at the exclusive
        // upper bound.
        let segs = cpu
            .materialize_scaled_segmentations(&det, &proto_data, Some([1.2, 1.2, 1.4, 1.4]), 32, 32)
            .expect("scaled path should clamp ROI starts and avoid OOB panics");
        assert_eq!(segs.len(), 1);
        let shape = segs[0].segmentation.shape();
        assert!(shape[0] > 0 && shape[1] > 0);
        assert_eq!(shape[2], 1);
    }

    #[test]
    fn test_materialize_nchw_i8_produces_correct_mask() {
        // NCHW protos [K=2, H=4, W=4]: channel 0 is +10, channel 1 is -10;
        // coefficients [1, 0] select channel 0.
        use edgefirst_tensor::{Quantization, Tensor};
        let cpu = CPUProcessor::new();
        let (proto_h, proto_w, num_protos) = (4, 4, 2);
        let mut proto_values = vec![0_i8; num_protos * proto_h * proto_w];
        for i in 0..proto_h * proto_w {
            proto_values[i] = 10; // channel 0
            proto_values[proto_h * proto_w + i] = -10; // channel 1
        }
        let mut protos_t =
            Tensor::<i8>::from_slice(&proto_values, &[num_protos, proto_h, proto_w]).unwrap();
        protos_t
            .set_quantization(Quantization::per_tensor(0.1, 0))
            .unwrap();
        let coeff_values = vec![1_i8, 0_i8];
        let mut coeff_t = Tensor::<i8>::from_slice(&coeff_values, &[1, num_protos]).unwrap();
        coeff_t
            .set_quantization(Quantization::per_tensor(1.0, 0))
            .unwrap();
        let proto_data = crate::ProtoData {
            mask_coefficients: coeff_t.into(),
            protos: protos_t.into(),
            layout: edgefirst_tensor::ProtoLayout::Nchw,
        };
        let det = [make_detect_box(0.0, 0.0, 1.0, 1.0)];
        let segs = cpu
            .materialize_segmentations(&det, &proto_data, None)
            .expect("NCHW i8 proto layout should be handled");
        assert_eq!(segs.len(), 1);
        assert!(
            seg_bytes(&segs[0]).iter().all(|&v| v == 255),
            "NCHW proto with positive channel selected should yield all-255 mask"
        );
    }

    #[test]
    fn test_materialize_nchw_float_rejected() {
        // NCHW layout with non-i8 protos returns NotSupported.
        use edgefirst_tensor::Tensor;
        let cpu = CPUProcessor::new();
        let proto_values = vec![1.0_f32; 2 * 4 * 4];
        let protos_t = Tensor::<f32>::from_slice(&proto_values, &[2, 4, 4]).unwrap();
        let coeff_values = vec![1.0_f32; 2];
        let coeff_t = Tensor::<f32>::from_slice(&coeff_values, &[1, 2]).unwrap();
        let proto_data = crate::ProtoData {
            mask_coefficients: coeff_t.into(),
            protos: protos_t.into(),
            layout: edgefirst_tensor::ProtoLayout::Nchw,
        };
        let det = [make_detect_box(0.0, 0.0, 1.0, 1.0)];
        let result = cpu.materialize_segmentations(&det, &proto_data, None);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(
            matches!(&err, crate::Error::NotSupported(s) if s.contains("NCHW")),
            "NCHW with non-i8 protos should return NotSupported: {err:?}"
        );
    }

    #[test]
    fn test_materialize_scaled_nchw_float_rejected() {
        // NCHW layout with non-i8 protos on the scaled path returns an error.
        use edgefirst_tensor::Tensor;
        let cpu = CPUProcessor::new();
        let proto_values = vec![1.0_f32; 2 * 4 * 4];
        let protos_t = Tensor::<f32>::from_slice(&proto_values, &[2, 4, 4]).unwrap();
        let coeff_values = vec![1.0_f32; 2];
        let coeff_t = Tensor::<f32>::from_slice(&coeff_values, &[1, 2]).unwrap();
        let proto_data = crate::ProtoData {
            mask_coefficients: coeff_t.into(),
            protos: protos_t.into(),
            layout: edgefirst_tensor::ProtoLayout::Nchw,
        };
        let det = [make_detect_box(0.1, 0.1, 0.9, 0.9)];
        let result = cpu.materialize_scaled_segmentations(&det, &proto_data, None, 64, 64);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(
            matches!(&err, crate::Error::NotSupported(s) if s.contains("NCHW")),
            "NCHW with non-i8 protos should return NotSupported: {err:?}"
        );
    }

    #[test]
    fn test_materialize_scaled_nchw_i8_produces_correct_mask() {
        // NCHW i8 protos on the scaled path.
        use edgefirst_tensor::{Quantization, Tensor};
        let cpu = CPUProcessor::new();
        let (proto_h, proto_w, num_protos) = (4, 4, 2);
        let mut proto_values = vec![0_i8; num_protos * proto_h * proto_w];
        // Channel 0: all +10, Channel 1: all -10
        for i in 0..proto_h * proto_w {
            proto_values[i] = 10;
            proto_values[proto_h * proto_w + i] = -10;
        }
        let mut protos_t =
            Tensor::<i8>::from_slice(&proto_values, &[num_protos, proto_h, proto_w]).unwrap();
        protos_t
            .set_quantization(Quantization::per_tensor(0.1, 0))
            .unwrap();
        // Coefficient selects channel 0 only: [+1, 0]
        let coeff_values = vec![1_i8, 0_i8];
        let mut coeff_t = Tensor::<i8>::from_slice(&coeff_values, &[1, num_protos]).unwrap();
        coeff_t
            .set_quantization(Quantization::per_tensor(1.0, 0))
            .unwrap();
        let proto_data = crate::ProtoData {
            mask_coefficients: coeff_t.into(),
            protos: protos_t.into(),
            layout: edgefirst_tensor::ProtoLayout::Nchw,
        };
        let det = [make_detect_box(0.0, 0.0, 1.0, 1.0)];
        let segs = cpu
            .materialize_scaled_segmentations(&det, &proto_data, None, 16, 16)
            .expect("NCHW i8 scaled path should succeed");
        assert_eq!(segs.len(), 1);
        assert!(
            seg_bytes(&segs[0]).iter().all(|&v| v == 255),
            "NCHW i8 scaled path with positive channel should yield all-255 mask"
        );
    }

    #[test]
    fn test_materialize_nchw_i8_asymmetric_quantization() {
        // Asymmetric quantization: real_value = (raw - zero_point) * scale.
        // Proto: zp=5, scale=0.1; ch0 raw 15 → 1.0, ch1 raw 5 → 0.0.
        // Coeff: zp=10, scale=0.5; raw [12, 10] → [1.0, 0.0].
        // dot = 1.0 > 0 → mask = 255.
        use edgefirst_tensor::{Quantization, Tensor};
        let cpu = CPUProcessor::new();
        let (proto_h, proto_w, num_protos) = (4, 4, 2);

        let mut proto_values = vec![0_i8; num_protos * proto_h * proto_w];
        let proto_zp: i8 = 5;
        for i in 0..proto_h * proto_w {
            proto_values[i] = 15; // channel 0
            proto_values[proto_h * proto_w + i] = proto_zp; // channel 1 (at zero-point)
        }
        let mut protos_t =
            Tensor::<i8>::from_slice(&proto_values, &[num_protos, proto_h, proto_w]).unwrap();
        protos_t
            .set_quantization(Quantization::per_tensor(0.1, proto_zp as i32))
            .unwrap();

        let coeff_zp: i8 = 10;
        let coeff_values = vec![12_i8, coeff_zp];
        let mut coeff_t = Tensor::<i8>::from_slice(&coeff_values, &[1, num_protos]).unwrap();
        coeff_t
            .set_quantization(Quantization::per_tensor(0.5, coeff_zp as i32))
            .unwrap();

        let proto_data = crate::ProtoData {
            mask_coefficients: coeff_t.into(),
            protos: protos_t.into(),
            layout: edgefirst_tensor::ProtoLayout::Nchw,
        };
        let det = [make_detect_box(0.0, 0.0, 1.0, 1.0)];
        let segs = cpu
            .materialize_segmentations(&det, &proto_data, None)
            .expect("NCHW i8 asymmetric quantization should succeed");
        assert_eq!(segs.len(), 1);
        assert!(
            seg_bytes(&segs[0]).iter().all(|&v| v == 255),
            "NCHW i8 with non-zero zero-points: positive dot should yield all-255"
        );

        // Coeff raw [10, 12] → [0.0, 1.0] selects channel 1: dot = 0.0 → mask = 0.
        let coeff_neg = vec![coeff_zp, 12_i8]; // selects channel 1 which is zero
        let mut coeff_t_neg = Tensor::<i8>::from_slice(&coeff_neg, &[1, num_protos]).unwrap();
        coeff_t_neg
            .set_quantization(Quantization::per_tensor(0.5, coeff_zp as i32))
            .unwrap();

        let mut protos_t2 =
            Tensor::<i8>::from_slice(&proto_values, &[num_protos, proto_h, proto_w]).unwrap();
        protos_t2
            .set_quantization(Quantization::per_tensor(0.1, proto_zp as i32))
            .unwrap();

        let proto_data_neg = crate::ProtoData {
            mask_coefficients: coeff_t_neg.into(),
            protos: protos_t2.into(),
            layout: edgefirst_tensor::ProtoLayout::Nchw,
        };
        let segs_neg = cpu
            .materialize_segmentations(&det, &proto_data_neg, None)
            .expect("NCHW i8 asymmetric: zero-dot case");
        assert_eq!(segs_neg.len(), 1);
        assert!(
            seg_bytes(&segs_neg[0]).iter().all(|&v| v == 0),
            "NCHW i8 with non-zero zero-points: zero dot should yield all-0"
        );
    }

    #[test]
    fn test_materialize_per_channel_i8_falls_through_to_f32() {
        // I8 protos with per-channel quantization take the f32 dequant path.
        use edgefirst_tensor::{Quantization, Tensor};
        let cpu = CPUProcessor::new();
        let (proto_h, proto_w, num_protos) = (4, 4, 2);

        // NHWC protos: [H, W, K] = [4, 4, 2]; channel 0 is +10, channel 1 is -10.
        let mut proto_values = vec![0_i8; proto_h * proto_w * num_protos];
        for y in 0..proto_h {
            for x in 0..proto_w {
                let base = (y * proto_w + x) * num_protos;
                proto_values[base] = 10; // channel 0
                proto_values[base + 1] = -10; // channel 1
            }
        }
        let mut protos_t =
            Tensor::<i8>::from_slice(&proto_values, &[proto_h, proto_w, num_protos]).unwrap();
        // Per-channel symmetric quantization on axis 2 (channel axis).
        let scales = vec![0.1_f32; num_protos];
        protos_t
            .set_quantization(Quantization::per_channel_symmetric(scales, 2).unwrap())
            .unwrap();

        // Per-tensor coeff: [1, 0] → selects channel 0 only
        let coeff_values = vec![1_i8, 0_i8];
        let mut coeff_t = Tensor::<i8>::from_slice(&coeff_values, &[1, num_protos]).unwrap();
        coeff_t
            .set_quantization(Quantization::per_tensor(1.0, 0))
            .unwrap();

        let proto_data = crate::ProtoData {
            mask_coefficients: coeff_t.into(),
            protos: protos_t.into(),
            layout: edgefirst_tensor::ProtoLayout::Nhwc,
        };
        let det = [make_detect_box(0.0, 0.0, 1.0, 1.0)];
        let segs = cpu
            .materialize_segmentations(&det, &proto_data, None)
            .expect("per-channel I8 protos should fall through to f32 path");
        assert_eq!(segs.len(), 1);
        assert!(
            seg_bytes(&segs[0]).iter().all(|&v| v == 255),
            "per-channel i8 fallback: positive dot should yield all-255"
        );
    }

    /// The quantized decode and CPU mask materialization reproduce the golden mask bytes.
    #[test]
    fn test_materialize_golden_i8_cached_model() {
        use edgefirst_decoder::{configs, ConfigOutput, DecoderBuilder, Nms};
        use edgefirst_tensor::{DetectBox, ProtoData};
        use edgefirst_tensor::{Tensor, TensorDyn};

        let boxes_raw = edgefirst_bench::testdata::read("yolov8_boxes_116x8400.bin");
        let boxes_i8 =
            unsafe { std::slice::from_raw_parts(boxes_raw.as_ptr() as *const i8, boxes_raw.len()) };
        let boxes_tensor: TensorDyn = Tensor::<i8>::from_slice(boxes_i8, &[1, 116, 8400])
            .expect("boxes tensor")
            .into();

        let protos_raw = edgefirst_bench::testdata::read("yolov8_protos_160x160x32.bin");
        let protos_i8 = unsafe {
            std::slice::from_raw_parts(protos_raw.as_ptr() as *const i8, protos_raw.len())
        };
        let protos_tensor: TensorDyn = Tensor::<i8>::from_slice(protos_i8, &[1, 160, 160, 32])
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
            .expect("yolov8-seg golden decoder must build");

        let inputs: Vec<&TensorDyn> = vec![&boxes_tensor, &protos_tensor];
        let mut detections: Vec<DetectBox> = Vec::with_capacity(50);
        let proto_data: ProtoData = decoder
            .decode_proto(&inputs, &mut detections)
            .expect("decode_proto must succeed")
            .expect("yolov8-seg config produces ProtoData");
        assert!(!detections.is_empty(), "fixture produced no detections");

        let cpu = CPUProcessor::new();
        let segs = cpu
            .materialize_segmentations(&detections, &proto_data, None)
            .expect("materialize_segmentations failed on cached fixture");
        assert_eq!(segs.len(), detections.len());

        // Detection 0's mask as bytes:
        //   [u16 roi_h_le][u16 roi_w_le][roi_h * roi_w bytes of u8 mask data]
        let det0_seg = &segs[0].segmentation;
        let sh = det0_seg.shape();
        assert_eq!(sh.len(), 3, "segmentation must be rank-3");
        assert_eq!(sh[2], 1, "YOLO-seg mask has channel dim 1");
        let roi_h = sh[0];
        let roi_w = sh[1];
        assert!(roi_h <= u16::MAX as usize && roi_w <= u16::MAX as usize);
        let mut actual: Vec<u8> = Vec::with_capacity(4 + roi_h * roi_w);
        actual.extend_from_slice(&(roi_h as u16).to_le_bytes());
        actual.extend_from_slice(&(roi_w as u16).to_le_bytes());
        let det0_map = det0_seg.as_u8().expect("U8 mask").map_read().unwrap();
        actual.extend_from_slice(det0_map.as_slice());

        // Resolves testdata via `EDGEFIRST_TESTDATA_DIR` when set, else the source tree.
        let golden_path = std::env::var_os("EDGEFIRST_TESTDATA_DIR")
            .map(|d| std::path::PathBuf::from(d).join("yolov8_mask0_160x160.bin"))
            .unwrap_or_else(|| {
                std::path::PathBuf::from(concat!(
                    env!("CARGO_MANIFEST_DIR"),
                    "/../../testdata/yolov8_mask0_160x160.bin"
                ))
            });

        if std::env::var_os("REGEN_GOLDEN").is_some() {
            std::fs::write(&golden_path, &actual).expect("could not write golden");
            eprintln!(
                "REGEN_GOLDEN: wrote {} bytes (roi_h={roi_h}, roi_w={roi_w}) to {}",
                actual.len(),
                golden_path.display()
            );
            return;
        }

        let expected = std::fs::read(&golden_path).unwrap_or_else(|e| {
            panic!(
                "missing golden at {}: {e}. Run \
                 `REGEN_GOLDEN=1 cargo test test_materialize_golden_i8_cached_model` \
                 once to create it, then commit the file.",
                golden_path.display()
            )
        });
        assert_eq!(
            actual.len(),
            expected.len(),
            "golden size mismatch: got {} bytes, expected {}",
            actual.len(),
            expected.len()
        );
        if actual != expected {
            let mut first_diff = None;
            for (i, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
                if a != e {
                    first_diff = Some((i, *a, *e));
                    break;
                }
            }
            let total_diff = actual
                .iter()
                .zip(expected.iter())
                .filter(|(a, e)| a != e)
                .count();
            panic!(
                "golden mismatch — {total_diff}/{} bytes differ; first at {first_diff:?}. \
                 If the change is intentional, verify visually and re-run with REGEN_GOLDEN=1.",
                actual.len()
            );
        }
    }

    // ── i8×i8 integer mask decode path tests ─────────────────────────────────────

    /// Build a ProtoData with I8 protos and I8 coefficients (with quantization).
    #[allow(clippy::too_many_arguments)]
    fn make_proto_data_i8(
        proto_h: usize,
        proto_w: usize,
        num_protos: usize,
        proto_values: Vec<i8>,
        proto_quant: (f32, i32),
        coeff_values: Vec<i8>,
        coeff_quant: (f32, i32),
        num_detections: usize,
    ) -> crate::ProtoData {
        use edgefirst_tensor::{Quantization, Tensor};
        assert_eq!(proto_values.len(), proto_h * proto_w * num_protos);
        assert_eq!(coeff_values.len(), num_detections * num_protos);

        let mut protos_t =
            Tensor::<i8>::from_slice(&proto_values, &[proto_h, proto_w, num_protos]).unwrap();
        protos_t
            .set_quantization(Quantization::per_tensor(proto_quant.0, proto_quant.1))
            .unwrap();

        let mut coeff_t =
            Tensor::<i8>::from_slice(&coeff_values, &[num_detections, num_protos]).unwrap();
        coeff_t
            .set_quantization(Quantization::per_tensor(coeff_quant.0, coeff_quant.1))
            .unwrap();

        crate::ProtoData {
            mask_coefficients: coeff_t.into(),
            protos: protos_t.into(),
            layout: edgefirst_tensor::ProtoLayout::Nhwc,
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn make_proto_data_i16_i8(
        proto_h: usize,
        proto_w: usize,
        num_protos: usize,
        proto_values: Vec<i8>,
        proto_quant: (f32, i32),
        coeff_values: Vec<i16>,
        coeff_quant: (f32, i32),
        num_detections: usize,
    ) -> crate::ProtoData {
        use edgefirst_tensor::{Quantization, Tensor};
        assert_eq!(proto_values.len(), proto_h * proto_w * num_protos);
        assert_eq!(coeff_values.len(), num_detections * num_protos);

        let mut protos_t =
            Tensor::<i8>::from_slice(&proto_values, &[proto_h, proto_w, num_protos]).unwrap();
        protos_t
            .set_quantization(Quantization::per_tensor(proto_quant.0, proto_quant.1))
            .unwrap();

        let mut coeff_t =
            Tensor::<i16>::from_slice(&coeff_values, &[num_detections, num_protos]).unwrap();
        coeff_t
            .set_quantization(Quantization::per_tensor(coeff_quant.0, coeff_quant.1))
            .unwrap();

        crate::ProtoData {
            mask_coefficients: coeff_t.into(),
            protos: protos_t.into(),
            layout: edgefirst_tensor::ProtoLayout::Nhwc,
        }
    }

    #[test]
    fn test_materialize_scaled_i8_i8_basic() {
        // Uniform positive protos + positive coefficients → all-255 mask.
        let proto_h = 4;
        let proto_w = 4;
        let num_protos = 2;
        let proto_values = vec![10_i8; proto_h * proto_w * num_protos];
        let coeff_values = vec![10_i8; num_protos];
        let proto_data = make_proto_data_i8(
            proto_h,
            proto_w,
            num_protos,
            proto_values,
            (0.1, 0),
            coeff_values,
            (0.1, 0),
            1,
        );
        let det = [make_detect_box(0.1, 0.1, 0.9, 0.9)];
        let cpu = CPUProcessor::new();
        let segs = cpu
            .materialize_scaled_segmentations(&det, &proto_data, None, 16, 16)
            .expect("i8×i8 scaled path should succeed");
        assert_eq!(segs.len(), 1);
        assert_eq!(segs[0].segmentation.shape()[2], 1);
        assert!(
            seg_bytes(&segs[0]).iter().all(|&v| v == 255),
            "uniform positive protos+coeffs should produce all-255 mask"
        );
    }

    #[test]
    fn test_materialize_scaled_i8_i8_all_negative() {
        // Positive protos with negative coefficients → all-0 mask.
        let proto_h = 4;
        let proto_w = 4;
        let num_protos = 2;
        let proto_values = vec![10_i8; proto_h * proto_w * num_protos];
        let coeff_values = vec![-10_i8; num_protos];
        let proto_data = make_proto_data_i8(
            proto_h,
            proto_w,
            num_protos,
            proto_values,
            (0.1, 0),
            coeff_values,
            (0.1, 0),
            1,
        );
        let det = [make_detect_box(0.1, 0.1, 0.9, 0.9)];
        let cpu = CPUProcessor::new();
        let segs = cpu
            .materialize_scaled_segmentations(&det, &proto_data, None, 16, 16)
            .expect("i8×i8 scaled path should succeed");
        assert_eq!(segs.len(), 1);
        assert!(
            seg_bytes(&segs[0]).iter().all(|&v| v == 0),
            "positive protos + negative coeffs should produce all-0 mask"
        );
    }

    #[test]
    fn test_materialize_scaled_i8_i8_mixed_sign_boundary() {
        // Proto field: left half positive, right half negative.
        let proto_h = 4;
        let proto_w = 4;
        let num_protos = 1;
        let mut proto_values = Vec::with_capacity(proto_h * proto_w * num_protos);
        for _y in 0..proto_h {
            for x in 0..proto_w {
                proto_values.push(if x < proto_w / 2 { 50_i8 } else { -50_i8 });
            }
        }
        let coeff_values = vec![1_i8]; // identity coefficient
        let proto_data = make_proto_data_i8(
            proto_h,
            proto_w,
            num_protos,
            proto_values,
            (0.1, 0),
            coeff_values,
            (0.1, 0),
            1,
        );
        let det = [make_detect_box(0.0, 0.0, 1.0, 1.0)];
        let cpu = CPUProcessor::new();
        let segs = cpu
            .materialize_scaled_segmentations(&det, &proto_data, None, 16, 16)
            .expect("i8×i8 with mixed-sign protos should succeed");
        assert_eq!(segs.len(), 1);
        let mask = seg_bytes(&segs[0]);
        let has_255 = mask.contains(&255);
        let has_0 = mask.contains(&0);
        assert!(
            has_255 && has_0,
            "mixed-sign proto field should produce mixed mask"
        );
        assert!(
            mask.iter().all(|&v| v == 0 || v == 255),
            "mask must be binarized (0 or 255)"
        );
    }

    #[test]
    fn test_materialize_scaled_i8_i8_with_zero_points() {
        // Protos with zp=-5, coeffs with zp=3.
        let proto_h = 4;
        let proto_w = 4;
        let num_protos = 2;
        // Raw proto values are 10, dequantized = (10 - (-5)) * 0.1 = 1.5
        let proto_values = vec![10_i8; proto_h * proto_w * num_protos];
        // Raw coeff values are 10, dequantized = (10 - 3) * 0.1 = 0.7
        let coeff_values = vec![10_i8; num_protos];
        let proto_data = make_proto_data_i8(
            proto_h,
            proto_w,
            num_protos,
            proto_values,
            (0.1, -5),
            coeff_values,
            (0.1, 3),
            1,
        );
        let det = [make_detect_box(0.1, 0.1, 0.9, 0.9)];
        let cpu = CPUProcessor::new();
        let segs = cpu
            .materialize_scaled_segmentations(&det, &proto_data, None, 16, 16)
            .expect("i8×i8 with zero-points should succeed");
        assert_eq!(segs.len(), 1);
        assert!(
            seg_bytes(&segs[0]).iter().all(|&v| v == 255),
            "positive dequantized values should produce all-255 mask"
        );
    }

    #[test]
    fn test_materialize_scaled_i8_i8_parity_with_float() {
        // Compares the i8×i8 integer path with the f32 path under symmetric
        // quantization (zp=0).
        let proto_h = 8;
        let proto_w = 8;
        let num_protos = 4;

        let mut proto_i8 = Vec::with_capacity(proto_h * proto_w * num_protos);
        for y in 0..proto_h {
            for x in 0..proto_w {
                for k in 0..num_protos {
                    let v = ((y as i32 * 7 + x as i32 * 13 + k as i32 * 3) % 127) as i8 - 60;
                    proto_i8.push(v);
                }
            }
        }

        let coeff_i8: Vec<i8> = (0..num_protos).map(|k| k as i8 * 10 + 5).collect();

        // f32 versions (symmetric quant, zp=0, scale=1.0).
        let proto_f32: Vec<f32> = proto_i8.iter().map(|&v| v as f32).collect();
        let coeff_f32: Vec<f32> = coeff_i8.iter().map(|&v| v as f32).collect();

        let proto_data_f32 =
            make_proto_data_with_values(proto_h, proto_w, num_protos, proto_f32, vec![coeff_f32]);
        let proto_data_i8 = make_proto_data_i8(
            proto_h,
            proto_w,
            num_protos,
            proto_i8,
            (1.0, 0),
            coeff_i8,
            (1.0, 0),
            1,
        );

        let det = [make_detect_box(0.05, 0.05, 0.95, 0.95)];
        let cpu = CPUProcessor::new();

        let segs_f32 = cpu
            .materialize_scaled_segmentations(&det, &proto_data_f32, None, 32, 32)
            .expect("f32 path");
        let segs_i8 = cpu
            .materialize_scaled_segmentations(&det, &proto_data_i8, None, 32, 32)
            .expect("i8 path");

        assert_eq!(segs_f32.len(), 1);
        assert_eq!(segs_i8.len(), 1);
        assert_eq!(
            segs_f32[0].segmentation.shape(),
            segs_i8[0].segmentation.shape(),
            "i8 and f32 paths should produce same-shaped masks"
        );

        let f32_mask: Vec<u8> = seg_bytes(&segs_f32[0]).to_vec();
        let i8_mask: Vec<u8> = seg_bytes(&segs_i8[0]).to_vec();
        assert_eq!(
            f32_mask, i8_mask,
            "i8×i8 integer path must produce identical binarized mask as f32 path (symmetric quant)"
        );
    }

    #[test]
    fn test_materialize_i16_i8_parity_with_float() {
        let proto_h = 8;
        let proto_w = 8;
        let num_protos = 4;

        let mut proto_i8 = Vec::with_capacity(proto_h * proto_w * num_protos);
        for y in 0..proto_h {
            for x in 0..proto_w {
                for k in 0..num_protos {
                    let v = ((y as i32 * 7 + x as i32 * 13 + k as i32 * 3) % 127) as i8 - 60;
                    proto_i8.push(v);
                }
            }
        }

        let coeff_i16 = vec![300_i16, -250, 180, -130];
        let proto_f32: Vec<f32> = proto_i8.iter().map(|&v| v as f32).collect();
        let coeff_f32: Vec<f32> = coeff_i16.iter().map(|&v| v as f32).collect();

        let proto_data_f32 =
            make_proto_data_with_values(proto_h, proto_w, num_protos, proto_f32, vec![coeff_f32]);
        let proto_data_i16 = make_proto_data_i16_i8(
            proto_h,
            proto_w,
            num_protos,
            proto_i8,
            (1.0, 0),
            coeff_i16,
            (1.0, 0),
            1,
        );

        let det = [make_detect_box(0.05, 0.05, 0.95, 0.95)];
        let cpu = CPUProcessor::new();

        let segs_f32 = cpu
            .materialize_segmentations(&det, &proto_data_f32, None)
            .expect("f32 path");
        let segs_i16 = cpu
            .materialize_segmentations(&det, &proto_data_i16, None)
            .expect("i16 path");

        let f32_mask: Vec<u8> = seg_bytes(&segs_f32[0]).to_vec();
        let i16_mask: Vec<u8> = seg_bytes(&segs_i16[0]).to_vec();
        assert_eq!(
            f32_mask, i16_mask,
            "i16×i8 proto path should match f32 fallback"
        );
    }

    #[test]
    fn test_materialize_scaled_i16_i8_parity_with_float() {
        let proto_h = 8;
        let proto_w = 8;
        let num_protos = 4;

        let mut proto_i8 = Vec::with_capacity(proto_h * proto_w * num_protos);
        for y in 0..proto_h {
            for x in 0..proto_w {
                for k in 0..num_protos {
                    let v = ((y as i32 * 11 + x as i32 * 5 + k as i32 * 17) % 127) as i8 - 60;
                    proto_i8.push(v);
                }
            }
        }

        let coeff_i16 = vec![220_i16, -180, 150, -90];
        let proto_f32: Vec<f32> = proto_i8.iter().map(|&v| v as f32).collect();
        let coeff_f32: Vec<f32> = coeff_i16.iter().map(|&v| v as f32).collect();

        let proto_data_f32 =
            make_proto_data_with_values(proto_h, proto_w, num_protos, proto_f32, vec![coeff_f32]);
        let proto_data_i16 = make_proto_data_i16_i8(
            proto_h,
            proto_w,
            num_protos,
            proto_i8,
            (1.0, 0),
            coeff_i16,
            (1.0, 0),
            1,
        );

        let det = [make_detect_box(0.05, 0.05, 0.95, 0.95)];
        let cpu = CPUProcessor::new();

        let segs_f32 = cpu
            .materialize_scaled_segmentations(&det, &proto_data_f32, None, 32, 32)
            .expect("f32 path");
        let segs_i16 = cpu
            .materialize_scaled_segmentations(&det, &proto_data_i16, None, 32, 32)
            .expect("i16 path");

        let f32_mask: Vec<u8> = seg_bytes(&segs_f32[0]).to_vec();
        let i16_mask: Vec<u8> = seg_bytes(&segs_i16[0]).to_vec();
        assert_eq!(
            f32_mask, i16_mask,
            "scaled i16×i8 path should match f32 fallback"
        );
    }

    #[test]
    fn test_materialize_nchw_i8_f32_coefficients_use_fallback() {
        use edgefirst_tensor::{Quantization, Tensor};
        let cpu = CPUProcessor::new();
        let (proto_h, proto_w, num_protos) = (4, 4, 2);
        let mut proto_values = vec![0_i8; num_protos * proto_h * proto_w];
        for i in 0..proto_h * proto_w {
            proto_values[i] = 10;
            proto_values[proto_h * proto_w + i] = -10;
        }
        let mut protos_t =
            Tensor::<i8>::from_slice(&proto_values, &[num_protos, proto_h, proto_w]).unwrap();
        protos_t
            .set_quantization(Quantization::per_tensor(0.1, 0))
            .unwrap();
        let coeff_t = Tensor::<f32>::from_slice(&[1.0_f32, 0.0], &[1, num_protos]).unwrap();
        let proto_data = crate::ProtoData {
            mask_coefficients: coeff_t.into(),
            protos: protos_t.into(),
            layout: edgefirst_tensor::ProtoLayout::Nchw,
        };
        let det = [make_detect_box(0.0, 0.0, 1.0, 1.0)];
        let segs = cpu
            .materialize_segmentations(&det, &proto_data, None)
            .expect("NCHW i8 protos should transpose in f32 fallback");
        assert!(seg_bytes(&segs[0]).iter().all(|&v| v == 255));
    }

    #[test]
    fn test_materialize_scaled_nchw_i8_f32_coefficients_use_fallback() {
        use edgefirst_tensor::{Quantization, Tensor};
        let cpu = CPUProcessor::new();
        let (proto_h, proto_w, num_protos) = (4, 4, 2);
        let mut proto_values = vec![0_i8; num_protos * proto_h * proto_w];
        for i in 0..proto_h * proto_w {
            proto_values[i] = 10;
            proto_values[proto_h * proto_w + i] = -10;
        }
        let mut protos_t =
            Tensor::<i8>::from_slice(&proto_values, &[num_protos, proto_h, proto_w]).unwrap();
        protos_t
            .set_quantization(Quantization::per_tensor(0.1, 0))
            .unwrap();
        let coeff_t = Tensor::<f32>::from_slice(&[1.0_f32, 0.0], &[1, num_protos]).unwrap();
        let proto_data = crate::ProtoData {
            mask_coefficients: coeff_t.into(),
            protos: protos_t.into(),
            layout: edgefirst_tensor::ProtoLayout::Nchw,
        };
        let det = [make_detect_box(0.0, 0.0, 1.0, 1.0)];
        let segs = cpu
            .materialize_scaled_segmentations(&det, &proto_data, None, 16, 16)
            .expect("NCHW i8 protos should transpose in scaled f32 fallback");
        assert!(seg_bytes(&segs[0]).iter().all(|&v| v == 255));
    }

    #[test]
    fn test_materialize_scaled_i8_i8_multiple_detections() {
        let proto_h = 4;
        let proto_w = 4;
        let num_protos = 2;
        let proto_values = vec![10_i8; proto_h * proto_w * num_protos];
        // Two detections: one positive, one negative.
        let mut coeff_values = Vec::with_capacity(2 * num_protos);
        coeff_values.extend_from_slice(&[10_i8, 10]); // det 0: positive
        coeff_values.extend_from_slice(&[-10_i8, -10]); // det 1: negative
        let proto_data = make_proto_data_i8(
            proto_h,
            proto_w,
            num_protos,
            proto_values,
            (0.1, 0),
            coeff_values,
            (0.1, 0),
            2,
        );
        let det = [
            make_detect_box(0.1, 0.1, 0.5, 0.5),
            make_detect_box(0.5, 0.5, 0.9, 0.9),
        ];
        let cpu = CPUProcessor::new();
        let segs = cpu
            .materialize_scaled_segmentations(&det, &proto_data, None, 16, 16)
            .expect("multiple detections i8×i8 should succeed");
        assert_eq!(segs.len(), 2);
        assert!(seg_bytes(&segs[0]).iter().all(|&v| v == 255));
        assert!(seg_bytes(&segs[1]).iter().all(|&v| v == 0));
    }

    // ── Multiplane PixelFormat::Nv12 tests ───────────────────────────────────────

    #[test]
    fn test_multiplane_nv12_creation() -> Result<()> {
        let luma = Tensor::<u8>::new(&[720, 1280], Some(TensorMemory::Mem), Some("luma"))?;
        let chroma = Tensor::<u8>::new(&[360, 1280], Some(TensorMemory::Mem), Some("chroma"))?;
        let img = {
            let __t = Tensor::<u8>::from_planes(luma, chroma, PixelFormat::Nv12)?;
            TensorDyn::from(__t)
        };

        assert_eq!(img.width(), Some(1280));
        assert_eq!(img.height(), Some(720));
        assert_eq!(img.format().unwrap(), PixelFormat::Nv12);
        assert!(img.as_u8().unwrap().is_multiplane());
        assert!(img.as_u8().unwrap().chroma().is_some());
        Ok(())
    }

    #[test]
    fn test_multiplane_is_multiplane() -> Result<()> {
        // Contiguous Nv12 is not multiplane.
        let contiguous = TensorDyn::image(
            640,
            480,
            PixelFormat::Nv12,
            DType::U8,
            Some(TensorMemory::Mem),
            edgefirst_tensor::CpuAccess::ReadWrite,
        )?;
        assert!(!contiguous.as_u8().unwrap().is_multiplane());
        assert!(contiguous.as_u8().unwrap().chroma().is_none());

        // `from_planes` is multiplane.
        let luma = Tensor::<u8>::new(&[480, 640], Some(TensorMemory::Mem), None)?;
        let chroma = Tensor::<u8>::new(&[240, 640], Some(TensorMemory::Mem), None)?;
        let multiplane = {
            let __t = Tensor::<u8>::from_planes(luma, chroma, PixelFormat::Nv12)?;
            TensorDyn::from(__t)
        };
        assert!(multiplane.as_u8().unwrap().is_multiplane());
        assert!(multiplane.as_u8().unwrap().chroma().is_some());

        // Rgb is not multiplane.
        let rgb = TensorDyn::image(
            640,
            480,
            PixelFormat::Rgb,
            DType::U8,
            Some(TensorMemory::Mem),
            edgefirst_tensor::CpuAccess::ReadWrite,
        )?;
        assert!(!rgb.as_u8().unwrap().is_multiplane());
        Ok(())
    }

    #[test]
    fn test_multiplane_invalid_shapes() {
        // Rgb is not a multiplane format.
        let luma = Tensor::<u8>::new(&[480, 640], Some(TensorMemory::Mem), None).unwrap();
        let chroma = Tensor::<u8>::new(&[240, 640], Some(TensorMemory::Mem), None).unwrap();
        assert!(Tensor::<u8>::from_planes(luma, chroma, PixelFormat::Rgb).is_err());

        // Chroma height mismatch for Nv12.
        let luma = Tensor::<u8>::new(&[480, 640], Some(TensorMemory::Mem), None).unwrap();
        let chroma = Tensor::<u8>::new(&[480, 640], Some(TensorMemory::Mem), None).unwrap();
        assert!(Tensor::<u8>::from_planes(luma, chroma, PixelFormat::Nv12).is_err());

        // Chroma width mismatch
        let luma = Tensor::<u8>::new(&[480, 640], Some(TensorMemory::Mem), None).unwrap();
        let chroma = Tensor::<u8>::new(&[240, 320], Some(TensorMemory::Mem), None).unwrap();
        assert!(Tensor::<u8>::from_planes(luma, chroma, PixelFormat::Nv12).is_err());

        // 3D luma.
        let luma = Tensor::<u8>::new(&[480, 640, 1], Some(TensorMemory::Mem), None).unwrap();
        let chroma = Tensor::<u8>::new(&[240, 640], Some(TensorMemory::Mem), None).unwrap();
        assert!(Tensor::<u8>::from_planes(luma, chroma, PixelFormat::Nv12).is_err());
    }

    #[test]
    fn test_multiplane_nv12_to_rgb_cpu() -> Result<()> {
        let nv12_bytes = edgefirst_bench::testdata::read("camera720p.nv12");
        let width = 1280usize;
        let height = 720usize;
        let y_size = width * height;
        let uv_size = width * (height / 2);

        // ── Contiguous path (baseline) ──────────────────────────────
        let contiguous = load_bytes_to_tensor(width, height, PixelFormat::Nv12, None, &nv12_bytes)?;
        let dst_contiguous = TensorDyn::image(
            width,
            height,
            PixelFormat::Rgb,
            DType::U8,
            mem(),
            edgefirst_tensor::CpuAccess::ReadWrite,
        )?;
        let mut converter = CPUProcessor::default();
        let contiguous_dyn = contiguous;
        let mut dst_contiguous_dyn = dst_contiguous;
        converter.convert(&contiguous_dyn, &mut dst_contiguous_dyn, None)?;

        // ── Multiplane path ─────────────────────────────────────────
        let luma = Tensor::<u8>::new(&[height, width], Some(TensorMemory::Mem), Some("luma"))?;
        luma.map()?.as_mut_slice()[..y_size].copy_from_slice(&nv12_bytes[..y_size]);

        let chroma = Tensor::<u8>::new(
            &[height / 2, width],
            Some(TensorMemory::Mem),
            Some("chroma"),
        )?;
        chroma.map()?.as_mut_slice()[..uv_size]
            .copy_from_slice(&nv12_bytes[y_size..y_size + uv_size]);

        let multiplane = {
            let __t = Tensor::<u8>::from_planes(luma, chroma, PixelFormat::Nv12)?;
            TensorDyn::from(__t)
        };
        let dst_multiplane = TensorDyn::image(
            width,
            height,
            PixelFormat::Rgb,
            DType::U8,
            mem(),
            edgefirst_tensor::CpuAccess::ReadWrite,
        )?;
        let multiplane_dyn = multiplane;
        let mut dst_multiplane_dyn = dst_multiplane;
        converter.convert(&multiplane_dyn, &mut dst_multiplane_dyn, None)?;

        // ── Compare: both paths must produce identical output ───────
        let contiguous_map = dst_contiguous_dyn.as_u8().unwrap().map()?;
        let multiplane_map = dst_multiplane_dyn.as_u8().unwrap().map()?;
        assert_eq!(
            contiguous_map.as_slice(),
            multiplane_map.as_slice(),
            "multiplane PixelFormat::Nv12→PixelFormat::Rgb must match contiguous path"
        );
        Ok(())
    }

    #[test]
    fn test_multiplane_nv12_to_rgba_cpu() -> Result<()> {
        let nv12_bytes = edgefirst_bench::testdata::read("camera720p.nv12");
        let width = 1280usize;
        let height = 720usize;
        let y_size = width * height;
        let uv_size = width * (height / 2);

        let contiguous = load_bytes_to_tensor(width, height, PixelFormat::Nv12, None, &nv12_bytes)?;
        let dst_contiguous = TensorDyn::image(
            width,
            height,
            PixelFormat::Rgba,
            DType::U8,
            mem(),
            edgefirst_tensor::CpuAccess::ReadWrite,
        )?;
        let mut converter = CPUProcessor::default();
        let contiguous_dyn = contiguous;
        let mut dst_contiguous_dyn = dst_contiguous;
        converter.convert(&contiguous_dyn, &mut dst_contiguous_dyn, None)?;

        let luma = Tensor::<u8>::new(&[height, width], Some(TensorMemory::Mem), Some("luma"))?;
        luma.map()?.as_mut_slice()[..y_size].copy_from_slice(&nv12_bytes[..y_size]);
        let chroma = Tensor::<u8>::new(
            &[height / 2, width],
            Some(TensorMemory::Mem),
            Some("chroma"),
        )?;
        chroma.map()?.as_mut_slice()[..uv_size]
            .copy_from_slice(&nv12_bytes[y_size..y_size + uv_size]);

        let multiplane = {
            let __t = Tensor::<u8>::from_planes(luma, chroma, PixelFormat::Nv12)?;
            TensorDyn::from(__t)
        };
        let dst_multiplane = TensorDyn::image(
            width,
            height,
            PixelFormat::Rgba,
            DType::U8,
            mem(),
            edgefirst_tensor::CpuAccess::ReadWrite,
        )?;
        let multiplane_dyn = multiplane;
        let mut dst_multiplane_dyn = dst_multiplane;
        converter.convert(&multiplane_dyn, &mut dst_multiplane_dyn, None)?;

        let contiguous_map = dst_contiguous_dyn.as_u8().unwrap().map()?;
        let multiplane_map = dst_multiplane_dyn.as_u8().unwrap().map()?;
        assert_eq!(
            contiguous_map.as_slice(),
            multiplane_map.as_slice(),
            "multiplane PixelFormat::Nv12→PixelFormat::Rgba must match contiguous path"
        );
        Ok(())
    }

    #[test]
    fn test_multiplane_nv12_to_grey_cpu() -> Result<()> {
        let nv12_bytes = edgefirst_bench::testdata::read("camera720p.nv12");
        let width = 1280usize;
        let height = 720usize;
        let y_size = width * height;
        let uv_size = width * (height / 2);

        let contiguous = load_bytes_to_tensor(width, height, PixelFormat::Nv12, None, &nv12_bytes)?;
        let dst_contiguous = TensorDyn::image(
            width,
            height,
            PixelFormat::Grey,
            DType::U8,
            mem(),
            edgefirst_tensor::CpuAccess::ReadWrite,
        )?;
        let mut converter = CPUProcessor::default();
        let contiguous_dyn = contiguous;
        let mut dst_contiguous_dyn = dst_contiguous;
        converter.convert(&contiguous_dyn, &mut dst_contiguous_dyn, None)?;

        let luma = Tensor::<u8>::new(&[height, width], Some(TensorMemory::Mem), Some("luma"))?;
        luma.map()?.as_mut_slice()[..y_size].copy_from_slice(&nv12_bytes[..y_size]);
        let chroma = Tensor::<u8>::new(
            &[height / 2, width],
            Some(TensorMemory::Mem),
            Some("chroma"),
        )?;
        chroma.map()?.as_mut_slice()[..uv_size]
            .copy_from_slice(&nv12_bytes[y_size..y_size + uv_size]);

        let multiplane = {
            let __t = Tensor::<u8>::from_planes(luma, chroma, PixelFormat::Nv12)?;
            TensorDyn::from(__t)
        };
        let dst_multiplane = TensorDyn::image(
            width,
            height,
            PixelFormat::Grey,
            DType::U8,
            mem(),
            edgefirst_tensor::CpuAccess::ReadWrite,
        )?;
        let multiplane_dyn = multiplane;
        let mut dst_multiplane_dyn = dst_multiplane;
        converter.convert(&multiplane_dyn, &mut dst_multiplane_dyn, None)?;

        let contiguous_map = dst_contiguous_dyn.as_u8().unwrap().map()?;
        let multiplane_map = dst_multiplane_dyn.as_u8().unwrap().map()?;
        assert_eq!(
            contiguous_map.as_slice(),
            multiplane_map.as_slice(),
            "multiplane PixelFormat::Nv12→PixelFormat::Grey must match contiguous path"
        );
        Ok(())
    }

    // ──────────────────────────────────────────────────────────────────────
    // src_rect clamping tests
    // ──────────────────────────────────────────────────────────────────────

    /// Create a synthetic RGB tensor where the left half is pure red and the
    /// right half is pure blue.
    fn make_red_blue_src(width: usize, height: usize) -> TensorDyn {
        let mut t = TensorDyn::image(
            width,
            height,
            PixelFormat::Rgb,
            DType::U8,
            mem(),
            edgefirst_tensor::CpuAccess::ReadWrite,
        )
        .unwrap();
        {
            let tensor_u8 = t.as_u8_mut().unwrap();
            let mut map = tensor_u8.map().unwrap();
            let data = map.as_mut_slice();
            let half = width / 2;
            for y in 0..height {
                for x in 0..width {
                    let idx = (y * width + x) * 3;
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

    /// Cropping the blue right half of a red|blue image and resizing leaves no red in the interior.
    #[test]
    fn test_src_rect_crop_no_bleed_cpu() -> Result<()> {
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
            mem(),
            edgefirst_tensor::CpuAccess::ReadWrite,
        )?;

        let mut cpu = CPUProcessor::default();

        let crop = Crop::new().with_source(Some(Region::new(src_w / 2, 0, src_w / 2, src_h)));

        cpu.convert(
            &src,
            &mut dst,
            Some(crate::ConvertOptions::new().crop(crop)),
        )?;

        // Skips the leftmost 2 columns.
        let map = dst.as_u8().unwrap().map()?;
        let data = map.as_slice();
        for y in 0..dst_h {
            for x in 2..dst_w {
                let i = y * dst_w + x;
                let r = data[i * 3];
                let g = data[i * 3 + 1];
                let b = data[i * 3 + 2];
                assert!(
                    r <= 2 && g <= 2 && b >= 253,
                    "CPU pixel ({x},{y}) has red bleed: RGB=({r},{g},{b}), expected pure blue"
                );
            }
        }
        Ok(())
    }

    /// Cropping the red left half and resizing leaves no blue in the interior.
    #[test]
    fn test_src_rect_crop_left_half_no_bleed_cpu() -> Result<()> {
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
            mem(),
            edgefirst_tensor::CpuAccess::ReadWrite,
        )?;

        let mut cpu = CPUProcessor::default();

        let crop = Crop::new().with_source(Some(Region::new(0, 0, src_w / 2, src_h)));

        cpu.convert(
            &src,
            &mut dst,
            Some(crate::ConvertOptions::new().crop(crop)),
        )?;

        // Skips the rightmost 2 columns.
        let map = dst.as_u8().unwrap().map()?;
        let data = map.as_slice();
        for y in 0..dst_h {
            for x in 0..(dst_w - 2) {
                let i = y * dst_w + x;
                let r = data[i * 3];
                let g = data[i * 3 + 1];
                let b = data[i * 3 + 2];
                assert!(
                    r >= 253 && g <= 2 && b <= 2,
                    "CPU pixel ({x},{y}) has blue bleed: RGB=({r},{g},{b}), expected pure red"
                );
            }
        }
        Ok(())
    }

    /// A crop at the exact red→blue boundary resizes to pure blue in the interior.
    #[test]
    fn test_src_rect_boundary_crop_no_bleed_cpu() -> Result<()> {
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
            mem(),
            edgefirst_tensor::CpuAccess::ReadWrite,
        )?;

        let mut cpu = CPUProcessor::default();

        let crop = Crop::new().with_source(Some(Region::new(src_w / 2, 0, src_w / 2, src_h)));

        cpu.convert(
            &src,
            &mut dst,
            Some(crate::ConvertOptions::new().crop(crop)),
        )?;

        // Skips the leftmost 2 columns.
        let map = dst.as_u8().unwrap().map()?;
        let data = map.as_slice();
        for y in 0..dst_h {
            for x in 2..dst_w {
                let i = y * dst_w + x;
                let r = data[i * 3];
                let g = data[i * 3 + 1];
                let b = data[i * 3 + 2];
                assert!(
                    r <= 2 && g <= 2 && b >= 253,
                    "CPU pixel ({x},{y}) contamination at boundary: RGB=({r},{g},{b})"
                );
            }
        }
        Ok(())
    }

    // =========================================================================
    // CPU F16/F32 float destination tests
    // =========================================================================

    /// RGBA8 → Rgb F32 at the same size: every output element equals `src_byte as f32 / 255.0`.
    #[test]
    fn cpu_convert_rgba_to_rgb_f32_identity() -> Result<()> {
        const W: usize = 4;
        const H: usize = 4;

        let src = TensorDyn::image(
            W,
            H,
            PixelFormat::Rgba,
            DType::U8,
            Some(TensorMemory::Mem),
            edgefirst_tensor::CpuAccess::ReadWrite,
        )?;
        {
            let mut map = src.as_u8().unwrap().map()?;
            let data = map.as_mut_slice();
            for i in 0..W * H {
                data[i * 4] = (i * 10) as u8; // R
                data[i * 4 + 1] = (i * 10 + 3) as u8; // G
                data[i * 4 + 2] = (i * 10 + 7) as u8; // B
                data[i * 4 + 3] = 255; // A
            }
        }

        let mut dst = TensorDyn::image(
            W,
            H,
            PixelFormat::Rgb,
            DType::F32,
            Some(TensorMemory::Mem),
            edgefirst_tensor::CpuAccess::ReadWrite,
        )?;
        {
            let mut cv = CPUProcessor::default();
            cv.convert(&src, &mut dst, None)?;
        }

        let src_map = src.as_u8().unwrap().map()?;
        let src_bytes = src_map.as_slice();
        let dst_map = dst.as_f32().unwrap().map()?;
        let dst_floats = dst_map.as_slice();

        assert_eq!(dst_floats.len(), W * H * 3, "wrong output element count");
        for i in 0..W * H {
            for c in 0..3 {
                let expected = src_bytes[i * 4 + c] as f32 / 255.0;
                let actual = dst_floats[i * 3 + c];
                assert_eq!(
                    actual, expected,
                    "pixel {i} channel {c}: got {actual}, expected {expected}"
                );
            }
        }
        Ok(())
    }

    /// RGBA8 → PlanarRgb F16 at the same size normalizes to [0,1] within 2^-9 and places each channel in its plane.
    #[test]
    fn cpu_convert_rgba_to_planar_rgb_f16_identity() -> Result<()> {
        const W: usize = 4;
        const H: usize = 4;

        // pixel (y,x): R = 50+x, G = 100+y*10, B = 200
        let src = TensorDyn::image(
            W,
            H,
            PixelFormat::Rgba,
            DType::U8,
            Some(TensorMemory::Mem),
            edgefirst_tensor::CpuAccess::ReadWrite,
        )?;
        {
            let mut map = src.as_u8().unwrap().map()?;
            let data = map.as_mut_slice();
            for y in 0..H {
                for x in 0..W {
                    let i = y * W + x;
                    data[i * 4] = (50 + x) as u8; // R
                    data[i * 4 + 1] = (100 + y * 10) as u8; // G
                    data[i * 4 + 2] = 200; // B
                    data[i * 4 + 3] = 255; // A
                }
            }
        }

        let mut dst = TensorDyn::image(
            W,
            H,
            PixelFormat::PlanarRgb,
            DType::F16,
            Some(TensorMemory::Mem),
            edgefirst_tensor::CpuAccess::ReadWrite,
        )?;
        {
            let mut cv = CPUProcessor::default();
            cv.convert(&src, &mut dst, None)?;
        }

        let tol = 2.0_f32.powi(-9);

        let src_map = src.as_u8().unwrap().map()?;
        let src_bytes = src_map.as_slice();
        let dst_map = dst.as_f16().unwrap().map()?;
        let dst_halfs = dst_map.as_slice();

        // PlanarRgb layout: [R_plane W*H, G_plane W*H, B_plane W*H]
        let plane = W * H;
        assert_eq!(dst_halfs.len(), plane * 3, "wrong output element count");

        for y in 0..H {
            for x in 0..W {
                let i = y * W + x;
                let r_expected = src_bytes[i * 4] as f32 / 255.0;
                let g_expected = src_bytes[i * 4 + 1] as f32 / 255.0;
                let b_expected = src_bytes[i * 4 + 2] as f32 / 255.0;

                let r_actual = dst_halfs[i].to_f32(); // R plane
                let g_actual = dst_halfs[plane + i].to_f32(); // G plane
                let b_actual = dst_halfs[2 * plane + i].to_f32(); // B plane

                assert!(
                    (r_actual - r_expected).abs() <= tol,
                    "R plane pixel ({x},{y}): got {r_actual}, expected {r_expected}"
                );
                assert!(
                    (g_actual - g_expected).abs() <= tol,
                    "G plane pixel ({x},{y}): got {g_actual}, expected {g_expected}"
                );
                assert!(
                    (b_actual - b_expected).abs() <= tol,
                    "B plane pixel ({x},{y}): got {b_actual}, expected {b_expected}"
                );

                let r_byte = src_bytes[i * 4];
                let g_byte = src_bytes[i * 4 + 1];
                let b_byte = src_bytes[i * 4 + 2];
                if r_byte != g_byte {
                    assert_ne!(
                        r_actual, g_actual,
                        "R and G planes must differ at ({x},{y})"
                    );
                }
                if g_byte != b_byte {
                    assert_ne!(
                        g_actual, b_actual,
                        "G and B planes must differ at ({x},{y})"
                    );
                }
            }
        }
        Ok(())
    }

    /// The `widen_scratch` cache is allocated on the first convert, reused at the same size and reallocated at a different one.
    #[test]
    fn cpu_widen_scratch_reused_across_converts() -> Result<()> {
        fn rgba(w: usize, h: usize) -> Result<TensorDyn> {
            let src = TensorDyn::image(
                w,
                h,
                PixelFormat::Rgba,
                DType::U8,
                Some(TensorMemory::Mem),
                edgefirst_tensor::CpuAccess::ReadWrite,
            )?;
            {
                let mut map = src.as_u8().unwrap().map()?;
                let data = map.as_mut_slice();
                for (i, px) in data.as_chunks_mut::<4>().0.iter_mut().enumerate() {
                    px[0] = (i * 7) as u8;
                    px[1] = (i * 7 + 1) as u8;
                    px[2] = (i * 7 + 2) as u8;
                    px[3] = 255;
                }
            }
            Ok(src)
        }

        let mut cv = CPUProcessor::default();

        let src4 = rgba(4, 4)?;
        let mut dst_a = TensorDyn::image(
            4,
            4,
            PixelFormat::Rgb,
            DType::F32,
            Some(TensorMemory::Mem),
            edgefirst_tensor::CpuAccess::ReadWrite,
        )?;
        let mut dst_b = TensorDyn::image(
            4,
            4,
            PixelFormat::Rgb,
            DType::F32,
            Some(TensorMemory::Mem),
            edgefirst_tensor::CpuAccess::ReadWrite,
        )?;

        // First convert: allocates the scratch.
        cv.convert(&src4, &mut dst_a, None)?;
        // Same dst format and size: reuses the scratch.
        cv.convert(&src4, &mut dst_b, None)?;

        let a = dst_a.as_f32().unwrap().map()?;
        let b = dst_b.as_f32().unwrap().map()?;
        assert_eq!(
            a.as_slice(),
            b.as_slice(),
            "cache-hit produced different output"
        );

        // A different size: reallocates the scratch.
        let src8 = rgba(8, 8)?;
        let mut dst_c = TensorDyn::image(
            8,
            8,
            PixelFormat::Rgb,
            DType::F32,
            Some(TensorMemory::Mem),
            edgefirst_tensor::CpuAccess::ReadWrite,
        )?;
        cv.convert(&src8, &mut dst_c, None)?;
        let c = dst_c.as_f32().unwrap().map()?;
        assert_eq!(c.as_slice().len(), 8 * 8 * 3);
        assert!(c.as_slice().iter().all(|v| (0.0..=1.0).contains(v)));
        Ok(())
    }

    /// `convert_dtype` returns `NotAnImage` when the destination tensor has no pixel format.
    #[test]
    fn cpu_convert_dst_without_format_errors() -> Result<()> {
        let src = TensorDyn::image(
            4,
            4,
            PixelFormat::Rgba,
            DType::U8,
            Some(TensorMemory::Mem),
            edgefirst_tensor::CpuAccess::ReadWrite,
        )?;
        let mut dst = TensorDyn::new(&[4, 4, 3], DType::F32, Some(TensorMemory::Mem), None)?;
        let mut cv = CPUProcessor::default();
        let err = cv.convert(&src, &mut dst, None).unwrap_err();
        assert!(
            matches!(err, Error::NotAnImage),
            "expected NotAnImage, got {err:?}"
        );
        Ok(())
    }

    /// OpenCV `INTER_LINEAR` sample of a `ch`-channel u8 image: two taps per
    /// axis at half-pixel centres, clamped at the borders.
    fn inter_linear(
        src: &[u8],
        ch: usize,
        (sw, sh): (usize, usize),
        (dw, dh): (usize, usize),
    ) -> Vec<f32> {
        let taps = |d: usize, n: usize, out: usize| {
            let s = ((d as f32 + 0.5) * (n as f32 / out as f32) - 0.5).clamp(0.0, (n - 1) as f32);
            let lo = s.floor() as usize;
            (lo, (lo + 1).min(n - 1), s - lo as f32)
        };
        let mut out = Vec::with_capacity(dw * dh * ch);
        for y in 0..dh {
            let (y0, y1, fy) = taps(y, sh, dh);
            for x in 0..dw {
                let (x0, x1, fx) = taps(x, sw, dw);
                for c in 0..ch {
                    let px = |yy: usize, xx: usize| f32::from(src[(yy * sw + xx) * ch + c]);
                    let top = px(y0, x0) * (1.0 - fx) + px(y0, x1) * fx;
                    let bottom = px(y1, x0) * (1.0 - fx) + px(y1, x1) * fx;
                    out.push(top * (1.0 - fy) + bottom * fy);
                }
            }
        }
        out
    }

    /// The CPU bilinear matches two-tap `INTER_LINEAR` on downscale and upscale for one, three and four channels.
    #[test]
    fn cpu_bilinear_matches_inter_linear_on_downscale_and_upscale() -> Result<()> {
        const SW: usize = 40;
        const SH: usize = 30;
        for (fmt, ch) in [
            (PixelFormat::Grey, 1),
            (PixelFormat::Rgb, 3),
            (PixelFormat::Rgba, 4),
        ] {
            // High-frequency content.
            let pixels: Vec<u8> = (0..SW * SH * ch)
                .map(|i| ((i * 2_654_435_761usize) >> 7) as u8)
                .collect();
            let src = TensorDyn::image(
                SW,
                SH,
                fmt,
                DType::U8,
                Some(TensorMemory::Mem),
                edgefirst_tensor::CpuAccess::ReadWrite,
            )?;
            src.as_u8()
                .unwrap()
                .map()?
                .as_mut_slice()
                .copy_from_slice(&pixels);
            for (dw, dh) in [(30, 22), (17, 13), (64, 48), (20, 15), (25, 45)] {
                let mut switched = CPUProcessor::new_nearest();
                switched.set_interpolation(crate::Interpolation::Bilinear);
                for (which, mut cpu) in [("new", CPUProcessor::new()), ("set", switched)] {
                    let mut dst = TensorDyn::image(
                        dw,
                        dh,
                        fmt,
                        DType::U8,
                        Some(TensorMemory::Mem),
                        edgefirst_tensor::CpuAccess::ReadWrite,
                    )?;
                    cpu.convert(&src, &mut dst, None)?;
                    let expected = inter_linear(&pixels, ch, (SW, SH), (dw, dh));
                    let map = dst.as_u8().unwrap().map()?;
                    assert_eq!(map.as_slice().len(), expected.len());
                    let worst = map
                        .as_slice()
                        .iter()
                        .zip(&expected)
                        .map(|(&got, &want)| (f32::from(got) - want).abs())
                        .fold(0.0f32, f32::max);
                    assert!(
                        worst <= 1.0 + 1e-3,
                        "{fmt} {which} {SW}x{SH} -> {dw}x{dh}: worst difference {worst}"
                    );
                }
            }
        }
        Ok(())
    }

    /// RGBA8 → Rgb F32 with an 8x8 → 4x4 downscale stays finite in [0,1] and keeps flat-colour regions exact.
    #[test]
    fn cpu_convert_rgba_to_rgb_f32_resize() -> Result<()> {
        const SW: usize = 8;
        const SH: usize = 8;
        const DW: usize = 4;
        const DH: usize = 4;

        // 8x8 source: the right half (x >= 4) is solid blue [0,0,255,255].
        let src = TensorDyn::image(
            SW,
            SH,
            PixelFormat::Rgba,
            DType::U8,
            Some(TensorMemory::Mem),
            edgefirst_tensor::CpuAccess::ReadWrite,
        )?;
        {
            let mut map = src.as_u8().unwrap().map()?;
            let data = map.as_mut_slice();
            for y in 0..SH {
                for x in 0..SW {
                    let i = y * SW + x;
                    if x >= 4 {
                        data[i * 4] = 0;
                        data[i * 4 + 1] = 0;
                        data[i * 4 + 2] = 255;
                        data[i * 4 + 3] = 255;
                    } else {
                        data[i * 4] = 128;
                        data[i * 4 + 1] = 64;
                        data[i * 4 + 2] = 32;
                        data[i * 4 + 3] = 255;
                    }
                }
            }
        }

        let mut dst = TensorDyn::image(
            DW,
            DH,
            PixelFormat::Rgb,
            DType::F32,
            Some(TensorMemory::Mem),
            edgefirst_tensor::CpuAccess::ReadWrite,
        )?;
        {
            let mut cv = CPUProcessor::default();
            cv.convert(&src, &mut dst, None)?;
        }

        let dst_map = dst.as_f32().unwrap().map()?;
        let dst_floats = dst_map.as_slice();

        assert_eq!(dst_floats.len(), DW * DH * 3, "wrong output element count");

        for (i, &v) in dst_floats.iter().enumerate() {
            assert!(
                v.is_finite() && (0.0..=1.0).contains(&v),
                "dst[{i}] = {v} is out of [0,1] or non-finite"
            );
        }

        // The rightmost column maps to src pixels x=6..7, inside the solid-blue half.
        for y in 0..DH {
            let x = DW - 1; // x=3
            let i = y * DW + x;
            let r = dst_floats[i * 3];
            let g = dst_floats[i * 3 + 1];
            let b = dst_floats[i * 3 + 2];
            assert_eq!(r, 0.0, "rightmost R must be 0.0 at ({x},{y})");
            assert_eq!(g, 0.0, "rightmost G must be 0.0 at ({x},{y})");
            assert_eq!(b, 1.0, "rightmost B must be 1.0 at ({x},{y})");
        }

        // The left src half is [128, 64, 32], normalised [~0.502, ~0.251, ~0.125];
        // x=0 maps inside it.
        for y in 0..DH {
            let i = (y * DW) * 3; // x=0
            assert!(
                (dst_floats[i] - 128.0 / 255.0).abs() < 0.02,
                "left R y={y}: {}",
                dst_floats[i]
            );
            assert!(
                (dst_floats[i + 1] - 64.0 / 255.0).abs() < 0.02,
                "left G y={y}: {}",
                dst_floats[i + 1]
            );
            assert!(
                (dst_floats[i + 2] - 32.0 / 255.0).abs() < 0.02,
                "left B y={y}: {}",
                dst_floats[i + 2]
            );
        }
        Ok(())
    }

    #[test]
    fn cpu_nv12_to_rgb_respects_tagged_709_vs_601() {
        use edgefirst_tensor::{ColorEncoding, ColorRange, Colorimetry};
        let make = |enc| {
            // 2x2 NV12: 4 luma + 1 UV pair (U=200,V=40).
            let mut src = load_bytes_to_tensor(
                2,
                2,
                PixelFormat::Nv12,
                None,
                &[180, 180, 180, 180, 200, 40],
            )
            .unwrap();
            src.set_colorimetry(Some(
                Colorimetry::default()
                    .with_encoding(enc)
                    .with_range(ColorRange::Limited),
            ));
            let mut dst = TensorDyn::image(
                2,
                2,
                PixelFormat::Rgb,
                DType::U8,
                None,
                edgefirst_tensor::CpuAccess::ReadWrite,
            )
            .unwrap();
            CPUProcessor::default()
                .convert(&src, &mut dst, None)
                .unwrap();
            dst.as_u8().unwrap().map().unwrap().as_slice().to_vec()
        };
        assert_ne!(make(ColorEncoding::Bt601), make(ColorEncoding::Bt709));
    }

    #[test]
    fn cpu_nv16_to_rgb_respects_tagged_709_vs_601() {
        use edgefirst_tensor::{ColorEncoding, ColorRange, Colorimetry};
        let make = |enc| {
            // 2x2 NV16 (4:2:2): 4 luma + 2 rows × 1 (Cb,Cr) pair = 4 chroma bytes (U=200,V=40).
            let mut src = load_bytes_to_tensor(
                2,
                2,
                PixelFormat::Nv16,
                None,
                &[180, 180, 180, 180, 200, 40, 200, 40],
            )
            .unwrap();
            src.set_colorimetry(Some(
                Colorimetry::default()
                    .with_encoding(enc)
                    .with_range(ColorRange::Limited),
            ));
            let mut dst = TensorDyn::image(
                2,
                2,
                PixelFormat::Rgb,
                DType::U8,
                None,
                edgefirst_tensor::CpuAccess::ReadWrite,
            )
            .unwrap();
            CPUProcessor::default()
                .convert(&src, &mut dst, None)
                .unwrap();
            dst.as_u8().unwrap().map().unwrap().as_slice().to_vec()
        };
        assert_ne!(make(ColorEncoding::Bt601), make(ColorEncoding::Bt709));
    }

    #[test]
    fn cpu_nv24_to_rgb_respects_tagged_709_vs_601() {
        use edgefirst_tensor::{ColorEncoding, ColorRange, Colorimetry};
        let make = |enc| {
            // 2x2 NV24 (4:4:4): 4 luma + 4 (Cb,Cr) pairs = 8 chroma bytes (U=200,V=40).
            let mut src = load_bytes_to_tensor(
                2,
                2,
                PixelFormat::Nv24,
                None,
                &[180, 180, 180, 180, 200, 40, 200, 40, 200, 40, 200, 40],
            )
            .unwrap();
            src.set_colorimetry(Some(
                Colorimetry::default()
                    .with_encoding(enc)
                    .with_range(ColorRange::Limited),
            ));
            let mut dst = TensorDyn::image(
                2,
                2,
                PixelFormat::Rgb,
                DType::U8,
                None,
                edgefirst_tensor::CpuAccess::ReadWrite,
            )
            .unwrap();
            CPUProcessor::default()
                .convert(&src, &mut dst, None)
                .unwrap();
            dst.as_u8().unwrap().map().unwrap().as_slice().to_vec()
        };
        assert_ne!(make(ColorEncoding::Bt601), make(ColorEncoding::Bt709));
    }

    #[test]
    fn cpu_convert_tagged_jfif_differs_from_untagged_heuristic_nv12() {
        use edgefirst_tensor::{ColorEncoding, ColorRange, Colorimetry};
        // NV12 2x2 = 4 Y-plane bytes + 2 UV-plane bytes = 6 total.
        let bytes = [180u8, 180, 180, 180, 200, 40];
        let convert = |cm: Option<Colorimetry>| {
            let mut src = load_bytes_to_tensor(2, 2, PixelFormat::Nv12, None, &bytes).unwrap();
            src.set_colorimetry(cm);
            let mut dst = TensorDyn::image(
                2,
                2,
                PixelFormat::Rgb,
                DType::U8,
                None,
                edgefirst_tensor::CpuAccess::ReadWrite,
            )
            .unwrap();
            CPUProcessor::default()
                .convert(&src, &mut dst, None)
                .unwrap();
            dst.as_u8().unwrap().map().unwrap().as_slice().to_vec()
        };
        // Tagged JFIF (BT.601 full-range).
        let jfif = convert(Some(Colorimetry::jfif()));
        // Untagged: resolves to BT.601 limited.
        let untagged = convert(None);
        assert_ne!(
            jfif, untagged,
            "JFIF (BT.601 full-range) must differ from heuristic (BT.601 limited)"
        );
        // Tagged BT.709 limited.
        let bt709 = convert(Some(
            Colorimetry::default()
                .with_encoding(ColorEncoding::Bt709)
                .with_range(ColorRange::Limited),
        ));
        assert_ne!(jfif, bt709, "BT.601-full must differ from BT.709-limited");
    }

    #[test]
    fn cpu_convert_into_heap_subviews_no_aliasing() {
        // Two distinct RGBA sources converted into two RGB sub-views of one
        // heap parent buffer; each window is compared with a standalone convert
        // of its source.
        let mut converter = CPUProcessor::default();

        let mut src0 = TensorDyn::image(
            4,
            4,
            PixelFormat::Rgba,
            DType::U8,
            Some(TensorMemory::Mem),
            edgefirst_tensor::CpuAccess::ReadWrite,
        )
        .unwrap();
        src0.as_u8_mut()
            .unwrap()
            .map()
            .unwrap()
            .as_mut_slice()
            .fill(50);
        let mut src1 = TensorDyn::image(
            4,
            4,
            PixelFormat::Rgba,
            DType::U8,
            Some(TensorMemory::Mem),
            edgefirst_tensor::CpuAccess::ReadWrite,
        )
        .unwrap();
        src1.as_u8_mut()
            .unwrap()
            .map()
            .unwrap()
            .as_mut_slice()
            .fill(200);

        let mut do_convert = |src: &TensorDyn, dst: &mut TensorDyn| {
            converter.convert(src, dst, None).unwrap();
        };

        // Standalone reference conversions into full, independent buffers.
        let mut ref0 = TensorDyn::image(
            4,
            4,
            PixelFormat::Rgb,
            DType::U8,
            Some(TensorMemory::Mem),
            edgefirst_tensor::CpuAccess::ReadWrite,
        )
        .unwrap();
        do_convert(&src0, &mut ref0);
        let mut ref1 = TensorDyn::image(
            4,
            4,
            PixelFormat::Rgb,
            DType::U8,
            Some(TensorMemory::Mem),
            edgefirst_tensor::CpuAccess::ReadWrite,
        )
        .unwrap();
        do_convert(&src1, &mut ref1);

        // One RGB parent holding two stacked 4x4 frames (4 wide, 8 tall).
        let frame = 4 * 4 * 3;
        let parent = TensorDyn::image(
            4,
            8,
            PixelFormat::Rgb,
            DType::U8,
            Some(TensorMemory::Mem),
            edgefirst_tensor::CpuAccess::ReadWrite,
        )
        .unwrap();
        let mut view0 = parent.view(Region::new(0, 0, 4, 4)).unwrap();
        let mut view1 = parent.view(Region::new(0, 4, 4, 4)).unwrap();
        assert_eq!(view1.plane_offset(), Some(frame));

        do_convert(&src0, &mut view0);
        do_convert(&src1, &mut view1);

        let parent_bytes = parent.as_u8().unwrap().map().unwrap().to_vec();
        let ref0_bytes = ref0.as_u8().unwrap().map().unwrap().to_vec();
        let ref1_bytes = ref1.as_u8().unwrap().map().unwrap().to_vec();
        assert_eq!(
            &parent_bytes[..frame],
            ref0_bytes.as_slice(),
            "view 0 window must match a standalone convert of src0"
        );
        assert_eq!(
            &parent_bytes[frame..2 * frame],
            ref1_bytes.as_slice(),
            "view 1 window must match a standalone convert of src1"
        );
    }

    /// A resize into a padded destination view lands each row at the view's stride.
    #[test]
    fn resize_into_padded_view_matches_standalone() {
        let tile_w = 320usize;
        let tile_h = 240usize;
        let channels = 3usize; // PixelFormat::Rgb

        // 640x480 Rgb source with non-uniform, deterministic content.
        let mut src = TensorDyn::image(
            640,
            480,
            PixelFormat::Rgb,
            DType::U8,
            Some(TensorMemory::Mem),
            edgefirst_tensor::CpuAccess::ReadWrite,
        )
        .unwrap();
        {
            let mut m = src.as_u8_mut().unwrap().map().unwrap();
            for (i, b) in m.as_mut_slice().iter_mut().enumerate() {
                *b = (i % 251) as u8;
            }
        }

        let mut converter = CPUProcessor::default();
        let mut do_convert = |dst: &mut TensorDyn| {
            converter.convert(&src, dst, None).unwrap();
        };

        // Reference: standalone tight 320x240 destination.
        let mut reference = TensorDyn::image(
            tile_w,
            tile_h,
            PixelFormat::Rgb,
            DType::U8,
            Some(TensorMemory::Mem),
            edgefirst_tensor::CpuAccess::ReadWrite,
        )
        .unwrap();
        do_convert(&mut reference);
        let reference_bytes = reference.as_u8().unwrap().map().unwrap().to_vec();

        // Parent: allocated wider than the tile, narrowed to `tile_w` via
        // `configure_image`, with a row stride 16 bytes past the tight pitch.
        let alloc_w = tile_w + 64;
        let mut parent = TensorDyn::image(
            alloc_w,
            2 * tile_h,
            PixelFormat::Rgb,
            DType::U8,
            Some(TensorMemory::Mem),
            edgefirst_tensor::CpuAccess::ReadWrite,
        )
        .unwrap();
        parent
            .configure_image(tile_w, 2 * tile_h, PixelFormat::Rgb)
            .unwrap();
        let padded_stride = tile_w * channels + 16;
        parent.set_row_stride(padded_stride).unwrap();

        // Second band: rows [tile_h, 2*tile_h) of the padded parent.
        let mut view = parent.view(Region::new(0, tile_h, tile_w, tile_h)).unwrap();
        do_convert(&mut view);

        // Compares each row's tight `width * channels` bytes at the parent's
        // stride with the standalone reference.
        let parent_bytes = parent.as_u8().unwrap().map().unwrap().to_vec();
        let band_offset = tile_h * padded_stride;
        let tight_row = tile_w * channels;
        for row in 0..tile_h {
            let parent_row_start = band_offset + row * padded_stride;
            let parent_row = &parent_bytes[parent_row_start..parent_row_start + tight_row];
            let ref_row = &reference_bytes[row * tight_row..(row + 1) * tight_row];
            assert_eq!(
                parent_row, ref_row,
                "row {row} of the padded destination view must match the standalone convert"
            );
        }
    }

    // ---- Fused NV→planar source region ----

    /// Slices each colour plane of `full` (a whole-frame NV12→PlanarRgb
    /// convert) down to `region`.
    fn manual_planar_rgb_crop(full: &TensorDyn, region: Region) -> Vec<u8> {
        let full_w = full.width().unwrap();
        let full_h = full.height().unwrap();
        let full_bytes = full.as_u8().unwrap().map().unwrap().to_vec();
        let plane_full = full_w * full_h;
        let plane_dst = region.width * region.height;
        let mut out = vec![0u8; plane_dst * 3];
        for p in 0..3 {
            let src_plane = &full_bytes[p * plane_full..(p + 1) * plane_full];
            let dst_plane = &mut out[p * plane_dst..(p + 1) * plane_dst];
            for row in 0..region.height {
                let s_off = (region.y + row) * full_w + region.x;
                let d_off = row * region.width;
                dst_plane[d_off..d_off + region.width]
                    .copy_from_slice(&src_plane[s_off..s_off + region.width]);
            }
        }
        out
    }

    /// Full-frame NV12→PlanarRgb convert of `camera720p.nv12`.
    fn nv12_720p_to_planar_rgb_full(converter: &mut CPUProcessor) -> (TensorDyn, TensorDyn) {
        let src = load_bytes_to_tensor(
            1280,
            720,
            PixelFormat::Nv12,
            None,
            &edgefirst_bench::testdata::read("camera720p.nv12"),
        )
        .unwrap();
        let mut full = TensorDyn::image(
            1280,
            720,
            PixelFormat::PlanarRgb,
            DType::U8,
            mem(),
            edgefirst_tensor::CpuAccess::ReadWrite,
        )
        .unwrap();
        converter.convert(&src, &mut full, None).unwrap();
        (src, full)
    }

    /// A scale-identity source crop with an even origin matches the whole-frame decode sliced to the same region.
    #[test]
    fn fused_region_matches_general_path() {
        let mut converter = CPUProcessor::default();
        let (src, full) = nv12_720p_to_planar_rgb_full(&mut converter);
        let region = Region::new(256, 128, 640, 360); // scale identity, even origin
        let reference = manual_planar_rgb_crop(&full, region);

        let mut cropped = TensorDyn::image(
            640,
            360,
            PixelFormat::PlanarRgb,
            DType::U8,
            mem(),
            edgefirst_tensor::CpuAccess::ReadWrite,
        )
        .unwrap();
        let crop = Crop::new().with_source(Some(region));
        converter
            .convert(
                &src,
                &mut cropped,
                Some(crate::ConvertOptions::new().crop(crop)),
            )
            .unwrap();

        let actual = cropped.as_u8().unwrap().map().unwrap().to_vec();
        assert_eq!(
            actual, reference,
            "fused NV12->PlanarRgb region convert must match the full-decode \
             + manual plane crop reference byte-for-byte"
        );
    }

    /// A scale-identity, chroma-aligned crop takes the fused path, shown by the `fused_hits` counter.
    #[test]
    fn scale_identity_crop_takes_fused_path() {
        let mut converter = CPUProcessor::default();
        let src = load_bytes_to_tensor(
            1280,
            720,
            PixelFormat::Nv12,
            None,
            &edgefirst_bench::testdata::read("camera720p.nv12"),
        )
        .unwrap();
        let mut cropped = TensorDyn::image(
            640,
            360,
            PixelFormat::PlanarRgb,
            DType::U8,
            mem(),
            edgefirst_tensor::CpuAccess::ReadWrite,
        )
        .unwrap();
        let crop = Crop::new().with_source(Some(Region::new(256, 128, 640, 360)));

        let before = converter.fused_hits();
        converter
            .convert(
                &src,
                &mut cropped,
                Some(crate::ConvertOptions::new().crop(crop)),
            )
            .unwrap();
        assert_eq!(
            converter.fused_hits(),
            before + 1,
            "scale-identity, chroma-aligned crop must take the fused NV->planar path"
        );
    }

    /// An odd-origin NV12 crop takes the general path and matches the oracle.
    #[test]
    fn odd_origin_nv12_crop_falls_through() {
        let mut converter = CPUProcessor::default();
        let (src, full) = nv12_720p_to_planar_rgb_full(&mut converter);
        let region = Region::new(257, 129, 640, 360); // odd left and top
        let reference = manual_planar_rgb_crop(&full, region);

        let mut cropped = TensorDyn::image(
            640,
            360,
            PixelFormat::PlanarRgb,
            DType::U8,
            mem(),
            edgefirst_tensor::CpuAccess::ReadWrite,
        )
        .unwrap();
        let crop = Crop::new().with_source(Some(region));

        let before = converter.fused_hits();
        converter
            .convert(
                &src,
                &mut cropped,
                Some(crate::ConvertOptions::new().crop(crop)),
            )
            .unwrap();
        assert_eq!(
            converter.fused_hits(),
            before,
            "odd-origin NV12 crop must never take the fused path (chroma \
             alignment requires even left/top) — never snap, never shift"
        );

        let actual = cropped.as_u8().unwrap().map().unwrap().to_vec();
        assert_eq!(
            actual, reference,
            "general-path output for the odd-origin crop must still be correct"
        );
    }

    /// A scale-identity, chroma-aligned crop with an odd width takes the fused NV12/NV16 path and matches the oracle.
    #[test]
    fn odd_width_scale_identity_crop_takes_fused_path_and_is_correct() {
        for (src_fmt, asset) in [
            (PixelFormat::Nv12, "camera720p.nv12"),
            (PixelFormat::Nv16, "camera720p.nv16"),
        ] {
            let mut converter = CPUProcessor::default();
            let src = load_bytes_to_tensor(
                1280,
                720,
                src_fmt,
                None,
                &edgefirst_bench::testdata::read(asset),
            )
            .unwrap();
            let mut full = TensorDyn::image(
                1280,
                720,
                PixelFormat::PlanarRgb,
                DType::U8,
                mem(),
                edgefirst_tensor::CpuAccess::ReadWrite,
            )
            .unwrap();
            converter.convert(&src, &mut full, None).unwrap();

            // Even left/top with an odd width.
            let region = Region::new(256, 128, 641, 360);
            let reference = manual_planar_rgb_crop(&full, region);

            let mut cropped = TensorDyn::image(
                region.width,
                region.height,
                PixelFormat::PlanarRgb,
                DType::U8,
                mem(),
                edgefirst_tensor::CpuAccess::ReadWrite,
            )
            .unwrap();
            let crop = Crop::new().with_source(Some(region));

            let before = converter.fused_hits();
            converter
                .convert(
                    &src,
                    &mut cropped,
                    Some(crate::ConvertOptions::new().crop(crop)),
                )
                .unwrap_or_else(|e| {
                    panic!("{src_fmt:?} odd-width scale-identity crop must not error: {e}")
                });
            assert_eq!(
                converter.fused_hits(),
                before + 1,
                "{src_fmt:?} odd-width, chroma-aligned crop must still take the fused path"
            );

            let actual = cropped.as_u8().unwrap().map().unwrap().to_vec();
            assert_eq!(
                actual, reference,
                "{src_fmt:?} odd-width fused region output must match the \
                 full-decode + manual plane crop reference"
            );
        }
    }

    /// `region.left == 0` crops stay fused and match the oracle for NV12 and NV16 with even and odd sizes.
    #[test]
    fn left_edge_crop_zero_copy_arm_is_fused_and_correct() {
        for (src_fmt, asset, region, last_row_tol, label) in [
            (
                PixelFormat::Nv12,
                "camera720p.nv12",
                Region::new(0, 80, 640, 640),
                0u8,
                "nv12 flush-bottom",
            ),
            (
                PixelFormat::Nv12,
                "camera720p.nv12",
                Region::new(0, 0, 641, 360),
                0u8,
                "nv12 odd-width even-height",
            ),
            (
                PixelFormat::Nv12,
                "camera720p.nv12",
                Region::new(0, 0, 641, 361),
                2u8,
                "nv12 odd-width odd-height",
            ),
            (
                PixelFormat::Nv16,
                "camera720p.nv16",
                Region::new(0, 0, 641, 361),
                0u8,
                "nv16 odd-width odd-height",
            ),
        ] {
            let mut converter = CPUProcessor::default();
            let src = load_bytes_to_tensor(
                1280,
                720,
                src_fmt,
                None,
                &edgefirst_bench::testdata::read(asset),
            )
            .unwrap();
            let mut full = TensorDyn::image(
                1280,
                720,
                PixelFormat::PlanarRgb,
                DType::U8,
                mem(),
                edgefirst_tensor::CpuAccess::ReadWrite,
            )
            .unwrap();
            converter.convert(&src, &mut full, None).unwrap();
            let reference = manual_planar_rgb_crop(&full, region);

            let mut cropped = TensorDyn::image(
                region.width,
                region.height,
                PixelFormat::PlanarRgb,
                DType::U8,
                mem(),
                edgefirst_tensor::CpuAccess::ReadWrite,
            )
            .unwrap();
            let crop = Crop::new().with_source(Some(region));

            let before = converter.fused_hits();
            converter
                .convert(
                    &src,
                    &mut cropped,
                    Some(crate::ConvertOptions::new().crop(crop)),
                )
                .unwrap_or_else(|e| panic!("left-edge {label} crop must not error: {e}"));
            assert_eq!(
                converter.fused_hits(),
                before + 1,
                "left-edge {label} crop must take the fused path"
            );

            let actual = cropped.as_u8().unwrap().map().unwrap().to_vec();
            if last_row_tol == 0 {
                assert_eq!(
                    actual, reference,
                    "left-edge {label} region output must match the \
                     full-decode + manual plane crop reference"
                );
            } else {
                // Odd-height NV12: all rows but the last are byte-exact; the last
                // row is compared with a tolerance.
                let (w, h) = (region.width, region.height);
                let plane = w * h;
                for p in 0..3 {
                    for row in 0..h {
                        let off = p * plane + row * w;
                        let (a, r) = (&actual[off..off + w], &reference[off..off + w]);
                        if row + 1 < h {
                            assert_eq!(
                                a, r,
                                "left-edge {label}: plane {p} row {row} must be byte-exact"
                            );
                        } else {
                            for (i, (x, y)) in a.iter().zip(r).enumerate() {
                                assert!(
                                    (i16::from(*x) - i16::from(*y)).abs() <= i16::from(last_row_tol),
                                    "left-edge {label}: plane {p} last row byte {i}: {x} vs {y} exceeds tolerance {last_row_tol}"
                                );
                            }
                        }
                    }
                }
            }
        }
    }

    /// Builds a `w`x`h` tensor in `fmt` whose mapped bytes follow a
    /// deterministic gradient.
    fn gradient_image(w: usize, h: usize, fmt: PixelFormat) -> TensorDyn {
        let t = TensorDyn::image(
            w,
            h,
            fmt,
            DType::U8,
            mem(),
            edgefirst_tensor::CpuAccess::ReadWrite,
        )
        .unwrap();
        {
            let u8t = t.as_u8().unwrap();
            let mut map = u8t.map_mut().unwrap();
            for (i, b) in map.as_mut_slice().iter_mut().enumerate() {
                *b = ((i * 7 + i / w * 13) & 0xFF) as u8;
            }
        }
        t
    }

    /// The crop-sized pre-resize intermediate is byte-identical to a full-frame intermediate for crops flush against every frame edge.
    #[test]
    fn crop_edge_matches_full_frame_intermediate() {
        const SRC_W: usize = 640;
        const SRC_H: usize = 480;
        const CW: usize = 128;
        const CH: usize = 96;

        for alg in ["bilinear", "nearest"] {
            for src_fmt in [PixelFormat::Nv12, PixelFormat::Nv16, PixelFormat::Nv24] {
                let mut converter = match alg {
                    "nearest" => CPUProcessor::new_nearest(),
                    _ => CPUProcessor::default(),
                };
                let src = gradient_image(SRC_W, SRC_H, src_fmt);

                // The full-frame intermediate: the whole frame, converted once.
                let mut full = gradient_image(SRC_W, SRC_H, PixelFormat::Rgb);
                converter.convert(&src, &mut full, None).unwrap();

                let origins = [
                    (0, 0),                   // top-left corner
                    (SRC_W - CW, 0),          // top-right, flush right
                    (0, SRC_H - CH),          // bottom-left, flush bottom
                    (SRC_W - CW, SRC_H - CH), // bottom-right, flush both
                    (101, 53),                // interior, odd on both axes
                ];
                for (left, top) in origins {
                    for (dst_w, dst_h) in [(CW, CH), (CW / 2, CH / 2)] {
                        let crop = Crop::new()
                            .with_source(Some(Region::new(left, top, CW, CH)))
                            .with_fit(crate::Fit::Stretch);

                        let mut actual = gradient_image(dst_w, dst_h, PixelFormat::Rgb);
                        converter
                            .convert(
                                &src,
                                &mut actual,
                                Some(crate::ConvertOptions::new().crop(crop)),
                            )
                            .unwrap();
                        let dims = converter.last_tmp_dims();

                        let mut expect = gradient_image(dst_w, dst_h, PixelFormat::Rgb);
                        converter
                            .convert(
                                &full,
                                &mut expect,
                                Some(crate::ConvertOptions::new().crop(crop)),
                            )
                            .unwrap();

                        let case = format!(
                            "{alg} {src_fmt:?} crop=({left},{top},{CW},{CH}) dst={dst_w}x{dst_h}"
                        );
                        assert_eq!(
                            actual.as_u8().unwrap().map().unwrap().to_vec(),
                            expect.as_u8().unwrap().map().unwrap().to_vec(),
                            "crop-sized intermediate changed output bytes for {case}"
                        );

                        // The intermediate is crop-sized.
                        let (tw, th) = dims.unwrap_or_else(|| {
                            panic!("{case}: expected a pre-resize intermediate")
                        });
                        assert!(
                            tw < SRC_W && th < SRC_H,
                            "{case}: intermediate {tw}x{th} did not shrink below the frame"
                        );
                        assert!(
                            tw >= CW && th >= CH,
                            "{case}: intermediate {tw}x{th} is smaller than the crop"
                        );
                    }
                }
            }
        }
    }

    /// A nearest scale-identity crop into a padded view matches a standalone tight convert and the full-frame intermediate.
    #[test]
    fn nearest_scale_identity_crop_into_padded_view() {
        const SRC_W: usize = 640;
        const SRC_H: usize = 480;
        let tile_w = 320usize;
        let tile_h = 240usize;
        let channels = 3usize; // PixelFormat::Rgb

        let src = gradient_image(SRC_W, SRC_H, PixelFormat::Nv12);
        // Even origin on both axes.
        let crop = Crop::new()
            .with_source(Some(Region::new(100, 52, tile_w, tile_h)))
            .with_fit(crate::Fit::Stretch);

        let mut converter = CPUProcessor::new_nearest();

        // Reference A: standalone, tightly-packed destination.
        let mut reference = gradient_image(tile_w, tile_h, PixelFormat::Rgb);
        converter
            .convert(
                &src,
                &mut reference,
                Some(crate::ConvertOptions::new().crop(crop)),
            )
            .unwrap();
        let reference_bytes = reference.as_u8().unwrap().map().unwrap().to_vec();

        // Reference B: the whole frame converted once into the packed-RGB
        // intermediate, then the same crop applied to it.
        let mut full = gradient_image(SRC_W, SRC_H, PixelFormat::Rgb);
        converter.convert(&src, &mut full, None).unwrap();
        let mut oracle = gradient_image(tile_w, tile_h, PixelFormat::Rgb);
        converter
            .convert(
                &full,
                &mut oracle,
                Some(crate::ConvertOptions::new().crop(crop)),
            )
            .unwrap();
        assert_eq!(
            reference_bytes,
            oracle.as_u8().unwrap().map().unwrap().to_vec(),
            "Nearest crop-sized intermediate diverged from the full-frame oracle"
        );

        // Padded parent: allocated wider, narrowed to the logical width, with a
        // row stride past the tight pitch.
        let mut parent = TensorDyn::image(
            tile_w + 64,
            2 * tile_h,
            PixelFormat::Rgb,
            DType::U8,
            mem(),
            edgefirst_tensor::CpuAccess::ReadWrite,
        )
        .unwrap();
        parent
            .configure_image(tile_w, 2 * tile_h, PixelFormat::Rgb)
            .unwrap();
        let padded_stride = tile_w * channels + 16;
        parent.set_row_stride(padded_stride).unwrap();

        let mut view = parent.view(Region::new(0, tile_h, tile_w, tile_h)).unwrap();
        converter
            .convert(
                &src,
                &mut view,
                Some(crate::ConvertOptions::new().crop(crop)),
            )
            .unwrap();

        let parent_bytes = parent.as_u8().unwrap().map().unwrap().to_vec();
        let band_offset = tile_h * padded_stride;
        let tight_row = tile_w * channels;
        for row in 0..tile_h {
            let start = band_offset + row * padded_stride;
            assert_eq!(
                &parent_bytes[start..start + tight_row],
                &reference_bytes[row * tight_row..(row + 1) * tight_row],
                "row {row} of the padded destination view must match the standalone convert"
            );
        }
    }

    /// A multiplane NV12 source extracts the same crop as the contiguous layout.
    #[test]
    fn crop_multiplane_nv12_matches_contiguous() {
        const W: usize = 640;
        const H: usize = 480;

        let contiguous = gradient_image(W, H, PixelFormat::Nv12);
        let (luma_bytes, chroma_bytes) = {
            let u8t = contiguous.as_u8().unwrap();
            let stride = u8t.effective_row_stride().unwrap_or(W);
            let map = u8t.map().unwrap();
            let s = map.as_slice();
            (s[..stride * H].to_vec(), s[stride * H..].to_vec())
        };

        let multiplane = {
            let luma = Tensor::<u8>::new(&[H, W], mem(), None).unwrap();
            let chroma = Tensor::<u8>::new(&[H / 2, W], mem(), None).unwrap();
            luma.map_mut().unwrap().as_mut_slice()[..luma_bytes.len()].copy_from_slice(&luma_bytes);
            let n = chroma_bytes
                .len()
                .min(chroma.map().unwrap().as_slice().len());
            chroma.map_mut().unwrap().as_mut_slice()[..n].copy_from_slice(&chroma_bytes[..n]);
            TensorDyn::from(Tensor::<u8>::from_planes(luma, chroma, PixelFormat::Nv12).unwrap())
        };

        let mut converter = CPUProcessor::default();
        // Cropped and scaled, with an odd origin.
        let crop = Crop::new()
            .with_source(Some(Region::new(101, 53, 320, 240)))
            .with_fit(crate::Fit::Stretch);

        let mut from_multi = gradient_image(160, 120, PixelFormat::Rgb);
        converter
            .convert(
                &multiplane,
                &mut from_multi,
                Some(crate::ConvertOptions::new().crop(crop)),
            )
            .unwrap();
        let mut from_contig = gradient_image(160, 120, PixelFormat::Rgb);
        converter
            .convert(
                &contiguous,
                &mut from_contig,
                Some(crate::ConvertOptions::new().crop(crop)),
            )
            .unwrap();

        assert_eq!(
            from_multi.as_u8().unwrap().map().unwrap().to_vec(),
            from_contig.as_u8().unwrap().map().unwrap().to_vec(),
            "multiplane NV12 crop extraction diverged from the contiguous layout"
        );
    }

    /// A cropped and scaled convert sizes its pre-resize intermediate to the crop plus the filter halo.
    #[test]
    fn pre_resize_intermediate_is_crop_sized() {
        const SRC_W: usize = 4032;
        const SRC_H: usize = 2268;
        const CROP: (usize, usize, usize, usize) = (512, 512, 640, 640);

        let mut converter = CPUProcessor::default();
        let src = TensorDyn::image(
            SRC_W,
            SRC_H,
            PixelFormat::Nv12,
            DType::U8,
            mem(),
            edgefirst_tensor::CpuAccess::ReadWrite,
        )
        .unwrap();
        {
            let u8t = src.as_u8().unwrap();
            let mut map = u8t.map_mut().unwrap();
            for (i, b) in map.as_mut_slice().iter_mut().enumerate() {
                *b = ((i * 7 + i / SRC_W * 13) & 0xFF) as u8;
            }
        }

        let mut dst = TensorDyn::image(
            320,
            320,
            PixelFormat::Rgb,
            DType::U8,
            mem(),
            edgefirst_tensor::CpuAccess::ReadWrite,
        )
        .unwrap();
        let crop = Crop::new()
            .with_source(Some(Region::new(CROP.0, CROP.1, CROP.2, CROP.3)))
            .with_fit(crate::Fit::Stretch);

        converter
            .convert(
                &src,
                &mut dst,
                Some(crate::ConvertOptions::new().crop(crop)),
            )
            .unwrap();

        let dims = converter
            .last_tmp_dims()
            .expect("a cropped NV12->Rgb scaled convert needs a pre-resize intermediate");
        assert_ne!(
            dims,
            (SRC_W, SRC_H),
            "the pre-resize intermediate is still frame-sized"
        );
        // At least the crop, and at most the crop plus 16 px per side.
        assert!(
            (CROP.2..=CROP.2 + 32).contains(&dims.0) && (CROP.3..=CROP.3 + 32).contains(&dims.1),
            "pre-resize intermediate {dims:?} is not crop-sized \
             (expected ~{}x{} + filter halo)",
            CROP.2,
            CROP.3
        );
    }
    /// The row-confined int8 bias changes only the pixel bytes of each logical row at the declared stride.
    #[test]
    fn int8_bias_rows_touches_only_the_pixel_bytes_at_the_declared_stride() {
        use crate::cpu::{apply_int8_xor_bias, apply_int8_xor_bias_rows};
        use edgefirst_tensor::{CpuAccess, PixelFormat, Tensor, TensorMapTrait, TensorMemory};

        const W: usize = 16;
        const H: usize = 4;
        const POISON: u8 = 0xAB;
        for fmt in [PixelFormat::Rgba, PixelFormat::Rgb, PixelFormat::Grey] {
            for wide in [W, W + 8] {
                let bpp = fmt.channels();
                let row_bytes = W * bpp;
                let stride = wide * bpp;
                // Allocates at the padded width, then declares the logical
                // geometry and the pitch.
                let mut t = Tensor::<u8>::image(
                    wide,
                    H,
                    fmt,
                    Some(TensorMemory::Mem),
                    CpuAccess::ReadWrite,
                )
                .unwrap();
                t.configure_image(W, H, fmt).unwrap();
                t.set_row_stride(stride).unwrap();
                // A distinct byte per pixel offset, poison in the padding.
                let mut before = vec![POISON; stride * H];
                for y in 0..H {
                    for i in 0..row_bytes {
                        before[y * stride + i] = (y * 31 + i * 7) as u8;
                    }
                }
                t.map_mut().unwrap().as_mut_slice().copy_from_slice(&before);

                apply_int8_xor_bias_rows(&mut t, fmt).unwrap();

                let after = t.map_read().unwrap().as_slice().to_vec();
                for y in 0..H {
                    // The flat helper on one row is the per-row oracle.
                    let mut want = before[y * stride..y * stride + row_bytes].to_vec();
                    apply_int8_xor_bias(&mut want, fmt);
                    assert_eq!(
                        &after[y * stride..y * stride + row_bytes],
                        &want[..],
                        "{fmt:?} at a {stride} B pitch: row {y} pixels"
                    );
                    assert!(
                        after[y * stride + row_bytes..(y + 1) * stride]
                            .iter()
                            .all(|&b| b == POISON),
                        "{fmt:?} at a {stride} B pitch: row {y} padding was biased"
                    );
                }
            }
        }
    }

    // ---------------------------------------------------------------------
    // Super-white luma
    // ---------------------------------------------------------------------

    /// Decodes a uniform 64x64 limited-range BT.601 NV12 frame and returns
    /// its first RGBA pixel.
    fn decode_uniform_nv12(y: u8, u: u8, v: u8) -> [u8; 4] {
        use edgefirst_tensor::{
            ColorEncoding, ColorRange, Colorimetry, DType, PixelFormat, TensorDyn, TensorMapTrait,
            TensorTrait,
        };
        let (w, h) = (64usize, 64usize);
        let mut src = TensorDyn::image(
            w,
            h,
            PixelFormat::Nv12,
            DType::U8,
            None,
            edgefirst_tensor::CpuAccess::ReadWrite,
        )
        .unwrap();
        {
            let bound = src.as_typed::<u8>().unwrap();
            let stride = bound.effective_row_stride().unwrap();
            let mut m = bound.map().unwrap();
            let buf = m.as_mut_slice();
            for r in 0..h {
                for c in 0..w {
                    buf[r * stride + c] = y;
                }
            }
            let uv = stride * h;
            for r in 0..h / 2 {
                for c in 0..w / 2 {
                    buf[uv + r * stride + c * 2] = u;
                    buf[uv + r * stride + c * 2 + 1] = v;
                }
            }
        }
        src.set_colorimetry(Some(
            Colorimetry::default()
                .with_encoding(ColorEncoding::Bt601)
                .with_range(ColorRange::Limited),
        ));
        let mut dst = TensorDyn::image(
            w,
            h,
            PixelFormat::Rgba,
            DType::U8,
            None,
            edgefirst_tensor::CpuAccess::ReadWrite,
        )
        .unwrap();
        CPUProcessor::new().convert(&src, &mut dst, None).unwrap();
        let bound = dst.as_typed::<u8>().unwrap();
        let m = bound.map().unwrap();
        let p = m.as_slice();
        [p[0], p[1], p[2], p[3]]
    }

    /// Exact BT.601 limited-range YUV→RGB, unclamped.
    fn exact_bt601_limited(y: f64, u: f64, v: f64) -> [f64; 3] {
        let l = 1.164383 * (y - 16.0);
        [
            l + 1.596027 * (v - 128.0),
            l - 0.391762 * (u - 128.0) - 0.812968 * (v - 128.0),
            l + 2.017232 * (u - 128.0),
        ]
    }

    /// Convert a `w`x`h` NV12 or NV24 frame built from `y_at` and `uv_at` to
    /// RGB on the CPU and return the largest channel error against `exact`
    /// (clamped and rounded) and each channel's mean signed error.
    fn nv_matrix_error(
        fmt: edgefirst_tensor::PixelFormat,
        (w, h): (usize, usize),
        cm: edgefirst_tensor::Colorimetry,
        y_at: impl Fn(usize, usize) -> u8,
        uv_at: impl Fn(usize, usize) -> (u8, u8),
        exact: impl Fn(f64, f64, f64) -> [f64; 3],
    ) -> (u8, [f64; 3]) {
        use edgefirst_tensor::{DType, PixelFormat, TensorDyn, TensorMapTrait, TensorTrait};
        let shift = usize::from(fmt == PixelFormat::Nv12);
        let mut src = TensorDyn::image(
            w,
            h,
            fmt,
            DType::U8,
            None,
            edgefirst_tensor::CpuAccess::ReadWrite,
        )
        .unwrap();
        {
            let bound = src.as_typed::<u8>().unwrap();
            let stride = bound.effective_row_stride().unwrap();
            let uv_stride = stride << (1 - shift);
            let mut m = bound.map().unwrap();
            let buf = m.as_mut_slice();
            for r in 0..h {
                for c in 0..w {
                    buf[r * stride + c] = y_at(c, r);
                }
            }
            for r in 0..h >> shift {
                for c in 0..w >> shift {
                    let (u, v) = uv_at(c, r);
                    buf[stride * h + r * uv_stride + c * 2] = u;
                    buf[stride * h + r * uv_stride + c * 2 + 1] = v;
                }
            }
        }
        src.set_colorimetry(Some(cm));
        let mut dst = TensorDyn::image(
            w,
            h,
            PixelFormat::Rgb,
            DType::U8,
            None,
            edgefirst_tensor::CpuAccess::ReadWrite,
        )
        .unwrap();
        CPUProcessor::new().convert(&src, &mut dst, None).unwrap();
        let bound = dst.as_typed::<u8>().unwrap();
        let stride = bound.effective_row_stride().unwrap();
        let m = bound.map().unwrap();
        let out = m.as_slice();
        let (mut max, mut sum) = (0u8, [0i64; 3]);
        for r in 0..h {
            for c in 0..w {
                let (u, v) = uv_at(c >> shift, r >> shift);
                let want = exact(f64::from(y_at(c, r)), f64::from(u), f64::from(v));
                for ch in 0..3 {
                    let want = want[ch].clamp(0.0, 255.0).round() as u8;
                    let have = out[r * stride + c * 3 + ch];
                    max = max.max(have.abs_diff(want));
                    sum[ch] += i64::from(have) - i64::from(want);
                }
            }
        }
        (max, sum.map(|s| s as f64 / (w * h) as f64))
    }

    /// Exact YUV→RGB for luma/red/blue weights `kr`, `kb`, unclamped except
    /// that limited-range footroom luma (Y < 16) is black.
    fn exact_yuv(kr: f64, kb: f64, limited: bool) -> impl Fn(f64, f64, f64) -> [f64; 3] {
        let (ys, cs) = if limited {
            (255.0 / 219.0, 255.0 / 224.0)
        } else {
            (1.0, 1.0)
        };
        let kg = 1.0 - kr - kb;
        move |y, u, v| {
            let l = (y - if limited { 16.0 } else { 0.0 }).max(0.0) * ys;
            let (cb, cr) = ((u - 128.0) * cs, (v - 128.0) * cs);
            [
                l + 2.0 * (1.0 - kr) * cr,
                l - 2.0 * kb * (1.0 - kb) / kg * cb - 2.0 * kr * (1.0 - kr) / kg * cr,
                l + 2.0 * (1.0 - kb) * cb,
            ]
        }
    }

    /// The CPU YUV→RGB conversion stays within one level of the exact matrix with no systematic bias.
    #[test]
    fn nv_to_rgb_is_unbiased_and_within_one_level() {
        use edgefirst_tensor::{ColorEncoding, ColorRange, Colorimetry, PixelFormat};
        let grid = |i: usize| (i % 52 * 5) as u8;
        for (enc, kr, kb) in [
            (ColorEncoding::Bt601, 0.299, 0.114),
            (ColorEncoding::Bt709, 0.2126, 0.0722),
        ] {
            for range in [ColorRange::Full, ColorRange::Limited] {
                let cm = Colorimetry::default().with_encoding(enc).with_range(range);
                let exact = exact_yuv(kr, kb, range == ColorRange::Limited);
                // NV24: x is luma, y walks the chroma grid.
                let nv24 = nv_matrix_error(
                    PixelFormat::Nv24,
                    (256, 52 * 52),
                    cm,
                    |x, _| x as u8,
                    |_, y| (grid(y), grid(y / 52)),
                    &exact,
                );
                // NV12: each 2x2 block shares one chroma sample.
                let nv12 = nv_matrix_error(
                    PixelFormat::Nv12,
                    (256, 2 * 52 * 52),
                    cm,
                    |x, y| (x as u8).wrapping_add(y as u8 % 2 * 128),
                    |_, cy| (grid(cy), grid(cy / 52)),
                    &exact,
                );
                for (fmt, (max, bias)) in [("NV24", nv24), ("NV12", nv12)] {
                    assert!(max <= 1, "{fmt} {enc:?} {range:?}: max error {max}");
                    for (ch, b) in bias.iter().enumerate() {
                        assert!(
                            b.abs() < 0.01,
                            "{fmt} {enc:?} {range:?}: channel {ch} bias {b:+.4}"
                        );
                    }
                }
            }
        }
    }

    /// Super-white luma (Y above 235) decodes to the clamped exact value.
    #[test]
    fn super_white_luma_never_decodes_to_black() {
        // Saturated chroma both ways, and a blue that only just saturates.
        for (y, u, v) in [(255u8, 255u8, 255u8), (255, 16, 16), (249, 253, 115)] {
            let got = decode_uniform_nv12(y, u, v);
            let exact = exact_bt601_limited(y as f64, u as f64, v as f64);
            for (ch, name) in ["red", "green", "blue"].iter().enumerate() {
                let want = exact[ch].clamp(0.0, 255.0);
                let have = got[ch] as f64;
                assert!(
                    (have - want).abs() <= 1.0,
                    "Y={y} U={u} V={v}: {name} decoded {have}, exact {want:.2}"
                );
            }
        }
    }

    /// The content region reported for a convert is the rectangle the convert paints.
    #[test]
    fn content_region_matches_painted_rect() {
        use crate::{ConvertOptions, LetterboxPlacement};
        use edgefirst_tensor::CpuAccess;
        let pad = [10u8, 20, 30];
        let mut cpu = CPUProcessor::new();
        for (sw, sh) in [(641, 479), (479, 641), (1, 3), (3, 1)] {
            let src = TensorDyn::image(
                sw,
                sh,
                PixelFormat::Rgb,
                DType::U8,
                Some(TensorMemory::Mem),
                CpuAccess::ReadWrite,
            )
            .unwrap();
            src.as_u8().unwrap().map().unwrap().as_mut_slice().fill(200);
            for (dw, dh) in [(640, 640), (320, 256)] {
                for placement in [LetterboxPlacement::Center, LetterboxPlacement::TopLeft] {
                    let opts = ConvertOptions::new()
                        .letterbox([pad[0], pad[1], pad[2], 255])
                        .placement(placement);
                    let mut dst = TensorDyn::image(
                        dw,
                        dh,
                        PixelFormat::Rgb,
                        DType::U8,
                        Some(TensorMemory::Mem),
                        CpuAccess::ReadWrite,
                    )
                    .unwrap();
                    cpu.convert(&src, &mut dst, Some(opts)).unwrap();
                    let map = dst.as_u8().unwrap().map_read().unwrap();
                    let px = map.as_slice();
                    let (mut x0, mut y0, mut x1, mut y1) = (usize::MAX, usize::MAX, 0, 0);
                    for y in 0..dh {
                        for x in 0..dw {
                            let i = (y * dw + x) * 3;
                            if px[i..i + 3] != pad {
                                (x0, y0) = (x0.min(x), y0.min(y));
                                (x1, y1) = (x1.max(x + 1), y1.max(y + 1));
                            }
                        }
                    }
                    let painted = Region::new(x0, y0, x1 - x0, y1 - y0);
                    let region = opts.content_region(sw, sh, dw, dh).unwrap();
                    assert_eq!(
                        region.pixels, painted,
                        "{sw}x{sh} -> {dw}x{dh} {placement:?}"
                    );
                }
            }
        }
        let stretch = ConvertOptions::new()
            .content_region(641, 479, 640, 640)
            .unwrap();
        assert!(stretch.is_full());
    }

    #[test]
    fn from_spec_matches_manual_options() {
        use crate::{
            ConvertOptions, FitMode, Interpolation, LetterboxPlacement, Normalization, PadSpace,
            PreprocessSpec,
        };
        let mut spec = PreprocessSpec::new();
        assert_eq!(
            ConvertOptions::from_spec(&spec).unwrap(),
            ConvertOptions::new()
                .letterbox([114, 114, 114, 255])
                .interpolation(Interpolation::Bilinear)
        );
        spec.normalization = Normalization::imagenet();
        spec.placement = LetterboxPlacement::TopLeft;
        spec.pad = [0.0; 3];
        spec.pad_space = PadSpace::Normalized;
        let manual = ConvertOptions::new()
            .letterbox([0, 0, 0, 255])
            .placement(LetterboxPlacement::TopLeft)
            .pad_output([0.0, 0.0, 0.0, 1.0])
            .normalization(Normalization::imagenet())
            .interpolation(Interpolation::Bilinear);
        assert_eq!(ConvertOptions::from_spec(&spec).unwrap(), manual);
        spec.fit = FitMode::Stretch;
        assert_eq!(
            ConvertOptions::from_spec(&spec).unwrap(),
            ConvertOptions::new()
                .normalization(Normalization::imagenet())
                .interpolation(Interpolation::Bilinear)
        );
        let rotated = ConvertOptions::new().rotation(Rotation::Clockwise90);
        assert_eq!(
            rotated.with_spec(&spec).unwrap().get_rotation(),
            Rotation::Clockwise90
        );
        spec.fit = FitMode::Letterbox;
        spec.pad_space = PadSpace::Pixel;
        spec.pad = [300.0; 3];
        assert!(matches!(
            ConvertOptions::from_spec(&spec),
            Err(Error::InvalidOptions(_))
        ));
    }

    #[test]
    fn unsupported_interpolation_falls_back_to_bilinear() {
        use crate::Interpolation;
        assert_eq!(
            crate::runnable_interpolation(Interpolation::Bicubic),
            Interpolation::Bilinear
        );
        assert_eq!(
            crate::runnable_interpolation(Interpolation::Area),
            Interpolation::Bilinear
        );
        assert_eq!(
            crate::runnable_interpolation(Interpolation::Nearest),
            Interpolation::Nearest
        );
        let opts = crate::ConvertOptions::from_spec(&crate::PreprocessSpec::new()).unwrap();
        assert_eq!(
            opts,
            crate::ConvertOptions::new()
                .letterbox([114, 114, 114, 255])
                .interpolation(Interpolation::Bilinear)
        );
    }

    // =========================================================================
    // Box outline width and colour alpha
    // =========================================================================

    #[test]
    fn test_box_line_width_is_honoured() {
        use crate::test_support::{box_drawn_columns, box_drawn_rows, box_expected_span};
        let mut cpu = CPUProcessor::new();
        // 8 reaches the left and top borders and 9 the right and bottom
        // ones; 12 and 40 cross them all.
        for width in [1u32, 3, 6, 7, 8, 9, 12, 40] {
            let flat = crate::test_support::draw_box_fixture(
                &mut cpu,
                [200, 100, 50, 255],
                Some(width),
                1.0,
            );
            let mid = crate::test_support::BOX_DIM / 2;
            let expected = box_expected_span(width as usize);
            assert_eq!(box_drawn_columns(&flat, mid), expected, "width {width} row");
            assert_eq!(box_drawn_rows(&flat, mid), expected, "width {width} column");
        }
    }

    #[test]
    fn test_box_line_width_defaults_to_three() {
        let mut cpu = CPUProcessor::new();
        let flat = crate::test_support::draw_box_fixture(&mut cpu, [200, 100, 50, 255], None, 1.0);
        let mid = crate::test_support::BOX_DIM / 2;
        assert_eq!(
            crate::test_support::box_drawn_columns(&flat, mid),
            crate::test_support::box_expected_span(crate::DEFAULT_BOX_LINE_WIDTH as usize)
        );
    }

    #[test]
    fn test_box_line_width_rejects_zero() {
        let mut cpu = CPUProcessor::new();
        cpu.set_box_line_width(5).unwrap();
        let err = cpu.set_box_line_width(0).unwrap_err();
        assert!(matches!(err, Error::InvalidOptions(_)), "{err:?}");
        let flat = crate::test_support::draw_box_fixture(&mut cpu, [200, 100, 50, 255], None, 1.0);
        let mid = crate::test_support::BOX_DIM / 2;
        assert_eq!(
            crate::test_support::box_drawn_columns(&flat, mid),
            crate::test_support::box_expected_span(5),
            "a rejected width must leave the previous one in place"
        );
    }

    #[test]
    fn test_box_colour_alpha() {
        use crate::test_support::{box_blend, box_expected_span, box_px, BOX_BG, BOX_DIM};
        let color = [200, 100, 50, 255];
        let mid = BOX_DIM / 2;
        let mut cpu = CPUProcessor::new();

        // Opaque: the colour itself, the destination alpha untouched.
        let flat = crate::test_support::draw_box_fixture(&mut cpu, color, Some(3), 1.0);
        let left = box_expected_span(3)[0];
        assert_eq!(box_px(&flat, left, mid), [200, 100, 50, BOX_BG[3]]);
        assert_eq!(
            box_px(&flat, mid, mid),
            BOX_BG,
            "box interior must not be drawn"
        );

        // Translucent: every outline pixel gets one blend with the colour's alpha.
        let translucent = [200, 100, 50, 128];
        let want = box_blend(translucent, 128);
        let flat = crate::test_support::draw_box_fixture(&mut cpu, translucent, Some(3), 1.0);
        for x in box_expected_span(3) {
            assert_eq!(box_px(&flat, x, mid), want, "row pixel x={x}");
            assert_eq!(box_px(&flat, mid, x), want, "column pixel y={x}");
        }
        let corner = BOX_DIM / 4 - 1;
        assert_eq!(box_px(&flat, corner, corner), want, "top-left corner");
        assert_eq!(
            box_px(&flat, BOX_DIM * 3 / 4, corner),
            want,
            "top-right corner"
        );

        // Opacity scales the colour's alpha.
        let flat = crate::test_support::draw_box_fixture(&mut cpu, color, Some(3), 0.5);
        assert_eq!(box_px(&flat, left, mid), box_blend(color, 128));

        // Alpha 0 draws nothing.
        let flat = crate::test_support::draw_box_fixture(&mut cpu, [200, 100, 50, 0], Some(3), 1.0);
        assert!(crate::test_support::box_drawn_columns(&flat, mid).is_empty());
    }

    #[test]
    fn full_range_saturated_red_saturates_yuyv_chroma() -> Result<()> {
        let cp = crate::cpu::ColorParams::uniform(
            edgefirst_tensor::ColorEncoding::Bt601,
            edgefirst_tensor::ColorRange::Full,
        );
        assert_eq!(
            CPUProcessor::rgba_to_yuyv([255, 0, 0, 255], cp),
            [76, 85, 76, 255]
        );

        let (w, h) = (8, 2);
        let src = Tensor::<u8>::image(
            w,
            h,
            PixelFormat::Rgba,
            Some(TensorMemory::Mem),
            edgefirst_tensor::CpuAccess::ReadWrite,
        )?;
        for px in src.map_mut()?.as_mut_slice().chunks_exact_mut(4) {
            px.copy_from_slice(&[255, 0, 0, 255]);
        }
        let src = TensorDyn::from(src);
        let mut dst = TensorDyn::image(
            w,
            h,
            PixelFormat::Yuyv,
            DType::U8,
            Some(TensorMemory::Mem),
            edgefirst_tensor::CpuAccess::ReadWrite,
        )?;
        dst.set_colorimetry(Some(
            edgefirst_tensor::Colorimetry::default()
                .with_encoding(edgefirst_tensor::ColorEncoding::Bt601)
                .with_range(edgefirst_tensor::ColorRange::Full),
        ));
        CPUProcessor::new().convert(&src, &mut dst, None)?;
        let map = dst.as_u8().unwrap().map_read()?;
        for macropixel in map.as_slice().chunks_exact(4) {
            assert_eq!(macropixel, [76, 85, 76, 255]);
        }
        Ok(())
    }

    fn bgra_test_pixel(x: usize, y: usize) -> [u8; 4] {
        [
            ((x * 5 + y) % 256) as u8,
            ((y * 7 + x * 3) % 256) as u8,
            ((x * 11 + y * 13) % 256) as u8,
            (255 - (x + y) % 200) as u8,
        ]
    }

    /// A tight RGBA tensor and a BGRA tensor (optionally padded to `stride`
    /// bytes per row) holding the same pixels.
    fn rgba_and_bgra_sources(
        w: usize,
        h: usize,
        stride: Option<usize>,
        pixel: impl Fn(usize, usize) -> [u8; 4],
    ) -> Result<(TensorDyn, TensorDyn)> {
        let rgba = Tensor::<u8>::image(
            w,
            h,
            PixelFormat::Rgba,
            Some(TensorMemory::Mem),
            edgefirst_tensor::CpuAccess::ReadWrite,
        )?;
        let stride = stride.unwrap_or(w * 4);
        let mut bgra = Tensor::<u8>::image(
            stride / 4,
            h,
            PixelFormat::Bgra,
            Some(TensorMemory::Mem),
            edgefirst_tensor::CpuAccess::ReadWrite,
        )?;
        bgra.configure_image(w, h, PixelFormat::Bgra)?;
        bgra.set_row_stride(stride)?;
        {
            let mut rmap = rgba.map_mut()?;
            let mut bmap = bgra.map_mut()?;
            bmap.as_mut_slice().fill(0xEE);
            for y in 0..h {
                for x in 0..w {
                    let [r, g, b, a] = pixel(x, y);
                    rmap.as_mut_slice()[(y * w + x) * 4..][..4].copy_from_slice(&[r, g, b, a]);
                    bmap.as_mut_slice()[y * stride + x * 4..][..4].copy_from_slice(&[b, g, r, a]);
                }
            }
        }
        Ok((TensorDyn::from(rgba), TensorDyn::from(bgra)))
    }

    /// The destination bytes of a tensor of any element type.
    fn dyn_dst_bytes(t: &TensorDyn) -> Vec<u8> {
        match t.dtype() {
            DType::U8 => t.as_u8().unwrap().map_read().unwrap().as_slice().to_vec(),
            DType::I8 => t
                .as_typed::<i8>()
                .unwrap()
                .map_read()
                .unwrap()
                .as_slice()
                .iter()
                .map(|&v| v as u8)
                .collect(),
            DType::F16 => t
                .as_typed::<half::f16>()
                .unwrap()
                .map_read()
                .unwrap()
                .as_slice()
                .iter()
                .flat_map(|v| v.to_bits().to_le_bytes())
                .collect(),
            DType::F32 => t
                .as_typed::<f32>()
                .unwrap()
                .map_read()
                .unwrap()
                .as_slice()
                .iter()
                .flat_map(|v| v.to_bits().to_le_bytes())
                .collect(),
            other => panic!("unexpected dtype {other:?}"),
        }
    }

    /// Converts `src` into a fresh `dst_w`x`dst_h` destination, returning its
    /// bytes, or `None` when the conversion is not supported.
    fn convert_to_bytes(
        cpu: &mut CPUProcessor,
        src: &TensorDyn,
        dst_w: usize,
        dst_h: usize,
        fmt: PixelFormat,
        dtype: DType,
        options: Option<crate::ConvertOptions>,
    ) -> Option<Vec<u8>> {
        let mut dst = TensorDyn::image(
            dst_w,
            dst_h,
            fmt,
            dtype,
            mem(),
            edgefirst_tensor::CpuAccess::ReadWrite,
        )
        .ok()?;
        dyn_fill_zero(&mut dst);
        cpu.convert(src, &mut dst, options).ok()?;
        Some(dyn_dst_bytes(&dst))
    }

    fn dyn_fill_zero(t: &mut TensorDyn) {
        if let Some(u) = t.as_u8() {
            u.map_mut().unwrap().as_mut_slice().fill(0);
        } else if let Some(i) = t.as_typed::<i8>() {
            i.map_mut().unwrap().as_mut_slice().fill(0);
        } else if let Some(h) = t.as_typed::<half::f16>() {
            h.map_mut()
                .unwrap()
                .as_mut_slice()
                .fill(half::f16::from_f32(0.0));
        } else if let Some(f) = t.as_typed::<f32>() {
            f.map_mut().unwrap().as_mut_slice().fill(0.0);
        }
    }

    const BGRA_TEST_DTYPES: [DType; 4] = [DType::U8, DType::I8, DType::F16, DType::F32];

    #[test]
    fn bgra_source_matches_rgba_for_every_destination() -> Result<()> {
        let (rgba, bgra) = rgba_and_bgra_sources(64, 48, None, bgra_test_pixel)?;
        let mut cpu = CPUProcessor::new();
        let mut compared = 0;
        for fmt in [
            PixelFormat::Rgb,
            PixelFormat::Rgba,
            PixelFormat::Grey,
            PixelFormat::Yuyv,
            PixelFormat::PlanarRgb,
            PixelFormat::PlanarRgba,
            PixelFormat::Nv16,
            PixelFormat::Nv12,
            PixelFormat::Bgra,
        ] {
            for dtype in BGRA_TEST_DTYPES {
                let want = convert_to_bytes(&mut cpu, &rgba, 64, 48, fmt, dtype, None);
                let got = convert_to_bytes(&mut cpu, &bgra, 64, 48, fmt, dtype, None);
                assert_eq!(got, want, "BGRA to {fmt} {dtype:?}");
                compared += 1;
            }
        }
        assert_eq!(compared, 36);
        Ok(())
    }

    #[test]
    fn bgra_source_matches_rgba_with_crop_letterbox_flip_rotation() -> Result<()> {
        let (rgba, bgra) = rgba_and_bgra_sources(640, 480, None, bgra_test_pixel)?;
        let cases: [(&str, crate::ConvertOptions); 4] = [
            (
                "source crop",
                crate::ConvertOptions::new()
                    .crop(Crop::new().with_source(Some(Region::new(100, 50, 320, 240)))),
            ),
            (
                "letterbox",
                crate::ConvertOptions::new().crop(Crop::letterbox([114, 114, 114, 255])),
            ),
            (
                "horizontal flip",
                crate::ConvertOptions::new().flip(Flip::Horizontal),
            ),
            (
                "rotation",
                crate::ConvertOptions::new().rotation(Rotation::Clockwise90),
            ),
        ];
        let mut cpu = CPUProcessor::new();
        for (name, options) in cases {
            for (fmt, dtype) in [
                (PixelFormat::Rgba, DType::U8),
                (PixelFormat::Nv12, DType::U8),
                (PixelFormat::PlanarRgb, DType::U8),
                (PixelFormat::PlanarRgb, DType::F16),
            ] {
                let want = convert_to_bytes(&mut cpu, &rgba, 320, 320, fmt, dtype, Some(options))
                    .unwrap_or_else(|| panic!("RGBA to {fmt} {dtype:?} ({name})"));
                let got = convert_to_bytes(&mut cpu, &bgra, 320, 320, fmt, dtype, Some(options))
                    .unwrap_or_else(|| panic!("BGRA to {fmt} {dtype:?} ({name})"));
                assert_eq!(got, want, "BGRA to {fmt} {dtype:?} ({name})");
            }
        }
        Ok(())
    }

    #[test]
    fn bgra_source_with_padded_stride_matches_rgba() -> Result<()> {
        let (rgba, bgra) = rgba_and_bgra_sources(100, 40, Some(448), bgra_test_pixel)?;
        let mut cpu = CPUProcessor::new();
        for fmt in [PixelFormat::Rgba, PixelFormat::Nv12] {
            let want = convert_to_bytes(&mut cpu, &rgba, 100, 40, fmt, DType::U8, None).unwrap();
            let got = convert_to_bytes(&mut cpu, &bgra, 100, 40, fmt, DType::U8, None).unwrap();
            assert_eq!(got, want, "padded BGRA to {fmt}");
        }
        Ok(())
    }

    #[test]
    fn bgra_scratch_follows_size_changes() -> Result<()> {
        let mut cpu = CPUProcessor::new();
        for (w, h, seed) in [(64, 48, 0), (32, 16, 0), (64, 48, 9)] {
            let pixel = move |x: usize, y: usize| bgra_test_pixel(x + seed, y + seed);
            let (rgba, bgra) = rgba_and_bgra_sources(w, h, None, pixel)?;
            let want = convert_to_bytes(&mut cpu, &rgba, w, h, PixelFormat::Rgba, DType::U8, None)
                .unwrap();
            let got = convert_to_bytes(&mut cpu, &bgra, w, h, PixelFormat::Rgba, DType::U8, None)
                .unwrap();
            assert_eq!(got, want, "{w}x{h} seed {seed}");
        }
        Ok(())
    }

    #[test]
    fn bgra_to_bgra_resize_keeps_channel_order() -> Result<()> {
        let (_, bgra) = rgba_and_bgra_sources(64, 48, None, bgra_test_pixel)?;
        let mut cpu = CPUProcessor::new();
        let got =
            convert_to_bytes(&mut cpu, &bgra, 32, 24, PixelFormat::Bgra, DType::U8, None).unwrap();
        let mut want =
            convert_to_bytes(&mut cpu, &bgra, 32, 24, PixelFormat::Rgba, DType::U8, None).unwrap();
        for px in want.as_chunks_mut::<4>().0 {
            px.swap(0, 2);
        }
        assert_eq!(got, want);
        Ok(())
    }

    #[test]
    fn bgra_to_bgra_resize_into_write_only_destination() -> Result<()> {
        let (_, bgra) = rgba_and_bgra_sources(64, 48, None, bgra_test_pixel)?;
        let mut cpu = CPUProcessor::new();
        let want =
            convert_to_bytes(&mut cpu, &bgra, 32, 24, PixelFormat::Bgra, DType::U8, None).unwrap();
        let mut dst = TensorDyn::image(
            32,
            24,
            PixelFormat::Bgra,
            DType::U8,
            mem(),
            edgefirst_tensor::CpuAccess::Write,
        )?;
        cpu.convert(&bgra, &mut dst, None)?;
        let map = dst.as_u8().unwrap().map_write()?;
        assert_eq!(map.as_slice(), want.as_slice());
        Ok(())
    }

    #[test]
    fn bgra_to_unsupported_destination_names_bgra() -> Result<()> {
        let (_, bgra) = rgba_and_bgra_sources(64, 48, None, bgra_test_pixel)?;
        let mut cpu = CPUProcessor::new();
        let mut dst = TensorDyn::image(
            64,
            48,
            PixelFormat::Vyuy,
            DType::U8,
            mem(),
            edgefirst_tensor::CpuAccess::ReadWrite,
        )?;
        let err = cpu.convert(&bgra, &mut dst, None).unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("BGRA") && !msg.contains("RGBA"), "{msg}");
        Ok(())
    }
}
