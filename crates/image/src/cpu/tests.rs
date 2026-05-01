// SPDX-FileCopyrightText: Copyright 2025 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod cpu_tests {

    use crate::{CPUProcessor, Crop, Error, Flip, ImageProcessorTrait, Rect, Result, Rotation};
    use edgefirst_decoder::DetectBox;
    use edgefirst_tensor::{
        DType, PixelFormat, Tensor, TensorDyn, TensorMapTrait, TensorMemory, TensorTrait,
    };

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

    fn compare_images_convert_to_rgb(
        img1: &TensorDyn,
        img2: &TensorDyn,
        threshold: f64,
        name: &str,
    ) {
        assert_eq!(img1.height(), img2.height(), "Heights differ");
        assert_eq!(img1.width(), img2.width(), "Widths differ");

        // Try converting both to RGB for comparison. If conversion is not
        // supported (e.g. from PlanarRgb), fall back to raw byte comparison.
        let mut converter = CPUProcessor::default();
        let w = img1.width().unwrap();
        let h = img1.height().unwrap();

        let (img_rgb1, img_rgb2) = {
            let mut rgb1 = TensorDyn::image(w, h, PixelFormat::Rgb, DType::U8, None).unwrap();
            let mut rgb2 = TensorDyn::image(w, h, PixelFormat::Rgb, DType::U8, None).unwrap();
            let r1 = converter.convert(
                img1,
                &mut rgb1,
                crate::Rotation::None,
                crate::Flip::None,
                crate::Crop::default(),
            );
            let r2 = converter.convert(
                img2,
                &mut rgb2,
                crate::Rotation::None,
                crate::Flip::None,
                crate::Crop::default(),
            );
            if r1.is_err() || r2.is_err() {
                // Fallback: compare raw bytes as greyscale strip
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

    fn load_bytes_to_tensor(
        width: usize,
        height: usize,
        format: PixelFormat,
        memory: Option<TensorMemory>,
        bytes: &[u8],
    ) -> Result<TensorDyn, Error> {
        log::debug!("Current function is {}", function!());
        let src = TensorDyn::image(width, height, format, DType::U8, memory)?;
        src.as_u8().unwrap().map()?.as_mut_slice()[0..bytes.len()].copy_from_slice(bytes);
        Ok(src)
    }

    macro_rules! generate_conversion_tests {
        (
        $src_fmt:expr,  $src_file:expr, $dst_fmt:expr, $dst_file:expr
    ) => {{
            // Load source
            let src = load_bytes_to_tensor(
                1280,
                720,
                $src_fmt,
                None,
                include_bytes!(concat!(
                    env!("CARGO_MANIFEST_DIR"),
                    "/../../testdata/",
                    $src_file
                )),
            )?;

            // Load destination reference
            let dst = load_bytes_to_tensor(
                1280,
                720,
                $dst_fmt,
                None,
                include_bytes!(concat!(
                    env!("CARGO_MANIFEST_DIR"),
                    "/../../testdata/",
                    $dst_file
                )),
            )?;

            let mut converter = CPUProcessor::default();

            let mut converted = TensorDyn::image(
                src.width().unwrap(),
                src.height().unwrap(),
                dst.format().unwrap(),
                DType::U8,
                None,
            )?;

            converter.convert(
                &src,
                &mut converted,
                Rotation::None,
                Flip::None,
                Crop::default(),
            )?;

            compare_images_convert_to_rgb(&dst, &converted, 0.99, function!());

            Ok(())
        }};
    }

    macro_rules! generate_conversion_tests_greyscale {
        (
        $src_fmt:expr,  $src_file:expr, $dst_fmt:expr, $dst_file:expr
    ) => {{
            // Load source
            let src = load_bytes_to_tensor(
                1280,
                720,
                $src_fmt,
                None,
                include_bytes!(concat!(
                    env!("CARGO_MANIFEST_DIR"),
                    "/../../testdata/",
                    $src_file
                )),
            )?;

            // Load destination reference
            let dst = load_bytes_to_tensor(
                1280,
                720,
                $dst_fmt,
                None,
                include_bytes!(concat!(
                    env!("CARGO_MANIFEST_DIR"),
                    "/../../testdata/",
                    $dst_file
                )),
            )?;

            let mut converter = CPUProcessor::default();

            let mut converted = TensorDyn::image(
                src.width().unwrap(),
                src.height().unwrap(),
                dst.format().unwrap(),
                DType::U8,
                None,
            )?;

            converter.convert(
                &src,
                &mut converted,
                Rotation::None,
                Flip::None,
                Crop::default(),
            )?;

            compare_images_convert_to_grey(&dst, &converted, 0.97, function!());

            Ok(())
        }};
    }

    // let mut dsts = [yuyv, rgb, rgba, grey, nv16, planar_rgb, planar_rgba];

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
        // Load source
        let src = load_bytes_to_tensor(2, 1, PixelFormat::Rgb, None, &[0, 0, 0, 255, 255, 255])?;

        let mut converter = CPUProcessor::new_nearest();

        let converted = TensorDyn::image(4, 1, PixelFormat::Rgb, DType::U8, None)?;
        let src_dyn = src;
        let mut converted_dyn = converted;

        converter.convert(
            &src_dyn,
            &mut converted_dyn,
            Rotation::None,
            Flip::None,
            Crop::default(),
        )?;

        assert_eq!(
            &converted_dyn.as_u8().unwrap().map()?.as_slice(),
            &[0, 0, 0, 0, 0, 0, 255, 255, 255, 255, 255, 255]
        );

        Ok(())
    }

    #[test]
    fn test_cpu_rotate_cw() -> Result<()> {
        // Load source
        let src = load_bytes_to_tensor(
            2,
            2,
            PixelFormat::Rgba,
            None,
            &[0, 0, 0, 255, 1, 1, 1, 255, 2, 2, 2, 255, 3, 3, 3, 255],
        )?;

        let mut converter = CPUProcessor::default();

        let converted = TensorDyn::image(4, 4, PixelFormat::Rgba, DType::U8, None)?;
        let src_dyn = src;
        let mut converted_dyn = converted;

        converter.convert(
            &src_dyn,
            &mut converted_dyn,
            Rotation::Clockwise90,
            Flip::None,
            Crop::default(),
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
        // Load source
        let src = load_bytes_to_tensor(
            2,
            2,
            PixelFormat::Rgba,
            None,
            &[0, 0, 0, 255, 1, 1, 1, 255, 2, 2, 2, 255, 3, 3, 3, 255],
        )?;

        let mut converter = CPUProcessor::default();

        let converted = TensorDyn::image(4, 4, PixelFormat::Rgba, DType::U8, None)?;
        let src_dyn = src;
        let mut converted_dyn = converted;

        converter.convert(
            &src_dyn,
            &mut converted_dyn,
            Rotation::CounterClockwise90,
            Flip::None,
            Crop::default(),
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
        // Load source
        let src = load_bytes_to_tensor(
            2,
            2,
            PixelFormat::Rgba,
            None,
            &[0, 0, 0, 255, 1, 1, 1, 255, 2, 2, 2, 255, 3, 3, 3, 255],
        )?;

        let mut converter = CPUProcessor::default();

        let converted = TensorDyn::image(4, 4, PixelFormat::Rgba, DType::U8, None)?;
        let src_dyn = src;
        let mut converted_dyn = converted;

        converter.convert(
            &src_dyn,
            &mut converted_dyn,
            Rotation::Rotate180,
            Flip::None,
            Crop::default(),
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
        // Load source
        let src = load_bytes_to_tensor(
            2,
            2,
            PixelFormat::Rgba,
            None,
            &[0, 0, 0, 255, 1, 1, 1, 255, 2, 2, 2, 255, 3, 3, 3, 255],
        )?;

        let mut converter = CPUProcessor::default();

        let converted = TensorDyn::image(4, 4, PixelFormat::Rgba, DType::U8, None)?;
        let src_dyn = src;
        let mut converted_dyn = converted;

        converter.convert(
            &src_dyn,
            &mut converted_dyn,
            Rotation::None,
            Flip::Vertical,
            Crop::default(),
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
        // Load source
        let src = load_bytes_to_tensor(
            2,
            2,
            PixelFormat::Rgba,
            None,
            &[0, 0, 0, 255, 1, 1, 1, 255, 2, 2, 2, 255, 3, 3, 3, 255],
        )?;

        let mut converter = CPUProcessor::default();

        let converted = TensorDyn::image(4, 4, PixelFormat::Rgba, DType::U8, None)?;
        let src_dyn = src;
        let mut converted_dyn = converted;

        converter.convert(
            &src_dyn,
            &mut converted_dyn,
            Rotation::None,
            Flip::Horizontal,
            Crop::default(),
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
        // Load source
        let src = load_bytes_to_tensor(2, 2, PixelFormat::Grey, None, &[10, 20, 30, 40])?;

        let mut converter = CPUProcessor::default();

        let converted = TensorDyn::image(2, 2, PixelFormat::Rgba, DType::U8, None)?;
        let src_dyn = src;
        let mut converted_dyn = converted;

        converter.convert(
            &src_dyn,
            &mut converted_dyn,
            Rotation::None,
            Flip::None,
            Crop::new().with_src_rect(Some(Rect::new(0, 0, 1, 2))),
        )?;

        assert_eq!(
            converted_dyn.as_u8().unwrap().map()?.as_slice(),
            &[10, 10, 10, 255, 13, 13, 13, 255, 30, 30, 30, 255, 33, 33, 33, 255]
        );
        Ok(())
    }

    #[test]
    fn test_cpu_dst_crop() -> Result<()> {
        // Load source
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
        let mut converted_dyn = converted;

        converter.convert(
            &src_dyn,
            &mut converted_dyn,
            Rotation::None,
            Flip::None,
            Crop::new().with_dst_rect(Some(Rect::new(0, 0, 2, 1))),
        )?;

        assert_eq!(
            converted_dyn.as_u8().unwrap().map()?.as_slice(),
            &[20, 128, 21, 128, 200, 128, 200, 128]
        );
        Ok(())
    }

    #[test]
    fn test_cpu_fill_rgba() -> Result<()> {
        // Load source
        let src = load_bytes_to_tensor(1, 1, PixelFormat::Rgba, None, &[3, 3, 3, 255])?;

        let mut converter = CPUProcessor::default();

        let converted = TensorDyn::image(2, 2, PixelFormat::Rgba, DType::U8, None)?;
        let src_dyn = src;
        let mut converted_dyn = converted;

        converter.convert(
            &src_dyn,
            &mut converted_dyn,
            Rotation::None,
            Flip::None,
            Crop {
                src_rect: None,
                dst_rect: Some(Rect {
                    left: 1,
                    top: 1,
                    width: 1,
                    height: 1,
                }),
                dst_color: Some([255, 0, 0, 255]),
            },
        )?;

        assert_eq!(
            converted_dyn.as_u8().unwrap().map()?.as_slice(),
            &[255, 0, 0, 255, 255, 0, 0, 255, 255, 0, 0, 255, 3, 3, 3, 255]
        );
        Ok(())
    }

    #[test]
    fn test_cpu_fill_yuyv() -> Result<()> {
        // Load source
        let src =
            load_bytes_to_tensor(2, 1, PixelFormat::Rgba, None, &[3, 3, 3, 255, 3, 3, 3, 255])?;

        let mut converter = CPUProcessor::default();

        let converted = TensorDyn::image(2, 3, PixelFormat::Yuyv, DType::U8, None)?;
        let src_dyn = src;
        let mut converted_dyn = converted;

        converter.convert(
            &src_dyn,
            &mut converted_dyn,
            Rotation::None,
            Flip::None,
            Crop {
                src_rect: None,
                dst_rect: Some(Rect {
                    left: 0,
                    top: 1,
                    width: 2,
                    height: 1,
                }),
                dst_color: Some([255, 0, 0, 255]),
            },
        )?;

        assert_eq!(
            converted_dyn.as_u8().unwrap().map()?.as_slice(),
            &[63, 102, 63, 240, 19, 128, 19, 128, 63, 102, 63, 240]
        );
        Ok(())
    }

    #[test]
    fn test_cpu_fill_grey() -> Result<()> {
        // Load source
        let src =
            load_bytes_to_tensor(2, 1, PixelFormat::Rgba, None, &[3, 3, 3, 255, 3, 3, 3, 255])?;

        let mut converter = CPUProcessor::default();

        let converted = TensorDyn::image(2, 3, PixelFormat::Grey, DType::U8, None)?;
        let src_dyn = src;
        let mut converted_dyn = converted;

        converter.convert(
            &src_dyn,
            &mut converted_dyn,
            Rotation::None,
            Flip::None,
            Crop {
                src_rect: None,
                dst_rect: Some(Rect {
                    left: 0,
                    top: 1,
                    width: 2,
                    height: 1,
                }),
                dst_color: Some([200, 200, 200, 255]),
            },
        )?;

        assert_eq!(
            converted_dyn.as_u8().unwrap().map()?.as_slice(),
            &[200, 200, 3, 3, 200, 200]
        );
        Ok(())
    }

    #[test]
    fn test_segmentation() {
        use edgefirst_decoder::Segmentation;
        use ndarray::Array3;

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
        use edgefirst_decoder::Segmentation;
        use ndarray::Array3;

        // draw_decoded_masks fully writes dst: we must pass the camera
        // frame as a background via MaskOverlay, not as the dst canvas.
        let bg = crate::load_image(
            include_bytes!(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/../../testdata/giraffe.jpg"
            )),
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
        let expected = crate::load_image(
            include_bytes!(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/../../testdata/output_render_cpu.jpg"
            )),
            Some(PixelFormat::Rgba),
            None,
        )
        .unwrap();
        compare_images_convert_to_rgb(&image, &expected, 0.99, function!());
    }

    // =========================================================================
    // Generic Conversion Tests
    // (These tests use TensorDyn for all image representations)
    // =========================================================================

    #[test]
    fn test_convert_rgb_to_planar_rgb_generic() {
        // Create PixelFormat::Rgb source image
        let src = TensorDyn::image(4, 4, PixelFormat::Rgb, DType::U8, None).unwrap();
        {
            let mut map = src.as_u8().unwrap().map().unwrap();
            let data = map.as_mut_slice();
            // Fill with pattern: pixel 0 = [10, 20, 30], pixel 1 = [40, 50, 60], etc.
            for i in 0..16 {
                data[i * 3] = (i * 10) as u8;
                data[i * 3 + 1] = (i * 10 + 1) as u8;
                data[i * 3 + 2] = (i * 10 + 2) as u8;
            }
        }

        // Create planar PixelFormat::Rgb destination using TensorDyn
        let mut dst = TensorDyn::image(4, 4, PixelFormat::PlanarRgb, DType::U8, None).unwrap();

        {
            let mut __cv = CPUProcessor::default();
            __cv.convert(
                &src,
                &mut dst,
                crate::Rotation::None,
                crate::Flip::None,
                crate::Crop::default(),
            )
            .unwrap();
        }

        // Verify the conversion - check first few pixels of each plane
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
        // Create PixelFormat::Rgba source image
        let src = TensorDyn::image(4, 4, PixelFormat::Rgba, DType::U8, None).unwrap();
        {
            let mut map = src.as_u8().unwrap().map().unwrap();
            let data = map.as_mut_slice();
            // Fill with pattern
            for i in 0..16 {
                data[i * 4] = (i * 10) as u8; // R
                data[i * 4 + 1] = (i * 10 + 1) as u8; // G
                data[i * 4 + 2] = (i * 10 + 2) as u8; // B
                data[i * 4 + 3] = 255; // A (ignored)
            }
        }

        // Create planar PixelFormat::Rgb destination
        let mut dst = TensorDyn::image(4, 4, PixelFormat::PlanarRgb, DType::U8, None).unwrap();

        {
            let mut __cv = CPUProcessor::default();
            __cv.convert(
                &src,
                &mut dst,
                crate::Rotation::None,
                crate::Flip::None,
                crate::Crop::default(),
            )
            .unwrap();
        }

        // Verify the conversion
        let map = dst.as_u8().unwrap().map().unwrap();
        let data = map.as_slice();

        assert_eq!(data[0], 0); // R of pixel 0
        assert_eq!(data[16], 1); // G of pixel 0
        assert_eq!(data[32], 2); // B of pixel 0
    }

    #[test]
    fn test_copy_image_generic_same_format() {
        // Create source image with data
        let src = TensorDyn::image(4, 4, PixelFormat::Rgb, DType::U8, None).unwrap();
        {
            let mut map = src.as_u8().unwrap().map().unwrap();
            let data = map.as_mut_slice();
            for (i, byte) in data.iter_mut().enumerate() {
                *byte = (i % 256) as u8;
            }
        }

        // Create destination tensor
        let mut dst = TensorDyn::image(4, 4, PixelFormat::Rgb, DType::U8, None).unwrap();

        {
            let mut __cv = CPUProcessor::default();
            __cv.convert(
                &src,
                &mut dst,
                crate::Rotation::None,
                crate::Flip::None,
                crate::Crop::default(),
            )
            .unwrap();
        }

        // Verify data was copied
        let src_map = src.as_u8().unwrap().map().unwrap();
        let dst_map = dst.as_u8().unwrap().map().unwrap();
        assert_eq!(src_map.as_slice(), dst_map.as_slice());
    }

    #[test]
    fn test_convert_unsupported_format_pair() {
        // Try NV12 -> NV12 (not supported by CPU converter)
        let src = TensorDyn::image(8, 8, PixelFormat::Nv12, DType::U8, None).unwrap();
        let mut dst = TensorDyn::image(8, 8, PixelFormat::Nv12, DType::U8, None).unwrap();

        let result = {
            let mut __cv = CPUProcessor::default();
            __cv.convert(
                &src,
                &mut dst,
                crate::Rotation::None,
                crate::Flip::None,
                crate::Crop::default(),
            )
        };
        assert!(result.is_err());
        assert!(matches!(result, Err(Error::NotSupported(_))));
    }

    #[test]
    fn test_fill_image_outside_crop_generic_rgba() {
        let mut dst = TensorDyn::image(4, 4, PixelFormat::Rgba, DType::U8, None).unwrap();
        // Initialize to zeros
        dst.as_u8().unwrap().map().unwrap().as_mut_slice().fill(0);

        // Fill outside a 2x2 crop in the center with red
        let crop = Rect::new(1, 1, 2, 2);
        CPUProcessor::fill_image_outside_crop_u8(dst.as_u8_mut().unwrap(), [255, 0, 0, 255], crop)
            .unwrap();

        let map = dst.as_u8().unwrap().map().unwrap();
        let data = map.as_slice();

        // Top-left corner should be filled (red)
        assert_eq!(&data[0..4], &[255, 0, 0, 255]);

        // Center pixel (1,1) should still be zero (inside crop)
        // row=1, col=1, width=4, bytes_per_pixel=4 -> offset = (1*4 + 1) * 4 = 20
        let center_offset = 20;
        assert_eq!(&data[center_offset..center_offset + 4], &[0, 0, 0, 0]);
    }

    #[test]
    fn test_fill_image_outside_crop_generic_rgb() {
        let mut dst = TensorDyn::image(4, 4, PixelFormat::Rgb, DType::U8, None).unwrap();
        dst.as_u8().unwrap().map().unwrap().as_mut_slice().fill(0);

        let crop = Rect::new(1, 1, 2, 2);
        CPUProcessor::fill_image_outside_crop_u8(dst.as_u8_mut().unwrap(), [0, 255, 0, 255], crop)
            .unwrap();

        let map = dst.as_u8().unwrap().map().unwrap();
        let data = map.as_slice();

        // Top-left corner should be green
        assert_eq!(&data[0..3], &[0, 255, 0]);

        // Center pixel (1,1): row=1, col=1, width=4, bytes=3 -> offset = (1*4 + 1) * 3
        // = 15
        let center_offset = 15;
        assert_eq!(&data[center_offset..center_offset + 3], &[0, 0, 0]);
    }

    #[test]
    fn test_fill_image_outside_crop_generic_planar_rgb() {
        let mut dst = TensorDyn::image(4, 4, PixelFormat::PlanarRgb, DType::U8, None).unwrap();
        dst.as_u8().unwrap().map().unwrap().as_mut_slice().fill(0);

        let crop = Rect::new(1, 1, 2, 2);
        CPUProcessor::fill_image_outside_crop_u8(
            dst.as_u8_mut().unwrap(),
            [128, 64, 32, 255],
            crop,
        )
        .unwrap();

        let map = dst.as_u8().unwrap().map().unwrap();
        let data = map.as_slice();

        // For planar: R plane is [0..16], G plane is [16..32], B plane is [32..48]
        // Top-left pixel (0,0) should have R=128, G=64, B=32
        assert_eq!(data[0], 128); // R plane, pixel 0
        assert_eq!(data[16], 64); // G plane, pixel 0
        assert_eq!(data[32], 32); // B plane, pixel 0

        // Center pixel (1,1): row=1, col=1, width=4 -> index = 1*4 + 1 = 5
        let center_idx = 5;
        assert_eq!(data[center_idx], 0); // R
        assert_eq!(data[16 + center_idx], 0); // G
        assert_eq!(data[32 + center_idx], 0); // B
    }

    #[test]
    fn test_convert_rgba_to_bgra() {
        use edgefirst_tensor::TensorMemory;
        // 2x1 image: pixel0 = [R=10, G=20, B=30, A=255], pixel1 = [R=40, G=50, B=60, A=128]
        let src =
            TensorDyn::image(2, 1, PixelFormat::Rgba, DType::U8, Some(TensorMemory::Mem)).unwrap();
        {
            let mut map = src.as_u8().unwrap().map().unwrap();
            let buf = map.as_mut_slice();
            buf[0..4].copy_from_slice(&[10, 20, 30, 255]);
            buf[4..8].copy_from_slice(&[40, 50, 60, 128]);
        }
        let mut dst =
            TensorDyn::image(2, 1, PixelFormat::Bgra, DType::U8, Some(TensorMemory::Mem)).unwrap();
        {
            let mut __cv = CPUProcessor::default();
            __cv.convert(
                &src,
                &mut dst,
                crate::Rotation::None,
                crate::Flip::None,
                crate::Crop::default(),
            )
            .unwrap();
        }
        let map = dst.as_u8().unwrap().map().unwrap();
        let buf = map.as_slice();
        // PixelFormat::Bgra byte order: [B, G, R, A]
        assert_eq!(&buf[0..4], &[30, 20, 10, 255]);
        assert_eq!(&buf[4..8], &[60, 50, 40, 128]);
    }

    #[test]
    fn test_convert_rgb_to_bgra() {
        // Convert PixelFormat::Rgb→PixelFormat::Rgba and PixelFormat::Rgb→PixelFormat::Bgra, verify R↔B swap matches
        let src =
            TensorDyn::image(2, 1, PixelFormat::Rgb, DType::U8, Some(TensorMemory::Mem)).unwrap();
        {
            let mut map = src.as_u8().unwrap().map().unwrap();
            let buf = map.as_mut_slice();
            buf[0..3].copy_from_slice(&[100, 150, 200]);
            buf[3..6].copy_from_slice(&[50, 75, 25]);
        }
        let mut rgba_dst =
            TensorDyn::image(2, 1, PixelFormat::Rgba, DType::U8, Some(TensorMemory::Mem)).unwrap();
        {
            let mut __cv = CPUProcessor::default();
            __cv.convert(
                &src,
                &mut rgba_dst,
                crate::Rotation::None,
                crate::Flip::None,
                crate::Crop::default(),
            )
            .unwrap();
        }

        let mut bgra_dst =
            TensorDyn::image(2, 1, PixelFormat::Bgra, DType::U8, Some(TensorMemory::Mem)).unwrap();
        {
            let mut __cv = CPUProcessor::default();
            __cv.convert(
                &src,
                &mut bgra_dst,
                crate::Rotation::None,
                crate::Flip::None,
                crate::Crop::default(),
            )
            .unwrap();
        }

        assert_bgra_matches_rgba(&bgra_dst, &rgba_dst);

        // Also verify the B,G,R channels are correct (alpha may vary)
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
        // 2x1 greyscale image
        let src =
            TensorDyn::image(2, 1, PixelFormat::Grey, DType::U8, Some(TensorMemory::Mem)).unwrap();
        {
            let mut map = src.as_u8().unwrap().map().unwrap();
            let buf = map.as_mut_slice();
            buf[0] = 128;
            buf[1] = 64;
        }
        let mut dst =
            TensorDyn::image(2, 1, PixelFormat::Bgra, DType::U8, Some(TensorMemory::Mem)).unwrap();
        {
            let mut __cv = CPUProcessor::default();
            __cv.convert(
                &src,
                &mut dst,
                crate::Rotation::None,
                crate::Flip::None,
                crate::Crop::default(),
            )
            .unwrap();
        }
        let map = dst.as_u8().unwrap().map().unwrap();
        let buf = map.as_slice();
        // Grey→PixelFormat::Bgra: all channels same value, A=255; R↔B swap is no-op on grey
        assert_eq!(&buf[0..4], &[128, 128, 128, 255]);
        assert_eq!(&buf[4..8], &[64, 64, 64, 255]);
    }

    #[test]
    fn test_convert_bgra_to_bgra_copy() {
        // Verify PixelFormat::Bgra→PixelFormat::Bgra is a straight copy
        let src =
            TensorDyn::image(2, 1, PixelFormat::Bgra, DType::U8, Some(TensorMemory::Mem)).unwrap();
        {
            let mut map = src.as_u8().unwrap().map().unwrap();
            let buf = map.as_mut_slice();
            buf[0..4].copy_from_slice(&[10, 20, 30, 255]);
            buf[4..8].copy_from_slice(&[40, 50, 60, 128]);
        }
        let mut dst =
            TensorDyn::image(2, 1, PixelFormat::Bgra, DType::U8, Some(TensorMemory::Mem)).unwrap();
        {
            let mut __cv = CPUProcessor::default();
            __cv.convert(
                &src,
                &mut dst,
                crate::Rotation::None,
                crate::Flip::None,
                crate::Crop::default(),
            )
            .unwrap();
        }
        let map = dst.as_u8().unwrap().map().unwrap();
        let buf = map.as_slice();
        assert_eq!(&buf[0..4], &[10, 20, 30, 255]);
        assert_eq!(&buf[4..8], &[40, 50, 60, 128]);
    }

    /// Helper: compare PixelFormat::Bgra output against PixelFormat::Rgba output by verifying R↔B swap.
    /// Since CPU PixelFormat::Bgra conversion is PixelFormat::Rgba conversion + R↔B swizzle, the results
    /// must be byte-exact after accounting for the channel swap.
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
            .chunks_exact(4)
            .zip(rgba_buf.chunks_exact(4))
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
            include_bytes!(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/../../testdata/camera720p.nv12"
            )),
        )
        .unwrap();

        // Convert to both PixelFormat::Rgba and PixelFormat::Bgra, then compare
        let mut rgba_dst = TensorDyn::image(1280, 720, PixelFormat::Rgba, DType::U8, None).unwrap();
        {
            let mut __cv = CPUProcessor::default();
            __cv.convert(
                &src,
                &mut rgba_dst,
                crate::Rotation::None,
                crate::Flip::None,
                crate::Crop::default(),
            )
            .unwrap();
        }

        let mut bgra_dst = TensorDyn::image(1280, 720, PixelFormat::Bgra, DType::U8, None).unwrap();
        {
            let mut __cv = CPUProcessor::default();
            __cv.convert(
                &src,
                &mut bgra_dst,
                crate::Rotation::None,
                crate::Flip::None,
                crate::Crop::default(),
            )
            .unwrap();
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
            include_bytes!(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/../../testdata/camera720p.yuyv"
            )),
        )
        .unwrap();

        let mut rgba_dst = TensorDyn::image(1280, 720, PixelFormat::Rgba, DType::U8, None).unwrap();
        {
            let mut __cv = CPUProcessor::default();
            __cv.convert(
                &src,
                &mut rgba_dst,
                crate::Rotation::None,
                crate::Flip::None,
                crate::Crop::default(),
            )
            .unwrap();
        }

        let mut bgra_dst = TensorDyn::image(1280, 720, PixelFormat::Bgra, DType::U8, None).unwrap();
        {
            let mut __cv = CPUProcessor::default();
            __cv.convert(
                &src,
                &mut bgra_dst,
                crate::Rotation::None,
                crate::Flip::None,
                crate::Crop::default(),
            )
            .unwrap();
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
            include_bytes!(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/../../testdata/camera720p.vyuy"
            )),
        )
        .unwrap();

        let mut rgba_dst = TensorDyn::image(1280, 720, PixelFormat::Rgba, DType::U8, None).unwrap();
        {
            let mut __cv = CPUProcessor::default();
            __cv.convert(
                &src,
                &mut rgba_dst,
                crate::Rotation::None,
                crate::Flip::None,
                crate::Crop::default(),
            )
            .unwrap();
        }

        let mut bgra_dst = TensorDyn::image(1280, 720, PixelFormat::Bgra, DType::U8, None).unwrap();
        {
            let mut __cv = CPUProcessor::default();
            __cv.convert(
                &src,
                &mut bgra_dst,
                crate::Rotation::None,
                crate::Flip::None,
                crate::Crop::default(),
            )
            .unwrap();
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
            include_bytes!(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/../../testdata/camera720p.nv16"
            )),
        )
        .unwrap();

        let mut rgba_dst = TensorDyn::image(1280, 720, PixelFormat::Rgba, DType::U8, None).unwrap();
        {
            let mut __cv = CPUProcessor::default();
            __cv.convert(
                &src,
                &mut rgba_dst,
                crate::Rotation::None,
                crate::Flip::None,
                crate::Crop::default(),
            )
            .unwrap();
        }

        let mut bgra_dst = TensorDyn::image(1280, 720, PixelFormat::Bgra, DType::U8, None).unwrap();
        {
            let mut __cv = CPUProcessor::default();
            __cv.convert(
                &src,
                &mut bgra_dst,
                crate::Rotation::None,
                crate::Flip::None,
                crate::Crop::default(),
            )
            .unwrap();
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
        use edgefirst_tensor::{Tensor, TensorDyn};
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
            mask_coefficients: TensorDyn::F32(coeff_t),
            protos: TensorDyn::F32(protos_t),
        }
    }

    fn make_detect_box(xmin: f32, ymin: f32, xmax: f32, ymax: f32) -> crate::DetectBox {
        crate::DetectBox {
            bbox: edgefirst_decoder::BoundingBox {
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
        // ProtoData from a detection-only model — caller passes no detections
        // to the materializer (detection-only models have a ProtoData stub
        // with an empty coefficient tensor). Materialization must not panic
        // and must return an empty segmentation list.
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
        // Segmentation should have shape (H, W, 1) with non-zero spatial dims
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
        // xmin should be clamped to 0.0
        assert!((segs[0].xmin - 0.0).abs() < 0.01);
        assert!((segs[0].ymin - 0.0).abs() < 0.01);
    }

    #[test]
    fn test_materialize_invalid_coeff_shape() {
        let cpu = CPUProcessor::new();
        // Proto has 4 channels but coefficients have 6 elements — mismatch
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
        assert!(
            segs[0]
                .segmentation
                .iter()
                .all(|&v| matches!(v, 0_u8 | 255_u8)),
            "scaled path output must remain binarized u8"
        );
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
            seg.segmentation.iter().all(|&v| v == 255),
            "positive logits should remain foreground after scaled letterbox mapping"
        );
    }

    #[test]
    fn test_materialize_scaled_out_of_range_letterbox_no_panic() {
        let cpu = CPUProcessor::new();
        let proto_data = make_proto_data_with_values(8, 8, 1, vec![1.0_f32; 64], vec![vec![1.0]]);
        let det = [make_detect_box(0.95, 0.95, 1.0, 1.0)];
        // Deliberately out-of-range letterbox values: this used to drive
        // proto_x0/proto_y0 to the exclusive upper bound and panic in scaled_run.
        let segs = cpu
            .materialize_scaled_segmentations(&det, &proto_data, Some([1.2, 1.2, 1.4, 1.4]), 32, 32)
            .expect("scaled path should clamp ROI starts and avoid OOB panics");
        assert_eq!(segs.len(), 1);
        let shape = segs[0].segmentation.shape();
        assert!(shape[0] > 0 && shape[1] > 0);
        assert_eq!(shape[2], 1);
    }

    /// Golden-byte anchor: runs the quantized decode + CPU mask materialization
    /// end-to-end on the cached YOLOv8-seg fixture and asserts the first
    /// detection's binarized u8 mask is bit-exact against a committed golden
    /// file.
    ///
    /// Regression-protection intent: any refactor that changes the i8 fused
    /// dequant+dot+sigmoid kernel, the NMS ordering, or the ROI coordinate
    /// mapping will perturb the mask bytes and fail this test.
    ///
    /// Generating / refreshing the golden:
    /// ```text
    /// REGEN_GOLDEN=1 cargo test -p edgefirst-image --lib \
    ///     test_materialize_golden_i8_cached_model
    /// ```
    /// writes `testdata/yolov8_mask0_160x160.bin` (or updates it). Commit
    /// the resulting file; subsequent runs verify bit-exact.
    #[test]
    fn test_materialize_golden_i8_cached_model() {
        use edgefirst_decoder::{
            yolo::impl_yolo_segdet_quant_proto, DetectBox, Nms, ProtoData, Quantization, XYWH,
        };

        let boxes_raw = include_bytes!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../testdata/yolov8_boxes_116x8400.bin"
        ));
        let boxes_i8 =
            unsafe { std::slice::from_raw_parts(boxes_raw.as_ptr() as *const i8, boxes_raw.len()) };
        let boxes = ndarray::Array2::from_shape_vec((116, 8400), boxes_i8.to_vec()).unwrap();

        let protos_raw = include_bytes!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../testdata/yolov8_protos_160x160x32.bin"
        ));
        let protos_i8 = unsafe {
            std::slice::from_raw_parts(protos_raw.as_ptr() as *const i8, protos_raw.len())
        };
        let protos = ndarray::Array3::from_shape_vec((160, 160, 32), protos_i8.to_vec()).unwrap();

        let quant_boxes = Quantization::new(0.019_484_945, 20);
        let quant_protos = Quantization::new(0.020_889_873, -115);

        let mut detections: Vec<DetectBox> = Vec::with_capacity(50);
        let proto_data: ProtoData = impl_yolo_segdet_quant_proto::<XYWH, _, _>(
            (boxes.view(), quant_boxes),
            (protos.view(), quant_protos),
            0.45,
            0.45,
            Some(Nms::ClassAgnostic),
            &mut detections,
        );
        assert!(!detections.is_empty(), "fixture produced no detections");

        let cpu = CPUProcessor::new();
        let segs = cpu
            .materialize_segmentations(&detections, &proto_data, None)
            .expect("materialize_segmentations failed on cached fixture");
        assert_eq!(segs.len(), detections.len());

        // Serialize detection 0's binarized mask as a compact byte blob:
        //   [u16 roi_h_le][u16 roi_w_le][roi_h * roi_w bytes of u8 mask data]
        // (Channels dim is always 1; no need to store it.)
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
        let det0_contig = det0_seg.as_standard_layout();
        actual.extend_from_slice(det0_contig.as_slice().expect("standard layout gives slice"));

        let golden_path = std::path::PathBuf::from(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../testdata/yolov8_mask0_160x160.bin"
        ));

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
        // Contiguous PixelFormat::Nv12 should NOT be multiplane
        let contiguous = TensorDyn::image(
            640,
            480,
            PixelFormat::Nv12,
            DType::U8,
            Some(TensorMemory::Mem),
        )?;
        assert!(!contiguous.as_u8().unwrap().is_multiplane());
        assert!(contiguous.as_u8().unwrap().chroma().is_none());

        // from_planes should be multiplane
        let luma = Tensor::<u8>::new(&[480, 640], Some(TensorMemory::Mem), None)?;
        let chroma = Tensor::<u8>::new(&[240, 640], Some(TensorMemory::Mem), None)?;
        let multiplane = {
            let __t = Tensor::<u8>::from_planes(luma, chroma, PixelFormat::Nv12)?;
            TensorDyn::from(__t)
        };
        assert!(multiplane.as_u8().unwrap().is_multiplane());
        assert!(multiplane.as_u8().unwrap().chroma().is_some());

        // PixelFormat::Rgb should NOT be multiplane
        let rgb = TensorDyn::image(
            640,
            480,
            PixelFormat::Rgb,
            DType::U8,
            Some(TensorMemory::Mem),
        )?;
        assert!(!rgb.as_u8().unwrap().is_multiplane());
        Ok(())
    }

    #[test]
    fn test_multiplane_invalid_shapes() {
        // Wrong format (PixelFormat::Rgb not supported for multiplane)
        let luma = Tensor::<u8>::new(&[480, 640], Some(TensorMemory::Mem), None).unwrap();
        let chroma = Tensor::<u8>::new(&[240, 640], Some(TensorMemory::Mem), None).unwrap();
        assert!(Tensor::<u8>::from_planes(luma, chroma, PixelFormat::Rgb).is_err());

        // Chroma height mismatch for PixelFormat::Nv12 (should be H/2)
        let luma = Tensor::<u8>::new(&[480, 640], Some(TensorMemory::Mem), None).unwrap();
        let chroma = Tensor::<u8>::new(&[480, 640], Some(TensorMemory::Mem), None).unwrap();
        assert!(Tensor::<u8>::from_planes(luma, chroma, PixelFormat::Nv12).is_err());

        // Chroma width mismatch
        let luma = Tensor::<u8>::new(&[480, 640], Some(TensorMemory::Mem), None).unwrap();
        let chroma = Tensor::<u8>::new(&[240, 320], Some(TensorMemory::Mem), None).unwrap();
        assert!(Tensor::<u8>::from_planes(luma, chroma, PixelFormat::Nv12).is_err());

        // 3D luma (should be 2D)
        let luma = Tensor::<u8>::new(&[480, 640, 1], Some(TensorMemory::Mem), None).unwrap();
        let chroma = Tensor::<u8>::new(&[240, 640], Some(TensorMemory::Mem), None).unwrap();
        assert!(Tensor::<u8>::from_planes(luma, chroma, PixelFormat::Nv12).is_err());
    }

    #[test]
    fn test_multiplane_nv12_to_rgb_cpu() -> Result<()> {
        // Load PixelFormat::Nv12 test data as contiguous buffer
        let nv12_bytes = include_bytes!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../testdata/camera720p.nv12"
        ));
        let width = 1280usize;
        let height = 720usize;
        let y_size = width * height;
        let uv_size = width * (height / 2);

        // ── Contiguous path (baseline) ──────────────────────────────
        let contiguous = load_bytes_to_tensor(width, height, PixelFormat::Nv12, None, nv12_bytes)?;
        let dst_contiguous = TensorDyn::image(width, height, PixelFormat::Rgb, DType::U8, None)?;
        let mut converter = CPUProcessor::default();
        let contiguous_dyn = contiguous;
        let mut dst_contiguous_dyn = dst_contiguous;
        converter.convert(
            &contiguous_dyn,
            &mut dst_contiguous_dyn,
            Rotation::None,
            Flip::None,
            Crop::default(),
        )?;

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
        let dst_multiplane = TensorDyn::image(width, height, PixelFormat::Rgb, DType::U8, None)?;
        let multiplane_dyn = multiplane;
        let mut dst_multiplane_dyn = dst_multiplane;
        converter.convert(
            &multiplane_dyn,
            &mut dst_multiplane_dyn,
            Rotation::None,
            Flip::None,
            Crop::default(),
        )?;

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
        let nv12_bytes = include_bytes!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../testdata/camera720p.nv12"
        ));
        let width = 1280usize;
        let height = 720usize;
        let y_size = width * height;
        let uv_size = width * (height / 2);

        let contiguous = load_bytes_to_tensor(width, height, PixelFormat::Nv12, None, nv12_bytes)?;
        let dst_contiguous = TensorDyn::image(width, height, PixelFormat::Rgba, DType::U8, None)?;
        let mut converter = CPUProcessor::default();
        let contiguous_dyn = contiguous;
        let mut dst_contiguous_dyn = dst_contiguous;
        converter.convert(
            &contiguous_dyn,
            &mut dst_contiguous_dyn,
            Rotation::None,
            Flip::None,
            Crop::default(),
        )?;

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
        let dst_multiplane = TensorDyn::image(width, height, PixelFormat::Rgba, DType::U8, None)?;
        let multiplane_dyn = multiplane;
        let mut dst_multiplane_dyn = dst_multiplane;
        converter.convert(
            &multiplane_dyn,
            &mut dst_multiplane_dyn,
            Rotation::None,
            Flip::None,
            Crop::default(),
        )?;

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
        let nv12_bytes = include_bytes!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../testdata/camera720p.nv12"
        ));
        let width = 1280usize;
        let height = 720usize;
        let y_size = width * height;
        let uv_size = width * (height / 2);

        let contiguous = load_bytes_to_tensor(width, height, PixelFormat::Nv12, None, nv12_bytes)?;
        let dst_contiguous = TensorDyn::image(width, height, PixelFormat::Grey, DType::U8, None)?;
        let mut converter = CPUProcessor::default();
        let contiguous_dyn = contiguous;
        let mut dst_contiguous_dyn = dst_contiguous;
        converter.convert(
            &contiguous_dyn,
            &mut dst_contiguous_dyn,
            Rotation::None,
            Flip::None,
            Crop::default(),
        )?;

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
        let dst_multiplane = TensorDyn::image(width, height, PixelFormat::Grey, DType::U8, None)?;
        let multiplane_dyn = multiplane;
        let mut dst_multiplane_dyn = dst_multiplane;
        converter.convert(
            &multiplane_dyn,
            &mut dst_multiplane_dyn,
            Rotation::None,
            Flip::None,
            Crop::default(),
        )?;

        let contiguous_map = dst_contiguous_dyn.as_u8().unwrap().map()?;
        let multiplane_map = dst_multiplane_dyn.as_u8().unwrap().map()?;
        assert_eq!(
            contiguous_map.as_slice(),
            multiplane_map.as_slice(),
            "multiplane PixelFormat::Nv12→PixelFormat::Grey must match contiguous path"
        );
        Ok(())
    }
}
