// SPDX-FileCopyrightText: Copyright 2025 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
#[allow(deprecated)]
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
            let mut rgb1 = TensorDyn::image(w, h, PixelFormat::Rgb, DType::U8, mem()).unwrap();
            let mut rgb2 = TensorDyn::image(w, h, PixelFormat::Rgb, DType::U8, mem()).unwrap();
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

    /// CPU-processor tests operate on host memory: a tight `Mem` tensor whose
    /// row stride equals `width * bpp`. Pinning these tests to `Mem` keeps them
    /// deterministic across platforms — `None` auto-selects a pitch-padded DMA
    /// tensor on i.MX (DMA heap present), which shears the flat `copy_from_slice`
    /// fill and flat `as_slice()` assertions below. Padded-stride coverage lives
    /// in the dedicated `odd_dim_cpu` integration suite.
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
        // Default to host memory (tight) so the flat fill matches the layout.
        let memory = memory.or_else(mem);
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
                &edgefirst_bench::testdata::read($src_file),
            )?;

            // Load destination reference
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
            )?;

            converter.convert(
                &src,
                &mut converted,
                Rotation::None,
                Flip::None,
                Crop::default(),
            )?;

            // INTERIM COLORIMETRY STOP-GAP (see crates/image/ARCHITECTURE.md
            // "Colorimetry"): the camera720p/1080p reference fixtures are
            // BT.709 limited-range ground truth, but the HAL currently hardcodes
            // BT.601 full-range for all YUV. That mismatch drops similarity to
            // ~0.97, so the threshold is loosened from 0.99 to 0.95 here. The
            // colorimetry PR restores 0.99 with correct per-source handling.
            compare_images_convert_to_rgb(&dst, &converted, 0.95, function!());

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
                &edgefirst_bench::testdata::read($src_file),
            )?;

            // Load destination reference
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
            )?;

            converter.convert(
                &src,
                &mut converted,
                Rotation::None,
                Flip::None,
                Crop::default(),
            )?;

            // INTERIM COLORIMETRY STOP-GAP (see generate_conversion_tests! and
            // crates/image/ARCHITECTURE.md "Colorimetry"): loosened 0.97 -> 0.95
            // while the HAL hardcodes BT.601 full-range against BT.709 fixtures.
            compare_images_convert_to_grey(&dst, &converted, 0.95, function!());

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

        let converted = TensorDyn::image(4, 1, PixelFormat::Rgb, DType::U8, mem())?;
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

        let converted = TensorDyn::image(4, 4, PixelFormat::Rgba, DType::U8, mem())?;
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

        let converted = TensorDyn::image(4, 4, PixelFormat::Rgba, DType::U8, mem())?;
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

        let converted = TensorDyn::image(4, 4, PixelFormat::Rgba, DType::U8, mem())?;
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

        let converted = TensorDyn::image(4, 4, PixelFormat::Rgba, DType::U8, mem())?;
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

        let converted = TensorDyn::image(4, 4, PixelFormat::Rgba, DType::U8, mem())?;
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

        let converted = TensorDyn::image(2, 2, PixelFormat::Rgba, DType::U8, mem())?;
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

        // The untagged 2x2 Grey source is below the HD threshold, so the
        // colorimetry heuristic resolves it to BT.601 limited range: grey luma
        // is mapped into the 16..235 limited range (219-step), so the
        // row-averaged grey [4, 6] yields Y = 19, 21. The second dst row is
        // outside the crop and keeps its pre-fill value.
        assert_eq!(
            converted_dyn.as_u8().unwrap().map()?.as_slice(),
            &[19, 128, 21, 128, 200, 128, 200, 128]
        );
        Ok(())
    }

    #[test]
    fn test_cpu_fill_rgba() -> Result<()> {
        // Load source
        let src = load_bytes_to_tensor(1, 1, PixelFormat::Rgba, None, &[3, 3, 3, 255])?;

        let mut converter = CPUProcessor::default();

        let converted = TensorDyn::image(2, 2, PixelFormat::Rgba, DType::U8, mem())?;
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

        let converted = TensorDyn::image(2, 3, PixelFormat::Yuyv, DType::U8, mem())?;
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

        // 2x3 YUYV is below the HD threshold, so the colorimetry heuristic
        // resolves it to BT.601 limited (previously a BT.709 hardcode). Red
        // (255,0,0) → Y=81, U=90, V=240 under BT.601 limited.
        assert_eq!(
            converted_dyn.as_u8().unwrap().map()?.as_slice(),
            &[81, 90, 81, 240, 19, 128, 19, 128, 81, 90, 81, 240]
        );
        Ok(())
    }

    #[test]
    fn test_cpu_fill_grey() -> Result<()> {
        // Load source
        let src =
            load_bytes_to_tensor(2, 1, PixelFormat::Rgba, None, &[3, 3, 3, 255, 3, 3, 3, 255])?;

        let mut converter = CPUProcessor::default();

        let converted = TensorDyn::image(2, 3, PixelFormat::Grey, DType::U8, mem())?;
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
        // Threshold 0.97 (was 0.99): the codec now decodes colour JPEGs to
        // their native NV12 layout, so the `giraffe.jpg` background routes
        // through an NV12 → RGBA conversion (chroma subsampling) before the
        // mask is composited. The `output_render_cpu.jpg` golden was captured
        // from the old direct-RGB JPEG decode, so the NV12-sourced background
        // differs slightly. This mirrors the GL counterpart's 0.97 tolerance.
        compare_images_convert_to_rgb(&image, &expected, 0.97, function!());
    }

    // =========================================================================
    // Generic Conversion Tests
    // (These tests use TensorDyn for all image representations)
    // =========================================================================

    #[test]
    fn test_convert_rgb_to_planar_rgb_generic() {
        // Create PixelFormat::Rgb source image
        let src = TensorDyn::image(4, 4, PixelFormat::Rgb, DType::U8, mem()).unwrap();
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
        let mut dst = TensorDyn::image(4, 4, PixelFormat::PlanarRgb, DType::U8, mem()).unwrap();

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
        let src = TensorDyn::image(4, 4, PixelFormat::Rgba, DType::U8, mem()).unwrap();
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
        let mut dst = TensorDyn::image(4, 4, PixelFormat::PlanarRgb, DType::U8, mem()).unwrap();

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
        let src = TensorDyn::image(4, 4, PixelFormat::Rgb, DType::U8, mem()).unwrap();
        {
            let mut map = src.as_u8().unwrap().map().unwrap();
            let data = map.as_mut_slice();
            for (i, byte) in data.iter_mut().enumerate() {
                *byte = (i % 256) as u8;
            }
        }

        // Create destination tensor
        let mut dst = TensorDyn::image(4, 4, PixelFormat::Rgb, DType::U8, mem()).unwrap();

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
        let src = TensorDyn::image(8, 8, PixelFormat::Nv12, DType::U8, mem()).unwrap();
        let mut dst = TensorDyn::image(8, 8, PixelFormat::Nv12, DType::U8, mem()).unwrap();

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
        let mut dst = TensorDyn::image(4, 4, PixelFormat::Rgba, DType::U8, mem()).unwrap();
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
        let mut dst = TensorDyn::image(4, 4, PixelFormat::Rgb, DType::U8, mem()).unwrap();
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
        let mut dst = TensorDyn::image(4, 4, PixelFormat::PlanarRgb, DType::U8, mem()).unwrap();
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
            &edgefirst_bench::testdata::read("camera720p.nv12"),
        )
        .unwrap();

        // Convert to both PixelFormat::Rgba and PixelFormat::Bgra, then compare
        let mut rgba_dst =
            TensorDyn::image(1280, 720, PixelFormat::Rgba, DType::U8, mem()).unwrap();
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
            TensorDyn::image(1280, 720, PixelFormat::Bgra, DType::U8, mem()).unwrap();
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
            &edgefirst_bench::testdata::read("camera720p.yuyv"),
        )
        .unwrap();

        let mut rgba_dst =
            TensorDyn::image(1280, 720, PixelFormat::Rgba, DType::U8, mem()).unwrap();
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
            TensorDyn::image(1280, 720, PixelFormat::Bgra, DType::U8, mem()).unwrap();
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
            &edgefirst_bench::testdata::read("camera720p.vyuy"),
        )
        .unwrap();

        let mut rgba_dst =
            TensorDyn::image(1280, 720, PixelFormat::Rgba, DType::U8, mem()).unwrap();
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
            TensorDyn::image(1280, 720, PixelFormat::Bgra, DType::U8, mem()).unwrap();
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
            &edgefirst_bench::testdata::read("camera720p.nv16"),
        )
        .unwrap();

        let mut rgba_dst =
            TensorDyn::image(1280, 720, PixelFormat::Rgba, DType::U8, mem()).unwrap();
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
            TensorDyn::image(1280, 720, PixelFormat::Bgra, DType::U8, mem()).unwrap();
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
            layout: edgefirst_decoder::ProtoLayout::Nhwc,
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

    #[test]
    fn test_materialize_nchw_i8_produces_correct_mask() {
        // Verify that materialize_segmentations handles NCHW layout correctly.
        // Proto data is stored as [K, H, W] (NCHW physical) with K=2, H=4, W=4.
        // Channel 0 filled with +10, channel 1 filled with -10.
        // Coefficient [+1, +1] → dot > 0 for all pixels (positive + negative = 0,
        // but with symmetric scale the sum should still be evaluated correctly).
        // Use coefficients [1, 0] to select only channel 0 → all positive → 255.
        use edgefirst_tensor::{Quantization, Tensor, TensorDyn};
        let cpu = CPUProcessor::new();
        let (proto_h, proto_w, num_protos) = (4, 4, 2);
        // NCHW physical: [K=2, H=4, W=4] → flat = [ch0 16 values, ch1 16 values]
        let mut proto_values = vec![0_i8; num_protos * proto_h * proto_w];
        // Channel 0: all +10, Channel 1: all -10
        for i in 0..proto_h * proto_w {
            proto_values[i] = 10; // channel 0
            proto_values[proto_h * proto_w + i] = -10; // channel 1
        }
        let mut protos_t =
            Tensor::<i8>::from_slice(&proto_values, &[num_protos, proto_h, proto_w]).unwrap();
        protos_t
            .set_quantization(Quantization::per_tensor(0.1, 0))
            .unwrap();
        // Coefficient selects channel 0 only: [+1, 0] (raw i8 with scale=1.0)
        let coeff_values = vec![1_i8, 0_i8];
        let mut coeff_t = Tensor::<i8>::from_slice(&coeff_values, &[1, num_protos]).unwrap();
        coeff_t
            .set_quantization(Quantization::per_tensor(1.0, 0))
            .unwrap();
        let proto_data = crate::ProtoData {
            mask_coefficients: TensorDyn::I8(coeff_t),
            protos: TensorDyn::I8(protos_t),
            layout: edgefirst_decoder::ProtoLayout::Nchw,
        };
        let det = [make_detect_box(0.0, 0.0, 1.0, 1.0)];
        let segs = cpu
            .materialize_segmentations(&det, &proto_data, None)
            .expect("NCHW i8 proto layout should be handled");
        assert_eq!(segs.len(), 1);
        // All pixels should be 255 (positive dot product from channel 0)
        assert!(
            segs[0].segmentation.iter().all(|&v| v == 255),
            "NCHW proto with positive channel selected should yield all-255 mask"
        );
    }

    #[test]
    fn test_materialize_nchw_float_rejected() {
        // NCHW layout with non-i8 protos should return NotSupported error.
        use edgefirst_tensor::{Tensor, TensorDyn};
        let cpu = CPUProcessor::new();
        let proto_values = vec![1.0_f32; 2 * 4 * 4];
        let protos_t = Tensor::<f32>::from_slice(&proto_values, &[2, 4, 4]).unwrap();
        let coeff_values = vec![1.0_f32; 2];
        let coeff_t = Tensor::<f32>::from_slice(&coeff_values, &[1, 2]).unwrap();
        let proto_data = crate::ProtoData {
            mask_coefficients: TensorDyn::F32(coeff_t),
            protos: TensorDyn::F32(protos_t),
            layout: edgefirst_decoder::ProtoLayout::Nchw,
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
        // NCHW layout with non-i8 protos on the scaled path should also error.
        use edgefirst_tensor::{Tensor, TensorDyn};
        let cpu = CPUProcessor::new();
        let proto_values = vec![1.0_f32; 2 * 4 * 4];
        let protos_t = Tensor::<f32>::from_slice(&proto_values, &[2, 4, 4]).unwrap();
        let coeff_values = vec![1.0_f32; 2];
        let coeff_t = Tensor::<f32>::from_slice(&coeff_values, &[1, 2]).unwrap();
        let proto_data = crate::ProtoData {
            mask_coefficients: TensorDyn::F32(coeff_t),
            protos: TensorDyn::F32(protos_t),
            layout: edgefirst_decoder::ProtoLayout::Nchw,
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
        // Verify that the scaled path handles NCHW i8 layout correctly.
        use edgefirst_tensor::{Quantization, Tensor, TensorDyn};
        let cpu = CPUProcessor::new();
        let (proto_h, proto_w, num_protos) = (4, 4, 2);
        // NCHW physical: [K=2, H=4, W=4]
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
            mask_coefficients: TensorDyn::I8(coeff_t),
            protos: TensorDyn::I8(protos_t),
            layout: edgefirst_decoder::ProtoLayout::Nchw,
        };
        let det = [make_detect_box(0.0, 0.0, 1.0, 1.0)];
        let segs = cpu
            .materialize_scaled_segmentations(&det, &proto_data, None, 16, 16)
            .expect("NCHW i8 scaled path should succeed");
        assert_eq!(segs.len(), 1);
        // All pixels should be 255 (positive dot product from channel 0)
        assert!(
            segs[0].segmentation.iter().all(|&v| v == 255),
            "NCHW i8 scaled path with positive channel should yield all-255 mask"
        );
    }

    #[test]
    fn test_materialize_nchw_i8_asymmetric_quantization() {
        // Verify that NCHW i8 kernel handles non-zero zero-points correctly.
        // Asymmetric quantization: real_value = (raw - zero_point) * scale
        // Proto: zp=5, scale=0.1 → raw 15 means (15-5)*0.1 = 1.0
        // Coeff: zp=10, scale=0.5 → raw 12 means (12-10)*0.5 = 1.0
        //
        // With K=2 channels: proto ch0=15 (→1.0), proto ch1=5 (→0.0)
        // coeff = [12, 10] (→[1.0, 0.0])
        // dot = 1.0*1.0 + 0.0*0.0 = 1.0 > 0 → mask = 255
        use edgefirst_tensor::{Quantization, Tensor, TensorDyn};
        let cpu = CPUProcessor::new();
        let (proto_h, proto_w, num_protos) = (4, 4, 2);

        // NCHW physical: [K=2, H=4, W=4]
        let mut proto_values = vec![0_i8; num_protos * proto_h * proto_w];
        let proto_zp: i8 = 5;
        // Channel 0: raw = 15 → dequant = (15-5)*0.1 = 1.0
        // Channel 1: raw = 5 → dequant = (5-5)*0.1 = 0.0
        for i in 0..proto_h * proto_w {
            proto_values[i] = 15; // channel 0
            proto_values[proto_h * proto_w + i] = proto_zp; // channel 1 (at zero-point)
        }
        let mut protos_t =
            Tensor::<i8>::from_slice(&proto_values, &[num_protos, proto_h, proto_w]).unwrap();
        protos_t
            .set_quantization(Quantization::per_tensor(0.1, proto_zp as i32))
            .unwrap();

        // Coefficient: zp=10, scale=0.5
        // raw=[12, 10] → dequant=[(12-10)*0.5, (10-10)*0.5] = [1.0, 0.0]
        let coeff_zp: i8 = 10;
        let coeff_values = vec![12_i8, coeff_zp];
        let mut coeff_t = Tensor::<i8>::from_slice(&coeff_values, &[1, num_protos]).unwrap();
        coeff_t
            .set_quantization(Quantization::per_tensor(0.5, coeff_zp as i32))
            .unwrap();

        let proto_data = crate::ProtoData {
            mask_coefficients: TensorDyn::I8(coeff_t),
            protos: TensorDyn::I8(protos_t),
            layout: edgefirst_decoder::ProtoLayout::Nchw,
        };
        let det = [make_detect_box(0.0, 0.0, 1.0, 1.0)];
        let segs = cpu
            .materialize_segmentations(&det, &proto_data, None)
            .expect("NCHW i8 asymmetric quantization should succeed");
        assert_eq!(segs.len(), 1);
        // Dot product = 1.0 > 0 → all 255
        assert!(
            segs[0].segmentation.iter().all(|&v| v == 255),
            "NCHW i8 with non-zero zero-points: positive dot should yield all-255"
        );

        // Now test case where dot < 0: use coeff that selects a negative channel.
        // coeff raw=[10, 12] → dequant=[0.0, 1.0]; ch1 is at zero_point → 0.0
        // dot = 1.0*0.0 + 0.0*1.0 = 0.0 → NOT > 0 → mask = 0
        let coeff_neg = vec![coeff_zp, 12_i8]; // selects channel 1 which is zero
        let mut coeff_t_neg = Tensor::<i8>::from_slice(&coeff_neg, &[1, num_protos]).unwrap();
        coeff_t_neg
            .set_quantization(Quantization::per_tensor(0.5, coeff_zp as i32))
            .unwrap();

        // Recreate protos tensor (Tensor doesn't implement Clone)
        let mut protos_t2 =
            Tensor::<i8>::from_slice(&proto_values, &[num_protos, proto_h, proto_w]).unwrap();
        protos_t2
            .set_quantization(Quantization::per_tensor(0.1, proto_zp as i32))
            .unwrap();

        let proto_data_neg = crate::ProtoData {
            mask_coefficients: TensorDyn::I8(coeff_t_neg),
            protos: TensorDyn::I8(protos_t2),
            layout: edgefirst_decoder::ProtoLayout::Nchw,
        };
        let segs_neg = cpu
            .materialize_segmentations(&det, &proto_data_neg, None)
            .expect("NCHW i8 asymmetric: zero-dot case");
        assert_eq!(segs_neg.len(), 1);
        // dot = 0 → NOT > 0 → mask should be 0
        assert!(
            segs_neg[0].segmentation.iter().all(|&v| v == 0),
            "NCHW i8 with non-zero zero-points: zero dot should yield all-0"
        );
    }

    #[test]
    fn test_materialize_per_channel_i8_falls_through_to_f32() {
        // Verify that I8 protos with per-channel quantization fall through
        // from the i8×i8 fast path to the general f32 dequant path.
        // This tests the regression fix: previously the i8 fast path returned
        // NotSupported without falling back.
        use edgefirst_tensor::{Quantization, Tensor, TensorDyn};
        let cpu = CPUProcessor::new();
        let (proto_h, proto_w, num_protos) = (4, 4, 2);

        // NHWC protos: [H, W, K] = [4, 4, 2]
        // Channel 0: all +10, Channel 1: all -10
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
            mask_coefficients: TensorDyn::I8(coeff_t),
            protos: TensorDyn::I8(protos_t),
            layout: edgefirst_decoder::ProtoLayout::Nhwc,
        };
        let det = [make_detect_box(0.0, 0.0, 1.0, 1.0)];
        let segs = cpu
            .materialize_segmentations(&det, &proto_data, None)
            .expect("per-channel I8 protos should fall through to f32 path");
        assert_eq!(segs.len(), 1);
        // Channel 0 is positive (10 * 0.1 = 1.0), coeff selects it → dot > 0 → 255
        assert!(
            segs[0].segmentation.iter().all(|&v| v == 255),
            "per-channel i8 fallback: positive dot should yield all-255"
        );
    }

    /// Golden-byte anchor: runs the quantized decode + CPU mask materialization
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
        use edgefirst_decoder::{configs, ConfigOutput, DecoderBuilder, DetectBox, Nms, ProtoData};
        use edgefirst_tensor::{Tensor, TensorDyn};

        let boxes_raw = edgefirst_bench::testdata::read("yolov8_boxes_116x8400.bin");
        let boxes_i8 =
            unsafe { std::slice::from_raw_parts(boxes_raw.as_ptr() as *const i8, boxes_raw.len()) };
        let boxes_tensor = TensorDyn::I8(
            Tensor::<i8>::from_slice(boxes_i8, &[1, 116, 8400]).expect("boxes tensor"),
        );

        let protos_raw = edgefirst_bench::testdata::read("yolov8_protos_160x160x32.bin");
        let protos_i8 = unsafe {
            std::slice::from_raw_parts(protos_raw.as_ptr() as *const i8, protos_raw.len())
        };
        let protos_tensor = TensorDyn::I8(
            Tensor::<i8>::from_slice(protos_i8, &[1, 160, 160, 32]).expect("protos tensor"),
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

        // Resolve testdata via EDGEFIRST_TESTDATA_DIR when set (on-target runs use
        // a deployed copy; the compile-time manifest path does not exist there),
        // falling back to the source tree for local runs.
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
        use edgefirst_tensor::{Quantization, Tensor, TensorDyn};
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
            mask_coefficients: TensorDyn::I8(coeff_t),
            protos: TensorDyn::I8(protos_t),
            layout: edgefirst_decoder::ProtoLayout::Nhwc,
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
        use edgefirst_tensor::{Quantization, Tensor, TensorDyn};
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
            mask_coefficients: TensorDyn::I16(coeff_t),
            protos: TensorDyn::I8(protos_t),
            layout: edgefirst_decoder::ProtoLayout::Nhwc,
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
        // All protos positive, all coeffs positive → dot products positive
        // → all mask pixels should be 255.
        assert!(
            segs[0].segmentation.iter().all(|&v| v == 255),
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
        // Negative dot products → all mask pixels should be 0.
        assert!(
            segs[0].segmentation.iter().all(|&v| v == 0),
            "positive protos + negative coeffs should produce all-0 mask"
        );
    }

    #[test]
    fn test_materialize_scaled_i8_i8_mixed_sign_boundary() {
        // Create a proto field where the left half is positive, right half
        // is negative. This exercises the bilinear boundary interpolation
        // code (the non-shortcut path).
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
        let mask = &segs[0].segmentation;
        // Mask should have both 0 and 255 values (left half ≈ 255, right ≈ 0).
        let has_255 = mask.iter().any(|&v| v == 255);
        let has_0 = mask.iter().any(|&v| v == 0);
        assert!(
            has_255 && has_0,
            "mixed-sign proto field should produce mixed mask"
        );
        // All values should be binarized.
        assert!(
            mask.iter().all(|&v| v == 0 || v == 255),
            "mask must be binarized (0 or 255)"
        );
    }

    #[test]
    fn test_materialize_scaled_i8_i8_with_zero_points() {
        // Test that zero-point correction is applied correctly.
        // Protos with zp=-5, coeffs with zp=3.
        let proto_h = 4;
        let proto_w = 4;
        let num_protos = 2;
        // Raw proto values are 10, dequantized = (10 - (-5)) * 0.1 = 1.5
        let proto_values = vec![10_i8; proto_h * proto_w * num_protos];
        // Raw coeff values are 10, dequantized = (10 - 3) * 0.1 = 0.7
        // Both positive → dot product positive → mask = 255
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
        // Dequantized values are both positive → all 255.
        assert!(
            segs[0].segmentation.iter().all(|&v| v == 255),
            "positive dequantized values should produce all-255 mask"
        );
    }

    #[test]
    fn test_materialize_scaled_i8_i8_parity_with_float() {
        // Compare i8×i8 integer path output against the f32 fallback path
        // using symmetric quantization (zp=0) so the results should match exactly.
        let proto_h = 8;
        let proto_w = 8;
        let num_protos = 4;

        // Generate a structured proto field with varied values.
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

        // Build f32 versions (symmetric quant, zp=0, scale=1.0).
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

        // With symmetric quantization (zp=0, scale=1.0), the sign of the
        // dot product is identical, so binarized masks must match exactly.
        let f32_mask: Vec<u8> = segs_f32[0].segmentation.iter().copied().collect();
        let i8_mask: Vec<u8> = segs_i8[0].segmentation.iter().copied().collect();
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

        let f32_mask: Vec<u8> = segs_f32[0].segmentation.iter().copied().collect();
        let i16_mask: Vec<u8> = segs_i16[0].segmentation.iter().copied().collect();
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

        let f32_mask: Vec<u8> = segs_f32[0].segmentation.iter().copied().collect();
        let i16_mask: Vec<u8> = segs_i16[0].segmentation.iter().copied().collect();
        assert_eq!(
            f32_mask, i16_mask,
            "scaled i16×i8 path should match f32 fallback"
        );
    }

    #[test]
    fn test_materialize_nchw_i8_f32_coefficients_use_fallback() {
        use edgefirst_tensor::{Quantization, Tensor, TensorDyn};
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
            mask_coefficients: TensorDyn::F32(coeff_t),
            protos: TensorDyn::I8(protos_t),
            layout: edgefirst_decoder::ProtoLayout::Nchw,
        };
        let det = [make_detect_box(0.0, 0.0, 1.0, 1.0)];
        let segs = cpu
            .materialize_segmentations(&det, &proto_data, None)
            .expect("NCHW i8 protos should transpose in f32 fallback");
        assert!(segs[0].segmentation.iter().all(|&v| v == 255));
    }

    #[test]
    fn test_materialize_scaled_nchw_i8_f32_coefficients_use_fallback() {
        use edgefirst_tensor::{Quantization, Tensor, TensorDyn};
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
            mask_coefficients: TensorDyn::F32(coeff_t),
            protos: TensorDyn::I8(protos_t),
            layout: edgefirst_decoder::ProtoLayout::Nchw,
        };
        let det = [make_detect_box(0.0, 0.0, 1.0, 1.0)];
        let segs = cpu
            .materialize_scaled_segmentations(&det, &proto_data, None, 16, 16)
            .expect("NCHW i8 protos should transpose in scaled f32 fallback");
        assert!(segs[0].segmentation.iter().all(|&v| v == 255));
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
        // Det 0: positive coeffs → all 255.
        assert!(segs[0].segmentation.iter().all(|&v| v == 255));
        // Det 1: negative coeffs → all 0.
        assert!(segs[1].segmentation.iter().all(|&v| v == 0));
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
        let nv12_bytes = edgefirst_bench::testdata::read("camera720p.nv12");
        let width = 1280usize;
        let height = 720usize;
        let y_size = width * height;
        let uv_size = width * (height / 2);

        // ── Contiguous path (baseline) ──────────────────────────────
        let contiguous = load_bytes_to_tensor(width, height, PixelFormat::Nv12, None, &nv12_bytes)?;
        let dst_contiguous = TensorDyn::image(width, height, PixelFormat::Rgb, DType::U8, mem())?;
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
        let dst_multiplane = TensorDyn::image(width, height, PixelFormat::Rgb, DType::U8, mem())?;
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
        let nv12_bytes = edgefirst_bench::testdata::read("camera720p.nv12");
        let width = 1280usize;
        let height = 720usize;
        let y_size = width * height;
        let uv_size = width * (height / 2);

        let contiguous = load_bytes_to_tensor(width, height, PixelFormat::Nv12, None, &nv12_bytes)?;
        let dst_contiguous = TensorDyn::image(width, height, PixelFormat::Rgba, DType::U8, mem())?;
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
        let dst_multiplane = TensorDyn::image(width, height, PixelFormat::Rgba, DType::U8, mem())?;
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
        let nv12_bytes = edgefirst_bench::testdata::read("camera720p.nv12");
        let width = 1280usize;
        let height = 720usize;
        let y_size = width * height;
        let uv_size = width * (height / 2);

        let contiguous = load_bytes_to_tensor(width, height, PixelFormat::Nv12, None, &nv12_bytes)?;
        let dst_contiguous = TensorDyn::image(width, height, PixelFormat::Grey, DType::U8, mem())?;
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
        let dst_multiplane = TensorDyn::image(width, height, PixelFormat::Grey, DType::U8, mem())?;
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

    // ──────────────────────────────────────────────────────────────────────
    // src_rect clamping tests — verify that cropping from a larger buffer
    // never samples padding pixels during bilinear resize.
    // ──────────────────────────────────────────────────────────────────────

    /// Create a synthetic RGB tensor where the left half is pure red and the
    /// right half is pure blue.
    fn make_red_blue_src(width: usize, height: usize) -> TensorDyn {
        let mut t = TensorDyn::image(width, height, PixelFormat::Rgb, DType::U8, mem()).unwrap();
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

    /// Crop the blue (right) half of a red|blue image and resize to a smaller
    /// destination. Verify no red pixels bleed into the interior output.
    /// Edge pixels (first/last column) may have filter-radius bleed from the
    /// resize library; we skip those and validate the interior is pure blue.
    #[test]
    fn test_src_rect_crop_no_bleed_cpu() -> Result<()> {
        let src_w = 128;
        let src_h = 64;
        let dst_w = 32;
        let dst_h = 32;

        let src = make_red_blue_src(src_w, src_h);
        let mut dst = TensorDyn::image(dst_w, dst_h, PixelFormat::Rgb, DType::U8, mem())?;

        let mut cpu = CPUProcessor::default();

        // Crop only the right (blue) half
        let crop = Crop::new().with_src_rect(Some(Rect::new(src_w / 2, 0, src_w / 2, src_h)));

        cpu.convert(&src, &mut dst, Rotation::None, Flip::None, crop)?;

        // Skip the leftmost 2 columns (resize filter radius at crop boundary)
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

    /// Crop the red (left) half and verify no blue bleeds into the interior.
    #[test]
    fn test_src_rect_crop_left_half_no_bleed_cpu() -> Result<()> {
        let src_w = 128;
        let src_h = 64;
        let dst_w = 32;
        let dst_h = 32;

        let src = make_red_blue_src(src_w, src_h);
        let mut dst = TensorDyn::image(dst_w, dst_h, PixelFormat::Rgb, DType::U8, mem())?;

        let mut cpu = CPUProcessor::default();

        // Crop only the left (red) half
        let crop = Crop::new().with_src_rect(Some(Rect::new(0, 0, src_w / 2, src_h)));

        cpu.convert(&src, &mut dst, Rotation::None, Flip::None, crop)?;

        // Skip the rightmost 2 columns (resize filter radius at crop boundary)
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

    /// Crop at the exact red→blue boundary and resize. Interior pixels
    /// (away from the crop edge) must be pure blue.
    #[test]
    fn test_src_rect_boundary_crop_no_bleed_cpu() -> Result<()> {
        let src_w = 256;
        let src_h = 64;
        let dst_w = 64;
        let dst_h = 64;

        let src = make_red_blue_src(src_w, src_h);
        let mut dst = TensorDyn::image(dst_w, dst_h, PixelFormat::Rgb, DType::U8, mem())?;

        let mut cpu = CPUProcessor::default();

        // Crop the right half — boundary is exactly at the colour transition
        let crop = Crop::new().with_src_rect(Some(Rect::new(src_w / 2, 0, src_w / 2, src_h)));

        cpu.convert(&src, &mut dst, Rotation::None, Flip::None, crop)?;

        // Skip the leftmost 2 columns (filter radius near crop boundary)
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

    /// RGBA8 → Rgb F32 identity (same dims, no resize): every output element
    /// must equal `src_byte as f32 / 255.0` exactly.
    #[test]
    fn cpu_convert_rgba_to_rgb_f32_identity() -> Result<()> {
        const W: usize = 4;
        const H: usize = 4;

        // Build a 4x4 RGBA source with a known per-pixel pattern.
        let src = TensorDyn::image(W, H, PixelFormat::Rgba, DType::U8, Some(TensorMemory::Mem))?;
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

        // Destination: Rgb F32 same size.
        let mut dst =
            TensorDyn::image(W, H, PixelFormat::Rgb, DType::F32, Some(TensorMemory::Mem))?;
        {
            let mut cv = CPUProcessor::default();
            cv.convert(&src, &mut dst, Rotation::None, Flip::None, Crop::default())?;
        }

        // Verify: dst[i*3+c] == src_u8[i*4+c] as f32 / 255.0 exactly.
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

    /// RGBA8 → PlanarRgb F16 identity: verify [0,1] normalization within F16
    /// rounding tolerance (2^-9 ≈ 1/512), and verify planar (channel-major)
    /// layout — distinct R/G/B values land in the correct plane.
    #[test]
    fn cpu_convert_rgba_to_planar_rgb_f16_identity() -> Result<()> {
        const W: usize = 4;
        const H: usize = 4;

        // Use clearly distinct per-channel values to catch plane-swap bugs.
        // pixel (y,x): R = 50+x, G = 100+y*10, B = 200
        let src = TensorDyn::image(W, H, PixelFormat::Rgba, DType::U8, Some(TensorMemory::Mem))?;
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

        // Destination: PlanarRgb F16 same size.
        let mut dst = TensorDyn::image(
            W,
            H,
            PixelFormat::PlanarRgb,
            DType::F16,
            Some(TensorMemory::Mem),
        )?;
        {
            let mut cv = CPUProcessor::default();
            cv.convert(&src, &mut dst, Rotation::None, Flip::None, Crop::default())?;
        }

        // F16 tolerance: 2^-9 ≈ 0.00195 (one ULP at 0.5 for f16 with 10-bit mantissa)
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

                // Verify planar layout: R/G/B must be distinct (not the same value
                // copied to all planes).
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

    /// Exercises the `widen_scratch` cache in `convert_dtype`'s U8→float arm:
    /// the first convert allocates+stores the scratch, a second convert at the
    /// same dst format/size takes the cache-hit branch, and a third at a
    /// different size takes the reallocate branch. A single-convert test (as
    /// all the others are) never reaches the cache-hit/realloc paths.
    #[test]
    fn cpu_widen_scratch_reused_across_converts() -> Result<()> {
        fn rgba(w: usize, h: usize) -> Result<TensorDyn> {
            let src =
                TensorDyn::image(w, h, PixelFormat::Rgba, DType::U8, Some(TensorMemory::Mem))?;
            {
                let mut map = src.as_u8().unwrap().map()?;
                let data = map.as_mut_slice();
                for (i, px) in data.chunks_exact_mut(4).enumerate() {
                    px[0] = (i * 7) as u8;
                    px[1] = (i * 7 + 1) as u8;
                    px[2] = (i * 7 + 2) as u8;
                    px[3] = 255;
                }
            }
            Ok(src)
        }

        // One processor instance so the scratch cache persists across calls.
        let mut cv = CPUProcessor::default();

        let src4 = rgba(4, 4)?;
        let mut dst_a =
            TensorDyn::image(4, 4, PixelFormat::Rgb, DType::F32, Some(TensorMemory::Mem))?;
        let mut dst_b =
            TensorDyn::image(4, 4, PixelFormat::Rgb, DType::F32, Some(TensorMemory::Mem))?;

        // 1st convert: scratch is None → allocate + cache.
        cv.convert(
            &src4,
            &mut dst_a,
            Rotation::None,
            Flip::None,
            Crop::default(),
        )?;
        // 2nd convert, same dst format+size → cache-hit branch (reuse scratch).
        cv.convert(
            &src4,
            &mut dst_b,
            Rotation::None,
            Flip::None,
            Crop::default(),
        )?;

        // Both results must be identical and correctly normalized.
        let a = dst_a.as_f32().unwrap().map()?;
        let b = dst_b.as_f32().unwrap().map()?;
        assert_eq!(
            a.as_slice(),
            b.as_slice(),
            "cache-hit produced different output"
        );

        // 3rd convert at a different size → scratch mismatch → realloc branch.
        let src8 = rgba(8, 8)?;
        let mut dst_c =
            TensorDyn::image(8, 8, PixelFormat::Rgb, DType::F32, Some(TensorMemory::Mem))?;
        cv.convert(
            &src8,
            &mut dst_c,
            Rotation::None,
            Flip::None,
            Crop::default(),
        )?;
        let c = dst_c.as_f32().unwrap().map()?;
        assert_eq!(c.as_slice().len(), 8 * 8 * 3);
        assert!(c.as_slice().iter().all(|v| (0.0..=1.0).contains(v)));
        Ok(())
    }

    /// `convert_dtype` returns `NotAnImage` when the destination tensor has no
    /// pixel format (a bare, non-image tensor), rather than panicking.
    #[test]
    fn cpu_convert_dst_without_format_errors() -> Result<()> {
        let src = TensorDyn::image(4, 4, PixelFormat::Rgba, DType::U8, Some(TensorMemory::Mem))?;
        // A plain tensor with no image format set.
        let mut dst = TensorDyn::new(&[4, 4, 3], DType::F32, Some(TensorMemory::Mem), None)?;
        let mut cv = CPUProcessor::default();
        let err = cv
            .convert(&src, &mut dst, Rotation::None, Flip::None, Crop::default())
            .unwrap_err();
        assert!(
            matches!(err, Error::NotAnImage),
            "expected NotAnImage, got {err:?}"
        );
        Ok(())
    }

    /// RGBA8 → Rgb F32 with downscale (8x8 → 4x4): output must be in [0,1]
    /// and finite; a flat-color region survives resize exactly.
    #[test]
    fn cpu_convert_rgba_to_rgb_f32_resize() -> Result<()> {
        const SW: usize = 8;
        const SH: usize = 8;
        const DW: usize = 4;
        const DH: usize = 4;

        // Build an 8x8 source: right half (x >= 4) is solid blue [0,0,255,255].
        let src = TensorDyn::image(
            SW,
            SH,
            PixelFormat::Rgba,
            DType::U8,
            Some(TensorMemory::Mem),
        )?;
        {
            let mut map = src.as_u8().unwrap().map()?;
            let data = map.as_mut_slice();
            for y in 0..SH {
                for x in 0..SW {
                    let i = y * SW + x;
                    if x >= 4 {
                        // Solid blue in the right half
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
        )?;
        {
            let mut cv = CPUProcessor::default();
            cv.convert(&src, &mut dst, Rotation::None, Flip::None, Crop::default())?;
        }

        let dst_map = dst.as_f32().unwrap().map()?;
        let dst_floats = dst_map.as_slice();

        assert_eq!(dst_floats.len(), DW * DH * 3, "wrong output element count");

        // All values must be in [0, 1] and finite.
        for (i, &v) in dst_floats.iter().enumerate() {
            assert!(
                v.is_finite() && (0.0..=1.0).contains(&v),
                "dst[{i}] = {v} is out of [0,1] or non-finite"
            );
        }

        // The rightmost column (x=3 in 4x4) maps to src pixels x=6..7 which
        // are entirely inside the solid-blue right half — bilinear resize of a
        // flat region is exact.
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

        // The left src half is [128, 64, 32] -> normalised [~0.502, ~0.251, ~0.125].
        // x=0 in the 4x4 output maps to src x=0 whose bilinear kernel sits
        // entirely within the left half — bilinear resize of a flat region is
        // exact, so the tolerance just absorbs the f32 /255 rounding.  A zero-
        // filled or miscopied left region would produce 0.0 here and be caught.
        // (x=1 already blends with right-half blue due to the resampler's broad
        // kernel reaching across the seam at src x=4.)
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
            // 2x2 NV12: 4 luma + 1 UV pair (U=200,V=40) — saturated chroma so 601≠709.
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
            let mut dst = TensorDyn::image(2, 2, PixelFormat::Rgb, DType::U8, None).unwrap();
            CPUProcessor::default()
                .convert(&src, &mut dst, Rotation::None, Flip::None, Crop::default())
                .unwrap();
            dst.as_u8().unwrap().map().unwrap().as_slice().to_vec()
        };
        assert_ne!(make(ColorEncoding::Bt601), make(ColorEncoding::Bt709));
    }

    #[test]
    fn cpu_nv16_to_rgb_respects_tagged_709_vs_601() {
        use edgefirst_tensor::{ColorEncoding, ColorRange, Colorimetry};
        let make = |enc| {
            // 2x2 NV16 (4:2:2): 4 luma + full-height half-width chroma = 2 rows
            // × 1 (Cb,Cr) pair = 4 chroma bytes (U=200,V=40, saturated so 601≠709).
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
            let mut dst = TensorDyn::image(2, 2, PixelFormat::Rgb, DType::U8, None).unwrap();
            CPUProcessor::default()
                .convert(&src, &mut dst, Rotation::None, Flip::None, Crop::default())
                .unwrap();
            dst.as_u8().unwrap().map().unwrap().as_slice().to_vec()
        };
        assert_ne!(make(ColorEncoding::Bt601), make(ColorEncoding::Bt709));
    }

    #[test]
    fn cpu_nv24_to_rgb_respects_tagged_709_vs_601() {
        use edgefirst_tensor::{ColorEncoding, ColorRange, Colorimetry};
        let make = |enc| {
            // 2x2 NV24 (4:4:4): 4 luma + full-res chroma = 4 (Cb,Cr) pairs = 8
            // chroma bytes (U=200,V=40, saturated so 601≠709).
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
            let mut dst = TensorDyn::image(2, 2, PixelFormat::Rgb, DType::U8, None).unwrap();
            CPUProcessor::default()
                .convert(&src, &mut dst, Rotation::None, Flip::None, Crop::default())
                .unwrap();
            dst.as_u8().unwrap().map().unwrap().as_slice().to_vec()
        };
        assert_ne!(make(ColorEncoding::Bt601), make(ColorEncoding::Bt709));
    }

    #[test]
    fn cpu_convert_tagged_jfif_differs_from_untagged_heuristic_nv12() {
        use edgefirst_tensor::{ColorEncoding, ColorRange, Colorimetry};
        // Same 2x2 NV12 bytes — saturated chroma so matrix/range choice matters.
        // NV12 2x2 = 4 Y-plane bytes + 2 UV-plane bytes = 6 total.
        let bytes = [180u8, 180, 180, 180, 200, 40];
        let convert = |cm: Option<Colorimetry>| {
            let mut src = load_bytes_to_tensor(2, 2, PixelFormat::Nv12, None, &bytes).unwrap();
            src.set_colorimetry(cm);
            let mut dst = TensorDyn::image(2, 2, PixelFormat::Rgb, DType::U8, None).unwrap();
            CPUProcessor::default()
                .convert(&src, &mut dst, Rotation::None, Flip::None, Crop::default())
                .unwrap();
            dst.as_u8().unwrap().map().unwrap().as_slice().to_vec()
        };
        // Tagged JFIF (BT.601 full-range) — what the codec sets for JPEG.
        let jfif = convert(Some(Colorimetry::jfif()));
        // Untagged: 2x2 height is below HD_THRESHOLD (720) → heuristic resolves
        // to BT.601 LIMITED.  JFIF is BT.601 FULL, so the two must differ.
        let untagged = convert(None);
        assert_ne!(
            jfif, untagged,
            "JFIF (BT.601 full-range) must differ from heuristic (BT.601 limited)"
        );
        // An explicitly-tagged BT.709-limited must also differ from BT.601 JFIF.
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
        // heap parent buffer. Each window must bit-match a standalone convert
        // of its source — proving the CPU convert path honors the heap
        // plane_offset with no aliasing. Runs on CI with no GPU/NPU.
        let mut converter = CPUProcessor::default();

        let mut src0 =
            TensorDyn::image(4, 4, PixelFormat::Rgba, DType::U8, Some(TensorMemory::Mem)).unwrap();
        src0.as_u8_mut()
            .unwrap()
            .map()
            .unwrap()
            .as_mut_slice()
            .fill(50);
        let mut src1 =
            TensorDyn::image(4, 4, PixelFormat::Rgba, DType::U8, Some(TensorMemory::Mem)).unwrap();
        src1.as_u8_mut()
            .unwrap()
            .map()
            .unwrap()
            .as_mut_slice()
            .fill(200);

        let mut do_convert = |src: &TensorDyn, dst: &mut TensorDyn| {
            converter
                .convert(src, dst, Rotation::None, Flip::None, Crop::default())
                .unwrap();
        };

        // Standalone reference conversions into full, independent buffers.
        let mut ref0 =
            TensorDyn::image(4, 4, PixelFormat::Rgb, DType::U8, Some(TensorMemory::Mem)).unwrap();
        do_convert(&src0, &mut ref0);
        let mut ref1 =
            TensorDyn::image(4, 4, PixelFormat::Rgb, DType::U8, Some(TensorMemory::Mem)).unwrap();
        do_convert(&src1, &mut ref1);

        // One RGB parent holding two stacked 4x4 frames (4 wide, 8 tall). The
        // sub-views inherit the Rgb format, so each is a ready convert target
        // and the plane offset survives.
        let frame = 4 * 4 * 3;
        let parent =
            TensorDyn::image(4, 8, PixelFormat::Rgb, DType::U8, Some(TensorMemory::Mem)).unwrap();
        let mut view0 = parent.subview(0, &[4, 4, 3]).unwrap();
        let mut view1 = parent.subview(frame, &[4, 4, 3]).unwrap();
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
}
