// SPDX-FileCopyrightText: Copyright 2025 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod cpu_tests {

    use crate::{
        CPUProcessor, Crop, Error, Flip, ImageProcessorTrait, Rect, Result, Rotation, TensorImage,
        TensorImageDst, TensorImageRef, BGRA, GREY, NV12, NV16, PLANAR_RGB, PLANAR_RGBA, RGB, RGBA,
        VYUY, YUYV,
    };
    use edgefirst_decoder::DetectBox;
    use edgefirst_tensor::{Tensor, TensorMapTrait, TensorMemory, TensorTrait};
    use four_char_code::FourCharCode;
    use image::buffer::ConvertBuffer;

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
        img1: &TensorImage,
        img2: &TensorImage,
        threshold: f64,
        name: &str,
    ) {
        assert_eq!(img1.height(), img2.height(), "Heights differ");
        assert_eq!(img1.width(), img2.width(), "Widths differ");

        let mut img_rgb1 = TensorImage::new(img1.width(), img1.height(), RGBA, None).unwrap();
        let mut img_rgb2 = TensorImage::new(img1.width(), img1.height(), RGBA, None).unwrap();
        CPUProcessor::convert_format(img1, &mut img_rgb1).unwrap();
        CPUProcessor::convert_format(img2, &mut img_rgb2).unwrap();

        let image1 = image::RgbaImage::from_vec(
            img_rgb1.width() as u32,
            img_rgb1.height() as u32,
            img_rgb1.tensor().map().unwrap().to_vec(),
        )
        .unwrap();

        let image2 = image::RgbaImage::from_vec(
            img_rgb2.width() as u32,
            img_rgb2.height() as u32,
            img_rgb2.tensor().map().unwrap().to_vec(),
        )
        .unwrap();

        let similarity = image_compare::gray_similarity_structure(
            &image_compare::Algorithm::RootMeanSquared,
            &image1.convert(),
            &image2.convert(),
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
        img1: &TensorImage,
        img2: &TensorImage,
        threshold: f64,
        name: &str,
    ) {
        assert_eq!(img1.height(), img2.height(), "Heights differ");
        assert_eq!(img1.width(), img2.width(), "Widths differ");

        let mut img_rgb1 = TensorImage::new(img1.width(), img1.height(), RGB, None).unwrap();
        let mut img_rgb2 = TensorImage::new(img1.width(), img1.height(), RGB, None).unwrap();
        CPUProcessor::convert_format(img1, &mut img_rgb1).unwrap();
        CPUProcessor::convert_format(img2, &mut img_rgb2).unwrap();

        let image1 = image::RgbImage::from_vec(
            img_rgb1.width() as u32,
            img_rgb1.height() as u32,
            img_rgb1.tensor().map().unwrap().to_vec(),
        )
        .unwrap();

        let image2 = image::RgbImage::from_vec(
            img_rgb2.width() as u32,
            img_rgb2.height() as u32,
            img_rgb2.tensor().map().unwrap().to_vec(),
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
        fourcc: FourCharCode,
        memory: Option<TensorMemory>,
        bytes: &[u8],
    ) -> Result<TensorImage, Error> {
        log::debug!("Current function is {}", function!());
        let src = TensorImage::new(width, height, fourcc, memory)?;
        src.tensor().map()?.as_mut_slice()[0..bytes.len()].copy_from_slice(bytes);
        Ok(src)
    }

    macro_rules! generate_conversion_tests {
        (
        $src_fmt:ident,  $src_file:expr, $dst_fmt:ident, $dst_file:expr
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

            let mut converted = TensorImage::new(src.width(), src.height(), dst.fourcc(), None)?;

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
        $src_fmt:ident,  $src_file:expr, $dst_fmt:ident, $dst_file:expr
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

            let mut converted = TensorImage::new(src.width(), src.height(), dst.fourcc(), None)?;

            converter.convert(
                &src,
                &mut converted,
                Rotation::None,
                Flip::None,
                Crop::default(),
            )?;

            compare_images_convert_to_grey(&dst, &converted, 0.985, function!());

            Ok(())
        }};
    }

    // let mut dsts = [yuyv, rgb, rgba, grey, nv16, planar_rgb, planar_rgba];

    #[test]
    fn test_cpu_yuyv_to_yuyv() -> Result<()> {
        generate_conversion_tests!(YUYV, "camera720p.yuyv", YUYV, "camera720p.yuyv")
    }

    #[test]
    fn test_cpu_yuyv_to_rgb() -> Result<()> {
        generate_conversion_tests!(YUYV, "camera720p.yuyv", RGB, "camera720p.rgb")
    }

    #[test]
    fn test_cpu_yuyv_to_rgba() -> Result<()> {
        generate_conversion_tests!(YUYV, "camera720p.yuyv", RGBA, "camera720p.rgba")
    }

    #[test]
    fn test_cpu_yuyv_to_grey() -> Result<()> {
        generate_conversion_tests!(YUYV, "camera720p.yuyv", GREY, "camera720p.y800")
    }

    #[test]
    fn test_cpu_yuyv_to_nv16() -> Result<()> {
        generate_conversion_tests!(YUYV, "camera720p.yuyv", NV16, "camera720p.nv16")
    }

    #[test]
    fn test_cpu_yuyv_to_planar_rgb() -> Result<()> {
        generate_conversion_tests!(YUYV, "camera720p.yuyv", PLANAR_RGB, "camera720p.8bps")
    }

    #[test]
    fn test_cpu_yuyv_to_planar_rgba() -> Result<()> {
        generate_conversion_tests!(YUYV, "camera720p.yuyv", PLANAR_RGBA, "camera720p.8bpa")
    }

    #[test]
    fn test_cpu_rgb_to_yuyv() -> Result<()> {
        generate_conversion_tests!(RGB, "camera720p.rgb", YUYV, "camera720p.yuyv")
    }

    #[test]
    fn test_cpu_rgb_to_rgb() -> Result<()> {
        generate_conversion_tests!(RGB, "camera720p.rgb", RGB, "camera720p.rgb")
    }

    #[test]
    fn test_cpu_rgb_to_rgba() -> Result<()> {
        generate_conversion_tests!(RGB, "camera720p.rgb", RGBA, "camera720p.rgba")
    }

    #[test]
    fn test_cpu_rgb_to_grey() -> Result<()> {
        generate_conversion_tests!(RGB, "camera720p.rgb", GREY, "camera720p.y800")
    }

    #[test]
    fn test_cpu_rgb_to_nv16() -> Result<()> {
        generate_conversion_tests!(RGB, "camera720p.rgb", NV16, "camera720p.nv16")
    }

    #[test]
    fn test_cpu_rgb_to_planar_rgb() -> Result<()> {
        generate_conversion_tests!(RGB, "camera720p.rgb", PLANAR_RGB, "camera720p.8bps")
    }

    #[test]
    fn test_cpu_rgb_to_planar_rgba() -> Result<()> {
        generate_conversion_tests!(RGB, "camera720p.rgb", PLANAR_RGBA, "camera720p.8bpa")
    }

    #[test]
    fn test_cpu_rgba_to_yuyv() -> Result<()> {
        generate_conversion_tests!(RGBA, "camera720p.rgba", YUYV, "camera720p.yuyv")
    }

    #[test]
    fn test_cpu_rgba_to_rgb() -> Result<()> {
        generate_conversion_tests!(RGBA, "camera720p.rgba", RGB, "camera720p.rgb")
    }

    #[test]
    fn test_cpu_rgba_to_rgba() -> Result<()> {
        generate_conversion_tests!(RGBA, "camera720p.rgba", RGBA, "camera720p.rgba")
    }

    #[test]
    fn test_cpu_rgba_to_grey() -> Result<()> {
        generate_conversion_tests!(RGBA, "camera720p.rgba", GREY, "camera720p.y800")
    }

    #[test]
    fn test_cpu_rgba_to_nv16() -> Result<()> {
        generate_conversion_tests!(RGBA, "camera720p.rgba", NV16, "camera720p.nv16")
    }

    #[test]
    fn test_cpu_rgba_to_planar_rgb() -> Result<()> {
        generate_conversion_tests!(RGBA, "camera720p.rgba", PLANAR_RGB, "camera720p.8bps")
    }

    #[test]
    fn test_cpu_rgba_to_planar_rgba() -> Result<()> {
        generate_conversion_tests!(RGBA, "camera720p.rgba", PLANAR_RGBA, "camera720p.8bpa")
    }

    #[test]
    fn test_cpu_nv12_to_rgb() -> Result<()> {
        generate_conversion_tests!(NV12, "camera720p.nv12", RGB, "camera720p.rgb")
    }

    #[test]
    fn test_cpu_nv12_to_yuyv() -> Result<()> {
        generate_conversion_tests!(NV12, "camera720p.nv12", YUYV, "camera720p.yuyv")
    }

    #[test]
    fn test_cpu_nv12_to_rgba() -> Result<()> {
        generate_conversion_tests!(NV12, "camera720p.nv12", RGBA, "camera720p.rgba")
    }

    #[test]
    fn test_cpu_nv12_to_grey() -> Result<()> {
        generate_conversion_tests!(NV12, "camera720p.nv12", GREY, "camera720p.y800")
    }

    #[test]
    fn test_cpu_nv12_to_nv16() -> Result<()> {
        generate_conversion_tests!(NV12, "camera720p.nv12", NV16, "camera720p.nv16")
    }

    #[test]
    fn test_cpu_nv12_to_planar_rgb() -> Result<()> {
        generate_conversion_tests!(NV12, "camera720p.nv12", PLANAR_RGB, "camera720p.8bps")
    }

    #[test]
    fn test_cpu_nv12_to_planar_rgba() -> Result<()> {
        generate_conversion_tests!(NV12, "camera720p.nv12", PLANAR_RGBA, "camera720p.8bpa")
    }

    #[test]
    fn test_cpu_grey_to_yuyv() -> Result<()> {
        generate_conversion_tests_greyscale!(GREY, "camera720p.y800", YUYV, "camera720p.yuyv")
    }

    #[test]
    fn test_cpu_grey_to_rgb() -> Result<()> {
        generate_conversion_tests_greyscale!(GREY, "camera720p.y800", RGB, "camera720p.rgb")
    }

    #[test]
    fn test_cpu_grey_to_rgba() -> Result<()> {
        generate_conversion_tests_greyscale!(GREY, "camera720p.y800", RGBA, "camera720p.rgba")
    }

    #[test]
    fn test_cpu_grey_to_grey() -> Result<()> {
        generate_conversion_tests_greyscale!(GREY, "camera720p.y800", GREY, "camera720p.y800")
    }

    #[test]
    fn test_cpu_grey_to_nv16() -> Result<()> {
        generate_conversion_tests_greyscale!(GREY, "camera720p.y800", NV16, "camera720p.nv16")
    }

    #[test]
    fn test_cpu_grey_to_planar_rgb() -> Result<()> {
        generate_conversion_tests_greyscale!(GREY, "camera720p.y800", PLANAR_RGB, "camera720p.8bps")
    }

    #[test]
    fn test_cpu_grey_to_planar_rgba() -> Result<()> {
        generate_conversion_tests_greyscale!(
            GREY,
            "camera720p.y800",
            PLANAR_RGBA,
            "camera720p.8bpa"
        )
    }

    #[test]
    fn test_cpu_nearest() -> Result<()> {
        // Load source
        let src = load_bytes_to_tensor(2, 1, RGB, None, &[0, 0, 0, 255, 255, 255])?;

        let mut converter = CPUProcessor::new_nearest();

        let mut converted = TensorImage::new(4, 1, RGB, None)?;

        converter.convert(
            &src,
            &mut converted,
            Rotation::None,
            Flip::None,
            Crop::default(),
        )?;

        assert_eq!(
            &converted.tensor().map()?.as_slice(),
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
            RGBA,
            None,
            &[0, 0, 0, 255, 1, 1, 1, 255, 2, 2, 2, 255, 3, 3, 3, 255],
        )?;

        let mut converter = CPUProcessor::default();

        let mut converted = TensorImage::new(4, 4, RGBA, None)?;

        converter.convert(
            &src,
            &mut converted,
            Rotation::Clockwise90,
            Flip::None,
            Crop::default(),
        )?;

        assert_eq!(&converted.tensor().map()?.as_slice()[0..4], &[2, 2, 2, 255]);
        assert_eq!(
            &converted.tensor().map()?.as_slice()[12..16],
            &[0, 0, 0, 255]
        );
        assert_eq!(
            &converted.tensor().map()?.as_slice()[48..52],
            &[3, 3, 3, 255]
        );

        assert_eq!(
            &converted.tensor().map()?.as_slice()[60..64],
            &[1, 1, 1, 255]
        );

        Ok(())
    }

    #[test]
    fn test_cpu_rotate_ccw() -> Result<()> {
        // Load source
        let src = load_bytes_to_tensor(
            2,
            2,
            RGBA,
            None,
            &[0, 0, 0, 255, 1, 1, 1, 255, 2, 2, 2, 255, 3, 3, 3, 255],
        )?;

        let mut converter = CPUProcessor::default();

        let mut converted = TensorImage::new(4, 4, RGBA, None)?;

        converter.convert(
            &src,
            &mut converted,
            Rotation::CounterClockwise90,
            Flip::None,
            Crop::default(),
        )?;

        assert_eq!(&converted.tensor().map()?.as_slice()[0..4], &[1, 1, 1, 255]);
        assert_eq!(
            &converted.tensor().map()?.as_slice()[12..16],
            &[3, 3, 3, 255]
        );
        assert_eq!(
            &converted.tensor().map()?.as_slice()[48..52],
            &[0, 0, 0, 255]
        );

        assert_eq!(
            &converted.tensor().map()?.as_slice()[60..64],
            &[2, 2, 2, 255]
        );

        Ok(())
    }

    #[test]
    fn test_cpu_rotate_180() -> Result<()> {
        // Load source
        let src = load_bytes_to_tensor(
            2,
            2,
            RGBA,
            None,
            &[0, 0, 0, 255, 1, 1, 1, 255, 2, 2, 2, 255, 3, 3, 3, 255],
        )?;

        let mut converter = CPUProcessor::default();

        let mut converted = TensorImage::new(4, 4, RGBA, None)?;

        converter.convert(
            &src,
            &mut converted,
            Rotation::Rotate180,
            Flip::None,
            Crop::default(),
        )?;

        assert_eq!(&converted.tensor().map()?.as_slice()[0..4], &[3, 3, 3, 255]);
        assert_eq!(
            &converted.tensor().map()?.as_slice()[12..16],
            &[2, 2, 2, 255]
        );
        assert_eq!(
            &converted.tensor().map()?.as_slice()[48..52],
            &[1, 1, 1, 255]
        );

        assert_eq!(
            &converted.tensor().map()?.as_slice()[60..64],
            &[0, 0, 0, 255]
        );

        Ok(())
    }

    #[test]
    fn test_cpu_flip_v() -> Result<()> {
        // Load source
        let src = load_bytes_to_tensor(
            2,
            2,
            RGBA,
            None,
            &[0, 0, 0, 255, 1, 1, 1, 255, 2, 2, 2, 255, 3, 3, 3, 255],
        )?;

        let mut converter = CPUProcessor::default();

        let mut converted = TensorImage::new(4, 4, RGBA, None)?;

        converter.convert(
            &src,
            &mut converted,
            Rotation::None,
            Flip::Vertical,
            Crop::default(),
        )?;

        assert_eq!(&converted.tensor().map()?.as_slice()[0..4], &[2, 2, 2, 255]);
        assert_eq!(
            &converted.tensor().map()?.as_slice()[12..16],
            &[3, 3, 3, 255]
        );
        assert_eq!(
            &converted.tensor().map()?.as_slice()[48..52],
            &[0, 0, 0, 255]
        );

        assert_eq!(
            &converted.tensor().map()?.as_slice()[60..64],
            &[1, 1, 1, 255]
        );

        Ok(())
    }

    #[test]
    fn test_cpu_flip_h() -> Result<()> {
        // Load source
        let src = load_bytes_to_tensor(
            2,
            2,
            RGBA,
            None,
            &[0, 0, 0, 255, 1, 1, 1, 255, 2, 2, 2, 255, 3, 3, 3, 255],
        )?;

        let mut converter = CPUProcessor::default();

        let mut converted = TensorImage::new(4, 4, RGBA, None)?;

        converter.convert(
            &src,
            &mut converted,
            Rotation::None,
            Flip::Horizontal,
            Crop::default(),
        )?;

        assert_eq!(&converted.tensor().map()?.as_slice()[0..4], &[1, 1, 1, 255]);
        assert_eq!(
            &converted.tensor().map()?.as_slice()[12..16],
            &[0, 0, 0, 255]
        );
        assert_eq!(
            &converted.tensor().map()?.as_slice()[48..52],
            &[3, 3, 3, 255]
        );

        assert_eq!(
            &converted.tensor().map()?.as_slice()[60..64],
            &[2, 2, 2, 255]
        );

        Ok(())
    }

    #[test]
    fn test_cpu_src_crop() -> Result<()> {
        // Load source
        let src = load_bytes_to_tensor(2, 2, GREY, None, &[10, 20, 30, 40])?;

        let mut converter = CPUProcessor::default();

        let mut converted = TensorImage::new(2, 2, RGBA, None)?;

        converter.convert(
            &src,
            &mut converted,
            Rotation::None,
            Flip::None,
            Crop::new().with_src_rect(Some(Rect::new(0, 0, 1, 2))),
        )?;

        assert_eq!(
            converted.tensor().map()?.as_slice(),
            &[10, 10, 10, 255, 13, 13, 13, 255, 30, 30, 30, 255, 33, 33, 33, 255]
        );
        Ok(())
    }

    #[test]
    fn test_cpu_dst_crop() -> Result<()> {
        // Load source
        let src = load_bytes_to_tensor(2, 2, GREY, None, &[2, 4, 6, 8])?;

        let mut converter = CPUProcessor::default();

        let mut converted =
            load_bytes_to_tensor(2, 2, YUYV, None, &[200, 128, 200, 128, 200, 128, 200, 128])?;

        converter.convert(
            &src,
            &mut converted,
            Rotation::None,
            Flip::None,
            Crop::new().with_dst_rect(Some(Rect::new(0, 0, 2, 1))),
        )?;

        assert_eq!(
            converted.tensor().map()?.as_slice(),
            &[20, 128, 21, 128, 200, 128, 200, 128]
        );
        Ok(())
    }

    #[test]
    fn test_cpu_fill_rgba() -> Result<()> {
        // Load source
        let src = load_bytes_to_tensor(1, 1, RGBA, None, &[3, 3, 3, 255])?;

        let mut converter = CPUProcessor::default();

        let mut converted = TensorImage::new(2, 2, RGBA, None)?;

        converter.convert(
            &src,
            &mut converted,
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
            converted.tensor().map()?.as_slice(),
            &[255, 0, 0, 255, 255, 0, 0, 255, 255, 0, 0, 255, 3, 3, 3, 255]
        );
        Ok(())
    }

    #[test]
    fn test_cpu_fill_yuyv() -> Result<()> {
        // Load source
        let src = load_bytes_to_tensor(2, 1, RGBA, None, &[3, 3, 3, 255, 3, 3, 3, 255])?;

        let mut converter = CPUProcessor::default();

        let mut converted = TensorImage::new(2, 3, YUYV, None)?;

        converter.convert(
            &src,
            &mut converted,
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
            converted.tensor().map()?.as_slice(),
            &[63, 102, 63, 240, 19, 128, 19, 128, 63, 102, 63, 240]
        );
        Ok(())
    }

    #[test]
    fn test_cpu_fill_grey() -> Result<()> {
        // Load source
        let src = load_bytes_to_tensor(2, 1, RGBA, None, &[3, 3, 3, 255, 3, 3, 3, 255])?;

        let mut converter = CPUProcessor::default();

        let mut converted = TensorImage::new(2, 3, GREY, None)?;

        converter.convert(
            &src,
            &mut converted,
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
            converted.tensor().map()?.as_slice(),
            &[200, 200, 3, 3, 200, 200]
        );
        Ok(())
    }

    #[test]
    fn test_segmentation() {
        use edgefirst_decoder::Segmentation;
        use ndarray::Array3;

        let mut image = TensorImage::load(
            include_bytes!(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/../../testdata/giraffe.jpg"
            )),
            Some(RGBA),
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

        let mut renderer = CPUProcessor::new();
        renderer.draw_masks(&mut image, &[], &[seg]).unwrap();

        image.save_jpeg("test_segmentation.jpg", 80).unwrap();
    }

    #[test]
    fn test_segmentation_yolo() {
        use edgefirst_decoder::Segmentation;
        use ndarray::Array3;

        let mut image = TensorImage::load(
            include_bytes!(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/../../testdata/giraffe.jpg"
            )),
            Some(RGBA),
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
        renderer.draw_masks(&mut image, &[detect], &[seg]).unwrap();
        let expected = TensorImage::load(
            include_bytes!(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/../../testdata/output_render_cpu.jpg"
            )),
            Some(RGBA),
            None,
        )
        .unwrap();
        compare_images_convert_to_rgb(&image, &expected, 0.99, function!());
    }

    // =========================================================================
    // Generic Conversion Tests (TensorImageRef support)
    // =========================================================================

    #[test]
    fn test_convert_rgb_to_planar_rgb_generic() {
        // Create RGB source image
        let mut src = TensorImage::new(4, 4, RGB, None).unwrap();
        {
            let mut map = src.tensor_mut().map().unwrap();
            let data = map.as_mut_slice();
            // Fill with pattern: pixel 0 = [10, 20, 30], pixel 1 = [40, 50, 60], etc.
            for i in 0..16 {
                data[i * 3] = (i * 10) as u8;
                data[i * 3 + 1] = (i * 10 + 1) as u8;
                data[i * 3 + 2] = (i * 10 + 2) as u8;
            }
        }

        // Create planar RGB destination using TensorImageRef
        let mut tensor = Tensor::<u8>::new(&[3, 4, 4], None, None).unwrap();
        let mut dst = TensorImageRef::from_borrowed_tensor(&mut tensor, PLANAR_RGB).unwrap();

        CPUProcessor::convert_format_generic(&src, &mut dst).unwrap();

        // Verify the conversion - check first few pixels of each plane
        let map = dst.tensor().map().unwrap();
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
        // Create RGBA source image
        let mut src = TensorImage::new(4, 4, RGBA, None).unwrap();
        {
            let mut map = src.tensor_mut().map().unwrap();
            let data = map.as_mut_slice();
            // Fill with pattern
            for i in 0..16 {
                data[i * 4] = (i * 10) as u8; // R
                data[i * 4 + 1] = (i * 10 + 1) as u8; // G
                data[i * 4 + 2] = (i * 10 + 2) as u8; // B
                data[i * 4 + 3] = 255; // A (ignored)
            }
        }

        // Create planar RGB destination
        let mut tensor = Tensor::<u8>::new(&[3, 4, 4], None, None).unwrap();
        let mut dst = TensorImageRef::from_borrowed_tensor(&mut tensor, PLANAR_RGB).unwrap();

        CPUProcessor::convert_format_generic(&src, &mut dst).unwrap();

        // Verify the conversion
        let map = dst.tensor().map().unwrap();
        let data = map.as_slice();

        assert_eq!(data[0], 0); // R of pixel 0
        assert_eq!(data[16], 1); // G of pixel 0
        assert_eq!(data[32], 2); // B of pixel 0
    }

    #[test]
    fn test_copy_image_generic_same_format() {
        // Create source image with data
        let mut src = TensorImage::new(4, 4, RGB, None).unwrap();
        {
            let mut map = src.tensor_mut().map().unwrap();
            let data = map.as_mut_slice();
            for (i, byte) in data.iter_mut().enumerate() {
                *byte = (i % 256) as u8;
            }
        }

        // Create destination tensor
        let mut tensor = Tensor::<u8>::new(&[4, 4, 3], None, None).unwrap();
        let mut dst = TensorImageRef::from_borrowed_tensor(&mut tensor, RGB).unwrap();

        CPUProcessor::convert_format_generic(&src, &mut dst).unwrap();

        // Verify data was copied
        let src_map = src.tensor().map().unwrap();
        let dst_map = dst.tensor().map().unwrap();
        assert_eq!(src_map.as_slice(), dst_map.as_slice());
    }

    #[test]
    fn test_convert_format_generic_unsupported() {
        // Try unsupported conversion (NV12 to PLANAR_RGB)
        let src = TensorImage::new(8, 8, NV12, None).unwrap();
        let mut tensor = Tensor::<u8>::new(&[3, 8, 8], None, None).unwrap();
        let mut dst = TensorImageRef::from_borrowed_tensor(&mut tensor, PLANAR_RGB).unwrap();

        let result = CPUProcessor::convert_format_generic(&src, &mut dst);
        assert!(result.is_err());
        assert!(matches!(result, Err(Error::NotSupported(_))));
    }

    #[test]
    fn test_fill_image_outside_crop_generic_rgba() {
        let mut tensor = Tensor::<u8>::new(&[4, 4, 4], None, None).unwrap();
        // Initialize to zeros
        tensor.map().unwrap().as_mut_slice().fill(0);

        let mut dst = TensorImageRef::from_borrowed_tensor(&mut tensor, RGBA).unwrap();

        // Fill outside a 2x2 crop in the center with red
        let crop = Rect::new(1, 1, 2, 2);
        CPUProcessor::fill_image_outside_crop_generic(&mut dst, [255, 0, 0, 255], crop).unwrap();

        let map = dst.tensor().map().unwrap();
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
        let mut tensor = Tensor::<u8>::new(&[4, 4, 3], None, None).unwrap();
        tensor.map().unwrap().as_mut_slice().fill(0);

        let mut dst = TensorImageRef::from_borrowed_tensor(&mut tensor, RGB).unwrap();

        let crop = Rect::new(1, 1, 2, 2);
        CPUProcessor::fill_image_outside_crop_generic(&mut dst, [0, 255, 0, 255], crop).unwrap();

        let map = dst.tensor().map().unwrap();
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
        let mut tensor = Tensor::<u8>::new(&[3, 4, 4], None, None).unwrap();
        tensor.map().unwrap().as_mut_slice().fill(0);

        let mut dst = TensorImageRef::from_borrowed_tensor(&mut tensor, PLANAR_RGB).unwrap();

        let crop = Rect::new(1, 1, 2, 2);
        CPUProcessor::fill_image_outside_crop_generic(&mut dst, [128, 64, 32, 255], crop).unwrap();

        let map = dst.tensor().map().unwrap();
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
        let src = TensorImage::new(2, 1, RGBA, Some(TensorMemory::Mem)).unwrap();
        {
            let mut map = src.tensor().map().unwrap();
            let buf = map.as_mut_slice();
            buf[0..4].copy_from_slice(&[10, 20, 30, 255]);
            buf[4..8].copy_from_slice(&[40, 50, 60, 128]);
        }
        let mut dst = TensorImage::new(2, 1, BGRA, Some(TensorMemory::Mem)).unwrap();
        CPUProcessor::convert_format(&src, &mut dst).unwrap();
        let map = dst.tensor().map().unwrap();
        let buf = map.as_slice();
        // BGRA byte order: [B, G, R, A]
        assert_eq!(&buf[0..4], &[30, 20, 10, 255]);
        assert_eq!(&buf[4..8], &[60, 50, 40, 128]);
    }

    #[test]
    fn test_convert_rgb_to_bgra() {
        // Convert RGB→RGBA and RGB→BGRA, verify R↔B swap matches
        let src = TensorImage::new(2, 1, RGB, Some(TensorMemory::Mem)).unwrap();
        {
            let mut map = src.tensor().map().unwrap();
            let buf = map.as_mut_slice();
            buf[0..3].copy_from_slice(&[100, 150, 200]);
            buf[3..6].copy_from_slice(&[50, 75, 25]);
        }
        let mut rgba_dst = TensorImage::new(2, 1, RGBA, Some(TensorMemory::Mem)).unwrap();
        CPUProcessor::convert_format(&src, &mut rgba_dst).unwrap();

        let mut bgra_dst = TensorImage::new(2, 1, BGRA, Some(TensorMemory::Mem)).unwrap();
        CPUProcessor::convert_format(&src, &mut bgra_dst).unwrap();

        assert_bgra_matches_rgba(&bgra_dst, &rgba_dst);

        // Also verify the B,G,R channels are correct (alpha may vary)
        let map = bgra_dst.tensor().map().unwrap();
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
        let src = TensorImage::new(2, 1, GREY, Some(TensorMemory::Mem)).unwrap();
        {
            let mut map = src.tensor().map().unwrap();
            let buf = map.as_mut_slice();
            buf[0] = 128;
            buf[1] = 64;
        }
        let mut dst = TensorImage::new(2, 1, BGRA, Some(TensorMemory::Mem)).unwrap();
        CPUProcessor::convert_format(&src, &mut dst).unwrap();
        let map = dst.tensor().map().unwrap();
        let buf = map.as_slice();
        // Grey→BGRA: all channels same value, A=255; R↔B swap is no-op on grey
        assert_eq!(&buf[0..4], &[128, 128, 128, 255]);
        assert_eq!(&buf[4..8], &[64, 64, 64, 255]);
    }

    #[test]
    fn test_convert_bgra_to_bgra_copy() {
        // Verify BGRA→BGRA is a straight copy
        let src = TensorImage::new(2, 1, BGRA, Some(TensorMemory::Mem)).unwrap();
        {
            let mut map = src.tensor().map().unwrap();
            let buf = map.as_mut_slice();
            buf[0..4].copy_from_slice(&[10, 20, 30, 255]);
            buf[4..8].copy_from_slice(&[40, 50, 60, 128]);
        }
        let mut dst = TensorImage::new(2, 1, BGRA, Some(TensorMemory::Mem)).unwrap();
        CPUProcessor::convert_format(&src, &mut dst).unwrap();
        let map = dst.tensor().map().unwrap();
        let buf = map.as_slice();
        assert_eq!(&buf[0..4], &[10, 20, 30, 255]);
        assert_eq!(&buf[4..8], &[40, 50, 60, 128]);
    }

    /// Helper: compare BGRA output against RGBA output by verifying R↔B swap.
    /// Since CPU BGRA conversion is RGBA conversion + R↔B swizzle, the results
    /// must be byte-exact after accounting for the channel swap.
    fn assert_bgra_matches_rgba(bgra: &TensorImage, rgba: &TensorImage) {
        assert_eq!(bgra.fourcc(), BGRA);
        assert_eq!(rgba.fourcc(), RGBA);
        assert_eq!(bgra.width(), rgba.width());
        assert_eq!(bgra.height(), rgba.height());

        let bgra_map = bgra.tensor().map().unwrap();
        let rgba_map = rgba.tensor().map().unwrap();
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
            NV12,
            None,
            include_bytes!(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/../../testdata/camera720p.nv12"
            )),
        )
        .unwrap();

        // Convert to both RGBA and BGRA, then compare
        let mut rgba_dst = TensorImage::new(1280, 720, RGBA, None).unwrap();
        CPUProcessor::convert_format(&src, &mut rgba_dst).unwrap();

        let mut bgra_dst = TensorImage::new(1280, 720, BGRA, None).unwrap();
        CPUProcessor::convert_format(&src, &mut bgra_dst).unwrap();

        assert_bgra_matches_rgba(&bgra_dst, &rgba_dst);
    }

    #[test]
    fn test_convert_yuyv_to_bgra() {
        let src = load_bytes_to_tensor(
            1280,
            720,
            YUYV,
            None,
            include_bytes!(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/../../testdata/camera720p.yuyv"
            )),
        )
        .unwrap();

        let mut rgba_dst = TensorImage::new(1280, 720, RGBA, None).unwrap();
        CPUProcessor::convert_format(&src, &mut rgba_dst).unwrap();

        let mut bgra_dst = TensorImage::new(1280, 720, BGRA, None).unwrap();
        CPUProcessor::convert_format(&src, &mut bgra_dst).unwrap();

        assert_bgra_matches_rgba(&bgra_dst, &rgba_dst);
    }

    #[test]
    fn test_convert_vyuy_to_bgra() {
        let src = load_bytes_to_tensor(
            1280,
            720,
            VYUY,
            None,
            include_bytes!(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/../../testdata/camera720p.vyuy"
            )),
        )
        .unwrap();

        let mut rgba_dst = TensorImage::new(1280, 720, RGBA, None).unwrap();
        CPUProcessor::convert_format(&src, &mut rgba_dst).unwrap();

        let mut bgra_dst = TensorImage::new(1280, 720, BGRA, None).unwrap();
        CPUProcessor::convert_format(&src, &mut bgra_dst).unwrap();

        assert_bgra_matches_rgba(&bgra_dst, &rgba_dst);
    }

    #[test]
    fn test_convert_nv16_to_bgra() {
        let src = load_bytes_to_tensor(
            1280,
            720,
            NV16,
            None,
            include_bytes!(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/../../testdata/camera720p.nv16"
            )),
        )
        .unwrap();

        let mut rgba_dst = TensorImage::new(1280, 720, RGBA, None).unwrap();
        CPUProcessor::convert_format(&src, &mut rgba_dst).unwrap();

        let mut bgra_dst = TensorImage::new(1280, 720, BGRA, None).unwrap();
        CPUProcessor::convert_format(&src, &mut bgra_dst).unwrap();

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
        crate::ProtoData {
            mask_coefficients: coefficients,
            protos: edgefirst_decoder::ProtoTensor::Float(ndarray::Array3::<f32>::zeros((
                proto_h, proto_w, num_protos,
            ))),
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
        let result = cpu.materialize_segmentations(&[], &proto_data);
        assert!(result.is_ok());
        assert!(result.unwrap().is_empty());
    }

    #[test]
    fn test_materialize_empty_proto_data() {
        let cpu = CPUProcessor::new();
        let proto_data = make_proto_data(8, 8, 4, vec![]);
        let det = [make_detect_box(0.1, 0.1, 0.5, 0.5)];
        let result = cpu.materialize_segmentations(&det, &proto_data);
        assert!(result.is_ok());
        assert!(result.unwrap().is_empty());
    }

    #[test]
    fn test_materialize_single_detection() {
        let cpu = CPUProcessor::new();
        let proto_data = make_proto_data(8, 8, 4, vec![vec![0.5; 4]]);
        let det = [make_detect_box(0.1, 0.1, 0.5, 0.5)];
        let result = cpu.materialize_segmentations(&det, &proto_data);
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
        let result = cpu.materialize_segmentations(&det, &proto_data);
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
        let result = cpu.materialize_segmentations(&det, &proto_data);
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
        let result = cpu.materialize_segmentations(&det, &proto_data);
        assert!(
            result.is_err(),
            "mismatched coeff count vs proto channels should error"
        );
        let err = result.unwrap_err();
        assert!(
            matches!(&err, crate::Error::Internal(s) if s.contains("coeff")),
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
        let result = cpu.materialize_segmentations(&det, &proto_data);
        assert!(result.is_ok());
        assert_eq!(result.unwrap().len(), 3);
    }

    #[test]
    fn test_materialize_zero_area_bbox() {
        let cpu = CPUProcessor::new();
        let proto_data = make_proto_data(8, 8, 4, vec![vec![0.5; 4]]);
        // xmin == xmax → zero-width bbox
        let det = [make_detect_box(0.5, 0.1, 0.5, 0.5)];
        let result = cpu.materialize_segmentations(&det, &proto_data);
        assert!(
            result.is_ok(),
            "zero-area bbox should return Ok with degenerate segmentation"
        );
        let segs = result.unwrap();
        assert_eq!(segs.len(), 1);
    }
}
