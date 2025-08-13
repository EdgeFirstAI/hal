//! EdgeFirst HAL - Image Converter
//!
//! The `image-converter` crate is part of the EdgeFirst Hardware Abstraction
//! Layer (HAL) and provides functionality for converting images between
//! different formats and sizes.  The crate is designed to work with hardware
//! acceleration when available, but also provides a CPU-based fallback for
//! environments where hardware acceleration is not present or not suitable.

use edgefirst_tensor::{Tensor, TensorMemory, TensorTrait as _};
use four_char_code::{FourCharCode, four_char_code};
use zune_jpeg::{
    JpegDecoder,
    zune_core::{colorspace::ColorSpace, options::DecoderOptions},
};

pub use cpu::CPUConverter;
pub use error::{Error, Result};
#[cfg(target_os = "linux")]
pub use g2d::G2DConverter;
#[cfg(target_os = "linux")]
pub use opengl_headless::GLConverter;
mod cpu;
mod error;
mod g2d;
mod opengl_headless;

pub const YUYV: FourCharCode = four_char_code!("YUYV");
pub const RGBA: FourCharCode = four_char_code!("RGBA");
pub const RGB: FourCharCode = four_char_code!("RGB ");

pub struct TensorImage {
    tensor: Tensor<u8>,
    fourcc: FourCharCode,
    is_planar: bool,
}

impl TensorImage {
    pub fn new(
        width: usize,
        height: usize,
        fourcc: FourCharCode,
        memory: Option<TensorMemory>,
    ) -> Result<Self> {
        let channels = fourcc_channels(fourcc)?;
        let shape = vec![height, width, channels];
        let tensor = Tensor::new(&shape, memory, None)?;

        Ok(Self {
            tensor,
            fourcc,
            is_planar: false,
        })
    }

    pub fn from_tensor(tensor: Tensor<u8>, fourcc: FourCharCode, is_planar: bool) -> Result<Self> {
        // Validate tensor shape based on the fourcc and is_planar flag
        let shape = tensor.shape();
        if shape.len() != 3 {
            return Err(Error::InvalidShape(format!(
                "Tensor shape must have 3 dimensions, got {}: {:?}",
                shape.len(),
                shape
            )));
        }

        let channels = if is_planar { shape[0] } else { shape[2] };
        if fourcc_channels(fourcc)? != channels {
            return Err(Error::InvalidShape(format!(
                "Invalid tensor shape {:?} for format {}",
                shape,
                fourcc.to_string()
            )));
        }

        Ok(Self {
            tensor,
            fourcc,
            is_planar,
        })
    }

    pub fn load(
        image: &[u8],
        format: Option<FourCharCode>,
        memory: Option<TensorMemory>,
    ) -> Result<Self> {
        let format = format.unwrap_or(RGB);
        let colour = match format {
            RGB => ColorSpace::RGB,
            RGBA => ColorSpace::RGBA,
            _ => {
                return Err(Error::NotImplemented(
                    "Unsupported image format".to_string(),
                ));
            }
        };

        let options = DecoderOptions::default().jpeg_set_out_colorspace(colour);
        let mut decoder = JpegDecoder::new_with_options(image, options);
        decoder.decode_headers().unwrap();
        let image_info = decoder.info().unwrap();

        let img = Self::new(
            image_info.width as usize,
            image_info.height as usize,
            format,
            memory,
        )?;

        {
            let mut tensor_map: edgefirst_tensor::TensorMap<u8> = img.tensor.map()?;
            decoder.decode_into(&mut tensor_map)?;
        }

        Ok(img)
    }

    pub fn save(&self, path: &str, quality: u8) -> Result<()> {
        if self.is_planar {
            return Err(Error::NotImplemented(
                "Saving planar images is not supported".to_string(),
            ));
        }

        let colour = if self.fourcc == RGB {
            jpeg_encoder::ColorType::Rgb
        } else if self.fourcc == RGBA {
            jpeg_encoder::ColorType::Rgba
        } else {
            return Err(Error::NotImplemented(
                "Unsupported image format for saving".to_string(),
            ));
        };

        let encoder = jpeg_encoder::Encoder::new_file(path, quality)?;
        let tensor_map = self.tensor.map()?;

        encoder.encode(
            &tensor_map,
            self.width() as u16,
            self.height() as u16,
            colour,
        )?;

        Ok(())
    }

    pub fn tensor(&self) -> &Tensor<u8> {
        &self.tensor
    }

    pub fn fourcc(&self) -> FourCharCode {
        self.fourcc
    }

    pub fn is_planar(&self) -> bool {
        self.is_planar
    }

    pub fn width(&self) -> usize {
        match self.is_planar {
            true => self.tensor.shape()[2],
            false => self.tensor.shape()[1],
        }
    }

    pub fn height(&self) -> usize {
        match self.is_planar {
            true => self.tensor.shape()[1],
            false => self.tensor.shape()[0],
        }
    }

    pub fn channels(&self) -> usize {
        match self.is_planar {
            true => self.tensor.shape()[0],
            false => self.tensor.shape()[2],
        }
    }

    pub fn row_stride(&self) -> usize {
        match self.is_planar {
            true => self.width(),
            false => self.width() * self.channels(),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Rotation {
    None = 0,
    Rotate90Clockwise = 1,
    Rotate180 = 2,
    Rotate90CounterClockwise = 3,
}
impl Rotation {
    pub fn from_degrees_clockwise(angle: usize) -> Rotation {
        match angle.rem_euclid(90) {
            0 => Rotation::None,
            90 => Rotation::Rotate90Clockwise,
            180 => Rotation::Rotate180,
            270 => Rotation::Rotate90CounterClockwise,
            _ => panic!("rotation angle is not a multiple of 90"),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Rect {
    pub left: usize,
    pub top: usize,
    pub width: usize,
    pub height: usize,
}

pub trait ImageConverterTrait {
    /// Converts the source image to the destination image format and size.
    ///
    /// # Arguments
    ///
    /// * `dst` - The destination image to be converted to.
    /// * `src` - The source image to convert from.
    /// * `rotation` - The rotation to apply to the destination image (after
    ///   crop if specified).
    /// * `crop` - An optional rectangle specifying the area to crop from the
    ///   source image
    ///
    /// # Returns
    ///
    /// A `Result` indicating success or failure of the conversion.
    fn convert(
        &mut self,
        dst: &mut TensorImage,
        src: &TensorImage,
        rotation: Rotation,
        crop: Option<Rect>,
    ) -> Result<()>;
}

pub enum ImageConverter {
    CPU(Box<CPUConverter>),
    #[cfg(target_os = "linux")]
    G2D(Box<G2DConverter>),
}

impl ImageConverter {
    pub fn new() -> Result<Self> {
        #[cfg(target_os = "linux")]
        match G2DConverter::new() {
            Ok(g2d_converter) => return Ok(Self::G2D(Box::new(g2d_converter))),
            Err(err) => log::debug!("Failed to initialize G2D converter: {err:?}"),
        }

        let cpu_converter = CPUConverter::new()?;
        Ok(Self::CPU(Box::new(cpu_converter)))
    }
}

impl ImageConverterTrait for ImageConverter {
    fn convert(
        &mut self,
        dst: &mut TensorImage,
        src: &TensorImage,
        rotation: Rotation,
        crop: Option<Rect>,
    ) -> Result<()> {
        match self {
            ImageConverter::CPU(converter) => converter.convert(dst, src, rotation, crop),
            #[cfg(target_os = "linux")]
            ImageConverter::G2D(converter) => converter.convert(dst, src, rotation, crop),
        }
    }
}

fn fourcc_channels(fourcc: FourCharCode) -> Result<usize> {
    match fourcc {
        RGBA => Ok(4), // RGBA has 4 channels (R, G, B, A)
        RGB => Ok(3),  // RGB has 3 channels (R, G, B)
        YUYV => Ok(2), // YUYV has 2 channels (Y and UV)
        _ => Err(Error::InvalidShape(format!(
            "Unsupported fourcc: {}",
            fourcc.to_string()
        ))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{CPUConverter, Rotation};
    use edgefirst_tensor::{TensorMapTrait, TensorMemory};
    use image::buffer::ConvertBuffer;
    use std::path::Path;

    #[ctor::ctor]
    fn init() {
        env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();
    }

    #[test]
    fn test_load_resize_save() {
        let path = Path::new("testdata/zidane.jpg");
        let path = match path.exists() {
            true => path,
            false => {
                let path = Path::new("../testdata/zidane.jpg");
                if path.exists() {
                    path
                } else {
                    Path::new("../../testdata/zidane.jpg")
                }
            }
        };
        assert!(path.exists(), "Test image not found at {path:?}");

        let file = std::fs::read(path).unwrap();
        let img = TensorImage::load(&file, Some(RGBA), None).unwrap();
        assert_eq!(img.width(), 1280);
        assert_eq!(img.height(), 720);

        let mut dst = TensorImage::new(640, 360, RGBA, None).unwrap();
        let mut converter = ImageConverter::new().unwrap();
        converter
            .convert(&mut dst, &img, Rotation::None, None)
            .unwrap();
        assert_eq!(dst.width(), 640);
        assert_eq!(dst.height(), 360);

        dst.save("zidane_resized.jpg", 80).unwrap();

        let file = std::fs::read("zidane_resized.jpg").unwrap();
        let img = TensorImage::load(&file, None, None).unwrap();
        assert_eq!(img.width(), 640);
        assert_eq!(img.height(), 360);
        assert_eq!(img.fourcc(), RGB);
    }

    #[test]
    #[cfg(target_os = "linux")]
    fn test_g2d() {
        let dst_width = 640;
        let dst_height = 360;
        let file = include_bytes!("../../../testdata/zidane.jpg").to_vec();
        let src = TensorImage::load(&file, Some(RGBA), Some(TensorMemory::Dma)).unwrap();

        let mut g2d_dst =
            TensorImage::new(dst_width, dst_height, RGBA, Some(TensorMemory::Dma)).unwrap();
        let mut g2d_converter = G2DConverter::new().unwrap();
        g2d_converter
            .convert(&mut g2d_dst, &src, Rotation::None, None)
            .unwrap();

        let mut cpu_dst = TensorImage::new(dst_width, dst_height, RGBA, None).unwrap();
        let mut cpu_converter = CPUConverter::new().unwrap();
        cpu_converter
            .convert(&mut cpu_dst, &src, Rotation::None, None)
            .unwrap();

        let g2d_image = image::RgbaImage::from_vec(
            dst_width as u32,
            dst_height as u32,
            g2d_dst.tensor().map().unwrap().to_vec(),
        )
        .unwrap();
        let cpu_image = image::RgbaImage::from_vec(
            dst_width as u32,
            dst_height as u32,
            cpu_dst.tensor().map().unwrap().to_vec(),
        )
        .unwrap();

        let similarity = image_compare::rgb_similarity_structure(
            &image_compare::Algorithm::RootMeanSquared,
            &g2d_image.convert(),
            &cpu_image.convert(),
        )
        .expect("Image Comparison failed");
        assert!(
            similarity.score > 0.98,
            "G2D and CPU converted image have similarity score too low: {}",
            similarity.score
        );

        drop(g2d_dst);
    }

    #[test]
    #[cfg(target_os = "linux")]
    fn test_opengl() {
        let dst_width = 640;
        let dst_height = 360;
        let file = include_bytes!("../../../testdata/zidane.jpg").to_vec();
        let src = TensorImage::load(&file, Some(RGBA), Some(TensorMemory::Dma)).unwrap();

        let mut cpu_dst = TensorImage::new(dst_width, dst_height, RGBA, None).unwrap();
        let mut cpu_converter = CPUConverter::new().unwrap();
        cpu_converter
            .convert(&mut cpu_dst, &src, Rotation::None, None)
            .unwrap();
        let mut gl_dst =
            TensorImage::new(dst_width, dst_height, RGBA, Some(TensorMemory::Dma)).unwrap();
        let mut gl_converter = GLConverter::new_with_size(dst_width, dst_height, false).unwrap();

        for _ in 0..5 {
            gl_converter
                .convert(&mut gl_dst, &src, Rotation::None, None)
                .unwrap();
            assert!(
                matches!(gl_dst.tensor, edgefirst_tensor::Tensor::DmaOpenGl(_)),
                "GL converted destination is not OpenGL DMA tensor",
            );

            let cpu_image = image::RgbaImage::from_vec(
                dst_width as u32,
                dst_height as u32,
                cpu_dst.tensor().map().unwrap().to_vec(),
            )
            .unwrap();
            let opengl_image = image::RgbaImage::from_vec(
                dst_width as u32,
                dst_height as u32,
                gl_dst.tensor().map().unwrap().to_vec(),
            )
            .unwrap();

            let similarity = image_compare::rgb_similarity_structure(
                &image_compare::Algorithm::RootMeanSquared,
                &opengl_image.convert(),
                &cpu_image.convert(),
            )
            .expect("Image Comparison failed");
            assert!(
                similarity.score > 0.98,
                "OpenGL and CPU converted image have similarity score too low: {}",
                similarity.score
            );
        }

        drop(gl_dst);
    }

    #[test]
    #[cfg(target_os = "linux")]
    fn test_opengl_10_threads() {
        let handles: Vec<_> = (0..10)
            .map(|i| {
                std::thread::Builder::new()
                    .name(format!("Thread {i}"))
                    .spawn(test_opengl)
                    .unwrap()
            })
            .collect();
        handles.into_iter().for_each(|h| {
            if let Err(e) = h.join() {
                std::panic::resume_unwind(e)
            }
        });
    }

    #[test]
    #[cfg(target_os = "linux")]
    fn test_g2d_crop() {
        let dst_width = 640;
        let dst_height = 360;
        let file = include_bytes!("../../../testdata/zidane.jpg").to_vec();
        let src = TensorImage::load(&file, Some(RGBA), None).unwrap();

        let mut cpu_dst = TensorImage::new(dst_width, dst_height, RGBA, None).unwrap();
        let mut cpu_converter = CPUConverter::new().unwrap();
        cpu_converter
            .convert(
                &mut cpu_dst,
                &src,
                Rotation::None,
                Some(Rect {
                    left: 320,
                    top: 180,
                    width: 1280 - 320,
                    height: 720 - 180,
                }),
            )
            .unwrap();

        let mut g2d_dst = TensorImage::new(dst_width, dst_height, RGBA, None).unwrap();
        let mut g2d_converter = G2DConverter::new().unwrap();
        g2d_converter
            .convert(
                &mut g2d_dst,
                &src,
                Rotation::None,
                Some(Rect {
                    left: 320,
                    top: 180,
                    width: 1280 - 320,
                    height: 720 - 180,
                }),
            )
            .unwrap();

        let g2d_image = image::RgbaImage::from_vec(
            dst_width as u32,
            dst_height as u32,
            g2d_dst.tensor().map().unwrap().to_vec(),
        )
        .unwrap();
        let cpu_image = image::RgbaImage::from_vec(
            dst_width as u32,
            dst_height as u32,
            cpu_dst.tensor().map().unwrap().to_vec(),
        )
        .unwrap();

        let similarity = image_compare::rgb_similarity_structure(
            &image_compare::Algorithm::RootMeanSquared,
            &g2d_image.convert(),
            &cpu_image.convert(),
        )
        .expect("Image Comparison failed");
        assert!(
            similarity.score > 0.98,
            "G2D and CPU converted image have similarity score too low: {}",
            similarity.score
        );

        drop(g2d_dst);
    }

    #[test]
    #[cfg(target_os = "linux")]
    fn test_opengl_crop() {
        let dst_width = 640;
        let dst_height = 360;
        let file = include_bytes!("../../../testdata/zidane.jpg").to_vec();
        let src = TensorImage::load(&file, Some(RGBA), Some(TensorMemory::Dma)).unwrap();

        let mut cpu_dst = TensorImage::new(dst_width, dst_height, RGBA, None).unwrap();
        let mut cpu_converter = CPUConverter::new().unwrap();
        cpu_converter
            .convert(
                &mut cpu_dst,
                &src,
                Rotation::None,
                Some(Rect {
                    left: 320,
                    top: 180,
                    width: 1280 - 320,
                    height: 720 - 180,
                }),
            )
            .unwrap();

        let mut gl_dst =
            TensorImage::new(dst_width, dst_height, RGBA, Some(TensorMemory::Dma)).unwrap();
        let mut gl_converter = GLConverter::new_with_size(dst_width, dst_height, false).unwrap();

        for _ in 0..5 {
            gl_converter
                .convert(
                    &mut gl_dst,
                    &src,
                    Rotation::None,
                    Some(Rect {
                        left: 320,
                        top: 180,
                        width: 1280 - 320,
                        height: 720 - 180,
                    }),
                )
                .unwrap();

            let opengl_image = image::RgbaImage::from_vec(
                dst_width as u32,
                dst_height as u32,
                gl_dst.tensor().map().unwrap().to_vec(),
            )
            .unwrap();
            let cpu_image = image::RgbaImage::from_vec(
                dst_width as u32,
                dst_height as u32,
                cpu_dst.tensor().map().unwrap().to_vec(),
            )
            .unwrap();

            let similarity = image_compare::rgb_similarity_structure(
                &image_compare::Algorithm::RootMeanSquared,
                &opengl_image.convert(),
                &cpu_image.convert(),
            )
            .expect("Image Comparison failed");
            assert!(
                similarity.score > 0.98,
                "OpenGL and CPU converted image have similarity score too low: {}",
                similarity.score
            );
        }

        drop(gl_dst);
    }

    #[test]
    #[cfg(target_os = "linux")]
    fn test_cpu_rotate() {
        for rot in [
            Rotation::Rotate90Clockwise,
            Rotation::Rotate180,
            Rotation::Rotate90CounterClockwise,
        ] {
            test_cpu_rotate_(rot);
        }
    }

    #[cfg(target_os = "linux")]
    fn test_cpu_rotate_(rot: Rotation) {
        // This test rotates the image 4 times and checks that the image was returned to
        // be the same Currently doesn't check if rotations actually rotated in
        // right direction
        let file = include_bytes!("../../../testdata/zidane.jpg").to_vec();

        let unchanged_src = TensorImage::load(&file, Some(RGBA), None).unwrap();
        let mut src = TensorImage::load(&file, Some(RGBA), None).unwrap();

        let (dst_width, dst_height) = match rot {
            Rotation::None | Rotation::Rotate180 => (src.width(), src.height()),
            Rotation::Rotate90Clockwise | Rotation::Rotate90CounterClockwise => {
                (src.height(), src.width())
            }
        };

        let mut cpu_dst = TensorImage::new(dst_width, dst_height, RGBA, None).unwrap();
        let mut cpu_converter = CPUConverter::new().unwrap();

        // After rotating 4 times, the image should be the same as the original

        cpu_converter
            .convert(&mut cpu_dst, &src, rot, None)
            .unwrap();

        cpu_converter
            .convert(&mut src, &cpu_dst, rot, None)
            .unwrap();

        cpu_converter
            .convert(&mut cpu_dst, &src, rot, None)
            .unwrap();

        cpu_converter
            .convert(&mut src, &cpu_dst, rot, None)
            .unwrap();

        let cpu_image = image::RgbaImage::from_vec(
            src.width() as u32,
            src.height() as u32,
            src.tensor().map().unwrap().to_vec(),
        )
        .unwrap();

        let src_image = image::RgbaImage::from_vec(
            unchanged_src.width() as u32,
            unchanged_src.height() as u32,
            unchanged_src.tensor().map().unwrap().to_vec(),
        )
        .unwrap();

        let similarity = image_compare::rgb_similarity_structure(
            &image_compare::Algorithm::RootMeanSquared,
            &src_image.convert(),
            &cpu_image.convert(),
        )
        .expect("Image Comparison failed");

        if similarity.score <= 0.999999 {
            let _ = src.save("cpu.jpg", 80);
            similarity
                .image
                .to_color_map()
                .save("cpu_similarity.png")
                .unwrap();
        }
        assert!(
            similarity.score > 0.999999,
            "OpenGL and CPU {:?} converted image have similarity score too low: {}",
            rot,
            similarity.score
        );
    }

    #[test]
    #[cfg(target_os = "linux")]
    fn test_opengl_rotate() {
        let size = (1280, 720);
        for rot in [
            Rotation::Rotate90Clockwise,
            Rotation::Rotate180,
            Rotation::Rotate90CounterClockwise,
        ] {
            for mem in [
                None,
                Some(TensorMemory::Dma),
                Some(TensorMemory::Shm),
                Some(TensorMemory::Mem),
            ] {
                test_opengl_rotate_(size, rot, mem);
            }
        }
    }

    #[cfg(target_os = "linux")]
    fn test_opengl_rotate_(
        size: (usize, usize),
        rot: Rotation,
        tensor_memory: Option<TensorMemory>,
    ) {
        let (dst_width, dst_height) = match rot {
            Rotation::None | Rotation::Rotate180 => size,
            Rotation::Rotate90Clockwise | Rotation::Rotate90CounterClockwise => (size.1, size.0),
        };

        let file = include_bytes!("../../../testdata/zidane.jpg").to_vec();
        let src = TensorImage::load(&file, Some(RGBA), tensor_memory).unwrap();

        let mut cpu_dst = TensorImage::new(dst_width, dst_height, RGBA, None).unwrap();
        let mut cpu_converter = CPUConverter::new().unwrap();

        cpu_converter
            .convert(&mut cpu_dst, &src, rot, None)
            .unwrap();

        let mut gl_dst = TensorImage::new(dst_width, dst_height, RGBA, tensor_memory).unwrap();
        let mut gl_converter = GLConverter::new_with_size(dst_width, dst_height, false).unwrap();

        for _ in 0..5 {
            gl_converter.convert(&mut gl_dst, &src, rot, None).unwrap();

            let opengl_image = image::RgbaImage::from_vec(
                dst_width as u32,
                dst_height as u32,
                gl_dst.tensor().map().unwrap().to_vec(),
            )
            .unwrap();
            let cpu_image = image::RgbaImage::from_vec(
                dst_width as u32,
                dst_height as u32,
                cpu_dst.tensor().map().unwrap().to_vec(),
            )
            .unwrap();

            let similarity = image_compare::rgb_similarity_structure(
                &image_compare::Algorithm::RootMeanSquared,
                &opengl_image.convert(),
                &cpu_image.convert(),
            )
            .expect("Image Comparison failed");
            if similarity.score <= 1.0 {
                let _ = gl_dst.save("opengl.jpg", 80);
                let _ = cpu_dst.save("cpu.jpg", 80);
                similarity
                    .image
                    .to_color_map()
                    .save("gl_cpu_similarity.png")
                    .unwrap();
            }
            assert!(
                similarity.score > 0.99,
                "OpenGL and CPU {:?} converted image have similarity score too low: {}",
                rot,
                similarity.score
            );
        }

        drop(gl_dst);
    }

    #[test]
    #[cfg(target_os = "linux")]
    fn test_g2d_rotate() {
        let size = (1280, 720);
        for rot in [
            Rotation::Rotate90Clockwise,
            Rotation::Rotate180,
            Rotation::Rotate90CounterClockwise,
        ] {
            test_g2d_rotate_(size, rot);
        }
    }

    #[cfg(target_os = "linux")]
    fn test_g2d_rotate_(size: (usize, usize), rot: Rotation) {
        let (dst_width, dst_height) = match rot {
            Rotation::None | Rotation::Rotate180 => size,
            Rotation::Rotate90Clockwise | Rotation::Rotate90CounterClockwise => (size.1, size.0),
        };

        let file = include_bytes!("../../../testdata/zidane.jpg").to_vec();
        let src = TensorImage::load(&file, Some(RGBA), Some(TensorMemory::Dma)).unwrap();

        let mut cpu_dst = TensorImage::new(dst_width, dst_height, RGBA, None).unwrap();
        let mut cpu_converter = CPUConverter::new().unwrap();

        cpu_converter
            .convert(&mut cpu_dst, &src, rot, None)
            .unwrap();

        let mut g2d_dst =
            TensorImage::new(dst_width, dst_height, RGBA, Some(TensorMemory::Dma)).unwrap();
        let mut g2d_converter = G2DConverter::new().unwrap();

        for _ in 0..5 {
            g2d_converter
                .convert(&mut g2d_dst, &src, rot, None)
                .unwrap();

            let g2d_image = image::RgbaImage::from_vec(
                dst_width as u32,
                dst_height as u32,
                g2d_dst.tensor().map().unwrap().to_vec(),
            )
            .unwrap();
            let cpu_image = image::RgbaImage::from_vec(
                dst_width as u32,
                dst_height as u32,
                cpu_dst.tensor().map().unwrap().to_vec(),
            )
            .unwrap();

            let similarity = image_compare::rgb_similarity_structure(
                &image_compare::Algorithm::RootMeanSquared,
                &g2d_image.convert(),
                &cpu_image.convert(),
            )
            .expect("Image Comparison failed");
            if similarity.score <= 0.99 {
                let _ = g2d_dst.save("g2d.jpg", 80);
                let _ = cpu_dst.save("cpu.jpg", 80);
                similarity
                    .image
                    .to_color_map()
                    .save("g2d_cpu_similarity.png")
                    .unwrap();
            }
            assert!(
                similarity.score > 0.99,
                "G2D and CPU {:?} converted image have similarity score too low: {}",
                rot,
                similarity.score
            );
        }
    }

    #[test]
    fn test_yuyv_to_rgba_cpu() {
        let file = include_bytes!("../../../testdata/camera720p.yuyv").to_vec();
        let src = TensorImage::new(1280, 720, YUYV, None).unwrap();
        src.tensor()
            .map()
            .unwrap()
            .as_mut_slice()
            .copy_from_slice(&file);

        let mut dst = TensorImage::new(1280, 720, RGBA, None).unwrap();
        let mut cpu_converter = CPUConverter::new().unwrap();

        cpu_converter
            .convert(&mut dst, &src, Rotation::None, None)
            .unwrap();

        let cpu_image =
            image::RgbaImage::from_vec(1280, 720, dst.tensor().map().unwrap().to_vec()).unwrap();

        let target_image = image::RgbaImage::from_vec(
            1280,
            720,
            include_bytes!("../../../testdata/camera720p.rgba").to_vec(),
        )
        .unwrap();

        let similarity = image_compare::rgb_similarity_structure(
            &image_compare::Algorithm::RootMeanSquared,
            &cpu_image.convert(),
            &target_image.convert(),
        )
        .expect("Image Comparison failed");
        if similarity.score <= 0.99 {
            let _ = dst.save("cpu.jpg", 80);
            similarity
                .image
                .to_color_map()
                .save("yuyv_to_rgba_cpu_similarity.png")
                .unwrap();
        }
        assert!(
            similarity.score > 0.99,
            "CPU converted image and target image have similarity score too low: {}",
            similarity.score
        );
    }

    #[test]
    fn test_yuyv_to_rgb_cpu() {
        let file = include_bytes!("../../../testdata/camera720p.yuyv").to_vec();
        let src = TensorImage::new(1280, 720, YUYV, None).unwrap();
        src.tensor()
            .map()
            .unwrap()
            .as_mut_slice()
            .copy_from_slice(&file);

        let mut dst = TensorImage::new(1280, 720, RGB, None).unwrap();
        let mut cpu_converter = CPUConverter::new().unwrap();

        cpu_converter
            .convert(&mut dst, &src, Rotation::None, None)
            .unwrap();

        let cpu_image =
            image::RgbImage::from_vec(1280, 720, dst.tensor().map().unwrap().to_vec()).unwrap();

        let target_image = image::RgbaImage::from_vec(
            1280,
            720,
            include_bytes!("../../../testdata/camera720p.rgba").to_vec(),
        )
        .unwrap();

        let similarity = image_compare::rgb_similarity_structure(
            &image_compare::Algorithm::RootMeanSquared,
            &cpu_image,
            &target_image.convert(),
        )
        .expect("Image Comparison failed");
        if similarity.score <= 0.99 {
            let _ = dst.save("cpu.jpg", 80);
            similarity
                .image
                .to_color_map()
                .save("yuyv_to_rgb_cpu_similarity.png")
                .unwrap();
        }
        assert!(
            similarity.score > 0.99,
            "CPU converted image and target image have similarity score too low: {}",
            similarity.score
        );
    }

    #[test]
    #[cfg(target_os = "linux")]
    fn test_yuyv_to_rgba_g2d() {
        let src = load_bytes_to_tensor(
            1280,
            720,
            YUYV,
            None,
            include_bytes!("../../../testdata/camera720p.yuyv"),
        )
        .unwrap();

        let mut dst = TensorImage::new(1280, 720, RGBA, Some(TensorMemory::Dma)).unwrap();
        let mut g2d_converter = G2DConverter::new().unwrap();

        g2d_converter
            .convert(&mut dst, &src, Rotation::None, None)
            .unwrap();

        let g2d_image =
            image::RgbaImage::from_vec(1280, 720, dst.tensor().map().unwrap().to_vec()).unwrap();

        let target_image = image::RgbaImage::from_vec(
            1280,
            720,
            include_bytes!("../../../testdata/camera720p.rgba").to_vec(),
        )
        .unwrap();

        let similarity = image_compare::rgb_similarity_structure(
            &image_compare::Algorithm::RootMeanSquared,
            &g2d_image.convert(),
            &target_image.convert(),
        )
        .expect("Image Comparison failed");
        if similarity.score <= 0.99 {
            let _ = dst.save("g2d.jpg", 80);
            similarity
                .image
                .to_color_map()
                .save("yuyv_to_rgba_g2d_similarity.png")
                .unwrap();
        }
        assert!(
            similarity.score > 0.99,
            "G2D converted image and target image have similarity score too low: {}",
            similarity.score
        );
    }

    #[test]
    #[cfg(target_os = "linux")]
    fn test_yuyv_to_rgba_opengl() {
        let src = load_bytes_to_tensor(
            1280,
            720,
            YUYV,
            Some(TensorMemory::Dma),
            include_bytes!("../../../testdata/camera720p.yuyv"),
        )
        .unwrap();

        let mut dst = TensorImage::new(1280, 720, RGBA, Some(TensorMemory::Mem)).unwrap();
        let mut gl_converter = GLConverter::new().unwrap();

        gl_converter
            .convert(&mut dst, &src, Rotation::None, None)
            .unwrap();

        let gl_image =
            image::RgbaImage::from_vec(1280, 720, dst.tensor().map().unwrap().to_vec()).unwrap();

        let target_image = image::RgbaImage::from_vec(
            1280,
            720,
            include_bytes!("../../../testdata/camera720p.rgba").to_vec(),
        )
        .unwrap();

        let similarity = image_compare::rgb_similarity_structure(
            &image_compare::Algorithm::RootMeanSquared,
            &gl_image.convert(),
            &target_image.convert(),
        )
        .expect("Image Comparison failed");
        if similarity.score <= 0.99 {
            let _ = dst.save("opengl.jpg", 80);
            similarity
                .image
                .to_color_map()
                .save("yuyv_to_rgba_opengl_similarity.png")
                .unwrap();
        }
        assert!(
            similarity.score > 0.99,
            "OpenGL converted image and target image have similarity score too low: {}",
            similarity.score
        );
    }

    #[test]
    fn test_yuyv_to_rgba_resize_cpu() {
        let src = load_bytes_to_tensor(
            1280,
            720,
            YUYV,
            None,
            include_bytes!("../../../testdata/camera720p.yuyv"),
        )
        .unwrap();

        let (dst_width, dst_height) = (960, 540);

        let mut dst = TensorImage::new(dst_width, dst_height, RGBA, None).unwrap();
        let mut cpu_converter = CPUConverter::new().unwrap();

        cpu_converter
            .convert(&mut dst, &src, Rotation::None, None)
            .unwrap();

        let cpu_image = image::RgbaImage::from_vec(
            dst_width as u32,
            dst_height as u32,
            dst.tensor().map().unwrap().to_vec(),
        )
        .unwrap();
        let mut dst_target = TensorImage::new(dst_width, dst_height, RGBA, None).unwrap();
        let src_target = load_bytes_to_tensor(
            1280,
            720,
            RGBA,
            None,
            include_bytes!("../../../testdata/camera720p.rgba"),
        )
        .unwrap();
        cpu_converter
            .convert(&mut dst_target, &src_target, Rotation::None, None)
            .unwrap();
        let target_image = image::RgbaImage::from_vec(
            dst_width as u32,
            dst_height as u32,
            dst_target.tensor().map().unwrap().to_vec(),
        )
        .unwrap();

        let similarity = image_compare::rgb_similarity_structure(
            &image_compare::Algorithm::RootMeanSquared,
            &cpu_image.convert(),
            &target_image.convert(),
        )
        .expect("Image Comparison failed");
        if similarity.score <= 0.99 {
            let _ = dst.save("cpu.jpg", 80);
            similarity
                .image
                .to_color_map()
                .save("yuyv_to_rgba_resize_cpu_similarity.png")
                .unwrap();
        }
        assert!(
            similarity.score > 0.99,
            "CPU converted image and target image have similarity score too low: {}",
            similarity.score
        );
    }

    fn load_bytes_to_tensor(
        width: usize,
        height: usize,
        fourcc: FourCharCode,
        memory: Option<TensorMemory>,
        bytes: &[u8],
    ) -> Result<TensorImage, Error> {
        let src = TensorImage::new(width, height, fourcc, memory)?;
        src.tensor().map()?.as_mut_slice().copy_from_slice(bytes);
        Ok(src)
    }
}
