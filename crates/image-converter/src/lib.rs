//! EdgeFirst HAL - Image Converter
//!
//! The `image-converter` crate is part of the EdgeFirst Hardware Abstraction
//! Layer (HAL) and provides functionality for converting images between
//! different formats and sizes.  The crate is designed to work with hardware
//! acceleration when available, but also provides a CPU-based fallback for
//! environments where hardware acceleration is not present or not suitable.

use four_char_code::{FourCharCode, four_char_code};
use tensor::{Tensor, TensorMemory, TensorTrait as _};
use zune_jpeg::{
    JpegDecoder,
    zune_core::{colorspace::ColorSpace, options::DecoderOptions},
};

pub use cpu::CPUConverter;
pub use error::{Error, Result};
#[cfg(target_os = "linux")]
pub use g2d::G2DConverter;

mod cpu;
mod error;
mod g2d;

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
            let mut tensor_map = img.tensor.map()?;
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
    None,
    Rotate90,
    Rotate180,
    Rotate270,
}

pub struct Rect {
    pub x: usize,
    pub y: usize,
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
    use std::path::Path;

    use super::*;

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
}
