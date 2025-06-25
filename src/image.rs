use crate::{Error, Result, Tensor, TensorMemory, TensorTrait};
use g2d_sys::{RGB3, RGBA, fourcc::FourCC};
use zune_jpeg::JpegDecoder;

pub struct Image {
    tensor: Tensor<u8>,
    format: FourCC,
}

impl Image {
    pub fn new(
        width: usize,
        height: usize,
        format: FourCC,
        memory: Option<TensorMemory>,
    ) -> Result<Self> {
        let shape = vec![height, width, format.channels()];
        let tensor = Tensor::new(&shape, memory, None)?;
        Ok(Self { tensor, format })
    }

    pub fn load(
        image: &[u8],
        format: Option<FourCC>,
        memory: Option<TensorMemory>,
    ) -> Result<Self> {
        let format = format.unwrap_or(RGB3);
        let mut decoder = JpegDecoder::new(image);
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
        let colour = if self.format == RGB3 {
            jpeg_encoder::ColorType::Rgb
        } else if self.format == RGBA {
            jpeg_encoder::ColorType::Rgba
        } else {
            return Err(Error::UnsupportedOperation(
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

    pub fn format(&self) -> FourCC {
        self.format
    }

    pub fn width(&self) -> usize {
        self.tensor.shape()[1]
    }

    pub fn height(&self) -> usize {
        self.tensor.shape()[0]
    }

    pub fn tensor(&self) -> &Tensor<u8> {
        &self.tensor
    }
}

pub enum ImageConverter {
    CPU(CPUConverter),
}

impl ImageConverter {
    pub fn new() -> Result<Self> {
        let cpu_converter = CPUConverter::new()?;
        Ok(Self::CPU(cpu_converter))
    }

    pub fn convert(&mut self, dst: &mut Image, src: &Image) -> Result<()> {
        match self {
            ImageConverter::CPU(converter) => converter.convert(dst, src),
        }
    }
}

struct CPUConverter {
    resizer: fast_image_resize::Resizer,
    options: fast_image_resize::ResizeOptions,
}

impl CPUConverter {
    fn new() -> Result<Self> {
        let resizer = fast_image_resize::Resizer::new();
        let options = fast_image_resize::ResizeOptions::new()
            .resize_alg(fast_image_resize::ResizeAlg::Convolution(
                fast_image_resize::FilterType::Hamming,
            ))
            .use_alpha(false);
        Ok(Self { resizer, options })
    }

    fn convert(&mut self, dst: &mut Image, src: &Image) -> Result<()> {
        let src_type = match src.format().channels() {
            1 => fast_image_resize::PixelType::U8,
            3 => fast_image_resize::PixelType::U8x3,
            4 => fast_image_resize::PixelType::U8x4,
            _ => {
                return Err(Error::UnsupportedOperation(
                    "Unsupported source image format".to_string(),
                ));
            }
        };

        let dst_type = match dst.format().channels() {
            1 => fast_image_resize::PixelType::U8,
            3 => fast_image_resize::PixelType::U8x3,
            4 => fast_image_resize::PixelType::U8x4,
            _ => {
                return Err(Error::UnsupportedOperation(
                    "Unsupported destination image format".to_string(),
                ));
            }
        };

        let mut src_map = src.tensor().map()?;
        let src_view = fast_image_resize::images::Image::from_slice_u8(
            src.width() as u32,
            src.height() as u32,
            &mut src_map,
            src_type,
        )?;

        let mut dst_map = dst.tensor().map()?;
        let mut dst_view = fast_image_resize::images::Image::from_slice_u8(
            dst.width() as u32,
            dst.height() as u32,
            &mut dst_map,
            dst_type,
        )?;

        let resized_image = self
            .resizer
            .resize(&src_view, &mut dst_view, &self.options)?;

        Ok(())
    }
}

struct G2DConverter {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_load_resize_save() {
        let file = std::fs::read("testdata/zidane.jpg").unwrap();
        let img = Image::load(&file, None, None).unwrap();
        assert_eq!(img.width(), 1280);
        assert_eq!(img.height(), 720);

        let mut dst = Image::new(640, 360, RGB3, None).unwrap();
        let mut converter = ImageConverter::new().unwrap();
        converter.convert(&mut dst, &img).unwrap();
        assert_eq!(dst.width(), 640);
        assert_eq!(dst.height(), 360);

        dst.save("zidane_resized.jpg", 80).unwrap();

        let file = std::fs::read("zidane_resized.jpg").unwrap();
        let img = Image::load(&file, None, None).unwrap();
        assert_eq!(img.width(), 640);
        assert_eq!(img.height(), 360);
        assert_eq!(img.format(), RGB3);
    }
}
