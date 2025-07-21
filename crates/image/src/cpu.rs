use crate::{Error, ImageConverterTrait, Rect, Result, Rotation, TensorImage};
use edgefirst_tensor::TensorTrait;

/// CPUConverter implements the ImageConverter trait using the fallback CPU
/// implementation for image processing.
pub struct CPUConverter {
    resizer: fast_image_resize::Resizer,
    options: fast_image_resize::ResizeOptions,
}

impl CPUConverter {
    pub fn new() -> Result<Self> {
        let resizer = fast_image_resize::Resizer::new();
        let options = fast_image_resize::ResizeOptions::new()
            .resize_alg(fast_image_resize::ResizeAlg::Convolution(
                fast_image_resize::FilterType::Hamming,
            ))
            .use_alpha(false);
        Ok(Self { resizer, options })
    }

    pub fn new_nearest() -> Result<Self> {
        let resizer = fast_image_resize::Resizer::new();
        let options = fast_image_resize::ResizeOptions::new()
            .resize_alg(fast_image_resize::ResizeAlg::Nearest)
            .use_alpha(false);
        Ok(Self { resizer, options })
    }
}

impl ImageConverterTrait for CPUConverter {
    fn convert(
        &mut self,
        dst: &mut TensorImage,
        src: &TensorImage,
        rotation: Rotation,
        crop: Option<Rect>,
    ) -> Result<()> {
        if rotation != Rotation::None {
            return Err(Error::NotImplemented(
                "Rotation is not supported in CPUConverter".to_string(),
            ));
        }

        let src_type = match src.channels() {
            1 => fast_image_resize::PixelType::U8,
            3 => fast_image_resize::PixelType::U8x3,
            4 => fast_image_resize::PixelType::U8x4,
            _ => {
                return Err(Error::NotImplemented(
                    "Unsupported source image format".to_string(),
                ));
            }
        };

        let dst_type = match dst.channels() {
            1 => fast_image_resize::PixelType::U8,
            3 => fast_image_resize::PixelType::U8x3,
            4 => fast_image_resize::PixelType::U8x4,
            _ => {
                return Err(Error::NotImplemented(
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
        let options = if let Some(crop) = crop {
            self.options.crop(
                crop.left as f64,
                crop.top as f64,
                crop.width as f64,
                crop.height as f64,
            )
        } else {
            self.options
        };

        self.resizer.resize(&src_view, &mut dst_view, &options)?;

        Ok(())
    }
}
