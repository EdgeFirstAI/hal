//! EdgeFirst HAL - Image Converter
//!
//! The `image-converter` crate is part of the EdgeFirst Hardware Abstraction
//! Layer (HAL) and provides functionality for converting images between
//! different formats and sizes.  The crate is designed to work with hardware
//! acceleration when available, but also provides a CPU-based fallback for
//! environments where hardware acceleration is not present or not suitable.

use edgefirst_tensor::{Tensor, TensorMapTrait, TensorMemory, TensorTrait as _};
use enum_dispatch::enum_dispatch;
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
#[cfg(feature = "opengl")]
pub use opengl_headless::GLConverter;
use zune_png::PngDecoder;
mod cpu;
mod error;
mod g2d;
mod opengl_headless;

/// 8 bit interleaved YUV422, limited range
pub const YUYV: FourCharCode = four_char_code!("YUYV");
/// 8 bit planar YUV420, limited range
pub const NV12: FourCharCode = four_char_code!("NV12");
/// 8 bit RGBA
pub const RGBA: FourCharCode = four_char_code!("RGBA");
/// 8 bit RGB
pub const RGB: FourCharCode = four_char_code!("RGB ");
/// 8 bit grayscale, full range
pub const GREY: FourCharCode = four_char_code!("Y800");

// TODO: planar RGB is 4BPS? https://fourcc.org/8bps/

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
        if let Ok(i) = Self::load_jpeg(image, format, memory) {
            return Ok(i);
        }
        if let Ok(i) = Self::load_png(image, format, memory) {
            return Ok(i);
        }

        Err(Error::NotSupported(
            "Could not decode as jpeg or png".to_string(),
        ))
    }

    pub fn load_jpeg(
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
        decoder.decode_headers()?;

        let image_info = decoder.info().unwrap();

        let (rotation, flip) = decoder
            .exif()
            .map(|x| Self::read_exif_orientation(x))
            .unwrap_or((Rotation::None, Flip::None));

        if (rotation, flip) == (Rotation::None, Flip::None) {
            let img = Self::new(
                image_info.width as usize,
                image_info.height as usize,
                format,
                memory,
            )?;

            let mut tensor_map: edgefirst_tensor::TensorMap<u8> = img.tensor.map()?;
            decoder.decode_into(&mut tensor_map)?;
            return Ok(img);
        }

        let tmp = Self::new(
            image_info.width as usize,
            image_info.height as usize,
            format,
            Some(TensorMemory::Mem),
        )?;
        decoder.decode_into(tmp.tensor.map()?.as_mut_slice())?;

        rotate_flip_to_tensor_image(&tmp, rotation, flip, memory)
    }

    pub fn load_png(
        image: &[u8],
        format: Option<FourCharCode>,
        memory: Option<TensorMemory>,
    ) -> Result<Self> {
        let format = format.unwrap_or(RGB);
        let alpha = match format {
            RGB => false,
            RGBA => true,
            _ => {
                return Err(Error::NotImplemented(
                    "Unsupported image format".to_string(),
                ));
            }
        };

        let options = DecoderOptions::default()
            .png_set_add_alpha_channel(alpha)
            .png_set_decode_animated(false);
        let mut decoder = PngDecoder::new_with_options(image, options);
        decoder.decode_headers()?;
        let image_info = decoder.get_info().unwrap();

        let (rotation, flip) = image_info
            .exif
            .as_ref()
            .map(|x| Self::read_exif_orientation(x))
            .unwrap_or((Rotation::None, Flip::None));

        if (rotation, flip) == (Rotation::None, Flip::None) {
            let img = Self::new(image_info.width, image_info.height, format, memory)?;
            decoder.decode_into(&mut img.tensor.map()?)?;
            return Ok(img);
        }

        let tmp = Self::new(
            image_info.width,
            image_info.height,
            format,
            Some(TensorMemory::Mem),
        )?;
        decoder.decode_into(&mut tmp.tensor.map()?)?;

        rotate_flip_to_tensor_image(&tmp, rotation, flip, memory)
    }

    fn read_exif_orientation(exif_: &[u8]) -> (Rotation, Flip) {
        let exifreader = exif::Reader::new();
        let Ok(exif_) = exifreader.read_raw(exif_.to_vec()) else {
            return (Rotation::None, Flip::None);
        };
        let Some(orientation) = exif_.get_field(exif::Tag::Orientation, exif::In::PRIMARY) else {
            return (Rotation::None, Flip::None);
        };
        match orientation.value.get_uint(0) {
            Some(1) => (Rotation::None, Flip::None),
            Some(2) => (Rotation::None, Flip::Horizontal),
            Some(3) => (Rotation::Rotate180, Flip::None),
            Some(4) => (Rotation::Rotate180, Flip::Horizontal),
            Some(5) => (Rotation::Clockwise90, Flip::Horizontal),
            Some(6) => (Rotation::Clockwise90, Flip::None),
            Some(7) => (Rotation::CounterClockwise90, Flip::Horizontal),
            Some(8) => (Rotation::CounterClockwise90, Flip::None),
            Some(v) => {
                log::warn!("broken orientation EXIF value: {v}");
                (Rotation::None, Flip::None)
            }
            None => (Rotation::None, Flip::None),
        }
    }

    pub fn save_jpeg(&self, path: &str, quality: u8) -> Result<()> {
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

/// Flips the image, and the rotates it.
fn rotate_flip_to_tensor_image(
    src: &TensorImage,
    rotation: Rotation,
    flip: Flip,
    memory: Option<TensorMemory>,
) -> Result<TensorImage, Error> {
    let src_map = src.tensor.map()?;
    let dst = match rotation {
        Rotation::None | Rotation::Rotate180 => {
            TensorImage::new(src.width(), src.height(), src.fourcc(), memory)?
        }
        Rotation::Clockwise90 | Rotation::CounterClockwise90 => {
            TensorImage::new(src.height(), src.width(), src.fourcc(), memory)?
        }
    };

    let mut dst_map = dst.tensor.map()?;

    CPUConverter::flip_rotate_ndarray(&src_map, &mut dst_map, &dst, rotation, flip)?;

    Ok(dst)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Rotation {
    None = 0,
    Clockwise90 = 1,
    Rotate180 = 2,
    CounterClockwise90 = 3,
}
impl Rotation {
    pub fn from_degrees_clockwise(angle: usize) -> Rotation {
        match angle.rem_euclid(90) {
            0 => Rotation::None,
            90 => Rotation::Clockwise90,
            180 => Rotation::Rotate180,
            270 => Rotation::CounterClockwise90,
            _ => panic!("rotation angle is not a multiple of 90"),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Flip {
    None = 0,
    Vertical = 1,
    Horizontal = 2,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Rect {
    pub left: usize,
    pub top: usize,
    pub width: usize,
    pub height: usize,
}

#[enum_dispatch(ImageConverter)]
pub trait ImageConverterTrait {
    /// Converts the source image to the destination image format and size. The
    /// image is cropped first, then flipped, then rotated
    ///
    /// # Arguments
    ///
    /// * `dst` - The destination image to be converted to.
    /// * `src` - The source image to convert from.
    /// * `rotation` - The rotation to apply to the destination image.
    /// * `flip` - Flips the image
    /// * `crop` - An optional rectangle specifying the area to crop from the
    ///   source image
    ///
    /// # Returns
    ///
    /// A `Result` indicating success or failure of the conversion.
    fn convert(
        &mut self,
        src: &TensorImage,
        dst: &mut TensorImage,
        rotation: Rotation,
        flip: Flip,
        crop: Option<Rect>,
    ) -> Result<()>;
}

pub struct ImageConverter {
    pub cpu: CPUConverter,

    #[cfg(target_os = "linux")]
    pub g2d: Option<G2DConverter>,
    #[cfg(target_os = "linux")]
    #[cfg(feature = "opengl")]
    pub opengl: Option<GLConverter>,
}

impl ImageConverter {
    pub fn new() -> Result<Self> {
        #[cfg(target_os = "linux")]
        let g2d = match G2DConverter::new() {
            Ok(g2d_converter) => Some(g2d_converter),
            Err(err) => {
                log::debug!("Failed to initialize G2D converter: {err:?}");
                None
            }
        };

        #[cfg(target_os = "linux")]
        #[cfg(feature = "opengl")]
        let opengl = match GLConverter::new() {
            Ok(gl_converter) => Some(gl_converter),
            Err(err) => {
                log::debug!("Failed to initialize GL converter: {err:?}");
                None
            }
        };

        let cpu = CPUConverter::new()?;
        Ok(Self {
            cpu,
            #[cfg(target_os = "linux")]
            g2d,
            #[cfg(feature = "opengl")]
            opengl,
        })
    }
}

impl ImageConverterTrait for ImageConverter {
    fn convert(
        &mut self,
        src: &TensorImage,
        dst: &mut TensorImage,
        rotation: Rotation,
        flip: Flip,
        crop: Option<Rect>,
    ) -> Result<()> {
        #[cfg(target_os = "linux")]
        if let Some(g2d) = self.g2d.as_mut()
            && g2d.convert(src, dst, rotation, flip, crop).is_ok()
        {
            log::debug!("image converted with g2d");
            return Ok(());
        }

        #[cfg(target_os = "linux")]
        #[cfg(feature = "opengl")]
        if let Some(opengl) = self.opengl.as_mut()
            && opengl.convert(src, dst, rotation, flip, crop).is_ok()
        {
            log::debug!("image converted with opengl");
            return Ok(());
        }

        self.cpu.convert(src, dst, rotation, flip, crop)?;
        log::debug!("image converted with cpu");
        Ok(())
    }
}

fn fourcc_channels(fourcc: FourCharCode) -> Result<usize> {
    match fourcc {
        RGBA => Ok(4), // RGBA has 4 channels (R, G, B, A)
        RGB => Ok(3),  // RGB has 3 channels (R, G, B)
        YUYV => Ok(2), // YUYV has 2 channels (Y and UV)
        GREY => Ok(1), // Y800 has 1 channel (Y)
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
        let img = TensorImage::load_jpeg(&file, Some(RGBA), None).unwrap();
        assert_eq!(img.width(), 1280);
        assert_eq!(img.height(), 720);

        let mut dst = TensorImage::new(640, 360, RGBA, None).unwrap();
        let mut converter = ImageConverter::new().unwrap();
        converter
            .convert(&img, &mut dst, Rotation::None, Flip::None, None)
            .unwrap();
        assert_eq!(dst.width(), 640);
        assert_eq!(dst.height(), 360);

        dst.save_jpeg("zidane_resized.jpg", 80).unwrap();

        let file = std::fs::read("zidane_resized.jpg").unwrap();
        let img = TensorImage::load_jpeg(&file, None, None).unwrap();
        assert_eq!(img.width(), 640);
        assert_eq!(img.height(), 360);
        assert_eq!(img.fourcc(), RGB);
    }

    // #[test]
    // fn test_opengl_grey() {
    //     let path = Path::new("testdata/zidane.jpg");
    //     let path = match path.exists() {
    //         true => path,
    //         false => {
    //             let path = Path::new("../testdata/zidane.jpg");
    //             if path.exists() {
    //                 path
    //             } else {
    //                 Path::new("../../testdata/zidane.jpg")
    //             }
    //         }
    //     };
    //     assert!(path.exists(), "Test image not found at {path:?}");

    //     let file = std::fs::read(path).unwrap();
    //     let img = TensorImage::load_jpeg(&file, Some(RGBA), None).unwrap();
    //     assert_eq!(img.width(), 1280);
    //     assert_eq!(img.height(), 720);

    //     let mut grey = TensorImage::new(1280, 720, GREY,
    // Some(TensorMemory::Dma)).unwrap();     let mut dst =
    // TensorImage::new(640, 640, GREY, Some(TensorMemory::Dma)).unwrap();

    //     let mut converter = CPUConverter::new().unwrap();

    //     converter
    //         .convert(&img, &mut grey, Rotation::None, Flip::None, None)
    //         .unwrap();

    //     let mut gl = GLConverter::new().unwrap();
    //     gl.convert(&grey, &mut dst, Rotation::None, Flip::None, None)
    //         .unwrap()
    // }

    #[test]
    #[cfg(target_os = "linux")]
    fn test_new_image_converter() {
        let dst_width = 640;
        let dst_height = 360;
        let file = include_bytes!("../../../testdata/zidane.jpg").to_vec();
        let src = TensorImage::load_jpeg(&file, Some(RGBA), None).unwrap();

        let mut converter_dst = TensorImage::new(dst_width, dst_height, RGBA, None).unwrap();
        let mut converter = ImageConverter::new().unwrap();
        converter
            .convert(&src, &mut converter_dst, Rotation::None, Flip::None, None)
            .unwrap();

        let mut cpu_dst = TensorImage::new(dst_width, dst_height, RGBA, None).unwrap();
        let mut cpu_converter = CPUConverter::new().unwrap();
        cpu_converter
            .convert(&src, &mut cpu_dst, Rotation::None, Flip::None, None)
            .unwrap();

        compare_images(&converter_dst, &cpu_dst, 0.98, function!());
    }

    #[test]
    fn test_load_with_exif() {
        let file = include_bytes!("../../../testdata/zidane_rotated_exif.jpg").to_vec();
        let loaded = TensorImage::load_jpeg(&file, Some(RGBA), None).unwrap();

        assert_eq!(loaded.height(), 1280);
        assert_eq!(loaded.width(), 720);

        let file = include_bytes!("../../../testdata/zidane.jpg").to_vec();
        let cpu_src = TensorImage::load_jpeg(&file, Some(RGBA), None).unwrap();

        let (dst_width, dst_height) = (cpu_src.height(), cpu_src.width());

        let mut cpu_dst = TensorImage::new(dst_width, dst_height, RGBA, None).unwrap();
        let mut cpu_converter = CPUConverter::new().unwrap();

        cpu_converter
            .convert(
                &cpu_src,
                &mut cpu_dst,
                Rotation::Clockwise90,
                Flip::None,
                None,
            )
            .unwrap();

        compare_images(&loaded, &cpu_dst, 0.98, function!());
    }

    #[test]
    #[cfg(target_os = "linux")]
    fn test_g2d() {
        let dst_width = 640;
        let dst_height = 360;
        let file = include_bytes!("../../../testdata/zidane.jpg").to_vec();
        let src = TensorImage::load_jpeg(&file, Some(RGBA), Some(TensorMemory::Dma)).unwrap();

        let mut g2d_dst =
            TensorImage::new(dst_width, dst_height, RGBA, Some(TensorMemory::Dma)).unwrap();
        let mut g2d_converter = G2DConverter::new().unwrap();
        g2d_converter
            .convert(&src, &mut g2d_dst, Rotation::None, Flip::None, None)
            .unwrap();

        let mut cpu_dst = TensorImage::new(dst_width, dst_height, RGBA, None).unwrap();
        let mut cpu_converter = CPUConverter::new().unwrap();
        cpu_converter
            .convert(&src, &mut cpu_dst, Rotation::None, Flip::None, None)
            .unwrap();

        compare_images(&g2d_dst, &cpu_dst, 0.98, function!());
    }

    #[test]
    #[cfg(target_os = "linux")]
    #[cfg(feature = "opengl")]
    fn test_opengl() {
        let dst_width = 640;
        let dst_height = 360;
        let file = include_bytes!("../../../testdata/zidane.jpg").to_vec();
        let src = TensorImage::load_jpeg(&file, Some(RGBA), Some(TensorMemory::Dma)).unwrap();

        let mut cpu_dst = TensorImage::new(dst_width, dst_height, RGBA, None).unwrap();
        let mut cpu_converter = CPUConverter::new().unwrap();
        cpu_converter
            .convert(&src, &mut cpu_dst, Rotation::None, Flip::None, None)
            .unwrap();
        let mut gl_dst =
            TensorImage::new(dst_width, dst_height, RGBA, Some(TensorMemory::Dma)).unwrap();
        let mut gl_converter = GLConverter::new_with_size(dst_width, dst_height, false).unwrap();

        for _ in 0..5 {
            gl_converter
                .convert(&src, &mut gl_dst, Rotation::None, Flip::None, None)
                .unwrap();

            compare_images(&gl_dst, &cpu_dst, 0.98, function!());
        }

        drop(gl_dst);
    }

    #[test]
    #[cfg(target_os = "linux")]
    #[cfg(feature = "opengl")]
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
        let dst_height = 640;
        let file = include_bytes!("../../../testdata/zidane.jpg").to_vec();
        let src = TensorImage::load_jpeg(&file, Some(RGBA), None).unwrap();

        let mut cpu_dst = TensorImage::new(dst_width, dst_height, RGBA, None).unwrap();
        let mut cpu_converter = CPUConverter::new().unwrap();
        cpu_converter
            .convert(
                &src,
                &mut cpu_dst,
                Rotation::None,
                Flip::None,
                Some(Rect {
                    left: 0,
                    top: 0,
                    width: 640,
                    height: 360,
                }),
            )
            .unwrap();

        let mut g2d_dst = TensorImage::new(dst_width, dst_height, RGBA, None).unwrap();
        let mut g2d_converter = G2DConverter::new().unwrap();
        g2d_converter
            .convert(
                &src,
                &mut g2d_dst,
                Rotation::None,
                Flip::None,
                Some(Rect {
                    left: 0,
                    top: 0,
                    width: 640,
                    height: 360,
                }),
            )
            .unwrap();

        compare_images(&g2d_dst, &cpu_dst, 0.98, function!());
    }

    #[test]
    #[cfg(target_os = "linux")]
    #[cfg(feature = "opengl")]
    fn test_opengl_crop() {
        let dst_width = 640;
        let dst_height = 360;
        let file = include_bytes!("../../../testdata/zidane.jpg").to_vec();
        let src = TensorImage::load_jpeg(&file, Some(RGBA), Some(TensorMemory::Dma)).unwrap();

        let mut cpu_dst = TensorImage::new(dst_width, dst_height, RGBA, None).unwrap();
        let mut cpu_converter = CPUConverter::new().unwrap();
        cpu_converter
            .convert(
                &src,
                &mut cpu_dst,
                Rotation::None,
                Flip::None,
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
                    &src,
                    &mut gl_dst,
                    Rotation::None,
                    Flip::None,
                    Some(Rect {
                        left: 320,
                        top: 180,
                        width: 1280 - 320,
                        height: 720 - 180,
                    }),
                )
                .unwrap();

            compare_images(&gl_dst, &cpu_dst, 0.98, function!());
        }
    }

    #[test]
    #[cfg(target_os = "linux")]
    fn test_cpu_rotate() {
        for rot in [
            Rotation::Clockwise90,
            Rotation::Rotate180,
            Rotation::CounterClockwise90,
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

        let unchanged_src = TensorImage::load_jpeg(&file, Some(RGBA), None).unwrap();
        let mut src = TensorImage::load_jpeg(&file, Some(RGBA), None).unwrap();

        let (dst_width, dst_height) = match rot {
            Rotation::None | Rotation::Rotate180 => (src.width(), src.height()),
            Rotation::Clockwise90 | Rotation::CounterClockwise90 => (src.height(), src.width()),
        };

        let mut cpu_dst = TensorImage::new(dst_width, dst_height, RGBA, None).unwrap();
        let mut cpu_converter = CPUConverter::new().unwrap();

        // After rotating 4 times, the image should be the same as the original

        cpu_converter
            .convert(&src, &mut cpu_dst, rot, Flip::None, None)
            .unwrap();

        cpu_converter
            .convert(&cpu_dst, &mut src, rot, Flip::None, None)
            .unwrap();

        cpu_converter
            .convert(&src, &mut cpu_dst, rot, Flip::None, None)
            .unwrap();

        cpu_converter
            .convert(&cpu_dst, &mut src, rot, Flip::None, None)
            .unwrap();

        compare_images(&src, &unchanged_src, 0.98, function!());
    }

    #[test]
    #[cfg(target_os = "linux")]
    #[cfg(feature = "opengl")]
    fn test_opengl_rotate() {
        let size = (1280, 720);
        for rot in [
            Rotation::Clockwise90,
            Rotation::Rotate180,
            Rotation::CounterClockwise90,
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
    #[cfg(feature = "opengl")]
    fn test_opengl_rotate_(
        size: (usize, usize),
        rot: Rotation,
        tensor_memory: Option<TensorMemory>,
    ) {
        let (dst_width, dst_height) = match rot {
            Rotation::None | Rotation::Rotate180 => size,
            Rotation::Clockwise90 | Rotation::CounterClockwise90 => (size.1, size.0),
        };

        let file = include_bytes!("../../../testdata/zidane.jpg").to_vec();
        let src = TensorImage::load_jpeg(&file, Some(RGBA), tensor_memory).unwrap();

        let mut cpu_dst = TensorImage::new(dst_width, dst_height, RGBA, None).unwrap();
        let mut cpu_converter = CPUConverter::new().unwrap();

        cpu_converter
            .convert(&src, &mut cpu_dst, rot, Flip::None, None)
            .unwrap();

        let mut gl_dst = TensorImage::new(dst_width, dst_height, RGBA, tensor_memory).unwrap();
        let mut gl_converter = GLConverter::new_with_size(dst_width, dst_height, false).unwrap();

        for _ in 0..5 {
            gl_converter
                .convert(&src, &mut gl_dst, rot, Flip::None, None)
                .unwrap();

            compare_images(&gl_dst, &cpu_dst, 0.98, function!());
        }
    }

    #[test]
    #[cfg(target_os = "linux")]
    fn test_g2d_rotate() {
        let size = (1280, 720);
        for rot in [
            Rotation::Clockwise90,
            Rotation::Rotate180,
            Rotation::CounterClockwise90,
        ] {
            test_g2d_rotate_(size, rot);
        }
    }

    #[cfg(target_os = "linux")]
    fn test_g2d_rotate_(size: (usize, usize), rot: Rotation) {
        let (dst_width, dst_height) = match rot {
            Rotation::None | Rotation::Rotate180 => size,
            Rotation::Clockwise90 | Rotation::CounterClockwise90 => (size.1, size.0),
        };

        let file = include_bytes!("../../../testdata/zidane.jpg").to_vec();
        let src = TensorImage::load_jpeg(&file, Some(RGBA), Some(TensorMemory::Dma)).unwrap();

        let mut cpu_dst = TensorImage::new(dst_width, dst_height, RGBA, None).unwrap();
        let mut cpu_converter = CPUConverter::new().unwrap();

        cpu_converter
            .convert(&src, &mut cpu_dst, rot, Flip::None, None)
            .unwrap();

        let mut g2d_dst =
            TensorImage::new(dst_width, dst_height, RGBA, Some(TensorMemory::Dma)).unwrap();
        let mut g2d_converter = G2DConverter::new().unwrap();

        g2d_converter
            .convert(&src, &mut g2d_dst, rot, Flip::None, None)
            .unwrap();

        compare_images(&g2d_dst, &cpu_dst, 0.98, function!());
    }

    #[test]
    fn test_rgba_to_yuyv_resize_cpu() {
        let src = load_bytes_to_tensor(
            1280,
            720,
            RGBA,
            None,
            include_bytes!("../../../testdata/camera720p.rgba"),
        )
        .unwrap();

        let (dst_width, dst_height) = (640, 360);

        let mut dst = TensorImage::new(dst_width, dst_height, YUYV, None).unwrap();

        let mut dst_through_yuyv = TensorImage::new(dst_width, dst_height, RGBA, None).unwrap();
        let mut dst_direct = TensorImage::new(dst_width, dst_height, RGBA, None).unwrap();

        let mut cpu_converter = CPUConverter::new().unwrap();

        cpu_converter
            .convert(&src, &mut dst, Rotation::None, Flip::None, None)
            .unwrap();

        cpu_converter
            .convert(
                &dst,
                &mut dst_through_yuyv,
                Rotation::None,
                Flip::None,
                None,
            )
            .unwrap();

        cpu_converter
            .convert(&src, &mut dst_direct, Rotation::None, Flip::None, None)
            .unwrap();

        compare_images(&dst_through_yuyv, &dst_direct, 0.98, function!());
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
            .convert(&src, &mut dst, Rotation::None, Flip::None, None)
            .unwrap();

        let target_image = TensorImage::new(1280, 720, RGBA, None).unwrap();
        target_image
            .tensor()
            .map()
            .unwrap()
            .as_mut_slice()
            .copy_from_slice(include_bytes!("../../../testdata/camera720p.rgba"));

        compare_images(&dst, &target_image, 0.98, function!());
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
            .convert(&src, &mut dst, Rotation::None, Flip::None, None)
            .unwrap();

        let target_image = TensorImage::new(1280, 720, RGB, None).unwrap();
        target_image
            .tensor()
            .map()
            .unwrap()
            .as_mut_slice()
            .as_chunks_mut::<3>()
            .0
            .iter_mut()
            .zip(
                include_bytes!("../../../testdata/camera720p.rgba")
                    .as_chunks::<4>()
                    .0,
            )
            .for_each(|(dst, src)| *dst = [src[0], src[1], src[2]]);

        compare_images(&dst, &target_image, 0.98, function!());
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
            .convert(&src, &mut dst, Rotation::None, Flip::None, None)
            .unwrap();

        let target_image = TensorImage::new(1280, 720, RGBA, None).unwrap();
        target_image
            .tensor()
            .map()
            .unwrap()
            .as_mut_slice()
            .copy_from_slice(include_bytes!("../../../testdata/camera720p.rgba"));

        compare_images(&dst, &target_image, 0.98, function!());
    }

    #[test]
    #[cfg(target_os = "linux")]
    #[cfg(feature = "opengl")]
    fn test_yuyv_to_rgba_opengl() {
        let src = load_bytes_to_tensor(
            1280,
            720,
            YUYV,
            Some(TensorMemory::Dma),
            include_bytes!("../../../testdata/camera720p.yuyv"),
        )
        .unwrap();

        let mut dst = TensorImage::new(1280, 720, RGBA, Some(TensorMemory::Dma)).unwrap();
        let mut gl_converter = GLConverter::new().unwrap();

        gl_converter
            .convert(&src, &mut dst, Rotation::None, Flip::None, None)
            .unwrap();

        let target_image = TensorImage::new(1280, 720, RGBA, None).unwrap();
        target_image
            .tensor()
            .map()
            .unwrap()
            .as_mut_slice()
            .copy_from_slice(include_bytes!("../../../testdata/camera720p.rgba"));

        compare_images(&dst, &target_image, 0.98, function!());
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
            .convert(&src, &mut dst, Rotation::None, Flip::None, None)
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
            .convert(
                &src_target,
                &mut dst_target,
                Rotation::None,
                Flip::None,
                None,
            )
            .unwrap();

        compare_images(&dst, &dst_target, 0.98, function!());
    }

    #[test]
    #[cfg(target_os = "linux")]
    fn test_yuyv_to_rgba_crop_flip_g2d() {
        let src = load_bytes_to_tensor(
            1280,
            720,
            YUYV,
            Some(TensorMemory::Dma),
            include_bytes!("../../../testdata/camera720p.yuyv"),
        )
        .unwrap();

        let (dst_width, dst_height) = (640, 640);

        let mut dst_g2d =
            TensorImage::new(dst_width, dst_height, RGBA, Some(TensorMemory::Dma)).unwrap();
        let mut g2d_converter = G2DConverter::new().unwrap();

        g2d_converter
            .convert(
                &src,
                &mut dst_g2d,
                Rotation::None,
                Flip::Horizontal,
                Some(Rect {
                    left: 20,
                    top: 15,
                    width: 400,
                    height: 300,
                }),
            )
            .unwrap();

        let mut dst_cpu =
            TensorImage::new(dst_width, dst_height, RGBA, Some(TensorMemory::Dma)).unwrap();
        let mut cpu_converter = CPUConverter::new().unwrap();

        cpu_converter
            .convert(
                &src,
                &mut dst_cpu,
                Rotation::None,
                Flip::Horizontal,
                Some(Rect {
                    left: 20,
                    top: 15,
                    width: 400,
                    height: 300,
                }),
            )
            .unwrap();
        compare_images(&dst_g2d, &dst_cpu, 0.98, function!());
    }

    #[test]
    #[cfg(target_os = "linux")]
    #[cfg(feature = "opengl")]
    fn test_yuyv_to_rgba_crop_flip_opengl() {
        let src = load_bytes_to_tensor(
            1280,
            720,
            YUYV,
            Some(TensorMemory::Dma),
            include_bytes!("../../../testdata/camera720p.yuyv"),
        )
        .unwrap();

        let (dst_width, dst_height) = (640, 640);

        let mut dst_gl =
            TensorImage::new(dst_width, dst_height, RGBA, Some(TensorMemory::Dma)).unwrap();
        let mut gl_converter = GLConverter::new().unwrap();

        gl_converter
            .convert(
                &src,
                &mut dst_gl,
                Rotation::None,
                Flip::Horizontal,
                Some(Rect {
                    left: 20,
                    top: 15,
                    width: 400,
                    height: 300,
                }),
            )
            .unwrap();

        let mut dst_cpu =
            TensorImage::new(dst_width, dst_height, RGBA, Some(TensorMemory::Dma)).unwrap();
        let mut cpu_converter = CPUConverter::new().unwrap();

        cpu_converter
            .convert(
                &src,
                &mut dst_cpu,
                Rotation::None,
                Flip::Horizontal,
                Some(Rect {
                    left: 20,
                    top: 15,
                    width: 400,
                    height: 300,
                }),
            )
            .unwrap();
        compare_images(&dst_gl, &dst_cpu, 0.98, function!());
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

    fn compare_images(img1: &TensorImage, img2: &TensorImage, threshold: f64, name: &str) {
        assert_eq!(img1.height(), img2.height(), "Heights differ");
        assert_eq!(img1.width(), img2.width(), "Widths differ");
        assert_eq!(img1.fourcc(), img2.fourcc(), "FourCC differ");
        assert!(
            matches!(img1.fourcc(), RGB | RGBA,),
            "FourCC must be RGB or RGBA for comparison"
        );
        let image1 = match img1.fourcc() {
            RGB => image::RgbImage::from_vec(
                img1.width() as u32,
                img1.height() as u32,
                img1.tensor().map().unwrap().to_vec(),
            )
            .unwrap(),
            RGBA => image::RgbaImage::from_vec(
                img1.width() as u32,
                img1.height() as u32,
                img1.tensor().map().unwrap().to_vec(),
            )
            .unwrap()
            .convert(),

            _ => return,
        };

        let image2 = match img2.fourcc() {
            RGB => image::RgbImage::from_vec(
                img2.width() as u32,
                img2.height() as u32,
                img2.tensor().map().unwrap().to_vec(),
            )
            .unwrap(),
            RGBA => image::RgbaImage::from_vec(
                img2.width() as u32,
                img2.height() as u32,
                img2.tensor().map().unwrap().to_vec(),
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
}
