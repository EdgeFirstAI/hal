#![cfg(target_os = "linux")]

use crate::{
    Error, Flip, GREY, ImageConverterTrait, NV12, RGB, Rect, Result, Rotation, TensorImage, YUYV,
};
use edgefirst_tensor::Tensor;
use g2d_sys::{G2D, G2DFormat, G2DPhysical, G2DSurface};
use log::debug;
use std::os::fd::AsRawFd;

/// G2DConverter implements the ImageConverter trait using the NXP G2D
/// library for hardware-accelerated image processing on i.MX platforms.
pub struct G2DConverter {
    g2d: G2D,
}

impl G2DConverter {
    pub fn new() -> Result<Self> {
        let mut g2d = G2D::new("libg2d.so.2")?;
        g2d.set_bt709_colorspace()?;
        Ok(Self { g2d })
    }

    pub fn version(&self) -> g2d_sys::Version {
        self.g2d.version()
    }
}

impl ImageConverterTrait for G2DConverter {
    /// Converts the source image to the destination image using G2D.
    ///
    /// # Arguments
    ///
    /// * `dst` - The destination image to be converted to.
    /// * `src` - The source image to convert from.
    /// * `rotation` - The rotation to apply to the destination image (after
    ///   crop if specified).
    /// * `crop` - An optional rectangle specifying the area to crop from the
    ///   source image.
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
    ) -> Result<()> {
        if matches!(dst.fourcc(), YUYV | GREY | NV12 | RGB) {
            return Err(Error::NotSupported(format!(
                "G2D does not support {} destination",
                dst.fourcc().display()
            )));
        }
        let mut src_surface: G2DSurface = src.try_into()?;
        let mut dst_surface: G2DSurface = dst.try_into()?;

        src_surface.rot = match flip {
            Flip::None => g2d_sys::g2d_rotation_G2D_ROTATION_0,
            Flip::Vertical => g2d_sys::g2d_rotation_G2D_FLIP_V,
            Flip::Horizontal => g2d_sys::g2d_rotation_G2D_FLIP_H,
        };

        dst_surface.rot = match rotation {
            Rotation::None => g2d_sys::g2d_rotation_G2D_ROTATION_0,
            Rotation::Clockwise90 => g2d_sys::g2d_rotation_G2D_ROTATION_90,
            Rotation::Rotate180 => g2d_sys::g2d_rotation_G2D_ROTATION_180,
            Rotation::CounterClockwise90 => g2d_sys::g2d_rotation_G2D_ROTATION_270,
        };

        if let Some(crop_rect) = crop {
            src_surface.left = crop_rect.left as i32;
            src_surface.top = crop_rect.top as i32;
            src_surface.right = (crop_rect.left + crop_rect.width) as i32;
            src_surface.bottom = (crop_rect.top + crop_rect.height) as i32;
        }

        debug!("Blitting from {src_surface:?} to {dst_surface:?}");

        self.g2d.blit(&src_surface, &dst_surface)?;

        Ok(())
    }
}

impl TryFrom<&TensorImage> for G2DSurface {
    type Error = Error;

    fn try_from(img: &TensorImage) -> Result<Self, Self::Error> {
        let phys: G2DPhysical = match img.tensor() {
            Tensor::Shm(t) => t.as_raw_fd(),
            Tensor::Dma(t) => t.as_raw_fd(),
            _ => {
                return Err(Error::NotImplemented(
                    "g2d only supports Shm or Dma memory".to_string(),
                ));
            }
        }
        .try_into()?;

        Ok(Self {
            planes: [phys.address(), 0, 0],
            format: G2DFormat::try_from(img.fourcc())?.format(),
            left: 0,
            top: 0,
            right: img.width() as i32,
            bottom: img.height() as i32,
            stride: img.width() as i32,
            width: img.width() as i32,
            height: img.height() as i32,
            blendfunc: 0,
            clrcolor: 0,
            rot: 0,
            global_alpha: 0,
        })
    }
}

impl TryFrom<&mut TensorImage> for G2DSurface {
    type Error = Error;

    fn try_from(img: &mut TensorImage) -> Result<Self, Self::Error> {
        let phys: G2DPhysical = match img.tensor() {
            Tensor::Shm(t) => t.as_raw_fd(),
            Tensor::Dma(t) => t.as_raw_fd(),
            _ => {
                return Err(Error::NotImplemented(
                    "g2d only supports Shm or Dma memory".to_string(),
                ));
            }
        }
        .try_into()?;

        Ok(Self {
            planes: [phys.address(), 0, 0],
            format: G2DFormat::try_from(img.fourcc())?.format(),
            left: 0,
            top: 0,
            right: img.width() as i32,
            bottom: img.height() as i32,
            stride: img.width() as i32,
            width: img.width() as i32,
            height: img.height() as i32,
            blendfunc: 0,
            clrcolor: 0,
            rot: 0,
            global_alpha: 0,
        })
    }
}
