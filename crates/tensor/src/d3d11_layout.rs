// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0
//! `(PixelFormat, DType, width, height) -> D3D11 texture layout`.
//!
//! Compiled on every platform so the table is unit-tested on every CI lane;
//! only the allocator that consumes it is Windows-only. Single source of
//! truth for which DXGI format and texture geometry back each HAL image
//! format; the image crate reads `gl_internal_format` when it builds the
//! `EGL_ANGLE_image_d3d11_texture` attribute list.

use crate::{DType, PixelFormat};

pub const DXGI_FORMAT_R32G32B32A32_FLOAT: u32 = 2;
pub const DXGI_FORMAT_R16G16B16A16_FLOAT: u32 = 10;
pub const DXGI_FORMAT_R8G8B8A8_UNORM: u32 = 28;
pub const DXGI_FORMAT_R8G8_UNORM: u32 = 49;
pub const DXGI_FORMAT_R8_UNORM: u32 = 61;
pub const DXGI_FORMAT_B8G8R8A8_UNORM: u32 = 87;

/// Values accepted for `EGL_TEXTURE_INTERNAL_FORMAT_ANGLE` by ANGLE's D3D11
/// client-buffer validation (`Renderer11::getD3DTextureInfo`).
pub const GL_RGBA: u32 = 0x1908;
pub const GL_BGRA_EXT: u32 = 0x80E1;
pub const GL_RED_EXT: u32 = 0x1903;
pub const GL_RG_EXT: u32 = 0x8227;

/// One texture's format and geometry for a HAL image. `texture_width` and
/// `texture_height` are texels and rows, not image pixels: the packed rows
/// fold several pixels or several planes into one texel, and a semi-planar
/// row is the image's row *stride* rather than its width.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct D3d11ImageLayout {
    pub dxgi_format: u32,
    pub texture_width: usize,
    pub texture_height: usize,
    pub bytes_per_texel: usize,
    pub gl_internal_format: u32,
}

impl D3d11ImageLayout {
    /// Bytes in one texture row, with no driver padding: the row a copy into
    /// or out of the texture moves.
    ///
    /// This is the image's own row for every format except a semi-planar one,
    /// whose texture row is the stride and can exceed the image width (see
    /// [`image_d3d11_layout`]). The image footprint of such a tensor is the
    /// storage's business, not this table's, because only the storage still
    /// knows the image width.
    pub fn tight_row_bytes(&self) -> usize {
        self.texture_width * self.bytes_per_texel
    }
    /// Bytes in the whole texture, padding-free: `tight_row_bytes` over every
    /// row, with the same semi-planar caveat.
    pub fn tight_bytes(&self) -> usize {
        self.tight_row_bytes() * self.texture_height
    }

    /// This layout with its rows widened to `texture_width` texels.
    ///
    /// The one sanctioned edit of a table entry, and only for a semi-planar
    /// row: its texture must be as wide as the row pitch the driver hands a
    /// CPU map, and a pitch is a property of a created texture rather than of
    /// the format, so the allocator probes for one and widens the entry here
    /// (see `D3d11TextureTensor::new_image`). A wrapped external texture is
    /// widened the same way, to the width its host chose.
    ///
    /// A width below the entry's own is not a widening and leaves it alone:
    /// the table's width is the floor every consumer of the layout may assume.
    pub fn widened_to(&self, texture_width: usize) -> Self {
        Self {
            texture_width: texture_width.max(self.texture_width),
            ..*self
        }
    }
}

fn lay(dxgi_format: u32, w: usize, h: usize, bpt: usize, gl: u32) -> Option<D3d11ImageLayout> {
    if w == 0 || h == 0 {
        return None;
    }
    Some(D3d11ImageLayout {
        dxgi_format,
        texture_width: w,
        texture_height: h,
        bytes_per_texel: bpt,
        gl_internal_format: gl,
    })
}

/// The D3D11 texture that backs `format`/`dtype` at `width` x `height`, or
/// `None` when the format has no zero-copy texture layout (it then falls to
/// PBO or Mem in the allocation chain).
pub fn image_d3d11_layout(
    format: PixelFormat,
    dtype: DType,
    width: usize,
    height: usize,
) -> Option<D3d11ImageLayout> {
    use PixelFormat::*;
    if width == 0 || height == 0 {
        return None;
    }
    // Four contiguous values per texel: three RGB bytes or floats per pixel
    // and four planar values per texel both need a width divisible by 4.
    let quarter = |n: usize| {
        if n.is_multiple_of(4) {
            Some(n / 4)
        } else {
            None
        }
    };
    match (format, dtype) {
        (Rgba, DType::U8 | DType::I8) => lay(DXGI_FORMAT_R8G8B8A8_UNORM, width, height, 4, GL_RGBA),
        (Bgra, DType::U8) => lay(DXGI_FORMAT_B8G8R8A8_UNORM, width, height, 4, GL_BGRA_EXT),
        (Rgb, DType::U8 | DType::I8) => lay(
            DXGI_FORMAT_R8G8B8A8_UNORM,
            quarter(width * 3)?,
            height,
            4,
            GL_RGBA,
        ),
        (Grey, DType::U8) => lay(DXGI_FORMAT_R8_UNORM, width, height, 1, GL_RED_EXT),
        // The HAL's combined luma + interleaved chroma plane, the layout the
        // Path B shader already decodes on Linux and macOS.
        //
        // The texture row is the plane's row *stride*, not the image width:
        // producers write luma and chroma lines at the stride and the shader
        // wraps chroma addressing at the sampled texture's width, so the two
        // have to be one number or a chroma line lands on the wrong row. The
        // even rounding is the floor -- an interleaved chroma pair cannot
        // straddle a row edge -- and the Windows allocator widens it further
        // to the driver's row pitch through
        // [`D3d11ImageLayout::widened_to`].
        (Nv12 | Nv16 | Nv24, DType::U8) => {
            let shape = format.allocation_shape(width, height)?;
            lay(
                DXGI_FORMAT_R8_UNORM,
                shape[1].next_multiple_of(2),
                shape[0],
                1,
                GL_RED_EXT,
            )
        }
        (Yuyv, DType::U8) => lay(DXGI_FORMAT_R8G8_UNORM, width, height, 2, GL_RG_EXT),
        (PlanarRgb | PlanarRgba, DType::F16) => lay(
            DXGI_FORMAT_R16G16B16A16_FLOAT,
            quarter(width)?,
            format.channels() * height,
            8,
            GL_RGBA,
        ),
        (PlanarRgb | PlanarRgba, DType::F32) => lay(
            DXGI_FORMAT_R32G32B32A32_FLOAT,
            quarter(width)?,
            format.channels() * height,
            16,
            GL_RGBA,
        ),
        (Rgb, DType::F16) => lay(
            DXGI_FORMAT_R16G16B16A16_FLOAT,
            quarter(width * 3)?,
            height,
            8,
            GL_RGBA,
        ),
        (Rgb, DType::F32) => lay(
            DXGI_FORMAT_R32G32B32A32_FLOAT,
            quarter(width * 3)?,
            height,
            16,
            GL_RGBA,
        ),
        (Rgba, DType::F16) => lay(DXGI_FORMAT_R16G16B16A16_FLOAT, width, height, 8, GL_RGBA),
        (Rgba, DType::F32) => lay(DXGI_FORMAT_R32G32B32A32_FLOAT, width, height, 16, GL_RGBA),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn packed_8bit_rows_are_direct() {
        let l = image_d3d11_layout(PixelFormat::Rgba, DType::U8, 640, 480).unwrap();
        assert_eq!(
            (
                l.dxgi_format,
                l.texture_width,
                l.texture_height,
                l.bytes_per_texel,
                l.gl_internal_format
            ),
            (DXGI_FORMAT_R8G8B8A8_UNORM, 640, 480, 4, GL_RGBA)
        );
        let l = image_d3d11_layout(PixelFormat::Bgra, DType::U8, 640, 480).unwrap();
        assert_eq!(
            (l.dxgi_format, l.gl_internal_format),
            (DXGI_FORMAT_B8G8R8A8_UNORM, GL_BGRA_EXT)
        );
        let l = image_d3d11_layout(PixelFormat::Grey, DType::U8, 641, 3).unwrap();
        assert_eq!(
            (
                l.dxgi_format,
                l.texture_width,
                l.bytes_per_texel,
                l.gl_internal_format
            ),
            (DXGI_FORMAT_R8_UNORM, 641, 1, GL_RED_EXT)
        );
        assert_eq!(l.tight_bytes(), 641 * 3);
    }

    #[test]
    fn packed_rgb_u8_packs_three_bytes_per_rgba_texel() {
        let l = image_d3d11_layout(PixelFormat::Rgb, DType::U8, 640, 480).unwrap();
        assert_eq!(
            (
                l.dxgi_format,
                l.texture_width,
                l.texture_height,
                l.bytes_per_texel
            ),
            (DXGI_FORMAT_R8G8B8A8_UNORM, 480, 480, 4)
        );
        assert!(
            image_d3d11_layout(PixelFormat::Rgb, DType::U8, 641, 480).is_none(),
            "W % 4 != 0 has no packing"
        );
    }

    #[test]
    fn semi_planar_is_the_combined_r8_plane() {
        let l = image_d3d11_layout(PixelFormat::Nv12, DType::U8, 128, 64).unwrap();
        assert_eq!(
            (
                l.dxgi_format,
                l.texture_width,
                l.texture_height,
                l.gl_internal_format
            ),
            (DXGI_FORMAT_R8_UNORM, 128, 96, GL_RED_EXT)
        );
        let l = image_d3d11_layout(PixelFormat::Nv16, DType::U8, 128, 64).unwrap();
        assert_eq!((l.texture_width, l.texture_height), (128, 128));
        let l = image_d3d11_layout(PixelFormat::Nv24, DType::U8, 128, 64).unwrap();
        assert_eq!((l.texture_width, l.texture_height), (128, 192));
        assert_eq!(
            l.tight_bytes(),
            PixelFormat::Nv24
                .allocation_shape(128, 64)
                .unwrap()
                .iter()
                .product::<usize>()
        );
    }

    /// An odd image width rounds the texture row up to the next even texel,
    /// because a chroma pair that straddles the row edge is unaddressable by
    /// the shader that decodes the combined plane.
    #[test]
    fn semi_planar_rounds_an_odd_width_up_to_an_even_texture_row() {
        for (fmt, rows) in [
            (PixelFormat::Nv12, 362),
            (PixelFormat::Nv16, 482),
            (PixelFormat::Nv24, 723),
        ] {
            let l = image_d3d11_layout(fmt, DType::U8, 321, 241).unwrap();
            assert_eq!(
                (l.texture_width, l.texture_height),
                (322, rows),
                "{fmt:?} 321x241"
            );
        }
        // An even width is untouched, so nothing that already fit changes.
        let l = image_d3d11_layout(PixelFormat::Nv12, DType::U8, 640, 480).unwrap();
        assert_eq!((l.texture_width, l.texture_height), (640, 720));
    }

    /// The allocator widens a semi-planar entry to the driver's row pitch and
    /// a wrap widens it to the host's texture width, so `widened_to` has to
    /// widen and never narrow: the table's width is the floor every consumer
    /// of a layout may assume.
    #[test]
    fn widened_to_widens_and_never_narrows() {
        let l = image_d3d11_layout(PixelFormat::Nv12, DType::U8, 321, 241).unwrap();
        assert_eq!(l.texture_width, 322, "an odd width already rounds up");

        // Below the entry's own width: not a widening, so it is ignored.
        assert_eq!(l.widened_to(64).texture_width, 322);
        assert_eq!(l.widened_to(321).texture_width, 322);
        // Equal, and above.
        assert_eq!(l.widened_to(322).texture_width, 322);
        assert_eq!(l.widened_to(384).texture_width, 384);
        assert_eq!(
            l.widened_to(384).tight_row_bytes(),
            384,
            "R8 rows are texels"
        );
        // Every other field is the entry's.
        let w = l.widened_to(384);
        assert_eq!(
            (
                w.dxgi_format,
                w.texture_height,
                w.bytes_per_texel,
                w.gl_internal_format
            ),
            (
                l.dxgi_format,
                l.texture_height,
                l.bytes_per_texel,
                l.gl_internal_format
            )
        );
    }

    #[test]
    fn yuyv_is_rg8_at_full_width() {
        let l = image_d3d11_layout(PixelFormat::Yuyv, DType::U8, 320, 240).unwrap();
        assert_eq!(
            (
                l.dxgi_format,
                l.texture_width,
                l.bytes_per_texel,
                l.gl_internal_format
            ),
            (DXGI_FORMAT_R8G8_UNORM, 320, 2, GL_RG_EXT)
        );
    }

    #[test]
    fn planar_float_packs_four_per_texel() {
        let l = image_d3d11_layout(PixelFormat::PlanarRgb, DType::F16, 640, 480).unwrap();
        assert_eq!(
            (
                l.dxgi_format,
                l.texture_width,
                l.texture_height,
                l.bytes_per_texel
            ),
            (DXGI_FORMAT_R16G16B16A16_FLOAT, 160, 1440, 8)
        );
        let l = image_d3d11_layout(PixelFormat::PlanarRgba, DType::F32, 640, 480).unwrap();
        assert_eq!(
            (
                l.dxgi_format,
                l.texture_width,
                l.texture_height,
                l.bytes_per_texel
            ),
            (DXGI_FORMAT_R32G32B32A32_FLOAT, 160, 1920, 16)
        );
        assert_eq!(l.tight_bytes(), 4 * 480 * 640 * 4);
        assert!(image_d3d11_layout(PixelFormat::PlanarRgb, DType::F16, 642, 480).is_none());
    }

    #[test]
    fn interleaved_float_rows() {
        let l = image_d3d11_layout(PixelFormat::Rgb, DType::F32, 640, 480).unwrap();
        assert_eq!(
            (
                l.dxgi_format,
                l.texture_width,
                l.texture_height,
                l.bytes_per_texel
            ),
            (DXGI_FORMAT_R32G32B32A32_FLOAT, 480, 480, 16)
        );
        let l = image_d3d11_layout(PixelFormat::Rgba, DType::F16, 640, 480).unwrap();
        assert_eq!(
            (
                l.dxgi_format,
                l.texture_width,
                l.texture_height,
                l.bytes_per_texel
            ),
            (DXGI_FORMAT_R16G16B16A16_FLOAT, 640, 480, 8)
        );
    }

    #[test]
    fn unsupported_pairs_are_none() {
        assert!(image_d3d11_layout(PixelFormat::Vyuy, DType::U8, 64, 64).is_none());
        assert!(image_d3d11_layout(PixelFormat::Grey, DType::F16, 64, 64).is_none());
        assert!(image_d3d11_layout(PixelFormat::Rgba, DType::U8, 0, 64).is_none());
    }
}
