// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0
//! Reading an image's `(width, height)` back out of an existing
//! `ID3D11Texture2D`'s own description.
//!
//! Compiled for both backends.

use super::com::hr;
use super::device::device;
use crate::d3d11_layout::image_d3d11_layout;
use crate::{DType, Error, PixelFormat, PixelLayout, Result};
use std::ffi::c_void;
use std::os::windows::io::RawHandle;
use windows::core::Interface;
use windows::Win32::Foundation::HANDLE;
use windows::Win32::Graphics::Direct3D11::{ID3D11Texture2D, D3D11_TEXTURE2D_DESC};

/// The image `(width, height)` a texture description encodes for `format`,
/// with `shape` supplying what the description cannot carry.
///
/// Proposes the dimensions each repacking (packed RGB in RGBA texels, four
/// planar values per texel, channels stacked vertically) would have come from,
/// re-runs [`image_d3d11_layout`] on every proposal, and keeps the one that
/// reproduces the description. A combined-plane semi-planar texture's width
/// comes from `shape` (see [`semi_planar_width`]).
fn geometry_of(
    tex: &ID3D11Texture2D,
    format: PixelFormat,
    shape: Option<&[usize]>,
) -> Result<(usize, usize)> {
    let mut desc = D3D11_TEXTURE2D_DESC::default();
    // SAFETY: `tex` is live and `desc` is a valid out-parameter.
    unsafe { tex.GetDesc(&mut desc) };
    let (tex_w, tex_h) = (desc.Width as usize, desc.Height as usize);
    let dxgi = desc.Format.0 as u32;

    if dxgi == crate::d3d11_layout::DXGI_FORMAT_NV12 {
        native_nv12_geometry(format, tex_w, tex_h)
    } else if format.layout() == PixelLayout::SemiPlanar {
        semi_planar_geometry(format, shape, tex_w, tex_h, dxgi)
    } else {
        // Every other format's dimensions are in the description alone;
        // `shape` is not read.
        packed_or_planar_geometry(format, tex_w, tex_h, dxgi)
    }
}

/// The image width of a semi-planar texture, read from index 1 of `shape` and
/// checked to fit the texture's row. Both spellings of a semi-planar shape put
/// the width at index 1: the allocation `[combined height, width]` and the
/// addressing `[height, width]`.
fn semi_planar_width(format: PixelFormat, shape: Option<&[usize]>, tex_w: usize) -> Result<usize> {
    let shape = shape.ok_or_else(|| {
        Error::InvalidArgument(format!(
            "d3d11 texture geometry: semi-planar textures carry a padded width; pass the tensor \
             shape so the {format:?} image width is known"
        ))
    })?;
    if shape.len() != 2 {
        return Err(Error::InvalidShape(format!(
            "d3d11 texture geometry: a {format:?} shape is rank 2 -- [combined height, width] or \
             [height, width] -- not {shape:?}"
        )));
    }
    let width = shape[1];
    let even = width.checked_next_multiple_of(2).ok_or_else(|| {
        Error::InvalidShape(format!(
            "d3d11 texture geometry: a {format:?} shape carries a width of {width}, which has no \
             even row"
        ))
    })?;
    if width == 0 || even > tex_w {
        return Err(Error::InvalidShape(format!(
            "d3d11 texture geometry: a {format:?} image {width} wide needs a texture row of at \
             least {even} texels, but this texture is {tex_w} wide"
        )));
    }
    Ok(width)
}

/// The image dimensions of a native NV12 texture: the luma grid.
fn native_nv12_geometry(format: PixelFormat, tex_w: usize, tex_h: usize) -> Result<(usize, usize)> {
    if format != PixelFormat::Nv12 {
        return Err(Error::InvalidArgument(format!(
            "d3d11 texture geometry: a DXGI_FORMAT_NV12 texture holds an Nv12 image, not {format:?}"
        )));
    }
    Ok((tex_w, tex_h))
}

/// The image dimensions of a semi-planar texture: the width from the shape,
/// the height inverted from the combined plane's row count.
fn semi_planar_geometry(
    format: PixelFormat,
    shape: Option<&[usize]>,
    tex_w: usize,
    tex_h: usize,
    dxgi: u32,
) -> Result<(usize, usize)> {
    let width = semi_planar_width(format, shape, tex_w)?;
    // The row count is the combined plane height; a count no image height
    // produces is an error.
    let height = format.image_height_from_combined(tex_h).ok_or_else(|| {
        Error::InvalidArgument(format!(
            "d3d11 texture geometry: {tex_h} texture rows are not a {format:?} combined plane \
             height, so this texture does not hold a {format:?} image"
        ))
    })?;
    // Checked against the combined-plane layout.
    let l = crate::d3d11_layout::combined_semi_planar_layout(format, width, height).ok_or_else(
        || {
            Error::InvalidArgument(format!(
                "d3d11 texture geometry: {format:?} has no D3D11 layout at {width}x{height}"
            ))
        },
    )?;
    if l.dxgi_format != dxgi {
        return Err(Error::InvalidArgument(format!(
            "d3d11 texture geometry: a {tex_w}x{tex_h} texture in DXGI format {dxgi} does not \
             hold a {format:?} image: {format:?} is allocated in DXGI format {} at that grid \
             (see edgefirst_tensor::d3d11_layout)",
            l.dxgi_format
        )));
    }
    Ok((width, height))
}

/// The image dimensions of a packed or planar texture, found by re-running the
/// layout table over the repackings that could have produced this grid.
fn packed_or_planar_geometry(
    format: PixelFormat,
    tex_w: usize,
    tex_h: usize,
    dxgi: u32,
) -> Result<(usize, usize)> {
    let heights: Vec<usize> = match format.layout() {
        // A planar image stacks its channels vertically.
        PixelLayout::Planar => vec![tex_h, tex_h / format.channels().max(1)],
        _ => vec![tex_h],
    };
    // Identity, the packed-RGB inverse (`w * 3 / 4`) and the four-values-per-
    // texel inverse (`w / 4`).
    let widths = [tex_w, tex_w * 4 / 3, tex_w * 4];

    let mut expected: Vec<u32> = Vec::new();
    for &dtype in DType::all() {
        for &h in &heights {
            for &w in &widths {
                let Some(l) = image_d3d11_layout(format, dtype, w, h) else {
                    continue;
                };
                if l.texture_width != tex_w || l.texture_height != tex_h {
                    continue;
                }
                if l.dxgi_format == dxgi {
                    return Ok((w, h));
                }
                if !expected.contains(&l.dxgi_format) {
                    expected.push(l.dxgi_format);
                }
            }
        }
    }
    // An empty list means no dtype of this format has a layout at this grid.
    let detail = if expected.is_empty() {
        format!("{format:?} has no D3D11 layout at a {tex_w}x{tex_h} texture grid")
    } else {
        format!("{format:?} is allocated in DXGI format one of {expected:?} at that grid")
    };
    Err(Error::InvalidArgument(format!(
        "d3d11 texture geometry: a {tex_w}x{tex_h} texture in DXGI format {dxgi} does not \
         hold a {format:?} image: {detail} (see edgefirst_tensor::d3d11_layout)"
    )))
}

/// The image `(width, height)` an existing `ID3D11Texture2D` holds for
/// `format`, read from its own description.
///
/// The height is the image height: for a semi-planar format the description
/// carries the combined luma + chroma row count, which is inverted through
/// [`PixelFormat::image_height_from_combined`].
///
/// A combined-plane semi-planar texture is as wide as the image's row stride.
/// For such an `Nv12`, `Nv16` or `Nv24` texture the width is taken from
/// `shape` (index 1 in both the allocation and the addressing spelling) and
/// checked to fit; omitting the shape is an error. For every other format
/// `shape` is unused.
///
/// The description is checked against the layout [`image_d3d11_layout`]
/// chooses for `format`: a texture of the wrong DXGI format or the wrong grid
/// is refused. Two formats that share a layout exactly are indistinguishable
/// by description and both answer.
///
/// # Errors
///
/// [`Error::InvalidArgument`] for a null texture, a semi-planar row count no
/// image height produces, a semi-planar texture with no `shape`, or a
/// description that is not the layout `format` is allocated with.
/// [`Error::InvalidShape`] for a `shape` that is not a rank-2 semi-planar
/// shape or whose width does not fit the texture's rows. Propagates the device
/// error when there is no process device.
///
/// # Safety
///
/// `texture` must be null or a live `ID3D11Texture2D`. Ownership stays with
/// the caller: this takes and drops its own reference.
pub unsafe fn d3d11_texture_geometry(
    texture: *mut c_void,
    format: PixelFormat,
    shape: Option<&[usize]>,
) -> Result<(usize, usize)> {
    // SAFETY: the caller guarantees `texture` is null or a live
    // `ID3D11Texture2D`; `cloned` takes the reference this call drops on
    // return.
    let tex = unsafe { ID3D11Texture2D::from_raw_borrowed(&texture) }
        .cloned()
        .ok_or_else(|| Error::InvalidArgument("d3d11_texture_geometry: null texture".to_owned()))?;
    geometry_of(&tex, format, shape)
}

/// [`d3d11_texture_geometry`] for a shared NT handle: opens it on the process
/// device and reads the same description. See that function for the contract.
///
/// # Errors
///
/// As [`d3d11_texture_geometry`], plus whatever `OpenSharedResource1` reports
/// for a handle that is not a shareable D3D11 texture.
///
/// # Safety
///
/// `handle` must be an NT shared handle of a D3D11 texture, valid in this
/// process. It stays owned by the caller: this opens a texture from it and
/// drops that texture on return.
pub unsafe fn d3d11_shared_handle_geometry(
    handle: RawHandle,
    format: PixelFormat,
    shape: Option<&[usize]>,
) -> Result<(usize, usize)> {
    let d = device()?;
    let dev1 = d.dev1().ok_or_else(|| {
        Error::NotImplemented("ID3D11Device1 required to open shared handles".to_owned())
    })?;
    // SAFETY: the caller guarantees an NT handle valid in this process.
    let tex: ID3D11Texture2D = hr("ID3D11Device1::OpenSharedResource1", unsafe {
        dev1.OpenSharedResource1(HANDLE(handle))
    })?;
    geometry_of(&tex, format, shape)
}

#[cfg(all(test, not(edgefirst_tensor_dynamic)))]
mod tests {
    use super::*;
    use crate::d3d11::texture::D3d11TextureTensor;
    use crate::CpuAccess;
    use std::os::windows::io::AsRawHandle;

    /// The geometry helpers read the image dimensions off the texture description.
    #[test]
    fn geometry_helpers_report_the_image_dimensions() {
        for (fmt, w, h) in [
            (PixelFormat::Rgba, 64usize, 32usize),
            (PixelFormat::Nv12, 640, 481),
        ] {
            let shape = fmt.allocation_shape(w, h).unwrap();
            let t = D3d11TextureTensor::<u8>::new_image(
                w,
                h,
                fmt,
                DType::U8,
                &shape,
                None,
                CpuAccess::None,
            )
            .unwrap();
            // SAFETY: `t` holds the texture live across the call.
            let by_ptr =
                unsafe { d3d11_texture_geometry(t.texture_ptr(), fmt, Some(&shape)) }.unwrap();
            assert_eq!(by_ptr, (w, h), "{fmt:?} through the pointer");

            let handle = t.shared_handle().unwrap();
            // SAFETY: `handle` is a shared NT handle this test owns.
            let by_handle =
                unsafe { d3d11_shared_handle_geometry(handle.as_raw_handle(), fmt, Some(&shape)) }
                    .unwrap();
            assert_eq!(by_handle, (w, h), "{fmt:?} through the shared handle");
        }
    }

    /// A semi-planar texture's width comes from the shape, in either spelling.
    #[test]
    fn semi_planar_geometry_takes_the_width_from_the_shape() {
        let fmt = PixelFormat::Nv12;
        let (w, h) = (640usize, 481usize);
        let allocation = fmt.allocation_shape(w, h).unwrap();
        let addressing = fmt.addressing_shape(w, h).unwrap();
        let t = D3d11TextureTensor::<u8>::new_image(
            w,
            h,
            fmt,
            DType::U8,
            &allocation,
            None,
            CpuAccess::None,
        )
        .unwrap();
        let handle = t.shared_handle().unwrap();
        for shape in [&allocation, &addressing] {
            // SAFETY: `t` holds the texture live across the call.
            let by_ptr =
                unsafe { d3d11_texture_geometry(t.texture_ptr(), fmt, Some(shape)) }.unwrap();
            assert_eq!(by_ptr, (w, h), "{shape:?} through the pointer");
            // SAFETY: `handle` is a shared NT handle this test owns.
            let by_handle =
                unsafe { d3d11_shared_handle_geometry(handle.as_raw_handle(), fmt, Some(shape)) }
                    .unwrap();
            assert_eq!(by_handle, (w, h), "{shape:?} through the shared handle");
        }
    }

    /// A semi-planar texture without a shape is refused by both helpers.
    #[test]
    fn semi_planar_geometry_without_a_shape_is_refused() {
        let fmt = PixelFormat::Nv12;
        let shape = fmt.allocation_shape(640, 481).unwrap();
        let t = D3d11TextureTensor::<u8>::new_image(
            640,
            481,
            fmt,
            DType::U8,
            &shape,
            None,
            CpuAccess::None,
        )
        .unwrap();
        // SAFETY: `t` holds the texture live across the call.
        let err = unsafe { d3d11_texture_geometry(t.texture_ptr(), fmt, None) }.unwrap_err();
        assert!(
            matches!(err, Error::InvalidArgument(_)) && err.to_string().contains("tensor shape"),
            "{err}"
        );

        let handle = t.shared_handle().unwrap();
        // SAFETY: `handle` is a shared NT handle this test owns.
        let err =
            unsafe { d3d11_shared_handle_geometry(handle.as_raw_handle(), fmt, None) }.unwrap_err();
        assert!(
            matches!(err, Error::InvalidArgument(_)) && err.to_string().contains("tensor shape"),
            "{err}"
        );
    }

    /// A shape width wider than the texture's row is refused.
    #[test]
    fn semi_planar_geometry_refuses_a_width_wider_than_the_texture() {
        let fmt = PixelFormat::Nv12;
        let shape = fmt.allocation_shape(640, 481).unwrap();
        let t = D3d11TextureTensor::<u8>::new_image(
            640,
            481,
            fmt,
            DType::U8,
            &shape,
            None,
            CpuAccess::None,
        )
        .unwrap();
        let too_wide = fmt.allocation_shape(641, 481).unwrap();
        // SAFETY: `t` holds the texture live across the call.
        let err =
            unsafe { d3d11_texture_geometry(t.texture_ptr(), fmt, Some(&too_wide)) }.unwrap_err();
        assert!(matches!(err, Error::InvalidShape(_)), "{err}");
    }

    /// A texture whose description is not the format's layout is refused.
    #[test]
    fn geometry_helpers_refuse_a_texture_of_another_format() {
        let shape = PixelFormat::Grey.allocation_shape(640, 481).unwrap();
        let t = D3d11TextureTensor::<u8>::new_image(
            640,
            481,
            PixelFormat::Grey,
            DType::U8,
            &shape,
            None,
            CpuAccess::None,
        )
        .unwrap();
        // SAFETY: `t` holds the texture live across the call.
        let err =
            unsafe { d3d11_texture_geometry(t.texture_ptr(), PixelFormat::Nv12, Some(&shape)) }
                .unwrap_err();
        // A Grey allocation shape is rank 3: the error is `InvalidShape`.
        assert!(
            matches!(err, Error::InvalidShape(_)) && err.to_string().contains("Nv12"),
            "{err}"
        );

        let handle = t.shared_handle().unwrap();
        // SAFETY: `handle` is a shared NT handle this test owns.
        let err = unsafe {
            d3d11_shared_handle_geometry(handle.as_raw_handle(), PixelFormat::Nv12, Some(&shape))
        }
        .unwrap_err();
        assert!(matches!(err, Error::InvalidShape(_)), "{err}");

        // A format with no layout at this texture grid.
        let shape = PixelFormat::Rgba.allocation_shape(64, 32).unwrap();
        let rgba = D3d11TextureTensor::<u8>::new_image(
            64,
            32,
            PixelFormat::Rgba,
            DType::U8,
            &shape,
            None,
            CpuAccess::None,
        )
        .unwrap();
        // SAFETY: `rgba` holds the texture live across the call.
        let err =
            unsafe { d3d11_texture_geometry(rgba.texture_ptr(), PixelFormat::PlanarRgb, None) }
                .unwrap_err();
        assert!(
            err.to_string()
                .contains("no D3D11 layout at a 64x32 texture grid"),
            "{err}"
        );
    }
}
