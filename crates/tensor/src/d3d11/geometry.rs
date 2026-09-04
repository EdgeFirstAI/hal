// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0
//! Reading an image's `(width, height)` back out of an existing
//! `ID3D11Texture2D`'s own description.
//!
//! Compiled for both backends, unlike the texture storage next door. Every
//! constructor handed a raw texture or a shared handle -- the C ABI, the
//! Python bindings, and the protocol and blob imports -- takes its dimensions
//! from here and treats the shape it was *also* given as a consistency check.
//! Only the C ABI links `static`, so gating this on that feature left the
//! others deriving dimensions from a shape that cannot tell an addressing grid
//! from an allocation grid, and the two backends then disagreed on which
//! spellings of a semi-planar shape were accepted.
//!
//! Nothing here touches `TensorStorage`: it needs the process device, the two
//! COM description calls, and the backend-agnostic layout table, all of which
//! exist under `dynamic` too.

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
/// The texture grid is not always the image grid: a semi-planar texture holds
/// the combined luma + chroma plane, packed RGB rides three bytes per RGBA
/// texel, and the planar float formats pack four values per texel. So this
/// proposes the dimensions each of those repackings would have come from and
/// re-runs [`image_d3d11_layout`] on every proposal, keeping the one that
/// reproduces this description exactly. A proposal that ever drifts from the
/// table therefore produces an error here rather than a wrong size.
///
/// A semi-planar texture is the exception the search cannot cover: it is as
/// wide as the image's row *stride*, which the driver chose and the
/// description does not distinguish from the width, so its width comes from
/// `shape` instead (see [`semi_planar_width`]).
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

    if format.layout() == PixelLayout::SemiPlanar {
        semi_planar_geometry(format, shape, tex_w, tex_h, dxgi)
    } else {
        // Every other format's dimensions are in the description alone, so a
        // shape is the caller's consistency check on the answer and stays
        // theirs: each entry point already compares it against both spellings
        // and reports a mismatch in the error class its own API promises --
        // `ValueError` from Python, `EfErrorClass::InvalidShape` from the C
        // ABI. Repeating the comparison here would pre-empt that with a
        // different class.
        packed_or_planar_geometry(format, tex_w, tex_h, dxgi)
    }
}

/// The image width of a semi-planar texture, which its description cannot
/// carry: the texture is as wide as the row stride, so the width comes from
/// the shape the caller was also given. Both spellings of a semi-planar shape
/// -- the allocation `[combined height, width]` and the addressing
/// `[height, width]` -- put the width at index 1.
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
    // Checked: the shape crosses the C and Python boundaries as a caller's
    // `uint64_t`, so a width near `usize::MAX` must error rather than wrap to a
    // small even number that then fits the texture.
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
    // The row count is the combined plane height, which must invert. A count
    // no image height produces is an error rather than the nearest height that
    // nearly fits.
    let height = format.image_height_from_combined(tex_h).ok_or_else(|| {
        Error::InvalidArgument(format!(
            "d3d11 texture geometry: {tex_h} texture rows are not a {format:?} combined plane \
             height, so this texture does not hold a {format:?} image"
        ))
    })?;
    // Semi-planar is `DType::U8` alone in the layout table, so there is one
    // entry to check rather than a search.
    let l = image_d3d11_layout(format, DType::U8, width, height).ok_or_else(|| {
        Error::InvalidArgument(format!(
            "d3d11 texture geometry: {format:?} has no D3D11 layout at {width}x{height}"
        ))
    })?;
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
    // `DType::all()` rather than a literal list: `DType` is `#[non_exhaustive]`
    // and the generated list grows with it, so a new element type cannot
    // silently drop out of this search.
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
    // An empty list is a different fact from a mismatched one: no dtype of
    // this format has a layout at this texture grid at all, so naming an
    // expected format would be naming nothing.
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
/// The unambiguous source of geometry for a constructor handed a raw texture.
/// Its intended callers are the C and Python entry points and the protocol and
/// blob imports, which tasks A6, A7, D1 and D2 add: each is meant to take its
/// dimensions from here and treat any `dims` or shape it was *also* given as a
/// consistency check, rather than deriving the size from a shape that cannot
/// distinguish an addressing grid from an allocation grid (see
/// [`crate::image_dims_from_shape`]). D1 and D2 pass the descriptor's and the
/// blob's shape here for the same reason every other caller does.
///
/// The height is the *image* height: for a semi-planar format the description
/// carries the combined luma + chroma row count, which is inverted through
/// [`PixelFormat::image_height_from_combined`].
///
/// `shape` is the exception to that for one family of formats. A semi-planar
/// texture is as wide as the image's row *stride* -- the driver's pitch, which
/// the HAL allocates the texture at so the combined plane's rows and the
/// texture's rows are one grid -- and no part of the description says how much
/// of that row is image. So for `Nv12`, `Nv16` and `Nv24` the width is *taken*
/// from `shape` (index 1 in both the allocation and the addressing spelling)
/// and checked to fit, and omitting the shape is an error rather than a guess.
/// For every other format `shape` is unused here and stays the caller's own
/// consistency check, in the error class that caller's API promises.
///
/// The description is checked against the layout [`image_d3d11_layout`] would
/// have chosen for `format`, so a texture of the wrong DXGI format or the
/// wrong grid is refused. Two formats that share a layout exactly (a Grey
/// texture whose rows happen to be an NV12 combined plane height, say) are
/// indistinguishable by description alone and both answer; nothing in the
/// texture records which one the producer meant.
///
/// # Errors
///
/// [`Error::InvalidArgument`] for a null texture, a semi-planar row count no
/// image height produces, a semi-planar texture with no `shape`, or a
/// description that is not the layout `format` is allocated with.
/// [`Error::InvalidShape`] for a `shape` that is not a rank-2 semi-planar
/// shape or whose width does not fit the texture's rows -- the C export
/// classifies that as `InvalidShape`, which is what a caller who passed the
/// wrong `dims` needs to hear. Propagates the device error when there is no
/// process device.
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
    // `ID3D11Texture2D`; borrowing takes no reference and `cloned` AddRefs the
    // one this call drops on return.
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

// The fixtures are textures, which only the `static` backend can allocate:
// `D3d11TextureTensor` is `TensorStorage`'s and `dynamic` forwards every
// tensor call to the C ABI instead. The code under test is the same in both
// builds, and the `dynamic` lint lane still compiles it.
#[cfg(all(test, feature = "static"))]
mod tests {
    use super::*;
    use crate::d3d11::texture::D3d11TextureTensor;
    use crate::CpuAccess;
    use std::os::windows::io::AsRawHandle;

    /// The geometry helpers read the image dimensions off the texture
    /// description, so a constructor handed a raw texture never has to trust a
    /// caller's dimensions -- it checks them.
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

    /// A semi-planar texture is as wide as its row pitch, not as wide as the
    /// image, so its width is unreadable from the description alone: the shape
    /// supplies it. Both spellings of a semi-planar shape carry the width at
    /// index 1, so either answers.
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

    /// Without a shape there is no image width to report, so both helpers say
    /// so rather than answering the texture's padded width as if it were one.
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

    /// A width the texture cannot hold an even row of is refused, so a shape
    /// that does not belong to this texture cannot pass itself off as one.
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

    /// A texture whose description is not the layout the format is allocated
    /// with is refused, rather than answering a size read off the wrong grid.
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
        // A Grey allocation shape is rank 3, so this is the shape that is
        // wrong, not the texture.
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

        // A format with no layout at this texture grid at all: the error says
        // so, rather than naming an empty list of expected DXGI formats.
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
