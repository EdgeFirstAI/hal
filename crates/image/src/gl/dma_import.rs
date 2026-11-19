// SPDX-FileCopyrightText: Copyright 2025 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! DMA-BUF plane resolution for EGL image import.
//!
//! Resolves plane parameters (fd, pitch, offset) and builds the
//! `eglCreateImage` attribute list without calling EGL.

use drm_fourcc::DrmFourcc;
use edgefirst_egl::{self as egl, Attrib};
use edgefirst_tensor::{ColorEncoding, ColorRange, PixelFormat, PixelLayout, Tensor, TensorTrait};
use std::os::fd::AsRawFd;
use std::os::unix::io::RawFd;

use super::context::egl_ext;
use super::fourcc::pixel_format_to_drm;
use super::processor::DMA_IMPORT_OFFSET_ALIGN;
use crate::colorimetry::resolve_colorimetry;
use crate::Error;

/// Resolved DMA-BUF plane parameters for EGL image creation.
///
/// Holds the values that build the `eglCreateImage` attribute list. The
/// three NV12 import scenarios produce different values:
///
/// | Scenario             | plane0_fd | plane1.fd        | plane1.offset           |
/// |----------------------|-----------|------------------|-------------------------|
/// | True multiplane      | fd_a      | fd_b (different) | chroma plane_offset     |
/// | Same-fd multiplane   | fd_a_dup1 | fd_a_dup2        | chroma plane_offset     |
/// | Contiguous single-fd | fd_a      | fd_a (same)      | p0_offset + pitch × h   |
#[derive(Debug, Clone)]
pub(super) struct DmaImportAttrs {
    pub width: usize,
    pub height: usize,
    pub drm_fourcc: DrmFourcc,
    pub plane0_fd: RawFd,
    pub plane0_pitch: usize,
    pub plane0_offset: usize,
    /// Texels between this import's first column and the tensor's own first
    /// pixel. Nonzero only for a source rebased onto an aligned base: `width`
    /// is widened by it and `plane0_offset` walked back by `x_shift_px * bpp`.
    /// The tensor's pixel `(x, y)` is import texel `(x + x_shift_px, y)`.
    pub x_shift_px: u32,
    /// Second plane for NV12.
    pub plane1: Option<DmaPlane1Attrs>,
    pub is_yuv: bool,
    /// Resolved YUV matrix encoding for the EGL color-space hint.
    /// Only meaningful when `is_yuv`; ignored otherwise.
    pub yuv_encoding: ColorEncoding,
    /// Resolved YUV sample range for the EGL sample-range hint.
    /// Only meaningful when `is_yuv`; ignored otherwise.
    pub yuv_range: ColorRange,
}

/// Resolved attributes for the chroma (UV) plane.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct DmaPlane1Attrs {
    pub fd: RawFd,
    pub pitch: usize,
    pub offset: usize,
}

/// The largest byte offset at or below `offset` that is a multiple of
/// `align` and a whole number of `bpp`-byte pixels away from `offset`,
/// with that distance in pixels. Bases step by `lcm(align, bpp)`.
///
/// Returns `None` when `offset` is not a whole number of pixels, or when
/// `bpp` or `align` is zero.
pub(super) fn aligned_source_base(offset: usize, bpp: usize, align: usize) -> Option<(usize, u32)> {
    if bpp == 0 || align == 0 || !offset.is_multiple_of(bpp) {
        return None;
    }
    // lcm(align, bpp), computed as `align / gcd * bpp`.
    let mut a = align;
    let mut b = bpp;
    while b != 0 {
        let t = a % b;
        a = b;
        b = t;
    }
    let step = (align / a).checked_mul(bpp)?;
    let base = offset - offset % step;
    Some((base, u32::try_from((offset - base) / bpp).ok()?))
}

/// Whether a source rebased by `x_shift_px` pixels fits the pitch it imports
/// at: the widened row `(width_px + x_shift_px) * bpp` does not run past
/// `plane0_pitch`. Arithmetic overflow counts as not fitting.
pub(super) fn rebase_fits_pitch(
    width_px: usize,
    x_shift_px: u32,
    bpp: usize,
    plane0_pitch: usize,
) -> bool {
    width_px
        .checked_add(x_shift_px as usize)
        .and_then(|w| w.checked_mul(bpp))
        .is_some_and(|row| row <= plane0_pitch)
}

/// The bytes per pixel the aligned-base rebase uses for `src_fmt`, or `None`
/// for the formats it does not cover. Covers `Rgba`, `Bgra`, `Rgb` and
/// `Grey`.
fn rebase_bpp(src_fmt: PixelFormat) -> Option<usize> {
    matches!(
        src_fmt,
        PixelFormat::Rgba | PixelFormat::Bgra | PixelFormat::Rgb | PixelFormat::Grey
    )
    .then(|| src_fmt.channels())
}

/// The plane-0 channel count the import uses: 1 for NV12 and for planar
/// formats, `PixelFormat::channels()` otherwise.
fn import_plane0_channels(src_fmt: PixelFormat) -> usize {
    if src_fmt == PixelFormat::Nv12 || src_fmt.layout() == PixelLayout::Planar {
        1
    } else {
        src_fmt.channels()
    }
}

/// The tightly-packed plane-0 pitch a tensor with no stored row stride
/// imports at.
fn tight_plane0_pitch(src_fmt: PixelFormat, width_px: usize, channels: usize) -> usize {
    if src_fmt == PixelFormat::Nv12 {
        // The NV12 luma plane is 1 byte per pixel.
        width_px
    } else {
        width_px * channels
    }
}

/// The plane-0 offset a source import of this shape presents to EGL, and the
/// texel shift folded into the sampling rectangle.
///
/// * A format the rebase does not cover, or an offset that is 64-byte
///   aligned: returned unchanged with no shift.
/// * An unaligned offset the rebase can move: the aligned base below it and
///   the remainder in whole pixels.
/// * An unaligned offset that is not a whole number of pixels, or a pitch
///   with no room to widen into: returned unchanged with no shift.
pub(super) fn resolve_source_plane0(
    src_fmt: PixelFormat,
    width_px: usize,
    plane0_pitch: usize,
    plane_offset: usize,
) -> (usize, u32) {
    let Some(bpp) = rebase_bpp(src_fmt) else {
        return (plane_offset, 0);
    };
    if plane_offset.is_multiple_of(DMA_IMPORT_OFFSET_ALIGN) {
        return (plane_offset, 0);
    }
    aligned_source_base(plane_offset, bpp, DMA_IMPORT_OFFSET_ALIGN)
        .filter(|&(_, shift)| rebase_fits_pitch(width_px, shift, bpp, plane0_pitch))
        .unwrap_or((plane_offset, 0))
}

/// [`resolve_source_plane0`]'s offset for a whole tensor, resolving the
/// width, pitch and offset the way [`DmaImportAttrs::from_tensor`] does for a
/// source.
pub(super) fn resolved_source_plane0_offset(src: &Tensor<u8>, src_fmt: PixelFormat) -> usize {
    let width = src.width().unwrap_or(0);
    let pitch = src
        .effective_row_stride()
        .unwrap_or_else(|| tight_plane0_pitch(src_fmt, width, import_plane0_channels(src_fmt)));
    resolve_source_plane0(src_fmt, width, pitch, src.plane_offset().unwrap_or(0)).0
}

/// `NotSupported` unless an R8 render target's row `pitch` is a multiple of 4.
fn require_r8_pitch(pitch: usize, what: &str) -> Result<(), Error> {
    if pitch.is_multiple_of(4) {
        Ok(())
    } else {
        Err(Error::NotSupported(format!(
            "{what}: an R8 render target needs a row pitch divisible by 4, got {pitch}"
        )))
    }
}

/// The raw DMA-BUF fd of `img`; `NotImplemented` when it has none, naming `what`.
fn dma_fd(img: &Tensor<u8>, what: &str) -> Result<RawFd, Error> {
    img.dmabuf().map(|fd| fd.as_raw_fd()).map_err(|e| {
        Error::NotImplemented(format!(
            "{what} requires a DMA tensor, got {:?} ({e})",
            img.memory()
        ))
    })
}

impl DmaImportAttrs {
    /// Resolves the fd, pitch and offset values `eglCreateImage` needs from a
    /// tensor and its pixel format, without calling EGL. With `for_dst` a view
    /// resolves its parent's geometry at offset 0; a source resolves its own
    /// region.
    pub fn from_tensor(
        src: &Tensor<u8>,
        src_fmt: PixelFormat,
        for_dst: bool,
    ) -> Result<Self, Error> {
        // A destination view imports its parent's geometry; a source imports its own.
        let view_origin = if for_dst { src.view_origin() } else { None };
        let (src_w, src_h) = match view_origin {
            Some(vo) => (vo.parent_width, vo.parent_height),
            None => (
                src.width().ok_or(Error::NotAnImage)?,
                src.height().ok_or(Error::NotAnImage)?,
            ),
        };
        let src_channels = src_fmt.channels();

        // Resolve the tensor's colorimetry; missing axes are filled by the
        // HD/SD height heuristic.
        let cm = resolve_colorimetry(src.colorimetry(), src.height());
        let yuv_encoding = cm.encoding.unwrap_or(ColorEncoding::Bt709);
        let yuv_range = cm.range.unwrap_or(ColorRange::Limited);

        let (width, height, drm_fourcc, channels) = if src_fmt == PixelFormat::Nv12 {
            if !src_w.is_multiple_of(2) {
                return Err(Error::NotSupported(format!(
                    "EGLImage requires even width for {src_fmt} (4:2:0 chroma), got {src_w}"
                )));
            }
            // Plane 0 is full-resolution single-channel luma.
            (src_w, src_h, pixel_format_to_drm(src_fmt)?, 1usize)
        } else if src_fmt.layout() == PixelLayout::Planar {
            if !src_w.is_multiple_of(16) {
                return Err(Error::NotSupported(format!(
                    "EGLImage requires width divisible by 16 for {src_fmt}, got {src_w}"
                )));
            }
            match src_fmt {
                PixelFormat::PlanarRgb | PixelFormat::PlanarRgba => {
                    (src_w, src_h * src_fmt.channels(), DrmFourcc::R8, 1)
                }
                _ => {
                    return Err(Error::NotSupported(format!(
                        "Unsupported Planar format {src_fmt:?}"
                    )));
                }
            }
        } else {
            // Formats other than Rgba, Bgra and Rgb require a plane-0 pitch
            // that is a multiple of 4.
            let needs_width_mod_4 = !matches!(
                src_fmt,
                PixelFormat::Rgba | PixelFormat::Bgra | PixelFormat::Rgb
            );
            let pitch = match view_origin {
                Some(vo) => vo.parent_row_stride,
                None => src
                    .effective_row_stride()
                    .unwrap_or_else(|| tight_plane0_pitch(src_fmt, src_w, src_channels)),
            };
            if needs_width_mod_4 && !pitch.is_multiple_of(4) {
                return Err(Error::NotSupported(format!(
                    "EGLImage requires a row pitch divisible by 4 for {src_fmt}, got {pitch} \
                     at width {src_w}"
                )));
            }
            (src_w, src_h, pixel_format_to_drm(src_fmt)?, src_channels)
        };

        debug_assert_eq!(
            channels,
            import_plane0_channels(src_fmt),
            "{src_fmt:?}: the import's plane-0 channel count and the gate's must agree"
        );

        let fd = dma_fd(src, "OpenGL EGLImage")?;

        // Multiplane NV12: the UV plane's fd comes from the chroma tensor.
        let uv_fd = if src.is_multiplane() {
            let chroma = src.chroma().unwrap();
            Some(dma_fd(chroma, "OpenGL EGLImage chroma plane")?)
        } else {
            None
        };

        // A view uses its parent's row stride. A whole tensor uses its stored
        // stride, or the tightly-packed pitch when none is set.
        let plane0_pitch = match view_origin {
            Some(vo) => vo.parent_row_stride,
            None => src
                .effective_row_stride()
                .unwrap_or_else(|| tight_plane0_pitch(src_fmt, width, channels)),
        };

        // A view imports its parent at offset 0; a whole tensor uses its own offset.
        let plane0_offset = if view_origin.is_some() {
            0
        } else {
            src.plane_offset().unwrap_or(0)
        };

        // A source imports from a 64-byte-aligned base and carries the remainder
        // as a texel shift. A destination keeps its offset.
        let (plane0_offset, x_shift_px) = if for_dst {
            (plane0_offset, 0)
        } else {
            resolve_source_plane0(src_fmt, width, plane0_pitch, plane0_offset)
        };
        let width = width + x_shift_px as usize;

        // NV12 carries a second plane of interleaved CbCr: ceil(H/2) rows of W bytes.
        let plane1 = if src_fmt == PixelFormat::Nv12 {
            let chroma_h = height.div_ceil(2);
            let default_chroma_pitch = plane0_pitch;
            let (plane1_fd, uv_offset) = if let Some(chroma_fd) = uv_fd {
                // Multiplane: the UV plane is a separate DMA-BUF at the chroma
                // tensor's plane offset, or 0.
                let chroma_offset = src.chroma().and_then(|c| c.plane_offset()).unwrap_or(0);
                (chroma_fd, chroma_offset)
            } else {
                // Contiguous: the UV plane follows the full-height Y plane in
                // the same buffer.
                (fd, plane0_offset + plane0_pitch * height)
            };
            let plane1_pitch = if let Some(chroma) = src.chroma() {
                // Multiplane: the chroma tensor's row stride if set, else the luma pitch.
                chroma
                    .effective_row_stride()
                    .unwrap_or(default_chroma_pitch)
            } else {
                default_chroma_pitch
            };
            // Validate that the chroma offset plus the chroma data fits within
            // the chroma fd's buffer.
            let chroma_data = plane1_pitch.saturating_mul(chroma_h);
            let required = uv_offset.saturating_add(chroma_data);
            let buf_size = {
                let mut stat: libc::stat = unsafe { std::mem::zeroed() };
                if unsafe { libc::fstat(plane1_fd, &mut stat) } == 0 && stat.st_size > 0 {
                    Some(stat.st_size as usize)
                } else {
                    None
                }
            };
            if let Some(sz) = buf_size {
                if required > sz {
                    return Err(Error::InvalidShape(format!(
                        "{src_fmt} chroma plane offset {uv_offset} + required {chroma_data} bytes \
                         exceeds chroma fd buffer size {sz} — the chroma PlaneDescriptor \
                         likely references the wrong fd (e.g. Y-only buffer with the \
                         vmeta global offset instead of the UV buffer's own fd)",
                    )));
                }
            }

            Some(DmaPlane1Attrs {
                fd: plane1_fd,
                pitch: plane1_pitch,
                offset: uv_offset,
            })
        } else {
            None
        };

        Ok(DmaImportAttrs {
            width,
            height,
            drm_fourcc,
            plane0_fd: fd,
            plane0_pitch,
            plane0_offset,
            x_shift_px,
            plane1,
            is_yuv: src_fmt.is_yuv(),
            yuv_encoding,
            yuv_range,
        })
    }

    /// Builds `DmaImportAttrs` for importing an NV12/NV16/NV24 semi-planar
    /// buffer, or a packed `Rgb` or `Yuyv` buffer, as a single-plane R8
    /// EGLImage.
    ///
    /// The luma and chroma rows are imported stacked as one R8 plane. A packed
    /// buffer imports as its `height` rows of `effective_row_stride` bytes. The
    /// returned attributes carry:
    ///   * `width`  = `effective_row_stride()`
    ///   * `height` = combined (luma + chroma) height
    ///   * `drm_fourcc` = `DrmFourcc::R8`
    ///   * `plane1` = `None`
    ///
    /// # Errors
    ///
    /// Returns `Error::NotSupported` for any other format and for a multiplane
    /// tensor.
    pub fn from_tensor_nv_r8(src: &Tensor<u8>, src_fmt: PixelFormat) -> Result<Self, Error> {
        let src_w = src.width().ok_or(Error::NotAnImage)?;
        let src_h = src.height().ok_or(Error::NotAnImage)?;

        if !matches!(
            src_fmt,
            PixelFormat::Nv12
                | PixelFormat::Nv16
                | PixelFormat::Nv24
                | PixelFormat::Rgb
                | PixelFormat::Yuyv
        ) {
            return Err(Error::NotSupported(format!(
                "from_tensor_nv_r8 only supports NV12/NV16/NV24/Rgb/Yuyv, got {src_fmt:?}"
            )));
        }

        if src.is_multiplane() {
            return Err(Error::NotSupported(
                "from_tensor_nv_r8: multiplane tensor not supported".into(),
            ));
        }

        // The R8 texture is as wide as the effective row stride, padding bytes included.
        let tex_width = src.effective_row_stride().unwrap_or(src_w);

        // Combined (luma + chroma) buffer height, matching `PixelFormat::allocation_shape`:
        //   NV12 (4:2:0): H luma + H/2 chroma          = H + ⌈H/2⌉
        //   NV16 (4:2:2): H luma + H chroma (W bytes/row) = 2H
        //   NV24 (4:4:4): H luma + 2H chroma (2W bytes/row laid out as 2H rows of W) = 3H
        // A packed buffer keeps its own height.
        let combined_h = if matches!(src_fmt, PixelFormat::Rgb | PixelFormat::Yuyv) {
            src_h
        } else {
            src_fmt.combined_plane_height(src_h).ok_or_else(|| {
                Error::NotImplemented(format!("R8 import: {src_fmt:?} is not semi-planar"))
            })?
        };

        let fd = dma_fd(src, "OpenGL R8 import")?;
        let plane0_offset = src.plane_offset().unwrap_or(0);

        Ok(DmaImportAttrs {
            width: tex_width,
            height: combined_h,
            drm_fourcc: DrmFourcc::R8,
            plane0_fd: fd,
            plane0_pitch: tex_width,
            plane0_offset,
            x_shift_px: 0,
            plane1: None,
            is_yuv: false,
            // The shader applies the YUV→RGB matrix; these hints are unused.
            yuv_encoding: ColorEncoding::Bt709,
            yuv_range: ColorRange::Limited,
        })
    }

    /// Import attributes for the chroma plane of a single-plane NV12/NV16/NV24
    /// tensor as its own `GR88` image: one texel per chroma sample, with the
    /// U byte in R and the V byte in G. The plane starts `row_stride * height`
    /// bytes after the luma plane and keeps the luma pitch per R8 row, which
    /// is `uv_rows_per_luma` rows per chroma row (two for NV24, whose CbCr
    /// row is `2W` bytes).
    pub fn from_tensor_nv_chroma_gr88(
        src: &Tensor<u8>,
        src_fmt: PixelFormat,
    ) -> Result<Self, Error> {
        let src_w = src.width().ok_or(Error::NotAnImage)?;
        let src_h = src.height().ok_or(Error::NotAnImage)?;
        let layout = src_fmt.chroma_layout().ok_or_else(|| {
            Error::NotSupported(format!("NV chroma import: {src_fmt:?} is not semi-planar"))
        })?;
        if src.is_multiplane() {
            return Err(Error::NotSupported(
                "NV chroma import: multiplane tensor not supported".into(),
            ));
        }
        let tex_width = src.effective_row_stride().unwrap_or(src_w);
        let fd = dma_fd(src, "NV chroma import")?;
        let (sx, sy) = (layout.shift_x, layout.shift_y);
        Ok(DmaImportAttrs {
            width: src_w.div_ceil(1 << sx),
            height: src_h.div_ceil(1 << sy),
            drm_fourcc: DrmFourcc::Gr88,
            plane0_fd: fd,
            plane0_pitch: tex_width * layout.uv_rows_per_luma,
            plane0_offset: src.plane_offset().unwrap_or(0) + tex_width * src_h,
            x_shift_px: 0,
            plane1: None,
            is_yuv: false,
            // The shader applies the YUV->RGB matrix; GR88 carries raw bytes.
            yuv_encoding: ColorEncoding::Bt709,
            yuv_range: ColorRange::Limited,
        })
    }

    /// Import attributes for a single-plane NV12 destination as one R8 render
    /// target: `width` = the row pitch, `height` = `h + h / 2` (luma rows,
    /// then chroma rows), at the tensor's plane offset.
    ///
    /// # Errors
    ///
    /// `NotSupported` for a multiplane tensor or a pitch that is not a
    /// multiple of 4; `NotImplemented` for a tensor without a DMA-BUF.
    pub fn from_tensor_nv12_dst_r8(img: &Tensor<u8>) -> Result<Self, Error> {
        let attrs = Self::from_tensor_nv_r8(img, PixelFormat::Nv12)?;
        require_r8_pitch(attrs.plane0_pitch, "NV12 destination")?;
        Ok(attrs)
    }

    /// Import attributes for a multiplane NV12 destination as two R8 render
    /// targets, each at its own buffer, pitch and offset and as wide as its
    /// pitch: the luma plane (`luma_pitch` × `h`) and the interleaved chroma
    /// plane (`chroma_pitch` × `h / 2`).
    ///
    /// # Errors
    ///
    /// `NotSupported` for a single-plane tensor or a plane pitch that is not
    /// a multiple of 4; `NotImplemented` for a plane without a DMA-BUF.
    pub fn from_tensor_nv12_dst_split(img: &Tensor<u8>) -> Result<(Self, Self), Error> {
        let h = img.height().ok_or(Error::NotAnImage)?;
        let chroma = img.chroma().ok_or_else(|| {
            Error::NotSupported(
                "NV12 destination split import: the chroma plane follows the luma plane".into(),
            )
        })?;
        let (luma_pitch, chroma_pitch) = crate::cpu::semi_planar_strides(img);
        require_r8_pitch(luma_pitch, "NV12 destination luma plane")?;
        require_r8_pitch(chroma_pitch, "NV12 destination chroma plane")?;
        let luma = Self::r8_render_target(
            dma_fd(img, "NV12 destination luma plane")?,
            h,
            luma_pitch,
            img.plane_offset().unwrap_or(0),
        );
        let chroma = Self::r8_render_target(
            dma_fd(chroma, "NV12 destination chroma plane")?,
            h / 2,
            chroma_pitch,
            chroma.plane_offset().unwrap_or(0),
        );
        Ok((luma, chroma))
    }

    /// Single-plane R8 attributes over `fd`, `pitch` texels wide.
    fn r8_render_target(fd: RawFd, height: usize, pitch: usize, offset: usize) -> Self {
        DmaImportAttrs {
            width: pitch,
            height,
            drm_fourcc: DrmFourcc::R8,
            plane0_fd: fd,
            plane0_pitch: pitch,
            plane0_offset: offset,
            x_shift_px: 0,
            plane1: None,
            is_yuv: false,
            yuv_encoding: ColorEncoding::Bt709,
            yuv_range: ColorRange::Limited,
        }
    }

    /// Build the EGL attribute list for `eglCreateImage`.
    pub fn to_egl_attribs(&self) -> Vec<Attrib> {
        let mut attrs = vec![
            egl_ext::LINUX_DRM_FOURCC as Attrib,
            self.drm_fourcc as Attrib,
            edgefirst_egl::WIDTH as Attrib,
            self.width as Attrib,
            edgefirst_egl::HEIGHT as Attrib,
            self.height as Attrib,
            egl_ext::DMA_BUF_PLANE0_PITCH as Attrib,
            self.plane0_pitch as Attrib,
            egl_ext::DMA_BUF_PLANE0_OFFSET as Attrib,
            self.plane0_offset as Attrib,
            egl_ext::DMA_BUF_PLANE0_FD as Attrib,
            self.plane0_fd as Attrib,
            egl::IMAGE_PRESERVED as Attrib,
            egl::TRUE as Attrib,
        ];

        if let Some(ref p1) = self.plane1 {
            attrs.extend_from_slice(&[
                egl_ext::DMA_BUF_PLANE1_FD as Attrib,
                p1.fd as Attrib,
                egl_ext::DMA_BUF_PLANE1_OFFSET as Attrib,
                p1.offset as Attrib,
                egl_ext::DMA_BUF_PLANE1_PITCH as Attrib,
                p1.pitch as Attrib,
            ]);
        }

        if self.is_yuv {
            // Color-space and sample-range hints from the resolved colorimetry.
            let cs_hint = match self.yuv_encoding {
                ColorEncoding::Bt601 => egl_ext::ITU_REC601,
                ColorEncoding::Bt709 => egl_ext::ITU_REC709,
                ColorEncoding::Bt2020 => egl_ext::ITU_REC2020,
                // Other encodings map to BT.709.
                _ => egl_ext::ITU_REC709,
            };
            let range_hint = match self.yuv_range {
                ColorRange::Full => egl_ext::YUV_FULL_RANGE,
                ColorRange::Limited => egl_ext::YUV_NARROW_RANGE,
                // Other ranges map to narrow range.
                _ => egl_ext::YUV_NARROW_RANGE,
            };
            attrs.extend_from_slice(&[
                egl_ext::YUV_COLOR_SPACE_HINT as Attrib,
                cs_hint as Attrib,
                egl_ext::SAMPLE_RANGE_HINT as Attrib,
                range_hint as Attrib,
            ]);
        }

        attrs.push(edgefirst_egl::NONE as Attrib);
        attrs
    }
}

/// Helper to extract the value for a given EGL attribute key from an attribute list.
#[cfg(test)]
pub(super) fn egl_attrib_value(attrs: &[Attrib], key: u32) -> Option<Attrib> {
    let mut i = 0;
    while i + 1 < attrs.len() {
        if attrs[i] == key as Attrib {
            return Some(attrs[i + 1]);
        }
        i += 2;
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    // ─── DmaImportAttrs::from_tensor tests ───────────────────────────
    // These require real DMA-BUF allocations; they skip without a heap unless HAL_TEST_REQUIRE_DMA=1.

    #[cfg(any(target_os = "linux", target_os = "macos", target_os = "windows"))]
    use crate::test_support::is_dma_available;
    #[cfg(any(target_os = "linux", target_os = "macos", target_os = "windows"))]
    use edgefirst_tensor::{PlaneDescriptor, TensorMemory};

    /// Reports a skip and returns `false` when this host has no usable DMA heap.
    #[cfg(any(target_os = "linux", target_os = "macos", target_os = "windows"))]
    fn dma_heap_or_skip(what: &str) -> bool {
        if is_dma_available() {
            return true;
        }
        crate::test_support::report_skip(&format!(
            "{}: {what}: DMA not available",
            crate::test_support::current_test()
        ));
        false
    }

    /// A failed allocation's reason, printed without the quotes `String`'s `Debug` adds.
    #[cfg(any(target_os = "linux", target_os = "macos", target_os = "windows"))]
    struct Failure(String);

    #[cfg(any(target_os = "linux", target_os = "macos", target_os = "windows"))]
    impl std::fmt::Debug for Failure {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            f.write_str(&self.0)
        }
    }

    /// Treats a DMA request that fell back to other memory as a failed allocation.
    #[cfg(any(target_os = "linux", target_os = "macos", target_os = "windows"))]
    fn dma_only(
        result: Result<Tensor<u8>, edgefirst_tensor::Error>,
    ) -> Result<Tensor<u8>, Failure> {
        match result {
            Ok(t) if t.memory() == TensorMemory::DmaBuf => Ok(t),
            Ok(t) => Err(Failure(format!("fell back to {:?}", t.memory()))),
            Err(e) => Err(Failure(format!("{e:?}"))),
        }
    }

    /// A DMA tensor of `bytes`, or `None` (warned) when the host has no usable DMA heap.
    #[cfg(any(target_os = "linux", target_os = "macos", target_os = "windows"))]
    fn alloc_dma(bytes: usize, name: &str) -> Option<Tensor<u8>> {
        if !dma_heap_or_skip(name) {
            return None;
        }
        crate::test_support::dma_allocated(
            dma_only(Tensor::<u8>::new(
                &[bytes],
                Some(TensorMemory::DmaBuf),
                Some(name),
            )),
            name,
        )
    }

    /// Helper: import an NV12 image via the public ImageProcessor API and
    /// return the resulting tensor for inspection.
    #[cfg(any(target_os = "linux", target_os = "macos", target_os = "windows"))]
    fn import_nv12(
        luma_pd: PlaneDescriptor,
        chroma_pd: Option<PlaneDescriptor>,
        width: usize,
        height: usize,
    ) -> Result<edgefirst_tensor::TensorDyn, crate::Error> {
        let proc = crate::ImageProcessor::new()?;
        proc.import_image(
            luma_pd,
            chroma_pd,
            width,
            height,
            PixelFormat::Nv12,
            edgefirst_tensor::DType::U8,
            None,
        )
    }

    /// A source view imports its own region; a destination view imports its parent.
    #[test]
    #[cfg(any(target_os = "linux", target_os = "macos", target_os = "windows"))]
    fn from_tensor_source_view_imports_own_region_dst_imports_parent() {
        if !is_dma_available() {
            crate::test_support::report_skip("from_tensor_source_view... - DMA not available");
            return;
        }
        let Some(parent) = crate::test_support::dma_allocated(
            Tensor::<u8>::image_with_stride(
                64,
                64,
                PixelFormat::Rgba,
                320,
                Some(TensorMemory::DmaBuf),
                edgefirst_tensor::CpuAccess::ReadWrite,
            ),
            "64x64 RGBA image_with_stride",
        ) else {
            return;
        };
        let view = parent
            .view(edgefirst_tensor::Region::new(8, 8, 32, 16))
            .unwrap();

        // Source: the view's own region, rebased from byte 2592 to base 2560
        // with an 8-pixel shift.
        let s = DmaImportAttrs::from_tensor(&view, PixelFormat::Rgba, false).unwrap();
        assert_eq!(
            (s.width, s.height),
            (32 + 8, 16),
            "source view imports its own dimensions, widened by the shift"
        );
        assert_eq!(s.x_shift_px, 8, "32 bytes of remainder is 8 RGBA pixels");
        assert_eq!(
            s.plane0_offset,
            8 * 320,
            "source view imports at the aligned base below its own byte offset"
        );
        assert_eq!(
            s.plane0_offset + s.x_shift_px as usize * 4,
            8 * 320 + 8 * 4,
            "and the shift closes that base back to the view's own offset"
        );
        assert_eq!(s.plane0_pitch, 320, "source view keeps the parent pitch");

        // Destination: imports the parent.
        let d = DmaImportAttrs::from_tensor(&view, PixelFormat::Rgba, true).unwrap();
        assert_eq!(
            (d.width, d.height),
            (64, 64),
            "dest view imports the parent dimensions"
        );
        assert_eq!(
            d.plane0_offset, 0,
            "dest view imports the parent at offset 0"
        );
        assert_eq!(d.plane0_pitch, 320, "dest view imports at the parent pitch");
    }

    /// True multiplane NV12 resolves two different fds with zero offsets.
    #[test]
    #[cfg(any(target_os = "linux", target_os = "macos", target_os = "windows"))]
    fn test_nv12_true_multiplane_attrs() {
        let width: usize = 1920;
        let height: usize = 1088;
        let stride: usize = 1920;
        let luma_bytes = stride * height;
        let chroma_bytes = stride * height.div_ceil(2);

        let Some(luma_buf) = alloc_dma(luma_bytes, "luma_buf") else {
            return;
        };
        let Some(chroma_buf) = alloc_dma(chroma_bytes, "chroma_buf") else {
            return;
        };

        let luma_fd = luma_buf.dmabuf().unwrap();
        let chroma_fd = chroma_buf.dmabuf().unwrap();
        let luma_pd = PlaneDescriptor::new(luma_fd).unwrap().with_stride(stride);
        let chroma_pd = PlaneDescriptor::new(chroma_fd).unwrap().with_stride(stride);

        let tensor = import_nv12(luma_pd, Some(chroma_pd), width, height).unwrap();
        let tensor_u8 = tensor.as_u8().unwrap();

        let attrs = DmaImportAttrs::from_tensor(tensor_u8, PixelFormat::Nv12, false).unwrap();

        let p1 = attrs.plane1.as_ref().expect("NV12 must have plane1");
        assert_ne!(
            attrs.plane0_fd, p1.fd,
            "true multiplane: plane0_fd and plane1_fd must be different"
        );

        assert_eq!(attrs.plane0_offset, 0, "luma offset must be 0");
        assert_eq!(p1.offset, 0, "chroma offset must be 0 for true multiplane");

        assert_eq!(attrs.plane0_pitch, stride);
        assert_eq!(p1.pitch, stride);

        assert_eq!(attrs.width, width);
        assert_eq!(attrs.height, height);
        assert!(attrs.is_yuv);
        assert_eq!(attrs.drm_fourcc, DrmFourcc::Nv12);
    }

    /// Same-fd multiplane NV12 resolves two dup'd fds with the chroma plane at its explicit offset.
    #[test]
    #[cfg(any(target_os = "linux", target_os = "macos", target_os = "windows"))]
    fn test_nv12_same_fd_multiplane_attrs() {
        let width: usize = 1920;
        let height: usize = 1088;
        let stride: usize = 1920;
        let luma_size = stride * height;
        let chroma_size = stride * height.div_ceil(2);
        let total_bytes = luma_size + chroma_size;

        let Some(buf) = alloc_dma(total_bytes, "shared_buf") else {
            return;
        };

        let fd = buf.dmabuf().unwrap();
        // Both descriptors dup the same fd.
        let luma_pd = PlaneDescriptor::new(fd)
            .unwrap()
            .with_stride(stride)
            .with_offset(0);
        let chroma_pd = PlaneDescriptor::new(fd)
            .unwrap()
            .with_stride(stride)
            .with_offset(luma_size);

        let tensor = import_nv12(luma_pd, Some(chroma_pd), width, height).unwrap();
        let tensor_u8 = tensor.as_u8().unwrap();

        let attrs = DmaImportAttrs::from_tensor(tensor_u8, PixelFormat::Nv12, false).unwrap();
        let p1 = attrs.plane1.as_ref().expect("NV12 must have plane1");

        assert_ne!(
            attrs.plane0_fd, p1.fd,
            "same-fd multiplane: raw fds must differ (each is a dup)"
        );

        assert_eq!(attrs.plane0_offset, 0);
        assert_eq!(
            p1.offset, luma_size,
            "chroma offset must be luma_size for same-fd multiplane"
        );

        assert_eq!(attrs.plane0_pitch, stride);
        assert_eq!(p1.pitch, stride);
        assert_eq!(attrs.width, width);
        assert_eq!(attrs.height, height);
    }

    /// An even NV12 width that is not 4-aligned imports; an odd width is rejected.
    #[test]
    #[cfg(any(target_os = "linux", target_os = "macos", target_os = "windows"))]
    fn test_nv12_even_width_relaxed_gate() {
        let (height, stride) = (64usize, 128usize);
        let total = stride * height + stride * height.div_ceil(2);
        let Some(buf) = alloc_dma(total, "nv12_even_w") else {
            return;
        };

        let luma_pd = PlaneDescriptor::new(buf.dmabuf().unwrap())
            .unwrap()
            .with_stride(stride);
        let t66 = import_nv12(luma_pd, None, 66, height).unwrap();
        let attrs = DmaImportAttrs::from_tensor(t66.as_u8().unwrap(), PixelFormat::Nv12, false)
            .expect("even-but-not-4-aligned NV12 width must import after the %4→%2 relaxation");
        assert_eq!(attrs.width, 66);

        let luma_pd2 = PlaneDescriptor::new(buf.dmabuf().unwrap())
            .unwrap()
            .with_stride(stride);
        if let Ok(t65) = import_nv12(luma_pd2, None, 65, height) {
            assert!(
                DmaImportAttrs::from_tensor(t65.as_u8().unwrap(), PixelFormat::Nv12, false)
                    .is_err(),
                "odd-width NV12 must still be rejected by the even-width gate"
            );
        }
    }

    /// Contiguous single-fd NV12 resolves plane 1 to the same raw fd at `plane0_pitch * height`.
    #[test]
    #[cfg(any(target_os = "linux", target_os = "macos", target_os = "windows"))]
    fn test_nv12_contiguous_single_fd_attrs() {
        let width: usize = 1920;
        let height: usize = 1088;
        let stride: usize = 1920;
        let total_h = height * 3 / 2; // NV12: Y + UV/2
        let total_bytes = stride * total_h;

        let Some(buf) = alloc_dma(total_bytes, "contiguous_buf") else {
            return;
        };

        let fd = buf.dmabuf().unwrap();
        let luma_pd = PlaneDescriptor::new(fd).unwrap().with_stride(stride);

        let tensor = import_nv12(luma_pd, None, width, height).unwrap();
        let tensor_u8 = tensor.as_u8().unwrap();

        let attrs = DmaImportAttrs::from_tensor(tensor_u8, PixelFormat::Nv12, false).unwrap();
        let p1 = attrs.plane1.as_ref().expect("NV12 must have plane1");

        assert_eq!(
            attrs.plane0_fd, p1.fd,
            "contiguous: plane0_fd and plane1_fd must be the same raw fd"
        );

        let expected_uv_offset = stride * height;
        assert_eq!(attrs.plane0_offset, 0);
        assert_eq!(
            p1.offset, expected_uv_offset,
            "contiguous UV offset must be stride * height"
        );

        assert_eq!(attrs.plane0_pitch, stride);
        assert_eq!(p1.pitch, stride);
    }

    /// Contiguous single-fd NV12 with a padded stride: the UV offset uses the stride.
    #[test]
    #[cfg(any(target_os = "linux", target_os = "macos", target_os = "windows"))]
    fn test_nv12_contiguous_padded_stride_attrs() {
        let width: usize = 1920;
        let height: usize = 1080;
        let stride: usize = 2048; // padded to 2048-byte alignment
        let total_h = height * 3 / 2;
        let total_bytes = stride * total_h;

        let Some(buf) = alloc_dma(total_bytes, "padded_buf") else {
            return;
        };

        let fd = buf.dmabuf().unwrap();
        let luma_pd = PlaneDescriptor::new(fd).unwrap().with_stride(stride);

        let tensor = import_nv12(luma_pd, None, width, height).unwrap();
        let tensor_u8 = tensor.as_u8().unwrap();

        let attrs = DmaImportAttrs::from_tensor(tensor_u8, PixelFormat::Nv12, false).unwrap();
        let p1 = attrs.plane1.as_ref().unwrap();

        let expected_uv_offset = stride * height;
        assert_eq!(
            p1.offset, expected_uv_offset,
            "contiguous padded: UV offset must use stride ({stride}), not width ({width})"
        );
        assert_eq!(attrs.plane0_pitch, stride);
        assert_eq!(p1.pitch, stride, "contiguous: UV pitch must match Y pitch");
    }

    /// Multiplane with padded strides: each plane has its own explicit stride.
    #[test]
    #[cfg(any(target_os = "linux", target_os = "macos", target_os = "windows"))]
    fn test_nv12_multiplane_padded_strides_attrs() {
        let width: usize = 1920;
        let height: usize = 1080;
        let luma_stride: usize = 2048;
        let chroma_stride: usize = 2048;
        let luma_bytes = luma_stride * height;
        let chroma_bytes = chroma_stride * height.div_ceil(2);

        let Some(luma_buf) = alloc_dma(luma_bytes, "luma_padded") else {
            return;
        };
        let Some(chroma_buf) = alloc_dma(chroma_bytes, "chroma_padded") else {
            return;
        };

        let luma_fd = luma_buf.dmabuf().unwrap();
        let chroma_fd = chroma_buf.dmabuf().unwrap();
        let luma_pd = PlaneDescriptor::new(luma_fd)
            .unwrap()
            .with_stride(luma_stride);
        let chroma_pd = PlaneDescriptor::new(chroma_fd)
            .unwrap()
            .with_stride(chroma_stride);

        let tensor = import_nv12(luma_pd, Some(chroma_pd), width, height).unwrap();
        let tensor_u8 = tensor.as_u8().unwrap();

        let attrs = DmaImportAttrs::from_tensor(tensor_u8, PixelFormat::Nv12, false).unwrap();
        let p1 = attrs.plane1.as_ref().unwrap();

        assert_eq!(attrs.plane0_pitch, luma_stride);
        assert_eq!(p1.pitch, chroma_stride);
        assert_eq!(attrs.plane0_offset, 0);
        assert_eq!(p1.offset, 0);
    }

    /// Same-fd multiplane NV12 with a non-zero luma offset.
    #[test]
    #[cfg(any(target_os = "linux", target_os = "macos", target_os = "windows"))]
    fn test_nv12_same_fd_nonzero_luma_offset() {
        let width: usize = 640;
        let height: usize = 480;
        let stride: usize = 640;
        let luma_offset: usize = 4096; // metadata header before luma
        let luma_size = stride * height;
        let chroma_offset = luma_offset + luma_size;
        let total_bytes = chroma_offset + stride * height.div_ceil(2);

        let Some(buf) = alloc_dma(total_bytes, "offset_buf") else {
            return;
        };

        let fd = buf.dmabuf().unwrap();
        let luma_pd = PlaneDescriptor::new(fd)
            .unwrap()
            .with_stride(stride)
            .with_offset(luma_offset);
        let chroma_pd = PlaneDescriptor::new(fd)
            .unwrap()
            .with_stride(stride)
            .with_offset(chroma_offset);

        let tensor = import_nv12(luma_pd, Some(chroma_pd), width, height).unwrap();
        let tensor_u8 = tensor.as_u8().unwrap();

        let attrs = DmaImportAttrs::from_tensor(tensor_u8, PixelFormat::Nv12, false).unwrap();
        let p1 = attrs.plane1.as_ref().unwrap();

        assert_eq!(attrs.plane0_offset, luma_offset);
        assert_eq!(p1.offset, chroma_offset);
    }

    /// A chroma offset at or past the end of the chroma fd's buffer is rejected.
    #[test]
    #[cfg(any(target_os = "linux", target_os = "macos", target_os = "windows"))]
    fn test_nv12_chroma_offset_exceeds_buffer() {
        let width: usize = 1920;
        let height: usize = 1088;
        let stride: usize = 1920;
        let y_size = stride * height;
        let chroma_h = height.div_ceil(2);
        let _chroma_size = stride * chroma_h;

        // A buffer sized for the Y plane only.
        let Some(y_buf) = alloc_dma(y_size, "y_only_buf") else {
            return;
        };

        let fd = y_buf.dmabuf().unwrap();
        let luma_pd = PlaneDescriptor::new(fd)
            .unwrap()
            .with_stride(stride)
            .with_offset(0);
        // The chroma descriptor uses the Y fd with offset `y_size`.
        let chroma_pd = PlaneDescriptor::new(fd)
            .unwrap()
            .with_stride(stride)
            .with_offset(y_size);

        let result = import_nv12(luma_pd, Some(chroma_pd), width, height);
        let tensor = result.unwrap();
        let tensor_u8 = tensor.as_u8().unwrap();

        let err = DmaImportAttrs::from_tensor(tensor_u8, PixelFormat::Nv12, false);
        assert!(
            err.is_err(),
            "from_tensor must reject chroma offset {y_size} on a {y_size}-byte buffer"
        );
        let msg = err.unwrap_err().to_string();
        assert!(
            msg.contains("exceeds chroma fd buffer size"),
            "error must mention buffer size, got: {msg}"
        );
    }

    // ─── to_egl_attribs tests ────────────────────────────────────────
    // Attribute list serialization with synthetic values.

    /// Verify EGL attribute list for true multiplane NV12.
    #[test]
    fn test_egl_attribs_true_multiplane() {
        let attrs = DmaImportAttrs {
            width: 1920,
            height: 1080,
            drm_fourcc: DrmFourcc::Nv12,
            plane0_fd: 10,
            plane0_pitch: 1920,
            plane0_offset: 0,
            x_shift_px: 0,
            plane1: Some(DmaPlane1Attrs {
                fd: 11,
                pitch: 1920,
                offset: 0,
            }),
            is_yuv: true,
            yuv_encoding: ColorEncoding::Bt709,
            yuv_range: ColorRange::Limited,
        };

        let egl = attrs.to_egl_attribs();

        assert_eq!(egl_attrib_value(&egl, egl_ext::DMA_BUF_PLANE0_FD), Some(10));
        assert_eq!(
            egl_attrib_value(&egl, egl_ext::DMA_BUF_PLANE0_PITCH),
            Some(1920)
        );
        assert_eq!(
            egl_attrib_value(&egl, egl_ext::DMA_BUF_PLANE0_OFFSET),
            Some(0)
        );

        assert_eq!(egl_attrib_value(&egl, egl_ext::DMA_BUF_PLANE1_FD), Some(11));
        assert_eq!(
            egl_attrib_value(&egl, egl_ext::DMA_BUF_PLANE1_PITCH),
            Some(1920)
        );
        assert_eq!(
            egl_attrib_value(&egl, egl_ext::DMA_BUF_PLANE1_OFFSET),
            Some(0)
        );

        assert_eq!(
            egl_attrib_value(&egl, egl_ext::YUV_COLOR_SPACE_HINT),
            Some(egl_ext::ITU_REC709 as Attrib)
        );
        assert_eq!(
            egl_attrib_value(&egl, egl_ext::SAMPLE_RANGE_HINT),
            Some(egl_ext::YUV_NARROW_RANGE as Attrib)
        );

        assert_eq!(*egl.last().unwrap(), edgefirst_egl::NONE as Attrib);
    }

    /// Verify EGL attribute list for same-fd multiplane NV12.
    #[test]
    fn test_egl_attribs_same_fd_multiplane() {
        let luma_size: usize = 1920 * 1080;
        let attrs = DmaImportAttrs {
            width: 1920,
            height: 1080,
            drm_fourcc: DrmFourcc::Nv12,
            plane0_fd: 10,
            plane0_pitch: 1920,
            plane0_offset: 0,
            x_shift_px: 0,
            plane1: Some(DmaPlane1Attrs {
                fd: 12, // different raw fd, same underlying buffer
                pitch: 1920,
                offset: luma_size,
            }),
            is_yuv: true,
            yuv_encoding: ColorEncoding::Bt709,
            yuv_range: ColorRange::Limited,
        };

        let egl = attrs.to_egl_attribs();

        assert_eq!(egl_attrib_value(&egl, egl_ext::DMA_BUF_PLANE0_FD), Some(10));
        assert_eq!(egl_attrib_value(&egl, egl_ext::DMA_BUF_PLANE1_FD), Some(12));
        assert_eq!(
            egl_attrib_value(&egl, egl_ext::DMA_BUF_PLANE1_OFFSET),
            Some(luma_size as Attrib)
        );
    }

    /// Verify EGL attribute list for contiguous single-fd NV12.
    #[test]
    fn test_egl_attribs_contiguous() {
        let height = 1080usize;
        let pitch = 1920usize;
        let attrs = DmaImportAttrs {
            width: 1920,
            height,
            drm_fourcc: DrmFourcc::Nv12,
            plane0_fd: 10,
            plane0_pitch: pitch,
            plane0_offset: 0,
            x_shift_px: 0,
            plane1: Some(DmaPlane1Attrs {
                fd: 10, // same raw fd
                pitch,
                offset: pitch * height,
            }),
            is_yuv: true,
            yuv_encoding: ColorEncoding::Bt709,
            yuv_range: ColorRange::Limited,
        };

        let egl = attrs.to_egl_attribs();

        let p0_fd = egl_attrib_value(&egl, egl_ext::DMA_BUF_PLANE0_FD).unwrap();
        let p1_fd = egl_attrib_value(&egl, egl_ext::DMA_BUF_PLANE1_FD).unwrap();
        assert_eq!(
            p0_fd, p1_fd,
            "contiguous: PLANE0_FD and PLANE1_FD must be the same raw fd"
        );

        assert_eq!(
            egl_attrib_value(&egl, egl_ext::DMA_BUF_PLANE1_OFFSET),
            Some((pitch * height) as Attrib)
        );
    }

    /// Verify that non-NV12 formats produce no PLANE1 attributes.
    #[test]
    fn test_egl_attribs_rgba_no_plane1() {
        let attrs = DmaImportAttrs {
            width: 640,
            height: 480,
            drm_fourcc: DrmFourcc::Abgr8888,
            plane0_fd: 10,
            plane0_pitch: 640 * 4,
            plane0_offset: 0,
            x_shift_px: 0,
            plane1: None,
            is_yuv: false,
            yuv_encoding: ColorEncoding::Bt709,
            yuv_range: ColorRange::Limited,
        };

        let egl = attrs.to_egl_attribs();

        assert_eq!(egl_attrib_value(&egl, egl_ext::DMA_BUF_PLANE1_FD), None);
        assert_eq!(egl_attrib_value(&egl, egl_ext::DMA_BUF_PLANE1_PITCH), None);
        assert_eq!(egl_attrib_value(&egl, egl_ext::DMA_BUF_PLANE1_OFFSET), None);

        assert_eq!(egl_attrib_value(&egl, egl_ext::YUV_COLOR_SPACE_HINT), None);

        assert_eq!(*egl.last().unwrap(), edgefirst_egl::NONE as Attrib);
    }

    /// Contiguous with padded stride: UV offset uses stride, not width.
    #[test]
    fn test_egl_attribs_contiguous_padded_stride() {
        let width = 1920usize;
        let height = 1080usize;
        let stride = 2048usize; // padded
        let attrs = DmaImportAttrs {
            width,
            height,
            drm_fourcc: DrmFourcc::Nv12,
            plane0_fd: 10,
            plane0_pitch: stride,
            plane0_offset: 0,
            x_shift_px: 0,
            plane1: Some(DmaPlane1Attrs {
                fd: 10,
                pitch: stride,
                offset: stride * height,
            }),
            is_yuv: true,
            yuv_encoding: ColorEncoding::Bt709,
            yuv_range: ColorRange::Limited,
        };

        let egl = attrs.to_egl_attribs();

        let uv_offset = egl_attrib_value(&egl, egl_ext::DMA_BUF_PLANE1_OFFSET).unwrap();
        assert_eq!(
            uv_offset,
            (stride * height) as Attrib,
            "UV offset must use stride ({stride}×{height}={}) not width ({width}×{height}={})",
            stride * height,
            width * height,
        );
    }

    // ─── Colorimetry-driven YUV hint tests ──────────────────────────────

    /// Helper: build an NV12 attr struct with the given colorimetry.
    fn nv12_attrs_with(encoding: ColorEncoding, range: ColorRange) -> DmaImportAttrs {
        DmaImportAttrs {
            width: 1920,
            height: 1080,
            drm_fourcc: DrmFourcc::Nv12,
            plane0_fd: 10,
            plane0_pitch: 1920,
            plane0_offset: 0,
            x_shift_px: 0,
            plane1: Some(DmaPlane1Attrs {
                fd: 10,
                pitch: 1920,
                offset: 1920 * 1080,
            }),
            is_yuv: true,
            yuv_encoding: encoding,
            yuv_range: range,
        }
    }

    /// BT.601 + Full range → ITU_REC601 + YUV_FULL_RANGE hints.
    #[test]
    fn test_egl_attribs_bt601_full() {
        let attrs = nv12_attrs_with(ColorEncoding::Bt601, ColorRange::Full);
        let egl = attrs.to_egl_attribs();
        assert_eq!(
            egl_attrib_value(&egl, egl_ext::YUV_COLOR_SPACE_HINT),
            Some(egl_ext::ITU_REC601 as Attrib),
            "BT.601 must map to ITU_REC601"
        );
        assert_eq!(
            egl_attrib_value(&egl, egl_ext::SAMPLE_RANGE_HINT),
            Some(egl_ext::YUV_FULL_RANGE as Attrib),
            "Full range must map to YUV_FULL_RANGE"
        );
    }

    /// BT.709 + Limited range → ITU_REC709 + YUV_NARROW_RANGE hints.
    #[test]
    fn test_egl_attribs_bt709_limited() {
        let attrs = nv12_attrs_with(ColorEncoding::Bt709, ColorRange::Limited);
        let egl = attrs.to_egl_attribs();
        assert_eq!(
            egl_attrib_value(&egl, egl_ext::YUV_COLOR_SPACE_HINT),
            Some(egl_ext::ITU_REC709 as Attrib),
            "BT.709 must map to ITU_REC709"
        );
        assert_eq!(
            egl_attrib_value(&egl, egl_ext::SAMPLE_RANGE_HINT),
            Some(egl_ext::YUV_NARROW_RANGE as Attrib),
            "Limited range must map to YUV_NARROW_RANGE"
        );
    }

    /// BT.2020 → ITU_REC2020 hint.
    #[test]
    fn test_egl_attribs_bt2020() {
        let attrs = nv12_attrs_with(ColorEncoding::Bt2020, ColorRange::Limited);
        let egl = attrs.to_egl_attribs();
        assert_eq!(
            egl_attrib_value(&egl, egl_ext::YUV_COLOR_SPACE_HINT),
            Some(egl_ext::ITU_REC2020 as Attrib),
            "BT.2020 must map to ITU_REC2020"
        );
    }

    // ─── Width-alignment pre-check (from_tensor) tests ──────────────────

    /// RGBA at a non-4-aligned width passes the `DmaImportAttrs::from_tensor` pre-check.
    #[cfg(any(target_os = "linux", target_os = "macos", target_os = "windows"))]
    #[test]
    fn test_from_tensor_accepts_non_4_aligned_rgba() {
        if !is_dma_available() {
            crate::test_support::report_skip(
                "test_from_tensor_accepts_non_4_aligned_rgba — DMA not available",
            );
            return;
        }
        use crate::{align_pitch_bytes_to_gpu_alignment, primary_plane_bpp};
        for &w in &[375usize, 427, 443] {
            let bpp = primary_plane_bpp(PixelFormat::Rgba, 1).unwrap();
            let aligned = align_pitch_bytes_to_gpu_alignment(w * bpp).unwrap();
            let Some(t) = crate::test_support::dma_allocated(
                Tensor::<u8>::image_with_stride(
                    w,
                    64,
                    PixelFormat::Rgba,
                    aligned,
                    Some(edgefirst_tensor::TensorMemory::DmaBuf),
                    edgefirst_tensor::CpuAccess::ReadWrite,
                ),
                &format!("RGBA image_with_stride at width {w}"),
            ) else {
                return;
            };
            let attrs =
                DmaImportAttrs::from_tensor(&t, PixelFormat::Rgba, false).unwrap_or_else(|e| {
                    panic!(
                        "RGBA width {w} must pass the pre-check; got {e}. \
                     Width%4 is not required for 4-bpp packed formats."
                    )
                });
            assert_eq!(attrs.width, w);
            assert_eq!(attrs.plane0_pitch, aligned);
        }
    }

    /// BGRA at a non-4-aligned width passes the `DmaImportAttrs::from_tensor` pre-check.
    #[cfg(any(target_os = "linux", target_os = "macos", target_os = "windows"))]
    #[test]
    fn test_from_tensor_accepts_non_4_aligned_bgra() {
        if !is_dma_available() {
            crate::test_support::report_skip(
                "test_from_tensor_accepts_non_4_aligned_bgra — DMA not available",
            );
            return;
        }
        use crate::{align_pitch_bytes_to_gpu_alignment, primary_plane_bpp};
        let bpp = primary_plane_bpp(PixelFormat::Bgra, 1).unwrap();
        let aligned = align_pitch_bytes_to_gpu_alignment(375 * bpp).unwrap();
        let Some(t) = crate::test_support::dma_allocated(
            Tensor::<u8>::image_with_stride(
                375,
                64,
                PixelFormat::Bgra,
                aligned,
                Some(edgefirst_tensor::TensorMemory::DmaBuf),
                edgefirst_tensor::CpuAccess::ReadWrite,
            ),
            "BGRA image_with_stride at width 375",
        ) else {
            return;
        };
        DmaImportAttrs::from_tensor(&t, PixelFormat::Bgra, false)
            .expect("BGRA width 375 must pass the pre-check");
    }

    /// Rgb passes the width pre-check at any width; the Mem tensor is refused for having no DMA-BUF.
    #[test]
    fn test_from_tensor_does_not_gate_rgb_width() {
        let t = Tensor::<u8>::image(
            375,
            64,
            PixelFormat::Rgb,
            Some(edgefirst_tensor::TensorMemory::Mem),
            edgefirst_tensor::CpuAccess::ReadWrite,
        )
        .unwrap();
        let err = DmaImportAttrs::from_tensor(&t, PixelFormat::Rgb, false)
            .expect_err("a Mem tensor has no DMA-BUF to import");
        assert!(
            !err.to_string().contains("divisible by 4"),
            "Rgb width 375 hit the width gate: {err}"
        );
    }

    /// A packed Rgb or YUYV buffer imports as an R8 view of `height` rows of `row_stride` bytes.
    #[test]
    fn test_from_tensor_nv_r8_takes_packed_as_stride_by_height() {
        if !crate::test_support::is_dma_available() {
            crate::test_support::report_skip("test_from_tensor_nv_r8_takes_packed - no DMA-BUF");
            return;
        }
        for fmt in [PixelFormat::Rgb, PixelFormat::Yuyv] {
            let t = Tensor::<u8>::image(
                322,
                241,
                fmt,
                Some(edgefirst_tensor::TensorMemory::DmaBuf),
                edgefirst_tensor::CpuAccess::ReadWrite,
            )
            .unwrap();
            let stride = t.effective_row_stride().unwrap();
            let attrs = DmaImportAttrs::from_tensor_nv_r8(&t, fmt).unwrap();
            assert_eq!(attrs.drm_fourcc, DrmFourcc::R8, "{fmt:?}");
            assert_eq!((attrs.width, attrs.height), (stride, 241), "{fmt:?}");
            assert_eq!(attrs.plane0_pitch, stride, "{fmt:?}");
            assert!(attrs.plane1.is_none(), "{fmt:?}");
        }
    }

    /// Grey at a tight pitch that is not a multiple of 4 is rejected.
    #[test]
    fn test_from_tensor_rejects_non_4_aligned_grey() {
        let t = Tensor::<u8>::image(
            375,
            64,
            PixelFormat::Grey,
            Some(edgefirst_tensor::TensorMemory::Mem),
            edgefirst_tensor::CpuAccess::ReadWrite,
        )
        .unwrap();
        let err = DmaImportAttrs::from_tensor(&t, PixelFormat::Grey, false)
            .expect_err("Grey width 375 must still be rejected");
        assert!(matches!(err, crate::Error::NotSupported(_)));
    }

    /// Grey at the same width with a 4-byte-aligned pitch passes the pitch pre-check.
    #[test]
    fn test_from_tensor_accepts_grey_at_aligned_pitch() {
        let t = Tensor::<u8>::image(
            375,
            64,
            PixelFormat::Grey,
            Some(edgefirst_tensor::TensorMemory::Mem),
            edgefirst_tensor::CpuAccess::ReadWrite,
        )
        .unwrap()
        .with_row_stride(384)
        .unwrap();
        let err = DmaImportAttrs::from_tensor(&t, PixelFormat::Grey, false)
            .expect_err("a Mem tensor has no DMA-BUF");
        assert!(
            matches!(err, crate::Error::NotImplemented(_)),
            "Grey 375 at pitch 384 was refused by the pitch gate: {err:?}"
        );
    }

    // ─── aligned_source_base tests ───────────────────────────────────

    /// RGBA offsets 0, 64, 256 and 2048 are unchanged; 32 and 2080 rebase to the 64-byte floor.
    #[test]
    fn aligned_source_base_rgba_floors_to_64_and_shifts_whole_pixels() {
        let f = |o| super::aligned_source_base(o, 4, 64);
        assert_eq!(f(0), Some((0, 0)));
        assert_eq!(f(64), Some((64, 0)));
        assert_eq!(f(256), Some((256, 0)));
        assert_eq!(f(2048), Some((2048, 0)));
        assert_eq!(f(32), Some((0, 8)));
        assert_eq!(f(2080), Some((2048, 8)));
    }

    /// Grey shifts by one texel per byte of the 64-byte residue.
    #[test]
    fn aligned_source_base_grey_shifts_every_residue_byte() {
        let f = |o| super::aligned_source_base(o, 1, 64);
        assert_eq!(f(0), Some((0, 0)));
        assert_eq!(f(32), Some((0, 32)));
        assert_eq!(f(2080), Some((2048, 32)));
        assert_eq!(f(63), Some((0, 63)));
    }

    /// RGB bases step by `lcm(64, 3) = 192`.
    #[test]
    fn aligned_source_base_rgb_steps_by_lcm_192() {
        let f = |o| super::aligned_source_base(o, 3, 64);
        assert_eq!(f(0), Some((0, 0)));
        assert_eq!(f(192), Some((192, 0)));
        assert_eq!(f(96), Some((0, 32)));
        assert_eq!(f(2079), Some((1920, 53)));
        assert_ne!(f(2079).map(|(b, _)| b), Some(2048));
    }

    /// An offset that is not a whole number of pixels, or a zero `bpp` or `align`, returns `None`.
    #[test]
    fn aligned_source_base_refuses_a_fractional_pixel_offset() {
        assert_eq!(super::aligned_source_base(2080, 3, 64), None);
        assert_eq!(super::aligned_source_base(2081, 4, 64), None);
        assert_eq!(super::aligned_source_base(1, 2, 64), None);
        assert_eq!(super::aligned_source_base(64, 0, 64), None);
        assert_eq!(super::aligned_source_base(64, 4, 0), None);
    }

    /// Over a sweep, the base is import-aligned and the shift is the whole-pixel distance to the offset.
    #[test]
    fn aligned_source_base_invariants_hold_across_a_sweep() {
        for bpp in [1usize, 3, 4] {
            for px in 0usize..4096 {
                let offset = px * bpp;
                let Some((base, shift)) = super::aligned_source_base(offset, bpp, 64) else {
                    panic!("a whole-pixel offset {offset} at {bpp} bpp must have a base");
                };
                assert_eq!(base % 64, 0, "base {base} must be 64-byte aligned");
                assert!(base <= offset, "base {base} must not pass offset {offset}");
                assert_eq!(
                    base + shift as usize * bpp,
                    offset,
                    "shift {shift} px at {bpp} bpp must close base {base} to {offset}"
                );
            }
        }
    }

    // ─── source rebase onto an aligned base ──────────────────────────

    /// Allocate the DMA parent the rebase tests view into, or `None` (warned)
    /// when this machine has no usable DMA heap.
    #[cfg(any(target_os = "linux", target_os = "macos", target_os = "windows"))]
    fn dma_parent(width: usize, height: usize, fmt: PixelFormat) -> Option<Tensor<u8>> {
        let what = format!("{width}x{height} {fmt:?} DMA image");
        if !dma_heap_or_skip(&what) {
            return None;
        }
        crate::test_support::dma_allocated(
            dma_only(Tensor::<u8>::image(
                width,
                height,
                fmt,
                Some(TensorMemory::DmaBuf),
                edgefirst_tensor::CpuAccess::ReadWrite,
            )),
            &what,
        )
    }

    /// A source at an unaligned RGBA offset imports from the 64-byte floor, widened by the remainder.
    #[test]
    #[cfg(any(target_os = "linux", target_os = "macos", target_os = "windows"))]
    fn from_tensor_rebases_an_unaligned_rgba_source_to_an_aligned_base() {
        let Some(parent) = dma_parent(64, 64, PixelFormat::Rgba) else {
            return;
        };
        let pitch = parent.effective_row_stride().unwrap_or(256);
        let view = parent
            .view(edgefirst_tensor::Region::new(8, 8, 16, 16))
            .unwrap();
        assert_eq!(view.plane_offset(), Some(8 * pitch + 32));
        let s = DmaImportAttrs::from_tensor(&view, PixelFormat::Rgba, false).unwrap();
        assert_eq!(s.plane0_offset % 64, 0, "the import base must be aligned");
        assert_eq!(s.plane0_offset, 8 * pitch + 32 - 32);
        assert_eq!(s.x_shift_px, 8, "32 bytes is 8 RGBA pixels");
        assert_eq!(s.width, 16 + 8, "the import widens by the shift");
        assert_eq!(
            s.plane0_pitch, pitch,
            "the pitch is the parent's, unchanged"
        );
    }

    /// An aligned source is not rebased, shifted or widened.
    #[test]
    #[cfg(any(target_os = "linux", target_os = "macos", target_os = "windows"))]
    fn from_tensor_leaves_an_aligned_source_alone() {
        let Some(parent) = dma_parent(64, 64, PixelFormat::Rgba) else {
            return;
        };
        let pitch = parent.effective_row_stride().unwrap_or(256);
        assert_eq!(pitch % 64, 0, "the premise: a 64-aligned parent pitch");
        let view = parent
            .view(edgefirst_tensor::Region::new(0, 8, 16, 16))
            .unwrap();
        let s = DmaImportAttrs::from_tensor(&view, PixelFormat::Rgba, false).unwrap();
        assert_eq!(s.plane0_offset, 8 * pitch);
        assert_eq!(s.x_shift_px, 0);
        assert_eq!(s.width, 16);
    }

    /// A destination is not rebased: it imports its parent at base 0.
    #[test]
    #[cfg(any(target_os = "linux", target_os = "macos", target_os = "windows"))]
    fn from_tensor_never_rebases_a_destination() {
        let Some(parent) = dma_parent(64, 64, PixelFormat::Rgba) else {
            return;
        };
        let view = parent
            .view(edgefirst_tensor::Region::new(8, 8, 16, 16))
            .unwrap();
        let d = DmaImportAttrs::from_tensor(&view, PixelFormat::Rgba, true).unwrap();
        assert_eq!(d.plane0_offset, 0, "a dst view imports its parent");
        assert_eq!(d.x_shift_px, 0, "and is never shifted");
        assert_eq!(d.width, 64, "at the parent's own width");
    }
    // ─── rebase_fits_pitch ───────────────────────────────────────────

    /// `rebase_fits_pitch` accepts a shift up to the pitch padding and treats overflow as not fitting.
    #[test]
    fn rebase_fits_pitch_declines_a_tight_foreign_pitch() {
        let fits = super::rebase_fits_pitch;
        assert!(!fits(100, 1, 4, 400));
        assert!(!fits(100, 8, 4, 400));
        assert!(fits(100, 0, 4, 400));
        assert!(fits(100, 12, 4, 448));
        assert!(!fits(100, 13, 4, 448));
        assert!(fits(16, 8, 4, 256));
        assert!(!fits(usize::MAX, 1, 4, usize::MAX));
    }

    // ─── which formats rebase ────────────────────────────────────────

    /// An unaligned `Rgb` source view rebases by the `lcm(64, 3) = 192` step.
    #[test]
    #[cfg(any(target_os = "linux", target_os = "macos", target_os = "windows"))]
    fn from_tensor_rebases_an_unaligned_rgb_source_by_the_lcm_step() {
        let Some(parent) = dma_parent(64, 64, PixelFormat::Rgb) else {
            return;
        };
        let pitch = parent.effective_row_stride().unwrap_or(192);
        let view = parent
            .view(edgefirst_tensor::Region::new(8, 8, 16, 16))
            .unwrap();
        let offset = view.plane_offset().unwrap_or(0);
        assert_eq!(offset, 8 * pitch + 24);
        let expect_base = offset - offset % 192;
        let expect_shift = (offset - expect_base) / 3;
        assert!(
            expect_shift > 0,
            "precondition: offset {offset} at pitch {pitch} must be unaligned"
        );
        let s = DmaImportAttrs::from_tensor(&view, PixelFormat::Rgb, false).unwrap();
        assert_eq!(s.plane0_offset, expect_base, "base steps by lcm(64, 3)");
        assert_eq!(s.plane0_offset % 64, 0, "and is still import-aligned");
        assert_eq!(s.x_shift_px as usize, expect_shift);
        assert_eq!(s.width, 16 + expect_shift, "the import widens by the shift");
        assert_eq!(
            s.plane0_pitch, pitch,
            "the pitch is the parent's, unchanged"
        );
    }

    /// An unaligned `Grey` source view rebases to the 64-byte floor with one texel of shift per byte.
    #[test]
    #[cfg(any(target_os = "linux", target_os = "macos", target_os = "windows"))]
    fn from_tensor_rebases_an_unaligned_grey_source_byte_for_texel() {
        let Some(parent) = dma_parent(64, 64, PixelFormat::Grey) else {
            return;
        };
        let pitch = parent.effective_row_stride().unwrap_or(64);
        let view = parent
            .view(edgefirst_tensor::Region::new(8, 8, 16, 16))
            .unwrap();
        let offset = view.plane_offset().unwrap_or(0);
        assert_eq!(offset, 8 * pitch + 8);
        let expect_base = offset - offset % 64;
        let expect_shift = offset - expect_base;
        assert!(
            expect_shift > 0,
            "precondition: offset {offset} at pitch {pitch} must be unaligned"
        );
        let s = DmaImportAttrs::from_tensor(&view, PixelFormat::Grey, false).unwrap();
        assert_eq!(s.plane0_offset, expect_base);
        assert_eq!(s.plane0_offset % 64, 0);
        assert_eq!(
            s.x_shift_px as usize, expect_shift,
            "1 bpp: a byte is a texel"
        );
        assert_eq!(s.width, 16 + expect_shift);
        assert_eq!(s.plane0_pitch, pitch);
    }

    // ─── resolved_source_plane0_offset matches from_tensor ────────────

    /// `resolve_source_plane0` yields an aligned offset for the rebased and aligned shapes only.
    #[test]
    fn the_resolved_offset_tells_the_four_shapes_apart() {
        let aligned = |fmt, w, pitch, off| {
            super::resolve_source_plane0(fmt, w, pitch, off)
                .0
                .is_multiple_of(DMA_IMPORT_OFFSET_ALIGN)
        };

        // Rebased.
        assert!(aligned(PixelFormat::Rgba, 16, 256, 2080));
        // Aligned.
        assert!(aligned(PixelFormat::Rgba, 64, 256, 2048));
        // Tight pitch with no room to widen.
        assert!(!aligned(PixelFormat::Rgba, 64, 256, 32));
        // Offset that is not a whole number of pixels.
        assert!(!aligned(PixelFormat::Rgb, 16, 320, 323));
    }

    /// `resolved_source_plane0_offset` equals the `plane0_offset` that `from_tensor` resolves.
    #[test]
    #[cfg(any(target_os = "linux", target_os = "macos", target_os = "windows"))]
    fn the_gate_reads_the_offset_the_import_presents() {
        if !is_dma_available() {
            use std::io::Write;
            let _ = writeln!(
                &mut std::io::stderr(),
                "WARNING: SKIPPED: {} - DMA not available",
                function!()
            );
            return;
        }
        // (format, parent w/h, view or whole-tensor offset)
        let cases: [(PixelFormat, usize, usize, Option<usize>); 5] = [
            // A rebased RGBA view: (8, 8) of 64x64, offset 2080.
            (PixelFormat::Rgba, 64, 64, None),
            // The tight-pitch whole tensor at a window offset.
            (PixelFormat::Rgba, 64, 64, Some(32)),
            // An aligned window offset.
            (PixelFormat::Rgba, 64, 64, Some(64)),
            // Rgb, whose 64-padded stride is not a multiple of 3.
            (PixelFormat::Rgb, 100, 8, Some(323)),
            // A format the rebase does not cover.
            (PixelFormat::Nv12, 64, 64, Some(32)),
        ];
        let total = cases.len();
        let mut compared = 0usize;
        for (fmt, w, h, whole_offset) in cases {
            let Some(mut parent) = dma_parent(w, h, fmt) else {
                return;
            };
            if fmt == PixelFormat::Rgb {
                let pitch = parent.effective_row_stride().unwrap_or(0);
                assert_eq!(pitch, 320, "precondition: the padded Rgb stride");
                let off = whole_offset.unwrap();
                assert_eq!(off, pitch + 3, "precondition: row 1, column 1");
                assert_ne!(off % 3, 0, "precondition: {off} is mid-pixel");
                assert_ne!(off % 64, 0, "precondition: {off} is unaligned");
            }
            let attrs = match whole_offset {
                Some(off) => {
                    parent.set_plane_offset(off);
                    DmaImportAttrs::from_tensor(&parent, fmt, false)
                        .unwrap_or_else(|e| panic!("{fmt:?} at {off}: {e}"))
                }
                None => {
                    let view = parent
                        .view(edgefirst_tensor::Region::new(8, 8, 16, 16))
                        .unwrap();
                    let a = DmaImportAttrs::from_tensor(&view, fmt, false).unwrap();
                    assert_eq!(
                        super::resolved_source_plane0_offset(&view, fmt),
                        a.plane0_offset,
                        "{fmt:?} view: the gate and the import must agree"
                    );
                    compared += 1;
                    continue;
                }
            };
            assert_eq!(
                super::resolved_source_plane0_offset(&parent, fmt),
                attrs.plane0_offset,
                "{fmt:?} at {whole_offset:?}: the gate and the import must agree"
            );
            compared += 1;
        }
        assert_eq!(
            compared, total,
            "a DMA heap is available, so every case must be comparable, but only \
             {compared} of {total} were (see the SKIPPED lines above); \
             the gate-versus-import equality went unpinned for the rest"
        );
    }

    /// A whole RGBA tensor at an unaligned window offset with a tight pitch keeps its own offset.
    #[test]
    #[cfg(any(target_os = "linux", target_os = "macos", target_os = "windows"))]
    fn from_tensor_cannot_rebase_a_window_into_a_tight_pitch() {
        let Some(mut t) = dma_parent(64, 64, PixelFormat::Rgba) else {
            return;
        };
        let pitch = t.effective_row_stride().unwrap_or(256);
        assert_eq!(pitch, 64 * 4, "precondition: the pitch is tight");
        t.set_plane_offset(32);
        let s = DmaImportAttrs::from_tensor(&t, PixelFormat::Rgba, false).unwrap();
        assert_eq!(s.plane0_offset, 32, "no room to widen, so no rebase");
        assert_eq!(s.x_shift_px, 0);
        assert_eq!(s.width, 64, "and no widening");
        assert!(
            !super::resolve_source_plane0(PixelFormat::Rgba, 64, pitch, 32)
                .0
                .is_multiple_of(DMA_IMPORT_OFFSET_ALIGN),
            "so Mali must still decline it"
        );
    }

    /// Formats the rebase does not cover import at their own offset with no shift.
    #[test]
    #[cfg(any(target_os = "linux", target_os = "macos", target_os = "windows"))]
    fn from_tensor_never_rebases_the_formats_the_fold_misses() {
        for fmt in [PixelFormat::Yuyv, PixelFormat::Nv12, PixelFormat::PlanarRgb] {
            let Some(mut t) = dma_parent(64, 64, fmt) else {
                return;
            };
            t.set_plane_offset(32);
            let s = DmaImportAttrs::from_tensor(&t, fmt, false)
                .unwrap_or_else(|e| panic!("{fmt:?} source import must resolve: {e}"));
            assert_eq!(s.x_shift_px, 0, "{fmt:?} must not be rebased");
            assert_eq!(
                s.plane0_offset, 32,
                "{fmt:?} keeps its own (unaligned) offset"
            );
            assert_eq!(s.width, 64, "{fmt:?} must not be widened");
        }
    }

    // ─── NV12 destination import tests ───────────────────────────────

    /// A `w`x`h` multiplane NV12 tensor over two tensors of `memory`, or
    /// `None` (warned) when a DMA plane cannot be allocated.
    fn nv12_planes(w: usize, h: usize, memory: TensorMemory) -> Option<Tensor<u8>> {
        let plane = |rows: usize, name: &str| {
            let t = Tensor::<u8>::new(&[rows, w], Some(memory), Some(name));
            if memory == TensorMemory::DmaBuf {
                crate::test_support::dma_allocated(dma_only(t), name)
            } else {
                Some(t.expect("heap plane"))
            }
        };
        let luma = plane(h, "nv12 luma")?;
        let chroma = plane(h / 2, "nv12 chroma")?;
        Some(Tensor::<u8>::from_planes(luma, chroma, PixelFormat::Nv12).expect("multiplane NV12"))
    }

    /// A single-plane NV12 destination imports as one R8 surface, pitch wide
    /// and `h + h / 2` rows tall, at its plane offset.
    #[test]
    fn nv12_dst_r8_covers_luma_and_chroma_rows_at_the_pitch() {
        let Some(t) = dma_parent(320, 240, PixelFormat::Nv12) else {
            return;
        };
        let pitch = t.effective_row_stride().unwrap_or(320);
        let a = DmaImportAttrs::from_tensor_nv12_dst_r8(&t).expect("single-plane NV12 imports");
        assert_eq!(a.drm_fourcc, DrmFourcc::R8);
        assert_eq!(a.width, pitch);
        assert_eq!(a.height, 360);
        assert_eq!(a.plane0_pitch, pitch);
        assert_eq!(a.plane0_offset, 0);
        assert_eq!(a.plane0_fd, t.dmabuf().unwrap().as_raw_fd());
        assert!(a.plane1.is_none());
        assert!(!a.is_yuv);
    }

    /// A multiplane NV12 destination imports its luma and chroma as separate
    /// R8 surfaces, each at its own buffer.
    #[test]
    fn nv12_dst_split_imports_each_plane_at_its_own_buffer() {
        if !dma_heap_or_skip("multiplane NV12 destination") {
            return;
        }
        let Some(t) = nv12_planes(320, 240, TensorMemory::DmaBuf) else {
            return;
        };
        let (luma, chroma) =
            DmaImportAttrs::from_tensor_nv12_dst_split(&t).expect("multiplane NV12 imports");
        assert_eq!(
            (luma.drm_fourcc, chroma.drm_fourcc),
            (DrmFourcc::R8, DrmFourcc::R8)
        );
        assert_eq!((luma.width, luma.height), (320, 240));
        assert_eq!((chroma.width, chroma.height), (320, 120));
        assert_eq!((luma.plane0_pitch, chroma.plane0_pitch), (320, 320));
        assert_eq!(luma.plane0_fd, t.dmabuf().unwrap().as_raw_fd());
        assert_eq!(
            chroma.plane0_fd,
            t.chroma().unwrap().dmabuf().unwrap().as_raw_fd()
        );
        assert_ne!(luma.plane0_fd, chroma.plane0_fd);
        assert!(luma.plane1.is_none() && chroma.plane1.is_none());
    }

    /// A multiplane NV12 destination with padded pitches imports each plane
    /// as wide as its own pitch.
    #[test]
    fn nv12_dst_split_imports_each_plane_as_wide_as_its_pitch() {
        if !dma_heap_or_skip("padded multiplane NV12 destination") {
            return;
        }
        let (w, h, luma_pitch, chroma_pitch) = (48, 64, 112, 80);
        let plane = |rows: usize, pitch: usize, name: &str| {
            let mut t = crate::test_support::dma_allocated(
                dma_only(Tensor::<u8>::new(
                    &[rows, pitch],
                    Some(TensorMemory::DmaBuf),
                    Some(name),
                )),
                name,
            )?;
            t.set_logical_shape(&[rows, w]).expect("logical width");
            Some(t)
        };
        let Some(luma) = plane(h, luma_pitch, "padded nv12 luma") else {
            return;
        };
        let Some(mut chroma) = plane(h / 2, chroma_pitch, "padded nv12 chroma") else {
            return;
        };
        chroma.set_row_stride_unchecked(chroma_pitch);
        let mut t =
            Tensor::<u8>::from_planes(luma, chroma, PixelFormat::Nv12).expect("multiplane NV12");
        t.set_row_stride(luma_pitch).expect("luma pitch");
        let (luma, chroma) =
            DmaImportAttrs::from_tensor_nv12_dst_split(&t).expect("multiplane NV12 imports");
        assert_eq!(
            (luma.width, luma.height, luma.plane0_pitch),
            (luma_pitch, h, luma_pitch)
        );
        assert_eq!(
            (chroma.width, chroma.height, chroma.plane0_pitch),
            (chroma_pitch, h / 2, chroma_pitch)
        );
    }

    /// Each builder refuses the other layout, and a plane pitch that is not a
    /// multiple of 4.
    #[test]
    fn nv12_dst_builders_refuse_the_other_layout_and_an_odd_pitch() {
        let Some(single) = dma_parent(320, 240, PixelFormat::Nv12) else {
            return;
        };
        let Some(split) = nv12_planes(320, 240, TensorMemory::DmaBuf) else {
            return;
        };
        let Some(mut narrow) = dma_parent(62, 240, PixelFormat::Nv12) else {
            return;
        };
        narrow.set_row_stride(62).expect("pitch 62 at width 62");
        let Some(narrow_split) = nv12_planes(62, 240, TensorMemory::DmaBuf) else {
            return;
        };
        const LAYOUT_SPLIT: &str = "the chroma plane follows the luma plane";
        const LAYOUT_R8: &str = "multiplane tensor not supported";
        const PITCH: &str = "needs a row pitch divisible by 4, got 62";
        for (what, err, expected) in [
            (
                "split of a single-plane tensor",
                DmaImportAttrs::from_tensor_nv12_dst_split(&single).map(|_| ()),
                LAYOUT_SPLIT,
            ),
            (
                "R8 of a multiplane tensor",
                DmaImportAttrs::from_tensor_nv12_dst_r8(&split).map(|_| ()),
                LAYOUT_R8,
            ),
            (
                "R8 at pitch 62",
                DmaImportAttrs::from_tensor_nv12_dst_r8(&narrow).map(|_| ()),
                PITCH,
            ),
            (
                "split at pitch 62",
                DmaImportAttrs::from_tensor_nv12_dst_split(&narrow_split).map(|_| ()),
                PITCH,
            ),
        ] {
            assert!(
                matches!(&err, Err(crate::Error::NotSupported(m)) if m.contains(expected)),
                "{what}: expected NotSupported containing {expected:?}, got {err:?}"
            );
        }
    }
}
