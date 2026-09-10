// SPDX-FileCopyrightText: Copyright 2025 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! DMA-BUF plane resolution for EGL image import.
//!
//! Separates the logic that resolves plane parameters (fd, pitch, offset)
//! from the actual `eglCreateImage` call so that the attribute construction
//! can be unit-tested without a GPU context.

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
/// Captures all the information needed to build the `eglCreateImage`
/// attribute list.  Three NV12 import scenarios produce different values:
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
    /// pixel. Nonzero only for a SOURCE rebased onto an aligned base (issue
    /// #170): `width` is widened by it and `plane0_offset` walked back by
    /// `x_shift_px * bpp`, so the tensor's pixel `(x, y)` is import texel
    /// `(x + x_shift_px, y)` at the same pitch. The engine folds it through
    /// `GlPlatform::import_origin` into the sampling rectangle.
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
/// `align` **and** a whole number of `bpp`-byte pixels away from `offset`,
/// with that distance in pixels.
///
/// Mali's EGL DMA-BUF import silently samples zeros unless
/// `EGL_DMA_BUF_PLANE0_OFFSET_EXT` is 64-byte aligned (issue #165, measured
/// on i.MX 95: offsets 32 and 2080 sample zeros, 0/64/256/2048 are correct).
/// Rather than decline such a source into an upload, the import starts at
/// this base and widens by the returned shift, so the tensor's first pixel
/// lands on texel `x_shift_px` of the imported texture and the engine folds
/// that shift into the sampling rectangle (issue #170).
///
/// The step is `lcm(align, bpp)`, not `align`: walking back by `align` alone
/// lands mid-pixel whenever `bpp` does not divide `align` — for `Rgb`'s 3
/// bytes a 64-byte floor of 2079 gives a 31-byte remainder, which is not a
/// whole RGB pixel, while the 192-byte step gives 159 bytes = 53 pixels.
///
/// `None` when no such base exists — `offset` is not a whole number of
/// pixels (a 128-byte padded pitch is not a multiple of 3, so an `Rgb` row
/// offset can land mid-pixel), or a degenerate `bpp`/`align`. The caller
/// declines the import and uploads, which is Stage F's behaviour.
pub(super) fn aligned_source_base(offset: usize, bpp: usize, align: usize) -> Option<(usize, u32)> {
    if bpp == 0 || align == 0 || !offset.is_multiple_of(bpp) {
        return None;
    }
    // lcm(align, bpp), computed as `align / gcd * bpp` so the intermediate
    // cannot overflow for any realistic pair.
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

/// Whether a source rebased by `x_shift_px` pixels still fits the pitch it
/// imports at: the widened row `(width_px + x_shift_px) * bpp` must not run
/// past `plane0_pitch`.
///
/// A driver that derives a row from WIDTH and PITCH rejects an import whose
/// row runs past its pitch, and the widened row is the only thing the rebase
/// can make run past. `false` declines the rebase, and the caller imports at
/// the original offset instead.
///
/// The room comes from the pitch padding. A view's `width_px` is its own
/// while its pitch is the parent's, so a 64-padded parent pitch normally has
/// room for every shift `aligned_source_base` can return. A pitch that is
/// merely tight has none: a foreign DMA-BUF adopted through `from_fd` at the
/// producer's unpadded stride, or a whole tensor at a `set_plane_offset`
/// window whose row already fills its pitch. Overflow counts as not fitting.
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
/// for the formats it does not cover.
///
/// Packed formats whose texel is exactly `channels()` bytes only: YUYV/VYUY
/// carry a 2-pixel macropixel phase a texel shift would break, NV12's second
/// plane has an offset of its own, and the R8 paths' pitch IS their width so
/// they cannot widen at all.
fn rebase_bpp(src_fmt: PixelFormat) -> Option<usize> {
    matches!(
        src_fmt,
        PixelFormat::Rgba | PixelFormat::Bgra | PixelFormat::Rgb | PixelFormat::Grey
    )
    .then(|| src_fmt.channels())
}

/// The plane-0 channel count the **import** uses, which is not always
/// `PixelFormat::channels()`: NV12's plane 0 is single-channel luma, and
/// PlanarRgb's is one R8 plane three times as tall rather than three
/// interleaved channels.
///
/// [`DmaImportAttrs::from_tensor`] derives the same number inside the branch
/// that also resolves width, height and fourcc, and debug-asserts it against
/// this function. [`resolved_source_plane0_offset`] calls this directly. The
/// two must agree, because the tight pitch they fall back on when a tensor
/// stores no row stride is what the gate and the import would otherwise
/// disagree about.
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
        // Luma plane is 1 byte/pixel for NV12 semi-planar YUV.
        width_px
    } else {
        width_px * channels
    }
}

/// The plane-0 offset a SOURCE import of this shape presents to EGL, and the
/// texel shift folded into the sampling rectangle to pay for it.
///
/// **The single source of truth for the source rebase (issue #170.)**
/// [`DmaImportAttrs::from_tensor`] builds its import from this, and the Mali
/// gate in `processor/mod.rs` asks it which offset EGL will actually be
/// handed — see [`resolved_source_plane0_offset`]. Neither may
/// reimplement the decision: the gate declines a source Mali would sample
/// zeros from, and it can only be right about that if it is reading the very
/// offset the import will carry.
///
/// Three outcomes, in order:
///
/// * A format the rebase does not cover, or an offset that is already
///   64-byte aligned — returned unchanged with no shift. The aligned case
///   is every ordinary import.
/// * An unaligned offset the rebase can move — the aligned base below it,
///   and the remainder in whole pixels. This is the zero-copy win.
/// * An unaligned offset it cannot move — an offset that is not a whole
///   number of pixels (`offset % bpp != 0`, which a 64-padded stride
///   produces whenever `bpp` does not divide it), or a pitch with no room
///   to widen into — returned
///   unchanged, unaligned. The drivers that import such an offset correctly
///   go on doing so; Mali declines it upstream and uploads instead, which is
///   exactly what this function exists to let the gate see.
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
/// source. This is what the Mali gate calls.
///
/// A source never collapses onto a parent import (`view_origin` is honored
/// for destinations only), so its geometry is its own and these three
/// accessors are the whole input. The tight-pitch fallback goes through
/// [`import_plane0_channels`], the same rule the import applies, so the two
/// cannot answer differently for a tensor that stores no row stride —
/// `PixelFormat::channels()` would say three for PlanarRgb where the import
/// says one. Unreachable while the rebase covers neither planar nor NV (the
/// caller returns before the pitch is read), but the gate must not depend on
/// that to be right.
pub(super) fn resolved_source_plane0_offset(src: &Tensor<u8>, src_fmt: PixelFormat) -> usize {
    let width = src.width().unwrap_or(0);
    let pitch = src
        .effective_row_stride()
        .unwrap_or_else(|| tight_plane0_pitch(src_fmt, width, import_plane0_channels(src_fmt)));
    resolve_source_plane0(src_fmt, width, pitch, src.plane_offset().unwrap_or(0)).0
}

impl DmaImportAttrs {
    /// Resolve plane parameters from a tensor and its pixel format.
    ///
    /// This extracts all the fd/pitch/offset values that `eglCreateImage`
    /// needs without actually calling EGL.
    pub fn from_tensor(
        src: &Tensor<u8>,
        src_fmt: PixelFormat,
        for_dst: bool,
    ) -> Result<Self, Error> {
        // The parent-import collapse is a DESTINATION technique: a dst tile is a
        // `glViewport` ROI into one parent EGLImage, so a `view()`/`batch()` dst
        // imports the parent geometry at offset 0 and every sibling shares it. A
        // SOURCE view must import its OWN region instead — otherwise it would
        // import the parent but `convert()` would sample using the view's
        // dimensions, reading the wrong region (source cropping is the view's
        // region, lowered to sampling UVs). So only honor `view_origin` for
        // destinations; a whole tensor (or any source) imports its own geometry
        // at its own (possibly foreign) offset.
        let view_origin = if for_dst { src.view_origin() } else { None };
        let (src_w, src_h) = match view_origin {
            Some(vo) => (vo.parent_width, vo.parent_height),
            None => (
                src.width().ok_or(Error::NotAnImage)?,
                src.height().ok_or(Error::NotAnImage)?,
            ),
        };
        let src_channels = src_fmt.channels();

        // Resolve the tensor's colorimetry (pure — does not mutate the
        // tensor). Missing axes are filled by the HD/SD height heuristic.
        // Drives the EGL YUV color-space + sample-range hints below.
        let cm = resolve_colorimetry(src.colorimetry(), src.height());
        let yuv_encoding = cm.encoding.unwrap_or(ColorEncoding::Bt709);
        let yuv_range = cm.range.unwrap_or(ColorRange::Limited);

        let (width, height, drm_fourcc, channels) = if src_fmt == PixelFormat::Nv12 {
            // NV12's genuine floor is even width (4:2:0 chroma is W/2 CbCr pairs).
            // The historical width%4 gate was for word-aligned row reads of the
            // tight luma pitch — now moot because the import uses the 64-aligned
            // `plane0_pitch` (`effective_row_stride`) below, not `width`. Relaxed
            // to width%2 so even-but-not-4-aligned camera widths take the
            // zero-copy sampler path; a driver that still rejects falls back via
            // `egl_create_image_with_fallback`.
            if !src_w.is_multiple_of(2) {
                return Err(Error::NotSupported(format!(
                    "EGLImage requires even width for {src_fmt} (4:2:0 chroma), got {src_w}"
                )));
            }
            // Luma (plane 0) is full-resolution R8; chroma subsampling
            // affects plane 1 (handled below).
            (src_w, src_h, pixel_format_to_drm(src_fmt)?, 1usize)
        } else if src_fmt.layout() == PixelLayout::Planar {
            if !src_w.is_multiple_of(16) {
                return Err(Error::NotSupported(format!(
                    "EGLImage requires width divisible by 16 for {src_fmt}, got {src_w}"
                )));
            }
            match src_fmt {
                PixelFormat::PlanarRgb => (src_w, src_h * 3, DrmFourcc::R8, 1),
                _ => {
                    return Err(Error::NotSupported(format!(
                        "Unsupported Planar format {src_fmt:?}"
                    )));
                }
            }
        } else {
            // For 3-bpp (Rgb) and 1-bpp (Grey) packed formats the natural
            // row pitch is `width * bpp`, which is only 4-byte aligned
            // when `width % 4 == 0`. EGL DMA-BUF import via
            // `eglCreateImage` requires word-aligned row reads on every
            // tested platform (Mali G310, Vivante GC7000UL, V3D, Tegra),
            // so we keep the blanket width%4 check for those.
            //
            // It is the TIGHT pitch this protects, and the aligned-base
            // rebase (issue #170) never changes that pitch: the rebase
            // widens the import and walks the offset back, both at the
            // pitch the tensor already had, so a width that satisfies this
            // check before the rebase still does after it.
            //
            // 4-bpp packed formats (Rgba, Bgra) have a row pitch of
            // `width * 4` which is trivially 4-byte aligned at any width.
            // The DRM fourcc spec (ABGR8888 / ARGB8888) imposes no
            // pixel-alignment requirement on top of pitch alignment, and
            // these drivers accept arbitrary widths as long as the
            // 64-byte pitch padding (handled by `padded_dma_pitch_for`)
            // is applied. Letting these through unlocks the zero-copy
            // EGL path for dataset-loader widths like 375 / 427 / 443 —
            // empirically verified on imx8mp-frdm, imx95-frdm,
            // rpi5-hailo, and orin-nano. If a driver fails the import,
            // `egl_create_image_with_fallback` downgrades to the CPU
            // texture upload path with a one-shot slow-path warning.
            let needs_width_mod_4 = !matches!(src_fmt, PixelFormat::Rgba | PixelFormat::Bgra);
            if needs_width_mod_4 && !src_w.is_multiple_of(4) {
                return Err(Error::NotSupported(format!(
                    "EGLImage requires width divisible by 4 for {src_fmt}, got {src_w}"
                )));
            }
            (src_w, src_h, pixel_format_to_drm(src_fmt)?, src_channels)
        };

        // The branch above resolves `channels` alongside width, height and
        // fourcc, so it cannot simply call `import_plane0_channels`; this
        // pins the two together instead, because the gate reads the helper
        // and both feed `tight_plane0_pitch`.
        debug_assert_eq!(
            channels,
            import_plane0_channels(src_fmt),
            "{src_fmt:?}: the import's plane-0 channel count and the gate's must agree"
        );

        // `dmabuf()`, not `as_dma().fd`. Both reach the same file
        // descriptor and both are on `Tensor<T>` with the same signature on
        // either backend -- but `as_dma` downcasts to `DmaTensor<T>`, which
        // is `static`-backend-internal storage that the `dynamic` backend
        // has no instance of and returns `None` for unconditionally. Going
        // through it meant a genuinely DMA-backed tensor was refused here
        // under `dynamic`, silently declining the zero-copy EGLImage import
        // and falling back to a PBO copy. Nothing failed and nothing said
        // so. See task P2b's review, F3.
        let fd = src
            .dmabuf()
            .map_err(|e| {
                Error::NotImplemented(format!(
                    "OpenGL EGLImage requires DMA tensor, got {:?} ({e})",
                    src.memory()
                ))
            })?
            .as_raw_fd();

        // For multiplane NV12, get the UV plane's fd from the chroma tensor
        let uv_fd = if src.is_multiplane() {
            let chroma = src.chroma().unwrap();
            let chroma_fd = chroma.dmabuf().map_err(|e| {
                Error::NotImplemented(format!("Multiplane chroma tensor must be DMA-backed ({e})"))
            })?;
            Some(chroma_fd.as_raw_fd())
        } else {
            None
        };

        // A view imports its PARENT, so it pitches on the parent's row stride
        // (`view_origin`), NOT the view's own `effective_row_stride` (which a
        // single-row view sets tight). A whole tensor uses its stored stride if
        // set (externally allocated buffers with row padding), otherwise the
        // tightly-packed pitch.
        let plane0_pitch = match view_origin {
            Some(vo) => vo.parent_row_stride,
            None => src
                .effective_row_stride()
                .unwrap_or_else(|| tight_plane0_pitch(src_fmt, width, channels)),
        };

        // A view imports its parent at offset 0 (its byte offset becomes the
        // viewport, never the import base); a whole tensor honors its own offset.
        let plane0_offset = if view_origin.is_some() {
            0
        } else {
            src.plane_offset().unwrap_or(0)
        };

        // Mali's EGL DMA-BUF import silently samples zeros unless the plane
        // offset is 64-byte aligned (issue #165: offsets 32 and 2080 on
        // i.MX 95). Rather than decline such a SOURCE into an upload, import
        // from an aligned base and widen by the remainder in pixels; the
        // engine folds the shift into the sampling rectangle through
        // `GlPlatform::import_origin` (issue #170). Applied on every driver,
        // not only Mali: one code path, exercised wherever an unaligned
        // source runs, rather than a Mali-only branch nothing else covers.
        //
        // Sources only. A destination view imports its parent at base 0 and
        // places the tile by viewport, and the driver that does fail on an
        // unaligned destination fails loudly (Vivante, EGL_BAD_ACCESS),
        // which `Platform::dst_import_places` already lowers.
        //
        // `resolve_source_plane0` owns the whole decision, and the Mali gate
        // in `processor/mod.rs` asks the same function what offset this
        // import will present. That shared answer is what makes the gate
        // safe to narrow: a source it exempts is one this call rebased onto
        // an aligned base, never merely one whose format is usually
        // rebasable. When the rebase cannot be applied -- an offset that is
        // not a whole number of pixels, or a pitch with no room to widen --
        // the import keeps its own unaligned offset, the drivers that read
        // it correctly go on doing so, and Mali declines it upstream.
        let (plane0_offset, x_shift_px) = if for_dst {
            (plane0_offset, 0)
        } else {
            resolve_source_plane0(src_fmt, width, plane0_pitch, plane0_offset)
        };
        let width = width + x_shift_px as usize;

        // Semi-planar YUV (NV12) carries a second (interleaved CbCr) plane.
        // NV12 (4:2:0): H/2 rows, W bytes/row (W/2 CbCr pairs).
        // Luma (plane 0) is full-resolution W×H, so the contiguous UV
        // offset is `plane0_offset + plane0_pitch * height`.
        let plane1 = if src_fmt == PixelFormat::Nv12 {
            // NV12 (4:2:0): ceil(H/2) chroma rows for odd-height images.
            let chroma_h = height.div_ceil(2);
            let default_chroma_pitch = plane0_pitch;
            let (plane1_fd, uv_offset) = if let Some(chroma_fd) = uv_fd {
                // Multiplane: UV in separate DMA-BUF — use chroma's plane_offset or 0
                let chroma_offset = src.chroma().and_then(|c| c.plane_offset()).unwrap_or(0);
                (chroma_fd, chroma_offset)
            } else {
                // Contiguous: UV follows the full-height Y plane in the same
                // buffer. Use stride-aware offset — if Y has padding, UV starts
                // at stride * height. Include the luma plane_offset so the UV
                // base is correct when pixel data does not start at byte 0.
                (fd, plane0_offset + plane0_pitch * height)
            };
            let plane1_pitch = if let Some(chroma) = src.chroma() {
                // Multiplane: use chroma's explicit stride if set (via
                // set_row_stride_unchecked during import), else the
                // subsampling-derived default.
                chroma
                    .effective_row_stride()
                    .unwrap_or(default_chroma_pitch)
            } else {
                default_chroma_pitch
            };
            // Validate that the chroma offset + required data fits within the
            // chroma fd's buffer.  Catches client bugs where the wrong fd is
            // used for the UV plane (e.g. Y-only fd with vmeta global offset)
            // and produces a clear error instead of an opaque EGL(BadAlloc).
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

    /// Build `DmaImportAttrs` for importing an NV* semi-planar buffer as a
    /// **single-plane R8** EGLImage (Path B, GL ES 3.0 texelFetch shader).
    ///
    /// The combined buffer (luma + chroma rows stacked) is imported as one R8
    /// plane of size `(effective_row_stride, luma_h + chroma_h)`.  This is
    /// the Path-B zero-copy import used by `draw_nv_texture_2d` together with
    /// the `generate_nv_to_rgba_shader_2d` shader.
    ///
    /// Returns the combined height and `uv_row_bytes` as extra fields via the
    /// existing `DmaImportAttrs` struct:
    ///   * `width`  = `effective_row_stride()` (even buffer width)
    ///   * `height` = combined (luma + chroma) height
    ///   * `drm_fourcc` = `DrmFourcc::R8`
    ///   * `plane1` = `None`  (single-plane import)
    ///
    /// Requires a contiguous DMA buffer (not multiplane): the R8 import maps
    /// the entire Y+UV region as one flat texture.
    pub fn from_tensor_nv_r8(src: &Tensor<u8>, src_fmt: PixelFormat) -> Result<Self, Error> {
        let src_w = src.width().ok_or(Error::NotAnImage)?;
        let src_h = src.height().ok_or(Error::NotAnImage)?;

        if !matches!(
            src_fmt,
            PixelFormat::Nv12 | PixelFormat::Nv16 | PixelFormat::Nv24
        ) {
            return Err(Error::NotSupported(format!(
                "from_tensor_nv_r8 only supports NV12/NV16/NV24, got {src_fmt:?}"
            )));
        }

        if src.is_multiplane() {
            // Path B requires the Y and UV regions in the same DMA-BUF.
            // Multiplane tensors have independent buffers; fall back to Path A.
            return Err(Error::NotSupported(
                "from_tensor_nv_r8: multiplane tensor not supported; use 2-plane Path A".into(),
            ));
        }

        // Use the effective row stride as the R8 texture width so padding bytes
        // are included in the linear addressing used by the shader. The stride is
        // always even and 64-byte aligned (`Tensor::image`), so this R8 import is
        // accepted on every tested GPU — odd LOGICAL dimensions import fine once
        // the buffer stride is 64-aligned (verified via gpu-probe on Mali,
        // Vivante, and V3D). Keeping width == stride also avoids clipping NV24's
        // stride-wrapped chroma on V3D/Tegra.
        let tex_width = src.effective_row_stride().unwrap_or(src_w);

        // Combined (luma + chroma) height — mirrors the PlanarRgb R8 import
        // (`src_h * 3` for 3-channel planar; here it is format-dependent).
        // Combined (luma + chroma) buffer height, matching `PixelFormat::allocation_shape`:
        //   NV12 (4:2:0): H luma + H/2 chroma          = H + ⌈H/2⌉
        //   NV16 (4:2:2): H luma + H chroma (W bytes/row) = 2H
        //   NV24 (4:4:4): H luma + 2H chroma (2W bytes/row laid out as 2H rows of W) = 3H
        // The shader addresses chroma at byte `H*tex_width + ...`, reaching row
        // 3H-1 for NV24, so the imported R8 texture MUST be 3H tall — a 2H import
        // leaves the NV24 chroma rows out of bounds (reads garbage on strict tiled
        // GPUs like Mali/V3D, tolerated on Vivante/Tegra).
        let combined_h = src_fmt.combined_plane_height(src_h).ok_or_else(|| {
            Error::NotImplemented(format!("Path-B R8 import: {src_fmt:?} is not semi-planar"))
        })?;

        // See the EGLImage import above for why this is `dmabuf()` and not
        // `as_dma().fd`.
        let fd = src
            .dmabuf()
            .map_err(|e| {
                Error::NotImplemented(format!(
                    "OpenGL Path-B R8 import requires DMA tensor, got {:?} ({e})",
                    src.memory()
                ))
            })?
            .as_raw_fd();
        let plane0_offset = src.plane_offset().unwrap_or(0);

        Ok(DmaImportAttrs {
            width: tex_width,
            height: combined_h,
            drm_fourcc: DrmFourcc::R8,
            plane0_fd: fd,
            plane0_pitch: tex_width,
            plane0_offset,
            // The R8 import's pitch IS its width, so it cannot widen: an
            // unaligned NV source keeps the Stage F decline and uploads
            // (issue #170).
            x_shift_px: 0,
            plane1: None,
            is_yuv: false, // R8 is not a multi-plane YUV fourcc; no driver hints
            // Path B applies the YUV→RGB matrix in-shader (per-tensor coeffs),
            // so the driver EGL color-space/range hints are unused here.
            yuv_encoding: ColorEncoding::Bt709,
            yuv_range: ColorRange::Limited,
        })
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
            // Drive the driver's YUV→RGB matrix + range from the resolved
            // per-tensor colorimetry rather than a fixed BT.709/limited.
            let cs_hint = match self.yuv_encoding {
                ColorEncoding::Bt601 => egl_ext::ITU_REC601,
                ColorEncoding::Bt709 => egl_ext::ITU_REC709,
                ColorEncoding::Bt2020 => egl_ext::ITU_REC2020,
                // Future encodings default to BT.709 (HD broadcast).
                _ => egl_ext::ITU_REC709,
            };
            let range_hint = match self.yuv_range {
                ColorRange::Full => egl_ext::YUV_FULL_RANGE,
                ColorRange::Limited => egl_ext::YUV_NARROW_RANGE,
                // Future ranges default to narrow (broadcast convention).
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
    // These require real DMA-BUF allocations, gated on dma_test_formats.

    #[cfg(feature = "dma_test_formats")]
    use edgefirst_tensor::{is_dma_available, PlaneDescriptor, TensorMemory};

    /// Helper: allocate a DMA tensor of the given byte count, or return None if unavailable.
    #[cfg(feature = "dma_test_formats")]
    fn alloc_dma(bytes: usize, name: &str) -> Option<Tensor<u8>> {
        if !is_dma_available() {
            return None;
        }
        match Tensor::<u8>::new(&[bytes], Some(TensorMemory::DmaBuf), Some(name)) {
            Ok(t) if t.memory() == TensorMemory::DmaBuf => Some(t),
            _ => None,
        }
    }

    /// Helper: import an NV12 image via the public ImageProcessor API and
    /// return the resulting tensor for inspection.
    #[cfg(feature = "dma_test_formats")]
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

    /// A SOURCE view imports its OWN region (own dims + offset) so `convert()`
    /// samples the right pixels; a DESTINATION view imports the PARENT (the tile
    /// is a `glViewport` ROI). Guards the `for_dst` distinction — without it a
    /// source view would import the parent but sample using the view's dims,
    /// reading the wrong region.
    #[test]
    #[cfg(feature = "dma_test_formats")]
    fn from_tensor_source_view_imports_own_region_dst_imports_parent() {
        if !is_dma_available() {
            crate::test_support::report_skip("from_tensor_source_view... - DMA not available");
            return;
        }
        // 64x64 RGBA with a padded 320-byte stride (tight row = 64*4 = 256).
        let parent = match Tensor::<u8>::image_with_stride(
            64,
            64,
            PixelFormat::Rgba,
            320,
            Some(TensorMemory::DmaBuf),
            edgefirst_tensor::CpuAccess::ReadWrite,
        ) {
            Ok(t) => t,
            Err(_) => {
                crate::test_support::report_skip("image_with_stride DMA unavailable");
                return;
            }
        };
        let view = parent
            .view(edgefirst_tensor::Region::new(8, 8, 32, 16))
            .unwrap();

        // SOURCE (for_dst = false): import the view's OWN region -- rebased
        // onto an aligned base, because (8, 8) at a 320-byte pitch is byte
        // 2592, which is not 64-byte aligned. The base is 2560 and the
        // 32-byte remainder is 8 RGBA pixels of shift, so the import is 8
        // texels wider than the view and starts it 8 texels in (issue #170).
        // The region is deliberately left at an unaligned origin: the
        // rebased numbers say more about what this test guards than an
        // aligned origin that exercises nothing.
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

        // DESTINATION (for_dst = true): import the PARENT.
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

    /// True multiplane: separate DMA-BUFs for Y and UV (libcamera style).
    ///
    /// Verifies that plane0_fd and plane1.fd are different raw fds pointing
    /// to different underlying DMA-BUFs, and that offsets default to 0.
    #[test]
    #[cfg(feature = "dma_test_formats")]
    fn test_nv12_true_multiplane_attrs() {
        let width: usize = 1920;
        let height: usize = 1088;
        let stride: usize = 1920;
        let luma_bytes = stride * height;
        let chroma_bytes = stride * height.div_ceil(2);

        let luma_buf = match alloc_dma(luma_bytes, "luma_buf") {
            Some(t) => t,
            None => {
                crate::test_support::report_skip(
                    "test_nv12_true_multiplane_attrs - DMA not available",
                );
                return;
            }
        };
        let chroma_buf = match alloc_dma(chroma_bytes, "chroma_buf") {
            Some(t) => t,
            None => {
                crate::test_support::report_skip(
                    "test_nv12_true_multiplane_attrs - DMA alloc failed",
                );
                return;
            }
        };

        let luma_fd = luma_buf.dmabuf().unwrap();
        let chroma_fd = chroma_buf.dmabuf().unwrap();
        let luma_pd = PlaneDescriptor::new(luma_fd).unwrap().with_stride(stride);
        let chroma_pd = PlaneDescriptor::new(chroma_fd).unwrap().with_stride(stride);

        let tensor = import_nv12(luma_pd, Some(chroma_pd), width, height).unwrap();
        let tensor_u8 = tensor.as_u8().unwrap();

        let attrs = DmaImportAttrs::from_tensor(tensor_u8, PixelFormat::Nv12, false).unwrap();

        // plane0 and plane1 should have DIFFERENT fds (separate DMA-BUFs)
        let p1 = attrs.plane1.as_ref().expect("NV12 must have plane1");
        assert_ne!(
            attrs.plane0_fd, p1.fd,
            "true multiplane: plane0_fd and plane1_fd must be different"
        );

        // Offsets should be 0 (no explicit offset set, each plane starts at its buffer origin)
        assert_eq!(attrs.plane0_offset, 0, "luma offset must be 0");
        assert_eq!(p1.offset, 0, "chroma offset must be 0 for true multiplane");

        // Pitches should match the set stride
        assert_eq!(attrs.plane0_pitch, stride);
        assert_eq!(p1.pitch, stride);

        // Dimensions
        assert_eq!(attrs.width, width);
        assert_eq!(attrs.height, height);
        assert!(attrs.is_yuv);
        assert_eq!(attrs.drm_fourcc, DrmFourcc::Nv12);
    }

    /// Same-fd multiplane: both planes reference the same DMA-BUF via dup'd fds
    /// with chroma at an explicit offset (V4L2 / GStreamer style).
    ///
    /// This is the scenario that causes EGL(BadAlloc) on Mali G310 — two
    /// PLANE descriptors with different raw fds that resolve to the same
    /// underlying DMA-BUF.
    #[test]
    #[cfg(feature = "dma_test_formats")]
    fn test_nv12_same_fd_multiplane_attrs() {
        let width: usize = 1920;
        let height: usize = 1088;
        let stride: usize = 1920;
        let luma_size = stride * height;
        let chroma_size = stride * height.div_ceil(2);
        let total_bytes = luma_size + chroma_size;

        let buf = match alloc_dma(total_bytes, "shared_buf") {
            Some(t) => t,
            None => {
                crate::test_support::report_skip(
                    "test_nv12_same_fd_multiplane_attrs - DMA not available",
                );
                return;
            }
        };

        let fd = buf.dmabuf().unwrap();
        // Both descriptors dup the SAME fd — different raw fd values, same buffer
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

        // plane0 and plane1 have DIFFERENT raw fds (each is a dup), but
        // they reference the same underlying DMA-BUF.
        assert_ne!(
            attrs.plane0_fd, p1.fd,
            "same-fd multiplane: raw fds must differ (each is a dup)"
        );

        // Luma starts at 0, chroma at luma_size
        assert_eq!(attrs.plane0_offset, 0);
        assert_eq!(
            p1.offset, luma_size,
            "chroma offset must be luma_size for same-fd multiplane"
        );

        // Both pitches match the stride
        assert_eq!(attrs.plane0_pitch, stride);
        assert_eq!(p1.pitch, stride);
        assert_eq!(attrs.width, width);
        assert_eq!(attrs.height, height);
    }

    /// Phase 5: even-but-not-4-aligned NV12 width imports (relaxed `%4→%2`
    /// gate); odd width is still rejected (4:2:0 chroma is W/2 — needs even W).
    /// The import uses the 64-aligned `plane0_pitch`, not `width`, so the old
    /// word-alignment rationale for `%4` no longer applies.
    #[test]
    #[cfg(feature = "dma_test_formats")]
    fn test_nv12_even_width_relaxed_gate() {
        let (height, stride) = (64usize, 128usize);
        let total = stride * height + stride * height.div_ceil(2);
        let buf = match alloc_dma(total, "nv12_even_w") {
            Some(t) => t,
            None => {
                crate::test_support::report_skip(
                    "test_nv12_even_width_relaxed_gate - DMA not available",
                );
                return;
            }
        };

        // 66 is even but NOT 4-aligned — previously rejected by the width%4 gate.
        let luma_pd = PlaneDescriptor::new(buf.dmabuf().unwrap())
            .unwrap()
            .with_stride(stride);
        let t66 = import_nv12(luma_pd, None, 66, height).unwrap();
        let attrs = DmaImportAttrs::from_tensor(t66.as_u8().unwrap(), PixelFormat::Nv12, false)
            .expect("even-but-not-4-aligned NV12 width must import after the %4→%2 relaxation");
        assert_eq!(attrs.width, 66);

        // Odd width still rejected (4:2:0 chroma needs even width).
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

    /// Contiguous single-fd: no chroma descriptor, UV follows Y in buffer.
    ///
    /// Plane1 fd must be the EXACT SAME raw fd as plane0 (not a dup), and
    /// the UV offset is computed as plane0_offset + plane0_pitch * height.
    #[test]
    #[cfg(feature = "dma_test_formats")]
    fn test_nv12_contiguous_single_fd_attrs() {
        let width: usize = 1920;
        let height: usize = 1088;
        let stride: usize = 1920;
        let total_h = height * 3 / 2; // NV12: Y + UV/2
        let total_bytes = stride * total_h;

        let buf = match alloc_dma(total_bytes, "contiguous_buf") {
            Some(t) => t,
            None => {
                crate::test_support::report_skip(
                    "test_nv12_contiguous_single_fd_attrs - DMA not available",
                );
                return;
            }
        };

        let fd = buf.dmabuf().unwrap();
        let luma_pd = PlaneDescriptor::new(fd).unwrap().with_stride(stride);

        // No chroma descriptor — single contiguous buffer
        let tensor = import_nv12(luma_pd, None, width, height).unwrap();
        let tensor_u8 = tensor.as_u8().unwrap();

        let attrs = DmaImportAttrs::from_tensor(tensor_u8, PixelFormat::Nv12, false).unwrap();
        let p1 = attrs.plane1.as_ref().expect("NV12 must have plane1");

        // Contiguous: plane1_fd must be the EXACT SAME raw fd as plane0
        assert_eq!(
            attrs.plane0_fd, p1.fd,
            "contiguous: plane0_fd and plane1_fd must be the same raw fd"
        );

        // UV offset computed from luma geometry
        let expected_uv_offset = stride * height;
        assert_eq!(attrs.plane0_offset, 0);
        assert_eq!(
            p1.offset, expected_uv_offset,
            "contiguous UV offset must be stride * height"
        );

        // Both pitches match
        assert_eq!(attrs.plane0_pitch, stride);
        assert_eq!(p1.pitch, stride);
    }

    /// Contiguous single-fd with padded stride.
    ///
    /// When the stride > width, the UV offset must use stride (not width).
    #[test]
    #[cfg(feature = "dma_test_formats")]
    fn test_nv12_contiguous_padded_stride_attrs() {
        let width: usize = 1920;
        let height: usize = 1080;
        let stride: usize = 2048; // padded to 2048-byte alignment
        let total_h = height * 3 / 2;
        let total_bytes = stride * total_h;

        let buf = match alloc_dma(total_bytes, "padded_buf") {
            Some(t) => t,
            None => {
                crate::test_support::report_skip(
                    "test_nv12_contiguous_padded_stride_attrs - DMA not available",
                );
                return;
            }
        };

        let fd = buf.dmabuf().unwrap();
        let luma_pd = PlaneDescriptor::new(fd).unwrap().with_stride(stride);

        let tensor = import_nv12(luma_pd, None, width, height).unwrap();
        let tensor_u8 = tensor.as_u8().unwrap();

        let attrs = DmaImportAttrs::from_tensor(tensor_u8, PixelFormat::Nv12, false).unwrap();
        let p1 = attrs.plane1.as_ref().unwrap();

        // UV offset must use stride, not width
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
    #[cfg(feature = "dma_test_formats")]
    fn test_nv12_multiplane_padded_strides_attrs() {
        let width: usize = 1920;
        let height: usize = 1080;
        let luma_stride: usize = 2048;
        let chroma_stride: usize = 2048;
        let luma_bytes = luma_stride * height;
        let chroma_bytes = chroma_stride * height.div_ceil(2);

        let luma_buf = match alloc_dma(luma_bytes, "luma_padded") {
            Some(t) => t,
            None => {
                crate::test_support::report_skip(
                    "test_nv12_multiplane_padded_strides_attrs - DMA not available",
                );
                return;
            }
        };
        let chroma_buf = match alloc_dma(chroma_bytes, "chroma_padded") {
            Some(t) => t,
            None => {
                crate::test_support::report_skip("DMA alloc failed");
                return;
            }
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
        // Multiplane: offsets are per-plane (both 0 here)
        assert_eq!(attrs.plane0_offset, 0);
        assert_eq!(p1.offset, 0);
    }

    /// Same-fd multiplane with non-zero luma offset.
    ///
    /// Some V4L2 drivers place the luma plane at a non-zero offset in the
    /// DMA-BUF (e.g. after a metadata header).
    #[test]
    #[cfg(feature = "dma_test_formats")]
    fn test_nv12_same_fd_nonzero_luma_offset() {
        let width: usize = 640;
        let height: usize = 480;
        let stride: usize = 640;
        let luma_offset: usize = 4096; // metadata header before luma
        let luma_size = stride * height;
        let chroma_offset = luma_offset + luma_size;
        let total_bytes = chroma_offset + stride * height.div_ceil(2);

        let buf = match alloc_dma(total_bytes, "offset_buf") {
            Some(t) => t,
            None => {
                crate::test_support::report_skip(
                    "test_nv12_same_fd_nonzero_luma_offset - DMA not available",
                );
                return;
            }
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

    /// Oversized chroma offset: chroma offset at/past the buffer end.
    ///
    /// Reproduces the v4l2h264dec bug: the cameraadaptor passes the Y
    /// plane's fd for the UV chroma descriptor with vmeta global offset
    /// (stride × aligned_height).  The Y fd's buffer only covers the Y
    /// plane, so the offset exceeds the buffer → must return a clear error
    /// instead of passing through to EGL (which returns opaque BadAlloc).
    #[test]
    #[cfg(feature = "dma_test_formats")]
    fn test_nv12_chroma_offset_exceeds_buffer() {
        let width: usize = 1920;
        let height: usize = 1088;
        let stride: usize = 1920;
        let y_size = stride * height; // 2,088,960 — also the vmeta global UV offset
        let chroma_h = height.div_ceil(2);
        let _chroma_size = stride * chroma_h;

        // Allocate a buffer sized for Y ONLY (not Y+UV)
        let y_buf = match alloc_dma(y_size, "y_only_buf") {
            Some(t) => t,
            None => {
                crate::test_support::report_skip(
                    "test_nv12_chroma_offset_exceeds_buffer - DMA not available",
                );
                return;
            }
        };

        let fd = y_buf.dmabuf().unwrap();
        let luma_pd = PlaneDescriptor::new(fd)
            .unwrap()
            .with_stride(stride)
            .with_offset(0);
        // Bug scenario: chroma uses same fd as Y but with offset = y_size
        let chroma_pd = PlaneDescriptor::new(fd)
            .unwrap()
            .with_stride(stride)
            .with_offset(y_size);

        let result = import_nv12(luma_pd, Some(chroma_pd), width, height);
        // The import itself succeeds (just stores metadata)
        let tensor = result.unwrap();
        let tensor_u8 = tensor.as_u8().unwrap();

        // from_tensor must detect that the chroma offset exceeds the buffer
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
    // These test the attribute list serialization with synthetic values
    // (no DMA allocation needed).

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
                fd: 11, // different fd
                pitch: 1920,
                offset: 0,
            }),
            is_yuv: true,
            yuv_encoding: ColorEncoding::Bt709,
            yuv_range: ColorRange::Limited,
        };

        let egl = attrs.to_egl_attribs();

        // Verify PLANE0 attributes
        assert_eq!(egl_attrib_value(&egl, egl_ext::DMA_BUF_PLANE0_FD), Some(10));
        assert_eq!(
            egl_attrib_value(&egl, egl_ext::DMA_BUF_PLANE0_PITCH),
            Some(1920)
        );
        assert_eq!(
            egl_attrib_value(&egl, egl_ext::DMA_BUF_PLANE0_OFFSET),
            Some(0)
        );

        // Verify PLANE1 attributes — fd must differ from PLANE0
        assert_eq!(egl_attrib_value(&egl, egl_ext::DMA_BUF_PLANE1_FD), Some(11));
        assert_eq!(
            egl_attrib_value(&egl, egl_ext::DMA_BUF_PLANE1_PITCH),
            Some(1920)
        );
        assert_eq!(
            egl_attrib_value(&egl, egl_ext::DMA_BUF_PLANE1_OFFSET),
            Some(0)
        );

        // YUV hints present
        assert_eq!(
            egl_attrib_value(&egl, egl_ext::YUV_COLOR_SPACE_HINT),
            Some(egl_ext::ITU_REC709 as Attrib)
        );
        assert_eq!(
            egl_attrib_value(&egl, egl_ext::SAMPLE_RANGE_HINT),
            Some(egl_ext::YUV_NARROW_RANGE as Attrib)
        );

        // Terminated with NONE
        assert_eq!(*egl.last().unwrap(), edgefirst_egl::NONE as Attrib);
    }

    /// Verify EGL attribute list for same-fd multiplane NV12.
    ///
    /// Both PLANE0_FD and PLANE1_FD are different raw fd values (dup'd)
    /// but reference the same DMA-BUF. The chroma has a non-zero offset.
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
    ///
    /// PLANE0_FD and PLANE1_FD must be the exact same raw fd value.
    /// UV offset = pitch × height.
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
                fd: 10, // SAME raw fd
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

        // No PLANE1 attributes
        assert_eq!(egl_attrib_value(&egl, egl_ext::DMA_BUF_PLANE1_FD), None);
        assert_eq!(egl_attrib_value(&egl, egl_ext::DMA_BUF_PLANE1_PITCH), None);
        assert_eq!(egl_attrib_value(&egl, egl_ext::DMA_BUF_PLANE1_OFFSET), None);

        // No YUV hints
        assert_eq!(egl_attrib_value(&egl, egl_ext::YUV_COLOR_SPACE_HINT), None);

        // Terminated
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
                offset: stride * height, // NOT width * height
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
    // The EGL color-space + sample-range hints are selected from the
    // resolved per-tensor colorimetry carried on `DmaImportAttrs`. These
    // build the attr Vec only (no GPU), so they're the testable surface
    // for the Linux EGL path.

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

    // ─── Width-alignment pre-check (from_tensor) regression tests ───────

    /// RGBA at a non-4-aligned width (e.g. 375, 427, 443) must pass the
    /// `DmaImportAttrs::from_tensor` pre-check — RGBA's natural pitch is
    /// `width × 4` which is always 4-byte aligned at any width, and the
    /// DRM fourcc ABGR8888 spec imposes no pixel-alignment requirement.
    ///
    /// This unlocks the zero-copy EGL path for dataset-loader widths that
    /// previously fell back to CPU texture upload.
    #[cfg(feature = "dma_test_formats")]
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
            let t = match Tensor::<u8>::image_with_stride(
                w,
                64,
                PixelFormat::Rgba,
                aligned,
                Some(edgefirst_tensor::TensorMemory::DmaBuf),
                edgefirst_tensor::CpuAccess::ReadWrite,
            ) {
                Ok(t) => t,
                Err(e) => {
                    crate::test_support::report_skip(&format!(
                        "image_with_stride failed at width {w}: {e}"
                    ));
                    return;
                }
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

    /// BGRA must also accept non-4-aligned widths (same reasoning as RGBA —
    /// 4-bpp packed, pitch always 4-byte aligned).
    #[cfg(feature = "dma_test_formats")]
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
        let t = match Tensor::<u8>::image_with_stride(
            375,
            64,
            PixelFormat::Bgra,
            aligned,
            Some(edgefirst_tensor::TensorMemory::DmaBuf),
            edgefirst_tensor::CpuAccess::ReadWrite,
        ) {
            Ok(t) => t,
            Err(e) => {
                crate::test_support::report_skip(&format!("image_with_stride failed: {e}"));
                return;
            }
        };
        DmaImportAttrs::from_tensor(&t, PixelFormat::Bgra, false)
            .expect("BGRA width 375 must pass the pre-check");
    }

    /// 3-bpp (Rgb) still requires width%4 — its natural pitch is
    /// `width × 3`, which is 4-byte aligned only when `width % 4 == 0`.
    #[test]
    fn test_from_tensor_rejects_non_4_aligned_rgb() {
        // We need a tensor with width=375 to hit the pre-check. Mem-backed
        // works for this — from_tensor only reads width/height/memory
        // metadata and the pre-check fires before the DMA fd lookup.
        let t = Tensor::<u8>::image(
            375,
            64,
            PixelFormat::Rgb,
            Some(edgefirst_tensor::TensorMemory::Mem),
            edgefirst_tensor::CpuAccess::ReadWrite,
        )
        .unwrap();
        let err = DmaImportAttrs::from_tensor(&t, PixelFormat::Rgb, false)
            .expect_err("RGB width 375 must still be rejected");
        match err {
            crate::Error::NotSupported(msg) => {
                assert!(msg.contains("divisible by 4"), "got: {msg}");
            }
            other => panic!("expected NotSupported, got {other:?}"),
        }
    }

    /// 1-bpp (Grey) likewise still requires width%4 — natural pitch is
    /// `width × 1`, which is 4-byte aligned only when `width % 4 == 0`.
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

    // ─── aligned_source_base tests ───────────────────────────────────

    /// The offsets the i.MX 95 probe classified (issue #165), at RGBA's
    /// 4 bytes per pixel: 32 and 2080 sampled zeros, 0/64/256/2048 were fine.
    /// `lcm(64, 4)` is 64, so the base is just the 64-byte floor and the
    /// remainder is always a whole number of RGBA pixels.
    #[test]
    fn aligned_source_base_rgba_floors_to_64_and_shifts_whole_pixels() {
        let f = |o| super::aligned_source_base(o, 4, 64);
        // Aligned offsets are untouched: no rebase, no shift, so every
        // driver sees exactly the import it sees today.
        assert_eq!(f(0), Some((0, 0)));
        assert_eq!(f(64), Some((64, 0)));
        assert_eq!(f(256), Some((256, 0)));
        assert_eq!(f(2048), Some((2048, 0)));
        // The two that sampled zeros. 32 is 8 RGBA pixels past base 0;
        // 2080 is 8 past base 2048 -- the (8, 0) and (8, 8) view origins.
        assert_eq!(f(32), Some((0, 8)));
        assert_eq!(f(2080), Some((2048, 8)));
    }

    /// Grey is 1 byte per pixel, so the remainder is whole by construction
    /// and every byte of the 64-byte residue becomes a texel of shift.
    #[test]
    fn aligned_source_base_grey_shifts_every_residue_byte() {
        let f = |o| super::aligned_source_base(o, 1, 64);
        assert_eq!(f(0), Some((0, 0)));
        assert_eq!(f(32), Some((0, 32)));
        assert_eq!(f(2080), Some((2048, 32)));
        assert_eq!(f(63), Some((0, 63)));
    }

    /// RGB is the case the `lcm` exists for: 3 does not divide 64, so the
    /// step is `lcm(64, 3) = 192` and the base walks back to a multiple of
    /// BOTH. A 64-byte floor alone would leave a fractional pixel.
    #[test]
    fn aligned_source_base_rgb_steps_by_lcm_192() {
        let f = |o| super::aligned_source_base(o, 3, 64);
        assert_eq!(f(0), Some((0, 0)));
        assert_eq!(f(192), Some((192, 0)));
        // 96 = 32 RGB pixels; base 0, because 96 is not a multiple of 192.
        assert_eq!(f(96), Some((0, 32)));
        // 2079 = 3 * 693. Base 1920 (= 10 * 192, and 1920 % 64 == 0),
        // remainder 159 bytes = 53 whole RGB pixels.
        assert_eq!(f(2079), Some((1920, 53)));
        // A 64-byte floor would have given base 2048 and a 31-byte
        // remainder, which is not a whole number of RGB pixels.
        assert_ne!(f(2079).map(|(b, _)| b), Some(2048));
    }

    /// An offset that is not a whole number of pixels has no aligned base
    /// with a whole-pixel remainder, and the import must decline rather
    /// than fold half a pixel.
    #[test]
    fn aligned_source_base_refuses_a_fractional_pixel_offset() {
        assert_eq!(super::aligned_source_base(2080, 3, 64), None);
        assert_eq!(super::aligned_source_base(2081, 4, 64), None);
        assert_eq!(super::aligned_source_base(1, 2, 64), None);
        // Degenerate inputs are refusals, not panics or divisions by zero.
        assert_eq!(super::aligned_source_base(64, 0, 64), None);
        assert_eq!(super::aligned_source_base(64, 4, 0), None);
    }

    /// The two invariants every caller depends on, over a sweep rather than
    /// a handful of points: the base is import-aligned, and the shift is an
    /// exact whole-pixel distance from it back to the tensor's own offset.
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

    // ─── source rebase onto an aligned base (issue #170) ─────────────

    /// Allocate the DMA parent the rebase tests view into, or `None` when
    /// this machine has no usable DMA heap (the heap is root-only on the
    /// desk, so an unprivileged run skips) or cannot allocate this format.
    #[cfg(feature = "dma_test_formats")]
    fn dma_parent(width: usize, height: usize, fmt: PixelFormat) -> Option<Tensor<u8>> {
        if !is_dma_available() {
            return None;
        }
        match Tensor::<u8>::image(
            width,
            height,
            fmt,
            Some(TensorMemory::DmaBuf),
            edgefirst_tensor::CpuAccess::ReadWrite,
        ) {
            Ok(t) if t.memory() == TensorMemory::DmaBuf => Some(t),
            _ => None,
        }
    }

    /// A source at an unaligned RGBA offset imports from the 64-byte floor,
    /// widened by the remainder in pixels, at the SAME pitch -- so the
    /// tensor's first pixel is texel `x_shift_px` of row 0. This is the
    /// import Mali samples correctly where the unrebased one sampled zeros
    /// (issue #170).
    #[test]
    #[cfg(feature = "dma_test_formats")]
    fn from_tensor_rebases_an_unaligned_rgba_source_to_an_aligned_base() {
        let Some(parent) = dma_parent(64, 64, PixelFormat::Rgba) else {
            use std::io::Write;
            let _ = writeln!(
                &mut std::io::stderr(),
                "SKIPPED: {} - DMA not available",
                function!()
            );
            return;
        };
        let pitch = parent.effective_row_stride().unwrap_or(256);
        // (8, 8) of a 256-byte-pitched RGBA surface: offset 2080, the origin
        // that sampled zeros on i.MX 95.
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

    /// An ALIGNED source is untouched -- no rebase, no shift, no widening --
    /// so every driver sees exactly the import it saw before issue #170.
    #[test]
    #[cfg(feature = "dma_test_formats")]
    fn from_tensor_leaves_an_aligned_source_alone() {
        let Some(parent) = dma_parent(64, 64, PixelFormat::Rgba) else {
            use std::io::Write;
            let _ = writeln!(
                &mut std::io::stderr(),
                "SKIPPED: {} - DMA not available",
                function!()
            );
            return;
        };
        let pitch = parent.effective_row_stride().unwrap_or(256);
        assert_eq!(pitch % 64, 0, "the premise: a 64-aligned parent pitch");
        // (0, 8): offset 8 * pitch, aligned whenever the pitch is.
        let view = parent
            .view(edgefirst_tensor::Region::new(0, 8, 16, 16))
            .unwrap();
        let s = DmaImportAttrs::from_tensor(&view, PixelFormat::Rgba, false).unwrap();
        assert_eq!(s.plane0_offset, 8 * pitch);
        assert_eq!(s.x_shift_px, 0);
        assert_eq!(s.width, 16);
    }

    /// A DESTINATION is never rebased: it imports its parent at base 0 and
    /// places the tile by viewport, and the drivers that fail on the
    /// destination side fail loudly rather than silently (Vivante,
    /// EGL_BAD_ACCESS), which `dst_import_places` already lowers.
    #[test]
    #[cfg(feature = "dma_test_formats")]
    fn from_tensor_never_rebases_a_destination() {
        let Some(parent) = dma_parent(64, 64, PixelFormat::Rgba) else {
            use std::io::Write;
            let _ = writeln!(
                &mut std::io::stderr(),
                "SKIPPED: {} - DMA not available",
                function!()
            );
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

    /// The pitch guard is about ROOM, and the room is the pitch padding.
    /// A pitch that is merely tight -- a foreign DMA-BUF adopted at the
    /// producer's own stride -- has none, and the rebase must be declined
    /// rather than handed to EGL as a row that runs past its pitch.
    #[test]
    fn rebase_fits_pitch_declines_a_tight_foreign_pitch() {
        let fits = super::rebase_fits_pitch;
        // 100 RGBA pixels at the producer's tight 400-byte stride: the row
        // already fills the pitch, so even one pixel of shift runs past it.
        assert!(!fits(100, 1, 4, 400));
        assert!(!fits(100, 8, 4, 400));
        // No shift is the unrebased import, which always fits.
        assert!(fits(100, 0, 4, 400));
        // The same width at the 64-padded pitch a HAL allocation gives it
        // (448) has room for a shift, but only up to the padding: 12 pixels
        // is 448 bytes exactly, 13 is 452 and runs past.
        assert!(fits(100, 12, 4, 448));
        assert!(!fits(100, 13, 4, 448));
        // The shape the pin test exercises: a 16-pixel view shifted 8
        // pixels inside a 256-byte parent pitch, with room to spare.
        assert!(fits(16, 8, 4, 256));
        // Overflow is "does not fit", never a wrap or a panic.
        assert!(!fits(usize::MAX, 1, 4, usize::MAX));
    }

    // ─── which formats rebase (issue #170) ───────────────────────────

    /// `Rgb` is 3 bytes per pixel, so its step is `lcm(64, 3) = 192` rather
    /// than 64 and the base walks back further than a 64-byte floor would.
    /// Asserted against the offset the view actually reports, so a board
    /// whose pitch padding differs is still pinned rather than skipped.
    #[test]
    #[cfg(feature = "dma_test_formats")]
    fn from_tensor_rebases_an_unaligned_rgb_source_by_the_lcm_step() {
        let Some(parent) = dma_parent(64, 64, PixelFormat::Rgb) else {
            use std::io::Write;
            let _ = writeln!(
                &mut std::io::stderr(),
                "SKIPPED: {} - DMA not available",
                function!()
            );
            return;
        };
        let pitch = parent.effective_row_stride().unwrap_or(192);
        let view = parent
            .view(edgefirst_tensor::Region::new(8, 8, 16, 16))
            .unwrap();
        // (8, 8) of a 192-byte-pitched RGB surface is offset 1560: base
        // 1536 (8 * 192), shift 8 pixels = 24 bytes.
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

    /// `Grey` is 1 byte per pixel, so every residue byte is a texel of
    /// shift and the step is just the 64-byte alignment.
    #[test]
    #[cfg(feature = "dma_test_formats")]
    fn from_tensor_rebases_an_unaligned_grey_source_byte_for_texel() {
        let Some(parent) = dma_parent(64, 64, PixelFormat::Grey) else {
            use std::io::Write;
            let _ = writeln!(
                &mut std::io::stderr(),
                "SKIPPED: {} - DMA not available",
                function!()
            );
            return;
        };
        let pitch = parent.effective_row_stride().unwrap_or(64);
        let view = parent
            .view(edgefirst_tensor::Region::new(8, 8, 16, 16))
            .unwrap();
        // (8, 8) of a 64-byte-pitched Grey surface is offset 520: base 512,
        // shift 8 bytes = 8 pixels.
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

    // ─── the gate and the import read the same answer ────────────────

    /// The four shapes the Mali gate has to tell apart, as a pure table.
    /// Keyed on the RESOLVED offset, so a packed format is exempt only when
    /// the fold actually moved it -- which is the whole point of issue
    /// #170's fix round: a format-keyed exemption waves the last two
    /// through into a black frame on Mali.
    #[test]
    fn the_resolved_offset_tells_the_four_shapes_apart() {
        // The gate's question, in the production resolver's terms.
        let aligned = |fmt, w, pitch, off| {
            super::resolve_source_plane0(fmt, w, pitch, off)
                .0
                .is_multiple_of(DMA_IMPORT_OFFSET_ALIGN)
        };

        // Rebased: a 16-pixel view 8 pixels into a 256-byte-pitched RGBA
        // parent, at the offset i.MX 95 sampled zeros from. Room to widen
        // ((16 + 8) * 4 = 96 <= 256), so the base is 2048 and it is exempt.
        assert!(aligned(PixelFormat::Rgba, 16, 256, 2080));
        // Already aligned: exempt without being touched.
        assert!(aligned(PixelFormat::Rgba, 64, 256, 2048));
        // Tight pitch, no room: a whole 64-wide RGBA tensor at a
        // `set_plane_offset(32)` window. Its stride is 256 because 256 is
        // already 64-aligned and nothing was padded on, so the widened row
        // would be (64 + 8) * 4 = 288 > 256. NOT exempt.
        assert!(!aligned(PixelFormat::Rgba, 64, 256, 32));
        // Stride not a whole pixel count: a 100-wide RGB surface pads to
        // 320 bytes, which is not a multiple of 3, so row 1 column 1 is at
        // 323 -- unaligned and not a whole number of pixels. NOT exempt.
        assert!(!aligned(PixelFormat::Rgb, 16, 320, 323));
    }

    /// The anti-drift pin the gate's safety rests on: for every shape, the
    /// offset `resolved_source_plane0_offset` reports (what the gate reads)
    /// is byte-for-byte the `plane0_offset` `from_tensor` puts in the import
    /// (what EGL is handed). If these two ever disagree the gate is deciding
    /// about an import that does not exist -- declining a good one, or
    /// waving through one that comes back black.
    #[test]
    #[cfg(feature = "dma_test_formats")]
    fn the_gate_reads_the_offset_the_import_presents() {
        if !is_dma_available() {
            use std::io::Write;
            let _ = writeln!(
                &mut std::io::stderr(),
                "SKIPPED: {} - DMA not available",
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
            // An aligned window offset, which must stay put.
            (PixelFormat::Rgba, 64, 64, Some(64)),
            // Rgb, whose 64-padded stride is not a multiple of 3.
            (PixelFormat::Rgb, 100, 8, Some(323)),
            // A format the fold never covers.
            (PixelFormat::Nv12, 64, 64, Some(32)),
        ];
        // The whole test already returned above when there is no DMA heap,
        // so on a host that reached here every case must be allocatable and
        // every case must be compared. Counting and asserting the full total
        // (rather than merely "more than none") is what stops this pin from
        // going quietly vacuous if a format stops allocating.
        let total = cases.len();
        let mut compared = 0usize;
        for (fmt, w, h, whole_offset) in cases {
            let Some(mut parent) = dma_parent(w, h, fmt) else {
                use std::io::Write;
                let _ = writeln!(
                    &mut std::io::stderr(),
                    "SKIPPED: {} {fmt:?} - not allocatable here",
                    function!()
                );
                continue;
            };
            if fmt == PixelFormat::Rgb {
                // The premise of this row: a 100-wide RGB surface pads its
                // 300-byte row to 320, which is not a multiple of 3, so byte
                // 323 is row 1 column 1 and is NOT a whole number of pixels
                // from the start. Asserted rather than assumed -- a board
                // that padded differently would quietly turn this row into
                // an ordinary aligned case.
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

    /// A whole 64-wide RGBA tensor at a `set_plane_offset` window is the
    /// shape a format-keyed Mali exemption got wrong: its stride is tight at
    /// 256 bytes (64 * 4 is already 64-aligned, so nothing was padded on),
    /// leaving no room to widen, so the import keeps its own unaligned
    /// offset and Mali must go on declining it.
    #[test]
    #[cfg(feature = "dma_test_formats")]
    fn from_tensor_cannot_rebase_a_window_into_a_tight_pitch() {
        let Some(mut t) = dma_parent(64, 64, PixelFormat::Rgba) else {
            use std::io::Write;
            let _ = writeln!(
                &mut std::io::stderr(),
                "SKIPPED: {} - DMA not available",
                function!()
            );
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

    /// The formats the fold does NOT cover import at their own offset with
    /// no shift, however unaligned that offset is: YUYV/VYUY carry a
    /// 2-pixel macropixel phase a texel shift would break, NV12's second
    /// plane has an offset of its own, and planar RGB stacks its planes by
    /// row. The Mali gate in the processor must keep declining these, which
    /// is only sound while they are not rebased here.
    #[test]
    #[cfg(feature = "dma_test_formats")]
    fn from_tensor_never_rebases_the_formats_the_fold_misses() {
        for fmt in [PixelFormat::Yuyv, PixelFormat::Nv12, PixelFormat::PlanarRgb] {
            let Some(mut t) = dma_parent(64, 64, fmt) else {
                use std::io::Write;
                let _ = writeln!(
                    &mut std::io::stderr(),
                    "SKIPPED: {} {fmt:?} - not allocatable here",
                    function!()
                );
                continue;
            };
            // 32 is one of the two offsets i.MX 95 sampled zeros from, and
            // is unaligned for every one of these formats.
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
}
