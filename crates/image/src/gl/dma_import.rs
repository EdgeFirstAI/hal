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

        let dma = src.as_dma().ok_or_else(|| {
            Error::NotImplemented(format!(
                "OpenGL EGLImage requires DMA tensor, got {:?}",
                src.memory()
            ))
        })?;
        let fd = dma.fd.as_raw_fd();

        // For multiplane NV12, get the UV plane's fd from the chroma tensor
        let uv_fd = if src.is_multiplane() {
            let chroma = src.chroma().unwrap();
            let chroma_dma = chroma.as_dma().ok_or_else(|| {
                Error::NotImplemented("Multiplane chroma tensor must be DMA-backed".to_string())
            })?;
            Some(chroma_dma.fd.as_raw_fd())
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
            None => src.effective_row_stride().unwrap_or_else(|| {
                if src_fmt == PixelFormat::Nv12 {
                    // Luma plane is 1 byte/pixel for NV12 semi-planar YUV.
                    width
                } else {
                    width * channels
                }
            }),
        };

        // A view imports its parent at offset 0 (its byte offset becomes the
        // viewport, never the import base); a whole tensor honors its own offset.
        let plane0_offset = if view_origin.is_some() {
            0
        } else {
            src.plane_offset().unwrap_or(0)
        };

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
        // Combined (luma + chroma) buffer height, matching `PixelFormat::image_shape`:
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

        let dma = src.as_dma().ok_or_else(|| {
            Error::NotImplemented(format!(
                "OpenGL Path-B R8 import requires DMA tensor, got {:?}",
                src.memory()
            ))
        })?;
        let fd = dma.fd.as_raw_fd();
        let plane0_offset = src.plane_offset().unwrap_or(0);

        Ok(DmaImportAttrs {
            width: tex_width,
            height: combined_h,
            drm_fourcc: DrmFourcc::R8,
            plane0_fd: fd,
            plane0_pitch: tex_width,
            plane0_offset,
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
        match Tensor::<u8>::new(&[bytes], Some(TensorMemory::Dma), Some(name)) {
            Ok(t) if t.memory() == TensorMemory::Dma => Some(t),
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
            eprintln!("SKIPPED: from_tensor_source_view... - DMA not available");
            return;
        }
        // 64x64 RGBA with a padded 320-byte stride (tight row = 64*4 = 256).
        let parent = match Tensor::<u8>::image_with_stride(
            64,
            64,
            PixelFormat::Rgba,
            320,
            Some(TensorMemory::Dma),
        ) {
            Ok(t) => t,
            Err(_) => {
                eprintln!("SKIPPED: image_with_stride DMA unavailable");
                return;
            }
        };
        let view = parent
            .view(edgefirst_tensor::Region::new(8, 8, 32, 16))
            .unwrap();

        // SOURCE (for_dst = false): import the view's OWN region.
        let s = DmaImportAttrs::from_tensor(&view, PixelFormat::Rgba, false).unwrap();
        assert_eq!(
            (s.width, s.height),
            (32, 16),
            "source view imports its own dimensions"
        );
        assert_eq!(
            s.plane0_offset,
            8 * 320 + 8 * 4,
            "source view imports at its own byte offset"
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
                eprintln!("SKIPPED: test_nv12_true_multiplane_attrs - DMA not available");
                return;
            }
        };
        let chroma_buf = match alloc_dma(chroma_bytes, "chroma_buf") {
            Some(t) => t,
            None => {
                eprintln!("SKIPPED: test_nv12_true_multiplane_attrs - DMA alloc failed");
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
                eprintln!("SKIPPED: test_nv12_same_fd_multiplane_attrs - DMA not available");
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
                eprintln!("SKIPPED: test_nv12_even_width_relaxed_gate - DMA not available");
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
                eprintln!("SKIPPED: test_nv12_contiguous_single_fd_attrs - DMA not available");
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
                eprintln!("SKIPPED: test_nv12_contiguous_padded_stride_attrs - DMA not available");
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
                eprintln!("SKIPPED: test_nv12_multiplane_padded_strides_attrs - DMA not available");
                return;
            }
        };
        let chroma_buf = match alloc_dma(chroma_bytes, "chroma_padded") {
            Some(t) => t,
            None => {
                eprintln!("SKIPPED: DMA alloc failed");
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
                eprintln!("SKIPPED: test_nv12_same_fd_nonzero_luma_offset - DMA not available");
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
                eprintln!("SKIPPED: test_nv12_chroma_offset_exceeds_buffer - DMA not available");
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
            eprintln!("SKIPPED: test_from_tensor_accepts_non_4_aligned_rgba — DMA not available");
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
                Some(edgefirst_tensor::TensorMemory::Dma),
            ) {
                Ok(t) => t,
                Err(e) => {
                    eprintln!("SKIPPED: image_with_stride failed at width {w}: {e}");
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
            eprintln!("SKIPPED: test_from_tensor_accepts_non_4_aligned_bgra — DMA not available");
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
            Some(edgefirst_tensor::TensorMemory::Dma),
        ) {
            Ok(t) => t,
            Err(e) => {
                eprintln!("SKIPPED: image_with_stride failed: {e}");
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
}
