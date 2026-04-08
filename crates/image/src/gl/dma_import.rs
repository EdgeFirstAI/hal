// SPDX-FileCopyrightText: Copyright 2025 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! DMA-BUF plane resolution for EGL image import.
//!
//! Separates the logic that resolves plane parameters (fd, pitch, offset)
//! from the actual `eglCreateImage` call so that the attribute construction
//! can be unit-tested without a GPU context.

use edgefirst_tensor::{PixelFormat, PixelLayout, Tensor, TensorTrait};
use gbm::drm::buffer::DrmFourcc;
use khronos_egl::{self as egl, Attrib};
use std::os::fd::AsRawFd;
use std::os::unix::io::RawFd;

use super::context::egl_ext;
use super::shaders::pixel_format_to_drm;
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
    pub fn from_tensor(src: &Tensor<u8>, src_fmt: PixelFormat) -> Result<Self, Error> {
        let src_w = src.width().ok_or(Error::NotAnImage)?;
        let src_h = src.height().ok_or(Error::NotAnImage)?;
        let src_channels = src_fmt.channels();

        let (width, height, drm_fourcc, channels) = if src_fmt == PixelFormat::Nv12 {
            if !src_w.is_multiple_of(4) {
                return Err(Error::NotSupported(format!(
                    "EGLImage requires width divisible by 4 for {src_fmt}, got {src_w}"
                )));
            }
            (
                src_w,
                src_h,
                pixel_format_to_drm(PixelFormat::Nv12)?,
                1usize,
            )
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
            if !src_w.is_multiple_of(4) {
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

        // Use the tensor's stored stride if set (for externally allocated buffers
        // with row padding), otherwise compute the tightly-packed pitch.
        let plane0_pitch = src.effective_row_stride().unwrap_or_else(|| {
            if src_fmt == PixelFormat::Nv12 {
                width
            } else {
                width * channels
            }
        });

        let plane0_offset = src.plane_offset().unwrap_or(0);

        // NV12 requires a second plane for UV data
        let plane1 = if src_fmt == PixelFormat::Nv12 {
            let (plane1_fd, uv_offset) = if let Some(chroma_fd) = uv_fd {
                // Multiplane: UV in separate DMA-BUF — use chroma's plane_offset or 0
                let chroma_offset = src.chroma().and_then(|c| c.plane_offset()).unwrap_or(0);
                (chroma_fd, chroma_offset)
            } else {
                // Contiguous: UV follows Y in same buffer.
                // Use stride-aware offset — if Y has padding, UV starts
                // at stride * height, not width * height.  Include the
                // luma plane_offset so the UV base is correct when pixel
                // data does not start at byte 0.
                (fd, plane0_offset + plane0_pitch * height)
            };
            let plane1_pitch = if let Some(chroma) = src.chroma() {
                // Multiplane: use chroma's explicit stride if set (via
                // set_row_stride_unchecked during import), or fall back to
                // the luma pitch (NV12 UV row width in bytes equals Y width)
                chroma.effective_row_stride().unwrap_or(plane0_pitch)
            } else {
                // Contiguous NV12: UV stride matches Y stride
                plane0_pitch
            };
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
        })
    }

    /// Build the EGL attribute list for `eglCreateImage`.
    pub fn to_egl_attribs(&self) -> Vec<Attrib> {
        let mut attrs = vec![
            egl_ext::LINUX_DRM_FOURCC as Attrib,
            self.drm_fourcc as Attrib,
            khronos_egl::WIDTH as Attrib,
            self.width as Attrib,
            khronos_egl::HEIGHT as Attrib,
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
            attrs.extend_from_slice(&[
                egl_ext::YUV_COLOR_SPACE_HINT as Attrib,
                egl_ext::ITU_REC709 as Attrib,
                egl_ext::SAMPLE_RANGE_HINT as Attrib,
                egl_ext::YUV_NARROW_RANGE as Attrib,
            ]);
        }

        attrs.push(khronos_egl::NONE as Attrib);
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
        )
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
        let chroma_bytes = stride * (height / 2);

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

        let attrs = DmaImportAttrs::from_tensor(tensor_u8, PixelFormat::Nv12).unwrap();

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
        let chroma_size = stride * (height / 2);
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

        let attrs = DmaImportAttrs::from_tensor(tensor_u8, PixelFormat::Nv12).unwrap();
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

        let attrs = DmaImportAttrs::from_tensor(tensor_u8, PixelFormat::Nv12).unwrap();
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

        let attrs = DmaImportAttrs::from_tensor(tensor_u8, PixelFormat::Nv12).unwrap();
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
        let chroma_bytes = chroma_stride * (height / 2);

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

        let attrs = DmaImportAttrs::from_tensor(tensor_u8, PixelFormat::Nv12).unwrap();
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
        let total_bytes = chroma_offset + stride * (height / 2);

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

        let attrs = DmaImportAttrs::from_tensor(tensor_u8, PixelFormat::Nv12).unwrap();
        let p1 = attrs.plane1.as_ref().unwrap();

        assert_eq!(attrs.plane0_offset, luma_offset);
        assert_eq!(p1.offset, chroma_offset);
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
        assert_eq!(*egl.last().unwrap(), khronos_egl::NONE as Attrib);
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
        };

        let egl = attrs.to_egl_attribs();

        // No PLANE1 attributes
        assert_eq!(egl_attrib_value(&egl, egl_ext::DMA_BUF_PLANE1_FD), None);
        assert_eq!(egl_attrib_value(&egl, egl_ext::DMA_BUF_PLANE1_PITCH), None);
        assert_eq!(egl_attrib_value(&egl, egl_ext::DMA_BUF_PLANE1_OFFSET), None);

        // No YUV hints
        assert_eq!(egl_attrib_value(&egl, egl_ext::YUV_COLOR_SPACE_HINT), None);

        // Terminated
        assert_eq!(*egl.last().unwrap(), khronos_egl::NONE as Attrib);
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
}
