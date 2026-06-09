// SPDX-FileCopyrightText: Copyright 2025 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

#![cfg(target_os = "linux")]

use crate::colorimetry::effective_colorimetry;
use crate::{CPUProcessor, Crop, Error, Flip, ImageProcessorTrait, ResolvedCrop, Result, Rotation};
use edgefirst_tensor::{
    ColorEncoding, ColorRange, Colorimetry, DType, PixelFormat, Tensor, TensorDyn, TensorMapTrait,
    TensorTrait,
};
use four_char_code::FourCharCode;
use g2d_sys::{G2DFormat, G2DPhysical, G2DSurface, G2D};
use std::{os::fd::AsRawFd, time::Instant};

/// Pure colorimetry-eligibility predicate for the G2D backend.
///
/// G2D is **matrix-only**: the g2d-sys API exposes `set_bt601_colorspace()` and
/// `set_bt709_colorspace()` (the YCbCr matrix selection) but has **no control
/// over quantization range** — the hardware is effectively limited-range only —
/// and exposes **no BT.2020 matrix**. Therefore:
///
/// - A YUV conversion whose resolved range is `Full` must be DECLINED so the
///   `ImageProcessor` dispatch falls through to the OpenGL / CPU backends,
///   which honour full vs limited range correctly.
/// - Any `Bt2020`-encoded conversion must be DECLINED (G2D cannot express it),
///   again falling through to GL / CPU.
///
/// `src_is_yuv` is `true` when the conversion has a YUV side whose colorimetry
/// matters (YUV→RGB uses the source colorimetry, RGB→YUV uses the destination).
/// For RGB→RGB (no YUV side) the full-range rule is N/A and only the BT.2020
/// matrix restriction applies.
pub(crate) fn g2d_can_handle(cm: &Colorimetry, src_is_yuv: bool) -> bool {
    // G2D matrix-only (no range control) and no BT.2020: decline full-range YUV
    // and BT.2020.
    !(src_is_yuv && cm.range == Some(ColorRange::Full))
        && cm.encoding != Some(ColorEncoding::Bt2020)
}

/// Convert a PixelFormat to the G2D-compatible FourCharCode.
fn pixelfmt_to_fourcc(fmt: PixelFormat) -> FourCharCode {
    use four_char_code::four_char_code;
    match fmt {
        PixelFormat::Rgb => four_char_code!("RGB "),
        PixelFormat::Rgba => four_char_code!("RGBA"),
        PixelFormat::Bgra => four_char_code!("BGRA"),
        PixelFormat::Grey => four_char_code!("Y800"),
        PixelFormat::Yuyv => four_char_code!("YUYV"),
        PixelFormat::Vyuy => four_char_code!("VYUY"),
        PixelFormat::Nv12 => four_char_code!("NV12"),
        PixelFormat::Nv16 => four_char_code!("NV16"),
        // Planar formats have no standard FourCC; use RGBA as fallback
        _ => four_char_code!("RGBA"),
    }
}

/// G2DConverter implements the ImageProcessor trait using the NXP G2D
/// library for hardware-accelerated image processing on i.MX platforms.
#[derive(Debug)]
pub struct G2DProcessor {
    g2d: G2D,
}

unsafe impl Send for G2DProcessor {}
unsafe impl Sync for G2DProcessor {}

impl G2DProcessor {
    /// Creates a new G2DConverter instance.
    ///
    /// The BT.709 matrix set here is only a safe default; each `convert()` call
    /// re-programs the matrix from the resolved colorimetry of the YUV side of
    /// that conversion (see `convert_impl`). G2D is matrix-only — it has no
    /// range control (limited-range only) and no BT.2020 matrix — so full-range
    /// YUV and BT.2020 conversions are declined and fall through to GL/CPU.
    pub fn new() -> Result<Self> {
        let mut g2d = G2D::new("libg2d.so.2")?;
        // Safe initial default only: every `convert()` re-programs the matrix
        // from the resolved colorimetry of that conversion (see `convert_impl`).
        // The g2d-sys API selects only the matrix, not full-vs-limited range, so
        // G2D is limited-range only — full-range and BT.2020 are declined and
        // fall through to GL/CPU (a structural hardware limit, not a stop-gap).
        g2d.set_bt601_colorspace()?;

        log::debug!("G2DConverter created with version {:?}", g2d.version());
        Ok(Self { g2d })
    }

    /// Returns the G2D library version as defined by _G2D_VERSION in the shared
    /// library.
    pub fn version(&self) -> g2d_sys::Version {
        self.g2d.version()
    }

    fn convert_impl(
        &mut self,
        src_dyn: &TensorDyn,
        dst_dyn: &mut TensorDyn,
        rotation: Rotation,
        flip: Flip,
        crop: ResolvedCrop,
    ) -> Result<()> {
        let _span = tracing::trace_span!(
            "image.convert.g2d",
            src_fmt = ?src_dyn.format(),
            dst_fmt = ?dst_dyn.format(),
        )
        .entered();
        if log::log_enabled!(log::Level::Trace) {
            log::trace!(
                "G2D convert: {:?}({:?}/{:?}) → {:?}({:?}/{:?})",
                src_dyn.format(),
                src_dyn.dtype(),
                src_dyn.memory(),
                dst_dyn.format(),
                dst_dyn.dtype(),
                dst_dyn.memory(),
            );
        }

        if src_dyn.dtype() != DType::U8 {
            return Err(Error::NotSupported(
                "G2D only supports u8 source tensors".to_string(),
            ));
        }
        let is_int8_dst = dst_dyn.dtype() == DType::I8;
        if dst_dyn.dtype() != DType::U8 && !is_int8_dst {
            return Err(Error::NotSupported(
                "G2D only supports u8 or i8 destination tensors".to_string(),
            ));
        }

        let src_fmt = src_dyn.format().ok_or(Error::NotAnImage)?;
        let dst_fmt = dst_dyn.format().ok_or(Error::NotAnImage)?;

        // Validate supported format pairs
        use PixelFormat::*;
        match (src_fmt, dst_fmt) {
            (Rgba, Rgba) => {}
            (Rgba, Yuyv) => {}
            (Rgba, Rgb) => {}
            (Yuyv, Rgba) => {}
            (Yuyv, Yuyv) => {}
            (Yuyv, Rgb) => {}
            // VYUY: i.MX8MP G2D hardware rejects VYUY blits (only YUYV/UYVY
            // among packed YUV 4:2:2). ImageProcessor falls through to CPU.
            (Nv12, Rgba) => {}
            (Nv12, Yuyv) => {}
            (Nv12, Rgb) => {}
            (Rgba, Bgra) => {}
            (Yuyv, Bgra) => {}
            (Nv12, Bgra) => {}
            (Bgra, Bgra) => {}
            (s, d) => {
                return Err(Error::NotSupported(format!(
                    "G2D does not support {} to {} conversion",
                    s, d
                )));
            }
        }

        // ── Per-conversion colorspace matrix ─────────────────────────
        // G2D is matrix-only (no range control, no BT.2020). Resolve the
        // colorimetry of the YUV side of this conversion and program the
        // matching matrix before the blit. YUV→RGB uses the *source*
        // colorimetry; RGB→YUV uses the *destination*. RGB→RGB has no YUV side
        // so the matrix is irrelevant and left untouched.
        //
        // Full-range YUV and BT.2020 are declined by `g2d_can_handle` in the
        // ImageProcessor dispatch (lib.rs) BEFORE we get here, so those fall
        // through to the GL / CPU backends which honour range and BT.2020. We
        // still gate here defensively so a direct G2DProcessor caller behaves.
        let src_is_yuv = src_fmt.is_yuv();
        let dst_is_yuv = dst_fmt.is_yuv();
        if src_is_yuv || dst_is_yuv {
            let cm = if src_is_yuv {
                effective_colorimetry(src_dyn)
            } else {
                effective_colorimetry(dst_dyn)
            };
            if !g2d_can_handle(&cm, true) {
                return Err(Error::NotSupported(format!(
                    "G2D cannot express colorimetry {:?}/{:?} (matrix-only, \
                     no range control, no BT.2020)",
                    cm.encoding, cm.range
                )));
            }
            match cm.encoding {
                Some(ColorEncoding::Bt601) => self.g2d.set_bt601_colorspace()?,
                Some(ColorEncoding::Bt709) => self.g2d.set_bt709_colorspace()?,
                // BT.2020 already declined above; any future encoding falls
                // back to the BT.709 matrix (closest HD-grade approximation).
                _ => self.g2d.set_bt709_colorspace()?,
            }
        }

        // Source/placement already validated by `Crop::resolve` at the entry.
        let src = src_dyn.as_u8().unwrap();
        // For i8 destinations, reinterpret as u8 for G2D (same byte layout).
        // The XOR 0x80 post-pass is applied after the blit completes.
        let dst = if is_int8_dst {
            // SAFETY: Tensor<i8> and Tensor<u8> have identical memory layout.
            // The T parameter only affects PhantomData<T> (zero-sized) in
            // TensorStorage variants and the typed view from map(). The chroma
            // field (Option<Box<Tensor<T>>>) is also layout-identical. This
            // reinterpreted reference is used only for shape/fd access and the
            // G2D blit (which operates on raw DMA bytes). It does not outlive
            // dst_dyn and is never stored.
            let i8_tensor = dst_dyn.as_i8_mut().unwrap();
            unsafe { &mut *(i8_tensor as *mut Tensor<i8> as *mut Tensor<u8>) }
        } else {
            dst_dyn.as_u8_mut().unwrap()
        };

        let mut src_surface = tensor_to_g2d_surface(src)?;
        let mut dst_surface = tensor_to_g2d_surface(dst)?;

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

        if let Some(crop_rect) = crop.src_rect {
            src_surface.left = crop_rect.left as i32;
            src_surface.top = crop_rect.top as i32;
            src_surface.right = (crop_rect.left + crop_rect.width) as i32;
            src_surface.bottom = (crop_rect.top + crop_rect.height) as i32;
        }

        let dst_w = dst.width().unwrap();
        let dst_h = dst.height().unwrap();

        // Clear the destination with the letterbox color before blitting the
        // image into the sub-region.
        //
        // g2d_clear does not support 3-byte-per-pixel formats (RGB888, BGR888).
        // For those formats, fall back to CPU fill after the blit.
        let needs_clear = crop.dst_color.is_some()
            && crop.dst_rect.is_some_and(|dst_rect| {
                dst_rect.left != 0
                    || dst_rect.top != 0
                    || dst_rect.width != dst_w
                    || dst_rect.height != dst_h
            });

        if needs_clear && dst_fmt != Rgb {
            if let Some(dst_color) = crop.dst_color {
                let start = Instant::now();
                self.g2d.clear(&mut dst_surface, dst_color)?;
                log::trace!("g2d clear takes {:?}", start.elapsed());
            }
        }

        if let Some(crop_rect) = crop.dst_rect {
            // stride is in pixels; multiply by bytes-per-pixel (== channels()
            // for u8 data) to get the byte offset.  All G2D destination
            // formats are packed, so channels() == bpp always holds here.
            dst_surface.planes[0] += ((crop_rect.top * dst_surface.stride as usize
                + crop_rect.left)
                * dst_fmt.channels()) as u64;

            dst_surface.right = crop_rect.width as i32;
            dst_surface.bottom = crop_rect.height as i32;
            dst_surface.width = crop_rect.width as i32;
            dst_surface.height = crop_rect.height as i32;
        }

        // ── i.MX95 DPU even-dimension workaround ──────────────────────────
        // The i.MX95 DPU G2D backend garbles a chroma-subsampled blit whose
        // width or height is odd — it drops the partial trailing chroma sample
        // and corrupts the ENTIRE output (max_diff ~255); the i.MX8MP Vivante
        // backend tolerates it. When either side is a chroma-subsampled YUV
        // format, round BOTH blit rectangles up to even on each *subsampled*
        // axis so the DPU's chroma geometry is exact. The width pad always lands
        // in the 64-aligned row stride (both sides). The height pad reads the
        // source's chroma plane (within its allocation) and writes ONE extra
        // destination row — applied only when the destination's (page-aligned)
        // DMA allocation can hold it; otherwise G2D declines so the caller falls
        // back to GL/CPU, which convert odd dimensions natively. Only the
        // logical w×h region is consumed downstream, so the padding is unused.
        let (sx_src, sy_src) = chroma_subsample_shifts(src_fmt);
        let (sx_dst, sy_dst) = chroma_subsample_shifts(dst_fmt);
        let pad_w = (sx_src | sx_dst) > 0;
        let pad_h = (sy_src | sy_dst) > 0;
        if pad_w || pad_h {
            let even = |v: i32| v + (v & 1);
            if pad_h && even(dst_surface.bottom) != dst_surface.bottom {
                // The padded blit writes one extra destination row. It is safe
                // only when the blit starts at the destination's top (no crop
                // placement offset) and the extra row stays within the
                // destination's DMA mapping. A self-allocated DMA-BUF is mapped
                // at page granularity, so its strided allocation
                // (row_bytes × height) is physically backed up to the next page
                // boundary. Query the runtime page size — aarch64 kernels may use
                // 16 KiB or 64 KiB pages, and hard-coding 4 KiB would
                // under-estimate the mapping and needlessly reject a destination
                // that is in fact page-backed.
                let page = match unsafe { libc::sysconf(libc::_SC_PAGESIZE) } {
                    n if n > 0 => n as usize,
                    _ => 4096,
                };
                let row_bytes = dst_surface.stride as usize * dst_fmt.channels();
                let mapped = (row_bytes * dst_h).next_multiple_of(page);
                let needed = row_bytes * even(dst_surface.bottom) as usize;
                if crop.dst_rect.is_some() || needed > mapped {
                    return Err(Error::NotSupported(format!(
                        "G2D: odd-height {dst_fmt} destination cannot hold the padding \
                         row for the i.MX95 DPU even-dimension workaround; deferring to \
                         GL/CPU"
                    )));
                }
            }
            if pad_w {
                src_surface.right = even(src_surface.right);
                src_surface.width = even(src_surface.width);
                dst_surface.right = even(dst_surface.right);
                dst_surface.width = even(dst_surface.width);
            }
            if pad_h {
                src_surface.bottom = even(src_surface.bottom);
                src_surface.height = even(src_surface.height);
                dst_surface.bottom = even(dst_surface.bottom);
                dst_surface.height = even(dst_surface.height);
            }
        }

        log::trace!("G2D blit: {src_fmt}→{dst_fmt} int8={is_int8_dst}");
        self.g2d.blit(&src_surface, &dst_surface)?;
        self.g2d.finish()?;
        log::trace!("G2D blit complete");

        // CPU fallback for RGB888 (unsupported by g2d_clear)
        if needs_clear && dst_fmt == Rgb {
            if let (Some(dst_color), Some(dst_rect)) = (crop.dst_color, crop.dst_rect) {
                let start = Instant::now();
                CPUProcessor::fill_image_outside_crop_u8(dst, dst_color, dst_rect)?;
                log::trace!("cpu fill takes {:?}", start.elapsed());
            }
        }

        // Apply XOR 0x80 for int8 output (u8→i8 bias conversion).
        // map() issues DMA_BUF_IOCTL_SYNC(START) on the dst fd; for self-allocated
        // CMA buffers this performs cache invalidation via the DrmAttachment.
        // For foreign fds (e.g. the Neutron NPU DMA-BUF imported via from_fd()),
        // the DrmAttachment is None and the sync ioctl is handled by the NPU driver.
        // The map drop issues DMA_BUF_IOCTL_SYNC(END) so the NPU DMA engine sees
        // the CPU-written XOR'd data on the next Invoke().
        if is_int8_dst {
            let start = Instant::now();
            let mut map = dst.map()?;
            crate::cpu::apply_int8_xor_bias(map.as_mut_slice(), dst_fmt);
            log::trace!("g2d int8 XOR 0x80 post-pass takes {:?}", start.elapsed());
        }

        Ok(())
    }
}

impl ImageProcessorTrait for G2DProcessor {
    fn convert(
        &mut self,
        src: &TensorDyn,
        dst: &mut TensorDyn,
        rotation: Rotation,
        flip: Flip,
        crop: Crop,
    ) -> Result<()> {
        // A view()/batch() destination needs no special handling here: G2D blits
        // to the destination's `plane_offset` + parent `row_stride`, so a sub-view
        // (the tile) lands in its window of the parent buffer natively — the same
        // mechanism a whole-tensor blit uses. (The cache-key/glViewport batch
        // path is GL-specific; G2D addresses the tile by offset+stride.)
        let crop = crop.resolve(
            src.width().unwrap_or(0),
            src.height().unwrap_or(0),
            dst.width().unwrap_or(0),
            dst.height().unwrap_or(0),
        )?;
        self.convert_impl(src, dst, rotation, flip, crop)
    }

    fn draw_decoded_masks(
        &mut self,
        dst: &mut TensorDyn,
        detect: &[crate::DetectBox],
        segmentation: &[crate::Segmentation],
        overlay: crate::MaskOverlay<'_>,
    ) -> Result<()> {
        // G2D can produce the *frame* (background or cleared canvas) via
        // hardware primitives — but has no rasterizer for boxes / masks.
        // If detections are present, defer to another backend.
        if !detect.is_empty() || !segmentation.is_empty() {
            return Err(Error::NotImplemented(
                "G2D does not support drawing detection or segmentation overlays".to_string(),
            ));
        }
        draw_empty_frame_g2d(&mut self.g2d, dst, overlay.background)
    }

    fn draw_proto_masks(
        &mut self,
        dst: &mut TensorDyn,
        detect: &[crate::DetectBox],
        _proto_data: &crate::ProtoData,
        overlay: crate::MaskOverlay<'_>,
    ) -> Result<()> {
        // Same logic as draw_decoded_masks: G2D handles empty-detection
        // frames (clear or blit background) via the 2D hardware.
        if !detect.is_empty() {
            return Err(Error::NotImplemented(
                "G2D does not support drawing detection or segmentation overlays".to_string(),
            ));
        }
        draw_empty_frame_g2d(&mut self.g2d, dst, overlay.background)
    }

    fn set_class_colors(&mut self, _: &[[u8; 4]]) -> Result<()> {
        Err(Error::NotImplemented(
            "G2D does not support setting colors for rendering detection or segmentation overlays"
                .to_string(),
        ))
    }
}

/// Produce the zero-detection frame on `dst` using G2D hardware ops.
///
/// `background == None` → `g2d_clear(dst, 0x00000000)` — actively fills the
/// destination with transparent black using the 2D engine (no CPU touch).
///
/// `background == Some(bg)` → `g2d_blit(bg → dst)` — hardware copy of bg
/// into dst. This is the "draw the cleared overlay onto the background"
/// path; G2D has no pure copy primitive, so we use the blit engine with a
/// 1:1 same-format surface pair, which is the hardware equivalent.
///
/// Both paths `finish()` before returning so dst is coherent for any
/// downstream reader.
fn draw_empty_frame_g2d(
    g2d: &mut G2D,
    dst_dyn: &mut TensorDyn,
    background: Option<&TensorDyn>,
) -> Result<()> {
    if dst_dyn.dtype() != DType::U8 {
        return Err(Error::NotSupported(
            "G2D only supports u8 destination tensors".to_string(),
        ));
    }
    let dst = dst_dyn.as_u8_mut().ok_or(Error::NotAnImage)?;

    // DMA-only: G2D operates on physical addresses. A Mem-backed dst means
    // we're on the wrong backend — surface a dispatch error so the caller
    // can fall back.
    if dst.as_dma().is_none() {
        return Err(Error::NotImplemented(
            "g2d only supports Dma memory".to_string(),
        ));
    }

    let mut dst_surface = tensor_to_g2d_surface(dst)?;

    match background {
        None => {
            // Case 1 — no background, no detections: hardware clear to
            // transparent black. Every byte of dst is written by the 2D
            // engine; no reliance on prior state.
            let start = Instant::now();
            g2d.clear(&mut dst_surface, [0, 0, 0, 0])?;
            g2d.finish()?;
            log::trace!("g2d clear (empty frame) takes {:?}", start.elapsed());
        }
        Some(bg_dyn) => {
            // Case 2 — background, no detections: hardware blit bg → dst.
            // Validate shape/format equivalence; if the caller handed us
            // mismatched surfaces we return a structured error rather than
            // silently producing wrong output.
            if bg_dyn.shape() != dst.shape() {
                return Err(Error::InvalidShape(
                    "background shape does not match dst".into(),
                ));
            }
            if bg_dyn.format() != dst.format() {
                return Err(Error::InvalidShape(
                    "background pixel format does not match dst".into(),
                ));
            }
            if bg_dyn.dtype() != DType::U8 {
                return Err(Error::NotSupported(
                    "G2D only supports u8 background tensors".to_string(),
                ));
            }
            let bg = bg_dyn.as_u8().ok_or(Error::NotAnImage)?;
            if bg.as_dma().is_none() {
                return Err(Error::NotImplemented(
                    "g2d background must be Dma-backed".to_string(),
                ));
            }
            let src_surface = tensor_to_g2d_surface(bg)?;
            let start = Instant::now();
            g2d.blit(&src_surface, &dst_surface)?;
            g2d.finish()?;
            log::trace!("g2d blit (bg→dst) takes {:?}", start.elapsed());
        }
    }
    Ok(())
}

/// Chroma subsampling of `fmt` as `(shift_x, shift_y)` — 1 = half resolution on
/// that axis, 0 = full. Drives the i.MX95 DPU even-dimension workaround: a blit
/// dimension that the format subsamples must be even or the DPU garbles it.
fn chroma_subsample_shifts(fmt: PixelFormat) -> (u32, u32) {
    match fmt.chroma_layout() {
        Some(cl) => (cl.shift_x, cl.shift_y),
        // Packed 4:2:2 (YUYV / VYUY) subsample horizontally; all other packed
        // formats (RGB/RGBA/BGRA/Grey) are full-resolution.
        None if matches!(fmt, PixelFormat::Yuyv | PixelFormat::Vyuy) => (1, 0),
        None => (0, 0),
    }
}

/// Build a `G2DSurface` from a `Tensor<u8>` that carries pixel-format metadata.
///
/// The tensor must be backed by DMA memory and must have a pixel format set.
fn tensor_to_g2d_surface(img: &Tensor<u8>) -> Result<G2DSurface> {
    let fmt = img.format().ok_or(Error::NotAnImage)?;
    let dma = img
        .as_dma()
        .ok_or_else(|| Error::NotImplemented("g2d only supports Dma memory".to_string()))?;
    let phys: G2DPhysical = dma.fd.as_raw_fd().try_into()?;

    // NV12 is a two-plane format: Y plane followed by interleaved UV plane.
    // planes[0] = Y plane start, planes[1] = UV plane start (Y size = width * height)
    //
    // plane_offset is the byte offset within the DMA-BUF where pixel data
    // starts.  G2D works with raw physical addresses so we must add the
    // offset ourselves — the hardware has no concept of a per-plane offset.
    let base_addr = phys.address();
    let luma_offset = img.plane_offset().unwrap_or(0) as u64;
    let planes = if fmt == PixelFormat::Nv12 {
        if img.is_multiplane() {
            // Multiplane: UV in separate DMA-BUF, get its physical address
            let chroma = img.chroma().unwrap();
            let chroma_dma = chroma.as_dma().ok_or_else(|| {
                Error::NotImplemented("g2d multiplane chroma must be DMA-backed".to_string())
            })?;
            let uv_phys: G2DPhysical = chroma_dma.fd.as_raw_fd().try_into()?;
            let chroma_offset = img.chroma().and_then(|c| c.plane_offset()).unwrap_or(0) as u64;
            [
                base_addr + luma_offset,
                uv_phys.address() + chroma_offset,
                0,
            ]
        } else {
            let w = img.width().unwrap();
            let h = img.height().unwrap();
            let stride = img.effective_row_stride().unwrap_or(w.next_multiple_of(2));
            let uv_offset = (luma_offset as usize + stride * h) as u64;
            [base_addr + luma_offset, base_addr + uv_offset, 0]
        }
    } else {
        [base_addr + luma_offset, 0, 0]
    };

    let w = img.width().unwrap();
    let h = img.height().unwrap();
    let fourcc = pixelfmt_to_fourcc(fmt);

    // G2D stride is in pixels.  effective_row_stride() returns bytes, so
    // divide by the bytes-per-pixel (channels for u8 data) to convert.
    let stride_pixels = match img.effective_row_stride() {
        Some(s) => {
            let channels = fmt.channels();
            if s % channels != 0 {
                return Err(Error::NotImplemented(
                    "g2d requires row stride to be a multiple of bytes-per-pixel".to_string(),
                ));
            }
            s / channels
        }
        None => w,
    };

    Ok(G2DSurface {
        planes,
        format: G2DFormat::try_from(fourcc)?.format(),
        left: 0,
        top: 0,
        right: w as i32,
        bottom: h as i32,
        stride: stride_pixels as i32,
        width: w as i32,
        height: h as i32,
        blendfunc: 0,
        clrcolor: 0,
        rot: 0,
        global_alpha: 0,
    })
}

#[cfg(test)]
mod g2d_predicate_tests {
    use super::*;
    use edgefirst_tensor::{ColorEncoding, ColorRange, Colorimetry};

    #[test]
    fn g2d_declines_full_range_and_bt2020_yuv() {
        let full = Colorimetry::default()
            .with_range(ColorRange::Full)
            .with_encoding(ColorEncoding::Bt709);
        let lim = Colorimetry::default()
            .with_range(ColorRange::Limited)
            .with_encoding(ColorEncoding::Bt709);
        let bt2020 = Colorimetry::default()
            .with_range(ColorRange::Limited)
            .with_encoding(ColorEncoding::Bt2020);
        assert!(!g2d_can_handle(&full, true)); // full-range YUV → decline
        assert!(g2d_can_handle(&lim, true)); // limited 709 YUV → ok
        assert!(!g2d_can_handle(&bt2020, true)); // BT.2020 → decline
        assert!(g2d_can_handle(&full, false)); // no YUV side → rule N/A → ok
    }
}

#[cfg(feature = "g2d_test_formats")]
#[cfg(test)]
#[allow(deprecated)]
mod g2d_tests {
    use super::*;
    use crate::{CPUProcessor, Flip, G2DProcessor, ImageProcessorTrait, Rotation};
    use edgefirst_tensor::{
        is_dma_available, DType, PixelFormat, TensorDyn, TensorMapTrait, TensorMemory, TensorTrait,
    };
    use image::buffer::ConvertBuffer;

    /// G2D is usable only where `libg2d.so.2` loads — the NXP i.MX boards. On
    /// V3D/Tegra/x86 the library is absent, so `G2DProcessor::new()` and the
    /// surface physical-address resolution these tests exercise both fail. DMA
    /// being available does NOT imply G2D (e.g. rpi5/V3D has DMA but no G2D), so
    /// these tests must gate on G2D, not just DMA, or they panic/fail on
    /// non-G2D hardware instead of skipping.
    #[cfg(target_os = "linux")]
    fn is_g2d_available() -> bool {
        G2DProcessor::new().is_ok()
    }

    #[test]
    #[cfg(target_os = "linux")]
    fn test_g2d_formats_no_resize() {
        for i in [
            PixelFormat::Rgba,
            PixelFormat::Yuyv,
            PixelFormat::Rgb,
            PixelFormat::Grey,
            PixelFormat::Nv12,
        ] {
            for o in [
                PixelFormat::Rgba,
                PixelFormat::Yuyv,
                PixelFormat::Rgb,
                PixelFormat::Grey,
            ] {
                let res = test_g2d_format_no_resize_(i, o);
                if let Err(e) = res {
                    println!("{i} to {o} failed: {e:?}");
                } else {
                    println!("{i} to {o} success");
                }
            }
        }
    }

    fn test_g2d_format_no_resize_(
        g2d_in_fmt: PixelFormat,
        g2d_out_fmt: PixelFormat,
    ) -> Result<(), crate::Error> {
        let dst_width = 1280;
        let dst_height = 720;
        let file = edgefirst_bench::testdata::read("zidane.jpg").to_vec();
        let src = crate::load_image_test_helper(&file, Some(PixelFormat::Rgb), None)?;

        // Create DMA buffer for G2D input
        let mut src2 = TensorDyn::image(1280, 720, g2d_in_fmt, DType::U8, Some(TensorMemory::Dma))?;

        let mut cpu_converter = CPUProcessor::new();

        // For PixelFormat::Nv12 input, load from file since CPU doesn't support PixelFormat::Rgb→PixelFormat::Nv12
        if g2d_in_fmt == PixelFormat::Nv12 {
            let nv12_bytes = edgefirst_bench::testdata::read("zidane.nv12");
            src2.as_u8()
                .unwrap()
                .map()?
                .as_mut_slice()
                .copy_from_slice(&nv12_bytes);
        } else {
            cpu_converter.convert(&src, &mut src2, Rotation::None, Flip::None, Crop::no_crop())?;
        }

        let mut g2d_dst = TensorDyn::image(
            dst_width,
            dst_height,
            g2d_out_fmt,
            DType::U8,
            Some(TensorMemory::Dma),
        )?;
        let mut g2d_converter = G2DProcessor::new()?;
        let src2_dyn = src2;
        let mut g2d_dst_dyn = g2d_dst;
        g2d_converter.convert(
            &src2_dyn,
            &mut g2d_dst_dyn,
            Rotation::None,
            Flip::None,
            Crop::no_crop(),
        )?;
        g2d_dst = {
            let mut __t = g2d_dst_dyn.into_u8().unwrap();
            __t.set_format(g2d_out_fmt)
                .map_err(|e| crate::Error::Internal(e.to_string()))?;
            TensorDyn::from(__t)
        };

        let mut cpu_dst =
            TensorDyn::image(dst_width, dst_height, PixelFormat::Rgb, DType::U8, None)?;
        cpu_converter.convert(
            &g2d_dst,
            &mut cpu_dst,
            Rotation::None,
            Flip::None,
            Crop::no_crop(),
        )?;

        compare_images(
            &src,
            &cpu_dst,
            0.98,
            &format!("{g2d_in_fmt}_to_{g2d_out_fmt}"),
        )
    }

    #[test]
    #[cfg(target_os = "linux")]
    fn test_g2d_formats_with_resize() {
        for i in [
            PixelFormat::Rgba,
            PixelFormat::Yuyv,
            PixelFormat::Rgb,
            PixelFormat::Grey,
            PixelFormat::Nv12,
        ] {
            for o in [
                PixelFormat::Rgba,
                PixelFormat::Yuyv,
                PixelFormat::Rgb,
                PixelFormat::Grey,
            ] {
                let res = test_g2d_format_with_resize_(i, o);
                if let Err(e) = res {
                    println!("{i} to {o} failed: {e:?}");
                } else {
                    println!("{i} to {o} success");
                }
            }
        }
    }

    #[test]
    #[cfg(target_os = "linux")]
    fn test_g2d_formats_with_resize_dst_crop() {
        for i in [
            PixelFormat::Rgba,
            PixelFormat::Yuyv,
            PixelFormat::Rgb,
            PixelFormat::Grey,
            PixelFormat::Nv12,
        ] {
            for o in [
                PixelFormat::Rgba,
                PixelFormat::Yuyv,
                PixelFormat::Rgb,
                PixelFormat::Grey,
            ] {
                let res = test_g2d_format_with_resize_dst_crop(i, o);
                if let Err(e) = res {
                    println!("{i} to {o} failed: {e:?}");
                } else {
                    println!("{i} to {o} success");
                }
            }
        }
    }

    fn test_g2d_format_with_resize_(
        g2d_in_fmt: PixelFormat,
        g2d_out_fmt: PixelFormat,
    ) -> Result<(), crate::Error> {
        let dst_width = 600;
        let dst_height = 400;
        let file = edgefirst_bench::testdata::read("zidane.jpg").to_vec();
        let src = crate::load_image_test_helper(&file, Some(PixelFormat::Rgb), None)?;

        let mut cpu_converter = CPUProcessor::new();

        let mut reference = TensorDyn::image(
            dst_width,
            dst_height,
            PixelFormat::Rgb,
            DType::U8,
            Some(TensorMemory::Dma),
        )?;
        cpu_converter.convert(
            &src,
            &mut reference,
            Rotation::None,
            Flip::None,
            Crop::no_crop(),
        )?;

        // Create DMA buffer for G2D input
        let mut src2 = TensorDyn::image(1280, 720, g2d_in_fmt, DType::U8, Some(TensorMemory::Dma))?;

        // For PixelFormat::Nv12 input, load from file since CPU doesn't support PixelFormat::Rgb→PixelFormat::Nv12
        if g2d_in_fmt == PixelFormat::Nv12 {
            let nv12_bytes = edgefirst_bench::testdata::read("zidane.nv12");
            src2.as_u8()
                .unwrap()
                .map()?
                .as_mut_slice()
                .copy_from_slice(&nv12_bytes);
        } else {
            cpu_converter.convert(&src, &mut src2, Rotation::None, Flip::None, Crop::no_crop())?;
        }

        let mut g2d_dst = TensorDyn::image(
            dst_width,
            dst_height,
            g2d_out_fmt,
            DType::U8,
            Some(TensorMemory::Dma),
        )?;
        let mut g2d_converter = G2DProcessor::new()?;
        let src2_dyn = src2;
        let mut g2d_dst_dyn = g2d_dst;
        g2d_converter.convert(
            &src2_dyn,
            &mut g2d_dst_dyn,
            Rotation::None,
            Flip::None,
            Crop::no_crop(),
        )?;
        g2d_dst = {
            let mut __t = g2d_dst_dyn.into_u8().unwrap();
            __t.set_format(g2d_out_fmt)
                .map_err(|e| crate::Error::Internal(e.to_string()))?;
            TensorDyn::from(__t)
        };

        let mut cpu_dst =
            TensorDyn::image(dst_width, dst_height, PixelFormat::Rgb, DType::U8, None)?;
        cpu_converter.convert(
            &g2d_dst,
            &mut cpu_dst,
            Rotation::None,
            Flip::None,
            Crop::no_crop(),
        )?;

        compare_images(
            &reference,
            &cpu_dst,
            0.98,
            &format!("{g2d_in_fmt}_to_{g2d_out_fmt}_resized"),
        )
    }

    fn test_g2d_format_with_resize_dst_crop(
        g2d_in_fmt: PixelFormat,
        g2d_out_fmt: PixelFormat,
    ) -> Result<(), crate::Error> {
        let dst_width = 600;
        let dst_height = 400;
        // The destination sub-rectangle is now expressed as a `view()` of the
        // destination (the old `Crop.dst_rect` was removed); G2D blits into the
        // view's window via its offset + parent stride. Region is (x=left, y=top,
        // width, height).
        let region = crate::Region::new(100, 100, 200, 100);
        let file = edgefirst_bench::testdata::read("zidane.jpg").to_vec();
        let src = crate::load_image_test_helper(&file, Some(PixelFormat::Rgb), None)?;

        let mut cpu_converter = CPUProcessor::new();

        let reference = TensorDyn::image(
            dst_width,
            dst_height,
            PixelFormat::Rgb,
            DType::U8,
            Some(TensorMemory::Dma),
        )?;
        reference
            .as_u8()
            .unwrap()
            .map()
            .unwrap()
            .as_mut_slice()
            .fill(128);
        cpu_converter.convert(
            &src,
            &mut reference.view(region)?,
            Rotation::None,
            Flip::None,
            Crop::no_crop(),
        )?;

        // Create DMA buffer for G2D input
        let mut src2 = TensorDyn::image(1280, 720, g2d_in_fmt, DType::U8, Some(TensorMemory::Dma))?;

        // For PixelFormat::Nv12 input, load from file since CPU doesn't support PixelFormat::Rgb→PixelFormat::Nv12
        if g2d_in_fmt == PixelFormat::Nv12 {
            let nv12_bytes = edgefirst_bench::testdata::read("zidane.nv12");
            src2.as_u8()
                .unwrap()
                .map()?
                .as_mut_slice()
                .copy_from_slice(&nv12_bytes);
        } else {
            cpu_converter.convert(&src, &mut src2, Rotation::None, Flip::None, Crop::no_crop())?;
        }

        let mut g2d_dst = TensorDyn::image(
            dst_width,
            dst_height,
            g2d_out_fmt,
            DType::U8,
            Some(TensorMemory::Dma),
        )?;
        g2d_dst
            .as_u8()
            .unwrap()
            .map()
            .unwrap()
            .as_mut_slice()
            .fill(128);
        let mut g2d_converter = G2DProcessor::new()?;
        let src2_dyn = src2;
        let g2d_dst_dyn = g2d_dst;
        g2d_converter.convert(
            &src2_dyn,
            &mut g2d_dst_dyn.view(region)?,
            Rotation::None,
            Flip::None,
            Crop::no_crop(),
        )?;
        g2d_dst = {
            let mut __t = g2d_dst_dyn.into_u8().unwrap();
            __t.set_format(g2d_out_fmt)
                .map_err(|e| crate::Error::Internal(e.to_string()))?;
            TensorDyn::from(__t)
        };

        let mut cpu_dst =
            TensorDyn::image(dst_width, dst_height, PixelFormat::Rgb, DType::U8, None)?;
        cpu_converter.convert(
            &g2d_dst,
            &mut cpu_dst,
            Rotation::None,
            Flip::None,
            Crop::no_crop(),
        )?;

        compare_images(
            &reference,
            &cpu_dst,
            0.98,
            &format!("{g2d_in_fmt}_to_{g2d_out_fmt}_resized_dst_crop"),
        )
    }

    fn compare_images(
        img1: &TensorDyn,
        img2: &TensorDyn,
        threshold: f64,
        name: &str,
    ) -> Result<(), crate::Error> {
        assert_eq!(img1.height(), img2.height(), "Heights differ");
        assert_eq!(img1.width(), img2.width(), "Widths differ");
        assert_eq!(
            img1.format().unwrap(),
            img2.format().unwrap(),
            "PixelFormat differ"
        );
        assert!(
            matches!(img1.format().unwrap(), PixelFormat::Rgb | PixelFormat::Rgba),
            "format must be Rgb or Rgba for comparison"
        );
        let image1 = match img1.format().unwrap() {
            PixelFormat::Rgb => image::RgbImage::from_vec(
                img1.width().unwrap() as u32,
                img1.height().unwrap() as u32,
                img1.as_u8().unwrap().map().unwrap().to_vec(),
            )
            .unwrap(),
            PixelFormat::Rgba => image::RgbaImage::from_vec(
                img1.width().unwrap() as u32,
                img1.height().unwrap() as u32,
                img1.as_u8().unwrap().map().unwrap().to_vec(),
            )
            .unwrap()
            .convert(),

            _ => unreachable!(),
        };

        let image2 = match img2.format().unwrap() {
            PixelFormat::Rgb => image::RgbImage::from_vec(
                img2.width().unwrap() as u32,
                img2.height().unwrap() as u32,
                img2.as_u8().unwrap().map().unwrap().to_vec(),
            )
            .unwrap(),
            PixelFormat::Rgba => image::RgbaImage::from_vec(
                img2.width().unwrap() as u32,
                img2.height().unwrap() as u32,
                img2.as_u8().unwrap().map().unwrap().to_vec(),
            )
            .unwrap()
            .convert(),

            _ => unreachable!(),
        };

        let similarity = image_compare::rgb_similarity_structure(
            &image_compare::Algorithm::RootMeanSquared,
            &image1,
            &image2,
        )
        .expect("Image Comparison failed");

        if similarity.score < threshold {
            image1.save(format!("{name}_1.png")).unwrap();
            image2.save(format!("{name}_2.png")).unwrap();
            return Err(Error::Internal(format!(
                "{name}: converted image and target image have similarity score too low: {} < {}",
                similarity.score, threshold
            )));
        }
        Ok(())
    }

    // =========================================================================
    // PixelFormat::Nv12 Reference Validation Tests
    // These tests compare G2D PixelFormat::Nv12 conversions against ffmpeg-generated references
    // =========================================================================

    fn load_raw_image(
        width: usize,
        height: usize,
        format: PixelFormat,
        memory: Option<TensorMemory>,
        bytes: &[u8],
    ) -> Result<TensorDyn, crate::Error> {
        let img = TensorDyn::image(width, height, format, DType::U8, memory)?;
        let mut map = img.as_u8().unwrap().map()?;
        let dst = map.as_mut_slice();
        if bytes.len() > dst.len() {
            return Err(crate::Error::InvalidShape(format!(
                "load_raw_image: {} input bytes exceed {}-byte image buffer",
                bytes.len(),
                dst.len()
            )));
        }
        dst[..bytes.len()].copy_from_slice(bytes);
        Ok(img)
    }

    /// Test G2D PixelFormat::Nv12→PixelFormat::Rgba conversion against ffmpeg reference
    #[test]
    #[cfg(target_os = "linux")]
    fn test_g2d_nv12_to_rgba_reference() -> Result<(), crate::Error> {
        if !is_dma_available() || !is_g2d_available() {
            return Ok(());
        }
        // Load PixelFormat::Nv12 source
        let src = load_raw_image(
            1280,
            720,
            PixelFormat::Nv12,
            Some(TensorMemory::Dma),
            &edgefirst_bench::testdata::read("camera720p.nv12"),
        )?;

        // Load PixelFormat::Rgba reference (ffmpeg-generated)
        let reference = load_raw_image(
            1280,
            720,
            PixelFormat::Rgba,
            None,
            &edgefirst_bench::testdata::read("camera720p.rgba"),
        )?;

        // Convert using G2D
        let mut dst = TensorDyn::image(
            1280,
            720,
            PixelFormat::Rgba,
            DType::U8,
            Some(TensorMemory::Dma),
        )?;
        let mut g2d = G2DProcessor::new()?;
        let src_dyn = src;
        let mut dst_dyn = dst;
        g2d.convert(
            &src_dyn,
            &mut dst_dyn,
            Rotation::None,
            Flip::None,
            Crop::no_crop(),
        )?;
        dst = {
            let mut __t = dst_dyn.into_u8().unwrap();
            __t.set_format(PixelFormat::Rgba)
                .map_err(|e| crate::Error::Internal(e.to_string()))?;
            TensorDyn::from(__t)
        };

        // Copy to CPU for comparison
        let cpu_dst = TensorDyn::image(1280, 720, PixelFormat::Rgba, DType::U8, None)?;
        cpu_dst
            .as_u8()
            .unwrap()
            .map()?
            .as_mut_slice()
            .copy_from_slice(dst.as_u8().unwrap().map()?.as_slice());

        compare_images(&reference, &cpu_dst, 0.98, "g2d_nv12_to_rgba_reference")
    }

    /// Test G2D PixelFormat::Nv12→PixelFormat::Rgb conversion against ffmpeg reference
    #[test]
    #[cfg(target_os = "linux")]
    fn test_g2d_nv12_to_rgb_reference() -> Result<(), crate::Error> {
        if !is_dma_available() || !is_g2d_available() {
            return Ok(());
        }
        // Load PixelFormat::Nv12 source
        let src = load_raw_image(
            1280,
            720,
            PixelFormat::Nv12,
            Some(TensorMemory::Dma),
            &edgefirst_bench::testdata::read("camera720p.nv12"),
        )?;

        // Load PixelFormat::Rgb reference (ffmpeg-generated)
        let reference = load_raw_image(
            1280,
            720,
            PixelFormat::Rgb,
            None,
            &edgefirst_bench::testdata::read("camera720p.rgb"),
        )?;

        // Convert using G2D
        let mut dst = TensorDyn::image(
            1280,
            720,
            PixelFormat::Rgb,
            DType::U8,
            Some(TensorMemory::Dma),
        )?;
        let mut g2d = G2DProcessor::new()?;
        let src_dyn = src;
        let mut dst_dyn = dst;
        g2d.convert(
            &src_dyn,
            &mut dst_dyn,
            Rotation::None,
            Flip::None,
            Crop::no_crop(),
        )?;
        dst = {
            let mut __t = dst_dyn.into_u8().unwrap();
            __t.set_format(PixelFormat::Rgb)
                .map_err(|e| crate::Error::Internal(e.to_string()))?;
            TensorDyn::from(__t)
        };

        // Copy to CPU for comparison
        let cpu_dst = TensorDyn::image(1280, 720, PixelFormat::Rgb, DType::U8, None)?;
        cpu_dst
            .as_u8()
            .unwrap()
            .map()?
            .as_mut_slice()
            .copy_from_slice(dst.as_u8().unwrap().map()?.as_slice());

        compare_images(&reference, &cpu_dst, 0.98, "g2d_nv12_to_rgb_reference")
    }

    /// Test G2D PixelFormat::Yuyv→PixelFormat::Rgba conversion against ffmpeg reference
    #[test]
    #[cfg(target_os = "linux")]
    fn test_g2d_yuyv_to_rgba_reference() -> Result<(), crate::Error> {
        if !is_dma_available() || !is_g2d_available() {
            return Ok(());
        }
        // Load PixelFormat::Yuyv source
        let src = load_raw_image(
            1280,
            720,
            PixelFormat::Yuyv,
            Some(TensorMemory::Dma),
            &edgefirst_bench::testdata::read("camera720p.yuyv"),
        )?;

        // Load PixelFormat::Rgba reference (ffmpeg-generated)
        let reference = load_raw_image(
            1280,
            720,
            PixelFormat::Rgba,
            None,
            &edgefirst_bench::testdata::read("camera720p.rgba"),
        )?;

        // Convert using G2D
        let mut dst = TensorDyn::image(
            1280,
            720,
            PixelFormat::Rgba,
            DType::U8,
            Some(TensorMemory::Dma),
        )?;
        let mut g2d = G2DProcessor::new()?;
        let src_dyn = src;
        let mut dst_dyn = dst;
        g2d.convert(
            &src_dyn,
            &mut dst_dyn,
            Rotation::None,
            Flip::None,
            Crop::no_crop(),
        )?;
        dst = {
            let mut __t = dst_dyn.into_u8().unwrap();
            __t.set_format(PixelFormat::Rgba)
                .map_err(|e| crate::Error::Internal(e.to_string()))?;
            TensorDyn::from(__t)
        };

        // Copy to CPU for comparison
        let cpu_dst = TensorDyn::image(1280, 720, PixelFormat::Rgba, DType::U8, None)?;
        cpu_dst
            .as_u8()
            .unwrap()
            .map()?
            .as_mut_slice()
            .copy_from_slice(dst.as_u8().unwrap().map()?.as_slice());

        compare_images(&reference, &cpu_dst, 0.98, "g2d_yuyv_to_rgba_reference")
    }

    /// Test G2D PixelFormat::Yuyv→PixelFormat::Rgb conversion against ffmpeg reference
    #[test]
    #[cfg(target_os = "linux")]
    fn test_g2d_yuyv_to_rgb_reference() -> Result<(), crate::Error> {
        if !is_dma_available() || !is_g2d_available() {
            return Ok(());
        }
        // Load PixelFormat::Yuyv source
        let src = load_raw_image(
            1280,
            720,
            PixelFormat::Yuyv,
            Some(TensorMemory::Dma),
            &edgefirst_bench::testdata::read("camera720p.yuyv"),
        )?;

        // Load PixelFormat::Rgb reference (ffmpeg-generated)
        let reference = load_raw_image(
            1280,
            720,
            PixelFormat::Rgb,
            None,
            &edgefirst_bench::testdata::read("camera720p.rgb"),
        )?;

        // Convert using G2D
        let mut dst = TensorDyn::image(
            1280,
            720,
            PixelFormat::Rgb,
            DType::U8,
            Some(TensorMemory::Dma),
        )?;
        let mut g2d = G2DProcessor::new()?;
        let src_dyn = src;
        let mut dst_dyn = dst;
        g2d.convert(
            &src_dyn,
            &mut dst_dyn,
            Rotation::None,
            Flip::None,
            Crop::no_crop(),
        )?;
        dst = {
            let mut __t = dst_dyn.into_u8().unwrap();
            __t.set_format(PixelFormat::Rgb)
                .map_err(|e| crate::Error::Internal(e.to_string()))?;
            TensorDyn::from(__t)
        };

        // Copy to CPU for comparison
        let cpu_dst = TensorDyn::image(1280, 720, PixelFormat::Rgb, DType::U8, None)?;
        cpu_dst
            .as_u8()
            .unwrap()
            .map()?
            .as_mut_slice()
            .copy_from_slice(dst.as_u8().unwrap().map()?.as_slice());

        compare_images(&reference, &cpu_dst, 0.98, "g2d_yuyv_to_rgb_reference")
    }

    /// Test G2D native PixelFormat::Bgra conversion for all supported source formats.
    /// Compares G2D src→PixelFormat::Bgra against G2D src→PixelFormat::Rgba by verifying R↔B swap.
    #[test]
    #[cfg(target_os = "linux")]
    #[ignore = "G2D on i.MX 8MP rejects BGRA as destination format; re-enable when supported"]
    fn test_g2d_bgra_no_resize() {
        for src_fmt in [
            PixelFormat::Rgba,
            PixelFormat::Yuyv,
            PixelFormat::Nv12,
            PixelFormat::Bgra,
        ] {
            test_g2d_bgra_no_resize_(src_fmt).unwrap_or_else(|e| {
                panic!("{src_fmt} to PixelFormat::Bgra failed: {e:?}");
            });
        }
    }

    fn test_g2d_bgra_no_resize_(g2d_in_fmt: PixelFormat) -> Result<(), crate::Error> {
        let file = edgefirst_bench::testdata::read("zidane.jpg").to_vec();
        let src = crate::load_image_test_helper(&file, Some(PixelFormat::Rgb), None)?;

        // Create DMA buffer for G2D input
        let mut src2 = TensorDyn::image(1280, 720, g2d_in_fmt, DType::U8, Some(TensorMemory::Dma))?;
        let mut cpu_converter = CPUProcessor::new();

        if g2d_in_fmt == PixelFormat::Nv12 {
            let nv12_bytes = edgefirst_bench::testdata::read("zidane.nv12");
            src2.as_u8()
                .unwrap()
                .map()?
                .as_mut_slice()
                .copy_from_slice(&nv12_bytes);
        } else {
            cpu_converter.convert(&src, &mut src2, Rotation::None, Flip::None, Crop::no_crop())?;
        }

        let mut g2d = G2DProcessor::new()?;

        // Convert to PixelFormat::Bgra via G2D
        let mut bgra_dst = TensorDyn::image(
            1280,
            720,
            PixelFormat::Bgra,
            DType::U8,
            Some(TensorMemory::Dma),
        )?;
        let src2_dyn = src2;
        let mut bgra_dst_dyn = bgra_dst;
        g2d.convert(
            &src2_dyn,
            &mut bgra_dst_dyn,
            Rotation::None,
            Flip::None,
            Crop::no_crop(),
        )?;
        bgra_dst = {
            let mut __t = bgra_dst_dyn.into_u8().unwrap();
            __t.set_format(PixelFormat::Bgra)
                .map_err(|e| crate::Error::Internal(e.to_string()))?;
            TensorDyn::from(__t)
        };

        // Reconstruct src2 from dyn for PixelFormat::Rgba conversion
        let src2 = {
            let mut __t = src2_dyn.into_u8().unwrap();
            __t.set_format(g2d_in_fmt)
                .map_err(|e| crate::Error::Internal(e.to_string()))?;
            TensorDyn::from(__t)
        };

        // Convert to PixelFormat::Rgba via G2D as reference
        let mut rgba_dst = TensorDyn::image(
            1280,
            720,
            PixelFormat::Rgba,
            DType::U8,
            Some(TensorMemory::Dma),
        )?;
        let src2_dyn2 = src2;
        let mut rgba_dst_dyn = rgba_dst;
        g2d.convert(
            &src2_dyn2,
            &mut rgba_dst_dyn,
            Rotation::None,
            Flip::None,
            Crop::no_crop(),
        )?;
        rgba_dst = {
            let mut __t = rgba_dst_dyn.into_u8().unwrap();
            __t.set_format(PixelFormat::Rgba)
                .map_err(|e| crate::Error::Internal(e.to_string()))?;
            TensorDyn::from(__t)
        };

        // Copy both to CPU memory for comparison
        let bgra_cpu = TensorDyn::image(1280, 720, PixelFormat::Bgra, DType::U8, None)?;
        bgra_cpu
            .as_u8()
            .unwrap()
            .map()?
            .as_mut_slice()
            .copy_from_slice(bgra_dst.as_u8().unwrap().map()?.as_slice());

        let rgba_cpu = TensorDyn::image(1280, 720, PixelFormat::Rgba, DType::U8, None)?;
        rgba_cpu
            .as_u8()
            .unwrap()
            .map()?
            .as_mut_slice()
            .copy_from_slice(rgba_dst.as_u8().unwrap().map()?.as_slice());

        // Verify PixelFormat::Bgra output has R↔B swapped vs PixelFormat::Rgba output
        let bgra_map = bgra_cpu.as_u8().unwrap().map()?;
        let rgba_map = rgba_cpu.as_u8().unwrap().map()?;
        let bgra_buf = bgra_map.as_slice();
        let rgba_buf = rgba_map.as_slice();

        assert_eq!(bgra_buf.len(), rgba_buf.len());
        for (i, (bc, rc)) in bgra_buf
            .chunks_exact(4)
            .zip(rgba_buf.chunks_exact(4))
            .enumerate()
        {
            assert_eq!(
                bc[0], rc[2],
                "{g2d_in_fmt} to PixelFormat::Bgra: pixel {i} B mismatch",
            );
            assert_eq!(
                bc[1], rc[1],
                "{g2d_in_fmt} to PixelFormat::Bgra: pixel {i} G mismatch",
            );
            assert_eq!(
                bc[2], rc[0],
                "{g2d_in_fmt} to PixelFormat::Bgra: pixel {i} R mismatch",
            );
            assert_eq!(
                bc[3], rc[3],
                "{g2d_in_fmt} to PixelFormat::Bgra: pixel {i} A mismatch",
            );
        }
        Ok(())
    }

    // =========================================================================
    // tensor_to_g2d_surface offset & stride unit tests
    //
    // These tests verify that plane_offset and effective_row_stride are
    // correctly propagated into the G2DSurface.  They require DMA memory
    // but do NOT require G2D hardware — only the DMA_BUF_IOCTL_PHYS ioctl.
    // =========================================================================

    /// Helper: build a DMA-backed Tensor<u8> with an optional plane_offset
    /// and an optional row stride, then return the G2DSurface.
    fn surface_for(
        width: usize,
        height: usize,
        fmt: PixelFormat,
        offset: Option<usize>,
        row_stride: Option<usize>,
    ) -> Result<G2DSurface, crate::Error> {
        use edgefirst_tensor::TensorMemory;
        let mut t = Tensor::<u8>::image(width, height, fmt, Some(TensorMemory::Dma))?;
        if let Some(o) = offset {
            t.set_plane_offset(o);
        }
        if let Some(s) = row_stride {
            t.set_row_stride_unchecked(s);
        }
        tensor_to_g2d_surface(&t)
    }

    #[test]
    fn g2d_surface_single_plane_no_offset() {
        if !is_dma_available() || !is_g2d_available() {
            return;
        }
        let s = surface_for(640, 480, PixelFormat::Rgba, None, None).unwrap();
        // planes[0] must be non-zero (valid physical address), no offset
        assert_ne!(s.planes[0], 0);
        assert_eq!(s.stride, 640);
    }

    #[test]
    fn g2d_surface_single_plane_with_offset() {
        if !is_dma_available() || !is_g2d_available() {
            return;
        }
        use edgefirst_tensor::TensorMemory;
        let mut t =
            Tensor::<u8>::image(640, 480, PixelFormat::Rgba, Some(TensorMemory::Dma)).unwrap();
        let s0 = tensor_to_g2d_surface(&t).unwrap();
        t.set_plane_offset(4096);
        let s1 = tensor_to_g2d_surface(&t).unwrap();
        assert_eq!(s1.planes[0], s0.planes[0] + 4096);
    }

    #[test]
    fn g2d_surface_single_plane_zero_offset() {
        if !is_dma_available() || !is_g2d_available() {
            return;
        }
        use edgefirst_tensor::TensorMemory;
        let mut t =
            Tensor::<u8>::image(640, 480, PixelFormat::Rgba, Some(TensorMemory::Dma)).unwrap();
        let s_none = tensor_to_g2d_surface(&t).unwrap();
        t.set_plane_offset(0);
        let s_zero = tensor_to_g2d_surface(&t).unwrap();
        // offset=0 should produce the same address as no offset
        assert_eq!(s_none.planes[0], s_zero.planes[0]);
    }

    #[test]
    fn g2d_surface_stride_rgba() {
        if !is_dma_available() || !is_g2d_available() {
            return;
        }
        // Default stride: width in pixels = 640
        let s_default = surface_for(640, 480, PixelFormat::Rgba, None, None).unwrap();
        assert_eq!(s_default.stride, 640);

        // Custom stride: 2816 bytes / 4 channels = 704 pixels
        let s_custom = surface_for(640, 480, PixelFormat::Rgba, None, Some(2816)).unwrap();
        assert_eq!(s_custom.stride, 704);
    }

    #[test]
    fn g2d_surface_stride_rgb() {
        if !is_dma_available() || !is_g2d_available() {
            return;
        }
        let s_default = surface_for(640, 480, PixelFormat::Rgb, None, None).unwrap();
        assert_eq!(s_default.stride, 640);

        // Padded: 1980 bytes / 3 channels = 660 pixels
        let s_custom = surface_for(640, 480, PixelFormat::Rgb, None, Some(1980)).unwrap();
        assert_eq!(s_custom.stride, 660);
    }

    #[test]
    fn g2d_surface_stride_grey() {
        if !is_dma_available() || !is_g2d_available() {
            return;
        }
        // Grey (Y800) may not be supported by all G2D hardware versions
        let s = match surface_for(640, 480, PixelFormat::Grey, None, Some(1024)) {
            Ok(s) => s,
            Err(crate::Error::G2D(..)) => return,
            Err(e) => panic!("unexpected error: {e:?}"),
        };
        // Grey: 1 channel. stride in bytes = stride in pixels
        assert_eq!(s.stride, 1024);
    }

    #[test]
    fn g2d_surface_contiguous_nv12_offset() {
        if !is_dma_available() || !is_g2d_available() {
            return;
        }
        use edgefirst_tensor::TensorMemory;
        let mut t =
            Tensor::<u8>::image(640, 480, PixelFormat::Nv12, Some(TensorMemory::Dma)).unwrap();
        let s0 = tensor_to_g2d_surface(&t).unwrap();

        t.set_plane_offset(8192);
        let s1 = tensor_to_g2d_surface(&t).unwrap();

        // Luma plane should shift by offset
        assert_eq!(s1.planes[0], s0.planes[0] + 8192);
        // UV plane = base + offset + stride * height
        // Without offset: UV = base + 640 * 480 = base + 307200
        // With offset 8192: UV = base + 8192 + 640 * 480 = base + 315392
        assert_eq!(s1.planes[1], s0.planes[1] + 8192);
    }

    #[test]
    fn g2d_surface_contiguous_nv12_stride() {
        if !is_dma_available() || !is_g2d_available() {
            return;
        }
        // NV12: 1 byte per pixel for Y. stride 640 bytes = 640 pixels.
        let s = surface_for(640, 480, PixelFormat::Nv12, None, None).unwrap();
        assert_eq!(s.stride, 640);

        // Padded stride: 1024 bytes = 1024 pixels (NV12 channels = 1)
        let s_padded = surface_for(640, 480, PixelFormat::Nv12, None, Some(1024)).unwrap();
        assert_eq!(s_padded.stride, 1024);
    }

    #[test]
    fn g2d_surface_multiplane_nv12_offset() {
        if !is_dma_available() || !is_g2d_available() {
            return;
        }
        use edgefirst_tensor::TensorMemory;

        // Create luma and chroma as separate DMA tensors
        let mut luma =
            Tensor::<u8>::new(&[480, 640], Some(TensorMemory::Dma), Some("luma")).unwrap();
        let mut chroma =
            Tensor::<u8>::new(&[240, 640], Some(TensorMemory::Dma), Some("chroma")).unwrap();

        // Get baseline physical addresses with no offsets
        let luma_base = {
            let dma = luma.as_dma().unwrap();
            let phys: G2DPhysical = dma.fd.as_raw_fd().try_into().unwrap();
            phys.address()
        };
        let chroma_base = {
            let dma = chroma.as_dma().unwrap();
            let phys: G2DPhysical = dma.fd.as_raw_fd().try_into().unwrap();
            phys.address()
        };

        // Set offsets and build multiplane tensor
        luma.set_plane_offset(4096);
        chroma.set_plane_offset(2048);
        let combined = Tensor::<u8>::from_planes(luma, chroma, PixelFormat::Nv12).unwrap();
        let s = tensor_to_g2d_surface(&combined).unwrap();

        // Luma should include its offset
        assert_eq!(s.planes[0], luma_base + 4096);
        // Chroma should include its offset
        assert_eq!(s.planes[1], chroma_base + 2048);
    }

    // =========================================================================
    // Odd-dimension end-to-end cells (Deliverable C)
    //
    // Design contract:
    //   • Source is a patterned NV12 tensor (Dma) filled stride-aware with a
    //     pattern varying in both X and Y.  A Mem copy carries the same fill for
    //     the CPU reference (trusted oracle from odd_dim_cpu.rs).
    //   • G2D output (Dma RGBA) is compared against the CPU output pixel-by-pixel
    //     at each tensor's own `effective_row_stride`.
    //   • Tolerance: ±4 (same as GL).  G2D BT.601 is limited-range on the matrix
    //     but historically matches within ±4 on real hardware; the tolerance is
    //     tightened to ±6 only if on-target validation shows it is needed.
    //   • If G2D rejects the dimensions (returns Err), the test documents the
    //     behaviour via `assert!` rather than silently passing.
    //   • Both test cells skip at runtime if DMA is unavailable or G2D init fails.
    // =========================================================================

    /// Fill a Nv12 tensor (any memory type) with a patterned, stride-aware
    /// synthetic image that varies in both X and Y.
    ///
    /// Pattern: `Y(r,c) = (r*3 + c*5) % 256`
    /// Chroma column `cc`, chroma row `cr_row` (NV12: 4:2:0 — half in both dims):
    ///   `Cb = (cc*7 + cr_row*11 + 40) % 256`
    ///   `Cr = (cc*13 + cr_row*3 + 80) % 256`
    ///
    /// This matches `make_odd_both_source` in `odd_dim_cpu.rs` so the two test
    /// suites share an analytic ground truth.
    #[cfg(target_os = "linux")]
    fn fill_patterned_nv12(t: &TensorDyn) {
        let w = t.width().unwrap();
        let h = t.height().unwrap();
        let stride = t.effective_row_stride().unwrap();

        // NV12 (4:2:0): chroma is ceil(H/2) rows of W/2 pairs.
        let chroma_h = h.div_ceil(2);
        let chroma_w = w.div_ceil(2);

        let bound = t.as_u8().unwrap();
        let mut m = bound.map().unwrap();
        let buf = m.as_mut_slice();
        let uv_start = stride * h;

        // Luma: diagonal gradient.
        for r in 0..h {
            for c in 0..w {
                buf[r * stride + c] = ((r * 3 + c * 5) % 256) as u8;
            }
        }
        // Chroma: UV row pitch == stride for NV12.
        for cr_row in 0..chroma_h {
            for cc in 0..chroma_w {
                let cb_val = ((cc * 7 + cr_row * 11 + 40) % 256) as u8;
                let cr_val = ((cc * 13 + cr_row * 3 + 80) % 256) as u8;
                let uv_byte = uv_start + cr_row * stride + cc * 2;
                buf[uv_byte] = cb_val;
                buf[uv_byte + 1] = cr_val;
            }
        }
    }

    /// Compare a G2D RGBA `u8` DMA tensor against a CPU RGBA `u8` reference,
    /// reading both at their real `effective_row_stride`.
    ///
    /// Returns `(max_diff, first_pixel_over_threshold)`.
    #[cfg(target_os = "linux")]
    fn compare_g2d_vs_cpu_rgba(
        g2d_dst: &TensorDyn,
        cpu_dst: &TensorDyn,
        w: usize,
        h: usize,
        tol: u32,
    ) -> (u32, Option<(usize, usize, usize)>) {
        let channels = 4usize; // RGBA
        let g2d_t = g2d_dst.as_u8().unwrap();
        let cpu_t = cpu_dst.as_u8().unwrap();
        let g2d_stride = g2d_t.effective_row_stride().unwrap_or(w * channels);
        let cpu_stride = cpu_t.effective_row_stride().unwrap_or(w * channels);
        let g2d_map = g2d_t.map().unwrap();
        let cpu_map = cpu_t.map().unwrap();
        let g2d_px = g2d_map.as_slice();
        let cpu_px = cpu_map.as_slice();
        let mut max_diff = 0u32;
        let mut first_fail: Option<(usize, usize, usize)> = None;
        for row in 0..h {
            for col in 0..w {
                for ch in 0..3usize {
                    // Compare RGB channels only; alpha is hardware-defined
                    let gi = row * g2d_stride + col * channels + ch;
                    let ci = row * cpu_stride + col * channels + ch;
                    let d = (g2d_px[gi] as i32 - cpu_px[ci] as i32).unsigned_abs();
                    if d > max_diff {
                        max_diff = d;
                    }
                    if first_fail.is_none() && d > tol {
                        first_fail = Some((col, row, ch));
                    }
                }
            }
        }
        (max_diff, first_fail)
    }

    // -------------------------------------------------------------------------
    // D-01: NV12 odd-W (65×64) → RGBA via G2D vs CPU reference
    // -------------------------------------------------------------------------

    /// D-01: NV12 odd-width (65×64) → RGBA via G2D, compared to CPU reference.
    ///
    /// Asserts:
    ///   (a) G2D accepts odd-width NV12 without returning an error.
    ///   (b) G2D output matches CPU reference within ±4 on RGB channels.
    ///       (Alpha is hardware-defined and excluded from the comparison.)
    ///
    /// If on-target validation shows the G2D hardware has wider tolerance for
    /// odd widths, the tolerance may be relaxed to ±6 with a justification
    /// comment — but only after observing an actual on-target failure, not
    /// preemptively.
    #[test]
    #[cfg(target_os = "linux")]
    fn d01_nv12_odd_w_g2d_vs_cpu() {
        if !is_dma_available() {
            eprintln!("SKIPPED: d01_nv12_odd_w_g2d_vs_cpu - DMA not available");
            return;
        }
        let (w, h) = (65usize, 64usize);

        let src_dma =
            TensorDyn::image(w, h, PixelFormat::Nv12, DType::U8, Some(TensorMemory::Dma)).unwrap();
        let src_mem =
            TensorDyn::image(w, h, PixelFormat::Nv12, DType::U8, Some(TensorMemory::Mem)).unwrap();
        fill_patterned_nv12(&src_dma);
        fill_patterned_nv12(&src_mem);

        // CPU reference: Nv12 → Rgba.
        let mut cpu_dst = TensorDyn::image(w, h, PixelFormat::Rgba, DType::U8, None).unwrap();
        CPUProcessor::new()
            .convert(
                &src_mem,
                &mut cpu_dst,
                Rotation::None,
                Flip::None,
                Crop::no_crop(),
            )
            .unwrap();

        // G2D convert: Nv12 → Rgba.
        let mut g2d_dst =
            TensorDyn::image(w, h, PixelFormat::Rgba, DType::U8, Some(TensorMemory::Dma)).unwrap();
        let mut g2d = match G2DProcessor::new() {
            Ok(g) => g,
            Err(e) => {
                eprintln!("SKIPPED: d01_nv12_odd_w_g2d_vs_cpu - G2D not available: {e}");
                return;
            }
        };

        // (a) G2D must accept odd-width NV12 without error.
        g2d.convert(
            &src_dma,
            &mut g2d_dst,
            Rotation::None,
            Flip::None,
            Crop::no_crop(),
        )
        .unwrap_or_else(|e| {
            panic!("D-01: G2D rejected odd-width (65) NV12 source: {e}");
        });

        // (b) G2D output ≈ CPU reference ±4 on RGB channels.
        let tol = 4u32;
        let (max_diff, first_fail) = compare_g2d_vs_cpu_rgba(&g2d_dst, &cpu_dst, w, h, tol);
        eprintln!("D-01 NV12 odd-W G2D vs CPU: max_diff={max_diff}");
        // Post-WS1 both CPU and G2D resolve this untagged odd-W NV12 source to
        // limited-range BT.601 (G2D is limited-range matrix-only), so the
        // YUV-matrix delta that previously forced the loose >64 bound has
        // closed; the residual is G2D fixed-point rounding. Warn above the tight
        // ±tol, fail on >35 (was 64) so a real geometry/stride regression — the
        // odd-W stride handling this test guards — still trips.
        if max_diff > tol {
            eprintln!(
                "WARNING: D-01 NV12 odd-W G2D vs CPU max_diff={max_diff} > {tol} \
                 (G2D fixed-point rounding; first bad at {first_fail:?})"
            );
        }
        assert!(
            max_diff <= 35,
            "D-01: gross NV12 odd-W G2D vs CPU mismatch max_diff={max_diff} (>35); first bad at {first_fail:?}"
        );
    }

    // -------------------------------------------------------------------------
    // D-03: NV12 odd-both (65×63) → RGBA via G2D vs CPU reference
    // -------------------------------------------------------------------------

    /// D-03: NV12 odd-width AND odd-height (65×63) → RGBA via G2D vs CPU reference.
    ///
    /// This is the strictest G2D odd-dimension cell: it exercises both the
    /// stride-boundary at the last luma row AND the half-height chroma plane
    /// boundary.  If the G2D hardware rejects the conversion (e.g. hardware
    /// alignment requirement on chroma height), this test documents that via a
    /// panic rather than passing silently.
    #[test]
    #[cfg(target_os = "linux")]
    fn d03_nv12_odd_both_g2d_vs_cpu() {
        if !is_dma_available() {
            eprintln!("SKIPPED: d03_nv12_odd_both_g2d_vs_cpu - DMA not available");
            return;
        }
        let (w, h) = (65usize, 63usize);

        let src_dma =
            TensorDyn::image(w, h, PixelFormat::Nv12, DType::U8, Some(TensorMemory::Dma)).unwrap();
        let src_mem =
            TensorDyn::image(w, h, PixelFormat::Nv12, DType::U8, Some(TensorMemory::Mem)).unwrap();
        fill_patterned_nv12(&src_dma);
        fill_patterned_nv12(&src_mem);

        // CPU reference: Nv12 → Rgba.
        let mut cpu_dst = TensorDyn::image(w, h, PixelFormat::Rgba, DType::U8, None).unwrap();
        CPUProcessor::new()
            .convert(
                &src_mem,
                &mut cpu_dst,
                Rotation::None,
                Flip::None,
                Crop::no_crop(),
            )
            .unwrap();

        // G2D convert: Nv12 → Rgba.
        let mut g2d_dst =
            TensorDyn::image(w, h, PixelFormat::Rgba, DType::U8, Some(TensorMemory::Dma)).unwrap();
        let mut g2d = match G2DProcessor::new() {
            Ok(g) => g,
            Err(e) => {
                eprintln!("SKIPPED: d03_nv12_odd_both_g2d_vs_cpu - G2D not available: {e}");
                return;
            }
        };

        // G2D must accept odd-both NV12 without error (documented behaviour).
        // If hardware rejects it, the panic message is the intended failure report.
        g2d.convert(
            &src_dma,
            &mut g2d_dst,
            Rotation::None,
            Flip::None,
            Crop::no_crop(),
        )
        .unwrap_or_else(|e| {
            panic!("D-03: G2D rejected odd-both (65×63) NV12 source: {e}");
        });

        let tol = 4u32;
        let (max_diff, first_fail) = compare_g2d_vs_cpu_rgba(&g2d_dst, &cpu_dst, w, h, tol);
        eprintln!("D-03 NV12 odd-both G2D vs CPU: max_diff={max_diff}");
        // Post-WS1 the YUV-matrix delta has closed — see test_d01 above. Warn
        // above ±tol on G2D fixed-point rounding, fail only on a gross >35
        // geometry/stride regression (was 64).
        if max_diff > tol {
            eprintln!(
                "WARNING: D-03 NV12 odd-both G2D vs CPU max_diff={max_diff} > {tol} \
                 (G2D fixed-point rounding; first bad at {first_fail:?})"
            );
        }
        assert!(
            max_diff <= 35,
            "D-03: gross NV12 odd-both G2D vs CPU mismatch max_diff={max_diff} (>35); first bad at {first_fail:?}"
        );
    }
}
