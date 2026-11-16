// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! Optional nvJPEG GPU JPEG-decoder backend (Linux + CUDA only).
//!
//! Decodes a JPEG into the destination tensor's CUDA device pointer, emitting
//! interleaved RGB. The destination is a `TensorMemory::Pbo` tensor; its CUDA
//! GL-buffer registration is mapped to a device pointer via
//! [`Tensor::cuda_map`], nvJPEG decodes into it on a per-decoder CUDA stream,
//! and the PBO is then unmapped.
//!
//! The CPU and V4L2 paths emit native NV12/Grey/NV24; this backend produces
//! packed `Rgb`.
//!
//! Capability-probed and `dlopen`-based: without libnvjpeg/libcudart, or with
//! a non-CUDA destination, the decoder falls through to the V4L2/CPU
//! backends.

mod ffi;
mod loader;

pub use loader::is_nvjpeg_available;

use crate::error::CodecError;
use crate::jpeg::markers::JpegHeaders;
use crate::options::ImageInfo;
use crate::pixel::ImagePixel;
use edgefirst_tensor::{PixelFormat, Tensor};
use ffi::*;
use std::os::raw::c_uchar;

/// Consecutive nvJPEG failures after which the backend is demoted to the
/// V4L2/CPU decoders for the rest of the session.
const MAX_CONSECUTIVE_FAILURES: u32 = 8;

/// Failure of an nvJPEG decode attempt.
pub(crate) enum NvJpegDecode {
    /// nvJPEG could not decode this image; decode it on V4L2/CPU. Carries a
    /// short reason.
    Fallback(String),
    /// A deterministic error to surface to the caller (e.g. tensor mapping).
    Fatal(CodecError),
}

/// Internal decode error, mapped to [`NvJpegDecode`] by the public entry point.
enum DecodeErr {
    /// An nvJPEG/CUDA failure: count it and fall back.
    Reset(String),
    /// A configuration this backend does not drive (e.g. RGB exceeds the PBO
    /// mapping): fall back without counting a failure.
    Unsupported(String),
    /// An error to surface (e.g. tensor reconfigure failure).
    Fatal(CodecError),
}

/// Lazily-probed nvJPEG backend state: the library is probed at most once and
/// the context (handle/state/stream) is reused across decodes.
#[allow(clippy::large_enum_variant)]
#[derive(Default)]
pub(crate) enum NvJpegProbe {
    #[default]
    Unprobed,
    Unavailable,
    Ready(NvJpegContext),
}

/// A persistent nvJPEG decode context: the library handle, a reusable decode
/// state, and one CUDA stream.
pub(crate) struct NvJpegContext {
    handle: NvjpegHandle,
    state: NvjpegJpegState,
    stream: edgefirst_tensor::CudaStream,
    failures: u32,
}

// SAFETY: the nvJPEG handle/state and the CUDA stream are owned exclusively by
// the per-`ImageDecoder` probe and only touched from its owning thread.
unsafe impl Send for NvJpegContext {}

impl NvJpegContext {
    /// Probe once: load libnvjpeg + libcudart, create a stream, an nvJPEG handle
    /// (DEFAULT backend) and a reusable decode state. `None` on any failure.
    fn new() -> Option<Self> {
        let lib = loader::lib()?;
        if !edgefirst_tensor::is_cuda_available() {
            return None;
        }
        let stream = edgefirst_tensor::stream_create()?;
        let mut handle: NvjpegHandle = std::ptr::null_mut();
        // SAFETY: valid out-pointer; allocators null, flags 0 per nvjpeg.h.
        let st = unsafe {
            (lib.create_ex)(
                NVJPEG_BACKEND_DEFAULT,
                std::ptr::null(),
                std::ptr::null(),
                0,
                &mut handle,
            )
        };
        if st != NVJPEG_STATUS_SUCCESS {
            log::debug!("nvjpegCreateEx failed (status {st}); nvjpeg unavailable");
            // SAFETY: stream is live and unused after this.
            unsafe { edgefirst_tensor::stream_destroy(stream) };
            return None;
        }
        let mut state: NvjpegJpegState = std::ptr::null_mut();
        // SAFETY: valid handle and out-pointer.
        if unsafe { (lib.jpeg_state_create)(handle, &mut state) } != NVJPEG_STATUS_SUCCESS {
            // SAFETY: handle/stream are live and unused after this.
            unsafe {
                (lib.destroy)(handle);
                edgefirst_tensor::stream_destroy(stream);
            }
            return None;
        }
        log::info!("nvjpeg context ready (BACKEND_DEFAULT)");
        Some(Self {
            handle,
            state,
            stream,
            failures: 0,
        })
    }

    /// Decode `data` into `dst` as packed RGB. Reconfigures `dst` to `Rgb`; on
    /// any failure after that reconfigure, restores `output_fmt` (the native
    /// NV12/Grey/NV24).
    fn decode<T: ImagePixel>(
        &self,
        data: &[u8],
        dst: &mut Tensor<T>,
        output_fmt: PixelFormat,
        img_w: usize,
        img_h: usize,
    ) -> Result<ImageInfo, DecodeErr> {
        let _span = tracing::trace_span!(
            "codec.decode_jpeg.nvjpeg",
            w = img_w,
            h = img_h,
            n_bytes = data.len(),
            target = "rgbi",
        )
        .entered();

        // Every failure path restores the caller's native format.
        match self.decode_reconfigured(data, dst, img_w, img_h) {
            Ok(info) => Ok(info),
            Err(e) => {
                let _ = dst.configure_image(img_w, img_h, output_fmt);
                Err(e)
            }
        }
    }

    /// Reconfigure `dst` to `Rgb` and decode into its device pointer. An error
    /// leaves `dst` configured as `Rgb`; [`decode`](Self::decode) restores the
    /// native format.
    fn decode_reconfigured<T: ImagePixel>(
        &self,
        data: &[u8],
        dst: &mut Tensor<T>,
        img_w: usize,
        img_h: usize,
    ) -> Result<ImageInfo, DecodeErr> {
        let lib =
            loader::lib().ok_or_else(|| DecodeErr::Reset("nvjpeg library vanished".into()))?;

        // Reconfigure the destination to Rgb. A failed reconfigure falls back to
        // V4L2/CPU.
        dst.configure_image(img_w, img_h, PixelFormat::Rgb)
            .map_err(|e| DecodeErr::Unsupported(format!("Rgb reconfigure failed: {e}")))?;

        // The row stride `configure_image` set.
        let rgb_stride = dst
            .effective_row_stride()
            .ok_or_else(|| DecodeErr::Reset("no row stride after Rgb reconfigure".into()))?;

        self.decode_into_device(lib, dst, data, img_w, img_h, rgb_stride)?;
        Ok(ImageInfo {
            width: img_w,
            height: img_h,
            format: PixelFormat::Rgb,
            row_stride: rgb_stride,
            rotation_degrees: 0,
            flip_horizontal: false,
        })
    }

    /// Map the destination PBO to a device pointer, decode RGB into it on the
    /// context's stream, synchronise, and unmap.
    fn decode_into_device<T: ImagePixel>(
        &self,
        lib: &loader::NvjpegLib,
        dst: &Tensor<T>,
        data: &[u8],
        img_w: usize,
        img_h: usize,
        rgb_stride: usize,
    ) -> Result<(), DecodeErr> {
        // Map the PBO's CUDA registration to a device pointer; the map stays
        // live across the decode and sync.
        let cuda = {
            let _s = tracing::trace_span!("codec.decode_jpeg.nvjpeg_map").entered();
            dst.cuda_map()
                .ok_or_else(|| DecodeErr::Reset("cuda_map returned None".into()))?
        };
        let base = cuda.device_ptr() as *mut c_uchar;
        if base.is_null() {
            return Err(DecodeErr::Reset(
                "cuda_map returned a null device pointer".into(),
            ));
        }
        let map_len = cuda.len();

        // Bounds-check the packed RGB write against the PBO allocation: an
        // overflowing size is `InvalidData`; a write past the mapping falls back
        // to V4L2/CPU.
        let offset = dst.plane_offset().unwrap_or(0);
        let needed = img_h.checked_mul(rgb_stride).ok_or_else(|| {
            DecodeErr::Fatal(CodecError::InvalidData(format!(
                "nvjpeg: RGB geometry {img_w}x{img_h} (stride {rgb_stride}) overflows usize"
            )))
        })?;
        if offset.checked_add(needed).is_none_or(|end| end > map_len) {
            return Err(DecodeErr::Unsupported(format!(
                "RGB {img_w}x{img_h} ({needed} B) + offset {offset} exceeds PBO mapping ({map_len} B)"
            )));
        }

        // Header info: confirm the dimensions match the parsed JPEG. Subsampling
        // is recorded on the span only.
        let mut ncomp: i32 = 0;
        let mut subsampling: i32 = 0;
        let mut widths = [0i32; NVJPEG_MAX_COMPONENT];
        let mut heights = [0i32; NVJPEG_MAX_COMPONENT];
        // SAFETY: valid handle, in-bounds buffer, and out-arrays of length
        // NVJPEG_MAX_COMPONENT per nvjpeg.h.
        let st = unsafe {
            (lib.get_image_info)(
                self.handle,
                data.as_ptr(),
                data.len(),
                &mut ncomp,
                &mut subsampling,
                widths.as_mut_ptr(),
                heights.as_mut_ptr(),
            )
        };
        if st != NVJPEG_STATUS_SUCCESS {
            return Err(DecodeErr::Reset(format!("nvjpegGetImageInfo status {st}")));
        }
        if widths[0] as usize != img_w || heights[0] as usize != img_h {
            return Err(DecodeErr::Reset(format!(
                "nvjpeg dims {}x{} != header {img_w}x{img_h}",
                widths[0], heights[0]
            )));
        }

        // Single interleaved RGB plane at the tensor's RGB pitch, at the (batch)
        // offset within the PBO.
        let mut image = NvjpegImage::default();
        // SAFETY: `offset + needed <= map_len` checked above.
        image.channel[0] = unsafe { base.add(offset) };
        image.pitch[0] = rgb_stride;

        {
            let _s = tracing::trace_span!("codec.decode_jpeg.nvjpeg_submit").entered();
            // SAFETY: valid handle/state, in-bounds input, valid device image,
            // and a live stream.
            let st = unsafe {
                (lib.decode)(
                    self.handle,
                    self.state,
                    data.as_ptr(),
                    data.len(),
                    NVJPEG_OUTPUT_RGBI,
                    &mut image,
                    self.stream,
                )
            };
            if st != NVJPEG_STATUS_SUCCESS {
                return Err(DecodeErr::Reset(format!("nvjpegDecode status {st}")));
            }
        }

        {
            let _s = tracing::trace_span!("codec.decode_jpeg.nvjpeg_sync").entered();
            // SAFETY: the stream is live for the context's lifetime.
            if !unsafe { edgefirst_tensor::stream_synchronize(self.stream) } {
                return Err(DecodeErr::Reset("cudaStreamSynchronize failed".into()));
            }
        }

        // Unmap the PBO after the sync.
        let _s = tracing::trace_span!("codec.decode_jpeg.nvjpeg_unmap").entered();
        drop(cuda);
        Ok(())
    }
}

impl Drop for NvJpegContext {
    fn drop(&mut self) {
        if let Some(lib) = loader::lib() {
            // SAFETY: handle/state are live and unused after this.
            unsafe {
                (lib.jpeg_state_destroy)(self.state);
                (lib.destroy)(self.handle);
            }
        }
        // SAFETY: stream is live and unused after this.
        unsafe { edgefirst_tensor::stream_destroy(self.stream) };
    }
}

impl NvJpegProbe {
    fn ensure_probed(&mut self) -> Option<&mut NvJpegContext> {
        if matches!(self, NvJpegProbe::Unprobed) {
            *self = match NvJpegContext::new() {
                Some(ctx) => NvJpegProbe::Ready(ctx),
                None => NvJpegProbe::Unavailable,
            };
        }
        match self {
            NvJpegProbe::Ready(ctx) => Some(ctx),
            _ => None,
        }
    }

    /// Attempt to decode `data` into `dst` using nvJPEG.
    ///
    /// - `Ok(Some(info))` — decoded on the GPU into the PBO as RGB.
    /// - `Ok(None)` — no nvJPEG, or the destination is not CUDA-backed; the
    ///   caller uses the V4L2/CPU decoders.
    /// - `Err(NvJpegDecode::Fallback)` — transient or unsupported; decode this
    ///   image on V4L2/CPU.
    /// - `Err(NvJpegDecode::Fatal)` — an error to surface.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn try_decode<T: ImagePixel>(
        &mut self,
        data: &[u8],
        _headers: &JpegHeaders,
        dst: &mut Tensor<T>,
        output_fmt: PixelFormat,
        img_w: usize,
        img_h: usize,
        _dst_stride: usize,
    ) -> Result<Option<ImageInfo>, NvJpegDecode> {
        let Some(ctx) = self.ensure_probed() else {
            return Ok(None);
        };

        // Only CUDA-backed destinations are decoded into; anything else is
        // left untouched.
        if !dst.is_cuda_attached() {
            return Ok(None);
        }

        let result = ctx.decode::<T>(data, dst, output_fmt, img_w, img_h);
        let out = match result {
            Ok(info) => {
                ctx.failures = 0;
                Ok(Some(info))
            }
            Err(DecodeErr::Reset(why)) => {
                ctx.failures = ctx.failures.saturating_add(1);
                Err(NvJpegDecode::Fallback(why))
            }
            // Unsupported configuration: fall back without counting a failure.
            Err(DecodeErr::Unsupported(why)) => Err(NvJpegDecode::Fallback(why)),
            Err(DecodeErr::Fatal(e)) => Err(NvJpegDecode::Fatal(e)),
        };

        // Demote a decoder that reached the failure limit.
        if let NvJpegProbe::Ready(ctx) = self {
            if ctx.failures >= MAX_CONSECUTIVE_FAILURES {
                log::warn!(
                    "nvjpeg decoder disabled after {} consecutive failures",
                    ctx.failures
                );
                *self = NvJpegProbe::Unavailable;
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use edgefirst_tensor::TensorMemory;

    /// A `Mem` tensor has no CUDA handle: `try_decode` returns `Ok(None)` and
    /// leaves the tensor's format untouched.
    #[test]
    fn mem_tensor_falls_through_untouched() {
        let mut probe = NvJpegProbe::default();
        let mut dst = Tensor::<u8>::image(
            64,
            64,
            PixelFormat::Nv12,
            Some(TensorMemory::Mem),
            edgefirst_tensor::CpuAccess::ReadWrite,
        )
        .unwrap();
        let headers =
            crate::jpeg::markers::parse_markers(MINIMAL_JPEG).expect("MINIMAL_JPEG must parse");
        let r = probe.try_decode::<u8>(
            MINIMAL_JPEG,
            &headers,
            &mut dst,
            PixelFormat::Nv12,
            8,
            8,
            16,
        );
        assert!(matches!(r, Ok(None)));
        assert_eq!(dst.format(), Some(PixelFormat::Nv12));
    }

    // 8x8 grayscale baseline JPEG.
    const MINIMAL_JPEG: &[u8] = &[
        0xFF, 0xD8, 0xFF, 0xDB, 0x00, 0x43, 0x00, 0x08, 0x06, 0x06, 0x07, 0x06, 0x05, 0x08, 0x07,
        0x07, 0x07, 0x09, 0x09, 0x08, 0x0A, 0x0C, 0x14, 0x0D, 0x0C, 0x0B, 0x0B, 0x0C, 0x19, 0x12,
        0x13, 0x0F, 0x14, 0x1D, 0x1A, 0x1F, 0x1E, 0x1D, 0x1A, 0x1C, 0x1C, 0x20, 0x24, 0x2E, 0x27,
        0x20, 0x22, 0x2C, 0x23, 0x1C, 0x1C, 0x28, 0x37, 0x29, 0x2C, 0x30, 0x31, 0x34, 0x34, 0x34,
        0x1F, 0x27, 0x39, 0x3D, 0x38, 0x32, 0x3C, 0x2E, 0x33, 0x34, 0x32, 0xFF, 0xC0, 0x00, 0x0B,
        0x08, 0x00, 0x08, 0x00, 0x08, 0x01, 0x01, 0x11, 0x00, 0xFF, 0xC4, 0x00, 0x1F, 0x00, 0x00,
        0x01, 0x05, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0A, 0x0B, 0xFF, 0xDA, 0x00,
        0x08, 0x01, 0x01, 0x00, 0x00, 0x3F, 0x00, 0xD2, 0xCF, 0x20, 0xFF, 0xD9,
    ];

    /// `NvJpegProbe::Unavailable` returns `Ok(None)` without touching the tensor.
    #[test]
    fn unavailable_probe_returns_none_without_touching_tensor() {
        let mut probe = NvJpegProbe::Unavailable;
        let mut dst = Tensor::<u8>::image(
            64,
            64,
            PixelFormat::Nv12,
            Some(TensorMemory::Mem),
            edgefirst_tensor::CpuAccess::ReadWrite,
        )
        .unwrap();
        let headers =
            crate::jpeg::markers::parse_markers(MINIMAL_JPEG).expect("MINIMAL_JPEG must parse");
        let r = probe.try_decode::<u8>(
            MINIMAL_JPEG,
            &headers,
            &mut dst,
            PixelFormat::Nv12,
            8,
            8,
            16,
        );
        assert!(
            matches!(r, Ok(None)),
            "Unavailable probe must return Ok(None)"
        );
        assert_eq!(dst.format(), Some(PixelFormat::Nv12));
    }

    /// `MAX_CONSECUTIVE_FAILURES` allows at least one retry.
    #[test]
    fn circuit_breaker_threshold_constant_is_positive() {
        const _: () = assert!(MAX_CONSECUTIVE_FAILURES >= 2);
    }

    /// Without CUDA/libnvjpeg the first `try_decode` call leaves the probe
    /// `Unavailable` and returns `Ok(None)`.
    #[test]
    fn unprobed_probe_degrades_to_unavailable_without_nvjpeg() {
        if is_nvjpeg_available() {
            // Skip on a CUDA host.
            return;
        }
        let mut probe = NvJpegProbe::default();
        assert!(
            matches!(probe, NvJpegProbe::Unprobed),
            "default must be Unprobed"
        );
        let mut dst = Tensor::<u8>::image(
            64,
            64,
            PixelFormat::Nv12,
            Some(TensorMemory::Mem),
            edgefirst_tensor::CpuAccess::ReadWrite,
        )
        .unwrap();
        let headers =
            crate::jpeg::markers::parse_markers(MINIMAL_JPEG).expect("MINIMAL_JPEG must parse");
        let r = probe.try_decode::<u8>(
            MINIMAL_JPEG,
            &headers,
            &mut dst,
            PixelFormat::Nv12,
            8,
            8,
            16,
        );
        assert!(matches!(r, Ok(None)), "no-CUDA probe must return Ok(None)");
        assert!(
            matches!(probe, NvJpegProbe::Unavailable),
            "probe must be Unavailable after first call on a non-CUDA host"
        );
    }
}
