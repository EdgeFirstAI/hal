// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! Optional nvJPEG GPU JPEG-decoder backend (Linux + CUDA only).
//!
//! When the CUDA nvJPEG library is present (NVIDIA Jetson / discrete GPU), this
//! backend decodes a JPEG straight into the destination tensor's CUDA device
//! pointer, emitting interleaved **RGB**. The destination is a
//! `TensorMemory::Pbo` tensor (what `ImageProcessor::create_image` yields on
//! Jetson, where there is no dma-heap); its CUDA GL-buffer registration is
//! mapped to a device pointer via [`Tensor::cuda_map`], nvJPEG decodes into it
//! on a per-decoder CUDA stream, and the unmapped PBO is then consumed
//! GPU-resident by `ImageProcessor::convert()` — no CPU bounce.
//!
//! Unlike the CPU/V4L2 paths (which emit native NV12/Grey/NV24 and never
//! colour-convert), this backend produces packed `Rgb`: nvJPEG does the
//! YCbCr→RGB on the GPU at near-zero marginal cost, the result is GPU-resident,
//! and `convert()` accepts an `Rgb` source directly. This is a deliberate,
//! documented exception to the codec's native-format contract.
//!
//! Capability-probed and entirely `dlopen`-based: absent libnvjpeg/libcudart,
//! or a non-CUDA destination, the decoder transparently falls through to the
//! V4L2/CPU backends.

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
/// V4L2/CPU decoders for the rest of the session (circuit breaker).
const MAX_CONSECUTIVE_FAILURES: u32 = 8;

/// Outcome of a hardware decode attempt, distinct from [`CodecError`] so the
/// caller can react to a transient failure by retrying on V4L2/CPU.
pub(crate) enum NvJpegDecode {
    /// nvJPEG could not decode this image; decode it on V4L2/CPU. Carries a
    /// short reason.
    Fallback(String),
    /// A deterministic error to surface to the caller (e.g. tensor mapping).
    Fatal(CodecError),
}

/// Internal decode error, mapped to [`NvJpegDecode`] by the public entry point.
enum DecodeErr {
    /// An nvJPEG/CUDA failure: count toward the circuit breaker and fall back.
    Reset(String),
    /// A configuration we don't drive (e.g. RGB exceeds the PBO mapping): fall
    /// back without counting it as a failure — the device is fine.
    Unsupported(String),
    /// A real error to surface (e.g. tensor reconfigure failure).
    Fatal(CodecError),
}

/// Lazily-probed nvJPEG backend state, stored on the reusable decoder state so
/// the library is probed at most once and the context (handle/state/stream) is
/// reused across decodes.
//
// `Ready` is much larger than the unit variants, but there is exactly one probe
// per decoder and it is `Ready` in the steady state, so boxing would add
// indirection on the hot path for no real saving.
#[allow(clippy::large_enum_variant)]
#[derive(Default)]
pub(crate) enum NvJpegProbe {
    #[default]
    Unprobed,
    Unavailable,
    Ready(NvJpegContext),
}

/// A persistent nvJPEG decode context: the library handle, a reusable decode
/// state (keeps nvJPEG's internal device scratch hot), and one CUDA stream so
/// concurrent decoders do not serialise on the default stream.
pub(crate) struct NvJpegContext {
    handle: NvjpegHandle,
    state: NvjpegJpegState,
    stream: edgefirst_tensor::CudaStream,
    failures: u32,
}

// SAFETY: the nvJPEG handle/state and the CUDA stream are owned exclusively by
// the per-`ImageDecoder` probe and only touched from its owning thread; the
// device pointers they operate on are process-global via the primary context.
// Send (not Sync) matches the one-decoder-per-thread usage.
unsafe impl Send for NvJpegContext {}

impl NvJpegContext {
    /// Probe once: load libnvjpeg + libcudart, create a stream, an nvJPEG handle
    /// (DEFAULT backend — the dedicated-hardware backend is unsupported on
    /// Orin), and a reusable decode state. `None` on any failure.
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
    /// NV12/Grey/NV24) so the V4L2/CPU fall-through sees a clean tensor.
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

        // Run the reconfigure + device decode in a helper so EVERY failure path
        // — including a failed `Rgb` reconfigure — restores the caller's native
        // format, leaving a correctly-shaped tensor for the V4L2/CPU fall-through.
        match self.decode_reconfigured(data, dst, img_w, img_h) {
            Ok(info) => Ok(info),
            Err(e) => {
                let _ = dst.configure_image(img_w, img_h, output_fmt);
                Err(e)
            }
        }
    }

    /// Reconfigure `dst` to `Rgb` and decode into its device pointer. Any error
    /// leaves `dst` configured as `Rgb` (possibly partially) — the caller's
    /// [`decode`](Self::decode) restores the native format.
    fn decode_reconfigured<T: ImagePixel>(
        &self,
        data: &[u8],
        dst: &mut Tensor<T>,
        img_w: usize,
        img_h: usize,
    ) -> Result<ImageInfo, DecodeErr> {
        let lib =
            loader::lib().ok_or_else(|| DecodeErr::Reset("nvjpeg library vanished".into()))?;

        // Reconfigure the destination NV12 → Rgb. `configure_image` preserves the
        // tensor's CUDA registration (same PBO + device pointer), and chooses the
        // row stride from the backing: a CUDA-backed DMA destination is rounded up
        // to a 64-byte-aligned pitch, a PBO stays tight (`width*3`). A failed
        // reconfigure — e.g. an NV12 buffer (1.5 B/px) is too small for RGB
        // (3 B/px) — is NOT fatal: fall back to V4L2/CPU.
        dst.configure_image(img_w, img_h, PixelFormat::Rgb)
            .map_err(|e| DecodeErr::Unsupported(format!("Rgb reconfigure failed: {e}")))?;

        // Use the stride `configure_image` actually set, NOT an assumed `width*3`:
        // on an alignment-padded (DMA) destination the rows are 64-aligned, and
        // writing at `width*3` would shear rows that `convert()` then samples at
        // the wider pitch.
        let rgb_stride = dst
            .effective_row_stride()
            .ok_or_else(|| DecodeErr::Reset("no row stride after Rgb reconfigure".into()))?;

        // The device decode borrows `dst` immutably (via `cuda_map`); scope it so
        // the format can be restored mutably on the error path by the caller.
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
    /// context's stream, synchronise, and unmap. Takes `&Tensor` so the
    /// immutable `cuda_map` borrow is scoped to this call.
    fn decode_into_device<T: ImagePixel>(
        &self,
        lib: &loader::NvjpegLib,
        dst: &Tensor<T>,
        data: &[u8],
        img_w: usize,
        img_h: usize,
        rgb_stride: usize,
    ) -> Result<(), DecodeErr> {
        // Map the PBO's CUDA registration to a device pointer (routes to the GL
        // worker thread that owns the PBO). The map must stay live across the
        // whole decode+sync — `device_ptr` is only valid while mapped, and
        // dropping it unmaps so `convert()` can re-take the PBO.
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

        // Bounds-check the packed RGB write against the true PBO allocation
        // (`configure_image` does NOT guard a packed format on GL memory).
        let offset = dst.plane_offset().unwrap_or(0);
        let needed = img_h.checked_mul(rgb_stride).ok_or(DecodeErr::Fatal(
            CodecError::InsufficientCapacity {
                image: (img_w, img_h),
                tensor: (0, 0),
            },
        ))?;
        if offset.checked_add(needed).is_none_or(|end| end > map_len) {
            return Err(DecodeErr::Unsupported(format!(
                "RGB {img_w}x{img_h} ({needed} B) + offset {offset} exceeds PBO mapping ({map_len} B)"
            )));
        }

        // Header info: confirm the dims match the parsed JPEG. Subsampling is
        // recorded on the span only; RGBI output normalises every subsampling.
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
            // and a live stream. Progressive/non-baseline JPEGs the GPU_HYBRID
            // backend rejects surface here and fall back to the CPU decoder.
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

        // Unmap the PBO (routes to the GL worker thread) so `convert()` can
        // sample it. Explicit drop after the sync guarantees the device write
        // is complete and visible first.
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
    /// - `Err(NvJpegDecode::Fallback)` — transient/unsupported; decode this
    ///   image on V4L2/CPU.
    /// - `Err(NvJpegDecode::Fatal)` — a real error to surface.
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

        // Only CUDA-backed (PBO/DMA-with-handle) destinations can be decoded
        // into zero-copy; anything else falls through to V4L2/CPU. Checked
        // before any reconfigure so a non-CUDA tensor is left untouched.
        if dst.cuda().is_none() {
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
            // A config we don't drive: fall back but do NOT count it — nvJPEG
            // is healthy, the CPU handles this image.
            Err(DecodeErr::Unsupported(why)) => Err(NvJpegDecode::Fallback(why)),
            Err(DecodeErr::Fatal(e)) => Err(NvJpegDecode::Fatal(e)),
        };

        // Circuit breaker: demote a persistently failing decoder for the session
        // so it stops costing per-image latency.
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

    /// A plain `Mem` tensor has no CUDA handle, so `try_decode` returns
    /// `Ok(None)` and leaves the tensor's format untouched for the V4L2/CPU
    /// fall-through — regardless of whether libnvjpeg is present.
    #[test]
    fn mem_tensor_falls_through_untouched() {
        let mut probe = NvJpegProbe::default();
        let mut dst =
            Tensor::<u8>::image(64, 64, PixelFormat::Nv12, Some(TensorMemory::Mem)).unwrap();
        // The `cuda()` gate fires before the headers are consulted, but parse the
        // fixture unconditionally so a broken `MINIMAL_JPEG` fails the test loudly
        // instead of silently skipping the `try_decode` call.
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
        // Format must be unchanged (still Nv12) for the fall-through.
        assert_eq!(dst.format(), Some(PixelFormat::Nv12));
    }

    // 8x8 grayscale baseline JPEG (smallest valid stream) for header parsing.
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
}
