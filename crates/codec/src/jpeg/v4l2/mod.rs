// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! Optional V4L2 hardware JPEG-decoder backend (Linux only).
//!
//! When a Linux device exposes a JPEG decoder through the standard V4L2
//! mem2mem API, this backend offloads the expensive Huffman + IDCT + chroma
//! upsample to hardware and runs only a light colour-conversion pass on the
//! CPU. It is probed lazily and entirely capability-based, so it is portable
//! across SoCs; anything it cannot drive transparently falls back to the
//! existing software decoder.
//!
//! Phase 1 implements a per-image MMAP decode for multi-planar M2M devices
//! (the lead target, i.MX `mxc-jpeg`). Single-planar devices, DMA-BUF
//! zero-copy, and a persistent buffer pool land in later phases; until then
//! those configurations fall back to the CPU decoder.

mod buffers;
mod device;
mod format;
mod ioctl;

use std::os::fd::{BorrowedFd, RawFd};
use std::os::raw::c_int;

use nix::poll::{poll, PollFd, PollFlags, PollTimeout};

use crate::error::CodecError;
use crate::jpeg::markers::JpegHeaders;
use crate::options::ImageInfo;
use crate::pixel::ImagePixel;
use buffers::Mmap;
use device::{ApiVariant, ProbedDevice};
use edgefirst_tensor::{PixelFormat, Tensor, TensorTrait};
use format::{CapKind, RowScratch};

/// Timeout waiting for the `SOURCE_CHANGE` event after queuing the JPEG (ms).
const SOURCE_CHANGE_TIMEOUT_MS: i32 = 200;
/// Timeout waiting for the hardware to finish a decode (ms).
const DECODE_TIMEOUT_MS: i32 = 2000;
/// Hard cap on the `DQEVENT` drain loop (an unbounded loop hangs — the driver
/// re-returns the same event repeatedly).
const MAX_EVENTS: usize = 16;

/// Outcome of a hardware decode attempt, distinct from [`CodecError`] so the
/// caller can react to a transient hardware failure by retrying on the CPU.
pub(crate) enum V4l2Decode {
    /// The hardware could not decode this image; the device has been reset and
    /// the caller should decode it on the CPU. Carries a short reason.
    Fallback(String),
    /// A deterministic input error the CPU decoder would also reject.
    Fatal(CodecError),
}

/// Internal decode error, mapped to [`V4l2Decode`] by the public entry point.
enum DecodeErr {
    /// A hardware/ioctl failure: reset the device and fall back to CPU.
    Reset(String),
    /// A configuration we don't drive (unsupported CAPTURE format, colorimetry
    /// mismatch, single-planar device, …): fall back to CPU.
    Unsupported(String),
    /// A real error to surface to the caller (e.g. tensor mapping failure).
    Fatal(CodecError),
}

/// Lazily-probed V4L2 backend state, stored on the reusable decoder state so
/// the device is probed at most once and the context is reused across decodes.
#[derive(Default)]
pub(crate) enum V4l2Probe {
    #[default]
    Unprobed,
    Unavailable,
    Ready(V4l2Context),
}

impl V4l2Probe {
    fn ensure_probed(&mut self) -> Option<&mut V4l2Context> {
        if matches!(self, V4l2Probe::Unprobed) {
            *self = match device::probe() {
                Some(dev) => V4l2Probe::Ready(V4l2Context::new(dev)),
                None => V4l2Probe::Unavailable,
            };
        }
        match self {
            V4l2Probe::Ready(ctx) => Some(ctx),
            _ => None,
        }
    }

    /// Attempt to decode `data` into `dst` using the hardware backend.
    ///
    /// - `Ok(Some(info))` — decoded in hardware.
    /// - `Ok(None)` — no usable device; the caller uses the CPU decoder.
    /// - `Err(V4l2Decode::Fallback)` — transient/unsupported; device reset,
    ///   caller decodes this image on the CPU.
    /// - `Err(V4l2Decode::Fatal)` — a real error to surface.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn try_decode<T: ImagePixel>(
        &mut self,
        data: &[u8],
        _headers: &JpegHeaders,
        dst: &mut Tensor<T>,
        output_fmt: PixelFormat,
        final_w: usize,
        final_h: usize,
        channels: usize,
        dst_stride: usize,
    ) -> Result<Option<ImageInfo>, V4l2Decode> {
        let Some(ctx) = self.ensure_probed() else {
            return Ok(None);
        };

        if ctx.device.api != ApiVariant::MultiPlanar {
            // Single-planar M2M decode is a later increment.
            return Ok(None);
        }

        let result = ctx.decode_mmap::<T>(
            data, dst, output_fmt, final_w, final_h, channels, dst_stride,
        );
        // Per-image teardown doubles as the device reset on the failure path:
        // STREAMOFF both queues + REQBUFS 0. Best-effort, errors ignored.
        ctx.teardown();

        match result {
            Ok(info) => Ok(Some(info)),
            Err(DecodeErr::Reset(why)) => Err(V4l2Decode::Fallback(why)),
            Err(DecodeErr::Unsupported(why)) => Err(V4l2Decode::Fallback(why)),
            Err(DecodeErr::Fatal(e)) => Err(V4l2Decode::Fatal(e)),
        }
    }
}

/// A persistent, capability-verified V4L2 decode context.
pub(crate) struct V4l2Context {
    device: ProbedDevice,
    /// Packed-u8 staging buffer (native stride), reused across decodes. The
    /// hardware path converts CAPTURE rows into here, then the shared typed
    /// tail writes into the destination tensor.
    staging: Vec<u8>,
    /// Per-row de-interleave scratch, reused across decodes.
    row_scratch: RowScratch,
}

impl V4l2Context {
    fn new(device: ProbedDevice) -> Self {
        Self {
            device,
            staging: Vec::new(),
            row_scratch: RowScratch::new(),
        }
    }

    /// Full multi-planar MMAP decode lifecycle for one image.
    #[allow(clippy::too_many_arguments)]
    fn decode_mmap<T: ImagePixel>(
        &mut self,
        data: &[u8],
        dst: &mut Tensor<T>,
        output_fmt: PixelFormat,
        final_w: usize,
        final_h: usize,
        channels: usize,
        dst_stride: usize,
    ) -> Result<ImageInfo, DecodeErr> {
        let fd = self.device.fd();
        const OUT: u32 = ioctl::V4L2_BUF_TYPE_VIDEO_OUTPUT_MPLANE;
        const CAP: u32 = ioctl::V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE;

        // --- Subscribe to the source-change event ---------------------------
        let sub = ioctl::v4l2_event_subscription {
            type_: ioctl::V4L2_EVENT_SOURCE_CHANGE,
            ..Default::default()
        };
        // SAFETY: `sub` is a valid, fully-initialised subscription struct.
        unsafe { ioctl::vidioc_subscribe_event(fd, &sub) }
            .map_err(|e| DecodeErr::Reset(format!("SUBSCRIBE_EVENT: {e}")))?;

        // --- OUTPUT queue: set JPEG format, allocate, map, stream, queue ----
        let mut ofmt = ioctl::v4l2_format {
            type_: OUT,
            ..Default::default()
        };
        // SAFETY: type_ selects the multi-planar variant.
        {
            let p = unsafe { ofmt.pix_mp() };
            p.width = final_w as u32;
            p.height = final_h as u32;
            p.pixelformat = ioctl::V4L2_PIX_FMT_JPEG;
            p.field = ioctl::V4L2_FIELD_NONE;
            p.num_planes = 1;
            p.plane_fmt[0].sizeimage = ((data.len() + 4095) & !4095) as u32;
        }
        // SAFETY: valid v4l2_format; kernel reads/writes it.
        unsafe { ioctl::vidioc_s_fmt(fd, &mut ofmt) }
            .map_err(|e| DecodeErr::Reset(format!("S_FMT OUTPUT: {e}")))?;

        reqbufs(fd, OUT, 1).map_err(|e| DecodeErr::Reset(format!("REQBUFS OUTPUT: {e}")))?;

        // QUERYBUF OUTPUT → mmap plane 0.
        let mut oplanes = [ioctl::v4l2_plane::default(); ioctl::VIDEO_MAX_PLANES];
        let mut ob = ioctl::v4l2_buffer {
            type_: OUT,
            memory: ioctl::V4L2_MEMORY_MMAP,
            index: 0,
            length: 1,
            ..Default::default()
        };
        ob.set_planes(oplanes.as_mut_ptr());
        // SAFETY: buffer + plane array valid for the QUERYBUF call.
        unsafe { ioctl::vidioc_querybuf(fd, &mut ob) }
            .map_err(|e| DecodeErr::Reset(format!("QUERYBUF OUTPUT: {e}")))?;
        let olen = oplanes[0].length as usize;
        let ooff = oplanes[0].mem_offset() as i64;
        let mut omap = Mmap::new(borrow(fd), olen, ooff)
            .map_err(|e| DecodeErr::Reset(format!("mmap OUTPUT: {e}")))?;

        streamon(fd, OUT).map_err(|e| DecodeErr::Reset(format!("STREAMON OUTPUT: {e}")))?;

        // Copy the JPEG bytestream in and queue it.
        omap.as_mut_slice()[..data.len()].copy_from_slice(data);
        let mut oqplanes = [ioctl::v4l2_plane::default(); ioctl::VIDEO_MAX_PLANES];
        oqplanes[0].bytesused = data.len() as u32;
        let mut oqb = ioctl::v4l2_buffer {
            type_: OUT,
            memory: ioctl::V4L2_MEMORY_MMAP,
            index: 0,
            length: 1,
            ..Default::default()
        };
        oqb.set_planes(oqplanes.as_mut_ptr());
        // SAFETY: valid buffer referencing the queued plane array.
        unsafe { ioctl::vidioc_qbuf(fd, &mut oqb) }
            .map_err(|e| DecodeErr::Reset(format!("QBUF OUTPUT: {e}")))?;

        // --- Wait for the driver to parse the header (best-effort) ----------
        poll_ready(fd, PollFlags::POLLPRI, SOURCE_CHANGE_TIMEOUT_MS);
        drain_events(fd);

        // --- CAPTURE queue: query driver-chosen format ----------------------
        let mut cfmt = ioctl::v4l2_format {
            type_: CAP,
            ..Default::default()
        };
        // SAFETY: valid v4l2_format for G_FMT.
        unsafe { ioctl::vidioc_g_fmt(fd, &mut cfmt) }
            .map_err(|e| DecodeErr::Reset(format!("G_FMT CAPTURE: {e}")))?;
        // SAFETY: type_ selects the multi-planar variant.
        let cap = *unsafe { cfmt.pix_mp() };

        let kind = format::classify(cap.pixelformat).ok_or_else(|| {
            DecodeErr::Unsupported(format!(
                "capture format {} unsupported",
                ioctl::fourcc_str(cap.pixelformat)
            ))
        })?;
        let plan = format::plan_output(output_fmt)
            .ok_or_else(|| DecodeErr::Unsupported(format!("output {output_fmt:?} unsupported")))?;
        if !colorimetry_ok(cap.colorspace, cap.ycbcr_enc, cap.quantization) {
            return Err(DecodeErr::Unsupported(format!(
                "capture colorimetry (cs={}, enc={}, quant={}) not BT.601 full-range",
                cap.colorspace, cap.ycbcr_enc, cap.quantization
            )));
        }

        let num_planes = cap.num_planes as usize;
        let cap_h = cap.height as usize;
        let luma_stride = cap.plane_fmt[0].bytesperline as usize;

        // --- CAPTURE queue: allocate, map every plane, stream, queue, dequeue
        reqbufs(fd, CAP, 1).map_err(|e| DecodeErr::Reset(format!("REQBUFS CAPTURE: {e}")))?;

        let mut cplanes = [ioctl::v4l2_plane::default(); ioctl::VIDEO_MAX_PLANES];
        let mut cqb = ioctl::v4l2_buffer {
            type_: CAP,
            memory: ioctl::V4L2_MEMORY_MMAP,
            index: 0,
            length: num_planes as u32,
            ..Default::default()
        };
        cqb.set_planes(cplanes.as_mut_ptr());
        // SAFETY: valid buffer + plane array for QUERYBUF.
        unsafe { ioctl::vidioc_querybuf(fd, &mut cqb) }
            .map_err(|e| DecodeErr::Reset(format!("QUERYBUF CAPTURE: {e}")))?;

        let mut maps: Vec<Mmap> = Vec::with_capacity(num_planes);
        for (p, plane) in cplanes.iter().take(num_planes).enumerate() {
            let len = plane.length as usize;
            let off = plane.mem_offset() as i64;
            maps.push(
                Mmap::new(borrow(fd), len, off)
                    .map_err(|e| DecodeErr::Reset(format!("mmap CAPTURE plane {p}: {e}")))?,
            );
        }

        streamon(fd, CAP).map_err(|e| DecodeErr::Reset(format!("STREAMON CAPTURE: {e}")))?;

        // Queue the CAPTURE buffer — both queues now streaming with buffers, so
        // the M2M hardware job runs.
        let mut cqplanes = [ioctl::v4l2_plane::default(); ioctl::VIDEO_MAX_PLANES];
        let mut cqb2 = ioctl::v4l2_buffer {
            type_: CAP,
            memory: ioctl::V4L2_MEMORY_MMAP,
            index: 0,
            length: num_planes as u32,
            ..Default::default()
        };
        cqb2.set_planes(cqplanes.as_mut_ptr());
        // SAFETY: valid buffer + plane array for QBUF.
        unsafe { ioctl::vidioc_qbuf(fd, &mut cqb2) }
            .map_err(|e| DecodeErr::Reset(format!("QBUF CAPTURE: {e}")))?;

        if !poll_ready(fd, PollFlags::POLLIN, DECODE_TIMEOUT_MS) {
            return Err(DecodeErr::Reset("CAPTURE decode timeout".into()));
        }

        let mut cdqplanes = [ioctl::v4l2_plane::default(); ioctl::VIDEO_MAX_PLANES];
        let mut cdqb = ioctl::v4l2_buffer {
            type_: CAP,
            memory: ioctl::V4L2_MEMORY_MMAP,
            length: num_planes as u32,
            ..Default::default()
        };
        cdqb.set_planes(cdqplanes.as_mut_ptr());
        // SAFETY: valid buffer + plane array for DQBUF.
        unsafe { ioctl::vidioc_dqbuf(fd, &mut cdqb) }
            .map_err(|e| DecodeErr::Reset(format!("DQBUF CAPTURE: {e}")))?;

        // --- Resolve the luma plane and (for NV12) the chroma plane ---------
        let (luma, chroma): (&[u8], Option<(&[u8], usize)>) = match (&kind, num_planes) {
            // NV12M: separate Y and CbCr buffers.
            (CapKind::Nv12, n) if n >= 2 => {
                let c_stride = cap.plane_fmt[1].bytesperline as usize;
                (maps[0].as_slice(), Some((maps[1].as_slice(), c_stride)))
            }
            // Single-buffer NV12: Y plane followed by CbCr in one allocation.
            (CapKind::Nv12, _) => {
                let m = maps[0].as_slice();
                let y_size = luma_stride * cap_h;
                (&m[..y_size], Some((&m[y_size..], luma_stride)))
            }
            // Single-plane packed/grey formats.
            _ => (maps[0].as_slice(), None),
        };

        // --- Convert the decoded rows into the staging buffer ---------------
        // Crop to the logical image: read only the first `final_h` rows and
        // `final_w` pixels per row from the MCU-rounded CAPTURE buffer.
        let native_stride = final_w * channels;
        self.staging.resize(native_stride * final_h, 0);
        let staging = &mut self.staging;
        let scratch = &mut self.row_scratch;
        for y in 0..final_h {
            let luma_row = &luma[y * luma_stride..];
            let chroma_row = chroma.map(|(c, cs)| &c[(y / 2) * cs..]);
            let dst_off = y * native_stride;
            format::convert_row(
                &kind,
                &plan,
                luma_row,
                chroma_row,
                final_w,
                scratch,
                &mut staging[dst_off..dst_off + native_stride],
            );
        }

        // --- Write staging into the destination tensor ----------------------
        let elem_size = std::mem::size_of::<T>();
        let mut map = dst.map().map_err(|e| DecodeErr::Fatal(e.into()))?;
        super::convert_rows_to_target::<T>(
            &self.staging,
            &mut map,
            final_w,
            final_h,
            channels,
            dst_stride,
            elem_size,
        );
        drop(map);

        // Keep the output mapping alive until decode completes.
        drop(omap);

        Ok(ImageInfo {
            width: final_w,
            height: final_h,
            format: output_fmt,
            row_stride: dst_stride,
        })
    }

    /// Best-effort reset: stop both queues and release their buffer pools.
    /// Doubles as the per-image cleanup (Phase 1 is non-persistent) and the
    /// recovery path after a hardware failure.
    fn teardown(&self) {
        let fd = self.device.fd();
        let _ = streamoff(fd, ioctl::V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE);
        let _ = streamoff(fd, ioctl::V4L2_BUF_TYPE_VIDEO_OUTPUT_MPLANE);
        let _ = reqbufs(fd, ioctl::V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE, 0);
        let _ = reqbufs(fd, ioctl::V4L2_BUF_TYPE_VIDEO_OUTPUT_MPLANE, 0);
    }
}

/// Accept only BT.601 full-range (JFIF) colorimetry — what the codec's colour
/// kernels assume. `DEFAULT` (0) fields defer to the colorspace, so a JPEG
/// colorspace is accepted; anything explicitly limited-range or non-601 is
/// rejected so we never emit mis-scaled colour.
fn colorimetry_ok(colorspace: u32, ycbcr_enc: u8, quantization: u8) -> bool {
    if colorspace == ioctl::V4L2_COLORSPACE_JPEG {
        return true;
    }
    let enc_ok = ycbcr_enc == 0 || ycbcr_enc == ioctl::V4L2_YCBCR_ENC_601;
    let quant_ok = quantization == ioctl::V4L2_QUANTIZATION_DEFAULT
        || quantization == ioctl::V4L2_QUANTIZATION_FULL_RANGE;
    enc_ok && quant_ok
}

// --- thin ioctl helpers -----------------------------------------------------

fn borrow(fd: RawFd) -> BorrowedFd<'static> {
    // SAFETY: callers hold the owning File alive for the duration of use.
    unsafe { BorrowedFd::borrow_raw(fd) }
}

fn reqbufs(fd: RawFd, buf_type: u32, count: u32) -> nix::Result<()> {
    let mut rb = ioctl::v4l2_requestbuffers {
        count,
        type_: buf_type,
        memory: ioctl::V4L2_MEMORY_MMAP,
        ..Default::default()
    };
    // SAFETY: valid requestbuffers struct.
    unsafe { ioctl::vidioc_reqbufs(fd, &mut rb) }.map(|_| ())
}

fn streamon(fd: RawFd, buf_type: u32) -> nix::Result<()> {
    let t: c_int = buf_type as c_int;
    // SAFETY: pointer to a valid c_int buffer type.
    unsafe { ioctl::vidioc_streamon(fd, &t) }.map(|_| ())
}

fn streamoff(fd: RawFd, buf_type: u32) -> nix::Result<()> {
    let t: c_int = buf_type as c_int;
    // SAFETY: pointer to a valid c_int buffer type.
    unsafe { ioctl::vidioc_streamoff(fd, &t) }.map(|_| ())
}

/// Drain pending events after `SOURCE_CHANGE`, bounded to avoid the
/// driver-re-returns-the-same-event hang.
fn drain_events(fd: RawFd) {
    for _ in 0..MAX_EVENTS {
        let mut ev = ioctl::v4l2_event::default();
        // SAFETY: valid event struct; best-effort, errors end the drain.
        if unsafe { ioctl::vidioc_dqevent(fd, &mut ev) }.is_err() {
            break;
        }
        if ev.pending == 0 {
            break;
        }
    }
}

/// Poll a single fd for `flags`, returning whether the event fired before the
/// timeout. Used best-effort for the source-change event and to wait for the
/// decode to complete.
fn poll_ready(fd: RawFd, flags: PollFlags, ms: i32) -> bool {
    let bfd = borrow(fd);
    let mut pfd = [PollFd::new(bfd, flags)];
    let timeout = PollTimeout::try_from(ms).unwrap_or(PollTimeout::ZERO);
    match poll(&mut pfd, timeout) {
        Ok(n) if n > 0 => pfd[0]
            .revents()
            .map(|r| r.intersects(flags))
            .unwrap_or(false),
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `V4L2_YCBCR_ENC_709` — a non-601 encoding we must reject.
    const ENC_709: u8 = 2;
    /// `V4L2_QUANTIZATION_LIM_RANGE` — limited range we must reject.
    const QUANT_LIMITED: u8 = 2;

    #[test]
    fn colorimetry_accepts_jpeg_and_full_range_601() {
        // JPEG colorspace is the shorthand and is always accepted.
        assert!(colorimetry_ok(
            ioctl::V4L2_COLORSPACE_JPEG,
            ENC_709,
            QUANT_LIMITED
        ));
        // Explicit 601 + full range.
        assert!(colorimetry_ok(
            ioctl::V4L2_COLORSPACE_SRGB,
            ioctl::V4L2_YCBCR_ENC_601,
            ioctl::V4L2_QUANTIZATION_FULL_RANGE
        ));
        // Driver "default" (0) fields defer to the colorspace and are accepted.
        assert!(colorimetry_ok(ioctl::V4L2_COLORSPACE_SRGB, 0, 0));
    }

    #[test]
    fn colorimetry_rejects_non_601_or_limited_range() {
        // 709 encoding → reject.
        assert!(!colorimetry_ok(
            ioctl::V4L2_COLORSPACE_SRGB,
            ENC_709,
            ioctl::V4L2_QUANTIZATION_FULL_RANGE
        ));
        // Limited-range quantization → reject.
        assert!(!colorimetry_ok(
            ioctl::V4L2_COLORSPACE_SRGB,
            ioctl::V4L2_YCBCR_ENC_601,
            QUANT_LIMITED
        ));
    }
}
