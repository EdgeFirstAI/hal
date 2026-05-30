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
//! Targets multi-planar M2M devices (the lead target, i.MX `mxc-jpeg`) with a
//! persistent streaming session: after the first decode at a given geometry
//! the OUTPUT/CAPTURE buffers stay mapped and streaming, so subsequent decodes
//! pay only the per-frame queue/dequeue cost. A geometry change rebuilds the
//! stream; a hardware failure resets it and (after repeated failures) demotes
//! the device to the CPU decoder. Single-planar devices and DMA-BUF zero-copy
//! fall back to the CPU decoder for now.

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
/// Consecutive hardware failures after which the device is demoted to the CPU
/// decoder for the rest of the session (circuit breaker).
const MAX_CONSECUTIVE_FAILURES: u32 = 8;

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
//
// `Ready` is much larger than the unit variants, but there is exactly one
// probe per decoder and it is `Ready` in the steady state, so boxing would add
// indirection on the hot path for no real saving.
#[allow(clippy::large_enum_variant)]
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

        let result = ctx.decode::<T>(
            data, dst, output_fmt, final_w, final_h, channels, dst_stride,
        );
        let out = match result {
            Ok(info) => {
                ctx.failures = 0;
                Ok(Some(info))
            }
            Err(DecodeErr::Reset(why)) | Err(DecodeErr::Unsupported(why)) => {
                // Reset the device and drop the persistent stream so the next
                // decode rebuilds cleanly, then fall back to the CPU.
                ctx.fail_reset();
                Err(V4l2Decode::Fallback(why))
            }
            Err(DecodeErr::Fatal(e)) => Err(V4l2Decode::Fatal(e)),
        };

        // Circuit breaker: a device that keeps failing is demoted so it stops
        // costing per-image latency; the CPU decoder takes over for good.
        if let V4l2Probe::Ready(ctx) = self {
            if ctx.failures >= MAX_CONSECUTIVE_FAILURES {
                log::warn!(
                    "v4l2 jpeg decoder disabled after {} consecutive failures",
                    ctx.failures
                );
                *self = V4l2Probe::Unavailable;
            }
        }
        out
    }
}

/// A persistent, capability-verified V4L2 decode context.
///
/// After the first decode at a given geometry the OUTPUT/CAPTURE buffers stay
/// allocated, mapped, and streaming, so subsequent same-geometry decodes skip
/// the ~0.9 ms per-image setup. A geometry change (or a larger JPEG than the
/// allocated OUTPUT buffer) rebuilds the stream; a hardware failure drops it.
pub(crate) struct V4l2Context {
    device: ProbedDevice,
    /// Packed-u8 staging buffer (native stride), reused across decodes. The
    /// hardware path converts CAPTURE rows into here, then the shared typed
    /// tail writes into the destination tensor.
    staging: Vec<u8>,
    /// Per-row de-interleave scratch, reused across decodes.
    row_scratch: RowScratch,
    /// The live streaming session (both queues `STREAMON` with mapped
    /// buffers), or `None` before the first decode / after a reset.
    stream: Option<Stream>,
    /// Consecutive hardware-failure count for the circuit breaker.
    failures: u32,
}

/// A live V4L2 streaming session: both queues set up, mapped, and `STREAMON`.
struct Stream {
    /// JPEG dimensions this OUTPUT format was configured for.
    jpeg_w: u32,
    jpeg_h: u32,
    /// Allocated OUTPUT plane size; a larger JPEG forces a rebuild.
    out_sizeimage: u32,
    /// Mapped OUTPUT (coded) buffer — the JPEG is copied in here each frame.
    out_map: Mmap,
    /// Driver-chosen CAPTURE (raw) format and its mapped planes.
    cap: Capture,
}

/// The CAPTURE side of a stream: driver-chosen format and mapped planes.
struct Capture {
    kind: CapKind,
    num_planes: usize,
    cap_h: usize,
    luma_stride: usize,
    /// CbCr-plane stride (NV12M two-plane); unused for single-plane formats.
    chroma_stride: usize,
    maps: Vec<Mmap>,
}

impl V4l2Context {
    fn new(device: ProbedDevice) -> Self {
        Self {
            device,
            staging: Vec::new(),
            row_scratch: RowScratch::new(),
            stream: None,
            failures: 0,
        }
    }

    /// Decode one image, reusing the persistent stream when the geometry
    /// matches and the JPEG fits the allocated OUTPUT buffer, otherwise
    /// rebuilding the stream first.
    #[allow(clippy::too_many_arguments)]
    fn decode<T: ImagePixel>(
        &mut self,
        data: &[u8],
        dst: &mut Tensor<T>,
        output_fmt: PixelFormat,
        final_w: usize,
        final_h: usize,
        channels: usize,
        dst_stride: usize,
    ) -> Result<ImageInfo, DecodeErr> {
        let needed = ((data.len() + 4095) & !4095) as u32;
        let reuse = matches!(
            &self.stream,
            Some(s) if s.jpeg_w == final_w as u32
                && s.jpeg_h == final_h as u32
                && s.out_sizeimage >= needed
        );

        if reuse {
            self.requeue_output(data)?;
        } else {
            self.rebuild_stream(data, final_w, final_h, needed)?;
        }
        self.collect::<T>(dst, output_fmt, final_w, final_h, channels, dst_stride)
    }

    /// Fast path: copy the JPEG into the already-mapped OUTPUT buffer and queue
    /// it. Both queues are already streaming from a prior decode.
    fn requeue_output(&mut self, data: &[u8]) -> Result<(), DecodeErr> {
        let fd = self.device.fd();
        let stream = self
            .stream
            .as_mut()
            .ok_or_else(|| DecodeErr::Reset("no stream".into()))?;
        stream.out_map.as_mut_slice()[..data.len()].copy_from_slice(data);
        qbuf_output(fd, data.len()).map_err(|e| DecodeErr::Reset(format!("QBUF OUTPUT: {e}")))
    }

    /// Tear down any existing stream, then set up the OUTPUT + CAPTURE queues
    /// for a new geometry, leaving the first JPEG queued and both queues
    /// streaming. Stores the resulting [`Stream`] for reuse.
    fn rebuild_stream(
        &mut self,
        data: &[u8],
        final_w: usize,
        final_h: usize,
        needed: u32,
    ) -> Result<(), DecodeErr> {
        self.drop_stream();
        let fd = self.device.fd();
        const OUT: u32 = ioctl::V4L2_BUF_TYPE_VIDEO_OUTPUT_MPLANE;
        const CAP: u32 = ioctl::V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE;

        // Subscribe to source-change events (harmless if already subscribed).
        let sub = ioctl::v4l2_event_subscription {
            type_: ioctl::V4L2_EVENT_SOURCE_CHANGE,
            ..Default::default()
        };
        // SAFETY: valid subscription struct.
        unsafe { ioctl::vidioc_subscribe_event(fd, &sub) }
            .map_err(|e| DecodeErr::Reset(format!("SUBSCRIBE_EVENT: {e}")))?;

        // OUTPUT: set JPEG format sized for this image, allocate, map, stream.
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
            p.plane_fmt[0].sizeimage = needed;
        }
        // SAFETY: valid v4l2_format; kernel reads/writes it.
        unsafe { ioctl::vidioc_s_fmt(fd, &mut ofmt) }
            .map_err(|e| DecodeErr::Reset(format!("S_FMT OUTPUT: {e}")))?;

        reqbufs(fd, OUT, 1).map_err(|e| DecodeErr::Reset(format!("REQBUFS OUTPUT: {e}")))?;
        let (olen, ooff) =
            querybuf(fd, OUT, 1).map_err(|e| DecodeErr::Reset(format!("QUERYBUF OUTPUT: {e}")))?;
        let mut out_map = Mmap::new(borrow(fd), olen, ooff)
            .map_err(|e| DecodeErr::Reset(format!("mmap OUTPUT: {e}")))?;

        streamon(fd, OUT).map_err(|e| DecodeErr::Reset(format!("STREAMON OUTPUT: {e}")))?;

        // Queue the first JPEG so the driver can parse the header.
        out_map.as_mut_slice()[..data.len()].copy_from_slice(data);
        qbuf_output(fd, data.len()).map_err(|e| DecodeErr::Reset(format!("QBUF OUTPUT: {e}")))?;

        // Wait for the driver to determine the CAPTURE format (best-effort).
        poll_ready(fd, PollFlags::POLLPRI, SOURCE_CHANGE_TIMEOUT_MS);
        drain_events(fd);

        // CAPTURE: query the driver-chosen format.
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
        if !colorimetry_ok(cap.colorspace, cap.ycbcr_enc, cap.quantization) {
            return Err(DecodeErr::Unsupported(format!(
                "capture colorimetry (cs={}, enc={}, quant={}) not BT.601 full-range",
                cap.colorspace, cap.ycbcr_enc, cap.quantization
            )));
        }

        let num_planes = cap.num_planes as usize;

        // CAPTURE: allocate, map every plane, stream.
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

        self.stream = Some(Stream {
            jpeg_w: final_w as u32,
            jpeg_h: final_h as u32,
            out_sizeimage: needed,
            out_map,
            cap: Capture {
                kind,
                num_planes,
                cap_h: cap.height as usize,
                luma_stride: cap.plane_fmt[0].bytesperline as usize,
                chroma_stride: cap.plane_fmt[1].bytesperline as usize,
                maps,
            },
        });
        Ok(())
    }

    /// Queue the CAPTURE buffer, wait for the decode, dequeue both buffers, and
    /// convert the result into `dst`. Shared by the build and reuse paths.
    #[allow(clippy::too_many_arguments)]
    fn collect<T: ImagePixel>(
        &mut self,
        dst: &mut Tensor<T>,
        output_fmt: PixelFormat,
        final_w: usize,
        final_h: usize,
        channels: usize,
        dst_stride: usize,
    ) -> Result<ImageInfo, DecodeErr> {
        let fd = self.device.fd();
        let plan = format::plan_output(output_fmt)
            .ok_or_else(|| DecodeErr::Unsupported(format!("output {output_fmt:?} unsupported")))?;

        let num_planes = self
            .stream
            .as_ref()
            .ok_or_else(|| DecodeErr::Reset("no stream".into()))?
            .cap
            .num_planes;

        // Queueing the CAPTURE buffer (both queues streaming) runs the job.
        qbuf_capture(fd, num_planes).map_err(|e| DecodeErr::Reset(format!("QBUF CAPTURE: {e}")))?;
        if !poll_ready(fd, PollFlags::POLLIN, DECODE_TIMEOUT_MS) {
            return Err(DecodeErr::Reset("CAPTURE decode timeout".into()));
        }
        dqbuf_capture(fd, num_planes)
            .map_err(|e| DecodeErr::Reset(format!("DQBUF CAPTURE: {e}")))?;
        // Recycle the consumed OUTPUT buffer so the next frame can re-queue it.
        dqbuf_output(fd).map_err(|e| DecodeErr::Reset(format!("DQBUF OUTPUT: {e}")))?;

        // Convert decoded planes → staging, cropping to the logical image.
        let native_stride = final_w * channels;
        self.staging.resize(native_stride * final_h, 0);

        let stream = self
            .stream
            .as_ref()
            .ok_or_else(|| DecodeErr::Reset("no stream".into()))?;
        let cap = &stream.cap;
        let luma_stride = cap.luma_stride;
        let (luma, chroma): (&[u8], Option<(&[u8], usize)>) = match (&cap.kind, cap.num_planes) {
            // NV12M: separate Y and CbCr buffers.
            (CapKind::Nv12, n) if n >= 2 => (
                cap.maps[0].as_slice(),
                Some((cap.maps[1].as_slice(), cap.chroma_stride)),
            ),
            // Single-buffer NV12: Y plane followed by CbCr in one allocation.
            (CapKind::Nv12, _) => {
                let m = cap.maps[0].as_slice();
                let y_size = cap.luma_stride * cap.cap_h;
                (&m[..y_size], Some((&m[y_size..], cap.luma_stride)))
            }
            // Single-plane packed/grey formats.
            _ => (cap.maps[0].as_slice(), None),
        };

        let staging = &mut self.staging;
        let scratch = &mut self.row_scratch;
        for y in 0..final_h {
            let luma_row = &luma[y * luma_stride..];
            let chroma_row = chroma.map(|(c, cs)| &c[(y / 2) * cs..]);
            let dst_off = y * native_stride;
            format::convert_row(
                &cap.kind,
                &plan,
                luma_row,
                chroma_row,
                final_w,
                scratch,
                &mut staging[dst_off..dst_off + native_stride],
            );
        }

        // Write staging into the destination tensor via the shared typed tail.
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

        Ok(ImageInfo {
            width: final_w,
            height: final_h,
            format: output_fmt,
            row_stride: dst_stride,
        })
    }

    /// Stop both queues and release their buffer pools, dropping the stream.
    /// Best-effort: errors are ignored (this is also the failure-recovery path).
    fn drop_stream(&mut self) {
        if self.stream.is_none() {
            return;
        }
        let fd = self.device.fd();
        let _ = streamoff(fd, ioctl::V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE);
        let _ = streamoff(fd, ioctl::V4L2_BUF_TYPE_VIDEO_OUTPUT_MPLANE);
        // Drop the mappings (munmap) before REQBUFS 0 frees the driver buffers.
        self.stream = None;
        let _ = reqbufs(fd, ioctl::V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE, 0);
        let _ = reqbufs(fd, ioctl::V4L2_BUF_TYPE_VIDEO_OUTPUT_MPLANE, 0);
    }

    /// Record a hardware failure and reset the device for the next attempt.
    fn fail_reset(&mut self) {
        self.drop_stream();
        self.failures = self.failures.saturating_add(1);
    }
}

impl Drop for V4l2Context {
    fn drop(&mut self) {
        // Release the streaming session and device buffers on decoder drop.
        self.drop_stream();
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

/// `VIDIOC_QUERYBUF` for index 0; returns plane 0's `(length, mem_offset)`.
fn querybuf(fd: RawFd, buf_type: u32, num_planes: usize) -> nix::Result<(usize, i64)> {
    let mut planes = [ioctl::v4l2_plane::default(); ioctl::VIDEO_MAX_PLANES];
    let mut b = ioctl::v4l2_buffer {
        type_: buf_type,
        memory: ioctl::V4L2_MEMORY_MMAP,
        index: 0,
        length: num_planes as u32,
        ..Default::default()
    };
    b.set_planes(planes.as_mut_ptr());
    // SAFETY: valid buffer + plane array for QUERYBUF.
    unsafe { ioctl::vidioc_querybuf(fd, &mut b) }?;
    Ok((planes[0].length as usize, planes[0].mem_offset() as i64))
}

/// `VIDIOC_QBUF` the OUTPUT (coded) buffer with `bytesused` JPEG bytes.
fn qbuf_output(fd: RawFd, bytesused: usize) -> nix::Result<()> {
    let mut planes = [ioctl::v4l2_plane::default(); ioctl::VIDEO_MAX_PLANES];
    planes[0].bytesused = bytesused as u32;
    let mut b = ioctl::v4l2_buffer {
        type_: ioctl::V4L2_BUF_TYPE_VIDEO_OUTPUT_MPLANE,
        memory: ioctl::V4L2_MEMORY_MMAP,
        index: 0,
        length: 1,
        ..Default::default()
    };
    b.set_planes(planes.as_mut_ptr());
    // SAFETY: valid buffer referencing the local plane array.
    unsafe { ioctl::vidioc_qbuf(fd, &mut b) }.map(|_| ())
}

/// `VIDIOC_DQBUF` the consumed OUTPUT buffer to recycle it.
fn dqbuf_output(fd: RawFd) -> nix::Result<()> {
    let mut planes = [ioctl::v4l2_plane::default(); ioctl::VIDEO_MAX_PLANES];
    let mut b = ioctl::v4l2_buffer {
        type_: ioctl::V4L2_BUF_TYPE_VIDEO_OUTPUT_MPLANE,
        memory: ioctl::V4L2_MEMORY_MMAP,
        length: 1,
        ..Default::default()
    };
    b.set_planes(planes.as_mut_ptr());
    // SAFETY: valid buffer + plane array for DQBUF.
    unsafe { ioctl::vidioc_dqbuf(fd, &mut b) }.map(|_| ())
}

/// `VIDIOC_QBUF` the CAPTURE (raw) buffer to run the decode.
fn qbuf_capture(fd: RawFd, num_planes: usize) -> nix::Result<()> {
    let mut planes = [ioctl::v4l2_plane::default(); ioctl::VIDEO_MAX_PLANES];
    let mut b = ioctl::v4l2_buffer {
        type_: ioctl::V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE,
        memory: ioctl::V4L2_MEMORY_MMAP,
        index: 0,
        length: num_planes as u32,
        ..Default::default()
    };
    b.set_planes(planes.as_mut_ptr());
    // SAFETY: valid buffer + plane array for QBUF.
    unsafe { ioctl::vidioc_qbuf(fd, &mut b) }.map(|_| ())
}

/// `VIDIOC_DQBUF` the decoded CAPTURE buffer.
fn dqbuf_capture(fd: RawFd, num_planes: usize) -> nix::Result<()> {
    let mut planes = [ioctl::v4l2_plane::default(); ioctl::VIDEO_MAX_PLANES];
    let mut b = ioctl::v4l2_buffer {
        type_: ioctl::V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE,
        memory: ioctl::V4L2_MEMORY_MMAP,
        length: num_planes as u32,
        ..Default::default()
    };
    b.set_planes(planes.as_mut_ptr());
    // SAFETY: valid buffer + plane array for DQBUF.
    unsafe { ioctl::vidioc_dqbuf(fd, &mut b) }.map(|_| ())
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
