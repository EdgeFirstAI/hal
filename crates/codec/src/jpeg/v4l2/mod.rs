// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! Optional V4L2 hardware JPEG-decoder backend (Linux only).
//!
//! When a Linux device exposes a JPEG decoder through the standard V4L2
//! mem2mem API, this backend offloads the decode to hardware, emitting the
//! codec's native `NV12`/`GREY` directly. It is probed lazily and entirely
//! capability-based, so it is portable across SoCs; anything it cannot drive
//! transparently falls back to the existing software decoder.
//!
//! Targets multi-planar M2M devices (the lead target, i.MX `mxc-jpeg`) with a
//! persistent streaming session. The OUTPUT (coded) buffer is allocated once
//! with headroom and survives geometry changes; the CAPTURE side is always
//! `V4L2_MEMORY_DMABUF`, where `REQBUFS` is allocation-free bookkeeping, so a
//! geometry change costs ~1 ms of ioctls instead of the ~110 ms CMA
//! allocate-and-map that `V4L2_MEMORY_MMAP` pays (there is no cheap MMAP
//! alternative: `S_FMT` returns `EBUSY` while MMAP buffers exist on a queue).
//! A hardware failure resets the session and (after repeated failures)
//! demotes the device to the CPU decoder.
//!
//! Two CAPTURE targets, tried in order:
//! - **Zero-copy ([`CaptureTarget::DstDma`]):** when the destination is a DMA
//!   tensor with MCU(16)-aligned dimensions and the driver accepts a
//!   single-plane contiguous CAPTURE at the tensor pitch, the tensor's dmabuf
//!   fd is imported as the CAPTURE buffer and the hardware decodes straight
//!   into it — no copy.
//! - **Scratch ([`CaptureTarget::Scratch`]):** the hardware decodes into a
//!   persistent codec-owned DMA scratch buffer (grown monotonically, never
//!   freed between frames) and the planes are copied — `NV12`/`GREY` cropped,
//!   `YUV3` (packed 4:4:4) deinterleaved to `NV24` — into the destination.
//!
//! The HAL is DMABUF-centric by design: hardware decode requires DMA buffer
//! allocation (dma_heap). Platforms without it use the CPU decoder, as do
//! single-planar M2M devices for now.

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
use edgefirst_tensor::{PixelFormat, Tensor, TensorMemory, TensorTrait};
use format::CapKind;
use std::os::fd::AsRawFd;

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
/// Minimum OUTPUT (coded) buffer allocation. Sized so virtually any JPEG fits
/// without a full stream rebuild; the buffer is allocated once per session.
const OUT_SIZE_FLOOR: u32 = 2 * 1024 * 1024;
/// Minimum CAPTURE scratch allocation (covers ~720p NV12 without regrowth).
const SCRATCH_SIZE_FLOOR: usize = 2 * 1024 * 1024;
/// Scratch tensor row width — the scratch is a GREY image used purely as a
/// linear byte buffer, so the row geometry only shapes the allocation size.
const SCRATCH_ROW_BYTES: usize = 4096;

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
        dst_stride: usize,
    ) -> Result<Option<ImageInfo>, V4l2Decode> {
        let Some(ctx) = self.ensure_probed() else {
            return Ok(None);
        };

        if ctx.device.api != ApiVariant::MultiPlanar {
            // Single-planar M2M decode is a later increment.
            return Ok(None);
        }

        let result = ctx.decode::<T>(data, dst, output_fmt, final_w, final_h, dst_stride);
        let out = match result {
            Ok(info) => {
                ctx.failures = 0;
                Ok(Some(info))
            }
            Err(DecodeErr::Reset(why)) => {
                // Hardware failure: reset + count toward the circuit breaker.
                ctx.fail_reset();
                Err(V4l2Decode::Fallback(why))
            }
            Err(DecodeErr::Unsupported(why)) => {
                // A config we don't drive (e.g. 4:4:4 YUV3 → NV12, or non-601
                // colorimetry). Reset the stream but do NOT count it as a
                // hardware failure — the CPU handles it; the device is fine.
                ctx.drop_stream();
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
/// After the first decode the OUTPUT buffer and the CAPTURE scratch stay
/// allocated, so a same-geometry decode pays only queue/dequeue and a
/// geometry change pays only a CAPTURE reconfigure (~1 ms) — never the
/// ~110 ms MMAP buffer reallocation. A hardware failure drops the session.
pub(crate) struct V4l2Context {
    device: ProbedDevice,
    /// The live streaming session (both queues `STREAMON` with buffers
    /// attached), or `None` before the first decode / after a reset.
    stream: Option<Stream>,
    /// Persistent CAPTURE scratch: a codec-owned DMA buffer the hardware
    /// decodes into when zero-copy is not possible. Grows monotonically on a
    /// new high-water mark and survives stream resets — only its capacity,
    /// dmabuf fd, and CPU mapping are used; its image geometry is irrelevant.
    scratch: Option<Tensor<u8>>,
    /// DMA scratch allocation failed once (no dma_heap): don't retry per
    /// frame. Hardware decode requires DMA buffers, so non-zero-copy images
    /// fall back to the CPU decoder for the rest of the session.
    scratch_failed: bool,
    /// Consecutive hardware-failure count for the circuit breaker.
    failures: u32,
}

/// A live V4L2 streaming session: both queues set up and `STREAMON`.
struct Stream {
    /// JPEG dimensions this session was configured for.
    jpeg_w: u32,
    jpeg_h: u32,
    /// The native output format the JPEG decodes to (`Nv12`/`Nv24`/`Grey`).
    /// Part of the reuse key: two same-sized JPEGs with different subsampling
    /// must not share a CAPTURE configuration (the driver would stall waiting
    /// for a format change).
    out_fmt: PixelFormat,
    /// Destination row pitch this session was configured for. Part of the
    /// reuse key for zero-copy: `S_FMT` baked this pitch into the CAPTURE
    /// format, so a same-geometry destination with a different stride must
    /// reconfigure. The scratch target copies out at the caller's stride
    /// every frame and ignores this.
    dst_stride: usize,
    /// Allocated OUTPUT plane size; a larger JPEG forces a full rebuild.
    out_sizeimage: u32,
    /// Mapped OUTPUT (coded) buffer — the JPEG is copied in here each frame.
    out_map: Mmap,
    /// Driver-chosen CAPTURE (raw) format and where the decode lands.
    cap: Capture,
}

/// The CAPTURE side of a stream: driver-chosen format and decode target.
/// The CAPTURE queue is always single-plane DMABUF — multi-plane formats
/// (NV12M) are renegotiated to their contiguous single-plane equivalents.
struct Capture {
    kind: CapKind,
    cap_h: usize,
    luma_stride: usize,
    target: CaptureTarget,
}

/// Where the hardware writes the decoded image.
enum CaptureTarget {
    /// Zero-copy: the destination tensor's dmabuf fd is imported per frame.
    DstDma,
    /// The context's persistent scratch dmabuf; planes are copied out after.
    Scratch,
}

impl V4l2Context {
    fn new(device: ProbedDevice) -> Self {
        Self {
            device,
            stream: None,
            scratch: None,
            scratch_failed: false,
            failures: 0,
        }
    }

    /// Decode one image through a three-tier path:
    /// - **reuse** — same geometry/format, OUTPUT fits: queue and go;
    /// - **reconfigure** — geometry changed but the session can keep its
    ///   buffers (DMABUF CAPTURE, persistent OUTPUT): ~1 ms of ioctls;
    /// - **rebuild** — first frame, OUTPUT overflow, or recovery: full setup.
    fn decode<T: ImagePixel>(
        &mut self,
        data: &[u8],
        dst: &mut Tensor<T>,
        output_fmt: PixelFormat,
        final_w: usize,
        final_h: usize,
        dst_stride: usize,
    ) -> Result<ImageInfo, DecodeErr> {
        let needed = ((data.len() + 4095) & !4095) as u32;
        let dma_capable = dst.memory() == TensorMemory::Dma;
        let dst_capacity = dst.capacity_bytes();

        // Hardware decode requires DMA buffers (the HAL is DMABUF-centric).
        // With no scratch and a non-DMA destination there is no possible
        // CAPTURE target — fail fast to the CPU decoder without touching the
        // device.
        if self.scratch_failed && !dma_capable {
            return Err(DecodeErr::Unsupported(
                "hardware decode requires DMA buffers (dma_heap unavailable)".into(),
            ));
        }

        // Reuse requires identical geometry AND native format (a same-sized
        // JPEG with different subsampling changes the CAPTURE format and would
        // stall the driver), plus a target the new destination supports:
        // zero-copy needs a DMA destination at the exact pitch the CAPTURE
        // format was configured with; Scratch copies out at the caller's
        // stride each frame, so any destination fits.
        let reuse = matches!(
            &self.stream,
            Some(s) if s.jpeg_w == final_w as u32
                && s.jpeg_h == final_h as u32
                && s.out_fmt == output_fmt
                && s.out_sizeimage >= needed
                && match s.cap.target {
                    CaptureTarget::DstDma => dma_capable && s.dst_stride == dst_stride,
                    CaptureTarget::Scratch => true,
                }
        );
        // Reconfigure keeps the OUTPUT buffer and re-imports the CAPTURE
        // buffers; only the OUTPUT capacity gates it.
        let reconfigure = !reuse && matches!(&self.stream, Some(s) if s.out_sizeimage >= needed);

        if reuse {
            self.requeue_output(data)?;
        } else if reconfigure {
            if let Err(e) = self.reconfigure_capture(
                data,
                output_fmt,
                final_w,
                final_h,
                dst_stride,
                dst_capacity,
                dma_capable,
            ) {
                match e {
                    // A reconfigure that cannot proceed (stale driver state,
                    // scratch exhausted) recovers with a full rebuild instead
                    // of burning a frame on the CPU.
                    DecodeErr::Reset(why) => {
                        log::debug!("v4l2: reconfigure failed ({why}); rebuilding stream");
                        self.rebuild_stream(
                            data,
                            output_fmt,
                            final_w,
                            final_h,
                            dst_stride,
                            dst_capacity,
                            dma_capable,
                            needed,
                        )?;
                    }
                    other => return Err(other),
                }
            }
        } else {
            self.rebuild_stream(
                data,
                output_fmt,
                final_w,
                final_h,
                dst_stride,
                dst_capacity,
                dma_capable,
                needed,
            )?;
        }
        self.collect::<T>(dst, output_fmt, final_w, final_h, dst_stride)
    }

    /// Fast path: copy the JPEG into the already-mapped OUTPUT buffer and queue
    /// it. Both queues are already streaming from a prior decode.
    fn requeue_output(&mut self, data: &[u8]) -> Result<(), DecodeErr> {
        let fd = self.device.fd();
        let stream = self
            .stream
            .as_mut()
            .ok_or_else(|| DecodeErr::Reset("no stream".into()))?;
        let staged = stage_jpeg(stream.out_map.as_mut_slice(), data);
        qbuf_output(fd, staged).map_err(|e| DecodeErr::Reset(format!("QBUF OUTPUT: {e}")))
    }

    /// Tear down any existing stream, then set up the OUTPUT + CAPTURE queues
    /// for a new geometry, leaving the first JPEG queued and both queues
    /// streaming. The OUTPUT buffer is allocated with headroom so subsequent
    /// geometry changes take [`Self::reconfigure_capture`] instead.
    #[allow(clippy::too_many_arguments)]
    fn rebuild_stream(
        &mut self,
        data: &[u8],
        output_fmt: PixelFormat,
        final_w: usize,
        final_h: usize,
        dst_stride: usize,
        dst_capacity: usize,
        dma_capable: bool,
        needed: u32,
    ) -> Result<(), DecodeErr> {
        let _span =
            tracing::trace_span!("codec.decode_jpeg.v4l2_rebuild", w = final_w, h = final_h)
                .entered();
        self.drop_stream();
        let fd = self.device.fd();
        const OUT: u32 = ioctl::V4L2_BUF_TYPE_VIDEO_OUTPUT_MPLANE;

        // Subscribe to source-change events (harmless if already subscribed).
        let sub = ioctl::v4l2_event_subscription {
            type_: ioctl::V4L2_EVENT_SOURCE_CHANGE,
            ..Default::default()
        };
        // SAFETY: valid subscription struct.
        unsafe { ioctl::vidioc_subscribe_event(fd, &sub) }
            .map_err(|e| DecodeErr::Reset(format!("SUBSCRIBE_EVENT: {e}")))?;

        // OUTPUT: JPEG format with headroom (so a later, larger image avoids a
        // full rebuild), allocate, map, stream. The allocation is persistent —
        // it survives every geometry change until OUTPUT overflow or reset.
        let out_request = needed.max(OUT_SIZE_FLOOR);
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
            p.plane_fmt[0].sizeimage = out_request;
        }
        // SAFETY: valid v4l2_format; kernel reads/writes it.
        unsafe { ioctl::vidioc_s_fmt(fd, &mut ofmt) }
            .map_err(|e| DecodeErr::Reset(format!("S_FMT OUTPUT: {e}")))?;

        reqbufs(fd, OUT, 1, ioctl::V4L2_MEMORY_MMAP)
            .map_err(|e| DecodeErr::Reset(format!("REQBUFS OUTPUT: {e}")))?;
        let (olen, ooff) =
            querybuf(fd, OUT, 1).map_err(|e| DecodeErr::Reset(format!("QUERYBUF OUTPUT: {e}")))?;
        let mut out_map = Mmap::new(borrow(fd), olen, ooff)
            .map_err(|e| DecodeErr::Reset(format!("mmap OUTPUT: {e}")))?;

        streamon(fd, OUT).map_err(|e| DecodeErr::Reset(format!("STREAMON OUTPUT: {e}")))?;

        // Queue the first JPEG so the driver can parse the header.
        let staged = stage_jpeg(out_map.as_mut_slice(), data);
        qbuf_output(fd, staged).map_err(|e| DecodeErr::Reset(format!("QBUF OUTPUT: {e}")))?;

        // Wait for the driver to determine the CAPTURE format (best-effort).
        poll_ready(fd, PollFlags::POLLPRI, SOURCE_CHANGE_TIMEOUT_MS);
        drain_events(fd);

        let cap = self.configure_capture(
            output_fmt,
            final_w,
            final_h,
            dst_stride,
            dst_capacity,
            dma_capable,
        )?;

        self.stream = Some(Stream {
            jpeg_w: final_w as u32,
            jpeg_h: final_h as u32,
            out_fmt: output_fmt,
            dst_stride,
            out_sizeimage: olen.min(u32::MAX as usize) as u32,
            out_map,
            cap,
        });
        Ok(())
    }

    /// Geometry-change hot path: keep the persistent OUTPUT buffer and the
    /// DMABUF CAPTURE memory mode, paying only ioctls (~1 ms measured on
    /// i.MX95 `mxc-jpeg`) instead of the ~110 ms MMAP buffer reallocation.
    ///
    /// Any `Reset` error here is recovered by the caller with a full rebuild —
    /// the half-torn queue state this can leave behind is exactly what
    /// [`Self::drop_stream`] (the first step of a rebuild) cleans up.
    #[allow(clippy::too_many_arguments)]
    fn reconfigure_capture(
        &mut self,
        data: &[u8],
        output_fmt: PixelFormat,
        final_w: usize,
        final_h: usize,
        dst_stride: usize,
        dst_capacity: usize,
        dma_capable: bool,
    ) -> Result<(), DecodeErr> {
        let _span = tracing::trace_span!(
            "codec.decode_jpeg.v4l2_reconfigure",
            w = final_w,
            h = final_h
        )
        .entered();
        let fd = self.device.fd();
        const CAP: u32 = ioctl::V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE;

        // Stop and free only the CAPTURE queue — DMABUF buffers are ours, so
        // this releases no memory, just vb2 bookkeeping. The OUTPUT queue
        // keeps streaming with its buffer allocated.
        if self.stream.is_none() {
            return Err(DecodeErr::Reset("no stream to reconfigure".into()));
        }
        streamoff(fd, CAP)
            .map_err(|e| DecodeErr::Reset(format!("STREAMOFF CAPTURE (reconf): {e}")))?;
        reqbufs(fd, CAP, 0, ioctl::V4L2_MEMORY_DMABUF)
            .map_err(|e| DecodeErr::Reset(format!("REQBUFS CAPTURE 0 (reconf): {e}")))?;

        // Queue the new JPEG; the driver parses the header and retargets the
        // CAPTURE format (raising SOURCE_CHANGE) while OUTPUT keeps streaming.
        let staged = {
            let stream = self
                .stream
                .as_mut()
                .ok_or_else(|| DecodeErr::Reset("no stream to reconfigure".into()))?;
            stage_jpeg(stream.out_map.as_mut_slice(), data)
        };
        qbuf_output(fd, staged)
            .map_err(|e| DecodeErr::Reset(format!("QBUF OUTPUT (reconf): {e}")))?;
        poll_ready(fd, PollFlags::POLLPRI, SOURCE_CHANGE_TIMEOUT_MS);
        drain_events(fd);

        let cap = self.configure_capture(
            output_fmt,
            final_w,
            final_h,
            dst_stride,
            dst_capacity,
            dma_capable,
        )?;

        let stream = self
            .stream
            .as_mut()
            .ok_or_else(|| DecodeErr::Reset("no stream to reconfigure".into()))?;
        stream.jpeg_w = final_w as u32;
        stream.jpeg_h = final_h as u32;
        stream.out_fmt = output_fmt;
        stream.dst_stride = dst_stride;
        stream.cap = cap;
        Ok(())
    }

    /// Shared CAPTURE-side configuration: read the driver-parsed format,
    /// negotiate, pick the decode target (zero-copy, else scratch), request
    /// the DMABUF buffer, and start the CAPTURE queue. The caller must have
    /// the new JPEG queued on a streaming OUTPUT and the CAPTURE queue
    /// stopped with no buffers.
    fn configure_capture(
        &mut self,
        output_fmt: PixelFormat,
        final_w: usize,
        final_h: usize,
        dst_stride: usize,
        dst_capacity: usize,
        dma_capable: bool,
    ) -> Result<Capture, DecodeErr> {
        let fd = self.device.fd();
        const CAP: u32 = ioctl::V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE;

        // CAPTURE: query the driver-chosen format for the queued JPEG.
        let mut cfmt = ioctl::v4l2_format {
            type_: CAP,
            ..Default::default()
        };
        // SAFETY: valid v4l2_format for G_FMT.
        unsafe { ioctl::vidioc_g_fmt(fd, &mut cfmt) }
            .map_err(|e| DecodeErr::Reset(format!("G_FMT CAPTURE: {e}")))?;
        // SAFETY: type_ selects the multi-planar variant.
        let mut cap = *unsafe { cfmt.pix_mp() };

        // The driver reports MCU-padded dimensions, so they must cover the
        // logical image. Smaller means the SOURCE_CHANGE never landed and the
        // format is stale from a previous image — recover with a rebuild.
        if (cap.width as usize) < final_w || (cap.height as usize) < final_h {
            return Err(DecodeErr::Reset(format!(
                "stale CAPTURE geometry {}x{} for {}x{} image",
                cap.width, cap.height, final_w, final_h
            )));
        }

        let mut kind = format::classify(cap.pixelformat).ok_or_else(|| {
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

        // The mxc-jpeg driver (and equivalent SoC decoders) select their native
        // CAPTURE format from the JPEG's chroma subsampling: a 4:4:4 JPEG
        // yields YUV3 (packed 4:4:4). When the caller explicitly requested
        // NV12 — and only then — negotiate it via S_FMT so the hardware
        // downconverts 4:4:4→4:2:0. An Nv24 request must NOT attempt this:
        // a driver that accepted NV12 would silently discard chroma
        // resolution and break the native-format contract with the CPU
        // decoder (4:4:4 → Nv24). S_FMT must happen before REQBUFS CAPTURE;
        // this is the only safe window (after G_FMT, before REQBUFS).
        if matches!(kind, CapKind::Yuv444Packed) && matches!(output_fmt, PixelFormat::Nv12) {
            let mut nfmt = ioctl::v4l2_format {
                type_: CAP,
                ..Default::default()
            };
            // SAFETY: mplane variant.
            {
                let p = unsafe { nfmt.pix_mp() };
                p.width = cap.width;
                p.height = cap.height;
                p.pixelformat = ioctl::V4L2_PIX_FMT_NV12;
                p.field = ioctl::V4L2_FIELD_NONE;
                p.colorspace = cap.colorspace;
                p.ycbcr_enc = cap.ycbcr_enc;
                p.quantization = cap.quantization;
                p.xfer_func = cap.xfer_func;
                p.num_planes = 1;
            }
            // SAFETY: valid format; best-effort — if refused, cap/kind stay YUV3
            // and the scratch path deinterleaves to NV24 instead.
            let _ = unsafe { ioctl::vidioc_s_fmt(fd, &mut nfmt) };
            let mut gfmt = ioctl::v4l2_format {
                type_: CAP,
                ..Default::default()
            };
            // SAFETY: valid format for G_FMT.
            if unsafe { ioctl::vidioc_g_fmt(fd, &mut gfmt) }.is_ok() {
                // SAFETY: mplane variant.
                let got = *unsafe { gfmt.pix_mp() };
                if let Some(new_kind) = format::classify(got.pixelformat) {
                    if matches!(new_kind, CapKind::Nv12) {
                        log::debug!(
                            "v4l2: negotiated 4:4:4 CAPTURE from YUV3 to NV12 (hardware 4:4:4→4:2:0)"
                        );
                        cap = got;
                        kind = new_kind;
                    } else {
                        log::debug!(
                            "v4l2: driver kept {} for 4:4:4 JPEG — will deinterleave to NV24",
                            ioctl::fourcc_str(got.pixelformat)
                        );
                    }
                }
            }
        }

        // Save the driver's natural CAPTURE format so a failed zero-copy probe
        // can restore it for the copy paths.
        let cap0 = cap;

        // Zero-copy eligibility: DMA destination, NV12/GREY output, and
        // MCU(16)-aligned dims so the CAPTURE geometry equals the logical image
        // (the decoded data lands exactly at the tensor's layout).
        let want_zc = dma_capable
            && matches!(output_fmt, PixelFormat::Nv12 | PixelFormat::Grey)
            && matches!(kind, CapKind::Nv12 | CapKind::Grey)
            && final_w.is_multiple_of(16)
            && final_h.is_multiple_of(16);

        let mut dst_dma = false;
        if want_zc {
            // MCU(16)-aligned logical dims mean the driver's MCU-padded
            // CAPTURE geometry equals them — the invariant the zero-copy
            // sizeimage math and the tensor layout both rely on.
            debug_assert_eq!(
                cap0.height as usize, final_h,
                "zero-copy requires the MCU-padded CAPTURE height to equal the image height"
            );
            debug_assert_eq!(
                cap0.width as usize, final_w,
                "zero-copy requires the MCU-padded CAPTURE width to equal the image width"
            );
            // Request a SINGLE-PLANE contiguous CAPTURE at the tensor pitch:
            // the whole NV12 (Y+CbCr) / GREY image becomes one fd at offset 0.
            // The driver's default NV12M is two *non-contiguous* planes, which
            // do not compose into a single tensor buffer via data_offset.
            let (target_fourcc, total_h) = match output_fmt {
                PixelFormat::Nv12 => (ioctl::V4L2_PIX_FMT_NV12, cap0.height as usize * 3 / 2),
                _ => (ioctl::V4L2_PIX_FMT_GREY, cap0.height as usize),
            };
            let mut sfmt = ioctl::v4l2_format {
                type_: CAP,
                ..Default::default()
            };
            // SAFETY: mplane variant.
            {
                let p = unsafe { sfmt.pix_mp() };
                p.width = cap0.width;
                p.height = cap0.height;
                p.pixelformat = target_fourcc;
                p.field = ioctl::V4L2_FIELD_NONE;
                p.colorspace = cap0.colorspace;
                p.ycbcr_enc = cap0.ycbcr_enc;
                p.quantization = cap0.quantization;
                p.xfer_func = cap0.xfer_func;
                p.num_planes = 1;
                p.plane_fmt[0].bytesperline = dst_stride as u32;
                p.plane_fmt[0].sizeimage = (dst_stride * total_h) as u32;
            }
            // SAFETY: valid format. Best-effort; read back what the driver kept.
            let _ = unsafe { ioctl::vidioc_s_fmt(fd, &mut sfmt) };
            let mut gfmt = ioctl::v4l2_format {
                type_: CAP,
                ..Default::default()
            };
            // SAFETY: valid format for G_FMT.
            unsafe { ioctl::vidioc_g_fmt(fd, &mut gfmt) }
                .map_err(|e| DecodeErr::Reset(format!("G_FMT CAPTURE (zc): {e}")))?;
            // SAFETY: mplane variant.
            let cap_zc = *unsafe { gfmt.pix_mp() };
            let ok = cap_zc.pixelformat == target_fourcc
                && cap_zc.num_planes == 1
                && cap_zc.width as usize == final_w
                && cap_zc.height as usize == final_h
                && cap_zc.plane_fmt[0].bytesperline as usize == dst_stride
                && dst_capacity >= cap_zc.plane_fmt[0].sizeimage as usize;
            if ok {
                cap = cap_zc;
                dst_dma = true;
            } else {
                // Driver refused single-plane contiguous output — restore its
                // natural format for the copy paths.
                let mut rfmt = ioctl::v4l2_format {
                    type_: CAP,
                    ..Default::default()
                };
                // SAFETY: mplane variant.
                {
                    let p = unsafe { rfmt.pix_mp() };
                    *p = cap0;
                }
                // SAFETY: valid format; best-effort restore.
                let _ = unsafe { ioctl::vidioc_s_fmt(fd, &mut rfmt) };
                cap = cap0;
            }
        }

        if dst_dma {
            // Zero-copy: no driver buffers; the tensor fd is imported per
            // frame in `collect`. REQBUFS(DMABUF) is bookkeeping only.
            reqbufs(fd, CAP, 1, ioctl::V4L2_MEMORY_DMABUF)
                .map_err(|e| DecodeErr::Reset(format!("REQBUFS CAPTURE (dmabuf): {e}")))?;
            streamon(fd, CAP).map_err(|e| DecodeErr::Reset(format!("STREAMON CAPTURE: {e}")))?;
            return Ok(Capture {
                kind,
                cap_h: cap.height as usize,
                luma_stride: cap.plane_fmt[0].bytesperline as usize,
                target: CaptureTarget::DstDma,
            });
        }

        // Scratch: the hardware decodes into the persistent codec-owned DMA
        // buffer and the planes are copied out. Needs a single-plane CAPTURE
        // so the import never depends on multi-plane data_offset semantics;
        // YUV3/GREY are naturally single-plane, NV12M is renegotiated.
        let mut single = cap;
        if cap.num_planes != 1 {
            let mut sfmt = ioctl::v4l2_format {
                type_: CAP,
                ..Default::default()
            };
            // SAFETY: mplane variant.
            {
                let p = unsafe { sfmt.pix_mp() };
                p.width = cap.width;
                p.height = cap.height;
                p.pixelformat = ioctl::V4L2_PIX_FMT_NV12;
                p.field = ioctl::V4L2_FIELD_NONE;
                p.colorspace = cap.colorspace;
                p.ycbcr_enc = cap.ycbcr_enc;
                p.quantization = cap.quantization;
                p.xfer_func = cap.xfer_func;
                p.num_planes = 1;
            }
            // SAFETY: valid format; best-effort — read back the result.
            let _ = unsafe { ioctl::vidioc_s_fmt(fd, &mut sfmt) };
            let mut gfmt = ioctl::v4l2_format {
                type_: CAP,
                ..Default::default()
            };
            // SAFETY: valid format for G_FMT.
            unsafe { ioctl::vidioc_g_fmt(fd, &mut gfmt) }
                .map_err(|e| DecodeErr::Reset(format!("G_FMT CAPTURE (scratch): {e}")))?;
            // SAFETY: mplane variant.
            single = *unsafe { gfmt.pix_mp() };
            if single.num_planes != 1 {
                return Err(DecodeErr::Unsupported(format!(
                    "driver insists on a {}-plane CAPTURE; single-plane DMABUF required",
                    single.num_planes
                )));
            }
        }
        let kind = format::classify(single.pixelformat).ok_or_else(|| {
            DecodeErr::Unsupported(format!(
                "capture format {} unsupported",
                ioctl::fourcc_str(single.pixelformat)
            ))
        })?;
        let sizeimage = single.plane_fmt[0].sizeimage as usize;
        if !self.ensure_scratch(sizeimage) {
            return Err(DecodeErr::Unsupported(
                "hardware decode requires DMA buffers (dma_heap unavailable)".into(),
            ));
        }
        reqbufs(fd, CAP, 1, ioctl::V4L2_MEMORY_DMABUF)
            .map_err(|e| DecodeErr::Reset(format!("REQBUFS CAPTURE (scratch): {e}")))?;
        streamon(fd, CAP).map_err(|e| DecodeErr::Reset(format!("STREAMON CAPTURE: {e}")))?;
        Ok(Capture {
            kind,
            cap_h: single.height as usize,
            luma_stride: single.plane_fmt[0].bytesperline as usize,
            target: CaptureTarget::Scratch,
        })
    }

    /// Ensure the persistent scratch dmabuf holds at least `sizeimage` bytes,
    /// growing geometrically on a new high-water mark. Returns whether the
    /// scratch is usable; a failed DMA allocation latches [`Self::scratch_failed`]
    /// so the session fails fast to the CPU decoder instead of retrying the
    /// allocation per frame.
    fn ensure_scratch(&mut self, sizeimage: usize) -> bool {
        if let Some(s) = &self.scratch {
            if s.capacity_bytes() >= sizeimage {
                return true;
            }
        }
        // Grow ×3/2 over the request (and never below the floor) so a slowly
        // increasing image size doesn't reallocate every frame.
        let want = (sizeimage + sizeimage / 2).max(SCRATCH_SIZE_FLOOR);
        let rows = want.div_ceil(SCRATCH_ROW_BYTES);
        // Free the outgrown scratch before allocating its replacement: CMA is
        // a scarce contiguous pool and holding both peaks at 2.5× the need.
        self.scratch = None;
        match Tensor::<u8>::image(
            SCRATCH_ROW_BYTES,
            rows,
            PixelFormat::Grey,
            Some(TensorMemory::Dma),
            // The decoder hardware writes the capture planes; the CPU only
            // reads them during copy-out (`map_read` syncs the read
            // direction).
            edgefirst_tensor::CpuAccess::Read,
        ) {
            Ok(t) => {
                log::debug!(
                    "v4l2: capture scratch dmabuf allocated ({} bytes)",
                    t.capacity_bytes()
                );
                self.scratch = Some(t);
                true
            }
            Err(e) => {
                log::debug!("v4l2: no DMA scratch ({e}); hardware decode unavailable");
                self.scratch_failed = true;
                false
            }
        }
    }

    /// Queue the CAPTURE buffer for the session's target, wait for the decode,
    /// dequeue both buffers, and (for the scratch target) write the decoded
    /// pixels into `dst`. Shared by all three decode paths.
    fn collect<T: ImagePixel>(
        &mut self,
        dst: &mut Tensor<T>,
        output_fmt: PixelFormat,
        final_w: usize,
        final_h: usize,
        dst_stride: usize,
    ) -> Result<ImageInfo, DecodeErr> {
        let fd = self.device.fd();
        let is_dst_dma = {
            let s = self
                .stream
                .as_ref()
                .ok_or_else(|| DecodeErr::Reset("no stream".into()))?;
            matches!(s.cap.target, CaptureTarget::DstDma)
        };
        let _span = tracing::trace_span!(
            "codec.decode_jpeg.v4l2_collect",
            target = if is_dst_dma { "dst_dma" } else { "scratch" },
        )
        .entered();

        if is_dst_dma {
            // Zero-copy: import the tensor's dmabuf fd as the CAPTURE buffer
            // so the hardware decodes straight into it — no copy. The whole
            // image (Y, then CbCr for NV12) is one contiguous plane at offset
            // 0; layout matches because dims are MCU-aligned and the CAPTURE
            // stride was forced to the tensor pitch.
            let dmabuf_fd = dst
                .dmabuf()
                .map_err(|e| DecodeErr::Fatal(e.into()))?
                .as_raw_fd();
            let capacity = dst.capacity_bytes();
            qbuf_capture_dmabuf(fd, dmabuf_fd, capacity)
                .map_err(|e| DecodeErr::Reset(format!("QBUF CAPTURE (dmabuf): {e}")))?;
            if !poll_ready(fd, PollFlags::POLLIN, DECODE_TIMEOUT_MS) {
                return Err(DecodeErr::Reset("CAPTURE decode timeout".into()));
            }
            dqbuf_capture(fd)
                .map_err(|e| DecodeErr::Reset(format!("DQBUF CAPTURE (dmabuf): {e}")))?;
            dqbuf_output(fd).map_err(|e| DecodeErr::Reset(format!("DQBUF OUTPUT: {e}")))?;
            // Decoded pixels are in the tensor's DMA buffer; the consumer's
            // `Tensor::map()` issues the cache sync on read.
            // DstDma only fires when output_fmt is Nv12|Grey (want_zc check),
            // so actual == requested and no dst reconfiguration is needed.
            return Ok(ImageInfo {
                width: final_w,
                height: final_h,
                format: output_fmt,
                row_stride: dst_stride,
                rotation_degrees: 0,
                flip_horizontal: false,
            });
        }

        // Persistent-scratch: the hardware decodes into the codec-owned DMA
        // buffer (single plane; QBUF length may exceed the format's
        // sizeimage — the import is by capacity), then planes copy out.
        let (scratch_fd, capacity) = {
            let t = self
                .scratch
                .as_ref()
                .ok_or_else(|| DecodeErr::Reset("capture scratch missing".into()))?;
            (
                t.dmabuf()
                    .map_err(|e| DecodeErr::Fatal(e.into()))?
                    .as_raw_fd(),
                t.capacity_bytes(),
            )
        };
        qbuf_capture_dmabuf(fd, scratch_fd, capacity)
            .map_err(|e| DecodeErr::Reset(format!("QBUF CAPTURE (scratch): {e}")))?;
        if !poll_ready(fd, PollFlags::POLLIN, DECODE_TIMEOUT_MS) {
            return Err(DecodeErr::Reset("CAPTURE decode timeout".into()));
        }
        dqbuf_capture(fd).map_err(|e| DecodeErr::Reset(format!("DQBUF CAPTURE (scratch): {e}")))?;
        dqbuf_output(fd).map_err(|e| DecodeErr::Reset(format!("DQBUF OUTPUT: {e}")))?;

        let stream = self
            .stream
            .as_ref()
            .ok_or_else(|| DecodeErr::Reset("no stream".into()))?;
        // `map()` issues the DMA_BUF_IOCTL_SYNC so CPU reads are coherent.
        let smap = self
            .scratch
            .as_mut()
            .ok_or_else(|| DecodeErr::Reset("capture scratch missing".into()))?
            .map_read()
            .map_err(|e| DecodeErr::Fatal(e.into()))?;
        write_planes(
            &stream.cap,
            &smap,
            dst,
            output_fmt,
            final_w,
            final_h,
            dst_stride,
        )
    }

    /// Stop both queues and release their buffer pools, dropping the stream.
    /// The CAPTURE scratch is *kept* — it is plain memory, not driver state,
    /// and survives resets so recovery never re-pays the DMA allocation.
    /// Best-effort: errors are ignored (this is also the failure-recovery path).
    fn drop_stream(&mut self) {
        let Some(stream) = self.stream.take() else {
            return;
        };
        let fd = self.device.fd();
        let _ = streamoff(fd, ioctl::V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE);
        let _ = streamoff(fd, ioctl::V4L2_BUF_TYPE_VIDEO_OUTPUT_MPLANE);
        // Drop the OUTPUT mapping (munmap) before REQBUFS 0 frees its buffer.
        drop(stream);
        let _ = reqbufs(
            fd,
            ioctl::V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE,
            0,
            ioctl::V4L2_MEMORY_DMABUF,
        );
        let _ = reqbufs(
            fd,
            ioctl::V4L2_BUF_TYPE_VIDEO_OUTPUT_MPLANE,
            0,
            ioctl::V4L2_MEMORY_MMAP,
        );
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

/// Write the decoded CAPTURE planes into `dst`, cropping to the logical image.
/// Handles single-plane NV12, GREY, and YUV3 (4:4:4 packed → NV24
/// deinterleave). `src` is the scratch buffer the hardware decoded into.
fn write_planes<T: ImagePixel>(
    cap: &Capture,
    src: &[u8],
    dst: &mut Tensor<T>,
    output_fmt: PixelFormat,
    final_w: usize,
    final_h: usize,
    dst_stride: usize,
) -> Result<ImageInfo, DecodeErr> {
    let _span = tracing::trace_span!(
        "codec.decode_jpeg.v4l2_copy_out",
        kind = match cap.kind {
            CapKind::Yuv444Packed => "yuv24_to_nv24",
            CapKind::Nv12 => "nv12",
            CapKind::Grey => "grey",
        },
    )
    .entered();
    let y_stride = cap.luma_stride;

    // YUV3 → NV24: the mxc-jpeg driver outputs 4:4:4 JPEGs as packed
    // [Y,Cb,Cr] triples (V4L2_PIX_FMT_YUV24). Deinterleave into the
    // semi-planar NV24 layout expected by the tensor and downstream convert.
    // NV24 UV addressing: Cb at [final_h*dst + y*2*dst + x*2],
    // Cr at [... + x*2 + 1] — matches the MCU writer in mcu.rs exactly.
    if matches!(cap.kind, CapKind::Yuv444Packed) {
        if !matches!(output_fmt, PixelFormat::Nv24) {
            return Err(DecodeErr::Unsupported(format!(
                "4:4:4 YUV3 capture but output format is {output_fmt:?} (expected Nv24)"
            )));
        }
        let mut map = dst.map_write().map_err(|e| DecodeErr::Fatal(e.into()))?;
        let dst_bytes: &mut [T] = &mut map;
        // SAFETY: NV24 is u8; JPEG entry guarantees T == u8.
        let d: &mut [u8] = unsafe {
            std::slice::from_raw_parts_mut(dst_bytes.as_mut_ptr() as *mut u8, dst_bytes.len())
        };
        deinterleave_yuv24_to_nv24(src, y_stride, d, dst_stride, final_w, final_h);
        drop(map);
        return Ok(ImageInfo {
            width: final_w,
            height: final_h,
            format: PixelFormat::Nv24,
            row_stride: dst_stride,
            rotation_degrees: 0,
            flip_horizontal: false,
        });
    }

    // Resolve the decoded planes from the single contiguous CAPTURE plane:
    // greyscale → Y only; NV12 → Y, then interleaved CbCr after the
    // (MCU-padded) luma plane.
    let (y_plane, chroma): (&[u8], Option<(&[u8], usize)>) = match &cap.kind {
        CapKind::Nv12 => {
            let ys = cap.luma_stride * cap.cap_h;
            (&src[..ys], Some((&src[ys..], cap.luma_stride)))
        }
        CapKind::Grey => (src, None),
        CapKind::Yuv444Packed => unreachable!("handled above"),
    };

    // Derive the format actually written from what hardware produced and
    // reconfigure dst when it differs (e.g. an Nv24 request where the 4:4:4
    // JPEG was hardware-downconverted is rejected above, but a caller may
    // request Nv12 for a 4:4:4 JPEG, where negotiation yields kind=Nv12).
    let actual_fmt = match &cap.kind {
        CapKind::Grey => PixelFormat::Grey,
        CapKind::Nv12 => PixelFormat::Nv12,
        CapKind::Yuv444Packed => unreachable!("handled above"),
    };
    if actual_fmt != output_fmt {
        // The tensor was configured for output_fmt (e.g. Nv24) by the
        // caller; reconfigure it to match what we're about to write.
        // The underlying allocation is always large enough because Nv24 is
        // the largest per-pixel format the caller pre-allocates for.
        dst.configure_image(final_w, final_h, actual_fmt)
            .map_err(|e| DecodeErr::Fatal(e.into()))?;
    }

    let mut map = dst.map_write().map_err(|e| DecodeErr::Fatal(e.into()))?;
    let dst_bytes: &mut [T] = &mut map;
    // SAFETY: native NV12/GREY are u8; the JPEG entry guarantees T == u8.
    let d: &mut [u8] = unsafe {
        std::slice::from_raw_parts_mut(dst_bytes.as_mut_ptr() as *mut u8, dst_bytes.len())
    };

    // Crop to the logical image (the CAPTURE buffer is MCU-rounded up).
    match actual_fmt {
        PixelFormat::Grey => {
            for y in 0..final_h {
                let s = y * y_stride;
                let o = y * dst_stride;
                d[o..o + final_w].copy_from_slice(&y_plane[s..s + final_w]);
            }
        }
        PixelFormat::Nv12 => {
            let Some((cbcr, c_stride)) = chroma else {
                return Err(DecodeErr::Unsupported(
                    "NV12 output requires a chroma plane from the decoder".into(),
                ));
            };
            // Y plane.
            for y in 0..final_h {
                let s = y * y_stride;
                let o = y * dst_stride;
                d[o..o + final_w].copy_from_slice(&y_plane[s..s + final_w]);
            }
            // Interleaved CbCr plane. An NV12 chroma row holds
            // `ceil(final_w/2)` (Cb,Cr) pairs == `even(final_w)` bytes;
            // copying only `final_w` drops the final Cr on ODD widths.
            // `ceil(final_h/2)` chroma rows (4:2:0 rounds the row count up).
            // Both `c_stride` (MCU-rounded CAPTURE pitch) and `dst_stride`
            // (64-aligned tensor pitch) are >= even(final_w), so the wider
            // copy stays in-bounds.
            let uv_base = final_h * dst_stride;
            copy_nv12_chroma(
                cbcr,
                c_stride,
                &mut d[uv_base..],
                dst_stride,
                final_w,
                final_h,
            );
        }
        other => {
            return Err(DecodeErr::Unsupported(format!(
                "output {other:?} unsupported"
            )));
        }
    }
    drop(map);

    Ok(ImageInfo {
        width: final_w,
        height: final_h,
        format: actual_fmt,
        row_stride: dst_stride,
        rotation_degrees: 0,
        flip_horizontal: false,
    })
}

/// Copy `data` into the OUTPUT buffer with metadata segments stripped,
/// returning the staged byte count.
///
/// The i.MX `mxc-jpeg` bitstream parser does not skip APPn payloads by their
/// length field: an APP13 (Photoshop IRB) carrying an embedded thumbnail JPEG
/// (nested SOI/SOF/SOS markers) wedges the hardware until the decode timeout
/// (observed deterministically on COCO val 000000122046.jpg, i.MX95). The
/// metadata carries nothing the hardware needs, so it is dropped during the
/// copy we already pay for. APP0 (JFIF) and APP14 (Adobe — its transform flag
/// changes component-colour interpretation) are kept; both are tiny and never
/// embed thumbnails. Falls back to a verbatim copy when the marker walk fails
/// (corrupt or unusual stream — the hardware error path handles it).
fn stage_jpeg(out: &mut [u8], data: &[u8]) -> usize {
    match copy_jpeg_stripped(out, data) {
        Some(n) => n,
        None => {
            out[..data.len()].copy_from_slice(data);
            data.len()
        }
    }
}

/// The marker walk behind [`stage_jpeg`]: copy every pre-scan segment except
/// APP1–APP13/APP15 and COM, then the scan (SOS onward) verbatim. `None` when
/// the stream doesn't parse as a baseline marker sequence or doesn't fit.
fn copy_jpeg_stripped(out: &mut [u8], data: &[u8]) -> Option<usize> {
    if data.len() < 4 || data[0] != 0xFF || data[1] != 0xD8 {
        return None;
    }
    out.get_mut(..2)?.copy_from_slice(&data[..2]); // SOI
    let mut o = 2;
    let mut i = 2;
    loop {
        if i + 4 > data.len() || data[i] != 0xFF {
            return None;
        }
        let marker = data[i + 1];
        if marker == 0xDA {
            // SOS: entropy-coded data follows — copy the remainder verbatim.
            let rest = &data[i..];
            out.get_mut(o..o + rest.len())?.copy_from_slice(rest);
            return Some(o + rest.len());
        }
        // Fill bytes or standalone markers before SOS aren't expected from a
        // well-formed encoder; hand the stream over verbatim.
        if marker == 0xFF || marker == 0x01 || (0xD0..=0xD9).contains(&marker) {
            return None;
        }
        let len = ((data[i + 2] as usize) << 8) | data[i + 3] as usize;
        if len < 2 || i + 2 + len > data.len() {
            return None;
        }
        let keep = !matches!(marker, 0xE1..=0xED | 0xEF | 0xFE);
        if keep {
            let seg = &data[i..i + 2 + len];
            out.get_mut(o..o + seg.len())?.copy_from_slice(seg);
            o += seg.len();
        }
        i += 2 + len;
    }
}

/// Deinterleave packed `[Y,Cb,Cr]` rows (`V4L2_PIX_FMT_YUV24`) into NV24:
/// a Y plane of `h` rows followed by an interleaved CbCr plane of `2h` rows
/// (each luma row owns two `dst_stride` UV rows — `uv_rows_per_luma = 2`).
/// NEON-accelerated on aarch64, scalar elsewhere.
fn deinterleave_yuv24_to_nv24(
    src: &[u8],
    src_stride: usize,
    dst: &mut [u8],
    dst_stride: usize,
    w: usize,
    h: usize,
) {
    let (y_region, uv_region) = dst.split_at_mut(h * dst_stride);
    for row in 0..h {
        let s = &src[row * src_stride..][..w * 3];
        let dy = &mut y_region[row * dst_stride..][..w];
        let duv = &mut uv_region[row * 2 * dst_stride..][..w * 2];
        deinterleave_row(s, dy, duv, w);
    }
}

/// One row of YUV24 → NV24: `src` is exactly `3w` bytes, `dy` `w`, `duv` `2w`.
#[inline]
fn deinterleave_row(src: &[u8], dy: &mut [u8], duv: &mut [u8], w: usize) {
    #[cfg(target_arch = "aarch64")]
    {
        // SAFETY: the slice lengths are checked by the caller's sub-slicing;
        // NEON is baseline on aarch64 (no runtime feature detection needed).
        unsafe { deinterleave_row_neon(src, dy, duv, w) }
    }
    #[cfg(not(target_arch = "aarch64"))]
    deinterleave_row_scalar(src, dy, duv, 0, w);
}

/// Scalar YUV24 row deinterleave from pixel `start` (the NEON tail).
fn deinterleave_row_scalar(src: &[u8], dy: &mut [u8], duv: &mut [u8], start: usize, w: usize) {
    for x in start..w {
        let s = x * 3;
        dy[x] = src[s];
        duv[2 * x] = src[s + 1];
        duv[2 * x + 1] = src[s + 2];
    }
}

/// NEON YUV24 row deinterleave: `vld3q_u8` splits 48 packed bytes into Y, Cb,
/// and Cr lanes; Y stores straight, Cb/Cr re-interleave via `vst2q_u8`.
/// 16 pixels per iteration, scalar tail.
///
/// # Safety
/// `src` must hold at least `3w` bytes, `dy` at least `w`, `duv` at least `2w`.
#[cfg(target_arch = "aarch64")]
unsafe fn deinterleave_row_neon(src: &[u8], dy: &mut [u8], duv: &mut [u8], w: usize) {
    use core::arch::aarch64::{uint8x16x2_t, vld3q_u8, vst1q_u8, vst2q_u8};
    let mut x = 0usize;
    while x + 16 <= w {
        // SAFETY (caller contract): x+16 <= w ⇒ 48 bytes readable at src[3x],
        // 16 writable at dy[x], 32 writable at duv[2x].
        let v = vld3q_u8(src.as_ptr().add(x * 3));
        vst1q_u8(dy.as_mut_ptr().add(x), v.0);
        vst2q_u8(duv.as_mut_ptr().add(x * 2), uint8x16x2_t(v.1, v.2));
        x += 16;
    }
    deinterleave_row_scalar(src, dy, duv, x, w);
}

/// Copy the interleaved NV12 CbCr plane from a hardware CAPTURE buffer into the
/// destination tensor, cropping to the logical image width.
///
/// An NV12 chroma row carries `ceil(final_w / 2)` (Cb, Cr) pairs, which is
/// `even(final_w)` bytes. Copying only `final_w` bytes would silently drop the
/// final Cr sample on **odd** widths. This function always copies
/// `final_w.next_multiple_of(2)` bytes per row, preserving the last Cr.
///
/// # Arguments
///
/// * `src`       — the hardware CAPTURE chroma plane (may be MCU-rounded larger).
/// * `src_stride`  — bytes per row in `src` (MCU-rounded CAPTURE pitch).
/// * `dst`       — the destination chroma region starting at the UV plane base.
/// * `dst_stride`  — bytes per row in `dst` (64-aligned tensor pitch).
/// * `final_w`   — logical image width (may be odd).
/// * `final_h`   — logical image height (may be odd).
fn copy_nv12_chroma(
    src: &[u8],
    src_stride: usize,
    dst: &mut [u8],
    dst_stride: usize,
    final_w: usize,
    final_h: usize,
) {
    let chroma_w = final_w.next_multiple_of(2);
    for cy in 0..final_h.div_ceil(2) {
        let s = cy * src_stride;
        let o = cy * dst_stride;
        dst[o..o + chroma_w].copy_from_slice(&src[s..s + chroma_w]);
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

fn reqbufs(fd: RawFd, buf_type: u32, count: u32, memory: u32) -> nix::Result<()> {
    let mut rb = ioctl::v4l2_requestbuffers {
        count,
        type_: buf_type,
        memory,
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

/// `VIDIOC_QBUF` the single-plane CAPTURE queue in DMABUF mode, importing
/// `dmabuf_fd` as the backing; `length` is the dmabuf size (may exceed the
/// format's `sizeimage`). The hardware decodes straight into the import.
fn qbuf_capture_dmabuf(fd: RawFd, dmabuf_fd: RawFd, length: usize) -> nix::Result<()> {
    let mut planes = [ioctl::v4l2_plane::default(); ioctl::VIDEO_MAX_PLANES];
    planes[0].set_fd(dmabuf_fd);
    planes[0].length = length as u32;
    let mut b = ioctl::v4l2_buffer {
        type_: ioctl::V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE,
        memory: ioctl::V4L2_MEMORY_DMABUF,
        index: 0,
        length: 1,
        ..Default::default()
    };
    b.set_planes(planes.as_mut_ptr());
    // SAFETY: valid buffer + plane array; `dmabuf_fd` outlives the call.
    unsafe { ioctl::vidioc_qbuf(fd, &mut b) }.map(|_| ())
}

/// `VIDIOC_DQBUF` the decoded single-plane DMABUF CAPTURE buffer.
fn dqbuf_capture(fd: RawFd) -> nix::Result<()> {
    let mut planes = [ioctl::v4l2_plane::default(); ioctl::VIDEO_MAX_PLANES];
    let mut b = ioctl::v4l2_buffer {
        type_: ioctl::V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE,
        memory: ioctl::V4L2_MEMORY_DMABUF,
        length: 1,
        ..Default::default()
    };
    b.set_planes(planes.as_mut_ptr());
    // SAFETY: valid buffer + plane array for DQBUF.
    unsafe { ioctl::vidioc_dqbuf(fd, &mut b) }.map(|_| ())
}

/// Drain pending events after `SOURCE_CHANGE`, bounded to avoid the
/// driver-re-returns-the-same-event hang.
///
/// The device fd is opened blocking (no `O_NONBLOCK`), and on the i.MX95
/// `mxc-jpeg` driver `VIDIOC_DQEVENT` *blocks* when no event is pending instead
/// of returning `EINVAL`. So whenever the best-effort `SOURCE_CHANGE` poll above
/// times out — no event arrived in the window — an unconditional dequeue wedges
/// the process, and the device, forever. (The driver itself decodes fine; this
/// is purely an event-dequeue protocol hazard.) Gate every dequeue on a
/// zero-timeout poll: no `POLLPRI` ⇒ no pending event ⇒ stop instead of block.
fn drain_events(fd: RawFd) {
    for _ in 0..MAX_EVENTS {
        if !poll_ready(fd, PollFlags::POLLPRI, 0) {
            break;
        }
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

    /// `copy_nv12_chroma` with an odd `final_w` must copy `even(final_w)` bytes
    /// per chroma row, preserving the final Cr sample that a naïve `final_w`-byte
    /// copy would have silently dropped.
    #[test]
    fn copy_nv12_chroma_odd_w_preserves_last_cr() {
        // Odd width 5: even(5) = 6. One chroma row (final_h = 2 → 1 chroma row).
        let final_w = 5usize;
        let final_h = 2usize;
        let src_stride = 8usize; // MCU-rounded (>= even(final_w)=6)
        let dst_stride = 64usize; // 64-aligned tensor pitch

        // Build a src chroma row: [Cb0, Cr0, Cb1, Cr1, Cb2, Cr2, PAD, PAD]
        // Index 4 is Cb2 and index 5 is Cr2 — the bytes that a naïve
        // `final_w=5` copy would have omitted (it would stop at index 4).
        let mut src = vec![0u8; src_stride];
        src[0] = 10; // Cb0
        src[1] = 11; // Cr0
        src[2] = 20; // Cb1
        src[3] = 21; // Cr1
        src[4] = 30; // Cb2
        src[5] = 31; // Cr2  ← last valid Cr, must NOT be dropped

        let mut dst = vec![0u8; dst_stride];
        copy_nv12_chroma(&src, src_stride, &mut dst, dst_stride, final_w, final_h);

        // Bytes 0..6 in dst must match src 0..6 exactly.
        assert_eq!(
            &dst[..6],
            &src[..6],
            "copy_nv12_chroma: first 6 bytes (even(5)) must be copied verbatim"
        );
        // Specifically, the last Cr (index 5) must NOT be zero / dropped.
        assert_eq!(
            dst[5], 31,
            "copy_nv12_chroma: last Cr at byte 5 must be preserved for odd final_w=5"
        );
    }

    /// Sanity-check the even-width case: `copy_nv12_chroma` with an even
    /// `final_w` behaves identically to copying `final_w` bytes (no implicit
    /// extra byte).
    #[test]
    fn copy_nv12_chroma_even_w_no_extra_byte() {
        let final_w = 4usize; // even
        let final_h = 2usize;
        let src_stride = 8usize;
        let dst_stride = 64usize;

        let mut src = vec![0u8; src_stride];
        src[0] = 1;
        src[1] = 2;
        src[2] = 3;
        src[3] = 4; // last valid byte for even(4)=4
        src[4] = 99; // must NOT be copied

        let mut dst = vec![0u8; dst_stride];
        copy_nv12_chroma(&src, src_stride, &mut dst, dst_stride, final_w, final_h);

        assert_eq!(&dst[..4], &src[..4]);
        assert_eq!(
            dst[4], 0,
            "copy_nv12_chroma: byte beyond even(final_w) must remain untouched"
        );
    }

    /// Build a minimal marker sequence: SOI + the given segments + SOS + scan.
    fn fake_jpeg(segments: &[(u8, &[u8])]) -> Vec<u8> {
        let mut v = vec![0xFF, 0xD8];
        for (marker, payload) in segments {
            v.extend_from_slice(&[0xFF, *marker]);
            let len = (payload.len() + 2) as u16;
            v.extend_from_slice(&len.to_be_bytes());
            v.extend_from_slice(payload);
        }
        // SOS with a 6-byte header + fake entropy data + EOI.
        v.extend_from_slice(&[0xFF, 0xDA, 0x00, 0x08, 1, 1, 0, 0, 63, 0]);
        v.extend_from_slice(&[0xAB; 32]);
        v.extend_from_slice(&[0xFF, 0xD9]);
        v
    }

    /// APP13 (and other metadata) must be stripped while DQT/SOF/DHT, APP0,
    /// APP14, and the scan survive byte-for-byte — including an APP13 payload
    /// that embeds JPEG markers (the mxc-jpeg wedge trigger).
    #[test]
    fn stage_jpeg_strips_metadata_segments() {
        let evil_app13 = [
            b"Photoshop 3.0\x00".as_slice(),
            &[0xFF, 0xD8, 0xFF, 0xDA, 0x12, 0x34],
        ]
        .concat();
        let src = fake_jpeg(&[
            (0xE0, b"JFIF\x00"),                       // APP0: kept
            (0xE1, &[0x55; 64]),                       // EXIF: stripped
            (0xED, &evil_app13),                       // Photoshop IRB: stripped
            (0xDB, &[0x11; 65]),                       // DQT: kept
            (0xC0, &[8, 0, 16, 0, 16, 1, 1, 0x11, 0]), // SOF0: kept
            (0xEE, b"Adobe\x00"),                      // APP14: kept
            (0xC4, &[0x22; 20]),                       // DHT: kept
            (0xFE, b"comment"),                        // COM: stripped
        ]);
        let want = fake_jpeg(&[
            (0xE0, b"JFIF\x00"),
            (0xDB, &[0x11; 65]),
            (0xC0, &[8, 0, 16, 0, 16, 1, 1, 0x11, 0]),
            (0xEE, b"Adobe\x00"),
            (0xC4, &[0x22; 20]),
        ]);
        let mut out = vec![0u8; src.len()];
        let n = stage_jpeg(&mut out, &src);
        assert_eq!(&out[..n], &want[..]);
    }

    /// A stream that doesn't parse as plain marker segments must be staged
    /// verbatim, not rejected.
    #[test]
    fn stage_jpeg_falls_back_to_verbatim() {
        for bad in [
            &b"not a jpeg at all"[..],
            &[0xFF, 0xD8, 0x00, 0x00, 0xFF][..], // garbage after SOI
            &[0xFF, 0xD8, 0xFF, 0xE1, 0xFF, 0xFF, 0x00][..], // length overruns
        ] {
            let mut out = vec![0u8; bad.len()];
            let n = stage_jpeg(&mut out, bad);
            assert_eq!(&out[..n], bad);
        }
    }

    /// `deinterleave_yuv24_to_nv24` must match a naïve reference for odd
    /// widths, padded strides, and partial NEON tails (w % 16 != 0). On
    /// aarch64 this verifies the NEON path against scalar addressing.
    #[test]
    fn deinterleave_yuv24_matches_reference() {
        for &(w, h) in &[(37usize, 5usize), (16, 3), (1, 1), (64, 2), (53, 4)] {
            let src_stride = (w * 3).next_multiple_of(64);
            let dst_stride = w.next_multiple_of(64);
            let mut src = vec![0u8; src_stride * h];
            for (i, v) in src.iter_mut().enumerate() {
                *v = (i % 251) as u8;
            }
            let mut got = vec![0u8; 3 * h * dst_stride];
            deinterleave_yuv24_to_nv24(&src, src_stride, &mut got, dst_stride, w, h);

            let mut want = vec![0u8; 3 * h * dst_stride];
            let uv_base = h * dst_stride;
            for y in 0..h {
                for x in 0..w {
                    let s = y * src_stride + x * 3;
                    want[y * dst_stride + x] = src[s];
                    let u = uv_base + y * 2 * dst_stride + x * 2;
                    want[u] = src[s + 1];
                    want[u + 1] = src[s + 2];
                }
            }
            assert_eq!(got, want, "mismatch at {w}x{h}");
        }
    }

    fn testdata_file(name: &str) -> Option<Vec<u8>> {
        let root = std::env::var("EDGEFIRST_TESTDATA_DIR")
            .map(std::path::PathBuf::from)
            .unwrap_or_else(|_| {
                std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../testdata")
            });
        std::fs::read(root.join(name)).ok()
    }

    /// On-target probe for the DMABUF-first reconfigure design: alternates two
    /// JPEG geometries while keeping the OUTPUT queue allocated and streaming,
    /// rebuilding only the CAPTURE queue in `V4L2_MEMORY_DMABUF` mode against a
    /// persistent, deliberately oversized DMA scratch buffer.
    ///
    /// Answers three driver-behavior questions the design depends on:
    /// 1. Is `REQBUFS(1, DMABUF)` allocation-free (sub-ms), unlike MMAP?
    /// 2. Is `S_FMT(CAPTURE)` accepted while the OUTPUT queue keeps streaming?
    /// 3. Is a QBUF `plane.length` larger than the format's `sizeimage` accepted?
    #[test]
    #[ignore = "on-target hardware probe; run with --ignored --nocapture on a JPEG M2M device"]
    fn probe_dmabuf_reconfigure() {
        use std::time::Instant;
        const OUT: u32 = ioctl::V4L2_BUF_TYPE_VIDEO_OUTPUT_MPLANE;
        const CAP: u32 = ioctl::V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE;

        let Some(dev) = device::probe() else {
            eprintln!("skip: no v4l2 jpeg decoder on this host");
            return;
        };
        if dev.api != ApiVariant::MultiPlanar {
            eprintln!("skip: device is not multi-planar");
            return;
        }
        let (Some(jpeg_a), Some(jpeg_b)) =
            (testdata_file("zidane.jpg"), testdata_file("giraffe.jpg"))
        else {
            eprintln!("skip: testdata not found (set EDGEFIRST_TESTDATA_DIR)");
            return;
        };

        // Persistent CAPTURE scratch: 4 MiB DMA buffer — larger than any
        // sizeimage in this probe, to confirm question 3.
        let scratch = match Tensor::<u8>::image(
            4096,
            1024,
            PixelFormat::Grey,
            Some(TensorMemory::Dma),
            edgefirst_tensor::CpuAccess::ReadWrite,
        ) {
            Ok(t) => t,
            Err(e) => {
                eprintln!("skip: no DMA heap ({e})");
                return;
            }
        };
        let scratch_fd = scratch.dmabuf().unwrap().as_raw_fd();
        let scratch_cap = scratch.capacity_bytes();

        let fd = dev.fd();
        let sub = ioctl::v4l2_event_subscription {
            type_: ioctl::V4L2_EVENT_SOURCE_CHANGE,
            ..Default::default()
        };
        // SAFETY: valid subscription struct.
        unsafe { ioctl::vidioc_subscribe_event(fd, &sub) }.expect("SUBSCRIBE_EVENT");

        // OUTPUT: one persistent 2 MiB coded buffer, set up once, streamed once.
        let mut ofmt = ioctl::v4l2_format {
            type_: OUT,
            ..Default::default()
        };
        // SAFETY: type_ selects the multi-planar variant.
        {
            let p = unsafe { ofmt.pix_mp() };
            p.width = 1280;
            p.height = 720;
            p.pixelformat = ioctl::V4L2_PIX_FMT_JPEG;
            p.field = ioctl::V4L2_FIELD_NONE;
            p.num_planes = 1;
            p.plane_fmt[0].sizeimage = 2 * 1024 * 1024;
        }
        // SAFETY: valid v4l2_format.
        unsafe { ioctl::vidioc_s_fmt(fd, &mut ofmt) }.expect("S_FMT OUTPUT");
        reqbufs(fd, OUT, 1, ioctl::V4L2_MEMORY_MMAP).expect("REQBUFS OUTPUT");
        let (olen, ooff) = querybuf(fd, OUT, 1).expect("QUERYBUF OUTPUT");
        let mut out_map = Mmap::new(borrow(fd), olen, ooff).expect("mmap OUTPUT");
        streamon(fd, OUT).expect("STREAMON OUTPUT");

        let mut cap_live = false;
        for round in 0..10 {
            for (name, jpeg) in [("zidane 1280x720", &jpeg_a), ("giraffe 640x640", &jpeg_b)] {
                let t0 = Instant::now();
                if cap_live {
                    streamoff(fd, CAP).expect("STREAMOFF CAPTURE");
                    reqbufs(fd, CAP, 0, ioctl::V4L2_MEMORY_DMABUF).expect("REQBUFS CAPTURE 0");
                }
                out_map.as_mut_slice()[..jpeg.len()].copy_from_slice(jpeg);
                qbuf_output(fd, jpeg.len()).expect("QBUF OUTPUT");
                poll_ready(fd, PollFlags::POLLPRI, SOURCE_CHANGE_TIMEOUT_MS);
                drain_events(fd);

                let mut gfmt = ioctl::v4l2_format {
                    type_: CAP,
                    ..Default::default()
                };
                // SAFETY: valid v4l2_format for G_FMT.
                unsafe { ioctl::vidioc_g_fmt(fd, &mut gfmt) }.expect("G_FMT CAPTURE");
                // SAFETY: mplane variant.
                let got = *unsafe { gfmt.pix_mp() };
                let t_gfmt = t0.elapsed();

                // Question 2: S_FMT(CAP) single-plane NV12 while OUTPUT streams.
                let mut sfmt = ioctl::v4l2_format {
                    type_: CAP,
                    ..Default::default()
                };
                // SAFETY: mplane variant.
                {
                    let p = unsafe { sfmt.pix_mp() };
                    p.width = got.width;
                    p.height = got.height;
                    p.pixelformat = ioctl::V4L2_PIX_FMT_NV12;
                    p.field = ioctl::V4L2_FIELD_NONE;
                    p.colorspace = got.colorspace;
                    p.num_planes = 1;
                }
                // SAFETY: valid v4l2_format.
                let sfmt_res = unsafe { ioctl::vidioc_s_fmt(fd, &mut sfmt) };
                let mut gfmt2 = ioctl::v4l2_format {
                    type_: CAP,
                    ..Default::default()
                };
                // SAFETY: valid v4l2_format for G_FMT.
                unsafe { ioctl::vidioc_g_fmt(fd, &mut gfmt2) }.expect("G_FMT CAPTURE (post S_FMT)");
                // SAFETY: mplane variant.
                let cap_fmt = *unsafe { gfmt2.pix_mp() };
                assert_eq!(
                    cap_fmt.num_planes, 1,
                    "probe requires the single-plane CAPTURE the backend uses"
                );
                let sizeimage = cap_fmt.plane_fmt[0].sizeimage as usize;
                assert!(
                    scratch_cap >= sizeimage,
                    "scratch too small: {scratch_cap} < {sizeimage}"
                );

                // Question 1: REQBUFS(DMABUF) cost in isolation.
                let t1 = Instant::now();
                reqbufs(fd, CAP, 1, ioctl::V4L2_MEMORY_DMABUF).expect("REQBUFS CAPTURE dmabuf");
                let t_reqbufs = t1.elapsed();
                streamon(fd, CAP).expect("STREAMON CAPTURE");
                cap_live = true;

                // Question 3: plane.length = 4 MiB scratch capacity > sizeimage.
                qbuf_capture_dmabuf(fd, scratch_fd, scratch_cap).expect("QBUF CAPTURE dmabuf");
                assert!(
                    poll_ready(fd, PollFlags::POLLIN, DECODE_TIMEOUT_MS),
                    "decode timeout"
                );
                dqbuf_capture(fd).expect("DQBUF CAPTURE");
                dqbuf_output(fd).expect("DQBUF OUTPUT");
                let total = t0.elapsed();

                let sfmt_str = match &sfmt_res {
                    Ok(_) => "ok".to_string(),
                    Err(e) => format!("ERR({e})"),
                };
                eprintln!(
                    "round {round:2} {name:18} total={total:9.2?} hdr+gfmt={t_gfmt:9.2?} \
                     reqbufs={t_reqbufs:9.2?} s_fmt={sfmt_str} fmt={} {}x{} sizeimage={}",
                    ioctl::fourcc_str(cap_fmt.pixelformat),
                    cap_fmt.width,
                    cap_fmt.height,
                    sizeimage,
                );
            }
        }

        let _ = streamoff(fd, CAP);
        let _ = streamoff(fd, OUT);
        let _ = reqbufs(fd, CAP, 0, ioctl::V4L2_MEMORY_DMABUF);
        let _ = reqbufs(fd, OUT, 0, ioctl::V4L2_MEMORY_MMAP);
    }

    // --- raw-throughput probe helpers (index-aware variants of the thin
    // ioctl wrappers, which all hardcode buffer index 0) ------------------

    fn reqbufs_n(fd: RawFd, buf_type: u32, count: u32, memory: u32) -> nix::Result<u32> {
        let mut rb = ioctl::v4l2_requestbuffers {
            count,
            type_: buf_type,
            memory,
            ..Default::default()
        };
        // SAFETY: valid requestbuffers struct.
        unsafe { ioctl::vidioc_reqbufs(fd, &mut rb) }.map(|_| rb.count)
    }

    fn querybuf_idx(fd: RawFd, buf_type: u32, index: u32) -> nix::Result<(usize, i64)> {
        let mut planes = [ioctl::v4l2_plane::default(); ioctl::VIDEO_MAX_PLANES];
        let mut b = ioctl::v4l2_buffer {
            type_: buf_type,
            memory: ioctl::V4L2_MEMORY_MMAP,
            index,
            length: 1,
            ..Default::default()
        };
        b.set_planes(planes.as_mut_ptr());
        // SAFETY: valid buffer + plane array for QUERYBUF.
        unsafe { ioctl::vidioc_querybuf(fd, &mut b) }?;
        Ok((planes[0].length as usize, planes[0].mem_offset() as i64))
    }

    fn qbuf_out_idx(fd: RawFd, index: u32, bytesused: usize) -> nix::Result<()> {
        let mut planes = [ioctl::v4l2_plane::default(); ioctl::VIDEO_MAX_PLANES];
        planes[0].bytesused = bytesused as u32;
        let mut b = ioctl::v4l2_buffer {
            type_: ioctl::V4L2_BUF_TYPE_VIDEO_OUTPUT_MPLANE,
            memory: ioctl::V4L2_MEMORY_MMAP,
            index,
            length: 1,
            ..Default::default()
        };
        b.set_planes(planes.as_mut_ptr());
        // SAFETY: valid buffer + plane array for QBUF.
        unsafe { ioctl::vidioc_qbuf(fd, &mut b) }.map(|_| ())
    }

    fn qbuf_cap_fd_idx(fd: RawFd, index: u32, dmabuf_fd: RawFd, len: usize) -> nix::Result<()> {
        let mut planes = [ioctl::v4l2_plane::default(); ioctl::VIDEO_MAX_PLANES];
        planes[0].set_fd(dmabuf_fd);
        planes[0].length = len as u32;
        let mut b = ioctl::v4l2_buffer {
            type_: ioctl::V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE,
            memory: ioctl::V4L2_MEMORY_DMABUF,
            index,
            length: 1,
            ..Default::default()
        };
        b.set_planes(planes.as_mut_ptr());
        // SAFETY: valid buffer + plane array for QBUF.
        unsafe { ioctl::vidioc_qbuf(fd, &mut b) }.map(|_| ())
    }

    fn dqbuf_idx(fd: RawFd, buf_type: u32, memory: u32) -> nix::Result<u32> {
        let mut planes = [ioctl::v4l2_plane::default(); ioctl::VIDEO_MAX_PLANES];
        let mut b = ioctl::v4l2_buffer {
            type_: buf_type,
            memory,
            length: 1,
            ..Default::default()
        };
        b.set_planes(planes.as_mut_ptr());
        // SAFETY: valid buffer + plane array for DQBUF.
        unsafe { ioctl::vidioc_dqbuf(fd, &mut b) }.map(|_| b.index)
    }

    /// Drive one M2M context at the given queue depth on a fixed JPEG and
    /// return steady-state decode FPS. OUTPUT buffers are prefilled once
    /// (every frame decodes the same bitstream — no per-frame copy), so this
    /// measures the pure driver + hardware pipeline rate.
    fn run_throughput(dev: &ProbedDevice, jpeg: &[u8], depth: u32, frames: usize) -> Option<f64> {
        use std::time::Instant;
        const OUT: u32 = ioctl::V4L2_BUF_TYPE_VIDEO_OUTPUT_MPLANE;
        const CAP: u32 = ioctl::V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE;
        let fd = dev.fd();

        let sub = ioctl::v4l2_event_subscription {
            type_: ioctl::V4L2_EVENT_SOURCE_CHANGE,
            ..Default::default()
        };
        // SAFETY: valid subscription struct.
        unsafe { ioctl::vidioc_subscribe_event(fd, &sub) }.ok()?;

        // OUTPUT: depth buffers, all prefilled with the same JPEG.
        let mut ofmt = ioctl::v4l2_format {
            type_: OUT,
            ..Default::default()
        };
        // SAFETY: mplane variant.
        {
            let p = unsafe { ofmt.pix_mp() };
            p.width = 1280;
            p.height = 720;
            p.pixelformat = ioctl::V4L2_PIX_FMT_JPEG;
            p.field = ioctl::V4L2_FIELD_NONE;
            p.num_planes = 1;
            p.plane_fmt[0].sizeimage = ((jpeg.len() + 4095) & !4095) as u32;
        }
        // SAFETY: valid v4l2_format.
        unsafe { ioctl::vidioc_s_fmt(fd, &mut ofmt) }.ok()?;
        let got_out = reqbufs_n(fd, OUT, depth, ioctl::V4L2_MEMORY_MMAP).ok()?;
        if got_out < depth {
            eprintln!("  driver clamped OUTPUT buffers: {depth} -> {got_out}");
        }
        let mut out_maps = Vec::new();
        for i in 0..got_out {
            let (len, off) = querybuf_idx(fd, OUT, i).ok()?;
            let mut m = Mmap::new(borrow(fd), len, off).ok()?;
            m.as_mut_slice()[..jpeg.len()].copy_from_slice(jpeg);
            out_maps.push(m);
        }
        streamon(fd, OUT).ok()?;
        qbuf_out_idx(fd, 0, jpeg.len()).ok()?;
        poll_ready(fd, PollFlags::POLLPRI, SOURCE_CHANGE_TIMEOUT_MS);
        drain_events(fd);

        // CAPTURE: single-plane NV12, depth scratch dmabufs.
        let mut gfmt = ioctl::v4l2_format {
            type_: CAP,
            ..Default::default()
        };
        // SAFETY: valid v4l2_format for G_FMT.
        unsafe { ioctl::vidioc_g_fmt(fd, &mut gfmt) }.ok()?;
        // SAFETY: mplane variant.
        let got = *unsafe { gfmt.pix_mp() };
        let mut sfmt = ioctl::v4l2_format {
            type_: CAP,
            ..Default::default()
        };
        // SAFETY: mplane variant.
        {
            let p = unsafe { sfmt.pix_mp() };
            p.width = got.width;
            p.height = got.height;
            p.pixelformat = ioctl::V4L2_PIX_FMT_NV12;
            p.field = ioctl::V4L2_FIELD_NONE;
            p.colorspace = got.colorspace;
            p.num_planes = 1;
        }
        // SAFETY: valid v4l2_format.
        unsafe { ioctl::vidioc_s_fmt(fd, &mut sfmt) }.ok()?;
        // SAFETY: valid v4l2_format for G_FMT.
        unsafe { ioctl::vidioc_g_fmt(fd, &mut gfmt) }.ok()?;
        // SAFETY: mplane variant.
        let cap_fmt = *unsafe { gfmt.pix_mp() };
        if cap_fmt.num_planes != 1 {
            eprintln!("  capture is not single-plane; skipping");
            return None;
        }
        let sizeimage = cap_fmt.plane_fmt[0].sizeimage as usize;

        let mut scratches = Vec::new();
        for _ in 0..depth {
            let rows = sizeimage.div_ceil(4096);
            let t = Tensor::<u8>::image(
                4096,
                rows,
                PixelFormat::Grey,
                Some(TensorMemory::Dma),
                edgefirst_tensor::CpuAccess::ReadWrite,
            )
            .ok()?;
            scratches.push(t);
        }
        let scratch_fds: Vec<RawFd> = scratches
            .iter()
            .map(|t| t.dmabuf().unwrap().as_raw_fd())
            .collect();

        let got_cap = reqbufs_n(fd, CAP, depth, ioctl::V4L2_MEMORY_DMABUF).ok()?;
        if got_cap < depth {
            eprintln!("  driver clamped CAPTURE buffers: {depth} -> {got_cap}");
        }
        streamon(fd, CAP).ok()?;

        // Fill both queues to the working depth.
        let live = depth.min(got_out).min(got_cap);
        for i in 0..live {
            qbuf_cap_fd_idx(
                fd,
                i,
                scratch_fds[i as usize],
                scratches[i as usize].capacity_bytes(),
            )
            .ok()?;
        }
        for i in 1..live {
            qbuf_out_idx(fd, i, jpeg.len()).ok()?;
        }

        // Warmup, then measure: each completion immediately requeues the same
        // buffer pair, keeping `live` decodes in flight.
        let warmup = 30usize;
        let mut t0 = Instant::now();
        for n in 0..(warmup + frames) {
            if n == warmup {
                t0 = Instant::now();
            }
            if !poll_ready(fd, PollFlags::POLLIN, DECODE_TIMEOUT_MS) {
                eprintln!("  decode timeout at frame {n}");
                return None;
            }
            let ci = dqbuf_idx(fd, CAP, ioctl::V4L2_MEMORY_DMABUF).ok()?;
            let oi = dqbuf_idx(fd, OUT, ioctl::V4L2_MEMORY_MMAP).ok()?;
            qbuf_out_idx(fd, oi, jpeg.len()).ok()?;
            qbuf_cap_fd_idx(
                fd,
                ci,
                scratch_fds[ci as usize],
                scratches[ci as usize].capacity_bytes(),
            )
            .ok()?;
        }
        let fps = frames as f64 / t0.elapsed().as_secs_f64();

        let _ = streamoff(fd, CAP);
        let _ = streamoff(fd, OUT);
        let _ = reqbufs(fd, CAP, 0, ioctl::V4L2_MEMORY_DMABUF);
        let _ = reqbufs(fd, OUT, 0, ioctl::V4L2_MEMORY_MMAP);
        Some(fps)
    }

    /// On-target probe: raw mxc-jpeg decode throughput (1280×720 4:2:0 →
    /// NV12, no userspace copies) versus queue depth and context count.
    /// Answers whether deeper buffer queues ("batch" decoding) or parallel
    /// M2M contexts (one per decode worker) raise hardware throughput.
    #[test]
    #[ignore = "on-target hardware probe; run with --ignored --nocapture on a JPEG M2M device"]
    fn probe_decode_throughput() {
        let Some(jpeg) = testdata_file("zidane.jpg") else {
            eprintln!("skip: testdata not found (set EDGEFIRST_TESTDATA_DIR)");
            return;
        };
        let frames = 300usize;

        for depth in [1u32, 2, 4] {
            let Some(dev) = device::probe() else {
                eprintln!("skip: no v4l2 jpeg decoder");
                return;
            };
            match run_throughput(&dev, &jpeg, depth, frames) {
                Some(fps) => eprintln!("single context, depth {depth}: {fps:7.1} FPS"),
                None => eprintln!("single context, depth {depth}: failed"),
            }
        }

        // Parallel contexts: one fd per thread, depth 1 each — the shape the
        // profiler's decode workers (one ImageDecoder each) already produce.
        for contexts in [2usize, 4] {
            let mut devs = Vec::new();
            for _ in 0..contexts {
                match device::probe() {
                    Some(d) => devs.push(d),
                    None => {
                        eprintln!("skip: could not open context");
                        return;
                    }
                }
            }
            let t0 = std::time::Instant::now();
            std::thread::scope(|s| {
                for dev in &devs {
                    let jpeg = &jpeg;
                    s.spawn(move || {
                        if run_throughput(dev, jpeg, 1, frames).is_none() {
                            eprintln!("  parallel context failed");
                        }
                    });
                }
            });
            // Each thread also runs its 30-frame warmup; include it in the
            // aggregate (small, and identical across configurations).
            let total = contexts * (frames + 30);
            let fps = total as f64 / t0.elapsed().as_secs_f64();
            eprintln!("{contexts} contexts, depth 1: {fps:7.1} FPS aggregate");
        }
    }
}
