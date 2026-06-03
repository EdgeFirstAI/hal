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
//! persistent streaming session: after the first decode at a given geometry
//! the OUTPUT/CAPTURE buffers stay set up and streaming, so subsequent decodes
//! pay only the per-frame queue/dequeue cost. A geometry change rebuilds the
//! stream; a hardware failure resets it and (after repeated failures) demotes
//! the device to the CPU decoder.
//!
//! Two CAPTURE paths:
//! - **Zero-copy (DMABUF):** when the destination is a DMA tensor with
//!   MCU(16)-aligned dimensions and the driver accepts a single-plane
//!   contiguous CAPTURE at the tensor pitch, the tensor's dmabuf fd is imported
//!   as the CAPTURE buffer and the hardware decodes straight into it — no copy.
//! - **MMAP + copy:** otherwise the driver buffers are mapped and the decoded
//!   `NV12`/`GREY` planes are copied (cropped to the logical image) into the
//!   destination. 4:4:4 (`YUV3`) capture is left to the CPU decoder.
//!
//! Single-planar M2M devices fall back to the CPU decoder for now.

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
/// After the first decode at a given geometry the OUTPUT/CAPTURE buffers stay
/// allocated, mapped, and streaming, so subsequent same-geometry decodes skip
/// the ~0.9 ms per-image setup. A geometry change (or a larger JPEG than the
/// allocated OUTPUT buffer) rebuilds the stream; a hardware failure drops it.
pub(crate) struct V4l2Context {
    device: ProbedDevice,
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
    /// CAPTURE queue is in DMABUF (zero-copy) mode: the hardware decodes
    /// straight into the caller's DMA tensor; `collect` imports the tensor fd
    /// each frame and performs no plane copy. `false` = MMAP + copy-out.
    dmabuf: bool,
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
            stream: None,
            failures: 0,
        }
    }

    /// Decode one image, reusing the persistent stream when the geometry
    /// matches and the JPEG fits the allocated OUTPUT buffer, otherwise
    /// rebuilding the stream first.
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
        // Reuse only when the geometry AND the CAPTURE memory mode match — a
        // change in destination memory kind (DMA ↔ Mem) needs a rebuild.
        let reuse = matches!(
            &self.stream,
            Some(s) if s.jpeg_w == final_w as u32
                && s.jpeg_h == final_h as u32
                && s.out_sizeimage >= needed
                && s.dmabuf == dma_capable
        );

        if reuse {
            self.requeue_output(data)?;
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
        stream.out_map.as_mut_slice()[..data.len()].copy_from_slice(data);
        qbuf_output(fd, data.len()).map_err(|e| DecodeErr::Reset(format!("QBUF OUTPUT: {e}")))
    }

    /// Tear down any existing stream, then set up the OUTPUT + CAPTURE queues
    /// for a new geometry, leaving the first JPEG queued and both queues
    /// streaming. Chooses DMABUF (zero-copy) CAPTURE when the destination is a
    /// DMA tensor with a matching layout, else MMAP. Stores the [`Stream`].
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

        reqbufs(fd, OUT, 1, ioctl::V4L2_MEMORY_MMAP)
            .map_err(|e| DecodeErr::Reset(format!("REQBUFS OUTPUT: {e}")))?;
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
        let mut cap = *unsafe { cfmt.pix_mp() };

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

        // Save the driver's natural CAPTURE format so a failed zero-copy probe
        // can restore it for the MMAP path.
        let cap0 = cap;

        // Zero-copy eligibility: DMA destination, NV12/GREY output, and
        // MCU(16)-aligned dims so the CAPTURE geometry equals the logical image
        // (the decoded data lands exactly at the tensor's layout).
        let want_zc = dma_capable
            && matches!(output_fmt, PixelFormat::Nv12 | PixelFormat::Grey)
            && matches!(kind, CapKind::Nv12 | CapKind::Grey)
            && final_w.is_multiple_of(16)
            && final_h.is_multiple_of(16);

        let mut dmabuf = false;
        if want_zc {
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
                && cap_zc.plane_fmt[0].bytesperline as usize == dst_stride
                && dst_capacity >= cap_zc.plane_fmt[0].sizeimage as usize;
            if ok {
                cap = cap_zc;
                dmabuf = true;
            } else {
                // Driver refused single-plane contiguous output — restore its
                // natural format for the MMAP copy path.
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

        let num_planes = cap.num_planes as usize;
        let cap_struct = Capture {
            kind,
            num_planes,
            cap_h: cap.height as usize,
            luma_stride: cap.plane_fmt[0].bytesperline as usize,
            chroma_stride: cap.plane_fmt[1].bytesperline as usize,
            maps: Vec::new(),
        };

        if dmabuf {
            // DMABUF (zero-copy): no driver buffers to map; the tensor fd is
            // imported per-frame in `collect`.
            reqbufs(fd, CAP, 1, ioctl::V4L2_MEMORY_DMABUF)
                .map_err(|e| DecodeErr::Reset(format!("REQBUFS CAPTURE (dmabuf): {e}")))?;
            streamon(fd, CAP).map_err(|e| DecodeErr::Reset(format!("STREAMON CAPTURE: {e}")))?;
            self.stream = Some(Stream {
                jpeg_w: final_w as u32,
                jpeg_h: final_h as u32,
                out_sizeimage: needed,
                out_map,
                cap: cap_struct,
                dmabuf: true,
            });
            return Ok(());
        }

        // MMAP: allocate, map every plane, stream.
        reqbufs(fd, CAP, 1, ioctl::V4L2_MEMORY_MMAP)
            .map_err(|e| DecodeErr::Reset(format!("REQBUFS CAPTURE: {e}")))?;

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
            cap: Capture { maps, ..cap_struct },
            dmabuf: false,
        });
        Ok(())
    }

    /// Queue the CAPTURE buffer, wait for the decode, dequeue both buffers, and
    /// copy the decoded native planes (NV12/GREY) into `dst`. Shared by the
    /// build and reuse paths.
    fn collect<T: ImagePixel>(
        &mut self,
        dst: &mut Tensor<T>,
        output_fmt: PixelFormat,
        final_w: usize,
        final_h: usize,
        dst_stride: usize,
    ) -> Result<ImageInfo, DecodeErr> {
        let fd = self.device.fd();
        let (num_planes, dmabuf) = {
            let s = self
                .stream
                .as_ref()
                .ok_or_else(|| DecodeErr::Reset("no stream".into()))?;
            (s.cap.num_planes, s.dmabuf)
        };

        let info = ImageInfo {
            width: final_w,
            height: final_h,
            format: output_fmt,
            row_stride: dst_stride,
            rotation_degrees: 0,
            flip_horizontal: false,
        };

        if dmabuf {
            // Zero-copy: import the tensor's dmabuf fd as the CAPTURE buffer so
            // the hardware decodes straight into it — no plane copy. Planes
            // share the one allocation: Y at offset 0, CbCr after the luma
            // plane (NV12). Layout matches because dims are MCU-aligned and the
            // CAPTURE stride was forced to the tensor pitch.
            let dmabuf_fd = dst
                .dmabuf()
                .map_err(|e| DecodeErr::Fatal(e.into()))?
                .as_raw_fd();
            let capacity = dst.capacity_bytes();
            let offsets = [0usize, final_h * dst_stride];
            qbuf_capture_dmabuf(fd, dmabuf_fd, num_planes, &offsets, capacity)
                .map_err(|e| DecodeErr::Reset(format!("QBUF CAPTURE (dmabuf): {e}")))?;
            if !poll_ready(fd, PollFlags::POLLIN, DECODE_TIMEOUT_MS) {
                return Err(DecodeErr::Reset("CAPTURE decode timeout".into()));
            }
            dqbuf_capture(fd, num_planes, ioctl::V4L2_MEMORY_DMABUF)
                .map_err(|e| DecodeErr::Reset(format!("DQBUF CAPTURE (dmabuf): {e}")))?;
            dqbuf_output(fd).map_err(|e| DecodeErr::Reset(format!("DQBUF OUTPUT: {e}")))?;
            // Decoded pixels are in the tensor's DMA buffer; the consumer's
            // `Tensor::map()` issues the cache sync on read.
            return Ok(info);
        }

        // MMAP path: queue the driver buffer, decode, copy planes out.
        qbuf_capture(fd, num_planes).map_err(|e| DecodeErr::Reset(format!("QBUF CAPTURE: {e}")))?;
        if !poll_ready(fd, PollFlags::POLLIN, DECODE_TIMEOUT_MS) {
            return Err(DecodeErr::Reset("CAPTURE decode timeout".into()));
        }
        dqbuf_capture(fd, num_planes, ioctl::V4L2_MEMORY_MMAP)
            .map_err(|e| DecodeErr::Reset(format!("DQBUF CAPTURE: {e}")))?;
        dqbuf_output(fd).map_err(|e| DecodeErr::Reset(format!("DQBUF OUTPUT: {e}")))?;

        // Resolve the decoded planes. Greyscale → Y only; NV12/NV12M → Y +
        // interleaved CbCr. 4:4:4 YUV3 → native NV12 is left to the CPU.
        let stream = self
            .stream
            .as_ref()
            .ok_or_else(|| DecodeErr::Reset("no stream".into()))?;
        let cap = &stream.cap;
        let y_stride = cap.luma_stride;
        let (y_plane, chroma): (&[u8], Option<(&[u8], usize)>) = match (&cap.kind, cap.num_planes) {
            (CapKind::Nv12, n) if n >= 2 => (
                cap.maps[0].as_slice(),
                Some((cap.maps[1].as_slice(), cap.chroma_stride)),
            ),
            (CapKind::Nv12, _) => {
                let m = cap.maps[0].as_slice();
                let ys = cap.luma_stride * cap.cap_h;
                (&m[..ys], Some((&m[ys..], cap.luma_stride)))
            }
            (CapKind::Grey, _) => (cap.maps[0].as_slice(), None),
            (CapKind::Yuv444Packed, _) => {
                return Err(DecodeErr::Unsupported(
                    "4:4:4 YUV3 capture → native NV12 not handled on hardware; using CPU".into(),
                ));
            }
        };

        let mut map = dst.map().map_err(|e| DecodeErr::Fatal(e.into()))?;
        let dst_bytes: &mut [T] = &mut map;
        // SAFETY: native NV12/GREY are u8; the JPEG entry guarantees T == u8.
        let d: &mut [u8] = unsafe {
            std::slice::from_raw_parts_mut(dst_bytes.as_mut_ptr() as *mut u8, dst_bytes.len())
        };

        // Crop to the logical image (the CAPTURE buffer is MCU-rounded up).
        match output_fmt {
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

        Ok(info)
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
        let _ = reqbufs(
            fd,
            ioctl::V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE,
            0,
            ioctl::V4L2_MEMORY_MMAP,
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

/// `VIDIOC_QBUF` the MMAP CAPTURE (raw) buffer to run the decode.
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

/// `VIDIOC_QBUF` the CAPTURE queue in DMABUF mode, importing `dmabuf_fd` as the
/// backing for every plane. `plane_offsets[p]` places plane `p` within the
/// (shared) buffer; `length` is the dmabuf size. Zero-copy: the hardware
/// decodes straight into the imported tensor.
fn qbuf_capture_dmabuf(
    fd: RawFd,
    dmabuf_fd: RawFd,
    num_planes: usize,
    plane_offsets: &[usize],
    length: usize,
) -> nix::Result<()> {
    let mut planes = [ioctl::v4l2_plane::default(); ioctl::VIDEO_MAX_PLANES];
    for (p, plane) in planes.iter_mut().take(num_planes).enumerate() {
        plane.set_fd(dmabuf_fd);
        plane.length = length as u32;
        plane.data_offset = plane_offsets.get(p).copied().unwrap_or(0) as u32;
    }
    let mut b = ioctl::v4l2_buffer {
        type_: ioctl::V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE,
        memory: ioctl::V4L2_MEMORY_DMABUF,
        index: 0,
        length: num_planes as u32,
        ..Default::default()
    };
    b.set_planes(planes.as_mut_ptr());
    // SAFETY: valid buffer + plane array; `dmabuf_fd` outlives the call.
    unsafe { ioctl::vidioc_qbuf(fd, &mut b) }.map(|_| ())
}

/// `VIDIOC_DQBUF` the decoded CAPTURE buffer (`memory` = MMAP or DMABUF).
fn dqbuf_capture(fd: RawFd, num_planes: usize, memory: u32) -> nix::Result<()> {
    let mut planes = [ioctl::v4l2_plane::default(); ioctl::VIDEO_MAX_PLANES];
    let mut b = ioctl::v4l2_buffer {
        type_: ioctl::V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE,
        memory,
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
}
