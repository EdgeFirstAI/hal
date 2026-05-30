// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! Raw V4L2 UAPI definitions: `#[repr(C)]` structs, FourCC/capability
//! constants, and `nix` ioctl macro bindings.
//!
//! Everything here mirrors `linux/videodev2.h` byte-for-byte. The rest of the
//! `v4l2` module works exclusively in safe Rust against these definitions —
//! this file is the only place raw structs and ioctl numbers live, mirroring
//! how `crates/tensor/src/dmabuf.rs` localises its DRM/dma-buf ABI.
//!
//! Every struct carries a compile-time `size_of` assertion against its known
//! 64-bit (`x86_64`/`aarch64`) layout. The `nix` ioctl macros encode the
//! request number from `sizeof(struct)`, so a layout mistake would otherwise
//! surface only as a runtime `ENOTTY` or a partial copy — the assertions turn
//! it into a build failure instead.
//!
//! The `m` union in `v4l2_buffer`/`v4l2_plane` is modelled as a single 8-byte
//! `u64`: on 64-bit targets every union member (`offset:u32`,
//! `userptr:unsigned long`, `planes:*mut`, `fd:s32`) fits, matching the
//! kernel's size and alignment. This module therefore targets 64-bit Linux.
#![allow(non_camel_case_types)]
#![allow(dead_code)] // ABI table: some constants are defined for completeness.

use nix::{ioctl_read, ioctl_readwrite, ioctl_write_ptr};

/// Maximum planes per buffer (`VIDEO_MAX_PLANES`).
pub const VIDEO_MAX_PLANES: usize = 8;

/// Build a V4L2 FourCC (`v4l2_fourcc`): four ASCII bytes packed little-endian.
pub const fn fourcc(a: u8, b: u8, c: u8, d: u8) -> u32 {
    (a as u32) | ((b as u32) << 8) | ((c as u32) << 16) | ((d as u32) << 24)
}

/// Render a FourCC back to a 4-byte ASCII string for logging.
pub fn fourcc_str(v: u32) -> String {
    let bytes = [v as u8, (v >> 8) as u8, (v >> 16) as u8, (v >> 24) as u8];
    bytes
        .iter()
        .map(|&b| if b.is_ascii_graphic() { b as char } else { '?' })
        .collect()
}

// ---- Coded (compressed) pixel formats, OUTPUT queue ------------------------
/// `V4L2_PIX_FMT_JPEG` — JFIF JPEG bytestream.
pub const V4L2_PIX_FMT_JPEG: u32 = fourcc(b'J', b'P', b'E', b'G');
/// `V4L2_PIX_FMT_MJPEG` — Motion-JPEG bytestream.
pub const V4L2_PIX_FMT_MJPEG: u32 = fourcc(b'M', b'J', b'P', b'G');

// ---- Raw (decoded) pixel formats, CAPTURE queue ----------------------------
/// `V4L2_PIX_FMT_YUV24` — packed Y/Cb/Cr 8-8-8, 4:4:4, single plane ("YUV3").
pub const V4L2_PIX_FMT_YUV24: u32 = fourcc(b'Y', b'U', b'V', b'3');
/// `V4L2_PIX_FMT_GREY` — 8-bit luma only.
pub const V4L2_PIX_FMT_GREY: u32 = fourcc(b'G', b'R', b'E', b'Y');
/// `V4L2_PIX_FMT_NV12` — Y plane + interleaved CbCr, 4:2:0, single buffer.
pub const V4L2_PIX_FMT_NV12: u32 = fourcc(b'N', b'V', b'1', b'2');
/// `V4L2_PIX_FMT_NV12M` — Y plane + interleaved CbCr, 4:2:0, two buffers
/// (non-contiguous planes; "NM12").
pub const V4L2_PIX_FMT_NV12M: u32 = fourcc(b'N', b'M', b'1', b'2');
/// `V4L2_PIX_FMT_YUYV` — packed Y/Cb/Y/Cr, 4:2:2.
pub const V4L2_PIX_FMT_YUYV: u32 = fourcc(b'Y', b'U', b'Y', b'V');
/// `V4L2_PIX_FMT_UYVY` — packed Cb/Y/Cr/Y, 4:2:2.
pub const V4L2_PIX_FMT_UYVY: u32 = fourcc(b'U', b'Y', b'V', b'Y');
/// `V4L2_PIX_FMT_YVYU` — packed Y/Cr/Y/Cb, 4:2:2.
pub const V4L2_PIX_FMT_YVYU: u32 = fourcc(b'Y', b'V', b'Y', b'U');

// ---- Capability flags (`v4l2_capability.device_caps`) ----------------------
pub const V4L2_CAP_VIDEO_M2M: u32 = 0x0000_8000;
pub const V4L2_CAP_VIDEO_M2M_MPLANE: u32 = 0x0000_4000;
pub const V4L2_CAP_STREAMING: u32 = 0x0400_0000;
pub const V4L2_CAP_DEVICE_CAPS: u32 = 0x8000_0000;

// ---- Buffer types (`v4l2_buf_type`) ----------------------------------------
pub const V4L2_BUF_TYPE_VIDEO_CAPTURE: u32 = 1;
pub const V4L2_BUF_TYPE_VIDEO_OUTPUT: u32 = 2;
pub const V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE: u32 = 9;
pub const V4L2_BUF_TYPE_VIDEO_OUTPUT_MPLANE: u32 = 10;

// ---- Memory types (`v4l2_memory`) ------------------------------------------
pub const V4L2_MEMORY_MMAP: u32 = 1;
pub const V4L2_MEMORY_USERPTR: u32 = 2;
pub const V4L2_MEMORY_DMABUF: u32 = 4;

// ---- Field / colorimetry ----------------------------------------------------
pub const V4L2_FIELD_NONE: u32 = 1;
pub const V4L2_COLORSPACE_JPEG: u32 = 7;
pub const V4L2_COLORSPACE_SRGB: u32 = 8;
pub const V4L2_YCBCR_ENC_601: u8 = 1;
pub const V4L2_QUANTIZATION_DEFAULT: u8 = 0;
pub const V4L2_QUANTIZATION_FULL_RANGE: u8 = 1;

// ---- Events -----------------------------------------------------------------
pub const V4L2_EVENT_SOURCE_CHANGE: u32 = 5;

/// `struct v4l2_capability` — returned by `VIDIOC_QUERYCAP` (104 bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct v4l2_capability {
    pub driver: [u8; 16],
    pub card: [u8; 32],
    pub bus_info: [u8; 32],
    pub version: u32,
    pub capabilities: u32,
    pub device_caps: u32,
    pub reserved: [u32; 3],
}
const _: () = assert!(std::mem::size_of::<v4l2_capability>() == 104);

impl Default for v4l2_capability {
    fn default() -> Self {
        // SAFETY: POD of integers/byte arrays; all-zero is a valid initial
        // state and the kernel overwrites it on QUERYCAP.
        unsafe { std::mem::zeroed() }
    }
}

impl v4l2_capability {
    /// The effective capabilities for *this* device node (per-node
    /// `device_caps` when `V4L2_CAP_DEVICE_CAPS` is set, else driver-wide).
    pub fn effective_caps(&self) -> u32 {
        if self.capabilities & V4L2_CAP_DEVICE_CAPS != 0 {
            self.device_caps
        } else {
            self.capabilities
        }
    }
}

/// `struct v4l2_fmtdesc` — one entry from `VIDIOC_ENUM_FMT` (64 bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct v4l2_fmtdesc {
    pub index: u32,
    pub type_: u32,
    pub flags: u32,
    pub description: [u8; 32],
    pub pixelformat: u32,
    pub reserved: [u32; 4],
}
const _: () = assert!(std::mem::size_of::<v4l2_fmtdesc>() == 64);

impl Default for v4l2_fmtdesc {
    fn default() -> Self {
        // SAFETY: POD; all-zero is the documented ENUM_FMT request state.
        unsafe { std::mem::zeroed() }
    }
}

/// `struct v4l2_plane_pix_format` — per-plane geometry (20 bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct v4l2_plane_pix_format {
    pub sizeimage: u32,
    pub bytesperline: u32,
    pub reserved: [u16; 6],
}
const _: () = assert!(std::mem::size_of::<v4l2_plane_pix_format>() == 20);

/// `struct v4l2_pix_format` — single-planar format (48 bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct v4l2_pix_format {
    pub width: u32,
    pub height: u32,
    pub pixelformat: u32,
    pub field: u32,
    pub bytesperline: u32,
    pub sizeimage: u32,
    pub colorspace: u32,
    pub priv_: u32,
    pub flags: u32,
    pub ycbcr_enc: u32,
    pub quantization: u32,
    pub xfer_func: u32,
}
const _: () = assert!(std::mem::size_of::<v4l2_pix_format>() == 48);

/// `struct v4l2_pix_format_mplane` — multi-planar format (192 bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct v4l2_pix_format_mplane {
    pub width: u32,
    pub height: u32,
    pub pixelformat: u32,
    pub field: u32,
    pub colorspace: u32,
    pub plane_fmt: [v4l2_plane_pix_format; VIDEO_MAX_PLANES],
    pub num_planes: u8,
    pub flags: u8,
    pub ycbcr_enc: u8,
    pub quantization: u8,
    pub xfer_func: u8,
    pub reserved: [u8; 7],
}
const _: () = assert!(std::mem::size_of::<v4l2_pix_format_mplane>() == 192);

impl Default for v4l2_pix_format_mplane {
    fn default() -> Self {
        // SAFETY: POD of integers; all-zero is a valid initial state.
        unsafe { std::mem::zeroed() }
    }
}

/// `struct v4l2_format` (208 bytes). The kernel's format union includes
/// `v4l2_window`, whose pointer members give the union 8-byte alignment — so
/// `{u32 type; union}` is `4 + 4 pad + 200 = 208`, and the 200-byte `fmt`
/// payload starts at offset 8. `_pad` reproduces that padding so the struct
/// size matches the kernel's (the ioctl request number encodes `sizeof`, so a
/// 204-byte struct yields the wrong number and `ENOTTY`).
#[repr(C)]
#[derive(Clone, Copy)]
pub struct v4l2_format {
    pub type_: u32,
    /// Alignment padding (the kernel union is 8-aligned); always zero.
    pub _pad: u32,
    pub fmt: [u8; 200],
}
const _: () = assert!(std::mem::size_of::<v4l2_format>() == 208);

impl Default for v4l2_format {
    fn default() -> Self {
        Self {
            type_: 0,
            _pad: 0,
            fmt: [0u8; 200],
        }
    }
}

impl v4l2_format {
    /// Typed mutable view of the single-planar payload.
    ///
    /// # Safety
    /// The caller must ensure `type_` selects the single-planar variant.
    pub unsafe fn pix(&mut self) -> &mut v4l2_pix_format {
        &mut *(self.fmt.as_mut_ptr() as *mut v4l2_pix_format)
    }

    /// Typed mutable view of the multi-planar payload.
    ///
    /// # Safety
    /// The caller must ensure `type_` selects the multi-planar variant.
    pub unsafe fn pix_mp(&mut self) -> &mut v4l2_pix_format_mplane {
        &mut *(self.fmt.as_mut_ptr() as *mut v4l2_pix_format_mplane)
    }
}

/// `struct v4l2_requestbuffers` — `VIDIOC_REQBUFS` (20 bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct v4l2_requestbuffers {
    pub count: u32,
    pub type_: u32,
    pub memory: u32,
    pub capabilities: u32,
    pub flags: u8,
    pub reserved: [u8; 3],
}
const _: () = assert!(std::mem::size_of::<v4l2_requestbuffers>() == 20);

/// `struct v4l2_timecode` (16 bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct v4l2_timecode {
    pub type_: u32,
    pub flags: u32,
    pub frames: u8,
    pub seconds: u8,
    pub minutes: u8,
    pub hours: u8,
    pub userbits: [u8; 4],
}
const _: () = assert!(std::mem::size_of::<v4l2_timecode>() == 16);

/// `struct v4l2_plane` (64 bytes). `m` overlays the `{mem_offset:u32,
/// userptr:unsigned long, fd:s32}` union as a single 8-byte slot.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct v4l2_plane {
    pub bytesused: u32,
    pub length: u32,
    pub m: u64,
    pub data_offset: u32,
    pub reserved: [u32; 11],
}
const _: () = assert!(std::mem::size_of::<v4l2_plane>() == 64);

impl Default for v4l2_plane {
    fn default() -> Self {
        // SAFETY: POD; all-zero is the documented initial state.
        unsafe { std::mem::zeroed() }
    }
}

impl v4l2_plane {
    /// `m.mem_offset` — MMAP plane offset (set by QUERYBUF).
    pub fn mem_offset(&self) -> u32 {
        self.m as u32
    }
    /// Set `m.fd` — import a dmabuf into this plane (DMABUF memory).
    pub fn set_fd(&mut self, fd: i32) {
        self.m = fd as u32 as u64;
    }
}

/// `struct v4l2_buffer` (88 bytes). `m` overlays the `{offset:u32,
/// userptr:unsigned long, planes:*mut v4l2_plane, fd:s32}` union.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct v4l2_buffer {
    pub index: u32,
    pub type_: u32,
    pub bytesused: u32,
    pub flags: u32,
    pub field: u32,
    pub timestamp: libc::timeval,
    pub timecode: v4l2_timecode,
    pub sequence: u32,
    pub memory: u32,
    pub m: u64,
    pub length: u32,
    pub reserved2: u32,
    pub request_fd: i32,
}
const _: () = assert!(std::mem::size_of::<v4l2_buffer>() == 88);

impl Default for v4l2_buffer {
    fn default() -> Self {
        // SAFETY: POD (libc::timeval is integers); all-zero is the documented
        // initial state for a buffer request.
        unsafe { std::mem::zeroed() }
    }
}

impl v4l2_buffer {
    /// Set `m.offset` — single-planar MMAP offset.
    pub fn set_offset(&mut self, offset: u32) {
        self.m = offset as u64;
    }
    /// `m.offset` — single-planar MMAP offset (from QUERYBUF).
    pub fn offset(&self) -> u32 {
        self.m as u32
    }
    /// Set `m.fd` — single-planar dmabuf import.
    pub fn set_fd(&mut self, fd: i32) {
        self.m = fd as u32 as u64;
    }
    /// Set `m.planes` — multi-planar plane array pointer. The pointed-to array
    /// must outlive the ioctl call.
    pub fn set_planes(&mut self, planes: *mut v4l2_plane) {
        self.m = planes as usize as u64;
    }
}

/// `struct v4l2_event` — `VIDIOC_DQEVENT` (136 bytes). The kernel's `u` union
/// has `u64` members (e.g. `v4l2_event_ctrl.value64`), giving it 8-byte
/// alignment, so it starts at offset 8 — `_pad` reproduces that so `pending`
/// lands at the right offset. The 64-byte `u` payload is opaque to us; we only
/// read `type_`/`pending`.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct v4l2_event {
    pub type_: u32,
    /// Alignment padding (the kernel union is 8-aligned); always zero.
    pub _pad: u32,
    pub u: [u8; 64],
    pub pending: u32,
    pub sequence: u32,
    pub timestamp: libc::timespec,
    pub id: u32,
    pub reserved: [u32; 8],
}
const _: () = assert!(std::mem::size_of::<v4l2_event>() == 136);

impl Default for v4l2_event {
    fn default() -> Self {
        // SAFETY: POD (libc::timespec is integers); all-zero is valid.
        unsafe { std::mem::zeroed() }
    }
}

/// `struct v4l2_event_subscription` — `VIDIOC_SUBSCRIBE_EVENT` (32 bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct v4l2_event_subscription {
    pub type_: u32,
    pub id: u32,
    pub flags: u32,
    pub reserved: [u32; 5],
}
const _: () = assert!(std::mem::size_of::<v4l2_event_subscription>() == 32);

// ---- ioctl bindings --------------------------------------------------------
// nix encodes the request number from (dir, type, nr, sizeof(struct)) at
// compile time, so these MUST be paired with the exact UAPI struct above.

ioctl_read!(
    /// `VIDIOC_QUERYCAP` = `_IOR('V', 0, struct v4l2_capability)`.
    vidioc_querycap,
    b'V',
    0,
    v4l2_capability
);
ioctl_readwrite!(
    /// `VIDIOC_ENUM_FMT` = `_IOWR('V', 2, struct v4l2_fmtdesc)`.
    vidioc_enum_fmt,
    b'V',
    2,
    v4l2_fmtdesc
);
ioctl_readwrite!(
    /// `VIDIOC_G_FMT` = `_IOWR('V', 4, struct v4l2_format)`.
    vidioc_g_fmt,
    b'V',
    4,
    v4l2_format
);
ioctl_readwrite!(
    /// `VIDIOC_S_FMT` = `_IOWR('V', 5, struct v4l2_format)`.
    vidioc_s_fmt,
    b'V',
    5,
    v4l2_format
);
ioctl_readwrite!(
    /// `VIDIOC_REQBUFS` = `_IOWR('V', 8, struct v4l2_requestbuffers)`.
    vidioc_reqbufs,
    b'V',
    8,
    v4l2_requestbuffers
);
ioctl_readwrite!(
    /// `VIDIOC_QUERYBUF` = `_IOWR('V', 9, struct v4l2_buffer)`.
    vidioc_querybuf,
    b'V',
    9,
    v4l2_buffer
);
ioctl_readwrite!(
    /// `VIDIOC_QBUF` = `_IOWR('V', 15, struct v4l2_buffer)`.
    vidioc_qbuf,
    b'V',
    15,
    v4l2_buffer
);
ioctl_readwrite!(
    /// `VIDIOC_DQBUF` = `_IOWR('V', 17, struct v4l2_buffer)`.
    vidioc_dqbuf,
    b'V',
    17,
    v4l2_buffer
);
ioctl_write_ptr!(
    /// `VIDIOC_STREAMON` = `_IOW('V', 18, int)`.
    vidioc_streamon,
    b'V',
    18,
    std::os::raw::c_int
);
ioctl_write_ptr!(
    /// `VIDIOC_STREAMOFF` = `_IOW('V', 19, int)`.
    vidioc_streamoff,
    b'V',
    19,
    std::os::raw::c_int
);
ioctl_read!(
    /// `VIDIOC_DQEVENT` = `_IOR('V', 89, struct v4l2_event)`.
    vidioc_dqevent,
    b'V',
    89,
    v4l2_event
);
ioctl_write_ptr!(
    /// `VIDIOC_SUBSCRIBE_EVENT` = `_IOW('V', 90, struct v4l2_event_subscription)`.
    vidioc_subscribe_event,
    b'V',
    90,
    v4l2_event_subscription
);
