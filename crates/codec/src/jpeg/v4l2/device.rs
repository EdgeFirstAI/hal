// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! Vendor-neutral V4L2 JPEG-decoder discovery.
//!
//! The probe is purely capability-based — no device-node, driver-name, or
//! output-format is hardcoded. Any Linux device that exposes a JPEG decoder
//! through the standard V4L2 mem2mem API (i.MX `mxc-jpeg`, Rockchip Hantro,
//! Chips&Media coda, Allwinner Cedrus, …) is discovered the same way.

use std::fs::{File, OpenOptions};
use std::os::fd::AsRawFd;
use std::path::{Path, PathBuf};

use super::ioctl;

/// Environment variable forcing the CPU decoder (skip all V4L2 probing).
const ENV_DISABLE: &str = "EDGEFIRST_DISABLE_V4L2";
/// Environment variable pinning a specific device node (skips enumeration).
const ENV_DEVICE: &str = "EDGEFIRST_CODEC_V4L2_DEVICE";

/// Which V4L2 streaming API a discovered node speaks.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ApiVariant {
    /// Multi-planar API (`*_MPLANE` buffer types, `m.planes[]`).
    MultiPlanar,
    /// Single-planar API (flat `v4l2_buffer` / `v4l2_pix_format`).
    SinglePlanar,
}

/// A discovered, capability-verified V4L2 JPEG decoder device.
///
/// Owns the open file descriptor for the device node; dropping it closes the
/// node and releases the M2M context.
pub struct ProbedDevice {
    /// Open device-node handle (owns the fd; `O_RDWR | O_CLOEXEC`).
    pub file: File,
    /// Streaming API variant this node uses.
    pub api: ApiVariant,
    /// Device path, retained for logging/diagnostics.
    pub path: PathBuf,
}

impl ProbedDevice {
    pub fn fd(&self) -> std::os::fd::RawFd {
        self.file.as_raw_fd()
    }
}

/// Probe for an accessible V4L2 JPEG decoder.
///
/// Returns `Ok(Some(dev))` for the first node that passes the capability
/// checks, `Ok(None)` when none is available (or probing is disabled), and is
/// otherwise infallible — discovery failure is never an error, it just means
/// the CPU decoder is used.
pub fn probe() -> Option<ProbedDevice> {
    if crate::jpeg::env_flag(ENV_DISABLE) {
        log::debug!("v4l2 jpeg decode disabled via {ENV_DISABLE}");
        return None;
    }

    for path in candidate_nodes() {
        match probe_node(&path) {
            Some(dev) => {
                log::info!(
                    "v4l2 jpeg decoder discovered at {} ({:?})",
                    dev.path.display(),
                    dev.api
                );
                return Some(dev);
            }
            None => continue,
        }
    }
    log::debug!("no v4l2 jpeg decoder found; using cpu decoder");
    None
}

/// The ordered list of device nodes to try: an explicit override if set,
/// otherwise every `/dev/video*` sorted numerically.
fn candidate_nodes() -> Vec<PathBuf> {
    if let Some(dev) = std::env::var_os(ENV_DEVICE) {
        return vec![PathBuf::from(dev)];
    }

    let mut nodes: Vec<(u32, PathBuf)> = Vec::new();
    let Ok(entries) = std::fs::read_dir("/dev") else {
        return Vec::new();
    };
    for entry in entries.flatten() {
        let name = entry.file_name();
        let Some(name) = name.to_str() else { continue };
        let Some(num) = name
            .strip_prefix("video")
            .and_then(|n| n.parse::<u32>().ok())
        else {
            continue;
        };
        nodes.push((num, entry.path()));
    }
    nodes.sort_by_key(|(num, _)| *num);
    nodes.into_iter().map(|(_, path)| path).collect()
}

/// Open and capability-check a single node. Returns `Some` only if it is a
/// streaming M2M device whose coded (OUTPUT) queue advertises JPEG/MJPEG.
fn probe_node(path: &Path) -> Option<ProbedDevice> {
    let file = OpenOptions::new().read(true).write(true).open(path).ok()?;
    let fd = file.as_raw_fd();

    // VIDIOC_QUERYCAP — must stream and be a mem2mem device.
    let mut cap = ioctl::v4l2_capability::default();
    // SAFETY: `cap` is a valid, correctly-sized v4l2_capability; the kernel
    // only writes into it.
    unsafe { ioctl::vidioc_querycap(fd, &mut cap) }.ok()?;
    let caps = cap.effective_caps();

    if caps & ioctl::V4L2_CAP_STREAMING == 0 {
        return None;
    }
    let api = if caps & ioctl::V4L2_CAP_VIDEO_M2M_MPLANE != 0 {
        ApiVariant::MultiPlanar
    } else if caps & ioctl::V4L2_CAP_VIDEO_M2M != 0 {
        ApiVariant::SinglePlanar
    } else {
        return None;
    };

    // VIDIOC_ENUM_FMT on the coded (OUTPUT) queue — the real "is this a JPEG
    // decoder?" test. Driver name is irrelevant.
    if !output_queue_has_jpeg(fd, api) {
        return None;
    }

    Some(ProbedDevice {
        file,
        api,
        path: path.to_path_buf(),
    })
}

/// Enumerate the OUTPUT-queue coded formats and report whether JPEG (or MJPEG)
/// is offered.
fn output_queue_has_jpeg(fd: std::os::fd::RawFd, api: ApiVariant) -> bool {
    let buf_type = match api {
        ApiVariant::MultiPlanar => ioctl::V4L2_BUF_TYPE_VIDEO_OUTPUT_MPLANE,
        ApiVariant::SinglePlanar => ioctl::V4L2_BUF_TYPE_VIDEO_OUTPUT,
    };

    for index in 0..64u32 {
        let mut desc = ioctl::v4l2_fmtdesc {
            index,
            type_: buf_type,
            ..Default::default()
        };
        // SAFETY: `desc` is a valid v4l2_fmtdesc with index/type set; the
        // kernel fills the remaining fields and returns EINVAL past the last
        // entry, which we treat as end-of-list.
        if unsafe { ioctl::vidioc_enum_fmt(fd, &mut desc) }.is_err() {
            break;
        }
        if desc.pixelformat == ioctl::V4L2_PIX_FMT_JPEG
            || desc.pixelformat == ioctl::V4L2_PIX_FMT_MJPEG
        {
            return true;
        }
    }
    false
}
