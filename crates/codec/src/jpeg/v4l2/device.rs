// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! Vendor-neutral V4L2 JPEG-decoder discovery.
//!
//! The probe is purely capability-based — no device-node, driver-name, or
//! output-format is hardcoded. Any Linux device that exposes a JPEG decoder
//! through the standard V4L2 mem2mem API (i.MX `mxc-jpeg`, Rockchip Hantro,
//! Chips&Media coda, Allwinner Cedrus, …) is discovered the same way.

use std::path::{Path, PathBuf};

use edgefirst_v4l2::device::{self, Device};
use edgefirst_v4l2::queue::BufType;
use edgefirst_v4l2::uapi;

/// Environment variable forcing the CPU decoder (skip all V4L2 probing).
const ENV_DISABLE: &str = "EDGEFIRST_DISABLE_V4L2";
/// Environment variable pinning a specific device node (skips enumeration).
const ENV_DEVICE: &str = "EDGEFIRST_CODEC_V4L2_DEVICE";

/// Which V4L2 streaming API a discovered node speaks.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ApiVariant {
    /// `V4L2_CAP_VIDEO_M2M_MPLANE` — the `_MPLANE` buffer types.
    MultiPlanar,
    /// `V4L2_CAP_VIDEO_M2M` — the single-planar buffer types.
    SinglePlanar,
}

/// A discovered, capability-verified V4L2 JPEG decoder device.
///
/// Owns the open device node; dropping it (and every queue built from it)
/// closes the node and releases the M2M context.
pub struct ProbedDevice {
    pub dev: Device,
    pub api: ApiVariant,
}

impl ProbedDevice {
    /// The device node's path.
    pub fn path(&self) -> &Path {
        self.dev.path()
    }
}

/// Probe for an accessible V4L2 JPEG decoder.
///
/// Returns the first node that passes the capability checks, or `None` when
/// none is available (or probing is disabled). Discovery failure is never an
/// error, it just means the CPU decoder is used.
pub fn probe() -> Option<ProbedDevice> {
    if crate::jpeg::env_flag(ENV_DISABLE) {
        log::debug!("v4l2 jpeg decode disabled via {ENV_DISABLE}");
        return None;
    }

    for path in candidate_nodes() {
        if let Some(dev) = probe_node(&path) {
            log::info!(
                "v4l2 jpeg decoder discovered at {} ({:?})",
                dev.path().display(),
                dev.api
            );
            return Some(dev);
        }
    }
    log::debug!("no v4l2 jpeg decoder found; using cpu decoder");
    None
}

/// The ordered list of device nodes to try: an explicit override if set,
/// otherwise every streaming M2M node among `/dev/video*`, sorted
/// numerically.
fn candidate_nodes() -> Vec<PathBuf> {
    if let Some(dev) = std::env::var_os(ENV_DEVICE) {
        return vec![PathBuf::from(dev)];
    }
    device::enumerate()
        .unwrap_or_default()
        .into_iter()
        .filter(|node| {
            node.capabilities
                .as_ref()
                .is_ok_and(|c| c.is_m2m() && c.has_streaming())
        })
        .map(|node| node.path)
        .collect()
}

/// Open and capability-check a single node. Returns `Some` only if it is a
/// streaming M2M device whose coded (OUTPUT) queue advertises JPEG/MJPEG.
fn probe_node(path: &Path) -> Option<ProbedDevice> {
    let dev = Device::open(path).ok()?;
    let caps = dev.capabilities();
    if !caps.is_m2m() || !caps.has_streaming() {
        return None;
    }
    let (api, output) = match caps.output_buf_type()? {
        BufType::VideoOutputMplane => (ApiVariant::MultiPlanar, BufType::VideoOutputMplane),
        _ => (ApiVariant::SinglePlanar, BufType::VideoOutput),
    };
    let has_jpeg = dev
        .formats(output)
        .ok()?
        .iter()
        .any(|f| f.fourcc == uapi::V4L2_PIX_FMT_JPEG || f.fourcc == uapi::V4L2_PIX_FMT_MJPEG);
    has_jpeg.then_some(ProbedDevice { dev, api })
}
