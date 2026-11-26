// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! V4L2 stateful memory-to-memory video codecs.
//!
//! Devices are found by capability, not by node or driver name.

pub(crate) mod decoder;
pub(crate) mod encoder;
mod media;
pub(crate) mod mjpeg;

use std::path::PathBuf;

use edgefirst_v4l2::device::{self, Device};
use edgefirst_v4l2::queue::BufType;

use edgefirst_tensor::PixelFormat;

use crate::{Capability, Codec, CodecError, Direction, Engine, Transfer};

/// Adds the H.264 encoders and decoders present here to `out`, as the
/// encoder and decoder would find them.
pub(crate) fn capabilities(out: &mut Vec<Capability>) {
    let entry = |path: &std::path::Path, direction, formats| Capability {
        engine: Engine::V4l2,
        device: path.display().to_string(),
        codec: Codec::H264,
        direction,
        hardware: true,
        formats,
        transfer: Transfer::ZeroCopy,
    };
    for path in candidates(encoder::ENV_ENCODER) {
        let Some(device) = M2mDevice::open(&path) else {
            continue;
        };
        for codec in [Codec::H264, Codec::Mjpeg] {
            let coded = encoder::coded_fourccs(codec);
            if coded.iter().any(|&f| device.lists(device.capture, f)) {
                let formats = encoder::input_formats(&device);
                if !formats.is_empty() {
                    let mut c = entry(&path, Direction::Encode, formats);
                    c.codec = codec;
                    out.push(c);
                }
            }
        }
    }
    for path in candidates(decoder::ENV_DECODER) {
        if M2mDevice::open(&path).is_some_and(|d| decoder::decodes(&d)) {
            out.push(entry(&path, Direction::Decode, vec![PixelFormat::Nv12]));
        }
    }
    if let Some(device) = crate::jpeg::v4l2_device() {
        out.push(Capability {
            device,
            codec: Codec::Mjpeg,
            formats: vec![PixelFormat::Nv12, PixelFormat::Grey],
            ..entry(std::path::Path::new(""), Direction::Decode, Vec::new())
        });
    }
}

/// Maps a V4L2 error to [`CodecError::Device`], keeping the errno.
pub(crate) fn device_error(e: edgefirst_v4l2::Error) -> CodecError {
    let source = match e.errno() {
        Some(errno) => std::io::Error::from_raw_os_error(errno as i32),
        None => std::io::Error::other(e.to_string()),
    };
    CodecError::Device {
        op: e.op().to_owned(),
        source,
    }
}

/// An open memory-to-memory device and its queue types.
pub(crate) struct M2mDevice {
    pub dev: Device,
    /// Raw frames (application to device for an encoder).
    pub output: BufType,
    /// Compressed frames (device to application for an encoder).
    pub capture: BufType,
}

impl M2mDevice {
    /// Opens `path` if it is a streaming memory-to-memory device.
    pub fn open(path: &std::path::Path) -> Option<Self> {
        Self::try_open(path).ok().flatten()
    }

    /// Opens `path` if it is a streaming memory-to-memory device: `Ok(None)`
    /// when it is not one or cannot be opened, and
    /// [`CodecError::Device`] when its media controller describes it as a
    /// codec but it is busy or out of resources for another user.
    pub fn try_open(path: &std::path::Path) -> Result<Option<Self>, CodecError> {
        let dev = match Device::open(path) {
            Ok(dev) => dev,
            Err(e) if is_busy(&e) && media::is_codec_node(path) => {
                let errno = e.errno().map_or(0, |n| n as i32);
                return Err(CodecError::Device {
                    op: format!("open {}", path.display()),
                    source: std::io::Error::from_raw_os_error(errno),
                });
            }
            Err(_) => return Ok(None),
        };
        Ok(Self::from_device(dev))
    }

    fn from_device(dev: Device) -> Option<Self> {
        let caps = dev.capabilities();
        if !caps.is_m2m() || !caps.has_streaming() {
            return None;
        }
        let output = caps.output_buf_type()?;
        let capture = caps.capture_buf_type()?;
        Some(Self {
            dev,
            output,
            capture,
        })
    }

    /// Whether the queues use the multi-planar API.
    pub fn multiplanar(&self) -> bool {
        self.output.is_multiplanar()
    }

    /// Whether `buf_type` lists `fourcc`.
    pub fn lists(&self, buf_type: BufType, fourcc: u32) -> bool {
        self.dev
            .formats(buf_type)
            .is_ok_and(|f| f.iter().any(|f| f.fourcc == fourcc))
    }
}

/// The device nodes to try: the node in `env` when set, otherwise every
/// streaming memory-to-memory node, and every codec node that is busy or
/// out of resources, in numeric order.
pub(crate) fn candidates(env: &str) -> Vec<PathBuf> {
    if let Some(path) = std::env::var_os(env) {
        return vec![PathBuf::from(path)];
    }
    device::enumerate()
        .unwrap_or_default()
        .into_iter()
        .filter(|node| match &node.capabilities {
            Ok(c) => c.is_m2m() && c.has_streaming(),
            Err(e) => is_busy(e) && media::is_codec_node(&node.path),
        })
        .map(|node| node.path)
        .collect()
}

/// Whether a failed open means the node is busy or out of resources.
fn is_busy(e: &edgefirst_v4l2::Error) -> bool {
    e.errno().is_some_and(|errno| {
        matches!(
            std::io::Error::from_raw_os_error(errno as i32).kind(),
            std::io::ErrorKind::OutOfMemory
                | std::io::ErrorKind::ResourceBusy
                | std::io::ErrorKind::WouldBlock
        )
    })
}
