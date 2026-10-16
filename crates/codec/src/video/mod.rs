// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! Hardware video encoding into H.264 elementary streams.
//!
//! [`VideoEncoder`] compresses image tensors with the platform's video
//! encoder. On Linux that is any V4L2 stateful memory-to-memory encoder
//! (i.MX 8M Plus `vsi_v4l2`, i.MX 95 Wave6, …), discovered by capability.
//! Source tensors in DMA-BUF memory are imported without a copy; only the
//! compressed bitstream is copied out.
//!
//! ```rust,no_run
//! use edgefirst_codec::video::{EncoderConfig, FrameOptions, VideoEncoder};
//! use edgefirst_tensor::{CpuAccess, PixelFormat, TensorDyn, TensorMemory};
//!
//! let mut encoder = VideoEncoder::new(EncoderConfig::h264(1920, 1080, PixelFormat::Nv12, 30.0))?;
//! let frame = TensorDyn::image(1920, 1080, PixelFormat::Nv12, edgefirst_tensor::DType::U8,
//!                              Some(TensorMemory::DmaBuf), CpuAccess::Write)?;
//! let mut stream = Vec::new();
//! for pts in 0..30u64 {
//!     let opts = FrameOptions { pts, ..FrameOptions::default() };
//!     if let Some(au) = encoder.encode(&frame, &opts)? {
//!         stream.extend_from_slice(&au.data);
//!     }
//! }
//! for au in encoder.flush()? {
//!     stream.extend_from_slice(&au.data);
//! }
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```

pub(crate) mod h264;
#[cfg(all(target_os = "linux", feature = "v4l2"))]
mod v4l2;

use edgefirst_tensor::{PixelFormat, Region, TensorDyn};

use crate::{CodecError, Result};

/// A compressed video format.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum VideoCodec {
    /// ITU-T H.264 / MPEG-4 AVC, as an Annex B byte stream.
    H264,
}

/// Target bitrate of an encoder.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Bitrate {
    /// The device's default bitrate and rate control.
    #[default]
    Auto,
    /// A constant target in bits per second. Selects constant-bitrate rate
    /// control where the device offers a choice.
    Bps(u32),
}

/// H.264 profile (ITU-T H.264 Annex A).
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum H264Profile {
    /// Baseline.
    Baseline,
    /// Constrained Baseline.
    ConstrainedBaseline,
    /// Main.
    Main,
    /// High.
    High,
}

/// H.264 level (ITU-T H.264 Table A-1).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[allow(missing_docs)]
pub enum H264Level {
    L1_0,
    L1b,
    L1_1,
    L1_2,
    L1_3,
    L2_0,
    L2_1,
    L2_2,
    L3_0,
    L3_1,
    L3_2,
    L4_0,
    L4_1,
    L4_2,
    L5_0,
    L5_1,
    L5_2,
    L6_0,
    L6_1,
    L6_2,
}

/// Configuration of a [`VideoEncoder`].
///
/// Start from [`EncoderConfig::h264`] and set the fields to change.
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq)]
pub struct EncoderConfig {
    /// Output codec.
    pub codec: VideoCodec,
    /// Width of the encoded picture in pixels.
    pub width: u32,
    /// Height of the encoded picture in pixels.
    pub height: u32,
    /// Pixel format of the source tensors.
    pub input_format: PixelFormat,
    /// Nominal frame rate, for rate control.
    pub frame_rate: f64,
    /// Target bitrate.
    pub bitrate: Bitrate,
    /// Frames from one key frame to the next. `None` keeps the device
    /// default.
    pub gop: Option<u32>,
    /// Emit the SPS and PPS with every key frame, so a receiver can start
    /// decoding at any key frame.
    pub repeat_headers: bool,
    /// Profile. `None` keeps the device default.
    pub profile: Option<H264Profile>,
    /// Level. `None` keeps the device default.
    pub level: Option<H264Level>,
}

impl EncoderConfig {
    /// An H.264 configuration with the device's default bitrate, GOP,
    /// profile and level, and headers repeated on every key frame.
    pub fn h264(width: u32, height: u32, input_format: PixelFormat, frame_rate: f64) -> Self {
        Self {
            codec: VideoCodec::H264,
            width,
            height,
            input_format,
            frame_rate,
            bitrate: Bitrate::Auto,
            gop: None,
            repeat_headers: true,
            profile: None,
            level: None,
        }
    }

    /// Checks the values that do not depend on a device.
    pub fn validate(&self) -> Result<()> {
        let invalid = |what: String| Err(CodecError::InvalidConfig(what));
        if self.width == 0 || self.height == 0 {
            return invalid(format!("size {}x{} is empty", self.width, self.height));
        }
        let (hsub, vsub) = chroma_subsampling(self.input_format);
        if !self.width.is_multiple_of(hsub) || !self.height.is_multiple_of(vsub) {
            return invalid(format!(
                "size {}x{} is not a whole number of {} chroma blocks",
                self.width, self.height, self.input_format
            ));
        }
        if !self.frame_rate.is_finite() || self.frame_rate <= 0.0 {
            return invalid(format!("frame rate {} is not positive", self.frame_rate));
        }
        if self.bitrate == Bitrate::Bps(0) {
            return invalid("bitrate is 0".into());
        }
        if self.gop == Some(0) {
            return invalid("GOP is 0".into());
        }
        Ok(())
    }
}

/// Horizontal and vertical chroma subsampling factors of `format`.
pub(crate) fn chroma_subsampling(format: PixelFormat) -> (u32, u32) {
    match format {
        PixelFormat::Nv12 => (2, 2),
        PixelFormat::Nv16 | PixelFormat::Yuyv | PixelFormat::Vyuy => (2, 1),
        _ => (1, 1),
    }
}

/// Per-frame options for [`VideoEncoder::encode`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct FrameOptions {
    /// Presentation timestamp, returned unchanged on the matching
    /// [`EncodedFrame`]. Any unit; the encoder never interprets it.
    pub pts: u64,
    /// The part of the source to encode, exactly the configured size.
    /// `None` encodes the whole source, which must then be the configured
    /// size.
    pub crop: Option<Region>,
    /// Encode this frame as a key frame (an IDR picture).
    pub keyframe: bool,
}

/// One encoded access unit.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EncodedFrame {
    /// Annex B bytes of the access unit. Key frames start with the SPS and
    /// PPS when [`EncoderConfig::repeat_headers`] is set, and the first frame
    /// always does.
    pub data: Vec<u8>,
    /// The access unit is an IDR picture.
    pub keyframe: bool,
    /// The [`FrameOptions::pts`] of the source frame.
    pub pts: u64,
}

/// What a video encoder backend implements.
pub(crate) trait EncoderBackend: Send {
    fn encode(&mut self, src: &TensorDyn, opts: &FrameOptions) -> Result<Option<EncodedFrame>>;
    fn flush(&mut self) -> Result<Vec<EncodedFrame>>;
    fn device(&self) -> String;
}

/// A hardware video encoder.
///
/// Feed source frames with [`encode`](Self::encode), which may return the
/// access unit of an earlier frame, and collect the rest with
/// [`flush`](Self::flush) at the end of the stream.
pub struct VideoEncoder {
    backend: Box<dyn EncoderBackend>,
}

impl std::fmt::Debug for VideoEncoder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("VideoEncoder")
            .field("device", &self.backend.device())
            .finish()
    }
}

impl VideoEncoder {
    /// Opens an encoder for `config`.
    ///
    /// On Linux the first V4L2 memory-to-memory device that produces the
    /// codec and accepts the input format is used. Setting
    /// `EDGEFIRST_CODEC_V4L2_ENCODER` to a device node uses that node only.
    ///
    /// # Errors
    ///
    /// [`CodecError::InvalidConfig`] when `config` is invalid or the device
    /// rejects it, [`CodecError::UnsupportedFormat`] when no device accepts
    /// the input format, and [`CodecError::NoDevice`] when there is no
    /// encoder.
    pub fn new(config: EncoderConfig) -> Result<Self> {
        config.validate()?;
        #[cfg(all(target_os = "linux", feature = "v4l2"))]
        {
            let backend = v4l2::encoder::V4l2Encoder::open(config)?;
            Ok(Self {
                backend: Box::new(backend),
            })
        }
        #[cfg(not(all(target_os = "linux", feature = "v4l2")))]
        {
            let _ = config;
            Err(CodecError::NoDevice(
                "no video encoder backend in this build".into(),
            ))
        }
    }

    /// Encodes `src`.
    ///
    /// `src` must have the configured input format and, without a crop, the
    /// configured size. DMA-BUF sources are read in place: the call returns
    /// once the device has finished reading `src`, so the caller may reuse
    /// it afterwards. Every source of one stream must have the same row
    /// stride.
    ///
    /// Returns the next finished access unit, which may belong to an earlier
    /// frame, or `None` when none is ready yet.
    pub fn encode(&mut self, src: &TensorDyn, opts: &FrameOptions) -> Result<Option<EncodedFrame>> {
        self.backend.encode(src, opts)
    }

    /// Drains the encoder: returns every access unit not yet returned, in
    /// order. The encoder accepts new frames afterwards.
    pub fn flush(&mut self) -> Result<Vec<EncodedFrame>> {
        self.backend.flush()
    }

    /// The device that encodes, for logs (a device node on Linux).
    pub fn device(&self) -> String {
        self.backend.device()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn nv12_1080p() -> EncoderConfig {
        EncoderConfig::h264(1920, 1080, PixelFormat::Nv12, 30.0)
    }

    #[test]
    fn default_h264_config_is_valid() {
        let c = nv12_1080p();
        assert!(c.validate().is_ok());
        assert_eq!(c.codec, VideoCodec::H264);
        assert_eq!(c.bitrate, Bitrate::Auto);
        assert!(c.repeat_headers);
        assert!(c.gop.is_none() && c.profile.is_none() && c.level.is_none());
    }

    #[test]
    fn rejects_empty_and_odd_sizes() {
        let mut c = nv12_1080p();
        c.width = 0;
        assert!(matches!(c.validate(), Err(CodecError::InvalidConfig(_))));
        let mut c = nv12_1080p();
        c.height = 1081;
        assert!(matches!(c.validate(), Err(CodecError::InvalidConfig(_))));
        let mut c = EncoderConfig::h264(641, 480, PixelFormat::Yuyv, 30.0);
        assert!(matches!(c.validate(), Err(CodecError::InvalidConfig(_))));
        c.width = 640;
        c.height = 481;
        assert!(c.validate().is_ok(), "YUYV subsamples horizontally only");
    }

    #[test]
    fn rejects_bad_rate_bitrate_and_gop() {
        for fps in [0.0, -1.0, f64::NAN, f64::INFINITY] {
            let mut c = nv12_1080p();
            c.frame_rate = fps;
            assert!(c.validate().is_err(), "{fps}");
        }
        let mut c = nv12_1080p();
        c.bitrate = Bitrate::Bps(0);
        assert!(c.validate().is_err());
        let mut c = nv12_1080p();
        c.gop = Some(0);
        assert!(c.validate().is_err());
        c.gop = Some(1);
        c.bitrate = Bitrate::Bps(4_000_000);
        assert!(c.validate().is_ok());
    }

    #[test]
    fn levels_are_ordered() {
        assert!(H264Level::L1b < H264Level::L1_1);
        assert!(H264Level::L4_0 < H264Level::L5_1);
    }
}
