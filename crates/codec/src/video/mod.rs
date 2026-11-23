// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! Hardware video encoding into, and decoding of, H.264 elementary streams.
//!
//! [`VideoEncoder`] compresses image tensors with the platform's video
//! encoder, and [`VideoDecoder`] turns access units back into image tensors.
//! Both run on a platform [`Engine`]: [`Engine::Auto`] picks the platform's
//! own, and [`capabilities`](crate::capabilities) lists what each engine here
//! can do. There is no CPU video codec in HAL: a codec the platform cannot
//! run fails with [`CodecError::Unsupported`].
//!
//! On Linux both are V4L2 stateful memory-to-memory devices, discovered by
//! capability. Source tensors in DMA-BUF memory are imported without a copy,
//! and decoded pictures are returned in the DMA-BUF buffers the device wrote;
//! only the compressed bitstream is copied.
//!
//! On Windows both are Media Foundation transforms (feature
//! `media-foundation`): a hardware encoder where the GPU driver registers
//! one, otherwise Microsoft's software encoder, and Microsoft's decoder.
//! Pixels stay on the GPU: a hardware encoder reads native NV12 D3D11 tensors
//! in place, and the decoder (DXVA) copies each picture, GPU to GPU, into a
//! pool of D3D11 tensors.
//!
//! On macOS and iOS both are VideoToolbox's (feature `videotoolbox`): a
//! hardware encoder where the system has one, otherwise Apple's software
//! encoder. A CoreVideo source of the configured size may be read in place;
//! every other source is copied into a pixel buffer from the encoder's own
//! pool. Decoded pictures are VideoToolbox's own IOSurface pixel buffers, lent
//! without a copy as multiplane NV12 tensors.
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
//!
//! Decoding takes one access unit at a time; [`access_units`] splits a byte
//! stream into them. Each call returns everything the decoder has ready.
//!
//! ```rust,no_run
//! use edgefirst_codec::video::{access_units, DecodeEvent, DecoderConfig, VideoDecoder};
//!
//! let stream = std::fs::read("clip.h264")?;
//! let mut decoder = VideoDecoder::new(DecoderConfig::h264())?;
//! let mut show = |event| match event {
//!     DecodeEvent::Frame(frame) => println!("{} {:?}", frame.pts(), frame.tensor().shape()),
//!     DecodeEvent::Reconfigured(info) => println!("{}x{}", info.width, info.height),
//! };
//! for (pts, au) in access_units(&stream).enumerate() {
//!     decoder.decode(au, pts as u64)?.into_iter().for_each(&mut show);
//! }
//! decoder.flush()?.into_iter().for_each(&mut show);
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```

#[cfg(any(
    all(target_os = "linux", feature = "v4l2"),
    all(windows, feature = "media-foundation"),
    all(any(target_os = "macos", target_os = "ios"), feature = "videotoolbox")
))]
pub(crate) mod copy;
#[cfg_attr(
    not(any(
        all(target_os = "linux", feature = "v4l2"),
        all(windows, feature = "media-foundation"),
        all(any(target_os = "macos", target_os = "ios"), feature = "videotoolbox")
    )),
    allow(dead_code)
)]
pub(crate) mod h264;
#[cfg_attr(
    not(all(any(target_os = "macos", target_os = "ios"), feature = "videotoolbox")),
    allow(dead_code)
)]
pub(crate) mod h264_syntax;
#[cfg(all(windows, feature = "media-foundation"))]
pub(crate) mod mf;
#[cfg(all(target_os = "linux", feature = "v4l2"))]
pub(crate) mod v4l2;
#[cfg(all(any(target_os = "macos", target_os = "ios"), feature = "videotoolbox"))]
pub(crate) mod vt;

use std::sync::{Arc, Mutex};

use edgefirst_tensor::{Colorimetry, DType, PixelFormat, Region, TensorDyn};

pub use h264::{access_units, AccessUnits};

use crate::engine::{compiled, engines_to_try, unsupported};
use crate::{Codec, CodecError, Direction, Engine, Result, Transfer};

/// The video formats of [`Codec`].
#[deprecated(note = "use edgefirst_codec::Codec, which names the video and still-image formats")]
pub type VideoCodec = Codec;

/// The engines this build has for video, in the order [`Engine::Auto`] tries
/// them.
fn video_engines() -> Vec<Engine> {
    [Engine::V4l2, Engine::MediaFoundation, Engine::VideoToolbox]
        .into_iter()
        .filter(|&e| compiled(e))
        .collect()
}

/// Whether `engine` runs `direction` of `codec` in this build.
fn runs(engine: Engine, codec: Codec, direction: Direction) -> bool {
    compiled(engine)
        && matches!(
            (engine, codec, direction),
            (
                Engine::V4l2 | Engine::MediaFoundation | Engine::VideoToolbox,
                Codec::H264,
                _
            ) | (Engine::V4l2, Codec::Mjpeg, _)
                | (Engine::MediaFoundation, Codec::Mjpeg, Direction::Decode)
        )
}

/// The engine that opens a `direction` stream of `codec` for `requested`:
/// the first engine to try that runs the codec.
///
/// # Errors
///
/// [`CodecError::NoDevice`] when this platform has no video engine, or
/// `requested` names one it does not have; [`CodecError::Unsupported`] when
/// no engine to try runs the codec.
fn pick(requested: Engine, codec: Codec, direction: Direction) -> Result<Engine> {
    let able = video_engines();
    if requested == Engine::Auto && able.is_empty() {
        return Err(CodecError::NoDevice(format!(
            "no video {direction} engine on this platform or in this build"
        )));
    }
    engines_to_try(requested, &able)?
        .into_iter()
        .find(|&e| runs(e, codec, direction))
        .ok_or_else(|| unsupported(codec, direction, requested))
}

/// Bits per pixel [`Bitrate::Auto`] targets: about 10 Mbps at 1080p30.
const AUTO_BITS_PER_PIXEL: f64 = 0.16;
/// Lowest and highest target [`Bitrate::Auto`] chooses, in bits per second.
const AUTO_BITRATE_RANGE: (f64, f64) = (1e6, 1e8);

/// Target bitrate of an encoder. Either way the encoder uses constant-bitrate
/// rate control where the device offers it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Bitrate {
    /// A target chosen from the frame size and rate: 0.16 bits per pixel,
    /// within 1 to 100 Mbps. See [`EncoderConfig::target_bitrate`].
    #[default]
    Auto,
    /// A constant target in bits per second.
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

impl H264Level {
    /// Every level, lowest first.
    const ALL: [Self; 20] = [
        Self::L1_0,
        Self::L1b,
        Self::L1_1,
        Self::L1_2,
        Self::L1_3,
        Self::L2_0,
        Self::L2_1,
        Self::L2_2,
        Self::L3_0,
        Self::L3_1,
        Self::L3_2,
        Self::L4_0,
        Self::L4_1,
        Self::L4_2,
        Self::L5_0,
        Self::L5_1,
        Self::L5_2,
        Self::L6_0,
        Self::L6_1,
        Self::L6_2,
    ];

    /// Table A-1 limits: frame size in macroblocks, macroblocks per second,
    /// and `MaxBR`, which [`Self::lowest_for`] scales by the profile's
    /// `cpbBrVclFactor` into bits per second.
    const fn limits(self) -> (u64, u64, u64) {
        match self {
            Self::L1_0 => (99, 1_485, 64),
            Self::L1b => (99, 1_485, 128),
            Self::L1_1 => (396, 3_000, 192),
            Self::L1_2 => (396, 6_000, 384),
            Self::L1_3 => (396, 11_880, 768),
            Self::L2_0 => (396, 11_880, 2_000),
            Self::L2_1 => (792, 19_800, 4_000),
            Self::L2_2 => (1_620, 20_250, 4_000),
            Self::L3_0 => (1_620, 40_500, 10_000),
            Self::L3_1 => (3_600, 108_000, 14_000),
            Self::L3_2 => (5_120, 216_000, 20_000),
            Self::L4_0 => (8_192, 245_760, 20_000),
            Self::L4_1 => (8_192, 245_760, 50_000),
            Self::L4_2 => (8_704, 522_240, 50_000),
            Self::L5_0 => (22_080, 589_824, 135_000),
            Self::L5_1 => (36_864, 983_040, 240_000),
            Self::L5_2 => (36_864, 2_073_600, 240_000),
            Self::L6_0 => (139_264, 4_177_920, 240_000),
            Self::L6_1 => (139_264, 8_355_840, 480_000),
            Self::L6_2 => (139_264, 16_711_680, 800_000),
        }
    }

    /// The lowest level whose frame size, frame dimension, macroblock rate
    /// and bitrate limits cover a `width`x`height` stream at `frame_rate` and
    /// `bps` in `profile`, or `None` when even level 6.2 does not.
    pub fn lowest_for(
        profile: H264Profile,
        width: u32,
        height: u32,
        frame_rate: f64,
        bps: u32,
    ) -> Option<Self> {
        let (width_mbs, height_mbs) = (
            u64::from(width.div_ceil(16)),
            u64::from(height.div_ceil(16)),
        );
        let mbs = width_mbs * height_mbs;
        let rate = (mbs as f64 * frame_rate).ceil();
        // cpbBrVclFactor of Table A-2: High allows 1.25 times the bitrate.
        let factor = match profile {
            H264Profile::High => 1_250,
            _ => 1_000,
        };
        Self::ALL.into_iter().find(|level| {
            let (frame, mb_rate, max_br) = level.limits();
            // A.3.1: neither dimension may exceed sqrt(8 * MaxFS) macroblocks.
            let side = (8 * frame).isqrt();
            mbs <= frame
                && width_mbs <= side
                && height_mbs <= side
                && rate <= mb_rate as f64
                && u64::from(bps) <= max_br * factor
        })
    }
}

/// Opens `engine`'s encoder for `config`.
fn open_encoder(engine: Engine, config: EncoderConfig) -> Result<Box<dyn EncoderBackend>> {
    match engine {
        #[cfg(all(target_os = "linux", feature = "v4l2"))]
        Engine::V4l2 => Ok(Box::new(v4l2::encoder::V4l2Encoder::open(config)?)),
        #[cfg(all(windows, feature = "media-foundation"))]
        Engine::MediaFoundation => Ok(Box::new(mf::encoder::MfEncoder::open(config)?)),
        #[cfg(all(any(target_os = "macos", target_os = "ios"), feature = "videotoolbox"))]
        Engine::VideoToolbox => Ok(Box::new(vt::encoder::VtEncoder::open(config)?)),
        other => Err(unsupported(config.codec, Direction::Encode, other)),
    }
}

/// Opens `engine`'s decoder for `config`.
fn open_decoder(engine: Engine, config: DecoderConfig) -> Result<Box<dyn DecoderBackend>> {
    match engine {
        #[cfg(all(target_os = "linux", feature = "v4l2"))]
        Engine::V4l2 if config.codec == Codec::Mjpeg => {
            Ok(Box::new(v4l2::mjpeg::V4l2MjpegDecoder::open(config)?))
        }
        #[cfg(all(target_os = "linux", feature = "v4l2"))]
        Engine::V4l2 => Ok(Box::new(v4l2::decoder::V4l2Decoder::open(config)?)),
        #[cfg(all(windows, feature = "media-foundation"))]
        Engine::MediaFoundation => Ok(Box::new(mf::decoder::MfDecoder::open(config)?)),
        #[cfg(all(any(target_os = "macos", target_os = "ios"), feature = "videotoolbox"))]
        Engine::VideoToolbox => Ok(Box::new(vt::decoder::VtDecoder::open(config)?)),
        other => Err(unsupported(config.codec, Direction::Decode, other)),
    }
}

/// Configuration of a [`VideoEncoder`].
///
/// Start from [`EncoderConfig::new`] or [`EncoderConfig::h264`] and set the
/// fields to change.
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq)]
pub struct EncoderConfig {
    /// Output codec.
    pub codec: Codec,
    /// The engine that encodes; [`Engine::Auto`] by default.
    pub engine: Engine,
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
    /// Emit the SPS and PPS with every key frame.
    pub repeat_headers: bool,
    /// Profile. `None` selects High where the device offers it, and keeps
    /// the device default where it does not. See
    /// [`EncoderConfig::target_profile`].
    pub profile: Option<H264Profile>,
    /// Level. `None` selects the lowest level that covers the stream where
    /// the device offers it, and keeps the device default where it does not.
    /// See [`EncoderConfig::target_level`].
    pub level: Option<H264Level>,
    /// JPEG quality of MJPEG pictures, 1 (smallest) to 100 (best). `None`
    /// uses [`DEFAULT_JPEG_QUALITY`](crate::DEFAULT_JPEG_QUALITY).
    pub quality: Option<u8>,
}

impl EncoderConfig {
    /// A configuration for `codec` on [`Engine::Auto`] with an automatic
    /// bitrate, profile and level, the device's default GOP, and headers
    /// repeated on every key frame.
    pub fn new(
        codec: Codec,
        width: u32,
        height: u32,
        input_format: PixelFormat,
        frame_rate: f64,
    ) -> Self {
        Self {
            codec,
            engine: Engine::Auto,
            width,
            height,
            input_format,
            frame_rate,
            bitrate: Bitrate::Auto,
            gop: None,
            repeat_headers: true,
            profile: None,
            level: None,
            quality: None,
        }
    }

    /// An H.264 configuration; see [`EncoderConfig::new`].
    pub fn h264(width: u32, height: u32, input_format: PixelFormat, frame_rate: f64) -> Self {
        Self::new(Codec::H264, width, height, input_format, frame_rate)
    }

    /// Sets [`engine`](Self::engine).
    pub fn with_engine(mut self, engine: Engine) -> Self {
        self.engine = engine;
        self
    }

    /// The target bitrate in bits per second: the [`Bitrate::Bps`] value, or
    /// for [`Bitrate::Auto`] one derived from the frame size and rate.
    pub fn target_bitrate(&self) -> u32 {
        match self.bitrate {
            Bitrate::Bps(bps) => bps,
            Bitrate::Auto => {
                let bps = f64::from(self.width)
                    * f64::from(self.height)
                    * self.frame_rate
                    * AUTO_BITS_PER_PIXEL;
                let (lo, hi) = AUTO_BITRATE_RANGE;
                bps.clamp(lo, hi) as u32
            }
        }
    }

    /// The profile to request: [`EncoderConfig::profile`], or High.
    pub fn target_profile(&self) -> H264Profile {
        self.profile.unwrap_or(H264Profile::High)
    }

    /// The level to request: [`EncoderConfig::level`], or the lowest that
    /// covers the size, frame rate and [target bitrate](Self::target_bitrate)
    /// in the [target profile](Self::target_profile). `None` when no level
    /// covers them.
    pub fn target_level(&self) -> Option<H264Level> {
        self.level.or_else(|| {
            H264Level::lowest_for(
                self.target_profile(),
                self.width,
                self.height,
                self.frame_rate,
                self.target_bitrate(),
            )
        })
    }

    /// The JPEG quality of MJPEG pictures: [`quality`](Self::quality), or
    /// [`DEFAULT_JPEG_QUALITY`](crate::DEFAULT_JPEG_QUALITY).
    pub fn jpeg_quality(&self) -> u8 {
        self.quality.unwrap_or(crate::DEFAULT_JPEG_QUALITY)
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
        if self.quality.is_some_and(|q| !(1..=100).contains(&q)) {
            return invalid(format!(
                "JPEG quality {} is not within 1 to 100",
                self.quality.unwrap_or_default()
            ));
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

/// Checks that `src` can be encoded as configured and returns the origin of
/// the region to encode.
#[cfg_attr(
    not(any(
        all(windows, feature = "media-foundation"),
        all(any(target_os = "macos", target_os = "ios"), feature = "videotoolbox")
    )),
    allow(dead_code)
)]
pub(crate) fn check_source(
    config: &EncoderConfig,
    src: &TensorDyn,
    opts: &FrameOptions,
) -> Result<(usize, usize)> {
    let invalid = |what: String| Err(CodecError::InvalidInput(what));
    if src.format() != Some(config.input_format) {
        return invalid(format!(
            "source format {:?} is not the configured {}",
            src.format(),
            config.input_format
        ));
    }
    if src.dtype() != DType::U8 {
        return invalid(format!(
            "source elements are {:?}; {} is 8-bit",
            src.dtype(),
            config.input_format
        ));
    }
    let (sw, sh) = match (src.width(), src.height()) {
        (Some(w), Some(h)) => (w, h),
        _ => return invalid("source is not an image".into()),
    };
    if src.view_origin().is_some() && config.input_format == PixelFormat::Nv12 {
        return invalid(format!(
            "a view of a {} image cannot be encoded; crop with FrameOptions::crop",
            config.input_format
        ));
    }
    let (w, h) = (config.width as usize, config.height as usize);
    let Some(crop) = opts.crop else {
        if (sw, sh) != (w, h) {
            return invalid(format!("source is {sw}x{sh}, the encoder {w}x{h}"));
        }
        return Ok((0, 0));
    };
    if (crop.width, crop.height) != (w, h) {
        return invalid(format!(
            "crop is {}x{}, the encoder {w}x{h}",
            crop.width, crop.height
        ));
    }
    if crop.x + crop.width > sw || crop.y + crop.height > sh {
        return invalid(format!(
            "crop {}x{}+{}+{} is outside the {sw}x{sh} source",
            crop.width, crop.height, crop.x, crop.y
        ));
    }
    let (hsub, vsub) = chroma_subsampling(config.input_format);
    if !crop.x.is_multiple_of(hsub as usize) || !crop.y.is_multiple_of(vsub as usize) {
        return invalid(format!(
            "crop origin {},{} is not on a {} chroma sample",
            crop.x, crop.y, config.input_format
        ));
    }
    Ok((crop.x, crop.y))
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

/// What runs an open [`VideoEncoder`], and how its sources reach it.
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EncoderInfo {
    /// The engine.
    pub engine: Engine,
    /// The device within the engine: a device node, or a transform's name.
    pub device: String,
    /// Whether the encoder is dedicated hardware rather than software the
    /// platform provides.
    pub hardware: bool,
    /// How a whole source of the configured size, in the memory the engine
    /// prefers (DMA-BUF; a D3D11 texture on Windows), reaches the encoder.
    /// Crops and host-memory sources may take a copy where this says none.
    /// A Media Foundation hardware encoder decides on its first frame
    /// whether it can read sources in place, and reports
    /// [`Transfer::GpuCopy`] until then. A V4L2 encoder reports how the
    /// latest source reached the device: [`Transfer::CpuCopy`] when it was
    /// copied into the device's layout, [`Transfer::ZeroCopy`] before the
    /// first frame and when the device read it in place, as it reads a
    /// source from [`VideoEncoder::create_source`].
    /// A VideoToolbox encoder reports how the last frame reached it,
    /// [`Transfer::CpuCopy`] before the first; [`Transfer::ZeroCopy`] means
    /// the last frame was read in place under the rule
    /// [`VideoEncoder::encode`] states.
    pub transfer: Transfer,
}

/// What a video encoder backend implements.
pub(crate) trait EncoderBackend: Send {
    fn encode(&mut self, src: &TensorDyn, opts: &FrameOptions) -> Result<Option<EncodedFrame>>;
    fn flush(&mut self) -> Result<Vec<EncodedFrame>>;
    fn device(&self) -> String;
    fn info(&self) -> EncoderInfo;
    fn create_source(&self) -> Result<TensorDyn>;
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
    /// Opens an encoder for `config`, on the first engine to try (see
    /// [`EncoderConfig::engine`]) that runs the codec.
    ///
    /// On Linux the first V4L2 memory-to-memory device that produces the
    /// codec and accepts the input format is used. Setting
    /// `EDGEFIRST_CODEC_V4L2_ENCODER` to a device node uses that node only.
    ///
    /// On Windows the first Media Foundation encoder transform that takes the
    /// input format (NV12 or YUYV) and the configuration is used, hardware
    /// transforms first. Setting `EDGEFIRST_CODEC_MF_ENCODER` keeps only the
    /// transforms whose name contains its value, or only the `hardware` or
    /// `software` ones.
    ///
    /// On macOS and iOS a VideoToolbox H.264 encoder is used, hardware where
    /// the system has one.
    ///
    /// # Errors
    ///
    /// [`CodecError::InvalidConfig`] when `config` is invalid or the device
    /// rejects it, [`CodecError::UnsupportedFormat`] when no device accepts
    /// the input format, [`CodecError::Unsupported`] when the engine (or, for
    /// [`Engine::Auto`], every engine here) does not encode the codec,
    /// [`CodecError::Device`] when no V4L2 encoder opens and a node is busy
    /// or out of resources, and [`CodecError::NoDevice`] when there is no
    /// encoder.
    pub fn new(config: EncoderConfig) -> Result<Self> {
        config.validate()?;
        let engine = pick(config.engine, config.codec, Direction::Encode)?;
        Ok(Self {
            backend: open_encoder(engine, config)?,
        })
    }

    /// Encodes `src`.
    ///
    /// `src` must have the configured input format and, without a crop, the
    /// configured size.
    ///
    /// On Linux and Windows, a source the device can read in place is read
    /// there; [`create_source`](Self::create_source) allocates one. Any
    /// other source, of any row stride and in any memory the CPU can read,
    /// is copied into a buffer the device can read first. Either way the
    /// call returns once `src` has been read, so the caller may reuse it
    /// afterwards.
    ///
    /// On macOS and iOS, every source of one stream must have the same row
    /// stride. A source whose IOSurface is a CoreVideo `420v`, `420f`,
    /// `yuvs` or `yuvf` surface of the configured size, with the row stride of
    /// each surface plane and encoded without a crop, may be read in place;
    /// [`EncoderInfo::transfer`] reports the route. Its IOSurface may stay in use
    /// (`IOSurfaceIsInUse` is true) after `encode()` returns, until its frame
    /// leaves the encoder (its access unit is returned, it is dropped, or
    /// `flush()` returns). CoreVideo pixel buffer pools do not hand out a
    /// surface while it is in use. Any other pool or writer must not reuse or
    /// write a surface while `IOSurfaceIsInUse` is true. Other sources are
    /// copied, and the caller may reuse them once `encode()` returns.
    ///
    /// Returns the next finished access unit, which may belong to an earlier
    /// frame, or `None` when none is ready yet.
    ///
    /// # Errors
    ///
    /// [`CodecError::InvalidInput`] when `src` does not match the
    /// configuration or the stream's earlier sources,
    /// [`CodecError::Device`] when the device fails, and
    /// [`CodecError::Timeout`] when the device does not take the frame in
    /// time.
    ///
    /// On macOS and iOS, at most 8 sources read in place are in use at once;
    /// another fails at once with [`CodecError::Timeout`] and is not encoded.
    /// A source tagged with a colour value that has no CoreVideo equivalent
    /// on the running platform (on iOS, `ColorTransfer::Linear`) fails
    /// with [`CodecError::InvalidInput`] naming the axis and value.
    ///
    /// On macOS and iOS, [`CodecError::Timeout`] means the frame was not
    /// taken because the encoder has too many frames waiting; the caller may
    /// submit it again later.
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

    /// The engine and device that encode, and how sources reach them.
    pub fn info(&self) -> EncoderInfo {
        self.backend.info()
    }

    /// Allocates a source image of the configured size and input format in
    /// the memory the encoder reads: one it reads in place, without a copy,
    /// where it can.
    ///
    /// On V4L2 the image is laid out as the device reads it: its row stride
    /// and, for NV12, the offset of its chroma plane can be larger than the
    /// picture needs, and the chroma plane can be a buffer of its own. On
    /// Windows it is a GPU image of the configured size for an encoder that
    /// reads D3D11 textures, and a host-memory image, which the encoder
    /// copies into its input sample, for one that does not. On macOS and
    /// iOS it is a CoreVideo IOSurface: `420v` for NV12 and `yuvs` for
    /// YUYV, or `420f` and `yuvf` once the stream's first source was full
    /// range. The encoder reads it in place under the rule
    /// [`encode`](Self::encode) states.
    ///
    /// # Errors
    ///
    /// [`CodecError::Tensor`] when the image cannot be allocated,
    /// [`CodecError::NoDevice`] when there is no GPU buffer allocator, and
    /// [`CodecError::Device`] when the device does not report its layout or,
    /// on macOS and iOS, CoreVideo cannot create the surface.
    pub fn create_source(&self) -> Result<TensorDyn> {
        self.backend.create_source()
    }
}

/// Where a [`VideoDecoder`] writes decoded pictures.
#[derive(Debug)]
pub enum OutputPool {
    /// The decoder allocates DMA-BUF frames, and allocates them again when
    /// the stream changes size.
    ///
    /// On macOS and iOS the frames are VideoToolbox's own pixel buffers,
    /// lent as multiplane NV12 tensors. A lent [`DecodedFrame`] keeps its
    /// pixel buffer out of VideoToolbox's pool until it drops.
    Allocate {
        /// Frames the caller may hold at once. The device's own minimum
        /// (the reference pictures it keeps) and one buffer for it to
        /// decode into are added to this. A V4L2 device grants a bounded
        /// number of buffers in all; a count it cannot cover makes
        /// [`VideoDecoder::decode`] fail with
        /// [`CodecError::InvalidConfig`] when the stream's size becomes
        /// known.
        count: usize,
    },
    /// Pictures are decoded into these DMA-BUF tensors, one buffer each.
    /// Each tensor must be a different buffer.
    ///
    /// On Windows, macOS and iOS pictures are copied into the pool (on the
    /// GPU where the device decodes there, otherwise by the CPU), and each
    /// tensor must be an NV12 image of exactly the stream's visible size;
    /// such a pool cannot follow a change of size. On macOS and iOS an image
    /// is one buffer or separate luma and chroma planes, and
    /// [`VideoDecoder::flush`] lends the pool every waiting picture or none.
    /// The rest of this description is about devices that decode in place.
    ///
    /// There must be at least one more than the device keeps for reference
    /// pictures and no more than it takes in all, and each must be large
    /// enough for the stream's coded picture at the row stride the device
    /// asks for; otherwise [`VideoDecoder::decode`] fails with
    /// [`CodecError::InvalidConfig`] when the stream's size becomes known. A
    /// tensor's own shape and format are not used: the decoded frames
    /// describe the device's layout over the same memory. Across a change of
    /// stream format the same tensors are used again, each once the frames
    /// still showing it have dropped.
    Provided(Vec<TensorDyn>),
}

impl Default for OutputPool {
    fn default() -> Self {
        Self::Allocate { count: 4 }
    }
}

/// Configuration of a [`VideoDecoder`].
#[non_exhaustive]
#[derive(Debug)]
pub struct DecoderConfig {
    /// Input codec.
    pub codec: Codec,
    /// The engine that decodes; [`Engine::Auto`] by default.
    pub engine: Engine,
    /// Where pictures are decoded to.
    pub output: OutputPool,
}

impl DecoderConfig {
    /// A configuration for `codec` on [`Engine::Auto`] with a
    /// decoder-allocated pool.
    pub fn new(codec: Codec) -> Self {
        Self {
            codec,
            engine: Engine::Auto,
            output: OutputPool::default(),
        }
    }

    /// An H.264 configuration; see [`DecoderConfig::new`].
    pub fn h264() -> Self {
        Self::new(Codec::H264)
    }

    /// Sets [`engine`](Self::engine).
    pub fn with_engine(mut self, engine: Engine) -> Self {
        self.engine = engine;
        self
    }

    /// Sets [`output`](Self::output).
    pub fn with_output(mut self, output: OutputPool) -> Self {
        self.output = output;
        self
    }
}

/// Geometry and format of the decoded pictures, known once the decoder has
/// parsed the stream's headers.
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq)]
pub struct StreamInfo {
    /// Width of the visible picture in pixels.
    pub width: usize,
    /// Height of the visible picture in pixels.
    pub height: usize,
    /// The visible picture within the coded one.
    pub crop: Region,
    /// Width of the coded picture, a whole number of macroblocks.
    pub coded_width: usize,
    /// Height of the coded picture, a whole number of macroblocks.
    pub coded_height: usize,
    /// Pixel format of the decoded frames.
    pub format: PixelFormat,
    /// Colorimetry the device reports, when it reports one.
    pub colorimetry: Option<Colorimetry>,
    /// Buffers in the output pool.
    pub buffers: usize,
    /// Whose memory the pictures are decoded into.
    pub memory: OutputMemory,
    /// The engine that decodes.
    pub engine: Engine,
    /// How each picture reaches its pool tensor: written in place
    /// ([`Transfer::ZeroCopy`], V4L2), lent in the decoder's own pixel
    /// buffers ([`Transfer::ZeroCopy`], VideoToolbox), or copied out of the
    /// decoder's own surfaces on the GPU ([`Transfer::GpuCopy`], Media
    /// Foundation with D3D11) or through CPU memory ([`Transfer::CpuCopy`]).
    pub transfer: Transfer,
}

/// Whose memory a decoder writes its pictures to. Either way the frames are
/// DMA-BUF tensors and reach the GPU without a copy.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OutputMemory {
    /// The pool's DMA-BUFs (allocated by the decoder or provided by the
    /// caller), imported by the device, or on Windows written by copying
    /// each picture out of the decoder, on the GPU where it decodes there.
    Imported,
    /// The device's own buffers, exported as DMA-BUFs, when it does not
    /// import the pool's. CPU reads of these may be uncached and slow. On
    /// macOS and iOS, VideoToolbox's own IOSurface pixel buffers.
    Exported,
}

/// What [`VideoDecoder::decode`] and [`VideoDecoder::flush`] produce, in
/// order.
#[derive(Debug)]
pub enum DecodeEvent {
    /// A decoded picture, in presentation order.
    Frame(DecodedFrame),
    /// The stream's size or format changed, or became known. Frames after
    /// this have the new [`StreamInfo`]; frames already returned keep their
    /// memory.
    Reconfigured(StreamInfo),
}

/// Buffers that callers have dropped, for the decoder to give back to the
/// device.
#[derive(Debug, Default)]
pub(crate) struct Returns(Mutex<Vec<(u64, u32)>>);

impl Returns {
    /// Takes the buffer indexes returned for pool `generation`, dropping
    /// those of earlier pools.
    #[cfg_attr(
        not(any(
            all(target_os = "linux", feature = "v4l2"),
            all(windows, feature = "media-foundation"),
            all(any(target_os = "macos", target_os = "ios"), feature = "videotoolbox")
        )),
        allow(dead_code)
    )]
    pub(crate) fn take(&self, generation: u64) -> Vec<u32> {
        let mut returned = self.0.lock().unwrap_or_else(|e| e.into_inner());
        returned
            .drain(..)
            .filter(|&(g, _)| g == generation)
            .map(|(_, index)| index)
            .collect()
    }
}

/// A decoded picture, lent from the decoder's pool.
///
/// The picture's buffer goes back to the device when the frame is dropped,
/// so the tensor is only reachable through the frame: hold the frame as long
/// as the picture is needed. To share one picture between consumers, share
/// the frame (`Arc<DecodedFrame>`); the buffer returns when the last one
/// drops. The pool is bounded, so frames held too long stop the decoder (see
/// [`CodecError::PoolExhausted`]).
#[derive(Debug)]
pub struct DecodedFrame {
    tensor: Arc<TensorDyn>,
    pts: u64,
    corrupted: bool,
    /// Pool generation and buffer index, given back on drop.
    slot: (u64, u32),
    returns: Arc<Returns>,
}

impl DecodedFrame {
    #[cfg_attr(
        not(any(
            all(target_os = "linux", feature = "v4l2"),
            all(windows, feature = "media-foundation"),
            all(any(target_os = "macos", target_os = "ios"), feature = "videotoolbox")
        )),
        allow(dead_code)
    )]
    pub(crate) fn new(
        tensor: Arc<TensorDyn>,
        pts: u64,
        corrupted: bool,
        slot: (u64, u32),
        returns: Arc<Returns>,
    ) -> Self {
        Self {
            tensor,
            pts,
            corrupted,
            slot,
            returns,
        }
    }

    /// The picture: an image tensor of the visible size in DMA-BUF memory
    /// (a D3D11 texture on Windows, or host memory where the process has no
    /// D3D11 device). On macOS and iOS it is a multiplane NV12 tensor over
    /// one of VideoToolbox's IOSurfaces with [`OutputPool::Allocate`], and
    /// one of the pool's tensors with [`OutputPool::Provided`].
    pub fn tensor(&self) -> &TensorDyn {
        &self.tensor
    }

    /// The `pts` passed with the access unit of this picture.
    pub fn pts(&self) -> u64 {
        self.pts
    }

    /// The device reported an error decoding this picture, so it may be
    /// partly wrong (a missing reference, a damaged slice).
    pub fn corrupted(&self) -> bool {
        self.corrupted
    }
}

impl Drop for DecodedFrame {
    fn drop(&mut self) {
        self.returns
            .0
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .push(self.slot);
    }
}

/// What a video decoder backend implements.
pub(crate) trait DecoderBackend: Send {
    fn decode(&mut self, access_unit: &[u8], pts: u64) -> Result<Vec<DecodeEvent>>;
    fn flush(&mut self) -> Result<Vec<DecodeEvent>>;
    fn stream_info(&self) -> Option<&StreamInfo>;
    fn device(&self) -> String;
    fn engine(&self) -> Engine;
}

/// A hardware video decoder.
///
/// Feed access units in decode order with [`decode`](Self::decode), which
/// returns every picture and format change that is ready, and collect the
/// rest with [`flush`](Self::flush) at the end of the stream.
pub struct VideoDecoder {
    backend: Box<dyn DecoderBackend>,
}

impl std::fmt::Debug for VideoDecoder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("VideoDecoder")
            .field("device", &self.backend.device())
            .field("stream", &self.backend.stream_info())
            .finish()
    }
}

impl VideoDecoder {
    /// Opens a decoder for `config`, on the first engine to try (see
    /// [`DecoderConfig::engine`]) that runs the codec.
    ///
    /// On Linux the first V4L2 memory-to-memory device that decodes the
    /// codec to NV12 is used. Setting `EDGEFIRST_CODEC_V4L2_DECODER` to a
    /// device node uses that node only.
    ///
    /// On Windows Microsoft's Media Foundation H.264 decoder is used.
    /// Setting `EDGEFIRST_CODEC_MF_DECODER` keeps only the transforms whose
    /// name contains its value, or only the `hardware` or `software` ones.
    ///
    /// On macOS and iOS VideoToolbox's H.264 decoder is used.
    ///
    /// # Errors
    ///
    /// [`CodecError::InvalidConfig`] when a provided pool is empty, holds one
    /// buffer twice or is not in DMA-BUF memory, [`CodecError::Unsupported`]
    /// when the engine (or, for [`Engine::Auto`], every engine here) does not
    /// decode the codec, [`CodecError::Device`] when no V4L2 decoder opens
    /// and a node is busy or out of resources, and [`CodecError::NoDevice`]
    /// when there is no decoder.
    pub fn new(config: DecoderConfig) -> Result<Self> {
        if let OutputPool::Provided(pool) = &config.output {
            if pool.is_empty() {
                return Err(CodecError::InvalidConfig(
                    "the provided pool is empty".into(),
                ));
            }
            // Reject a pool that holds one buffer twice.
            let mut seen = std::collections::HashMap::with_capacity(pool.len());
            for (i, t) in pool.iter().enumerate() {
                if let Some(first) = seen.insert(t.buffer_identity().id(), i) {
                    return Err(CodecError::InvalidConfig(format!(
                        "provided tensors {first} and {i} are the same buffer; each pool buffer must be distinct"
                    )));
                }
            }
            if let Some(t) = pool
                .iter()
                .find(|t| t.memory() != edgefirst_tensor::TensorMemory::DmaBuf)
            {
                return Err(CodecError::InvalidConfig(format!(
                    "the provided pool has a tensor in {:?} memory; the decoder writes DMA-BUF tensors",
                    t.memory()
                )));
            }
        }
        let engine = pick(config.engine, config.codec, Direction::Decode)?;
        Ok(Self {
            backend: open_decoder(engine, config)?,
        })
    }

    /// Decodes one access unit (see [`access_units`]), whose picture is
    /// tagged with `pts`.
    ///
    /// Returns, in order, every event ready by the time the device has taken
    /// the access unit; it may be empty, and frames may belong to earlier
    /// access units. The first access unit yields
    /// [`DecodeEvent::Reconfigured`] once the headers are parsed.
    ///
    /// # Errors
    ///
    /// [`CodecError::PoolExhausted`] when the caller holds the frames in
    /// every output buffer, so the device cannot decode further; drop frames
    /// and decode the same access unit again. [`CodecError::Timeout`] when
    /// the device makes no progress otherwise. [`CodecError::InvalidInput`]
    /// when the access unit is empty or larger than the device's bitstream
    /// buffers. Events that became ready before an error are returned by
    /// the next call.
    pub fn decode(&mut self, access_unit: &[u8], pts: u64) -> Result<Vec<DecodeEvent>> {
        self.backend.decode(access_unit, pts)
    }

    /// Drains the decoder: returns every event not yet returned, in order.
    ///
    /// The decoder accepts new access units afterwards, starting a new
    /// sequence: the next access unit must be decodable on its own, an IDR
    /// picture with its SPS and PPS, as at the start of a stream or after a
    /// seek.
    ///
    /// # Errors
    ///
    /// [`CodecError::PoolExhausted`] when the caller holds the frames in
    /// output buffers the drain needs; drop frames and flush again.
    /// [`CodecError::InvalidConfig`] on Windows, macOS and iOS when an
    /// [`OutputPool::Provided`] pool holds fewer tensors than the pictures
    /// the drain releases. [`CodecError::Timeout`] when the device does not
    /// finish draining.
    ///
    /// On macOS and iOS a flush into an [`OutputPool::Provided`] pool lends
    /// every waiting picture or none. It returns them all when they fit the
    /// free tensors. Otherwise it lends nothing and the pictures stay
    /// queued: it returns [`CodecError::PoolExhausted`] while the caller
    /// holds frames, and [`CodecError::InvalidConfig`] when the pool holds
    /// fewer tensors than the pictures waiting. Frames an earlier call left
    /// after an error come first; when they fill the tensors the pictures
    /// need, the flush returns only those frames, and the next flush
    /// continues the drain.
    pub fn flush(&mut self) -> Result<Vec<DecodeEvent>> {
        self.backend.flush()
    }

    /// The current stream's geometry, once known.
    pub fn stream_info(&self) -> Option<&StreamInfo> {
        self.backend.stream_info()
    }

    /// The device that decodes, for logs (a device node on Linux).
    pub fn device(&self) -> String {
        self.backend.device()
    }

    /// The engine that decodes.
    pub fn engine(&self) -> Engine {
        self.backend.engine()
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
        assert_eq!((c.codec, c.engine), (Codec::H264, Engine::Auto));
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
    fn auto_bitrate_scales_with_size_and_rate_within_its_range() {
        let mut c = nv12_1080p();
        assert_eq!(c.target_bitrate(), 9_953_280);
        c.frame_rate = 60.0;
        assert_eq!(c.target_bitrate(), 19_906_560);
        let c = EncoderConfig::h264(320, 240, PixelFormat::Nv12, 15.0);
        assert_eq!(c.target_bitrate(), 1_000_000);
        let c = EncoderConfig::h264(7680, 4320, PixelFormat::Nv12, 120.0);
        assert_eq!(c.target_bitrate(), 100_000_000);
        let mut c = nv12_1080p();
        c.bitrate = Bitrate::Bps(4_000_000);
        assert_eq!(c.target_bitrate(), 4_000_000);
    }

    #[test]
    fn levels_cover_size_rate_and_bitrate() {
        use H264Level as L;
        use H264Profile::{High, Main};
        assert_eq!(
            L::lowest_for(High, 1920, 1080, 30.0, 10_000_000),
            Some(L::L4_0)
        );
        assert_eq!(
            L::lowest_for(High, 1920, 1080, 60.0, 20_000_000),
            Some(L::L4_2)
        );
        assert_eq!(
            L::lowest_for(High, 1920, 1080, 30.0, 40_000_000),
            Some(L::L4_1)
        );
        assert_eq!(
            L::lowest_for(Main, 1920, 1080, 30.0, 24_000_000),
            Some(L::L4_1)
        );
        assert_eq!(
            L::lowest_for(High, 640, 480, 30.0, 2_000_000),
            Some(L::L3_0)
        );
        assert_eq!(
            L::lowest_for(High, 3840, 2160, 30.0, 40_000_000),
            Some(L::L5_1)
        );
        assert_eq!(
            L::lowest_for(High, 7680, 4320, 120.0, 100_000_000),
            Some(L::L6_2)
        );
        assert_eq!(L::lowest_for(High, 7680, 4320, 240.0, 100_000_000), None);
    }

    #[test]
    fn levels_cover_each_frame_dimension() {
        use H264Level as L;
        use H264Profile::High;
        assert_eq!(
            L::lowest_for(High, 4096, 256, 30.0, 5_000_000),
            Some(L::L4_0)
        );
        assert_eq!(
            L::lowest_for(High, 256, 4096, 30.0, 5_000_000),
            Some(L::L4_0)
        );
        assert_eq!(
            L::lowest_for(High, 4112, 256, 30.0, 5_000_000),
            Some(L::L4_2)
        );
    }

    #[test]
    fn unset_profile_and_level_resolve_to_high_at_the_lowest_level() {
        let mut c = nv12_1080p();
        assert_eq!(c.target_profile(), H264Profile::High);
        assert_eq!(c.target_level(), Some(H264Level::L4_0));
        c.frame_rate = 60.0;
        assert_eq!(c.target_level(), Some(H264Level::L4_2));
        c.profile = Some(H264Profile::Main);
        c.level = Some(H264Level::L5_1);
        assert_eq!(c.target_profile(), H264Profile::Main);
        assert_eq!(c.target_level(), Some(H264Level::L5_1));
    }

    #[test]
    fn frames_can_be_shared_between_threads() {
        fn shareable<T: Send + Sync>() {}
        shareable::<DecodedFrame>();
        shareable::<Arc<DecodedFrame>>();
    }

    #[test]
    fn a_dropped_frame_returns_its_buffer_to_its_own_pool() {
        let returns = Arc::new(Returns::default());
        let tensor =
            |n| Arc::new(TensorDyn::new(&[n], edgefirst_tensor::DType::U8, None, None).unwrap());
        let old = DecodedFrame::new(tensor(16), 7, false, (1, 3), Arc::clone(&returns));
        let current = DecodedFrame::new(tensor(16), 8, true, (2, 5), Arc::clone(&returns));
        assert_eq!((current.pts(), current.corrupted()), (8, true));
        assert!(
            returns.take(2).is_empty(),
            "nothing returned while frames are held"
        );
        drop(old);
        drop(current);
        assert_eq!(
            returns.take(2),
            vec![5],
            "a buffer of an earlier pool is not returned"
        );
        assert!(returns.take(1).is_empty(), "take drains every generation");
    }

    #[test]
    fn only_platform_engines_run_video() {
        for direction in [Direction::Encode, Direction::Decode] {
            assert!(!runs(Engine::Cpu, Codec::H264, direction));
            assert!(!runs(Engine::NvJpeg, Codec::H264, direction));
            assert!(!runs(Engine::Cpu, Codec::Mjpeg, direction));
        }
        for codec in [Codec::Hevc, Codec::Vp8, Codec::Vp9, Codec::Av1, Codec::Jpeg] {
            for engine in [Engine::V4l2, Engine::MediaFoundation, Engine::VideoToolbox] {
                assert!(!runs(engine, codec, Direction::Decode), "{engine} {codec}");
            }
        }
    }

    #[test]
    fn an_engine_that_cannot_run_the_codec_is_unsupported() {
        let refused = |engine, codec| {
            pick(engine, codec, Direction::Encode)
                .map(|_| ())
                .unwrap_err()
        };
        assert!(matches!(
            refused(Engine::Cpu, Codec::H264),
            CodecError::Unsupported(crate::UnsupportedFeature::Codec {
                codec: Codec::H264,
                direction: Direction::Encode,
                engine: Engine::Cpu,
            })
        ));
        if cfg!(all(
            any(target_os = "macos", target_os = "ios"),
            feature = "videotoolbox"
        )) {
            assert_eq!(
                pick(Engine::VideoToolbox, Codec::H264, Direction::Encode).unwrap(),
                Engine::VideoToolbox
            );
        } else {
            assert!(matches!(
                refused(Engine::VideoToolbox, Codec::H264),
                CodecError::NoDevice(_)
            ));
        }
        for direction in [Direction::Encode, Direction::Decode] {
            assert!(!runs(Engine::VideoToolbox, Codec::Mjpeg, direction));
        }
        if !video_engines().is_empty() {
            assert!(matches!(
                refused(Engine::Auto, Codec::Hevc),
                CodecError::Unsupported(crate::UnsupportedFeature::Codec {
                    engine: Engine::Auto,
                    ..
                })
            ));
        }
    }

    #[test]
    fn levels_are_ordered() {
        assert!(H264Level::L1b < H264Level::L1_1);
        assert!(H264Level::L4_0 < H264Level::L5_1);
    }
}
