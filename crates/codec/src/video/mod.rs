// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! Hardware video encoding into, and decoding of, H.264 elementary streams.
//!
//! [`VideoEncoder`] compresses image tensors with the platform's video
//! encoder, and [`VideoDecoder`] turns access units back into image tensors.
//! On Linux both are V4L2 stateful memory-to-memory devices (i.MX 8M Plus
//! `vsi_v4l2`, i.MX 95 Wave6, …), discovered by capability. Source tensors in
//! DMA-BUF memory are imported without a copy, and decoded pictures are
//! returned in the DMA-BUF buffers the device wrote; only the compressed
//! bitstream is copied.
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

#[cfg_attr(not(all(target_os = "linux", feature = "v4l2")), allow(dead_code))] // backend use only
pub(crate) mod h264;
#[cfg(all(target_os = "linux", feature = "v4l2"))]
mod v4l2;

use std::sync::{Arc, Mutex};

use edgefirst_tensor::{Colorimetry, PixelFormat, Region, TensorDyn};

pub use h264::{access_units, AccessUnits};

use crate::{CodecError, Result};

/// A compressed video format.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum VideoCodec {
    /// ITU-T H.264 / MPEG-4 AVC, as an Annex B byte stream.
    H264,
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
    /// Profile. `None` selects High where the device offers it, and keeps
    /// the device default where it does not. See
    /// [`EncoderConfig::target_profile`].
    pub profile: Option<H264Profile>,
    /// Level. `None` selects the lowest level that covers the stream where
    /// the device offers it, and keeps the device default where it does not.
    /// See [`EncoderConfig::target_level`].
    pub level: Option<H264Level>,
}

impl EncoderConfig {
    /// An H.264 configuration with an automatic bitrate, profile and level,
    /// the device's default GOP, and headers repeated on every key frame.
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

/// Where a [`VideoDecoder`] writes decoded pictures.
#[derive(Debug)]
pub enum OutputPool {
    /// The decoder allocates DMA-BUF frames, and allocates them again when
    /// the stream changes size.
    Allocate {
        /// Frames the caller may hold at once. The device's own minimum
        /// (the reference pictures it keeps) and one buffer for it to
        /// decode into are added to this.
        count: usize,
    },
    /// Pictures are decoded into these DMA-BUF tensors, one buffer each.
    /// There must be at least one more than the device keeps for reference
    /// pictures, and each must be large enough for the stream's coded
    /// picture at the row stride the device asks for; otherwise
    /// [`VideoDecoder::decode`] fails with [`CodecError::InvalidConfig`] when
    /// the stream's size becomes known. A tensor's own shape and format are
    /// not used: the decoded frames describe the device's layout over the
    /// same memory. Across a change of stream format the same tensors are
    /// used again, each once the frames still showing it have dropped.
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
    pub codec: VideoCodec,
    /// Where pictures are decoded to.
    pub output: OutputPool,
}

impl DecoderConfig {
    /// An H.264 configuration with a decoder-allocated pool.
    pub fn h264() -> Self {
        Self {
            codec: VideoCodec::H264,
            output: OutputPool::default(),
        }
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
}

/// Whose memory a decoder writes its pictures to. Either way the frames are
/// DMA-BUF tensors and reach the GPU without a copy.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OutputMemory {
    /// The pool's DMA-BUFs (allocated by the decoder or provided by the
    /// caller), imported by the device.
    Imported,
    /// The device's own buffers, exported as DMA-BUFs because it would not
    /// import the pool's. CPU reads of these may be uncached and slow.
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
    #[cfg_attr(not(all(target_os = "linux", feature = "v4l2")), allow(dead_code))] // backend use only
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
    #[cfg_attr(not(all(target_os = "linux", feature = "v4l2")), allow(dead_code))] // backend use only
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

    /// The picture: an image tensor of the visible size in DMA-BUF memory.
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
    /// Opens a decoder for `config`.
    ///
    /// On Linux the first V4L2 memory-to-memory device that decodes the
    /// codec to NV12 is used. Setting `EDGEFIRST_CODEC_V4L2_DECODER` to a
    /// device node uses that node only.
    ///
    /// # Errors
    ///
    /// [`CodecError::InvalidConfig`] when a provided pool is empty or not in
    /// DMA-BUF memory, and [`CodecError::NoDevice`] when there is no decoder.
    pub fn new(config: DecoderConfig) -> Result<Self> {
        if let OutputPool::Provided(pool) = &config.output {
            if pool.is_empty() {
                return Err(CodecError::InvalidConfig(
                    "the provided pool is empty".into(),
                ));
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
        #[cfg(all(target_os = "linux", feature = "v4l2"))]
        {
            let backend = v4l2::decoder::V4l2Decoder::open(config)?;
            Ok(Self {
                backend: Box::new(backend),
            })
        }
        #[cfg(not(all(target_os = "linux", feature = "v4l2")))]
        {
            let _ = config;
            Err(CodecError::NoDevice(
                "no video decoder backend in this build".into(),
            ))
        }
    }

    /// Decodes one access unit (see [`access_units`]), whose picture is
    /// tagged with `pts`.
    ///
    /// Returns, in order, every event ready by the time the device has taken
    /// the access unit; it may be empty, and frames may belong to earlier
    /// access units, since devices hold pictures back for reordering. The
    /// first access unit yields [`DecodeEvent::Reconfigured`] once the
    /// headers are parsed.
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
    /// every output buffer the drain needs; drop frames and flush again.
    /// [`CodecError::Timeout`] when the device does not finish draining.
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
        // Main has no 1.25 bitrate allowance: 24 Mbps exceeds level 4.0's 20.
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
        // 256 macroblocks wide fits level 3.2's frame size but not its
        // sqrt(8 * 5120) = 202 macroblock side; level 4.0 allows 256.
        assert_eq!(
            L::lowest_for(High, 4096, 256, 30.0, 5_000_000),
            Some(L::L4_0)
        );
        assert_eq!(
            L::lowest_for(High, 256, 4096, 30.0, 5_000_000),
            Some(L::L4_0)
        );
        // 257 macroblocks exceed level 4.1's side; 4.2 allows sqrt(8 * 8704) = 263.
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
    fn levels_are_ordered() {
        assert!(H264Level::L1b < H264Level::L1_1);
        assert!(H264Level::L4_0 < H264Level::L5_1);
    }
}
