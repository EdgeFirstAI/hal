// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! Codecs, the engines that run them, and what each engine can do here.
//!
//! Every codec object takes an [`Engine`]. [`Engine::Auto`] lets HAL pick the
//! first engine that can do the request on this platform; any other value
//! uses that engine alone and fails rather than falling back. Video codecs
//! run only on platform engines: a codec the platform cannot run is
//! [`CodecError::Unsupported`], never a CPU encoder in HAL.
//!
//! [`capabilities`] lists what the engines present on this machine can do,
//! without opening a stream.

use std::fmt;
use std::str::FromStr;

use edgefirst_tensor::PixelFormat;

use crate::{CodecError, Result, UnsupportedFeature};

/// A compressed image or video format.
///
/// The enum names known formats; which ones work depends on the platform's
/// engines (see [`capabilities`]).
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Codec {
    /// JPEG still images.
    Jpeg,
    /// PNG still images.
    Png,
    /// ITU-T H.264 / MPEG-4 AVC, as an Annex B byte stream.
    H264,
    /// ITU-T H.265 / HEVC, as an Annex B byte stream.
    Hevc,
    /// Motion JPEG: a stream of JPEG pictures.
    Mjpeg,
    /// VP8.
    Vp8,
    /// VP9.
    Vp9,
    /// AV1.
    Av1,
}

impl Codec {
    /// Whether this is a stream format, for [`VideoEncoder`] and
    /// [`VideoDecoder`], rather than a still-image format.
    ///
    /// [`VideoEncoder`]: crate::video::VideoEncoder
    /// [`VideoDecoder`]: crate::video::VideoDecoder
    pub fn is_video(self) -> bool {
        !matches!(self, Self::Jpeg | Self::Png)
    }

    /// The format's usual name, for messages.
    pub fn name(self) -> &'static str {
        match self {
            Self::Jpeg => "JPEG",
            Self::Png => "PNG",
            Self::H264 => "H.264",
            Self::Hevc => "HEVC",
            Self::Mjpeg => "MJPEG",
            Self::Vp8 => "VP8",
            Self::Vp9 => "VP9",
            Self::Av1 => "AV1",
        }
    }
}

impl fmt::Display for Codec {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

/// Encoding or decoding.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Direction {
    /// Pixels to a compressed format.
    Encode,
    /// A compressed format to pixels.
    Decode,
}

impl fmt::Display for Direction {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Encode => "encoding",
            Self::Decode => "decoding",
        })
    }
}

/// What runs a codec.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum Engine {
    /// The first engine that can do the request, in the platform's order:
    /// on Linux V4L2 for video and nvJPEG (when enabled), V4L2, then the CPU
    /// for JPEG; on Windows Media Foundation for video. The environment
    /// variable [`ENGINE_ENV`] narrows and orders this choice.
    #[default]
    Auto,
    /// Linux V4L2 memory-to-memory devices.
    V4l2,
    /// Windows Media Foundation transforms.
    MediaFoundation,
    /// Apple VideoToolbox.
    VideoToolbox,
    /// NVIDIA nvJPEG, on Linux with CUDA.
    NvJpeg,
    /// HAL's own CPU code, for still images only.
    Cpu,
}

/// Environment variable naming, in order and separated by commas, the
/// engines [`Engine::Auto`] may use, for example `media-foundation` or
/// `v4l2,cpu`. Engines it leaves out are not used by `Auto`; codec objects
/// that name an engine explicitly ignore it.
pub const ENGINE_ENV: &str = "EDGEFIRST_CODEC_ENGINE";

impl Engine {
    /// The engine's name, as [`ENGINE_ENV`] and [`FromStr`] take it.
    pub fn name(self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::V4l2 => "v4l2",
            Self::MediaFoundation => "media-foundation",
            Self::VideoToolbox => "videotoolbox",
            Self::NvJpeg => "nvjpeg",
            Self::Cpu => "cpu",
        }
    }
}

impl fmt::Display for Engine {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

impl FromStr for Engine {
    type Err = CodecError;

    /// Parses an engine name, ignoring case; `mf` and `vt` are accepted for
    /// Media Foundation and VideoToolbox.
    fn from_str(s: &str) -> Result<Self> {
        Ok(match s.trim().to_ascii_lowercase().as_str() {
            "auto" => Self::Auto,
            "v4l2" => Self::V4l2,
            "media-foundation" | "mediafoundation" | "mf" => Self::MediaFoundation,
            "videotoolbox" | "vt" => Self::VideoToolbox,
            "nvjpeg" => Self::NvJpeg,
            "cpu" => Self::Cpu,
            other => {
                return Err(CodecError::InvalidConfig(format!(
                    "unknown codec engine {other:?}; expected auto, v4l2, media-foundation, \
                     videotoolbox, nvjpeg or cpu"
                )))
            }
        })
    }
}

/// Parses an [`ENGINE_ENV`] value: `None` when it names no engine (empty, or
/// only `auto`).
fn parse_engine_list(value: &str) -> Result<Option<Vec<Engine>>> {
    let mut engines = Vec::new();
    for name in value.split(',').filter(|n| !n.trim().is_empty()) {
        let engine: Engine = name.parse()?;
        if engine != Engine::Auto && !engines.contains(&engine) {
            engines.push(engine);
        }
    }
    Ok((!engines.is_empty()).then_some(engines))
}

/// The engines to try for `requested`, in order. `able` lists, in the
/// platform's order, the engines this build has that can do the request;
/// [`Engine::Auto`] picks from it as [`ENGINE_ENV`] allows, and an explicit
/// engine is used alone, whether or not `able` lists it.
///
/// # Errors
///
/// [`CodecError::NoDevice`] for an explicit engine this platform or build
/// does not have, and [`CodecError::InvalidConfig`] for an [`ENGINE_ENV`]
/// naming an unknown engine.
pub(crate) fn engines_to_try(requested: Engine, able: &[Engine]) -> Result<Vec<Engine>> {
    if requested != Engine::Auto {
        if !compiled(requested) {
            return Err(not_compiled(requested));
        }
        return Ok(vec![requested]);
    }
    let listed = match std::env::var(ENGINE_ENV) {
        Ok(value) => parse_engine_list(&value)?,
        Err(_) => None,
    };
    Ok(order(able, listed.as_deref()))
}

/// `able`, or the engines of `listed` that `able` has, in `listed`'s order.
fn order(able: &[Engine], listed: Option<&[Engine]>) -> Vec<Engine> {
    match listed {
        None => able.to_vec(),
        Some(listed) => listed
            .iter()
            .copied()
            .filter(|e| able.contains(e))
            .collect(),
    }
}

/// The error for `engine` (or, for [`Engine::Auto`], every engine here) not
/// doing `direction` of `codec`.
pub(crate) fn unsupported(codec: Codec, direction: Direction, engine: Engine) -> CodecError {
    CodecError::Unsupported(UnsupportedFeature::Codec {
        codec,
        direction,
        engine,
    })
}

/// Whether an engine exists on this platform and in this build.
pub(crate) fn compiled(engine: Engine) -> bool {
    match engine {
        Engine::Auto | Engine::Cpu => true,
        Engine::V4l2 => cfg!(all(target_os = "linux", feature = "v4l2")),
        Engine::MediaFoundation => cfg!(all(windows, feature = "media-foundation")),
        Engine::VideoToolbox => false,
        Engine::NvJpeg => cfg!(all(target_os = "linux", feature = "nvjpeg")),
    }
}

/// The error for an explicit engine this platform or build does not have.
pub(crate) fn not_compiled(engine: Engine) -> CodecError {
    CodecError::NoDevice(format!(
        "the {engine} codec engine is not available on this platform or in this build"
    ))
}

/// How pixels move between a tensor and an engine.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Transfer {
    /// The engine reads or writes the tensor's own memory: a device importing
    /// or exporting a DMA-BUF, a D3D11 texture read in place, or the CPU
    /// working in the mapped tensor.
    ZeroCopy,
    /// One copy per frame on the GPU, between the tensor's texture and one of
    /// the engine's.
    GpuCopy,
    /// One copy per frame through CPU memory.
    CpuCopy,
}

impl fmt::Display for Transfer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::ZeroCopy => "zero-copy",
            Self::GpuCopy => "GPU copy",
            Self::CpuCopy => "CPU copy",
        })
    }
}

/// One thing an engine on this machine can do.
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Capability {
    /// The engine.
    pub engine: Engine,
    /// The device within the engine: a device node, a transform's name, or
    /// `cpu`.
    pub device: String,
    /// The compressed format.
    pub codec: Codec,
    /// Encoding or decoding.
    pub direction: Direction,
    /// Whether the work runs on dedicated hardware rather than the CPU.
    pub hardware: bool,
    /// The pixel formats of the images: what an encoder takes, or what a
    /// decoder produces.
    pub formats: Vec<PixelFormat>,
    /// How frames move in the best case, for a source or pool in the memory
    /// the engine prefers. An open stream reports what it does in
    /// [`EncoderInfo`](crate::video::EncoderInfo) and
    /// [`StreamInfo`](crate::video::StreamInfo).
    pub transfer: Transfer,
}

/// What the codec engines on this machine can do, engine by engine, in the
/// order [`Engine::Auto`] tries them. Probing opens no stream, but it does
/// open device nodes and enumerate (on Windows, activate) transforms, so
/// it is not free: call it once.
pub fn capabilities() -> Vec<Capability> {
    let mut found = Vec::new();
    #[cfg(all(target_os = "linux", feature = "v4l2"))]
    crate::video::v4l2::capabilities(&mut found);
    #[cfg(all(windows, feature = "media-foundation"))]
    crate::video::mf::capabilities(&mut found);
    crate::jpeg::capabilities(&mut found);
    #[cfg(feature = "png")]
    found.push(Capability {
        engine: Engine::Cpu,
        device: "cpu".into(),
        codec: Codec::Png,
        direction: Direction::Decode,
        hardware: false,
        formats: vec![PixelFormat::Rgb, PixelFormat::Rgba, PixelFormat::Grey],
        // Rows are decoded into scratch memory, then copied into the tensor.
        transfer: Transfer::CpuCopy,
    });
    found
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn engine_names_round_trip() {
        for e in [
            Engine::Auto,
            Engine::V4l2,
            Engine::MediaFoundation,
            Engine::VideoToolbox,
            Engine::NvJpeg,
            Engine::Cpu,
        ] {
            assert_eq!(e.name().parse::<Engine>().unwrap(), e);
            assert_eq!(e.to_string().to_uppercase().parse::<Engine>().unwrap(), e);
        }
        assert_eq!("mf".parse::<Engine>().unwrap(), Engine::MediaFoundation);
        assert_eq!(" vt ".parse::<Engine>().unwrap(), Engine::VideoToolbox);
        assert!(matches!(
            "ffmpeg".parse::<Engine>(),
            Err(CodecError::InvalidConfig(_))
        ));
    }

    #[test]
    fn an_engine_list_keeps_named_engines_in_order() {
        assert_eq!(parse_engine_list("").unwrap(), None);
        assert_eq!(parse_engine_list(" auto ").unwrap(), None);
        assert_eq!(
            parse_engine_list("cpu, v4l2,cpu").unwrap(),
            Some(vec![Engine::Cpu, Engine::V4l2])
        );
        assert!(parse_engine_list("v4l2,gstreamer").is_err());
    }

    #[test]
    fn auto_picks_from_the_able_engines_as_the_list_allows() {
        let able = [Engine::NvJpeg, Engine::V4l2, Engine::Cpu];
        assert_eq!(order(&able, None), able.to_vec());
        assert_eq!(
            order(&able, Some(&[Engine::Cpu, Engine::V4l2])),
            vec![Engine::Cpu, Engine::V4l2]
        );
        assert_eq!(
            order(&able, Some(&[Engine::MediaFoundation])),
            Vec::<Engine>::new()
        );
    }

    #[test]
    fn an_explicit_engine_is_used_alone() {
        assert_eq!(
            engines_to_try(Engine::Cpu, &[Engine::V4l2]).unwrap(),
            vec![Engine::Cpu]
        );
    }

    #[test]
    fn video_codecs_are_told_from_still_images() {
        assert!(Codec::H264.is_video() && Codec::Mjpeg.is_video() && Codec::Av1.is_video());
        assert!(!Codec::Jpeg.is_video() && !Codec::Png.is_video());
    }

    #[test]
    fn the_cpu_engine_is_always_compiled() {
        assert!(compiled(Engine::Cpu) && compiled(Engine::Auto));
        assert!(!compiled(Engine::VideoToolbox));
        assert_eq!(
            compiled(Engine::MediaFoundation),
            cfg!(all(windows, feature = "media-foundation"))
        );
    }

    #[test]
    fn capabilities_list_the_cpu_jpeg_decoder() {
        let caps = capabilities();
        assert!(caps.iter().any(|c| c.engine == Engine::Cpu
            && c.codec == Codec::Jpeg
            && c.direction == Direction::Decode));
    }
}
