// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! [`ImageEncoder`]: still images from tensors.

use edgefirst_tensor::{ColorRange, CpuAccess, DType, PixelFormat, TensorDyn};
use jpeg_encoder::{ColorType, Encoder, ImageBuffer, JpegColorType, SamplingFactor};

use crate::engine::{engines_to_try, unsupported};
use crate::{Codec, CodecError, Direction, Engine, Result};

/// JPEG quality [`ImageEncoder::new`] starts with.
pub const DEFAULT_JPEG_QUALITY: u8 = 85;

/// The pixel formats the CPU JPEG encoder takes.
pub(crate) const CPU_JPEG_FORMATS: [PixelFormat; 6] = [
    PixelFormat::Nv12,
    PixelFormat::Nv16,
    PixelFormat::Yuyv,
    PixelFormat::Grey,
    PixelFormat::Rgb,
    PixelFormat::Rgba,
];

/// Encodes image tensors into still-image files.
///
/// Create one and reuse it: one encoder serves images of any size and format
/// it takes, and keeps a hardware encoder open for the last geometry it
/// encoded.
///
/// ```rust,no_run
/// use edgefirst_codec::{Codec, ImageEncoder};
/// # let image: edgefirst_tensor::TensorDyn = unimplemented!();
/// let mut encoder = ImageEncoder::new(Codec::Jpeg)?;
/// encoder.set_quality(90)?;
/// let jpeg = encoder.encode(&image)?;
/// std::fs::write("frame.jpg", jpeg)?;
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
#[derive(Debug)]
pub struct ImageEncoder {
    codec: Codec,
    engine: Engine,
    quality: u8,
    /// The V4L2 JPEG encoder, open for one source geometry.
    #[cfg(all(target_os = "linux", feature = "v4l2"))]
    v4l2: Option<(V4l2Key, crate::video::VideoEncoder)>,
}

/// The source geometry and quality a V4L2 JPEG encoder was opened for.
#[cfg(all(target_os = "linux", feature = "v4l2"))]
type V4l2Key = (usize, usize, PixelFormat, usize, u8);

/// The engines this build has that encode JPEG, in the order
/// [`Engine::Auto`] tries them.
fn encode_engines() -> Vec<Engine> {
    [
        (
            Engine::V4l2,
            cfg!(all(target_os = "linux", feature = "v4l2")),
        ),
        (Engine::Cpu, true),
    ]
    .into_iter()
    .filter_map(|(engine, present)| present.then_some(engine))
    .collect()
}

impl ImageEncoder {
    /// An encoder for `codec` on [`Engine::Auto`], at
    /// [`DEFAULT_JPEG_QUALITY`].
    ///
    /// # Errors
    ///
    /// [`CodecError::Unsupported`] carrying
    /// [`UnsupportedFeature::Codec`](crate::UnsupportedFeature::Codec) for a
    /// codec no engine here encodes as a still image (only JPEG today).
    pub fn new(codec: Codec) -> Result<Self> {
        if codec != Codec::Jpeg {
            return Err(unsupported(codec, Direction::Encode, Engine::Auto));
        }
        Ok(Self {
            codec,
            engine: Engine::Auto,
            quality: DEFAULT_JPEG_QUALITY,
            #[cfg(all(target_os = "linux", feature = "v4l2"))]
            v4l2: None,
        })
    }

    /// Selects the engine that encodes. [`Engine::Auto`] (the default) tries
    /// V4L2 (a hardware JPEG encoder, for DMA-BUF sources), then the CPU,
    /// which encodes every format in [`encode_into`](Self::encode_into)'s
    /// list, as [`ENGINE_ENV`](crate::ENGINE_ENV) allows. Any other engine
    /// encodes alone.
    pub fn set_engine(&mut self, engine: Engine) {
        self.engine = engine;
    }

    /// The engine selected with [`set_engine`](Self::set_engine).
    pub fn engine(&self) -> Engine {
        self.engine
    }

    /// Sets the JPEG quality, 1 (smallest) to 100 (best).
    ///
    /// # Errors
    ///
    /// [`CodecError::InvalidConfig`] outside 1 to 100.
    pub fn set_quality(&mut self, quality: u8) -> Result<()> {
        if !(1..=100).contains(&quality) {
            return Err(CodecError::InvalidConfig(format!(
                "JPEG quality {quality} is not within 1 to 100"
            )));
        }
        self.quality = quality;
        Ok(())
    }

    /// The JPEG quality.
    pub fn quality(&self) -> u8 {
        self.quality
    }

    /// Encodes `src` and returns the file's bytes.
    ///
    /// See [`encode_into`](Self::encode_into).
    pub fn encode(&mut self, src: &TensorDyn) -> Result<Vec<u8>> {
        let mut out = Vec::new();
        self.encode_into(src, &mut out)?;
        Ok(out)
    }

    /// Encodes `src`, replacing the contents of `out`, whose capacity is
    /// reused.
    ///
    /// The CPU encoder takes `u8` images in NV12 and NV16 (chroma subsampled
    /// 4:2:0 and 4:2:2 in the file), YUYV (4:2:2), Grey, RGB and RGBA, at any
    /// row stride, of up to 65535 pixels a side. YUV samples are written as
    /// the file's YCbCr, rescaled to full range when the tensor's colorimetry
    /// says limited range; the matrix is not changed, so a BT.709 source
    /// decodes with the BT.601 colours every JPEG reader assumes.
    ///
    /// # Errors
    ///
    /// [`CodecError::UnsupportedFormat`] or [`CodecError::UnsupportedDtype`]
    /// for a source no engine to try takes, [`CodecError::InvalidInput`] for
    /// one too large, and [`CodecError::Tensor`] when the source cannot be
    /// mapped for reading.
    pub fn encode_into(&mut self, src: &TensorDyn, out: &mut Vec<u8>) -> Result<()> {
        let _span = tracing::trace_span!("codec.encode_jpeg").entered();
        #[cfg(all(target_os = "linux", feature = "v4l2"))]
        let explicit = self.engine != Engine::Auto;
        // In a build without a hardware JPEG encoder the first engine is the
        // CPU or refused, so the loop body always leaves it.
        #[cfg_attr(
            not(all(target_os = "linux", feature = "v4l2")),
            allow(clippy::never_loop)
        )]
        for engine in engines_to_try(self.engine, &encode_engines())? {
            match engine {
                #[cfg(all(target_os = "linux", feature = "v4l2"))]
                Engine::V4l2 => match self.encode_v4l2(src, out) {
                    Ok(()) => return Ok(()),
                    Err(e) if explicit => return Err(e),
                    Err(e) => log::debug!("v4l2 jpeg encode passed on: {e}"),
                },
                Engine::Cpu => return encode_jpeg_cpu(src, self.quality, out),
                other => return Err(unsupported(self.codec, Direction::Encode, other)),
            }
        }
        Err(unsupported(self.codec, Direction::Encode, self.engine))
    }

    /// Encodes `src` on the V4L2 JPEG encoder, opened for its geometry.
    #[cfg(all(target_os = "linux", feature = "v4l2"))]
    fn encode_v4l2(&mut self, src: &TensorDyn, out: &mut Vec<u8>) -> Result<()> {
        use crate::video::{EncoderConfig, FrameOptions, VideoEncoder};
        let format = src
            .format()
            .ok_or(CodecError::UnsupportedDtype(src.dtype()))?;
        let (w, h) = (src.width().unwrap_or(0), src.height().unwrap_or(0));
        let stride = src.effective_row_stride().unwrap_or(0);
        let key = (w, h, format, stride, self.quality);
        if self.v4l2.as_ref().map(|(k, _)| *k) != Some(key) {
            self.v4l2 = None;
            let mut config = EncoderConfig::new(Codec::Mjpeg, w as u32, h as u32, format, 30.0)
                .with_engine(Engine::V4l2);
            config.quality = Some(self.quality);
            self.v4l2 = Some((key, VideoEncoder::new(config)?));
        }
        let (_, encoder) = self.v4l2.as_mut().expect("opened above");
        let frame = match encoder.encode(src, &FrameOptions::default())? {
            Some(frame) => frame,
            None => encoder.flush()?.into_iter().next().ok_or_else(|| {
                CodecError::Timeout("the V4L2 JPEG encoder returned no picture".into())
            })?,
        };
        out.clear();
        out.extend_from_slice(&frame.data);
        Ok(())
    }
}

/// The JPEG of `src` on the CPU.
fn encode_jpeg_cpu(src: &TensorDyn, quality: u8, out: &mut Vec<u8>) -> Result<()> {
    if src.dtype() != DType::U8 {
        return Err(CodecError::UnsupportedDtype(src.dtype()));
    }
    let format = src
        .format()
        .ok_or_else(|| CodecError::InvalidInput("the source is not an image".into()))?;
    if !CPU_JPEG_FORMATS.contains(&format) {
        return Err(CodecError::UnsupportedFormat(format));
    }
    let (w, h) = (src.width().unwrap_or(0), src.height().unwrap_or(0));
    let side = |v: usize| u16::try_from(v).ok().filter(|&v| v > 0);
    let (Some(width), Some(height)) = (side(w), side(h)) else {
        return Err(CodecError::InvalidInput(format!(
            "a {w}x{h} image cannot be a JPEG (1 to 65535 pixels a side)"
        )));
    };
    let stride = src
        .effective_row_stride()
        .ok_or_else(|| CodecError::InvalidInput("the source has no row stride".into()))?;
    let map = src.map_bytes(CpuAccess::Read)?;
    let limited = src
        .colorimetry()
        .and_then(|c| c.range)
        .is_some_and(|r| r == ColorRange::Limited);
    let image = Source {
        bytes: &map,
        format,
        width,
        height,
        stride,
        levels: Levels::new(limited),
    };
    let needed = image.bytes_needed();
    if map.len() < needed {
        return Err(CodecError::InvalidInput(format!(
            "a {w}x{h} {format} image at stride {stride} needs {needed} bytes; the tensor maps {}",
            map.len()
        )));
    }
    out.clear();
    let mut encoder = Encoder::new(&mut *out, quality);
    match format {
        PixelFormat::Nv12 => encoder.set_sampling_factor(SamplingFactor::F_2_2),
        PixelFormat::Nv16 | PixelFormat::Yuyv => encoder.set_sampling_factor(SamplingFactor::F_2_1),
        _ => {}
    }
    let encoded = match format {
        PixelFormat::Rgb | PixelFormat::Rgba if stride == w * format.channels() => {
            let colour = if format == PixelFormat::Rgb {
                ColorType::Rgb
            } else {
                ColorType::Rgba
            };
            encoder.encode(&map[..needed], width, height, colour)
        }
        _ => encoder.encode_image(image),
    };
    encoded.map_err(|e| CodecError::InvalidInput(format!("JPEG encode: {e}")))
}

/// Maps limited-range samples to full range, or leaves them.
struct Levels {
    luma: [u8; 256],
    chroma: [u8; 256],
}

impl Levels {
    fn new(limited: bool) -> Self {
        let table = |offset: f32, span: f32, centre: f32| {
            let mut t = [0u8; 256];
            for (v, out) in t.iter_mut().enumerate() {
                *out = if limited {
                    (centre + (v as f32 - offset) * 255.0 / span)
                        .round()
                        .clamp(0.0, 255.0) as u8
                } else {
                    v as u8
                };
            }
            t
        };
        Self {
            luma: table(16.0, 219.0, 0.0),
            chroma: table(128.0, 224.0, 128.0),
        }
    }
}

/// A mapped source image, as the JPEG encoder reads it row by row.
struct Source<'a> {
    bytes: &'a [u8],
    format: PixelFormat,
    width: u16,
    height: u16,
    stride: usize,
    levels: Levels,
}

impl Source<'_> {
    /// Bytes the source spans in its map.
    fn bytes_needed(&self) -> usize {
        let (w, h) = (usize::from(self.width), usize::from(self.height));
        let rows = match self.format {
            PixelFormat::Nv12 => h + h.div_ceil(2),
            PixelFormat::Nv16 => 2 * h,
            _ => h,
        };
        let last = match self.format {
            PixelFormat::Nv12 | PixelFormat::Nv16 => w.next_multiple_of(2),
            other => w * other.channels(),
        };
        (rows - 1) * self.stride + last
    }
}

impl ImageBuffer for Source<'_> {
    fn get_jpeg_color_type(&self) -> JpegColorType {
        match self.format {
            PixelFormat::Grey => JpegColorType::Luma,
            _ => JpegColorType::Ycbcr,
        }
    }

    fn width(&self) -> u16 {
        self.width
    }

    fn height(&self) -> u16 {
        self.height
    }

    fn fill_buffers(&self, y: u16, buffers: &mut [Vec<u8>; 4]) {
        let (w, h, y) = (
            usize::from(self.width),
            usize::from(self.height),
            usize::from(y),
        );
        let row = &self.bytes[y * self.stride..];
        let Levels { luma, chroma } = &self.levels;
        match self.format {
            PixelFormat::Grey => buffers[0].extend(row[..w].iter().map(|&v| luma[v as usize])),
            PixelFormat::Nv12 | PixelFormat::Nv16 => {
                let chroma_row = match self.format {
                    PixelFormat::Nv12 => h + y / 2,
                    _ => h + y,
                };
                let uv = &self.bytes[chroma_row * self.stride..];
                for x in 0..w {
                    let pair = x & !1;
                    buffers[0].push(luma[row[x] as usize]);
                    buffers[1].push(chroma[uv[pair] as usize]);
                    buffers[2].push(chroma[uv[pair + 1] as usize]);
                }
            }
            PixelFormat::Yuyv => {
                for x in 0..w {
                    let pair = 4 * (x / 2);
                    buffers[0].push(luma[row[2 * x] as usize]);
                    buffers[1].push(chroma[row[pair + 1] as usize]);
                    buffers[2].push(chroma[row[pair + 3] as usize]);
                }
            }
            PixelFormat::Rgb | PixelFormat::Rgba => {
                let channels = self.format.channels();
                for px in row[..w * channels].chunks_exact(channels) {
                    let (r, g, b) = (f32::from(px[0]), f32::from(px[1]), f32::from(px[2]));
                    let to_u8 = |v: f32| v.round().clamp(0.0, 255.0) as u8;
                    buffers[0].push(to_u8(0.299 * r + 0.587 * g + 0.114 * b));
                    buffers[1].push(to_u8(128.0 - 0.168_736 * r - 0.331_264 * g + 0.5 * b));
                    buffers[2].push(to_u8(128.0 + 0.5 * r - 0.418_688 * g - 0.081_312 * b));
                }
            }
            other => unreachable!("{other} is not a CPU JPEG source format"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn quality_is_one_to_one_hundred() {
        let mut e = ImageEncoder::new(Codec::Jpeg).unwrap();
        assert_eq!(e.quality(), DEFAULT_JPEG_QUALITY);
        assert!(e.set_quality(0).is_err() && e.set_quality(101).is_err());
        e.set_quality(100).unwrap();
        assert_eq!(e.quality(), 100);
    }

    #[test]
    fn only_jpeg_is_a_still_image_encoder() {
        for codec in [Codec::Png, Codec::H264, Codec::Mjpeg] {
            assert!(matches!(
                ImageEncoder::new(codec),
                Err(CodecError::Unsupported(
                    crate::UnsupportedFeature::Codec { .. }
                ))
            ));
        }
    }

    #[test]
    fn limited_range_samples_are_stretched_to_full_range() {
        let l = Levels::new(true);
        assert_eq!(
            (l.luma[16], l.luma[235], l.luma[0], l.luma[255]),
            (0, 255, 0, 255)
        );
        assert_eq!((l.chroma[16], l.chroma[128], l.chroma[240]), (1, 128, 255));
        let f = Levels::new(false);
        assert_eq!((f.luma[16], f.chroma[240]), (16, 240));
    }
}
