// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! Baseline JPEG decoder with strided native output.
//!
//! The decoder emits the source's native format by default: `Grey` for
//! greyscale (1-component) JPEGs, and the matching semi-planar format for
//! colour (3-component) JPEGs by subsampling: `Nv12` for 4:2:0, `Nv16` for
//! 4:2:2, `Nv24` for 4:4:4. [`crate::ImageDecoder::set_output_format`]
//! selects a fused output of the software decoder: `Rgb` (4:4:4 or native
//! 4:2:0 sources) or `Nv12` (any colour source), converted at the MCU write
//! stage. The decoder never rotates; EXIF orientation is reported in
//! [`ImageInfo`].

pub mod bitstream;
pub mod color;
pub mod cpu;
pub mod huffman;
pub mod idct;
pub mod markers;
pub mod mcu;
pub mod types;

/// Optional V4L2 hardware JPEG-decoder backend (Linux only, `v4l2` feature).
#[cfg(all(target_os = "linux", feature = "v4l2"))]
mod v4l2;

/// Optional nvJPEG GPU JPEG-decoder backend (Linux + CUDA, `nvjpeg` feature).
#[cfg(all(target_os = "linux", feature = "nvjpeg"))]
mod nvjpeg;

/// Runtime probe: `true` when the nvJPEG GPU decoder is available.
#[cfg(all(target_os = "linux", feature = "nvjpeg"))]
pub(crate) use nvjpeg::is_nvjpeg_available as nvjpeg_available;

/// Runtime probe: `true` when a V4L2 JPEG decoder device is present (and not
/// opted out via `EDGEFIRST_DISABLE_V4L2`). Opens and drops the device once.
#[cfg(all(target_os = "linux", feature = "v4l2"))]
pub(crate) fn v4l2_available() -> bool {
    v4l2::device_available()
}

/// The node of the V4L2 JPEG decoder this backend drives, when there is one.
#[cfg(all(target_os = "linux", feature = "v4l2"))]
pub(crate) fn v4l2_device() -> Option<String> {
    v4l2::probe_path().map(|p| p.display().to_string())
}

/// Parse a boolean environment variable: `true` only for `1`/`true`/`yes`
/// (case-insensitive, trimmed); absent or anything else is `false`.
#[cfg(all(target_os = "linux", any(feature = "v4l2", feature = "nvjpeg")))]
pub(crate) fn env_flag(name: &str) -> bool {
    std::env::var(name)
        .map(|v| matches!(v.trim().to_ascii_lowercase().as_str(), "1" | "true" | "yes"))
        .unwrap_or(false)
}

use crate::engine::{engines_to_try, unsupported, Capability, Codec, Direction, Engine, Transfer};
use crate::error::{CodecError, UnsupportedFeature};
use crate::exif::read_exif_orientation;
use crate::options::ImageInfo;
use crate::pixel::ImagePixel;
use edgefirst_tensor::{PixelFormat, Tensor};

/// The engines this build has that decode JPEG, in the order
/// [`Engine::Auto`] tries them. nvJPEG is listed only when it is enabled.
fn decode_engines() -> Vec<Engine> {
    [
        (Engine::NvJpeg, crate::nvjpeg_available()),
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

/// Adds the JPEG decoders present here to `out`.
pub(crate) fn capabilities(out: &mut Vec<Capability>) {
    let jpeg = |engine, device: String, hardware, formats: Vec<PixelFormat>| Capability {
        engine,
        device,
        codec: Codec::Jpeg,
        direction: Direction::Decode,
        hardware,
        formats,
        transfer: Transfer::ZeroCopy,
    };
    #[cfg(all(target_os = "linux", feature = "nvjpeg"))]
    if nvjpeg_available() {
        out.push(jpeg(
            Engine::NvJpeg,
            "nvjpeg".into(),
            true,
            vec![PixelFormat::Rgb],
        ));
    }
    #[cfg(all(target_os = "linux", feature = "v4l2"))]
    if let Some(path) = v4l2::probe_path() {
        out.push(jpeg(
            Engine::V4l2,
            path.display().to_string(),
            true,
            vec![PixelFormat::Nv12, PixelFormat::Grey],
        ));
    }
    out.push(jpeg(
        Engine::Cpu,
        "cpu".into(),
        false,
        vec![
            PixelFormat::Nv12,
            PixelFormat::Nv16,
            PixelFormat::Nv24,
            PixelFormat::Grey,
            PixelFormat::Rgb,
        ],
    ));
}

/// Reusable JPEG decoder state.
///
/// Holds scratch buffers that grow to the high-water mark and are reused across
/// decode calls.
pub struct JpegDecoderState {
    /// MCU scratch buffers (per-component IDCT output for one MCU row band).
    mcu_scratch: Option<mcu::McuScratch>,
    /// Huffman LUTs reused across frames when the DHT payload is unchanged.
    /// Tables are moved into [`markers::JpegHeaders`] for the decode and
    /// restored afterwards.
    huffman_cache: huffman::HuffmanCache,
    /// Optional fused-output preference (see [`resolve_output_format`]).
    pub(crate) preferred_format: Option<PixelFormat>,
    /// Chroma upsampling filter for a fused native-4:2:0 `Rgb` decode; see
    /// [`crate::ChromaUpsample`].
    pub(crate) chroma_upsample: crate::decoder::ChromaUpsample,
    /// Fast (AAN `ifast`-class) IDCT; see [`crate::DctMethod`]. False by
    /// default; `EDGEFIRST_CODEC_DCT=fast` makes it the constructor default.
    pub(crate) fast_dct: bool,
    /// The engine that decodes; see [`crate::ImageDecoder::set_engine`].
    pub(crate) engine: Engine,
    /// V4L2 hardware decoder, probed on first decode, at most once; a ready
    /// context is reused across decodes.
    #[cfg(all(target_os = "linux", feature = "v4l2"))]
    v4l2: v4l2::V4l2Probe,
    /// nvJPEG GPU decoder, probed on first decode; one nvJPEG
    /// handle/state/stream is reused across decodes.
    #[cfg(all(target_os = "linux", feature = "nvjpeg"))]
    nvjpeg: nvjpeg::NvJpegProbe,
}

impl JpegDecoderState {
    /// Whether the V4L2 decoder wrote the last picture straight into its
    /// destination.
    #[cfg(all(target_os = "linux", feature = "v4l2"))]
    pub(crate) fn v4l2_wrote_in_place(&self) -> bool {
        self.v4l2.wrote_in_place()
    }

    pub fn new() -> Self {
        Self {
            mcu_scratch: None,
            huffman_cache: huffman::HuffmanCache::default(),
            preferred_format: None,
            chroma_upsample: crate::decoder::ChromaUpsample::default(),
            fast_dct: std::env::var("EDGEFIRST_CODEC_DCT")
                .map(|v| v.trim().eq_ignore_ascii_case("fast"))
                .unwrap_or(false),
            engine: Engine::Auto,
            #[cfg(all(target_os = "linux", feature = "v4l2"))]
            v4l2: v4l2::V4l2Probe::default(),
            #[cfg(all(target_os = "linux", feature = "nvjpeg"))]
            nvjpeg: nvjpeg::NvJpegProbe::default(),
        }
    }
}

impl Default for JpegDecoderState {
    fn default() -> Self {
        Self::new()
    }
}

/// The codec's native output format for a JPEG: `Grey` for 1-component
/// (greyscale) images, and the matching semi-planar format for 3-component
/// (YCbCr) images by subsampling — `Nv24` (4:4:4), `Nv16` (4:2:2), or `Nv12`
/// (4:2:0); non-standard subsamplings downsample to `Nv12`.
fn native_format(headers: &markers::JpegHeaders) -> crate::Result<PixelFormat> {
    let comps = &headers.header.components;
    match comps.len() {
        1 => Ok(PixelFormat::Grey),
        3 => {
            // Native chroma format by subsampling:
            //   4:4:4 (full-res chroma)        → NV24
            //   4:2:2 (half-width chroma)      → NV16
            //   4:2:0 (quarter-res chroma)     → NV12
            // Non-standard subsamplings (4:1:1, 4:1:0, mismatched Cb/Cr) map to NV12.
            let max_h = headers.header.max_h_samp as usize;
            let max_v = headers.header.max_v_samp as usize;
            let cb = comps[1].sampling;
            let cr = comps[2].sampling;
            // Chroma components with different sampling map to NV12.
            if cb.h == cr.h && cb.v == cr.v && cb.h as usize > 0 && cb.v as usize > 0 {
                let h_ratio = max_h / cb.h as usize;
                let v_ratio = max_v / cb.v as usize;
                return Ok(match (h_ratio, v_ratio) {
                    (1, 1) => PixelFormat::Nv24, // 4:4:4
                    (2, 1) => PixelFormat::Nv16, // 4:2:2
                    (2, 2) => PixelFormat::Nv12, // 4:2:0
                    _ => PixelFormat::Nv12,      // exotic → downsample to 4:2:0
                });
            }
            Ok(PixelFormat::Nv12)
        }
        n => Err(CodecError::Unsupported(
            UnsupportedFeature::JpegComponentCount {
                components: n as u8,
            },
        )),
    }
}

/// Resolve the decode output format from the native format and an optional
/// caller preference (fused decode outputs):
///
/// - `Rgb`: honoured for 4:4:4 colour JPEGs (native `Nv24`) and for native
///   4:2:0 colour JPEGs (sampling factors matching [`mcu::is_native_420`]).
///   Any other source (4:2:2, greyscale, or a non-standard subsampling)
///   returns [`CodecError::UnsupportedFormat`].
/// - `Nv12`: honoured for any colour JPEG; the MCU writer downsamples
///   chroma (2×2 for 4:4:4, vertical for 4:2:2; native 4:2:0 passes through).
/// - Anything else (or `None`): native format.
fn resolve_output_format(
    headers: &markers::JpegHeaders,
    native: PixelFormat,
    preferred: Option<PixelFormat>,
) -> crate::Result<PixelFormat> {
    Ok(match preferred {
        Some(PixelFormat::Rgb) if native == PixelFormat::Nv24 => PixelFormat::Rgb,
        Some(PixelFormat::Rgb)
            if native == PixelFormat::Nv12 && mcu::is_native_420(&headers.header) =>
        {
            PixelFormat::Rgb
        }
        Some(PixelFormat::Rgb) => return Err(CodecError::UnsupportedFormat(PixelFormat::Rgb)),
        Some(PixelFormat::Nv12)
            if matches!(
                native,
                PixelFormat::Nv12 | PixelFormat::Nv16 | PixelFormat::Nv24
            ) =>
        {
            PixelFormat::Nv12
        }
        _ => native,
    })
}

/// Native luma/primary-plane row stride in bytes for a freshly decoded image:
/// `even(width)` rounded up to 64 bytes.
fn native_row_stride(width: usize) -> usize {
    width.next_multiple_of(2).next_multiple_of(64)
}

/// Parse JPEG headers and return native dimensions, format, and EXIF
/// orientation without decoding pixels. `width` and `height` are the source's
/// dimensions before the orientation is applied.
pub fn peek_jpeg_info(data: &[u8]) -> crate::Result<ImageInfo> {
    let headers = markers::parse_markers(data)?;
    let img_w = headers.header.width as usize;
    let img_h = headers.header.height as usize;
    let format = native_format(&headers)?;
    let (rotation_degrees, flip_horizontal) = headers
        .exif_data
        .as_deref()
        .map(read_exif_orientation)
        .unwrap_or((0, false));
    Ok(ImageInfo {
        width: img_w,
        height: img_h,
        format,
        row_stride: native_row_stride(img_w),
        rotation_degrees,
        flip_horizontal,
    })
}

/// Decode a JPEG image from `data` into `dst`, configuring `dst`'s dimensions
/// and pixel format to the decoded native format.
///
/// The destination must be a `u8` tensor with enough underlying capacity for
/// the decoded image; smaller allocations yield
/// [`CodecError::InsufficientCapacity`]. Decoded rows are written at the
/// tensor's `effective_row_stride()` offsets. The returned [`ImageInfo`]
/// reports the EXIF orientation for the caller to apply downstream.
pub fn decode_jpeg_into<T: ImagePixel>(
    data: &[u8],
    dst: &mut Tensor<T>,
    state: &mut JpegDecoderState,
) -> crate::Result<ImageInfo> {
    let _span = tracing::trace_span!(
        "codec.decode_jpeg",
        dtype = std::any::type_name::<T>(),
        n_bytes = data.len(),
    )
    .entered();

    let mut headers = {
        let _s = tracing::trace_span!("codec.decode_jpeg.parse_markers").entered();
        markers::parse_markers_with_cache(data, Some(&mut state.huffman_cache))?
    };
    let result = decode_jpeg_into_parsed(data, dst, state, &headers);
    state
        .huffman_cache
        .restore(&mut headers.dc_tables, &mut headers.ac_tables);
    result
}

/// The picture a decode writes: its format, size and the destination's row
/// stride.
#[cfg_attr(
    not(all(target_os = "linux", any(feature = "v4l2", feature = "nvjpeg"))),
    allow(dead_code)
)]
struct Target {
    format: PixelFormat,
    width: usize,
    height: usize,
    stride: usize,
    /// Whether the format is the JPEG's own; fused outputs are produced by
    /// the CPU decoder only.
    native: bool,
}

/// What a hardware decoder did with an image.
#[cfg_attr(not(all(target_os = "linux", feature = "v4l2")), allow(dead_code))]
enum Attempt {
    /// Decoded; the orientation is still to be filled in.
    Decoded(ImageInfo),
    /// Not this image, for the reason given.
    PassedOn(String),
    /// The decoder is not present.
    NoDevice(String),
}

/// Decodes on hardware `engine`.
#[cfg_attr(
    not(all(target_os = "linux", any(feature = "v4l2", feature = "nvjpeg"))),
    allow(unused_variables)
)]
fn try_hardware<T: ImagePixel>(
    engine: Engine,
    data: &[u8],
    headers: &markers::JpegHeaders,
    dst: &mut Tensor<T>,
    state: &mut JpegDecoderState,
    target: &Target,
) -> crate::Result<Attempt> {
    match engine {
        #[cfg(all(target_os = "linux", any(feature = "v4l2", feature = "nvjpeg")))]
        Engine::NvJpeg | Engine::V4l2 if !target.native => {
            Ok(Attempt::PassedOn("a fused output format".into()))
        }
        // nvJPEG decodes interleaved RGB into a CUDA-backed (PBO) destination.
        #[cfg(all(target_os = "linux", feature = "nvjpeg"))]
        Engine::NvJpeg => match state.nvjpeg.try_decode::<T>(
            data,
            headers,
            dst,
            target.format,
            target.width,
            target.height,
            target.stride,
        ) {
            Ok(Some(info)) => {
                // RGB output carries no chroma colorimetry: clear it.
                dst.set_colorimetry(None);
                Ok(Attempt::Decoded(info))
            }
            Ok(None) => Ok(Attempt::PassedOn("no nvJPEG decoder".into())),
            Err(nvjpeg::NvJpegDecode::Fallback(why)) => Ok(Attempt::PassedOn(why)),
            Err(nvjpeg::NvJpegDecode::Fatal(e)) => Err(e),
        },
        #[cfg(all(target_os = "linux", feature = "v4l2"))]
        Engine::V4l2 => match state.v4l2.try_decode::<T>(
            data,
            headers,
            dst,
            target.format,
            target.width,
            target.height,
            target.stride,
        ) {
            Ok(Some(info)) => {
                dst.set_colorimetry(Some(edgefirst_tensor::Colorimetry::jfif()));
                Ok(Attempt::Decoded(info))
            }
            Ok(None) => Ok(Attempt::NoDevice("no V4L2 JPEG decoder".into())),
            Err(v4l2::V4l2Decode::Fallback(why)) => Ok(Attempt::PassedOn(why)),
            Err(v4l2::V4l2Decode::Fatal(e)) => Err(e),
        },
        other => Err(unsupported(Codec::Jpeg, Direction::Decode, other)),
    }
}

/// Decode using already-parsed headers. Huffman tables live in `headers` for
/// the duration of this call; the caller restores them to the cache after.
fn decode_jpeg_into_parsed<T: ImagePixel>(
    data: &[u8],
    dst: &mut Tensor<T>,
    state: &mut JpegDecoderState,
    headers: &markers::JpegHeaders,
) -> crate::Result<ImageInfo> {
    let img_w = headers.header.width as usize;
    let img_h = headers.header.height as usize;
    let native_fmt = native_format(headers)?;
    let output_fmt = resolve_output_format(headers, native_fmt, state.preferred_format)?;

    // Reject non-u8 destinations.
    if T::dtype() != edgefirst_tensor::DType::U8 {
        return Err(CodecError::UnsupportedDtype(T::dtype()));
    }

    // EXIF orientation is reported, never applied.
    let (rotation_degrees, flip_horizontal) = headers
        .exif_data
        .as_deref()
        .map(read_exif_orientation)
        .unwrap_or((0, false));

    // Configure the destination tensor to the decoded image (or error if its
    // allocation is too small).
    let cap_w = dst.width().unwrap_or(0);
    let cap_h = dst.height().unwrap_or(0);
    dst.configure_image(img_w, img_h, output_fmt)
        .map_err(|e| match e {
            edgefirst_tensor::Error::InsufficientCapacity { .. } => {
                CodecError::InsufficientCapacity {
                    image: (img_w, img_h),
                    tensor: (cap_w, cap_h),
                }
            }
            other => CodecError::Tensor(other),
        })?;
    let dst_stride = dst
        .effective_row_stride()
        .unwrap_or(native_row_stride(img_w));

    // Try the engines in order. Under `Auto` a hardware decoder that cannot
    // take this image passes it on; an engine asked for by name fails instead.
    // An image every listed engine passes on is refused.
    let target = Target {
        format: output_fmt,
        width: img_w,
        height: img_h,
        stride: dst_stride,
        native: output_fmt == native_fmt,
    };
    let mut cpu = false;
    for engine in engines_to_try(state.engine, &decode_engines())? {
        if engine == Engine::Cpu {
            cpu = true;
            break;
        }
        match try_hardware(engine, data, headers, dst, state, &target)? {
            Attempt::Decoded(mut info) => {
                info.rotation_degrees = rotation_degrees;
                info.flip_horizontal = flip_horizontal;
                return Ok(info);
            }
            Attempt::PassedOn(why) if state.engine == Engine::Auto => {
                log::debug!("{engine} jpeg decode passed on: {why}");
            }
            Attempt::PassedOn(why) => {
                log::debug!("{engine} jpeg decode refused: {why}");
                return Err(unsupported(Codec::Jpeg, Direction::Decode, engine));
            }
            Attempt::NoDevice(why) if state.engine == Engine::Auto => {
                log::debug!("{engine} jpeg decode passed on: {why}");
            }
            Attempt::NoDevice(why) => return Err(CodecError::NoDevice(why)),
        }
    }
    if !cpu {
        return Err(unsupported(Codec::Jpeg, Direction::Decode, state.engine));
    }

    // CPU decode: the MCU loop writes u8 pixels into the tensor.
    dst.cpu_access()
        .require(edgefirst_tensor::CpuAccess::Write)?;
    match &mut state.mcu_scratch {
        Some(scratch) => scratch.ensure_capacity(headers),
        None => state.mcu_scratch = Some(mcu::McuScratch::new(headers)),
    }
    let mcu_scratch = state.mcu_scratch.as_mut().unwrap();

    {
        let mut map = crate::pixel::map_decode_dst(dst)?;
        let dst_bytes: &mut [T] = &mut map;
        // SAFETY: T is u8 (checked above) — layout-identical reinterpret.
        let dst_u8: &mut [u8] = unsafe {
            std::slice::from_raw_parts_mut(dst_bytes.as_mut_ptr() as *mut u8, dst_bytes.len())
        };
        let _s = tracing::trace_span!("codec.decode_jpeg.mcu_loop").entered();
        mcu::decode_image(
            data,
            headers,
            mcu_scratch,
            dst_u8,
            dst_stride,
            output_fmt,
            state.fast_dct,
        )?;
    }

    // RGB output carries no chroma colorimetry: clear it.
    if output_fmt == PixelFormat::Rgb {
        dst.set_colorimetry(None);
    } else {
        dst.set_colorimetry(Some(edgefirst_tensor::Colorimetry::jfif()));
    }

    Ok(ImageInfo {
        width: img_w,
        height: img_h,
        format: output_fmt,
        row_stride: dst_stride,
        rotation_degrees,
        flip_horizontal,
    })
}
