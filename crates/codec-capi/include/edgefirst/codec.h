#ifndef EDGEFIRST_CODEC_H
#define EDGEFIRST_CODEC_H

/**
 * @file codec.h
 * @brief EdgeFirst image and video encode/decode C API
 *
 * SPDX-License-Identifier: Apache-2.0
 * Copyright (c) 2026 Au-Zone Technologies. All Rights Reserved.
 *
 * Still images: decode JPEG and PNG into a tensor (ef_image_decoder), and
 * encode a tensor as JPEG (ef_image_encoder). Video: encode tensors to H.264
 * or MJPEG access units (ef_video_encoder), and decode access units into
 * frames whose pictures are tensors (ef_video_decoder).
 *
 * Image decoding writes *into* a tensor you already have, rather than
 * allocating one: that is what lets a decode target be a DMA buffer the GPU
 * will read next, with no copy in between. The tensor may come from any
 * EdgeFirst library.
 *
 * Every codec runs on an engine (ef_codec_engine): V4L2 on Linux, Media
 * Foundation on Windows, nvJPEG on Linux with CUDA, or the CPU for still
 * images. EF_CODEC_ENGINE_AUTO takes the first engine that can do the
 * request; ef_codec_capabilities_new lists what this machine's engines do.
 * There is no software video codec: where no engine runs a codec, opening
 * it fails with ENOTSUP or ENODEV.
 *
 * Calls return 0 or an errno. Functions that create an object return NULL
 * and set errno (on Windows, the thread's last-error code, read with
 * GetLastError). ef_codec_last_error says why the calling thread's last
 * call failed. Each object is used from one thread at a time.
 *
 * Platform-specific entry points are declared on every platform and fail at
 * run time where no engine runs them; a "Platforms:" line says where:
 *   - Linux, Windows: ef_video_encoder_new, ef_video_decoder_new,
 *     ef_video_decoder_new_with_pool, and the calls on their objects
 */

#include <stdarg.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>

#include "edgefirst/tensor.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief An opaque image decoder.
 *
 * Forward-declared so callers write `ef_image_decoder *` without the `struct`
 * keyword. Its definition is private: decoder state is not part of the ABI.
 */
typedef struct ef_image_decoder ef_image_decoder;


/* Generated with cbindgen:0.29.4 */

/* WARNING: The generated portion of this file is produced by cbindgen. Do not modify it directly. */

#include <stdarg.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>

/**
 * Encode the frame as a key frame (an IDR picture); a flag of
 * [`ef_video_encoder_encode`].
 */
#define EF_VIDEO_ENCODE_KEYFRAME 1

/**
 * A compressed image or video format.
 */
enum ef_codec
#if defined(__cplusplus) || __STDC_VERSION__ >= 202311L
  : uint32_t
#endif // defined(__cplusplus) || __STDC_VERSION__ >= 202311L
 {
  /**
   * JPEG still images.
   */
  EF_CODEC_JPEG = 0,
  /**
   * PNG still images.
   */
  EF_CODEC_PNG = 1,
  /**
   * H.264, as an Annex B byte stream.
   */
  EF_CODEC_H264 = 2,
  /**
   * HEVC, as an Annex B byte stream. No engine runs it yet.
   */
  EF_CODEC_HEVC = 3,
  /**
   * Motion JPEG: a stream of JPEG pictures.
   */
  EF_CODEC_MJPEG = 4,
  /**
   * VP8. No engine runs it yet.
   */
  EF_CODEC_VP8 = 5,
  /**
   * VP9. No engine runs it yet.
   */
  EF_CODEC_VP9 = 6,
  /**
   * AV1. No engine runs it yet.
   */
  EF_CODEC_AV1 = 7,
};
#ifndef __cplusplus
#if __STDC_VERSION__ >= 202311L
typedef enum ef_codec ef_codec;
#else
typedef uint32_t ef_codec;
#endif // __STDC_VERSION__ >= 202311L
#endif // __cplusplus

/**
 * What runs a codec.
 */
enum ef_codec_engine
#if defined(__cplusplus) || __STDC_VERSION__ >= 202311L
  : uint32_t
#endif // defined(__cplusplus) || __STDC_VERSION__ >= 202311L
 {
  /**
   * The first engine that can do the request, in the platform's order.
   * The environment variable `EDGEFIRST_CODEC_ENGINE` (engine names,
   * comma-separated) narrows and orders the choice.
   */
  EF_CODEC_ENGINE_AUTO = 0,
  /**
   * Linux V4L2 memory-to-memory devices.
   */
  EF_CODEC_ENGINE_V4L2 = 1,
  /**
   * Windows Media Foundation transforms.
   */
  EF_CODEC_ENGINE_MEDIA_FOUNDATION = 2,
  /**
   * Apple VideoToolbox.
   */
  EF_CODEC_ENGINE_VIDEOTOOLBOX = 3,
  /**
   * NVIDIA nvJPEG, on Linux with CUDA.
   */
  EF_CODEC_ENGINE_NVJPEG = 4,
  /**
   * HAL's own CPU code, for still images only.
   */
  EF_CODEC_ENGINE_CPU = 5,
};
#ifndef __cplusplus
#if __STDC_VERSION__ >= 202311L
typedef enum ef_codec_engine ef_codec_engine;
#else
typedef uint32_t ef_codec_engine;
#endif // __STDC_VERSION__ >= 202311L
#endif // __cplusplus

/**
 * Encoding or decoding.
 */
enum ef_codec_direction
#if defined(__cplusplus) || __STDC_VERSION__ >= 202311L
  : uint32_t
#endif // defined(__cplusplus) || __STDC_VERSION__ >= 202311L
 {
  /**
   * Pixels to a compressed format.
   */
  EF_CODEC_DIRECTION_ENCODE = 0,
  /**
   * A compressed format to pixels.
   */
  EF_CODEC_DIRECTION_DECODE = 1,
};
#ifndef __cplusplus
#if __STDC_VERSION__ >= 202311L
typedef enum ef_codec_direction ef_codec_direction;
#else
typedef uint32_t ef_codec_direction;
#endif // __STDC_VERSION__ >= 202311L
#endif // __cplusplus

/**
 * How pixels move between a tensor and an engine.
 */
enum ef_codec_transfer
#if defined(__cplusplus) || __STDC_VERSION__ >= 202311L
  : uint32_t
#endif // defined(__cplusplus) || __STDC_VERSION__ >= 202311L
 {
  /**
   * The engine reads or writes the tensor's own memory.
   */
  EF_CODEC_TRANSFER_ZERO_COPY = 0,
  /**
   * One copy per frame on the GPU.
   */
  EF_CODEC_TRANSFER_GPU_COPY = 1,
  /**
   * One copy per frame through CPU memory.
   */
  EF_CODEC_TRANSFER_CPU_COPY = 2,
};
#ifndef __cplusplus
#if __STDC_VERSION__ >= 202311L
typedef enum ef_codec_transfer ef_codec_transfer;
#else
typedef uint32_t ef_codec_transfer;
#endif // __STDC_VERSION__ >= 202311L
#endif // __cplusplus

/**
 * H.264 profile of an encoder.
 */
enum ef_h264_profile
#if defined(__cplusplus) || __STDC_VERSION__ >= 202311L
  : uint32_t
#endif // defined(__cplusplus) || __STDC_VERSION__ >= 202311L
 {
  /**
   * High where the device offers it, else the device default.
   */
  EF_H264_PROFILE_AUTO = 0,
  /**
   * Baseline.
   */
  EF_H264_PROFILE_BASELINE = 1,
  /**
   * Constrained Baseline.
   */
  EF_H264_PROFILE_CONSTRAINED_BASELINE = 2,
  /**
   * Main.
   */
  EF_H264_PROFILE_MAIN = 3,
  /**
   * High.
   */
  EF_H264_PROFILE_HIGH = 4,
};
#ifndef __cplusplus
#if __STDC_VERSION__ >= 202311L
typedef enum ef_h264_profile ef_h264_profile;
#else
typedef uint32_t ef_h264_profile;
#endif // __STDC_VERSION__ >= 202311L
#endif // __cplusplus

/**
 * What [`ef_video_decoder_next_event`] returned.
 */
enum ef_video_event
#if defined(__cplusplus) || __STDC_VERSION__ >= 202311L
  : uint32_t
#endif // defined(__cplusplus) || __STDC_VERSION__ >= 202311L
 {
  /**
   * Nothing is ready: decode more, or flush.
   */
  EF_VIDEO_EVENT_NONE = 0,
  /**
   * A decoded frame, in presentation order.
   */
  EF_VIDEO_EVENT_FRAME = 1,
  /**
   * The stream's size or format changed, or became known; read it with
   * [`ef_video_decoder_stream_info`]. Frames after this have the new
   * format; frames already returned keep their memory.
   */
  EF_VIDEO_EVENT_RECONFIGURED = 2,
};
#ifndef __cplusplus
#if __STDC_VERSION__ >= 202311L
typedef enum ef_video_event ef_video_event;
#else
typedef uint32_t ef_video_event;
#endif // __STDC_VERSION__ >= 202311L
#endif // __cplusplus

/**
 * Whose memory a decoder writes its pictures to.
 */
enum ef_video_memory
#if defined(__cplusplus) || __STDC_VERSION__ >= 202311L
  : uint32_t
#endif // defined(__cplusplus) || __STDC_VERSION__ >= 202311L
 {
  /**
   * The pool's tensors, which the device imports, or which each picture
   * is copied into.
   */
  EF_VIDEO_MEMORY_IMPORTED = 0,
  /**
   * The device's own buffers, exported as DMA-BUFs because it would not
   * import the pool's. CPU reads of these may be uncached and slow.
   */
  EF_VIDEO_MEMORY_EXPORTED = 1,
};
#ifndef __cplusplus
#if __STDC_VERSION__ >= 202311L
typedef enum ef_video_memory ef_video_memory;
#else
typedef uint32_t ef_video_memory;
#endif // __STDC_VERSION__ >= 202311L
#endif // __cplusplus

/**
 * What the codec engines on this machine can do: a list of
 * `ef_codec_capability`, engine by engine, in the order
 * `EF_CODEC_ENGINE_AUTO` tries them.
 */
typedef struct ef_codec_capabilities ef_codec_capabilities;

/**
 * An opaque still-image encoder.
 */
typedef struct ef_image_encoder ef_image_encoder;

/**
 * An opaque video decoder.
 */
typedef struct ef_video_decoder ef_video_decoder;

/**
 * An opaque video encoder.
 */
typedef struct ef_video_encoder ef_video_encoder;

/**
 * The configuration of a video encoder.
 */
typedef struct ef_video_encoder_config ef_video_encoder_config;

/**
 * An opaque decoded frame. It holds its pool buffer until it is freed.
 */
typedef struct ef_video_frame ef_video_frame;

/**
 * One thing an engine on this machine can do.
 */
typedef struct ef_codec_capability {
  /**
   * The engine, an `ef_codec_engine`.
   */
  uint32_t engine;
  /**
   * The compressed format, an `ef_codec`.
   */
  uint32_t codec;
  /**
   * An `ef_codec_direction`.
   */
  uint32_t direction;
  /**
   * How frames move in the best case (an `ef_codec_transfer`), for a
   * source or pool in the memory the engine prefers. An open stream says
   * what it does in its own info.
   */
  uint32_t transfer;
  /**
   * The work runs on dedicated hardware rather than the CPU.
   */
  bool hardware;
  /**
   * The device within the engine: a device node, a transform's name, or
   * `"cpu"`.
   */
  const char *device;
  /**
   * The pixel formats of the images (`"NV12"`, `"YUYV"`): what an encoder
   * takes, or what a decoder produces.
   */
  const char *const *formats;
  /**
   * Entries in `formats`.
   */
  uintptr_t format_count;
} ef_codec_capability;

/**
 * A rectangle of an image, in pixels.
 */
typedef struct ef_codec_region {
  /**
   * Left edge.
   */
  uint32_t x;
  /**
   * Top edge.
   */
  uint32_t y;
  /**
   * Width.
   */
  uint32_t width;
  /**
   * Height.
   */
  uint32_t height;
} ef_codec_region;

/**
 * One encoded access unit.
 */
typedef struct ef_video_packet {
  /**
   * The bytes: Annex B for H.264, a JPEG file for MJPEG.
   */
  const uint8_t *data;
  /**
   * Bytes at `data`.
   */
  uintptr_t len;
  /**
   * The `pts` the source frame was passed with.
   */
  uint64_t pts;
  /**
   * The access unit is a key frame (an IDR picture).
   */
  bool keyframe;
} ef_video_packet;

/**
 * What runs an open encoder, and how its sources reach it.
 */
typedef struct ef_video_encoder_info {
  /**
   * The engine, an `ef_codec_engine`.
   */
  uint32_t engine;
  /**
   * How a whole source of the configured size, in the memory the engine
   * prefers (DMA-BUF; a D3D11 texture on Windows), reaches the encoder,
   * an `ef_codec_transfer`. Crops and host-memory sources may take a
   * copy where this says none. A Media Foundation hardware encoder
   * decides on its first frame whether it reads sources in place, and
   * reports `EF_CODEC_TRANSFER_GPU_COPY` until then.
   */
  uint32_t transfer;
  /**
   * The encoder is dedicated hardware rather than software the platform
   * provides.
   */
  bool hardware;
  /**
   * The device within the engine: a device node, or a transform's name.
   */
  const char *device;
} ef_video_encoder_info;

/**
 * Geometry and format of a stream's decoded pictures.
 */
typedef struct ef_video_stream_info {
  /**
   * Width of the visible picture in pixels.
   */
  uint32_t width;
  /**
   * Height of the visible picture in pixels.
   */
  uint32_t height;
  /**
   * The visible picture within the coded one.
   */
  struct ef_codec_region crop;
  /**
   * Width of the coded picture.
   */
  uint32_t coded_width;
  /**
   * Height of the coded picture.
   */
  uint32_t coded_height;
  /**
   * Pixel format of the decoded frames (`"NV12"`).
   */
  const char *format;
  /**
   * Colorimetry the device reports, packed as `ef_tensor_colorimetry`
   * returns it; 0 when it reports none.
   */
  uint32_t colorimetry;
  /**
   * Buffers in the output pool.
   */
  uint32_t buffers;
  /**
   * Whose memory the pictures are decoded into, an `ef_video_memory`.
   */
  uint32_t memory;
  /**
   * The engine that decodes, an `ef_codec_engine`.
   */
  uint32_t engine;
  /**
   * How each picture reaches its pool tensor, an `ef_codec_transfer`:
   * written in place (V4L2), or copied out of the decoder's own surfaces
   * on the GPU (Media Foundation with D3D11) or through CPU memory.
   */
  uint32_t transfer;
} ef_video_stream_info;

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

/**
 * ABI version of this library's C surface.
 *
 * 2: `ef_codec_colorimetry_from_v4l2` takes an `is_rgb_or_hsv` argument
 * before `out`.
 *
 * 3: engines, capabilities, `ef_image_encoder`, `ef_video_encoder` and
 * `ef_video_decoder`; `ef_codec_last_error`; image decoding answers
 * `EBADMSG` for undecodable data and `ENOTSUP` for unsupported features,
 * which were `EIO`.
 */
uint32_t ef_codec_abi_version(void);

/**
 * Create a decoder. `NULL` on failure.
 */
ef_image_decoder *ef_image_decoder_new(void);

/**
 * Free a decoder. Freeing `NULL` is a no-op, matching `free(3)`.
 *
 * # Safety
 * `d` must be `NULL` or have come from [`ef_image_decoder_new`].
 */
void ef_image_decoder_free(ef_image_decoder *d);

/**
 * Decode JPEG or PNG bytes into `dst`, sizing and formatting it to the image.
 *
 * The container is detected from magic bytes. `dst` must be large enough;
 * a smaller allocation is an error rather than a truncated image.
 *
 * @return 0 on success, `ENOSPC` when `dst` is too small, `EACCES` when the
 *         CPU decoder must write a `dst` not declared CPU-writable,
 *         `EBADMSG` for data that is not a decodable image, `ENOTSUP` for a
 *         feature or destination format the decoder does not support,
 *         otherwise an errno, with the reason in `ef_codec_last_error`.
 *
 * # Safety
 * `data` must be readable for `len` bytes; `dst` must be a live handle.
 */
int ef_image_decoder_decode_into(ef_image_decoder *d,
                                 const uint8_t *data,
                                 uintptr_t len,
                                 ef_tensor *dst);

/**
 * Decode an image file into `dst`.
 *
 * @return as [`ef_image_decoder_decode_into`], plus `ENOENT` for a missing file.
 *
 * # Safety
 * `path` must be a NUL-terminated string; `dst` must be a live handle.
 */
int ef_image_decoder_decode_file_into(ef_image_decoder *d, const char *path, ef_tensor *dst);

/**
 * Whether a hardware V4L2 JPEG decoder is present.
 */
int ef_codec_v4l2_available(void);

/**
 * Whether nvJPEG is present.
 */
int ef_codec_nvjpeg_available(void);

/**
 * Selects the engine that decodes JPEG, an `ef_codec_engine`:
 * `EF_CODEC_ENGINE_AUTO` (nvJPEG, V4L2, then the CPU, where present), or
 * one engine used alone (`EF_CODEC_ENGINE_CPU`, `EF_CODEC_ENGINE_V4L2`,
 * `EF_CODEC_ENGINE_NVJPEG`). PNG is decoded on the CPU under `AUTO` and
 * `CPU` only.
 *
 * @return 0, or `EINVAL`.
 *
 * # Safety
 * `d` must be a live decoder.
 */
int ef_image_decoder_set_engine(ef_image_decoder *d, uint32_t engine);

/**
 * Select the software JPEG IDCT kernel. `0` = accurate, `1` = fast.
 *
 * # Safety
 * `d` must be `NULL` or a live decoder.
 */
int ef_image_decoder_set_dct_method(ef_image_decoder *d, uint32_t method);

/**
 * Request a fused JPEG output format (`"rgb8"`, `"NV12"`, …). `NULL` resets.
 *
 * # Safety
 * `format` must be `NULL` or a NUL-terminated string.
 */
int ef_image_decoder_set_output_format(ef_image_decoder *d, const char *format);

/**
 * Reset fused JPEG output to the source's native format.
 *
 * # Safety
 * `d` must be `NULL` or a live decoder.
 */
int ef_image_decoder_reset_output_format(ef_image_decoder *d);

/**
 * Map raw V4L2 colorimetry integers to the packed `ef_tensor_colorimetry` form.
 *
 * `is_rgb_or_hsv` is non-zero when the negotiated format's pixel encoding
 * is RGB or HSV. Per `V4L2_MAP_QUANTIZATION_DEFAULT`, a `DEFAULT` (0)
 * `quant` is full range for an RGB/HSV format, whatever the colorspace, and
 * for `V4L2_COLORSPACE_JPEG`; limited for a YUV format in any other
 * recognised colorspace; and unset (range byte 0) for a YUV format in a
 * `DEFAULT` or unrecognised colorspace, so the at-use height heuristic
 * decides.
 *
 * # Safety
 * `out` must be writable.
 */
int ef_codec_colorimetry_from_v4l2(uint32_t colorspace,
                                   uint32_t xfer,
                                   uint32_t ycbcr_enc,
                                   uint32_t quant,
                                   int is_rgb_or_hsv,
                                   uint32_t *out);

/**
 * Encode a packed RGB/RGBA u8 tensor as JPEG.
 *
 * `quality` in 1–100; `0` or out-of-range uses 80.
 *
 * @return 0 on success, `EACCES` when `t` is not declared CPU-readable,
 *         otherwise an errno.
 *
 * # Safety
 * `path` must be a NUL-terminated string; `t` must be a live handle.
 */
int ef_codec_save_jpeg(const ef_tensor *t, const char *path, int quality);

/**
 * Creates an encoder for `codec`, an `ef_codec`. Only `EF_CODEC_JPEG` is
 * encoded; the quality starts at 85 and the engine at
 * `EF_CODEC_ENGINE_AUTO`.
 *
 * Under `EF_CODEC_ENGINE_AUTO` a V4L2 JPEG encoder takes DMA-BUF sources it
 * can read and the CPU takes the rest. The CPU encodes `u8` NV12, NV16,
 * YUYV, GREY, RGB and RGBA images at any row stride.
 *
 * @return the encoder, freed with [`ef_image_encoder_free`]; `NULL` with
 *         `errno` `ENOTSUP` for a codec it does not encode or `EINVAL` for
 *         a value outside `ef_codec`.
 */
struct ef_image_encoder *ef_image_encoder_new(uint32_t codec);

/**
 * Frees an encoder. Freeing `NULL` is a no-op.
 *
 * # Safety
 * `e` must be `NULL` or have come from [`ef_image_encoder_new`].
 */
void ef_image_encoder_free(struct ef_image_encoder *e);

/**
 * Selects the engine, an `ef_codec_engine`: `EF_CODEC_ENGINE_AUTO`, or
 * one engine used alone (`EF_CODEC_ENGINE_CPU`, `EF_CODEC_ENGINE_V4L2`).
 *
 * @return 0, or `EINVAL`.
 *
 * # Safety
 * `e` must be a live encoder.
 */
int ef_image_encoder_set_engine(struct ef_image_encoder *e, uint32_t engine);

/**
 * Sets the JPEG quality, 1 (smallest) to 100 (best).
 *
 * @return 0, or `EINVAL` outside 1 to 100.
 *
 * # Safety
 * `e` must be a live encoder.
 */
int ef_image_encoder_set_quality(struct ef_image_encoder *e, int quality);

/**
 * Encodes `src` and points `*data` and `*len` at the file.
 *
 * The bytes belong to the encoder and stay valid until its next encode or
 * until it is freed; the next encode reuses the buffer.
 *
 * @return 0; `EACCES` when the CPU must read a `src` not declared
 *         CPU-readable; `ENOTSUP` for a source format or element type no
 *         engine takes; `EINVAL` for a bad argument or source (for example
 *         an odd-width YUYV image); `ENODEV` when the engine selected by
 *         name is not present; otherwise an errno, with the reason in
 *         [`ef_codec_last_error`](crate::error::ef_codec_last_error).
 *
 * # Safety
 * `e` must be a live encoder, `src` a live tensor handle, and `data` and
 * `len` writable.
 */
int ef_image_encoder_encode(struct ef_image_encoder *e,
                            const ef_tensor *src,
                            const uint8_t **data,
                            uintptr_t *len);

/**
 * The usual name of an `ef_codec` value (`"H.264"`, `"JPEG"`), or `NULL`
 * for a value outside the enum. The string is static.
 */
const char *ef_codec_name(uint32_t codec);

/**
 * The name of an `ef_codec_engine` value, as `EDGEFIRST_CODEC_ENGINE` and
 * [`ef_codec_engine_from_name`] take it (`"v4l2"`, `"media-foundation"`),
 * or `NULL` for a value outside the enum. The string is static.
 */
const char *ef_codec_engine_name(uint32_t engine);

/**
 * Parses an engine name, ignoring case (`"mf"` and `"vt"` are accepted for
 * Media Foundation and VideoToolbox), into `*out`.
 *
 * @return 0, or `EINVAL` for an unknown name.
 *
 * # Safety
 * `name` must be a NUL-terminated string and `out` writable.
 */
int ef_codec_engine_from_name(const char *name, uint32_t *out);

/**
 * Probes the codec engines on this machine.
 *
 * Probing opens no stream, but it opens device nodes and enumerates (on
 * Windows, activates) transforms, so it is not free: call it once.
 *
 * @return the list, freed with [`ef_codec_capabilities_free`]; `NULL` only
 *         if allocation fails.
 */
struct ef_codec_capabilities *ef_codec_capabilities_new(void);

/**
 * Entries in the list; 0 for `NULL`.
 *
 * # Safety
 * `list` must be `NULL` or a live list.
 */
uintptr_t ef_codec_capabilities_count(const struct ef_codec_capabilities *list);

/**
 * Entry `index`, valid until the list is freed, or `NULL` past the end.
 *
 * # Safety
 * `list` must be `NULL` or a live list.
 */
const struct ef_codec_capability *ef_codec_capabilities_get(const struct ef_codec_capabilities *list,
                                                            uintptr_t index);

/**
 * Frees a list. Freeing `NULL` is a no-op.
 *
 * # Safety
 * `list` must be `NULL` or have come from [`ef_codec_capabilities_new`].
 */
void ef_codec_capabilities_free(struct ef_codec_capabilities *list);

/**
 * Why the calling thread's last failed codec call failed, or `NULL` when
 * its last call succeeded.
 *
 * The text is for people, not for parsing. The pointer stays valid until
 * the same thread's next codec call.
 */
const char *ef_codec_last_error(void);

/**
 * Creates an encoder configuration for `codec` (an `ef_codec`): sources of
 * `width` x `height` pixels in pixel format `format` (`"NV12"`, `"YUYV"`)
 * at a nominal `frame_rate`, on `EF_CODEC_ENGINE_AUTO`, with an automatic
 * bitrate, profile and level, the device's default GOP, and the parameter
 * sets repeated on every key frame.
 *
 * @return the configuration, freed with
 *         [`ef_video_encoder_config_free`]; `NULL` with `errno` `EINVAL`
 *         for a value outside `ef_codec` or an unknown pixel format.
 *
 * # Safety
 * `format` must be a NUL-terminated string.
 */
struct ef_video_encoder_config *ef_video_encoder_config_new(uint32_t codec,
                                                            uint32_t width,
                                                            uint32_t height,
                                                            const char *format,
                                                            double frame_rate);

/**
 * Frees a configuration. Freeing `NULL` is a no-op.
 *
 * # Safety
 * `c` must be `NULL` or have come from [`ef_video_encoder_config_new`].
 */
void ef_video_encoder_config_free(struct ef_video_encoder_config *c);

/**
 * Selects the engine, an `ef_codec_engine`.
 *
 * @return 0, or `EINVAL`.
 *
 * # Safety
 * `c` must be a live configuration.
 */
int ef_video_encoder_config_set_engine(struct ef_video_encoder_config *c, uint32_t engine);

/**
 * Sets a constant target bitrate in bits per second; 0 chooses one from
 * the frame size and rate (0.16 bits per pixel, within 1 to 100 Mbps).
 *
 * @return 0, or `EINVAL`.
 *
 * # Safety
 * `c` must be a live configuration.
 */
int ef_video_encoder_config_set_bitrate(struct ef_video_encoder_config *c, uint32_t bps);

/**
 * Sets the frames from one key frame to the next; 0 keeps the device
 * default.
 *
 * @return 0, or `EINVAL`.
 *
 * # Safety
 * `c` must be a live configuration.
 */
int ef_video_encoder_config_set_gop(struct ef_video_encoder_config *c, uint32_t gop);

/**
 * Whether to emit the parameter sets with every key frame, so a receiver
 * can start at any key frame (on by default). The first frame always
 * carries them.
 *
 * @return 0, or `EINVAL`.
 *
 * # Safety
 * `c` must be a live configuration.
 */
int ef_video_encoder_config_set_repeat_headers(struct ef_video_encoder_config *c, bool repeat);

/**
 * Sets the H.264 profile, an `ef_h264_profile`.
 *
 * @return 0, or `EINVAL`.
 *
 * # Safety
 * `c` must be a live configuration.
 */
int ef_video_encoder_config_set_h264_profile(struct ef_video_encoder_config *c, uint32_t profile);

/**
 * Sets the H.264 level: the level number times ten (`41` for 4.1), `9`
 * for level 1b, or 0 for the lowest level that covers the size, frame
 * rate and bitrate.
 *
 * @return 0, or `EINVAL` for a level H.264 does not define.
 *
 * # Safety
 * `c` must be a live configuration.
 */
int ef_video_encoder_config_set_h264_level(struct ef_video_encoder_config *c, uint32_t level);

/**
 * Sets the JPEG quality of MJPEG pictures, 1 (smallest) to 100 (best);
 * 0 uses the default, 85.
 *
 * @return 0, or `EINVAL` above 100.
 *
 * # Safety
 * `c` must be a live configuration.
 */
int ef_video_encoder_config_set_quality(struct ef_video_encoder_config *c, uint32_t quality);

/**
 * Opens an encoder for the configuration, on the first engine to try that
 * runs the codec.
 *
 * Platforms: Linux (V4L2 memory-to-memory encoders), Windows (Media
 * Foundation hardware encoder transforms). Elsewhere there is no engine
 * and this fails with `ENODEV`.
 *
 * On Linux the first V4L2 device that produces the codec and accepts the
 * input format is used (`EDGEFIRST_CODEC_V4L2_ENCODER` names one device
 * node). On Windows the first hardware encoder transform that takes the
 * input format is used (`EDGEFIRST_CODEC_MF_ENCODER` keeps only the
 * transforms whose name contains its value). There is no software video
 * encoder.
 *
 * @return the encoder, freed with [`ef_video_encoder_free`]; `NULL` with
 *         `errno` `EINVAL` for an invalid configuration or one the device
 *         rejects, `ENOTSUP` when no engine here encodes the codec or takes
 *         the input format, or `ENODEV` when there is no encoder.
 *
 * # Safety
 * `c` must be a live configuration; it is not kept.
 */
struct ef_video_encoder *ef_video_encoder_new(const struct ef_video_encoder_config *c);

/**
 * Frees an encoder, dropping access units not yet pulled. Freeing `NULL` is
 * a no-op.
 *
 * # Safety
 * `e` must be `NULL` or have come from [`ef_video_encoder_new`].
 */
void ef_video_encoder_free(struct ef_video_encoder *e);

/**
 * Encodes `src`, queueing any access unit that is ready.
 *
 * `src` must have the configured pixel format and, without `crop`, the
 * configured size. `crop`, when not `NULL`, is the part of `src` to
 * encode, exactly the configured size. DMA-BUF and D3D11 sources are read
 * in place: the call returns once the device has finished reading `src`,
 * so the caller may reuse it afterwards. Every source of one stream must
 * have the same row stride. `pts` is returned unchanged on the access
 * unit; `flags` is 0 or `EF_VIDEO_ENCODE_KEYFRAME`.
 *
 * Platforms: Linux, Windows.
 *
 * @return 0; `EINVAL` for a source that does not match the configuration;
 *         `ETIMEDOUT` when the device does not finish in time; otherwise an
 *         errno, with the reason in `ef_codec_last_error`.
 *
 * # Safety
 * `e` must be a live encoder, `src` a live tensor handle, and `crop`
 * `NULL` or readable.
 */
int ef_video_encoder_encode(struct ef_video_encoder *e,
                            const ef_tensor *src,
                            uint64_t pts,
                            uint32_t flags,
                            const struct ef_codec_region *crop);

/**
 * Drains the encoder, queueing every access unit not yet queued. The
 * encoder accepts new frames afterwards.
 *
 * @return 0, or an errno.
 *
 * # Safety
 * `e` must be a live encoder.
 */
int ef_video_encoder_flush(struct ef_video_encoder *e);

/**
 * The next queued access unit, in decode order, or `NULL` when none is
 * queued. It belongs to the encoder and stays valid until the next call of
 * this function or until the encoder is freed.
 *
 * # Safety
 * `e` must be `NULL` or a live encoder.
 */
const struct ef_video_packet *ef_video_encoder_next_packet(struct ef_video_encoder *e);

/**
 * The engine and device that encode, and how sources reach them, valid
 * until the next call of this function or until the encoder is freed;
 * `NULL` for a `NULL` encoder.
 *
 * # Safety
 * `e` must be `NULL` or a live encoder.
 */
const struct ef_video_encoder_info *ef_video_encoder_get_info(struct ef_video_encoder *e);

/**
 * Opens a decoder for `codec` (an `ef_codec`) on `engine` (an
 * `ef_codec_engine`), which allocates its own frames: DMA-BUF tensors on
 * Linux, D3D11 textures on Windows (host memory where the process has no
 * D3D11 device), allocated again when the stream changes size.
 *
 * `pool_count` is the frames the caller may hold at once, 4 if 0; the
 * device's own reference pictures and one buffer to decode into are added
 * to it.
 * A V4L2 device takes a bounded number of buffers (31 on the i.MX 95's
 * Wave6, its own included); a count beyond that makes
 * [`ef_video_decoder_decode`] fail with `EINVAL` once the stream's format is
 * known.
 *
 * Platforms: Linux (V4L2 memory-to-memory decoders), Windows (Media
 * Foundation: Microsoft's H.264 decoder with DXVA, and MJPEG decoders).
 * Elsewhere there is no engine and this fails with `ENODEV`.
 *
 * @return the decoder, freed with [`ef_video_decoder_free`]; `NULL` with
 *         `errno` `ENOTSUP` when no engine here decodes the codec, `ENODEV`
 *         when there is no decoder, or `EINVAL`.
 */
struct ef_video_decoder *ef_video_decoder_new(uint32_t codec,
                                              uint32_t engine,
                                              uintptr_t pool_count);

/**
 * Opens a decoder that decodes into the caller's tensors, one buffer each.
 * Each must be a different buffer: a pool that holds one buffer twice, as
 * the same handle or two handles over it, is `EINVAL`.
 *
 * On success the decoder takes over the caller's reference to each tensor:
 * the caller must not free them. A caller that keeps using one takes its
 * own reference first with `ef_tensor_retain`; while that reference is
 * held, the decoder's CPU writes into the tensor (software decoders) are
 * refused. On failure the caller keeps its references.
 *
 * On Windows pictures are copied into the pool, and each tensor must be an
 * NV12 image of exactly the stream's visible size; such a pool cannot
 * follow a change of size. On Linux the device imports the tensors'
 * DMA-BUFs: there must be at least one more than the device keeps for
 * reference pictures, each large enough for the coded picture at the row
 * stride the device asks for, or decoding fails with `EINVAL` once the
 * stream's size is known.
 *
 * Platforms: Linux, Windows.
 *
 * @return as [`ef_video_decoder_new`].
 *
 * # Safety
 * `pool` must point to `count` live tensor handles.
 */
struct ef_video_decoder *ef_video_decoder_new_with_pool(uint32_t codec,
                                                        uint32_t engine,
                                                        ef_tensor *const *pool,
                                                        uintptr_t count);

/**
 * Frees a decoder, dropping events not yet pulled. Frames already pulled
 * stay valid until they are freed. Freeing `NULL` is a no-op.
 *
 * # Safety
 * `d` must be `NULL` or have come from [`ef_video_decoder_new`] or
 * [`ef_video_decoder_new_with_pool`].
 */
void ef_video_decoder_free(struct ef_video_decoder *d);

/**
 * Decodes one access unit, queueing every frame and format change that is
 * ready.
 *
 * Access units go in decode order: for H.264 an Annex B access unit (see
 * `edgefirst_codec::video::access_units` for splitting a byte stream), for
 * MJPEG one JPEG picture. `pts` is returned unchanged on the frame.
 *
 * Platforms: Linux, Windows.
 *
 * @return 0; `EAGAIN` when the caller holds every frame of the pool (free
 *         some, then decode the same access unit again); `EBADMSG` for data
 *         the decoder cannot parse; `EINVAL` for a provided pool that does
 *         not fit the stream; `ETIMEDOUT` when the device does not finish
 *         in time; otherwise an errno, with the reason in
 *         `ef_codec_last_error`.
 *
 * # Safety
 * `d` must be a live decoder and `data` readable for `len` bytes.
 */
int ef_video_decoder_decode(struct ef_video_decoder *d,
                            const uint8_t *data,
                            uintptr_t len,
                            uint64_t pts);

/**
 * Drains the decoder at the end of the stream, queueing every frame not yet
 * queued.
 *
 * @return 0, or an errno.
 *
 * # Safety
 * `d` must be a live decoder.
 */
int ef_video_decoder_flush(struct ef_video_decoder *d);

/**
 * Pulls the next queued event into `*kind`, an `ef_video_event`.
 *
 * For `EF_VIDEO_EVENT_FRAME`, `*frame` receives the frame, which the caller
 * frees with [`ef_video_frame_free`] to give its buffer back to the pool;
 * otherwise `*frame` is `NULL`. For `EF_VIDEO_EVENT_RECONFIGURED`,
 * [`ef_video_decoder_stream_info`] now describes the frames that follow.
 * `EF_VIDEO_EVENT_NONE` means the queue is empty.
 *
 * @return 0, or `EINVAL`.
 *
 * # Safety
 * `d` must be a live decoder; `kind` and `frame` writable.
 */
int ef_video_decoder_next_event(struct ef_video_decoder *d,
                                uint32_t *kind,
                                struct ef_video_frame **frame);

/**
 * The stream's geometry and format as of the last
 * `EF_VIDEO_EVENT_RECONFIGURED` pulled, valid until the next one is pulled
 * or the decoder is freed; `NULL` before the first.
 *
 * # Safety
 * `d` must be `NULL` or a live decoder.
 */
const struct ef_video_stream_info *ef_video_decoder_stream_info(const struct ef_video_decoder *d);

/**
 * The decoded picture: an image tensor of the visible size, `NULL` for a
 * `NULL` frame.
 *
 * The handle belongs to the frame and is valid until the frame is freed;
 * do not free it, or change its shape or format. Pass it to any EdgeFirst
 * library as a source. A caller that keeps it past the frame takes its own
 * reference with `ef_tensor_retain`, but the pool may decode a later
 * picture into the same memory once the frame is freed.
 *
 * # Safety
 * `f` must be `NULL` or a live frame.
 */
ef_tensor *ef_video_frame_tensor(const struct ef_video_frame *f);

/**
 * The `pts` passed with the frame's access unit; 0 for `NULL`.
 *
 * # Safety
 * `f` must be `NULL` or a live frame.
 */
uint64_t ef_video_frame_pts(const struct ef_video_frame *f);

/**
 * Whether the device reported an error decoding this picture, so it may be
 * partly wrong (a missing reference, a damaged slice).
 *
 * # Safety
 * `f` must be `NULL` or a live frame.
 */
bool ef_video_frame_corrupted(const struct ef_video_frame *f);

/**
 * Frees a frame, giving its buffer back to the decoder's pool. Freeing
 * `NULL` is a no-op.
 *
 * # Safety
 * `f` must be `NULL` or have come from [`ef_video_decoder_next_event`].
 */
void ef_video_frame_free(struct ef_video_frame *f);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* EDGEFIRST_CODEC_H */
