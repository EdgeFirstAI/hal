#ifndef EDGEFIRST_IMAGE_H
#define EDGEFIRST_IMAGE_H

/**
 * @file image.h
 * @brief EdgeFirst image processing C API
 *
 * SPDX-License-Identifier: Apache-2.0
 * Copyright (c) 2026 Au-Zone Technologies. All Rights Reserved.
 *
 * Hardware-accelerated image processing: conversion, scaling and colour
 * handling across GL, G2D and CPU backends.
 *
 * This library MINTS tensors. `ef_image_processor_create_image` returns an
 * `ef_tensor`, and allocating a PBO is the one operation that genuinely
 * requires a processor, because only the GL context owner can create one.
 *
 * A tensor from here is an ordinary `ef_tensor`. There is one tensor
 * header, `edgefirst/tensor.h`. Detection primitives used when drawing
 * masks live in header-only `edgefirst/detect.h`.
 *
 * Platform-specific entry points. Both are declared on every platform and
 * refuse at run time off their platform, so linking never depends on the
 * host. The tie is stated per function with a "Platforms:" line:
 *   - Linux, macOS, iOS, Android: ef_image_processor_convert_fence
 *     (sync-fence fd)
 *   - Windows: ef_image_processor_convert_fence_handle (event handle)
 */

#include <stdarg.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>

#include "edgefirst/detect.h"
#include "edgefirst/tensor.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief An opaque image processor.
 *
 * Forward-declared so callers write `ef_image_processor *` without the `struct`
 * keyword. Its definition is private: processor state is not part of the ABI.
 */
typedef struct ef_image_processor ef_image_processor;

/**
 * @brief Opaque convert options.
 *
 * Built with `ef_convert_options_new` and the `ef_convert_options_set_*`
 * setters, freed with `ef_convert_options_free`. `NULL` where options are
 * accepted means the defaults.
 */
typedef struct ef_convert_options ef_convert_options;

/** @brief An owned list of materialized masks. */
typedef struct ef_mask_list ef_mask_list;

/** @brief Opaque list of tile specs from `ef_tile_grid`. */
typedef struct ef_tile_spec_list ef_tile_spec_list;

/** @brief Opaque list of tile placements from plan/tile_into. */
typedef struct ef_tile_placement_list ef_tile_placement_list;


/* Generated with cbindgen:0.29.4 */

/* WARNING: The generated portion of this file is produced by cbindgen. Do not modify it directly. */

#include <stdarg.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>

/**
 * A quarter-turn rotation, for `ef_image_processor_convert` and friends.
 */
enum ef_rotation
#if defined(__cplusplus) || __STDC_VERSION__ >= 202311L
  : uint32_t
#endif // defined(__cplusplus) || __STDC_VERSION__ >= 202311L
 {
  EF_ROTATION_NONE = 0,
  EF_ROTATION_CLOCKWISE90 = 1,
  EF_ROTATION_ROTATE180 = 2,
  EF_ROTATION_COUNTER_CLOCKWISE90 = 3,
};
#ifndef __cplusplus
#if __STDC_VERSION__ >= 202311L
typedef enum ef_rotation ef_rotation;
#else
typedef uint32_t ef_rotation;
#endif // __STDC_VERSION__ >= 202311L
#endif // __cplusplus

/**
 * A mirror, for `ef_image_processor_convert` and friends.
 */
enum ef_flip
#if defined(__cplusplus) || __STDC_VERSION__ >= 202311L
  : uint32_t
#endif // defined(__cplusplus) || __STDC_VERSION__ >= 202311L
 {
  EF_FLIP_NONE = 0,
  /**
   * Mirror top to bottom.
   */
  EF_FLIP_VERTICAL = 1,
  /**
   * Mirror left to right.
   */
  EF_FLIP_HORIZONTAL = 2,
};
#ifndef __cplusplus
#if __STDC_VERSION__ >= 202311L
typedef enum ef_flip ef_flip;
#else
typedef uint32_t ef_flip;
#endif // __STDC_VERSION__ >= 202311L
#endif // __cplusplus

/**
 * How a mask's palette colour is chosen, for the draw functions.
 */
enum ef_color_mode
#if defined(__cplusplus) || __STDC_VERSION__ >= 202311L
  : uint32_t
#endif // defined(__cplusplus) || __STDC_VERSION__ >= 202311L
 {
  /**
   * By class label.
   */
  EF_COLOR_MODE_CLASS = 0,
  /**
   * By detection index.
   */
  EF_COLOR_MODE_INSTANCE = 1,
  /**
   * By track ID.
   */
  EF_COLOR_MODE_TRACK = 2,
};
#ifndef __cplusplus
#if __STDC_VERSION__ >= 202311L
typedef enum ef_color_mode ef_color_mode;
#else
typedef uint32_t ef_color_mode;
#endif // __STDC_VERSION__ >= 202311L
#endif // __cplusplus

/**
 * The backend `ef_image_processor_new_with_backend` forces.
 */
enum ef_compute_backend
#if defined(__cplusplus) || __STDC_VERSION__ >= 202311L
  : uint32_t
#endif // defined(__cplusplus) || __STDC_VERSION__ >= 202311L
 {
  /**
   * Auto-detect, with the fallback chain.
   */
  EF_COMPUTE_BACKEND_AUTO = 0,
  /**
   * CPU only.
   */
  EF_COMPUTE_BACKEND_CPU = 1,
  /**
   * G2D only.
   */
  EF_COMPUTE_BACKEND_G2D = 2,
  /**
   * OpenGL only.
   */
  EF_COMPUTE_BACKEND_OPEN_GL = 3,
};
#ifndef __cplusplus
#if __STDC_VERSION__ >= 202311L
typedef enum ef_compute_backend ef_compute_backend;
#else
typedef uint32_t ef_compute_backend;
#endif // __STDC_VERSION__ >= 202311L
#endif // __cplusplus

/**
 * A named normalization, `(x - mean) / std` on 0-255 values, for
 * `ef_convert_options_set_normalization_preset`.
 */
enum ef_normalization
#if defined(__cplusplus) || __STDC_VERSION__ >= 202311L
  : uint32_t
#endif // defined(__cplusplus) || __STDC_VERSION__ >= 202311L
 {
  /**
   * `x / 255`, giving `[0, 1]`.
   */
  EF_NORMALIZATION_UNSIGNED = 0,
  /**
   * `x / 127.5 - 1`, giving `[-1, 1]`.
   */
  EF_NORMALIZATION_SIGNED = 1,
  /**
   * `x`, giving `[0, 255]`.
   */
  EF_NORMALIZATION_RAW = 2,
  /**
   * ImageNet mean and standard deviation.
   */
  EF_NORMALIZATION_IMAGENET = 3,
};
#ifndef __cplusplus
#if __STDC_VERSION__ >= 202311L
typedef enum ef_normalization ef_normalization;
#else
typedef uint32_t ef_normalization;
#endif // __STDC_VERSION__ >= 202311L
#endif // __cplusplus

/**
 * Static tiling geometry. Seed with [`ef_tiling_config_default`].
 *
 * How each tile is converted (fit, letterbox placement and pad,
 * normalization) is the `const ef_convert_options *` the tiling functions
 * take beside it; `NULL` stretches each tile with the default mapping.
 */
typedef struct ef_tiling_config {
  uintptr_t tile_w;
  uintptr_t tile_h;
  float overlap_ratio;
} ef_tiling_config;

/**
 * One tile's native-frame crop and grid coordinates.
 */
typedef struct ef_tile_spec {
  uint32_t x;
  uint32_t y;
  uint32_t width;
  uint32_t height;
  uintptr_t index;
  uintptr_t row;
  uintptr_t col;
} ef_tile_spec;

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

/**
 * ABI version of this library's C surface.
 *
 * 4: `ef_tiling_config` lost `pad` and `fit` (24 bytes, was 32) and
 * `ef_image_processor_plan_tiles`, `ef_image_processor_tile_into` and
 * `ef_image_processor_tile_one` take a trailing `const ef_convert_options *`
 * that says how each tile is converted. `EF_FIT_*` and
 * `EF_LETTERBOX_PLACEMENT_*` moved to `edgefirst/detect.h` with their values
 * unchanged, `ef_image_processor_set_interpolation` accepts
 * `EF_INTERPOLATION_BICUBIC` and `EF_INTERPOLATION_AREA`, and
 * `ef_convert_options_set_interpolation` sets a per-convert interpolation.
 *
 * 3: the convert functions take a `const ef_convert_options *` (built with
 * the `ef_convert_options_*` setters) in place of `rotation`, `flip` and a
 * `const ef_crop *`, and `ef_crop` is gone. A caller built against
 * generation 2 fails to compile; one linked against it passes arguments the
 * functions no longer read the same way.
 *
 * 2: an unknown `color_mode` passed to `ef_image_processor_draw_decoded_masks`
 * or `ef_image_processor_draw_proto_masks` returns `EINVAL` (it was drawn as
 * `EF_COLOR_MODE_CLASS`), and an unknown `ef_tiling_config.fit` fails the
 * tiling functions (any non-zero value meant letterbox). No symbol or layout
 * changed, so a caller built against generation 1 gets no link error; the
 * probe is the only signal.
 */
uint32_t ef_image_abi_version(void);

/**
 * Draw boxes and decoded masks onto `dst`.
 *
 * On Windows the destination's `ef_tensor_gpu_completion` reflects this draw
 * afterwards, as it does after a convert, so another device can wait on the
 * drawn frame instead of the CPU.
 *
 * A `BGRA` background onto a zero-copy destination (a D3D11 texture, an
 * IOSurface, a dma-buf) returns `ENOTSUP`: the GL base-layer draw has no
 * `BGRA` arm and the CPU backend renders only `RGBA`/`RGB`.
 *
 * `content_region` is `NULL` or four floats: where the image sits in the
 * model input as normalized `[xmin, ymin, xmax, ymax]`
 * (`ef_convert_options_content_region`), used to map boxes and masks back
 * to the image.
 *
 * `color_mode` is an `EF_COLOR_MODE_*` code; any other value is `EINVAL`.
 * A CPU draw blends into `dst`, so one not declared `EF_CPU_ACCESS_READ_WRITE`
 * (or a `background` not declared readable) is `EACCES`.
 *
 * @return 0 on success, `EINVAL` for an invalid argument or aliased
 * buffers, `ENOTSUP` for a format no backend draws, `EACCES` as above,
 * otherwise `EIO` or the backend's errno.
 *
 * # Safety
 * Pointers must be live or NULL as documented.
 */
int ef_image_processor_draw_decoded_masks(ef_image_processor *p,
                                          ef_tensor *dst,
                                          const ef_detect_box *boxes,
                                          uintptr_t n_boxes,
                                          const ef_segmentation *masks,
                                          uintptr_t n_masks,
                                          const ef_tensor *background,
                                          float opacity,
                                          const float *content_region,
                                          uint32_t color_mode);

/**
 * Draw proto masks onto `dst`. `protos` and `coeffs` are borrowed, not taken.
 *
 * On Windows the destination's `ef_tensor_gpu_completion` reflects this draw
 * afterwards, as it does after a convert.
 *
 * The same `BGRA` background restriction, `content_region`, `color_mode`
 * validation, `EACCES` rule and return codes as
 * `ef_image_processor_draw_decoded_masks`.
 *
 * # Safety
 * Tensor handles must stay live for the call.
 */
int ef_image_processor_draw_proto_masks(ef_image_processor *p,
                                        ef_tensor *dst,
                                        const ef_detect_box *boxes,
                                        uintptr_t n_boxes,
                                        ef_tensor *protos,
                                        ef_tensor *coeffs,
                                        uint32_t layout,
                                        const ef_tensor *background,
                                        float opacity,
                                        const float *content_region,
                                        uint32_t color_mode);

/**
 * Materialize per-instance masks from proto tensors. Caller frees the list.
 * `content_region` as for `ef_image_processor_draw_decoded_masks`.
 *
 * # Safety
 * Handles must be live.
 */
ef_mask_list *ef_image_processor_materialize_masks(ef_image_processor *p,
                                                   const ef_detect_box *boxes,
                                                   uintptr_t n_boxes,
                                                   ef_tensor *protos,
                                                   ef_tensor *coeffs,
                                                   uint32_t layout,
                                                   const float *content_region);

/**
 * Number of masks. Zero for NULL.
 *
 * # Safety
 * `l` must be `NULL` or a live handle from this library.
 */
uintptr_t ef_mask_list_len(const ef_mask_list *l);

/**
 * Borrow masks as `ef_segmentation` values. Valid until the list is freed.
 *
 * # Safety
 * `l` must be `NULL` or a live handle from this library.
 */
const ef_segmentation *ef_mask_list_data(ef_mask_list *l);

/**
 * Free a mask list. NULL is a no-op.
 *
 * # Safety
 * `l` must be `NULL` or have come from this library.
 */
void ef_mask_list_free(ef_mask_list *l);

/**
 * Create a processor, probing the platform's converters. `NULL` on failure.
 */
ef_image_processor *ef_image_processor_new(void);

/**
 * Free a processor. Freeing `NULL` is a no-op, matching `free(3)`.
 *
 * # Safety
 * `p` must be `NULL` or have come from [`ef_image_processor_new`].
 */
void ef_image_processor_free(ef_image_processor *p);

/**
 * Allocate an image, returning a tensor.
 *
 * `format` is the wire descriptor (`"NV12"`, `"rgb8"`), matching
 * `ef_tensor_builder_format` rather than introducing a second vocabulary.
 * `dtype` and `access` are the shared integer codes.
 *
 * `storage` selects the backing; pass `EF_STORAGE_KIND_PBO` for the case that
 * actually needs a processor. The result is an ordinary `ef_tensor` — read it
 * with `ef_tensor_shape`, release it with `ef_tensor_free`.
 *
 * The returned tensor's `ef_tensor_cpu_access` may be wider than `access`:
 * what this processor's own convert may do to it as a destination on the
 * CPU is added, `EF_CPU_ACCESS_WRITE` when it may fill the whole image and
 * `EF_CPU_ACCESS_READ_WRITE` for passes that read back or write part of it
 * (`EF_DTYPE_I8`, a CPU `bgra8` swizzle, a G2D `rgb8` letterbox border). A
 * PBO image is always `EF_CPU_ACCESS_READ_WRITE`. Nothing is added for use
 * as a convert source or for a CPU mask draw; declare those yourself.
 *
 * @return a tensor the caller owns, or `NULL`.
 *
 * # Safety
 * `format` must be `NULL` or a NUL-terminated string.
 */
ef_tensor *ef_image_processor_create_image(ef_image_processor *p,
                                           uintptr_t width,
                                           uintptr_t height,
                                           const char *format,
                                           uint32_t dtype,
                                           uint32_t storage,
                                           uint32_t access);

/**
 * Allocate the image an `ef_tensor_image_desc` request describes.
 *
 * The request comes from `libedgefirst-tensor`, the type's single
 * implementation home (see `edgefirst_tensor_capi::desc`); this library
 * never dereferences the handle itself, only the scalar view
 * `ef_tensor_image_desc_get` fills. The request is not consumed and may be
 * reused, so one description can fill a pool.
 *
 * The declared access is widened as for `ef_image_processor_create_image`,
 * except for a compression request, which keeps `EF_CPU_ACCESS_NONE`.
 *
 * @return a tensor the caller owns, or `NULL`.
 *
 * # Safety
 * `p` and `d` must be live.
 */
ef_tensor *ef_image_processor_create_image_desc(ef_image_processor *p,
                                                const ef_tensor_image_desc *d);

/**
 * New default convert options. `NULL` on allocation failure. Free with
 * [`ef_convert_options_free`].
 */
ef_convert_options *ef_convert_options_new(void);

/**
 * Free options from [`ef_convert_options_new`]. `NULL` is a no-op.
 *
 * # Safety
 * `o` must be `NULL` or a live handle, not used afterwards.
 */
void ef_convert_options_free(ef_convert_options *o);

/**
 * Rotate the output: an `EF_ROTATION_*` code.
 *
 * @return 0, or `EINVAL` for an unknown code or `NULL`.
 *
 * # Safety
 * `o` must be `NULL` or a live handle.
 */
int ef_convert_options_set_rotation(ef_convert_options *o, uint32_t rotation);

/**
 * Flip the output: an `EF_FLIP_*` code.
 *
 * @return 0, or `EINVAL` for an unknown code or `NULL`.
 *
 * # Safety
 * `o` must be `NULL` or a live handle.
 */
int ef_convert_options_set_flip(ef_convert_options *o, uint32_t flip);

/**
 * Sample only this rectangle of the source, in pixels. A zero `width` or
 * `height` samples the whole source.
 *
 * @return 0, or `EINVAL` for `NULL`.
 *
 * # Safety
 * `o` must be `NULL` or a live handle.
 */
int ef_convert_options_set_source(ef_convert_options *o,
                                  uint32_t x,
                                  uint32_t y,
                                  uint32_t width,
                                  uint32_t height);

/**
 * Letterbox: keep the source aspect ratio and pad the rest with the RGBA
 * pixel `pad`, mapped like the image. `NULL` stretches instead.
 *
 * @return 0, or `EINVAL` for a `NULL` handle.
 *
 * # Safety
 * `o` must be `NULL` or a live handle; `pad` `NULL` or four bytes.
 */
int ef_convert_options_set_letterbox(ef_convert_options *o, const uint8_t *pad);

/**
 * Where a letterboxed image sits: an `EF_LETTERBOX_PLACEMENT_*` code.
 *
 * @return 0, or `EINVAL` for an unknown code or `NULL`.
 *
 * # Safety
 * `o` must be `NULL` or a live handle.
 */
int ef_convert_options_set_letterbox_placement(ef_convert_options *o, uint32_t placement);

/**
 * Pad a letterbox with these four values as written to the destination
 * (after normalization; quantized for integer destinations) instead of
 * mapping the pixel pad.
 *
 * @return 0, or `EINVAL` for `NULL`.
 *
 * # Safety
 * `o` must be `NULL` or a live handle; `pad` must be four floats.
 */
int ef_convert_options_set_pad_output(ef_convert_options *o, const float *pad);

/**
 * Normalize with a preset: an `EF_NORMALIZATION_*` code.
 *
 * @return 0, or `EINVAL` for an unknown code or `NULL`.
 *
 * # Safety
 * `o` must be `NULL` or a live handle.
 */
int ef_convert_options_set_normalization_preset(ef_convert_options *o, uint32_t preset);

/**
 * Normalize as `(x - mean) / std` on 0-255 values, `n` values each: 1 for
 * every channel or 3 for RGB.
 *
 * @return 0, or `EINVAL` for `NULL`, another `n`, a non-finite mean or a
 * zero or non-finite std.
 *
 * # Safety
 * `o` must be `NULL` or a live handle; `mean` and `std` must hold `n` floats.
 */
int ef_convert_options_set_normalization(ef_convert_options *o,
                                         const float *mean,
                                         const float *std,
                                         uintptr_t n);

/**
 * Resample with this interpolation when a convert with `o` resizes, instead
 * of the processor's (see [`ef_image_processor_set_interpolation`]): an
 * `EF_INTERPOLATION_*` code. `EF_INTERPOLATION_BICUBIC` and
 * `EF_INTERPOLATION_AREA` run as bilinear, logged once per mode. The
 * `EDGEFIRST_INTERPOLATION` environment variable, when set, overrides it.
 *
 * @return 0, or `EINVAL` for an unknown code or `NULL`.
 *
 * # Safety
 * `o` must be `NULL` or a live handle.
 */
int ef_convert_options_set_interpolation(ef_convert_options *o, uint32_t mode);

/**
 * Options that apply a model's preprocessing: its normalization, fit,
 * letterbox placement, pad and interpolation (bicubic and area run as
 * bilinear). `spec` usually comes from the decoder library's
 * `ef_model_config_preprocessing`.
 *
 * @return options to free with [`ef_convert_options_free`], or `NULL` with
 * `errno` `EINVAL` for a `NULL` spec, an unknown code, a zero or non-finite
 * `std`, or a pad outside 0-255 in pixel space.
 *
 * # Safety
 * `spec` must be `NULL` or valid for reads.
 */
ef_convert_options *ef_convert_options_from_spec(const ef_preprocess_spec *spec);

/**
 * Where a convert with `o` puts the image content in a `dst_w`x`dst_h`
 * destination for a `src_w`x`src_h` source: the whole destination for a
 * stretch, the letterbox rectangle otherwise, the same pixels the convert
 * paints. Pass `normalized_out` to `ef_semantic_options_set_content_region`
 * or the draw functions' `content_region`.
 *
 * `pixels_out` (`NULL` to skip) receives `[x, y, width, height]` in
 * destination pixels; `normalized_out` (`NULL` to skip) receives
 * `[xmin, ymin, xmax, ymax]` normalized by the destination size.
 *
 * @return 0, or `EINVAL` for a `NULL` handle, a zero size, or a source
 * crop outside the source.
 *
 * # Safety
 * `o` must be `NULL` or live; `pixels_out` and `normalized_out` `NULL` or
 * four writable elements.
 */
int ef_convert_options_content_region(const ef_convert_options *o,
                                      uintptr_t src_w,
                                      uintptr_t src_h,
                                      uintptr_t dst_w,
                                      uintptr_t dst_h,
                                      uint32_t *pixels_out,
                                      float *normalized_out);

/**
 * Convert `src` into `dst`, scaling, converting colour and rotating as needed.
 *
 * `src`/`dst` may have been minted by any EdgeFirst library -- every library
 * links the same shared tensor implementation, so both are read the same
 * way regardless of which one minted them. `options` may be `NULL` for the
 * defaults.
 *
 * @return 0 on success; `EINVAL` for options that cannot apply to `dst`
 * (such as a normalization on an integer destination without quantization),
 * an invalid crop or shape, or a tensor with no pixel format; `ENOTSUP`
 * when no backend supports the conversion; `EACCES` when a tensor's
 * declared `ef_cpu_access` does not cover the CPU pass, including when
 * GL/G2D declined the convert and the CPU fallback was refused; otherwise
 * `EIO` or the backend's errno.
 *
 * # Safety
 * `p`, `src` and `dst` must be live handles.
 */
int ef_image_processor_convert(ef_image_processor *p,
                               const ef_tensor *src,
                               ef_tensor *dst,
                               const ef_convert_options *options);

/**
 * Like [`ef_image_processor_convert`], but does not wait for the GPU.
 *
 * @return the codes [`ef_image_processor_convert`] returns.
 *
 * # Safety
 * `p`, `src` and `dst` must be live handles.
 */
int ef_image_processor_convert_deferred(ef_image_processor *p,
                                        const ef_tensor *src,
                                        ef_tensor *dst,
                                        const ef_convert_options *options);

/**
 * DMA-BUF row pitch alignment the GL backend requires, in bytes.
 */
uintptr_t ef_gpu_dma_buf_pitch_alignment_bytes(void);

/**
 * Round `width` up so `width * bpp` meets the GPU pitch alignment.
 */
uintptr_t ef_align_width_for_gpu_pitch(uintptr_t width, uintptr_t bpp);

/**
 * Align `width` for `format` (`"NV12"`, `"rgba8"`, …) and `dtype`.
 *
 * # Safety
 * `format` must be a NUL-terminated string.
 */
uintptr_t ef_align_width_for_pixel_format(uintptr_t width, const char *format, uint32_t dtype);

/**
 * Create a processor forced to one backend.
 *
 * `backend` is an `EF_COMPUTE_BACKEND_*` code. A forced backend
 * disables the fallback chain entirely — if it is unavailable the call fails
 * rather than quietly using another, which is the point of forcing one.
 *
 * @return `NULL` when the backend is unknown or unavailable here.
 */
ef_image_processor *ef_image_processor_new_with_backend(uint32_t backend);

/**
 * Set the RGBA palette used when drawing class masks.
 *
 * Copied, not borrowed, so the caller may free its array immediately.
 *
 * @return 0 on success, `EINVAL` for a null argument or zero colours.
 *
 * # Safety
 * `colors` must point to `count` RGBA quads.
 */
int ef_image_processor_set_class_colors(ef_image_processor *p,
                                        const uint8_t (*colors)[4],
                                        uintptr_t count);

/**
 * Set the thickness, in `dst` pixels, of the box outlines drawn by
 * `ef_image_processor_draw_decoded_masks` and
 * `ef_image_processor_draw_proto_masks`.
 *
 * The outline is drawn outside the box edges and clipped at the image
 * border. The default is 3. Applies to every drawing backend the processor
 * holds.
 *
 * @return 0 on success, `EINVAL` for a null processor or a zero `width`,
 * `ENOTSUP` when the processor's backend does not draw (G2D).
 *
 * # Safety
 * `p` must be a processor from `ef_image_processor_new*` or `NULL`.
 */
int ef_image_processor_set_box_line_width(ef_image_processor *p, uint32_t width);

/**
 * Set how `ef_image_processor_convert` resamples when it resizes, for
 * converts whose options do not set their own
 * ([`ef_convert_options_set_interpolation`]).
 *
 * `mode` is an `EF_INTERPOLATION_*` code: `EF_INTERPOLATION_BILINEAR` (the
 * default; half-pixel centres, as OpenCV `INTER_LINEAR`) or
 * `EF_INTERPOLATION_NEAREST`. `EF_INTERPOLATION_BICUBIC` and
 * `EF_INTERPOLATION_AREA` are accepted and run as bilinear, logged once per
 * mode. Applies to every backend the processor holds; the
 * `EDGEFIRST_INTERPOLATION` environment variable (`nearest` | `bilinear`),
 * read when the processor is created, overrides it.
 *
 * @return 0 on success, `EINVAL` for a null processor or unknown mode,
 * `EIO` when a backend cannot apply the mode.
 *
 * # Safety
 * `p` must be a live processor.
 */
int ef_image_processor_set_interpolation(ef_image_processor *p, uint32_t mode);

/**
 * Flush any queued GPU work and wait for it.
 *
 * @return 0 on success, otherwise an errno.
 *
 * # Safety
 * `p` must be a live processor.
 */
int ef_image_processor_flush(ef_image_processor *p);

/**
 * Platforms: Linux, macOS, iOS, Android.
 *
 * Convert, returning a sync-fence fd instead of blocking on the GPU.
 *
 * The GL to NPU handoff. `*fence_fd` receives a descriptor the caller owns and
 * must close, or `-1` when the platform has no native fence and the convert
 * therefore completed synchronously — in which case the destination is already
 * safe to read.
 *
 * `options` as for [`ef_image_processor_convert`].
 *
 * @return the codes [`ef_image_processor_convert`] returns, and `ENOTSUP`
 * off Unix.
 *
 * # Safety
 * `p`, `src`, `dst` must be live handles; `fence_fd` must be writable.
 */
int ef_image_processor_convert_fence(ef_image_processor *p,
                                     const ef_tensor *src,
                                     ef_tensor *dst,
                                     const ef_convert_options *options,
                                     int *fence_fd);

/**
 * Platforms: Windows.
 *
 * Convert, returning an event handle instead of blocking on the GPU. The
 * event is set when the destination is complete; the caller owns it and
 * closes it with `CloseHandle`. `*fence` is `NULL` when the convert
 * completed synchronously (no fence on this display).
 *
 * `options` as for [`ef_image_processor_convert`].
 *
 * @return the codes [`ef_image_processor_convert`] returns, and `ENOTSUP`
 * off Windows.
 *
 * # Safety
 * `p`, `src`, `dst` must be live handles; `fence` must be writable.
 */
int ef_image_processor_convert_fence_handle(ef_image_processor *p,
                                            const ef_tensor *src,
                                            ef_tensor *dst,
                                            const ef_convert_options *options,
                                            void **fence);

/**
 * Deploy defaults: tile size = model input, overlap 0.2.
 */
struct ef_tiling_config ef_tiling_config_default(uintptr_t tile_w, uintptr_t tile_h);

/**
 * EvenDist tile grid. Free with [`ef_tile_spec_list_free`].
 */
ef_tile_spec_list *ef_tile_grid(uintptr_t frame_h,
                                uintptr_t frame_w,
                                uintptr_t tile_h,
                                uintptr_t tile_w,
                                float overlap_ratio);

/**
 * Number of tile specs. Zero for NULL.
 *
 * # Safety
 * `list` must be `NULL` or a live handle from this library.
 */
uintptr_t ef_tile_spec_list_len(const ef_tile_spec_list *list);

/**
 * Copy one tile spec into `out`. Returns 0 on success.
 *
 * # Safety
 * `list` and `out` must be live or NULL as documented.
 */
int ef_tile_spec_list_get(const ef_tile_spec_list *list, uintptr_t index, struct ef_tile_spec *out);

/**
 * Free a tile-spec list. NULL is a no-op.
 *
 * # Safety
 * `list` must be `NULL` or have come from this library.
 */
void ef_tile_spec_list_free(ef_tile_spec_list *list);

/**
 * Number of tile placements. Zero for NULL.
 *
 * # Safety
 * `list` must be `NULL` or a live handle from this library.
 */
uintptr_t ef_tile_placement_list_len(const ef_tile_placement_list *list);

/**
 * Copy one tile placement into `out`. Returns 0 on success.
 *
 * # Safety
 * `list` and `out` must be live or NULL as documented.
 */
int ef_tile_placement_list_get(const ef_tile_placement_list *list,
                               uintptr_t index,
                               ef_tile_placement *out);

/**
 * Free a tile-placement list. NULL is a no-op.
 *
 * # Safety
 * `list` must be `NULL` or have come from this library.
 */
void ef_tile_placement_list_free(ef_tile_placement_list *list);

/**
 * Allocate a tall packed batch that stacks `n` tiles.
 *
 * The declared access is widened as for `ef_image_processor_create_image`;
 * tiles are written through views, so any predicted CPU write is added as
 * `EF_CPU_ACCESS_READ_WRITE`.
 *
 * # Safety
 * `format` is a NUL-terminated wire code.
 */
ef_tensor *ef_image_processor_alloc_tile_batch(ef_image_processor *p,
                                               uintptr_t n,
                                               const struct ef_tiling_config *config,
                                               const char *format,
                                               uint32_t dtype,
                                               uint32_t storage,
                                               uint32_t access);

/**
 * Plan tile placements for a frame. Free with [`ef_tile_placement_list_free`].
 *
 * `options` (`NULL` for a plain stretch) are the convert options every tile
 * is rendered with; a letterboxing one gives each placement a content
 * region. Pass the same options to [`ef_image_processor_tile_into`] or
 * [`ef_image_processor_tile_one`].
 *
 * @return the list, or `NULL` with `errno` `EINVAL` for a `NULL` argument,
 * a zero tile size, an overlap outside `[0, 1)`, or options that rotate or
 * flip.
 *
 * # Safety
 * `p` and `config` must be live or NULL as documented; `options` `NULL` or
 * live.
 */
ef_tile_placement_list *ef_image_processor_plan_tiles(ef_image_processor *p,
                                                      uintptr_t src_w,
                                                      uintptr_t src_h,
                                                      const struct ef_tiling_config *config,
                                                      const ef_convert_options *options);

/**
 * Render every tile of `src` into `dst`, each converted with `options`
 * (`NULL` for a plain stretch) sampling the tile's crop.
 *
 * @return a placement list to free with [`ef_tile_placement_list_free`], or
 * `NULL` with `errno` set: `EINVAL` for bad arguments or options that rotate
 * or flip, `EACCES` when a tensor's declared CPU access does not cover the
 * CPU pass, otherwise the codes `ef_image_processor_convert` returns.
 *
 * # Safety
 * Pointers must be live or NULL as documented.
 */
ef_tile_placement_list *ef_image_processor_tile_into(ef_image_processor *p,
                                                     const ef_tensor *src,
                                                     ef_tensor *dst,
                                                     const struct ef_tiling_config *config,
                                                     const ef_convert_options *options);

/**
 * Render one planned tile of `src` into `dst`, converted with `options`
 * (`NULL` for a plain stretch) sampling the tile's crop.
 *
 * @return 0 on success, `EINVAL` for bad arguments or options that rotate
 * or flip, otherwise the codes `ef_image_processor_convert` returns
 * (`EACCES` for a refused CPU access).
 *
 * # Safety
 * Pointers must be live or NULL as documented.
 */
int ef_image_processor_tile_one(ef_image_processor *p,
                                const ef_tensor *src,
                                ef_tensor *dst,
                                const ef_tile_placement *placement,
                                const struct ef_tiling_config *config,
                                const ef_convert_options *options);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* EDGEFIRST_IMAGE_H */
