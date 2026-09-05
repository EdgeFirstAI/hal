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
 * Geometry for a convert: source rectangle and letterbox padding.
 *
 * `NULL` anywhere one is accepted means the whole source, stretched to fill.
 */
typedef struct ef_crop {
  /**
   * Source rectangle in pixels. A zero `width` or `height` means the whole
   * source, so a zeroed struct is the same as passing `NULL`.
   */
  uint32_t x;
  uint32_t y;
  uint32_t width;
  uint32_t height;
  /**
   * Non-zero to preserve aspect ratio, padding the remainder.
   */
  int letterbox;
  /**
   * Letterbox fill colour, RGBA. Ignored unless `letterbox` is set.
   */
  uint8_t pad[4];
} ef_crop;

/**
 * Static tiling configuration. Seed with [`ef_tiling_config_default`].
 */
typedef struct ef_tiling_config {
  uintptr_t tile_w;
  uintptr_t tile_h;
  float overlap_ratio;
  uint8_t pad[4];
  /**
   * 0 = stretch, 1 = letterbox.
   */
  int fit;
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

/**
 * ABI version of this library's C surface.
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
 * IOSurface, a dma-buf) returns `EIO`: the GL base-layer draw has no `BGRA`
 * arm and the CPU backend renders only `RGBA`/`RGB`. It previously returned
 * 0 with the destination unwritten.
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
                                          const float *letterbox,
                                          uint32_t color_mode);

/**
 * Draw proto masks onto `dst`. `protos` and `coeffs` are borrowed, not taken.
 *
 * On Windows the destination's `ef_tensor_gpu_completion` reflects this draw
 * afterwards, as it does after a convert.
 *
 * The same `BGRA` background restriction as
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
                                        const float *letterbox,
                                        uint32_t color_mode);

/**
 * Materialize per-instance masks from proto tensors. Caller frees the list.
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
                                                   const float *letterbox);

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
 * @return a tensor the caller owns, or `NULL`.
 *
 * # Safety
 * `p` and `d` must be live.
 */
ef_tensor *ef_image_processor_create_image_desc(ef_image_processor *p,
                                                const ef_tensor_image_desc *d);

/**
 * Convert `src` into `dst`, scaling, converting colour and rotating as needed.
 *
 * `src`/`dst` may have been minted by any EdgeFirst library -- every library
 * links the same shared tensor implementation, so both are read the same
 * way regardless of which one minted them. `crop` may be `NULL` for the
 * whole source.
 *
 * @return 0 on success, otherwise an errno.
 *
 * # Safety
 * `p`, `src` and `dst` must be live handles.
 */
int ef_image_processor_convert(ef_image_processor *p,
                               const ef_tensor *src,
                               ef_tensor *dst,
                               uint32_t rotation,
                               uint32_t flip,
                               const struct ef_crop *crop);

/**
 * Like [`ef_image_processor_convert`], but does not wait for the GPU.
 *
 * # Safety
 * `p`, `src` and `dst` must be live handles.
 */
int ef_image_processor_convert_deferred(ef_image_processor *p,
                                        const ef_tensor *src,
                                        ef_tensor *dst,
                                        uint32_t rotation,
                                        uint32_t flip,
                                        const struct ef_crop *crop);

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
 * `backend`: 0 = auto, 1 = CPU, 2 = G2D, 3 = OpenGL. A forced backend
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
 * @return 0 on success, `ENOTSUP` off Unix, otherwise an errno.
 *
 * # Safety
 * `p`, `src`, `dst` must be live handles; `fence_fd` must be writable.
 */
int ef_image_processor_convert_fence(ef_image_processor *p,
                                     const ef_tensor *src,
                                     ef_tensor *dst,
                                     uint32_t rotation,
                                     uint32_t flip,
                                     const struct ef_crop *crop,
                                     int *fence_fd);

/**
 * Platforms: Windows.
 *
 * Convert, returning an event handle instead of blocking on the GPU. The
 * event is set when the destination is complete; the caller owns it and
 * closes it with `CloseHandle`. `*fence` is `NULL` when the convert
 * completed synchronously (no fence on this display).
 *
 * @return 0 on success, `ENOTSUP` off Windows, otherwise an errno.
 *
 * # Safety
 * `p`, `src`, `dst` must be live handles; `fence` must be writable.
 */
int ef_image_processor_convert_fence_handle(ef_image_processor *p,
                                            const ef_tensor *src,
                                            ef_tensor *dst,
                                            uint32_t rotation,
                                            uint32_t flip,
                                            const struct ef_crop *crop,
                                            void **fence);

/**
 * Deploy defaults: overlap 0.2, stretch, pad `[114,114,114,255]`.
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
 * # Safety
 * `p` and `config` must be live or NULL as documented.
 */
ef_tile_placement_list *ef_image_processor_plan_tiles(ef_image_processor *p,
                                                      uintptr_t src_w,
                                                      uintptr_t src_h,
                                                      const struct ef_tiling_config *config);

/**
 * Render every tile of `src` into `dst`.
 *
 * # Safety
 * Pointers must be live or NULL as documented.
 */
ef_tile_placement_list *ef_image_processor_tile_into(ef_image_processor *p,
                                                     const ef_tensor *src,
                                                     ef_tensor *dst,
                                                     const struct ef_tiling_config *config);

/**
 * Render one planned tile of `src` into `dst`.
 *
 * # Safety
 * Pointers must be live or NULL as documented.
 */
int ef_image_processor_tile_one(ef_image_processor *p,
                                const ef_tensor *src,
                                ef_tensor *dst,
                                const ef_tile_placement *placement,
                                const struct ef_tiling_config *config);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* EDGEFIRST_IMAGE_H */
