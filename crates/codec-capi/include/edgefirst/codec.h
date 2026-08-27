#ifndef EDGEFIRST_CODEC_H
#define EDGEFIRST_CODEC_H

/**
 * @file codec.h
 * @brief EdgeFirst image decode/encode C API
 *
 * SPDX-License-Identifier: Apache-2.0
 * Copyright (c) 2026 Au-Zone Technologies. All Rights Reserved.
 *
 * Decode JPEG and PNG into a tensor, and encode a tensor back out.
 *
 * Decoding writes *into* a tensor you already have, rather than allocating one:
 * that is what lets a decode target be a DMA buffer the GPU will read next,
 * with no copy in between. The tensor may come from any EdgeFirst library.
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
 * ABI version of this library's C surface.
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
 * @return 0 on success, `ENOSPC` when `dst` is too small, otherwise an
 *         errno.
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
 * # Safety
 * `out` must be writable.
 */
int ef_codec_colorimetry_from_v4l2(uint32_t colorspace,
                                   uint32_t xfer,
                                   uint32_t ycbcr_enc,
                                   uint32_t quant,
                                   uint32_t *out);

/**
 * Encode a packed RGB/RGBA u8 tensor as JPEG.
 *
 * `quality` in 1–100; `0` or out-of-range uses 80.
 *
 * # Safety
 * `path` must be a NUL-terminated string; `t` must be a live handle.
 */
int ef_codec_save_jpeg(const ef_tensor *t, const char *path, int quality);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* EDGEFIRST_CODEC_H */
