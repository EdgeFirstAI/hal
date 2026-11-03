#ifndef EDGEFIRST_KERNELS_H
#define EDGEFIRST_KERNELS_H

/**
 * @file kernels.h
 * @brief EdgeFirst quantized compute kernels C API
 *
 * SPDX-License-Identifier: Apache-2.0
 * Copyright (c) 2026 Au-Zone Technologies. All Rights Reserved.
 *
 * Softmax, sigmoid, 8-bit table lookup and DFL box decoding over caller-owned
 * tensors. Build a plan once with ef_<op>_plan_new(), then call
 * ef_<op>_plan_run() any number of times, concurrently if needed. A plan_new
 * that fails with ENOTSUP means the operation is valid but not handled.
 *
 * Every function returns 0, or -1 with errno set (EINVAL, ENOTSUP or ENOMEM)
 * and a message available from ef_kernels_last_error(). A plan_new failure
 * always leaves *out NULL. A plan_run rejects with EINVAL a view whose data is
 * not aligned to its element type, and an input and output whose memory spans
 * overlap, interleaved views of one buffer included.
 *
 * Adding a field to any params struct, or any other change to a C type's
 * layout, bumps ef_kernels_abi_version().
 */

#include <errno.h>
#include <stdarg.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>

#ifdef __cplusplus
extern "C" {
#endif

/** @brief An opaque int8 softmax plan. */
typedef struct ef_softmax_i8_plan ef_softmax_i8_plan;

/** @brief An opaque int16 softmax plan. */
typedef struct ef_softmax_i16_plan ef_softmax_i16_plan;

/** @brief An opaque sigmoid plan. */
typedef struct ef_sigmoid_plan ef_sigmoid_plan;

/** @brief An opaque DFL decode plan. */
typedef struct ef_dfl_plan ef_dfl_plan;

/** @brief An opaque distance-to-box decode plan. */
typedef struct ef_dist2bbox_plan ef_dist2bbox_plan;

/** @brief An opaque fused DFL plus distance-to-box plan. */
typedef struct ef_dfl_box_plan ef_dfl_box_plan;

/** @brief An opaque 8-bit table lookup plan. */
typedef struct ef_lut_i8_plan ef_lut_i8_plan;


/* Generated with cbindgen:0.29.4 */

/* WARNING: The generated portion of this file is produced by cbindgen. Do not modify it directly. */

#include <stdarg.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>

/**
 * Element type of a tensor. Params and views carry it as a `uint32_t` holding one of these values.
 */
enum ef_kernel_dtype
#if defined(__cplusplus) || __STDC_VERSION__ >= 202311L
  : uint32_t
#endif // defined(__cplusplus) || __STDC_VERSION__ >= 202311L
 {
  EF_KERNEL_DTYPE_I8 = 0,
  EF_KERNEL_DTYPE_U8 = 1,
  EF_KERNEL_DTYPE_I16 = 2,
  EF_KERNEL_DTYPE_U16 = 3,
  /**
   * 16-bit IEEE half, stored as `uint16_t`.
   */
  EF_KERNEL_DTYPE_F16 = 4,
  EF_KERNEL_DTYPE_F32 = 5,
};
#ifndef __cplusplus
#if __STDC_VERSION__ >= 202311L
typedef enum ef_kernel_dtype ef_kernel_dtype;
#else
typedef uint32_t ef_kernel_dtype;
#endif // __STDC_VERSION__ >= 202311L
#endif // __cplusplus

/**
 * Layout of the four decoded box values.
 *
 */
enum ef_kernel_box_format
#if defined(__cplusplus) || __STDC_VERSION__ >= 202311L
  : uint32_t
#endif // defined(__cplusplus) || __STDC_VERSION__ >= 202311L
 {
  /**
   * `[xc, yc, w, h]`.
   */
  EF_KERNEL_BOX_XYWH = 0,
  /**
   * `[x1, y1, x2, y2]`.
   */
  EF_KERNEL_BOX_XYXY = 1,
};
#ifndef __cplusplus
#if __STDC_VERSION__ >= 202311L
typedef enum ef_kernel_box_format ef_kernel_box_format;
#else
typedef uint32_t ef_kernel_box_format;
#endif // __STDC_VERSION__ >= 202311L
#endif // __cplusplus

/**
 * Affine quantization: `real = scale * (q - zero_point)`.
 */
typedef struct ef_kernel_quant {
  float scale;
  int32_t zero_point;
} ef_kernel_quant;

/**
 * Parameters of an int8 softmax; the output must be quantized (1/256, -128).
 */
typedef struct ef_softmax_i8_params {
  struct ef_kernel_quant input;
  struct ef_kernel_quant output;
  float beta;
} ef_softmax_i8_params;

/**
 * A caller-owned tensor: data pointer, dtype, shape and optional element strides.
 *
 * A plan run rejects with EINVAL a view whose `data` is not aligned to its element type
 * (2 bytes for 16-bit types, 4 for float32), and an input and output whose memory spans
 * overlap, interleaved views of one buffer included. A span runs from `data` to one past the
 * last element the shape and strides address.
 */
typedef struct ef_kernel_view {
  /**
   * First element, aligned to the element type; may be NULL only when the view has no elements.
   */
  void *data;
  /**
   * An `ef_kernel_dtype` value.
   */
  uint32_t dtype;
  /**
   * Number of dimensions, at most 6.
   */
  size_t ndim;
  /**
   * `ndim` extents.
   */
  const size_t *shape;
  /**
   * `ndim` strides in elements (not bytes), or NULL for a contiguous row-major layout.
   */
  const ptrdiff_t *strides;
} ef_kernel_view;

/**
 * A caller-supplied thread pool.
 *
 * `run` must call `task(arg, i)` for each `i` in `0..n_tasks` (on any threads, in any order)
 * and return only after all calls have finished. `ctx` is passed back unchanged. `run` is
 * invoked from whichever thread calls `plan_run`, so it must tolerate concurrent calls.
 * Tasks keep staging buffers of up to 32 KiB on the stack, so worker threads need about
 * 64 KiB of stack headroom beyond the pool's own use.
 */
typedef struct ef_kernel_pool {
  void *ctx;
  void (*run)(void *ctx, size_t n_tasks, void (*task)(void *arg, size_t index), void *arg);
} ef_kernel_pool;

/**
 * Parameters of an int16 softmax.
 */
typedef struct ef_softmax_i16_params {
  struct ef_kernel_quant input;
  struct ef_kernel_quant output;
  float beta;
} ef_softmax_i16_params;

/**
 * Parameters of a sigmoid.
 *
 * A float `input_dtype` ignores `input` and a float `output_dtype` ignores `output`, so a
 * float tensor may pass `{0, 0}`.
 */
typedef struct ef_sigmoid_params {
  uint32_t input_dtype;
  struct ef_kernel_quant input;
  uint32_t output_dtype;
  struct ef_kernel_quant output;
} ef_sigmoid_params;

/**
 * Parameters of a distribution-focal-loss decode.
 *
 * A float `input_dtype` ignores `input` and a float `output_dtype` ignores `output`, so a
 * float tensor may pass `{0, 0}`.
 */
typedef struct ef_dfl_params {
  uint32_t input_dtype;
  struct ef_kernel_quant input;
  uint32_t output_dtype;
  struct ef_kernel_quant output;
  size_t reg_max;
} ef_dfl_params;

/**
 * One FPN level of a concatenated anchor axis.
 */
typedef struct ef_kernel_level {
  size_t anchors;
  size_t grid_w;
  float stride;
} ef_kernel_level;

/**
 * Parameters of a distance-to-box decode.
 *
 * A float `input_dtype` ignores `input` and a float `output_dtype` ignores `output`, so a
 * float tensor may pass `{0, 0}`.
 */
typedef struct ef_dist2bbox_params {
  uint32_t input_dtype;
  struct ef_kernel_quant input;
  uint32_t output_dtype;
  struct ef_kernel_quant output;
  /**
   * `n_levels` entries.
   */
  const struct ef_kernel_level *levels;
  size_t n_levels;
  /**
   * An `ef_kernel_box_format` value.
   */
  uint32_t box_format;
} ef_dist2bbox_params;

/**
 * Parameters of the fused DFL plus distance-to-box decode.
 *
 * A float `input_dtype` ignores `input` and a float `output_dtype` ignores `output`, so a
 * float tensor may pass `{0, 0}`.
 */
typedef struct ef_dfl_box_params {
  uint32_t input_dtype;
  struct ef_kernel_quant input;
  uint32_t output_dtype;
  struct ef_kernel_quant output;
  size_t reg_max;
  /**
   * `n_levels` entries.
   */
  const struct ef_kernel_level *levels;
  size_t n_levels;
  /**
   * An `ef_kernel_box_format` value.
   */
  uint32_t box_format;
} ef_dfl_box_params;

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

/**
 * ABI version of this library's C surface.
 *
 * It is bumped by any change to the layout of a C type, including a field added to any
 * params struct.
 */
uint32_t ef_kernels_abi_version(void);

/**
 * Message of the calling thread's last failed kernels call, or NULL if the last call succeeded.
 *
 * The pointer is valid until the same thread's next `ef_*_plan_new*` or `ef_*_plan_run` call,
 * each of which replaces or clears the message; copy the text out if it must outlive that.
 *
 * @return A NUL-terminated string, or NULL when there is no error.
 */
const char *ef_kernels_last_error(void);

/**
 * Creates a plan for an int8 softmax over the last axis.
 * `beta` scales the logits; the output must be quantized (1/256, -128).
 *
 * @param params Plan parameters; read during the call only.
 * @param out Receives the new plan, or NULL on failure.
 * @return 0 on success; -1 with errno EINVAL (NULL pointer, unknown dtype, invalid
 *         parameters) or ENOTSUP (valid but not handled), and a message for
 *         ef_kernels_last_error().
 *
 * # Safety
 * `params` must point to a valid params struct (and any array it points to) and
 * `out` to writable storage for one pointer.
 */
int ef_softmax_i8_plan_new(const struct ef_softmax_i8_params *params, ef_softmax_i8_plan **out);

/**
 * Runs an int8 softmax over the last axis. Safe to call concurrently on one plan from several threads.
 *
 * @param plan A plan from the matching `_plan_new`.
 * @param input Input tensor; must stay readable for the call.
 * @param output Output tensor; must stay writable for the call. Views whose memory spans
 *        overlap the input's, interleaved views of one buffer included, are EINVAL.
 * @param pool Optional thread pool; NULL runs on the calling thread.
 * @return 0 on success; -1 with errno EINVAL (NULL pointer, bad dtype, rank or view,
 *         misaligned data, overlapping views, shape mismatch, a pool task that panicked) or
 *         ENOTSUP, and a message for ef_kernels_last_error().
 *
 * # Safety
 * All pointers must be NULL or valid, and the views must address memory the caller owns.
 */
int ef_softmax_i8_plan_run(const ef_softmax_i8_plan *plan,
                           const struct ef_kernel_view *input,
                           const struct ef_kernel_view *output,
                           const struct ef_kernel_pool *pool);

/**
 * Frees a plan. NULL is a no-op.
 *
 * @param plan A plan from the matching `_plan_new`, not used afterwards.
 *
 * # Safety
 * `plan` must be NULL or an unfreed pointer from the matching `_plan_new`.
 */
void ef_softmax_i8_plan_free(ef_softmax_i8_plan *plan);

/**
 * Creates a plan for an int16 softmax over the last axis.
 *
 * @param params Plan parameters; read during the call only.
 * @param out Receives the new plan, or NULL on failure.
 * @return 0 on success; -1 with errno EINVAL (NULL pointer, unknown dtype, invalid
 *         parameters) or ENOTSUP (valid but not handled), and a message for
 *         ef_kernels_last_error().
 *
 * # Safety
 * `params` must point to a valid params struct (and any array it points to) and
 * `out` to writable storage for one pointer.
 */
int ef_softmax_i16_plan_new(const struct ef_softmax_i16_params *params, ef_softmax_i16_plan **out);

/**
 * Runs an int16 softmax over the last axis. Safe to call concurrently on one plan from several threads.
 *
 * @param plan A plan from the matching `_plan_new`.
 * @param input Input tensor; must stay readable for the call.
 * @param output Output tensor; must stay writable for the call. Views whose memory spans
 *        overlap the input's, interleaved views of one buffer included, are EINVAL.
 * @param pool Optional thread pool; NULL runs on the calling thread.
 * @return 0 on success; -1 with errno EINVAL (NULL pointer, bad dtype, rank or view,
 *         misaligned data, overlapping views, shape mismatch, a pool task that panicked) or
 *         ENOTSUP, and a message for ef_kernels_last_error().
 *
 * # Safety
 * All pointers must be NULL or valid, and the views must address memory the caller owns.
 */
int ef_softmax_i16_plan_run(const ef_softmax_i16_plan *plan,
                            const struct ef_kernel_view *input,
                            const struct ef_kernel_view *output,
                            const struct ef_kernel_pool *pool);

/**
 * Frees a plan. NULL is a no-op.
 *
 * @param plan A plan from the matching `_plan_new`, not used afterwards.
 *
 * # Safety
 * `plan` must be NULL or an unfreed pointer from the matching `_plan_new`.
 */
void ef_softmax_i16_plan_free(ef_softmax_i16_plan *plan);

/**
 * Creates a plan for an elementwise sigmoid.
 *
 * @param params Plan parameters; read during the call only.
 * @param out Receives the new plan, or NULL on failure.
 * @return 0 on success; -1 with errno EINVAL (NULL pointer, unknown dtype, invalid
 *         parameters) or ENOTSUP (valid but not handled), and a message for
 *         ef_kernels_last_error().
 *
 * # Safety
 * `params` must point to a valid params struct (and any array it points to) and
 * `out` to writable storage for one pointer.
 */
int ef_sigmoid_plan_new(const struct ef_sigmoid_params *params, ef_sigmoid_plan **out);

/**
 * Runs an elementwise sigmoid. Safe to call concurrently on one plan from several threads.
 *
 * @param plan A plan from the matching `_plan_new`.
 * @param input Input tensor; must stay readable for the call.
 * @param output Output tensor; must stay writable for the call. Views whose memory spans
 *        overlap the input's, interleaved views of one buffer included, are EINVAL.
 * @param pool Optional thread pool; NULL runs on the calling thread.
 * @return 0 on success; -1 with errno EINVAL (NULL pointer, bad dtype, rank or view,
 *         misaligned data, overlapping views, shape mismatch, a pool task that panicked) or
 *         ENOTSUP, and a message for ef_kernels_last_error().
 *
 * # Safety
 * All pointers must be NULL or valid, and the views must address memory the caller owns.
 */
int ef_sigmoid_plan_run(const ef_sigmoid_plan *plan,
                        const struct ef_kernel_view *input,
                        const struct ef_kernel_view *output,
                        const struct ef_kernel_pool *pool);

/**
 * Frees a plan. NULL is a no-op.
 *
 * @param plan A plan from the matching `_plan_new`, not used afterwards.
 *
 * # Safety
 * `plan` must be NULL or an unfreed pointer from the matching `_plan_new`.
 */
void ef_sigmoid_plan_free(ef_sigmoid_plan *plan);

/**
 * Creates a plan for a distribution-focal-loss decode.
 *
 * @param params Plan parameters; read during the call only.
 * @param out Receives the new plan, or NULL on failure.
 * @return 0 on success; -1 with errno EINVAL (NULL pointer, unknown dtype, invalid
 *         parameters) or ENOTSUP (valid but not handled), and a message for
 *         ef_kernels_last_error().
 *
 * # Safety
 * `params` must point to a valid params struct (and any array it points to) and
 * `out` to writable storage for one pointer.
 */
int ef_dfl_plan_new(const struct ef_dfl_params *params, ef_dfl_plan **out);

/**
 * Runs a distribution-focal-loss decode. Safe to call concurrently on one plan from several threads.
 *
 * @param plan A plan from the matching `_plan_new`.
 * @param input Input tensor; must stay readable for the call.
 * @param output Output tensor; must stay writable for the call. Views whose memory spans
 *        overlap the input's, interleaved views of one buffer included, are EINVAL.
 * @param pool Optional thread pool; NULL runs on the calling thread.
 * @return 0 on success; -1 with errno EINVAL (NULL pointer, bad dtype, rank or view,
 *         misaligned data, overlapping views, shape mismatch, a pool task that panicked) or
 *         ENOTSUP, and a message for ef_kernels_last_error().
 *
 * # Safety
 * All pointers must be NULL or valid, and the views must address memory the caller owns.
 */
int ef_dfl_plan_run(const ef_dfl_plan *plan,
                    const struct ef_kernel_view *input,
                    const struct ef_kernel_view *output,
                    const struct ef_kernel_pool *pool);

/**
 * Frees a plan. NULL is a no-op.
 *
 * @param plan A plan from the matching `_plan_new`, not used afterwards.
 *
 * # Safety
 * `plan` must be NULL or an unfreed pointer from the matching `_plan_new`.
 */
void ef_dfl_plan_free(ef_dfl_plan *plan);

/**
 * Creates a plan for a distance-to-box decode.
 *
 * @param params Plan parameters; read during the call only.
 * @param out Receives the new plan, or NULL on failure.
 * @return 0 on success; -1 with errno EINVAL (NULL pointer, unknown dtype, invalid
 *         parameters) or ENOTSUP (valid but not handled), and a message for
 *         ef_kernels_last_error().
 *
 * # Safety
 * `params` must point to a valid params struct (and any array it points to) and
 * `out` to writable storage for one pointer.
 */
int ef_dist2bbox_plan_new(const struct ef_dist2bbox_params *params, ef_dist2bbox_plan **out);

/**
 * Runs a distance-to-box decode. Safe to call concurrently on one plan from several threads.
 *
 * @param plan A plan from the matching `_plan_new`.
 * @param input Input tensor; must stay readable for the call.
 * @param output Output tensor; must stay writable for the call. Views whose memory spans
 *        overlap the input's, interleaved views of one buffer included, are EINVAL.
 * @param pool Optional thread pool; NULL runs on the calling thread.
 * @return 0 on success; -1 with errno EINVAL (NULL pointer, bad dtype, rank or view,
 *         misaligned data, overlapping views, shape mismatch, a pool task that panicked) or
 *         ENOTSUP, and a message for ef_kernels_last_error().
 *
 * # Safety
 * All pointers must be NULL or valid, and the views must address memory the caller owns.
 */
int ef_dist2bbox_plan_run(const ef_dist2bbox_plan *plan,
                          const struct ef_kernel_view *input,
                          const struct ef_kernel_view *output,
                          const struct ef_kernel_pool *pool);

/**
 * Frees a plan. NULL is a no-op.
 *
 * @param plan A plan from the matching `_plan_new`, not used afterwards.
 *
 * # Safety
 * `plan` must be NULL or an unfreed pointer from the matching `_plan_new`.
 */
void ef_dist2bbox_plan_free(ef_dist2bbox_plan *plan);

/**
 * Creates a plan for the fused DFL plus distance-to-box decode.
 *
 * @param params Plan parameters; read during the call only.
 * @param out Receives the new plan, or NULL on failure.
 * @return 0 on success; -1 with errno EINVAL (NULL pointer, unknown dtype, invalid
 *         parameters) or ENOTSUP (valid but not handled), and a message for
 *         ef_kernels_last_error().
 *
 * # Safety
 * `params` must point to a valid params struct (and any array it points to) and
 * `out` to writable storage for one pointer.
 */
int ef_dfl_box_plan_new(const struct ef_dfl_box_params *params, ef_dfl_box_plan **out);

/**
 * Runs the fused DFL plus distance-to-box decode. Safe to call concurrently on one plan from several threads.
 *
 * @param plan A plan from the matching `_plan_new`.
 * @param input Input tensor; must stay readable for the call.
 * @param output Output tensor; must stay writable for the call. Views whose memory spans
 *        overlap the input's, interleaved views of one buffer included, are EINVAL.
 * @param pool Optional thread pool; NULL runs on the calling thread.
 * @return 0 on success; -1 with errno EINVAL (NULL pointer, bad dtype, rank or view,
 *         misaligned data, overlapping views, shape mismatch, a pool task that panicked) or
 *         ENOTSUP, and a message for ef_kernels_last_error().
 *
 * # Safety
 * All pointers must be NULL or valid, and the views must address memory the caller owns.
 */
int ef_dfl_box_plan_run(const ef_dfl_box_plan *plan,
                        const struct ef_kernel_view *input,
                        const struct ef_kernel_view *output,
                        const struct ef_kernel_pool *pool);

/**
 * Frees a plan. NULL is a no-op.
 *
 * @param plan A plan from the matching `_plan_new`, not used afterwards.
 *
 * # Safety
 * `plan` must be NULL or an unfreed pointer from the matching `_plan_new`.
 */
void ef_dfl_box_plan_free(ef_dfl_box_plan *plan);

/**
 * Creates a plan that maps every 8-bit element through a 256-entry table.
 *
 * @param dtype `EF_KERNEL_DTYPE_I8` or `EF_KERNEL_DTYPE_U8`.
 * @param table 256 bytes, indexed by the element reinterpreted as `uint8_t`.
 * @param out Receives the new plan, or NULL on failure.
 * @return 0 on success; -1 with errno EINVAL or ENOTSUP, and a message for
 *         ef_kernels_last_error().
 *
 * # Safety
 * `table` must be NULL or point to 256 readable bytes; `out` must be NULL or writable.
 */
int ef_lut_i8_plan_new_table(uint32_t dtype, const uint8_t *table, ef_lut_i8_plan **out);

/**
 * Creates a plan that applies a quantized sigmoid through a table built from the quantization.
 *
 * @param dtype `EF_KERNEL_DTYPE_I8` or `EF_KERNEL_DTYPE_U8`.
 * @param input Input quantization.
 * @param output Output quantization.
 * @param out Receives the new plan, or NULL on failure.
 * @return 0 on success; -1 with errno EINVAL or ENOTSUP, and a message for
 *         ef_kernels_last_error().
 *
 * # Safety
 * `out` must be NULL or writable.
 */
int ef_lut_i8_plan_new_sigmoid(uint32_t dtype,
                               struct ef_kernel_quant input,
                               struct ef_kernel_quant output,
                               ef_lut_i8_plan **out);

/**
 * Runs a table plan. Safe to call concurrently on one plan from several threads.
 *
 * @param plan A plan from `ef_lut_i8_plan_new_table` or `ef_lut_i8_plan_new_sigmoid`.
 * @param input Input tensor of the plan's dtype.
 * @param output Output tensor of the same dtype and shape. Views whose memory spans
 *        overlap the input's, interleaved views of one buffer included, are EINVAL.
 * @param pool Optional thread pool; NULL runs on the calling thread.
 * @return 0 on success; -1 with errno EINVAL or ENOTSUP, and a message for
 *         ef_kernels_last_error().
 *
 * # Safety
 * All pointers must be NULL or valid, and the views must address memory the caller owns.
 */
int ef_lut_i8_plan_run(const ef_lut_i8_plan *plan,
                       const struct ef_kernel_view *input,
                       const struct ef_kernel_view *output,
                       const struct ef_kernel_pool *pool);

/**
 * Frees a table plan. NULL is a no-op.
 *
 * @param plan A plan from a `ef_lut_i8_plan_new_*` function, not used afterwards.
 *
 * # Safety
 * `plan` must be NULL or an unfreed pointer from a `ef_lut_i8_plan_new_*` function.
 */
void ef_lut_i8_plan_free(ef_lut_i8_plan *plan);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* EDGEFIRST_KERNELS_H */
