#ifndef EDGEFIRST_KERNELS_H
#define EDGEFIRST_KERNELS_H

/**
 * @file kernels.h
 * @brief EdgeFirst quantized compute kernels C API
 *
 * SPDX-License-Identifier: Apache-2.0
 * Copyright (c) 2026 Au-Zone Technologies. All Rights Reserved.
 *
 * Softmax, sigmoid, 8-bit table lookup, DFL box decoding, TFLite-exact int8
 * ADD/SUB/MUL and matrix multiplication, a fused DFL softmax-and-projection
 * head and fused int8 attention over caller-owned tensors. Build a plan once with ef_<op>_plan_new(), then call
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

/** @brief An opaque int8 ADD/SUB/MUL plan. */
typedef struct ef_binary_i8_plan ef_binary_i8_plan;

/** @brief An opaque int8 matrix multiplication plan. */
typedef struct ef_matmul_i8_plan ef_matmul_i8_plan;

/** @brief An opaque int8 softmax-then-dot (DFL head) plan. */
typedef struct ef_softmax_dot_i8_plan ef_softmax_dot_i8_plan;

/** @brief An opaque fused int8 attention plan. */
typedef struct ef_attention_i8_plan ef_attention_i8_plan;

/** @brief Opaque packing scratch for one attention call shape. */
typedef struct ef_attention_scratch ef_attention_scratch;


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
 * Which of TFLite's int8 kernels a plan reproduces, where they round differently. Params carry
 * it as a `uint32_t` holding one of these values.
 */
enum ef_tflite_kernels
#if defined(__cplusplus) || __STDC_VERSION__ >= 202311L
  : uint32_t
#endif // defined(__cplusplus) || __STDC_VERSION__ >= 202311L
 {
  /**
   * int8 `ADD` and `MUL` in 16-element NEON blocks, matrix products through ruy: every aarch64
   * build.
   */
  EF_TFLITE_KERNELS_NEON_RUY = 0,
  /**
   * NEON blocks, matrix products through gemmlowp (rounded as the reference kernels): x86
   * builds with SSE4.1, such as the LiteRT Python wheels.
   */
  EF_TFLITE_KERNELS_NEON_GEMMLOWP = 1,
  /**
   * Scalar `ADD` and `MUL`, matrix products through ruy: builds without NEON.
   */
  EF_TFLITE_KERNELS_RUY = 2,
  /**
   * The reference kernels (`BuiltinRefOpResolver`).
   */
  EF_TFLITE_KERNELS_REFERENCE = 3,
};
#ifndef __cplusplus
#if __STDC_VERSION__ >= 202311L
typedef enum ef_tflite_kernels ef_tflite_kernels;
#else
typedef uint32_t ef_tflite_kernels;
#endif // __STDC_VERSION__ >= 202311L
#endif // __cplusplus

/**
 * A fused activation, applied as the output clamp TFLite derives from the output quantization.
 * The values are TFLite's `TfLiteFusedActivation`; params carry it as a `uint32_t`.
 */
enum ef_kernel_activation
#if defined(__cplusplus) || __STDC_VERSION__ >= 202311L
  : uint32_t
#endif // defined(__cplusplus) || __STDC_VERSION__ >= 202311L
 {
  EF_KERNEL_ACTIVATION_NONE = 0,
  EF_KERNEL_ACTIVATION_RELU = 1,
  /**
   * Clamp to [-1, 1].
   */
  EF_KERNEL_ACTIVATION_RELU_N1_TO_1 = 2,
  EF_KERNEL_ACTIVATION_RELU6 = 3,
};
#ifndef __cplusplus
#if __STDC_VERSION__ >= 202311L
typedef enum ef_kernel_activation ef_kernel_activation;
#else
typedef uint32_t ef_kernel_activation;
#endif // __STDC_VERSION__ >= 202311L
#endif // __cplusplus

/**
 * The elementwise operation of an int8 binary plan; params carry it as a `uint32_t`.
 */
enum ef_binary_op
#if defined(__cplusplus) || __STDC_VERSION__ >= 202311L
  : uint32_t
#endif // defined(__cplusplus) || __STDC_VERSION__ >= 202311L
 {
  EF_BINARY_OP_ADD = 0,
  EF_BINARY_OP_SUB = 1,
  EF_BINARY_OP_MUL = 2,
};
#ifndef __cplusplus
#if __STDC_VERSION__ >= 202311L
typedef enum ef_binary_op ef_binary_op;
#else
typedef uint32_t ef_binary_op;
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

/**
 * A TFLite fixed-point requantization: `acc * multiplier * 2^(shift - 31)`, rounded as the
 * plan's kernels variant does.
 */
typedef struct ef_kernel_requant {
  int32_t multiplier;
  int32_t shift;
} ef_kernel_requant;

/**
 * Parameters of an int8 `ADD`, `SUB` or `MUL`: `output = lhs op rhs`.
 */
typedef struct ef_binary_i8_params {
  /**
   * An `ef_binary_op` value.
   */
  uint32_t op;
  struct ef_kernel_quant lhs;
  struct ef_kernel_quant rhs;
  struct ef_kernel_quant output;
  /**
   * An `ef_kernel_activation` value.
   */
  uint32_t activation;
  /**
   * An `ef_tflite_kernels` value.
   */
  uint32_t kernels;
} ef_binary_i8_params;

/**
 * Parameters of an int8 matrix multiplication
 * `out[.., m, n] = clamp(zo + R[n](sum_k (lhs[.., m, k] - zl) * (rhs[.., k, n] - zr) + bias[n]))`.
 */
typedef struct ef_matmul_i8_params {
  int32_t lhs_zero_point;
  int32_t rhs_zero_point;
  /**
   * The output zero point and scale; the scale only places the activation clamp.
   */
  struct ef_kernel_quant output;
  /**
   * `n_requant` entries: one for every column, or one per output column.
   */
  const struct ef_kernel_requant *requant;
  size_t n_requant;
  /**
   * `n_bias` entries added to each column's accumulator; `n_bias` 0 for none.
   */
  const int32_t *bias;
  size_t n_bias;
  /**
   * An `ef_kernel_activation` value.
   */
  uint32_t activation;
  /**
   * An `ef_tflite_kernels` value.
   */
  uint32_t kernels;
} ef_matmul_i8_params;

/**
 * Parameters of an int8 softmax over the last axis followed by a product with one weight
 * column (a YOLO DFL head); `projection` must have one requantization, at most one bias and
 * the softmax output's zero point as its left zero point.
 */
typedef struct ef_softmax_dot_i8_params {
  struct ef_softmax_i8_params softmax;
  /**
   * `n_weights` entries, one per element of a row.
   */
  const int8_t *weights;
  size_t n_weights;
  struct ef_matmul_i8_params projection;
} ef_softmax_dot_i8_params;

/**
 * The scalar `MUL` a TFLite graph applies to Q before the attention scores.
 */
typedef struct ef_attention_q_mul {
  /**
   * The `MUL` output for every Q value, indexed by the raw byte (see
   * `ef_binary_i8_plan_table_for_rhs`).
   */
  uint8_t table[256];
  /**
   * The quantization of the `MUL` output.
   */
  struct ef_kernel_quant output;
} ef_attention_q_mul;

/**
 * Parameters of an int8 attention block: `[MUL] -> BATCH_MATMUL -> SOFTMAX -> BATCH_MATMUL`.
 */
typedef struct ef_attention_i8_params {
  struct ef_kernel_quant q;
  /**
   * The scale applied to Q first, or NULL when the graph has none.
   */
  const struct ef_attention_q_mul *q_mul;
  struct ef_kernel_quant k;
  struct ef_kernel_quant v;
  /**
   * The quantization of the int8 scores.
   */
  struct ef_kernel_quant scores;
  /**
   * The softmax `beta`.
   */
  float beta;
  /**
   * The quantization of the int8 probabilities: scale 1/256, zero point -128.
   */
  struct ef_kernel_quant probs;
  struct ef_kernel_quant output;
  /**
   * An `ef_tflite_kernels` value.
   */
  uint32_t kernels;
} ef_attention_i8_params;

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

/**
 * Computes the requantization of TFLite's int8 `BATCH_MATMUL` (and per-tensor
 * `FULLY_CONNECTED`): `(lhs_scale * rhs_scale)` in float, then divided by `output_scale` in
 * double, as `QuantizeMultiplier`.
 *
 * @param lhs_scale Left operand scale.
 * @param rhs_scale Right operand scale.
 * @param output_scale Output scale.
 * @param out Receives the requantization.
 * @return 0 on success; -1 with errno EINVAL (NULL `out`, a multiplier that is not finite
 *         and positive), and a message for ef_kernels_last_error().
 *
 * # Safety
 * `out` must be NULL or writable.
 */
int ef_kernel_requant_tflite_batch_matmul(float lhs_scale,
                                          float rhs_scale,
                                          float output_scale,
                                          struct ef_kernel_requant *out);

/**
 * Computes the requantization of one output channel of TFLite's int8 `CONV_2D` and
 * `DEPTHWISE_CONV_2D`: the whole product `input_scale * filter_scale / output_scale` in
 * double, as `QuantizeMultiplier`.
 *
 * @param input_scale Input scale.
 * @param filter_scale The channel's filter scale.
 * @param output_scale Output scale.
 * @param out Receives the requantization.
 * @return 0 on success; -1 with errno EINVAL (NULL `out`, a multiplier that is not finite
 *         and positive), and a message for ef_kernels_last_error().
 *
 * # Safety
 * `out` must be NULL or writable.
 */
int ef_kernel_requant_tflite_conv(float input_scale,
                                  float filter_scale,
                                  float output_scale,
                                  struct ef_kernel_requant *out);

/**
 * Creates a plan for an int8 `ADD`, `SUB` or `MUL` reproducing the TFLite kernels `kernels`
 * names, bit for bit.
 *
 * @param params Plan parameters; read during the call only.
 * @param out Receives the new plan, or NULL on failure.
 * @return 0 on success; -1 with errno EINVAL (NULL pointer, unknown enum value, invalid
 *         parameters) or ENOTSUP (scale ratios or an activation range TFLite does not take),
 *         and a message for ef_kernels_last_error().
 *
 * # Safety
 * `params` must point to a valid params struct and `out` to writable storage for one
 * pointer.
 */
int ef_binary_i8_plan_new(const struct ef_binary_i8_params *params, ef_binary_i8_plan **out);

/**
 * Runs an int8 binary op elementwise: all three int8 with the same shape (any rank); a
 * broadcast input is a view with zero strides. Safe to call concurrently on one plan from
 * several threads.
 *
 * @param plan A plan from `ef_binary_i8_plan_new`.
 * @param lhs Left input; must stay readable for the call.
 * @param rhs Right input; must stay readable for the call. It may overlap `lhs`.
 * @param output Output tensor; must stay writable for the call. Views whose memory spans
 *        overlap either input's, interleaved views of one buffer included, are EINVAL.
 * @param pool Optional thread pool; NULL runs on the calling thread.
 * @return 0 on success; -1 with errno EINVAL (NULL pointer, bad dtype, rank or view,
 *         misaligned data, overlapping views, shape mismatch, a pool task that panicked) or
 *         ENOTSUP, and a message for ef_kernels_last_error().
 *
 * # Safety
 * All pointers must be NULL or valid, and the views must address memory the caller owns.
 */
int ef_binary_i8_plan_run(const ef_binary_i8_plan *plan,
                          const struct ef_kernel_view *lhs,
                          const struct ef_kernel_view *rhs,
                          const struct ef_kernel_view *output,
                          const struct ef_kernel_pool *pool);

/**
 * Fills `table` with the plan's output for every `lhs` value with the right operand fixed at
 * `rhs`, indexed by the left value's raw byte, valid for every element of a call over `len`
 * elements; it can feed `ef_lut_i8_plan_new_table` or an `ef_attention_q_mul`.
 *
 * @param plan A plan from `ef_binary_i8_plan_new`.
 * @param rhs The constant right operand.
 * @param len Elements of the call the table stands for.
 * @param table Receives 256 bytes.
 * @return 0 on success; -1 with errno EINVAL (NULL pointer) or ENOTSUP (the call would give
 *         some elements TFLite's NEON-block arithmetic and others its scalar code, and the
 *         two disagree for this `rhs`), and a message for ef_kernels_last_error().
 *
 * # Safety
 * `plan` must be NULL or a live plan; `table` must be NULL or point to 256 writable bytes.
 */
int ef_binary_i8_plan_table_for_rhs(const ef_binary_i8_plan *plan,
                                    int8_t rhs,
                                    size_t len,
                                    uint8_t *table);

/**
 * Frees a plan. NULL is a no-op.
 *
 * @param plan A plan from `ef_binary_i8_plan_new`, not used afterwards.
 *
 * # Safety
 * `plan` must be NULL or an unfreed pointer from `ef_binary_i8_plan_new`.
 */
void ef_binary_i8_plan_free(ef_binary_i8_plan *plan);

/**
 * Creates a plan for an int8 matrix multiplication over the last two axes, with TFLite's
 * integer requantization (`BATCH_MATMUL`, `FULLY_CONNECTED`, 1x1 `CONV_2D`).
 *
 * @param params Plan parameters; read during the call only, arrays included.
 * @param out Receives the new plan, or NULL on failure.
 * @return 0 on success; -1 with errno EINVAL (NULL pointer, NULL array with a nonzero count,
 *         unknown enum value, invalid parameters), ENOTSUP (ruy rounding of a multiplier of
 *         0.5 or more, an empty activation range) or ENOMEM, and a message for
 *         ef_kernels_last_error().
 *
 * # Safety
 * `params` must point to a valid params struct (and the arrays it points to) and `out` to
 * writable storage for one pointer.
 */
int ef_matmul_i8_plan_new(const struct ef_matmul_i8_params *params, ef_matmul_i8_plan **out);

/**
 * Runs `out = lhs * rhs` over the last two axes: `[.., M, K] x [.., K, N] -> [.., M, N]`, all
 * int8 and of the same rank (2 to 6); the other axes are batch axes (a broadcast operand has
 * zero batch strides) and transposed operands are views with swapped strides. Safe to call
 * concurrently on one plan from several threads.
 *
 * @param plan A plan from `ef_matmul_i8_plan_new`.
 * @param lhs Left operand; must stay readable for the call.
 * @param rhs Right operand; must stay readable for the call. It may overlap `lhs`.
 * @param output Output tensor; must stay writable for the call. Views whose memory spans
 *        overlap either operand's are EINVAL.
 * @param pool Optional thread pool; NULL runs on the calling thread.
 * @return 0 on success; -1 with errno EINVAL (NULL pointer, bad dtype, rank or view,
 *         misaligned data, overlapping views, shapes that do not chain, a requantization or
 *         bias count that is not one per column, a pool task that panicked) or ENOTSUP (a
 *         reduction axis above 4096), and a message for ef_kernels_last_error().
 *
 * # Safety
 * All pointers must be NULL or valid, and the views must address memory the caller owns.
 */
int ef_matmul_i8_plan_run(const ef_matmul_i8_plan *plan,
                          const struct ef_kernel_view *lhs,
                          const struct ef_kernel_view *rhs,
                          const struct ef_kernel_view *output,
                          const struct ef_kernel_pool *pool);

/**
 * Whether the matmul kernels take a reduction axis of `k` elements.
 *
 * @param k Reduction axis length.
 * @return 0 when they do; -1 with errno ENOTSUP past 4096, and a message for
 *         ef_kernels_last_error().
 */
int ef_matmul_i8_check_depth(size_t k);

/**
 * Frees a plan. NULL is a no-op.
 *
 * @param plan A plan from `ef_matmul_i8_plan_new`, not used afterwards.
 *
 * # Safety
 * `plan` must be NULL or an unfreed pointer from `ef_matmul_i8_plan_new`.
 */
void ef_matmul_i8_plan_free(ef_matmul_i8_plan *plan);

/**
 * Creates a plan for an int8 softmax over the last axis followed by a product with one weight
 * column (a YOLO DFL head) in one pass. The probabilities are not rounded to 8 bits: the
 * result is within a few LSB of the float model and not TFLite's pair bit for bit.
 *
 * @param params Plan parameters; read during the call only, arrays included.
 * @param out Receives the new plan, or NULL on failure.
 * @return 0 on success; -1 with errno EINVAL (NULL pointer, NULL array with a nonzero count,
 *         no weights, a projection with more than one requantization or bias or another left
 *         zero point than the softmax output's, invalid parameters), ENOTSUP or ENOMEM, and a
 *         message for ef_kernels_last_error().
 *
 * # Safety
 * `params` must point to a valid params struct (and the arrays it points to) and `out` to
 * writable storage for one pointer.
 */
int ef_softmax_dot_i8_plan_new(const struct ef_softmax_dot_i8_params *params,
                               ef_softmax_dot_i8_plan **out);

/**
 * Runs the softmax-then-dot of `input` (`[.., n]`, `n` the number of weights, unit stride
 * along the last axis) into `output` (`[.., 1]`), both int8. Safe to call concurrently on one
 * plan from several threads.
 *
 * @param plan A plan from `ef_softmax_dot_i8_plan_new`.
 * @param input Input logits; must stay readable for the call.
 * @param output Output tensor; must stay writable for the call. Views whose memory spans
 *        overlap the input's are EINVAL.
 * @param pool Optional thread pool; NULL runs on the calling thread.
 * @return 0 on success; -1 with errno EINVAL (NULL pointer, bad dtype, rank or view,
 *         misaligned data, overlapping views, shape mismatch, a pool task that panicked) or
 *         ENOTSUP (an input row axis that is not unit-stride), and a message for
 *         ef_kernels_last_error().
 *
 * # Safety
 * All pointers must be NULL or valid, and the views must address memory the caller owns.
 */
int ef_softmax_dot_i8_plan_run(const ef_softmax_dot_i8_plan *plan,
                               const struct ef_kernel_view *input,
                               const struct ef_kernel_view *output,
                               const struct ef_kernel_pool *pool);

/**
 * Frees a plan. NULL is a no-op.
 *
 * @param plan A plan from `ef_softmax_dot_i8_plan_new`, not used afterwards.
 *
 * # Safety
 * `plan` must be NULL or an unfreed pointer from `ef_softmax_dot_i8_plan_new`.
 */
void ef_softmax_dot_i8_plan_free(ef_softmax_dot_i8_plan *plan);

/**
 * Creates a plan for fused int8 attention, `softmax(q' * k^T) * v` per head with `q'` Q
 * through the optional `q_mul`, reproducing TFLite's `[MUL] -> BATCH_MATMUL -> SOFTMAX ->
 * BATCH_MATMUL` chain.
 *
 * @param params Plan parameters; read during the call only, `q_mul` included.
 * @param out Receives the new plan, or NULL on failure.
 * @return 0 on success; -1 with errno EINVAL (NULL pointer, unknown enum value, invalid
 *         scales or zero points) or ENOTSUP (probabilities not quantized (1/256, -128), a
 *         requantization the rounding cannot apply), and a message for
 *         ef_kernels_last_error().
 *
 * # Safety
 * `params` must point to a valid params struct (`q_mul` NULL or valid) and `out` to writable
 * storage for one pointer.
 */
int ef_attention_i8_plan_new(const struct ef_attention_i8_params *params,
                             ef_attention_i8_plan **out);

/**
 * Whether the attention kernels take `tokens` key/value tokens per head, `dk`-long queries
 * and keys and `dv`-long values.
 *
 * @param tokens Key/value tokens per head.
 * @param dk Query and key length.
 * @param dv Value length.
 * @return 0 when they do; -1 with errno ENOTSUP past 2048 tokens or a head dim of 256, and a
 *         message for ef_kernels_last_error().
 */
int ef_attention_i8_check_shape(size_t tokens, size_t dk, size_t dv);

/**
 * Creates the packing scratch `plan` needs for one call shape: `heads` heads of `queries`
 * queries attending over `tokens` key/value tokens, `dk`-long queries and keys and `dv`-long
 * values. A scratch serves any number of calls of that shape on that plan (or on a plan with
 * the same kernel), one call at a time.
 *
 * @param plan A plan from `ef_attention_i8_plan_new`; read during the call only.
 * @param heads Heads.
 * @param queries Queries per head.
 * @param tokens Key/value tokens per head.
 * @param dk Query and key length.
 * @param dv Value length.
 * @param out Receives the new scratch, or NULL on failure.
 * @return 0 on success; -1 with errno EINVAL (NULL pointer, sizes that overflow) or ENOTSUP
 *         (a shape `ef_attention_i8_check_shape` refuses, an allocation that failed), and a
 *         message for ef_kernels_last_error().
 *
 * # Safety
 * `plan` must be NULL or a live plan; `out` must be NULL or writable.
 */
int ef_attention_scratch_new(const ef_attention_i8_plan *plan,
                             size_t heads,
                             size_t queries,
                             size_t tokens,
                             size_t dk,
                             size_t dv,
                             ef_attention_scratch **out);

/**
 * Frees a scratch. NULL is a no-op.
 *
 * @param scratch A scratch from `ef_attention_scratch_new`, not used afterwards.
 *
 * # Safety
 * `scratch` must be NULL or an unfreed pointer from `ef_attention_scratch_new`.
 */
void ef_attention_scratch_free(ef_attention_scratch *scratch);

/**
 * Runs attention of `q` `[heads, queries, dk]` over `k` `[heads, tokens, dk]` and `v`
 * `[heads, tokens, dv]` into `output` `[heads, queries, dv]`, all int8 views with any
 * strides (head slices of one token-major buffer included). Safe to call concurrently on
 * one plan from several threads, each call with its own scratch.
 *
 * @param plan A plan from `ef_attention_i8_plan_new`.
 * @param q Queries; must stay readable for the call.
 * @param k Keys; must stay readable for the call. Inputs may overlap each other.
 * @param v Values; must stay readable for the call.
 * @param output Output tensor; must stay writable for the call. Views whose memory spans
 *        overlap any input's are EINVAL.
 * @param scratch A scratch from `ef_attention_scratch_new` for this shape, used by no other
 *        call during this one. A scratch built for another shape, or by a plan with another
 *        kernel, is EINVAL and nothing is written.
 * @param pool Optional thread pool; NULL runs on the calling thread.
 * @return 0 on success; -1 with errno EINVAL (NULL pointer, bad dtype, rank or view,
 *         misaligned data, overlapping views, shape mismatch, a mismatched scratch, a pool
 *         task that panicked) or ENOTSUP, and a message for ef_kernels_last_error().
 *
 * # Safety
 * All pointers must be NULL or valid, the views must address memory the caller owns, and no
 * other call may use `scratch` concurrently.
 */
int ef_attention_i8_plan_run(const ef_attention_i8_plan *plan,
                             const struct ef_kernel_view *q,
                             const struct ef_kernel_view *k,
                             const struct ef_kernel_view *v,
                             const struct ef_kernel_view *output,
                             ef_attention_scratch *scratch,
                             const struct ef_kernel_pool *pool);

/**
 * Frees a plan. NULL is a no-op.
 *
 * @param plan A plan from `ef_attention_i8_plan_new`, not used afterwards.
 *
 * # Safety
 * `plan` must be NULL or an unfreed pointer from `ef_attention_i8_plan_new`.
 */
void ef_attention_i8_plan_free(ef_attention_i8_plan *plan);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* EDGEFIRST_KERNELS_H */
