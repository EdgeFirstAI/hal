#ifndef EDGEFIRST_TENSOR_H
#define EDGEFIRST_TENSOR_H

/**
 * @file tensor.h
 * @brief EdgeFirst tensor C API
 *
 * SPDX-License-Identifier: Apache-2.0
 * Copyright (c) 2026 Au-Zone Technologies. All Rights Reserved.
 *
 * The one tensor header. Every EdgeFirst application includes this — they all
 * need at least ef_tensor_free() — and every sibling library's header includes
 * it rather than restating the tensor surface.
 *
 * The handle is fully opaque. Every accessor is an exported function of
 * libedgefirst_tensor, the single implementation home of the tensor type: a
 * tensor from ef_image_processor_create_image() and one from ef_tensor_new()
 * are the same kind of object and behave identically, ef_tensor_free()
 * included.
 *
 * Three surfaces, and a layout appears in exactly one of them:
 *   - Opaque handle + exported accessors, for in-process access.
 *   - A builder, for construction, with sticky errors.
 *   - A (blob, fds) pair, for IPC. Fds travel out of band; a struct with an
 *     fd field is meaningless across a process boundary.
 */

#include <stdarg.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief An opaque tensor handle.
 */
typedef struct ef_tensor ef_tensor;

/**
 * @brief Opaque builder handle.
 *
 * Forward-declared so callers can write `ef_tensor_builder *` without the
 * `struct` keyword. The definition is private to the library: construction
 * state is not part of the ABI.
 */
typedef struct ef_tensor_builder ef_tensor_builder;


/* Generated with cbindgen:0.29.4 */

/* WARNING: The generated portion of this file is produced by cbindgen. Do not modify it directly. */

#include <stdarg.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>

/**
 * Log severity. Maps 1:1 to Rust `log::Level`.
 */
enum ef_log_level
#if defined(__cplusplus) || __STDC_VERSION__ >= 202311L
  : uint32_t
#endif // defined(__cplusplus) || __STDC_VERSION__ >= 202311L
 {
  EF_LOG_LEVEL_ERROR = 1,
  EF_LOG_LEVEL_WARN = 2,
  EF_LOG_LEVEL_INFO = 3,
  EF_LOG_LEVEL_DEBUG = 4,
  EF_LOG_LEVEL_TRACE = 5,
};
#ifndef __cplusplus
#if __STDC_VERSION__ >= 202311L
typedef enum ef_log_level ef_log_level;
#else
typedef uint32_t ef_log_level;
#endif // __STDC_VERSION__ >= 202311L
#endif // __cplusplus

/**
 * Element type of a tensor's addressing grid.
 */
enum ef_dtype
#if defined(__cplusplus) || __STDC_VERSION__ >= 202311L
  : uint32_t
#endif // defined(__cplusplus) || __STDC_VERSION__ >= 202311L
 {
  EF_DTYPE_U8 = 0,
  EF_DTYPE_I8 = 1,
  EF_DTYPE_U16 = 2,
  EF_DTYPE_I16 = 3,
  EF_DTYPE_U32 = 4,
  EF_DTYPE_I32 = 5,
  EF_DTYPE_U64 = 6,
  EF_DTYPE_I64 = 7,
  EF_DTYPE_F16 = 8,
  EF_DTYPE_F32 = 9,
  EF_DTYPE_F64 = 10,
};
#ifndef __cplusplus
#if __STDC_VERSION__ >= 202311L
typedef enum ef_dtype ef_dtype;
#else
typedef uint32_t ef_dtype;
#endif // __STDC_VERSION__ >= 202311L
#endif // __cplusplus

/**
 * Backing store for a tensor.
 *
 * Every kind is declared on every platform. Whether one can be *materialised*
 * here is a runtime question — an IOSurface is a meaningful thing to name on
 * Linux even though nothing will allocate one — and platform-gating the
 * vocabulary would make the same integer mean different things per target.
 */
enum ef_storage_kind
#if defined(__cplusplus) || __STDC_VERSION__ >= 202311L
  : uint32_t
#endif // defined(__cplusplus) || __STDC_VERSION__ >= 202311L
 {
  EF_STORAGE_KIND_MEM = 0,
  EF_STORAGE_KIND_SHM = 1,
  EF_STORAGE_KIND_DMA_BUF = 2,
  EF_STORAGE_KIND_IO_SURFACE = 3,
  EF_STORAGE_KIND_PBO = 4,
  EF_STORAGE_KIND_CUDA = 5,
};
#ifndef __cplusplus
#if __STDC_VERSION__ >= 202311L
typedef enum ef_storage_kind ef_storage_kind;
#else
typedef uint32_t ef_storage_kind;
#endif // __STDC_VERSION__ >= 202311L
#endif // __cplusplus

/**
 * CPU access direction for a map window.
 *
 * Mirrors `edgefirst_tensor::CpuAccess`'s semantics; the values are the wire
 * codes. `None` names the no-CPU-access declaration and is never a valid map
 * direction.
 */
enum ef_cpu_access
#if defined(__cplusplus) || __STDC_VERSION__ >= 202311L
  : uint32_t
#endif // defined(__cplusplus) || __STDC_VERSION__ >= 202311L
 {
  EF_CPU_ACCESS_NONE = 0,
  EF_CPU_ACCESS_READ = 1,
  EF_CPU_ACCESS_WRITE = 2,
  EF_CPU_ACCESS_READ_WRITE = 3,
};
#ifndef __cplusplus
#if __STDC_VERSION__ >= 202311L
typedef enum ef_cpu_access ef_cpu_access;
#else
typedef uint32_t ef_cpu_access;
#endif // __STDC_VERSION__ >= 202311L
#endif // __cplusplus

/**
 * Vendor tile-compression scheme actually in force for a tensor's
 * allocation.
 *
 * A *result*, not a request: [`EfImageDescView::compression`] carries what
 * an allocation asked for ("any scheme" / "a specific one"), whereas this
 * names the scheme the allocator actually resolved to. `None` is a real
 * enumerator rather than a presence flag because "linear" is the answer for
 * almost every allocation on almost every platform -- there is no absent
 * case to distinguish from it, unlike `EfViewOrigin`/`EfQuantizationInfo`
 * where every field value is legitimate.
 *
 * Mirrors `edgefirst_tensor::CompressionScheme`'s variants, plus the
 * `None` the Rust side spells as `Option::None`.
 */
enum ef_compression
#if defined(__cplusplus) || __STDC_VERSION__ >= 202311L
  : uint32_t
#endif // defined(__cplusplus) || __STDC_VERSION__ >= 202311L
 {
  /**
   * Linear layout -- no vendor tile compression.
   */
  EF_COMPRESSION_NONE = 0,
  /**
   * Qualcomm Adreno Universal Bandwidth Compression.
   */
  EF_COMPRESSION_UBWC = 1,
  /**
   * Arm Mali/Immortalis Framebuffer Compression.
   */
  EF_COMPRESSION_AFBC = 2,
  /**
   * Imagination PowerVR Image Compression.
   */
  EF_COMPRESSION_PVRIC = 3,
  /**
   * Samsung Xclipse (AMD RDNA) Delta Color Compression.
   */
  EF_COMPRESSION_DCC = 4,
};
#ifndef __cplusplus
#if __STDC_VERSION__ >= 202311L
typedef enum ef_compression ef_compression;
#else
typedef uint32_t ef_compression;
#endif // __STDC_VERSION__ >= 202311L
#endif // __cplusplus

/**
 * Which *kind* of failure the calling thread's last failing
 * `ef_tensor_*` call was.
 *
 * The companion to `ef_tensor_last_error_message`, and the reason it
 * exists: an entry point that reports failure by returning `NULL` has no
 * errno to carry a class, so a Rust consumer rebuilding a typed
 * `edgefirst_tensor::Error` from it had nothing to go on but the advisory
 * message -- which the message's own contract says must never be parsed.
 * `ef_tensor_batch` really did match on a fragment of a `Display` string
 * for exactly this reason.
 *
 * A **class**, not a one-to-one mirror of `edgefirst_tensor::Error`. Its
 * variants are the distinctions a caller acts on differently; several Rust
 * variants deliberately collapse into one here, and adding a Rust variant
 * does not oblige adding one here. `Unspecified` is the honest answer when
 * a failure path recorded only a message, and is what every such path
 * resets this to -- a stale class from an earlier failure would be read as
 * this call's, which is the confident-falsehood shape this whole mechanism
 * exists to remove.
 */
enum ef_error_class
#if defined(__cplusplus) || __STDC_VERSION__ >= 202311L
  : uint32_t
#endif // defined(__cplusplus) || __STDC_VERSION__ >= 202311L
 {
  /**
   * No class recorded for this failure; the message is all there is.
   */
  EF_ERROR_CLASS_UNSPECIFIED = 0,
  /**
   * A caller argument was malformed, out of range, or not recognized.
   */
  EF_ERROR_CLASS_INVALID_ARGUMENT = 1,
  /**
   * A shape was rejected: wrong rank, wrong element count, or not a
   * shape the format admits.
   */
  EF_ERROR_CLASS_INVALID_SHAPE = 2,
  /**
   * A shape or window did not fit the allocation behind it.
   */
  EF_ERROR_CLASS_INSUFFICIENT_CAPACITY = 3,
  /**
   * An index was outside the leading (batch) dimension.
   */
  EF_ERROR_CLASS_BATCH_INDEX_OUT_OF_BOUNDS = 4,
  /**
   * A spatial region did not fit inside its parent frame.
   */
  EF_ERROR_CLASS_REGION_OUT_OF_BOUNDS = 5,
  /**
   * The operation is not available for this tensor's backing, on this
   * platform, or in this build.
   */
  EF_ERROR_CLASS_NOT_SUPPORTED = 6,
  /**
   * The operation is legal but not permitted right now -- a live map, a
   * shared handle, a lock another holder owns.
   */
  EF_ERROR_CLASS_INVALID_OPERATION = 7,
  /**
   * An allocation failed, or a syscall backing one did.
   */
  EF_ERROR_CLASS_ALLOCATION_FAILED = 8,
  /**
   * A quantization payload was rejected.
   */
  EF_ERROR_CLASS_QUANTIZATION_INVALID = 9,
};
#ifndef __cplusplus
#if __STDC_VERSION__ >= 202311L
typedef enum ef_error_class ef_error_class;
#else
typedef uint32_t ef_error_class;
#endif // __STDC_VERSION__ >= 202311L
#endif // __cplusplus

/**
 * An image request, built up field by field.
 */
typedef struct ef_tensor_image_desc ef_tensor_image_desc;

/**
 * Flattened, `#[repr(C)]` view of an image-request descriptor's fields.
 *
 * `ef_tensor_image_desc` is opaque (handles are opaque both ways: a
 * receiving library never dereferences, sizes, or copies one). This is the
 * scalar block `ef_tensor_image_desc_get` fills instead -- the same shape as
 * `ef_tensor_plane`, which lets one library read a tensor it did not mint
 * without touching the other's private layout.
 *
 * `memory` and `compression` are each a value plus an explicit presence
 * flag rather than a sentinel: every code in `ef_storage_kind` (0..=5) is a
 * real value, so there is no unused number to repurpose as "no request"
 * without colliding with `ef_storage_kind`'s `MEM == 0`. `compression` is 1
 * for "any scheme" and 2 for "a specific vendor scheme", the latter not
 * further decodable through this view -- no `ef_tensor_image_desc_set_*`
 * entry point can request one, so this view has never needed to carry more
 * detail than "present, and it's a specific one."
 */
typedef struct ef_image_desc_view {
  /**
   * Requested width in pixels.
   */
  uint64_t width;
  /**
   * Requested height in pixels.
   */
  uint64_t height;
  /**
   * The requested pixel format's wire code (`PixelFormat::code()`).
   */
  uint32_t format;
  /**
   * `ef_dtype`.
   */
  uint32_t dtype;
  /**
   * `ef_cpu_access`.
   */
  uint32_t access;
  /**
   * `ef_storage_kind`, meaningful only when `has_memory != 0`.
   */
  uint32_t memory;
  /**
   * Non-zero when a specific memory backing was requested (`None` on the
   * Rust side auto-selects).
   */
  uint32_t has_memory;
  /**
   * 1 = any scheme the platform offers; 2 = a specific vendor scheme.
   * Meaningful only when `has_compression != 0`.
   */
  uint32_t compression;
  /**
   * Non-zero when a compression request was made.
   */
  uint32_t has_compression;
} ef_image_desc_view;

/**
 * One plane's location, mirroring `TensorPlane` on the wire.
 *
 * The route by which one library consumes a tensor another minted: the Rust
 * types are not shared across `.so` boundaries, so the planes are the
 * interface. Emitted into `edgefirst/tensor.h` as `ef_tensor_plane` by
 * cbindgen.
 */
typedef struct ef_tensor_plane {
  /**
   * dma-buf fd, IOSurface id, or -1 when host memory.
   */
  int64_t handle;
  /**
   * Byte offset of this plane within the handle.
   */
  uint64_t offset;
  /**
   * Bytes per line.
   */
  uint64_t stride;
  /**
   * Plane extent in bytes.
   */
  uint64_t size;
  /**
   * Valid payload bytes.
   */
  uint64_t used;
  /**
   * DRM format modifier; 0 = linear.
   */
  uint64_t modifier;
} ef_tensor_plane;

/**
 * Parent-region snapshot for a tensor that is a `view`/`batch` sub-region.
 *
 * The scalar block `ef_tensor_view_origin` fills, the same shape as
 * `ef_tensor_plane` and `ef_tensor_image_desc_view` -- one library reading a
 * tensor it did not mint. `has_origin` is a presence flag rather than a
 * sentinel value because every field is a legitimate 0 for a view pinned at
 * the parent's top-left corner; the other fields are meaningful only when
 * it is non-zero. Mirrors `edgefirst_tensor::ViewOrigin`.
 */
typedef struct EfViewOrigin {
  /**
   * Logical width of the root parent image, in pixels.
   */
  uint64_t parent_width;
  /**
   * Logical height of the root parent image, in pixels.
   */
  uint64_t parent_height;
  /**
   * The parent's row stride in bytes.
   */
  uint64_t parent_row_stride;
  /**
   * This view's top-left x origin within the root parent, in pixels.
   */
  uint64_t x;
  /**
   * This view's top-left y origin within the root parent, in pixels.
   */
  uint64_t y;
  /**
   * Non-zero when this tensor is a view/batch sub-region; 0 for a whole
   * tensor, in which case the other fields are all zero and unused.
   */
  uint32_t has_origin;
} EfViewOrigin;

/**
 * Callback invoked for each log record.
 */
typedef void (*EfLogCallback)(ef_log_level level,
                              const char *target,
                              const char *message,
                              void *userdata);

/**
 * A mapped CPU window over a tensor's bytes.
 *
 * By-value with no version field and no reserved tail, so its size
 * is baked into consumer call sites, so it evolves by a suffixed successor
 * (`ef_tensor_view2` + new entry points), never in place. The pointer is
 * valid from `ef_tensor_map` until the matching `ef_tensor_unmap`; writing
 * through it is allowed only when the map was taken with a writable access.
 */
typedef struct ef_tensor_view {
  uint8_t *ptr;
  uintptr_t len;
} ef_tensor_view;

/**
 * Presence/shape summary of a tensor's quantization metadata.
 *
 * The first half of the two-call idiom `ef_tensor_quantization_info` /
 * `ef_tensor_quantization_get` use: this scalar block tells the caller
 * *whether* quantization is attached and *how big* a buffer the second call
 * needs, without allocating on either side of the boundary --
 * `Quantization` itself is variable-length (an axis plus per-axis `scales`
 * and `zero_points`), so unlike `ef_tensor_plane`/`ef_tensor_view_origin`
 * this view cannot carry the payload itself.
 *
 * `has_quantization` is a presence flag, not a sentinel, for the same
 * reason `EfViewOrigin::has_origin` is: axis `0` and a scale of `0.0` are
 * both legitimate values, so there is no unused bit pattern to repurpose as
 * "absent" without colliding with a real one.
 */
typedef struct ef_quantization_info {
  /**
   * Channel axis for per-channel quantization; `-1` for per-tensor (no
   * axis). Meaningful only when `has_quantization != 0`.
   */
  int32_t axis;
  /**
   * Number of entries in the `scale`/`zero_point` arrays (1 for
   * per-tensor). Meaningful only when `has_quantization != 0`.
   */
  uint32_t count;
  /**
   * Non-zero when this tensor carries quantization metadata; 0 when it
   * does not, in which case `axis` and `count` are both zeroed and
   * unused.
   */
  uint32_t has_quantization;
} ef_quantization_info;

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

/**
 * ABI version of this library's C surface.
 *
 * Bumped only when the C ABI breaks. Adding a new exported accessor does
 * *not* bump this: an existing consumer's header simply never names the new
 * symbol, so linking against an older `libedgefirst_tensor.so` still
 * resolves everything it actually calls.
 */
uint32_t ef_tensor_abi_version(void);

/**
 * Create a builder. Returns `NULL` only on allocation failure.
 */
ef_tensor_builder *ef_tensor_builder_new(void);

/**
 * Free a builder. Freeing `NULL` is a no-op, matching `free(3)`.
 *
 * # Safety
 * `b` must have come from [`ef_tensor_builder_new`].
 */
void ef_tensor_builder_free(ef_tensor_builder *b);

/**
 * The first error recorded, or 0. `EINVAL` for a `NULL` builder.
 *
 * # Safety
 * `b` must be `NULL` or a live builder.
 */
int ef_tensor_builder_error(const ef_tensor_builder *b);

/**
 * Set the element type.
 *
 * Takes the integer rather than `ef_dtype` deliberately: a C caller can pass
 * any value, and transmuting an out-of-range one into a Rust enum is undefined
 * behaviour, while validating an integer is not. Pass an `EF_DTYPE_*`
 * enumerator; an unknown code is `EINVAL`.
 *
 * # Safety
 * `b` must be `NULL` or a live builder.
 */
int ef_tensor_builder_dtype(ef_tensor_builder *b, uint32_t dtype);

/**
 * Set the addressing grid.
 *
 * # Safety
 * `dims` must point to `ndim` readable `uint64_t`.
 */
int ef_tensor_builder_shape(ef_tensor_builder *b, const uint64_t *dims, uint32_t ndim);

/**
 * Set strides, **in bytes**.
 *
 * Must have the same rank as the shape: a partial stride array has no
 * meaning, matching the blob format's all-or-nothing rule.
 *
 * # Safety
 * `str_` must point to `ndim` readable `int64_t`.
 */
int ef_tensor_builder_strides(ef_tensor_builder *b, const int64_t *str, uint32_t ndim);

/**
 * Set the backing store.
 *
 * Takes the integer rather than `ef_storage_kind`, for the same reason as
 * [`ef_tensor_builder_dtype`]. Pass an `EF_STORAGE_KIND_*` enumerator.
 *
 * # Safety
 * `b` must be `NULL` or a live builder.
 */
int ef_tensor_builder_storage(ef_tensor_builder *b, uint32_t kind);

/**
 * Add a plane. Required by `wrap`, rejected by `alloc`.
 *
 * Every field is recorded here, but [`ef_tensor_builder_wrap`] cannot honour
 * all of them equally -- the underlying `Tensor` can only represent a
 * subset of what a real V4L2/DRM plane carries:
 *
 * * `handle` -- adopted as the tensor's fd.
 * * `stride` -- carried onto the tensor as its row stride, applied after any
 *   format the builder also carries (`ef_tensor_builder_format`) is
 *   attached -- `set_row_stride` itself requires a format already be set,
 *   so applying it any earlier would reject every `wrap` call that supplied
 *   both a format and a nonzero stride, unconditionally, regardless of
 *   whether the stride was actually valid.
 * * `offset` -- carried onto the tensor as its plane offset, for the same
 *   reason applied after any format (this is the one field `wrap` used to
 *   silently drop).
 * * `size` -- validated against the extent `shape` and `stride` imply, never
 *   stored: the tensor derives its own extent from shape and stride, and a
 *   caller-supplied `size` is only ever a sanity bound. `0` means
 *   "unspecified" and is not checked (matching `stride`'s own convention). A
 *   nonzero `size` **smaller** than required is rejected; **larger** is
 *   accepted -- a padded or over-allocated buffer is a normal thing to hand
 *   a wrapper.
 * * `used` -- rejected unless equal to `size`. The tensor has no
 *   partial-fill/`bytes_used` concept, so any other value is unrepresentable.
 *   `used > size` is rejected right here, by this function, with `EINVAL`;
 *   `used < size` is rejected by `wrap` itself, with `EBADMSG` -- same
 *   underlying fact caught at two different points in the two functions'
 *   own validation order, not an accidental split.
 * * `modifier` -- rejected unless `0` (linear). The `Tensor` type has no
 *   representation for a DRM format modifier: adopting a tiled or
 *   compressed buffer under a nonzero modifier would read it back as
 *   linear, which is silently wrong data in every pixel, so `wrap` refuses
 *   it instead.
 * * a second and later plane -- rejected: `wrap` adopts exactly one handle
 *   per tensor. Combined multi-plane geometry (e.g. NV12's Y and UV planes
 *   at different offsets within one dma-buf) is not something this builder
 *   composes; wrap each plane as its own single-plane tensor instead.
 *
 * # Safety
 * `b` must be `NULL` or a live builder.
 */
int ef_tensor_builder_add_plane(ef_tensor_builder *b,
                                int64_t handle,
                                uint64_t offset,
                                uint64_t stride,
                                uint64_t size,
                                uint64_t used,
                                uint64_t modifier);

/**
 * Set the format descriptor (`"NV12"`, `"rgb8"`); `""` means not an image.
 *
 * # Safety
 * `f` must be `NULL` or a NUL-terminated string.
 */
int ef_tensor_builder_format(ef_tensor_builder *b, const char *f);

/**
 * Set the four colorimetry axes. Any may be `""` for unspecified.
 *
 * # Safety
 * Each argument must be `NULL` or a NUL-terminated string.
 */
int ef_tensor_builder_colorimetry(ef_tensor_builder *b,
                                  const char *space,
                                  const char *transfer,
                                  const char *encoding,
                                  const char *range);

/**
 * Set quantization. `axis` is `-2` for none, `-1` per-tensor, `>= 0` per-channel.
 *
 * # Safety
 * `scales` must point to `n` floats; `zps` must be `NULL` or point to `n` ints.
 */
int ef_tensor_builder_quantization(ef_tensor_builder *b,
                                   int32_t axis,
                                   const float *scales,
                                   const int32_t *zps,
                                   uint32_t n);

/**
 * Set the acquire fence fd, or `-1` for none.
 *
 * # Safety
 * `b` must be `NULL` or a live builder.
 */
int ef_tensor_builder_fence(ef_tensor_builder *b, int fd);

/**
 * Allocate storage and produce a tensor. Returns `NULL` on failure.
 *
 * Requires **no** planes: `alloc` derives storage from the format, shape and
 * alignment. Supplying planes means the caller wanted [`ef_tensor_builder_wrap`],
 * so it is rejected rather than silently ignored.
 *
 * The builder survives and may be called again.
 *
 * # Safety
 * `b` must be `NULL` or a live builder.
 */
ef_tensor *ef_tensor_builder_alloc(ef_tensor_builder *b);

/**
 * Adopt externally-owned handles and produce a tensor. Returns `NULL` on failure.
 *
 * Requires **at least one** plane carrying a real handle — that is the
 * difference from [`ef_tensor_builder_alloc`], and the reason misuse is a
 * per-field error rather than a convention.
 *
 * Adopts the handle: the resulting tensor owns it and the caller must not
 * close it.
 *
 * **Behaviour change**: earlier versions of this function silently ignored
 * `offset`, `size`, `used`, `modifier`, and every plane past the first --
 * a caller who supplied any of them got a tensor that looked valid but read
 * from the wrong place, or under the wrong layout. They are now carried,
 * validated, or rejected; see [`ef_tensor_builder_add_plane`]'s doc for the
 * field-by-field disposition. A third-party caller that was previously
 * getting a silently-wrong tensor now gets one of the errors below instead
 * -- the correct trade, but a real behaviour change for anyone linking this
 * library. Also fixed in the same pass: a `wrap` call that supplied both a
 * format (`ef_tensor_builder_format`) and a nonzero plane `stride` used to
 * fail unconditionally with `EINVAL` regardless of whether the stride was
 * valid, because the stride was applied before the format was; that
 * combination now succeeds when the stride is actually valid for the format.
 *
 * @retval non-`NULL` success.
 * @retval `NULL` on failure; [`ef_tensor_builder_error`] distinguishes why:
 *   - `EINVAL` no plane was added, `handle` is negative or does not fit an
 *     `int`, no dtype/shape was set, `stride` is nonzero and smaller than
 *     the format's minimum row size (only checked once a format is
 *     attached; same check as [`crate::mutate::ef_tensor_set_row_stride`]),
 *     `from_fd` itself failed (e.g. an unrecognized fd type), or the
 *     platform is non-Unix.
 *   - `ERANGE` `size` is nonzero and smaller than the extent `shape` and
 *     `stride` require.
 *   - `EBADMSG` `used` does not equal `size`.
 *   - `EDOM` `modifier` is nonzero -- only linear (`0`) is representable.
 *   - `ENOTSUP` more than one plane was added -- `wrap` adopts a single
 *     handle per tensor.
 *
 * # Safety
 * `b` must be `NULL` or a live builder, and any plane handle must be a valid
 * file descriptor this process may adopt.
 */
ef_tensor *ef_tensor_builder_wrap(ef_tensor_builder *b);

/**
 * Map `t` for CUDA use. Returns an opaque map, or NULL if CUDA is unavailable.
 *
 * The map retains `t`. The caller may `ef_tensor_free` their own handle
 * while the map is outstanding; [`ef_tensor_cuda_unmap`] releases that
 * retain.
 *
 * # Safety
 * `t` must be `NULL` or a live handle.
 */
void *ef_tensor_cuda_map(const ef_tensor *t);

/**
 * Device pointer from a map returned by [`ef_tensor_cuda_map`].
 *
 * # Safety
 * `map` must be `NULL` or a live map. `out_size` may be NULL.
 */
void *ef_tensor_cuda_device_ptr(const void *map, uintptr_t *out_size);

/**
 * Release a map from [`ef_tensor_cuda_map`]. NULL is a no-op.
 *
 * Drops the CUDA mapping first, then releases the retain taken at map.
 *
 * # Safety
 * `map` must be `NULL` or have come from [`ef_tensor_cuda_map`].
 */
void ef_tensor_cuda_unmap(void *map);

/**
 * Create a request for a `width`×`height` image.
 *
 * `format` is the wire descriptor (`"NV12"`, `"rgb8"`), matching every other
 * entry point rather than introducing a second vocabulary. `dtype` is the
 * shared code.
 *
 * @return `NULL` for an unknown format or dtype, or zero dimensions.
 *
 * # Safety
 * `format` must be a NUL-terminated string.
 */
struct ef_tensor_image_desc *ef_tensor_image_desc_new(uintptr_t width,
                                                      uintptr_t height,
                                                      const char *format,
                                                      uint32_t dtype);

/**
 * Free a request. Freeing `NULL` is a no-op.
 *
 * # Safety
 * `d` must be `NULL` or have come from this library.
 */
void ef_tensor_image_desc_free(struct ef_tensor_image_desc *d);

/**
 * Request a specific backing store, by `ef_storage_kind` code.
 *
 * # Safety
 * `d` must be `NULL` or a live request.
 */
int ef_tensor_image_desc_set_memory(struct ef_tensor_image_desc *d, uint32_t kind);

/**
 * Declare CPU access: 0 none, 1 read, 2 write, 3 read-write.
 *
 * # Safety
 * `d` must be `NULL` or a live request.
 */
int ef_tensor_image_desc_set_access(struct ef_tensor_image_desc *d, uint32_t access);

/**
 * Request compression: 0 = none, 1 = any scheme the platform offers.
 *
 * `Any` allocates linear when the format is not eligible and counts the
 * fallback, which is the right default for a pipeline that wants the
 * bandwidth win without a portability failure. Requesting a *specific*
 * scheme is deliberately not exposed here — it fails outright on a device
 * whose scheme differs, and that belongs behind a named entry point rather
 * than an integer a caller might pass by accident.
 *
 * # Safety
 * `d` must be `NULL` or a live request.
 */
int ef_tensor_image_desc_set_compression(struct ef_tensor_image_desc *d, uint32_t compression);

/**
 * Read a request's fields into `out`, mirroring `ef_tensor_plane_at`'s
 * shape: a scalar block a foreign library copies rather than a pointer it
 * would have to dereference into this library's private layout.
 *
 * @return 0 on success, `EINVAL` for a `NULL` argument.
 *
 * # Safety
 * `d` must be `NULL` or a live request. `out` must be a valid pointer to a
 * writable `ef_image_desc_view`.
 */
int ef_tensor_image_desc_get(const struct ef_tensor_image_desc *d, struct ef_image_desc_view *out);

/**
 * Allocate an image tensor from a finished request -- the primitive that
 * makes the `ef_tensor_image_desc` family a real constructor rather than a
 * request-only echo. Triggers [`edgefirst_tensor::TensorDyn::image_desc`]
 * (dispatching on `desc.dtype()`) inside `libedgefirst_tensor.so`, the same
 * geometry-computing code any Rust caller of `TensorDyn::image_desc`
 * reaches -- this does not reimplement or approximate it.
 *
 * @retval a new tensor the caller must free with `ef_tensor_free`, on
 *         success.
 * @retval `NULL` for a `NULL` request, or if the underlying allocation
 *         fails (see `TensorDyn::image_desc`'s error conditions --
 *         incompatible `CpuAccess`/compression combination, an
 *         unsupported compression request, or an invalid `width`x`height`
 *         for the format) -- `ef_tensor_last_error_message` carries the
 *         reason.
 *
 * # Safety
 * `d` must be `NULL` or a live request from this library.
 */
ef_tensor *ef_tensor_image_desc_alloc(const struct ef_tensor_image_desc *d);

/**
 * Number of dimensions in the addressing grid; 0 for an invalid handle.
 *
 * # Safety
 * `t` must be `NULL` or a live handle from an EdgeFirst library.
 */
uint32_t ef_tensor_ndim(const ef_tensor *t);

/**
 * Borrowed pointer to `ndim` dimension extents, valid while `t` lives.
 *
 * Returns NULL for an invalid or NULL handle.
 *
 * # Safety
 * `t` must be `NULL` or a live handle from an EdgeFirst library.
 */
const uint64_t *ef_tensor_shape(const ef_tensor *t);

/**
 * Borrowed pointer to `ndim` strides in BYTES, valid while `t` lives.
 *
 * Returns NULL for an invalid or NULL handle.
 *
 * # Safety
 * `t` must be `NULL` or a live handle from an EdgeFirst library.
 */
const int64_t *ef_tensor_strides(const ef_tensor *t);

/**
 * The `ef_dtype` code of the addressing grid's element type.
 *
 * Returns 0 for an invalid or NULL handle.
 *
 * # Safety
 * `t` must be `NULL` or a live handle from an EdgeFirst library.
 */
uint32_t ef_tensor_dtype(const ef_tensor *t);

/**
 * The `ef_storage_kind` code of the backing store.
 *
 * Returns 0 for an invalid or NULL handle.
 *
 * # Safety
 * `t` must be `NULL` or a live handle from an EdgeFirst library.
 */
uint32_t ef_tensor_storage_kind(const ef_tensor *t);

/**
 * Number of planes; 1 for a bare (formatless) tensor.
 *
 * Returns 0 for an invalid or NULL handle.
 *
 * # Safety
 * `t` must be `NULL` or a live handle from an EdgeFirst library.
 */
uint32_t ef_tensor_plane_count(const ef_tensor *t);

/**
 * Borrowed format descriptor, "" when this is not an image.
 *
 * Returns NULL for an invalid or NULL handle.
 *
 * # Safety
 * `t` must be `NULL` or a live handle from an EdgeFirst library.
 */
const char *ef_tensor_format(const ef_tensor *t);

/**
 * Packed colorimetry (`space | transfer<<8 | encoding<<16 | range<<24`),
 * or 0 when undefined. See `edgefirst_tensor::Colorimetry::pack`.
 *
 * Returns 0 for an invalid or NULL handle -- indistinguishable from a
 * genuinely undefined colorimetry, which is the same ambiguity every other
 * "0 means absent" wire value in this ABI already accepts.
 *
 * # Safety
 * `t` must be `NULL` or a live handle from an EdgeFirst library.
 */
uint32_t ef_tensor_colorimetry(const ef_tensor *t);

/**
 * Describe plane `index`. Returns 0 on success, EINVAL otherwise.
 *
 * # Safety
 * `t` must be `NULL` or a live handle; `out` must be writable.
 */
int ef_tensor_plane_at(const ef_tensor *t, uint32_t index, struct ef_tensor_plane *out);

/**
 * Bytes of the underlying allocation, which is `>=` the tensor's logical
 * size -- a pool tensor holding a smaller decoded image, or a
 * pitch-aligned image whose padding the shape alone cannot express.
 *
 * The producer side of [`edgefirst_tensor::TensorDesc::capacity`]: a
 * consumer re-importing this tensor's memory needs the real allocation
 * size, not the size the shape implies. Not derivable from
 * `ef_tensor_plane_at`: for a *formatted* tensor that reports per-plane
 * geometry over a computed plane table, whose sum is the logical image
 * size and not the allocation's.
 *
 * `-1` is a genuine sentinel, matching `ef_tensor_plane_offset`'s: a byte
 * count can never be negative.
 *
 * @retval `>= 0` the allocation's byte count.
 * @retval `-1` `t` is `NULL` or invalid.
 *
 * # Safety
 * `t` must be `NULL` or a live handle.
 */
int64_t ef_tensor_capacity_bytes(const ef_tensor *t);

/**
 * The *recorded* row stride in bytes, or `-1` when none is recorded
 * (tightly packed).
 *
 * Deliberately distinct from `ef_tensor_plane_at`'s `stride`, which is the
 * *effective* pitch -- the recorded one when there is one, else a pitch
 * computed from the format and width. The difference is load-bearing for
 * [`edgefirst_tensor::TensorDyn::descriptor`]: the cross-package protocol
 * carries `None` for a tight tensor and lets the consumer recompute, and
 * baking a computed pitch in instead would turn "no stride recorded" into
 * "this exact stride is required" across a package boundary.
 *
 * `-1` is a genuine sentinel, matching `ef_tensor_plane_offset`'s: a byte
 * pitch can never be negative.
 *
 * @retval `>= 0` the recorded row stride in bytes.
 * @retval `-1` `t` is `NULL`/invalid, or no row stride is recorded.
 *
 * # Safety
 * `t` must be `NULL` or a live handle.
 */
int64_t ef_tensor_row_stride(const ef_tensor *t);

/**
 * The vendor tile-compression scheme this tensor's allocation actually
 * resolved to; `EF_COMPRESSION_NONE` (0) for a linear layout, which is the
 * answer everywhere except an Android AHardwareBuffer allocation that both
 * requested compression and got it.
 *
 * Returns 0 rather than an error code for a `NULL`/invalid handle: the
 * return type is the vocabulary itself, with no spare bit pattern to carry
 * a failure, and "linear" is the conservative answer -- a consumer that
 * treats an unreadable handle as linear reads plausible bytes, whereas one
 * that treated it as tiled would decode garbage. Callers who need to
 * distinguish an invalid handle have `ef_tensor_dtype`/`ef_tensor_ndim`
 * for that.
 *
 * # Safety
 * `t` must be `NULL` or a live handle.
 */
uint32_t ef_tensor_compression(const ef_tensor *t);

/**
 * Describe this tensor's parent-region snapshot, if it is a `view`/`batch`
 * sub-region. `out->has_origin` is 0 for a whole tensor, in which case the
 * rest of `out` is zeroed and unused.
 *
 * Returns 0 on success (`out` always written), EINVAL for a NULL/invalid
 * handle or a NULL `out`.
 *
 * # Safety
 * `t` must be `NULL` or a live handle; `out` must be writable.
 */
int ef_tensor_view_origin(const ef_tensor *t, struct EfViewOrigin *out);

/**
 * Allocate a host-memory tensor.
 *
 * The simplest constructor: `edgefirst-tensor` handles `mem` allocation
 * itself, with no `ImageProcessor` involved. Returns `NULL` on failure.
 *
 * # Safety
 * `dims` must point to `ndim` readable `uint64_t`.
 */
ef_tensor *ef_tensor_new(uint32_t dtype, const uint64_t *dims, uint32_t ndim);

/**
 * Caller-owned debug name. Free with `free(3)`.
 *
 * # Safety
 * `t` must be `NULL` or a live handle.
 */
char *ef_tensor_name(const ef_tensor *t);

/**
 * Wrap a caller-owned host allocation as a tensor, aliasing it rather than
 * copying or owning it -- the consumer half of the cross-package capsule
 * protocol's `HOST` kind.
 *
 * `capacity` is the producer's real allocation size, which is `>=` the
 * tight footprint `dims` implies: a pool tensor, or one padded to a
 * decoder's pitch alignment, is larger than the shape it currently
 * reports, and without carrying it the alias would be clamped to today's
 * shape and unable to grow back into memory the producer actually has.
 * Pass 0 to mean "exactly the tight footprint".
 *
 * **The returned tensor does not keep `ptr` alive.** It is valid only
 * while the producer keeps that memory alive, which for the capsule
 * protocol is the capsule keepalive's job; nothing here takes a reference
 * to extend it. See
 * [`edgefirst_tensor::TensorDyn::import_descriptor`].
 *
 * @retval a new tensor the caller must free with `ef_tensor_free`.
 * @retval `NULL` for a `NULL` `ptr`/`dims`, `ndim == 0`, or an
 *         unrecognized `dtype` -- `ef_tensor_last_error_message` carries
 *         the reason.
 *
 * # Safety
 * `ptr` must be non-null, aligned for `dtype`, and valid for
 * `max(capacity, product(dims) * sizeof(dtype))` bytes for as long as the
 * returned tensor and every view/map sharing its backing is used. `dims`
 * must point to `ndim` readable `uint64_t`.
 */
ef_tensor *ef_tensor_wrap_host(uint8_t *ptr,
                               uintptr_t capacity,
                               uint32_t dtype,
                               const uint64_t *dims,
                               uint32_t ndim);

/**
 * Wrap a live IOSurface, named by its cross-process `IOSurfaceID`, as a
 * tensor (macOS/iOS only) -- the consumer half of the capsule protocol's
 * `IOSURFACE` kind.
 *
 * Declared on every platform and refused at runtime off Apple, rather than
 * existing only in an Apple build: this library's ABI surface is the same
 * set of symbols everywhere, the same rule `ef_storage_kind` follows for
 * naming `IO_SURFACE` on Linux. A platform-conditional symbol would make
 * "does this build have it" a link-time question for every consumer.
 *
 * IDs are reused after a surface is freed, so a stale one fails rather
 * than resolving to an unrelated buffer.
 *
 * @retval a new tensor the caller must free with `ef_tensor_free`. It
 *         holds its own retain on the surface.
 * @retval `NULL` off Apple platforms, for a `NULL` `dims`/`ndim == 0`/
 *         unrecognized `dtype`, or for an `id` no live surface has --
 *         `ef_tensor_last_error_message` carries the reason.
 *
 * # Safety
 * `dims` must point to `ndim` readable `uint64_t`.
 */
ef_tensor *ef_tensor_from_iosurface_id(uint32_t id,
                                       uint32_t dtype,
                                       const uint64_t *dims,
                                       uint32_t ndim);

/**
 * Release one reference to a tensor handle. Freeing `NULL` is a no-op.
 *
 * `ef_tensor` is refcounted: `ef_tensor_retain` adds a reference and this
 * function is the release. The tensor is destroyed only when the *last*
 * reference is released -- whoever still holds a reference keeps the tensor
 * alive, the `GstBuffer`/`CVPixelBuffer` convention. A handle you hold a
 * reference to remains valid after another reference's `ef_tensor_free`
 * call returns; it is a use-after-free only once *your own* last reference
 * has been released.
 *
 * Works on a handle from **any** EdgeFirst library: every library links
 * this one's shared implementation, so every handle is the exact same
 * layout, allocated by the exact same allocator.
 *
 * # Safety
 * `t` must be `NULL` or a live handle from an EdgeFirst library. `t` must
 * not be used after the release of the caller's own last reference to it.
 */
void ef_tensor_free(ef_tensor *t);

/**
 * Add one reference, keeping the tensor alive until a matching
 * `ef_tensor_free` releases it.
 *
 * # Safety
 * `t` must be NULL or a live handle from an EdgeFirst library.
 */
int ef_tensor_retain(ef_tensor *t);

/**
 * Attach/clear colorimetry metadata on a live handle. `packed` is
 * `Colorimetry::pack`'s wire form; 0 clears it (matching `pack`'s own
 * all-`None`-maps-to-0 convention, so a caller can round-trip
 * [`ef_tensor_colorimetry`] straight back through this without special-
 * casing "no colorimetry").
 *
 * **Concurrency.** `ef_tensor` is refcounted and its handles are designed
 * to cross threads (`ef_tensor_retain` is exactly how two threads come to
 * legitimately hold the same handle). This function and
 * [`ef_tensor_colorimetry`] are safe to call concurrently -- from any
 * number of threads, holding valid references to the same handle, in any
 * interleaving -- with no external locking required around either call.
 *
 * # Safety
 * `t` must be `NULL` or a live handle from an EdgeFirst library.
 */
int ef_tensor_set_colorimetry(ef_tensor *t, uint32_t packed);

/**
 * Wrap an AHardwareBuffer. `NULL` / `ENOTSUP` off Android.
 *
 * # Safety
 * `buffer` and `dims` must be valid when non-NULL.
 */
ef_tensor *ef_tensor_from_hardware_buffer(uint32_t dtype,
                                          void *buffer,
                                          const uint64_t *dims,
                                          uint32_t ndim,
                                          const char *name);

/**
 * Borrowed AHardwareBuffer pointer, or NULL / `ENOTSUP`.
 *
 * # Safety
 * `t` must be `NULL` or a live handle.
 */
void *ef_tensor_hardware_buffer_ptr(const ef_tensor *t);

/**
 * Physical AHardwareBuffer dimensions in texels.
 *
 * # Safety
 * `width` and `height` must be writable when non-NULL.
 */
int ef_tensor_hardware_buffer_physical_dims(const ef_tensor *t,
                                            uintptr_t *width,
                                            uintptr_t *height);

/**
 * Borrowed IOSurfaceRef, or NULL / `ENOTSUP` off Apple.
 *
 * # Safety
 * `t` must be `NULL` or a live handle.
 */
void *ef_tensor_iosurface_ref(const ef_tensor *t);

/**
 * Allocate an image tensor of `width` x `height` in `format`/`dtype`. See
 * [`edgefirst_tensor::TensorDyn::image`].
 *
 * @retval a new tensor the caller must free with `ef_tensor_free`, on
 *         success.
 * @retval `NULL` for a `NULL`/unrecognized `format`, an unknown `dtype` or
 *         `memory` code, an unrecognized `access` code, or if the
 *         underlying allocation fails (invalid `width`x`height` for
 *         `format`, or the requested `memory` is unavailable) --
 *         `ef_tensor_last_error_message` carries the reason.
 *
 * # Safety
 * `format` must be `NULL` or a NUL-terminated string.
 */
ef_tensor *ef_tensor_image_alloc(uintptr_t width,
                                 uintptr_t height,
                                 const char *format,
                                 uint32_t dtype,
                                 int has_memory,
                                 uint32_t memory,
                                 uint32_t access);

/**
 * Allocate a DMA-backed image tensor with an explicit row stride, for a
 * pitch wider than the format's natural `width * channels * sizeof(dtype)`
 * (GPU pitch alignment). See
 * [`edgefirst_tensor::TensorDyn::image_with_stride`].
 *
 * @retval a new tensor the caller must free with `ef_tensor_free`, on
 *         success.
 * @retval `NULL` for the same argument reasons as [`ef_tensor_image_alloc`],
 *         plus a `row_stride_bytes` smaller than the format's minimum row
 *         size, a non-packed `format` (only packed layouts support a
 *         padded stride), or non-DMA `memory` --
 *         `ef_tensor_last_error_message` carries the reason.
 *
 * # Safety
 * `format` must be `NULL` or a NUL-terminated string.
 */
ef_tensor *ef_tensor_image_with_stride_alloc(uintptr_t width,
                                             uintptr_t height,
                                             const char *format,
                                             uint32_t dtype,
                                             uintptr_t row_stride_bytes,
                                             int has_memory,
                                             uint32_t memory,
                                             uint32_t access);

/**
 * Borrow a rectangular `(x, y, width, height)` pixel sub-region of `t` as a
 * new, independent handle that shares `t`'s underlying allocation and
 * identity (zero-copy) -- never a new allocation. See
 * [`edgefirst_tensor::TensorDyn::view`].
 *
 * @retval a new tensor the caller must free with `ef_tensor_free`
 *         (independently of `t`; both stay valid, sharing the same
 *         backing), on success.
 * @retval `NULL` for a `NULL`/invalid `t`, or a region that does not fit
 *         within `t`'s frame -- `ef_tensor_last_error_message` carries the
 *         reason.
 *
 * # Safety
 * `t` must be `NULL` or a live handle.
 */
ef_tensor *ef_tensor_view_region(const ef_tensor *t,
                                 uint64_t x,
                                 uint64_t y,
                                 uint64_t width,
                                 uint64_t height);

/**
 * Borrow batch element `n` of a batched tensor (leading `N` dimension) as a
 * new, independent handle that shares `t`'s underlying allocation and
 * identity (zero-copy) -- never a new allocation. See
 * [`edgefirst_tensor::TensorDyn::batch`].
 *
 * Distinct from `ef_tensor_view_region`, which crops a *spatial*
 * rectangle within one image: this indexes the leading dimension, and the
 * result has `t`'s shape with that dimension dropped. A tensor whose
 * leading dimension is not a batch axis has no meaningful element `n`, and
 * the underlying call refuses.
 *
 * @retval a new tensor the caller must free with `ef_tensor_free`
 *         (independently of `t`; both stay valid, sharing the same
 *         backing), on success.
 * @retval `NULL` for a `NULL`/invalid `t`, or an `n` outside the leading
 *         dimension -- `ef_tensor_last_error_message` carries the reason.
 *
 * # Safety
 * `t` must be `NULL` or a live handle.
 */
ef_tensor *ef_tensor_batch(const ef_tensor *t, uint64_t n);

/**
 * Combine separate luma and chroma plane tensors into one semi-planar
 * (NV12/NV16) tensor. See [`edgefirst_tensor::Tensor::from_planes`].
 *
 * **Ownership: consumes both `luma` and `chroma`, but only past this
 * function's precondition checks.** A real Rust caller of
 * `Tensor::<T>::from_planes` always passes both by value, with no "give
 * them back on error" path -- but every precondition that call requires
 * (matching element types chief among them, since the C boundary is
 * type-erased and the compiler cannot enforce it the way it does for a
 * real Rust caller) is checked here *first*, before either handle is
 * reclaimed. Consuming happens once, right before the underlying
 * `Tensor::from_planes` call; from that point on both outcomes (success or
 * a validation failure *inside* `Tensor::from_planes`, e.g. an
 * incompatible format/shape) leave `luma`/`chroma` invalidated, matching
 * Rust's own by-value semantics. See the `@retval` list below for exactly
 * which failures consume and which do not.
 *
 * @retval a new tensor the caller must free with `ef_tensor_free`, on
 *         success (`luma`/`chroma` consumed).
 * @retval `NULL`, **`luma`/`chroma` left valid and unconsumed** -- a
 *         `NULL`/invalid `luma` or `chroma`, an unrecognized `format`, a
 *         `luma`/`chroma` element-type mismatch, or an
 *         outstanding `ef_tensor_retain`/`ef_tensor_map` on either handle
 *         (consuming a handle another reference still points at, or that
 *         has a live map guard, would dangle it).
 * @retval `NULL`, **`luma`/`chroma` consumed regardless** -- every
 *         precondition above passed but `Tensor::from_planes` itself
 *         refused (see its constraints: only NV12/NV16, matching
 *         luma/chroma widths, and the format-specific height ratio).
 *
 * Every `NULL` case sets `ef_tensor_last_error_message` with the reason.
 *
 * # Safety
 * `luma` and `chroma` must each be `NULL` or a live handle; `format` must
 * be `NULL` or a NUL-terminated string.
 */
ef_tensor *ef_tensor_from_planes(ef_tensor *luma, ef_tensor *chroma, const char *format);

/**
 * Advisory detail for the calling thread's last failing `tensor-capi` call,
 * `""` if none has failed yet.
 *
 * The returned pointer is valid only until this thread's next failing
 * `tensor-capi` call (or the thread's exit); a caller that wants to keep
 * the text must copy it out before calling in again. Never parse this
 * string -- program against the errno return, this is a log line.
 *
 * # Safety
 * The returned pointer must not be read after this thread makes another
 * `tensor-capi` call, or after this thread exits.
 */
const char *ef_tensor_last_error_message(void);

/**
 * Which *kind* of failure the calling thread's last failing
 * `ef_tensor_*` call was; `EF_ERROR_CLASS_UNSPECIFIED` (0) if none has
 * failed yet, or if that failure recorded no class.
 *
 * The companion to [`ef_tensor_last_error_message`], and unlike that
 * string this **is** meant to be programmed against. It exists for the
 * entry points that report failure by returning `NULL`: those have no
 * errno to carry a class, so before this a caller rebuilding a typed error
 * from one had only the advisory message -- and `ef_tensor_batch`'s Rust
 * wrapper really did match on a fragment of it, because there was nowhere
 * else for the distinction to live.
 *
 * Same lifetime rules as the message: thread-local, set by every failing
 * call, unchanged by a successful one. Read it immediately after the call
 * that returned `NULL`, before making another.
 *
 * # Safety
 * Safe to call at any time; declared `unsafe` only for symmetry with the
 * rest of this ABI.
 */
uint32_t ef_tensor_last_error_class(void);

/**
 * Initialise logging to a `FILE*`. First successful call wins (`EALREADY` after).
 *
 * # Safety
 * `stream` must remain valid for the process lifetime.
 */
int ef_log_init_file(FILE *stream, uint32_t max_level);

/**
 * Initialise logging with a callback. First successful call wins.
 *
 * # Safety
 * `cb` must remain valid for the process lifetime.
 */
int ef_log_init_callback(EfLogCallback cb, void *userdata, uint32_t max_level);

/**
 * Map a tensor's whole extent for CPU access.
 *
 * Only one map may be outstanding per tensor at a time; a second call
 * before the matching `ef_tensor_unmap` returns `EBUSY`. `access` selects
 * the mapping direction: `EF_CPU_ACCESS_READ`, `_WRITE`, or `_READ_WRITE`.
 * `EF_CPU_ACCESS_NONE` is not a mappable direction.
 *
 * A read-only map (`EF_CPU_ACCESS_READ`) still populates `out->ptr` --
 * `ef_tensor_view` is one shared shape for both directions -- but writing
 * through it is a contract violation this C signature cannot itself
 * enforce; the Rust-side guard does enforce it (a debug assertion fires if
 * the guard is ever asked for a mutable slice while read-only), so misuse
 * is a caught bug on the Rust side even though nothing stops the raw
 * pointer write from C directly.
 *
 * Exclusive write, CPU-side only: a writable map (`_WRITE`/`_READ_WRITE`)
 * is refused with `EBUSY` unless the tensor's CPU-side handle count
 * (`ef_tensor_retain`/`ef_tensor_free`) is exactly one -- a second EBUSY
 * trigger distinct from the double-map one above, with its own
 * `ef_tensor_last_error_message` text so the two are distinguishable. This
 * is the C surface's honesty limit on exclusivity: the gate sees only the
 * CPU-side handle count, because a refcount cannot see a device -- the
 * GPU/NPU hold no reference of their own, so a dma-buf a device is
 * concurrently writing is not exclusive no matter what this count says.
 * Device-side ordering is not this gate's job; it stays with the fence
 * field.
 *
 * @return 0 on success, `EINVAL` (null tensor/out, or bad access code),
 *         `EBUSY` (a map is already outstanding on this tensor, or a
 *         writable map was requested while the CPU-side handle count is
 *         greater than one -- see above), `EACCES` (a writable map was
 *         requested on a tensor whose declared CPU access is not writable
 *         -- enforced at this boundary; read-direction mismatches follow
 *         the Rust layer's warn-and-allow policy instead. Backend-level
 *         mapping refusals, e.g. AHardwareBuffer lock exclusivity, also
 *         surface as `EACCES`), or another errno translated from the
 *         backend's error.
 *
 * # Safety
 * `t` must be `NULL` or a live handle; `out` must be writable for one
 * `ef_tensor_view`.
 */
int ef_tensor_map(ef_tensor *t, uint32_t access, struct ef_tensor_view *out);

/**
 * Release the outstanding map taken by `ef_tensor_map`.
 *
 * Dropping the guard runs the platform sync bracket (mmap stays resident,
 * but e.g. IOSurface/dma-buf run their unlock/sync-for-device here). The
 * pointer handed out by the matching `ef_tensor_map` is invalid the instant
 * this returns 0.
 *
 * @return 0 on success, `EINVAL` (null tensor, or no map is outstanding).
 *
 * # Safety
 * `t` must be `NULL` or a live handle.
 */
int ef_tensor_unmap(ef_tensor *t);

/**
 * Acquire the buffer for CPU access -- the standalone cache-maintenance
 * bracket, without a mapping.
 *
 * `DMA_BUF_IOCTL_SYNC` with `DMA_BUF_SYNC_START` on Linux; the IOSurface
 * lock on Apple platforms; a no-op for coherent host memory. Pairs with
 * [`ef_tensor_sync_for_device`], which **must** be called with the same
 * `access`: the direction tells the kernel which half of the maintenance
 * this access needs, and a mismatched pair skips one of them (a read-only
 * bracket lets the kernel skip the writeback, a write-only one skips the
 * invalidate).
 *
 * Distinct from `ef_tensor_map`, which establishes an address *and* the
 * coherency window together. This is for a caller that already holds the
 * address -- one that mapped once at init and now brackets each frame's
 * CPU access -- so it takes no map state and leaves none behind.
 *
 * `EF_CPU_ACCESS_NONE` is not a sync direction and is refused with
 * `EINVAL`, exactly as it is for `ef_tensor_map`.
 *
 * @return 0 on success, `EINVAL` (null tensor, or a bad/`NONE` access
 *         code), `ENOTSUP` (a backing with no coherency window independent
 *         of its map -- PBO, and AHardwareBuffer on Android), or another
 *         errno translated from the backend's error.
 *
 * # Safety
 * `t` must be `NULL` or a live handle.
 */
int ef_tensor_sync_for_cpu(const ef_tensor *t, uint32_t access);

/**
 * Release the buffer back to the device -- the CPU is done accessing it.
 *
 * `DMA_BUF_SYNC_END`. See [`ef_tensor_sync_for_cpu`] for the pairing rule
 * and the direction's meaning; `access` must match the one that opened the
 * bracket.
 *
 * @return the same codes as [`ef_tensor_sync_for_cpu`].
 *
 * # Safety
 * `t` must be `NULL` or a live handle.
 */
int ef_tensor_sync_for_device(const ef_tensor *t, uint32_t access);

/**
 * Copy a tensor's whole extent into a caller-provided buffer.
 *
 * Needs no outstanding `ef_tensor_map`: it takes its own short-lived read
 * guard, copies, and drops the guard before returning. On `edgefirst-tensor`
 * backends today (`Mem`, at minimum -- see the test below) a plain read
 * guard coexists freely with an outstanding stored map, because the
 * underlying platform mapping carries no single-writer lock of its own;
 * this call does not special-case that, so if a future backend's map ever
 * does refuse a second concurrent guard, the refusal surfaces here as
 * whatever errno `errno_for` gives that backend's error (in practice
 * `EACCES` or `EBUSY` depending on the backend), not a hardcoded one.
 *
 * @return bytes written (`>= 0`) on success, or a negative errno: `-EINVAL`
 *         (null tensor/out), `-ERANGE` (`cap` is smaller than the tensor's
 *         byte length), or another negative errno translated from the
 *         backend's error.
 *
 * # Safety
 * `t` must be `NULL` or a live handle; `out` must be writable for `cap`
 * bytes.
 */
int64_t ef_tensor_copy_to(ef_tensor *t, uint8_t *out, uintptr_t cap);

/**
 * Attach pixel format metadata to a live handle, validating that its shape
 * is compatible with the format's layout. See
 * [`edgefirst_tensor::TensorDyn::set_format`].
 *
 * @retval 0 success; the format is attached and `ef_tensor_format` /
 *         `ef_tensor_plane_count` / `ef_tensor_plane_at` now reflect it.
 * @retval EINVAL `t` is `NULL`, `format` is `NULL`/non-UTF8/unrecognized, or
 *         the tensor's current shape does not match the format's layout
 *         (packed expects `[H, W, C]`, planar `[C, H, W]`, semi-planar
 *         `[H*k, W]`).
 *
 * @warning Not safe to call concurrently with any other `tensor-capi` call
 * on the same handle from another thread -- see this file's module docs.
 *
 * # Safety
 * `t` must be `NULL` or a live handle; `format` must be `NULL` or a
 * NUL-terminated string.
 */
int ef_tensor_set_format(ef_tensor *t, const char *format);

/**
 * Set the row stride in bytes for a live handle with padded rows (e.g. a
 * V4L2/GStreamer allocator's buffer). Must be called after
 * [`ef_tensor_set_format`]. See
 * [`edgefirst_tensor::TensorDyn::set_row_stride`].
 *
 * @retval 0 success.
 * @retval EINVAL `t` is `NULL`, no pixel format is set on this tensor yet,
 *         or `stride` is smaller than the format's minimum row size at the
 *         tensor's current width.
 *
 * @warning Not safe to call concurrently with any other `tensor-capi` call
 * on the same handle from another thread -- see this file's module docs.
 *
 * # Safety
 * `t` must be `NULL` or a live handle.
 */
int ef_tensor_set_row_stride(ef_tensor *t, uintptr_t stride);

/**
 * Set the row stride in bytes without format validation. See
 * [`edgefirst_tensor::TensorDyn::set_row_stride_unchecked`].
 *
 * Unlike [`ef_tensor_set_row_stride`], this never requires a pixel format
 * and never validates `stride` against one -- for a raw sub-tensor that by
 * contract carries no format (the multiplane chroma plane
 * [`crate::image::ef_tensor_from_planes`] combines, or the standalone plane
 * tensors `ef_tensor_builder_wrap` produces before a format is attached),
 * there is no minimum to check the caller's stride against. Same escape
 * hatch as `static`'s `Tensor::set_row_stride_unchecked` (`lib.rs`): the
 * caller is responsible for the stride being valid for whatever it goes on
 * to describe.
 *
 * Updates `ef_tensor_plane_at`'s cached geometry the same way
 * [`ef_tensor_set_row_stride`] does (`refresh_caches`) -- unlike that
 * setter, a formatless tensor's plane-0 geometry falls back to reporting
 * this stride directly (see `ef_tensor_plane_at`'s own doc comment for why
 * that fallback exists).
 *
 * @retval 0 success.
 * @retval EINVAL `t` is `NULL`.
 *
 * @warning Not safe to call concurrently with any other `tensor-capi` call
 * on the same handle from another thread -- see this file's module docs.
 *
 * # Safety
 * `t` must be `NULL` or a live handle.
 */
int ef_tensor_set_row_stride_unchecked(ef_tensor *t, uintptr_t stride);

/**
 * Set the byte offset within the backing buffer where image data starts.
 * Format-independent, unlike [`ef_tensor_set_row_stride`]. See
 * [`edgefirst_tensor::TensorDyn::set_plane_offset`].
 *
 * Does not touch `ef_tensor_shape`/`ef_tensor_strides`/`ef_tensor_format`'s
 * cached values -- `plane_offset` is read live from `inner` by
 * `ef_tensor_plane_at`, not cached, so no refresh is needed here (contrast
 * [`ef_tensor_set_format`]/[`ef_tensor_set_row_stride`]).
 *
 * @retval 0 success.
 * @retval EINVAL `t` is `NULL`.
 *
 * @warning Not safe to call concurrently with any other `tensor-capi` call
 * on the same handle from another thread -- see this file's module docs.
 *
 * # Safety
 * `t` must be `NULL` or a live handle.
 */
int ef_tensor_set_plane_offset(ef_tensor *t, uintptr_t offset);

/**
 * Read back the byte offset within the backing buffer where image data
 * starts, as set by [`ef_tensor_set_plane_offset`] (or a producer's
 * `ef_tensor_builder_add_plane` at construction). See
 * [`edgefirst_tensor::TensorDyn::plane_offset`].
 *
 * `-1` is a genuine sentinel here, not a presence-flag substitute: a byte
 * offset can never be negative, so it is unambiguous, matching
 * `ef_tensor_plane`'s own `handle: -1` convention for "none" -- unlike
 * axis or scale (where every value including 0 is legitimate), there is no
 * real offset this collides with.
 *
 * @retval `>= 0` the current plane offset in bytes.
 * @retval `-1` `t` is `NULL`, or no offset has been set (the default).
 *
 * # Safety
 * `t` must be `NULL` or a live handle.
 */
int64_t ef_tensor_plane_offset(const ef_tensor *t);

/**
 * Change a tensor's logical shape, keeping the same element count.
 *
 * The product of `dims` must equal the current element count. See
 * [`ef_tensor_set_logical_shape`] for the capacity-based sibling a pool
 * tensor needs.
 *
 * @retval 0 success; `ef_tensor_shape`/`ef_tensor_strides` now reflect the
 *         new geometry.
 * @retval EINVAL `t` or `dims` is `NULL`, `ndim` is 0, or a dimension is
 *         out of range for this host.
 * @retval ERANGE the new shape's element count differs from the current
 *         one -- the tensor is left untouched.
 *
 * @warning Not safe to call concurrently with any other `tensor-capi` call
 * on the same handle from another thread -- see this file's module docs.
 *
 * # Safety
 * `t` must be `NULL` or a live handle; `dims` must point to `ndim`
 * readable `uint64_t`.
 */
int ef_tensor_reshape(ef_tensor *t, const uint64_t *dims, uint32_t ndim);

/**
 * Change a tensor's logical shape to anything its **allocation** can hold.
 *
 * The capacity-based counterpart to [`ef_tensor_reshape`]
 * (`TensorTrait::set_logical_shape`): an oversized reusable pool tensor
 * reconfigured to a smaller image without reallocating, which
 * `ef_tensor_reshape`'s equal-count rule refuses.
 *
 * Two entry points rather than one with a flag, because they are two
 * contracts a caller picks between deliberately -- the same reason
 * `ef_tensor_sync_for_cpu` and `_sync_for_device` are two entries.
 *
 * Task P2b wrote this, found nothing called it, and **deleted it before
 * committing** -- unreferenced ABI surface drifts unwatched. Task P2e gave
 * it a caller: `Tensor<T>` never overrode
 * `TensorTrait::set_logical_shape`, so both backends silently applied
 * `reshape`'s strict rule under a name promising the opposite, and fixing
 * that on the `dynamic` side needs exactly this primitive. Added back on
 * the strength of the caller, not of the idea.
 *
 * @retval 0 success.
 * @retval EINVAL `t` or `dims` is `NULL`, `ndim` is 0, or a dimension is
 *         out of range for this host.
 * @retval ERANGE the new shape does not fit the existing allocation --
 *         the tensor is left untouched.
 *
 * @warning Not safe to call concurrently with any other `tensor-capi` call
 * on the same handle from another thread -- see this file's module docs.
 *
 * # Safety
 * `t` must be `NULL` or a live handle; `dims` must point to `ndim`
 * readable `uint64_t`.
 */
int ef_tensor_set_logical_shape(ef_tensor *t, const uint64_t *dims, uint32_t ndim);

/**
 * Duplicate the file descriptor backing this tensor, for any storage kind
 * that has one.
 *
 * Deliberately **not** derivable from `ef_tensor_plane_at`: that reports a
 * plane's *native handle*, which is a dma-buf fd on Linux and an IOSurface
 * id on Apple and `-1` for everything else -- so a consumer deriving
 * "clone this tensor's fd" from it refuses SHM-backed tensors, which do
 * have a real fd. The library owns the storage and knows which kinds have
 * one; asking it is the whole point of the split.
 *
 * @retval `>= 0` a new file descriptor the caller owns and must `close()`.
 * @retval a negative errno: `-EINVAL` for a `NULL`/invalid handle,
 *         `-ENOTSUP` for a backing with no file descriptor at all,
 *         or another negative errno from the underlying `dup`.
 *
 * # Safety
 * `t` must be `NULL` or a live handle.
 */
int ef_tensor_clone_fd(const ef_tensor *t);

/**
 * Retag a tensor's element type without touching its bytes.
 *
 * The recorded dtype is metadata over the same allocation: `EF_DTYPE_U8`
 * and `EF_DTYPE_I8` address identical bytes and differ only in how a
 * consumer reads them. `edgefirst-image` allocates a PBO or DMA buffer as
 * `u8` and hands it back as `i8`, with the int8 shader applying an XOR 0x80
 * bias over the same buffer -- this is the primitive that makes the handle
 * agree with that.
 *
 * A dtype of a different width is refused. That is not a retag but a
 * reinterpretation: `ef_tensor_shape` times the element width is what a
 * consumer multiplies out, so widening or narrowing here would silently
 * change the element count over an allocation whose size did not change.
 *
 * @retval 0 success; `ef_tensor_dtype` now reports `dtype`.
 * @retval EINVAL `t` is `NULL`, or `dtype` is not a recognized code.
 * @retval ERANGE `dtype` has a different element width than the current one
 *         -- the tensor is left untouched.
 *
 * @warning Not safe to call concurrently with any other `tensor-capi` call
 * on the same handle from another thread -- see this file's module docs.
 *
 * # Safety
 * `t` must be `NULL` or a live handle.
 */
int ef_tensor_set_dtype(ef_tensor *t, uint32_t dtype);

/**
 * Reconfigure a live handle's logical dimensions and pixel format, reusing
 * its existing allocation -- the pool-reuse primitive a JPEG
 * decode-into-pool destination tensor needs before each decode. See
 * [`edgefirst_tensor::TensorDyn::configure_image`].
 *
 * @retval 0 success; `ef_tensor_shape`/`ef_tensor_strides`/`ef_tensor_format`
 *         /`ef_tensor_plane_at` now reflect the new geometry.
 * @retval EINVAL `t` is `NULL`, `format` is `NULL`/non-UTF8/unrecognized, or
 *         `width`x`height` is not a valid size for `format`.
 * @retval ERANGE the existing allocation cannot hold `width`x`height` in
 *         `format` (a pool tensor sized too small for this request).
 *
 * @warning Not safe to call concurrently with any other `tensor-capi` call
 * on the same handle from another thread -- see this file's module docs.
 *
 * # Safety
 * `t` must be `NULL` or a live handle; `format` must be `NULL` or a
 * NUL-terminated string.
 */
int ef_tensor_configure_image(ef_tensor *t, uintptr_t width, uintptr_t height, const char *format);

/**
 * Whether CUDA interop symbols resolved.
 */
int ef_is_cuda_available(void);

/**
 * Whether Linux DMA-BUF allocation is available.
 */
int ef_is_dma_available(void);

/**
 * Whether a platform GPU-coherent buffer kind can be allocated.
 */
int ef_is_gpu_buffer_available(void);

/**
 * Whether IOSurface allocation is available.
 */
int ef_is_iosurface_available(void);

/**
 * Whether POSIX shared memory allocation is available.
 */
int ef_is_shm_available(void);

/**
 * Whether this platform can honour a tile-compression request for `format`/`dtype`.
 *
 * `format` is a wire code (`"NV12"`, `"rgba8"`). Returns 1 when a request can
 * be honoured, 0 otherwise (including unknown format/dtype).
 *
 * # Safety
 * `format` must be `NULL` or a NUL-terminated string.
 */
int ef_platform_compression_support(const char *format, uint32_t dtype);

/**
 * `HAL_COMPRESSION_ANY` requests that fell back to a linear layout.
 */
uint64_t ef_compression_fallback_count(void);

/**
 * Maps that exceeded a buffer's declared CPU access.
 */
uint64_t ef_unplanned_cpu_access_count(void);

/**
 * Report whether a live handle carries quantization metadata, and how many
 * `scale`/`zero_point` entries it has.
 *
 * The first half of the two-call idiom; see this module's docs.
 *
 * @retval 0 success (`out` is always fully written, whether or not
 *         quantization is present).
 * @retval EINVAL `t` or `out` is `NULL`.
 *
 * # Safety
 * `t` must be `NULL` or a live handle; `out` must be writable for one
 * `ef_quantization_info`.
 */
int ef_tensor_quantization_info(const ef_tensor *t, struct ef_quantization_info *out);

/**
 * Fill caller-provided buffers with a live handle's quantization scales and
 * zero-points. The second half of the two-call idiom; `n` must equal the
 * `count` [`ef_tensor_quantization_info`] reported.
 *
 * @retval 0 success; `scales[0..n]` is filled, and `zps[0..n]` too when
 *         `zps` is non-`NULL` (zero-filled for a symmetric quantization).
 * @retval EINVAL `t` or `scales` is `NULL`, this tensor has no quantization
 *         attached, or `n` does not match its actual entry count.
 *
 * # Safety
 * `t` must be `NULL` or a live handle; `scales` must point to `n` writable
 * `float`s; `zps` must be `NULL` or point to `n` writable `int`s.
 */
int ef_tensor_quantization_get(const ef_tensor *t, float *scales, int32_t *zps, uint32_t n);

/**
 * Attach quantization metadata to a live handle. `axis` is `-1` for
 * per-tensor (`n` must be 1) or `>= 0` for per-channel; `zps` may be `NULL`
 * for symmetric quantization (zero-point implicitly 0).
 *
 * Only meaningful for an integer-dtype tensor; a float tensor is refused,
 * matching [`edgefirst_tensor::TensorDyn::set_quantization`]'s own
 * `QuantizationInvalid { field: "dtype_is_integer", .. }` refusal.
 *
 * @retval 0 success.
 * @retval EINVAL `t` or `scales` is `NULL`, `n == 0`, `axis` is `< -1`,
 *         `axis == -1` with `n != 1`, or the backend rejects the resulting
 *         `Quantization` (dtype not integer, or `axis`/`n` incompatible
 *         with the tensor's shape).
 *
 * @warning Not safe to call concurrently with any other `tensor-capi` call
 * on the same handle from another thread -- see this module's docs.
 *
 * # Safety
 * `t` must be `NULL` or a live handle; `scales` must point to `n` readable
 * `float`s; `zps` must be `NULL` or point to `n` readable `int`s.
 */
int ef_tensor_quantization_set(ef_tensor *t,
                               int32_t axis,
                               const float *scales,
                               const int32_t *zps,
                               uint32_t n);

/**
 * Clear any quantization metadata on a live handle. A no-op if none is
 * attached.
 *
 * @retval 0 success.
 * @retval EINVAL `t` is `NULL`.
 *
 * @warning Not safe to call concurrently with any other `tensor-capi` call
 * on the same handle from another thread -- see this module's docs.
 *
 * # Safety
 * `t` must be `NULL` or a live handle.
 */
int ef_tensor_quantization_clear(ef_tensor *t);

/**
 * Serialize a tensor into a caller-provided blob buffer and handle table.
 *
 * Follows the standard two-call C pattern: pass `blob_cap`/`fds_cap` of 0 (or
 * `NULL` buffers) to learn the sizes required, then call again with buffers
 * that large. `blob_len` and `fds_len` are always written when non-`NULL`, so
 * a caller learns the requirement even from a failed call.
 *
 * The transport mode is chosen from the tensor's storage: a backing with a
 * shareable handle is exported by reference, and one without — `mem`, `pbo` —
 * is inlined, because there is nothing to refer to. A caller that needs bytes
 * from a shareable tensor (sending over a network, where a handle is
 * meaningless off-host) should copy into a `mem` tensor first; a mode-selecting
 * variant can be appended later without breaking this signature.
 *
 * @return 0 on success, `ENOSPC` when a buffer is too small (with the
 *         required lengths written), `EINVAL` on a null tensor or null
 *         out-parameter, `EIO` if the tensor cannot be serialized.
 *
 * # Safety
 * `blob` must be writable for `blob_cap` bytes and `fds` for `fds_cap` ints.
 */
int ef_tensor_export(const ef_tensor *t,
                     uint8_t *blob_out,
                     uintptr_t blob_cap,
                     uintptr_t *blob_len,
                     int *fds_out,
                     uintptr_t fds_cap,
                     uintptr_t *fds_len);

/**
 * Reconstruct a tensor from a blob and its handle table.
 *
 * **Import dups** every handle it retains, so the result is independent and
 * the sender may close its own copies as soon as this returns. There is no
 * keepalive protocol.
 *
 * `blob` is untrusted — it may have arrived from another library, another
 * process, or a network — so every length and index inside it is validated
 * against the buffer and against `fds_len` before use.
 *
 * @return a new tensor the caller must free with `ef_tensor_free`, or `NULL`.
 *
 * # Safety
 * `blob` must be readable for `blob_len` bytes and `fds` for `fds_len` ints.
 */
ef_tensor *ef_tensor_import(const uint8_t *blob_in,
                            uintptr_t blob_len,
                            const int *fds_in,
                            uintptr_t fds_len);

/**
 * Start a process-wide trace to `path`. Only one session per process.
 *
 * # Safety
 * `path` must be a NUL-terminated UTF-8 string.
 */
int ef_start_tracing(const char *path);

/**
 * Stop tracing and flush. No-op if inactive or tracing is not compiled in.
 */
void ef_stop_tracing(void);

/**
 * 1 if a session is active, else 0.
 */
int ef_is_tracing_active(void);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* EDGEFIRST_TENSOR_H */
