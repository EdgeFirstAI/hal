#ifndef EDGEFIRST_TENSOR_H
#define EDGEFIRST_TENSOR_H

/**
 * @file tensor.h
 * @brief EdgeFirst tensor C API
 *
 * SPDX-License-Identifier: Apache-2.0
 * Copyright (c) 2026 Au-Zone Technologies. All Rights Reserved.
 *
 * The tensor header. Every sibling library's header includes it.
 *
 * The handle is opaque. Every accessor is an exported function of
 * libedgefirst_tensor. A tensor from ef_image_processor_create_image() and
 * one from ef_tensor_new() are the same kind of object.
 *
 * Three surfaces:
 *   - Opaque handle + exported accessors, for in-process access.
 *   - A builder, for construction, with sticky errors.
 *   - A (blob, fds) pair, for IPC. Fds travel out of band.
 *
 * Platform-specific entry points. Every function is declared on every
 * platform and refuses at run time off its platform. Each function states
 * its platforms on a "Platforms:" line:
 *   - Linux:   dma-buf fds (the DMA_BUF export kind); ef_tensor_clone_fd
 *              duplicates the fd of any Unix backing that has one
 *   - macOS, iOS: IOSurface (ef_tensor_iosurface_ref, ef_tensor_from_iosurface_id)
 *   - Android: AHardwareBuffer (ef_tensor_from_hardware_buffer, ef_tensor_hardware_buffer_*)
 *   - Windows: D3D11 textures (ef_d3d11_*, ef_tensor_d3d11_*, ef_tensor_gpu_completion,
 *              ef_tensor_gpu_write_value, ef_tensor_set_gpu_write, ef_tensor_from_d3d11_*)
 *   - any host with a CUDA driver: ef_tensor_cuda_*
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
 * Every kind is declared on every platform; whether one can be allocated is a
 * runtime question.
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
  /**
   * Windows `ID3D11Texture2D`, named specifically. A tensor's own
   * storage kind reports `DmaBuf` on Windows; this is the code the
   * cross-package descriptor carries for that backing.
   */
  EF_STORAGE_KIND_D3D11_TEXTURE = 6,
};
#ifndef __cplusplus
#if __STDC_VERSION__ >= 202311L
typedef enum ef_storage_kind ef_storage_kind;
#else
typedef uint32_t ef_storage_kind;
#endif // __STDC_VERSION__ >= 202311L
#endif // __cplusplus

/**
 * CPU access: a tensor's declaration, or the direction of a map window.
 *
 * Mirrors `edgefirst_tensor::CpuAccess`; the values are the wire codes.
 * `None` names the no-CPU-access declaration and is never a valid map
 * direction.
 *
 * The declaration, fixed when an image is allocated, covers all CPU access
 * to it, the library's own included, and is enforced: `ef_tensor_map`,
 * `ef_tensor_sync_*` and the copies fail with `EACCES` outside it, and a
 * library CPU pass (a CPU convert or draw, a decode) that would need more
 * fails instead of running. Hardware (GPU, NPU, G2D, a compositor) access is
 * always allowed.
 *
 * - `EF_CPU_ACCESS_NONE`: no CPU access. On Linux a DMA-BUF comes from the
 *   uncached CMA heap where available.
 * - `EF_CPU_ACCESS_READ`: CPU reads; cached memory.
 * - `EF_CPU_ACCESS_WRITE`: CPU writes that cover the whole buffer (decode
 *   targets, CPU-drawn overlays). On Linux a DMA-BUF comes from the uncached
 *   CMA heap where available, and CPU writes are write-combined.
 * - `EF_CPU_ACCESS_READ_WRITE`: both; cached memory. The default.
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
 * Vendor tile-compression scheme in force for a tensor's allocation.
 *
 * A result, not a request: [`EfImageDescView::compression`] carries what an
 * allocation asked for, and this names the scheme the allocator resolved to.
 * `None` means a linear layout.
 *
 * Mirrors `edgefirst_tensor::CompressionScheme`'s variants, plus the `None`
 * the Rust side spells as `Option::None`.
 */
enum ef_compression
#if defined(__cplusplus) || __STDC_VERSION__ >= 202311L
  : uint32_t
#endif // defined(__cplusplus) || __STDC_VERSION__ >= 202311L
 {
  /**
   * Linear layout: no vendor tile compression.
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
 * Compression request, as `ef_tensor_image_desc_set_compression` takes it and
 * [`EfImageDescView::compression`] reports it.
 *
 * Distinct from [`EfCompression`], which names the scheme an allocation
 * resolved to. No C setter can request `Scheme`; the view reports it for a
 * request made through the Rust API.
 */
enum ef_compression_request
#if defined(__cplusplus) || __STDC_VERSION__ >= 202311L
  : uint32_t
#endif // defined(__cplusplus) || __STDC_VERSION__ >= 202311L
 {
  /**
   * No compression request: allocate linear.
   */
  EF_COMPRESSION_REQUEST_NONE = 0,
  /**
   * Any scheme the platform offers, linear when the format is not eligible.
   */
  EF_COMPRESSION_REQUEST_ANY = 1,
  /**
   * A specific vendor scheme, not further decodable through the view.
   */
  EF_COMPRESSION_REQUEST_SCHEME = 2,
};
#ifndef __cplusplus
#if __STDC_VERSION__ >= 202311L
typedef enum ef_compression_request ef_compression_request;
#else
typedef uint32_t ef_compression_request;
#endif // __STDC_VERSION__ >= 202311L
#endif // __cplusplus

/**
 * Whether a tensor's memory is known to be physically contiguous
 * (`ef_tensor_contiguity`).
 *
 * Mirrors `edgefirst_tensor::Contiguity`.
 */
enum ef_contiguity
#if defined(__cplusplus) || __STDC_VERSION__ >= 202311L
  : uint32_t
#endif // defined(__cplusplus) || __STDC_VERSION__ >= 202311L
 {
  /**
   * Not known: an imported fd, or memory not allocated from a DMA heap.
   */
  EF_CONTIGUITY_UNKNOWN = 0,
  /**
   * Allocated from the CMA heap.
   */
  EF_CONTIGUITY_CONTIGUOUS = 1,
  /**
   * Allocated from the system heap.
   */
  EF_CONTIGUITY_NON_CONTIGUOUS = 2,
};
#ifndef __cplusplus
#if __STDC_VERSION__ >= 202311L
typedef enum ef_contiguity ef_contiguity;
#else
typedef uint32_t ef_contiguity;
#endif // __STDC_VERSION__ >= 202311L
#endif // __cplusplus

/**
 * Which kind of failure the calling thread's last failing `ef_tensor_*` call
 * was. The companion to `ef_tensor_last_error_message`.
 *
 * A class, not a one-to-one mirror of `edgefirst_tensor::Error`: several Rust
 * variants collapse into one here. `Unspecified` is reported when a failure
 * path recorded only a message, and every such path resets the class to it.
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
   * The operation is legal but not permitted right now: a live map, a
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
 * Client-owned state a library entry point calls back into. Passed by value;
 * its size and field offsets are fixed.
 *
 * # What `retain`/`release` govern
 *
 * They extend the life of the callback channel only. They never transfer
 * ownership of the GL buffer, the CUDA resource, or anything else the context
 * addresses:
 *
 * * The producing side's own destructor stays the sole caller of the real
 *   `delete_buffer`.
 * * A library that reconstructed operations from this struct treats
 *   `delete_buffer` as a no-op: it borrows the resource.
 *
 * # Ownership at an entry point
 *
 * An entry point that stores this struct calls `retain(ctx)` itself and calls
 * `release(ctx)` exactly once when the object holding it is destroyed. The
 * caller keeps its own count and may drop it at any time after the call
 * returns.
 *
 * `ctx` is opaque to the library: only the producing copy of the code
 * dereferences it.
 */
typedef struct ef_client_state {
  /**
   * Opaque to every caller except this struct's own function pointers.
   */
  const void *ctx;
  /**
   * Add one reference to the callback channel. Never NULL in a valid
   * state; an entry point receiving NULL must fail with `EINVAL`.
   */
  void (*retain)(const void *ctx);
  /**
   * Drop one reference to the callback channel. Never NULL in a valid
   * state; an entry point receiving NULL must fail with `EINVAL`.
   */
  void (*release)(const void *ctx);
} ef_client_state;

/**
 * Map a CUDA graphics resource to a device pointer. `0` on success.
 *
 * May be NULL where an entry point takes one; such an entry point refuses a
 * NULL.
 */
typedef int (*EfCudaMapFnNullable)(const void *ctx,
                                   void *resource,
                                   void **out_ptr,
                                   uintptr_t *out_len);

/**
 * Unmap a resource previously mapped by an `ef_cuda_map_fn`.
 *
 * May be NULL where an entry point takes one; such an entry point refuses a
 * NULL.
 */
typedef void (*EfCudaUnmapFnNullable)(const void *ctx, void *resource);

/**
 * Unregister a CUDA graphics resource.
 *
 * May be NULL where an entry point takes one; such an entry point refuses a
 * NULL.
 */
typedef void (*EfCudaUnregisterFnNullable)(const void *ctx, void *resource);

/**
 * The D3D11 texture behind a Windows texture tensor.
 *
 * The scalar block `ef_tensor_d3d11_layout` fills. Mirrors
 * `edgefirst_tensor::d3d11_layout::D3d11ImageLayout`.
 *
 * The dimensions are the texture's, in texels and rows, not the image's in
 * pixels: a combined-plane semi-planar image is one texture whose row count
 * covers both planes, and a YUYV image is one texel per two pixels. Read the
 * image's own dimensions from `ef_tensor_shape`.
 *
 * An NV12 image of even width and height is a native two-plane texture,
 * `dxgi_format` 103 (`DXGI_FORMAT_NV12`): `texture_width` x `texture_height`
 * is the luma grid, the image's own size, and the chroma plane is a second
 * plane of the same texture at half each dimension, which GL and video APIs
 * address by plane index. A CPU map of it is laid out as luma rows then
 * chroma rows at `ef_tensor_row_stride`.
 *
 * Every other semi-planar image (`nv16`, `nv24`, and `nv12` at an odd size)
 * is the combined-plane `R8` texture, whose `texture_width` is the driver's
 * row pitch: at least `even(width)`, and possibly more. It is the pitch the
 * combined plane's rows are spaced by and the width a sampler addresses the
 * texture at. `ef_tensor_row_stride` carries the same number.
 *
 * Passed by value; its size and field offsets are fixed.
 */
typedef struct ef_d3d11_layout {
  /**
   * `DXGI_FORMAT` the texture was created with.
   */
  uint32_t dxgi_format;
  /**
   * Texture width in texels.
   */
  uint32_t texture_width;
  /**
   * Texture height in rows.
   */
  uint32_t texture_height;
  /**
   * Bytes per texel of `dxgi_format`.
   */
  uint32_t bytes_per_texel;
  /**
   * The GL internal format an importer binds this texture as; 0 when the
   * format has no GL equivalent.
   */
  uint32_t gl_internal_format;
} ef_d3d11_layout;

/**
 * Flattened, `#[repr(C)]` view of an image-request descriptor's fields.
 *
 * `ef_tensor_image_desc` is opaque; this is the scalar block
 * `ef_tensor_image_desc_get` fills.
 *
 * `memory` and `compression` are each a value plus an explicit presence flag.
 * `compression` is an [`EfCompressionRequest`]: `Any`, or `Scheme` for a
 * specific vendor scheme, which this view does not decode further.
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
   * `ef_compression_request`, meaningful only when `has_compression != 0`.
   */
  uint32_t compression;
  /**
   * Non-zero when a compression request was made.
   */
  uint32_t has_compression;
} ef_image_desc_view;

/**
 * One plane's location, mirroring `TensorPlane` on the wire.
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
 * The scalar block `ef_tensor_view_origin` fills. The other fields are
 * meaningful only when `has_origin` is non-zero. Mirrors
 * `edgefirst_tensor::ViewOrigin`.
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
 * Passed by value; its size is fixed. The pointer is valid from
 * `ef_tensor_map` until the matching `ef_tensor_unmap`; writing through it is
 * allowed only when the map was taken with a writable access.
 */
typedef struct ef_tensor_view {
  uint8_t *ptr;
  uintptr_t len;
} ef_tensor_view;

/**
 * Map a PBO for CPU access. `0` on success; `-1` when the GL context is
 * gone; any other non-zero value a generic failure.
 *
 * May be NULL where an entry point takes one; such an entry point refuses a
 * NULL.
 */
typedef int (*ef_pbo_map_fn)(const void *ctx,
                             uint32_t buffer_id,
                             uintptr_t size,
                             uint8_t **out_ptr,
                             uintptr_t *out_len);

/**
 * Unmap a PBO previously mapped by an `ef_pbo_map_fn`. Same return codes.
 *
 * May be NULL where an entry point takes one; such an entry point refuses a
 * NULL.
 */
typedef int (*ef_pbo_unmap_fn)(const void *ctx, uint32_t buffer_id);

/**
 * Presence and shape summary of a tensor's quantization metadata.
 *
 * The first half of the two-call idiom `ef_tensor_quantization_info` /
 * `ef_tensor_quantization_get` use: this scalar block reports whether
 * quantization is attached and how many entries the second call's buffers
 * need.
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
 * Incremented when the C ABI breaks; adding an exported function does not
 * change it.
 */
uint32_t ef_tensor_abi_version(void);

/**
 * Create a builder. Returns `NULL` only on allocation failure.
 */
ef_tensor_builder *ef_tensor_builder_new(void);

/**
 * Free a builder. Freeing `NULL` is a no-op.
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
 * Set the element type. `dtype` is an `EF_DTYPE_*` code; an unknown code
 * is `EINVAL`.
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
 * Set strides, in bytes. Must have the same rank as the shape.
 *
 * # Safety
 * `str_` must point to `ndim` readable `int64_t`.
 */
int ef_tensor_builder_strides(ef_tensor_builder *b, const int64_t *str, uint32_t ndim);

/**
 * Set the backing store. `kind` is an `EF_STORAGE_KIND_*` code; an unknown
 * code is `EINVAL`.
 *
 * # Safety
 * `b` must be `NULL` or a live builder.
 */
int ef_tensor_builder_storage(ef_tensor_builder *b, uint32_t kind);

/**
 * Add a plane. Required by `wrap`, rejected by `alloc`.
 *
 * [`ef_tensor_builder_wrap`] treats the fields as follows:
 *
 * * `handle` -- adopted as the tensor's fd.
 * * `stride` -- carried onto the tensor as its row stride, applied after
 *   any format the builder carries. `0` means unspecified.
 * * `offset` -- carried onto the tensor as its plane offset, applied after
 *   any format.
 * * `size` -- validated against the extent `shape` and `stride` imply, and
 *   not stored. `0` means unspecified and is not checked. A nonzero `size`
 *   smaller than required is rejected; a larger one is accepted.
 * * `used` -- rejected unless equal to `size`: `used > size` by this
 *   function with `EINVAL`, `used < size` by `wrap` with `EBADMSG`.
 * * `modifier` -- rejected unless `0` (linear).
 * * a second and later plane -- rejected: `wrap` adopts one handle per
 *   tensor.
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
 * Fails when planes were added. The builder survives and may be used again.
 *
 * # Safety
 * `b` must be `NULL` or a live builder.
 */
ef_tensor *ef_tensor_builder_alloc(ef_tensor_builder *b);

/**
 * Adopt an externally-owned handle and produce a tensor. Returns `NULL` on
 * failure.
 *
 * Requires exactly one plane carrying a handle.
 *
 * Adopts the handle only on success: the resulting tensor owns it and the
 * caller must not close it. On failure the handle is left open and remains
 * the caller's to close.
 *
 * See [`ef_tensor_builder_add_plane`] for how each plane field is treated.
 * A nonzero plane `stride` supplied together with a format
 * (`ef_tensor_builder_format`) is validated against that format.
 *
 * @retval non-`NULL` success.
 * @retval `NULL` on failure; [`ef_tensor_builder_error`] distinguishes why:
 *   - `EINVAL` no plane was added, `handle` is negative or does not fit an
 *     `int`, no dtype/shape was set, `stride` is nonzero and smaller than
 *     the format's minimum row size (checked only when a format is
 *     attached), or the fd could not be imported.
 *   - `ERANGE` `size` is nonzero and smaller than the extent `shape` and
 *     `stride` require.
 *   - `EBADMSG` `used` does not equal `size`.
 *   - `EDOM` `modifier` is nonzero.
 *   - `ENOTSUP` more than one plane was added.
 *   - `ENOSYS` the platform is non-Unix.
 *
 * # Safety
 * `b` must be `NULL` or a live builder, and any plane handle must be a valid
 * file descriptor this process may adopt.
 */
ef_tensor *ef_tensor_builder_wrap(ef_tensor_builder *b);

/**
 * Platforms: any host with a CUDA driver.
 *
 * Map `t` for CUDA use. Returns an opaque map, or NULL if CUDA is unavailable.
 *
 * The map retains `t`. The caller may `ef_tensor_free` their own handle
 * while the map is outstanding; [`ef_tensor_cuda_unmap`] releases that
 * retain.
 *
 * On Windows the mapping of a D3D11 texture tensor is tight rows of
 * `width * bytes_per_texel`, and the size [`ef_tensor_cuda_device_ptr`]
 * reports is their sum. `ef_tensor_row_stride` is the D3D11 staging pitch a
 * CPU map sees, which is larger on a padded backing: a consumer must not
 * stride the device pointer by it.
 *
 * For a semi-planar texture (NV12, NV16, NV24) the mapping is
 * `ef_tensor_row_stride` times the combined height, and striding by
 * `ef_tensor_row_stride` is correct.
 *
 * # Safety
 * `t` must be `NULL` or a live handle.
 */
void *ef_tensor_cuda_map(const ef_tensor *t);

/**
 * Platforms: any host with a CUDA driver.
 *
 * Writable mapping; `ef_tensor_cuda_unmap` writes the device buffer back
 * into the tensor on backings that do not alias (Windows D3D11 textures).
 *
 * On Windows the mapping of a D3D11 texture tensor is tight rows of
 * `width * bytes_per_texel`, and the size [`ef_tensor_cuda_device_ptr`]
 * reports is their sum. `ef_tensor_row_stride` is the D3D11 staging pitch a
 * CPU map sees, which is larger on a padded backing: a consumer must not
 * stride the device pointer by it.
 *
 * For a semi-planar texture (NV12, NV16, NV24) the mapping is
 * `ef_tensor_row_stride` times the combined height, and striding by
 * `ef_tensor_row_stride` is correct.
 *
 * Retains `t` and returns NULL when CUDA is unavailable, as
 * [`ef_tensor_cuda_map`] does.
 *
 * # Safety
 * `t` must be `NULL` or a live handle.
 */
void *ef_tensor_cuda_map_mut(const ef_tensor *t);

/**
 * Platforms: any host with a CUDA driver.
 *
 * Device pointer from a map returned by [`ef_tensor_cuda_map`] or
 * [`ef_tensor_cuda_map_mut`].
 *
 * `*out_size` is the mapping's length in bytes. On Windows that is the
 * D3D11 texture's tight rows of `width * bytes_per_texel` summed over the
 * rows, not `ef_tensor_row_stride` times the row count.
 *
 * For a semi-planar texture (NV12, NV16, NV24) the mapping is
 * `ef_tensor_row_stride` times the combined height.
 *
 * # Safety
 * `map` must be `NULL` or a live map. `out_size` may be NULL.
 */
void *ef_tensor_cuda_device_ptr(const void *map, uintptr_t *out_size);

/**
 * Platforms: any host with a CUDA driver.
 *
 * Release a map from [`ef_tensor_cuda_map`] or [`ef_tensor_cuda_map_mut`].
 * NULL is a no-op.
 *
 * Drops the CUDA mapping first, then releases the retain taken at map. For
 * a writable map, the device buffer is copied into the tensor and the copy
 * is synchronized before this returns.
 *
 * # Safety
 * `map` must be `NULL` or have come from [`ef_tensor_cuda_map`] or
 * [`ef_tensor_cuda_map_mut`].
 */
void ef_tensor_cuda_unmap(void *map);

/**
 * Attach a GL-buffer CUDA registration to an existing tensor.
 *
 * `state` is the callback channel this library keeps for the registration's
 * life: it calls `state.retain` before returning and `state.release` when
 * the tensor (and therefore the `CudaHandle`) is freed.
 *
 * `resource` is the client-owned GL buffer handle. This entry point never
 * deletes it; the caller remains the sole caller of `glDeleteBuffers`. On
 * success, the CUDA graphics registration for that buffer is consumed by
 * the library tensor and torn down through `unregister_fn` when the tensor
 * is freed: do not call `cudaGraphicsUnregisterResource` on that
 * registration after success. On failure (`EINVAL`), nothing is stored and
 * the registration remains caller-owned.
 *
 * `map_fn` / `unmap_fn` / `unregister_fn` must be non-NULL. They run on
 * the caller's thread (typically the GL worker).
 *
 * @retval 0 success.
 * @retval EINVAL `t` is NULL, the channel is incomplete, or an op is NULL.
 *
 * @warning Not safe to call concurrently with any other `tensor-capi` call
 * on the same handle from another thread, or while a live map from
 * [`ef_tensor_cuda_map`] / [`ef_tensor_cuda_map_mut`] exists on `t`.
 *
 * # Safety
 * `t` must be `NULL` or a live handle. `state.ctx` must remain valid until
 * this library's `release` call.
 */
int ef_tensor_cuda_attach(ef_tensor *t,
                          struct ef_client_state state,
                          void *resource,
                          uintptr_t size,
                          EfCudaMapFnNullable map_fn,
                          EfCudaUnmapFnNullable unmap_fn,
                          EfCudaUnregisterFnNullable unregister_fn);

/**
 * Whether `t` carries a CUDA registration (GL attach or D3D11 import).
 *
 * @retval 1 attached.
 * @retval 0 `t` is NULL or has no registration.
 *
 * # Safety
 * `t` must be `NULL` or a live handle.
 */
int ef_tensor_cuda_attached(const ef_tensor *t);

/**
 * Platforms: Windows.
 *
 * The process `ID3D11Device*`, created on first call. Borrowed: no
 * reference is transferred. A caller that keeps the pointer past this
 * library's lifetime must `AddRef` it, and must not `Release` the pointer
 * returned here.
 *
 * Every texture tensor this library allocates lives on this device.
 *
 * @retval `NULL` with `errno` set: `EIO` when no device could be created
 *         (no adapter, or `D3D11CreateDevice` failed --
 *         `ef_tensor_last_error_message` carries the reason), `ENOTSUP`
 *         off Windows.
 */
void *ef_d3d11_device(void);

/**
 * Platforms: Windows.
 *
 * Adopt a host-owned `ID3D11Device*` as the process device. The reference
 * stays the caller's; this library takes its own when it first uses the
 * device.
 *
 * Must be called before anything that creates the device --
 * `ef_d3d11_device`, `ef_is_gpu_buffer_available`, or any texture
 * allocation. Once the device exists it cannot be replaced.
 *
 * @return 0 on success, `EINVAL` for `NULL` or a pointer that is not a live
 *         `ID3D11Device`, `EBUSY` when the device is already initialized,
 *         `ENOTSUP` off Windows.
 *
 * # Safety
 * `device` must be `NULL` or a live `ID3D11Device *`.
 */
int ef_d3d11_use_external_device(void *device);

/**
 * Platforms: Windows.
 *
 * The `ID3D11Texture2D*` backing this tensor. Borrowed: valid while the
 * tensor lives, and never `Release`d by the caller. To hand the texture to
 * another device or process use [`ef_tensor_d3d11_shared_handle`].
 *
 * @retval `NULL` with `errno` set: `EINVAL` for a `NULL` tensor, `ENOTSUP`
 *         for a tensor that is not a D3D11 texture and off Windows.
 *
 * # Safety
 * `t` must be `NULL` or a live handle.
 */
void *ef_tensor_d3d11_texture(const ef_tensor *t);

/**
 * Platforms: Windows.
 *
 * The texture geometry the HAL chose for this image: DXGI format, texel
 * dimensions, bytes per texel, and the matching GL internal format. The
 * dimensions are the texture's, which differ from the image's for
 * semi-planar and packed-YUV formats (see `ef_d3d11_layout`).
 *
 * `out` is written only on success.
 *
 * @return 0 on success, `EINVAL` for a `NULL` tensor or `out`, `ENOTSUP`
 *         for a tensor that is not a D3D11 texture and off Windows.
 *
 * # Safety
 * `t` must be `NULL` or a live handle; `out` must be writable for one
 * `ef_d3d11_layout`.
 */
int ef_tensor_d3d11_layout(const ef_tensor *t, struct ef_d3d11_layout *out);

/**
 * Platforms: Windows.
 *
 * An owned duplicate of the texture's NT shared handle; close it with
 * `CloseHandle`. It can be opened on a D3D12 device, on another D3D11
 * device, in CUDA, or duplicated into another process. Closing it does not
 * affect the tensor.
 *
 * @retval `NULL` with `errno` set: `EINVAL` for a `NULL` tensor, `ENOTSUP`
 *         for a tensor that is not a D3D11 texture and off Windows, or
 *         another errno translated from the backend's error.
 *
 * # Safety
 * `t` must be `NULL` or a live handle.
 */
void *ef_tensor_d3d11_shared_handle(const ef_tensor *t);

/**
 * Platforms: Windows.
 *
 * The fence a GPU consumer waits on before reading this texture, and the
 * timeline value to wait for.
 *
 * `*fence` and `*value` are set to `NULL`/0 first and stay that way on
 * every failure path. When no write has been recorded the call returns 0
 * with `*fence == NULL`.
 *
 * When a write is recorded, `*fence` is an owned duplicate of the shared
 * fence's NT handle; close it with `CloseHandle`. Open it with
 * `ID3D12Device::OpenSharedHandle`, `ID3D11Device5::OpenSharedFence`, or
 * `cudaImportExternalSemaphore`, and wait for `*value`.
 *
 * @return 0 on success, whether or not a write is recorded; `EINVAL` for a
 *         `NULL` tensor, `fence` or `value`; `ENOTSUP` for a tensor that is
 *         not a D3D11 texture and off Windows; or another errno translated
 *         from the backend's error.
 *
 * # Safety
 * `t` must be `NULL` or a live handle; `fence` and `value` must be writable.
 */
int ef_tensor_gpu_completion(const ef_tensor *t, void **fence, uint64_t *value);

/**
 * Platforms: Windows.
 *
 * The fence value of the newest GPU write recorded on this tensor, or 0 when
 * there is none: the `*value` [`ef_tensor_gpu_completion`] reports, without
 * the duplicated fence handle.
 *
 * @return the recorded value, or 0 with `errno` set: `EINVAL` for a `NULL`
 *         tensor, `ENOTSUP` for a tensor that is not a D3D11 texture and
 *         off Windows. A texture with no recorded write returns 0 and
 *         leaves `errno` unchanged.
 *
 * # Safety
 * `t` must be `NULL` or a live handle.
 */
uint64_t ef_tensor_gpu_write_value(const ef_tensor *t);

/**
 * Platforms: Windows.
 *
 * Record that the GPU work writing this texture completes at `value` of the
 * process device's shared fence -- the value a later
 * [`ef_tensor_gpu_completion`] reports. Monotonic: an older value never
 * displaces a newer one.
 *
 * May be called while other threads hold the same handle and read the
 * tensor; it needs no exclusive access.
 *
 * @return 0 on success, `EINVAL` for a `NULL` tensor, `ENOTSUP` for a
 *         tensor that is not a D3D11 texture and off Windows, or another
 *         errno translated from the backend's error.
 *
 * # Safety
 * `t` must be `NULL` or a live handle.
 */
int ef_tensor_set_gpu_write(ef_tensor *t, uint64_t value);

/**
 * Platforms: Windows.
 *
 * Wrap an existing `ID3D11Texture2D` as a tensor. The texture must live on
 * the process device ([`ef_d3d11_device`]) and match the layout the HAL
 * would have chosen for `format` at the texture's own dimensions.
 * Ownership stays with the caller: this takes its own reference, and the
 * tensor releases it when freed.
 *
 * `dims`/`ndim` describe the tensor's grid: the allocation shape --
 * `[height, width, channels]` packed, `[channels, height, width]` planar,
 * `[combined_height, width]` semi-planar -- or the addressing shape. The
 * width and height are read from the texture description; a `dims` that
 * matches neither shape is `EINVAL`. The returned tensor carries the
 * allocation shape.
 *
 * `format` is the wire descriptor (`"NV12"`, `"rgba8"`); `dtype` is an
 * `EF_DTYPE_*` code; `access` is an `EF_CPU_ACCESS_*` code and decides
 * whether a CPU staging texture is created alongside: `EF_CPU_ACCESS_NONE`
 * creates none. `name` may be `NULL`.
 *
 * A semi-planar texture (`"NV12"`, `"NV16"`, `"NV24"`) carries the image's
 * row stride as its width, and is accepted only when that width is even
 * and equal to the staging row pitch the driver gives a texture of that
 * width; anything else is `EINVAL` with both numbers in the message.
 *
 * @retval a new tensor the caller must free with `ef_tensor_free`.
 * @retval `NULL` with `errno` set: `EINVAL` for a `NULL`/unshaped argument,
 *         an unknown format/dtype/access code, or a `dims` the texture does
 *         not have; the errno the backend's own failure maps to when the
 *         import itself fails; `ENOTSUP` off Windows.
 *         `ef_tensor_last_error_message` carries the reason.
 *
 * # Safety
 * `texture` must be a live `ID3D11Texture2D *` created on the process
 * device; `dims` must point to `ndim` readable `uint64_t`; `format` and
 * `name` must be `NULL` or NUL-terminated.
 */
ef_tensor *ef_tensor_from_d3d11_texture(void *texture,
                                        uint32_t dtype,
                                        const uint64_t *dims,
                                        uint32_t ndim,
                                        const char *format,
                                        uint32_t access,
                                        const char *name);

/**
 * Platforms: Windows.
 *
 * Open a shared D3D11 texture by its NT handle on the process device.
 *
 * `handle` stays owned by the caller: this opens its own texture from it,
 * and the caller closes the handle with `CloseHandle`. When `fence` is
 * non-`NULL` it names a shared fence whose value `fence_value` is waited
 * for on the process device's immediate context; the fence handle is
 * duplicated and stays owned by the caller. Pass `NULL`/0 when there is
 * nothing to wait for.
 *
 * `dims`, `format`, `dtype`, `access` and `name` follow
 * [`ef_tensor_from_d3d11_texture`], geometry check included.
 *
 * @retval a new tensor the caller must free with `ef_tensor_free`.
 * @retval `NULL` with `errno` set, as [`ef_tensor_from_d3d11_texture`].
 *
 * # Safety
 * `handle` must be an NT shared handle of a D3D11 texture, valid in this
 * process, and `fence` `NULL` or a shared-fence NT handle; `dims` must
 * point to `ndim` readable `uint64_t`; `format` and `name` must be `NULL`
 * or NUL-terminated.
 */
ef_tensor *ef_tensor_from_d3d11_shared_handle(void *handle,
                                              uint32_t dtype,
                                              const uint64_t *dims,
                                              uint32_t ndim,
                                              const char *format,
                                              uint32_t access,
                                              void *fence,
                                              uint64_t fence_value,
                                              const char *name);

/**
 * Create a request for a `width`×`height` image.
 *
 * `format` is the wire descriptor (`"NV12"`, `"rgb8"`). `dtype` is an
 * `ef_dtype` code. CPU access defaults to `EF_CPU_ACCESS_READ_WRITE`.
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
 * Declare CPU access, by `ef_cpu_access` code (`EF_CPU_ACCESS_*`).
 *
 * # Safety
 * `d` must be `NULL` or a live request.
 */
int ef_tensor_image_desc_set_access(struct ef_tensor_image_desc *d, uint32_t access);

/**
 * Request compression, by `ef_compression_request` code:
 * `EF_COMPRESSION_REQUEST_NONE` or `EF_COMPRESSION_REQUEST_ANY` (any scheme
 * the platform offers). Any other code is `EINVAL`.
 *
 * `Any` allocates linear when the format is not eligible and counts the
 * fallback.
 *
 * # Safety
 * `d` must be `NULL` or a live request.
 */
int ef_tensor_image_desc_set_compression(struct ef_tensor_image_desc *d, uint32_t compression);

/**
 * Require physically contiguous memory: 0 = no, 1 = yes.
 *
 * A contiguous request is allocated from the Linux CMA DMA heap only:
 * `ef_tensor_image_desc_alloc` fails, naming the heap, when that heap
 * cannot serve it.
 *
 * # Safety
 * `d` must be `NULL` or a live request.
 */
int ef_tensor_image_desc_set_contiguous(struct ef_tensor_image_desc *d, int contiguous);

/**
 * Whether the request requires physically contiguous memory: 1 if it does,
 * 0 if not or `d` is `NULL`.
 *
 * # Safety
 * `d` must be `NULL` or a live request.
 */
int ef_tensor_image_desc_contiguous(const struct ef_tensor_image_desc *d);

/**
 * Read a request's fields into `out`.
 *
 * @return 0 on success, `EINVAL` for a `NULL` argument.
 *
 * # Safety
 * `d` must be `NULL` or a live request. `out` must be a valid pointer to a
 * writable `ef_image_desc_view`.
 */
int ef_tensor_image_desc_get(const struct ef_tensor_image_desc *d, struct ef_image_desc_view *out);

/**
 * Allocate an image tensor from a finished request. See
 * [`edgefirst_tensor::TensorDyn::image_desc`].
 *
 * @retval a new tensor the caller must free with `ef_tensor_free`, on
 *         success.
 * @retval `NULL` for a `NULL` request, or if the underlying allocation
 *         fails (an incompatible `CpuAccess`/compression combination, an
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
 * Borrowed pointer to `ndim` strides in bytes, valid while `t` lives.
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
 * Returns 0 for an invalid or NULL handle.
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
 * size.
 *
 * @retval `>= 0` the allocation's byte count.
 * @retval `-1` `t` is `NULL` or invalid.
 *
 * # Safety
 * `t` must be `NULL` or a live handle.
 */
int64_t ef_tensor_capacity_bytes(const ef_tensor *t);

/**
 * The recorded row stride in bytes, or `-1` when none is recorded
 * (tightly packed).
 *
 * `ef_tensor_plane_at`'s `stride` is the effective pitch: the recorded one
 * when there is one, else a pitch computed from the format and width.
 *
 * @retval `>= 0` the recorded row stride in bytes.
 * @retval `-1` `t` is `NULL`/invalid, or no row stride is recorded.
 *
 * # Safety
 * `t` must be `NULL` or a live handle.
 */
int64_t ef_tensor_row_stride(const ef_tensor *t);

/**
 * The vendor tile-compression scheme this tensor's allocation resolved to,
 * as an `ef_compression` code; `EF_COMPRESSION_NONE` (0) for a linear
 * layout. Non-zero only for an Android AHardwareBuffer allocation that
 * requested compression and received it.
 *
 * Returns `EF_COMPRESSION_NONE` (0) for a `NULL`/invalid handle.
 *
 * # Safety
 * `t` must be `NULL` or a live handle.
 */
uint32_t ef_tensor_compression(const ef_tensor *t);

/**
 * The CPU access declared for this tensor, as an `ef_cpu_access` code: set
 * at allocation, possibly narrowed by `ef_tensor_restrict_cpu_access`. A
 * map, copy or sync bracket whose direction it does not cover fails with
 * `EACCES`.
 *
 * Returns `EF_CPU_ACCESS_NONE` (0) for a `NULL`/invalid handle.
 *
 * # Safety
 * `t` must be `NULL` or a live handle.
 */
uint32_t ef_tensor_cpu_access(const ef_tensor *t);

/**
 * Narrow this tensor's declared CPU access to the directions both it and
 * `access` (an `ef_cpu_access` code) cover, on the chroma plane too.
 * Never widens: a code wider than the declaration leaves it unchanged and
 * is not an error.
 *
 * @return 0, or `EINVAL` for a `NULL`/invalid handle or an unknown code.
 *
 * # Safety
 * `t` must be `NULL` or a live handle with no concurrent `tensor-capi`
 * calls on it.
 */
int ef_tensor_restrict_cpu_access(ef_tensor *t, uint32_t access);

/**
 * Whether this tensor's memory is known to be physically contiguous, as an
 * `ef_contiguity` code: contiguous for DMA-BUF allocated from the CMA heap,
 * non-contiguous for the system heap, unknown for an imported fd and every
 * other kind of memory.
 *
 * Returns unknown (0) for a `NULL`/invalid handle.
 *
 * # Safety
 * `t` must be `NULL` or a live handle.
 */
uint32_t ef_tensor_contiguity(const ef_tensor *t);

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
 * Whether this tensor addresses part of a larger buffer: a `view`/`batch`
 * sub-region, or a tensor rebuilt from one's descriptor and marked with
 * `ef_tensor_mark_view`. A CPU write into such a tensor maps read-write.
 *
 * Returns 1 for a view, 0 for a whole tensor or a `NULL`/invalid handle.
 *
 * # Safety
 * `t` must be `NULL` or a live handle.
 */
int ef_tensor_is_view(const ef_tensor *t);

/**
 * Record that this tensor addresses part of a larger buffer, which
 * `ef_tensor_is_view` then reports. Never cleared.
 *
 * @return 0, or `EINVAL` for a `NULL`/invalid handle.
 *
 * # Safety
 * `t` must be `NULL` or a live handle with no concurrent `tensor-capi`
 * calls on it.
 */
int ef_tensor_mark_view(ef_tensor *t);

/**
 * Allocate a host-memory tensor.
 *
 * Returns `NULL` on failure.
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
 * Wrap a caller-owned host allocation as a tensor that aliases it, without
 * copying or owning it.
 *
 * `capacity` is the allocation's size in bytes, which is `>=` the tight
 * footprint `dims` implies. Pass 0 for the tight footprint.
 *
 * The returned tensor does not keep `ptr` alive: it is valid only while
 * the caller keeps that memory alive.
 *
 * @retval a new tensor the caller must free with `ef_tensor_free`.
 * @retval `NULL` for a `NULL` `ptr`/`dims`, `ndim == 0`, or an
 *         unrecognized `dtype` -- `ef_tensor_last_error_message` carries
 *         the reason, `ef_tensor_last_error_class` is
 *         `EF_ERROR_CLASS_INVALID_ARGUMENT`, and `errno` is `EINVAL`.
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
 * Platforms: macOS, iOS.
 *
 * Wrap a live IOSurface, named by its cross-process `IOSurfaceID`, as a
 * tensor.
 *
 * The symbol is exported on every platform and returns `NULL` off Apple
 * platforms.
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
 * function releases one. The tensor is destroyed when the last reference
 * is released. A handle the caller holds a reference to remains valid
 * after another reference's `ef_tensor_free` call returns.
 *
 * Accepts a handle from any EdgeFirst library.
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
 * Set or clear colorimetry metadata on a live handle. `packed` is
 * `Colorimetry::pack`'s wire form; 0 clears it.
 *
 * This function and [`ef_tensor_colorimetry`] may be called concurrently
 * on the same handle from any number of threads, with no external locking.
 *
 * # Safety
 * `t` must be `NULL` or a live handle from an EdgeFirst library.
 */
int ef_tensor_set_colorimetry(ef_tensor *t, uint32_t packed);

/**
 * Platforms: Android.
 *
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
 * Platforms: Android.
 *
 * Borrowed AHardwareBuffer pointer, or NULL / `ENOTSUP`.
 *
 * # Safety
 * `t` must be `NULL` or a live handle.
 */
void *ef_tensor_hardware_buffer_ptr(const ef_tensor *t);

/**
 * Platforms: Android.
 *
 * Physical AHardwareBuffer dimensions in texels.
 *
 * # Safety
 * `width` and `height` must be writable when non-NULL.
 */
int ef_tensor_hardware_buffer_physical_dims(const ef_tensor *t,
                                            uintptr_t *width,
                                            uintptr_t *height);

/**
 * Platforms: macOS, iOS.
 *
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
 * identity (zero-copy). See [`edgefirst_tensor::TensorDyn::view`].
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
 * identity (zero-copy). See [`edgefirst_tensor::TensorDyn::batch`].
 *
 * The result has `t`'s shape with the leading dimension dropped. A tensor
 * whose leading dimension is not a batch axis is refused.
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
 * Consumes both `luma` and `chroma` once the precondition checks pass:
 * from that point both handles are invalid whether the call succeeds or
 * fails. A failed precondition check leaves both handles valid.
 *
 * @retval a new tensor the caller must free with `ef_tensor_free`, on
 *         success (`luma`/`chroma` consumed).
 * @retval `NULL`, `luma`/`chroma` left valid and unconsumed -- a
 *         `NULL`/invalid `luma` or `chroma`, an unrecognized `format`, a
 *         `luma`/`chroma` element-type mismatch, or an outstanding
 *         `ef_tensor_retain`/`ef_tensor_map` on either handle.
 * @retval `NULL`, `luma`/`chroma` consumed -- every precondition above
 *         passed but `Tensor::from_planes` refused (it accepts only
 *         NV12/NV16, matching luma/chroma widths, and the format-specific
 *         height ratio).
 *
 * Every `NULL` case sets `ef_tensor_last_error_message` with the reason.
 *
 * # Safety
 * `luma` and `chroma` must each be `NULL` or a live handle; `format` must
 * be `NULL` or a NUL-terminated string.
 */
ef_tensor *ef_tensor_from_planes(ef_tensor *luma, ef_tensor *chroma, const char *format);

/**
 * Whether `t` was assembled from separate luma/chroma allocations.
 *
 * @retval 1 `t` is a two-allocation NV12/NV16 tensor (`Tensor::from_planes`).
 * @retval 0 `t` is `NULL`, invalid, or a single contiguous buffer.
 *
 * # Safety
 * `t` must be `NULL` or a live handle.
 */
int ef_tensor_is_multiplane(const ef_tensor *t);

/**
 * Retained handle to the chroma plane of a multiplane tensor.
 *
 * The result is an independent handle the caller must free with
 * `ef_tensor_free`. Geometry mutations (`ef_tensor_set_row_stride_unchecked`,
 * `ef_tensor_set_plane_offset`) apply to this handle. Stride/offset that
 * must live on the combined tensor are set on the chroma input before
 * `ef_tensor_from_planes`.
 *
 * @retval a new tensor on success.
 * @retval `NULL` if `t` is `NULL`, not multiplane, or the chroma plane
 *         cannot be cloned into its own handle.
 *
 * # Safety
 * `t` must be `NULL` or a live handle.
 */
ef_tensor *ef_tensor_chroma(const ef_tensor *t);

/**
 * Advisory detail for the calling thread's last failing `tensor-capi` call,
 * `""` if none has failed yet.
 *
 * The returned pointer is valid only until this thread's next failing
 * `tensor-capi` call (or the thread's exit); a caller that wants to keep
 * the text must copy it out before calling in again. The string is a log
 * line and is not meant to be parsed.
 *
 * # Safety
 * The returned pointer must not be read after this thread makes another
 * `tensor-capi` call, or after this thread exits.
 */
const char *ef_tensor_last_error_message(void);

/**
 * Which kind of failure the calling thread's last failing `ef_tensor_*`
 * call was; `EF_ERROR_CLASS_UNSPECIFIED` (0) if none has failed yet, or if
 * that failure recorded no class.
 *
 * The companion to [`ef_tensor_last_error_message`]. Unlike that string,
 * this value may be programmed against. It serves the entry points that
 * report failure by returning `NULL`.
 *
 * Thread-local, set by every failing call, unchanged by a successful one.
 * Read it immediately after the call that returned `NULL`, before making
 * another.
 *
 * # Safety
 * Safe to call at any time.
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
 * A read-only map (`EF_CPU_ACCESS_READ`) populates `out->ptr`; the caller
 * must not write through it.
 *
 * A writable map (`_WRITE`/`_READ_WRITE`) is refused with `EBUSY` unless
 * the tensor's CPU-side handle count (`ef_tensor_retain`/`ef_tensor_free`)
 * is exactly one. `ef_tensor_last_error_message` distinguishes this from
 * the double-map `EBUSY`. The count covers CPU-side handles only; it does
 * not order access against a device writing the same buffer.
 *
 * @return 0 on success, `EINVAL` (null tensor/out, or bad access code),
 *         `EBUSY` (a map is already outstanding on this tensor, or a
 *         writable map was requested while the CPU-side handle count is
 *         greater than one), `EACCES` (the requested direction is not
 *         covered by the tensor's declared CPU access, see
 *         `ef_tensor_cpu_access`, or the backend refused the mapping), or
 *         another errno translated from the backend's error.
 *
 * # Safety
 * `t` must be `NULL` or a live handle; `out` must be writable for one
 * `ef_tensor_view`.
 */
int ef_tensor_map(ef_tensor *t, uint32_t access, struct ef_tensor_view *out);

/**
 * Non-blocking [`ef_tensor_map`]: identical except that a backing whose
 * map has to wait for a GPU copy returns `EAGAIN` instead of waiting.
 *
 * Only the Windows D3D11 texture has such a copy. Every other backing
 * behaves as `ef_tensor_map` and never returns `EAGAIN`.
 *
 * On `EAGAIN` no map is left outstanding and `out` is untouched. Yield or
 * sleep between attempts.
 *
 * @return the codes [`ef_tensor_map`] returns, plus `EAGAIN` when the map
 *         would have had to wait for a GPU copy.
 *
 * # Safety
 * `t` must be `NULL` or a live handle; `out` must be writable for one
 * `ef_tensor_view`.
 */
int ef_tensor_try_map(ef_tensor *t, uint32_t access, struct ef_tensor_view *out);

/**
 * Release the outstanding map taken by `ef_tensor_map`.
 *
 * Runs the platform sync bracket (for example the IOSurface unlock or the
 * dma-buf sync-for-device). The pointer returned by the matching
 * `ef_tensor_map` is invalid once this returns 0.
 *
 * @return 0 on success, `EINVAL` (null tensor, or no map is outstanding).
 *
 * # Safety
 * `t` must be `NULL` or a live handle.
 */
int ef_tensor_unmap(ef_tensor *t);

/**
 * Acquire the buffer for CPU access: the cache-maintenance bracket,
 * without a mapping.
 *
 * `DMA_BUF_IOCTL_SYNC` with `DMA_BUF_SYNC_START` on Linux; the IOSurface
 * lock on Apple platforms; a no-op for coherent host memory. Pairs with
 * [`ef_tensor_sync_for_device`], which must be called with the same
 * `access`.
 *
 * Unlike `ef_tensor_map`, this takes no map state and returns no address.
 *
 * `EF_CPU_ACCESS_NONE` is not a sync direction and is refused with
 * `EINVAL`.
 *
 * @return 0 on success, `EINVAL` (null tensor, or a bad/`NONE` access
 *         code), `EACCES` (the direction is not covered by the tensor's
 *         declared CPU access; see `ef_tensor_cpu_access`), `ENOTSUP` (a
 *         backing with no coherency window independent of its map -- PBO,
 *         and AHardwareBuffer on Android), or another errno translated
 *         from the backend's error.
 *
 * # Safety
 * `t` must be `NULL` or a live handle.
 */
int ef_tensor_sync_for_cpu(const ef_tensor *t, uint32_t access);

/**
 * Release the buffer back to the device after CPU access.
 *
 * `DMA_BUF_SYNC_END` on Linux. `access` must match the one passed to the
 * [`ef_tensor_sync_for_cpu`] that opened the bracket.
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
 * guard, copies, and drops the guard before returning.
 *
 * @return bytes written (`>= 0`) on success, or a negative errno: `-EINVAL`
 *         (null tensor/out), `-EACCES` (the tensor's declared CPU access
 *         has no read direction), `-ERANGE` (`cap` is smaller than the
 *         tensor's byte length), or another negative errno translated from
 *         the backend's error.
 *
 * # Safety
 * `t` must be `NULL` or a live handle; `out` must be writable for `cap`
 * bytes.
 */
int64_t ef_tensor_copy_to(ef_tensor *t, uint8_t *out, uintptr_t cap);

/**
 * Runs `f(ctx)` while no tensor CPU mapping can be created in this process.
 *
 * Every tensor mapping this library creates waits until `f` returns; `f`
 * must not map a tensor itself. `f` is called exactly once, on the calling
 * thread; a NULL `f` is a no-op.
 *
 * # Safety
 *
 * `f`, when non-NULL, must be safe to call with `ctx`.
 */
void ef_tensor_with_cpu_mappings_excluded(void (*f)(void *ctx), void *ctx);

/**
 * Attach pixel format metadata to a live handle, validating that its shape
 * is compatible with the format's layout. See
 * [`edgefirst_tensor::TensorDyn::set_format`].
 *
 * @retval 0 success; the format is attached and `ef_tensor_format` /
 *         `ef_tensor_plane_count` / `ef_tensor_plane_at` reflect it.
 * @retval EINVAL `t` is `NULL`, `format` is `NULL`/non-UTF8/unrecognized, or
 *         the tensor's current shape does not match the format's layout
 *         (packed expects `[H, W, C]`, planar `[C, H, W]`, semi-planar
 *         `[H*k, W]`).
 *
 * @warning Not safe to call concurrently with any other `tensor-capi` call
 * on the same handle from another thread.
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
 * on the same handle from another thread.
 *
 * # Safety
 * `t` must be `NULL` or a live handle.
 */
int ef_tensor_set_row_stride(ef_tensor *t, uintptr_t stride);

/**
 * Set the row stride in bytes without format validation. See
 * [`edgefirst_tensor::TensorDyn::set_row_stride_unchecked`].
 *
 * Unlike [`ef_tensor_set_row_stride`], this requires no pixel format and
 * does not validate `stride` against one. The caller is responsible for
 * the stride being valid.
 *
 * For a tensor with no format, `ef_tensor_plane_at` reports this stride as
 * plane 0's stride.
 *
 * @retval 0 success.
 * @retval EINVAL `t` is `NULL`.
 *
 * @warning Not safe to call concurrently with any other `tensor-capi` call
 * on the same handle from another thread.
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
 * @retval 0 success.
 * @retval EINVAL `t` is `NULL`.
 *
 * @warning Not safe to call concurrently with any other `tensor-capi` call
 * on the same handle from another thread.
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
 * [`ef_tensor_set_logical_shape`] for the capacity-based variant.
 *
 * @retval 0 success; `ef_tensor_shape`/`ef_tensor_strides` reflect the new
 *         geometry.
 * @retval EINVAL `t` or `dims` is `NULL`, `ndim` is 0, or a dimension is
 *         out of range for this host.
 * @retval ERANGE the new shape's element count differs from the current
 *         one -- the tensor is left untouched.
 *
 * @warning Not safe to call concurrently with any other `tensor-capi` call
 * on the same handle from another thread.
 *
 * # Safety
 * `t` must be `NULL` or a live handle; `dims` must point to `ndim`
 * readable `uint64_t`.
 */
int ef_tensor_reshape(ef_tensor *t, const uint64_t *dims, uint32_t ndim);

/**
 * Change a tensor's logical shape to anything its allocation can hold.
 *
 * The capacity-based counterpart to [`ef_tensor_reshape`]: the new shape's
 * element count may differ from the current one.
 *
 * @retval 0 success.
 * @retval EINVAL `t` or `dims` is `NULL`, `ndim` is 0, or a dimension is
 *         out of range for this host.
 * @retval ERANGE the new shape does not fit the existing allocation --
 *         the tensor is left untouched.
 *
 * @warning Not safe to call concurrently with any other `tensor-capi` call
 * on the same handle from another thread.
 *
 * # Safety
 * `t` must be `NULL` or a live handle; `dims` must point to `ndim`
 * readable `uint64_t`.
 */
int ef_tensor_set_logical_shape(ef_tensor *t, const uint64_t *dims, uint32_t ndim);

/**
 * Platforms: Linux, macOS, iOS, Android.
 *
 * Duplicate the file descriptor backing this tensor, for any storage kind
 * that has one: a dma-buf fd on Linux, a shared-memory fd wherever `Shm`
 * allocates.
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
 * consumer reads them. A dtype of a different width is refused.
 *
 * @retval 0 success; `ef_tensor_dtype` reports `dtype`.
 * @retval EINVAL `t` is `NULL`, or `dtype` is not a recognized code.
 * @retval ERANGE `dtype` has a different element width than the current one
 *         -- the tensor is left untouched.
 *
 * @warning Not safe to call concurrently with any other `tensor-capi` call
 * on the same handle from another thread.
 *
 * # Safety
 * `t` must be `NULL` or a live handle.
 */
int ef_tensor_set_dtype(ef_tensor *t, uint32_t dtype);

/**
 * Reconfigure a live handle's logical dimensions and pixel format, reusing
 * its existing allocation. See
 * [`edgefirst_tensor::TensorDyn::configure_image`].
 *
 * @retval 0 success; `ef_tensor_shape`/`ef_tensor_strides`/`ef_tensor_format`
 *         /`ef_tensor_plane_at` reflect the new geometry.
 * @retval EINVAL `t` is `NULL`, `format` is `NULL`/non-UTF8/unrecognized, or
 *         `width`x`height` is not a valid size for `format`.
 * @retval ERANGE the existing allocation cannot hold `width`x`height` in
 *         `format`.
 *
 * @warning Not safe to call concurrently with any other `tensor-capi` call
 * on the same handle from another thread.
 *
 * # Safety
 * `t` must be `NULL` or a live handle; `format` must be `NULL` or a
 * NUL-terminated string.
 */
int ef_tensor_configure_image(ef_tensor *t, uintptr_t width, uintptr_t height, const char *format);

/**
 * Wrap a client-owned OpenGL Pixel Buffer Object as a tensor.
 *
 * `state` is the callback channel this library keeps for the tensor's
 * life: it calls `state.retain` before returning and `state.release` once,
 * when the last reference to the returned tensor is freed. The caller
 * keeps its own reference and may drop it as soon as this returns.
 *
 * `retain`/`release` govern the channel only. They do not transfer
 * ownership of the GL buffer: this library never deletes it, and the
 * caller remains the sole caller of `glDeleteBuffers`.
 *
 * `size` is the GL allocation's full byte count, which may exceed the
 * product of `dims`.
 *
 * `map_fn` and `unmap_fn` are called on the caller's own thread, never on
 * a thread this library creates. They must be safe to call concurrently
 * and must not unwind.
 *
 * `state.ctx`, `state.retain`, `state.release`, `map_fn` and `unmap_fn`
 * must all be non-NULL; a NULL one is rejected before anything is
 * constructed.
 *
 * @retval a new tensor the caller must free with `ef_tensor_free`.
 * @retval `NULL` for a NULL `state.ctx`/`state.retain`/`state.release`/
 *         `map_fn`/`unmap_fn`, a NULL `dims`, `ndim == 0`, an
 *         unrecognized `dtype`, or a `size` smaller than the shape needs
 *         — `ef_tensor_last_error_message` carries the reason and
 *         `ef_tensor_last_error_class` its class. `errno` is set to
 *         `EINVAL` for these argument-validation refusals.
 *
 * # Safety
 * `dims` must point to `ndim` readable `uint64_t`. `state.ctx` must remain
 * valid until this library's `release` call, and `map_fn`/`unmap_fn`, when
 * non-NULL, must remain callable for the same span.
 */
ef_tensor *ef_tensor_wrap_pbo(struct ef_client_state state,
                              uint32_t buffer_id,
                              uintptr_t size,
                              uint32_t dtype,
                              const uint64_t *dims,
                              uint32_t ndim,
                              ef_pbo_map_fn map_fn,
                              ef_pbo_unmap_fn unmap_fn);

/**
 * The GL buffer name behind a PBO-backed tensor.
 *
 * @retval 0 success; `*out_id` holds the buffer name.
 * @retval EINVAL `t` is NULL/unresolvable, `out_id` is NULL, or `t` is not
 *         PBO-backed. Use `ef_tensor_storage_kind` to test whether a tensor
 *         is PBO-backed.
 *
 * # Safety
 * `t` must be NULL or a live handle; `out_id` must be NULL or a writable
 * `uint32_t`.
 */
int ef_tensor_pbo_id(const ef_tensor *t, uint32_t *out_id);

/**
 * Whether a PBO-backed tensor holds, or is establishing, a CPU mapping.
 *
 * @retval 1 mapped, or a map/unmap is in flight.
 * @retval 0 fully unmapped.
 * @retval -1 `t` is NULL/unresolvable or not PBO-backed.
 *
 * @warning Three-valued: the error value `-1` is truthy. Test `< 0` before
 * testing for non-zero.
 *
 * # Safety
 * `t` must be NULL or a live handle.
 */
int ef_tensor_pbo_is_mapped(const ef_tensor *t);

/**
 * The address of this PBO's callback vtable, as carried in the
 * cross-package descriptor's `ptr` field for the `PBO` kind.
 *
 * Borrowed: valid for as long as `t` is. A consumer that reconstructs
 * operations from it takes its own reference through the `ef_client_state`
 * embedded in it.
 *
 * @retval the vtable address.
 * @retval `NULL` when `t` is NULL/unresolvable or not PBO-backed.
 *
 * # Safety
 * `t` must be NULL or a live handle.
 */
const void *ef_tensor_pbo_vtable(const ef_tensor *t);

/**
 * Whether CUDA interop symbols resolved. Declared and answered on every
 * platform; `1` only where a CUDA runtime loaded.
 */
int ef_is_cuda_available(void);

/**
 * Whether Linux DMA-BUF allocation is available. Declared and answered on
 * every platform; `1` only on Linux. `ef_is_gpu_buffer_available` is the
 * portable question.
 */
int ef_is_dma_available(void);

/**
 * Whether a platform GPU-coherent buffer kind can be allocated: a DMA-BUF on
 * Linux, an IOSurface on macOS and iOS, an AHardwareBuffer on Android, a
 * D3D11 texture on Windows. Declared and answered on every platform.
 */
int ef_is_gpu_buffer_available(void);

/**
 * Whether IOSurface allocation is available. Declared and answered on every
 * platform; `1` only on macOS and iOS. `ef_is_gpu_buffer_available` is the
 * portable question.
 */
int ef_is_iosurface_available(void);

/**
 * Whether POSIX shared memory allocation is available. Declared and answered
 * on every platform; `1` only where `/dev/shm` is writable, so not on
 * Windows.
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
 * Number of `EF_COMPRESSION_REQUEST_ANY` requests that fell back to a
 * linear layout.
 */
uint64_t ef_compression_fallback_count(void);

/**
 * Report whether a live handle carries quantization metadata, and how many
 * `scale`/`zero_point` entries it has.
 *
 * Call this before [`ef_tensor_quantization_get`] to size its buffers.
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
 * zero-points. `n` must equal the `count` [`ef_tensor_quantization_info`]
 * reported.
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
 * A float tensor is refused.
 *
 * @retval 0 success.
 * @retval EINVAL `t` or `scales` is `NULL`, `n == 0`, `axis` is `< -1`,
 *         `axis == -1` with `n != 1`, or the backend rejects the resulting
 *         `Quantization` (dtype not integer, or `axis`/`n` incompatible
 *         with the tensor's shape).
 *
 * @warning Not safe to call concurrently with any other `tensor-capi` call
 * on the same handle from another thread.
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
 * on the same handle from another thread.
 *
 * # Safety
 * `t` must be `NULL` or a live handle.
 */
int ef_tensor_quantization_clear(ef_tensor *t);

/**
 * Serialize a tensor into a caller-provided blob buffer and handle table.
 *
 * Two-call pattern: pass `blob_cap`/`fds_cap` of 0 (or `NULL` buffers) to
 * learn the sizes required, then call again with buffers that large.
 * `*blob_len` and `*fds_len` are written even when the call fails with
 * `ENOSPC`.
 *
 * A `NULL` `fds_out` is a zero-capacity handle table: it is refused only
 * when the export produced handles. An inline export, and every export on
 * a platform that shares by handle value inside the blob (Windows),
 * produces none: it succeeds with `fds_out, fds_cap` of `NULL, 0` and
 * reports `*fds_len == 0`.
 *
 * The transport mode is chosen from the tensor's storage: a backing with a
 * shareable handle is exported by reference, and one without (`mem`, `pbo`)
 * is inlined.
 *
 * @return 0 on success; `ENOSPC` when a buffer is too small (with the
 *         required lengths written), including a `NULL` `fds_out` on an
 *         export that produced handles; `EINVAL` on a null tensor, a null
 *         `blob_len` or a null `fds_len`; `EIO` if the tensor cannot be
 *         serialized.
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
 * Import dups every handle it retains: the result is independent and the
 * sender may close its own copies as soon as this returns.
 *
 * Every length and index inside `blob` is validated against the buffer and
 * against `fds_len` before use.
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
