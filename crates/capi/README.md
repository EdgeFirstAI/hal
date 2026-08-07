# edgefirst-hal-capi

[![GitHub release](https://img.shields.io/github/v/release/EdgeFirstAI/hal?label=release)](https://github.com/EdgeFirstAI/hal/releases)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://github.com/EdgeFirstAI/hal/blob/main/LICENSE)

> **Distribution:** the C library is shipped as a GitHub Release tarball,
> not via crates.io. The `edgefirst-hal-capi` crate is marked
> `publish = false` because its useful artifact is the static/shared
> library plus the cbindgen-generated header — not the Rust source.

**EdgeFirst HAL C API** — C language bindings for the EdgeFirst Hardware
Abstraction Layer, providing zero-copy tensor operations,
hardware-accelerated image processing, ML model output decoding, and
multi-object tracking.

## Role in edgefirst-hal

`edgefirst-hal-capi` is the FFI bridge over the EdgeFirst HAL Rust workspace:

- Wraps [`edgefirst-tensor`](https://github.com/EdgeFirstAI/hal/blob/main/crates/tensor/), [`edgefirst-image`](https://github.com/EdgeFirstAI/hal/blob/main/crates/image/), [`edgefirst-decoder`](https://github.com/EdgeFirstAI/hal/blob/main/crates/decoder/), [`edgefirst-tracker`](https://github.com/EdgeFirstAI/hal/blob/main/crates/tracker/) as opaque-handle C APIs.
- Builds as both `staticlib` (`libedgefirst_hal.a`) and `cdylib` (`libedgefirst_hal.so` / `.dylib`). On mobile the two artifacts take different jobs: the `cdylib` is the Android JNI library, and the `staticlib` is what iOS embeds and what both mobile link-closure checks link against.
- Used by GStreamer plugins, OpenCV pipelines, NPU delegates, and any C/C++ consumer that needs the HAL outside Rust.
- Defines the [Delegate DMA-BUF Framework](https://github.com/EdgeFirstAI/hal/blob/main/crates/capi/ARCHITECTURE.md#delegate-dma-buf-framework) ABI that NXP Neutron, VxDelegate, and other TFLite delegates implement to expose their internal DMA-BUF tensors.

## Features

- **Tensor** - Create, reshape, map, and manage typed multi-dimensional tensors
  with DMA, shared memory, or system memory backing
- **Image** - Load, save, and convert images between pixel formats (RGB, RGBA,
  NV12, YUYV, etc.) with hardware acceleration when available (G2D, OpenGL, CPU)
- **Decoder** - Decode ML model outputs into detection boxes and segmentation
  masks (YOLO Ultralytics, ModelPack formats)
- **Tracker** - Multi-object tracking with ByteTrack (track-by-detection with
  UUID-based track identity)
- **Tiling** - SAHI-style input tiling plus the matching lift/merge/accumulate
  postprocessing for small-object detection on high-resolution frames

## Supported Platforms

Release tarballs are published for three targets:

| Platform | Architecture | Library Files |
|----------|-------------|---------------|
| Linux | x86_64 | `libedgefirst_hal.so`, `libedgefirst_hal.a` |
| Linux | aarch64 | `libedgefirst_hal.so`, `libedgefirst_hal.a` |
| macOS | Apple Silicon | `libedgefirst_hal.dylib`, `libedgefirst_hal.a` |

Android (`aarch64`/`x86_64`) and iOS (`aarch64`, plus the simulator) build and
link-check in CI, but no prebuilt archive ships for them — build those from
source with `scripts/build-android.sh` or `scripts/build-ios.sh`. Windows
(`x86_64-pc-windows-msvc`) only gets a `cargo check` in CI — it compiles, but
there is no release archive, no C test run, and no implementation behind the
DMA-BUF, EGL or IOSurface paths.

## Installation

Download the release archive for your platform from
[GitHub Releases](https://github.com/EdgeFirstAI/hal/releases) and extract it:

```sh
tar xzf edgefirst-hal-capi-<version>-<target>.tar.gz
```

The archive contains:

```
edgefirst-hal-capi-<version>-<target>/
  README.md
  LICENSE
  NOTICE
  include/
    edgefirst/
      hal.h
  lib/
    libedgefirst_hal.a                  # All platforms
    libedgefirst_hal.so                 # Linux: symlink -> .so.<major>
    libedgefirst_hal.so.<major>         # Linux: symlink -> .so.<major>.<minor>
    libedgefirst_hal.so.<major>.<minor> # Linux: symlink -> the real file
    libedgefirst_hal.so.<X>.<Y>.<Z>     # Linux: the real shared object
    libedgefirst_hal.dylib              # macOS
    pkgconfig/
      edgefirst-hal.pc
```

On Linux the shared library ships as the usual three-level symlink chain.
`DT_SONAME` inside the ELF is `libedgefirst_hal.so.<major>`, which matches the
second link in the chain, so the dynamic linker resolves it at runtime.

## Linking

### Linux (gcc/g++)

```sh
gcc -I/path/to/include -L/path/to/lib -o myapp myapp.c -ledgefirst_hal -lm -lpthread -ldl
```

With rpath for runtime library resolution:

```sh
gcc -I/path/to/include -L/path/to/lib -Wl,-rpath,/path/to/lib \
    -o myapp myapp.c -ledgefirst_hal -lm -lpthread -ldl
```

### macOS (gcc/clang)

```sh
gcc -I/path/to/include -L/path/to/lib -o myapp myapp.c -ledgefirst_hal -lm -lpthread
```

### Static linking

Replace `-ledgefirst_hal` with the full path to `libedgefirst_hal.a`:

```sh
gcc -I/path/to/include -o myapp myapp.c /path/to/lib/libedgefirst_hal.a -lm -lpthread -ldl
```

### pkg-config

The archive ships `lib/pkgconfig/edgefirst-hal.pc`. Its `prefix` is `/usr`, so
either install the archive under `/usr` or override the prefix on the command
line:

```sh
export PKG_CONFIG_PATH=/path/to/lib/pkgconfig
gcc -o myapp myapp.c \
    $(pkg-config --cflags --libs --define-variable=prefix=/path/to edgefirst-hal) \
    -lm -lpthread -ldl
```

## Quick Start

```c
#include <edgefirst/hal.h>
#include <errno.h>
#include <stdio.h>
#include <string.h>

int main(void) {
    // Create a 1x3x224x224 float32 tensor.
    size_t shape[] = {1, 3, 224, 224};
    struct hal_tensor *tensor = hal_tensor_new(
        HAL_DTYPE_F32, shape, 4, HAL_TENSOR_MEMORY_MEM, "input");
    if (!tensor) {
        fprintf(stderr, "hal_tensor_new: %s\n", strerror(errno));
        return 1;
    }

    // Map the tensor for CPU access.
    struct hal_tensor_map *map = hal_tensor_map_create(tensor);
    if (!map) {
        fprintf(stderr, "hal_tensor_map_create: %s\n", strerror(errno));
        hal_tensor_free(tensor);
        return 1;
    }

    memset(hal_tensor_map_data(map), 0, hal_tensor_map_size(map));

    hal_tensor_map_unmap(map);
    hal_tensor_free(tensor);
    return 0;
}
```

Build it against an extracted release archive:

```sh
gcc -I/path/to/include -L/path/to/lib -Wl,-rpath,/path/to/lib \
    -o quickstart quickstart.c -ledgefirst_hal -lm -lpthread -ldl
```

## Error Handling

All functions follow a consistent error convention:

- **Functions returning `int`**: `0` on success, `-1` on error with `errno` set
- **Functions returning pointers**: valid pointer on success, `NULL` on error
  with `errno` set
- **Functions returning `size_t`**: `0` if the handle is `NULL`

Check `errno` after any failure for the specific error code (e.g. `EINVAL`,
`ENOMEM`, `EIO`).

## Memory Management

- **`hal_*_new()` / `hal_*_load()`** - Caller owns the returned handle and must
  call the corresponding `hal_*_free()` to release it
- **`hal_*_get_*()`** - Returned pointers are borrowed references valid only
  during the parent object's lifetime; do not free them
- **Functions returning `char *`** (`hal_tensor_name()`,
  `hal_decoder_model_type()`) - Newly allocated C strings owned by the caller;
  release them with plain `free()`, not a `hal_*_free()`
- **`hal_*_clone_fd()`** - Creates a new owned file descriptor; caller must
  `close()` it
- **`hal_*_from_fd()`** - Duplicates the file descriptor internally; caller
  retains ownership and must `close()` it when done
- **`hal_tensor_map_create()` / `hal_tensor_map_unmap()`** - Map provides CPU
  access to tensor data; unmap when done to ensure cache coherency (especially
  for DMA tensors)
- All `_free()` functions accept `NULL` safely (no-op)

Three handles break the plain create/free symmetry, so they are worth calling
out:

| Handle | Deviation |
|--------|-----------|
| `struct hal_plane_descriptor *` | `hal_import_image()` consumes it on **both** the success and failure paths. Calling `hal_plane_descriptor_free()` afterwards is a double-free. |
| `struct hal_decoder_params *` | `hal_decoder_new()` takes it as `const` and clones the configuration. The caller still owns the params and must free them. |
| `struct hal_tiled_frame_accumulator *` | `hal_tiled_frame_accumulator_finalize()` and `_finalize_normalized()` consume the handle. Use `_free()` only to abandon an accumulator without finalizing. |

`struct HalImageDesc *` is deliberately *not* on that list: it is a reusable
builder that the create calls only read, so free it whenever you are done with
it, independently of the tensors it produced.

## Logging API

HAL logging is off by default. Initialise it once per process before any other
HAL calls:

```c
#include <edgefirst/hal.h>
#include <stdio.h>

// Option 2: forward each record to a callback of your own.
static void my_logger(hal_log_level level, const char *target,
                      const char *message, void *userdata) {
    (void)userdata;
    fprintf(stderr, "[%d] %s: %s\n", level, target, message);
}

int main(void) {
    // Option 1: write "[LEVEL] target: message" lines to a FILE*.
    hal_log_init_file(stderr, HAL_LOG_LEVEL_DEBUG);

    // Option 2 (pick one — the second call returns -1 with errno EALREADY).
    // hal_log_init_callback(my_logger, NULL, HAL_LOG_LEVEL_INFO);

    // ... the rest of your program ...
    return 0;
}
```

Only the first successful call takes effect; subsequent calls return `-1` with
`errno = EALREADY`. Available log levels: `HAL_LOG_LEVEL_ERROR`,
`HAL_LOG_LEVEL_WARN`, `HAL_LOG_LEVEL_INFO`, `HAL_LOG_LEVEL_DEBUG`,
`HAL_LOG_LEVEL_TRACE`.

## Zero-Copy Buffer Import

`hal_import_image()` wraps an externally-allocated DMA-BUF (e.g. from a V4L2
camera or video decoder) as a HAL tensor without copying:

```c
#include <edgefirst/hal.h>
#include <errno.h>
#include <stdio.h>
#include <string.h>

// Wrap one RGBA camera buffer and one NV12 decoder buffer, then release both.
// dmabuf_fd, y_fd and uv_fd come from V4L2 / the codec; the caller keeps them.
int import_example(int dmabuf_fd, size_t bytesperline, int y_fd, int uv_fd) {
    struct hal_image_processor *proc = hal_image_processor_new();
    if (!proc) {
        fprintf(stderr, "hal_image_processor_new: %s\n", strerror(errno));
        return -1;
    }

    // Single-plane (e.g. RGBA from a camera).
    struct hal_plane_descriptor *pd = hal_plane_descriptor_new(dmabuf_fd);
    hal_plane_descriptor_set_stride(pd, bytesperline);  // only for padded rows
    struct hal_tensor *src = hal_import_image(
        proc, pd, NULL, 1920, 1080, HAL_PIXEL_FORMAT_RGBA, HAL_DTYPE_U8,
        NULL);  // NULL colorimetry = unspecified, HAL picks the default
    // pd is consumed by hal_import_image — do NOT hal_plane_descriptor_free() it

    // Multi-plane NV12 (Y and UV planes on separate fds).
    struct hal_plane_descriptor *y_pd  = hal_plane_descriptor_new(y_fd);
    struct hal_plane_descriptor *uv_pd = hal_plane_descriptor_new(uv_fd);
    struct hal_tensor *nv12 = hal_import_image(
        proc, y_pd, uv_pd, 1920, 1080, HAL_PIXEL_FORMAT_NV12, HAL_DTYPE_U8,
        NULL);

    if (!src || !nv12) {
        fprintf(stderr, "hal_import_image: %s\n", strerror(errno));
        hal_tensor_free(src);   // NULL-safe
        hal_tensor_free(nv12);
        hal_image_processor_free(proc);
        return -1;
    }

    // Use src/nv12 with hal_image_processor_convert() or the draw_masks family.

    hal_tensor_free(nv12);
    hal_tensor_free(src);
    hal_image_processor_free(proc);
    return 0;
}
```

**Lifecycle rules:**
- `hal_plane_descriptor_new(fd)` dups the fd immediately; the caller keeps the
  original.
- `hal_plane_descriptor_set_stride(pd, stride)` and
  `hal_plane_descriptor_set_offset(pd, offset)` configure the plane before
  import.
- `hal_import_image()` **always consumes** both plane descriptors (even on
  error). Never call `hal_plane_descriptor_free()` on a descriptor passed to
  `hal_import_image()`.
- The trailing `colorimetry` argument is optional. Pass `NULL` to leave the
  four axes unspecified, or fill a `struct hal_colorimetry` when the source
  reports them — `hal_colorimetry_from_v4l2()` converts the four raw V4L2
  integers into the HAL constants.
- Linux only (`ENOTSUP` on other platforms).

## Allocating Images

Buffers you allocate yourself — intermediates, model inputs, render targets —
should come from `hal_image_processor_create_image()`, not `hal_tensor_new()`.
The processor picks the memory backend that its active GPU can consume
zero-copy: DMA-BUF for EGLImage-capable GPUs (Vivante, Mali), a PBO for desktop
GPUs, heap memory when there is no GPU. `hal_tensor_new()` takes a hardcoded
memory type and skips that negotiation, which can force a slow upload or
readback path.

Both `hal_image_processor_create_image()` and the standalone
`hal_tensor_new_image()` take a required `enum HalCpuAccess` as their last
argument. It declares what the **CPU** intends to do with the buffer; hardware
(GPU/NPU) access is always implied:

| Value | Declares | Use for |
|-------|----------|---------|
| `HAL_CPU_ACCESS_NONE` | No CPU mapping | Pure hardware pipelines — the cheapest option, and the only one eligible for vendor tile compression on Android |
| `HAL_CPU_ACCESS_READ` | CPU reads | Verification, CPU consumers |
| `HAL_CPU_ACCESS_WRITE` | CPU writes | Decode targets |
| `HAL_CPU_ACCESS_READ_WRITE` | Both | Reproduces the pre-declaration behaviour byte for byte |

```c
// Model input the GPU writes and the NPU reads — no CPU mapping needed.
struct hal_tensor *dst = hal_image_processor_create_image(
    proc, 640, 640, HAL_PIXEL_FORMAT_PLANAR_RGB, HAL_DTYPE_U8,
    HAL_CPU_ACCESS_NONE);
```

Mapping a buffer beyond its declaration is best effort: the platform may refuse
it or serve it slowly, and each occurrence warns once per buffer and increments
`hal_unplanned_cpu_access_count()`. Poll that counter in development to catch
declarations that no longer match how the buffer is used.

For anything beyond width, height, format, dtype and access, build a
`struct HalImageDesc` instead:

```c
struct HalImageDesc *desc = hal_image_desc_new(
    1920, 1080, HAL_PIXEL_FORMAT_RGBA, HAL_DTYPE_U8);
hal_image_desc_set_memory(desc, HAL_TENSOR_MEMORY_DMA);
hal_image_desc_set_access(desc, HAL_CPU_ACCESS_NONE);
hal_image_desc_set_compression(desc, HAL_COMPRESSION_ANY);

struct hal_tensor *t = hal_image_processor_create_image_desc(proc, desc);
hal_image_desc_free(desc);  // the desc is read, never consumed — reuse or free

// HAL_COMPRESSION_ANY is a request, not a guarantee. Read back what you got.
if (hal_tensor_compression(t) == HAL_COMPRESSION_NONE) {
    // Fell back to a linear layout (also counted by
    // hal_compression_fallback_count()).
}
```

Compression requires `HAL_CPU_ACCESS_NONE` — a CPU-mapped buffer pins a linear
layout by definition. `HAL_COMPRESSION_ANY` falls back to linear wherever the
device has no native scheme; naming a *specific* scheme the device cannot
provide fails the allocation with `ENOTSUP`. Probe first with
`hal_platform_compression_support(format, dtype)`.

## Object Tracking

ByteTrack multi-object tracking assigns stable UUID identities to detections
across frames:

```c
#include <edgefirst/hal.h>
#include <stdio.h>
#include <time.h>

static uint64_t monotonic_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ull + (uint64_t)ts.tv_nsec;
}

// `detections` comes from hal_decoder_decode(); the caller still owns it.
void track_frame(struct hal_bytetrack *tracker,
                 const struct hal_detect_box_list *detections) {
    struct hal_track_info_list *tracks =
        hal_bytetrack_update(tracker, detections, monotonic_ns());
    if (!tracks) {
        return;
    }

    size_t n = hal_track_info_list_len(tracks);
    for (size_t i = 0; i < n; i++) {
        struct hal_track_info info;
        if (hal_track_info_list_get(tracks, i, &info) != 0) {
            continue;
        }

        char uuid_str[37];
        hal_uuid_to_string(&info.uuid, uuid_str, sizeof(uuid_str));
        printf("track %s  box=[%.1f,%.1f,%.1f,%.1f]  count=%d\n",
               uuid_str,
               info.location[0], info.location[1],
               info.location[2], info.location[3],
               info.count);
    }
    hal_track_info_list_free(tracks);
}

int main(void) {
    // Custom parameters.
    struct hal_bytetrack *tracker = hal_bytetrack_new(
        0.25f,  // track_update: smoothness threshold
        0.70f,  // high_thresh:  high-confidence detection threshold
        0.25f,  // match_thresh: IOU matching threshold
        30,     // frame_rate:   expected fps
        60      // track_buffer: frames to hold lost tracks
    );
    // Or the library defaults (track_update=0.25, high_thresh=0.7,
    // match_thresh=0.25, track_extra_lifespan=500ms):
    //   struct hal_bytetrack *tracker = hal_bytetrack_new_default();
    if (!tracker) {
        return 1;
    }

    // ... call track_frame(tracker, detections) once per frame ...

    // Query the currently active tracks without feeding new detections.
    struct hal_track_info_list *active = hal_bytetrack_get_active_tracks(tracker);
    hal_track_info_list_free(active);

    hal_bytetrack_free(tracker);
    return 0;
}
```

`hal_track_info` fields:
| Field | Type | Description |
|-------|------|-------------|
| `uuid` | `uint8_t[16]` | 128-bit track identity (RFC 4122) |
| `location` | `float[4]` | Predicted box in XYXY format |
| `count` | `int32_t` | Number of times this track has been updated |
| `created` | `uint64_t` | Nanosecond timestamp when track was first created |
| `last_updated` | `uint64_t` | Nanosecond timestamp of last update |

## Mask Rendering

Draw segmentation masks (and detection boxes) directly onto an output image:

```c
#include <edgefirst/hal.h>

// Fused decode + render, no tracker.
int render(struct hal_image_processor *processor,
           const struct hal_decoder *decoder,
           const struct hal_tensor *const *outputs, size_t num_outputs,
           struct hal_tensor *dst) {
    struct hal_detect_box_list *boxes = NULL;

    int rc = hal_image_processor_draw_masks(
        processor,
        decoder,
        outputs,
        num_outputs,
        dst,                    // destination image tensor
        NULL,                   // background (NULL = clear dst first)
        0.6f,                   // opacity [0.0, 1.0]
        NULL,                   // letterbox [x0,y0,x1,y1] (NULL = none)
        HAL_COLOR_MODE_CLASS,   // colour per class
        &boxes);

    hal_detect_box_list_free(boxes);
    return rc;
}

// Fused decode + render + tracking.
int render_tracked(struct hal_image_processor *processor,
                   const struct hal_decoder *decoder,
                   struct hal_bytetrack *tracker, uint64_t timestamp_ns,
                   const struct hal_tensor *const *outputs, size_t num_outputs,
                   struct hal_tensor *dst) {
    struct hal_detect_box_list *boxes = NULL;
    struct hal_track_info_list *tracks = NULL;

    int rc = hal_image_processor_draw_masks_tracked(
        processor,
        decoder,
        tracker,
        timestamp_ns,
        outputs,
        num_outputs,
        dst,
        NULL,                   // background
        0.6f,                   // opacity
        NULL,                   // letterbox
        HAL_COLOR_MODE_TRACK,   // colour per track identity
        &boxes,
        &tracks);               // pass NULL when track output is not needed

    hal_detect_box_list_free(boxes);
    hal_track_info_list_free(tracks);  // NULL-safe
    return rc;
}
```

- `dst` is **always fully written**; its prior contents are discarded. To
  composite over an existing frame, pass that frame as `background` rather than
  pre-filling `dst`.
- `background`: optional source image composited under the masks. It must have
  the same dimensions and format as `dst`, and must not alias it. `NULL` clears
  `dst` to transparent black.
- `opacity`: clamped to `[0.0, 1.0]`; `1.0` = fully opaque masks.
- `letterbox`: four normalized coordinates `[x0, y0, x1, y1]` describing the
  content bounds on the model input, or `NULL` when the model saw the whole
  frame.
- `color_mode`: `HAL_COLOR_MODE_CLASS`, `_INSTANCE`, or `_TRACK`.
- `out_boxes` is always populated and must be freed; `out_tracks` in the tracked
  variant may be `NULL` if track output is not required.

## Tiling (SAHI)

Small objects in a 4K frame are only a handful of pixels once the whole frame is
squeezed into a 640×640 model input. SAHI-style tiling covers the frame with a
uniform overlapping grid, runs the same small model on each tile, then lifts the
per-tile detections back to full-frame coordinates and merges the duplicates
that land on tile seams.

The API splits along that seam. `hal_tile_grid()` and the
`hal_image_processor_*_tile*` calls handle the input side; `hal_lift_tile_boxes()`,
`hal_merge_tiled_detections()` and `hal_tiled_frame_accumulator_*` handle the
output side.

**Seed the config structs; never `memset(0)` them.** `struct hal_tiling_config`
and `struct hal_merge_config` are passed by value, and a zeroed one is not the
library default: `overlap_ratio` would be `0.0` instead of `0.2`, and the merge
would use IOU with `max_det = 0` instead of IOS with `max_det = 300`. Always
start from `hal_tiling_config_default()` / `hal_merge_config_default()` and
override individual fields.

```c
#include <edgefirst/hal.h>

// Your own helper: run the model on band `i` of the batched input and return
// that tile's decoded, tile-local normalized boxes.
struct hal_detect_box_list *decode_tile(const struct hal_decoder *decoder,
                                        struct hal_tensor *batch, size_t i);

// One frame, batched: render every tile into a single tall destination.
int tile_frame(struct hal_image_processor *proc,
               const struct hal_tensor *frame,
               const struct hal_decoder *decoder) {
    struct hal_tiling_config cfg = hal_tiling_config_default(640, 640);
    cfg.overlap_ratio = 0.2f;   // minimum overlap; realized overlap is >= this

    struct hal_tile_placement_list *plan =
        hal_image_processor_plan_tiles(proc, 3840, 2160, &cfg);
    if (!plan) {
        return -1;
    }
    size_t n_tiles = hal_tile_placement_list_len(plan);

    // A tall packed parent that stacks all n_tiles model inputs vertically.
    struct hal_tensor *batch = hal_image_processor_alloc_tile_batch(
        proc, n_tiles, &cfg, HAL_PIXEL_FORMAT_PLANAR_RGB, HAL_DTYPE_U8,
        HAL_TENSOR_MEMORY_DMA, HAL_CPU_ACCESS_NONE);

    // Render every tile and flush the GPU once.
    struct hal_tile_placement_list *placements =
        hal_image_processor_tile_into(proc, frame, batch, &cfg);

    // Collect each tile's detections as inference completes.
    struct hal_merge_config merge = hal_merge_config_default();  // IOS / 0.5 / 300
    struct hal_tiled_frame_accumulator *acc =
        hal_tiled_frame_accumulator_new(3840.0f, 2160.0f, n_tiles, &merge, 16);

    for (size_t i = 0; i < n_tiles; i++) {
        struct hal_tile_placement p;
        hal_tile_placement_list_get(placements, i, &p);

        // Run the model on tile i and decode it with a LOW score threshold
        // (see the note below), yielding tile-local normalized boxes.
        struct hal_detect_box_list *tile_boxes = decode_tile(decoder, batch, i);

        hal_tiled_frame_accumulator_push_tile(acc, tile_boxes, &p);
        hal_detect_box_list_free(tile_boxes);  // the accumulator copied them
    }

    // finalize CONSUMES acc — do not free it afterwards.
    struct hal_detect_box_list *merged =
        hal_tiled_frame_accumulator_finalize(acc);   // full-frame pixels
    // ...or _finalize_normalized(acc) for [0,1] boxes, e.g. to feed the tracker.

    hal_detect_box_list_free(merged);
    hal_tile_placement_list_free(placements);
    hal_tile_placement_list_free(plan);
    hal_tensor_free(batch);
    return 0;
}
```

The accumulator is the streaming path: push each tile as its inference lands,
and `hal_tiled_frame_accumulator_is_complete()` / `_remaining()` tell you when
the frame's fan-in is done. Pushes are idempotent per `placement.index`, so a
retried tile is not double-counted — the push returns `1` when the tile is newly
accepted and `0` when it is a duplicate or out of range.

If you do the work yourself instead, `hal_lift_tile_boxes()` converts one tile's
normalized boxes to full-frame pixels and `hal_merge_tiled_detections()` merges
the accumulated result. Both borrow their input list; the caller still owns and
frees it.

**Decode each tile with a low score threshold** (roughly 0.05) and class-aware
NMS. The merge uses GREEDYNMM with the intersection-over-smaller metric
precisely because an object split across a seam shows up as two partial boxes
with low IoU but high IoS. A high per-tile threshold discards those fragments
before the merge can rejoin them. Do the final score gating in
`hal_merge_config.score_threshold`, which defaults to `0.0` for that reason.

**Known limitation:** an object larger than one tile cannot be reconstructed.
Every tile sees only a fragment, and with no whole-object box to anchor the
union the fragments may not mutually clear the IoS threshold. Pick a tile size
larger than the biggest object you expect, or push one extra full-frame
downscaled pass into the accumulator as an additional tile.

## Tensor Extensions

### Attaching pixel format metadata

Tensors created from a raw DMA-BUF fd (via `hal_tensor_from_fd()`) carry no
image metadata. Use `hal_tensor_set_format()` to attach a pixel format so they
can be passed to image-processing functions:

```c
// Tensor shape [height, width, channels] — created from a DMA-BUF fd.
// hal_tensor_from_fd dups the fd; the caller still owns and closes its copy.
size_t shape[] = {1080, 1920, 3};
struct hal_tensor *t = hal_tensor_from_fd(HAL_DTYPE_U8, fd, shape, 3, "rgb");
if (!t || hal_tensor_set_format(t, HAL_PIXEL_FORMAT_RGB) != 0) {
    perror("tensor setup failed");
    hal_tensor_free(t);  // NULL-safe
    return -1;
}

// Now hal_tensor_width(), hal_tensor_height(), hal_tensor_row_stride() work.
size_t w      = hal_tensor_width(t);
size_t h      = hal_tensor_height(t);
size_t stride = hal_tensor_row_stride(t);  // explicit or computed
```

`hal_tensor_row_stride()` returns the stride set by
`hal_plane_descriptor_set_stride()` if one was recorded, otherwise computes
the minimum packed stride from the format, width, and element size.

### Cloning a DMA-BUF file descriptor

`hal_tensor_dmabuf_clone()` clones the DMA-BUF fd backing a tensor and returns
a clear `ENOTSUP` error for non-DMA-backed tensors (Mem, Shm), unlike
`hal_tensor_clone_fd()` which returns a generic I/O error in that case:

```c
int dmabuf_fd = hal_tensor_dmabuf_clone(tensor);
if (dmabuf_fd < 0) {
    if (errno == ENOTSUP)
        fprintf(stderr, "tensor is not DMA-backed\n");
    else
        perror("dmabuf clone failed");
} else {
    // use dmabuf_fd for zero-copy hardware import, then:
    close(dmabuf_fd);
}
```

Linux only (`ENOTSUP` on other platforms).

## CUDA Zero-Copy (TensorRT)

`hal_is_cuda_available()` queries whether the CUDA runtime (`libcudart`) is
loaded and all interop symbols resolved. The result is cached after the first
call — subsequent calls are cheap. Gate CUDA-specific paths on this before
calling `hal_tensor_cuda_map`.

```c
if (hal_is_cuda_available()) {
    // libcudart is present; zero-copy paths are usable.
}
```

### CUDA map lifecycle

| Step | Function | Notes |
|------|----------|-------|
| Obtain device ptr | `hal_tensor_cuda_map(tensor)` | Returns opaque handle or NULL |
| Read device address | `hal_tensor_cuda_device_ptr(handle, &size)` | Usable cross-thread via the CUDA primary context |
| Release handle | `hal_tensor_cuda_unmap(handle)` | Must be called before freeing the tensor |

**Ownership and lifetime rules:**
- Call `hal_tensor_cuda_unmap` before `hal_tensor_free` — unmapping after
  freeing is undefined behavior.
- Do not write to the tensor's host buffer while a CUDA map is live.
- The device pointer is valid until `hal_tensor_cuda_unmap` is called; do
  not cache it beyond that point.

### Fallback pattern

`hal_tensor_cuda_map` returns `NULL` for tensors without a registered CUDA
handle (all Mem and Shm tensors, and DMA tensors on systems without CUDA).
Always check for `NULL` and fall back to the host map:

```c
#include <edgefirst/hal.h>
#include <assert.h>

size_t shape[] = {1, 3, 640, 640};
struct hal_tensor* t = hal_tensor_new(
    HAL_DTYPE_F32, shape, 4, HAL_TENSOR_MEMORY_DMA, "input");

void* cm = hal_tensor_cuda_map(t);
if (cm) {
    // Zero-copy: feed the device pointer directly to TensorRT.
    size_t sz = 0;
    void* dptr = hal_tensor_cuda_device_ptr(cm, &sz);
    trt_context_set_input_tensor_address("input", dptr);
    trt_context_execute_async_v3(stream);
    // Always unmap before freeing the tensor.
    hal_tensor_cuda_unmap(cm);
} else {
    // Fallback: CPU-side host map (e.g. no libcudart, Mem tensor).
    struct hal_tensor_map* m = hal_tensor_map_create(t);
    assert(m != NULL);
    float* data = (float*)hal_tensor_map_data(m);
    assert(data != NULL);
    // ... fill data for CPU inference ...
    hal_tensor_map_unmap(m);
}
hal_tensor_free(t);
```

## Delegate DMA-BUF API

The delegate DMA-BUF framework defines the ABI contract that external NPU
delegates (e.g. NXP Neutron, VxDelegate) use to expose DMA-BUF tensor
information and camera format negotiation to the HAL.

These are **type definitions owned by the HAL**; the function implementations
live in the delegate shared libraries. Each delegate ships a
`hal_dmabuf.h` header that uses these types.

### `hal_dmabuf_tensor_info`

Describes a single delegate tensor's DMA-BUF allocation:

| Field | Type | Description |
|-------|------|-------------|
| `size` | `size_t` | Buffer size in bytes |
| `offset` | `size_t` | Byte offset within the DMA-BUF |
| `shape` | `size_t[8]` | Tensor dimensions (up to `HAL_DMABUF_MAX_NDIM = 8`) |
| `ndim` | `size_t` | Number of valid entries in `shape` |
| `fd` | `int` | DMA-BUF file descriptor — **borrowed, do not close** |
| `dtype` | `hal_dtype` | Element data type |

The struct must be zero-initialised with `memset(info, 0, info_size)` before
passing to the delegate's `hal_dmabuf_get_tensor_info()`. The `info_size`
parameter allows the struct to grow in future versions without breaking ABI.
Total size: 96 bytes on LP64.

### `hal_camera_adaptor_format_info`

Describes the channel mapping and V4L2 FourCC code for a camera format adaptor:

| Field | Type | Description |
|-------|------|-------------|
| `input_channels` | `int` | Number of input channels (e.g. 4 for RGBA) |
| `output_channels` | `int` | Number of output channels (e.g. 3 for RGB) |
| `fourcc` | `char[HAL_FOURCC_MAX_LEN]` | NUL-terminated V4L2 FourCC string (`HAL_FOURCC_MAX_LEN` is 8; the string is at most 4 bytes plus NUL) |

Used by consumers to negotiate upstream formats without requiring
vendor-specific symbols. Populated by the delegate's
`hal_camera_adaptor_get_format_info()`.

## API Reference

The full API is documented with Doxygen comments in
[`include/edgefirst/hal.h`](https://github.com/EdgeFirstAI/hal/blob/main/crates/capi/include/edgefirst/hal.h),
which `cbindgen` regenerates on every `cargo build -p edgefirst-hal-capi`. The
header is committed in-tree so C consumers can read the ABI without a Rust
toolchain; it is the authority on exact signatures, and this README is a guide
to using them.

## Documentation

- Architecture overview (incl. performance recommendations and Delegate DMA-BUF framework): [ARCHITECTURE.md](https://github.com/EdgeFirstAI/hal/blob/main/crates/capi/ARCHITECTURE.md)
- Testing guide: [TESTING.md](https://github.com/EdgeFirstAI/hal/blob/main/crates/capi/TESTING.md)
- Reference C benchmark / canonical example: [`bench_preproc.c`](https://github.com/EdgeFirstAI/hal/blob/main/crates/capi/tests/bench_preproc.c)
- Project README: [../../README.md](https://github.com/EdgeFirstAI/hal/blob/main/README.md)

## License

Apache-2.0 - see [LICENSE](https://github.com/EdgeFirstAI/hal/blob/main/LICENSE) for details.
