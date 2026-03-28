# EdgeFirst HAL C API

C language bindings for the EdgeFirst Hardware Abstraction Layer, providing
zero-copy tensor operations, hardware-accelerated image processing, ML model
output decoding, and multi-object tracking.

## Features

- **Tensor** - Create, reshape, map, and manage typed multi-dimensional tensors
  with DMA, shared memory, or system memory backing
- **Image** - Load, save, and convert images between pixel formats (RGB, RGBA,
  NV12, YUYV, etc.) with hardware acceleration when available (G2D, OpenGL, CPU)
- **Decoder** - Decode ML model outputs into detection boxes and segmentation
  masks (YOLO Ultralytics, ModelPack formats)
- **Tracker** - Multi-object tracking with ByteTrack (track-by-detection with
  UUID-based track identity)

## Supported Platforms

| Platform | Architecture | Library Files |
|----------|-------------|---------------|
| Linux | x86_64 | `libedgefirst_hal.so`, `libedgefirst_hal.a` |
| Linux | aarch64 | `libedgefirst_hal.so`, `libedgefirst_hal.a` |
| macOS | Apple Silicon | `libedgefirst_hal.dylib`, `libedgefirst_hal.a` |

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
    libedgefirst_hal.so    # Linux
    libedgefirst_hal.a     # All platforms
    libedgefirst_hal.dylib # macOS
```

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

## Quick Start

```c
#include <edgefirst/hal.h>
#include <stdio.h>
#include <string.h>

int main(void) {
    // Create a 1x3x224x224 float32 tensor
    size_t shape[] = {1, 3, 224, 224};
    struct hal_tensor *tensor = hal_tensor_new(
        HAL_DTYPE_F32, shape, 4, HAL_TENSOR_MEMORY_MEM, "input");
    if (!tensor) {
        fprintf(stderr, "Failed to create tensor: %s\n", strerror(errno));
        return 1;
    }

    // Map tensor for CPU access
    struct hal_tensor_map *map = hal_tensor_map_create(tensor);
    float *data = (float *)hal_tensor_map_data(map);

    // Fill with zeros
    memset(data, 0, hal_tensor_map_size(map));

    // Unmap and free
    hal_tensor_map_unmap(map);
    hal_tensor_free(tensor);
    return 0;
}
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
- **`hal_*_clone_fd()`** - Creates a new owned file descriptor; caller must
  `close()` it
- **`hal_*_from_fd()`** - Duplicates the file descriptor internally; caller
  retains ownership and must `close()` it when done
- **`hal_tensor_map_create()` / `hal_tensor_map_unmap()`** - Map provides CPU
  access to tensor data; unmap when done to ensure cache coherency (especially
  for DMA tensors)
- All `_free()` functions accept `NULL` safely (no-op)

## Logging API

HAL logging is off by default. Initialise it once per process before any other
HAL calls:

```c
#include <edgefirst/hal.h>
#include <stdio.h>

// Option 1: write [LEVEL] target: message lines to a FILE*
hal_log_init_file(stderr, HAL_LOG_LEVEL_DEBUG);

// Option 2: forward each record to a custom callback
void my_logger(hal_log_level level, const char *target,
               const char *message, void *userdata) {
    fprintf(stderr, "[%d] %s: %s\n", level, target, message);
}
hal_log_init_callback(my_logger, NULL, HAL_LOG_LEVEL_INFO);
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

// Create the image processor
struct hal_image_processor *proc = hal_image_processor_new();

// Single-plane (e.g. RGBA from a camera)
struct hal_plane_descriptor *pd = hal_plane_descriptor_new(dmabuf_fd);
hal_plane_descriptor_set_stride(pd, bytesperline); // optional, for padded rows
struct hal_tensor *src = hal_import_image(
    proc, pd, NULL, 1920, 1080, HAL_PIXEL_FORMAT_RGBA, HAL_DTYPE_U8);
// pd is consumed by hal_import_image — do NOT call hal_plane_descriptor_free()

// Multi-plane NV12 (Y + UV planes on separate fds)
struct hal_plane_descriptor *y_pd  = hal_plane_descriptor_new(y_fd);
struct hal_plane_descriptor *uv_pd = hal_plane_descriptor_new(uv_fd);
struct hal_tensor *nv12 = hal_import_image(
    proc, y_pd, uv_pd, 1920, 1080, HAL_PIXEL_FORMAT_NV12, HAL_DTYPE_U8);

// Use src with hal_image_processor_convert() or hal_image_processor_draw_masks()
hal_tensor_free(src);
hal_tensor_free(nv12);
hal_image_processor_free(proc);
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
- Linux only (`ENOTSUP` on other platforms).

## Object Tracking

ByteTrack multi-object tracking assigns stable UUID identities to detections
across frames:

```c
#include <edgefirst/hal.h>
#include <stdio.h>

// Create a tracker (custom parameters)
struct hal_bytetrack *tracker = hal_bytetrack_new(
    0.25f,  // track_update: smoothness threshold
    0.70f,  // high_thresh:  high-confidence detection threshold
    0.25f,  // match_thresh: IOU matching threshold
    30,     // frame_rate:   expected fps
    60      // track_buffer: frames to hold lost tracks
);
// Or use defaults (track_update=0.25, high_thresh=0.7, match_thresh=0.25,
//                  track_extra_lifespan=500ms)
// struct hal_bytetrack *tracker = hal_bytetrack_new_default();

// Each frame: update with decoded detections
uint64_t timestamp_ns = ...; // monotonic nanoseconds
struct hal_track_info_list *tracks =
    hal_bytetrack_update(tracker, detect_box_list, timestamp_ns);

size_t n = hal_track_info_list_len(tracks);
for (size_t i = 0; i < n; i++) {
    struct hal_track_info info;
    hal_track_info_list_get(tracks, i, &info);

    char uuid_str[37];
    hal_uuid_to_string(&info.uuid, uuid_str, sizeof(uuid_str));
    printf("track %s  box=[%.1f,%.1f,%.1f,%.1f]  count=%d\n",
           uuid_str,
           info.location[0], info.location[1],
           info.location[2], info.location[3],
           info.count);
}
hal_track_info_list_free(tracks);

// Query currently active tracks without updating
struct hal_track_info_list *active = hal_bytetrack_get_active_tracks(tracker);
hal_track_info_list_free(active);

hal_bytetrack_free(tracker);
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

struct hal_detect_box_list *boxes = NULL;

// Fused decode + render (no tracker)
int rc = hal_image_processor_draw_masks(
    processor,
    decoder,
    outputs,      // const struct hal_tensor **
    num_outputs,
    dst,          // destination image tensor
    NULL,         // background (NULL = draw over dst)
    0.6f,         // opacity [0.0, 1.0]
    &boxes
);

// Fused decode + render + tracking
struct hal_track_info_list *tracks = NULL;
rc = hal_image_processor_draw_masks_tracked(
    processor,
    decoder,
    tracker,
    timestamp_ns,
    outputs,
    num_outputs,
    dst,
    NULL,         // background (NULL = draw over dst)
    0.6f,         // opacity
    &boxes,
    &tracks       // may be NULL if track output not needed
);

hal_detect_box_list_free(boxes);
hal_track_info_list_free(tracks); // safe no-op if NULL
```

- `background`: optional source image to composite masks onto before writing to
  `dst`. Must not alias `dst`.
- `opacity`: clamped to `[0.0, 1.0]`; `1.0` = fully opaque masks.
- `out_boxes` is always populated; `out_tracks` in the tracked variant may be
  `NULL` if track output is not required.

## Tensor Extensions

### Attaching pixel format metadata

Tensors created from a raw DMA-BUF fd (via `hal_tensor_from_fd()`) carry no
image metadata. Use `hal_tensor_set_format()` to attach a pixel format so they
can be passed to image-processing functions:

```c
// Tensor shape [height, width, channels] — created from a DMA-BUF fd
size_t shape[] = {1080, 1920, 3};
struct hal_tensor *t = hal_tensor_from_fd(fd, HAL_DTYPE_U8, shape, 3, "rgb");
hal_tensor_set_format(t, HAL_PIXEL_FORMAT_RGB);

// Now hal_tensor_width(), hal_tensor_height(), hal_tensor_row_stride() work
size_t w      = hal_tensor_width(t);
size_t h      = hal_tensor_height(t);
size_t stride = hal_tensor_row_stride(t); // explicit or computed
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

## Delegate DMA-BUF API

The delegate DMA-BUF framework defines the ABI contract that external NPU
delegates (e.g. NXP Neutron, VxDelegate) use to expose DMA-BUF tensor
information and camera format negotiation to the HAL.

These are **type definitions owned by the HAL**; the function implementations
live in the delegate shared libraries. Each delegate ships a
`hal_dmabuf.h` header that uses these types.

### `HalDmabufTensorInfo`

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

### `HalCameraAdaptorFormatInfo`

Describes the channel mapping and V4L2 FourCC code for a camera format adaptor:

| Field | Type | Description |
|-------|------|-------------|
| `input_channels` | `int` | Number of input channels (e.g. 4 for RGBA) |
| `output_channels` | `int` | Number of output channels (e.g. 3 for RGB) |
| `fourcc` | `char[8]` | NUL-terminated V4L2 FourCC string |

Used by consumers to negotiate upstream formats without requiring
vendor-specific symbols. Populated by the delegate's
`hal_camera_adaptor_get_format_info()`.

## API Reference

The full API is documented with Doxygen comments in
[`include/edgefirst/hal.h`](include/edgefirst/hal.h).

## License

Apache-2.0 - see [LICENSE](../../LICENSE) for details.
