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
- **`hal_*_from_fd()`** - Takes ownership of the file descriptor; caller must
  NOT `close()` it after the call
- **`hal_tensor_map_create()` / `hal_tensor_map_unmap()`** - Map provides CPU
  access to tensor data; unmap when done to ensure cache coherency (especially
  for DMA tensors)
- All `_free()` functions accept `NULL` safely (no-op)

## API Reference

The full API is documented with Doxygen comments in
[`include/edgefirst/hal.h`](include/edgefirst/hal.h).

## License

Apache-2.0 - see [LICENSE](../../LICENSE) for details.
