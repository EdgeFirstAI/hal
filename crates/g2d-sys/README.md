# g2d-sys

[![Crates.io](https://img.shields.io/crates/v/g2d-sys.svg)](https://crates.io/crates/g2d-sys)
[![Documentation](https://docs.rs/g2d-sys/badge.svg)](https://docs.rs/g2d-sys)
[![License](https://img.shields.io/crates/l/g2d-sys.svg)](LICENSE)

**Rust FFI bindings for NXP i.MX G2D 2D graphics accelerator.**

This crate provides low-level unsafe bindings to `libg2d.so` for hardware-accelerated 2D graphics operations on NXP i.MX8 platforms.

## Requirements

- NXP i.MX8 platform with G2D support
- `libg2d.so.2` installed (typically at `/usr/lib/libg2d.so.2`)

## Features

The G2D library provides hardware-accelerated:

- **Blitting** - Fast memory-to-memory copies with format conversion
- **Scaling** - High-quality image resize
- **Rotation** - 90/180/270 degree rotation
- **Color space conversion** - YUV to RGB
- **Alpha blending** - Porter-Duff compositing operations
- **Clear** - Fast rectangle fills

## Supported Formats

| Format | Description |
|--------|-------------|
| `G2D_RGBA8888` | 32-bit RGBA |
| `G2D_RGBX8888` | 32-bit RGBx (alpha ignored) |
| `G2D_RGB565` | 16-bit RGB |
| `G2D_NV12` | YUV 4:2:0 semi-planar |
| `G2D_NV16` | YUV 4:2:2 semi-planar |
| `G2D_YUYV` | YUV 4:2:2 packed |

## Usage

This is a `-sys` crate providing raw FFI bindings. For a safe Rust API, use [`edgefirst-image`](https://crates.io/crates/edgefirst-image) which wraps these bindings.

```rust
use g2d_sys::*;

unsafe {
    let handle = g2d_open();
    if handle.is_null() {
        panic!("Failed to open G2D");
    }

    // Configure surfaces and perform operations...

    g2d_close(handle);
}
```

## Library Loading

The bindings use dynamic loading via `libloading`. The library is loaded from:
1. `LIBG2D_PATH` environment variable (if set)
2. `/usr/lib/libg2d.so.2` (default)

## License

Licensed under the MIT License. See [LICENSE](LICENSE) for details.

The G2D API header (`g2d.h`) is provided by NXP under their license terms.
