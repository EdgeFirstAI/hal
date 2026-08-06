# edgefirst-gl

Minimal OpenGL ES bindings used by the EdgeFirst HAL.

This crate is a **trimmed, maintained fork** of [`gls`](https://github.com/vaxpl/gls-rs)
0.1.6 by Varphone Wong, which became unmaintained. Only the surface the HAL
actually uses is retained:

- the raw `gl` bindings (generated at build time by `gl_generator`, GLES 3.2 +
  the `GL_OES_EGL_image` / `GL_OES_EGL_image_external` / `GL_EXT_YUV_target`
  extensions),
- a small set of safe wrappers (`apis`),
- the `Error` type.

The upstream higher-level helpers (textures, framebuffers, shaders, buffers,
viewport math) and their `nalgebra` / `serde` / `libc` / `winapi` dependencies
have been removed, so this crate has **no runtime dependencies**.

The library is imported as `edgefirst_gl`:

```rust
use edgefirst_gl as gls; // optional alias
```

## Loading the function pointers

`gl_generator`'s `GlobalGenerator` emits a process-global `static mut` function
table, so `gl::load_with` must run exactly once per process — not once per
context or per object. Calling it again from a second thread while GL calls are
in flight is a data race. The HAL wraps this in a `load_gl_once` step on its
platform seam; anything else embedding this crate needs the same discipline.

## License

Licensed under either of Apache License, Version 2.0 or MIT license at your
option, matching the original `gls`. See `LICENSE-APACHE` and `LICENSE-MIT`.
