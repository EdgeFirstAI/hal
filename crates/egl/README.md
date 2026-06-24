# edgefirst-egl

Rust bindings for the [Khronos EGL](https://www.khronos.org/egl/) API.

This crate is a maintained fork of [`khronos-egl`](https://github.com/timothee-haudebourg/khronos-egl)
6.0.0 by Timothée Haudebourg and Sean Kerr, which became unmaintained. It is
vendored into the EdgeFirst HAL so we can keep its dependencies current —
notably tracking `libloading` 0.9+ — and is published as `edgefirst-egl`.

Relative to upstream it tracks `libloading` 0.9 and drops the static-linking
path (the `static`/`no-pkg-config` features, the `pkg-config` build dependency,
and the `Static`/`API` types); only runtime (dynamic) loading is supported. The
dynamic-loading API is otherwise unchanged. The library is imported as
`edgefirst_egl`:

```rust
use edgefirst_egl as egl;
```

## License

Licensed under either of Apache License, Version 2.0 or MIT license at your
option, matching the original `khronos-egl`. See `LICENSE-APACHE` and
`LICENSE-MIT`.
