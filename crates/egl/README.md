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

## Features

`dynamic` pulls in `libloading` and enables the `Dynamic` instance — without it
the crate exposes the types but no way to load a driver, so enable it unless you
are supplying your own loader. The HAL's workspace dependency turns it on:

```toml
edgefirst-egl = { version = "0.28", features = ["dynamic"] }
```

The `1_0` through `1_5` features gate the EGL API version, cumulatively, and
default to `1_5`. Both this crate and its consumer must resolve to the same
`libloading` version — the `Borrow<Library>` bound on `Dynamic` will not hold
across a version mismatch.

## License

Licensed under either of Apache License, Version 2.0 or MIT license at your
option, matching the original `khronos-egl`. See `LICENSE-APACHE` and
`LICENSE-MIT`.
