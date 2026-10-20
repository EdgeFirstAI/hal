# edgefirst-build-support

Build-script helpers shared by the HAL's shared-library leaves: the C libraries and the Python extensions. Not published; a `[build-dependencies]` entry only.

The tensor-linked leaves are built with edgefirst-tensor's dynamic backend (their crate's `.cargo/config.toml`, or `--config .cargo/tensor-dynamic.toml` from the root) and call into `libedgefirst_tensor` instead of embedding the tensor implementation. This crate holds the build-script half of that arrangement (backend guard, producer build, linking, rpath, SONAME) in one place; the module docs in `src/lib.rs` list each helper.

See [ARCHITECTURE.md § Tensor Backend Selection](https://github.com/EdgeFirstAI/hal/blob/main/ARCHITECTURE.md#tensor-backend-selection).
