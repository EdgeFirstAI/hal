<!--
SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
SPDX-License-Identifier: Apache-2.0
-->

# edgefirst-image-capi — Architecture

C ABI for `edgefirst-image`. Ships as **`libedgefirst_image`** with `edgefirst/image.h`.

This library mints tensors (`ef_image_processor_create_image`) and converts, resizes, letterboxes and draws. It links `libedgefirst_tensor.so` dynamically. Detection primitives used when drawing live in header-only `edgefirst/detect.h` (`Requires.private: edgefirst-decoder-abi`). pkg-config: `Requires: edgefirst-tensor`.

Exported symbols are `ef_*` only. Tensor accessors are not re-exported.

**ABI versioning.** `ef_image_abi_version()` returns `1`; the rules that govern when it moves, when a by-value struct may gain a field in place, and what mixing versions across the five libraries is allowed to do are in [§ C ABI Stability and Versioning](https://github.com/EdgeFirstAI/hal/blob/main/ARCHITECTURE.md#c-abi-stability-and-versioning). `image.h` declares three by-value structs (`ef_crop`, `ef_tiling_config`, `ef_tile_spec`) and, unlike `tensor` and `decoder`, has **no** `tests/c/test_layout_goldens.c` pinning them — a gap to close before 1.0.

See [README.md](README.md) for usage and [tests/c/](tests/c/) for the link-and-run consumer.
