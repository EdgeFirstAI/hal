<!--
SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
SPDX-License-Identifier: Apache-2.0
-->

# edgefirst-image-capi — Architecture

C ABI for `edgefirst-image`. Ships as **`libedgefirst_image`** with `edgefirst/image.h`.

This library mints tensors (`ef_image_processor_create_image`) and converts, resizes, letterboxes and draws. It links `libedgefirst_tensor.so` dynamically. Detection primitives used when drawing live in header-only `edgefirst/detect.h` (`Requires.private: edgefirst-decoder-abi`). pkg-config: `Requires: edgefirst-tensor`.

Exported symbols are `ef_*` only. Tensor accessors are not re-exported.

**ABI versioning.** `ef_image_abi_version()` returns `4`. It moved from `3` when `ef_tiling_config` lost `pad` and `fit` (24 bytes, was 32) and `ef_image_processor_plan_tiles`, `_tile_into` and `_tile_one` gained a trailing `const ef_convert_options *` that says how each tile is converted; `EF_FIT_*` and `EF_LETTERBOX_PLACEMENT_*` moved to `detect.h` with their values unchanged, `ef_image_processor_set_interpolation` began accepting `EF_INTERPOLATION_BICUBIC`/`_AREA` (run as bilinear), and `ef_convert_options_set_interpolation` added a per-convert interpolation. It moved from `2` to `3` when the convert functions began taking an opaque `ef_convert_options` handle in place of `rotation`, `flip` and `ef_crop`, which was removed. It moved from `1` when unknown `color_mode` codes (drawn as `EF_COLOR_MODE_CLASS` before) and unknown `ef_tiling_config.fit` codes (letterbox before) began to be refused: the same calls and layouts, read differently. The rules that govern when it moves, when a by-value struct may gain a field in place, and what mixing versions across the five libraries is allowed to do are in [§ C ABI Stability and Versioning](https://github.com/EdgeFirstAI/hal/blob/main/ARCHITECTURE.md#c-abi-stability-and-versioning). `image.h` declares two by-value structs (`ef_tiling_config`, `ef_tile_spec`) and, unlike `tensor` and `decoder`, has **no** `tests/c/test_layout_goldens.c` pinning them — a gap to close before 1.0.

`ef_convert_options_from_spec` consumes the `ef_preprocess_spec` the decoder library produces, interpolation included; the struct is described in [the decoder's ARCHITECTURE.md](../decoder-capi/ARCHITECTURE.md).

See [README.md](README.md) for usage and [tests/c/](tests/c/) for the link-and-run consumer.
