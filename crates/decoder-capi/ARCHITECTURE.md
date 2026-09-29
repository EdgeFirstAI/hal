<!--
SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
SPDX-License-Identifier: Apache-2.0
-->

# edgefirst-decoder-capi — Architecture

C ABI for `edgefirst-decoder`. Ships as **`libedgefirst_decoder`** with `edgefirst/decoder.h`.

This library turns model-output tensors into `ef_detect_box` lists and segmentation masks. Box / mask / tile layouts are declared in header-only `edgefirst/detect.h`. It links `libedgefirst_tensor.so` dynamically. pkg-config: `Requires: edgefirst-tensor` and `Requires.private: edgefirst-decoder-abi`.

Exported symbols are `ef_*` only. Tiled merge helpers (`ef_tiled_frame_accumulator`, `ef_lift_tile_boxes`, `ef_merge_tiled_detections`) live here; tile *planning* that needs an `ImageProcessor` stays on `libedgefirst_image`.

**ABI versioning.** `ef_decoder_abi_version()` returns `2`: it moved from `1` when `ef_merge_config` gained `mode` in its tail pad, a size-safe but not value-safe change that no link error or size mismatch reveals. The default tiled merge became keep-best in the same release; a changed default does not move the probe on its own. The rules that govern when it moves, when a by-value struct may gain a field in place (`ef_merge_config` gained `mode` in its tail pad), and what mixing versions across the five libraries is allowed to do are in [§ C ABI Stability and Versioning](https://github.com/EdgeFirstAI/hal/blob/main/ARCHITECTURE.md#c-abi-stability-and-versioning). Layouts declared in `detect.h` are pinned by [`tests/c/test_layout_goldens.c`](tests/c/test_layout_goldens.c).

See [README.md](README.md) for usage and [tests/c/](tests/c/) for the link-and-run consumer.
