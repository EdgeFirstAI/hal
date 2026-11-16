<!--
SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
SPDX-License-Identifier: Apache-2.0
-->

# edgefirst-decoder-capi — Architecture

C ABI for `edgefirst-decoder`. Ships as **`libedgefirst_decoder`** with `edgefirst/decoder.h`.

This library turns model-output tensors into `ef_detect_box` lists and segmentation masks. Box / mask / tile layouts are declared in header-only `edgefirst/detect.h`. It links `libedgefirst_tensor.so` dynamically. pkg-config: `Requires: edgefirst-tensor` and `Requires.private: edgefirst-decoder-abi`.

Exported symbols are `ef_*` only. Tiled merge helpers (`ef_tiled_frame_accumulator`, `ef_lift_tile_boxes`, `ef_merge_tiled_detections`) live here; tile *planning* that needs an `ImageProcessor` stays on `libedgefirst_image`.

**ABI versioning.** `ef_decoder_abi_version()` returns `4`. The rules that govern when it moves, when a by-value struct may gain a field in place, and what mixing versions across the five libraries is allowed to do are in [§ C ABI Stability and Versioning](https://github.com/EdgeFirstAI/hal/blob/main/ARCHITECTURE.md#c-abi-stability-and-versioning). Layouts declared in `detect.h`, including `ef_preprocess_spec`, are pinned by [`tests/c/test_layout_goldens.c`](tests/c/test_layout_goldens.c).

**Model configuration.** `ef_model_config_resolve` wraps `edgefirst_decoder::model_config`. Its preprocessing leaves as an `ef_preprocess_spec`, a plain value declared in header-only `detect.h` together with the `EF_FIT_*`, `EF_LETTERBOX_PLACEMENT_*`, `EF_PAD_SPACE_*` and `EF_INTERPOLATION_*` codes it carries, so `libedgefirst_image` consumes it without either library linking the other. Like every by-value struct here it has no `struct_size` handshake: it grows by major bump or successor type, as the root ARCHITECTURE.md sets out, and its layout is pinned by the decoder's layout goldens.

See [README.md](README.md) for usage and [tests/c/](tests/c/) for the link-and-run consumer.
