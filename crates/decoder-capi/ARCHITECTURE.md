<!--
SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
SPDX-License-Identifier: Apache-2.0
-->

# edgefirst-decoder-capi — Architecture

C ABI for `edgefirst-decoder`. Ships as **`libedgefirst_decoder`** with `edgefirst/decoder.h`.

This library turns model-output tensors into `ef_detect_box` lists and segmentation masks. Box / mask / tile layouts are declared in header-only `edgefirst/detect.h`. It links `libedgefirst_tensor.so` dynamically. pkg-config: `Requires: edgefirst-tensor` and `Requires.private: edgefirst-decoder-abi`.

Exported symbols are `ef_*` only. Tiled merge helpers (`ef_tiled_frame_accumulator`, `ef_lift_tile_boxes`, `ef_merge_tiled_detections`) live here; tile *planning* that needs an `ImageProcessor` stays on `libedgefirst_image`.

**ABI versioning.** `ef_decoder_abi_version()` returns `4`. It moved from `3` when `ef_tile_placement.has_letterbox`/`letterbox` were renamed `has_content_region`/`content_region` (same offsets and meaning) and `ef_semantic_options_set_letterbox` became `ef_semantic_options_set_content_region`: a source and link break with no layout change. It moved from `2` to `3` when the `dtype` argument of `ef_infer_signals_add_input`/`ef_infer_signals_add_output` became an `EF_DTYPE_*` code instead of a `0x100`-based one and `ef_decoder_params_add_output` began refusing unknown `dims` codes: the same calls and layouts, read differently. It moved from `1` to `2` when `ef_merge_config` gained `mode` in its tail pad, a size-safe but not value-safe change that no link error or size mismatch reveals. The default tiled merge became keep-best in the same release; a changed default does not move the probe on its own. The rules that govern when it moves, when a by-value struct may gain a field in place (`ef_merge_config` gained `mode` in its tail pad), and what mixing versions across the five libraries is allowed to do are in [§ C ABI Stability and Versioning](https://github.com/EdgeFirstAI/hal/blob/main/ARCHITECTURE.md#c-abi-stability-and-versioning). Layouts declared in `detect.h`, including `ef_preprocess_spec`, are pinned by [`tests/c/test_layout_goldens.c`](tests/c/test_layout_goldens.c).

**Model configuration.** `ef_model_config_resolve` wraps `edgefirst_decoder::model_config`. Its preprocessing leaves as an `ef_preprocess_spec`, a plain value declared in header-only `detect.h` together with the `EF_FIT_*`, `EF_LETTERBOX_PLACEMENT_*`, `EF_PAD_SPACE_*` and `EF_INTERPOLATION_*` codes it carries, so `libedgefirst_image` consumes it without either library linking the other. Like every by-value struct here it has no `struct_size` handshake: it grows by major bump or successor type, as the root ARCHITECTURE.md sets out, and its layout is pinned by the decoder's layout goldens.

See [README.md](README.md) for usage and [tests/c/](tests/c/) for the link-and-run consumer.
