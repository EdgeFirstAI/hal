<!--
SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
SPDX-License-Identifier: Apache-2.0
-->

# edgefirst-decoder-capi — Architecture

C ABI for `edgefirst-decoder`. Ships as **`libedgefirst_decoder`** with `edgefirst/decoder.h`.

This library turns model-output tensors into `ef_detect_box` lists and segmentation masks. Box / mask / tile layouts are declared in header-only `edgefirst/detect.h`. It links `libedgefirst_tensor.so` dynamically. pkg-config: `Requires: edgefirst-tensor` and `Requires.private: edgefirst-decoder-abi`.

Exported symbols are `ef_*` only. Tiled merge helpers (`ef_tiled_frame_accumulator`, `ef_lift_tile_boxes`, `ef_merge_tiled_detections`) live here; tile *planning* that needs an `ImageProcessor` stays on `libedgefirst_image`.

See [README.md](README.md) for usage and [tests/c/](tests/c/) for the link-and-run consumer.
