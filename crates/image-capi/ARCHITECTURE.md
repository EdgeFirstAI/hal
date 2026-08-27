<!--
SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
SPDX-License-Identifier: Apache-2.0
-->

# edgefirst-image-capi — Architecture

C ABI for `edgefirst-image`. Ships as **`libedgefirst_image`** with `edgefirst/image.h`.

This library mints tensors (`ef_image_processor_create_image`) and converts, resizes, letterboxes and draws. It links `libedgefirst_tensor.so` dynamically. Detection primitives used when drawing live in header-only `edgefirst/detect.h` (`Requires.private: edgefirst-decoder-abi`). pkg-config: `Requires: edgefirst-tensor`.

Exported symbols are `ef_*` only. Tensor accessors are not re-exported.

See [README.md](README.md) for usage and [tests/c/](tests/c/) for the link-and-run consumer.
