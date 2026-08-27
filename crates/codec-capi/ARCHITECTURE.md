<!--
SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
SPDX-License-Identifier: Apache-2.0
-->

# edgefirst-codec-capi — Architecture

C ABI for `edgefirst-codec`. Ships as **`libedgefirst_codec`** with `edgefirst/codec.h`.

This library decodes JPEG/PNG into a caller-supplied `ef_tensor`. It links `libedgefirst_tensor.so` (`DT_NEEDED` + `RUNPATH $ORIGIN`) instead of embedding a private tensor copy. pkg-config: `Requires: edgefirst-tensor`.

Exported symbols are `ef_*` only (version script / exported-symbols list). Constructors (`ef_image_decoder_new`, `ef_image_decoder_decode_into`, …) live here; tensor accessors live only in `libedgefirst_tensor`.

See [README.md](README.md) for usage and [tests/c/](tests/c/) for the link-and-run consumer.
