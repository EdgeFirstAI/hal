<!--
SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
SPDX-License-Identifier: Apache-2.0
-->

# edgefirst-codec-capi — Architecture

C ABI for `edgefirst-codec`. Ships as **`libedgefirst_codec`** with `edgefirst/codec.h`.

This library decodes JPEG/PNG into a caller-supplied `ef_tensor`. It links `libedgefirst_tensor.so` (`DT_NEEDED` + `RUNPATH $ORIGIN`) instead of embedding a private tensor copy. pkg-config: `Requires: edgefirst-tensor`.

Exported symbols are `ef_*` only (version script / exported-symbols list). Constructors (`ef_image_decoder_new`, `ef_image_decoder_decode_into`, …) live here; tensor accessors live only in `libedgefirst_tensor`.

**ABI versioning.** `ef_codec_abi_version()` returns `1`; the rules that govern when it moves, and what mixing versions across the five libraries is allowed to do, are in [§ C ABI Stability and Versioning](https://github.com/EdgeFirstAI/hal/blob/main/ARCHITECTURE.md#c-abi-stability-and-versioning). `codec.h` declares only opaque handles — no by-value struct — so this library has no layout goldens and needs none.

See [README.md](README.md) for usage and [tests/c/](tests/c/) for the link-and-run consumer.
