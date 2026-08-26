<!--
SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
SPDX-License-Identifier: Apache-2.0
-->

# EdgeFirst HAL — C libraries

Five independently linkable libraries from the [EdgeFirst Hardware Abstraction Layer](https://github.com/EdgeFirstAI/hal). Siblings load `libedgefirst_tensor` at runtime instead of embedding a second copy.

| Library | Header | Role |
|---|---|---|
| `libedgefirst_tensor` | `edgefirst/tensor.h` | Zero-copy tensors |
| `libedgefirst_codec` | `edgefirst/codec.h` | JPEG/PNG decode into a tensor |
| `libedgefirst_image` | `edgefirst/image.h` | Convert, resize, letterbox, draw |
| `libedgefirst_decoder` | `edgefirst/decoder.h` | YOLO / ModelPack post-process |
| `libedgefirst_tracker` | `edgefirst/tracker.h` | ByteTrack |

Detection boxes live in header-only `edgefirst/detect.h`. See `INSTALL.txt` for extract, pkg-config, runtime search path, and a JPEG→tensor example.

**No ABI stability is offered before 1.0.** SONAME is `.so.0` (major 0); pin the archive version you built against.

0.29 ships static `.a` archives in the same tarball as the shared libraries (they dominate uncompressed size; the download still compresses to ~21 MB). Most consumers only need the `.so` / `.dylib` / DLL chain. See `INSTALL.txt` for a recipe that actually static-links — `pkg-config --static` alone still prefers the shared library.

The five per-crate READMEs (`crates/*-capi/README.md` in the source tree) document each library's API. This file is the archive index.
