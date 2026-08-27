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

The archive is the C ABI only: headers plus shared libraries. Linux ships a `.tar.gz`; Windows and macOS ship a `.zip`. Linux and macOS put the `.so` / `.dylib` in `lib/`. Windows ships `bin/edgefirst_*.dll` and `lib/edgefirst_*.lib` import libraries (never cargo's `*.dll.lib`, never the Rust staticlib). See `INSTALL.txt`.

The five per-crate READMEs (`crates/*-capi/README.md` in the source tree) document each library's API. This file is the archive index.
