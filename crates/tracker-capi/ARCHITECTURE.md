<!--
SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
SPDX-License-Identifier: Apache-2.0
-->

# edgefirst-tracker-capi — Architecture

C ABI for `edgefirst-tracker`. Ships as **`libedgefirst_tracker`** with `edgefirst/tracker.h`.

ByteTrack. Detections cross as a plain `ef_detect_box` array, so this library has **no** `DT_NEEDED` on tensor or decoder. `tracker.h` includes `edgefirst/detect.h` (and historically `decoder.h`) for the type; pkg-config uses `Requires.private: edgefirst-decoder` so Cflags arrive without forcing a dynamic link.

Exported symbols are `ef_*` only.

**ABI versioning.** `ef_tracker_abi_version()` returns `1`; the rules that govern when it moves, when a by-value struct may gain a field in place, and what mixing versions across the five libraries is allowed to do are in [§ C ABI Stability and Versioning](https://github.com/EdgeFirstAI/hal/blob/main/ARCHITECTURE.md#c-abi-stability-and-versioning). `tracker.h` declares one by-value struct (`ef_track_info`) and has **no** `tests/c/test_layout_goldens.c` pinning it — a gap to close before 1.0.

See [README.md](README.md) for usage and [tests/c/](tests/c/) for the link-and-run consumer.
