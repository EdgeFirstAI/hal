<!--
SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
SPDX-License-Identifier: Apache-2.0
-->

# edgefirst-tracker-capi — Architecture

C ABI for `edgefirst-tracker`. Ships as **`libedgefirst_tracker`** with `edgefirst/tracker.h`.

ByteTrack. Detections cross as a plain `ef_detect_box` array, so this library has **no** `DT_NEEDED` on tensor or decoder. `tracker.h` includes `edgefirst/detect.h` (and historically `decoder.h`) for the type; pkg-config uses `Requires.private: edgefirst-decoder` so Cflags arrive without forcing a dynamic link.

Exported symbols are `ef_*` only.

See [README.md](README.md) for usage and [tests/c/](tests/c/) for the link-and-run consumer.
