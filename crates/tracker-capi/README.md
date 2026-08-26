<!--
SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
SPDX-License-Identifier: Apache-2.0
-->

# edgefirst-tracker (C API)

ByteTrack multi-object tracking — gives each object a stable UUID across frames.
Ships as `libedgefirst_tracker` with `edgefirst/tracker.h`.

```sh
cc my_app.c $(pkg-config --cflags --libs edgefirst-tracker) -o my_app
```

## What this library links, and what it does not

`tracker.h` includes `edgefirst/decoder.h`, because detections are
`ef_detect_box` values and that is where they are declared. So compiling against
this library needs decoder's headers — which is why the pkg-config file says
`Requires.private: edgefirst-decoder`.

`libedgefirst_tracker.so` itself has **no** `DT_NEEDED` entry for any sibling. It
links neither `libedgefirst_decoder` nor `libedgefirst_tensor`. Detections arrive
as a plain C array from `ef_detect_box_list_data`, and reading six scalars per
element does not require linking the library that produced them. `Requires.private`
is what expresses that precisely: its cflags always apply so the header resolves,
while its libs apply only under `pkg-config --static`, where a static link
genuinely does need decoder's symbols present.

A tracking-only consumer therefore installs decoder's headers but loads one
shared library at runtime.

## Usage

```c
#include <edgefirst/tracker.h>

ef_bytetrack *t = ef_bytetrack_new_default();

/* detections from libedgefirst_decoder, or built by hand */
const ef_detect_box *dets = ef_detect_box_list_data(boxes);
uintptr_t n = ef_detect_box_list_len(boxes);

ef_track_info_list *tracks = ef_bytetrack_update(t, dets, n, timestamp_ns);

for (uintptr_t i = 0; i < ef_track_info_list_len(tracks); i++) {
    ef_track_info info = ef_track_info_list_get(tracks, i);
    char uuid[37];
    ef_uuid_to_string(info.id, uuid);
    /* ... */
}

ef_track_info_list_free(tracks);
ef_bytetrack_free(t);
```

`ef_bytetrack_active_tracks` reports every track currently alive without
advancing the tracker, for a caller that wants to inspect state between frames.
