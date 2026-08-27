#ifndef EDGEFIRST_TRACKER_H
#define EDGEFIRST_TRACKER_H

/**
 * @file tracker.h
 * @brief EdgeFirst multi-object tracking C API
 *
 * SPDX-License-Identifier: Apache-2.0
 * Copyright (c) 2026 Au-Zone Technologies. All Rights Reserved.
 *
 * Associates detections across frames, giving each object a stable identity.
 *
 * Takes detections as a plain `(const ef_detect_box *, size_t)` pair — the
 * same view `ef_detect_box_list_data()`/`_len()` give back — so this library
 * needs only the by-value `ef_detect_box` type, declared in
 * edgefirst/detect.h, and never links the decoder.
 */

#include <stdarg.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>

#include "edgefirst/detect.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief An opaque tracker.
 *
 * Forward-declared so callers write `ef_bytetrack *` without the `struct`
 * keyword. Its definition is private: tracker state is not part of the ABI.
 */
typedef struct ef_bytetrack ef_bytetrack;

/** @brief An opaque list of tracks. */
typedef struct ef_track_info_list ef_track_info_list;


/* Generated with cbindgen:0.29.4 */

/* WARNING: The generated portion of this file is produced by cbindgen. Do not modify it directly. */

#include <stdarg.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>

/**
 * One track: a stable identity, its predicted location, and its history.
 */
typedef struct ef_track_info {
  /**
   * Stable identity for this track, as raw UUID bytes.
   *
   * Bytes rather than a string so the struct stays a plain value with no
   * allocation; format it with [`ef_uuid_to_string`] when a human needs it.
   */
  uint8_t uuid[16];
  /**
   * Smoothed predicted location, XYXY normalized.
   */
  float location[4];
  /**
   * How many times this track has been updated.
   */
  int32_t count;
  /**
   * Nanosecond timestamp when the track was created.
   */
  uint64_t created;
  /**
   * Nanosecond timestamp of the most recent update.
   */
  uint64_t last_updated;
} ef_track_info;

/**
 * ABI version of this library's C surface.
 */
uint32_t ef_tracker_abi_version(void);

/**
 * Create a tracker with explicit parameters.
 *
 * @param track_update  Smoothness of track updates; higher is more stable,
 *                      lower more responsive.
 * @param high_thresh   Confidence above which a detection can start a track.
 * @param match_thresh  IoU required to match a detection to a track.
 */
ef_bytetrack *ef_bytetrack_new(float track_update, float high_thresh, float match_thresh);

/**
 * Create a tracker with the library's defaults.
 */
ef_bytetrack *ef_bytetrack_new_default(void);

/**
 * Free a tracker. Freeing `NULL` is a no-op.
 *
 * # Safety
 * `t` must be `NULL` or have come from this library.
 */
void ef_bytetrack_free(ef_bytetrack *t);

/**
 * Advance the tracker by one frame.
 *
 * `timestamp` is in nanoseconds and must be monotonic; the tracker uses it for
 * track age and expiry, so a non-monotonic value corrupts both.
 *
 * @return a track list the caller must free, or `NULL` on a null argument.
 *
 * # Safety
 * `detections` must point to `count` readable elements, or be `NULL` when
 * `count` is 0 — obtain them from `ef_detect_box_list_data`.
 */
ef_track_info_list *ef_bytetrack_update(ef_bytetrack *t,
                                        const ef_detect_box *detections,
                                        uintptr_t count,
                                        uint64_t timestamp);

/**
 * Every track currently alive, without advancing the tracker.
 *
 * @return a track list the caller must free, or `NULL`.
 *
 * # Safety
 * `t` must be live.
 */
ef_track_info_list *ef_bytetrack_active_tracks(const ef_bytetrack *t);

/**
 * Number of tracks. Zero for a `NULL` list.
 *
 * # Safety
 * `l` must be `NULL` or valid.
 */
uintptr_t ef_track_info_list_len(const ef_track_info_list *l);

/**
 * Copy track `index` into `out`.
 *
 * Copies rather than lending a pointer: a borrowed element dangles the moment
 * the list is freed, and C gives the caller no way to notice.
 *
 * @return 0 on success, `EINVAL` for a null argument or out-of-range index.
 *
 * # Safety
 * `out` must be writable.
 */
int ef_track_info_list_get(const ef_track_info_list *l, uintptr_t index, struct ef_track_info *out);

/**
 * Free a track list. Freeing `NULL` is a no-op.
 *
 * # Safety
 * `l` must be `NULL` or have come from this library.
 */
void ef_track_info_list_free(ef_track_info_list *l);

/**
 * Format a track's UUID into `out` as 36 characters plus a NUL.
 *
 * `out` must have room for 37 bytes. Writing into a caller's buffer rather
 * than returning an allocation keeps ownership with the caller and needs no
 * matching free.
 *
 * @return 0 on success, `EINVAL` on a null argument.
 *
 * # Safety
 * `uuid` must be 16 readable bytes; `out` must be 37 writable bytes.
 */
int ef_uuid_to_string(const uint8_t *uuid, char *out);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* EDGEFIRST_TRACKER_H */
