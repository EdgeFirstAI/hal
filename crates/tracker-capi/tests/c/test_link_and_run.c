/* SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
 * SPDX-License-Identifier: Apache-2.0
 *
 * Links and RUNS against libedgefirst_tracker only.
 * A header that compiles is not the same as a library that loads. */
#include <edgefirst/tracker.h>
#include <stdio.h>

int main(void) {
    struct ef_bytetrack *t = ef_bytetrack_new_default();
    if (!t) { fprintf(stderr, "FAIL: ef_bytetrack_new_default returned NULL\n"); return 1; }
    struct ef_track_info_list *l = ef_bytetrack_active_tracks(t);
    if (!l) { fprintf(stderr, "FAIL: active_tracks returned NULL\n"); return 2; }
    printf("PASS: tracker links and runs (%zu active)\n", ef_track_info_list_len(l));
    ef_track_info_list_free(l);
    ef_bytetrack_free(t);
    return 0;
}
