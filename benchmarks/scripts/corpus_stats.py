#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
# SPDX-License-Identifier: Apache-2.0
"""JPEG corpus statistics for benchmark reporting (stdlib only).

Walks a directory of JPEGs and reports the distribution facts a published
benchmark must state about its corpus: image count, dimensions/megapixels,
file sizes, chroma subsampling histogram, baseline-vs-progressive split,
greyscale count, and restart-interval (DRI) presence.

The DRI count is what turns "COCO has no restart-marker files" from an
assumption into a measurement: libjpeg-turbo documents that its fast Huffman
decoder is disabled for DRI streams (up to ~20% slower), so whether a corpus
carries DRI markers changes which turbo code path a benchmark exercises.

Markers are parsed structurally up to the first SOS (where cjpeg/jpegtran
place DRI), not text-searched, so EXIF thumbnails cannot false-positive.

Usage:
    python3 corpus_stats.py DIR [DIR ...]
"""

import struct
import sys
from pathlib import Path

SOF_MARKERS = {
    0xC0,
    0xC1,
    0xC2,
    0xC3,
    0xC5,
    0xC6,
    0xC7,
    0xC9,
    0xCA,
    0xCB,
    0xCD,
    0xCE,
    0xCF,
}
PROGRESSIVE_SOFS = {0xC2, 0xC6, 0xCA, 0xCE}


def parse_jpeg(path):
    """Return dict of facts for one JPEG, or None if it does not parse."""
    data = path.read_bytes()
    if len(data) < 4 or data[0:2] != b"\xff\xd8":
        return None
    facts = {
        "bytes": len(data),
        "width": 0,
        "height": 0,
        "ncomp": 0,
        "subsampling": "?",
        "progressive": False,
        "dri": False,
    }
    i = 2
    n = len(data)
    while i + 4 <= n:
        if data[i] != 0xFF:
            i += 1
            continue
        marker = data[i + 1]
        if marker in (0xFF, 0x00) or 0xD0 <= marker <= 0xD8:
            i += 2
            continue
        if marker == 0xD9:  # EOI
            break
        seglen = struct.unpack(">H", data[i + 2 : i + 4])[0]
        # A truncated/corrupt file can declare a segment running past EOF (or
        # an impossible <2 length); stop parsing and let the caller count the
        # file as unparsed rather than raising out of the whole run.
        if seglen < 2 or i + 2 + seglen > n:
            break
        seg = data[i + 4 : i + 2 + seglen]
        if marker == 0xDD:
            facts["dri"] = True
        elif marker in SOF_MARKERS:
            if len(seg) < 6:  # truncated SOF payload
                break
            facts["progressive"] = marker in PROGRESSIVE_SOFS
            _prec, h, w, ncomp = struct.unpack(">BHHB", seg[0:6])
            facts["height"], facts["width"], facts["ncomp"] = h, w, ncomp
            if ncomp == 1:
                facts["subsampling"] = "grey"
            elif ncomp >= 3 and len(seg) >= 8:
                # Luma sampling factors relative to 1x1 chroma.
                hv = seg[6 + 1]
                lh, lv = hv >> 4, hv & 0xF
                facts["subsampling"] = {
                    (1, 1): "4:4:4",
                    (2, 1): "4:2:2",
                    (1, 2): "4:4:0",
                    (2, 2): "4:2:0",
                    (4, 1): "4:1:1",
                }.get((lh, lv), f"{lh}x{lv}")
        elif marker == 0xDA:  # first SOS: tables (incl. DRI) precede this
            break
        i += 2 + seglen
    return facts


def pctl(sorted_vals, p):
    if not sorted_vals:
        return 0
    idx = min(round(p * (len(sorted_vals) - 1)), len(sorted_vals) - 1)
    return sorted_vals[idx]


def report(directory):
    d = Path(directory).expanduser()
    files = sorted(
        p for p in d.iterdir() if p.suffix.lower() in (".jpg", ".jpeg") and p.is_file()
    )
    if not files:
        print(f"{d}: no JPEG files")
        return 1
    stats, unparsed = [], 0
    for p in files:
        f = parse_jpeg(p)
        if f is None or f["width"] == 0:
            unparsed += 1
            continue
        stats.append(f)
    if not stats:
        print(f"{d}: {unparsed} files, none parsed as JPEG")
        return 1

    mpix = sorted(f["width"] * f["height"] / 1e6 for f in stats)
    sizes = sorted(f["bytes"] / 1024.0 for f in stats)
    sub_hist = {}
    for f in stats:
        sub_hist[f["subsampling"]] = sub_hist.get(f["subsampling"], 0) + 1
    n_dri = sum(f["dri"] for f in stats)
    n_prog = sum(f["progressive"] for f in stats)

    print(f"=== {d} ===")
    print(
        f"  files: {len(stats)} parsed" + (f", {unparsed} unparsed" if unparsed else "")
    )
    print(
        f"  megapixels: min={mpix[0]:.3f}  p50={pctl(mpix, 0.5):.3f}  max={mpix[-1]:.3f}"
    )
    print(
        f"  file KB:    min={sizes[0]:.0f}  p50={pctl(sizes, 0.5):.0f}  max={sizes[-1]:.0f}"
    )
    subs = "  ".join(
        f"{k}={v}" for k, v in sorted(sub_hist.items(), key=lambda kv: -kv[1])
    )
    print(f"  subsampling: {subs}")
    print(f"  progressive: {n_prog}    restart-interval (DRI): {n_dri}")
    return 0


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    rc = 0
    for d in sys.argv[1:]:
        rc |= report(d)
    return rc


if __name__ == "__main__":
    sys.exit(main())
