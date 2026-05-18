#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
# SPDX-License-Identifier: Apache-2.0
"""Generate EXIF orientation fixtures for codec tests/benchmarks.

For each EXIF orientation value (1..=8), produce a JPEG and a PNG whose
**pixel bytes are identical to `zidane.jpg`** and whose **only difference
is the EXIF orientation tag**. This lets tests assert:

  - apply_exif=false on all 8 variants yields the same decoded bytes
    (since the JPEG/PNG IDAT/SOS content is unchanged).
  - apply_exif=true on each variant yields the orientation-specific
    transform applied to the orientation=1 reference.

EXIF orientation reference (per EXIF/TIFF spec):
   1: Identity                  (no transform)
   2: Mirror horizontal         (flip on the vertical axis)
   3: Rotate 180°
   4: Mirror vertical           (flip on the horizontal axis = 180° + flip-H)
   5: Rotate 90° CW + mirror H  (or transpose along main diagonal)
   6: Rotate 90° CW
   7: Rotate 90° CCW + mirror H (or transpose along anti-diagonal)
   8: Rotate 90° CCW            (= 270° CW)

Run once when fixtures are missing or zidane.jpg changes:

    source venv/bin/activate
    python scripts/generate_exif_fixtures.py

The generated files (`testdata/zidane_exif_<N>.jpg|.png` for N in 1..=8)
are committed to the repository so cross-target benchmark deployments
don't need to regenerate them.
"""

from __future__ import annotations
import io
import struct
import sys
from pathlib import Path

import piexif
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parent.parent
TESTDATA = REPO_ROOT / "testdata"


def jpeg_with_orientation(zidane_jpg_bytes: bytes, orientation: int) -> bytes:
    """Return a JPEG identical to `zidane_jpg_bytes` except for the EXIF
    orientation tag. Uses piexif to rewrite the APP1 segment; the entropy-
    coded scan data is untouched."""
    # piexif works on a fresh EXIF dict so we can set just orientation
    exif_dict = {"0th": {piexif.ImageIFD.Orientation: orientation}, "Exif": {}, "GPS": {}, "1st": {}, "thumbnail": None}
    exif_bytes = piexif.dump(exif_dict)
    out = io.BytesIO()
    piexif.insert(exif_bytes, zidane_jpg_bytes, out)
    return out.getvalue()


import binascii


def _png_chunk(ctype: bytes, payload: bytes) -> bytes:
    """Assemble a PNG chunk: length (4) + type (4) + payload + CRC (4)."""
    length = struct.pack(">I", len(payload))
    crc = struct.pack(">I", binascii.crc32(ctype + payload) & 0xFFFFFFFF)
    return length + ctype + payload + crc


def png_with_orientation(zidane_jpg_bytes: bytes, orientation: int) -> bytes:
    """Encode zidane's pixels as PNG with an `eXIf` chunk carrying the
    given orientation.

    PIL's `PngInfo.add()` does not actually write custom chunks like
    `eXIf` (silently dropped on save), so we post-process: render the
    PNG without EXIF, then splice an `eXIf` chunk in between IHDR and
    the first IDAT. PNG IDAT content is reproducible (Pillow's
    deterministic deflate), so all 8 PNG variants differ only in the
    inserted `eXIf` chunk."""
    src = Image.open(io.BytesIO(zidane_jpg_bytes)).convert("RGB")

    # Build a TIFF-formatted EXIF block (no JPEG APP1 "Exif\0\0" prefix —
    # PNG eXIf chunks carry only the TIFF data per the PNG eXIf chunk
    # spec).
    exif_dict = {"0th": {piexif.ImageIFD.Orientation: orientation}, "Exif": {}, "GPS": {}, "1st": {}, "thumbnail": None}
    raw = piexif.dump(exif_dict)
    if raw.startswith(b"Exif\x00\x00"):
        tiff = raw[6:]
    else:
        tiff = raw
    exif_chunk = _png_chunk(b"eXIf", tiff)

    # Render the base PNG (no EXIF), then splice our chunk in after IHDR.
    out = io.BytesIO()
    src.save(out, format="PNG", optimize=False, compress_level=6)
    base = out.getvalue()
    sig = base[:8]
    # IHDR is always the first chunk: length(4) + "IHDR"(4) + 13 data + CRC(4)
    ihdr_end = 8 + 4 + 4 + 13 + 4
    assert base[8 + 4: 8 + 8] == b"IHDR"
    return sig + base[8:ihdr_end] + exif_chunk + base[ihdr_end:]


def main() -> int:
    zidane_path = TESTDATA / "zidane.jpg"
    if not zidane_path.is_file():
        print(f"ERROR: {zidane_path} not found", file=sys.stderr)
        return 1
    src_bytes = zidane_path.read_bytes()

    print(f"Source: {zidane_path} ({len(src_bytes):,} bytes)")
    print("Generating EXIF orientation fixtures (1..=8)…")

    for orientation in range(1, 9):
        jpg_path = TESTDATA / f"zidane_exif_{orientation}.jpg"
        png_path = TESTDATA / f"zidane_exif_{orientation}.png"

        jpg_bytes = jpeg_with_orientation(src_bytes, orientation)
        png_bytes = png_with_orientation(src_bytes, orientation)

        jpg_path.write_bytes(jpg_bytes)
        png_path.write_bytes(png_bytes)

        # Sanity: round-trip the EXIF tag we just wrote.
        rt = piexif.load(jpg_bytes)
        got = rt["0th"].get(piexif.ImageIFD.Orientation)
        ok = "OK" if got == orientation else f"BAD (got {got})"
        print(
            f"  orientation={orientation}  jpg={len(jpg_bytes):,}B  png={len(png_bytes):,}B  [{ok}]"
        )

    # Verify orientation=1 JPEG decodes to the same pixels as zidane.jpg.
    src_pixels = Image.open(io.BytesIO(src_bytes)).convert("RGB")
    one_jpg = (TESTDATA / "zidane_exif_1.jpg").read_bytes()
    one_pixels = Image.open(io.BytesIO(one_jpg)).convert("RGB")
    if list(src_pixels.getdata()) != list(one_pixels.getdata()):
        print(
            "ERROR: orientation=1 JPEG pixels differ from source — piexif altered scan data",
            file=sys.stderr,
        )
        return 1
    print("✓ orientation=1 JPEG pixels match source byte-for-byte")
    return 0


if __name__ == "__main__":
    sys.exit(main())
