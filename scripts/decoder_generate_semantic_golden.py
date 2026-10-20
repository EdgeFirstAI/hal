#!/usr/bin/env python3
"""Generate the PyTorch golden fixture for the semantic segmentation decoder.

Writes ``testdata/decoder/semantic_torch_golden.json``: a small random logit
map and the class maps PyTorch produces for it. Accurate cases bilinearly
upsample every channel and take the argmax; fast cases take the argmax first
and upsample with ``nearest-exact``. Cropped (letterboxed) cases sample the
content region of the full map, so taps past the region's edge read the
padding rows as the two-step reference (resize to the input, then crop)
does; ``grid_sample`` expresses that for ``align_corners=False``.

Usage: python scripts/decoder_generate_semantic_golden.py (needs torch).
"""

import json
from pathlib import Path

import torch
import torch.nn.functional as F

C, H, W = 5, 7, 9
OUT_H, OUT_W = 13, 17
CROP_H, CROP_W = 6, 8


def region_bilinear(logits, rh, rw):
    """Half-pixel bilinear samples of rows [0, rh) and columns [0, rw), with taps clamped to the full map."""

    def axis(out, n, roi):
        src = ((torch.arange(out, dtype=torch.float64) + 0.5) * (roi / out) - 0.5).clamp(min=0.0)
        return ((2 * src + 1) / n - 1).float()

    gy, gx = axis(OUT_H, H, rh), axis(OUT_W, W, rw)
    grid = torch.stack(torch.meshgrid(gx, gy, indexing="xy"), dim=-1)[None]
    return F.grid_sample(logits, grid, mode="bilinear", padding_mode="border", align_corners=False)


def main():
    g = torch.Generator().manual_seed(2164)
    # Rounded to 1/64 so the values are exact in JSON and in f16.
    logits = torch.round(torch.randn(1, C, H, W, generator=g) * 64) / 64
    cases = []
    for crop in (False, True):
        x = logits[:, :, :CROP_H, :CROP_W] if crop else logits
        letterbox = [0.0, 0.0, CROP_W / W, CROP_H / H] if crop else [0.0, 0.0, 1.0, 1.0]
        for align in (False, True):
            if crop and not align:
                up = region_bilinear(logits, CROP_H, CROP_W)
            else:
                up = F.interpolate(x, size=(OUT_H, OUT_W), mode="bilinear", align_corners=align)
            cases.append({"mode": "accurate", "align_corners": align, "letterbox": letterbox, "classes": up.argmax(1)[0].flatten().tolist()})
        am = x.argmax(1, keepdim=True).float()
        near = F.interpolate(am, size=(OUT_H, OUT_W), mode="nearest-exact")
        cases.append({"mode": "fast", "align_corners": False, "letterbox": letterbox, "classes": near[0, 0].long().flatten().tolist()})
    out = {
        "torch": torch.__version__,
        "shape": [1, C, H, W],
        "out": [OUT_H, OUT_W],
        "logits": logits.flatten().tolist(),
        "cases": cases,
    }
    path = Path(__file__).resolve().parent.parent / "testdata/decoder/semantic_torch_golden.json"
    path.write_text(json.dumps(out) + "\n")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
