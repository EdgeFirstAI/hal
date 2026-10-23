#!/usr/bin/env python3
"""Generate SegFormer reference fixtures for the semantic segmentation decoder.

Runs a SegFormer ONNX export on a ``testdata`` image and writes
``testdata/decoder/segformer-b1-ade-<shape>_<image>.safetensors`` with:

- ``logits``: the model's float32 output ``[1, 150, h, w]``;
- ``logits_i8``: the same logits quantized per tensor to int8;
- ``ref.accurate`` / ``ref.fast``: class maps at the image size from the
  reference decode of ``logits``;
- ``ref.margin``: float16 gap between the two highest upsampled logits of the
  accurate reference, which marks the near-ties where a fused resize may differ;
- ``ref.accurate_i8`` / ``ref.fast_i8``: the same from the dequantized
  ``logits_i8``.

The accurate reference is the mmsegmentation two-step decode: bilinear to the
model input, crop the letterbox padding, bilinear to the image size, argmax.
The fast reference takes the argmax at the model's resolution, crops at the
rounded-up cell count and upsamples with PyTorch's ``nearest``.

``__metadata__`` holds the ``edgefirst.json`` schema for each dtype
(``edgefirst_json``, ``edgefirst_json_i8``), the normalized letterbox
content region (``letterbox``), the source image and the model.

Usage (needs torch, onnxruntime, opencv-python, safetensors):

    python scripts/decoder_generate_semantic_fixture.py \\
        --onnx segformer-b1-ade-512x672.onnx --image testdata/zidane.jpg --resize letterbox
"""

import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
import torch
import torch.nn.functional as F
from safetensors.numpy import save_file

MEAN = np.array([123.675, 116.28, 103.53], dtype=np.float32)
STD = np.array([58.395, 57.12, 57.375], dtype=np.float32)


def preprocess(img_bgr, resize, in_h, in_w):
    """Normalized ``[1, 3, in_h, in_w]`` input and the ``(h, w)`` of its content."""
    h, w = img_bgr.shape[:2]
    if resize == "stretch":
        vh, vw = in_h, in_w
    else:
        s = min(in_h / h, in_w / w)
        vh, vw = min(in_h, int(h * s + 0.5)), min(in_w, int(w * s + 0.5))
    img = cv2.resize(img_bgr, (vw, vh), interpolation=cv2.INTER_LINEAR)
    x = (cv2.cvtColor(img.astype(np.float32), cv2.COLOR_BGR2RGB) - MEAN) / STD
    padded = np.zeros((3, in_h, in_w), dtype=np.float32)
    padded[:, :vh, :vw] = x.transpose(2, 0, 1)
    return padded[None], (vh, vw)


def reference(logits, in_hw, valid, ori_hw):
    """Accurate and fast reference class maps as uint8 ``[H, W]``, and the accurate top-two margin."""
    t = torch.from_numpy(logits)
    in_h, in_w = in_hw
    vh, vw = valid
    _, _, h, w = t.shape
    up = F.interpolate(t, size=in_hw, mode="bilinear", align_corners=False)[:, :, :vh, :vw]
    scores = F.interpolate(up, size=ori_hw, mode="bilinear", align_corners=False)[0]
    top2 = scores.topk(2, dim=0).values
    accurate = scores.argmax(0)
    qh, qw = math.ceil(vh * h / in_h), math.ceil(vw * w / in_w)
    am = t.argmax(1, keepdim=True)[:, :, :qh, :qw].float()
    fast = F.interpolate(am, size=ori_hw, mode="nearest")[0, 0]
    return accurate.numpy().astype(np.uint8), fast.numpy().astype(np.uint8), (top2[0] - top2[1]).numpy().astype(np.float16)


def quantize(logits):
    """Asymmetric per-tensor int8 quantization: ``(q, scale, zero_point)``."""
    lo, hi = float(logits.min()), float(logits.max())
    scale = (hi - lo) / 255.0
    zero_point = int(round(-128 - lo / scale))
    q = np.clip(np.round(logits / scale) + zero_point, -128, 127).astype(np.int8)
    return q, scale, zero_point


def schema(shape, in_shape, resize, dtype, quant=None):
    output = {
        "name": "logits",
        "type": "segmentation",
        "decoder": "semantic",
        "shape": list(shape),
        "dshape": [{"batch": 1}, {"num_classes": shape[1]}, {"height": shape[2]}, {"width": shape[3]}],
        "dtype": dtype,
        "quantization": None if quant is None else {"scale": quant[0], "zero_point": quant[1], "dtype": dtype},
        "stride": in_shape[2] // shape[2],
        "upsample": {"method": "bilinear", "align_corners": False},
        "decode_mode": "accurate",
    }
    preprocessing = {"normalization": "imagenet", "resize": resize, "interpolation": "bilinear"}
    if resize == "letterbox":
        preprocessing["letterbox"] = {"placement": "topleft", "pad": [0.0, 0.0, 0.0], "pad_space": "normalized"}
    return {
        "schema_version": 2,
        "input": {
            "shape": list(in_shape),
            "dshape": [{"batch": 1}, {"channels": 3}, {"height": in_shape[2]}, {"width": in_shape[3]}],
            "cameraadaptor": "rgb",
            "preprocessing": preprocessing,
        },
        "outputs": [output],
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--onnx", required=True, type=Path)
    ap.add_argument("--image", required=True, type=Path)
    ap.add_argument("--resize", choices=["stretch", "letterbox"], required=True)
    ap.add_argument("--out-dir", type=Path, default=Path(__file__).resolve().parent.parent / "testdata" / "decoder")
    args = ap.parse_args()

    session = ort.InferenceSession(str(args.onnx), providers=["CPUExecutionProvider"])
    in_shape = session.get_inputs()[0].shape
    in_h, in_w = in_shape[2], in_shape[3]
    img = cv2.imread(str(args.image), cv2.IMREAD_COLOR)
    ori_hw = img.shape[:2]
    x, (vh, vw) = preprocess(img, args.resize, in_h, in_w)
    logits = session.run(None, {session.get_inputs()[0].name: x})[0].astype(np.float32)

    q, scale, zero_point = quantize(logits)
    dequant = (q.astype(np.float32) - zero_point) * np.float32(scale)
    accurate, fast, margin = reference(logits, (in_h, in_w), (vh, vw), ori_hw)
    accurate_i8, fast_i8, _ = reference(dequant, (in_h, in_w), (vh, vw), ori_hw)

    letterbox = [0.0, 0.0, vw / in_w, vh / in_h]
    metadata = {
        "edgefirst_json": json.dumps(schema(logits.shape, in_shape, args.resize, "float32")),
        "edgefirst_json_i8": json.dumps(schema(logits.shape, in_shape, args.resize, "int8", (scale, zero_point))),
        "letterbox": json.dumps(letterbox),
        "image": args.image.name,
        "model": args.onnx.name,
        "onnxruntime": ort.__version__,
        "torch": torch.__version__,
    }
    tensors = {
        "logits": logits,
        "logits_i8": q,
        "ref.accurate": accurate,
        "ref.fast": fast,
        "ref.margin": margin,
        "ref.accurate_i8": accurate_i8,
        "ref.fast_i8": fast_i8,
    }
    name = f"{args.onnx.stem}_{args.image.stem}.safetensors"
    save_file(tensors, str(args.out_dir / name), metadata=metadata)
    print(f"{name}: logits {list(logits.shape)}, image {ori_hw[1]}x{ori_hw[0]}, content {vw}x{vh}, int8 scale {scale:.6f} zp {zero_point}")


if __name__ == "__main__":
    main()
